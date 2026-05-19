#!/usr/bin/env bash
# Encoder-cache policy sweep on 8xA100.
#
# Pipeline:
#   1. Generate fixed pool of K images (one-time).
#   2. Probe the server (FIFO policy) to measure per-type c_i, m_i.
#   3. Solve lambda*.
#   4. For each policy in {fifo, nocache, offline}:
#        start server with that policy,
#        sweep request rates with --num-prompts requests x --repeats runs,
#        dump per-run JSON + cumulative cache-stats snapshot from server log.
#   5. Aggregate everything into summary.csv + summary.md (+ optional plots).
#
# All knobs live as env vars at the top so you can override one or two
# without forking the script. Example:
#   RPS_LIST="8 16 32" REPEATS=1 ./run_sweep.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
TP=${TP:-8}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}

# Output / cache directories
RESULT_DIR=${RESULT_DIR:-$(pwd)/ec_eval_$(date +%Y%m%d_%H%M%S)}
POOL_DIR=${POOL_DIR:-$RESULT_DIR/pool}

# Pool spec
K=${K:-100}
DISTRIBUTION=${DISTRIBUTION:-zipf}
DISTRIBUTION_PARAM=${DISTRIBUTION_PARAM:-1.1}
BUCKETS=${BUCKETS:-"360x640 720x1280 1080x1920 1440x2560"}

# Workload
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-512}
NUM_WARMUPS=${NUM_WARMUPS:-32}
REPEATS=${REPEATS:-3}

# Targeted prewarm: hit each of the K pool images exactly once before the
# bench window so that CUDA-graph capture and vision-tower lazy init
# happen outside the measurement. Disable with PREWARM=0 to fall back to
# pure bench --num-warmups.
PREWARM=${PREWARM:-1}
PREWARM_CONCURRENCY=${PREWARM_CONCURRENCY:-4}
RPS_LIST=${RPS_LIST:-"4 8 12 16 20 24 28 32"}
NUM_MM_BASE=${NUM_MM_BASE:-1}   # base images per request
NUM_MM_RANGE=${NUM_MM_RANGE:-0.0}

# vLLM server flags
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
ENCODER_CACHE_SIZE=${ENCODER_CACHE_SIZE:-}   # leave empty to use vLLM default
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}

# Solver
CACHE_CAPACITY=${CACHE_CAPACITY:-}   # auto-detected from server log if empty

# Sweep
POLICIES=${POLICIES:-"fifo nocache offline"}
STATS_INTERVAL_SEC=${STATS_INTERVAL_SEC:-5}

# Aggregation
AGGREGATE_PLOT=${AGGREGATE_PLOT:-1}   # set 0 to skip matplotlib plots

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$RESULT_DIR/runs" "$RESULT_DIR/server_logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Check for and clean up leftover processes from a previously aborted run.
# Defined further down; called from main pipeline after server lifecycle helpers.

# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------
SERVER_PIDFILE=""

wait_for_server() {
  local start=$SECONDS
  while [ $((SECONDS - start)) -lt $HEALTH_TIMEOUT ]; do
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      log "server healthy after $((SECONDS - start))s"
      return 0
    fi
    sleep 5
  done
  log "ERROR: server did not become healthy within ${HEALTH_TIMEOUT}s"
  if [ -n "$SERVER_PIDFILE" ] && [ -f "$SERVER_PIDFILE" ]; then
    tail -n 100 "$(dirname "$SERVER_PIDFILE")/$(basename "$SERVER_PIDFILE" .pid).log" || true
  fi
  return 1
}

start_server() {
  local policy=$1 logfile=$2 pidfile=$3
  SERVER_PIDFILE=$pidfile
  log "starting server policy=$policy log=$logfile"

  local ec_size_arg=""
  if [ -n "$ENCODER_CACHE_SIZE" ]; then
    ec_size_arg="--encoder-cache-size $ENCODER_CACHE_SIZE"
  fi

  local env_extra=""
  if [ "$policy" = "offline" ]; then
    if [ ! -f "$POOL_DIR/mm_pool.json" ]; then
      log "ERROR: $POOL_DIR/mm_pool.json missing — run solve first"
      return 1
    fi
    env_extra="VLLM_ENCODER_CACHE_POLICY_CONFIG=$POOL_DIR/mm_pool.json"
  fi

  (
    cd "$REPO_ROOT"
    env VLLM_ENCODER_CACHE_POLICY="$policy" \
        VLLM_ENCODER_CACHE_STATS_INTERVAL_SEC="$STATS_INTERVAL_SEC" \
        VLLM_SERVER_DEV_MODE=1 \
        $env_extra \
        nohup vllm serve "$MODEL" \
          --host "$HOST" --port "$PORT" \
          --tensor-parallel-size "$TP" \
          --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
          --max-model-len "$MAX_MODEL_LEN" \
          $ec_size_arg \
          $SERVER_EXTRA_ARGS \
          > "$logfile" 2>&1 &
    echo $! > "$pidfile"
  )
  wait_for_server
}

stop_server() {
  local pidfile=$1
  if [ ! -f "$pidfile" ]; then
    SERVER_PIDFILE=""
    return 0
  fi
  local pid
  pid=$(cat "$pidfile")
  log "stopping server pid=$pid"

  # Step 1: gentle SIGTERM on the parent and on its direct children
  #         (engine core, worker processes spawned by vLLM).
  kill -TERM "$pid" 2>/dev/null || true
  pkill -TERM -P "$pid" 2>/dev/null || true

  # Step 2: wait up to 30s for clean exit.
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done

  # Step 3: if still alive, escalate to SIGKILL on the whole subtree.
  if kill -0 "$pid" 2>/dev/null; then
    log "  SIGTERM did not exit cleanly; sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
    pkill -KILL -P "$pid" 2>/dev/null || true
    sleep 2
  fi

  # Step 4: belt-and-suspenders — kill any leftover vLLM workers / engine
  # cores that may have detached from the parent's process group. These
  # are the processes that most often hold GPU handles and block
  # subsequent nvidia-smi calls.
  pkill -KILL -f "vllm serve" 2>/dev/null || true
  pkill -KILL -f "vllm.entrypoints" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  pkill -KILL -f "ray::" 2>/dev/null || true

  rm -f "$pidfile"
  SERVER_PIDFILE=""

  # Give the NVIDIA driver a few seconds to reclaim GPU device handles.
  sleep 3
}

# Sanity check: warn if there are still vLLM-related processes from a
# previous failed run. They will hold GPU memory and confuse health
# checks for the new server.
check_leftover_processes() {
  local found
  found=$(pgrep -fa "vllm serve|EngineCore" 2>/dev/null || true)
  if [ -n "$found" ]; then
    log "WARNING: leftover vLLM processes detected:"
    echo "$found" | sed 's/^/  /'
    log "  attempting to clean up before starting…"
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "vllm.entrypoints" 2>/dev/null || true
    pkill -KILL -f "EngineCore" 2>/dev/null || true
    pkill -KILL -f "ray::" 2>/dev/null || true
    sleep 5
  fi
}

cleanup() {
  if [ -n "$SERVER_PIDFILE" ] && [ -f "$SERVER_PIDFILE" ]; then
    log "trap: cleaning up server"
    stop_server "$SERVER_PIDFILE"
  fi
}
trap cleanup EXIT INT TERM

# Pre-flight: clean up any leftover processes from a previously aborted run.
check_leftover_processes

# ---------------------------------------------------------------------------
# Stage 1: pool generation
# ---------------------------------------------------------------------------
if [ ! -f "$POOL_DIR/pool_spec.json" ]; then
  log "generate pool K=$K dist=$DISTRIBUTION/$DISTRIBUTION_PARAM"
  python "$REPO_ROOT/tools/precompute_mm_pool.py" --pool-dir "$POOL_DIR" generate \
    --k "$K" --seed 0 \
    --distribution "$DISTRIBUTION" --distribution-param "$DISTRIBUTION_PARAM" \
    --bucket-config $BUCKETS
else
  log "reusing existing pool: $POOL_DIR/pool_spec.json"
fi

# ---------------------------------------------------------------------------
# Stage 2: measure c_i, m_i  (server runs with FIFO policy)
# ---------------------------------------------------------------------------
if ! python - "$POOL_DIR/pool_spec.json" <<'PY' >/dev/null 2>&1
import json, sys
spec = json.load(open(sys.argv[1]))
ok = all(t.get("c_seconds") and t.get("m_tokens") for t in spec["types"])
sys.exit(0 if ok else 1)
PY
then
  log "measuring per-type c_i + m_i (warmup=2 repeats=5)"
  start_server fifo "$RESULT_DIR/server_logs/measure.log" \
                    "$RESULT_DIR/server_logs/measure.pid"
  python "$REPO_ROOT/tools/precompute_mm_pool.py" --pool-dir "$POOL_DIR" measure \
    --base-url "http://${HOST}:${PORT}" \
    --model "$MODEL" --warmup 2 --repeats 5 --use-token-query \
    | tee "$RESULT_DIR/measure_log.txt"
  stop_server "$RESULT_DIR/server_logs/measure.pid"
else
  log "reusing existing c_i/m_i in pool_spec.json"
fi

# ---------------------------------------------------------------------------
# Determine encoder cache capacity B for lambda* solving.
#
# vLLM does NOT print encoder_cache_size at startup, and by default it
# equals --max-num-batched-tokens (see vllm/config/scheduler.py). The
# precedence we use:
#   1. $CACHE_CAPACITY explicitly set.
#   2. $ENCODER_CACHE_SIZE explicitly set (also passed to the server).
#   3. Fallback: $MAX_NUM_BATCHED_TOKENS (the vLLM default).
# ---------------------------------------------------------------------------
if [ -z "$CACHE_CAPACITY" ]; then
  # Try grep first in case a future vLLM version exposes it directly.
  CACHE_CAPACITY=$(grep -oE "encoder_cache_size[^0-9]*[0-9]+" \
                    "$RESULT_DIR/server_logs/measure.log" 2>/dev/null \
                  | grep -oE "[0-9]+" | tail -1 || true)
fi
if [ -z "$CACHE_CAPACITY" ] && [ -n "$ENCODER_CACHE_SIZE" ]; then
  CACHE_CAPACITY=$ENCODER_CACHE_SIZE
  log "using explicit ENCODER_CACHE_SIZE=$CACHE_CAPACITY as cache capacity"
fi
if [ -z "$CACHE_CAPACITY" ]; then
  CACHE_CAPACITY=$MAX_NUM_BATCHED_TOKENS
  log "encoder cache capacity not provided; falling back to" \
      "MAX_NUM_BATCHED_TOKENS=$CACHE_CAPACITY (vLLM default)"
else
  log "encoder cache capacity = $CACHE_CAPACITY tokens"
fi

# ---------------------------------------------------------------------------
# Stage 3: solve lambda*
# ---------------------------------------------------------------------------
if [ ! -f "$POOL_DIR/mm_pool.json" ]; then
  log "solving lambda* with B=$CACHE_CAPACITY"
  python "$REPO_ROOT/tools/precompute_mm_pool.py" --pool-dir "$POOL_DIR" solve \
    --cache-capacity "$CACHE_CAPACITY" --dump-dual-curve \
    | tee "$RESULT_DIR/lambda_solve.log"
else
  log "reusing existing lambda* in mm_pool.json"
fi

# ---------------------------------------------------------------------------
# Stage 4: policy sweep
# ---------------------------------------------------------------------------
for POLICY in $POLICIES; do
  log "==================== POLICY: $POLICY ===================="
  LOGFILE="$RESULT_DIR/server_logs/${POLICY}.log"
  PIDFILE="$RESULT_DIR/server_logs/${POLICY}.pid"
  start_server "$POLICY" "$LOGFILE" "$PIDFILE"

  if [ "$PREWARM" = "1" ]; then
    log "  prewarming encoder cache (K=$K targeted requests, concurrency=$PREWARM_CONCURRENCY)"
    python "$REPO_ROOT/tools/precompute_mm_pool.py" --pool-dir "$POOL_DIR" prewarm \
      --base-url "http://${HOST}:${PORT}" \
      --model "$MODEL" \
      --concurrency "$PREWARM_CONCURRENCY" \
      --seed 0 2>&1 \
      | tee -a "$RESULT_DIR/server_logs/${POLICY}.prewarm.txt"
    # Tiny breather to let the cumulative stats log line emit before bench starts.
    sleep "$STATS_INTERVAL_SEC"
  fi

  for RPS in $RPS_LIST; do
    RPS_TAG=$(printf "%02d" "$RPS")
    for REP in $(seq 0 $((REPEATS - 1))); do
      OUT="$RESULT_DIR/runs/${POLICY}_rps${RPS_TAG}_rep${REP}.json"
      BENCH_LOG="$RESULT_DIR/runs/${POLICY}_rps${RPS_TAG}_rep${REP}.bench.log"
      CACHE_SNAPSHOT="$RESULT_DIR/runs/${POLICY}_rps${RPS_TAG}_rep${REP}.cache.txt"
      log "  run policy=$POLICY rps=$RPS rep=$REP -> $(basename "$OUT")"

      ( cd "$REPO_ROOT" && vllm bench serve \
          --backend openai-chat \
          --base-url "http://${HOST}:${PORT}" \
          --endpoint /v1/chat/completions \
          --model "$MODEL" \
          --dataset-name mm-fixed-pool \
          --mm-pool-dir "$POOL_DIR" \
          --random-mm-base-items-per-request "$NUM_MM_BASE" \
          --random-mm-num-mm-items-range-ratio "$NUM_MM_RANGE" \
          --random-input-len "$INPUT_LEN" \
          --random-output-len "$OUTPUT_LEN" \
          --random-range-ratio 0.0 \
          --num-prompts "$NUM_PROMPTS" \
          --num-warmups "$NUM_WARMUPS" \
          --request-rate "$RPS" \
          --ignore-eos \
          --percentile-metrics ttft,tpot,itl,e2el \
          --metric-percentiles 50,90,95,99 \
          --save-result --result-filename "$OUT" \
        ) > "$BENCH_LOG" 2>&1

      # Capture the most recent cache-policy summary line from the server log.
      grep "encoder_cache " "$LOGFILE" | tail -1 > "$CACHE_SNAPSHOT" || true
    done
  done

  stop_server "$PIDFILE"
done

# ---------------------------------------------------------------------------
# Stage 5: aggregate
# ---------------------------------------------------------------------------
PLOT_ARG=""
if [ "$AGGREGATE_PLOT" = "1" ]; then
  PLOT_ARG="--plot-dir $RESULT_DIR/plots"
fi
log "aggregating results"
python "$SCRIPT_DIR/aggregate.py" \
  --result-dir "$RESULT_DIR/runs" \
  --pool-spec "$POOL_DIR/pool_spec.json" \
  --output-csv "$RESULT_DIR/summary.csv" \
  --output-md "$RESULT_DIR/summary.md" \
  $PLOT_ARG

log "DONE. See $RESULT_DIR/summary.md"
