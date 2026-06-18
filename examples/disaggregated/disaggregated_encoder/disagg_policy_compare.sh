#!/bin/bash
# Compare encoder-cache eviction policies under E/PD disaggregation, using the
# SAME controllable fixed-pool workload as benchmarks/encoder_cache_eval/run_sweep.sh.
#
# Topology (1E1PD): a dedicated encoder worker (producer, GPU_E) encodes images
# and persists embeddings to a shared store; a prefill+decode worker (consumer,
# GPU_PD) reads them and runs the LLM. A proxy routes E -> PD. The encoder-cache
# policy lives on the producer and governs re-encoding.
#
# REQUIRES the connector delete-propagation (policy evictions delete the
# persisted embedding); otherwise the store is append-only and all policies
# behave identically.
#
# Pipeline:
#   A. generate a fixed pool of K images (one-time)                  [no server]
#   B. if "offline" requested: measure c_i/m_i on a temp single-proc
#      server, then solve lambda* (with PIN_MARGIN)                  [temp server]
#   C. for each policy: bring up the 1E1PD deployment, sweep RPS x REPEATS with
#      --dataset-name mm-fixed-pool (all the run_sweep request knobs), capture
#      per-RPS cache/encoder snapshots, tear down.
#
# All run_sweep request knobs are honored (same env var names):
#   K DISTRIBUTION DISTRIBUTION_PARAM BUCKETS HF_PROCESSOR_KWARGS
#   INPUT_LEN OUTPUT_LEN NUM_PROMPTS NUM_WARMUPS REPEATS RPS_LIST
#   NUM_MM_BASE NUM_MM_RANGE NOVELTY_RATE
#   ENCODER_CACHE_SIZE CACHE_CAPACITY PIN_MARGIN POLICIES
set -euo pipefail

GIT_ROOT=$(git rev-parse --show-toplevel)

###############################################################################
# Config (override via env) -- request knobs mirror run_sweep.sh
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"   # small LLM => encode-heavier
# Read SERVE_HOST, NOT HOST: conda compiler activation exports
# HOST=x86_64-conda-linux-gnu, which is not a bindable address and makes
# `vllm serve --host` fail with socket.gaierror. Decouple from it.
HOST="${SERVE_HOST:-127.0.0.1}"
# Dev routes (/reset_encoder_cache) are required by the c_i measurement and
# the per-RPS cache reset between bench runs; enable by default.
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
# The JIT-built GDN kernel needs the conda libstdc++ (GLIBCXX_3.4.32); make
# sure the active env's lib dir is on the loader path for child vllm serves.
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
RESULT_DIR="${RESULT_DIR:-$(pwd)/disagg_policy_$(date +%Y%m%d_%H%M%S)}"
POOL_DIR="${POOL_DIR:-$RESULT_DIR/pool}"

# Pool spec
K="${K:-100}"
DISTRIBUTION="${DISTRIBUTION:-zipf}"
DISTRIBUTION_PARAM="${DISTRIBUTION_PARAM:-1.1}"
BUCKETS="${BUCKETS:-360x640 720x1280 1080x1920 1440x2560}"
HF_PROCESSOR_KWARGS="${HF_PROCESSOR_KWARGS:-{}}"

# Workload (bench-time)
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
REPEATS="${REPEATS:-1}"
RPS_LIST="${RPS_LIST:-4 8 12}"
NUM_MM_BASE="${NUM_MM_BASE:-1}"
NUM_MM_RANGE="${NUM_MM_RANGE:-0.0}"
NOVELTY_RATE="${NOVELTY_RATE:-0.0}"

# Policy / cache
POLICIES="${POLICIES:-fifo nocache offline}"
ENCODER_CACHE_SIZE="${ENCODER_CACHE_SIZE:-}"     # producer cache B (tokens)
CACHE_CAPACITY="${CACHE_CAPACITY:-}"             # B used to solve lambda*; falls back to ENCODER_CACHE_SIZE
PIN_MARGIN="${PIN_MARGIN:-0.0}"
STATS_INTERVAL="${STATS_INTERVAL:-2}"

# Disagg deployment
ENCODE_PORT="${ENCODE_PORT:-19534}"
PD_PORT="${PD_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"
MEASURE_PORT="${MEASURE_PORT:-19540}"
GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"
EC_STORE="${EC_STORE:-/tmp/ec_cache_policy}"
ENCODE_MAX_NUM_SEQS="${ENCODE_MAX_NUM_SEQS:-16}"  # throttle encode -> bottleneck
PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL_E="${GPU_MEM_UTIL_E:-0.30}"
GPU_MEM_UTIL_PD="${GPU_MEM_UTIL_PD:-0.70}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

mkdir -p "$RESULT_DIR/runs"
declare -a PIDS=()
# Processor-kwargs server flag, built once (array-safe for JSON with spaces).
declare -a PROC_ARG=()
if [ "$HF_PROCESSOR_KWARGS" != "{}" ]; then
  PROC_ARG=(--mm-processor-kwargs "$HF_PROCESSOR_KWARGS")
fi
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c "
    until curl -s localhost:$port/v1/chat/completions >/dev/null 2>&1; do sleep 1; done" \
    && return 0 || { log "ERROR: server :$port not ready in ${TIMEOUT_SECONDS}s"; return 1; }
}
kill_pids() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]:-}"; do kill -9 "$pid" 2>/dev/null || true; done
  pkill -9 -f "vllm serve.*--port $ENCODE_PORT" 2>/dev/null || true
  pkill -9 -f "vllm serve.*--port $PD_PORT" 2>/dev/null || true
  pkill -9 -f "vllm serve.*--port $MEASURE_PORT" 2>/dev/null || true
  pkill -9 -f "disagg_epd_proxy.py.*--port $PROXY_PORT" 2>/dev/null || true
  PIDS=()
  sleep 3
}
cleanup() { log "cleanup..."; trap - INT TERM EXIT; kill_pids; exit "${1:-0}"; }
trap 'cleanup 1' INT TERM
trap 'cleanup 0' EXIT

###############################################################################
# Stage A: pool generation (images + p_i)
###############################################################################
if [ ! -f "$POOL_DIR/pool_spec.json" ]; then
  log "generate pool K=$K dist=$DISTRIBUTION/$DISTRIBUTION_PARAM"
  python "$GIT_ROOT/tools/precompute_mm_pool.py" --pool-dir "$POOL_DIR" generate \
    --k "$K" --seed 0 \
    --distribution "$DISTRIBUTION" --distribution-param "$DISTRIBUTION_PARAM" \
    --bucket-config $BUCKETS \
    --model-id "$MODEL" \
    --hf-processor-kwargs "$HF_PROCESSOR_KWARGS"
else
  log "reusing pool $POOL_DIR/pool_spec.json"
fi

###############################################################################
# Stage B: measure c_i/m_i + solve lambda* (only if offline is requested)
###############################################################################
needs_offline=0
for p in $POLICIES; do [ "$p" = "offline" ] && needs_offline=1; done

if [ "$needs_offline" = "1" ] && [ ! -f "$POOL_DIR/mm_pool.json" ]; then
  log "measuring c_i/m_i on temp single-process server :$MEASURE_PORT"
  CUDA_VISIBLE_DEVICES="$GPU_E" \
  env VLLM_ENCODER_CACHE_POLICY=fifo \
      vllm serve "$MODEL" \
        --host "$HOST" --port "$MEASURE_PORT" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL_PD" \
        --enforce-eager \
        "${PROC_ARG[@]}" \
        >"$RESULT_DIR/measure.log" 2>&1 &
  PIDS+=($!)
  wait_for_server "$MEASURE_PORT"
  python "$GIT_ROOT/tools/precompute_mm_pool.py" --pool-dir "$POOL_DIR" measure \
    --base-url "http://${HOST}:${MEASURE_PORT}" \
    --model "$MODEL" --warmup 2 --repeats 5 --use-token-query \
    | tee "$RESULT_DIR/measure_cim.log"
  kill_pids

  # Capacity B used to solve lambda*. Prefer CACHE_CAPACITY, else
  # ENCODER_CACHE_SIZE, else the per-step token budget (vLLM default for B).
  local_cap="$CACHE_CAPACITY"
  [ -z "$local_cap" ] && local_cap="$ENCODER_CACHE_SIZE"
  [ -z "$local_cap" ] && local_cap="$MAX_NUM_BATCHED_TOKENS"
  log "solving lambda* B=$local_cap pin_margin=$PIN_MARGIN"
  python "$GIT_ROOT/tools/precompute_mm_pool.py" --pool-dir "$POOL_DIR" solve \
    --cache-capacity "$local_cap" --pin-margin "$PIN_MARGIN" --dump-dual-curve \
    | tee "$RESULT_DIR/lambda_solve.log"
fi

###############################################################################
# Stage C: per-policy disagg sweep
###############################################################################
start_disagg() {
  local policy=$1 enc_log=$2 pd_log=$3 proxy_log=$4
  rm -rf "$EC_STORE"; mkdir -p "$EC_STORE"

  local enc_ec_env="" offline_env=""
  [ -n "$ENCODER_CACHE_SIZE" ] && enc_ec_env="VLLM_ENCODER_CACHE_SIZE=$ENCODER_CACHE_SIZE"
  if [ "$policy" = "offline" ]; then
    offline_env="VLLM_ENCODER_CACHE_POLICY_CONFIG=$POOL_DIR/mm_pool.json"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_E" \
  env VLLM_ENCODER_CACHE_POLICY="$policy" \
      VLLM_ENCODER_CACHE_STATS_INTERVAL_SEC="$STATS_INTERVAL" \
      VLLM_TRACK_ENCODER_FORWARD_TIME=1 \
      VLLM_ENCODER_FORWARD_LOG_INTERVAL_SEC="$STATS_INTERVAL" \
      $enc_ec_env $offline_env \
      vllm serve "$MODEL" \
        --host "$HOST" --port "$ENCODE_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL_E" \
        --enforce-eager --no-async-scheduling \
        --enable-request-id-headers --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs "$ENCODE_MAX_NUM_SEQS" \
        --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
        "${PROC_ARG[@]}" \
        --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_producer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
        >"$enc_log" 2>&1 &
  PIDS+=($!)

  CUDA_VISIBLE_DEVICES="$GPU_PD" \
      vllm serve "$MODEL" \
        --host "$HOST" --port "$PD_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL_PD" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager --no-async-scheduling --enable-request-id-headers \
        --max-num-seqs "$PD_MAX_NUM_SEQS" \
        --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
        "${PROC_ARG[@]}" \
        --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_consumer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
        >"$pd_log" 2>&1 &
  PIDS+=($!)

  wait_for_server "$ENCODE_PORT"
  wait_for_server "$PD_PORT"

  ( cd "${GIT_ROOT}/examples/disaggregated/disaggregated_encoder" &&
    python disagg_epd_proxy.py --host 0.0.0.0 --port "$PROXY_PORT" \
      --encode-servers-urls "http://localhost:$ENCODE_PORT" \
      --prefill-servers-urls "disable" \
      --decode-servers-urls "http://localhost:$PD_PORT" \
      >"$proxy_log" 2>&1 ) &
  PIDS+=($!)
  wait_for_server "$PROXY_PORT"
}

run_policy() {
  local policy=$1
  local enc_log="$RESULT_DIR/${policy}.encoder.log"
  log "=== policy=$policy ==="
  start_disagg "$policy" "$enc_log" \
    "$RESULT_DIR/${policy}.pd.log" "$RESULT_DIR/${policy}.proxy.log"

  for RPS in $RPS_LIST; do
    local rps_tag; rps_tag=$(printf "%07.2f" "$RPS")
    for REP in $(seq 0 $((REPEATS - 1))); do
      local out="$RESULT_DIR/runs/${policy}_rps${rps_tag}_rep${REP}.json"
      log "  policy=$policy rps=$RPS rep=$REP"
      vllm bench serve \
        --backend openai-chat --base-url "http://${HOST}:${PROXY_PORT}" \
        --endpoint /v1/chat/completions --model "$MODEL" \
        --dataset-name mm-fixed-pool --mm-pool-dir "$POOL_DIR" \
        --random-mm-base-items-per-request "$NUM_MM_BASE" \
        --random-mm-num-mm-items-range-ratio "$NUM_MM_RANGE" \
        --mm-novelty-rate "$NOVELTY_RATE" \
        --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio 0.0 \
        --num-prompts "$NUM_PROMPTS" --num-warmups "$NUM_WARMUPS" \
        --request-rate "$RPS" --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
        --save-result --result-filename "$out" \
        >"$RESULT_DIR/runs/${policy}_rps${rps_tag}_rep${REP}.bench.log" 2>&1 || \
        log "    WARN: bench nonzero (see ${policy}_rps${rps_tag}_rep${REP}.bench.log)"
      # Per-RPS cumulative producer-side snapshots.
      grep "encoder_cache "   "$enc_log" | tail -1 > "${out%.json}.cache.txt" || true
      grep "encoder_forward "  "$enc_log" | tail -1 > "${out%.json}.enc.txt"   || true
    done
  done
  kill_pids
}

log "model=$MODEL policies=[$POLICIES] B=${ENCODER_CACHE_SIZE:-default} pin_margin=$PIN_MARGIN"
log "workload: K=$K img/req=$NUM_MM_BASE novelty=$NOVELTY_RATE in=$INPUT_LEN out=$OUTPUT_LEN n=$NUM_PROMPTS rps=[$RPS_LIST]"
for policy in $POLICIES; do
  run_policy "$policy"
done

log "done. results in $RESULT_DIR"

# ── Aggregate + plot (set SKIP_AGGREGATE=1 to skip) ─────────────────────────
AGG="$GIT_ROOT/benchmarks/encoder_cache_eval/aggregate.py"
if [ "${SKIP_AGGREGATE:-0}" = "1" ]; then
  log "SKIP_AGGREGATE=1 — aggregate manually:"
  log "  python $AGG --result-dir $RESULT_DIR/runs --pool-spec $POOL_DIR/pool_spec.json --output-csv $RESULT_DIR/summary.csv --output-md $RESULT_DIR/summary.md --plot-dir $RESULT_DIR/plots"
else
  log "aggregating -> summary.csv / summary.md / plots/  (matplotlib needed for PNGs; csv+md written regardless)"
  python "$AGG" \
    --result-dir "$RESULT_DIR/runs" \
    --pool-spec  "$POOL_DIR/pool_spec.json" \
    --output-csv "$RESULT_DIR/summary.csv" \
    --output-md  "$RESULT_DIR/summary.md" \
    --plot-dir   "$RESULT_DIR/plots" \
    && {
      if [ -f "$RESULT_DIR/summary.md" ]; then
        echo "==================== summary.md ===================="
        cat "$RESULT_DIR/summary.md"
        echo "===================================================="
      fi
      log "summary: $RESULT_DIR/summary.md   csv: $RESULT_DIR/summary.csv   plots: $RESULT_DIR/plots/"
    } || log "WARN: aggregate failed — run manually (args logged above with SKIP_AGGREGATE=1)"
fi
