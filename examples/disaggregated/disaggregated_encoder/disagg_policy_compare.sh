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
# 1E + N PD: set GPU_PD_LIST="1 2 3" for 1E3PD. Each PD i serves on PD_PORT+i
# and the proxy random-balances across them. Tripling PD capacity shifts the
# bottleneck onto the single encoder, so the encoder-cache policy can finally
# show up in throughput / TTFT. Defaults to the single GPU_PD (1E1PD).
GPU_PD_LIST="${GPU_PD_LIST:-$GPU_PD}"
read -r -a GPU_PD_ARR <<< "$GPU_PD_LIST"
NUM_PD=${#GPU_PD_ARR[@]}
# CUDA graphs for the E/PD serving engines.
#   ENFORCE_EAGER=1 (default): --enforce-eager. Fast startup, robust.
#   ENFORCE_EAGER=0: enable PIECEWISE CUDA graphs. We force
#     cudagraph_mode=PIECEWISE because the auto-default FULL_AND_PIECEWISE
#     crashes qwen3_next's GDN linear attention during FULL decode-graph
#     capture (KeyError ...linear_attn). PIECEWISE captures fine and still
#     speeds decode (less than FULL). Longer warmup; helps decode/TPOT, not
#     the prefill/encoder bottleneck. Same for fifo/offline -> comparison
#     unchanged. The c_i measurement server stays eager regardless.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
declare -a CG_ARG=()
if [ "$ENFORCE_EAGER" = "1" ]; then
  CG_ARG=(--enforce-eager)
else
  CG_ARG=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi
EC_STORE="${EC_STORE:-/tmp/ec_cache_policy}"
ENCODE_MAX_NUM_SEQS="${ENCODE_MAX_NUM_SEQS:-16}"  # throttle encode -> bottleneck
PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL_E="${GPU_MEM_UTIL_E:-0.30}"
GPU_MEM_UTIL_PD="${GPU_MEM_UTIL_PD:-0.70}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
# Queue sampler interval (seconds, fractional ok). Finer => catches shorter
# transient queues that a 1s tick misses. Note num_requests_waiting itself is
# only refreshed per scheduler step (~tens of ms), so below ~0.05 is pointless.
QUEUE_SAMPLE_SEC="${QUEUE_SAMPLE_SEC:-0.25}"

mkdir -p "$RESULT_DIR/runs"
declare -a PIDS=()
# Processor-kwargs server flag, built once (array-safe for JSON with spaces).
declare -a PROC_ARG=()
if [ "$HF_PROCESSOR_KWARGS" != "{}" ]; then
  PROC_ARG=(--mm-processor-kwargs "$HF_PROCESSOR_KWARGS")
fi
# mm_processor_cache type (applied to E/producer only). Default "" keeps vLLM's
# default "lru": API-side sender cache (skips HF preprocess on repeats) but the
# worker has NO receiver cache, so pixel_values are re-sent over API->worker IPC
# every step. MM_CACHE_TYPE=shm puts processed pixel_values in a shared-memory
# object store the worker reads directly -> repeated images skip the IPC
# re-transfer, targeting the engine-orchestration (~134ms) bucket of
# encoder_fanout. Only effective on a single-API-process engine (E qualifies).
declare -a MM_CACHE_ARG=()
if [ "${MM_CACHE_TYPE:-}" = "shm" ]; then
  MM_CACHE_ARG=(--mm-processor-cache-type shm
    --mm-shm-cache-max-object-size-mb "${MM_SHM_OBJ_MB:-256}")
fi
# Torch-profiler trace pass (PROFILE=1). Captures a BOUNDED steady-state trace
# at the REAL operating point (same RPS / policy / warmed cache) rather than the
# clean RPS sweep. The profiled run's latency is overhead-inflated, so its bench
# numbers are DISCARDED — only the trace (Chrome/Perfetto, viewable at
# https://ui.perfetto.dev) is kept. When PROFILE=1, run_policy does one trace
# pass per policy and skips the measurement sweep + aggregation entirely, so
# clean numbers and traces never come from the same run. Profile E by default;
# PROFILE_PD=1 also traces a PD worker. The torch schedule keeps overhead off
# the wait window and records only PROFILE_ACTIVE steady steps.
PROFILE="${PROFILE:-0}"
PROFILE_PD="${PROFILE_PD:-0}"
PROFILE_RPS="${PROFILE_RPS:-}"                  # default = last value of RPS_LIST
PROFILE_PROMPTS="${PROFILE_PROMPTS:-64}"
PROFILE_WAIT="${PROFILE_WAIT:-20}"              # steps skipped before recording (zero overhead)
PROFILE_WARMUP="${PROFILE_WARMUP:-2}"           # discarded steps (JIT noise)
PROFILE_ACTIVE="${PROFILE_ACTIVE:-15}"          # recorded steady steps
PROFILE_DIR="$RESULT_DIR/traces"
declare -a PROFILE_ARG=()
if [ "$PROFILE" = "1" ]; then
  mkdir -p "$PROFILE_DIR"
  PROFILE_ARG=(--profiler-config \
    "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":true,\"wait_iterations\":$PROFILE_WAIT,\"warmup_iterations\":$PROFILE_WARMUP,\"active_iterations\":$PROFILE_ACTIVE}")
  [ -z "$PROFILE_RPS" ] && PROFILE_RPS="${RPS_LIST##* }"  # last token of RPS_LIST
fi
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c "
    until curl -s localhost:$port/v1/chat/completions >/dev/null 2>&1; do sleep 1; done" \
    && return 0 || { log "ERROR: server :$port not ready in ${TIMEOUT_SECONDS}s"; return 1; }
}
# Snapshot the queue/prefill/decode/ttft histogram _sum and _count across all
# PD workers' /metrics into $1. Diffing two snapshots gives per-RPS averages.
scrape_phases() {
  local f=$1 i
  : > "$f"
  for i in $(seq 0 $((NUM_PD - 1))); do
    curl -s "http://${HOST}:$((PD_PORT + i))/metrics" 2>/dev/null
  done | grep -E "^vllm:(time_to_first_token_seconds|request_queue_time_seconds|request_prefill_time_seconds|request_decode_time_seconds)_(sum|count)" >> "$f" 2>/dev/null || true
}
# Background sampler: every QUEUE_SAMPLE_SEC seconds, record each engine's
# instantaneous queue depth (num_requests_waiting), running count, and KV-cache
# usage into $1 (CSV). Run during a bench and kill afterwards; queue_summary.py
# then shows which stage (E encoder vs PD workers) starts queuing first as load
# rises. `t` is a high-resolution (sub-second) timestamp so each sampling round
# is unique — queue_summary.py groups PD engines by `t`, which would mis-sum if
# multiple sub-second rounds shared the same integer second.
sample_queues() {
  local out=$1 s name port body w r kv now
  local -a specs=("E:$ENCODE_PORT")
  local j
  for j in $(seq 0 $((NUM_PD - 1))); do specs+=("PD${j}:$((PD_PORT + j))"); done
  echo "t,engine,waiting,running,kv" > "$out"
  while true; do
    now=$(date +%s.%N)
    for s in "${specs[@]}"; do
      name="${s%%:*}"; port="${s##*:}"
      body=$(curl -s "http://${HOST}:${port}/metrics" 2>/dev/null)
      w=$(printf '%s\n' "$body" | awk '/^vllm:num_requests_waiting/{x+=$NF} END{print x+0}')
      r=$(printf '%s\n' "$body" | awk '/^vllm:num_requests_running/{x+=$NF} END{print x+0}')
      kv=$(printf '%s\n' "$body" | awk '/^vllm:kv_cache_usage_perc/{print $NF; exit}')
      echo "${now},${name},${w:-0},${r:-0},${kv:-0}" >> "$out"
    done
    sleep "$QUEUE_SAMPLE_SEC"
  done
}
kill_pids() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]:-}"; do kill -9 "$pid" 2>/dev/null || true; done
  pkill -9 -f "vllm serve.*--port $ENCODE_PORT" 2>/dev/null || true
  local i
  for i in $(seq 0 $((NUM_PD - 1))); do
    pkill -9 -f "vllm serve.*--port $((PD_PORT + i))" 2>/dev/null || true
  done
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
        "${CG_ARG[@]}" --no-async-scheduling \
        --enable-request-id-headers --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs "$ENCODE_MAX_NUM_SEQS" \
        --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
        "${PROC_ARG[@]}" "${MM_CACHE_ARG[@]}" "${PROFILE_ARG[@]}" \
        --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_producer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
        >"$enc_log" 2>&1 &
  PIDS+=($!)

  # Launch NUM_PD consumer (PD) workers, one per GPU in GPU_PD_LIST, each on
  # PD_PORT+i. Collect their URLs for the proxy to balance across.
  local d_urls="" i pd_gpu pd_port this_pd_log
  for i in "${!GPU_PD_ARR[@]}"; do
    pd_gpu="${GPU_PD_ARR[$i]}"
    pd_port=$((PD_PORT + i))
    this_pd_log="${pd_log%.log}.${i}.log"
    CUDA_VISIBLE_DEVICES="$pd_gpu" \
        vllm serve "$MODEL" \
          --host "$HOST" --port "$pd_port" \
          --gpu-memory-utilization "$GPU_MEM_UTIL_PD" \
          --max-model-len "$MAX_MODEL_LEN" \
          "${CG_ARG[@]}" --no-async-scheduling --enable-request-id-headers \
          --max-num-seqs "$PD_MAX_NUM_SEQS" \
          --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
          "${PROC_ARG[@]}" "${PROFILE_ARG[@]}" \
          --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_consumer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
          >"$this_pd_log" 2>&1 &
    PIDS+=($!)
    d_urls="${d_urls:+$d_urls,}http://localhost:$pd_port"
  done

  wait_for_server "$ENCODE_PORT"
  for i in "${!GPU_PD_ARR[@]}"; do wait_for_server "$((PD_PORT + i))"; done

  ( cd "${GIT_ROOT}/examples/disaggregated/disaggregated_encoder" &&
    python disagg_epd_proxy.py --host 0.0.0.0 --port "$PROXY_PORT" \
      --encode-servers-urls "http://localhost:$ENCODE_PORT" \
      --prefill-servers-urls "disable" \
      --decode-servers-urls "$d_urls" \
      >"$proxy_log" 2>&1 ) &
  PIDS+=($!)
  wait_for_server "$PROXY_PORT"
}

# Drive a short load through the proxy at $1 RPS (numbers discarded; used to
# warm the cache and to fill the profiler's recording window). Output -> $2.
_profile_bench() {
  vllm bench serve \
    --backend openai-chat --base-url "http://${HOST}:${PROXY_PORT}" \
    --endpoint /v1/chat/completions --model "$MODEL" \
    --dataset-name mm-fixed-pool --mm-pool-dir "$POOL_DIR" \
    --random-mm-base-items-per-request "$NUM_MM_BASE" \
    --random-mm-num-mm-items-range-ratio "$NUM_MM_RANGE" \
    --mm-novelty-rate "$NOVELTY_RATE" \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --num-prompts "$PROFILE_PROMPTS" --num-warmups 0 \
    --request-rate "$1" --ignore-eos \
    >"$2" 2>&1 || log "    WARN: profile bench nonzero (see $2)"
}

# One bounded trace pass at the real operating point: warm the cache, then
# record a steady-state window on E (and optionally a PD). Latencies here are
# profiler-inflated, so the bench numbers are discarded — only the torch trace
# (under $PROFILE_DIR) is kept.
profile_pass() {
  local policy=$1
  local pdir="$PROFILE_DIR/$policy"
  mkdir -p "$pdir"
  local -a turls=("http://${HOST}:${ENCODE_PORT}")
  [ "$PROFILE_PD" = "1" ] && turls+=("http://${HOST}:${PD_PORT}")
  log "  PROFILE policy=$policy rps=$PROFILE_RPS engines=[${turls[*]}] (numbers discarded)"
  log "    warming cache..."
  _profile_bench "$PROFILE_RPS" "$pdir/warmup.bench.log"
  local u
  for u in "${turls[@]}"; do
    curl -s -X POST "$u/start_profile" >/dev/null && log "    started profile @ $u" \
      || log "    WARN: start_profile $u failed (is --profiler-config set?)"
  done
  log "    recording steady window..."
  _profile_bench "$PROFILE_RPS" "$pdir/trace.bench.log"
  for u in "${turls[@]}"; do
    curl -s -X POST "$u/stop_profile" >/dev/null && log "    stopped profile @ $u" \
      || log "    WARN: stop_profile $u failed"
  done
  log "  PROFILE done -> $PROFILE_DIR/  (*.json.gz; open in https://ui.perfetto.dev)"
}

run_policy() {
  local policy=$1
  local enc_log="$RESULT_DIR/${policy}.encoder.log"
  log "=== policy=$policy ==="
  start_disagg "$policy" "$enc_log" \
    "$RESULT_DIR/${policy}.pd.log" "$RESULT_DIR/${policy}.proxy.log"

  # PROFILE mode: one bounded trace pass at the real operating point, then
  # tear down. No measurement sweep / aggregation (kept on the clean PROFILE=0
  # run), so profiler overhead never contaminates the reported numbers.
  if [ "$PROFILE" = "1" ]; then
    profile_pass "$policy"
    kill_pids
    return
  fi

  for RPS in $RPS_LIST; do
    local rps_tag; rps_tag=$(printf "%07.2f" "$RPS")
    for REP in $(seq 0 $((REPEATS - 1))); do
      local out="$RESULT_DIR/runs/${policy}_rps${rps_tag}_rep${REP}.json"
      log "  policy=$policy rps=$RPS rep=$REP"
      scrape_phases "${out%.json}.phase_before.txt"
      sample_queues "${out%.json}.queues.csv" &
      local sampler_pid=$!
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
      kill "$sampler_pid" 2>/dev/null || true
      scrape_phases "${out%.json}.phase_after.txt"
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

# PROFILE run produces traces, not measurement JSONs — skip aggregation/plots.
if [ "$PROFILE" = "1" ]; then
  log "PROFILE run: torch traces under $PROFILE_DIR/<policy>/ (*.json.gz)."
  log "  view: download + open in https://ui.perfetto.dev (or chrome://tracing)."
  log "  NOTE: latencies in these traces carry profiler overhead — read structure,"
  log "        not absolute ms. Run with PROFILE=0 for clean measurement numbers."
  cleanup 0
fi

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

# ── TTFT phase composition (queue / prefill / decode from PD /metrics) ───────
PHASE="$GIT_ROOT/benchmarks/encoder_cache_eval/phase_breakdown.py"
if [ -f "$PHASE" ]; then
  log "TTFT phase breakdown -> phase_breakdown.md"
  python "$PHASE" "$RESULT_DIR/runs" --md "$RESULT_DIR/phase_breakdown.md" \
    && { echo "================ phase_breakdown.md ================"; \
         cat "$RESULT_DIR/phase_breakdown.md"; \
         echo "===================================================="; } \
    || log "WARN: phase_breakdown failed (non-fatal)"
fi

# ── Bottleneck: which stage (E vs PD) starts queuing first ───────────────────
QSUM="$GIT_ROOT/benchmarks/encoder_cache_eval/queue_summary.py"
if [ -f "$QSUM" ]; then
  log "queue-depth bottleneck summary -> queue_summary.md"
  python "$QSUM" "$RESULT_DIR/runs" --md "$RESULT_DIR/queue_summary.md" \
    && { echo "================ queue_summary.md ================"; \
         cat "$RESULT_DIR/queue_summary.md"; \
         echo "===================================================="; } \
    || log "WARN: queue_summary failed (non-fatal)"
fi

# ── Fine TTFT breakdown: stacked components vs RPS + GPU-encode vs RPS ───────
PLOT="$GIT_ROOT/benchmarks/encoder_cache_eval/plot_ttft_breakdown.py"
if [ -f "$PLOT" ]; then
  log "TTFT fine-breakdown plots -> plots/ttft_breakdown_*.png + producer_encode_vs_rps.png"
  python "$PLOT" "$RESULT_DIR" --md "$RESULT_DIR/ttft_breakdown.md" \
    && { echo "================ ttft_breakdown.md ================"; \
         cat "$RESULT_DIR/ttft_breakdown.md"; \
         echo "===================================================="; } \
    || log "WARN: plot_ttft_breakdown failed (non-fatal)"
fi
