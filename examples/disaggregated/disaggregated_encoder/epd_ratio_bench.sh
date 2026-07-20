#!/bin/bash
# EPD-disaggregation ratio benchmark.
#
# Deploys NUM_E encoder workers + NUM_PD prefill-decode workers behind the
# upstream disagg_epd_proxy.py and benchmarks ONE ratio, then tears everything
# down. Run it once per ratio (deliberately NOT a sweep in a single process, so
# each ratio gets a cold, independent deployment):
#
#   NUM_E=1 bash epd_ratio_bench.sh     # 1E / 7PD
#   NUM_E=2 bash epd_ratio_bench.sh     # 2E / 6PD
#   ...
#   NUM_E=7 bash epd_ratio_bench.sh     # 7E / 1PD
#
# This is a pure harness: it only launches `vllm serve` / the upstream proxy /
# `vllm bench serve`. It does NOT modify any vLLM engine code, so it runs on a
# clean upstream checkout.
#
# Layout: encoders take GPUs [0, NUM_E), PD workers take [NUM_E, TOTAL_GPUS).
# The proxy fans encode requests across all encoders and load-balances the
# prompt across all PD workers (both are comma-separated URL lists).
set -euo pipefail

###############################################################################
# Configuration -- override via env
###############################################################################
MODEL="${MODEL:?set MODEL, e.g. MODEL=/path/to/Qwen3.5-9B}"
NUM_E="${NUM_E:?set NUM_E, the number of encoder workers (1..TOTAL_GPUS-1)}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"
NUM_PD=$((TOTAL_GPUS - NUM_E))

if [ "$NUM_E" -lt 1 ] || [ "$NUM_PD" -lt 1 ]; then
  echo "ERROR: need 1 <= NUM_E <= $((TOTAL_GPUS - 1)); got NUM_E=$NUM_E -> NUM_PD=$NUM_PD" >&2
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
ENCODE_PORT_BASE="${ENCODE_PORT_BASE:-19534}"
PD_PORT_BASE="${PD_PORT_BASE:-19600}"
PROXY_PORT="${PROXY_PORT:-10001}"

# Serve args (defaults follow the upstream disagg_1e1pd_example.sh)
GPU_MEM_UTIL_E="${GPU_MEM_UTIL_E:-0.30}"
GPU_MEM_UTIL_PD="${GPU_MEM_UTIL_PD:-0.70}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# Benchmark args
DATASET_NAME="${DATASET_NAME:-hf}"
DATASET_PATH="${DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
NUM_PROMPTS="${NUM_PROMPTS:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"          # open loop; or use MAX_CONCURRENCY
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"       # set for closed loop (recommended)
SEED="${SEED:-0}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
GIT_ROOT=$(git rev-parse --show-toplevel)
RESULT_DIR="${RESULT_DIR:-$(pwd)/epd_ratio_${NUM_E}E${NUM_PD}PD_$(date +%Y%m%d_%H%M%S)}"
EC_STORE="${EC_STORE:-/tmp/ec_store_${NUM_E}e${NUM_PD}pd}"

mkdir -p "$RESULT_DIR"
declare -a PIDS=()
log() { echo "[$(date '+%H:%M:%S')] $*"; }

declare -a EAGER_ARG=()
[ "$ENFORCE_EAGER" = "1" ] && EAGER_ARG=(--enforce-eager)

###############################################################################
# Helpers
###############################################################################
wait_for_server() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c "
    until curl -s localhost:$port/v1/chat/completions >/dev/null 2>&1; do sleep 1; done" \
    && return 0 || { log "ERROR: server :$port not ready in ${TIMEOUT_SECONDS}s"; return 1; }
}

cleanup() {
  log "cleanup..."
  trap - INT TERM EXIT
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]:-}"; do kill -9 "$pid" 2>/dev/null || true; done
  local i
  for i in $(seq 0 $((NUM_E - 1))); do
    pkill -9 -f "vllm serve.*--port $((ENCODE_PORT_BASE + i))" 2>/dev/null || true
  done
  for i in $(seq 0 $((NUM_PD - 1))); do
    pkill -9 -f "vllm serve.*--port $((PD_PORT_BASE + i))" 2>/dev/null || true
  done
  pkill -9 -f "disagg_epd_proxy.py.*--port $PROXY_PORT" 2>/dev/null || true
  sleep 2
  exit "${1:-0}"
}
trap 'cleanup 1' INT TERM
trap 'cleanup 0' EXIT

###############################################################################
# Fresh EC store
###############################################################################
rm -rf "$EC_STORE"; mkdir -p "$EC_STORE"

log "=== ratio ${NUM_E}E / ${NUM_PD}PD  (TOTAL_GPUS=$TOTAL_GPUS) ==="
log "model=$MODEL  store=$EC_STORE  results=$RESULT_DIR"

###############################################################################
# Encoder workers: GPUs [0, NUM_E)
###############################################################################
e_urls=""
for i in $(seq 0 $((NUM_E - 1))); do
  port=$((ENCODE_PORT_BASE + i))
  CUDA_VISIBLE_DEVICES="$i" vllm serve "$MODEL" \
    --host "$HOST" --port "$port" \
    --gpu-memory-utilization "$GPU_MEM_UTIL_E" \
    "${EAGER_ARG[@]}" \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_producer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
    >"$RESULT_DIR/encoder.$i.log" 2>&1 &
  PIDS+=($!)
  e_urls="${e_urls:+$e_urls,}http://localhost:$port"
  log "  encoder[$i] gpu=$i port=$port"
done

###############################################################################
# Prefill+Decode workers: GPUs [NUM_E, TOTAL_GPUS)
###############################################################################
d_urls=""
for j in $(seq 0 $((NUM_PD - 1))); do
  gpu=$((NUM_E + j))
  port=$((PD_PORT_BASE + j))
  CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$MODEL" \
    --host "$HOST" --port "$port" \
    --gpu-memory-utilization "$GPU_MEM_UTIL_PD" \
    "${EAGER_ARG[@]}" \
    --enable-request-id-headers \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_consumer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
    >"$RESULT_DIR/pd.$j.log" 2>&1 &
  PIDS+=($!)
  d_urls="${d_urls:+$d_urls,}http://localhost:$port"
  log "  pd[$j] gpu=$gpu port=$port"
done

for i in $(seq 0 $((NUM_E - 1))); do wait_for_server $((ENCODE_PORT_BASE + i)); done
for j in $(seq 0 $((NUM_PD - 1))); do wait_for_server $((PD_PORT_BASE + j)); done

###############################################################################
# Proxy (upstream): comma-separated URL lists, random.choice load balancing
###############################################################################
( cd "${GIT_ROOT}/examples/disaggregated/disaggregated_encoder" &&
  python disagg_epd_proxy.py --host 0.0.0.0 --port "$PROXY_PORT" \
    --encode-servers-urls "$e_urls" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "$d_urls" \
    >"$RESULT_DIR/proxy.log" 2>&1 ) &
PIDS+=($!)
wait_for_server "$PROXY_PORT"
log "all services up (E=$NUM_E, PD=$NUM_PD)"

###############################################################################
# Benchmark
###############################################################################
declare -a LOAD_ARG=()
if [ -n "$MAX_CONCURRENCY" ]; then
  LOAD_ARG=(--max-concurrency "$MAX_CONCURRENCY" --request-rate inf)
  log "load: closed loop, max-concurrency=$MAX_CONCURRENCY"
else
  LOAD_ARG=(--request-rate "$REQUEST_RATE")
  log "load: open loop, request-rate=$REQUEST_RATE"
fi

log "running benchmark (n=$NUM_PROMPTS)..."
vllm bench serve \
  --model "$MODEL" \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --base-url "http://${HOST}:${PROXY_PORT}" \
  --dataset-name "$DATASET_NAME" \
  --dataset-path "$DATASET_PATH" \
  --seed "$SEED" \
  --num-prompts "$NUM_PROMPTS" \
  "${LOAD_ARG[@]}" \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
  --save-result --result-filename "$RESULT_DIR/bench.json" \
  2>&1 | tee "$RESULT_DIR/bench.log"

# Record the exact configuration next to the results for reproducibility.
cat >"$RESULT_DIR/meta.json" <<EOF
{
  "num_e": $NUM_E, "num_pd": $NUM_PD, "total_gpus": $TOTAL_GPUS,
  "model": "$MODEL", "git_sha": "$(git -C "$GIT_ROOT" rev-parse HEAD)",
  "branch": "$(git -C "$GIT_ROOT" rev-parse --abbrev-ref HEAD)",
  "gpu_mem_util_e": $GPU_MEM_UTIL_E, "gpu_mem_util_pd": $GPU_MEM_UTIL_PD,
  "max_num_seqs": $MAX_NUM_SEQS, "max_model_len": $MAX_MODEL_LEN,
  "dataset_name": "$DATASET_NAME", "dataset_path": "$DATASET_PATH",
  "num_prompts": $NUM_PROMPTS,
  "request_rate": "$REQUEST_RATE", "max_concurrency": "${MAX_CONCURRENCY:-}",
  "ec_store": "$EC_STORE"
}
EOF

log "done -> $RESULT_DIR  (bench.json / meta.json / *.log)"
