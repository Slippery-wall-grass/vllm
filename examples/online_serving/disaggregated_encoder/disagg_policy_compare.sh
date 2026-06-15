#!/bin/bash
# Compare encoder-cache eviction policies under E/PD disaggregation.
#
# Topology (1E1PD): a dedicated encoder worker (producer) on GPU_E encodes
# images and persists embeddings to a shared store; a prefill+decode worker
# (consumer) on GPU_PD reads them and runs the LLM. A proxy routes E -> PD.
#
# The encoder-cache *policy* (fifo / nocache / offline) lives on the producer
# and governs whether a repeat image is re-encoded. IMPORTANT: this comparison
# is only meaningful with the connector's delete-propagation (policy evictions
# delete the persisted embedding); otherwise the store is append-only, the
# producer never re-encodes, and all policies behave identically.
#
# For each policy we restart both workers (so the policy env var takes effect),
# wipe the shared store, run the benchmark, and save the encoder worker's log
# (which carries the periodic `encoder_cache ...` policy/occupancy lines).
#
# Override any knob via env, e.g.:
#   POLICIES="fifo offline" NUM_PROMPTS=300 ./disagg_policy_compare.sh
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"   # small LLM => encode-heavier
RESULT_DIR="${RESULT_DIR:-./disagg_policy_$(date +%Y%m%d_%H%M%S)}"
POLICIES="${POLICIES:-fifo nocache offline}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PD_PORT="${PD_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"
GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"

EC_STORE="${EC_STORE:-/tmp/ec_cache_policy}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

# Workload
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-inf}"          # inf = closed-loop max load
DATASET_NAME="${DATASET_NAME:-hf}"
DATASET_PATH="${DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
OUTPUT_LEN="${OUTPUT_LEN:-}"                  # empty = dataset default

# Encoder-cache size B on the producer (tokens). Decoupled from the per-step
# token budget via VLLM_ENCODER_CACHE_SIZE (read in SchedulerConfig).
ENCODER_CACHE_SIZE="${ENCODER_CACHE_SIZE:-}"
# offline policy config (mm_pool.json from tools/precompute_mm_pool.py solve).
OFFLINE_POOL_JSON="${OFFLINE_POOL_JSON:-}"
PIN_MARGIN="${PIN_MARGIN:-0.0}"   # informational; bake into the pool json's solve
STATS_INTERVAL="${STATS_INTERVAL:-2}"

# Throttle the encoder worker so it can become the pipeline bottleneck (that is
# the regime where the policy matters). Lower => encode saturates sooner.
ENCODE_MAX_NUM_SEQS="${ENCODE_MAX_NUM_SEQS:-16}"
PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-128}"

GIT_ROOT=$(git rev-parse --show-toplevel)
mkdir -p "$RESULT_DIR"
declare -a PIDS=()

###############################################################################
# Helpers
###############################################################################
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c "
    until curl -s localhost:$port/v1/chat/completions >/dev/null 2>&1; do sleep 1; done" \
    && return 0 || { log "ERROR: server on :$port not ready in ${TIMEOUT_SECONDS}s"; return 1; }
}

kill_pids() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]:-}"; do kill -9 "$pid" 2>/dev/null || true; done
  # Backstop: any leftover servers on our ports / EC workers.
  pkill -9 -f "vllm serve.*--port $ENCODE_PORT" 2>/dev/null || true
  pkill -9 -f "vllm serve.*--port $PD_PORT" 2>/dev/null || true
  pkill -9 -f "disagg_epd_proxy.py.*--port $PROXY_PORT" 2>/dev/null || true
  PIDS=()
  sleep 3
}

cleanup() { log "cleanup..."; trap - INT TERM EXIT; kill_pids; exit "${1:-0}"; }
trap 'cleanup 1' INT TERM
trap 'cleanup 0' EXIT

###############################################################################
# Per-policy run
###############################################################################
run_policy() {
  local policy=$1
  local enc_log="$RESULT_DIR/${policy}.encoder.log"
  local pd_log="$RESULT_DIR/${policy}.pd.log"
  local proxy_log="$RESULT_DIR/${policy}.proxy.log"
  local bench_json="$RESULT_DIR/${policy}.result.json"

  log "=== policy=$policy ==="

  # Fresh store every policy so prior embeddings don't leak across policies.
  rm -rf "$EC_STORE"; mkdir -p "$EC_STORE"

  local enc_ec_env=""
  if [ -n "$ENCODER_CACHE_SIZE" ]; then
    enc_ec_env="VLLM_ENCODER_CACHE_SIZE=$ENCODER_CACHE_SIZE"
  fi
  local offline_env=""
  if [ "$policy" = "offline" ]; then
    if [ -z "$OFFLINE_POOL_JSON" ] || [ ! -f "$OFFLINE_POOL_JSON" ]; then
      log "ERROR: policy=offline needs OFFLINE_POOL_JSON pointing at mm_pool.json"
      return 1
    fi
    offline_env="VLLM_ENCODER_CACHE_POLICY_CONFIG=$OFFLINE_POOL_JSON"
  fi

  # --- Encoder worker (producer): runs the policy ---
  CUDA_VISIBLE_DEVICES="$GPU_E" \
  env VLLM_ENCODER_CACHE_POLICY="$policy" \
      VLLM_ENCODER_CACHE_STATS_INTERVAL_SEC="$STATS_INTERVAL" \
      VLLM_TRACK_ENCODER_FORWARD_TIME=1 \
      VLLM_ENCODER_FORWARD_LOG_INTERVAL_SEC="$STATS_INTERVAL" \
      $enc_ec_env $offline_env \
      vllm serve "$MODEL" \
        --gpu-memory-utilization 0.30 \
        --port "$ENCODE_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs "$ENCODE_MAX_NUM_SEQS" \
        --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
        --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_producer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
        >"$enc_log" 2>&1 &
  PIDS+=($!)

  # --- Prefill+Decode worker (consumer) ---
  CUDA_VISIBLE_DEVICES="$GPU_PD" \
      vllm serve "$MODEL" \
        --gpu-memory-utilization 0.70 \
        --port "$PD_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs "$PD_MAX_NUM_SEQS" \
        --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
        --ec-transfer-config "{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_consumer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"$EC_STORE\"}}" \
        >"$pd_log" 2>&1 &
  PIDS+=($!)

  wait_for_server "$ENCODE_PORT"
  wait_for_server "$PD_PORT"

  # --- Proxy ---
  ( cd "${GIT_ROOT}/examples/online_serving/disaggregated_encoder" &&
    python disagg_epd_proxy.py \
      --host 0.0.0.0 --port "$PROXY_PORT" \
      --encode-servers-urls "http://localhost:$ENCODE_PORT" \
      --prefill-servers-urls "disable" \
      --decode-servers-urls "http://localhost:$PD_PORT" \
      >"$proxy_log" 2>&1 ) &
  PIDS+=($!)
  wait_for_server "$PROXY_PORT"

  # --- Benchmark ---
  log "  benchmarking policy=$policy -> $(basename "$bench_json")"
  local out_len_arg=()
  [ -n "$OUTPUT_LEN" ] && out_len_arg=(--hf-output-len "$OUTPUT_LEN")
  vllm bench serve \
    --model "$MODEL" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --seed 0 \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --port "$PROXY_PORT" \
    --save-result --result-dir "$RESULT_DIR" --result-filename "${policy}.result.json" \
    "${out_len_arg[@]}" >"$RESULT_DIR/${policy}.bench.log" 2>&1 || \
    log "  WARN: bench for $policy returned nonzero (see ${policy}.bench.log)"

  # Capture the final cumulative encoder_cache stats line (hits / forced_unpin /
  # occupancy) from the producer log.
  grep "encoder_cache " "$enc_log" | tail -1 > "$RESULT_DIR/${policy}.cache.txt" || true
  grep "encoder_forward " "$enc_log" | tail -1 > "$RESULT_DIR/${policy}.enc.txt" || true

  kill_pids
}

###############################################################################
# Main
###############################################################################
log "model=$MODEL  policies=[$POLICIES]  result_dir=$RESULT_DIR"
log "store=$EC_STORE  encode_max_seqs=$ENCODE_MAX_NUM_SEQS  pd_max_seqs=$PD_MAX_NUM_SEQS"
for policy in $POLICIES; do
  run_policy "$policy"
done

echo
echo "=================== summary ==================="
printf "%-10s %-12s %-10s %-14s %-12s\n" policy throughput hit_rate forced_unpin enc_secs
for policy in $POLICIES; do
  bj="$RESULT_DIR/${policy}.result.json"
  cache="$RESULT_DIR/${policy}.cache.txt"
  enc="$RESULT_DIR/${policy}.enc.txt"
  tput=$(python -c "import json;print(f\"{json.load(open('$bj')).get('request_throughput',0):.3f}\")" 2>/dev/null || echo "-")
  hr=$(grep -oE "hit_rate=[0-9.]+" "$cache" 2>/dev/null | cut -d= -f2 || echo "-")
  fu=$(grep -oE "forced_unpin=[0-9]+" "$cache" 2>/dev/null | cut -d= -f2 || echo "-")
  es=$(grep -oE "cumulative_secs=[0-9.]+" "$enc" 2>/dev/null | cut -d= -f2 || echo "-")
  printf "%-10s %-12s %-10s %-14s %-12s\n" "$policy" "${tput:--}" "${hr:--}" "${fu:--}" "${es:--}"
done
echo "==============================================="
echo "per-policy logs + result.json in $RESULT_DIR"
