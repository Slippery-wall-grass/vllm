#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end comparison of None / FIFO / Distribution-Aware encoder cache
# policies.
#
# This script:
#   1. Generates K synthetic test images
#   2. Profiles encoder computation time for each image type (offline)
#   3. Solves for optimal lambda*
#   4. Deploys 1E1P1D and benchmarks with no-cache baseline
#   5. Restarts with FIFO and benchmarks
#   6. Restarts with distribution-aware and benchmarks
#   7. Prints 3-way comparison: TTFT, throughput, hit-rate, improvement
#      vs the no-cache baseline
#
# Usage:
#   bash run_cache_comparison.sh
#
# Override defaults via environment variables:
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct NUM_TYPES=5 NUM_REQUESTS=2000 \
#     BENCH_MODE=closed CONCURRENCY=32 WARMUP_REQUESTS=400 \
#     bash run_cache_comparison.sh
#
# BENCH_MODE=closed (default) runs a fixed-concurrency saturation test and
# reports max throughput; BENCH_MODE=open fires at fixed QPS (uses $QPS).
#
# To use images from a real dataset (e.g. VisionArena-Chat) instead of the
# synthetic generator, run preprocess_real_dataset.py first, then:
#   SKIP_GENERATION=1 WORK_DIR=/tmp/real_data IMAGE_DIR=/tmp/real_data/images \
#     DISTRIBUTION="$(cat /tmp/real_data/distribution_real.json)" \
#     bash run_cache_comparison.sh
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
NUM_TYPES="${NUM_TYPES:-5}"            # number of image types K
NUM_VIDEOS="${NUM_VIDEOS:-0}"          # number of video types V (0 = no video)
# Fixed duration (used when min/max are not both set)
VIDEO_DURATION_S="${VIDEO_DURATION_S:-2}"
# When both set, each video gets a random duration uniformly in [min, max]
VIDEO_DURATION_MIN_S="${VIDEO_DURATION_MIN_S:-}"
VIDEO_DURATION_MAX_S="${VIDEO_DURATION_MAX_S:-}"
VIDEO_FPS="${VIDEO_FPS:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-2000}"
# Load mode: "closed" drives the server to saturation for throughput
# measurement (recommended); "open" fires at a fixed QPS for latency-at-load.
BENCH_MODE="${BENCH_MODE:-closed}"
QPS="${QPS:-2}"
CONCURRENCY="${CONCURRENCY:-32}"
# Warmup requests are excluded from metrics so the cache reaches steady
# state before measurement begins.
WARMUP_REQUESTS="${WARMUP_REQUESTS:-400}"
# Number of independent rounds per policy. Each round uses a different
# seed; results are aggregated with mean ± std to reduce noise.
NUM_ROUNDS="${NUM_ROUNDS:-3}"
# Trimmed mean: drop the N highest and N lowest values for each metric
# when aggregating across rounds. Recommended TRIM=1 for NUM_ROUNDS>=5
# to remove outliers (e.g. residual first-round warmup effects).
TRIM="${TRIM:-0}"
# Global warmup before round 1 — throwaway requests to warm up CUDA / cuDNN.
GLOBAL_WARMUP="${GLOBAL_WARMUP:-20}"
# When 1, force the global warmup workload to include each type at least
# once (useful when some types have very low p_i and would otherwise be
# missed by random sampling).
GUARANTEE_EACH_TYPE_WARMUP="${GUARANTEE_EACH_TYPE_WARMUP:-0}"
GUARANTEE_EACH_TYPE_WARMUP_FLAG=""
if [ "$GUARANTEE_EACH_TYPE_WARMUP" = "1" ]; then
    GUARANTEE_EACH_TYPE_WARMUP_FLAG="--guarantee-each-type-warmup"
fi
SEED="${SEED:-42}"
# CACHE_BUDGET is informational only: solve_lambda.py uses it to print an
# offline preview of lambda*, but at runtime DistributionAwareCacheManager
# re-solves lambda* using vLLM's actual encoder cache size. Leave empty
# to let solve_lambda.py pick a placeholder (M // 2).
CACHE_BUDGET="${CACHE_BUDGET:-}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-2}"
GPU_P="${GPU_P:-2}"
GPU_D="${GPU_D:-3}"

# Encoder worker needs enough KV cache to fit visual tokens from test images.
# With 0.01 utilization the KV cache is too small for most image sizes.
# Since E and P share a GPU, keep it modest (E=0.10, P=0.60).
GPU_MEM_E="${GPU_MEM_E:-0.10}"
GPU_MEM_P="${GPU_MEM_P:-0.60}"
GPU_MEM_D="${GPU_MEM_D:-0.70}"

# Encoder cache size knob. vLLM derives encoder_cache_size from
# --max-num-batched-tokens (see vllm/config/scheduler.py:228), so this
# value also caps how many encoder embedding tokens fit in the cache.
# Lower values create cache pressure (good for seeing policy
# differences); higher values let everything fit (all policies tie).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-114688}"

# Use /dev/shm (tmpfs in memory) by default to avoid disk IO jitter on
# every encoder->prefill transfer. /dev/shm is a built-in tmpfs on almost
# every Linux system - no mount / sudo required.
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/dev/shm/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

# Working directories
WORK_DIR="${WORK_DIR:-/tmp/encoder_cache_benchmark}"
IMAGE_DIR="${WORK_DIR}/images"
LOG_PATH="${WORK_DIR}/logs"

# Distribution: JSON mapping type_id -> p_i
# Either pass DISTRIBUTION as a JSON string directly, OR pick a preset
# via DISTRIBUTION_PRESET (uniform, zipf, zipf-light, zipf-heavy/skewed,
# extreme, pareto-80-20, bimodal). DISTRIBUTION takes precedence; the
# preset is only used when DISTRIBUTION is empty. Default behaviour
# (both empty) falls back to a built-in zipf below.
DISTRIBUTION="${DISTRIBUTION:-}"
DISTRIBUTION_PRESET="${DISTRIBUTION_PRESET:-}"

# How to ship media to the server. "file" (default) uses file:// URLs
# so the server reads the media directly from disk via
# --allowed-local-media-path; request bodies stay tiny so TTFT isn't
# dominated by base64 upload + JSON parse. "base64" inlines the file
# (slower; needed when client and server don't share a filesystem).
MEDIA_MODE="${MEDIA_MODE:-file}"

# Approximate target text-prompt length in tokens (~4 chars/token).
# 0 = keep the original short "Describe this video briefly." prompt.
# Use this to study how prefill cost scales with prompt length and to
# shift the relative weight of encoder vs prefill in TTFT.
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"

export UCX_TLS=all
export UCX_NET_DEVICES=all

# GPU frequency locking: set to a fixed clock (MHz) to reduce variance.
# Set to "" to skip locking. Use `nvidia-smi -q -d SUPPORTED_CLOCKS` to
# find valid values for your GPU.
GPU_LOCK_GC="${GPU_LOCK_GC:-}"    # graphics clock, e.g. "1410"
GPU_LOCK_MC="${GPU_LOCK_MC:-}"    # memory clock, e.g. "1593"

###############################################################################
# Setup
###############################################################################
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORK_DIR" "$IMAGE_DIR" "$LOG_PATH"

declare -a PIDS=()

START_TIME=$(date +"%Y%m%d_%H%M%S")

# Collect unique GPU IDs used by this benchmark
declare -a USED_GPUS=()
for g in "$GPU_E" "$GPU_P" "$GPU_D"; do
    local_dup=false
    for existing in "${USED_GPUS[@]+"${USED_GPUS[@]}"}"; do
        if [ "$existing" = "$g" ]; then
            local_dup=true
            break
        fi
    done
    if [ "$local_dup" = false ]; then
        USED_GPUS+=("$g")
    fi
done

lock_gpu_clocks() {
    if [ -z "$GPU_LOCK_GC" ] && [ -z "$GPU_LOCK_MC" ]; then
        return
    fi
    echo "Locking GPU clocks for GPUs: ${USED_GPUS[*]}"
    for gpu_id in "${USED_GPUS[@]}"; do
        nvidia-smi -pm 1 -i "$gpu_id"
        if [ -n "$GPU_LOCK_GC" ]; then
            nvidia-smi -lgc "$GPU_LOCK_GC","$GPU_LOCK_GC" -i "$gpu_id"
            echo "  GPU $gpu_id: graphics clock locked to ${GPU_LOCK_GC} MHz"
        fi
        if [ -n "$GPU_LOCK_MC" ]; then
            nvidia-smi -lmc "$GPU_LOCK_MC","$GPU_LOCK_MC" -i "$gpu_id"
            echo "  GPU $gpu_id: memory clock locked to ${GPU_LOCK_MC} MHz"
        fi
    done
}

unlock_gpu_clocks() {
    if [ -z "$GPU_LOCK_GC" ] && [ -z "$GPU_LOCK_MC" ]; then
        return
    fi
    echo "Unlocking GPU clocks for GPUs: ${USED_GPUS[*]}"
    for gpu_id in "${USED_GPUS[@]}"; do
        if [ -n "$GPU_LOCK_GC" ]; then
            nvidia-smi -rgc -i "$gpu_id" 2>/dev/null || true
        fi
        if [ -n "$GPU_LOCK_MC" ]; then
            nvidia-smi -rmc -i "$gpu_id" 2>/dev/null || true
        fi
    done
}

wait_for_server() {
    local port=$1
    echo "Waiting for server on port $port..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null 2>&1; do
            sleep 2
        done" && echo "Server on port $port is ready" && return 0 \
        || { echo "Timeout waiting for server on port $port"; return 1; }
}

cleanup_servers() {
    echo "Stopping all servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    PIDS=()
    echo "All servers stopped."
}

trap 'cleanup_servers; unlock_gpu_clocks; exit 1' INT TERM

###############################################################################
# Helper: Start full 1E1P1D stack
###############################################################################
start_1e1p1d() {
    local policy="${1:-fifo}"
    local label="${2:-default}"
    local config_path="${3:-}"

    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"

    # When video types are present, allow at most 1 video per prompt and
    # at most 1 image per prompt (each request carries one media item).
    local mm_limit_arg=""
    if [ "$NUM_VIDEOS" -gt 0 ]; then
        mm_limit_arg='--limit-mm-per-prompt={"image":1,"video":1}'
    fi

    # Encoder worker (dev mode enables /reset_encoder_cache endpoint).
    # VLLM_ENCODER_CACHE_TRACE=1 makes the encoder cache manager log
    # one INFO line per check_and_update_cache call so plot_real_hitrate
    # can reconstruct the per-request hit rate. Set EC_TRACE=0 to skip.
    CUDA_VISIBLE_DEVICES="$GPU_E" \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_ENCODER_CACHE_POLICY="$policy" \
    VLLM_ENCODER_CACHE_CONFIG_PATH="$config_path" \
    VLLM_ENCODER_CACHE_TRACE="${EC_TRACE:-1}" \
    VLLM_REQUEST_TIMING_TRACE="${TIMING_TRACE:-1}" \
    vllm serve "$MODEL" \
        --gpu-memory-utilization "$GPU_MEM_E" \
        --port "$ENCODE_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
        $mm_limit_arg \
        --ec-transfer-config '{
            "ec_connector": "ECExampleConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        >"${LOG_PATH}/encoder_${label}_${START_TIME}.log" 2>&1 &
    PIDS+=($!)

    # Prefill worker (also reads encoder cache policy since it has its own
    # scheduler / EncoderCacheManager that decides when to load from EC)
    CUDA_VISIBLE_DEVICES="$GPU_P" \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
    VLLM_ENCODER_CACHE_POLICY="$policy" \
    VLLM_ENCODER_CACHE_CONFIG_PATH="$config_path" \
    VLLM_REQUEST_TIMING_TRACE="${TIMING_TRACE:-1}" \
    VLLM_EC_DELETE_AFTER_LOAD="${EC_DELETE_AFTER_LOAD:-1}" \
    vllm serve "$MODEL" \
        --gpu-memory-utilization "$GPU_MEM_P" \
        --port "$PREFILL_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
        $mm_limit_arg \
        --ec-transfer-config '{
            "ec_connector": "ECExampleConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer"
        }' \
        >"${LOG_PATH}/prefill_${label}_${START_TIME}.log" 2>&1 &
    PIDS+=($!)

    # Decode worker
    CUDA_VISIBLE_DEVICES="$GPU_D" \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
    VLLM_REQUEST_TIMING_TRACE="${TIMING_TRACE:-1}" \
    vllm serve "$MODEL" \
        --gpu-memory-utilization "$GPU_MEM_D" \
        --port "$DECODE_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
        $mm_limit_arg \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer"
        }' \
        >"${LOG_PATH}/decode_${label}_${START_TIME}.log" 2>&1 &
    PIDS+=($!)

    # Wait for all workers
    wait_for_server "$ENCODE_PORT"
    wait_for_server "$PREFILL_PORT"
    wait_for_server "$DECODE_PORT"

    # Start proxy
    python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
        --host "0.0.0.0" \
        --port "$PROXY_PORT" \
        --encode-servers-urls "http://localhost:$ENCODE_PORT" \
        --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
        --decode-servers-urls "http://localhost:$DECODE_PORT" \
        >"${LOG_PATH}/proxy_${label}_${START_TIME}.log" 2>&1 &
    PIDS+=($!)

    wait_for_server "$PROXY_PORT"
    echo "All 1E1P1D services are up! ($label)"
}

###############################################################################
# Step 1 + 2: Generate test images and distribution (or reuse existing)
###############################################################################
if [ "${SKIP_GENERATION:-0}" = "1" ]; then
    echo "============================================================"
    echo "SKIP_GENERATION=1 — reusing existing manifest and distribution"
    echo "============================================================"
    MANIFEST_PATH="${MANIFEST_PATH:-$WORK_DIR/manifest.json}"
    if [ ! -f "$MANIFEST_PATH" ]; then
        echo "ERROR: SKIP_GENERATION=1 but $MANIFEST_PATH does not exist."
        echo "Run preprocess_real_dataset.py first, or unset SKIP_GENERATION."
        exit 1
    fi
    if [ -z "$DISTRIBUTION" ]; then
        echo "ERROR: SKIP_GENERATION=1 but DISTRIBUTION env var is empty."
        echo "Pass DISTRIBUTION=\"\$(cat .../distribution_real.json)\""
        exit 1
    fi
    echo "Using manifest: $MANIFEST_PATH"
else
    echo "============================================================"
    echo "Step 1: Generating $NUM_TYPES test image types"
    echo "============================================================"

    python "$SCRIPT_DIR/generate_test_images.py" \
        --num-types "$NUM_TYPES" \
        --output-dir "$IMAGE_DIR" \
        --manifest-path "$WORK_DIR/manifest.json"

    MANIFEST_PATH="$WORK_DIR/manifest.json"

    ###########################################################################
    # Step 1.5: Generate test videos (if NUM_VIDEOS > 0) and append to manifest
    ###########################################################################
    if [ "$NUM_VIDEOS" -gt 0 ]; then
        duration_args=(--duration-s "$VIDEO_DURATION_S")
        if [ -n "$VIDEO_DURATION_MIN_S" ] && \
           [ -n "$VIDEO_DURATION_MAX_S" ]; then
            duration_args=(
                --duration-min-s "$VIDEO_DURATION_MIN_S"
                --duration-max-s "$VIDEO_DURATION_MAX_S"
            )
            echo "Generating $NUM_VIDEOS test video types "
            echo "(duration in [${VIDEO_DURATION_MIN_S}s, "
            echo "${VIDEO_DURATION_MAX_S}s], fps=$VIDEO_FPS)..."
        else
            echo "Generating $NUM_VIDEOS test video types "
            echo "(duration=${VIDEO_DURATION_S}s, fps=$VIDEO_FPS)..."
        fi
        python "$SCRIPT_DIR/generate_test_videos.py" \
            --num-videos "$NUM_VIDEOS" \
            "${duration_args[@]}" \
            --fps "$VIDEO_FPS" \
            --seed "$SEED" \
            --output-dir "$IMAGE_DIR" \
            --manifest-path "$MANIFEST_PATH" \
            --start-type-id "$NUM_TYPES"
    fi

    ###########################################################################
    # Step 2: Generate distribution if not provided
    ###########################################################################
    if [ -z "$DISTRIBUTION" ]; then
        K_TOTAL=$(( NUM_TYPES + NUM_VIDEOS ))
        if [ -n "$DISTRIBUTION_PRESET" ]; then
            echo "Generating distribution from preset '$DISTRIBUTION_PRESET' "
            echo "for $K_TOTAL types..."
            DISTRIBUTION=$(python "$SCRIPT_DIR/gen_distribution.py" \
                --preset "$DISTRIBUTION_PRESET" \
                --num-types "$K_TOTAL")
        else
            echo "Generating default Zipf-like distribution for "
            echo "$K_TOTAL types (no DISTRIBUTION / DISTRIBUTION_PRESET set)..."
            DISTRIBUTION=$(python -c "
import json, random
K = $K_TOTAL
# Zipf weights: 1/1, 1/2, ..., 1/K (covers both image and video types)
raw = [1.0/(i+1) for i in range(K)]
# Shuffle so the highest probability is NOT always on the smallest
# resolution (type_0). This way p_i and m_i are decorrelated.
random.seed($SEED)
random.shuffle(raw)
total = sum(raw)
dist = {f'type_{i}': round(raw[i]/total, 4) for i in range(K)}
# Fix rounding
remainder = round(1.0 - sum(dist.values()), 4)
first_key = f'type_0'
dist[first_key] = round(dist[first_key] + remainder, 4)
print(json.dumps(dist))
")
        fi
    fi
fi

echo "Distribution: $DISTRIBUTION"
echo "Media mode: $MEDIA_MODE"
echo "Prompt tokens: $PROMPT_TOKENS"

###############################################################################
# Step 3: Profile encoder computation time (c_i)
###############################################################################
echo "============================================================"
echo "Step 3: Profiling encoder computation times"
echo "============================================================"

# Directly load the vision encoder and time it. No server needed.
CUDA_VISIBLE_DEVICES="$GPU_E" python "$SCRIPT_DIR/profile_encoder.py" \
    --manifest-path "$MANIFEST_PATH" \
    --model "$MODEL" \
    --device cuda:0 \
    --num-warmup 3 \
    --num-iterations 10 \
    --output-path "$WORK_DIR/profile.json"

PROFILE_PATH="$WORK_DIR/profile.json"

###############################################################################
# Step 4: Solve for lambda*
###############################################################################
echo "============================================================"
echo "Step 4: Solving for optimal lambda*"
echo "============================================================"

# Default cache budget = MAX_NUM_BATCHED_TOKENS, which is exactly what
# vLLM uses for encoder_cache_size (vllm/config/scheduler.py:228). Using
# the same value here keeps solve_lambda's offline analysis (printed
# reservations, lambda*) consistent with what the runtime
# DistributionAwareCacheManager will re-derive. Without this, the
# offline analysis used a placeholder of M//2, which made
# lambda_config.json look unrelated to runtime behaviour — exactly the
# "diagnosis 4" symptom we hit in the QPS sweep.
CACHE_BUDGET="${CACHE_BUDGET:-$MAX_NUM_BATCHED_TOKENS}"
echo "Cache budget for solver: $CACHE_BUDGET (matches "
echo "  vLLM's encoder_cache_size = max-num-batched-tokens)"

python "$SCRIPT_DIR/solve_lambda.py" \
    --profile-path "$PROFILE_PATH" \
    --distribution "$DISTRIBUTION" \
    --cache-budget "$CACHE_BUDGET" \
    --output-path "$WORK_DIR/lambda_config.json"

LAMBDA_CONFIG_PATH="$WORK_DIR/lambda_config.json"

###############################################################################
# Step 4.5: Build distribution config for the dist-aware cache manager
###############################################################################
DIST_CONFIG_PATH="$WORK_DIR/dist_cache_config.json"

python -c "
import json

with open('$LAMBDA_CONFIG_PATH') as f:
    lam_config = json.load(f)

with open('$MANIFEST_PATH') as f:
    manifest = json.load(f)

config = {'types': {}, 'hash_to_type': {}}

for t in lam_config['types']:
    tid = t['type_id']
    config['types'][tid] = {
        'p_i': t['p_i'],
        'm_i': t['m_i'],
        'c_i': t['c_i'],
    }

# run_benchmark.py sends each MM part with uuid=<type_id>. With no
# hf_processor_mm_kwargs set, vLLM uses that uuid verbatim as the
# mm_hash, so the runtime lookup key is the type_id itself.
# (Without this, the mapping was keyed by file path and never matched
# the runtime mm_hash, silently degrading distribution-aware to FIFO.)
for tid in manifest.keys():
    config['hash_to_type'][tid] = tid

with open('$DIST_CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=2)

print(f'Distribution cache config written to $DIST_CONFIG_PATH')
"

###############################################################################
# Helper: run a single benchmark trial under a given policy
###############################################################################
run_trial() {
    local policy="$1"
    local label="$2"
    local config_path="${3:-}"

    echo "============================================================"
    echo "Benchmarking policy=$policy (label=$label)"
    echo "============================================================"

    start_1e1p1d "$policy" "$label" "$config_path"

    python "$SCRIPT_DIR/run_benchmark.py" \
        --manifest-path "$MANIFEST_PATH" \
        --distribution "$DISTRIBUTION" \
        --num-requests "$NUM_REQUESTS" \
        --server-url "http://localhost:$PROXY_PORT" \
        --model "$MODEL" \
        --mode "$BENCH_MODE" \
        --qps "$QPS" \
        --concurrency "$CONCURRENCY" \
        --warmup-requests "$WARMUP_REQUESTS" \
        --num-rounds "$NUM_ROUNDS" \
        --trim "$TRIM" \
        --global-warmup "$GLOBAL_WARMUP" \
        $GUARANTEE_EACH_TYPE_WARMUP_FLAG \
        --encoder-url "http://localhost:$ENCODE_PORT" \
        --encoder-log-path "${LOG_PATH}/encoder_${label}_${START_TIME}.log" \
        --seed "$SEED" \
        --label "$label" \
        --media-mode "$MEDIA_MODE" \
        --prompt-tokens "$PROMPT_TOKENS" \
        --output-path "$WORK_DIR/results_${label}.json"

    cleanup_servers
    sleep 5
}

###############################################################################
# Step 5: Lock GPU clocks, pre-warm page cache, benchmark
###############################################################################
# Pre-read all images into the page cache so the first round doesn't pay the
# cold-read penalty (improves consistency across rounds).
echo "Pre-loading test images into page cache..."
cat "$IMAGE_DIR"/*.jpg > /dev/null 2>&1 || true

lock_gpu_clocks

run_trial "none" "none"

###############################################################################
# Step 6: Benchmark FIFO policy (with EC connector cleanup — apples-to-apples)
###############################################################################
run_trial "fifo" "fifo"

###############################################################################
# Step 6a: Benchmark FIFO + persistent EC connector files
#
# This is the "today's vLLM ships like this" baseline: in-memory FIFO
# eviction in EncoderCacheManager, but EC connector files in /dev/shm
# persist forever. The persistent file cache silently rescues every
# in-memory miss, so this configuration tends to look much faster than
# real FIFO would in a longer-running deployment where /dev/shm
# eventually fills up (or where prefill and encoder do not share a
# filesystem). Useful as a stress baseline for our delete-after-load
# fix.
#
# Skip with SKIP_FIFO_PERSISTENT=1.
###############################################################################
if [ "${SKIP_FIFO_PERSISTENT:-0}" != "1" ]; then
    EC_DELETE_AFTER_LOAD=0 run_trial "fifo" "fifo_persistent"
fi

###############################################################################
# Step 6.5: Benchmark Distribution-Aware policy
###############################################################################
run_trial "distribution_aware" "dist_aware" "$DIST_CONFIG_PATH"

unlock_gpu_clocks

###############################################################################
# Step 7: Compare results (None / FIFO / Dist-Aware)
###############################################################################
echo "============================================================"
echo "Step 7: Comparison Results"
echo "============================================================"

python -c "
import json
import os

with open('$WORK_DIR/results_none.json') as f:
    none_data = json.load(f)
with open('$WORK_DIR/results_fifo.json') as f:
    fifo_data = json.load(f)
with open('$WORK_DIR/results_dist_aware.json') as f:
    dist_data = json.load(f)

# fifo_persistent is optional (may have been skipped via SKIP_FIFO_PERSISTENT=1)
fifo_pers_path = '$WORK_DIR/results_fifo_persistent.json'
fifo_pers_data = None
if os.path.exists(fifo_pers_path):
    with open(fifo_pers_path) as f:
        fifo_pers_data = json.load(f)

# Use aggregated means when available (multi-round), fall back to metrics
def get_val(data, key):
    agg = data.get('aggregated', {})
    if key in agg:
        return agg[key]['mean']
    return data.get('metrics', {}).get(key, 0)

def get_std(data, key):
    agg = data.get('aggregated', {})
    if key in agg:
        return agg[key].get('std', 0)
    return 0

num_rounds = none_data.get('config', {}).get('num_rounds', 1)

def fmt(data, key):
    v = get_val(data, key)
    s = get_std(data, key)
    if s > 0:
        return f'{v:.2f}+/-{s:.2f}'
    return f'{v:.2f}' if isinstance(v, (int, float)) else str(v)

def imp(base_data, new_data, key, higher_is_better=False):
    base = get_val(base_data, key)
    new = get_val(new_data, key)
    if base in (0, None) or new in (0, None):
        return 'N/A'
    delta = (new - base) / base * 100
    if higher_is_better:
        return f'{delta:+.1f}%'
    return f'{-delta:+.1f}%'

print()
print('=' * 130)
print(f'Encoder Cache Policy Comparison (vs no-cache baseline, {num_rounds} round(s))')
print('=' * 130)
header_cols = [('Metric', 22), ('None', 16), ('FIFO+EC-clean', 16),
               ('FIFO+EC-persist', 18), ('Dist-Aware', 16),
               ('FIFO vs None', 14), ('Dist vs FIFO+pers', 18)]
print(''.join(f'{name:<{w}}' for name, w in header_cols))
print('-' * 130)

# (metric_key, label, higher_is_better)
metrics = [
    ('ttft_mean_ms',     'TTFT Mean (ms)',     False),
    ('ttft_median_ms',   'TTFT Median (ms)',   False),
    ('ttft_p95_ms',      'TTFT P95 (ms)',      False),
    ('ttft_p99_ms',      'TTFT P99 (ms)',      False),
    ('latency_mean_ms',  'Latency Mean (ms)',  False),
    ('throughput_rps',   'Throughput (req/s)', True),
    ('cache_hit_rate',   'Cache Hit Rate',     True),
]

for key, label, higher in metrics:
    fp_str = fmt(fifo_pers_data, key) if fifo_pers_data else 'skipped'
    # Headline: how much does our algorithm (dist-aware + EC clean)
    # improve over the today's-vLLM baseline (FIFO + persistent EC)?
    if fifo_pers_data is not None:
        dist_vs_pers = imp(fifo_pers_data, dist_data, key, higher)
    else:
        dist_vs_pers = 'N/A'
    print(f'{label:<22} {fmt(none_data, key):<16} '
          f'{fmt(fifo_data, key):<16} {fp_str:<18} '
          f'{fmt(dist_data, key):<16}'
          f'{imp(none_data, fifo_data, key, higher):<14} '
          f'{dist_vs_pers:<18}')

print()
ns = fmt(none_data, 'successful')
fs = fmt(fifo_data, 'successful')
fps = fmt(fifo_pers_data, 'successful') if fifo_pers_data else 'skipped'
ds = fmt(dist_data, 'successful')
print(f\"{'Successful':<22} {ns:<16} {fs:<16} {fps:<18} {ds:<16}\")
nf = fmt(none_data, 'failed')
ff = fmt(fifo_data, 'failed')
fpf = fmt(fifo_pers_data, 'failed') if fifo_pers_data else 'skipped'
df = fmt(dist_data, 'failed')
print(f\"{'Failed':<22} {nf:<16} {ff:<16} {fpf:<18} {df:<16}\")

# Per-type breakdown uses last-round metrics (not aggregated)
none_r = none_data['metrics']
fifo_r = fifo_data['metrics']
dist_r = dist_data['metrics']

print()
print('Per-Type TTFT Median (ms) [last round]:')
print(f\"{'Type':<10} {'None':<14} {'FIFO':<14} {'Dist-Aware':<14}\"
      f\"{'FIFO vs None':<14} {'Dist vs None':<14}\")
print('-' * 80)

def imp_val(base, new):
    if base in (0, None) or new in (0, None):
        return 'N/A'
    return f'{-(new - base) / base * 100:+.1f}%'

all_types = (
    set(none_r.get('per_type', {}).keys())
    | set(fifo_r.get('per_type', {}).keys())
    | set(dist_r.get('per_type', {}).keys())
)
for tid in sorted(all_types):
    nt = none_r.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    ft = fifo_r.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    dt = dist_r.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    print(f'{tid:<10} {nt:<14.2f} {ft:<14.2f} {dt:<14.2f}'
          f'{imp_val(nt, ft):<14} {imp_val(nt, dt):<14}')

print()
if num_rounds > 1:
    print(f'Values shown as mean+/-std across {num_rounds} rounds.')
print('Improvement: TTFT/latency reduction or throughput gain (positive = better).')
print()
print(f'Full results saved in: $WORK_DIR/')
"

echo ""
echo "Done! Results are in $WORK_DIR/"
echo "  - results_none.json             (no-cache baseline)"
echo "  - results_fifo.json             (FIFO/LRU + EC delete-after-load)"
echo "  - results_fifo_persistent.json  (FIFO/LRU + persistent EC files;"
echo "                                    skipped if SKIP_FIFO_PERSISTENT=1)"
echo "  - results_dist_aware.json (distribution-aware)"
echo "  - lambda_config.json"
echo "  - profile.json"
