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
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct NUM_TYPES=5 NUM_REQUESTS=200 \
#     bash run_cache_comparison.sh
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
NUM_TYPES="${NUM_TYPES:-5}"
NUM_REQUESTS="${NUM_REQUESTS:-200}"
QPS="${QPS:-2}"
SEED="${SEED:-42}"
CACHE_BUDGET="${CACHE_BUDGET:-2000}"

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

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

# Working directories
WORK_DIR="${WORK_DIR:-/tmp/encoder_cache_benchmark}"
IMAGE_DIR="${WORK_DIR}/images"
LOG_PATH="${WORK_DIR}/logs"

# Distribution: JSON mapping type_id -> p_i
# Default: generate a skewed Zipf-like distribution
DISTRIBUTION="${DISTRIBUTION:-}"

export UCX_TLS=all
export UCX_NET_DEVICES=all

###############################################################################
# Setup
###############################################################################
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORK_DIR" "$IMAGE_DIR" "$LOG_PATH"

declare -a PIDS=()

START_TIME=$(date +"%Y%m%d_%H%M%S")

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

trap 'cleanup_servers; exit 1' INT TERM

###############################################################################
# Helper: Start full 1E1P1D stack
###############################################################################
start_1e1p1d() {
    local policy="${1:-fifo}"
    local label="${2:-default}"
    local config_path="${3:-}"

    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"

    # Encoder worker
    CUDA_VISIBLE_DEVICES="$GPU_E" \
    VLLM_ENCODER_CACHE_POLICY="$policy" \
    VLLM_ENCODER_CACHE_CONFIG_PATH="$config_path" \
    vllm serve "$MODEL" \
        --gpu-memory-utilization "$GPU_MEM_E" \
        --port "$ENCODE_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
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
    vllm serve "$MODEL" \
        --gpu-memory-utilization "$GPU_MEM_P" \
        --port "$PREFILL_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
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
    vllm serve "$MODEL" \
        --gpu-memory-utilization "$GPU_MEM_D" \
        --port "$DECODE_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
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
# Step 1: Generate test images
###############################################################################
echo "============================================================"
echo "Step 1: Generating $NUM_TYPES test image types"
echo "============================================================"

python "$SCRIPT_DIR/generate_test_images.py" \
    --num-types "$NUM_TYPES" \
    --output-dir "$IMAGE_DIR" \
    --manifest-path "$WORK_DIR/manifest.json"

MANIFEST_PATH="$WORK_DIR/manifest.json"

###############################################################################
# Step 2: Generate distribution if not provided
###############################################################################
if [ -z "$DISTRIBUTION" ]; then
    echo "Generating Zipf-like distribution for $NUM_TYPES types..."
    DISTRIBUTION=$(python -c "
import json, math
K = $NUM_TYPES
# Zipf distribution: p_i proportional to 1/i
raw = [1.0/(i+1) for i in range(K)]
total = sum(raw)
dist = {f'type_{i}': round(raw[i]/total, 4) for i in range(K)}
# Fix rounding
remainder = round(1.0 - sum(dist.values()), 4)
dist['type_0'] = round(dist['type_0'] + remainder, 4)
print(json.dumps(dist))
")
fi

echo "Distribution: $DISTRIBUTION"

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

# Map image paths to type_ids. Note: at runtime the cache uses real mm_hashes,
# this mapping is used by tests / debugging.
for tid, info in manifest.items():
    config['hash_to_type'][info['path']] = tid

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
        --qps "$QPS" \
        --seed "$SEED" \
        --label "$label" \
        --output-path "$WORK_DIR/results_${label}.json"

    cleanup_servers
    sleep 5
}

###############################################################################
# Step 5: Benchmark no-cache baseline
###############################################################################
run_trial "none" "none"

###############################################################################
# Step 6: Benchmark FIFO policy
###############################################################################
run_trial "fifo" "fifo"

###############################################################################
# Step 6.5: Benchmark Distribution-Aware policy
###############################################################################
run_trial "distribution_aware" "dist_aware" "$DIST_CONFIG_PATH"

###############################################################################
# Step 7: Compare results (None / FIFO / Dist-Aware)
###############################################################################
echo "============================================================"
echo "Step 7: Comparison Results"
echo "============================================================"

python -c "
import json

with open('$WORK_DIR/results_none.json') as f:
    none_r = json.load(f)['metrics']
with open('$WORK_DIR/results_fifo.json') as f:
    fifo_r = json.load(f)['metrics']
with open('$WORK_DIR/results_dist_aware.json') as f:
    dist_r = json.load(f)['metrics']

def fmt(v):
    return f'{v:.2f}' if isinstance(v, (int, float)) else str(v)

def imp(base, new, higher_is_better=False):
    if base in (0, None) or new in (0, None):
        return 'N/A'
    delta = (new - base) / base * 100
    if higher_is_better:
        return f'{delta:+.1f}%'
    return f'{-delta:+.1f}%'  # show reduction as positive

print()
print('=' * 90)
print('Encoder Cache Policy Comparison (vs no-cache baseline)')
print('=' * 90)
print(f\"{'Metric':<22} {'None':<14} {'FIFO':<14} {'Dist-Aware':<14}\"
      f\"{'FIFO vs None':<14} {'Dist vs None':<14}\")
print('-' * 90)

# (metric_key, label, higher_is_better)
metrics = [
    ('ttft_mean_ms',     'TTFT Mean (ms)',     False),
    ('ttft_median_ms',   'TTFT Median (ms)',   False),
    ('ttft_p95_ms',      'TTFT P95 (ms)',      False),
    ('ttft_p99_ms',      'TTFT P99 (ms)',      False),
    ('latency_mean_ms',  'Latency Mean (ms)',  False),
    ('throughput_rps',   'Throughput (req/s)', True),
    ('wall_clock_s',     'Wall Clock (s)',     False),
]

for key, label, higher in metrics:
    nv = none_r.get(key, 0)
    fv = fifo_r.get(key, 0)
    dv = dist_r.get(key, 0)
    print(f'{label:<22} {fmt(nv):<14} {fmt(fv):<14} {fmt(dv):<14}'
          f'{imp(nv, fv, higher):<14} {imp(nv, dv, higher):<14}')

print()
print(f\"{'Successful':<22} {none_r['successful']:<14} {fifo_r['successful']:<14} \"
      f\"{dist_r['successful']:<14}\")
print(f\"{'Failed':<22} {none_r['failed']:<14} {fifo_r['failed']:<14} \"
      f\"{dist_r['failed']:<14}\")
print()
print('Per-Type TTFT Median (ms):')
print(f\"{'Type':<10} {'None':<14} {'FIFO':<14} {'Dist-Aware':<14}\"
      f\"{'FIFO vs None':<14} {'Dist vs None':<14}\")
print('-' * 80)

all_types = (
    set(none_r.get('per_type', {}).keys())
    | set(fifo_r.get('per_type', {}).keys())
    | set(dist_r.get('per_type', {}).keys())
)
for tid in sorted(all_types):
    nt = none_r.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    ft = fifo_r.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    dt = dist_r.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    print(f'{tid:<10} {fmt(nt):<14} {fmt(ft):<14} {fmt(dt):<14}'
          f'{imp(nt, ft):<14} {imp(nt, dt):<14}')

print()
print('Note: Improvement columns show TTFT/latency reduction (positive is')
print('better) or throughput gain (positive is better).')
print()
print(f'Full results saved in: $WORK_DIR/')
"

echo ""
echo "Done! Results are in $WORK_DIR/"
echo "  - results_none.json       (no-cache baseline)"
echo "  - results_fifo.json       (FIFO/LRU)"
echo "  - results_dist_aware.json (distribution-aware)"
echo "  - lambda_config.json"
echo "  - profile.json"
