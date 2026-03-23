#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end comparison of FIFO vs Distribution-Aware encoder cache policies.
#
# This script:
#   1. Generates K synthetic test images
#   2. Deploys 1E1P1D (encoder + prefill + decode) via disagg setup
#   3. Profiles encoder computation time for each image type
#   4. Solves for optimal lambda*
#   5. Runs benchmark with FIFO cache policy
#   6. Restarts with distribution-aware policy and runs benchmark again
#   7. Prints comparison of TTFT and hit rate
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
    echo "Waiting for server on port $port (health check)..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -sf http://localhost:$port/health > /dev/null 2>&1; do
            sleep 3
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
    local extra_env="${1:-}"
    local label="${2:-default}"

    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"

    # Encoder worker
    CUDA_VISIBLE_DEVICES="$GPU_E" \
    $extra_env \
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

    # Prefill worker
    CUDA_VISIBLE_DEVICES="$GPU_P" \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
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
# Step 3: Start full 1E1P1D stack and profile encoder
###############################################################################
echo "============================================================"
echo "Step 3: Starting full 1E1P1D stack for profiling"
echo "============================================================"

# Profiling must go through the full pipeline (encoder -> prefill -> decode)
# because the encoder worker alone (ec_producer, gpu-memory-utilization 0.01)
# cannot serve complete chat completion requests.
start_1e1p1d "" "profile"

echo "Profiling encoder computation times..."
python "$SCRIPT_DIR/profile_encoder.py" \
    --manifest-path "$MANIFEST_PATH" \
    --server-url "http://localhost:$PROXY_PORT" \
    --model "$MODEL" \
    --num-warmup 2 \
    --num-iterations 5 \
    --output-path "$WORK_DIR/profile.json"

PROFILE_PATH="$WORK_DIR/profile.json"

# Stop profiling stack
cleanup_servers
sleep 5

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
# Step 5: Benchmark FIFO policy
###############################################################################
echo "============================================================"
echo "Step 5: Benchmarking FIFO cache policy"
echo "============================================================"

start_1e1p1d "" "fifo"

python "$SCRIPT_DIR/run_benchmark.py" \
    --manifest-path "$MANIFEST_PATH" \
    --distribution "$DISTRIBUTION" \
    --num-requests "$NUM_REQUESTS" \
    --server-url "http://localhost:$PROXY_PORT" \
    --model "$MODEL" \
    --qps "$QPS" \
    --seed "$SEED" \
    --label "fifo" \
    --output-path "$WORK_DIR/results_fifo.json"

cleanup_servers
sleep 5

###############################################################################
# Step 6: Benchmark Distribution-Aware policy
###############################################################################
echo "============================================================"
echo "Step 6: Benchmarking Distribution-Aware cache policy"
echo "============================================================"

# Pass distribution config via environment variable
export ENCODER_CACHE_DISTRIBUTION_CONFIG="$WORK_DIR/dist_cache_config.json"

# Generate the distribution config file for the cache manager
python -c "
import json

# Load lambda config
with open('$LAMBDA_CONFIG_PATH') as f:
    lam_config = json.load(f)

# Load manifest to get hash mappings (for testing, we use image paths as hashes)
with open('$MANIFEST_PATH') as f:
    manifest = json.load(f)

# Build cache manager config
config = {
    'types': {},
    'hash_to_type': {}
}

for t in lam_config['types']:
    tid = t['type_id']
    config['types'][tid] = {
        'p_i': t['p_i'],
        'm_i': t['m_i'],
        'c_i': t['c_i'],
    }

# Map image paths to type_ids (hash_to_type will be populated at runtime
# when actual mm_hashes are known)
for tid, info in manifest.items():
    config['hash_to_type'][info['path']] = tid

with open('$ENCODER_CACHE_DISTRIBUTION_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)

print(f'Distribution cache config written to $ENCODER_CACHE_DISTRIBUTION_CONFIG')
"

start_1e1p1d "" "dist_aware"

python "$SCRIPT_DIR/run_benchmark.py" \
    --manifest-path "$MANIFEST_PATH" \
    --distribution "$DISTRIBUTION" \
    --num-requests "$NUM_REQUESTS" \
    --server-url "http://localhost:$PROXY_PORT" \
    --model "$MODEL" \
    --qps "$QPS" \
    --seed "$SEED" \
    --label "distribution_aware" \
    --output-path "$WORK_DIR/results_dist_aware.json"

cleanup_servers

###############################################################################
# Step 7: Compare results
###############################################################################
echo "============================================================"
echo "Step 7: Comparison Results"
echo "============================================================"

python -c "
import json

with open('$WORK_DIR/results_fifo.json') as f:
    fifo = json.load(f)
with open('$WORK_DIR/results_dist_aware.json') as f:
    dist = json.load(f)

fm = fifo['metrics']
dm = dist['metrics']

print()
print('=' * 70)
print('FIFO vs Distribution-Aware Encoder Cache Comparison')
print('=' * 70)
print(f\"{'Metric':<25} {'FIFO':<15} {'Dist-Aware':<15} {'Improvement':<15}\")
print('-' * 70)

for metric, label in [
    ('ttft_mean_ms', 'TTFT Mean (ms)'),
    ('ttft_median_ms', 'TTFT Median (ms)'),
    ('ttft_p95_ms', 'TTFT P95 (ms)'),
    ('ttft_p99_ms', 'TTFT P99 (ms)'),
    ('latency_mean_ms', 'Latency Mean (ms)'),
]:
    fv = fm.get(metric, 0)
    dv = dm.get(metric, 0)
    if fv > 0:
        improvement = (fv - dv) / fv * 100
        print(f'{label:<25} {fv:<15.2f} {dv:<15.2f} {improvement:+.1f}%')
    else:
        print(f'{label:<25} {fv:<15.2f} {dv:<15.2f} N/A')

print()
print(f\"{'Successful':<25} {fm['successful']:<15} {dm['successful']:<15}\")
print(f\"{'Failed':<25} {fm['failed']:<15} {dm['failed']:<15}\")
print()
print('Per-Type TTFT Comparison (median ms):')
print(f\"{'Type':<12} {'FIFO':<15} {'Dist-Aware':<15} {'Improvement':<15}\")
print('-' * 57)

all_types = set(list(fm.get('per_type', {}).keys()) +
                list(dm.get('per_type', {}).keys()))
for tid in sorted(all_types):
    ft = fm.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    dt = dm.get('per_type', {}).get(tid, {}).get('ttft_median_ms', 0)
    if ft > 0:
        imp = (ft - dt) / ft * 100
        print(f'{tid:<12} {ft:<15.2f} {dt:<15.2f} {imp:+.1f}%')
    else:
        print(f'{tid:<12} {ft:<15.2f} {dt:<15.2f} N/A')

print()
print(f'Full results saved in: $WORK_DIR/')
"

echo ""
echo "Done! Results are in $WORK_DIR/"
echo "  - results_fifo.json"
echo "  - results_dist_aware.json"
echo "  - lambda_config.json"
echo "  - profile.json"
