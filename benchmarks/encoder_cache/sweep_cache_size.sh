#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Sweep encoder cache size and run the 3-policy comparison at each point.
#
# vLLM derives encoder_cache_size from --max-num-batched-tokens
# (see vllm/config/scheduler.py:228). We vary that value to find the
# pressure regime where Distribution-Aware separates from FIFO.
#
# Outputs one subdir per cache size under $SWEEP_DIR, each containing
# results_{none,fifo,dist_aware}.json from run_cache_comparison.sh.
# A manifest.csv summarises the sweep for plot_sweep.py.
#
# Usage (assumes you already preprocessed a dataset into $SOURCE_DIR):
#   SOURCE_DIR=/tmp/vmmu_data \
#   SWEEP_DIR=/tmp/cache_sweep \
#   DISTRIBUTION="$(cat /tmp/vmmu_data/distribution_videommmu.json)" \
#   NUM_VIDEOS=5 NUM_TYPES=0 \
#   bash sweep_cache_size.sh
#
#   # Custom sizes (in tokens, must be ≥ largest single mm item):
#   SWEEP_SIZES="4096 8192 16384 32768 65536" \
#   bash sweep_cache_size.sh
#
# Then plot:
#   python plot_sweep.py --sweep-dir /tmp/cache_sweep -o /tmp/sweep.png
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where the existing manifest/videos live (must already contain
# manifest.json + the media files referenced by it).
SOURCE_DIR="${SOURCE_DIR:-${IMAGE_DIR:-/tmp/vmmu_data}}"
SOURCE_MANIFEST="${MANIFEST_PATH:-$SOURCE_DIR/manifest.json}"

if [ ! -f "$SOURCE_MANIFEST" ]; then
    echo "ERROR: $SOURCE_MANIFEST not found." >&2
    echo "Run preprocess_videommmu.py (or another preprocessor) first," >&2
    echo "or set SOURCE_DIR=<dir containing manifest.json>." >&2
    exit 1
fi

SWEEP_DIR="${SWEEP_DIR:-/tmp/cache_sweep}"
SWEEP_SIZES="${SWEEP_SIZES:-8192 16384 32768 49152 65536 98304 131072}"

mkdir -p "$SWEEP_DIR"
manifest_csv="$SWEEP_DIR/manifest.csv"
echo "cache_size,subdir,timestamp,status" > "$manifest_csv"

echo "Source manifest:  $SOURCE_MANIFEST"
echo "Source media dir: $SOURCE_DIR"
echo "Sweep sizes:      $SWEEP_SIZES"
echo "Output dir:       $SWEEP_DIR"
echo

START_TS=$(date +%s)
for size in $SWEEP_SIZES; do
    sub="$SWEEP_DIR/cache_${size}"
    mkdir -p "$sub"
    echo
    echo "============================================================"
    echo "  Sweep: encoder_cache_size = $size tokens"
    echo "  Output: $sub"
    echo "============================================================"

    # Per-iteration WORK_DIR ($sub) catches per-size artifacts
    # (profile.json, lambda_config.json, dist_cache_config.json,
    # results_*.json, logs/). The MANIFEST_PATH and IMAGE_DIR stay
    # pinned to the source so we don't re-download videos.
    if WORK_DIR="$sub" \
       IMAGE_DIR="$SOURCE_DIR" \
       MANIFEST_PATH="$SOURCE_MANIFEST" \
       MAX_NUM_BATCHED_TOKENS="$size" \
       SKIP_GENERATION=1 \
       bash "$SCRIPT_DIR/run_cache_comparison.sh"; then
        status="ok"
    else
        status="failed"
        echo "Run failed for size=$size; continuing to next size"
    fi
    echo "$size,$sub,$(date +%Y-%m-%dT%H:%M:%S),$status" >> "$manifest_csv"
done

DUR=$(( $(date +%s) - START_TS ))
echo
echo "Sweep done in ${DUR}s. Manifest: $manifest_csv"
echo "Plot with:"
echo "  python $SCRIPT_DIR/plot_sweep.py --sweep-dir $SWEEP_DIR"
