#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Sweep open-loop QPS and run the 4-policy comparison at each point.
#
# At each QPS value, run_cache_comparison.sh runs all 4 trials
# (none / fifo / fifo_persistent / dist_aware) and writes
# results_<label>.json under a per-QPS subdir. Use plot_qps_sweep.py
# afterwards to draw TTFT / hit-rate / throughput vs QPS.
#
# Usage (assumes you already preprocessed a dataset into $SOURCE_DIR):
#   SOURCE_DIR=/tmp/vmmu_data \
#   SWEEP_DIR=/tmp/qps_sweep \
#   DISTRIBUTION="$(cat /tmp/vmmu_data/distribution_videommmu.json)" \
#   NUM_VIDEOS=5 NUM_TYPES=0 \
#   bash sweep_qps.sh
#
#   # Custom QPS list:
#   SWEEP_QPS="1 2 3 4 5 6 8" \
#   bash sweep_qps.sh
#
# Then plot:
#   python plot_qps_sweep.py --sweep-dir /tmp/qps_sweep
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

SWEEP_DIR="${SWEEP_DIR:-/tmp/qps_sweep}"
SWEEP_QPS="${SWEEP_QPS:-1 2 3 4 5}"

mkdir -p "$SWEEP_DIR"
manifest_csv="$SWEEP_DIR/manifest.csv"
echo "qps,subdir,timestamp,status" > "$manifest_csv"

echo "Source manifest:  $SOURCE_MANIFEST"
echo "Source media dir: $SOURCE_DIR"
echo "Sweep QPS:        $SWEEP_QPS"
echo "Output dir:       $SWEEP_DIR"
echo

START_TS=$(date +%s)
for qps in $SWEEP_QPS; do
    sub="$SWEEP_DIR/qps_${qps}"
    mkdir -p "$sub"
    echo
    echo "============================================================"
    echo "  Sweep: open-loop QPS = $qps"
    echo "  Output: $sub"
    echo "============================================================"

    # Force open-loop BENCH_MODE so QPS is meaningful.
    # WORK_DIR is per-QPS; the source manifest/media stays pinned.
    if WORK_DIR="$sub" \
       IMAGE_DIR="$SOURCE_DIR" \
       MANIFEST_PATH="$SOURCE_MANIFEST" \
       BENCH_MODE="open" \
       QPS="$qps" \
       SKIP_GENERATION=1 \
       bash "$SCRIPT_DIR/run_cache_comparison.sh"; then
        status="ok"
    else
        status="failed"
        echo "Run failed for qps=$qps; continuing to next QPS"
    fi
    echo "$qps,$sub,$(date +%Y-%m-%dT%H:%M:%S),$status" >> "$manifest_csv"
done

DUR=$(( $(date +%s) - START_TS ))
echo
echo "Sweep done in ${DUR}s. Manifest: $manifest_csv"
echo "Plot with:"
echo "  python $SCRIPT_DIR/plot_qps_sweep.py --sweep-dir $SWEEP_DIR"
