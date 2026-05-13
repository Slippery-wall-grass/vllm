# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a scan-resistant workload for the encoder-cache benchmark.

The workload mixes two cohorts:

  * K_hot recurring "hot" video types — followed Zipfian within the
    cohort. These are the items a smart cache *should* retain.

  * K_cold one-shot "cold" video types — each appears with very low
    probability, so over a typical benchmark window most are visited
    at most once. They are "scan traffic" that pollutes any naive
    LRU/FIFO cache: every cold visit evicts something useful to make
    room for an item that will never be revisited.

Distribution layout:
    p_i_hot  = HOT_FRACTION * zipf(i; s=hot_skew)     for i in hot
    p_i_cold = (1 - HOT_FRACTION) / K_cold            for i in cold

Distribution-Aware should recognise that cold types have low p_i and
high keep_threshold_d, so it refuses to admit them to the cache (or
marks them immediately evictable), protecting the hot working set.
FIFO/LRU has no such signal — every cold request evicts a hot entry
in turn. Persistent EC connector files retain everything forever
which "solves" the miss problem at the cost of unbounded shared
storage.

This is the prototypical workload where Distribution-Aware should
significantly outperform FIFO/LRU on hit rate and on hot-item TTFT,
while FIFO+EC-persist gets best raw throughput but uses ballooning
storage that won't survive long-running deployments.

Usage:
    python generate_scan_workload.py \
        --num-hot 5 --num-cold 100 \
        --hot-fraction 0.8 --hot-skew 1.0 \
        --duration-s 2 --fps 8 \
        --output-dir /tmp/scan_data

The script writes:
    <output-dir>/manifest.json
    <output-dir>/distribution.json
    <output-dir>/<type_id>_<W>x<H>_d<dur>s_fps<F>.mp4   for each type

Then the existing benchmark pipeline can use it:
    SOURCE_DIR=/tmp/scan_data \
    MANIFEST_PATH=/tmp/scan_data/manifest.json \
    DISTRIBUTION="$(cat /tmp/scan_data/distribution.json)" \
    NUM_VIDEOS=$((K_hot + K_cold))  NUM_TYPES=0 \
    bash run_cache_comparison.sh
"""

import argparse
import json
import os
import random
from pathlib import Path

# Reuse generate_test_videos's video writer for consistency.
from generate_test_videos import (  # type: ignore[import-not-found]
    DEFAULT_VIDEO_RESOLUTIONS,
    generate_video,
)


def _build_hot_distribution(num_hot: int, hot_skew: float) -> dict[str, float]:
    """Distribution over the HOT pool only — Zipfian, sums to 1.0.

    This is what feeds into solve_lambda / DistributionAwareCacheManager:
    cold one-shot items are excluded entirely so the algorithm never
    "reserves" cache for things that will never be revisited. Cold items
    arrive at the manager without an entry in `type_metadata` and are
    immediately marked evictable (see _compute_evictability).
    """
    assert num_hot >= 1
    weights = [1.0 / (i + 1) ** hot_skew for i in range(num_hot)]
    norm = sum(weights)
    return {f"type_{i}": w / norm for i, w in enumerate(weights)}


def _pick_resolution(rng: random.Random,
                     candidates: list[tuple[int, int]]) -> tuple[int, int]:
    return rng.choice(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-hot", type=int, default=5,
                        help="Number of recurring 'hot' video types "
                             "(default: 5)")
    parser.add_argument("--num-cold", type=int, default=100,
                        help="Number of one-shot 'cold' video types "
                             "(default: 100)")
    parser.add_argument("--hot-fraction", type=float, default=0.8,
                        help="Fraction of probability mass routed to "
                             "hot types (default: 0.8). The remaining "
                             "1-hot_fraction is spread uniformly across "
                             "cold types.")
    parser.add_argument("--hot-skew", type=float, default=1.0,
                        help="Zipfian exponent within the hot cohort "
                             "(default: 1.0; 0=uniform, 2=heavy-tail)")
    parser.add_argument("--duration-s", type=float, default=2.0,
                        help="Video duration in seconds (default: 2)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Video FPS (default: 8)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write videos + manifest + "
                             "distribution")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Output manifest path (default: "
                             "<output-dir>/manifest.json)")
    parser.add_argument("--distribution-path", type=str, default=None,
                        help="Output distribution path (default: "
                             "<output-dir>/distribution.json)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for resolution sampling.")
    parser.add_argument(
        "--cold-resolution-pool", type=str, default=None,
        help="Optional override for the resolution pool used for cold "
             "videos. Comma-separated WxH (e.g. "
             "'224x224,280x280,336x336'). Default: same pool as hot.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest_path
                     or str(out_dir / "manifest.json"))
    dist_path = (args.distribution_path
                 or str(out_dir / "distribution.json"))

    rng = random.Random(args.seed)
    cold_pool = list(DEFAULT_VIDEO_RESOLUTIONS)
    if args.cold_resolution_pool:
        cold_pool = []
        for chunk in args.cold_resolution_pool.split(","):
            w, h = chunk.lower().split("x")
            cold_pool.append((int(w), int(h)))
    hot_pool = list(DEFAULT_VIDEO_RESOLUTIONS)

    total = args.num_hot + args.num_cold
    print(f"Generating scan workload: {args.num_hot} hot + "
          f"{args.num_cold} cold = {total} total types")
    print(f"  hot_fraction={args.hot_fraction}, "
          f"hot_skew={args.hot_skew}")
    print(f"  duration={args.duration_s}s @ {args.fps}fps")
    print(f"  output: {out_dir}")

    manifest: dict[str, dict] = {}

    # Hot videos
    for i in range(args.num_hot):
        type_id = f"type_{i}"
        resolution = _pick_resolution(rng, hot_pool)
        w, h = resolution
        filename = (f"{type_id}_{w}x{h}_d{args.duration_s}s_"
                    f"fps{args.fps}_hot.mp4")
        filepath = str(out_dir / filename)
        num_frames = generate_video(
            resolution=resolution,
            duration_s=args.duration_s,
            fps=args.fps,
            type_id=type_id,
            color_seed=i * 1000 + 7,
            output_path=filepath,
        )
        manifest[type_id] = {
            "media_type": "video",
            "path": os.path.abspath(filepath),
            "resolution": list(resolution),
            "filename": filename,
            "duration_s": args.duration_s,
            "fps": args.fps,
            "num_frames": num_frames,
            "cohort": "hot",
        }
        print(f"  hot  {type_id}: {w}x{h} {num_frames}f")

    # Cold videos
    for j in range(args.num_cold):
        type_id = f"type_{args.num_hot + j}"
        resolution = _pick_resolution(rng, cold_pool)
        w, h = resolution
        filename = (f"{type_id}_{w}x{h}_d{args.duration_s}s_"
                    f"fps{args.fps}_cold.mp4")
        filepath = str(out_dir / filename)
        num_frames = generate_video(
            resolution=resolution,
            duration_s=args.duration_s,
            fps=args.fps,
            type_id=type_id,
            color_seed=(args.num_hot + j) * 1000 + 7,
            output_path=filepath,
        )
        manifest[type_id] = {
            "media_type": "video",
            "path": os.path.abspath(filepath),
            "resolution": list(resolution),
            "filename": filename,
            "duration_s": args.duration_s,
            "fps": args.fps,
            "num_frames": num_frames,
            "cohort": "cold",
        }
        if j < 3 or j == args.num_cold - 1:
            print(f"  cold {type_id}: {w}x{h} {num_frames}f")
        elif j == 3:
            print(f"  cold ... ({args.num_cold - 4} more not printed)")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")

    # distribution.json contains ONLY the hot pool. Distribution-aware
    # solver operates on this — cold one-shots are deliberately excluded
    # so the algorithm doesn't waste reservation on items that will never
    # be revisited. At runtime, cold items arrive at the cache manager
    # without an entry in `type_metadata` and `_compute_evictability`
    # returns True for them (immediately evictable).
    dist = _build_hot_distribution(
        num_hot=args.num_hot,
        hot_skew=args.hot_skew,
    )
    with open(dist_path, "w") as f:
        json.dump(dist, f, indent=2)
    print(f"Distribution written (hot only, sums to 1.0): {dist_path}")

    # Sidecar metadata that run_benchmark.py reads to know the
    # hot/cold mix ratio for the scan-mode workload.
    meta_path = str(out_dir / "scan_meta.json")
    meta = {
        "num_hot": args.num_hot,
        "num_cold": args.num_cold,
        "hot_fraction": args.hot_fraction,
        "hot_skew": args.hot_skew,
        "schema_note": (
            "distribution.json contains only hot types (sum=1.0). "
            "Use hot_fraction here as the Bernoulli p for picking hot "
            "vs cold per request in WORKLOAD_MODE=scan."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Scan metadata written: {meta_path}")

    # Quick summary
    top_hot = dist.get("type_0", 0.0)
    print()
    print(f"Top hot p_i (within hot pool):  {top_hot:.4f}")
    print(f"Hot fraction (per-request Bernoulli): {args.hot_fraction:.2f}")
    if args.num_cold > 0:
        cold_per_req = (1.0 - args.hot_fraction) / args.num_cold
        print(f"Effective cold p_i: {cold_per_req:.6f}  "
              f"(~1 visit per {int(1 / cold_per_req):,} requests)")

    print()
    print("Next step:")
    print(f"  SOURCE_DIR={out_dir} \\")
    print(f"  MANIFEST_PATH={manifest_path} \\")
    print(f"  DISTRIBUTION=\"$(cat {dist_path})\" \\")
    print(f"  HOT_FRACTION={args.hot_fraction} \\")
    print(f"  NUM_VIDEOS={total} NUM_TYPES=0 SKIP_GENERATION=1 \\")
    print(f"  WORKLOAD_MODE=scan \\")
    print("  bash run_cache_comparison.sh")


if __name__ == "__main__":
    main()
