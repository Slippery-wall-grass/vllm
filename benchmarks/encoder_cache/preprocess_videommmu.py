# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preprocess the lmms-lab/VideoMMMU dataset into the (manifest +
distribution) format used by the encoder-cache benchmark.

The dataset has 300 unique videos, each with 3 questions (Perception,
Comprehension, Adaptation). Videos are referenced via `link_selected`
URLs (web-hosted), so this script downloads them locally before adding
them to the manifest.

Each video appears in 3 rows by construction, giving natural cache
repetition for benchmarking. The script picks K videos and emits the
same manifest schema as `generate_test_videos.py` (media_type="video",
path, resolution, fps, num_frames, duration_s) so all downstream
scripts (profile_encoder.py, run_benchmark.py, run_cache_comparison.sh)
work without changes.

Dependencies:
    pip install datasets yt-dlp opencv-python

Note: VideoMMMU is gated (CC-BY-NC-SA-4.0). Run `huggingface-cli login`
once and accept the dataset's license at https://huggingface.co/
datasets/lmms-lab/VideoMMMU before using this script.

Usage:
    python preprocess_videommmu.py \
        --num-types 5 \
        --output-dir /tmp/vmmu_data
"""

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

VALID_SPLITS = ["perception", "comprehension", "adaptation", "all"]


def _normalize_split_name(s: str) -> str:
    """Map user-friendly names to actual HF split names (capitalised)."""
    s = s.strip().lower()
    if s == "perception":
        return "Perception"
    if s == "comprehension":
        return "Comprehension"
    if s == "adaptation":
        return "Adaptation"
    raise ValueError(f"unknown split {s}")


def collect_url_counts(split: str, hf_token: str | None) -> Counter:
    """Stream the dataset and count occurrences of each link_selected URL."""
    from datasets import load_dataset

    counts: Counter = Counter()
    splits_to_scan = (
        ["Perception", "Comprehension", "Adaptation"]
        if split == "all" else [_normalize_split_name(split)]
    )
    for sp in splits_to_scan:
        print(f"Loading lmms-lab/VideoMMMU split={sp}...")
        ds = load_dataset(
            "lmms-lab/VideoMMMU", split=sp, token=hf_token,
        )
        for row in ds:
            url = row.get("link_selected")
            if url:
                counts[url] += 1
    print(f"Found {len(counts)} unique videos across {sum(counts.values())} "
          f"questions")
    return counts


def have_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None


def download_video(url: str, output_path: str, timeout: int) -> bool:
    """Download a single video using yt-dlp.

    Returns True on success, False on failure or timeout.
    """
    if not have_yt_dlp():
        print("  ERROR: yt-dlp not on PATH. pip install yt-dlp")
        return False
    try:
        # Force mp4 output, single file, quiet operation
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_path,
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--socket-timeout", str(min(timeout, 60)),
            url,
        ]
        result = subprocess.run(
            cmd, timeout=timeout, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  yt-dlp failed (rc={result.returncode}): "
                  f"{result.stderr[:200]}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def probe_video(path: str) -> dict | None:
    """Use cv2 to read width / height / fps / num_frames from a local video.

    Returns None if the file cannot be opened.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or num_frames <= 0:
        return None
    duration_s = num_frames / fps if fps > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "duration_s": duration_s,
    }


def select_videos(
    downloaded: list[tuple[str, str, int, dict]],
    k: int,
    diversify_duration: bool,
) -> list[tuple[str, str, int, dict]]:
    """Pick K videos from the downloaded set.

    `downloaded` items are (url, local_path, count, probe_info).
    Default: top-K by count.
    With `diversify_duration`: bucket by duration into K bins, pick the
    highest-count video per bucket.
    """
    if not diversify_duration or k >= len(downloaded):
        return sorted(downloaded, key=lambda x: -x[2])[:k]

    durations = [d["duration_s"] for *_, d in downloaded]
    min_d, max_d = min(durations), max(durations)
    if min_d == max_d:
        return sorted(downloaded, key=lambda x: -x[2])[:k]
    width = (max_d - min_d) / k
    buckets: list[list] = [[] for _ in range(k)]
    for item in downloaded:
        d = item[3]["duration_s"]
        idx = min(k - 1, int((d - min_d) / width))
        buckets[idx].append(item)
    chosen = []
    used: set[str] = set()
    for b in buckets:
        b.sort(key=lambda x: -x[2])
        if b:
            chosen.append(b[0])
            used.add(b[0][0])
    if len(chosen) < k:
        for item in sorted(downloaded, key=lambda x: -x[2]):
            if item[0] in used:
                continue
            chosen.append(item)
            used.add(item[0])
            if len(chosen) >= k:
                break
    chosen.sort(key=lambda x: x[3]["duration_s"])
    return chosen[:k]


def load_or_init_manifest(manifest_path: str) -> dict:
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def compute_start_idx(manifest: dict, override: int | None) -> int:
    if override is not None:
        return override
    existing = []
    for tid in manifest.keys():
        if tid.startswith("type_"):
            try:
                existing.append(int(tid.split("_", 1)[1]))
            except ValueError:
                pass
    return max(existing, default=-1) + 1


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VideoMMMU into manifest + distribution"
    )
    parser.add_argument("--split", type=str, default="all",
                        choices=VALID_SPLITS,
                        help="Which dataset split to scan")
    parser.add_argument("--num-types", type=int, default=5,
                        help="Number of videos K to pick")
    parser.add_argument("--max-download-count", type=int, default=None,
                        help="Number of candidates to attempt downloading "
                             "(default: K * 3, allowing for failures)")
    parser.add_argument("--output-dir", type=str,
                        default="/tmp/vmmu_data",
                        help="Directory for downloaded videos and outputs")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Existing manifest to extend "
                             "(default: <output-dir>/manifest.json)")
    parser.add_argument("--start-type-id", type=int, default=None,
                        help="Numeric index for the first new type "
                             "(default: max existing index + 1)")
    parser.add_argument("--download-timeout", type=int, default=120,
                        help="Seconds before a single download is killed")
    parser.add_argument("--diversify-duration", action="store_true",
                        help="Pick videos spanning a range of durations "
                             "instead of pure count-descending")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (or use huggingface-cli "
                             "login). VideoMMMU is gated.")
    args = parser.parse_args()

    if not have_yt_dlp():
        print("WARNING: yt-dlp not found; downloads will fail. "
              "Install with: pip install yt-dlp")

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or str(output_dir / "manifest.json")

    counts = collect_url_counts(args.split, args.hf_token)
    if not counts:
        raise RuntimeError("No videos found in dataset")

    max_dl = args.max_download_count or (args.num_types * 3)
    candidates = counts.most_common(max_dl)
    print(f"\nAttempting to download {len(candidates)} candidate videos...")

    downloaded: list[tuple[str, str, int, dict]] = []
    for i, (url, count) in enumerate(candidates):
        out_path = str(video_dir / f"vmmu_{i:04d}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"  [{i+1}/{len(candidates)}] cached: {url}")
        else:
            print(f"  [{i+1}/{len(candidates)}] downloading: {url}")
            ok = download_video(url, out_path, args.download_timeout)
            if not ok:
                continue
        info = probe_video(out_path)
        if info is None:
            print(f"    probe failed; skipping {out_path}")
            continue
        downloaded.append((url, out_path, count, info))
        print(f"    OK: {info['width']}x{info['height']} "
              f"@{info['fps']:.1f}fps, {info['num_frames']} frames "
              f"({info['duration_s']:.1f}s)")
        if len(downloaded) >= args.num_types and not args.diversify_duration:
            # Enough for top-K by count
            break

    if not downloaded:
        raise RuntimeError(
            "All downloads failed. Check yt-dlp installation, network "
            "access, and HuggingFace credentials.")
    if len(downloaded) < args.num_types:
        print(f"WARNING: only {len(downloaded)} downloads succeeded "
              f"(requested {args.num_types}). Reducing K.")

    chosen = select_videos(
        downloaded,
        k=min(args.num_types, len(downloaded)),
        diversify_duration=args.diversify_duration,
    )

    manifest = load_or_init_manifest(manifest_path)
    start_idx = compute_start_idx(manifest, args.start_type_id)

    chosen_ordered = []
    for i, (url, path, count, info) in enumerate(chosen):
        type_id = f"type_{start_idx + i}"
        manifest[type_id] = {
            "media_type": "video",
            "path": os.path.abspath(path),
            "resolution": [info["width"], info["height"]],
            "filename": os.path.basename(path),
            "duration_s": info["duration_s"],
            "fps": info["fps"],
            "num_frames": info["num_frames"],
            "source_url": url,
            "dataset_count": count,
        }
        chosen_ordered.append((type_id, count))
        print(f"\n  {type_id}: {info['width']}x{info['height']} "
              f"{info['duration_s']:.1f}s {info['num_frames']}f "
              f"(count={count})")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Distribution from real counts
    total = sum(c for _, c in chosen_ordered)
    dist = {tid: c / total for tid, c in chosen_ordered}
    dist_path = str(output_dir / "distribution_videommmu.json")
    with open(dist_path, "w") as f:
        json.dump(dist, f, indent=2)
    print(f"Distribution written to {dist_path}")
    print(f"  {dist}")

    print("\nNext step (videos only):")
    print(f"  SKIP_GENERATION=1 WORK_DIR={output_dir} "
          f"IMAGE_DIR={output_dir} \\")
    print(f"    DISTRIBUTION=\"$(cat {dist_path})\" \\")
    print(f"    NUM_VIDEOS={len(chosen_ordered)} \\")
    print("    bash run_cache_comparison.sh")


if __name__ == "__main__":
    main()
