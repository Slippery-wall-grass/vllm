# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preprocess a real multimodal dataset (default: lmarena-ai/VisionArena-Chat)
into the (manifest + distribution) format expected by the rest of the
encoder-cache benchmark pipeline.

Strategy
--------
Real VLM datasets contain tens of thousands of unique images, so a naive
replay gets near-zero cache hits for any policy. Instead, we extract K
representative images and treat them as the K "types" used by the existing
distribution-aware algorithm:

  1. Stream the dataset and hash every image we see.
  2. Pick K images that are (a) frequently requested and (b) diverse in
     resolution (so m_i / c_i differs across types).
  3. Save those K images as JPEGs and emit manifest.json matching the
     synthetic-image pipeline's schema.
  4. Compute p_i from the real counts. If duplication is too sparse to form
     a meaningful distribution, also emit a Zipf fallback.

Dependencies
------------
  pip install datasets pillow

Usage
-----
  python preprocess_real_dataset.py \
      --num-types 5 \
      --max-scan 50000 \
      --output-dir /tmp/real_data
"""

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def extract_images(row: dict) -> list[tuple[bytes, tuple[int, int]]]:
    """Extract (bytes, resolution) for every image in a single dataset row.

    VisionArena-Chat stores images under `row["images"]` as a list of dicts,
    each dict contains `"bytes"`. Handles variations defensively.
    """
    out: list[tuple[bytes, tuple[int, int]]] = []
    images = row.get("images") or row.get("image") or []
    if not isinstance(images, list):
        images = [images]

    for img in images:
        try:
            if isinstance(img, dict) and "bytes" in img and img["bytes"]:
                img_bytes = img["bytes"]
            elif isinstance(img, (bytes, bytearray)):
                img_bytes = bytes(img)
            elif isinstance(img, Image.Image):
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=95)
                img_bytes = buf.getvalue()
            else:
                continue

            # Decode to get resolution
            pil = Image.open(io.BytesIO(img_bytes))
            resolution = (pil.width, pil.height)
            out.append((img_bytes, resolution))
        except Exception:
            # Skip unreadable / corrupt rows rather than aborting.
            continue
    return out


def scan_dataset(
    dataset_name: str,
    split: str,
    max_scan: int,
) -> tuple[Counter, dict[str, tuple[bytes, tuple[int, int]]]]:
    """Stream the dataset and return (hash_counts, hash_to_sample).

    hash_to_sample maps a hash to one representative (bytes, resolution)
    tuple so we can later save the image to disk without rescanning.
    """
    from datasets import load_dataset

    print(f"Streaming {dataset_name} (split={split})...")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    counts: Counter = Counter()
    samples: dict[str, tuple[bytes, tuple[int, int]]] = {}

    scanned = 0
    image_total = 0
    for row in ds:
        scanned += 1
        for img_bytes, resolution in extract_images(row):
            h = hash_image_bytes(img_bytes)
            counts[h] += 1
            if h not in samples:
                samples[h] = (img_bytes, resolution)
            image_total += 1

        if scanned % 1000 == 0:
            print(f"  scanned {scanned} rows, {image_total} images, "
                  f"{len(counts)} unique hashes")

        if max_scan and scanned >= max_scan:
            break

    print(f"Done: {scanned} rows, {image_total} images, "
          f"{len(counts)} unique hashes")
    return counts, samples


def pick_representative_images(
    counts: Counter,
    samples: dict[str, tuple[bytes, tuple[int, int]]],
    k: int,
    candidate_pool_multiplier: int = 50,
) -> list[tuple[str, int, tuple[int, int]]]:
    """Pick K hashes by count-descending + resolution-diversity.

    Step 1: take top `candidate_pool_multiplier * K` by count.
    Step 2: bucket by resolution (max(w, h)) into K evenly-spaced bins
            spanning the candidates' resolution range; pick the highest-count
            hash in each bin. Empty bins fall back to next-highest candidate.

    Returns list of (hash, count, resolution) sorted by resolution ascending,
    one per bucket.
    """
    pool_size = max(k, candidate_pool_multiplier * k)
    candidates = counts.most_common(pool_size)
    if len(candidates) < k:
        print(f"WARNING: only {len(candidates)} unique images available, "
              f"less than requested K={k}")
        return [(h, c, samples[h][1]) for h, c in candidates]

    # Bucket candidates by max(w, h)
    cand_with_res = [
        (h, c, samples[h][1], max(samples[h][1]))
        for h, c in candidates
    ]
    min_res = min(r for *_, r in cand_with_res)
    max_res = max(r for *_, r in cand_with_res)

    if k == 1 or min_res == max_res:
        return [(h, c, res) for h, c, res, _ in cand_with_res[:k]]

    # K evenly-spaced buckets on [min_res, max_res]
    bucket_width = (max_res - min_res) / k
    bucket_boundaries = [min_res + i * bucket_width for i in range(k + 1)]
    # Assign each candidate to a bucket
    buckets: list[list] = [[] for _ in range(k)]
    for h, c, res, rmax in cand_with_res:
        idx = min(k - 1, int((rmax - min_res) / bucket_width))
        buckets[idx].append((h, c, res))
    # Sort each bucket by count desc, pick best
    chosen: list[tuple[str, int, tuple[int, int]]] = []
    used_hashes: set[str] = set()
    for b in buckets:
        b.sort(key=lambda x: -x[1])
        if b:
            h, c, res = b[0]
            chosen.append((h, c, res))
            used_hashes.add(h)

    # Any empty bucket → top leftover candidate not yet used
    if len(chosen) < k:
        for h, c, res, _ in cand_with_res:
            if h in used_hashes:
                continue
            chosen.append((h, c, res))
            used_hashes.add(h)
            if len(chosen) >= k:
                break

    # Sort by resolution ascending so type_0 is smallest, type_{K-1} biggest
    chosen.sort(key=lambda x: max(x[2]))
    return chosen[:k]


def save_image(img_bytes: bytes, output_path: str) -> None:
    """Save bytes as JPEG, converting through PIL for format consistency."""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil.save(output_path, "JPEG", quality=95)


def build_manifest(
    chosen: list[tuple[str, int, tuple[int, int]]],
    output_dir: Path,
) -> dict:
    manifest: dict[str, dict[str, Any]] = {}
    for i, (_h, _c, resolution) in enumerate(chosen):
        type_id = f"type_{i}"
        w, h = resolution
        filename = f"{type_id}_{w}x{h}.jpg"
        filepath = str(output_dir / filename)
        manifest[type_id] = {
            "path": os.path.abspath(filepath),
            "resolution": list(resolution),
            "filename": filename,
        }
    return manifest


def build_real_distribution(
    chosen: list[tuple[str, int, tuple[int, int]]],
) -> dict[str, float]:
    total = sum(c for _, c, _ in chosen)
    if total == 0:
        return {f"type_{i}": 1.0 / len(chosen) for i in range(len(chosen))}
    dist = {f"type_{i}": c / total for i, (_, c, _) in enumerate(chosen)}
    # Normalize to sum exactly 1.0 after rounding
    return dist


def build_zipf_distribution(k: int) -> dict[str, float]:
    raw = [1.0 / (i + 1) for i in range(k)]
    total = sum(raw)
    return {f"type_{i}": raw[i] / total for i in range(k)}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess real VLM dataset into manifest + distribution"
    )
    parser.add_argument("--dataset", type=str,
                        default="lmarena-ai/VisionArena-Chat",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--num-types", type=int, default=5,
                        help="Number of image types K")
    parser.add_argument("--max-scan", type=int, default=50000,
                        help="Max number of dataset rows to scan")
    parser.add_argument("--output-dir", type=str,
                        default="/tmp/real_data",
                        help="Directory for images and JSON outputs")
    parser.add_argument("--min-count-threshold", type=int, default=5,
                        help="If the least-frequent chosen image has count "
                             "below this, emit a Zipf fallback distribution.")
    parser.add_argument("--candidate-pool-multiplier", type=int, default=50,
                        help="Size of the top-count candidate pool, as a "
                             "multiple of K. Larger = more resolution "
                             "diversity at the cost of including less-"
                             "frequent images.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    counts, samples = scan_dataset(
        args.dataset, args.split, args.max_scan,
    )

    if not counts:
        raise RuntimeError("No images extracted from dataset")

    chosen = pick_representative_images(
        counts, samples, args.num_types,
        candidate_pool_multiplier=args.candidate_pool_multiplier,
    )

    print(f"\nSelected {len(chosen)} representative images:")
    for i, (h, c, res) in enumerate(chosen):
        print(f"  type_{i}: hash={h[:12]}..., count={c}, resolution={res}")

    # Save images
    manifest = build_manifest(chosen, image_dir)
    for i, (h, _c, _res) in enumerate(chosen):
        type_id = f"type_{i}"
        img_bytes = samples[h][0]
        save_image(img_bytes, manifest[type_id]["path"])

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Write real distribution
    real_dist = build_real_distribution(chosen)
    real_dist_path = output_dir / "distribution_real.json"
    with open(real_dist_path, "w") as f:
        json.dump(real_dist, f, indent=2)
    print(f"Real distribution written to {real_dist_path}")
    print(f"  {real_dist}")

    # Write Zipf fallback if counts are too sparse
    min_count = min(c for _, c, _ in chosen) if chosen else 0
    if min_count < args.min_count_threshold:
        print(f"\nWARNING: min count = {min_count} < threshold "
              f"{args.min_count_threshold}. Real distribution may be noisy.")
        zipf_dist = build_zipf_distribution(len(chosen))
        zipf_path = output_dir / "distribution_zipf.json"
        with open(zipf_path, "w") as f:
            json.dump(zipf_dist, f, indent=2)
        print(f"Zipf fallback distribution written to {zipf_path}")
        print(f"  {zipf_dist}")

    print("\nNext step:")
    print(f"  SKIP_GENERATION=1 WORK_DIR={output_dir} "
          f"IMAGE_DIR={image_dir} \\")
    print(f"    DISTRIBUTION=\"$(cat {real_dist_path})\" \\")
    print("    bash run_cache_comparison.sh")


if __name__ == "__main__":
    main()
