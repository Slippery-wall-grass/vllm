# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate V synthetic test videos for the encoder cache benchmark.

Each video is an mp4 with `duration_s * fps` frames, drawn at a specified
resolution. Per-video, the frames vary (color cycle + frame-index text)
so the encoder produces non-trivial outputs and each video has a unique
mm_hash. Different videos use different resolutions so m_i / c_i differ.

Appends video entries to an existing manifest.json (produced by
generate_test_images.py) so downstream scripts see a unified manifest
of mixed image+video types.

Dependencies:
    pip install opencv-python

Usage:
    python generate_test_videos.py \
        --num-videos 2 \
        --duration-s 2 \
        --fps 8 \
        --output-dir /tmp/encoder_cache_test_images \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json \
        --start-type-id 5
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

# Default resolutions for video. Smaller than max image resolution because
# total m_i = num_frames * tokens_per_frame, so even at 224x224 a 16-frame
# video is much larger than any single image.
DEFAULT_VIDEO_RESOLUTIONS = [
    (224, 224),
    (252, 252),
    (280, 280),
    (308, 308),
    (336, 336),
]


def generate_video(
    resolution: tuple[int, int],
    duration_s: float,
    fps: int,
    type_id: str,
    color_seed: int,
    output_path: str,
) -> int:
    """Write an mp4 with cycling colors + frame index text.

    Returns total number of frames written.
    """
    w, h = resolution
    num_frames = max(1, int(round(duration_s * fps)))

    # Use mp4v (widely supported, OpenCV ships with it)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    rng = np.random.RandomState(color_seed)
    base_colors = rng.randint(0, 256, size=(num_frames, 3), dtype=np.uint8)

    for f in range(num_frames):
        # Solid color background that changes per frame
        b, g, r = int(base_colors[f, 0]), int(base_colors[f, 1]), int(
            base_colors[f, 2]
        )
        frame = np.full((h, w, 3), (b, g, r), dtype=np.uint8)

        # Diagonal pattern so encoder sees structure (not just flat color)
        for i in range(0, w, max(1, w // 10)):
            cv2.line(frame, (i, 0), (w - i, h),
                     (255 - b, 255 - g, 255 - r), 1)

        # Frame index + type label so each frame's content is distinct
        text = f"{type_id} f{f}"
        cv2.putText(frame, text, (max(2, w // 16), max(20, h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 600),
                    (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    return num_frames


def load_or_init_manifest(manifest_path: str) -> dict:
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test videos for encoder cache benchmark"
    )
    parser.add_argument("--num-videos", type=int, default=2,
                        help="Number of distinct video types V")
    parser.add_argument("--duration-s", type=float, default=2.0,
                        help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second")
    parser.add_argument("--output-dir", type=str,
                        default="/tmp/encoder_cache_test_images",
                        help="Directory to save mp4 files")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Existing manifest.json to extend "
                             "(default: <output-dir>/manifest.json)")
    parser.add_argument("--start-type-id", type=int, default=None,
                        help="Numeric index for the first video type. "
                             "Defaults to (max existing image type idx) + 1.")
    args = parser.parse_args()

    V = args.num_videos
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or str(output_dir / "manifest.json")

    manifest = load_or_init_manifest(manifest_path)

    # Compute starting type ID
    if args.start_type_id is not None:
        start_idx = args.start_type_id
    else:
        existing_indices = []
        for tid in manifest.keys():
            if tid.startswith("type_"):
                try:
                    existing_indices.append(int(tid.split("_", 1)[1]))
                except ValueError:
                    continue
        start_idx = max(existing_indices, default=-1) + 1

    # Pick V resolutions
    if V <= len(DEFAULT_VIDEO_RESOLUTIONS):
        resolutions = DEFAULT_VIDEO_RESOLUTIONS[:V]
    else:
        min_res = DEFAULT_VIDEO_RESOLUTIONS[0][0]
        max_res = DEFAULT_VIDEO_RESOLUTIONS[-1][0]
        step = (max_res - min_res) / (V - 1) if V > 1 else 0
        resolutions = [
            (round(min_res + i * step), round(min_res + i * step))
            for i in range(V)
        ]

    for i in range(V):
        type_idx = start_idx + i
        type_id = f"type_{type_idx}"
        resolution = resolutions[i]
        w, h = resolution
        filename = (f"{type_id}_{w}x{h}_d{args.duration_s}s_"
                    f"fps{args.fps}.mp4")
        filepath = str(output_dir / filename)

        num_frames = generate_video(
            resolution=resolution,
            duration_s=args.duration_s,
            fps=args.fps,
            type_id=type_id,
            color_seed=type_idx * 1000 + 7,
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
        }
        print(f"Generated {type_id}: {w}x{h} {args.duration_s}s "
              f"@ {args.fps}fps ({num_frames} frames) -> {filepath}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest updated at {manifest_path}")
    print(f"Total entries: {len(manifest)} "
          f"(images + videos)")


if __name__ == "__main__":
    main()
