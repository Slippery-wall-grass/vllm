# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate K synthetic test images of varying resolutions for encoder cache
benchmarking.

Usage:
    python generate_test_images.py \
        --num-types 5 \
        --output-dir /tmp/encoder_cache_test_images \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

# Default resolutions to generate different types of images.
# Each resolution produces different m_i (encoder embedding counts).
# Resolutions are kept moderate so the encoder worker (which runs with
# gpu-memory-utilization 0.01 in disagg mode) has enough KV cache to
# fit the visual tokens.  Patch-size 14 gives token counts:
#   224 -> 256,  280 -> 400,  336 -> 576,  392 -> 784,  448 -> 1024
DEFAULT_RESOLUTIONS = [
    (224, 224),
    (252, 252),
    (280, 280),
    (308, 308),
    (336, 336),
    (364, 364),
    (392, 392),
    (448, 448),
]

# Colors for different image types to ensure distinct content hashes.
COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (128, 0, 0),      # dark red
    (0, 128, 0),      # dark green
    (0, 0, 128),      # dark blue
    (128, 128, 0),    # olive
]


def generate_image(resolution: tuple[int, int], color: tuple[int, int, int],
                   type_id: str, output_path: str) -> None:
    """Generate a synthetic image with colored patterns."""
    w, h = resolution
    img = Image.new("RGB", (w, h), color=color)
    draw = ImageDraw.Draw(img)

    # Draw a distinctive pattern so each image has unique content
    for i in range(0, w, max(1, w // 10)):
        draw.line([(i, 0), (w - i, h)], fill=(255, 255, 255), width=2)
    for i in range(0, h, max(1, h // 10)):
        draw.line([(0, i), (w, h - i)], fill=(200, 200, 200), width=1)

    # Add type_id text at center
    draw.text((w // 4, h // 2), type_id, fill=(255, 255, 255))

    img.save(output_path, "JPEG", quality=95)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test images for encoder cache benchmark"
    )
    parser.add_argument("--num-types", type=int, default=5,
                        help="Number of distinct image types to generate")
    parser.add_argument("--output-dir", type=str,
                        default="/tmp/encoder_cache_test_images",
                        help="Directory to save generated images")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Path for output manifest JSON "
                        "(default: <output-dir>/manifest.json)")
    args = parser.parse_args()

    K = args.num_types
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest_path or str(output_dir / "manifest.json")

    if K <= len(DEFAULT_RESOLUTIONS):
        resolutions = DEFAULT_RESOLUTIONS[:K]
    else:
        # Generate K resolutions evenly spaced within the default range
        # (224 to 448) so we never exceed the largest default resolution.
        min_res = DEFAULT_RESOLUTIONS[0][0]
        max_res = DEFAULT_RESOLUTIONS[-1][0]
        step = (max_res - min_res) / (K - 1) if K > 1 else 0
        resolutions = [
            (round(min_res + i * step), round(min_res + i * step))
            for i in range(K)
        ]

    manifest = {}
    for i in range(K):
        type_id = f"type_{i}"
        resolution = resolutions[i]
        color = COLORS[i % len(COLORS)]
        filename = f"{type_id}_{resolution[0]}x{resolution[1]}.jpg"
        filepath = str(output_dir / filename)

        generate_image(resolution, color, type_id, filepath)

        manifest[type_id] = {
            "path": os.path.abspath(filepath),
            "resolution": list(resolution),
            "filename": filename,
        }
        print(f"Generated {type_id}: {resolution[0]}x{resolution[1]} -> "
              f"{filepath}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"Generated {K} image types in {output_dir}")


if __name__ == "__main__":
    main()
