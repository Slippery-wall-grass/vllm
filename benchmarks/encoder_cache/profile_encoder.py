# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile encoder computation time (c_i) for each image type.

Directly loads the model's vision encoder and runs it on each test image
with GPU-synchronised timing.  No server needed.

Usage:
    python profile_encoder.py \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-warmup 3 \
        --num-iterations 10 \
        --output-path /tmp/encoder_cache_test_images/profile.json
"""

import argparse
import json
import statistics
import time

import torch
from PIL import Image
from transformers import AutoProcessor


def profile_image(model, processor, image: Image.Image, device: str,
                  num_warmup: int, num_iterations: int) -> list[float]:
    """Run the vision encoder on *image* and return per-iteration times."""
    # Preprocess
    inputs = processor(
        images=image,
        text="Describe this image.",
        return_tensors="pt",
    ).to(device)

    # Extract pixel_values (the vision encoder input)
    if "pixel_values" not in inputs:
        raise RuntimeError("Processor did not produce pixel_values")

    pixel_values = inputs["pixel_values"]

    # Some models need image_grid_thw or similar
    extra_kwargs = {}
    for key in ("image_grid_thw", "image_sizes", "image_bound"):
        if key in inputs:
            extra_kwargs[key] = inputs[key]

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            if hasattr(model, "visual"):
                # Qwen2-VL style
                model.visual(pixel_values, grid_thw=extra_kwargs.get(
                    "image_grid_thw"))
            elif hasattr(model, "vision_tower"):
                # LLaVA style
                model.vision_tower(pixel_values)
            elif hasattr(model, "get_image_features"):
                model.get_image_features(pixel_values)
            else:
                # Generic: try embed_multimodal or vision_model
                if hasattr(model, "vision_model"):
                    model.vision_model(pixel_values)
                else:
                    raise RuntimeError(
                        "Cannot find vision encoder on model. "
                        f"Model type: {type(model).__name__}"
                    )
            torch.cuda.synchronize(device)

    # Measure
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(num_iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            if hasattr(model, "visual"):
                model.visual(pixel_values, grid_thw=extra_kwargs.get(
                    "image_grid_thw"))
            elif hasattr(model, "vision_tower"):
                model.vision_tower(pixel_values)
            elif hasattr(model, "get_image_features"):
                model.get_image_features(pixel_values)
            elif hasattr(model, "vision_model"):
                model.vision_model(pixel_values)

            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

    return times


def main():
    parser = argparse.ArgumentParser(
        description="Profile encoder computation time for each image type"
    )
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--m-i-override", type=str, default=None,
                        help="JSON mapping type_id -> m_i")
    args = parser.parse_args()

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    output_path = args.output_path
    if output_path is None:
        from pathlib import Path
        output_path = str(Path(args.manifest_path).parent / "profile.json")

    m_i_override = {}
    if args.m_i_override:
        m_i_override = json.loads(args.m_i_override)

    device = args.device
    print(f"Loading model {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Load just the vision part — use AutoModel to get the full model,
    # then access its vision encoder.
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()

    print(f"Model loaded on {device}\n")

    profile = {}
    for type_id, info in manifest.items():
        image_path = info["path"]
        print(f"Profiling {type_id} ({info['resolution']})...")

        image = Image.open(image_path).convert("RGB")
        times = profile_image(
            model, processor, image, device,
            args.num_warmup, args.num_iterations,
        )

        for i, t in enumerate(times):
            print(f"    iter {i}: c_i={t:.4f}s")

        result = {
            "c_i": statistics.median(times),
            "c_i_mean": statistics.mean(times),
            "c_i_std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "c_i_all": times,
        }

        if type_id in m_i_override:
            result["m_i"] = m_i_override[type_id]
        else:
            w, h = info["resolution"]
            result["m_i"] = max(1, (w * h) // (14 * 14))

        profile[type_id] = result
        print(f"  c_i={result['c_i']:.4f}s, m_i={result['m_i']}\n")

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile written to {output_path}")


if __name__ == "__main__":
    main()
