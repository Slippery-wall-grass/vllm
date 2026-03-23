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


def _run_vision_encoder(model, pixel_values, extra_kwargs):
    """Call the vision encoder and return the output tensor."""
    if hasattr(model, "visual"):
        # Qwen2-VL style
        return model.visual(pixel_values,
                            grid_thw=extra_kwargs.get("image_grid_thw"))
    elif hasattr(model, "vision_tower"):
        # LLaVA style
        return model.vision_tower(pixel_values)
    elif hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values)
    elif hasattr(model, "vision_model"):
        return model.vision_model(pixel_values)
    else:
        raise RuntimeError(
            "Cannot find vision encoder on model. "
            f"Model type: {type(model).__name__}"
        )


def profile_image(model, processor, image: Image.Image, device: str,
                  num_warmup: int, num_iterations: int
                  ) -> tuple[list[float], int]:
    """Run the vision encoder on *image*.

    Returns (per_iteration_times, m_i) where m_i is the number of
    encoder output embeddings (from the output tensor shape).
    """
    # Preprocess
    inputs = processor(
        images=image,
        text="Describe this image.",
        return_tensors="pt",
    ).to(device)

    if "pixel_values" not in inputs:
        raise RuntimeError("Processor did not produce pixel_values")

    pixel_values = inputs["pixel_values"]

    extra_kwargs = {}
    for key in ("image_grid_thw", "image_sizes", "image_bound"):
        if key in inputs:
            extra_kwargs[key] = inputs[key]

    # Warmup and get m_i from output shape
    m_i = 0
    with torch.inference_mode():
        for _ in range(num_warmup):
            out = _run_vision_encoder(model, pixel_values, extra_kwargs)
            torch.cuda.synchronize(device)
        # m_i = number of output embeddings
        if isinstance(out, torch.Tensor):
            # Shape is typically (num_embeds, hidden_dim) or
            # (batch, num_embeds, hidden_dim)
            if out.dim() == 2:
                m_i = out.shape[0]
            elif out.dim() == 3:
                m_i = out.shape[1]
        elif isinstance(out, (tuple, list)):
            t = out[0] if isinstance(out[0], torch.Tensor) else out
            if isinstance(t, torch.Tensor):
                m_i = t.shape[1] if t.dim() == 3 else t.shape[0]

    # Measure
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(num_iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _run_vision_encoder(model, pixel_values, extra_kwargs)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

    return times, m_i


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
        times, m_i = profile_image(
            model, processor, image, device,
            args.num_warmup, args.num_iterations,
        )

        for i, t in enumerate(times):
            print(f"    iter {i}: c_i={t:.4f}s")

        if type_id in m_i_override:
            m_i = m_i_override[type_id]

        result = {
            "c_i": statistics.median(times),
            "c_i_mean": statistics.mean(times),
            "c_i_std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "c_i_all": times,
            "m_i": m_i,
        }

        profile[type_id] = result
        print(f"  c_i={result['c_i']:.4f}s, m_i={m_i}\n")

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile written to {output_path}")


if __name__ == "__main__":
    main()
