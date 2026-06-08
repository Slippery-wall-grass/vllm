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


def _run_vision_encoder(model, pixel_values, extra_kwargs,
                        is_video: bool = False):
    """Call the vision encoder and return the output tensor.

    For video inputs (Qwen2.5-VL), pass video_grid_thw. For images, pass
    image_grid_thw. The same `model.visual()` handles both.
    """
    grid_thw_key = "video_grid_thw" if is_video else "image_grid_thw"
    if hasattr(model, "visual"):
        return model.visual(pixel_values,
                            grid_thw=extra_kwargs.get(grid_thw_key))
    elif hasattr(model, "vision_tower"):
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


def _load_video_frames(video_path: str, num_frames: int = 32) -> list:
    """Load frames from an mp4, uniformly sampled to `num_frames` to match
    vLLM's runtime VideoMediaIO behavior. Pass num_frames=-1 to load all.

    Returns a list of HxWx3 uint8 numpy arrays in RGB order (compatible with
    HF Qwen2.5-VL processor)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fall back to read-all if metadata is missing
        target_idx_set: set[int] | None = None
    elif num_frames > 0 and num_frames < total_frames:
        target_idx_set = set(
            np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        )
    else:
        target_idx_set = None  # use all frames

    frames = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if target_idx_set is None or idx in target_idx_set:
            ret, frame_bgr = cap.retrieve()
            if ret:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


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


def profile_video(model, processor, video_path: str, device: str,
                  num_warmup: int, num_iterations: int,
                  num_frames: int = 32,
                  ) -> tuple[list[float], int]:
    """Run the vision encoder on a video.

    Frames are uniformly sampled to `num_frames` to match what vLLM's
    runtime VideoMediaIO produces (default 32). Mismatching this value
    will produce m_i in profile.json that doesn't match the runtime cache.

    Returns (per_iteration_times, m_i). For Qwen2.5-VL the processor
    accepts `videos=[list_of_frames]` and emits pixel_values_videos plus
    video_grid_thw. The visual encoder itself is the same module used for
    images, but with the video grid.
    """
    frames = _load_video_frames(video_path, num_frames=num_frames)
    inputs = processor(
        videos=[frames],
        text="Describe this video.",
        return_tensors="pt",
    ).to(device)

    if "pixel_values_videos" not in inputs:
        raise RuntimeError(
            "Processor did not produce pixel_values_videos; "
            "this model probably does not support video input."
        )

    pixel_values = inputs["pixel_values_videos"]
    extra_kwargs = {}
    for key in ("video_grid_thw", "second_per_grid_ts"):
        if key in inputs:
            extra_kwargs[key] = inputs[key]

    m_i = 0
    with torch.inference_mode():
        for _ in range(num_warmup):
            out = _run_vision_encoder(
                model, pixel_values, extra_kwargs, is_video=True,
            )
            torch.cuda.synchronize(device)
        if isinstance(out, torch.Tensor):
            if out.dim() == 2:
                m_i = out.shape[0]
            elif out.dim() == 3:
                m_i = out.shape[1]
        elif isinstance(out, (tuple, list)):
            t = out[0] if isinstance(out[0], torch.Tensor) else out
            if isinstance(t, torch.Tensor):
                m_i = t.shape[1] if t.dim() == 3 else t.shape[0]

    times: list[float] = []
    with torch.inference_mode():
        for _ in range(num_iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _run_vision_encoder(
                model, pixel_values, extra_kwargs, is_video=True,
            )
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
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="torch_dtype for the profiled model. Default 'bfloat16' "
             "to match what `vllm serve` uses on Ampere+ GPUs. The "
             "previous default 'float16' caused profile c_i to be "
             "noticeably larger than runtime c_i.",
    )
    parser.add_argument(
        "--attn-impl", type=str, default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="attn_implementation passed to HF from_pretrained. "
             "Default 'auto' tries flash_attention_2 → sdpa → eager. "
             "vLLM uses a FlashAttention-based kernel internally for "
             "Qwen2.5-VL's vision tower; using flash_attention_2 here "
             "puts the HF profile path on the same playing field. If "
             "flash-attn isn't installed, sdpa is the next best.",
    )
    parser.add_argument("--video-num-frames", type=int, default=32,
                        help="Frames to sample per video before encoding. "
                             "Must match VideoMediaIO at serve time (default "
                             "32). Pass -1 to use all frames.")
    parser.add_argument(
        "--skip-cohort", type=str, default="cold", nargs="?",
        const="cold",
        help="Skip manifest entries whose 'cohort' field matches this "
             "value. Default 'cold': skip one-shot videos generated by "
             "generate_scan_workload.py since they're never reused and "
             "don't participate in the distribution-aware solve. Set "
             "to '' to profile everything.",
    )
    args = parser.parse_args()

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    # Drop cold one-shot entries — they don't go through solve_lambda
    # and their c_i/m_i are never read.
    if args.skip_cohort:
        kept = {
            tid: entry for tid, entry in manifest.items()
            if (entry.get("cohort") or "hot") != args.skip_cohort
        }
        skipped = len(manifest) - len(kept)
        if skipped > 0:
            print(f"Skipping {skipped} manifest entries with "
                  f"cohort='{args.skip_cohort}' (typically cold "
                  f"one-shots in scan workloads).")
        manifest = kept

    output_path = args.output_path
    if output_path is None:
        from pathlib import Path
        output_path = str(Path(args.manifest_path).parent / "profile.json")

    m_i_override = {}
    if args.m_i_override:
        m_i_override = json.loads(args.m_i_override)

    device = args.device
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model {args.model} (dtype={args.dtype}, "
          f"attn_impl={args.attn_impl})...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Load just the vision part — use AutoModel to get the full model,
    # then access its vision encoder. Try attn implementations in order
    # so we get vLLM-comparable speed when flash-attn is available.
    from transformers import AutoModel
    attn_chain = (
        [args.attn_impl] if args.attn_impl != "auto"
        else ["flash_attention_2", "sdpa", "eager"]
    )
    model = None
    last_err: Exception | None = None
    for attn in attn_chain:
        try:
            kwargs = dict(
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            # `attn_implementation` is supported on newer transformers
            # versions; pass it only when meaningful.
            if attn:
                kwargs["attn_implementation"] = attn
            model = AutoModel.from_pretrained(args.model, **kwargs)
            print(f"  using attn_implementation={attn}")
            break
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"  attn_implementation={attn} unavailable: {e}")
            last_err = e
            continue
    if model is None:
        raise RuntimeError(
            f"Could not load model with any attn_implementation in "
            f"{attn_chain}. Last error: {last_err}"
        )
    model = model.to(device).eval()

    # Warn if user is still on float16 — measured c_i will be ~3x
    # larger than vLLM's runtime (which defaults to bfloat16 + flash
    # attn). The downstream solve_lambda will be fed an inflated c_i.
    if torch_dtype == torch.float16:
        print("  WARNING: profiling in float16. vLLM uses bfloat16 by "
              "default on Ampere+ GPUs; consider --dtype=bfloat16 for "
              "an apples-to-apples c_i.")

    print(f"Model loaded on {device}\n")

    profile = {}
    for type_id, info in manifest.items():
        media_type = info.get("media_type", "image")
        media_path = info["path"]
        if media_type == "video":
            print(f"Profiling {type_id} (video {info['resolution']}, "
                  f"{info.get('num_frames', '?')} frames)...")
            times, m_i = profile_video(
                model, processor, media_path, device,
                args.num_warmup, args.num_iterations,
                num_frames=args.video_num_frames,
            )
        else:
            print(f"Profiling {type_id} (image {info['resolution']})...")
            image = Image.open(media_path).convert("RGB")
            times, m_i = profile_image(
                model, processor, image, device,
                args.num_warmup, args.num_iterations,
            )

        for i, t in enumerate(times):
            print(f"    iter {i}: c_i={t:.4f}s")

        if type_id in m_i_override:
            m_i = m_i_override[type_id]

        result = {
            "media_type": media_type,
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
