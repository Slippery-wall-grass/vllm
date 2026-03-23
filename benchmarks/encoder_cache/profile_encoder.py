# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile encoder computation time (c_i) and memory cost (m_i) for each
image type by sending requests to a running vLLM encoder server.

Usage:
    python profile_encoder.py \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json \
        --server-url http://localhost:19534 \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-warmup 2 \
        --num-iterations 5 \
        --output-path /tmp/encoder_cache_test_images/profile.json
"""

import argparse
import base64
import json
import statistics
import time

import requests as http_requests


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def send_image_request(server_url: str, model: str, image_path: str,
                       use_base64: bool = True,
                       max_tokens: int = 10) -> float:
    """Send a single image request and measure TTFT.

    Returns TTFT in seconds.
    """
    if use_base64:
        img_b64 = encode_image_to_base64(image_path)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        }
    else:
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"file://{image_path}"},
        }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    start_time = time.perf_counter()
    ttft = None

    with http_requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    # Continue reading to complete the request

    if ttft is None:
        raise RuntimeError(f"No streaming response received for {image_path}")

    return ttft


def profile_image_type(server_url: str, model: str, image_path: str,
                       num_warmup: int, num_iterations: int) -> dict:
    """Profile a single image type by measuring TTFT over multiple iterations.

    Returns dict with c_i (median TTFT) and all measurements.
    """
    # Warmup runs
    for _ in range(num_warmup):
        send_image_request(server_url, model, image_path)

    # Measurement runs
    ttfts = []
    for _ in range(num_iterations):
        ttft = send_image_request(server_url, model, image_path)
        ttfts.append(ttft)

    return {
        "c_i": statistics.median(ttfts),
        "ttft_mean": statistics.mean(ttfts),
        "ttft_std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
        "ttft_all": ttfts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile encoder computation time for each image type"
    )
    parser.add_argument("--manifest-path", type=str, required=True,
                        help="Path to image manifest JSON")
    parser.add_argument("--server-url", type=str,
                        default="http://localhost:19534",
                        help="URL of the vLLM encoder server")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Model name")
    parser.add_argument("--num-warmup", type=int, default=2,
                        help="Number of warmup iterations per image type")
    parser.add_argument("--num-iterations", type=int, default=5,
                        help="Number of measurement iterations per type")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path for output profile JSON")
    parser.add_argument("--m-i-override", type=str, default=None,
                        help="JSON mapping type_id -> m_i to manually set "
                        "encoder embedding counts")
    args = parser.parse_args()

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    output_path = args.output_path
    if output_path is None:
        from pathlib import Path
        output_path = str(
            Path(args.manifest_path).parent / "profile.json"
        )

    m_i_override = {}
    if args.m_i_override:
        m_i_override = json.loads(args.m_i_override)

    profile = {}
    for type_id, info in manifest.items():
        image_path = info["path"]
        print(f"Profiling {type_id} ({info['resolution']})...")

        result = profile_image_type(
            args.server_url, args.model, image_path,
            args.num_warmup, args.num_iterations,
        )

        # m_i: use override if provided, otherwise estimate from resolution
        # In practice, m_i depends on the model's vision encoder and should
        # be measured from the actual model. Here we provide a reasonable
        # default based on typical VL model behavior.
        if type_id in m_i_override:
            result["m_i"] = m_i_override[type_id]
        else:
            # Default estimate: (w * h) / (patch_size^2)
            # For Qwen2.5-VL with patch_size=14: tokens = (w*h)/(14*14)
            w, h = info["resolution"]
            result["m_i"] = max(1, (w * h) // (14 * 14))

        profile[type_id] = result
        print(f"  c_i={result['c_i']:.4f}s, m_i={result['m_i']}, "
              f"ttft_mean={result['ttft_mean']:.4f}s")

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nProfile written to {output_path}")


if __name__ == "__main__":
    main()
