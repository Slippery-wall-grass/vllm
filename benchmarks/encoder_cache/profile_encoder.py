# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile encoder computation time (c_i) and memory cost (m_i) for each
image type.

c_i is the pure encoder computation time, i.e. the time saved on a cache hit.
For each measurement iteration we create a *unique variant* of the source
image (by drawing a small random marker) so its mm_hash differs from all
previous requests.  This guarantees the first send is a cache miss.  We then
immediately send the identical image again (guaranteed cache hit) and take
TTFT_miss - TTFT_hit = c_i.

Usage:
    python profile_encoder.py \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json \
        --server-url http://localhost:10001 \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-warmup 2 \
        --num-iterations 5 \
        --output-path /tmp/encoder_cache_test_images/profile.json
"""

import argparse
import base64
import io
import json
import random
import statistics
import time

import requests as http_requests
from PIL import Image, ImageDraw


def image_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image to a JPEG base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_unique_variant(image_path: str, seed: int) -> str:
    """Return a base64 string of *image_path* with a tiny unique marker.

    Drawing a small random rectangle in a corner ensures the image content
    (and therefore its mm_hash) differs from every other variant while
    keeping the encoder workload essentially identical.
    """
    rng = random.Random(seed)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # 2x2 pixel marker in a random corner position
    x = rng.randint(0, max(0, img.width - 3))
    y = rng.randint(0, max(0, img.height - 3))
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    draw.rectangle([x, y, x + 1, y + 1], fill=color)
    return image_to_base64(img)


def send_b64_request(server_url: str, model: str, img_b64: str,
                     max_tokens: int = 10) -> float:
    """Send a base64 image request and return TTFT in seconds."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    },
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    max_retries = 3
    for attempt in range(max_retries):
        start_time = time.perf_counter()
        ttft = None

        try:
            with http_requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as response:
                if response.status_code != 200:
                    body = response.text
                    print(f"  [attempt {attempt+1}] HTTP {response.status_code}"
                          f" from {server_url}: {body[:500]}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if (decoded.startswith("data: ")
                                and decoded != "data: [DONE]"):
                            if ttft is None:
                                ttft = time.perf_counter() - start_time

            if ttft is None:
                print(f"  [attempt {attempt+1}] No streaming data received")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise RuntimeError("No streaming response received")
            return ttft

        except http_requests.exceptions.ConnectionError as e:
            print(f"  [attempt {attempt+1}] Connection error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise

    raise RuntimeError(f"All {max_retries} attempts failed")


def measure_encoder_compute_time(
    server_url: str, model: str, image_path: str,
    num_warmup: int, num_iterations: int,
) -> dict:
    """Measure c_i for a single image type.

    For each iteration:
      1. Create a unique variant of the image (different mm_hash) so the
         first request is a guaranteed cache miss.
      2. Send the *same* variant again — guaranteed cache hit.
      3. c_i = TTFT_miss - TTFT_hit.
    """
    # Warmup: exercise the encoder path so CUDA kernels are compiled.
    for i in range(num_warmup):
        warmup_b64 = make_unique_variant(image_path, seed=-(i + 1))
        send_b64_request(server_url, model, warmup_b64)

    ttft_miss_list: list[float] = []
    ttft_hit_list: list[float] = []
    c_i_list: list[float] = []

    for iteration in range(num_iterations):
        # Unique variant → guaranteed fresh mm_hash → guaranteed miss
        variant_b64 = make_unique_variant(image_path, seed=iteration * 1000)

        ttft_miss = send_b64_request(server_url, model, variant_b64)
        ttft_hit = send_b64_request(server_url, model, variant_b64)

        ttft_miss_list.append(ttft_miss)
        ttft_hit_list.append(ttft_hit)

        c_i = max(0.0, ttft_miss - ttft_hit)
        c_i_list.append(c_i)
        print(f"    iter {iteration}: miss={ttft_miss:.4f}s  "
              f"hit={ttft_hit:.4f}s  c_i={c_i:.4f}s")

    return {
        "c_i": statistics.median(c_i_list),
        "c_i_mean": statistics.mean(c_i_list),
        "c_i_std": statistics.stdev(c_i_list) if len(c_i_list) > 1 else 0.0,
        "c_i_all": c_i_list,
        "ttft_miss_median": statistics.median(ttft_miss_list),
        "ttft_hit_median": statistics.median(ttft_hit_list),
        "ttft_miss_all": ttft_miss_list,
        "ttft_hit_all": ttft_hit_list,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile encoder computation time for each image type"
    )
    parser.add_argument("--manifest-path", type=str, required=True,
                        help="Path to image manifest JSON")
    parser.add_argument("--server-url", type=str,
                        default="http://localhost:10001",
                        help="URL of the vLLM proxy/server")
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

        result = measure_encoder_compute_time(
            args.server_url, args.model, image_path,
            args.num_warmup, args.num_iterations,
        )

        # m_i: use override if provided, otherwise estimate from resolution
        if type_id in m_i_override:
            result["m_i"] = m_i_override[type_id]
        else:
            # Default estimate: (w * h) / (patch_size^2)
            # For Qwen2.5-VL with patch_size=14: tokens = (w*h)/(14*14)
            w, h = info["resolution"]
            result["m_i"] = max(1, (w * h) // (14 * 14))

        profile[type_id] = result
        print(f"  c_i={result['c_i']:.4f}s (encoder compute time), "
              f"m_i={result['m_i']}, "
              f"ttft_miss={result['ttft_miss_median']:.4f}s, "
              f"ttft_hit={result['ttft_hit_median']:.4f}s")

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nProfile written to {output_path}")


if __name__ == "__main__":
    main()
