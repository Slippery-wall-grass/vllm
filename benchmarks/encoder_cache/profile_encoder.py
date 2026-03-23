# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile encoder computation time (c_i) and memory cost (m_i) for each
image type.

c_i is measured directly via GPU-synchronised timing in encoder_runner.
The encoder worker writes each encode time to a shared file (controlled
by VLLM_ENCODE_TIME_FILE env var).  For each iteration we:
  1. Clear the encoder cache via /reset_encoder_cache.
  2. Send the image (guaranteed cache miss, encoder runs).
  3. Read the latest encode time from the shared file.

Usage:
    python profile_encoder.py \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json \
        --server-url http://localhost:10001 \
        --encode-time-file /tmp/encode_times.txt \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-warmup 2 \
        --num-iterations 10 \
        --output-path /tmp/encoder_cache_test_images/profile.json
"""

import argparse
import base64
import json
import os
import statistics
import time

import requests as http_requests


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def reset_encoder_cache(server_url: str) -> None:
    resp = http_requests.post(f"{server_url}/reset_encoder_cache", timeout=30)
    resp.raise_for_status()


def read_encode_times(filepath: str) -> list[float]:
    """Read all encode times from the shared file."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        times = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    times.append(float(line))
                except ValueError:
                    pass
        return times


def clear_encode_time_file(filepath: str) -> None:
    """Clear the shared encode time file."""
    with open(filepath, "w") as f:
        f.truncate(0)


def send_image_request(server_url: str, model: str, img_b64: str,
                       max_tokens: int = 10) -> float:
    """Send a base64 image request. Returns TTFT in seconds."""
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
    encode_time_file: str, num_warmup: int, num_iterations: int,
) -> dict:
    """Measure c_i for a single image type.

    For each iteration:
      1. Clear the encode time file and reset encoder cache.
      2. Send image (cache miss -> encoder runs, writes compute time to file).
      3. Read the compute time from the file.
    """
    img_b64 = encode_image_to_base64(image_path)

    # Warmup
    for _ in range(num_warmup):
        send_image_request(server_url, model, img_b64)

    c_i_list: list[float] = []
    ttft_list: list[float] = []

    for iteration in range(num_iterations):
        # Clear file + cache
        clear_encode_time_file(encode_time_file)
        reset_encoder_cache(server_url)

        ttft = send_image_request(server_url, model, img_b64)
        ttft_list.append(ttft)

        # Read the encode time(s) written during this request
        # Small delay to ensure file write is complete
        time.sleep(0.2)
        times = read_encode_times(encode_time_file)

        if times:
            # Use the last entry (in case multiple were written)
            c_i = times[-1]
            c_i_list.append(c_i)
            print(f"    iter {iteration}: c_i={c_i:.4f}s  ttft={ttft:.4f}s")
        else:
            print(f"    iter {iteration}: c_i=N/A (file empty)  "
                  f"ttft={ttft:.4f}s")

    if not c_i_list:
        raise RuntimeError(
            f"No encoder compute times found in {encode_time_file}. "
            f"Make sure the encoder worker is started with "
            f"VLLM_ENCODE_TIME_FILE={encode_time_file}"
        )

    return {
        "c_i": statistics.median(c_i_list),
        "c_i_mean": statistics.mean(c_i_list),
        "c_i_std": statistics.stdev(c_i_list) if len(c_i_list) > 1 else 0.0,
        "c_i_all": c_i_list,
        "ttft_median": statistics.median(ttft_list),
        "ttft_all": ttft_list,
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
    parser.add_argument("--encode-time-file", type=str,
                        default="/tmp/vllm_encode_times.txt",
                        help="Path to shared encode time file "
                        "(must match VLLM_ENCODE_TIME_FILE on encoder worker)")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Model name")
    parser.add_argument("--num-warmup", type=int, default=2,
                        help="Number of warmup iterations per image type")
    parser.add_argument("--num-iterations", type=int, default=10,
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
            args.encode_time_file, args.num_warmup, args.num_iterations,
        )

        if type_id in m_i_override:
            result["m_i"] = m_i_override[type_id]
        else:
            w, h = info["resolution"]
            result["m_i"] = max(1, (w * h) // (14 * 14))

        profile[type_id] = result
        print(f"  c_i={result['c_i']:.4f}s (encoder compute time), "
              f"m_i={result['m_i']}")

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nProfile written to {output_path}")


if __name__ == "__main__":
    main()
