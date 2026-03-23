# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile encoder computation time (c_i) and memory cost (m_i) for each
image type.

c_i is the pure encoder computation time measured directly via GPU-
synchronised timing in the encoder_runner.  For each iteration we:
  1. Clear the encoder cache via /reset_encoder_cache.
  2. Send the image (guaranteed cache miss, encoder runs).
  3. Parse the encoder worker log for "Encoder compute time: X.XXXXs".

Usage:
    python profile_encoder.py \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json \
        --server-url http://localhost:10001 \
        --encoder-log /tmp/encoder_cache_benchmark/logs/encoder_profile_*.log \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --num-warmup 2 \
        --num-iterations 5 \
        --output-path /tmp/encoder_cache_test_images/profile.json
"""

import argparse
import base64
import glob
import json
import re
import statistics
import time

import requests as http_requests

# Pattern matching the log line emitted by encoder_runner.py
_ENCODE_TIME_RE = re.compile(
    r"Encoder compute time: ([\d.]+)s \((\d+) items?\)"
)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def reset_encoder_cache(server_url: str) -> None:
    resp = http_requests.post(f"{server_url}/reset_encoder_cache", timeout=30)
    resp.raise_for_status()


def read_last_encode_time(log_path: str) -> float | None:
    """Read the last 'Encoder compute time' entry from the encoder log."""
    # Resolve glob (e.g. logs/encoder_profile_*.log)
    paths = sorted(glob.glob(log_path))
    if not paths:
        return None
    # Use the latest log file
    with open(paths[-1], "r", errors="replace") as f:
        lines = f.readlines()
    # Search backwards for the last occurrence
    for line in reversed(lines):
        m = _ENCODE_TIME_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def count_encode_time_entries(log_path: str) -> int:
    """Count total 'Encoder compute time' entries in the log."""
    paths = sorted(glob.glob(log_path))
    if not paths:
        return 0
    count = 0
    with open(paths[-1], "r", errors="replace") as f:
        for line in f:
            if _ENCODE_TIME_RE.search(line):
                count += 1
    return count


def send_image_request(server_url: str, model: str, img_b64: str,
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
    encoder_log: str, num_warmup: int, num_iterations: int,
) -> dict:
    """Measure c_i for a single image type.

    For each iteration:
      1. Reset encoder cache.
      2. Send image (cache miss -> encoder runs, logs compute time).
      3. Parse encoder log for the GPU-synchronised compute time.
    """
    img_b64 = encode_image_to_base64(image_path)

    # Warmup
    for _ in range(num_warmup):
        send_image_request(server_url, model, img_b64)

    c_i_list: list[float] = []
    ttft_list: list[float] = []

    for iteration in range(num_iterations):
        # Record how many log entries exist before this request
        count_before = count_encode_time_entries(encoder_log)

        # Clear cache -> next request is guaranteed miss
        reset_encoder_cache(server_url)

        ttft = send_image_request(server_url, model, img_b64)
        ttft_list.append(ttft)

        # Wait briefly for log flush
        time.sleep(0.5)

        # Read the latest encode time from the log
        c_i = read_last_encode_time(encoder_log)

        # Verify a new entry appeared
        count_after = count_encode_time_entries(encoder_log)
        if c_i is not None and count_after > count_before:
            c_i_list.append(c_i)
            print(f"    iter {iteration}: c_i={c_i:.4f}s  ttft={ttft:.4f}s")
        else:
            print(f"    iter {iteration}: c_i=N/A (log entry not found)  "
                  f"ttft={ttft:.4f}s")

    if not c_i_list:
        print("  WARNING: no encoder compute times found in log, "
              "falling back to TTFT")
        c_i_list = ttft_list

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
    parser.add_argument("--encoder-log", type=str, default=None,
                        help="Glob pattern for the encoder worker log file "
                        "(e.g. /tmp/.../logs/encoder_profile_*.log)")
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

    encoder_log = args.encoder_log
    if encoder_log is None:
        from pathlib import Path
        encoder_log = str(
            Path(args.manifest_path).parent.parent
            / "logs" / "encoder_profile_*.log"
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
            encoder_log, args.num_warmup, args.num_iterations,
        )

        # m_i: use override if provided, otherwise estimate from resolution
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
