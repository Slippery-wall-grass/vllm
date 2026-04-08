# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run encoder cache benchmark by sending requests following a specified
distribution to a running vLLM server and measuring TTFT.

Usage:
    python run_benchmark.py \
        --manifest-path /tmp/encoder_cache_test_images/manifest.json \
        --distribution '{"type_0": 0.3, "type_1": 0.2, ...}' \
        --num-requests 500 \
        --server-url http://localhost:10001 \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --output-path /tmp/benchmark_results.json \
        --qps 2
"""

import argparse
import asyncio
import base64
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_workload(
    manifest: dict,
    distribution: dict[str, float],
    num_requests: int,
    seed: int = 42,
) -> list[dict]:
    """Generate a workload of requests following the specified distribution.

    Returns list of dicts with type_id and image_path.
    """
    rng = random.Random(seed)

    type_ids = list(distribution.keys())
    weights = [distribution[t] for t in type_ids]

    workload = []
    for _ in range(num_requests):
        type_id = rng.choices(type_ids, weights=weights, k=1)[0]
        image_path = manifest[type_id]["path"]
        workload.append({
            "type_id": type_id,
            "image_path": image_path,
        })

    return workload


async def send_request(
    session: aiohttp.ClientSession,
    server_url: str,
    model: str,
    image_path: str,
    max_tokens: int = 20,
) -> dict:
    """Send a single streaming request and measure TTFT and total latency."""
    img_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    start_time = time.perf_counter()
    ttft = None
    output_tokens = 0

    try:
        async with session.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            response.raise_for_status()
            async for line in response.content:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue
                for part in decoded.split("\n"):
                    part = part.strip()
                    if part.startswith("data: ") and part != "data: [DONE]":
                        if ttft is None:
                            ttft = time.perf_counter() - start_time
                        try:
                            chunk = json.loads(part[6:])
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    output_tokens += 1
                        except json.JSONDecodeError:
                            pass

        total_latency = time.perf_counter() - start_time

        return {
            "ttft": ttft,
            "total_latency": total_latency,
            "output_tokens": output_tokens,
            "success": ttft is not None,
        }
    except Exception as e:
        return {
            "ttft": None,
            "total_latency": time.perf_counter() - start_time,
            "output_tokens": 0,
            "success": False,
            "error": str(e),
        }


async def run_benchmark(
    workload: list[dict],
    server_url: str,
    model: str,
    qps: float,
    max_tokens: int = 20,
) -> tuple[list[dict], float]:
    """Run the benchmark by sending requests at the specified QPS.

    Returns (results, wall_clock_seconds).
    """
    results = []
    interval = 1.0 / qps if qps > 0 else 0

    connector = aiohttp.TCPConnector(limit=100)
    bench_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, item in enumerate(workload):
            if i > 0 and interval > 0:
                await asyncio.sleep(interval)

            task = asyncio.create_task(
                send_request(session, server_url, model,
                             item["image_path"], max_tokens)
            )
            tasks.append((i, item["type_id"], task))

        for i, type_id, task in tasks:
            result = await task
            result["request_id"] = i
            result["type_id"] = type_id
            results.append(result)

    wall_clock = time.perf_counter() - bench_start
    return results, wall_clock


def compute_metrics(results: list[dict], wall_clock: float = 0.0) -> dict:
    """Compute aggregate metrics from benchmark results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    throughput = (len(successful) / wall_clock) if wall_clock > 0 else 0.0

    if not successful:
        return {
            "total_requests": len(results),
            "successful": 0,
            "failed": len(failed),
            "wall_clock_s": wall_clock,
            "throughput_rps": throughput,
            "error": "All requests failed",
        }

    ttfts = [r["ttft"] for r in successful]
    latencies = [r["total_latency"] for r in successful]

    # Per-type TTFT
    type_ttfts: dict[str, list[float]] = {}
    for r in successful:
        tid = r["type_id"]
        if tid not in type_ttfts:
            type_ttfts[tid] = []
        type_ttfts[tid].append(r["ttft"])

    type_metrics = {}
    for tid, tts in type_ttfts.items():
        type_metrics[tid] = {
            "count": len(tts),
            "ttft_mean_ms": statistics.mean(tts) * 1000,
            "ttft_median_ms": statistics.median(tts) * 1000,
            "ttft_p95_ms": sorted(tts)[int(len(tts) * 0.95)] * 1000
            if len(tts) > 1 else tts[0] * 1000,
        }

    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "wall_clock_s": wall_clock,
        "throughput_rps": throughput,
        "ttft_mean_ms": statistics.mean(ttfts) * 1000,
        "ttft_median_ms": statistics.median(ttfts) * 1000,
        "ttft_p95_ms": sorted(ttfts)[int(len(ttfts) * 0.95)] * 1000
        if len(ttfts) > 1 else ttfts[0] * 1000,
        "ttft_p99_ms": sorted(ttfts)[int(len(ttfts) * 0.99)] * 1000
        if len(ttfts) > 1 else ttfts[0] * 1000,
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "per_type": type_metrics,
    }


def print_results(metrics: dict, label: str = "") -> None:
    """Print benchmark results in a table."""
    header = f"Benchmark Results{f' ({label})' if label else ''}"
    print("=" * 60)
    print(header)
    print("=" * 60)
    print(f"Total requests:    {metrics['total_requests']}")
    print(f"Successful:        {metrics['successful']}")
    print(f"Failed:            {metrics['failed']}")
    print(f"Wall clock:        {metrics.get('wall_clock_s', 0):.2f} s")
    print(f"Throughput:        {metrics.get('throughput_rps', 0):.2f} req/s")
    print(f"TTFT mean:         {metrics.get('ttft_mean_ms', 'N/A'):.2f} ms")
    print(f"TTFT median:       {metrics.get('ttft_median_ms', 'N/A'):.2f} ms")
    print(f"TTFT p95:          {metrics.get('ttft_p95_ms', 'N/A'):.2f} ms")
    print(f"TTFT p99:          {metrics.get('ttft_p99_ms', 'N/A'):.2f} ms")
    print(f"Latency mean:      {metrics.get('latency_mean_ms', 'N/A'):.2f} ms")
    print()

    if "per_type" in metrics:
        print(f"{'Type':<12} {'Count':<8} {'TTFT mean':<12} "
              f"{'TTFT median':<12} {'TTFT p95':<12}")
        print("-" * 60)
        for tid, tm in sorted(metrics["per_type"].items()):
            print(f"{tid:<12} {tm['count']:<8} "
                  f"{tm['ttft_mean_ms']:<12.2f} "
                  f"{tm['ttft_median_ms']:<12.2f} "
                  f"{tm['ttft_p95_ms']:<12.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run encoder cache benchmark"
    )
    parser.add_argument("--manifest-path", type=str, required=True,
                        help="Path to image manifest JSON")
    parser.add_argument("--distribution", type=str, required=True,
                        help="JSON dict mapping type_id -> probability")
    parser.add_argument("--num-requests", type=int, default=200,
                        help="Number of requests to send")
    parser.add_argument("--server-url", type=str,
                        default="http://localhost:10001",
                        help="URL of the vLLM proxy/server")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Model name")
    parser.add_argument("--qps", type=float, default=2.0,
                        help="Requests per second")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Max output tokens per request")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for workload generation")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path for output results JSON")
    parser.add_argument("--label", type=str, default="",
                        help="Label for this benchmark run (e.g., 'fifo' or "
                        "'distribution_aware')")
    args = parser.parse_args()

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    distribution = json.loads(args.distribution)

    # Normalize
    total_p = sum(distribution.values())
    if abs(total_p - 1.0) > 0.01:
        print(f"Normalizing distribution (sum={total_p})")
        distribution = {k: v / total_p for k, v in distribution.items()}

    print(f"Generating workload: {args.num_requests} requests, "
          f"QPS={args.qps}, seed={args.seed}")
    workload = generate_workload(
        manifest, distribution, args.num_requests, args.seed
    )

    # Print distribution summary
    type_counts: dict[str, int] = {}
    for item in workload:
        type_counts[item["type_id"]] = type_counts.get(
            item["type_id"], 0
        ) + 1
    print("Workload distribution:")
    for tid, count in sorted(type_counts.items()):
        print(f"  {tid}: {count} requests "
              f"({count/len(workload)*100:.1f}%)")

    print(f"\nSending requests to {args.server_url}...")
    results, wall_clock = asyncio.run(
        run_benchmark(workload, args.server_url, args.model,
                      args.qps, args.max_tokens)
    )

    metrics = compute_metrics(results, wall_clock)
    print_results(metrics, args.label)

    # Save results
    output_path = args.output_path
    if output_path is None:
        output_path = str(
            Path(args.manifest_path).parent
            / f"benchmark_results_{args.label or 'default'}.json"
        )

    output = {
        "label": args.label,
        "config": {
            "num_requests": args.num_requests,
            "qps": args.qps,
            "seed": args.seed,
            "distribution": distribution,
        },
        "metrics": metrics,
        "raw_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
