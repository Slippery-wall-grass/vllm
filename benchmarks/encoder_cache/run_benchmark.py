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


async def run_benchmark_open(
    workload: list[dict],
    server_url: str,
    model: str,
    qps: float,
    max_tokens: int = 20,
) -> tuple[list[dict], float]:
    """Open-loop benchmark: dispatch requests at a fixed QPS regardless of
    downstream latency. Useful for measuring steady-state TTFT at a given
    load level, NOT for measuring max throughput.

    Returns (results, wall_clock_seconds).
    """
    results = []
    interval = 1.0 / qps if qps > 0 else 0

    connector = aiohttp.TCPConnector(limit=1000)
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
            result["phase"] = "measure"
            results.append(result)

    wall_clock = time.perf_counter() - bench_start
    return results, wall_clock


async def run_benchmark_closed(
    workload: list[dict],
    server_url: str,
    model: str,
    concurrency: int,
    warmup_requests: int,
    max_tokens: int = 20,
) -> tuple[list[dict], float]:
    """Closed-loop (saturation) benchmark: N workers continuously pull from a
    shared queue and fire requests as fast as the server accepts them. This
    drives the system to its throughput ceiling.

    The first `warmup_requests` items are marked as "warmup" and excluded
    from throughput/latency metrics so that the cache reaches steady state
    before measurement starts. The measurement window begins when the first
    non-warmup request is dispatched and ends when the last non-warmup
    request completes.

    Returns (results, measurement_wall_clock_seconds).
    """
    results: list[dict] = []
    results_lock = asyncio.Lock()

    # Shared index into the workload
    next_idx = 0
    idx_lock = asyncio.Lock()

    # Track the measurement window independently of warmup
    measure_start: float | None = None
    measure_end: float | None = None

    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 100))
    total = len(workload)

    async def worker(session: aiohttp.ClientSession, worker_id: int) -> None:
        nonlocal next_idx, measure_start, measure_end
        while True:
            async with idx_lock:
                if next_idx >= total:
                    return
                i = next_idx
                next_idx += 1

            item = workload[i]
            is_warmup = i < warmup_requests

            # Start the measurement clock on the first non-warmup dispatch
            if not is_warmup and measure_start is None:
                measure_start = time.perf_counter()

            result = await send_request(
                session, server_url, model,
                item["image_path"], max_tokens,
            )
            result["request_id"] = i
            result["type_id"] = item["type_id"]
            result["phase"] = "warmup" if is_warmup else "measure"

            # Stamp completion time; the last non-warmup completion wins
            if not is_warmup:
                measure_end = time.perf_counter()

            async with results_lock:
                results.append(result)

    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(worker(session, w))
            for w in range(concurrency)
        ]
        await asyncio.gather(*workers)

    if measure_start is None or measure_end is None:
        # Either no warmup occurred or nothing was measured
        return results, 0.0
    return results, measure_end - measure_start


def compute_metrics(results: list[dict], wall_clock: float = 0.0) -> dict:
    """Compute aggregate metrics from benchmark results.

    Requests whose `phase` is "warmup" are excluded from all metrics so that
    throughput / TTFT reflect steady-state cache behavior. `wall_clock` must
    already be the measurement-window duration (warmup excluded).
    """
    measured = [r for r in results if r.get("phase", "measure") != "warmup"]
    successful = [r for r in measured if r["success"]]
    failed = [r for r in measured if not r["success"]]

    throughput = (len(successful) / wall_clock) if wall_clock > 0 else 0.0

    if not successful:
        return {
            "total_requests": len(results),
            "measured_requests": len(measured),
            "warmup_requests": len(results) - len(measured),
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
        "measured_requests": len(measured),
        "warmup_requests": len(results) - len(measured),
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


def aggregate_rounds(round_metrics: list[dict]) -> dict:
    """Aggregate metrics across multiple rounds into mean ± std."""
    if len(round_metrics) == 1:
        # Single round: just copy the metrics, std = 0
        m = round_metrics[0]
        agg = {}
        for key in ("throughput_rps", "ttft_mean_ms", "ttft_median_ms",
                     "ttft_p95_ms", "ttft_p99_ms", "latency_mean_ms",
                     "latency_median_ms"):
            val = m.get(key)
            if val is not None:
                agg[key] = {"mean": val, "std": 0.0}
        agg["successful"] = {
            "mean": m.get("successful", 0), "std": 0.0,
        }
        agg["failed"] = {
            "mean": m.get("failed", 0), "std": 0.0,
        }
        return agg

    agg = {}
    for key in ("throughput_rps", "ttft_mean_ms", "ttft_median_ms",
                 "ttft_p95_ms", "ttft_p99_ms", "latency_mean_ms",
                 "latency_median_ms"):
        vals = [m[key] for m in round_metrics if key in m and m[key] is not None]
        if vals:
            agg[key] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
    for key in ("successful", "failed"):
        vals = [m.get(key, 0) for m in round_metrics]
        agg[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return agg


def print_results(metrics: dict, label: str = "",
                  aggregated: dict | None = None,
                  num_rounds: int = 1) -> None:
    """Print benchmark results in a table.

    If aggregated is provided (multi-round), show mean ± std format.
    Otherwise show single-round metrics.
    """
    header = f"Benchmark Results{f' ({label})' if label else ''}"
    print("=" * 60)
    print(header)
    if num_rounds > 1:
        print(f"  ({num_rounds} rounds aggregated)")
    print("=" * 60)

    if aggregated and num_rounds > 1:
        def fmt_agg(key: str) -> str:
            v = aggregated.get(key)
            if v is None:
                return "N/A"
            return f"{v['mean']:.2f} ± {v['std']:.2f}"

        print(f"Successful:        {fmt_agg('successful')}")
        print(f"Failed:            {fmt_agg('failed')}")
        print(f"Throughput:        {fmt_agg('throughput_rps')} req/s")
        print(f"TTFT mean:         {fmt_agg('ttft_mean_ms')} ms")
        print(f"TTFT median:       {fmt_agg('ttft_median_ms')} ms")
        print(f"TTFT p95:          {fmt_agg('ttft_p95_ms')} ms")
        print(f"TTFT p99:          {fmt_agg('ttft_p99_ms')} ms")
        print(f"Latency mean:      {fmt_agg('latency_mean_ms')} ms")
    else:
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
    parser.add_argument("--mode", type=str, default="open",
                        choices=["open", "closed"],
                        help="'open' = fixed QPS (latency-at-load test); "
                        "'closed' = fixed concurrency saturation test "
                        "(max throughput)")
    parser.add_argument("--qps", type=float, default=2.0,
                        help="[open mode] Requests per second")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="[closed mode] Number of in-flight workers")
    parser.add_argument("--warmup-requests", type=int, default=0,
                        help="[closed mode] Number of initial requests to "
                        "exclude from metrics so the cache reaches steady "
                        "state. Recommended: at least 2x cache capacity "
                        "in items.")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Max output tokens per request")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for workload generation")
    parser.add_argument("--num-rounds", type=int, default=1,
                        help="Number of independent rounds to run. Each "
                        "round uses a different seed (seed + round_idx) "
                        "and the results are aggregated with mean ± std.")
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

    num_rounds = args.num_rounds
    per_round_metrics: list[dict] = []
    last_metrics: dict = {}
    last_results: list[dict] = []

    for round_idx in range(num_rounds):
        round_seed = args.seed + round_idx

        if num_rounds > 1:
            print(f"\n{'='*60}")
            print(f"Round {round_idx + 1}/{num_rounds} (seed={round_seed})")
            print(f"{'='*60}")

        if args.mode == "closed":
            print(f"Generating workload: {args.num_requests} requests "
                  f"(warmup={args.warmup_requests}), "
                  f"concurrency={args.concurrency}, seed={round_seed}")
        else:
            print(f"Generating workload: {args.num_requests} requests, "
                  f"QPS={args.qps}, seed={round_seed}")

        workload = generate_workload(
            manifest, distribution, args.num_requests, round_seed,
        )

        # Print distribution summary (only on first round)
        if round_idx == 0:
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
        if args.mode == "closed":
            results, wall_clock = asyncio.run(
                run_benchmark_closed(
                    workload, args.server_url, args.model,
                    args.concurrency, args.warmup_requests, args.max_tokens,
                )
            )
        else:
            results, wall_clock = asyncio.run(
                run_benchmark_open(
                    workload, args.server_url, args.model,
                    args.qps, args.max_tokens,
                )
            )

        metrics = compute_metrics(results, wall_clock)
        per_round_metrics.append(metrics)
        last_metrics = metrics
        last_results = results

        if num_rounds > 1:
            print(f"  Round {round_idx + 1}: throughput="
                  f"{metrics.get('throughput_rps', 0):.2f} req/s, "
                  f"ttft_mean={metrics.get('ttft_mean_ms', 0):.2f} ms, "
                  f"successful={metrics.get('successful', 0)}, "
                  f"failed={metrics.get('failed', 0)}")

    # Aggregate across rounds
    aggregated = aggregate_rounds(per_round_metrics)
    print_results(last_metrics, args.label, aggregated, num_rounds)

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
            "mode": args.mode,
            "num_requests": args.num_requests,
            "qps": args.qps,
            "concurrency": args.concurrency,
            "warmup_requests": args.warmup_requests,
            "seed": args.seed,
            "num_rounds": num_rounds,
            "distribution": distribution,
        },
        "metrics": last_metrics,
        "aggregated": aggregated,
        "per_round": per_round_metrics,
        "raw_results": last_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
