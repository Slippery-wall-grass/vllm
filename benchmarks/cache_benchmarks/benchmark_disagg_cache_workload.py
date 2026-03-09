#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
 
import argparse
import asyncio
import base64
import glob
import json
import os
import sys
import time
from pathlib import Path
 
import numpy as np
 
try:
    import aiohttp
except ImportError:
    print("please install aiohttp: pip install aiohttp")
    sys.exit(1)
 
try:
    from PIL import Image
except ImportError:
    Image = None

PROMPTS = [
    "What is shown in this image? Describe it briefly.",
    "Describe the contents of this image in one paragraph.",
    "What objects and colors can you see in this image?",
    "What is the most prominent feature in this image?",
    "If you had to give this image a title, what would it be?",
    "Describe the spatial layout of elements in this image.",
    "What mood or atmosphere does this image convey?",
    "List the main visual elements in this image.",
]
 
 
# ========================== image generation ==========================
def prepare_images(image_dir: str, num_images: int, seed: int = 42):
    """generate synthetic images with different sizes and content, save to disk."""
    if Image is None:
        print("please install Pillow: pip install Pillow")
        sys.exit(1)
 
    os.makedirs(image_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
 
    sizes = [
        (224, 224),   # small
        (336, 336),   # medium
        (448, 448),   # standard
        (512, 512),   # large
        (640, 480),   # wide
    ]
 
    for i in range(num_images):
        w, h = sizes[i % len(sizes)]
        # different content: random pixels + different color tendencies, ensuring different mm_hash
        base_color = rng.randint(0, 200, size=3)
        noise = rng.randint(0, 56, (h, w, 3), dtype=np.uint8)
        pixels = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels, "RGB")
 
        path = os.path.join(image_dir, f"img_{i:04d}.jpg")
        img.save(path, "JPEG", quality=85)
 
    print(f"Created {num_images} images -> {image_dir}")
    from collections import Counter
    size_counts = Counter(sizes[i % len(sizes)] for i in range(num_images))
    for (w, h), count in sorted(size_counts.items()):
        print(f"  {w}x{h}: {count} images")
 
 
# ======================== distribution ========================
def generate_request_sequence(
    num_images: int,
    num_prompts: int,
    distribution: str = "zipf",
    zipf_alpha: float = 1.5,
    seed: int = 42,
) -> list[int]:
    """Generate a sequence of image references according to the specified distribution."""
    rng = np.random.RandomState(seed)
 
    if distribution == "zipf":
        ranks = np.arange(1, num_images + 1, dtype=float)
        probs = 1.0 / (ranks ** zipf_alpha)
    elif distribution == "uniform":
        probs = np.ones(num_images)
    elif distribution == "bimodal":
        probs = np.ones(num_images)
        probs[: num_images // 4] = 20.0 
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
 
    probs /= probs.sum()
    return rng.choice(num_images, size=num_prompts, p=probs).tolist()
 
 
# ======================== request sending ========================
async def send_one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    image_path: str,
    prompt: str,
    request_idx: int,
) -> dict:
    """Send a chat completion request with a file:// image."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{image_path}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 32,
        "stream": False,
    }
 
    start = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=180)
        ) as resp:
            elapsed = time.perf_counter() - start
            body = await resp.json()
 
            output_tokens = 0
            if resp.status == 200 and "usage" in body:
                output_tokens = body["usage"].get("completion_tokens", 0)
 
            return {
                "idx": request_idx,
                "success": resp.status == 200,
                "ttft": elapsed,
                "status": resp.status,
                "output_tokens": output_tokens,
            }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "idx": request_idx,
            "success": False,
            "ttft": elapsed,
            "error": str(e),
        }
 
 
async def run_workload(
    model: str,
    proxy_port: int,
    image_dir: str,
    image_indices: list[int],
    concurrency: int = 8,
) -> list[dict]:
    """Send requests in sequence (with limited concurrency) and collect results."""
    url = f"http://localhost:{proxy_port}/v1/chat/completions"
 
    image_files = sorted(glob.glob(os.path.join(image_dir, "img_*.jpg")))
    if not image_files:
        print(f"No images found in {image_dir}")
        sys.exit(1)
 
    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
 
    async def bounded(session, idx, img_idx):
        async with sem:
            prompt = PROMPTS[idx % len(PROMPTS)]
            img_path = image_files[img_idx]
            return await send_one_request(
                session, url, model, img_path, prompt, idx
            )
 
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            bounded(session, i, img_idx)
            for i, img_idx in enumerate(image_indices)
        ]
 
        total = len(tasks)
        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                print(f"  Progress: {done}/{total}")
 
    return results
 
 
# ======================== Result Analysis ========================
def analyze(results: list[dict], policy: str, config: dict) -> dict:
    """Calculate summary statistics."""
    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
 
    if not ok:
        return {"policy": policy, "successful": 0, "failed": len(fail),
                "error": "No successful requests"}
 
    ttfts = sorted(r["ttft"] for r in ok)
    out_toks = [r.get("output_tokens", 0) for r in ok]
 
    total_wall = max(r["ttft"] for r in ok) if ok else 0
 
    summary = {
        "policy": policy,
        "total_requests": len(results),
        "successful": len(ok),
        "failed": len(fail),
        "ttft_mean":   float(np.mean(ttfts)),
        "ttft_median": float(np.median(ttfts)),
        "ttft_p50":    float(np.percentile(ttfts, 50)),
        "ttft_p90":    float(np.percentile(ttfts, 90)),
        "ttft_p95":    float(np.percentile(ttfts, 95)),
        "ttft_p99":    float(np.percentile(ttfts, 99)),
        "ttft_min":    float(np.min(ttfts)),
        "ttft_max":    float(np.max(ttfts)),
        "avg_output_tokens": float(np.mean(out_toks)),
        "throughput_rps": len(ok) / total_wall if total_wall > 0 else 0,
    }
    return summary
 
 
def compare_results(results_dir: str, timestamp: str | None = None):
    """Compare multiple result JSON files and print a summary table. If timestamp is provided, only compare files with that timestamp."""
    pattern = os.path.join(results_dir, "results_*")
    if timestamp:
        pattern = os.path.join(results_dir, f"results_*_{timestamp}.json")
 
    files = sorted(glob.glob(pattern))
    if len(files) < 2:
        print(f"At least 2 result files are required, found {len(files)}: {pattern}")
        if files:
            for f in files:
                with open(f) as fp:
                    data = json.load(fp)
                s = data.get("summary", data)
                print(f"\n  Policy: {s.get('policy', '?')}")
                for k, v in s.items():
                    if k != "policy":
                        print(f"    {k}: {v:.4f}" if isinstance(v, float)
                              else f"    {k}: {v}")
        return
 
    summaries = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        summaries.append(data.get("summary", data))
 
    # output table
    metrics = [
        "successful", "failed",
        "ttft_mean", "ttft_median", "ttft_p90", "ttft_p95", "ttft_p99",
        "throughput_rps",
    ]
 
    col_w = 16
    header = f"{'Metric':<20}"
    for s in summaries:
        header += f"  {s.get('policy', '?'):>{col_w}}"
    print(header)
    print("-" * len(header))
 
    for m in metrics:
        row = f"  {m:<18}"
        for s in summaries:
            v = s.get(m, "N/A")
            row += f"  {v:>{col_w}.4f}" if isinstance(v, float) else \
                   f"  {str(v):>{col_w}}"
        print(row)
 
    # Calculate improvement
    lru_s = next((s for s in summaries if s.get("policy") == "lru"), None)
    od_s = next((s for s in summaries if s.get("policy") == "online_dual"), None)
    if lru_s and od_s:
        print(f"\n{'─'*50}")
        if lru_s.get("ttft_mean") and od_s.get("ttft_mean"):
            imp = (lru_s["ttft_mean"] - od_s["ttft_mean"]) / lru_s["ttft_mean"] * 100
            print(f"  TTFT Mean Improvement (OnlineDual vs LRU): {imp:+.2f}%")
        if lru_s.get("ttft_p90") and od_s.get("ttft_p90"):
            imp = (lru_s["ttft_p90"] - od_s["ttft_p90"]) / lru_s["ttft_p90"] * 100
            print(f"  TTFT P90 Improvement (OnlineDual vs LRU):  {imp:+.2f}%")
        if lru_s.get("throughput_rps") and od_s.get("throughput_rps"):
            imp = (od_s["throughput_rps"] - lru_s["throughput_rps"]) / lru_s["throughput_rps"] * 100
            print(f"  Throughput Improvement (OnlineDual vs LRU):     {imp:+.2f}%")
 
 
# ======================== Main Entry ========================
def main():
    parser = argparse.ArgumentParser(
        description="Disagg encoder cache policy benchmark"
    )
 
    # modes
    parser.add_argument("--prepare-images", action="store_true",
                        help="generate synthetic images for the benchmark")
    parser.add_argument("--compare", action="store_true",
                        help="compare results from multiple runs")
 
    # server & proxy
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--proxy-port", type=int, default=10001,
                        help="Disagg proxy port")
 
    # image database
    parser.add_argument("--image-dir", type=str,
                        default="/tmp/vllm_disagg_bench_images")
    parser.add_argument("--num-images", type=int, default=30,
                        help="image database size")
 
    # workload
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="request count")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="concurrent request count")
    parser.add_argument("--distribution", type=str, default="zipf",
                        choices=["zipf", "uniform", "bimodal"])
    parser.add_argument("--zipf-alpha", type=float, default=1.5,
                        help="Zipf distribution parameter, larger values lead to more concentration")
    parser.add_argument("--seed", type=int, default=42)
 
    # output
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--policy-name", type=str, default="unknown")
    parser.add_argument("--results-dir", type=str,
                        default="./logs/disagg_cache_bench")
    parser.add_argument("--timestamp", type=str, default=None)
 
    args = parser.parse_args()
 
    # ---- Mode Selection ----
    if args.prepare_images:
        prepare_images(args.image_dir, args.num_images, args.seed)
        return
 
    if args.compare:
        compare_results(args.results_dir, args.timestamp)
        return
 
    # ---- Generate Request Sequence ----
    image_indices = generate_request_sequence(
        num_images=args.num_images,
        num_prompts=args.num_prompts,
        distribution=args.distribution,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
 
    from collections import Counter
    counts = Counter(image_indices)
    print(f"\n distribution ({args.distribution}, alpha={args.zipf_alpha}):")
    print(f"  Total Requests: {args.num_prompts}")
    print(f"  Unique Images Referenced: {len(counts)}/{args.num_images}")
    top5 = counts.most_common(5)
    print(f"  Top 5 Most Popular Images: {[(f'img_{k:04d}', v) for k, v in top5]}")
    print()
 
    # ---- Send Requests ----
    print(f"Policy: {args.policy_name}, Concurrency: {args.concurrency}")
    results = asyncio.run(
        run_workload(
            model=args.model,
            proxy_port=args.proxy_port,
            image_dir=args.image_dir,
            image_indices=image_indices,
            concurrency=args.concurrency,
        )
    )
 
    # ---- Analyze Results ----
    config = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "num_images": args.num_images,
        "distribution": args.distribution,
        "zipf_alpha": args.zipf_alpha,
        "concurrency": args.concurrency,
        "seed": args.seed,
    }
    summary = analyze(results, args.policy_name, config)
 
    print(f"\n{'='*50}")
    print(f"Policy: {args.policy_name}")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
 
    # ---- Save Results ----
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "config": config,
                        "raw_results": results}, f, indent=2)
        print(f"\nResults saved: {args.output_json}")
 
 
if __name__ == "__main__":
    main()