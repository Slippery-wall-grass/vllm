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
import os
import random
import re
import statistics
import time
from pathlib import Path

import aiohttp


import requests as http_requests


_CACHE_STATS_RE = re.compile(
    r"Encoder cache stats before reset:\s*"
    r"hits=(\d+)\s+misses=(\d+)\s+total=(\d+)\s+hit_rate=([\d.]+)"
)

# Pulls per-group encoder-forward timing from the encoder worker log.
# Emitted by gpu_model_runner.py:timed_encoder_operation when
# VLLM_REQUEST_TIMING_TRACE=1. duration_ms is total wallclock for the
# group; per_request_ms is duration_ms / num_items.
_ENCODER_FWD_RE = re.compile(
    r"EncoderForwardTrace\s+req_ids=\S+\s+num_items=(\d+)\s+"
    r"duration_ms=([\d.]+)\s+per_request_ms=([\d.]+)"
)


def parse_latest_cache_stats(log_path: str) -> dict | None:
    """Scan an encoder-worker log file for the latest cache-stats line.

    Returns a dict with cache_hits / cache_misses / cache_total /
    cache_hit_rate, or None if the file is missing / no stats line found.
    Reads the file in chunks from the end so it stays cheap on big logs.
    """
    if not log_path or not os.path.exists(log_path):
        return None
    try:
        size = os.path.getsize(log_path)
        if size == 0:
            return None
        chunk = min(size, 65536)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - chunk))
            tail = f.read().decode("utf-8", errors="replace")
        matches = _CACHE_STATS_RE.findall(tail)
        if not matches:
            return None
        hits, misses, total, hit_rate = matches[-1]
        return {
            "cache_hits": int(hits),
            "cache_misses": int(misses),
            "cache_total": int(total),
            "cache_hit_rate": float(hit_rate),
        }
    except OSError:
        return None


def parse_encoder_forward_stats(
    log_path: str, after_byte_offset: int = 0,
) -> tuple[dict | None, int]:
    """Parse EncoderForwardTrace lines from `log_path` starting at byte
    offset `after_byte_offset`. Used to compute per-round encoder
    forward statistics — call once before a round to record the file
    size, then again after the round with that offset to scope reads.

    Returns:
        (stats_dict_or_None, new_byte_offset)

        stats_dict has:
          encoder_forward_events:   # of EncoderForwardTrace lines
          encoder_forward_items:    sum of num_items across events
          encoder_forward_total_ms: sum of duration_ms across events
          encoder_forward_mean_ms:  total_ms / items
                                    (= mean GPU forward time per mm
                                    item that went through the encoder
                                    — the work cache hits avoid)
        Returns None when the file is missing.

    Endpoints measured:
        START: model.embed_multimodal(**mm_kwargs_group) invocation
        END:   embed_multimodal() returns (after torch.cuda.synchronize)
    Cache HITs never call embed_multimodal → emit no event → counted
    as 0 implicitly when downstream code divides by N_measured_requests.
    """
    if not log_path or not os.path.exists(log_path):
        return None, after_byte_offset
    try:
        size = os.path.getsize(log_path)
        if size <= after_byte_offset:
            return {
                "encoder_forward_events": 0,
                "encoder_forward_items": 0,
                "encoder_forward_total_ms": 0.0,
                "encoder_forward_mean_ms": 0.0,
            }, size
        with open(log_path, "rb") as f:
            f.seek(after_byte_offset)
            chunk = f.read().decode("utf-8", errors="replace")
        events = _ENCODER_FWD_RE.findall(chunk)
        n_events = len(events)
        n_items = sum(int(e[0]) for e in events)
        total_ms = sum(float(e[1]) for e in events)
        mean_per_item = (total_ms / n_items) if n_items > 0 else 0.0
        return {
            "encoder_forward_events": n_events,
            "encoder_forward_items": n_items,
            "encoder_forward_total_ms": total_ms,
            "encoder_forward_mean_ms": mean_per_item,
        }, size
    except OSError:
        return None, after_byte_offset


def reset_encoder_cache(encoder_url: str) -> None:
    """Reset the encoder cache on the encoder worker.

    Requires VLLM_SERVER_DEV_MODE=1 on the encoder worker.
    """
    resp = http_requests.post(
        f"{encoder_url}/reset_encoder_cache", timeout=30,
    )
    resp.raise_for_status()
    print(f"  Encoder cache reset via {encoder_url}")


def encode_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_to_base64(image_path: str) -> str:
    return encode_file_to_base64(image_path)


def generate_workload(
    manifest: dict,
    distribution: dict[str, float],
    num_requests: int,
    seed: int = 42,
    workload_mode: str = "prob",
    hot_fraction: float | None = None,
) -> list[dict]:
    """Generate a workload of requests.

    Two modes:

    * "prob" (default):  Each request samples a type from `distribution`
      with replacement. This is the original probabilistic mode — best
      for measuring steady-state behaviour on a closed type set.

    * "scan":  Per-request Bernoulli(hot_fraction) decides whether to
      pick from the hot cohort (with replacement, weighted by p_i
      restricted to hot types) or to consume the next unused cold type
      (without replacement). Cold types are identified by
      `manifest[tid].cohort == "cold"` (set by generate_scan_workload.py);
      anything else is treated as hot.

      This is the streaming scan-resistant workload: 80% of requests
      revisit the working set, 20% bring in items that will never be
      seen again. Classic adversarial pattern for LRU/FIFO; the
      benchmark for our distribution-aware algorithm.

      Requires num_cold_types >= ceil(num_requests * (1-hot_fraction)).

    If `hot_fraction` is None in scan mode, it is computed as the sum
    of probabilities of hot types in the distribution (e.g. 0.8 for a
    distribution where hot types sum to 0.8 and cold types sum to 0.2).

    Returns list of dicts with type_id, media_path, and media_type.
    """
    rng = random.Random(seed)

    if workload_mode == "prob":
        type_ids = list(distribution.keys())
        weights = [distribution[t] for t in type_ids]
        workload = []
        for _ in range(num_requests):
            type_id = rng.choices(type_ids, weights=weights, k=1)[0]
            entry = manifest[type_id]
            workload.append({
                "type_id": type_id,
                "media_path": entry["path"],
                "media_type": entry.get("media_type", "image"),
                "image_path": entry["path"],
            })
        return workload

    if workload_mode != "scan":
        raise ValueError(
            f"workload_mode must be 'prob' or 'scan', got {workload_mode!r}")

    # ---------------- scan mode ----------------
    # Classification source of truth is the manifest cohort tag.
    # `distribution` contains ONLY hot types (sums to 1.0); cold types
    # live in the manifest with cohort='cold' and are NOT in the
    # distribution — they bypass dist-aware's reservation logic
    # entirely. This matches the design intent that cold one-shot data
    # should never participate in the policy decision.
    hot_types: list[str] = []
    cold_types: list[str] = []
    for tid, entry in manifest.items():
        cohort = (entry.get("cohort") or "hot").lower()
        if cohort == "cold":
            cold_types.append(tid)
        elif tid in distribution:
            hot_types.append(tid)
        # else: hot in manifest but missing from distribution — ignore
        #       to avoid silently changing the workload mix.

    if not hot_types:
        raise ValueError(
            "scan mode requires at least one hot type. Use "
            "generate_scan_workload.py; hot types must appear in both "
            "manifest (cohort='hot') and distribution.json.")
    if not cold_types:
        raise ValueError(
            "scan mode requires cold types in the manifest "
            "(cohort='cold'). Got 0.")

    # Hot picks are weighted by p_i restricted to hot types
    hot_weights = [distribution[t] for t in hot_types]
    if sum(hot_weights) <= 0:
        hot_weights = [1.0] * len(hot_types)

    if hot_fraction is None:
        # No CLI override. distribution.json sums to 1.0 over hot only,
        # so we cannot infer the hot/cold mix from it. Fall back to 0.8
        # with a warning — that's the generate_scan_workload.py default.
        print("WARNING: scan mode received no --hot-fraction. "
              "distribution.json contains only the hot pool (sum=1.0) "
              "so the hot/cold mix cannot be inferred from it. "
              "Defaulting hot_fraction=0.8. Pass it explicitly or "
              "write a scan_meta.json sidecar to silence this.")
        hot_fraction = 0.8
    if not 0.0 <= hot_fraction <= 1.0:
        raise ValueError(
            f"hot_fraction must be in [0, 1], got {hot_fraction}")
    expected_cold = int(round(num_requests * (1.0 - hot_fraction)))
    if expected_cold > len(cold_types):
        raise ValueError(
            f"scan mode needs at least {expected_cold} cold types for "
            f"{num_requests} requests at hot_fraction={hot_fraction}, "
            f"but only {len(cold_types)} are available. Regenerate "
            f"the workload with --num-cold >= {expected_cold} (use "
            f"generate_scan_workload.py).")

    # Shuffle cold types so the one-shot order isn't always the same
    rng.shuffle(cold_types)
    cold_iter = iter(cold_types)

    workload: list[dict] = []
    for _ in range(num_requests):
        if rng.random() < hot_fraction:
            tid = rng.choices(hot_types, weights=hot_weights, k=1)[0]
        else:
            try:
                tid = next(cold_iter)
            except StopIteration:
                # Fewer cold visits than the pool ratio predicted — fall
                # back to a hot pick rather than crash mid-run.
                tid = rng.choices(hot_types, weights=hot_weights, k=1)[0]
        entry = manifest[tid]
        workload.append({
            "type_id": tid,
            "media_path": entry["path"],
            "media_type": entry.get("media_type", "image"),
            "image_path": entry["path"],
        })

    return workload


def build_guaranteed_warmup_workload(
    manifest: dict,
    distribution: dict[str, float],
    target_count: int,
    seed: int = 0,
) -> list[dict]:
    """Build a warmup workload that includes at least one request per type.

    The first len(types) entries are exactly one of each type (in
    deterministic order: type_0, type_1, ...). Any remaining slots up to
    target_count are filled by sampling from `distribution` as usual.

    If target_count < num_types, the workload still has one of each type
    so all kernels get warmed - the actual length may exceed target_count
    in that case.
    """
    rng = random.Random(seed)
    type_ids = list(distribution.keys())

    # First: one of each type, in manifest order
    workload: list[dict] = []
    for tid in type_ids:
        if tid not in manifest:
            continue
        entry = manifest[tid]
        workload.append({
            "type_id": tid,
            "media_path": entry["path"],
            "media_type": entry.get("media_type", "image"),
            "image_path": entry["path"],
        })

    # Then: fill remaining slots with sampled requests (if any)
    weights = [distribution[t] for t in type_ids]
    while len(workload) < target_count:
        tid = rng.choices(type_ids, weights=weights, k=1)[0]
        entry = manifest[tid]
        workload.append({
            "type_id": tid,
            "media_path": entry["path"],
            "media_type": entry.get("media_type", "image"),
            "image_path": entry["path"],
        })
    return workload


def _build_media_content(
    media_path: str,
    media_type: str,
    type_id: str | None = None,
    media_mode: str = "file",
) -> dict:
    """Build the OpenAI-style content part for an image or video.

    `media_mode` controls how the media is referenced:

    * "file" (default): emit a `file://<absolute_path>` URL. The server
      reads the file directly from disk via `--allowed-local-media-path`.
      This keeps request bodies tiny so TTFT isn't dominated by base64
      upload + JSON parse — important for measuring encoder cache
      effects, which would otherwise be drowned in transport overhead.
    * "base64": inline the file as a base64 data URL. Useful when the
      server can't share a filesystem with the client (containerized
      remote setups).

    When `type_id` is provided, it is attached as the OpenAI-extension
    `uuid` field. vLLM uses that uuid directly as the mm_hash (when no
    hf_processor_mm_kwargs are set), which keeps the benchmark's notion
    of "type" aligned with the runtime's notion of "cache key" — needed
    for the distribution-aware cache to look up `hash_to_type` correctly.
    """
    media_mode = (media_mode or "file").lower()
    if media_mode not in ("file", "base64"):
        raise ValueError(
            f"media_mode must be 'file' or 'base64', got {media_mode!r}")

    if media_mode == "file":
        # Use file:// so server short-circuits to local disk read. Make
        # the path absolute so it's unambiguous regardless of the
        # server's CWD, and forward-slash-normalised so it works on
        # Windows clients too.
        abs_path = os.path.abspath(media_path).replace("\\", "/")
        url = f"file://{abs_path}"
    else:
        b64 = encode_file_to_base64(media_path)
        if media_type == "video":
            url = f"data:video/mp4;base64,{b64}"
        else:
            url = f"data:image/jpeg;base64,{b64}"

    if media_type == "video":
        item = {"type": "video_url", "video_url": {"url": url}}
    else:
        item = {"type": "image_url", "image_url": {"url": url}}

    if type_id is not None:
        item["uuid"] = type_id
    return item


# A long deterministic English passage used to pad prompts to a target
# token count. Repeats a fixed paragraph so the same target_tokens
# always yields the same string (no randomness across runs).
_LOREM_PARAGRAPH = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut "
    "enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat "
    "nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
    "sunt in culpa qui officia deserunt mollit anim id est laborum. "
)


def make_prompt_text(media_type: str, target_tokens: int = 0) -> str:
    """Construct the text part of the prompt with a target length.

    `target_tokens` is approximate (uses ~4 characters/token rule of
    thumb). When 0 (the default), returns the original short prompt so
    behaviour is unchanged for callers that don't care.

    The text uses a fixed lorem-ipsum-like paragraph so the same
    `target_tokens` produces the same string every run — no per-request
    variability that would muddy the encoder-cache measurement.
    """
    suffix = (
        "Describe this video briefly."
        if media_type == "video"
        else "Describe this image briefly."
    )
    if target_tokens <= 0:
        return suffix

    # ~4 chars per token (English BPE rule-of-thumb).
    target_chars = max(0, target_tokens * 4 - len(suffix))
    if target_chars <= 0:
        return suffix
    n_repeats = (target_chars // len(_LOREM_PARAGRAPH)) + 1
    body = (_LOREM_PARAGRAPH * n_repeats)[:target_chars]
    return f"{body} {suffix}"


async def send_request(
    session: aiohttp.ClientSession,
    server_url: str,
    model: str,
    image_path: str,
    max_tokens: int = 20,
    media_type: str = "image",
    type_id: str | None = None,
    media_mode: str = "file",
    prompt_text: str | None = None,
) -> dict:
    """Send a single streaming request and measure TTFT and total latency.

    `image_path` is kept as the parameter name for backwards compatibility
    but accepts any media path (image or video). `media_type` controls the
    payload type sent to the server. `type_id`, if provided, is sent as the
    `uuid` of the MM content part so vLLM uses it as the mm_hash.

    `media_mode` is "file" (default) or "base64" — see
    `_build_media_content` for details. `prompt_text`, if not None,
    overrides the default short prompt; pre-build it via
    `make_prompt_text(media_type, target_tokens)` to avoid recomputing
    it on every call.

    TTFT is measured as the time from POST dispatch to the first SSE
    chunk that carries a non-empty `delta.content` field. This excludes
    the leading role chunk (`{"delta": {"role": "assistant"}}`) that
    OpenAI-compatible servers usually send first, so the metric is a
    true Time-To-First-Token instead of Time-To-First-Chunk.
    """
    media_content = _build_media_content(
        image_path, media_type, type_id=type_id, media_mode=media_mode,
    )
    if prompt_text is None:
        prompt_text = make_prompt_text(media_type, target_tokens=0)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    media_content,
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    start_time = time.perf_counter()
    ttft = None
    output_tokens = 0
    server_request_id: str | None = None

    try:
        async with session.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            response.raise_for_status()
            # The server emits X-Request-Id when --enable-request-id-headers
            # is set on the prefill worker. Capture it so we can correlate
            # with VLLM_REQUEST_TIMING_TRACE log lines on the server side.
            server_request_id = (
                response.headers.get("X-Request-Id")
                or response.headers.get("x-request-id")
            )
            async for line in response.content:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue
                for part in decoded.split("\n"):
                    part = part.strip()
                    if not (part.startswith("data: ")
                            and part != "data: [DONE]"):
                        continue
                    try:
                        chunk = json.loads(part[6:])
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    # Only count chunks that carry actual generated
                    # text. This skips the leading {"role": "assistant"}
                    # chunk so TTFT measures Time-To-First-Token, not
                    # Time-To-First-Chunk.
                    if not content:
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    output_tokens += 1

        end_time = time.perf_counter()
        total_latency = end_time - start_time

        return {
            "ttft": ttft,
            "total_latency": total_latency,
            "output_tokens": output_tokens,
            "success": ttft is not None,
            "send_time": start_time,
            "complete_time": end_time,
            "server_request_id": server_request_id,
        }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "ttft": None,
            "total_latency": end_time - start_time,
            "output_tokens": 0,
            "success": False,
            "error": str(e),
            "send_time": start_time,
            "complete_time": end_time,
            "server_request_id": server_request_id,
        }


async def run_benchmark_open(
    workload: list[dict],
    server_url: str,
    model: str,
    qps: float,
    max_tokens: int = 20,
    media_mode: str = "file",
    prompt_tokens: int = 0,
) -> tuple[list[dict], float]:
    """Open-loop benchmark: dispatch requests at a fixed QPS regardless of
    downstream latency. Useful for measuring steady-state TTFT at a given
    load level, NOT for measuring max throughput.

    Returns (results, wall_clock_seconds).
    """
    results = []
    interval = 1.0 / qps if qps > 0 else 0

    # Pre-build prompt strings once per media_type so we don't recompute
    # the lorem-padded text on every request.
    prompt_cache = {
        mt: make_prompt_text(mt, target_tokens=prompt_tokens)
        for mt in {item.get("media_type", "image") for item in workload}
    }

    connector = aiohttp.TCPConnector(limit=1000)
    bench_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, item in enumerate(workload):
            if i > 0 and interval > 0:
                await asyncio.sleep(interval)

            mt = item.get("media_type", "image")
            task = asyncio.create_task(
                send_request(session, server_url, model,
                             item["image_path"], max_tokens,
                             media_type=mt,
                             type_id=item["type_id"],
                             media_mode=media_mode,
                             prompt_text=prompt_cache[mt])
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
    media_mode: str = "file",
    prompt_tokens: int = 0,
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

    # Pre-build prompt strings once per media_type.
    prompt_cache = {
        mt: make_prompt_text(mt, target_tokens=prompt_tokens)
        for mt in {item.get("media_type", "image") for item in workload}
    }

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

            mt = item.get("media_type", "image")
            result = await send_request(
                session, server_url, model,
                item["image_path"], max_tokens,
                media_type=mt,
                type_id=item["type_id"],
                media_mode=media_mode,
                prompt_text=prompt_cache[mt],
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

    # --- Three throughput metrics ---
    # 1. Effective throughput: successful / wall_clock (includes sleep gaps)
    throughput_eff = (len(successful) / wall_clock) if wall_clock > 0 else 0.0

    # 2. Offered throughput: successful / (last_complete - first_send)
    #    Measures actual server-facing window span.
    send_times = [r["send_time"] for r in successful if "send_time" in r]
    complete_times = [r["complete_time"] for r in successful
                      if "complete_time" in r]
    if send_times and complete_times:
        server_window = max(complete_times) - min(send_times)
        throughput_offered = (
            len(successful) / server_window if server_window > 0 else 0.0
        )
    else:
        server_window = wall_clock
        throughput_offered = throughput_eff

    # 3. Server capacity (Little's Law estimate):
    #    capacity ≈ concurrency / mean_latency
    #    Equivalent to: successful / sum(latency) — the average number of
    #    requests the server can handle per second if fully utilised.
    latencies_all = [r["total_latency"] for r in successful]
    if latencies_all:
        total_busy = sum(latencies_all)
        throughput_capacity = (
            len(successful) ** 2 / total_busy / len(successful)
            if total_busy > 0 else 0.0
        )
        # Simplifies to: len(successful) / total_busy * concurrency
        # But we don't know concurrency here, so use the simpler form:
        # capacity = 1 / mean_latency * avg_concurrency
        # avg_concurrency = total_busy / server_window
        avg_concurrency = total_busy / server_window if server_window > 0 else 1
        mean_latency = statistics.mean(latencies_all)
        throughput_capacity = (
            avg_concurrency / mean_latency if mean_latency > 0 else 0.0
        )
    else:
        throughput_capacity = 0.0

    if not successful:
        return {
            "total_requests": len(results),
            "measured_requests": len(measured),
            "warmup_requests": len(results) - len(measured),
            "successful": 0,
            "failed": len(failed),
            "wall_clock_s": wall_clock,
            "throughput_rps": throughput_eff,
            "throughput_offered_rps": 0.0,
            "throughput_capacity_rps": 0.0,
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
        "throughput_rps": throughput_eff,
        "throughput_offered_rps": throughput_offered,
        "throughput_capacity_rps": throughput_capacity,
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


def _trimmed_mean_std(vals: list[float],
                      trim: int = 0) -> tuple[float, float]:
    """Return (mean, std) of vals after dropping `trim` highest and `trim`
    lowest values. Falls back to full data if trimming would leave <2 values.
    """
    if not vals:
        return 0.0, 0.0
    if trim > 0 and len(vals) > 2 * trim + 1:
        sv = sorted(vals)
        kept = sv[trim:len(sv) - trim]
    else:
        kept = vals
    mean = statistics.mean(kept)
    std = statistics.stdev(kept) if len(kept) > 1 else 0.0
    return mean, std


def aggregate_rounds(round_metrics: list[dict], trim: int = 0) -> dict:
    """Aggregate metrics across multiple rounds into mean ± std.

    If trim > 0, drop the `trim` highest and `trim` lowest values for each
    metric before computing mean/std (trimmed mean — robust to outliers
    such as the inflated first-round TTFT).
    """
    keys_metric = (
        "throughput_rps", "throughput_offered_rps",
        "throughput_capacity_rps",
        "ttft_mean_ms", "ttft_median_ms",
        "ttft_p95_ms", "ttft_p99_ms",
        "latency_mean_ms", "latency_median_ms",
        "cache_hit_rate", "cache_hits", "cache_misses",
        # Per-round encoder forward stats (parsed from encoder log).
        "encoder_forward_mean_per_req_ms",
        "encoder_forward_mean_per_miss_ms",
        "encoder_forward_events", "encoder_forward_items",
    )
    keys_count = ("successful", "failed")

    if len(round_metrics) == 1:
        m = round_metrics[0]
        agg = {}
        for key in keys_metric:
            val = m.get(key)
            if val is not None:
                agg[key] = {"mean": val, "std": 0.0}
        for key in keys_count:
            agg[key] = {"mean": m.get(key, 0), "std": 0.0}
        return agg

    agg = {}
    for key in keys_metric:
        vals = [
            m[key] for m in round_metrics
            if key in m and m[key] is not None
        ]
        if vals:
            mean, std = _trimmed_mean_std(vals, trim=trim)
            agg[key] = {"mean": mean, "std": std}
    for key in keys_count:
        vals = [m.get(key, 0) for m in round_metrics]
        mean, std = _trimmed_mean_std(vals, trim=trim)
        agg[key] = {"mean": mean, "std": std}
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
        print(f"Throughput (eff):   {fmt_agg('throughput_rps')} req/s")
        print(f"Throughput (offer): {fmt_agg('throughput_offered_rps')} req/s")
        print(f"Throughput (cap):  {fmt_agg('throughput_capacity_rps')} req/s")
        if "cache_hit_rate" in aggregated:
            print(f"Cache hit rate:    {fmt_agg('cache_hit_rate')}")
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
        print(f"Throughput (eff):   {metrics.get('throughput_rps', 0):.2f} req/s")
        print(f"Throughput (offer): {metrics.get('throughput_offered_rps', 0):.2f} req/s")
        print(f"Throughput (cap):  {metrics.get('throughput_capacity_rps', 0):.2f} req/s")
        if "cache_hit_rate" in metrics:
            print(f"Cache hit rate:    "
                  f"{metrics.get('cache_hit_rate', 0):.4f}")
        # `metrics.get(key, default)` returns either a float or the
        # default ("N/A"); applying `:.2f` to the latter raises
        # ValueError. Format only when we have a numeric value.
        def _fmt_ms(key: str) -> str:
            v = metrics.get(key)
            if isinstance(v, (int, float)):
                return f"{v:.2f} ms"
            return "N/A"
        print(f"TTFT mean:         {_fmt_ms('ttft_mean_ms')}")
        print(f"TTFT median:       {_fmt_ms('ttft_median_ms')}")
        print(f"TTFT p95:          {_fmt_ms('ttft_p95_ms')}")
        print(f"TTFT p99:          {_fmt_ms('ttft_p99_ms')}")
        print(f"Latency mean:      {_fmt_ms('latency_mean_ms')}")

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
    parser.add_argument("--workload-mode", type=str, default="prob",
                        choices=["prob", "scan"],
                        help="'prob' = sample each request from "
                             "`distribution` with replacement (default). "
                             "'scan' = per-request Bernoulli(hot_fraction) "
                             "picks hot-vs-cold; hot picks are weighted "
                             "by distribution restricted to "
                             "cohort=='hot' types, cold picks are taken "
                             "WITHOUT replacement from cohort=='cold' "
                             "types. Use this with manifests produced "
                             "by generate_scan_workload.py.")
    parser.add_argument("--hot-fraction", type=float, default=None,
                        help="[scan mode] Probability that each request "
                             "is a hot pick. Default: sum of p_i over "
                             "hot types in the distribution.")
    parser.add_argument("--num-rounds", type=int, default=1,
                        help="Number of independent rounds to run. "
                        "Results are aggregated with mean ± std.")
    parser.add_argument("--fix-seed-across-rounds", action="store_true",
                        default=False,
                        help="Use the same seed for all rounds (isolate "
                        "hardware noise). Default: each round uses "
                        "seed + round_idx (captures ordering sensitivity).")
    parser.add_argument("--trim", type=int, default=0,
                        help="When aggregating across rounds, drop the N "
                        "highest and N lowest values for each metric "
                        "(trimmed mean). Recommended: 1 for >=5 rounds. "
                        "Default 0 (no trimming).")
    parser.add_argument("--encoder-url", type=str, default=None,
                        help="URL of the encoder worker (e.g. "
                        "http://localhost:19534). If provided, the encoder "
                        "cache is reset via POST /reset_encoder_cache "
                        "before each round (requires VLLM_SERVER_DEV_MODE=1 "
                        "on the encoder worker).")
    parser.add_argument("--encoder-log-path", type=str, default=None,
                        help="Path to the encoder worker's log file. When "
                        "set, after each round we trigger a reset (which "
                        "logs the round's hit/miss stats) and parse the "
                        "latest stats line into per-round metrics.")
    parser.add_argument("--global-warmup", type=int, default=20,
                        help="Number of requests to send before round 1 to "
                        "warm up CUDA kernels, cuDNN autotuning, and TCP "
                        "connections. These are fully discarded and do not "
                        "appear in any metrics. Set to 0 to skip.")
    parser.add_argument("--guarantee-each-type-warmup", action="store_true",
                        default=False,
                        help="Force the global warmup workload to include at "
                        "least one request per type in the manifest. Useful "
                        "when some types have very low p_i and would "
                        "otherwise be missed by random sampling, leaving "
                        "their kernels cold for the first measured round.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path for output results JSON")
    parser.add_argument("--label", type=str, default="",
                        help="Label for this benchmark run (e.g., 'fifo' or "
                        "'distribution_aware')")
    parser.add_argument("--media-mode", type=str, default="file",
                        choices=("file", "base64"),
                        help="How to send media references. 'file' (default) "
                        "uses file:// URLs so the server reads from "
                        "local disk via --allowed-local-media-path; "
                        "request bodies stay tiny so TTFT isn't "
                        "dominated by base64 upload + JSON parse. "
                        "'base64' inlines the file as a data URL "
                        "(needed when client and server don't share a "
                        "filesystem).")
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="Approximate target length of the text part of "
                        "the prompt, in tokens (~4 chars/token). 0 "
                        "(default) keeps the original short prompt. "
                        "Useful for studying how prefill cost scales "
                        "with prompt length, since longer prompts "
                        "increase TTFT and shift the relative weight "
                        "of encoder vs prefill.")
    args = parser.parse_args()

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    distribution = json.loads(args.distribution)

    # Normalize
    total_p = sum(distribution.values())
    if abs(total_p - 1.0) > 0.01:
        print(f"Normalizing distribution (sum={total_p})")
        distribution = {k: v / total_p for k, v in distribution.items()}

    # Global warmup: send a few throwaway requests to warm up CUDA kernels,
    # cuDNN autotuning, PyTorch memory allocator, and TCP connection pools.
    # Without this, the first round has ~5x higher TTFT than subsequent ones.
    if args.global_warmup > 0:
        if args.guarantee_each_type_warmup:
            warmup_wl = build_guaranteed_warmup_workload(
                manifest, distribution, args.global_warmup, seed=0,
            )
            print(f"\nGlobal warmup: sending {len(warmup_wl)} throwaway "
                  f"requests (covers each of {len(distribution)} types "
                  f"at least once)...")
        else:
            # Global warmup always uses prob mode so that warmup doesn't
            # consume cold types we need for the measured rounds.
            warmup_wl = generate_workload(
                manifest, distribution, args.global_warmup, seed=0,
                workload_mode="prob",
            )
            print(f"\nGlobal warmup: sending {len(warmup_wl)} throwaway "
                  f"requests to warm up CUDA / cuDNN / connections...")
        if args.mode == "closed":
            asyncio.run(run_benchmark_closed(
                warmup_wl, args.server_url, args.model,
                min(args.concurrency, len(warmup_wl)),
                warmup_requests=0, max_tokens=args.max_tokens,
                media_mode=args.media_mode,
                prompt_tokens=args.prompt_tokens,
            ))
        else:
            asyncio.run(run_benchmark_open(
                warmup_wl, args.server_url, args.model,
                args.qps, args.max_tokens,
                media_mode=args.media_mode,
                prompt_tokens=args.prompt_tokens,
            ))
        print("Global warmup done.\n")

    num_rounds = args.num_rounds
    per_round_metrics: list[dict] = []
    last_metrics: dict = {}
    last_results: list[dict] = []

    for round_idx in range(num_rounds):
        round_seed = (args.seed if args.fix_seed_across_rounds
                       else args.seed + round_idx)

        if num_rounds > 1:
            print(f"\n{'='*60}")
            print(f"Round {round_idx + 1}/{num_rounds} (seed={round_seed})")
            print(f"{'='*60}")

        # Reset encoder cache before round 1 so rounds always start cold.
        # For subsequent rounds the previous-round-end reset already cleared
        # the cache; resetting again is harmless but skipped to avoid
        # logging an empty stats line.
        if args.encoder_url and round_idx == 0:
            reset_encoder_cache(args.encoder_url)

        # Snapshot encoder log size so per-round encoder forward stats
        # only count events emitted during this round.
        encoder_log_offset_before_round = 0
        if args.encoder_log_path and os.path.exists(args.encoder_log_path):
            try:
                encoder_log_offset_before_round = os.path.getsize(
                    args.encoder_log_path
                )
            except OSError:
                encoder_log_offset_before_round = 0

        if args.mode == "closed":
            print(f"Generating workload: {args.num_requests} requests "
                  f"(warmup={args.warmup_requests}), "
                  f"concurrency={args.concurrency}, seed={round_seed}")
        else:
            print(f"Generating workload: {args.num_requests} requests, "
                  f"QPS={args.qps}, seed={round_seed}")

        workload = generate_workload(
            manifest, distribution, args.num_requests, round_seed,
            workload_mode=args.workload_mode,
            hot_fraction=args.hot_fraction,
        )
        if args.workload_mode == "scan":
            n_cold_in_workload = sum(
                1 for w in workload
                if (manifest.get(w["type_id"]) or {}).get("cohort", "hot")
                == "cold"
            )
            print(f"  scan mode: {n_cold_in_workload}/{len(workload)} "
                  f"requests are one-shot cold "
                  f"(target = {1 - (args.hot_fraction or 0.0):.2f} of "
                  f"requests if hot_fraction given; otherwise derived "
                  f"from distribution)")

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
                    media_mode=args.media_mode,
                    prompt_tokens=args.prompt_tokens,
                )
            )
        else:
            results, wall_clock = asyncio.run(
                run_benchmark_open(
                    workload, args.server_url, args.model,
                    args.qps, args.max_tokens,
                    media_mode=args.media_mode,
                    prompt_tokens=args.prompt_tokens,
                )
            )

        metrics = compute_metrics(results, wall_clock)

        # Capture per-round cache hit/miss stats by triggering a reset
        # (which causes the encoder worker to log this round's stats)
        # and then parsing the latest stats line from the worker log.
        # This also resets the cache for the next round.
        if args.encoder_url:
            reset_encoder_cache(args.encoder_url)
            if args.encoder_log_path:
                # Allow log buffer to flush
                time.sleep(0.3)
                stats = parse_latest_cache_stats(args.encoder_log_path)
                if stats is not None:
                    metrics.update(stats)
                    print(f"  Cache stats: hits={stats['cache_hits']} "
                          f"misses={stats['cache_misses']} "
                          f"hit_rate={stats['cache_hit_rate']:.4f}")
                else:
                    print("  Cache stats: (could not parse encoder log)")

                # Encoder forward time (this round only): scope by byte
                # offset captured at the start of the round.
                fwd_stats, _new_off = parse_encoder_forward_stats(
                    args.encoder_log_path,
                    after_byte_offset=encoder_log_offset_before_round,
                )
                if fwd_stats is not None:
                    n_meas = metrics.get("measured_requests", 0) or 1
                    # Two related views:
                    # 1) mean over all measured requests (hits == 0):
                    #    expected encoder cost per request given policy
                    # 2) mean over only the requests that ran encoder
                    #    (= forward time per miss, ~constant per type)
                    metrics["encoder_forward_mean_per_req_ms"] = (
                        fwd_stats["encoder_forward_total_ms"] / n_meas
                    )
                    metrics["encoder_forward_mean_per_miss_ms"] = (
                        fwd_stats["encoder_forward_mean_ms"]
                    )
                    metrics["encoder_forward_events"] = (
                        fwd_stats["encoder_forward_events"]
                    )
                    metrics["encoder_forward_items"] = (
                        fwd_stats["encoder_forward_items"]
                    )
                    print(f"  Encoder fwd: {fwd_stats['encoder_forward_events']} "
                          f"events, {fwd_stats['encoder_forward_items']} items, "
                          f"per_req={metrics['encoder_forward_mean_per_req_ms']:.2f}ms "
                          f"per_miss={metrics['encoder_forward_mean_per_miss_ms']:.2f}ms")

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
    aggregated = aggregate_rounds(per_round_metrics, trim=args.trim)
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
