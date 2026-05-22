# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pool builder for the encoder-cache Lagrangian policy.

This script prepares the artifacts consumed by

- :class:`vllm.benchmarks.datasets.MMFixedPoolDataset` (the
  ``mm-fixed-pool`` benchmark dataset), and
- :class:`vllm.v1.core.encoder_cache_policy.OfflineLagrangianEncoderCachePolicy`
  (the cache-side pinning policy).

Three subcommands are provided:

1. ``generate``: create K canonical PNG images with stable byte
   contents, compute their mm_hashes, and write ``pool_spec.json``.
2. ``measure``: probe a running vLLM OpenAI-compatible server, send one
   request per type, and record the encoder-side compute time per
   type (``c_i``). Results are written back into ``pool_spec.json``.
3. ``solve``: read the populated ``pool_spec.json``, solve the dual
   ``max_lambda D(lambda)`` for the configured cache capacity, and
   emit ``mm_pool.json`` consumed by the cache policy.

All three subcommands share ``--pool-dir``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

# Allow running from a vllm checkout without installation.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vllm.multimodal.hasher import MultiModalHasher  # noqa: E402
from vllm.multimodal.image import convert_image_mode  # noqa: E402
from vllm.v1.core.encoder_cache_lambda import (  # noqa: E402
    TypeStats,
    dual_value,
    solve_lambda_star,
)


# 4 buckets per user specification (resolution-independent of distribution).
DEFAULT_BUCKETS: list[tuple[int, int]] = [
    (360, 640),
    (720, 1280),
    (1080, 1920),
    (1440, 2560),
]


# ---------------------------------------------------------------------------
# Image generation + canonical hashing
# ---------------------------------------------------------------------------


def _generate_image(
    height: int,
    width: int,
    seed: int,
) -> bytes:
    """Generate a deterministic PNG byte string for a given (size, seed)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    buf = io.BytesIO()
    # optimize=False keeps the encoder cheap and reproducible across PIL
    # versions; pnginfo is omitted so we never embed timestamps/exif.
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _hash_png_bytes(
    png_bytes: bytes,
    model_id: str | None = None,
    hf_processor_mm_kwargs: dict | None = None,
) -> str:
    """Hash the *decoded* image the way vLLM does server-side.

    vLLM's input pipeline decodes the data URL into a ``PIL.Image`` and
    hashes via
    :meth:`MultiModalHasher.hash_kwargs(model_id=..., image=...,
    **hf_processor_mm_kwargs)`` (see
    ``vllm/multimodal/processing/inputs.py``).

    ``model_id`` MUST match the value vLLM uses (typically the
    ``--model`` path/name passed to ``vllm serve``); otherwise the
    offline-computed hash will not collide with the server-computed
    one and ``Offline`` / ``Oracle`` policies will see every image as
    unknown.
    """
    img = Image.open(io.BytesIO(png_bytes))
    img.load()
    img = convert_image_mode(img, "RGB")
    kwargs: dict[str, object] = {"image": img}
    if model_id is not None:
        kwargs["model_id"] = model_id
    if hf_processor_mm_kwargs:
        kwargs.update(hf_processor_mm_kwargs)
    return MultiModalHasher.hash_kwargs(**kwargs)


# ---------------------------------------------------------------------------
# Distribution shapes
# ---------------------------------------------------------------------------


def _build_distribution(
    kind: str,
    k: int,
    param: float | None,
    custom_probs: list[float] | None,
    seed: int,
) -> list[float]:
    """Return a length-K probability vector summing to 1."""
    if kind == "uniform":
        return [1.0 / k] * k
    if kind == "zipf":
        s = float(param) if param is not None else 1.1
        if s <= 0.0:
            raise ValueError("zipf parameter must be > 0")
        weights = np.array([1.0 / ((i + 1) ** s) for i in range(k)], dtype=np.float64)
        return (weights / weights.sum()).tolist()
    if kind == "geom":
        # Geometric-like decay over the K bins. ``param`` is the decay
        # ratio in (0, 1); larger -> flatter.
        r = float(param) if param is not None else 0.7
        if not (0.0 < r < 1.0):
            raise ValueError("geom parameter must be in (0, 1)")
        weights = np.array([r**i for i in range(k)], dtype=np.float64)
        return (weights / weights.sum()).tolist()
    if kind == "dirichlet":
        # Random skewed distribution from a symmetric Dirichlet. ``param``
        # is the concentration alpha. Useful to get a fresh shape for
        # repeated experiments without committing to a parametric family.
        alpha = float(param) if param is not None else 0.5
        rng = np.random.default_rng(seed)
        sample = rng.dirichlet([alpha] * k)
        # Sort descending so type 0 is most frequent, matching zipf/geom.
        sample = np.sort(sample)[::-1]
        return sample.tolist()
    if kind == "custom":
        if not custom_probs:
            raise ValueError("--distribution-probs is required for kind=custom")
        if len(custom_probs) != k:
            raise ValueError(
                f"custom probs has {len(custom_probs)} entries, expected K={k}"
            )
        total = sum(custom_probs)
        if total <= 0:
            raise ValueError("custom probs must sum to a positive value")
        return [p / total for p in custom_probs]
    raise ValueError(f"Unknown distribution kind: {kind}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.bucket_config:
        buckets = [tuple(map(int, b.split("x"))) for b in args.bucket_config]
    else:
        buckets = DEFAULT_BUCKETS

    pool_dir = args.pool_dir
    os.makedirs(os.path.join(pool_dir, "images"), exist_ok=True)

    probs = _build_distribution(
        kind=args.distribution,
        k=args.k,
        param=args.distribution_param,
        custom_probs=args.distribution_probs,
        seed=args.seed,
    )

    # Bucket assignment: cycle through buckets so resolution and frequency
    # are independent (user's stated preference).
    bucket_rng = random.Random(args.seed + 1)
    bucket_assignments = [buckets[bucket_rng.randrange(len(buckets))] for _ in range(args.k)]

    if not args.model_id:
        print(
            "WARNING: --model-id not provided. The offline mm_hash will not "
            "match the server-side hash, and Offline/Oracle policies will see "
            "every pool image as 'unknown'. Pass --model-id with the same "
            "value you use for `vllm serve <model>`.",
            file=sys.stderr,
        )

    types: list[dict[str, Any]] = []
    for i in range(args.k):
        h, w = bucket_assignments[i]
        png_bytes = _generate_image(h, w, seed=args.seed * 100003 + i)
        filename = f"{i:04d}.png"
        with open(os.path.join(pool_dir, "images", filename), "wb") as f:
            f.write(png_bytes)
        mm_hash = _hash_png_bytes(png_bytes, model_id=args.model_id)
        types.append(
            {
                "idx": i,
                "height": h,
                "width": w,
                "filename": filename,
                "p": float(probs[i]),
                "mm_hash": mm_hash,
                # m_tokens / c_seconds populated by `measure` later.
                "m_tokens": None,
                "c_seconds": None,
            }
        )

    spec = {
        "version": 1,
        "seed": args.seed,
        "model_id": args.model_id,   # remembered for downstream re-hashing
        "distribution": {
            "kind": args.distribution,
            "param": args.distribution_param,
            "probs": args.distribution_probs,
        },
        "images_dir": "images",
        "types": types,
    }
    with open(os.path.join(pool_dir, "pool_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(
        f"Wrote {args.k} images and pool_spec.json to {pool_dir}.\n"
        f"  buckets={sorted(set(bucket_assignments))}\n"
        f"  distribution={args.distribution} (p_min={min(probs):.4f},"
        f" p_max={max(probs):.4f})"
    )


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def _png_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def _send_chat_image(
    base_url: str,
    api_key: str | None,
    model: str,
    png_bytes: bytes,
    output_tokens: int,
    timeout: float,
) -> float:
    """Send a single chat-completions request with the image and return
    the response's total wall-clock latency. We approximate ``c_i`` with
    TTFT after warmup so it excludes decode time."""
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe in one word."},
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_to_data_url(png_bytes)},
                    },
                ],
            }
        ],
        "max_tokens": output_tokens,
        "stream": True,
        "temperature": 0.0,
        "stream_options": {"include_usage": True},
        "extra_body": {"ignore_eos": True},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    t0 = time.perf_counter()
    first_token_time = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            if line == "[DONE]":
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            # We only care about TTFT; consume the rest to drain the stream.
            del evt
    if first_token_time is None:
        first_token_time = time.perf_counter()
    return first_token_time - t0


def _query_num_image_tokens(
    base_url: str,
    api_key: str | None,
    model: str,
    png_bytes: bytes,
    timeout: float,
) -> int | None:
    """Best-effort: query ``/tokenize`` or use ``usage.prompt_tokens`` after
    a tiny chat call to infer the image token count. Returns ``None`` if
    the server does not expose enough information; caller falls back to
    a heuristic estimate based on pixel count.
    """
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "."},
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_to_data_url(png_bytes)},
                    },
                ],
            }
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "extra_body": {"ignore_eos": False},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    usage = payload.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        return None
    # The image token count is approximately (prompt_tokens - text overhead).
    # The text content above is intentionally tiny; this is a best-effort.
    return int(prompt_tokens) - 4


def _server_wakeup(args: argparse.Namespace, spec: dict) -> None:
    """Force the vision tower / vLLM CUDA graphs to finish first-time
    initialization before we start timing requests.

    ``/health`` returns 200 as soon as the LLM core is alive, but the
    vision tower is lazily loaded on the first multimodal request, which
    can take well over the default 120s socket timeout. We send one
    request using the smallest image in the pool with a very generous
    timeout so the rest of the measurement loop can run with the normal
    timeout.
    """
    images_dir = os.path.join(args.pool_dir, spec.get("images_dir", "images"))
    # Pick the smallest image by m_tokens if available, else by area.
    types = list(spec["types"])
    def _size(t):
        return (t.get("m_tokens") or (t["height"] * t["width"]))
    smallest = min(types, key=_size)
    with open(os.path.join(images_dir, smallest["filename"]), "rb") as f:
        png = f.read()
    wakeup_timeout = max(args.timeout, args.wakeup_timeout)
    print(
        f"wakeup: sending one request to warm vision tower / CUDA graphs "
        f"(image={smallest['filename']}, timeout={wakeup_timeout}s)",
        flush=True,
    )
    t0 = time.perf_counter()
    # Run the wake-up request on a worker thread so the main thread can
    # emit periodic progress lines. Without this the script appears
    # frozen for up to wakeup_timeout seconds, which makes it impossible
    # to tell "still warming" from "actually stuck".
    import threading

    result_box: dict[str, object] = {}

    def _do_request() -> None:
        try:
            _send_chat_image(
                args.base_url,
                args.api_key,
                args.model,
                png,
                output_tokens=1,
                timeout=wakeup_timeout,
            )
            result_box["ok"] = True
        except Exception as exc:  # noqa: BLE001
            result_box["err"] = exc

    th = threading.Thread(target=_do_request, daemon=True)
    th.start()
    while th.is_alive():
        th.join(timeout=15.0)
        if th.is_alive():
            print(
                f"wakeup: still waiting ({time.perf_counter() - t0:.0f}s "
                f"elapsed of {wakeup_timeout:.0f}s)…",
                flush=True,
            )
    if "err" in result_box:
        raise result_box["err"]  # type: ignore[misc]
    print(f"wakeup: done in {time.perf_counter() - t0:.1f}s", flush=True)


def cmd_measure(args: argparse.Namespace) -> None:
    spec_path = os.path.join(args.pool_dir, "pool_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    _server_wakeup(args, spec)

    images_dir = os.path.join(args.pool_dir, spec.get("images_dir", "images"))
    # Warmup once per type before timing so we don't include compile/cuda
    # graph capture in the latency.
    for t in spec["types"]:
        with open(os.path.join(images_dir, t["filename"]), "rb") as f:
            png = f.read()
        # Warmup
        for _ in range(max(1, args.warmup)):
            _send_chat_image(
                args.base_url, args.api_key, args.model, png,
                output_tokens=1, timeout=args.timeout,
            )
        # Time
        samples: list[float] = []
        for _ in range(args.repeats):
            lat = _send_chat_image(
                args.base_url, args.api_key, args.model, png,
                output_tokens=1, timeout=args.timeout,
            )
            samples.append(lat)
        t["c_seconds"] = float(statistics.median(samples))
        # m_tokens
        if args.use_token_query:
            m = _query_num_image_tokens(
                args.base_url, args.api_key, args.model, png, args.timeout
            )
        else:
            m = None
        if m is None:
            # Heuristic: ~588 tokens per megapixel for Qwen2-VL-like models.
            m = max(1, int(round(t["height"] * t["width"] / 1e6 * 588)))
        t["m_tokens"] = int(m)
        print(
            f"type {t['idx']:3d} {t['height']}x{t['width']}: "
            f"c={t['c_seconds']:.4f}s  m={t['m_tokens']} tokens"
        )

    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"Updated {spec_path}")


# ---------------------------------------------------------------------------
# prewarm
# ---------------------------------------------------------------------------


def cmd_prewarm(args: argparse.Namespace) -> None:
    """Send one cheap request per pool image type to force-encode it
    server-side. This populates the encoder cache and exercises every
    CUDA graph / vision-tower code path before the measurement phase,
    so the bench window is not contaminated by per-shape lazy init.

    Submission is concurrent (ThreadPoolExecutor) to keep total wall
    time short: K=100 images at concurrency=4 typically finishes in
    5-10 seconds vs ~30 seconds sequential. Order is shuffled so the
    FIFO eviction tail is randomized rather than always keeping the
    last-by-filename types.

    By default the encoder cache is flushed via the server's
    ``/reset_encoder_cache`` dev endpoint after prewarm so the
    measurement phase starts from a clean cache state while still
    benefiting from preserved CUDA graphs and lazy init. Disable with
    ``--no-reset-after`` (kept for debugging).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    spec_path = os.path.join(args.pool_dir, "pool_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    images_dir = os.path.join(args.pool_dir, spec.get("images_dir", "images"))

    _server_wakeup(args, spec)

    payloads: list[tuple[int, bytes]] = []
    for t in spec["types"]:
        with open(os.path.join(images_dir, t["filename"]), "rb") as f:
            payloads.append((int(t["idx"]), f.read()))

    rng = random.Random(args.seed)
    rng.shuffle(payloads)

    def worker(item: tuple[int, bytes]) -> tuple[int, float, str | None]:
        idx, png = item
        t0 = time.perf_counter()
        try:
            _send_chat_image(
                args.base_url,
                args.api_key,
                args.model,
                png,
                output_tokens=1,
                timeout=args.timeout,
            )
            return idx, time.perf_counter() - t0, None
        except Exception as exc:  # noqa: BLE001
            return idx, time.perf_counter() - t0, str(exc)

    done = 0
    failed = 0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, item) for item in payloads]
        for fut in as_completed(futures):
            idx, lat, err = fut.result()
            done += 1
            if err is not None:
                failed += 1
                print(
                    f"  prewarm type={idx:3d} FAILED ({lat:.2f}s): {err[:120]}",
                    file=sys.stderr,
                )
    elapsed = time.perf_counter() - t_start
    print(
        f"prewarm: {done - failed}/{len(payloads)} succeeded in {elapsed:.1f}s "
        f"(concurrency={args.concurrency})"
    )
    if failed > 0 and failed >= max(1, len(payloads) // 5):
        # If more than 20% fail something is wrong; surface non-zero exit so
        # run_sweep.sh aborts early instead of running a bad experiment.
        raise SystemExit(
            f"prewarm: {failed}/{len(payloads)} requests failed — aborting."
        )

    if args.reset_after:
        _reset_encoder_cache_endpoint(args.base_url, args.api_key, args.timeout)


def _reset_encoder_cache_endpoint(
    base_url: str,
    api_key: str | None,
    timeout: float,
) -> None:
    """POST /reset_encoder_cache. Requires VLLM_SERVER_DEV_MODE=1 on the
    server, otherwise the route is not registered and returns 404."""
    import urllib.request

    url = base_url.rstrip("/") + "/reset_encoder_cache"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=b"", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                print(
                    "prewarm: encoder cache reset OK "
                    "(CUDA graphs and lazy init preserved)"
                )
                return
            raise RuntimeError(f"unexpected status {resp.status}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                "WARNING: /reset_encoder_cache returned 404. "
                "Start the server with VLLM_SERVER_DEV_MODE=1 to enable "
                "the dev routes, or pass --no-reset-after to skip.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: encoder-cache reset failed: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def cmd_solve(args: argparse.Namespace) -> None:
    spec_path = os.path.join(args.pool_dir, "pool_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    types_raw = spec["types"]
    types = [
        TypeStats(
            p=float(t["p"]),
            m=float(t["m_tokens"]),
            c=float(t["c_seconds"]),
        )
        for t in types_raw
        if t.get("m_tokens") is not None and t.get("c_seconds") is not None
    ]
    if len(types) != len(types_raw):
        raise ValueError(
            "pool_spec.json has types without measured m_tokens/c_seconds. "
            "Run the 'measure' subcommand first."
        )
    lam, dual_opt = solve_lambda_star(types, args.cache_capacity)
    print(
        f"solved lambda*={lam:.6g}, D(lambda*)={dual_opt:.6g} "
        f"(capacity={args.cache_capacity}, K={len(types)})"
    )
    if args.dump_dual_curve:
        # Helpful sanity-check dump: D(lambda) on a log grid.
        if lam > 0:
            grid = [lam * (2.0 ** k) for k in range(-6, 7)]
        else:
            grid = [10.0 ** k for k in range(-10, 0)]
        for g in grid:
            print(f"  lambda={g:.6g}  D={dual_value(types, args.cache_capacity, g):.6g}")

    entries: dict[str, dict[str, Any]] = {}
    for t_raw, t in zip(types_raw, types):
        entries[t_raw["mm_hash"]] = {
            "type_idx": int(t_raw["idx"]),
            "p": t.p,
            "m": t.m,
            "c": t.c,
        }
    out_path = os.path.join(args.pool_dir, "mm_pool.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "lambda_star": lam,
                "cache_capacity": args.cache_capacity,
                "entries": entries,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        prog="precompute_mm_pool",
        description="Generate / measure / solve a fixed-pool encoder-cache config.",
    )
    p.add_argument("--pool-dir", required=True, help="Working directory for the pool.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate", help="Create K PNGs and pool_spec.json.")
    pg.add_argument("--k", type=int, default=100, help="Pool size.")
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument(
        "--distribution",
        choices=["uniform", "zipf", "geom", "dirichlet", "custom"],
        default="zipf",
    )
    pg.add_argument(
        "--distribution-param",
        type=float,
        default=None,
        help="zipf s / geom ratio / dirichlet alpha.",
    )
    pg.add_argument(
        "--distribution-probs",
        type=float,
        nargs="+",
        default=None,
        help="Custom probability vector (length K) for --distribution custom.",
    )
    pg.add_argument(
        "--bucket-config",
        nargs="+",
        default=None,
        help='Resolution buckets as "HxW" tokens, e.g. 360x640 720x1280. '
        "Default: 360x640 720x1280 1080x1920 1440x2560.",
    )
    pg.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model identifier used by vLLM when hashing multimodal inputs "
        "(see vllm/multimodal/processing/inputs.py). MUST match the value "
        "passed to `vllm serve` (typically the model path or HF model name) "
        "or the Offline / Oracle policies will see every pool image as "
        "'unknown'.",
    )
    pg.set_defaults(func=cmd_generate)

    pm = sub.add_parser(
        "measure",
        help="Probe a running vLLM server to fill in m_tokens and c_seconds.",
    )
    pm.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    pm.add_argument("--model", required=True)
    pm.add_argument("--api-key", default=None)
    pm.add_argument("--warmup", type=int, default=2)
    pm.add_argument("--repeats", type=int, default=5)
    pm.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds after the wake-up phase.",
    )
    pm.add_argument(
        "--wakeup-timeout",
        type=float,
        default=1200.0,
        help="HTTP timeout for the first vision-tower wake-up request. "
        "First call after server startup includes CUDA graph capture and "
        "lazy module loading, which routinely exceeds the normal timeout.",
    )
    pm.add_argument(
        "--use-token-query",
        action="store_true",
        help="Ask the server for prompt_tokens to infer m_tokens. "
        "When omitted, fall back to a pixel-count heuristic.",
    )
    pm.set_defaults(func=cmd_measure)

    pw = sub.add_parser(
        "prewarm",
        help="Send one chat-completion request per pool image to force the "
        "vision encoder through every type. Useful right before the bench "
        "phase so CUDA graph capture and lazy module loading do not "
        "contaminate the measurement window.",
    )
    pw.add_argument("--base-url", required=True)
    pw.add_argument("--model", required=True)
    pw.add_argument("--api-key", default=None)
    pw.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Parallel in-flight requests. 4 is a safe default for one A100.",
    )
    pw.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds after the wake-up phase.",
    )
    pw.add_argument(
        "--wakeup-timeout",
        type=float,
        default=1200.0,
        help="HTTP timeout for the first wake-up request that primes the "
        "vision tower / CUDA graphs.",
    )
    pw.add_argument("--seed", type=int, default=0)
    pw.add_argument(
        "--reset-after",
        dest="reset_after",
        action="store_true",
        default=True,
        help="POST /reset_encoder_cache after prewarm so the measurement "
        "phase starts from a clean cache (default).",
    )
    pw.add_argument(
        "--no-reset-after",
        dest="reset_after",
        action="store_false",
        help="Skip the post-prewarm cache reset.",
    )
    pw.set_defaults(func=cmd_prewarm)

    ps = sub.add_parser("solve", help="Solve lambda* and emit mm_pool.json.")
    ps.add_argument(
        "--cache-capacity",
        type=float,
        required=True,
        help="Encoder cache capacity in encoder embedding tokens (B).",
    )
    ps.add_argument("--dump-dual-curve", action="store_true")
    ps.set_defaults(func=cmd_solve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
