# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Aggregator for ``run_sweep.sh`` outputs.

Reads per-run benchmark JSONs and the sidecar cache-stats snapshots,
emits a long-form CSV, a markdown summary table per metric, and (when
matplotlib is installed) line plots of each metric vs RPS, grouped by
policy.

Designed to be safe to re-run as new files appear in ``--result-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


_FILE_RX = re.compile(r"^(?P<policy>[a-zA-Z0-9_-]+)_rps(?P<rps>\d+)_rep(?P<rep>\d+)\.json$")


# Cumulative cache stats line printed by EncoderCacheManager.maybe_log_stats():
# "encoder_cache policy=Foo hits=N misses=M hit_rate=R forced_unpin=F pinned=P
#  freeable=Q free_slots=S"
_CACHE_FIELDS = {
    "hits": r"hits=(\d+)",
    "misses": r"misses=(\d+)",
    "hit_rate": r"hit_rate=([\d.eE+\-]+)",
    "forced_unpin_evictions": r"forced_unpin=(\d+)",
    "num_pinned": r"pinned=(\d+)",
    "num_freeable": r"freeable=(\d+)",
}


def parse_cache_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return {}
    out: dict[str, Any] = {}
    for key, pat in _CACHE_FIELDS.items():
        m = re.search(pat, txt)
        if not m:
            continue
        out[key] = float(m.group(1)) if "." in m.group(1) or "e" in m.group(1).lower() else int(m.group(1))
    return out


_BENCH_METRICS = [
    "completed",
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
]


def load_bench_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: data.get(k) for k in _BENCH_METRICS}


def median_safe(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else None


def collect_rows(result_dir: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cache_deltas: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)

    by_policy_rps: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for f in sorted(result_dir.iterdir()):
        m = _FILE_RX.match(f.name)
        if not m:
            continue
        policy = m["policy"]
        rps = int(m["rps"])
        rep = int(m["rep"])
        bench = load_bench_json(f)
        cache_path = f.with_suffix(".cache.txt")
        if cache_path.name.endswith(".cache.txt"):
            # `with_suffix(".cache.txt")` replaces only the final suffix,
            # which is what we want.
            pass
        cache_snapshot = parse_cache_snapshot(cache_path)
        row = {
            "policy": policy,
            "rps": rps,
            "rep": rep,
            **bench,
            **{f"cache_{k}": v for k, v in cache_snapshot.items()},
        }
        rows.append(row)
        by_policy_rps[(policy, rps)].append(row)

    # Cache counters in the log are cumulative. Compute per-(policy, RPS) deltas
    # by subtracting the median of the *previous* RPS step's snapshot from
    # this step's median. (For the smallest RPS we use the snapshot directly.)
    policies = sorted({p for p, _ in by_policy_rps})
    rps_levels = sorted({r for _, r in by_policy_rps})
    for policy in policies:
        prev = {"cache_hits": 0, "cache_misses": 0, "cache_forced_unpin_evictions": 0}
        for rps in rps_levels:
            bucket = by_policy_rps.get((policy, rps), [])
            if not bucket:
                continue
            cum = {}
            for key in ["cache_hits", "cache_misses", "cache_forced_unpin_evictions"]:
                cum[key] = median_safe([r.get(key) for r in bucket])
            delta = {}
            for key, val in cum.items():
                if val is None:
                    continue
                delta[f"delta_{key}"] = max(0.0, val - prev.get(key, 0))
                prev[key] = val
            # Derived hit rate for this RPS step.
            dh = delta.get("delta_cache_hits")
            dm = delta.get("delta_cache_misses")
            if dh is not None and dm is not None and (dh + dm) > 0:
                delta["delta_hit_rate"] = dh / (dh + dm)
            cache_deltas[(policy, rps)] = delta

    return rows, cache_deltas


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(
    rows: list[dict[str, Any]],
    cache_deltas: dict[tuple[str, int], dict[str, Any]],
    pool_spec_path: Path | None,
    out_path: Path,
) -> None:
    policies = sorted({r["policy"] for r in rows})
    rps_levels = sorted({r["rps"] for r in rows})

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["policy"], r["rps"])].append(r)

    metrics = [
        ("request_throughput", "Request throughput (req/s)"),
        ("output_throughput", "Output throughput (tok/s)"),
        ("mean_ttft_ms", "TTFT mean (ms)"),
        ("p50_ttft_ms", "TTFT p50 (ms)"),
        ("p90_ttft_ms", "TTFT p90 (ms)"),
        ("p99_ttft_ms", "TTFT p99 (ms)"),
        ("mean_tpot_ms", "TPOT mean (ms)"),
        ("p99_tpot_ms", "TPOT p99 (ms)"),
        ("p99_itl_ms", "ITL p99 (ms)"),
        ("p99_e2el_ms", "E2EL p99 (ms)"),
    ]

    derived = [
        ("delta_hit_rate", "Cache hit rate (per RPS step)"),
        ("delta_cache_forced_unpin_evictions", "Forced-unpin evictions (per step)"),
    ]

    lines: list[str] = []
    lines.append("# Encoder-cache policy sweep")
    lines.append("")
    if pool_spec_path and pool_spec_path.exists():
        spec = json.load(open(pool_spec_path))
        dist = spec.get("distribution", {})
        types = spec.get("types", [])
        ms = [t.get("m_tokens") for t in types if t.get("m_tokens")]
        cs = [t.get("c_seconds") for t in types if t.get("c_seconds")]
        lines.append(
            f"- Pool: K={len(types)} distribution={dist.get('kind')}"
            f" param={dist.get('param')}"
        )
        if ms:
            lines.append(
                f"- m_tokens: min={min(ms)} median={int(statistics.median(ms))}"
                f" max={max(ms)} sum={sum(ms)}"
            )
        if cs:
            lines.append(
                f"- c_seconds: min={min(cs):.4f} median={statistics.median(cs):.4f}"
                f" max={max(cs):.4f}"
            )
        lines.append("")

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    for key, label in metrics:
        lines.append(f"## {label}")
        lines.append("")
        header = "| RPS | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [str(rps)]
            for pol in policies:
                vals = [r.get(key) for r in grouped[(pol, rps)]]
                cells.append(fmt(median_safe(vals)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    for key, label in derived:
        lines.append(f"## {label}")
        lines.append("")
        header = "| RPS | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [str(rps)]
            for pol in policies:
                cells.append(fmt(cache_deltas.get((pol, rps), {}).get(key)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def maybe_plot(
    rows: list[dict[str, Any]],
    cache_deltas: dict[tuple[str, int], dict[str, Any]],
    plot_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    policies = sorted({r["policy"] for r in rows})
    rps_levels = sorted({r["rps"] for r in rows})
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["policy"], r["rps"])].append(r)

    panels = [
        ("request_throughput", "Request throughput (req/s)"),
        ("output_throughput", "Output throughput (tok/s)"),
        ("mean_ttft_ms", "TTFT mean (ms)"),
        ("p99_ttft_ms", "TTFT p99 (ms)"),
        ("p99_e2el_ms", "E2EL p99 (ms)"),
    ]

    for key, label in panels:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pol in policies:
            ys = [median_safe([r.get(key) for r in grouped.get((pol, rps), [])]) for rps in rps_levels]
            xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
            if not xs_ys:
                continue
            xs, vals = zip(*xs_ys)
            ax.plot(xs, vals, marker="o", label=pol)
        ax.set_xlabel("Request rate (RPS)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{key}.png", dpi=120)
        plt.close(fig)

    # Hit rate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pol in policies:
        ys = [cache_deltas.get((pol, rps), {}).get("delta_hit_rate") for rps in rps_levels]
        xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
        if not xs_ys:
            continue
        xs, vals = zip(*xs_ys)
        ax.plot(xs, vals, marker="o", label=pol)
    ax.set_xlabel("Request rate (RPS)")
    ax.set_ylabel("Hit rate (per-step delta)")
    ax.set_title("Encoder cache hit rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "hit_rate.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--pool-spec", default=None)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--plot-dir", default=None)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    rows, deltas = collect_rows(result_dir)
    if not rows:
        print(f"No benchmark JSONs found in {result_dir}")
        return
    write_csv(rows, Path(args.output_csv))
    pool_path = Path(args.pool_spec) if args.pool_spec else None
    write_markdown(rows, deltas, pool_path, Path(args.output_md))
    if args.plot_dir:
        maybe_plot(rows, deltas, Path(args.plot_dir))
    print(f"Wrote {args.output_csv} and {args.output_md}")
    if args.plot_dir:
        print(f"Plots (if matplotlib available): {args.plot_dir}")


if __name__ == "__main__":
    main()
