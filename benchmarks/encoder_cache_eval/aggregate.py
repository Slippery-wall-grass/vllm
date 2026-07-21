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

# Display label for the swept axis. The filename slot stays `_rps<tag>` for
# back-compat, but in a closed-loop run the value is max-concurrency, not RPS;
# disagg_policy_compare.sh sets SWEEP_XLABEL=concurrency so tables/plots read
# correctly. Defaults to RPS for the open-loop sweep.
XLABEL = os.environ.get("SWEEP_XLABEL", "RPS")
XAXIS = "Request rate (RPS)" if XLABEL == "RPS" else "Max concurrency (in flight)"


_FILE_RX = re.compile(
    r"^(?P<policy>[a-zA-Z0-9_-]+)_rps(?P<rps>\d+(?:\.\d+)?)_rep(?P<rep>\d+)\.json$"
)


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

# Cumulative encoder-forward log line emitted by the worker when
# VLLM_TRACK_ENCODER_FORWARD_TIME=1 and the periodic logger is enabled:
#   "encoder_forward cumulative_secs=12.345 cumulative_forwards=42"
_ENC_FIELDS = {
    "cumulative_encoder_secs": r"cumulative_secs=([\d.eE+\-]+)",
    "cumulative_encoder_forwards": r"cumulative_forwards=(\d+)",
}

# Cumulative per-step stage timing line emitted by the worker when
# VLLM_TRACK_STEP_TIME=1 and the periodic logger is enabled. Mirrors the
# encoder_forward sidecar but adds prefill/decode cumulative seconds and
# step/token counts:
#   "stage_breakdown encoder_secs=X prefill_secs=Y decode_secs=Z
#    encoder_forwards=A prefill_steps=B decode_steps=C
#    prefill_tokens=D decode_tokens=E mixed_steps=F"
# prefill_secs/decode_secs are split per step in proportion to the prefill vs
# decode token counts processed that step (mixed chunked-prefill+decode steps
# contribute to both buckets). `mixed_steps` counts steps that carried both
# prefill and decode tokens (a subset of prefill_steps); it is absent in logs
# produced before that field was added, in which case the parser omits it.
_STAGE_FIELDS = {
    "stage_encoder_secs": r"encoder_secs=([\d.eE+\-]+)",
    "stage_prefill_secs": r"prefill_secs=([\d.eE+\-]+)",
    "stage_decode_secs": r"decode_secs=([\d.eE+\-]+)",
    "stage_encoder_forwards": r"encoder_forwards=(\d+)",
    "stage_prefill_steps": r"prefill_steps=(\d+)",
    "stage_decode_steps": r"decode_steps=(\d+)",
    "stage_prefill_tokens": r"prefill_tokens=(\d+)",
    "stage_decode_tokens": r"decode_tokens=(\d+)",
    "stage_mixed_steps": r"mixed_steps=(\d+)",
    # Cumulative request preemptions. >0 means prompts are being recomputed
    # under KV-cache pressure (extra prefill work). Absent in older logs.
    "stage_preempted_reqs": r"preempted_reqs=(\d+)",
}


def parse_encoder_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return {}
    out: dict[str, Any] = {}
    for key, pat in _ENC_FIELDS.items():
        m = re.search(pat, txt)
        if not m:
            continue
        raw = m.group(1)
        out[key] = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
    return out


def parse_stage_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return {}
    out: dict[str, Any] = {}
    for key, pat in _STAGE_FIELDS.items():
        m = re.search(pat, txt)
        if not m:
            continue
        raw = m.group(1)
        out[key] = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
    return out


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


def _fmt_rps(rps: float) -> str:
    """Display an RPS as int when it's integral (so '2' not '2.0'),
    otherwise with enough decimals to be distinct from neighbors."""
    if float(rps).is_integer():
        return str(int(rps))
    # Two decimals is enough for typical use (e.g. 1.5, 2.25); strip
    # trailing zeros for readability.
    s = f"{rps:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def collect_rows(result_dir: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, float], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cache_deltas: dict[tuple[str, float], dict[str, Any]] = defaultdict(dict)

    by_policy_rps: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)

    for f in sorted(result_dir.iterdir()):
        m = _FILE_RX.match(f.name)
        if not m:
            continue
        policy = m["policy"]
        rps = float(m["rps"])
        rep = int(m["rep"])
        bench = load_bench_json(f)
        cache_path = f.with_suffix(".cache.txt")
        enc_path = f.with_suffix(".enc.txt")
        stage_path = f.with_suffix(".stage.txt")
        cache_snapshot = parse_cache_snapshot(cache_path)
        enc_snapshot = parse_encoder_snapshot(enc_path)
        stage_snapshot = parse_stage_snapshot(stage_path)
        row = {
            "policy": policy,
            "rps": rps,
            "rep": rep,
            **bench,
            **{f"cache_{k}": v for k, v in cache_snapshot.items()},
            **enc_snapshot,
            **stage_snapshot,
        }
        rows.append(row)
        by_policy_rps[(policy, rps)].append(row)

    # Cache + encoder counters in the log are cumulative. Compute per
    # -(policy, RPS) deltas by subtracting the median of the *previous*
    # RPS step's snapshot from this step's median. (For the smallest
    # RPS we use the snapshot directly.)
    policies = sorted({p for p, _ in by_policy_rps})
    rps_levels = sorted({r for _, r in by_policy_rps})
    cumulative_keys = [
        "cache_hits",
        "cache_misses",
        "cache_forced_unpin_evictions",
        "cumulative_encoder_secs",
        "cumulative_encoder_forwards",
        "stage_encoder_secs",
        "stage_prefill_secs",
        "stage_decode_secs",
        "stage_encoder_forwards",
        "stage_prefill_steps",
        "stage_decode_steps",
        "stage_mixed_steps",
        "stage_prefill_tokens",
        "stage_decode_tokens",
        "stage_preempted_reqs",
    ]
    for policy in policies:
        prev = {k: 0 for k in cumulative_keys}
        for rps in rps_levels:
            bucket = by_policy_rps.get((policy, rps), [])
            if not bucket:
                continue
            cum: dict[str, float | None] = {}
            for key in cumulative_keys:
                cum[key] = median_safe([r.get(key) for r in bucket])
            delta: dict[str, float] = {}
            for key, val in cum.items():
                if val is None:
                    continue
                delta[f"delta_{key}"] = max(0.0, float(val) - float(prev.get(key, 0)))
                prev[key] = float(val)
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


def _expected_c_seconds(pool_spec_path: Path | None) -> float | None:
    """Compute E[c_i] = sum_i p_i * c_i from pool_spec.json.

    This is the average encoder-forward cost saved by a single cache
    hit (under the pool distribution; novelty hits contribute the same
    amount on average since novelty resolutions are drawn from the same
    bucket set as the pool).
    """
    if pool_spec_path is None or not pool_spec_path.exists():
        return None
    spec = json.load(open(pool_spec_path))
    types = spec.get("types", [])
    total = 0.0
    pool_total = 0.0
    for t in types:
        p = t.get("p")
        c = t.get("c_seconds")
        if p is None or c is None:
            continue
        total += float(p) * float(c)
        pool_total += float(p)
    if pool_total <= 0:
        return None
    return total / pool_total


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

    expected_c = _expected_c_seconds(pool_spec_path)

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
        header = f"| {XLABEL} | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [_fmt_rps(rps)]
            for pol in policies:
                vals = [r.get(key) for r in grouped[(pol, rps)]]
                cells.append(fmt(median_safe(vals)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    for key, label in derived:
        lines.append(f"## {label}")
        lines.append("")
        header = f"| {XLABEL} | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [_fmt_rps(rps)]
            for pol in policies:
                cells.append(fmt(cache_deltas.get((pol, rps), {}).get(key)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Measured encoder-forward wall-clock per RPS step (from the
    # worker-side cumulative counter). This is the ground-truth
    # equivalent of the analytical estimate below.
    has_measured = any(
        cache_deltas.get((pol, rps), {}).get("delta_cumulative_encoder_secs") is not None
        for pol in policies for rps in rps_levels
    )
    if has_measured:
        lines.append("## Encoder forward time MEASURED (s per RPS step)")
        lines.append("")
        header = f"| {XLABEL} | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [_fmt_rps(rps)]
            for pol in policies:
                v = cache_deltas.get((pol, rps), {}).get("delta_cumulative_encoder_secs")
                cells.append(f"{v:.2f}" if v is not None else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        lines.append(
            "## Encoder forward time MEASURED — saved vs nocache (s)"
        )
        lines.append("")
        header = f"| {XLABEL} | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [_fmt_rps(rps)]
            base = cache_deltas.get(("nocache", rps), {}).get(
                "delta_cumulative_encoder_secs"
            )
            for pol in policies:
                v = cache_deltas.get((pol, rps), {}).get(
                    "delta_cumulative_encoder_secs"
                )
                if v is None or base is None:
                    cells.append("—")
                else:
                    saved = float(base) - float(v)
                    pct = (
                        100.0 * saved / float(base) if base > 0 else 0.0
                    )
                    cells.append(f"{saved:.2f}s ({pct:+.1f}%)")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Encoder compute time saved per RPS step (seconds and as % of
    # pure no-cache baseline). Uses E[c_i] weighted by p_i.
    if expected_c is not None:
        lines.append(
            f"## Encoder compute saved (s per RPS step) — E[c_i] = "
            f"{expected_c * 1000:.1f} ms/hit"
        )
        lines.append("")
        header = f"| {XLABEL} | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [_fmt_rps(rps)]
            for pol in policies:
                hits = cache_deltas.get((pol, rps), {}).get("delta_cache_hits")
                if hits is None:
                    cells.append("—")
                else:
                    saved_s = float(hits) * expected_c
                    cells.append(f"{saved_s:.2f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # Per-policy savings RELATIVE to nocache and to fifo.
        lines.append("## Encoder compute saved vs nocache (s and %)")
        lines.append("")
        header = f"| {XLABEL} | " + " | ".join(policies) + " |"
        sep = "|" + "---|" * (len(policies) + 1)
        lines.append(header)
        lines.append(sep)
        for rps in rps_levels:
            cells = [_fmt_rps(rps)]
            base_hits = cache_deltas.get(("nocache", rps), {}).get("delta_cache_hits")
            for pol in policies:
                hits = cache_deltas.get((pol, rps), {}).get("delta_cache_hits")
                if hits is None or base_hits is None:
                    cells.append("—")
                else:
                    extra = (float(hits) - float(base_hits)) * expected_c
                    total_requests = sum(
                        r.get("completed", 0)
                        for r in grouped.get((pol, rps), [])
                    )
                    total_workload_compute = (
                        float(total_requests) * expected_c
                        if total_requests
                        else None
                    )
                    if total_workload_compute and total_workload_compute > 0:
                        pct = 100.0 * extra / total_workload_compute
                        cells.append(f"{extra:.2f}s ({pct:+.1f}%)")
                    else:
                        cells.append(f"{extra:.2f}s")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def maybe_plot(
    rows: list[dict[str, Any]],
    cache_deltas: dict[tuple[str, int], dict[str, Any]],
    plot_dir: Path,
    expected_c_seconds: float | None = None,
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
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["policy"], r["rps"])].append(r)

    # All panels (throughput AND latency) show the full RPS range so
    # the user can observe what happens when the server saturates.
    panels = [
        ("request_throughput", "Request throughput (req/s)"),
        ("output_throughput", "Output throughput (tok/s)"),
        ("mean_ttft_ms", "TTFT mean (ms)"),
        ("p50_ttft_ms", "TTFT p50 (ms)"),
        ("p99_ttft_ms", "TTFT p99 (ms)"),
        ("mean_tpot_ms", "TPOT mean (ms)"),
        ("p99_tpot_ms", "TPOT p99 (ms)"),
        ("p99_e2el_ms", "E2EL p99 (ms)"),
    ]
    for key, label in panels:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_plotted = False
        for pol in policies:
            ys = [
                median_safe([r.get(key) for r in grouped.get((pol, rps), [])])
                for rps in rps_levels
            ]
            xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
            if not xs_ys:
                continue
            xs, vals = zip(*xs_ys)
            ax.plot(xs, vals, marker="o", label=pol)
            any_plotted = True
        ax.set_xlabel(XAXIS)
        ax.set_ylabel(label)
        ax.set_title(label)
        if any_plotted:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{key}.png", dpi=120)
        plt.close(fig)

    # Hit rate (full range — independent of latency saturation).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pol in policies:
        ys = [
            cache_deltas.get((pol, rps), {}).get("delta_hit_rate")
            for rps in rps_levels
        ]
        xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
        if not xs_ys:
            continue
        xs, vals = zip(*xs_ys)
        ax.plot(xs, vals, marker="o", label=pol)
    ax.set_xlabel(XAXIS)
    ax.set_ylabel("Hit rate (per-step delta)")
    ax.set_title("Encoder cache hit rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "hit_rate.png", dpi=120)
    plt.close(fig)

    # Measured encoder forward time per RPS step (ground truth from
    # the worker's cumulative counter). Plotted only when the worker
    # was started with VLLM_TRACK_ENCODER_FORWARD_TIME=1.
    has_measured = any(
        cache_deltas.get((pol, rps), {}).get("delta_cumulative_encoder_secs")
        is not None
        for pol in policies
        for rps in rps_levels
    )
    if has_measured:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pol in policies:
            ys = [
                cache_deltas.get((pol, rps), {}).get("delta_cumulative_encoder_secs")
                for rps in rps_levels
            ]
            xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
            if not xs_ys:
                continue
            xs, vals = zip(*xs_ys)
            ax.plot(xs, vals, marker="o", label=pol)
        ax.set_xlabel(XAXIS)
        ax.set_ylabel("Encoder forward time (s, measured)")
        ax.set_title("Measured encoder forward time per RPS step")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "encoder_time_measured.png", dpi=120)
        plt.close(fig)

        # Stacked-bar stage breakdown per RPS, one figure per policy.
        # Bars show absolute seconds spent in encoder / prefill / decode
        # within each RPS window; percentages annotated on top so the
        # ratio is visible at a glance.
        for pol in policies:
            xs: list[float] = []
            enc_vals: list[float] = []
            pre_vals: list[float] = []
            dec_vals: list[float] = []
            for rps in rps_levels:
                d = cache_deltas.get((pol, rps), {})
                e = d.get("delta_stage_encoder_secs")
                p = d.get("delta_stage_prefill_secs")
                de = d.get("delta_stage_decode_secs")
                if e is None or p is None or de is None:
                    continue
                xs.append(float(rps))
                enc_vals.append(float(e))
                pre_vals.append(float(p))
                dec_vals.append(float(de))
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8))
            x_pos = list(range(len(xs)))
            bar_w = 0.6
            ax.bar(x_pos, enc_vals, bar_w, label="encoder",
                   color="#ff9f43")
            ax.bar(x_pos, pre_vals, bar_w, bottom=enc_vals,
                   label="prefill", color="#5f6caf")
            decode_bottom = [a + b for a, b in zip(enc_vals, pre_vals)]
            ax.bar(x_pos, dec_vals, bar_w, bottom=decode_bottom,
                   label="decode", color="#3ec1d3")
            # Annotate ratios on top of each stack.
            for i, (e, p, de) in enumerate(zip(enc_vals, pre_vals, dec_vals)):
                tot = e + p + de
                if tot <= 0:
                    continue
                ax.text(
                    i, tot,
                    f"{100*e/tot:.0f}/{100*p/tot:.0f}/{100*de/tot:.0f}%",
                    ha="center", va="bottom", fontsize=8,
                )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([_fmt_rps(x) for x in xs])
            ax.set_xlabel(XAXIS)
            ax.set_ylabel("Cumulative time per RPS window (s)")
            ax.set_title(
                f"Stage time breakdown — {pol}\n"
                "(enc/prefill/decode %)"
            )
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(plot_dir / f"stage_breakdown_{pol}.png", dpi=120)
            plt.close(fig)

        # Cross-policy ratio plot: stacked 100% bars showing stage share
        # for each policy at each RPS. One panel per RPS so the user can
        # compare fifo vs offline vs nocache stage mix at the same load.
        for rps in rps_levels:
            pol_xs: list[str] = []
            enc_pcts: list[float] = []
            pre_pcts: list[float] = []
            dec_pcts: list[float] = []
            for pol in policies:
                d = cache_deltas.get((pol, rps), {})
                e = d.get("delta_stage_encoder_secs")
                p = d.get("delta_stage_prefill_secs")
                de = d.get("delta_stage_decode_secs")
                if e is None or p is None or de is None:
                    continue
                tot = float(e) + float(p) + float(de)
                if tot <= 0:
                    continue
                pol_xs.append(pol)
                enc_pcts.append(100 * float(e) / tot)
                pre_pcts.append(100 * float(p) / tot)
                dec_pcts.append(100 * float(de) / tot)
            if not pol_xs:
                continue
            fig, ax = plt.subplots(figsize=(6, 4.2))
            x_pos = list(range(len(pol_xs)))
            ax.bar(x_pos, enc_pcts, 0.55, label="encoder",
                   color="#ff9f43")
            ax.bar(x_pos, pre_pcts, 0.55, bottom=enc_pcts,
                   label="prefill", color="#5f6caf")
            bot = [a + b for a, b in zip(enc_pcts, pre_pcts)]
            ax.bar(x_pos, dec_pcts, 0.55, bottom=bot,
                   label="decode", color="#3ec1d3")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(pol_xs)
            ax.set_ylabel("Share of stage time (%)")
            ax.set_ylim(0, 105)
            ax.set_title(
                f"Stage share by policy @ RPS={_fmt_rps(rps)}"
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(
                plot_dir / f"stage_share_rps{_fmt_rps(rps)}.png", dpi=120,
            )
            plt.close(fig)

        # Saved vs nocache (measured).
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pol in policies:
            ys = []
            for rps in rps_levels:
                v = cache_deltas.get((pol, rps), {}).get(
                    "delta_cumulative_encoder_secs"
                )
                base = cache_deltas.get(("nocache", rps), {}).get(
                    "delta_cumulative_encoder_secs"
                )
                if v is None or base is None:
                    ys.append(None)
                else:
                    ys.append(float(base) - float(v))
            xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
            if not xs_ys:
                continue
            xs, vals = zip(*xs_ys)
            ax.plot(xs, vals, marker="o", label=pol)
        ax.set_xlabel(XAXIS)
        ax.set_ylabel("Encoder time saved vs nocache (s)")
        ax.set_title("Measured encoder time saved (vs nocache baseline)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "encoder_time_saved_measured.png", dpi=120)
        plt.close(fig)

    # Encoder compute saved: per-step delta_cache_hits * E[c_i].
    if expected_c_seconds is not None and expected_c_seconds > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pol in policies:
            ys = []
            for rps in rps_levels:
                hits = cache_deltas.get((pol, rps), {}).get("delta_cache_hits")
                ys.append(
                    float(hits) * expected_c_seconds if hits is not None else None
                )
            xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
            if not xs_ys:
                continue
            xs, vals = zip(*xs_ys)
            ax.plot(xs, vals, marker="o", label=pol)
        ax.set_xlabel(XAXIS)
        ax.set_ylabel("Encoder compute saved (s)")
        ax.set_title(
            f"Encoder compute saved per RPS step  "
            f"(E[c_i] = {expected_c_seconds * 1000:.1f} ms/hit)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "encoder_time_saved.png", dpi=120)
        plt.close(fig)

        # Also normalize as "saved / requested total compute" to show
        # the *fraction* of encoder work eliminated.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pol in policies:
            ys = []
            for rps in rps_levels:
                hits = cache_deltas.get((pol, rps), {}).get("delta_cache_hits")
                completed = sum(
                    int(r.get("completed", 0) or 0)
                    for r in grouped.get((pol, rps), [])
                )
                if hits is None or completed <= 0:
                    ys.append(None)
                    continue
                # fraction = hits / total_requests = encoder work avoided
                ys.append(100.0 * float(hits) / float(completed))
            xs_ys = [(x, y) for x, y in zip(rps_levels, ys) if y is not None]
            if not xs_ys:
                continue
            xs, vals = zip(*xs_ys)
            ax.plot(xs, vals, marker="o", label=pol)
        ax.set_xlabel(XAXIS)
        ax.set_ylabel("Encoder forwards avoided (% of arrivals)")
        ax.set_title("Encoder compute saved (% of arrivals)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "encoder_time_saved_pct.png", dpi=120)
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
        expected_c = _expected_c_seconds(pool_path)
        maybe_plot(
            rows,
            deltas,
            Path(args.plot_dir),
            expected_c_seconds=expected_c,
        )
    print(f"Wrote {args.output_csv} and {args.output_md}")
    if args.plot_dir:
        print(f"Plots (if matplotlib available): {args.plot_dir}")


if __name__ == "__main__":
    main()
