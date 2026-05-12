# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot TTFT / hit-rate / throughput vs open-loop QPS for the 4 cache
strategies (none / fifo / fifo_persistent / dist_aware).

Reads the per-QPS subdirectories produced by sweep_qps.sh:

    $SWEEP_DIR/
        qps_1/results_<label>.json
        qps_2/results_<label>.json
        ...

Produces a 4-panel figure:
    * Mean TTFT (ms) vs QPS
    * P95 TTFT (ms) vs QPS
    * Cache hit rate (%) vs QPS
    * Effective throughput (req/s) vs QPS

Use it to see how each policy degrades (or stays flat) as offered
load increases. In a well-scaled system, TTFT should stay roughly
constant up to the server's capacity, then knee upward. The cache
policies should differ most in the regime where the encoder is the
bottleneck — at very low QPS or oversaturated QPS they all converge.

Usage:
    python plot_qps_sweep.py --sweep-dir /tmp/qps_sweep
"""

import argparse
import json
import re
from pathlib import Path

STRATS = [
    ("none", "No cache", "tab:gray", "o"),
    ("fifo", "LRU + EC clean", "tab:blue", "s"),
    ("fifo_persistent", "LRU + EC persist", "tab:cyan", "D"),
    ("dist_aware", "OUR", "tab:red", "^"),
]


def _load_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _extract(d: dict | None, key: str) -> float | None:
    """Pull `key` from `aggregated[key].mean` or fall back to
    `metrics[key]`. Returns None when the run failed or the key is
    absent.
    """
    if not d:
        return None
    agg = d.get("aggregated") or {}
    if key in agg and isinstance(agg[key], dict):
        v = agg[key].get("mean")
        if v is not None:
            return float(v)
    metrics = d.get("metrics") or {}
    if key in metrics:
        try:
            return float(metrics[key])
        except (TypeError, ValueError):
            return None
    return None


def _discover(sweep_dir: Path) -> list[tuple[float, Path]]:
    """Return sorted list of (qps, subdir) tuples found under sweep_dir.
    Recognises subdirs named `qps_<float>` (also accepts ints).
    """
    pattern = re.compile(r"^qps_([\d.]+)$")
    found: list[tuple[float, Path]] = []
    for child in sweep_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        try:
            qps = float(m.group(1))
        except ValueError:
            continue
        found.append((qps, child))
    found.sort(key=lambda p: p[0])
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep-dir", type=str, required=True,
                        help="Directory containing qps_<value>/ "
                             "subdirectories (matches sweep_qps.sh's "
                             "$SWEEP_DIR)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output PNG path (default: "
                             "<sweep-dir>/qps_sweep.png)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--annotate", action="store_true",
                        help="Annotate each point with its numeric "
                             "value")
    parser.add_argument("--errorbars", action="store_true",
                        help="Draw mean ± std error bars (uses "
                             "aggregated std across rounds)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    sweep_dir = Path(args.sweep_dir)
    points = _discover(sweep_dir)
    if not points:
        raise SystemExit(
            f"No qps_<value>/ subdirs found under {sweep_dir}")

    print(f"Discovered {len(points)} sweep points:")
    for qps, sub in points:
        files = sorted(p.name for p in sub.glob("results_*.json"))
        print(f"  qps={qps:>5}  {sub.name}  ({len(files)} files)")

    # Build (qps, strategy) -> metric tables.
    table_ttft_mean: dict[str, list[tuple[float, float, float]]] = {
        k: [] for k, *_ in STRATS}
    table_ttft_p95: dict[str, list[tuple[float, float, float]]] = {
        k: [] for k, *_ in STRATS}
    # Mean encoder forward time (the GPU work cache hits save).
    # Endpoints: model.embed_multimodal() entry → return.
    # "per_req" averages over ALL measured requests (hits = 0); "per_miss"
    # averages only over requests that actually ran encoder forward.
    table_enc_per_req: dict[str, list[tuple[float, float, float]]] = {
        k: [] for k, *_ in STRATS}
    table_enc_per_miss: dict[str, list[tuple[float, float, float]]] = {
        k: [] for k, *_ in STRATS}

    def _push(table, key, qps, d, metric_key):
        v = _extract(d, metric_key)
        if v is None:
            return
        # Std (only present when num_rounds > 1)
        agg = (d or {}).get("aggregated") or {}
        std = 0.0
        if metric_key in agg and isinstance(agg[metric_key], dict):
            std = float(agg[metric_key].get("std") or 0.0)
        table[key].append((qps, v, std))

    for qps, sub in points:
        for key, *_ in STRATS:
            d = _load_results(sub / f"results_{key}.json")
            _push(table_ttft_mean, key, qps, d, "ttft_mean_ms")
            _push(table_ttft_p95, key, qps, d, "ttft_p95_ms")
            _push(table_enc_per_req, key, qps, d,
                  "encoder_forward_mean_per_req_ms")
            _push(table_enc_per_miss, key, qps, d,
                  "encoder_forward_mean_per_miss_ms")

    # 2x2 layout: TTFT mean / P95 / encoder-per-req / encoder-per-miss
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_mean, ax_p95 = axes[0]
    ax_enc_req, ax_enc_miss = axes[1]

    def _plot(ax, table, ylabel, title, ylim=None):
        any_data = False
        for key, label, color, marker in STRATS:
            data = table.get(key) or []
            if not data:
                continue
            any_data = True
            xs = [p[0] for p in data]
            ys = [p[1] for p in data]
            errs = [p[2] for p in data] if args.errorbars else None
            if args.errorbars and any(e > 0 for e in errs or []):
                ax.errorbar(xs, ys, yerr=errs, marker=marker, color=color,
                            linewidth=2.0, markersize=7, capsize=3,
                            label=label)
            else:
                ax.plot(xs, ys, marker=marker, color=color, linewidth=2.0,
                        markersize=8, label=label)
            if args.annotate:
                for x, y in zip(xs, ys):
                    ax.annotate(f"{y:.0f}", (x, y),
                                textcoords="offset points",
                                xytext=(4, 4), fontsize=8, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if any_data:
            ax.legend(loc="best", fontsize=8)
        return any_data

    _plot(ax_mean, table_ttft_mean, "Mean TTFT (ms)",
          "Mean TTFT vs offered QPS")
    _plot(ax_p95, table_ttft_p95, "P95 TTFT (ms)",
          "P95 TTFT vs offered QPS")
    _plot(ax_enc_req, table_enc_per_req,
          "Encoder forward (ms / request)",
          "Mean encoder GPU work per request "
          "(model.embed_multimodal entry→return; hit=0)")
    _plot(ax_enc_miss, table_enc_per_miss,
          "Encoder forward (ms / miss)",
          "Mean encoder GPU work per miss "
          "(only requests that actually ran the encoder)")

    for ax in (ax_enc_req, ax_enc_miss):
        ax.set_xlabel("Offered QPS")

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out = args.output or str(sweep_dir / "qps_sweep.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    # Numerical summary
    def _fmt(v: float | None, prec: int) -> str:
        return f"{v:.{prec}f}" if v is not None else "-"

    print("\nNumerical summary (TTFT mean / encoder fwd per request, ms):")
    header = (f"{'qps':>5}  "
              f"{'none':>22}  {'LRU+clean':>22}  "
              f"{'LRU+persist':>22}  {'OUR':>22}")
    print(header)
    print("-" * len(header))
    for qps, _ in points:
        row = {}
        for key, *_ in STRATS:
            t = dict((q, v) for q, v, _s in table_ttft_mean[key]).get(qps)
            e = dict((q, v) for q, v, _s in table_enc_per_req[key]).get(qps)
            row[key] = (t, e)
        cells = []
        for key, *_ in STRATS:
            t, e = row[key]
            cells.append(
                f"ttft={_fmt(t, 0):>5} "
                f"enc/req={_fmt(e, 0):>5}"
            )
        print(f"{qps:>5.1f}  " + "  ".join(f"{c:>22}" for c in cells))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
