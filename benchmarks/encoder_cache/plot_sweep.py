# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot TTFT and hit-rate vs encoder cache size from a sweep.

Reads the per-size subdirectories produced by sweep_cache_size.sh:

    $SWEEP_DIR/
        cache_8192/results_{none,fifo,dist_aware}.json
        cache_16384/results_{none,fifo,dist_aware}.json
        ...

Produces three panels in one figure:
    * Mean TTFT (ms) vs cache size, one line per strategy
    * P95 TTFT (ms) vs cache size
    * Cache hit rate (%) vs cache size (skipping "none" which has no cache)

Use it to identify the cache-pressure "sweet spot" where
Distribution-Aware separates from FIFO. At very small sizes everything
collapses to no-cache; at very large sizes everything ties at full-hit.
The interesting region is in between.

Usage:
    python plot_sweep.py --sweep-dir /tmp/cache_sweep -o /tmp/sweep.png
"""

import argparse
import json
import re
from pathlib import Path

STRATS = [
    ("none", "No cache", "tab:gray", "o"),
    ("fifo", "FIFO + EC clean", "tab:blue", "s"),
    ("fifo_persistent", "FIFO + EC persist", "tab:cyan", "D"),
    ("dist_aware", "Distribution-Aware", "tab:red", "^"),
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


def _discover(sweep_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (cache_size, subdir) tuples found under
    sweep_dir. Recognises subdirs named `cache_<int>` or `cache=<int>`.
    """
    pattern = re.compile(r"^cache[_=](\d+)$")
    found: list[tuple[int, Path]] = []
    for child in sweep_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        found.append((int(m.group(1)), child))
    found.sort(key=lambda p: p[0])
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep-dir", type=str, required=True,
                        help="Directory containing cache_<size>/ "
                             "subdirectories (matches "
                             "sweep_cache_size.sh's $SWEEP_DIR)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output PNG path (default: "
                             "<sweep-dir>/sweep.png)")
    parser.add_argument("--metric", type=str, default="ttft_mean_ms",
                        choices=("ttft_mean_ms", "ttft_median_ms",
                                 "ttft_p95_ms", "ttft_p99_ms",
                                 "throughput_rps", "throughput_offered_rps",
                                 "latency_mean_ms"),
                        help="Primary metric for the top panel "
                             "(default: ttft_mean_ms)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--log-x", action="store_true",
                        help="Use log scale on the cache-size axis")
    parser.add_argument("--annotate", action="store_true",
                        help="Annotate each point with its numeric "
                             "value")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    sweep_dir = Path(args.sweep_dir)
    points = _discover(sweep_dir)
    if not points:
        raise SystemExit(
            f"No cache_<size>/ subdirs found under {sweep_dir}")

    print(f"Discovered {len(points)} sweep points:")
    for size, sub in points:
        files = sorted(p.name for p in sub.glob("results_*.json"))
        print(f"  cache_size={size:>6}  {sub.name}  ({len(files)} files)")

    # Build (size, strategy) -> metric tables.
    # Use the user-chosen primary metric for the top panel, p95 for
    # mid panel, and cache_hit_rate for bottom panel.
    table_primary: dict[str, list[tuple[int, float]]] = {
        k: [] for k, *_ in STRATS}
    table_p95: dict[str, list[tuple[int, float]]] = {
        k: [] for k, *_ in STRATS}
    table_hit: dict[str, list[tuple[int, float]]] = {
        k: [] for k, *_ in STRATS}

    for size, sub in points:
        for key, *_ in STRATS:
            d = _load_results(sub / f"results_{key}.json")
            v = _extract(d, args.metric)
            if v is not None:
                table_primary[key].append((size, v))
            v95 = _extract(d, "ttft_p95_ms")
            if v95 is not None:
                table_p95[key].append((size, v95))
            hr = _extract(d, "cache_hit_rate")
            if hr is not None:
                # hit_rate stored as a fraction in [0,1]
                table_hit[key].append((size, hr * 100.0))

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(10, 11), sharex=True)

    def _plot(ax, table, ylabel, title, ylim=None):
        any_data = False
        for key, label, color, marker in STRATS:
            data = table.get(key) or []
            if not data:
                continue
            any_data = True
            xs = [p[0] for p in data]
            ys = [p[1] for p in data]
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
        if args.log_x:
            ax.set_xscale("log", base=2)
        if any_data:
            ax.legend(loc="best", fontsize=9)
        return any_data

    pretty = {
        "ttft_mean_ms": "Mean TTFT (ms)",
        "ttft_median_ms": "Median TTFT (ms)",
        "ttft_p95_ms": "P95 TTFT (ms)",
        "ttft_p99_ms": "P99 TTFT (ms)",
        "throughput_rps": "Throughput (req/s)",
        "throughput_offered_rps": "Offered throughput (req/s)",
        "latency_mean_ms": "Mean latency (ms)",
    }
    primary_label = pretty.get(args.metric, args.metric)

    _plot(ax_top, table_primary, primary_label,
          f"{primary_label} vs encoder cache size")
    _plot(ax_mid, table_p95, "P95 TTFT (ms)",
          "P95 TTFT vs encoder cache size")
    _plot(ax_bot, table_hit, "Cache hit rate (%)",
          "Cache hit rate vs encoder cache size", ylim=(0, 100))

    ax_bot.set_xlabel("Encoder cache size (tokens, "
                      "= --max-num-batched-tokens)")

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out = args.output or str(sweep_dir / "sweep.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    # Numerical takeaway
    print("\nNumerical summary (lower TTFT / higher hit-rate is better):")
    print(f"{'cache_size':>10}  "
          f"{'none_ttft':>10}  {'fifo_ttft':>10}  {'dist_ttft':>10}  "
          f"{'fifo_hr':>8}  {'dist_hr':>8}  {'gain':>7}")
    def _fmt(v: float | None, prec: int) -> str:
        return f"{v:.{prec}f}" if v is not None else "-"

    for size, _ in points:
        row_t = {k: dict(table_primary[k]).get(size) for k, *_ in STRATS}
        row_h = {k: dict(table_hit[k]).get(size) for k, *_ in STRATS}
        gain = ""
        if row_t["fifo"] is not None and row_t["dist_aware"] is not None:
            gain = f"{row_t['fifo'] - row_t['dist_aware']:+.0f}ms"
        print(f"{size:>10}  "
              f"{_fmt(row_t['none'], 0):>10}  "
              f"{_fmt(row_t['fifo'], 0):>10}  "
              f"{_fmt(row_t['dist_aware'], 0):>10}  "
              f"{_fmt(row_h['fifo'], 1):>8}  "
              f"{_fmt(row_h['dist_aware'], 1):>8}  "
              f"{gain:>7}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
