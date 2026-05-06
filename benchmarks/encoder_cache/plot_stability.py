# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot per-request TTFT and (approximate) hit-rate trajectories so you can
see when the cache reaches steady state.

The x-axis is the **request dispatch index** within a round, taken from
`raw_results[*].request_id` in each strategy's results JSON. (Only the
last round's raw results are persisted by run_benchmark.py, but that's
sufficient: every round starts with a cold cache, so one round captures
the cold-start → steady-state shape.)

Two panels:
  * Top: per-request TTFT (light scatter) + rolling mean (window W),
    one color per strategy. A vertical band marks the warmup phase.
  * Bottom: cumulative "ideal" hit rate, defined as the fraction of
    requests up to index i whose `type_id` already appeared earlier in
    the same round. This is a workload property — it's the upper bound
    on hit rate any policy can achieve (and it's tight when the cache
    fits all distinct types seen so far). The line is the same for
    every strategy by construction; we plot it once.

Usage:
    python plot_stability.py --work-dir /tmp/vmmu_data
    # or:
    python plot_stability.py \
        --none /path/results_none.json \
        --fifo /path/results_fifo.json \
        --dist /path/results_dist_aware.json \
        --window 50 \
        --output /tmp/stability.png
"""

import argparse
import json
from pathlib import Path

STRATS = [
    ("none", "No cache", "tab:gray", "o"),
    ("fifo", "FIFO", "tab:blue", "s"),
    ("dist_aware", "Distribution-Aware", "tab:red", "^"),
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  missing: {path}")
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  corrupt JSON {path}: {e}")
        return None


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.work_dir:
        wd = Path(args.work_dir)
        return {
            "none": wd / "results_none.json",
            "fifo": wd / "results_fifo.json",
            "dist_aware": wd / "results_dist_aware.json",
        }
    paths = {}
    if args.none:
        paths["none"] = Path(args.none)
    if args.fifo:
        paths["fifo"] = Path(args.fifo)
    if args.dist:
        paths["dist_aware"] = Path(args.dist)
    if not paths:
        raise SystemExit(
            "Pass --work-dir or at least one of --none/--fifo/--dist")
    return paths


def _rolling_mean(xs: list[float], window: int) -> list[float | None]:
    """Right-aligned rolling mean. Result has same length as xs; positions
    < window-1 are None so the line starts at index window-1.
    """
    out: list[float | None] = []
    s = 0.0
    q: list[float] = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q) if len(q) >= window else None)
    return out


def _ideal_hitrate_curve(type_ids: list[str]) -> list[float]:
    """Fraction of requests in prefix [0..i] whose type_id has appeared
    earlier in the same prefix. Upper bound on any cache policy's
    cumulative hit rate (tight when the cache fits all distinct types
    seen so far).
    """
    seen: set[str] = set()
    hits = 0
    out: list[float] = []
    for i, t in enumerate(type_ids):
        if t in seen:
            hits += 1
        else:
            seen.add(t)
        out.append(hits / (i + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory with results_{none,fifo,"
                             "dist_aware}.json (matches "
                             "run_cache_comparison.sh's $WORK_DIR).")
    parser.add_argument("--none", type=str, default=None)
    parser.add_argument("--fifo", type=str, default=None)
    parser.add_argument("--dist", type=str, default=None)
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling-mean window width in requests "
                             "(default: 50). Bigger → smoother but "
                             "lags steady state more.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output PNG path (default: "
                             "<work-dir>/stability.png or "
                             "./stability.png)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--ttft-log", action="store_true",
                        help="Use log scale on the TTFT axis")
    parser.add_argument("--include-warmup", action="store_true",
                        help="Plot warmup requests too (default: "
                             "warmup is shaded but excluded from "
                             "rolling-mean line so steady-state is "
                             "easier to see)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    paths = _resolve_paths(args)
    print("Loading:")
    raw_per_strat: dict[str, list[dict]] = {}
    config_per_strat: dict[str, dict] = {}
    for key, p in paths.items():
        d = _load(p)
        if d is None:
            continue
        raw = d.get("raw_results") or []
        if not raw:
            print(f"  {key}: no raw_results in {p}")
            continue
        config_per_strat[key] = d.get("config", {})
        # Sort by request_id (closed-loop appends in completion order, not
        # dispatch order)
        raw.sort(key=lambda r: r.get("request_id", 0))
        raw_per_strat[key] = raw
        print(f"  {key:<10} {len(raw)} requests  ({p})")
    if not raw_per_strat:
        raise SystemExit("No usable raw_results; nothing to plot.")

    # Warmup count: assume the same across strategies; pick from any
    warmup_count = 0
    for raw in raw_per_strat.values():
        warmup_count = sum(1 for r in raw if r.get("phase") == "warmup")
        break

    fig, (ax_ttft, ax_hit) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True)

    max_n = 0
    for key, label, color, marker in STRATS:
        raw = raw_per_strat.get(key)
        if raw is None:
            continue

        ids = [r.get("request_id", i) for i, r in enumerate(raw)]
        ttfts_s = [r.get("ttft") for r in raw]
        # Convert to ms; replace failed (None) with NaN-like skip
        valid_idx = [i for i, t in enumerate(ttfts_s) if t is not None]
        if not valid_idx:
            print(f"  {key}: no successful TTFTs; skipping")
            continue
        x_all = [ids[i] for i in valid_idx]
        y_all = [ttfts_s[i] * 1000.0 for i in valid_idx]
        max_n = max(max_n, max(x_all) + 1 if x_all else 0)

        # Scatter (light)
        ax_ttft.scatter(x_all, y_all, s=8, color=color, alpha=0.18,
                        edgecolors="none")

        # Rolling mean — exclude warmup unless asked
        if args.include_warmup:
            x_for_mean = x_all
            y_for_mean = y_all
        else:
            x_for_mean = [x for x in x_all if x >= warmup_count]
            y_for_mean = [y for x, y in zip(x_all, y_all)
                          if x >= warmup_count]
        if len(y_for_mean) >= args.window:
            rm = _rolling_mean(y_for_mean, args.window)
            xs_line = [x for x, v in zip(x_for_mean, rm) if v is not None]
            ys_line = [v for v in rm if v is not None]
            ax_ttft.plot(xs_line, ys_line, color=color, linewidth=2.0,
                         label=f"{label} (rolling mean, w={args.window})")
        else:
            ax_ttft.plot([], [], color=color, marker=marker,
                         label=f"{label} (only {len(y_for_mean)} samples)")

    if warmup_count > 0:
        for ax in (ax_ttft, ax_hit):
            ax.axvspan(0, warmup_count, color="orange", alpha=0.10,
                       label="_nolegend_")
        ax_ttft.text(warmup_count / 2, ax_ttft.get_ylim()[1] * 0.95,
                     "warmup", ha="center", va="top", fontsize=9,
                     color="darkorange")

    ax_ttft.set_ylabel("TTFT (ms)")
    ax_ttft.set_title("Per-request TTFT — last round, sorted by dispatch index")
    if args.ttft_log:
        ax_ttft.set_yscale("log")
    ax_ttft.grid(True, alpha=0.3)
    ax_ttft.legend(loc="best", fontsize=9)

    # Bottom: ideal cumulative hit rate (workload-determined; same for any
    # strategy when cache is large enough to hold all distinct types).
    # Use any one strategy's type_id sequence (they all see the same
    # workload modulo seed if --fix-seed-across-rounds was used; otherwise
    # we plot each strategy's own curve).
    plotted_hit = False
    for key, label, color, marker in STRATS:
        raw = raw_per_strat.get(key)
        if raw is None:
            continue
        type_ids = [r.get("type_id", "") for r in raw]
        ideal = _ideal_hitrate_curve(type_ids)
        xs_h = [r.get("request_id", i) for i, r in enumerate(raw)]
        # Hit rate is undefined for "no cache" — but the workload curve is
        # still informative as a reference, so we still draw the no-cache
        # one as the workload baseline. To avoid duplicate near-identical
        # lines, only plot once if seeds were fixed.
        ax_hit.plot(xs_h, [v * 100 for v in ideal], color=color,
                    linewidth=1.6, alpha=0.85,
                    label=f"{label} workload upper bound")
        plotted_hit = True

    ax_hit.set_xlabel("Request index (dispatch order, within last round)")
    ax_hit.set_ylabel("Cumulative ideal hit rate (%)")
    ax_hit.set_title("Cumulative upper-bound hit rate "
                     "(workload property — assumes infinite cache)")
    ax_hit.set_ylim(0, 100)
    ax_hit.grid(True, alpha=0.3)
    if plotted_hit:
        ax_hit.legend(loc="best", fontsize=9)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out = args.output
    if out is None:
        if args.work_dir:
            out = str(Path(args.work_dir) / "stability.png")
        else:
            out = "stability.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    # Quick text report: where each strategy's rolling-mean TTFT enters a
    # ±5% band around its tail mean.
    print("\nStabilization estimate (when rolling-mean TTFT enters ±5% of "
          "its final value, within the measure phase):")
    for key, label, _, _ in STRATS:
        raw = raw_per_strat.get(key)
        if raw is None:
            continue
        ttfts = [r["ttft"] * 1000.0 for r in raw
                 if r.get("ttft") is not None
                 and r.get("phase", "measure") != "warmup"]
        if len(ttfts) < args.window * 2:
            continue
        rm = _rolling_mean(ttfts, args.window)
        rm_clean = [v for v in rm if v is not None]
        if len(rm_clean) < 2:
            continue
        tail = rm_clean[-min(50, len(rm_clean)):]
        tail_mean = sum(tail) / len(tail)
        band = 0.05 * tail_mean
        # Find smallest k such that all rm_clean[k:] are within ±5% of tail_mean
        stable_at: int | None = None
        for k in range(len(rm_clean)):
            if all(abs(v - tail_mean) <= band for v in rm_clean[k:]):
                stable_at = k + args.window  # add window offset
                break
        if stable_at is None:
            print(f"  {label:<22} never enters ±5% band "
                  f"(tail mean={tail_mean:.0f}ms)")
        else:
            print(f"  {label:<22} stabilises near request "
                  f"#{warmup_count + stable_at} "
                  f"(tail mean={tail_mean:.0f}ms)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
