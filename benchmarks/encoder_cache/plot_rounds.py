# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot per-round hit-rate and TTFT for the three cache strategies.

Reads the three result JSONs produced by run_cache_comparison.sh:
    $WORK_DIR/results_none.json
    $WORK_DIR/results_fifo.json
    $WORK_DIR/results_dist_aware.json

Each JSON has a `per_round` list; each entry contains at least
`cache_hit_rate` and `ttft_mean_ms`. We plot two subplots: hit-rate vs
round and mean TTFT vs round, with one line per strategy.

Usage:
    python plot_rounds.py --work-dir /tmp/vmmu_data
    # or pass files explicitly:
    python plot_rounds.py \
        --none /tmp/.../results_none.json \
        --fifo /tmp/.../results_fifo.json \
        --dist /tmp/.../results_dist_aware.json \
        --output /tmp/rounds.png

Dependencies: matplotlib (`pip install matplotlib`).
"""

import argparse
import json
from pathlib import Path

# Three strategies, in fixed display order.
STRATS = [
    ("none", "No cache", "tab:gray", "o"),
    ("fifo", "LRU + EC clean", "tab:blue", "s"),
    ("fifo_persistent", "LRU + EC persist", "tab:cyan", "D"),
    ("dist_aware", "OUR", "tab:red", "^"),
]


def _load(path: Path) -> list[dict] | None:
    """Return the per_round list from a results file, or None if missing."""
    if not path.exists():
        print(f"  missing: {path}")
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  corrupt JSON {path}: {e}")
        return None
    rounds = data.get("per_round") or []
    if not rounds:
        print(f"  no per_round entries in {path}")
        return None
    return rounds


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory containing results_{none,fifo,"
                             "dist_aware}.json (matches "
                             "run_cache_comparison.sh's $WORK_DIR).")
    parser.add_argument("--none", type=str, default=None,
                        help="Path to results_none.json")
    parser.add_argument("--fifo", type=str, default=None,
                        help="Path to results_fifo.json")
    parser.add_argument("--dist", type=str, default=None,
                        help="Path to results_dist_aware.json")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output PNG path (default: "
                             "<work-dir>/rounds.png or ./rounds.png)")
    parser.add_argument("--show", action="store_true",
                        help="Open the plot interactively after saving")
    parser.add_argument("--title", type=str, default=None,
                        help="Optional supertitle for the figure")
    args = parser.parse_args()

    # Lazy import so --help works without matplotlib installed.
    import matplotlib.pyplot as plt

    paths = _resolve_paths(args)
    print("Loading:")
    series: dict[str, list[dict]] = {}
    for key, p in paths.items():
        rounds = _load(p)
        if rounds is not None:
            series[key] = rounds
            print(f"  {key:<10} {len(rounds)} rounds  ({p})")
    if not series:
        raise SystemExit("No usable result files; nothing to plot.")

    # X-axis = 1..max_rounds across whichever strategies we have.
    max_rounds = max(len(v) for v in series.values())
    xs = list(range(1, max_rounds + 1))

    fig, (ax_hit, ax_ttft) = plt.subplots(
        1, 2, figsize=(12, 4.5), sharex=True)

    for key, label, color, marker in STRATS:
        rounds = series.get(key)
        if not rounds:
            continue

        # Hit rate: skip "none" (no cache → undefined / always 0).
        # `cache_hit_rate` is None when the encoder log wasn't parsed.
        hit_xs, hit_ys = [], []
        ttft_xs, ttft_ys = [], []
        for i, r in enumerate(rounds, start=1):
            hr = r.get("cache_hit_rate")
            if hr is not None and key != "none":
                hit_xs.append(i)
                hit_ys.append(hr * 100.0)  # percent
            t = r.get("ttft_mean_ms")
            if t is not None:
                ttft_xs.append(i)
                ttft_ys.append(t)

        if hit_ys:
            ax_hit.plot(hit_xs, hit_ys, marker=marker, color=color,
                        label=label, linewidth=2)
        if ttft_ys:
            ax_ttft.plot(ttft_xs, ttft_ys, marker=marker, color=color,
                         label=label, linewidth=2)

    ax_hit.set_title("Encoder cache hit rate per round")
    ax_hit.set_xlabel("Round")
    ax_hit.set_ylabel("Hit rate (%)")
    ax_hit.set_ylim(0, 100)
    ax_hit.set_xticks(xs)
    ax_hit.grid(True, alpha=0.3)
    ax_hit.legend(loc="best")

    ax_ttft.set_title("Mean TTFT per round")
    ax_ttft.set_xlabel("Round")
    ax_ttft.set_ylabel("TTFT mean (ms)")
    ax_ttft.set_xticks(xs)
    ax_ttft.grid(True, alpha=0.3)
    ax_ttft.legend(loc="best")

    if args.title:
        fig.suptitle(args.title)

    fig.tight_layout()

    out = args.output
    if out is None:
        if args.work_dir:
            out = str(Path(args.work_dir) / "rounds.png")
        else:
            out = "rounds.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
