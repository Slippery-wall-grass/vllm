# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot REAL per-event cache hit rate by parsing the encoder worker log.

Requires the encoder worker to have been launched with
`VLLM_ENCODER_CACHE_TRACE=1`, which causes EncoderCacheManager to emit
one INFO line per `check_and_update_cache` call:

    EncoderCacheTrace req_id=<rid> mm_hash=<hash> hit=<0|1> \
        cum_hits=<N> cum_misses=<M>

This script reads the encoder log for each strategy (none / fifo /
dist_aware) and plots, for each strategy:

    * Top: cumulative hit rate vs cache event index
    * Bottom: rolling hit rate (configurable window) vs event index

These are the GROUND-TRUTH per-request hit rates as seen by the
EncoderCacheManager — not the workload-derived upper bound that the
older plot_stability.py script produced. They tell you both how
quickly the cache warms up and where it plateaus, separately for each
policy.

Usage:
    # Easiest: point at the work dir where run_cache_comparison.sh
    # writes its logs (logs/encoder_<label>_*.log).
    python plot_real_hitrate.py --work-dir /tmp/vmmu_data --window 100

    # Or pass logs explicitly:
    python plot_real_hitrate.py \
        --none-log path/to/encoder_none_*.log \
        --fifo-log path/to/encoder_fifo_*.log \
        --dist-log path/to/encoder_dist_aware_*.log \
        -o /tmp/real_hitrate.png

Note: these traces span the entire worker lifetime (warmup + all
rounds). The script splits the trace at every `Encoder cache stats
before reset` log line so each round shows up as its own segment.
"""

import argparse
import re
from pathlib import Path

# Format: EncoderCacheTrace req_id=<rid> mm_hash=<hash> hit=<0|1>
#         cum_hits=<N> cum_misses=<M>
TRACE_RE = re.compile(
    r"EncoderCacheTrace\s+"
    r"req_id=(?P<req_id>\S+)\s+"
    r"mm_hash=(?P<mm_hash>\S+)\s+"
    r"hit=(?P<hit>\d+)\s+"
    r"cum_hits=(?P<hits>\d+)\s+"
    r"cum_misses=(?P<misses>\d+)"
)
RESET_RE = re.compile(
    r"Encoder cache stats before reset:\s+"
    r"hits=(?P<hits>\d+)\s+misses=(?P<misses>\d+)"
)

STRATS = [
    ("none", "No cache", "tab:gray", "o"),
    ("fifo", "LRU + EC clean", "tab:blue", "s"),
    ("fifo_persistent", "LRU + EC persist", "tab:cyan", "D"),
    ("dist_aware", "OUR", "tab:red", "^"),
]


def _parse_log(path: Path) -> list[list[dict]]:
    """Return a list of round-segments. Each segment is a list of
    per-event dicts with keys: req_id, mm_hash, hit (0/1), hits,
    misses. Boundaries are determined by `Encoder cache stats before
    reset` lines (which fire just before each cache.reset()).
    """
    segments: list[list[dict]] = [[]]
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRACE_RE.search(line)
            if m:
                segments[-1].append({
                    "req_id": m.group("req_id"),
                    "mm_hash": m.group("mm_hash"),
                    "hit": int(m.group("hit")),
                    "hits": int(m.group("hits")),
                    "misses": int(m.group("misses")),
                })
                continue
            if RESET_RE.search(line):
                # End of a round; start a fresh segment unless current is empty
                if segments[-1]:
                    segments.append([])
    # Drop trailing empty segment
    if segments and not segments[-1]:
        segments.pop()
    return segments


def _resolve_logs(args: argparse.Namespace) -> dict[str, Path]:
    if args.work_dir:
        wd = Path(args.work_dir) / "logs"
        out: dict[str, Path] = {}
        for key, *_ in STRATS:
            # logs/encoder_<label>_<timestamp>.log — pick the most recent
            matches = sorted(wd.glob(f"encoder_{key}_*.log"))
            if matches:
                out[key] = matches[-1]
        return out
    out = {}
    if args.none_log:
        out["none"] = Path(args.none_log)
    if args.fifo_log:
        out["fifo"] = Path(args.fifo_log)
    if args.dist_log:
        out["dist_aware"] = Path(args.dist_log)
    if not out:
        raise SystemExit(
            "Pass --work-dir or at least one of --none-log/--fifo-log/"
            "--dist-log")
    return out


def _rolling_hit_rate(events: list[dict], window: int) -> list[float]:
    """Compute right-aligned rolling hit rate over the last `window`
    events. Returns one value per event; the first window-1 entries
    are computed over a growing window.
    """
    out: list[float] = []
    queue: list[int] = []
    s = 0
    for e in events:
        h = e["hit"]
        queue.append(h)
        s += h
        if len(queue) > window:
            s -= queue.pop(0)
        out.append(s / len(queue))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory where run_cache_comparison.sh "
                             "wrote logs/ (looks for "
                             "logs/encoder_{none,fifo,dist_aware}_*.log)")
    parser.add_argument("--none-log", type=str, default=None)
    parser.add_argument("--fifo-log", type=str, default=None)
    parser.add_argument("--dist-log", type=str, default=None)
    parser.add_argument("--window", type=int, default=100,
                        help="Rolling window width in events "
                             "(default: 100)")
    parser.add_argument("--round", type=int, default=-1,
                        help="Which round-segment to plot. Default -1 "
                             "= last (post-warmup, steady state). 0 = "
                             "first. Use --all-rounds to overlay every "
                             "round on top of each other.")
    parser.add_argument("--all-rounds", action="store_true",
                        help="Plot every round-segment with decreasing "
                             "alpha so you can see how stabilisation "
                             "improves across rounds.")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    logs = _resolve_logs(args)
    print("Parsing logs:")
    parsed: dict[str, list[list[dict]]] = {}
    for key, p in logs.items():
        segs = _parse_log(p)
        if not segs:
            print(f"  {key:<10} {p}: no trace lines found "
                  f"(was VLLM_ENCODER_CACHE_TRACE=1 set?)")
            continue
        parsed[key] = segs
        sizes = [len(s) for s in segs]
        print(f"  {key:<10} {p.name}: {len(segs)} round-segments, "
              f"sizes={sizes}")
    if not parsed:
        raise SystemExit(
            "No trace data found. Set VLLM_ENCODER_CACHE_TRACE=1 on the "
            "encoder worker and re-run.")

    fig, (ax_cum, ax_roll) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True)

    def _segments_to_plot(
        segs: list[list[dict]],
    ) -> list[tuple[int, list[dict]]]:
        if args.all_rounds:
            return list(enumerate(segs))
        idx = args.round
        if idx < 0:
            idx = len(segs) + idx
        if not 0 <= idx < len(segs):
            return []
        return [(idx, segs[idx])]

    for key, label, color, marker in STRATS:
        segs = parsed.get(key)
        if not segs:
            continue
        for round_idx, events in _segments_to_plot(segs):
            if not events:
                continue
            xs = list(range(1, len(events) + 1))
            cum = [(e["hits"]) / (e["hits"] + e["misses"])
                   for e in events]
            roll = _rolling_hit_rate(events, args.window)

            alpha = 1.0
            if args.all_rounds:
                alpha = 0.4 + 0.6 * (round_idx + 1) / len(segs)
            lab = label if not args.all_rounds else \
                f"{label} (round {round_idx + 1})"

            ax_cum.plot(xs, [v * 100 for v in cum], color=color,
                        linewidth=2.0, alpha=alpha, label=lab)
            ax_roll.plot(xs, [v * 100 for v in roll], color=color,
                         linewidth=1.8, alpha=alpha, label=lab)

    ax_cum.set_ylabel("Cumulative hit rate (%)")
    ax_cum.set_title("REAL cumulative hit rate "
                      "(from EncoderCacheManager trace)")
    ax_cum.set_ylim(0, 100)
    ax_cum.grid(True, alpha=0.3)
    ax_cum.legend(loc="best", fontsize=9)

    ax_roll.set_xlabel("Cache event index "
                        "(check_and_update_cache call order)")
    ax_roll.set_ylabel("Rolling hit rate (%)")
    ax_roll.set_title(f"REAL rolling hit rate (window={args.window})")
    ax_roll.set_ylim(0, 100)
    ax_roll.grid(True, alpha=0.3)
    ax_roll.legend(loc="best", fontsize=9)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out = args.output
    if out is None:
        if args.work_dir:
            out = str(Path(args.work_dir) / "real_hitrate.png")
        else:
            out = "real_hitrate.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    print("\nFinal stats per strategy / round:")
    for key, label, *_ in STRATS:
        segs = parsed.get(key)
        if not segs:
            continue
        for ri, events in enumerate(segs):
            if not events:
                continue
            last = events[-1]
            total = last["hits"] + last["misses"]
            hr = last["hits"] / total * 100 if total > 0 else 0
            print(f"  {label:<22} round {ri + 1:>2}: "
                  f"hits={last['hits']:>5} misses={last['misses']:>5} "
                  f"hit_rate={hr:>5.1f}%")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
