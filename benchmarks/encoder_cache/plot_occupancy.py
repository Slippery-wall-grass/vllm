# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot REAL cache occupancy over time (pinned vs freeable vs free).

Requires the encoder worker to have been launched with
`VLLM_ENCODER_CACHE_TRACE=1` (default in run_cache_comparison.sh).
EncoderCacheManager then emits one INFO line per state change:

    EncoderCacheOccupancy event=<allocate|free|evict> t=<seconds> \\
        cache_size=<S> pinned=<P> freeable=<F> free=<E> \\
        num_pinned_entries=<NP> num_freeable_entries=<NF>

This script reads the encoder log for each strategy and plots, per
strategy, a stacked-area chart over time:

    * red    = pinned    (held by in-flight requests, NOT evictable)
    * yellow = freeable  (no live ref, can be evicted on demand)
    * gray   = free      (physically empty)

The total stacked height equals `cache_size`. The pinned band tells you
how much capacity the cache is forced to spend on staging in-flight
requests — that's the slice the eviction policy CANNOT touch. Only the
remaining (freeable + free) is "where caching actually happens", and
the policy comparison only matters in that region.

Usage:
    python plot_occupancy.py --work-dir /tmp/vmmu_data
    # or per strategy:
    python plot_occupancy.py --fifo-log path/to/encoder_fifo_*.log
"""

import argparse
import re
from pathlib import Path

OCC_RE = re.compile(
    r"EncoderCacheOccupancy\s+"
    r"event=(?P<event>\S+)\s+"
    r"t=(?P<t>[\d.]+)\s+"
    r"cache_size=(?P<size>\d+)\s+"
    r"pinned=(?P<pinned>\d+)\s+"
    r"freeable=(?P<freeable>\d+)\s+"
    r"free=(?P<free>\d+)"
)
RESET_RE = re.compile(
    r"Encoder cache stats before reset:\s+"
    r"hits=(?P<hits>\d+)\s+misses=(?P<misses>\d+)"
)

STRATS = [
    ("none", "No cache", "tab:gray"),
    ("fifo", "LRU + EC clean", "tab:blue"),
    ("fifo_persistent", "LRU + EC persist", "tab:cyan"),
    ("dist_aware", "OUR", "tab:red"),
]


def _parse(path: Path) -> list[list[dict]]:
    """Return list of round-segments. Each segment is a list of state
    samples in chronological order. Round boundaries come from the
    `Encoder cache stats before reset` line that fires on cache.reset().
    """
    if not path.exists():
        return []
    segments: list[list[dict]] = [[]]
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = OCC_RE.search(line)
            if m:
                segments[-1].append({
                    "event": m.group("event"),
                    "t": float(m.group("t")),
                    "size": int(m.group("size")),
                    "pinned": int(m.group("pinned")),
                    "freeable": int(m.group("freeable")),
                    "free": int(m.group("free")),
                })
                continue
            if RESET_RE.search(line):
                if segments[-1]:
                    segments.append([])
    if segments and not segments[-1]:
        segments.pop()
    return segments


def _resolve_logs(args: argparse.Namespace) -> dict[str, Path]:
    if args.work_dir:
        wd = Path(args.work_dir) / "logs"
        out: dict[str, Path] = {}
        for key, *_ in STRATS:
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
            "Pass --work-dir or --{none,fifo,dist}-log")
    return out


def _build_step(samples: list[dict]) -> tuple[list[float], list[int],
                                              list[int], list[int]]:
    """Convert event samples to step-function arrays for fill_between.

    Each event holds the post-event state, so the value is constant
    from this event's t until the next event's t. We duplicate the t
    of the next event so matplotlib draws a horizontal segment.
    """
    if not samples:
        return [], [], [], []
    ts: list[float] = []
    pin: list[int] = []
    fre: list[int] = []
    fre_only: list[int] = []
    for i, s in enumerate(samples):
        next_t = samples[i + 1]["t"] if i + 1 < len(samples) else s["t"]
        for t in (s["t"], next_t):
            ts.append(t)
            pin.append(s["pinned"])
            fre.append(s["pinned"] + s["freeable"])
            fre_only.append(s["freeable"])
    return ts, pin, fre, fre_only


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory with logs/encoder_<label>_*.log")
    parser.add_argument("--none-log", type=str, default=None)
    parser.add_argument("--fifo-log", type=str, default=None)
    parser.add_argument("--dist-log", type=str, default=None)
    parser.add_argument("--round", type=int, default=-1,
                        help="Which round-segment to plot. Default -1 "
                             "= last round (steady state).")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--time-axis", type=str, default="time",
                        choices=("time", "event"),
                        help="X-axis: 'time' (seconds since first event) "
                             "or 'event' (event index).")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    logs = _resolve_logs(args)
    print("Parsing logs:")
    parsed: dict[str, list[list[dict]]] = {}
    for key, p in logs.items():
        segs = _parse(p)
        if not segs:
            print(f"  {key:<10} {p}: no occupancy lines "
                  f"(VLLM_ENCODER_CACHE_TRACE off?)")
            continue
        parsed[key] = segs
        sizes = [len(s) for s in segs]
        print(f"  {key:<10} {p.name}: {len(segs)} rounds, "
              f"events={sizes}")
    if not parsed:
        raise SystemExit("No occupancy data found.")

    n_strats = len(parsed)
    fig, axes = plt.subplots(
        n_strats, 1, figsize=(11, 3.2 * n_strats), sharex=False,
    )
    if n_strats == 1:
        axes = [axes]

    cache_size = None
    for ax, (key, label, color) in zip(
            axes, [s for s in STRATS if s[0] in parsed]):
        segs = parsed[key]
        idx = args.round if args.round >= 0 else len(segs) + args.round
        idx = max(0, min(idx, len(segs) - 1))
        samples = segs[idx]
        if not samples:
            ax.set_title(f"{label} — round {idx + 1}: no events")
            continue

        if args.time_axis == "time":
            ts, pin, pin_plus_fre, fre_only = _build_step(samples)
            xlabel = "Time since first event (s)"
        else:
            # one event index per sample, no step expansion
            ts = list(range(len(samples)))
            pin = [s["pinned"] for s in samples]
            pin_plus_fre = [s["pinned"] + s["freeable"] for s in samples]
            fre_only = [s["freeable"] for s in samples]
            xlabel = "Event index"

        size = samples[0]["size"]
        cache_size = size
        # Stack: pinned (red) on bottom, freeable (yellow) middle,
        # free (gray) top.
        ax.fill_between(ts, 0, pin, color="tab:red", alpha=0.55,
                        label="Pinned (held by in-flight reqs)",
                        step="post" if args.time_axis == "event" else None)
        ax.fill_between(ts, pin, pin_plus_fre, color="gold", alpha=0.55,
                        label="Freeable (cached, no live ref)",
                        step="post" if args.time_axis == "event" else None)
        ax.fill_between(ts, pin_plus_fre, [size] * len(ts),
                        color="lightgray", alpha=0.55, label="Free",
                        step="post" if args.time_axis == "event" else None)
        ax.axhline(y=size, color="black", linewidth=0.8,
                   linestyle="--", label=f"cache_size={size}")

        # Numeric summary annotations
        max_pin = max(s["pinned"] for s in samples)
        mean_pin = sum(s["pinned"] for s in samples) / len(samples)
        max_pin_pct = max_pin / size * 100
        mean_pin_pct = mean_pin / size * 100
        ax.set_title(
            f"{label} — round {idx + 1}: "
            f"max pinned={max_pin} ({max_pin_pct:.0f}%), "
            f"mean pinned={mean_pin:.0f} ({mean_pin_pct:.0f}%)")
        ax.set_ylabel("Tokens")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, size * 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncols=2)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out = args.output
    if out is None:
        if args.work_dir:
            out = str(Path(args.work_dir) / "occupancy.png")
        else:
            out = "occupancy.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    # Numeric per-strategy summary
    print("\nOccupancy summary (last round only):")
    print(f"{'strategy':<22} {'max_pin':>8} {'mean_pin':>9} "
          f"{'max_pin%':>10} {'mean_pin%':>10}")
    for key, label, _ in STRATS:
        if key not in parsed:
            continue
        segs = parsed[key]
        idx = args.round if args.round >= 0 else len(segs) + args.round
        idx = max(0, min(idx, len(segs) - 1))
        samples = segs[idx]
        if not samples:
            continue
        size = samples[0]["size"]
        pins = [s["pinned"] for s in samples]
        max_pin = max(pins); mean_pin = sum(pins) / len(pins)
        print(f"{label:<22} {max_pin:>8} {mean_pin:>9.0f} "
              f"{max_pin/size*100:>9.1f}% {mean_pin/size*100:>9.1f}%")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
