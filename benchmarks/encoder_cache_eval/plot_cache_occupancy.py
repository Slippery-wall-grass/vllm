# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the encoder-cache token-occupancy trajectory over time.

The ``EncoderCacheManager`` emits a periodic ``encoder_cache ...`` line to
the server log (gated by ``VLLM_ENCODER_CACHE_STATS_INTERVAL_SEC``). Each
line carries an instantaneous split of the cache, in encoder-embedding
tokens, into three mutually exclusive buckets that tile the capacity:

  - referenced_tokens : entries still referenced by an in-flight request
                        (refcount > 0) -> cannot be evicted.
  - pinned_tokens     : unreferenced entries the policy still protects
                        ("kept because judged valuable").
  - evictable_tokens  : unreferenced entries past their unlock tick
                        ("droppable right now").

This script greps every such line out of one or more server logs and draws
a stacked-area trajectory (referenced / pinned / evictable), with the cache
capacity as a reference line. Multiple logs (e.g. fifo vs offline) are drawn
as separate stacked subplots sharing axes.

Usage:
  python plot_cache_occupancy.py SERVER_LOG [SERVER_LOG ...] \
      --out cache_occupancy.png [--labels fifo offline]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Logger prefix timestamp, e.g. "INFO 06-08 22:16:21 [..] encoder_cache ..."
_TS_RX = re.compile(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
_FIELDS = {
    "referenced": r"referenced_tokens=(\d+)",
    "pinned": r"pinned_tokens=(\d+)",
    "evictable": r"evictable_tokens=(\d+)",
    "capacity": r"capacity=(\d+)",
}
# Encoder-cache reset (e.g. prewarm --reset-after via POST /reset_encoder_cache)
# zeroes the cache instantly. Detected so the plot can show the true drop
# instead of interpolating a misleading ramp across the idle gap that follows.
_RESET_RX = re.compile(r"Resetting encoder cache")


def _ts_seconds(m: re.Match) -> float:
    """Convert a 'MM-DD HH:MM:SS' match to absolute seconds (year-agnostic).

    Good enough for relative-time plotting within a single run; assumes the
    run does not span a month boundary.
    """
    mo, da, hh, mm, ss = (int(g) for g in m.groups())
    return ((mo * 31 + da) * 24 + hh) * 3600 + mm * 60 + ss


def parse_log(path: Path) -> dict[str, list[float]]:
    """Extract the occupancy trajectory (and reset events) from one log.

    A synthetic all-zero sample is injected at each reset that falls within
    the trajectory, so the stacked area shows the instantaneous drop to 0
    rather than a straight line interpolated across the post-reset idle gap.
    """
    samples: list[tuple[float, float, float, float, float]] = []
    reset_abs: list[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tm = _TS_RX.search(line)
            if _RESET_RX.search(line):
                if tm is not None:
                    reset_abs.append(_ts_seconds(tm))
                continue
            if "encoder_cache " not in line or "referenced_tokens=" not in line:
                continue
            vals = {k: re.search(p, line) for k, p in _FIELDS.items()}
            if tm is None or not all(vals.values()):
                continue
            samples.append(
                (
                    _ts_seconds(tm),
                    float(vals["referenced"].group(1)),
                    float(vals["pinned"].group(1)),
                    float(vals["evictable"].group(1)),
                    float(vals["capacity"].group(1)),
                )
            )
    if not samples:
        return {"t": [], "referenced": [], "pinned": [], "evictable": [],
                "cap": [], "resets": []}

    t0 = samples[0][0]
    cap_val = samples[-1][4]
    # Inject a zero-occupancy sample at each in-window reset so the stack
    # truly drops to 0 there instead of being interpolated over.
    for r in reset_abs:
        if t0 <= r <= samples[-1][0]:
            samples.append((r, 0.0, 0.0, 0.0, cap_val))
    samples.sort(key=lambda s: s[0])

    return {
        "t": [s[0] - t0 for s in samples],
        "referenced": [s[1] for s in samples],
        "pinned": [s[2] for s in samples],
        "evictable": [s[3] for s in samples],
        "cap": [s[4] for s in samples],
        "resets": [r - t0 for r in reset_abs if t0 <= r <= samples[-1][0]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path, help="server log file(s)")
    ap.add_argument("--out", type=Path, default=Path("cache_occupancy.png"))
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="title per log (defaults to file stem)",
    )
    args = ap.parse_args()

    labels = args.labels or [p.stem for p in args.logs]
    if len(labels) != len(args.logs):
        ap.error("--labels must match the number of logs")

    series = [parse_log(p) for p in args.logs]
    n = len(series)
    fig, axes = plt.subplots(
        n, 1, figsize=(11, 3.2 * n), sharex=True, sharey=True, squeeze=False
    )

    # referenced (bottom) -> pinned -> evictable (top); free is the gap to cap.
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    names = ["referenced (in use)", "pinned (valuable)", "evictable (cold)"]
    for ax, s, label in zip(axes[:, 0], series, labels):
        if not s["t"]:
            ax.set_title(f"{label} — no encoder_cache lines found")
            continue
        ax.stackplot(
            s["t"],
            s["referenced"],
            s["pinned"],
            s["evictable"],
            labels=names,
            colors=colors,
            alpha=0.9,
        )
        cap = s["cap"][-1] if s["cap"] else None
        if cap:
            ax.axhline(cap, ls="--", lw=1.2, color="black", label=f"capacity={int(cap)}")
        for i, r in enumerate(s.get("resets", [])):
            ax.axvline(
                r,
                ls=":",
                lw=1.3,
                color="red",
                alpha=0.8,
                label="cache reset" if i == 0 else None,
            )
        ax.set_title(f"Encoder-cache token occupancy — {label}")
        ax.set_ylabel("encoder tokens")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.margins(x=0)
    axes[-1, 0].set_xlabel("time since first stats line (s)")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
