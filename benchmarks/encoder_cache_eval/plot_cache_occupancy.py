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


def _ts_seconds(m: re.Match) -> float:
    """Convert a 'MM-DD HH:MM:SS' match to absolute seconds (year-agnostic).

    Good enough for relative-time plotting within a single run; assumes the
    run does not span a month boundary.
    """
    mo, da, hh, mm, ss = (int(g) for g in m.groups())
    return ((mo * 31 + da) * 24 + hh) * 3600 + mm * 60 + ss


def parse_log(path: Path) -> dict[str, list[float]]:
    """Extract the occupancy trajectory from one server log."""
    t: list[float] = []
    ref: list[float] = []
    pin: list[float] = []
    evi: list[float] = []
    cap: list[float] = []
    t0: float | None = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "encoder_cache " not in line or "referenced_tokens=" not in line:
                continue
            tm = _TS_RX.search(line)
            vals = {k: re.search(p, line) for k, p in _FIELDS.items()}
            if tm is None or not all(vals.values()):
                continue
            secs = _ts_seconds(tm)
            if t0 is None:
                t0 = secs
            t.append(secs - t0)
            ref.append(float(vals["referenced"].group(1)))
            pin.append(float(vals["pinned"].group(1)))
            evi.append(float(vals["evictable"].group(1)))
            cap.append(float(vals["capacity"].group(1)))
    return {"t": t, "referenced": ref, "pinned": pin, "evictable": evi, "cap": cap}


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
