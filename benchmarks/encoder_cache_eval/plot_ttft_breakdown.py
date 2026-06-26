#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the fine-grained TTFT composition vs RPS for a disagg_policy_compare run.

We decompose the client-observed TTFT into a stack that SUMS to it, using the
打点 already emitted by the instrumented servers:

  Stage 1 — encoder fanout (E side, proxy waits for embeds to hit the store)
    1. fetch/decode/tokenize  = EStage.render_ms - MMPrep.apply_ms
    2. preprocess (HF)        = MMPrep.apply_ms                 <- mm_processor_cache
    3. gpu encode             = ProducerEncode.encode_save_ms   <- ENCODER cache (policy)
    4. engine orchestration   = EStage.engine_ms - ProducerEncode
    5. proxy<->E network       = ProxyStageTiming.encoder_ms - (render+engine)
  Stage 2 — PD first chunk (PD side, prompt -> first token)
    6. PD queue               = vllm:request_queue_time_seconds (PD /metrics diff)
    7. PD prefill             = vllm:request_prefill_time_seconds
    8. PD residual            = client_ttft - encoder_fanout - PD_queue - PD_prefill
                                (proxy->PD dispatch + PD sched + first decode; not
                                 finely instrumented)

Per-RPS attribution: the E-side / proxy 打点 live in ONE continuous per-policy
log. disagg_policy_compare.sh brackets every RPS step with a Prometheus scrape
(``*.phase_before.txt`` / ``*.phase_after.txt``); their FILE MTIMES give the
[start, end] wall-clock window for that step, so we bucket each log line (by its
timestamp) into the matching RPS.

Outputs (to <run>/plots/):
  ttft_breakdown_abs_<policy>.png   stacked bars, absolute ms vs rps
  ttft_breakdown_pct_<policy>.png   stacked bars, % of TTFT vs rps
  producer_encode_vs_rps.png        GPU encode (ProducerEncode) vs rps, all policies
And a text table to stdout (+ optional --md).

Usage:
    python plot_ttft_breakdown.py <result-dir> [--md out.md]
where <result-dir> is the disagg_policy_* dir (the parent of runs/).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

# Swept-axis display label (set SWEEP_XLABEL=concurrency for closed-loop runs).
XLABEL = os.environ.get("SWEEP_XLABEL", "rps")
from datetime import datetime

# ── timestamp parsing ───────────────────────────────────────────────────────
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_VLLM = re.compile(r"(?<!\d)(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def parse_ts(line: str, year: int) -> datetime | None:
    """Parse a leading timestamp from a log line. Handles both the proxy's ISO
    ``YYYY-MM-DD HH:MM:SS`` and vLLM's ``MM-DD HH:MM:SS`` (year supplied)."""
    m = _ISO.search(line)
    if m:
        y, mo, d, h, mi, s = (int(x) for x in m.groups())
        try:
            return datetime(y, mo, d, h, mi, s)
        except ValueError:
            return None
    m = _VLLM.search(line)
    if m:
        mo, d, h, mi, s = (int(x) for x in m.groups())
        try:
            return datetime(year, mo, d, h, mi, s)
        except ValueError:
            return None
    return None


# ── log-line value extractors ───────────────────────────────────────────────
_RE_MMPREP = re.compile(r"\[MMPrep\].*?apply_ms=([\d.]+)")
_RE_ENCODE = re.compile(r"\[ProducerEncode\].*?encode_save_ms=([\d.]+)")
_RE_ESTAGE = re.compile(r"\[EStage\].*?render_ms=([\d.]+) engine_ms=([\d.]+)")
_RE_FANOUT = re.compile(r"\[ProxyStageTiming\].*?encoder_ms=([\d.]+)")


def scan_log(path: str, year: int) -> list[tuple[datetime, str, tuple]]:
    """Return [(dt, kind, values)] for every recognized 打点 line in a log."""
    out: list[tuple[datetime, str, tuple]] = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            if "[MMPrep]" in line:
                m = _RE_MMPREP.search(line)
                kind, vals = "mmprep", (float(m.group(1)),) if m else (None, None)
            elif "[ProducerEncode]" in line:
                m = _RE_ENCODE.search(line)
                kind, vals = "encode", (float(m.group(1)),) if m else (None, None)
            elif "[EStage]" in line:
                m = _RE_ESTAGE.search(line)
                kind, vals = (
                    "estage",
                    (float(m.group(1)), float(m.group(2))) if m else None,
                )
            elif "[ProxyStageTiming]" in line:
                m = _RE_FANOUT.search(line)
                kind, vals = "fanout", (float(m.group(1)),) if m else (None,)
            else:
                continue
            if vals is None or vals[0] is None:
                continue
            dt = parse_ts(line, year)
            if dt is not None:
                out.append((dt, kind, vals))
    return out


def window_mean(
    records: list[tuple[datetime, str, tuple]],
    kind: str,
    lo: datetime,
    hi: datetime,
    idx: int = 0,
) -> float:
    vals = [r[2][idx] for r in records if r[1] == kind and lo <= r[0] <= hi]
    return sum(vals) / len(vals) if vals else float("nan")


# ── PD-side Prometheus snapshot diff (same logic as phase_breakdown.py) ──────
def parse_snapshot(path: str) -> dict[str, tuple[float, float]]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            name = parts[0].split("{", 1)[0]
            try:
                val = float(parts[1])
            except ValueError:
                continue
            if name.endswith("_sum"):
                sums[name[:-4]] += val
            elif name.endswith("_count"):
                counts[name[:-6]] += val
    keys = set(sums) | set(counts)
    return {k: (sums.get(k, 0.0), counts.get(k, 0.0)) for k in keys}


def avg_ms(before: dict, after: dict, metric: str) -> float:
    bs, bc = before.get(metric, (0.0, 0.0))
    as_, ac = after.get(metric, (0.0, 0.0))
    dc = ac - bc
    return (as_ - bs) / dc * 1000.0 if dc > 0 else float("nan")


# ── component definition (bottom -> top of the stack) ───────────────────────
COMPONENTS = [
    ("fetch/decode/tok", "#9ecae1"),
    ("preprocess (HF)", "#3182bd"),
    ("gpu encode", "#e6550d"),
    ("engine orch", "#fdae6b"),
    ("proxy<->E net", "#74c476"),
    ("PD queue", "#c7c7c7"),
    ("PD prefill", "#756bb1"),
    ("PD residual", "#bcbddc"),
]


def components(row: dict) -> dict[str, float]:
    render = row.get("render", float("nan"))
    engine = row.get("engine", float("nan"))
    mmprep = row.get("mmprep", float("nan"))
    encode = row.get("encode", float("nan"))
    fanout = row.get("fanout", float("nan"))
    ttft = row.get("client_ttft", float("nan"))
    queue = row.get("pd_queue", float("nan"))
    prefill = row.get("pd_prefill", float("nan"))

    def pos(x):
        return x if (x == x and x > 0) else 0.0

    # If proxy fanout is missing, fall back to E-internal render+engine.
    stage1 = fanout if fanout == fanout else (render + engine)
    return {
        "fetch/decode/tok": pos(render - mmprep),
        "preprocess (HF)": pos(mmprep),
        "gpu encode": pos(encode),
        "engine orch": pos(engine - encode),
        "proxy<->E net": pos(stage1 - render - engine),
        "PD queue": pos(queue),
        "PD prefill": pos(prefill),
        "PD residual": pos(ttft - stage1 - pos(queue) - pos(prefill)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="the disagg_policy_* result dir (parent of runs/)")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    runs_dir = os.path.join(run_dir, "runs")

    # Year for vLLM-format timestamps: pull from the dir name disagg_policy_YYYYMMDD_*.
    ym = re.search(r"(\d{4})\d{4}_\d{6}", os.path.basename(run_dir))
    year = int(ym.group(1)) if ym else datetime.now().year  # noqa: DTZ005

    # policies present (one encoder.log each)
    policies = sorted(
        os.path.basename(p)[: -len(".encoder.log")]
        for p in glob.glob(os.path.join(run_dir, "*.encoder.log"))
    )
    if not policies:
        print(f"no <policy>.encoder.log in {run_dir}")
        return

    # data[policy] = list of (rps, components_dict, raw_row)
    data: dict[str, list] = {}
    for policy in policies:
        recs = scan_log(os.path.join(run_dir, f"{policy}.encoder.log"), year)
        recs += scan_log(os.path.join(run_dir, f"{policy}.proxy.log"), year)
        rows = []
        pat = os.path.join(runs_dir, f"{policy}_rps*_rep*.phase_before.txt")
        for bf in sorted(glob.glob(pat)):
            stem = bf[: -len(".phase_before.txt")]
            af = stem + ".phase_after.txt"
            base = os.path.basename(stem)
            m = re.match(r".+_rps(?P<rps>[0-9.]+)_rep(?P<rep>\d+)$", base)
            if not m or not os.path.exists(af):
                continue
            rps = float(m.group("rps"))
            lo = datetime.fromtimestamp(os.path.getmtime(bf))  # noqa: DTZ006
            hi = datetime.fromtimestamp(os.path.getmtime(af))  # noqa: DTZ006
            if hi < lo:
                lo, hi = hi, lo
            before, after = parse_snapshot(bf), parse_snapshot(af)
            client_ttft = float("nan")
            jp = stem + ".json"
            if os.path.exists(jp):
                try:
                    client_ttft = float(json.load(open(jp)).get("mean_ttft_ms", "nan"))
                except (ValueError, json.JSONDecodeError, OSError):
                    pass
            estage_r = [r for r in recs if r[1] == "estage" and lo <= r[0] <= hi]
            row = {
                "mmprep": window_mean(recs, "mmprep", lo, hi),
                "encode": window_mean(recs, "encode", lo, hi),
                "render": (
                    sum(r[2][0] for r in estage_r) / len(estage_r)
                    if estage_r
                    else float("nan")
                ),
                "engine": (
                    sum(r[2][1] for r in estage_r) / len(estage_r)
                    if estage_r
                    else float("nan")
                ),
                "fanout": window_mean(recs, "fanout", lo, hi),
                "pd_queue": avg_ms(before, after, "vllm:request_queue_time_seconds"),
                "pd_prefill": avg_ms(
                    before, after, "vllm:request_prefill_time_seconds"
                ),
                "client_ttft": client_ttft,
                "n_estage": len(estage_r),
            }
            rows.append((rps, components(row), row))
        rows.sort(key=lambda r: r[0])
        data[policy] = rows

    # ── text table ──────────────────────────────────────────────────────────
    lines = ["# TTFT fine breakdown vs RPS (ms avg/request)", ""]
    hdr = f"| policy | {XLABEL} | " + " | ".join(c for c, _ in COMPONENTS) + " | TTFT(sum) |"
    lines += [hdr, "|" + "---|" * (len(COMPONENTS) + 3)]
    for policy in policies:
        for rps, comp, raw in data[policy]:
            tot = sum(comp.values())
            cells = " | ".join(f"{comp[c]:.0f}" for c, _ in COMPONENTS)
            lines.append(f"| {policy} | {rps:g} | {cells} | {tot:.0f} |")
    table = "\n".join(lines)
    print(table)
    if args.md:
        with open(args.md, "w") as f:
            f.write(table + "\n")

    # ── plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); wrote table only.")
        return

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    labels = [c for c, _ in COMPONENTS]
    colors = [c for _, c in COMPONENTS]

    def stacked(policy, rows, normalize, fname, title):
        if not rows:
            return
        xs = [f"{r[0]:g}" for r in rows]
        x = range(len(xs))
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(xs) + 2), 5))
        bottoms = [0.0] * len(xs)
        for lab, col in zip(labels, colors):
            vals = []
            for (_, comp, _) in rows:
                tot = sum(comp.values()) or 1.0
                vals.append(comp[lab] / tot * 100.0 if normalize else comp[lab])
            ax.bar(x, vals, bottom=bottoms, label=lab, color=col, width=0.7)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(list(x))
        ax.set_xticklabels(xs)
        ax.set_xlabel("request rate (RPS)" if XLABEL == "rps" else "max concurrency (in flight)")
        ax.set_ylabel("% of TTFT" if normalize else "ms")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, fname), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {os.path.join(plot_dir, fname)}")

    for policy in policies:
        stacked(policy, data[policy], False,
                f"ttft_breakdown_abs_{policy}.png",
                f"TTFT composition vs RPS — {policy} (absolute ms)")
        stacked(policy, data[policy], True,
                f"ttft_breakdown_pct_{policy}.png",
                f"TTFT composition vs RPS — {policy} (% of TTFT)")

    # ProducerEncode (gpu encode) vs rps, all policies overlaid
    fig, ax = plt.subplots(figsize=(7, 5))
    any_pt = False
    for policy in policies:
        pts = [(r[0], r[2]["encode"]) for r in data[policy]
               if r[2]["encode"] == r[2]["encode"]]
        if pts:
            any_pt = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", label=policy)
    if any_pt:
        ax.set_xlabel("request rate (RPS)" if XLABEL == "rps" else "max concurrency (in flight)")
        ax.set_ylabel("ProducerEncode (GPU encode+save) ms/step")
        ax.set_title("GPU encode time vs RPS")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fp = os.path.join(plot_dir, "producer_encode_vs_rps.png")
        fig.savefig(fp, dpi=130)
        print(f"[plot] wrote {fp}")
    plt.close(fig)


if __name__ == "__main__":
    main()
