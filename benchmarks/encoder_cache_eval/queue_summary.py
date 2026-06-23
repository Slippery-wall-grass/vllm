#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Show which stage (the single encoder E vs the PD workers) starts queuing
first as offered load rises — the core "where is the bottleneck" question.

disagg_policy_compare.sh's background sampler writes, every QUEUE_SAMPLE_SEC
seconds (default 0.25) during each RPS bench, one row per engine into
``<run_dir>/<policy>_rps<rps>_rep<rep>.queues.csv`` with columns
``t,engine,waiting,running,kv`` (waiting = vllm:num_requests_waiting,
kv = vllm:kv_cache_usage_perc). ``engine`` is ``E`` for the encoder and
``PD0/PD1/...`` for the consumers.

For each (policy, rps) we report the PEAK queue depth on E and the peak
*summed* queue depth across the PD workers, plus peak KV usage. Whichever
side's ``Waiting`` lifts off zero at the LOWER rps is the first to queue =
the bottleneck. Theory for this experiment: you WANT E (encode) to queue
first, so the encoder-cache policy governs throughput.

Usage:
    python queue_summary.py <run_dir> [--md out.md] [--threshold 1.0]
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", default=None)
    ap.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="peak Waiting >= this counts as 'this stage is queuing'",
    )
    args = ap.parse_args()

    # data[(policy, rps)] = {"E": {"wait": [...], "kv": [...]}, "PD": {...}}
    data: dict = defaultdict(
        lambda: {"E": {"wait": [], "kv": []}, "PD": {"wait": [], "kv": []}}
    )
    for f in glob.glob(os.path.join(args.run_dir, "*.queues.csv")):
        base = os.path.basename(f)[: -len(".queues.csv")]
        m = re.match(r"(?P<policy>.+)_rps(?P<rps>[0-9.]+)_rep(?P<rep>\d+)$", base)
        if not m:
            continue
        key = (m.group("policy"), float(m.group("rps")))
        # Sum PD waiting across PD engines at each timestamp, then take peaks.
        pd_wait_t: dict = defaultdict(float)
        pd_kv_t: dict = defaultdict(float)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                try:
                    w = float(row["waiting"])
                    kv = float(row["kv"])
                except (ValueError, KeyError, TypeError):
                    continue
                if row.get("engine") == "E":
                    data[key]["E"]["wait"].append(w)
                    data[key]["E"]["kv"].append(kv)
                else:
                    pd_wait_t[row["t"]] += w
                    pd_kv_t[row["t"]] = max(pd_kv_t[row["t"]], kv)
        data[key]["PD"]["wait"].extend(pd_wait_t.values())
        data[key]["PD"]["kv"].extend(pd_kv_t.values())

    def peak(xs):
        return max(xs) if xs else 0.0

    rows = sorted(data.items())
    lines = [
        "# Where the queue builds first (E encoder vs PD workers)",
        "",
        "Peak `Waiting` per RPS step. The side whose Waiting lifts off zero at "
        "the LOWER rps is the first to queue = the bottleneck. For this study "
        "you want **E (encode)** to queue first.",
        "",
        "| policy | rps | E wait | PD wait | E kv% | PD kv% | queuing |",
        "|---|---|---|---|---|---|---|",
    ]
    first: dict = defaultdict(lambda: {"E": None, "PD": None})
    for (pol, rps), d in rows:
        ew, pw = peak(d["E"]["wait"]), peak(d["PD"]["wait"])
        ekv, pkv = peak(d["E"]["kv"]) * 100, peak(d["PD"]["kv"]) * 100
        tags = []
        if ew >= args.threshold:
            tags.append("E")
            if first[pol]["E"] is None:
                first[pol]["E"] = rps
        if pw >= args.threshold:
            tags.append("PD")
            if first[pol]["PD"] is None:
                first[pol]["PD"] = rps
        lines.append(
            f"| {pol} | {rps:g} | {ew:.0f} | {pw:.0f} | {ekv:.0f}% | {pkv:.0f}% | "
            f"{'+'.join(tags) or '—'} |"
        )

    lines += ["", "## First RPS at which each stage starts queuing", ""]
    for pol, fq in sorted(first.items()):
        e, p = fq["E"], fq["PD"]
        if e is not None and (p is None or e <= p):
            verdict = "ENCODE queues first → encoder-bound ✅"
        elif p is not None:
            verdict = "PD queues first → prefill/decode-bound"
        else:
            verdict = "neither queued (raise RPS)"
        lines.append(f"- {pol}: E@{e}  PD@{p}  →  {verdict}")

    out = "\n".join(lines)
    print(out)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(out + "\n")


if __name__ == "__main__":
    main()
