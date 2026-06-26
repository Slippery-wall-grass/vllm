#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize server-side TTFT phase composition for a disagg_policy_compare run.

disagg_policy_compare.sh snapshots the PD workers' Prometheus ``/metrics``
before and after each RPS bench into
``<result-dir>/runs/<policy>_rps<rps>_rep<rep>.phase_{before,after}.txt``.

This script diffs each before/after pair (so the numbers reflect ONLY that RPS
step, not the server's whole lifetime), turns the histogram ``_sum`` / ``_count``
into a per-request average, and prints a table of where first-token latency goes:

    queue   = vllm:request_queue_time_seconds      (waiting in the PD queue)
    prefill = vllm:request_prefill_time_seconds     (PD prefill compute)
    decode  = vllm:request_decode_time_seconds      (PD decode, for context)
    TTFT    = vllm:time_to_first_token_seconds       (server-side, ~queue+prefill)

NOTE: these are PD-side phases. The encoder forward time (~c_i, on the E
worker) is reported separately by aggregate.py as "Encoder forward time
MEASURED"; add it to queue+prefill for the full client-observed TTFT.

Usage:
    python phase_breakdown.py <result-dir>/runs [--md out.md]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

# Swept-axis display label (set SWEEP_XLABEL=concurrency for closed-loop runs).
XLABEL = os.environ.get("SWEEP_XLABEL", "rps")

METRICS = {
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "ttft": "vllm:time_to_first_token_seconds",
}


def parse_snapshot(path: str) -> dict[str, tuple[float, float]]:
    """Return {metric_base: (sum, count)} summed over all PD workers."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)  # "<name>{labels} <value>"
            if len(parts) != 2:
                continue
            name_full = parts[0].split("{", 1)[0]
            try:
                val = float(parts[1])
            except ValueError:
                continue
            if name_full.endswith("_sum"):
                sums[name_full[:-4]] += val
            elif name_full.endswith("_count"):
                counts[name_full[:-6]] += val
    keys = set(sums) | set(counts)
    return {k: (sums.get(k, 0.0), counts.get(k, 0.0)) for k in keys}


def avg_ms(before: dict, after: dict, metric: str) -> float:
    bs, bc = before.get(metric, (0.0, 0.0))
    as_, ac = after.get(metric, (0.0, 0.0))
    dc = ac - bc
    return (as_ - bs) / dc * 1000.0 if dc > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="the <result-dir>/runs directory")
    ap.add_argument("--md", default=None, help="also write the table to this file")
    args = ap.parse_args()

    rows = []
    for bf in sorted(glob.glob(os.path.join(args.run_dir, "*.phase_before.txt"))):
        stem = bf[: -len(".phase_before.txt")]
        af = stem + ".phase_after.txt"
        base = os.path.basename(stem)
        m = re.match(r"(?P<policy>.+)_rps(?P<rps>[0-9.]+)_rep(?P<rep>\d+)$", base)
        if not m:
            continue
        before, after = parse_snapshot(bf), parse_snapshot(af)
        v = {k: avg_ms(before, after, metric) for k, metric in METRICS.items()}
        # Client-observed TTFT (from the bench result JSON) includes the encode
        # + transfer that the PD-side metric cannot see.
        client_ttft = float("nan")
        jp = stem + ".json"
        if os.path.exists(jp):
            try:
                client_ttft = float(json.load(open(jp)).get("mean_ttft_ms", "nan"))
            except (ValueError, json.JSONDecodeError, OSError):
                pass
        # encode + transfer + proxy ≈ client TTFT − PD-side TTFT.
        enc_net = (
            client_ttft - v["ttft"]
            if client_ttft == client_ttft and v["ttft"] == v["ttft"]
            else float("nan")
        )
        rows.append((m.group("policy"), float(m.group("rps")), v, client_ttft, enc_net))

    rows.sort(key=lambda r: (r[0], r[1]))

    def pct(x: float, whole: float) -> str:
        return f"{x / whole * 100:.0f}%" if whole and whole == whole and x == x else "—"

    lines = [
        "# TTFT composition (ms avg/request)",
        "",
        "client TTFT = encode+net + PD-queue + PD-prefill. `encode+net` is the "
        "gap between the client-measured TTFT and the PD-side TTFT, i.e. the "
        "encoder forward + E->PD transfer + proxy. Percentages are of client TTFT.",
        "",
        f"| policy | {XLABEL} | client TTFT | encode+net | PD queue | PD prefill | "
        "enc% | queue% | prefill% | (decode) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pol, rps, v, client_ttft, enc_net in rows:
        ct = client_ttft
        lines.append(
            f"| {pol} | {rps:g} | {ct:.0f} | {enc_net:.0f} | {v['queue']:.0f} | "
            f"{v['prefill']:.0f} | {pct(enc_net, ct)} | {pct(v['queue'], ct)} | "
            f"{pct(v['prefill'], ct)} | {v['decode']:.0f} |"
        )

    out = "\n".join(lines)
    print(out)
    if args.md:
        with open(args.md, "w") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
