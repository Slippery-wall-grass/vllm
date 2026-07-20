#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the per-worker service rate (mu) of every engine in an EPD
deployment -- the encoder E and each prefill-decode worker PD_i.

WHY NOT 1/E[service_time]:
  vLLM serves a BATCH of requests concurrently, so the per-request service time
  from ``vllm:request_inference_time_seconds`` does NOT invert into the service
  rate: with batch size B, mu ~= B / E[S], not 1 / E[S]. Inverting the mean
  service time underestimates mu by roughly the batch size.

WHAT WE ACTUALLY MEASURE:
  mu = (requests completed) / (time the worker was BUSY)
  where "busy" means ``vllm:num_requests_running > 0``. This is the
  queueing-theory service rate, needs no assumption about batching, and is
  exactly the ceiling that determines where the system saturates.

  We also report 1/E[S] and Little's-law mu (E[running]/E[S]) as a CROSS-CHECK:
  mu_busy and mu_little should agree within sampling error. If they diverge
  badly, the sampling interval is too coarse relative to request duration.

Usage -- two phases:

  # 1. during the benchmark (background), poll every engine's /metrics
  python worker_service_rate.py sample \
      --engines E=http://localhost:19534,PD0=http://localhost:19600 \
      --out metrics.csv --interval 0.5

  # 2. after it finishes
  python worker_service_rate.py report metrics.csv [--md service_rate.md]

Only reads Prometheus endpoints that stock upstream vLLM already exposes, so it
works against an unmodified engine.
"""
import argparse
import csv
import os
import signal
import sys
import time
import urllib.request
from collections import defaultdict

# Metric -> CSV column. Counters/histograms are cumulative; we diff them.
FIELDS = [
    ("running", "vllm:num_requests_running", "gauge"),
    ("waiting", "vllm:num_requests_waiting", "gauge"),
    ("kv", "vllm:kv_cache_usage_perc", "gauge"),
    ("success", "vllm:request_success_total", "counter"),
    ("infer_sum", "vllm:request_inference_time_seconds_sum", "counter"),
    ("infer_cnt", "vllm:request_inference_time_seconds_count", "counter"),
    ("prefill_sum", "vllm:request_prefill_time_seconds_sum", "counter"),
    ("prefill_cnt", "vllm:request_prefill_time_seconds_count", "counter"),
    ("queue_sum", "vllm:request_queue_time_seconds_sum", "counter"),
    ("queue_cnt", "vllm:request_queue_time_seconds_count", "counter"),
    ("decode_sum", "vllm:request_decode_time_seconds_sum", "counter"),
    ("decode_cnt", "vllm:request_decode_time_seconds_count", "counter"),
    ("prompt_tok", "vllm:prompt_tokens_total", "counter"),
    ("gen_tok", "vllm:generation_tokens_total", "counter"),
    ("mm_queries", "vllm:mm_cache_queries_total", "counter"),
    ("mm_hits", "vllm:mm_cache_hits_total", "counter"),
]
COLUMNS = ["t", "engine"] + [f[0] for f in FIELDS]


def scrape(base_url: str, timeout: float = 5.0) -> dict[str, float]:
    """Fetch /metrics and sum each series across its label sets."""
    out: dict[str, float] = defaultdict(float)
    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as r:
        text = r.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0].split("{", 1)[0]] += float(parts[1])
        except ValueError:
            continue
    return out


def cmd_sample(args: argparse.Namespace) -> None:
    engines: list[tuple[str, str]] = []
    for spec in args.engines.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, _, url = spec.partition("=")
        engines.append((name.strip(), url.strip()))
    if not engines:
        sys.exit("no engines given")

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        while not stop["now"]:
            t = time.time()
            for name, url in engines:
                try:
                    m = scrape(url)
                except Exception:  # noqa: BLE001 - a transient scrape miss is fine
                    continue
                w.writerow([f"{t:.3f}", name] + [f"{m.get(k, 0.0):g}" for _, k, _ in FIELDS])
            fh.flush()
            time.sleep(args.interval)


def _stats(rows: list[dict]) -> dict:
    """Per-engine service-rate statistics from its time-ordered samples."""
    rows = sorted(rows, key=lambda r: r["t"])
    if len(rows) < 2:
        return {}
    wall = rows[-1]["t"] - rows[0]["t"]

    # Busy time: rectangle-rule integral of 1{running > 0}. Attribute each
    # interval to the state observed at its START.
    busy = 0.0
    run_busy: list[float] = []
    peak_wait = 0.0
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if a["running"] > 0:
            busy += dt
            run_busy.append(a["running"])
        peak_wait = max(peak_wait, a["waiting"])
    peak_wait = max(peak_wait, rows[-1]["waiting"])

    def delta(k):
        return rows[-1][k] - rows[0][k]

    def mean_hist(sum_k, cnt_k):
        n = delta(cnt_k)
        return (delta(sum_k) / n) if n > 0 else float("nan")

    done = delta("success")
    mean_S = mean_hist("infer_sum", "infer_cnt")
    mean_run_busy = (sum(run_busy) / len(run_busy)) if run_busy else 0.0
    mm_q = delta("mm_queries")

    return {
        "wall": wall,
        "busy": busy,
        "rho": busy / wall if wall > 0 else float("nan"),
        "completed": done,
        # THE service rate: completions per second of busy time.
        "mu_busy": done / busy if busy > 0 else float("nan"),
        "throughput": done / wall if wall > 0 else float("nan"),
        "mean_S": mean_S,
        "mean_queue": mean_hist("queue_sum", "queue_cnt"),
        "mean_prefill": mean_hist("prefill_sum", "prefill_cnt"),
        "mean_batch": mean_run_busy,
        # Cross-checks -- see module docstring.
        "mu_naive": (1.0 / mean_S) if mean_S and mean_S == mean_S else float("nan"),
        "mu_little": (mean_run_busy / mean_S)
        if mean_S and mean_S == mean_S and mean_S > 0
        else float("nan"),
        "peak_wait": peak_wait,
        "mm_per_s": (mm_q / busy) if busy > 0 else float("nan"),
        "mm_hit_rate": (delta("mm_hits") / mm_q) if mm_q > 0 else float("nan"),
        "gen_tok_per_s": (delta("gen_tok") / busy) if busy > 0 else float("nan"),
    }


def cmd_report(args: argparse.Namespace) -> None:
    by_engine: dict[str, list[dict]] = defaultdict(list)
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            rec = {"t": float(row["t"])}
            ok = True
            for name, _, _ in FIELDS:
                try:
                    rec[name] = float(row[name])
                except (ValueError, KeyError, TypeError):
                    ok = False
                    break
            if ok:
                by_engine[row["engine"]].append(rec)

    stats = {e: _stats(rs) for e, rs in by_engine.items()}
    stats = {e: s for e, s in stats.items() if s}
    if not stats:
        sys.exit("no usable samples in " + args.csv)

    def is_encoder(name):
        return name.upper().startswith("E")

    def order(name):
        return (0 if is_encoder(name) else 1, name)

    names = sorted(stats, key=order)
    L = [
        "# Per-worker service rate",
        "",
        "`mu_busy` = completions / busy-time  — **this is the service rate**.",
        "`rho` = busy / wall = utilisation. `mean_batch` = mean concurrent",
        "requests while busy. `mu_little` = mean_batch / mean_S is a cross-check",
        "of `mu_busy`; `mu_naive` = 1/mean_S is shown only to demonstrate how far",
        "off the naive (batch-ignoring) estimate is.",
        "",
        "| engine | mu_busy (req/s) | rho | thrpt (req/s) | mean_S (s) | mean_batch |"
        " mu_little | mu_naive | queue (s) | prefill (s) | peak_wait |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        s = stats[n]
        L.append(
            f"| {n} | **{s['mu_busy']:.2f}** | {s['rho']:.2f} | {s['throughput']:.2f} | "
            f"{s['mean_S']:.2f} | {s['mean_batch']:.1f} | {s['mu_little']:.2f} | "
            f"{s['mu_naive']:.2f} | {s['mean_queue']:.3f} | {s['mean_prefill']:.3f} | "
            f"{s['peak_wait']:.0f} |"
        )

    # Aggregate capacity per side, and which side is the binding constraint.
    e_names = [n for n in names if is_encoder(n)]
    pd_names = [n for n in names if not is_encoder(n)]
    e_mu = sum(stats[n]["mu_busy"] for n in e_names)
    pd_mu = sum(stats[n]["mu_busy"] for n in pd_names)
    n_pd = len(pd_names)
    L += ["", "## Aggregate capacity", ""]
    if e_mu:
        L.append(f"- E side  ({len(e_names)} worker(s)): **{e_mu:.2f} req/s**")
    if n_pd:
        L.append(f"- PD side ({n_pd} workers): **{pd_mu:.2f} req/s**")
    if e_mu and pd_mu:
        if e_mu < pd_mu:
            L.append(
                f"- **Bottleneck = E** (encode). System ceiling ~{e_mu:.2f} req/s; "
                f"PD side could absorb {pd_mu / e_mu:.1f}x more."
            )
        else:
            L.append(
                f"- **Bottleneck = PD**. System ceiling ~{pd_mu:.2f} req/s; "
                f"E could feed {e_mu / pd_mu:.1f}x more."
            )

    mm_rates = [stats[n]["mm_per_s"] for n in e_names if stats[n]["mm_per_s"] == stats[n]["mm_per_s"]]
    if mm_rates:
        hits = [stats[n]["mm_hit_rate"] for n in e_names if stats[n]["mm_hit_rate"] == stats[n]["mm_hit_rate"]]
        L += [
            "",
            "## Encoder detail",
            "",
            f"- multimodal items/s while busy (summed over E): **{sum(mm_rates):.1f}**",
        ]
        if hits:
            L.append(f"- mm_processor_cache hit rate (mean over E): {sum(hits) / len(hits):.1%}")

    L += [
        "",
        "## Reading the cross-check",
        "",
        "`mu_busy` and `mu_little` agreeing means the measurement is sound. A big",
        "gap means the sampling interval is too coarse relative to how long a",
        "request stays resident -- lower --interval and re-run.",
    ]

    out = "\n".join(L)
    print(out)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(out + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sample", help="poll /metrics of every engine into a CSV")
    ps.add_argument(
        "--engines",
        required=True,
        help="comma list of NAME=BASE_URL, e.g. E=http://localhost:19534,PD0=http://localhost:19600",
    )
    ps.add_argument("--out", required=True)
    ps.add_argument("--interval", type=float, default=0.5)
    ps.set_defaults(func=cmd_sample)

    pr = sub.add_parser("report", help="compute per-worker service rates")
    pr.add_argument("csv")
    pr.add_argument("--md", default=None)
    pr.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
