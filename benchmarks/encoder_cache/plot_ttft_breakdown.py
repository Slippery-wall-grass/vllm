# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decompose client-observed TTFT into server-side phases per request.

Requires VLLM_REQUEST_TIMING_TRACE=1 on both encoder and prefill workers
(default in run_cache_comparison.sh). Two log line formats are parsed:

  Encoder worker:
    EncoderForwardTrace req_ids=<id1,id2,...> num_items=N \
        duration_ms=D per_request_ms=P

  Prefill worker:
    RequestPhases req_id=<rid> arrival=<ts> pre_queue_ms=A \
        queued_ms=Q prefill_ms=PF first_token_latency_ms=L prompt_len=N

Each request is correlated by `request_id` (the server-side id captured
client-side via the X-Request-Id response header).

The script computes, per request:
  * client_ttft       — what the client observed (from raw_results.ttft)
  * encoder_forward   — pulled from EncoderForwardTrace; 0 ⇒ cache HIT
  * prefill_first_tok — from RequestPhases.prefill_ms
  * queue_prefill     — from RequestPhases.queued_ms
  * other             — client_ttft − (encoder + queue + prefill)
                        (transport, EC save/load, proxy hop, network)

It then plots two panels per strategy:
  * Top:    stacked bar of mean phase durations (one bar per strategy)
  * Bottom: per-request breakdown over time (sample of N requests)

Usage:
    python plot_ttft_breakdown.py --work-dir /tmp/vmmu_data
"""

import argparse
import json
import re
from pathlib import Path

ENC_RE = re.compile(
    r"EncoderForwardTrace\s+"
    # req_ids contains comma-separated request ids; each id may include
    # the chatcmpl- prefix, dashes (uuid), and colon-separated suffixes
    # (input_id, mm hash chunks). Stop at whitespace.
    r"req_ids=(?P<ids>\S+)\s+"
    r"num_items=(?P<n>\d+)\s+"
    r"duration_ms=(?P<dur>[\d.]+)\s+"
    r"per_request_ms=(?P<per>[\d.]+)"
)
PHASE_RE = re.compile(
    r"RequestPhases\s+"
    r"req_id=(?P<rid>\S+)\s+"
    r"arrival=(?P<arr>[\d.]+)\s+"
    r"pre_queue_ms=(?P<pre>[\d.]+)\s+"
    r"queued_ms=(?P<q>[\d.]+)\s+"
    r"prefill_ms=(?P<pf>[\d.]+)\s+"
    r"first_token_latency_ms=(?P<lat>[\d.]+)\s+"
    r"prompt_len=(?P<plen>\d+)"
)

STRATS = [
    ("none", "No cache", "tab:gray"),
    ("fifo", "FIFO", "tab:blue"),
    ("dist_aware", "Distribution-Aware", "tab:red"),
]


def _parse_logs(work_dir: Path, strat: str
                ) -> tuple[dict[str, float], dict[str, dict]]:
    """Return (encoder_per_req_ms, prefill_phases_per_req).

    encoder_per_req_ms: rid -> per_request_ms (sum if multiple groups)
    prefill_phases_per_req: rid -> {pre_queue_ms, queued_ms, prefill_ms,
                                    first_token_latency_ms, prompt_len}
    """
    enc_logs = sorted((work_dir / "logs").glob(f"encoder_{strat}_*.log"))
    pref_logs = sorted((work_dir / "logs").glob(f"prefill_{strat}_*.log"))

    encoder: dict[str, float] = {}
    if enc_logs:
        with enc_logs[-1].open(encoding="utf-8", errors="replace") as f:
            for line in f:
                m = ENC_RE.search(line)
                if not m:
                    continue
                per_req = float(m.group("per"))
                for rid in m.group("ids").split(","):
                    rid = rid.strip()
                    if rid:
                        encoder[rid] = encoder.get(rid, 0.0) + per_req

    prefill: dict[str, dict] = {}
    if pref_logs:
        with pref_logs[-1].open(encoding="utf-8", errors="replace") as f:
            for line in f:
                m = PHASE_RE.search(line)
                if not m:
                    continue
                rid = m.group("rid")
                if rid in prefill:
                    continue  # only first occurrence per rid
                prefill[rid] = {
                    "pre_queue_ms": float(m.group("pre")),
                    "queued_ms": float(m.group("q")),
                    "prefill_ms": float(m.group("pf")),
                    "first_token_latency_ms": float(m.group("lat")),
                    "prompt_len": int(m.group("plen")),
                }
    return encoder, prefill


def _index_by_substring(d: dict[str, "T"]) -> dict[str, "T"]:
    """Build a lookup by 'core uuid' substring of each log key.

    Server-side log keys look like `chatcmpl-<uuid>:<input_id>:<8hex>-<8hex>`
    (or with `-<8hex>` suffix from input_processor). The client only sees
    the bare `<uuid>` from the proxy's X-Request-Id header. We canonicalise
    by extracting the longest UUID-shaped substring from each key.
    """
    import re
    uuid_re = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
                         r"-[0-9a-f]{4}-[0-9a-f]{12}")
    out: dict[str, "T"] = {}
    for k, v in d.items():
        m = uuid_re.search(k)
        if m:
            out[m.group(0)] = v
        out[k] = v  # also keep exact form
    return out


def _build_breakdown(results: list[dict], encoder: dict[str, float],
                     prefill: dict[str, dict]) -> list[dict]:
    """Join client raw_results with server-side phase logs by request_id.

    The client captures the proxy's X-Request-Id (a bare uuid). The
    server-side logs use vLLM's internal request_id which prepends
    `chatcmpl-` and may append further mangling. We index server logs
    by the embedded uuid substring so the join is robust to those
    transformations.
    """
    encoder_idx = _index_by_substring(encoder)
    prefill_idx = _index_by_substring(prefill)
    out = []
    for r in results:
        rid = r.get("server_request_id")
        if not rid or not r.get("success") or r.get("ttft") is None:
            continue
        enc_ms = encoder_idx.get(rid, 0.0)
        ph = prefill_idx.get(rid)
        if ph is None:
            # Server-side trace missing for this rid (e.g. log rotated or
            # request still in flight at log read time). Skip.
            continue
        client_ttft_ms = r["ttft"] * 1000.0
        queue_ms = ph["queued_ms"]
        prefill_ms = ph["prefill_ms"]
        # 'other' captures everything we don't have explicit numbers for:
        # client→proxy network, proxy→encoder hop, EC save+load, proxy→
        # prefill hop, prefill→client first chunk send, etc. Can be
        # negative if clocks drift or encoder forward overlaps; floor at 0
        # for stacked plot but keep raw for printing.
        other_ms = client_ttft_ms - enc_ms - queue_ms - prefill_ms
        out.append({
            "request_id": rid,
            "type_id": r.get("type_id"),
            "send_time": r["send_time"],
            "client_ttft_ms": client_ttft_ms,
            "encoder_ms": enc_ms,
            "queue_ms": queue_ms,
            "prefill_ms": prefill_ms,
            "other_ms": other_ms,
            "cache_hit": enc_ms == 0.0,  # no encoder forward → hit
            "phase": r.get("phase", "measure"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--work-dir", type=str, required=True,
                        help="Dir with results_<label>.json + logs/")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--exclude-warmup", action="store_true",
                        default=True,
                        help="Exclude warmup requests from breakdown")
    parser.add_argument("--sample", type=int, default=200,
                        help="Sample size for per-request timeline panel")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import statistics

    work_dir = Path(args.work_dir)
    breakdowns: dict[str, list[dict]] = {}

    print("Parsing per-strategy logs:")
    for key, label, _ in STRATS:
        results_file = work_dir / f"results_{key}.json"
        if not results_file.exists():
            print(f"  {key:<10} no results JSON")
            continue
        with results_file.open() as f:
            data = json.load(f)
        raw = data.get("raw_results") or []
        if not raw:
            print(f"  {key:<10} no raw_results in JSON")
            continue
        encoder, prefill = _parse_logs(work_dir, key)
        if not prefill:
            print(f"  {key:<10} no RequestPhases in prefill log "
                  f"(VLLM_REQUEST_TIMING_TRACE off?)")
            continue
        rows = _build_breakdown(raw, encoder, prefill)
        if args.exclude_warmup:
            rows = [r for r in rows if r["phase"] != "warmup"]
        if not rows:
            continue
        breakdowns[key] = rows
        n_hit = sum(1 for r in rows if r["cache_hit"])
        print(f"  {key:<10} merged={len(rows)} of {len(raw)} requests; "
              f"cache hits in trace = {n_hit} "
              f"({n_hit/len(rows)*100:.1f}%)")

    if not breakdowns:
        raise SystemExit("Nothing to plot — no merged breakdowns")

    # ---------- Figure ----------
    fig, (ax_bar, ax_time) = plt.subplots(2, 1, figsize=(11, 9))

    # Top: stacked-bar of mean phase durations per strategy
    phase_keys = ["encoder_ms", "queue_ms", "prefill_ms", "other_ms"]
    phase_labels = ["Encoder fwd", "Queue (prefill)", "Prefill→1st tok",
                    "Other (transport/EC/network)"]
    phase_colors = ["tab:red", "tab:orange", "tab:purple", "lightgray"]

    strat_keys = [k for k, *_ in STRATS if k in breakdowns]
    strat_pretty = {k: lbl for k, lbl, _ in STRATS}

    means: dict[str, dict[str, float]] = {}
    for k in strat_keys:
        rows = breakdowns[k]
        means[k] = {
            ph: statistics.mean(r[ph] for r in rows)
            for ph in phase_keys
        }
        means[k]["client_ttft_ms"] = statistics.mean(
            r["client_ttft_ms"] for r in rows
        )

    bar_x = list(range(len(strat_keys)))
    bottoms = [0.0] * len(strat_keys)
    for ph, lbl, col in zip(phase_keys, phase_labels, phase_colors):
        # Floor at 0 for stacked plot (other_ms can be negative when
        # encoder runs concurrently with downstream).
        ys = [max(0.0, means[k][ph]) for k in strat_keys]
        ax_bar.bar(bar_x, ys, bottom=bottoms, label=lbl, color=col,
                   edgecolor="black", linewidth=0.4)
        bottoms = [b + y for b, y in zip(bottoms, ys)]

    # Overlay client-observed TTFT (which may differ from sum-of-phases
    # when 'other' is negative/clipped)
    ttfts = [means[k]["client_ttft_ms"] for k in strat_keys]
    ax_bar.scatter(bar_x, ttfts, marker="D", color="black", s=70, zorder=3,
                   label="Client TTFT (mean)")

    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels([strat_pretty[k] for k in strat_keys])
    ax_bar.set_ylabel("Time (ms)")
    ax_bar.set_title("Mean TTFT decomposed into server-side phases")
    ax_bar.legend(loc="upper left", fontsize=8, ncols=2)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Annotate each bar with TTFT total
    for x, t in zip(bar_x, ttfts):
        ax_bar.annotate(f"{t:.0f}ms", xy=(x, t),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")

    # Bottom: per-request timeline (encoder vs other) for one strategy
    # — pick the most populous one with cache active.
    pick = "fifo" if "fifo" in breakdowns else strat_keys[0]
    rows = sorted(breakdowns[pick], key=lambda r: r["send_time"])
    if len(rows) > args.sample:
        step = len(rows) // args.sample
        rows = rows[::step][:args.sample]
    xs = list(range(len(rows)))
    enc = [r["encoder_ms"] for r in rows]
    pf = [r["prefill_ms"] for r in rows]
    q = [r["queue_ms"] for r in rows]
    other = [max(0.0, r["other_ms"]) for r in rows]
    ttft = [r["client_ttft_ms"] for r in rows]

    ax_time.bar(xs, enc, color="tab:red", alpha=0.75, label="Encoder fwd",
                width=1.0, edgecolor="none")
    ax_time.bar(xs, q, bottom=enc, color="tab:orange", alpha=0.75,
                label="Queue", width=1.0, edgecolor="none")
    ax_time.bar(xs, pf, bottom=[a + b for a, b in zip(enc, q)],
                color="tab:purple", alpha=0.75, label="Prefill",
                width=1.0, edgecolor="none")
    ax_time.bar(xs, other, bottom=[a + b + c for a, b, c in zip(enc, q, pf)],
                color="lightgray", alpha=0.6, label="Other",
                width=1.0, edgecolor="none")
    ax_time.plot(xs, ttft, color="black", linewidth=0.8, alpha=0.6,
                 label="Client TTFT")
    ax_time.set_xlabel(f"Request index (sampled, sorted by send time) "
                       f"— strategy={strat_pretty[pick]}")
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_title("Per-request TTFT decomposition over time")
    ax_time.legend(loc="upper right", fontsize=8, ncols=2)
    ax_time.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    out = args.output or str(work_dir / "ttft_breakdown.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")

    # Numerical summary
    print("\nMean phase breakdown per strategy (ms):")
    header = (f"{'strategy':<22} {'TTFT':>7} {'enc':>7} {'queue':>7} "
              f"{'prefill':>7} {'other':>7} {'hit%':>6}")
    print(header)
    print("-" * len(header))
    for k in strat_keys:
        rows = breakdowns[k]
        n_hit = sum(1 for r in rows if r["cache_hit"])
        m = means[k]
        print(f"{strat_pretty[k]:<22} "
              f"{m['client_ttft_ms']:>7.1f} "
              f"{m['encoder_ms']:>7.1f} "
              f"{m['queue_ms']:>7.1f} "
              f"{m['prefill_ms']:>7.1f} "
              f"{m['other_ms']:>7.1f} "
              f"{n_hit/len(rows)*100:>5.1f}%")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
