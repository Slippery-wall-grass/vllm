#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hit-rate summary for the CLEAN-tree L3 disagg setup.

The upstream `aggregate.py` parses the fork L2 policy line
(`encoder_cache policy=... hit_rate=R`), which the clean EncoderCacheManager
never emits -> its "Cache hit rate" table comes out empty. This sidecar parses
the three hit signals that DO exist in the clean tree and writes hit_rate.md:

  * skip-E hit rate  : proxy `[SkipE] ... items=N hit(skip)=S encode=E`
                       (fraction of mm items whose whole E was skipped)
  * L3 read hit rate : engine `[L3 stats] load-hit=H miss=M` (recoverable-miss
                       rate on the L3 load path) + final save/evict/used
  * MM content hit   : PD `MM cache hit rate: X%` (L1/L2 content-hash cache)

Aggregate over the whole arm (all concurrency steps pooled). Per-concurrency
segmentation is future work (needs mapping bench request-id prefixes to runs).
"""
import argparse
import glob
import re
from pathlib import Path


def _skipe(proxy_log: Path) -> dict:
    tot = skip = 0
    if proxy_log.exists():
        pat = re.compile(r"\[SkipE\].*items=(\d+) hit\(skip\)=(\d+) encode=(\d+)")
        for line in proxy_log.open(errors="ignore"):
            m = pat.search(line)
            if m:
                it, sk, _ = map(int, m.groups())
                tot += it
                skip += sk
    return {"items": tot, "skipped": skip,
            "rate": (skip / tot) if tot else None}


def _l3(engine_logs: list[Path]) -> dict:
    load_hit = miss = 0
    save = evict = used_mb = cap_mb = entries = 0
    load_avg = save_avg = None
    hit_pat = re.compile(
        r"\[L3 stats\] load-hit=(\d+) miss=(\d+)"
        r"(?: load_avg=([\d.]+)ms)?")
    save_pat = re.compile(
        r"\[L3 stats\] save=(\d+) evict=(\d+) hit=\d+ miss=\d+ "
        r"used=(\d+)MB/(\d+)MB entries=(\d+)(?: save_avg=([\d.]+)ms)?")
    for lg in engine_logs:
        if not lg.exists():
            continue
        for line in lg.open(errors="ignore"):
            m = hit_pat.search(line)
            if m:
                load_hit = max(load_hit, int(m.group(1)))
                miss = max(miss, int(m.group(2)))
                if m.group(3):
                    load_avg = float(m.group(3))
            m = save_pat.search(line)
            if m:
                save = max(save, int(m.group(1)))
                evict = max(evict, int(m.group(2)))
                used_mb, cap_mb, entries = (int(m.group(3)), int(m.group(4)),
                                            int(m.group(5)))
                if m.group(6):
                    save_avg = float(m.group(6))
    total = load_hit + miss
    return {"load_hit": load_hit, "miss": miss,
            "read_rate": (load_hit / total) if total else None,
            "save": save, "evict": evict, "used_mb": used_mb,
            "cap_mb": cap_mb, "entries": entries,
            "load_avg_ms": load_avg, "save_avg_ms": save_avg}


def _mm(pd_log: Path) -> dict:
    vals = []
    if pd_log.exists():
        pat = re.compile(r"MM cache hit rate: ([\d.]+)%")
        for line in pd_log.open(errors="ignore"):
            m = pat.search(line)
            if m:
                vals.append(float(m.group(1)))
    return {"rate": (sum(vals) / len(vals) / 100.0) if vals else None,
            "n_samples": len(vals)}


def _fmt(x, pct=False, suf=""):
    if x is None:
        return "—"
    return f"{x*100:.1f}%" if pct else f"{x:g}{suf}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--policy", default="fifo",
                    help="log basename prefix (e.g. fifo)")
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args()

    rd = Path(a.result_dir)
    proxy = rd / f"{a.policy}.proxy.log"
    pd = rd / f"{a.policy}.pd.0.log"
    engines = [Path(p) for p in glob.glob(str(rd / f"{a.policy}.*.log"))]

    se, l3, mm = _skipe(proxy), _l3(engines), _mm(pd)

    lines = [
        "# Cache / skip hit rates (clean-tree L3 disagg)\n",
        "Aggregate over all concurrency steps in this arm.\n",
        "| signal | value | detail |",
        "|---|---|---|",
        f"| **skip-E hit** (E fully skipped) | **{_fmt(se['rate'], pct=True)}** "
        f"| {se['skipped']}/{se['items']} mm items |",
        f"| **L3 read hit** (load path, 1-miss) | {_fmt(l3['read_rate'], pct=True)} "
        f"| load-hit={l3['load_hit']} miss={l3['miss']} |",
        f"| **MM content hit** (L1/L2 hash) | {_fmt(mm['rate'], pct=True)} "
        f"| {mm['n_samples']} samples |",
        "",
        "## L3 cache occupancy / transfer",
        "| save | evict | used | cap | entries | load_avg | save_avg |",
        "|---|---|---|---|---|---|---|",
        f"| {l3['save']} | {l3['evict']} | {l3['used_mb']}MB | {l3['cap_mb']}MB "
        f"| {l3['entries']} | {_fmt(l3['load_avg_ms'], suf='ms')} "
        f"| {_fmt(l3['save_avg_ms'], suf='ms')} |",
        "",
    ]
    out = "\n".join(lines)
    print(out)
    dest = Path(a.output_md) if a.output_md else rd / "hit_rate.md"
    dest.write_text(out, encoding="utf-8")
    print(f"[hit_rate] wrote {dest}")


if __name__ == "__main__":
    main()
