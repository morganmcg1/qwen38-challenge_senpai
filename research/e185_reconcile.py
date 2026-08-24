#!/usr/bin/env python3
"""E185 instrument reconciliation: hardware idle counter against the ledger.

    usage: research/e185_reconcile.py TAG [--skip-rounds N] [--json OUT]

Stage 1 and stage 2 measured the same decode round on two instruments and
disagreed by about three times. This reader needs one leg that carries both,
so the two numbers describe the same rounds rather than two sessions.

The two instruments do not define idle the same way.

* `powermetrics` reports `idle_ns` from a GPU hardware residency counter. A
  gap shorter than the counter's own reaction has no chance to appear, and any
  background GPU work in the same sample hides part of a real gap. It is a
  lower bound on the idle a mechanism could address.
* The E90 ledger reports the complement of the command-buffer execution union.
  The GPU can still be draining or starting work at a buffer boundary, and any
  GPU work outside the instrumented submit path is invisible to it, so its
  complement is an upper bound.

Two comparisons are made:

1. **Per sample.** Each hardware sample window is a window on the mach uptime
   axis, so the ledger idle of that same window is computable. The pair
   (ledger idle, hardware idle) over all samples gives a slope, which is the
   fraction of ledger idle the hardware counter registers.
2. **By gap size.** Ledger gaps are bucketed by duration. The bucket where the
   running total from the largest gap down first reaches the hardware total
   names the shortest gap the counter can be shown to register.

harness=local. Observation only.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e185_gap_table import gaps_in, read_anchors, read_intervals, union  # noqa: E402
from e185_residency import (  # noqa: E402
    NS,
    fit_sample_axis,
    largest_round_block,
    read_meta,
    read_plist_stream,
    read_rounds,
)

BUCKET_EDGES_US = [0.0, 1.0, 5.0, 20.0, 100.0, 300.0, 1000.0, float("inf")]


def bucket_label(us: float) -> str:
    for lo, hi in zip(BUCKET_EDGES_US, BUCKET_EDGES_US[1:]):
        if lo <= us < hi:
            return f"{lo:g}-{hi:g}us" if hi != float("inf") else f">={lo:g}us"
    return "unclassified"


def ols(xs: list[float], ys: list[float]) -> dict:
    n = len(xs)
    if n < 3:
        return {}
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx if sxx else float("nan")
    return {
        "n": n,
        "slope": slope,
        "intercept_us": my - slope * mx,
        "r": sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan"),
        "slope_through_origin": (
            sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
            if any(xs)
            else float("nan")
        ),
    }


def idle_host_duty(path: Path) -> float | None:
    raw = read_plist_stream(path)
    if not raw:
        return None
    elapsed = sum(r[1] for r in raw)
    idle = sum(r[2] for r in raw)
    return 100.0 * (1.0 - idle / elapsed) if elapsed else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--skip-rounds", type=int, default=8)
    ap.add_argument("--json", dest="json_path")
    args = ap.parse_args()

    out = Path(args.tag)
    if not out.exists():
        out = Path("research/out") / args.tag
    meta = read_meta(out / "meta.txt")

    raw = read_plist_stream(out / "powermetrics.plist")
    if not raw:
        print("e185_reconcile: no gpu samples", file=sys.stderr)
        return 2
    pre_wall, pre_mono = meta["e185_clock_pre_wall_mono"].split()
    post_wall, post_mono = meta["e185_clock_post_wall_mono"].split()
    offset_pre = int(round(float(pre_wall) * NS)) - int(pre_mono)
    offset_post = int(round(float(post_wall) * NS)) - int(post_mono)
    offset = (offset_pre + offset_post) // 2
    samples, axis_residual_span = fit_sample_axis(raw, offset)

    block = largest_round_block(read_rounds(out / "trace.txt"))
    if len(block) <= args.skip_rounds + 4:
        print("e185_reconcile: decode block too short", file=sys.stderr)
        return 2
    analysed = block[args.skip_rounds :]
    t0, t1 = analysed[0].t0, analysed[-1].t1
    pid = analysed[0].pid
    n_rounds = len(analysed)

    anchors = [a for a in read_anchors(out / "trace.txt") if a.get("pid") == pid]
    merged = union(read_intervals(out / "gpu-intervals.jsonl", pid))
    if not merged:
        print("e185_reconcile: empty interval ledger", file=sys.stderr)
        return 2

    # 1. Per-sample pairing. Only samples wholly inside the decode block are
    # used, so no sample mixes decode with the seed or the reference work.
    pairs = []
    hw_elapsed = hw_idle = 0
    for s in samples:
        if s.start_mono_ns < t0 or s.end_mono_ns > t1:
            continue
        ledger_idle = s.elapsed_ns - _busy(merged, s.start_mono_ns, s.end_mono_ns)
        pairs.append((ledger_idle / 1000.0, s.idle_ns / 1000.0))
        hw_elapsed += s.elapsed_ns
        hw_idle += s.idle_ns
    if len(pairs) < 10:
        print("e185_reconcile: too few paired samples", file=sys.stderr)
        return 2

    regression = ols([p[0] for p in pairs], [p[1] for p in pairs])
    ledger_idle_paired_us = sum(p[0] for p in pairs)
    hw_idle_paired_us = sum(p[1] for p in pairs)

    # 2. Gap-size buckets over the whole analysed block, including the gaps
    # between rounds, so the two totals cover the same span.
    gaps_us = [(b - a) / 1000.0 for a, b in gaps_in(merged, t0, t1)]
    buckets: dict[str, list[float]] = {}
    for g in gaps_us:
        buckets.setdefault(bucket_label(g), []).append(g)
    bucket_rows = []
    for lo, hi in zip(BUCKET_EDGES_US, BUCKET_EDGES_US[1:]):
        label = f"{lo:g}-{hi:g}us" if hi != float("inf") else f">={lo:g}us"
        vals = buckets.get(label, [])
        bucket_rows.append(
            {
                "bucket": label,
                "gaps": len(vals),
                "gaps_per_round": len(vals) / n_rounds,
                "total_us_per_round": sum(vals) / n_rounds,
                "share_of_ledger_idle_pct": (
                    100.0 * sum(vals) / sum(gaps_us) if gaps_us else None
                ),
            }
        )

    # Running total from the largest gaps down. The hardware counter's total
    # is reached at some cut; gaps shorter than that cut are, at best, what the
    # counter did not see.
    hw_per_round_us = hw_idle / n_rounds / 1000.0
    running = 0.0
    cut_us = None
    for g in sorted(gaps_us, reverse=True):
        running += g
        if running / n_rounds >= hw_per_round_us:
            cut_us = g
            break

    report = {
        "tag": meta.get("tag"),
        "harness": "local",
        "official_or_ranked_score": False,
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "leg_role": "instrument-reconciliation",
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "tokens": meta.get("tokens"),
        "rounds_total": len(block),
        "rounds_skipped": args.skip_rounds,
        "rounds_analysed": n_rounds,
        "decode_block_seconds": (t1 - t0) / NS,
        "axis_residual_span_s": axis_residual_span,
        "clock_offset_drift_ns": offset_post - offset_pre,
        "anchor_rounds_seen": len(anchors),
        "ledger_intervals": len(merged),
        "paired_samples": len(pairs),
        "paired_coverage_pct": 100.0 * hw_elapsed / (t1 - t0),
        "hardware_idle_us_per_round": hw_per_round_us,
        "hardware_idle_pct": 100.0 * hw_idle / hw_elapsed,
        "ledger_idle_us_per_round_paired": ledger_idle_paired_us / n_rounds,
        "ledger_idle_us_per_round_block": sum(gaps_us) / n_rounds,
        "ledger_over_hardware_ratio": (
            ledger_idle_paired_us / hw_idle_paired_us if hw_idle_paired_us else None
        ),
        "per_sample_regression": regression,
        "gap_size_buckets": bucket_rows,
        "hardware_total_reached_at_gap_us": cut_us,
        "background_duty_pct_pre": idle_host_duty(out / "powermetrics-idle-pre.plist"),
        "background_duty_pct_post": idle_host_duty(out / "powermetrics-idle-post.plist"),
    }

    score_path = out / "score.json"
    if score_path.exists():
        metrics = json.loads(score_path.read_text()).get("metrics", {})
        for key in (
            "mtp_seconds_per_token",
            "serial_seconds_per_token",
            "effective_mean_draft_len",
            "accepted_draft_rate",
            "all_tokens_matched",
            "decode_tokens",
        ):
            if key in metrics:
                report[key] = metrics[key]

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n")
    return 0


def _busy(merged: list[tuple[int, int]], t0: int, t1: int) -> int:
    total = 0
    for a, b in merged:
        if b <= t0:
            continue
        if a >= t1:
            break
        total += min(b, t1) - max(a, t0)
    return total


if __name__ == "__main__":
    raise SystemExit(main())
