#!/usr/bin/env python3
"""Price one launched threadgroup column against the real width distribution.

The pipeline log's `by_width` is a WARM census, not decode traffic. Its
`first_index_by_width` runs in sequential blocks of 257, so the warm phase
walks each legal width once in order and the counts say nothing about how
often a drafting round actually verifies at that width.

257 is the number of QMV dispatches in one forward pass: 4 per layer across 64
layers plus one vocabulary readout. One drafting round runs one verify forward
pass at width `d + 1`, so

    columns per round = 257 * columns_by_width[d + 1]

averaged over the run's own per-round draft depths, which the trace records as
`d=`. Draft length is deterministic on this fixture, so one traced leg fixes
the distribution for every arm in the session.

    python3 research/e135_width_histogram.py --label c1
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_report as report

ARMS = ("base", "composed", "composed67")
QMV_DISPATCHES_PER_FORWARD_PASS = 257
PAIRS = (
    ("base", "composed", 1.9394, 0.0257),
    ("composed", "composed67", 0.1912, 0.0253),
    ("base", "composed67", 2.1305, 0.0226),
)


def depth_histogram(trace: pathlib.Path) -> collections.Counter:
    text = trace.read_text()
    depths = [int(x) for x in re.findall(r"round=\d+ d=(\d+)", text)]
    return collections.Counter(d + 1 for d in depths)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="c1")
    ap.add_argument("--trace", default="research/out/e87t-g128-2/trace.txt")
    args = ap.parse_args()

    hist = depth_histogram(pathlib.Path(args.trace))
    total = sum(hist.values())
    print(f"verify width histogram over {total} rounds, from {args.trace}")
    for w in sorted(hist):
        print(f"  width {w}  {hist[w]:>3} rounds  {100.0 * hist[w] / total:5.1f} %")
    print(f"  mean verify width {sum(w * n for w, n in hist.items()) / total:.4f}")

    cols = {}
    for arm in ARMS:
        path = pathlib.Path(f"research/out/e135{args.label}w{arm}/pipelines.json")
        cols[arm] = json.loads(path.read_text())["columns_by_width"]

    per_round = {}
    for arm in ARMS:
        acc = 0
        for w, n in hist.items():
            # An unrouted width falls back to MLX, which launches one column
            # per row exactly as the wide grid does.
            columns = int(cols[arm].get(str(w), w))
            acc += n * QMV_DISPATCHES_PER_FORWARD_PASS * columns
        per_round[arm] = acc / total
    print()
    print("columns launched per drafting round")
    for arm in ARMS:
        print(f"  {arm:<12}{per_round[arm]:>10.1f}")

    report.configure("composition")
    report.ARMS = ARMS
    rows = [r for r in report.legs(args.label)
            if r["metrics"].get("mtp_seconds_per_token")]
    mean = {}
    for arm in ARMS:
        v = [report.fnum(r["metrics"]["mtp_seconds_per_token"])
             for r in rows if r["arm"] == arm]
        if v:
            mean[arm] = statistics.fmean(v)

    print()
    print(f"{'contrast':<26}{'saved/round':>12}{'pct':>10}{'us/column':>18}")
    out = []
    for ref, cand, pct, se in PAIRS:
        round_s = mean[ref] * 512 / 78
        saved = per_round[ref] - per_round[cand]
        us = pct / 100.0 * round_s * 1e6 / saved
        us_se = se / 100.0 * round_s * 1e6 / saved
        out.append((f"{ref}->{cand}", saved, pct, us, us_se))
        print(f"  {ref + ' -> ' + cand:<24}{saved:>12.1f}{pct:>+10.4f}"
              f"{us:>12.4f} +- {us_se:.4f}")

    noop = next(o for o in out if o[0] == "base->composed")
    working = next(o for o in out if o[0] == "composed->composed67")
    print()
    print("The two contrasts that share no arm DISAGREE by"
          f" {working[3] / noop[3]:.1f}x, so the effect is not linear in"
          " launched columns and the two column kinds must be priced apart.")
    print(f"  a no-op column the tight grid deletes  {noop[3]:.4f}"
          f" +- {noop[4]:.4f} us")
    print(f"  a column that reads the whole weight matrix"
          f"  {working[3]:.4f} +- {working[4]:.4f} us")
    share = 100.0 * sum(hist[w] for w in (6, 7)) / total
    print(f"  onepass67 only touches widths 6 and 7, which are {share:.2f} %"
          " of local rounds, because width 8 takes 76.9 % of this fixture")

    artifacts = pathlib.Path("research/e135-artifacts")
    artifacts.mkdir(exist_ok=True)
    (artifacts / f"{args.label}-per-width.json").write_text(json.dumps({
        "qmv_dispatches_per_forward_pass": QMV_DISPATCHES_PER_FORWARD_PASS,
        "verify_width_histogram": {str(w): hist[w] for w in sorted(hist)},
        "columns_per_round": per_round,
        "estimates": [{"contrast": c, "columns_saved": s, "pct": p,
                       "us_per_column": u, "us_per_column_se": e}
                      for c, s, p, u, e in out],
        "e135_launch_cost_us_per_column": noop[3],
        "e135_launch_cost_us_per_column_se": noop[4],
        "e135_working_column_us": working[3],
        "e135_working_column_us_se": working[4],
        "e135_verify_width_mean":
            sum(w * n for w, n in hist.items()) / total,
        "e135_width_6_7_round_share_pct": share,
    }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
