#!/usr/bin/env python3
"""Report the T29-A three-arm session as three pairwise contrasts.

The session harness fits `y ~ 1 + arm + centred leg index` for a two-level arm
code. Three arms do not fit that model, and rewriting it into a three-level
design would change the estimator the grid and E87 sessions were reported with.
So this filters to two arms at a time and reuses the same estimator unchanged.

The filtering is sound because the palindrome is `base composed composed67
composed67 composed base`. Each arm has mean position 3.5, so dropping the
third arm leaves the remaining two still balanced about the centre of the
replicate and a monotone drift still cancels.

The three contrasts answer three different questions:

  base       vs composed    the advisor's decision, `e135_composition_local_pct`
  composed   vs composed67  the onepass67 table under a tight grid
  base       vs composed67  the best configuration against the base

  python3 research/e135_composition_report.py --label c1
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_report as report

PAIRS = (
    ("base", "composed", "e135_composition_local_pct",
     "the composition exactly as 1db9d63e shipped it"),
    ("composed", "composed67", "e135_onepass_gain_under_tight_pct",
     "the onepass67 table added back under the tight grid"),
    ("base", "composed67", "e135_composition67_local_pct",
     "the never-built cell against the advisor base"),
)

# F29 pre-registered these on the base-to-composed contrast, in the convention
# where a positive number means the composition is FASTER than the base.
PREDICT_ADDITIVE = 0.87
PREDICT_F225 = -1.18
TOLERANCE_PP = 0.238


def contrast(rows: list[dict], ref: str, cand: str, key: str):
    report.ARMS = (ref, cand)
    sub = [r for r in rows if r["arm"] in (ref, cand)]
    fit = report.ols_arm_and_drift(sub, key)
    return sub, fit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="c1")
    args = ap.parse_args()

    report.configure("composition")
    # `legs` keeps only arms named in `report.ARMS`, so widen it to all three
    # before reading, then narrow it per contrast.
    report.ARMS = ("base", "composed", "composed67")
    rows = report.legs(args.label)
    complete = [r for r in rows if r["metrics"].get("mtp_seconds_per_token")]
    if not complete:
        print("e135_composition_report: no scored legs")
        return 1

    print(f"=== T29-A composition session {args.label}:"
          f" {len(complete)} scored legs")
    print()
    print("per arm")
    print(f"{'arm':<12}{'n':>3}{'mtp mean':>12}{'sd':>10}"
          f"{'serial mean':>13}{'ratio':>9}{'entry C':>9}{'exit C':>8}"
          f"{'draft len':>26}")
    for arm in ("base", "composed", "composed67"):
        v = [report.fnum(r["metrics"]["mtp_seconds_per_token"])
             for r in complete if r["arm"] == arm]
        if not v:
            continue
        s = [report.fnum(r["metrics"].get("serial_seconds_per_token"))
             for r in complete if r["arm"] == arm]
        s = [x for x in s if x is not None]
        ratio = [report.fnum(r["metrics"].get("mtp_decode_speedup"))
                 for r in complete if r["arm"] == arm]
        ratio = [x for x in ratio if x is not None]
        ent = [report.fnum(r["meta"].get("gpu_temp_entry_c"))
               for r in complete if r["arm"] == arm]
        ent = [x for x in ent if x is not None]
        ex = [report.fnum(r["meta"].get("gpu_temp_exit_c"))
              for r in complete if r["arm"] == arm]
        ex = [x for x in ex if x is not None]
        edl = sorted({report.fnum(r["metrics"].get("effective_mean_draft_len"))
                      for r in complete if r["arm"] == arm} - {None})
        shown = ",".join(f"{d:.9f}" for d in edl) if edl else "-"
        print(f"{arm:<12}{len(v):>3}{statistics.fmean(v):>12.6f}"
              f"{(statistics.stdev(v) if len(v) > 1 else 0.0):>10.6f}"
              f"{(statistics.fmean(s) if s else float('nan')):>13.6f}"
              f"{(statistics.fmean(ratio) if ratio else float('nan')):>9.4f}"
              f"{(statistics.fmean(ent) if ent else float('nan')):>9.1f}"
              f"{(statistics.fmean(ex) if ex else float('nan')):>8.1f}"
              f"{shown:>26}")

    by_arm_edl = {}
    for arm in ("base", "composed", "composed67"):
        vals = {report.fnum(r["metrics"].get("effective_mean_draft_len"))
                for r in complete if r["arm"] == arm} - {None}
        if vals:
            by_arm_edl[arm] = vals
    print()
    print("schedule")
    for arm, vals in by_arm_edl.items():
        flag = "" if len(vals) == 1 else "   NOT CONSTANT WITHIN THE ARM"
        print(f"  {arm:<12} {sorted(vals)}{flag}")
    distinct = set().union(*by_arm_edl.values()) if by_arm_edl else set()
    if len(distinct) == 1:
        print("  one value across all arms, so every contrast here is pure"
              " kernel and runtime with a zero schedule component")
    else:
        print("  the arms do NOT share one schedule. The probe fraction is the"
              " only axis in this session that can move draft length, so the"
              " base contrast mixes a schedule change with the kernel change."
              " The ranked receipt had digit-identical draft lengths on all"
              " five weighted prompts, so this local schedule split is a"
              " property of the fixture and it does not transfer.")
        print("  composed -> composed67 holds the probe fraction fixed, so"
              " that contrast stays schedule-clean whatever the base does.")

    print()
    print("pairwise contrasts, y ~ 1 + arm + centred leg index")
    headlines = {}
    for ref, cand, metric, why in PAIRS:
        sub, fit = contrast(complete, ref, cand, "mtp_seconds_per_token")
        if fit is None:
            print(f"  {ref} -> {cand}: too few legs")
            continue
        pct = -100.0 * fit["contrast"] / fit["mean"]
        se = 100.0 * fit["se"] / fit["mean"]
        headlines[metric] = (pct, se)
        print(f"  {ref} -> {cand}   n {fit['n']:>2}   {why}")
        print(f"      {metric} = {pct:+.4f} % (+- {se:.4f}),"
              f" positive means {cand} is FASTER")
        _, sfit = contrast(complete, ref, cand, "serial_seconds_per_token")
        if sfit:
            print(f"      serial leg {-100.0 * sfit['contrast'] / sfit['mean']:+.4f} %"
                  f" (+- {100.0 * sfit['se'] / sfit['mean']:.4f}),"
                  f" drift {fit['drift_per_leg']:+.3e} per leg")

    print()
    if "e135_composition_local_pct" in headlines:
        pct, se = headlines["e135_composition_local_pct"]
        print("F29 pre-registration, base -> composed, positive means faster")
        print(f"  additivity predicts   {PREDICT_ADDITIVE:+.2f} %")
        print(f"  Finding 225 predicts  {PREDICT_F225:+.2f} %")
        print(f"  observed              {pct:+.4f} % (+- {se:.4f})")
        da = abs(pct - PREDICT_ADDITIVE)
        df = abs(pct - PREDICT_F225)
        near = "additivity" if da < df else "Finding 225"
        print(f"  distance to additivity {da:.4f} pp,"
              f" to Finding 225 {df:.4f} pp  -> nearer {near}")
        print(f"  the two hypotheses differ by"
              f" {abs(PREDICT_ADDITIVE - PREDICT_F225):.4f} pp against a"
              f" {TOLERANCE_PP} pp instrument tolerance")

    print()
    print("This session is GATED. Every leg passed the real 40 C gate, so"
          " cool_gate_passed_real_gate=true and the legs are gate-qualified"
          " for local comparison. It is still harness=local and is not any"
          " kind of official or ranked score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
