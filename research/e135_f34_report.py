#!/usr/bin/env python3
"""Report the F34 three-arm palindrome as three pairwise contrasts.

The session harness fits `y ~ 1 + arm + centred leg index` for a two-level arm
code, so this filters to two arms at a time and reuses that estimator
unchanged, exactly as the T29-A reporter does. The palindrome

    c67ship c67pb6 c678pb6 c678pb6 c67pb6 c67ship

gives every arm mean position 3.5, so dropping the third arm leaves the
remaining two balanced about the centre and a monotone drift still cancels.

    python3 research/e135_f34_report.py --label f34
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_composition_report as composition
import e135_report as report

ARMS = ("c67ship", "c67pb6", "c678pb6")

PAIRS = (
    ("c67ship", "c67pb6", "e135_pb6_under_tight_pct",
     "the pb6 depth price under the tight launch grid"),
    ("c67pb6", "c678pb6", "e135_onepass678_local_pct",
     "onePass678 on top of the shipping schedule"),
    ("c67ship", "c678pb6", "e135_f34_full_move_pct",
     "both restorations together against the ship schedule"),
)

# T45 pre-registered onePass678 from the working-column price. Under tight the
# table takes width 8 from 2 launched columns to 1, so it saves
# `P(width 8) * 257` columns per round at 13.8715 us each. F34 section 5
# withdraws the physical story behind that price and keeps the number, so the
# forecast is a linear extension of a contrast measured over 26.4 columns per
# round to one over about 198, and a large signed gap is information about the
# column model rather than an error.
PREDICT_ONEPASS678 = 1.434

# F34 decision thresholds, in the convention where a positive number means the
# SECOND arm is faster.
PB6_STOP_IF_WORSE_THAN = -0.50
ONEPASS678_INCLUDE_AT = 0.50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="f34")
    args = ap.parse_args()

    report.configure("composition")
    report.ARMS = ARMS
    rows = report.legs(args.label)
    complete = [r for r in rows if r["metrics"].get("mtp_seconds_per_token")]
    if not complete:
        print("e135_f34_report: no scored legs")
        return 1

    print(f"=== F34 session {args.label}: {len(complete)} scored legs")
    print()
    print("per arm")
    print(f"{'arm':<12}{'n':>3}{'mtp mean':>12}{'sd':>10}"
          f"{'serial mean':>13}{'ratio':>9}{'entry C':>9}{'exit C':>8}"
          f"{'draft len':>26}")
    for arm in ARMS:
        v = [report.fnum(r["metrics"]["mtp_seconds_per_token"])
             for r in complete if r["arm"] == arm]
        if not v:
            continue
        s = [x for x in (report.fnum(r["metrics"].get(
            "serial_seconds_per_token")) for r in complete
            if r["arm"] == arm) if x is not None]
        ratio = [x for x in (report.fnum(r["metrics"].get(
            "mtp_decode_speedup")) for r in complete
            if r["arm"] == arm) if x is not None]
        ent = [x for x in (report.fnum(r["meta"].get("gpu_temp_entry_c"))
                           for r in complete if r["arm"] == arm)
               if x is not None]
        ex = [x for x in (report.fnum(r["meta"].get("gpu_temp_exit_c"))
                          for r in complete if r["arm"] == arm)
              if x is not None]
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

    print()
    print("per leg")
    print(f"{'i':>3}{'tag':<26}{'arm':<10}{'mtp s/tok':>12}"
          f"{'serial s/tok':>14}{'rounds':>8}{'entry C':>9}{'exit C':>8}")
    for i, r in enumerate(complete):
        m = report.fnum(r["metrics"]["mtp_seconds_per_token"])
        s = report.fnum(r["metrics"].get("serial_seconds_per_token"))
        n = composition.rounds(r)
        print(f"{i:>3}{r['tag']:<26}{r['arm']:<10}{m:>12.6f}"
              f"{(s if s is not None else float('nan')):>14.6f}"
              f"{(n if n is not None else -1):>8}"
              f"{report.fnum(r['meta'].get('gpu_temp_entry_c')) or 0.0:>9.1f}"
              f"{report.fnum(r['meta'].get('gpu_temp_exit_c')) or 0.0:>8.1f}")

    print()
    print("exactness")
    matched = {r["metrics"].get("all_tokens_matched") for r in complete}
    diverged = {r["metrics"].get("residual_divergence_count")
                for r in complete}
    print(f"  all_tokens_matched         {sorted(map(str, matched))}")
    print(f"  residual_divergence_count  {sorted(map(str, diverged))}")

    print()
    print("gate")
    real = {r["meta"].get("cool_gate_passed_real_gate") for r in complete}
    qual = {r["meta"].get("gate_qualified_for_timing") for r in complete}
    print(f"  cool_gate_passed_real_gate {sorted(map(str, real))}")
    print(f"  gate_qualified_for_timing  {sorted(map(str, qual))}")

    print()
    print("provenance")
    for key in ("base_sha", "worker_sha256", "post_run_worker_sha256",
                "dirty_candidate_paths", "e135_session_commit"):
        seen = sorted({str(r["meta"].get(key)) for r in complete})
        print(f"  {key:<26} {seen}")

    print()
    print("contrasts, y ~ 1 + arm + centred leg index."
          " Positive means the SECOND arm is FASTER.")
    results = {}
    for ref, cand, key, why in PAIRS:
        sub, fit = composition.contrast(complete, ref, cand,
                                        "mtp_seconds_per_token")
        if fit is None:
            print(f"  {ref} -> {cand}: not enough legs")
            continue
        pct = -100.0 * fit["contrast"] / fit["mean"]
        se = 100.0 * fit["se"] / fit["mean"]
        results[key] = (pct, se)
        print(f"  {ref} -> {cand}   n {fit['n']:>2}   {why}")
        print(f"      {key} = {pct:+.4f} % (+- {se:.4f})")
        _, sfit = composition.contrast(complete, ref, cand,
                                       "serial_seconds_per_token")
        if sfit:
            print("      serial-leg null"
                  f" {-100.0 * sfit['contrast'] / sfit['mean']:+.4f} %"
                  f" (+- {100.0 * sfit['se'] / sfit['mean']:.4f}),"
                  f" drift {fit['drift_per_leg']:+.3e} per leg")

    print()
    print("decisions")
    pb6 = results.get("e135_pb6_under_tight_pct")
    if pb6 is not None:
        verdict = ("PROCEED" if pb6[0] >= PB6_STOP_IF_WORSE_THAN
                   else "STOP, do not submit")
        print(f"  pb6 under tight  {pb6[0]:+.4f} % (+- {pb6[1]:.4f})"
              f"  threshold {PB6_STOP_IF_WORSE_THAN:+.2f} %  -> {verdict}")
    op = results.get("e135_onepass678_local_pct")
    if op is not None:
        include = ("INCLUDE onePass678" if op[0] >= ONEPASS678_INCLUDE_AT
                   else "HOLD onePass678 as a finding")
        print(f"  onePass678       {op[0]:+.4f} % (+- {op[1]:.4f})"
              f"  threshold {ONEPASS678_INCLUDE_AT:+.2f} %  -> {include}")
        print(f"  pre-registered   {PREDICT_ONEPASS678:+.4f} %"
              f"  signed gap {op[0] - PREDICT_ONEPASS678:+.4f} pp")
        print("  The forecast extends a price measured over 26.4 columns per"
              " round to one over about 198 columns per round, so treat a"
              " large signed gap as information about the column model.")

    print()
    print("This session is GATED and harness=local. It is not any kind of"
          " official or ranked score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
