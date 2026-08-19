#!/usr/bin/env python3
"""E46 pre-registration. Committed before the first timed run of this experiment.

E41 fitted `T(M) = c + b*streams(M) + a*M` with b = 20.291 ms per marginal weight
stream, on the NA<=5 table at `04ad6bf1`. That table has ONE stream boundary
(5->6), so b is carried by one contrast, and at that contrast the stream count
and the widest group width move together: `<T,5,5>` -> `<T,6,3>` is 1->2 streams
AND 5 rows -> 3 rows. The name on the mechanism is therefore not identified.

E46 identifies it twice more, at FIXED M, where the `a*M` term cancels exactly:

  contrast A   M=6, IPG 3 -> 4     streams 2 -> 2      group width 3 -> {4,2}
  contrast B   M=8, IPG 4 -> 3     streams 2 -> 3      group width 4 -> {3,2}

and re-measures the width curve on the shipped NA<=4 table, whose stream vector
[1,1,2,2,2,2,3] puts boundaries at 4->5 and 8->9 and has NO boundary at 5->6.

Everything here is fixed before the GPU runs. `research/e46_analyze.py` applies
these constants and rules to the measured curves without re-deriving any of them.

  python3 research/e46_prereg.py --out research/e46-prereg.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
WIDTHS = list(range(3, 10))

# --- provenance ---------------------------------------------------------------
BASE_SHA = "01f69e18f3878c9565fee479581581d85cf481ce"
E41_BASE_SHA = "04ad6bf11437c269df85a47e91faa769c74fe6da"
# senpai/verify-kernel-table.sh table 04ad6bf1 01f69e18, verbatim.
TABLES = {
    "04ad6bf": {"ipg": [3, 4, 5, 3, 4, 4, 5], "streams": [1, 1, 1, 2, 2, 2, 2],
                "boundaries": ["5->6"], "na_ceiling": 5, "twin_equals_header": True},
    "01f69e1": {"ipg": [3, 4, 3, 3, 4, 4, 3], "streams": [1, 1, 2, 2, 2, 2, 3],
                "boundaries": ["4->5", "8->9"], "na_ceiling": 4,
                "twin_equals_header": True},
}

# --- E41 coefficients, on E41's table -----------------------------------------
# T(M) = C + B*streams(M) + A*M, max|residual| 1.674 ms over 72.8-164.7 ms.
E41_A = 11.798   # ms per input row
E41_B = 20.291   # ms per marginal weight stream  <- the quantity under test
E41_C = 16.432

# --- step 2: predicted curve on the SHIPPED stream vector ---------------------
# Levels do NOT transfer: the shared allocation dropped 129 -> 108 between the
# two trees, so every absolute time moves. The registered test is the PATTERN of
# the first difference, which is invariant to any additive constant and robust
# to a uniform occupancy factor.
PRED_T = [round(E41_C + E41_B * s + E41_A * m, 3)
          for m, s in zip(WIDTHS, TABLES["01f69e1"]["streams"])]
PRED_D1 = [round(PRED_T[i + 1] - PRED_T[i], 3) for i in range(len(PRED_T) - 1)]
D1_LABELS = ["%d->%d" % (m, m + 1) for m in WIDTHS[:-1]]
# For contrast, E41's own measured d1 on the NA<=5 table (base-r2).
E41_MEASURED_D1 = [9.911, 13.495, 32.673, 9.827, 11.010, 14.948]

HYPOTHESES = {
    "H_streams": {
        "claim": "T is set by ceil(M/IPG), the number of threadgroups that read "
                 "the full weight tile; +20.291 ms per marginal stream",
        "step2": "argmax d1 in {4->5, 8->9}, both ~32.089; the other four ~11.798",
        "contrast_A": "delta = 0 (2 streams both arms)",
        "contrast_B": "delta = +20.291 ms (2 -> 3 streams)",
    },
    "H_groupwidth": {
        "claim": "T is set by the widest group's row count / group balance",
        "step2": "no specific prediction; the shipped table's widest group is "
                 "4 at M=4,7,8 and 3 elsewhere, so d1 would not be flat",
        "contrast_A": "delta > 0 (widest group 3 -> 4 rows, split 3+3 -> 4+2)",
        "contrast_B": "delta < 0 (widest group 4 -> 3 rows)",
    },
    "H_M6breakpoint": {
        "claim": "E43's fitted 36.278*[M>=6] indicator, a property of the width "
                 "itself rather than of the table",
        "step2": "argmax d1 = 5->6",
        "contrast_A": "delta = 0 (M fixed)",
        "contrast_B": "delta = 0 (M fixed)",
    },
}

# --- decision rules -----------------------------------------------------------
# n=2 per build, ABBA-counterbalanced (base, arm, arm, base) so the two arms
# share a mean sweep position of 2.5 and linear drift cancels exactly. With n=2
# the honest floor is the observed same-build spread, not a t-statistic.
MDE_RULE = ("MDE(M) = max(|T_base1 - T_base2|, |T_arm1 - T_arm2|) at that width; "
            "a signed effect requires |delta(M)| > MDE(M)")
SIGN_TEST = ("secondary, distribution-free: the per-shape sign of delta over the "
             "8 scored shapes; 8/8 agreement is p = 0.0078 two-sided")
# Contrast B is a level, and levels move between trees, so it is scored in bands
# around the E41 coefficient rather than against a point value.
B_BAND_STRICT = (0.75 * E41_B, 1.25 * E41_B)
B_BAND_LENIENT = (0.50 * E41_B, 1.50 * E41_B)

STOP_RULES = [
    "1. step 2's argmax d1 is 5->6  -> the stream reading is falsified; stop",
    "2. |delta_A| > MDE(6), or delta_B outside the lenient band -> the mechanism "
    "is renamed; stop and report",
    "3. any arm's measured kernel-wide register max exceeds 108 -> stop",
    "4. measured r=2 tax at NA=5 >= 21.20% -> the NA=5 axis closes; stop",
]
NOT_SHIPPED = ("contrast B makes M=8 worse on purpose and contrast A is a probe; "
               "the branch's scored surface must return to zero diff vs BASE_SHA")

# --- measurement configuration, fixed here so it cannot drift -----------------
RUN_CONFIG = {
    "harness": "research/run-qmv-curve.sh",
    "flags": "--widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 "
             "--skip-stock",
    "order": ["e46-base-r1", "e46-arm-r1", "e46-arm-r2", "e46-base-r2"],
    "metric": "T(M) = sum over the 8 scored shapes of calls_per_verify * "
              "seconds_per_call, in ms",
    "arm_edit": "case 6: <T,6,3,true> -> <T,6,4,true>; case 8: <T,8,4,true> -> "
                "<T,8,3,true>, in quantized.h AND its generated twin",
    "controls": "widths 1,2,3,4,5,7,9 are byte-identical between the builds",
    "thermal": "this host's real 40C gate is unreachable (~43.4C floor); every "
               "arm records entry/exit GPU temperature and preserves "
               "cool_gate_passed_real_gate=false, gate_qualified_for_timing=false",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e46-prereg.json")
    args = ap.parse_args()

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    payload = {
        "experiment": "qwen38-r1-e46-stream-vs-groupwidth-fixed-m",
        "head_at_preregistration": head,
        "base_sha": BASE_SHA,
        "e41_base_sha": E41_BASE_SHA,
        "widths": WIDTHS,
        "tables": TABLES,
        "e41_coefficients": {"a_per_row": E41_A, "b_per_stream": E41_B,
                             "c_intercept": E41_C},
        "predicted_T_ms": dict(zip(map(str, WIDTHS), PRED_T)),
        "predicted_d1_ms": dict(zip(D1_LABELS, PRED_D1)),
        "e41_measured_d1_on_na5_table": dict(zip(D1_LABELS, E41_MEASURED_D1)),
        "hypotheses": HYPOTHESES,
        "mde_rule": MDE_RULE,
        "sign_test": SIGN_TEST,
        "contrast_B_band_strict_ms": list(B_BAND_STRICT),
        "contrast_B_band_lenient_ms": list(B_BAND_LENIENT),
        "stop_rules": STOP_RULES,
        "not_shipped": NOT_SHIPPED,
        "run_config": RUN_CONFIG,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("E46 pre-registration   head=%s" % head[:8])
    print("\ntable at BASE_SHA 01f69e1: IPG %s  streams %s  boundaries %s"
          % (TABLES["01f69e1"]["ipg"], TABLES["01f69e1"]["streams"],
             TABLES["01f69e1"]["boundaries"]))
    print("table at E41 base 04ad6bf: IPG %s  streams %s  boundaries %s"
          % (TABLES["04ad6bf"]["ipg"], TABLES["04ad6bf"]["streams"],
             TABLES["04ad6bf"]["boundaries"]))
    print("\nM            " + "".join("%9d" % m for m in WIDTHS))
    print("predicted T  " + "".join("%9.3f" % t for t in PRED_T))
    print("\nstep         " + "".join("%9s" % s for s in D1_LABELS))
    print("predicted d1 " + "".join("%9.3f" % d for d in PRED_D1))
    print("E41 d1 (NA<=5)" + "".join("%9.3f" % d for d in E41_MEASURED_D1))
    print("\nH_streams      argmax d1 in {4->5, 8->9}")
    print("H_M6breakpoint argmax d1 = 5->6")
    print("\ncontrast A  M=6 IPG 3->4: H_streams 0, H_groupwidth > 0")
    print("contrast B  M=8 IPG 4->3: H_streams +%.3f ms  strict [%.3f, %.3f]  "
          "lenient [%.3f, %.3f]"
          % (E41_B, *B_BAND_STRICT, *B_BAND_LENIENT))
    print("\n" + MDE_RULE)
    print(SIGN_TEST)
    for rule in STOP_RULES:
        print(rule)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
