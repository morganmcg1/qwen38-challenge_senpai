#!/usr/bin/env python3
"""Pre-registration for E54: lone-vs-sibling NA=5 law.

Emits every timing prediction from an explicit model so the arithmetic is
auditable before any GPU second is spent.  Run:

    python3 research/e54_prereg.py --out research/e54-artifacts/e54-prereg.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------- dispatch ---
# Shipped inner-product-group width per decode width, read from the scored
# dispatch table at BASE_SHA (Vendor/.../quantized.h qmv_quad / qmv_fast tree).
# Each pair swaps one `case` label to NA=5 in an isolated single-case build.
PAIRS = {
    "P1": {"m": 5, "shipped_ipg": 3, "cand_ipg": 5},
    "P2": {"m": 7, "shipped_ipg": 4, "cand_ipg": 5},
    "P3": {"m": 8, "shipped_ipg": 4, "cand_ipg": 5},
    "M9": {"m": 9, "shipped_ipg": 3, "cand_ipg": 5},  # measured in E49
}
M9_MEASURED_PCT = -12.255  # E49 isolated Arm 1, <T,9,3> -> <T,9,5>

# E46 refit of isolated-build block latency (us):
#   T(M) = 16.757 + 27.532 * working_groups + 9.624 * M
E46_FIT = {"const": 16.757, "per_group": 27.532, "per_m": 9.624}

# PR #8 effective weight-read bandwidth by accumulator width (GB/s).
BW_NA5 = 95.5
BW_NA_LE4 = 165.6

# Pre-registered decision bar (percent).  Sources, all from E49 on this host:
#   same-arm leg-to-leg noise: 0.066..0.377 % mean, 0.770 % max
#   control-width |delta| in isolated builds: 0.00..0.88 %
#   replicate spread at the treated width: 0.107..1.504 ms (0.18 % at M=9)
BAR_PCT = 0.770


def groups(m: int, ipg: int) -> int:
    """Working groups = ceil(m / ipg).  Every group re-reads the whole matrix."""
    return -(-m // ipg)


def split(m: int, ipg: int) -> list[int]:
    """Accumulator width actually used by each working group."""
    out = []
    left = m
    while left > 0:
        out.append(min(ipg, left))
        left -= out[-1]
    return out


def e46_time(m: int, ipg: int) -> float:
    return (
        E46_FIT["const"]
        + E46_FIT["per_group"] * groups(m, ipg)
        + E46_FIT["per_m"] * m
    )


def bw_time(m: int, ipg: int, theta: float) -> float:
    """Weight-read time in units of (matrix bytes).

    theta interpolates between two extremes of how the NA=5 register-pressure
    penalty interacts with the weight stream:
      theta = 0  serialized: each group pays its own NA-specific bandwidth
      theta = 1  fully overlapped: every group reads at the NA<=4 rate
    """
    ser = sum(1.0 / (BW_NA5 if na == 5 else BW_NA_LE4) for na in split(m, ipg))
    ovl = groups(m, ipg) / BW_NA_LE4
    return (1.0 - theta) * ser + theta * ovl


def pct(cand: float, base: float) -> float:
    return (cand - base) / base * 100.0


def solve_theta() -> float:
    """Calibrate theta so the bandwidth model reproduces the measured M=9."""
    m, sh, ca = PAIRS["M9"]["m"], PAIRS["M9"]["shipped_ipg"], PAIRS["M9"]["cand_ipg"]
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d = pct(bw_time(m, ca, mid), bw_time(m, sh, mid))
        if d > M9_MEASURED_PCT:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build() -> dict:
    theta = solve_theta()

    # E46 refit calibration: scale raw refit deltas so M=9 matches measurement.
    m9 = PAIRS["M9"]
    raw_m9 = pct(e46_time(m9["m"], m9["cand_ipg"]), e46_time(m9["m"], m9["shipped_ipg"]))
    calib = M9_MEASURED_PCT / raw_m9

    cells = {}
    for name, p in PAIRS.items():
        m, sh, ca = p["m"], p["shipped_ipg"], p["cand_ipg"]
        raw = pct(e46_time(m, ca), e46_time(m, sh))
        cells[name] = {
            "pair": name,
            "width": m,
            "shipped_template": f"<T,{m},{sh}>",
            "candidate_template": f"<T,{m},{ca}>",
            "shipped_split": split(m, sh),
            "candidate_split": split(m, ca),
            "shipped_groups": groups(m, sh),
            "candidate_groups": groups(m, ca),
            "groups_change": groups(m, ca) - groups(m, sh),
            "pred_pct": {
                "working_group_law_raw": round(raw, 3),
                "working_group_law_calibrated": round(raw * calib, 3),
                "bandwidth_serialized": round(
                    pct(bw_time(m, ca, 0.0), bw_time(m, sh, 0.0)), 3
                ),
                "bandwidth_overlapped": round(
                    pct(bw_time(m, ca, 1.0), bw_time(m, sh, 1.0)), 3
                ),
                "bandwidth_theta_calibrated": round(
                    pct(bw_time(m, ca, theta), bw_time(m, sh, theta)), 3
                ),
            },
            "measured_pct": M9_MEASURED_PCT if name == "M9" else None,
        }

    return {
        "experiment": "E54",
        "assignment_id": "qwen38-r1-e54-lone-vs-sibling-na5-law",
        "revision_id": "r1",
        "pr": 58,
        "base_sha": "a35bb006fd47785dc916241df63ec8780bda8e5c",
        "orientation": (
            "delta_pct = (candidate_NA5 - shipped_IPG) / shipped_IPG * 100 at the "
            "treated width; negative means the NA=5 build is faster"
        ),
        "decision_bar_pct": BAR_PCT,
        "decision_bar_rule": (
            "max(0.770, worst in-session control-width |delta|, treated-width "
            "replicate spread) - whichever is largest wins"
        ),
        "theta_calibrated": round(theta, 4),
        "e46_calibration_factor": round(calib, 4),
        "cells": cells,
        "laws": {
            "A_as_written_in_brief": {
                "claim": "every NA=5 swap wins; M=5 wins most",
                "predicts": {"P1": "< -12 %", "P2": "< 0 %", "P3": "< 0 %"},
                "falsified_if": "P2 or P3 fails to win by more than the bar",
            },
            "A_prime_working_group_traffic": {
                "claim": (
                    "weight traffic is proportional to working groups = ceil(M/IPG); "
                    "only a swap that removes a group can win"
                ),
                "predicts": {
                    "P1": "-23 % .. -13 % (group count 2 -> 1)",
                    "P2": "null within the bar (2 -> 2)",
                    "P3": "null within the bar (2 -> 2)",
                },
                "falsified_if": (
                    "P1 |delta| < bar, or P2 or P3 wins by more than 3 %"
                ),
            },
            "B_na5_always_loses": {
                "claim": "NA=5 register pressure always costs more than it saves",
                "predicts": {"P1": "> +bar", "P2": "> +bar", "P3": "> +bar"},
                "status": "already falsified at M=9 by E49 (-12.255 %)",
                "falsified_if": "any pair wins by more than the bar",
            },
            "C_sibling_overlap_advisor": {
                "claim": (
                    "a lone NA=5 group has no NA<=4 sibling to hide its latency, so "
                    "the lone cell regresses while mixed cells win"
                ),
                "predicts": {
                    "P1": "> +bar (regression)",
                    "P2": "-12 % +/- 3",
                    "P3": "-12 % +/- 3",
                },
                "falsified_if": (
                    "P1 wins by more than the bar, or P2/P3 do not win by >= 3 %"
                ),
                "requires_for_P1_regression": (
                    "the lone NA=5 group must sustain < 82.8 GB/s, which is 13 % "
                    "below the 95.5 GB/s measured for NA=5 in PR #8"
                ),
            },
            "null": {
                "claim": "IPG choice does not move isolated-build block latency",
                "predicts": {"P1": "|d| < bar", "P2": "|d| < bar", "P3": "|d| < bar"},
            },
        },
        "control_widths": {
            "P1": [1, 2, 3, 4, 6, 7, 8, 9, 10],
            "P2": [1, 2, 3, 4, 5, 6, 8, 9, 10],
            "P3": [1, 2, 3, 4, 5, 6, 7, 9, 10],
        },
        "stop_rules": [
            "Hard stop before any timing if the parity leg shows a bitwise delta at "
            "M<=9 for a timed arm.",
            "Hard stop before any timing if the lane-perturbation positive control "
            "does not diverge at M=5.",
            "Run P4 (e27_full on the real table) only if at least one treated-width "
            "|delta| >= 3 x bar and every control width is within the bar.",
            "Add reps only when a decision-relevant treated-width |delta| lands in "
            "the ambiguous band [bar, 2 x bar].",
            "Do not re-run M=9; E49 already measured it.",
        ],
        "measurement_plan": {
            "method": "E49 Arm 1 isolated single-case builds, ABBA-mirrored legs",
            "widths": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "reps": 21,
            "inner": 10,
            "leg_order": ["A1", "B1", "B2", "A2"],
            "cool_gate": "real 40 C gate, not bypassed; entry/exit temps recorded",
            "wandb_per_leg": True,
        },
        "downstream_pricing": {
            "note": (
                "pricing is reported separately from the timing decision and under "
                "both live mixtures; E54 does not adjudicate between them"
            ),
            "mixtures": ["askeladd_E48_78_round_corpus", "edward_E53_burst_envelope"],
            "harness": "ranked",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    doc = build()
    text = json.dumps(doc, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")

    print(f"theta (overlap parameter, calibrated on M=9) = {doc['theta_calibrated']}")
    print(f"E46 refit calibration factor = {doc['e46_calibration_factor']}")
    print(f"decision bar = {doc['decision_bar_pct']} %\n")
    hdr = (
        f"{'pair':4} {'M':>2} {'shipped':>9} {'cand':>9} {'grp':>7} "
        f"{'A-raw':>8} {'A-cal':>8} {'BW-ser':>8} {'BW-ovl':>8} {'BW-th':>8} {'meas':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, c in doc["cells"].items():
        p = c["pred_pct"]
        meas = "-" if c["measured_pct"] is None else f"{c['measured_pct']:+8.3f}"
        print(
            f"{name:4} {c['width']:>2} {c['shipped_template']:>9} "
            f"{c['candidate_template']:>9} "
            f"{c['shipped_groups']}->{c['candidate_groups']:<5} "
            f"{p['working_group_law_raw']:+8.2f} "
            f"{p['working_group_law_calibrated']:+8.2f} "
            f"{p['bandwidth_serialized']:+8.2f} "
            f"{p['bandwidth_overlapped']:+8.2f} "
            f"{p['bandwidth_theta_calibrated']:+8.2f} {meas:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
