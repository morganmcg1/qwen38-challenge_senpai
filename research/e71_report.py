#!/usr/bin/env python3
"""Turn one E71 census session into the width-tax map.

    usage: research/e71_report.py research/out/TAG/census.json [--json OUT]

Every arm is contrasted against the baseline blocks that bracket it inside the
same ABBA quartet, so monotone thermal drift cancels to first order. The closure
gap is reported as a first-class result, not as an error term.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

# --- fixed campaign constants, each with its source ---------------------------

# senpai/campaign-ledger.md 201(D). One weight-stream removal, ranked candidate
# leg, roughly flat in width. The assignment quotes +/- 0.294 % (the 471-tree
# standard error); 201(D) itself states +/- 0.313 %, which is what is used here.
FLAT_LAW_PCT = -0.639
FLAT_LAW_SE_PCT = 0.313
FLAT_LAW_SE_PCT_ASSIGNMENT = 0.294

# senpai/campaign-ledger.md 199(B). Local cost of ONE extra weight stream, ms.
LOCAL_STREAM_MS = {1: 65.009, 2: 64.40, 3: 72.17, 4: 82.24, 5: 95.48,
                   6: 122.34, 7: 147.21}

# senpai/campaign-ledger.md 200(E). In-situ V = verify_build + eval_wall, us.
LEDGER_V_US = {1: 64979, 2: 69509, 3: 73985, 4: 89610, 5: 114934,
               6: 131749, 7: 150629, 8: 167074, 9: 190483}

# research/e70_double_roofline.py:18-19 (E1). Full candidate round, ms.
# The M=6 row rests on N=2 and the ledger says at :14884-14886 that it must not
# be used as a baseline. The M=9 row was corrected post-E55 to 184.970 ms
# (:14902-14903); both values are carried so the report can show the disagreement.
E1_ROUND_MS = {1: 65.009, 2: 70.482, 3: 75.519, 4: 91.288, 5: 115.691,
               6: 134.668, 7: 154.169, 8: 172.827, 9: 198.237}
E1_ROUND_MS_M9_CORRECTED = 184.970

# Ranked verify-width mixture supplied by the assignment.
RANKED_WIDTH_MIX = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334, 7: 0.122,
                    8: 0.0735, 9: 0.0575}

# 201(E): flat law -1.278 % candidate leg -> +1.295 % raw ratio -> +1.171 %
# published. The published/raw damping is the ratio of those last two.
PUBLISHED_PER_RAW = 1.171 / 1.295

# Quantized weight bytes per family, reproduced in rung 0 and matching ledger
# 199(A)'s 14,412,349,440 B total. Only the interceptable families appear as
# census arms; the rest are the named floor on the closure gap.
FAMILY_BYTES = {
    "mlp_all": 6_417_285_120 + 3_208_642_560,
    "mlp_down": 3_208_642_560,
    "lm_head": 715_161_600,
    "fa_o_proj": 283_115_520,
    "gdn_out_proj": 849_346_560,
    "null": 0,
    "baseline": 0,
}
TOTAL_WEIGHT_BYTES = 14_412_349_440
NOT_INTERCEPTABLE_BYTES = (
    1_415_577_600 + 849_346_560 + 660_602_880 + 13_271_040)

# `mlp_down` is contained in `mlp_all`, so it must never enter the closure sum.
DISJOINT_ARMS = ["mlp_all", "lm_head", "fa_o_proj", "gdn_out_proj"]

# Pre-registered in PR #74 before any timing ran.
PREREGISTERED_MS = {
    "null": (0.0, -0.30, 0.30),
    "lm_head": (2.9, 2.0, 4.2),
    "mlp_all": (32.0, 22.0, 45.0),
    "mlp_down": (11.0, 7.0, 16.0),
    "fa_o_proj": (1.2, 0.6, 2.2),
    "gdn_out_proj": (3.2, 1.8, 5.0),
}

# research/e63-artifacts/e63-cost-curve.json. Standalone lm_head, seconds.
E63_LM_HEAD_M1_S = 0.0028712875
E63_LM_HEAD_M6_S = 0.0057467708


def median_ms(block: dict) -> float:
    return 1e3 * block["seconds_median"]


def block_spread_ms(block: dict) -> float:
    """Half the inter-quartile range of the reps inside one block, in ms."""
    xs = sorted(block["seconds"])
    n = len(xs)
    if n < 4:
        return 1e3 * (xs[-1] - xs[0]) / 2
    lo = statistics.median(xs[: n // 2])
    hi = statistics.median(xs[(n + 1) // 2:])
    return 1e3 * (hi - lo) / 2


def quartets(blocks: list[dict]) -> list[dict]:
    """Recover the `baseline, arm, arm, baseline` quartets by block order."""
    out = []
    i = 0
    while i + 3 < len(blocks):
        b0, a1, a2, b3 = blocks[i:i + 4]
        same = (b0["arm"] == "baseline" and b3["arm"] == "baseline"
                and a1["arm"] == a2["arm"] and a1["arm"] != "baseline"
                and len({b["width"] for b in (b0, a1, a2, b3)}) == 1)
        if not same:
            i += 1
            continue
        base = 0.5 * (median_ms(b0) + median_ms(b3))
        arm = 0.5 * (median_ms(a1) + median_ms(a2))
        out.append({
            "arm": a1["arm"],
            "width": a1["width"],
            "pin_rows": a1["pin_rows"],
            "baseline_ms": base,
            "arm_ms": arm,
            "tax_ms": base - arm,
            "baseline_half_range_ms": abs(median_ms(b0) - median_ms(b3)) / 2,
            "arm_half_range_ms": abs(median_ms(a1) - median_ms(a2)) / 2,
            "within_block_spread_ms": max(block_spread_ms(b) for b in
                                          (b0, a1, a2, b3)),
            "orders": [b["order"] for b in (b0, a1, a2, b3)],
            "entry_temps_c": [b.get("gpu_temp_entry_c") for b in
                              (b0, a1, a2, b3)],
        })
        i += 4
    return out


def curve(blocks: list[dict]) -> dict:
    """Rung 1. The forward and reversed baseline sweeps, averaged per width."""
    per_width: dict[int, list[float]] = {}
    for b in blocks:
        if b["arm"] != "baseline":
            continue
        per_width.setdefault(b["width"], []).append(median_ms(b))
    return {
        width: {
            "mean_ms": sum(v) / len(v),
            "values_ms": v,
            "half_range_ms": (max(v) - min(v)) / 2 if len(v) > 1 else 0.0,
            "n": len(v),
        }
        for width, v in sorted(per_width.items())
    }


def price(tax_ms: float, width: int) -> dict:
    """Price a local tax with the 201(D) flat law, never with L * (f_r/f_l)."""
    stream_ms = LOCAL_STREAM_MS.get(width)
    if stream_ms is None:
        return {"streams_equivalent": None}
    streams = tax_ms / stream_ms
    cand_pct = streams * FLAT_LAW_PCT
    cand_se_pct = abs(streams) * FLAT_LAW_SE_PCT
    frac = -cand_pct / 100.0
    raw_pct = 100.0 * frac / (1.0 - frac) if frac < 1 else float("inf")
    return {
        "local_stream_ms": stream_ms,
        "streams_equivalent": streams,
        "candidate_leg_pct": cand_pct,
        "candidate_leg_se_pct": cand_se_pct,
        "raw_ratio_gain_pct": raw_pct,
        "published_gain_pct": raw_pct * PUBLISHED_PER_RAW,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("census")
    parser.add_argument("--json")
    args = parser.parse_args()

    payload = json.load(open(args.census))
    blocks = payload["blocks"]

    report: dict = {
        "experiment": "e71-in-situ-width-tax-census",
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "identity": payload.get("identity", {}),
    }

    # --- rung 1: the width curve ---------------------------------------------
    curve_ms = curve(blocks)
    report["rung1_curve"] = {
        str(w): {
            **v,
            "ledger_200E_V_ms": LEDGER_V_US.get(w, 0) / 1e3 or None,
            "e1_round_ms": E1_ROUND_MS.get(w),
        }
        for w, v in curve_ms.items()
    }
    session_null = max(
        (v["half_range_ms"] for v in curve_ms.values()), default=0.0)
    report["session_null_ms"] = session_null

    # --- rungs 2 and 3: the pinned-width ablations ----------------------------
    qs = quartets(blocks)
    per_arm: dict[str, dict[int, dict]] = {}
    for q in qs:
        per_arm.setdefault(q["arm"], {})[q["width"]] = q
    report["quartets"] = qs

    headline = {}
    for arm, by_width in sorted(per_arm.items()):
        for width, q in sorted(by_width.items()):
            entry = dict(q)
            entry["preregistered_ms"] = PREREGISTERED_MS.get(arm)
            entry["bytes"] = FAMILY_BYTES.get(arm)
            if width == 6:
                entry["price"] = price(q["tax_ms"], width)
            headline.setdefault(arm, {})[str(width)] = entry
    report["arms"] = headline

    # --- controls -------------------------------------------------------------
    controls = {}
    null_by_width = per_arm.get("null", {})
    if null_by_width:
        worst_null = max(abs(q["tax_ms"]) for q in null_by_width.values())
        measurable = [
            abs(q["tax_ms"]) for a, bw in per_arm.items() if a != "null"
            for q in bw.values()
        ]
        smallest = min(measurable) if measurable else float("nan")
        controls["null"] = {
            "worst_abs_tax_ms": worst_null,
            "smallest_measured_tax_ms": smallest,
            "fraction_of_smallest": worst_null / smallest if measurable else None,
            "gate": "stop if fraction_of_smallest > 0.25",
            "passed": bool(measurable) and worst_null / smallest <= 0.25,
        }
    if 6 in per_arm.get("lm_head", {}):
        measured = per_arm["lm_head"][6]["tax_ms"]
        e63 = 1e3 * (E63_LM_HEAD_M6_S - E63_LM_HEAD_M1_S)
        ratio = measured / e63 if e63 else float("nan")
        controls["lm_head_positive"] = {
            "measured_tax_ms": measured,
            "e63_standalone_delta_ms": e63,
            "ratio": ratio,
            "gate": "stop if ratio outside [0.5, 2.0]",
            "passed": 0.5 <= ratio <= 2.0,
        }
    report["controls"] = controls

    # --- closure --------------------------------------------------------------
    closure = {}
    for width in sorted({q["width"] for q in qs}):
        parts = {a: per_arm[a][width]["tax_ms"]
                 for a in DISJOINT_ARMS if width in per_arm.get(a, {})}
        if not parts:
            continue
        attributed = sum(parts.values())
        base_v = LEDGER_V_US.get(1, 0) / 1e3
        this_v = LEDGER_V_US.get(width, 0) / 1e3
        measured_here = None
        if width in curve_ms and 1 in curve_ms:
            measured_here = curve_ms[width]["mean_ms"] - curve_ms[1]["mean_ms"]
        total = measured_here if measured_here is not None else this_v - base_v
        closure[str(width)] = {
            "parts_ms": parts,
            "attributed_ms": attributed,
            "total_width_tax_ms_this_session": measured_here,
            "total_width_tax_ms_ledger_200E": this_v - base_v,
            "gap_ms": total - attributed,
            "gap_fraction": (total - attributed) / total if total else None,
            "attributed_fraction": attributed / total if total else None,
            "non_interceptable_weight_bytes": NOT_INTERCEPTABLE_BYTES,
            "non_interceptable_weight_share": (
                NOT_INTERCEPTABLE_BYTES / TOTAL_WEIGHT_BYTES),
        }
    report["closure"] = closure

    # --- ranked width mixture -------------------------------------------------
    mixture = {}
    for arm in DISJOINT_ARMS:
        by_width = per_arm.get(arm, {})
        if not by_width:
            continue
        known = sorted(by_width)
        weighted = 0.0
        for w, p in RANKED_WIDTH_MIX.items():
            nearest = min(known, key=lambda k: abs(k - w))
            weighted += p * by_width[nearest]["tax_ms"]
        mixture[arm] = {
            "mixture_weighted_tax_ms": weighted,
            "measured_widths": known,
            "note": ("widths not measured take the nearest measured width; "
                     "this is interpolation, not measurement"),
            "price_at_m6_stream_cost": price(weighted, 6),
        }
    report["ranked_width_mixture"] = mixture
    report["flat_law"] = {
        "pct_per_stream_removed": FLAT_LAW_PCT,
        "se_pct_ledger_201D": FLAT_LAW_SE_PCT,
        "se_pct_as_quoted_in_assignment": FLAT_LAW_SE_PCT_ASSIGNMENT,
        "source": "senpai/campaign-ledger.md 201(D)",
        "bridge_assumption": (
            "A local tax of X ms at width M is priced as X / LOCAL_STREAM_MS[M] "
            "weight streams and then at -0.639 % per stream. This assumes the "
            "local-to-ranked transfer of a census tax matches that of a stream "
            "removal. 201(D)'s own ratio column spans 0.048 to 0.480 across "
            "widths, so this bridge is the largest single uncertainty in the "
            "conversion. It is an assumption, not a measurement."),
        "flatness_significance": (
            "The flatness of the flat law is about 1.1 sigma "
            "(campaign-ledger.md :16733-16739). It is not established."),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        with open(args.json, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
