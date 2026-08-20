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
    "all_interceptable": (6_417_285_120 + 3_208_642_560 + 715_161_600
                          + 283_115_520 + 849_346_560),
    "null": 0,
    "baseline": 0,
}
TOTAL_WEIGHT_BYTES = 14_412_349_440
NOT_INTERCEPTABLE_BYTES = (
    1_415_577_600 + 849_346_560 + 660_602_880 + 13_271_040)

# `mlp_down` is contained in `mlp_all`, so it must never enter the closure sum.
DISJOINT_ARMS = ["mlp_all", "lm_head", "fa_o_proj", "gdn_out_proj"]

# Shape of every scored quantized linear the census can reach, read from
# weights/config.json and Vendor/mlx-swift-lm/.../Qwen35.swift. `k` is the
# reduction dimension, `n` the output dimension, `calls` the dispatches per
# round. `k // 64` is the number of affine group-64 k-blocks one lane walks,
# which is the kernel's inner loop trip count.
#
# `mlp_gate_up` is not an arm: it is recovered as `mlp_all - mlp_down` so that
# the fit has a large-bytes, shallow-k point to sit against `mlp_down`.
FAMILY_SHAPE = {
    "mlp_gate_up": {"k": 5120, "n": 34816, "calls": 64,
                    "bytes": 6_417_285_120, "derived_from": "mlp_all - mlp_down"},
    "mlp_down": {"k": 17408, "n": 5120, "calls": 64, "bytes": 3_208_642_560},
    "gdn_out_proj": {"k": 6144, "n": 5120, "calls": 48, "bytes": 849_346_560},
    "fa_o_proj": {"k": 6144, "n": 5120, "calls": 16, "bytes": 283_115_520},
    "lm_head": {"k": 5120, "n": 248320, "calls": 1, "bytes": 715_161_600},
}

# Kernel-selection audit for the ranked host (applegpu_g17s) against this host
# (applegpu_g16s). Every selection predicate below was read from source.
#
#   Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:84-125
#     get_qmv_batch_limit is the ONLY site in the whole Metal backend that reads
#     get_architecture_gen(). It special-cases gen 13 and 14; gen 16 and 17 both
#     fall through to the same branch, and arch_size is 's' for both hosts, so
#     it returns the same limit on both.
#   quantized.cpp:1483  vector_limit = get_qmv_batch_limit(K, N, d)
#   quantized.cpp:1484  M >= vector_limit -> qmm, else -> dispatch_qmv
#   quantized.cpp:259   fast = (N % 8 == 0 && K % 512 == 0)
#   kernels/quantized.h:1917-1980  the cross-row partition switch keys on
#     ntg.x (= M) and out_vec_size (= N). Neither is architecture-derived.
#
# Every other architecture-sensitive site in the backend (matmul.cpp:208,372,
# 918,2303,2514 and scaled_dot_product_attention.cpp:443,747) branches only on
# get_architecture().back(), which is 's' on both hosts.
QMV_BATCH_LIMIT = 10  # K > 4096 and N > 4096 for every census family
GEN_SENSITIVE_SITES = [
    "quantized.cpp:84-125 get_qmv_batch_limit (get_architecture_gen)",
]
CHAR_SENSITIVE_SITES = [
    "matmul.cpp:208", "matmul.cpp:372", "matmul.cpp:918",
    "matmul.cpp:2303", "matmul.cpp:2514",
    "scaled_dot_product_attention.cpp:443",
    "scaled_dot_product_attention.cpp:747",
]
# Cross-row partition <T, M, IPG, true> actually selected at each width,
# kernels/quantized.h:1929-1975. The IPG column is campaign-tuned.
CROSSROW_PARTITION = {3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 5}
# Widths whose IPG was retuned by a merged campaign commit rather than inherited
# from the promoted crown snapshot 1033e1a.
CAMPAIGN_RETUNED_WIDTHS = {5: "b757237 <T,5,5>", 6: "aa8ce50 <T,6,6>",
                           9: "t55 <T,9,5>"}

# Pre-registered in PR #74 before any timing ran.
PREREGISTERED_MS = {
    "null": (0.0, -0.30, 0.30),
    "lm_head": (2.9, 2.0, 4.2),
    "mlp_all": (32.0, 22.0, 45.0),
    "mlp_down": (11.0, 7.0, 16.0),
    "fa_o_proj": (1.2, 0.6, 2.2),
    "gdn_out_proj": (3.2, 1.8, 5.0),
    # Added after the plumbing smoke, before the census. Registered as the exact
    # sum of the four disjoint arms, i.e. the hypothesis is perfect additivity.
    "all_interceptable": (39.3, 30.6, 56.4),
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


def shape_table(per_arm: dict[str, dict[int, dict]]) -> dict:
    """Tax, bytes, k, n and k-blocks for every family, at every census width.

    `mlp_gate_up` is recovered as `mlp_all - mlp_down`. It carries the SwiGLU
    and the fused gate/up compiled kernel as well as the two projections, so it
    is an upper bound on the projection pair alone.
    """
    widths = sorted(per_arm.get("mlp_all", {}))
    rows: dict[str, dict] = {}
    for fam, shape in FAMILY_SHAPE.items():
        gb = shape["bytes"] / 1e9
        row = {**shape, "k_blocks": shape["k"] // 64, "gb": gb, "by_width": {}}
        for w in widths:
            if fam == "mlp_gate_up":
                if w not in per_arm.get("mlp_all", {}) or w not in per_arm.get("mlp_down", {}):
                    continue
                tax = per_arm["mlp_all"][w]["tax_ms"] - per_arm["mlp_down"][w]["tax_ms"]
            else:
                if w not in per_arm.get(fam, {}):
                    continue
                tax = per_arm[fam][w]["tax_ms"]
            row["by_width"][str(w)] = {
                "tax_ms": tax,
                "ms_per_gb": tax / gb,
                # ms per GB per row added beyond width 1, the quantity the fit
                # below models. Dividing by (w - 1) removes the trivial linear
                # growth so a residual k-dependence is visible.
                "ms_per_gb_per_extra_row": tax / gb / (w - 1),
            }
        rows[fam] = row
    return rows


def separated_fit(shapes: dict) -> dict:
    """Split the per-GB width tax into a bytes term and a reduction-depth term.

    At each width, fit  ms_per_gb(F) = alpha + beta * k_blocks(F)  by ordinary
    least squares over the five families. `alpha` is the width tax a family pays
    per GB regardless of how deep its reduction is; `beta` is the extra ms per
    GB for each additional group-64 k-block a lane must walk.

    Five points and two parameters, so this is a fit and not an identity. The
    two families that share a shape exactly (`fa_o_proj` and `gdn_out_proj`,
    both k=6144, n=5120) are reported separately as a shape-invariance check
    that does not depend on the fit at all.
    """
    widths = sorted(
        {w for r in shapes.values() for w in r["by_width"]}, key=int)
    out: dict = {"per_width": {}}
    for w in widths:
        pts = [(r["k_blocks"], r["by_width"][w]["ms_per_gb"])
               for r in shapes.values() if w in r["by_width"]]
        if len(pts) < 3:
            continue
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        det = n * sxx - sx * sx
        beta = (n * sxy - sx * sy) / det
        alpha = (sy - beta * sx) / n
        resid = [y - (alpha + beta * x) for x, y in pts]
        ss_res = sum(r * r for r in resid)
        ybar = sy / n
        ss_tot = sum((y - ybar) ** 2 for _, y in pts)
        out["per_width"][w] = {
            "n_families": n,
            "alpha_ms_per_gb": alpha,
            "beta_ms_per_gb_per_kblock": beta,
            "r_squared": 1 - ss_res / ss_tot if ss_tot else None,
            "max_abs_residual_ms_per_gb": max(abs(r) for r in resid),
            # What the deep-k family pays above a shallow-k family of the same
            # size, in ms at this width, on mlp_down's 3.209 GB.
            "mlp_down_k_penalty_ms": beta * (272 - 80) * (3_208_642_560 / 1e9),
        }
    fa = shapes.get("fa_o_proj", {}).get("by_width", {})
    gdn = shapes.get("gdn_out_proj", {}).get("by_width", {})
    out["identical_shape_check"] = {
        "families": ["fa_o_proj", "gdn_out_proj"],
        "shared_shape": {"k": 6144, "n": 5120, "k_blocks": 96},
        "note": ("Same kernel, same k, same n, different call count and so "
                 "different bytes. If the width tax is a property of the shape "
                 "then their ms/GB must agree. This check is independent of "
                 "the fit."),
        "by_width": {
            w: {"fa_o_proj_ms_per_gb": fa[w]["ms_per_gb"],
                "gdn_out_proj_ms_per_gb": gdn[w]["ms_per_gb"],
                "ratio": fa[w]["ms_per_gb"] / gdn[w]["ms_per_gb"]}
            for w in fa if w in gdn
        },
    }
    return out


def kernel_selection_map(shapes: dict, widths: list[int]) -> dict:
    """Per family, does the scored path reach the same kernel on g16s and g17s?

    Answers the advisor's ranked-regression question. Selection identity is a
    source fact; performance identity is not, and the two are reported apart.
    """
    fams = {}
    for fam, r in shapes.items():
        k, n = r["k"], r["n"]
        fast = (n % 8 == 0) and (k % 512 == 0)
        fams[fam] = {
            "k": k, "n": n,
            "qmv_batch_limit_both_hosts": QMV_BATCH_LIMIT,
            "stays_on_qmv_for_all_census_widths": max(widths) < QMV_BATCH_LIMIT,
            "qmv_fast": fast,
            "crossrow_wide_path": n >= 4096,
            "kernel": ("affine_qmv_fast -> qmv_fast_crossrow_affine4_g64_m"
                       if fast and n >= 4096 else "see quantized.h switch"),
            "selection_identical_g16s_vs_g17s": True,
            "selection_inputs": ["ntg.x (= M)", "out_vec_size (= N)",
                                 "in_vec_size (= K)", "group_size", "bits"],
        }
    return {
        "families": fams,
        "generation_reading_sites": GEN_SENSITIVE_SITES,
        "arch_char_reading_sites": CHAR_SENSITIVE_SITES,
        "arch_char_both_hosts": "s",
        "verdict": (
            "Kernel SELECTION is identical on applegpu_g16s and applegpu_g17s "
            "for every family at every census width. The only site in the Metal "
            "backend that reads the architecture generation is "
            "get_qmv_batch_limit; gen 16 and gen 17 both miss its gen-13/14 "
            "special case and both hosts report arch_size 's', so it returns "
            "the same limit. Every other architecture-sensitive site branches "
            "on get_architecture().back(), which is 's' on both."),
        "caveat": (
            "Selection identity is not performance identity. All five families "
            "land on the SAME campaign-tuned cross-row partition kernel, whose "
            "IPG template argument was hand-selected on g16s at widths 5, 6 and "
            "9. Every family in this census is therefore exposed to the "
            "mechanism suspected in the ff73cbbd ranked regression, and the "
            "exposure is uniform across families rather than concentrated in "
            "one. What differs between families is k_blocks, which sets the "
            "inner-loop trip count and the register pressure that the IPG "
            "choice trades against."),
        "crossrow_partition_ipg": {str(w): CROSSROW_PARTITION.get(w)
                                   for w in widths},
        "campaign_retuned_widths": {str(w): c for w, c
                                    in CAMPAIGN_RETUNED_WIDTHS.items()},
        "census_widths_on_campaign_retuned_partitions": [
            w for w in widths if w in CAMPAIGN_RETUNED_WIDTHS],
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

    # --- shape and the separated bytes/reduction-depth fit --------------------
    shapes = shape_table(per_arm)
    fit = separated_fit(shapes)

    # --- controls -------------------------------------------------------------
    controls = {}
    null_by_width = per_arm.get("null", {})
    if null_by_width:
        # The gate is applied per width, because the null is a property of the
        # wrapper at that width and the smallest tax is a property of the
        # families measured there. A family arm whose tax does not clear the
        # null at its own width is reported as NOT RESOLVED rather than as a
        # measurement.
        by_width = {}
        for w, nq in sorted(null_by_width.items()):
            null_abs = abs(nq["tax_ms"])
            others = {a: bw[w]["tax_ms"] for a, bw in per_arm.items()
                      if a != "null" and w in bw}
            unresolved = sorted(a for a, t in others.items()
                                if abs(t) <= null_abs / 0.25)
            resolved = {a: t for a, t in others.items() if a not in unresolved}
            smallest_resolved = min((abs(t) for t in resolved.values()),
                                    default=float("nan"))
            by_width[str(w)] = {
                "null_tax_ms": nq["tax_ms"],
                "null_abs_ms": null_abs,
                "smallest_arm_tax_ms": min((abs(t) for t in others.values()),
                                           default=None),
                "fraction_of_smallest_arm": (
                    null_abs / min(abs(t) for t in others.values())
                    if others else None),
                "arms_not_resolved_at_this_width": unresolved,
                "smallest_resolved_arm_tax_ms": smallest_resolved,
                "passed_for_resolved_arms": bool(resolved),
            }
        # Advisor's caution: if the wrapper's own cost grows with width, the
        # closure residual shrinks for the wrong reason. Test it directly.
        ws = sorted(null_by_width)
        taxes = [null_by_width[w]["tax_ms"] for w in ws]
        n = len(ws)
        slope = ((n * sum(w * t for w, t in zip(ws, taxes))
                  - sum(ws) * sum(taxes))
                 / (n * sum(w * w for w in ws) - sum(ws) ** 2)) if n > 1 else None
        controls["null"] = {
            "by_width": by_width,
            "gate": "per width: stop if abs(null) > 0.25 * smallest arm tax",
            "worst_abs_tax_ms": max(abs(t) for t in taxes),
            "slope_ms_per_row": slope,
            "grows_with_width": bool(slope is not None and slope > 0),
            "slope_note": (
                "A positive slope would mean the wrapper gets more expensive at "
                "wider M, which would inflate every tax and shrink the closure "
                "gap for the wrong reason. A negative or zero slope means the "
                "reported gap is not an artefact of the wrapper."),
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
        joint = per_arm.get("all_interceptable", {}).get(width)
        closure[str(width)] = {
            "parts_ms": parts,
            "attributed_ms": attributed,
            "joint_all_interceptable_ms": joint["tax_ms"] if joint else None,
            "additivity_residual_ms": (
                joint["tax_ms"] - attributed if joint else None),
            "additivity_note": (
                "joint minus the sum of the disjoint single-family arms. Near "
                "zero means the families are additive and the remaining gap "
                "belongs to families this harness cannot reach. A large value "
                "means the per-family map is incomplete on its own."),
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
    report["ranked_width_mixture"] = {
        "harness": "local",
        "score_conversion_withheld": True,
        "score_conversion_note": (
            "The advisor directed on 2026-08-20 (PR #74 feedback "
            "e71-mlp-down-is-the-headline-and-two-things-to-add): 'Do not "
            "convert any local number to a ranked score yourself; report "
            "harness=local and leave the conversion to me.' The flat-law "
            "arithmetic below is retained only so the advisor can audit the "
            "inputs. No ranked score is claimed by this experiment."),
        "per_arm": mixture,
    }
    report["shapes"] = shapes
    report["separated_fit"] = fit
    report["kernel_selection"] = kernel_selection_map(
        shapes, sorted(per_arm.get("mlp_all", {})))
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
