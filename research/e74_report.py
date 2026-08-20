#!/usr/bin/env python3
"""E74 rungs 1 to 3: locate the working-threadgroup knee in situ.

    usage: research/e74_report.py research/out/e74-census-r1/census.json \
               [--prior research/out/e71-census-r1/census.json] \
               [--rung0 research/out/e74-rung0.json] [--json OUT]

Reduction is E71's, unchanged: the same ABBA quartet contrast, the same null
gate, the same shape table. This file adds the grid terms, the knee fit, the
cross-instrument control against E33 and the recommendation surface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e71_report as e71  # noqa: E402
import e74_rung0 as r0  # noqa: E402

# Measured local weight-stream bandwidth, senpai/campaign-ledger.md 205(D):
# "Our M4 Pro 20-core is 273 GB/s published and 226.0 GB/s measured."
LOCAL_BW_GB_S = 226.0

# The eight scored quantized-linear shapes, ledger item 130 table at :3457.
SCORED_SHAPES = [
    ("head.lm_head", 248320, 5120),
    ("head.compact_draft_vocab", 98336, 5120),
    ("mlp.gate_up_fused", 34816, 5120),
    ("linear_attn.in_proj", 16480, 5120),
    ("full_attn.qkv_proj", 14336, 5120),
    ("full_attn.o_proj", 5120, 6144),
    ("linear_attn.out_proj", 5120, 6144),
    ("mlp.down", 5120, 17408),
]

# Census family -> the scored shape it is the same kernel cell as.
FAMILY_TO_SCORED = {
    "lm_head": "head.lm_head",
    "mlp_gate_up": "mlp.gate_up_fused",
    "fa_o_proj": "full_attn.o_proj",
    "gdn_out_proj": "linear_attn.out_proj",
    "mlp_down": "mlp.down",
}

# Ranked-host core counts to carry. The ranked runner is labelled
# m5-qwen38-27b-mtp and reports applegpu_g17s (ledger 154(1)); its tier is not
# probed (ledger :17127), so the core count is an inference, never a
# measurement. 20 is the M4 Pro count on this host, read from ioreg.
RANKED_CORE_ASSUMPTIONS = [20, 24, 40]

# Ranked verify-width mixture, from the E74 assignment's leg table.
RANKED_WIDTH_MIX = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334, 7: 0.122,
                    8: 0.0735, 9: 0.0575}

# The log cost of one extra x-group weight pass. This census CANNOT identify it:
# every family it measures has out_vec_size >= 4096, so all five families share
# one IPG at a given width, and the per-width intercept absorbs the whole group
# term. The value is imported from prior art measured on this host family at
# M=6, senpai/campaign-ledger.md item 157: "R1, the second weight pass: +0.1196
# as attributed", pre-registered band [0.130, 0.200], point estimate +0.1658.
# The low end is the conservative attributed value; the high end is the top of
# the registered band. That +12 % rather than +100 % for a doubled weight stream
# is itself evidence that the second pass is mostly cache-served.
G_GROUP_PASS_LOG = math.log(1.1196)
G_GROUP_PASS_LOG_RANGE = (math.log(1.1196), math.log(1.200))

# Independent prior-art bracket on the same knee, ledger item 157 R3: grid
# thinning by 2x costs "+7.4 pp at 1280 TGs decaying to ~0 at >=4120 TGs" at
# identical n and identical traffic. Under the hard-knee model a 2x thinning of
# a grid at T working threadgroups costs A*[max(0, lnK - ln(T/2)) - max(0, lnK -
# lnT)]. Zero cost at T=4120 requires K <= 2060; the full A*ln2 cost at T=1280
# requires K >= 1280 and gives A = 0.074/ln2.
E33_R3_KNEE_BRACKET = (1280.0, 2060.0)
E33_R3_IMPLIED_A = 0.074 / math.log(2.0)


def reduce_session(path: str) -> dict:
    payload = json.load(open(path))
    blocks = payload["blocks"]
    qs = e71.quartets(blocks)
    per_arm: dict[str, dict[int, dict]] = {}
    for q in qs:
        per_arm.setdefault(q["arm"], {})[q["width"]] = q
    return {"payload": payload, "blocks": blocks, "quartets": qs,
            "per_arm": per_arm, "curve": e71.curve(blocks),
            "shapes": e71.shape_table(per_arm)}


def null_control(per_arm: dict) -> dict:
    """E71's rule, unchanged: per width, stop if |null| > 0.25 * smallest arm."""
    out = {}
    for w, nq in sorted(per_arm.get("null", {}).items()):
        null_abs = abs(nq["tax_ms"])
        others = {a: bw[w]["tax_ms"] for a, bw in per_arm.items()
                  if a != "null" and w in bw}
        unresolved = sorted(a for a, t in others.items() if abs(t) <= null_abs / 0.25)
        smallest = min((abs(t) for t in others.values()), default=None)
        out[str(w)] = {
            "null_tax_ms": nq["tax_ms"],
            "smallest_arm_tax_ms": smallest,
            "fraction_of_smallest_arm": null_abs / smallest if smallest else None,
            "arms_not_resolved_at_this_width": unresolved,
            "passed": bool(smallest and null_abs <= 0.25 * smallest),
        }
    return out


def cell_surface(sessions: dict[str, dict], cores: int) -> list[dict]:
    """One row per (family, width, session) with the grid terms attached."""
    rows = []
    for label, sess in sessions.items():
        for fam, shape in sess["shapes"].items():
            for w_str, cell in shape["by_width"].items():
                w = int(w_str)
                n = shape["n"]
                tgs = r0.working_tgs(w, n)
                unresolved = fam in sess["null"].get(w_str, {}).get(
                    "arms_not_resolved_at_this_width", [])
                if fam == "mlp_gate_up":
                    unresolved = ("mlp_all" in sess["null"].get(w_str, {}).get(
                        "arms_not_resolved_at_this_width", []))
                rows.append({
                    "session": label, "family": fam, "width": w,
                    "n": n, "k": shape["k"], "k_blocks": shape["k_blocks"],
                    "gb": shape["gb"], "calls": shape["calls"],
                    "ipg": r0.ipg_for(w, n), "groups": math.ceil(w / r0.ipg_for(w, n)),
                    "working_tgs": tgs, "tgs_per_core": tgs / cores,
                    "launched_tgs": r0.launched_tgs(w, n),
                    "tax_ms": cell["tax_ms"], "ms_per_gb": cell["ms_per_gb"],
                    "resolved": not unresolved,
                })
    return rows


def width_stats(rows: list[dict], session: str) -> dict:
    out = {}
    for w in sorted({r["width"] for r in rows if r["session"] == session}):
        cells = {r["family"]: r["ms_per_gb"] for r in rows
                 if r["session"] == session and r["width"] == w}
        if len(cells) < 5:
            continue
        level = statistics.fmean(cells.values())
        d = statistics.fmean([cells["fa_o_proj"], cells["gdn_out_proj"]]) - \
            statistics.fmean([cells["lm_head"], cells["mlp_gate_up"]])
        r = cells["mlp_down"] - statistics.fmean([cells["fa_o_proj"], cells["gdn_out_proj"]])
        out[str(w)] = {
            "ms_per_gb": cells, "level": level,
            "D": d, "D_over_level": d / level,
            "R": r, "R_over_level": r / level,
            "spread_abs": max(cells.values()) - min(cells.values()),
            "spread_ratio": max(cells.values()) / min(cells.values()),
            "groups": math.ceil(w / r0.SHIPPED_IPG[w]),
            "ipg": r0.SHIPPED_IPG[w],
        }
    return out


def ols(design: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares by Gaussian elimination on the normal equations."""
    p = len(design[0])
    ata = [[sum(row[i] * row[j] for row in design) for j in range(p)] for i in range(p)]
    aty = [sum(row[i] * yi for row, yi in zip(design, y)) for i in range(p)]
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(ata[r][i]))
        if abs(ata[piv][i]) < 1e-12:
            return [float("nan")] * p
        ata[i], ata[piv] = ata[piv], ata[i]
        aty[i], aty[piv] = aty[piv], aty[i]
        for r in range(p):
            if r == i:
                continue
            f = ata[r][i] / ata[i][i]
            for c in range(i, p):
                ata[r][c] -= f * ata[i][c]
            aty[r] -= f * aty[i]
    return [aty[i] / ata[i][i] for i in range(p)]


def fit_knee(rows: list[dict], knee_grid=None) -> dict:
    """ln(ms/GB) = a_M + A * max(0, ln(knee) - ln(tgs)) + B_w * (k_blocks - 96).

    Per-width intercepts absorb every uniform width term, including the whole
    effect of the shipped group count, because all five families here have
    out_vec_size >= 4096 and therefore share one IPG at a given width. `A` is
    identified only by cross-family variation in the working threadgroup count
    at a fixed width. The depth slope is split at width 6, which E71 found is
    the one width whose reduction-depth penalty is anomalous; width 6 is also
    the only width with IPG=6, so the two readings are not separable here.
    """
    widths = sorted({r["width"] for r in rows})
    if knee_grid is None:
        knee_grid = [400 * 1.03 ** i for i in range(190)]  # 400 .. ~100k

    def design_for(knee):
        d = []
        for r in rows:
            row = [1.0 if r["width"] == w else 0.0 for w in widths]
            row.append(max(0.0, math.log(knee) - math.log(r["working_tgs"])))
            row.append((r["k_blocks"] - 96) * (1.0 if r["width"] != 6 else 0.0))
            row.append((r["k_blocks"] - 96) * (1.0 if r["width"] == 6 else 0.0))
            d.append(row)
        return d

    y = [math.log(r["ms_per_gb"]) for r in rows]
    profile = []
    best = None
    for knee in knee_grid:
        d = design_for(knee)
        beta = ols(d, y)
        if any(math.isnan(b) for b in beta):
            continue
        sse = sum((yi - sum(b * x for b, x in zip(beta, row))) ** 2
                  for row, yi in zip(d, y))
        profile.append({"knee": knee, "sse": sse, "A": beta[len(widths)]})
        if best is None or sse < best["sse"]:
            best = {"knee": knee, "sse": sse, "beta": beta}
    if best is None:
        return {"resolved": False}
    beta = best["beta"]
    d = design_for(best["knee"])
    resid = [yi - sum(b * x for b, x in zip(beta, row)) for row, yi in zip(d, y)]
    dof = max(1, len(rows) - len(beta) - 1)
    # Profile interval: every knee whose SSE is within an F(1, dof) factor of
    # the minimum is not separable from the optimum at 68 %.
    thresh = best["sse"] * (1.0 + 1.0 / dof)
    inside = [p["knee"] for p in profile if p["sse"] <= thresh]
    return {
        "resolved": True,
        "widths": widths,
        "width_intercepts": {str(w): beta[i] for i, w in enumerate(widths)},
        "A_per_log_deficit": beta[len(widths)],
        "B_depth_per_k_block": beta[len(widths) + 1],
        "B_depth_per_k_block_width6": beta[len(widths) + 2],
        "knee_working_tgs": best["knee"],
        "knee_interval_working_tgs": [min(inside), max(inside)] if inside else None,
        "knee_spans_whole_grid": bool(inside and
                                      min(inside) <= knee_grid[1] and
                                      max(inside) >= knee_grid[-2]),
        "sse": best["sse"],
        "rmse_log": math.sqrt(best["sse"] / len(rows)),
        "residual_max_abs_log": max(abs(x) for x in resid),
        "n_cells": len(rows),
        "profile": profile,
    }


def bootstrap_knee(rows: list[dict], draws: int = 300, seed: int = 20260820) -> dict:
    rng = random.Random(seed)
    fams = sorted({r["family"] for r in rows})
    knees, a_vals = [], []
    for _ in range(draws):
        pick = [rng.choice(fams) for _ in fams]
        sample = [r for f in pick for r in rows if r["family"] == f]
        fit = fit_knee(sample)
        if fit.get("resolved") and not math.isnan(fit["A_per_log_deficit"]):
            knees.append(fit["knee_working_tgs"])
            a_vals.append(fit["A_per_log_deficit"])
    if not knees:
        return {"resolved": False}
    knees.sort()
    a_vals.sort()
    q = lambda xs, p: xs[max(0, min(len(xs) - 1, int(p * len(xs))))]
    return {
        "resolved": True, "draws": len(knees),
        "knee_p16": q(knees, 0.16), "knee_p50": q(knees, 0.50), "knee_p84": q(knees, 0.84),
        "A_p16": q(a_vals, 0.16), "A_p50": q(a_vals, 0.50), "A_p84": q(a_vals, 0.84),
        "resampled": "families with replacement, all widths of a family kept together",
    }


def e33_positive_control(fit: dict, tax_by_family: dict) -> dict:
    """The fit must reproduce E33's sign flip between 1792 and 2060 arm groups.

    E33 measured a total per-call cost ratio. This census measures a marginal
    per-byte width tax. The two are converted with a bandwidth-anchored base
    cost so that the comparison is on one axis.
    """
    a = fit["A_per_log_deficit"]
    knee = fit["knee_working_tgs"]
    pen = lambda t: max(0.0, math.log(knee) - math.log(t))

    preds = []
    for name, n, k, t_old, t_new, obs in r0.E33_SHAPES:
        preds.append({"shape": name, "shipped_tgs": t_old, "arm_tgs": t_new,
                      "observed_ratio": obs,
                      "grid_log_change": a * (pen(t_new) - pen(t_old))})
    # One free level term: the E33 arm also moved ROWS_PER_SIMD 4 -> 2, which is
    # shape-independent. Only the shape-dependent part is a grid signal.
    c = statistics.fmean([math.log(p["observed_ratio"]) - p["grid_log_change"]
                          for p in preds])
    for p in preds:
        p["predicted_ratio"] = math.exp(c + p["grid_log_change"])
        p["residual"] = p["observed_ratio"] - p["predicted_ratio"]

    def solve_flip(level_log, scale):
        lo, hi, mid = 1.0, 1e7, 1.0
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            step = math.exp(a * (pen(mid / 2) - pen(mid))) - 1.0
            val = level_log + math.log1p(step * scale)
            if val > 0:
                lo = mid
            else:
                hi = mid
        return mid

    raw = {
        "predicted_sign_flip_shipped_tgs": solve_flip(c, 1.0),
        "level_term_C": math.exp(c),
        "ordering_tau_predicted_vs_observed": r0.kendall_tau_b(
            [p["predicted_ratio"] for p in preds], [p["observed_ratio"] for p in preds]),
        "max_abs_residual": max(abs(p["residual"]) for p in preds),
    }
    raw["passed"] = 3584.0 <= raw["predicted_sign_flip_shipped_tgs"] <= 4120.0

    # Instrument bridge. E33 reports a ratio of TOTAL per-call cost. This census
    # reports a MARGINAL per-byte width tax, which is only part of that total.
    # A relative change of x in the width tax shows up as x * tax_share in E33's
    # ratio, where tax_share = tax / (tax + weight-stream base). The base is the
    # bandwidth-bound M=1 cost of the same weights.
    shares = {}
    for fam, scored in FAMILY_TO_SCORED.items():
        tax = tax_by_family.get(fam)
        if tax is None:
            continue
        base_ms = tax["gb"] / LOCAL_BW_GB_S * 1e3
        shares[scored] = {
            "family": fam, "census_m6_tax_ms": tax["tax_ms"],
            "bandwidth_base_ms": base_ms,
            "tax_share_of_total": tax["tax_ms"] / (tax["tax_ms"] + base_ms),
        }
    measured_shares = [v["tax_share_of_total"] for v in shares.values()]
    mean_share = statistics.fmean(measured_shares) if measured_shares else 1.0

    conv = []
    for p in preds:
        s = shares.get(p["shape"], {}).get("tax_share_of_total", mean_share)
        conv.append({**p, "tax_share": s, "tax_share_measured": p["shape"] in shares,
                     "shape_term": math.exp(p["grid_log_change"]) - 1.0})
    c2 = statistics.fmean([math.log(p["observed_ratio"])
                           - math.log1p(p["shape_term"] * p["tax_share"]) for p in conv])
    for p in conv:
        p["predicted_ratio"] = math.exp(c2) * (1 + p["shape_term"] * p["tax_share"])
        p["residual"] = p["observed_ratio"] - p["predicted_ratio"]
    converted = {
        "predicted_sign_flip_shipped_tgs": solve_flip(c2, mean_share),
        "level_term_C": math.exp(c2),
        "mean_tax_share": mean_share,
        "ordering_tau_predicted_vs_observed": r0.kendall_tau_b(
            [p["predicted_ratio"] for p in conv], [p["observed_ratio"] for p in conv]),
        "max_abs_residual": max(abs(p["residual"]) for p in conv),
        "per_shape": conv,
        "tax_shares": shares,
    }
    converted["passed"] = 3584.0 <= converted["predicted_sign_flip_shipped_tgs"] <= 4120.0

    return {
        "gate": ("the fit must place the E33 sign flip between 3584 and 4120 shipped "
                 "working threadgroups, that is between 1792 and 2060 in the arm"),
        "passed": bool(converted["passed"]),
        "primary_variant": "tax_share_converted",
        "raw_variant": {**raw, "per_shape": preds,
                        "note": ("compares a marginal per-byte quantity directly with a "
                                 "total-cost ratio, so it overstates the predicted "
                                 "spread by about 1/tax_share")},
        "tax_share_converted": converted,
        "caveat": ("the level term C is free in both variants, so this control tests "
                   "the shape dependence and the flip location, not the absolute "
                   "level. The E33 arm also changed ROWS_PER_SIMD 4 -> 2, which C "
                   "absorbs. Three of the eight E33 shapes are not in this census and "
                   "carry the mean measured tax share instead of their own."),
    }


def ranked_extrapolation(fit: dict, cores_local: int) -> dict:
    knee_per_core = fit["knee_working_tgs"] / cores_local
    iv = fit.get("knee_interval_working_tgs")
    out = {"local_cores": cores_local,
           "knee_working_tgs_local": fit["knee_working_tgs"],
           "knee_per_core": knee_per_core,
           "knee_per_core_interval": [iv[0] / cores_local, iv[1] / cores_local] if iv else None,
           "assumption": ("the knee is a machine-capacity boundary, so it scales with "
                          "the core count. Generation 17 may also change the resident "
                          "threadgroups per core, which this cannot see."),
           "how_to_test_the_core_count": (
               "no ranked probe exists. Two indirect reads: the ranked serial leg's "
               "achieved weight-stream bandwidth against Apple's published figure for "
               "each SKU, and the width at which a ranked candidate's cost curve "
               "changes slope. Both need a receipt, neither is a measurement of the "
               "runner's core count."),
           "by_cores": {}}
    for cores in RANKED_CORE_ASSUMPTIONS:
        knee = knee_per_core * cores
        shapes = {}
        for name, n, k in SCORED_SHAPES:
            per_m = {}
            for m in range(3, 10):
                tgs = r0.working_tgs(m, n)
                per_m[m] = {"working_tgs": tgs, "tgs_per_core": tgs / cores,
                            "below_knee": tgs < knee,
                            "log_deficit": max(0.0, math.log(knee) - math.log(tgs))}
            shapes[name] = {"n": n, "k": k, "by_width": per_m,
                            "below_knee_at_every_width": all(v["below_knee"] for v in per_m.values()),
                            "below_knee_at_m6": per_m[6]["below_knee"]}
        out["by_cores"][str(cores)] = {
            "knee_working_tgs": knee,
            "shapes": shapes,
            "shapes_below_knee_at_m6": sorted(s for s, v in shapes.items() if v["below_knee_at_m6"]),
            "n_threshold_at_m6_groups1": knee * 8,
        }
    return out


def r3_bracket_crosscheck(fit: dict) -> dict:
    """Compare the fitted knee with ledger item 157's independent R3 bracket.

    R3 was measured at identical n and identical traffic, so it isolates the
    same occupancy term this census fits, on the same host family, without
    sharing an instrument with it.
    """
    lo, hi = E33_R3_KNEE_BRACKET
    knee = fit["knee_working_tgs"]
    interval = fit.get("knee_interval_working_tgs")
    overlaps = bool(interval and interval[0] <= hi and interval[1] >= lo)
    return {
        "prior_bracket_working_tgs": [lo, hi],
        "prior_implied_A": E33_R3_IMPLIED_A,
        "fitted_knee_working_tgs": knee,
        "fitted_A": fit["A_per_log_deficit"],
        "point_inside_bracket": bool(lo <= knee <= hi),
        "interval_overlaps_bracket": overlaps,
        "A_ratio_fitted_over_prior": fit["A_per_log_deficit"] / E33_R3_IMPLIED_A,
        "note": ("independent corroboration, not a pre-registered gate. The "
                 "pre-registered E33 control is the sign-flip window."),
    }


def recommendation(fit: dict, cores_local: int, width_taxes: dict) -> dict:
    """Rung 3. A recommended IPG per (M, out_vec_size band), with predictions.

    Cost model at fixed width and fixed shape, in the fitted log units:

        cost(IPG) = A * max(0, ln(knee) - ln(working_tgs))   # occupancy, fitted
                  + G * (groups - 1)                         # weight passes
                  + depth term

    The two terms come from different instruments and that is deliberate. A is
    fitted here, but G is NOT identified by this design: every family measured
    has out_vec_size >= 4096, so one width has one IPG for all five families and
    the per-width intercept absorbs the whole group term. G is therefore
    imported from ledger item 157 R1. Each cell is scored at the low, point and
    high value of G, and a recommendation counts as robust only when the
    argument of the minimum does not move across that range.
    """
    a = fit["A_per_log_deficit"]
    knee = fit["knee_working_tgs"]
    pen = lambda t: max(0.0, math.log(knee) - math.log(t))
    g_variants = {"g_zero": 0.0, "g_point": G_GROUP_PASS_LOG,
                  "g_high": G_GROUP_PASS_LOG_RANGE[1]}

    bands = [("1024-4095 pair kernel", 2048), ("4096-8191", 5120),
             ("8192-16383", 14336), ("16384-32767", 16480), ("32768+", 34816)]
    table = {}
    for m in range(3, 10):
        for band_name, n_rep in bands:
            shipped_ipg = r0.ipg_for(m, n_rep)
            candidates = {}
            for ipg in (2, 3, 4, 5, 6):
                if ipg > m or m % ipg == 1:
                    continue  # wrapper static_asserts, quantized.h:1168-1169
                groups = math.ceil(m / ipg)
                tgs = groups * math.ceil(n_rep / 8)
                occ = a * pen(tgs)
                candidates[ipg] = {
                    "working_tgs": tgs, "tgs_per_core": tgs / cores_local,
                    "groups": groups,
                    "occupancy_log_cost": occ,
                    "predicted_log_cost": {
                        k: occ + g * (groups - 1)
                        for k, g in g_variants.items()},
                }
            if not candidates:
                continue
            argmin = {k: min(candidates,
                             key=lambda i: candidates[i]["predicted_log_cost"][k])
                      for k in g_variants}
            best = argmin["g_point"]
            safe = min((i for i in candidates if candidates[i]["groups"] <= 2),
                       key=lambda i: candidates[i]["predicted_log_cost"]["g_point"])
            shipped = candidates.get(shipped_ipg)
            gain = None
            if shipped is not None:
                gain = {k: 1 - math.exp(candidates[best]["predicted_log_cost"][k]
                                        - shipped["predicted_log_cost"][k])
                        for k in g_variants}
            # The 1024-4095 band runs the separate pair kernel at NA=2 for every
            # width, quantized.h:1980-2023. Its constraints are not the IPG-table
            # constraints, and no family in this census lands in it, so the model
            # is outside its identified domain there.
            applies = n_rep >= 4096
            table[f"M{m}/{band_name}"] = {
                "M": m, "band": band_name, "representative_n": n_rep,
                "model_applies": applies,
                "shipped_ipg": shipped_ipg,
                "recommended_ipg": best if applies else shipped_ipg,
                "recommended_ipg_groups_le_2": safe if applies else shipped_ipg,
                "change": bool(applies and best != shipped_ipg),
                "argmin_by_g": argmin,
                "robust_to_group_pass_cost": len(set(argmin.values())) == 1,
                "predicted_local_gain_fraction_of_that_cell_width_tax": gain,
                "candidates": candidates,
                "extrapolated_group_count": bool(candidates[best]["groups"] > 2),
                "uniform_group_term_not_identified_here": True,
                "uniform_group_term_source":
                    "ledger item 157 R1, second weight pass at M=6, same host family",
            }

    # Value, expressed only as a share of the measured verify-width tax.
    value = {}
    for m, mix in sorted(RANKED_WIDTH_MIX.items()):
        cells = [v for v in table.values() if v["M"] == m and v["change"]]
        value[str(m)] = {
            "ranked_width_share": mix,
            "cells_recommended_to_change": [c["band"] for c in cells],
        }
    return {"table": table, "by_width": value,
            "measured_width_tax_share": width_taxes,
            "note": ("no ranked score conversion. Value is expressed as a fraction of "
                     "the measured verify-width tax, per the assignment.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--prior", default="research/out/e71-census-r1/census.json")
    ap.add_argument("--rung0", default="research/out/e74-rung0.json")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cores, core_source = r0.gpu_core_count()
    sessions = {"e74": reduce_session(args.census)}
    if args.prior and os.path.exists(args.prior):
        sessions["e71"] = reduce_session(args.prior)
    for label, sess in sessions.items():
        sess["null"] = null_control(sess["per_arm"])

    report = {
        "experiment": "e74-in-situ-threadgroup-knee",
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "device": {"gpu_cores": cores, "core_count_source": core_source,
                   "architecture": sessions["e74"]["payload"]["identity"]["device"]["architecture"]},
        "identity": sessions["e74"]["payload"]["identity"],
    }

    # --- rung 1: curve, controls, surface ------------------------------------
    report["rung1"] = {
        label: {
            "curve": {str(w): v for w, v in sess["curve"].items()},
            "session_null_ms": max((v["half_range_ms"] for v in sess["curve"].values()),
                                   default=0.0),
            "null_control": sess["null"],
            "arm_taxes": {arm: {str(w): {"tax_ms": q["tax_ms"],
                                         "arm_half_range_ms": q["arm_half_range_ms"],
                                         "baseline_half_range_ms": q["baseline_half_range_ms"],
                                         "entry_temps_c": q["entry_temps_c"]}
                                for w, q in sorted(bw.items())}
                          for arm, bw in sorted(sess["per_arm"].items())},
        }
        for label, sess in sessions.items()
    }

    rows = cell_surface(sessions, cores)
    report["surface"] = rows
    report["width_stats"] = {label: width_stats(rows, label) for label in sessions}

    # Cross-session bridge on the widths both sessions measured.
    bridge = {}
    for w in sorted({r["width"] for r in rows if r["session"] == "e74"}):
        a = {r["family"]: r["ms_per_gb"] for r in rows if r["session"] == "e74" and r["width"] == w}
        b = {r["family"]: r["ms_per_gb"] for r in rows if r["session"] == "e71" and r["width"] == w}
        shared = sorted(set(a) & set(b))
        if shared:
            bridge[str(w)] = {"families": shared,
                              "e74_over_e71": {f: a[f] / b[f] for f in shared},
                              "median_ratio": statistics.median([a[f] / b[f] for f in shared])}
    report["cross_session_bridge"] = bridge

    # --- rung 2: the knee fit -------------------------------------------------
    primary_rows = [r for r in rows if r["session"] == "e74" and r["resolved"]]
    pooled_rows = [r for r in rows if r["resolved"]]
    fit = fit_knee(primary_rows)
    report["rung2"] = {
        "primary_fit": {"scope": "E74 session only, resolved cells", **fit},
        "pooled_fit": {"scope": "both sessions, resolved cells",
                       **fit_knee(pooled_rows)},
        "bootstrap": bootstrap_knee(primary_rows),
    }
    report["rung2"]["primary_fit"].pop("profile", None)
    report["rung2"]["pooled_fit"].pop("profile", None)
    report["rung2"]["knee_profile"] = [{"knee": p["knee"], "sse": p["sse"], "A": p["A"]}
                                       for p in fit.get("profile", [])]

    tax_by_family = {r["family"]: {"tax_ms": r["tax_ms"], "gb": r["gb"]}
                     for r in rows if r["session"] == "e74" and r["width"] == 6}
    if not tax_by_family:
        tax_by_family = {r["family"]: {"tax_ms": r["tax_ms"], "gb": r["gb"]}
                         for r in rows if r["session"] == "e71" and r["width"] == 6}
    if fit.get("resolved"):
        report["rung2"]["positive_control_e33"] = e33_positive_control(fit, tax_by_family)
        report["rung2"]["independent_r3_bracket"] = r3_bracket_crosscheck(fit)
        report["rung2"]["ranked_extrapolation"] = ranked_extrapolation(fit, cores)
        total_tax = sum(v["tax_ms"] for v in tax_by_family.values())
        width_taxes = {f: v["tax_ms"] / total_tax for f, v in tax_by_family.items()}
        report["rung3"] = recommendation(fit, cores, width_taxes)
    else:
        report["rung2"]["positive_control_e33"] = {"skipped": "fit unresolved"}
        report["rung2"]["ranked_extrapolation"] = {"skipped": "fit unresolved"}
        report["rung3"] = {"skipped": "fit unresolved"}

    # --- pre-registration verdict --------------------------------------------
    if os.path.exists(args.rung0):
        pre = json.load(open(args.rung0))["preregistration"]
        ws = report["width_stats"]["e74"]
        got = {w: ws[w]["D_over_level"] for w in ("7", "8") if w in ws}
        knee_band = pre["primary_statistic_D"]["H_knee_prediction"]["D_over_level_7_8"]
        null_band = pre["primary_statistic_D"]["H_null_prediction"]["D_over_level_7_8"]
        verdict = "unresolved"
        if len(got) < 2:
            verdict = "incomplete: needs both M=7 and M=8"
        elif all(v <= knee_band[1] for v in got.values()):
            verdict = "knee confirmed in situ"
        elif got and all(v >= null_band[0] for v in got.values()):
            verdict = "knee falsified in situ"
        report["preregistration_verdict"] = {
            "D_over_level_measured": got,
            "H_knee_band": knee_band, "H_null_band": null_band,
            "verdict": verdict,
            "R_measured": {w: ws[w]["R"] for w in ws},
            "spread_measured": {w: ws[w]["spread_abs"] for w in ws},
            "per_family_predictions": pre["per_family_point_predictions"],
        }

    out_path = args.json or os.path.join(os.path.dirname(args.census), "report.json")
    json.dump(report, open(out_path, "w"), indent=1, sort_keys=True)
    print(f"wrote {out_path}")

    ws = report["width_stats"]["e74"]
    print("\nM  grp IPG  level   D      D/L    R      spread")
    for w in sorted(ws, key=int):
        v = ws[w]
        print(f"{w}  {v['groups']}   {v['ipg']}   {v['level']:6.3f} {v['D']:+6.3f} "
              f"{v['D_over_level']:+6.3f} {v['R']:+6.3f} {v['spread_abs']:6.3f}")
    print(f"\nknee {fit.get('knee_working_tgs', float('nan')):.0f} tgs "
          f"({fit.get('knee_working_tgs', float('nan')) / cores:.1f}/core), "
          f"A={fit.get('A_per_log_deficit', float('nan')):.4f}, "
          f"interval {fit.get('knee_interval_working_tgs')}")
    print("verdict:", report.get("preregistration_verdict", {}).get("verdict"))
    pc = report["rung2"]["positive_control_e33"]
    if "skipped" in pc:
        print("E33 control skipped:", pc["skipped"])
    else:
        print("E33 control passed:", pc["passed"],
              "| converted flip at", round(pc["tax_share_converted"]["predicted_sign_flip_shipped_tgs"]),
              "| raw flip at", round(pc["raw_variant"]["predicted_sign_flip_shipped_tgs"]),
              "| gate window [3584, 4120]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
