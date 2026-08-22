#!/usr/bin/env python3
"""E118: reduce the metadata-load arms, and name the binding resource.

    research/e118_analysis.py --rate research/out/e118-full/rate.json \
        --census research/e118-artifacts/census.json \
        --out research/e118-artifacts/summary.json

The identified-set columns need four board facts per prompt. CAMPAIGN RULE 40
forbids an analysis script reading a gitignored host-local path for a number
that reaches the report, so those facts are extracted ONCE into a committed
slice and only the slice is read afterwards:

    research/e118_analysis.py --extract-receipt /tmp/yukon-board/full.json \
        --receipt b8b8b860 --slice research/e118-artifacts/e114_receipt_slice.json

Sign convention throughout: a POSITIVE percentage means the arm is FASTER than
`a_base`. E114's arm tables use the opposite sign, so the two are never mixed in
one column here.

harness=local everywhere. Every number below comes from a standalone probe that
compiles its own copy of the kernel. Finding 28 says the `quantized` metallib is
dead for the scored worker and only the worker binary carries the arm, so
nothing here is an end-to-end measurement of the shipped path.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scoring_weights as sw  # noqa: E402

HEADLINE_SHAPE = "mlp_gate_up_k5120_n34816"
WARMUP_BLOCKS = 1
KILL_RULE_PCT = 0.5
# From the assignment: the arms whose mechanism the primary metric may rank.
# `p_prefetch_w` is bit exact but spills on g16s at the widths that matter, so
# it is reported beside them and never inside the headline.
PROMOTION_ARMS = ("s_bcast", "s_bcast_all", "s_bcast_scale", "p_split_meta",
                  "g_pack32", "s_bcast_pack32")
DIAGNOSTIC_ARMS = ("n_nosums", "l_loadonly", "n_nobias", "d_bias1",
                   "y_algebra", "y_hsum_tree", "n_halfsums",
                   "n_halfsums_free")
# Rung 2, Finding 53. These are bit exact but they are NOT metadata-load arms,
# so they are reported beside the primary metric and never inside it. Moving
# the metric's arm set after seeing their numbers would be goalpost-shifting;
# they carry their own pre-registered bar of +1.0 % instead.
RUNG2_EXACT_ARMS = ("x_sumshare_min", "x_sumshare_split", "x_sumshare_owner")
RUNG2_BAR_PCT = 1.0
# E111's bias arms, folded in at every width. `e_bias6` is bit exact but is not
# part of the metadata-load screen: it needs a repacked metadata array that this
# experiment does not build, so it is reported beside the primary metric.
BIAS_ARMS = ("n_nobias", "n_nosums", "d_bias1", "e_bias6")
# `z_ballast` carries no candidate mechanism. It adds dead loop-carried
# arithmetic that never reaches `y`, only to force the compiler to spill, so it
# separates "this arm's mechanism broke exactness" from "spilling broke
# exactness". It is a control and never enters the primary metric.
CONTROL_ARMS = ("z_ballast",)
# The instruction-price ladder. Each rung injects an exactly known number of
# instructions of ONE class per k-block iteration while holding the register
# footprint fixed, so `a_base` at zero plus the rungs give a measured slope in
# percent of `a_base` per instruction. `k_alu16w` is the ILP control: the same
# 16 injected ALU instructions as `k_alu16` over four chains instead of two.
CAL_LADDER = {
    "alu": (("k_alu8", 8), ("k_alu16", 16)),
    "ld": (("k_ld8", 8), ("k_ld16", 16)),
    "shuf": (("k_shuf8", 8), ("k_shuf16", 16)),
}
CAL_ILP_CONTROL = ("k_alu16w", "k_alu16", 16)
CAL_ARMS = tuple(a for rungs in CAL_LADDER.values() for a, _ in rungs) + \
    (CAL_ILP_CONTROL[0],)
# The two arithmetic-axis ceilings. Both reassociate, so neither is promotable.
CEILING_ARMS = ("y_algebra", "y_hsum_tree")

# Metadata instructions issued per k-block iteration, counted from each arm's
# own source. `a_base` issues 4 rows x (1 scale + 1 bias) = 8 device loads and
# no shuffles, and every entry below is the difference from that.
#
#   s_bcast         4 x (2 predicated loads + 2 shuffles). Predication masks
#                   lanes, it does not delete the instruction, so the load
#                   count is UNCHANGED and only shuffles are added.
#   s_bcast_all     2 fully active loads hoisted out of the r loop, then
#                   4 x 2 shuffles. This is the only arm that truly deletes
#                   metadata load instructions.
#   s_bcast_scale   scales broadcast, biases loaded as shipped.
#   g_pack32        4 x 1 interleaved uint32 load, no shuffles.
#   s_bcast_pack32  4 x (1 predicated uint32 load + 1 shuffle).
#   p_split_meta    the same instructions in a different order.
ARM_META_DELTA = {
    "s_bcast": {"ld": 0, "shuf": 8},
    "s_bcast_all": {"ld": -6, "shuf": 8},
    "s_bcast_scale": {"ld": 0, "shuf": 4},
    "g_pack32": {"ld": -4, "shuf": 0},
    "s_bcast_pack32": {"ld": -4, "shuf": 4},
    "p_split_meta": {"ld": 0, "shuf": 0},
}


def mols(rows: list[list[float]], ys: list[float],
         names: list[str]) -> dict | None:
    """Multivariate least squares with an intercept, normal equations.

    Written out rather than imported so this script keeps to the standard
    library and anybody can rerun it on a bare host. Gauss-Jordan on the
    (p+1)x(p+1) normal matrix is exact enough at this size and refuses rather
    than guesses when the design is rank deficient.
    """
    n, p = len(rows), len(rows[0])
    if n < p + 2:
        return None
    design = [[1.0] + list(r) for r in rows]
    k = p + 1
    ata = [[sum(design[i][a] * design[i][b] for i in range(n))
            for b in range(k)] for a in range(k)]
    aty = [sum(design[i][a] * ys[i] for i in range(n)) for a in range(k)]
    # augment with the identity so the same elimination yields the inverse,
    # which is what the coefficient standard errors need
    aug = [ata[a][:] + [1.0 if a == b else 0.0 for b in range(k)]
           for a in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(k):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col])]
    inv = [row[k:] for row in aug]
    beta = [sum(inv[a][b] * aty[b] for b in range(k)) for a in range(k)]
    fitted = [sum(beta[a] * design[i][a] for a in range(k)) for i in range(n)]
    resid = [y - f for y, f in zip(ys, fitted)]
    sse = sum(r * r for r in resid)
    my = sum(ys) / n
    sst = sum((y - my) ** 2 for y in ys)
    dof = n - k
    sigma2 = sse / dof if dof > 0 else float("nan")
    return {
        "intercept": beta[0],
        "coefficients": {names[a - 1]: beta[a] for a in range(1, k)},
        "standard_errors": {
            names[a - 1]: math.sqrt(sigma2 * inv[a][a])
            if sigma2 == sigma2 and sigma2 * inv[a][a] > 0 else float("nan")
            for a in range(1, k)},
        "r2": (1.0 - sse / sst) if sst > 0 else float("nan"),
        "n": n, "dof": dof,
        "residuals": resid,
    }


def ols(xs: list[float], ys: list[float]) -> dict | None:
    """Least squares through an intercept, with the slope standard error."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    se = math.sqrt(sse / (n - 2) / sxx) if n > 2 else float("nan")
    return {"slope": slope, "intercept": intercept, "se": se,
            "r2": (1.0 - sse / sst) if sst > 0 else float("nan"), "n": n}


def cost_model(rate: dict, cells: dict, census: dict | None,
               us: dict | None = None) -> dict:
    """The measured price of one instruction, by class, from the ladder.

    Every screen arm changes several things at once, so no screen arm can price
    an instruction. The ladder changes exactly one thing per class. The slope is
    reported in percent of `a_base` per injected instruction per k-block
    iteration, which is the form another experiment can apply directly.

    A rung is dropped wherever the census says it spills on the timed
    architecture, because E118's own spill defect shows spilling changes both
    the time and the correctness of this kernel.
    """
    arms = rate["arms"]
    arch = census["local_arch"] if census else None
    out: dict = {"unit": "percent of a_base per injected instruction "
                         "per k-block iteration",
                 "note": "positive means the instruction COSTS that much",
                 "classes": {}, "excluded_spilling": [], "per_cell": {}}

    def spills(arm: str, m: int) -> bool:
        if not census or arm not in census.get("arms", {}):
            return False
        rec = census["arms"][arm].get(arch, {}).get(str(m), {})
        return bool(rec.get("spill_bytes", 0))

    for klass, rungs in CAL_LADDER.items():
        if not all(a in arms for a, _ in rungs):
            continue
        fits: list[dict] = []
        for (shape, m), by_arm in sorted(cells.items()):
            xs, ys = [0.0], [0.0]
            skip = False
            for arm, count in rungs:
                if spills(arm, m):
                    skip = True
                    out["excluded_spilling"].append(
                        {"arm": arm, "shape": shape, "m": m})
                    continue
                med = statistics.median(by_arm[arm])
                xs.append(float(count))
                ys.append(-med)  # cost, not speed-up
            if skip or len(xs) < 3:
                continue
            fit = ols(xs, ys)
            if fit is None:
                continue
            # The price is taken from the DIFFERENCE between the two rungs and
            # not from the three-point fit. Both rungs carry the same injection
            # scaffold -- the extra address arithmetic and the sink that stops
            # the compiler deleting the work -- so differencing them cancels it
            # exactly, while a fit anchored on `a_base` at x=0 charges 1/16th of
            # that scaffold to every instruction. `scaffold_pct` publishes the
            # size of what was cancelled, so the reader can see how far the
            # ladder is from linear down to zero.
            lo, hi = ys[1], ys[2]
            span = xs[2] - xs[1]
            fit["slope_three_point"] = fit["slope"]
            fit["slope"] = (hi - lo) / span
            fit["scaffold_pct"] = lo - xs[1] * fit["slope"]
            # The same slope in absolute microseconds of whole-dispatch time,
            # which is the form the advisor asked for. It is cell-specific
            # because a bigger shape runs more k-block iterations.
            base_us = (us or {}).get((shape, m), {}).get("a_base")
            fit["us_per_instruction"] = (fit["slope"] * base_us / 100.0
                                         if base_us else float("nan"))
            fit["a_base_us"] = base_us
            fit.update({"shape": shape, "m": m})
            fits.append(fit)
            out["per_cell"].setdefault(klass, []).append(fit)
        if not fits:
            continue
        slopes = [f["slope"] for f in fits]
        us_slopes = [f["us_per_instruction"] for f in fits
                     if f["us_per_instruction"] == f["us_per_instruction"]]
        out["classes"][klass] = {
            "pct_per_instruction_median": statistics.median(slopes),
            "pct_per_instruction_mean": statistics.fmean(slopes),
            "sem": (statistics.stdev(slopes) / math.sqrt(len(slopes))
                    if len(slopes) > 1 else float("nan")),
            "us_per_instruction_median": (statistics.median(us_slopes)
                                          if us_slopes else float("nan")),
            "r2_median": statistics.median([f["r2"] for f in fits]),
            "slope_three_point_median": statistics.median(
                [f["slope_three_point"] for f in fits]),
            "scaffold_pct_median": statistics.median(
                [f["scaffold_pct"] for f in fits]),
            "cells": len(fits),
            # The price is also resolved per width, because a screen arm has to
            # be predicted at the width it was measured at: NA changes both the
            # register pressure and the number of k-block iterations.
            "per_na": {
                m: {"pct_per_instruction_median": statistics.median(
                        [f["slope"] for f in fits if f["m"] == m]),
                    "us_per_instruction_median": statistics.median(
                        [f["us_per_instruction"] for f in fits
                         if f["m"] == m and
                         f["us_per_instruction"] == f["us_per_instruction"]]
                        or [float("nan")]),
                    "r2_median": statistics.median(
                        [f["r2"] for f in fits if f["m"] == m]),
                    "cells": len([f for f in fits if f["m"] == m])}
                for m in sorted({f["m"] for f in fits})
            },
        }

    # The ILP control. If two chains and four chains price the same, the ladder
    # is measuring issue throughput and not dependency-chain latency.
    wide, narrow, count = CAL_ILP_CONTROL
    if wide in arms and narrow in arms:
        pairs = []
        for (shape, m), by_arm in sorted(cells.items()):
            if spills(wide, m) or spills(narrow, m):
                continue
            pairs.append((-statistics.median(by_arm[narrow]) / count,
                          -statistics.median(by_arm[wide]) / count))
        if pairs:
            two = statistics.median([a * count for a, _ in pairs])
            four = statistics.median([b * count for _, b in pairs])
            out["ilp_control"] = {
                "injected_instructions": count,
                "two_chain_cost_pct": two,
                "four_chain_cost_pct": four,
                "four_minus_two_pct": four - two,
                "two_chain_pct_per_instruction": two / count,
                "four_chain_pct_per_instruction": four / count,
                "reads": ("issue throughput" if four >= two * 0.75
                          else "dependency-chain latency"),
                "cells": len(pairs),
            }

    # Apply the prices to the screen arms. This is the whole point of the
    # ladder: it turns the metadata screen into a prediction with no free
    # parameter, so agreement is a real test of the issue model.
    #
    # The comparison is made per width and never pooled across widths. Pooling
    # is the trap: a spilling arm loses different widths from a non-spilling
    # arm, so a pooled `measured` column would compare two arms over two
    # different sets of cells and read as a difference in the arm. Each row
    # below therefore states the width, and the round-weighted column is filled
    # in only when every standing-weight width survived for that arm.
    widths_all = sorted({m for _, m in cells})
    if "ld" in out["classes"] and "shuf" in out["classes"]:
        ld_na = out["classes"]["ld"]["per_na"]
        shuf_na = out["classes"]["shuf"]["per_na"]
        preds = {}
        for arm, delta in ARM_META_DELTA.items():
            if arm not in arms:
                continue
            per_na: dict[int, dict] = {}
            for m in widths_all:
                if m not in ld_na or m not in shuf_na:
                    per_na[m] = {"status": "no_ladder_price"}
                    continue
                if spills(arm, m):
                    per_na[m] = {"status": "arm_spills"}
                    continue
                obs = [statistics.median(by_arm[arm])
                       for (shape, mm), by_arm in sorted(cells.items())
                       if mm == m]
                if not obs:
                    per_na[m] = {"status": "no_cells"}
                    continue
                predicted = -(
                    delta["ld"] * ld_na[m]["pct_per_instruction_median"] +
                    delta["shuf"] * shuf_na[m]["pct_per_instruction_median"])
                per_na[m] = {
                    "status": "ok",
                    "predicted_pct_faster": predicted,
                    "measured_pct_faster": statistics.median(obs),
                    "residual_pct": statistics.median(obs) - predicted,
                    "shapes": len(obs),
                }
            ok = {m: v for m, v in per_na.items() if v["status"] == "ok"}
            complete = all(m in ok for m in sw.STANDING_WEIGHTS)
            wsum = sum(sw.STANDING_WEIGHTS[m] for m in ok
                       if m in sw.STANDING_WEIGHTS)
            preds[arm] = {
                "delta_device_loads": delta["ld"],
                "delta_shuffles": delta["shuf"],
                "per_na": per_na,
                "widths_ok": sorted(ok),
                "widths_dropped": sorted(m for m in per_na
                                         if per_na[m]["status"] != "ok"),
                "complete_over_standing_weights": complete,
                # Renormalised over the widths that survived, with the covered
                # standing weight printed next to it so the reader can see how
                # much of the round the number actually speaks for. It is never
                # comparable across two arms with different coverage, which is
                # exactly why the coverage travels with the number.
                "standing_weight_covered": wsum,
                "weighted_predicted_over_covered": (
                    sum(sw.STANDING_WEIGHTS[m] * ok[m]["predicted_pct_faster"]
                        for m in ok if m in sw.STANDING_WEIGHTS) / wsum
                    if wsum else None),
                "weighted_measured_over_covered": (
                    sum(sw.STANDING_WEIGHTS[m] * ok[m]["measured_pct_faster"]
                        for m in ok if m in sw.STANDING_WEIGHTS) / wsum
                    if wsum else None),
            }
        out["screen_prediction"] = preds
    return out


AIR_REGRESSORS = ("device_loads", "shuffles", "arithmetic_lanes",
                  "convert", "address")


def air_regression(cells: dict, census: dict | None, us: dict) -> dict | None:
    """The observational regression: measured microseconds on AIR counts.

    This is the model the assignment asked for in its literal form -- every arm
    and every cell, time against the static AIR instruction count per entry
    point. It is reported next to the ladder rather than instead of it, because
    the two disagree and the disagreement is the point. Every arm here changes
    several categories at once and changes register pressure as well, so the
    coefficients are correlational; the ladder holds everything but one class
    fixed, so its slopes are causal.

    Spilling arm-widths are dropped: a spilled kernel is a different machine and
    its time is not a function of the AIR count.
    """
    if not census:
        return None
    arch = census["local_arch"]
    air = {a: r.get("air", {}) for a, r in census.get("arms", {}).items()}
    out: dict = {"note": "observational, correlational, not causal; the "
                         "calibration ladder is the causal instrument",
                 "regressors": list(AIR_REGRESSORS), "per_cell": {}}
    uni, multi = [], []
    for (shape, m), by_arm in sorted(cells.items()):
        xs_uni, xs_multi, ys, used = [], [], [], []
        for arm in sorted(by_arm):
            counts = air.get(arm, {}).get(str(m))
            base = census.get("arms", {}).get(arm, {}).get(arch, {}).get(str(m))
            if counts is None or base is None or base.get("spill_bytes"):
                continue
            value = us.get((shape, m), {}).get(arm)
            if value is None:
                continue
            xs_uni.append([float(counts.get("issue_lanes", 0))])
            xs_multi.append([float(counts.get(k, 0)) for k in AIR_REGRESSORS])
            ys.append(value)
            used.append(arm)
        if len(ys) < len(AIR_REGRESSORS) + 3:
            continue
        fit_u = mols(xs_uni, ys, ["issue_lanes"])
        fit_m = mols(xs_multi, ys, list(AIR_REGRESSORS))
        if fit_u is None or fit_m is None:
            continue
        row = {"shape": shape, "m": m, "arms": used,
               "univariate": {k: fit_u[k] for k in
                              ("intercept", "coefficients",
                               "standard_errors", "r2", "n")},
               "multivariate": {k: fit_m[k] for k in
                                ("intercept", "coefficients",
                                 "standard_errors", "r2", "n")},
               "worst_residuals": sorted(
                   ({"arm": a, "residual_us": r}
                    for a, r in zip(used, fit_u["residuals"])),
                   key=lambda d: -abs(d["residual_us"]))[:5]}
        out["per_cell"]["%s|NA%d" % (shape, m)] = row
        uni.append(fit_u)
        multi.append(fit_m)
    if not uni:
        return None
    out["univariate_summary"] = {
        "us_per_issue_lane_median": statistics.median(
            [f["coefficients"]["issue_lanes"] for f in uni]),
        "se_median": statistics.median(
            [f["standard_errors"]["issue_lanes"] for f in uni]),
        "r2_median": statistics.median([f["r2"] for f in uni]),
        "cells": len(uni),
    }
    out["multivariate_summary"] = {
        k: {"us_per_instruction_median": statistics.median(
                [f["coefficients"][k] for f in multi]),
            "se_median": statistics.median([f["standard_errors"][k]
                                            for f in multi])}
        for k in AIR_REGRESSORS}
    out["multivariate_summary"]["r2_median"] = statistics.median(
        [f["r2"] for f in multi])
    out["multivariate_summary"]["cells"] = len(multi)
    return out


def block_dispersion(rate: dict, cells: dict) -> dict:
    """Harness defect 19: a rare low-clock tail that pooling would hide.

    Thorfinn's E115 saw a whole timed region run at about half speed in four
    blocks of eight. That is external interruption, not first-position ramp
    bias, so the block-0 discard does not catch it. Any kept block whose time
    exceeds 1.5x the cell median is flagged here rather than pooled silently.
    """
    flagged, cell_rows = [], []
    per_cell: dict[tuple, dict[str, list[float]]] = {}
    for row in rate["measurements"]:
        if row["kind"] != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        key = (row["shape"], row["m"])
        for arm, sec in row["seconds"].items():
            per_cell.setdefault(key, {}).setdefault(arm, []).append(sec * 1e6)
    for (shape, m), by_arm in sorted(per_cell.items()):
        worst = 0.0
        for arm, vals in by_arm.items():
            med = statistics.median(vals)
            if med <= 0:
                continue
            spread = (max(vals) - min(vals)) / med * 100.0
            worst = max(worst, spread)
            for i, v in enumerate(vals):
                if v > 1.5 * med:
                    flagged.append({"shape": shape, "m": m, "arm": arm,
                                    "block_index": i, "us": v,
                                    "cell_median_us": med,
                                    "ratio": v / med})
        cell_rows.append({"shape": shape, "m": m,
                          "worst_arm_spread_pct": worst})
    return {"threshold_ratio": 1.5, "flagged": flagged,
            "flagged_count": len(flagged), "per_cell": cell_rows,
            "max_cell_spread_pct": max((c["worst_arm_spread_pct"]
                                        for c in cell_rows), default=0.0)}


def rung2_decomposition(delta_by_arm: dict, census: dict | None,
                        widths: list[int], shape: str) -> dict:
    """Finding 53, decomposed into mechanism, scaffolding and exchange.

    The registered ceiling `n_halfsums` puts a uniform branch around a
    duplicated `i` loop, so it prices the mechanism and the scaffolding
    together. `n_halfsums_free` drops the same half of the add tree at compile
    time in both simdgroups: same executed instruction count, no branch and no
    duplication, so it is the mechanism alone. The three differences below are
    therefore identified without any fitted parameter.
    """
    rows = []
    for na in widths:
        def g(arm: str) -> float | None:
            return delta_by_arm.get(arm, {}).get(na)
        free, reg = g("n_halfsums_free"), g("n_halfsums")
        mn, sp, ow = (g("x_sumshare_min"), g("x_sumshare_split"),
                      g("x_sumshare_owner"))
        rows.append({
            "na": na,
            "mechanism_ceiling_pct": free,
            "registered_ceiling_pct": reg,
            "body_duplication_cost_pp": (
                None if free is None or reg is None else free - reg),
            "min_exchange_pct": mn,
            "exchange_cost_pp": (
                None if free is None or mn is None else free - mn),
            "split_pct": sp,
            "owner_pct": ow,
            "captured_fraction": (
                None if not free or mn is None else mn / free),
        })
    occ = []
    if census is not None:
        for arm in ("a_base", "n_halfsums", "n_halfsums_free",
                    "x_sumshare_min", "x_sumshare_split", "x_sumshare_owner"):
            row = census["arms"].get(arm)
            if row is None:
                continue
            for na in widths:
                air = row["air"].get(str(na), {})
                tgmap = row.get("threadgroup_bytes") or {}
                tg = tgmap.get(str(na), tgmap.get(na, 0)) or 0
                budget = row.get("threadgroup_budget_bytes", 32768)
                cell = {"arm": arm, "na": na,
                        "threadgroup_bytes": tg,
                        "threadgroup_budget_bytes": budget,
                        "threadgroups_allowed_by_shared_memory":
                            (budget // tg) if tg else None,
                        "air_threadgroup_ops": air.get("threadgroup"),
                        "air_total_instructions": air.get(
                            "total_instructions")}
                for arch in (census["local_arch"], census["ranked_arch"]):
                    v = row.get(arch, {}).get(str(na))
                    if v is None:
                        continue
                    cell[arch] = {"registers": v["registers"],
                                  "spill_bytes": v["spill_bytes"],
                                  "text_bytes": v["text_bytes"]}
                occ.append(cell)
    return {"shape": shape, "rows": rows, "occupancy": occ,
            "note": "n_halfsums and n_halfsums_free are DIAGNOSTIC and are not "
                    "bit exact; x_sumshare_min, x_sumshare_split and "
                    "x_sumshare_owner are bit exact"}


def print_rung2(d: dict) -> None:
    print("\n-- rung 2, Finding 53 decomposed on %s" % d["shape"])
    print("   `free` drops the same half of the add tree at compile time in "
          "both simdgroups: the mechanism with no branch and no duplicated "
          "body. Both ceiling arms are DIAGNOSTIC and wrong by construction; "
          "the x_sumshare_* arms are bit exact.")
    print("   %4s %10s %10s %9s %10s %9s %10s %10s %8s"
          % ("NA", "free", "n_halfsum", "dup cost", "min", "xchg cost",
             "split", "owner", "capture"))
    for r in d["rows"]:
        def f(v, w=10):
            return ("%*s" % (w, "-")) if v is None else ("%*.3f" % (w, v))
        print("   %4d %s %s %s %s %s %s %s %s"
              % (r["na"], f(r["mechanism_ceiling_pct"]),
                 f(r["registered_ceiling_pct"]),
                 f(r["body_duplication_cost_pp"], 9),
                 f(r["min_exchange_pct"]), f(r["exchange_cost_pp"], 9),
                 f(r["split_pct"]), f(r["owner_pct"]),
                 f(r["captured_fraction"], 8)))
    if d["occupancy"]:
        print("\n   threadgroup bytes against the 32768 budget, and the "
              "register/spill/text cost of each scaffolding")
        arch = sorted({k for c in d["occupancy"] for k in c
                       if k.startswith("applegpu")})
        print("   %-18s %3s %6s %7s %5s %s"
              % ("arm", "NA", "tgB", "tg/32768", "tgops",
                 "  ".join("%-18s" % a for a in arch)))
        for c in d["occupancy"]:
            cells = []
            for a in arch:
                v = c.get(a)
                cells.append("%-18s" % ("-" if v is None else "%d/%d/%d"
                                        % (v["registers"], v["spill_bytes"],
                                           v["text_bytes"])))
            print("   %-18s %3d %6d %7s %5s %s"
                  % (c["arm"], c["na"], c["threadgroup_bytes"],
                     c["threadgroups_allowed_by_shared_memory"]
                     if c["threadgroups_allowed_by_shared_memory"]
                     else "unlim", c["air_threadgroup_ops"],
                     "  ".join(cells)))


# --- the committed receipt slice ----------------------------------------------

def extract_receipt(board: str, prefix: str, out: pathlib.Path) -> int:
    """Write the four board facts per prompt that the identified set needs."""
    import e114_width_recovery as wr

    rec = wr.load_receipt(board, prefix)
    keep = ("mean_width", "p_width1", "rounds", "round_us", "mtp_us_per_token",
            "raw", "mean_draft_len", "zero_draft_rounds")
    slim = {"id": rec["id"], "status": rec["status"],
            "published": rec["published"], "source_board": board,
            "prompts": {name: {k: p[k] for k in keep}
                        for name, p in rec["prompts"].items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slim, indent=2, sort_keys=True) + "\n")
    print("wrote %s for receipt %s" % (out, rec["id"]))
    return 0


# --- reduction ----------------------------------------------------------------

def paired_pct(rate: dict) -> dict:
    """Per (shape, NA, arm) percent faster than `a_base`, block by block."""
    arms = rate["arms"]
    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rate["measurements"]:
        if row.get("kind") != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        key = (row["shape"], row["m"])
        seconds = row["seconds"]
        bucket = cells.setdefault(key, {a: [] for a in arms})
        base = seconds[arms[0]]
        for arm in arms:
            bucket[arm].append(100.0 * (base - seconds[arm]) / base)
    return cells


def absolute_us(rate: dict) -> dict:
    """Per (shape, NA, arm) median absolute microseconds, same blocks."""
    arms = rate["arms"]
    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rate["measurements"]:
        if row.get("kind") != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        bucket = cells.setdefault((row["shape"], row["m"]), {a: [] for a in arms})
        for arm in arms:
            bucket[arm].append(row["seconds"][arm] * 1e6)
    return {k: {a: statistics.median(v) for a, v in b.items()}
            for k, b in cells.items()}


def summarise(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {"n": n, "median": statistics.median(values), "mean": mean,
            "sd": sd, "sem": sd / math.sqrt(n) if n > 1 else 0.0,
            "min": min(values), "max": max(values)}


def forward_reverse_gap(rate: dict) -> dict:
    """Harness defect 16 residual: slot `a` against slot `2N - 1 - a`.

    A fixed cost paid by the first timed slot of a block does not cancel in the
    palindrome mean, so it survives as a positive forward-minus-reverse gap on
    arm 0 and on nothing else. E115 saw +61.6 % on arm 0 with every other arm
    under 0.4 %. Both the all-block and the post-warm-up figure are reported,
    because the first is what the fix has to remove.
    """
    arms = rate["arms"]
    n = len(arms)
    allb: dict[str, list[float]] = {a: [] for a in arms}
    kept: dict[str, list[float]] = {a: [] for a in arms}
    for row in rate["measurements"]:
        if row.get("kind") != "timing":
            continue
        slots = row["slots"]
        for i, arm in enumerate(arms):
            fwd, rev = slots[i], slots[2 * n - 1 - i]
            gap = 100.0 * (fwd - rev) / rev
            allb[arm].append(gap)
            if row["block"] >= WARMUP_BLOCKS:
                kept[arm].append(gap)
    return {arm: {"all_blocks_median_pct": statistics.median(allb[arm]),
                  "all_blocks_max_abs_pct": max(abs(v) for v in allb[arm]),
                  "post_warmup_median_pct": statistics.median(kept[arm]),
                  "post_warmup_max_abs_pct": max(abs(v) for v in kept[arm])}
            for arm in arms}


def fidelity_rows(rate: dict) -> dict:
    exact_failures, control_failures, diag_seen = [], [], []
    weakened, base_ref = [], []
    for row in rate["measurements"]:
        if row.get("kind") == "fidelity":
            if row.get("base_nonfinite"):
                weakened.append({"shape": row["shape"], "m": row["m"],
                                 "base_nonfinite": row["base_nonfinite"],
                                 "base_elements": row["base_elements"]})
            base_ref.append({"shape": row["shape"], "m": row["m"],
                             "max_rel": row["base_vs_double_max_rel"],
                             "rms": row["base_vs_double_rms_over_signal"]})
            for entry in row["arms"]:
                if entry["exact_required"] and not entry["bit_identical"]:
                    exact_failures.append(
                        {"shape": row["shape"], "m": row["m"], **entry})
                if not entry["exact_required"]:
                    diag_seen.append(entry["arm"])
        elif row.get("kind") == "positive_control" and not row["detected"]:
            control_failures.append(row)
    return {"exact_failures": exact_failures,
            "control_failures": control_failures,
            "screen_weakened_by_nonfinite": weakened,
            "base_vs_double": base_ref,
            "diagnostic_arms_seen": sorted(set(diag_seen))}


def spill_exactness(rate: dict, census: dict | None) -> dict | None:
    """Join each arm's exactness verdict to its spill bytes on the local arch.

    An arm that disagrees with `a_base` may be wrong for its own reasons. The
    join answers a narrower question: does the disagreement track the compiler
    spilling rather than the mechanism? `z_ballast` decides it, because it
    spills without changing any arithmetic that reaches the output.
    """
    if census is None:
        return None
    arch = census["local_arch"]
    per_arm: dict[str, dict[int, dict]] = {}
    for row in rate["measurements"]:
        if row.get("kind") != "fidelity":
            continue
        for entry in row["arms"]:
            if not entry["exact_required"]:
                continue
            cell = per_arm.setdefault(entry["arm"], {}).setdefault(
                row["m"], {"exact": True, "shapes_wrong": [],
                           "vs_double_max_rel": 0.0})
            if not entry["bit_identical"]:
                cell["exact"] = False
                cell["shapes_wrong"].append(row["shape"])
            cell["vs_double_max_rel"] = max(cell["vs_double_max_rel"],
                                            entry.get("vs_double_max_rel", 0.0))
    out = {"arch": arch, "arms": {}}
    for arm, by_na in per_arm.items():
        static = census["arms"].get(arm, {}).get(arch, {})
        out["arms"][arm] = {
            str(na): {"spill_bytes": (static.get(str(na)) or {}).get(
                          "spill_bytes", 0) or 0,
                      "registers": (static.get(str(na)) or {}).get("registers"),
                      "exact": cell["exact"],
                      "shapes_wrong": cell["shapes_wrong"],
                      "vs_double_max_rel": cell["vs_double_max_rel"]}
            for na, cell in sorted(by_na.items())}
    pairs = [(c["spill_bytes"], c["exact"])
             for arm in out["arms"].values() for c in arm.values()]
    wrong = [s for s, e in pairs if not e]
    right = [s for s, e in pairs if e]
    out["max_spill_while_exact"] = max(right) if right else None
    out["min_spill_while_wrong"] = min(wrong) if wrong else None
    out["separates"] = bool(
        wrong and right and min(wrong) > max(right))
    return out


# --- weighting ----------------------------------------------------------------

def identified_range(delta: dict[int, float], slice_path: pathlib.Path,
                     rates: dict) -> tuple[float, float]:
    """Exact extremum of the published-weighted arm value over E114's set."""
    import e114_rerank as rr

    rec = json.loads(slice_path.read_text())
    rec["prompts"] = {k: dict(v) for k, v in rec["prompts"].items()}
    mix = rr.prompt_mix(rec)
    policy, _ = rr.load_policy_shapes("research/e114-artifacts/rung1b.json")
    vsets, _ = rr.build(rec, rates, True, policy)
    lo = rr.arm_range(delta, vsets, mix, rec, rates, hi=False)
    hi = rr.arm_range(delta, vsets, mix, rec, rates, hi=True)
    return lo, hi


def bias_axis(per_arm_na: dict[str, dict[int, float]],
              weights: dict[int, float], widths: list[int]) -> dict | None:
    """E111's bias arms, decomposed and reduced at every width.

    E111 ran these at NA 5 only. Two of the four are deliberately wrong and
    price a ceiling; `e_bias6` is the real, bit-exact recoding. Two differences
    are also reported, because `n_nobias - n_nosums` isolates the bias LOAD from
    its arithmetic and `d_bias1 - e_bias6` isolates the reconstruction from the
    one-byte read.
    """
    if any(arm not in per_arm_na for arm in BIAS_ARMS):
        return None
    if set(widths) != set(sw.NA_CELLS):
        return None

    def row(table: dict[int, float]) -> dict:
        return {"na": table, "weighted": sw.weighted(table, weights)}

    diff = {m: per_arm_na["n_nobias"][m] - per_arm_na["n_nosums"][m]
            for m in widths}
    recon = {m: per_arm_na["d_bias1"][m] - per_arm_na["e_bias6"][m]
             for m in widths}
    return {"whole_bias_axis": row(per_arm_na["n_nobias"]),
            "bias_arithmetic": row(per_arm_na["n_nosums"]),
            "bias_load": row(diff),
            "bias6_ceiling": row(per_arm_na["d_bias1"]),
            "bias6_real": row(per_arm_na["e_bias6"]),
            "bias6_reconstruction": row(recon)}


def point_shapes(delta: dict[int, float]) -> dict[str, float]:
    """The four E114 candidate shapes. Every one FAILED its own rung-0 gate."""
    table = json.loads(
        pathlib.Path("research/e114-artifacts/rung1.json").read_text())
    out = {}
    for shape, shift in table["delta_weights"].items():
        weights = {na: sw.STANDING_WEIGHTS[na] + shift[str(na)]
                   for na in sw.NA_CELLS}
        out[shape] = sw.weighted(delta, weights)
    return out


# --- report -------------------------------------------------------------------

def report(rate_path: pathlib.Path, census_path: pathlib.Path | None,
           slice_path: pathlib.Path | None, out_path: pathlib.Path | None,
           shape: str) -> int:
    rate = json.loads(rate_path.read_text())
    arms = rate["arms"]
    cells = paired_pct(rate)
    us = absolute_us(rate)
    widths = sorted({m for (s, m) in cells if s == shape})
    shapes = sorted({s for (s, _) in cells})

    print("=" * 96)
    print("E118 - the metadata-load instruction axis of the wide affine-4 QMV")
    print("=" * 96)
    print("harness=local   device %s   architecture %s" %
          (rate["device"], rate["architecture"]))
    print("standalone probe, own kernel copy: NOT an end-to-end measurement")
    print("blocks per cell %d, first %d discarded, palindrome order, "
          "cool_gate_passed_real_gate=false, gate_qualified_for_timing=false"
          % (rate["pairs"], WARMUP_BLOCKS))
    print("sign convention: POSITIVE percent means FASTER than a_base")

    fid = fidelity_rows(rate)
    print("\n-- fidelity")
    print("   exact-arm failures  : %d" % len(fid["exact_failures"]))
    print("   positive-control failures: %d" % len(fid["control_failures"]))
    print("   diagnostic arms (difference expected): %s"
          % ", ".join(fid["diagnostic_arms_seen"]))
    for row in fid["exact_failures"]:
        print("   FAILED %-16s %s NA%d  %d/%d cells  max_ulp=%s max_rel=%s"
              % (row["arm"], row["shape"], row["m"], row["differing"],
                 row["total"], row.get("max_ulp", "?"),
                 ("%.3e" % row["max_rel"]) if "max_rel" in row else "?"))

    gaps = forward_reverse_gap(rate)
    print("\n-- harness defect 16 residual, forward slot against reverse slot")
    print("   %-16s %12s %12s %12s %12s"
          % ("arm", "all med %", "all |max| %", "kept med %", "kept |max| %"))
    for arm in arms:
        g = gaps[arm]
        print("   %-16s %12.3f %12.3f %12.3f %12.3f"
              % (arm, g["all_blocks_median_pct"], g["all_blocks_max_abs_pct"],
                 g["post_warmup_median_pct"], g["post_warmup_max_abs_pct"]))

    print("\n-- %s, percent faster than a_base, median over kept blocks "
          "(sem in brackets)" % shape)
    header = "   %-16s" % "arm" + "".join("  %18s" % ("NA%d" % m)
                                          for m in widths)
    print(header)
    per_arm_na: dict[str, dict[int, float]] = {}
    stats: dict[str, dict[int, dict]] = {}
    for arm in arms:
        line = "   %-16s" % arm
        per_arm_na[arm], stats[arm] = {}, {}
        for m in widths:
            st = summarise(cells[(shape, m)][arm])
            per_arm_na[arm][m] = st["median"]
            stats[arm][m] = st
            line += "  %+10.3f (%.3f)" % (st["median"], st["sem"])
        print(line)

    print("\n-- absolute microseconds, %s, median over kept blocks" % shape)
    print("   %-16s" % "arm" + "".join("  %10s" % ("NA%d" % m) for m in widths))
    for arm in arms:
        print("   %-16s" % arm
              + "".join("  %10.1f" % us[(shape, m)][arm] for m in widths))

    census_data = (json.loads(census_path.read_text())
                   if census_path is not None and census_path.exists()
                   else None)

    # --- Finding 44 placement -------------------------------------------------
    print("\n-- Finding 44 placement on %s: a_base against its own load "
          "ceiling" % shape)
    print("   %4s %12s %12s %10s" % ("NA", "a_base us", "l_loadonly us",
                                     "gap %"))
    f44 = {}
    for m in widths:
        base_us, load_us = us[(shape, m)]["a_base"], us[(shape, m)]["l_loadonly"]
        gap = 100.0 * (base_us - load_us) / load_us
        f44[m] = {"a_base_us": base_us, "l_loadonly_us": load_us,
                  "gap_pct": gap}
        print("   %4d %12.1f %12.1f %10.2f" % (m, base_us, load_us, gap))
    weights = {na: sw.STANDING_WEIGHTS[na] for na in sw.NA_CELLS}
    if set(widths) == set(sw.NA_CELLS):
        f44_weighted = sw.weighted({m: f44[m]["gap_pct"] for m in widths},
                                   weights)
        print("   round weighted gap: %+.2f %%" % f44_weighted)
    else:
        f44_weighted = float("nan")

    # --- the headline ---------------------------------------------------------
    print("\n-- round-weighted percent faster than a_base, %s, standing "
          "weights %s" % (shape, sw.STANDING_WEIGHTS))
    rows = {}
    for arm in arms:
        if arm == "a_base" or set(widths) != set(sw.NA_CELLS):
            continue
        value = sw.weighted(per_arm_na[arm], weights)
        row = {"standing_pct": value, "na": per_arm_na[arm],
               "role": ("diagnostic" if arm in DIAGNOSTIC_ARMS
                        else "promotion" if arm in PROMOTION_ARMS
                        else "rung2exact" if arm in RUNG2_EXACT_ARMS
                        else "control" if arm in CONTROL_ARMS
                        else "other"),
               "points": point_shapes(per_arm_na[arm])}
        if slice_path is not None:
            lo, hi = identified_range(per_arm_na[arm], slice_path,
                                      sw.ONE_GROUP_GBPS)
            rlo, rhi = identified_range(per_arm_na[arm], slice_path,
                                        sw.RANKED_ONE_GROUP_GBPS)
            row["identified_local"] = [lo, hi]
            row["identified_ranked"] = [rlo, rhi]
        rows[arm] = row
    order = sorted(rows, key=lambda a: -rows[a]["standing_pct"])
    print("   %-16s %10s %10s %22s %22s"
          % ("arm", "role", "standing", "identified local", "identified ranked"))
    for arm in order:
        r = rows[arm]
        loc = ("[%+7.3f, %+7.3f]" % tuple(r["identified_local"])
               if "identified_local" in r else "n/a")
        rnk = ("[%+7.3f, %+7.3f]" % tuple(r["identified_ranked"])
               if "identified_ranked" in r else "n/a")
        print("   %-16s %10s %+10.3f %22s %22s"
              % (arm, r["role"][:10], r["standing_pct"], loc, rnk))

    print("\n   the four point shapes are DIAGNOSTIC: every one failed E114's "
          "own rung-0 gate")
    print("   %-16s %10s %10s %10s %10s" % ("arm", "maxent", "gt1", "gt2",
                                            "policy"))
    for arm in order:
        p = rows[arm]["points"]
        print("   %-16s %+10.3f %+10.3f %+10.3f %+10.3f"
              % (arm, p["maxent"], p["gt1"], p["gt2"], p["policy"]))

    best, best_value = None, float("-inf")
    for arm in PROMOTION_ARMS:
        if arm in rows and rows[arm]["standing_pct"] > best_value:
            best, best_value = arm, rows[arm]["standing_pct"]
    print("\n   PRIMARY METRIC e118_best_bit_exact_arm_round_weighted_pct_"
          "faster_vs_a_base = %+.4f  (%s)" % (best_value, best))
    print("   kill rule %+.2f %% -> %s"
          % (KILL_RULE_PCT,
             "CLEARED" if best_value >= KILL_RULE_PCT else "NOT CLEARED, null"))
    print("   the metric ranks the metadata-load arms only. `p_prefetch_w` and")
    print("   `e_bias6` are bit exact but carry other mechanisms, so they are")
    print("   reported beside it and never inside it.")

    r2best, r2value = None, float("-inf")
    for arm in RUNG2_EXACT_ARMS:
        if arm in rows and rows[arm]["standing_pct"] > r2value:
            r2best, r2value = arm, rows[arm]["standing_pct"]
    print("\n   SEPARATE, NOT THE PRIMARY METRIC: best bit-exact rung-2 "
          "reduction-sharing arm = %+.4f (%s)" % (r2value, r2best))
    print("   rung-2 pre-registered bar %+.2f %% -> %s"
          % (RUNG2_BAR_PCT,
             "CLEARED" if r2value >= RUNG2_BAR_PCT else "not cleared"))
    print("   this arm shares a reduction across simdgroups. It is not a")
    print("   metadata-load arm, so it is NOT folded into the primary metric.")
    rung2_verdict = {"best_arm": r2best, "standing_pct": r2value,
                     "bar_pct": RUNG2_BAR_PCT,
                     "cleared": bool(r2value >= RUNG2_BAR_PCT),
                     "in_primary_metric": False}
    # A width where the arm spills on THIS host but not on the ranked one is a
    # host artefact, not a property of the mechanism. Report the same number
    # again over the widths whose local measurement is not spill-confounded,
    # with the round-weight coverage stated, and never instead of the number
    # above.
    if r2best in rows and census_data is not None:
        crow = census_data["arms"].get(r2best, {})
        loc = crow.get(census_data["local_arch"], {})
        rnk = crow.get(census_data["ranked_arch"], {})
        clean = [m for m in widths
                 if (loc.get(str(m), {}).get("spill_bytes") or 0) == 0]
        cov = sum(weights.get(m, 0.0) for m in clean)
        if clean and cov > 0:
            val = (sum(weights.get(m, 0.0) * per_arm_na[r2best][m]
                       for m in clean) / cov)
            dropped = {m: {"local_spill_bytes":
                           loc.get(str(m), {}).get("spill_bytes"),
                           "ranked_spill_bytes":
                           rnk.get(str(m), {}).get("spill_bytes"),
                           "pct": per_arm_na[r2best][m]}
                       for m in widths if m not in clean}
            rung2_verdict["standing_pct_excluding_local_spill"] = val
            rung2_verdict["excluding_local_spill_coverage"] = cov
            rung2_verdict["dropped_widths"] = dropped
            print("   over the %d widths whose LOCAL build does not spill "
                  "(round-weight coverage %.3f) the same arm is %+.4f %%"
                  % (len(clean), cov, val))
            for m, d in dropped.items():
                print("   NA%d dropped: %s spills %s B locally and %s B on %s, "
                      "measured %+.3f %%"
                      % (m, r2best, d["local_spill_bytes"],
                         d["ranked_spill_bytes"], census_data["ranked_arch"],
                         d["pct"]))

    bias = bias_axis(per_arm_na, weights, widths)
    if bias is not None:
        print("\n-- E111 bias axis, folded in at every width, %s" % shape)
        print("   E111 measured these at NA 5 only, which carries %.3f of the "
              "standing weight." % sw.STANDING_WEIGHTS[5])
        print("   %-26s" % "quantity"
              + "".join("  %9s" % ("NA%d" % m) for m in widths)
              + "  %11s" % "weighted")
        for key, label in (
                ("whole_bias_axis", "n_nobias, whole axis"),
                ("bias_arithmetic", "n_nosums, arithmetic"),
                ("bias_load", "difference, the load"),
                ("bias6_ceiling", "d_bias1, Bias6 ceiling"),
                ("bias6_real", "e_bias6, real, bit exact"),
                ("bias6_reconstruction", "difference, reconstruct")):
            row = bias[key]
            print("   %-26s" % label
                  + "".join("  %+9.3f" % row["na"][m] for m in widths)
                  + "  %+11.3f" % row["weighted"])
        print("   e_bias6 is bit exact at every cell reported above; the two "
              "diagnostic")
        print("   rows above it are deliberately wrong and price ceilings "
              "only.")

    # --- every other shape ----------------------------------------------------
    print("\n-- every shape, round-weighted percent faster than a_base, "
          "standing weights")
    print("   %-16s" % "arm" + "".join("  %26s" % s for s in shapes))
    per_shape = {}
    for arm in arms:
        if arm == "a_base":
            continue
        line = "   %-16s" % arm
        per_shape[arm] = {}
        for s in shapes:
            ws = sorted({m for (ss, m) in cells if ss == s})
            if set(ws) != set(sw.NA_CELLS):
                line += "  %26s" % "-"
                continue
            table = {m: statistics.median(cells[(s, m)][arm]) for m in ws}
            value = sw.weighted(table, weights)
            per_shape[arm][s] = {"weighted_pct": value, "na": table}
            line += "  %+26.3f" % value
        print(line)

    # --- the discriminator ----------------------------------------------------
    print("\n-- discriminator")
    s_b = rows.get("s_bcast", {}).get("standing_pct", float("nan"))
    s_ba = rows.get("s_bcast_all", {}).get("standing_pct", float("nan"))
    p_sm = rows.get("p_split_meta", {}).get("standing_pct", float("nan"))
    n_ns = rows.get("n_nosums", {}).get("standing_pct", float("nan"))
    bar = KILL_RULE_PCT
    if s_b >= bar and p_sm < bar:
        verdict = "load-issue port"
    elif s_b >= bar and p_sm >= bar:
        verdict = "memory latency"
    elif n_ns >= bar and s_b < bar and p_sm < bar:
        verdict = "total instruction issue or ALU"
    else:
        verdict = "no arm cleared the bar; the probe does not select a resource"
    print("   s_bcast %+.3f   s_bcast_all %+.3f   p_split_meta %+.3f   "
          "n_nosums %+.3f" % (s_b, s_ba, p_sm, n_ns))
    print("   binding resource selected by the data: %s" % verdict)

    # --- harness defect 19, the low-clock tail --------------------------------
    disp = block_dispersion(rate, cells)
    print("\n-- harness defect 19, kept-block dispersion")
    print("   a block is flagged when it exceeds %.1fx its cell median"
          % disp["threshold_ratio"])
    print("   flagged blocks: %d of %d cell-arm series"
          % (disp["flagged_count"], len(disp["per_cell"]) * len(arms)))
    print("   worst within-cell spread over any arm: %.2f %%"
          % disp["max_cell_spread_pct"])
    for row in disp["flagged"][:12]:
        print("   FLAGGED %-14s %s NA%d  %.1f us vs median %.1f (%.2fx)"
              % (row["arm"], row["shape"], row["m"], row["us"],
                 row["cell_median_us"], row["ratio"]))

    # --- rung 2, Finding 53 ----------------------------------------------------
    rung2 = rung2_decomposition(per_arm_na, census_data, widths, shape)
    print_rung2(rung2)

    # --- the instruction price ladder -----------------------------------------
    model = cost_model(rate, cells, census_data, us)
    print("\n-- instruction price, measured from the calibration ladder")
    print("   unit: %s" % model["unit"])
    print("   price is the 8-to-16 rung contrast, which cancels the injection "
          "scaffold exactly; `scaffold` is what was cancelled and `3pt` is the "
          "biased fit that anchors on a_base at zero, shown so the bias is "
          "visible rather than hidden")
    print("   %-6s %12s %9s %13s %8s %10s %8s %6s" %
          ("class", "pct/instr", "sem", "us/instr med", "R2 med",
           "scaffold %", "3pt", "cells"))
    for klass in ("ld", "alu", "shuf"):
        c = model["classes"].get(klass)
        if c:
            print("   %-6s %12.5f %9.5f %13.4f %8.3f %10.3f %8.5f %6d"
                  % (klass, c["pct_per_instruction_median"], c["sem"],
                     c["us_per_instruction_median"], c["r2_median"],
                     c["scaffold_pct_median"],
                     c["slope_three_point_median"], c["cells"]))
    print("   the same price resolved per width, pct/instr "
          "(`-` where every rung of that class spilled)")
    print("   %-6s %s" % ("class", "  ".join("%12s" % ("NA%d" % m)
                                             for m in widths)))
    for klass in ("ld", "alu", "shuf"):
        c = model["classes"].get(klass)
        if not c:
            continue
        cells_txt = []
        for m in widths:
            v = c["per_na"].get(m)
            cells_txt.append("%12s" % (
                "-" if v is None else "%.5f/%.3fus"
                % (v["pct_per_instruction_median"],
                   v["us_per_instruction_median"])))
        print("   %-6s %s" % (klass, "  ".join(cells_txt)))
    ilp = model.get("ilp_control")
    if ilp:
        print("   ILP control, %d injected ALU instructions either way: "
              "2 chains cost %.3f %%, 4 chains cost %.3f %%, difference "
              "%+.3f pp over %d cells"
              % (ilp["injected_instructions"], ilp["two_chain_cost_pct"],
                 ilp["four_chain_cost_pct"], ilp["four_minus_two_pct"],
                 ilp["cells"]))
        print("   halving the dependency depth at constant instruction count "
              "did not reduce the cost, so the ladder reads %s"
              % ilp["reads"])
    pred = model.get("screen_prediction")
    if pred:
        print("\n   the ladder applied to the screen arms, no free parameter")
        print("   compared per width; a width where the arm spills is shown "
              "as `spill` and never pooled away")
        hdr = "   %-16s %6s %6s" % ("arm", "d_ld", "d_shuf")
        for m in widths:
            hdr += " %17s" % ("NA%d pred/meas" % m)
        print(hdr + " %19s" % "wtd pred/meas (cov)")
        for arm in sorted(pred, key=lambda a: (
                pred[a]["weighted_measured_over_covered"]
                if pred[a]["weighted_measured_over_covered"] is not None
                else 1e9)):
            p = pred[arm]
            line = "   %-16s %6d %6d" % (
                arm, p["delta_device_loads"], p["delta_shuffles"])
            for m in widths:
                cell = p["per_na"].get(m, {"status": "no_cells"})
                if cell["status"] != "ok":
                    line += " %17s" % cell["status"]
                else:
                    line += " %8.2f/%8.2f" % (cell["predicted_pct_faster"],
                                              cell["measured_pct_faster"])
            wp = p["weighted_predicted_over_covered"]
            wm = p["weighted_measured_over_covered"]
            line += (" %19s" % "-" if wm is None else
                     " %6.2f/%6.2f (%.3f)" % (wp, wm,
                                              p["standing_weight_covered"]))
            print(line)
        print("   `cov` is the share of the standing round weight the two "
              "weighted columns speak for; they are not comparable between "
              "two arms with different cov")

    # --- the observational regression, for contrast with the ladder ----------
    obs = air_regression(cells, census_data, us)
    if obs:
        u = obs["univariate_summary"]
        mv = obs["multivariate_summary"]
        print("\n-- observational regression, microseconds on AIR counts, "
              "every arm x every cell")
        print("   %s" % obs["note"])
        print("   univariate on issue lanes: %.4f us/instruction "
              "(se %.4f), R2 median %.3f over %d cells"
              % (u["us_per_issue_lane_median"], u["se_median"],
                 u["r2_median"], u["cells"]))
        print("   multivariate, R2 median %.3f over %d cells"
              % (mv["r2_median"], mv["cells"]))
        print("   %-18s %16s %12s" % ("AIR category", "us/instruction", "se"))
        for k in AIR_REGRESSORS:
            print("   %-18s %16.4f %12.4f"
                  % (k, mv[k]["us_per_instruction_median"], mv[k]["se_median"]))
        head = obs["per_cell"].get("%s|NA4" % shape)
        if head:
            print("   worst univariate residuals on %s NA4, the arms that "
                  "cost more than one slot each" % shape)
            for r in head["worst_residuals"]:
                print("      %-16s %+9.1f us" % (r["arm"], r["residual_us"]))

    payload = {
        "harness": "local",
        "block_dispersion": disp,
        "rung2_finding53": rung2,
        "rung2_exact_verdict": rung2_verdict,
        "air_regression": obs,
        "cost_model": model,
        "device": rate["device"], "architecture": rate["architecture"],
        "shape": shape, "widths": widths, "warmup_blocks": WARMUP_BLOCKS,
        "pairs": rate["pairs"], "ramp_ms": rate.get("ramp_ms"),
        "sign_convention": "positive percent means FASTER than a_base",
        "standing_weights": sw.STANDING_WEIGHTS,
        "fidelity": fid, "forward_reverse_gap_pct": gaps,
        "per_arm_na_pct": per_arm_na,
        "per_arm_na_stats": stats,
        "absolute_us": {"%s|NA%d" % k: v for k, v in us.items()},
        "finding44": {"per_na": f44, "round_weighted_gap_pct": f44_weighted},
        "weighted": rows, "per_shape": per_shape, "bias_axis": bias,
        "primary_metric": {
            "name": "e118_best_bit_exact_arm_round_weighted_pct_faster_vs_"
                    "a_base",
            "value": best_value, "arm": best, "kill_rule_pct": KILL_RULE_PCT,
            "cleared": bool(best_value >= KILL_RULE_PCT)},
        "discriminator": {"s_bcast": s_b, "s_bcast_all": s_ba,
                          "p_split_meta": p_sm, "n_nosums": n_ns,
                          "verdict": verdict},
    }
    if census_data is not None:
        payload["census"] = census_data
        print_census(payload["census"])
        se = spill_exactness(rate, payload["census"])
        payload["spill_exactness"] = se
        if se is not None:
            print("\n-- spill bytes against exactness on %s" % se["arch"])
            print("   an arm is `wrong` when it differs from a_base on any "
                  "shape at that width")
            print("   %-16s %s" % ("arm", "  ".join(
                "%14s" % ("NA%d" % m) for m in widths)))
            for arm in sorted(se["arms"]):
                cells = []
                for m in widths:
                    c = se["arms"][arm].get(str(m))
                    cells.append("%14s" % ("?" if c is None else "%dB %s" % (
                        c["spill_bytes"], "exact" if c["exact"] else "WRONG")))
                print("   %-16s %s" % (arm, "  ".join(cells)))
            print("   largest spill that stayed exact: %s B"
                  % se["max_spill_while_exact"])
            print("   smallest spill that went wrong:  %s B"
                  % se["min_spill_while_wrong"])
            print("   spill separates exact from wrong: %s" % se["separates"])
            zb = se["arms"].get("z_ballast", {})
            zwrong = sorted(int(m) for m, c in zb.items() if not c["exact"])
            print("   z_ballast control: changes no arithmetic that reaches y, "
                  "wrong at NA %s"
                  % (zwrong if zwrong else "nowhere"))
            print("   -> the NA=5 divergence is caused by SPILLING, not by any "
                  "arm's mechanism" if zwrong else
                  "   -> spilling alone does not break exactness here")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % out_path)
    return 0


def print_census(census: dict) -> None:
    widths = census["widths"]
    print("\n-- AIR device loads per entry point (and simd_shuffle calls)")
    print("   %-16s %s" % ("arm", "  ".join("%9s" % ("NA%d" % na)
                                            for na in widths)))
    for arm, row in census["arms"].items():
        cells = []
        for na in widths:
            c = row["air"].get(str(na), {})
            sh = c.get("shuffles", 0)
            cells.append("%9s" % ("%s+%dsh" % (c.get("device_loads", "?"), sh)
                                  if sh else str(c.get("device_loads", "?"))))
        print("   %-16s %s" % (arm, "  ".join(cells)))
    for arch in (census["local_arch"], census["ranked_arch"]):
        print("\n-- %s registers / spill bytes / machine text bytes" % arch)
        for arm, row in census["arms"].items():
            cells = []
            for na in widths:
                v = row.get(arch, {}).get(str(na))
                if v is None:
                    cells.append("NA%d=?" % na)
                    continue
                spill = v["spill_bytes"] or 0
                cells.append("NA%d=%s%s/%s" % (na, v["registers"],
                                               "s%d" % spill if spill else "",
                                               v["text_bytes"]))
            print("   %-16s %s" % (arm, "  ".join(cells)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--slice", type=pathlib.Path,
                    default=pathlib.Path(
                        "research/e118-artifacts/e114_receipt_slice.json"))
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--shape", default=HEADLINE_SHAPE)
    ap.add_argument("--extract-receipt")
    ap.add_argument("--receipt", default="b8b8b860")
    args = ap.parse_args()

    if args.extract_receipt:
        return extract_receipt(args.extract_receipt, args.receipt, args.slice)
    if args.rate is None:
        ap.error("--rate is required unless --extract-receipt is given")
    slice_path = args.slice if args.slice and args.slice.exists() else None
    return report(args.rate, args.census, slice_path, args.out, args.shape)


if __name__ == "__main__":
    raise SystemExit(main())
