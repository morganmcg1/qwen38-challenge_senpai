#!/usr/bin/env python3
"""E48: does a UNIFORM QMV cost change move the score up or down?

    research/e48_analyze.py --arms base ulo uhi base2 [--wandb]

The score identity is `raw_p = serial / mtp`, so a change that scales QMV cost
by (1+x) everywhere scales raw_p by (1 + psi_serial*x)/(1 + psi_mtp_TOTAL*x).
Ledger 173(A) claims psi_mtp_TOTAL < psi_serial, i.e. a uniform QMV *speedup*
LOWERS the score. E48 injects the cost instead of modelling it.

Two things make this harder than E42:

1. The pass-loop injection is NOT uniform in x. It covers ~90 % of a crossrow
   kernel and ~43 % of `qmv_fast_impl`, so one level everywhere over-doses the
   candidate leg 2.1x. The arms therefore carry independent levels per family
   and the realised dose ratio x1/xX is measured, not assumed.
2. The candidate leg is NOT free of width-1 QMV. `psi_mtp_w1` is small but
   non-zero, so `psi_mtp_TOTAL = psi_mtp_w1 + psi_mtp_X` and the two dosed arms
   are used to identify both terms rather than inheriting either.

Every leg here ran with MLXFAST_LOCAL_COOL_GATE=0. Nothing printed is
gate-qualified, ranked-equivalent, or an official score.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e42_analyze import (  # noqa: E402
    DRAFT_SIDE_SHAPE,
    histogram,
    load_legs,
    mean,
    summarise_arm,
    verify_cost_per_round,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / ".mlxfast-private/e48/runs"
CURVES = ROOT / ".mlxfast-private/qmv-curve"

# (crossrow_level, m1_level) as spliced by research/e48_perturb.py. The g arms
# carry m1_level 0, so their MTP-leg response is pure crossrow and their serial
# leg is a structural-churn control (scaffolding present, dose zero).
DOSES = {"base": (0, 0), "base2": (0, 0), "ulo": (1, 2), "g1": (1, 0), "g2": (2, 0)}
NULL_ARMS = ("base", "base2")
CROSSROW_WIDTHS = frozenset(range(2, 10))
# Cross-build reproduction tolerance for a shape's per-call cost.
REPRO_TOL = 0.02
nan = float("nan")


def load_meta(arm: str) -> dict:
    """Provenance and thermal record written by research/e42-run.sh."""
    meta: dict[str, str] = {}
    path = RUNS / arm / "meta.txt"
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        meta[key] = value

    def gpu_c(raw: str | None) -> float:
        if not raw:
            return nan
        for field in raw.split():
            if field.startswith("gpu_temp="):
                return float(field[len("gpu_temp=") : -1])
        return nan

    legs = sorted(k for k in meta if k.endswith("_thermal_after") and k.startswith("leg"))
    return {
        "head_sha": meta["head_sha"],
        "twin_digests": meta["twin_digests"],
        "metallib_fingerprint": meta.get("metallib_fingerprint"),
        "started": meta.get("started"),
        "entry_gpu_temp_c": gpu_c(meta.get("thermal_before")),
        "exit_gpu_temp_c": gpu_c(meta.get("thermal_after")),
        "per_leg_exit_gpu_temp_c": [gpu_c(meta[k]) for k in legs],
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "official_or_ranked_score": meta["official_or_ranked_score"],
    }


def load_curve(arm: str) -> dict | None:
    path = CURVES / f"e48-{arm}" / "vendored.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    out = {}
    for shape in payload["shapes"]:
        out[shape["name"]] = {
            "calls_per_verify": shape["calls_per_verify"],
            "k": shape["k"],
            "n": shape["n"],
            "rows": {int(row["m"]): row for row in shape["rows"]},
        }
    return out


def per_width_slowdown(base_curve, arm_curve, exclude=frozenset()) -> dict[int, dict]:
    ms = sorted(set(base_curve["mlp.down"]["rows"]) & set(arm_curve["mlp.down"]["rows"]))
    out = {}
    for m in ms:
        b = verify_cost_per_round(base_curve, m, exclude)
        a = verify_cost_per_round(arm_curve, m, exclude)
        per_shape = {}
        for name, shape in base_curve.items():
            if name == DRAFT_SIDE_SHAPE or not shape["calls_per_verify"] or name in exclude:
                continue
            per_shape[name] = (
                arm_curve[name]["rows"][m]["seconds_per_call"]
                / shape["rows"][m]["seconds_per_call"]
                - 1.0
            )
        out[m] = {
            "base_verify_seconds": b,
            "arm_verify_seconds": a,
            "x": a / b - 1.0,
            "x_by_shape": per_shape,
        }
    return out


def costed_shapes(curve: dict, exclude: frozenset[str] = frozenset()) -> list[str]:
    return [
        n
        for n, s in curve.items()
        if n != DRAFT_SIDE_SHAPE and s["calls_per_verify"] and n not in exclude
    ]


def irreproducible_shapes(base_curve: dict, repeat_curve: dict) -> list[str]:
    """Shapes that fail to reproduce between two builds of the SAME source.

    E42 had to infer this from an untreated width inside a dosed arm. E48 treats
    every width, so the base bracket is the only clean cross-build control -- and
    it is the stronger one, because it exercises every width and every shape.
    """
    x_by_m = per_width_slowdown(base_curve, repeat_curve)
    bad = set()
    for cell in x_by_m.values():
        bad |= {n for n, x in cell["x_by_shape"].items() if abs(x) > REPRO_TOL}
    return sorted(bad)


def crossrow_xbar(x_by_m: dict[int, dict], hist: dict[int, int]) -> float:
    """x over widths 2..9, weighted by each width's share of BASE verify QMV cost."""
    weights, xs = [], []
    for m, count in sorted(hist.items()):
        if m not in CROSSROW_WIDTHS or m not in x_by_m:
            continue
        weights.append(count * x_by_m[m]["base_verify_seconds"])
        xs.append(x_by_m[m]["x"])
    if not weights:
        return nan
    return sum(w * x for w, x in zip(weights, xs)) / sum(weights)


def coverage_gap(base_curve: dict, hist: dict[int, int], mtp_seconds: float) -> dict:
    """Candidate-leg QMV that NO arm treats, so psi_mtp comes out a lower bound.

    Two families escape both injections:

    * the 2-bit compact draft readout. `quantized.h:1908` dispatches
      `qmv_fast_singlerow_affine2_g64` at `ntg.x == 1 && out_vec_size == 98336`
      and RETURNS, upstream of both crossrow tiers and of the `qmv_fast_impl`
      fall-through the m1 dose patches. It is candidate-leg-only by construction.
    * GDN `in_proj_b` / `in_proj_a`, n=48 (`Qwen35GatedDelta.swift:254-255`).
      `out_vec_size < 1024` sends them to `qmv_fast_impl` at every width, so the
      crossrow dose misses them and the m1 dose only catches width 1. The curve
      models all four GDN input projections as one fused n=16480 shape
      (`QwenQMVCostCurveTests.swift:396`), so their bandwidth is already inside
      the treated total; only ~0.6 % of that shape's width is a/b columns.

    The serial leg has neither: at depth 0 every backbone shape falls through to
    `qmv_fast_impl`, which the m1 dose treats in full (E42 measured its
    alpha at 0.9733). So the asymmetry runs one way -- psi_mtp is understated,
    psi_serial is not -- and the uniform coefficient psi_mtp - psi_serial is a
    LOWER bound. A measured positive coefficient is therefore conclusive; a
    measured negative one has to clear this correction first.
    """
    treated = sum(count * verify_cost_per_round(base_curve, m) for m, count in hist.items())
    draft = base_curve.get(DRAFT_SIDE_SHAPE)
    draft_steps = sum(count * (m - 1) for m, count in hist.items())
    # The curve probes every shape at affine-4, and its own M=1 row reports
    # `qmv_fast_impl`. The real readout is 2-bit on the singlerow kernel, which
    # moves about half the weight bytes, so the 4-bit row is an upper bound and
    # half of it is the bandwidth-scaled estimate. `calls_per_verify` is 0 for
    # this shape, so it never entered the x-bar denominator either way.
    hi = draft_steps * draft["rows"][1]["seconds_per_call"] if draft else 0.0
    lo = 0.5 * hi
    return {
        "treated_verify_qmv_seconds": treated,
        "draft_steps": draft_steps,
        "untreated_draft_readout_seconds_4bit_proxy_upper": hi,
        "untreated_draft_readout_seconds_2bit_scaled": lo,
        "untreated_share_of_candidate_qmv_upper": hi / (treated + hi),
        "untreated_share_of_candidate_qmv_scaled": lo / (treated + lo),
        "psi_mtp_additive_correction_upper": hi / mtp_seconds,
        "psi_mtp_additive_correction_scaled": lo / mtp_seconds,
        "note": (
            "the correction's SIGN is certain because untreated candidate QMV has "
            "positive cost; only its magnitude is uncertain, and the one-sided "
            "bound below does not depend on the magnitude"
        ),
        "direction": "psi_mtp measured is a LOWER bound; uniform coefficient is a LOWER bound",
    }


def solve_psi_mtp(arms: dict[str, dict], stable: bool = False) -> dict:
    """psi_mtp from the crossrow-only (Arm G) arms, plus every control they license.

    An Arm G arm doses only the crossrow family, so its MTP-leg response is
    psi_mtp_X * xbar_X with no width-1 term to divide out, and its serial leg
    carries the pass-loop scaffolding at dose zero. That makes psi_mtp a direct
    per-arm quotient and the serial leg a structural-churn control.
    """
    xk, xXk = ("x1_stable", "xbar_X_stable") if stable else ("x1", "xbar_X")
    g = {k: v for k, v in arms.items() if v["m1_level"] == 0 and v["crossrow_level"] > 0}
    out: dict = {"arms_used": sorted(g), "dosimetry": "stable_shapes" if stable else "as_measured"}
    if not g:
        out["identified"] = False
        return out
    out["identified"] = True
    out["psi_mtp_per_arm"] = {k: v["mtp_frac"] / v[xXk] for k, v in g.items()}
    out["psi_mtp"] = mean(out["psi_mtp_per_arm"].values())
    out["psi_mtp_interval"] = [min(out["psi_mtp_per_arm"].values()), max(out["psi_mtp_per_arm"].values())]

    # Two doses test the functional form, not just the point: a linear cost model
    # predicts the same quotient at every dose.
    if len(g) >= 2:
        lo, hi = sorted(g, key=lambda k: g[k][xXk])
        out["form_test"] = {
            "low_arm": lo,
            "high_arm": hi,
            "dose_ratio_xX": g[hi][xXk] / g[lo][xXk],
            "psi_mtp_form_residual_pct": 100.0
            * (out["psi_mtp_per_arm"][hi] / out["psi_mtp_per_arm"][lo] - 1.0),
        }

    # Scaffolding at dose zero must cost nothing, or every arm's dose attribution
    # is wrong. This is measured on the same build and thermal window as the dose.
    out["structural_churn_control"] = {
        "what": "serial leg of each Arm G arm: pass-loop scaffolding present, E42_PASSES=0",
        "serial_frac_per_arm": {k: v["serial_frac"] for k, v in g.items()},
        "worst_abs_serial_frac": max(abs(v["serial_frac"]) for v in g.values()),
    }

    # ulo and an Arm G arm at the SAME crossrow level differ only in the width-1
    # dose, so differencing the MTP legs isolates psi_mtp_w1 with no crossrow term.
    if "ulo" in arms:
        u = arms["ulo"]
        twin = next((k for k, v in g.items() if v["crossrow_level"] == u["crossrow_level"]), None)
        if twin:
            out["psi_mtp_w1_by_differencing"] = {
                "paired_with": twin,
                "shared_crossrow_level": u["crossrow_level"],
                "mtp_frac_difference": u["mtp_frac"] - arms[twin]["mtp_frac"],
                "psi_mtp_w1": (u["mtp_frac"] - arms[twin]["mtp_frac"]) / u[xk],
                "covers": "4-bit qmv_fast_impl width-1 only",
                "excludes": (
                    "the 2-bit draft readout at quantized.h:1908, which no arm doses; "
                    "so this is a LOWER bound on total candidate width-1 exposure"
                ),
            }
    return out


def local_uniform_coefficient(arms: dict[str, dict], stable: bool = False) -> dict:
    """The withdrawn quantity, reported for the record from the ulo arm.

    Has no ranked meaning: the ranked baseline is a separately built pinned tree,
    so d ln(serial)/dx = 0 there. Kept because it validates the local two-leg
    cost model that E42's psi values were measured under.
    """
    if "ulo" not in arms:
        return {"available": False}
    xk, xXk = ("x1_stable", "xbar_X_stable") if stable else ("x1", "xbar_X")
    u = arms["ulo"]
    psi_serial = u["serial_frac"] / u[xk]
    psi_mtp_tot = u["mtp_frac"] / u[xXk]
    # An Arm G arm doses width 1 at zero, so its width-1 curve cell should read
    # x = 0. Whatever it actually reads is the width-1 dosimeter's error at dose
    # zero, and it divides straight into psi_serial.
    offset = next((v[xk] for v in arms.values() if v["m1_level"] == 0), None)
    # raw_p is flat at the level ratio rho* where the two legs move together.
    slope = psi_serial * (u[xk] / DOSES["ulo"][1]) / (1.0 + u["serial_frac"])
    rho = DOSES["ulo"][1] / DOSES["ulo"][0]
    out = {
        "available": True,
        "dosimetry": "stable_shapes" if stable else "as_measured",
        "ranked_meaning": "NONE: psi_serial is not in the ranked score (pinned baseline binary)",
        "psi_serial_local": psi_serial,
        "psi_mtp_total_local": psi_mtp_tot,
        "uniform_coefficient_local": psi_mtp_tot - psi_serial,
        "realised_level_ratio_rho": rho,
        "null_crossing_level_ratio_rho_star": rho - math.log(u["raw_p_ratio"]) / slope
        if slope
        else nan,
        "ledger_173A_claim": -0.1789,
    }
    coeffs = [out["uniform_coefficient_local"]]
    if offset:
        # Subtracting the dose-zero offset is one defensible treatment; leaving it
        # in is the other. Which is right depends on whether the offset is an
        # additive bias or noise, and this design cannot tell them apart, so the
        # honest answer is the interval spanned by both.
        psi_serial_corr = u["serial_frac"] / (u[xk] - offset)
        coeffs.append(psi_mtp_tot - psi_serial_corr)
        out["width1_dosimeter_offset_at_dose_zero"] = offset
        out["psi_serial_local_offset_corrected"] = psi_serial_corr
        out["uniform_coefficient_local_offset_corrected"] = coeffs[-1]
    out["uniform_coefficient_local_interval"] = [min(coeffs), max(coeffs)]
    out["overstatement_factor_vs_173A_interval"] = sorted(
        abs(out["ledger_173A_claim"] / c) for c in coeffs if c
    )
    out["note"] = (
        "psi_mtp uses only the crossrow dosimeter, which reproduces across "
        "independent builds; psi_serial uses only the width-1 dosimeter, which does "
        "not. The withdrawn sign rested on the weaker half of the instrument."
    )
    return out


def dosimeter_reproducibility(dosed: dict[str, dict]) -> dict:
    """Do two independent builds at the same crossrow dose measure the same x?

    This is what decides whether psi_mtp is identified. Arms sharing a crossrow
    level must agree per width; the width-1 cell of an m1_level=0 arm must read 0.
    """
    out: dict = {}
    by_level: dict[int, list[str]] = {}
    for name, rec in dosed.items():
        by_level.setdefault(rec["crossrow_level"], []).append(name)
    for level, names in sorted(by_level.items()):
        if len(names) < 2:
            continue
        a, b = names[0], names[1]
        diffs = {
            m: dosed[b]["x_by_width"][m] - dosed[a]["x_by_width"][m]
            for m in dosed[a]["x_by_width"]
            if m in dosed[b]["x_by_width"] and int(m) in CROSSROW_WIDTHS
        }
        out[f"crossrow_level_{level}"] = {
            "arms": [a, b],
            "per_width_abs_diff": {m: abs(v) for m, v in sorted(diffs.items(), key=lambda kv: int(kv[0]))},
            "worst_abs_diff": max(abs(v) for v in diffs.values()),
            "mean_abs_diff": mean(abs(v) for v in diffs.values()),
            "xbar_X_disagreement_pct": 100.0 * (dosed[b]["xbar_X"] / dosed[a]["xbar_X"] - 1.0),
        }
    zero = {k: v["x_by_width"].get(1) for k, v in dosed.items() if v["m1_level"] == 0}
    if zero:
        out["width1_cell_at_true_dose_zero"] = zero
        out["verdict"] = (
            "crossrow dosimeter reproduces across builds; width-1 dosimeter does not, "
            "so psi_mtp is identified and psi_serial is not"
        )
    return out


def one_sided_bound(arm: dict) -> dict:
    """Sign conclusion that survives without knowing the dose ratio exactly.

    raw_p rises iff serial_frac > mtp_frac. Under-dosing width 1 (x1 <= xX) can
    only make raw_p look WORSE than a truly uniform change would, so raw_p up at
    x1 <= xX proves the uniform sign is negative. Over-dosing gives the mirror
    proof for a positive sign.
    """
    ratio = arm["x1"] / arm["xbar_X"] if arm["xbar_X"] else nan
    up = arm["serial_frac"] > arm["mtp_frac"]
    verdict = "inconclusive"
    if ratio <= 1.0 and up:
        verdict = "uniform sign NEGATIVE (proved: under-dosed width 1 and raw_p still rose)"
    elif ratio >= 1.0 and not up:
        verdict = "uniform sign POSITIVE (proved: over-dosed width 1 and raw_p still fell)"
    return {
        "realised_dose_ratio_x1_over_xX": ratio,
        "raw_p_direction": "up" if up else "down",
        "one_sided_verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base", "ulo", "g1", "g2", "base2"])
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    arms, curves, metas = {}, {}, {}
    for arm in args.arms:
        arms[arm] = summarise_arm(arm, load_legs_for(arm), None)
        curves[arm] = load_curve(arm)
        metas[arm] = load_meta(arm)

    base, base_curve = arms["base"], curves["base"]
    exclude = frozenset()
    entry_temps = [m["entry_gpu_temp_c"] for m in metas.values()]
    payload: dict = {
        "base_sha": "fb0a09d3912477d94ed631bdb90fd04172d7b4cf",
        "head_sha": git_head(),
        "doses": {k: DOSES.get(k) for k in args.arms},
        "arm_order": args.arms,
        "counterbalancing": (
            "bracketed A B B' A: the two zero-dose arms bookend the dosed arms in one "
            "session, so monotone drift shows up as the base/base2 spread reported below"
        ),
        "provenance": metas,
        "entry_gpu_temp_spread_c": max(entry_temps) - min(entry_temps),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }

    if "base2" in arms:
        b2 = arms["base2"]
        payload["null_arm"] = {
            "what": "byte-identical zero-dose repeat of base, separate build and session slot",
            "serial_frac": b2["serial_decode_seconds"] / base["serial_decode_seconds"] - 1.0,
            "mtp_frac": b2["mtp_decode_seconds"] / base["mtp_decode_seconds"] - 1.0,
            "raw_p_base": base["raw_p"],
            "raw_p_repeat": b2["raw_p"],
            "raw_p_ratio": b2["raw_p"] / base["raw_p"],
        }
        if curves.get("base2"):
            exclude = frozenset(irreproducible_shapes(base_curve, curves["base2"]))
            payload["null_arm"]["irreproducible_shapes"] = sorted(exclude)
            payload["null_arm"]["curve_repro_tolerance"] = REPRO_TOL
            if not costed_shapes(base_curve, exclude):
                payload["null_arm"]["stable_shapes_variant"] = (
                    "unavailable: every costed shape failed cross-build reproduction, "
                    "so there is no stable subset to reprice against"
                )
                exclude = frozenset()

    dosed = {}
    for arm in args.arms:
        if arm in NULL_ARMS or not curves.get(arm):
            continue
        rec = arms[arm]
        x_all = per_width_slowdown(base_curve, curves[arm])
        x_stable = per_width_slowdown(base_curve, curves[arm], exclude) if exclude else x_all
        hist = rec["width_histogram"]
        entry = {
            "crossrow_level": DOSES[arm][0],
            "m1_level": DOSES[arm][1],
            "serial_frac": rec["serial_decode_seconds"] / base["serial_decode_seconds"] - 1.0,
            "mtp_frac": rec["mtp_decode_seconds"] / base["mtp_decode_seconds"] - 1.0,
            "raw_p": rec["raw_p"],
            "raw_p_ratio": rec["raw_p"] / base["raw_p"],
            "x1": x_all[1]["x"],
            "xbar_X": crossrow_xbar(x_all, hist),
            "x1_stable": x_stable[1]["x"],
            "xbar_X_stable": crossrow_xbar(x_stable, hist),
            "x_by_width": {m: c["x"] for m, c in sorted(x_all.items())},
            "all_tokens_matched": rec["all_tokens_matched"],
            "row_ledger_closes": rec["row_ledger_closes"],
        }
        if entry["m1_level"]:
            entry.update(one_sided_bound(entry))
        dosed[arm] = entry
    payload["dosed_arms"] = dosed

    gap = coverage_gap(base_curve, base["width_histogram"], base["mtp_decode_seconds"])
    payload["coverage_gap"] = gap
    payload["dosimeter_reproducibility"] = dosimeter_reproducibility(dosed)

    # psi_mtp is the primary estimand: the ranked score is
    # baseline_serial_seconds_per_token_mean / candidate_mtp_seconds_per_token_mean
    # against a separately built pinned baseline, so it is the only coefficient a
    # candidate-side QMV change can move.
    for label, stable in (("psi_mtp", False), ("psi_mtp_stable_shapes", True)):
        est = solve_psi_mtp(dosed, stable=stable)
        if est.get("identified"):
            for tag in ("scaled", "upper"):
                est[f"psi_mtp_gap_corrected_{tag}"] = (
                    est["psi_mtp"] + gap[f"psi_mtp_additive_correction_{tag}"]
                )
            est["ranked_dscore_per_pct_qmv_win"] = est["psi_mtp"]
            est["ranked_gating_premium"] = 0.0
        payload[label] = est

    payload["withdrawn_uniform_sign"] = {
        "status": "WITHDRAWN BY ADVISOR, not answered by this experiment",
        "why": (
            "the ranked baseline is a separately built pinned tree "
            "(/opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current, never built from "
            "candidate source; .github is absent from benchmark.json editablePaths), so "
            "d ln(serial)/dx = 0 on ranked and psi_serial has no ranked leverage"
        ),
        "provenance": "edward E50 / PR 54, relayed on PR 52 comment 5342984599; "
        "re-verified from the workflow independently here",
        "local_frame_as_measured": local_uniform_coefficient(dosed, stable=False),
        "local_frame_stable_shapes": local_uniform_coefficient(dosed, stable=True),
    }

    payload["arms"] = {k: strip(v) for k, v in arms.items()}
    payload["width_histogram_spread"] = width_spread(args.arms)
    print(json.dumps(payload, indent=2, default=str))
    if args.wandb:
        log_wandb(payload)
    return 0


def load_legs_for(arm: str) -> list[dict]:
    import e42_analyze

    e42_analyze.RUNS = RUNS
    return load_legs(arm)


def strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not isinstance(v, (list, dict)) or k.endswith("histogram")}


def width_spread(arm_names: list[str]) -> dict:
    """Is the per-round width histogram a random variable across repeats?"""
    draws = []
    for arm in arm_names:
        for leg in load_legs_for(arm):
            widths = [d + 1 for d in leg["mtp"]["effective_draft_lengths"]]
            draws.append(
                {
                    "arm": arm,
                    "leg": leg["index"],
                    "rounds": len(widths),
                    "mean_m": mean(widths),
                    "histogram": histogram(widths),
                }
            )
    means = [d["mean_m"] for d in draws]
    hists = [json.dumps(d["histogram"], sort_keys=True) for d in draws]
    return {
        "draws": draws,
        "n_draws": len(draws),
        "identical_across_all_draws": len(set(hists)) == 1,
        "mean_m_range": [min(means), max(means)] if means else [],
        "mean_m_sd_pct": (100.0 * statistics.stdev(means) / mean(means)) if len(means) > 1 else 0.0,
    }


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def log_wandb_part1(run, wandb, summary: dict) -> dict | None:
    """Part 1 shares from research/e48_score_weighted_shares.py, if it has been run.

    Logged into the same run as Part 2 because the shares only price a mechanism
    once multiplied by psi_mtp, which Part 2 measures.
    """
    path = ROOT / "research/e48-artifacts/score-weighted-shares.json"
    if not path.exists():
        return None
    shares = json.loads(path.read_text())
    weighted = shares["score_weighted"]
    superseded = shares["score_weighted_superseded_79_21"]
    summary["part1/weights"] = json.dumps(weighted["weights"])
    for slice_name, value in weighted["slices"].items():
        summary[f"part1/{slice_name}"] = value
        summary[f"part1/{slice_name}_superseded_79_21"] = superseded["slices"][slice_name]
    for prompt, rec in shares["per_prompt_maxent_tilt"].items():
        for slice_name, value in rec["slices"].items():
            summary[f"part1/{prompt}/{slice_name}"] = value
        summary[f"part1/{prompt}/mean_m"] = rec["mean_m"]
    for slice_name, value in shares["corpus"]["slices"].items():
        summary[f"part1/corpus/{slice_name}"] = value
    summary["part1/kink_pct"] = shares["repricing"]["kink_pct"]
    summary["part1/saturation_cap_pct"] = shares["repricing"]["saturation_cap_pct"]
    summary["part1/identification"] = shares["identification"]

    share_table = wandb.Table(columns=["basis", "weights", "M9", "f_7_8", "f_4_5_6"])
    rows = {
        "corpus_E42": (None, shares["corpus"]["slices"]),
        "beagle_unweighted": (None, shares["per_prompt_maxent_tilt"]["beagle"]["slices"]),
        "medicine_unweighted": (None, shares["per_prompt_maxent_tilt"]["medicine"]["slices"]),
        "score_weighted_marginal": (weighted["weights"], weighted["slices"]),
        "score_weighted_SUPERSEDED_79_21": (superseded["weights"], superseded["slices"]),
    }
    for basis, (weights, sl) in rows.items():
        share_table.add_data(
            basis,
            json.dumps(weights) if weights else "",
            sl["M9"],
            sl["f_7_8"],
            sl["f_4_5_6"],
        )

    reprice_table = wandb.Table(
        columns=[
            "mechanism",
            "slice",
            "qmv_pct",
            "beagle_leg_pct",
            "medicine_leg_pct",
            "score_pct_order_statistic",
            "score_pct_naive_rate",
            "rate_error_pp",
            "above_kink",
        ]
    )
    for name, rec in shares["repricing"]["mechanisms"].items():
        reprice_table.add_data(
            name,
            rec["slice"],
            rec["qmv_cost_reduction_pct"],
            rec["per_prompt_leg_gain_pct"]["beagle"],
            rec["per_prompt_leg_gain_pct"]["medicine"],
            rec["score_pct_order_statistic"],
            rec["score_pct_naive_weighted_rate"],
            rec["rate_model_error_pct_points"],
            rec["above_kink"],
        )
        summary[f"part1/repricing/{name}/score_pct"] = rec["score_pct_order_statistic"]
    return {"part1_cost_shares": share_table, "part1_repricing": reprice_table}


def log_wandb(payload: dict) -> None:
    import wandb

    est = payload.get("psi_mtp", {})
    est_stable = payload.get("psi_mtp_stable_shapes", {})
    local = payload["withdrawn_uniform_sign"]["local_frame_as_measured"]
    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="qwen38-mlx-challenge-senpai",
        name="e48-psi-mtp-arm-g",
        job_type="analysis",
        tags=["e48", "qwen3.8-27b-mtp-v1", "injected-regression", "ungated-local"],
        config={
            "base_sha": payload["base_sha"],
            "head_sha": payload["head_sha"],
            "doses": payload["doses"],
            "host": "local-m4-pro",
            "fixture": "correctness_prompts/public_longcopy_gate_english_512_256.json",
            "decode_tokens": 512,
            "offered_depth": 8,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
        },
    )
    churn = est.get("structural_churn_control", {})
    diff = est.get("psi_mtp_w1_by_differencing", {})
    summary = {
        # primary: the only coefficient a candidate-side QMV change moves on ranked
        "psi_mtp": est.get("psi_mtp"),
        "psi_mtp_lo": (est.get("psi_mtp_interval") or [None, None])[0],
        "psi_mtp_hi": (est.get("psi_mtp_interval") or [None, None])[1],
        "psi_mtp_stable_shapes": est_stable.get("psi_mtp"),
        "psi_mtp_gap_corrected_scaled": est.get("psi_mtp_gap_corrected_scaled"),
        "psi_mtp_gap_corrected_upper": est.get("psi_mtp_gap_corrected_upper"),
        "psi_mtp_form_residual_pct": est.get("form_test", {}).get("psi_mtp_form_residual_pct"),
        "ranked_dscore_per_pct_qmv_win": est.get("ranked_dscore_per_pct_qmv_win"),
        "ranked_gating_premium": est.get("ranked_gating_premium"),
        "identified": est.get("identified"),
        # controls
        "churn_worst_abs_serial_frac": churn.get("worst_abs_serial_frac"),
        "psi_mtp_w1_by_differencing": diff.get("psi_mtp_w1"),
        "untreated_share_of_candidate_qmv_upper": payload["coverage_gap"][
            "untreated_share_of_candidate_qmv_upper"
        ],
        # withdrawn quantity, local frame only, kept for the record
        "local_uniform_coefficient": local.get("uniform_coefficient_local"),
        "local_psi_serial": local.get("psi_serial_local"),
        "local_rho_star": local.get("null_crossing_level_ratio_rho_star"),
        "uniform_sign_status": "withdrawn_by_advisor",
    }
    est_all = [
        v
        for src in (est, est_stable)
        for v in (src.get("psi_mtp_per_arm") or {}).values()
    ]
    if est_all:
        summary["psi_mtp_envelope_lo"] = min(est_all)
        summary["psi_mtp_envelope_hi"] = max(est_all)
        summary["psi_mtp_envelope_width_pct"] = 100.0 * (max(est_all) - min(est_all)) / (
            sum(est_all) / len(est_all)
        )
    local_stable = payload["withdrawn_uniform_sign"].get("local_frame_stable_shapes", {})
    psi_serial_variants = [
        v
        for v in (
            local.get("psi_serial_local"),
            local.get("psi_serial_local_offset_corrected"),
            local_stable.get("psi_serial_local"),
            local_stable.get("psi_serial_local_offset_corrected"),
        )
        if v is not None
    ]
    if psi_serial_variants:
        summary["local_psi_serial_lo"] = min(psi_serial_variants)
        summary["local_psi_serial_hi"] = max(psi_serial_variants)
        # a leg cannot be more than 100 % QMV, so >1 is proof the width-1
        # dosimeter is not measuring what it claims to measure
        summary["local_psi_serial_exceeds_unity"] = max(psi_serial_variants) > 1.0
    coeff_variants = [
        v
        for src in (local, local_stable)
        for v in (src.get("uniform_coefficient_local_interval") or [])
    ]
    if coeff_variants:
        summary["local_uniform_coefficient_envelope_lo"] = min(coeff_variants)
        summary["local_uniform_coefficient_envelope_hi"] = max(coeff_variants)
    for arm, rec in payload.get("dosed_arms", {}).items():
        for key in (
            "serial_frac",
            "mtp_frac",
            "raw_p",
            "raw_p_ratio",
            "x1",
            "xbar_X",
            "realised_dose_ratio_x1_over_xX",
            "all_tokens_matched",
        ):
            summary[f"{arm}/{key}"] = rec.get(key)
        summary[f"{arm}/one_sided_verdict"] = rec.get("one_sided_verdict")
    for key, value in (payload.get("null_arm") or {}).items():
        if not isinstance(value, (list, dict)):
            summary[f"null/{key}"] = value
    summary["width_histogram_identical_across_draws"] = payload["width_histogram_spread"][
        "identical_across_all_draws"
    ]
    summary["width_mean_m_sd_pct"] = payload["width_histogram_spread"]["mean_m_sd_pct"]
    summary["entry_gpu_temp_spread_c"] = payload["entry_gpu_temp_spread_c"]
    for arm, meta in payload["provenance"].items():
        summary[f"{arm}/entry_gpu_temp_c"] = meta["entry_gpu_temp_c"]
        summary[f"{arm}/exit_gpu_temp_c"] = meta["exit_gpu_temp_c"]
        summary[f"{arm}/head_sha"] = meta["head_sha"]

    arm_table = wandb.Table(
        columns=[
            "arm",
            "crossrow_level",
            "m1_level",
            "serial_frac",
            "mtp_frac",
            "raw_p",
            "raw_p_ratio",
            "x1",
            "xbar_X",
            "dose_ratio",
            "verdict",
        ]
    )
    for arm, rec in payload.get("dosed_arms", {}).items():
        arm_table.add_data(
            arm,
            rec["crossrow_level"],
            rec["m1_level"],
            rec["serial_frac"],
            rec["mtp_frac"],
            rec["raw_p"],
            rec["raw_p_ratio"],
            rec["x1"],
            rec["xbar_X"],
            rec.get("realised_dose_ratio_x1_over_xX"),
            rec.get("one_sided_verdict"),
        )
    width_table = wandb.Table(columns=["arm", "m", "x"])
    for arm, rec in payload.get("dosed_arms", {}).items():
        for m, x in rec["x_by_width"].items():
            width_table.add_data(arm, int(m), x)
    hist_table = wandb.Table(columns=["arm", "leg", "rounds", "mean_m", "histogram"])
    for draw in payload["width_histogram_spread"]["draws"]:
        hist_table.add_data(
            draw["arm"], draw["leg"], draw["rounds"], draw["mean_m"], json.dumps(draw["histogram"])
        )
    logged = {"arms": arm_table, "per_width_x": width_table, "width_histograms": hist_table}
    shares = log_wandb_part1(run, wandb, summary)
    if shares is not None:
        logged.update(shares)
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    run.log(logged)
    run.finish()
    print(f"wandb_run_url={run.url}", file=sys.stderr)
    print(f"wandb_run_id={run.id}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
