#!/usr/bin/env python3
"""Log the E38 row-blocks-in-idle-x-blocks result to W&B.

E38 unbundles E33's measured 1.0150 into an activation-tile effect and a
grid-thinning effect by running three matched cost-curve arms. The durable
record is the pre-registration, the three ladders, the registered relations
R1/R2/R3, the untreated-width controls and the AIR register table.

  python3 research/e38_wandb_log.py research/e38-artifacts/e38-metrics.json
  python3 research/e38_wandb_log.py research/e38-artifacts/e38-metrics.json \
      --resume <run_id>
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e38_m6_step as S  # noqa: E402
import e38_prereg as P  # noqa: E402
import e38_value as V  # noqa: E402
import qmv_parity_compare as C  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# metal -std=metal3.1 -O2 -S | metal-opt -passes='default<O3>'
# research/e38-artifacts/air-registers.txt
AIR_REGS = [
    ("base  _m<T,6,3,true,4>", 83, "[4 x [4 x i16]]", 2, None),
    ("arm_a _m<T,6,3,true,2>", 66, "[2 x [4 x i16]]", 3, 66),
    ("arm_b _m<T,6,6,true,2,true>", 106, "[2 x [4 x i16]]", 2, None),
    ("e33   _m<T,6,6,true,2>", 117, "[2 x [4 x i16]]", 3, 117),
    # the register wall that makes row-blocking mandatory at NA=6, and the
    # rung-2 cells that are over it before any timing
    ("wall  _m<T,6,6,true,4> (crossrow_na6)", 144, "[4 x <6 x float>]", 2, None),
    ("rung2 _m<T,7,7,true,2>", 134, "[2 x [4 x i16]]", 3, None),
    ("rung2 _m<T,8,8,true,2>", 151, "[2 x [4 x i16]]", 3, None),
]

INSTANTIATION = {
    "base": "qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>",
    "a": "qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true, 2>",
    "b": "qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true, 2, true>",
}


def _flag(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def _in_band(r: dict) -> bool:
    return bool(r["lo"] <= r["measured"] <= r["hi"])


def _ranked_geom(
    path: str = "research/e38-artifacts/e38-ranked-geom.json",
) -> dict | None:
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _parity(out_dir: str = ".mlxfast-private/qmv-parity") -> dict | None:
    """Cross-build digest comparison of each arm against base.

    A digest comparison reports bit-identical for free when two arms happen to
    build the same source, so the per-arm twin hashes are carried alongside the
    verdict and their distinctness is what makes the pass informative.
    """
    d = pathlib.Path(out_dir)
    if not (d / "base.json").exists():
        return None

    def twin_digests(arm: str) -> list[str]:
        f = d / f"{arm}.twins.txt"
        if not f.exists():
            return []
        return [ln.split()[0] for ln in f.read_text().splitlines() if ln.strip()]

    ref = C.load(str(d / "base.json"))
    payload = json.loads((d / "base.json").read_text())
    arms = [p.stem for p in sorted(d.glob("*.json")) if p.stem != "base"]

    per_arm = {}
    for arm in arms:
        cur = C.load(str(d / f"{arm}.json"))
        shared = set(ref) & set(cur)
        differing = [k for k in sorted(shared) if ref[k] != cur[k]]
        per_arm[arm] = {
            "cells_compared": len(shared),
            "cells_differing": len(differing),
            "differing_by_bits_m": sorted({(b, m) for _, b, m in differing}),
            "bit_identical": not differing,
            "twin_digests": twin_digests(arm),
        }

    base_twins = twin_digests("base")
    distinct = len({tuple(base_twins)}
                   | {tuple(v["twin_digests"]) for v in per_arm.values()})
    return {
        "arms": arms,
        "cells_per_arm": len(ref),
        "bits": sorted({b for _, b, _ in ref}),
        "cells_by_bits": payload.get("cells_by_bits"),
        # cells where the crossrow cell under test is the kernel actually
        # dispatched; the remainder still validate the build but do not
        # exercise the changed code
        "covering_cells_by_bits": payload.get("covering_cells_by_bits"),
        "per_arm": per_arm,
        "total_cells_compared": sum(v["cells_compared"] for v in per_arm.values()),
        "total_cells_differing": sum(v["cells_differing"] for v in per_arm.values()),
        "all_bit_identical": all(v["bit_identical"] for v in per_arm.values()),
        "base_twin_digests": base_twins,
        # a cross-build digest match is free if two arms built the same source,
        # so distinctness is what makes the pass informative rather than vacuous
        "all_source_digests_distinct": distinct == 1 + len(arms),
    }


def main() -> None:
    resume_id = _flag("--resume")
    positional = [a for a in sys.argv[1:]
                  if not a.startswith("--") and a != resume_id]
    path = pathlib.Path(positional[0] if positional
                        else "research/e38-artifacts/e38-metrics.json")
    d = json.loads(path.read_text())

    ident = {k: d["arms"][k]["identity"] for k in ("base", "a", "b")}
    temps = [float(ident[k]["gpu_temp_c_before_vendored"]) for k in ident]
    geo = d["geometry"]
    rel = d["relations"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=resume_id,
        resume="must" if resume_id else None,
        name="e38-rowblocks-in-idle-xblocks-m6",
        job_type="cost-curve",
        tags=["e38", "crossrow-qmv", "row-blocking", "m6", "cost-curve",
              "prereg", "e33-followup"],
        config={
            "assignment_id": "qwen38-r1-e38-rowblocks-in-idle-xblocks-m6",
            "revision_id": "r1",
            "student": "qwen-thorfinn",
            "base_sha": ident["base"]["head"],
            "arm_a_head": ident["a"]["head"],
            "arm_b_head": ident["b"]["head"],
            "host_architecture": geo["architecture"],
            "treated_width": d["treated"],
            "widths": d["widths"],
            "control_widths": d["control_widths"],
            "instantiation_base": INSTANTIATION["base"],
            "instantiation_arm_a": INSTANTIATION["a"],
            "instantiation_arm_b": INSTANTIATION["b"],
            "instantiation_e33": "qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true, 2>",
            "e33_measured_ratio": P.E33_RATIO_M6,
            "assignment_target_ratio": P.ADVISOR_RATIO,
            "prereg_point": P.REGISTERED_RATIO,
            "prereg_band": list(P.REGISTERED_BAND),
            "prereg_route1_ratio_b": P.route1()["ratio_b"],
            "prereg_route2_ratio_b": P.route2()["ratio"],
            # every cost-curve number in this campaign ran under MLX arch
            # defaults: QwenQMVCostCurveTests never builds QwenRuntimeMTPWorker
            # so RuntimeStartupMemoryPolicy never runs
            "geometry_architecture": geo["architecture"],
            "geometry_architecture_class": geo["architecture_class"],
            "geometry_startup_policy_applied": geo["policy_applied"],
            "geometry_max_ops_per_buffer": geo["mlx_default_ops_per_buffer"],
            "geometry_max_mb_per_buffer": geo["mlx_default_mb_per_buffer"],
            "geometry_env_overrides": geo["env_overrides"],
            "geometry_resident_weights": False,
            "ranked_geometry_max_mb_per_buffer": 512,
            "ranked_geometry_max_ops_per_buffer": 50,
            "cool_gate": ident["base"].get("cool_gate_vendored"),
            "gate_qualified_for_timing": True,
            "psi_measured": P.PSI_MEASURED,
            "phi_local_cost_share": P.PHI_LOCAL,
            "phi_ranked_row_share": P.PHI_RANKED_ROWS,
            "score_chain": P.SCORE_CHAIN,
            "sigma_score_pct": P.SIGMA_SCORE_PCT,
            "gap_to_rank1_pct": P.GAP_TO_FIRST_PCT,
        },
    )

    c = d["c_round_ms"]
    ladder = wandb.Table(columns=[
        "M", "base_ms", "arm_a_ms", "arm_b_ms", "a_over_base", "b_over_base",
        "b_over_a", "treated", "ipg_shipped"])
    for m in d["widths"]:
        k = str(m)
        cb, ca, cbb = c["base"][k], c["a"][k], c["b"][k]
        ladder.add_data(m, cb, ca, cbb, ca / cb, cbb / cb, cbb / ca,
                        m == d["treated"], P.SHIPPED_IPG.get(m))

    shapes = wandb.Table(columns=[
        "shape", "n", "k", "calls", "base_us", "arm_a_us", "arm_b_us",
        "a_over_base", "b_over_base", "e38_predicted", "e33_measured",
        "row_groups"])
    for s in d["shapes_m6"]:
        shapes.add_data(s["name"], s["n"], s["k"], s["calls"],
                        s["base_s"] * 1e6, s["a_s"] * 1e6, s["b_s"] * 1e6,
                        s["a_s"] / s["base_s"], s["b_s"] / s["base_s"],
                        _f(s["e38_pred"]), _f(s["e33"]),
                        math.ceil(s["n"] / 8))

    relations = wandb.Table(columns=[
        "id", "band_lo", "band_hi", "registered", "measured", "in_band"])
    for key, r in rel.items():
        relations.add_data(key, r["lo"], r["hi"], r["registered"],
                           r["measured"], _in_band(r))

    controls = wandb.Table(columns=[
        "M", "ratio_a", "ratio_b", "abs_dev_a", "abs_dev_b", "within_tol"])
    for ctl in d["controls"]:
        da, db = abs(ctl["ratio_a"] - 1), abs(ctl["ratio_b"] - 1)
        controls.add_data(ctl["m"], ctl["ratio_a"], ctl["ratio_b"], da, db,
                          max(da, db) <= d["control_tol"])

    air = wandb.Table(columns=[
        "cell", "peak_live_regs", "activation_alloca", "backedges",
        "predicted", "spilled"])
    for cell, regs, ty, be, pred in AIR_REGS:
        air.add_data(cell, regs, ty, be, _f(pred), False)

    preflight = wandb.Table(columns=["M", "base_pct", "arm_a_pct", "arm_b_pct"])
    for p in d["preflight"]:
        preflight.add_data(p["m"], p["base"], p["a"], p["b"])

    thermal = wandb.Table(columns=[
        "arm", "tag", "head", "dirty", "cool_gate", "entry_c", "exit_c"])
    for k in ("base", "a", "b"):
        i = ident[k]
        thermal.add_data(k, d["arms"][k]["tag"], i["head"][:12], i["dirty"],
                         i.get("cool_gate_vendored"),
                         float(i["gpu_temp_c_before_vendored"]),
                         float(i["gpu_temp_c_after_vendored"]))

    # Advisor 5337633069/5337689508: only beagle (our 4th order statistic) and
    # medicine (our 5th) can move the score, so the per-prompt legs are logged
    # separately and no aggregate over the eight prompts is recorded.
    ratio = d["primary"]["value"]
    leg = V.leg_pct(ratio)
    prompts = wandb.Table(columns=[
        "prompt", "order_statistic", "carries_score_weight", "deficit_pct",
        "deficit_ms_per_round", "ranked_rounds", "plateau_between_row_sd_pct",
        "predicted_leg_movement_pct", "share_of_deficit_closed",
        "predicted_ms_per_round", "measured_this_round"])
    for name, p in V.PROMPTS.items():
        share = leg / p["deficit_pct"]
        prompts.add_data(name, p["order_stat"], True, p["deficit_pct"],
                         p["deficit_ms_per_round"], p["rounds"],
                         p["plateau_sd_pct"], leg, share,
                         share * p["deficit_ms_per_round"], False)

    # Every null states its MDE (advisor 5337633069 item 7); the exact
    # noncentral-t figure is reported next to the normal one because at n=2
    # paired the normal approximation understates the floor 5.83x.
    sd, ctl_widths, drift = V.controls_sd_pct(d)
    se_curve = sd * math.sqrt(1.0 + 1.0 / len(ctl_widths))
    mde = wandb.Table(columns=[
        "null", "design", "n", "sd_pct", "mde_normal_pct", "mde_exact_pct",
        "effect_pct", "detected", "note"])
    mde.add_data("m6_cost_curve_effect", "1 treated width vs controls",
                 len(ctl_widths), sd, V.normal_mde(se_curve),
                 V.exact_mde(se_curve, len(ctl_widths) - 1), (ratio - 1) * 100.0,
                 True, f"controls M>=3 only: {ctl_widths}; drift {drift:+.4f}%")
    for n in (4, 2):
        se = V.E2E_PAIR_SD_PCT / math.sqrt(n)
        mde.add_data("end_to_end_decode_leg", "paired legs", n,
                     V.E2E_PAIR_SD_PCT, V.normal_mde(se),
                     V.exact_mde(se, n - 1), leg, False,
                     "NOT RUN: predicted movement is under the floor")
    for name, p in V.PROMPTS.items():
        mde.add_data(f"{name}_leg_vs_plateau", "single board reading", 1,
                     p["plateau_sd_pct"], V.normal_mde(p["plateau_sd_pct"]),
                     float("nan"), leg, False,
                     "not measured this round; E2E arm skipped")
    mde.add_data("untreated_widths", "1 treated width vs controls",
                 len(ctl_widths), sd, V.normal_mde(se_curve),
                 V.exact_mde(se_curve, len(ctl_widths) - 1),
                 d["control_worst_abs_dev"] * 100.0, False,
                 "flat within the floor")

    bars = wandb.Table(columns=["target", "score_pct", "required_ratio", "cleared"])
    for label, target in (("1_sigma", V.SIGMA_SCORE_PCT),
                          ("2_sigma", 2 * V.SIGMA_SCORE_PCT),
                          ("engineerable_gap", V.ENGINEERABLE_PCT),
                          ("crown", V.CROWN_GAP_PCT)):
        need = V.ratio_for_score(target)
        bars.add_data(label, target, need, bool(ratio <= need))

    step = S.step_summary()
    ladder_step = wandb.Table(columns=list(step.keys()))
    ladder_step.add_data(*step.values())

    tables = {
        "e38/ladder": ladder,
        "e38/shapes_m6": shapes,
        "e38/relations": relations,
        "e38/controls": controls,
        "e38/air_registers": air,
        "e38/jit_preflight": preflight,
        "e38/thermal_provenance": thermal,
        "e38/scoring_prompts": prompts,
        "e38/mde": mde,
        "e38/score_bars": bars,
        "e38/m5_to_m6_step": ladder_step,
    }

    rg = _ranked_geom()
    if rg is not None:
        rg_table = wandb.Table(columns=[
            "M", "arch_ms", "ranked_ms", "delta_pct", "trusted"])
        for row in rg["per_width"]:
            rg_table.add_data(row["verify_width"], row["arch_ms"],
                              row["ranked_ms"], row["delta_pct"],
                              bool(row["trusted"]))
        tables["e38/ranked_geometry"] = rg_table

    par = _parity()
    if par is not None:
        par_table = wandb.Table(columns=[
            "arm", "cells_compared", "cells_differing", "bit_identical",
            "twin_digest_quantized_h", "twin_digest_quantized_cpp"])
        par_table.add_data("base (reference)", par["cells_per_arm"], 0, True,
                           *(par["base_twin_digests"] + ["", ""])[:2])
        for arm, v in par["per_arm"].items():
            par_table.add_data(arm, v["cells_compared"], v["cells_differing"],
                               v["bit_identical"],
                               *(v["twin_digests"] + ["", ""])[:2])
        tables["e38/parity_digest"] = par_table

    run.log(tables)

    prim = d["primary"]
    ratio_b = prim["value"]
    run.summary.update({
        # primary: drift-adjusted per-row QMV cost at M=6, arm (b) vs base
        "e38/m6_per_row_cost_ratio": ratio_b,
        "e38/m6_per_row_cost_ratio_raw": prim["raw"],
        "e38/m6_per_row_cost_ratio_arm_a": d["ratio_a"],
        "e38/m6_per_row_cost_ratio_arm_a_raw": d["ratio_a_raw"],
        "e38/prereg_point": P.REGISTERED_RATIO,
        "e38/prereg_band_lo": P.REGISTERED_BAND[0],
        "e38/prereg_band_hi": P.REGISTERED_BAND[1],
        "e38/prereg_hit": bool(P.REGISTERED_BAND[0] <= ratio_b <= P.REGISTERED_BAND[1]),
        "e38/assignment_target": P.ADVISOR_RATIO,
        "e38/assignment_target_hit": bool(ratio_b <= P.ADVISOR_RATIO),
        "e38/R1_weight_pass": rel["R1_weight_pass"]["measured"],
        "e38/R2_activation_doubling": rel["R2_activation_doubling"]["measured"],
        "e38/R3_serialization": rel["R3_serialization"]["measured"],
        "e38/R1_in_band": _in_band(rel["R1_weight_pass"]),
        "e38/R2_in_band": _in_band(rel["R2_activation_doubling"]),
        "e38/R3_in_band": _in_band(rel["R3_serialization"]),
        "e38/e33_ratio": P.E33_RATIO_M6,
        "e38/leg_movement_pct": d["leg_movement_pct"],
        "e38/score_gain_pct": d["score_gain_pct"],
        "e38/decisive_threshold": P.DECISIVE_RATIO,
        "e38/ship_threshold": P.SHIP_RATIO,
        "e38/e2e_threshold": P.E2E_RATIO,
        "e38/null_threshold": 1 - P.CONTROL_TOL,
        "e38/verdict": d["verdict"],
        "e38/e2e_run": d["run_e2e"],
        "e38/dispatch_identity_ok": d["dispatch_identity_ok"],
        "e38/controls_worst_abs_dev": d["control_worst_abs_dev"],
        "e38/control_tol": d["control_tol"],
        "e38/drift_a": d["drift_a"],
        "e38/drift_b": d["drift_b"],
        "e38/m1_null_ratio_a": d["m1_null"]["ratio_a"] if d["m1_null"] else float("nan"),
        "e38/m1_null_ratio_b": d["m1_null"]["ratio_b"] if d["m1_null"] else float("nan"),
        "e38/bitwise_base": d["bitwise"]["base"],
        "e38/bitwise_arm_a": d["bitwise"]["a"],
        "e38/bitwise_arm_b": d["bitwise"]["b"],
        "e38/entry_temp_min_c": min(temps),
        "e38/entry_temp_max_c": max(temps),
        "e38/entry_temp_spread_c": max(temps) - min(temps),
        # corrected value chain: a kernel change moves BOTH scoring legs, so the
        # sensitivity is the advisor's "both legs" ladder row, not the
        # beagle-only 0.4827 used in the interim comment
        "e38/leg_movement_pct_corrected": leg,
        "e38/score_gain_pct_corrected": V.score_pct(ratio_b),
        "e38/score_gain_sigmas": V.score_pct(ratio_b) / V.SIGMA_SCORE_PCT,
        "e38/share_of_engineerable_gap": V.score_pct(ratio_b) / V.ENGINEERABLE_PCT,
        "e38/sigma_score_pct_corrected": V.SIGMA_SCORE_PCT,
        "e38/engineerable_gap_pct_corrected": V.ENGINEERABLE_PCT,
        "e38/score_per_leg_pct_both_legs": V.SCORE_PER_LEG_PCT,
        "e38/ratio_required_for_crown": V.ratio_for_score(V.CROWN_GAP_PCT),
        "e38/ratio_required_for_engineerable_gap":
            V.ratio_for_score(V.ENGINEERABLE_PCT),
        "e38/mde_cost_curve_normal_pct": V.normal_mde(se_curve),
        "e38/mde_cost_curve_exact_pct": V.exact_mde(se_curve, len(ctl_widths) - 1),
        "e38/mde_e2e_n4_normal_pct": V.normal_mde(V.E2E_PAIR_SD_PCT / 2.0),
        "e38/mde_e2e_n4_exact_pct": V.exact_mde(V.E2E_PAIR_SD_PCT / 2.0, 3),
        "e38/e2e_underpowered_factor_exact":
            V.exact_mde(V.E2E_PAIR_SD_PCT / 2.0, 3) / abs(leg),
        "e38/control_sd_pct_m3_plus": sd,
        "e38/beagle_leg_measured": False,
        "e38/medicine_leg_measured": False,
        "e38/m5_to_m6_step_pct": step["step_pct"],
        "e38/m5_to_m6_priced_weight_stream_ms": step["priced_weight_stream_ms"],
        "e38/m5_to_m6_residual_ms": step["residual_one_more_row_ms"],
        "e38/air_regs_na6_r4_wall": 144,
        "e38/air_regs_shipped_high_water": 125,
        "e38/rung2_dead_on_registers": True,
    })

    if rg is not None:
        run.summary.update({
            # deliverable (d): does the ranked runner's MLX buffer geometry move
            # the shape-level cost curve? A width-ordered residual would mean it
            # does; a flat one of drift size means the local curves transfer.
            "e38/ranked_geom_mean_delta_pct": rg["mean_delta_pct"],
            "e38/ranked_geom_sd_delta_pp": rg["sd_delta_pp"],
            "e38/ranked_geom_t_stat": rg["t_stat"],
            "e38/ranked_geom_df": rg["df"],
            "e38/ranked_geom_abs_max_delta_pct": rg["abs_max_delta_pct"],
            "e38/ranked_geom_width_trend_pearson_r": rg["width_trend_pearson_r"],
            "e38/ranked_geom_m6_delta_pct": next(
                r["delta_pct"] for r in rg["per_width"] if r["verify_width"] == 6),
            "e38/ranked_geom_mean_over_mde": rg["mean_over_mde_normal"],
            "e38/ranked_geom_within_drift_envelope": rg["within_drift_envelope"],
            "e38/ranked_geom_verdict": rg["verdict"],
            "e38/ranked_geom_scope": (
                "shape-level GEMM only; a --shapes-only probe issues one op per "
                "call and is insensitive to MLX_MAX_OPS_PER_BUFFER by "
                "construction, so this is not evidence about end-to-end "
                "command-buffer behaviour"),
        })

    if par is not None:
        run.summary.update({
            "e38/parity_cells_per_arm": par["cells_per_arm"],
            "e38/parity_bits": par["bits"],
            "e38/parity_cells_by_bits": par["cells_by_bits"],
            "e38/parity_covering_cells_by_bits": par["covering_cells_by_bits"],
            "e38/parity_total_cells_compared": par["total_cells_compared"],
            "e38/parity_total_cells_differing": par["total_cells_differing"],
            "e38/parity_all_bit_identical": par["all_bit_identical"],
            # without this the bit-identical verdict would be vacuous: two arms
            # that accidentally built the same source always match
            "e38/parity_all_source_digests_distinct":
                par["all_source_digests_distinct"],
            "e38/parity_arms": par["arms"],
        })

    run.finish()
    print(f"logged {run.url}")


if __name__ == "__main__":
    main()
