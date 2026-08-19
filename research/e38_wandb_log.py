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
import e38_prereg as P  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# metal -std=metal3.1 -O2 -S | metal-opt -passes='default<O3>'
# research/e38-artifacts/air-registers.txt
AIR_REGS = [
    ("base  _m<T,6,3,true,4>", 83, "[4 x [4 x i16]]", 2, None),
    ("arm_a _m<T,6,3,true,2>", 66, "[2 x [4 x i16]]", 3, 66),
    ("arm_b _m<T,6,6,true,2,true>", 106, "[2 x [4 x i16]]", 2, None),
    ("e33   _m<T,6,6,true,2>", 117, "[2 x [4 x i16]]", 3, 117),
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

    run.log({
        "e38/ladder": ladder,
        "e38/shapes_m6": shapes,
        "e38/relations": relations,
        "e38/controls": controls,
        "e38/air_registers": air,
        "e38/jit_preflight": preflight,
        "e38/thermal_provenance": thermal,
    })

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
    })
    run.finish()
    print(f"logged {run.url}")


if __name__ == "__main__":
    main()
