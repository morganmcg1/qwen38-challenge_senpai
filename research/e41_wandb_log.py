#!/usr/bin/env python3
"""Log the E41 K-tile re-read-distance ladder to W&B.

E41 resolves E38's R2 (+10.54 % for halving the row tile at M=6) into two
candidate mechanisms: the doubled/distant activation re-read (MEM) versus the
halved register tile and extra loop (ILP). The durable record is the
pre-registration, the compile-only AIR census that gated the GPU time, the
dispatch readback, the ladder, the untreated-width controls, the thermal record
and the cross-build parity digests.

  python3 research/e41_wandb_log.py research/e41-artifacts/e41-metrics.json
  python3 research/e41_wandb_log.py research/e41-artifacts/e41-metrics.json \
      --resume <run_id>
"""

from __future__ import annotations

import json
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e41_prereg as P  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
CENSUS = pathlib.Path("research/e41-ktile-census.json")
PARITY = pathlib.Path(".mlxfast-private/qmv-parity")


def _flag(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def main() -> None:
    metrics_path = sys.argv[1]
    d = json.loads(pathlib.Path(metrics_path).read_text())
    resume_id = _flag("--resume")
    ident = d["identity"]
    base_id, arm_id, base2_id = ident["base"], ident["arm"], ident["base2"]

    census = json.loads(CENSUS.read_text()) if CENSUS.exists() else {"cells": []}
    cells = {c["name"]: c for c in census["cells"]}

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=resume_id,
        resume="must" if resume_id else None,
        name="e41-ktile-reread-distance-ladder",
        job_type="cost-curve",
        tags=["e41", "crossrow-qmv", "k-tiling", "row-blocking", "cost-curve",
              "prereg", "e38-followup", "mechanism-discrimination"],
        config={
            "assignment_id": "qwen38-r1-e41-r2-confound-before-ktiling",
            "revision_id": "r1",
            "student": "qwen-thorfinn",
            "pr_number": 46,
            "base_sha": "04ad6bf11437c269df85a47e91faa769c74fe6da",
            "base_build_head": base_id.get("head"),
            "arm_build_head": arm_id.get("head"),
            "host": base_id.get("host"),
            "reps": base_id.get("reps"),
            "inner": base_id.get("inner"),
            "widths": d["widths"],
            "treated_widths": d["treated"],
            "control_widths": d["control"],
            "arm_map": {str(m): v[0] for m, v in P.ARM_MAP.items()},
            "arm_map_roles": {str(m): v[1] for m, v in P.ARM_MAP.items()},
            "discriminating_step": "M=4 (KT=64) -> M=8 (KT=4), pure NA=4",
            "total_recovery_bound": "M=7 (KT=1), mixed NA=4+3",
            "anchor": "M=6 (r=2, BPC=1) replicates E38 arm(a)",
            "e38_arm_a_m6_ratio": P.E38_ARM_A_M6,
            "prereg_control_band": P.CONTROL_BAND,
            "prereg_anchor_band": list(P.PRED_M6_ANCHOR),
            "prereg_ktall_na4_band": list(P.PRED_KTALL_NA4),
            "prereg_mem_threshold": P.LOCALITY_STEP_MEM,
            "prereg_ilp_threshold": P.LOCALITY_STEP_ILP,
            "prereg_amendments": [a[0] for a in P.CENSUS_AMENDMENTS],
            # the thermal contract, recorded honestly: the real gate ran and
            # stalled above its 40C target on this host, so these numbers are
            # directional causal evidence, not a gate-qualified score.
            "cool_gate": base_id.get("cool_gate_vendored"),
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "gpu_entry_c_base": _f(base_id.get("gpu_temp_c_before_vendored")),
            "gpu_entry_c_arm": _f(arm_id.get("gpu_temp_c_before_vendored")),
            "gpu_exit_c_base": _f(base_id.get("gpu_temp_c_after_vendored")),
            "gpu_exit_c_arm": _f(arm_id.get("gpu_temp_c_after_vendored")),
            # base-r2's much hotter exit is load-bearing: it agrees with base-r1
            # at M=9 to 0.05 %, which is what rules out a thermal explanation for
            # the M=9 control residual.
            "base2_build_head": base2_id.get("head"),
            "gpu_entry_c_base2": _f(base2_id.get("gpu_temp_c_before_vendored")),
            "gpu_exit_c_base2": _f(base2_id.get("gpu_temp_c_after_vendored")),
            "counterbalancing": ("A-B-A bracket across three sequential runs, "
                                 "not one interleaved ABBA session"),
            "e2e_leg_run": False,
            "e2e_leg_reason": "predicted 0.07-0.5 % vs n=4 MDE 0.417 %/0.632 %",
            "psi_phi_backsolved": P.PSI_PHI_BACKSOLVED,
            "psi_phi_is_measured": False,
            "score_sensitivity": P.SCORE_SENSITIVITY,
            "crown_pct": P.CROWN_PCT,
            "engineerable_gap_pct": P.GAP_PCT,
            "sigma_score_pct": P.SIGMA_SCORE_PCT,
        },
    )

    tables = {}

    c = d["c_round_ms"]
    ladder = wandb.Table(columns=["M", "base_ms", "arm_ms", "ratio_raw",
                                  "ratio_drift_adjusted", "role", "treated"])
    drift = d["control"]["median_drift"]
    for m in d["widths"]:
        k = str(m)
        role = P.ARM_MAP[m][1] if m in P.ARM_MAP else "untreated control"
        ladder.add_data(m, c[k]["base"], c[k]["arm"], c[k]["ratio_raw"],
                        c[k]["ratio_raw"] / drift, role, m in P.ARM_MAP)
    tables["e41/curve_by_width"] = ladder

    rungs = wandb.Table(columns=["M", "KT", "rho", "tax", "regs", "device_loads",
                                 "vector_float_ops", "loop_backedges"])
    for m, kt, cell in ((4, 64, "xkt_na4_r2_kt64"), (8, 4, "xkt_na4_r2_kt4"),
                        (7, 1, "xkt_na4_r2_kt1")):
        rho = d["ladder_rho"][str(m)]
        z = cells.get(cell, {})
        rungs.add_data(m, kt, rho, rho - 1.0, z.get("peak_live_regs"),
                       z.get("device_loads"), z.get("vector_float_ops"),
                       z.get("loop_backedges"))
    tables["e41/na4_ladder"] = rungs

    ctl = wandb.Table(columns=["M", "arm_over_base", "abs_dev", "inside_band"])
    for m, v in sorted(d["control"]["spread"].items(), key=lambda kv: int(kv[0])):
        ctl.add_data(int(m), v, abs(v - 1), abs(v - 1) <= P.CONTROL_BAND)
    tables["e41/untreated_controls"] = ctl

    cen = wandb.Table(columns=["cell", "arm", "NA", "rows_per_tile", "KT",
                               "peak_live_regs", "device_loads",
                               "vector_float_ops", "loop_backedges", "allocas",
                               "acc_spill", "fits_reg_wall", "timed"])
    for z in census["cells"]:
        if z.get("status") != "ok":
            continue
        cen.add_data(z["name"], z["arm"], z["na"], z["r"], z["kt"],
                     z["peak_live_regs"], z["device_loads"],
                     z["vector_float_ops"], z["loop_backedges"], z["allocas"],
                     z["acc_spill"], z["fits_reg_wall"], z["timed"])
    tables["e41/air_census"] = cen

    pred = wandb.Table(columns=["cell", "predicted_lo", "predicted_hi",
                                "measured_regs", "in_band", "fits_wall"])
    for name, measured in P.CENSUS_MEASURED_REGS.items():
        na, r, kt, _ = P.REGISTERED_CELLS[name]
        lo, hi = P.predicted_regs(na, kt, r)
        pred.add_data(name, lo, hi, measured, lo <= measured <= hi,
                      measured <= P.REG_WALL)
    tables["e41/registered_vs_measured_regs"] = pred

    jit = wandb.Table(columns=["M", "base_mean_over_min_pct", "arm_mean_over_min_pct"])
    for m, v in sorted(d["jit_spread_pct"].items(), key=lambda kv: int(kv[0])):
        jit.add_data(int(m), v["base"], v["arm"])
    tables["e41/jit_leak_check"] = jit

    amend = wandb.Table(columns=["amendment", "reason"])
    for what, why in P.CENSUS_AMENDMENTS:
        amend.add_data(what, why)
    tables["e41/prereg_amendments"] = amend

    # run-qmv-parity.sh writes one digest file per arm and prints the comparison
    # to stdout; it never writes a compare.json. Rejoin the per-cell digests here
    # so the durable record holds the actual evidence rather than a verdict line.
    parity = None
    bp, ap = PARITY / "base.json", PARITY / "arm.json"
    if bp.exists() and ap.exists():
        bd, ad = json.loads(bp.read_text()), json.loads(ap.read_text())
        key = lambda e: (e["bits"], e["m"], e["shape"])  # noqa: E731
        bmap = {key(e): e for e in bd["entries"]}
        pt = wandb.Table(columns=["bits", "m", "shape", "k", "n", "in_kernel_path",
                                  "digests_match"])
        compared, differing = 0, []
        for e in ad["entries"]:
            b = bmap.get(key(e))
            if b is None:
                continue
            same = b["digest"] == e["digest"]
            compared += 1
            if not same:
                differing.append(key(e))
            pt.add_data(e["bits"], e["m"], e["shape"], e["k"], e["n"],
                        e["in_kernel_path"], same)
        tables["e41/cross_build_parity"] = pt
        # Distinct twin digests prove two different builds were actually compared;
        # identical ones would make a bit-identical verdict vacuous.
        tw = {n: (PARITY / f"{n}.twins.txt").read_text().split()[0]
              for n in ("base", "arm") if (PARITY / f"{n}.twins.txt").exists()}
        parity = {
            "cells_compared": compared,
            "cells_differing": len(differing),
            "all_identical": compared > 0 and not differing,
            "differing_cells": differing[:20],
            "cells_by_bits": ad.get("cells_by_bits"),
            "covering_cells_by_bits": ad.get("covering_cells_by_bits"),
            "builds_distinct": len(set(tw.values())) == len(tw) == 2,
            "twin_header_sha256": tw,
        }

    if "width_floors" in d:
        wf = wandb.Table(columns=["m", "role", "replicate_floor_pct", "arm_dev_pct",
                                  "exceeds_own_floor"])
        for width, rec in sorted(d["width_floors"].items(), key=lambda kv: int(kv[0])):
            role = "treated" if int(width) in [int(x) for x in d["treated"]] else "control"
            wf.add_data(int(width), role, rec["floor"] * 100.0,
                        rec["arm_dev"] * 100.0, rec["exceeds_own_floor"])
        tables["e41/per_width_noise_floor"] = wf

    run.log(tables)

    summary = {
        # primary: does shortening the activation re-read recover R2?
        "e41/locality_recovery_fraction": _f(d["locality_recovery"]),
        "e41/total_recovery_fraction": _f(d["total_recovery"]),
        "e41/rho_m4_kt64": _f(d["ladder_rho"]["4"]),
        "e41/rho_m8_kt4": _f(d["ladder_rho"]["8"]),
        "e41/rho_m7_kt1": _f(d["ladder_rho"]["7"]),
        "e41/locality_step_absolute": _f(d["ladder_rho"]["4"] - d["ladder_rho"]["8"]),
        "e41/verdict": d["verdict"],
        "e41/anchor_rho_m6": _f(d["anchor"]["rho"]),
        "e41/anchor_replicates_e38": bool(d["anchor"]["pass"]),
        "e41/na3_rho_m6": _f(d["na3"]["rho_m6"]),
        "e41/na3_rho_m3": _f(d["na3"]["rho_m3"]),
        "e41/na3_recovery_fraction": _f(d["na3"]["recovery"]),
        "e41/control_median_drift": _f(drift),
        "e41/control_worst_abs_dev": _f(d["control"]["worst_abs_dev"]),
        "e41/controls_pass": bool(d["control"]["pass"]),
        "e41/dispatch_readback_ok": bool(d["dispatch_readback_ok"]),
        "e41/bitwise_failures_arm": len(d["bitwise_failures"]["arm"]),
        "e41/bitwise_failures_base": len(d["bitwise_failures"]["base"]),
        "e41/all_gates_pass": bool(d["all_gates_pass"]),
        "e41/census_ladder_is_one_mechanism": not census.get("ladder_failures"),
        "e41/census_incumbent_unperturbed": not census.get("anchor_failures"),
        "e41/census_no_timed_cell_spills": not census.get("timed_failures"),
        "e41/na6_r1_ktile_regs": P.CENSUS_MEASURED_REGS["xkt_na6_r1_kt64"],
        "e41/na6_r2_ktile_regs": P.CENSUS_MEASURED_REGS["xkt_na6_r2_kt64"],
        "e41/deliverable_b_licensed": d["verdict"].startswith("MEM"),
    }
    if parity:
        summary.update({
            "e41/parity_all_bit_identical": bool(parity["all_identical"]),
            "e41/parity_cells_compared": parity["cells_compared"],
            "e41/parity_cells_differing": parity["cells_differing"],
            "e41/parity_covering_cells_bits4":
                (parity["covering_cells_by_bits"] or {}).get("4"),
            "e41/parity_builds_distinct": parity["builds_distinct"],
        })
    if "width_floors" in d:
        fl = d["width_floors"]
        lad = [str(x) for x in (4, 6, 7, 8)]
        summary.update({
            "e41/floor_worst_ladder_anchor_pct":
                max(fl[w]["floor"] for w in lad if w in fl) * 100.0,
            "e41/floor_m1_pct": fl["1"]["floor"] * 100.0 if "1" in fl else None,
            "e41/controls_above_own_floor":
                [int(w) for w, r in fl.items() if r["exceeds_own_floor"]],
        })
    run.summary.update(summary)

    print(f"logged {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
