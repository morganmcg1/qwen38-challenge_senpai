#!/usr/bin/env python3
"""Log E49 -- is the 2-stream M=9 prize local, shared, or absent -- to W&B.

Ledger 173(C) prices a 2-stream M=9 at +5.36 % of score from a cell nobody has
timed. E49 times it in isolation (arm 1) and, separately, turns the kernel-wide
register allocation with an unreachable switch case so the shared-allocation tax
can be measured on widths whose instructions did not change (arm 2).

The durable record is the pre-registration, the compile-only dose census, the
contention instruments and their two-directional self-tests, every leg's
dispatch readback and source digests, the per-width curves with their same-build
replicate floors, and the thermal record.

  python3 research/e49_wandb_log.py research/e49-artifacts/e49-metrics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
PREREG = pathlib.Path("research/e49-prereg.json")
CENSUS = pathlib.Path("research/e49-reg-census.json")
SELFTEST = pathlib.Path(".mlxfast-private/e49-legs/gpu-gate-selftest.json")


def _flag(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text())
    prereg = json.loads(PREREG.read_text())
    census = json.loads(CENSUS.read_text()) if CENSUS.exists() else {}
    selftest = json.loads(SELFTEST.read_text()) if SELFTEST.exists() else {}
    resume_id = _flag("--resume")

    legs = [l for l in d["legs"] if l["status"] == "ok"]
    arm1 = d.get("arm1", {})
    arm2 = d.get("arm2", {})

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=resume_id,
        resume="must" if resume_id else None,
        name=_flag("--name") or "e49-m9-two-stream-local-vs-shared",
        job_type="kernel-probe",
        tags=["e49", "qwen38-mtp", "crossrow-qmv", "register-ceiling",
              "research-only", "not-gate-qualified"],
        notes=("Arm 1: <T,9,5> vs <T,9,3> compiled as the only crossrow case, "
               "so the cell under test owns the shared allocation. Arm 2: an "
               "unreachable case 10 turns the kernel-wide allocation without "
               "changing any dispatched width's instructions. Local timing on "
               "an ungated host; directional causal evidence inside one "
               "counterbalanced session, never a ranked score."),
        config={
            "assignment": prereg["assignment"],
            "base_sha": prereg["base_sha"],
            "pr": 53,
            "harness": "research/run-qmv-curve.sh (E46's, unchanged)",
            "metric": "T(M) = sum over 8 scored shapes of calls_per_verify * "
                      "seconds_per_call, ms",
            "prereg": prereg,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "run_lock_dir": "/tmp/mlxfast-shared",
            "gpu_gate_selftest_passed": selftest.get("selftest_passed"),
        },
    )

    tables = {}

    leg_tbl = wandb.Table(columns=[
        "tag", "arm", "gpu_gate_state", "gate_max_util", "gate_priming_discarded",
        "entry_temp_c", "exit_temp_c", "cool_gate", "header_sha256",
        "twin_sha256", "bitwise_failures"])
    for leg in legs:
        ident = leg["identity"]
        src = leg["sources_as_measured"]
        gate = leg["gpu_gate"]
        leg_tbl.add_data(
            leg["tag"], leg["arm"], gate.get("state"), _f(gate.get("max")),
            _f(gate.get("priming_discarded")),
            _f(ident.get("gpu_temp_c_before_vendored")),
            _f(ident.get("gpu_temp_c_after_vendored")),
            str(ident.get("cool_gate_vendored")),
            src["Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"][:16],
            src["Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"][:16],
            len(leg["bitwise_failures"]))
    tables["legs"] = leg_tbl

    curve = wandb.Table(columns=["tag", "arm", "M", "t_ms", "jit_spread_pct",
                                 "in_kernel_path", "weight_streams",
                                 "inputs_per_group"])
    for leg in legs:
        for m, t in sorted(leg["t_ms"].items(), key=lambda kv: int(kv[0])):
            paths, strm, ipg = leg["dispatch"].get(str(m), leg["dispatch"].get(m, ([], [], [])))
            curve.add_data(leg["tag"], leg["arm"], int(m), _f(t),
                           _f(leg["jit_spread_pct"].get(str(m),
                                                        leg["jit_spread_pct"].get(m))),
                           ",".join(map(str, paths)), ",".join(map(str, strm)),
                           ",".join(map(str, ipg)))
    tables["width_curves"] = curve

    if arm1:
        a1 = wandb.Table(columns=["M", "iso3_ms", "iso5_ms", "delta_ms",
                                  "delta_pct", "mde_ms", "exceeds_mde", "role"])
        for m, r in sorted(arm1["per_width"].items(), key=lambda kv: int(kv[0])):
            a1.add_data(int(m), r["control_ms"], r["treated_ms"], r["delta_ms"],
                        r["delta_pct"], r["mde_ms"], r["exceeds_mde"],
                        "contrast" if int(m) == 9 else "unchanged code")
        tables["arm1_isolated"] = a1

    if arm2 and arm2.get("doses"):
        a2 = wandb.Table(columns=["dose", "cell_max_regs", "entry_heuristic",
                                  "M", "control_ms", "treated_ms", "delta_ms",
                                  "delta_pct", "mde_ms"])
        doses = {x["arm"]: x for x in prereg["arm2_shared"]["doses"]}
        for name, dose in arm2["doses"].items():
            meta = doses.get(name, {})
            for m, r in sorted(dose["per_width"].items(), key=lambda kv: int(kv[0])):
                a2.add_data(name, meta.get("cell_max"), meta.get("entry_heuristic"),
                            int(m), r["control_ms"], r["treated_ms"],
                            r["delta_ms"], r["delta_pct"], r["mde_ms"])
        tables["arm2_dose_ladder"] = a2

        pooled = wandb.Table(columns=["dose", "cell_max_regs", "pooled_tax_pct",
                                      "worst_width_pct", "n_widths_slower",
                                      "n_widths"])
        for name, dose in arm2["doses"].items():
            pooled.add_data(name, doses.get(name, {}).get("cell_max"),
                            dose["pooled_tax_pct"], dose["worst_width_pct"],
                            dose["n_widths_slower"], dose["n_widths"])
        tables["arm2_dose_response"] = pooled

    if census:
        reg = wandb.Table(columns=["arm", "family", "dispatched_reg_max",
                                   "entry_point_reg_max", "doc"])
        for a in census.get("arms", []):
            if a.get("status") == "ok":
                reg.add_data(a["name"], a["family"], a["dispatched_reg_max"],
                             a["entry_point_reg_max"], a["doc"])
        tables["register_census_heuristic"] = reg

        cells = wandb.Table(columns=["cell", "peak_live_regs", "streams",
                                     "na_cells", "acc_alloca_types"])
        for name, c in census.get("dose_candidates", {}).items():
            if c.get("status") == "ok":
                cells.add_data(name, c["peak_live_regs"], c["streams"],
                               str(c["na_cells"]), str(c["acc_alloca_types"]))
        tables["dose_cell_census"] = cells

    if selftest:
        st = wandb.Table(columns=["phase", "state", "samples", "max",
                                  "priming_discarded", "longest_busy_run"])
        for phase in ("under_load", "after_load"):
            s = selftest.get(phase, {})
            st.add_data(phase, s.get("state"), str(s.get("samples")),
                        _f(s.get("max")), _f(s.get("priming_discarded")),
                        _f(s.get("longest_busy_run")))
        tables["gpu_gate_selftest"] = st

    run.log(tables)

    if arm1:
        v = arm1["verdict"]
        run.summary["arm1_delta_pct_m9"] = v["delta_pct_m9"]
        run.summary["arm1_delta_ms_m9"] = v["delta_ms_m9"]
        run.summary["arm1_mde_ms_m9"] = v["mde_ms_m9"]
        run.summary["arm1_predicted_pct_refit"] = v["predicted_pct"][0]
        run.summary["arm1_predicted_pct_contrast_b"] = v["predicted_pct"][1]
        run.summary["arm1_worst_unchanged_width_pct"] = v[
            "worst_unchanged_code_width_pct"]
        run.summary["arm1_label"] = v["label"]
    if arm2 and arm2.get("doses"):
        for name, dose in arm2["doses"].items():
            run.summary["arm2_%s_pooled_tax_pct" % name] = dose["pooled_tax_pct"]
        run.summary["arm2_label"] = arm2.get("label")
    run.summary["legs_measured"] = len(legs)
    run.summary["bitwise_failures_total"] = sum(
        len(l["bitwise_failures"]) for l in legs)
    run.summary["gate_qualified_for_timing"] = False
    run.summary["cool_gate_passed_real_gate"] = all(
        str(l["identity"].get("cool_gate_vendored")) == "passed" for l in legs)

    print("logged %s tables to %s" % (len(tables), run.url))
    run.finish()


if __name__ == "__main__":
    main()
