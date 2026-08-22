#!/usr/bin/env python3
"""Publish the E123 instruction price ladder and priced deletion audit to W&B.

    usage: research/e123_wandb_log.py [--only RUN]

  `e123-ladder`         the four new instruction classes -- threadgroup load,
                        threadgroup store, barrier and bf16 widening -- priced
                        beside the three E118 classes, plus the held-out
                        predictions that score the ladder.
  `e123-static-budget`  AIR counts, registers, spill bytes and ISA text for
                        every arm on the local `applegpu_g16s` and the ranked
                        `applegpu_g17s`, plus the all-widths entry-point
                        census that E118 never ran.
  `e123-deletion-price` the injection price against the deletion price of one
                        instruction class, measured in one counterbalanced
                        session instead of across two.
  `e123-rung1-audit`    the priced instruction census of
                        `qmv_fast_crossrow_affine4_g64_wide`, every group
                        classified bit-exact deletable, precision changing or
                        not deletable, ranked by round-weighted percent.

Every timed leg here is a standalone Metal microbenchmark. It holds no model
and runs no benchmark wrapper, so it passes no thermal gate. Each run logs
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false, and no leg here is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e123-instruction-price-ladder-and-priced-deletion-audit"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e123-artifacts")

PR = 124
ASSIGNMENT_BASE_SHA = "61ed64fe02346bd1fc021f1c664a9cd2c67286c4"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
HOLDOUT_GATE_PP = 1.0
PRIMARY_BASELINE_PP = 0.66
RUNG2_BAR_PCT = 1.0

# Republishing must correct a run in place. A second run of the same evidence
# would leave two disagreeing records of one experiment.
RUN_IDS = {
    "e123-ladder": "e123ladd1",
    "e123-static-budget": "e123stat1",
    "e123-deletion-price": "e123del01",
    "e123-rung1-audit": "e123rng11",
}

# The keys a reader of the report arrives looking for, per run id. Consumed by
# `research/e118_wandb_check.py --experiment e123 --verify`.
EXPECTED_SUMMARY_KEYS = {
    "e123ladd1": (
        "primary_metric_name",
        "e123_ladder_out_of_sample_prediction_error_pp",
        "primary_metric_baseline_pp",
        "kill_rule_gate_pp",
        "kill_rule_fired",
        "max_holdout_error_pp",
        "e123_threadgroup_load_pct_per_instr_per_kblock_na4",
        "e123_threadgroup_store_pct_per_instr_per_kblock_na4",
        "e123_barrier_pct_per_barrier_per_kblock_na4",
        "e123_bf16_to_float_conversion_pct_per_instr_per_kblock_na4",
        "session_valid",
    ),
    "e123stat1": (
        "entrypoint_registers_a_base_local",
        "entrypoint_registers_a_base_ranked",
        "entrypoint_simdgroups_a_base_ranked",
        "x_cvtshift_rejected_before_timing",
    ),
    "e123del01": (
        "injection_over_deletion_ratio_na4",
        "ratio_inside_prereg_band",
        "rung1_deletion_price_divisor",
        "deletion_verdict",
    ),
    "e123rng11": (
        "e123_largest_predicted_bit_exact_deletion_pct_round_weighted",
        "rung2_bar_pct",
        "rung2_justified",
        "reconstruction_ratio_na4",
    ),
}


def summary() -> dict:
    return json.loads((ART / "summary.json").read_text())


def meta() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ART / "meta.txt").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def start(job_type: str, name: str, config: dict[str, object]):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name, id=RUN_IDS.get(name), resume="allow", config=config,
        reinit=True)


def gate_flags(instrument: str, timing_valid: bool) -> dict[str, object]:
    return {
        "timing_valid": timing_valid,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "instrument": instrument,
    }


def identity() -> dict[str, object]:
    m = meta()
    return {
        "experiment": GROUP,
        "pr": PR,
        "assignment_base_sha": ASSIGNMENT_BASE_SHA,
        "leg_commit": m.get("git_head"),
        "host": HOST,
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "chip": "Apple M4 Pro",
        "metal_version": m.get("metal_version"),
        "swift_version": m.get("swift_version"),
        "entry_points": "e118_iso_na2..na5",
        "transcribed_from": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "quantized.h qmv_fast_crossrow_affine4_g64_wide"
        ),
        "candidate_files_changed": 0,
        "fast_math": False,
        "preregistration": "research/e123-artifacts/preregistration.md",
    }


def _gates_summary(doc: dict) -> dict[str, object]:
    g = doc["validity_gates"]
    return {
        "session_valid": g["session_valid"],
        "gate_bandwidth_passed": g["bandwidth"]["passed"],
        "gate_bandwidth_max_implied_gb_s": g["bandwidth"]["max_implied_gb_s"],
        "gate_null_scaffold_passed": g["null_scaffold"]["passed"],
        "gate_positive_controls_passed": g["positive_controls"]["passed"],
        "positive_control_failure_count":
            len(g["positive_controls"]["failures"]),
        "positive_controls_excused_on_excluded_cells":
            len(g["positive_controls"]["excused_on_excluded_cells"]),
    }


def log_ladder() -> None:
    doc = summary()
    sec = doc["secondary_metrics"]
    pm = doc["primary_metric"]
    run = start(
        job_type="isolated-metal-arms", name="e123-ladder",
        config={
            "question": (
                "what does one threadgroup load, one threadgroup store, one "
                "threadgroup barrier and one bf16 to float conversion cost "
                "per instruction per k-block in the wide qmv inner loop"
            ),
            "leg_command": (
                "research/e123_probe.sh e123-full --shapes 0,1,2,3,4 "
                "--widths 2,3,4,5 --pairs 8 --samples 24"
            ),
            "widths": [2, 3, 4, 5],
            "holdout_arms": list(pm["per_holdout_na4_error_pp"].keys()),
            "holdout_gate_pp": HOLDOUT_GATE_PP,
            "primary_metric_baseline_pp": PRIMARY_BASELINE_PP,
            "sign_convention": "cost = percent slower than a_base",
            **identity(),
            **gate_flags("standalone Metal microbenchmark, one GPU", True),
        },
    )

    prices = wandb.Table(columns=[
        "class", "na2", "na3", "na4", "na5", "prereg_point", "prereg_lo",
        "prereg_hi", "na4_inside_band", "dropped_rungs_na4"])
    for klass, per in doc["prices"].items():
        band = doc["band_verdicts"].get(klass) or {}
        lo, hi = (band.get("band") or [None, None])
        prices.add_data(
            klass,
            *[(per.get(str(na)) or {}).get("price") for na in (2, 3, 4, 5)],
            band.get("point"), lo, hi, band.get("inside"),
            ", ".join((per.get("4") or {}).get("dropped_rungs") or []))
    run.log({"instruction_prices_by_width": prices})

    costs = wandb.Table(columns=["arm", "na", "cost_pct_slower_than_a_base",
                                 "iqr", "n"])
    for na, bucket in doc["cost_by_width"].items():
        for arm, cell in sorted(bucket.items()):
            costs.add_data(arm, int(na), cell.get("median"), cell.get("iqr"),
                           cell.get("n"))
    run.log({"arm_costs_by_width": costs})

    hold = wandb.Table(columns=[
        "arm", "na", "predicted_pct", "measured_pct", "error_pp",
        "signed_error_pp", "inside_gate"])
    for arm, per in doc["holdouts"].items():
        for na, cell in sorted(per.items()):
            err = cell["error_pp"]
            hold.add_data(arm, int(na), cell["predicted_pct"],
                          cell["measured_pct"], err, cell["signed_error_pp"],
                          err is not None and err <= HOLDOUT_GATE_PP)
    run.log({"held_out_predictions": hold})

    free = wandb.Table(columns=["prediction", "predicted", "measured",
                                "error_pp", "inside_gate"])
    for key in ("threadgroup_access", "exchange_cost"):
        cell = doc["free_predictions"].get(key)
        if cell is None:
            continue
        free.add_data(key, cell.get("predicted", cell.get("predicted_pp")),
                      cell.get("measured", cell.get("measured_pp")),
                      cell["error_pp"], cell["inside_gate"])
    run.log({"free_predictions": free})

    add = wandb.Table(columns=[
        "na", "scaffold_pct", "k_ld4_pct", "k_shuf4_pct",
        "additive_prediction_pct", "measured_pct", "excess_pp",
        "superadditive"])
    for na, cell in sorted(doc["additivity"].items()):
        add.add_data(int(na), cell["scaffold_pct"], cell["k_ld4_pct"],
                     cell["k_shuf4_pct"], cell["additive_prediction_pct"],
                     cell["measured_pct"], cell["excess_pp"],
                     cell["superadditive"])
    run.log({"cross_class_additivity": add})

    gaps = wandb.Table(columns=["arm", "na", "forward_reverse_gap_pct"])
    for key, value in sorted((doc.get("forward_reverse_gap") or {}).items()):
        arm, _, na = key.partition("|")
        gaps.add_data(arm, na, value)
    run.log({"defect16_forward_reverse_gap": gaps})

    bd = doc["block_dispersion"]
    disp = wandb.Table(columns=["shape", "na", "worst_arm_spread_pct"])
    for cell in bd["per_cell"]:
        disp.add_data(cell["shape"], cell["m"], cell["worst_arm_spread_pct"])
    run.log({"defect19_per_block_dispersion": disp})

    run.summary.update({
        "primary_metric_name": "e123_ladder_out_of_sample_prediction_error_pp",
        "e123_ladder_out_of_sample_prediction_error_pp":
            pm["e123_ladder_out_of_sample_prediction_error_pp"],
        "primary_metric_baseline_pp": PRIMARY_BASELINE_PP,
        "primary_metric_direction": "minimize",
        "kill_rule_gate_pp": pm["kill_rule_gate_pp"],
        "kill_rule_fired": pm["kill_rule_fired"],
        "max_holdout_error_pp": pm["max_error_pp"],
        **{"holdout_error_pp_" + a: v
           for a, v in pm["per_holdout_na4_error_pp"].items()},
        "e123_threadgroup_load_pct_per_instr_per_kblock_na4":
            sec["e123_threadgroup_load_pct_per_instr_per_kblock_na4"],
        "e123_threadgroup_store_pct_per_instr_per_kblock_na4":
            sec["e123_threadgroup_store_pct_per_instr_per_kblock_na4"],
        "e123_barrier_pct_per_barrier_per_kblock_na4":
            sec["e123_barrier_pct_per_barrier_per_kblock_na4"],
        "e123_bf16_to_float_conversion_pct_per_instr_per_kblock_na4":
            sec["e123_bf16_to_float_conversion_pct_per_instr_per_kblock_na4"],
        "bank_conflict_ratio_na4": doc["bank_conflict_ratio_na4"],
        "bank_conflict_inside_band": doc["bank_conflict_band"]["inside"],
        "conversion_inside_band": doc["conversion_band"]["inside"],
        "barrier_merge_k_bar8_na4_pp": doc["barrier_merge_k_bar8_na4_pp"],
        "exact_failure_count": len(doc["fidelity"]["exact_failures"]),
        "defect19_flagged_cell_count": bd["flagged_count"],
        "defect19_max_cell_spread_pct": bd["max_cell_spread_pct"],
        **_gates_summary(doc),
    })
    run.finish()


def log_static() -> None:
    doc = summary()
    entry = doc["entrypoint_census"]
    run = start(
        job_type="static-budget", name="e123-static-budget",
        config={
            "question": (
                "what does each arm cost in registers, spill bytes and ISA "
                "text on the local and the ranked architecture, both per "
                "width and at the all-widths entry point the scored kernel "
                "actually presents"
            ),
            "leg_command": (
                "research/e123_arms.py --emit DIR && "
                "research/e123_arms.py --census DIR --out census.json && "
                "research/e123_arms.py --entrypoint-census DIR "
                "--out entrypoint-census.json"
            ),
            **identity(),
            **gate_flags("offline metal-arch translation on CPU, no GPU",
                         False),
        },
    )

    census = json.loads((ART / "census.json").read_text())
    static = wandb.Table(columns=[
        "arm", "arch", "na", "registers", "spill_bytes", "text_bytes"])
    for arm, row in census["arms"].items():
        for arch in (census["local_arch"], census["ranked_arch"]):
            for na, v in sorted((row.get(arch) or {}).items()):
                static.add_data(arm, arch, int(na), v["registers"],
                                v["spill_bytes"] or 0, v["text_bytes"])
    run.log({"registers_spill_text_by_width": static})

    ep = wandb.Table(columns=[
        "arm", "arch", "registers", "spill_bytes", "text_bytes",
        "resident_simdgroups"])
    for arch, table in entry["simdgroups"].items():
        for arm, sg in table.items():
            cell = entry["arms"][arm][arch]["0"]
            ep.add_data(arm, arch, cell["registers"], cell["spill_bytes"] or 0,
                        cell["text_bytes"], sg)
    run.log({"entrypoint_census_all_widths": ep})

    local_sg = entry["simdgroups"][LOCAL_ARCH]
    ranked_sg = entry["simdgroups"][RANKED_ARCH]
    run.summary.update({
        "entrypoint_registers_a_base_local":
            entry["arms"]["a_base"][LOCAL_ARCH]["0"]["registers"],
        "entrypoint_registers_a_base_ranked":
            entry["arms"]["a_base"][RANKED_ARCH]["0"]["registers"],
        "entrypoint_simdgroups_a_base_local": local_sg["a_base"],
        "entrypoint_simdgroups_a_base_ranked": ranked_sg["a_base"],
        "entrypoint_simdgroups_x_cvtshift_ranked": ranked_sg.get("x_cvtshift"),
        "x_cvtshift_spill_bytes_local":
            entry["arms"]["x_cvtshift"][LOCAL_ARCH]["0"]["spill_bytes"],
        "x_cvtshift_rejected_before_timing": True,
        "excluded_cells_local_spill": json.dumps(doc["excluded_cells"]),
        "conclusion": (
            "Per-width census hides the register pressure the scored kernel "
            "presents, because the shipped entry point inlines every width "
            "into one function. At the entry point `a_base` needs %d "
            "registers on the ranked %s and holds %d resident simdgroups, "
            "against %d and %d on the local %s. `x_cvtshift` was rejected "
            "before any timing on this evidence."
            % (entry["arms"]["a_base"][RANKED_ARCH]["0"]["registers"],
               RANKED_ARCH, ranked_sg["a_base"],
               entry["arms"]["a_base"][LOCAL_ARCH]["0"]["registers"],
               local_sg["a_base"], LOCAL_ARCH)),
    })
    run.finish()


def log_deletion() -> None:
    doc = summary()
    dele = doc["injection_vs_deletion"]
    run = start(
        job_type="deletion-price", name="e123-deletion-price",
        config={
            "question": (
                "does the price of adding one instruction equal the price of "
                "removing one, measured inside a single counterbalanced "
                "session instead of across two"
            ),
            "leg_command": (
                "research/e123_probe.sh e123-full --shapes 0,1,2,3,4 "
                "--widths 2,3,4,5 --pairs 8 --samples 24"
            ),
            "prereg_ratio_point": dele["prediction"]["point"],
            "prereg_ratio_band": dele["prediction"]["band"],
            "bytes_per_instruction": 8.25,
            **identity(),
            **gate_flags("standalone Metal microbenchmark, one GPU", True),
        },
    )

    tbl = wandb.Table(columns=[
        "na", "injection_price", "deletion_price", "ratio",
        "n_halfsums_free_cost_pct", "n_nosums_cost_pct",
        "deleted_instructions_from_text", "deleted_instructions_from_source"])
    for na, r in sorted(dele["per_width"].items()):
        tbl.add_data(int(na), r.get("injection_price"),
                     r.get("deletion_price"), r.get("ratio"),
                     r.get("n_halfsums_free_cost_pct"),
                     r.get("n_nosums_cost_pct"),
                     r.get("deleted_instructions_from_text"),
                     r.get("deleted_instructions_from_source"))
    run.log({"injection_against_deletion_by_width": tbl})

    lad = wandb.Table(columns=[
        "contrast", "na", "deleted_instructions_from_text", "saving_pct",
        "deletion_price", "injection_price", "ratio", "preregistered"])
    for name, per in doc["deletion_ladder"].items():
        for na, cell in sorted(per.items()):
            lad.add_data(name, int(na),
                         cell["deleted_instructions_from_text"],
                         cell["saving_pct"], cell["deletion_price"],
                         cell["injection_price"], cell["ratio"],
                         name == "second_half")
    run.log({"deletion_price_by_contrast": lad})

    repro = wandb.Table(columns=[
        "class", "e118_pct_per_instruction", "e123_pct_per_instruction",
        "drift_pct"])
    for klass, cell in doc["e118_reproduction"].items():
        repro.add_data(klass, cell["e118_pct_per_instruction"],
                       cell["e123_pct_per_instruction"], cell["drift_pct"])
    run.log({"e118_price_reproduction_na4": repro})

    calib = wandb.Table(columns=[
        "na", "deleted_instructions", "census_predicted_saving_pct",
        "measured_saving_pct", "over_prediction_factor"])
    for na, cell in sorted(doc["census_calibration"].items()):
        calib.add_data(int(na), cell["deleted_instructions"],
                       cell["census_predicted_saving_pct"],
                       cell["measured_saving_pct"],
                       cell["over_prediction_factor"])
    run.log({"census_against_a_measured_deletion": calib})

    lad4 = {n: doc["deletion_ladder"][n]["4"] for n in doc["deletion_ladder"]}
    run.summary.update({
        "injection_over_deletion_ratio_na4": dele["ratio_na4"],
        "ratio_na4_first_half_contrast": lad4["first_half"]["ratio"],
        "ratio_na4_whole_tree_contrast": lad4["whole_tree"]["ratio"],
        "deletion_price_na4_first_half": lad4["first_half"]["deletion_price"],
        "deletion_price_na4_second_half": lad4["second_half"]["deletion_price"],
        "deletion_price_na4_whole_tree": lad4["whole_tree"]["deletion_price"],
        "e118_p_alu_drift_pct": doc["e118_reproduction"]["alu"]["drift_pct"],
        "e118_p_ld_drift_pct": doc["e118_reproduction"]["ld"]["drift_pct"],
        "e118_p_shuf_drift_pct": doc["e118_reproduction"]["shuf"]["drift_pct"],
        "census_over_prediction_factor_na4":
            doc["census_calibration"]["4"]["over_prediction_factor"],
        "ratio_inside_prereg_band": dele["prediction"]["inside_band"],
        "prereg_ratio_point": dele["prediction"]["point"],
        "alphonse_cross_session_ratio": dele["alphonse_cross_session_ratio"],
        "rung1_deletion_price_divisor": dele["rung1_divisor"],
        "deletion_verdict": dele["verdict"],
        **_gates_summary(doc),
    })
    run.finish()


def log_rung1() -> None:
    doc = summary()
    audit = doc["rung1"]
    run = start(
        job_type="priced-census", name="e123-rung1-audit",
        config={
            "question": (
                "priced against the completed ladder, which instruction group "
                "in the wide qmv inner loop is worth deleting, and is any "
                "bit-exact deletion worth more than 1.0 percent round "
                "weighted"
            ),
            "round_weights": {"2": 0.024, "3": 0.275, "4": 0.667, "5": 0.034},
            "rung2_bar_pct": RUNG2_BAR_PCT,
            "reconstruction_band": audit["reconstruction_band"],
            **identity(),
            **gate_flags("priced census, derived from the timed ladder",
                         False),
        },
    )

    tbl = wandb.Table(columns=[
        "group", "ladder_class", "deletability", "count_na4", "price_na4",
        "pct_na2", "pct_na3", "pct_na4", "pct_na5", "round_weighted_pct",
        "round_weighted_pct_deletion_priced", "note"])
    for g in audit["groups"]:
        # `summary.json` is the only input, so every width key is a string.
        per = g["per_width"]
        tbl.add_data(g["group"], g["class"], g["deletable"],
                     per["4"]["count"], per["4"]["price"],
                     *[per[str(na)]["pct"] for na in (2, 3, 4, 5)],
                     g["round_weighted_pct"],
                     g["round_weighted_pct_deletion_priced"], g["note"])
    run.log({"priced_instruction_census": tbl})

    recon = wandb.Table(columns=["na", "total_pct_of_a_base", "ratio",
                                 "unpriced_groups"])
    for na, r in sorted(audit["reconstruction"].items()):
        recon.add_data(int(na), r["total_pct_of_a_base"], r["ratio"],
                       ", ".join(r["unpriced_groups"]))
    run.log({"census_reconstruction": recon})

    rank = wandb.Table(columns=["rank", "group", "round_weighted_pct",
                                "round_weighted_pct_deletion_priced"])
    for i, row in enumerate(audit["bit_exact_ranking"], start=1):
        rank.add_data(i, row["group"], row["round_weighted_pct"],
                      row["round_weighted_pct_deletion_priced"])
    run.log({"bit_exact_deletion_ranking": rank})

    key = "e123_largest_predicted_bit_exact_deletion_pct_round_weighted"
    run.summary.update({
        key: audit[key],
        "rung2_bar_pct": audit["rung2_bar_pct"],
        "rung2_justified": audit["rung2_justified"],
        "deletion_price_divisor": audit["deletion_price_divisor"],
        "reconstruction_ratio_na4": audit["reconstruction"]["4"]["ratio"],
        **_gates_summary(doc),
    })
    run.finish()


RUNS = {
    "e123-ladder": log_ladder,
    "e123-static-budget": log_static,
    "e123-deletion-price": log_deletion,
    "e123-rung1-audit": log_rung1,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(RUNS))
    args = ap.parse_args()
    for name, fn in RUNS.items():
        if args.only and name != args.only:
            continue
        print("==", name)
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
