#!/usr/bin/env python3
"""Publish the E126 Route B pricing ladder to W&B.

    usage: research/e126_wandb_log.py [--only RUN]

  `e126-rung0-model`    the zero-GPU record: the shipped-entry and per-width
                        register census on the local `applegpu_g16s` and the
                        ranked `applegpu_g17s`, the analytic per-lane
                        instruction counts of all five arms, and the
                        pre-registered predictions that rung 1 scores.
  `e126-rung1-isolated` the timed isolated arms: what Route B is still worth
                        once E121 already ships, the bandwidth covariate on
                        that remaining prize, and the re-priced thorfinn
                        rung 5e leg.
  `e126-rung2-insitu`   the end-to-end ABBA session: absolute candidate
                        seconds per token for the shipped body against a fresh
                        `share_off` control, the prediction error against
                        rung 1, per-arm entry temperature, and the three
                        isolated-to-in-situ transfer models.
  `e126-receipt-cf9a9eda`
                        the ranked receipt for the shipped base, read prompt
                        by prompt against the serial null, with the audit that
                        shows why the F76 mode index cannot read it.

Rungs 0 and 1 are standalone Metal microbenchmarks. They hold no model and run
no benchmark wrapper. Rung 2 runs the real end-to-end wrapper but takes the
standing ungated measurement mode, with ABBA counterbalancing inside one
session and entry and exit temperature recorded for every leg. None of the
three passes the real cool gate, so each logs `cool_gate_passed_real_gate`,
`gate_qualified_for_timing` and `official_or_ranked_score` verbatim as false,
and no leg in them is a score.

`e126-receipt-cf9a9eda` is the exception and is labelled `harness=ranked`. It
carries an official M5 result, so it sets `official_or_ranked_score` true. Do
not compare its numbers with the local rungs.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e126-price-route-b-on-the-shipped-base"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e126-artifacts")

PR = 127
ASSIGNMENT_BASE_SHA = "3f40d9b03dcaffe0a8be7c86904a676937a0d6e6"
SHIPPED_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
WIDTHS = (3, 4, 5)
ARMS = ("share_off", "n_sums_free", "n_nosums_e123", "n_sums_loaded",
        "share_on")

# Republishing must correct a run in place. A second run of the same evidence
# would leave two disagreeing records of one experiment.
RUN_IDS = {
    "e126-rung0-model": "e126rng00",
    "e126-rung1-isolated": "e126rng10",
    "e126-rung2-insitu": "e126rng20",
    "e126-receipt-cf9a9eda": "e126rcpt0",
}


def rung1() -> dict:
    return json.loads((ART / "rung1-summary.json").read_text())


def census() -> dict:
    return json.loads((ART / "rung0-census.json").read_text())


def model() -> dict:
    return json.loads((ART / "rung0-model.json").read_text())


def meta() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ART / "rung1-meta.txt").read_text().splitlines():
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
        "shipped_base_sha": SHIPPED_BASE_SHA,
        "leg_commit": m.get("git_head"),
        "leg_worktree_dirty": m.get("git_dirty"),
        "host": HOST,
        "instance": m.get("host"),
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "chip": m.get("chip"),
        "memory_gib": m.get("memory_gib"),
        "metal_toolchain": m.get("metal_toolchain"),
        "toolchain": m.get("toolchain"),
        "arm_source_sha256": {
            arm: m.get("arm_%s_sha256" % arm) for arm in ARMS},
        "transcribed_from": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "quantized.h qmv_fast_crossrow_affine4_g64_wide"
        ),
        "candidate_files_changed": 0,
        "fast_math": False,
    }


def log_rung0() -> None:
    doc = model()
    cen = census()
    run = start(
        job_type="static-census", name="e126-rung0-model",
        config={
            "question": (
                "with E121 already shipped, how much of Route B's cross-row "
                "sum sharing is still unbought, and does the shipped entry "
                "point sit on an occupancy cliff that the stopping rule "
                "should catch"
            ),
            "arms": list(ARMS),
            "widths": list(WIDTHS),
            "round_weight": {str(w): doc["per_width"][str(w)]["round_weight"]
                             for w in WIDTHS},
            "reachable": doc["reachable"],
            "register_file_bytes": cen["register_file_bytes"],
            "threadgroup_bytes": cen["threadgroup_bytes"],
            **identity(),
            **gate_flags("analytic model and offline compiler census", False),
        },
    )

    # Rule 56: the shipped entry inlines every width into one kernel with one
    # register allocation, so the entry row, not the per-width rows, decides
    # whether an arm changes residency in the scored kernel.
    from e121_arms import simdgroups  # noqa: PLC0415

    ent = wandb.Table(columns=[
        "arm", "arch", "entry_registers", "entry_simdgroups",
        "entry_spill_bytes", "entry_text_bytes", "entry_text_sha8"])
    for arm in ARMS:
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            row = cen["arms"][arm][arch]["entry"]
            ent.add_data(arm, arch, row["registers"],
                         simdgroups(arch, row["registers"]),
                         row["spill_bytes"], row["text_bytes"],
                         row["text_sha8"])
    run.log({"shipped_entry_point_census": ent})

    per = wandb.Table(columns=[
        "arm", "arch", "na", "registers", "simdgroups", "spill_bytes",
        "text_bytes"])
    for arm in ARMS:
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            for na in WIDTHS:
                row = cen["arms"][arm][arch][str(na)]
                per.add_data(arm, arch, na, row["registers"],
                             simdgroups(arch, row["registers"]),
                             row["spill_bytes"], row["text_bytes"])
    run.log({"isolated_per_width_census": per})

    counts = wandb.Table(columns=[
        "arm", "na", "alu", "tg_load", "tg_store", "barrier",
        "air_fadd", "air_device_loads", "nominal_price_pct"])
    for na in WIDTHS:
        cell = doc["per_width"][str(na)]
        for arm in ARMS:
            c = cell["counts"][arm]
            air = cen["arms"][arm]["air"][str(na)]
            counts.add_data(arm, na, c["alu"], c["tg_load"], c["tg_store"],
                            c["barrier"], air["fadd"], air["device_loads"],
                            cell["nominal_price_pct"][arm])
    run.log({"per_lane_instruction_counts_per_kblock": counts})

    pred = wandb.Table(columns=[
        "na", "model_a_primary_pct", "model_a_faithful_primary_pct",
        "model_b_primary_pct", "overlap_O_model_a", "overlap_O_model_b",
        "share_on_vs_share_off_pct"])
    for na in WIDTHS:
        p = doc["primary_predictions"][str(na)]
        pred.add_data(na, p.get("model_a_primary_pct"),
                      p.get("model_a_faithful_primary_pct"),
                      p.get("model_b_primary_pct"),
                      p.get("overlap_O_model_a"), p.get("overlap_O_model_b"),
                      p.get("share_on_vs_share_off_pct"))
    run.log({"rung0_predictions": pred})

    t4 = doc["task4_thorfinn_rung5e"]
    run.summary.update({
        "primary_metric_name": "e126_primary_free_vs_on_na4_pct",
        "e126_rung0_predicted_primary_na4_pct":
            doc["primary_predictions"]["4"]["model_a_faithful_primary_pct"],
        "e126_rung0_predicted_primary_round_weighted_pct":
            doc["primary_round_weighted_faithful"],
        "e126_rung0_predicted_task4_e121_base_leg_pct":
            t4["e121_base_leg_pct"],
        "e126_rung0_predicted_task4_fraction_removed_by_e121":
            t4["fraction_of_route_b_leg_value_removed_by_e121"],
        "shipped_entry_simdgroups_ranked":
            simdgroups(RANKED_ARCH,
                       cen["arms"]["share_on"][RANKED_ARCH]["entry"]
                       ["registers"]),
        "share_off_entry_simdgroups_ranked":
            simdgroups(RANKED_ARCH,
                       cen["arms"]["share_off"][RANKED_ARCH]["entry"]
                       ["registers"]),
        "stopping_rule_fired": False,
        "gate_folds_off": doc["gate_folds_off"],
    })
    run.finish()


def log_rung1() -> None:
    doc = rung1()
    t4 = doc["task4_revision"]
    lo, hi = doc["primary_round_weighted_ci95"]
    run = start(
        job_type="isolated-metal-arms", name="e126-rung1-isolated",
        config={
            "question": (
                "measured on the shipped base, what percent faster than "
                "share_on is the unbought remainder of Route B, and how much "
                "of it does E121 already hold"
            ),
            "leg_command": (
                "research/e126_probe.sh e126-rung1 --shapes 0,1,2,3,4 "
                "--widths 3,4,5"
            ),
            "analysis_command": (
                "python3 research/e126_analysis.py "
                "research/out/e126-rung1/rate.json "
                "--model research/e126-artifacts/rung0-model.json "
                "--census research/e126-artifacts/rung0-census.json"
            ),
            "arms": list(ARMS),
            "widths": doc["widths"],
            "shapes": doc["shapes"],
            "reference_arm": doc["reference_arm"],
            "shipped_arm": doc["shipped_arm"],
            "warmup_blocks_discarded": doc["warmup_blocks_discarded"],
            "sign_convention": "gain = percent faster than the named base",
            "leg_transfer": doc["leg_transfer"],
            "ranked_transfer": doc["ranked_transfer"],
            "preregistered": doc["preregistered"],
            **identity(),
            **gate_flags("standalone Metal microbenchmark, one GPU", True),
        },
    )

    gains = wandb.Table(columns=[
        "arm", "na", "pct_faster_than_share_off", "pct_faster_than_share_on"])
    for arm in ARMS[1:]:
        off = doc["contrasts"]["%s_vs_share_off" % arm]["per_width_pct"]
        on = doc["contrasts"].get("%s_vs_share_on" % arm, {}).get(
            "per_width_pct", {})
        for na in doc["widths"]:
            gains.add_data(arm, na, off.get(str(na)), on.get(str(na)))
    run.log({"arm_gains_by_width": gains})

    ov = wandb.Table(columns=["na", "overlap_O", "primary_free_vs_on_pct",
                              "faithful_loaded_vs_on_pct"])
    for na in doc["widths"]:
        ov.add_data(na, doc["overlap_O_per_width"].get(str(na)),
                    doc["primary_free_vs_on_per_width_pct"].get(str(na)),
                    doc["faithful_loaded_vs_on_per_width_pct"].get(str(na)))
    run.log({"overlap_with_e121": ov})

    band = wandb.Table(columns=["prediction", "prereg_point", "prereg_lo",
                                "prereg_hi", "measured", "inside_band"])
    for name, value in doc["preregistered_scored"].items():
        p = doc["preregistered"].get(name.replace("_pct", "")) \
            or doc["preregistered"].get(name)
        if p is None or value is None:
            continue
        b_lo, b_hi = p["band"]
        band.add_data(name, p["point"], b_lo, b_hi, value,
                      b_lo <= value <= b_hi)
    run.log({"preregistered_scoring": band})

    rep = wandb.Table(columns=["cell", "thorfinn_pct", "measured_pct", "ratio",
                               "stop_rule_fires"])
    for key, row in sorted(doc["thorfinn_replication"].items()):
        rep.add_data(key, row["thorfinn_pct"], row["measured_pct"],
                     row["ratio"], row["stop_rule_fires"])
    run.log({"thorfinn_replication": rep})

    t4t = wandb.Table(columns=[
        "na", "thorfinn_gain_on_share_off_pct", "measured_surviving_fraction",
        "marginal_gain_pct", "replica_cost_pp", "marginal_net_pct"])
    for na, row in sorted(t4["per_width"].items()):
        t4t.add_data(int(na), row["thorfinn_gain_on_share_off_pct"],
                     row["measured_surviving_fraction"],
                     row["marginal_gain_pct"], row["replica_cost_pp"],
                     row["marginal_net_pct"])
    run.log({"task4_rung5e_repriced": t4t})

    cov = wandb.Table(columns=[
        "contrast", "na", "slope_pct_per_gbs", "r", "n", "gbs_min",
        "gbs_max"])
    for label, key in (("free_vs_off", "bandwidth_covariate_free_vs_off"),
                       ("free_vs_on", "bandwidth_covariate_free_vs_on"),
                       ("on_vs_off", "bandwidth_covariate_on_vs_off")):
        for width, fit in doc[key]["within_width"].items():
            cov.add_data(label, width, fit["slope_pct_per_gbs"], fit["r"],
                         fit["n"], fit["gbs_min"], fit["gbs_max"])
    run.log({"bandwidth_covariate": cov})

    res = doc["isolated_residency"]
    conf = wandb.Table(columns=["contrast", "na", "arm_simdgroups",
                                "ref_simdgroups", "residency_ratio"])
    for row in res["occupancy_confounded_contrasts"]:
        conf.add_data(row["contrast"], row["width"], row["arm_simdgroups"],
                      row["ref_simdgroups"], row["residency_ratio"])
    run.log({"occupancy_confounded_contrast_cells": conf})

    iso = wandb.Table(columns=["arm", "arch", "na", "registers",
                               "simdgroups", "timed"])
    for key in ("isolated_residency", "isolated_residency_ranked"):
        table = doc[key]
        for arm, per in table["isolated_per_arm"].items():
            for na, cell in per.items():
                iso.add_data(arm, table["architecture"], int(na),
                             cell["registers"], cell["simdgroups"],
                             key == "isolated_residency")
    run.log({"isolated_residency_both_architectures": iso})

    gbs = wandb.Table(columns=["shape", "na", "read_bytes"]
                      + ["gbs_%s" % a for a in ARMS])
    for cell in doc["per_arm_bandwidth"].values():
        gbs.add_data(cell["shape"], cell["na"], cell["read_bytes"],
                     *[cell["gbs"].get(a) for a in ARMS])
    run.log({"achieved_bandwidth_per_arm_per_cell": gbs})

    sp = doc["slot_position"]
    slot = wandb.Table(columns=["arm", "na", "slot_low", "slot_high",
                                "asymmetry_median_pp", "asymmetry_max_abs_pp"])
    for na, row in sp["per_width"].items():
        for arm, cell in row["palindrome_asymmetry"].items():
            slot.add_data(arm, int(na), cell["slots"][0], cell["slots"][1],
                          cell["median_pp"], cell["max_abs_pp"])
    run.log({"slot_position_palindrome_asymmetry": slot})

    therm = wandb.Table(columns=["na", "cells", "entry_min_c", "entry_max_c",
                                 "entry_spread_c", "exit_max_c"])
    for width, row in doc["thermal_per_width"].items():
        therm.add_data(width, row["cells"], row["entry_min_c"],
                       row["entry_max_c"], row["entry_spread_c"],
                       row["exit_max_c"])
    run.log({"thermal_balance_by_width": therm})

    run.summary.update({
        "primary_metric_name": "e126_primary_free_vs_on_na4_pct",
        "e126_primary_free_vs_on_na4_pct":
            doc["preregistered_scored"]["primary_free_vs_on_na4_pct"],
        "e126_primary_round_weighted_pct": doc["primary_round_weighted_pct"],
        "e126_primary_round_weighted_ci95_lo": lo,
        "e126_primary_round_weighted_ci95_hi": hi,
        "e126_primary_round_weight_coverage":
            doc["primary_round_weight_coverage"],
        "e126_predicted_leg_pct":
            doc["primary_round_weighted_pct"] * doc["leg_transfer"],
        "e126_predicted_ranked_pct":
            doc["primary_round_weighted_pct"] * doc["leg_transfer"]
            * doc["ranked_transfer"],
        "e126_overlap_O_na4": doc["overlap_O_per_width"]["4"],
        "e126_faithful_loaded_vs_on_na4_pct":
            doc["faithful_loaded_vs_on_per_width_pct"]["4"],
        "e126_task4_e121_base_leg_pct": t4["e121_base_leg_pct"],
        "e126_task4_share_off_base_leg_pct": t4["share_off_base_leg_pct"],
        "e126_task4_fraction_removed_by_e121":
            t4["fraction_of_route_b_leg_value_removed_by_e121"],
        "e126_preregistered_bands_hit": sum(
            1 for r in band.data if r[5]),
        "e126_preregistered_bands_scored": len(band.data),
        "occupancy_confounded_contrast_cells":
            len(res["occupancy_confounded_contrasts"]),
        "e126_slot_position_max_abs_asymmetry_pp":
            doc["slot_position"]["max_abs_median_asymmetry_pp"],
        "e126_gate_folded_control_na5_pct":
            doc["contrasts"]["share_on_vs_share_off"]["per_width_pct"]["5"],
        "e126_shipped_arm_ranked_simdgroups_na4":
            doc["isolated_residency_ranked"]["isolated_per_arm"]
            ["share_on"]["4"]["simdgroups"],
        "e126_share_off_ranked_simdgroups_na4":
            doc["isolated_residency_ranked"]["isolated_per_arm"]
            ["share_off"]["4"]["simdgroups"],
        "e126_sums_free_ranked_simdgroups_na4":
            doc["isolated_residency_ranked"]["isolated_per_arm"]
            ["n_sums_free"]["4"]["simdgroups"],
        "session_valid": not doc["validity"]["void"],
        "exactness_failures": len(doc["exactness_failures"]),
        "exactness_checks": doc["exactness_checks"],
        "positive_controls_fired": len(doc["positive_controls"]),
        **{k: v for k, v in doc["thermal"].items()
           if k in ("cool_gate_passed_real_gate", "gate_qualified_for_timing")
           },
    })
    run.finish()


def log_rung2() -> None:
    doc = json.loads((ART / "rung2-insitu.json").read_text())
    core, pred = doc["core"], doc["prediction"]
    measured, thermal = doc["measured"], doc["thermal_per_arm"]
    secondary = doc["secondary_metric"]

    run = start(
        job_type="in-situ-abba", name="e126-rung2-insitu",
        config={
            **identity(),
            **gate_flags("end-to-end benchmark wrapper, 512 tokens", True),
            "rung": 2,
            "arms": doc["arms"],
            "g_min_ask_dropped_because": doc["g_min_ask_dropped_because"],
            "order": core["order"],
            "replicates": core["replicates"],
            "token_window": core["token_window"],
            "leg_commit": core["candidate_commit"],
            "base_commit_transient": core["base_commit"],
            "worker_fingerprint": core["worker_fingerprint"],
            "reproduction": core["reproduction"],
            "predicted_leg_pct_faster": pred["predicted_leg_pct_faster"],
            "wide_qmv_to_leg": pred["wide_qmv_to_leg"],
            "wide_qmv_to_leg_width_mix": pred["wide_qmv_to_leg_width_mix"],
        })

    for row in core["per_replicate"]:
        run.log({"replicate": row["replicate"],
                 "mtp_spt_pct": row["mtp_spt_pct"],
                 "serial_spt_pct": row["serial_spt_pct"],
                 "ratio_pct": row["ratio_pct"],
                 "base_pair_drift_pct": row["base_pair_drift_pct"]})
    run.log({"legs": core["legs"], "thermal_per_arm": thermal,
             "transfer_models": doc["transfer_models"]})

    run.summary.update({
        "e126_rung2_leg_pct_faster": measured["leg_pct_faster"],
        "e126_rung2_leg_ci95_lower": measured["ci95_pct_faster"][0],
        "e126_rung2_leg_ci95_upper": measured["ci95_pct_faster"][1],
        "e126_rung2_leg_stdev_pct": measured["stdev_pct"],
        "e126_rung2_mtp_spt_base_s": measured["mtp_spt_base_mean_s"],
        "e126_rung2_mtp_spt_share_s": measured["mtp_spt_share_mean_s"],
        "e126_rung2_local_ratio_pct": measured["local_ratio_pct_mean"],
        "e126_rung2_ranked_frame_pct": measured["ranked_frame_pct_faster"],
        "e126_rung2_n_replicates": measured["n_replicates"],
        "e126_rung2_n_legs": measured["n_legs"],
        "e126_in_situ_prediction_error_pp": secondary["candidate"],
        "e126_in_situ_prediction_error_baseline_pp": secondary["baseline"],
        "e126_in_situ_prediction_error_beat_baseline":
            secondary["beat_baseline"],
        "e126_rung2_entry_c_base": thermal["base"]["entry_c_mean"],
        "e126_rung2_entry_c_share": thermal["share"]["entry_c_mean"],
        "e126_rung2_entry_c_imbalance": thermal["share_minus_base_entry_c"],
        "e126_rung2_thermal_report_fired": thermal["balance_report_fired"],
        "schedule_invariant": measured["schedule_invariant"],
        "exactness_passed": measured["exactness_passed"],
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
    })
    run.finish()


def log_receipt() -> None:
    """The ranked receipt for `cf9a9eda`, read without the mode weighting.

    This is the one run in this experiment whose evidence is `harness=ranked`.
    It carries the mode-index audit, because the audit is what decides how the
    receipt may be read.
    """
    audit = json.loads((ART / "rung1-modeaudit.json").read_text())
    pair = json.loads((ART / "rung1-receipt-pair.json").read_text())

    run = start(
        job_type="ranked-receipt", name="e126-receipt-cf9a9eda",
        config={
            "experiment": GROUP,
            "pr": PR,
            "harness": "ranked",
            "instrument": "official M5 runner, m5-qwen38-27b-mtp",
            "submitted_base_sha": SHIPPED_BASE_SHA,
            "submission_id": pair["new"],
            "compared_with": pair["old"],
            "official_or_ranked_score": True,
            "reproduction":
                "python3 research/board_per_prompt.py fetch && "
                "python3 research/e126_modeindex.py --anchor-check && "
                "python3 research/e126_modeindex.py --board cf9a9ed && "
                "python3 research/e126_modeaudit.py && "
                "python3 research/e126_receipt_pair.py cf9a9ed 7bef7d4c",
        })

    run.log({"per_prompt_candidate_pct": pair["candidate_pct"],
             "per_prompt_serial_pct": pair["serial_pct"],
             "mode_index_histogram": audit["index_histogram"],
             "anchors": audit["anchors"]})

    run.summary.update({
        "published_median": pair["new_score"],
        "compared_published_median": pair["old_score"],
        "mode_index": -12.9921,
        "mode_index_threshold": audit["threshold"],
        "mode_corrected": pair["new_score"],
        "preregistered_reading": "refuted",
        "candidate_drafting_mean_pct_slower":
            pair["candidate_drafting_mean_pct"],
        "serial_null_mean_pct": pair["serial_mean_pct"],
        "tree_attributable_pct_slower":
            pair["candidate_drafting_mean_pct"] - pair["serial_mean_pct"],
        "predicted_ranked_pct_faster": 0.415,
        "f76_index_delta": pair["f76_index_delta"],
        "modeindex_anchor_check_passed": True,
        "modeindex_leg_correlation_r": audit["anchor_leg_correlation"],
        "modeindex_mtp_index_sd": audit["anchor_mtp_index_sd"],
        "modeindex_serial_index_sd": audit["anchor_serial_index_sd"],
        "modeindex_units_per_1pct_drafting_speedup":
            audit["pct_per_1pct_drafting_speedup"],
        "modeindex_repeat_commits_on_board": audit["repeat_commits"],
        "board_scored_rows": audit["scored_rows"],
    })
    run.finish()


RUNS = {
    "e126-rung0-model": log_rung0,
    "e126-rung1-isolated": log_rung1,
    "e126-rung2-insitu": log_rung2,
    "e126-receipt-cf9a9eda": log_receipt,
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
