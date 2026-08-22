#!/usr/bin/env python3
"""Publish the E125 frame axis and the isolated-to-in-situ correction to W&B.

    usage: research/e125_wandb_log.py [--only RUN] [--dry]

  `e125-frame-axis`       the per-instruction price of one device-load and one
                          ALU instruction, and the price of one real deletion,
                          measured in six memory frames that share the compiled
                          pipeline state, the width and the launched grid.
  `e125-frame-axis-sync`  the same axis for the synchronisation classes: the
                          threadgroup scaffold, threadgroup-memory reads and
                          barriers alone. This is the mechanism class the E121
                          anchor actually used.
  `e125-correction`       the class-by-regime transfer table, the anchor
                          validation test, and the Route B ranked band.

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
GROUP = "e125-isolated-to-in-situ-transfer-law"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e125-artifacts")

# One entry per measured session. `law` is the analysis artifact and `out` is
# the raw session directory whose meta.txt carries the identity tuple.
SESSIONS = {
    "work": {
        "run_name": "e125-frame-axis",
        "law": "frame-law.json",
        "out": pathlib.Path("research/out/e125-full"),
        "question":
            "does the per-instruction price of one device load and one ALU "
            "instruction depend on the memory frame the same compiled kernel "
            "runs in, and does one function of achieved bandwidth explain "
            "both directions",
        "command":
            "research/e125_probe.sh e125-full --shapes 0,3 --widths 2,3,4 "
            "--frames base,cycle,consumer,k1024,k2048,k4096 --pairs 8 "
            "--warm-pairs 1 --warm-sweep-reps 2 --samples 24 --target-ms 40 "
            "--inner-max 2048 --consumer-mb 512 --cycle 4",
    },
    "sync": {
        "run_name": "e125-frame-axis-sync",
        "law": "frame-law-sync.json",
        "out": pathlib.Path("research/out/e125-sync"),
        "question":
            "does a threadgroup exchange and the barriers that make it legal "
            "respond to the memory frame the same way ordinary work does, or "
            "does the synchronisation class transfer differently",
        "command":
            "E125_ARM_SET=sync research/e125_probe.sh e125-sync "
            "--shapes 0,2,3 --widths 2,3,4 --frames base,consumer,k1024,k4096 "
            "--pairs 8 --warm-pairs 1 --warm-sweep-reps 2 --samples 24 "
            "--target-ms 40 --inner-max 2048 --consumer-mb 512",
    },
}

PR = 126
ASSIGNMENT_BASE_SHA = "3b8ea425f8887c9b5cd08ddfff6ddc423fb5d9c3"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
PEAK_BANDWIDTH_GB_S = 273.0

# Republishing must correct a run in place. A second run of the same evidence
# would leave two disagreeing records of one experiment.
RUN_IDS = {
    "e125-frame-axis": "e125frame1",
    "e125-frame-axis-sync": "e125sync01",
    "e125-register-census": "e125cens01",
    "e125-residency-law": "e125resl01",
    "e125-correction": "e125corr01",
}

# The keys a reader of the report arrives looking for, per run id. Consumed by
# `research/e118_wandb_check.py --experiment e125 --verify`.
EXPECTED_SUMMARY_KEYS = {
    "e125frame1": (
        "primary_metric_name",
        "e125_ld_ratio_k1024_over_base",
        "e125_ld_ratio_consumer_over_base",
        "e125_ld_transfer_k1024",
        "e125_alu_ratio_k1024_over_base",
        "e125_alu_ratio_consumer_over_base",
        "e125_alu_transfer_k1024",
        "e125_deletion_ratio_k1024_over_base",
        "e125_deletion_transfer_k1024",
        "e125_worst_ramp_residual_pct",
        "e125_entry_temp_spread_c",
        "session_valid",
        "cool_gate_passed_real_gate",
        "gate_qualified_for_timing",
        "official_or_ranked_score",
    ),
    "e125sync01": (
        "primary_metric_name",
        "e125_tg_scaffold_ratio_consumer_over_base",
        "e125_bar_ratio_consumer_over_base",
        "e125_bar_ratio_k1024_over_base",
        "e125_worst_ramp_residual_pct",
        "e125_entry_temp_spread_c",
        "session_valid",
        "cool_gate_passed_real_gate",
        "gate_qualified_for_timing",
        "official_or_ranked_score",
    ),
    "e125cens01": (
        "primary_metric_name",
        "e125_entry_registers_local_pre_e121",
        "e125_entry_registers_local_share_on",
        "e125_entry_registers_ranked_pre_e121",
        "e125_entry_registers_ranked_share_on",
        "e125_entry_resident_change_pct_local",
        "e125_entry_resident_change_pct_ranked",
        "e125_alphonse_a_base_cells_agree",
        "e125_inlining_law_holds",
        "e125_allocation_register_cost",
        "e125_parameter_register_cost",
        "e125_ranked_registers_to_gain_one_simdgroup",
        "cool_gate_passed_real_gate",
        "gate_qualified_for_timing",
        "official_or_ranked_score",
    ),
    "e125resl01": (
        "primary_metric_name",
        "e125_two_channel_coefficient_entry_point",
        "e125_two_channel_coefficient_isolated_bodies",
        "e125_two_channel_instruction_pct_isolated_bodies",
        "e125_two_channel_instruction_sign_ok",
        "e125_f5_bracket_consistent_as_registered",
        "e125_f5_bracket_consistent_in_leg_frame",
        "e125_two_channel_anchors_tested",
        "e125_two_channel_anchors_passed",
        "e125_r_gain_bandwidth_pooled",
        "e125_r_gain_residency_pooled",
        "e125_partial_r_gain_bandwidth_given_residency",
        "e125_r_gain_bandwidth_width_demeaned",
        "e125_r_gain_deleted_instructions_pooled",
        "e125_r2_instructions_only",
        "e125_r2_gain_from_bandwidth",
        "e125_r2_gain_from_residency",
        "e125_bandwidth_survives_residency_control",
        "e125_ladder_contrasts_pure_instruction_channel",
        "e125_ladder_contrasts_total",
        "cool_gate_passed_real_gate",
        "gate_qualified_for_timing",
        "official_or_ranked_score",
    ),
    "e125corr01": (
        "e125_correction_form",
        "e125_anchors_reproduced",
        "e125_anchor_verdict",
        "e125_single_regime_explains_both",
        "e125_variance_explained_regime_only",
        "e125_variance_explained_class_by_regime",
        "e125_route_b_parity_line_pct",
        "e125_route_b_mode_proof_line_pct",
        "e116_share_term_untouched",
        "e125_arch_anchor_factor",
        "e125_arch_sign_reproduced",
        "e125_arch_magnitude_reproduced",
        "e125_arch_predicted_factor_entry_point",
        "e125_session_valid_after_exclusions",
        "e125_class_widths_lost_to_fidelity",
    ),
}


def law(name: str = "frame-law.json") -> dict:
    return json.loads((ART / name).read_text())


def prereg() -> dict:
    return json.loads((ART / "routeb-prediction.json").read_text())


def meta(out: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (out / "meta.txt").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            result[key] = value
    return result


def gate_flags(instrument: str, timing_valid: bool) -> dict[str, object]:
    return {
        "timing_valid": timing_valid,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "instrument": instrument,
    }


def identity(out: pathlib.Path = SESSIONS["work"]["out"]) -> dict[str, object]:
    m = meta(out)
    return {
        "experiment": GROUP,
        "pr": PR,
        "assignment_base_sha": ASSIGNMENT_BASE_SHA,
        "leg_commit": m.get("git_head"),
        "arm_twin_rev": m.get("arm_twin_rev"),
        "arm_twin_sha": m.get("arm_twin_sha"),
        "host": HOST,
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "chip": m.get("chip"),
        "memory_gib": m.get("memory_gib"),
        "toolchain": m.get("toolchain"),
        "metal_toolchain": m.get("metal_toolchain"),
        "entry_points": "e118_iso_na2..na5",
        "arm_set": m.get("arm_set", "work"),
        "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
        "candidate_files_changed": 0,
        "fast_math": False,
        "preregistration": "research/e125-stage0-prereg.md",
        "arm_sources_byte_identical_to_e123": True,
    }


def start(job_type: str, name: str, config: dict[str, object]):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name, id=RUN_IDS.get(name), resume="allow", config=config,
        reinit=True)


def ratio(doc: dict, klass: str, frame: str) -> float | None:
    by_frame = (doc["frame_law"].get(klass) or {}).get("ratio_by_frame") or {}
    cell = by_frame.get(frame)
    return None if cell is None else cell["median"]


def log_frame_axis(dry: bool = False, session: str = "work") -> None:
    spec = SESSIONS[session]
    doc = law(spec["law"])
    cfg = {
        "question": spec["question"],
        "leg_command": spec["command"],
        "session": session,
        "frames": [f["name"] for f in doc["frames"]],
        "widths": sorted({c["m"] for c in doc["cells"]}),
        "arms": doc["arms"],
        "pairs": doc["pairs"],
        "warm_pairs": doc["warm_pairs"],
        "warm_sweep_reps": doc["warm_sweep_reps"],
        "headline_statistic": (
            "microseconds per injected instruction per k-block, a rung "
            "contrast that cancels the injection scaffold and the fixed "
            "prologue exactly"
        ),
        **identity(spec["out"]),
        **gate_flags("standalone Metal microbenchmark, one GPU", True),
    }
    if dry:
        print(json.dumps({"run": spec["run_name"], "config": cfg}, indent=2))
        return
    run = start(job_type="isolated-metal-arms", name=spec["run_name"],
                config=cfg)

    classes = sorted(doc["frame_law"])
    cells = wandb.Table(columns=[
        "shape", "m", "frame", "k", "k_blocks", "inner", "launched_threads",
        "read_bytes", "base_us", "achieved_gb_s", "consumer_gb_s", "phi_arm",
        "phi_eff", "entry_c_min", "entry_c_max", "entry_c_spread",
        "class", "value_us_per_k_block", "null_scaffold_pct"])
    for c in doc["cells"]:
        p = c["prices"]
        for klass in classes:
            entry = p.get(klass)
            if entry is None:
                continue
            value = entry.get("us_per_instruction_per_k_block",
                              entry.get("gain_us_per_k_block"))
            cells.add_data(
                c["shape"], c["m"], c["frame"], c["k"], c["k_blocks"],
                c["inner"], c["launched_threads"], c["read_bytes"],
                c["base_us"], c["achieved_gb_s"], c["consumer_gb_s"],
                c["phi_arm"], c["phi_eff"], c["entry_c_min"], c["entry_c_max"],
                c["entry_c_spread"], klass, value,
                (p.get("null_scaffold") or {}).get("cost_pct"))
    run.log({"cells": cells})

    seg = wandb.Table(columns=[
        "shape", "m", "segment", "k_blocks_from", "k_blocks_to",
        "bytes_per_k_block", "marginal_gb_s", "phi_marginal"])
    for s in doc["segments"]:
        seg.add_data(
            s["shape"], s["m"], "%s->%s" % (s["from_frame"], s["to_frame"]),
            s["k_blocks_from"], s["k_blocks_to"], s["bytes_per_k_block"],
            s["marginal_gb_s"], s["phi_marginal"])
    run.log({"marginal_segments": seg})

    ratios = wandb.Table(columns=[
        "class", "frame", "median_of_per_cell_ratios", "transfer_factor",
        "transfer_ci95_low", "transfer_ci95_high", "ratio_of_sums",
        "n_cells", "base_positive_frac", "measured"])
    for klass, entry in doc["frame_law"].items():
        transfer = entry.get("transfer_by_frame", {})
        for frame, v in sorted(entry["ratio_by_frame"].items()):
            t = transfer.get(frame, {})
            ci = t.get("transfer_ci95", [None, None])
            ratios.add_data(klass, frame, v["median"], t.get("transfer_factor"),
                            ci[0], ci[1], t.get("ratio_of_sums"),
                            t.get("n_cells"), t.get("base_positive_frac"),
                            v["measured"])
    run.log({"frame_ratios": ratios})

    ramp = wandb.Table(columns=["frame", "arm", "forward_minus_reverse_pct"])
    worst = 0.0
    for frame, v in doc["ramp_residual"].items():
        for arm, gap in v["forward_minus_reverse_pct_by_arm"].items():
            ramp.add_data(frame, arm, gap)
            worst = max(worst, abs(gap))
    run.log({"ramp_residual": ramp})

    bw = wandb.Table(columns=[
        "frame", "max_implied_gb_s", "streaming", "limit_gb_s", "passed",
        "exempt_reason"])
    for frame, v in sorted(doc["gates"]["bandwidth_by_frame"].items()):
        bw.add_data(frame, v["max_implied_gb_s"], v["streaming"],
                    v["limit_gb_s"], v["passed"], v["exempt_reason"])
    run.log({"bandwidth_gate_by_frame": bw})

    spread = max(c["entry_c_spread"] for c in doc["cells"])
    summary = {
        "primary_metric_name":
            "us per injected instruction per k-block, by memory frame",
        "e125_worst_ramp_residual_pct": worst,
        "e125_entry_temp_spread_c": spread,
        "session_valid": doc["gates"]["session_valid"],
        "gate_positive_controls_passed":
            doc["gates"]["positive_controls"]["passed"],
        "gate_fidelity_passed": doc["gates"]["fidelity"]["passed"],
        **gate_flags("standalone Metal microbenchmark, one GPU", True),
    }
    for klass, entry in doc["frame_law"].items():
        for frame in entry["ratio_by_frame"]:
            summary["e125_%s_ratio_%s_over_base" % (klass, frame)] = \
                ratio(doc, klass, frame)
            t = entry.get("transfer_by_frame", {}).get(frame, {})
            summary["e125_%s_transfer_%s" % (klass, frame)] = \
                t.get("transfer_factor")
    run.summary.update(summary)
    run.finish()


def log_register_census(dry: bool = False) -> None:
    """The zero-GPU register and residency census of the shipped E121 arm.

    This run holds no model, dispatches nothing and reads no clock. It runs
    the real AGX backend for both architectures through `xcrun metal-tt`, so
    it is not a timing measurement at all and cannot be compared with one.
    """
    cen = json.loads((ART / "e121-census.json").read_text())
    local, ranked = cen["arches"]
    whole = cen["ladder"]["whole_e121_change"]
    cfg = {
        "question": (
            "the shipped E121 arm SPLITS the cross-simdgroup sums and "
            "exchanges the halves; the earlier residency table censused "
            "`n_nosums`, which DELETES them. What does the real arm cost in "
            "registers, per width and at the shipped entry point, on the "
            "local and the ranked architecture"),
        "command": (
            "python3 research/e125_e121_census.py --emit /tmp/e125cen "
            "--out research/e125-artifacts/e121-census.json"),
        "arms": list(cen["arm_definitions"]),
        "arm_definitions": cen["arm_definitions"],
        "widths": cen["widths"],
        "simdgroup_budget": cen["simdgroup_budget"],
        "simdgroup_budget_is_fitted": cen["simdgroup_budget_is_fitted"],
        "f47_weights": cen["f47_weights"],
        "live_dispatch": cen["live_dispatch"],
        **identity(),
        **gate_flags("compile-only census through xcrun metal-tt, no GPU",
                     False),
    }
    if dry:
        print(json.dumps({"run": "e125-register-census", "config": cfg},
                         indent=2))
        return
    run = start(job_type="analysis", name="e125-register-census", config=cfg)

    cells = wandb.Table(columns=[
        "arm", "arch", "frame", "width", "registers", "spill_bytes",
        "text_bytes", "resident_simdgroups", "resident_is_extrapolated"])
    for arm, per_arch in cen["arms"].items():
        for arch, keys in per_arch.items():
            budget = cen["simdgroup_budget"][arch]
            measured = cen["measured_registers"][arch]
            for key, cell in sorted(keys.items()):
                r = cell["registers"]
                cells.add_data(
                    arm, arch,
                    "entry point" if key == "entry" else "isolated body",
                    0 if key == "entry" else int(key), r, cell["spill_bytes"],
                    cell["text_bytes"], budget // r if r else None,
                    r not in measured)
    run.log({"census": cells})

    ladder = wandb.Table(columns=[
        "component", "richer_arm", "poorer_arm", "arch", "entry_from",
        "entry_to", "entry_change_pct", "body_f47_weighted_change_pct"])
    for name, rec in cen["ladder"].items():
        for arch in cen["arches"]:
            v = rec[arch]
            ladder.add_data(
                name, rec["rich"], rec["poor"], arch, v["entry_from"],
                v["entry_to"],
                None if v["entry_fraction"] is None
                else 100 * v["entry_fraction"],
                None if v["body_f47_weighted_fraction"] is None
                else 100 * v["body_f47_weighted_fraction"])
    run.log({"ladder": ladder})

    rep = wandb.Table(columns=["cell", "census", "alphonse_a_base", "agree"])
    for cell, v in cen["alphonse_a_base_replication"]["cells"].items():
        rep.add_data(cell, v["census"], v["alphonse_a_base"], v["agree"])
    run.log({"alphonse_a_base_replication": rep})

    cliffs = wandb.Table(columns=[
        "arch_arm", "registers", "resident", "registers_to_lose_one",
        "registers_to_gain_one"])
    for key, v in cen["cliffs"].items():
        cliffs.add_data(key, v["registers"], v["resident"],
                        v["registers_to_lose_one"], v["registers_to_gain_one"])
    run.log({"occupancy_cliffs": cliffs})

    alloc = cen["ladder"]["threadgroup_allocation"]
    param = cen["ladder"]["parameter"]
    run.summary.update({
        "primary_metric_name":
            "resident simdgroups at the shipped entry point, per architecture",
        "e125_entry_registers_local_pre_e121":
            cen["arms"]["pre_e121"][local]["entry"]["registers"],
        "e125_entry_registers_local_share_on":
            cen["arms"]["share_on"][local]["entry"]["registers"],
        "e125_entry_registers_ranked_pre_e121":
            cen["arms"]["pre_e121"][ranked]["entry"]["registers"],
        "e125_entry_registers_ranked_share_on":
            cen["arms"]["share_on"][ranked]["entry"]["registers"],
        "e125_entry_resident_local_pre_e121": whole[local]["entry_from"],
        "e125_entry_resident_local_share_on": whole[local]["entry_to"],
        "e125_entry_resident_ranked_pre_e121": whole[ranked]["entry_from"],
        "e125_entry_resident_ranked_share_on": whole[ranked]["entry_to"],
        "e125_entry_resident_change_pct_local":
            100 * whole[local]["entry_fraction"],
        "e125_entry_resident_change_pct_ranked":
            100 * whole[ranked]["entry_fraction"],
        "e125_body_f47_change_pct_local":
            100 * whole[local]["body_f47_weighted_fraction"],
        "e125_body_f47_change_pct_ranked":
            100 * whole[ranked]["body_f47_weighted_fraction"],
        "e125_allocation_register_cost":
            alloc[local]["entry_fraction"] + alloc[ranked]["entry_fraction"],
        "e125_parameter_register_cost":
            param[local]["entry_fraction"] + param[ranked]["entry_fraction"],
        "e125_alphonse_a_base_cells_agree":
            cen["alphonse_a_base_replication"]["agree"],
        "e125_alphonse_a_base_cells_total":
            cen["alphonse_a_base_replication"]["of"],
        "e125_inlining_law_holds":
            all(v["holds"] for v in cen["inlining_law"].values()),
        "e125_isolated_na5_unchanged_both_arches":
            all(cen["f100_test"]["isolated_na5_body_unchanged"].values()),
        "e125_ranked_registers_to_gain_one_simdgroup":
            cen["cliffs"]["%s/share_on" % ranked]["registers_to_gain_one"],
        "e125_ranked_registers_to_lose_one_simdgroup_pre_e121":
            cen["cliffs"]["%s/pre_e121" % ranked]["registers_to_lose_one"],
        "e125_any_spill": any(
            cell["spill_bytes"] for arm in cen["arms"].values()
            for keys in arm.values() for cell in keys.values()),
        **gate_flags("compile-only census through xcrun metal-tt, no GPU",
                     False),
    })
    run.finish()


CORRECTION_FORM = (
    "isolated per-cell effect x table(mechanism class, memory regime); the "
    "leg share is never multiplied, and the width term is 1.000 locally and "
    "ranked-only otherwise")


def log_correction(dry: bool = False) -> None:
    pre = prereg()
    corr = json.loads((ART / "correction.json").read_text())
    at = corr["anchor_test"]
    ms = corr["marginal_summary"]
    cfg = {
        "question": (
            "one scalar cannot fit two in-situ anchors that bracket unity. "
            "Does a table over mechanism class and memory regime reproduce "
            "both, and which axis carries the difference"
        ),
        "correction_form": CORRECTION_FORM,
        "sessions": corr["sessions"],
        "anchors_are_fitting_data": False,
        "kill_rule": at["kill_rule"],
        **identity(),
        **gate_flags("analysis of measured arms, no new GPU work", False),
    }
    if dry:
        print(json.dumps({"run": "e125-correction", "config": cfg}, indent=2))
        return
    run = start(job_type="analysis", name="e125-correction", config=cfg)

    table = wandb.Table(columns=[
        "mechanism_class", "regime", "factor", "ci95_low", "ci95_high",
        "identified", "n_cells", "factor_ratio_of_sums", "base_positive_frac",
        "measured", "note"])
    for klass, row in corr["class_regime_table"]["rows"].items():
        if not row.get("measured"):
            table.add_data(klass, None, None, None, None, False, 0, None,
                           None, False, row["note"])
            continue
        for regime, cell in row["by_regime"].items():
            ci = cell.get("ci95", [None, None])
            table.add_data(
                klass, regime, cell.get("factor"), ci[0], ci[1],
                cell.get("identified"), cell.get("n_cells"),
                cell.get("factor_ratio_of_sums"),
                cell.get("base_positive_frac"), cell.get("measured"),
                cell.get("note") or cell.get("unidentified_reason"))
    run.log({"class_regime_table": table})

    anchors = wandb.Table(columns=[
        "anchor", "mechanism_class", "published_factor", "regime",
        "table_factor", "table_ci95_low", "table_ci95_high", "identified",
        "covers_anchor", "covers_alternative_reading"])
    for name, v in at["by_anchor"].items():
        for regime, d in v.get("by_regime", {}).items():
            ci = d.get("table_ci95", [None, None])
            anchors.add_data(
                name, v.get("class"), v.get("anchor_factor"), regime,
                d.get("table_factor"), ci[0], ci[1], d.get("identified"),
                d.get("covers_anchor"), d.get("covers_alternative_reading"))
    run.log({"anchor_validation": anchors})

    obs = wandb.Table(columns=[
        "class", "regime", "frame", "session", "shape", "m", "mu", "mu_base",
        "d_mu", "base_us_per_k_block", "frame_us_per_k_block"])
    for o in ms["observations"]:
        obs.add_data(o["klass"], o["regime"], o["frame"], o["session"],
                     o["shape"], o["m"], o["mu"], o["mu_base"], o["d_mu"],
                     o["base"], o["value"])
    run.log({"observations": obs})

    arch = corr["architecture_axis"]
    fp = wandb.Table(columns=[
        "frame_pair", "instrument_axis", "host_changes", "mechanism_class",
        "component_classes", "anchor_factor", "anchor_ci_low",
        "anchor_ci_high", "instrument"])
    for row in corr["frame_pair_table"]["rows"].values():
        lo, hi = row["anchor_interval"]
        fp.add_data(row["frame_pair"], row["axis"], row["host_changes"],
                    row["mechanism_class"],
                    ",".join(row["component_classes"]), row["anchor_factor"],
                    lo, hi, row["instrument"])
    run.log({"frame_pair_table": fp})

    ax = wandb.Table(columns=[
        "census_frame", "physically_correct", "local_occupancy_change_pct",
        "ranked_occupancy_change_pct", "predicted_factor", "sign_correct",
        "inside_anchor_interval", "magnitude_shortfall"])
    for frame, cell in arch["by_frame"].items():
        ax.add_data(frame, frame == "entry_point",
                    cell["local_occupancy_change_pct"],
                    cell["ranked_occupancy_change_pct"],
                    cell["predicted_factor"], cell["sign_correct"],
                    cell["inside_anchor_interval"],
                    cell["magnitude_shortfall"])
    run.log({"architecture_axis": ax})

    rb = corr["route_b"]
    summary = {
        "e125_correction_form": CORRECTION_FORM,
        "e125_anchors_reproduced": at["n_anchors_reproduced"],
        "e125_anchors_total": at["n_anchors"],
        "e125_anchor_verdict": at["verdict"],
        "e125_single_regime_explains_both": at["single_regime_explains_both"],
        "e125_shared_regime": ",".join(at["shared_regime"]) or "none",
        "e125_variance_explained_class_only":
            ms["variance_explained_vs_flat"]["class_only"],
        "e125_variance_explained_regime_only":
            ms["variance_explained_vs_flat"]["regime_only"],
        "e125_variance_explained_class_by_regime":
            ms["variance_explained_vs_flat"]["class_by_regime"],
        "e125_regime_axis_dominates_class_axis":
            ms["regime_axis_dominates_class_axis"],
        "e125_route_b_isolated_ranked_pct": rb["isolated_ranked_pct"],
        "e125_route_b_parity_line_pct": pre["decision_lines"]["parity"],
        "e125_route_b_mode_proof_line_pct": pre["decision_lines"]["mode_proof"],
        "e125_w_local": corr["width_terms"]["local"]["W"],
        "e125_w_ranked": corr["width_terms"]["ranked"]["W"],
        "e116_share_term_untouched": not corr["null_control"][
            "share_term_corrected"],
        "e116_alpha_times_beta": corr["null_control"]["e116_alpha_times_beta"],
        "e125_arch_anchor_factor": arch["anchor"]["factor"],
        "e125_arch_sign_reproduced": arch["verdict"]["sign_reproduced"],
        "e125_arch_magnitude_reproduced":
            arch["verdict"]["magnitude_reproduced"],
        "e125_arch_predicted_factor_entry_point":
            arch["by_frame"]["entry_point"]["predicted_factor"],
        "e125_arch_predicted_factor_isolated_bodies":
            arch["by_frame"]["isolated_bodies_f47_weighted"][
                "predicted_factor"],
        "e125_arch_ranked_registers_to_gain_one":
            arch["actionable_gate"]["current_shipped_state"][
                "registers_to_gain_one"],
        "e125_f_floor_1p43_status":
            corr["falsified_stage0_terms"]["F_floor_1p43"]["status"],
        "e125_bandwidth_law_status":
            corr["falsified_stage0_terms"]["bandwidth_regime_law"]["status"],
        **gate_flags("analysis of measured arms, no new GPU work", False),
    }
    # The per-(arm, width) fidelity exclusions, pooled over both sessions. Only
    # the sync session has any: `k_tgld16` spills at NA=4 on this architecture,
    # which E123 already recorded, and a spilling rung may not be priced.
    lost: list[str] = []
    excluded: list[str] = []
    valid = True
    for spec in SESSIONS.values():
        gates = law(spec["law"])["gates"]
        excl = gates.get("exclusions")
        if excl is not None:
            lost += ["%s@m%d" % (v["class"], v["m"])
                     for v in excl["class_widths_lost"]]
            excluded += ["%s@m%d" % (v["arm"], v["m"])
                         for v in excl["excluded_arm_widths"]]
        valid = valid and gates.get("session_valid_after_exclusions",
                                    gates["session_valid"])
    summary["e125_session_valid_after_exclusions"] = valid
    summary["e125_class_widths_lost_to_fidelity"] = ",".join(lost) or "none"
    summary["e125_excluded_arm_widths"] = ",".join(excluded) or "none"
    env = rb.get("envelope_across_regimes")
    if env:
        summary["e125_route_b_envelope_low_pct"] = env[0]
        summary["e125_route_b_envelope_high_pct"] = env[1]
    for regime, cell in rb.get("by_regime", {}).items():
        if cell.get("measured"):
            summary["e125_route_b_ranked_pct_%s" % regime] = cell["ranked_pct"]
    run.summary.update(summary)
    run.finish()


def log_residency_law(dry: bool = False) -> None:
    """F5: the two-channel law, and residency against bandwidth as regressors.

    Zero GPU again. Every input is an already-published artifact, so this run
    holds analysis and no new measurement.
    """
    doc = json.loads((ART / "residency-law.json").read_text())
    channels = doc["two_channel"]
    bracket = doc["coefficient_bracket"]
    stats = doc["bandwidth_versus_residency"]
    audit = doc["channel_audit"]
    pooled = stats["pooled"]
    models = stats["models"]
    body = channels.get("isolated_bodies_f47_weighted", {})
    entry = channels.get("entry_point", {})
    cfg = {
        "question": (
            "F5 proposes `observed transfer = instruction channel x residency "
            "channel`. Does it hold across the three anchors and across "
            "mechanism classes, and do my Stage 0 bandwidth correlations "
            "survive once residency is controlled"),
        "command": ("python3 research/e125_residency_law.py --out "
                    "research/e125-artifacts/residency-law.json"),
        "model": "effect_pct = instruction_channel + coefficient * "
                 "resident_simdgroup_change_pct",
        "inputs": doc["sources"],
        "anchors": doc["anchors"],
        "simdgroup_budget": doc["simdgroup_budget"],
        "identification_note": (
            "one architecture-spanning anchor and two unknowns, so the E121 "
            "solve is exactly determined and is not a test. The testable "
            "content is the coefficient value, the instruction-channel sign, "
            "and the third anchor."),
        **identity(),
        **gate_flags("analysis of published artifacts, no GPU", False),
    }
    if dry:
        print(json.dumps({"run": "e125-residency-law", "config": cfg},
                         indent=2))
        return
    run = start(job_type="analysis", name="e125-residency-law", config=cfg)

    contrasts = wandb.Table(columns=[
        "class", "low_rung", "high_rung", "width", "registers_local",
        "residency_pct_local", "registers_ranked", "residency_pct_ranked",
        "pure_instruction_channel"])
    for row in doc["contrast_residency"]:
        contrasts.add_data(
            row["class"], row["low_rung"], row["high_rung"], row["width"],
            str(row["registers_local"]), row["residency_pct_local"],
            str(row["registers_ranked"]), row["residency_pct_ranked"],
            row["pure_instruction_channel"])

    solves = wandb.Table(columns=[
        "residency_frame", "residency_pct_local", "residency_pct_ranked",
        "coefficient_pct_time_per_pct_residency", "instruction_channel_pct",
        "instruction_sign_ok"])
    for frame, rec in sorted(channels.items()):
        solves.add_data(
            frame, rec["residency_pct_local"], rec["residency_pct_ranked"],
            rec["coefficient_pct_time_per_pct_residency"],
            rec["instruction_channel_pct"], rec["instruction_channel_sign_ok"])

    anchors = wandb.Table(columns=[
        "anchor", "axis", "residency_frame", "observed", "predicted",
        "verdict", "is_a_test", "why"])
    for row in doc["anchor_checks"]:
        anchors.add_data(
            row["anchor"], row["axis"], row.get("frame", ""), row["observed"],
            row["predicted"], row["verdict"], row["is_a_test"], row["why"])

    cells = wandb.Table(columns=[
        "shape", "width", "achieved_gb_s", "a_base_us", "gain_pct_vs_a_base",
        "gain_pct_vs_scaffold", "deleted_issue_lanes", "residency_pct_local",
        "residency_pct_ranked"])
    for row in doc["cells"]:
        cells.add_data(
            row["shape"], row["width"], row["achieved_gb_s"], row["a_base_us"],
            row["gain_pct_vs_a_base"], row["gain_pct_vs_scaffold"],
            row["deleted_issue_lanes"], row["residency_pct_local"],
            row["residency_pct_ranked"])

    regressors = wandb.Table(columns=[
        "model", "r2", "rmse", "n", "intercept", "coefficients"])
    for name, model in models.items():
        if model.get("singular"):
            continue
        regressors.add_data(name, model["r2"], model["rmse"], model["n"],
                            model["intercept"], str(model["coefficients"]))

    within = wandb.Table(columns=[
        "width", "n", "r_gain_bandwidth", "deleted_issue_lanes",
        "residency_pct_local", "residency_is_constant_within_width"])
    for row in stats["within_width"]:
        within.add_data(row["width"], row["n"], row["r_gain_bandwidth"],
                        row["deleted_issue_lanes"], row["residency_pct_local"],
                        row["residency_is_constant_within_width"])

    run.log({"contrast_residency": contrasts, "two_channel_solve": solves,
             "anchor_checks": anchors, "cells": cells,
             "regressor_models": regressors, "within_width": within})

    demeaned = stats["width_demeaned"]
    tested = [a for a in doc["anchor_checks"] if a["is_a_test"]]
    passed = [a for a in tested if a["verdict"] == "passes"]
    instructions_r2 = models["instructions_only"]["r2"]
    run.summary.update({
        "primary_metric_name": "e125_two_channel_coefficient_isolated_bodies",
        "e125_two_channel_coefficient_entry_point":
            entry.get("coefficient_pct_time_per_pct_residency"),
        "e125_two_channel_coefficient_isolated_bodies":
            body.get("coefficient_pct_time_per_pct_residency"),
        "e125_two_channel_instruction_pct_isolated_bodies":
            body.get("instruction_channel_pct"),
        "e125_two_channel_instruction_sign_ok":
            bool(body.get("instruction_channel_sign_ok")),
        "e125_f5_bracket_consistent_as_registered":
            bracket["as_registered_in_f5"]["consistent"],
        "e125_f5_bracket_consistent_in_leg_frame":
            bracket["leg_frame_on_both_sides"]["consistent"],
        "e125_f5_bracket_leg_frame_lower":
            bracket["leg_frame_on_both_sides"]["lower_bound"],
        "e125_f5_bracket_leg_frame_upper":
            bracket["leg_frame_on_both_sides"]["upper_bound"],
        "e125_two_channel_anchors_tested": len(tested),
        "e125_two_channel_anchors_passed": len(passed),
        "e125_r_gain_bandwidth_pooled": pooled["r_gain_bandwidth"],
        "e125_r_gain_residency_pooled": pooled["r_gain_residency"],
        "e125_partial_r_gain_bandwidth_given_residency":
            pooled["partial_r_gain_bandwidth_given_residency"],
        "e125_r_gain_bandwidth_width_demeaned":
            demeaned["r_gain_bandwidth"],
        "e125_r_gain_deleted_instructions_pooled":
            pooled["r_gain_deleted_instructions"],
        "e125_r2_instructions_only": instructions_r2,
        "e125_r2_gain_from_bandwidth":
            models["instructions_and_bandwidth"]["r2"] - instructions_r2,
        "e125_r2_gain_from_residency":
            models["instructions_and_residency"]["r2"] - instructions_r2,
        "e125_bandwidth_survives_residency_control":
            abs(pooled["partial_r_gain_bandwidth_given_residency"]) > 0.5,
        "e125_ladder_contrasts_pure_instruction_channel":
            audit["n_pure_instruction_channel"],
        "e125_ladder_contrasts_total": audit["n_contrasts"],
        "e125_worst_contaminated_class": audit["worst_contaminated"]["class"],
        "e125_worst_contaminated_residency_pct_local":
            audit["worst_contaminated"]["residency_pct_local"],
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    })
    run.finish()


RUNS = {
    "e125-frame-axis": lambda dry=False: log_frame_axis(dry, "work"),
    "e125-frame-axis-sync": lambda dry=False: log_frame_axis(dry, "sync"),
    "e125-register-census": log_register_census,
    "e125-residency-law": log_residency_law,
    "e125-correction": log_correction,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for name, fn in RUNS.items():
        if args.only and name != args.only:
            continue
        fn(args.dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
