#!/usr/bin/env python3
"""Publish the E125 frame axis and the isolated-to-in-situ correction to W&B.

    usage: research/e125_wandb_log.py [--only RUN] [--dry]

  `e125-frame-axis`   the per-instruction price of one device-load and one ALU
                      instruction, and the price of one real deletion, measured
                      in six memory frames that share the compiled pipeline
                      state, the width and the launched grid volume.
  `e125-correction`   the correction applied: the predicted ranked band for
                      Route B, the back-check against alphonse's in-situ
                      figure, and the class table with unmeasured rows named.

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
OUT = pathlib.Path("research/out/e125-full")

PR = 126
ASSIGNMENT_BASE_SHA = "3b8ea425f8887c9b5cd08ddfff6ddc423fb5d9c3"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
PEAK_BANDWIDTH_GB_S = 273.0

# Republishing must correct a run in place. A second run of the same evidence
# would leave two disagreeing records of one experiment.
RUN_IDS = {
    "e125-frame-axis": "e125frame1",
    "e125-correction": "e125corr01",
}

# The keys a reader of the report arrives looking for, per run id. Consumed by
# `research/e118_wandb_check.py --experiment e125 --verify`.
EXPECTED_SUMMARY_KEYS = {
    "e125frame1": (
        "primary_metric_name",
        "e125_ld_price_ratio_k1024_over_base",
        "e125_ld_price_ratio_consumer_over_base",
        "e125_alu_price_ratio_k1024_over_base",
        "e125_alu_price_ratio_consumer_over_base",
        "e125_deletion_ratio_k1024_over_base",
        "e125_worst_ramp_residual_pct",
        "e125_entry_temp_spread_c",
        "session_valid",
        "cool_gate_passed_real_gate",
        "gate_qualified_for_timing",
        "official_or_ranked_score",
    ),
    "e125corr01": (
        "e125_route_b_predicted_ranked_pct",
        "e125_route_b_parity_line_pct",
        "e125_route_b_mode_proof_line_pct",
        "e125_alphonse_back_check_sd",
        "e125_correction_form",
    ),
}


def law() -> dict:
    return json.loads((ART / "frame-law.json").read_text())


def prereg() -> dict:
    return json.loads((ART / "routeb-prediction.json").read_text())


def meta() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (OUT / "meta.txt").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


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


def log_frame_axis(dry: bool) -> None:
    doc = law()
    cfg = {
        "question": (
            "does the per-instruction price of one device load and one ALU "
            "instruction depend on the memory frame the same compiled kernel "
            "runs in, and does one function of achieved bandwidth explain "
            "both directions"
        ),
        "leg_command": (
            "research/e125_probe.sh e125-full --shapes 0,3 --widths 2,3,4 "
            "--frames base,cycle,consumer,k1024,k2048,k4096 --pairs 8 "
            "--warm-pairs 1 --warm-sweep-reps 2 --samples 24 --target-ms 40 "
            "--inner-max 2048 --consumer-mb 512 --cycle 4"
        ),
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
        **identity(),
        **gate_flags("standalone Metal microbenchmark, one GPU", True),
    }
    if dry:
        print(json.dumps({"run": "e125-frame-axis", "config": cfg}, indent=2))
        return
    run = start(job_type="isolated-metal-arms", name="e125-frame-axis",
                config=cfg)

    cells = wandb.Table(columns=[
        "shape", "m", "frame", "k", "k_blocks", "inner", "launched_threads",
        "read_bytes", "base_us", "achieved_gb_s", "consumer_gb_s", "phi_arm",
        "phi_eff", "entry_c_min", "entry_c_max", "entry_c_spread",
        "ld_us_per_instr_per_kblock", "alu_us_per_instr_per_kblock",
        "deletion_us_per_kblock", "deletion_vs_a_base_us_per_kblock",
        "null_scaffold_pct"])
    for c in doc["cells"]:
        p = c["prices"]
        cells.add_data(
            c["shape"], c["m"], c["frame"], c["k"], c["k_blocks"], c["inner"],
            c["launched_threads"], c["read_bytes"], c["base_us"],
            c["achieved_gb_s"], c["consumer_gb_s"], c["phi_arm"],
            c["phi_eff"], c["entry_c_min"], c["entry_c_max"],
            c["entry_c_spread"],
            (p.get("ld") or {}).get("us_per_instruction_per_k_block"),
            (p.get("alu") or {}).get("us_per_instruction_per_k_block"),
            (p.get("deletion") or {}).get("gain_us_per_k_block"),
            (p.get("deletion_vs_a_base") or {}).get("gain_us_per_k_block"),
            (p.get("null_scaffold") or {}).get("cost_pct"))
    run.log({"cells": cells})

    seg = wandb.Table(columns=[
        "shape", "m", "segment", "k_blocks_from", "k_blocks_to",
        "bytes_per_k_block", "marginal_gb_s", "phi_marginal",
        "ld_us_per_instr_per_kblock", "alu_us_per_instr_per_kblock"])
    for s in doc["segments"]:
        seg.add_data(
            s["shape"], s["m"], "%s->%s" % (s["from_frame"], s["to_frame"]),
            s["k_blocks_from"], s["k_blocks_to"], s["bytes_per_k_block"],
            s["marginal_gb_s"], s["phi_marginal"],
            s.get("ld_us_per_instruction_per_k_block"),
            s.get("alu_us_per_instruction_per_k_block"))
    run.log({"marginal_segments": seg})

    ratios = wandb.Table(columns=[
        "class", "frame", "ratio_to_base", "n_points", "measured", "note"])
    for klass, entry in doc["frame_law"].items():
        for frame, v in sorted(entry["ratio_by_frame"].items()):
            ratios.add_data(klass, frame, v["median"], v["n"], v["measured"],
                            v["note"])
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
    run.summary.update({
        "primary_metric_name":
            "us per injected instruction per k-block, by memory frame",
        "e125_ld_price_ratio_k1024_over_base": ratio(doc, "ld", "k1024"),
        "e125_ld_price_ratio_k2048_over_base": ratio(doc, "ld", "k2048"),
        "e125_ld_price_ratio_k4096_over_base": ratio(doc, "ld", "k4096"),
        "e125_ld_price_ratio_consumer_over_base": ratio(doc, "ld", "consumer"),
        "e125_ld_price_ratio_cycle_over_base": ratio(doc, "ld", "cycle"),
        "e125_alu_price_ratio_k1024_over_base": ratio(doc, "alu", "k1024"),
        "e125_alu_price_ratio_consumer_over_base":
            ratio(doc, "alu", "consumer"),
        "e125_alu_price_ratio_cycle_over_base": ratio(doc, "alu", "cycle"),
        "e125_deletion_ratio_k1024_over_base": ratio(doc, "deletion", "k1024"),
        "e125_deletion_ratio_consumer_over_base":
            ratio(doc, "deletion", "consumer"),
        "e125_deletion_ratio_cycle_over_base": ratio(doc, "deletion", "cycle"),
        "e125_worst_ramp_residual_pct": worst,
        "e125_entry_temp_spread_c": spread,
        "session_valid": doc["gates"]["session_valid"],
        "gate_positive_controls_passed":
            doc["gates"]["positive_controls"]["passed"],
        "gate_fidelity_passed": doc["gates"]["fidelity"]["passed"],
        **gate_flags("standalone Metal microbenchmark, one GPU", True),
    })
    run.finish()


def log_correction(dry: bool) -> None:
    doc = law()
    pre = prereg()
    corr = json.loads((ART / "correction.json").read_text())
    cfg = {
        "question": (
            "what correction turns an isolated per-cell effect into an "
            "in-situ one, and what does it predict for Route B"
        ),
        "correction_form": corr["form"],
        "fitted_points": corr["fitted_points"],
        "held_out_points": corr["held_out_points"],
        **identity(),
        **gate_flags("analysis of measured arms, no new GPU work", False),
    }
    if dry:
        print(json.dumps({"run": "e125-correction", "config": cfg}, indent=2))
        return
    run = start(job_type="analysis", name="e125-correction", config=cfg)

    table = wandb.Table(columns=[
        "mechanism_class", "isolated_to_in_situ_factor", "interval_low",
        "interval_high", "n_independent_points", "measured", "evidence"])
    for row in corr["class_table"]:
        table.add_data(row["mechanism_class"], row.get("factor"),
                       row.get("low"), row.get("high"),
                       row["n_independent_points"], row["measured"],
                       row["evidence"])
    run.log({"class_table": table})

    run.summary.update({
        "e125_route_b_predicted_ranked_pct": corr["route_b"]["point"],
        "e125_route_b_band_low_pct": corr["route_b"]["low"],
        "e125_route_b_band_high_pct": corr["route_b"]["high"],
        "e125_route_b_parity_line_pct": pre["decision_lines"]["parity"],
        "e125_route_b_mode_proof_line_pct": pre["decision_lines"]["mode_proof"],
        "e125_alphonse_back_check_predicted_pct":
            corr["alphonse_back_check"]["predicted_pct"],
        "e125_alphonse_back_check_measured_pct":
            corr["alphonse_back_check"]["measured_pct"],
        "e125_alphonse_back_check_sd": corr["alphonse_back_check"]["sd_away"],
        "e125_e124_corrected_ranked_pct_low": corr["e124"]["low"],
        "e125_e124_corrected_ranked_pct_high": corr["e124"]["high"],
        "e125_correction_form": corr["form"],
        "e116_share_term_untouched": True,
        **gate_flags("analysis of measured arms, no new GPU work", False),
    })
    run.finish()


RUNS = {"e125-frame-axis": log_frame_axis, "e125-correction": log_correction}


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
