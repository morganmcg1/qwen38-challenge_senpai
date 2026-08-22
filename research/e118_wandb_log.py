#!/usr/bin/env python3
"""Publish the E118 wide-qmv metadata-load instruction screen to W&B.

    usage: research/e118_wandb_log.py [--only RUN]

  `e118-arms`           the isolated Metal arms at NA = 2, 3, 4 and 5 over five
                        scored shapes, the primary metric and the discriminator.
  `e118-static-budget`  AIR device loads, registers, spill bytes and ISA text
                        for the local `applegpu_g16s` and the ranked
                        `applegpu_g17s`.
  `e118-spill-defect`   the NA=5 exactness failure and the `z_ballast` control
                        that attributes it to spilling.

Every timed leg here is a standalone Metal microbenchmark. It holds no model
and runs no benchmark wrapper, so it passes no thermal gate. Each run logs
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` verbatim as
false, and no leg here is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e118-wide-qmv-metadata-load-instruction-screen"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e118-artifacts")

PR = 120
ASSIGNMENT_BASE_SHA = "1d2320bece29cddc94b95e5f99f00331b05a5025"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
KILL_RULE_PCT = 0.5

# Republishing must correct a run in place. A second run of the same evidence
# would leave two disagreeing records of one experiment.
RUN_IDS = {
    "e118-arms": "e118arms1",
    "e118-static-budget": "e118stat1",
    "e118-spill-defect": "e118spil1",
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
    }


def log_arms() -> None:
    doc = summary()
    run = start(
        job_type="isolated-metal-arms", name="e118-arms",
        config={
            "question": (
                "does removing or repacking the per-group metadata loads in "
                "the wide qmv inner loop make the kernel faster"
            ),
            "leg_command": (
                "research/e118_probe.sh e118-full2 --shapes 0,1,2,3,4 "
                "--widths 2,3,4,5 --pairs 8 --samples 24"
            ),
            "shapes": list(doc["per_shape"].get("g_pack32", {}).keys()),
            "headline_shape": doc["shape"],
            "widths": doc["widths"],
            "pairs": doc["pairs"],
            "warmup_blocks": doc["warmup_blocks"],
            "standing_weights": doc["standing_weights"],
            "kill_rule_pct": KILL_RULE_PCT,
            "sign_convention": doc["sign_convention"],
            **identity(),
            **gate_flags("standalone Metal microbenchmark, one GPU", True),
        },
    )

    weighted = doc["weighted"]
    table = wandb.Table(columns=[
        "arm", "role", "round_weighted_pct_faster", "na2_pct", "na3_pct",
        "na4_pct", "na5_pct"])
    for arm, row in sorted(weighted.items(),
                           key=lambda kv: -kv[1]["standing_pct"]):
        na = row["na"]
        table.add_data(arm, row["role"], row["standing_pct"],
                       na.get("2"), na.get("3"), na.get("4"), na.get("5"))
    run.log({"arms_round_weighted": table})

    abs_us = wandb.Table(columns=["cell", "median_us"])
    for cell, value in sorted(doc["absolute_us"].items()):
        abs_us.add_data(cell, value)
    run.log({"absolute_microseconds": abs_us})

    shape_tbl = wandb.Table(columns=["arm", "shape", "round_weighted_pct"])
    for arm, per in doc["per_shape"].items():
        for shape, row in per.items():
            shape_tbl.add_data(arm, shape, row["weighted_pct"])
    run.log({"per_shape_round_weighted": shape_tbl})

    gaps = wandb.Table(columns=["arm", "na", "forward_reverse_gap_pct"])
    for key, value in sorted(doc["forward_reverse_gap_pct"].items()):
        arm, _, na = key.partition("|")
        gaps.add_data(arm, na, value)
    run.log({"defect16_forward_reverse_gap": gaps})

    bias = doc.get("bias_axis")
    if bias:
        bias_tbl = wandb.Table(columns=[
            "quantity", "na2_pct", "na3_pct", "na4_pct", "na5_pct",
            "round_weighted_pct"])
        for key in ("whole_bias_axis", "bias_arithmetic", "bias_load",
                    "bias6_ceiling", "bias6_real", "bias6_reconstruction"):
            row = bias[key]
            na = row["na"]
            bias_tbl.add_data(key, na.get("2"), na.get("3"), na.get("4"),
                              na.get("5"), row["weighted"])
        run.log({"e111_bias_axis_by_width": bias_tbl})

    pm = doc["primary_metric"]
    disc = doc["discriminator"]
    run.summary.update({
        "primary_metric_name": pm["name"],
        "e118_best_bit_exact_arm_round_weighted_pct_faster_vs_a_base":
            pm["value"],
        "primary_metric_arm": pm["arm"],
        "kill_rule_pct": pm["kill_rule_pct"],
        "kill_rule_cleared": pm["cleared"],
        "discriminator_verdict": disc["verdict"],
        "discriminator_s_bcast_pct": disc["s_bcast"],
        "discriminator_s_bcast_all_pct": disc["s_bcast_all"],
        "discriminator_p_split_meta_pct": disc["p_split_meta"],
        "discriminator_n_nosums_pct": disc["n_nosums"],
        "finding44_round_weighted_gap_pct":
            doc["finding44"]["round_weighted_gap_pct"],
        "exact_failure_count": len(doc["fidelity"]["exact_failures"]),
        "positive_control_failure_count":
            len(doc["fidelity"]["control_failures"]),
        "screen_weakened_cells":
            len(doc["fidelity"].get("screen_weakened_by_nonfinite", [])),
        "conclusion": (
            "The shipped wide qmv inner loop already issues only 7 device "
            "loads per entry point, so the 8 scalar metadata reads are "
            "already coalesced by the front end. Broadcasting them costs "
            "registers and shuffles and loses. Removing one load with a "
            "packed scale-bias word gains %+.3f %%, below the %+.2f %% kill "
            "rule. The load-issue port is not the binding resource."
            % (pm["value"], KILL_RULE_PCT)),
    })
    run.finish()


def log_static() -> None:
    doc = summary()
    census = doc["census"]
    run = start(
        job_type="static-budget", name="e118-static-budget",
        config={
            "question": (
                "how many device loads, registers, spill bytes and ISA text "
                "bytes does each arm cost on the local and the ranked "
                "architecture"
            ),
            "leg_command": (
                "research/e118_arms.py --emit DIR && "
                "research/e118_arms.py --census DIR --out census.json"
            ),
            "widths": census["widths"],
            **identity(),
            **gate_flags("offline metal-arch translation on CPU, no GPU",
                         False),
        },
    )

    air = wandb.Table(columns=["arm", "na", "device_loads", "shuffles"])
    static = wandb.Table(columns=[
        "arm", "arch", "na", "registers", "spill_bytes", "text_bytes"])
    for arm, row in census["arms"].items():
        for na in census["widths"]:
            cell = row["air"].get(str(na), {})
            air.add_data(arm, na, cell.get("device_loads"),
                         cell.get("shuffles", 0))
            for arch in (census["local_arch"], census["ranked_arch"]):
                v = (row.get(arch) or {}).get(str(na))
                if v is None:
                    continue
                static.add_data(arm, arch, na, v["registers"],
                                v["spill_bytes"] or 0, v["text_bytes"])
    run.log({"air_device_loads": air, "registers_spill_text": static})

    base_loads = census["arms"]["a_base"]["air"]["5"]["device_loads"]
    run.summary.update({
        "shipped_device_loads_per_entry_point": base_loads,
        "conclusion": (
            "a_base already issues %d device loads. The eight scalar metadata "
            "reads are coalesced before any arm touches them, so the "
            "broadcast arms remove zero loads and only add registers and "
            "shuffles." % base_loads),
    })
    run.finish()


def log_spill_defect() -> None:
    doc = summary()
    se = doc.get("spill_exactness")
    fid = doc["fidelity"]
    run = start(
        job_type="exactness-defect", name="e118-spill-defect",
        config={
            "question": (
                "is the NA=5 bit-exactness failure caused by an arm's "
                "mechanism or by the compiler spilling"
            ),
            "control": (
                "z_ballast adds 12 dead loop-carried floats that are consumed "
                "only inside a branch that never runs, so no value it "
                "computes reaches y. It changes the spill budget and nothing "
                "else."
            ),
            "reference": (
                "every arm is scored against an exact affine-4 reference "
                "evaluated in double on the CPU"
            ),
            **identity(),
            **gate_flags("standalone Metal microbenchmark, one GPU", False),
        },
    )

    failures = wandb.Table(columns=[
        "arm", "shape", "na", "differing", "total", "max_ulp", "max_rel",
        "vs_double_max_rel"])
    for row in fid["exact_failures"]:
        failures.add_data(row["arm"], row["shape"], row["m"], row["differing"],
                          row["total"], row["max_ulp"], row["max_rel"],
                          row.get("vs_double_max_rel"))
    run.log({"exactness_failures": failures})

    base_ref = wandb.Table(columns=["shape", "na", "base_vs_double_max_rel",
                                    "base_vs_double_rms_over_signal"])
    for row in fid["base_vs_double"]:
        base_ref.add_data(row["shape"], row["m"], row["max_rel"], row["rms"])
    run.log({"base_against_double_reference": base_ref})

    if se:
        join = wandb.Table(columns=[
            "arm", "na", "spill_bytes", "registers", "exact",
            "vs_double_max_rel"])
        for arm, by_na in sorted(se["arms"].items()):
            for na, cell in sorted(by_na.items()):
                join.add_data(arm, int(na), cell["spill_bytes"],
                              cell["registers"], cell["exact"],
                              cell["vs_double_max_rel"])
        run.log({"spill_against_exactness": join})
        zb = se["arms"].get("z_ballast", {})
        zwrong = sorted(int(na) for na, c in zb.items() if not c["exact"])
        run.summary.update({
            "arch": se["arch"],
            "max_spill_while_exact_bytes": se["max_spill_while_exact"],
            "min_spill_while_wrong_bytes": se["min_spill_while_wrong"],
            "spill_separates_exact_from_wrong": se["separates"],
            "z_ballast_wrong_at_na": zwrong,
            "conclusion": (
                "z_ballast changes no arithmetic that reaches y and is still "
                "wrong at NA %s, so the NA=5 divergence is caused by the "
                "compiler spilling, not by any arm's mechanism. On this host "
                "and toolchain, keep the wide qmv NA=5 entry point at or "
                "below %s B of spill and check exactness whenever it spills."
                % (zwrong, se["max_spill_while_exact"])),
        })
    run.finish()


RUNS = {
    "e118-arms": log_arms,
    "e118-static-budget": log_static,
    "e118-spill-defect": log_spill_defect,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS))
    args = ap.parse_args()
    for name, fn in RUNS.items():
        if args.only in (None, name):
            print(f"== {name}")
            fn()


if __name__ == "__main__":
    main()
