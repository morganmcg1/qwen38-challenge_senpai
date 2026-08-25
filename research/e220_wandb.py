#!/usr/bin/env python3
"""Publish the E220 integer-path census and verdict to W&B.

    usage: research/e220_wandb.py [--name NAME] [--notes TEXT]

E220 spent ZERO GPU seconds. Every number here comes from `xcrun metal` and
`xcrun metal-tt`, which run the real AGX backend for a named architecture on
any Mac. Nothing here is a timing measurement, an official score or a ranked
score, and the thermal gate never applied because no leg was ever timed.

`g16s` is the local generation and `g17s` is the ranked M5 generation. A
negative `text_delta_pct` means the variant emitted LESS machine code than the
base.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e220-artifacts"
REPO = HERE.parent

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e220-dequant-integer-path"
GROUP = "qwen38-r1-e220-dequant-integer-path"


def shell(*argv: str) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    return done.stdout.strip()


def identity() -> dict:
    return {
        "experiment": EXPERIMENT,
        "harness": "local",
        "measurement": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "base_sha": shell("git", "rev-parse", "HEAD"),
        "base_ref": "senpai/qwen38-mtp-r1",
        "assignment_base_sha": "d8de14a2b56a127652d8bb34b9fd3eb16be1a291",
        "dirty_candidate_paths": len([
            line for line in shell(
                "git", "status", "--porcelain", "--",
                "Sources", "Vendor", "Package.swift").splitlines() if line]),
        "host": platform.node(),
        "chip": shell("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(shell("sysctl", "-n", "hw.memsize") or 0),
        "os": shell("sw_vers", "-productVersion"),
        "swift": shell("swift", "--version").splitlines()[0]
        if shell("swift", "--version") else "",
        "metal": shell("xcrun", "metal", "--version").splitlines()[0]
        if shell("xcrun", "metal", "--version") else "",
        "local_arch": "applegpu_g16s",
        "ranked_arch": "applegpu_g17s",
    }


def variant_table(report: dict) -> wandb.Table:
    columns = ["variant", "is_cost_oracle", "fp_sequence_identical",
               "agx_text_identical", "introduces_spill",
               "g16s_text_delta_pct", "g17s_text_delta_pct",
               "dequant_int_ops_NA5", "verdict"]
    rows = []
    for name, v in report["variants"].items():
        rows.append([
            name, v["is_cost_oracle"], v["fp_sequence_identical"],
            v["agx_text_identical"], v["introduces_spill"],
            v["mean_text_delta_pct"]["g16s"],
            v["mean_text_delta_pct"]["g17s"],
            v["dequant_block_int_ops"].get("NA5_tbl"),
            v["verdict"],
        ])
    return wandb.Table(columns=columns, data=rows)


def cell_table(report: dict) -> wandb.Table:
    columns = ["variant", "cell", "arch", "base_text_bytes", "text_bytes",
               "text_delta_bytes", "text_delta_pct", "text_identical",
               "base_registers", "registers", "base_spill_bytes",
               "spill_bytes"]
    rows = []
    for name, v in report["variants"].items():
        for cell, per_arch in v["per_cell"].items():
            for arch, e in per_arch.items():
                rows.append([
                    name, cell, arch, e["base_text_bytes"], e["text_bytes"],
                    e["text_delta_bytes"], e["text_delta_pct"],
                    e["text_identical"], e["base_registers"], e["registers"],
                    e["base_spill_bytes"], e["spill_bytes"],
                ])
    return wandb.Table(columns=columns, data=rows)


def instantiation_table(census: dict) -> wandb.Table:
    columns = ["m", "ipg", "variant", "weight_passes", "na_instantiated",
               "cells"]
    rows = []
    for entry in census["reachable_instantiations"].values():
        rows.append([entry["m"], entry["ipg"], entry["variant"],
                     entry["groups"], str(entry["na_instantiated"]),
                     ", ".join(sorted(set(entry["cells"])))])
    return wandb.Table(columns=columns, data=rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="e220-dequant-integer-path-census")
    parser.add_argument("--notes", default=(
        "E220 static integer-path census and verdict. Zero GPU seconds; no "
        "timing, no score."))
    args = parser.parse_args()

    report = json.loads((ARTIFACTS / "e220-report.json").read_text())
    census = json.loads((ARTIFACTS / "e220_census_base.json").read_text())

    config = identity()
    config.update({
        "base_source_sha8": report["base_source_sha8"],
        "width_plan_rule": report["width_plan_rule"],
        "base_dequant_block_int_ops": report["base_dequant_block_int_ops"],
        "nibble_extraction_ops_per_4_values": 6,
        "promotion_bar_ms_per_round":
            report["perfect_extraction_prize"]["promotion_bar_ms_per_round"],
    })

    run = wandb.init(project=PROJECT, entity=ENTITY, group=GROUP,
                     name=args.name, notes=args.notes, config=config,
                     job_type="static-census",
                     tags=["e220", "integer-path", "static-compile",
                           "zero-gpu", "harness-local", "negative-result"])

    prize = report["perfect_extraction_prize"]
    run.summary.update({
        "verdict": "not useful: integer path already minimal",
        "candidates_screened": len(
            [v for v in report["variants"].values()
             if not v["is_cost_oracle"]]),
        "candidates_carried_forward": len(
            [v for v in report["variants"].values()
             if v["verdict"] == "carry forward"]),
        "bfe_agx_text_identical":
            report["variants"]["bfe"]["agx_text_identical"],
        "vecload_agx_text_identical_size":
            report["variants"]["vecload"]["mean_text_delta_pct"]["g16s"] == 0.0,
        "prize_instruction_share_pct_g16s":
            prize["instruction_share_pct"]["g16s"],
        "prize_instruction_share_pct_g17s":
            prize["instruction_share_pct"]["g17s"],
        "prize_ms_per_round_b_term_g16s":
            prize["ms_per_round_on_b_term"]["g16s"],
        "prize_ms_per_round_b_term_g17s":
            prize["ms_per_round_on_b_term"]["g17s"],
        "prize_ms_per_round_na5_pass_g16s":
            prize["ms_per_round_on_whole_na5_pass"]["g16s"],
        "prize_ms_per_round_na5_pass_g17s":
            prize["ms_per_round_on_whole_na5_pass"]["g17s"],
        "prize_reachable": prize["reachable"],
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
    })

    run.log({
        "variants": variant_table(report),
        "per_cell": cell_table(report),
        "reachable_instantiations": instantiation_table(census),
        "base_dequant_block_sequence": wandb.Table(
            columns=["index", "air_instruction"],
            data=[[i, line] for i, line in enumerate(
                report["base_dequant_block_sequence"])]),
    })

    artifact = wandb.Artifact("e220-integer-path-census", type="census")
    for path in sorted(ARTIFACTS.glob("*.json")):
        artifact.add_file(str(path))
    run.log_artifact(artifact)

    print("run:", run.url)
    print("run id:", run.id)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
