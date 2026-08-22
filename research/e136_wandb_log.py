#!/usr/bin/env python3
"""Publish an E136 rung to W&B.

    usage: research/e136_wandb_log.py --rung 0|0b [--dry]

Rung 0 times Metal kernels but it is NOT a decode leg, a gated measurement or
a score. It times the survivor-selection dispatches standing alone, off the
scored path, so every run logs `cool_gate_passed_real_gate`,
`gate_qualified_for_timing` and `official_or_ranked_score` verbatim as false.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e136-c1-sketch-readout-build"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
BASE_SHA = "35d8cf586b8671dc3d01faf3cdbd724ec603801b"

ARM_COLUMNS = [
    "arm", "survivors", "tiles", "dispatches", "added_us_batched",
    "added_us_isolated", "added_us_parts_plus_ledger_f", "added_ranked_pct",
    "added_ranked_pct_e93_rate", "recall_of_true_top32",
    "us_per_call", "batched_us_per_selection",
]

PART_COLUMNS = ["arm", "part", "us_above_floor"]

FLOOR_COLUMNS = [
    "statistic", "stratum", "n", "gain_events_b", "loss_events_c",
    "discordant_D", "acceptance_delta", "ranked_pct",
    "floor_pure_gain_ranked_pct", "two_sigma_band_null_ranked_pct",
    "sign_test_p", "null_survives_2sigma",
]

ARM_F_COLUMNS = [
    "stratum", "n", "rows_tied_at_max_shipped_arith", "fp32_changed",
    "fp32_changed_on_a_base_miss", "fp32_changed_on_a_tied_row",
    "fp32_changed_live", "fp32_changed_live_new_is_target",
    "fp32_changed_live_old_is_target", "fp32_max_attainable_ranked_pct",
]

RUNGS = {
    "0": {
        "run_name": "e136-rung0-widened-selection-microbenchmark",
        "file": "research/e136-selection-bench.json",
        "question":
            "does selecting the top N of 34,424 sketch-scored rows with a "
            "histogram threshold cost less added GPU time per draft step "
            "than the 37.483 MB the C1 readout removes is worth",
        "command":
            "research/e133_job.sh research/await-lock-then-run.sh 1800 "
            "python3 research/e136_selection_bench.py --replicates 3",
    },
    "0b": {
        "run_name": "e136-rung0b-fp32-rerank-tiebreak",
        "file": "research/e136-fp32-floor.json",
        "companion": "research/e136-attrib-fp32.json",
        "job_type": "attribution-replay",
        "question":
            "is the one narrowing store in the affine-4 rerank kernel "
            "(Qwen35.swift:4118, typedef bfloat16_t InT at :4143) worth "
            "removing, priced on realised acceptance against a replay "
            "baseline remodelled on that same kernel's arithmetic",
        "command":
            "research/e133_job.sh python3 research/e133_screen.py attrib "
            "--per-seed --out research/e136-attrib-fp32.json && "
            "python3 research/e136_null_floor.py "
            "--attrib research/e136-attrib-fp32.json "
            "--out research/e136-fp32-floor.json",
    },
}


def rung0_summary(payload: dict, spec: dict) -> tuple[dict, dict]:
    verdict = payload["verdict"]
    arms, parts = [], []
    for label, row in payload["arms"].items():
        if "added_us_per_step" not in row:
            continue
        plan = row.get("plan", {})
        arms.append({
            "arm": label,
            "survivors": plan.get("real_count"),
            "tiles": plan.get("tiles"),
            "dispatches": row.get("dispatches_per_call"),
            "added_us_batched": row["added_us_per_step"],
            "added_us_isolated": row["added_us_isolated"],
            "added_us_parts_plus_ledger_f":
                row["added_us_from_parts_plus_ledger_f"],
            "added_ranked_pct": row["added_ranked_pct"],
            "added_ranked_pct_e93_rate":
                row["added_us_per_step"] / 247.2,
            "recall_of_true_top32":
                payload["correctness"][label]["recall_of_true_top32"],
            "us_per_call": row["us_per_call"],
            "batched_us_per_selection": row["batch"]["us_per_selection"],
        })
        for part, value in row.get("parts_us_above_floor", {}).items():
            parts.append({"arm": label, "part": part, "us_above_floor": value})

    summary = {
        # The metric the assignment names. The whole-arm batched slope is the
        # estimator: it replicates to 0.4 percent where the part arms move by
        # a factor of three.
        "e136_widened_selection_us_per_draft_step":
            verdict["added_us_per_draft_step"],
        "e136_widened_selection_us_spread":
            verdict["added_us_per_draft_step_spread"],
        "e136_widened_selection_ranked_pct":
            verdict["added_ranked_pct_at_265gbs"],
        "e136_widened_selection_ranked_pct_e93_rate":
            verdict["added_ranked_pct_at_measured_186_7gbs"],
        "e136_rung0_stop_rule": verdict["stop_rule"],
        "e136_rung0_stop_rule_on_worst_estimate":
            verdict["stop_rule_on_worst_estimate"],
        "e136_rung0_net_pct_after_selection_cost":
            verdict["net_pct_after_selection_cost"],
        "e136_rung0_recall_min":
            min(c["recall_of_true_top32"]
                for c in payload["correctness"].values()),
        "e136_rung0_positive_control_fires":
            all(c["positive_control_detects_the_change"]
                for c in payload["correctness"].values()),
        "e136_rung0_device_threshold_matches_host":
            all(c["device_threshold_matches_host"]
                for c in payload["correctness"].values()),
    }
    for key, value in verdict.items():
        if isinstance(value, (int, float, str, bool)):
            summary["verdict/%s" % key] = value
    for key, value in payload["anchor_check"].items():
        if isinstance(value, (int, float, str, bool)):
            summary["anchor/%s" % key] = value
    for i, rep in enumerate(payload["replicates"]):
        for label, value in rep.items():
            summary["replicate/%d/%s" % (i, label)] = value
    return summary, {
        "rung0_arms": table(ARM_COLUMNS, arms),
        "rung0_parts": table(PART_COLUMNS, parts),
    }


# The advisor's rung-0b bars, stated here so the logged verdict cannot drift
# from the rule that produced it. Both are ranked percent on `pool:corpus`.
RUNG0B_ADVANCE_PCT = 0.40
RUNG0B_CLOSE_PCT = 0.25
RUNG0B_BEAGLE_ALONE_PCT = 0.50


def rung0b_summary(payload: dict, spec: dict) -> tuple[dict, dict]:
    attrib = json.loads(pathlib.Path(spec["companion"]).read_text())
    stats = payload["statistics"]
    rows = []
    for name, stat in stats.items():
        for stratum, r in stat["by_stratum"].items():
            rows.append(dict(r, statistic=name, stratum=stratum))
        for stratum, r in stat["pooled"].items():
            rows.append(dict(r, statistic=name, stratum="pool:" + stratum))

    decide = stats["fp32_rerank"]["pooled"]["corpus"]
    beagle = stats["fp32_rerank"]["by_stratum"]["beagle"]
    ship = stats["shipped_arithmetic_model"]["pooled"]["corpus"]
    pct = decide["ranked_pct"]
    verdict = ("advance" if pct >= RUNG0B_ADVANCE_PCT
               else "close" if pct < RUNG0B_CLOSE_PCT else "unresolved")

    summary = {
        # The decision statistic and the two bars it is read against.
        "e136_rung0b_fp32_rerank_ranked_pct": pct,
        "e136_rung0b_verdict": verdict,
        "e136_rung0b_advance_bar_pct": RUNG0B_ADVANCE_PCT,
        "e136_rung0b_close_bar_pct": RUNG0B_CLOSE_PCT,
        "e136_rung0b_beagle_alone_bar_pct": RUNG0B_BEAGLE_ALONE_PCT,
        "e136_rung0b_beagle_ranked_pct": beagle["ranked_pct"],
        # Every estimate is reported beside its own detection floor.
        "e136_rung0b_floor_pct": decide["floor_pure_gain_ranked_pct"],
        "e136_rung0b_beagle_floor_pct": beagle["floor_pure_gain_ranked_pct"],
        "e136_rung0b_two_sigma_band_pct":
            decide["two_sigma_band_null_ranked_pct"],
        "e136_rung0b_sign_test_p": decide["sign_test_p"],
        "e136_rung0b_beagle_sign_test_p": beagle["sign_test_p"],
        "e136_rung0b_null_survives_2sigma": decide["null_survives_2sigma"],
        "e136_rung0b_n": decide["n"],
        "e136_rung0b_gain_events_b": decide["gain_events_b"],
        "e136_rung0b_loss_events_c": decide["loss_events_c"],
        "e136_rung0b_discordant_D": decide["discordant_D"],
        # The corrected baseline. A non-significant shift here is what lets
        # the earlier attribution conclusions stand.
        "e136_rung0b_shipped_arith_baseline_shift_pct": ship["ranked_pct"],
        "e136_rung0b_shipped_arith_sign_test_p": ship["sign_test_p"],
        "e136_rung0b_shipped_arith_null_survives_2sigma":
            ship["null_survives_2sigma"],
        # False is the finding: MLX bfloat16 quantized_matmul is not
        # "accumulate in float32 and round once".
        "e136_rung0b_mlx_bf16_qmm_is_rounded_f32":
            attrib["fp32_roundtrip_exact"],
        "e136_rung0b_output_changes_on_a_base_miss": sum(
            r["fp32_changed_on_a_base_miss"]
            for r in attrib["by_stratum"].values()),
        "e136_rung0b_output_changes_on_a_tied_row": sum(
            r["fp32_changed_on_a_tied_row"]
            for r in attrib["by_stratum"].values()),
        "e136_rung0b_rows_tied_at_max": sum(
            r["rows_tied_at_max_shipped_arith"]
            for r in attrib["by_stratum"].values()),
    }
    arm_f = [dict(r, stratum=s) for s, r in attrib["by_stratum"].items()]
    return summary, {
        "rung0b_floor": table(FLOOR_COLUMNS, rows),
        "rung0b_arm_f_events": table(ARM_F_COLUMNS, arm_f),
    }


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


BUILDERS = {"0": rung0_summary, "0b": rung0b_summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    spec = RUNGS[args.rung]
    job_type = spec.get("job_type", "kernel-microbenchmark")
    payload = json.loads(pathlib.Path(spec["file"]).read_text())
    summary, tables = BUILDERS[args.rung](payload, spec)
    summary.update({
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "timing_valid": False,
        "host": HOST,
        "base_sha": BASE_SHA,
        "rung": args.rung,
        "question": spec["question"],
        "command": spec["command"],
    })

    if args.dry:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=spec["run_name"], job_type=job_type,
                     config={"experiment": "E136", "rung": args.rung,
                             "base_sha": BASE_SHA, "host": HOST,
                             "command": spec["command"],
                             "question": spec["question"]})
    for name, value in tables.items():
        run.log({name: value})
    run.summary.update(summary)
    artifact = wandb.Artifact("e136-rung%s" % args.rung, type=job_type)
    artifact.add_file(spec["file"])
    if "companion" in spec:
        artifact.add_file(spec["companion"])
    run.log_artifact(artifact)
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
