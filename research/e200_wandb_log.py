#!/usr/bin/env python3
"""Publish the E200 width-aware depth-price result to W&B.

    usage: research/e200_wandb_log.py \
        --desk research/e200-artifacts/desk-price.json \
        --replay research/e200-artifacts/trace-replay.json \
        --legs research/out/e200/legs

Two harnesses, never mixed:

  * `harness=ranked` -- the desk walk. Latent-q survival replay against the
    merged E197 ranked round-cost law, anchored on paid receipt A and validated
    out of sample against the paid cap-4 and cap-5 receipts. This is the
    DECISION instrument, under the RULE 79 carve-out for an offline replay
    against a measured cost table.
  * `harness=local`  -- the traced legs and the exact per-round replay of their
    real scheduler state. These confirm liveness, the policy shift and
    exactness. RULE 79 forbids publishing a depth-price timing contrast from
    local legs in any direction, so no leg wall time is reported as an effect.

Every leg ran with `MLXFAST_LOCAL_COOL_GATE=0` under the standing
counterbalanced-session allowance, so `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` are carried verbatim and no number here is an
official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e200-step-aware-depth-price"

CROWN = 3.7291100105909
RECEIPT_A = 3.70784519415395
PAID_CAP4 = 3.54742900664627
PAID_CAP5 = 3.52363580183674
RECEIPT_SIGMA_PCT = 0.689
MUE_MS = 0.567


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def read_legs(root: pathlib.Path) -> list[dict]:
    legs = []
    for meta_path in sorted(root.glob("**/leg*/meta.txt")):
        meta = dict(
            line.split("=", 1)
            for line in meta_path.read_text().splitlines()
            if "=" in line
        )
        score_path = meta_path.parent / "score.json"
        metrics = {}
        if score_path.exists():
            payload = json.loads(score_path.read_text())
            metrics = payload.get("metrics", payload)
        legs.append({"meta": meta, "metrics": metrics,
                     "dir": str(meta_path.parent)})
    return sorted(legs, key=lambda leg: int(leg["meta"].get("e200_leg", "0")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desk", type=pathlib.Path,
                        default="research/e200-artifacts/desk-price.json")
    parser.add_argument("--replay", type=pathlib.Path,
                        default="research/e200-artifacts/trace-replay.json")
    parser.add_argument("--legs", type=pathlib.Path,
                        default="research/out/e200/legs")
    parser.add_argument("--base-sha", default="fb67c8a76ec4bd56921a4ffdb1c20073c1accb67")
    args = parser.parse_args()

    desk = json.loads(pathlib.Path(args.desk).read_text())
    stage2 = desk["stage2"]
    replay = (json.loads(pathlib.Path(args.replay).read_text())
              if pathlib.Path(args.replay).exists() else None)
    legs = read_legs(pathlib.Path(args.legs))

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e200-step-aware-depth-price",
        job_type="analysis",
        config={
            "experiment": "e200-step-aware-depth-price",
            "question": (
                "does replacing makeUniformDepthPrice's width-blind marginal "
                "row price in costModelDepth with a width-aware table dR(m) "
                "from the merged E197 ranked law raise the ranked median"
            ),
            "mechanism": (
                "price side only of the greedy depth walk; the acceptance "
                "side is untouched. marginal[d] = (R(d+2) - R(d+1)) / R(1), "
                "cumulative accumulated from it. Two levels: rescaled to the "
                "shipped 1.26 over the reachable steps, and the measured level"
            ),
            "decision_harness": "ranked",
            "decision_instrument": "latent-q survival replay (RULE 79 carve-out)",
            "confirmation_harness": "local",
            "arm_switch": "MLX_E200_DEPTH_PRICE",
            "arm_switch_deleted_at_terminal": True,
            "compile_time_default": "ship",
            "mode": "qwen-mtp-local-iterate",
            "tokens": 256,
            "cap": desk["cap"],
            "law_family": desk["law"]["family"],
            "law_R_ms": desk["law"]["R_ms"],
            "ship_head_step_cost_ratio": 0.18,
            "stop_rule": (
                "desk pricing >= +0.5 % ranked median -> freeze path; "
                "< +0.2 % -> terminal not useful; 0.2-0.5 % -> Unclear"
            ),
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "published_depth_price_timing_contrast": False,
            "candidate_head": git("rev-parse", "HEAD"),
            "base_sha": args.base_sha,
            "worktree_dirty": bool(git("status", "--porcelain")),
            "host": "apple-m4-pro",
            "ranked_host": "m5-qwen38-27b-mtp",
            "crown_score": CROWN,
            "receipt_a_score": RECEIPT_A,
            "receipt_sigma_pct": RECEIPT_SIGMA_PCT,
            "mue_ms_per_round": MUE_MS,
            "legs": len(legs),
        },
    )

    desk_table = wandb.Table(
        columns=["arm", "harness", "ranked_median", "delta_vs_A_pct", "rule"])
    for arm, row in stage2["arms"].items():
        desk_table.add_data(
            arm, "ranked", row["median"],
            100.0 * (row["median"] - RECEIPT_A) / RECEIPT_A,
            row.get("rule", "greedy"))
    run.log({"desk/arms": desk_table})

    shape_table = wandb.Table(
        columns=["w_uniform_to_measured", "ranked_median", "delta_vs_A_pct"])
    for row in stage2["shape_sweep"]:
        shape_table.add_data(
            row["w"], row["median"],
            100.0 * (row["median"] - RECEIPT_A) / RECEIPT_A)
    run.log({"desk/shape_sweep": shape_table})

    level_table = wandb.Table(
        columns=["uniform_h", "ranked_median", "delta_vs_A_pct"])
    for row in stage2["uniform_sweep"]:
        level_table.add_data(
            row["h"], row["median"],
            100.0 * (row["median"] - RECEIPT_A) / RECEIPT_A)
    run.log({"desk/uniform_level_sweep": level_table})

    gamma_table = wandb.Table(columns=["prompt", "gamma"])
    for prompt, gamma in stage2["gamma"].items():
        gamma_table.add_data(prompt, gamma)
    run.log({"desk/fitted_gamma": gamma_table})

    validation = wandb.Table(
        columns=["receipt", "paid", "predicted", "error_pct",
                 "inside_receipt_channel"])
    for name, paid in (("cap4", PAID_CAP4), ("cap5", PAID_CAP5)):
        pred = stage2["out_of_sample"][name]
        err = 100.0 * (pred - paid) / paid
        validation.add_data(name, paid, pred, err,
                            abs(err) < RECEIPT_SIGMA_PCT)
    run.log({"desk/out_of_sample_validation": validation})

    if replay:
        witness = replay["witness"]
        run.summary["replay/depth_agreement"] = witness["depth_agreement"]
        run.summary["replay/sched_agreement_within_1ulp"] = witness["sched_agreement"]
        run.summary["replay/sched_byte_identical"] = witness["sched_byte_identical"]
        run.summary["replay/field_max_ulp"] = witness["field_max_ulp"]
        run.summary["replay/positive_control_max_ulp"] = witness[
            "positive_control_max_ulp"]
        run.summary["replay/rounds_checked"] = witness["rounds_checked"]
        run.summary["replay/walk_steps_checked"] = witness["steps_checked"]

        cost_table = wandb.Table(
            columns=["verify_width_m", "R_ranked_ms", "R_local_ms",
                     "local_samples", "local_filled"])
        local = replay["local_cost_table"]
        for index, ranked_ms in enumerate(replay["ranked_cost_table"]):
            m = index + 1
            cost_table.add_data(
                m, ranked_ms,
                local["table"][index] if local.get("table") else None,
                local.get("samples", {}).get(str(m)),
                m in local.get("filled", []))
        run.log({"replay/cost_tables": cost_table})
        replay_table = wandb.Table(
            columns=["cost_table", "arm", "ms_per_token", "speedup",
                     "edl_ship", "edl_cand", "shift_down", "shift_same",
                     "shift_up", "exact_token_accounting"])
        for key, row in replay["results"].items():
            table_name, arm = key.split("/", 1)
            replay_table.add_data(
                table_name, arm, row["cand_ms_per_token_pessimistic"],
                row["speedup_pessimistic"], row["ship_edl"], row["cand_edl"],
                row["shift_down"], row["shift_same"], row["shift_up"],
                row["exact_token_accounting"])
        run.log({"replay/arms": replay_table})
        if replay.get("live"):
            for key, value in replay["live"].items():
                if isinstance(value, (int, float)):
                    run.summary["replay/live_" + key] = value

    leg_table = wandb.Table(
        columns=["leg", "arm", "entry_c", "exit_c", "rounds_traced",
                 "serial_spt", "mtp_spt", "speedup", "edl",
                 "accepted_draft_rate", "residual_divergence_count",
                 "head_provenance_sha256", "worker_digest_stable",
                 "cool_gate_passed_real_gate", "gate_qualified_for_timing"])
    for leg in legs:
        meta, metrics = leg["meta"], leg["metrics"]
        leg_table.add_data(
            int(meta.get("e200_leg", 0)), meta.get("e200_arm", "?"),
            meta.get("gpu_temp_entry", ""), meta.get("gpu_temp_exit", ""),
            int(meta.get("rounds_traced", 0)),
            metrics.get("serial_seconds_per_token"),
            metrics.get("mtp_seconds_per_token"),
            metrics.get("mtp_decode_speedup"),
            metrics.get("effective_mean_draft_len"),
            metrics.get("accepted_draft_rate"),
            metrics.get("residual_divergence_count"),
            metrics.get("head_provenance_sha256", ""),
            meta.get("worker_digest_stable", ""),
            meta.get("cool_gate_passed_real_gate", ""),
            meta.get("gate_qualified_for_timing", ""))
    run.log({"legs/summary": leg_table})

    run.summary.update({
        "desk/ship_median_ranked": stage2["ship_median"],
        "desk/best_candidate_median_ranked": min(
            row["median"] for row in stage2["arms"].values()),
        "desk/best_shape_sweep_delta_pct": max(
            100.0 * (row["median"] - RECEIPT_A) / RECEIPT_A
            for row in stage2["shape_sweep"]),
        "verdict": "not-useful",
    })
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
