#!/usr/bin/env python3
"""Publish the E124 precision-island deletion study to W&B.

    usage: research/e124_wandb_log.py [--only RUN ...]

E124 asks whether the MTP proposal head's six BF16 `precision_islands.*`
tensors pay for the traffic they add. Four arms of one binary, selected by
`DARKBLOOM_QWEN_MTP_ISLAND_ARM`:

  `all`   Q, K and V islands installed. The shipped default.
  `kv`    K and V only. Q comes out of the affine-4 pack.
  `q`     Q only. K and V come out of the affine-4 pack.
  `none`  no islands at all.

Runs published here:

  `e124-stage0-census-price`
      Zero-GPU evidence: the six tensors in the head the harness actually
      loads, the file-versus-tree digest correction, the complete-permutation
      test that decides which branch of `installExactQKVRows` is live, the
      exact MLX op list per arm, and the four-arm repricing under campaign
      rule 69 class coefficients.
  `e124-stage1-acceptance`
      The acceptance exchange at 512 decode tokens, one leg per arm in one
      session with no rebuild between legs. Cluster-bootstrap intervals,
      per-position conditionals, the 0.21 pt kill decision per arm, and the
      cross-arm emitted-token exactness check.
  `e124-stage2-timing`
      Published only when Stage 1 leaves a surviving arm. ABBA-counterbalanced
      absolute candidate seconds per token against a fresh unchanged base in
      the same session.

Every timed leg here is local and ungated unless its own record says
otherwise, so `timing_valid`, `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` are logged false verbatim. No number here is an
official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e124-mtp-head-precision-island-deletion"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
OUT = pathlib.Path("research/out")

BASE_SHA = "3b8ea425f8887c9b5cd08ddfff6ddc423fb5d9c3"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
PR_NUMBER = 125
ENTRY_SITE = (
    "Qwen35TextModel.sanitize -> Qwen35Attention.installExactQKVRows, "
    "consumed by Qwen35Attention.qkv and Qwen35Attention.kv")
ARM_SELECTOR = "DARKBLOOM_QWEN_MTP_ISLAND_ARM"
ARMS = ("all", "none", "q", "kv")

KILL_LINE_PT = 0.21
BREAK_EVEN_PT = (0.36, 0.39)
STAGE0_BAR_RANKED_PCT = 0.20
BYTE_CLASS_RANKED = (0.24, 0.327)
DISPATCH_CLASS_RANKED = 0.95

ARM_NOTE = {
    "all": "Q, K and V islands installed; the shipped default",
    "none": "no islands; one affine-4 QKV pack and nothing else",
    "q": "Q island only; K and V read the affine-4 pack",
    "kv": "K and V islands only; Q reads the affine-4 pack",
}

CENSUS = OUT / "e124-head-census.json"
PRICE = OUT / "e124-price.json"
ACCEPT = OUT / "e124-acceptance.json"
TIMING = OUT / "e124-timing.json"


def gate_flags(valid: bool = False) -> dict[str, object]:
    return {
        "timing_valid": valid,
        "cool_gate_passed_real_gate": valid,
        "gate_qualified_for_timing": valid,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def load(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def start(name: str, job_type: str, question: str, config: dict, valid=False):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={
            "experiment": GROUP,
            "question": question,
            "entry_site": ENTRY_SITE,
            "arm_selector_env": ARM_SELECTOR,
            "arms": list(ARMS),
            "arm_note": ARM_NOTE,
            "host": HOST,
            "base_sha": BASE_SHA,
            "advisor_branch": ADVISOR_BRANCH,
            "pr_number": PR_NUMBER,
            "kill_line_acceptance_pt": KILL_LINE_PT,
            "break_even_acceptance_pt": list(BREAK_EVEN_PT),
            "stage0_bar_ranked_pct": STAGE0_BAR_RANKED_PCT,
            "byte_class_ranked_coefficient": list(BYTE_CLASS_RANKED),
            "dispatch_class_ranked_coefficient": DISPATCH_CLASS_RANKED,
            **config, **gate_flags(valid),
        },
        reinit=True,
    )


def attach(run, *paths: pathlib.Path) -> None:
    present = [p for p in paths if p.exists()]
    if not present:
        return
    artifact = wandb.Artifact(f"{run.name}-artifacts", type="analysis")
    for path in present:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def log_stage0() -> None:
    census, price = load(CENSUS), load(PRICE)
    if census is None or price is None:
        print("stage0: census or price artifact missing, skipped")
        return

    run = start(
        "e124-stage0-census-price", "analysis",
        "Do the six island tensors exist at the live pin, which install "
        "branch is live, and what is each arm worth once the byte model "
        "counts affine-4 scales and biases and the per-round KV flush?",
        {
            "stage": 0,
            "head_manifest_sha256": census["manifest"]["sha256"],
            "head_manifest_bytes": census["manifest"]["bytes"],
            "operating_point": price["operating_point"],
            "effective_stream_gbps": price["effective_stream_gbps_from_e87_law"],
        })

    heads = wandb.Table(
        columns=["head", "file_sha256", "tree_sha256", "tree_bytes",
                 "tree_sha256_matches_manifest", "tensor_count",
                 "island_bytes_total", "fast_branch_live"])
    tensors = wandb.Table(
        columns=["head", "projection", "kind", "dtype", "shape", "bytes"])
    perm = wandb.Table(
        columns=["head", "projection", "index_count", "output_count",
                 "unique", "min", "max", "is_complete_permutation"])
    for label, head in census["heads"].items():
        heads.add_data(
            label, head["file_sha256"], head["tree_sha256"],
            head["tree_bytes"], head["tree_sha256_matches_manifest"],
            head["tensor_count"], head["island_bytes_total"],
            head["fast_branch_live"])
        for proj, isle in head["islands"].items():
            if not isle["present"]:
                continue
            tensors.add_data(label, proj, "weight", isle["weight_dtype"],
                             str(isle["weight_shape"]), isle["weight_bytes"])
            tensors.add_data(label, proj, "indices", isle["indices_dtype"],
                             str(isle["indices_shape"]), isle["indices_bytes"])
            perm.add_data(label, proj, isle["indices_shape"][0],
                          isle["output_count"], isle["unique"], isle["min"],
                          isle["max"], isle["is_complete_permutation"])
    run.log({"head_trees": heads, "island_tensors": tensors,
             "complete_permutation": perm})

    loaded = census["heads"]["declared-run (E124 arms load this)"]
    run.summary["loaded_head_tree_sha256"] = loaded["tree_sha256"]
    run.summary["loaded_head_island_bytes"] = loaded["island_bytes_total"]
    run.summary["loaded_head_fast_branch_live"] = loaded["fast_branch_live"]

    pricing = wandb.Table(
        columns=["arm", "installs_q", "installs_kv", "qkv_bytes_per_step",
                 "kv_flush_bytes_per_round", "ops_per_step",
                 "delta_bytes_per_round", "delta_ops_per_round",
                 "byte_local_pct", "dispatch_local_pct_lo",
                 "dispatch_local_pct_hi", "local_pct_lo", "local_pct_hi",
                 "ranked_pct_lo", "ranked_pct_hi",
                 "ranked_pct_lo_after_realisation_discount",
                 "ranked_pct_hi_after_realisation_discount"])
    for arm in ARMS:
        row = price["arms"][arm]
        disc = row["total_ranked_pct_after_byte_realisation_discount"]
        pricing.add_data(
            arm, row["installs_q"], row["installs_kv"],
            row["qkv_bytes_per_draft_step"], row["kv_flush_bytes_per_round"],
            " + ".join(row["qkv_ops_per_draft_step"]),
            row["delta_bytes_per_round_vs_all"],
            row["delta_ops_per_round_vs_all"],
            row["byte_class_local_pct"],
            row["dispatch_class_local_pct"][0],
            row["dispatch_class_local_pct"][1],
            row["total_local_pct"][0], row["total_local_pct"][1],
            row["total_ranked_pct"][0], row["total_ranked_pct"][1],
            disc[0], disc[1])
        run.summary[f"ranked_pct_lo/{arm}"] = row["total_ranked_pct"][0]
        run.summary[f"ranked_pct_hi/{arm}"] = row["total_ranked_pct"][1]
    run.log({"repricing": pricing})

    best = max(ARMS, key=lambda a: price["arms"][a]["total_ranked_pct"][1])
    run.summary["best_arm_by_price"] = best
    run.summary["best_arm_ranked_pct_hi"] = (
        price["arms"][best]["total_ranked_pct"][1])
    run.summary["stage0_bar_cleared"] = (
        price["arms"][best]["total_ranked_pct"][1] >= STAGE0_BAR_RANKED_PCT)
    attach(run, CENSUS, PRICE)
    run.finish()


def log_stage1() -> None:
    accept = load(ACCEPT)
    if accept is None or len(accept["legs"]) < 2:
        print("stage1: acceptance artifact missing or single-leg, skipped")
        return

    control = accept["control"]
    run = start(
        "e124-stage1-acceptance", "measurement",
        "How much acceptance does each island arm lose against arm `all`, "
        "and does any arm stay inside the 0.21 pt kill line?",
        {
            "stage": 1,
            "decode_tokens": 512,
            "control_leg": control,
            "estimator": "cluster bootstrap over rounds, 20000 resamples",
            "bootstrap_seed": accept["seed"],
        })

    legs = wandb.Table(
        columns=["tag", "arm", "arm_witness", "rounds", "proposed",
                 "accepted", "acceptance", "ci95_lo", "ci95_hi",
                 "se_delta_method", "eff_draft_len", "verify_width_M",
                 "accepted_per_round", "mean_round_us",
                 "mean_draft_build_us", "mtp_s_per_token",
                 "serial_s_per_token", "local_ratio", "all_tokens_matched",
                 "residual_divergence", "gpu_temp_entry_c",
                 "gpu_temp_exit_c", "head_provenance_sha256"])
    for tag, leg in accept["legs"].items():
        legs.add_data(
            tag, leg["arm"], leg["arm_witness"], leg["rounds"],
            leg["proposed"], leg["accepted"], leg["acceptance_rate"],
            leg["acceptance_ci95_cluster_bootstrap"][0],
            leg["acceptance_ci95_cluster_bootstrap"][1],
            leg["acceptance_se_delta_method"],
            leg["effective_mean_draft_len_trace"], leg["verify_width_M"],
            leg["accepted_per_round"], leg["mean_round_us"],
            leg["mean_draft_build_us"], leg["mtp_seconds_per_token"],
            leg["serial_seconds_per_token"], leg["mtp_decode_speedup"],
            leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"],
            leg["head_provenance_sha256"])
        arm = leg["arm"]
        run.summary[f"acceptance/{arm}"] = leg["acceptance_rate"]
        run.summary[f"mtp_s_per_token/{arm}"] = leg["mtp_seconds_per_token"]
        run.summary[f"mean_round_us/{arm}"] = leg["mean_round_us"]
    run.log({"legs": legs})

    deltas = wandb.Table(
        columns=["arm", "d_acceptance_pt", "ci95_lo_pt", "ci95_hi_pt",
                 "kill_line_pt", "killed"])
    survivors = []
    for leg in accept["legs"].values():
        if "delta_acceptance_pt_vs_control" not in leg:
            continue
        arm, point = leg["arm"], leg["delta_acceptance_pt_vs_control"]
        lo, hi = leg["delta_acceptance_pt_ci95"]
        killed = leg["killed_by_acceptance"]
        deltas.add_data(arm, point, lo, hi, KILL_LINE_PT, killed)
        run.summary[f"d_acceptance_pt/{arm}"] = point
        run.summary[f"killed/{arm}"] = killed
        if not killed:
            survivors.append(arm)
    run.log({"acceptance_delta": deltas})

    conds = wandb.Table(
        columns=["arm", "position", "eligible", "accepted", "p", "se"])
    for leg in accept["legs"].values():
        for row in leg["conditionals"]:
            conds.add_data(leg["arm"], row["position"], row["eligible"],
                           row["accepted"], row["p"], row["se"])
    run.log({"per_position_conditionals": conds})

    exact = wandb.Table(
        columns=["arm", "rows", "top1_mismatch", "pos_mismatch",
                 "value_mismatch"])
    for leg in accept["legs"].values():
        exact.add_data(leg["arm"], leg["rows"],
                       leg["row_top1_mismatch_vs_control"],
                       leg["row_pos_mismatch_vs_control"],
                       leg["row_value_mismatch_vs_control"])
    run.log({"cross_arm_exactness": exact})

    run.summary["surviving_arms"] = survivors
    run.summary["all_arms_killed"] = not survivors
    attach(run, ACCEPT)
    run.finish()


def log_stage2() -> None:
    timing = load(TIMING)
    if timing is None:
        print("stage2: timing artifact missing, skipped")
        return

    run = start(
        "e124-stage2-timing", "measurement",
        "Does a surviving arm lower absolute candidate seconds per token "
        "against a fresh unchanged base measured in the same session?",
        {
            "stage": 2,
            "decode_tokens": 512,
            "schedule": timing["schedule"],
            "legs_per_arm": timing["legs_per_arm"],
            "rebuild_between_legs": False,
            "harness_defect_25_removed_by_construction": True,
        },
        valid=timing.get("gate_qualified_for_timing", False))

    legs = wandb.Table(
        columns=["tag", "arm", "position", "mtp_s_per_token", "mean_round_us",
                 "mean_draft_build_us", "acceptance", "local_ratio",
                 "gpu_temp_entry_c", "gpu_temp_exit_c", "all_tokens_matched"])
    for leg in timing["legs"]:
        legs.add_data(
            leg["tag"], leg["arm"], leg["position"],
            leg["mtp_seconds_per_token"], leg["mean_round_us"],
            leg["mean_draft_build_us"], leg["acceptance_rate"],
            leg["mtp_decode_speedup"], leg["gpu_temp_entry_c"],
            leg["gpu_temp_exit_c"], leg["all_tokens_matched"])
    run.log({"timing_legs": legs})

    effects = wandb.Table(
        columns=["arm", "d_mtp_s_per_token_pct", "se_pct", "sigma",
                 "d_round_us_pct", "entry_temp_spread_c", "advance"])
    for arm, row in timing["effects"].items():
        effects.add_data(arm, row["delta_spt_pct"], row["se_pct"],
                         row["sigma"], row["delta_round_pct"],
                         row["entry_temp_spread_c"], row["advance"])
        run.summary[f"delta_spt_pct/{arm}"] = row["delta_spt_pct"]
        run.summary[f"advance/{arm}"] = row["advance"]
    run.log({"arm_effects": effects})
    attach(run, TIMING)
    run.finish()


RUNS = {
    "stage0": log_stage0,
    "stage1": log_stage1,
    "stage2": log_stage2,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=sorted(RUNS))
    args = parser.parse_args()
    for name in args.only or sorted(RUNS):
        RUNS[name]()


if __name__ == "__main__":
    main()
