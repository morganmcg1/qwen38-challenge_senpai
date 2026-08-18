#!/usr/bin/env python3
"""qwen38-r1-e25 r2: publish the forced-depth price curve and the realised-rate
optimum to W&B.

usage:
  research/e25r2_refit.py --out research/e25r2-pool.json
  research/e25r2_log_wandb.py research/e25r2-pool.json [--group G] [--name N]

Consumes the pooled report written by research/e25r2_refit.py. That report is
built from the TRUSTED PARENT's own per-round clock (`block_request_seconds`
paired with `effective_draft_lengths` in `reports/04-mtp-timed.json`), not from
the editable session's trace, so every published T(d) is measured on the clock
that produces the score and depth 0 is covered.

Three things in here are the actual result and are logged as first-class
summary metrics rather than buried in a table:

  e25r2/wall_depths            which measured prices exceed their 1/(d+1)
                               admissibility ceiling. Exactly one does, c_3,
                               which is why arm D is a hard DEEP_CAP = 3.
  e25r2/rate_argmax_depth      the realised-rate optimum over constant-depth
                               policies. It is 2-3, NOT the depth 7 I
                               pre-registered in r1 under a hypothetical p = 1.
  e25r2/greedy_leaves_on_table_pct
                               how much modelled global argmax beats modelled
                               greedy local ascent by. ~0, which is why no
                               Swift arm G was implemented.

The r1 headline is logged as config so the group shows the r2 diagnosis next to
the timed result it explains, and every projection is anchored on the LIVE
promoted bar rather than on r1's stale frontier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e25-per-row-draft-price"

BASE_SHA = "d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602"
R1_BASE_SHA = "0d2eef9cac75d890de06a5eef4fd686c3c34c1ef"
PROMOTED_FRONTIER_SHA = "474c75013f333f119bdc465d849f23917b195b20"
PROMOTED_SUBMISSION_ID = "942e5ab2-1c46-4c50-b7c3-eaf948878ed0"
LIVE_RANKED_BAR = 3.2341518328631

# research/e25-build.sh installs these and records sha256.txt per arm. Swift
# release builds are not bit-reproducible on this host, so arm identity rides
# on the source blob digest; the binary digest is recorded for completeness.
ARM_SOURCE_BLOBS = {
    "BASE": "7ce81abe55275f6b25712026ba3d5908396ebe30",
    "PRICE": "64673bab9c024d38639c291d71269be8ac0046ef",
    "FORCE": "7ce81abe55275f6b25712026ba3d5908396ebe30+patch",
}
FORCE_PATCH_SHA256 = \
    "0ff4ab291f4f2743a361814271563ab459e90cad1059ffe1ccc58b25d0602ed8"

# Manifest digest is the SINGLE-FILE tree digest. The local run tree adds the
# organizer head-family config.json (3570 B) because benchmark-qwen-mtp.sh:215
# requires a non-empty one, so the observed tree digest differs by that file
# alone; model.safetensors is hardlinked and byte-identical.
DECLARED_HEAD = {
    "manifest_tree_sha256":
        "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71",
    "safetensors_sha256":
        "d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1",
    "safetensors_bytes": 427742600,
    "local_run_tree_sha256":
        "dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9",
    "local_run_tree_adds": "config.json (3570 B, fb2a5bd0..., organizer head family)",
    "r1_stale_head_bytes": 238934129,
}

R1_HEADLINE = {
    "r1_metric": "e25/mtp_true_decode_gain_pct_median_of_8",
    "r1_median_of_8_pct": 3.8346226261260976,
    "r1_wandb_run": "ilsehgqd",
    "r1_prompts_won": "8/8",
    "r1_row_step_ratio": [0.0, 0.095904, 0.152261, 0.442442],
}


def flat_time(report: dict, key: str, prefix: str) -> dict:
    out = {}
    for d, v in report[key]["round_ms"].items():
        out[f"{prefix}/T_mean_ms/d{d}"] = v["mean_ms"]
        out[f"{prefix}/T_sem_ms/d{d}"] = v["sem_ms"]
        out[f"{prefix}/T_median_ms/d{d}"] = v["median_ms"]
        out[f"{prefix}/n/d{d}"] = v["n"]
    return out


def flat_admissibility(adm: dict, prefix: str) -> tuple[dict, list[int]]:
    out, walls = {}, []
    for d, v in adm.items():
        out[f"{prefix}/c/d{d}"] = v["measured_c"]
        out[f"{prefix}/ceiling/d{d}"] = v["ceiling"]
        out[f"{prefix}/shipped_scalar_c/d{d}"] = v["shipped_scalar_c"]
        out[f"{prefix}/admissible/d{d}"] = v["admissible"]
        if not v["admissible"]:
            walls.append(int(d))
    return out, sorted(walls)


def flat_rate(rate: dict, prefix: str) -> dict:
    out = {}
    for d, v in rate["by_depth"].items():
        out[f"{prefix}/tokens_per_round/d{d}"] = v["mean_tokens"]
        out[f"{prefix}/tokens_per_round_sem/d{d}"] = v["sem_tokens"]
        out[f"{prefix}/ms_per_token/d{d}"] = v["ms_per_token"]
        out[f"{prefix}/tokens_per_ms/d{d}"] = v["tokens_per_ms"]
        out[f"{prefix}/tokens_per_ms_sem/d{d}"] = v["sem_tokens_per_ms"]
        out[f"{prefix}/rate_vs_argmax_pct/d{d}"] = rate["rate_vs_argmax_pct"][d]
    for d, share in rate["bootstrap_argmax_share"].items():
        out[f"{prefix}/bootstrap_argmax_share/d{d}"] = share
    return out


def quartile_order_holds(report: dict) -> bool:
    """True when T(d) is non-decreasing in d within every position quartile."""
    for per_depth in report["position_control"]["by_quartile"].values():
        depths = sorted(per_depth, key=int)
        means = [per_depth[d]["mean_ms"] for d in depths if per_depth[d]["n"] > 0]
        if any(b < a for a, b in zip(means, means[1:])):
            return False
    return True


def tables(report: dict) -> dict:
    t = report["time_parent_clock"]["round_ms"]
    adm = report["admissibility_parent"]
    rate = report["rate_empirical"]["by_depth"]
    vs_argmax = report["rate_empirical"]["rate_vs_argmax_pct"]

    curve = wandb.Table(columns=[
        "depth", "n", "T_mean_ms", "T_sem_ms", "T_median_ms",
        "weight_stream_passes", "c_d", "ceiling", "admissible",
        "tokens_per_round", "sem_tokens", "ms_per_token", "rate_vs_argmax_pct",
    ])
    for d in sorted(t, key=int):
        v, r = t[d], rate.get(d, {})
        a = adm.get(d, {})
        curve.add_data(int(d), v["n"], v["mean_ms"], v["sem_ms"], v["median_ms"],
                       v["weight_passes"], a.get("measured_c"), a.get("ceiling"),
                       a.get("admissible"), r.get("mean_tokens"),
                       r.get("sem_tokens"), r.get("ms_per_token"),
                       vs_argmax.get(d))

    accept = wandb.Table(columns=["row_index", "reached", "accepted", "p", "sem"])
    for i, v in sorted(report["acceptance"].items(), key=lambda kv: int(kv[0])):
        accept.add_data(int(i), v["reached"], v["accepted"], v["p"], v["sem"])

    byq = report["position_control"]["by_quartile"]
    quart = wandb.Table(columns=["depth", "q0_ms", "q1_ms", "q2_ms", "q3_ms"])
    for d in sorted(t, key=int):
        quart.add_data(int(d), *[byq[f"q{q}"].get(d, {}).get("mean_ms")
                                 for q in range(4)])

    fid = wandb.Table(columns=[
        "prompt", "all_tokens_matched", "parity_all_ok",
        "residual_divergence_count", "decode_token_count",
        "emitted_token_total", "round_robin_speedup", "head_origin",
        "head_bytes", "head_file_count",
    ])
    for p, v in report["fidelity"].items():
        fid.add_data(p, v["all_tokens_matched"], v["parity_all_ok"],
                     v["residual_divergence_count"], v["decode_token_count"],
                     v["emitted_token_total"], v.get("round_robin_speedup"),
                     v.get("head_origin"), v.get("head_bytes"),
                     v.get("head_file_count"))

    return {"e25r2/depth_cost_curve": curve,
            "e25r2/conditional_acceptance": accept,
            "e25r2/position_quartiles": quart,
            "e25r2/fidelity_and_head": fid}


def policy_payload(policy: dict) -> tuple[dict, dict]:
    """Offline replay of the whole price design space on one fixed tape."""
    arms = policy["arms"]
    base = arms["base_shipped_h0.18"]["ms_per_token"]
    table = wandb.Table(columns=[
        "arm", "ms_per_token", "gain_vs_shipped_pct", "mean_depth", "rounds",
        "emitted_tokens", "depth_histogram",
    ])
    for name in sorted(arms, key=lambda a: arms[a]["ms_per_token"]):
        row = arms[name]
        table.add_data(name, row["ms_per_token"],
                       100.0 * (base - row["ms_per_token"]) / base,
                       row["mean_depth"], row["rounds"], row["emitted_tokens"],
                       json.dumps(row["depth_histogram"], sort_keys=True))

    best = min(arms, key=lambda a: arms[a]["ms_per_token"])
    cal = policy["calibration"]
    obs = policy["observed_runtime_depths"]
    validation = policy["replay_validation"]
    summary = {
        "e25r2/policy_rounds": policy["rounds"],
        "e25r2/policy_replay_exact": validation["replay_exact"],
        "e25r2/policy_best_arm": best,
        "e25r2/policy_best_gain_vs_shipped_pct":
            100.0 * (base - arms[best]["ms_per_token"]) / base,
        "e25r2/policy_shipped_ms_per_token": base,
        # arm D recomposed on the refit curve is numerically identical to the
        # shipped rule truncated at depth 3: deliverable (a), a second way.
        "e25r2/policy_arm_d_refit_equals_deep_cap_3": (
            arms["arm_d_refit_measured"]["depth_histogram"]
            == arms["base_shipped_deep_cap_3"]["depth_histogram"]
            and arms["arm_d_refit_measured"]["ms_per_token"]
            == arms["base_shipped_deep_cap_3"]["ms_per_token"]),
        "e25r2/policy_belief_ratio": cal["ratio"],
        "e25r2/policy_belief_shrink": policy["belief_shrink_applied"],
        "e25r2/observed_max_depth": obs["max_depth_observed"],
        "e25r2/observed_rounds_at_depth_ge_4": obs["rounds_at_depth_ge_4"],
        "e25r2/observed_mean_depth": obs["mean_depth"],
    }
    for name, row in arms.items():
        summary[f"e25r2/policy_arm/{name}/ms_per_token"] = row["ms_per_token"]
        summary[f"e25r2/policy_arm/{name}/mean_depth"] = row["mean_depth"]
    return {"e25r2/policy_arms": table}, summary


def timed_payload(timed: dict) -> tuple[dict, dict]:
    """Matched ABBA BASE vs PRICE legs: the experiment's primary metric."""
    head = timed["headline"]
    per_prompt = timed["per_prompt"]
    table = wandb.Table(columns=[
        "prompt", "base_ms_per_token", "candidate_ms_per_token", "gain_pct",
        "serial_control_delta_pct", "base_mean_draft_len",
        "candidate_mean_draft_len", "base_max_draft_len",
        "candidate_max_draft_len", "base_local_ratio", "candidate_local_ratio",
        "base_rounds", "candidate_rounds", "base_accepted_rate",
        "candidate_accepted_rate",
    ])
    for prompt in sorted(per_prompt):
        row = per_prompt[prompt]
        b, c = row["base"], row["candidate"]
        table.add_data(
            prompt, b["decode_ms_per_token"], c["decode_ms_per_token"],
            row["gain_pct"], row["serial_delta_pct"],
            b["counters"]["effective_mean_draft_len"],
            c["counters"]["effective_mean_draft_len"],
            b["counters"]["effective_max_draft_len"],
            c["counters"]["effective_max_draft_len"],
            b["local_ratio"], c["local_ratio"],
            b["counters"]["round_count"], c["counters"]["round_count"],
            b["counters"]["accepted_draft_rate"],
            c["counters"]["accepted_draft_rate"])

    drift = timed["host_drift_control"]
    summary = {
        # the assignment's primary metric, logged under its contract name
        "e25/mtp_true_decode_gain_pct_median_of_8": head["median_gain_pct"],
        "e25r2/timed_mean_gain_pct": head["mean_gain_pct"],
        "e25r2/timed_min_gain_pct": head["min_gain_pct"],
        "e25r2/timed_max_gain_pct": head["max_gain_pct"],
        "e25r2/timed_prompts_improved": head["prompts_improved"],
        "e25r2/timed_n_prompts": head["n_prompts"],
        "e25r2/timed_gates_pass": timed["gates"]["all_pass"],
        "e25r2/timed_gate_failure_count": len(timed["gates"]["failures"]),
        "e25r2/timed_max_abs_serial_drift_pct": drift["max_abs_serial_delta_pct"],
        "e25r2/timed_median_abs_serial_drift_pct":
            drift["median_abs_serial_delta_pct"],
        "e25r2/timed_vs_r1_headline_pct_points":
            head["median_gain_pct"] - R1_HEADLINE["r1_median_of_8_pct"]
            if head["median_gain_pct"] is not None else None,
    }
    for prompt, gain in head["per_prompt_gain_pct"].items():
        summary[f"e25r2/timed_gain_pct/{prompt}"] = gain
    for prompt, delta in drift["serial_delta_pct"].items():
        summary[f"e25r2/timed_serial_drift_pct/{prompt}"] = delta
    return {"e25r2/timed_pairs": table}, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--name", default="e25-r2-forced-depth-price-curve")
    ap.add_argument("--notes", default=(
        "E25 r2: forced-depth legs d=0..7 on the trusted parent's own clock. "
        "Proves arm D is a hard DEEP_CAP=3 (c_3 = 0.44 > 1/4) and that the "
        "wall is intrinsic to the measured weight-stream pass cliff at M=5, "
        "not to arm D's max(). The realised-rate optimum over constant-depth "
        "policies is d=2-3, refuting my own r1 depth-7 pre-registration."))
    ap.add_argument("--policy", type=Path,
                    help="research/e25r2-policy.json offline replay report")
    ap.add_argument("--timed", type=Path,
                    help="research/e25r2-timed.json matched BASE vs PRICE report")
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    prompts = report["prompts"]

    config = {
        "assignment": "qwen38-r1-e25-per-row-draft-price",
        "revision": "r2",
        "pr": 29,
        "student": "qwen-edward",
        "credit": "thorfinn E22 follow-up #1 (two-piece boundary-aware price)",
        "base_sha": BASE_SHA,
        "r1_base_sha": R1_BASE_SHA,
        "promoted_frontier_sha": PROMOTED_FRONTIER_SHA,
        "promoted_submission_id": PROMOTED_SUBMISSION_ID,
        "live_ranked_bar": LIVE_RANKED_BAR,
        "arm": report["arm"],
        "arm_source_blobs": ARM_SOURCE_BLOBS,
        "force_patch_sha256": FORCE_PATCH_SHA256,
        "declared_head": DECLARED_HEAD,
        "prompts": prompts,
        "n_prompts": len(prompts),
        "decode_tokens_per_leg": 512,
        "seed_tokens_per_leg": 512,
        "offered_depth": 8,
        "forced_depth_cycle": [0, 1, 2, 3, 4, 5, 6, 7],
        "warmup_rounds_dropped_per_prompt":
            report["warmup_rounds_dropped_per_prompt"],
        "primary_timer": "parent block_request_seconds x effective_draft_lengths",
        "host": "Apple M4 Pro / Mac16,11 / applegpu_g16s (NOT the ranked M5)",
        "cool_gate_bypassed": True,
        "ipg_by_m": {1: None, 2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3},
        **R1_HEADLINE,
    }

    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name,
                     group=args.group, job_type="analysis", config=config,
                     notes=args.notes)

    adm_parent, walls = flat_admissibility(report["admissibility_parent"],
                                           "e25r2/price_mean")
    adm_median, walls_median = flat_admissibility(
        report["admissibility_parent_median"], "e25r2/price_median")

    rate = report["rate_empirical"]
    modelled = report["rate_modelled"]
    fit = report["pass_count_model"]
    replay = report["replay"]
    inst = report["instrument"]
    fid = report["fidelity"]

    summary = {
        "e25r2/rounds_total": report["rounds_total"],
        "e25r2/rounds_analysed": report["rounds_analysed"],
        "e25r2/n_prompts": len(prompts),

        # --- the DEEP_CAP proof, as numbers ---
        "e25r2/wall_depths": json.dumps(walls),
        "e25r2/wall_depths_median_T": json.dumps(walls_median),
        "e25r2/n_wall_depths": len(walls),
        "e25r2/c3_measured": report["admissibility_parent"]["3"]["measured_c"],
        "e25r2/c3_ceiling": report["admissibility_parent"]["3"]["ceiling"],
        "e25r2/c3_replicates_r1": abs(
            report["admissibility_parent"]["3"]["measured_c"] - 0.442442) < 0.01,
        "e25r2/arm_d_is_deep_cap_3": 3 in walls,

        # --- the realised-rate optimum ---
        "e25r2/rate_argmax_depth": rate["argmax_depth"],
        "e25r2/rate_argmax_depth_median_T": rate["argmax_depth_median_T"],
        "e25r2/rate_modelled_argmax_depth": modelled["global_argmax_depth"],
        "e25r2/greedy_first_local_max": modelled["greedy_first_local_max_depth"],
        "e25r2/greedy_leaves_on_table_pct": modelled["greedy_leaves_on_table_pct"],
        "e25r2/depth7_rate_vs_argmax_pct": rate["rate_vs_argmax_pct"]["7"],
        "e25r2/r1_preregistered_argmax_depth": 7,
        "e25r2/r1_prereg_refuted": rate["argmax_depth"] != 7,

        # --- the mechanism ---
        "e25r2/pass_model_intercept_ms": fit["intercept_ms"],
        "e25r2/pass_model_per_row_ms": fit["per_row_ms"],
        "e25r2/pass_model_per_weight_pass_ms": fit["per_weight_pass_ms"],
        "e25r2/pass_model_r2": fit["r_squared"],
        "e25r2/pass_model_max_abs_resid_ms": fit["max_abs_residual_ms"],

        # --- confound controls ---
        "e25r2/position_drift_ms_per_1000_tokens":
            report["position_control"]["ols_ms_per_1000_tokens"],
        "e25r2/depth_order_replicates_all_quartiles": quartile_order_holds(report),
        "e25r2/timer_max_disagreement_ms": max(
            abs(v["core_minus_parent"])
            for v in report["timer_agreement_ms"].values()),

        # --- instrument gate ---
        "e25r2/rounds_carrying_forced_mark": inst["rounds_carrying_forced_mark"],
        "e25r2/taken_depth_equals_forced": inst["taken_depth_equals_forced"],
        "e25r2/forced_mark_disagreements":
            inst["rounds_carrying_forced_mark"]
            - inst["taken_depth_equals_forced"],

        # --- policy replay ---
        "e25r2/replay_shipped_ms_per_token":
            replay["shipped_counterfactual"]["ms_per_token"],
        "e25r2/replay_shipped_mean_depth":
            replay["shipped_counterfactual"]["mean_depth"],
        "e25r2/replay_forced_ms_per_token":
            replay["forced_as_run"]["ms_per_token"],
        "e25r2/replay_arm_g_ms_per_token":
            replay["arm_g_global_argmax"]["ms_per_token"],
        "e25r2/replay_arm_g_mean_depth":
            replay["arm_g_global_argmax"]["mean_depth"],
        "e25r2/arm_g_gain_over_shipped_pct": 100.0 * (
            replay["shipped_counterfactual"]["ms_per_token"]
            - replay["arm_g_global_argmax"]["ms_per_token"])
            / replay["shipped_counterfactual"]["ms_per_token"],

        # --- fidelity, every leg ---
        "e25r2/all_prompts_tokens_matched":
            all(v["all_tokens_matched"] for v in fid.values()),
        "e25r2/all_prompts_parity_ok":
            all(v["parity_all_ok"] for v in fid.values()),
        "e25r2/max_residual_divergence_count":
            max(v["residual_divergence_count"] for v in fid.values()),
        "e25r2/all_prompts_declared_head_q2q4": all(
            "q2-q4-rerank-v1" in (v.get("head_origin") or "")
            for v in fid.values()),
        "e25r2/all_row_ledgers_closed": all(
            v["rows_closed"] for v in report["ledger_crosscheck"].values()),
    }
    summary.update(adm_parent)
    summary.update(adm_median)
    summary.update(flat_time(report, "time_parent_clock", "e25r2/parent"))
    summary.update(flat_time(report, "time_core", "e25r2/trace_core"))
    summary.update(flat_rate(rate, "e25r2/rate"))
    summary["e25r2/gain_over_depth3_pct"] = rate["gain_over_depth3_pct"]
    for p, d in report["per_prompt_argmax"].items():
        summary[f"e25r2/per_prompt_argmax/{p}"] = d

    logged = tables(report)
    for path, builder in ((args.policy, policy_payload),
                          (args.timed, timed_payload)):
        if path is None:
            continue
        extra_tables, extra_summary = builder(json.loads(path.read_text()))
        logged.update(extra_tables)
        summary.update(extra_summary)

    run.summary.update(summary)
    run.log(logged)

    artifact = wandb.Artifact("e25r2-forced-depth-pool", type="analysis")
    artifact.add_file(str(args.report))
    for path in (args.policy, args.timed):
        if path is not None:
            artifact.add_file(str(path))
    run.log_artifact(artifact)

    print(f"logged {run.url}")
    print(f"  walls={walls} rate_argmax={rate['argmax_depth']} "
          f"greedy_leaves={modelled['greedy_leaves_on_table_pct']:+.3f}%")
    run.finish()


if __name__ == "__main__":
    main()
