#!/usr/bin/env python3
"""E159 R1: publish the decode round's cost law and budget to W&B.

One run, five record groups.

  legs       every timed leg, with its honesty flags, temperatures, head
             provenance, exact round accounting and per-leg R.
  depth      R(D) at each pinned PROPOSED draft count, with a and q kept
             separate, alpha, non-drafting rounds and the fit residual.
  fit        R(D) = s + h*D, its standard errors, its quadratic term, and the
             pre-registered headline rho = 8h/s against the advisor's window.
  trace      the sync-head traced split of the round, and h measured in situ.
  budget     the four shares of the round at the adaptive operating point.

harness=local. Every leg is ungated by construction, so nothing here is a
ranked score and nothing here is gate-qualified.

Usage:
  python3 research/e159_wandb_log.py --run-name e159-r1-round-budget
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e159-artifacts"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
BASE_SHA = "1ded9d70d19e98742264b642455384c05e40d508"

# Pre-registered by the advisor in PR 159 comment e159-f3-rho-preregistration.
RHO_PREDICTED = (1.2, 1.6)
RHO_FEASIBLE = (0.0, 1.580)
S_FEASIBLE = (0.02925, 0.03365)
H_FEASIBLE = (0.0, 0.0057778)
RHO_GATES = ((0.32, "widen_producer_fusion"), (0.95, "run_both"))


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def table(columns, rows):
    return wandb.Table(columns=list(columns), data=[list(r) for r in rows])


def decision(rho: float) -> str:
    for threshold, verdict in RHO_GATES:
        if rho < threshold:
            return verdict
    return "pivot_to_head_cost"


def log_legs(run, doc: dict) -> None:
    columns = [
        "dir", "arm", "block", "pinned_depth", "tokens", "rounds",
        "rounds_identity", "accepted_draft_total", "non_drafting_round_count",
        "decode_seconds", "R_ms", "round_ms_trimmed", "first_round_ms",
        "a_accepted_per_round", "q_proposed_per_round",
        "alpha_accept_fraction", "mtp_seconds_per_token", "R_over_1_plus_a_ms",
        "all_tokens_matched", "residual_divergence_count",
        "head_provenance_sha256", "uses_pinned_mtp_head",
        "gpu_temp_entry_c", "gpu_temp_exit_c", "leg_wall_seconds",
    ]
    rows = [
        (
            leg["dir"], leg["arm"], leg["block"],
            -1 if leg["pinned_depth"] is None else leg["pinned_depth"],
            leg["tokens"], leg["rounds"], leg["rounds_identity"],
            leg["accepted_draft_total"], leg["non_drafting_round_count"],
            leg["decode_seconds"], 1e3 * leg["R_seconds"],
            1e-3 * leg["round_us_trimmed"], 1e-3 * leg["first_round_us"],
            leg["a_accepted_per_round"], leg["q_proposed_per_round"],
            leg["alpha_accept_fraction"], leg["mtp_seconds_per_token"],
            1e3 * leg["R_over_1_plus_a"], leg["all_tokens_matched"],
            leg["residual_divergence_count"], leg["head_provenance_sha256"],
            leg["uses_pinned_mtp_head"], leg["entry_c"], leg["exit_c"],
            leg["wall_seconds"],
        )
        for leg in doc["legs"]
    ]
    run.log({"e159/legs": table(columns, rows)})
    entries = [leg["entry_c"] for leg in doc["legs"]]
    run.summary.update({
        "e159_leg_count": len(doc["legs"]),
        "e159_all_tokens_matched": doc["all_tokens_matched"],
        "e159_entry_temp_spread_c": max(entries) - min(entries),
        "e159_entry_temp_min_c": min(entries),
        "e159_entry_temp_max_c": max(entries),
        "e159_head_provenance_sha256": ",".join(doc["head_provenance_sha256"]),
    })


def log_depth(run, doc: dict) -> None:
    columns = [
        "D", "legs", "R_ms", "R_spread_ms", "rounds", "accepted_draft_total",
        "a_accepted_per_round", "q_proposed_per_round", "alpha_accept_fraction",
        "realized_acceptance_a_over_D", "non_drafting_round_count",
        "mtp_seconds_per_token", "fit_R_ms", "residual_ms",
        "entry_c", "exit_c",
    ]
    rows = []
    for row in doc["depth_table"]:
        rows.append((
            row["D"], row["legs"], 1e3 * row["R_seconds_mean"],
            1e3 * row["R_seconds_spread"], json.dumps(row["rounds"]),
            json.dumps(row["accepted_draft_total"]),
            row["a_accepted_per_round"], row["q_proposed_per_round"],
            row["alpha_accept_fraction"],
            row["a_accepted_per_round"] / row["D"] if row["D"] else 0.0,
            json.dumps(row["non_drafting_round_count"]),
            row["mtp_seconds_per_token"], 1e3 * row["fit_R_seconds"],
            1e3 * row["residual_seconds"], json.dumps(row["entry_c"]),
            json.dumps(row["exit_c"]),
        ))
    run.log({"e159/depth/R_of_D": table(columns, rows)})
    for row in doc["depth_table"]:
        run.log({
            "e159/depth/D": row["D"],
            "e159/depth/R_ms": 1e3 * row["R_seconds_mean"],
            "e159/depth/a": row["a_accepted_per_round"],
            "e159/depth/q": row["q_proposed_per_round"],
            "e159/depth/alpha": row["alpha_accept_fraction"],
            "e159/depth/residual_ms": 1e3 * row["residual_seconds"],
        })


def log_fit(run, doc: dict) -> None:
    fit, curve = doc["fit"], doc["quadratic_term"]
    rho = doc["rho_8h_over_s"]
    s, h = doc["s_fixed_seconds"], doc["h_slope_seconds"]
    run.log({
        "e159/fit/residuals": table(
            ["point", "residual_ms"],
            [(i, 1e3 * r) for i, r in enumerate(doc["fit_residuals_seconds"])],
        )
    })
    run.summary.update({
        "e159_rho_8h_over_s": rho,
        "e159_rho_se": doc["rho_se"],
        "e159_rho_decision": decision(rho),
        "e159_rho_in_advisor_prediction": RHO_PREDICTED[0] <= rho <= RHO_PREDICTED[1],
        "e159_rho_in_board_feasible_range": RHO_FEASIBLE[0] <= rho <= RHO_FEASIBLE[1],
        "e159_rho_trimmed_round_clock": doc["rho_trimmed_round_clock"],
        "e159_s_fixed_seconds": s,
        "e159_s_se_seconds": fit["se_intercept"],
        "e159_s_in_board_feasible_range": S_FEASIBLE[0] <= s <= S_FEASIBLE[1],
        "e159_h_slope_seconds": h,
        "e159_h_se_seconds": fit["se_slope"],
        "e159_h_positive": h > 0.0,
        "e159_h_in_board_feasible_range": H_FEASIBLE[0] <= h <= H_FEASIBLE[1],
        "e159_fit_r_squared": fit["r_squared"],
        "e159_fit_residual_sd_ms": 1e3 * fit["residual_sd"],
        "e159_fit_n_points": fit["n"],
        "e159_quadratic_c2_ms": 1e3 * curve["c2"],
        "e159_quadratic_c2_sigma": curve["t_c2"],
        "e159_linear_within_two_sigma": abs(curve["t_c2"]) < 2.0,
        "e159_s_trimmed_seconds": doc["fit_trimmed_round_clock"]["intercept"],
        "e159_h_trimmed_seconds": doc["fit_trimmed_round_clock"]["slope"],
    })


def log_trace(run, doc: dict) -> None:
    traces = doc.get("trace_legs") or []
    if not traces:
        return
    columns = [
        "dir", "arm", "pinned_depth", "rounds", "round_us", "draft_build_us",
        "d_pre_us", "d_flush_us", "d_head1_us", "d_chain_us",
        "head_segment_us", "verify_build_us", "eval_wall_us", "readout_us",
        "commit_us", "upkeep_us", "session_tail_us",
    ]
    run.log({
        "e159/trace/legs": table(
            columns, [[t[c] for c in columns] for t in traces])
    })
    head_fit = doc.get("head_fit")
    if head_fit:
        run.summary.update({
            "e159_h_head_seconds": doc["h_head_seconds"],
            "e159_h_head_se_seconds": head_fit["se_slope"],
            "e159_h_head_r_squared": head_fit["r_squared"],
            "e159_t1_marginal_verify_seconds": doc[
                "t1_marginal_verify_seconds"],
            "e159_rho_head_only": doc["rho_head_only"],
            "e159_head_share_of_slope": (
                doc["h_head_seconds"] / doc["h_slope_seconds"]),
        })


def log_budget(run, doc: dict) -> None:
    adapt = doc.get("adaptive_operating_point")
    if not adapt:
        return
    shares = [
        ("mtp_session_plus_target_batch_1_fixed", adapt["fixed_share"]),
        ("drafting_total", adapt["drafting_share"]),
        ("proposal_head", adapt["head_share"]),
        ("extra_verify_rows", adapt["marginal_verify_share"]),
    ]
    run.log({
        "e159/budget/shares": table(
            ["component", "share_of_round"],
            [(name, value) for name, value in shares if value is not None],
        )
    })
    run.summary.update({
        "e159_adaptive_q_proposed": adapt["q_proposed_per_round"],
        "e159_adaptive_a_accepted": adapt["a_accepted_per_round"],
        "e159_adaptive_alpha": adapt["alpha_accept_fraction"],
        "e159_adaptive_R_ms": 1e3 * adapt["R_seconds"],
        "e159_adaptive_mtp_seconds_per_token": adapt["mtp_seconds_per_token"],
        "e159_budget_fixed_share": adapt["fixed_share"],
        "e159_budget_drafting_share": adapt["drafting_share"],
        "e159_budget_head_share": adapt["head_share"],
        "e159_budget_marginal_verify_share": adapt["marginal_verify_share"],
        "e159_R0_over_R_adaptive": adapt["R0_over_Radapt"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="e159-r1-round-budget")
    parser.add_argument("--doc", type=pathlib.Path,
                        default=ARTIFACTS / "e159_round_budget.json")
    args = parser.parse_args()

    doc = json.loads(args.doc.read_text())
    rho = doc["rho_8h_over_s"]

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=args.run_name,
        job_type="measurement",
        tags=["e159", "r1", "round-budget", "depth-sweep", "harness=local",
              "ungated"],
        config={
            "experiment": "e159-r1-round-budget",
            "rung": "r1",
            "pr": 159,
            "base_sha": BASE_SHA,
            "analysis_commit": git_sha(),
            "harness": "local",
            "official_or_ranked_score": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "thermal_mode": (
                "MLXFAST_LOCAL_COOL_GATE=0, ABBA-counterbalanced within one "
                "session, entry and exit GPU temperature recorded per leg"
            ),
            "decode_tokens": 512,
            "fixture": "correctness_prompts/public_longcopy_gate_english_512_256.json",
            "pinned_depths": [0, 1, 2, 4, 8],
            "depth_override_env": "MLX_E159_FIXED_DRAFT_DEPTH",
            "depth_override_submitted": False,
            "cost_model_depth_changed": False,
            "round_arithmetic": (
                "rounds = tokens - acceptedDraftTotal; R = decodeSeconds / "
                "rounds; a = acceptedDraftTotal / rounds; q = "
                "effective_mean_draft_len; mtp = R / (1 + a)"
            ),
            "rho_definition": "8 * slope / intercept of R(D) = s + h*D",
            "rho_predicted_low": RHO_PREDICTED[0],
            "rho_predicted_high": RHO_PREDICTED[1],
            "rho_gate_widen_producer_fusion_below": RHO_GATES[0][0],
            "rho_gate_pivot_to_head_above": RHO_GATES[1][0],
            "session": doc["session"],
            "trace_session": doc["trace_session"],
        },
    )
    log_legs(run, doc)
    log_depth(run, doc)
    log_fit(run, doc)
    log_trace(run, doc)
    log_budget(run, doc)
    run.summary.update({"e159_headline": f"rho = {rho:.4f} -> {decision(rho)}"})
    print(run.url, run.id, sep="\n")
    run.finish()


if __name__ == "__main__":
    main()
