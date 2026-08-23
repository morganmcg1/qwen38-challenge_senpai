#!/usr/bin/env python3
"""E159 R1: publish the decode round's cost law and budget to W&B.

One run, seven record groups.

  legs       every timed leg, with its honesty flags, temperatures, head
             provenance, exact round accounting, seed prefill and per-leg R.
  depth      R(D) at each pinned PROPOSED draft count, for the prescribed
             depths and for the dense 0..8 sweep, with a and q kept separate,
             alpha, realized acceptance, non-drafting rounds and the residual.
  segments   the marginal price of each extra proposed draft.
  fit        R(D) = s + h*D, its standard errors, its quadratic term, and the
             pre-registered headline rho = 8h/s against the advisor's window,
             plus every rho variant the data supports.
  width      the verify-width step table and the width-6 wall.
  trace      the sync-head traced split of the round, and h measured in situ.
  budget     the four shares of the round at the adaptive operating point.

R is the DECODE round cost: the per-leg seed prefill is removed before the
fit, because prefill is paid once per leg and not once per round.

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
        "decode_seconds", "seed_prefill_seconds", "R_decode_ms",
        "R_prescribed_ms", "prefill_per_round_ms", "round_ms_trimmed",
        "first_round_ms", "a_accepted_per_round", "q_proposed_per_round",
        "rejected_per_round", "alpha_accept_fraction", "mtp_seconds_per_token",
        "R_over_1_plus_a_ms", "draft_length_histogram", "all_tokens_matched",
        "residual_divergence_count", "head_provenance_sha256",
        "uses_pinned_mtp_head", "gpu_temp_entry_c", "gpu_temp_exit_c",
        "leg_wall_seconds", "git_head", "worker_sha256",
    ]
    rows = [
        (
            leg["dir"], leg["arm"], leg["block"],
            -1 if leg["pinned_depth"] is None else leg["pinned_depth"],
            leg["tokens"], leg["rounds"], leg["rounds_identity"],
            leg["accepted_draft_total"], leg["non_drafting_round_count"],
            leg["decode_seconds"], leg["seed_prefill_seconds"],
            1e3 * leg["R_decode_seconds"], 1e3 * leg["R_seconds"],
            1e3 * leg["prefill_per_round_seconds"],
            1e-3 * leg["round_us_trimmed"], 1e-3 * leg["first_round_us"],
            leg["a_accepted_per_round"], leg["q_proposed_per_round"],
            leg["rejected_per_round"], leg["alpha_accept_fraction"],
            leg["mtp_seconds_per_token"], 1e3 * leg["R_over_1_plus_a"],
            json.dumps(leg["draft_length_histogram"]),
            leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["head_provenance_sha256"], leg["uses_pinned_mtp_head"],
            leg["entry_c"], leg["exit_c"], leg["wall_seconds"],
            leg["git_head"], leg["worker_sha256"],
        )
        for leg in doc["legs"]
    ]
    run.log({"e159/legs": table(columns, rows)})
    entries = [leg["entry_c"] for leg in doc["legs"]]
    exits = [leg["exit_c"] for leg in doc["legs"]]
    prefills = [leg["seed_prefill_seconds"] for leg in doc["legs"]]
    run.summary.update({
        "e159_leg_count": len(doc["legs"]),
        "e159_all_tokens_matched": doc["all_tokens_matched"],
        "e159_entry_temp_spread_c": max(entries) - min(entries),
        "e159_entry_temp_min_c": min(entries),
        "e159_entry_temp_max_c": max(entries),
        "e159_exit_temp_min_c": min(exits),
        "e159_exit_temp_max_c": max(exits),
        "e159_seed_prefill_seconds_mean": sum(prefills) / len(prefills),
        "e159_head_provenance_sha256": ",".join(doc["head_provenance_sha256"]),
    })


DEPTH_COLUMNS = [
    "D", "verify_width", "legs", "R_decode_ms", "R_decode_spread_ms",
    "R_prescribed_ms", "prefill_per_round_ms", "rounds",
    "accepted_draft_total", "a_accepted_per_round", "q_proposed_per_round",
    "rejected_per_round", "alpha_accept_fraction",
    "realized_acceptance_a_over_D", "implied_uniform_p",
    "non_drafting_round_count", "mtp_seconds_per_token", "fit_R_ms",
    "residual_ms", "entry_c", "exit_c",
]


def depth_rows(rows):
    return [
        (
            row["D"], row["D"] + 1, row["legs"],
            1e3 * row["R_decode_seconds_mean"],
            1e3 * row["R_decode_seconds_spread"],
            1e3 * row["R_seconds_mean"],
            1e3 * row["prefill_per_round_seconds"],
            json.dumps(row["rounds"]),
            json.dumps(row["accepted_draft_total"]),
            row["a_accepted_per_round"], row["q_proposed_per_round"],
            row["rejected_per_round"], row["alpha_accept_fraction"],
            row["realized_acceptance_a_over_D"], row["implied_uniform_p"],
            json.dumps(row["non_drafting_round_count"]),
            row["mtp_seconds_per_token"], 1e3 * row["fit_R_seconds"],
            1e3 * row["residual_seconds"], json.dumps(row["entry_c"]),
            json.dumps(row["exit_c"]),
        )
        for row in rows
    ]


def log_depth(run, doc: dict) -> None:
    run.log({
        "e159/depth/R_of_D_prescribed": table(
            DEPTH_COLUMNS, depth_rows(doc["depth_table"])),
        "e159/depth/R_of_D_dense": table(
            DEPTH_COLUMNS, depth_rows(doc["dense_depth_table"])),
    })
    for row in doc["dense_depth_table"]:
        run.log({
            "e159/depth/D": row["D"],
            "e159/depth/R_decode_ms": 1e3 * row["R_decode_seconds_mean"],
            "e159/depth/a": row["a_accepted_per_round"],
            "e159/depth/q": row["q_proposed_per_round"],
            "e159/depth/alpha": row["alpha_accept_fraction"],
            "e159/depth/mtp_ms": 1e3 * row["mtp_seconds_per_token"],
            "e159/depth/residual_ms": 1e3 * row["residual_seconds"],
        })
    best = min(doc["dense_depth_table"],
               key=lambda r: r["mtp_seconds_per_token"])
    run.summary.update({
        "e159_dense_best_depth_D": best["D"],
        "e159_dense_best_mtp_seconds_per_token": best["mtp_seconds_per_token"],
        "e159_rule79_not_evidence": True,
    })


def log_segments(run, doc: dict) -> None:
    columns = [
        "from_D", "to_D", "delta_R_ms", "marginal_ms_per_draft",
        "delta_rejected_per_round",
    ]
    for key, name in (
        ("segment_marginals", "e159/segments/prescribed"),
        ("dense_segment_marginals", "e159/segments/dense"),
    ):
        rows = [
            (s["from_D"], s["to_D"], 1e3 * s["delta_R_seconds"],
             1e3 * s["marginal_seconds_per_draft"],
             s["delta_rejected_per_round"])
            for s in doc[key]
        ]
        run.log({name: table(columns, rows)})


def log_fit(run, doc: dict) -> None:
    fit, curve = doc["fit"], doc["quadratic_term"]
    prescribed_fit = doc["fit_prescribed_includes_prefill"]
    rho = doc["rho_8h_over_s"]
    s, h = doc["s_fixed_seconds"], doc["h_slope_seconds"]
    variants = doc["rho_variants"]
    chord = doc["chord_estimate"]
    run.log({
        "e159/fit/residuals": table(
            ["point", "residual_ms"],
            [(i, 1e3 * r) for i, r in enumerate(doc["fit_residuals_seconds"])],
        ),
        "e159/fit/rho_variants": table(
            ["variant", "rho", "note"],
            [
                ("ols_prescribed_depths_0_8",
                 variants["ols_prescribed_depths_0_8"],
                 "headline: OLS on the prescribed depths, prefill removed"),
                ("ols_dense_0_8", variants["ols_dense_0_8"],
                 "OLS on every dense depth 0..8"),
                ("ols_dense_0_7_shipped_cap",
                 variants["ols_dense_0_7_shipped_cap"],
                 "OLS truncated at the shipped depth cap 7"),
                ("chord_0_8", variants["chord_0_8"]["rho"],
                 "chord: (R(8)-R(0))/8 over R(0)"),
                ("chord_0_7_shipped_cap",
                 variants["chord_0_7_shipped_cap"]["rho"],
                 "chord truncated at the shipped depth cap 7"),
                ("trimmed_round_clock", variants["trimmed_round_clock"],
                 "OLS on the trimmed in-session round clock"),
                ("head_only", doc["rho_head_only"],
                 "8*h_head/s: the head's share of the marginal price"),
            ],
        ),
    })
    run.summary.update({
        "e159_rho_8h_over_s": rho,
        "e159_rho_se": doc["rho_se"],
        "e159_rho_decision": decision(rho),
        "e159_rho_in_advisor_prediction": (
            RHO_PREDICTED[0] <= rho <= RHO_PREDICTED[1]),
        "e159_rho_in_board_feasible_range": (
            RHO_FEASIBLE[0] <= rho <= RHO_FEASIBLE[1]),
        "e159_rho_trimmed_round_clock": variants["trimmed_round_clock"],
        "e159_rho_ols_dense_0_8": variants["ols_dense_0_8"],
        "e159_rho_ols_dense_0_7_shipped_cap": variants[
            "ols_dense_0_7_shipped_cap"],
        "e159_rho_chord_0_8": variants["chord_0_8"]["rho"],
        "e159_rho_chord_0_7_shipped_cap": variants[
            "chord_0_7_shipped_cap"]["rho"],
        "e159_rho_prescribed_includes_prefill": doc[
            "rho_prescribed_includes_prefill"],
        "e159_chord_s_seconds": chord["s_seconds"],
        "e159_chord_h_seconds": chord["h_seconds"],
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
        "e159_fit_prescribed_intercept_seconds": prescribed_fit["intercept"],
        "e159_fit_prescribed_slope_seconds": prescribed_fit["slope"],
        "e159_quadratic_c2_ms": 1e3 * curve["c2"],
        "e159_quadratic_c2_sigma": curve["t_c2"],
        "e159_linear_within_two_sigma": abs(curve["t_c2"]) < 2.0,
        "e159_linear_model_rejected": doc["linear_model_rejected"],
        "e159_s_trimmed_seconds": doc["fit_trimmed_round_clock"]["intercept"],
        "e159_h_trimmed_seconds": doc["fit_trimmed_round_clock"]["slope"],
    })


def log_width_wall(run, doc: dict) -> None:
    wall = doc.get("width_wall")
    if not wall:
        return
    steps = wall["steps_by_verify_width"]
    run.log({
        "e159/width/steps": table(
            ["verify_width", "step_ms", "is_wall"],
            [(int(w), 1e3 * v, int(w) == wall["wall_verify_width"])
             for w, v in sorted(steps.items(), key=lambda kv: int(kv[0]))],
        ),
        "e159/width/adaptive_draft_lengths": table(
            ["draft_length", "rounds"],
            [(int(k), v) for k, v in
             sorted(wall["adaptive_draft_length_histogram"].items(),
                    key=lambda kv: int(kv[0]))],
        ),
    })
    run.summary.update({
        "e159_wall_depth_D": wall["wall_depth_D"],
        "e159_wall_verify_width": wall["wall_verify_width"],
        "e159_wall_step_ms": 1e3 * wall["wall_step_seconds"],
        "e159_wall_neighbour_step_ms": 1e3 * wall["neighbour_step_seconds"],
        "e159_wall_excess_ms": 1e3 * wall["wall_excess_seconds"],
        "e159_wall_share_of_reaching_depth_7": wall[
            "wall_share_of_reaching_depth_7"],
        "e159_cost_of_reaching_depth_7_ms": 1e3 * wall[
            "cost_of_reaching_depth_7_seconds"],
        "e159_adaptive_rounds_at_or_above_wall": wall[
            "adaptive_rounds_at_or_above_wall"],
        "e159_adaptive_share_at_or_above_wall": wall[
            "adaptive_share_at_or_above_wall"],
        "e159_shipped_width_cap": wall["shipped_width_cap"],
    })


def log_trace(run, doc: dict) -> None:
    traces = doc.get("trace_legs") or []
    if not traces:
        return
    columns = [
        "dir", "arm", "pinned_depth", "rounds", "round_us", "draft_build_us",
        "d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_submit2_us",
        "d_chain_us", "head_segment_us", "verify_build_us", "eval_wall_us",
        "readout_us", "commit_us", "upkeep_us", "session_tail_us",
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
            "e159_h_head_intercept_seconds": head_fit["intercept"],
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
    budget = adapt.get("trace_anchored_budget") or {}
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
        ),
        "e159/budget/trace_anchored": table(
            ["component", "seconds", "share_of_round"],
            [
                ("target_batch_1", budget.get("target_batch_1_seconds"),
                 budget.get("target_batch_1_share")),
                ("extra_verify_rows",
                 budget.get("extra_verify_rows_seconds"),
                 budget.get("extra_verify_rows_share")),
                ("proposal_head", budget.get("proposal_head_seconds"),
                 budget.get("proposal_head_share")),
                ("session_overhead", budget.get("session_overhead_seconds"),
                 budget.get("session_overhead_share")),
            ],
        ),
    })
    run.summary.update({
        "e159_adaptive_q_proposed": adapt["q_proposed_per_round"],
        "e159_adaptive_a_accepted": adapt["a_accepted_per_round"],
        "e159_adaptive_alpha": adapt["alpha_accept_fraction"],
        "e159_adaptive_R_decode_ms": 1e3 * adapt["R_decode_seconds"],
        "e159_adaptive_R_prescribed_ms": 1e3 * adapt["R_seconds_prescribed"],
        "e159_adaptive_mtp_seconds_per_token": adapt["mtp_seconds_per_token"],
        "e159_budget_fixed_share": adapt["fixed_share"],
        "e159_budget_drafting_share": adapt["drafting_share"],
        "e159_budget_head_share": adapt["head_share"],
        "e159_budget_marginal_verify_share": adapt["marginal_verify_share"],
        "e159_budget_trace_target_batch_1_share": budget.get(
            "target_batch_1_share"),
        "e159_budget_trace_extra_verify_share": budget.get(
            "extra_verify_rows_share"),
        "e159_budget_trace_head_share": budget.get("proposal_head_share"),
        "e159_budget_trace_session_share": budget.get(
            "session_overhead_share"),
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
            "rule79_not_evidence": True,
            "host": "Apple M4 Pro, 20 GPU cores, 48 GiB, applegpu_g16s",
            "thermal_mode": (
                "MLXFAST_LOCAL_COOL_GATE=0, ABBA-counterbalanced within one "
                "session, entry and exit GPU temperature recorded per leg"
            ),
            "decode_tokens": 512,
            "fixture": (
                "correctness_prompts/"
                "public_longcopy_gate_english_512_256.json"),
            "pinned_depths": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "depth_override_env": "MLX_E159_FIXED_DRAFT_DEPTH",
            "depth_override_submitted": False,
            "cost_model_depth_changed": False,
            "round_arithmetic": (
                "rounds = tokens - acceptedDraftTotal; R_decode = "
                "(decodeSeconds - seedPrefillSeconds) / rounds; a = "
                "acceptedDraftTotal / rounds; q = effective_mean_draft_len; "
                "mtp = seedPrefillSeconds / tokens + R_decode / (1 + a)"
            ),
            "round_time_definition": doc["round_time_definition"],
            "rho_definition": "8 * slope / intercept of R_decode(D) = s + h*D",
            "rho_predicted_low": RHO_PREDICTED[0],
            "rho_predicted_high": RHO_PREDICTED[1],
            "rho_gate_widen_producer_fusion_below": RHO_GATES[0][0],
            "rho_gate_pivot_to_head_above": RHO_GATES[1][0],
            "session": doc["session"],
            "dense_sessions": doc["dense_sessions"],
            "trace_session": doc["trace_session"],
        },
    )
    log_legs(run, doc)
    log_depth(run, doc)
    log_segments(run, doc)
    log_fit(run, doc)
    log_width_wall(run, doc)
    log_trace(run, doc)
    log_budget(run, doc)
    run.summary.update({"e159_headline": f"rho = {rho:.4f} -> {decision(rho)}"})
    print(run.url, run.id, sep="\n")
    run.finish()


if __name__ == "__main__":
    main()
