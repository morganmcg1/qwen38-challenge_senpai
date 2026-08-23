#!/usr/bin/env python3
"""Publish the E153 terminal result to Weights and Biases.

One run for the whole assignment.

R1 ships `derivedClusterRowsPerLeaf = 16`, the derived cluster index leaf
width. R2 built a merged wide-decode SDPA kernel, proved it bit-exact, and
measured it slower in a thermally gated ABBA session. R2 is stripped from the
submitted surface and preserved at commit `e4a9ea2a` for E156.

Frame labels follow Rule 144. Every measured number here is `harness=local`.
Every ranked projection is a model, is named as one, and carries its
`fitted_on` tag. Rule 134's 524.5 us/round per 1 percent is `fitted_on=0cf1637e`
and reproduces in the TOTAL-LEG frame, so the frame-consistent leaf16 reading
is the total-leg one.
"""

from __future__ import annotations

import json
import pathlib

import wandb

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

PR_NUMBER = 153
BRANCH = "qwen-askeladd/e153-merged-sdpa-kernel-and-leaf16"
BASE_SHA = "b27c004afd515d0998ef86f252ce0cf048b4c197"
BUDGET_BASE = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
SESSION_COMMIT = "a37efb2450ba3eca43b461255efbc641a6d51cc2"
GATE_WORKER_SHA = "18090ae3e012535b4a1c17329688d3c6691f2159c7653a77de4f2e0f0affbd98"
ABBA_WORKER_SHA = "4f3abde77ca30e5858268dc91a304da0d7c03a3fc9a362fd3873889671f0fd49"
R2_PRESERVED_COMMIT = "e4a9ea2a"

RESUME_ID = "e153askeladdr1"

GATED_LEG_SD_PCT = 0.052
RULE134_US_PER_ROUND_PER_PCT = 524.5
RULE134_FITTED_ON = "0cf1637e"

# `research/e153_frame_check.py`, reproduced in this session.
DEPTH_DISCOUNT = 0.74453
LOCAL_EDL_BENCHFIXTURE = 6.3590
RANKED_STEPS_PER_ROUND = 4.7345
R1_LOCAL_US_PER_ROUND_DECODE = 132.55
R1_LOCAL_US_PER_ROUND_TOTAL = 169.84

FILES = [
    "e153_exactness.sh", "e153_timing.sh", "e153_frame_check.py",
    "e153_r2_abba.sh", "e153_r2_predict.py", "e153_r2_report.py",
    "e153_r2_warm_confound.py", "e153_r1_gate_chain.sh",
    "e153_wandb_result.py",
    "e153-r2-abba.json", "e153-r2-prediction.json",
]
OUT_FILES = [
    "e153-r2-exactness.json", "e153-r2-timing.json",
    "e153-r2-warm-confound.json", "e153-r1-leaf-witness.json",
]

# `research/e153_r1_gate_chain.sh`, job 0234b74c, exit 0.
#
# Gate provenance, reported verbatim rather than summarised. The chain ran the
# real 40 C gate in a SEPARATE process immediately before each leg, through
# `benchmark.sh --local-cool-gate-only`, and the chain recorded
# `e153_cool_gate_status=passed`. The leg's own `meta.txt` therefore carries
# `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`,
# because the leg itself did not run a gate. These are exactness legs. No
# timing claim is made from them, so the in-leg timing qualification is not
# load-bearing, but the flags are preserved as written.
EXACTNESS_LEGS = {
    "beagle_a": {
        "all_tokens_matched": True, "residual_divergence_count": 0,
        "round_count": 119, "effective_mean_draft_len": 4.2436974789915967,
        "accept_rate": 0.77821782178217824,
        "gpu_temp_entry_c": 40.23643493652344,
        "gpu_temp_exit_c": 61.80331802368164,
        "cool_gate_passed_real_gate_in_leg": False,
        "external_real_gate_status": "passed",
        "external_gate_exit_temp_c": 39.9,
        "gate_qualified_for_timing": False,
        "head_provenance_sha256":
            "dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9",
    },
    "essays_montaigne": {
        "all_tokens_matched": True, "residual_divergence_count": 0,
        "round_count": 146, "effective_mean_draft_len": 3.4383561643835616,
        "accept_rate": 0.72908366533864544,
        "gpu_temp_entry_c": 41.577152252197266,
        "gpu_temp_exit_c": 61.957157135009766,
        "cool_gate_passed_real_gate_in_leg": False,
        "external_real_gate_status": "passed",
        "external_gate_exit_temp_c": 39.4,
        "gate_qualified_for_timing": False,
        "head_provenance_sha256":
            "dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9",
    },
}

GATE_CHAIN = {
    "phase0-rebuild-and-assert-worker":
        "pass, two-sided: leaf16 selectors present, "
        "qwen35_merged_sdpa_vector absent",
    "phase1-swift-test":
        "pass, 41 issues under the documented 10 names, per-test issue "
        "locations byte-identical to research/out/e149-swift-test.log. The "
        "run total reads 42 because of one extra pre-existing base-drift "
        "issue at E145WidthPinTests.swift:65, which this branch does not "
        "touch",
    "phase2-twin-audit": "pass, 29 runtime-effective twins",
    "phase3-leaf-override-positive-control":
        "pass, the divisibility guard fires on MLX_E141_ROWS_PER_LEAF=12",
    "phase4-512-token-exactness-legs":
        "pass, two legs behind the real 40C gate, all tokens matched, "
        "zero residual divergences",
    "validate-assignment-scope": "pass",
    "check-editable-budget":
        "pass, growth_enforced 197356/262144 against 770a3ff2",
    "verify-ranked-score-boundary": "pass",
}


def load(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    abba = load(HERE / "e153-r2-abba.json")
    pred = load(HERE / "e153-r2-prediction.json")
    exact = load(OUT / "e153-r2-exactness.json")
    timing = load(OUT / "e153-r2-timing.json")
    confound = load(OUT / "e153-r2-warm-confound.json")
    witness = load(OUT / "e153-r1-leaf-witness.json")

    cells = exact["cells"]
    positive_control_min = min(c["positive_control_max_abs"] for c in cells)
    merged_max_abs = max(c["merged_vs_split_max_abs"] for c in cells)
    mismatch_bytes = max(c["merged_vs_split_bit_mismatch_bytes"] for c in cells)

    r1_ranked_us_total = R1_LOCAL_US_PER_ROUND_TOTAL * DEPTH_DISCOUNT
    r1_ranked_us_decode = R1_LOCAL_US_PER_ROUND_DECODE * DEPTH_DISCOUNT
    r1_ranked_pct_total = r1_ranked_us_total / RULE134_US_PER_ROUND_PER_PCT
    r1_ranked_pct_decode = r1_ranked_us_decode / RULE134_US_PER_ROUND_PER_PCT

    metrics: dict[str, float | int] = {
        # ---- primary: the shipped candidate, frame-consistent total-leg model
        "e153_primary_ranked_pct": r1_ranked_pct_total,

        # ---- R1 leaf16, ranked projection (a MODEL, not a measurement)
        "e153_leaf16_ranked_pct_total_leg": r1_ranked_pct_total,
        "e153_leaf16_ranked_pct_decode_frame_mixed": r1_ranked_pct_decode,
        "e153_leaf16_ranked_us_per_round_total_leg": r1_ranked_us_total,
        "e153_leaf16_ranked_us_per_round_decode": r1_ranked_us_decode,
        "e153_leaf16_depth_discount": DEPTH_DISCOUNT,
        "e153_leaf16_local_edl_benchfixture": LOCAL_EDL_BENCHFIXTURE,
        "e153_leaf16_ranked_steps_per_round": RANKED_STEPS_PER_ROUND,

        # ---- R1 leaf16, the underlying gated ABBA measurement (E149 arm A)
        "e153_leaf16_local_us_per_round_total_leg": R1_LOCAL_US_PER_ROUND_TOTAL,
        "e153_leaf16_local_us_per_round_decode": R1_LOCAL_US_PER_ROUND_DECODE,
        "e153_leaf16_source_n_timed_legs": 12,
        "e153_leaf16_source_divergences": 0,

        # ---- R1 leaf16, structure
        "e153_leaf16_leaves": 6146,
        "e153_leaf16_probes": 1537,
        "e153_leaf16_probed_rows": 24592,
        "e153_leaf16_baseline_leaves": 12292,
        "e153_leaf16_baseline_probes": 3073,
        "e153_leaf16_baseline_probed_rows": 24584,
        "e153_leaf16_coarse_pass_byte_reduction_pct": 50.0,
        "e153_leaf16_fine_pass_row_growth_pct":
            100.0 * (24592 - 24584) / 24584,

        # ---- R1 leaf16, mechanism identity claims verified this session
        "e153_leaf16_touches_outputs_per_thread": 0,
        "e153_leaf16_is_qmv_output_row_tile": 0,
        "e153_leaf16_nax_scaffold_dependency": 0,
        "e153_leaf16_submitted_files_changed": 1,
        "e153_leaf16_override_positive_control_passed":
            float(witness["e149_leaf_override_positive_control_passed"]),

        # ---- R1 gate chain, phase 4
        "e153_r1_exactness_legs": len(EXACTNESS_LEGS),
        "e153_r1_all_tokens_matched": 1.0,
        "e153_r1_residual_divergence_count": 0,
        "e153_r1_entry_temp_c_min": min(
            v["gpu_temp_entry_c"] for v in EXACTNESS_LEGS.values()),
        "e153_r1_entry_temp_c_max": max(
            v["gpu_temp_entry_c"] for v in EXACTNESS_LEGS.values()),
        "e153_r1_swift_test_issues": 41,
        "e153_r1_swift_test_names": 10,
        "e153_r1_swift_test_base_drift_issues": 1,
        "e153_r1_cool_gate_passed_real_gate_in_leg": 0.0,
        "e153_r1_external_real_gate_passed": 1.0,
        "e153_r1_gate_qualified_for_timing": 0.0,

        # ---- R1 budget
        "e153_growth_enforced_bytes": 197356,
        "e153_growth_enforced_limit_bytes": 262144,
        "e153_growth_attributable_bytes": 1860,
        "e153_growth_reclaimed_bytes": 15842,
        "e153_team_headroom_bytes": 64788,

        # ---- R2 merged SDPA, exactness
        "e153_r2_cells": len(cells),
        "e153_r2_merged_vs_split_max_abs": merged_max_abs,
        "e153_r2_merged_vs_split_bit_mismatch_bytes": mismatch_bytes,
        "e153_r2_positive_control_min_max_abs": positive_control_min,
        "e153_r2_merged_ran_all_cells": float(
            all(c["merged_ran"] for c in cells)),

        # ---- R2 merged SDPA, gated ABBA (the decisive measurement)
        "e153_merged_sdpa_local_pct": abba["e153_merged_sdpa_local_pct"],
        "e153_merged_sdpa_local_pct_decode_frame":
            abba["e153_merged_sdpa_local_pct_decode_frame"],
        "e153_merged_sdpa_round_cost_us":
            abba["e153_merged_sdpa_round_cost_us"],
        "e153_merged_sdpa_round_cost_us_total_frame":
            abba["e153_merged_sdpa_round_cost_us_total_frame"],
        "e153_merged_sdpa_divergences":
            abba["e153_merged_sdpa_divergences"],
        "e153_merged_sdpa_sigma_vs_gated_floor_total":
            abba["sigma_vs_gated_floor_total"],
        "e153_merged_sdpa_n_timed_legs": abba["n_timed_legs"],
        "e153_merged_sdpa_entry_temp_c_spread": abba["entry_c_spread"],
        "e153_merged_sdpa_unweighted_block_decode_pct":
            abba["unweighted_block_decode_pct"],

        # ---- R2, the isolated probe that got it wrong
        "e153_r2_probe_predicted_us_per_round":
            abba["c1_predicted_us_per_round"],
        "e153_r2_probe_predicted_se": abba["c1_predicted_se"],
        "e153_r2_probe_prediction_ratio": abba["c1_prediction_ratio"],
        "e153_r2_probe_prediction_z": abba["c1_prediction_z"],
        "e153_r2_probe_implied_eligible_round_saving_us":
            abba["implied_eligible_round_saving_us"],
        "e153_r2_probe_streaming_slope_us_per_key_per_layer":
            timing["c1_streaming_slope_us_per_key_per_layer"],
        "e153_r2_probe_structural_intercept_us_per_layer":
            timing["c1_structural_intercept_us_per_layer"],

        # ---- R2, the advisor's cold-fallback confound, rejected
        "e153_r2_confound_fixed_us_per_leg":
            confound["e153_r2_fixed_us_per_leg"],
        "e153_r2_confound_variable_us_per_eligible_round":
            confound["e153_r2_variable_us_per_eligible_round"],
        "e153_r2_confound_corrected_decode_pct":
            confound["e153_r2_decode_pct_after_removing_all_fixed_cost"],
        "e153_r2_confound_corrected_sigma":
            confound["e153_r2_sigma_after_removing_all_fixed_cost"],
        "e153_r2_confound_rejected":
            float(confound["e153_r2_warm_confound_rejected"]),

        # ---- shared campaign constants
        "e153_gated_leg_sd_pct": GATED_LEG_SD_PCT,
        "e153_rule134_us_per_round_per_pct": RULE134_US_PER_ROUND_PER_PCT,
        "e153_ranked_p_m_ge_6": abba["ranked_p_m_ge_6"],
    }
    for prompt, leg in EXACTNESS_LEGS.items():
        for key in ("round_count", "effective_mean_draft_len", "accept_rate",
                    "gpu_temp_entry_c", "gpu_temp_exit_c"):
            metrics[f"e153_r1_{prompt}_{key}"] = leg[key]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=RESUME_ID,
        resume="allow",
        name="e153-terminal-leaf16-ship-merged-sdpa-negative",
        job_type="analysis",
        tags=["e153", "askeladd", "leaf16", "merged-sdpa", "terminal"],
        config={
            "experiment": "E153",
            "pr_number": PR_NUMBER,
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "budget_base_sha": BUDGET_BASE,
            "session_commit": SESSION_COMMIT,
            "gate_chain_worker_sha256": GATE_WORKER_SHA,
            "r2_abba_worker_sha256": ABBA_WORKER_SHA,
            "r2_preserved_commit": R2_PRESERVED_COMMIT,
            "host": "Apple M4 Pro, Mac mini Mac16,11, 48 GB",
            "ranked_host": "m5-qwen38-27b-mtp (not this machine)",
            "decode_tokens": 512,
            "harness": "local",
            "upstream_synced_commit":
                "c0dbec051c58bccf5435ee1e1e5b01271dc7e179",
            "promoted_submission_id":
                "684821ed-f7b5-48f5-9ce1-df99b59e19b6",
            "promoted_source_ref":
                "eb5eadc7a165047d4321ce883b9ff30894d8bd19",
            "promoted_score": 3.71959723,
            "head_provenance_sha256":
                "dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9",
            "head_manifest_tree_sha256":
                "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71",
            "r1_gate_provenance":
                "the real 40 C gate ran in a separate process immediately "
                "before each exactness leg; the leg's own meta.txt therefore "
                "records cool_gate_passed_real_gate=false. No timing claim "
                "uses these legs.",
            "rule134_fitted_on": RULE134_FITTED_ON,
            "rule134_frame": "total-leg; it reproduces at 521.0 total-leg "
                             "against 468.8 decode-frame",
            "r1_mechanism": "derivedClusterRowsPerLeaf 8 -> 16, the derived "
                            "cluster ANN index leaf width over the 98,336-row "
                            "padded compact draft head",
            "r1_mechanism_class": "per_draft_step",
            "r1_mechanism_is_not":
                "the wide-QMV output row tile n/entry.rps, and not "
                "activeInputGroups; the E135 x-side grid trim is already the "
                "compiled default on this base",
            "r1_submitted_file":
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
            "r2_mechanism": "merged wide-decode SDPA kernel, width gated at 6",
            "r2_mechanism_class": "width_gated_at_6",
            "r2_verdict": "not useful; bit-exact but slower",
            "r2_arm_a": abba["arm_a"],
            "r2_arm_b": abba["arm_b"],
            "r2_estimator": "within-prompt ABBA, one binary, real 40C gate",
            "r2_sign_convention": abba["sign_convention"],
            "r2_all_gate_qualified": abba["all_gate_qualified"],
            "r2_all_tokens_matched": abba["all_tokens_matched"],
            "r2_schedule_identical_all_blocks":
                abba["schedule_identical_all_blocks"],
            "r2_exactness_reference_arm": exact["reference_arm"],
            "r2_probe_gate_qualified_for_timing":
                timing["gate_qualified_for_timing"],
            "r2_probe_cool_gate_passed_real_gate":
                timing["cool_gate_passed_real_gate"],
            "gate_chain": GATE_CHAIN,
            "e153_r2_prediction_preregistered": pred is not None,
        },
    )
    run.log(metrics)

    exactness = wandb.Table(columns=[
        "query_layout", "kL", "qL", "merged_ran", "merged_vs_split_max_abs",
        "merged_vs_split_bit_mismatch_bytes", "split_vs_split_max_abs",
        "positive_control_max_abs"])
    for cell in cells:
        exactness.add_data(
            cell["query_layout"], cell["kL"], cell["qL"], cell["merged_ran"],
            cell["merged_vs_split_max_abs"],
            cell["merged_vs_split_bit_mismatch_bytes"],
            cell["split_vs_split_max_abs"], cell["positive_control_max_abs"])
    run.log({"e153_r2_exactness_cells": exactness})

    blocks = wandb.Table(columns=[
        "prompt", "replicate", "rounds", "eligible_rounds", "eligible_frac",
        "decode_pct", "total_pct", "decode_us_per_round", "us_per_eligible_round",
        "edl", "accept_rate_delta_pp", "schedule_identical",
        "merged_arm_kernels", "gate_state"])
    for block in abba["per_block"]:
        hist = {int(k): v for k, v in block["width_hist"].items()}
        rounds = block["rounds"][0]
        eligible = sum(v for k, v in hist.items() if k >= 6)
        blocks.add_data(
            block["prompt"], block["replicate"], rounds, eligible,
            eligible / rounds, block["decode_pct"], block["total_pct"],
            block["decode_us_per_round"],
            block["decode_us_per_round"] * rounds / eligible,
            float(block["edl"][0]), block["accept_rate_delta_pp"],
            block["schedule_identical_across_arms"],
            block["arms_witnessed"][0], block["gate_states"][0])
    run.log({"e153_r2_abba_blocks": blocks})

    legs = wandb.Table(columns=[
        "prompt", "all_tokens_matched", "residual_divergence_count",
        "round_count", "effective_mean_draft_len", "accept_rate",
        "gpu_temp_entry_c", "gpu_temp_exit_c",
        "cool_gate_passed_real_gate_in_leg", "external_real_gate_status",
        "external_gate_exit_temp_c", "gate_qualified_for_timing",
        "head_provenance_sha256"])
    for prompt, leg in EXACTNESS_LEGS.items():
        legs.add_data(
            prompt, leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["round_count"], leg["effective_mean_draft_len"],
            leg["accept_rate"], leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"],
            leg["cool_gate_passed_real_gate_in_leg"],
            leg["external_real_gate_status"],
            leg["external_gate_exit_temp_c"],
            leg["gate_qualified_for_timing"],
            leg["head_provenance_sha256"])
    run.log({"e153_r1_exactness_legs": legs})

    widths = wandb.Table(columns=["prompt", "width", "rounds", "eligible"])
    for block in abba["per_block"]:
        for width, count in sorted(
                ((int(k), v) for k, v in block["width_hist"].items())):
            widths.add_data(block["prompt"], width, count, width >= 6)
    run.log({"e153_r2_width_histogram": widths})

    art = wandb.Artifact("e153-terminal", type="analysis")
    for name in FILES:
        path = HERE / name
        if path.exists():
            art.add_file(str(path))
    for name in OUT_FILES:
        path = OUT / name
        if path.exists():
            art.add_file(str(path))
    run.log_artifact(art)

    print(run.id)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
