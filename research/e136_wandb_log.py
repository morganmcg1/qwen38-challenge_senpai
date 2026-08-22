#!/usr/bin/env python3
"""Publish an E136 rung to W&B.

    usage: research/e136_wandb_log.py --rung 0|0b|1 [--dry]

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
            "(Qwen35.swift:4117, typedef bfloat16_t InT at :4143) worth "
            "removing, priced on realised acceptance against a replay "
            "baseline remodelled on that same kernel's arithmetic",
        "command":
            "research/e133_job.sh python3 research/e133_screen.py attrib "
            "--per-seed --out research/e136-attrib-fp32.json && "
            "python3 research/e136_null_floor.py "
            "--attrib research/e136-attrib-fp32.json "
            "--out research/e136-fp32-floor.json",
    },
    "1": {
        "run_name": "e136-rung1-c1-sketch-readout-risk-gate",
        "file": "research/e136-c1-gate/verify.json",
        "companion": "research/e136-basis-probe.json",
        "job_type": "exactness-gate",
        "question":
            "does every dispatch of the C1 sketch shortlist compute what it "
            "claims, and what did the editable byte budget cost the arm when "
            "it forced a checkpoint-derived basis in place of the "
            "query-fitted one",
        "command":
            "research/await-lock-then-run.sh 3600 research/e133_job.sh env "
            "MLXFAST_RUN_MLX_RUNTIME_TESTS=1 "
            "MLXFAST_C1_OUT_DIR=research/e136-c1-gate "
            "swift test --force-resolved-versions --filter E136C1SketchTests",
    },
    "2": {
        "run_name": "e136-rung2-c1-sketch-readout-abba",
        "file": "research/e136-rung2.json",
        "companion": "research/e136-cliff-census.json",
        "job_type": "decode-abba",
        "question":
            "does the C1 sketch shortlist lower ABSOLUTE candidate MTP "
            "seconds per token on a matched ABBA decode pair, and does it "
            "leave the accepted token stream unchanged",
        "command":
            "senpai/rebuild-and-assert-worker.sh --require MLX_E136_C1_SKETCH "
            "&& research/await-lock-then-run.sh 3600 research/e133_job.sh "
            "research/e136_abba.sh 512 r2 && "
            "python3 research/e136_analyse.py --label r2 "
            "--json research/e136-rung2.json",
    },
    "5a": {
        "run_name": "e136-5a-probe-fraction-acceptance-replay",
        "file": "research/e136-probe-grid-priced.json",
        "companion": "research/e136-probe-grid.json",
        "job_type": "corpus-replay",
        "question":
            "where is the probe-fraction argmax on the shipped readout and "
            "on the C1 sketch readout, once the acceptance side is replayed "
            "on the corpus instead of assumed",
        "command":
            "research/await-lock-then-run.sh 1200 research/e133_job.sh "
            "python3 research/e133_screen.py screen "
            "--families exact0,lowrank256 --widths 4096 "
            "--probes 0.10,0.125,0.15,0.175,0.20,0.25,0.35 "
            "--stage-a sketch --out research/e136-probe-grid.json && "
            "python3 research/e136_probe_grid.py "
            "--screen research/e136-probe-grid.json "
            "--json research/e136-probe-grid-priced.json --self-check",
    },
}

LADDER_COLUMNS = [
    "family", "arm", "p", "anchor_p", "probes", "coarse_rows", "recall_wg",
    "acc_loss_wg", "d_acc_loss_pp", "gross_pct", "d_gross_pct",
    "d_gross_pct_scaled", "d_net_pct", "d_net_pct_scaled",
    "board_scale_applied", "predicted_pct_gating", "predicted_pct_absolute",
    "bytes_per_row", "n_gating", "passes_t0", "passes_t0b",
]

# F5 resolved ADVISOR ERROR 136: the PR body section D bars govern, and the
# F4 section 8 restatement is retracted. Advancing also needs the headline to
# be separated from the session's own within-arm 2 sigma null, so a large
# point estimate inside a wide null cannot advance.
RUNG2_ADVANCE_PCT = 0.30
RUNG2_HOLD_PCT = 0.10

# Rung 1 predicted the C1 acceptance loss from the offline E133 corpus screen
# on the shipping `lowrank256` basis. Logging the prediction next to the
# measurement is what turns a negative result into a calibration of the
# screen.
RUNG1_PREDICTED_ACC_LOSS_PP = 100.0 * 7.8756e-4

# Rule 107 net for C1, gross byte gain minus C1's own predicted acceptance
# cost, quoted on both bandwidth lines because the byte-to-time coefficient is
# not settled. The 265 GB/s line is the ceiling; E93 measured 186.7 GB/s on
# the head pass.
RUNG1_NET_PCT_AT_265_GB_S = 0.546
RUNG1_NET_PCT_AT_186_7_GB_S = 0.353

# Local `benchfixture` drafts deeper than the F83-weighted ranked prompt mix
# (effective mean draft length 6.359 against about 4.9), so a local per-cent
# is worth less on the ranked board. F5 requires this to be STATED, never
# silently applied: every logged per-cent below is local.
LOCAL_TO_RANKED_HAIRCUT = 1.3

LEG_COLUMNS = [
    "tag", "arm", "position", "flag", "arm_witnessed", "c1_draft_steps",
    "shipped_selection_draft_steps", "mtp_s_per_tok", "serial_s_per_tok",
    "speedup", "mean_d", "mean_acc", "realised_acceptance",
    "median_round_us", "median_draft_build_us", "rounds", "decode_tokens",
    "all_tokens_matched", "residual_divergence_count", "entry_c", "exit_c",
    "status", "worker_sha256", "session_commit", "base_sha",
    "dirty_candidate_paths",
]


def rung2_summary(payload: dict, spec: dict) -> tuple[dict, dict]:
    mtp = payload["mtp_s_per_tok"]
    headline = payload.get("e136_c1_candidate_leg_pct")
    null = payload.get("e136_session_null_2sigma_pct")
    exact = (payload["all_tokens_matched"]
             and payload["accepted_stream_divergences"] == 0)
    separated = (headline is not None and null is not None
                 and abs(headline) > null)
    if headline is None:
        verdict = "no-contrast"
    elif not payload["all_arms_witnessed"] or not exact:
        verdict = "invalid"
    elif headline >= RUNG2_ADVANCE_PCT and separated:
        verdict = "ADVANCE"
    elif headline >= RUNG2_HOLD_PCT:
        verdict = "HOLD"
    else:
        verdict = "CLOSE"

    measured_acc_loss_pp = payload.get("e136_realised_acceptance_delta_pp")
    acc_ratio = (abs(measured_acc_loss_pp) / RUNG1_PREDICTED_ACC_LOSS_PP
                 if measured_acc_loss_pp else None)

    # FINDING 181 clearance. This host clamps at 96 registers from NA=5 up
    # while the ranked host grants 126, so a register delta on a routed
    # wide-QMV entry point would make g16s absolute time an unsafe screen.
    census = json.loads(pathlib.Path(spec["companion"]).read_text())
    worst_reg_delta = max(
        abs(cell[arch]["registers_delta"])
        for cell in census["cells"]
        for arch in ("applegpu_g16s", "applegpu_g17s"))

    summary = {
        # Primary. Absolute candidate MTP time is the only quantity the ranked
        # numerator responds to; the sign is flipped so faster reads positive.
        "e136_c1_candidate_leg_pct": headline,
        "e136_realised_acceptance_delta_pp":
            payload.get("e136_realised_acceptance_delta_pp"),
        "e136_accepted_stream_divergences":
            payload["accepted_stream_divergences"],
        "e136_c1_candidate_leg_pct_two_sigma":
            payload.get("e136_c1_candidate_leg_pct_two_sigma"),
        "e136_session_null_2sigma_pct": null,
        "e136_rung2_separated_from_session_null": separated,
        "e136_rung2_verdict": verdict,
        "e136_rung2_advance_bar_pct": RUNG2_ADVANCE_PCT,
        "e136_rung2_hold_bar_pct": RUNG2_HOLD_PCT,
        # The calibration result. The offline corpus screen predicted the
        # acceptance cost of the shipping basis; the live session measured it.
        "e136_rung2_predicted_acc_loss_pp": RUNG1_PREDICTED_ACC_LOSS_PP,
        "e136_rung2_acc_loss_underprediction_ratio": acc_ratio,
        "e136_rung2_rule107_net_pct_at_265_gb_s": RUNG1_NET_PCT_AT_265_GB_S,
        "e136_rung2_rule107_net_pct_at_186_7_gb_s":
            RUNG1_NET_PCT_AT_186_7_GB_S,
        # Stated, not applied: every per-cent logged here is a LOCAL per-cent.
        "e136_local_to_ranked_haircut_stated_not_applied":
            LOCAL_TO_RANKED_HAIRCUT,
        "e136_rung2_cliff_census_verdict": census["verdict"],
        "e136_rung2_cliff_census_worst_registers_delta": worst_reg_delta,
        "e136_rung2_finding181_applies": worst_reg_delta != 0,
        # The binary that actually decoded the tokens. A session whose worker
        # digest moved is not a comparison, and `base_sha` may legitimately
        # move across a research-tooling commit that the worker never links.
        "e136_rung2_worker_uniform":
            payload.get("identity", {}).get("worker_uniform"),
        "e136_rung2_base_sha_uniform":
            payload.get("identity", {}).get("base_sha_uniform"),
        "e136_rung2_no_dirty_candidate_paths":
            payload.get("identity", {}).get("no_dirty_candidate_paths"),
        "e136_rung2_all_arms_witnessed": payload["all_arms_witnessed"],
        "e136_rung2_all_tokens_matched": payload["all_tokens_matched"],
        "e136_rung2_entry_temp_spread_c": payload["entry_temp_spread_c"],
        "e136_rung2_legs": len(payload["legs"]),
        "e136_rung2_decode_tokens": payload["legs"][0].get("decode_tokens"),
        # Absolute times, which lead, and the local ratio, which is a valid
        # SECOND readout here only because arm C1 changes work confined to the
        # candidate MTP leg.
        "e136_rung2_mtp_s_per_tok_off": mtp["off"]["mean"],
        "e136_rung2_mtp_s_per_tok_on": mtp["on"]["mean"],
        "e136_rung2_mtp_s_per_tok_sd_off": mtp["off"]["sd"],
        "e136_rung2_mtp_s_per_tok_sd_on": mtp["on"]["sd"],
        "e136_rung2_serial_s_per_tok_delta_pct":
            payload["serial_s_per_tok"].get("delta_pct"),
        "e136_rung2_local_ratio_delta_pct":
            payload["speedup"].get("delta_pct"),
    }
    for key in ("mean_draft_len", "realised_acceptance", "median_round_us",
                "median_draft_build_us"):
        block = payload.get(key, {})
        if "delta_pct" in block:
            summary["e136_rung2_%s_off" % key] = block["off"]["mean"]
            summary["e136_rung2_%s_on" % key] = block["on"]["mean"]
            summary["e136_rung2_%s_delta_pct" % key] = block["delta_pct"]
    return summary, {"rung2_legs": table(LEG_COLUMNS, payload["legs"])}


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


# Rung 1 ships no timing. It publishes the risk gate the arm needs before it
# is allowed to spend a matched decode pair, plus the basis-substitution
# numbers that changed the arm's expected value.
BASIS_COLUMNS = [
    "basis", "fitted_on", "cross_fit", "recall_wg", "acc_loss_wg",
    "predicted_pct_gating", "predicted_pct_absolute", "ships",
]

# The E133 screen cells, both at 264 B/row and gross pct_head_share_7 0.8109.
BASIS_ROWS = [
    {
        "basis": "qlowrank256", "fitted_on": "e133 query corpus",
        "cross_fit": True, "recall_wg": 0.999564, "acc_loss_wg": 2.1777e-4,
        "predicted_pct_gating": 0.767, "predicted_pct_absolute": 0.678,
        "ships": False,
    },
    {
        "basis": "lowrank256", "fitted_on": "transformed checkpoint rows",
        "cross_fit": False, "recall_wg": 0.995275, "acc_loss_wg": 7.8756e-4,
        "predicted_pct_gating": 0.651, "predicted_pct_absolute": -0.028,
        "ships": True,
    },
]

RUNG1_SELECTION_COST_PCT = 0.105


def rung1_summary(payload: dict, spec: dict) -> tuple[dict, dict]:
    probe = json.loads(pathlib.Path(spec["companion"]).read_text())
    ships = next(r for r in BASIS_ROWS if r["ships"])
    blocked = next(r for r in BASIS_ROWS if not r["ships"])

    summary = {
        # Arithmetic. fp32 reassociation only.
        "e136_rung1_project_rel_error": payload["project_rel_error"],
        "e136_rung1_centroid_sketch_rel_error":
            payload["centroid_sketch_rel_error"],
        "e136_rung1_row_sketch_rel_error": payload["row_sketch_rel_error"],
        # The gate has to be able to fail.
        "e136_rung1_positive_control_rel_error":
            payload["positive_control_rel_error"],
        "e136_rung1_positive_control_fires":
            payload["positive_control_rel_error"] > 1e-2,
        # Exactness of the parts that are exact by construction.
        "e136_rung1_rescore_contiguous_mismatches":
            payload["rescore_contiguous_mismatches"],
        "e136_rung1_rescore_gathered_mismatches":
            payload["rescore_gathered_mismatches"],
        "e136_rung1_tau_is_exact": payload["tau_is_exact"],
        "e136_rung1_above_is_exact": payload["above_is_exact"],
        "e136_rung1_cursor_is_exact": payload["cursor_is_exact"],
        "e136_rung1_survivors_distinct": payload["survivors_distinct"],
        "e136_rung1_survivor_capacity": payload["survivor_capacity"],
        "e136_rung1_survivors_below_tau": payload["survivors_below_tau"],
        "e136_rung1_sketch_top32_recall": payload["sketch_top32_recall"],
        "e136_rung1_shortlist_mismatches": payload["shortlist_mismatches"],
        "e136_rung1_gate_all_pass": (
            payload["project_rel_error"] < 1e-5
            and payload["centroid_sketch_rel_error"] < 1e-5
            and payload["row_sketch_rel_error"] < 1e-5
            and payload["positive_control_rel_error"] > 1e-2
            and payload["rescore_contiguous_mismatches"] == 0
            and payload["rescore_gathered_mismatches"] == 0
            and payload["tau_is_exact"] and payload["above_is_exact"]
            and payload["cursor_is_exact"]
            and payload["survivors_distinct"] == payload["survivor_capacity"]
            and payload["survivors_below_tau"] == 0
            and payload["sketch_top32_recall"] == 1.0
            and payload["shortlist_mismatches"] == 0),
        # The forced basis substitution and what it cost in expected value.
        "e136_rung1_basis_shipped": ships["basis"],
        "e136_rung1_basis_blocked": blocked["basis"],
        "e136_rung1_basis_blocked_bytes": 5120 * 256 * 2,
        "e136_rung1_growth_headroom_bytes_at_decision": 103166,
        "e136_rung1_predicted_pct_gating": ships["predicted_pct_gating"],
        "e136_rung1_predicted_pct_gating_forgone":
            blocked["predicted_pct_gating"] - ships["predicted_pct_gating"],
        "e136_rung1_predicted_pct_absolute": ships["predicted_pct_absolute"],
        "e136_rung1_selection_cost_pct": RUNG1_SELECTION_COST_PCT,
        # Both pricings, because they disagree in sign.
        "e136_rung1_net_pct_rule107":
            ships["predicted_pct_gating"] - RUNG1_SELECTION_COST_PCT,
        "e136_rung1_net_pct_miss_rate_proxy":
            ships["predicted_pct_absolute"] - RUNG1_SELECTION_COST_PCT,
        # Why the shipped basis is weaker: it is aligned with row variance,
        # not with where the queries live.
        "e136_rung1_row_energy_at_rank256":
            probe["exact_row_energy"],
        "e136_rung1_resident_bytes": 31_847_712,
        "e136_rung1_residency_side": "active_at_sizing",
        "e136_rung1_slack_consumed_bytes": 0,
    }
    for stratum, row in probe.get("query_energy", {}).items():
        for key, value in row.items():
            summary["query_energy/%s/%s" % (stratum, key)] = value
    return summary, {"rung1_basis_choice": table(BASIS_COLUMNS, BASIS_ROWS)}


def rung5a_summary(payload: dict, spec: dict) -> tuple[dict, dict]:
    shipped = payload["argmax"]["exact"]
    c1 = payload["argmax"]["lowrank"]
    floor = payload["null_floors"]["perfect_readout:corpus"]
    rows = payload["ladders"]["exact"] + payload["ladders"]["lowrank"]
    low = {r["family"]: r for r in rows if r["p"] == 0.10}

    summary = {
        "e136_5a_shipped_argmax_p": shipped["p"],
        "e136_5a_shipped_argmax_anchor_p": shipped["anchor_p"],
        "e136_5a_shipped_argmax_d_net_pct": shipped["d_net_pct"],
        "e136_5a_shipped_argmax_d_net_pct_scaled":
            shipped["d_net_pct_scaled"],
        "e136_5a_shipped_beats_pooled_2sigma_clustered":
            shipped["beats_pooled_2sigma_clustered"],
        "e136_5a_shipped_beats_pooled_2sigma_clustered_scaled":
            shipped["beats_pooled_2sigma_clustered_scaled"],
        "e136_5a_c1_argmax_p": c1["p"],
        "e136_5a_c1_argmax_d_net_pct": c1["d_net_pct"],
        "e136_5a_c1_beats_pooled_2sigma_clustered":
            c1["beats_pooled_2sigma_clustered"],
        "e136_5a_pooled_2sigma_clustered_floor_pct":
            floor["two_sigma_clustered_pct"],
        "e136_5a_marginal_pct_per_0.01p_shipped":
            payload["marginal_pct_per_0.01_p_shipped"],
        "e136_5a_marginal_pct_per_0.01p_c1":
            payload["marginal_pct_per_0.01_p_c1"],
        # Pre-registration scoring. The outcome held and the stated mechanism
        # did not: recall never falls on the shipped ladder, so the measured
        # argmax is set by the lowest probe fraction sampled, not by
        # acceptance. The true shipped argmax lies below this grid.
        "e136_5a_prediction_5a_argmax_below_0.15": shipped["p"] < 0.15,
        "e136_5a_prediction_5a_mechanism_held":
            low["exact"]["recall_wg"] < 1.0,
        "e136_5a_prediction_5b_no_c1_argmax":
            c1["p"] == c1["anchor_p"] and c1["d_net_pct"] == 0.0,
        # The named falsifier: the sketch ordering must not be materially
        # BETTER than the exact ordering at low probe fractions.
        "e136_5a_falsifier_fired":
            low["lowrank"]["recall_wg"] > low["exact"]["recall_wg"],
        "e136_5a_recall_at_p010_shipped": low["exact"]["recall_wg"],
        "e136_5a_recall_at_p010_c1": low["lowrank"]["recall_wg"],
    }
    return summary, {"probe_ladder": table(LADDER_COLUMNS, rows)}


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


BUILDERS = {"0": rung0_summary, "0b": rung0b_summary,
            "5a": rung5a_summary,
            "1": rung1_summary, "2": rung2_summary}


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
        # Rung 2 is the only rung that times a decode leg. Its timing is real
        # but ungated and counterbalanced, so it stays `harness=local` with
        # both cool-gate fields false.
        "timing_valid": spec.get("timing_valid", False),
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
