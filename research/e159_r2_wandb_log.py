#!/usr/bin/env python3
"""E159 R2: publish the depth/cap/width-wall evidence and the two withdrawals.

One run, six tables plus the deliverable summary.

  depth_sweep     the dense fixed proposed-draft-depth sweep, D = 0..8, with
                  the four per-leg quantities F9 asked for: window tokens,
                  rounds consumed, proposals issued and drafts accepted, and
                  the mean verified row count M_all.
  cap_sweep       the step 2 verify-width cap replay under four prices for the
                  width-6 marginal, with leave-one-prompt-out folds.
  ranked_prompts  the ranked anchor receipt 5a9f130a, per prompt.
  law_fits        the structured round-cost law a + b*E[streams] + c*E[width]
                  fitted on both hosts.
  pinned_b_scan   ranked c refitted at pinned values of the stream price.
  withdrawals     every claim this experiment retracted, with the evidence
                  that refuted it.

harness is labelled per row. No GPU leg ran in r2, so nothing here is a ranked
score and nothing here is gate-qualified. The local rows are replays of the r1
sweep, which was itself ungated.

Usage:
  python3 research/e159_r2_wandb_log.py --run-name e159-r2-width-wall-verdict
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

ASSIGNMENT_ID = "qwen38-r1-e159-every-arm-that-ever-lost-moved-the-same-one-number"
REVISION = "r2"
BASE_SHA = "74b5137e583d526fa86d859ea36166a8e2174d85"
HEAD_DIGEST = "dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9"
ANCHOR_RECEIPT = "5a9f130a"
WINDOW_TOKENS = 512

# D, width, R_dec ms, R_tot ms, accepted per round, alpha, mtp ms/token.
# Measured on M4 Pro under MLX_E159_FIXED_DRAFT_DEPTH, every leg
# all_tokens_matched = true.
DEPTH_SWEEP = [
    (0, 1, 65.813, 73.634, 0.0000, None, 73.6338),
    (1, 2, 69.484, 84.929, 0.9807, 0.9807, 42.9621),
    (2, 3, 71.561, 94.310, 1.9091, 0.9545, 32.4192),
    (3, 4, 78.323, 108.164, 2.8209, 0.9450, 28.3086),
    (4, 5, 91.634, 128.098, 3.6545, 0.9199, 27.5210),
    (5, 6, 127.084, 168.468, 4.2784, 0.8628, 31.9167),
    (6, 7, 138.065, 186.355, 5.1687, 0.8684, 30.2098),
    (7, 8, 145.655, 199.772, 5.9189, 0.8555, 28.8733),
    (8, 9, 185.427, 244.281, 6.5294, 0.8222, 32.4435),
]

# Inputs per group table, Qwen35.swift:1568, and the resulting stream count
# from activeInputGroups, Qwen35.swift:1715.
STREAMS_BY_WIDTH = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}

ADAPTIVE_LEG = {
    "window_tokens": WINDOW_TOKENS,
    "rounds": 78,
    "mean_proposed_q": 6.359,
    "accepted_per_round": 5.564,
    "alpha": 0.875,
    "mtp_ms_per_token": 28.68,
    "M_all": 7.359,
    "proposed_depth_histogram": {"3": 4, "4": 7, "5": 5, "6": 3, "7": 59},
}

CAP_SWEEP = [
    ("board deconvolved as fitted", 6.03, 3, 0.6189, -0.1778, 4, 3, 5, False),
    ("w6 at neighbour mean", 7.49, 3, 3.3793, 1.9435, 5, 3, 3, True),
    ("local wall transferred", 16.29, 3, 16.2014, 10.9661, 5, 3, 4, False),
    ("local wall transferred, w6 healed", 5.58, 6, 0.0000, -0.0951, 1, 5, 6, False),
]

# name, rounds, mean proposed q, accepted per round, round ms, serial ms, raw
RANKED_PROMPTS = [
    ("plutarch", 488, 0.1557, 0.049, 30.522, 37.9740, 1.26070),
    ("drama", 252, 2.2976, 1.032, 34.241, 37.9454, 2.12220),
    ("travel", 213, 2.6479, 1.404, 35.109, 37.9646, 2.42770),
    ("beagle", 110, 4.3818, 3.655, 44.983, 37.9208, 3.54530),
    ("republic", 93, 4.9892, 4.505, 47.723, 38.0130, 3.91990),
    ("essays", 92, 5.0870, 4.565, 48.932, 38.0201, 3.87040),
    ("medicine", 90, 5.2556, 4.689, 49.350, 37.9068, 3.90650),
    ("botany", 81, 6.1481, 5.321, 54.474, 37.9495, 3.93320),
]

LAW_FITS = [
    ("local", "a + b*G + c*M", 18.11, 2.62, 32.245, 3.302, 7.799, 1.003, 2.814, 0.99681),
    ("local", "a + c*M", 21.95, 10.58, None, None, 16.628, 1.776, 11.508, 0.93595),
    ("ranked", "a + b*G + c*M", 19.65, 2.00, 6.118, 2.638, 2.885, 0.658, 1.593, 0.97596),
    ("ranked", "a + c*M", 22.51, 2.07, None, None, 4.242, 0.397, 2.095, 0.95009),
    ("ranked, drop plutarch", "a + b*G + c*M", 16.13, 0.73, -0.271, 1.117, 5.421, 0.379, 0.449, 0.99766),
]

PINNED_B_SCAN = [
    (0.0, 5.335, 0.405),
    (5.0, 3.739, 1.030),
    (10.0, 2.143, 1.891),
    (20.291, -1.143, 3.722),
    (30.212, -4.310, 5.500),
]

WITHDRAWALS = [
    (
        "e159 arm projects +11.90 % published median from the local width-6 wall price",
        "withdrawn",
        "The local 30.2 ms per stream cannot transfer. Applied to the ranked "
        "median pair it forces a negative per-verified-row cost of -20.88 ms.",
        "own arithmetic on the F9 ranked table",
    ),
    (
        "e159 arm projects +10.08 % published median from a ranked stream price "
        "of 6.118 +- 2.638 ms",
        "withdrawn",
        "FINDING 279 measured the direct ranked intervention. onePass6 removes "
        "the second stream at width 6 and reads -0.0390 % +- 0.0799, z = -0.49. "
        "The cross-sectional fit predicted about +12.5 % on essays.",
        "ledger 318.1 FINDING 279",
    ),
    (
        "e159_width6_excess_mechanism = the cross-row QMV input-group partition, "
        "each group costing a full extra weight pass",
        "withdrawn",
        "FINDING 279 shows the ceil(m/ipg) re-reads are cache-served at 0.64 % "
        "of the DRAM prediction, and falsifies the source claim at "
        "Qwen35.swift:1817-1819 that weight traffic dominates every routed cell.",
        "ledger 318.1 FINDING 279",
    ),
    (
        "instantiate IPG = 6 so activeInputGroups(6) == 1",
        "withdrawn, previously closed",
        "E137 Route B built this bit-exactly. RULE 155 prices it at ranked "
        "+1.0583 % because it replaces the already balanced [3+3] partition and "
        "still pays the wider template's register and occupancy tier.",
        "ledger 314.7 RULE 155, item 230",
    ),
    (
        "width 9 at IPG = 5 is an uncosted stream win",
        "withdrawn, void",
        "SEGMENTED_VERIFY_DEPTH_CAP = 7 caps M at 8, so width 9 is never "
        "reached by the shipped scheduler. My D=8 leg reached it only because "
        "MLX_E159_FIXED_DRAFT_DEPTH replaces the draft policy and bypasses the "
        "cost model cap.",
        "ledger 318.2 ADVISOR ERROR 188(b)",
    ),
]

SURVIVING = [
    (
        "e159_verdict = no_arm_clears",
        "The verify-width cap arm does not clear +0.30 % candidate leg under "
        "any of the four width-6 prices, and its leave-one-prompt-out "
        "selection is unstable in three of the four.",
    ),
    (
        "the width-6 step is independent of the QMV pass count",
        "FINDING 185 measured the 5 to 6 step under onePass67, where plan(5) "
        "and plan(6) are both a single pass. This sweep measured the same step "
        "under the shipped table, where plan(6) is the two-pass [3+3]. Two "
        "different pass counts at width 6, the same step. The pass-count "
        "explanation is now dead by measurement as well as by construction.",
    ),
    (
        "e159_split_fires_at_width_6 = true, and it is bounded out as the "
        "dominant term",
        "The wide-decode exactness chunk does run at width 6. It adds 96 graph "
        "nodes per round and 34 to 67 MB of extra KV traffic, which is 0.3 to "
        "1.3 ms at M4 Pro bandwidth, at most 6 % of the 23.3 ms excess.",
    ),
    (
        "the GDN scan geometry explanation is excluded at source",
        "GatedDelta.swift:167-176 dispatches grid (32, Dv, B*Hv) with T as a "
        "scalar and a sequential t loop, so the scan is T independent. The "
        "comment at Qwen36MTPBlockSession.swift:1029 is factually wrong.",
    ),
    (
        "e159_double_update_is_safe = true, but it is a regression",
        "Two KVCacheSimple updates write disjoint contiguous ranges and give "
        "byte-identical windows, but AttentionUtils.swift:117 shows the single "
        "update already produces both windows, so the second call only adds an "
        "MLX buffer donation loss of about 67 MB per round.",
    ),
]

DELIVERABLES = {
    "e159_wall_identified": True,
    "e159_wall_index": 6,
    "e159_wall_second_index": 9,
    "e159_wall_marginal_W_ms_w6": 35.450,
    "e159_wall_marginal_W_ms_w9": 39.772,
    "e159_wall_excess_over_neighbours_ms_w6": 23.304,
    "e159_wall_excess_over_neighbours_ms_w9": 30.486,
    "e159_flat_marginal_mean_ms": 7.399,
    "e159_flat_marginal_min_ms": 2.077,
    "e159_flat_marginal_max_ms": 13.311,
    "e159_flat_marginal_n": 6,
    "e159_h_subwall_ms_per_row": 7.399,
    "e159_law_form": "step_at_width_6_and_9_over_a_flat_sub_wall_slope",
    "e159_arm_marginal_ms": None,
    "e159_predicted_median_pct": None,
    "e159_plutarch_nondrafting": 449,
    "e159_plutarch_nondrafting_of": 488,
    "e159_median_pair_changed": False,
    "e159_verdict": "no_arm_clears",
    "e159_local_divergence": 0,
    "e159_head_provenance_sha256": HEAD_DIGEST,
    "e159_split_fires_at_width_6": True,
    "e159_extra_graph_nodes_per_round_at_width_6": 96,
    "e159_width6_excess_mechanism": "unexplained",
    "e159_double_update_is_safe": True,
    "e159_double_update_recommended": False,
    "rule79_not_evidence": True,
    "official_or_ranked_score": False,
    "cool_gate_passed_real_gate": False,
    "gate_qualified_for_timing": False,
    "harness": "local",
    "gpu_legs_run_in_r2": 0,
}


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def depth_table() -> wandb.Table:
    cols = [
        "proposed_depth_D",
        "verify_width",
        "window_tokens",
        "rounds",
        "proposals_issued",
        "drafts_accepted",
        "accepted_per_round",
        "M_all",
        "alpha",
        "R_decode_ms",
        "R_total_ms",
        "mtp_ms_per_token",
        "marginal_R_decode_ms",
        "input_groups",
        "delta_groups",
        "all_tokens_matched",
    ]
    table = wandb.Table(columns=cols)
    prev = None
    for depth, width, r_dec, r_tot, acc, alpha, mtp in DEPTH_SWEEP:
        rounds = round(WINDOW_TOKENS / (acc + 1.0))
        marginal = None if prev is None else round(r_dec - prev, 3)
        groups = STREAMS_BY_WIDTH[width]
        prev_groups = STREAMS_BY_WIDTH.get(width - 1, groups)
        table.add_data(
            depth,
            width,
            WINDOW_TOKENS,
            rounds,
            rounds * depth,
            round(rounds * acc),
            acc,
            float(depth + 1),
            alpha,
            r_dec,
            r_tot,
            mtp,
            marginal,
            groups,
            groups - prev_groups,
            True,
        )
        prev = r_dec
    return table


def cap_table() -> wandb.Table:
    table = wandb.Table(
        columns=[
            "scenario",
            "width6_price_ms",
            "best_cap",
            "predicted_median_pct",
            "predicted_candidate_leg_pct",
            "same_sign_of_8",
            "loo_cap_lo",
            "loo_cap_hi",
            "loo_stable",
        ]
    )
    for row in CAP_SWEEP:
        table.add_data(*row)
    return table


def ranked_table() -> wandb.Table:
    priced = json.loads((ARTIFACTS / "e159_ranked_stream_price.json").read_text())
    streams = {r["prompt"]: r["expected_streams"] for r in priced["per_prompt"]}
    table = wandb.Table(
        columns=[
            "prompt",
            "rounds",
            "mean_proposed_q",
            "accepted_per_round",
            "round_ms",
            "serial_ms",
            "raw",
            "M_all",
            "expected_weight_streams",
        ]
    )
    for name, rounds, q, acc, round_ms, serial_ms, raw in RANKED_PROMPTS:
        table.add_data(
            name,
            rounds,
            q,
            acc,
            round_ms,
            serial_ms,
            raw,
            round(1.0 + q, 4),
            streams[name],
        )
    return table


def fit_table() -> wandb.Table:
    table = wandb.Table(
        columns=[
            "harness",
            "fit",
            "a_ms",
            "a_se",
            "b_ms_per_stream",
            "b_se",
            "c_ms_per_row",
            "c_se",
            "resid_sd_ms",
            "r2",
        ]
    )
    for row in LAW_FITS:
        table.add_data(*row)
    return table


def scan_table() -> wandb.Table:
    table = wandb.Table(columns=["b_pinned_ms", "c_fitted_ms_per_row", "resid_sd_ms"])
    for row in PINNED_B_SCAN:
        table.add_data(*row)
    return table


def withdrawal_table() -> wandb.Table:
    table = wandb.Table(columns=["claim", "status", "refuting_evidence", "source"])
    for row in WITHDRAWALS:
        table.add_data(*row)
    return table


def surviving_table() -> wandb.Table:
    table = wandb.Table(columns=["result", "evidence"])
    for row in SURVIVING:
        table.add_data(*row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="e159-r2-width-wall-verdict")
    args = parser.parse_args()

    head = git_sha()
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.run_name,
        job_type="analysis",
        tags=["e159", "r2", "depth-price", "width-wall", "harness-local", "no-gpu-leg"],
        config={
            "assignment_id": ASSIGNMENT_ID,
            "revision": REVISION,
            "pr": 159,
            "base_sha": BASE_SHA,
            "head_sha": head,
            "anchor_receipt": ANCHOR_RECEIPT,
            "head_provenance_sha256": HEAD_DIGEST,
            "host": "M4 Pro",
            "window_tokens": WINDOW_TOKENS,
            "harness": "local",
            "official_or_ranked_score": False,
            "rule79_not_evidence": True,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "gpu_legs_run_in_r2": 0,
            "adaptive_leg": ADAPTIVE_LEG,
        },
    )

    run.log(
        {
            "depth_sweep": depth_table(),
            "cap_sweep": cap_table(),
            "ranked_prompts": ranked_table(),
            "law_fits": fit_table(),
            "pinned_b_scan": scan_table(),
            "withdrawals": withdrawal_table(),
            "surviving_results": surviving_table(),
        }
    )

    run.summary.update(DELIVERABLES)

    artifact = wandb.Artifact("e159-r2-analysis", type="analysis")
    for path in sorted(ARTIFACTS.glob("*.json")):
        artifact.add_file(str(path), name=path.name)
    for script in ("e159_cap_replay.py", "e159_qmv_group_law.py", "e159_ranked_stream_price.py"):
        candidate = HERE / script
        if candidate.exists():
            artifact.add_file(str(candidate), name=script)
    run.log_artifact(artifact)

    print(json.dumps({"run_id": run.id, "url": run.url}, indent=1))
    run.finish()


if __name__ == "__main__":
    main()
