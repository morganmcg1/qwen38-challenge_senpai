"""Record the E9 r3 source-analysis result in W&B.

E9 r3 produced no timed arm: the lever it was assigned to tune was deleted from
the base, so Part A ended the experiment before any timing budget was spent.
Logging the finding anyway leaves a searchable record in the experiment group,
so the next agent to reach for a draft-readout bit width finds the negative
instead of an empty group.

Every value here is a byte count, a symbol count, a source fact, or a probe
counter. No speed or acceptance number is logged, because none was measured
validly.
"""

import wandb

ROWS_PADDED = 98_336
COLS = 5_120
GROUP = 64
STREAM_PEAK_BPS = 227_128_791_836.97
PINNED_BF16_HEAD_BYTES = 849_400_347
DECLARED_Q4_HEAD_BYTES = 238_934_129
KERNEL_SPEEDUP_3BIT = 0.242


def affine_bytes(bits: int) -> int:
    weight = ROWS_PADDED * COLS * bits // 8
    scales = ROWS_PADDED * (COLS // GROUP) * 2
    return weight + 2 * scales


def main() -> None:
    b4, b3, b2 = affine_bytes(4), affine_bytes(3), affine_bytes(2)

    run = wandb.init(
        project="qwen38-mlx-challenge-senpai",
        entity="wandb-applied-ai-team",
        group="qwen38-r1-e9-draft-bits-default",
        job_type="source-analysis",
        name="e9r3-draft-bits-lever-void",
        tags=["e9", "r3", "negative", "no-timed-arm", "source-analysis"],
        config={
            "assignment_id": "qwen38-r1-e9-draft-bits-default",
            "revision_id": "r3",
            "student": "qwen-askeladd",
            "base_sha": "bc5e15fd8121db9b6dd26570c8fa42a07c1a4ce6",
            "research_base_sha": "fe38ecc21e4084e4d17dac3aa76264bb5897a614",
            "prior_r2_base_sha": "8970d775a63a28b610fd418c68873c236ce6b86c",
            "submitted_path_bytes_changed": 0,
            "editable_source_bytes": 2_402_203,
            "editable_growth_bytes": 0,
            "timed_arms_run": 0,
            # Draft schedule constants on this base, which differ from r2's and
            # are why r2's acceptance result does not carry over.
            "head_step_cost_ratio": 0.18,
            "sdpa_width_wall_depth_cap": 5,
            "segmented_verify_depth_cap": 8,
            "segmented_streak_gate": 3,
            "compact_draft_padded_count": ROWS_PADDED,
            "hidden_size": COLS,
            "quant_group_size": GROUP,
            "backbone_lm_head_bits": 4,
        },
    )

    # A1: symbol counts in Qwen35.swift, old base vs new base.
    for symbol, old, new in (
        ("draftHeadBits", 3, 0),
        ("requantizedDraftHead", 2, 0),
        ("MLX_QWEN_MTP_DRAFT_BITS", 2, 0),
        ("makeCompactDraftHead", 3, 3),
    ):
        run.summary[f"a1/symbol_count_old_base/{symbol}"] = old
        run.summary[f"a1/symbol_count_new_base/{symbol}"] = new

    run.summary["a1/lever_present_on_new_base"] = False
    run.summary["a1/three_bit_default_is_reimplementation"] = True
    run.summary["a1/route_b_shipped"] = False
    run.summary["a1/declared_head_was_already_q4_at_old_base"] = True

    # A1 runtime probe (untimed; path counters only).
    run.summary["a1/probe_draft_head_w_present"] = False
    run.summary["a1/probe_make_compact_draft_head_reached"] = True
    run.summary["a1/probe_compact_fused_calls"] = 1
    run.summary["a1/probe_declared_head_calls"] = 0
    run.summary["a1/probe_lm_head_bits"] = 4
    run.summary["a1/probe_lm_head_group_size"] = GROUP
    run.summary["a1/probe_all_tokens_matched"] = True
    run.summary["a1/probe_reference_checked_rows"] = 16
    run.summary["a1/probe_worker_sha256"] = (
        "4aa4cdf6b7910326dd0c0cc91c76467e1b3cd521b8cd964772d48dacf87a40aa"
    )

    # A2: the slice is of the backbone head, so r2 was always double quantizing.
    run.summary["a2/slice_source"] = "backbone_lm_head"
    run.summary["a2/was_always_double_quantization"] = True
    run.summary["a2/r2_acceptance_delta_english"] = 0.019
    run.summary["a2/r2_acceptance_delta_technical"] = 0.010
    run.summary["a2/r2_acceptance_delta_narrative"] = -0.007

    # A3: readout byte model, identical on both bases.
    for bits, total in ((4, b4), (3, b3), (2, b2)):
        run.summary[f"a3/readout_bytes_{bits}bit"] = total
        run.summary[f"a3/readout_ms_{bits}bit"] = 1e3 * total / STREAM_PEAK_BPS
    run.summary["a3/readout_bytes_saved_3bit"] = b4 - b3
    run.summary["a3/readout_bytes_saved_frac_3bit"] = (b4 - b3) / b4
    run.summary["a3/readout_ms_saved_3bit"] = 1e3 * (b4 - b3) / STREAM_PEAK_BPS

    local_share = b4 / (PINNED_BF16_HEAD_BYTES + b4)
    ranked_share = b4 / (DECLARED_Q4_HEAD_BYTES + b4)
    run.summary["a3/pinned_bf16_head_bytes"] = PINNED_BF16_HEAD_BYTES
    run.summary["a3/declared_q4_head_bytes"] = DECLARED_Q4_HEAD_BYTES
    run.summary["a3/readout_share_local_path"] = local_share
    run.summary["a3/readout_share_ranked_path"] = ranked_share
    run.summary["a3/head_step_cap_local_path"] = -local_share * KERNEL_SPEEDUP_3BIT
    run.summary["a3/head_step_cap_ranked_path"] = -ranked_share * KERNEL_SPEEDUP_3BIT
    run.summary["a3/local_understates_ranked_effect"] = True

    run.summary["decision"] = "dead"
    run.summary["recommendation"] = "close"
    run.summary["r2_gain_was_acceptance_coin_flip"] = True
    run.summary["r2_floors_valid_on_this_base"] = False

    run.summary["notes"] = (
        "Lever deleted upstream: draftHeadBits/requantizedDraftHead/"
        "MLX_QWEN_MTP_DRAFT_BITS absent from Qwen35.swift on base bc5e15fd. "
        "makeCompactDraftHead now inherits bits: quantized.bits. No timed arm "
        "run; no speed or acceptance metric is logged because none was measured "
        "validly. Untimed probe confirms the compact readout path is reached."
    )

    artifact = wandb.Artifact("e9r3-source-analysis", type="result")
    artifact.add_file("research/results/qwen38-r1-e9-draft-bits-default.md")
    artifact.add_file("research/e9r3_readout_share.py")
    run.log_artifact(artifact)

    print(f"wandb run id: {run.id}")
    print(f"wandb run url: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
