"""Log the E156 reconciliation against advisor comment 5387591462 to the run."""
import json
import pathlib

import wandb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = json.loads(
    (ROOT / "research" / "e156-advisor-law-change-reconciliation.json").read_text()
)

PINNED = "62516c6f3799b66c91171ee13aa6816db5af197aa8c527cec0f6bb4026f0c7b7"

summary = {
    "e156_law_reconciliation": json.dumps(DOC, indent=1),
    "e156_uses_retracted_2_7769_scalar": 0,
    "e156_uses_finding_286_round_cost_law": 0,
    "e156_pricing_tag": "clean-no-host-transfer-term",
    "e156_uses_rule_163_1_42_scalar": 0,
    "e156_published_pct_constant": 0.100436,
    "e156_published_pct_constant_provenance":
        "FINDING 254 corrected model, campaign-ledger.md ~line 56988, an "
        "eight-raw-ratio order-statistic rebuild. Ranked prefill percent in, "
        "published median percent out. NO host-transfer term, so the RULE 166 "
        "retraction and RULE 174 both leave it untouched. It is the small-cut "
        "limit of that table, whose secant slope runs 0.1013 at a 2 percent cut "
        "to 0.1032 at 20 percent, so my constant is mildly conservative.",
    "e156_rule_174_applies": 0,
    "e156_rule_173_head_sha256_candidate_leg": PINNED,
    "e156_rule_173_head_sha256_base_arm_leg": PINNED,
    "e156_rule_173_head_digests_match_across_arms": 1,
    "e156_rule_173_head_is_pinned_not_declared": 1,
    "e156_finding_302_candidate_increases_depth": 0,
    "e156_finding_302_edl_candidate": 6.3766233766233764,
    "e156_finding_302_edl_base_arm": 6.376623376623376,
    "e156_finding_302_gate_biased": 0,
    "e156_finding_302_note":
        "The realised draft-length distribution is identical across arms to the "
        "last printed digit, with equal mtp_rounds and accepted_draft_rate, so "
        "the pinned-head penalty is paid by both arms and cancels exactly. "
        "Measured, not assumed. Must be re-checked on M5, where the changed "
        "kernel actually executes.",
}

run = wandb.init(
    project="qwen38-mlx-challenge-senpai",
    entity="wandb-applied-ai-team",
    id="e156alphonser1",
    resume="must",
)
run.summary.update(summary)
run.finish()
print("logged %d reconciliation fields to %s" % (len(summary), run.url))
