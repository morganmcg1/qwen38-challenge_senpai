#!/usr/bin/env python3
"""Publish the E211 evidence to W&B.

    python3 research/e211_wandb.py [--dry-run]

One run per cost law in group `qwen38-r1-e211-step-aware-depth-price`, plus a
`receipt-proof` run for the minimax table that pays under every law. Every
number is harness=ranked and comes from the FINDING 520 instrument. There is
no timed leg and no GPU in this experiment, so every measurement field records
`desk-replay` rather than borrowing a value from an unrelated run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e211-step-aware-depth-price"
ARTIFACTS = pathlib.Path("research/e211-artifacts")
ASSIGNMENT_BASE = "d9460346662bc8229696550550e9211aa40fdb6a"
LAW_KEYS = ("smooth", "step", "step_e208")


def head_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def payload():
    return json.loads((ARTIFACTS / "step-price.json").read_text())


def identity(blob, extra):
    return {
        "harness": "ranked",
        "experiment": "e211-step-aware-depth-price",
        "assignmentBaseSha": ASSIGNMENT_BASE,
        "commitSha": head_sha(),
        "instrument": "FINDING 520 survival-pinned latent-q, imported "
                      "unmodified from e201_online_cap",
        "policyFamily": "greedy costModelDepth over a free marginal price "
                        "table; inputs are the row index and the shipped "
                        "acceptance signal only",
        "seed": blob["seed"],
        "worker": "desk-replay", "host": "desk-replay",
        "gpuUsed": False,
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "timingClaimsPermitted": False,
        "officialOrRankedScore": False,
        **extra,
    }


def gate_summary(blob):
    gate = blob["reproduction_gate"]
    val = blob["instrument_validation"]
    return {
        "reproductionGateMaxRelativeError": gate["max_relative_error"],
        "reproductionGatePass": gate["pass"],
        "validationCap7Pct": val["error_pct"]["cap7_pct"],
        "validationCap4Pct": val["error_pct"]["cap4_pct"],
        "validationCap5Pct": val["error_pct"]["cap5_pct"],
        "validationPass": val["pass"],
        "receiptChannelPct": val["receipt_channel_pct"],
        "receiptA": blob["receipt_A"], "crown": blob["crown"],
    }


def law_summary(blob, key):
    law = blob["laws"][key]
    dec = law["decomposition"]
    ship, opt = law["shipped"], law["optimum"]
    out = {
        "harness": "ranked", "law": key,
        "rankedDr9Ms": blob["nine_row_cell"]["ranked_dr9_ms"][key],
        "shippedPublishedMedian": ship["published_median"],
        "optimumPublishedMedian": opt["published_median"],
        "inSampleDeltaPct": law["in_sample_delta_pct"],
        "looHonestPublishedMedian": law["loo_honest"]["published_median"],
        "looHonestDeltaPct": law["loo_honest_delta_pct"],
        "looWorstDeltaPct": law["loo_worst_delta_pct"],
        "looWorstPrompt": law["loo_worst_prompt"],
        "oracleQPublishedMedian": law["oracle_q_median"],
        "fractionOfOracleCeiling":
            (opt["published_median"] - ship["published_median"])
            / (law["oracle_q_median"] - ship["published_median"]),
        "levelOnlyDeltaPct": dec["level_pct"],
        "levelOnlyLooHonestPct": dec["loo_honest_level_pct"],
        "shapeAtShippedLevelDeltaPct": dec["structure_at_shipped_level_pct"],
        "structurePremiumPct": dec["structure_premium_pct"],
        "structurePremiumLooHonestPct":
            dec["loo_honest_structure_premium_pct"],
        "bestUniformPrice": dec["level_only"]["marginal"][0],
        "greedyTableAgreement": opt["greedy_agreement"],
        "greedyTableMismatches": opt["greedy_mismatches"],
        "worstRoundMassRelocated":
            law["anchor_distance"]["optimum"]["worst_round_mass_relocated"],
    }
    for d, value in enumerate(opt["price_marginal"]):
        out["optimumPrice/row%d" % d] = value
        out["optimumThreshold/row%d" % d] = opt["thresholds"][d]
    for name in ship["per_prompt_raw"]:
        out["prompt/%s/shippedRaw" % name] = ship["per_prompt_raw"][name]
        out["prompt/%s/optimumRaw" % name] = opt["per_prompt_raw"][name]
        out["prompt/%s/looHonestRaw" % name] = \
            law["loo_honest"]["per_prompt_raw"][name]
        out["prompt/%s/shippedEdl" % name] = ship["edl"][name]
        out["prompt/%s/optimumEdl" % name] = opt["edl"][name]
        out["prompt/%s/looDropDeltaPct" % name] = \
            law["loo_robust"][name]["delta_pct"]
    for fitted, row in blob["cross_law"][key]["fitted"].items():
        out["paidUnderThisLaw/fitted_%s/deltaPct" % fitted] = row["delta_pct"]
    return out


def proof_summary(blob):
    proof = blob["receipt_proof"]
    guarded = proof["guarded"]
    env = blob["validated_envelope"]
    out = {
        "harness": "ranked", "law": "minimax-over-all-three",
        "worstDeltaPct": proof["worst_delta_pct"],
        "worstLooHonestDeltaPct": proof["worst_loo_honest_delta_pct"],
        "worstSinglePromptDeltaPct": proof["worst_prompt_delta_pct"],
        "greedyTableAgreement": proof["greedy_agreement"],
        "greedyTableMismatches": proof["greedy_mismatches"],
        "worstCrossLawTransferOfPerLawTablePct":
            blob["cross_law"]["worst_transfer_delta_pct"],
        "worstRoundMassRelocated":
            proof["anchor_distance"]["worst_round_mass_relocated"],
        "guardedWorstDeltaPct": guarded["worst_delta_pct"],
        "guardedWorstLooHonestDeltaPct":
            guarded["worst_loo_honest_delta_pct"],
        "guardedWorstSinglePromptDeltaPct":
            guarded["worst_prompt_delta_pct"],
        "guardedGreedyTableAgreement": guarded["greedy_agreement"],
        "validatedEnvelopeBar": env["validated_bar"],
    }
    for d, value in enumerate(proof["price_marginal"]):
        out["proofPrice/row%d" % d] = value
        out["proofThreshold/row%d" % d] = proof["thresholds"][d]
        out["guardedPrice/row%d" % d] = guarded["price_marginal"][d]
        out["guardedThreshold/row%d" % d] = guarded["thresholds"][d]
    for key in LAW_KEYS:
        out["paidUnder/%s/publishedMedian" % key] = \
            proof["per_law"][key]["published_median"]
        out["paidUnder/%s/deltaPct" % key] = proof["per_law"][key]["delta_pct"]
        out["paidUnder/%s/looHonestDeltaPct" % key] = \
            proof["loo_honest"][key]["delta_pct"]
        out["paidUnder/%s/worstPromptDeltaPct" % key] = \
            proof["per_law"][key]["worst_prompt_delta_pct"]
        out["guardedPaidUnder/%s/publishedMedian" % key] = \
            guarded["per_law"][key]["published_median"]
        out["guardedPaidUnder/%s/deltaPct" % key] = \
            guarded["per_law"][key]["delta_pct"]
        out["guardedPaidUnder/%s/looHonestDeltaPct" % key] = \
            guarded["loo_honest"][key]["delta_pct"]
        out["guardedPaidUnder/%s/worstPromptDeltaPct" % key] = \
            guarded["per_law"][key]["worst_prompt_delta_pct"]
    for name, value in proof["edl"].items():
        out["prompt/%s/proofEdl" % name] = value
    for name, arm in env["arms"].items():
        out["envelope/%s/relocationVsCap7" % name] = arm["relocation_vs_cap7"]
        out["envelope/%s/envelopeRatio" % name] = arm["envelope_ratio"]
        out["envelope/%s/insideValidatedEnvelope" % name] = \
            arm["inside_validated_envelope"]
    for name, value in env["validated_relocation"].items():
        out["envelope/validated/%s" % name] = value
    for arm in ("forward", "forward_guarded"):
        fw = proof[arm]
        out["%s/lawSet" % arm] = ",".join(fw["law_set"])
        out["%s/worstDeltaPct" % arm] = fw["worst_delta_pct"]
        out["%s/worstLooHonestDeltaPct" % arm] = \
            fw["worst_loo_honest_delta_pct"]
        out["%s/worstSinglePromptDeltaPct" % arm] = \
            fw["worst_prompt_delta_pct"]
        out["%s/paidUnderUncorrectedStepPct" % arm] = \
            fw["paid_under_uncorrected_step_pct"]
        out["%s/greedyTableAgreement" % arm] = fw["greedy_agreement"]
        for d, value in enumerate(fw["price_marginal"]):
            out["%s/price/row%d" % (arm, d)] = value
            out["%s/threshold/row%d" % (arm, d)] = fw["thresholds"][d]
        for key in fw["law_set"]:
            out["%s/paidUnder/%s/publishedMedian" % (arm, key)] = \
                fw["per_law"][key]["published_median"]
            out["%s/paidUnder/%s/deltaPct" % (arm, key)] = \
                fw["per_law"][key]["delta_pct"]
            out["%s/paidUnder/%s/looHonestDeltaPct" % (arm, key)] = \
                fw["loo_honest"][key]["delta_pct"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blob = payload()
    gates = gate_summary(blob)
    specs = {}
    for key in LAW_KEYS:
        specs[key] = (
            {"job_type": "desk", "name": "e211-price-%s" % key.replace("_",
                                                                       "-"),
             "config": identity(blob, {
                 "costLaw": key,
                 "costTableMs": blob["laws"][key]["cost_table_ms"],
                 "deliverable": "free optimum over the marginal depth-price "
                                "table under one 9-row reading"})},
            {**gates, **law_summary(blob, key)})
    specs["proof"] = (
        {"job_type": "desk", "name": "e211-receipt-proof",
         "config": identity(blob, {
             "costLaw": "minimax over smooth, step and step_e208",
             "deliverable": "one price table that pays under every live "
                            "reading of the 9-row cell"})},
        {**gates, **proof_summary(blob)})

    if args.dry_run:
        print(json.dumps({k: {"config": s["config"], "summary": v}
                          for k, (s, v) in specs.items()},
                         indent=1, sort_keys=True, default=float))
        return

    import wandb

    # Re-publishing must update the existing runs, not add a second set that
    # a reader would have to disambiguate.
    existing = {}
    path = ARTIFACTS / "wandb.json"
    if path.exists():
        existing = {k: v["run"] for k, v in json.loads(path.read_text()).items()}

    published = {}
    for key, (spec, summary) in specs.items():
        extra = ({"id": existing[key], "resume": "allow"}
                 if key in existing else {})
        run = wandb.init(project=PROJECT.split("/")[-1],
                         entity=PROJECT.split("/")[0], group=GROUP,
                         **spec, **extra)
        run.summary.update(summary)
        published[key] = {"run": run.id, "url": run.url}
        run.finish()
    (ARTIFACTS / "wandb.json").write_text(
        json.dumps(published, indent=1) + "\n")
    print(json.dumps(published, indent=1))


if __name__ == "__main__":
    main()
