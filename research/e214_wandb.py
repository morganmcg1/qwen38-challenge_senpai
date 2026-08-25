#!/usr/bin/env python3
"""Publish the E214 implementation evidence to W&B.

    python3 research/e214_wandb.py [--dry-run]

Two runs in group `qwen38-r1-e214-step-price-impl`:

  gate     the open-loop agreement gate (`research/e214_open_loop_gate.py`):
           table identity against the fitted artifact, grid agreement, the
           round-for-round replay of the traced fixture, and the positive
           control that proves the comparison can fail.
  confirm  the gated 512-token `--local-submit` confirmation: exactness, row
           ledger, schedule census against the same run on the base, and the
           local absolute seconds per token.

RULE 79. No local timing number here is a price for this change. The local
serial and MTP legs both run the candidate binary, and a schedule change moves
the MTP leg only, so the local ratio is not the ranked ratio. Every local
figure is published with `NOT-A-PRICE` in its name or its config, and the value
claim stays the harness=ranked desk projection the receipt settles.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e214-step-price-impl"
ARTIFACTS = pathlib.Path("research/e214-artifacts")
DESK = pathlib.Path("research/e211-artifacts/step-price.json")
ASSIGNMENT_BASE = "e0c7a026aebc9dbdd579577964ad413a11ede6fc"
DESK_ARTIFACT_SHA256 = \
    "3c5295cc1d05816e01da571a8053e73aa09ed1cc9e72e0ea192aaa415318f4ae"


def head_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def load(name):
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def identity(extra):
    return {
        "experiment": "e214-step-price-implementation",
        "assignmentBaseSha": ASSIGNMENT_BASE,
        "commitSha": head_sha(),
        "deskArtifact": "research/e211-artifacts/step-price.json",
        "deskArtifactSha256": DESK_ARTIFACT_SHA256,
        "deskArtifactKey": "receipt_proof.forward_guarded",
        "shippedArm": "stepq",
        "policyInputs": "marginal row index and the shipped acceptance signal "
                        "only; no prompt feature, no runtime prompt "
                        "conditioning, no phase detection, no cross-request "
                        "state",
        "officialOrRankedScore": False,
        **extra,
    }


def desk_projection():
    blob = json.loads(DESK.read_text())
    fg = blob["receipt_proof"]["forward_guarded"]
    env = blob["validated_envelope"]["arms"]["forward_guarded"]
    out = {
        "harness": "ranked",
        "deskWorstReadingDeltaPct": fg["worst_delta_pct"],
        "deskWorstLooHonestDeltaPct": fg["worst_loo_honest_delta_pct"],
        "deskWorstSinglePromptDeltaPct": fg["worst_prompt_delta_pct"],
        "deskPaidUnderUncorrectedStepPct":
            fg["paid_under_uncorrected_step_pct"],
        "deskLawSet": ",".join(fg["law_set"]),
        "deskGreedyTableAgreement": fg["greedy_agreement"],
        "deskRelocationVsCap7": env["relocation_vs_cap7"],
        "deskInsideValidatedEnvelope": env["inside_validated_envelope"],
        "deskValidatedEnvelopeBar":
            blob["validated_envelope"]["validated_bar"],
        "receiptChannelPct":
            blob["instrument_validation"]["receipt_channel_pct"],
        "receiptA": blob["receipt_A"],
        "crown": blob["crown"],
    }
    for d, value in enumerate(fg["price_marginal"]):
        out["shippedPrice/row%d" % d] = value
        out["shippedThreshold/row%d" % d] = fg["thresholds"][d]
        out["shippedCut/row%d" % d] = fg["cuts"][d]
    return out


def gate_summary(gate):
    identity_block = gate["table_identity"]
    grid = gate["grid"]
    live = gate["round_for_round"]
    control = gate["positive_control"]
    census = gate["census"]
    out = {
        "gatePass": gate["pass"],
        "tableArm": identity_block["arm"],
        "tableArmIsShipped": identity_block["arm_is_shipped"],
        "tableWitnessIdentical": identity_block["witness_identical"],
        "tableMarginalBitIdentical":
            identity_block["marginal_bit_identical"],
        "tableCumulativeBitIdentical":
            identity_block["cumulative_bit_identical"],
        "tableMaxAbsMarginalDelta":
            identity_block["max_abs_marginal_delta"],
        "gridNodes": grid["grid_nodes"],
        "gridCutsMatch": grid["cuts_match"],
        "gridGreedyAgreement": grid["greedy_agreement"],
        "gridGreedyMismatches": grid["greedy_mismatches"],
        "replayRounds": live["rounds_checked"],
        "replayWalkSteps": live["steps_checked"],
        "replayDepthAgreement": live["depth_agreement"],
        "replaySchedByteIdentical": live["sched_byte_identical"],
        "replayFieldMaxUlp": live["field_max_ulp"],
        "replayMismatches": live["mismatches_total"],
        "controlDepthAgreement": control["depth_agreement"],
        "controlMismatches": control["mismatches_total"],
        "controlFieldMaxUlp": control["field_max_ulp"],
        "legEdl": census["edl"],
        "legAcceptedMean": census["accepted_mean"],
        "legTokens": gate["tokens"],
        "legAllTokensMatched":
            gate.get("score_metrics", {}).get("all_tokens_matched"),
        "legResidualDivergenceCount":
            gate.get("score_metrics", {}).get("residual_divergence_count"),
        "legHeadProvenanceSha256":
            gate.get("score_metrics", {}).get("head_provenance_sha256"),
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "timingClaimsPermitted": False,
    }
    for d, value in enumerate(census["depth_hist"]):
        out["legDepthHist/d%d" % d] = value
    for d, value in enumerate(identity_block["shipped_marginal"]):
        out["builtPrice/row%d" % d] = value
    return out


def confirm_summary(confirm):
    """The gated 512-token confirmation, plus the same run on the base."""
    out = {"harness": "local"}
    for arm in ("candidate", "base"):
        leg = confirm.get(arm)
        if not leg:
            continue
        for key, value in leg.items():
            out["%s/%s" % (arm, key)] = value
    for key, value in confirm.get("delta", {}).items():
        out["delta/%s" % key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    projection = desk_projection()
    gate = load("open-loop-gate.json")
    confirm = load("confirm.json")

    specs = {}
    if gate:
        specs["gate"] = (
            {"job_type": "gate", "name": "e214-open-loop-gate",
             "config": identity({
                 "harness": "local trace against the ranked desk table",
                 "deliverable": "the shipped table reproduces the fitted "
                                "depth map on the grid and round for round "
                                "on the traced fixture",
                 "gpuUsed": True,
                 "coolGatePassedRealGate": False,
                 "gateQualifiedForTiming": False,
                 "timingClaimsPermitted": False})},
            {**projection, **gate_summary(gate)})
    if confirm:
        specs["confirm"] = (
            {"job_type": "confirm", "name": "e214-gated-512-confirm",
             "config": identity({
                 "harness": "local",
                 "deliverable": "exactness, row-ledger closure and the "
                                "schedule census of the shipped table over "
                                "512 decode tokens",
                 "gpuUsed": True,
                 "coolGatePassedRealGate": True,
                 "gateQualifiedForTiming": True,
                 "timingClaimsPermitted": False,
                 "localTimingIsNotAPrice": True})},
            {**projection, **confirm_summary(confirm)})

    if not specs:
        raise SystemExit("no E214 artifacts to publish yet")

    if args.dry_run:
        print(json.dumps({k: {"config": s["config"], "summary": v}
                          for k, (s, v) in specs.items()},
                         indent=1, sort_keys=True, default=float))
        return

    import wandb

    existing = {}
    path = ARTIFACTS / "wandb.json"
    if path.exists():
        existing = {k: v["run"] for k, v in json.loads(path.read_text()).items()}

    published = dict(json.loads(path.read_text())) if path.exists() else {}
    for key, (spec, summary) in specs.items():
        extra = ({"id": existing[key], "resume": "allow"}
                 if key in existing else {})
        run = wandb.init(project=PROJECT.split("/")[-1],
                         entity=PROJECT.split("/")[0], group=GROUP,
                         **spec, **extra)
        run.summary.update(summary)
        published[key] = {"run": run.id, "url": run.url}
        run.finish()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(published, indent=1) + "\n")
    print(json.dumps(published, indent=1))


if __name__ == "__main__":
    main()
