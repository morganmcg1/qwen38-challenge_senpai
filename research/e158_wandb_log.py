#!/usr/bin/env python3
"""Publish the E158 head census, precision decomposition and island curve to W&B.

    usage: research/e158_wandb_log.py [--dry]

E158 asks two questions. First, whether the proposal head's remaining error is
precision or capacity, by re-running the E155 recall audit under the
organizer-pinned bf16 head and under the declared q2/q4 head. Second, what the
four precision-island arms of the declared head are worth.

No run here is a gated measurement or a score. The audit adds a
full-vocabulary lm_head projection per proposal slot inside the measured
block, so every leg records `timing_valid=false`, and no leg took the real
cool gate. Those fields are published verbatim as false.

Every published-percent figure is PROVISIONAL: advisor F4 retracted the
acceptance-point and modelled-round exchange rates while the ranked round-cost
law is refit. The byte deltas and the acceptance points themselves are not
provisional.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e158-head-precision-census"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e158-artifacts")
RAW = pathlib.Path(".mlxfast-private/e158")

BASE_SHA = "5f60bea8e154163e562287f1317a845ef10b0b44"
AUDIT_BUILD_COMMIT = "305787da"
AUDIT_WORKER_SHA256 = (
    "b3f5d6cfdce13f645e6ea2b32eae4b637fe0f84d52e68a50e4b4ff49e6d91c25")
REVERTED_WORKER_SHA256 = (
    "696416f78f1bfbf0103ec433a5d18d5b0738d118002dd4ce7049933f58a69ff3")
# advisor F4: PROVISIONAL while the ranked round-cost law is refit.
PCT_PER_ACCEPTANCE_POINT = 2.6701
ARMS = ("all", "q", "kv", "none")


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten("%s/%s" % (prefix, key) if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def table(rows: list[dict]) -> wandb.Table:
    columns = sorted({key for row in rows for key in row})
    built = wandb.Table(columns=columns)
    for row in rows:
        built.add_data(*[row.get(column) for column in columns])
    return built


def build() -> tuple[dict, dict, dict]:
    census = load("head-census.json")
    bytes_art = load("draft-step-bytes.json")
    declared = load("recall-declared.json")["pooled"]
    pinned = load("recall-pinned.json")["pooled"]
    curve = load("island-curve.json")
    probe = load("gate-probe.json")

    cd = declared["conditional_population"]
    cp = pinned["conditional_population"]
    recoverable_pp = 100.0 * (cp["p_exact_full"] - cd["p_exact_full"])
    delta_bytes = (bytes_art["pinned_bf16_head"]["total_bytes_per_draft_step"]
                   - bytes_art["declared_head"]["total_bytes_per_draft_step"])
    delta_mb = delta_bytes / 1e6
    gain_pct = recoverable_pp * PCT_PER_ACCEPTANCE_POINT
    cost_pct = delta_mb * 0.01793
    net_pct = gain_pct - cost_pct

    summary = {
        # R0/R1.A -- which head each harness loads.
        "e158_heads_differ_local_vs_ranked": True,
        "e158_local_head_provenance_sha256_e128_sessions":
            curve["arms"]["all"]["legs"]["beagle_a"][
                "head_provenance_sha256"],
        "e158_declared_head_sha256": census["declared_tree_digest"]["sha256"],
        "e158_pinned_head_sha256": census["pinned_tree_digest"]["sha256"],
        "e158_head_swap_works": True,
        "e158_head_has_own_vocab_projection_pinned": False,
        "e158_head_has_own_vocab_projection_declared": True,
        "e158_tree_digest_recipe_verified": True,

        # R1 -- the precision decomposition.
        "e158_p_shipped_declared": cd["p_shipped"],
        "e158_p_shipped_pinned_bf16": cp["p_shipped"],
        "e158_p_exact_compact_declared": cd["p_exact_compact"],
        "e158_p_exact_compact_pinned_bf16": cp["p_exact_compact"],
        "e158_p_exact_full_declared": cd["p_exact_full"],
        "e158_p_exact_full_pinned_bf16": cp["p_exact_full"],
        "e158_head_precision_recoverable_pp": recoverable_pp,
        "e158_irreducible_head_error_pp_declared":
            declared["three_bucket_split_conditional"]["bucket_c_pp"],
        "e158_irreducible_head_error_pp_pinned":
            pinned["three_bucket_split_conditional"]["bucket_c_pp"],
        "e158_bf16_bytes_per_draft_step_delta_MB": delta_mb,
        "e158_precision_gain_pct_PROVISIONAL": gain_pct,
        "e158_precision_cost_pct_PROVISIONAL": cost_pct,
        "e158_precision_net_pct_PROVISIONAL": net_pct,
        "e158_verdict": "close_head_precision",

        # Byte attribution, not provisional.
        "e158_bytes_per_draft_step_declared":
            bytes_art["declared_head"]["total_bytes_per_draft_step"],
        "e158_bytes_per_draft_step_pinned":
            bytes_art["pinned_bf16_head"]["total_bytes_per_draft_step"],

        # Gate witness, two-sided and non-vacuous.
        "e158_exact_path_is_gated_off_by_default": True,
        "e158_gate_probe_file_created": probe["audit_file_created"],
        "e158_gate_probe_rows_written": probe["audit_rows_written"],
        "e158_gate_probe_exit_code": probe["leg_exit_code"],
        "e158_reverted_worker_forbids_audit_symbols": True,

        # Rule 101 positive controls.
        "e158_masked_control_moved_declared":
            declared["counts"]["masked_control_moved"],
        "e158_masked_control_moved_pinned":
            pinned["counts"]["masked_control_moved"],
        "e158_ann_outside_compact_set_declared":
            declared["counts"]["ann_outside_compact_set"],
        "e158_ann_outside_compact_set_pinned":
            pinned["counts"]["ann_outside_compact_set"],
        "e158_index_miss_declared": declared["counts"]["index_miss"],
        "e158_index_miss_pinned": pinned["counts"]["index_miss"],
    }

    for arm in ARMS:
        entry = curve["arms"][arm]
        summary["e158_island_accept_%s" % arm] = entry["p_shipped_conditional"]
        summary["e158_island_rounds_%s" % arm] = entry["rounds_total"]
        summary["e158_island_loss_vs_all_pt_%s" % arm] = entry[
            "acceptance_loss_vs_all_pt"]
        summary["e158_island_bytes_delta_MB_%s" % arm] = entry[
            "bytes_per_draft_step_delta_vs_all_MB"]
        summary["e158_island_pct_vs_all_PROVISIONAL_%s" % arm] = entry[
            "published_pct_delta_vs_all_PROVISIONAL"]
        summary["e158_island_arm_witness_%s" % arm] = entry["legs"][
            "beagle_a"]["island_arm_witness"]
    summary["e158_island_accept_all_minus_none_pt"] = curve["arms"]["none"][
        "acceptance_loss_vs_all_pt"]
    summary["e158_island_two_se_pt"] = curve["arms"]["none"][
        "acceptance_loss_two_se_pt"]
    summary["e158_island_verdict"] = "drop_the_islands"

    flatten("census", census, summary)
    flatten("bucket_declared", declared["three_bucket_split_conditional"],
            summary)
    flatten("bucket_pinned", pinned["three_bucket_split_conditional"], summary)

    head_rows = []
    for arm_name, pooled in (("declared", declared), ("pinned", pinned)):
        cond = pooled["conditional_population"]
        head_rows.append({
            "head": arm_name,
            "slots_marginal": pooled["slots_total"],
            "slots_conditional": cond["slots"],
            "rounds": pooled["rounds"],
            "mean_offered_depth": pooled["mean_offered_depth"],
            "p_shipped_conditional": cond["p_shipped"],
            "p_exact_compact_conditional": cond["p_exact_compact"],
            "p_exact_full_conditional": cond["p_exact_full"],
            "p_shipped_marginal": pooled["e155_p_shipped"],
            "p_exact_full_marginal": pooled["e155_p_exact_full"],
            "bucket_a_pp": pooled["three_bucket_split_conditional"][
                "bucket_a_pp"],
            "bucket_b_pp": pooled["three_bucket_split_conditional"][
                "bucket_b_pp"],
            "bucket_c_pp": pooled["three_bucket_split_conditional"][
                "bucket_c_pp"],
            "index_miss": pooled["counts"]["index_miss"],
            "masked_control_moved": pooled["counts"]["masked_control_moved"],
        })

    island_rows = []
    for arm in ARMS:
        entry = curve["arms"][arm]
        for prompt, leg in entry["legs"].items():
            island_rows.append({
                "arm": arm,
                "prompt": prompt,
                "rounds": leg["rounds"],
                "tokens_per_round": leg["tokens_per_round"],
                "accepted_draft_rate": leg["accepted_draft_rate"],
                "effective_mean_draft_len": leg["effective_mean_draft_len"],
                "all_tokens_matched": leg["all_tokens_matched"],
                "residual_divergence_count": leg["residual_divergence_count"],
                "head_provenance_sha256": leg["head_provenance_sha256"],
                "island_arm_witness": leg["island_arm_witness"],
            })
        island_rows.append({
            "arm": arm,
            "prompt": "pooled",
            "rounds": entry["rounds_total"],
            "tokens_per_round": entry["tokens_per_round_pooled"],
            "p_shipped_conditional": entry["p_shipped_conditional"],
            "acceptance_loss_vs_all_pt": entry["acceptance_loss_vs_all_pt"],
            "bytes_delta_vs_all": entry["bytes_per_draft_step_delta_vs_all"],
            "published_pct_vs_all_PROVISIONAL": entry[
                "published_pct_delta_vs_all_PROVISIONAL"],
        })

    byte_rows = []
    for head in ("declared_head", "pinned_bf16_head"):
        for section in ("trunk_reads_per_draft_step",
                        "readout_reads_per_draft_step"):
            for name, value in bytes_art[head][section].items():
                byte_rows.append({
                    "head": head,
                    "section": section,
                    "tensor": name,
                    "bytes_per_draft_step": value,
                })

    tables = {
        "e158_head_comparison": table(head_rows),
        "e158_island_curve": table(island_rows),
        "e158_bytes_per_draft_step": table(byte_rows),
    }
    config = {
        "experiment": "E158",
        "pr": 158,
        "base_sha": BASE_SHA,
        "audit_build_commit": AUDIT_BUILD_COMMIT,
        "audit_worker_sha256": AUDIT_WORKER_SHA256,
        "reverted_worker_sha256": REVERTED_WORKER_SHA256,
        "instrument_patch": "research/e155-patches/recall-audit.patch",
        "host": HOST,
        "tokens_per_leg": 512,
        "offered_depth": 8,
        "prompts": ["beagle_a", "essays_montaigne", "benchfixture"],
        "question":
            "is the proposal head's remaining acceptance loss precision or "
            "capacity, and what are the four precision-island arms worth",
        "command":
            "research/e158_r1_session.sh {declared,pinned} beagle_a "
            "essays_montaigne benchfixture; "
            "research/e158_island_session.sh all,q,kv,none beagle_a "
            "essays_montaigne benchfixture",
        "published_percent_is_provisional_per_advisor_F4": True,
    }
    return summary, tables, config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    summary, tables, config = build()
    summary.update({
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local for rates, ranked for published percent",
        "timing_valid": False,
        "host": HOST,
        "base_sha": BASE_SHA,
    })

    if args.dry:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e158-head-precision-census", job_type="recall-audit",
        config=config)
    for name, value in tables.items():
        run.log({name: value})
    run.summary.update(summary)
    artifact = wandb.Artifact("e158-head-precision-census", type="recall-audit")
    for path in sorted(ART.glob("*.json")):
        artifact.add_file(str(path))
    for path in sorted(RAW.glob("**/*.jsonl")):
        artifact.add_file(str(path), name=str(path.relative_to(RAW)))
    run.log_artifact(artifact)
    print(run.url)
    print(run.id)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
