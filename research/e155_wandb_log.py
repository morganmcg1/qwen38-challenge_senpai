#!/usr/bin/env python3
"""Publish the E155 draft-head recall audit to W&B.

    usage: research/e155_wandb_log.py [--dry]

E155 asks how much of the shipped proposal head's per-slot miss rate an exact
readout could recover, and converts that headroom into published percent.

No run here is a gated measurement or a score. The audit-ON legs add a
full-vocabulary lm_head projection per proposal slot inside the measured
block, so `timing_valid` is false for them by construction, and no leg took
the real cool gate. Every such field is published verbatim as false.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e155-draft-head-recall-audit"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e155-artifacts")
# Raw per-slot rows stay out of Git and travel in the W&B artifact instead.
RAW = pathlib.Path(".mlxfast-private/e155")


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten("%s/%s" % (prefix, key) if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    built = wandb.Table(columns=columns)
    for row in rows:
        built.add_data(*[row.get(column) for column in columns])
    return built


def build() -> tuple[dict, dict, dict]:
    recall = load("recall.json")
    width = load("width-cost-fit.json")
    witness = load("witness.json")
    pooled = recall["pooled"]

    summary = {
        "e155_p_shipped": pooled["e155_p_shipped"],
        "e155_p_exact_compact": pooled["e155_p_exact_compact"],
        "e155_p_exact_full": pooled["e155_p_exact_full"],
        "e155_index_miss_rate": pooled["e155_index_miss_rate"],
        "e155_recoverable_index_pp": pooled["e155_recoverable_index_pp"],
        "e155_recoverable_vocab_pp": pooled["e155_recoverable_vocab_pp"],
        "e155_irreducible_head_error_pp":
            pooled["e155_irreducible_head_error_pp"],
        "e155_axis_verdict": recall["e155_axis_verdict"],
        "e155_warm_verdict": recall["e155_warm_verdict"],
        "e155_slots_audited": pooled["slots_total"],
        "e155_rounds_audited": pooled["rounds"],
        "e155_exact_path_is_gated_off_by_default":
            witness["e155_exact_path_is_gated_off_by_default"],
        "e155_permutation_positive_control_passed":
            pooled["permutation_control"]["magnitude_pp"] > 0.0,
        "e155_permutation_control_magnitude_pp":
            pooled["permutation_control"]["magnitude_pp"],
        "e155_masked_winner_control_magnitude_pp":
            pooled["masked_winner_control"]["magnitude_pp"],
        "e155_masked_winner_control_moved_fraction":
            pooled["masked_winner_control"]["moved_fraction"],
        "e155_ann_outside_compact_set":
            pooled["counts"]["ann_outside_compact_set"],
        "e155_growth_reclaimed_bytes": witness["e155_growth_reclaimed_bytes"],
        "e155_published_pct_per_acceptance_point_own_derivation":
            pooled["derived_exchange_rate"][
                "published_pct_per_acceptance_point"],
        "e155_within_prompt_width_cost_us_per_row":
            width["within_prompt_slope_mean_us"],
        "e155_within_prompt_width_cost_us_per_row_drift_controlled":
            width["within_prompt_slope_mean_us_drift_controlled"],
        "e155_cross_prompt_width_cost_us_per_row":
            width["cross_prompt_reference_slope_us_per_row"],
    }
    flatten("conversion", recall["conversion"], summary)
    flatten("witness", witness, summary)
    flatten("acceptance_points", recall["e155_recoverable_acceptance_points"],
            summary)
    flatten("bucket_b_price", recall["bucket_b_untrim_price"], summary)
    flatten("bucket_conditional", pooled["three_bucket_split_conditional"],
            summary)
    flatten("bucket_marginal", pooled["three_bucket_split_marginal"], summary)
    flatten("head_margin", pooled["head_margin_when_head_disagrees"], summary)
    summary["e155_vocab_axis_verdict"] = recall["e155_vocab_axis_verdict"]
    summary["e155_recoverable_acceptance_points_total"] = recall[
        "e155_recoverable_acceptance_points"]["total_recoverable_points"]

    prompt_rows = []
    for entry in recall["per_prompt"] + [pooled]:
        row = {
            "prompt": entry["prompt"],
            "slots": entry["slots_total"],
            "rounds": entry["rounds"],
            "mean_offered_depth": entry["mean_offered_depth"],
            "p_shipped": entry["e155_p_shipped"],
            "p_exact_compact": entry["e155_p_exact_compact"],
            "p_exact_full": entry["e155_p_exact_full"],
            "index_miss_rate": entry["e155_index_miss_rate"],
            "recoverable_index_pp": entry["e155_recoverable_index_pp"],
            "recoverable_vocab_pp": entry["e155_recoverable_vocab_pp"],
            "irreducible_head_error_pp":
                entry["e155_irreducible_head_error_pp"],
            "shipped_hits": entry["counts"]["shipped_hits"],
            "exact_compact_hits": entry["counts"]["exact_compact_hits"],
            "exact_full_hits": entry["counts"]["exact_full_hits"],
            "index_miss": entry["counts"]["index_miss"],
            "index_miss_exact_ties": entry["counts"]["index_miss_exact_ties"],
            "distinct_target_tokens": entry["distinct_target_tokens"],
            "distinct_proposed_tokens": entry["distinct_proposed_tokens"],
            "most_common_target_share": entry["most_common_target_share"],
        }
        for population in ("conditional", "marginal"):
            split = entry["three_bucket_split_%s" % population]
            for key in ("bucket_a_selection_defect",
                        "bucket_b_candidate_set_trim",
                        "bucket_c_irreducible_head_disagreement",
                        "lucky_shipped_win_exact_miss",
                        "bucket_a_net_of_lucky_pp", "bucket_b_pp",
                        "bucket_c_pp"):
                row["%s_%s" % (population, key)] = split[key]
        prompt_rows.append(row)

    slot_rows = []
    for entry in recall["per_prompt"] + [pooled]:
        for slot, values in entry["conditional_slot_table"].items():
            slot_rows.append({
                "prompt": entry["prompt"],
                "population": "conditional",
                "slot": int(slot),
                **values,
            })
        for slot, values in entry["marginal_slot_table"].items():
            slot_rows.append({
                "prompt": entry["prompt"],
                "population": "marginal",
                "slot": int(slot),
                **values,
            })

    width_rows = []
    for fit in width["within_prompt_fits"]:
        width_rows.append({
            "prompt": fit["prompt"],
            "rounds_used": fit["rounds_used"],
            "mean_round_us": fit["mean_round_us"],
            "slope_b_us_per_row": fit["fit_round_us_vs_M"][
                "slope_b_us_per_row"],
            "slope_stderr_us": fit["fit_round_us_vs_M"]["slope_stderr_us"],
            "intercept_a_us": fit["fit_round_us_vs_M"]["intercept_a_us"],
            "r_squared": fit["fit_round_us_vs_M"]["r_squared"],
            "slope_b_us_per_row_drift_controlled":
                fit["fit_round_us_vs_M_and_round_index"][
                    "slope_b_us_per_row"],
            "drift_c_us_per_round":
                fit["fit_round_us_vs_M_and_round_index"][
                    "drift_c_us_per_round"],
        })

    tables = {
        "e155_prompt_summary": table(
            sorted({k for r in prompt_rows for k in r}), prompt_rows),
        "e155_per_slot_table": table(
            sorted({k for r in slot_rows for k in r}), slot_rows),
        "e155_within_prompt_width_cost_fit": table(
            sorted({k for r in width_rows for k in r}), width_rows),
    }
    config = {
        "experiment": "E155",
        "pr": 155,
        "base_sha": witness["base_sha"],
        # The audit legs ran on the instrumented build. The submitted head
        # reverts the instrument, so it cannot reproduce the rows by itself;
        # research/e155-patches/recall-audit.patch restores it.
        "audit_build_commit": "3d52eac6",
        "audit_worker_sha256": witness["legs"][0]["worker_sha256_on"],
        "instrument_patch": "research/e155-patches/recall-audit.patch",
        "host": HOST,
        "question":
            "how much of the shipped proposal head's per-slot miss rate would "
            "an exact compact readout, and then an exact full-vocabulary "
            "readout, recover",
        "command": "research/e155_audit_session.sh beagle_a "
                   "essays_montaigne benchfixture",
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
        "harness": "local",
        "timing_valid": False,
        "host": HOST,
        "base_sha": config["base_sha"],
    })

    if args.dry:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e155-draft-head-recall-audit", job_type="recall-audit",
        config=config)
    for name, value in tables.items():
        run.log({name: value})
    run.summary.update(summary)
    artifact = wandb.Artifact("e155-recall-audit", type="recall-audit")
    for path in sorted(ART.glob("*.json")):
        artifact.add_file(str(path))
    for path in sorted(RAW.glob("*.jsonl")):
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    print(run.url)
    print(run.id)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
