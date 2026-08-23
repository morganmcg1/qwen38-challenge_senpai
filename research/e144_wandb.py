"""Publish the E144 evidence to Weights & Biases.

Reads the JSON artifacts the rungs already wrote and logs them as one run:
summary scalars for every `e144_*` metric in the assignment's evidence contract,
a per-tensor table for the reconstruction sweep, and a second table for the
achievability bounds. Missing artifacts are skipped and named in the summary, so
a partial campaign still publishes what it has.
"""

import argparse
import json
import os

import wandb

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="qwen38-mlx-challenge-senpai")
    parser.add_argument("--entity", default="wandb-applied-ai-team")
    parser.add_argument("--name", default="e144-data-free-head-requantization")
    parser.add_argument("--commit", required=True, help="candidate commit under test")
    parser.add_argument("--base-sha", required=True)
    arguments = parser.parse_args()

    r0 = load("e144-r0.json")
    ra = load("e144-ra.json")
    rb = load("e144-rb.json")
    rb2 = load("e144-rb2.json")
    bounds = load("e144-rb-bounds.json")
    price = load("e144-price.json")
    sweep = load("e144-group-sweep.json")
    f219 = load("e144-f219.json")
    missing = [
        label
        for label, payload in [
            ("e144-r0.json", r0),
            ("e144-ra.json", ra),
            ("e144-rb.json", rb),
            ("e144-rb2.json", rb2),
            ("e144-rb-bounds.json", bounds),
            ("e144-price.json", price),
            ("e144-group-sweep.json", sweep),
            ("e144-f219.json", f219),
        ]
        if payload is None
    ]

    run = wandb.init(
        entity=arguments.entity,
        project=arguments.project,
        name=arguments.name,
        job_type="analysis",
        config={
            "experiment": "e144",
            "harness": "local",
            "hypothesis": (
                "a better data-free quantizer of the master-bf16 MTP head recovers the "
                "0.82 pt of acceptance the incumbent lost to round-to-nearest affine-4 g64"
            ),
            "data_free": True,
            "gpu_used": False,
            "commit": arguments.commit,
            "base_sha": arguments.base_sha,
            "bits": 4,
            "group_size": 64,
            "master_head": "EigenLabs/Qwen3.8-27B-MTP-bf16@26a328e0",
            "declared_head": "amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827",
            "missing_artifacts": missing,
        },
    )

    summary = {}
    if ra:
        summary["e144_declared_reproduction_exact"] = ra["summary"][
            "e144_declared_reproduction_exact"
        ]
        summary["e144_tensors_with_master_ancestor"] = ra["summary"].get(
            "tensors_with_master_ancestor"
        )
        summary["e144_tensors_without_master_ancestor"] = ra["summary"].get(
            "tensors_without_master_ancestor"
        )
    if rb:
        summary["e144_rel_l2_improvement_factor_pooled"] = rb["summary"][
            "e144_rel_l2_improvement_factor_pooled"
        ]
        summary["e144_rel_l2_improvement_factor_worst_tensor"] = rb["summary"][
            "e144_rel_l2_improvement_factor_worst_tensor"
        ]
        summary["e144_rb_stop_rule_passed"] = rb["summary"]["stop_rule_passed"]
        for label, value in rb["pooled_rel_l2"].items():
            summary[f"e144_pooled_rel_l2_{label}"] = value
        if "emitted" in rb:
            summary["e144_head_bytes_delta"] = rb["emitted"]["e144_head_bytes_delta"]
            summary["e144_head_bytes"] = rb["emitted"]["bytes"]

        table = wandb.Table(
            columns=["tensor", "rows", "columns", "declared", "rtn", "clip", "als", "best", "factor"]
        )
        for name, entry in rb["tensors"].items():
            table.add_data(
                name,
                entry["shape"][0],
                entry["shape"][1],
                entry["rel_l2"]["declared"],
                entry["rel_l2"]["rtn"],
                entry["rel_l2"]["clip"],
                entry["rel_l2"]["als"],
                entry["rel_l2"]["best"],
                entry["improvement_factor_vs_declared"],
            )
        run.log({"e144_rel_l2_by_tensor": table})

    if rb2:
        summary["e144_head_payload_bytes_delta"] = rb2["e144_head_payload_bytes_delta"]
        summary["e144_head_bytes_delta"] = rb2["e144_head_bytes_delta"]
        summary["e144_readout_tensors_byte_identical"] = rb2["readout_tensors_byte_identical"]
        summary["e144_changed_tensor_count"] = len(rb2["changed_tensors"])

    if sweep:
        table = wandb.Table(
            columns=["group_size", "rel_l2", "factor_vs_g64", "trunk_metadata_bytes_delta"]
        )
        for size, entry in sorted(sweep["pooled"].items(), key=lambda item: int(item[0])):
            table.add_data(
                int(size),
                entry["rel_l2"],
                entry["rel_l2_factor_vs_g64"],
                entry["trunk_metadata_bytes_delta_vs_g64"],
            )
            summary[f"e144_group_sweep_rel_l2_g{size}"] = entry["rel_l2"]
        run.log({"e144_group_size_sweep": table})

    if bounds:
        summary["e144_minmax_reference_factor"] = bounds["pooled"]["minmax_rel_l2_factor"]
        summary["e144_affine_ceiling_factor"] = bounds["pooled"]["affine_rel_l2_factor"]
        summary["e144_codebook_ceiling_factor"] = bounds["pooled"]["codebook_rel_l2_factor"]
        table = wandb.Table(
            columns=["tensor", "groups", "minmax_factor", "affine_factor", "codebook_factor"]
        )
        for name, entry in bounds["tensors"].items():
            table.add_data(
                name,
                entry["groups"],
                entry["minmax_rel_l2_factor"],
                entry["affine_rel_l2_factor"],
                entry["codebook_rel_l2_factor"],
            )
        run.log({"e144_ceiling_by_tensor": table})

    if price:
        verdict = price["verdict"]
        summary["e144_inferred_recovery_pt_at_measured_low"] = verdict[
            "recovery_pt_at_measured_range"
        ][0]
        summary["e144_inferred_recovery_pt_at_measured_high"] = verdict[
            "recovery_pt_at_measured_range"
        ][1]
        summary["e144_inferred_recovery_pt_at_ceiling_low"] = verdict[
            "recovery_pt_at_ceiling_range"
        ][0]
        summary["e144_inferred_recovery_pt_at_ceiling_high"] = verdict[
            "recovery_pt_at_ceiling_range"
        ][1]
        summary["e144_required_factor_min"] = verdict["min_required_factor"]
        summary["e144_required_factor_max"] = verdict["max_required_factor"]
        summary["e144_ceiling_reaches_target"] = verdict["ceiling_reaches_target"]
        table = wandb.Table(
            columns=["model", "k", "recovery_at_measured_pt", "recovery_at_ceiling_pt", "factor_for_0.30pt"]
        )
        for row in price["rows"]:
            table.add_data(
                row["model"],
                row["k"],
                row["recovery_pt_at_measured"],
                row["recovery_pt_at_affine_ceiling"],
                row["factor_required_for_0.30pt"],
            )
        run.log({"e144_inferred_acceptance_price": table})

    if price and "group_lever" in price:
        lever = price["group_lever"]
        summary["e144_median_pct_per_acceptance_pt"] = lever["median_pct_per_acceptance_pt"]
        table = wandb.Table(
            columns=[
                "group_size",
                "rel_l2_factor_vs_g64",
                "head_bytes_pct",
                "score_pct_from_bytes",
                "score_pct_from_acceptance_low",
                "score_pct_from_acceptance_high",
                "net_score_pct_low",
                "net_score_pct_high",
            ]
        )
        for row in lever["rows"]:
            table.add_data(
                row["group_size"],
                row["rel_l2_factor_vs_g64"],
                row["head_bytes_pct"],
                row["score_pct_from_bytes"],
                row["score_pct_from_acceptance"][0],
                row["score_pct_from_acceptance"][1],
                row["net_score_pct_range"][0],
                row["net_score_pct_range"][1],
            )
            if row["group_size"] != 64:
                summary[f"e144_group_lever_net_score_pct_low_g{row['group_size']}"] = row[
                    "net_score_pct_range"
                ][0]
                summary[f"e144_group_lever_net_score_pct_high_g{row['group_size']}"] = row[
                    "net_score_pct_range"
                ][1]
        run.log({"e144_group_size_lever_value": table})

    if f219:
        arms = f219["arms"]
        chain = wandb.Table(
            columns=[
                "delta_pt",
                "regime",
                "prompt",
                "draft_len",
                "e144_new_draft_len",
                "round_cost_us",
                "e144_new_round_cost_us",
                "round_cost_pct",
                "seconds_per_token",
                "e144_new_seconds_per_token",
                "candidate_leg_gain_pct",
                "e144_new_raw_ratio",
            ]
        )
        value = wandb.Table(
            columns=[
                "delta_pt",
                "regime",
                "e144_expected_median_pct",
                "median_pct_sd",
                "e144_median_p05_pct",
                "expected_published_median",
                "e144_p_beat_bar",
                "e144_upper_slot_owner_before",
                "e144_upper_slot_owner_after",
            ]
        )
        for arm in arms:
            delta = arm["inferred_acceptance_delta_pt"]
            for name in sorted(arm["prompts"]):
                entry = arm["prompts"][name]
                chain.add_data(
                    delta,
                    arm["regime"],
                    name,
                    entry["draft_len"],
                    entry["e144_new_draft_len"],
                    entry["round_cost_us"],
                    entry["e144_new_round_cost_us"],
                    entry["round_cost_pct"],
                    entry["seconds_per_token"],
                    entry["e144_new_seconds_per_token"],
                    entry["candidate_leg_gain_pct"],
                    arm["e144_new_raw_ratio"][name],
                )
            block = arm["value"]
            value.add_data(
                delta,
                arm["regime"],
                block["e144_expected_median_pct"],
                block["median_pct_sd"],
                block["e144_median_p05_pct"],
                block["expected_published_median"],
                block["e144_p_beat_bar"],
                json.dumps(block["e144_upper_slot_owner_before"]),
                json.dumps(block["e144_upper_slot_owner_after"]),
            )
        run.log({"e144_f219_chain": chain, "e144_f219_value": value})

        reachable = [
            arm
            for arm in arms
            if price
            and arm["inferred_acceptance_delta_pt"]
            <= price["verdict"]["recovery_pt_at_ceiling_range"][1] + 1e-9
        ]
        if reachable:
            best = max(reachable, key=lambda arm: arm["value"]["e144_p_beat_bar"])
            worst = min(reachable, key=lambda arm: arm["value"]["e144_p_beat_bar"])
            for label, arm in (("", best), ("_low", worst)):
                block = arm["value"]
                summary[f"e144_expected_median_pct{label}"] = block["e144_expected_median_pct"]
                summary[f"e144_median_p05_pct{label}"] = block["e144_median_p05_pct"]
                summary[f"e144_p_beat_bar{label}"] = block["e144_p_beat_bar"]
                summary[f"e144_expected_published_median{label}"] = block[
                    "expected_published_median"
                ]
                summary[f"e144_priced_regime{label}"] = arm["regime"]
                summary[f"e144_priced_delta_pt{label}"] = arm["inferred_acceptance_delta_pt"]
            owner_before = best["value"]["e144_upper_slot_owner_before"]
            owner_after = best["value"]["e144_upper_slot_owner_after"]
            summary["e144_upper_slot_owner_before"] = max(owner_before, key=owner_before.get)
            summary["e144_upper_slot_owner_after"] = max(owner_after, key=owner_after.get)

    # Rungs the stop rule prevented. Name them so the record is unambiguous.
    for metric in (
        "e144_recall_at_32",
        "e144_acceptance_delta_pp",
        "e144_mcnemar_p",
        "e144_width6plus_mass_delta",
        "e144_beagle_cost_pct",
        "e144_zero_weight_gain_share",
    ):
        summary[f"{metric}_status"] = "not measured: R-C skipped under stop rule 3"

    if r0:
        measured = r0["q3_archive_or_diff_step_breaks"]["measured"]
        summary["e144_in_branch_archive_bytes"] = measured["archive_bytes_with_head"]
        summary["e144_in_branch_archive_cap_bytes"] = measured["archive_cap"]
        summary["e144_in_branch_archive_overage_factor"] = measured["archive_overage_factor"]

    run.summary.update({key: value for key, value in summary.items() if value is not None})
    print(json.dumps({"run_id": run.id, "url": run.url}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
