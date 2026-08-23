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
    bounds = load("e144-rb-bounds.json")
    price = load("e144-price.json")
    missing = [
        label
        for label, payload in [
            ("e144-r0.json", r0),
            ("e144-ra.json", ra),
            ("e144-rb.json", rb),
            ("e144-rb-bounds.json", bounds),
            ("e144-price.json", price),
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

    if bounds:
        summary["e144_affine_ceiling_factor"] = bounds["pooled"]["affine_rel_l2_factor"]
        summary["e144_codebook_ceiling_factor"] = bounds["pooled"]["codebook_rel_l2_factor"]
        table = wandb.Table(columns=["tensor", "groups", "affine_factor", "codebook_factor"])
        for name, entry in bounds["tensors"].items():
            table.add_data(
                name, entry["groups"], entry["affine_rel_l2_factor"], entry["codebook_rel_l2_factor"]
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
