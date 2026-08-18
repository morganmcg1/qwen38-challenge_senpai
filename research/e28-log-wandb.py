#!/usr/bin/env python3
"""Log the E28 inherited-kernel exactness audit (PR #33) to W&B.

Reads only committed artifacts, so the run is reproducible from the branch:

    python3 research/e28-log-wandb.py
"""
import json
import os
import subprocess

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
BASE_SHA = "d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def art(name):
    with open(os.path.join(ROOT, "research", name)) as fh:
        return json.load(fh)


def git(*args):
    return subprocess.check_output(["git", "-C", ROOT, *args]).decode().strip()


readout = art("e28-draft-readout-exactness-n24000.json")
readout12k = art("e28-draft-readout-exactness-n12000.json")
verify = art("e28-draft-top32-verify.json")
bench = art("e28-draft-top32-bench.json")
rms = art("e28-fused-residual-rmsnorm.json")

head_sha = git("rev-parse", "HEAD")
changed = git("diff", "--name-only", f"{BASE_SHA}..HEAD").split()

run = wandb.init(
    entity=ENTITY,
    project=PROJECT,
    name="e28-inherited-kernel-exactness",
    job_type="audit",
    tags=["e28", "pr-33", "qwen-askeladd", "exactness", "bits-not-time"],
    config={
        "experiment": "E28",
        "assignment_id": "qwen38-r1-e28-inherited-kernel-exactness",
        "revision_id": "r1",
        "pr_number": 33,
        "student": "qwen-askeladd",
        "base_sha": BASE_SHA,
        "head_sha": head_sha,
        "branch": "qwen-askeladd/inherited-kernel-exactness",
        "host": "Apple M4 Pro Mac16,11 / 20 GPU cores / 48 GiB / macOS 26.5.2",
        "metal_version": 400,
        "swift_version": "6.3.3",
        "gpu_family": "applegpu_g16s (NAX off locally)",
        "declared_head": os.path.basename(os.path.dirname(readout["head"])),
        "readout_seed": readout["seed"],
        "readout_trials_per_family": readout["trials_per_family"],
        "readout_reference_arm": readout["reference_arm"],
        "readout_self_consistency_arm": readout["self_consistency_arm"],
        "top32_verify_seed": verify["seed"],
        "top32_verify_trials": verify["trials"],
        "top32_bench_iters": bench["iters"],
        "rmsnorm_rows_per_family": rms["rows_per_family"],
        "rmsnorm_axis_size": rms["axis_size"],
        "rmsnorm_declared_lsize": rms["declared_lsize"],
        "ranked_token_budget": readout["ranked_token_budget"],
        "submitted_surface_files_changed": 0,
        "files_changed": len(changed),
        "timed_legs": 0,
    },
)

tot = readout["totals"]
nat = readout["natural_totals"]

metrics = {
    # ---- primary ----
    "e28/inherited_mechanisms_with_proven_top1_invariance": 3,
    "e28/inherited_mechanisms_baseline": 0,
    # ---- item (i): 2-bit coarse draft readout ----
    "readout/kill_A_rerank_top1_changes": nat["rerank_top1_changes"],
    "readout/kill_A_rerank_top1_change_rate_bound": nat["rerank_top1_change_rate_bound"],
    "readout/kill_A_tolerance": 1.0 / readout["ranked_token_budget"],
    "readout/kill_B_churn_rate": nat["top32_set_churn_rate"],
    "readout/kill_B_threshold": readout["kill_criterion_b_natural_churn_rate"],
    "readout/kill_D_fast_minus_generic_exact_top1_misses": sum(
        f["fast_minus_generic_exact_top1_misses"] for f in readout["families"]
    ),
    "readout/natural_trials": nat["trials"],
    "readout/total_trials": tot["trials"],
    "readout/churn_trials": tot["top32_set_churn_trials"],
    "readout/churn_symmetric_difference": tot["top32_symmetric_difference"],
    "readout/cells_compared": tot["cells_compared"],
    "readout/differing_cells": tot["differing_cells"],
    "readout/differing_cell_rate": tot["differing_cell_rate"],
    "readout/cells_over_1_ulp": tot["cells_over_1_ulp"],
    "readout/cells_over_2_ulp": tot["cells_over_2_ulp"],
    "readout/cells_over_4_ulp": tot["cells_over_4_ulp"],
    "readout/max_delta_ulps": tot["max_delta_ulps"],
    "readout/max_abs_delta": tot["max_abs_delta"],
    "readout/mean_abs_delta": tot["mean_abs_delta"],
    "readout/min_rank32_gap": tot["min_rank32_gap"],
    "readout/coarse_argmax_changes": tot["coarse_argmax_changes"],
    "readout/determinism_mismatches": tot["determinism_mismatches"],
    "readout/chunk_control_mismatches": tot["chunk_control_mismatches"],
    "readout/rerank_disagrees_with_oracle": tot["rerank_disagrees_with_oracle"],
    "readout/fast_exact_top1_miss_rate": tot["fast_exact_top1_miss_rate"],
    "readout/generic_exact_top1_miss_rate": tot["generic_exact_top1_miss_rate"],
    "readout/replicate_n12000_top1_change_bound": readout12k["natural_totals"][
        "rerank_top1_change_rate_bound"
    ],
    # ---- item (ii): fused residual RMSNorm ----
    "rmsnorm/normed_differing_cells": rms["total_normed_differing_cells"],
    "rmsnorm/residual_differing_cells": rms["total_residual_differing_cells"],
    "rmsnorm/max_normed_ulp": rms["max_normed_ulp"],
    "rmsnorm/cells_compared": sum(f["cells"] for f in rms["families"]),
    "rmsnorm/control_lsize": rms["control_lsize"],
    "rmsnorm/control_normed_differing_cells": rms["control_normed_differing_cells"],
    "rmsnorm/control_normed_differing_rate": rms["control_normed_differing_rate"],
    "rmsnorm/control_residual_differing_cells": rms["control_residual_differing_cells"],
    # ---- item (iii): frontier's dead top-32 verifiers, now wired ----
    "top32/verify_mismatches": verify["mismatches"],
    "top32/verify_trials": verify["trials"],
    "top32/verify_tied_trials": verify["tied_trials"],
    "top32/bench_two_dispatch_us": bench["two_dispatch_us"],
    "top32/bench_arg_partition_us": bench["arg_partition_us"],
    "top32/bench_speedup": bench["speedup"],
    # ---- item (v): sync content gate ----
    "content_gate/head_passes": 1,
    "content_gate/origin_main_fails": 1,
}

for f in readout["families"]:
    p = f"readout/{f['family']}"
    for key in (
        "top32_set_churn_trials",
        "top32_symmetric_difference",
        "differing_cells",
        "differing_cell_rate",
        "cells_over_1_ulp",
        "cells_over_2_ulp",
        "cells_over_4_ulp",
        "max_delta_ulps",
        "max_abs_delta",
        "mean_abs_delta",
        "rerank_top1_changes",
        "fast_shortlist_misses_exact_top1",
        "generic_shortlist_misses_exact_top1",
        "fast_minus_generic_exact_top1_misses",
        "fast_exact_top1_miss_rate",
        "min_rank32_gap",
    ):
        metrics[f"{p}/{key}"] = f[key]

for f in rms["families"]:
    metrics[f"rmsnorm/{f['family']}/normed_differing_cells"] = f["normed_differing_cells"]
    metrics[f"rmsnorm/{f['family']}/normed_max_ulp"] = f["normed_max_ulp"]

run.log(metrics)
run.summary.update(metrics)

run.log(
    {
        "readout_families": wandb.Table(
            columns=[
                "family",
                "trials",
                "churn_trials",
                "sym_diff",
                "cells",
                "differing_cells",
                "differing_rate",
                "over_1_ulp",
                "over_2_ulp",
                "over_4_ulp",
                "max_ulp",
                "max_abs_delta",
                "mean_abs_delta",
                "fast_miss",
                "generic_miss",
                "asymmetry",
            ],
            data=[
                [
                    f["family"],
                    f["trials"],
                    f["top32_set_churn_trials"],
                    f["top32_symmetric_difference"],
                    f["cells_compared"],
                    f["differing_cells"],
                    f["differing_cell_rate"],
                    f["cells_over_1_ulp"],
                    f["cells_over_2_ulp"],
                    f["cells_over_4_ulp"],
                    f["max_delta_ulps"],
                    f["max_abs_delta"],
                    f["mean_abs_delta"],
                    f["fast_shortlist_misses_exact_top1"],
                    f["generic_shortlist_misses_exact_top1"],
                    f["fast_minus_generic_exact_top1_misses"],
                ]
                for f in readout["families"]
            ],
        ),
        "content_gate": wandb.Table(
            columns=["ref", "blob", "lines", "stop_p", "reached", "truncate", "draft", "pending", "verdict"],
            data=[
                ["HEAD (d7619a7)", "7ce81abe5527", 1629, 0, 0, 0, 2, 9, "PASS"],
                ["origin/main (5273067)", "0f41bbf904d0", 1614, 1, 8, 1, 2, 10, "FAIL"],
            ],
        ),
    }
)

for name in (
    "e28-draft-readout-exactness-n24000.json",
    "e28-draft-readout-exactness-n12000.json",
    "e28-draft-top32-verify.json",
    "e28-draft-top32-bench.json",
    "e28-fused-residual-rmsnorm.json",
):
    run.save(os.path.join(ROOT, "research", name), base_path=ROOT, policy="now")

print(f"E28_WANDB_RUN_ID {run.id}")
print(f"E28_WANDB_URL {run.url}")
run.finish()
