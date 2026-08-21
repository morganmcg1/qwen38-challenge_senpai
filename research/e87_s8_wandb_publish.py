"""Publish the two recorded section-8 measurement artefacts as W&B runs.

The advisor requires a run link next to every section-8 claim. Both sessions ran
through run_job with no W&B run attached, so this republishes the recorded JSON
rather than re-measuring. Nothing here produces a new number.
"""

import json
import pathlib
import sys

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
ROOT = pathlib.Path(__file__).resolve().parent

COMMON = {
    "experiment": "E87",
    "section": 8,
    "student": "qwen-thorfinn",
    "pr": 89,
    "assignment_id": "qwen38-r1-e87-coarse-draft-shortlist-traffic",
    "revision_id": "r2",
    "harness": "local",
    "host": "ip-10-231-2-95.ec2.internal",
    "chip": "Apple M4 Pro Mac16,11",
    "gpu_cores": 20,
    "memory_gib": 48,
    "kernel": "qwen_mtp_probe_sort",
    "gate_env": "MLX_E87_PROBE_SORT",
    "cool_gate_passed_real_gate": False,
    "gate_qualified_for_timing": False,
    "republished_from_recorded_json": True,
}


def publish_chain():
    src = ROOT / "e87-s8-isolated-chain.json"
    d = json.loads(src.read_text())
    ident = d["identity"]
    result = d["result"]
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e87-s8-isolated-chain-census",
        job_type="dispatch-census",
        group="e87-section8",
        tags=["e87", "section8", "probe-sort", "dispatch-census", "upper-bound"],
        config={
            **COMMON,
            "measurement": "isolated chain cost, one dispatch per command buffer",
            "label": d["label"],
            "buffer_limit_mb": 1,
            "tokens": ident["tokens"],
            "forced_drafts": ident["forced_drafts"],
            "mode": ident["mode"],
            "width_phase": ident["width_phase"],
            "steady_snapshots": ident["steady_snapshots"],
            "worker_sha256_instrumented": ident["worker_sha256_instrumented"],
            "worker_sha256_clean_candidate": ident["worker_sha256_clean_candidate"],
            "arm_selector": ident["arm_selector"],
        },
    )
    base, kern = d["legs"]["base"], d["legs"]["kernel"]
    flat = {
        "base/phase_us_per_draft": base["phase_us_per_draft"],
        "base/phase_dispatches_per_draft": base["phase_dispatches_per_draft"],
        "kernel/phase_us_per_draft": kern["phase_us_per_draft"],
        "kernel/phase_dispatches_per_draft": kern["phase_dispatches_per_draft"],
        "session_drift/kernel_over_base": d["session_drift"]["kernel_over_base"],
    }
    for row in d["roster_delta"]["removed_from_base"]:
        flat[f"removed/{row['role']}/us_per_draft"] = row["us_per_draft"]
    for row in d["roster_delta"]["added_in_kernel"]:
        flat[f"added/{row['role']}/us_per_draft"] = row["us_per_draft"]
    for k, v in result.items():
        flat[f"result/{k}"] = v
    run.log(flat)
    run.summary.update(flat)
    art = wandb.Artifact("e87-s8-isolated-chain", type="measurement")
    art.add_file(str(src))
    run.log_artifact(art)
    rid, url = run.id, run.url
    run.finish()
    return rid, url


def publish_abba():
    src = ROOT / "e87-s8-round-abba.json"
    d = json.loads(src.read_text())
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e87-s8-round-abba",
        job_type="abba-timing",
        group="e87-section8",
        tags=["e87", "section8", "probe-sort", "abba", "round-level"],
        config={
            **COMMON,
            "measurement": "round-level ABBA, 7 legs B K B K B K B, one session, one binary",
            "decode_tokens": 512,
            "legs": len(d["order"]),
            "order": d["order"],
            "all_legs_exact": d["all_legs_exact"],
            "distinct_worker_digests": d["distinct_worker_digests"],
            "distinct_draft_len": d["distinct_draft_len"],
            "primary_metric": "mtp_seconds_per_token",
            "primary_direction": "minimize",
        },
    )
    spt, spd = d["mtp_seconds_per_token"], d["mtp_decode_speedup"]
    for i, leg in enumerate(d["order"]):
        arm, n = leg.split("-")
        j = int(n) - 1
        run.log(
            {
                "leg_index": i,
                "arm_is_kernel": int(arm == "kernel"),
                "leg/mtp_seconds_per_token": spt[arm][j],
                "leg/mtp_decode_speedup": spd[arm][j],
            },
            step=i,
        )
    flat = {}
    for name, blk in (("mtp_seconds_per_token", spt), ("mtp_decode_speedup", spd)):
        for k in (
            "drift_slope_per_leg",
            "session_null_sd",
            "session_null_pct",
            "delta",
            "delta_pct",
            "contrast_sd",
            "inside_session_null",
        ):
            flat[f"{name}/{k}"] = blk[k]
        for i, c in enumerate(blk["contrasts"]):
            flat[f"{name}/contrast_{i + 1}"] = c
    run.summary.update(flat)
    art = wandb.Artifact("e87-s8-round-abba", type="measurement")
    art.add_file(str(src))
    run.log_artifact(art)
    rid, url = run.id, run.url
    run.finish()
    return rid, url


if __name__ == "__main__":
    out = {"isolated_chain": publish_chain(), "round_abba": publish_abba()}
    json.dump(out, sys.stdout, indent=1)
    print()
