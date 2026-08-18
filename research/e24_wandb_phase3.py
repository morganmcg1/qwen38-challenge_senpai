#!/usr/bin/env python3
"""E24: publish the Phase 3 paired BASE/MEMO prose measurement to W&B.

One run per arm carrying the per-prompt series, plus one analysis run carrying
the headline effect on BOTH legs, the Phase 1 realization factor, the
correctness ledger and the cool-gate honesty fields.

usage:
  research/e24_wandb_phase3.py [--json research/results/e24-phase3.json] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
GROUP = "qwen38-r1-e24-constant-scalar-dispatch-tax"
BASE_SHA = "55c727e959e26cf24333d3e8c0896f7d97ab1224"
PR_NUMBER = 28
CAST_US = 9.711e-6
SITES_PER_FORWARD = 96

ARM_ROLE = {
    "BASE": "campaign-base-inline-constants",
    "MEMO": "candidate-cached-constants",
}


def sh(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def host_config() -> dict:
    return {
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "host_os": platform.mac_ver()[0],
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_base_sha": BASE_SHA,
        "git_dirty_files": len(sh("git", "status", "--porcelain").splitlines()),
        "pr_number": PR_NUMBER,
        "assignment": GROUP,
        "decode_tokens": 512,
        "phase": "phase3-paired-prose-abba",
        # Stated in config so no reader has to infer it from the plots.
        "headline_instrument": "absolute prefill-subtracted decode wall seconds",
        "why_not_ratio": (
            "E24 edits GDN constants the depth-0 serial leg also executes, so "
            "the serial-to-MTP ratio partly cancels the effect"),
        "ranked_host": "M5 (this host is M4 Pro: directional only)",
        "metallib_stale_both_arms": True,
        # This host idles above COOL_GATE_TEMP_C=40, so the wrapper gate is
        # unsatisfiable; timing ran under the E15-authorized
        # MLXFAST_LOCAL_COOL_GATE=0 policy (ABBA, entry/exit temps, spread
        # reported, flags carried verbatim).
        "cool_gate_policy": "settle_then_gate_off_e15_authorized",
        "cool_gate_real": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/results/e24-phase3.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report = json.loads(Path(args.json).read_text())
    rows = report["rows"]
    arms = report["arms"]
    gates = report.get("gates", {})

    mtp_e = [r["mtp_effect_pct"] for r in rows]
    ser_e = [r["ser_effect_pct"] for r in rows]

    if args.dry_run:
        print(json.dumps({"prompts": len(rows),
                          "mtp_median_pct": statistics.median(mtp_e),
                          "ser_median_pct": statistics.median(ser_e)}, indent=2))
        return 0

    import wandb

    base_config = host_config()
    for arm in ("BASE", "MEMO"):
        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP,
            job_type="phase3-arm", name=f"e24-phase3-{arm}",
            config={**base_config, "arm": arm, "arm_role": ARM_ROLE[arm]},
            reinit=True,
        )
        for step, r in enumerate(rows):
            prompt = r["prompt"]
            v = arms[prompt][arm]
            g = gates.get(f"{prompt}-{arm}", {})
            run.log({
                "prompt_index": step,
                "mtp_true_decode_s": r[f"mtp_{arm.lower()}_s"],
                "serial_true_decode_s": r[f"ser_{arm.lower()}_s"],
                "mtp_spt": v["mtp_spt"],
                "serial_spt": v["serial_spt"],
                "mtp_prefill_s": v["mtp_prefill_s"],
                "serial_prefill_s": v["serial_prefill_s"],
                "rounds": v["rounds"],
                "mean_depth": v["mean_depth"],
                "max_depth": v["max_depth"],
                "accept_rate": v["accept_rate"],
                "accepted": v["accepted"],
                "rejected": v["rejected"],
                "replays": v["replays"],
                "declared_rows": v["declared_rows"],
                "checked_rows": v["checked_rows"],
                "residual_divergence_count": v["divergence"],
                "raw_ratio": v["raw"],
                "gate_entry_temp_c": g.get("entry_temp_c", float("nan")),
                "gate_waited_s": g.get("waited_s", float("nan")),
            }, step=step)
        run.summary.update({
            "mtp_true_decode_s_median":
                statistics.median(r[f"mtp_{arm.lower()}_s"] for r in rows),
            "serial_true_decode_s_median":
                statistics.median(r[f"ser_{arm.lower()}_s"] for r in rows),
            "prompts_completed": len(rows),
            "worker_sha256": arms[rows[0]["prompt"]][arm]["meta"].get("worker_sha256"),
            "source_sha256": arms[rows[0]["prompt"]][arm]["meta"].get("source_sha256"),
        })
        run.finish()

    pred_mtp = [r["rounds_base"] * SITES_PER_FORWARD * CAST_US for r in rows]
    meas_mtp = [r["mtp_base_s"] - r["mtp_memo_s"] for r in rows]
    pred_ser = 512 * SITES_PER_FORWARD * CAST_US
    meas_ser = [r["ser_base_s"] - r["ser_memo_s"] for r in rows]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="phase3-analysis", name="e24-phase3-analysis",
        config=base_config, reinit=True,
    )
    run.summary.update({
        "mtp_effect_pct_median": statistics.median(mtp_e),
        "mtp_effect_pct_mean": statistics.fmean(mtp_e),
        "mtp_effect_pct_min": min(mtp_e),
        "mtp_effect_pct_max": max(mtp_e),
        "mtp_prompts_memo_faster": sum(1 for e in mtp_e if e > 0),
        "serial_effect_pct_median": statistics.median(ser_e),
        "serial_effect_pct_mean": statistics.fmean(ser_e),
        "serial_effect_pct_min": min(ser_e),
        "serial_effect_pct_max": max(ser_e),
        "serial_prompts_memo_faster": sum(1 for e in ser_e if e > 0),
        "prompts_completed": len(rows),
        "phase1_upper_bound_pct": 1.164,
        "phase1_coherence_forecast_pct": 0.58,
        "prereg_threshold_pct": 0.50,
        "realization_vs_phase1_mtp_median":
            statistics.median(m / p for m, p in zip(meas_mtp, pred_mtp)),
        "realization_vs_phase1_serial_median":
            statistics.median(m / pred_ser for m in meas_ser),
        "correctness_all_clean": report["correctness_all_clean"],
        "cool_gate_passed_real_gate": report["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": report["gate_qualified_for_timing"],
        "entry_temp_spread_c": report["entry_temp_spread_c"],
        "gate_passes_captured": len(gates),
        "timed_runs": 2 * len(rows),
    })
    table = wandb.Table(columns=[
        "prompt", "mtp_base_s", "mtp_memo_s", "mtp_effect_pct",
        "ser_base_s", "ser_memo_s", "ser_effect_pct",
        "rounds_base", "rounds_memo", "raw_base", "raw_memo"])
    for r in rows:
        table.add_data(*(r[c] for c in table.columns))
    run.log({"per_prompt": table})
    run.finish()
    print("logged 3 runs to", f"{ENTITY}/{PROJECT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
