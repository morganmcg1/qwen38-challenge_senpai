#!/usr/bin/env python3
"""Publish the E212 width-cost census to W&B.

    usage: research/e212_wandb.py research/e212-report.json [--name NAME]
                                  [--notes TEXT]

Every number in the document is `harness=local` and ungated: the legs run with
the per-round phase trace on and `MLXFAST_LOCAL_COOL_GATE=0` under the standing
counterbalanced conditions. No number here is an official or ranked score, and
no absolute leg time is a candidate speed claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e212-width-cost-law"
GROUP = "qwen38-r1-e212-width-cost-law"

PHASE_COLUMNS = ["round_us", "draft_build_us", "d_head1_us", "d_chain_us",
                 "d_submit1_us", "d_submit2_us", "verify_build_us",
                 "eval_wall_us", "readout_us", "commit_us", "upkeep_us"]


def width_table(report: dict) -> wandb.Table:
    columns = (["m", "round_ms", "leg_spread_ms", "exceeds_noise_floor",
                "rounds_at_width", "accepted_mean", "tokens_per_round",
                "mtp_seconds_per_token", "weight_bytes_per_round",
                "e182_dated_round_ms", "delta_vs_dated_ms"]
               + [f"{p[:-3]}_ms" for p in PHASE_COLUMNS])
    rows = []
    dated = report["dated_table_e182_round_ms"]
    for key, entry in sorted(report["width_table"].items(), key=lambda kv: int(kv[0])):
        m = int(key)
        old = dated.get(str(m)) or dated.get(m)
        phases = entry.get("phase_ms", {})
        rows.append([
            m, entry.get("round_ms"), entry.get("round_ms_leg_spread"),
            entry.get("leg_spread_exceeds_noise_floor"),
            sum(entry.get("rounds_at_width", [])),
            entry.get("accepted_mean"), entry.get("tokens_per_round_mean"),
            entry.get("mtp_seconds_per_token"),
            entry.get("weight_bytes_per_round"),
            old, (entry.get("round_ms") - old) if old else None,
        ] + [phases.get(p, {}).get("value") for p in PHASE_COLUMNS])
    return wandb.Table(columns=columns, data=rows)


def leg_table(report: dict) -> wandb.Table:
    columns = ["leg", "session", "arm", "width_m", "band_sync", "leg_index",
               "rounds_at_width", "round_ms_median", "round_ms_ci_lo",
               "round_ms_ci_hi", "gpu_temp_entry_c", "gpu_temp_exit_c",
               "all_tokens_matched", "residual_divergence_count",
               "mtp_seconds_per_token", "serial_seconds_per_token",
               "effective_mean_draft_len", "accepted_draft_rate",
               "worker_sha256", "base_sha"]
    rows = []
    for leg in report["legs"]:
        ci = leg["phase_ci_ms"].get("round_us", [None, None])
        rows.append([
            leg["leg"], leg["session"], leg["arm"], leg["width_m"],
            leg["band_sync"], leg["leg_index"], leg["rounds_at_width"],
            leg["phases_ms"].get("round_us", {}).get("median"), ci[0], ci[1],
            leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"],
            leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["mtp_seconds_per_token"], leg["serial_seconds_per_token"],
            leg["effective_mean_draft_len"], leg["accepted_draft_rate"],
            leg["worker_sha256"], leg["base_sha"],
        ])
    return wandb.Table(columns=columns, data=rows)


def step_table(report: dict) -> wandb.Table:
    columns = ["boundary", "m", "raw_step_ms", "shipped_share_at_or_above_m",
               "round_weighted_step_ms", "weight_bytes_step",
               "leg_spread_lo_ms", "leg_spread_hi_ms"]
    rows = [[s["boundary"], s["m"], s["raw_step_ms"],
             s["shipped_share_at_or_above_m"], s["round_weighted_step_ms"],
             s["weight_bytes_step"], s["leg_spread_lo"], s["leg_spread_hi"]]
            for s in report["steps"]]
    return wandb.Table(columns=columns, data=rows)


def served_table(report: dict) -> wandb.Table:
    distribution = report["shipped_served_width_distribution"]
    columns = ["m", "rounds", "share", "share_at_or_above"]
    rows = []
    for key, count in sorted(distribution["counts"].items(), key=lambda kv: int(kv[0])):
        rows.append([int(key), count, distribution["share"].get(key),
                     distribution["share_at_or_above"].get(key)])
    return wandb.Table(columns=columns, data=rows)


def band_table(report: dict) -> wandb.Table:
    columns = ["leg", "width_m", "band_pre_ms", "band_gdn_mixer_ms",
               "band_gdn_mlp_ms", "band_fa_mixer_ms", "band_fa_mlp_ms",
               "band_fwd", "round_ms", "d_submit2_ms"]
    rows = []
    for leg in report["legs"]:
        if not leg["band_sync"]:
            continue
        phases = leg["phases_ms"]
        rows.append([
            leg["leg"], leg["width_m"],
            phases.get("band_pre_us", {}).get("median"),
            phases.get("band_gdn_mixer_us", {}).get("median"),
            phases.get("band_gdn_mlp_us", {}).get("median"),
            phases.get("band_fa_mixer_us", {}).get("median"),
            phases.get("band_fa_mlp_us", {}).get("median"),
            phases.get("band_fwd", {}).get("median"),
            phases.get("round_us", {}).get("median"),
            phases.get("d_submit2_us", {}).get("median"),
        ])
    return wandb.Table(columns=columns, data=rows)


def band_step_table(report: dict) -> wandb.Table:
    bands = ["band_pre_us", "band_gdn_mixer_us", "band_gdn_mlp_us",
             "band_fa_mixer_us", "band_fa_mlp_us"]
    columns = (["boundary", "band_arm_round_step_ms", "band_step_total_ms"]
               + [f"{b[len('band_'):-len('_us')]}_step_ms" for b in bands]
               + [f"{b[len('band_'):-len('_us')]}_share" for b in bands])
    rows = []
    for step in report.get("band_steps", []):
        rows.append([step["boundary"], step["band_arm_round_step_ms"],
                     step["band_step_total_ms"]]
                    + [step["band_step_ms"].get(b) for b in bands]
                    + [step["band_share"].get(b) for b in bands])
    return wandb.Table(columns=columns, data=rows)


def ipg_table(report: dict) -> wandb.Table:
    law = report.get("ipg_law") or {}
    columns = ["ipg", "m_one_pass", "round_ms_one_pass", "m_two_pass",
               "round_ms_two_pass", "ratio", "implied_fixed_overhead_ms"]
    rows = [[p["ipg"], p["m_one_pass"], p["round_ms_one_pass"],
             p["m_two_pass"], p["round_ms_two_pass"], p["ratio"],
             p["implied_fixed_overhead_ms"]]
            for p in law.get("fixed_ipg_pass_doubling", [])]
    return wandb.Table(columns=columns, data=rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--name", default=EXPERIMENT)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    first = report["legs"][0]
    config = {
        "experiment": EXPERIMENT,
        "harness": "local",
        "official_score": False,
        "rankable": False,
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "trace_perturbs_timing": True,
        "noise_floor_ms": report["noise_floor_ms"],
        "sessions": ",".join(report["sessions"]),
        "base_sha": first["base_sha"],
        "worker_sha256": first["worker_sha256"],
        "head_provenance_sha256": first["head_provenance_sha256"],
        "decode_tokens": first["decode_tokens"],
        "fixture": "public --local-iterate fixture",
        "reference_source": "candidate-generated reference rows",
        "surface": "cap-8 + staged (9,5) + selective-m6 single-pass",
    }

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=args.name, notes=args.notes, config=config,
                     job_type="analysis")
    for key, entry in report["width_table"].items():
        run.summary[f"round_ms/m{key}"] = entry.get("round_ms")
        run.summary[f"leg_spread_ms/m{key}"] = entry.get("round_ms_leg_spread")
    for step in report["steps"]:
        run.summary[f"step_raw_ms/{step['boundary']}"] = step["raw_step_ms"]
        run.summary[f"step_weighted_ms/{step['boundary']}"] = \
            step["round_weighted_step_ms"]
    for model, block in (report.get("shape_fit") or {}).items():
        if isinstance(block, dict) and "aicc" in block:
            run.summary[f"shape_fit/{model}_aicc"] = block["aicc"]
            run.summary[f"shape_fit/{model}_rss"] = block["rss"]
    run.summary["all_tokens_matched_every_leg"] = all(
        leg["all_tokens_matched"] is True for leg in report["legs"])

    law = report.get("ipg_law") or {}
    for ipg, ms in (law.get("single_pass_ipg_ladder_ms") or {}).items():
        run.summary[f"single_pass_cost_ms/ipg{ipg}"] = ms
    for pair in law.get("fixed_ipg_pass_doubling", []):
        run.summary[
            f"pass_doubling_ratio/ipg{pair['ipg']}_m{pair['m_two_pass']}"
        ] = pair["ratio"]
    for step in report.get("band_steps", []):
        for band, share in step["band_share"].items():
            run.summary[f"band_share/{step['boundary']}/{band}"] = share

    run.log({"width_table": width_table(report),
             "legs": leg_table(report),
             "steps": step_table(report),
             "served_width_distribution": served_table(report),
             "band_arms": band_table(report),
             "band_steps": band_step_table(report),
             "ipg_pass_doubling": ipg_table(report)})
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
