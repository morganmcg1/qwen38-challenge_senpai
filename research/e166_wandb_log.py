#!/usr/bin/env python3
"""E166: publish the depth-policy reading and the island prelude to W&B.

One run, six record groups.

  legs        every leg of the two sessions, with honesty flags, temperatures,
              head provenance, the exact accept ledger and the width histogram.
  island      the FINDING 372 mechanism contrast: same binary, one environment
              variable apart, arm witness read back off each leg.
  price       the shipped dimensionless price translated into both harnesses.
  policy      the per-prompt depth the shipped rule chooses at the measured
              ranked `q`, and the flat-`q` disagreement map between prices.
  acceptance  step 2: `q` solved from the E159 accepted counts at each depth.
  replay      the traced adaptive leg, its realised width histogram, and the
              open-loop counterfactual histograms.

harness=local for every measured leg. The sessions are ungated by
construction, so nothing here is a ranked score or gate-qualified.

Usage:
  python3 research/e166_wandb_log.py --run-name e166-r0-depth-policy-reading
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS = HERE / "e166-artifacts"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
BASE_SHA = "4b6eb4f59a1f7b2722b9ec48a46d55f67a233753"


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def table(columns, rows):
    return wandb.Table(columns=list(columns), data=[list(r) for r in rows])


def read_meta(path):
    out = {}
    with open(path) as handle:
        for line in handle:
            if "=" in line:
                key, value = line.rstrip("\n").split("=", 1)
                out[key] = value
    return out


def read_witness(leg_dir):
    trace = os.path.join(leg_dir, "trace.txt")
    if not os.path.exists(trace):
        return ""
    with open(trace, errors="replace") as handle:
        for line in handle:
            if line.startswith("qwen-mtp-island-arm:"):
                return line.strip()
    return ""


def collect_legs(session_globs):
    legs = []
    for pattern in session_globs:
        for leg_dir in sorted(glob.glob(pattern)):
            report_path = os.path.join(leg_dir, "report.json")
            meta_path = os.path.join(leg_dir, "meta.txt")
            if not (os.path.exists(report_path) and os.path.exists(meta_path)):
                continue
            report = json.load(open(report_path))
            meta = read_meta(meta_path)
            blocks = report["block_request_seconds"]
            steady = blocks[1:] if len(blocks) > 1 else blocks
            hist = {}
            for value in report["effective_draft_lengths"]:
                hist[str(int(value))] = hist.get(str(int(value)), 0) + 1
            legs.append({
                "dir": leg_dir,
                "session": leg_dir.split("/")[2],
                "arm": meta.get("arm_label", ""),
                "arm_env": meta.get("arm_env", ""),
                "witness": read_witness(leg_dir),
                "tokens": int(meta.get("tokens", 0)),
                "rounds": report["round_count"],
                "edl": report["effective_mean_draft_len"],
                "accepted_draft_total": report["accepted_draft_total"],
                "rejected_draft_total": report["rejected_draft_total"],
                "accepted_draft_rate": report["accepted_draft_rate"],
                "declared_rows_total": report["declared_rows_total"],
                "reference_checked_row_total": report["reference_checked_row_total"],
                "emitted_token_total": report["emitted_token_total"],
                "all_tokens_matched": report["all_tokens_matched"],
                "parity_all_ok": report["parity_all_ok"],
                "residual_divergence_count": report["residual_divergence_count"],
                "mtp_seconds_per_token": report["parent_measured_seconds_per_token"],
                "R_steady_ms": 1e3 * sum(steady) / max(len(steady), 1),
                "hist": json.dumps(hist),
                "head_sha256": report["head_provenance"]["sha256"],
                "entry_c": float(meta.get("gpu_temp_entry_c", "nan")),
                "exit_c": float(meta.get("gpu_temp_exit_c", "nan")),
                "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate", "false"),
                "gate_qualified_for_timing": meta.get("gate_qualified_for_timing", "false"),
                "worker_sha256": meta.get("worker_sha256", ""),
                "cli_sha256": meta.get("cli_sha256", ""),
                "base_sha": meta.get("base_sha", ""),
            })
    return legs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="e166-r0-depth-policy-reading")
    args = parser.parse_args()

    doc = json.load(open(ARTIFACTS / "e166_policy_reading.json"))
    legs = collect_legs([
        "research/out/e166s0/b*-*", "research/out/e166s0w/b*-*",
        "research/out/e166trace/b*-*",
    ])

    run = wandb.init(project=PROJECT, entity=ENTITY, name=args.run_name,
                     job_type="analysis",
                     config={
                         "experiment": "e166-ranked-depth-optimum",
                         "harness": "local",
                         "base_sha": BASE_SHA,
                         "git_head": git_sha(),
                         "host_chip": "Apple M4 Pro",
                         "official_or_ranked_score": False,
                         "gate_qualified_for_timing": False,
                         "headStepCostRatio": doc["step1_constants"]["headStepCostRatio"],
                         "depthPriceArm": doc["step1_constants"]["depthPriceArm"],
                         "segmentedVerifyDepthCap":
                             doc["step1_constants"]["segmentedVerifyDepthCap"],
                     })

    leg_columns = list(legs[0].keys()) if legs else []
    if legs:
        run.log({"e166/legs": table(leg_columns, [[leg[c] for c in leg_columns]
                                                  for leg in legs])})

    price = doc["step1_price_translation"]
    run.log({"e166/price": table(
        ["harness", "law", "round_at_d0_ms", "marginal_row_ms", "implied_ratio",
         "shipped_ratio_in_ms", "shipped_over_truth"],
        [[name, entry.get("law", entry.get("note", "")),
          entry.get("round_at_d0_ms"), entry.get("marginal_row_ms",
                                                 entry.get("mean_marginal_row_ms")),
          entry.get("implied_ratio"), entry.get("shipped_ratio_in_ms"),
          entry.get("shipped_over_truth")]
         for name, entry in price.items()])})

    run.log({"e166/policy_per_prompt": table(
        ["prompt", "q", "shipped_edl", "d_star_ranked_global",
         "d_ship_greedy_at_flat_q", "d_ranked_greedy_at_flat_q"],
        [[r["prompt"], r["q"], r["shipped_edl"], r["d_star_ranked_global"],
          r["d_ship_greedy_at_flat_q"], r["d_ranked_greedy_at_flat_q"]]
         for r in doc["step1_prompt_table"]])})

    run.log({"e166/flat_q_map": table(
        ["q", "d_ship", "d_ranked"],
        [[m["q"], m["d_ship"], m["d_ranked"]] for m in doc["step1_flat_q_map"]])})

    step2 = doc["step2"]
    run.log({"e166/acceptance": table(
        ["pinned_depth", "rows_M", "accepted_draft_rate", "accepted_count", "implied_q"],
        [[r["pinned_depth"], r["rows_M"], r["accepted_draft_rate"],
          r["accepted_count"], r["implied_q"]] for r in step2["table"]])})

    summary = {
        "e166_width_conditional_term_in_shipped_policy": False,
        "e166_shipped_price_ratio": doc["step1_constants"]["headStepCostRatio"],
        "e166_ranked_price_ratio": price["harness=ranked"]["implied_ratio"],
        "e166_shipped_price_in_ranked_ms": price["harness=ranked"]["shipped_ratio_in_ms"],
        "e166_ranked_marginal_row_ms": price["harness=ranked"]["marginal_row_ms"],
        "e166_q_mean": step2["q_mean"],
        "e166_q_sd": step2["q_sd"],
        "e166_q_spread_pct": step2["q_spread_pct"],
        "e166_q_collapses_with_depth": step2["q_collapses_with_depth"],
        "e166_leg_count": len(legs),
    }

    replay = doc.get("step1_replay")
    if replay:
        run.log({"e166/replay": table(
            ["variant", "price_ratio", "margin_clamp", "mean_depth", "hist"],
            [[name, v["price_ratio"], v["margin_clamp"], v["mean_depth"],
              json.dumps(v["hist"])] for name, v in replay["variants"].items()])})
        run.log({"e166/break_even": table(
            ["variant", "mean_offered_rows", "ranked_round_ms",
             "tokens_per_round_at_e159_acceptance", "ranked_spt_ms",
             "ranked_spt_pct_vs_ship", "break_even_token_gain_pct",
             "modelled_token_gain_pct"],
            [[name, v["mean_offered_rows"], v["ranked_round_ms"],
              v["tokens_per_round_at_e159_acceptance"], v["ranked_spt_ms"],
              v["ranked_spt_pct_vs_ship"], v["break_even_token_gain_pct"],
              v["modelled_token_gain_pct"]]
             for name, v in replay["break_even"].items()])})
        summary.update({
            "e166_rounds_traced": replay["rounds_traced"],
            "e166_rounds_where_margin_clamp_binds":
                replay["rounds_where_margin_clamp_binds"],
            "e166_mean_depth_realised": replay["mean_depth_realised"],
            "e166_sched_positive_control_ok":
                replay["sched_positive_control"]["model_reproduces_trace_fields"],
            "e166_sched_steps_checked":
                replay["sched_positive_control"]["steps_checked"],
            "e166_replay_interior_match_rate":
                replay["walk_control"]["interior_match_rate"],
            "e166_replay_reproduces_every_interior_round":
                replay["walk_control"]["reproduces_every_interior_round"],
            "e166_ranked_spt_pct_ranked_price":
                replay["break_even"]["ranked_price"]["ranked_spt_pct_vs_ship"],
            "e166_ranked_spt_pct_no_margin_clamp":
                replay["break_even"]["no_margin_clamp"]["ranked_spt_pct_vs_ship"],
            "e166_mean_depth_ranked_price":
                replay["variants"]["ranked_price"]["mean_depth"],
            "e166_mean_depth_no_margin_clamp":
                replay["variants"]["no_margin_clamp"]["mean_depth"],
        })

    if legs:
        island = [leg for leg in legs if leg["arm"] in ("S", "P")]
        if island:
            summary["e166_island_ledger_invariant"] = (
                len({leg["accepted_draft_total"] for leg in island}) == 1
                and len({leg["rounds"] for leg in island}) == 1)
            summary["e166_island_witness_present"] = all(
                leg["witness"] for leg in island if leg["session"] == "e166s0w")
        summary["e166_all_tokens_matched"] = all(leg["all_tokens_matched"] for leg in legs)
        summary["e166_residual_divergence_total"] = sum(
            leg["residual_divergence_count"] for leg in legs)
        summary["e166_entry_temp_min_c"] = min(leg["entry_c"] for leg in legs)
        summary["e166_entry_temp_max_c"] = max(leg["entry_c"] for leg in legs)
        summary["e166_worker_sha256"] = sorted({leg["worker_sha256"] for leg in legs})[-1]

    run.summary.update(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
