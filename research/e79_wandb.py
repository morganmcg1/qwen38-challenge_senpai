#!/usr/bin/env python3
"""E79: stream the head-economics census and pricing to W&B.

One run holds every leg's identity, the per-position acceptance census for
both provisioned heads, the per-draft phase decomposition, the compact draft
vocabulary break-even table and the rung-3 score arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def read_meta(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def leg_record(tag):
    out = Path("research/out") / tag
    meta = read_meta(out / "meta.txt")
    rec = {"tag": tag}
    rec.update({k: meta.get(k) for k in (
        "base_sha", "host", "chip", "memory_bytes", "tokens", "sync_head",
        "cool_gate", "cool_gate_passed_real_gate", "gate_qualified_for_timing",
        "official_or_ranked_score", "head_dir", "worker_sha256", "cli_sha256",
        "gpu_temp_entry_c", "gpu_temp_exit_c", "trace_rounds",
        "dirty_candidate_paths", "started", "finished", "exit")})
    score = out / "score.json"
    if score.exists():
        m = json.loads(score.read_text())["metrics"]
        rec.update({k: m.get(k) for k in (
            "mtp_decode_speedup", "serial_seconds_per_token",
            "mtp_seconds_per_token", "effective_mean_draft_len",
            "accepted_draft_rate", "all_tokens_matched",
            "residual_divergence_count", "decode_tokens", "mtp_depth",
            "head_provenance_sha256")})
    return rec


def census_tables(run, tag, census_path):
    leg = json.loads(Path(census_path).read_text())["legs"][0]
    pos = wandb.Table(columns=["tag", "position", "reached", "accepted", "p",
                               "wilson_lo", "wilson_hi"])
    for r in leg["positions"]:
        pos.add_data(tag, r["position"], r["reached"], r["accepted"], r["p"],
                     r["lo"], r["hi"])
    run.log({f"census/{tag}/per_position": pos})

    ema = wandb.Table(columns=["tag", "position", "converged_ema",
                               "shipped_seed"])
    for i, (a, b) in enumerate(zip(leg.get("final_ema", []),
                                   leg.get("shipped_ema_seed", [])), 1):
        ema.add_data(tag, i, a, b)
    run.log({f"census/{tag}/accept_ema": ema})

    fit = wandb.Table(columns=["tag", "phase", "ms_per_draft", "fixed_ms",
                               "r2", "n"])
    for phase, f in leg.get("phase_fit", {}).items():
        fit.add_data(tag, phase, f["slope_ms_per_draft"], f["fixed_ms"],
                     f["r2"], f["n"])
    run.log({f"phase/{tag}/per_draft_fit": fit})

    width = wandb.Table(columns=["tag", "verify_width", "rounds",
                                 "round_ms_median"])
    for w, v in sorted(leg.get("round_ms_by_width", {}).items(),
                       key=lambda kv: int(kv[0])):
        width.add_data(tag, int(w), v["n"], v["median"])
    run.log({f"phase/{tag}/round_ms_by_width": width})
    return leg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e79-head-economics-census")
    ap.add_argument("--legs", nargs="+", required=True,
                    help="TAG=CENSUS_JSON pairs")
    ap.add_argument("--price", required=True)
    ap.add_argument("--reprice", required=True)
    ap.add_argument("--chainfit")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    pairs = [p.split("=", 1) for p in args.legs]
    legs = [leg_record(tag) for tag, _ in pairs]
    price = json.loads(Path(args.price).read_text())
    reprice = json.loads(Path(args.reprice).read_text())

    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name,
                     job_type="research-census", notes=args.notes,
                     config={"experiment": "qwen38-r1-e79-head-economics-census",
                             "harness": "local", "legs": legs,
                             "cost_model": price["cost_model"],
                             "head_step_bytes": price["head_step_bytes"]})

    identity = wandb.Table(columns=list(legs[0].keys()))
    for rec in legs:
        identity.add_data(*[rec.get(k) for k in legs[0].keys()])
    run.log({"legs/identity": identity})

    for tag, census in pairs:
        census_tables(run, tag, census)

    work = wandb.Table(columns=["prompt", "drafts_per_round", "constant_p",
                                "head_share_of_round"])
    for name, v in price["working_point"].items():
        work.add_data(name, v["drafts_per_round"], v["p"],
                      v["head_share_of_round"])
    run.log({"rung3/ranked_working_point": work})

    arms = wandb.Table(columns=["arm", "head_scale", "acceptance",
                                "vocabulary_rows", "breakeven_coverage",
                                "median", "delta_pct"])
    for a in price["arms"]:
        arms.add_data(a["arm"], a.get("head_scale"),
                      a.get("p_scale", a.get("acceptance_floor")),
                      a.get("vocabulary_rows"), a.get("breakeven_coverage"),
                      a["median"], a["delta_pct"])
    run.log({"rung3/score_arms": arms})

    sched = wandb.Table(columns=["arm", "head_step_ms", "M",
                                 "tokens_per_round", "ms_per_token",
                                 "delta_pct_vs_ship"])
    for a in reprice["arms"]:
        sched.add_data(a["arm"], a["head_step_ms"], a["M"],
                       a["tokens_per_round"], a["ms_per_token"],
                       a["delta_pct_vs_ship"])
    run.log({"rung0/reprice": sched})

    if args.chainfit:
        cf = json.loads(Path(args.chainfit).read_text())["prompts"]
        cols = [k for k, v in cf[0].items() if not isinstance(v, list)]
        run.log({"rung0/chainfit": wandb.Table(
            columns=cols, data=[[r[k] for k in cols] for r in cf])})

    free = next(a for a in price["arms"] if a["arm"] == "head cost x 0.00")
    v32 = next(a for a in price["arms"] if a.get("vocabulary_rows") == 32794)
    p99 = next(a for a in price["arms"]
               if a["arm"] == "p_i -> 0.990 every position")
    run.summary.update({
        "baseline_published_median": price["baseline_median"],
        "measured_head_step_cost_ratio": price["cost_model"]["measured_h"],
        "shipped_head_step_cost_ratio": price["cost_model"]["shipped_h"],
        "head_ms_per_draft": price["cost_model"]["head_slope_ms"],
        "round_ms_per_draft": price["cost_model"]["round_slope_ms"],
        "score_delta_pct_free_head": free["delta_pct"],
        "score_delta_pct_vocab_32768": v32["delta_pct"],
        "breakeven_coverage_vocab_32768": v32["breakeven_coverage"],
        "measured_coverage_vocab_32768": v32["coverage_id_prefix"],
        "score_delta_pct_p_099": p99["delta_pct"],
    })
    print("wandb_run_id=%s" % run.id)
    print("wandb_url=%s" % run.url)
    run.finish()


if __name__ == "__main__":
    main()
