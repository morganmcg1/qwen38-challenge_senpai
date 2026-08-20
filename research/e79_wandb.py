#!/usr/bin/env python3
"""E79: stream the head-economics census and pricing to W&B.

One run holds every leg's identity, the per-position acceptance census for
both provisioned heads, the per-draft phase decomposition, the compact draft
vocabulary break-even table and the rung-3 score arms.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
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
    # p_i is reported beside the width mix that produced it. A position is
    # only sampled by rounds whose verify width reaches it, so a p_i measured
    # under this fixture's width mix does not transfer to a ranked prompt
    # that runs at a different mix.
    hist = {int(k): v for k, v in leg["width_histogram"].items()}
    total = sum(hist.values())
    pos = wandb.Table(columns=["tag", "position", "reached", "accepted", "p",
                               "wilson_lo", "wilson_hi",
                               "rounds_at_width_eq_position",
                               "share_of_rounds_reaching_position"])
    for r in leg["positions"]:
        i = r["position"]
        pos.add_data(tag, i, r["reached"], r["accepted"], r["p"],
                     r["lo"], r["hi"], hist.get(i + 1, 0), r["reached"] / total)
    run.log({f"census/{tag}/per_position": pos})

    width = wandb.Table(columns=["tag", "verify_width", "rounds", "share"])
    for w in sorted(hist):
        width.add_data(tag, w, hist[w], hist[w] / total)
    run.log({f"census/{tag}/width_histogram": width})

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


def head_variant(run, pinned, declared):
    """The rung-3a natural experiment: two real heads, one machine, one base,
    one schedule, one fixture. The serial leg never uses the candidate head,
    so an unchanged serial time is the control that validates the pair."""
    if not (pinned and declared):
        return {}
    table = wandb.Table(columns=["head", "tag", "head_provenance_sha256",
                                 "mtp_decode_speedup",
                                 "mtp_seconds_per_token",
                                 "serial_seconds_per_token",
                                 "effective_mean_draft_len",
                                 "accepted_draft_rate",
                                 "gate_qualified_for_timing",
                                 "gpu_temp_entry_c", "gpu_temp_exit_c"])
    agg = {}
    for head, tags in (("pinned", pinned), ("declared", declared)):
        recs = [leg_record(t) for t in tags]
        for r in recs:
            table.add_data(head, r["tag"], r.get("head_provenance_sha256"),
                           r.get("mtp_decode_speedup"),
                           r.get("mtp_seconds_per_token"),
                           r.get("serial_seconds_per_token"),
                           r.get("effective_mean_draft_len"),
                           r.get("accepted_draft_rate"),
                           r.get("gate_qualified_for_timing"),
                           r.get("gpu_temp_entry_c"), r.get("gpu_temp_exit_c"))
        agg[head] = {k: st.mean([r[k] for r in recs]) for k in
                     ("mtp_decode_speedup", "mtp_seconds_per_token",
                      "serial_seconds_per_token", "accepted_draft_rate")}
    run.log({"rung3a/head_variant": table})
    p, d = agg["pinned"], agg["declared"]
    return {
        "head_variant_speedup_pinned": p["mtp_decode_speedup"],
        "head_variant_speedup_declared": d["mtp_decode_speedup"],
        "head_variant_speedup_delta_pct":
            100 * (d["mtp_decode_speedup"] / p["mtp_decode_speedup"] - 1),
        "head_variant_candidate_spt_delta_pct":
            100 * (d["mtp_seconds_per_token"]
                   / p["mtp_seconds_per_token"] - 1),
        "head_variant_serial_spt_delta_pct":
            100 * (d["serial_seconds_per_token"]
                   / p["serial_seconds_per_token"] - 1),
        "head_variant_acceptance_pinned": p["accepted_draft_rate"],
        "head_variant_acceptance_declared": d["accepted_draft_rate"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e79-head-economics-census")
    ap.add_argument("--legs", nargs="+", required=True,
                    help="TAG=CENSUS_JSON pairs")
    ap.add_argument("--price", required=True)
    ap.add_argument("--price-pinned")
    ap.add_argument("--reprice", required=True)
    ap.add_argument("--reprice-pinned")
    ap.add_argument("--chainfit")
    ap.add_argument("--calibrate")
    ap.add_argument("--pinned-legs", nargs="*", default=[],
                    help="tags of the organizer-pinned head variant")
    ap.add_argument("--declared-legs", nargs="*", default=[],
                    help="tags of the manifest-declared head variant")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    pairs = [p.split("=", 1) for p in args.legs]
    legs = [leg_record(tag) for tag, _ in pairs]
    census = {tag: json.loads(Path(c).read_text())["legs"][0]
              for tag, c in pairs}
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

    if args.reprice_pinned:
        rp = json.loads(Path(args.reprice_pinned).read_text())
        t = wandb.Table(columns=sched.columns)
        for a in rp["arms"]:
            t.add_data(a["arm"], a["head_step_ms"], a["M"],
                       a["tokens_per_round"], a["ms_per_token"],
                       a["delta_pct_vs_ship"])
        run.log({"rung0/reprice_pinned_head": t})

    if args.price_pinned:
        pp = json.loads(Path(args.price_pinned).read_text())
        t = wandb.Table(columns=arms.columns)
        for a in pp["arms"]:
            t.add_data(a["arm"], a.get("head_scale"),
                       a.get("p_scale", a.get("acceptance_floor")),
                       a.get("vocabulary_rows"), a.get("breakeven_coverage"),
                       a["median"], a["delta_pct"])
        run.log({"rung3/score_arms_pinned_head": t})

    if args.chainfit:
        cf = json.loads(Path(args.chainfit).read_text())["prompts"]
        cols = [k for k, v in cf[0].items() if not isinstance(v, list)]
        run.log({"rung0/chainfit": wandb.Table(
            columns=cols, data=[[r[k] for k in cols] for r in cf])})

    calib = {}
    if args.calibrate:
        cal = json.loads(Path(args.calibrate).read_text())
        sweep = wandb.Table(columns=list(cal["h_sweep"][0].keys()))
        for r in cal["h_sweep"]:
            sweep.add_data(*r.values())
        run.log({"calibration/h_sweep_out_of_sample": sweep})
        cols = [k for k, v in cal["arm3"][0].items()
                if not isinstance(v, list)]
        run.log({"calibration/arm3_chain": wandb.Table(
            columns=cols, data=[[r[k] for k in cols] for r in cal["arm3"]])})
        rr = cal["ranked_round_cost"]
        calib = {
            "ranked_round_fixed_ms": rr["intercept"],
            "ranked_round_slope_ms_per_draft": rr["slope"],
            "ranked_round_fit_r2": rr["r2"],
            "ranked_total_marginal_ratio": rr["marginal_ratio"],
            "ranked_head_only_ratio": rr["head_only_ratio"],
            "ranked_head_share_of_marginal": rr["head_share_of_marginal"],
            "h_sweep_mean_abs_relative_error":
                st.mean(abs(r["error"] / r["observed_drafts_h_hi"])
                        for r in cal["h_sweep"]),
            "h_sweep_direction_correct": sum(
                1 for r in cal["h_sweep"]
                if (r["predicted_drafts_h_hi"] - r["observed_drafts_h_lo"])
                * (r["observed_drafts_h_hi"] - r["observed_drafts_h_lo"]) > 0),
            "arm3_chain_identified_prompts":
                sum(1 for r in cal["arm3"] if r["identified"]),
            "arm3_predicted_median": cal["arm3_predicted_median"],
            "arm3_observed_median": cal["arm3_observed"]["published_median"],
        }

    variant = head_variant(run, args.pinned_legs, args.declared_legs)

    free = next(a for a in price["arms"] if a["arm"] == "head cost x 0.00")
    v32 = next(a for a in price["arms"] if a.get("vocabulary_rows") == 32794)
    p99 = next(a for a in price["arms"]
               if a["arm"] == "p_i -> 0.990 every position")
    deep = next((a for a in price["arms"]
                 if a["arm"] == "deepest 3 restored to position 1"), None)
    shape = next((a for a in price["arms"]
                  if a["arm"] == "measured position shape"), None)
    free_sched = next(a for a in reprice["arms"]
                      if a["arm"] == "head free, shipped flat price")
    summary = {
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
        "rung0_free_head_ms_per_token_delta_pct":
            free_sched["delta_pct_vs_ship"],
        # The objective is the published MEDIAN, so the rung-0 stop rule is
        # read against the median arm. Pooled local ms/token is a secondary
        # observation that no prompt is scored on.
        "rung0_free_head_median_delta_pct": free["delta_pct"],
        "rung0_stop_rule_triggered": abs(free["delta_pct"]) < 0.5,
    }
    if deep is not None and shape is not None:
        summary["score_delta_pct_measured_position_shape"] = shape["delta_pct"]
        summary["score_delta_pct_deepest3_to_pos1"] = deep["delta_pct"]
        summary["score_gain_pct_from_deepest3"] = (deep["delta_pct"]
                                                   - shape["delta_pct"])
    if args.declared_legs:
        widths = [census[t]["M"] for t in args.declared_legs if t in census]
        if widths:
            summary["local_mean_verify_width"] = st.mean(widths)
    ranked_widths = sorted(1 + v["drafts_per_round"]
                           for v in price["working_point"].values())
    summary["ranked_mean_verify_width"] = st.mean(ranked_widths)
    summary["ranked_median_setting_verify_width"] = st.mean(ranked_widths[3:5])
    summary.update(calib)
    summary.update(variant)
    run.summary.update(summary)
    print("wandb_run_id=%s" % run.id)
    print("wandb_url=%s" % run.url)
    run.finish()


if __name__ == "__main__":
    main()
