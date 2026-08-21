#!/usr/bin/env python3
"""Publish the E106 rung 0 dispatch census to W&B.

    usage:
      research/e106_wandb_log.py --tag TAG [--tag TAG ...] \
          --rung0-json research/out/e106/rung0-d4.json \
          [--trace-json research/out/e106/trace-d4.json]

One run per census leg. The config carries the full experiment identity tuple
and the census contract; the metrics carry the round anchor, the fitted
per-dispatch fixed cost and streaming rate, and every family residual. The
tables carry the per-family and per-tensor lines.

A census leg is NEVER a timing leg: the census swizzle serialises every
dispatch, so host wall clock is invalid and only Metal's GPU clock counts.
Every run therefore logs `timing_valid=false` beside the standard
`cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e106-dispatch-fixed-cost-law"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
DRAM_PEAK_GB_S = 273.0

# Finding 36, fitted by the advisor on the pre-E100 E96 round census.
FINDING_36 = {
    "F_us_per_dispatch": 9.90,
    "S_us_per_gb": 3670.2,
    "S_gb_per_s": 272.5,
    "streaming_dispatches_per_round": 514,
    "round_us": 127_533.0,
    "n5120_excess_us_per_dispatch": 8.43,
    "n5120_excess_pct_of_round": 1.693,
}


def meta(tag):
    path = pathlib.Path("research/out") / tag / "meta.txt"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                out[key] = value
    return out


def score(tag):
    path = pathlib.Path("research/out") / tag / "score.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("metrics", {})


def git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def tensor_table(tensors):
    return wandb.Table(
        columns=["tensor", "K", "N", "per_round", "gb_per_disp", "us_per_disp",
                 "p05_us", "median_us", "min_us", "max_us", "sd_us", "law_us",
                 "excess_us", "excess_p05_us", "excess_pct",
                 "x_bytes_per_threadgroup"],
        data=[[name, t["k"], t["n"], t["per_round"], t["gb"], t["mean_us"],
               t["p05_us"], t["median_us"], t["min_us"], t["max_us"],
               t["stdev_us"], t["law_us"], t["excess_us"], t["excess_p05_us"],
               t["excess_pct"], t["x_bytes_per_tg"]]
              for name, t in tensors.items()])


def round_account(tensors, round_us, m1_f=10.73, m1_s=4003.3):
    """Price the per-tensor split over the whole census round."""
    total = dict(us=0.0, exc=0.0, gb=0.0, n=0.0, m1=0.0)
    for t in tensors.values():
        n = t["per_round"]
        total["us"] += n * t["mean_us"]
        total["exc"] += n * t["excess_us"]
        total["gb"] += n * t["gb"]
        total["n"] += n
        total["m1"] += n * (m1_f + t["gb"] * m1_s)
    gross = total["gb"] / (total["us"] * 1e-6)
    return {
        "streaming_gb_per_round": total["gb"],
        "streaming_us_per_round": total["us"],
        "streaming_pct_of_round": 100.0 * total["us"] / round_us,
        "streaming_gross_gb_per_s": gross,
        "streaming_gross_pct_dram_peak": 100.0 * gross / DRAM_PEAK_GB_S,
        "n5120_excess_us_per_round": total["exc"],
        "n5120_excess_pct_of_round": 100.0 * total["exc"] / round_us,
        "m1_law_us_per_round": total["m1"],
        "rate_na_gap_us_per_round": total["us"] - total["m1"],
        "rate_na_gap_pct_of_round": 100.0 * (total["us"] - total["m1"])
        / round_us,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--rung0-json", required=True)
    ap.add_argument("--trace-json")
    ap.add_argument("--m1-json",
                    help="serial M=1 control split from the same census leg")
    args = ap.parse_args()

    rung0 = json.loads(pathlib.Path(args.rung0_json).read_text())
    trace = (json.loads(pathlib.Path(args.trace_json).read_text())
             if args.trace_json else {})
    m1 = (json.loads(pathlib.Path(args.m1_json).read_text())
          if args.m1_json else {})

    for tag in args.tag:
        entry = rung0.get(tag)
        if entry is None:
            print(f"e106_wandb_log: no rung0 payload for {tag}, skipping")
            continue
        info = meta(tag)
        metrics_json = score(tag)
        config = {
            "experiment": "e106",
            "rung": 0,
            "harness": "local",
            "leg": tag,
            "local_mode": info.get("local_mode", "--local-iterate"),
            "census": True,
            "buffer_limit_ops": 0,
            "buffer_limit_mb": 1,
            "dispatch_trace": info.get("dispatch_trace", "0") == "1",
            "forced_drafts": int(info.get("forced_drafts", -1)),
            "verify_width": entry["width"],
            "tokens": int(info.get("tokens", 0)),
            "host": HOST,
            "chip": "Apple M4 Pro",
            "gpu_cores": 20,
            "toolchain": "swift 6.3.3 / macOS 26.5.2",
            "base_sha": info.get("base_sha"),
            "candidate_sha": git_sha(),
            "worker_sha256": info.get("worker_sha256"),
            "head_dir": info.get("head_dir"),
            "head_provenance_sha256": metrics_json.get(
                "head_provenance_sha256"),
            "timing_valid": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "finding_36_reference": FINDING_36,
        }
        run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                         job_type="dispatch-census", name=f"e106-{tag}",
                         config=config, reinit=True)

        fit = entry["fit"]
        families = entry["families"]
        stream_disp = sum(f["dispatches_per_round"] for f in families.values())
        metrics = {
            "census_rounds": entry["rounds"],
            "round_us": entry["round_us"],
            "round_dispatches": entry["round_dispatches"],
            "streaming_dispatches_per_round": stream_disp,
            "fit_F_us_per_dispatch": fit["F_us"],
            "fit_S_us_per_gb": fit["S_us_per_gb"],
            "fit_S_gb_per_s": 1e6 / fit["S_us_per_gb"],
            "fit_S_pct_of_dram_peak": 100.0 * (1e6 / fit["S_us_per_gb"])
            / DRAM_PEAK_GB_S,
            "fit_r2": fit["r2"],
            "fixed_cost_us_per_round": fit["F_us"] * stream_disp,
            "fixed_cost_pct_of_round": 100.0 * fit["F_us"] * stream_disp
            / entry["round_us"],
        }
        for name, f in families.items():
            key = name.replace(" + ", "_and_").replace(".", "_").replace(
                " ", "_")
            metrics[f"{key}_dispatches_per_round"] = f["dispatches_per_round"]
            metrics[f"{key}_gb_per_dispatch"] = f["gb_per_dispatch"]
            metrics[f"{key}_us_per_dispatch"] = f["us_per_dispatch"]
            metrics[f"{key}_us_per_round"] = f["us_per_round"]
            metrics[f"{key}_gb_per_s"] = f["gb_per_s"]
            metrics[f"{key}_gb_per_s_after_f"] = f["gb_per_s_after_f"]
            metrics[f"{key}_pct_dram_peak_after_f"] = (
                100.0 * f["gb_per_s_after_f"] / DRAM_PEAK_GB_S)
            metrics[f"{key}_resid_us"] = f["resid_us"]
            metrics[f"{key}_resid_pct"] = f["resid_pct"]
            metrics[f"{key}_excess_us_per_round"] = f["excess_us_per_round"]
            metrics[f"{key}_excess_pct_of_round"] = f["excess_pct_of_round"]

        family_rows = [
            [name, f["dispatches_per_round"], f["gb_per_dispatch"],
             f["us_per_dispatch"], f["gb_per_s"], f["gb_per_s_after_f"],
             100.0 * f["gb_per_s_after_f"] / DRAM_PEAK_GB_S, f["refit_us"],
             f["resid_us"], f["resid_pct"], f["excess_pct_of_round"],
             str(f["threadgroups"]), str(f["grids"]), str(f["threadgroup_dims"])]
            for name, f in families.items()]
        run.log({"streaming_families": wandb.Table(
            columns=["family", "disp_per_round", "gb_per_disp", "us_per_disp",
                     "gb_per_s", "gb_per_s_after_F", "pct_dram_peak_after_F",
                     "law_us", "resid_us", "resid_pct", "excess_pct_of_round",
                     "threadgroups", "grid", "threadgroup"],
            data=family_rows)})

        shape_rows = [
            [s["kernel"][:80], str(s["grid"]), str(s["tg"]), s["phase"],
             s["per_round"], s["us_per_dispatch"], s["min_us"], s["max_us"],
             s["us_per_round"], s["threadgroups"], s["waves_20_cores"],
             s["gb_per_dispatch"]]
            for s in entry["shapes"] if s["us_per_round"] >= 1.0]
        run.log({"dispatch_census": wandb.Table(
            columns=["kernel", "grid", "threadgroup", "phase", "per_round",
                     "us_per_dispatch", "min_us", "max_us", "us_per_round",
                     "threadgroups", "threadgroups_per_core", "gb_per_disp"],
            data=shape_rows)})

        tentry = trace.get(tag)
        if tentry:
            tensors = tentry["tensors"]
            for name, t in tensors.items():
                key = name.replace(".", "_")
                metrics[f"tensor_{key}_us_per_dispatch"] = t["mean_us"]
                metrics[f"tensor_{key}_excess_us"] = t["excess_us"]
                metrics[f"tensor_{key}_excess_p05_us"] = t["excess_p05_us"]
                metrics[f"tensor_{key}_excess_median_us"] = t[
                    "excess_median_us"]
                metrics[f"tensor_{key}_excess_pct"] = t["excess_pct"]
                metrics[f"tensor_{key}_gb_per_s_after_f"] = t[
                    "gb_per_s_after_f"]
            run.log({"tensor_split": tensor_table(tensors)})
            metrics.update(round_account(tensors, entry["round_us"]))

        mentry = m1.get(tag)
        if mentry:
            mfit = mentry["fit"]
            metrics["m1_fit_F_us_per_dispatch"] = mfit["F_us"]
            metrics["m1_fit_S_us_per_gb"] = mfit["S_us_per_gb"]
            metrics["m1_fit_S_gb_per_s"] = 1e6 / mfit["S_us_per_gb"]
            metrics["m1_fit_S_pct_of_dram_peak"] = (
                100.0 * (1e6 / mfit["S_us_per_gb"]) / DRAM_PEAK_GB_S)
            metrics["m1_fit_r2"] = mfit["r2"]
            for name, t in mentry["tensors"].items():
                key = name.replace(".", "_")
                metrics[f"m1_tensor_{key}_us_per_dispatch"] = t["mean_us"]
                metrics[f"m1_tensor_{key}_excess_us"] = t["excess_us"]
            run.log({"m1_tensor_split": tensor_table(mentry["tensors"])})

        run.log(metrics)
        print(f"{tag} {run.id} {run.url}")
        run.finish()


if __name__ == "__main__":
    main()
