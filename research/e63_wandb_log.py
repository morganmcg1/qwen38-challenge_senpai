#!/usr/bin/env python3
"""Log E63 -- is the QMV width cliff a memory-level-parallelism collapse? -- to W&B.

Two records, one run:

  rung 0   the compile-only NA census that decides the experiment. In-flight
           weight loads, registers, peak live, allocas, private traffic, and the
           three-way model comparison against askeladd's ladder.
  rung 1   this host's own roofline and weight-stream bandwidth ladder from
           `QwenQMVCostCurveTests`, next to askeladd's E61 rung 1 numbers.

Rung 0 is the terminal readout: its preregistered kill rule fired, so no
intervention arm exists and there is no candidate timing leg.

  python3 research/e63_wandb_log.py \
      --rung0 research/e63-artifacts/rung0.json \
      --rung1 research/e63-artifacts/e63-cost-curve.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

# askeladd's E61 rung 1 ladder, the sole prior source of these numbers.
ASKELADD_BW = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946, 6: 117.8, 7: 97.9}
ASKELADD_PEAK_GB_S = 227.9

# Ranked QMV share by verify width, supplied in the assignment. Used to price
# any cell effect with the RANKED mixture rather than the local one.
RANKED_WIDTH_SHARE = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334, 7: 0.122,
                      8: 0.0735, 9: 0.0575}
LOCAL_WIDTH_SHARE = {4: 0.0377, 5: 0.0668, 6: 0.1889, 7: 0.0382, 8: 0.0428,
                     9: 0.6257}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True,
                          text=True).stdout.strip()


def ladder_from_cost_curve(curve: dict) -> list[dict]:
    """Achieved weight-stream bandwidth per NA, from the scored-shape sweep.

    The shipped dispatch table reaches NA in {2, 3, 4, 5} only, and NA = 5
    appears only inside the two-stream M = 9 cell, so a clean single-stream rung
    exists for NA = 2, 3, 4. Mixed cells are reported and marked, never averaged
    into a rung as though they were clean.
    """
    rows = []
    for shape in curve.get("shapes", []):
        for row in shape.get("rows", []):
            ipg, streams = row.get("inputs_per_group"), row.get("weight_streams")
            if not isinstance(ipg, int) or not isinstance(streams, int):
                continue
            if not row.get("crossrow"):
                continue
            seconds = row["seconds_per_call"]
            bytes_moved = shape["weight_bytes"] * streams
            m = row["m"]
            tail = m % ipg
            rows.append({
                "shape": shape["name"],
                "k": shape["k"],
                "n": shape["n"],
                "m": m,
                "na": ipg,
                "weight_streams": streams,
                "clean_single_na": streams == 1 and tail == 0,
                "tail_na": (max(tail, 2) if tail else None),
                "seconds_per_call": seconds,
                "seconds_per_call_min": row["seconds_per_call_min"],
                "seconds_per_call_max": row["seconds_per_call_max"],
                "weight_bytes_per_call": bytes_moved,
                "achieved_gb_per_s": bytes_moved / seconds / 1e9,
                "row0_bitwise_matches_m1": row.get("row0_bitwise_matches_m1"),
            })
    return rows


def rung_summary(rows: list[dict]) -> dict:
    """Byte-weighted achieved bandwidth per NA over the clean single-stream cells."""
    out = {}
    for na in sorted({r["na"] for r in rows if r["clean_single_na"]}):
        clean = [r for r in rows if r["clean_single_na"] and r["na"] == na]
        total_bytes = sum(r["weight_bytes_per_call"] for r in clean)
        total_seconds = sum(r["seconds_per_call"] for r in clean)
        out[na] = {
            "cells": len(clean),
            "achieved_gb_per_s": total_bytes / total_seconds / 1e9,
            "per_cell_gb_per_s": sorted(r["achieved_gb_per_s"] for r in clean),
            "askeladd_gb_per_s": ASKELADD_BW.get(na),
        }
        here = out[na]["achieved_gb_per_s"]
        there = ASKELADD_BW.get(na)
        out[na]["ratio_here_over_askeladd"] = (here / there) if there else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung0", default="research/e63-artifacts/rung0.json")
    ap.add_argument("--rung1", default="research/e63-artifacts/e63-cost-curve.json")
    ap.add_argument("--log", default="research/e63-artifacts/e63-rung1.log")
    args = ap.parse_args()

    rung0 = json.loads(pathlib.Path(args.rung0).read_text())
    rung1_path = pathlib.Path(args.rung1)
    rung1 = json.loads(rung1_path.read_text()) if rung1_path.exists() else None
    log_path = pathlib.Path(args.log)
    log = log_path.read_text() if log_path.exists() else ""

    kill = rung0["kill_rule"]
    mc = rung0["model_comparison"]
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="kernel-census",
        name="e63-qmv-width-cliff-mlp",
        tags=["e63", "qmv", "width-cliff", "rung0-kill", "compile-only",
              "qwen-edward"],
        config={
            "experiment": "e63-qmv-width-cliff-memory-level-parallelism",
            "hypothesis": "the QMV width cliff is a memory-level-parallelism "
                          "collapse, not an occupancy collapse",
            "base_sha": "d67d8d194d3495cdac1261082029f078da342deb",
            "head_sha": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "pr": 66,
            "student": "qwen-edward",
            "host_chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True).stdout.strip(),
            "metal_flags": rung0["flags"],
            "air_pipeline": rung0["pipeline"],
            "askeladd_peak_gb_per_s": ASKELADD_PEAK_GB_S,
            "ranked_width_share": RANKED_WIDTH_SHARE,
            "local_width_share": LOCAL_WIDTH_SHARE,
            # No timed candidate leg exists, so no thermal gate was engaged.
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
        },
    )

    census = wandb.Table(columns=[
        "na", "reg_law", "peak_live_regs", "allocas", "alloca_types",
        "max_total_threads_per_threadgroup", "device_loads_total",
        "weight_loads", "max_weight_loads_in_flight",
        "weight_loads_before_first_consumer", "loop_device_loads",
        "loop_private_loads", "loop_private_stores", "loop_fadd", "loop_fmul",
        "accumulator_in_private_memory"])
    for na_str, cell in sorted(rung0["cells"].items(), key=lambda kv: int(kv[0])):
        if "mlp" not in cell:
            continue
        na = int(na_str)
        m, b = cell["mlp"], cell["loop_body"]
        census.add_data(
            na, 22 + 20 * na, cell["peak_live_regs"], cell["allocas"],
            ", ".join(cell["alloca_types"]),
            cell.get("occupancy", {}).get("max_total_threads_per_threadgroup"),
            cell["device_loads_total"], m["weight_loads"],
            m["max_weight_loads_in_flight_arith"],
            m["weight_loads_before_first_consumer_arith"],
            b["device_loads"], b["private_loads"], b["private_stores"],
            b["fadd"], b["fmul"],
            any("float" in a for a in cell["alloca_types"]))

    models = wandb.Table(columns=["model", "free_parameters", "rms_rel",
                                  "max_abs_rel", "residual_signs",
                                  "coefficients"])
    for name, fit in mc["models"].items():
        if "rms_rel" not in fit:
            continue
        models.add_data(name, fit["free_parameters"], fit["rms_rel"],
                        fit["max_abs_rel"], "".join(fit["residual_sign"]),
                        json.dumps([round(c, 4) for c in fit["coefficients"]]))

    summary = {
        "rung0/kill_rule_fired": bool(kill["identical"]),
        "rung0/max_weight_loads_in_flight_na2": kill["max_weight_loads_in_flight_na2"],
        "rung0/max_weight_loads_in_flight_na6": kill["max_weight_loads_in_flight_na6"],
        "rung0/in_flight_ratio_na2_over_na6": kill["ratio"],
        "rung0/advance_requires_ratio_ge": kill["advance_requires_ratio_ge"],
        "rung0/entry_points_defined": len(
            rung0["entry"].get("defined_kernels", [])),
        "rung0/single_entry_point_confirmed": sorted(
            rung0["entry"].get("defined_kernels", [])) == sorted([
                "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0",
                "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_1"]),
        "rung0/accumulator_spill_first_na": next(
            (int(k) for k, v in sorted(rung0["accumulator_spill"].items(),
                                       key=lambda kv: int(kv[0]))
             if v["accumulator_in_private_memory"]), None),
    }
    for name, fit in mc["models"].items():
        if "rms_rel" in fit:
            summary["rung0/rms_rel/%s" % name] = fit["rms_rel"]

    spill = mc["models"].get("alu_work_plus_spill_step", {})
    if "coefficients" in spill:
        a, b, s = spill["coefficients"]
        summary.update({
            "rung0/fit_fixed_offset_ms": a,
            "rung0/fit_per_na_slope_ms": b,
            "rung0/fit_spill_step_ms": s,
        })
        t6 = 14.412 / ASKELADD_BW[6] * 1000.0
        summary["rung0/na6_spill_share_of_cell"] = s / t6
        summary["rung0/na6_bw_if_step_removed_gb_per_s"] = \
            14.412 / (t6 - s) * 1000.0
        summary["rung0/ranked_qmv_effect_if_step_removed"] = \
            -(s / t6) * RANKED_WIDTH_SHARE[6]

    logged = {"rung0/na_census": census, "rung0/model_comparison": models}

    if rung1 is not None:
        roof = rung1.get("roofline", {})
        rows = ladder_from_cost_curve(rung1)
        rungs = rung_summary(rows)
        ladder = wandb.Table(columns=[
            "shape", "k", "n", "m", "na", "weight_streams", "clean_single_na",
            "tail_na", "seconds_per_call", "seconds_per_call_min",
            "seconds_per_call_max", "achieved_gb_per_s",
            "row0_bitwise_matches_m1"])
        for r in rows:
            ladder.add_data(r["shape"], r["k"], r["n"], r["m"], r["na"],
                            r["weight_streams"], r["clean_single_na"],
                            r["tail_na"], r["seconds_per_call"],
                            r["seconds_per_call_min"], r["seconds_per_call_max"],
                            r["achieved_gb_per_s"],
                            r["row0_bitwise_matches_m1"])
        compare = wandb.Table(columns=["na", "here_gb_per_s",
                                       "askeladd_gb_per_s", "ratio", "cells"])
        for na, v in sorted(rungs.items()):
            compare.add_data(na, v["achieved_gb_per_s"], v["askeladd_gb_per_s"],
                             v["ratio_here_over_askeladd"], v["cells"])
        logged["rung1/ladder"] = ladder
        logged["rung1/ladder_vs_askeladd"] = compare
        peak = roof.get("peak_bandwidth_bytes_per_second")

        # Per-shape decline across the clean rungs. The rung aggregate hides a
        # 2x spread in fraction of peak between shapes at one NA, so report the
        # shapes separately and check that the NA2 -> NA4 decline replicates in
        # each of them rather than only in the byte-weighted total.
        per_shape = wandb.Table(columns=[
            "shape", "k", "n", "gb_per_s_na2", "gb_per_s_na3", "gb_per_s_na4",
            "fraction_of_local_peak_na2", "fraction_of_local_peak_na4",
            "decline_na4_over_na2"])
        by_shape: dict[str, dict[int, float]] = {}
        meta: dict[str, tuple[int, int]] = {}
        for r in rows:
            if not r["clean_single_na"]:
                continue
            by_shape.setdefault(r["shape"], {})[r["na"]] = r["achieved_gb_per_s"]
            meta[r["shape"]] = (r["k"], r["n"])
        peak_gb = (peak / 1e9) if peak else None
        for shape in sorted(by_shape, key=lambda s: -by_shape[s].get(4, 0.0)):
            v = by_shape[shape]
            k, n = meta[shape]
            b2, b4 = v.get(2), v.get(4)
            per_shape.add_data(
                shape, k, n, b2, v.get(3), b4,
                (b2 / peak_gb) if (b2 and peak_gb) else None,
                (b4 / peak_gb) if (b4 and peak_gb) else None,
                (b4 / b2) if (b2 and b4) else None)
        logged["rung1/per_shape_decline"] = per_shape
        declines = [v[4] / v[2] for v in by_shape.values() if 2 in v and 4 in v]
        if declines:
            summary.update({
                "rung1/decline_na4_over_na2_min": min(declines),
                "rung1/decline_na4_over_na2_max": max(declines),
                "rung1/decline_na4_over_na2_shapes": len(declines),
            })
        if peak_gb:
            fr4 = [v[4] / peak_gb for v in by_shape.values() if 4 in v]
            if fr4:
                summary.update({
                    "rung1/fraction_of_local_peak_na4_min": min(fr4),
                    "rung1/fraction_of_local_peak_na4_max": max(fr4),
                })

        # Declared null: the widest rep-to-rep spread inside one unchanged cell.
        # Any claimed effect must exceed this to be readable on this host.
        spreads = [(r["seconds_per_call_max"] - r["seconds_per_call_min"])
                   / r["seconds_per_call"] for r in rows
                   if r["seconds_per_call"] > 0]
        if spreads:
            spreads.sort()
            summary.update({
                "rung1/same_cell_rel_spread_max": spreads[-1],
                "rung1/same_cell_rel_spread_p50": spreads[len(spreads) // 2],
                "rung1/same_cell_rel_spread_cells": len(spreads),
            })
        summary.update({
            "rung1/peak_bandwidth_gb_per_s": (peak / 1e9) if peak else None,
            "rung1/askeladd_peak_gb_per_s": ASKELADD_PEAK_GB_S,
            "rung1/peak_flops_tflop_per_s":
                (roof["peak_flops_per_second"] / 1e12)
                if roof.get("peak_flops_per_second") else None,
            "rung1/device": rung1.get("device"),
            "rung1/reps": rung1.get("reps"),
            "rung1/inner_calls_per_rep": rung1.get("inner_calls_per_rep"),
            "rung1/clean_rungs_measured": sorted(rungs),
        })
        for na, v in rungs.items():
            summary["rung1/achieved_gb_per_s/na%d" % na] = v["achieved_gb_per_s"]
            if v["ratio_here_over_askeladd"]:
                summary["rung1/ratio_vs_askeladd/na%d" % na] = \
                    v["ratio_here_over_askeladd"]
        if peak:
            for na, v in rungs.items():
                summary["rung1/fraction_of_local_peak/na%d" % na] = \
                    v["achieved_gb_per_s"] / (peak / 1e9)

    for line in log.splitlines():
        if "entry_thermal" in line:
            summary["rung1/thermal_entry"] = line.split("entry_thermal", 1)[1].strip()
        if "exit_thermal" in line:
            summary["rung1/thermal_exit"] = line.split("exit_thermal", 1)[1].strip()

    run.log(logged)
    run.summary.update(summary)
    for path in (args.rung0, args.rung1, args.log):
        if pathlib.Path(path).exists():
            run.save(path, policy="now")
    print("run_id", run.id)
    print("url", run.url)
    run.finish()


if __name__ == "__main__":
    main()
