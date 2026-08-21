#!/usr/bin/env python3
"""Publish the E104 x-group rate experiment to W&B.

    usage: research/e104_wandb_log.py [--only RUN]

Four runs:

  `e104-census`     rung 0: the per-NA AIR census of the shipped kernel and of
                    the three variant arms. Device loads, nibble unpack, float
                    lanes, peak live registers and allocas per k-block.
  `e104-rate-sweep` rung 1: the four-arm counterbalanced sweep over five scored
                    shapes and NA=2..6. The rate(NA) curve, the roofline
                    overhead factor, the two-regime model fit and the
                    `xw_widex` verdict.
  `e104-occupancy`  the runtime occupancy read: maxTotalThreadsPerThreadgroup
                    per arm per NA.
  `e104-controls`   the two negative controls that withdrew the FMA-fusion
                    explanation, plus the cold-start slot bias correction.

Every leg here ran with the local cool gate off, so `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are logged false
verbatim. No number here is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e104-why-a-wide-x-group-streams-slowly"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
OUT = pathlib.Path("research/out")

BASE_SHA = "5c2c3b8b613841d0d9677d4540e6a08e8bd40759"
UPSTREAM_NOTE = "senpai/qwen38-mtp-r1"

DRAM_PEAK_GBS = 273.0
ARMS = ("a_base", "l_loadonly", "z_noxload", "xw_widex")
WARMUP_BLOCKS = 1


def gate_flags(qualified: bool = False) -> dict[str, object]:
    return {
        "timing_valid": qualified,
        "cool_gate_passed_real_gate": qualified,
        "gate_qualified_for_timing": qualified,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def start(name: str, job_type: str, question: str, rung: int,
          config: dict, qualified: bool = False):
    meta = read_meta(OUT / "e104-r1-sweep" / "meta.txt")
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={
            "experiment": GROUP, "rung": rung, "question": question,
            "host": HOST, "hostname": meta.get("host"),
            "chip": meta.get("chip"), "toolchain": meta.get("toolchain"),
            "metal_toolchain": meta.get("metal_toolchain"),
            "base_sha": BASE_SHA, "advisor_branch": UPSTREAM_NOTE,
            "shipped_source_touched": False,
            **config, **gate_flags(qualified),
        },
        reinit=True,
    )


def attach(run, *paths: pathlib.Path) -> None:
    present = [p for p in paths if p.exists()]
    if not present:
        return
    artifact = wandb.Artifact(f"{run.name}-artifacts", type="analysis")
    for path in present:
        artifact.add_file(str(path))
    run.log_artifact(artifact)


def load_cells(doc: dict) -> dict:
    cells: dict[tuple[str, int], dict] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        key = (row["shape"], row["m"])
        cell = cells.setdefault(key, {"bytes": row["read_bytes"],
                                      "flops": row["flops"],
                                      "seconds": {}})
        for arm, sec in row["seconds"].items():
            cell["seconds"].setdefault(arm, []).append(sec)
    return cells


def log_census() -> None:
    path = OUT / "e104-r0" / "census.json"
    doc = json.loads(path.read_text())
    run = start(
        "e104-census", "air-census",
        "How does the shipped kernel's per-k-block instruction mix scale with "
        "the number of activation rows?",
        0,
        {"tool": "research/e104_arm_census.py",
         "note": "counts are weighted by loop trip count per k-block"},
    )
    table = wandb.Table(columns=[
        "arm", "na", "device_loads", "nibble_and", "nibble_lshr",
        "fmul_lanes", "fadd_lanes", "fpext_lanes", "peak_live_regs",
        "allocas"])
    for arm, widths in doc["arms"].items():
        for na in sorted(widths, key=int):
            cell = widths[na]
            if cell.get("status") != "ok":
                continue
            pk = cell["per_kblock"]
            table.add_data(arm, int(na), pk["device_loads"], pk["nibble_and"],
                           pk["nibble_lshr"], pk["fmul_lanes"],
                           pk["fadd_lanes"], pk.get("fpext_lanes"),
                           cell["peak_live_regs"], cell["allocas"])
            if arm == "a_base":
                run.log({"na": int(na),
                         "a_base/device_loads": pk["device_loads"],
                         "a_base/nibble_and": pk["nibble_and"],
                         "a_base/fmul_lanes": pk["fmul_lanes"],
                         "a_base/fadd_lanes": pk["fadd_lanes"],
                         "a_base/peak_live_regs": cell["peak_live_regs"],
                         "a_base/allocas": cell["allocas"]})
    run.log({"census": table})
    run.summary.update({
        "h3_dequant_repeats_refuted": True,
        "nibble_unpack_flat_in_na": True,
        "device_loads_model": "24 + 16*NA",
        "xw_widex_device_loads_model": "24 + 2*NA",
        "activation_load_share_at_na5": 80 / 104,
    })
    attach(run, path, OUT / "e104-r0" / "census.log")
    run.finish()


def log_rate_sweep() -> None:
    d = OUT / "e104-r1-sweep"
    doc = json.loads((d / "rate.json").read_text())
    cells = load_cells(doc)
    meta = read_meta(d / "meta.txt")
    run = start(
        "e104-rate-sweep", "kernel-rate-probe",
        "Why does a lone wide x-group achieve a lower streaming rate at NA=5 "
        "than at NA=2 when it reads the same bytes?",
        1,
        {"shapes": 5, "widths": "2..6", "arms": list(ARMS),
         "order": doc["order"], "blocks_per_cell": doc["pairs"],
         "warmup_blocks_discarded": WARMUP_BLOCKS,
         "dram_peak_gbs": DRAM_PEAK_GBS,
         "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
         "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
         "arm_sha256": {a: meta.get(f"arm_{a}_sha256") for a in ARMS}},
    )

    table = wandb.Table(columns=[
        "shape", "na", "arm", "us", "gbs", "tflops", "dram_floor_us",
        "overhead_factor"])
    xw = wandb.Table(columns=["shape", "na", "base_us", "xw_us", "change_pct",
                              "base_overhead", "xw_overhead"])
    for (shape, m), c in sorted(cells.items()):
        dram_floor = c["bytes"] / (DRAM_PEAK_GBS * 1e9)
        per_arm = {}
        for arm in ARMS:
            if arm not in c["seconds"]:
                continue
            sec = statistics.median(c["seconds"][arm])
            per_arm[arm] = sec
            table.add_data(shape, m, arm, sec * 1e6,
                           c["bytes"] / sec / 1e9, c["flops"] / sec / 1e12,
                           dram_floor * 1e6, sec / dram_floor)
        if "a_base" in per_arm and "xw_widex" in per_arm:
            b, x = per_arm["a_base"], per_arm["xw_widex"]
            xw.add_data(shape, m, b * 1e6, x * 1e6, 100.0 * (x - b) / b,
                        b / dram_floor, x / dram_floor)
        run.log({"na": m, f"{shape}/a_base_us": per_arm["a_base"] * 1e6,
                 f"{shape}/l_loadonly_us": per_arm["l_loadonly"] * 1e6,
                 f"{shape}/a_base_tflops": c["flops"] / per_arm["a_base"] / 1e12,
                 f"{shape}/a_base_overhead": per_arm["a_base"] / dram_floor})
    run.log({"per_cell": table, "xw_widex_vs_base": xw})

    # The headline shape is the only one that clearly exceeds the SLC.
    head = "mlp_gate_up_k5120_n34816"
    base5 = statistics.median(cells[(head, 5)]["seconds"]["a_base"])
    xw5 = statistics.median(cells[(head, 5)]["seconds"]["xw_widex"])
    floor5 = cells[(head, 5)]["bytes"] / (DRAM_PEAK_GBS * 1e9)
    lo2 = statistics.median(cells[(head, 2)]["seconds"]["l_loadonly"])
    lo6 = statistics.median(cells[(head, 6)]["seconds"]["l_loadonly"])
    b2 = statistics.median(cells[(head, 2)]["seconds"]["a_base"])
    b6 = statistics.median(cells[(head, 6)]["seconds"]["a_base"])
    run.summary.update({
        "headline_shape": head,
        "h4_load_stream_refuted": True,
        "l_loadonly_growth_na2_to_na6": lo6 / lo2,
        "a_base_growth_na2_to_na6": b6 / b2,
        "na5_overhead_base": base5 / floor5,
        "na5_overhead_xw_widex": xw5 / floor5,
        "na5_overhead_promotion_bar": 1.5,
        "promotion_bar_met": bool(xw5 / floor5 < 1.5),
        "xw_widex_change_pct_na5": 100.0 * (xw5 - base5) / base5,
        "integrated": False,
        "exactness_all_cells_bit_identical": True,
    })
    attach(run, d / "rate.json", d / "meta.txt", d / "probe.log",
           d / "census.json")
    run.finish()


def log_occupancy() -> None:
    path = OUT / "e104-r1-sweep" / "occupancy.json"
    doc = json.loads(path.read_text())
    run = start(
        "e104-occupancy", "occupancy-probe",
        "Does register pressure cap the launchable threadgroup as NA grows?",
        2,
        {"tool": "research/e104_occupancy_probe.m",
         "device_max_threads_per_threadgroup":
             doc["device_max_threads_per_threadgroup"]},
    )
    table = wandb.Table(columns=["arm", "na", "max_total_threads",
                                 "thread_execution_width", "tg_memory_bytes"])
    caps = set()
    for row in doc["rows"]:
        table.add_data(row["arm"], row["na"],
                       row["max_total_threads_per_threadgroup"],
                       row["thread_execution_width"],
                       row["static_threadgroup_memory_bytes"])
        caps.add(row["max_total_threads_per_threadgroup"])
    run.log({"occupancy": table})
    run.summary.update({
        "h1_hard_occupancy_cliff": False,
        "max_total_threads_distinct_values": sorted(caps),
        "note": "Metal exposes the launch cap, not concurrent residency, so "
                "this rules out a hard cliff only.",
    })
    attach(run, path)
    run.finish()


def log_controls() -> None:
    run = start(
        "e104-controls", "negative-controls",
        "Do the controls support the FMA-fusion explanation for the "
        "arithmetic ceiling?",
        2,
        {"tool": "research/e104_fma_ceiling.m + -ffast-math codegen diff"},
    )
    codegen = wandb.Table(columns=["build", "llvm_fma", "fmul", "fadd"])
    codegen.add_data("-O2", 0, 207, 260)
    codegen.add_data("-O2 -ffast-math", 0, 207, 260)

    scaling = wandb.Table(columns=["form", "trip_count_ratio",
                                   "time_ratio", "expected"])
    scaling.add_data("ceiling_mul_add", 8.0, 2.17, 8.0)
    scaling.add_data("ceiling_fma", 8.0, 2.18, 8.0)

    slot = wandb.Table(columns=["blocks", "slot0_over_mirror_median",
                                "slot0_over_mirror_max", "n"])
    slot.add_data("dropped (block 0)", 2.598, 3.441, 25)
    slot.add_data("kept (blocks 1..3)", 0.999, 1.303, 75)

    run.log({"fast_math_codegen": codegen,
             "ceiling_scaling_check": scaling,
             "cold_start_slot_bias": slot})
    run.summary.update({
        "fma_fusion_explanation_withdrawn": True,
        "fast_math_changes_air": False,
        "ceiling_harness_valid": False,
        "ceiling_harness_failed_closed": True,
        "machine_arithmetic_peak_quoted": False,
        "arithmetic_ceiling_cause": "unresolved",
        "cold_start_correction_applied": True,
    })
    attach(run, pathlib.Path("research/e104_fma_ceiling.m"))
    run.finish()


RUNS = {"census": log_census, "rate-sweep": log_rate_sweep,
        "occupancy": log_occupancy, "controls": log_controls}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS))
    args = ap.parse_args()
    for name, fn in RUNS.items():
        if args.only and name != args.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
