#!/usr/bin/env python3
"""Publish the E104 x-group rate experiment to W&B.

    usage: research/e104_wandb_log.py [--only RUN]

Six runs:

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
  `e104-partition-ladder`
                    rung 0.5: isolated one-group versus split dispatch at
                    NA=2..8 over five shapes, with the NA=2 and NA=3 null
                    controls. Gives `A_local` per width against the
                    ranked-neutral threshold, plus registers and spill on
                    both target architectures.
  `e104-arithmetic-arms`
                    rung 2: four arithmetic variants that cut or reorder the
                    floating-point work at fixed memory traffic. Tests whether
                    floating-point issue is the binding constraint, and
                    compares AIR operation counts with compiled ISA text size.

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

# Advisor break-even law (PR #106): A_ranked = A_local * 1.244, and collapsing
# a split dispatch is ranked-neutral at A_ranked == 2.
LOCAL_TO_RANKED_A = 1.244
RANKED_NEUTRAL_A = 2.0 / LOCAL_TO_RANKED_A
FP_OPS_PER_NA = {"a_base": 160, "n_nosums": 148, "xf_exactfma": 112,
                 "f_fmamax": 88, "s_splitacc": 160}


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


def _ladder_cells(name: str):
    d = OUT / name
    doc = json.loads((d / "rate.json").read_text())
    return d, doc, load_cells(doc), read_meta(d / "meta.txt")


def log_partition_ladder() -> None:
    d, doc, cells, meta = _ladder_cells("e104-r05-ladder")
    arms = doc["arms"]
    one, split = arms[0], arms[1]
    spec = json.loads((d / "census.json").read_text())["partitions"][split]
    run = start(
        "e104-partition-ladder", "partition-ladder",
        "At which activation width does collapsing a split dispatch into one "
        "wide group stop paying, once the local-to-ranked transfer is applied?",
        0.5,
        {"shapes": 5, "widths": "2..8", "arms": arms,
         "order": doc["order"], "blocks_per_cell": doc["pairs"],
         "warmup_blocks_discarded": WARMUP_BLOCKS,
         "local_to_ranked_a": LOCAL_TO_RANKED_A,
         "ranked_neutral_a": RANKED_NEUTRAL_A,
         "splits": {k: v["partition"] for k, v in spec.items()},
         "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
         "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
         "arm_sha256": {a: meta.get(f"arm_{a}_sha256") for a in arms}},
    )
    per_cell = wandb.Table(columns=[
        "shape", "na", "split", "t1_us", "t2_us", "r1_gbs", "r2_gbs",
        "a_local"])
    ladder = wandb.Table(columns=[
        "na", "split", "a_local_median", "a_local_min", "a_local_max",
        "a_ranked", "ranked_gain_pct", "verdict"])
    by_na: dict[int, list[float]] = {}
    for (shape, m), c in sorted(cells.items()):
        t1 = statistics.median(c["seconds"][one])
        t2 = statistics.median(c["seconds"][split])
        # `read_bytes` prices one weight stream plus the activations; a G-way
        # split rereads the weights G times but the activations only once.
        weights = c["flops"] / (2.0 * m) * (0.5 + 4.0 / 64.0)
        acts = c["bytes"] - weights
        r1 = c["bytes"] / t1 / 1e9
        r2 = (spec[str(m)]["weight_streams"] * weights + acts) / t2 / 1e9
        a = r2 / r1
        by_na.setdefault(m, []).append(a)
        per_cell.add_data(shape, m, spec[str(m)]["partition"], t1 * 1e6,
                          t2 * 1e6, r1, r2, a)
    for na in sorted(by_na):
        vals = by_na[na]
        med = statistics.median(vals)
        ranked = med * LOCAL_TO_RANKED_A
        gain = 100.0 * (1.0 - ranked / 2.0)
        verdict = ("null control" if abs(med - 1.0) < 0.02
                   else "collapse wins" if med < RANKED_NEUTRAL_A
                   else "collapse loses")
        ladder.add_data(na, spec[str(na)]["partition"], med, min(vals),
                        max(vals), ranked, gain, verdict)
        run.log({"na": na, "a_local": med, "a_ranked": ranked,
                 "ranked_gain_pct": gain})
    run.log({"per_cell": per_cell, "a_ladder": ladder})
    med5 = statistics.median(by_na[5])
    med6 = statistics.median(by_na[6])
    run.summary.update({
        "null_control_a_na2": statistics.median(by_na[2]),
        "null_control_a_na3": statistics.median(by_na[3]),
        "instrument_noise_floor_pct": 0.2,
        "a_local_na5": med5,
        "a_local_na6": med6,
        "ranked_gain_pct_na5": 100.0 * (1.0 - med5 * LOCAL_TO_RANKED_A / 2.0),
        "ranked_gain_pct_na6": 100.0 * (1.0 - med6 * LOCAL_TO_RANKED_A / 2.0),
        "last_width_where_collapse_pays": 5,
        "collapse_recommended_at_m6": False,
        "exactness_all_cells_bit_identical": True,
        "g16s_register_budget": 96,
        "g17s_register_budget": 124,
        "g16s_spills_from_na": 6,
        "g17s_spills_from_na": 8,
    })
    attach(run, d / "rate.json", d / "meta.txt", d / "census.json",
           d / "analysis.txt")
    run.finish()


def log_arithmetic_arms() -> None:
    d, doc, cells, meta = _ladder_cells("e104-r2-arith")
    arms = doc["arms"]
    census = json.loads((d / "census.json").read_text())
    regs = {r["arm"]: r for r in census["arms"]}
    fidelity: dict[str, list[bool]] = {}
    for row in doc["measurements"]:
        if row["kind"] != "fidelity":
            continue
        for a in row["arms"]:
            fidelity.setdefault(a["arm"], []).append(a["bit_identical"])
    run = start(
        "e104-arithmetic-arms", "arithmetic-arms",
        "Is floating-point instruction issue the binding constraint that makes "
        "a wide x-group stream slowly at NA=5?",
        2,
        {"shapes": 5, "widths": "2..8", "arms": arms,
         "order": doc["order"], "blocks_per_cell": doc["pairs"],
         "warmup_blocks_discarded": WARMUP_BLOCKS,
         "air_fp_ops_per_na": FP_OPS_PER_NA,
         "exact_required_arms": ["xf_exactfma"],
         "promotion_bar_rate_lift_pct": 10.0,
         "closure_bar_na5_move_pct": 3.0,
         "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
         "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
         "arm_sha256": {a: meta.get(f"arm_{a}_sha256") for a in arms}},
    )
    per_cell = wandb.Table(columns=["shape", "na", "arm", "us", "gbs",
                                    "change_pct_vs_base"])
    codegen = wandb.Table(columns=[
        "arm", "na", "air_fp_ops_per_kblock", "g16s_regs", "g16s_spill_b",
        "g16s_text_b", "g16s_text_change_pct", "g17s_regs", "g17s_spill_b"])
    by_na_arm: dict[tuple[int, str], list[float]] = {}
    for (shape, m), c in sorted(cells.items()):
        base = statistics.median(c["seconds"]["a_base"])
        for arm in arms:
            sec = statistics.median(c["seconds"][arm])
            chg = 100.0 * (sec - base) / base
            by_na_arm.setdefault((m, arm), []).append(chg)
            per_cell.add_data(shape, m, arm, sec * 1e6,
                              c["bytes"] / sec / 1e9, chg)
    for arm in arms:
        for na in range(2, 9):
            g16 = regs[arm]["applegpu_g16s"][str(na)]
            g17 = regs[arm]["applegpu_g17s"][str(na)]
            base_text = regs["a_base"]["applegpu_g16s"][str(na)]["text_bytes"]
            codegen.add_data(
                arm, na, FP_OPS_PER_NA[arm] * na, g16["registers"],
                g16["spill_bytes"], g16["text_bytes"],
                100.0 * (g16["text_bytes"] / base_text - 1.0),
                g17["registers"], g17["spill_bytes"])
    run.log({"per_cell": per_cell, "codegen": codegen})
    for na in range(2, 9):
        run.log({"na": na, **{f"{a}/change_pct": statistics.median(
            by_na_arm[(na, a)]) for a in arms if a != "a_base"}})

    exact5 = statistics.median(by_na_arm[(5, "xf_exactfma")])
    nosums5 = statistics.median(by_na_arm[(5, "n_nosums")])
    run.summary.update({
        "xf_exactfma_bit_identical_cells":
            f"{sum(fidelity['xf_exactfma'])}/{len(fidelity['xf_exactfma'])}",
        "xf_exactfma_air_fp_ops_cut_pct": -30.0,
        "xf_exactfma_g16s_text_change_pct_na5": 100.0 * (
            regs["xf_exactfma"]["applegpu_g16s"]["5"]["text_bytes"]
            / regs["a_base"]["applegpu_g16s"]["5"]["text_bytes"] - 1.0),
        "xf_exactfma_change_pct_na5": exact5,
        "n_nosums_change_pct_na5": nosums5,
        "f_fmamax_change_pct_na5": statistics.median(
            by_na_arm[(5, "f_fmamax")]),
        "f_fmamax_g16s_spill_b_na5":
            regs["f_fmamax"]["applegpu_g16s"]["5"]["spill_bytes"],
        "s_splitacc_change_pct_na5": statistics.median(
            by_na_arm[(5, "s_splitacc")]),
        "s_splitacc_g16s_spill_b_na5":
            regs["s_splitacc"]["applegpu_g16s"]["5"]["spill_bytes"],
        "h4_fp_issue_saturation_refuted": True,
        "h5_split_accumulators_refuted": True,
        "air_op_count_predicts_time": False,
        "isa_text_bytes_predicts_time": True,
        "backend_contracts_fma_by_default": True,
        "promotion_bar_met": False,
        "closure_bar_met_for_exact_arms": abs(exact5) <= 3.0,
        "rate_na_axis_recommended_closed": True,
        "n_nosums_is_legal_candidate": False,
        "n_nosums_reason_illegal": "drops the affine bias-sum term; sums is "
                                   "already hoisted out of the row loop, so "
                                   "there is no bit-exact route to this win",
    })
    attach(run, d / "rate.json", d / "meta.txt", d / "census.json",
           d / "analysis.txt", d / "air-census.json")
    run.finish()


RUNS = {"census": log_census, "rate-sweep": log_rate_sweep,
        "occupancy": log_occupancy, "controls": log_controls,
        "partition-ladder": log_partition_ladder,
        "arithmetic-arms": log_arithmetic_arms}


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
