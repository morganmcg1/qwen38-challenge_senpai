#!/usr/bin/env python3
"""Publish the E103 FA-history SDPA evidence to W&B.

    usage: research/e103_wandb_log.py [--only RUN]

  `e103-rung0-census`   the in-situ dispatch census: which kernel the scored
                        decode actually reaches, its verbatim signature, grid,
                        threadgroup shape, dispatch count and exclusive GPU
                        time per dispatch, plus the editable-versus-trusted
                        split of the files that contain it.
  `e103-rung1-registers`
                        the static budget: registers, spill bytes, text bytes
                        and text digest for eleven arms of the scored kernel,
                        cross-compiled for the local `applegpu_g16s` and the
                        ranked `applegpu_g17s`.
  `e103-rung2-isolated` the isolated dose: one process, one queue, palindrome
                        ordering, twenty cells over N in {512, 576, 768, 1024}
                        and M in {1..5}, ten arms per cell, every arm compared
                        bit for bit against the shipped transcription.

Rung 0 is an instrumented census leg, never a timing leg. Rung 1 is static
toolchain output. Rung 2 is a standalone Metal microbenchmark that holds no
model and runs no benchmark wrapper, so it passes no thermal gate. Every run
therefore logs `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` verbatim, and no leg here is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e103-sdpa-fa-history-head-packing"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out/e103")

BASE_SHA = "97511edbd452e6d1f4fbe1c3d03297805bbf5020"
SUBMIT_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
FRONTIER = "f04b102e"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"

# The scored cell, verbatim from the census leg.
SCORED_KERNEL_M5 = "sdpa_vector_bfloat16_t_256_256_nomask_qnt_c_nosinks grid=24x5x1 tg=1024x1x1"
SCORED_KERNEL_M1 = "sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks grid=24x1x1 tg=1024x1x1"
METALLIB_FINGERPRINT = (
    "mlxfast-metallib-fingerprint-v1 "
    "7ae5c5a3d8fabe72ee19bfc09dd737281338a6be658deca49ba97eefdbe3611c"
)

# Assignment thresholds, PR #105.
LOCAL_ROUND_US = 127176.0  # measured round GPU busy at width 5, census leg
MIN_USEFUL_US_PER_ROUND = 383.0  # 0.30 % of the local round
LATENCY_CLASS_FACTOR = 2.40
INSITU_DISCOUNT = (1.65, 2.59)
PUBLISHED_FLOOR_PCT = 0.277
SERIAL_FREE_FLOOR_PCT = 0.160

# Round composition at verify width 5, from the census leg.
VERIFY_DISPATCHES = 16
DRAFT_HEAD_DISPATCHES = 4

ARMS = [
    "a_shipped_c",
    "b_vecload_c",
    "c_fastpath_c",
    "d_pack1_c",
    "d_pack2_c",
    "d_pack3_c",
    "d_pack6_c",
    "e_resident_c",
    "f_nosoftmax_c",
    "g_double_c",
]

ARM_ROLE = {
    "a_shipped_c": "shipped transcription, the reference for every comparison",
    "b_vecload_c": "4-wide K and V loads, arithmetic order unchanged",
    "c_fastpath_c": "b plus a skipped rescale when the running max does not move",
    "d_pack1_c": "pack kernel at P=1, an identity check on the pack rewrite",
    "d_pack2_c": "two query heads per threadgroup",
    "d_pack3_c": "three query heads per threadgroup",
    "d_pack6_c": "six query heads per threadgroup, the full GQA group",
    "e_resident_c": "traffic-free control: K and V pointers never advance",
    "f_nosoftmax_c": "softmax-free control: traffic kept, online softmax removed",
    "g_double_c": "positive control: the key loop runs twice",
}

ARM_PACK = {a: (2 if a == "d_pack2_c" else 3 if a == "d_pack3_c"
                else 6 if a == "d_pack6_c" else 1) for a in ARMS}


def gate_flags(instrument: str, gpu_seconds: float) -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "gpu_seconds_used": gpu_seconds,
        "instrument": instrument,
    }


def identity() -> dict[str, object]:
    return {
        "experiment": GROUP,
        "pr": 105,
        "base_sha": BASE_SHA,
        "submit_base_sha": SUBMIT_BASE_SHA,
        "frontier_submission": FRONTIER,
        "host": HOST,
        "instance": "ip-10-231-2-227.ec2.internal",
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "scored_kernel_m5": SCORED_KERNEL_M5,
        "scored_kernel_m1": SCORED_KERNEL_M1,
        "metallib_fingerprint_unmodified": METALLIB_FINGERPRINT,
        "kernel_source_form": "AOT into mlx.metallib, no mlx-generated twin",
        "dispatcher_file": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/"
            "scaled_dot_product_attention.cpp"
        ),
        "dispatcher_editable": False,
        "kernel_file": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "sdpa_vector.h"
        ),
        "kernel_editable": True,
    }


def log_census() -> None:
    payload = json.loads((OUT / "census_costs.json").read_text())
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="dispatch-census",
        name="e103-rung0-census",
        config={
            "rung": "0",
            "question": (
                "which kernel does the scored decode dispatch over the "
                "full-attention history, and what does it cost"
            ),
            "leg": "research/out/e103r0-d4-ops0",
            "leg_command": "research/e96_census_leg.sh e103r0-d4-ops0 4 64 0",
            "buffer_limit_ops": 0,
            "buffer_limit_mb": 1,
            "forced_draft_depth": 4,
            "decode_tokens": 64,
            "steel_attention_selected": False,
            "sdpa_vector_2pass_selected": False,
            **identity(),
            **gate_flags("E58 one-dispatch-per-buffer census, GPU", 151.0),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=[
            "width_phase_shape",
            "rounds",
            "dispatches",
            "dispatches_per_round",
            "us_per_dispatch",
            "us_per_round",
            "min_us",
            "max_us",
        ]
    )
    flat: dict[str, object] = {}
    for key, v in sorted(payload["sdpa"].items()):
        table.add_data(
            key,
            v["rounds"],
            v["dispatches"],
            v["dispatches_per_round"],
            v["us_per_dispatch"],
            v["us_per_round"],
            v["min_us"],
            v["max_us"],
        )
        short = key.split("|")[0] + "_" + key.split("|")[1]
        if "grid=24x5" in key or "grid=24x1" in key:
            flat[f"census/{short}/{key.split('grid=')[1].split(' ')[0]}/us_per_dispatch"] = v["us_per_dispatch"]
    run.log({"census/sdpa_dispatches": table})

    wp = wandb.Table(columns=["width_phase", "rounds", "gpu_us_per_round"])
    for key, v in sorted(payload["width_phase"].items()):
        wp.add_data(key, v["rounds"], v["gpu_us_per_round"])
    run.log({"census/width_phase_gpu_busy": wp})

    verify = payload["sdpa"][f"w5|target_verify|{SCORED_KERNEL_M5}"]
    draft = payload["sdpa"][f"w5|draft_head|{SCORED_KERNEL_M1}"]
    fwd = payload["sdpa"][f"w1|target_forward|{SCORED_KERNEL_M1}"]
    round_busy = (
        payload["width_phase"]["w5|target_verify"]["gpu_us_per_round"]
        + payload["width_phase"]["w5|draft_head"]["gpu_us_per_round"]
    )
    sdpa_round = verify["us_per_round"] + draft["us_per_round"]

    # Cost as a function of verify row count, at the same history length.
    m4 = payload["sdpa"][
        "w4|target_verify|sdpa_vector_bfloat16_t_256_256_nomask_qnt_c_nosinks"
        " grid=24x4x1 tg=1024x1x1"
    ]
    slope = (verify["us_per_dispatch"] - fwd["us_per_dispatch"]) / 4.0
    intercept = verify["us_per_dispatch"] - 5.0 * slope

    run.summary.update(
        {
            "census/verify_us_per_dispatch": verify["us_per_dispatch"],
            "census/verify_us_per_round": verify["us_per_round"],
            "census/draft_head_us_per_dispatch": draft["us_per_dispatch"],
            "census/target_forward_us_per_dispatch": fwd["us_per_dispatch"],
            "census/m4_verify_us_per_dispatch": m4["us_per_dispatch"],
            "census/round_gpu_busy_us": round_busy,
            "census/sdpa_us_per_round": sdpa_round,
            "census/sdpa_share_of_round_pct": 100.0 * sdpa_round / round_busy,
            "census/row_cost_slope_us": slope,
            "census/row_cost_intercept_us": intercept,
            "census/assignment_anchor_us_per_round": 1267.0,
            "census/assignment_anchor_us_per_dispatch": 79.19,
            "census/reproduces_anchor": True,
        }
    )
    run.finish()


def log_registers() -> None:
    payload = json.loads((OUT / "regs.json").read_text())
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="static-register-budget",
        name="e103-rung1-registers",
        config={
            "rung": "1",
            "question": (
                "what is the largest number of query heads per threadgroup "
                "that stays spill-free on the ranked g17s"
            ),
            "arms": len(payload[LOCAL_ARCH]),
            "g17s_register_ceiling": 126,
            "threadgroup_memory_bytes": 4352,
            "arm_source": "research/e103_sdpa_arms.metal",
            **identity(),
            **gate_flags("xcrun metal, metallib and metal-tt, CPU only", 0.0),
        },
        reinit=True,
    )
    table = wandb.Table(
        columns=[
            "arm",
            "role",
            "pack_heads_per_threadgroup",
            "g16s_registers",
            "g16s_spill_bytes",
            "g16s_text_bytes",
            "g17s_registers",
            "g17s_spill_bytes",
            "g17s_text_bytes",
            "g17s_text_sha8",
        ]
    )
    flat: dict[str, object] = {}
    for name in sorted(payload[RANKED_ARCH]):
        g16 = payload[LOCAL_ARCH][name]
        g17 = payload[RANKED_ARCH][name]
        table.add_data(
            name,
            ARM_ROLE.get(name, "shipped transcription, non-causal form"),
            ARM_PACK.get(name, 1),
            g16["registers"],
            g16["spill_bytes"],
            g16["text_bytes"],
            g17["registers"],
            g17["spill_bytes"],
            g17["text_bytes"],
            g17["text_sha8"],
        )
        flat[f"regs/{name}/g17s_registers"] = g17["registers"]
        flat[f"regs/{name}/g17s_spill_bytes"] = g17["spill_bytes"]
        flat[f"regs/{name}/g16s_registers"] = g16["registers"]
        flat[f"regs/{name}/g16s_spill_bytes"] = g16["spill_bytes"]
    run.log({"registers/arms": table})
    run.summary.update(flat)
    run.summary.update(
        {
            "regs/largest_spill_free_pack": 3,
            "regs/pack6_spill_bytes_g17s": payload[RANKED_ARCH]["d_pack6_c"][
                "spill_bytes"
            ],
            "regs/shipped_registers_g17s": payload[RANKED_ARCH]["a_shipped_c"][
                "registers"
            ],
        }
    )
    run.finish()


def _cell_medians(measurements: list[dict]) -> dict[tuple[int, int], dict[str, float]]:
    cells: dict[tuple[int, int], list[dict]] = {}
    for r in measurements:
        if r["kind"] != "timing":
            continue
        cells.setdefault((r["n"], r["m"]), []).append(r)
    return {
        key: {a: st.median(x["seconds"][a] for x in rows) for a in ARMS}
        for key, rows in cells.items()
    }


def _round_saving(med: dict[tuple[int, int], dict[str, float]], n: int,
                  arm: str) -> float:
    """Microseconds saved per width-5 round if `arm` replaced the shipped
    kernel at history length `n`: 16 verify dispatches at M=5 plus 4 draft-head
    dispatches at M=1."""
    d5 = (med[(n, 5)]["a_shipped_c"] - med[(n, 5)][arm]) * 1e6
    d1 = (med[(n, 1)]["a_shipped_c"] - med[(n, 1)][arm]) * 1e6
    return VERIFY_DISPATCHES * d5 + DRAFT_HEAD_DISPATCHES * d1


def _window_mean(values: dict[int, float]) -> float:
    """Trapezoidal mean over the ranked history window N in [512, 1024]."""
    ns = sorted(values)
    total = 0.0
    for lo, hi in zip(ns, ns[1:]):
        total += 0.5 * (values[lo] + values[hi]) * (hi - lo)
    return total / (ns[-1] - ns[0])


def log_isolated() -> None:
    payload = json.loads((OUT / "rung2.json").read_text())
    meas = payload["measurements"]
    med = _cell_medians(meas)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="isolated-kernel-ab",
        name="e103-rung2-isolated",
        config={
            "rung": "2",
            "question": (
                "how much of the 79 us FA-history SDPA dispatch is device "
                "traffic, how much is the online softmax, and how much can "
                "packing query heads into one threadgroup recover"
            ),
            "harness_source": "research/e103_sdpa_ab.m",
            "arm_source": "research/e103_sdpa_arms.metal",
            "order": "palindrome, ten arms forward then reversed",
            "blocks_per_cell": payload["pairs"],
            "kv_copies": payload["kv_copies"],
            "device": payload["device"],
            "architecture": payload["architecture"],
            "heads": payload["heads"],
            "kv_heads": payload["kv_heads"],
            "head_dim": payload["head_dim"],
            "gqa_factor": payload["heads"] // payload["kv_heads"],
            "redundancy_factor_m5": 5 * payload["heads"] // payload["kv_heads"],
            "min_useful_us_per_round": MIN_USEFUL_US_PER_ROUND,
            "local_round_us": LOCAL_ROUND_US,
            "latency_class_factor": LATENCY_CLASS_FACTOR,
            **identity(),
            **gate_flags(
                "standalone Metal microbenchmark, one process, one queue", 43.2
            ),
        },
        reinit=True,
    )

    timing = wandb.Table(
        columns=["n", "m", "arm", "pack", "us", "delta_vs_a_pct",
                 "logical_gb_s", "actual_gb_s", "actual_read_mb"]
    )
    for (n, m), arms in sorted(med.items()):
        a0 = arms["a_shipped_c"]
        logical = 2.0 * payload["kv_heads"] * n * payload["head_dim"] * 2.0
        for arm in ARMS:
            sec = arms[arm]
            actual = (payload["heads"] / ARM_PACK[arm]) * m * 2.0 * n * \
                payload["head_dim"] * 2.0
            timing.add_data(
                n, m, arm, ARM_PACK[arm], sec * 1e6,
                100.0 * (sec - a0) / a0,
                logical / sec / 1e9, actual / sec / 1e9, actual / 1e6,
            )
    run.log({"isolated/timing": timing})

    fid = wandb.Table(
        columns=["n", "m", "arm", "expect_bit_exact", "bit_identical",
                 "differing", "total", "worst_abs"]
    )
    violations = 0
    for r in meas:
        if r["kind"] != "fidelity":
            continue
        fid.add_data(r["n"], r["m"], r["arm"], r["expect_bit_exact"],
                     r["bit_identical"], r["differing"], r["total"],
                     r["worst_abs"])
        if r["expect_bit_exact"] and not r["bit_identical"]:
            violations += 1
    run.log({"isolated/fidelity": fid})

    ctrl = wandb.Table(columns=["n", "m", "kind", "value"])
    ref_max, pos_detected = [], []
    for r in meas:
        if r["kind"] == "reference":
            ctrl.add_data(r["n"], r["m"], "a_vs_double_max_abs", r["max_abs"])
            ctrl.add_data(r["n"], r["m"], "a_vs_double_rms_over_signal",
                          r["rms_over_signal"])
            ref_max.append(r["max_abs"])
        elif r["kind"] == "positive_control":
            ctrl.add_data(r["n"], r["m"], "perturbed_key_differing",
                          r["differing"])
            pos_detected.append(bool(r["detected"]))
    run.log({"isolated/controls": ctrl})

    # --- round-level and ranked pricing -----------------------------------
    lens = sorted({n for n, _ in med})
    pricing = wandb.Table(
        columns=["arm", "us_saved_per_round_n512", "us_saved_per_round_n576",
                 "us_saved_per_round_n768", "us_saved_per_round_n1024",
                 "us_saved_per_round_window_mean", "local_pct_of_round",
                 "fraction_of_min_useful_effect", "ranked_pct_undiscounted",
                 "ranked_pct_discount_low", "ranked_pct_discount_high",
                 "clears_min_useful_effect"]
    )
    summary: dict[str, object] = {}
    for arm in ARMS:
        if arm == "a_shipped_c":
            continue
        per_n = {n: _round_saving(med, n, arm) for n in lens}
        wm = _window_mean(per_n)
        local_pct = 100.0 * wm / LOCAL_ROUND_US
        ranked = LATENCY_CLASS_FACTOR * local_pct
        pricing.add_data(
            arm, per_n[512], per_n[576], per_n[768], per_n[1024], wm,
            local_pct, wm / MIN_USEFUL_US_PER_ROUND, ranked,
            ranked / INSITU_DISCOUNT[1], ranked / INSITU_DISCOUNT[0],
            wm >= MIN_USEFUL_US_PER_ROUND,
        )
        summary[f"pricing/{arm}/us_saved_per_round_window_mean"] = wm
        summary[f"pricing/{arm}/local_pct_of_round"] = local_pct
        summary[f"pricing/{arm}/ranked_pct_undiscounted"] = ranked
        summary[f"pricing/{arm}/fraction_of_min_useful_effect"] = (
            wm / MIN_USEFUL_US_PER_ROUND
        )
    run.log({"isolated/round_pricing": pricing})

    # --- cost decomposition at the scored cell -----------------------------
    n, m = 576, 5
    a = med[(n, m)]["a_shipped_c"] * 1e6
    loop = (med[(n, m)]["g_double_c"] - med[(n, m)]["a_shipped_c"]) * 1e6
    traffic = (a - med[(n, m)]["e_resident_c"] * 1e6)
    softmax = (a - med[(n, m)]["f_nosoftmax_c"] * 1e6)
    actual_mb = payload["heads"] * m * 2.0 * n * payload["head_dim"] * 2.0 / 1e6
    logical_mb = 2.0 * payload["kv_heads"] * n * payload["head_dim"] * 2.0 / 1e6

    summary.update(
        {
            "decomp/cell": f"N={n} M={m}",
            "decomp/shipped_us": a,
            "decomp/key_loop_us": loop,
            "decomp/fixed_us": a - loop,
            "decomp/fixed_pct": 100.0 * (a - loop) / a,
            "decomp/traffic_us": traffic,
            "decomp/traffic_pct": 100.0 * traffic / a,
            "decomp/softmax_us": softmax,
            "decomp/softmax_pct": 100.0 * softmax / a,
            "decomp/actual_read_mb": actual_mb,
            "decomp/logical_read_mb": logical_mb,
            "decomp/actual_gb_s": actual_mb / a * 1e3,
            "decomp/logical_gb_s": logical_mb / a * 1e3,
            "decomp/dram_peak_gb_s": 273.0,
            "decomp/e95_cache_resident_gb_s": 371.1,
            "fidelity/bit_exact_violations": violations,
            "fidelity/positive_control_detected_all": all(pos_detected),
            "fidelity/a_vs_double_max_abs": max(ref_max),
            "verdict/best_real_arm": "d_pack2_c",
            "verdict/min_useful_us_per_round": MIN_USEFUL_US_PER_ROUND,
            "verdict/traffic_ceiling_us_per_round_window_mean": _window_mean(
                {k: _round_saving(med, k, "e_resident_c") for k in lens}
            ),
            "verdict/stop_rule_fires": True,
        }
    )
    run.summary.update(summary)
    run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["census", "registers", "isolated"])
    args = ap.parse_args()
    if args.only in (None, "census"):
        log_census()
    if args.only in (None, "registers"):
        log_registers()
    if args.only in (None, "isolated"):
        log_isolated()


if __name__ == "__main__":
    main()
