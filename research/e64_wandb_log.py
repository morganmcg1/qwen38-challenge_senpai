#!/usr/bin/env python3
"""Log E64 rung 0b -- the accumulator private-memory dose test -- to W&B.

Three records in one run:

  air      the compile-only certification of each arm: allocas, the `phi
           <NA x float>` promotion, k-loop floating-point counts, private
           traffic, and CFG peak live registers, at NA=5 and NA=6.
  legs     every timed leg with its rep, leg position, arm, seconds and achieved
           weight bandwidth, so the palindrome balance is auditable.
  decision the preregistered effect, the null bars, and the verdict.

  python3 research/e64_wandb_log.py \
      --air research/e64-artifacts/rung0b-air.json \
      --analysis research/e64-artifacts/rung0b-analysis.json \
      --timing research/e64-artifacts/rung0b-timing-na5.json \
      --log research/e64-artifacts/rung0b-na5.log
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

# askeladd's E61 rung 1 single-stream ladder, the sole prior source of the step
# this experiment exists to explain.
ASKELADD_LADDER_GB_S = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946,
                        6: 117.8, 7: 97.9}
ASKELADD_LADDER_MS = {2: 64.40, 3: 72.17, 4: 82.24, 5: 95.48, 6: 122.34,
                      7: 147.21}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True,
                          text=True).stdout.strip()


def gate_status(log: str) -> tuple[bool, bool]:
    passed = "cool_gate_passed_real_gate=true" in log
    return passed, passed


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--air", type=pathlib.Path,
                        default="research/e64-artifacts/rung0b-air.json")
    parser.add_argument("--analysis", type=pathlib.Path,
                        default="research/e64-artifacts/rung0b-analysis.json")
    parser.add_argument("--timing", type=pathlib.Path, nargs="+",
                        default=["research/e64-artifacts/rung0b-timing-na5.json"])
    parser.add_argument("--log", type=pathlib.Path, nargs="+",
                        default=["research/e64-artifacts/rung0b-na5.log"])
    parser.add_argument("--name", default="e64-wide-qmv-accumulator-private-memory")
    args = parser.parse_args()

    air = json.loads(pathlib.Path(args.air).read_text())
    analysis = json.loads(pathlib.Path(args.analysis).read_text())
    logs = "\n".join(pathlib.Path(path).read_text()
                     for path in args.log if pathlib.Path(path).exists())
    gate_passed, gate_qualified = gate_status(logs)

    sessions = analysis["sessions"]
    primary = sessions[0]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="kernel-microbenchmark",
        name=args.name,
        tags=["e64", "qmv", "wide-crossrow", "rung0b", "accumulator",
              "private-memory", "qwen-edward"],
        config={
            "experiment": "e64-wide-qmv-accumulator-private-memory",
            "hypothesis": "the NA=5 -> 6 step in the wide crossrow QMV ladder "
                          "is `acc` leaving the registers for private memory",
            "rung": "0b",
            "base_sha": "3bf0e1f20fcdcc9b90d6e5ded52329bf74e4b52c",
            "head_sha": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "pr": 67,
            "student": "qwen-edward",
            "host": subprocess.run(["hostname", "-s"], capture_output=True,
                                   text=True).stdout.strip(),
            "host_chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True).stdout.strip(),
            "device": primary["device"],
            "metal_flags": air["flags"],
            "air_pipeline": air["pipeline"],
            "jit_compile_options": "MTLLanguageVersion4_0, fastMath off",
            "arms": ["plain", "forced", "ballast"],
            "leg_order": primary["order"],
            "reps": primary["reps"],
            "warmup_reps_discarded": primary["warmup_reps_discarded"],
            "na": primary["na"],
            "prereg_confirm_threshold": 0.10,
            "prereg_partial_threshold": 0.04,
            "cool_gate_passed_real_gate": gate_passed,
            "gate_qualified_for_timing": gate_qualified,
            "askeladd_ladder_gb_per_s": ASKELADD_LADDER_GB_S,
            "askeladd_ladder_ms": ASKELADD_LADDER_MS,
        },
    )

    census = wandb.Table(columns=[
        "na", "arm", "allocas", "alloca_types", "accumulator_alloca",
        "phi_acc_width", "peak_live_cfg_loop", "peak_live_text_order",
        "loop_fadd", "loop_fmul", "loop_fma", "loop_device_loads",
        "loop_private_loads", "loop_private_stores", "air_lines"])
    for na, cells in sorted(air["cells"].items(), key=lambda kv: int(kv[0])):
        for arm, stats in cells.items():
            census.add_data(
                int(na), arm, stats["allocas"], ", ".join(stats["alloca_types"]),
                bool(stats["acc_allocas"]), stats["phi_acc_width"],
                stats["peak_live_cfg_loop"], stats["peak_live_text_order"],
                stats["loop"]["fadd"], stats["loop"]["fmul"],
                stats["loop"]["fma"], stats["loop"]["device_loads"],
                stats["loop"]["private_loads"], stats["loop"]["private_stores"],
                stats["air_lines"])

    legs = wandb.Table(columns=["na", "shape", "rep", "position", "arm",
                                "seconds_per_dispatch", "gb_per_s",
                                "gpu_seconds", "wall_seconds"])
    for path in args.timing:
        timing = json.loads(pathlib.Path(path).read_text())
        for shape in timing["shapes"]:
            for leg in shape["legs"]:
                legs.add_data(timing["na"], shape["shape"], leg["rep"],
                              leg["position"], leg["arm"],
                              leg["seconds_per_dispatch"], leg["gbps"],
                              leg["gpu_seconds"], leg["wall_seconds"])

    effects = wandb.Table(columns=[
        "na", "shape", "arm", "effect_median_pct", "effect_min_pct",
        "effect_max_pct", "sign_stable_pct", "null_position_split_pct",
        "widest_same_arm_spread_pct", "widest_mirrored_leg_spread_pct",
        "verdict", "median_ms", "gb_per_s", "parity_differing_vs_plain",
        "entry_gpu_temp_c", "exit_gpu_temp_c"])
    for session in sessions:
        for shape in session["shapes"]:
            for arm, effect in shape["effects"].items():
                effects.add_data(
                    session["na"], shape["shape"], arm,
                    effect["median"] * 100, effect["min"] * 100,
                    effect["max"] * 100, effect["sign_stable_fraction"] * 100,
                    shape["null_position_split_median"] * 100,
                    shape["widest_same_arm_spread"] * 100,
                    shape["widest_mirrored_leg_spread"] * 100,
                    effect["verdict_vs_spread_bar"],
                    shape["median_seconds_per_dispatch"][arm] * 1e3,
                    shape["gb_per_s"][arm],
                    shape["parity_differing_vs_plain"].get(arm),
                    shape["entry_gpu_temp_c"], shape["exit_gpu_temp_c"])

    summary = {
        "rung0b/forced_effect_median": primary.get("forced_effect_median_over_shapes"),
        "rung0b/ballast_effect_median": primary.get("ballast_effect_median_over_shapes"),
        "rung0b/widest_same_arm_spread": primary["widest_same_arm_spread_over_shapes"],
        "rung0b/decision_forced": primary["decision_forced"],
        "rung0b/entry_gpu_temp_c": primary["entry_gpu_temp_c"],
        "rung0b/exit_gpu_temp_c": primary["exit_gpu_temp_c"],
        "rung0b/parity_differing_total": sum(
            sum(shape["parity_differing_vs_plain"].values())
            for session in sessions for shape in session["shapes"]),
        "air/forced_acc_alloca_present":
            bool(air["cells"]["5"]["forced"]["acc_allocas"]),
        "air/plain_acc_alloca_present_na5":
            bool(air["cells"]["5"]["plain"]["acc_allocas"]),
        "air/plain_acc_alloca_present_na6":
            bool(air["cells"]["6"]["plain"]["acc_allocas"]),
        "air/forced_fp_unchanged":
            air["checks"]["5"]["forced_loop_fp_unchanged"]
            and air["checks"]["5"]["forced_total_fp_unchanged"],
        "air/ballast_raises_live_registers":
            air["checks"]["5"]["ballast_raises_live_registers"],
    }
    run.log({
        "air_census": census,
        "legs": legs,
        "effects": effects,
        **{key: value for key, value in summary.items()
           if not isinstance(value, str)},
    })
    run.summary.update(summary)
    print(f"logged {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
