#!/usr/bin/env python3
"""Publish the E223 per-column inner-loop result to W&B.

    usage: research/e223_wandb.py [--session DIR] [--name NAME]

Two stages with DIFFERENT harness labels, both recorded verbatim:

  stage 0  static attribution census. Zero GPU seconds. `xcrun metal`,
           `xcrun metal-opt` and `agx_crossarch.translate` for the local
           `applegpu_g16s` and the ranked `applegpu_g17s`. No timing.
  stage 1  standalone kernel microbenchmark. harness=local-microbench. It
           holds no model, so the benchmark wrapper's process lock and 40C
           cool gate never applied and are NOT claimed:
           `cool_gate_passed_real_gate=false`,
           `gate_qualified_for_timing=false`.

Nothing here is an official or ranked score, and nothing here is a whole-leg
number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e223-artifacts"
REPO = HERE.parent

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e223-percolumn-inner-loop"
GROUP = "qwen38-r1-e223-percolumn-inner-loop"

RELIEF_CEILING_MS = 23.066
STOP_BAR_MS = 0.3
ADVANCE_BAR_MS = 1.0


def shell(*argv: str) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    return done.stdout.strip()


def identity(screen: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "assignment_pr": 221,
        "assignment_revision": "e223-r0",
        "candidate_sha": shell("git", "rev-parse", "HEAD"),
        "base_ref": "senpai/qwen38-mtp-r1",
        "assignment_base_sha":
            "a384559870722a20e7c48354afc72e623a9afefb",
        "dirty_candidate_paths": len([
            line for line in shell(
                "git", "status", "--porcelain", "--",
                "Sources", "Vendor", "Package.swift").splitlines() if line]),
        "host": platform.node(),
        "chip": shell("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(shell("sysctl", "-n", "hw.memsize") or 0),
        "os": shell("sw_vers", "-productVersion"),
        "swift": (shell("swift", "--version").splitlines() or [""])[0],
        "metal": (shell("xcrun", "metal", "--version").splitlines() or [""])[0],
        "local_arch": "applegpu_g16s",
        "ranked_arch": "applegpu_g17s",
        # Stage 1 harness labels, verbatim.
        "harness": "local-microbench",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "whole_leg_or_ranked_number": False,
        "reference_source":
            "self-generated random affine-4/group-64 cells at scored shapes; "
            "no model checkpoint is loaded",
        "scored_kernel": "qwen_e120_qmv_wide (qwen35E120QMVHeader)",
        "relief_ceiling_ms_per_round": RELIEF_CEILING_MS,
        "stop_bar_ms_per_round": STOP_BAR_MS,
        "advance_bar_ms_per_round": ADVANCE_BAR_MS,
        "rule_410_b_stream_us_per_mib":
            screen.get("rule_410_b_stream_us_per_mib"),
        "rule_410_cache_served_fraction_bound":
            screen.get("rule_410_cache_served_fraction_bound"),
    }


def arm_table(screen: dict) -> wandb.Table:
    columns = [
        "cell", "na", "groups", "m", "cold", "k", "n", "pooled_cell",
        "invocations_per_round",
        "xbf16_us_per_invocation", "xf32_us_per_invocation",
        "ratio_xf32_over_xbf16",
        "xbf16_ms_per_round", "xf32_ms_per_round",
        "measured_in_kernel_relief_ms_per_round",
        "ci95_half_width_ms_per_round", "ci95_overlaps_xbf16",
        "removed_instructions_per_invocation",
        "extra_issued_activation_mb_per_invocation",
        "access_regime_tag", "cache_served_fraction",
        "first_touch_cost_ms_per_round_at_b",
        "cache_served_cost_ms_per_round_at_0p36b",
        "desk_gross_relief_ms_per_round",
        "desk_predicted_in_kernel_net_ms_per_round",
        "desk_prediction_error_ms_per_round",
    ]
    rows = []
    for a in screen["arms"]:
        regime = a["access_regime"]
        d = a.get("desk", {})
        rows.append([
            a["cell"], a["na"], a["groups"], a["m"], a["cold"], a["k"], a["n"],
            a["pooled_cell"], a["invocations_per_round"],
            a["xbf16_us_per_invocation"], a["xf32_us_per_invocation"],
            a["ratio_xf32_over_xbf16"],
            a["xbf16_ms_per_round"], a["xf32_ms_per_round"],
            a["measured_in_kernel_relief_ms_per_round"],
            a["measured_ci95_half_width_ms_per_round"],
            a["ci95_overlaps_xbf16"],
            a["removed_instructions_per_invocation"],
            a["extra_issued_activation_mb_per_invocation"],
            regime["tag"], regime["cache_served_fraction"],
            regime["first_touch_cost_ms_per_round_at_b"],
            regime["cache_served_cost_ms_per_round_at_0p36b"],
            d.get("gross_relief_ms_per_round"),
            d.get("predicted_in_kernel_net_ms_per_round"),
            a.get("desk_prediction_error_ms_per_round"),
        ])
    return wandb.Table(columns=columns, data=rows)


def pooled_table(screen: dict) -> wandb.Table:
    columns = ["arm", "cells", "xbf16_ms_per_round",
               "measured_in_kernel_relief_ms_per_round",
               "ci95_half_width_ms_per_round", "separated_from_zero",
               "meets_stop_bar", "meets_advance_bar",
               "measured_shipped_net_ms_per_round",
               "desk_gross_relief_ms_per_round",
               "desk_predicted_in_kernel_net_ms_per_round"]
    rows = []
    for tag in sorted(screen["pooled"]):
        e = screen["pooled"][tag]
        rows.append([
            tag, ",".join(e["cells"]), e["xbf16_ms_per_round"],
            e["measured_in_kernel_relief_ms_per_round"],
            e["ci95_half_width_ms_per_round"], e["separated_from_zero"],
            e["meets_stop_bar"], e["meets_advance_bar"],
            e.get("measured_shipped_net_ms_per_round"),
            e.get("desk_gross_relief_ms_per_round"),
            e.get("desk_predicted_in_kernel_net_ms_per_round"),
        ])
    return wandb.Table(columns=columns, data=rows)


def occupancy_table(screen: dict) -> wandb.Table:
    """RULE 407: registers and spill for the EXACT timed instantiation."""
    columns = ["arch", "na", "registers_bf16", "registers_f32",
               "registers_delta", "spill_bytes_bf16", "spill_bytes_f32",
               "spill_bytes_delta", "text_bytes_bf16", "text_bytes_f32"]
    rows = []
    for arch, per_na in sorted(screen["rule_407_occupancy_witness"].items()):
        for na_key in sorted(per_na, key=lambda s: int(s[2:])):
            e = per_na[na_key]
            rows.append([
                arch, int(na_key[2:]), e["registers_bf16"],
                e["registers_f32"], e["registers_delta"],
                e["spill_bytes_bf16"], e["spill_bytes_f32"],
                e["spill_bytes_delta"], e["text_bytes_bf16"],
                e["text_bytes_f32"],
            ])
    return wandb.Table(columns=columns, data=rows)


def census_table(censuses: dict[str, dict]) -> wandb.Table:
    """Stage 0: the per-column instruction slope of every variant."""
    columns = ["variant", "path", "arch", "regime", "instructions_per_na",
               "slope_bytes_per_na", "r_squared", "air_lane_ops_intercept",
               "air_lane_ops_per_na", "air_r_squared", "source_sha8"]
    rows = []
    for variant, blob in sorted(censuses.items()):
        for key, fit in sorted(blob.get("fits", {}).items()):
            if not key.startswith("agx/") or "text_bytes_by_regime" not in key:
                continue
            _, arch, path, _ = key.split("/")
            air = blob["fits"].get("air/%s/ops_per_k_block/TOTAL" % path) or {}
            for regime, entry in sorted((fit or {}).items()):
                if not isinstance(entry, dict):
                    continue
                rows.append([
                    variant, path, arch, regime,
                    entry.get("instructions_per_na"), entry.get("slope"),
                    entry.get("r_squared"),
                    air.get("intercept"), air.get("slope"),
                    air.get("r_squared"), blob.get("source_sha8"),
                ])
    return wandb.Table(columns=columns, data=rows)


def coefficient_table(screen: dict) -> wandb.Table:
    """The instruction-relief bound E226 can hold out as a self-test target."""
    coef = screen["coefficient"]
    insn = coef["min_byte_arms"]["removed_instructions_per_round_pooled"]
    table = wandb.Table(columns=[
        "arm", "removed_instructions_per_round_pooled",
        "measured_ms_per_round", "ci95_half_width_ms_per_round",
        "measured_upper_bound_ms_per_round",
        "op_proportional_prediction_ms_per_round", "op_proportional_excluded",
        "us_per_1e9_removed_instructions_upper_bound"])
    for tag in ("na2/g1/cold", "na2/g1/hot"):
        e = coef.get(tag)
        if not e:
            continue
        table.add_data(
            tag, insn, e["measured_ms_per_round"],
            e["ci95_half_width_ms_per_round"],
            e["measured_upper_bound_ms_per_round"],
            e["op_proportional_prediction_ms_per_round"],
            e["op_proportional_excluded"],
            e["us_per_removed_instruction_upper_bound"] * 1e9)
    return table


def exactness_table(sanity: dict) -> wandb.Table:
    columns = ["cell", "k", "n", "m", "activation_dtype", "positive_control",
               "elements", "differing", "non_finite", "bit_exact"]
    rows = []
    for e in sanity["value_comparisons"]:
        control = bool(e.get("positive_control"))
        rows.append([
            e["cell"], e["k"], e["n"], e["m"], e["activation_dtype"],
            control, e["elements"], e["differing"], e.get("non_finite"),
            (e["differing"] > 0) if control else (e["differing"] == 0),
        ])
    return wandb.Table(columns=columns, data=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=pathlib.Path,
                    default=REPO / "research/out/e219/e223-xdtype")
    ap.add_argument("--sanity", type=pathlib.Path,
                    default=REPO / "research/out/e219/e223-sanity")
    ap.add_argument("--name", default="e223-percolumn-inner-loop")
    ap.add_argument("--run-id", default=None,
                    help="resume this run instead of creating a new one")
    args = ap.parse_args()

    screen = json.loads((ARTIFACTS / "e223-screen.json").read_text())
    sanity = json.loads((args.sanity / "sanity_xdtype.json").read_text())
    censuses = {
        path.stem.replace("e223-census-", ""): json.loads(path.read_text())
        for path in sorted(ARTIFACTS.glob("e223-census-*.json"))
    }

    verdict = screen["verdict"]
    coef = screen["coefficient"]
    best = screen["pooled"][verdict["best_pooled_arm"]]
    worst_tag = min(
        screen["pooled"],
        key=lambda t: screen["pooled"][t][
            "measured_in_kernel_relief_ms_per_round"])

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name, group=GROUP,
        id=args.run_id, resume="allow" if args.run_id else None,
        job_type="kernel-microbenchmark",
        tags=["e223", "qmv", "activation-dtype", "negative-result",
              "local-microbench", "rule-410"],
        config=identity(screen) | {
            "variants_screened": sorted(censuses),
            "gpu_temperature_c": screen.get("gpu_temperature_c"),
        },
        notes=(
            "E223: does an NA-contiguous activation access path relieve the "
            "per-column QMV cost slope? REFUTED. The only variant that "
            "survived the stage-0 desk, float32 activations in the shipped "
            "[m, k] layout, is bit-exact and removes 10.2 percent of the "
            "machine instructions per column per k-block, yet it is slower at "
            "every scored width above NA=2. The gather is not "
            "instruction-issue-limited."),
    )

    run.summary.update({
        "best_pooled_arm": verdict["best_pooled_arm"],
        "best_pooled_in_kernel_relief_ms_per_round":
            best["measured_in_kernel_relief_ms_per_round"],
        "best_pooled_ci95_half_width_ms_per_round":
            best["ci95_half_width_ms_per_round"],
        "best_pooled_separated_from_zero": best["separated_from_zero"],
        "worst_pooled_arm": worst_tag,
        "worst_pooled_in_kernel_relief_ms_per_round":
            screen["pooled"][worst_tag][
                "measured_in_kernel_relief_ms_per_round"],
        "any_arm_meets_advance_bar": verdict["any_arm_meets_advance_bar"],
        "any_arm_meets_stop_bar": verdict["any_arm_meets_stop_bar"],
        "decision_uses_b_stream": verdict["decision_uses_b_stream"],
        "decision_flips_anywhere_in_b_prime":
            verdict["decision_flips_anywhere_in_b_prime"],
        "instructions_removed_per_column_per_kblock_g17s":
            screen["per_column_instructions_g17s"]["bfloat16"]
            - screen["per_column_instructions_g17s"]["float32"],
        "instruction_reduction_fraction_g17s":
            1.0 - screen["per_column_instructions_g17s"]["float32"]
            / screen["per_column_instructions_g17s"]["bfloat16"],
        "bit_exact_rows": sum(
            1 for e in sanity["value_comparisons"]
            if not e.get("positive_control") and e["differing"] == 0),
        "bit_exact_failures": sum(
            1 for e in sanity["value_comparisons"]
            if not e.get("positive_control") and e["differing"] != 0),
        "positive_control_rows": sum(
            1 for e in sanity["value_comparisons"]
            if e.get("positive_control")),
        "positive_control_failures": sum(
            1 for e in sanity["value_comparisons"]
            if e.get("positive_control") and e["differing"] == 0),
        "coefficient_removed_instructions_per_extra_byte":
            coef["removed_instructions_per_extra_byte"][0],
        "coefficient_collinear_not_separately_identifiable": coef["collinear"],
        "coefficient_min_byte_arm_removed_instructions_per_round":
            coef["min_byte_arms"]["removed_instructions_per_round_pooled"],
        # Upper bound on the worth of one removed non-arithmetic instruction,
        # scaled to a readable unit. E226 can hold this out as a self-test.
        "coefficient_us_per_1e9_removed_instructions_upper_bound_hot":
            coef["na2/g1/hot"]["us_per_removed_instruction_upper_bound"] * 1e9,
        "coefficient_us_per_1e9_removed_instructions_upper_bound_cold":
            coef["na2/g1/cold"]["us_per_removed_instruction_upper_bound"] * 1e9,
        "coefficient_op_proportional_excluded_hot":
            coef["na2/g1/hot"]["op_proportional_excluded"],
        "coefficient_op_proportional_excluded_cold":
            coef["na2/g1/cold"]["op_proportional_excluded"],
        "g16s_f32_spill_onset_na": 3,
        "g16s_bf16_spill_onset_na": 6,
        "g17s_f32_spills_at_scored_na": False,
        "local_to_ranked_transfer_blocked": True,
        "desk_stage0_predicted_best_pooled_ms_per_round": 2.00,
        "verdict": "refuted",
        "official_or_ranked_score": False,
        "gate_qualified_for_timing": False,
    })

    run.log({
        "timed_arms": arm_table(screen),
        "pooled_verdict": pooled_table(screen),
        "rule_407_occupancy": occupancy_table(screen),
        "instruction_relief_coefficient": coefficient_table(screen),
        "stage0_variant_census": census_table(censuses),
        "bitexactness": exactness_table(sanity),
    })

    artifact = wandb.Artifact("e223-percolumn-inner-loop", type="experiment")
    for path in sorted(ARTIFACTS.glob("*.json")):
        artifact.add_file(str(path))
    for path in sorted(ARTIFACTS.glob("*.md")):
        artifact.add_file(str(path))
    for directory, names in (
        (args.session, ("xdtype.json", "xdtype.meta.txt", "xdtype.log")),
        (args.sanity, ("sanity_xdtype.json", "sanity.json",
                       "sanity_rows.json", "sanity.meta.txt")),
    ):
        for name in names:
            path = directory / name
            if path.exists():
                artifact.add_file(
                    str(path), name="session/%s/%s" % (directory.name, name))
    run.log_artifact(artifact)

    print("desk stage-0 best pooled prediction: +2.00 ms/round")
    print("measured best pooled: %+.3f +/- %.3f ms/round (%s)"
          % (best["measured_in_kernel_relief_ms_per_round"],
             best["ci95_half_width_ms_per_round"],
             verdict["best_pooled_arm"]))
    print("run:", run.url)
    print("run id:", run.id)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
