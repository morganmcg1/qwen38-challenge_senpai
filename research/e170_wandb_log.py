#!/usr/bin/env python3
"""Publish the E170 `rows_per_simd` occupancy study to W&B.

    usage: research/e170_wandb_log.py --screen DIR --exactness DIR

THE QUESTION. The advisor's bandwidth model says the wide affine-4/g64 QMV
reaches 98.1 % of the 273 GB/s roof at NA = 3 and only 73.6 % at NA = 5, and
attributes the dip to register-limited occupancy. If that is the cause, then
halving the row block each simdgroup owns, from 4 output rows to 2, should buy
the residency back and recover streaming efficiency at NA = 5.

THREE RUNS:

  `e170-exactness`
      56 cells of 7 scored shapes by widths 2..9. Every cell digests the MLX
      `quantizedMM` launcher, the shipped `rows_per_simd = 4` kernel and the
      `rows_per_simd = 2` kernel, and all three must agree. Two positive
      controls prove the comparison can fail: a one-ulp bfloat16 activation
      perturbation, and a starved grid that covers half the output rows.
  `e170-screen`
      The ABBA timing screen at NA = 5, 4 and 3 over the same 7 shapes, with
      the per-leg noise floor and the session drift taken from the timed legs
      themselves. This is the run that decides the experiment.
  `e170-occupancy`
      The register and residency census from the AGX backend, for this host
      and for the ranked generation. Zero GPU seconds. This is the run that
      explains the screen.

The screen ran behind the real cool gate, so `cool_gate_passed_real_gate` is
true and the entry and exit GPU temperatures are logged. No number here is an
official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e170-rows-per-simd-occupancy"
BASE_SHA = "99a81d6a02f9f0d121b0f8b7edd81eb438bdca4e"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
PR_NUMBER = 170
HOST = "apple-m4-pro-applegpu_g16s-48gib"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
ENTRY_CELL = ("qwen_e120_qmv_wide<NA, USE_TABLE, ROWS> reached through "
              "Qwen35CustomQMV.matmul, sumTable arm")

# The advisor's model for this host, which the experiment is testing.
MODEL_FIXED_MS = 11.792
MODEL_PASS_MS = {2: 55.008, 3: 53.841, 4: 61.336, 5: 71.758}
MODEL_ROOF_FRACTION = {2: 0.960, 3: 0.981, 4: 0.861, 5: 0.736}
MINIMUM_USEFUL_FRACTION_OF_T = 0.04


def modelled_t_ms(m: int) -> float:
    groups = -(-m // {2: 2, 3: 3, 4: 4, 5: 5}[m])
    return MODEL_FIXED_MS + groups * MODEL_PASS_MS[m]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def read_identity(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if value:
            out[key.strip().replace("e170: ", "")] = value.strip()
    return out


def start(name: str, job_type: str, question: str, config: dict,
          gate_qualified: bool):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={
            "experiment": GROUP, "question": question,
            "entry_cell": ENTRY_CELL, "host": HOST,
            "local_arch": LOCAL_ARCH, "ranked_arch": RANKED_ARCH,
            "base_sha": BASE_SHA, "candidate_sha": git_head(),
            "advisor_branch": ADVISOR_BRANCH, "pr_number": PR_NUMBER,
            "harness": "local",
            "official_or_ranked_score": False,
            "cool_gate_passed_real_gate": gate_qualified,
            "gate_qualified_for_timing": gate_qualified,
            "timing_valid": gate_qualified,
            **config,
        },
        reinit=True,
    )


def log_exactness(path: pathlib.Path) -> str:
    gate = json.loads(path.read_text())
    run = start(
        "e170-exactness", "exactness",
        "Does rows_per_simd = 2 produce bit-identical output to the shipped "
        "rows_per_simd = 4 kernel and to the MLX launcher?",
        {"cells": len(gate["cells"]),
         "device": gate.get("device"),
         "widths": sorted({c["m"] for c in gate["cells"]}),
         "shapes": sorted({c["shape"] for c in gate["cells"]})},
        gate_qualified=False)
    digests = {c["digest_rows4"] for c in gate["cells"]}
    run.log({
        "exactness/cells": len(gate["cells"]),
        "exactness/mismatches_rows2_vs_rows4": gate["mismatches_rows2_vs_rows4"],
        "exactness/mismatches_rows4_vs_mlx": gate["mismatches_rows4_vs_mlx"],
        "exactness/distinct_digests": len(digests),
        "control/one_ulp_activation_changes_output":
            int(gate["control_one_ulp_activation_changes_output"]),
        "control/starved_grid_changes_output":
            int(gate["control_starved_grid_changes_output"]),
    })
    run.summary["verdict"] = (
        "bit-exact"
        if gate["mismatches_rows2_vs_rows4"] == 0
        and gate["mismatches_rows4_vs_mlx"] == 0
        and gate["control_one_ulp_activation_changes_output"]
        and gate["control_starved_grid_changes_output"]
        else "FAILED")
    url = run.url
    run.finish()
    return url


def log_screen(screen_path: pathlib.Path, identity_path: pathlib.Path) -> str:
    payload = json.loads(screen_path.read_text())
    identity = read_identity(identity_path)
    widths = payload["widths"]

    run = start(
        "e170-screen", "timing",
        "Does halving rows_per_simd at high NA recover streaming efficiency "
        "in the wide affine-4/g64 QMV?",
        {"widths": widths, "reps": payload["reps"],
         "inner_calls_per_timed_region": payload["inner_calls_per_timed_region"],
         "block_order": payload["block_order"],
         "arms_rows_per_simd": payload["arms_rows_per_simd"],
         "device": payload["device"],
         "gpu_temp_c_entry": float(identity.get("gpu_temp_c_before_screen", "nan")),
         "gpu_temp_c_exit": float(identity.get("gpu_temp_c_after_screen", "nan")),
         "minimum_useful_fraction_of_t": MINIMUM_USEFUL_FRACTION_OF_T},
        gate_qualified=identity.get("cool_gate_screen") == "passed")

    weighted: dict[tuple[int, int], float] = {}
    replicate: dict[int, list[float]] = {}
    drift: dict[int, list[float]] = {}
    signs = [0, 0]
    for shape in payload["shapes"]:
        calls = shape["calls_per_verify"]
        by_block = {(r["m"], r["block"]): r["seconds_per_call"]
                    for r in shape["rows"]}
        for m in widths:
            legs = [by_block.get((m, b)) for b in range(4)]
            if any(v is None for v in legs):
                continue
            a1, b1, b2, a2 = legs
            arm4, arm2 = statistics.mean((a1, a2)), statistics.mean((b1, b2))
            weighted[(m, 4)] = weighted.get((m, 4), 0.0) + arm4 * calls * 1e3
            weighted[(m, 2)] = weighted.get((m, 2), 0.0) + arm2 * calls * 1e3
            replicate.setdefault(m, []).append(
                100.0 * abs(b1 - b2) / statistics.mean((b1, b2)))
            drift.setdefault(m, []).append(100.0 * (a2 - a1) / arm4)
            signs[1] += 1
            signs[0] += 1 if arm2 > arm4 else 0
            run.log({
                "cell/m": m, "cell/shape": shape["name"],
                "cell/us_per_call_rows4": arm4 * 1e6,
                "cell/us_per_call_rows2": arm2 * 1e6,
                "cell/delta_pct": 100.0 * (arm2 - arm4) / arm4,
            })

    verdicts = {}
    for m in widths:
        qmv4, qmv2 = weighted[(m, 4)], weighted[(m, 2)]
        delta = 100.0 * (qmv2 - qmv4) / qmv4
        groups = -(-m // {2: 2, 3: 3, 4: 4, 5: 5}[m])
        d_t = (delta / 100.0) * MODEL_PASS_MS[m] * groups
        t_model = modelled_t_ms(m)
        d_t_pct = 100.0 * d_t / t_model
        rep_max = max(replicate[m])
        verdicts[m] = d_t_pct
        run.log({
            "width/m": m,
            "width/qmv_ms_rows4": qmv4,
            "width/qmv_ms_rows2": qmv2,
            "width/delta_pct": delta,
            "width/implied_delta_t_ms": d_t,
            "width/implied_delta_t_pct": d_t_pct,
            "width/replicate_spread_pct_median": statistics.median(replicate[m]),
            "width/replicate_spread_pct_max": rep_max,
            "width/session_drift_pct_median": statistics.median(drift[m]),
            "width/effect_over_max_replicate_spread": delta / rep_max,
            "width/modelled_roof_fraction": MODEL_ROOF_FRACTION[m],
        })

    run.summary["noise_floor_pct_median"] = statistics.median(
        v for values in replicate.values() for v in values)
    run.summary["sign_cells_rows2_slower"] = signs[0]
    run.summary["sign_cells_total"] = signs[1]
    run.summary["implied_delta_t_pct_at_m5"] = verdicts[5]
    run.summary["verdict"] = (
        "promote" if verdicts[5] <= -100.0 * MINIMUM_USEFUL_FRACTION_OF_T
        else "stop")
    url = run.url
    run.finish()
    return url


def log_occupancy(path: pathlib.Path) -> str:
    census = json.loads(path.read_text())
    law = census["occupancy_law"]
    run = start(
        "e170-occupancy", "static_compile",
        "How much register-limited occupancy does halving rows_per_simd "
        "actually buy, and could a larger register file remove the NA = 5 dip?",
        {"register_file_bytes": census["register_file_bytes"],
         "bytes_per_simdgroup_register": census["bytes_per_simdgroup_register"],
         "ipg_table": census["ipg_table_read_live"],
         "gpu_seconds": 0},
        gate_qualified=False)

    for arch, block in law.items():
        for key, row in block["per_na_table_pipeline"].items():
            na = int(key[2:])
            run.log({
                "occupancy/na": na,
                "occupancy/arch": arch,
                f"occupancy/{arch}/registers_rows4": row["registers_rows4"],
                f"occupancy/{arch}/registers_rows2": row["registers_rows2"],
                f"occupancy/{arch}/simdgroups_rows4":
                    row["resident_simdgroups_rows4"],
                f"occupancy/{arch}/simdgroups_rows2":
                    row["resident_simdgroups_rows2"],
                f"occupancy/{arch}/simdgroup_ratio":
                    row["occupancy_ratio_rows2_over_rows4"],
                "occupancy/modelled_roof_fraction_rows4":
                    MODEL_ROOF_FRACTION.get(na),
            })
        entry = block["entry_table_pipeline"]
        run.summary[f"{arch}/entry_registers_rows4"] = entry["rows4"]["registers"]
        run.summary[f"{arch}/entry_registers_rows2"] = entry["rows2"]["registers"]
        run.summary[f"{arch}/entry_simdgroups_rows4"] = \
            entry["rows4"]["resident_simdgroups"]
        run.summary[f"{arch}/entry_simdgroups_rows2"] = \
            entry["rows2"]["resident_simdgroups"]
        threshold = block.get("dip_vanishes_if")
        if threshold:
            for key, value in threshold.items():
                run.summary[f"{arch}/dip_vanishes_if/{key}"] = value

    # The falsification in one number: at rows_per_simd = 2 and NA = 5 the
    # kernel already holds more resident simdgroups than the NA = 3 cell that
    # the model puts at 98.1 % of the roof, and it is still slower.
    for arch, block in law.items():
        na5_rows2 = block["per_na_table_pipeline"]["na5"]["resident_simdgroups_rows2"]
        na3_rows4 = block["per_na_table_pipeline"]["na3"]["resident_simdgroups_rows4"]
        run.summary[f"{arch}/na5_rows2_simdgroups_minus_na3_rows4"] = \
            na5_rows2 - na3_rows4

    occ = census.get("local_occupancy") or {}
    run.summary["local_pipeline_channel_max_threads"] = sorted({
        rec["max_total_threads_per_threadgroup"] for rec in occ.values()
        if isinstance(rec, dict)})
    url = run.url
    run.finish()
    return url


def log_transfer(path: pathlib.Path) -> str:
    """Does the NA = 5 deficit come from the source or from this host?"""
    census = json.loads(path.read_text())
    recon = census["arithmetic_reconciliation"]
    air = census["air_vector_census"]
    run = start(
        "e170-transfer", "static_compile",
        "Is the wide-QMV NA = 5 deficit fixed by the Metal source, and so "
        "carried to the ranked M5, or fixed by this host's register file?",
        {"expression": next(iter(recon.values()))["expression"],
         "register_file_bytes": census["register_file_bytes"],
         "ipg_table": census["ipg_table_read_live"],
         "gpu_seconds": 0},
        gate_qualified=False)

    for arch, block in recon.items():
        for cell in block["cells"]:
            run.log({
                "transfer/na": cell["na"],
                "transfer/rows_per_simd": cell["rows"],
                f"transfer/{arch}/data_registers_from_source":
                    cell["data_registers_from_source"],
                f"transfer/{arch}/compiled_registers":
                    cell["compiled_registers"],
                f"transfer/{arch}/scaffolding_headroom":
                    cell["scaffolding_headroom"],
            })
        run.summary[f"{arch}/widest_rows4_headroom"] = \
            block["widest_rows4_headroom"]
        run.summary[f"{arch}/other_cells_min_headroom"] = \
            block["other_cells_min_headroom"]
        run.summary[f"{arch}/median_headroom"] = block["median_headroom"]

    for name, rec in sorted(air.items()):
        run.summary[f"air/{name}/fma_total"] = rec["fma_total"]
        run.summary[f"air/{name}/lane_issues"] = rec["lane_issues"]
        run.summary[f"air/{name}/widths"] = json.dumps(rec["fma_widths"])

    run.summary["air_emits_native_v5f32"] = any(
        "v5f32" in rec["fma_widths"] for rec in air.values())
    run.summary["air_identical_rows4_vs_rows2"] = all(
        air[name]["fma_widths"] == air[name.replace("_r4_", "_r2_")]["fma_widths"]
        and air[name]["lane_issues"] == air[name.replace("_r4_", "_r2_")]["lane_issues"]
        for name in air if "_r4_" in name)
    run.summary["verdict"] = (
        "Source-fixed in mechanism, hardware-scaled in magnitude: the same "
        "register demand collapses the scaffolding headroom at NA = 5 and "
        "rows_per_simd = 4 on both architectures, but the M5 architecture "
        "keeps three times the residual headroom.")
    url = run.url
    run.finish()
    return url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=pathlib.Path)
    parser.add_argument("--exactness", type=pathlib.Path)
    parser.add_argument("--census", type=pathlib.Path,
                        default=pathlib.Path("research/e170-artifacts/register_census.json"))
    parser.add_argument("--only", default="",
                        help="comma separated subset of run names to log")
    args = parser.parse_args()

    wanted = {v.strip() for v in args.only.split(",") if v.strip()}
    plan = {
        "e170-exactness": lambda: log_exactness(args.exactness / "exactness.json"),
        "e170-screen": lambda: log_screen(args.screen / "screen.json",
                                          args.screen / "identity.txt"),
        "e170-occupancy": lambda: log_occupancy(args.census),
        "e170-transfer": lambda: log_transfer(args.census),
    }
    for name, make in plan.items():
        if wanted and name not in wanted:
            continue
        print(f"{name}  {make()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
