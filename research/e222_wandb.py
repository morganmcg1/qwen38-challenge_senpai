#!/usr/bin/env python3
"""Publish the E222 dequant value-sharing result to W&B.

    usage: research/e222_wandb.py SESSION_DIR [--name NAME] [--notes TEXT]

`SESSION_DIR` is a `research/out/e222/TAG` directory written by
`research/e222_session.sh`. The analysis file that
`research/e222_analyze.py` produced is read from `research/e222-artifacts/`.

harness=local-microbench, Apple M4 Pro (`g16s`). This probe holds no model, so
the benchmark wrapper's process lock and 40C cool gate never applied. Every run
carries `coolGatePassedRealGate=false` and `gateQualifiedForTiming=false`
verbatim, and no number here is a whole-leg, official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e222-artifacts"
REPO = HERE.parent

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
EXPERIMENT = "e222-dequant-value-sharing"
GROUP = "qwen38-r1-e222-dequant-value-sharing"

PROMOTION_BAR_MS_PER_ROUND = 1.0
STOP_RULE_MS_PER_ROUND = 0.3


def shell(*argv: str) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    return done.stdout.strip()


def identity(analysis: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "harness": "local-microbench",
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "officialOrRankedScore": False,
        "wholeLegOrRankedNumber": False,
        "baseSha": shell("git", "rev-parse", "HEAD"),
        "baseRef": "senpai/qwen38-mtp-r1",
        "assignmentBaseSha": "1963bb5604e4eeab0c88f4892d969e7875793e40",
        "assignmentId": "e222-dequant-value-sharing",
        "revisionId": "e222-r0",
        "prNumber": 220,
        "dirtyCandidatePaths": len([
            line for line in shell(
                "git", "status", "--porcelain", "--",
                "Sources", "Vendor", "Package.swift").splitlines() if line]),
        "host": platform.node(),
        "chip": shell("sysctl", "-n", "machdep.cpu.brand_string"),
        "memoryBytes": int(shell("sysctl", "-n", "hw.memsize") or 0),
        "os": shell("sw_vers", "-productVersion"),
        "localArch": "applegpu_g16s",
        "rankedArch": "applegpu_g17s",
        "baselineArm": analysis["baseline_arm"],
        "arms": analysis["arms"],
        "widths": analysis["widths"],
        "blocks": analysis["blocks"],
        "chains": analysis["chains"],
        "abbaCounterbalanced": analysis["abba_counterbalanced"],
        "promotionBarMsPerRound": PROMOTION_BAR_MS_PER_ROUND,
        "stopRuleMsPerRound": STOP_RULE_MS_PER_ROUND,
        "widthCensusCap7": analysis["width_census"]["cap7"],
        "widthCensusCap8": analysis["width_census"]["cap8"],
        "registerBudgets": analysis["register_budgets"],
    }


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    return wandb.Table(
        columns=columns, data=[[r.get(c) for c in columns] for r in rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    parser.add_argument("--name", default="e222-value-sharing-screen")
    parser.add_argument("--notes", default=(
        "E222 fused dequant value sharing. Decisive negative: the shipped "
        "staged plan is ONE dispatch whose redundant weight group pass runs "
        "concurrently and is cache-served, so fusing it saves at most 36 "
        "percent of E219's fitted stream coefficient and loses outright once "
        "the wider accumulator costs occupancy."))
    args = parser.parse_args()

    session = pathlib.Path(args.session_dir)
    analysis = json.loads(
        (ARTIFACTS / "e222-screen-analysis.json").read_text())
    exact = json.loads((session / "exact.json").read_text())
    coverage = json.loads((session / "write-coverage.json").read_text())
    desk = json.loads((ARTIFACTS / "e222-desk.json").read_text())

    cal = analysis["fusion_saving_calibration"]
    ceiling = analysis["cellwise_ceiling"]["cap7/chain1"]
    comparisons = exact["comparisons"]
    census = coverage["census"]

    config = identity(analysis)
    config.update({
        "e219BUsPerMib": cal["e219_b_us_per_mib"],
        "e219AUsPerDispatchNotApplicable":
            cal["e219_a_us_per_dispatch_not_applicable"],
        "shippedStagedPlanIsOneDispatch":
            cal["shipped_staged_plan_is_one_dispatch"],
        "shippedGroupIndexSource": cal["shipped_group_index_source"],
        # Stage 1's desk prediction, kept so the size of the miss is on record.
        "deskArgmaxPooledMsPerRound":
            desk["pooled"]["per_width_argmax_over_ranked_legal_arms"][
                "pooled_ms_per_round"],
        "deskFusedR2PooledMsPerRound":
            desk["pooled"]["fused_r2/all_g2_widths"]["pooled_ms_per_round"],
        "deskByteModelSelftestWorstRelativeError":
            desk["byte_model_selftest"].get("worst_relative_error"),
    })

    run = wandb.init(
        project=PROJECT, entity=ENTITY, group=GROUP, name=args.name,
        notes=args.notes, config=config, job_type="microbench-screen",
        tags=["e222", "dequant-value-sharing", "qmv", "harness-local",
              "negative-result", "microbench"])

    run.summary.update({
        "verdict": "not useful: the redundant weight group pass is already "
                   "cache-served",
        # Correctness gates.
        "exact/comparisons": len(comparisons),
        "exact/elementsCompared": sum(c["elements"] for c in comparisons),
        "exact/differingTotal": sum(c["differing"] for c in comparisons),
        "exact/nonFiniteTotal": sum(c["non_finite"] for c in comparisons),
        "exact/bitExact": all(c["differing"] == 0 for c in comparisons),
        # RULE 408 write coverage.
        "rule408/instantiations": len(census),
        "rule408/keysVerified": sum(c["keys_written"] for c in census),
        "rule408/allExactlyOneWrite": all(
            c["keys_written"] == c["keys_expected"] and c["min_writes"] == 1
            and c["max_writes"] == 1 for c in census),
        "rule408/shortGridControlTrips": all(
            c["control_short_grid_keys_written"] < c["keys_expected"]
            for c in census),
        "rule408/doubledGridControlTrips": all(
            c["control_doubled_grid_max_writes"] == 2 for c in census),
        # The decisive mechanism number.
        "calibration/impliedMarginalBUsPerMibMean":
            cal["implied_marginal_b_us_per_mib_mean"],
        "calibration/impliedMarginalBUsPerMibBestCell":
            cal["implied_marginal_b_us_per_mib_max"],
        "calibration/bestCellFractionOfE219B": max(
            r["fraction_of_e219_b"] for r in cal["per_cell"]),
        # The verdict against the bar.
        "ceiling/cap7Chain1MsPerRound": ceiling["pooled_ms_per_round"],
        "ceiling/cap7Chain4MsPerRound":
            analysis["cellwise_ceiling"]["cap7/chain4"]["pooled_ms_per_round"],
        "ceiling/clearsPromotionBar":
            ceiling["pooled_ms_per_round"] >= PROMOTION_BAR_MS_PER_ROUND,
        "ceiling/clearsStopRule":
            ceiling["pooled_ms_per_round"] >= STOP_RULE_MS_PER_ROUND,
        "ceiling/shortfallVersusBar":
            PROMOTION_BAR_MS_PER_ROUND - ceiling["pooled_ms_per_round"],
        # Thermal record, kept verbatim for the permitted ungated mode.
        "thermal/minC": analysis["gpu_temperature_min_c"],
        "thermal/maxC": analysis["gpu_temperature_max_c"],
        "thermal/spreadC": analysis["gpu_temperature_spread_c"],
        "thermal/blockEntrySpreadC":
            analysis["block_entry_temperature_spread_c"],
        "gpuSeconds": 80,
    })

    for census_name in ("cap7", "cap8"):
        for entry in analysis["pooled"][census_name]:
            stem = f"pooled/{census_name}/{entry['arm']}"
            run.summary[f"{stem}/msPerRound"] = entry["pooled_ms_per_round"]
            run.summary[f"{stem}/ci95Lo"] = (
                entry["pooled_ms_per_round_ci95"][0])
            run.summary[f"{stem}/ci95Hi"] = (
                entry["pooled_ms_per_round_ci95"][1])
            run.summary[f"{stem}/allTermsTwoSided"] = (
                entry["all_terms_two_sided"])

    for net in analysis["width_nets"]:
        stem = f"widthNet/m{net['m']}/{net['arm']}"
        run.summary[f"{stem}/msPerRound"] = net["ms_per_round"]
        run.summary[f"{stem}/rankedLegal"] = net["ranked_legal"]
        run.summary[f"{stem}/twoSided"] = net["two_sided"]

    run.log({
        "calibration": table(
            ["cell", "baseline_group_passes", "candidate_group_passes",
             "stream_saved_mib", "predicted_b_term_us", "measured_isolated_us",
             "measured_slope_us", "implied_marginal_b_us_per_mib",
             "fraction_of_e219_b"], cal["per_cell"]),
        "unitFits": table(
            ["label", "cell", "m", "arm", "rows_per_simd", "simdgroups",
             "x_groups", "slope_us", "intercept_us", "mean_us_per_unit",
             "stream_bytes", "pass_stream_bytes", "activation_read_bytes"],
            analysis["unit_fits"]),
        "widthNets": table(
            ["m", "arm", "ms_per_round", "ranked_legal", "two_sided",
             "g17s_headroom", "complete"], analysis["width_nets"]),
        "perChain": table(
            ["cell", "m", "arm", "chain", "baseline_us", "candidate_us",
             "ratio_candidate_over_baseline", "delta_us"],
            analysis["per_chain"]["rows"]),
        "signStability": table(
            ["cell", "m", "arm", "faster_at_every_chain",
             "slower_at_every_chain", "sign_stable"],
            analysis["per_chain"]["sign_stability"]),
        "registerProbe": wandb.Table(
            columns=["arm_width", "instantiation", "g16s_registers",
                     "g16s_spill_bytes", "g16s_text_sha8", "g17s_registers",
                     "g17s_spill_bytes", "g17s_text_sha8",
                     "air_fp_sequence_sha8"],
            data=[[key] + [v[c] for c in (
                "instantiation", "g16s_registers", "g16s_spill_bytes",
                "g16s_text_sha8", "g17s_registers", "g17s_spill_bytes",
                "g17s_text_sha8", "air_fp_sequence_sha8")]
                for key, v in sorted(
                    analysis["register_probe_timed_instantiations"].items())]),
        "exactness": table(
            ["arm", "cell", "m", "rows_per_simd", "elements", "differing",
             "non_finite"], comparisons),
        "writeCoverage": table(
            ["arm", "cell", "m", "rows_per_simd", "keys_expected",
             "keys_written", "min_writes", "max_writes", "min_key", "max_key",
             "control_short_grid_keys_written",
             "control_doubled_grid_max_writes"], census),
        "cellwiseCeilingPicks": table(
            ["m", "cell", "arm", "ms_per_round"], ceiling["picks"]),
    })

    artifact = wandb.Artifact("e222-value-sharing", type="microbench")
    for path in sorted(ARTIFACTS.glob("*.json")):
        artifact.add_file(str(path))
    for path in sorted(session.glob("*.json")):
        artifact.add_file(str(path), name=f"session/{path.name}")
    run.log_artifact(artifact)

    print("run:", run.url)
    print("run id:", run.id)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
