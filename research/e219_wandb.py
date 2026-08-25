#!/usr/bin/env python3
"""Publish the E219 QMV pass-anatomy census to W&B.

    usage: research/e219_wandb.py ARTIFACT_JSON [--dry-run]

`ARTIFACT_JSON` is the file `research/e219_analyze.py` wrote, by default
`research/e219-artifacts/pass-anatomy.json`.

One run per measured phase plus one `census` run that carries the fitted
coefficients, the reconciliation gate and the priced shortlist, all in group
`qwen38-r1-e219-qmv-pass-anatomy`.

harness=local-microbench, Apple M4 Pro. This probe holds no model, so the
benchmark wrapper's process lock and 40C cool gate never applied: every run
carries `coolGatePassedRealGate=false` and `gateQualifiedForTiming=false`
verbatim, and no number here is a whole-leg or ranked score (RULE 79).
"""

from __future__ import annotations

import argparse
import json
import pathlib

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e219-qmv-pass-anatomy"


def identity(blob: dict, extra: dict) -> dict:
    ident = blob["identity"]
    return {
        "harness": "local-microbench",
        "experiment": "e219-qmv-pass-anatomy",
        "baseSha": ident["base_sha"],
        "campaignBaseSha": ident["campaign_base_sha"],
        "host": ident["host"],
        "referenceSource": ident["reference_source"],
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "officialOrRankedScore": False,
        "wholeLegOrRankedNumber": False,
        "rowsPerSimd": 4,
        "threadgroup": "32x2x1",
        **extra,
    }


def phase_specs(blob: dict) -> dict:
    specs = {}
    for phase, meta in sorted(blob["phases"].items()):
        summary: dict = {
            "phase": phase,
            "blocks": meta.get("blocks"),
        }
        temps = [v for v in (meta.get("gpu_temperature_c") or {}).values()
                 if isinstance(v, (int, float)) and v > 0]
        if temps:
            summary["gpuTempMinC"] = min(temps)
            summary["gpuTempMaxC"] = max(temps)
            summary["gpuTempSpreadC"] = max(temps) - min(temps)
        for ring in meta.get("replica_rings") or []:
            stem = "replicas/%s" % ring.get("cell", "?")
            if "dispatches" in ring:
                stem += "/d%s" % ring["dispatches"]
            summary["%s/count" % stem] = ring["replicas"]
            summary["%s/setBytes" % stem] = ring["replica_set_bytes"]

        if phase == "sanity":
            rows = meta.get("comparisons") or []
            summary["comparisons"] = len(rows)
            summary["elementsCompared"] = sum(r["elements"] for r in rows)
            summary["differingTotal"] = sum(r["differing"] for r in rows)
            summary["nonFiniteTotal"] = sum(r["non_finite"] for r in rows)
        else:
            for unit in blob["units"]:
                if unit.get("phase") != phase:
                    continue
                stem = "unit/%s" % unit["label"].replace(".", "_")
                summary["%s/slopeUs" % stem] = unit["slope_us"]
                summary["%s/interceptUs" % stem] = unit["intercept_us"]
                summary["%s/ci95Lo" % stem] = unit["slope_ci95"][0]
                summary["%s/ci95Hi" % stem] = unit["slope_ci95"][1]

        config = identity(blob, {
            "stage": "phase-%s" % phase,
            "deliverable": meta.get("deliverable"),
            "chains": meta.get("chains"),
            "gpuUsed": phase != "sanity",
        })
        specs[phase] = ("e219-%s" % phase.replace("/", "-"), config, summary)
    return specs


def census_spec(blob: dict) -> tuple[str, dict, dict]:
    fit = blob.get("composition_fit") or {}
    recon = blob.get("reconciliation") or {}
    summary: dict = {
        "fit/designRank": fit.get("design_rank"),
        "fit/designColumns": fit.get("design_columns"),
        "fit/identifiable": fit.get("identifiable"),
        "fit/weightedR2": fit.get("weighted_r2"),
        "fit/nUnits": fit.get("n_units"),
        "fit/scaledConditionNumber": fit.get("scaled_condition_number"),
        "gate/aPrimePass": recon.get("gate_a_prime_pass"),
        "gate/aPass": recon.get("gate_a_pass"),
        "gate/bPass": recon.get("gate_b_pass"),
        "phi/identifiable": recon.get("phi_identifiable"),
        "phi/cacheServedFraction": recon.get("phi_cache_served_fraction"),
    }

    gate_ap = recon.get("gate_a_prime") or {}
    summary["gateAPrime/worstErrorPct"] = gate_ap.get("worst_error_pct")
    summary["gateAPrime/widthsChecked"] = len(
        gate_ap.get("widths_checked") or [])
    summary["gateAPrime/widthsMissing"] = len(
        gate_ap.get("widths_missing") or [])
    for row in gate_ap.get("rows") or []:
        stem = "gateAPrime/m%d" % row["m"]
        summary["%s/finding559MsPerRound" % stem] = (
            row["finding_559_r_local_ms_per_round"])
        if row.get("census_ms_per_round") is None:
            continue
        summary["%s/censusMsPerRound" % stem] = row["census_ms_per_round"]
        summary["%s/errorPct" % stem] = row["error_pct"]
        summary["%s/within15pct" % stem] = row["within_15pct"]

    for key, name in (
        ("us_per_dispatch", "term/aUsPerDispatch"),
        ("ms_per_round", "term/aMsPerRound"),
    ):
        block = (blob.get("term_interpretation") or {}).get("a_dispatch") or {}
        if block.get(key) is not None:
            summary[name] = block[key]
    ti = blob.get("term_interpretation") or {}
    b_term = ti.get("b_stream") or {}
    summary["term/bMsPerPass"] = b_term.get("ms_per_pass_all_cells")
    summary["term/bImpliedGBps"] = b_term.get("implied_stream_gb_per_s")
    summary["term/bExceedsDramPeak"] = b_term.get("exceeds_dram_peak")
    summary["term/bLaneCyclesPerWeightElement"] = b_term.get(
        "nominal_lane_cycles_per_weight_element")
    c_term = ti.get("c_activation_row") or {}
    summary["term/cMsPerPassPerNa"] = c_term.get("ms_per_pass_per_na")
    whole = ti.get("whole_pass_5_1") or {}
    summary["term/pass51MsPerRound"] = whole.get("modelled_ms_per_round")
    summary["term/pass51AchievedGBps"] = whole.get("achieved_gb_per_s")
    summary["term/pass51FractionOfDramPeak"] = whole.get(
        "fraction_of_dram_peak")

    dec = recon.get("finding_543_decomposition") or {}
    summary["decomp543/censusIpg3G3MsPerRound"] = dec.get(
        "census_ipg3_g3_ms_per_round")
    summary["decomp543/censusIpg5G2MsPerRound"] = dec.get(
        "census_ipg5_g2_ms_per_round")
    summary["decomp543/weightPassDeltaMsPerRound"] = dec.get(
        "census_weight_pass_delta_ms_per_round")
    summary["decomp543/weightPassShareOf543"] = dec.get(
        "weight_pass_share_of_543")

    lay = recon.get("layout_headroom") or {}
    summary["layout/achievedGBps"] = lay.get("achieved_gb_per_s")
    summary["layout/fractionOfPeak"] = lay.get("fraction_of_peak")
    summary["layout/pooledCeilingMsPerRound"] = lay.get(
        "pooled_ceiling_ms_per_round")

    gate_b = recon.get("gate_b") or {}
    summary["gateB/scoredWholeRoundRatio"] = gate_b.get("scored_ratio")
    summary["gateB/coldG1MsPerRound"] = gate_b.get("cold_g1_ms_per_round")
    summary["gateB/coldG2MsPerRound"] = gate_b.get("cold_g2_ms_per_round")
    summary["gateB/hotWholeRoundRatio"] = gate_b.get("hot_whole_round_ratio")

    gate_a = recon.get("gate_a") or {}
    for key, name in (
        ("cold_minus_hot_ms_per_round", "gateA/coldMinusHotMsPerRound"),
        ("cold_minus_hot_pct_of_cold", "gateA/coldMinusHotPctOfCold"),
        ("bracket_noise_2sigma_ms_per_round", "gateA/noise2SigmaMsPerRound"),
    ):
        if gate_a.get(key) is not None:
            summary[name] = gate_a[key]
    for key, name in (
        ("bracket_ms_per_round", "gateA/bracketMsPerRound"),
        ("contains_point_anchor", "gateA/containsPointAnchor"),
        ("cold_over_insitu", "gateA/coldOverInsitu"),
        ("hot_over_insitu", "gateA/hotOverInsitu"),
        ("residual_dram_served_fraction", "phi/dramResidualFraction"),
        ("phi_from_anchor_ci", "phi/fromAnchorCi"),
    ):
        if gate_a.get(key) is not None:
            summary[name] = gate_a[key]
    for tag in ("cold", "hot"):
        block = gate_a.get(tag)
        if block:
            summary["gateA/%s/scaledMsPerRound" % tag] = \
                block["scaled_to_all_cells_ms_per_round"]
            summary["gateA/%s/measuredCellsMsPerRound" % tag] = \
                block["measured_cells_ms_per_round"]
    for row in (recon.get("gate_b") or {}).get("rows") or []:
        stem = "gateB/%s/%s" % (row["cell"].replace(".", "_"), row["arm"])
        summary["%s/coldRatio" % stem] = row["cold_ratio"]
        summary["%s/hotRatio" % stem] = row["hot_ratio"]
        summary["%s/contains559Band" % stem] = row["contains_559_band"]
    for name, entry in (fit.get("coefficients") or {}).items():
        summary["coef/%s" % name] = entry["value"]
        summary["coef/%s/ci95Lo" % name] = entry["ci95"][0]
        summary["coef/%s/ci95Hi" % name] = entry["ci95"][1]
        summary["coef/%s/resolved" % name] = entry["resolved_away_from_zero"]
        if entry.get("implied_gb_per_s") is not None:
            summary["coef/%s/impliedGBps" % name] = entry["implied_gb_per_s"]
    for check in recon.get("finding_536_context") or []:
        stem = "context536/%s" % check["cell"].replace(".", "_")
        summary["%s/coldMsPerRound" % stem] = \
            check["cold_marginal_pass_ms_per_round"]
        summary["%s/hotMsPerRound" % stem] = \
            check["hot_marginal_pass_ms_per_round"]
        summary["%s/ratioColdVs536" % stem] = check["ratio_cold_vs_536"]
    for pair in blob.get("hot_vs_cold") or []:
        stem = "cache/%s/na%s/g%s" % (
            pair["cell"].replace(".", "_"), pair["na"], pair["groups"])
        summary["%s/coldUs" % stem] = pair["cold_us"]
        summary["%s/hotUs" % stem] = pair["hot_us"]
        summary["%s/hotMinusColdPct" % stem] = pair["hot_minus_cold_pct"]
        summary["%s/coldGBps" % stem] = pair["cold_implied_gb_per_s"]
    for entry in blob.get("priced_shortlist") or []:
        stem = "priced/%s" % entry["mechanism"].split(" (")[0].replace(" ", "_")
        summary["%s/pooledMsPerRound" % stem] = entry["pooled_ms_per_round"]
        summary["%s/publicMsPerRound" % stem] = entry["public_ms_per_round"]
        summary["%s/decisionMsPerRound" % stem] = entry["decision_ms_per_round"]
        summary["%s/clears1msBar" % stem] = entry["clears_1ms_bar"]
        summary["%s/pooledMsPerRoundCiUpper" % stem] = \
            entry["pooled_ms_per_round_ci_upper"]
        summary["%s/resolvedByCensus" % stem] = not entry.get(
            "not_resolved_by_this_census", False)

    config = identity(blob, {
        "stage": "census",
        "deliverable":
            "decompose one wide-QMV pass into dispatch, weight-stream, "
            "activation, output and threadgroup terms, and price the next "
            "mechanism wave under the FINDING 564 pooled census",
        "gpuUsed": False,
        "sessions": blob["identity"]["sessions"],
    })
    return "e219-census", config, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blob = json.loads(args.artifact.read_text())
    specs = phase_specs(blob)
    name, config, summary = census_spec(blob)
    specs["census"] = (name, config, summary)

    if args.dry_run:
        print(json.dumps(
            {k: {"name": n, "config": c, "summary": s}
             for k, (n, c, s) in specs.items()},
            indent=1, sort_keys=True, default=str))
        return 0

    import wandb

    published = {}
    for key, (run_name, cfg, summ) in specs.items():
        run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                         job_type="local-measurement", name=run_name,
                         config=cfg)
        run.summary.update(summ)
        published[key] = {"run": run.id, "url": run.url}
        run.finish()
    print(json.dumps(published, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
