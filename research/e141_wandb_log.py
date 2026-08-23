#!/usr/bin/env python3
"""E141: log every required metric to one W&B run.

The assignment names six metrics. Five are read straight out of the artifacts
the rungs wrote; the sixth, `e141_added_us_per_round_at_p015`, is DERIVED and
is labelled as such in the run config, because
`qwen35DerivedClusterProbeFraction` is not this experiment's editable surface.

The primary metric is reported twice under the pre-registered coefficient
range, as the pre-registered PREDICTION, and separately as the measured value.
The timed arms already contain both the extra accepted tokens and the extra
probe cost, so no coefficient is needed to combine them.

The measured value is reported on two clearly separated harnesses, because
CAMPAIGN RULE 115 makes them different numbers:

  harness=local   what the ABBA session measured directly on this M4 Pro.
  harness=ranked  the same evidence with the absolute per-round cost
                  re-expressed over the 52,726 us ranked round rather than the
                  ~195,000 us local one. `e141_net_ranked_pct` is this one.

An unlabelled score model is invalid, so every metric here carries its harness
in the name or in the run config.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# The advisor's pre-registered price: recoverable mass times a coefficient in
# this range gives the published-median gain in percent.
COEFF_LOW = 65.0
COEFF_HIGH = 203.0

# Cluster geometry, from Qwen35.swift. Rows per leaf is 8 and the probe
# fraction is 0.25 on the shipped surface.
ROWS_PER_LEAF = 8
PROBE_FRACTION_SHIPPED = 0.25


def load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def geometry(padded_rows: int) -> dict:
    leaves = padded_rows // ROWS_PER_LEAF
    probes = max(1, math.ceil(PROBE_FRACTION_SHIPPED * leaves))
    return {
        "padded_rows": padded_rows,
        "leaves": leaves,
        "probes_at_p025": probes,
        "rows_scored_at_p025": probes * ROWS_PER_LEAF,
        "probes_at_p015": max(1, math.ceil(0.15 * leaves)),
        "rows_scored_at_p015": max(1, math.ceil(0.15 * leaves)) * ROWS_PER_LEAF,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="research/e141-census.json")
    ap.add_argument("--rung1", default="research/e141-rung1-screen.json")
    ap.add_argument("--rung3", default="research/e141-rung3.json")
    ap.add_argument("--rung2", default="research/e141-rung2.json")
    ap.add_argument("--name", default="e141-unproposable-token-channel")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    census = load(args.census)
    rung1 = load(args.rung1)
    rung3 = load(args.rung3)
    rung2 = load(args.rung2)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    shipped = geometry(98_336)
    full = geometry(248_320)

    config: dict = {
        "experiment": "E141",
        "pr": 141,
        "commit": commit,
        "base_sha": "bdba19f66e84e7e2aa1f8162eeaa0a82579edc94",
        "arm_selector": "MLX_E141_DRAFT_PREFIX",
        "shipped_prefix": 98_304,
        "full_prefix": 248_320,
        "measurement_harness": "local",
        "score_model_harness": "ranked, CAMPAIGN RULE 115 conversion",
        "ranked_round_us": 52_860.0,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "cluster_geometry_shipped": shipped,
        "cluster_geometry_full": full,
        "probe_fraction": PROBE_FRACTION_SHIPPED,
        "p015_is_derived_not_measured": True,
        "p015_reason": (
            "qwen35DerivedClusterProbeFraction (Qwen35.swift:4937) is a plain "
            "let on another experiment's surface, so p=0.15 cannot be selected "
            "at run time from this branch"
        ),
        "coefficient_low": COEFF_LOW,
        "coefficient_high": COEFF_HIGH,
    }

    metrics: dict = {}
    summary_notes: list[str] = []

    if rung1:
        ok = float(rung1.get("e141_coarse_table_bit_reproduction", 0.0))
        metrics["e141_coarse_table_bit_reproduction"] = ok
        config["rung1_positive_control_failed_as_required"] = rung1.get(
            "positive_control_failed_as_required"
        )
        config["rung1_shipped_head_bytes_added"] = 0

    if census:
        mass = census["medpair"]["recoverable_pct_full_vocabulary"]
        metrics["e141_medpair_recoverable_fraction_pct"] = mass
        config["census_curve"] = census["medpair"]["curve"]
        config["census_share_prefix"] = {
            k: v["share_prefix"] for k, v in census["families"].items()
        }
        # The pre-registered price, evaluated at the MEASURED mass.
        metrics["e141_predicted_ranked_pct_at_coeff65"] = mass / 100.0 * COEFF_LOW
        metrics["e141_predicted_ranked_pct_at_coeff203"] = mass / 100.0 * COEFF_HIGH

    if rung3:
        delta = rung3.get("delta") or {}
        if delta:
            metrics["e141_full_vocab_accept_delta_pp"] = delta["accept_rate_delta_pp"]
            metrics["e141_rounds_saved_pct"] = delta["rounds_saved_pct"]
            metrics["e141_mean_tokens_per_round_shipped"] = delta[
                "mean_tokens_per_round_shipped"
            ]
            metrics["e141_mean_tokens_per_round_full"] = delta[
                "mean_tokens_per_round_full"
            ]
        divergences = 0
        witness_rows = 0
        for arm, blob in rung3["arms"].items():
            for seed, s in blob["seeds"].items():
                divergences += s.get("residual_divergence_count") or 0
                if not s.get("all_tokens_matched"):
                    divergences += 1
                if arm != "shipped":
                    witness_rows += s["widened_draft_rows"]
                config[f"rung3_{arm}_{seed}"] = s
        metrics["e141_exactness_divergences"] = float(divergences)
        metrics["e141_widened_draft_rows_witnessed"] = float(witness_rows)
        control = rung3.get("selector_positive_control")
        if control:
            config["selector_positive_control"] = control
            metrics["e141_selector_proven_live"] = float(
                bool(control["selector_proven_live"])
            )

    if rung2:
        metrics.update(
            {
                k: v
                for k, v in rung2.items()
                if isinstance(v, (int, float)) and k.startswith("e141_")
            }
        )
        config["rung2"] = rung2
        net = rung2.get("e141_net_ranked_pct")
        if net is not None:
            metrics["e141_net_ranked_pct"] = net
            bracket = rung2.get("e141_net_ranked_pct_bracket") or [net, net]
            metrics["e141_net_ranked_pct_low"] = bracket[0]
            metrics["e141_net_ranked_pct_high"] = bracket[1]
            summary_notes.append(
                f"measured net {net:.4f} % (harness=ranked, Rule 115; bracket "
                f"{bracket[0]:.4f} to {bracket[1]:.4f}) against the "
                "pre-registered "
                f"{metrics.get('e141_predicted_ranked_pct_at_coeff65', 0):.4f} % "
                f"to {metrics.get('e141_predicted_ranked_pct_at_coeff203', 0):.4f} %"
            )
            gross = rung2.get("e141_net_ranked_pct_gross_rounds_only")
            if gross is not None:
                summary_notes.append(
                    f"gross rounds-only term {gross:.4f} %, so the cost of "
                    "widening consumes "
                    f"{100.0 * (gross - net) / gross:.0f} % of it"
                )

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        job_type="experiment",
        tags=["e141", "mtp", "draft-vocabulary", "qwen38-27b"],
        config=config,
        mode="offline" if args.offline else "online",
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    if summary_notes:
        run.summary["e141_note"] = "; ".join(summary_notes)
    print(f"run_id  {run.id}")
    print(f"run_url {run.url}")
    for key in sorted(metrics):
        print(f"  {key} = {metrics[key]}")
    run.finish()


if __name__ == "__main__":
    main()
