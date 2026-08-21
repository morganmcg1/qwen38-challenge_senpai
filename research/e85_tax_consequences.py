#!/usr/bin/env python3
"""Turn the measured buffer-tax slope into a prediction for arms (a)+(b).

    usage: research/e85_tax_consequences.py TAX_SLOPE_JSON [--json OUT]

One tax unit is one dispatch boundary plus one materialised `[1, 1, 5120]`
bf16 intermediate, so its price splits into traffic, which is computable from
bandwidth, and the dispatch boundary, which is the residue. The residue is the
quantity nobody in this campaign has measured directly.

Arms (a) and (b) remove 6 dispatch boundaries and 105,280 bytes of intermediate
traffic per draft token, which is a different traffic-to-boundary mix from the
tax unit, so the two parts are applied separately.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BANDWIDTH_GBPS = 226.035          # local peak measured, advisor feedback
TAX_BUFFER_BYTES = 1 * 1 * 5120 * 2
ARM_BUFFERS = 6
ARM_BYTES = 105_280               # census total for (a)+(b)
TOKENS_PER_ROUND = 512.0 / 78.0   # 512-token window over the observed 78 rounds


def traffic_us(bytes_moved: float) -> float:
    """A materialised intermediate is written once and read once."""
    return 2.0 * bytes_moved / (BANDWIDTH_GBPS * 1e9) * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tax_json")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = json.load(open(args.tax_json))
    dpt = d["drafts_per_token"]
    levels = d["tax_levels"]
    per = d["per_level"]
    baseline = d["baseline_mtp_s_per_tok"]

    out: dict = {
        "drafts_per_token": dpt,
        "baseline_mtp_s_per_tok": baseline,
        "tax_unit_traffic_us": traffic_us(TAX_BUFFER_BYTES),
        "arm_traffic_us_per_draft": traffic_us(ARM_BYTES),
        "arm_buffers": ARM_BUFFERS,
    }

    # Segment slopes, to test whether the marginal price is constant.
    segs = {}
    for lo, hi in zip(levels, levels[1:]):
        dy = per[str(hi)]["mtp_s_per_tok_mean"] - per[str(lo)]["mtp_s_per_tok_mean"]
        slope_tok = dy / (hi - lo)
        sem_lo = per[str(lo)]["mtp_s_per_tok_sd"] / (per[str(lo)]["legs"] ** 0.5)
        sem_hi = per[str(hi)]["mtp_s_per_tok_sd"] / (per[str(hi)]["legs"] ** 0.5)
        sem = ((sem_lo ** 2 + sem_hi ** 2) ** 0.5) / (hi - lo)
        segs[f"{lo}-{hi}"] = {
            "us_per_token_per_unit": slope_tok * 1e6,
            "us_per_buffer": slope_tok * 1e6 / dpt,
            "us_per_buffer_sem": sem * 1e6 / dpt,
            "us_per_buffer_ci95_lo": (slope_tok - 1.96 * sem) * 1e6 / dpt,
            "us_per_buffer_ci95_hi": (slope_tok + 1.96 * sem) * 1e6 / dpt,
        }
    out["segment_slopes"] = segs
    lin = d["linearity_residual_us"]
    interior = levels[1]
    sem_interior = (per[str(interior)]["mtp_s_per_tok_sd"]
                    / (per[str(interior)]["legs"] ** 0.5)) * 1e6
    out["linearity_residual_us"] = lin
    out["linearity_residual_in_sem"] = {
        k: v / sem_interior for k, v in lin.items()
    }
    out["convexity_significant"] = any(
        abs(v) > 1.96 for v in out["linearity_residual_in_sem"].values())

    # Arm prediction from each candidate unit price.
    def predict(name: str, us_per_buffer: float) -> dict:
        boundary = us_per_buffer - out["tax_unit_traffic_us"]
        per_draft = ARM_BUFFERS * boundary + out["arm_traffic_us_per_draft"]
        per_token = per_draft * dpt
        return {
            "us_per_buffer_used": us_per_buffer,
            "dispatch_boundary_us": boundary,
            "arm_us_per_draft": per_draft,
            "arm_us_per_token": per_token,
            "arm_us_per_round": per_token * TOKENS_PER_ROUND,
            "arm_pct_of_candidate": 100.0 * per_token / (baseline * 1e6),
        }

    ols = d["estimators"]["ols"]
    out["prediction"] = {
        "full_range_ols": predict("ols", ols["slope_us_per_buffer"]),
        "full_range_ols_ci_lo": predict("lo", ols["ci95_lo_us_per_buffer"]),
        "full_range_ols_ci_hi": predict("hi", ols["ci95_hi_us_per_buffer"]),
        "low_segment": predict("low", segs[f"{levels[0]}-{levels[1]}"]["us_per_buffer"]),
    }
    pcts = [out["prediction"][k]["arm_pct_of_candidate"]
            for k in ("low_segment", "full_range_ols")]
    out["arm_pct_range"] = [min(pcts), max(pcts)]
    out["advisor_prior_pct"] = [0.03, 0.12]
    out["actionable_threshold_pct"] = 0.05
    out["straddles_threshold"] = bool(
        min(pcts) < 0.05 < max(pcts))

    print(json.dumps(out, indent=2, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
