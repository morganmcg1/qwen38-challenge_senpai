#!/usr/bin/env python3
"""E48 Part 1: score-weighted per-width cost shares of candidate-leg QMV time.

Cost model is thorfinn's E46 refit T(M) = 16.757 + 27.532*ceil(M/IPG(M)) + 9.624*M,
with IPG read from the dispatch switch of THIS base (kernels/quantized.h), not the
E27 table that E42 was measured on.

The beagle/medicine prompts are hidden (fixtures/qwen3_8_27b_mtp_track.json lists
them as R2 paths only), so their per-round width histogram cannot be measured
locally. This script instead reports the maximum-entropy exponential tilt of the
measured corpus histogram to each prompt's known mean draft width, which is the
only per-prompt width statistic the ranked receipts expose.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from qmv_score_leverage import (
    kink_pct,
    marginal_weights,
    saturation_cap_pct,
    score_pct_from_leg_gains,
)

# Dispatch switch of this base, kernels/quantized.h:1922-1979 (out_vec_size >= 4096).
# M=2 uses the pair kernel qmv_fast_crossrow_affine4_g64<T,2> (inputs_per_group 2).
IPG = {2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# E42 local_fixture_census: 78 verify rounds of ONE 512-token --local-iterate decode
# of correctness_prompts/public_longcopy_gate_english_512_256.json at offered depth 8.
CORPUS_HISTOGRAM = {2: 1, 3: 0, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}

# E42 ranked_census, derived from officialMetrics per-prompt effective_mean_draft_len.
MEAN_M = {"beagle": 5.5327102803738315, "medicine": 5.767676767676767}

# 🔴 SUPERSEDED. The brief called this "the marginal value ratio from the
# order-statistic structure"; the advisor withdrew that on PR 52 (comment
# 5343772907). It is really E40's per-prompt LEG-EFFECT split (+0.363 % vs
# +0.088 %), i.e. how much room each prompt has, not what a unit of gain in each
# is worth. Retained only so the propagation of the mislabelled constant stays on
# the record.
SUPERSEDED_WEIGHT = {"beagle": 0.79, "medicine": 0.21}
SUPERSEDED_REASON = (
    "E40 leg-effect split (+0.363 % beagle vs +0.088 % medicine, 4.1x) mislabelled "
    "in the E48 brief as an order-statistic marginal weight; withdrawn by the "
    "advisor on PR 52 comment 5343772907. Using it as a weight double-counts the "
    "heterogeneity and biases every slice DOWNWARD, because medicine drafts longer "
    "than beagle (mean_m 5.7677 vs 5.5327)."
)

# E48 Arm G, two doses, this base. Measured, not inherited.
PSI_MTP = 0.693390924409709
PSI_MTP_INTERVAL = (0.6922916384758034, 0.6944902103436144)

# Mechanisms to re-price, as candidate-leg QMV cost reductions at the named slice.
# Only alphonse's magnitude is measured; the M=9 row deliberately reuses the SAME
# 11.421 % so the two slices are comparable, and so the order-statistic kink shows
# up on a mechanism large enough to cross it. It is NOT a claim about the size of
# any particular M=9 proposal.
MECHANISMS = {
    "alphonse_E44r2_simdgroup_M7_8": {
        "slice": "f_7_8",
        "qmv_pct": 11.421,
        "magnitude_provenance": "measured, alphonse E44 r1/r2 mean over attn_out and "
        "mlp_down at M in {7,8}",
    },
    "hypothetical_M9_at_the_same_11_421_pct": {
        "slice": "M9",
        "qmv_pct": 11.421,
        "magnitude_provenance": "NOT MEASURED. Same size as alphonse's cell, chosen "
        "only for slice-to-slice comparability and to exercise the kink.",
    },
}


def cost(m: int) -> float:
    return 16.757 + 27.532 * math.ceil(m / IPG[m]) + 9.624 * m


def cost_shares(histogram: dict[int, float]) -> dict[int, float]:
    weighted = {m: n * cost(m) for m, n in histogram.items()}
    total = sum(weighted.values())
    return {m: w / total for m, w in weighted.items()}


def mean_width(histogram: dict[int, float]) -> float:
    total = sum(histogram.values())
    return sum(m * n for m, n in histogram.items()) / total


def exponential_tilt(histogram: dict[int, float], target_mean: float) -> tuple[dict[int, float], float]:
    """Max-entropy reweighting of `histogram` subject to a mean-width constraint."""
    lo, hi = -5.0, 5.0
    lam = 0.0
    tilted: dict[int, float] = {}
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        weights = {m: n * math.exp(lam * m) for m, n in histogram.items()}
        total = sum(weights.values())
        tilted = {m: w / total for m, w in weights.items()}
        if mean_width({m: p for m, p in tilted.items()}) * 1.0 < target_mean:
            lo = lam
        else:
            hi = lam
    return tilted, lam


@dataclass
class Slices:
    m9: float
    f78: float
    f456: float

    def as_dict(self) -> dict[str, float]:
        return {"M9": self.m9, "f_7_8": self.f78, "f_4_5_6": self.f456}


def slices(shares: dict[int, float]) -> Slices:
    return Slices(shares[9], shares[7] + shares[8], shares[4] + shares[5] + shares[6])


def main() -> None:
    out: dict[str, object] = {
        "cost_model": "E46 refit T(M)=16.757+27.532*ceil(M/IPG)+9.624*M",
        "ipg_table_source": "kernels/quantized.h dispatch switch on base fb0a09d",
        "ipg": IPG,
        "per_width_cost_ms": {m: round(cost(m), 3) for m in sorted(IPG)},
    }

    corpus = cost_shares(CORPUS_HISTOGRAM)
    out["corpus"] = {
        "histogram": CORPUS_HISTOGRAM,
        "denominator": "78 verify rounds, one 512-token local-iterate decode of "
        "public_longcopy_gate_english_512_256.json, offered depth 8, local M4 Pro",
        "mean_m": mean_width(CORPUS_HISTOGRAM),
        "cost_shares": {m: round(v, 6) for m, v in corpus.items()},
        "slices": slices(corpus).as_dict(),
    }

    per_prompt = {}
    for name, target in MEAN_M.items():
        tilted, lam = exponential_tilt(CORPUS_HISTOGRAM, target)
        shares = cost_shares(tilted)
        per_prompt[name] = {
            "mean_m": target,
            "tilt_lambda": lam,
            "dispatch_shares": {m: round(v, 6) for m, v in tilted.items()},
            "cost_shares": {m: round(v, 6) for m, v in shares.items()},
            "slices": slices(shares).as_dict(),
        }
    out["per_prompt_maxent_tilt"] = per_prompt

    def weighted(weights: dict[str, float]) -> dict[int, float]:
        return {
            m: sum(weights[p] * per_prompt[p]["cost_shares"][m] for p in weights)
            for m in sorted(IPG)
        }

    weights = marginal_weights()
    combined = weighted(weights)
    out["score_weighted"] = {
        "weights": weights,
        "weights_source": "qmv_score_leverage.marginal_weights(), computed from the "
        "crown order statistics; NOT re-inlined here",
        "cost_shares": {m: round(v, 6) for m, v in combined.items()},
        "slices": slices(combined).as_dict(),
    }

    superseded = weighted(SUPERSEDED_WEIGHT)
    out["score_weighted_superseded_79_21"] = {
        "weights": SUPERSEDED_WEIGHT,
        "status": "SUPERSEDED, retained for the record",
        "reason": SUPERSEDED_REASON,
        "cost_shares": {m: round(v, 6) for m, v in superseded.items()},
        "slices": slices(superseded).as_dict(),
    }

    # Re-pricing. The score is the mean of order statistics 4 and 5, so a constant
    # %/% rate is only valid while the scored pair keeps its membership. Compute the
    # score change by re-sorting the per-prompt ratios instead of multiplying a rate.
    repricing = {}
    for mech, spec in MECHANISMS.items():
        leg_gains = {
            p: PSI_MTP * per_prompt[p]["slices"][spec["slice"]] * spec["qmv_pct"]
            for p in MEAN_M
        }
        order_stat_pct = score_pct_from_leg_gains(leg_gains)
        naive_rate_pct = sum(
            weights[p] * leg_gains[p] for p in weights
        )
        repricing[mech] = {
            "slice": spec["slice"],
            "qmv_cost_reduction_pct": spec["qmv_pct"],
            "magnitude_provenance": spec["magnitude_provenance"],
            "psi_mtp_used": PSI_MTP,
            "per_prompt_leg_gain_pct": {p: round(v, 6) for p, v in leg_gains.items()},
            "score_pct_order_statistic": order_stat_pct,
            "score_pct_naive_weighted_rate": naive_rate_pct,
            "rate_model_error_pct_points": naive_rate_pct - order_stat_pct,
            "above_kink": max(leg_gains.values()) > kink_pct(),
        }
    out["repricing"] = {
        "method": "qmv_score_leverage.score_pct_from_leg_gains(), which re-sorts the "
        "eight order statistics; the naive weighted rate is reported beside it only "
        "to show the size of the error the rate model makes",
        "kink_pct": kink_pct(),
        "saturation_cap_pct": saturation_cap_pct(),
        "mechanisms": repricing,
    }
    out["identification"] = (
        "PREDICTION ONLY. beagle/medicine prompts are hidden (R2-only in the track "
        "fixture) and officialMetrics exposes no per-round width histogram, so these "
        "shares are a max-entropy extrapolation from one measured statistic per "
        "prompt, not a measurement."
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
