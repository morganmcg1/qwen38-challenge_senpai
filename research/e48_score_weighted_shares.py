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

# Dispatch switch of this base, kernels/quantized.h:1922-1979 (out_vec_size >= 4096).
# M=2 uses the pair kernel qmv_fast_crossrow_affine4_g64<T,2> (inputs_per_group 2).
IPG = {2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# E42 local_fixture_census: 78 verify rounds of ONE 512-token --local-iterate decode
# of correctness_prompts/public_longcopy_gate_english_512_256.json at offered depth 8.
CORPUS_HISTOGRAM = {2: 1, 3: 0, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}

# E42 ranked_census, derived from officialMetrics per-prompt effective_mean_draft_len.
MEAN_M = {"beagle": 5.5327102803738315, "medicine": 5.767676767676767}

# Marginal value ratio from the order-statistic structure (advisor, PR 52).
SCORE_WEIGHT = {"beagle": 0.79, "medicine": 0.21}


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

    combined = {
        m: sum(SCORE_WEIGHT[p] * per_prompt[p]["cost_shares"][m] for p in SCORE_WEIGHT)
        for m in sorted(IPG)
    }
    out["score_weighted"] = {
        "weights": SCORE_WEIGHT,
        "cost_shares": {m: round(v, 6) for m, v in combined.items()},
        "slices": slices(combined).as_dict(),
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
