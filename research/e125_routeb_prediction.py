#!/usr/bin/env python3
"""E125 Stage 0 - arithmetic behind the registered rung 5e prediction.

Every input is a published number: thorfinn's 7x7 per-width ranked conversion,
the ledger's per-prompt ranked mean widths, beagle's median weight, and the two
published isolated-to-in-situ frame transfers. Nothing here is fitted.
"""

from __future__ import annotations

import json

# thorfinn's 7x7 grid, ranked % by wide-QMV width M
RANKED_BY_WIDTH = {3: 0.01, 4: 2.36, 5: 2.78, 6: 1.32, 7: 2.37, 8: 3.28, 9: 1.44}

BEAGLE_MEAN_M = 5.382
BEAGLE_MEDIAN_WEIGHT = 0.484
SECOND_MEDIAN_CANDIDATES = {
    "republic": 5.989,
    "essays": 6.087,
    "medicine": 6.256,
    "botany": 7.148,
}
BEAGLE_ACCEPT = 0.834

# frame transfers published before this file existed
F_RULE58 = 22.75 / 13.12  # launched-volume / grouping class
F_ALPHONSE_RAW = 0.890 / 0.436  # threadgroup exchange, raw
F_ALPHONSE_REWEIGHTED = 0.545 / 0.436  # after his own cell re-weighting
ALPHONSE_SHIPPED_PREDICTION = 0.890
ALPHONSE_MEASURED = 0.436
ALPHONSE_SD = 0.093

PARITY_LINE = 0.53
MODE_PROOF_LINE = 1.86


def ranked_at(m: float) -> float:
    lo = int(m)
    hi = lo + 1
    if hi not in RANKED_BY_WIDTH:
        return RANKED_BY_WIDTH[lo]
    return RANKED_BY_WIDTH[lo] + (m - lo) * (RANKED_BY_WIDTH[hi] - RANKED_BY_WIDTH[lo])


def truncated_geometric(accept: float, cap: int = 9) -> dict[int, float]:
    """Realised width M = 1 + accepted drafts, capped at `cap`."""
    dist = {k: accept ** (k - 1) * (1.0 - accept) for k in range(1, cap)}
    dist[cap] = accept ** (cap - 1)
    return dist


def main() -> None:
    beagle = ranked_at(BEAGLE_MEAN_M)
    others = {k: ranked_at(v) for k, v in SECOND_MEDIAN_CANDIDATES.items()}
    mean_other = sum(others.values()) / len(others)
    w = BEAGLE_MEDIAN_WEIGHT
    isolated = w * beagle + (1 - w) * mean_other
    isolated_lo = w * beagle + (1 - w) * min(others.values())
    isolated_hi = w * beagle + (1 - w) * max(others.values())

    # Jensen term: E[f(M)] against f(E[M]) under a realised width histogram
    jensen = {}
    for accept in (BEAGLE_ACCEPT, 0.87):
        dist = truncated_geometric(accept)
        mean_m = sum(k * p for k, p in dist.items())
        ef = sum(p * RANKED_BY_WIDTH.get(k, 0.0) for k, p in dist.items())
        jensen[accept] = {"mean_m": mean_m, "E_f": ef, "factor": beagle / ef}

    w_term = 1.33  # geometric mid of [1.00, 1.76]
    branches = {
        "primary": 1.43,  # sqrt(1.00 x 2.04)
        "T_flat_reading": F_ALPHONSE_REWEIGHTED,
        "A_roofline_law": F_ALPHONSE_RAW,
        "N_no_frame_term": 1.00,
    }
    preds = {
        name: {"F": f_term, "C": w_term * f_term, "ranked_pct": isolated / (w_term * f_term)}
        for name, f_term in branches.items()
    }

    c_primary = w_term * branches["primary"]
    alphonse_pred = ALPHONSE_SHIPPED_PREDICTION / c_primary
    alphonse_sd_error = abs(alphonse_pred - ALPHONSE_MEASURED) / ALPHONSE_SD

    out = {
        "ranked_at_beagle_mean_width": beagle,
        "second_median_candidates_ranked": others,
        "isolated_ranked_recomputed_on_7x7": isolated,
        "isolated_ranked_band": [isolated_lo, isolated_hi],
        "carried_headline": 2.47,
        "grid_reread_factor": 2.47 / isolated,
        "jensen": jensen,
        "W": {"central": w_term, "band": [1.00, 1.76]},
        "frame_transfers_published": {
            "rule58_launched_volume": F_RULE58,
            "alphonse_raw": F_ALPHONSE_RAW,
            "alphonse_reweighted": F_ALPHONSE_REWEIGHTED,
        },
        "predictions": preds,
        "band_80pct": [isolated / 2.70, isolated / 1.30],
        "envelope": [isolated_lo / (1.76 * 2.04), isolated_hi / 1.00],
        "alphonse_backcheck": {
            "predicted": alphonse_pred,
            "measured": ALPHONSE_MEASURED,
            "sd": ALPHONSE_SD,
            "error_in_sd": alphonse_sd_error,
        },
        "decision_lines": {"parity": PARITY_LINE, "mode_proof": MODE_PROOF_LINE},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
