#!/usr/bin/env python3
"""Commit the E75 rung D prediction before the session runs.

Rung C predicted the 2x2 from a closed cost ladder. Rung B then MEASURED the
crown table's verify-cost curve, so the honest pre-registration for rung D is
the re-priced prediction, not the ladder one. This script writes it to a file
so the prediction is on the record with a commit timestamp, exactly as the
rung B cell predictions were.

It also writes the three falsification rules the session must be read against.
Two of them can invalidate the predictor rather than the prediction:

  * the positive control. `O-pbfit - O-ship` re-derives E68's headline inside
    this session. If it does not land near -3.500 % the session is not
    comparable to E68 and nothing else in it may be quoted.
  * fixed-width round latency. The predictor assumes round latency at a fixed
    verify width does not depend on the depth-price arm, which E68 rung 2
    established to within 0.8 % ON OUR TABLE. It is not automatic on the crown
    table. If `C-ship` and `C-pbfit` disagree by more than 1 % at a fixed
    width, the histogram is no longer the only channel and the prediction is
    void rather than wrong.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e75_rungB_analyze import (  # noqa: E402
    A_PER_ROUND_MS,
    B_PER_ROW_MS,
    CHANGED,
    HIST,
    PREREGISTERED_2X2,
    ROWS,
    decode_ms,
    load,
    mean_sd,
)

TOKENS = 512
OUT = HERE / "e75-artifacts/e75-rungD-prereg.json"

# E68 rung 3, nine legs, mirrored palindrome, the result this session must
# reproduce in its `ours` half.
E68_SHIP_S_PER_TOKEN = 0.031457267
E68_PBFIT_S_PER_TOKEN = 0.030356223
E68_PBFIT_EFFECT_PCT = -3.500
E68_SESSION_NULL_PCT = 0.143

# The ranked measurement of the identical eight-line dispatch-table diff,
# receipt 9b241879, plutarch-corrected scoring-prompt mean. Recorded here only
# so the two harnesses sit in one file; it is NOT converted.
RANKED_TABLE_EFFECT_PCT = -0.298


def main() -> None:
    per_arm, _ = load()
    widths = sorted(per_arm["ours"][0])
    cost = {
        arm: {w: mean_sd([c[w] for c in curves])[0] for w in widths}
        for arm, curves in per_arm.items()
    }

    cells_ms = {
        "O-ship": decode_ms(HIST["ship"], ROWS["ship"], cost["ours"]),
        "O-pbfit": decode_ms(HIST["pbfit"], ROWS["pbfit"], cost["ours"]),
        "C-ship": decode_ms(HIST["ship"], ROWS["ship"], cost["crown"]),
        "C-pbfit": decode_ms(HIST["pbfit"], ROWS["pbfit"], cost["crown"]),
    }
    s_per_token = {k: v / 1000.0 / TOKENS for k, v in cells_ms.items()}

    pbfit_on_ours = (cells_ms["O-pbfit"] - cells_ms["O-ship"]) / cells_ms["O-ship"] * 100
    pbfit_on_crown = (cells_ms["C-pbfit"] - cells_ms["C-ship"]) / cells_ms["C-ship"] * 100
    table_at_ship = (cells_ms["C-ship"] - cells_ms["O-ship"]) / cells_ms["O-ship"] * 100
    table_at_pbfit = (cells_ms["C-pbfit"] - cells_ms["O-pbfit"]) / cells_ms["O-pbfit"] * 100

    payload = {
        "experiment": "E75 rung D",
        "harness": "local",
        "committed_before": "any rung D leg was timed",
        "design": "2x2 within one thermal session, mirrored palindrome "
                  "O-ship, O-pbfit, C-ship, C-pbfit, C-pbfit, C-ship, O-pbfit, O-ship",
        "model": {
            "form": "decode_ms = sum_w n(w) * C(w) + a * rounds + b * rows",
            "a_ms_per_round": A_PER_ROUND_MS,
            "b_ms_per_row": B_PER_ROW_MS,
            "round_histograms": {k: {str(w): n for w, n in v.items()}
                                 for k, v in HIST.items()},
            "rows": ROWS,
            "verify_cost_ms": {arm: {str(w): round(c, 4) for w, c in curve.items()}
                               for arm, curve in cost.items()},
            "changed_cells": list(CHANGED),
            "source_of_crown_curve": "E75 rung B, 8 legs, measured",
            "out_of_sample_error_pct": -0.32,
        },
        "predicted_decode_seconds": {k: round(v / 1000.0, 4) for k, v in cells_ms.items()},
        "predicted_candidate_mtp_seconds_per_token":
            {k: round(v, 9) for k, v in s_per_token.items()},
        "predicted_effects_pct": {
            "pbfit_on_ours": round(pbfit_on_ours, 3),
            "pbfit_on_crown": round(pbfit_on_crown, 3),
            "table_at_ship": round(table_at_ship, 3),
            "table_at_pbfit": round(table_at_pbfit, 3),
            "interaction_pp": round(pbfit_on_crown - pbfit_on_ours, 3),
        },
        "ladder_prediction_superseded": PREREGISTERED_2X2,
        "e68_reference": {
            "ship_seconds_per_token": E68_SHIP_S_PER_TOKEN,
            "pbfit_seconds_per_token": E68_PBFIT_S_PER_TOKEN,
            "pbfit_effect_pct": E68_PBFIT_EFFECT_PCT,
            "session_null_pct": E68_SESSION_NULL_PCT,
        },
        "ranked_table_effect_pct_do_not_convert": RANKED_TABLE_EFFECT_PCT,
        "falsification": {
            "positive_control":
                "O-pbfit - O-ship must land within about 1 pp of -3.500 %. "
                "Outside that, the session is not comparable to E68 and the "
                "C-ship vs O-ship calibration may not be quoted.",
            "predictor_validity":
                "Round latency at a fixed verify width must be arm-independent "
                "to within about 1 % on the crown table. If C-ship and C-pbfit "
                "disagree by more than that at a fixed width, the histogram is "
                "not the only channel and this prediction is void, not wrong.",
            "single_mechanism_calibration":
                "If C-ship and O-ship share a round histogram, the whole local "
                "table effect is per-cell cost with no schedule reaction, and "
                "the pair is a clean single-mechanism calibration of the "
                "harness transfer function.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("E75 rung D pre-registration, written to %s" % OUT.relative_to(HERE.parent))
    print("\n  cell      decode s   s/token")
    for k in ("O-ship", "O-pbfit", "C-ship", "C-pbfit"):
        print("  %-8s  %8.3f  %.9f" % (k, cells_ms[k] / 1000.0, s_per_token[k]))
    print("\n  pbfit on ours   %+7.3f %%   (positive control, E68 measured %.3f %%)"
          % (pbfit_on_ours, E68_PBFIT_EFFECT_PCT))
    print("  pbfit on crown  %+7.3f %%" % pbfit_on_crown)
    print("  table at ship   %+7.3f %%   (harness=local; ranked was %+.3f %%, DO NOT CONVERT)"
          % (table_at_ship, RANKED_TABLE_EFFECT_PCT))
    print("  interaction     %+7.3f pp" % (pbfit_on_crown - pbfit_on_ours))


if __name__ == "__main__":
    main()
