#!/usr/bin/env python3
"""Post-registration corrections to E38's sizing constants.

research/e38_prereg.py is frozen: it was committed before the kernel existed and
must keep saying what it said.  Advisor comment 5337633069 then corrected two of
the constants it used.  This module applies those corrections on top rather than
editing the pre-registration, so both numbers stay auditable.

  sigma_score  0.0923 % -> 0.0978 %   (second route: the serial leg is a
                                       nominally identical computation on all 8
                                       prompts => 94 quasi-replicates)
  engineerable 0.561 %  -> 0.2586 %   (the +0.5193 % crown gap splits almost
                                       exactly in half; +0.2600 % of it is the
                                       rival's SLOWER SERIAL LEG, which is the
                                       score denominator and is not engineerable
                                       -- degrading our own serial leg to chase
                                       it would be metric gaming)

  python3 research/e38_corrections.py [--self-test]
"""

from __future__ import annotations

import sys

# --- corrected campaign constants (advisor 5337633069) -----------------------
SIGMA_SCORE_PCT = 0.0978
CROWN_GAP_PCT = 0.5193
SERIAL_LEG_PCT = 0.2600      # denominator, not engineerable
ENGINEERABLE_PCT = 0.2586    # the MTP-leg half we can actually attack

# --- score chain (unchanged; provenance in advisor 5337266846) ---------------
SCORE_CHAIN = 0.4827         # d(score)/score per unit relative gain in beagle raw_p
PHI = 0.201                  # M=6 share of QMV cost   (this fixture)
PSI = 0.228                  # QMV share of candidate-leg wall (measured, E33)

# E2E instrument resolution on the decode leg, measured in E33.
E2E_RESOLUTION_PCT = 0.3

# Order statistics: score is the mean of the 4th and 5th of eight per-prompt
# ratios.  Ours sort with beagle 4th and medicine 5th, so only those two carry
# any score weight at all.
SCORING_PROMPTS = ("beagle", "medicine")


def leg_pct(ratio: float) -> float:
    """Predicted MTP-leg movement, percent, from an M=6 per-row cost ratio."""
    return PSI * PHI * (1.0 - ratio) * 100.0


def score_pct(ratio: float) -> float:
    """Predicted published-score movement, percent."""
    return SCORE_CHAIN * leg_pct(ratio)


def sigmas(ratio: float) -> float:
    return score_pct(ratio) / SIGMA_SCORE_PCT


def ratio_for_score(target_pct: float) -> float:
    """The M=6 cost ratio needed to buy `target_pct` of published score."""
    return 1.0 - target_pct / (SCORE_CHAIN * PSI * PHI * 100.0)


def mde_ratio() -> float:
    """Smallest M=6 ratio whose predicted leg movement the E2E rig can resolve."""
    return 1.0 - E2E_RESOLUTION_PCT / (PSI * PHI * 100.0)


THRESHOLDS = {
    "1 sigma of score": ratio_for_score(SIGMA_SCORE_PCT),
    "2 sigma of score": ratio_for_score(2 * SIGMA_SCORE_PCT),
    "engineerable gap": ratio_for_score(ENGINEERABLE_PCT),
    "full crown gap": ratio_for_score(CROWN_GAP_PCT),
    "E2E resolvable (MDE)": mde_ratio(),
}


def report(measured: float) -> str:
    out = [
        "E38 sizing under the corrected constants (advisor 5337633069)",
        f"  sigma_score      {SIGMA_SCORE_PCT:.4f} %  (was 0.0923 %)",
        f"  engineerable gap {ENGINEERABLE_PCT:.4f} %  (was 0.561 %; half of"
        f" the {CROWN_GAP_PCT:.4f} % crown gap is their slower serial leg)",
        f"  chain            score = {SCORE_CHAIN} x psi {PSI} x phi {PHI} x x",
        "",
        "  ratio needed to buy:",
    ]
    for name, r in THRESHOLDS.items():
        out.append(f"    {name:<22} ratio <= {r:.4f}  (x = {(1-r)*100:5.2f} %)")
    out += [
        "",
        f"  MEASURED ratio {measured:.4f}  =>  x = {(1-measured)*100:.2f} %",
        f"    predicted MTP-leg movement {leg_pct(measured):+.4f} %",
        f"    predicted score movement   {score_pct(measured):+.4f} %"
        f"  = {sigmas(measured):.2f} sigma",
        f"    fraction of the engineerable gap closed:"
        f" {100*score_pct(measured)/ENGINEERABLE_PCT:.1f} %",
        "",
        f"  E2E: predicted leg {leg_pct(measured):+.4f} % against a"
        f" +/-{E2E_RESOLUTION_PCT} % instrument",
        f"       => under-powered by {E2E_RESOLUTION_PCT/abs(leg_pct(measured)):.1f}x;"
        f" the rig could only have detected ratio <= {mde_ratio():.4f}",
    ]
    return "\n".join(out)


def _self_test() -> None:
    # The advisor's own worked example must reproduce: ratio 0.840 with psi 0.4
    # gives leg -1.29 % and score +0.62 %.
    leg = 0.4 * PHI * (1 - 0.840) * 100
    assert abs(leg - 1.29) < 0.01, leg
    assert abs(SCORE_CHAIN * leg - 0.62) < 0.01, SCORE_CHAIN * leg
    # ratio_for_score must invert score_pct.
    for t in (0.05, 0.2586, 0.5193):
        assert abs(score_pct(ratio_for_score(t)) - t) < 1e-9
    # The halves of the crown gap must add up.
    assert abs(SERIAL_LEG_PCT + ENGINEERABLE_PCT - CROWN_GAP_PCT) < 0.002
    print("e38 corrections self-test OK")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        arg = [a for a in sys.argv[1:] if not a.startswith("--")]
        print(report(float(arg[0]) if arg else 0.9891))
