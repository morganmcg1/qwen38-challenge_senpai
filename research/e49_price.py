#!/usr/bin/env python3
"""Convert E49's measured times into ranked score, using the advisor's module.

Timing is the deliverable; this file is only the price. The advisor's pricing
module is not on this experiment's base (`fb0a09d`), so it is extracted from a
pinned advisor-branch ref rather than copied into this branch: a fork of that
file is exactly how a retracted constant survives.

Everything here is `harness=ranked`. The ranked serial leg is a pinned separate
binary (E50), so `d ln(serial)/dx = 0` and no local serial share is subtracted.

  python3 research/e49_price.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import tempfile

LEVERAGE_REF = "ccd1af6"          # advisor branch tip carrying the order-statistic score
LEVERAGE_PATH = "research/qmv_score_leverage.py"

# Arm 1, measured: isolated <T,9,5> vs <T,9,3>, 4 ABBA legs, 21 reps.
M9_WIN_PCT = 12.255
# Arm 2, measured: worst pooled tax over untouched widths, shipped-referenced,
# and the control-free dose_null-referenced contrast.
CEILING_TAX_PCT = 0.213
CEILING_TAX_CONTROL_FREE_PCT = 0.130
# E27's observed board score change and its register delta (108 -> 129).
E27_SCORE_PCT = -0.3321
# M=9 share of scored candidate-leg QMV cost. 20.48 % is E48 Part 1 as relayed
# by the advisor; it is a moving number computed at weights the advisor has
# since corrected, so it is quoted with a sensitivity band, never hard-coded.
M9_SHARE_PCT = 20.48
SHARE_BAND_PCT = (15.0, 20.48, 30.0, 53.45)


def _load_leverage():
    text = subprocess.run(["git", "show", f"{LEVERAGE_REF}:{LEVERAGE_PATH}"],
                          capture_output=True, text=True, check=True).stdout
    tmp = pathlib.Path(tempfile.mkdtemp()) / "qmv_score_leverage.py"
    tmp.write_text(text)
    spec = importlib.util.spec_from_file_location("qmv_score_leverage", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    L = _load_leverage()
    both = lambda pct: {p: pct for p in L.SCORED_PROMPTS}  # noqa: E731
    out: dict = {"harness": "ranked", "leverage_ref": LEVERAGE_REF,
                 "measured": {"m9_win_pct": M9_WIN_PCT,
                              "ceiling_tax_pct": CEILING_TAX_PCT,
                              "ceiling_tax_control_free_pct": CEILING_TAX_CONTROL_FREE_PCT}}

    print(f"module {LEVERAGE_PATH} @ {LEVERAGE_REF}   harness=ranked")
    print(f"  psi_mtp {L.PSI_MTP}   kink {L.kink_pct():.4f} %   "
          f"saturation cap {L.saturation_cap_pct():.4f} %")
    weights = L.marginal_weights()
    print("  marginal weights " + "  ".join(f"{k} {v:.6f}" for k, v in weights.items()))
    out["kink_pct"] = L.kink_pct()
    out["saturation_cap_pct"] = L.saturation_cap_pct()
    out["marginal_weights"] = weights

    print("\nthe advisor's +10.6 % ceiling, checked against the score function")
    print(f"  target_for(10.6) -> {L.target_for(10.6)}   (None = above the cap)")
    out["target_for_10_6"] = L.target_for(10.6)

    print(f"\nARM 1 PRIZE: a {M9_WIN_PCT:.3f} % win at M=9, per share of scored QMV cost")
    print("  share %   QMV removed %   leg gain %   score %")
    out["prize_by_share"] = {}
    for share in SHARE_BAND_PCT:
        removed = share / 100.0 * M9_WIN_PCT
        leg = L.PSI_MTP * removed
        score = L.score_pct_from_leg_gains(both(leg))
        flag = "  <- E48 P1, relayed" if share == M9_SHARE_PCT else ""
        print(f"  {share:6.2f}   {removed:11.3f}   {leg:10.3f}   {score:+7.4f}{flag}")
        out["prize_by_share"][share] = {"qmv_removed_pct": removed,
                                        "leg_gain_pct": leg, "score_pct": score}
    prize = out["prize_by_share"][M9_SHARE_PCT]["score_pct"]

    print("\nARM 2 CEILING: a UNIFORM slowdown on widths whose code did not change")
    out["ceiling"] = {}
    for label, tax in (("shipped-referenced", CEILING_TAX_PCT),
                       ("control-free", CEILING_TAX_CONTROL_FREE_PCT)):
        leg = -L.PSI_MTP * tax
        score = L.score_pct_from_leg_gains(both(leg))
        print(f"  {label:20s} tax {tax:.3f} % of QMV  ->  leg {leg:+.4f} %  "
              f"->  score {score:+.4f} %")
        out["ceiling"][label] = {"tax_pct": tax, "leg_pct": leg, "score_pct": score}
    bound = abs(out["ceiling"]["shipped-referenced"]["score_pct"])

    print("\nE27 RECONCILIATION (board anchor vs this experiment)")
    residual = E27_SCORE_PCT - prize
    print(f"  E27 observed score change        {E27_SCORE_PCT:+.4f} %  (board, trusted)")
    print(f"  E49 M=9 half, priced             {prize:+.4f} %")
    print(f"  residual                         {residual:+.4f} %")
    print(f"  E49 upper bound on the ceiling    {bound:.4f} %")
    print(f"  still unexplained                {residual + bound:+.4f} %"
          f"   (ceiling explains at most {100.0 * bound / abs(residual):.1f} %)")
    out["e27"] = {"observed_score_pct": E27_SCORE_PCT, "e49_m9_score_pct": prize,
                  "residual_pct": residual, "ceiling_bound_pct": bound,
                  "unexplained_pct": residual + bound}

    print("\n  to close it, E27's UNMEASURED M=5 half must ADD QMV cost:")
    need_leg = -residual / L.PSI_MTP        # % of QMV cost the M=5 half must add
    out["e27"]["m5_must_add_qmv_pct"] = need_leg
    print(f"    required added QMV cost {need_leg:+.3f} %")
    out["e27"]["m5_requirement"] = {}
    for share in (10.0, 15.0, 20.0, 30.0):
        slower = need_leg / (share / 100.0)
        out["e27"]["m5_requirement"][share] = slower
        print(f"    if M=5 is {share:4.1f} % of scored QMV cost, <T,5,5> must be "
              f"{slower:+.1f} % SLOWER than <T,5,3>")

    dest = pathlib.Path("research/e49-artifacts/e49-price.json")
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
