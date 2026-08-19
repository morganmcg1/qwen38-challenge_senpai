#!/usr/bin/env python3
"""Pre-registered addendum: what must M=5 do to explain E27's board score?

E27 changed two cells on the real table, `case 5 -> <T,5,5>` and
`case 9 -> <T,9,5>`, and the board moved -0.3321 %. E49 measured the M=9 half in
isolation and it WON 12.255 %. So the M=9 half alone predicts a gain, and
something else must pay for the observed loss.

This inverts the pricing. Given the measured M=9 win and a width mixture, it
solves for the M=5 delta that reproduces the board number exactly. That turns
E27's board result into a falsifiable prediction about P1 BEFORE P1 is measured:

  if measured P1 is close to the solved value, the two cells explain E27
  if measured P1 is a large WIN instead, the two cells CANNOT explain E27, and
    the residual belongs to another mechanism, most likely the shared
    [[kernel]] allocation tax that E49 bounded

Committed before any timed leg, like the main pre-registration.

  python3 research/e54_reconcile.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from e54_prereg import BW_NA5, BW_NA_LE4
from e54_price import E27_SCORE_PCT, M9_WIN_PCT, SHARES, _load_leverage


def implied_gbps(m5_delta_pct: float) -> float:
    """Bandwidth the LONE NA=5 group must sustain to be `m5_delta_pct` slower.

    `<T,5,3>` reads the whole matrix twice at the NA<=4 rate; `<T,5,5>` reads it
    once. Halving the traffic and still losing time pins the rate.
    """
    return BW_NA_LE4 / (2.0 * (1.0 + m5_delta_pct / 100.0))


def solve_m5_delta(L, s5: float, s9: float, target_score: float) -> float | None:
    """M=5 delta percent that makes the composite price hit `target_score`.

    Returned with the same sign convention as the measurement: negative means
    `<T,5,5>` is FASTER than `<T,5,3>`, positive means it is slower.
    """
    def score_of(delta5: float) -> float:
        # A positive delta is a slowdown, so it removes negative QMV cost.
        removed = s5 * (-delta5) + s9 * M9_WIN_PCT
        leg = L.PSI_MTP * removed
        return L.score_pct_from_leg_gains({p: leg for p in L.SCORED_PROMPTS})

    lo, hi = -200.0, 400.0
    if not (min(score_of(hi), score_of(lo)) <= target_score
            <= max(score_of(lo), score_of(hi))):
        return None
    # score_of is monotone decreasing in delta5: a slower M=5 costs score.
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if score_of(mid) > target_score:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e54-artifacts/e54-reconcile.json")
    args = ap.parse_args()

    L = _load_leverage()
    shares = json.loads((REPO / SHARES).read_text())

    mixes = {"e48": (shares["askeladd_e48"]["per_width"]["5"],
                     shares["askeladd_e48"]["per_width"]["9"])}
    env = shares["edward_e53"]["envelope"]["per_width"]
    for k in ("low", "mid", "high"):
        mixes[f"e53_{k}"] = (env["5"][k], env["9"][k])

    out: dict = {
        "harness": "ranked",
        "board_observed_pct": E27_SCORE_PCT,
        "m9_win_pct_measured_e49": M9_WIN_PCT,
        "psi_mtp": L.PSI_MTP,
        "question": "what M=5 delta reproduces E27's board score, given the "
                    "measured M=9 win?",
        "sign_convention": "negative = <T,5,5> faster than <T,5,3>",
        "solved": {},
    }

    print("E27 RECONCILIATION, inverted. board %+.4f %%, M=9 measured %+.3f %% win"
          % (E27_SCORE_PCT, M9_WIN_PCT))
    print("  %-10s %-11s %-11s %-14s %-16s %s"
          % ("mixture", "share M5 %", "share M9 %", "M=5 must be",
             "implied GB/s", "M=9 half alone"))
    for label, (s5, s9) in mixes.items():
        need = solve_m5_delta(L, s5, s9, E27_SCORE_PCT)
        leg9 = L.PSI_MTP * s9 * M9_WIN_PCT
        alone = L.score_pct_from_leg_gains({p: leg9 for p in L.SCORED_PROMPTS})
        out["solved"][label] = {
            "share_pct_m5": round(100.0 * s5, 4),
            "share_pct_m9": round(100.0 * s9, 4),
            "m5_delta_pct_required": None if need is None else round(need, 3),
            "implied_lone_group_gbps": (
                None if need is None else round(implied_gbps(need), 2)),
            "m9_half_alone_score_pct": round(alone, 4),
            "reachable": need is not None,
        }
        shown = "unreachable" if need is None else "%+.2f %%" % need
        gbps = ("-" if need is None
                else "%.1f (%.0f %% low)" % (implied_gbps(need),
                                             100.0 * (1.0 - implied_gbps(need) / BW_NA5)))
        print("  %-10s %-11.4f %-11.4f %-14s %-16s %+.4f %%"
              % (label, 100.0 * s5, 100.0 * s9, shown, gbps, alone))
    out["bandwidth_reference"] = {
        "bw_na5_measured_pr8_gbps": BW_NA5,
        "bw_na_le4_measured_pr8_gbps": BW_NA_LE4,
        "note": "<T,5,3> reads the matrix twice at the NA<=4 rate and <T,5,5> "
                "reads it once, so a required M=5 SLOWDOWN pins the bandwidth "
                "the lone NA=5 group would have to sustain. Every value below "
                "the measured 95.5 GB/s is a claim that a lone group is slower "
                "than the only NA=5 bandwidth measurement we have.",
    }

    # The pre-registered discriminator, written down before P1 is measured.
    out["prereg_discriminator"] = {
        "if_p1_matches_solved_within_3pct": "the two cells explain E27; Law C "
                                            "gains strong support",
        "if_p1_is_a_win_beyond_the_bar": "the two cells CANNOT explain E27 under "
                                         "any mixture, because both halves would "
                                         "then predict a GAIN while the board "
                                         "lost. The residual must come from a "
                                         "mechanism outside these two cells, and "
                                         "the shared [[kernel]] allocation tax is "
                                         "the leading candidate",
        "note": "E49 bounded that tax at 0.213 % of QMV cost shipped-referenced, "
                "which is far too small to cover the board loss on its own, so a "
                "P1 win would leave E27 genuinely unexplained and worth its own "
                "experiment",
    }
    print("\nPRE-REGISTERED DISCRIMINATOR")
    print("  P1 near the solved value      -> the two cells explain E27, Law C supported")
    print("  P1 a win beyond the bar       -> the two cells CANNOT explain E27;")
    print("                                   residual belongs to another mechanism")

    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print("\nwrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
