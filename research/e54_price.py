#!/usr/bin/env python3
"""Price E54's measured cells under BOTH live width mixtures.

Timing is the deliverable; this file is only the price. E54 does not adjudicate
between askeladd's E48 corpus mixture and edward's E53 burst envelope, so every
cell is priced under both and the two answers are reported side by side.

The advisor's pricing module is not on this experiment's base, so it is
extracted from a pinned advisor-branch ref rather than copied into this branch:
a fork of that file is exactly how a retracted constant survives.

Everything here is `harness=ranked`. The ranked serial leg is a pinned separate
binary, so `d ln(serial)/dx = 0` and no local serial share is subtracted.

  python3 research/e54_price.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import tempfile

LEVERAGE_REF = "ccd1af6"          # advisor branch tip carrying the order-statistic score
LEVERAGE_PATH = "research/qmv_score_leverage.py"

REPO = pathlib.Path(__file__).resolve().parent.parent
SHARES = REPO / "research/e54-artifacts/e54-shares.json"
METRICS = REPO / "research/e54-artifacts/e54-metrics.json"
PREREG = REPO / "research/e54-artifacts/e54-prereg.json"

# E49's measured cell, carried forward rather than re-run.
M9_WIN_PCT = 12.255
# E27's observed board score change: the composite of the M=5 and M=9 cells.
E27_SCORE_PCT = -0.3321


def _load_leverage():
    text = subprocess.run(["git", "show", f"{LEVERAGE_REF}:{LEVERAGE_PATH}"],
                          capture_output=True, text=True, check=True).stdout
    tmp = pathlib.Path(tempfile.mkdtemp()) / "qmv_score_leverage.py"
    tmp.write_text(text)
    spec = importlib.util.spec_from_file_location("qmv_score_leverage", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def share_points(shares: dict, m: int) -> dict[str, float]:
    """Every share this width takes under the two mixtures, as fractions."""
    out = {"e48": shares["askeladd_e48"]["per_width"][str(m)]}
    env = shares["edward_e53"]["envelope"]["per_width"][str(m)]
    out.update({f"e53_{k}": env[k] for k in ("low", "mid", "high")})
    return out


def price(L, win_pct: float, share_frac: float) -> dict:
    """Score change from removing `win_pct` of the QMV cost carried by a width.

    `win_pct` is oriented as a saving: a positive value is a speedup.
    """
    removed = share_frac * win_pct
    leg = L.PSI_MTP * removed
    gains = {p: leg for p in L.SCORED_PROMPTS}
    return {
        "share_pct": round(100.0 * share_frac, 4),
        "qmv_removed_pct": round(removed, 4),
        "leg_gain_pct": round(leg, 4),
        "score_pct": round(L.score_pct_from_leg_gains(gains), 4),
    }


def measured_wins() -> dict[int, float]:
    """Measured saving per width, positive = faster. M=9 comes from E49."""
    wins = {9: M9_WIN_PCT}
    if not METRICS.exists():
        return wins
    metrics = json.loads(METRICS.read_text())
    for pair, rep in metrics.get("pairs", {}).items():
        if pair == "P4":
            continue
        for m, t in rep.get("treated", {}).items():
            wins[int(m)] = -t["delta_pct"]     # delta is signed; win is a saving
    return wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e54-artifacts/e54-price.json")
    args = ap.parse_args()

    L = _load_leverage()
    shares = json.loads(SHARES.read_text())
    wins = measured_wins()

    out: dict = {
        "harness": "ranked",
        "leverage_ref": LEVERAGE_REF,
        "psi_mtp": L.PSI_MTP,
        "kink_pct": L.kink_pct(),
        "saturation_cap_pct": L.saturation_cap_pct(),
        "marginal_weights": L.marginal_weights(),
        "measured_win_pct_by_width": wins,
        "mixtures_are_not_adjudicated_here": True,
        "per_width": {},
    }

    print(f"module {LEVERAGE_PATH} @ {LEVERAGE_REF}   harness=ranked")
    print(f"  psi_mtp {L.PSI_MTP}   kink {L.kink_pct():.4f} %   "
          f"saturation cap {L.saturation_cap_pct():.4f} %")
    print("\nPER-WIDTH PRICE, both mixtures. positive win = faster candidate")
    print("  %-3s %-9s %-10s %-11s %-11s %-11s %-11s"
          % ("M", "win %", "mixture", "share %", "QMV rm %", "leg %", "score %"))
    for m in sorted(wins):
        pts = share_points(shares, m)
        out["per_width"][m] = {"win_pct": wins[m], "prices": {}}
        for label, frac in pts.items():
            p = price(L, wins[m], frac)
            out["per_width"][m]["prices"][label] = p
            print("  %-3d %-9.3f %-10s %-11.4f %-11.4f %-11.4f %+11.4f"
                  % (m, wins[m], label, p["share_pct"], p["qmv_removed_pct"],
                     p["leg_gain_pct"], p["score_pct"]))

    # E27 is the composite of the M=5 and M=9 cells on the real table, so it is
    # priced as one combined removal rather than as two independent scores.
    if 5 in wins:
        print("\nE27 COMPOSITE (M=5 and M=9 together), against the board anchor")
        out["e27_composite"] = {}
        combos = {
            "e48": (shares["askeladd_e48"]["per_width"]["5"],
                    shares["askeladd_e48"]["per_width"]["9"]),
        }
        env = shares["edward_e53"]["envelope"]["per_width"]
        for k in ("low", "mid", "high"):
            combos[f"e53_{k}"] = (env["5"][k], env["9"][k])
        print("  %-10s %-11s %-11s %-11s %-11s"
              % ("mixture", "QMV rm %", "leg %", "score %", "vs board"))
        for label, (s5, s9) in combos.items():
            removed = s5 * wins[5] + s9 * wins[9]
            leg = L.PSI_MTP * removed
            score = L.score_pct_from_leg_gains({p: leg for p in L.SCORED_PROMPTS})
            out["e27_composite"][label] = {
                "share_pct_m5": round(100.0 * s5, 4),
                "share_pct_m9": round(100.0 * s9, 4),
                "qmv_removed_pct": round(removed, 4),
                "leg_gain_pct": round(leg, 4),
                "score_pct": round(score, 4),
                "board_observed_pct": E27_SCORE_PCT,
                "residual_pct": round(E27_SCORE_PCT - score, 4),
                "sign_matches_board": bool((score < 0) == (E27_SCORE_PCT < 0)),
            }
            r = out["e27_composite"][label]
            print("  %-10s %-11.4f %-11.4f %+11.4f %+11.4f %s"
                  % (label, r["qmv_removed_pct"], r["leg_gain_pct"],
                     r["score_pct"], r["residual_pct"],
                     "sign OK" if r["sign_matches_board"] else "SIGN DIFFERS"))
        print(f"  board observed {E27_SCORE_PCT:+.4f} % (trusted)")

    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
