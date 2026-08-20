#!/usr/bin/env python3
"""Project the measured M=6 cell delta onto the whole decode leg.

Rung 1 measured what the single-stream M=6 cell costs. The promotion rule is
written on whole-leg candidate MTP seconds per token, so the cell delta has to
be carried through two factors:

  f6       the share of local QMV verify time that width 6 accounts for,
           recomputed here from THIS session's measured `shipped` per-width
           costs rather than carried over from E54; and
  psi_mtp  E55's preregistered transfer constant from QMV time to MTP-leg time,
           0.693391, with E55's measured realisation factor 0.946 applied
           afterwards.

The ranked projection uses the assignment's ranked M=6 share band, 30.9-34.7 %,
because no per-width ranked mixture exists on this base. `research/e53-width-
mixture.json` gives bucket fractions (f456, f78, f9, f123), not per-width ones,
so it corroborates that widths 4-6 dominate at rank but cannot supply f6 alone.

  python3 research/e61_project.py --out research/e61-artifacts/e61-projection.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# Deterministic across all six E55 legs at 512 tokens (research/e61-prereg.md).
LOCAL_ROUNDS = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
# E55 preregistered transfer constant and its measured realisation factor.
PSI_MTP = 0.693391
REALISATION = 0.946
# Assignment: M=6 share of ranked QMV time.
RANKED_F6_BAND = (0.309, 0.347)
# research/e61-prereg.md.
LOCAL_NULL_FLOOR_PCT = 0.0629
PROMOTE_PCT = -0.30
REPORT_ONLY_PCT = 0.10
PREREG_F6 = 0.2671
PREREG_CELL_DELTA_PCT = -9.95
PREREG_WHOLE_LEG_PCT = -1.84


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth", default="research/e61-artifacts/e61-bandwidth.json")
    ap.add_argument("--out", default="research/e61-artifacts/e61-projection.json")
    args = ap.parse_args()

    bw = json.loads((REPO / args.bandwidth).read_text())
    shipped = bw["per_arm_per_width"]["shipped"]
    cell = bw["cell_deltas"]["t6"]["cells"]

    # Time-weighted share of local QMV verify time per width, on the shipped
    # table, using this session's own per-width costs.
    weighted = {m: LOCAL_ROUNDS[m] * shipped[str(m)]["weighted_seconds_per_verify"]
                for m in LOCAL_ROUNDS if str(m) in shipped}
    total = sum(weighted.values())
    shares = {m: w / total for m, w in weighted.items()}
    f6 = shares[6]

    m6_delta_pct = cell["6"]["seconds_delta_pct"]

    qmv_delta_pct = f6 * m6_delta_pct
    leg_delta_pct = PSI_MTP * qmv_delta_pct
    leg_delta_realised_pct = leg_delta_pct * REALISATION

    ranked = {}
    for name, f in (("low", RANKED_F6_BAND[0]), ("high", RANKED_F6_BAND[1])):
        q = f * m6_delta_pct
        ranked[name] = {
            "f6": f,
            "qmv_delta_pct": q,
            "leg_delta_pct": PSI_MTP * q,
            "leg_delta_realised_pct": PSI_MTP * q * REALISATION,
        }

    verdict = ("promote" if leg_delta_realised_pct <= PROMOTE_PCT
               else "report_only" if leg_delta_realised_pct < REPORT_ONLY_PCT
               else "stop")

    out = {
        "inputs": {
            "local_rounds": LOCAL_ROUNDS,
            "psi_mtp": PSI_MTP,
            "realisation_factor": REALISATION,
            "measured_m6_cell_delta_pct": m6_delta_pct,
            "shipped_seconds_per_verify": {
                m: shipped[str(m)]["weighted_seconds_per_verify"] for m in LOCAL_ROUNDS
                if str(m) in shipped},
        },
        "local_time_weighted_shares": shares,
        "local": {
            "f6": f6,
            "f6_prereg": PREREG_F6,
            "qmv_delta_pct": qmv_delta_pct,
            "leg_delta_pct": leg_delta_pct,
            "leg_delta_realised_pct": leg_delta_realised_pct,
            "multiple_of_null_floor": abs(leg_delta_realised_pct) / LOCAL_NULL_FLOOR_PCT,
        },
        "ranked": ranked,
        "prereg_comparison": {
            "prereg_cell_delta_pct": PREREG_CELL_DELTA_PCT,
            "measured_cell_delta_pct": m6_delta_pct,
            "cell_realisation_ratio": m6_delta_pct / PREREG_CELL_DELTA_PCT,
            "prereg_whole_leg_pct": PREREG_WHOLE_LEG_PCT,
            "revised_whole_leg_pct": leg_delta_realised_pct,
        },
        "decision_bands": {
            "promote_at_or_below_pct": PROMOTE_PCT,
            "report_only_below_pct": REPORT_ONLY_PCT,
            "local_null_floor_pct": LOCAL_NULL_FLOOR_PCT,
        },
        "predicted_verdict_if_realised": verdict,
        "caveat": ("This is a projection from a microbenchmark cell, not a "
                   "measurement. Rung 3 measures the whole leg directly and "
                   "that measurement decides."),
    }

    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("local time-weighted QMV shares, this session's shipped costs:")
    for m, s in sorted(shares.items()):
        mark = "  <- treated" if m == 6 else ""
        print("   M=%d  %6.2f %%%s" % (m, 100 * s, mark))
    print("\nf6 = %.4f (prereg %.4f, from E54 costs)" % (f6, PREREG_F6))
    print("measured M=6 cell delta = %+.2f %% (prereg model %+.2f %%, realised %.2fx)"
          % (m6_delta_pct, PREREG_CELL_DELTA_PCT, m6_delta_pct / PREREG_CELL_DELTA_PCT))
    print("\nlocal whole-leg projection")
    print("   QMV time      %+.3f %%" % qmv_delta_pct)
    print("   MTP leg       %+.3f %%   (x psi_mtp %.6f)" % (leg_delta_pct, PSI_MTP))
    print("   after E55 realisation %.3f: %+.3f %%   (%.1fx the %.4f %% null floor)"
          % (REALISATION, leg_delta_realised_pct,
             abs(leg_delta_realised_pct) / LOCAL_NULL_FLOOR_PCT, LOCAL_NULL_FLOOR_PCT))
    print("\nranked projection, assignment f6 band %.1f-%.1f %%"
          % (100 * RANKED_F6_BAND[0], 100 * RANKED_F6_BAND[1]))
    for name in ("low", "high"):
        print("   f6=%.3f -> MTP leg %+.3f %% realised"
              % (ranked[name]["f6"], ranked[name]["leg_delta_realised_pct"]))
    print("\npromote at <= %.2f %%, report-only below %+.2f %% -> projection says %s"
          % (PROMOTE_PCT, REPORT_ONLY_PCT, verdict))
    print("wrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
