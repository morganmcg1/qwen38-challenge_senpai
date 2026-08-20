#!/usr/bin/env python3
"""E66 rung 4: convert the measured local leg effects to a ranked projection.

Two rules from the ledger govern this file.

202(C): project a kernel win to rank as `L_local * (f_ranked / f_local)`. Never
use `psi` as a standalone factor. `psi_mtp_ranked_leg = 0.82127` is retracted as
a leg-conversion factor and is not used here.

202(J): width histograms are host- and session-specific. `f5_local` and
`f6_local` come from THIS session's own round census, read out of the rung 2 row
ledger, never from another session's histogram.

`f5_ranked` and `f6_ranked` are QMV TIME shares by verify width, so a
commensurate `f_local` must also be time weighted. The round census alone gives
a COUNT share. Both are reported. The time weighting needs a per-width cell cost
and this session did not measure one, so the E61 cost curve is used for that one
purpose and is labelled at every use. It was measured on this same host at the
pre-`t6` table, which is arm A of this experiment. It is a cost model, not a
divisor, and 202(J)'s rule is about divisors.

  python3 research/e66_project.py --out research/e66-projection.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

# Ranked QMV time share by verify width. The assignment names f5 and f6; the
# rest of the row comes from the same ledger 199/200 beagle midpoints.
RANKED_SHARE = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334,
                7: 0.122, 8: 0.0735, 9: 0.0575}

# The independent second estimate of the same quantity: E53's fitted ranked
# width mixture, time weighted. Its six variants bracket the ranked share.
# Source: research/e61-artifacts/e61-projection.json, "ranked_e53".
E53_TIME_SHARES = {
    "A_flat": {5: 0.26200570660585193, 6: 0.5351718101034941},
    "B_decay_098": {5: 0.28186613153632833, 6: 0.5235419334488544},
    "C_decay_095": {5: 0.3056519636108851, 6: 0.5096885547829557},
    "D_margin_8": {5: 0.2000148817099925, 6: 0.5176178752412137},
    "E_margin_4": {5: 0.1780607179162684, 6: 0.4738809892832299},
    "F_margin_2": {5: 0.1807247955570762, 6: 0.4048274533767755},
}

# Per-width cell cost measured on THIS host by E61's `--curve` sweep at the
# pre-t6 table, which is arm A here. Used only to time weight the local width
# census. Source: research/e61-artifacts/e61-projection.json,
# "inputs.shipped_seconds_per_verify".
E61_SECONDS_PER_VERIFY_THIS_HOST = {
    2: 0.0662173643, 4: 0.0821751763, 5: 0.11995656605, 6: 0.12840456985,
    7: 0.138224612, 8: 0.1487185918, 9: 0.1638910192,
}

# Ledger 201(D): the board-anchored flat law, 83 ranked runs across 13
# contrasts. One weight-stream removal, as a fraction of the ranked candidate
# leg, roughly flat in width.
FLAT_LAW_PCT = -0.639
FLAT_LAW_SE_PCT = 0.313


def time_weighted(rounds: dict[int, int]) -> dict[int, float]:
    cost = {m: rounds.get(m, 0) * E61_SECONDS_PER_VERIFY_THIS_HOST.get(m, 0.0)
            for m in rounds}
    total = sum(cost.values())
    return {m: c / total for m, c in sorted(cost.items())}


def count_share(rounds: dict[int, int]) -> dict[int, float]:
    total = sum(rounds.values())
    return {m: c / total for m, c in sorted(rounds.items())}


def project(local_pct: float, f_local: float, f_ranked: float) -> float:
    """Ledger 202(C): ranked_leg_delta = L_local * (f_ranked / f_local)."""
    return local_pct * (f_ranked / f_local)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung3", default="research/e66-rung3.json")
    ap.add_argument("--exactness", default="research/e66-exactness.json")
    ap.add_argument("--out", default="research/e66-projection.json")
    args = ap.parse_args()

    rung3 = json.loads(pathlib.Path(args.rung3).read_text())
    exact = json.loads(pathlib.Path(args.exactness).read_text())

    census = {int(k): v for k, v in
              exact["width_census_candidate"]["rounds_by_width"].items()}
    counts = count_share(census)
    times = time_weighted(census)

    measured = rung3["measured_contrasts_pct"]
    # t6 alone is B - A and lives in the M=6 cell; t55 alone is the incremental
    # C - B and lives in the M=5 cell.
    mech = {
        "t6": {"local_pct": measured["B_minus_A"], "width": 6,
               "cell": "<T,6,3> -> <T,6,6>"},
        "t55": {"local_pct": measured["C_minus_B"], "width": 5,
                "cell": "<T,5,3> -> <T,5,5>"},
    }

    # Measured round-fraction-of-leg, this session, this host. Reported because
    # 202(J) requires the session's own value in place of the retracted
    # psi_mtp_ranked_leg constant. It does not enter the psi-free projection.
    timed = [l for l in rung3["legs"] if not l["discarded"]]
    frac = {}
    for arm in ("a_neither", "b_t6", "c_t55_t6"):
        legs = [l for l in timed if l["arm"] == arm]
        frac[arm] = {
            "p50_round_seconds_over_seconds_per_token": statistics.fmean(
                l["mtp_p50_block_seconds"] / l["mtp_seconds_per_token"]
                for l in legs),
            "round_seconds_share_of_decode": statistics.fmean(
                l["mtp_round_count"] * l["mtp_p50_block_seconds"]
                / l["mtp_decode_seconds"] for l in legs),
            "mtp_round_count": statistics.fmean(l["mtp_round_count"] for l in legs),
        }

    out_mech = {}
    for name, spec in mech.items():
        w = spec["width"]
        f_local_time = times.get(w)
        f_local_count = counts.get(w)
        f_ranked = RANKED_SHARE[w]
        e53 = [v[w] for v in E53_TIME_SHARES.values()]
        band = sorted({f_ranked, min(e53), max(e53)})
        out_mech[name] = {
            "cell": spec["cell"],
            "local_leg_pct": spec["local_pct"],
            "f_local_time_weighted": f_local_time,
            "f_local_count": f_local_count,
            "f_ranked_headline": f_ranked,
            "f_ranked_e53_min": min(e53),
            "f_ranked_e53_max": max(e53),
            "ranked_leg_pct_headline": project(spec["local_pct"], f_local_time, f_ranked),
            "ranked_leg_pct_low": project(spec["local_pct"], f_local_time, band[0]),
            "ranked_leg_pct_high": project(spec["local_pct"], f_local_time, band[-1]),
            "ranked_leg_pct_headline_count_basis":
                project(spec["local_pct"], f_local_count, f_ranked),
        }

    comp_headline = sum(v["ranked_leg_pct_headline"] for v in out_mech.values())
    comp_low = sum(v["ranked_leg_pct_low"] for v in out_mech.values())
    comp_high = sum(v["ranked_leg_pct_high"] for v in out_mech.values())
    comp_count = sum(v["ranked_leg_pct_headline_count_basis"]
                     for v in out_mech.values())

    flat_two = 2 * FLAT_LAW_PCT
    flat_two_se = 2 * FLAT_LAW_SE_PCT
    gap = comp_headline - flat_two
    agrees = abs(gap) <= (flat_two_se ** 2 + (0.2 * abs(comp_headline)) ** 2) ** 0.5

    report = {
        "experiment": "qwen38-r1-e66-composition-certification",
        "rung": 4,
        "law": "ledger 202(C): ranked_leg_delta = L_local * (f_ranked / f_local)",
        "psi_used": False,
        "retracted_constants_not_used": ["psi_mtp_ranked_leg = 0.82127",
                                         "LOCAL_NULL_FLOOR_PCT = 0.0629"],
        "session_round_census": census,
        "session_round_count": sum(census.values()),
        "f_local_count_share": counts,
        "f_local_time_weighted_share": times,
        "time_weighting_basis": {
            "seconds_per_verify": E61_SECONDS_PER_VERIFY_THIS_HOST,
            "provenance": ("E61 --curve sweep, this host, pre-t6 table = arm A. "
                           "Used only to time weight THIS session's own census, "
                           "because f_ranked is a time share and the census "
                           "alone is a count share."),
        },
        "measured_round_fraction_of_leg": frac,
        "mechanisms": out_mech,
        "composition": {
            "ranked_leg_pct_headline": comp_headline,
            "ranked_leg_pct_low": comp_low,
            "ranked_leg_pct_high": comp_high,
            "ranked_leg_pct_count_basis": comp_count,
            "method": "sum of the two per-mechanism projections; each mechanism "
                      "is projected with its own width share",
        },
        "cross_check_ledger_201_flat_law": {
            "one_stream_removal_pct": FLAT_LAW_PCT,
            "one_stream_removal_se_pct": FLAT_LAW_SE_PCT,
            "two_removals_pct": flat_two,
            "two_removals_se_pct": flat_two_se,
            "gap_vs_202c_projection_pct": gap,
            "agrees_within_combined_uncertainty": agrees,
        },
    }

    pathlib.Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("session round census: %s  (%d rounds)"
          % (census, sum(census.values())))
    print("f_local count share      f5=%.6f f6=%.6f"
          % (counts.get(5, 0), counts.get(6, 0)))
    print("f_local time weighted    f5=%.6f f6=%.6f"
          % (times.get(5, 0), times.get(6, 0)))
    print("f_ranked (assignment)    f5=%.4f f6=%.4f"
          % (RANKED_SHARE[5], RANKED_SHARE[6]))
    print()
    for arm, v in frac.items():
        print("%-11s round cost / s-per-token %.4f   round share of decode %.4f "
              "(%.1f rounds)"
              % (arm, v["p50_round_seconds_over_seconds_per_token"],
                 v["round_seconds_share_of_decode"], v["mtp_round_count"]))
    print()
    for name, v in out_mech.items():
        print("%-5s local %+.4f %%  ->  ranked %+.4f %%  (range %+.4f .. %+.4f, "
              "count basis %+.4f)"
              % (name, v["local_leg_pct"], v["ranked_leg_pct_headline"],
                 v["ranked_leg_pct_high"], v["ranked_leg_pct_low"],
                 v["ranked_leg_pct_headline_count_basis"]))
    print("comp  ranked %+.4f %%  (range %+.4f .. %+.4f, count basis %+.4f)"
          % (comp_headline, comp_high, comp_low, comp_count))
    print("flat law (201) two removals %+.4f %% +- %.4f   gap %+.4f  agrees: %s"
          % (flat_two, flat_two_se, gap, agrees))
    print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
