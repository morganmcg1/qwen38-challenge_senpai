#!/usr/bin/env python3
"""E55 Risk 3, answered from published campaign ratios with no GPU second.

Risk 3 in the assignment poses a dichotomy: either the wide-5 load path and
`qmv_fast_crossrow_affine4_g64_m<T,9,5>` are DIFFERENT families and the crossrow
tier escapes PR #8's group-throughput collapse, or they are the SAME family and
the isolated single-body build is an artefact.

I decline both. A one-parameter cost model says the collapse is real AND the M=9
cell still wins, because NA=5 wins exactly when it removes a weight stream.

The model
---------
`streams(M, IPG) = ceil(M / IPG)`, and the group NA values are IPG repeated with
a final `TAIL = M % IPG` (or IPG when that is zero). Every working group re-reads
the whole weight matrix. Assume:

  A1  per-dispatch cost is additive over working groups;
  A2  each group moves the same weight bytes;
  A3  one constant `r = cost(NA=5 group) / cost(NA<=4 group)` covers all M;
  A4  the measured cell ratios are dominated by that weight traffic.

Then `ratio(M) = cost(candidate groups) / cost(base groups)` depends only on `r`.

Why this is not the fabrication I refused earlier
-------------------------------------------------
I previously declined to derive absolute GB/s from E49's millisecond figures,
because I do not know its shape or iteration count. This script never uses an
absolute time or an absolute bandwidth. It uses only DIMENSIONLESS ratios of two
arms measured under one matched identity, which are invariant to shape and
iteration count. No number from PR #8 enters the fit; PR #8's quoted range is
used once, at the end, as an out-of-sample consistency check.

The decisive test is that `r` is over-determined. Two independent experiments on
two different cells each identify it, and they must agree or the model is wrong.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Shipped table on the campaign base, as stated in the assignment body.
# M -> IPG actually instantiated for !batched && g64 && b4 && out_vec_size >= 4096.
SHIPPED_IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}
CANDIDATE_IPG = 5

# Independently measured cell ratios (candidate / base), matched identity, ABBA.
# E49 Arm 1, run 92a0u0fl: isolated <T,9,5> vs <T,9,3>, 186.113 -> 163.305 ms.
E49_M9_RATIO = 163.305 / 186.113
# E27, run hy0qq9sk: M=5 cell ratio, quoted by the advisor.
E27_M5_RATIO = 0.7990
# E27's M=9 cell ratio from the same arm, quoted by the advisor.
E27_M9_RATIO = 0.8854

# PR #8, as quoted to me in the assignment. Used ONLY as an out-of-sample check.
PR8_BOUNDARY_SLOWDOWN_RANGE = (1.13, 1.54)
PR8_QUOTED_GBPS = {"na5_group": 95.5, "na_le4_group": 165.6, "m9_aggregate": 239.5}

SCORED_WIDTHS = (2, 3, 4, 5, 6, 7, 8, 9)


def groups(m: int, ipg: int) -> list[int]:
    """NA value of each working group for row count `m` at items-per-group `ipg`."""
    n = math.ceil(m / ipg)
    tail = m % ipg or ipg
    return [ipg] * (n - 1) + [tail]


def cost(nas: list[int], r: float) -> float:
    """Additive cost in units of one NA<=4 group, per A1..A3."""
    return sum(r if na == 5 else 1.0 for na in nas)


def ratio(m: int, r: float) -> float:
    base = groups(m, SHIPPED_IPG[m]) if m in SHIPPED_IPG else None
    if base is None:
        return float("nan")
    cand = groups(m, CANDIDATE_IPG)
    return cost(cand, r) / cost(base, r)


def solve_r(m: int, observed_ratio: float) -> float:
    """Invert ratio(M) for r. Linear in r, so this is exact, not a search."""
    base = cost(groups(m, SHIPPED_IPG[m]), 1.0)  # base has no NA=5 group
    cand_nas = groups(m, CANDIDATE_IPG)
    n5 = sum(1 for na in cand_nas if na == 5)
    n_other = len(cand_nas) - n5
    if n5 == 0:
        return float("nan")
    return (observed_ratio * base - n_other) / n5


def stream_table() -> dict:
    rows = []
    for m in SCORED_WIDTHS:
        if m not in SHIPPED_IPG:
            rows.append({"m": m, "in_shipped_switch": False})
            continue
        b = groups(m, SHIPPED_IPG[m])
        c = groups(m, CANDIDATE_IPG)
        rows.append(
            {
                "m": m,
                "in_shipped_switch": True,
                "base_ipg": SHIPPED_IPG[m],
                "base_groups": b,
                "base_streams": len(b),
                "candidate_groups": c,
                "candidate_streams": len(c),
                "stream_reduction": len(b) - len(c),
                "removes_a_stream": len(c) < len(b),
                "lone_na5_group": c == [5],
            }
        )
    reducing = [x["m"] for x in rows if x.get("removes_a_stream")]
    lone = [x["m"] for x in rows if x.get("lone_na5_group")]
    return {
        "rows": rows,
        "widths_where_na5_removes_a_stream": reducing,
        "widths_with_a_lone_na5_group": lone,
        "reading": (
            "Across every scored width, NA=5 removes a weight stream at M={red} "
            "only. Those are exactly the two cells E27 changed. M={lone} is the "
            "only configuration in the whole table that produces a LONE NA=5 "
            "group with no NA<=4 sibling, which is the configuration PR #8 "
            "measured the collapse on."
        ).format(red=reducing, lone=lone),
    }


def identify_r() -> dict:
    """The decisive test: two independent cells must return the same r."""
    r_from_e49_m9 = solve_r(9, E49_M9_RATIO)
    r_from_e27_m5 = solve_r(5, E27_M5_RATIO)
    r_from_e27_m9 = solve_r(9, E27_M9_RATIO)

    estimates = {
        "e49_arm1_m9_cell": r_from_e49_m9,
        "e27_m5_cell": r_from_e27_m5,
        "e27_m9_cell": r_from_e27_m9,
    }
    primary = [r_from_e49_m9, r_from_e27_m5]
    spread = max(primary) - min(primary)
    mean_r = sum(primary) / len(primary)
    return {
        "observed_ratios": {
            "e49_arm1_m9": E49_M9_RATIO,
            "e27_m5": E27_M5_RATIO,
            "e27_m9": E27_M9_RATIO,
        },
        "r_estimates": estimates,
        "r_primary_mean": mean_r,
        "r_primary_spread": spread,
        "r_primary_spread_pct": 100.0 * spread / mean_r,
        "independent_cells_agree_within_5pct": (100.0 * spread / mean_r) < 5.0,
        "collapse_is_real": mean_r > 1.0,
        "reading": (
            "One constant, identified twice. E49's M=9 cell gives r = {a:.5f} and "
            "E27's M=5 cell gives r = {b:.5f}, agreeing to {p:.2f} %. These are "
            "different cells, different experiments, different stream counts and "
            "different sessions, so the agreement is an out-of-sample validation "
            "of the model rather than a fit. r > 1 means an NA=5 group really is "
            "slower per byte, so PR #8's collapse is REAL and present in the "
            "crossrow tier. The M=9 cell wins anyway, because it drops from three "
            "streams to two."
        ).format(a=r_from_e49_m9, b=r_from_e27_m5, p=100.0 * spread / mean_r),
    }


def predictions(r_lo: float, r_hi: float) -> dict:
    rows = []
    for m in SCORED_WIDTHS:
        if m not in SHIPPED_IPG:
            continue
        lo, hi = sorted((ratio(m, r_lo), ratio(m, r_hi)))
        rows.append(
            {
                "m": m,
                "predicted_ratio_lo": lo,
                "predicted_ratio_hi": hi,
                "predicted_pct_lo": 100.0 * (lo - 1.0),
                "predicted_pct_hi": 100.0 * (hi - 1.0),
                "predicted_direction": "faster" if hi < 1.0 else ("slower" if lo > 1.0 else "mixed"),
            }
        )
    by_m = {x["m"]: x for x in rows}
    m78 = [by_m[7], by_m[8]]
    pr8_lo = min(x["predicted_ratio_lo"] for x in m78)
    pr8_hi = max(x["predicted_ratio_hi"] for x in m78)
    consistent = (
        pr8_hi >= PR8_BOUNDARY_SLOWDOWN_RANGE[0]
        and pr8_lo <= PR8_BOUNDARY_SLOWDOWN_RANGE[1]
    )
    return {
        "rows": rows,
        "m7_m8_predicted_ratio_range": [pr8_lo, pr8_hi],
        "pr8_quoted_boundary_slowdown_range": list(PR8_BOUNDARY_SLOWDOWN_RANGE),
        "pr8_range_consistent_with_model": consistent,
        "reading": (
            "The model predicts NA=5 at M=7 and M=8 is HARMFUL by {a:.1f} % to "
            "{b:.1f} %, because the stream count does not fall there: two groups "
            "before, two groups after, and one of them becomes slower. PR #8 "
            "measured its boundary widths at {p0}x to {p1}x slower, which "
            "brackets the prediction. That is a second out-of-sample check, and "
            "PR #8 contributed nothing to the fit."
        ).format(
            a=100.0 * (pr8_lo - 1.0),
            b=100.0 * (pr8_hi - 1.0),
            p0=PR8_BOUNDARY_SLOWDOWN_RANGE[0],
            p1=PR8_BOUNDARY_SLOWDOWN_RANGE[1],
        ),
    }


def break_even(r_lo: float, r_hi: float) -> dict:
    """At what `r` does each cell stop paying? This is the transfer-risk statement.

    `ratio(M)` is linear and increasing in `r`, so each cell has one break-even
    value. It matters because `r` was identified on M4 Pro cells and the ranked
    host need not share it.
    """
    rows = []
    for m in SCORED_WIDTHS:
        if m not in SHIPPED_IPG:
            continue
        base = cost(groups(m, SHIPPED_IPG[m]), 1.0)
        cand_nas = groups(m, CANDIDATE_IPG)
        n5 = sum(1 for na in cand_nas if na == 5)
        n_other = len(cand_nas) - n5
        be = float("inf") if n5 == 0 else (base - n_other) / n5
        rows.append(
            {
                "m": m,
                "break_even_r": be,
                "profitable_while_r_below": be if n5 else None,
                "identified_r_headroom_pct": (
                    100.0 * (be - r_hi) / r_hi if n5 and be > 1.0 else None
                ),
                "unprofitable_for_every_r_above_1": bool(n5 and be <= 1.0),
            }
        )
    return {
        "rows": rows,
        "identified_r_interval": [r_lo, r_hi],
        "reading": (
            "M=5 and M=9 both break even at r = 2, so they stay profitable until "
            "an NA=5 group costs twice an NA<=4 group. The identified r of "
            "{lo:.3f}..{hi:.3f} leaves {head:.1f} % of headroom, which is the "
            "margin the local-to-ranked transfer has to eat before the M=9 cell "
            "stops paying. M=6, M=7 and M=8 break even at r = 1, so they are "
            "unprofitable for ANY r > 1: NA=5 cannot pay there at any hardware "
            "operating point, because the stream count does not fall."
        ).format(
            lo=r_lo,
            hi=r_hi,
            head=100.0 * (2.0 - r_hi) / r_hi,
        ),
    }


def risk3_verdict(ident: dict) -> dict:
    return {
        "assignment_answer_1_different_families": False,
        "assignment_answer_2_isolated_build_is_artefact": False,
        "answer": (
            "Neither. Same family, the collapse is real at r = {r:.3f}, and NA=5 "
            "is profitable exactly when it removes a weight stream. The isolated "
            "single-body build is not an artefact: its ratio is reproduced by a "
            "model calibrated on a different cell from a different experiment."
        ).format(r=ident["r_primary_mean"]),
        "stream_corrected_reading": (
            "r is the stream-corrected quantity Risk 3 asked for, expressed as a "
            "ratio rather than as GB/s. An NA=5 group costs {r:.3f}x an NA<=4 "
            "group per unit of weight traffic. PR #8's own quoted figures give "
            "165.6 / 95.5 = {p:.3f}, which agrees with r to {d:.1f} %. I report "
            "that agreement as a check and did not use it to fit anything."
        ).format(
            r=ident["r_primary_mean"],
            p=PR8_QUOTED_GBPS["na_le4_group"] / PR8_QUOTED_GBPS["na5_group"],
            d=100.0
            * abs(
                ident["r_primary_mean"]
                - PR8_QUOTED_GBPS["na_le4_group"] / PR8_QUOTED_GBPS["na5_group"]
            )
            / ident["r_primary_mean"],
        ),
        "honest_limits": [
            "A1 assumes working groups are sequential and additive; a concurrent "
            "bandwidth-sharing execution model would change the algebra.",
            "A3 lumps NA=2, 3 and 4 into one cost class. The two-cell agreement "
            "supports that, but one measurement cannot resolve an NA-dependent "
            "cost inside the NA<=4 class.",
            "The model is a LOCAL per-cell cost model. It predicts cell ratios, "
            "not published score, and it does NOT explain E27's ranked loss: it "
            "says both E27 cells were locally faster, which E27 also measured.",
            "r is identified on M4 Pro cells. The advisor's own transfer result "
            "says the M4 Pro ladder over-prices depth on ranked M5, so r need "
            "not carry to the ranked host unchanged.",
        ],
    }


def negative_controls() -> dict:
    c = {}

    # An r of exactly 1 must predict no change at any width, or the algebra is wrong.
    c["r_equals_1_predicts_stream_count_ratio_only"] = (
        abs(ratio(9, 1.0) - 2.0 / 3.0) < 1e-12 and abs(ratio(7, 1.0) - 1.0) < 1e-12
    )

    # The group decomposition must reproduce the shipped table stated in the brief.
    c["group_decomposition_matches_shipped_table"] = (
        groups(9, 3) == [3, 3, 3]
        and groups(5, 3) == [3, 2]
        and groups(7, 4) == [4, 3]
        and groups(8, 4) == [4, 4]
        and groups(4, 4) == [4]
        and groups(9, 5) == [5, 4]
        and groups(5, 5) == [5]
    )

    # solve_r must be a true inverse of ratio.
    r_test = 1.7
    c["solve_r_inverts_ratio"] = all(
        abs(solve_r(m, ratio(m, r_test)) - r_test) < 1e-12 for m in (5, 7, 8, 9)
    )

    # A wrong observed ratio must break the two-cell agreement.
    bad = solve_r(5, 0.50)
    good = solve_r(9, E49_M9_RATIO)
    c["agreement_claim_can_fail"] = abs(bad - good) / good > 0.05

    # The harmful prediction must not be an artefact of r's sign.
    c["m7_predicted_slower_for_every_r_above_1"] = all(
        ratio(7, rr) > 1.0 for rr in (1.01, 1.3, 1.6, 2.5)
    )

    # M=9 must be predicted faster only while r stays below the break-even.
    c["m9_break_even_exists"] = ratio(9, 2.0) == 1.0 and ratio(9, 2.5) > 1.0

    return {"controls": c, "all_fire": all(c.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e55-stream-model.json")
    args = ap.parse_args()

    streams = stream_table()
    ident = identify_r()
    r_lo = min(ident["r_estimates"]["e49_arm1_m9_cell"], ident["r_estimates"]["e27_m5_cell"])
    r_hi = max(ident["r_estimates"]["e49_arm1_m9_cell"], ident["r_estimates"]["e27_m5_cell"])
    preds = predictions(r_lo, r_hi)
    controls = negative_controls()

    doc = {
        "experiment": "E55",
        "answers": "assignment Risk 3, stream-corrected, no GPU second",
        "uses_only_dimensionless_ratios": True,
        "pr8_numbers_used_in_fit": False,
        "stream_table": streams,
        "identification": ident,
        "predictions": preds,
        "break_even": break_even(r_lo, r_hi),
        "risk3": risk3_verdict(ident),
        "r_interval": [r_lo, r_hi],
        "negative_controls": controls,
    }
    doc["verdict_ok"] = bool(
        ident["independent_cells_agree_within_5pct"]
        and ident["collapse_is_real"]
        and preds["pr8_range_consistent_with_model"]
        and controls["all_fire"]
    )

    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"wrote {args.out}")
    print("stream decomposition on the shipped table:")
    for row in streams["rows"]:
        if not row.get("in_shipped_switch"):
            continue
        print(
            f"  M={row['m']}  base {row['base_groups']} ({row['base_streams']} streams)"
            f"  ->  cand {row['candidate_groups']} ({row['candidate_streams']} streams)"
            f"   removes_stream={row['removes_a_stream']}"
            f"   lone_na5={row['lone_na5_group']}"
        )
    print(f"  NA=5 removes a stream at M = {streams['widths_where_na5_removes_a_stream']}")
    print(f"  lone NA=5 group at M      = {streams['widths_with_a_lone_na5_group']}")
    print()
    print("identification of r = cost(NA=5 group) / cost(NA<=4 group):")
    for k, v in ident["r_estimates"].items():
        print(f"  r from {k:<22} = {v:.5f}")
    print(f"  primary mean {ident['r_primary_mean']:.5f}, spread {ident['r_primary_spread_pct']:.2f} %")
    print(f"  independent cells agree within 5 % : {ident['independent_cells_agree_within_5pct']}")
    print()
    print("predicted cell ratios:")
    for row in preds["rows"]:
        print(
            f"  M={row['m']}  {row['predicted_ratio_lo']:.4f}..{row['predicted_ratio_hi']:.4f}"
            f"  ({row['predicted_pct_lo']:+.2f} % .. {row['predicted_pct_hi']:+.2f} %)"
            f"  {row['predicted_direction']}"
        )
    print(f"  M=7/M=8 predicted {preds['m7_m8_predicted_ratio_range'][0]:.4f}"
          f"..{preds['m7_m8_predicted_ratio_range'][1]:.4f}x ; PR #8 quoted "
          f"{PR8_BOUNDARY_SLOWDOWN_RANGE[0]}..{PR8_BOUNDARY_SLOWDOWN_RANGE[1]}x ; "
          f"consistent={preds['pr8_range_consistent_with_model']}")
    print()
    print(f"  negative controls all fire : {controls['all_fire']}")
    print(f"  verdict_ok                 : {doc['verdict_ok']}")
    return 0 if doc["verdict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
