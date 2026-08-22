#!/usr/bin/env python3
"""E129 -- price the width table as DELETED INSTRUCTIONS PER OUTPUT ELEMENT.

F13 asks for this frame instead of resident simdgroups, because F114 says a
genuine deletion of instructions is the one category that converts to time,
and F131 measured the residency coefficient at about zero.

WHERE THE COUNTS COME FROM. They are read off `qwen_e120_qmv_wide<NA, RPS,
USE_TABLE>` in `Qwen35.swift`, not estimated. Inside one k block one thread
runs two disjoint statement families:

  row-keyed   the `for r in 0 ..< RPS` prologue. Weight halfwords, scale,
              bias, their index arithmetic, their bf16 widenings, and the
              nibble extraction. The count is proportional to RPS and does
              NOT depend on NA.
  m-keyed     the `for m in 0 ..< NA` bodies. The sum-table read and the four
              activation `vec<bfloat16_t, 4>` loads with their 16 widenings.
              The count is proportional to NA and does NOT depend on RPS.

One thread's k block covers `NA * RPS` output elements, so dividing gives

    row-keyed cost per output element  =  (count per row) / NA   =  c / IPG
    m-keyed   cost per output element  =  (count per m)   / RPS

for a full group, where NA == IPG. The `partial[r] +=` chain and the
`acc[r] +=` update are NA*RPS-keyed and therefore constant per output element;
that is the same statement the E129 contraction census proved at the IR level.

WHY THE ARM IS NOT A PURE DELETION. `{6:6, 7:7}` raises IPG and lowers RPS at
the same two widths. Raising IPG deletes row-keyed work. Lowering RPS ADDS
m-keyed work, at exactly the same factor of two. In raw instruction counts the
two nearly cancel. They do not cancel in time, because a weight halfword is a
streamed read of the whole quantized matrix and an activation vec4 is a read
of an `M x in_vec_size` working set that fits in cache. So the raw count is
reported for honesty and the priced number uses E123's measured per-class
shares of QMV time as the cost weights.

HOW THE PRICING WORKS. For class `c` with per-output-element scaling
`s_c(m)`, E123's aggregate share over the ranked mix at the SHIPPED table is

    share_c  =  u_c * sum_m  mass(m) * s_c_shipped(m)

which fixes the unit cost `u_c`. The candidate table then costs

    delta_c  =  u_c * sum_m  mass(m) * (s_c_new(m) - s_c_shipped(m))

reported as a percentage of QMV time. `harness=ranked`: the mass is the F83
median-sensitivity weighting over the E114 max-entropy routed width
histograms, never the local fixture histogram.

Zero GPU seconds.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e129_entry_point_census as census  # noqa: E402

RANKED_WIDTHS = tuple(range(3, 9))

# Statement counts per k block, read off the kernel body.
#
#   `per_row`  executed once for each of the RPS output rows.
#   `per_m`    executed once for each of the NA activation lanes.
#
# `nibble_int_ops` is 3 shifts and 4 masks for each of the 4 `i` values.
ROW_KEYED = {
    "weight_element_loads": 4,
    "weight_address_arith": 1,
    "metadata_loads": 2,
    "metadata_index_arith": 1,
    "metadata_widenings": 2,
    "nibble_int_ops": 28,
}
M_KEYED = {
    "sums_table_loads": 1,
    "activation_vec4_loads": 4,
    "activation_widenings": 16,
    "activation_address_arith": 4,
}

# E123's priced census, as a percentage of QMV time at the shipped table over
# the ranked mix. `sums_add_tree` is carried under two readings because the
# routed path uses `USE_TABLE=true`, where the class is the `st[m]` read plus
# the `sums * bias_local[r]` term. The read is m-keyed; the multiply-add is
# NA*RPS-keyed and therefore constant per output element.
E123_SHARE = {
    "weight_element_loads": 6.420,
    "metadata_loads": 3.210,
    "metadata_widenings": 1.218,
    "activation_vec4_loads": 6.211,
}
SUMS_ADD_TREE_SHARE = 8.476


def plan_map(name: str) -> dict[int, tuple[int, int]]:
    return {m: (ipg, rps) for m, ipg, rps in census.PLANS[name]}


def per_element(plan: dict[int, tuple[int, int]], m: int) -> dict[str, float]:
    ipg, rps = plan[m]
    out = {k: v / ipg for k, v in ROW_KEYED.items()}
    out.update({k: v / rps for k, v in M_KEYED.items()})
    return out


def ranked_mass() -> dict[int, float]:
    """F83-weighted routed width mass over the ranked prompts."""
    hists = census.ranked_histograms()
    weights = {p: w for p, (_, w) in census.RANKED_WIDTH_MIX.items()}
    total = sum(weights.values())
    mass = {m: 0.0 for m in RANKED_WIDTHS}
    for prompt, hist in hists.items():
        w = weights[prompt] / total
        for m in RANKED_WIDTHS:
            mass[m] += w * hist[m]
    scale = sum(mass.values())
    return {m: v / scale for m, v in mass.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="onepass67r2")
    ap.add_argument("--base", default="shipped")
    ap.add_argument("--json", type=pathlib.Path,
                    default=pathlib.Path("research/out/e129-deleted-ops.json"))
    args = ap.parse_args()

    base = plan_map(args.base)
    cand = plan_map(args.table)
    mass = ranked_mass()

    print("harness=ranked  official_or_ranked_score=false  timing_valid=false")
    print("base=%s candidate=%s" % (args.base, args.table))
    print()
    print("ranked routed width mass: %s" % {
        m: round(v, 5) for m, v in mass.items()})
    print()

    # 1. Raw statement counts per output element, per k block, per width.
    print("--- statements per output element, per k block ---")
    print("%3s %11s %11s %10s %10s %10s" % (
        "M", "base ipg:rps", "cand ipg:rps", "base", "cand", "delta"))
    per_width = {}
    for m in RANKED_WIDTHS:
        b = per_element(base, m)
        c = per_element(cand, m)
        bt, ct = sum(b.values()), sum(c.values())
        per_width[m] = {"base": b, "candidate": c,
                        "base_total": bt, "candidate_total": ct,
                        "delta_total": ct - bt}
        print("%3d %11s %11s %10.4f %10.4f %+10.4f" % (
            m, "%d:%d" % base[m], "%d:%d" % cand[m], bt, ct, ct - bt))
    raw_base = sum(mass[m] * per_width[m]["base_total"] for m in RANKED_WIDTHS)
    raw_cand = sum(mass[m] * per_width[m]["candidate_total"]
                   for m in RANKED_WIDTHS)
    print("ranked-weighted  %10.4f %10.4f %+10.4f  (%+.3f %%)" % (
        raw_base, raw_cand, raw_cand - raw_base,
        100.0 * (raw_cand - raw_base) / raw_base))
    print()

    # 2. The two families separately. This is where the arm actually lives.
    print("--- by family, ranked-weighted statements per output element ---")
    families = {"row_keyed": ROW_KEYED, "m_keyed": M_KEYED}
    family_rows = {}
    for fam, members in families.items():
        b = sum(mass[m] * sum(per_width[m]["base"][k] for k in members)
                for m in RANKED_WIDTHS)
        c = sum(mass[m] * sum(per_width[m]["candidate"][k] for k in members)
                for m in RANKED_WIDTHS)
        family_rows[fam] = {"base": b, "candidate": c, "delta": c - b}
        print("%-10s base %8.4f  cand %8.4f  delta %+8.4f  (%+.2f %%)" % (
            fam, b, c, c - b, 100.0 * (c - b) / b))
    print()

    # 3. Priced with E123's measured shares.
    print("--- priced with E123 shares, %% of QMV time ---")
    print("%-24s %8s %10s %10s %9s" % (
        "class", "share", "unit", "new", "delta"))
    priced = {}
    total_delta = 0.0
    for name, share in E123_SHARE.items():
        agg_base = sum(mass[m] * per_width[m]["base"][name]
                       for m in RANKED_WIDTHS)
        agg_cand = sum(mass[m] * per_width[m]["candidate"][name]
                       for m in RANKED_WIDTHS)
        unit = share / agg_base
        new = unit * agg_cand
        priced[name] = {"share_percent": share, "unit_cost": unit,
                        "new_percent": new, "delta_percent": new - share}
        total_delta += new - share
        print("%-24s %8.3f %10.4f %10.3f %+9.3f" % (
            name, share, unit, new, new - share))

    # `sums_add_tree` under both readings.
    agg_base = sum(mass[m] * per_width[m]["base"]["sums_table_loads"]
                   for m in RANKED_WIDTHS)
    agg_cand = sum(mass[m] * per_width[m]["candidate"]["sums_table_loads"]
                   for m in RANKED_WIDTHS)
    sums_adverse = SUMS_ADD_TREE_SHARE * (agg_cand / agg_base - 1.0)
    print("%-24s %8.3f %10s %10s %+9.3f   (m-keyed reading, adverse)" % (
        "sums_add_tree", SUMS_ADD_TREE_SHARE, "-", "-", sums_adverse))
    print("%-24s %8.3f %10s %10s %+9.3f   (NA*RPS-keyed reading, neutral)" % (
        "sums_add_tree", SUMS_ADD_TREE_SHARE, "-", "-", 0.0))
    print()
    print("TOTAL named classes, sums neutral   %+8.3f %% of QMV time" %
          total_delta)
    print("TOTAL named classes, sums adverse   %+8.3f %% of QMV time" %
          (total_delta + sums_adverse))

    payload = {
        "harness": "ranked",
        "official_or_ranked_score": False,
        "timing_valid": False,
        "instrument": "statement census of qwen_e120_qmv_wide, "
                      "priced with the E123 class shares",
        "base_table": args.base,
        "candidate_table": args.table,
        "ranked_width_mass": mass,
        "row_keyed_counts_per_row": ROW_KEYED,
        "m_keyed_counts_per_m": M_KEYED,
        "per_width": {str(m): per_width[m] for m in RANKED_WIDTHS},
        "family": family_rows,
        "raw_weighted": {"base": raw_base, "candidate": raw_cand,
                         "delta": raw_cand - raw_base,
                         "delta_percent":
                             100.0 * (raw_cand - raw_base) / raw_base},
        "priced": priced,
        "sums_add_tree": {"share_percent": SUMS_ADD_TREE_SHARE,
                          "delta_percent_m_keyed": sums_adverse,
                          "delta_percent_narps_keyed": 0.0},
        "total_delta_percent_qmv_sums_neutral": total_delta,
        "total_delta_percent_qmv_sums_adverse": total_delta + sums_adverse,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
