#!/usr/bin/env python3
"""Can any widened-vocabulary design fit inside the mechanism's own ceiling?

The `full` arm already recovers every binding event the draft head can reach,
so the whole mechanism is worth CEILING_MEDIAN_PCT of published median and no
more. Dividing by the measured ranked conversion turns that into a hard
per-round cost budget. A design costing more than the budget cannot win
however well it is engineered.

An additive model of the cluster search:

    added_us_per_round = d_leaves * c_leaf + d_rows * c_row + t * c_table + w

  c_leaf   coarse centroid scan, per leaf centroid per round
  c_row    row QMV, per probed row per round
  c_table  fixed penalty for gathering from the 248,320-row table instead of
           the 98,336-row one, at constant probed rows; t is 0 or 1
  w        kernel penalty for a leaf wider than the shipped 8, w(8) = 0

Three timed arms give three equations in five unknowns, so `c_table` is NOT
identified. This script reports what the arms do pin down, states the
degeneracy plainly, and then asks the only question that matters: can a
two-region design fit the budget even under the assumptions most favourable
to it?

WHY leaf16 DID NOT SETTLE THIS. The pre-registered rule expected leaf16 to
add either about 0 or about 579 us/round. It removed 264 us/round instead,
because it halves the leaf count at the same time as it doubles the leaf
width. That confound was flagged before the run. It bounds the width penalty
from above and the centroid cost from below, which is useful, but it cannot
separate `c_table` from `w20`.
"""

import argparse
import json
from pathlib import Path

CEILING_MEDIAN_PCT = 0.4055  # `full` arm f209 price: beagle +0.8475, essays 0
RANKED_PCT_PER_US = 0.0012229  # 0.7084 % of median per 579.3 us/round

SHIPPED_LEAVES = 12_292
SHIPPED_ROWS = 24_584
CORE_ROWS = 98_336
TAIL_ROWS = 248_320 - CORE_ROWS

ARMS = {
    #          leaves  probed rows  widened table  leaf width
    "leaf16": (6_146, 24_576, False, 16),
    "armB20": (12_416, 24_580, True, 20),
    "full": (31_040, 62_080, True, 8),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e141-cost-model.json")
    args = ap.parse_args()

    added: dict[str, float] = {}
    for arm, suffix in (("leaf16", "-leaf16"), ("armB20", "-armB20"), ("full", "")):
        blob = json.loads(Path(f"research/e141-rung2{suffix}.json").read_text())
        added[arm] = blob["e141_added_us_per_round_at_p025"]

    budget_us = CEILING_MEDIAN_PCT / RANKED_PCT_PER_US

    # leaf16: -6146*c_leaf - 8*c_row + w16 = added. Every term is a physical
    # cost so w16 >= 0, which makes the no-penalty reading the lower bound on
    # c_leaf. The -8 rows are 0.03 % of the probe and are dropped.
    d_leaves_16 = SHIPPED_LEAVES - ARMS["leaf16"][0]
    c_leaf_min = -added["leaf16"] / d_leaves_16

    # armB20 holds probed rows fixed, so it pins the sum c_table + w20 and
    # nothing finer.
    d_leaves_b = ARMS["armB20"][0] - SHIPPED_LEAVES
    c_table_plus_w20 = added["armB20"] - d_leaves_b * c_leaf_min

    # full has leaf width 8, so w = 0 there and it caps the row cost.
    d_leaves_f = ARMS["full"][0] - SHIPPED_LEAVES
    d_rows_f = ARMS["full"][1] - SHIPPED_ROWS
    c_row_max = (added["full"] - d_leaves_f * c_leaf_min) / d_rows_f

    # Best case for a two-region design: the widened table is free, the wider
    # kernel is free, and the centroid scan is as cheap as leaf16 allows.
    # Only the tail index is new; the core index is untouched by construction.
    feasibility = []
    for tail_leaf in (8, 16, 32, 64, 128):
        tail_leaves = TAIL_ROWS / tail_leaf
        for probe_fraction in (0.25, 0.05, 0.01):
            probed = TAIL_ROWS * probe_fraction
            cost = tail_leaves * c_leaf_min + probed * c_row_max
            feasibility.append(
                {
                    "tail_leaf_width": tail_leaf,
                    "tail_leaves": tail_leaves,
                    "tail_probe_fraction": probe_fraction,
                    "tail_probed_rows": probed,
                    "best_case_us_per_round": cost,
                    "fits_budget_with_c_table_zero": cost < budget_us,
                }
            )

    report = {
        "measured_added_us_per_round": added,
        "ceiling_median_pct": CEILING_MEDIAN_PCT,
        "ranked_pct_per_added_us": RANKED_PCT_PER_US,
        "break_even_budget_us_per_round": budget_us,
        "c_leaf_us_lower_bound": c_leaf_min,
        "c_row_us_upper_bound": c_row_max,
        "c_table_plus_w20_us": c_table_plus_w20,
        "c_table_us_identified": False,
        "c_table_us_range": [0.0, c_table_plus_w20],
        "shipped_search_us_per_round_lower_bound": (
            SHIPPED_LEAVES * c_leaf_min + SHIPPED_ROWS * c_row_max
        ),
        "two_region_feasibility": feasibility,
        "identifying_experiment": (
            "armA (248320@3073:8) shares leaf count, table and leaf width with "
            "full and differs only in probed rows, so full minus armA gives "
            "c_row exactly and armA then gives c_table exactly."
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print("measured added us/round (medpair, ABBA)")
    for arm in ("leaf16", "armB20", "full"):
        print(f"  {arm:8s} {added[arm]:+9.1f}")
    print(
        f"\nmechanism ceiling {CEILING_MEDIAN_PCT:.4f} % of median"
        f"  =>  break-even budget {budget_us:.1f} us/round"
    )
    print("\nwhat the three arms pin down")
    print(f"  c_leaf   >= {c_leaf_min:.5f} us per leaf centroid per round")
    print(f"  c_row    <= {c_row_max:.5f} us per probed row per round")
    print(
        f"  c_table + w20 = {c_table_plus_w20:.1f} us, so c_table is in "
        f"[0.0, {c_table_plus_w20:.1f}] and is NOT identified"
    )
    print(
        "  shipped cluster search >= "
        f"{report['shipped_search_us_per_round_lower_bound']:.0f} us of its "
        "round"
    )

    print(
        f"\ntwo-region design, best case (c_table = 0, wider kernel free), "
        f"budget {budget_us:.0f} us"
    )
    print("  tail leaf   tail leaves      p   probed rows   us/round   fits")
    for f in feasibility:
        print(
            f"  {f['tail_leaf_width']:9d}   {f['tail_leaves']:11.0f}   "
            f"{f['tail_probe_fraction']:.2f}   {f['tail_probed_rows']:11.0f}   "
            f"{f['best_case_us_per_round']:8.0f}   "
            f"{'yes' if f['fits_budget_with_c_table_zero'] else 'no'}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
