#!/usr/bin/env python3
"""E141 F1 item 2: the byte table and `plan.perThread` for every candidate arm.

Every constant here is read from the source this session, not remembered:

  qwen35RowTop32Tiles      = 32          Qwen35.swift:4503
  qwen35Top32TG            = 256         Qwen35.swift:4255
  stride                   = 32 * 256    Qwen35.swift:4272 via :4511
  precondition perThread  <= 32          Qwen35.swift:4512
  derivedClusterRowsPerLeaf = 8          Qwen35.swift:5504
  probe fraction            = 0.25       Qwen35.swift:4937 (advisor base)
  probes = ceil(fraction * leaves)       Qwen35.swift:6057

The advisor's F1 table is written for thorfinn's `ProbeArm.compiledDefault =
.p15` tree. The advisor base this branch sits on still carries 0.25, so both
are reported: the p15 column reproduces F1 exactly and is the bridge, and the
p25 column is what this branch actually measures.

BYTES. One coarse row is 1600 B (F144 readout census). The centroid pass reads
every leaf; the row pass reads `probes * rowsPerLeaf` rows; the exact rerank
reads 32 rows and does not move.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROW_BYTES = 1600
ROWS_PER_LEAF = 8
RERANK_MB = 0.09
DRAFT_STEP_MB = 323.59  # all sources, F144
DRAFT_HEAD_SHARE = (0.075, 0.09)  # E131 rung 0b census, F143

TILES = 32  # qwen35RowTop32Tiles
THREADGROUP = 256  # qwen35Top32TG
STRIDE = TILES * THREADGROUP
PER_THREAD_LIMIT = 32

SHIPPED_PADDED = 98_336
FULL_PADDED = 248_320

# Measured this session from the 16-leg ABBA run, research/e141-rung2.json.
MEASURED_US_PER_MB = 28.91
MEASURED_LOCAL_ROUND_US = 129_195.4
RANKED_ROUND_US = 52_860.0  # F1 section 9
KAPPA_OVERHEAD = 0.646

# Rung 3 round counts, deterministic.
ROUNDS = {"beagle_a": (118, 117), "essays_montaigne": (151, 151)}
MEDPAIR = {"beagle_a": 0.478, "essays_montaigne": 0.522}
REFUTATION_FLOOR_PCT = 0.15


def leaves(padded: int, rows_per_leaf: int = ROWS_PER_LEAF) -> int:
    return padded // rows_per_leaf


def per_thread(rows: int) -> int:
    return (rows + STRIDE - 1) // STRIDE


def arm(name: str, padded: int, probes: int, rows_per_leaf: int) -> dict:
    lv = leaves(padded, rows_per_leaf)
    probed_rows = probes * rows_per_leaf
    centroid_mb = lv * ROW_BYTES / 1e6
    row_mb = probed_rows * ROW_BYTES / 1e6
    pt = per_thread(probed_rows)
    return {
        "arm": name,
        "padded_rows": padded,
        "rows_per_leaf": rows_per_leaf,
        "leaves": lv,
        "probes": probes,
        "probe_fraction_effective": probes / lv,
        "probed_rows": probed_rows,
        "centroid_mb": centroid_mb,
        "row_mb": row_mb,
        "total_mb": centroid_mb + row_mb + RERANK_MB,
        "plan_per_thread": pt,
        "per_thread_within_limit": pt <= PER_THREAD_LIMIT,
    }


def price(delta_mb: float) -> dict:
    """Price an added byte cost two ways: the campaign share model and the
    conversion measured directly by this session's ABBA legs."""
    added_us = delta_mb * MEASURED_US_PER_MB
    share = tuple(
        100.0 * delta_mb / DRAFT_STEP_MB * s for s in DRAFT_HEAD_SHARE
    )
    net = {}
    for label, f in (
        ("overhead", KAPPA_OVERHEAD * added_us / RANKED_ROUND_US),
        ("bandwidth", added_us / MEASURED_LOCAL_ROUND_US),
    ):
        net[label] = sum(
            MEDPAIR[p] * 100.0 * (1.0 - (full / ship) * (1.0 + f))
            for p, (ship, full) in ROUNDS.items()
        )
    return {
        "delta_mb": delta_mb,
        "added_us_per_round_measured_conversion": added_us,
        "round_cost_pct_share_model": share,
        "round_cost_pct_measured": 100.0 * added_us / MEASURED_LOCAL_ROUND_US,
        "net_ranked_pct": net,
    }


def budget() -> dict:
    """Largest added cost that still clears the refutation floor.

    The gross round-count term is fixed by rung 3 at +0.4051 %, so the whole
    question is how much per-round cost it can absorb.
    """
    out = {}
    for label, scale in (
        ("overhead", KAPPA_OVERHEAD / RANKED_ROUND_US),
        ("bandwidth", 1.0 / MEASURED_LOCAL_ROUND_US),
    ):
        lo, hi = 0.0, 20000.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            net = sum(
                MEDPAIR[p] * 100.0 * (1.0 - (full / ship) * (1.0 + scale * mid))
                for p, (ship, full) in ROUNDS.items()
            )
            if net >= REFUTATION_FLOOR_PCT:
                lo = mid
            else:
                hi = mid
        out[label] = {
            "max_added_us_per_round": lo,
            "max_added_mb": lo / MEASURED_US_PER_MB,
        }
    return out


# F2 Rule 121. The medpair weights above are infinitesimal weights; the
# published median is an order statistic over SORTED positions, so a finite
# move must be re-sorted. These figures come from the advisor's
# research/f209_reorder_value.py driven with this session's MEASURED rung 3
# halves (beagle +0.8475 %, essays +0.0000 %) and a uniform negative gain for
# the arm's round cost. Anchor 623e77af, the highest-scoring row carrying
# per-prompt evidence in the board cache available to this student.
F209_NET_MEDIAN_PCT = {
    "gross_prize_no_cost": 0.4055,
    "widened_hold_fraction": -1.6085,
    "widened_arm_A_hold_probes": -0.2655,
    "widened_arm_B_rowsPerLeaf16": 0.2905,
    "widened_arm_B_rowsPerLeaf20": 0.4015,
}

CAVEATS = {
    "f209_supersedes_the_linear_medpair_weight": (
        "net_ranked_pct below uses the linear Rule 116 weights. For a finite "
        "move the published median must be re-sorted, so F209_NET_MEDIAN_PCT "
        "is the decision quantity and net_ranked_pct is the intuition. The two "
        "agree on sign for every arm in this table."
    ),
    "essays_gain_is_zero_at_source": (
        "The F3 composition assumed both halves of the prize pay. Rung 3 "
        "measured the essays half at EXACTLY zero, and the truncation census "
        "explains why: essays produced 0 rounds truncated by an unproposable "
        "token across 151 rounds. The prize is beagle-only, which is why "
        "f209 prices the gross move at +0.4055 % and not the +0.9495 % F3 "
        "quoted."
    ),
    "price_is_a_zero_recall_loss_ceiling": (
        "price() reuses the rung 3 round counts, which were measured only for "
        "the hold-fraction arm at rowsPerLeaf 8. Holding the probe count "
        "(arm A) drops the effective probe fraction, and changing rowsPerLeaf "
        "(arm B) repartitions every cluster. Both can only lose recall "
        "relative to the measured arm, so every net figure below is an upper "
        "bound, not a forecast."
    ),
    "arm_B_keeps_the_fused_row_qmv": (
        "RESOLVED this session. qwen35ClusterRowQMV guarded `rowsPerCluster == "
        "8`, but the guard was the only thing pinning the width: the kernel "
        "source already reads rows_per from w_shape[1] and one simdgroup "
        "already emits 4 rows. The dispatch now derives simdgroups = "
        "rowsPerCluster / 4 and accepts any multiple of 4 in [4, 32], so arm B "
        "keeps the fused path instead of falling back to gatherQuantizedMM. "
        "qwen35VerifyClusterRowQMV shows widths 8, 16 and 20 reproduce "
        "gatherQuantizedMM bit for bit (maxAbsDiff 0), and width 8 dispatches "
        "exactly the grid it dispatched before."
    ),
    "arm_B_needs_the_padding_rule_generalised": (
        "qwen35CompactDraftCounts pads to a multiple of 8 at Qwen35.swift:4229 "
        "and qwen35BalancedBisecting2Means assumes `count / rowsPerLeaf` is "
        "exact at Qwen35.swift:5009. 248,320 divides by 16 and 20, but the "
        "shipped 98,336 does not divide by 20, so rowsPerLeaf must stay "
        "arm-scoped or the padding rule must follow it."
    ),
    "arm_B_keeps_the_fused_finalize": (
        "Qwen35RowTop32 already threads rowsPerCluster through "
        "qwen35Top32FinalizeSource at Qwen35.swift:4526, so the top-32 "
        "finalize stage is shape generic and does not block arm B."
    ),
}


def main() -> None:
    fractions = {"p25_advisor_base": 0.25, "p15_thorfinn_tree": 0.15}
    report: dict = {
        "constants": {
            "row_bytes": ROW_BYTES,
            "stride": STRIDE,
            "per_thread_limit": PER_THREAD_LIMIT,
            "measured_us_per_mb": MEASURED_US_PER_MB,
            "ranked_round_us": RANKED_ROUND_US,
        },
        "caveats": CAVEATS,
        "trees": {},
        "refutation_budget": budget(),
        "f209_net_median_pct": F209_NET_MEDIAN_PCT,
    }

    for tree, frac in fractions.items():
        ship_leaves = leaves(SHIPPED_PADDED)
        ship_probes = math.ceil(frac * ship_leaves)
        shipped = arm("shipped", SHIPPED_PADDED, ship_probes, ROWS_PER_LEAF)

        arms = [shipped]
        # As specified: hold the FRACTION. This is what rung 2 measured.
        arms.append(
            arm(
                "widened_hold_fraction",
                FULL_PADDED,
                math.ceil(frac * leaves(FULL_PADDED)),
                ROWS_PER_LEAF,
            )
        )
        # Arm A: hold the absolute PROBE COUNT.
        arms.append(
            arm("widened_arm_A_hold_probes", FULL_PADDED, ship_probes, ROWS_PER_LEAF)
        )
        # Arm B: raise rowsPerLeaf so the centroid pass stops growing.
        for rpl in (16, 20):
            probes_b = max(1, round(shipped["probed_rows"] / rpl))
            arms.append(
                arm(f"widened_arm_B_rowsPerLeaf{rpl}", FULL_PADDED, probes_b, rpl)
            )

        for a in arms:
            a["delta_mb_vs_shipped"] = a["total_mb"] - shipped["total_mb"]
            if a["arm"] != "shipped":
                a["price"] = price(a["delta_mb_vs_shipped"])
        report["trees"][tree] = {"probe_fraction": frac, "arms": arms}

    out = Path("research/e141-arm-geometry.json")
    out.write_text(json.dumps(report, indent=2) + "\n")

    for tree, blob in report["trees"].items():
        print(f"\n=== {tree}, probe fraction {blob['probe_fraction']} ===")
        print(
            f"{'arm':32s} {'leaves':>7s} {'probes':>7s} {'p_eff':>7s} "
            f"{'rows':>7s} {'cen':>7s} {'row':>7s} {'tot':>7s} {'dMB':>8s} {'pT':>3s}"
        )
        for a in blob["arms"]:
            print(
                f"{a['arm']:32s} {a['leaves']:7d} {a['probes']:7d} "
                f"{a['probe_fraction_effective']:7.4f} {a['probed_rows']:7d} "
                f"{a['centroid_mb']:7.2f} {a['row_mb']:7.2f} {a['total_mb']:7.2f} "
                f"{a['delta_mb_vs_shipped']:+8.2f} {a['plan_per_thread']:3d}"
                + ("" if a["per_thread_within_limit"] else "  PRECONDITION FAIL")
            )
        for a in blob["arms"]:
            if "price" not in a:
                continue
            p = a["price"]
            lo, hi = p["round_cost_pct_share_model"]
            print(
                f"  {a['arm']:30s} +{p['added_us_per_round_measured_conversion']:7.1f} us"
                f"  round cost {p['round_cost_pct_measured']:5.3f} % measured"
                f" ({lo:.2f}-{hi:.2f} % share model)"
                f"  net {p['net_ranked_pct']['overhead']:+.3f} to "
                f"{p['net_ranked_pct']['bandwidth']:+.3f} %"
            )

    print("\n=== F209 sorted-median value, the decision quantity ===")
    for name, pct in F209_NET_MEDIAN_PCT.items():
        print(f"  {name:32s} {pct:+.4f} %")

    b = report["refutation_budget"]
    print(
        f"\nTo clear the +{REFUTATION_FLOOR_PCT} % refutation floor the added cost "
        "must stay under"
    )
    for label, v in b.items():
        print(
            f"  {label:10s} {v['max_added_us_per_round']:7.1f} us/round "
            f"= {v['max_added_mb']:6.2f} MB per draft step"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
