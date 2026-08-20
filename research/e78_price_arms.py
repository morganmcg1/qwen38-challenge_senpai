#!/usr/bin/env python3
"""Price every E78 arm from the rung 2a cell table.

  python3 research/e78_price_arms.py [--cells research/e78-artifacts/rung2a-cells.json]

Rung 2a measured absolute time for each `(shape, M, IPG)` cell. Every arm is a
map from `(M, out_vec_size)` to `IPG`, so once the cells are measured, each
arm's verify-round cost follows by lookup and no further GPU time is needed.
This is DERIVED. It assumes the round cost is the sum of its dispatches, which
rung 2b tests directly.

The arms only ever disagree at M = 5, 6 and 9, so widths outside that set
cancel exactly in every arm difference. The reweighted deltas below therefore
need no assumption about the widths the session did not measure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e73_pairs import CROWN, SHIPPED  # noqa: E402

# shape -> (calls per verify round). `linear_attn.out_proj` also serves
# `full_attn.o_proj`, which is the same (6144, 5120) cell.
CELL_CALLS = {
    "mlp.gate_up_fused": 64,
    "mlp.down": 64,
    "linear_attn.in_proj_fused_qkvzba": 48,
    "full_attn.qkv_proj_fused": 16,
    "head.lm_head": 1,
    "linear_attn.out_proj": 48 + 16,
}

# Widths where the tables disagree. Everything else cancels in an arm delta.
MOVED_WIDTHS = (5, 6, 9)

# Local public fixture under the shipped schedule, E78 rung 1, 78 rounds.
LOCAL_ROUNDS = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
LOCAL_TOTAL_ROUNDS = sum(LOCAL_ROUNDS.values())
# Ranked pooled verify-width time share, from the assignment's receipts.
RANKED_SHARE_PCT = {4: 14.2, 5: 24.1, 6: 33.4, 7: 12.2, 8: 7.35, 9: 5.75}


def arm_ipg(arm: str, m: int, n: int, k: int) -> int:
    """The IPG an arm selects for width `m` at out_vec_size `n`, in_vec_size `k`."""
    ship = SHIPPED[m]
    if arm == "kdown_8192":
        # Splits at M = 6 only, and only for the one cell rung 2a measured
        # faster at two groups. Every other width and shape keeps the base.
        return 3 if (m == 6 and k >= 8192) else ship
    if m not in MOVED_WIDTHS:
        return ship
    crown = CROWN[m]
    if arm == "a_ship":
        return ship
    if arm == "b_crown":
        return crown
    if arm.startswith("hybrid"):
        cutoff = int(arm.split("_", 1)[1])
        return ship if n >= cutoff else crown
    raise KeyError(arm)


ARMS = {
    "a_ship": "the campaign base table",
    "b_crown": "the promoted crown's table, IPG 3 at M = 5, 6 and 9",
    "hybrid_24928": "c_hybrid24928: crown IPG below out_vec_size 24928",
    "hybrid_8192": "d_hybrid8192: crown IPG below out_vec_size 8192",
    "kdown_8192": "e_kdown: IPG 3 at M = 6 when in_vec_size >= 8192",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells",
                    default="research/e78-artifacts/rung2a-cells.json")
    ap.add_argument("--out", default="research/e78-artifacts/rung2a-arms.json")
    ap.add_argument("--markdown",
                    default="research/e78-artifacts/rung2a-arms.md")
    args = ap.parse_args()

    doc = json.loads(pathlib.Path(args.cells).read_text())
    cell = {(c["shape"], c["m"], c["ipg"]): c for c in doc["cells"]}
    shape_n = {c["shape"]: c["n"] for c in doc["cells"]}
    shape_k = {c["shape"]: c["k"] for c in doc["cells"]}
    widths = sorted({c["m"] for c in doc["cells"]})

    def round_ms(arm: str, m: int) -> float | None:
        total = 0.0
        for shape, calls in CELL_CALLS.items():
            found = cell.get(
                (shape, m, arm_ipg(arm, m, shape_n[shape], shape_k[shape])))
            if found is None:
                return None
            total += calls * found["seconds_per_dispatch"]["mean"]
        return total * 1000.0

    # An arm cannot do better than picking the faster measured cell every time.
    # This is not a legal arm: it needs both M and k, and no arm here reads k.
    def oracle_ms(m: int) -> float | None:
        total = 0.0
        for shape, calls in CELL_CALLS.items():
            options = [cell.get((shape, m, SHIPPED[m])),
                       cell.get((shape, m, CROWN[m]))]
            options = [o for o in options if o is not None]
            if not options:
                return None
            total += calls * min(o["seconds_per_dispatch"]["mean"]
                                 for o in options)
        return total * 1000.0

    per_width = []
    for m in widths:
        row = {"m": m, "shipped_ipg": SHIPPED[m], "crown_ipg": CROWN[m],
               "local_rounds": LOCAL_ROUNDS.get(m, 0),
               "ranked_share_pct": RANKED_SHARE_PCT.get(m),
               "oracle_round_ms": oracle_ms(m)}
        for arm in ARMS:
            row[f"{arm}_round_ms"] = round_ms(arm, m)
        per_width.append(row)

    by_width = {row["m"]: row for row in per_width}
    baseline = "a_ship"

    def weighted_delta(arm: str, weights: dict[int, float],
                       total: float) -> float | None:
        out = 0.0
        for m in MOVED_WIDTHS:
            row = by_width.get(m)
            if row is None:
                return None
            a, b = row[f"{arm}_round_ms"], row[f"{baseline}_round_ms"]
            if a is None or b is None:
                return None
            out += (weights.get(m, 0.0) / total) * (a - b)
        return out

    def weighted_oracle(weights: dict[int, float],
                        total: float) -> float | None:
        out = 0.0
        for m in MOVED_WIDTHS:
            row = by_width.get(m)
            if row is None or row["oracle_round_ms"] is None:
                return None
            out += (weights.get(m, 0.0) / total) * (
                row["oracle_round_ms"] - row[f"{baseline}_round_ms"])
        return out

    local_w = {m: float(v) for m, v in LOCAL_ROUNDS.items()}
    ranked_w = dict(RANKED_SHARE_PCT)
    ranked_total = sum(RANKED_SHARE_PCT.values())

    mixes = {}
    for arm in ARMS:
        mixes[arm] = {
            "local_delta_ms_per_round":
                weighted_delta(arm, local_w, float(LOCAL_TOTAL_ROUNDS)),
            "ranked_delta_ms_per_round":
                weighted_delta(arm, ranked_w, ranked_total),
        }
    mixes["oracle_k_and_m"] = {
        "local_delta_ms_per_round":
            weighted_oracle(local_w, float(LOCAL_TOTAL_ROUNDS)),
        "ranked_delta_ms_per_round":
            weighted_oracle(ranked_w, ranked_total),
    }

    # The design question: can `out_vec_size` separate the cells that want the
    # crown's IPG from the cells that do not? It can only do so if, at every
    # width, the sign of (shipped - crown) is a monotone function of n.
    separable = []
    for m in MOVED_WIDTHS:
        rows = []
        for shape in CELL_CALLS:
            ship = cell.get((shape, m, SHIPPED[m]))
            crown = cell.get((shape, m, CROWN[m]))
            if ship is None or crown is None:
                continue
            rows.append({
                "shape": shape,
                "n": shape_n[shape],
                "k": shape_k[shape],
                "k_blocks_g64": shape_k[shape] // 64,
                "shipped_ms": ship["seconds_per_dispatch"]["mean"] * 1000.0,
                "crown_ms": crown["seconds_per_dispatch"]["mean"] * 1000.0,
                "crown_wins": (crown["seconds_per_dispatch"]["mean"]
                               < ship["seconds_per_dispatch"]["mean"]),
            })
        wins = [r for r in rows if r["crown_wins"]]
        losses = [r for r in rows if not r["crown_wins"]]
        # Separable by n only if no losing cell sits at or below a winning n.
        clash = [(w["shape"], l["shape"]) for w in wins for l in losses
                 if l["n"] <= w["n"]]
        separable.append({
            "m": m,
            "cells": sorted(rows, key=lambda r: r["n"]),
            "crown_wins_at": [r["shape"] for r in wins],
            "separable_by_out_vec_size": not clash,
            "n_collisions": [{"crown_wins": a, "shipped_wins": b}
                             for a, b in clash],
        })

    record = {
        "experiment": "e78",
        "rung": "2a-derived",
        "harness": "local",
        "instrument": "isolated",
        "derived_from": args.cells,
        "derivation": ("arm round cost = sum over the seven scored linear "
                       "families of (calls per round) x (measured cell time "
                       "for the IPG that arm selects at that width and n)"),
        "gate_qualified_for_timing": doc.get("gate_qualified_for_timing", False),
        "cool_gate_passed_real_gate":
            doc.get("cool_gate_passed_real_gate", False),
        "official_or_ranked_score": False,
        "device": doc.get("device"),
        "baseline_arm": baseline,
        "moved_widths": list(MOVED_WIDTHS),
        "per_width_round_ms": per_width,
        "mix_weighted_delta_vs_a_ship": mixes,
        "out_vec_size_separability": separable,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n")

    lines = ["# E78 rung 2a derived: arm prices from the measured cells", "",
             "Derived from absolute cell times. No extra GPU time. Rung 2b "
             "tests the additivity assumption.", "",
             "## Verify-round cost by width and arm (ms, derived)", "",
             "| M | ship IPG | crown IPG | a_ship | b_crown | c_hyb24928 | "
             "d_hyb8192 | oracle (M and k) | local rounds | ranked % |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_width:
        def fmt(key: str) -> str:
            value = row[key]
            return "-" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row['m']} | {row['shipped_ipg']} | {row['crown_ipg']} | "
            f"{fmt('a_ship_round_ms')} | {fmt('b_crown_round_ms')} | "
            f"{fmt('hybrid_24928_round_ms')} | {fmt('hybrid_8192_round_ms')} | "
            f"{fmt('oracle_round_ms')} | {row['local_rounds']} | "
            f"{row['ranked_share_pct']} |")

    lines += ["", "## Mix-weighted delta against a_ship (ms per verify round)",
              "", "Negative is faster. Only M = 5, 6 and 9 move, so every "
              "other width cancels exactly.", "",
              "| arm | local fixture mix | ranked pooled mix |",
              "|---|---:|---:|"]
    for arm, row in mixes.items():
        def fmt2(key: str) -> str:
            value = row[key]
            return "-" if value is None else f"{value:+.4f}"
        lines.append(f"| {arm} | {fmt2('local_delta_ms_per_round')} | "
                     f"{fmt2('ranked_delta_ms_per_round')} |")

    lines += ["", "## Can out_vec_size separate the cells that want IPG 3?", ""]
    for row in separable:
        lines.append(f"### M = {row['m']}")
        lines.append("")
        lines.append("| n | k | k blocks | shipped ms | crown ms | winner |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for c in row["cells"]:
            lines.append(
                f"| {c['n']} | {c['k']} | {c['k_blocks_g64']} | "
                f"{c['shipped_ms']:.5f} | {c['crown_ms']:.5f} | "
                f"{'crown' if c['crown_wins'] else 'shipped'} |")
        lines.append("")
        lines.append(f"separable_by_out_vec_size = "
                     f"{row['separable_by_out_vec_size']}")
        for clash in row["n_collisions"]:
            lines.append(f"  collision: {clash['crown_wins']} wants IPG 3 and "
                         f"{clash['shipped_wins']} does not, at the same or "
                         f"lower out_vec_size")
        lines.append("")

    pathlib.Path(args.markdown).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"e78_price_arms: wrote {args.out} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
