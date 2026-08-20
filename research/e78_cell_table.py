#!/usr/bin/env python3
"""Turn the E78 rung 2a cell session into the advisor's per-cell table.

  python3 research/e78_cell_table.py --cells RAW.json --out CELLS.json
                                     [--markdown CELLS.md]

The deliverable is ABSOLUTE time per cell at both group counts, per width and
per `n`, never collapsed into one leg number. Ratios and reweighted round costs
are reported after the absolute times, clearly marked as derived.

The reweighting is given under two width mixes, because they disagree badly:
the local public fixture (E66, 78 rounds) and the ranked pooled mix. A cell
table is the only form in which a reader can apply their own mix.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e73_pairs import CROWN, SHIPPED  # noqa: E402

CORES_LOCAL = 20      # M4 Pro, applegpu_g16s
CORES_RANKED = 40     # M5 Max, applegpu_g17s

# Calls per verify round, E33 section 8.2 via research/xgroup_census.py.
# `linear_attn.out_proj` and `full_attn.o_proj` are the same (6144, 5120) cell,
# so the measured cell carries the calls of both families.
CELL_CALLS = {
    "mlp.gate_up_fused": 64,
    "mlp.down": 64,
    "linear_attn.in_proj_fused_qkvzba": 48,
    "full_attn.qkv_proj_fused": 16,
    "head.lm_head": 1,
    "linear_attn.out_proj": 48 + 16,
}

# Round share by width. Local is the public fixture under the shipped schedule
# (E66, 78 rounds). Ranked is the pooled receipt mix the advisor supplied.
LOCAL_ROUND_SHARE = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
RANKED_ROUND_SHARE_PCT = {4: 14.2, 5: 24.1, 6: 33.4, 7: 12.2, 8: 7.35, 9: 5.75}


def summarize(values: list[float]) -> dict:
    out = {
        "n_legs": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) > 1:
        out["stdev"] = statistics.stdev(values)
        out["sem"] = out["stdev"] / math.sqrt(len(values))
        out["rel_sem_pct"] = 100.0 * out["sem"] / out["mean"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown")
    args = ap.parse_args()

    raw = json.loads(pathlib.Path(args.cells).read_text())
    if args.config:
        # Provenance and gate state live in the harness config, not the result.
        for key, value in json.loads(
                pathlib.Path(args.config).read_text()).items():
            raw.setdefault(key, value)
    arms = {a["arm"]: a for a in raw["arms"]}

    cells = []
    for shape in raw["shapes"]:
        name, k, n = shape["shape"], shape["k"], shape["n"]
        per_arm: dict[str, list[float]] = {}
        for leg in shape["legs"]:
            per_arm.setdefault(leg["arm"], []).append(leg["seconds_per_dispatch"])
        for arm, values in sorted(per_arm.items()):
            m, ipg = arms[arm]["m"], arms[arm]["ipg"]
            g = arms[arm]["groups"]
            working_tgs = g * math.ceil(n / 8)
            stats = summarize(values)
            bytes_moved = g * shape["bytes_per_stream"]
            cells.append({
                "shape": name,
                "k": k,
                "n": n,
                "m": m,
                "ipg": ipg,
                "groups": g,
                "is_shipped": SHIPPED.get(m) == ipg,
                "is_crown": CROWN.get(m) == ipg,
                "working_threadgroups": working_tgs,
                "threadgroups_per_core_local": working_tgs / CORES_LOCAL,
                "threadgroups_per_core_ranked": working_tgs / CORES_RANKED,
                "k_blocks_g64": k // 64,
                "bytes_per_stream": shape["bytes_per_stream"],
                "gb_moved": bytes_moved / 1e9,
                "seconds_per_dispatch": stats,
                "ms_per_dispatch": {key: (value * 1000.0 if key != "n_legs"
                                          and key != "rel_sem_pct" else value)
                                    for key, value in stats.items()},
                "ms_per_gb": stats["mean"] * 1000.0 / (bytes_moved / 1e9),
                "shape_entry_gpu_temp_c": shape["entry_gpu_temp_c"],
                "shape_exit_gpu_temp_c": shape["exit_gpu_temp_c"],
                "inner_dispatches_per_leg": shape["inner"],
            })

    # The contrast: shipped group count against the crown's, per cell. Absolute
    # first, ratio second.
    contrasts = []
    by_key = {(c["shape"], c["m"], c["ipg"]): c for c in cells}
    widths = sorted({c["m"] for c in cells})
    for shape in raw["shapes"]:
        name = shape["shape"]
        for m in widths:
            ship = by_key.get((name, m, SHIPPED.get(m)))
            crown = by_key.get((name, m, CROWN.get(m)))
            if ship is None or crown is None or ship["ipg"] == crown["ipg"]:
                # M = 4 ships the same count in both tables; contrast it against
                # the only other legal partition instead.
                alt = [c for c in cells
                       if c["shape"] == name and c["m"] == m
                       and c["ipg"] != (SHIPPED.get(m) or -1)]
                if ship is None or not alt:
                    continue
                crown = alt[0]
            delta = (ship["seconds_per_dispatch"]["mean"]
                     - crown["seconds_per_dispatch"]["mean"])
            contrasts.append({
                "shape": name,
                "n": shape["n"],
                "k": shape["k"],
                "m": m,
                "shipped_ipg": ship["ipg"],
                "other_ipg": crown["ipg"],
                "shipped_ms": ship["seconds_per_dispatch"]["mean"] * 1000.0,
                "other_ms": crown["seconds_per_dispatch"]["mean"] * 1000.0,
                "delta_ms_shipped_minus_other": delta * 1000.0,
                "ratio_shipped_over_other": (
                    ship["seconds_per_dispatch"]["mean"]
                    / crown["seconds_per_dispatch"]["mean"]),
                "shipped_wins": delta < 0,
                "sem_ms_shipped": ship["seconds_per_dispatch"].get("sem", 0.0) * 1000.0,
                "sem_ms_other": crown["seconds_per_dispatch"].get("sem", 0.0) * 1000.0,
            })

    # Derived, and only derived: what one verify round costs under each table if
    # every family paid its measured cell time. Reported under both width mixes
    # because they disagree by 7.6x at M = 9.
    def round_cost(m: int, table: dict[int, int]) -> float | None:
        total = 0.0
        for name, calls in CELL_CALLS.items():
            cell = by_key.get((name, m, table.get(m)))
            if cell is None:
                return None
            total += calls * cell["seconds_per_dispatch"]["mean"]
        return total * 1000.0

    rounds = []
    for m in widths:
        ship_cost = round_cost(m, SHIPPED)
        crown_cost = round_cost(m, CROWN)
        rounds.append({
            "m": m,
            "shipped_ipg": SHIPPED.get(m),
            "crown_ipg": CROWN.get(m),
            "shipped_round_ms": ship_cost,
            "crown_round_ms": crown_cost,
            "delta_ms": None if ship_cost is None or crown_cost is None
            else ship_cost - crown_cost,
            "local_rounds": LOCAL_ROUND_SHARE.get(m, 0),
            "ranked_share_pct": RANKED_ROUND_SHARE_PCT.get(m),
        })

    def mix_total(key: str, weights: dict[int, float]) -> float | None:
        total, seen = 0.0, 0.0
        for row in rounds:
            weight = weights.get(row["m"])
            if weight is None or row[key] is None:
                continue
            total += weight * row[key]
            seen += weight
        return total / seen if seen else None

    local_weights = {m: float(c) for m, c in LOCAL_ROUND_SHARE.items()}
    ranked_weights = {m: v for m, v in RANKED_ROUND_SHARE_PCT.items()}

    record = {
        "experiment": "e78",
        "rung": "2a",
        "harness": "local",
        "instrument": "isolated",
        "instrument_note": (
            "E73 cell harness: every arm compiled into one process, shapes "
            "walked shape-major, arms timed back to back in a palindrome "
            "order within each shape. No surrounding round, no model cache."),
        "cool_gate_passed_real_gate": raw.get("cool_gate_passed_real_gate", False),
        "gate_qualified_for_timing": raw.get("gate_qualified_for_timing", False),
        "official_or_ranked_score": False,
        "device": raw.get("device"),
        "gpu_cores_local": CORES_LOCAL,
        "gpu_cores_ranked": CORES_RANKED,
        "reps": raw.get("reps"),
        "warmup_reps": raw.get("warmup_reps"),
        "target_bytes": raw.get("target_bytes"),
        "session_entry_gpu_temp_c": raw.get("session_entry_gpu_temp_c"),
        "session_exit_gpu_temp_c": raw.get("session_exit_gpu_temp_c"),
        "head_sha": raw.get("head_sha"),
        "source_sha256": raw.get("source_sha256"),
        "grid": raw.get("grid"),
        "excluded_shape": {
            "name": "head.compact_draft_vocab",
            "n": 98336,
            "reason": (
                "draft-side readout, bits = 2 and M = 1 only, so it never "
                "reaches the bits == 4 affine gate this table lives behind; "
                "xgroup_census prices it at 0 calls per verify round"),
        },
        "cells": cells,
        "contrasts": contrasts,
        "round_costs": rounds,
        "derived_mix_totals": {
            "note": ("Renormalized over the measured widths only. Widths the "
                     "session did not measure carry no weight, so this is a "
                     "conditional mean round cost, not a leg time."),
            "mix_widths": widths,
            "local_fixture_weighted_shipped_round_ms":
                mix_total("shipped_round_ms", local_weights),
            "local_fixture_weighted_crown_round_ms":
                mix_total("crown_round_ms", local_weights),
            "ranked_weighted_shipped_round_ms":
                mix_total("shipped_round_ms", ranked_weights),
            "ranked_weighted_crown_round_ms":
                mix_total("crown_round_ms", ranked_weights),
        },
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    lines = ["# E78 rung 2a: absolute isolated cell times", "",
             f"device {record['device']}, reps {record['reps']}, "
             f"harness=local, instrument=isolated, "
             f"gate_qualified_for_timing={record['gate_qualified_for_timing']}",
             "",
             "## Absolute time per cell (ms per dispatch)", "",
             "| shape | k | n | M | IPG | groups | working TGs | TGs/core (20) | "
             "GB moved | ms | +/- SEM ms | ms/GB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        lines.append(
            f"| {c['shape']} | {c['k']} | {c['n']} | {c['m']} | {c['ipg']} | "
            f"{c['groups']} | {c['working_threadgroups']} | "
            f"{c['threadgroups_per_core_local']:.1f} | {c['gb_moved']:.4f} | "
            f"{c['seconds_per_dispatch']['mean'] * 1000:.5f} | "
            f"{c['seconds_per_dispatch'].get('sem', 0.0) * 1000:.5f} | "
            f"{c['ms_per_gb']:.4f} |")
    lines += ["", "## Shipped group count against the alternative", "",
              "| shape | n | M | shipped IPG | other IPG | shipped ms | "
              "other ms | delta ms | ratio | shipped wins |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in contrasts:
        lines.append(
            f"| {row['shape']} | {row['n']} | {row['m']} | {row['shipped_ipg']} | "
            f"{row['other_ipg']} | {row['shipped_ms']:.5f} | "
            f"{row['other_ms']:.5f} | {row['delta_ms_shipped_minus_other']:+.5f} | "
            f"{row['ratio_shipped_over_other']:.4f} | "
            f"{'yes' if row['shipped_wins'] else 'no'} |")
    lines += ["", "## Derived: one verify round from the measured cells", "",
              "Cell time times calls per round, summed over the seven scored "
              "linear families. Derived, not measured end to end.", "",
              "| M | shipped IPG | crown IPG | shipped ms | crown ms | "
              "delta ms | local rounds | ranked share % |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rounds:
        if row["shipped_round_ms"] is None or row["crown_round_ms"] is None:
            continue
        lines.append(
            f"| {row['m']} | {row['shipped_ipg']} | {row['crown_ipg']} | "
            f"{row['shipped_round_ms']:.4f} | {row['crown_round_ms']:.4f} | "
            f"{row['delta_ms']:+.4f} | {row['local_rounds']} | "
            f"{row['ranked_share_pct']} |")
    if args.markdown:
        pathlib.Path(args.markdown).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\ne78_cell_table: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
