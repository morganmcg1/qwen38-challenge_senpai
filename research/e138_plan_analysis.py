#!/usr/bin/env python3
"""E138: reduce the isolated (M, IPG, RPS) plan-surface blocks to a decision.

Reads one or more artifacts written by `research/e138_plan_surface.sh` and
answers three questions:

  1. Does any plan flatten the isolated dispatch-weighted M=5 -> M=6 step?
  2. Is the best plan global (one (IPG, RPS) for the width) or shape-keyed
     (one (IPG, RPS) per scored shape at that width)?
  3. What is the E138 primary metric, `e138_best_plan_isolated_step_reduction_pct`?

Drift correction
----------------
Every cell was ABBA-interleaved against the SAME reference cell inside one
timed block, so the reference median of that block carries the block's clock
state. The instrument sorts each arm, so per-rep pairing is gone; the ratio of
the two medians is the paired quantity that survives. Each cell is therefore
reported as

    corrected = median(cell) / median(reference in that block) * anchor

where `anchor` is the median of every reference sample of that shape across
the whole artifact. A cell measured in a cold block and a cell measured in a
hot block become comparable, and the reference cell itself is exactly `anchor`.

Ranked pricing
--------------
CAMPAIGN RULE 115: convert through absolute microseconds, never through a
local ratio. An isolated saving is priced as

    ranked_pct = saving_us * IN_SITU_TRANSFER * HOST_TRANSFER * mass / RANKED_ROUND_US

`harness=local` for every number this script produces.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

# The E137 in-situ anchor. The isolated dispatch-weighted step measured by that
# experiment was 30,750.8 us and the in-situ Route B round step was 39,134.9 us,
# so an isolated microsecond is worth 0.7858 in-situ microseconds.
IN_SITU_TRANSFER = 30750.8 / 39134.9
HOST_TRANSFER = 0.65  # RULE 115, M4 Pro g16s -> ranked M5 g17s
RANKED_ROUND_US = 54000.0  # RULE 115, F83-weighted ranked round
LOCAL_ROUND_US = 196000.0  # RULE 115, local round

# F83 ranked width masses.
RANKED_WIDTH_MASS = {
    4: 0.1017,
    5: 0.1251,
    6: 0.188,
    7: 0.211,
    8: 0.1871,
}


def median(values: list[float]) -> float:
    return statistics.median(values)


def load(paths: list[pathlib.Path]) -> list[dict]:
    out = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        side = path.with_suffix(".session.json")
        payload["_session"] = (
            json.loads(side.read_text()) if side.exists() else None
        )
        out.append(payload)
    return out


def corrected_table(artifacts: list[dict]) -> tuple[dict, dict, list[str]]:
    """(shape, cell) -> corrected microseconds per call, plus the shape table."""
    problems: list[str] = []
    # Drift correction divides by the block reference and multiplies by one
    # anchor per shape. Pooling two artifacts that used different reference
    # cells would silently mix two anchors, so refuse instead.
    references = {art["reference_cell"] for art in artifacts}
    if len(references) > 1:
        raise SystemExit(
            "refusing to pool artifacts with different reference cells: "
            + ", ".join(sorted(references))
        )
    grids = {art["grid"] for art in artifacts}
    if len(grids) > 1 and references != {"6:stock"}:
        raise SystemExit(
            "refusing to pool two launch grids unless the reference cell is "
            "grid-independent (`m:stock`); got reference "
            + ", ".join(sorted(references))
        )

    ref_samples: dict[str, list[float]] = {}
    for art in artifacts:
        for shape in art["shapes"]:
            if not shape["exactness_positive_control_rejects"]:
                problems.append(
                    f"{art['_path']}: {shape['name']} positive control did "
                    "not reject a one-step perturbation"
                )
            for row in shape["rows"]:
                ref_samples.setdefault(shape["name"], []).extend(
                    row["reference_samples"]
                )

    anchor = {name: median(s) for name, s in ref_samples.items()}

    table: dict[tuple[str, str], dict] = {}
    shapes: dict[str, dict] = {}
    for art in artifacts:
        for shape in art["shapes"]:
            shapes[shape["name"]] = {
                "k": shape["k"],
                "n": shape["n"],
                "calls_per_verify": shape["calls_per_verify"],
            }
            for row in shape["rows"]:
                if not row["matches_incumbent_bitwise"]:
                    problems.append(
                        f"{art['_path']}: {shape['name']} {row['cell']} is not "
                        "bit-exact against the incumbent gate "
                        f"(max_abs_delta={row['max_abs_delta_vs_incumbent']})"
                    )
                block_ref = median(row["reference_samples"])
                raw = median(row["samples"])
                ratio = raw / block_ref
                # Two grids in one analysis would collide on the cell label,
                # so the grid becomes part of the identity.
                label = row["cell"]
                if len(grids) > 1 and not row["cell"].endswith(":stock"):
                    label = f"{row['cell']}@{art['grid']}"
                key = (shape["name"], label)
                entry = table.setdefault(
                    key,
                    {
                        "shape": shape["name"],
                        "cell": label,
                        "plan": row["cell"],
                        "grid": art["grid"],
                        "m": row["m"],
                        "ipg": row["ipg"],
                        "rps": row["rps"],
                        "passes": row["passes"],
                        "launched_columns": row["launched_columns"],
                        "threadgroups_per_column": row[
                            "threadgroups_per_column"
                        ],
                        "ratios": [],
                        "raw_us": [],
                        "tap_us": [],
                    },
                )
                entry["ratios"].append(ratio)
                entry["raw_us"].append(raw * 1e6)
                entry["tap_us"].append(row["tap_overhead_seconds_per_call"] * 1e6)

    for key, entry in table.items():
        entry["ratio"] = median(entry["ratios"])
        entry["us_per_call"] = entry["ratio"] * anchor[entry["shape"]] * 1e6
        entry["raw_us_per_call"] = median(entry["raw_us"])
        entry["tap_us_per_call"] = median(entry["tap_us"])
        entry["blocks"] = len(entry["ratios"])
        entry["ratio_spread"] = (
            max(entry["ratios"]) - min(entry["ratios"])
            if len(entry["ratios"]) > 1
            else 0.0
        )

    return table, shapes, problems


def weighted_total(table: dict, shapes: dict, choice: dict[str, str]) -> float:
    """Dispatch-weighted microseconds of one verify forward under `choice`."""
    total = 0.0
    for name, cell in choice.items():
        entry = table[(name, cell)]
        total += entry["us_per_call"] * shapes[name]["calls_per_verify"]
    return total


def sort_key(label: str) -> tuple:
    plan, _, grid = label.partition("@")
    parts = plan.split(":")
    if parts[-1] == "stock":
        return (int(parts[0]), 99, 99, grid)
    return tuple(int(p) for p in parts) + (grid,)


def cells_at_width(table: dict, m: int) -> list[str]:
    return sorted(
        {entry["cell"] for entry in table.values() if entry["m"] == m},
        key=sort_key,
    )


def complete_cells(table: dict, shapes: dict, m: int) -> list[str]:
    """Cells measured on every shape, so a weighted total is well defined."""
    out = []
    for cell in cells_at_width(table, m):
        if all((name, cell) in table for name in shapes):
            out.append(cell)
    return out


def analyse(artifacts: list[dict], shipped: dict[int, str]) -> dict:
    table, shapes, problems = corrected_table(artifacts)
    widths = sorted({entry["m"] for entry in table.values()})

    per_width: dict[int, dict] = {}
    for m in widths:
        measured = complete_cells(table, shapes, m)
        if not measured:
            continue
        # The stock MLX kernel is a reference arm, not a shippable plan, so it
        # never competes for the plan optimum. It is reported beside it.
        stock = [c for c in measured if c.endswith(":stock")]
        cells = [c for c in measured if not c.endswith(":stock")]
        globals_: dict[str, float] = {}
        for cell in measured:
            globals_[cell] = weighted_total(
                table, shapes, {name: cell for name in shapes}
            )
        keyed_choice = {}
        for name in shapes:
            keyed_choice[name] = min(
                cells, key=lambda c: table[(name, c)]["us_per_call"]
            )
        keyed_total = weighted_total(table, shapes, keyed_choice)
        best_global = min(cells, key=lambda c: globals_[c]) if cells else None
        ship = shipped.get(m)
        per_width[m] = {
            "cells": cells,
            "stock_cells": stock,
            "global_totals_us": globals_,
            "best_global_cell": best_global,
            "best_global_total_us": globals_[best_global],
            "shipped_cell": ship,
            "shipped_total_us": globals_.get(ship) if ship else None,
            "shape_keyed_choice": keyed_choice,
            "shape_keyed_total_us": keyed_total,
            "shape_keyed_distinct_plans": sorted(set(keyed_choice.values())),
            "stock_total_us": {c: globals_[c] for c in stock},
        }

    out = {
        "harness": "local",
        "artifacts": [a["_path"] for a in artifacts],
        "grid": sorted({a["grid"] for a in artifacts}),
        "reference_cell": sorted({a["reference_cell"] for a in artifacts}),
        "arm": sorted({a["arm"] for a in artifacts}),
        "shapes": shapes,
        "cells": {
            f"{k[0]}|{k[1]}": {
                key: value
                for key, value in entry.items()
                if key not in ("ratios", "raw_us", "tap_us")
            }
            for k, entry in sorted(table.items())
        },
        "per_width": per_width,
        "problems": problems,
        "sessions": [a["_session"] for a in artifacts],
    }

    # The E138 step and the primary metric. A plan TABLE chooses a plan at
    # every width, so the step it produces is `total6(its plan) -
    # total5(its plan)`, not `total6(its plan) - total5(shipped)`. Both are
    # reported, because a table that also speeds M=5 can widen the step while
    # making the round strictly cheaper, and that must not read as a loss.
    if 5 in per_width and 6 in per_width:
        ship5 = per_width[5]["shipped_total_us"]
        ship6 = per_width[6]["shipped_total_us"]
        if ship5 is not None and ship6 is not None:
            shipped_step = ship6 - ship5
            variants = {
                "best_global": (
                    per_width[5]["best_global_total_us"],
                    per_width[6]["best_global_total_us"],
                ),
                "shape_keyed": (
                    per_width[5]["shape_keyed_total_us"],
                    per_width[6]["shape_keyed_total_us"],
                ),
            }
            step = {
                "shipped_m5_us": ship5,
                "shipped_m6_us": ship6,
                "shipped_step_us": shipped_step,
            }
            for label, (m5, m6) in variants.items():
                step[f"{label}_m5_us"] = m5
                step[f"{label}_m6_us"] = m6
                step[f"{label}_step_us"] = m6 - m5
                step[f"{label}_reduction_pct"] = (
                    100.0 * (shipped_step - (m6 - m5)) / shipped_step
                )
                # The M=6 total against the shipped M=6 total. This is the
                # quantity that actually pays in the round.
                step[f"{label}_m6_saving_us"] = ship6 - m6
            out["step"] = step
            best = max(
                step["best_global_reduction_pct"],
                step["shape_keyed_reduction_pct"],
            )
            out["e138_best_plan_isolated_step_reduction_pct"] = best
            # A cliff is plan-invariant when no plan removes a useful part of
            # it. The pre-registered threshold for "useful" is 3 %.
            out["e138_cliff_is_plan_invariant"] = 1.0 if best < 3.0 else 0.0

    # Item 2: a shape-keyed table multiplies compiled entry points. Count the
    # distinct plans the table needs across every measured width.
    keyed_plans = sorted(
        {
            cell
            for block in per_width.values()
            for cell in block["shape_keyed_distinct_plans"]
        },
        key=sort_key,
    )
    shipped_plans = sorted(
        {
            block["shipped_cell"]
            for block in per_width.values()
            if block["shipped_cell"]
        },
        key=sort_key,
    )
    out["pipeline_count"] = {
        "shipped_distinct_plans": shipped_plans,
        "shipped_pipelines": len(shipped_plans),
        "shape_keyed_distinct_plans": keyed_plans,
        "shape_keyed_pipelines": len(keyed_plans),
        "finding_133_ceiling": 14,
        "over_finding_133_ceiling": len(keyed_plans) > 14,
        "note": "FINDING 167.1 charges +0.35 % per extra resident pipeline "
        "even when the plans are identical, so a table is only worth its "
        "pipelines if the plan saving clears that tax.",
    }

    # RULE 115 ranked pricing of every width where a plan beats the shipped one.
    pricing = {}
    for m, block in per_width.items():
        ship = block["shipped_total_us"]
        if ship is None:
            continue
        for label, total in (
            ("best_global", block["best_global_total_us"]),
            ("shape_keyed", block["shape_keyed_total_us"]),
        ):
            saving = ship - total
            mass = RANKED_WIDTH_MASS.get(m, 0.0)
            pricing[f"m{m}_{label}"] = {
                "saving_us_isolated": saving,
                "in_situ_us": saving * IN_SITU_TRANSFER,
                "ranked_us_at_this_width": saving
                * IN_SITU_TRANSFER
                * HOST_TRANSFER,
                "ranked_mass": mass,
                "ranked_round_pct": 100.0
                * saving
                * IN_SITU_TRANSFER
                * HOST_TRANSFER
                * mass
                / RANKED_ROUND_US,
            }
    out["ranked_pricing"] = pricing
    out["ranked_pricing_constants"] = {
        "in_situ_transfer": IN_SITU_TRANSFER,
        "host_transfer": HOST_TRANSFER,
        "ranked_round_us": RANKED_ROUND_US,
        "local_round_us": LOCAL_ROUND_US,
        "rule": "CAMPAIGN RULE 115",
    }
    if pricing:
        out["ranked_pricing_total_pct"] = sum(
            block["ranked_round_pct"]
            for name, block in pricing.items()
            if name.endswith("_shape_keyed")
        )
    return out


def matrix(result: dict) -> str:
    """Item 1a: the seven scored shapes against the plan, microseconds per
    call, with the dispatch count beside each row."""
    lines = []
    for m, block in sorted(result["per_width"].items()):
        cells = block["cells"] + block["stock_cells"]
        ship = block["shipped_cell"]
        lines.append(f"M={m}  per-shape microseconds per call  (shipped {ship})")
        lines.append(
            "%-34s %4s " % ("shape", "disp")
            + "".join("%11s" % c for c in cells)
        )
        for name in sorted(result["shapes"]):
            row = result["cells"]
            disp = result["shapes"][name]["calls_per_verify"]
            values = [row[f"{name}|{c}"]["us_per_call"] for c in cells]
            base = row[f"{name}|{ship}"]["us_per_call"] if ship else None
            lines.append(
                "%-34s %4d " % (name, disp)
                + "".join("%11.1f" % v for v in values)
            )
            if base:
                lines.append(
                    "%-34s %4s " % ("    vs shipped", "")
                    + "".join(
                        "%10.1f%%" % (100.0 * (v - base) / base) for v in values
                    )
                )
        lines.append(
            "%-34s %4s " % ("WEIGHTED TOTAL", "")
            + "".join("%11.0f" % block["global_totals_us"][c] for c in cells)
        )
        lines.append("")
    return "\n".join(lines)


def report(result: dict) -> str:
    lines = ["E138 plan surface  harness=local", ""]
    for m, block in sorted(result["per_width"].items()):
        lines.append(f"width M={m}   shipped={block['shipped_cell']}")
        totals = block["global_totals_us"]
        ship = block["shipped_total_us"]
        for cell in block["cells"] + block["stock_cells"]:
            delta = "" if ship is None else f"  {totals[cell] - ship:+9.1f} us"
            mark = " <-- shipped" if cell == block["shipped_cell"] else ""
            if cell.endswith(":stock"):
                mark = " <-- stock MLX, reference only"
            lines.append(
                f"    global {cell:<8} {totals[cell]:10.1f} us{delta}{mark}"
            )
        keyed = block["shape_keyed_total_us"]
        delta = "" if ship is None else f"  {keyed - ship:+9.1f} us"
        lines.append(f"    shape-keyed      {keyed:10.1f} us{delta}")
        for name, cell in sorted(block["shape_keyed_choice"].items()):
            lines.append(f"        {name:<34} {cell}")
        lines.append("")

    if "step" in result:
        step = result["step"]
        lines += [
            "isolated dispatch-weighted M=5 -> M=6 step",
            f"    shipped        {step['shipped_step_us']:10.1f} us",
            f"    best global    {step['best_global_step_us']:10.1f} us"
            f"   reduction {step['best_global_reduction_pct']:+6.2f} %"
            f"   M=6 saving {step['best_global_m6_saving_us']:+8.1f} us",
            f"    shape-keyed    {step['shape_keyed_step_us']:10.1f} us"
            f"   reduction {step['shape_keyed_reduction_pct']:+6.2f} %"
            f"   M=6 saving {step['shape_keyed_m6_saving_us']:+8.1f} us",
            "",
            "PRIMARY  e138_best_plan_isolated_step_reduction_pct = "
            f"{result['e138_best_plan_isolated_step_reduction_pct']:.2f}",
            "SECONDARY e138_cliff_is_plan_invariant = "
            f"{result['e138_cliff_is_plan_invariant']:.1f}",
            "",
        ]

    pipes = result["pipeline_count"]
    lines += [
        "pipeline count (FINDING 133 ceiling 14, FINDING 167.1 tax +0.35 %/each)",
        f"    shipped      {pipes['shipped_pipelines']:2d}  "
        f"{', '.join(pipes['shipped_distinct_plans'])}",
        f"    shape-keyed  {pipes['shape_keyed_pipelines']:2d}  "
        f"{', '.join(pipes['shape_keyed_distinct_plans'])}",
        f"    over ceiling: {pipes['over_finding_133_ceiling']}",
        "",
    ]

    if result["ranked_pricing"]:
        lines.append("RULE 115 ranked pricing (absolute microseconds)")
        for name, block in sorted(result["ranked_pricing"].items()):
            lines.append(
                f"    {name:<24} saving {block['saving_us_isolated']:9.1f} us"
                f"  mass {block['ranked_mass']:.4f}"
                f"  ranked {block['ranked_round_pct']:+6.3f} %"
            )
        lines.append("")

    if result["problems"]:
        lines.append("PROBLEMS")
        lines += [f"    {p}" for p in result["problems"]]
    else:
        lines.append("every cell is bit-exact and every positive control rejects")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument(
        "--shipped",
        default="5:5:5:4,6:6:6:4,7:7:7:4,8:8:4:4,9:9:3:4",
        help="m:m:ipg:rps entries naming the shipped cell at each width",
    )
    args = parser.parse_args()

    shipped = {}
    for entry in args.shipped.split(","):
        parts = entry.split(":")
        shipped[int(parts[0])] = ":".join(parts[1:])

    result = analyse(load(args.artifacts), shipped)
    if args.matrix:
        print(matrix(result))
    text = report(result)
    print(text)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
