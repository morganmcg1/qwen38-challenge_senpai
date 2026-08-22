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
local ratio.

CAMPAIGN RULE 116 supersedes the single-number form of that conversion. The
published score is `median(raw_1 .. raw_8)`, and FINDING 196 shows the sorted
order of the eight prompts is stable, so the median is always the mean of the
beagle and essays ratios. A mass-weighted sum cannot represent that, because
the saturation cap is a property of sorting rather than of a linear
combination. This script therefore reports the saving as absolute microseconds
per round at each verify width, per shape, with the dispatch count it
multiplied by, and stops short of forming a headline ranked percentage.

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
LOCAL_ROUND_US = 196000.0  # RULE 115, local round

# FINDING 196. The sorted order of the eight hidden prompts is stable, so the
# published median is always the mean of the beagle and essays ratios. Only
# these two prompts carry marginal weight, and only their own width masses can
# convert a per-round microsecond saving into a score movement. The earlier
# F83 campaign-wide width table is withdrawn as an instrument: its weights sum
# to 0.9192, so it understated every broad mechanism by 8.79 % relative.
F196_PROMPT_WIDTH_MASS = {
    "beagle": {5: 0.0920, 6: 0.0739, 7: 0.0589, 8: 0.3388},
    "essays": {5: 0.1995, 6: 0.1558, 7: 0.1408, 8: 0.2958},
}
F196_MEDIAN_WEIGHT = {"beagle": 0.478, "essays": 0.522}


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
        # A width may be measured with the stock kernel alone, as a reference
        # arm. That width has no plan to choose between, so it carries its
        # stock timings and no plan optimum.
        keyed_choice = {
            name: min(cells, key=lambda c: table[(name, c)]["us_per_call"])
            for name in shapes
        } if cells else {}
        keyed_total = (
            weighted_total(table, shapes, keyed_choice) if cells else None
        )
        best_global = min(cells, key=lambda c: globals_[c]) if cells else None
        ship = shipped.get(m)
        per_width[m] = {
            "cells": cells,
            "stock_cells": stock,
            "global_totals_us": globals_,
            "best_global_cell": best_global,
            "best_global_total_us": (
                globals_[best_global] if best_global else None
            ),
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

    # Advisor feedback F3: edward's curve-perturbation arm consumes the step
    # into each width as an absolute in-situ figure, before and after the plan
    # change, plus the fraction of that step the plan removes. The E137
    # isolated-to-in-situ factor 0.7858 is applied here and named as applied.
    # The host transfer and the ranked round are deliberately NOT applied:
    # every figure below is g16s in-situ microseconds, harness=local.
    ladder = []
    for m in sorted(per_width):
        lo, hi = per_width.get(m - 1), per_width[m]
        if lo is None or lo["shipped_total_us"] is None:
            continue
        if hi["shipped_total_us"] is None:
            continue
        entry = {"into_width": m, "from_width": m - 1}
        for label, key in (
            ("shipped", "shipped_total_us"),
            ("best_global", "best_global_total_us"),
            ("shape_keyed", "shape_keyed_total_us"),
        ):
            step_us = hi[key] - lo[key]
            entry[f"{label}_isolated_step_us"] = step_us
            entry[f"{label}_in_situ_step_us"] = step_us * IN_SITU_TRANSFER
        base = entry["shipped_in_situ_step_us"]
        for label in ("best_global", "shape_keyed"):
            entry[f"{label}_step_reduction_fraction"] = (
                (base - entry[f"{label}_in_situ_step_us"]) / base
                if base else None
            )
        ladder.append(entry)
    if ladder:
        out["step_ladder"] = ladder
        out["step_ladder_transfer_applied"] = "e137_isolated_to_in_situ"
        out["step_ladder_transfer_value"] = IN_SITU_TRANSFER
        out["step_ladder_host_transfer_applied"] = False

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

    # CAMPAIGN RULE 116. Report the saving as absolute microseconds per round
    # at each verify width, per shape and in total, with the dispatch count
    # that produced the weighting. No mass is applied and no percentage of a
    # ranked round is formed here: the published score is the mean of the two
    # central sorted prompts, and a weighted sum cannot represent that.
    ladder = {}
    for m, block in sorted(per_width.items()):
        ship = block["shipped_cell"]
        if ship is None or ship not in block["global_totals_us"]:
            continue
        arms = {}
        for label, cell_of in (
            ("best_global", {n: block["best_global_cell"] for n in shapes}),
            ("shape_keyed", block["shape_keyed_choice"]),
        ):
            per_shape = {}
            for name in shapes:
                calls = shapes[name]["calls_per_verify"]
                one = (
                    table[(name, ship)]["us_per_call"]
                    - table[(name, cell_of[name])]["us_per_call"]
                )
                per_shape[name] = {
                    "plan": cell_of[name],
                    "calls_per_round": calls,
                    "saving_us_per_call_isolated": one,
                    "saving_us_per_round_isolated": one * calls,
                    "saving_us_per_round_in_situ": one * calls
                    * IN_SITU_TRANSFER,
                    "saving_us_per_round_ranked_host": one * calls
                    * IN_SITU_TRANSFER * HOST_TRANSFER,
                }
            total = sum(
                s["saving_us_per_round_isolated"] for s in per_shape.values()
            )
            arms[label] = {
                "per_shape": per_shape,
                "distinct_plans": sorted({s["plan"] for s in
                                          per_shape.values()}),
                "total_us_per_round_isolated": total,
                "total_us_per_round_in_situ": total * IN_SITU_TRANSFER,
                "total_us_per_round_ranked_host": total * IN_SITU_TRANSFER
                * HOST_TRANSFER,
            }
        ladder[m] = {"shipped_cell": ship, "arms": arms}
    out["width_ladder"] = ladder
    out["width_ladder_constants"] = {
        "in_situ_transfer": IN_SITU_TRANSFER,
        "in_situ_transfer_source": "E137, 30750.8 isolated / 39134.9 in-situ",
        "host_transfer": HOST_TRANSFER,
        "host_transfer_source": "RULE 115, M4 Pro g16s -> ranked M5 g17s",
        "rule": "CAMPAIGN RULE 116",
        "note": "absolute microseconds per round at each verify width; "
        "no width mass and no ranked percentage are applied here",
    }

    # FINDING 196 rebuild. Only beagle and essays can move the published
    # score, so the round saving is rebuilt for those two prompts from their
    # own width masses. This is a per-prompt round time, not a median, and it
    # is deliberately not summed into one ranked number.
    rebuild = {}
    for prompt, mass in F196_PROMPT_WIDTH_MASS.items():
        arms = {}
        for label in ("best_global", "shape_keyed"):
            covered = {m: mass[m] for m in mass if m in ladder}
            arms[label] = {
                "per_width_us": {
                    m: mass[m]
                    * ladder[m]["arms"][label]["total_us_per_round_in_situ"]
                    for m in covered
                },
                "saving_us_per_round_in_situ": sum(
                    mass[m]
                    * ladder[m]["arms"][label]["total_us_per_round_in_situ"]
                    for m in covered
                ),
                "saving_us_per_round_ranked_host": sum(
                    mass[m]
                    * ladder[m]["arms"][label][
                        "total_us_per_round_ranked_host"]
                    for m in covered
                ),
                "width_mass_covered": sum(covered.values()),
                "width_mass_total": sum(mass.values()),
            }
        rebuild[prompt] = {"width_mass": mass, "arms": arms}
    out["finding_196_prompt_rebuild"] = rebuild
    out["finding_196_note"] = (
        "published score = (beagle_raw + essays_raw) / 2. These are per-prompt "
        "round savings in microseconds. Converting them to a score needs each "
        "prompt's own baseline round time and the essays saturation cap, so "
        "this script does not form a headline ranked percentage."
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

    if "step_ladder" in result:
        lines += [
            "F3 step ladder, in-situ g16s microseconds  harness=local",
            "    transfer applied: E137 isolated -> in-situ "
            f"{result['step_ladder_transfer_value']:.4f}",
            "    host transfer and ranked round NOT applied; edward converts",
            "    %-11s %12s %12s %12s %9s %9s"
            % ("step", "shipped", "best global", "shape-keyed",
               "red glob", "red keyed"),
        ]
        for e in result["step_ladder"]:
            keyed = e["shape_keyed_step_reduction_fraction"]
            glob = e["best_global_step_reduction_fraction"]
            lines.append(
                "    %-11s %12.1f %12.1f %12.1f %8.4f %9.4f"
                % (
                    "%d -> %d" % (e["from_width"], e["into_width"]),
                    e["shipped_in_situ_step_us"],
                    e["best_global_in_situ_step_us"],
                    e["shape_keyed_in_situ_step_us"],
                    glob if glob is not None else float("nan"),
                    keyed if keyed is not None else float("nan"),
                )
            )
        lines.append("")

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

    if result["width_ladder"]:
        lines.append(
            "RULE 116 width ladder: microseconds saved per round, by width"
        )
        lines.append(
            "  in-situ applies %.4f (E137). ranked-host also applies %.2f."
            % (IN_SITU_TRANSFER, HOST_TRANSFER)
        )
        lines.append("  no width mass and no ranked percentage are applied")
        for m, block in sorted(result["width_ladder"].items()):
            lines.append("")
            lines.append(
                f"  M={m}   shipped {block['shipped_cell']}"
            )
            for label, arm in sorted(block["arms"].items()):
                lines.append(
                    f"    {label}  plans {','.join(arm['distinct_plans'])}"
                )
                lines.append(
                    "      %-34s %5s %11s %11s"
                    % ("shape", "calls", "us/call", "us/round")
                )
                for name, s in sorted(
                    arm["per_shape"].items(),
                    key=lambda kv: -abs(kv[1]["saving_us_per_round_isolated"]),
                ):
                    lines.append(
                        "      %-34s %5d %11.3f %11.1f"
                        % (name, s["calls_per_round"],
                           s["saving_us_per_call_isolated"],
                           s["saving_us_per_round_isolated"])
                    )
                lines.append(
                    "      %-34s %5s %11s %11.1f isolated"
                    % ("TOTAL", "", "", arm["total_us_per_round_isolated"])
                )
                lines.append(
                    "      %-34s %5s %11s %11.1f in-situ"
                    % ("", "", "", arm["total_us_per_round_in_situ"])
                )
                lines.append(
                    "      %-34s %5s %11s %11.1f ranked-host"
                    % ("", "", "", arm["total_us_per_round_ranked_host"])
                )
        lines.append("")
        lines.append(
            "FINDING 196 rebuild: only beagle and essays move the median"
        )
        for prompt, block in sorted(result["finding_196_prompt_rebuild"]
                                    .items()):
            arm = block["arms"]["shape_keyed"]
            best = block["arms"]["best_global"]
            masses = " ".join(
                f"m{m}={v:.4f}" for m, v in sorted(block["width_mass"].items())
            )
            lines.append(f"  {prompt}   {masses}")
            lines.append(
                "    per-round saving in-situ  best_global %8.1f us"
                "   shape_keyed %8.1f us"
                % (best["saving_us_per_round_in_situ"],
                   arm["saving_us_per_round_in_situ"])
            )
            lines.append(
                "    width mass covered by this sweep: %.4f of %.4f"
                % (arm["width_mass_covered"], arm["width_mass_total"])
            )
        lines.append("")
        lines.append(
            "  a headline ranked percentage is deliberately NOT formed here; "
            "see finding_196_note"
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
