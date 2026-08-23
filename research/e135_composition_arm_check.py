#!/usr/bin/env python3
"""Check every axis of a T29-A composition arm against one leg's own log.

The arms move four things at once: the dispatch table, the launch grid, the
leaf probe fraction, and whether width 2 is routed. Each is selected by an
environment variable and every one of those selectors fails soft. `Table`,
`Grid` and `ProbeArm` fall back to their compiled default on an unrecognized
string, and `MLX_E120_QMV_WIDTH2` acts only on the exact string `0`. A selector
that never reached the worker is therefore indistinguishable from one that
arrived and was ignored, unless the leg states what it actually did.

So this reads behaviour, not the parsed enum, wherever behaviour is recorded:

  table      the `plan` literal, which is the width plan interpolated into
             the Metal source the leg compiled
  grid       `columns_by_width`, read back off the dispatch argument
  probe      `probe_count`, the leaf count the kernel was actually given
  width 2    the `by_width` keys, which are the widths that were routed

`columns_by_width` is the strongest of the four. `launch(m:n:)` computes `m`
under wide and `ceil(m / ipg)` under tight, so the expected map is a function
of the plan and the grid together. An arm that got its table but not its grid,
or the reverse, produces a map that matches neither expectation.

Rule 101. Run a leg against an arm it is not and this must exit non-zero. The
session script does that once per arm before it times anything.

  python3 research/e135_composition_arm_check.py LOG.json --arm composed
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

# arm -> (table, grid, probe arm, lowest routed width)
ARMS = {
    "base": ("onepass67", "wide", "p25", 3),
    "composed": ("shipped", "tight", "p15", 2),
    "composed67": ("onepass67", "tight", "p15", 2),
    "composed678": ("onepass678", "tight", "p15", 2),
}

FRACTIONS = {"p25": 0.25, "p15": 0.15, "p10": 0.10}


def parse_plan(witness: str) -> dict[int, int]:
    """Map each planned width to its `ipg` from the emitted plan literal."""
    if "/" not in witness:
        return {}
    out = {}
    for triple in witness.split("/", 1)[1].split(","):
        parts = triple.split(":")
        if len(parts) == 3:
            out[int(parts[0])] = int(parts[1])
    return out


def check(log: dict, arm: str) -> list[str]:
    table, grid, probe, low = ARMS[arm]
    bad = []

    if log.get("table") != table:
        bad.append(f"table {log.get('table')!r} wanted {table!r}")
    if log.get("grid") != grid:
        bad.append(f"grid {log.get('grid')!r} wanted {grid!r}")
    if log.get("probe_arm") != probe:
        bad.append(f"probe_arm {log.get('probe_arm')!r} wanted {probe!r}")

    leaves, count = log.get("probe_leaves"), log.get("probe_count")
    if leaves:
        want = max(1, math.ceil(FRACTIONS[probe] * leaves))
        if count != want:
            bad.append(f"probe_count {count} wanted {want} for {probe} on"
                       f" {leaves} leaves, so the fraction missed the kernel")

    routed = sorted(int(w) for w in (log.get("by_width") or {}))
    if not routed:
        bad.append("no width was routed, so nothing is witnessed")
    elif min(routed) != low:
        bad.append(f"lowest routed width {min(routed)} wanted {low}")

    plan = parse_plan(log.get("plan", ""))
    columns = {int(w): c for w, c in (log.get("columns_by_width") or {}).items()}
    for width, actual in sorted(columns.items()):
        ipg = plan.get(width)
        if ipg is None:
            bad.append(f"width {width} launched but is absent from the plan")
            continue
        want = width if grid == "wide" else (width + ipg - 1) // ipg
        if actual != want:
            bad.append(f"width {width} launched {actual} columns, {grid} with"
                       f" ipg {ipg} wants {want}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = ap.parse_args()

    if not args.log.exists():
        print(f"e135_composition_arm_check: {args.log} is missing")
        return 2

    log = json.loads(args.log.read_text())
    bad = check(log, args.arm)
    table, grid, probe, low = ARMS[args.arm]
    print(f"arm {args.arm}: want table={table} grid={grid} probe={probe}"
          f" widths>={low}")
    print(f"  observed table={log.get('table')} grid={log.get('grid')}"
          f" probe={log.get('probe_arm')} count={log.get('probe_count')}"
          f" routed={sorted(int(w) for w in (log.get('by_width') or {}))}")
    print(f"  columns_by_width {log.get('columns_by_width')}")
    for line in bad:
        print(f"  MISMATCH {line}")
    print(f"  {'ok' if not bad else f'{len(bad)} mismatch(es)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
