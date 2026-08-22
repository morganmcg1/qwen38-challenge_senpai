#!/usr/bin/env python3
"""E138 F5: `Table.shipped` against `onePass67`, at M=6 and M=7, in both grids.

    usage: research/e138_f5_report.py ARTIFACT...

F5 named `Table.shipped` `[(6,3,4),(7,4,4)]` as the reference plan on the board,
because it is the verbatim width plan of the crown `08b67f12`. `onePass67` is
our promoted `[(6,6,4),(7,7,4)]`. This report answers F5 in its requested
order: the `(plan x grid)` interaction against the pre-registered band first,
then the two named points on the N axis, then the plan main effect in absolute
microseconds per round at each width.

Every number is `harness=local`, isolated QMV dispatch, dispatch-weighted with
the shipped counts, and drift-corrected against the grid-independent `6:stock`
anchor.
"""

from __future__ import annotations

import importlib.util
import pathlib
import statistics
import sys

# RULE 115 transfers. Isolated -> in-situ is E137's measured 30,750.8 / 39,134.9
# and in-situ -> the ranked M5 host is the campaign's 0.65.
IN_SITU_TRANSFER = 30750.8 / 39134.9
HOST_TRANSFER = 0.65

BAND_US = (1200.0, 3400.0)
ARMS = ((6, "6:6:4", "6:3:4"), (7, "7:7:4", "7:4:4"))
NAMED_N_AXIS = ("full_attn.qkv_proj_fused", "linear_attn.in_proj_fused_qkvzba")


def _grid_control():
    spec = importlib.util.spec_from_file_location(
        "e138_grid_control",
        pathlib.Path(__file__).with_name("e138_grid_control.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted(data: dict, plan: str, grid: str) -> float:
    return sum(
        data["us"][(n, plan, grid)] * data["dispatch"][n]
        for n in data["dispatch"]
    )


def main() -> int:
    paths = [pathlib.Path(a) for a in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    gc = _grid_control()
    pooled = gc.load(paths)

    print("E138 F5 report   harness=local   %d replicate(s)" % len(paths))
    print("Table.shipped = the crown 08b67f12 width plan, verbatim.")
    print("onePass67     = our promoted plan.")
    print()
    print("=" * 74)
    print("1. THE (plan x grid) INTERACTION, ABSOLUTE MICROSECONDS PER ROUND")
    print("=" * 74)
    print("Sign convention: interaction = advantage(tight) - advantage(wide),")
    print("advantage = us(onePass67) - us(Table.shipped). A POSITIVE value")
    print("means Table.shipped buys more once the grid tightens, which is the")
    print("direction F5 pre-registered at +2,266 us.")
    print()
    for width, one_pass, shipped in ARMS:
        report = gc.factorial(paths, one_pass)
        value = report["interaction_us_where_grid_differs"][shipped]
        floor = report["interaction_resolution_floor_us"]
        print("width %d   %s against %s" % (width, shipped, one_pass))
        print("    measured interaction      %+10.1f us" % value)
        print("    pre-registered band       %+10.1f to %+.1f us" % BAND_US)
        print("    inside the band           %s"
              % ("YES" if BAND_US[0] <= value <= BAND_US[1] else "NO"))
        print("    resolution floor          %10.1f us" % floor)
        print("    |interaction| over floor  %s"
              % ("YES" if abs(value) > floor else "NO"))
        print("    band is testable here     %s"
              % ("YES" if floor < BAND_US[0] else
                 "NO, the floor is above the band"))
        for row in report["null_control_total_us"]:
            print("      null control %-14s true 0 us, measured %+9.1f us"
                  % (row["plan"], row["measured_interaction_us"]))
        for row in report["matched_geometry_total_us"]:
            print("      matched pair %-14s true 0 us, measured %+9.1f us"
                  % ("/".join(row["plans"]),
                     row["measured_interaction_spread_us"]))
        print()

    print("=" * 74)
    print("2. THE TWO NAMED POINTS ON THE N AXIS")
    print("=" * 74)
    print("%-34s %6s %6s %6s %10s %9s %10s"
          % ("shape", "N", "plan", "grid", "us/call", "spread", "us/round"))
    for name in NAMED_N_AXIS:
        for _, one_pass, shipped in ARMS:
            for plan in (one_pass, shipped):
                for grid in ("wide", "tight"):
                    value = pooled["us"][(name, plan, grid)]
                    print("%-34s %6d %6s %6s %10.3f %8.2f %% %10.1f"
                          % (name, pooled["n"][name], plan, grid, value,
                             pooled["spread_pct"][(name, plan, grid)],
                             value * pooled["dispatch"][name]))
        print()

    print("=" * 74)
    print("3. THE PLAN MAIN EFFECT, ABSOLUTE MICROSECONDS PER ROUND")
    print("=" * 74)
    print("Positive means onePass67 is cheaper than Table.shipped.")
    print()
    print("%-6s %-6s %11s %11s %11s %11s %11s"
          % ("width", "grid", "onePass67", "Table", "onePass-Tbl",
             "in-situ", "ranked-host"))
    per_replicate: dict[tuple[int, str], list[float]] = {}
    for path in paths:
        single = gc.load([path])
        for width, one_pass, shipped in ARMS:
            for grid in ("wide", "tight"):
                per_replicate.setdefault((width, grid), []).append(
                    weighted(single, one_pass, grid)
                    - weighted(single, shipped, grid)
                )
    for width, one_pass, shipped in ARMS:
        for grid in ("tight", "wide"):
            a, b = (weighted(pooled, one_pass, grid),
                    weighted(pooled, shipped, grid))
            print("%-6d %-6s %11.1f %11.1f %+11.1f %+11.1f %+11.1f"
                  % (width, grid, a, b, a - b,
                     (a - b) * IN_SITU_TRANSFER,
                     (a - b) * IN_SITU_TRANSFER * HOST_TRANSFER))
    print()
    print("Per-replicate main effect, and whether its sign ever changes")
    print("%-6s %-6s %10s %10s %10s %8s %12s"
          % ("width", "grid", "rep1", "rep2", "rep3", "spread", "sign stable"))
    for (width, grid), values in sorted(per_replicate.items()):
        stable = min(values) * max(values) > 0
        print("%-6d %-6s %10s %8.1f %% %11s"
              % (width, grid,
                 " ".join("%9.1f" % v for v in values),
                 100.0 * (max(values) - min(values))
                 / abs(statistics.median(values)),
                 "YES" if stable else "NO"))
    print()

    print("=" * 74)
    print("4. PER-SHAPE BREAKDOWN UNDER THE SHIPPED TIGHT GRID")
    print("=" * 74)
    for width, one_pass, shipped in ARMS:
        print("width %d   %s (onePass67) against %s (Table.shipped)"
              % (width, one_pass, shipped))
        print("  %-34s %7s %6s %11s %11s %12s"
              % ("shape", "N", "calls", "onePass", "Table", "onePass wins"))
        total = 0.0
        order = sorted(
            pooled["dispatch"],
            key=lambda s: -abs(
                (pooled["us"][(s, one_pass, "tight")]
                 - pooled["us"][(s, shipped, "tight")])
                * pooled["dispatch"][s]
            ),
        )
        for name in order:
            calls = pooled["dispatch"][name]
            a = pooled["us"][(name, one_pass, "tight")] * calls
            b = pooled["us"][(name, shipped, "tight")] * calls
            total += b - a
            print("  %-34s %7d %6d %11.1f %11.1f %+12.1f"
                  % (name, pooled["n"][name], calls, a, b, b - a))
        print("  %-34s %7s %6s %11s %11s %+12.1f"
              % ("TOTAL", "", "", "", "", total))
        print()

    # The pooled session spread mixes drift between sessions into a comparison
    # that is actually paired inside each session, so it overstates the noise
    # on this contrast. Sign agreement across replicates is the honest test.
    print("=" * 74)
    print("5. PER-REPLICATE SIGN CHECK ON THE TWO NAMED SHAPES, TIGHT GRID")
    print("=" * 74)
    print("Each value is us/round that Table.shipped costs over onePass67.")
    print("%-34s %6s %5s %9s %9s %9s %8s"
          % ("shape", "plan", "width", "rep1", "rep2", "rep3", "agree"))
    singles = [gc.load([p]) for p in paths]
    for name in NAMED_N_AXIS + ("mlp.down",):
        for width, one_pass, shipped in ARMS:
            values = [
                (s["us"][(name, shipped, "tight")]
                 - s["us"][(name, one_pass, "tight")]) * s["dispatch"][name]
                for s in singles
            ]
            print("%-34s %6s %5d %s %8s"
                  % (name, shipped, width,
                     " ".join("%+9.1f" % v for v in values),
                     "YES" if min(values) * max(values) > 0 else "NO"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
