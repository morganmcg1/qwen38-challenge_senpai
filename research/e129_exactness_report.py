#!/usr/bin/env python3
"""E129 — read the three width-exactness legs and decide pass or fail.

    usage: research/e129_exactness_report.py OUTDIR

A leg passes only when every row is bit exact AND every row's positive control
fired. A row that is bit exact with a dead control proves nothing: it is the
same reading a kernel that never ran would give.
"""

from __future__ import annotations

import json
import pathlib
import sys

LEGS = (("base", "shared switch, shipped, wide grid"),
        ("tier", "tiered switch, shipped, wide grid"),
        ("onep", "tiered switch, one-pass, wide grid"),
        ("basetight", "shared switch, shipped, tight grid"),
        ("tiertight", "tiered switch, shipped, tight grid"),
        ("oneptight", "tiered switch, one-pass, tight grid"))

# A tight-grid leg must reproduce its wide-grid leg element for element. That
# is a stronger statement than "both are bit exact against quantizedMM",
# because it also covers the rows where the one-pass table is already wrong:
# dropping the empty threadgroups must not change even a wrong answer.
GRID_PAIRS = (("base", "basetight"), ("tier", "tiertight"), ("onep", "oneptight"))


def read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())["records"]


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    failures: list[str] = []
    summary = {}

    for leg, description in LEGS:
        records = read(root / ("%s.json" % leg))
        if not records:
            failures.append("%s: no records" % leg)
            continue
        exact = [r for r in records if r["bit_exact"]]
        live = [r for r in records if r["positive_control_can_fail"]]
        tabled = [r for r in records if "table_hit" in r]
        widths = sorted({r["width"] for r in records})
        summary[leg] = {
            "description": description,
            "rows": len(records),
            "bit_exact": len(exact),
            "controls_live": len(live),
            "widths": widths,
            "shapes": sorted({r["shape"] for r in records}),
            "arms": sorted({r["arm"] for r in records}),
            "max_abs_diff": max(r["max_abs_diff"] for r in records),
            "table_rows": len(tabled),
            "table_controls_live": sum(
                1 for r in tabled if r["table_hit"] > 0 and r["restored_diff"] == 0),
        }
        if len(exact) != len(records):
            bad = [(r["shape"], r["width"], r["arm"], r["differing_elements"])
                   for r in records if not r["bit_exact"]]
            failures.append("%s: %d of %d rows not bit exact %s"
                            % (leg, len(records) - len(exact), len(records), bad))
        if len(live) != len(records):
            dead = [(r["shape"], r["width"], r["arm"])
                    for r in records if not r["positive_control_can_fail"]]
            failures.append("%s: %d dead positive controls %s"
                            % (leg, len(records) - len(live), dead))
        if tabled and summary[leg]["table_controls_live"] != len(tabled):
            failures.append("%s: a chunk-sum table control did not fire" % leg)

    grid_pairs = {}
    for wide, tight in GRID_PAIRS:
        left, right = read(root / ("%s.json" % wide)), read(root / ("%s.json" % tight))
        if not left or not right:
            continue
        key = lambda r: (r["shape"], r["width"], r["arm"])  # noqa: E731
        by_key = {key(r): r for r in left}
        moved = [key(r) for r in right
                 if key(r) in by_key
                 and (r["differing_elements"] != by_key[key(r)]["differing_elements"]
                      or r["max_abs_diff"] != by_key[key(r)]["max_abs_diff"])]
        grid_pairs["%s->%s" % (wide, tight)] = {
            "rows_compared": len(by_key), "rows_moved": len(moved), "moved": moved}
        if moved:
            failures.append("%s vs %s: the tight grid changed %d rows %s"
                            % (wide, tight, len(moved), moved[:6]))

    print("%-10s %-38s %5s %10s %9s %s" % (
        "leg", "configuration", "rows", "bit exact", "controls", "widths"))
    for leg, _ in LEGS:
        row = summary.get(leg)
        if row is None:
            print("%-10s %-38s  MISSING" % (leg, "-"))
            continue
        print("%-10s %-38s %5d %10d %9d %s" % (
            leg, row["description"], row["rows"], row["bit_exact"],
            row["controls_live"],
            "".join(str(w) for w in row["widths"])))

    if grid_pairs:
        print("\ndropping the empty x threadgroups, element for element:")
        for name, pair in grid_pairs.items():
            print("  %-22s %d of %d rows moved" % (
                name, pair["rows_moved"], pair["rows_compared"]))

    (root / "summary.json").write_text(json.dumps(
        {"legs": summary, "grid_pairs": grid_pairs, "failures": failures,
         "pass": not failures}, indent=2) + "\n")

    if failures:
        print("\nFAIL")
        for line in failures:
            print("  %s" % line)
        return 1
    print("\nPASS: every routed width is bit exact against quantizedMM under "
          "every table, and every positive control fired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
