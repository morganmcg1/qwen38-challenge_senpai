#!/usr/bin/env python3
"""E108: decide the pre-registered stop rule from the probe output.

    python3 research/e108_analysis.py research/out/e108-rung0/probe.json \
        --census research/out/e108/instrument-validation.json

Every arm executes the same instructions at every measured width, so a per-cell
time delta can only be instruction fetch or code layout. The pre-registered rule
is: report a negative if `i_pruneall` moves no scored cell by more than 1.0 % and
the pooled effect is inside the plus or minus 0.2 % E104 instrument floor.

The paired estimate per block is already the mean of the two palindrome slots the
arm occupies, so the delta is formed block by block and then reduced by median.
That keeps a single drifting block from moving the verdict.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

CELL_LIMIT_PCT = 1.0
POOLED_FLOOR_PCT = 0.2
ARCHES = ("applegpu_g16s", "applegpu_g17s")


def deltas(doc: dict, base: str, warmup: int) -> dict:
    """(shape, M) -> arm -> list of per-block percentage deltas against base."""
    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < warmup:
            continue
        sec = row["seconds"]
        cell = cells.setdefault((row["shape"], row["m"]), {})
        for arm, value in sec.items():
            if arm == base:
                continue
            cell.setdefault(arm, []).append(100.0 * (value - sec[base])
                                            / sec[base])
    return cells


def base_times(doc: dict, base: str, warmup: int) -> dict:
    out: dict[tuple[str, int], list[float]] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < warmup:
            continue
        out.setdefault((row["shape"], row["m"]), []).append(
            row["seconds"][base])
    return out


def exactness(doc: dict) -> tuple[dict, list[str], list[str]]:
    """arm -> (shape, M) -> differing count, plus failures and controls."""
    matrix: dict[str, dict[tuple[str, int], int]] = {}
    failures, controls = [], []
    for row in doc["measurements"]:
        if row["kind"] == "fidelity":
            for arm in row["arms"]:
                matrix.setdefault(arm["arm"], {})[(row["shape"], row["m"])] = \
                    arm["differing"]
                if arm["exact_required"] and not arm["bit_identical"]:
                    failures.append(
                        f"{row['shape']} M={row['m']} {arm['arm']}: "
                        f"{arm['differing']}/{arm['total']} differ")
        elif row["kind"] == "positive_control":
            controls.append(
                f"{row['shape']} M={row['m']} {row['arm']}: "
                f"{row['differing']}/{row['total']} differ, "
                f"detected={row['detected']}")
    return matrix, failures, controls


def census_text(path: pathlib.Path | None) -> dict:
    if not path:
        return {}
    doc = json.loads(path.read_text())
    return {name: {arch: (rec[arch]["text_bytes"], rec[arch]["spill_bytes"],
                          rec[arch]["registers"])
                   for arch in ARCHES if arch in rec}
            for name, rec in doc["arms"].items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--base", default="a_base")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    doc = json.loads(args.probe.read_text())
    arms = [a for a in doc["arms"] if a != args.base]
    cells = deltas(doc, args.base, args.warmup)
    base_us = base_times(doc, args.base, args.warmup)
    text = census_text(args.census)

    print(f"device={doc['device']} arch={doc['architecture']} "
          f"function={doc['function']}")
    print(f"base arm={args.base}  arms={doc['arms']}  "
          f"blocks kept per cell={len(next(iter(cells.values()))[arms[0]]) if cells else 0}")

    if text:
        print("\n== static census beside the timed arms ==")
        print(f"{'arm':<16}" + "".join(
            f"{a.replace('applegpu_', '') + 'B':>12}"
            f"{a.replace('applegpu_', '') + 'reg':>10}"
            f"{a.replace('applegpu_', '') + 'spl':>10}" for a in ARCHES))
        for name in [args.base] + arms:
            if name not in text:
                continue
            row = f"{name:<16}"
            for arch in ARCHES:
                b, spill, reg = text[name][arch]
                row += f"{b:>12}{reg:>10}{spill:>10}"
            print(row)

    matrix, failures, controls = exactness(doc)
    if matrix:
        print("\n== bit-exactness against the base arm, differing elements ==")
        keys = sorted({k for arm in matrix.values() for k in arm})
        print(f"{'shape':<38}{'M':>3}" + "".join(f"{a:>16}" for a in arms))
        for key in keys:
            row = f"{key[0]:<38}{key[1]:>3}"
            for arm in arms:
                row += f"{matrix.get(arm, {}).get(key, '-'):>16}"
            print(row)
    for line in failures:
        print(f"EXACTNESS FAILURE: {line}")
    for line in controls:
        print(f"harness positive control: {line}")

    print("\n== per-cell paired deltas against the base arm, percent ==")
    header = f"{'shape':<38}{'M':>3}{'base_us':>10}"
    for arm in arms:
        header += f"{arm + '_med':>16}{arm + '_min':>10}{arm + '_max':>10}"
    print(header)
    pooled: dict[str, list[float]] = {a: [] for a in arms}
    per_width: dict[int, dict[str, list[float]]] = {}
    breaches: dict[str, list[str]] = {a: [] for a in arms}
    for key in sorted(cells):
        shape, m = key
        row = (f"{shape:<38}{m:>3}"
               f"{1e6 * statistics.median(base_us[key]):>10.1f}")
        for arm in arms:
            values = cells[key][arm]
            med = statistics.median(values)
            row += f"{med:>16.3f}{min(values):>10.3f}{max(values):>10.3f}"
            pooled[arm].append(med)
            per_width.setdefault(m, {}).setdefault(arm, []).append(med)
            if abs(med) > CELL_LIMIT_PCT:
                breaches[arm].append(f"{shape} M={m} {med:+.3f} %")
        print(row)

    print("\n== pooled over cells, percent ==")
    for arm in arms:
        values = sorted(pooled[arm])
        n = len(values)
        positive = sum(1 for v in values if v > 0)
        print(f"{arm:<16} median={statistics.median(values):+.3f} "
              f"mean={statistics.fmean(values):+.3f} "
              f"q1={values[n // 4]:+.3f} q3={values[(3 * n) // 4]:+.3f} "
              f"min={values[0]:+.3f} max={values[-1]:+.3f} "
              f"cells={n} positive={positive}")

    print("\n== pooled by width, percent ==")
    print(f"{'M':>3}" + "".join(f"{a:>16}" for a in arms))
    for m in sorted(per_width):
        print(f"{m:>3}" + "".join(
            f"{statistics.median(per_width[m][a]):>16.3f}" for a in arms))

    # A `_null` arm compiles to the same machine code as the arm it mirrors, so
    # its delta is the harness's slot artefact at that palindrome position and
    # nothing else. A real arm effect has to clear it.
    nulls = {a: a[:-len("_null")] for a in arms if a.endswith("_null")}
    if nulls:
        print("\n== null controls: same machine code, different slot ==")
        for null, twin in nulls.items():
            if twin == args.base:
                values = pooled[null]
            elif twin in pooled:
                values = [n - t for n, t in zip(pooled[null], pooled[twin])]
            else:
                continue
            print(f"{null:<16} mirrors {twin:<16} "
                  f"median={statistics.median(values):+.3f} "
                  f"min={min(values):+.3f} max={max(values):+.3f} "
                  f"max_abs={max(abs(v) for v in values):.3f} % over "
                  f"{len(values)} cells")

    print("\n== pre-registered stop rule ==")
    verdict = {}
    for arm in arms:
        med = statistics.median(pooled[arm])
        inside_floor = abs(med) <= POOLED_FLOOR_PCT
        no_cell = not breaches[arm]
        stop = inside_floor and no_cell
        verdict[arm] = {"pooled_median_pct": med,
                        "cells_over_1pct": breaches[arm],
                        "negative_by_rule": stop}
        print(f"{arm:<16} pooled median {med:+.3f} % "
              f"({'inside' if inside_floor else 'OUTSIDE'} the "
              f"{POOLED_FLOOR_PCT} % floor), "
              f"{len(breaches[arm])} cells over {CELL_LIMIT_PCT} % "
              f"-> {'NEGATIVE, stop' if stop else 'MOVES, go to rung 1'}")
        for line in breaches[arm]:
            print(f"    over limit: {line}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "base_arm": args.base,
            "warmup_blocks": args.warmup,
            "cells": {f"{s}|{m}": {a: cells[(s, m)][a] for a in arms}
                      for s, m in cells},
            "pooled": {a: pooled[a] for a in arms},
            "verdict": verdict,
            "exactness_failures": failures,
            "harness_positive_controls": controls,
        }, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
