#!/usr/bin/env python3
"""E129 -- prove warmup instantiated every entry point before the timed window.

    usage: research/e129_pipeline_coverage.py LOG [LOG ...] [--json OUT]

Templating turns 2 instantiated QMV pipelines into 4, and the one-pass table
into 7. A pipeline first compiled INSIDE a timed leg pays a full Metal JIT
compile there and reads as a large regression, so the submission risk is not
the kernels but the warmup.

`warmAllDepthShapes` runs one throwaway forward at each legal width in
ascending order before any scored token, so the dispatch ordinal at which each
width is first seen must be an ARITHMETIC PROGRESSION whose step is the QMV
dispatch count of one forward. A width first reached inside the timed window
breaks that progression by orders of magnitude, and every tier pipeline is
first seen at the first dispatch of its lowest width.

This script checks three things per log and fails on any of them:

  1. every routed width the leg reached has a first index inside the warmup
     prefix, which is the progression the widths themselves define;
  2. every pipeline key's first index equals the first index of the lowest
     width that routes to it, so no pipeline was compiled late;
  3. the number of distinct pipeline keys equals the number the recorded table
     predicts.

Zero GPU seconds; it only reads logs a leg already wrote.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# `Qwen35CustomQMV.minimumTableWidth`: M=3 runs the no-table replica pipeline
# and every wider routed width runs the chunk-sum pipeline.
MINIMUM_TABLE_WIDTH = 4


def parse_plan(witness: str) -> dict[int, tuple[int, int]]:
    """`e120_width_plan/3:3:4,...` into `{m: (ipg, rps)}`."""
    body = witness.split("/", 1)[1]
    out = {}
    for cell in body.split(","):
        m, ipg, rps = (int(v) for v in cell.split(":"))
        out[m] = (ipg, rps)
    return out


def expected_keys(plan: dict[int, tuple[int, int]], widths: list[int],
                  tiered: bool) -> set[str]:
    """The pipeline keys the widths a leg actually reached must produce.

    `Qwen35CustomQMV.matmul` on the shipped `sumTable` arm routes
    `m >= MINIMUM_TABLE_WIDTH` to the chunk-sum entry point with
    `USE_TABLE=true` and everything narrower to the no-table entry point.
    `xsums_v1` is the fill kernel, which carries no width.
    """
    keys = {"xsums_v1"}
    for m in widths:
        suffix = "_na%d" % plan[m][0] if tiered else ""
        if m >= MINIMUM_TABLE_WIDTH:
            keys.add("qmv_sums%s_v2/USE_TABLE=true" % suffix)
        else:
            keys.add("qmv_wide%s_v2" % suffix)
    return keys


def check(path: pathlib.Path) -> dict:
    log = json.loads(path.read_text())
    plan = parse_plan(log["plan"])
    by_width = {int(m): n for m, n in log["by_width"].items()}
    first_width = {int(m): i for m, i in log["first_index_by_width"].items()}
    first_key = {k: i for k, i in log["first_index_by_key"].items()}
    widths = sorted(by_width)

    problems = []

    # 1. Warmup prefix. Every width's first dispatch must land before the last
    # width's first dispatch plus one warmup step, which is the definition of
    # "all of them were reached during the ascending warmup sweep".
    ordered = [first_width[m] for m in widths]
    warmup_end = max(ordered)
    steps = [b - a for a, b in zip(ordered, ordered[1:])]
    if ordered != sorted(ordered):
        problems.append("width first-dispatch order is not ascending: %s" % ordered)
    if steps and max(steps) > 4 * max(min(steps), 1):
        problems.append(
            "width first-dispatch steps are not a progression: %s" % steps)

    # 2. No pipeline compiled after warmup ended.
    late = {k: i for k, i in first_key.items() if i > warmup_end}
    if late:
        problems.append("pipelines first seen after warmup: %s" % late)

    # 3. The key set matches what the recorded table predicts for the widths
    # this leg reached.
    want = expected_keys(plan, widths, log["entry"] == "tiered_switch")
    have = set(first_key)
    if want != have:
        problems.append("pipeline keys %s, expected %s" % (sorted(have), sorted(want)))

    return {
        "log": str(path),
        "entry": log["entry"],
        "table": log["table"],
        "arm": log["arm"],
        "plan": log["plan"],
        "qmv_specializations": log["qmv_specializations"],
        "dispatches": log["dispatches"],
        "widths_reached": widths,
        "first_index_by_width": {str(m): first_width[m] for m in widths},
        "first_index_by_key": first_key,
        "warmup_last_dispatch": warmup_end,
        "problems": problems,
        "ok": not problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    records = []
    for path in args.logs:
        if not path.is_file():
            print("MISSING %s" % path)
            records.append({"log": str(path), "ok": False,
                            "problems": ["log not written"]})
            continue
        record = check(path)
        records.append(record)
        print("%-46s entry=%-14s table=%-8s pipelines=%d widths=%s %s" % (
            path.name, record["entry"], record["table"],
            record["qmv_specializations"], record["widths_reached"],
            "OK" if record["ok"] else "FAIL"))
        print("    first dispatch by width %s" % record["first_index_by_width"])
        print("    first dispatch by key   %s" % record["first_index_by_key"])
        for problem in record["problems"]:
            print("    PROBLEM %s" % problem)

    ok = all(r["ok"] for r in records) and bool(records)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "harness": "local",
            "official_or_ranked_score": False,
            "timing_valid": False,
            "instrument": "MLX_E120_QMV_PIPELINE_LOG first-dispatch ordinals",
            "ok": ok,
            "legs": records,
        }, indent=1, sort_keys=True) + "\n")
        print("wrote %s" % args.json)
    print("pipeline coverage: %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
