#!/usr/bin/env python3
"""Grade the E110 row-split arms against the zero-GPU decision rule.

The screen asks one question: does processing the four rows of a k-block as two
sequential half-passes buy enough register relief to be worth a GPU hour? The
rule is written on `applegpu_g16s` at NA = 5, so this script reports that cell
first, then prices every cell under the E77 occupancy law because NA = 5 is only
3.4 per cent of realised rounds.

    python3 research/e110_arms.py --emit /tmp/e110-xr
    python3 research/e110_arms.py --census /tmp/e110-xr \
        --out research/out/e110/xr-split-census.json
    python3 research/e110_xr_split_grade.py

Research-only: nothing here is on the scored path.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agx_crossarch import LOCAL_ARCH, RANKED_ARCH  # noqa: E402

CENSUS = pathlib.Path("research/out/e110/xr-split-census.json")
OUT = pathlib.Path("research/out/e110/xr-split-grade.json")

WIDTHS = ("2", "3", "4", "5")

# Realised width histogram of the fixture, from `rung1-summary.json`. NA = 4
# carries two thirds of the rounds; NA = 5, which the decision rule is written
# on, carries 3.4 per cent.
ROUND_WEIGHTS = {"2": 0.024, "3": 0.275, "4": 0.667, "5": 0.034}

# E77's corrected occupancy law. Occupancy is a smooth and very weak function of
# the register count, so a large simdgroup swing buys a small time change.
GAMMA = 0.01346
FILE_KIB = {LOCAL_ARCH: 384, RANKED_ARCH: 496}

# `a_scaffold` rebuilds the pre-xv4 body the decision rule was written against.
# `a_base` is the current worktree kernel, which now ships xv4.
RULE_BASE = "a_scaffold"
LIVE_BASE = "a_base"
ARMS = ("a_scaffold", "a_base", "xr_split2", "xr_split2u")

RULE_ARCH = LOCAL_ARCH
RULE_WIDTH = "5"
RULE_DROP = 8


def simdgroups(arch: str, registers: int) -> int:
    return (FILE_KIB[arch] * 1024) // (128 * registers)


def omega(count: int) -> float:
    return (32.0 / count) ** GAMMA


def cell(rows: dict, arm: str, arch: str, width: str) -> dict:
    value = rows[arm][arch][width]
    registers = value["registers"]
    return {
        "registers": registers,
        "spill_bytes": value.get("spill_bytes") or 0,
        "text_bytes": value["text_bytes"],
        "simdgroups": simdgroups(arch, registers),
    }


def main() -> int:
    rows = json.loads(CENSUS.read_text())["arms"]
    report: dict = {"round_weights": ROUND_WEIGHTS, "gamma": GAMMA, "arch": {}}

    for arch in (LOCAL_ARCH, RANKED_ARCH):
        label = "S_ranked" if arch == RANKED_ARCH else "S_local"
        print("\n%s   registers / spill / text / %s" % (arch, label))
        print("  %-11s %s" % ("arm", "  ".join(
            "%-22s" % ("NA" + w) for w in WIDTHS)))
        per_arm = {}
        for arm in ARMS:
            cells = {w: cell(rows, arm, arch, w) for w in WIDTHS}
            per_arm[arm] = cells
            print("  %-11s %s" % (arm, "  ".join(
                "%-22s" % ("%d%s / %d / %d" % (
                    cells[w]["registers"],
                    "s%d" % cells[w]["spill_bytes"]
                    if cells[w]["spill_bytes"] else "",
                    cells[w]["text_bytes"], cells[w]["simdgroups"]))
                for w in WIDTHS)))
        report["arch"][arch] = per_arm

        # Occupancy price against both baselines, weighted by realised width.
        for base in (RULE_BASE, LIVE_BASE):
            for arm in ("xr_split2", "xr_split2u"):
                ratio = sum(
                    ROUND_WEIGHTS[w]
                    * omega(per_arm[arm][w]["simdgroups"])
                    / omega(per_arm[base][w]["simdgroups"])
                    for w in WIDTHS)
                print("  occupancy-only time ratio %-10s vs %-10s %.5f "
                      "(%+.3f %%)" % (arm, base, ratio, 100.0 * (ratio - 1.0)))
                report.setdefault("occupancy_ratio", {})[
                    "%s|%s|%s" % (arch, arm, base)] = ratio

    print("\nDecision rule: %s NA=%s registers must fall by >= %d from %s "
          "at zero spill" % (RULE_ARCH, RULE_WIDTH, RULE_DROP, RULE_BASE))
    base_cell = cell(rows, RULE_BASE, RULE_ARCH, RULE_WIDTH)
    verdicts = {}
    for arm in ("xr_split2", "xr_split2u"):
        arm_cell = cell(rows, arm, RULE_ARCH, RULE_WIDTH)
        drop = base_cell["registers"] - arm_cell["registers"]
        passed = drop >= RULE_DROP and arm_cell["spill_bytes"] == 0
        verdicts[arm] = {"drop": drop, "spill_bytes": arm_cell["spill_bytes"],
                         "passed": passed}
        print("  %-11s %d -> %d (drop %+d), spill %d B  =>  %s"
              % (arm, base_cell["registers"], arm_cell["registers"], drop,
                 arm_cell["spill_bytes"], "PASS" if passed else "FAIL"))
    report["rule"] = {"arch": RULE_ARCH, "width": RULE_WIDTH,
                      "base": RULE_BASE, "required_drop": RULE_DROP,
                      "base_registers": base_cell["registers"],
                      "verdicts": verdicts}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
