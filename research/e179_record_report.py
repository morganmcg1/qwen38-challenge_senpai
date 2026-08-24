#!/usr/bin/env python3
"""E179 step 1 report: price the launch-config cache from the host record probe.

Reads the four ABBA legs written by research/e179_record_probe.sh and reports,
per cell and over cells, the per-call record cost of each arm, their difference
and what that difference is worth over the 387 kernel records of one decode
round.

harness=local. Host time only, no thermal gate, no ranked claim.
"""

import json
import pathlib
import statistics
import sys

OUT = pathlib.Path(__file__).resolve().parent / "out"
LEGS = ["on1", "off1", "off2", "on2"]
RECORDS_PER_ROUND = 387
QMV_RECORDS = 257
FILL_RECORDS = 130
ROUND_MS = 181.0
MUE_MS = 0.54


def load(leg):
    path = OUT / f"e179-record-probe-{leg}.json"
    if not path.exists():
        sys.exit(f"missing leg file: {path}")
    return json.loads(path.read_text())


def cell_record_us(report, cell, batch=256):
    return report["cells"][cell][f"batch_{batch}"]["record_us_per_call_median"]


def main():
    legs = {leg: load(leg) for leg in LEGS}

    for leg, report in legs.items():
        arm_on = report["arm_resolved_enabled"]
        expected = leg.startswith("on")
        if arm_on != expected:
            sys.exit(f"leg {leg} resolved arm_enabled={arm_on}, expected {expected}")
        if report["batch_scaling_control_failures"]:
            sys.exit(f"leg {leg} failed the batch-scaling control")
        if expected and report["cache_hits"] <= 0:
            sys.exit(f"leg {leg} is the on arm but recorded no cache hit")
        if not expected and (report["cache_hits"] or report["cache_misses"]):
            sys.exit(f"leg {leg} is the off arm but touched the cache")

    cells = sorted(legs["on1"]["cells"])
    result = {
        "experiment": "E179-step1-config-cache-record-report",
        "harness": "local",
        "timed_leg": False,
        "gate_qualified_for_timing": False,
        "records_per_round": RECORDS_PER_ROUND,
        "round_ms": ROUND_MS,
        "minimum_useful_effect_ms_per_round": MUE_MS,
        "legs": LEGS,
        "cache_hits_on1": legs["on1"]["cache_hits"],
        "cache_misses_on1": legs["on1"]["cache_misses"],
        "cells": {},
    }

    deltas = []
    for cell in cells:
        on = [cell_record_us(legs[leg], cell) for leg in LEGS if leg.startswith("on")]
        off = [cell_record_us(legs[leg], cell) for leg in LEGS if leg.startswith("off")]
        delta = statistics.mean(off) - statistics.mean(on)
        deltas.append(delta)
        result["cells"][cell] = {
            "record_us_per_call_on": on,
            "record_us_per_call_off": off,
            "record_us_per_call_on_mean": statistics.mean(on),
            "record_us_per_call_off_mean": statistics.mean(off),
            "delta_us_per_call": delta,
            "delta_share_of_off_record": delta / statistics.mean(off),
        }

    # Weight the cells by what a decode round actually launches: 257 records
    # are table-paying QMV cells and 130 are standalone chunk-sum fills.
    qmv = statistics.mean(
        [result["cells"][c]["delta_us_per_call"] for c in cells if c.startswith("qmv_")]
    )
    fill = result["cells"]["xsums_hidden_5120"]["delta_us_per_call"]
    weighted_ms_per_round = (QMV_RECORDS * qmv + FILL_RECORDS * fill) / 1000.0
    result["delta_us_per_call_qmv"] = qmv
    result["delta_us_per_call_fill"] = fill
    result["weighted_ceiling_ms_per_round"] = weighted_ms_per_round
    result["weighted_ceiling_pct_of_round"] = 100.0 * weighted_ms_per_round / ROUND_MS
    result["weighted_ceiling_clears_mue"] = weighted_ms_per_round >= MUE_MS

    delta = statistics.mean(deltas)
    ms_per_round = delta * RECORDS_PER_ROUND / 1000.0
    result["delta_us_per_call_mean_over_cells"] = delta
    result["delta_us_per_call_min_over_cells"] = min(deltas)
    result["delta_us_per_call_max_over_cells"] = max(deltas)
    result["ceiling_ms_per_round"] = ms_per_round
    result["ceiling_pct_of_round"] = 100.0 * ms_per_round / ROUND_MS
    result["ceiling_clears_mue"] = ms_per_round >= MUE_MS
    result["ceiling_ms_per_round_min"] = min(deltas) * RECORDS_PER_ROUND / 1000.0
    result["ceiling_ms_per_round_max"] = max(deltas) * RECORDS_PER_ROUND / 1000.0

    (OUT / "e179-record-report.json").write_text(
        json.dumps(result, indent=1, sort_keys=True)
    )
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
