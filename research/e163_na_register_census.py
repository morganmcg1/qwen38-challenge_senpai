#!/usr/bin/env python3
"""E163: register and occupancy cost of the minimum-NA width plans.

E159's census answered "does a WIDER accumulator fit". This one prices the
opposite direction: each arm replaces one wide group with more, narrower ones,
so the question is how many registers the narrower body holds and how many
simdgroups that leaves resident.

It reuses `research/e159_na_register_census.py` verbatim -- the same live
header read out of `Qwen35.swift`, the same entry template, the same
`agx_crossarch` backend -- and only changes the cells. Every cell here is a
`(M, IPG)` pair that one of this experiment's arms actually instantiates,
including the tail bodies, so the numbers describe the kernels that run and not
a nearby template.

Zero GPU seconds. `xcrun metal-tt` runs the AGX backend for a named
architecture on any Mac.

  python3 research/e163_na_register_census.py
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e159_na_register_census as e159  # noqa: E402

OUT = HERE / "e163-artifacts" / "e163_na_register_census.json"

SHIPPED = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
ARMS = {
    "shipped": {},
    "minna5": {5: 3},
    "minna456": {4: 2, 5: 3, 6: 2},
}

# (M, IPG, role). `shipped` is the control ladder at the touched widths;
# `minna` cells are what the arms compile instead. Widths 7, 8 and 9 are here
# as untouched controls: their text digest must be identical across arms.
CELLS = [
    (4, 4, "shipped_w4"),
    (4, 2, "minna_w4"),
    (5, 5, "shipped_w5"),
    (5, 3, "minna_w5"),
    (6, 3, "shipped_w6"),
    (6, 2, "minna_w6"),
    (7, 4, "untouched_w7"),
    (8, 4, "untouched_w8"),
    (9, 3, "untouched_w9"),
]


def main() -> None:
    e159.CELLS = CELLS
    header = e159.live_header()

    result = {
        "experiment": "e163-r1-minimum-accumulator-width-plans",
        "harness": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "source": "Qwen35.swift qwen35E120QMVHeader, read live",
        "template_signature": "template <int NA, bool USE_TABLE> qwen_e120_qmv_wide",
        "arms": ARMS,
        "shipped_table": SHIPPED,
        "cells": {},
    }

    for table in (True, False):
        try:
            records = e159.census(header, table)
        except Exception as exc:  # noqa: BLE001
            result["cells"][f"use_table_{table}"] = {"error": str(exc)[:2000]}
            continue
        block = {}
        for m, ipg, role in CELLS:
            name = e159.cell_name(m, ipg, table)
            entry = records.get(name, {})
            record = {
                "M": m,
                "IPG": ipg,
                "NA": ipg,
                "role": role,
                "active_groups": -(-m // ipg),
                "tail_lanes": ipg if m % ipg == 0 else max(m % ipg, 2),
                "shipped_ipg_for_this_m": SHIPPED[m],
            }
            for arch, rec in entry.items():
                record[arch] = {
                    "registers": rec.get("registers"),
                    "spill_bytes": rec.get("spill_bytes"),
                    "text_bytes": rec.get("text_bytes"),
                    "text_sha8": rec.get("text_sha8"),
                }
            block[f"m{m}_ipg{ipg}"] = record
        result["cells"][f"use_table_{table}"] = block

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    for table_key, block in result["cells"].items():
        print(f"\n=== {table_key} ===")
        if "error" in block:
            print("  ERROR:", block["error"][:400])
            continue
        print(f"  {'cell':12s} {'role':14s} {'G x NA':>8s}"
              f" {'g16s reg/spill':>15s} {'g17s reg/spill':>15s}  text_sha8")
        for name, rec in block.items():
            g16 = rec.get("g16s", {})
            g17 = rec.get("g17s", {})
            print(
                f"  {name:12s} {rec['role']:14s}"
                f" {str(rec['active_groups']) + 'x' + str(rec['NA']):>8s}"
                f" {str(g16.get('registers')) + '/' + str(g16.get('spill_bytes')):>15s}"
                f" {str(g17.get('registers')) + '/' + str(g17.get('spill_bytes')):>15s}"
                f"  {g17.get('text_sha8')}"
            )
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
