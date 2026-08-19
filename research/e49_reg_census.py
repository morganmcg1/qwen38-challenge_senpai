#!/usr/bin/env python3
"""E49 step 1: what each timed arm does to the register allocation. No GPU.

This does NOT measure the tax; it calibrates the DOSE. The advisor's rule holds:
there is no true register readout on this box, `peak_live_regs` is a textual
peak-live-SSA heuristic, and its absolute level is not comparable to hardware.
What it is good for is telling us whether an arm moved the shared allocation at
all, and in which direction -- i.e. whether the knob we are about to turn is
connected to anything.

Two readouts per arm, exactly as E46:

  cell            one width case `..._m<T, M, IPG, true>` compiled alone.
                  The max over the cases a table dispatches is the campaign's
                  "kernel-wide max" (108 at the tip, 129 on the NA<=5 table).
  entry           the real `affine_qmv_fast`, into which every case, the 2-bit
                  draft readout and `qmv_fast_impl` all inline. Over-counts by
                  the same amount in every arm, so a MOVE is evidence.

The working tree is never modified: each arm is a patched copy of quantized.h in
a shadow include directory searched ahead of the vendored one.

  python3 research/e49_reg_census.py --out research/e49-reg-census.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e46_reg_census import (  # noqa: E402
    CELL,
    ENTRIES,
    HEADER,
    HEADER_REL,
    compile_probe,
    ipg_table,
    na_cells,
    streams,
)
from e49_arms import ARMS, apply_arm  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
WIDTHS = list(range(3, 10))
SHIPPED_MAX = 108

# Cells offered as ceiling doses, censused standalone so the ladder is chosen
# from measured pressure rather than from arithmetic. `M % IPG != 1` and
# `M >= 3`; NA and M asserts are relaxed in the shadow header only.
DOSE_CANDIDATES = [(4, 4), (7, 4), (5, 5), (8, 5), (9, 5), (10, 5), (12, 6),
                   (14, 7), (16, 8)]


def census_arm(name: str, workdir: pathlib.Path) -> dict:
    shadow = workdir / name
    dst = shadow / HEADER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = apply_arm(HEADER.read_text(), name)
    dst.write_text(text)

    table = ipg_table(text)
    dispatched = {m: ipg for m, ipg in table.items() if ipg}
    out = {"name": name, "family": ARMS[name]["family"], "doc": ARMS[name]["doc"],
           "ipg_table": table, "streams": streams(table),
           "dispatched_widths": sorted(dispatched),
           "na_cells": {m: na_cells(m, table[m]) for m in WIDTHS}}

    cells = {}
    for m, ipg in sorted(dispatched.items()):
        res = compile_probe(shadow, "cell_m%d_ipg%d" % (m, ipg),
                            {"E46_CELL_M": m, "E46_CELL_IPG": ipg}, (CELL,))
        if res["status"] != "ok":
            return dict(out, status="cell_%d_%s" % (m, res["status"]),
                        error=res.get("error"))
        cells[m] = dict(res["functions"][CELL], ipg=ipg, na_cells=na_cells(m, ipg))

    entry = compile_probe(shadow, "entry", {}, ENTRIES)
    if entry["status"] != "ok":
        return dict(out, status="entry_%s" % entry["status"],
                    error=entry.get("error"))

    width_max = max(c["peak_live_regs"] for c in cells.values()) if cells else 0
    return dict(out, status="ok", width_cells=cells,
                entry={k: v["peak_live_regs"] for k, v in entry["functions"].items()},
                entry_allocas={k: v["allocas"] for k, v in entry["functions"].items()},
                entry_acc_allocas={k: v["acc_alloca_types"]
                                   for k, v in entry["functions"].items()},
                dispatched_reg_max=width_max,
                entry_point_reg_max=max(v["peak_live_regs"]
                                        for v in entry["functions"].values()))


def census_dose_cells(workdir: pathlib.Path) -> dict:
    """Standalone pressure of every candidate dose cell, on a relaxed header."""
    shadow = workdir / "_dose_cells"
    dst = shadow / HEADER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(apply_arm(HEADER.read_text(), "dose_129"))

    out = {}
    for m, ipg in DOSE_CANDIDATES:
        res = compile_probe(shadow, "dose_m%d_ipg%d" % (m, ipg),
                            {"E46_CELL_M": m, "E46_CELL_IPG": ipg}, (CELL,))
        key = "<T,%d,%d>" % (m, ipg)
        if res["status"] != "ok":
            out[key] = {"status": res["status"], "error": res.get("error")}
            continue
        fn = res["functions"][CELL]
        out[key] = {"status": "ok", "peak_live_regs": fn["peak_live_regs"],
                    "allocas": fn["allocas"],
                    "acc_alloca_types": fn["acc_alloca_types"],
                    "na_cells": na_cells(m, ipg),
                    "streams": (m + ipg - 1) // ipg,
                    "over_shipped_max": fn["peak_live_regs"] - SHIPPED_MAX}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e49-reg-census.json")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="e49-reg-census-"))
    try:
        doses = census_dose_cells(workdir)
        arms = [census_arm(name, workdir) for name in ARMS]
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    ok = [a for a in arms if a["status"] == "ok"]
    payload = {
        "head": head,
        "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
        "instrument_caveat":
            "peak_live_regs is a textual peak-live-SSA heuristic (ledger 157). "
            "Shape and direction only; the absolute number is not a hardware "
            "register count and no decision here rests on its level.",
        "shipped_max": SHIPPED_MAX,
        "dose_candidates": doses,
        "arms": arms,
        "all_ok": len(ok) == len(arms),
        "dispatched_reg_max": {a["name"]: a["dispatched_reg_max"] for a in ok},
        "entry_point_reg_max": {a["name"]: a["entry_point_reg_max"] for a in ok},
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("E49 register census   head=%s" % head[:8])
    print("\ndose candidates (standalone cell pressure, heuristic):")
    print("  %-12s %-6s %-8s %-10s %s" % ("cell", "regs", "streams", "NA cells", "status"))
    for key, v in doses.items():
        if v["status"] != "ok":
            print("  %-12s %s %s" % (key, v["status"], v.get("error", "")))
            continue
        print("  %-12s %-6d %-8d %-10s %s"
              % (key, v["peak_live_regs"], v["streams"], v["na_cells"],
                 "+%d over shipped max" % v["over_shipped_max"]
                 if v["over_shipped_max"] > 0 else "at/below shipped max"))

    print("\narms:")
    print("  %-14s %-10s %-8s %-8s %s"
          % ("arm", "family", "cellmax", "entry", "dispatched widths"))
    for a in arms:
        if a["status"] != "ok":
            print("  %-14s %-10s FAILED %s %s"
                  % (a["name"], a["family"], a["status"], a.get("error", "")))
            continue
        print("  %-14s %-10s %-8d %-8d %s"
              % (a["name"], a["family"], a["dispatched_reg_max"],
                 a["entry_point_reg_max"], a["dispatched_widths"]))
    print("\nwrote %s" % args.out)
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
