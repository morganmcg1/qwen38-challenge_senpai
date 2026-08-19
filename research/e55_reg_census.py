#!/usr/bin/env python3
"""E55: does `<T,9,5>` raise the register allocation the whole kernel shares?

The advisor's warning: E27 moved `case 5` and `case 9` off NA<=4 together, won
locally per width, and lost 0.3321 % on the board. Its recorded mechanism is a
kernel-wide register max of 129 against 108, shared by every width because there
is exactly one `[[kernel]]` and every helper is `METAL_FUNC` inline. E55 moves
`case 9` alone, so the same channel is open.

Three arms, each a header taken from a named git rev so the reading is a property
of a commit and not of whatever the worktree currently holds:

  base_na4_table    the base twins, NA<=4 everywhere. Recorded: 108 / 163.
  m9two_candidate   the candidate. THE MEASUREMENT.
  e27_both_cells    base + assert relaxation + case 5 AND case 9 at NA=5.

The third arm is the instrument's positive control, and it is calibrated against
a recorded historical value rather than against itself: E27's table must report
129. An arm that cannot reproduce a known move cannot be trusted to report a
null, per ledger 178(E).

It also answers the question the advisor's table cannot: if the candidate and the
E27 table report the SAME max, then the candidate carries E27's full allocation
penalty and `case 5` contributed none of it.

No GPU work is dispatched. Compiling AIR and creating a pipeline both run on the
driver.

  python3 research/e55_reg_census.py --out research/e55-reg-census.json
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
    HEADER_REL,
    WIDTHS,
    build_occupancy_tool,
    compile_probe,
    ipg_table,
    na_cells,
    occupancy,
    streams,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER_PATH = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"

BASE_REV = "f2ec48a"
CAND_REV = "2267a84"

ASSERT_4 = 'static_assert(NA >= 2 && NA <= 4, "wide multi-row QMV supports NA in [2, 4]");'
ASSERT_5 = 'static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");'
M5_SHIPPED = "qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>"
M5_E27 = "qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>"
M9_SHIPPED = "qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>"
M9_NA5 = "qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>"

# The harm-only control. It adds an NA=5 body to the ONE `[[kernel]]` without
# changing any cell the host can dispatch: the offered draft depth caps M at 9,
# so `case 10` is reachable code that never runs. If the shared-allocation
# channel is what E27 lost the board to, this arm carries that cost with zero
# M=9 gain, so a timed arm measures the harm term directly instead of inferring
# it from a model residual.
CASE9_TAIL = """        case 9:
          qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        default:"""
CASE9_TAIL_PLUS_10 = """        case 9:
          qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 10:
          qmv_fast_crossrow_affine4_g64_m<T, 10, 5, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        default:"""

# The campaign's recorded values for the two arms whose numbers are on the record.
RECORDED = {
    "base_na4_table": {"kernel_wide_reg_max": 108, "entry_batch0": 163},
    "e27_both_cells": {"kernel_wide_reg_max": 129, "entry_batch0": 183},
}

ARMS = [
    {"name": "base_na4_table", "rev": BASE_REV, "edits": [],
     "role": "reference"},
    {"name": "m9two_candidate", "rev": CAND_REV, "edits": [],
     "role": "measurement"},
    {"name": "e27_both_cells", "rev": BASE_REV,
     "edits": [(ASSERT_4, ASSERT_5), (M5_SHIPPED, M5_E27), (M9_SHIPPED, M9_NA5)],
     "role": "positive_control"},
    {"name": "harm_only_case10", "rev": BASE_REV,
     "edits": [(ASSERT_4, ASSERT_5), (CASE9_TAIL, CASE9_TAIL_PLUS_10)],
     "role": "harm_only_design"},
]


def header_at(rev: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", "%s:%s" % (rev, HEADER_PATH)],
        capture_output=True, text=True, check=True)
    return out.stdout


def run_arm(arm: dict, workdir: pathlib.Path,
            tool: pathlib.Path | None) -> dict:
    shadow = workdir / arm["name"]
    dst = shadow / HEADER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)

    text = header_at(arm["rev"])
    for old, new in arm["edits"]:
        if text.count(old) != 1:
            return dict(arm, status="edit_not_unique", pattern=old,
                        occurrences=text.count(old))
        text = text.replace(old, new)
    dst.write_text(text)

    table = ipg_table(text)
    out = dict(arm, ipg_table=table, streams=streams(table),
               na_cells={m: na_cells(m, table[m]) for m in WIDTHS},
               assert_bound=5 if ASSERT_5 in text else 4)

    cells = {}
    for m in WIDTHS:
        res = compile_probe(shadow, "cell_m%d" % m,
                            {"E46_CELL_M": m, "E46_CELL_IPG": table[m]}, (CELL,))
        if res["status"] != "ok":
            return dict(out, status="cell_%d_%s" % (m, res["status"]),
                        error=res.get("error"))
        cells[m] = dict(res["functions"][CELL], ipg=table[m],
                        na_cells=na_cells(m, table[m]))

    entry = compile_probe(shadow, "entry", {}, ENTRIES)
    if entry["status"] != "ok":
        return dict(out, status="entry_%s" % entry["status"],
                    error=entry.get("error"))

    width_max = max(c["peak_live_regs"] for c in cells.values())
    out = dict(
        out, status="ok", width_cells=cells,
        entry={k: v["peak_live_regs"] for k, v in entry["functions"].items()},
        kernel_wide_reg_max=width_max,
        argmax_width=max(cells, key=lambda m: cells[m]["peak_live_regs"]),
        entry_batch0=entry["functions"][ENTRIES[0]]["peak_live_regs"],
        entry_batch1=entry["functions"][ENTRIES[1]]["peak_live_regs"],
        acc_alloca_types_by_width={
            m: cells[m]["acc_alloca_types"] for m in WIDTHS},
    )
    if tool is not None:
        out["occupancy"] = occupancy(shadow, tool)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e55-reg-census.json")
    ap.add_argument("--no-occupancy", action="store_true")
    args = ap.parse_args()

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="e55-reg-census-"))
    try:
        tool = None if args.no_occupancy else build_occupancy_tool(workdir)
        arms = [run_arm(a, workdir, tool) for a in ARMS]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    by = {a["name"]: a for a in arms}
    ok = [a for a in arms if a["status"] == "ok"]

    checks = {}
    for name, want in RECORDED.items():
        a = by.get(name)
        if a is None or a["status"] != "ok":
            checks[name] = "arm_failed"
            continue
        checks[name] = {
            "kernel_wide_reg_max": [a["kernel_wide_reg_max"],
                                    want["kernel_wide_reg_max"],
                                    a["kernel_wide_reg_max"] == want["kernel_wide_reg_max"]],
            "entry_batch0": [a["entry_batch0"], want["entry_batch0"],
                             a["entry_batch0"] == want["entry_batch0"]],
        }
    control_fired = (
        by["e27_both_cells"].get("kernel_wide_reg_max")
        != by["base_na4_table"].get("kernel_wide_reg_max")
        if len(ok) == len(ARMS) else None)

    payload = {
        "head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
        "base_rev": BASE_REV,
        "candidate_rev": CAND_REV,
        "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
        "arms": arms,
        "all_ok": len(ok) == len(ARMS),
        "kernel_wide_reg_max": {a["name"]: a["kernel_wide_reg_max"] for a in ok},
        "entry_batch0": {a["name"]: a["entry_batch0"] for a in ok},
        "entry_batch1": {a["name"]: a["entry_batch1"] for a in ok},
        "recorded_value_checks": checks,
        "positive_control_fired": control_fired,
    }
    if len(ok) == len(ARMS):
        b = by["base_na4_table"]
        c = by["m9two_candidate"]
        e = by["e27_both_cells"]
        payload["candidate_vs_base"] = {
            "kernel_wide_reg_max_delta": c["kernel_wide_reg_max"] - b["kernel_wide_reg_max"],
            "entry_batch0_delta": c["entry_batch0"] - b["entry_batch0"],
            "candidate_rises_above_base": c["kernel_wide_reg_max"] > b["kernel_wide_reg_max"],
        }
        payload["candidate_vs_e27"] = {
            "kernel_wide_reg_max_delta": c["kernel_wide_reg_max"] - e["kernel_wide_reg_max"],
            "entry_batch0_delta": c["entry_batch0"] - e["entry_batch0"],
            "candidate_carries_full_e27_allocation":
                c["kernel_wide_reg_max"] == e["kernel_wide_reg_max"],
        }
        payload["per_width_reg"] = {
            a["name"]: {m: a["width_cells"][m]["peak_live_regs"] for m in WIDTHS}
            for a in ok}
        payload["widths_unchanged_vs_base"] = sorted(
            m for m in WIDTHS
            if c["width_cells"][m]["peak_live_regs"]
            == b["width_cells"][m]["peak_live_regs"])
        h = by["harm_only_case10"]
        payload["harm_only_vs_base"] = {
            "entry_batch0_delta": h["entry_batch0"] - b["entry_batch0"],
            "kernel_wide_reg_max_delta":
                h["kernel_wide_reg_max"] - b["kernel_wide_reg_max"],
            "dispatched_widths_all_identical_to_base": all(
                h["width_cells"][m]["peak_live_regs"]
                == b["width_cells"][m]["peak_live_regs"] for m in WIDTHS),
            "carries_candidate_entry_share":
                (h["entry_batch0"] - b["entry_batch0"])
                / max(c["entry_batch0"] - b["entry_batch0"], 1),
        }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("E55 register census   head=%s   base=%s candidate=%s"
          % (payload["head"][:8], BASE_REV, CAND_REV))
    print("%-17s %-16s %-8s %-6s %-6s %-6s %s"
          % ("arm", "role", "status", "wide", "e_b0", "e_b1", "per-width M=3..9"))
    for a in arms:
        if a["status"] != "ok":
            print("%-17s %-16s %-8s" % (a["name"], a["role"], a["status"]),
                  a.get("error", ""), a.get("pattern", ""))
            continue
        per = " ".join("%d" % a["width_cells"][m]["peak_live_regs"] for m in WIDTHS)
        print("%-17s %-16s %-8s %-6d %-6d %-6d %s"
              % (a["name"], a["role"], a["status"], a["kernel_wide_reg_max"],
                 a["entry_batch0"], a["entry_batch1"], per))
    for a in ok:
        print("  %-17s assert<=%d  IPG(M=3..9) = %s  streams = %s"
              % (a["name"], a["assert_bound"],
                 " ".join(str(a["ipg_table"][m]) for m in WIDTHS),
                 " ".join(str(a["streams"][m]) for m in WIDTHS)))
        print("  %-17s acc alloca M=9 = %s"
              % ("", a["acc_alloca_types_by_width"][9]))

    print("\nrecorded-value checks (campaign record, not self-comparison):")
    for name, res in checks.items():
        print("  %-17s %s" % (name, res))
    print("positive control fired (E27 table != base): %s" % control_fired)
    if payload.get("harm_only_vs_base"):
        print("harm-only design: %s" % payload["harm_only_vs_base"])
    if payload.get("candidate_vs_base"):
        print("candidate vs base : %s" % payload["candidate_vs_base"])
        print("candidate vs E27  : %s" % payload["candidate_vs_e27"])
        print("widths with reg count unchanged vs base: %s"
              % payload["widths_unchanged_vs_base"])
    if "occupancy" in by["m9two_candidate"]:
        for a in ok:
            fns = a["occupancy"].get("functions", {})
            print("  occupancy %-17s %s" % (a["name"], fns))
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
