#!/usr/bin/env python3
"""E61: what does a single weight stream at M=6 cost in registers?

The advisor's census law, exact on six configurations:

    reg = 20 + 21*max(NA) + 4*[two distinct NA group sizes]

E61 predicts `<T,6,6>` at 146 and the whole-table maximum at 146, against the
129 the `d2139c92` table sits at. Predictions are registered in
`research/e61-prereg.md` BEFORE this script runs; the constants below are
copied from that file so the verdict is computed rather than chosen.

Every arm is a patched copy of quantized.h in a shadow include directory, so
the worktree is never modified and each reading is a property of a named rev
plus a named patch. No GPU work is dispatched: compiling AIR and creating a
pipeline both run on the driver.

    python3 research/e61_reg_census.py --out research/e61-artifacts/e61-reg-census.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
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
from e61_arms import apply_arm, relax_na  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER_PATH = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
BASE_REV = "d2139c924c7a7d98ca6026eea63867c2776abbca"

# --- pre-registered in research/e61-prereg.md, before any measurement --------
PREDICTED_CELL = {
    ("shipped", 9): 129,
    ("t6", 6): 146,
    ("iso_m6_ipg3", 6): 83,
    ("iso_m6_ipg6", 6): 146,
}
PREDICTED_TABLE_MAX = {
    "shipped": 129,
    "t6": 146,
    "iso_m6_ipg3": 83,
    "iso_m6_ipg6": 146,
}
PREDICTED_ENTRY_INTERVAL = {"shipped": (178, 184), "t6": (196, 205)}
# Isolated cells the census reads but never times, one compile each.
PREDICTED_EXTRA_CELLS = {(7, 7): 167, (8, 8): 188}

ARMS = [
    {"name": "shipped", "role": "reference", "never_time": False},
    {"name": "t6", "role": "measurement", "never_time": False},
    {"name": "t6_rbx", "role": "measurement", "never_time": False},
    {"name": "shipped_rbx", "role": "measurement", "never_time": False},
    {"name": "iso_m6_ipg3", "role": "census_probe", "never_time": True},
    {"name": "iso_m6_ipg6", "role": "census_probe", "never_time": True},
    {"name": "ballast", "role": "census_probe", "never_time": True},
]


DISPATCHED_CELL = re.compile(
    r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>")


def law(max_na: int, distinct_group_sizes: int) -> int:
    """The advisor's census law, as this instrument reads it."""
    return 20 + 21 * max_na + (4 if distinct_group_sizes > 1 else 0)


def instantiated_cells(text: str) -> dict[int, int]:
    """Every `<T,M,IPG>` the arm instantiates, including unreachable ones.

    `e46_reg_census.ipg_table` scans widths 3..9 only, so a ballast cell placed
    outside that range would be invisible to the table maximum it is there to
    raise. The register allocator sees every instantiated body, so the census
    must too.
    """
    return {int(m): int(ipg) for m, ipg in DISPATCHED_CELL.findall(text)}


def header_at(rev: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", "%s:%s" % (rev, HEADER_PATH)],
        capture_output=True, text=True, check=True)
    return out.stdout


def run_arm(arm: dict, base_text: str, workdir: pathlib.Path,
            tool: pathlib.Path | None) -> dict:
    shadow = workdir / arm["name"]
    dst = shadow / HEADER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = apply_arm(base_text, arm["name"])
    dst.write_text(text)

    table = ipg_table(text)
    live = instantiated_cells(text)
    out = dict(arm, family="census_probe" if arm["never_time"] else "timed_arm",
               ipg_table=table, streams=streams(live),
               na_cells={m: na_cells(m, ipg) for m, ipg in live.items()})

    cells = {}
    for m, ipg in live.items():
        res = compile_probe(shadow, "cell_m%d" % m,
                            {"E46_CELL_M": m, "E46_CELL_IPG": ipg}, (CELL,))
        if res["status"] != "ok":
            return dict(out, status="cell_%d_%s" % (m, res["status"]),
                        error=res.get("error"))
        na = na_cells(m, ipg)
        cells[m] = dict(res["functions"][CELL], ipg=ipg, na_cells=na,
                        law_prediction=law(max(na), len(na)))

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
        acc_alloca_types_by_width={m: cells[m]["acc_alloca_types"] for m in cells},
        law_residual_by_width={
            m: cells[m]["peak_live_regs"] - cells[m]["law_prediction"]
            for m in cells},
    )
    if tool is not None:
        out["occupancy"] = occupancy(shadow, tool)
    return out


def extra_cells(base_text: str, workdir: pathlib.Path) -> dict:
    """Isolated `<T,7,7>` and `<T,8,8>`: census only, never timed, never built."""
    shadow = workdir / "extra_cells"
    dst = shadow / HEADER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for (m, ipg), predicted in PREDICTED_EXTRA_CELLS.items():
        dst.write_text(relax_na(base_text, ipg))
        res = compile_probe(shadow, "extra_m%d_ipg%d" % (m, ipg),
                            {"E46_CELL_M": m, "E46_CELL_IPG": ipg}, (CELL,))
        key = "<T,%d,%d>" % (m, ipg)
        if res["status"] != "ok":
            out[key] = {"status": res["status"], "error": res.get("error"),
                        "family": "census_probe", "never_time": True}
            continue
        na = na_cells(m, ipg)
        stats = res["functions"][CELL]
        out[key] = dict(stats, status="ok", family="census_probe",
                        never_time=True, na_cells=na,
                        law_prediction=law(max(na), len(na)),
                        predicted=predicted,
                        hit=stats["peak_live_regs"] == predicted)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e61-artifacts/e61-reg-census.json")
    ap.add_argument("--rev", default=BASE_REV)
    ap.add_argument("--no-occupancy", action="store_true")
    args = ap.parse_args()

    base_text = header_at(args.rev)
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="e61-reg-census-"))
    try:
        tool = None if args.no_occupancy else build_occupancy_tool(workdir)
        arms = [run_arm(a, base_text, workdir, tool) for a in ARMS]
        extras = extra_cells(base_text, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    by = {a["name"]: a for a in arms}
    ok = [a for a in arms if a["status"] == "ok"]

    cell_checks = {}
    for (name, m), predicted in PREDICTED_CELL.items():
        a = by.get(name)
        if a is None or a["status"] != "ok" or m not in a["width_cells"]:
            cell_checks["%s M=%d" % (name, m)] = "unavailable"
            continue
        got = a["width_cells"][m]["peak_live_regs"]
        cell_checks["%s M=%d" % (name, m)] = {
            "measured": got, "predicted": predicted, "hit": got == predicted}

    table_checks = {}
    for name, predicted in PREDICTED_TABLE_MAX.items():
        a = by.get(name)
        if a is None or a["status"] != "ok":
            table_checks[name] = "unavailable"
            continue
        table_checks[name] = {"measured": a["kernel_wide_reg_max"],
                              "predicted": predicted,
                              "hit": a["kernel_wide_reg_max"] == predicted}

    entry_checks = {}
    for name, (lo, hi) in PREDICTED_ENTRY_INTERVAL.items():
        a = by.get(name)
        if a is None or a["status"] != "ok":
            entry_checks[name] = "unavailable"
            continue
        got = a["entry_batch0"]
        entry_checks[name] = {"measured": got, "interval": [lo, hi],
                              "hit": lo <= got <= hi}

    law_hits = {a["name"]: a["law_residual_by_width"] for a in ok}
    law_survives = all(all(v == 0 for v in r.values()) for r in law_hits.values())
    law_survives = law_survives and all(
        c.get("hit") for c in extras.values() if isinstance(c, dict) and c.get("status") == "ok")

    payload = {
        "head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
        "base_rev": args.rev,
        "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
        "law": "reg = 20 + 21*max(NA) + 4*[two distinct NA group sizes]",
        "law_note": ("the +4 term reproduces this instrument's documented "
                     "constant over-count on mixed-group cells (E54), so the "
                     "hardware ceiling of a mixed cell is very likely 4 lower "
                     "than reported here; single-group cells carry no "
                     "over-count"),
        "arms": arms,
        "extra_cells_census_only": extras,
        "kernel_wide_reg_max": {a["name"]: a["kernel_wide_reg_max"] for a in ok},
        "entry_batch0": {a["name"]: a["entry_batch0"] for a in ok},
        "entry_batch1": {a["name"]: a["entry_batch1"] for a in ok},
        "cell_prediction_checks": cell_checks,
        "table_max_prediction_checks": table_checks,
        "entry_prediction_checks": entry_checks,
        "law_residual_by_arm": law_hits,
        "law_survives": law_survives,
        "all_ok": all(a["status"] == "ok" for a in arms),
    }
    if by["shipped"]["status"] == "ok" and by["t6"]["status"] == "ok":
        b, c = by["shipped"], by["t6"]
        payload["t6_vs_shipped"] = {
            "kernel_wide_reg_max_delta":
                c["kernel_wide_reg_max"] - b["kernel_wide_reg_max"],
            "entry_batch0_delta": c["entry_batch0"] - b["entry_batch0"],
            "shipped_argmax_width": b["argmax_width"],
            "t6_argmax_width": c["argmax_width"],
        }
    if by["ballast"]["status"] == "ok" and by["t6"]["status"] == "ok":
        payload["ballast_matches_t6_ceiling"] = (
            by["ballast"]["kernel_wide_reg_max"] == by["t6"]["kernel_wide_reg_max"])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: payload[k] for k in (
        "kernel_wide_reg_max", "entry_batch0", "cell_prediction_checks",
        "table_max_prediction_checks", "entry_prediction_checks",
        "law_survives", "all_ok")}, indent=2, sort_keys=True))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
