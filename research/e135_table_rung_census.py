#!/usr/bin/env python3
"""F45 section 5: register, spill and residency census of all four table rungs.

    usage: research/e135_table_rung_census.py [--out PATH] [--keep DIR]

The four `Table` arms differ only in which widths they serve with one pass:

    shipped      6 -> [3+3]   7 -> [4+3]   8 -> [4+4]
    onePass6     6 -> [6]     7 -> [4+3]   8 -> [4+4]
    onePass67    6 -> [6]     7 -> [7]     8 -> [4+4]
    onePass678   6 -> [6]     7 -> [7]     8 -> [8]

`onePass678` was retired because width 8 spills on `applegpu_g17s`. F45 asks
whether the same objection reaches width 7, because that is the only cell that
separates `onePass6` from the shipped `onePass67` default. This census answers
it without a GPU: `xcrun metal-tt` runs the real AGX backend for a named
architecture, wrapped by `research/agx_crossarch.py`.

CAMPAIGN RULE 101. Six cells are pinned from advisor F22 and F45 and are
asserted before any new cell is read. Controls are scored per architecture,
because Rule 83 already restricts the closure to `applegpu_g17s`: a moved
g17s control voids the answer, and a moved g16s control voids only the
completeness column.

CAMPAIGN RULE 99. This is a detector, not a judge. A residency difference earns
a measurement, not a claim.

CAMPAIGN RULE 83. A g16s closure is not g17s evidence. Both hosts are reported
so the two-host table is complete, and only the g17s column decides.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e120_g17s_census as base  # noqa: E402
import e135_receipt_read as rr  # noqa: E402
import e135_rung_residency as res  # noqa: E402

# `tablePays(m) = m >= 4` (Qwen35.swift:2322), so widths 2 and 3 run the
# non-table entry point and are censused without the `xsums` binding.
MINIMUM_TABLE_WIDTH = 4

# `m: items per group` for every width each arm routes. The rps is 4 throughout.
ARMS = {
    "shipped":    {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3},
    "onePass6":   {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 3},
    "onePass67":  {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3},
    "onePass678": {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 3},
}

# Pinned cells from advisor F22 section 1 and F45 section 5.
CONTROLS = {
    ("applegpu_g17s", 6, 3): {"registers": 94, "spill_bytes": 0,
                              "resident_simdgroups": 42},
    ("applegpu_g17s", 6, 6): {"registers": 105, "spill_bytes": 0,
                              "resident_simdgroups": 37},
    ("applegpu_g17s", 7, 7): {"registers": 118, "spill_bytes": 0,
                              "resident_simdgroups": 33},
    ("applegpu_g17s", 7, 4): {"registers": 96, "spill_bytes": 0,
                              "resident_simdgroups": 41},
    ("applegpu_g16s", 7, 7): {"registers": 96, "spill_bytes": 32},
    ("applegpu_g16s", 7, 4): {"registers": 96, "spill_bytes": 0},
}


def kernel_name(m: int, ipg: int) -> str:
    stem = ("qwen35_custom_affine4_g64_qmv_wide_sums_v2"
            if m >= MINIMUM_TABLE_WIDTH
            else "qwen35_custom_affine4_g64_qmv_wide_v2")
    return "%s_m%d_na%d" % (stem, m, ipg)


def source(header: str, m: int, ipg: int, rps: int = 4) -> str:
    """One library holding the single width case this plan compiles to."""
    table = m >= MINIMUM_TABLE_WIDTH
    inputs = base.QMV_INPUTS + ([("xsums", "float")] if table else [])
    template = [("bool", "USE_TABLE", "true")] if table else None
    text = base.generate(kernel_name(m, ipg), inputs, base.QMV_OUTPUTS,
                         base.qmv_body(table, ((m, ipg, rps),)), template)
    return "\n".join([base.PRELUDE, header, "", text]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(
        "research/e135-artifacts/table-rung-census.json"))
    ap.add_argument("--keep", type=pathlib.Path)
    ap.add_argument("--board", default="/tmp/yukon-board/read.json")
    ap.add_argument("--receipt", default="0cf1637e")
    ap.add_argument("--trace", default="research/out/e135shipexact/trace.txt")
    args = ap.parse_args()

    header = base.swift_literal("qwen35E120QMVHeader")
    plans = sorted({(m, ipg) for arm in ARMS.values() for m, ipg in arm.items()})

    cells: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for m, ipg in plans:
            tag = "m%d_ipg%d" % (m, ipg)
            name = kernel_name(m, ipg)
            text = source(header, m, ipg)
            if args.keep:
                args.keep.mkdir(parents=True, exist_ok=True)
                (args.keep / (tag + ".metal")).write_text(text)
            try:
                got = base.census(text, tag, workdir)
            except subprocess.CalledProcessError as error:
                got = {"error": (error.stderr or b"").decode()[-2000:]}
            cells[tag] = {"m": m, "ipg": ipg, "name": name, "census": got}

    def cell(arch: str, m: int, ipg: int) -> dict:
        entry = cells["m%d_ipg%d" % (m, ipg)]
        return entry["census"].get(arch, {}).get(entry["name"], {})

    print("## Controls, Rule 101")
    print("  Convention: one library per (m, ipg) plan, so each row is the")
    print("  single-width specialization, not the tiered switch's maximum.")
    failed = {arch: 0 for arch in base.ARCHS}
    for (arch, m, ipg), want in sorted(CONTROLS.items()):
        got = cell(arch, m, ipg)
        bad = [k for k, v in want.items() if got.get(k) != v]
        failed[arch] += bool(bad)
        print("  %-14s m%d ipg%d  %s" % (
            arch.replace("applegpu_", ""), m, ipg,
            "OK" if not bad else "MISMATCH " + str(
                {k: (want[k], got.get(k)) for k in bad})))
    decider = "applegpu_g17s"
    print("\n  %s controls %s. Rule 83 makes this the only deciding column."
          % (decider.replace("applegpu_", ""),
             "all pass" if not failed[decider] else "MOVED: the census is void"))
    for arch in base.ARCHS:
        if arch != decider and failed[arch]:
            print("  %s carries %d moved control(s). Its column is reported"
                  % (arch.replace("applegpu_", ""), failed[arch]))
            print("  for completeness and closes nothing.")

    print("\n## Per-width cells")
    print("  arch    m  ipg  cols  registers  spill B  simdgroups")
    for arch in base.ARCHS:
        for m, ipg in plans:
            got = cell(arch, m, ipg)
            print("  %-6s %2d %4d %5d %10s %8s %11s" % (
                arch.replace("applegpu_", ""), m, ipg,
                math.ceil(m / ipg), got.get("registers", "?"),
                got.get("spill_bytes", "?"),
                got.get("resident_simdgroups", "?")))

    row = rr.find(rr.load_board(args.board), args.receipt)
    entries = rr.per_prompt(row)
    ranked = {m: 0.0 for m in res.SUPPORT}
    for name, weight in rr.RULE148_WEIGHTS.items():
        dist = res.maxent(1.0 + entries[name]["effective_mean_draft_len"])
        for m in res.SUPPORT:
            ranked[m] += weight * dist[m]
    scale = sum(rr.RULE148_WEIGHTS.values())
    ranked = {m: p / scale for m, p in ranked.items()}
    local, rounds = res.local_histogram(args.trace)

    print("\n## Arm residency, weighted by where the run sits")
    print("  Ranked weights are the Rule 148 vector over a maximum-entropy")
    print("  width reconstruction; the local column is the measured histogram")
    print("  of %d traced rounds. Occupancy is weighted, not summed." % rounds)
    print("  arm          arch    ranked sg   local sg   spilling widths")
    summary: dict = {}
    for name, plan in ARMS.items():
        for arch in base.ARCHS:
            wr = ws = 0.0
            spills = []
            for m, ipg in sorted(plan.items()):
                got = cell(arch, m, ipg)
                sg = got.get("resident_simdgroups")
                if sg is None:
                    continue
                wr += ranked.get(m, 0.0) * sg
                ws += local.get(m, 0.0) * sg
                if got.get("spill_bytes"):
                    spills.append("%d(%dB)" % (m, got["spill_bytes"]))
            summary["%s/%s" % (name, arch)] = {"ranked_sg": wr, "local_sg": ws,
                                               "spills": spills}
            print("  %-11s  %-6s %10.3f %10.3f   %s" % (
                name, arch.replace("applegpu_", ""), wr, ws,
                ", ".join(spills) or "none"))

    print("\n## F45 section 5, the two closure conditions on %s"
          % decider.replace("applegpu_", ""))
    spill6 = summary["onePass6/" + decider]["spills"]
    spill67 = summary["onePass67/" + decider]["spills"]
    print("  1. no spill in either arm:  onePass6 %s, onePass67 %s  -> %s"
          % (spill6 or ["none"], spill67 or ["none"],
             "MET" if not spill6 and not spill67 else "NOT MET"))
    two_pass = cell(decider, 7, 4)
    one_pass = cell(decider, 7, 7)
    better = (one_pass.get("resident_simdgroups", 0)
              > two_pass.get("resident_simdgroups", 0))
    print("  2. strictly better residency at width 7 for onePass67:")
    print("     two-pass (7,4,4) %s regs / %s sg   one-pass (7,7,4) %s regs"
          " / %s sg  -> %s"
          % (two_pass.get("registers"), two_pass.get("resident_simdgroups"),
             one_pass.get("registers"), one_pass.get("resident_simdgroups"),
             "MET" if better else "NOT MET"))
    print("  The one-pass width-7 body buys one fewer column with %.1f %%"
          % (100.0 * (1.0 - one_pass.get("resident_simdgroups", 0)
                      / max(two_pass.get("resident_simdgroups", 1), 1))))
    print("  less occupancy. Width 6 pays %.1f %% for the same trade and a"
          % (100.0 * (1.0 - cell(decider, 6, 6).get("resident_simdgroups", 0)
                      / max(cell(decider, 6, 3).get("resident_simdgroups", 1),
                            1))))
    print("  ranked receipt already measured that composite net positive.")
    print("  Ranked residency: width 6 %.4f, width 7 %.4f of rounds."
          % (ranked[6], ranked[7]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "harness": "local",
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "instrument": "xcrun metal-tt via research/agx_crossarch.py",
        "simdgroup_budget": base.SIMDGROUP_BUDGET,
        "deciding_arch": decider,
        "controls_moved_by_arch": failed,
        "controls_passed": failed[decider] == 0,
        "receipt": row["id"],
        "ranked_width_maxent": ranked,
        "local_width_histogram": local,
        "local_round_count": rounds,
        "arms": ARMS,
        "cells": cells,
        "summary": summary,
    }, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
