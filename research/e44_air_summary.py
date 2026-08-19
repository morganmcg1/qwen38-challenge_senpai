#!/usr/bin/env python3
"""E44 Gate 0 summary: kernel-wide register/spill/threadgroup readout per arm.

Three readouts, because the campaign has no true register readout on this box
(see campaign-ledger item on E40 E: `-mllvm -stats` disabled, `-Rpass*` silent,
`metal-objdump` stops at AIR, pipeline reflection cannot discriminate):

  regs      lane-weighted peak-live-SSA textual heuristic. Shape usable, the
            absolute number is not. Printed naive AND lane-corrected, because
            AIR models an 8x8 simdgroup matrix as one simdgroup-wide
            `<64 x float>` value whose 64 elements live across 32 lanes, so the
            naive count over-reports it by exactly 32x.
  allocas   private-memory arrays that survived -O2/-O3. A genuine compiler
            outcome, and the discriminator E40 recommended over the register
            heuristic.
  tg_bytes  static threadgroup-memory bytes. `affine_qmv_fast` is ONE kernel, so
            a threadgroup array declared for one width cell is allocated for
            every dispatch of every width -- the same shared-allocation channel
            as registers, which no earlier experiment has had to measure.

Run with --selftest to check the lane correction against a synthetic body.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from air_kernel_stats import (  # noqa: E402
    ALLOCA,
    kernels,
    peak_live_breakdown,
    peak_live_registers,
)

# `@x = internal addrspace(3) global [64 x float] undef, align 4`
TG_GLOBAL = re.compile(r"^@([\w.]+)\s*=.*addrspace\(3\)\s+global\s+(.+?),\s*align")
ARRAY = re.compile(r"\[(\d+)\s+x\s+(.+)\]$")
SCALAR_BYTES = {
    "float": 4,
    "i32": 4,
    "i64": 8,
    "double": 8,
    "half": 2,
    "bfloat": 2,
    "i16": 2,
    "i8": 1,
    "i1": 1,
}

# The >=4096 dispatch table of the post-E27-revert base tree, and the candidate
# table, keyed by the runtime `ntg.x` value each case answers. `narrow_m2` is the
# same `qmv_fast_crossrow_affine4_g64<T,2>` both branches call at M=2.
BASE_TABLE = {
    2: "e44_narrow_m2",
    3: "e44_m3_ipg3",
    4: "e44_m4_ipg4",
    5: "e44_m5_ipg3",
    6: "e44_m6_ipg3",
    7: "e44_m7_ipg4",
    8: "e44_m8_ipg4",
    9: "e44_m9_ipg3",
}
CAND_TABLE = {
    2: "e44_narrow_m2",
    3: "e44_m3_ipg3",
    4: "e44_sgmm_runtime",
    5: "e44_sgmm_runtime",
    6: "e44_sgmm_runtime",
    7: "e44_sgmm_runtime",
    8: "e44_sgmm_runtime",
    9: "e44_sgmm_runtime",
}
# A hardcoded dispatch table is the campaign's most repeated error: it is quoted
# from whichever tree the author last read. `--dispatch-from` derives the table
# from the header actually being measured, so the arm cannot silently disagree
# with the tree, and the tables above become a self-check rather than an input.
WIDE_BRANCH = re.compile(
    r"if \(out_vec_size >= 4096\) \{(.*?)\n    \} else \{", re.DOTALL
)
CASE_LABEL = re.compile(r"^\s*case (\d+):\s*$")
CALL_M = re.compile(r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>")
CALL_N = re.compile(r"qmv_fast_crossrow_affine4_g64<T, (\d+)>")
CALL_SGMM = re.compile(r"qmv_fast_crossrow_affine4_g64_sgmm<T>")


def dispatch_table_from_header(path: pathlib.Path) -> dict[int, str]:
    """Map each `ntg.x` case in the >=4096 branch to its probe cell name."""
    branch = WIDE_BRANCH.search(path.read_text())
    if not branch:
        raise SystemExit(f"{path}: no >=4096 dispatch branch found")
    table: dict[int, str] = {}
    pending: list[int] = []
    for line in branch.group(1).splitlines():
        label = CASE_LABEL.match(line)
        if label:
            pending.append(int(label.group(1)))
            continue
        if not pending:
            continue
        if CALL_SGMM.search(line):
            cell = "e44_sgmm_runtime"
        elif (match := CALL_M.search(line)) is not None:
            cell = f"e44_m{match.group(1)}_ipg{match.group(2)}"
        elif (match := CALL_N.search(line)) is not None:
            cell = f"e44_narrow_m{match.group(1)}"
        else:
            continue
        for m in pending:
            table[m] = cell
        pending = []
    if not table:
        raise SystemExit(f"{path}: dispatch branch parsed but no cells matched")
    return table


# The <4096 branch is inlined into the same kernel by both arms, so it bounds
# the kernel-wide maximum from below for both.
NARROW_TABLE = {m: f"e44_narrow_m{m}" for m in range(2, 10)}
ENTRY_KERNELS = (
    "e40_affine_qmv_fast_bf16_gs64_b4_batch0",
    "e40_affine_qmv_fast_bf16_gs64_b4_batch1",
    "e40_affine_qmv_fast_bf16_gs64_b2_batch0",
)


def type_bytes(text: str) -> int:
    text = text.strip()
    match = ARRAY.match(text)
    if match:
        return int(match.group(1)) * type_bytes(match.group(2))
    return SCALAR_BYTES.get(text.split()[0], 4)


def threadgroup_bytes(path: pathlib.Path) -> list[tuple[str, int]]:
    out = []
    for line in path.read_text().splitlines():
        match = TG_GLOBAL.match(line)
        if match:
            out.append((match.group(1), type_bytes(match.group(2))))
    return out


def measure(path: pathlib.Path) -> dict[str, dict[str, int]]:
    stats = {}
    for name, body in kernels(path).items():
        allocas = [ALLOCA.search(line).group(1) for line in body if ALLOCA.search(line)]
        naive = peak_live_registers(body, False)[0]
        lane = peak_live_registers(body, True)[0]
        split = peak_live_breakdown(body)
        # The breakdown is a second, independent sweep over the same intervals;
        # if it disagreed with the published peaks it would not be evidence.
        assert split["naive_peak_naive_total"] == naive, name
        assert split["lane_peak_lane_total"] == lane, name
        stats[name] = {
            "naive": naive,
            "lane": lane,
            "allocas": len(allocas),
            "alloca_types": sorted(set(allocas)),
            "loads": sum(1 for line in body if re.search(r"=\s*load\s", line)),
            "mma": sum(1 for line in body if "simdgroup_matrix" in line),
            "split": split,
        }
    return stats


def report_split(label: str, stats: dict, names: list[str]) -> None:
    """Adjudicate the naive-vs-corrected verdict split value by value.

    A naive number above the gate is only an instrument artifact if the whole
    excess sits in distributed `<32k x T>` values. Printing the two peak points
    separately is what makes that checkable: the ordinary-value subtotal at the
    lane peak is the part of the pressure that is NOT a modelling artifact.
    """
    shown = [n for n in names if n in stats and stats[n]["naive"] != stats[n]["lane"]]
    if not shown:
        return
    print()
    print("%s: verdict-split accounting (distributed = simdgroup matrices)" % label)
    print("  at the NAIVE peak point            | at the LANE peak point")
    print("  %-30s %7s %5s %7s | %7s %5s %7s"
          % ("kernel", "total", "dist", "ordin.", "total", "dist", "ordin."))
    for name in shown:
        s = stats[name]["split"]
        print("  %-30s %7d %5d %7d | %7d %5d %7d"
              % (name,
                 s["naive_peak_naive_total"],
                 s["naive_peak_distributed_values"],
                 s["naive_peak_ordinary"],
                 s["lane_peak_lane_total"],
                 s["lane_peak_distributed_values"],
                 s["lane_peak_ordinary"]))


def table_max(stats: dict, table: dict[int, str], key: str) -> tuple[str, int]:
    cells = set(table.values()) | set(NARROW_TABLE.values())
    present = [c for c in cells if c in stats]
    binding = max(present, key=lambda c: stats[c][key])
    return binding, stats[binding][key]


def report_cells(path: pathlib.Path, header: pathlib.Path | None = None) -> None:
    stats = measure(path)
    cand_table = CAND_TABLE
    if header is not None:
        cand_table = dispatch_table_from_header(header)
        print(f"candidate dispatch table derived from {header}")
        print("  " + "  ".join(f"{m}:{cell}" for m, cell in sorted(cand_table.items())))
        print()
    have_cand = all(c in stats for c in cand_table.values())

    print("per-cell footprint (regs: naive / lane-corrected)")
    print("%-24s %8s %8s %8s %8s %s" % ("cell", "naive", "lane", "allocas", "loads", "alloca types"))
    for name in sorted(stats):
        s = stats[name]
        print("%-24s %8d %8d %8d %8d %s"
              % (name, s["naive"], s["lane"], s["allocas"], s["loads"],
                 s["alloca_types"] or ""))

    print()
    print("KERNEL-WIDE MAXIMUM (what one register allocation must satisfy)")
    print("%-10s %-24s %8s %8s" % ("arm", "binding cell", "naive", "lane"))
    for label, table in (("base", BASE_TABLE), ("cand", cand_table)):
        if label == "cand" and not have_cand:
            print("%-10s %-24s %8s %8s" % (label, "(cell absent from tree)", "-", "-"))
            continue
        b_naive, v_naive = table_max(stats, table, "naive")
        b_lane, v_lane = table_max(stats, table, "lane")
        print("%-10s %-24s %8d %8d" % (label, f"{b_naive} / {b_lane}", v_naive, v_lane))

    print()
    print("per-M dispatch table (naive / lane-corrected)")
    print("%-4s %-24s %14s %-24s %14s" % ("M", "base cell", "base regs", "cand cell", "cand regs"))
    for m in sorted(BASE_TABLE):
        base = BASE_TABLE[m]
        cand = cand_table[m]
        cand_txt = ("%d / %d" % (stats[cand]["naive"], stats[cand]["lane"])) if cand in stats else "-"
        print("%-4d %-24s %14s %-24s %14s"
              % (m, base, "%d / %d" % (stats[base]["naive"], stats[base]["lane"]),
                 cand if cand in stats else "(absent)", cand_txt))

    anchors = [n for n in sorted(stats) if n.startswith("e44_wide_na")]
    if anchors:
        print()
        print("inner packing-factor anchor (E13/E27/E32/E40 read 62/83/104 for NA=2/3/4)")
        for name in anchors:
            print("  %-20s naive=%d lane=%d" % (name, stats[name]["naive"], stats[name]["lane"]))

    report_split("cells", stats, sorted(stats))

    tg = threadgroup_bytes(path)
    print()
    print("static threadgroup memory in module: %d bytes%s"
          % (sum(b for _, b in tg), (" " + repr(tg)) if tg else " (none)"))


def report_entry(base_path: pathlib.Path, cand_path: pathlib.Path) -> None:
    base, cand = measure(base_path), measure(cand_path)
    print("%-44s %-18s %-18s %s" % ("entry kernel", "base naive/lane", "cand naive/lane", "allocas b->c"))
    for name in ENTRY_KERNELS:
        if name not in base or name not in cand:
            print("  MISSING %s" % name)
            continue
        b, c = base[name], cand[name]
        new_types = sorted(set(c["alloca_types"]) - set(b["alloca_types"]))
        print("%-44s %-18s %-18s %s"
              % (name,
                 "%d / %d" % (b["naive"], b["lane"]),
                 "%d / %d" % (c["naive"], c["lane"]),
                 "%d -> %d %s" % (b["allocas"], c["allocas"],
                                  ("NEW TYPES " + repr(new_types)) if new_types
                                  else "no new alloca type")))
        print("      base alloca types: %s" % (b["alloca_types"] or "none"))
        print("      cand alloca types: %s" % (c["alloca_types"] or "none"))
    for label, path in (("base", base_path), ("cand", cand_path)):
        tg = threadgroup_bytes(path)
        print("  %-4s static threadgroup memory: %d bytes%s"
              % (label, sum(b for _, b in tg), (" " + repr(tg)) if tg else ""))
    report_split("entry (cand)", cand, list(ENTRY_KERNELS))


def selftest() -> int:
    # A `<64 x float>` simdgroup-matrix value is one value per SIMDGROUP: the
    # naive lane weighting must read 64 and the corrected one 2, while an
    # ordinary `<4 x float>` must be untouched by the flag.
    body = [
        "  %1 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float 0.0)",
        "  %2 = fadd <4 x float> %1, %1",
        "  ret void %1 %2",
    ]
    naive = peak_live_registers(body, False)[0]
    lane = peak_live_registers(body, True)[0]
    ok = naive == 68 and lane == 6
    print("selftest: naive=%d (expect 68 = 64 + 4) lane=%d (expect 6 = 2 + 4) -> %s"
          % (naive, lane, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("air_ll", nargs="*")
    ap.add_argument("--cells", action="store_true")
    ap.add_argument("--dispatch-from", type=pathlib.Path,
                    help="derive the candidate table from this header instead of "
                         "the built-in one")
    ap.add_argument("--entry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.cells:
        report_cells(pathlib.Path(args.air_ll[0]), args.dispatch_from)
        return 0
    if args.entry:
        report_entry(pathlib.Path(args.air_ll[0]), pathlib.Path(args.air_ll[1]))
        return 0
    ap.error("one of --cells, --entry, --selftest is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
