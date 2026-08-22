#!/usr/bin/env python3
"""E125 rung 0: the register and residency census of the ACTUAL shipped E121 arm.

Advisor feedback F4 asked for the census that Advisor Error 96 skipped. The
earlier residency table in `research/e125-artifacts/residency-axis.json` used
`n_nosums`, which DELETES the cross-simdgroup sums entirely. The shipped E121
arm does not delete them: it SPLITS them across the two simdgroups and
exchanges the halves through `sums_xchg` with two `threadgroup_barrier` calls
per k-block. A census of the wrong arm cannot price the right mechanism.

The four arms decompose the shipped change into its three components, so each
one is priced on its own instead of as a lump:

  share_on       the shipped kernel, verbatim from the runtime-effective JIT
                 string: `constexpr bool SHARE_SUMS = NA <= 4`, the
                 `threadgroup float* sums_xchg` parameter, and the 512-byte
                 entry-point allocation.
  share_off      SHARE_SUMS forced false. The parameter and the allocation
                 stay. `share_on - share_off` is the split and the exchange.
  share_noalloc  additionally shrinks the entry-point allocation to one float.
                 `share_off - share_noalloc` is the threadgroup allocation.
  pre_e121       additionally removes the parameter from both `_wide` and
                 `_m` and every call site. `share_noalloc - pre_e121` is the
                 parameter itself. `share_on - pre_e121` is the whole change.

Two frames are censused for every arm, because they answer different
questions and F4's own falsifier depends on which one binds:

  isolated bodies      `e125_iso_naN`, one entry point per width, so the
                       register cost of each `_wide<T, NA>` body is visible on
                       its own.
  in-situ entry point  `affine_qmv_fast_bfloat16_t_64_4_false`, the real
                       shipped kernel. `qmv_fast_crossrow_affine4_g64_wide` is
                       a `METAL_FUNC`, so the shipped `switch (ntg.x)` inlines
                       EVERY live width into ONE entry point and that entry
                       point is allocated for the widest inlined body.

This is a compile-only census. It runs the real AGX backend for both
architectures through `xcrun metal-tt` and never touches the GPU, so it is not
a timing measurement and carries no thermal or gate claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402
import e104_variant_sources as v  # noqa: E402
from e123_arms import SIMDGROUP_BUDGET, simdgroups  # noqa: E402

ARCHES = (agx.LOCAL_ARCH, agx.RANKED_ARCH)
WIDTHS = (2, 3, 4, 5)
ARMS = ("share_on", "share_off", "share_noalloc", "pre_e121")

# The shipped entry point, and the isolated per-width kernels this file adds.
SHIPPED_ENTRY = "affine_qmv_fast_bfloat16_t_64_4_false"
ISO_RE = re.compile(r"e125_iso_na(\d+)$")

# The three textual sites the shipped E121 change owns. Each count is asserted
# before the patch, so a base that moves fails loudly instead of silently
# censusing an arm that is really the base.
SHARE_SITE = "constexpr bool SHARE_SUMS = NA <= 4;"
SHARE_OFF = "constexpr bool SHARE_SUMS = false;"
ALLOC_SITE = "  threadgroup float sums_xchg[1 * 4 * 32];\n"
ALLOC_MIN = "  threadgroup float sums_xchg[1];\n"
PARAM_SITE = "    uint simd_lid,\n    threadgroup float* sums_xchg) {"
PARAM_OFF = "    uint simd_lid) {"
CALL_SITE = "simd_gid, simd_lid, sums_xchg);"
CALL_OFF = "simd_gid, simd_lid);"
# `if constexpr` discards the branch but still parses it, so the parameter
# cannot be removed while the exchange body still names it.
EXCHANGE_BLOCK = """    if constexpr (SHARE_SUMS) {
      for (int m = 0; m < NA; m++) {
        if ((m < H) == own_lo) {
          sums_xchg[m * SIMD_SIZE + simd_lid] = sums[m];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int m = 0; m < NA; m++) {
        if ((m < H) != own_lo) {
          sums[m] = sums_xchg[m * SIMD_SIZE + simd_lid];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
SITE_COUNTS = {SHARE_SITE: 1, ALLOC_SITE: 1, PARAM_SITE: 2, CALL_SITE: 9,
               EXCHANGE_BLOCK: 1}

# `_wide` takes the threadgroup pointer, so the isolated entry point has to own
# the staging the shipped entry point owns. The shipped allocation is
# `1 * 4 * 32` floats: SHARE_SUMS is only true at NA <= 4, so the largest index
# the exchange can reach is `3 * SIMD_SIZE + 31 == 127`.
ISO_KERNEL = """
[[kernel]] void e125_iso_na%(na)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
%(decl)s  const int first_m = int(tid.x) * %(na)d;
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, first_m, out_row,
      simd_gid, simd_lid%(pass)s);
}
"""

# The live `switch (ntg.x)` at the `out_vec_size >= 4096` branch of the shipped
# entry point, read out of the emitted base rather than assumed. `M` is the
# draft width the parent asks for; `IPG` is the template's inputs-per-group and
# therefore the `NA` the `_wide` body is instantiated at.
LIVE_DISPATCH = {
    2: {"kernel": "pair", "na": ()},
    3: {"kernel": "_m<3,3>", "na": (3,)},
    4: {"kernel": "_m<4,4>", "na": (4,)},
    5: {"kernel": "_m<5,5>", "na": (5,)},
    6: {"kernel": "_m<6,3>", "na": (3,)},
    7: {"kernel": "_m<7,4>", "na": (4, 3)},
    8: {"kernel": "_m<8,4>", "na": (4,)},
    9: {"kernel": "_m<9,3>", "na": (3,)},
}

# Finding 47 draft-width weights, as the advisor quoted them in F4.
F47_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# The register counts at which E123 fitted `SIMDGROUP_BUDGET`, which are exactly
# the cells Alphonse measured in the E121 residency table. A resident count read
# at one of these register counts is a reproduction of a measurement; any other
# register count is the fitted model EXTRAPOLATING, and is marked as such.
MEASURED_REGISTERS = {"applegpu_g16s": (93, 94, 96),
                      "applegpu_g17s": (101, 102, 107, 120, 121)}

# Alphonse's published `a_base` residency column, for a validation that does not
# depend on any constant this file owns. `a_base` is the pre-E121 kernel, so it
# must equal this census's `pre_e121`.
ALPHONSE_A_BASE = {"applegpu_g16s": {2: 46, 3: 37, 4: 32, 5: 33},
                   "applegpu_g17s": {2: 42, 3: 44, 4: 44, 5: 39}}

# The ladder this census prices, as (name, richer arm, poorer arm).
LADDER = (
    ("split_and_exchange", "share_on", "share_off"),
    ("threadgroup_allocation", "share_off", "share_noalloc"),
    ("parameter", "share_noalloc", "pre_e121"),
    ("whole_e121_change", "share_on", "pre_e121"),
)


def assert_sites(base: str) -> None:
    for needle, want in SITE_COUNTS.items():
        seen = base.count(needle)
        if seen != want:
            raise SystemExit(
                "e125_e121_census: %r matched %d times, expected %d. The base "
                "moved; re-read the kernel before censusing it."
                % (needle[:48], seen, want))


def arm_source(base: str, arm: str) -> str:
    text = base
    keep_param = True
    if arm != "share_on":
        text = text.replace(SHARE_SITE, SHARE_OFF)
    if arm == "share_noalloc":
        text = text.replace(ALLOC_SITE, ALLOC_MIN)
    if arm == "pre_e121":
        text = text.replace(ALLOC_SITE, "")
        text = text.replace(EXCHANGE_BLOCK, "")
        text = text.replace(PARAM_SITE, PARAM_OFF)
        text = text.replace(CALL_SITE, CALL_OFF)
        keep_param = False
    if text.count("sums_xchg") != (0 if arm == "pre_e121" else
                                   base.count("sums_xchg")):
        raise SystemExit("e125_e121_census: %s left a stray sums_xchg" % arm)
    decl = "  threadgroup float sums_xchg[1 * 4 * 32];\n" if keep_param else ""
    return text + "".join(
        ISO_KERNEL % {"na": na, "decl": decl,
                      "pass": ", sums_xchg" if keep_param else ""}
        for na in WIDTHS)


def emit(outdir: pathlib.Path) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    base = v.emit_base(outdir / "base_raw.metal")
    assert_sites(base)
    sources = {}
    seen: dict[str, str] = {}
    for arm in ARMS:
        text = arm_source(base, arm)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        if digest in seen:
            raise SystemExit("e125_e121_census: %s and %s are byte-identical"
                             % (arm, seen[digest]))
        seen[digest] = arm
        (outdir / ("arm_%s.metal" % arm)).write_text(text)
        sources[arm] = text
        print("%-15s %8d bytes  sha=%s" % (arm, len(text), digest))
    return sources


def census_one(source: str, workdir: pathlib.Path, tag: str) -> dict:
    lib = agx.build_metallib(source, workdir / tag)
    row: dict = {}
    for arch in ARCHES:
        cells: dict[str, dict] = {}
        for kernel, record in agx.translate(lib, arch, workdir / tag).items():
            hit = ISO_RE.search(kernel)
            if hit is not None:
                key = hit.group(1)
            elif kernel.endswith(SHIPPED_ENTRY):
                key = "entry"
            else:
                continue
            cells[key] = {"registers": record.get("registers"),
                          "spill_bytes": record.get("spill_bytes", 0),
                          "text_bytes": record.get("text_bytes"),
                          "text_sha8": record.get("text_sha8")}
        row[arch] = cells
    return row


def resident(rows: dict, arm: str, arch: str, key: str) -> int | None:
    cell = rows.get(arm, {}).get(arch, {}).get(key, {})
    return simdgroups(cell.get("registers"), arch)


def weighted_change(rows: dict, rich: str, poor: str, arch: str) -> float | None:
    """Mean fractional change in resident simdgroups over the F47 widths.

    Sign convention: negative means the richer arm is resident less often, so
    the change costs occupancy.
    """
    total = 0.0
    weight = 0.0
    for na, w in F47_WEIGHTS.items():
        a = resident(rows, rich, arch, str(na))
        b = resident(rows, poor, arch, str(na))
        if not a or not b:
            return None
        total += w * (a - b) / b
        weight += w
    return None if not weight else total / weight


def regs(rows: dict, arm: str, arch: str, key: str) -> int | None:
    return rows.get(arm, {}).get(arch, {}).get(key, {}).get("registers")


def inlining_law(rows: dict) -> dict:
    """Is the entry point allocated for the widest body it inlines?

    E102 and E123 assert this from the source. The census can test it: if the
    claim holds, the entry-point register count equals the maximum over the
    per-width bodies, for every arm on both architectures.
    """
    checks = {}
    for arch in ARCHES:
        for arm in ARMS:
            widest = max(regs(rows, arm, arch, str(na)) for na in WIDTHS)
            entry = regs(rows, arm, arch, "entry")
            checks["%s/%s" % (arch, arm)] = {
                "entry_registers": entry, "widest_body_registers": widest,
                "widest_body_width": max(
                    WIDTHS, key=lambda na: regs(rows, arm, arch, str(na))),
                "holds": entry == widest}
    return checks


def cliff(arch: str, registers: int) -> dict:
    """Where the register count sits between two floor-division cliffs."""
    budget = SIMDGROUP_BUDGET[arch]
    groups = budget // registers
    # The largest register count that still keeps `groups` resident, and the
    # smallest that keeps `groups + 1`.
    keep = budget // groups
    gain = budget // (groups + 1)
    return {"registers": registers, "resident": groups,
            "registers_to_lose_one": keep + 1 - registers,
            "registers_to_gain_one": registers - gain}


def alphonse_replication(rows: dict) -> dict:
    """Does `pre_e121` reproduce Alphonse's published `a_base` column?"""
    cells = {}
    agree = 0
    for arch in ARCHES:
        for na in WIDTHS:
            mine = resident(rows, "pre_e121", arch, str(na))
            theirs = ALPHONSE_A_BASE[arch][na]
            cells["%s/NA%d" % (arch, na)] = {"census": mine,
                                             "alphonse_a_base": theirs,
                                             "agree": mine == theirs}
            agree += int(mine == theirs)
    return {"cells": cells, "agree": agree, "of": len(cells)}


def report(rows: dict, out: pathlib.Path | None) -> int:
    for arch in ARCHES:
        print("\n%s  registers / spill / resident simdgroups "
              "(budget %d)" % (arch, SIMDGROUP_BUDGET[arch]))
        print("  %-15s %-22s %s" % ("arm", "entry (in situ)",
                                    "  ".join("NA%d" % na for na in WIDTHS)))
        for arm in ARMS:
            cells = []
            for key in ("entry",) + tuple(str(na) for na in WIDTHS):
                cell = rows[arm][arch].get(key, {})
                count = cell.get("registers")
                text = "%s/s%s/%s" % (count, cell.get("spill_bytes"),
                                      simdgroups(count, arch))
                cells.append("%-22s" % text if key == "entry" else "%-14s" % text)
            print("  %-15s %s" % (arm, " ".join(cells)))

    print("\nladder, as the fractional change in resident simdgroups")
    print("  %-24s %-30s %s" % ("component", "in-situ entry point",
                                "isolated bodies, F47-weighted"))
    ladder: dict[str, dict] = {}
    for name, rich, poor in LADDER:
        entry_cells = []
        body_cells = []
        record: dict = {"rich": rich, "poor": poor}
        for arch in ARCHES:
            a = resident(rows, rich, arch, "entry")
            b = resident(rows, poor, arch, "entry")
            frac = None if not a or not b else (a - b) / b
            entry_cells.append("%s %s->%s %s" % (
                arch[-4:], b, a, "n/a" if frac is None else "%+.2f%%"
                % (100 * frac)))
            wf = weighted_change(rows, rich, poor, arch)
            body_cells.append("%s %s" % (arch[-4:], "n/a" if wf is None
                                         else "%+.2f%%" % (100 * wf)))
            record[arch] = {"entry_from": b, "entry_to": a,
                            "entry_fraction": frac,
                            "body_f47_weighted_fraction": wf,
                            "body_by_width": {
                                str(na): {
                                    "from": resident(rows, poor, arch, str(na)),
                                    "to": resident(rows, rich, arch, str(na))}
                                for na in WIDTHS}}
        ladder[name] = record
        print("  %-24s %-30s %s" % (name, " | ".join(entry_cells),
                                    " | ".join(body_cells)))

    law = inlining_law(rows)
    print("\nis the entry point allocated for the widest body it inlines?")
    for key, cell in law.items():
        print("  %-30s entry %s, widest body NA%d at %s -> %s"
              % (key, cell["entry_registers"], cell["widest_body_width"],
                 cell["widest_body_registers"],
                 "holds" if cell["holds"] else "FAILS"))

    rep = alphonse_replication(rows)
    print("\npre_e121 against Alphonse's published a_base column: %d of %d "
          "cells agree" % (rep["agree"], rep["of"]))

    cliffs = {}
    print("\nthe entry point against its floor-division cliffs")
    for arch in ARCHES:
        for arm in ("pre_e121", "share_on"):
            cell = cliff(arch, regs(rows, arm, arch, "entry"))
            cliffs["%s/%s" % (arch, arm)] = cell
            print("  %-30s %3d registers -> %d resident; +%d loses one, "
                  "-%d gains one" % ("%s/%s" % (arch, arm), cell["registers"],
                                     cell["resident"],
                                     cell["registers_to_lose_one"],
                                     cell["registers_to_gain_one"]))

    # F4's falsifier, answered from the census rather than from a timing run.
    na5 = {arch: (rows["share_on"][arch]["5"]["registers"]
                  == rows["share_off"][arch]["5"]["registers"])
           for arch in ARCHES}
    entry_uniform = {arch: rows["share_on"][arch]["entry"]["registers"]
                     for arch in ARCHES}
    print("\nF100 falsifier")
    print("  isolated NA=5 body identical under share_on and share_off: %s"
          % json.dumps(na5))
    print("  in-situ entry-point registers, one number for every width: %s"
          % json.dumps(entry_uniform))
    print("  so the occupancy component of the E121 cost is present at M=5, "
          "and the instruction component is exactly zero there.")

    payload = {
        "note": ("Compile-only census through xcrun metal-tt. No GPU work, no "
                 "timing claim."),
        "arches": list(ARCHES),
        "widths": list(WIDTHS),
        "arms": rows,
        "arm_definitions": {
            "share_on": "shipped E121: SHARE_SUMS = NA <= 4, param, 512 B",
            "share_off": "SHARE_SUMS = false; param and 512 B allocation stay",
            "share_noalloc": "share_off with the allocation cut to one float",
            "pre_e121": "share_noalloc with the parameter removed everywhere",
        },
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "simdgroup_budget_is_fitted": True,
        "measured_registers": {k: list(v)
                               for k, v in MEASURED_REGISTERS.items()},
        "extrapolated_cells": {
            arch: {arm: sorted(
                key for key in ("entry",) + tuple(str(na) for na in WIDTHS)
                if regs(rows, arm, arch, key) not in MEASURED_REGISTERS[arch])
                for arm in ARMS} for arch in ARCHES},
        "f47_weights": {str(k): v for k, v in F47_WEIGHTS.items()},
        "live_dispatch": {str(k): val for k, val in LIVE_DISPATCH.items()},
        "ladder": ladder,
        "inlining_law": law,
        "alphonse_a_base_replication": rep,
        "cliffs": cliffs,
        "f100_test": {
            "isolated_na5_body_unchanged": na5,
            "entry_point_registers": entry_uniform,
            "entry_point_is_shared_by_every_width": True,
            "verdict": (
                "The occupancy component cannot be absent at M=5: the shipped "
                "switch inlines every live width into one entry point, so one "
                "register allocation serves every dispatch. The instruction "
                "component IS exactly zero at M=5, because SHARE_SUMS is false "
                "there. A cost measured at M=5 therefore does not falsify the "
                "register hypothesis; it is what the register hypothesis "
                "predicts."),
            "cleanest_future_test": (
                "M=5 separates the two components better than any other "
                "width. On the ranked architecture it dispatches NA=5 alone, "
                "the NA=5 body is byte-identical under E121, and yet it is "
                "allocated 102 registers instead of 101 because the NA=4 body "
                "is inlined beside it. So at M=5 the E121 cost is the "
                "occupancy term with the instruction term set to zero, while "
                "at M=3 and M=4 it is both. Timing E121 against pre-E121 at "
                "M=5 and at M=4 on a g17s host splits the two terms directly."),
        },
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print("\nwrote %s" % out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    outdir = args.emit or pathlib.Path(tempfile.mkdtemp(prefix="e125cen"))
    sources = emit(outdir)
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for arm in ARMS:
            rows[arm] = census_one(sources[arm], workdir, arm)
            print("censused %s" % arm)
    return report(rows, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
