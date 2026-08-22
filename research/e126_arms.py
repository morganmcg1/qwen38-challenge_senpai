#!/usr/bin/env python3
"""E126 rung 0 and rung 1 arms: price the activation chunk-sum tree Route B deletes.

    research/e126_arms.py --emit /tmp/e126-arms
    research/e126_arms.py --census /tmp/e126-arms --out research/e126-artifacts/rung0-census.json

Every arm is a textual transform of the RUNTIME-EFFECTIVE shipped source, taken
through `emit_base`, not a re-expression of the body against an older revision.
E121 finding F3 is the reason: the same mechanism written against two different
base revisions differed by 2.2x in measured effect. The shipped base already
carries the E121 signature and the entry-point exchange buffer, so no dispatch
patching is needed here and none is done.

Arms:

  share_off      `SHARE_SUMS` forced false everywhere. Bit exact, and identical
                 to the pre-E121 body: `owns_m` folds to true and the
                 `if constexpr (SHARE_SUMS)` exchange is not emitted.
  share_on       the shipped source, unmodified. `SHARE_SUMS = NA <= 4`.
  n_sums_free    the assignment's Route-B-shaped body: the sums tree AND the
                 `sums * bias_local[r]` consumer term removed.
  n_nosums_e123  E123's `n_nosums` construction: the tree removed, but the
                 consumer keeps a broadcast `+ bias_local[r]`. Carried so the
                 cross-instrument comparison against E123 and against
                 thorfinn's grid is exact rather than approximate.
  n_sums_loaded  what Route B's QMV would actually run: no tree, no barrier,
                 the full `sums * bias_local[r]` consumer, and `sums` read once
                 per row per k-block from the exchange buffer.

WHY THE FOURTH AND FIFTH ARMS EXIST. `n_sums_free` deletes consumer work that
Route B still pays, so it is an UPPER BOUND on Route B's QMV-side benefit and
not an estimate of it. `n_nosums_e123` is a slightly tighter upper bound.
`n_sums_loaded` is the only arm whose instruction mix matches the mechanism
being priced. Reporting the primary metric from an upper bound alone would
overstate Route B on the shipped base, which is the exact failure this
experiment exists to prevent.

Every `n_` arm is NUMERICALLY INVALID by construction. It computes a wrong
result on purpose, it is research-only, it is never submitted, and it never
runs end to end. The probe must carry it as `:diag`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH, RANKED_ARCH, build_metallib, translate,
)
from e104_variant_sources import (  # noqa: E402
    emit_base, wide_fn_span, widen_asserts,
)
from e121_arms import (  # noqa: E402
    ARM_KERNEL, KERNEL_RE, ENTRY_RE, REGISTER_FILE, TG_MEMORY_BYTES, WIDTHS,
    air_stats, simdgroups,
)

# Arm 0 is the exactness reference and the timing base, so it is `share_off`:
# every per-cell percentage is then directly comparable with E121 rung 2 and
# with E123's `a_base`. The harness runs its positive control on the LAST arm,
# so the last arm must be exact against arm 0, which puts `share_on` at the end.
ARMS = ("share_off", "n_sums_free", "n_nosums_e123", "n_sums_loaded",
        "share_on")
EXACT_ARMS = ("share_off", "share_on")

# Slabs of NA*32 floats each arm needs in the exchange buffer. `n_sums_loaded`
# alternates between two slabs so its address depends on the k-block index.
BUFFER_SLABS = {"n_sums_loaded": 2}
ENTRY_BUFFER = "  threadgroup float sums_xchg[1 * 4 * 32];\n"

# --- the exact shipped text each transform anchors on -------------------------

GATE = "  constexpr bool SHARE_SUMS = NA <= 4;\n"
GATE_OFF = "  constexpr bool SHARE_SUMS = false;\n"

TREE_DIRECT = """          if (owns_m) {
            sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
          }
"""
TREE_INDIRECT = """          if (owns_m) {
            sums[m] += xsum;
          }
"""
CONSUMER = "      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];\n"
CONSUMER_FREE = "      acc[r] += scale_local[r] * partial[r];\n"
CONSUMER_E123 = "      acc[r] += scale_local[r] * partial[r] + bias_local[r];\n"

# Route B publishes the sums from a replica kernel in an earlier dispatch, so
# the QMV reads them and needs no barrier of its own. Reading through the
# exchange buffer prices one threadgroup load per row per k-block; E123's
# `tgld` price converts that to a device read if the advisor prefers one.
#
# The ring index is not decoration. Route B's replica writes a DISTINCT value
# for every (row, k-block), so the read must depend on `k`. The first form of
# this arm addressed `[m * SIMD_SIZE + simd_lid]`, which is loop invariant, and
# the census caught the compiler hoisting all NA loads out of the k loop and
# reporting zero threadgroup traffic. Alternating over two slabs restores the
# dependence at the cost of one mask and one add per k-block.
LOAD_SUMS = """    const int sums_slab = ((k / block_size) & 1) * NA * SIMD_SIZE;
    for (int m = 0; m < NA; m++) {
      sums[m] = sums_xchg[sums_slab + m * SIMD_SIZE + simd_lid];
    }
"""

# Threadgroup memory starts undefined. Without a defined producer the compiler
# folds every read of `sums_xchg` to undef and deletes the loads, which the
# first census caught. Route B's real producer is a replica kernel in an
# earlier dispatch, so this arm seeds both slabs once per threadgroup from a
# runtime value and fences. The seed is 2*NA stores and one barrier per
# threadgroup, spread over the 10 k-blocks of the 5120-wide cell, so it
# over-charges the arm by about one fifth of a k-block of exchange traffic.
SEED_SUMS = """  if (simd_gid == 0) {
    for (int m = 0; m < NA; m++) {
      sums_xchg[m * SIMD_SIZE + simd_lid] = float(first_m + m);
      sums_xchg[(NA + m) * SIMD_SIZE + simd_lid] = float(first_m - m);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
"""
K_LOOP_OPEN = ("  for (int k = 0; k < in_vec_size; k += block_size) {\n"
               "    thread uint16_t packed[rows_per_simd][4];\n")
CONSUMER_LOOP_OPEN = "    for (int r = 0; r < rows_per_simd; r++) {\n" \
                     "      acc[r] += scale_local[r] * partial[r]"


def expect(text: str, needle: str, count: int, label: str) -> str:
    seen = text.count(needle)
    if seen != count:
        raise SystemExit(
            "e126_arms: %s matched %d times, expected %d" % (label, seen, count))
    return text


def in_wide(text: str, needle: str, replacement: str, label: str) -> str:
    """Replace inside the wide template only; the k loop text is not unique."""
    start, end = wide_fn_span(text)
    region = expect(text[start:end], needle, 1, label)
    return text[:start] + region.replace(needle, replacement) + text[end:]


def drop_tree(text: str) -> str:
    """Remove the chunk-sum add tree from both nibble branches."""
    expect(text, TREE_DIRECT, 1, "direct sums tree")
    expect(text, TREE_INDIRECT, 1, "indirect sums tree")
    text = text.replace(TREE_DIRECT, "").replace(TREE_INDIRECT, "")
    return text.replace(
        "          const float xsum = load_vector<T, float, 4, 4>(xm, xc);\n",
        "          load_vector<T, float, 4, 4>(xm, xc);\n")


def gate_off(text: str) -> str:
    expect(text, GATE, 1, "SHARE_SUMS gate")
    return text.replace(GATE, GATE_OFF)


def arm_source(base: str, arm: str) -> str:
    if arm == "share_on":
        text = base
    elif arm == "share_off":
        text = gate_off(base)
    elif arm == "n_sums_free":
        text = expect(drop_tree(gate_off(base)), CONSUMER, 1, "consumer")
        text = text.replace(CONSUMER, CONSUMER_FREE)
    elif arm == "n_nosums_e123":
        text = expect(drop_tree(gate_off(base)), CONSUMER, 1, "consumer")
        text = text.replace(CONSUMER, CONSUMER_E123)
    elif arm == "n_sums_loaded":
        text = drop_tree(gate_off(base))
        expect(text, CONSUMER_LOOP_OPEN, 1, "consumer loop")
        text = text.replace(CONSUMER_LOOP_OPEN, LOAD_SUMS + CONSUMER_LOOP_OPEN)
        text = in_wide(text, K_LOOP_OPEN, SEED_SUMS + K_LOOP_OPEN,
                       "wide k loop")
    else:
        raise SystemExit("e126_arms: unknown arm %s" % arm)
    slabs = BUFFER_SLABS.get(arm, 1)
    if slabs != 1:
        expect(text, ENTRY_BUFFER, 1, "entry exchange buffer")
        text = text.replace(ENTRY_BUFFER, "  threadgroup float sums_xchg[%d"
                            " * %d * 32];\n" % (slabs, max(WIDTHS)))
    return text + "".join(
        ARM_KERNEL % {"na": na, "slots": na * slabs} for na in WIDTHS)


def emit(outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    base = widen_asserts(emit_base(outdir / "base_raw.metal"))
    expect(base, GATE, 1, "shipped SHARE_SUMS gate")
    (outdir / "base_lone.metal").write_text(base)
    seen: dict[str, str] = {}
    for arm in ARMS:
        text = arm_source(base, arm)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        if digest in seen:
            raise SystemExit(
                "e126_arms: %s and %s are byte-identical" % (arm, seen[digest]))
        seen[digest] = arm
        (outdir / ("arm_%s.metal" % arm)).write_text(text)
        print("%-14s %8d bytes  sha=%s  exact=%s"
              % (arm, len(text), digest, arm in EXACT_ARMS))
    print("\n--arms %s" % ",".join(
        a if a in EXACT_ARMS else a + ":diag" for a in ARMS))


def census(directory: pathlib.Path, out: pathlib.Path | None) -> int:
    rows: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for arm in ARMS:
            source = directory / ("arm_%s.metal" % arm)
            air_dir = workdir / ("air_" + arm)
            air_dir.mkdir(parents=True, exist_ok=True)
            row: dict = {"air": air_stats(source, air_dir)}
            lib = build_metallib(source.read_text(), workdir / arm)
            for arch in (LOCAL_ARCH, RANKED_ARCH):
                for kernel, record in translate(lib, arch, workdir / arm).items():
                    hit = KERNEL_RE.search(kernel)
                    key = hit.group(1) if hit else (
                        "entry" if ENTRY_RE.search(kernel) else None)
                    if key is None:
                        continue
                    row.setdefault(arch, {})[key] = {
                        "registers": record.get("registers"),
                        "spill_bytes": record.get("spill_bytes", 0),
                        "text_bytes": record.get("text_bytes"),
                        "text_sha8": record.get("text_sha8"),
                    }
            rows[arm] = row
            print("censused %s" % arm)

    print("\nAIR per isolated width: fadd / fmul / tg ld+st / barriers / lines")
    for arm in ARMS:
        cells = []
        for na in WIDTHS:
            cell = rows[arm]["air"].get(str(na), {})
            cells.append("NA%d=%s/%s+%s/%s/%s" % (
                na, cell.get("fadd", "?"),
                cell.get("threadgroup_loads", "?"),
                cell.get("threadgroup_stores", "?"),
                cell.get("barriers", "?"), cell.get("air_lines", "?")))
        print("  %-14s %s" % (arm, "  ".join(cells)))

    for arch in (LOCAL_ARCH, RANKED_ARCH):
        print("\n%s isolated widths: registers / spill / text bytes" % arch)
        for arm in ARMS:
            cells = []
            for na in WIDTHS:
                value = rows[arm].get(arch, {}).get(str(na))
                if value is None:
                    cells.append("NA%d=?" % na)
                    continue
                spill = value["spill_bytes"] or 0
                cells.append("NA%d=%s%s/%s" % (
                    na, value["registers"], "s%d" % spill if spill else "",
                    value["text_bytes"]))
            print("  %-14s %s" % (arm, "  ".join(cells)))

    print("\nShipped entry point affine_qmv_fast, every width inlined (Rule 56)")
    for arm in ARMS:
        cells = []
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            value = rows[arm].get(arch, {}).get("entry")
            if value is None:
                cells.append("%s=?" % arch)
                continue
            spill = value["spill_bytes"] or 0
            cells.append("%s R=%s%s text=%s sg=%d" % (
                arch, value["registers"], "s%d" % spill if spill else "",
                value["text_bytes"], simdgroups(arch, value["registers"] or 96)))
        print("  %-14s %s" % (arm, "  ".join(cells)))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"widths": list(WIDTHS), "arms": rows,
             "register_file_bytes": {k: v for k, v in REGISTER_FILE.items()},
             "threadgroup_bytes": TG_MEMORY_BYTES,
             "exact_arms": list(EXACT_ARMS)}, indent=2) + "\n")
        print("\nwrote %s" % out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    if args.emit is not None:
        emit(args.emit)
    if args.census is not None:
        return census(args.census, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
