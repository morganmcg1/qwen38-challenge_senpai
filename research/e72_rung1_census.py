#!/usr/bin/env python3
"""E72 rung 1: register and spill census of the scored QMV on two GPU generations.

`research/agx_crossarch.py` reaches the real AGX backend for a generation this
host does not have, so the ranked runner's `applegpu_g17s` can be measured here
against the local `applegpu_g16s`. Two questions are answered together:

  wide   `qmv_fast_crossrow_affine4_g64_wide` at NA = 2..6. Does the NA = 6
         spill that rung 2 attacks also exist on the ranked generation, and at
         which width does each generation start spilling?

  cells  `qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true>` at the
         twelve cells that separate our shipped group partition from the
         frontier's, so `t55`, `t6` and `E55` can be priced before a ranked
         slot is spent on them.

  arms   the same `<T, 6, 6>` cell rebuilt against each rung-2 arm body, which
         is the before-and-after register pair for the shipped instantiation
         rather than for the standalone probe. E38 read AIR `peak_live_regs`
         = 144 for na6 and inferred that it cannot fit 128 registers; these
         are device numbers from the real backend, for both generations.

This is a selection and provenance instrument. It never times anything.

  python3 research/e72_rung1_census.py --out research/e72-artifacts/rung1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import PREAMBLES, preamble  # noqa: E402
from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH,
    RANKED_ARCH,
    build_metallib,
    translate,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"

# The twelve cells from the assignment. Each M carries the group size we ship
# and the group size the frontier ships; they differ only at M = 5, 6 and 9.
CELLS = [(3, 3, 3), (4, 4, 4), (5, 5, 3), (6, 6, 3),
         (7, 4, 4), (8, 4, 4), (9, 5, 3)]
WIDE_WIDTHS = [2, 3, 4, 5, 6]

INCLUDES = """#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
"""

PREFIX = """
#define E72_ARGS                                                           \\
    const device uint32_t* w [[buffer(0)]],                                \\
    const device bfloat16_t* scales [[buffer(1)]],                         \\
    const device bfloat16_t* biases [[buffer(2)]],                         \\
    const device bfloat16_t* x [[buffer(3)]],                              \\
    device bfloat16_t* y [[buffer(4)]],                                    \\
    const constant int& in_vec_size [[buffer(5)]],                         \\
    const constant int& out_vec_size [[buffer(6)]],                        \\
    uint3 tid [[threadgroup_position_in_grid]],                            \\
    uint simd_gid [[simdgroup_index_in_threadgroup]],                      \\
    uint simd_lid [[thread_index_in_simdgroup]]

"""

# `_m` is the scored entry shape: it takes the same arguments the worker passes
# and derives `first_m` and `out_row` itself.
CELL_KERNEL = """[[kernel]] void cell_m{m}_ipg{ipg}(E72_ARGS) {{
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, {m}, {ipg}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}}

"""

WIDE_KERNEL = """[[kernel]] void wide_na{na}(E72_ARGS) {{
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      int(tid.x) * {na}, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);
}}

"""

# `qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true>` with the body swapped for a
# rung-2 arm. At M = IPG = 6 the wrapper's TAIL is 0, so the shipped wrapper
# reduces to exactly this guard plus one `_wide<T, 6, true>` call; `arm_plain`
# must therefore emit byte-identical machine code to `cell_m6_ipg6`, and the
# census checks that rather than assuming it.
ARM_CELL_KERNEL = """[[kernel]] void arm_{arm}(E72_ARGS) {{
  const int first_m = int(tid.x) * 6;
  if (first_m >= 6) {{
    return;
  }}
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  {symbol}<bfloat16_t, 6, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid);
}}

"""


def arm_symbols() -> list[tuple[str, str]]:
    """The arms exactly as `research/e72_wide_gen.py` emitted them.

    The X-macro in the generated header is the single source of truth for arm
    names, so the census cannot drift from the generator or from rung 2.
    """
    text = (REPO / "research/generated/e72_wide_arms.h").read_text()
    block = text[text.index("#define E72_FOR_EACH_ARM(X)"):]
    found = []
    for line in block.splitlines():
        line = line.strip().rstrip("\\").strip()
        if line.startswith("X("):
            arm, symbol = line[2:line.rindex(")")].split(",")
            found.append((arm.strip(), symbol.strip()))
    if not found:
        raise SystemExit("no arms found in E72_FOR_EACH_ARM")
    return found


def source(jit: bool) -> tuple[str, dict[str, dict]]:
    """Build the probe either from headers or from the worker's JIT string.

    The scored worker does not compile these headers. It concatenates the
    checked-in `mlx-generated/*.cpp` preambles and hands that string to
    `newLibrary` with no include path. A register census taken through the
    headers is therefore a census of a different translation unit, so `--jit`
    reproduces the exact text the worker compiles.
    """
    head = "".join(preamble(stem, None) for stem in PREAMBLES) if jit else INCLUDES
    arms = arm_symbols()
    head += '#include "%s"\n' % (REPO / "research/generated/e72_wide_arms.h")
    parts, labels = [head, PREFIX], {}
    for na in WIDE_WIDTHS:
        parts.append(WIDE_KERNEL.format(na=na))
        labels[f"wide_na{na}"] = {"kind": "wide", "na": na}
    seen = set()
    for m, ours, theirs in CELLS:
        for ipg, who in ((ours, "ours"), (theirs, "frontier")):
            name = f"cell_m{m}_ipg{ipg}"
            if name not in seen:
                parts.append(CELL_KERNEL.format(m=m, ipg=ipg))
                seen.add(name)
            labels.setdefault(name, {"kind": "cell", "m": m, "ipg": ipg,
                                     "role": []})
            labels[name]["role"].append(who)
    for arm, symbol in arms:
        parts.append(ARM_CELL_KERNEL.format(arm=arm, symbol=symbol))
        labels[f"arm_{arm}"] = {"kind": "arm", "arm": arm, "m": 6, "ipg": 6,
                                "symbol": symbol}
    return "".join(parts), labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--arch", nargs="+", default=[LOCAL_ARCH, RANKED_ARCH])
    parser.add_argument("--jit", action="store_true",
                        help="compile the worker's JIT string, not the headers")
    args = parser.parse_args()

    text, labels = source(args.jit)
    result = {"architectures": args.arch, "kernels": labels,
              "translation_unit": "jit_string" if args.jit else "headers",
              "census": {}}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = build_metallib(text, workdir,
                             include=None if args.jit else INCLUDE)
        for arch in args.arch:
            result["census"][arch] = translate(lib, arch, workdir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"{'kernel':<20}" + "".join(
        f"{arch.replace('applegpu_', ''):>30}" for arch in args.arch))
    print(f"{'':<20}" + "".join(f"{'regs':>7}{'spill':>7}{'bytes':>7}{'code':>9}"
                                for _ in args.arch))
    for name in labels:
        row = f"{name:<20}"
        for arch in args.arch:
            record = result["census"][arch][name]
            row += (f"{record['registers']:>7}{record['spill_bytes']:>7}"
                    f"{record['text_bytes']:>7}{record['text_sha8']:>9}")
        print(row)

    # A census is only readable if the kernels really are distinct programs.
    for arch in args.arch:
        digests = {}
        for name in labels:
            digests.setdefault(
                result["census"][arch][name]["text_sha8"], []).append(name)
        for digest, shared in sorted(digests.items()):
            if len(shared) > 1:
                print(f"NOTE {arch}: identical machine code {digest} for "
                      + ", ".join(shared))

    # The whole before-and-after pair rests on `arm_plain` being the shipped
    # `<T,6,6>` object. It is a hand-written copy of a wrapper this experiment
    # does not own, so prove the copy is faithful instead of trusting it.
    result["checks"] = {}
    for arch in args.arch:
        shipped = result["census"][arch]["cell_m6_ipg6"]
        replica = result["census"][arch]["arm_plain"]
        same = shipped["text_sha8"] == replica["text_sha8"]
        result["checks"][arch] = {
            "arm_plain_is_shipped_m6_ipg6": same,
            "shipped_text_sha8": shipped["text_sha8"],
            "replica_text_sha8": replica["text_sha8"],
        }
        print(f"CHECK {arch}: arm_plain == cell_m6_ipg6 "
              f"{'PASS' if same else 'FAIL'} "
              f"({shipped['text_sha8']} vs {replica['text_sha8']})")

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0 if all(c["arm_plain_is_shipped_m6_ipg6"]
                    for c in result["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
