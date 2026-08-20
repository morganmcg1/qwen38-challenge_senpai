#!/usr/bin/env python3
"""E76 rung 1: register and spill census of every arm on both GPU generations.

The question is a single number per arm: how many `applegpu_g17s` registers does
this restructuring of the one-group `_wide` body allocate, and does it still emit
a program at all without a spill frame. The shipped `<T,5,5>` cell allocates 98
and `<T,6,6>` allocates 111; the frontier's three-input cells allocate 90. An arm
that reaches 90 or fewer keeps our single weight stream at the frontier's ranked
occupancy.

The scored worker does not compile the vendored headers. It concatenates the
checked-in `mlx-generated/*.cpp` preambles and hands that string to `newLibrary`
with no include path, so `--jit` (the default) censuses the exact translation
unit the ranked runner compiles. `--headers` is available as a cross-check.

Every arm is emitted into ONE source string, so no arm can differ from another
by preamble text or compiler options, and each arm is its own entry point, so
each gets its own register allocation. `shipped_na{N}` calls the unmodified
kernel from the same string; the census asserts that `plain` reproduces it
byte-for-byte, because a control that is not the shipped object makes every
other row unreadable.

  python3 research/e76_rung1_census.py --out research/e76-artifacts/rung1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import PREAMBLES, preamble  # noqa: E402
from e76_wide_gen import ARMS, BASE_SYMBOL, GENERATED  # noqa: E402
from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH,
    RANKED_ARCH,
    build_metallib,
    translate,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"

# The frontier's three-input cells allocate this many g17s registers. An arm at
# or below it matches their ranked occupancy while keeping one weight stream.
TARGET_G17S_REGISTERS = 90

INCLUDES = """#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
"""

PREFIX = """
#define E76_ARGS                                                           \\
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

# The frozen host geometry: `grid_dims(M, ceil(N/8), B)` with
# `group_dims(32, 2, 1)`, so a simdgroup owns four output rows whatever the
# arm's internal row block is.
ENTRY = """[[kernel]] void {name}(E76_ARGS) {{
  {symbol}<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      int(tid.x) * {na}, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);
}}

"""

SHIPPED_SYMBOL = "qmv_fast_crossrow_affine4_g64_wide"


def source(jit: bool, widths: list[int]) -> tuple[str, dict[str, dict]]:
    head = "".join(preamble(stem, None) for stem in PREAMBLES) if jit else INCLUDES
    head += (REPO / GENERATED).read_text().replace("#pragma once", "")
    parts, labels = [head, PREFIX], {}
    for na in widths:
        name = f"shipped_na{na}"
        parts.append(ENTRY.format(name=name, symbol=SHIPPED_SYMBOL, na=na))
        labels[name] = {"kind": "shipped", "arm": "shipped", "na": na,
                        "rows_per_simd": 4}
        for arm, rps, unroll, rewrites in ARMS:
            name = f"e76_{arm}_na{na}"
            parts.append(ENTRY.format(
                name=name, symbol=f"{BASE_SYMBOL}_{arm}", na=na))
            labels[name] = {"kind": "arm", "arm": arm, "na": na,
                            "rows_per_simd": rps,
                            "row_block_loop_unrolled": unroll,
                            "body_rewrites": len(rewrites)}
    return "".join(parts), labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--na", nargs="+", type=int, default=[5, 6])
    parser.add_argument("--arch", nargs="+", default=[LOCAL_ARCH, RANKED_ARCH])
    parser.add_argument("--headers", action="store_true",
                        help="compile the vendored headers, not the JIT string")
    args = parser.parse_args()

    text, labels = source(not args.headers, args.na)
    result = {"architectures": args.arch, "kernels": labels,
              "widths": args.na,
              "target_g17s_registers": TARGET_G17S_REGISTERS,
              "translation_unit": "headers" if args.headers else "jit_string",
              "census": {}}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = build_metallib(text, workdir,
                             include=INCLUDE if args.headers else None)
        for arch in args.arch:
            result["census"][arch] = translate(lib, arch, workdir)

    header = f"{'kernel':<22}" + "".join(
        f"{arch.replace('applegpu_', ''):>30}" for arch in args.arch)
    print(header)
    print(f"{'':<22}" + "".join(f"{'regs':>7}{'spill':>7}{'bytes':>7}{'code':>9}"
                                for _ in args.arch))
    for name in labels:
        row = f"{name:<22}"
        for arch in args.arch:
            record = result["census"][arch][name]
            row += (f"{record['registers']:>7}{record['spill_bytes']:>7}"
                    f"{record['text_bytes']:>7}{record['text_sha8']:>9}")
        print(row)

    # A control that is not the shipped object makes every other row unreadable.
    result["checks"] = {}
    ok = True
    for arch in args.arch:
        for na in args.na:
            shipped = result["census"][arch][f"shipped_na{na}"]
            replica = result["census"][arch][f"e76_plain_na{na}"]
            same = shipped["text_sha8"] == replica["text_sha8"]
            ok = ok and same
            result["checks"][f"{arch}_na{na}"] = {
                "plain_is_shipped_wide": same,
                "shipped_text_sha8": shipped["text_sha8"],
                "plain_text_sha8": replica["text_sha8"],
            }
            print(f"CHECK {arch} na={na}: plain == shipped "
                  f"{'PASS' if same else 'FAIL'} "
                  f"({shipped['text_sha8']} vs {replica['text_sha8']})")

    # The one question this rung exists to answer.
    reached = []
    for na in args.na:
        for arm, _, _, _ in ARMS:
            record = result["census"][RANKED_ARCH].get(f"e76_{arm}_na{na}")
            if record and record["registers"] <= TARGET_G17S_REGISTERS \
                    and not record["spill_bytes"]:
                reached.append({"arm": arm, "na": na,
                                "registers": record["registers"]})
    result["reached_target"] = reached
    if reached:
        for hit in reached:
            print(f"TARGET na={hit['na']} arm={hit['arm']} "
                  f"g17s registers={hit['registers']} "
                  f"<= {TARGET_G17S_REGISTERS}")
    else:
        print(f"TARGET: no arm reaches {TARGET_G17S_REGISTERS} g17s registers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
