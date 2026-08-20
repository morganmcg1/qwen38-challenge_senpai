#!/usr/bin/env python3
"""Assemble ONE self-contained source string holding every E72 rung-2 arm.

Same construction as research/e69_emit_arms.py: the scored worker concatenates
the checked-in `mlx-generated/*.cpp` preambles and hands the string to
`newLibrary` with no include path, so the arms are appended to that exact
concatenation. One source string for every arm means the arms cannot differ by
preamble text or compiler options; each arm is still its own entry point, so
each gets its own register allocation.

  python3 research/e72_emit_arms.py --na 6 --out /tmp/e72/arms.metal
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import PREAMBLES, preamble  # noqa: E402
from e72_wide_gen import ARMS as GEN_ARMS, BASE_SYMBOL  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
ARMS_HEADER = REPO / "research/generated/e72_wide_arms.h"

PROBE_ARGS = """    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]"""

# Every E72 arm keeps the shipped geometry: rows_per_simd = 4 on the frozen host
# grid `grid_dims(M, ceil(N/8), B)` with `group_dims(32, 2, 1)`.
ENTRY = """
[[kernel]] void e72_cell_{arm}(
{args}) {{
  const int first_m = int(tid.x) * {na};
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  {symbol}<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid);
}}
"""

# Derived from the generator so the entry points can never drift from the arms
# actually emitted into research/generated/e72_wide_arms.h.
ARMS = {tag.removeprefix("e72"): f"{BASE_SYMBOL}_{tag}" for tag in GEN_ARMS}


def assemble(na: int) -> str:
    parts = [preamble(stem, None) for stem in PREAMBLES]
    parts.append(ARMS_HEADER.read_text().replace("#pragma once", ""))
    for arm, symbol in ARMS.items():
        parts.append(ENTRY.format(arm=arm, symbol=symbol, na=na,
                                  args=PROBE_ARGS))
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--na", type=int, default=6)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = assemble(args.na)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    print(f"wrote {args.out} na={args.na} bytes={len(source)} "
          f"sha256={hashlib.sha256(source.encode()).hexdigest()[:16]} "
          f"arms={len(ARMS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
