#!/usr/bin/env python3
"""Assemble ONE self-contained source string holding the three E64 rung-0b arms.

The scored worker does not compile the readable `kernels/*.metal`; for the
`quantized` family it concatenates the checked-in `mlx-generated/*.cpp`
preambles and hands the string to `newLibrary` with no include path
(jit_kernels.cpp get_quantized_kernel, device.cpp build_library_from_source).
research/jit_string_compile.py already reproduces that concatenation, so this
reuses it and appends the generated arms plus one `[[kernel]]` entry point per
arm.

One source string for all three arms means the arms cannot differ by preamble
text or compiler options: the only difference is the arm body itself. Each arm
is still its own entry point, so each gets its own register allocation, which is
the isolation the ladder measurement needs.

  python3 research/e64_emit_arms.py --na 5 --out /tmp/e64/arms.metal
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import PREAMBLES, preamble  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
ARMS_HEADER = REPO / "research/generated/e64_wide_arms.h"
ARMS = {
    "plain": "qmv_fast_crossrow_affine4_g64_wide_e64plain",
    "forced": "qmv_fast_crossrow_affine4_g64_wide_e64forced",
    "ballast": "qmv_fast_crossrow_affine4_g64_wide_e64ballast",
}

ENTRY = """
[[kernel]] void e64_cell_{arm}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {{
  const int first_m = int(tid.x) * {na};
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  {symbol}<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid);
}}
"""


# The scored `affine_qmv_fast` is ONE [[kernel]] and every verify width is a
# `case` of a runtime `switch (ntg.x)` inside it, so all widths share one
# register allocation and one occupancy. Every isolated NA arm above is its own
# entry point and gets its own allocation. This arm reproduces the shipped
# structure so the two can be compared: if a step between two widths survives
# here, it is a property of the width; if it does not, it is a property of
# compiling that width alone.
MERGED_ENTRY = """
[[kernel]] void e64_cell_merged(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& na_runtime [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {{
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  switch (na_runtime) {{
{cases}  }}
}}
"""

MERGED_CASE = """    case {na}: {{
      qmv_fast_crossrow_affine4_g64_wide_e64plain<bfloat16_t, {na}, true>(
          w, scales, biases, x, y, in_vec_size, out_vec_size,
          int(tid.x) * {na}, out_row, simd_lid);
      return;
    }}
"""


def assemble(na: int, merged_widths: list[int]) -> str:
    parts = [preamble(stem, None) for stem in PREAMBLES]
    body = ARMS_HEADER.read_text()
    parts.append(body.replace("#pragma once", ""))
    for arm, symbol in ARMS.items():
        parts.append(ENTRY.format(arm=arm, symbol=symbol, na=na))
    if merged_widths:
        cases = "".join(MERGED_CASE.format(na=width) for width in merged_widths)
        parts.append(MERGED_ENTRY.format(cases=cases))
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--na", type=int, default=5)
    parser.add_argument("--merged-widths", type=int, nargs="*", default=[])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = assemble(args.na, args.merged_widths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    print(f"{args.out} {len(source)} bytes na={args.na} "
          f"source_sha256={hashlib.sha256(source.encode()).hexdigest()} "
          f"arms_header_sha256="
          f"{hashlib.sha256(ARMS_HEADER.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
