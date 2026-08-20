#!/usr/bin/env python3
"""Assemble ONE self-contained source string holding every E69 rung-1 arm.

The scored worker does not compile the readable `kernels/*.metal`; for the
`quantized` family it concatenates the checked-in `mlx-generated/*.cpp`
preambles and hands the string to `newLibrary` with no include path
(jit_kernels.cpp get_quantized_kernel, device.cpp build_library_from_source).
research/jit_string_compile.py already reproduces that concatenation, so this
reuses it and appends the generated arms plus one `[[kernel]]` entry point per
arm.

One source string for every arm means the arms cannot differ by preamble text
or compiler options: the only difference is the arm body. Each arm is still its
own entry point, so each gets its own register allocation, which is the
isolation the cell measurement needs.

The entry points mirror research/e69_wide_probe.metal exactly. The host grid is
FROZEN at `grid_dims(M, ceil(N/8), B)` with `group_dims(32, 2, 1)`
(backend/metal/quantized.cpp:253-254, which is NOT an editable path), so the
rows_per_simd = 8 arms fold two adjacent 8-row tiles into one threadgroup
instead of asking for a different grid.

  python3 research/e69_emit_arms.py --na 5 --out /tmp/e69/arms.metal
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import PREAMBLES, preamble  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
ARMS_HEADER = REPO / "research/generated/e69_wide_arms.h"

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

# rows_per_simd = 4: the shipped geometry.
ENTRY_R4 = """
[[kernel]] void e69_cell_{arm}(
{args}) {{
  const int first_m = int(tid.x) * {na};
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  {symbol}<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid);
}}
"""

# Arm B stages x in threadgroup memory once per threadgroup, so it needs the
# simdgroup index and the staging buffer that the shipped signature does not
# carry.
ENTRY_TG = """
[[kernel]] void e69_cell_{arm}(
{args}) {{
  threadgroup bfloat16_t xs[{na} * 512];
  const int first_m = int(tid.x) * {na};
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  {symbol}<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid, simd_gid, xs);
}}
"""

# rows_per_simd = 8, tile folding: threadgroup 2t covers rows 16t..16t+15 and
# threadgroup 2t+1 returns. Needs out_vec_size % 16 == 0, which every scored
# affine-4 projection satisfies.
ENTRY_R8 = """
[[kernel]] void e69_cell_{arm}(
{args}) {{
  if ((tid.y & 1u) != 0u) {{
    return;
  }}
  const int first_m = int(tid.x) * {na};
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 8;
  {symbol}<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid);
}}
"""

# rows_per_simd = 8, idle half: every threadgroup keeps its own 8-row tile and
# the second simdgroup returns. Same simdgroup work as `rows8` spread over twice
# the threadgroups, so the contrast separates occupancy from dispatch count.
ENTRY_R8_IDLE = """
[[kernel]] void e69_cell_{arm}(
{args}) {{
  if (simd_gid != 0u) {{
    return;
  }}
  const int first_m = int(tid.x) * {na};
  const int out_row = int(tid.y) * 8;
  {symbol}<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid);
}}
"""

ARMS = {
    "plain": ("qmv_fast_crossrow_affine4_g64_wide_e69plain", ENTRY_R4),
    "wvec": ("qmv_fast_crossrow_affine4_g64_wide_e69wvec", ENTRY_R4),
    "xvec": ("qmv_fast_crossrow_affine4_g64_wide_e69xvec", ENTRY_R4),
    "wxvec": ("qmv_fast_crossrow_affine4_g64_wide_e69wxvec", ENTRY_R4),
    "tgx": ("qmv_fast_crossrow_affine4_g64_wide_e69tgx", ENTRY_TG),
    "rows8": ("qmv_fast_crossrow_affine4_g64_wide_e69rows8", ENTRY_R8),
    "rows8wxvec": ("qmv_fast_crossrow_affine4_g64_wide_e69rows8wxvec", ENTRY_R8),
    "rows8idle": ("qmv_fast_crossrow_affine4_g64_wide_e69rows8", ENTRY_R8_IDLE),
}


def assemble(na: int) -> str:
    parts = [preamble(stem, None) for stem in PREAMBLES]
    parts.append(ARMS_HEADER.read_text().replace("#pragma once", ""))
    for arm, (symbol, template) in ARMS.items():
        parts.append(template.format(arm=arm, symbol=symbol, na=na,
                                     args=PROBE_ARGS))
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--na", type=int, default=5)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = assemble(args.na)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    print(f"{args.out} {len(source)} bytes na={args.na} "
          f"arms={','.join(ARMS)} "
          f"source_sha256={hashlib.sha256(source.encode()).hexdigest()} "
          f"arms_header_sha256="
          f"{hashlib.sha256(ARMS_HEADER.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
