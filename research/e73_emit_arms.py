#!/usr/bin/env python3
"""Assemble ONE self-contained source string holding every legal `(M, IPG)` arm.

The scored worker does not compile the readable `kernels/*.metal`; for the
`quantized` family it concatenates the checked-in `mlx-generated/*.cpp`
preambles and hands the string to `newLibrary` with no include path.
`research/jit_string_compile.py` already reproduces that concatenation, so this
reuses it and appends one `[[kernel]]` entry point per legal partition.

One source string for all 19 arms means no arm can differ by preamble text or
compiler options. Each arm is still its own entry point, so each gets its own
register allocation, which is the isolation the cell measurement needs.

Every arm calls the SHIPPED wrapper `qmv_fast_crossrow_affine4_g64_m`
unchanged, with `DIRECT_NIBBLES = true` exactly as the scored switch does.

  python3 research/e73_emit_arms.py --out /tmp/e73/arms.metal
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import PREAMBLES, preamble  # noqa: E402
from e73_pairs import name, pairs  # noqa: E402

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

ENTRY = """
[[kernel]] void e73_cell_{arm}(
{args}) {{
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, {m}, {ipg}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}}
"""


def assemble() -> str:
    parts = [preamble(stem, None) for stem in PREAMBLES]
    for m, ipg in pairs():
        parts.append(ENTRY.format(arm=name(m, ipg), m=m, ipg=ipg,
                                  args=PROBE_ARGS))
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = assemble()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    print(f"{args.out} {len(source)} bytes arms={len(pairs())} "
          f"source_sha256={hashlib.sha256(source.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
