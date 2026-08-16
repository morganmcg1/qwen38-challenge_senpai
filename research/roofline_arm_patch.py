#!/usr/bin/env python3
"""Apply one measurement-only arm to `qmv_fast_crossrow_affine4_g64_wide`.

The roofline question is whether the crossrow verify kernel is limited by
memory bandwidth or by ALU throughput at M >= 4. Two arms separate them:

  arm1  cut arithmetic ~4x, hold every load.  ALU-bound => time falls.
  arm2  cut unique weight bytes ~4x, hold every arithmetic op.
        Bandwidth-bound => time falls.

Both arms compute WRONG numerics on purpose. They exist to be timed and must
never reach the product path.

`arm2-naive` is the literal "read row `out_row` from all four slots" form. It is
built for AIR inspection only: it lets the compiler prove the four rows carry
identical values, so it collapses the arithmetic too and answers both arms at
once, i.e. answers neither. `arm2` defeats that with a runtime-opaque zero.

The readable header and its runtime-effective `mlx-generated` twin are edited
together, so `research/twin_audit.py` stays clean.

  research/roofline_arm_patch.py arm1 [--check]
  git checkout -- Vendor/mlx-swift/Source/Cmlx   # revert
"""

from __future__ import annotations

import argparse
import pathlib
import sys

TWINS = [
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
]

WIDE_DECL = "METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide("

FULL_QDOT = """      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * (packed[r][i] & 0x00f0) +
                       a2 * (packed[r][i] & 0x0f00) +
                       a3 * (packed[r][i] & 0xf000));
      }
"""

ONE_TERM_QDOT = """      for (int r = 0; r < rows_per_simd; r++) {
        // Measurement-only arm 1: three of the four nibble terms are dropped.
        // `packed[r][i]` stays live through the surviving term and load_vector
        // still consumes all four x values through its returned sum, so weight
        // and activation traffic are untouched while the FMA count falls ~4x.
        partial[r] += a0 * (packed[r][i] & 0x000f);
      }
"""

K_LOOP = "  for (int k = 0; k < in_vec_size; k += block_size) {\n"

K_LOOP_WITH_SPAN = """  // Measurement-only arm 2: a runtime zero the compiler cannot fold. The four
  // row addresses stay formally distinct, so no load or value CSE fires and the
  // arithmetic is untouched, while at runtime all four resolve to one tile and
  // unique weight traffic falls ~4x.
  const int row_span = in_vec_size >> 30;
  for (int k = 0; k < in_vec_size; k += block_size) {
"""

ROW_STRIDED = "      const int row = out_row + r;\n"
ROW_SPANNED = "      const int row = out_row + r * row_span;\n"
ROW_COLLAPSED = "      const int row = out_row;\n"

FMA_QDOT = """      for (int r = 0; r < rows_per_simd; r++) {
        const int q = packed[r][i];
        VF p = partial[r];
        p = metal::fma(a0, VF(float(q & 0x000f)), p);
        p = metal::fma(a1, VF(float(q & 0x00f0)), p);
        p = metal::fma(a2, VF(float(q & 0x0f00)), p);
        p = metal::fma(a3, VF(float(q & 0xf000)), p);
        partial[r] = p;
      }
"""

# Every nibble product is exact in fp32 (a bf16 activation scaled by a power of
# two carries <= 8 mantissa bits; the masked nibble carries <= 4), so an FMA
# cannot change a product. Only the addition order can change a bit. This form
# keeps the association the unfused source compiles to --
# `((a0*q0 + a1*q1) + partial) + a2*q2 + a3*q3` -- so it is bit-identical while
# still trading three multiply/add pairs for three FMAs.
FMA_QDOT_ORDERED = """      for (int r = 0; r < rows_per_simd; r++) {
        const int q = packed[r][i];
        VF p = metal::fma(a1, VF(float(q & 0x00f0)), a0 * VF(float(q & 0x000f)));
        p += partial[r];
        p = metal::fma(a2, VF(float(q & 0x0f00)), p);
        partial[r] = metal::fma(a3, VF(float(q & 0xf000)), p);
      }
"""

ARMS = {
    "arm1": [(FULL_QDOT, ONE_TERM_QDOT)],
    "fma": [(FULL_QDOT, FMA_QDOT)],
    "fma-ordered": [(FULL_QDOT, FMA_QDOT_ORDERED)],
    "arm2": [(K_LOOP, K_LOOP_WITH_SPAN), (ROW_STRIDED, ROW_SPANNED)],
    "arm2-naive": [(ROW_STRIDED, ROW_COLLAPSED)],
}


def wide_region(text: str) -> tuple[int, int]:
    start = text.index(WIDE_DECL)
    return start, text.index("\n}\n", start) + 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--check", action="store_true", help="verify only, do not write")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    for rel in TWINS:
        path = root / rel
        text = path.read_text()
        start, end = wide_region(text)
        region = text[start:end]
        for old, new in ARMS[args.arm]:
            if region.count(old) != 1:
                print(f"{rel}: anchor not unique in the _wide body "
                      f"({region.count(old)} matches); source drifted", file=sys.stderr)
                return 1
            region = region.replace(old, new)
        if not args.check:
            path.write_text(text[:start] + region + text[end:])
        print(f"{rel}: {args.arm} {'ok' if args.check else 'applied'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
