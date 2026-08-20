#!/usr/bin/env python3
"""Derive the E69 arms FROM the shipped wide crossrow QMV kernel.

E69 tests one claim: `qmv_fast_crossrow_affine4_g64_wide` is bound by the
traffic and the load instructions of its `x` operand, not by weight bytes and
not by arithmetic. Every arm below is the shipped body plus a closed list of
exact substitutions, so no arm can drift away from what ships.

The AIR census (research/e69_air_census.py) shows the shipped k-block issues,
per lane at NA=6: 16 scalar `i16` weight loads, 8 scalar `bfloat` scale and bias
loads, and 4 * 4 * NA = 96 scalar `bfloat` x loads. x is 80 % of the load
instructions and every one of them moves 2 bytes. The compiler cannot widen
either operand itself, because `in_vec_size` is a runtime value and it can only
assume 2-byte alignment. The host gates this kernel on `K % 512 == 0`, so the
alignment does hold and the code can assert it.

  plain     the shipped body, renamed. Control.
  wvec      arm A. One 8-byte `ushort4` load replaces the four scalar
            `uint16_t` loads of the packed weights. 16 -> 4 loads per lane per
            k-block. Bytes unchanged.
  xvec      arm A on the larger operand. One 8-byte `vec<T, 4>` load replaces
            the four scalar T loads of x. 96 -> 24 loads per lane per k-block at
            NA=6. Bytes unchanged.
  wxvec     wvec + xvec: 120 -> 36 loads per lane per k-block at NA=6.
  tgx       arm B. The two simdgroups of a threadgroup read the SAME
            `NA * block_size` x values for a k block. Stage them once in
            threadgroup memory, and keep the scalar reads, so device x BYTES
            halve while the read issue count does not change. Contrasted with
            `xvec` this separates traffic from issue slots.
  rows8     arm C. `rows_per_simd` 4 -> 8, so the x term per output row halves.
            The frozen host grid cannot be changed, so the caller folds two
            adjacent 8-row tiles into one threadgroup and returns on odd
            `tid.y`. Tests the payoff, and pays in registers.
  rows8wxvec arm C + wxvec.
  rows8idle arm C with the alternative caller geometry: every threadgroup keeps
            its own 8-row tile and `simd_gid == 1` returns. Separates
            "fewer threadgroups" from "half the threads idle". It reuses the
            rows8 body, so it is a probe entry point, not an arm here.

Bit-identity holds for every arm by construction. Output row `r` sums over the
same k in the same order and finishes with the same 32-lane `simd_sum`, and
there is no cross-row reduction in this function, so `rows_per_simd` cannot
change any emitted value. `wvec` and `xvec` move the same bytes into the same
slots. `tgx` routes the same T values to the same FMAs in the same order.

  python3 research/e69_wide_gen.py            # write the generated header
  python3 research/e69_wide_gen.py --check    # verify it is still in sync
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

SHIPPED = pathlib.Path(
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
)
GENERATED = pathlib.Path("research/generated/e69_wide_arms.h")

TEMPLATE_LINE = "template <typename T, int NA, bool DIRECT_NIBBLES = false>"
SIGNATURE = "METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide("
NA_ASSERT = (
    '  static_assert(NA >= 2 && NA <= 6, "wide multi-row QMV supports NA in [2, 6]");'
)
PROBE_ASSERT = '  static_assert(NA >= 2 && NA <= 9, "probe-only NA bound");'

ROWS_DECL = "  constexpr int rows_per_simd = 4;"
ROWS8_DECL = """  // E69 arm C: each simdgroup owns 8 output rows instead of 4. Per output row
  // the x term of the k-block cost, 32*NA/rows_per_simd bytes, halves; the
  // weight term is unchanged. Row r still sums over the same k in the same
  // order, and nothing in this function reduces across r, so every emitted
  // value is bit-identical. The caller must cover 16 rows per threadgroup.
  constexpr int rows_per_simd = 8;"""

WEIGHT_LOAD = """      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }"""
WEIGHT_LOAD_VEC = """      // E69 arm A: one 8-byte load in place of four 2-byte loads. `ws` is
      // 8-byte aligned because in_vec_size_w, k/2 and simd_lid*bytes_per_lane
      // are each multiples of 8, and the same bytes land in the same slots in
      // the same order, so the arm is bit-identical by construction.
      const ushort4 wv = *reinterpret_cast<const device ushort4*>(ws);
      packed[r][0] = wv.x;
      packed[r][1] = wv.y;
      packed[r][2] = wv.z;
      packed[r][3] = wv.w;"""

TAIL_PARAM = "    uint simd_lid) {"
TAIL_PARAM_TG = """    uint simd_lid,
    uint simd_gid,
    threadgroup T* xs) {"""

SUMS_ANCHOR = "    VF sums = VF(0.0f);"
STAGE_X = """    // E69 arm B: both simdgroups of this threadgroup read the SAME
    // NA * block_size x values for this k block. Stage them once, so the
    // device-side x traffic per threadgroup halves. The staged values are the
    // same T bit patterns and they reach the same FMAs in the same order.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int m = 0; m < NA; m++) {
      const device vec<T, 4>* xsrc = reinterpret_cast<const device vec<T, 4>*>(
          x + (first_m + m) * in_vec_size + k);
      threadgroup vec<T, 4>* xdst =
          reinterpret_cast<threadgroup vec<T, 4>*>(xs + m * block_size);
      for (int t = int(simd_gid) * SIMD_SIZE + int(simd_lid);
           t < block_size / 4; t += 2 * SIMD_SIZE) {
        xdst[t] = xsrc[t];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

X_SCALAR_READ = """        thread float xc[4];
        if (DIRECT_NIBBLES) {
          xc[0] = static_cast<float>(xm[0]);
          xc[1] = static_cast<float>(xm[1]);
          xc[2] = static_cast<float>(xm[2]);
          xc[3] = static_cast<float>(xm[3]);
          // Preserve the incumbent BF16 expression tree used for the affine
          // bias correction; only the qdot nibble extraction changes.
          sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        } else {"""
X_VECTOR_READ = """        thread float xc[4];
        if (DIRECT_NIBBLES) {
          // E69 arm X: one 8-byte load in place of four 2-byte loads. The
          // compiler cannot widen these itself because `in_vec_size` is a
          // runtime value, so it can only assume 2-byte alignment. The host
          // gates this kernel on `K % 512 == 0`, and every other term of this
          // address is a multiple of 4 elements, so the 8-byte alignment does
          // hold. The same T values reach the same expressions in the same
          // order, so the arm is bit-identical by construction.
          const vec<T, 4> xv = *reinterpret_cast<const device vec<T, 4>*>(xm);
          xc[0] = static_cast<float>(xv[0]);
          xc[1] = static_cast<float>(xv[1]);
          xc[2] = static_cast<float>(xv[2]);
          xc[3] = static_cast<float>(xv[3]);
          // Preserve the incumbent BF16 expression tree used for the affine
          // bias correction; only the qdot nibble extraction changes.
          sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
        } else {"""

X_READ = """        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        thread float xc[4];
        if (DIRECT_NIBBLES) {
          xc[0] = static_cast<float>(xm[0]);
          xc[1] = static_cast<float>(xm[1]);
          xc[2] = static_cast<float>(xm[2]);
          xc[3] = static_cast<float>(xm[3]);
          // Preserve the incumbent BF16 expression tree used for the affine
          // bias correction; only the qdot nibble extraction changes.
          sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        } else {"""
X_READ_TG = """        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        // The staged copy carries the same T values at the same offsets. The
        // device pointer above stays for the unscaled branch, which no scored
        // instantiation takes, and dies with it under DIRECT_NIBBLES.
        const threadgroup T* xt = xs + m * block_size +
            simd_lid * values_per_thread + 4 * i;
        thread float xc[4];
        if (DIRECT_NIBBLES) {
          xc[0] = static_cast<float>(xt[0]);
          xc[1] = static_cast<float>(xt[1]);
          xc[2] = static_cast<float>(xt[2]);
          xc[3] = static_cast<float>(xt[3]);
          // Preserve the incumbent BF16 expression tree used for the affine
          // bias correction; only the qdot nibble extraction changes.
          sums[m] += xt[0] + xt[1] + xt[2] + xt[3];
        } else {"""

HEADER = """// GENERATED by research/e69_wide_gen.py -- do not edit by hand.
//
// Body extracted verbatim from {src} lines {lo}-{hi}
// (extracted-body sha256 {digest}), then rewritten per arm by exact
// substitutions that must each match once.
//
// Research only. Nothing here is on the scored path. The arms exist so one
// question can be answered: is the wide crossrow QMV kernel bound by the
// traffic and the load instructions of its x operand?

#pragma once

"""


def rename(name: str) -> tuple[str, str]:
    return SIGNATURE, f"METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide_{name}("


WVEC = ("load the four packed weight halves as one 8-byte vector",
        WEIGHT_LOAD, WEIGHT_LOAD_VEC)
XVEC = ("load the four x values as one 8-byte vector",
        X_SCALAR_READ, X_VECTOR_READ)
ROWS8 = ("raise rows_per_simd from 4 to 8", ROWS_DECL, ROWS8_DECL)
TGX = [
    ("take the staging buffer and the simdgroup index", TAIL_PARAM, TAIL_PARAM_TG),
    ("stage one k block of x per threadgroup", SUMS_ANCHOR, STAGE_X + SUMS_ANCHOR),
    ("read x from the staged copy", X_READ, X_READ_TG),
]


def arms() -> dict[str, list[tuple[str, str, str]]]:
    """arm -> [(why, exact old text, exact new text)], each matching once."""
    common = lambda name: [  # noqa: E731
        ("rename so the arm cannot shadow the shipped symbol", *rename(name)),
        ("relax the NA bound inside the probe only", NA_ASSERT, PROBE_ASSERT),
    ]
    return {
        "e69plain": common("e69plain"),
        "e69wvec": common("e69wvec") + [WVEC],
        "e69xvec": common("e69xvec") + [XVEC],
        "e69wxvec": common("e69wxvec") + [WVEC, XVEC],
        "e69tgx": common("e69tgx") + TGX,
        "e69rows8": common("e69rows8") + [ROWS8],
        "e69rows8wxvec": common("e69rows8wxvec") + [ROWS8, WVEC, XVEC],
    }


def extract(text: str) -> tuple[str, int, int]:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(SIGNATURE):
            if lines[index - 1] != TEMPLATE_LINE:
                raise SystemExit(
                    f"{SHIPPED}: line {index}: expected {TEMPLATE_LINE!r}"
                )
            start = index - 1
            break
    if start is None:
        raise SystemExit(f"{SHIPPED}: no {SIGNATURE!r}")
    end = None
    for index in range(start, len(lines)):
        if lines[index] == "}":
            end = index
            break
    if end is None:
        raise SystemExit(f"{SHIPPED}: unterminated body from line {start + 1}")
    return "\n".join(lines[start:end + 1]) + "\n", start + 1, end + 1


def apply(body: str, rewrites: list[tuple[str, str, str]]) -> str:
    out = body
    for why, old, new in rewrites:
        if out.count(old) != 1:
            raise SystemExit(
                f"rewrite {why!r} matched {out.count(old)} times, expected 1"
            )
        out = out.replace(old, new)
    return out


def generate() -> str:
    body, lo, hi = extract(SHIPPED.read_text())
    digest = hashlib.sha256(body.encode()).hexdigest()
    parts = [HEADER.format(src=SHIPPED, lo=lo, hi=hi, digest=digest)]
    for name, rewrites in arms().items():
        parts.append(f"// --- arm {name} " + "-" * (58 - len(name)) + "\n//\n")
        for why, _, _ in rewrites:
            parts.append(f"//   {why}\n")
        parts.append("\n" + apply(body, rewrites) + "\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    text = generate()
    if args.check:
        if not GENERATED.exists() or GENERATED.read_text() != text:
            print(f"{GENERATED} is stale; rerun research/e69_wide_gen.py")
            return 1
        print(f"{GENERATED} is in sync with {SHIPPED}")
        return 0
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(text)
    print(f"{GENERATED} {len(text)} bytes, {len(arms())} arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
