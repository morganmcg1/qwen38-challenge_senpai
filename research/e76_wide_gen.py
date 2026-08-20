#!/usr/bin/env python3
"""Derive the E76 register-search arms FROM the shipped wide crossrow QMV kernel.

E72 made the ranked generation's register allocator readable on this host, so a
kernel restructuring can be priced before any GPU time is spent. E76 asks one
question with that instrument: can the one-group `_wide` body at NA = 5 and
NA = 6 be restructured to 90 or fewer `applegpu_g17s` registers while emitting
the same bits?

The live set at the peak is thirteen `vec<float, NA>` values: `acc[4]`,
`partial[4]`, `sums`, and `a0..a3`. At NA = 6 that is 78 floats, which the
`agx_crossarch.py wall` curve prices near the observed 111 g17s registers.
`a0..a3` cannot leave without changing the addition order inside `partial[r]`,
so the only large lever is the row block: `acc` and `partial` are the two arrays
indexed by `r`, and halving `rows_per_simd` removes two `VF` values from each.

The arms are the cross product of two independent levers.

Row block, which sets how many `VF` values `acc` and `partial` hold:

  (none)    `rows_per_simd` = 4, the shipped value.
  rps2      4 -> 2 behind the coverage-preserving row-block wrapper: two
            sequential blocks of two rows per simdgroup. Each weight row is
            still read exactly once; the x side is read once per block.
  rps1      the same at one row per block, four blocks.

Operand staging, which sets how long the scalar operands stay live across the
peak:

  (none)    the shipped staging: `packed[4][4]`, `scale_local[4]` and
            `bias_local[4]` are all loaded at the top of the k-block.
  lazysb    `scale_local` and `bias_local` are used only in the k-block
            epilogue, so this loads them there.
  lazyw     each `packed[r][i]` is used at exactly one `i`, so this loads it at
            that use site instead of staging sixteen `uint16_t`.
  lazy      both.

`plain` is the shipped body, renamed, and the census asserts it emits the same
machine code as the shipped instantiation. `rps2nu` and `rps1nu` repeat `rps2`
and `rps1` with the row-block loop held rolled: the loop has a compile-time trip
count, so an unrolled block loop lets the allocator interleave the blocks and
put the pressure straight back, and these two separate that outcome from a real
reduction.

Every rewrite is a storage or scheduling change. No arm changes which values are
computed, the order of any floating-point accumulation, or the reduction. That
is the claim, not the proof: E72 saw `mfull` pass an arithmetic-level gate and
still change 30720 output rows, so every arm here is also checked bit-for-bit on
the device across all seven scored shapes.

  python3 research/e76_wide_gen.py            # write the generated header
  python3 research/e76_wide_gen.py --check    # verify it is still in sync
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

SHIPPED = pathlib.Path(
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
)
GENERATED = pathlib.Path("research/generated/e76_wide_arms.h")

TEMPLATE_LINE = "template <typename T, int NA, bool DIRECT_NIBBLES = false>"
SIGNATURE = "METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide("
BASE_SYMBOL = "qmv_fast_crossrow_affine4_g64_wide_e76"

# ---------------------------------------------------------------------------
# Anchors. Every one must appear exactly once in the extracted body, so the
# generator fails loudly if the shipped kernel moves under the search.
# ---------------------------------------------------------------------------

RPS_DECL = "  constexpr int rows_per_simd = 4;"

STAGE = """    thread uint16_t packed[rows_per_simd][4];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * bytes_per_lane);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }"""

STAGE_WEIGHTS_ONLY = """    thread uint16_t packed[rows_per_simd][4];
    for (int r = 0; r < rows_per_simd; r++) {
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) +
          (out_row + r) * in_vec_size_w + k / 2 + simd_lid * bytes_per_lane);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
    }"""

STAGE_SCALES_ONLY = """    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int group_index =
          (out_row + r) * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }"""

STAGE_NONE = "    // E76 lazywsb: nothing is staged; every operand loads at its use site."

USE = """      for (int r = 0; r < rows_per_simd; r++) {
        if (DIRECT_NIBBLES) {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * ((packed[r][i] >> 4) & 0x000f) +
                         a2 * ((packed[r][i] >> 8) & 0x000f) +
                         a3 * ((packed[r][i] >> 12) & 0x000f));
        } else {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * (packed[r][i] & 0x00f0) +
                         a2 * (packed[r][i] & 0x0f00) +
                         a3 * (packed[r][i] & 0xf000));
        }
      }"""

# The same expression tree over the same uint16_t, loaded at the use site rather
# than read out of a staged array.
USE_LAZY = """      for (int r = 0; r < rows_per_simd; r++) {
        const uint16_t packed_ri = reinterpret_cast<const device uint16_t*>(
            reinterpret_cast<const device uint8_t*>(w) +
            (out_row + r) * in_vec_size_w + k / 2 +
            simd_lid * bytes_per_lane)[i];
        if (DIRECT_NIBBLES) {
          partial[r] += (a0 * (packed_ri & 0x000f) +
                         a1 * ((packed_ri >> 4) & 0x000f) +
                         a2 * ((packed_ri >> 8) & 0x000f) +
                         a3 * ((packed_ri >> 12) & 0x000f));
        } else {
          partial[r] += (a0 * (packed_ri & 0x000f) +
                         a1 * (packed_ri & 0x00f0) +
                         a2 * (packed_ri & 0x0f00) +
                         a3 * (packed_ri & 0xf000));
        }
      }"""

EPILOGUE = """    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }"""

# `scale_local[r]` is a `float` written from a `T`, so the use-site form keeps
# the same T -> float conversion before the same multiply.
EPILOGUE_LAZY = """    for (int r = 0; r < rows_per_simd; r++) {
      const int group_index =
          (out_row + r) * in_vec_size_g + k / 64 + simd_lid / 4;
      const float scale_local_r = scales[group_index];
      const float bias_local_r = biases[group_index];
      acc[r] += scale_local_r * partial[r] + sums * bias_local_r;
    }"""

# The two levers are independent, so the search is their cross product: the row
# block sets how many `VF` values `acc` and `partial` hold, and the staging
# choice sets how long the scalar operands stay live across that peak.
STAGING = {
    "": [],
    "lazysb": [(STAGE, STAGE_WEIGHTS_ONLY), (EPILOGUE, EPILOGUE_LAZY)],
    "lazyw": [(STAGE, STAGE_SCALES_ONLY), (USE, USE_LAZY)],
    "lazy": [(STAGE, STAGE_NONE), (USE, USE_LAZY), (EPILOGUE, EPILOGUE_LAZY)],
}

# (arm, rows_per_simd, unroll the row-block loop, [(anchor, replacement), ...])
ARMS = []
for _rps in (4, 2, 1):
    for _staging, _rewrites in STAGING.items():
        _name = f"rps{_rps}{_staging}" if _rps != 4 else (_staging or "plain")
        ARMS.append((_name, _rps, True, _rewrites))
# The row-block loop has a compile-time trip count, so the allocator is free to
# unroll it and interleave the blocks, which would put the pressure straight
# back. These two hold it rolled and separate that outcome from a real win.
ARMS.append(("rps2nu", 2, False, []))
ARMS.append(("rps1nu", 1, False, []))

HEADER = """// GENERATED by research/e76_wide_gen.py -- do not edit by hand.
//
// Body extracted verbatim from {src} lines {lo}-{hi}
// (extracted-body sha256 {digest}), then rewritten once per arm by a closed set
// of substitutions, each of which must match exactly once.
//
// Shared by every arm: the symbol is renamed, `rows_per_simd` becomes a
// template parameter, and the row-block wrapper below restores the four rows
// per simdgroup that the frozen host grid requires. Nothing else differs from
// the shipped kernel unless the arm's own list says so: the k-loop, the load
// order, the qdot expression tree, the epilogue and the simd_sum reduction are
// the shipped text.
//
// Re-verify with `python3 research/e76_wide_gen.py --check`.

#pragma once

"""

# The frozen host grid gives each simdgroup four output rows
# (`grid_dims(M, ceil(N/8), B)` with `group_dims(32, 2, 1)`), and
# `backend/metal/quantized.cpp` is not editable, so an arm that covers fewer
# rows per body call must be called 4 / ROWS_PER_SIMD times. Weight rows are
# still read exactly once each; the x side is read once per block.
WRAPPER_BLOCKED = """
template <typename T, int NA, bool DIRECT_NIBBLES = false>
METAL_FUNC void {symbol}(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    int first_m,
    int out_row,
    uint simd_lid) {{
  constexpr int kRowsPerSimd = {rps};
  static_assert(4 % kRowsPerSimd == 0, "row blocks must tile 4 rows exactly");
{pragma}  for (int b = 0; b < 4 / kRowsPerSimd; b++) {{
    {symbol}_body<T, NA, DIRECT_NIBBLES, kRowsPerSimd>(
        w, scales, biases, x, y, in_vec_size, out_vec_size, first_m,
        out_row + b * kRowsPerSimd, simd_lid);
  }}
}}
"""

WRAPPER_DIRECT = """
template <typename T, int NA, bool DIRECT_NIBBLES = false>
METAL_FUNC void {symbol}(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    int first_m,
    int out_row,
    uint simd_lid) {{
  {symbol}_body<T, NA, DIRECT_NIBBLES, 4>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, first_m, out_row,
      simd_lid);
}}
"""

NO_UNROLL = "  #pragma clang loop unroll(disable)\n"


def extract(text: str) -> tuple[str, int, int]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(SIGNATURE)]
    if len(starts) != 1:
        sys.exit(f"expected exactly 1 definition of {SIGNATURE!r}, found {len(starts)}")
    start = starts[0]
    if lines[start - 1] != TEMPLATE_LINE:
        sys.exit(f"unexpected template line above the signature: {lines[start - 1]!r}")
    start -= 1
    end = next((i for i in range(start, len(lines)) if lines[i] == "}"), None)
    if end is None:
        sys.exit("no closing brace at column 0 found for the crossrow body")
    return "\n".join(lines[start:end + 1]) + "\n", start + 1, end + 1


def substitute(body: str, arm: str, pairs: list[tuple[str, str]]) -> str:
    out = body
    for old, new in pairs:
        if out.count(old) != 1:
            sys.exit(
                f"arm {arm}: rewrite matched {out.count(old)} times, expected 1.\n"
                f"  looked for: {old!r}\n"
                "  The shipped kernel changed shape; re-read it before trusting "
                "any register number."
            )
        out = out.replace(old, new)
    return out


def render_arm(body: str, arm: str, rps: int, unroll: bool,
               pairs: list[tuple[str, str]]) -> str:
    symbol = f"{BASE_SYMBOL}_{arm}"
    shared = [
        (TEMPLATE_LINE,
         "template <typename T, int NA, bool DIRECT_NIBBLES, int ROWS_PER_SIMD>"),
        (SIGNATURE, f"METAL_FUNC void {symbol}_body("),
        (RPS_DECL, "  constexpr int rows_per_simd = ROWS_PER_SIMD;"),
    ]
    text = substitute(body, arm, shared + pairs)
    note = f"// ---- arm {arm}: rows_per_simd = {rps}, row-block loop "
    note += "unrolled" if unroll else "rolled"
    note += f", {len(pairs)} body rewrite(s)\n"
    if rps == 4:
        wrapper = WRAPPER_DIRECT.format(symbol=symbol)
    else:
        wrapper = WRAPPER_BLOCKED.format(
            symbol=symbol, rps=rps, pragma="" if unroll else NO_UNROLL)
    return note + text + wrapper


def render(body: str, lo: int, hi: int) -> str:
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    parts = [HEADER.format(src=SHIPPED, lo=lo, hi=hi, digest=digest)]
    for arm, rps, unroll, pairs in ARMS:
        parts.append(render_arm(body, arm, rps, unroll, pairs))
    parts.append("\n// One list of arms, so no consumer can drift from the generator.\n")
    parts.append("#define E76_FOR_EACH_ARM(X) \\\n")
    parts.append(" \\\n".join(
        f"    X({arm}, {BASE_SYMBOL}_{arm}, {rps})" for arm, rps, _, _ in ARMS))
    parts.append("\n")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    body, lo, hi = extract(SHIPPED.read_text())
    generated = render(body, lo, hi)

    if args.check:
        if not GENERATED.exists():
            sys.exit(f"{GENERATED} is missing; run without --check")
        if GENERATED.read_text() != generated:
            sys.exit(
                f"{GENERATED} is STALE relative to {SHIPPED}.\n"
                "The arms would no longer be the shipped body."
            )
        print(f"OK: {GENERATED} matches {SHIPPED} lines {lo}-{hi}, "
              f"{len(ARMS)} arms")
        return

    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(generated)
    print(f"wrote {GENERATED} from {SHIPPED} lines {lo}-{hi}, {len(ARMS)} arms")


if __name__ == "__main__":
    main()
