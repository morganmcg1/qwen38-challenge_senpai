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

The arms are the cross product of three independent levers.

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

Accumulator layout, which asks whether `vec<float, NA>` itself costs registers
at non-native widths. The measured g17s step structure is 83, 90, 91, 98, 111 at
NA = 2..6: +7, +1, +7, +13 for a live-float demand that rises by exactly 4 per
step. If the allocator charges an alignment class for a non-native vector width
rather than for live state, replacing the vectors with flat `float` arrays over
the same expression trees recovers it.

  (none)    the shipped `vec<float, NA>` values.
  facc      `acc` only: `VF acc[rows_per_simd]` -> `float acc[rows_per_simd][NA]`.
            The k-block epilogue and the `simd_sum` readout already index by
            `[r][m]`, so the arithmetic text is otherwise unchanged.
  fall      every `VF`: `acc`, `partial`, `sums` and `a0..a3`. Element `m` of
            each vector expression becomes the same scalar expression in the
            same order, so no accumulation is reassociated.

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

# ---------------------------------------------------------------------------
# Accumulator-layout anchors. `vec<float, NA>` indexed by `[m]` and
# `float[...][NA]` indexed by `[m]` read identically, so the readout loop and
# every already-indexed expression are shared text between the two layouts.
# ---------------------------------------------------------------------------

ACC_DECL = """  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }"""

ACC_DECL_FLAT = """  float acc[rows_per_simd][NA];
  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      acc[r][m] = 0.0f;
    }
  }"""

PARTIAL_DECL = """    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }"""

PARTIAL_DECL_FLAT = """    float sums[NA];
    float partial[rows_per_simd][NA];
    for (int m = 0; m < NA; m++) {
      sums[m] = 0.0f;
    }
    for (int r = 0; r < rows_per_simd; r++) {
      for (int m = 0; m < NA; m++) {
        partial[r][m] = 0.0f;
      }
    }"""

A_DECL = "      VF a0, a1, a2, a3;"
A_DECL_FLAT = "      float a0[NA], a1[NA], a2[NA], a3[NA];"

# Element `m` of the vector form, written out. The four products are added in
# the same left-associated order, so nothing is reassociated.
USE_FLAT = """      for (int r = 0; r < rows_per_simd; r++) {
        for (int m = 0; m < NA; m++) {
          if (DIRECT_NIBBLES) {
            partial[r][m] += (a0[m] * (packed[r][i] & 0x000f) +
                              a1[m] * ((packed[r][i] >> 4) & 0x000f) +
                              a2[m] * ((packed[r][i] >> 8) & 0x000f) +
                              a3[m] * ((packed[r][i] >> 12) & 0x000f));
          } else {
            partial[r][m] += (a0[m] * (packed[r][i] & 0x000f) +
                              a1[m] * (packed[r][i] & 0x00f0) +
                              a2[m] * (packed[r][i] & 0x0f00) +
                              a3[m] * (packed[r][i] & 0xf000));
          }
        }
      }"""

# The use-site load stays outside the `m` loop so each `packed_ri` is still read
# exactly once per (r, i), as in the vector form.
USE_LAZY_FLAT = """      for (int r = 0; r < rows_per_simd; r++) {
        const uint16_t packed_ri = reinterpret_cast<const device uint16_t*>(
            reinterpret_cast<const device uint8_t*>(w) +
            (out_row + r) * in_vec_size_w + k / 2 +
            simd_lid * bytes_per_lane)[i];
        for (int m = 0; m < NA; m++) {
          if (DIRECT_NIBBLES) {
            partial[r][m] += (a0[m] * (packed_ri & 0x000f) +
                              a1[m] * ((packed_ri >> 4) & 0x000f) +
                              a2[m] * ((packed_ri >> 8) & 0x000f) +
                              a3[m] * ((packed_ri >> 12) & 0x000f));
          } else {
            partial[r][m] += (a0[m] * (packed_ri & 0x000f) +
                              a1[m] * (packed_ri & 0x00f0) +
                              a2[m] * (packed_ri & 0x0f00) +
                              a3[m] * (packed_ri & 0xf000));
          }
        }
      }"""

EPILOGUE_FLAT = """    for (int r = 0; r < rows_per_simd; r++) {
      for (int m = 0; m < NA; m++) {
        acc[r][m] += scale_local[r] * partial[r][m] + sums[m] * bias_local[r];
      }
    }"""

EPILOGUE_LAZY_FLAT = """    for (int r = 0; r < rows_per_simd; r++) {
      const int group_index =
          (out_row + r) * in_vec_size_g + k / 64 + simd_lid / 4;
      const float scale_local_r = scales[group_index];
      const float bias_local_r = biases[group_index];
      for (int m = 0; m < NA; m++) {
        acc[r][m] += scale_local_r * partial[r][m] + sums[m] * bias_local_r;
      }
    }"""

# ---------------------------------------------------------------------------
# The proposal-width chunk lever.
#
# The row block cuts registers by re-reading x, and the timed session prices
# that at +14 % to +65 % per verify round. Chunking the proposal width instead
# costs no extra traffic at all: the k-block still stages `packed`,
# `scale_local` and `bias_local` exactly once, and each chunk touches a disjoint
# set of `m`, so every x element is still read exactly once. Only `acc` stays
# live across the chunks; `partial`, `sums` and `a0..a3` become chunk-local, and
# each chunk's vectors sit in a narrower lane class.
#
# Three chunk slots cover every legal case at NA <= 6 for a maximum width of 2
# or more. An unused slot has a constexpr width of zero, so its guard and its
# placeholder one-lane vector are removed at compile time.
# ---------------------------------------------------------------------------

K_INTERIOR = """    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
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
        } else {
          sums[m] += load_vector<T, float, 4, 4>(xm, xc);
        }
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      for (int r = 0; r < rows_per_simd; r++) {
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
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }"""

READOUT = """  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] =
            static_cast<T>(reduced);
      }
    }
  }"""

CHUNK_SLOTS = 3


def chunk_widths(width: int) -> str:
    """constexpr chunk widths, greedy at `width`, over `CHUNK_SLOTS` slots."""
    lines = [f"  constexpr int kChunkMax = {width};",
             "  constexpr int C0 = NA < kChunkMax ? NA : kChunkMax;"]
    for slot in range(1, CHUNK_SLOTS):
        taken = " - ".join(f"C{i}" for i in range(slot))
        lines.append(
            f"  constexpr int kLeft{slot} = NA - {taken};\n"
            f"  constexpr int C{slot} = kLeft{slot} <= 0 ? 0"
            f" : (kLeft{slot} < kChunkMax ? kLeft{slot} : kChunkMax);")
    total = " + ".join(f"C{i}" for i in range(CHUNK_SLOTS))
    lines.append(f'  static_assert({total} == NA, "chunks must tile NA");')
    for slot in range(CHUNK_SLOTS):
        lines.append(f"  typedef vec<float, C{slot} == 0 ? 1 : C{slot}> V{slot};")
        lines.append(f"  V{slot} acc{slot}[rows_per_simd];")
    lines.append("  for (int r = 0; r < rows_per_simd; r++) {")
    for slot in range(CHUNK_SLOTS):
        lines.append(f"    acc{slot}[r] = V{slot}(0.0f);")
    lines.append("  }")
    return "\n".join(lines)


def chunk_offset(slot: int) -> str:
    return "" if slot == 0 else " + " + " + ".join(f"C{i}" for i in range(slot))


def chunk_block(slot: int) -> str:
    """One chunk of the k-block. Same text as the shipped body, narrower."""
    off = chunk_offset(slot)
    return f"""    if (C{slot} > 0) {{
      V{slot} sums = V{slot}(0.0f);
      V{slot} partial[rows_per_simd];
      for (int r = 0; r < rows_per_simd; r++) {{
        partial[r] = V{slot}(0.0f);
      }}
      for (int i = 0; i < 4; i++) {{
        V{slot} a0, a1, a2, a3;
        for (int m = 0; m < C{slot}; m++) {{
          const device T* xm = x + (first_m{off} + m) * in_vec_size + k +
              simd_lid * values_per_thread + 4 * i;
          thread float xc[4];
          if (DIRECT_NIBBLES) {{
            xc[0] = static_cast<float>(xm[0]);
            xc[1] = static_cast<float>(xm[1]);
            xc[2] = static_cast<float>(xm[2]);
            xc[3] = static_cast<float>(xm[3]);
            sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
          }} else {{
            sums[m] += load_vector<T, float, 4, 4>(xm, xc);
          }}
          a0[m] = xc[0];
          a1[m] = xc[1];
          a2[m] = xc[2];
          a3[m] = xc[3];
        }}
        for (int r = 0; r < rows_per_simd; r++) {{
          if (DIRECT_NIBBLES) {{
            partial[r] += (a0 * (packed[r][i] & 0x000f) +
                           a1 * ((packed[r][i] >> 4) & 0x000f) +
                           a2 * ((packed[r][i] >> 8) & 0x000f) +
                           a3 * ((packed[r][i] >> 12) & 0x000f));
          }} else {{
            partial[r] += (a0 * (packed[r][i] & 0x000f) +
                           a1 * (packed[r][i] & 0x00f0) +
                           a2 * (packed[r][i] & 0x0f00) +
                           a3 * (packed[r][i] & 0xf000));
          }}
        }}
      }}
      for (int r = 0; r < rows_per_simd; r++) {{
        acc{slot}[r] += scale_local[r] * partial[r] + sums * bias_local[r];
      }}
    }}"""


def chunk_readout(slot: int) -> str:
    off = chunk_offset(slot)
    return f"""  if (C{slot} > 0) {{
    for (int r = 0; r < rows_per_simd; r++) {{
      for (int m = 0; m < C{slot}; m++) {{
        const float reduced = simd_sum(acc{slot}[r][m]);
        if (simd_lid == 0) {{
          y[(first_m{off} + m) * out_vec_size + out_row + r] =
              static_cast<T>(reduced);
        }}
      }}
    }}
  }}"""


def mchunk_rewrites(width: int) -> list[tuple[str, str]]:
    return [
        (ACC_DECL, chunk_widths(width)),
        (K_INTERIOR, "\n".join(chunk_block(s) for s in range(CHUNK_SLOTS))),
        (READOUT, "\n".join(chunk_readout(s) for s in range(CHUNK_SLOTS))),
    ]


# The levers are independent, so the search is their cross product: the row
# block sets how many accumulator values are live, the staging choice sets how
# long the scalar operands stay live across that peak, and the layout choice
# sets whether those values are non-native vectors or flat floats.
STAGING = {
    "": [],
    "lazysb": [(STAGE, STAGE_WEIGHTS_ONLY), (EPILOGUE, EPILOGUE_LAZY)],
    "lazyw": [(STAGE, STAGE_SCALES_ONLY), (USE, USE_LAZY)],
    "lazy": [(STAGE, STAGE_NONE), (USE, USE_LAZY), (EPILOGUE, EPILOGUE_LAZY)],
}


def layout_rewrites(layout: str, staging: str) -> list[tuple[str, str]]:
    """Rewrites applied after the staging rewrites, so they see their output."""
    if not layout:
        return []
    pairs = [(ACC_DECL, ACC_DECL_FLAT)]
    if layout == "fall":
        pairs.append((PARTIAL_DECL, PARTIAL_DECL_FLAT))
        pairs.append((A_DECL, A_DECL_FLAT))
        pairs.append((USE_LAZY, USE_LAZY_FLAT) if staging in ("lazyw", "lazy")
                     else (USE, USE_FLAT))
    pairs.append((EPILOGUE_LAZY, EPILOGUE_LAZY_FLAT)
                 if staging in ("lazysb", "lazy") else (EPILOGUE, EPILOGUE_FLAT))
    return pairs


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
# The layout lever is carried on the shipped staging, which isolates it, and on
# the winning staging, which shows whether the two effects stack. It is carried
# at the shipped row block and at the smallest one for the same reason.
for _layout in ("facc", "fall"):
    for _rps in (4, 1):
        for _staging in ("", "lazy"):
            _name = (f"rps{_rps}" if _rps != 4 else "") + _staging + _layout
            ARMS.append((_name, _rps, True,
                         STAGING[_staging] + layout_rewrites(_layout, _staging)))
# The chunk lever keeps the shipped row block and the shipped staging, because
# its whole claim is that it reduces register pressure without adding traffic.
for _width in (4, 3, 2):
    ARMS.append((f"mc{_width}", 4, True, mchunk_rewrites(_width)))
# Every multi-chunk arm above fails device parity, and `mc4` at NA=5 has exactly
# the same [4,1] partition as `rps1mc4`, which passes. The chunk source is
# therefore not the variable and the row block is. This cross carries both chunk
# widths on all three row blocks so the boundary can be located, and so the
# result separates a row-block dependence from a register-pressure dependence.
for _rps in (2, 1):
    for _width in (4, 2):
        ARMS.append((f"rps{_rps}mc{_width}", _rps, True, mchunk_rewrites(_width)))

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
