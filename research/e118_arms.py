#!/usr/bin/env python3
"""E118: the metadata-load instruction axis of the wide affine-4 QMV.

    research/e118_arms.py --emit /tmp/e118-arms
    research/e118_arms.py --census /tmp/e118-arms --out research/e118-artifacts/census.json

Every lane of a 32-lane simdgroup computes `group_index = row * G + k / 64 +
simd_lid / 4`, so four consecutive lanes ask for the identical `scales` and
`biases` word. E111 (Finding 43) showed that the cost of this field is its LOAD
INSTRUCTION and not its bytes, and then stop-listed every lossless recoding.
This module builds the arms that delete metadata load instructions instead of
metadata bytes, and the two roofline anchors that place them on Finding 44.

Arms, all instantiated per width as their own `e118_iso_na<NA>` entry point so
the backend allocates registers for one width instead of a max over branches:

  a_base          the shipped wide template, unmodified.
  q_scaffold      the shipped body re-emitted through the shared
                  prologue/body/epilogue scaffold, with NOTHING changed. It is
                  the null arm: it must be bit-identical AND, on both
                  architectures, byte-identical in machine text to `a_base`, and
                  its timing spread is this session's instrument noise floor.
  s_bcast         the assignment's broadcast: lanes 0 to 7 load the eight
                  distinct `scales`/`biases` words for one row and every lane
                  takes `simd_shuffle(v, simd_lid / 4)`. Eight load instructions
                  per k-block stay, but only a quarter of the lanes are active
                  in each.
  s_bcast_all     the same idea taken to the instruction level. rows_per_simd *
                  groups_per_block == 4 * 8 == 32 == SIMD_SIZE, so ONE fully
                  active load instruction fetches the metadata of all four rows
                  and eight `simd_shuffle` calls distribute it. Eight metadata
                  load instructions per k-block become two.
  s_bcast_scale   `s_bcast` for `scales` only; `biases` is loaded as shipped.
                  Prices the two fields separately.
  p_split_meta    the metadata loads hoisted out of the weight-load loop into
                  their own `r` loop, so the eight metadata loads issue back to
                  back. Identical instructions, identical registers, different
                  memory-level parallelism.
  g_pack32        E111's weak lead: the same values read as one interleaved
                  uint32. Same bytes, four metadata loads instead of eight.
  s_bcast_pack32  `s_bcast` and `g_pack32` composed.
  p_prefetch_w    E104 arm P, never measured on any host: the
                  `packed[rows_per_simd][4]` weight loads are double-buffered
                  one k-block ahead. Eight extra registers, which g16s does not
                  have at NA = 4 and NA = 5.
  n_nosums        DIAGNOSTIC. E111's arm: the bias is still loaded, the x-sum
                  chain is removed. The mandated third reading of the
                  E111 (+6.132 %) / E104 (-4.47 %) contradiction.
  l_loadonly      DIAGNOSTIC. E104's load-only arm, reproduced byte for byte, as
                  the Finding 44 load-ceiling anchor.
  n_nobias        DIAGNOSTIC. E111's arm: the bias load AND the x-sum chain are
                  both removed, 34 B per group. The ceiling of the whole bias
                  axis.
  d_bias1         DIAGNOSTIC. E111's arm: the bias is read from a one-byte
                  array and used raw, so the value is deliberately wrong.
                  35 B per group with no reconstruction cost, so this is the
                  ceiling of Bias6 itself.
  e_bias6         E111's arm: the bias is reconstructed from (scale, 6-bit code)
                  with integer arithmetic. Must be bit-identical to `a_base`.

E111 measured its bias arms at NA = 5 only, which carries 0.034 of the standing
round weight. The last three arms above repeat them at NA = 2, 3, 4 and 5 so the
bias axis gets a round-weighted number instead of a corner one.

Every arm except those marked DIAGNOSTIC keeps every floating-point value and
every accumulation order of `a_base`, so all of them must be bit-identical to it
at every cell. The probe checks that with a positive control that fires.

Research-only: nothing here is on the scored path, and this experiment does not
modify any submitted file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH, RANKED_ARCH, build_metallib, translate,
)
from e104_variant_sources import (  # noqa: E402
    EPILOGUE, PROLOGUE, emit_base, wide_fn_span, widen_asserts,
)

WIDTHS = (2, 3, 4, 5)

# --- the shipped inner body, in the shared scaffold ---------------------------
# `DIRECT_NIBBLES == true` is the branch every scored cell takes, so the
# scaffold carries it unconditionally. E104's `xw_widex` and E110's `b_barrier`
# were both built on these strings and were bit-identical to the unmodified
# shipped kernel at every cell; `q_scaffold` re-proves that here.
BODY_BASE = """
    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        a0[m] = static_cast<float>(xm[0]);
        a1[m] = static_cast<float>(xm[1]);
        a2[m] = static_cast<float>(xm[2]);
        a3[m] = static_cast<float>(xm[3]);
      }
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
"""

# E104's load-only arm, reproduced verbatim. The activation loads survive
# because the bias-correction sum reads all four values; only the nibble
# extraction and three quarters of the vector FMA go.
BODY_LOADONLY = """
    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0;
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        a0[m] = static_cast<float>(xm[0]);
      }
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += a0 * float(packed[r][i] & 0x000f);
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
"""

# E111's `n_nosums`, line for line: the bias load stays live, the `sums` chain
# and the `sums * bias_local[r]` product are removed.
BODY_NOSUMS = """
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        a0[m] = static_cast<float>(xm[0]);
        a1[m] = static_cast<float>(xm[1]);
        a2[m] = static_cast<float>(xm[2]);
        a3[m] = static_cast<float>(xm[3]);
      }
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + bias_local[r];
    }
"""

# E111's `n_nobias`: `n_nosums` with the bias load deleted as well, 34 B per
# group. `bias_local` is left declared and never written, so the backend drops
# it and the arm prices the whole bias axis, traffic and arithmetic together.
BODY_NOBIAS = BODY_NOSUMS.replace(
    "acc[r] += scale_local[r] * partial[r] + bias_local[r];",
    "acc[r] += scale_local[r] * partial[r];")

# --- rung 2, Finding 53: the two simdgroups compute `sums` twice ---------------
#
# The activation pointer inside this loop is
#     x + (first_m + m) * in_vec_size + k + simd_lid * values_per_thread + 4 * i
# which contains no `out_row` and no `simd_gid`. Lane L of simdgroup 0 and lane
# L of simdgroup 1 therefore read byte-identical addresses and compute
# byte-identical `sums`. The whole activation side is computed twice per
# threadgroup and only `sums` is a reduction, so only `sums` is worth sharing:
# NA floats exchanged against 4 * NA * 5 add-tree instructions removed.
#
# The split is a UNIFORM branch on `simd_gid` around the whole `i` loop, with
# the body duplicated. A branch on `simd_gid` is uniform inside a simdgroup so
# it is free; predicating inside the `m` loop would not be, which is exactly
# what the `s_bcast` to `s_bcast_all` decomposition measured.
#
# `H` splits the components. Simdgroup 0 owns m < H, simdgroup 1 owns m >= H.
SUMSHARE_HEAD_NOH = """
    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
"""
SUMSHARE_HEAD = SUMSHARE_HEAD_NOH + "    constexpr int H = (NA + 1) / 2;\n"


def sumshare_half(own: str) -> str:
    """One duplicated `i` loop whose `sums` add tree is gated by `own`."""
    return ("""      for (int i = 0; i < 4; i++) {
        VF a0, a1, a2, a3;
        for (int m = 0; m < NA; m++) {
          const device T* xm = x + (first_m + m) * in_vec_size + k +
              simd_lid * values_per_thread + 4 * i;
          if (%s) {
            sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
          }
          a0[m] = static_cast<float>(xm[0]);
          a1[m] = static_cast<float>(xm[1]);
          a2[m] = static_cast<float>(xm[2]);
          a3[m] = static_cast<float>(xm[3]);
        }
        for (int r = 0; r < rows_per_simd; r++) {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * ((packed[r][i] >> 4) & 0x000f) +
                         a2 * ((packed[r][i] >> 8) & 0x000f) +
                         a3 * ((packed[r][i] >> 12) & 0x000f));
        }
      }
""" % own)


SUMSHARE_TAIL = """    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
"""

# n_halfsums: the zero-cost upper bound on the mechanism. Each simdgroup drops
# the other half's add tree and never gets it back, so half of `sums` stays at
# zero and the arm MUST fail bit-exactness. Diagnostic, in the same spirit as
# `n_nosums`.
BODY_HALFSUMS = (SUMSHARE_HEAD +
                 "    if (simd_gid == 0) {\n" + sumshare_half("m < H") +
                 "    } else {\n" + sumshare_half("m >= H") +
                 "    }\n" + SUMSHARE_TAIL)

# The exchange. Lane index is the FASTEST dimension: with [sg][m][lane]
# consecutive lanes touch consecutive floats and there is no bank conflict.
# [sg][lane][m] would stride by NA floats and collide four ways at NA=4.
SUMSHARE_XCHG = """    for (int m = 0; m < NA; m++) {
      if ((simd_gid == 0) == (m < H)) {
        sums_xchg[(int(simd_gid) * NA + m) * SIMD_SIZE + int(simd_lid)] =
            sums[m];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int m = 0; m < NA; m++) {
      if ((simd_gid == 0) != (m < H)) {
        sums[m] =
            sums_xchg[((1 - int(simd_gid)) * NA + m) * SIMD_SIZE +
                      int(simd_lid)];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# x_sumshare_split: bit exact. Each `sums[m]` is still accumulated across `i` in
# the original order by exactly one simdgroup, and the float written to
# threadgroup memory is the float read back. The other simdgroup would have
# computed the identical value from the identical addresses. The epilogue is
# untouched, so no summation order changes anywhere.
BODY_SUMSHARE_SPLIT = (SUMSHARE_HEAD +
                       "    if (simd_gid == 0) {\n" + sumshare_half("m < H") +
                       "    } else {\n" + sumshare_half("m >= H") +
                       "    }\n" + SUMSHARE_XCHG + SUMSHARE_TAIL)

# x_sumshare_owner: bit exact. Simdgroup 0 computes every component and writes
# them; simdgroup 1 computes none and reads them all. Fewer threadgroup
# operations than the split arm and a worse critical path, at the same total
# issued instruction count, so the pair is a direct test of whether total issue
# dominates the critical path in this loop.
SUMSHARE_XCHG_OWNER = """    if (simd_gid == 0) {
      for (int m = 0; m < NA; m++) {
        sums_xchg[m * SIMD_SIZE + int(simd_lid)] = sums[m];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid != 0) {
      for (int m = 0; m < NA; m++) {
        sums[m] = sums_xchg[m * SIMD_SIZE + int(simd_lid)];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
BODY_SUMSHARE_OWNER = (SUMSHARE_HEAD_NOH +
                       "    if (simd_gid == 0) {\n" +
                       sumshare_half("true") +
                       "    } else {\n" + sumshare_half("false") +
                       "    }\n" + SUMSHARE_XCHG_OWNER + SUMSHARE_TAIL)

# --- prologue surgery ---------------------------------------------------------

SIG_TAIL = "    uint simd_lid) {\n"
# The default keeps the shipped `qmv_fast_crossrow_affine4_g64_m` dispatcher
# compiling unchanged. That wrapper is never dispatched by this probe, which
# reaches `_wide` directly, so the null pointer is never dereferenced.
SIG_TAIL_SB = ("    uint simd_lid,\n"
               "    const device uint32_t* packed_sb = nullptr) {\n")
SIG_TAIL_CODES = ("    uint simd_lid,\n"
                  "    const device uint8_t* bias_codes = nullptr) {\n")
# Rung 2 needs the simdgroup index inside the kernel body, and a threadgroup
# staging buffer the entry point owns. Both default so the shipped
# `qmv_fast_crossrow_affine4_g64_m` wrapper still compiles untouched.
SIG_TAIL_GID = ("    uint simd_lid,\n"
                "    uint simd_gid = 0) {\n")
SIG_TAIL_XCHG = ("    uint simd_lid,\n"
                 "    uint simd_gid = 0,\n"
                 "    threadgroup float* sums_xchg = nullptr) {\n")

# E111's one-byte bias code. The low four bits are an integer multiple of the
# group scale and the high two bits are a BF16 ULP correction, so the whole
# reconstruction is integer arithmetic on the product's bit pattern and the CPU
# packer and the GPU agree with no dependence on a rounding mode.
BIAS6_HELPER = """static inline float e118_bias_from_code(float scale, uint code) {
  const float prod = -float(code & 0xFu) * scale;
  uint u = as_type<uint>(prod);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u += ((code & 0x30u) << 12) - 0x10000u;
  return as_type<float>(u & 0xFFFF0000u);
}

"""

WEIGHT_LOAD = """      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * bytes_per_lane);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
"""

META_LOAD = """      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
"""

WEIGHT_META_LOOP = ("    for (int r = 0; r < rows_per_simd; r++) {\n"
                    "      const int row = out_row + r;\n"
                    + WEIGHT_LOAD + META_LOAD + "    }\n")

# s_bcast: the assignment's broadcast, one row at a time. Lanes 8 to 31 must not
# load, because `group_base + simd_lid` leaves the row on the last k-block and
# leaves the buffer on the last row.
META_BCAST = """      const int group_base = row * in_vec_size_g + k / 64;
      const int meta_lane = int(simd_lid);
      const float s_src =
          meta_lane < 8 ? float(scales[group_base + meta_lane]) : 0.0f;
      const float b_src =
          meta_lane < 8 ? float(biases[group_base + meta_lane]) : 0.0f;
      scale_local[r] = simd_shuffle(s_src, ushort(simd_lid / 4));
      bias_local[r] = simd_shuffle(b_src, ushort(simd_lid / 4));
"""

META_BCAST_SCALE = """      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      const int group_base = row * in_vec_size_g + k / 64;
      const int meta_lane = int(simd_lid);
      const float s_src =
          meta_lane < 8 ? float(scales[group_base + meta_lane]) : 0.0f;
      scale_local[r] = simd_shuffle(s_src, ushort(simd_lid / 4));
      bias_local[r] = biases[group_index];
"""

META_PACK32 = """      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      const uint32_t sb = packed_sb[group_index];
      scale_local[r] = float(as_type<T>(uint16_t(sb & 0xFFFFu)));
      bias_local[r] = float(as_type<T>(uint16_t(sb >> 16)));
"""

META_NOBIAS = """      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
"""

META_CODE1 = """      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = float(bias_codes[group_index]);
"""

META_BIAS6 = """      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      const float scale = scales[group_index];
      scale_local[r] = scale;
      bias_local[r] = e118_bias_from_code(scale, uint(bias_codes[group_index]));
"""

META_BCAST_PACK32 = """      const int group_base = row * in_vec_size_g + k / 64;
      const int meta_lane = int(simd_lid);
      const uint32_t sb_src =
          meta_lane < 8 ? packed_sb[group_base + meta_lane] : 0u;
      const uint32_t sb = simd_shuffle(sb_src, ushort(simd_lid / 4));
      scale_local[r] = float(as_type<T>(uint16_t(sb & 0xFFFFu)));
      bias_local[r] = float(as_type<T>(uint16_t(sb >> 16)));
"""

# s_bcast_all: rows_per_simd * groups_per_k_block == 4 * 8 == 32 == SIMD_SIZE, so
# one fully active load instruction covers every metadata word the simdgroup
# needs for the whole k-block. Lane j takes row `out_row + j / 8` and group
# offset `j % 8`, so the word lane L wants for row r sits in lane
# `r * 8 + L / 4`. Every index stays inside its own row, so no guard is needed.
WEIGHT_META_LOOP_BCAST_ALL = """    {
      const int meta_row = out_row + int(simd_lid) / 8;
      const int meta_index =
          meta_row * in_vec_size_g + k / 64 + int(simd_lid) % 8;
      const float s_src = float(scales[meta_index]);
      const float b_src = float(biases[meta_index]);
      for (int r = 0; r < rows_per_simd; r++) {
        const ushort src_lane = ushort(r * 8 + simd_lid / 4);
        scale_local[r] = simd_shuffle(s_src, src_lane);
        bias_local[r] = simd_shuffle(b_src, src_lane);
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
""" + WEIGHT_LOAD + "    }\n"

# p_split_meta: identical instructions, issued in two back-to-back runs instead
# of interleaved with the weight loads.
WEIGHT_META_LOOP_SPLIT = (
    "    for (int r = 0; r < rows_per_simd; r++) {\n"
    "      const int row = out_row + r;\n" + WEIGHT_LOAD + "    }\n"
    "    for (int r = 0; r < rows_per_simd; r++) {\n"
    "      const int row = out_row + r;\n" + META_LOAD + "    }\n")

# z_ballast: register pressure with NO arithmetic change. `ballast` is
# loop-carried and data-dependent, so it cannot be hoisted, and it is consumed
# only inside a branch the backend cannot prove dead, so nothing it holds ever
# reaches `y`. Every value and every accumulation order that produces an output
# is the shipped one, which makes this the control for the question "does
# spilling alone change the answer?".
BALLAST_N = 12
BALLAST_DECL = """  thread float ballast[%d];
  for (int i = 0; i < %d; i++) {
    ballast[i] = float(i) + float(simd_lid);
  }

  VF acc[rows_per_simd];
""" % (BALLAST_N, BALLAST_N)

BALLAST_STEP = ("    for (int i = 0; i < %d; i++) {\n"
                "      ballast[i] = fma(ballast[i], 1.0000001f,\n"
                "                       scale_local[i %% rows_per_simd]);\n"
                "    }\n") % BALLAST_N

BALLAST_SINK = """  float ballast_sum = 0.0f;
  for (int i = 0; i < %d; i++) {
    ballast_sum += ballast[i];
  }
  if (in_vec_size < 0) {
    y[0] = static_cast<T>(ballast_sum);
  }

""" % BALLAST_N


def expect(text: str, needle: str, count: int, label: str) -> None:
    seen = text.count(needle)
    if seen != count:
        raise SystemExit(
            "e118_arms: %s matched %d times, expected %d" % (label, seen, count))


# --- the instruction-price calibration ladder ---------------------------------
# The screen arms answer "is this mechanism worth anything". They cannot answer
# "what does one instruction cost", because every screen arm changes several
# things at once. These arms change exactly one thing: the number of injected
# instructions of ONE class, per k-block iteration, holding the register
# footprint fixed across the whole ladder.
#
# `lanes` independent accumulators are carried across the k loop and `depth`
# rounds of `lanes` operations are injected per iteration, so the arm issues
# exactly `lanes * depth` extra instructions of the named class per iteration
# while allocating exactly `lanes` extra registers. Two rungs of a class plus
# `a_base` at zero give the slope in microseconds per instruction directly,
# with no fitted intermediary and no cross-arm confound.
#
# Holding registers fixed across rungs is the point. E118's own spill defect
# shows that register pressure, not instruction count, is what breaks the wide
# kernel at NA=5, so a ladder that changed both would measure neither.
CAL_KINDS = {
    # one fused multiply-add reading a live value, so it cannot be folded.
    "alu": "      cal[%(j)d] = fma(cal[%(j)d], 1.0000001f, "
           "scale_local[%(jr)d]);\n",
    # one simdgroup shuffle, the primitive `s_bcast` substitutes for a load.
    "shuf": "      cal[%(j)d] = simd_shuffle(cal[%(j)d], "
            "ushort((simd_lid + %(rot)d) & 31));\n",
    # one device load from the metadata block this k-block already touched, so
    # the rung prices the load INSTRUCTION and not a cache miss. Every injected
    # load gets a DISTINCT address: the eight groups of this k-block, then the
    # next row's eight. Without that the optimiser common-subexpression folds
    # the repeats and the upper rung silently collapses onto the lower one.
    "ld": "      cal[%(j)d] += float(scales[cal_base + %(row)d * in_vec_size_g"
          " + ((simd_lid + %(rot)d) & 7)]);\n",
}


def cal_decl(lanes: int) -> str:
    return ("  thread float cal[%d];\n"
            "  for (int i = 0; i < %d; i++) {\n"
            "    cal[i] = float(i) + float(simd_lid);\n"
            "  }\n"
            "\n"
            "  VF acc[rows_per_simd];\n") % (lanes, lanes)


def cal_step(kind: str, lanes: int, depth: int) -> str:
    """Exactly `lanes * depth` injected instructions, fully unrolled in source.

    Lanes alternate inside each round so consecutive injected operations are
    independent of one another; only operations `lanes` apart share a chain.
    """
    template = CAL_KINDS[kind]
    out = []
    if kind == "ld":
        out.append("    const int cal_base = out_row * in_vec_size_g + k / 64;\n")
    out.append("    {\n")
    for d in range(depth):
        for j in range(lanes):
            n = d * lanes + j
            out.append(template % {"j": j, "jr": j % 4,
                                   "rot": n % 8, "row": n // 8})
    out.append("    }\n")
    return "".join(out)


def cal_sink(lanes: int) -> str:
    return ("  float cal_sum = 0.0f;\n"
            "  for (int i = 0; i < %d; i++) {\n"
            "    cal_sum += cal[i];\n"
            "  }\n"
            "  if (in_vec_size < 0) {\n"
            "    y[0] = static_cast<T>(cal_sum);\n"
            "  }\n"
            "\n") % lanes


def cal_prologue(kind: str, lanes: int, depth: int) -> str:
    text = PROLOGUE
    expect(text, "  VF acc[rows_per_simd];\n", 1, "accumulator declaration")
    text = text.replace("  VF acc[rows_per_simd];\n", cal_decl(lanes))
    expect(text, WEIGHT_META_LOOP, 1, "prologue weight+metadata loop")
    return text.replace(WEIGHT_META_LOOP,
                        WEIGHT_META_LOOP + cal_step(kind, lanes, depth))


def cal_epilogue(lanes: int) -> str:
    head = "  for (int r = 0; r < rows_per_simd; r++) {\n"
    expect(EPILOGUE, head, 1, "epilogue store loop")
    return EPILOGUE.replace(head, cal_sink(lanes) + head)


def cal_plan(kind: str, lanes: int, depth: int):
    return ((cal_prologue(kind, lanes, depth), BODY_BASE, cal_epilogue(lanes)),
            "")


# --- the two arithmetic-axis ceilings the advisor asked for -------------------
# DIAGNOSTIC, both of them. Neither is promotable and neither is bit exact.
#
# `y_algebra` prices Finding 40's algebraic route. Over all 400,343,040 scored
# groups `bias == q0 * scale` exactly, so
#     scale*partial + sums*bias  ==  scale*(partial + sums*q0)
# in exact arithmetic, one multiply per row per k-block cheaper. In floating
# point it reassociates, so it can never ship. This arm keeps every load of
# `a_base` and changes ONLY the shape of the accumulate expression, so it
# isolates the arithmetic saving with no load-side confound. Its output values
# are deliberately not `a_base`'s.
expect(BODY_BASE, "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
       1, "base accumulate expression")
BODY_ALGEBRA = BODY_BASE.replace(
    "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
    "acc[r] += scale_local[r] * (partial[r] + sums * bias_local[r]);")

# `y_hsum_tree` prices the DEPENDENCY DEPTH of the activation sum chain. The
# shipped source writes `xm[0] + xm[1] + xm[2] + xm[3]`, and the front end emits
# three strictly left-associated scalar bfloat adds, a serial chain of depth 3.
# The tree form is the same three adds at depth 2. Same instruction count, same
# class, shorter chain, so it separates "issue bound" from "latency bound"
# inside the single largest term on the table. It reassociates, so it is a
# ceiling and not a candidate.
expect(BODY_BASE, "sums[m] += xm[0] + xm[1] + xm[2] + xm[3];", 1,
       "base activation sum chain")
BODY_HSUM_TREE = BODY_BASE.replace(
    "sums[m] += xm[0] + xm[1] + xm[2] + xm[3];",
    "sums[m] += (xm[0] + xm[1]) + (xm[2] + xm[3]);")


def prologue_with(meta: str | None = None, loop: str | None = None,
                  extra: str = "", helper: str = "") -> str:
    """The shared prologue with the metadata load or the whole loop replaced."""
    text = PROLOGUE
    if meta is not None:
        expect(text, META_LOAD, 1, "prologue metadata load")
        text = text.replace(META_LOAD, meta)
    if loop is not None:
        expect(text, WEIGHT_META_LOOP, 1, "prologue weight+metadata loop")
        text = text.replace(WEIGHT_META_LOOP, loop)
    if extra:
        tail = {"packed_sb": SIG_TAIL_SB, "bias_codes": SIG_TAIL_CODES,
                "simd_gid": SIG_TAIL_GID,
                "simd_gid, sums_xchg": SIG_TAIL_XCHG}[extra]
        expect(text, SIG_TAIL, 1, "prologue signature tail")
        text = text.replace(SIG_TAIL, tail)
    return helper + text


def ballast_prologue() -> str:
    text = PROLOGUE
    expect(text, "  VF acc[rows_per_simd];\n", 1, "accumulator declaration")
    text = text.replace("  VF acc[rows_per_simd];\n", BALLAST_DECL)
    expect(text, WEIGHT_META_LOOP, 1, "prologue weight+metadata loop")
    return text.replace(WEIGHT_META_LOOP, WEIGHT_META_LOOP + BALLAST_STEP)


def ballast_epilogue() -> str:
    head = "  for (int r = 0; r < rows_per_simd; r++) {\n"
    expect(EPILOGUE, head, 1, "epilogue store loop")
    return EPILOGUE.replace(head, BALLAST_SINK + head)


# --- p_prefetch_w -------------------------------------------------------------
# The k loop is restructured, so this arm replaces the whole template rather
# than a prologue. `k_next` is clamped instead of branched, so the last
# iteration re-reads the last block into registers it never uses and the load
# stays unpredicated. Values, operand types and accumulation order are
# untouched, so the arm is bit exact by construction.
PREFETCH_PROLOGUE = """template <typename T, int NA, bool DIRECT_NIBBLES = false>
METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    int first_m,
    int out_row,
    uint simd_lid) {
  static_assert(NA >= 2 && NA <= 8, "e118 probe admits NA in [2, 8]");
  typedef vec<float, NA> VF;
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }

  thread uint16_t ahead[rows_per_simd][4];
  for (int r = 0; r < rows_per_simd; r++) {
    const device uint16_t* ws0 = reinterpret_cast<const device uint16_t*>(
        reinterpret_cast<const device uint8_t*>(w) +
        (out_row + r) * in_vec_size_w + simd_lid * bytes_per_lane);
    for (int i = 0; i < 4; i++) {
      ahead[r][i] = ws0[i];
    }
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread uint16_t packed[rows_per_simd][4];
    for (int r = 0; r < rows_per_simd; r++) {
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ahead[r][i];
      }
    }
    const int k_next = min(k + block_size, in_vec_size - block_size);
    for (int r = 0; r < rows_per_simd; r++) {
      const device uint16_t* wn = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) +
          (out_row + r) * in_vec_size_w + k_next / 2 +
          simd_lid * bytes_per_lane);
      for (int i = 0; i < 4; i++) {
        ahead[r][i] = wn[i];
      }
    }
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int group_index =
          (out_row + r) * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }
"""

# --- entry points -------------------------------------------------------------
# `qmv_fast_crossrow_affine4_g64_m<T, NA, NA, true>` has `TAIL == M % IPG == 0`,
# so its only reachable branch is the direct `_wide` call below. Calling `_wide`
# here instead lets the pack32 arms pass a ninth buffer without patching the
# shipped dispatcher, and the census proves the two routes produce identical
# machine text for `a_base`.
ISO_KERNEL = """
[[kernel]] void e118_iso_na%(na)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const device uint32_t* packed_sb [[buffer(7)]],
    const device uint8_t* bias_codes [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * %(na)d;
  if (first_m >= %(na)d) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
%(decl)s  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, first_m, out_row,
      simd_lid%(extra)s);
}
"""

# Threadgroup staging the entry point owns, for the rung-2 arms only. The
# budget is 32768 bytes per threadgroup; the largest of these is 1280 at NA=5,
# so occupancy is not limited by it. `2 * NA` slots for the split arm because
# both simdgroups write, `NA` for the owner arm because only simdgroup 0 does.
ENTRY_DECL = {
    "n_halfsums": "",
    "x_sumshare_split":
        "  threadgroup float sums_xchg[2 * %(na)d * 32];\n",
    "x_sumshare_owner":
        "  threadgroup float sums_xchg[%(na)d * 32];\n",
}
# Threadgroup bytes per arm per width, for the occupancy arithmetic the
# assignment asks for. Derived from the same expressions as the declarations
# above so the two cannot drift.
THREADGROUP_BYTES = {
    "x_sumshare_split": {na: 2 * na * 32 * 4 for na in WIDTHS},
    "x_sumshare_owner": {na: na * 32 * 4 for na in WIDTHS},
}
THREADGROUP_BUDGET_BYTES = 32768

# Census-only control: the same arm reached through the shipped dispatcher
# wrapper, to prove the direct call above is not itself a change.
ISO_KERNEL_VIA_M = """
[[kernel]] void e118_iso_na%(na)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const device uint32_t* packed_sb [[buffer(7)]],
    const device uint8_t* bias_codes [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, %(na)d, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid,
      simd_lid);
}
"""

# arm -> (prologue, body, epilogue) applied to the wide template, or None for
# the unmodified shipped template, plus the extra buffer the arm reads.
PLANS = {
    "a_base": (None, ""),
    "q_scaffold": ((PROLOGUE, BODY_BASE, EPILOGUE), ""),
    "s_bcast": ((prologue_with(meta=META_BCAST), BODY_BASE, EPILOGUE), ""),
    "s_bcast_all": ((prologue_with(loop=WEIGHT_META_LOOP_BCAST_ALL), BODY_BASE,
                     EPILOGUE), ""),
    "s_bcast_scale": ((prologue_with(meta=META_BCAST_SCALE), BODY_BASE,
                       EPILOGUE), ""),
    "p_split_meta": ((prologue_with(loop=WEIGHT_META_LOOP_SPLIT), BODY_BASE,
                      EPILOGUE), ""),
    "g_pack32": ((prologue_with(meta=META_PACK32, extra="packed_sb"), BODY_BASE,
                  EPILOGUE), "packed_sb"),
    "s_bcast_pack32": ((prologue_with(meta=META_BCAST_PACK32,
                                      extra="packed_sb"),
                        BODY_BASE, EPILOGUE), "packed_sb"),
    "p_prefetch_w": ((PREFETCH_PROLOGUE, BODY_BASE, EPILOGUE), ""),
    "n_nosums": ((PROLOGUE, BODY_NOSUMS, EPILOGUE), ""),
    "l_loadonly": ((PROLOGUE, BODY_LOADONLY, EPILOGUE), ""),
    "n_nobias": ((prologue_with(meta=META_NOBIAS), BODY_NOBIAS, EPILOGUE), ""),
    "d_bias1": ((prologue_with(meta=META_CODE1, extra="bias_codes"), BODY_BASE,
                 EPILOGUE), "bias_codes"),
    "e_bias6": ((prologue_with(meta=META_BIAS6, extra="bias_codes",
                               helper=BIAS6_HELPER),
                 BODY_BASE, EPILOGUE), "bias_codes"),
    "z_ballast": ((ballast_prologue(), BODY_BASE, ballast_epilogue()), ""),
    # the instruction-price ladder: two rungs per class, `a_base` is the zero
    # rung, and every rung of every class allocates the same two extra
    # registers so only the instruction count varies.
    "k_alu8": cal_plan("alu", 2, 4),
    "k_alu16": cal_plan("alu", 2, 8),
    "k_ld8": cal_plan("ld", 2, 4),
    "k_ld16": cal_plan("ld", 2, 8),
    "k_shuf8": cal_plan("shuf", 2, 4),
    "k_shuf16": cal_plan("shuf", 2, 8),
    # the ILP control for the ladder: the same 16 injected ALU instructions as
    # `k_alu16` spread over four independent chains instead of two. If the two
    # agree, the ladder is measuring issue throughput and not chain latency.
    "k_alu16w": cal_plan("alu", 4, 4),
    # the two arithmetic-axis ceilings, both deliberately not bit exact.
    "y_algebra": ((PROLOGUE, BODY_ALGEBRA, EPILOGUE), ""),
    "y_hsum_tree": ((PROLOGUE, BODY_HSUM_TREE, EPILOGUE), ""),
    # rung 2, Finding 53. `n_halfsums` is the ceiling and must fail exactness;
    # the two `x_sumshare_*` arms are bit exact by construction.
    "n_halfsums": ((prologue_with(extra="simd_gid"),
                    BODY_HALFSUMS, EPILOGUE), "simd_gid"),
    "x_sumshare_split": ((prologue_with(extra="simd_gid, sums_xchg"),
                          BODY_SUMSHARE_SPLIT, EPILOGUE),
                         "simd_gid, sums_xchg"),
    "x_sumshare_owner": ((prologue_with(extra="simd_gid, sums_xchg"),
                          BODY_SUMSHARE_OWNER, EPILOGUE),
                         "simd_gid, sums_xchg"),
}

# Arms that are NOT required to reproduce `a_base` bit for bit.
DIAGNOSTIC_ARMS = ("n_nosums", "l_loadonly", "n_nobias", "d_bias1",
                   "y_algebra", "y_hsum_tree", "n_halfsums")
# Rung 2, run as its own session on one cell first under its own kill rule.
RUNG2_ARMS = ("n_halfsums", "x_sumshare_split", "x_sumshare_owner")

# The calibration ladder, by injected instruction class, as
# (arm, injected instructions per k-block iteration, independent chains).
CAL_LADDER = {
    "alu": (("k_alu8", 8, 2), ("k_alu16", 16, 2), ("k_alu16w", 16, 4)),
    "ld": (("k_ld8", 8, 2), ("k_ld16", 16, 2)),
    "shuf": (("k_shuf8", 8, 2), ("k_shuf16", 16, 2)),
}
CAL_ARMS = tuple(a for rungs in CAL_LADDER.values() for a, _, _ in rungs)
# Arms the primary metric may rank: bit exact AND not register-confounded on
# this host. `p_prefetch_w` is bit exact but spills on g16s at the widths that
# matter, so it is reported separately.
PROMOTION_ARMS = ("s_bcast", "s_bcast_all", "s_bcast_scale", "p_split_meta",
                  "g_pack32", "s_bcast_pack32")

ARMS = tuple(PLANS)


def arm_source(base: str, arm: str, via_m: bool = False) -> str:
    plan, extra = PLANS[arm]
    text = base
    if plan is not None:
        prologue, body, epilogue = plan
        start, end = wide_fn_span(base)
        text = base[:start] + prologue + body + epilogue + base[end:]
    template = ISO_KERNEL_VIA_M if via_m else ISO_KERNEL
    decl = ENTRY_DECL.get(arm, "")
    if via_m:
        return text + "".join(template % {"na": na} for na in WIDTHS)
    return text + "".join(
        template % {"na": na, "extra": ", " + extra if extra else "",
                    "decl": decl % {"na": na} if decl else ""}
        for na in WIDTHS)


def emit(outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    base = widen_asserts(emit_base(outdir / "base_raw.metal"))
    (outdir / "base_lone.metal").write_text(base)
    seen: dict[str, str] = {}
    for arm in ARMS:
        text = arm_source(base, arm)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        if digest in seen:
            raise SystemExit(
                "e118_arms: %s and %s are byte-identical" % (arm, seen[digest]))
        seen[digest] = arm
        (outdir / ("arm_%s.metal" % arm)).write_text(text)
        print("%-15s %8d bytes  sha=%s  exact=%-5s  extra=%s"
              % (arm, len(text), digest, arm not in DIAGNOSTIC_ARMS,
                 PLANS[arm][1] or "-"))
    (outdir / "ctl_a_base_via_m.metal").write_text(
        arm_source(base, "a_base", via_m=True))
    names = ",".join(a + (":diag" if a in DIAGNOSTIC_ARMS else "") for a in ARMS)
    print("\n--arms %s" % names)


# --- census -------------------------------------------------------------------

KERNEL_RE = re.compile(r"e118_iso_na(\d+)$")
DEVICE_LOAD = re.compile(r"=\s*load\s.*addrspace\(1\)")
SHUFFLE = re.compile(r"simd_shuffle|@air\.simd_shuffle")

# The instruction categories the cost model regresses against.
#
# Two things have to be right for an AIR count to mean anything, and both are
# easy to get wrong:
#
#  1. AIR is pre-register-allocation LLVM IR, so a large part of it is SSA
#     bookkeeping the backend never issues: `phi`, `br`, `insertelement`,
#     `shufflevector`, `alloca`, `bitcast`, `llvm.lifetime.*`. Counting those
#     as instructions is most of why Finding 36 says AIR operation counts do
#     not predict time. They are counted here, into their own bucket, and then
#     excluded from the issue count rather than silently dropped.
#  2. AIR is vector-typed and this GPU issues one lane at a time, so a
#     `<4 x float>` `llvm.fmuladd` is four machine instructions, not one. Every
#     count is therefore also reported width-weighted as `lanes`.
TG_ACCESS = re.compile(r"addrspace\(3\)")
OPCODE = re.compile(r"^\s+(?:%[\w.]+\s*=\s*)?(?:tail\s+)?([a-z][\w.]*)\s")
CALLEE = re.compile(r"@([\w.$]+)\(")
VECTOR_WIDTH = re.compile(r"<(\d+) x ")

FREE_OPS = {"br", "phi", "ret", "switch", "unreachable", "alloca",
            "insertelement", "extractelement", "shufflevector", "bitcast",
            "fence", "landingpad", "indirectbr"}
FLOAT_OPS = {"fadd", "fsub", "fmul", "fdiv", "fneg", "frem", "fcmp"}
INT_OPS = {"add", "sub", "mul", "shl", "lshr", "ashr", "and", "or", "xor",
           "icmp", "select", "sdiv", "udiv", "srem", "urem"}
CONVERT_OPS = {"fpext", "fptrunc", "sitofp", "uitofp", "fptosi", "fptoui",
               "trunc", "zext", "sext", "ptrtoint", "inttoptr"}
ADDRESS_OPS = {"getelementptr", "addrspacecast"}
AIR_CLASSES = ("device_loads", "device_stores", "threadgroup", "shuffles",
               "simd_reduce", "float_alu", "int_alu", "convert", "address",
               "free", "other")


def classify_air(line: str) -> tuple[str, int]:
    """One AIR line to (class, lane count). Lane count is the vector width."""
    hit = OPCODE.match(line)
    if hit is None:
        return ("", 0)
    op = hit.group(1)
    width_hit = VECTOR_WIDTH.search(line)
    lanes = int(width_hit.group(1)) if width_hit else 1
    if op == "call":
        callee = CALLEE.search(line)
        target = callee.group(1) if callee else ""
        if target.startswith("llvm.lifetime") or target.startswith("llvm.dbg"):
            return ("free", 0)
        if "simd_shuffle" in target:
            return ("shuffles", lanes)
        if target.startswith("air.simd_") or "simd_sum" in target:
            return ("simd_reduce", lanes)
        if "fmuladd" in target or target.startswith("air.fma"):
            return ("float_alu", lanes)
        if target.startswith("air.convert"):
            return ("convert", lanes)
        return ("other", lanes)
    if op in ("load", "store"):
        if TG_ACCESS.search(line):
            return ("threadgroup", lanes)
        if "addrspace(1)" in line:
            return ("device_loads" if op == "load" else "device_stores", lanes)
        # addrspace(0) is the stack slot the compiler uses for a spilled or
        # not-yet-promoted array; it is charged to `other`, never to a device
        # access, because calling it a device load would fabricate memory
        # traffic that does not exist.
        return ("other", lanes)
    if op in FREE_OPS:
        return ("free", 0)
    if op in FLOAT_OPS:
        return ("float_alu", lanes)
    if op in CONVERT_OPS:
        return ("convert", lanes)
    if op in INT_OPS:
        return ("int_alu", lanes)
    if op in ADDRESS_OPS:
        return ("address", lanes)
    return ("other", lanes)


def air_categories(body: list[str]) -> dict:
    """Disjoint AIR instruction counts for one entry point.

    `<class>` is the raw line count and `<class>_lanes` is the same count
    weighted by vector width, which is what the machine actually issues.
    """
    counts = {c: 0 for c in AIR_CLASSES}
    lanes = {c: 0 for c in AIR_CLASSES}
    total = 0
    for line in body:
        klass, width = classify_air(line)
        if not klass:
            continue
        total += 1
        counts[klass] += 1
        lanes[klass] += width
    out: dict = {c: counts[c] for c in AIR_CLASSES}
    out.update({c + "_lanes": lanes[c] for c in AIR_CLASSES})
    out["total_instructions"] = total
    out["arithmetic"] = counts["float_alu"] + counts["int_alu"]
    out["arithmetic_lanes"] = lanes["float_alu"] + lanes["int_alu"]
    # What the backend plausibly issues: everything except the SSA bookkeeping
    # the register allocator dissolves.
    out["issue_lanes"] = sum(lanes[c] for c in AIR_CLASSES if c != "free")
    return out


def air_stats(source: pathlib.Path, workdir: pathlib.Path) -> dict:
    """Device loads and shuffle calls per entry point, after -O2.

    AIR is what the front end hands the backend, so a load still present here
    survived every dead-code pass the optimiser runs. The count is the direct
    test of the claim that a broadcast arm removes load instructions.
    """
    ll = workdir / "air.ll"
    done = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", str(source), "-o", str(ll)],
        capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip().splitlines()[-8:]}
    found: dict[str, dict] = {}
    name, body = None, []
    for line in ll.read_text().splitlines():
        if line.startswith("define "):
            match = re.search(r"@([\w.]+)\(", line)
            name, body = (match.group(1) if match else None), []
        elif line == "}" and name is not None:
            hit = KERNEL_RE.search(name)
            if hit:
                cat = air_categories(body)
                # `device_loads` and `shuffles` keep the meaning they had when
                # the screen was first reported, so the earlier numbers in this
                # experiment stay comparable: a shuffle is charged to `shuffles`
                # and never also to `device_loads`.
                cat["air_lines"] = len(body)
                found[hit.group(1)] = cat
            name = None
        elif name is not None:
            body.append(line)
    return found


def census_one(source: pathlib.Path, workdir: pathlib.Path, tag: str) -> dict:
    air_dir = workdir / ("air_" + tag)
    air_dir.mkdir(parents=True, exist_ok=True)
    row: dict = {"air": air_stats(source, air_dir),
                 "threadgroup_bytes": THREADGROUP_BYTES.get(tag, {na: 0
                                                                  for na
                                                                  in WIDTHS}),
                 "threadgroup_budget_bytes": THREADGROUP_BUDGET_BYTES}
    lib = build_metallib(source.read_text(), workdir / tag)
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for kernel, record in translate(lib, arch, workdir / tag).items():
            hit = KERNEL_RE.search(kernel)
            if hit is None:
                continue
            row.setdefault(arch, {})[hit.group(1)] = {
                "registers": record.get("registers"),
                "spill_bytes": record.get("spill_bytes", 0),
                "text_bytes": record.get("text_bytes"),
                "text_sha8": record.get("text_sha8"),
            }
    return row


def census(directory: pathlib.Path, out: pathlib.Path | None) -> int:
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for arm in ARMS:
            rows[arm] = census_one(directory / ("arm_%s.metal" % arm), workdir,
                                   arm)
            print("censused %s" % arm)
        rows["ctl_a_base_via_m"] = census_one(
            directory / "ctl_a_base_via_m.metal", workdir, "ctl_via_m")
        print("censused ctl_a_base_via_m")

    print("\nAIR device loads (and simd_shuffle calls) per entry point")
    print("  %-15s %s" % ("arm", "  ".join("NA%d" % na for na in WIDTHS)))
    for arm in ARMS:
        cells = []
        for na in WIDTHS:
            cell = rows[arm]["air"].get(str(na), {})
            shuffles = cell.get("shuffles", 0)
            cells.append("%4s%s" % (cell.get("device_loads", "?"),
                                    "+%dsh" % shuffles if shuffles else "    "))
        print("  %-15s %s" % (arm, " ".join(cells)))

    for arch in (LOCAL_ARCH, RANKED_ARCH):
        print("\n%s registers / spill / machine text bytes" % arch)
        for arm in ARMS:
            cells = []
            for na in WIDTHS:
                value = rows[arm].get(arch, {}).get(str(na))
                if value is None:
                    cells.append("NA%d=?" % na)
                    continue
                spill = value["spill_bytes"] or 0
                cells.append("NA%d=%s%s/%s" % (
                    na, value["registers"], "s%d" % spill if spill else "",
                    value["text_bytes"]))
            print("  %-15s %s" % (arm, "  ".join(cells)))

    print("\nScaffold control: q_scaffold against a_base (must be identical "
          "machine text)")
    print("Dispatcher control: ctl_a_base_via_m against a_base (must be "
          "identical machine text)")
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for other in ("q_scaffold", "ctl_a_base_via_m"):
            verdict = []
            for na in WIDTHS:
                a = rows["a_base"].get(arch, {}).get(str(na), {})
                b = rows[other].get(arch, {}).get(str(na), {})
                verdict.append("NA%d=%s" % (
                    na, "same" if a.get("text_sha8") == b.get("text_sha8")
                    else "DIFFERENT"))
            print("  %-14s %-18s %s" % (arch, other, " ".join(verdict)))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"widths": list(WIDTHS), "local_arch": LOCAL_ARCH,
             "ranked_arch": RANKED_ARCH, "diagnostic_arms": DIAGNOSTIC_ARMS,
             "promotion_arms": PROMOTION_ARMS, "arms": rows},
            indent=2) + "\n")
        print("\nwrote %s" % out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--arm-list", action="store_true",
                    help="print the probe --arms string and exit, so the "
                         "runner cannot drift out of step with PLANS")
    args = ap.parse_args()
    if args.arm_list:
        print(",".join(a + (":diag" if a in DIAGNOSTIC_ARMS else "")
                       for a in ARMS))
        return 0
    if args.emit is not None:
        emit(args.emit)
    if args.census is not None:
        return census(args.census, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
