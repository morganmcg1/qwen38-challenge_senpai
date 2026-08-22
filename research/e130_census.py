#!/usr/bin/env python3
"""Census the wide-QMV entry point and every body it inlines, per variant.

WHAT THE ENTRY POINT IS. The scored cell is one JIT kernel,
`affine_qmv_fast<bfloat16_t, 64, 4, false>`. The host names it
`affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` and creates ONE pipeline for it
(mlx/backend/metal/quantized.cpp qmv). The draft width M reaches the kernel as
the runtime value `ntg.x`, so every width runs the same pipeline and therefore
the same per-thread register allocation. That allocation must cover every body
the entry point inlines, so it is the MAXIMUM over those bodies.

WHICH BODIES ARE INLINED at group_size 64, bits 4, batched false:

  out_vec_size >= 4096 : pair<T,2>, and _m<T,M,IPG,true> for M in [3,9],
                         which expands to wide<T,NA,true> for NA in {3,4,5}
  out_vec_size <  4096 : pair<T,M> for M in [2,9]
  fallthrough          : qmv_fast_impl<T,64,4>
  bits == 2 branch     : dead at bits == 4, constant-folded away

`wide<T,2,true>` is NOT reachable today: no dispatched width has TAIL == 2.
A census that prices it as a live cell prices a body the kernel never contains.

RESIDENCY IS DERIVED, NOT MEASURED. `resident_simdgroups` is
`floor(register_file_bytes / (128 * registers))`, an arithmetic consequence of
the measured register count. Reporting it back as a verified "floor law" would
be circular, so `--occupancy` adds the one independent channel this host can
supply: `maxTotalThreadsPerThreadgroup` from a real local pipeline, which Apple
caps by the same register allocation. It validates the register channel on
`applegpu_g16s` only. No local channel can measure `applegpu_g17s` residency.

Registers, spill and residency are cost observations and are never correctness
evidence (Rule 73).

    python3 research/e130_census.py --out research/e130-artifacts/rung0.json
    python3 research/e130_census.py --variants base prune_na5_pair --occupancy
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH, RANKED_ARCH, build_metallib, translate,
)
from e104_variant_sources import emit_base  # noqa: E402
from e121_arms import ENTRY_RE, REGISTER_FILE, simdgroups  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The shipped dispatch, read back from the source before it is trusted.
M_IPG = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
WIDE_NAS = (2, 3, 4, 5)
PAIR_MS = tuple(range(2, 10))

# Draft-round weights over the weight-stream group size NA, standing since
# E110 and re-confirmed by E114. Used only to weight residency, never time.
NA_WEIGHT = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

WIDE_KERNEL = """
[[kernel]] void e130_wide%(na)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * %(na)d;
  if (first_m >= %(na)d) {
    return;
  }
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);
}
"""

SCALAR_WIDE_KERNEL = WIDE_KERNEL.replace(
    "e130_wide%(na)d", "e130_swide%(na)d").replace(
    "qmv_fast_crossrow_affine4_g64_wide<",
    "qmv_fast_crossrow_affine4_g64_wide_s<")

PAIR_KERNEL = """
[[kernel]] void e130_pair%(m)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_fast_crossrow_affine4_g64<bfloat16_t, %(m)d>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}
"""

M_KERNEL = """
[[kernel]] void e130_m%(m)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, %(m)d, %(ipg)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}
"""

IMPL_KERNEL = """
[[kernel]] void e130_impl(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_fast_impl<bfloat16_t, 64, 4>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}
"""

PROBE_RE = re.compile(r"e130_(wide\d|swide\d|pair\d|m\d|impl)$")


def dispatch_table(source: str, strict: bool = False) -> dict[int, int]:
    found = {int(m): int(ipg) for m, ipg in re.findall(
        r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>", source)}
    if strict and found != M_IPG:
        raise SystemExit(
            "e130_census: dispatch table changed: %s" % sorted(found.items()))
    return found


def live_bodies(source: str) -> list[str]:
    """Body cells the entry point actually inlines, read from the source."""
    table = {int(m): int(ipg) for m, ipg in re.findall(
        r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>", source)}
    nas = set()
    for m, ipg in table.items():
        # Both arms of the TAIL test are instantiated, but the arm guarded by a
        # compile-time-false condition is deleted before register allocation,
        # so only a reachable arm is a live body.
        nas.add(ipg)
        tail = m % ipg
        if tail != 0:
            nas.add(max(tail, 2))
    pairs = {int(m) for m in re.findall(
        r"qmv_fast_crossrow_affine4_g64<T, (\d+)>", source)}
    names = ["wide%d" % na for na in sorted(nas)]
    if "qmv_fast_crossrow_affine4_g64_wide_s(" in source:
        # The TAIL == 2 call site is routed to the scalar body instead.
        names = [n for n in names if n != "wide2"] + ["swide2"]
    return names + ["pair%d" % m for m in sorted(pairs)] + ["impl"]


def probe_source(source: str) -> str:
    """Append one probe kernel per body cell, plus the shipped M wrappers.

    The M wrappers always use the base dispatch table, so a variant that
    changes one width still reports the same per-width columns.
    """
    text = source
    for na in WIDE_NAS:
        text += WIDE_KERNEL % {"na": na}
    if "qmv_fast_crossrow_affine4_g64_wide_s(" in source:
        text += SCALAR_WIDE_KERNEL % {"na": 2}
    for m in PAIR_MS:
        text += PAIR_KERNEL % {"m": m}
    for m, ipg in sorted(M_IPG.items()):
        text += M_KERNEL % {"m": m, "ipg": ipg}
    return text + IMPL_KERNEL


# --- variants ----------------------------------------------------------------
# Each variant is a text transform of the JIT source. A transform that does not
# find its anchor exactly once raises, so a silent no-op cannot be priced as a
# result.

M5_WIDE = """        case 5:
          qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;"""
M5_PAIR = """        case 5:
          qmv_fast_crossrow_affine4_g64<T, 5>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;"""
M5_IPG3 = """        case 5:
          qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    seen = text.count(old)
    if seen != 1:
        raise SystemExit("e130_census: %s matched %d times, expected 1"
                         % (label, seen))
    return text.replace(old, new)


# --- the occupancy ladder ----------------------------------------------------
# The entry point is allocated against the maximum over the bodies it inlines,
# so ballast added to a body that a given width never executes still lowers
# that width's residency. Ballast in wide<5>, timed at M != 5, therefore moves
# residency with the executed instruction stream held EXACTLY fixed. That is a
# regression discontinuity with no instruction-count covariate to confound it.
#
# The ballast is seeded per lane so it cannot be held in the uniform register
# file, and it is consumed only under `out_vec_size < 0`, which the compiler
# cannot prove and the host never supplies, so no output changes.

BALLAST_ANCHOR_INIT = """  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }
"""
BALLAST_INIT = """  constexpr int E130_BALLAST = (NA == 5) ? %d : 0;
  float e130_ballast[E130_BALLAST > 0 ? E130_BALLAST : 1];
  for (int b = 0; b < E130_BALLAST; b++) {
    e130_ballast[b] = float(int(simd_lid) + b);
  }
"""
BALLAST_ANCHOR_STEP = """    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }
"""
BALLAST_STEP = """    for (int b = 0; b < E130_BALLAST; b++) {
      e130_ballast[b] = fma(e130_ballast[b], 1.0f, float(k));
    }
  }
"""
BALLAST_ANCHOR_SINK = """  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      const float reduced = simd_sum(acc[r][m]);"""
BALLAST_SINK = """  if (E130_BALLAST > 0 && out_vec_size < 0) {
    for (int b = 0; b < E130_BALLAST; b++) {
      y[b] = static_cast<T>(e130_ballast[b]);
    }
  }
"""


def ballast(count: int):
    def apply(text: str) -> str:
        text = replace_once(text, BALLAST_ANCHOR_INIT,
                            BALLAST_ANCHOR_INIT + BALLAST_INIT % count,
                            "ballast init anchor")
        text = replace_once(
            text, BALLAST_ANCHOR_STEP,
            BALLAST_ANCHOR_STEP[:-len("  }\n")] + BALLAST_STEP,
            "ballast step anchor")
        return replace_once(text, BALLAST_ANCHOR_SINK,
                            BALLAST_SINK + BALLAST_ANCHOR_SINK,
                            "ballast sink anchor")
    return apply


BALLAST_RUNGS = (0, 2, 3, 4, 6, 7, 9, 10, 12)

# --- the downward ladder, which is the one that works on this host -----------
# Ballast can only raise the allocation, and `applegpu_g16s` caps a kernel at
# 96 registers and spills past it, so upward ballast never moves local
# residency: 94, 95 and 96 registers all hold 32 simdgroups. The usable local
# ladder therefore goes DOWN. Widths 3, 6 and 9 execute wide<3> only, so
# removing wide<4> and wide<5> from the kernel changes those widths' residency
# while leaving the instructions they execute bit-identical. A synthetic body
# behind an unreachable switch case then sets any allocation between the
# pruned floor and the cap, one register at a time.

PRUNE_WIDE45 = {
    "qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true>(":
        "qmv_fast_crossrow_affine4_g64<T, 4>(",
    "qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>(":
        "qmv_fast_crossrow_affine4_g64<T, 5>(",
    "qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>(":
        "qmv_fast_crossrow_affine4_g64<T, 7>(",
    "qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>(":
        "qmv_fast_crossrow_affine4_g64<T, 8>(",
}

REG_BODY = """
template <typename T, int B>
METAL_FUNC void e130_reg_ballast(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  float v[B];
  for (int i = 0; i < B; i++) {
    v[i] = float(int(simd_lid) + i) + float(int(tid.x));
  }
  for (int k = 0; k < in_vec_size; k += 512) {
    for (int i = 0; i < B; i++) {
      v[i] = fma(v[i], 1.0009765625f, float(k));
    }
  }
  float s = 0.0f;
  for (int i = 0; i < B; i++) {
    s += v[i];
  }
  if (out_vec_size < 0) {
    y[int(simd_gid)] = static_cast<T>(s);
  }
}

"""
REG_BODY_ANCHOR = ("template <typename T, int group_size, int bits, "
                   "bool batched>\n[[kernel]] void affine_qmv_fast(")

CASE9_ANCHOR = """        case 9:
          qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
"""
CASE31 = """        case 31:
          e130_reg_ballast<T, %d>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
"""


def prune_wide45(text: str) -> str:
    for old, new in PRUNE_WIDE45.items():
        text = replace_once(text, old, new, "prune %s" % old[:44])
    return text


def rung(width: int | None):
    """Prune wide<4> and wide<5>, then set the allocation with a dead body."""
    def apply(text: str) -> str:
        text = prune_wide45(text)
        if width is None:
            return text
        text = replace_once(text, REG_BODY_ANCHOR, REG_BODY + REG_BODY_ANCHOR,
                            "reg ballast body anchor")
        return replace_once(text, CASE9_ANCHOR, CASE9_ANCHOR + CASE31 % width,
                            "unreachable case anchor")
    return apply


REG_RUNGS = (0,) + tuple(range(56, 116))

# The six spill-free applegpu_g16s treads the ballast can reach, named by the
# entry register count they produce. Every one executes the SAME instructions at
# M = 3, 6 and 9, and the entry ISA text grows by exactly 38 bytes per ballast
# register, so nothing outside the dead body moves.
LADDER = {"l82": 78, "l84": 81, "l86": 83, "l88": 85, "l91": 88, "l94": 91}

# --- C-C: make the NA = 5 body cheaper instead of retiring it ----------------
# `vec<float,5>` is the only object in the wide body whose width is NA, and the
# cross-architecture asymmetry sits exactly there: NA=5 costs 101 registers on
# g17s but 93 on g16s, while NA=4 costs 90 and 94. Scalarising the vector into
# `float[NA]` performs the SAME elementwise operations in the SAME order, so it
# is bit-exact by construction, and it removes any vector-width rounding the
# g17s allocator may apply. Loading the group metadata after the inner block
# shortens the live range of eight more values without moving any arithmetic.

SCALARISE = [
    ("""  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }
""", """  float acc[rows_per_simd][NA];
  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      acc[r][m] = 0.0f;
    }
  }
"""),
    ("""    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
""", """    float sums[NA];
    for (int m = 0; m < NA; m++) {
      sums[m] = 0.0f;
    }
    float partial[rows_per_simd][NA];
    for (int r = 0; r < rows_per_simd; r++) {
      for (int m = 0; m < NA; m++) {
        partial[r][m] = 0.0f;
      }
    }
"""),
    ("      VF a0, a1, a2, a3;\n", "      float a0[NA], a1[NA], a2[NA], a3[NA];\n"),
    ("""        if (DIRECT_NIBBLES) {
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
""", """        for (int m = 0; m < NA; m++) {
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
"""),
    ("      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];\n",
     """      for (int m = 0; m < NA; m++) {
        acc[r][m] += scale_local[r] * partial[r][m] + sums[m] * bias_local[r];
      }
"""),
]

META_OLD = """      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }
"""
META_NEW = """    }
"""
META_LATE_ANCHOR = "    for (int r = 0; r < rows_per_simd; r++) {\n      acc[r]"
META_LATE = """    for (int r = 0; r < rows_per_simd; r++) {
      const int group_index =
          (out_row + r) * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }
"""


def scalarise(text: str) -> str:
    for old, new in SCALARISE:
        text = replace_once(text, old, new, "scalarise %s" % old.strip()[:40])
    return text


def late_meta(text: str) -> str:
    text = replace_once(text, META_OLD, META_NEW, "metadata load site")
    return replace_once(text, META_LATE_ANCHOR, META_LATE + META_LATE_ANCHOR,
                        "metadata reload site")

# --- selective source form: one body cell keeps its own arithmetic form ------
# Scalarising every width is a net loss because NA = 4 regresses. Scalarising a
# single width needs two coexisting templates and a compile-time selector, so
# only the selected instantiation is emitted and the other width columns keep
# their base allocation.

WIDE_HEAD = ("template <typename T, int NA, bool DIRECT_NIBBLES = false>\n"
             "METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide(")

TAIL_CALL = """    qmv_fast_crossrow_affine4_g64_wide<
        T, (TAIL >= 2 ? TAIL : 2), DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
"""
TAIL_CALL_NEW = """    e130_wide_tail<T, (TAIL >= 2 ? TAIL : 2), DIRECT_NIBBLES>::run(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
"""

TAIL_ARGS = """      const device uint32_t* w,
      const device T* scales,
      const device T* biases,
      const device T* x,
      device T* y,
      const int in_vec_size,
      const int out_vec_size,
      int first_m,
      int out_row,
      uint simd_lid"""

TAIL_SELECTOR = """
template <typename T, int NA, bool DIRECT_NIBBLES>
struct e130_wide_tail {
  static METAL_FUNC void run(
%(args)s) {
    qmv_fast_crossrow_affine4_g64_wide<T, NA, DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
  }
};

template <typename T, bool DIRECT_NIBBLES>
struct e130_wide_tail<T, 2, DIRECT_NIBBLES> {
  static METAL_FUNC void run(
%(args)s) {
    qmv_fast_crossrow_affine4_g64_wide_s<T, 2, DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
  }
};

""" % {"args": TAIL_ARGS}

M_HEAD = "template <typename T, int M, int IPG, bool DIRECT_NIBBLES = false>\n"


def extract_function(text: str, head: str) -> str:
    start = text.index(head)
    depth = 0
    i = text.index("{", start)
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1


def scalar_tail2(text: str) -> str:
    """Add a scalarised NA == 2 body and route only the TAIL == 2 call to it."""
    body = extract_function(text, WIDE_HEAD)
    scalar = scalarise(body).replace(
        "qmv_fast_crossrow_affine4_g64_wide(",
        "qmv_fast_crossrow_affine4_g64_wide_s(")
    text = replace_once(text, M_HEAD, scalar + "\n\n" + TAIL_SELECTOR + M_HEAD,
                        "scalar tail body")
    return replace_once(text, TAIL_CALL, TAIL_CALL_NEW, "tail call site")


VARIANTS = {
    # The current base, unchanged.
    "base": lambda t: t,
    # Route the only NA == 5 dispatch to the pair kernel the small-N branch
    # already instantiates, so the entry point loses the wide<5> body without
    # gaining any new body.
    "prune_na5_pair": lambda t: replace_once(t, M5_WIDE, M5_PAIR, "M5 -> pair"),
    # Route it to a 3 + 2 wide split instead, which drops wide<5> but adds
    # wide<2>.
    "prune_na5_ipg3": lambda t: replace_once(t, M5_WIDE, M5_IPG3, "M5 -> IPG 3"),
    # C-C: keep <T,5,5,true> and make its body cheaper, bit-exactly.
    "cc_scalarise": scalarise,
    "cc_late_meta": late_meta,
    "cc_both": lambda t: scalarise(late_meta(t)),
    # C-A plus a scalarised NA == 2 tail, which is the only body C-A adds.
    "ca_scalar_tail2": lambda t: scalar_tail2(
        replace_once(t, M5_WIDE, M5_IPG3, "M5 -> IPG 3")),
}
VARIANTS.update({"ballast%d" % n: ballast(n) for n in BALLAST_RUNGS if n})
VARIANTS.update({"rung%d" % n: rung(n or None) for n in REG_RUNGS})
VARIANTS.update({name: rung(b) for name, b in LADDER.items()})


def census(name: str, source: str, workdir: pathlib.Path) -> dict:
    probe = probe_source(source)
    lib = build_metallib(probe, workdir / name)
    row: dict = {"source_bytes": len(source), "probe_bytes": len(probe),
                 "live_bodies": live_bodies(source), "cells": {}}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for kernel, record in translate(lib, arch, workdir / name).items():
            hit = PROBE_RE.search(kernel)
            if hit is not None:
                key = hit.group(1)
            elif ENTRY_RE.search(kernel):
                key = "entry"
            else:
                continue
            registers = record.get("registers")
            row["cells"].setdefault(arch, {})[key] = {
                "registers": registers,
                "spill_bytes": record.get("spill_bytes", 0),
                "text_bytes": record.get("text_bytes"),
                "text_sha8": record.get("text_sha8"),
                "resident_simdgroups_derived":
                    simdgroups(arch, registers) if registers else None,
            }
    row["metallib"] = str(lib)
    return row


def occupancy(lib: pathlib.Path, workdir: pathlib.Path) -> dict:
    """maxTotalThreadsPerThreadgroup per kernel from a real local pipeline."""
    binary = workdir / "occupancy"
    if not binary.exists():
        subprocess.run(
            ["swiftc", "-O", str(ROOT / "research/crossrow_na_occupancy.swift"),
             "-o", str(binary)], check=True, capture_output=True)
    done = subprocess.run([str(binary), str(lib)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip()[-400:]}
    found = {}
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1].isdigit():
            hit = PROBE_RE.search(parts[0])
            key = hit.group(1) if hit else (
                "entry" if ENTRY_RE.search(parts[0]) else None)
            if key:
                found[key] = {"max_threads": int(parts[1]),
                              "exec_width": int(parts[2])}
    return found


def audit(rows: dict, occ: dict) -> dict:
    """The two identities rung 0 must decide, plus the honest caveat."""
    out: dict = {"per_arch": {}, "notes": []}
    out["notes"].append(
        "resident_simdgroups_derived is floor(register_file_bytes / (128 * "
        "registers)); it is arithmetic on the measured register count, not an "
        "independent occupancy measurement")
    base = rows["base"]
    live = base["live_bodies"]
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        cells = base["cells"][arch]
        entry = cells["entry"]["registers"]
        live_regs = {k: cells[k]["registers"] for k in live if k in cells}
        peak = max(live_regs.values())
        argmax = sorted(k for k, v in live_regs.items() if v == peak)
        dead = {k: cells[k]["registers"] for k in cells
                if k not in live and k != "entry"}
        out["per_arch"][arch] = {
            "entry_registers": entry,
            "live_body_registers": live_regs,
            "max_live_body_registers": peak,
            "argmax_live_bodies": argmax,
            "entry_equals_max_live_body": entry == peak,
            "not_inlined_cells": dead,
            "entry_resident_simdgroups":
                cells["entry"]["resident_simdgroups_derived"],
            "spill_free": all(
                (cells[k]["spill_bytes"] or 0) == 0 for k in cells),
        }
    if occ:
        out["local_occupancy_channel"] = occ
        pairs = []
        for key, record in sorted(occ.items()):
            regs = base["cells"][LOCAL_ARCH].get(key, {}).get("registers")
            if regs:
                pairs.append((key, regs, record["max_threads"]))
        out["local_occupancy_pairs"] = [
            {"cell": k, "registers": r, "max_threads": t} for k, r, t in pairs]
        # Apple caps maxTotalThreadsPerThreadgroup by the same allocation, so a
        # monotone non-increasing relation with register count is the strongest
        # independent statement this host can make about the register channel.
        ordered = sorted(pairs, key=lambda p: p[1])
        out["local_occupancy_monotone"] = all(
            ordered[i][2] >= ordered[i + 1][2] for i in range(len(ordered) - 1))
    return out


def weighted_residency(rows: dict, name: str, arch: str) -> dict:
    """Residency change at the entry point, weighted over the NA rounds.

    One pipeline serves every width, so the entry allocation is what every
    round pays and the weighted figure is the uniform change. It is written as
    a weighted sum anyway so a future per-width pipeline could reuse the same
    reporting shape.
    """
    before = rows["base"]["cells"][arch]["entry"]["resident_simdgroups_derived"]
    after = rows[name]["cells"][arch]["entry"]["resident_simdgroups_derived"]
    per_cell = {na: 100.0 * (after - before) / before for na in NA_WEIGHT}
    total = sum(NA_WEIGHT[na] * per_cell[na] for na in NA_WEIGHT)
    return {
        "entry_sg_before": before,
        "entry_sg_after": after,
        "per_cell_pct": per_cell,
        "weighted_pct": total / sum(NA_WEIGHT.values()),
        "min_cell_pct": min(per_cell.values()),
    }


def report(rows: dict, occ: dict, out: pathlib.Path | None) -> int:
    keys = ["entry", "wide2", "wide3", "wide4", "wide5"] + \
        ["pair%d" % m for m in PAIR_MS] + ["impl"] + \
        ["m%d" % m for m in sorted(M_IPG)]
    for name, row in rows.items():
        print("\n=== variant %s ===" % name)
        print("live bodies: %s" % ", ".join(row["live_bodies"]))
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            print("  %s  (register file %d bytes, %d simdgroup slots)" % (
                arch, REGISTER_FILE[arch], REGISTER_FILE[arch] // 128))
            for key in keys:
                cell = row["cells"][arch].get(key)
                if cell is None:
                    continue
                live = "live" if (key in row["live_bodies"]
                                  or key == "entry") else "dead"
                print("    %-7s %-4s regs %3d  spill %d  text %6d  sg %2d" % (
                    key, live, cell["registers"], cell["spill_bytes"] or 0,
                    cell["text_bytes"], cell["resident_simdgroups_derived"]))

    verdict = audit(rows, occ)
    print("\n=== rung 0 audit ===")
    for arch, record in verdict["per_arch"].items():
        print("  %s: entry %d regs, max live body %d (%s) -> entry==max %s, "
              "spill-free %s, entry sg %d" % (
                  arch, record["entry_registers"],
                  record["max_live_body_registers"],
                  ",".join(record["argmax_live_bodies"]),
                  record["entry_equals_max_live_body"], record["spill_free"],
                  record["entry_resident_simdgroups"]))
        print("    not inlined by the entry point: %s" % record[
            "not_inlined_cells"])
    if "local_occupancy_pairs" in verdict:
        print("  local independent channel (maxTotalThreadsPerThreadgroup):")
        for record in verdict["local_occupancy_pairs"]:
            print("    %-7s regs %3d  max_threads %4d" % (
                record["cell"], record["registers"], record["max_threads"]))
        print("    monotone in register count: %s"
              % verdict["local_occupancy_monotone"])

    gains = {}
    for name in rows:
        if name == "base":
            continue
        gains[name] = {arch: weighted_residency(rows, name, arch)
                       for arch in (LOCAL_ARCH, RANKED_ARCH)}
        print("\n  variant %s residency vs base:" % name)
        for arch, record in gains[name].items():
            print("    %s  entry sg %d -> %d  weighted %+.2f %%  worst cell "
                  "%+.2f %%" % (arch, record["entry_sg_before"],
                                record["entry_sg_after"],
                                record["weighted_pct"],
                                record["min_cell_pct"]))

    payload = {
        "harness": "local",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "base_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
            capture_output=True, text=True).stdout.strip(),
        "register_file_bytes": dict(REGISTER_FILE),
        "na_weights": NA_WEIGHT,
        "dispatch_table": M_IPG,
        "variants": {k: {x: y for x, y in v.items() if x != "metallib"}
                     for k, v in rows.items()},
        "audit": verdict,
        "residency_vs_base": gains,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print("\nwrote %s" % out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--keep", type=pathlib.Path)
    ap.add_argument("--occupancy", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["base"])
    ap.add_argument("--emit-arms", type=pathlib.Path,
                    help="write each variant's JIT source as an A/B arm")
    args = ap.parse_args()
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit("e130_census: unknown variants %s" % unknown)
    if "base" not in args.variants:
        args.variants = ["base"] + list(args.variants)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = args.keep or pathlib.Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        base = emit_base(workdir / "base.metal")
        dispatch_table(base, strict=True)
        rows, occ = {}, {}
        if args.emit_arms is not None:
            args.emit_arms.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(args.variants):
            source = VARIANTS[name](base)
            if args.emit_arms is not None:
                (args.emit_arms / ("arm_%s.metal" % name)).write_text(source)
            rows[name] = census(name, source, workdir)
            rows[name]["arm_index"] = index
            print("censused %s" % name)
        if args.emit_arms is not None:
            (args.emit_arms / "manifest.json").write_text(json.dumps(
                {"arms": args.variants,
                 "function": "affine_qmv_fast_bfloat16_t_64_4_false"},
                indent=2) + "\n")
        if args.occupancy:
            occ = occupancy(pathlib.Path(rows["base"]["metallib"]), workdir)
        return report(rows, occ, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
