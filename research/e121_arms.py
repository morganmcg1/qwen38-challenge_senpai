#!/usr/bin/env python3
"""E121: share the activation chunk-sum term across the two simdgroups.

    research/e121_arms.py --emit /tmp/e121-arms
    research/e121_arms.py --census /tmp/e121-arms --out research/out/e121/x.json

Both simdgroups of the wide-QMV threadgroup see the identical `first_m`, the
identical `simd_lid` and the identical `k`, so they compute bit-identical
`sums[m]` values. Every one of them is computed exactly twice. Only the add
tree is redundant: `partial[r]` consumes all NA components of `a0..a3`, so both
simdgroups must keep loading and converting every activation row.

Arms, all on the isolated per-width entry points so each width gets its own
register allocation:

  a_base       the shipped wide kernel, unmodified. Contains xv4.
  a_scaffold   the same body under the E121 signature, which adds `simd_gid`
               and the exchange pointer, plus the threadgroup allocation in the
               entry point. Isolates the signature and the allocation. Must be
               bit exact and should be time-neutral.
  b_barrier2   `a_scaffold` plus the two threadgroup barriers per k-block that
               the exchange needs, and nothing else. Prices the barrier pair.
  n_halfsums   each simdgroup computes only its own half of `sums` and the
               halves are NEVER exchanged, so the result is deliberately WRONG.
               It must fail the bit-exactness check. This is the ceiling, and
               it uses the predicated loop so that it differs from
               `x_split_pred` by the exchange alone.
  x_split_dup  duplicated `i` loop under `if (simd_gid == 0)`, single exchange
               buffer, two barriers per k-block.
  x_split_pred one `i` loop with the `sums[m] +=` line predicated on a uniform
               `simd_gid`-derived condition. Minimum instruction text.
  x_split_pp   `x_split_dup` with a double-buffered exchange, which drops the
               second barrier.
  x_split_pred_pp
               the predicated loop with the double-buffered exchange.

Exchange layout is `[m][lane]`, lane fastest, so the 32 lanes of a simdgroup
touch 32 consecutive floats and no bank conflict is possible. The buffer is
`NA * 32` floats (640 B at NA = 5) for the single-buffer arms and twice that
for the ping-pong arm, against a 32,768 B threadgroup pool. That is the
structural difference from E110's `xs_stage`, whose NA x 1024 B tile capped
residency below the ranked operating point.

`n_halfsums` changes the answer on purpose and is a timing-only diagnostic.
Every other arm must reproduce `a_base` bit for bit at every cell, and the
harness runs a positive control that proves the check can fail.

Research-only: nothing here is on the scored path. The winning body is
transplanted into `Vendor/.../kernels/quantized.h` and its generated twin.
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
    emit_base, wide_fn_span, widen_asserts,
)

WIDTHS = (2, 3, 4, 5)
TG_MEMORY_BYTES = 32768
REGISTER_FILE = {LOCAL_ARCH: 384 * 1024, RANKED_ARCH: 496 * 1024}

# --- shared scaffold ---------------------------------------------------------
# The E121 signature: `simd_gid` selects the owned half and `sums_xchg` is the
# exchange buffer the entry point allocates. `DIRECT_NIBBLES == true` is the
# branch every scored cell takes, so the scaffold carries it unconditionally,
# exactly as E104 and E110 did.

PROLOGUE = """template <typename T, int NA, bool DIRECT_NIBBLES = false>
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
    uint simd_gid,
    uint simd_lid,
    threadgroup float* sums_xchg) {
  static_assert(NA >= 2 && NA <= 8, "e121 probe admits NA in [2, 8]");
  typedef vec<float, NA> VF;
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  constexpr int H = NA / 2;
  constexpr int HA = (NA + 1) / 2;
  const bool own_lo = simd_gid == 0;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread uint16_t packed[rows_per_simd][4];
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
    }
"""

PROLOGUE_PP = PROLOGUE.replace(
    """  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }
""",
    """  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }

  int par = 0;
""")

EPILOGUE = """  }

  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] =
            static_cast<T>(reduced);
      }
    }
  }
}
"""

# The shipped k-block body, which is the xv4 form merged at 2127858b.
LOOP_BASE = """    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        const vec<T, 4> xv = *reinterpret_cast<const device vec<T, 4>*>(xm);
        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
        a0[m] = static_cast<float>(xv[0]);
        a1[m] = static_cast<float>(xv[1]);
        a2[m] = static_cast<float>(xv[2]);
        a3[m] = static_cast<float>(xv[3]);
      }
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
"""


def loop_owned(condition: str, indent: str = "    ") -> str:
    """The k-block `i` loop with the chunk-sum term guarded by `condition`.

    `m` is a fully unrolled index, so a condition written in `m` and the
    constexpr `H` folds at compile time and the guarded adds disappear from the
    instruction stream. A condition written in `own_lo` stays a uniform runtime
    value and the backend chooses predication or a branch.
    """
    body = LOOP_BASE.replace(
        "        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];\n",
        "        if (%s) {\n"
        "          sums[m] += xv[0] + xv[1] + xv[2] + xv[3];\n"
        "        }\n" % condition,
    )
    if indent == "    ":
        return body
    return "".join(
        (indent[4:] + line if line.strip() else line)
        for line in body.splitlines(keepends=True)
    )


HEAD = """    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
"""

TAIL = """    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
"""

BARRIER = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"


def exchange(store: str, load: str, buffer_expr: str = "sums_xchg") -> str:
    """Store the owned half, barrier, read the other half, barrier."""
    return (
        "    for (int m = 0; m < NA; m++) {\n"
        "      if (%s) {\n"
        "        %s[m * SIMD_SIZE + simd_lid] = sums[m];\n"
        "      }\n"
        "    }\n" % (store, buffer_expr)
        + BARRIER
        + "    for (int m = 0; m < NA; m++) {\n"
        "      if (%s) {\n"
        "        sums[m] = %s[m * SIMD_SIZE + simd_lid];\n"
        "      }\n"
        "    }\n" % (load, buffer_expr)
    )


BODY_SCAFFOLD = HEAD + LOOP_BASE + TAIL
BODY_BARRIER2 = HEAD + LOOP_BASE + BARRIER + BARRIER + TAIL

# n_halfsums: the ceiling. Each simdgroup keeps only its own half and the
# halves are never exchanged, so `sums` is incomplete for the other half and
# the output is wrong on purpose.
#
# It uses the predicated loop, not the duplicated one, so it differs from
# `x_split_pred` by exactly the exchange and its barriers. That makes
# `a_base - n_halfsums` the ceiling and `n_halfsums - x_split_pred` the price
# of the exchange, both read inside one register-allocation family. A
# duplicated-branch ceiling would instead confound the removed adds with the
# doubled loop text that the rung-0 census already prices.
BODY_HALFSUMS = HEAD + loop_owned("(m < H) == own_lo") + TAIL

# x_split_dup: duplicated `i` loop under a uniform branch, so the guarded adds
# fold away statically. One shared buffer, two barriers per k-block: the first
# publishes this k-block, the second protects it from the next k-block's store.
BODY_SPLIT_DUP = (
    HEAD
    + "    if (own_lo) {\n"
    + loop_owned("m < H", indent="      ")
    + "    } else {\n"
    + loop_owned("m >= H", indent="      ")
    + "    }\n"
    + exchange("(m < H) == own_lo", "(m < H) != own_lo")
    + BARRIER
    + TAIL
)

# x_split_pred: one `i` loop, the chunk-sum line predicated on a uniform value.
BODY_SPLIT_PRED = (
    HEAD
    + loop_owned("(m < H) == own_lo")
    + exchange("(m < H) == own_lo", "(m < H) != own_lo")
    + BARRIER
    + TAIL
)

# x_split_pp: `x_split_dup` with parity-alternating buffers. The next k-block
# writes the other half of the buffer, so the write-after-read barrier goes.
BODY_SPLIT_PP = (
    HEAD
    + "    if (own_lo) {\n"
    + loop_owned("m < H", indent="      ")
    + "    } else {\n"
    + loop_owned("m >= H", indent="      ")
    + "    }\n"
    + exchange("(m < H) == own_lo", "(m < H) != own_lo",
               "(sums_xchg + par * NA * SIMD_SIZE)")
    + "    par ^= 1;\n"
    + TAIL
)

# x_split_pred_pp: the predicated loop with the parity-alternating buffer, so
# the cheap branch form also drops the write-after-read barrier. On the rung-0
# census this is the only arm that keeps base-like register pressure while
# paying one barrier per k-block.
BODY_SPLIT_PRED_PP = (
    HEAD
    + loop_owned("(m < H) == own_lo")
    + exchange("(m < H) == own_lo", "(m < H) != own_lo",
               "(sums_xchg + par * NA * SIMD_SIZE)")
    + "    par ^= 1;\n"
    + TAIL
)

# --- askeladd's x_sumshare_min, ported onto the xv4 base ---------------------
# E118 built its arms on the pre-xv4 body, where the add tree re-reads `xm[0..3]`
# from device memory. Duplicating only the add tree there needs an explicit
# `T xv[NA][4]` register cache, or the duplicated arm re-issues the loads. This
# base already holds the four values in one `vec<T, 4>`, so the port keeps his
# structure -- values cached, add tree duplicated behind a uniform `simd_gid`
# branch, never predicated inside the `m` loop -- while sourcing the cache from
# the single vector load. His `H = (NA + 1) / 2` and his `[sg][m][lane]`
# exchange are preserved so the comparison is against his arm, not a variant.
ASK_LOOP = """    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      vec<T, 4> xvs[NA];
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        xvs[m] = *reinterpret_cast<const device vec<T, 4>*>(xm);
        a0[m] = static_cast<float>(xvs[m][0]);
        a1[m] = static_cast<float>(xvs[m][1]);
        a2[m] = static_cast<float>(xvs[m][2]);
        a3[m] = static_cast<float>(xvs[m][3]);
      }
      if (own_lo) {
        for (int m = 0; m < HA; m++) {
          sums[m] += xvs[m][0] + xvs[m][1] + xvs[m][2] + xvs[m][3];
        }
      } else {
        for (int m = HA; m < NA; m++) {
          sums[m] += xvs[m][0] + xvs[m][1] + xvs[m][2] + xvs[m][3];
        }
      }
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
"""

ASK_XCHG = """    for (int m = 0; m < NA; m++) {
      if (own_lo == (m < HA)) {
        sums_xchg[(int(simd_gid) * NA + m) * SIMD_SIZE + int(simd_lid)] =
            sums[m];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int m = 0; m < NA; m++) {
      if (own_lo != (m < HA)) {
        sums[m] = sums_xchg[((1 - int(simd_gid)) * NA + m) * SIMD_SIZE +
                            int(simd_lid)];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

BODY_MIN_ASK = HEAD + ASK_LOOP + ASK_XCHG + TAIL


def reindent(text: str, extra: str = "  ") -> str:
    return "".join(
        (extra + line if line.strip() else line)
        for line in text.splitlines(keepends=True)
    )


def gated(shared: str) -> str:
    """Run `shared` only where it pays; fall back to the incumbent elsewhere.

    E118 measured `x_sumshare_min` at +1.465 % at NA=4 and +0.024 % standing,
    because it spills at NA=5 and loses 31 % there. NA=5 carries 0.034 of the
    round weight. `NA` is a template parameter, so this branch is resolved at
    compile time and costs no runtime instruction.
    """
    return (
        HEAD
        + "    if constexpr (NA <= 4) {\n"
        + reindent(shared)
        + "    } else {\n"
        + reindent(LOOP_BASE)
        + "    }\n"
        + TAIL
    )


BODIES = {
    "a_base": None,
    "a_scaffold": (PROLOGUE, BODY_SCAFFOLD, EPILOGUE),
    "b_barrier2": (PROLOGUE, BODY_BARRIER2, EPILOGUE),
    "n_halfsums": (PROLOGUE, BODY_HALFSUMS, EPILOGUE),
    "x_split_dup": (PROLOGUE, BODY_SPLIT_DUP, EPILOGUE),
    "x_split_pred": (PROLOGUE, BODY_SPLIT_PRED, EPILOGUE),
    "x_split_pp": (PROLOGUE_PP, BODY_SPLIT_PP, EPILOGUE),
    "x_split_pred_pp": (PROLOGUE_PP, BODY_SPLIT_PRED_PP, EPILOGUE),
    "x_min_ask": (PROLOGUE, BODY_MIN_ASK, EPILOGUE),
    "g_min_ask": (PROLOGUE, gated(ASK_LOOP + ASK_XCHG), EPILOGUE),
    "g_split_pred": (
        PROLOGUE,
        gated(loop_owned("(m < H) == own_lo")
              + exchange("(m < H) == own_lo", "(m < H) != own_lo")
              + BARRIER),
        EPILOGUE),
    "g_split_pred_pp": (
        PROLOGUE_PP,
        gated(loop_owned("(m < H) == own_lo")
              + exchange("(m < H) == own_lo", "(m < H) != own_lo",
                         "(sums_xchg + par * NA * SIMD_SIZE)")
              + "    par ^= 1;\n"),
        EPILOGUE),
}

# Buffer depth in units of NA * SIMD_SIZE floats. Askeladd's layout gives each
# simdgroup its own NA-wide region, so it needs two.
BUFFERS = {"x_split_pp": 2, "x_split_pred_pp": 2, "g_split_pred_pp": 2,
           "x_min_ask": 2, "g_min_ask": 2}

# A gated arm never touches the buffer at NA = 5, so it allocates for NA = 4.
GATED = ("g_min_ask", "g_split_pred", "g_split_pred_pp")

EXACT_ARMS = ("a_scaffold", "b_barrier2", "x_split_dup", "x_split_pred",
              "x_split_pp", "x_split_pred_pp", "x_min_ask", "g_min_ask",
              "g_split_pred", "g_split_pred_pp")

# Rung 0 censused every form. The timed set drops the ungated ping-pong, whose
# barrier question `g_split_pred_pp` already answers, and keeps `x_min_ask` as
# the guaranteed-skip reference: its uniform branch is real control flow, while
# the predicated form may compile to a masked add that still issues.
#
# The probe runs its positive control on the last arm, so the last arm must be
# one that is required to be bit exact. `g_split_pred` is last because it is
# the primary candidate.
ARMS = ("a_base", "a_scaffold", "x_split_pred", "x_min_ask", "g_min_ask",
        "g_split_pred_pp", "g_split_pred")

BASE_KERNEL = """
[[kernel]] void e121_iso_na%(na)d(
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
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      int(tid.x) * %(na)d, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);
}
"""

ARM_KERNEL = """
[[kernel]] void e121_iso_na%(na)d(
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
  threadgroup float sums_xchg[%(slots)d * 32];
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      int(tid.x) * %(na)d, int(tid.y) * 8 + int(simd_gid) * 4, simd_gid,
      simd_lid, sums_xchg);
}
"""


# --- the shipped dispatch, carried by every arm ------------------------------
# The wide kernel is reached only through `qmv_fast_crossrow_affine4_g64_m`,
# which is reached only from the `affine_qmv_fast` entry point. Both must carry
# the exchange buffer, and the buffer must be declared ONCE in the entry point:
# Metal allocates threadgroup memory statically per pipeline, so a declaration
# inside the wide template would be allocated per inlined instantiation.

M_MARKER = "  static_assert(M % IPG != 1,"
M_SIGNATURE_OLD = "    uint simd_lid) {\n"
M_SIGNATURE_NEW = "    uint simd_lid,\n    threadgroup float* sums_xchg) {\n"

WIDE_CALL_OLD = """        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);"""
WIDE_CALL_NEW = """        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_gid, simd_lid, sums_xchg);"""

M_DISPATCH_CALL = "qmv_fast_crossrow_affine4_g64_m<T, "
M_CALL_TAIL = "tid, simd_gid, simd_lid);"
M_CALL_TAIL_NEW = "tid, simd_gid, simd_lid, sums_xchg);"

ENTRY_ANCHOR = """  if (!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024) {
    if (out_vec_size >= 4096) {"""


def patch_dispatch(text: str, depth: int, max_na: int = 5) -> str:
    """Thread the exchange buffer from the entry point to the wide kernel."""
    marker = text.count(M_MARKER)
    if marker != 1:
        raise SystemExit("e121_arms: dispatch marker matched %d times" % marker)
    at = text.rindex(M_SIGNATURE_OLD, 0, text.index(M_MARKER))
    text = text[:at] + M_SIGNATURE_NEW + text[at + len(M_SIGNATURE_OLD):]
    seen = text.count(WIDE_CALL_OLD)
    if seen != 2:
        raise SystemExit(
            "e121_arms: wide call patch matched %d times, expected 2" % seen)
    text = text.replace(WIDE_CALL_OLD, WIDE_CALL_NEW)

    # Only the dispatcher's own call sites take the buffer; the pair kernel and
    # the narrow paths share the same trailing argument text.
    calls = 0
    at = 0
    while True:
        at = text.find(M_DISPATCH_CALL, at)
        if at < 0:
            break
        end = text.index(M_CALL_TAIL, at)
        text = text[:end] + M_CALL_TAIL_NEW + text[end + len(M_CALL_TAIL):]
        at = end
        calls += 1
    if calls != 7:
        raise SystemExit(
            "e121_arms: dispatcher call sites matched %d, expected 7" % calls)
    seen = text.count(ENTRY_ANCHOR)
    if seen != 1:
        raise SystemExit("e121_arms: entry anchor matched %d times" % seen)
    return text.replace(
        ENTRY_ANCHOR,
        "  threadgroup float sums_xchg[%d * %d * 32];\n" % (depth, max_na)
        + ENTRY_ANCHOR)


def arm_source(base: str, arm: str) -> str:
    plan = BODIES[arm]
    if plan is None:
        return base + "".join(BASE_KERNEL % {"na": na} for na in WIDTHS)
    prologue, body, epilogue = plan
    start, end = wide_fn_span(base)
    text = base[:start] + prologue + body + epilogue + base[end:]
    depth = BUFFERS.get(arm, 1)
    max_na = 4 if arm in GATED else 5
    text = patch_dispatch(text, depth, max_na)
    return text + "".join(
        ARM_KERNEL % {"na": na, "slots": depth * min(na, max_na)}
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
                "e121_arms: %s and %s are byte-identical" % (arm, seen[digest]))
        seen[digest] = arm
        (outdir / ("arm_%s.metal" % arm)).write_text(text)
        print("%-13s %8d bytes  sha=%s  exact=%s"
              % (arm, len(text), digest, arm in EXACT_ARMS))
    manifest = {
        arm: {str(na): {"ipg": na, "partition": str(na), "weight_streams": 1}
              for na in WIDTHS}
        for arm in ARMS
    }
    (outdir / "partitions.json").write_text(json.dumps(manifest, indent=2) + "\n")
    names = ",".join(a if a in EXACT_ARMS or a == "a_base" else a + ":diag"
                     for a in ARMS)
    print("\n--arms %s" % names)


# --- census ------------------------------------------------------------------

KERNEL_RE = re.compile(r"e121_iso_na(\d+)$")
# The shipped entry point. It carries every width branch and the whole model's
# other qmv shapes, so its allocation is what an in-situ arm actually pays.
ENTRY_RE = re.compile(r"affine_qmv_fast")
DEVICE_LOAD = re.compile(r"=\s*load\s.*addrspace\(1\)")
TG_LOAD = re.compile(r"=\s*load\s.*addrspace\(3\)")
TG_STORE = re.compile(r"^\s*store\s.*addrspace\(3\)")
BARRIER_CALL = re.compile(r"call.*(air\.wg\.barrier|threadgroup_barrier)")
FADD = re.compile(r"=\s*fadd\s")
TG_GLOBAL = re.compile(
    r"^@(\S+)\s*=.*addrspace\(3\) global \[(\d+) x (\w+)\]")
TYPE_BYTES = {"bfloat": 2, "half": 2, "float": 4, "i8": 1, "i16": 2, "i32": 4}


def simdgroups(arch: str, registers: int) -> int:
    return REGISTER_FILE[arch] // (128 * registers)


def air_stats(source: pathlib.Path, workdir: pathlib.Path) -> dict:
    """Per entry point AIR facts after -O2, which is what the backend sees."""
    ll = workdir / "air.ll"
    done = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", str(source), "-o", str(ll)],
        capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip().splitlines()[-8:]}
    text = ll.read_text()
    tg_bytes: dict[str, int] = {}
    for line in text.splitlines():
        hit = TG_GLOBAL.match(line)
        if hit is None:
            continue
        owner = re.search(r"e121_iso_na(\d+)", hit.group(1))
        key = owner.group(1) if owner else (
            "entry" if ENTRY_RE.search(hit.group(1)) else None)
        if key is None:
            continue
        unit = TYPE_BYTES.get(hit.group(3))
        if unit is None:
            raise SystemExit(
                "e121_arms: unknown threadgroup element type %s" % hit.group(3))
        tg_bytes[key] = tg_bytes.get(key, 0) + int(hit.group(2)) * unit
    found: dict[str, dict] = {}
    name, body = None, []
    for line in text.splitlines():
        if line.startswith("define "):
            match = re.search(r"@([\w.]+)\(", line)
            name, body = (match.group(1) if match else None), []
        elif line == "}" and name is not None:
            hit = KERNEL_RE.search(name)
            key = hit.group(1) if hit else (
                "entry" if ENTRY_RE.search(name) else None)
            if key:
                found[key] = {
                    "device_loads": sum(1 for x in body if DEVICE_LOAD.search(x)),
                    "threadgroup_loads": sum(1 for x in body if TG_LOAD.search(x)),
                    "threadgroup_stores": sum(1 for x in body if TG_STORE.search(x)),
                    "threadgroup_bytes": tg_bytes.get(key, 0),
                    "barriers": sum(1 for x in body if BARRIER_CALL.search(x)),
                    "fadd": sum(1 for x in body if FADD.search(x)),
                    "air_lines": len(body),
                }
            name = None
        elif name is not None:
            body.append(line)
    return found


def census(directory: pathlib.Path, out: pathlib.Path | None) -> int:
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for arm in ARMS:
            source = directory / ("arm_%s.metal" % arm)
            air_dir = workdir / ("air_" + arm)
            air_dir.mkdir(parents=True, exist_ok=True)
            row: dict = {"air": air_stats(source, air_dir)}
            lib = build_metallib(source.read_text(), workdir / arm)
            for arch in (LOCAL_ARCH, RANKED_ARCH):
                for kernel, record in translate(lib, arch, workdir / arm).items():
                    hit = KERNEL_RE.search(kernel)
                    key = hit.group(1) if hit else (
                        "entry" if ENTRY_RE.search(kernel) else None)
                    if key is None:
                        continue
                    row.setdefault(arch, {})[key] = {
                        "registers": record.get("registers"),
                        "spill_bytes": record.get("spill_bytes", 0),
                        "text_bytes": record.get("text_bytes"),
                        "text_sha8": record.get("text_sha8"),
                    }
            rows[arm] = row
            print("censused %s" % arm)

    print("\nAIR per entry point: lines / fadd / device loads / tg ld+st / barriers")
    for arm in ARMS:
        cells = []
        for na in WIDTHS:
            cell = rows[arm]["air"].get(str(na), {})
            cells.append("NA%d=%s/%s/%s/%s+%s/%s" % (
                na, cell.get("air_lines", "?"), cell.get("fadd", "?"),
                cell.get("device_loads", "?"), cell.get("threadgroup_loads", "?"),
                cell.get("threadgroup_stores", "?"), cell.get("barriers", "?")))
        print("  %-13s %s" % (arm, "  ".join(cells)))

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
            print("  %-13s %s" % (arm, "  ".join(cells)))

    print("\nThreadgroup allocation and the residency it caps, 64-thread groups")
    for arm in ARMS:
        cells = []
        for na in WIDTHS:
            tg = rows[arm]["air"].get(str(na), {}).get("threadgroup_bytes", 0)
            cells.append("NA%d=%dB%s" % (
                na, tg, "/%dsg" % (2 * (TG_MEMORY_BYTES // tg)) if tg else ""))
        print("  %-13s %s" % (arm, "  ".join(cells)))
    print("  register-limited simdgroups per core at R = 96: %s %d, %s %d"
          % (LOCAL_ARCH, simdgroups(LOCAL_ARCH, 96),
             RANKED_ARCH, simdgroups(RANKED_ARCH, 96)))

    print("\nShipped entry point affine_qmv_fast, which every qmv shape pays")
    for arm in ARMS:
        air = rows[arm]["air"].get("entry", {})
        tg = air.get("threadgroup_bytes", 0)
        cells = []
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            value = rows[arm].get(arch, {}).get("entry")
            if value is None:
                cells.append("%s=?" % arch)
                continue
            spill = value["spill_bytes"] or 0
            cells.append("%s R=%s%s text=%s sg=%d" % (
                arch, value["registers"], "s%d" % spill if spill else "",
                value["text_bytes"], simdgroups(arch, value["registers"] or 96)))
        print("  %-13s tg=%dB%s  %s" % (
            arm, tg, "/%dsg" % (2 * (TG_MEMORY_BYTES // tg)) if tg else "",
            "  ".join(cells)))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"widths": list(WIDTHS), "arms": rows},
                                  indent=2) + "\n")
        print("\nwrote %s" % out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    if args.emit is not None:
        emit(args.emit)
    if args.census is not None:
        return census(args.census, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
