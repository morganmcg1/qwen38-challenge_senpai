#!/usr/bin/env python3
"""E110 rung 1: loop-nest order and activation staging in the wide QMV kernel.

Rung 0 refuted H1 and H2. The activation stream does not cost bandwidth: it
costs only when it sits next to the dense arithmetic, so the remaining
mechanism is the extraction schedule inside the k-block. This module builds the
arms that change that schedule.

    research/e110_arms.py --emit /tmp/e110-arms
    research/e110_arms.py --census /tmp/e110-arms --out research/out/e110/x.json

Arms, all instantiated per width as their own `e110_iso_na<NA>` entry point so
the backend allocates registers for one width instead of a max over branches:

  a_base      the shipped wide kernel, unmodified, routed to IPG == NA.
  l_loadonly  E104's pure-load arm, reproduced byte for byte. The LOAD half of
              the rule 36b roofline pair.
  b_constw    both operand streams replaced by compile-time constants while the
              full 160 x NA scalar body is retained. The ALU half of the
              roofline pair. The constants differ per lane and per row so no
              vector work collapses to scalar work.
  b_barrier   `a_base` plus the two `threadgroup_barrier` calls per k-block that
              staging needs, and nothing else. It prices the barrier alone,
              which a staged arm cannot separate from its own gain.
  xs_stage    the NA x 512 activation tile for the k-block is staged in
              threadgroup memory once per THREADGROUP, then read from there by
              both simdgroups. Only the memory space of the activation read
              changes.
  mo_swap     the `i` outer / `m` inner nest is swapped to `m` outer / `i`
              inner. Each lane then finishes its 32 bytes of activation row `m`
              before it moves to row `m + 1` instead of interleaving NA address
              streams, and the four staged conversions become scalars instead
              of `vec<float, NA>`, which is `4 * NA` fewer live registers.
  mo_stage    `mo_swap` and `xs_stage` together: additive or the same effect?

`b_constw` and `l_loadonly` change the arithmetic on purpose and are
timing-only diagnostics. `b_barrier`, `xs_stage`, `mo_swap` and `mo_stage` must
reproduce `a_base` bit for bit; the harness checks that at every cell and the
positive control proves the check can fail.

Research-only: nothing here is on the scored path.
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

# NA = 6 is not reachable in the shipped wide switch, so rung 1 drops it. The
# realised width histogram of the same fixture weights the four survivors.
WIDTHS = (2, 3, 4, 5)

# Apple's documented threadgroup memory limit, which is also the per-core pool
# the resident threadgroups share. A tile of `B` bytes therefore caps residency
# at `floor(TG_MEMORY_BYTES / B)` threadgroups of 64 threads, whatever the
# register file allows. The probe prints the device's own
# `maxThreadgroupMemoryLength`, so this constant is checkable, not assumed.
TG_MEMORY_BYTES = 32768

# Register file per core, in bytes, and 128 bytes per simdgroup register.
REGISTER_FILE = {LOCAL_ARCH: 384 * 1024, RANKED_ARCH: 496 * 1024}


def simdgroups(arch: str, registers: int) -> int:
    """Register-limited resident simdgroups per core."""
    return REGISTER_FILE[arch] // (128 * registers)


# --- the shipped inner body, in the shared scaffold ---------------------------
# `DIRECT_NIBBLES == true` is the branch every scored cell takes, so the
# scaffold carries it unconditionally. E104 proved this scaffold faithful: its
# `xw_widex` arm, built on exactly these strings, was bit-identical to the
# unmodified shipped kernel at every cell.
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

# E104's `l_loadonly`, reproduced verbatim so the DCE question is asked of the
# arm that produced the finding and not of a lookalike.
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

# The four activation values a lane would have loaded, as compile-time
# constants. `m` and `i` are unrolled loop indices, so every value below folds
# at compile time and no activation address is ever formed. They differ per
# lane, which keeps the `vec<float, NA>` arithmetic from collapsing to scalar
# arithmetic and makes the arm a load-stream control rather than a smaller
# kernel.
X_CONSTANTS = """        const float c0 = 0.5f + 0.0625f * float(m) + 0.00390625f * float(i);
        const float c1 = c0 * 0.5f;
        const float c2 = c0 * 0.25f;
        const float c3 = c0 * 0.125f;
"""

BODY_WONLY = """
    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
%s        sums[m] += c0 + c1 + c2 + c3;
        a0[m] = c0;
        a1[m] = c1;
        a2[m] = c2;
        a3[m] = c3;
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
""" % X_CONSTANTS

# mo_swap: the `i` outer / `m` inner nest of the shipped body, swapped.
#
# Bit exactness by construction. `sums` and `partial[r]` are both zeroed inside
# the k loop. For each fixed pair (r, m) the accumulation into `partial[r][m]`
# still runs over i = 0, 1, 2, 3 in that order over the same expression, and for
# each fixed m the accumulation into `sums[m]` still runs over i = 0, 1, 2, 3 in
# that order over the same BF16 expression tree. The `acc[r]` update and the
# `simd_sum` reduction are untouched. `a0..a3` become scalars because the m loop
# no longer has to hold one lane of every row at once.
BODY_MOSWAP = """
    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int m = 0; m < NA; m++) {
      float sm = 0.0f;
      for (int i = 0; i < 4; i++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        const float a0 = static_cast<float>(xm[0]);
        const float a1 = static_cast<float>(xm[1]);
        const float a2 = static_cast<float>(xm[2]);
        const float a3 = static_cast<float>(xm[3]);
        sm += xm[0] + xm[1] + xm[2] + xm[3];
        for (int r = 0; r < rows_per_simd; r++) {
          partial[r][m] += (a0 * (packed[r][i] & 0x000f) +
                            a1 * ((packed[r][i] >> 4) & 0x000f) +
                            a2 * ((packed[r][i] >> 8) & 0x000f) +
                            a3 * ((packed[r][i] >> 12) & 0x000f));
        }
      }
      sums[m] = sm;
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
"""

# The weight-side loads of the shared prologue, and the constant stand-in that
# `b_constw` puts in their place.
WEIGHT_LOADS = """    for (int r = 0; r < rows_per_simd; r++) {
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

WEIGHT_CONSTANTS = """    for (int r = 0; r < rows_per_simd; r++) {
      for (int i = 0; i < 4; i++) {
        packed[r][i] = uint16_t(0x1357u + 0x0248u * uint(r) + 0x0011u * uint(i));
      }
      scale_local[r] = 0.00625f + 0.00025f * float(r);
      bias_local[r] = -0.046875f - 0.001f * float(r);
    }
"""

BARRIERS = """    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

K_LOOP_OPEN = "  for (int k = 0; k < in_vec_size; k += block_size) {\n"


def expect(text: str, needle: str, count: int, label: str) -> None:
    seen = text.count(needle)
    if seen != count:
        raise SystemExit(
            "e110_arms: %s matched %d times, expected %d" % (label, seen, count))


def prologue_constw() -> str:
    expect(PROLOGUE, WEIGHT_LOADS, 1, "prologue weight loads")
    return PROLOGUE.replace(WEIGHT_LOADS, WEIGHT_CONSTANTS)


def prologue_barrier() -> str:
    expect(PROLOGUE, K_LOOP_OPEN, 1, "k loop open")
    return PROLOGUE.replace(K_LOOP_OPEN, K_LOOP_OPEN + BARRIERS)


# --- the staged arm ----------------------------------------------------------
# The tile is `NA * block_size` bf16 values, 5,120 B at NA = 5, against a 32 KB
# threadgroup budget. All 64 threads of the threadgroup cooperate on a coalesced
# 16-byte-per-thread-per-row copy, so both simdgroups then read the same tile
# and the device-side activation fetch count halves before any cache effect.
#
# Only the ADDRESS SPACE of the activation read changes. The values, the
# per-lane `partial[r]` accumulation order, the `sums[m]` expression tree and
# the `simd_sum` reduction order are all unchanged, so the arm is bit exact by
# construction and the probe's fidelity pass proves it at every cell.
STAGED_FN = """
template <typename T, int NA, bool DIRECT_NIBBLES = false>
METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide_staged%(suffix)s(
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
    threadgroup T* xtile) {
  static_assert(NA >= 2 && NA <= 8, "e110 probe admits NA in [2, 8]");
  typedef vec<float, NA> VF;
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  constexpr int copy_per_thread = 8;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;
  const int tg_lid = int(simd_gid) * SIMD_SIZE + int(simd_lid);

  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int m = 0; m < NA; m++) {
      const device uint4* src = reinterpret_cast<const device uint4*>(
          x + (first_m + m) * in_vec_size + k + tg_lid * copy_per_thread);
      threadgroup uint4* dst = reinterpret_cast<threadgroup uint4*>(
          xtile + m * block_size + tg_lid * copy_per_thread);
      *dst = *src;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

%(body)s  }

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

# The shipped nest, reading the staged tile. Only the address space changes.
STAGED_BODY_XS = """    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
        const threadgroup T* xm = xtile + m * block_size +
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

# The swapped nest, reading the staged tile: both changes at once.
STAGED_BODY_MO = """    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int m = 0; m < NA; m++) {
      float sm = 0.0f;
      for (int i = 0; i < 4; i++) {
        const threadgroup T* xm = xtile + m * block_size +
            simd_lid * values_per_thread + 4 * i;
        const float a0 = static_cast<float>(xm[0]);
        const float a1 = static_cast<float>(xm[1]);
        const float a2 = static_cast<float>(xm[2]);
        const float a3 = static_cast<float>(xm[3]);
        sm += xm[0] + xm[1] + xm[2] + xm[3];
        for (int r = 0; r < rows_per_simd; r++) {
          partial[r][m] += (a0 * (packed[r][i] & 0x000f) +
                            a1 * ((packed[r][i] >> 4) & 0x000f) +
                            a2 * ((packed[r][i] >> 8) & 0x000f) +
                            a3 * ((packed[r][i] >> 12) & 0x000f));
        }
      }
      sums[m] = sm;
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
"""

# The entry point the harness dispatches. `first_m` is threadgroup-uniform, so
# the early return of the NA - 1 non-weight-reading x-groups is uniform too and
# the barriers inside the staged body stay legal.
ISO_KERNEL = """
[[kernel]] void e110_iso_na%(na)d(
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
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, %(na)d, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid,
      simd_lid);
}
"""

STAGED_KERNEL = """
[[kernel]] void e110_iso_na%(na)d(
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
  threadgroup bfloat16_t xtile[%(na)d * 512];
  const int first_m = int(tid.x) * %(na)d;
  if (first_m >= %(na)d) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  qmv_fast_crossrow_affine4_g64_wide_staged%(suffix)s<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, first_m, out_row,
      simd_gid, simd_lid, xtile);
}
"""

# arm -> (prologue, body, epilogue) applied to the wide template, or the staged
# body the arm appends instead.
BODIES = {
    "a_base": None,
    "l_loadonly": (PROLOGUE, BODY_LOADONLY, EPILOGUE),
    "b_constw": (prologue_constw(), BODY_WONLY, EPILOGUE),
    "b_barrier": (prologue_barrier(), BODY_BASE, EPILOGUE),
    "xs_stage": None,
    "mo_stage": None,
    "mo_swap": (PROLOGUE, BODY_MOSWAP, EPILOGUE),
}

STAGED = {"xs_stage": ("_xs", STAGED_BODY_XS),
          "mo_stage": ("_mo", STAGED_BODY_MO)}

# Arms that must reproduce `a_base` bit for bit. The rest change the arithmetic
# on purpose and are timing-only.
EXACT_ARMS = ("b_barrier", "xs_stage", "mo_stage", "mo_swap")

# The harness runs its positive control on the LAST arm, so the last arm has to
# be an exact one. `mo_swap` is the priority arm, so it takes that slot.
ARMS = ("a_base", "l_loadonly", "b_constw", "b_barrier", "xs_stage",
        "mo_stage", "mo_swap")


def arm_source(base: str, arm: str) -> str:
    plan = BODIES[arm]
    text = base
    if plan is not None:
        prologue, body, epilogue = plan
        start, end = wide_fn_span(base)
        text = base[:start] + prologue + body + epilogue + base[end:]
    if arm in STAGED:
        suffix, staged_body = STAGED[arm]
        fields = {"suffix": suffix, "body": staged_body}
        return text + STAGED_FN % fields + "".join(
            STAGED_KERNEL % dict(fields, na=na) for na in WIDTHS)
    return text + "".join(ISO_KERNEL % {"na": na} for na in WIDTHS)


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
                "e110_arms: %s and %s are byte-identical" % (arm, seen[digest]))
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

KERNEL_RE = re.compile(r"e110_iso_na(\d+)$")
DEVICE_LOAD = re.compile(r"=\s*load\s.*addrspace\(1\)")
TG_LOAD = re.compile(r"=\s*load\s.*addrspace\(3\)")
TG_STORE = re.compile(r"^\s*store\s.*addrspace\(3\)")
# A `threadgroup` declaration inside a kernel becomes a module-level global in
# address space 3 whose mangled name carries the entry point it belongs to.
# Summing those per entry point is how the compiled allocation is read without a
# device, which is the question the isolated per-width probe cannot answer for
# the composed dispatcher.
TG_GLOBAL = re.compile(
    r"^@(\S+)\s*=.*addrspace\(3\) global \[(\d+) x (\w+)\]")
TYPE_BYTES = {"bfloat": 2, "half": 2, "float": 4, "i8": 1, "i16": 2, "i32": 4}


def air_loads(source: pathlib.Path, workdir: pathlib.Path) -> dict:
    """Device and threadgroup memory accesses per entry point, after -O2.

    AIR is what the front end hands the backend, so a load that is still here
    survived every dead-code pass the optimiser runs. A load whose value reaches
    a store cannot be removed later either, which is why this count answers the
    DCE question the timing arms cannot.
    """
    ll = workdir / "air.ll"
    done = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", str(source), "-o", str(ll)],
        capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip().splitlines()[-8:]}
    found: dict[str, dict] = {}
    tg_bytes: dict[str, int] = {}
    for line in ll.read_text().splitlines():
        global_hit = TG_GLOBAL.match(line)
        if global_hit is None:
            continue
        owner = re.search(r"e110_iso_na(\d+)", global_hit.group(1))
        if owner is None:
            continue
        width = int(global_hit.group(2))
        unit = TYPE_BYTES.get(global_hit.group(3))
        if unit is None:
            raise SystemExit("e110_arms: unknown threadgroup element type %s"
                             % global_hit.group(3))
        tg_bytes[owner.group(1)] = tg_bytes.get(owner.group(1), 0) + width * unit
    name, body = None, []
    for line in ll.read_text().splitlines():
        if line.startswith("define "):
            match = re.search(r"@([\w.]+)\(", line)
            name, body = (match.group(1) if match else None), []
        elif line == "}" and name is not None:
            hit = KERNEL_RE.search(name)
            if hit:
                found[hit.group(1)] = {
                    "device_loads": sum(1 for x in body if DEVICE_LOAD.search(x)),
                    "threadgroup_loads": sum(1 for x in body if TG_LOAD.search(x)),
                    "threadgroup_stores": sum(1 for x in body if TG_STORE.search(x)),
                    "threadgroup_bytes": tg_bytes.get(hit.group(1), 0),
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
            row: dict = {"air": air_loads(source, air_dir)}
            lib = build_metallib(source.read_text(), workdir / arm)
            for arch in (LOCAL_ARCH, RANKED_ARCH):
                for kernel, record in translate(lib, arch, workdir / arm).items():
                    hit = KERNEL_RE.search(kernel)
                    if hit is None:
                        continue
                    row.setdefault(arch, {})[hit.group(1)] = {
                        "registers": record.get("registers"),
                        "spill_bytes": record.get("spill_bytes", 0),
                        "text_bytes": record.get("text_bytes"),
                        "text_sha8": record.get("text_sha8"),
                    }
            rows[arm] = row
            print("censused %s" % arm)

    print("\nAIR device loads per k-block body, per entry point")
    print("  %-13s %s" % ("arm", "  ".join("NA%d" % na for na in WIDTHS)))
    for arm in ARMS:
        air = rows[arm]["air"]
        cells = []
        for na in WIDTHS:
            cell = air.get(str(na), {})
            tg = cell.get("threadgroup_loads", 0)
            cells.append("%3s%s" % (cell.get("device_loads", "?"),
                                    "+%dtg" % tg if tg else "    "))
        print("  %-13s %s" % (arm, " ".join(cells)))

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

    print("\nCompiled threadgroup memory per entry point, and the occupancy it "
          "implies")
    print("  a threadgroup is 64 threads = 2 simdgroups; the cap is "
          "floor(%d / tile) threadgroups" % TG_MEMORY_BYTES)
    for arm in ARMS:
        cells = []
        for na in WIDTHS:
            tg = rows[arm]["air"].get(str(na), {}).get("threadgroup_bytes", 0)
            cells.append("NA%d=%dB%s" % (
                na, tg, "/%dsg" % (2 * (TG_MEMORY_BYTES // tg)) if tg else ""))
        print("  %-13s %s" % (arm, "  ".join(cells)))
    print("  register-limited simdgroups per core: %s %d, %s %d at R = 96"
          % (LOCAL_ARCH, simdgroups(LOCAL_ARCH, 96),
             RANKED_ARCH, simdgroups(RANKED_ARCH, 96)))

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
