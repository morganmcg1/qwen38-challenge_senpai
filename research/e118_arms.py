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

Every arm except the two marked DIAGNOSTIC keeps every floating-point value and
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

# --- prologue surgery ---------------------------------------------------------

SIG_TAIL = "    uint simd_lid) {\n"
# The default keeps the shipped `qmv_fast_crossrow_affine4_g64_m` dispatcher
# compiling unchanged. That wrapper is never dispatched by this probe, which
# reaches `_wide` directly, so the null pointer is never dereferenced.
SIG_TAIL_SB = ("    uint simd_lid,\n"
               "    const device uint32_t* packed_sb = nullptr) {\n")

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


def expect(text: str, needle: str, count: int, label: str) -> None:
    seen = text.count(needle)
    if seen != count:
        raise SystemExit(
            "e118_arms: %s matched %d times, expected %d" % (label, seen, count))


def prologue_with(meta: str | None = None, loop: str | None = None,
                  wants_sb: bool = False) -> str:
    """The shared prologue with the metadata load or the whole loop replaced."""
    text = PROLOGUE
    if meta is not None:
        expect(text, META_LOAD, 1, "prologue metadata load")
        text = text.replace(META_LOAD, meta)
    if loop is not None:
        expect(text, WEIGHT_META_LOOP, 1, "prologue weight+metadata loop")
        text = text.replace(WEIGHT_META_LOOP, loop)
    if wants_sb:
        expect(text, SIG_TAIL, 1, "prologue signature tail")
        text = text.replace(SIG_TAIL, SIG_TAIL_SB)
    return text


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
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * %(na)d;
  if (first_m >= %(na)d) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, first_m, out_row,
      simd_lid%(sb)s);
}
"""

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
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, %(na)d, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid,
      simd_lid);
}
"""

# arm -> (prologue, body, epilogue) applied to the wide template, or None for
# the unmodified shipped template. `sb` marks an arm that reads buffer 7.
PLANS = {
    "a_base": (None, False),
    "q_scaffold": ((PROLOGUE, BODY_BASE, EPILOGUE), False),
    "s_bcast": ((prologue_with(meta=META_BCAST), BODY_BASE, EPILOGUE), False),
    "s_bcast_all": ((prologue_with(loop=WEIGHT_META_LOOP_BCAST_ALL), BODY_BASE,
                     EPILOGUE), False),
    "s_bcast_scale": ((prologue_with(meta=META_BCAST_SCALE), BODY_BASE,
                       EPILOGUE), False),
    "p_split_meta": ((prologue_with(loop=WEIGHT_META_LOOP_SPLIT), BODY_BASE,
                      EPILOGUE), False),
    "g_pack32": ((prologue_with(meta=META_PACK32, wants_sb=True), BODY_BASE,
                  EPILOGUE), True),
    "s_bcast_pack32": ((prologue_with(meta=META_BCAST_PACK32, wants_sb=True),
                        BODY_BASE, EPILOGUE), True),
    "p_prefetch_w": ((PREFETCH_PROLOGUE, BODY_BASE, EPILOGUE), False),
    "n_nosums": ((PROLOGUE, BODY_NOSUMS, EPILOGUE), False),
    "l_loadonly": ((PROLOGUE, BODY_LOADONLY, EPILOGUE), False),
}

# Arms that must reproduce `a_base` bit for bit.
DIAGNOSTIC_ARMS = ("n_nosums", "l_loadonly")
# Arms the primary metric may rank: bit exact AND not register-confounded on
# this host. `p_prefetch_w` is bit exact but spills on g16s at the widths that
# matter, so it is reported separately.
PROMOTION_ARMS = ("s_bcast", "s_bcast_all", "s_bcast_scale", "p_split_meta",
                  "g_pack32", "s_bcast_pack32")

ARMS = tuple(PLANS)


def arm_source(base: str, arm: str, via_m: bool = False) -> str:
    plan, wants_sb = PLANS[arm]
    text = base
    if plan is not None:
        prologue, body, epilogue = plan
        start, end = wide_fn_span(base)
        text = base[:start] + prologue + body + epilogue + base[end:]
    template = ISO_KERNEL_VIA_M if via_m else ISO_KERNEL
    return text + "".join(
        template % {"na": na, "sb": ", packed_sb" if wants_sb else ""}
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
        print("%-15s %8d bytes  sha=%s  exact=%s  buffer7=%s"
              % (arm, len(text), digest, arm not in DIAGNOSTIC_ARMS,
                 PLANS[arm][1]))
    (outdir / "ctl_a_base_via_m.metal").write_text(
        arm_source(base, "a_base", via_m=True))
    names = ",".join(a + (":diag" if a in DIAGNOSTIC_ARMS else "") for a in ARMS)
    print("\n--arms %s" % names)


# --- census -------------------------------------------------------------------

KERNEL_RE = re.compile(r"e118_iso_na(\d+)$")
DEVICE_LOAD = re.compile(r"=\s*load\s.*addrspace\(1\)")
SHUFFLE = re.compile(r"simd_shuffle|@air\.simd_shuffle")


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
                found[hit.group(1)] = {
                    "device_loads": sum(1 for x in body if DEVICE_LOAD.search(x)),
                    "shuffles": sum(1 for x in body if SHUFFLE.search(x)),
                    "air_lines": len(body),
                }
            name = None
        elif name is not None:
            body.append(line)
    return found


def census_one(source: pathlib.Path, workdir: pathlib.Path, tag: str) -> dict:
    air_dir = workdir / ("air_" + tag)
    air_dir.mkdir(parents=True, exist_ok=True)
    row: dict = {"air": air_stats(source, air_dir)}
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
    args = ap.parse_args()
    if args.emit is not None:
        emit(args.emit)
    if args.census is not None:
        return census(args.census, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
