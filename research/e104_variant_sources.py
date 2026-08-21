#!/usr/bin/env python3
"""Emit the E104 rate-curve arms as self-contained Metal sources.

    research/e104_variant_sources.py --outdir /tmp/e104-arms

E104 asks why one wide x-group loses a third of its streaming rate between
NA = 2 and NA = 5 while reading exactly the same weight bytes. Every arm is the
runtime-effective JIT string for `affine_qmv_fast<bfloat16_t, 64, 4, false>`
with two shared changes plus one arm-specific rewrite of
`qmv_fast_crossrow_affine4_g64_wide`.

Shared changes (identical in every arm, so the NA sweep is matched):

  * the M switch in the `out_vec_size >= 4096` branch routes M in [2, 6] to
    IPG == M, so exactly ONE x-group reads weights at every measured NA. M = 3,
    4 and 5 already ship that way; M = 2 and M = 6 do not, and forcing them
    makes the whole sweep a single-variable NA ladder over one kernel family.
  * the NA and M static_asserts are widened to admit NA = 6.

Arms:

  a_base      the shipped wide kernel, unmodified.
  l_loadonly  every device load and every bf16 -> f32 conversion of a_base, with
              the four-nibble FMA block collapsed to one. Isolates the load
              stream: if its rate still falls with NA, the load stream itself is
              throttled; if it is flat, the constraint is on the compute side.
  z_noxload   weights, scales and biases only; the activation stream is not read
              at all. Nothing in it scales with NA except the NA output writes,
              so a non-flat rate here would mean the grid's NA - 1 early-return
              x-groups, and not the kernel body, carry the penalty.
  xw_widex    a_base with the activation loads widened from 8 to 16 bytes: eight
              x values per instruction instead of four, halving the activation
              load count. The candidate fix. Every arithmetic operation, its
              operand types and the accumulation order are unchanged, so this
              arm must be BIT-IDENTICAL to a_base.

`l_loadonly` and `z_noxload` change the arithmetic on purpose. They are
timing-only diagnostics and must never reach a candidate path.

Every substitution count is asserted, so source drift fails loudly instead of
silently producing an arm that is identical to another arm.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
ENTRY = "affine_qmv_fast_bfloat16_t_64_4_false"

ARMS = ("a_base", "l_loadonly", "z_noxload", "xw_widex")
PROBE_NA = (2, 3, 4, 5, 6)

# --- rung 0.5: the partition ladder ------------------------------------------
# The break-even law needs the isolated ONE-group rate `r1` and the two-group
# rate `r2` at every verify width. `x_onegroup` routes every M to IPG == M, so
# one x-group reads the weights. `y_split` uses the two-group partition that a
# collapse at that width replaces: the shipped one at M = 6, 7 and 8, and the
# pre-E100 one at M = 5.
#
# M = 2 and M = 3 have no legal two-group partition, because `M % IPG == 1` is
# not instantiated and IPG = 1 would need `vec<float, 1>`. Both arms therefore
# run the same one-group kernel there, which makes those two cells a null
# control that measures the instrument's own noise.
#
# Both arms must agree bit for bit at every width: each group runs the same
# accumulation over the same k order for its own rows.
LADDER_NA = (2, 3, 4, 5, 6, 7, 8)
LADDER_SPLIT = {2: 2, 3: 3, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}

WIDE_FN = "qmv_fast_crossrow_affine4_g64_wide"

# --- shared patches ----------------------------------------------------------

NA_ASSERT = (
    'static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in '
    '[2, 5]");'
)
NA_ASSERT_NEW = (
    'static_assert(NA >= 2 && NA <= 8, "e104 probe admits NA in [2, 8]");'
)
M_ASSERT = (
    'static_assert(M >= 3 && M <= 9, "wide multi-row QMV dispatch covers M in '
    '[3, 9]");'
)
M_ASSERT_NEW = (
    'static_assert(M >= 2 && M <= 9, "e104 probe admits M in [2, 9]");'
)

# The `out_vec_size >= 4096` switch is the one the five scored shapes reach.
# Isolating it by span keeps the `< 4096` switch, which calls the pair kernel
# under names that would otherwise also match, untouched.
WIDE_SWITCH_OPEN = "    if (out_vec_size >= 4096) {\n"
WIDE_SWITCH_CLOSE = "    } else {\n"

CASE2_OLD = """        case 2:
          qmv_fast_crossrow_affine4_g64<T, 2>(
"""
CASE2_NEW = """        case 2:
          qmv_fast_crossrow_affine4_g64_m<T, 2, 2, true>(
"""
CASE6_OLD = "          qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>(\n"
CASE6_NEW = "          qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true>(\n"


def emit_base(path: pathlib.Path) -> str:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "research/jit_string_compile.py"),
            "--emit",
            str(path),
            "--",
            CELL,
        ],
        check=True,
        cwd=str(ROOT),
    )
    return path.read_text()


def expect(text: str, needle: str, count: int, label: str) -> None:
    seen = text.count(needle)
    if seen != count:
        raise SystemExit(
            "e104_variant_sources: %s matched %d times, expected %d"
            % (label, seen, count)
        )


def widen_asserts(text: str) -> str:
    """Admit the research-only widths the probe instantiates."""
    expect(text, NA_ASSERT, 1, "NA static_assert")
    expect(text, M_ASSERT, 1, "M static_assert")
    return text.replace(NA_ASSERT, NA_ASSERT_NEW).replace(M_ASSERT, M_ASSERT_NEW)


def force_lone_group(text: str) -> str:
    """Route every measured M to one weight-reading x-group."""
    text = widen_asserts(text)

    open_at = text.index(WIDE_SWITCH_OPEN)
    close_at = text.index(WIDE_SWITCH_CLOSE, open_at)
    block = text[open_at:close_at]
    expect(block, CASE2_OLD, 1, "wide-branch case 2")
    expect(block, CASE6_OLD, 1, "wide-branch case 6")
    block = block.replace(CASE2_OLD, CASE2_NEW).replace(CASE6_OLD, CASE6_NEW)
    return text[:open_at] + block + text[close_at:]


def wide_fn_span(text: str) -> tuple[int, int]:
    """Span of the whole `qmv_fast_crossrow_affine4_g64_wide` template."""
    sig = "METAL_FUNC void %s(\n" % WIDE_FN
    expect(text, sig, 1, "wide kernel signature")
    at = text.index(sig)
    start = text.rindex("template <", 0, at)
    end = text.index("\n}\n", at) + len("\n}\n")
    return start, end


# --- arm bodies --------------------------------------------------------------

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
    uint simd_lid) {
  static_assert(NA >= 2 && NA <= 8, "e104 probe admits NA in [2, 8]");
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

# l_loadonly: every load and every bf16 -> f32 conversion of the base arm. The
# activation loads survive because the bias-correction sum reads all four
# values; only the nibble extraction and three quarters of the vector FMA go.
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

# z_noxload: the weight and metadata stream on its own. Nothing here depends on
# NA except the NA output writes, so its rate curve prices the grid's
# early-returning x-groups rather than the kernel body.
BODY_NOXLOAD = """
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += VF(float(packed[r][i] & 0x000f));
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r];
    }
"""

# xw_widex: one 16-byte activation load replaces two 8-byte loads. The eight
# values are consumed in the same order, in the same operand types, and are
# accumulated into `sums` and `partial` in two separate statements per 16-byte
# load, so every FP32 and every BF16 rounding step of the base arm survives.
BODY_WIDEX = """
    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i += 2) {
      VF a0, a1, a2, a3, a4, a5, a6, a7;
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        const uint4 raw = *reinterpret_cast<const device uint4*>(xm);
        const T t0 = as_type<T>(static_cast<uint16_t>(raw.x & 0xffffu));
        const T t1 = as_type<T>(static_cast<uint16_t>(raw.x >> 16));
        const T t2 = as_type<T>(static_cast<uint16_t>(raw.y & 0xffffu));
        const T t3 = as_type<T>(static_cast<uint16_t>(raw.y >> 16));
        const T t4 = as_type<T>(static_cast<uint16_t>(raw.z & 0xffffu));
        const T t5 = as_type<T>(static_cast<uint16_t>(raw.z >> 16));
        const T t6 = as_type<T>(static_cast<uint16_t>(raw.w & 0xffffu));
        const T t7 = as_type<T>(static_cast<uint16_t>(raw.w >> 16));
        sums[m] += t0 + t1 + t2 + t3;
        sums[m] += t4 + t5 + t6 + t7;
        a0[m] = static_cast<float>(t0);
        a1[m] = static_cast<float>(t1);
        a2[m] = static_cast<float>(t2);
        a3[m] = static_cast<float>(t3);
        a4[m] = static_cast<float>(t4);
        a5[m] = static_cast<float>(t5);
        a6[m] = static_cast<float>(t6);
        a7[m] = static_cast<float>(t7);
      }
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
        partial[r] += (a4 * (packed[r][i + 1] & 0x000f) +
                       a5 * ((packed[r][i + 1] >> 4) & 0x000f) +
                       a6 * ((packed[r][i + 1] >> 8) & 0x000f) +
                       a7 * ((packed[r][i + 1] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
"""

BODIES = {
    "l_loadonly": BODY_LOADONLY,
    "z_noxload": BODY_NOXLOAD,
    "xw_widex": BODY_WIDEX,
}


def arm_source(base: str, arm: str) -> str:
    if arm == "a_base":
        return base
    start, end = wide_fn_span(base)
    return base[:start] + PROLOGUE + BODIES[arm] + EPILOGUE + base[end:]


# --- isolated per-NA entry points for the AIR census -------------------------

ISO_KERNEL = """
[[kernel]] void e104_iso_na%(na)d(
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


def iso_source(arm_text: str) -> str:
    """One instantiation per NA, so the census can price a single NA cell.

    The shipped entry point inlines every width branch, so its register peak is
    a max over branches and cannot answer an NA question.
    """
    return arm_text + "".join(ISO_KERNEL % {"na": na} for na in PROBE_NA)


# --- rung 0.5: isolated partition entry points -------------------------------

LADDER_KERNEL = """
// M = %(na)d as %(label)s
[[kernel]] void e104_iso_na%(na)d(
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
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, %(na)d, %(ipg)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid,
      simd_lid);
}
"""


def partition_label(m: int, ipg: int) -> str:
    groups, left = [], m
    while left > 0:
        groups.append(min(ipg, left))
        left -= ipg
    return "+".join(str(g) for g in groups)


def ladder_source(base: str, ipg_of: dict[int, int]) -> str:
    """Every width as its own kernel, so registers are allocated per cell."""
    return base + "".join(
        LADDER_KERNEL
        % {"na": na, "ipg": ipg_of[na], "label": partition_label(na, ipg_of[na])}
        for na in LADDER_NA
    )


def emit_rate_arms(outdir: pathlib.Path, raw: str) -> tuple[str, ...]:
    base = force_lone_group(raw)
    (outdir / "base_lone.metal").write_text(base)
    for arm in ARMS:
        text = arm_source(base, arm)
        (outdir / ("arm_%s.metal" % arm)).write_text(text)
        (outdir / ("iso_%s.metal" % arm)).write_text(iso_source(text))
        print("%-12s %8d bytes  arm_%s.metal" % (arm, len(text), arm))
    return ARMS


def emit_ladder_arms(outdir: pathlib.Path, raw: str) -> tuple[str, ...]:
    base = widen_asserts(raw)
    plans = {
        "x_onegroup": {na: na for na in LADDER_NA},
        "y_split": LADDER_SPLIT,
    }
    manifest = {}
    for arm, ipg_of in plans.items():
        text = ladder_source(base, ipg_of)
        (outdir / ("arm_%s.metal" % arm)).write_text(text)
        manifest[arm] = {
            str(na): {
                "ipg": ipg_of[na],
                "partition": partition_label(na, ipg_of[na]),
                "weight_streams": -(-na // ipg_of[na]),
            }
            for na in LADDER_NA
        }
        table = " ".join(
            "M%d=[%s]" % (na, partition_label(na, ipg_of[na])) for na in LADDER_NA
        )
        print("%-12s %8d bytes  %s" % (arm, len(text), table))
    (outdir / "partitions.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return tuple(plans)


SETS = {"rate": emit_rate_arms, "ladder": emit_ladder_arms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/e104-arms")
    ap.add_argument("--set", dest="which", default="rate", choices=sorted(SETS))
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = emit_base(outdir / "base_raw.metal")
    arms = SETS[args.which](outdir, raw)

    distinct = {(outdir / ("arm_%s.metal" % a)).read_text() for a in arms}
    if len(distinct) != len(arms):
        raise SystemExit("e104_variant_sources: two arms are byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
