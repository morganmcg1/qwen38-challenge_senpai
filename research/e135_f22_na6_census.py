#!/usr/bin/env python3
"""F22 kill gate: can `na6` hold 96 or fewer g17s registers without spilling?

    usage: research/e135_f22_na6_census.py [--out PATH] [--keep DIR]

Advisor F22 asks one question. The shipped two-pass plan runs width 6 as
`(6, 3, 4)`, entry point `na3`, which allocates 94 registers and 42 resident
simdgroups on `applegpu_g17s`. The one-pass plan runs it as `(6, 6, 4)`, entry
point `na6`, which wins 8,405.2 us per round on g16s and allocates 105
registers for 37 simdgroups on g17s, an 11.9 % occupancy loss that a ranked
receipt then charged us 0.2646 % for. If `na6` can be brought to 96 registers
or fewer on g17s without spilling, the body win is available with no occupancy
tax.

CAMPAIGN RULE 83. g16s clamps at 96 and shows nothing, so the closure is a
g17s closure. g16s numbers are reported beside it only to keep the two-host
table complete. CAMPAIGN RULE 99: this census is a detector, not a judge. A
pass earns an implementation and one ranked receipt, not a claim.

Zero GPU seconds. `xcrun metal-tt` runs the real AGX backend for a named
architecture on any Mac, wrapped by `research/agx_crossarch.py`.

THE ARMS

`na3_control`     the shipped two-pass width-6 body. It must read 94 / 42 on
                  g17s. This is the channel check: if it does not reproduce
                  the recorded figure, nothing else here is trustworthy.
`na6_baseline`    the one-pass width-6 body as it stands. It must read 105 / 37.
`na6_maxtpt64`    ARM A. `[[max_total_threads_per_threadgroup(64)]]` on the
                  entry point. Our threadgroup is (32, 2, 1) = 64 threads. The
                  advisor states plainly that he cannot verify this is a
                  register-capping lever in MSL, and in most toolchains it
                  permits more registers rather than fewer. Five minutes.
`na6_pairN`       ARM B. Process the NA columns in chunks of N inside the k
                  block instead of all NA at once, re-extracting the nibble
                  and re-reading the scale and bias per chunk.

WHY ARM B SHOULD MOVE THE COUNT. Per k block the baseline keeps live, in
units of float lanes: `acc[RPS]` at RPS*NA, `partial[RPS]` at RPS*NA,
`a0..a3` at 4*NA, and `sums` at NA. At NA = 6, RPS = 4 that is 24 + 24 + 24 +
6 = 78 lanes. `acc` must stay at RPS*NA because it accumulates across k, but
`partial`, `a0..a3` and `sums` only live inside one k block, so chunking them
to width N cuts them to RPS*N + 4*N + N. At N = 2 the total falls to
24 + 8 + 8 + 2 = 42 lanes.

THE BIT-EXACTNESS INVARIANT, which is the reason this arm is legal at all.
Each column `m` keeps its own reduction order exactly: `partial[r][m]`
accumulates over `i` in the same order, and `acc[r][m]` accumulates over `k`
in the same order, because the vector lanes were always independent. Only
which columns are computed together changes. Re-extracting the same nibble
with the same shift and mask, and re-reading the same `scales[group_index]`
and `biases[group_index]`, returns the same value. This is therefore NOT a
Rule 92 accumulation-order change. This census does not prove that: it is a
register census. The proof is a bit comparison at the touched cells with a
failing Rule 101 control, and it belongs to the implementation step.

TRANSCRIPTION CONTROL. Arm B is a hand-written template, so a transcription
error would be indistinguishable from a register win. `na6_pair6` is the same
hand-written template instantiated at chunk width 6, which is the baseline
algorithm expressed through the chunk loop. It must reproduce
`na6_baseline` exactly. If it does not, the transcription is wrong and every
arm-B row is void.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch  # noqa: E402
import e120_g17s_census as base  # noqa: E402

# `qwen_e120_qmv_wide` with the NA columns walked in chunks of `CHUNK`.
# Everything outside the chunk loop is copied from the shipped template.
PAIR_TEMPLATE = """
    template <int NA, int RPS, bool USE_TABLE, int CHUNK>
    inline void qwen_e120_qmv_wide_pairs(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
        const int sums_stride,
        int first_m,
        int out_row,
        uint simd_lid
    ) {
        static_assert(NA % CHUNK == 0, "the chunk must divide the width");
        typedef vec<float, NA> VF;
        typedef vec<float, CHUNK> VC;
        constexpr int rows_per_simd = RPS;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int bytes_per_lane = 8;
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;

        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] = VF(0.0f);
        }

        for (int k = 0; k < in_vec_size; k += block_size) {
            thread uint16_t packed[rows_per_simd][4];
            for (int r = 0; r < rows_per_simd; r++) {
                const int row = out_row + r;
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }
            }

            for (int c = 0; c < NA; c += CHUNK) {
                VC sums = VC(0.0f);
                VC partial[rows_per_simd];
                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] = VC(0.0f);
                }
                for (int i = 0; i < 4; i++) {
                    VC a0, a1, a2, a3;
                    for (int m = 0; m < CHUNK; m++) {
                        const device bfloat16_t* xm =
                            x + (first_m + c + m) * in_vec_size + k +
                            simd_lid * values_per_thread + 4 * i;
                        const vec<bfloat16_t, 4> xv =
                            *reinterpret_cast<
                                const device vec<bfloat16_t, 4>*>(xm);
                        a0[m] = static_cast<float>(xv[0]);
                        a1[m] = static_cast<float>(xv[1]);
                        a2[m] = static_cast<float>(xv[2]);
                        a3[m] = static_cast<float>(xv[3]);
                        if (!USE_TABLE) {
                            sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                        }
                    }
                    for (int r = 0; r < rows_per_simd; r++) {
                        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                                       a3 * ((packed[r][i] >> 12) & 0x000f));
                    }
                }
                if (USE_TABLE) {
                    const device float* st =
                        xsums + ((k / block_size) * 32 + int(simd_lid)) *
                        sums_stride + first_m + c;
                    for (int m = 0; m < CHUNK; m++) {
                        sums[m] = st[m];
                    }
                }
                for (int r = 0; r < rows_per_simd; r++) {
                    const int group_index =
                        (out_row + r) * in_vec_size_g + k / 64 +
                        int(simd_lid) / 4;
                    const float scale_local_r = scales[group_index];
                    const float bias_local_r = biases[group_index];
                    for (int m = 0; m < CHUNK; m++) {
                        acc[r][c + m] += scale_local_r * partial[r][m] +
                            sums[m] * bias_local_r;
                    }
                }
            }
        }

        for (int r = 0; r < rows_per_simd; r++) {
            for (int m = 0; m < NA; m++) {
                const float reduced = simd_sum(acc[r][m]);
                if (simd_lid == 0) {
                    y[(first_m + m) * out_vec_size + out_row + r] =
                        static_cast<bfloat16_t>(reduced);
                }
            }
        }
    }

    template <int M, int IPG, int RPS, bool USE_TABLE, int CHUNK>
    inline void qwen_e120_qmv_m_pairs(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
        const int sums_stride,
        int group_x,
        int out_row,
        uint simd_lid
    ) {
        static_assert(M % IPG != 1, "a one-input tail group is not built");
        constexpr int TAIL = M % IPG;
        const int first_m = group_x * IPG;
        if (first_m >= M) {
            return;
        }
        if (TAIL == 0 || M - first_m >= IPG) {
            qwen_e120_qmv_wide_pairs<IPG, RPS, USE_TABLE, CHUNK>(
                w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
                sums_stride, first_m, out_row, simd_lid);
        } else {
            qwen_e120_qmv_wide<(TAIL >= 2 ? TAIL : 2), RPS, USE_TABLE>(
                w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
                sums_stride, first_m, out_row, simd_lid);
        }
    }
"""

# The same chunked body with `acc` held as a plain float array rather than a
# `vec<float, NA>`. The arithmetic and the per-column reduction order are
# identical; only the storage declaration changes, so the compiler is free to
# allocate the accumulators independently instead of as one vector.
SCALARACC_TEMPLATE = """
    template <int NA, int RPS, bool USE_TABLE, int CHUNK>
    inline void qwen_e120_qmv_wide_scalaracc(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
        const int sums_stride,
        int first_m,
        int out_row,
        uint simd_lid
    ) {
        static_assert(NA % CHUNK == 0, "the chunk must divide the width");
        typedef vec<float, CHUNK> VC;
        constexpr int rows_per_simd = RPS;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int bytes_per_lane = 8;
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;

        float acc[rows_per_simd][NA];
        for (int r = 0; r < rows_per_simd; r++) {
            for (int m = 0; m < NA; m++) {
                acc[r][m] = 0.0f;
            }
        }

        for (int k = 0; k < in_vec_size; k += block_size) {
            thread uint16_t packed[rows_per_simd][4];
            for (int r = 0; r < rows_per_simd; r++) {
                const int row = out_row + r;
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }
            }

            for (int c = 0; c < NA; c += CHUNK) {
                VC sums = VC(0.0f);
                VC partial[rows_per_simd];
                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] = VC(0.0f);
                }
                for (int i = 0; i < 4; i++) {
                    VC a0, a1, a2, a3;
                    for (int m = 0; m < CHUNK; m++) {
                        const device bfloat16_t* xm =
                            x + (first_m + c + m) * in_vec_size + k +
                            simd_lid * values_per_thread + 4 * i;
                        const vec<bfloat16_t, 4> xv =
                            *reinterpret_cast<
                                const device vec<bfloat16_t, 4>*>(xm);
                        a0[m] = static_cast<float>(xv[0]);
                        a1[m] = static_cast<float>(xv[1]);
                        a2[m] = static_cast<float>(xv[2]);
                        a3[m] = static_cast<float>(xv[3]);
                        if (!USE_TABLE) {
                            sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                        }
                    }
                    for (int r = 0; r < rows_per_simd; r++) {
                        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                                       a3 * ((packed[r][i] >> 12) & 0x000f));
                    }
                }
                if (USE_TABLE) {
                    const device float* st =
                        xsums + ((k / block_size) * 32 + int(simd_lid)) *
                        sums_stride + first_m + c;
                    for (int m = 0; m < CHUNK; m++) {
                        sums[m] = st[m];
                    }
                }
                for (int r = 0; r < rows_per_simd; r++) {
                    const int group_index =
                        (out_row + r) * in_vec_size_g + k / 64 +
                        int(simd_lid) / 4;
                    const float scale_local_r = scales[group_index];
                    const float bias_local_r = biases[group_index];
                    for (int m = 0; m < CHUNK; m++) {
                        acc[r][c + m] += scale_local_r * partial[r][m] +
                            sums[m] * bias_local_r;
                    }
                }
            }
        }

        for (int r = 0; r < rows_per_simd; r++) {
            for (int m = 0; m < NA; m++) {
                const float reduced = simd_sum(acc[r][m]);
                if (simd_lid == 0) {
                    y[(first_m + m) * out_vec_size + out_row + r] =
                        static_cast<bfloat16_t>(reduced);
                }
            }
        }
    }

    template <int M, int IPG, int RPS, bool USE_TABLE, int CHUNK>
    inline void qwen_e120_qmv_m_scalaracc(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
        const int sums_stride,
        int group_x,
        int out_row,
        uint simd_lid
    ) {
        static_assert(M % IPG == 0, "this census arm builds no tail group");
        const int first_m = group_x * IPG;
        if (first_m >= M) {
            return;
        }
        qwen_e120_qmv_wide_scalaracc<IPG, RPS, USE_TABLE, CHUNK>(
            w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
            sums_stride, first_m, out_row, simd_lid);
    }
"""

DISPATCH = """    const int qmv_m = x_shape[x_ndim - 2];
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = qmv_m <= 8 ? 8 : 16;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_gx = int(qmv_tid.x);
    switch (qmv_m) {
        case %(m)d:
            %(callee)s<%(m)d, %(ipg)d, %(rps)d, USE_TABLE%(extra)s>(
                w, scales, biases, x, xsums, y,
                qmv_k, qmv_n, qmv_stride,
                qmv_gx,
                int(qmv_tid.y) * %(ty)d + int(qmv_sgid) * %(rps)d,
                qmv_lid);
            break;
        default:
            break;
    }"""


def dispatch(m: int, ipg: int, rps: int, callee: str, extra: str = "") -> str:
    return DISPATCH % {"m": m, "ipg": ipg, "rps": rps, "callee": callee,
                       "ty": 2 * rps, "extra": extra}


def kernel(name: str, body: str, attribute: str = "") -> str:
    text = base.generate(name, base.QMV_INPUTS + [("xsums", "float")],
                         base.QMV_OUTPUTS, body,
                         [("bool", "USE_TABLE", "true")])
    if attribute:
        text = text.replace("[[kernel]] void %s(" % name,
                            "[[kernel]] %s void %s(" % (attribute, name), 1)
    return text


def library(header: str, name: str, body: str, attribute: str = "") -> str:
    return "\n".join([base.PRELUDE, header, "", PAIR_TEMPLATE, "",
                      SCALARACC_TEMPLATE, "",
                      kernel(name, body, attribute)]) + "\n"


# The recorded two-host figures this census must reproduce before any arm-B row
# is read, from advisor F22 section 1.
EXPECTED = {
    "na3_control": {"applegpu_g16s": 94, "applegpu_g17s": 94},
    "na6_baseline": {"applegpu_g16s": 96, "applegpu_g17s": 105},
}
TARGET_REGISTERS = 96

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("research/out/e135-f22-na6-census.json"))
parser.add_argument("--keep", type=pathlib.Path,
                    help="write the reproduced Metal sources here")
args = parser.parse_args()

header = base.swift_literal("qwen35E120QMVHeader")
name = "qwen35_custom_affine4_g64_qmv_wide_sums_v2_na6"

arms = {
    "na3_control": library(header, name,
                           dispatch(6, 3, 4, "qwen_e120_qmv_m")),
    "na6_baseline": library(header, name,
                            dispatch(6, 6, 4, "qwen_e120_qmv_m")),
    "na6_maxtpt64": library(header, name,
                            dispatch(6, 6, 4, "qwen_e120_qmv_m"),
                            "[[max_total_threads_per_threadgroup(64)]]"),
    "na6_pair6": library(header, name,
                         dispatch(6, 6, 4, "qwen_e120_qmv_m_pairs", ", 6")),
    "na6_pair3": library(header, name,
                         dispatch(6, 6, 4, "qwen_e120_qmv_m_pairs", ", 3")),
    "na6_pair2": library(header, name,
                         dispatch(6, 6, 4, "qwen_e120_qmv_m_pairs", ", 2")),
    "na6_pair1": library(header, name,
                         dispatch(6, 6, 4, "qwen_e120_qmv_m_pairs", ", 1")),
    # `na6_pair1` lands at 98, two registers above the target, so the two
    # cheapest remaining levers are tried against it rather than against the
    # baseline. Neither is a new mechanism.
    "na6_pair1_tpt64": library(header, name,
                               dispatch(6, 6, 4, "qwen_e120_qmv_m_pairs",
                                        ", 1"),
                               "[[max_total_threads_per_threadgroup(64)]]"),
    "na6_pair1_scalaracc": library(
        header, name,
        dispatch(6, 6, 4, "qwen_e120_qmv_m_scalaracc", ", 1")),
}

result = {
    "harness": "local",
    "instrument": "xcrun metal-tt, AGX backend, zero GPU seconds",
    "timing_valid": False,
    "gate_qualified_for_timing": False,
    "official_or_ranked_score": False,
    "simdgroup_budget": base.SIMDGROUP_BUDGET,
    "target_g17s_registers": TARGET_REGISTERS,
    "expected_from_f22": EXPECTED,
    "arms": {},
}

with tempfile.TemporaryDirectory() as tmp:
    workdir = pathlib.Path(tmp)
    for tag, source in arms.items():
        if args.keep:
            args.keep.mkdir(parents=True, exist_ok=True)
            (args.keep / ("%s.metal" % tag)).write_text(source)
        try:
            result["arms"][tag] = base.census(source, tag, workdir)
        except subprocess.CalledProcessError as error:
            result["arms"][tag] = {
                "error": (error.stderr or b"").decode()[-3000:]}

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def row(tag, arch):
    entry = result["arms"].get(tag, {})
    if "error" in entry:
        return None
    return entry.get(arch, {}).get(name)


print("%-14s %-6s %9s %7s %9s %11s" % (
    "arm", "arch", "registers", "spill", "text B", "simdgroups"))
for tag in arms:
    if "error" in result["arms"][tag]:
        print("%-14s BUILD FAILED" % tag)
        print(result["arms"][tag]["error"])
        continue
    for arch in base.ARCHS:
        record = row(tag, arch)
        print("%-14s %-6s %9d %7d %9d %11d" % (
            tag, arch.replace("applegpu_", ""), record["registers"],
            record["spill_bytes"], record["text_bytes"],
            record["resident_simdgroups"]))

print()
verdict = 0
for tag, want in EXPECTED.items():
    for arch, registers in want.items():
        got = row(tag, arch)
        got = got and got["registers"]
        state = "ok  " if got == registers else "FAIL"
        if got != registers:
            verdict = 1
        print("%s channel check %-14s %-14s expected %d, read %s"
              % (state, tag, arch, registers, got))

six = row("na6_pair6", "applegpu_g17s")
zero = row("na6_baseline", "applegpu_g17s")
if six and zero:
    same = six["registers"] == zero["registers"]
    print("%s transcription control: na6_pair6 reads %d and na6_baseline "
          "reads %d on g17s" % ("ok  " if same else "FAIL",
                                six["registers"], zero["registers"]))
    if not same:
        verdict = 1

print()
best = None
for tag in arms:
    if tag in ("na3_control", "na6_baseline", "na6_pair6"):
        continue
    record = row(tag, "applegpu_g17s")
    if not record:
        continue
    if record["registers"] <= TARGET_REGISTERS and record["spill_bytes"] == 0:
        if best is None or record["registers"] < best[1]["registers"]:
            best = (tag, record)

# CAMPAIGN RULE 82, reported before any timing. The recorded F47 weight table
# is indexed by NA over 2, 3, 4, 5 and has no NA = 6 cell, because it predates
# the one-pass tier. This change moves width 6 from NA = 3 to NA = 6, so the
# table cannot price it and substituting one of its cells would be a different
# denominator. The weight used instead is the realised width-6 round mass on
# the two prompts that set the published median, from advisor F24 section 5.
WIDTH6_MASS = {"beagle": 0.0739, "medicine": 0.0641}
SHIPPED = "na3_control"

shipped = row(SHIPPED, "applegpu_g17s")
print("\ng17s residency against the shipped %s, and the Rule 82 weighted "
      "delta at the realised width-6 mass %s:" % (SHIPPED, WIDTH6_MASS))
mass = sum(WIDTH6_MASS.values()) / len(WIDTH6_MASS)
for tag in arms:
    record = row(tag, "applegpu_g17s")
    if not record or tag == SHIPPED:
        continue
    cell = (record["resident_simdgroups"] / shipped["resident_simdgroups"]
            - 1) * 100
    print("  %-20s %2d simdgroups, cell %+7.2f %%, weighted %+7.3f %%%s"
          % (tag, record["resident_simdgroups"], cell, cell * mass,
             "   <- Rule 82 veto" if cell < 0 else ""))

if best is None:
    print("\nKILL RULE FIRES: no arm reaches %d or fewer g17s registers "
          "without spilling. Close the arm." % TARGET_REGISTERS)
else:
    tag, record = best
    gain = record["resident_simdgroups"] - zero["resident_simdgroups"]
    print("KILL RULE DOES NOT FIRE: %s reads %d registers, spill %d, %d "
          "resident simdgroups, %+d against na6_baseline and %+d against the "
          "na3 control." % (
              tag, record["registers"], record["spill_bytes"],
              record["resident_simdgroups"], gain,
              record["resident_simdgroups"]
              - row("na3_control", "applegpu_g17s")["resident_simdgroups"]))

sys.exit(verdict)
