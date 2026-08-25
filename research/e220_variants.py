#!/usr/bin/env python3
"""E220 integer-path variants, as textual transforms of the live QMV header.

Each variant is an exact substitution applied to `qwen35E120QMVHeader` read
live out of `Qwen35.swift`, so a variant cannot silently carry a stale copy of
the kernel and cannot change any line it does not name. A substitution that
does not match raises, so a base edit that moves the target text fails the
census instead of censusing the wrong source.

Every variant changes only integer, address or load instructions. The
floating-point expression tree is byte-identical in every case, which
`e220_census.py` checks through the ordered FP opcode digest of the dequant
block.

  bfe       nibble extraction through `extract_bits` instead of shift+mask.
            `extract_bits` is unsigned, then cast back to `int`, so the
            operand handed to the FP tree is still a signed `int` in 0..15 and
            the conversion instruction is unchanged.

  vecload   the four packed 16-bit weight words per row read as one 8-byte
            `vec<uint16_t, 4>` load instead of four 2-byte loads. The address
            is `w + row * (K/2) + k/2 + simd_lid * 8` bytes; `K` is 5120, 6144
            or 17408 for every scored cell, so `K/2` is a multiple of 8, `k` is
            a multiple of 512 and the lane term is a multiple of 8. The load is
            therefore 8-byte aligned at every scored cell.

  hoist     k-loop address strength reduction. The weight, scale/bias, x and
            xsums addresses are all affine in `k`, but the source recomputes
            `row * (K/2)`, `row * (K/64)` and `(first_m + m) * K` inside the k
            loop. This variant computes each base once and advances it by a
            constant stride per k block, so the k loop carries no multiply.

  all       bfe + vecload + hoist.
"""

from __future__ import annotations

# (needle, replacement) pairs. The needles are copied verbatim out of
# `qwen35E120QMVHeader`.

BFE = [
    (
        """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * ((packed[r][i] >> 4) & 0x000f) +
                                   a2 * ((packed[r][i] >> 8) & 0x000f) +
                                   a3 * ((packed[r][i] >> 12) & 0x000f));
                }""",
        """                for (int r = 0; r < rows_per_simd; r++) {
                    const uint pw = uint(packed[r][i]);
                    partial[r] += (a0 * int(extract_bits(pw, 0, 4)) +
                                   a1 * int(extract_bits(pw, 4, 4)) +
                                   a2 * int(extract_bits(pw, 8, 4)) +
                                   a3 * int(extract_bits(pw, 12, 4)));
                }""",
    ),
]

VECLOAD = [
    (
        """                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }""",
        """                const device vec<uint16_t, 4>* ws =
                    reinterpret_cast<const device vec<uint16_t, 4>*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                const vec<uint16_t, 4> wv = *ws;
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = wv[i];
                }""",
    ),
]

HOIST = [
    # Bases computed once, outside the k loop.
    (
        """        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] = VF(0.0f);
        }

        for (int k = 0; k < in_vec_size; k += block_size) {""",
        """        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] = VF(0.0f);
        }

        // Every address in the k loop is affine in k. Each base is formed once
        // and advanced by a constant stride, so the loop carries no multiply.
        thread const device uint8_t* w_row[rows_per_simd];
        thread int group_row[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            const int row = out_row + r;
            w_row[r] = reinterpret_cast<const device uint8_t*>(w) +
                row * in_vec_size_w + simd_lid * bytes_per_lane;
            group_row[r] = row * in_vec_size_g + int(simd_lid) / 4;
        }
        thread const device bfloat16_t* x_row[NA];
        for (int m = 0; m < NA; m++) {
            x_row[m] = x + (first_m + m) * in_vec_size +
                simd_lid * values_per_thread;
        }
        const device float* sums_row =
            xsums + int(simd_lid) * sums_stride + first_m;
        constexpr int w_step = block_size / 2;
        constexpr int group_step = block_size / 64;
        constexpr int sums_step = 32;

        for (int k = 0; k < in_vec_size; k += block_size) {""",
    ),
    # The weight and scale/bias reads now step their own bases.
    (
        """            for (int r = 0; r < rows_per_simd; r++) {
                const int row = out_row + r;
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }
                const int group_index =
                    row * in_vec_size_g + k / 64 + int(simd_lid) / 4;
                scale_local[r] = scales[group_index];
                bias_local[r] = biases[group_index];
            }""",
        """            for (int r = 0; r < rows_per_simd; r++) {
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(w_row[r]);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }
                const int group_index = group_row[r];
                scale_local[r] = scales[group_index];
                bias_local[r] = biases[group_index];
                w_row[r] += w_step;
                group_row[r] += group_step;
            }""",
    ),
    (
        """            VF sums = VF(0.0f);
            if (USE_TABLE) {
                const device float* st =
                    xsums + ((k / block_size) * 32 + int(simd_lid)) *
                    sums_stride + first_m;
                for (int m = 0; m < NA; m++) {
                    sums[m] = st[m];
                }
            }""",
        """            VF sums = VF(0.0f);
            if (USE_TABLE) {
                const device float* st = sums_row;
                for (int m = 0; m < NA; m++) {
                    sums[m] = st[m];
                }
                sums_row += sums_step * sums_stride;
            }""",
    ),
    (
        """                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;""",
        """                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm = x_row[m] + 4 * i;""",
    ),
    # The x bases advance once per k block, after the element loop has read
    # them. `k` itself is no longer an address term.
    (
        """            for (int r = 0; r < rows_per_simd; r++) {
                acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
            }
        }""",
        """            for (int r = 0; r < rows_per_simd; r++) {
                acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
            }
            for (int m = 0; m < NA; m++) {
                x_row[m] += block_size;
            }
        }""",
    ),
]

# `hoist` puts one base per row and one per input in private arrays, which is
# itself a register-pressure claim. `hoist_scalar` removes the same k-invariant
# terms without introducing any array, so the address axis is tested without
# the confound of a new private allocation.
HOIST_SCALAR = [
    (
        """        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] = VF(0.0f);
        }

        for (int k = 0; k < in_vec_size; k += block_size) {""",
        """        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] = VF(0.0f);
        }

        const device uint8_t* w_base =
            reinterpret_cast<const device uint8_t*>(w) +
            out_row * in_vec_size_w + simd_lid * bytes_per_lane;
        const int group_base = out_row * in_vec_size_g + int(simd_lid) / 4;
        const device bfloat16_t* x_base =
            x + first_m * in_vec_size + simd_lid * values_per_thread;

        for (int k = 0; k < in_vec_size; k += block_size) {""",
    ),
    (
        """            for (int r = 0; r < rows_per_simd; r++) {
                const int row = out_row + r;
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }
                const int group_index =
                    row * in_vec_size_g + k / 64 + int(simd_lid) / 4;
                scale_local[r] = scales[group_index];
                bias_local[r] = biases[group_index];
            }""",
        """            for (int r = 0; r < rows_per_simd; r++) {
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        w_base + r * in_vec_size_w + k / 2);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }
                const int group_index =
                    group_base + r * in_vec_size_g + k / 64;
                scale_local[r] = scales[group_index];
                bias_local[r] = biases[group_index];
            }""",
    ),
    (
        """                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;""",
        """                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x_base + m * in_vec_size + k + 4 * i;""",
    ),
]

# Eight nibbles extracted from one 32-bit word instead of four from each of two
# 16-bit words. This is the paired-extraction form: it removes the two 16-to-32
# bit widenings per eight values and lets one register hold both halves.
WORD32 = [
    (
        """            thread uint16_t packed[rows_per_simd][4];""",
        """            thread uint32_t packed[rows_per_simd][2];""",
    ),
    (
        """                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }""",
        """                const device uint32_t* ws =
                    reinterpret_cast<const device uint32_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 2; i++) {
                    packed[r][i] = ws[i];
                }""",
    ),
    (
        """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * ((packed[r][i] >> 4) & 0x000f) +
                                   a2 * ((packed[r][i] >> 8) & 0x000f) +
                                   a3 * ((packed[r][i] >> 12) & 0x000f));
                }""",
        """                for (int r = 0; r < rows_per_simd; r++) {
                    const int shift = 16 * (i & 1);
                    const uint pw = packed[r][i / 2] >> shift;
                    partial[r] += (a0 * int(pw & 0x000f) +
                                   a1 * int((pw >> 4) & 0x000f) +
                                   a2 * int((pw >> 8) & 0x000f) +
                                   a3 * int((pw >> 12) & 0x000f));
                }""",
    ),
]

# `hoist` rewrites the same weight-load block `vecload` targets, so the
# combination applies `hoist` first and then vectorizes the hoisted load.
VECLOAD_AFTER_HOIST = [
    (
        """                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(w_row[r]);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }""",
        """                const vec<uint16_t, 4> wv =
                    *reinterpret_cast<const device vec<uint16_t, 4>*>(w_row[r]);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = wv[i];
                }""",
    ),
]

# COST ORACLE, NOT A CANDIDATE. Every nibble is read from bit 0, so the three
# shifts and two of the three masks fold away under CSE and the per-4-value
# extraction drops from six integer operations to one. The kernel computes the
# WRONG ANSWER by construction; it exists only to price the extraction inside
# the real kernel, so that "already minimal" can be stated as a bound rather
# than as a null result. It must never be timed, gated or shipped.
PROBE_ZERO_SHIFT = [
    (
        """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * ((packed[r][i] >> 4) & 0x000f) +
                                   a2 * ((packed[r][i] >> 8) & 0x000f) +
                                   a3 * ((packed[r][i] >> 12) & 0x000f));
                }""",
        """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * (packed[r][i] & 0x000f) +
                                   a2 * (packed[r][i] & 0x000f) +
                                   a3 * (packed[r][i] & 0x000f));
                }""",
    ),
]

# COST ORACLE, NOT A CANDIDATE. Four DISTINCT values, each reachable in ONE
# integer operation, so the extraction costs four operations per four values
# instead of six while the floating-point tree stays exactly as it is: four
# separate conversions, four separate broadcasts, one fmul and three fmas. This
# is the code a machine with a free single-instruction nibble extract would
# emit, so the base-minus-oracle difference is the WHOLE prize available to any
# nibble-extraction rewrite. The values are scaled wrong on purpose, so the
# kernel computes the WRONG ANSWER and must never be timed, gated or shipped.
PROBE_MIN_EXTRACT = [
    (
        """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * ((packed[r][i] >> 4) & 0x000f) +
                                   a2 * ((packed[r][i] >> 8) & 0x000f) +
                                   a3 * ((packed[r][i] >> 12) & 0x000f));
                }""",
        """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * (packed[r][i] & 0x00f0) +
                                   a2 * (packed[r][i] & 0x0f00) +
                                   a3 * (packed[r][i] & 0xf000));
                }""",
    ),
]

VARIANTS = {
    "base": [],
    "probe_zero_shift": PROBE_ZERO_SHIFT,
    "probe_min_extract": PROBE_MIN_EXTRACT,
    "bfe": BFE,
    "vecload": VECLOAD,
    "hoist": HOIST,
    "hoist_scalar": HOIST_SCALAR,
    "word32": WORD32,
    "hoist_vecload": HOIST + VECLOAD_AFTER_HOIST,
    "all": BFE + HOIST + VECLOAD_AFTER_HOIST,
}


def apply(header: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}; "
                         f"have {sorted(VARIANTS)}")
    out = header
    for needle, replacement in VARIANTS[variant]:
        if out.count(needle) != 1:
            raise SystemExit(
                f"variant {variant}: needle matched {out.count(needle)} times, "
                f"expected 1:\n{needle}")
        out = out.replace(needle, replacement)
    return out
