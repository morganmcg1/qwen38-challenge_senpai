#!/usr/bin/env python3
"""E223 stage 0.3: candidate activation access paths, as source transforms.

Each variant rewrites the activation gather of the live `qwen35E120QMVHeader`
and nothing else. The floating-point expression tree, its order and its types
stay exactly as shipped, so a variant that survives the desk can go to the
0-ULP gate unchanged in meaning.

The variants exist to be COMPILED, not run: `research/e223_census.py --variant`
reports what each one costs in real AGX machine code (instructions per column
per k-block, registers, spill bytes) on `applegpu_g16s` and `applegpu_g17s`.
A variant that does not shrink the per-column instruction slope is dead before
any GPU second is spent.

  base       shipped text, unmodified
  hoistbase  the per-column base pointer `(first_m + m) * in_vec_size` is
             computed once per dispatch instead of once per (i, m). Removes
             per-column address arithmetic only.
  wideload   one 16-byte `vec<bfloat16_t,8>` per column per half-i-step instead
             of two 8-byte `vec<bfloat16_t,4>` loads. Halves the activation
             load count and doubles the live activation registers.
  xtf32      the NA-contiguous path: activations are read from a transposed
             float32 slab `xT[k][m]`, so one contiguous NA-wide load per
             `a0..a3` per i-step replaces NA strided loads plus 4*NA bf16 to
             f32 conversions. COMPILE-ONLY PROBE: the slab is passed through
             the `xsums` buffer, so the probe measures instruction structure,
             never a value. It is compiled at `USE_TABLE = false` only.
  narrowload METHOD CONTROL, not a candidate. The same four values per column
             per i-step are read as four separate 2-byte scalar loads instead
             of one 8-byte `vec<bfloat16_t,4>`. Values and order are identical.
             If the witness sees no cost, the backend re-coalesces narrow
             source loads and the activation load count is not a source-level
             lever. If the witness sees +3 loads per column, the witness
             resolves single-instruction-per-column effects and `wideload`'s
             null result is a real null.
  xf32flat   the convert lever in its cheapest form: activations are already
             float32 in the SHIPPED [m, K] layout, so the 4*NA bf16 to f32
             conversions per i-step disappear and no transpose is needed. It
             doubles the activation bytes the kernel issues. COMPILE-ONLY
             PROBE: the f32 slab is passed through the `xsums` argument, so the
             probe measures instruction structure, never a value.
"""

from __future__ import annotations

NAMES = ("base", "hoistbase", "wideload", "xtf32", "narrowload", "xf32flat")

# The shipped gather, verbatim. Every transform below replaces this exact text,
# and the replacement count is asserted, so a base move breaks the variant
# loudly instead of silently patching the wrong lines.
SHIPPED_GATHER = """            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;
                    const vec<bfloat16_t, 4> xv =
                        *reinterpret_cast<const device vec<bfloat16_t, 4>*>(
                            xm);
                    a0[m] = static_cast<float>(xv[0]);
                    a1[m] = static_cast<float>(xv[1]);
                    a2[m] = static_cast<float>(xv[2]);
                    a3[m] = static_cast<float>(xv[3]);
                    if (!USE_TABLE) {
                        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                    }
                }"""

HOISTBASE_PROLOGUE = """        const device bfloat16_t* xbase[NA];
        for (int m = 0; m < NA; m++) {
            xbase[m] = x + (first_m + m) * in_vec_size +
                simd_lid * values_per_thread;
        }
"""

HOISTBASE_GATHER = """            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm = xbase[m] + k + 4 * i;
                    const vec<bfloat16_t, 4> xv =
                        *reinterpret_cast<const device vec<bfloat16_t, 4>*>(
                            xm);
                    a0[m] = static_cast<float>(xv[0]);
                    a1[m] = static_cast<float>(xv[1]);
                    a2[m] = static_cast<float>(xv[2]);
                    a3[m] = static_cast<float>(xv[3]);
                    if (!USE_TABLE) {
                        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                    }
                }"""

# Two i-steps per iteration. The eight values a column contributes are the same
# eight values in the same order, and each is inserted into the same lane of the
# same accumulator, so the arithmetic is untouched.
WIDELOAD_GATHER = """            for (int i = 0; i < 4; i += 2) {
                VF a0, a1, a2, a3, b0, b1, b2, b3;
                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;
                    const vec<bfloat16_t, 8> xv =
                        *reinterpret_cast<const device vec<bfloat16_t, 8>*>(
                            xm);
                    a0[m] = static_cast<float>(xv[0]);
                    a1[m] = static_cast<float>(xv[1]);
                    a2[m] = static_cast<float>(xv[2]);
                    a3[m] = static_cast<float>(xv[3]);
                    b0[m] = static_cast<float>(xv[4]);
                    b1[m] = static_cast<float>(xv[5]);
                    b2[m] = static_cast<float>(xv[6]);
                    b3[m] = static_cast<float>(xv[7]);
                    if (!USE_TABLE) {
                        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                        sums[m] += xv[4] + xv[5] + xv[6] + xv[7];
                    }
                }"""

# The second half of the unrolled pair reuses the shipped row loop body, so the
# transform also duplicates the FMA block. `wideload` therefore needs the row
# loop text as well.
SHIPPED_ROWS = """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * ((packed[r][i] >> 4) & 0x000f) +
                                   a2 * ((packed[r][i] >> 8) & 0x000f) +
                                   a3 * ((packed[r][i] >> 12) & 0x000f));
                }
            }"""

WIDELOAD_ROWS = """                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * ((packed[r][i] >> 4) & 0x000f) +
                                   a2 * ((packed[r][i] >> 8) & 0x000f) +
                                   a3 * ((packed[r][i] >> 12) & 0x000f));
                    partial[r] += (b0 * (packed[r][i + 1] & 0x000f) +
                                   b1 * ((packed[r][i + 1] >> 4) & 0x000f) +
                                   b2 * ((packed[r][i + 1] >> 8) & 0x000f) +
                                   b3 * ((packed[r][i + 1] >> 12) & 0x000f));
                }
            }"""

# `xsums` carries the transposed slab in this probe, so the slab pointer is
# named once and the gather reads `xT[(k + 4*i + j) * NA + m]` in one contiguous
# NA-wide run per accumulator.
XTF32_GATHER = """            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                const device float* xt =
                    xsums + (k + simd_lid * values_per_thread + 4 * i) * NA;
                for (int m = 0; m < NA; m++) {
                    a0[m] = xt[m];
                    a1[m] = xt[NA + m];
                    a2[m] = xt[2 * NA + m];
                    a3[m] = xt[3 * NA + m];
                }"""


NARROWLOAD_GATHER = """            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;
                    const bfloat16_t xv0 = xm[0];
                    const bfloat16_t xv1 = xm[1];
                    const bfloat16_t xv2 = xm[2];
                    const bfloat16_t xv3 = xm[3];
                    a0[m] = static_cast<float>(xv0);
                    a1[m] = static_cast<float>(xv1);
                    a2[m] = static_cast<float>(xv2);
                    a3[m] = static_cast<float>(xv3);
                    if (!USE_TABLE) {
                        sums[m] += xv0 + xv1 + xv2 + xv3;
                    }
                }"""


XF32FLAT_GATHER = """            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device float* xm =
                        xsums + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;
                    const vec<float, 4> xv =
                        *reinterpret_cast<const device vec<float, 4>*>(xm);
                    a0[m] = xv[0];
                    a1[m] = xv[1];
                    a2[m] = xv[2];
                    a3[m] = xv[3];
                }"""


def apply(header: str, variant: str) -> str:
    """The header with one activation-access transform applied."""
    if variant == "base":
        return header
    if variant not in NAMES:
        raise SystemExit(f"e223: unknown variant {variant!r}")

    def swap(text: str, old: str, new: str) -> str:
        if text.count(old) != 1:
            raise SystemExit(
                f"e223: variant {variant} expected one match, found "
                f"{text.count(old)}")
        return text.replace(old, new)

    if variant == "hoistbase":
        out = swap(header, SHIPPED_GATHER, HOISTBASE_GATHER)
        return swap(out, "        for (int k = 0; k < in_vec_size;",
                    HOISTBASE_PROLOGUE
                    + "        for (int k = 0; k < in_vec_size;")
    if variant == "wideload":
        out = swap(header, SHIPPED_GATHER, WIDELOAD_GATHER)
        return swap(out, SHIPPED_ROWS, WIDELOAD_ROWS)
    if variant == "narrowload":
        return swap(header, SHIPPED_GATHER, NARROWLOAD_GATHER)
    if variant == "xf32flat":
        return swap(header, SHIPPED_GATHER, XF32FLAT_GATHER)
    return swap(header, SHIPPED_GATHER, XTF32_GATHER)
