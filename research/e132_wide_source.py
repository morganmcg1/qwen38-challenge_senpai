#!/usr/bin/env python3
"""Generate `qwen_e120_qmv_wide` variants that are bit-exact by construction.

`Qwen35.swift` is owned by thorfinn this round, so E132 never edits it. This
module reads the shipped `qwen_e120_qmv_wide` template out of the Swift string
literal, regenerates it from parameters, and hands the result to the census.

The generator is trusted only because of `assert_faithful`: with every flag off
it must reproduce the shipped template BYTE FOR BYTE. Any variant therefore
differs from the shipped kernel only by the flags that were switched on, and the
diff a reader has to check is the flag, not a hand-retyped kernel.

RULE 92. No flag changes a floating-point operation or the order in which the
products of one output element are summed. Each flag carries its argument:

  msplit      Splits the NA activation columns into tiles of MSPLIT and runs
              the i-loop once per tile. For a fixed (r, m) the products are
              still added over i in the order 0,1,2,3 with the same expression,
              and `acc[r][m]` still accumulates over k-blocks in order. Columns
              m never combine with each other anywhere in the kernel, so
              reordering across m cannot move a rounding boundary. `packed` is
              loaded once per k-block and stays live across the tiles, and each
              activation element is still read exactly once, so neither weight
              nor activation traffic changes.
  late_meta   Moves the `scales`/`biases` loads from the top of the k-block to
              the accumulate loop that consumes them. Same addresses, same
              number of loads, same expression; only the live range shortens.
  rows2_at    Sets `rows_per_simd = 2` for one NA. `rows_per_simd` decides how
              many OUTPUT rows a simdgroup owns. Each output element still
              accumulates over exactly the same k values in the same order. The
              host must launch twice as many row groups.
  sink_sums   Moves the `xsums` table read below the i-loop. Under USE_TABLE
              `sums` is a pure load consumed only by the accumulate loop, so
              this is code motion with no arithmetic change. Under !USE_TABLE
              `sums` is accumulated inside the i-loop and the flag is ignored.
"""

from __future__ import annotations

import hashlib

# Digest of the shipped template span, as a guard against silent drift in
# `Qwen35.swift`. A mismatch means the generator must be re-derived, not that
# the census may proceed.
SHIPPED_SHA8 = "64cee0c6"

OPEN = "template <int NA, bool USE_TABLE>"
NEXT = "template <int M, int IPG, bool USE_TABLE>"


def wide_span(header: str) -> tuple[int, int]:
    start = header.index(OPEN)
    return start, header.index(NEXT, start)


def shipped_wide(header: str) -> str:
    start, end = wide_span(header)
    return header[start:end]


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _ind(lines: list[str], pad: int) -> list[str]:
    space = " " * pad
    return [space + line if line else "" for line in lines]


def _msplit_selector(msplit: dict[int, int]) -> list[str]:
    expr = "na"
    for na in sorted(msplit, reverse=True):
        expr = "na == %d ? %d : (%s)" % (na, msplit[na], expr)
    if expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1]
    return ["constexpr int qwen_e120_msplit(int na) {",
            "    return %s;" % expr,
            "}",
            ""]


def rows_per_simd(na: int, rows2_at: int | None = None, **_flags) -> int:
    """Output rows one simdgroup computes, under the variant's flags."""
    return 2 if rows2_at == na else 4


def output_elements_per_thread(na: int, **flags) -> int:
    """Output elements one thread accumulates across the whole k loop.

    A lane holds `rows_per_simd` accumulators of `NA` lanes each and the
    simdgroup reduces them at the end, so this is the denominator that makes a
    `rows_per_simd = 2` body comparable with a `rows_per_simd = 4` one.
    """
    return rows_per_simd(na, **flags) * na


def dynamic_ops(na: int, table: bool, **flags) -> dict:
    """Scalar-lane operations one thread issues per k block.

    The static instruction count that E132-F2 asks for is only comparable
    between two kernels the backend unrolled the same way. It does not compare
    `qwen_e120_qmv_wide<4>`, which the backend unrolls fully, with
    `qwen_e120_qmv_wide<8>`, which it does not. This model counts the work the
    SOURCE asks for with exact trip counts, so it is invariant to the unroll
    decision and is comparable across widths.

    It counts one operation per scalar lane of a `vec<float, N>` expression,
    because AGX has no wide vector ALU: a width-N vector operation issues N
    times. Address arithmetic is charged at the number of integer operations
    the expression spells out. It does not model spill, scheduling, latency or
    cache behaviour, so it is a work model and never a time model.
    """
    rows = rows_per_simd(na, **flags)
    split = flags.get("msplit") or {}
    m = split.get(na, na)
    tiles = na // m
    meta_repeats = tiles if flags.get("late_meta") else 1

    weight_loads = rows * 4
    weight_int = rows * 6
    meta_loads = meta_repeats * rows * 2
    meta_int = meta_repeats * rows * 3

    table_loads = tiles * m if table else 0
    table_int = tiles * 4 if table else 0

    act_loads = tiles * 4 * m
    act_int = tiles * 4 * m * 3
    act_cvt = tiles * 4 * m * 4
    sums_add = 0 if table else tiles * 4 * m * 4

    nibble_int = tiles * 4 * rows * 8
    nibble_cvt = tiles * 4 * rows * 4
    fma = tiles * 4 * rows * 4 * m
    combine = tiles * rows * 3 * m

    loads = weight_loads + meta_loads + table_loads + act_loads
    integer = weight_int + meta_int + table_int + act_int + nibble_int
    floating = act_cvt + sums_add + nibble_cvt + fma + combine
    total = loads + integer + floating
    elements = rows * na
    return {
        "loads": loads,
        "integer": integer,
        "floating": floating,
        "total": total,
        "output_elements_per_thread": elements,
        "ops_per_output_element": round(total / elements, 3),
    }


def wide_source(msplit: dict[int, int] | None = None,
                late_meta: bool = False,
                rows2_at: int | None = None,
                sink_sums: bool = False,
                shell_only: bool = False) -> str:
    """The `qwen_e120_qmv_wide` template, with every requested flag applied.

    `shell_only` keeps the signature, the accumulator declarations, the k loop
    control and the reduction epilogue, and replaces the k-block body with one
    dependent add per accumulator. The k loop therefore survives dead-code
    elimination and `simd_sum` cannot be constant folded, so the machine code
    that remains is the part of the kernel that is *not* the k-block body.
    Subtracting it from the full kernel isolates the inner-loop machine code
    that E132-F2 asks for. It is a measurement instrument only and is never a
    candidate.
    """
    tiled = msplit is not None
    vec = "VT" if tiled else "VF"
    span = "MSPLIT" if tiled else "NA"
    shift = " + base_m" if tiled else ""
    acc_t = "[t]" if tiled else ""

    out: list[str] = []
    if tiled:
        out += _msplit_selector(msplit)
        out.append("template <int NA, bool USE_TABLE, "
                   "int MSPLIT = qwen_e120_msplit(NA)>")
    else:
        out.append(OPEN)
    out += [
        "inline void qwen_e120_qmv_wide(",
        "    const device uint32_t* w,",
        "    const device bfloat16_t* scales,",
        "    const device bfloat16_t* biases,",
        "    const device bfloat16_t* x,",
        "    const device float* xsums,",
        "    device bfloat16_t* y,",
        "    const int in_vec_size,",
        "    const int out_vec_size,",
        "    const int sums_stride,",
        "    int first_m,",
        "    int out_row,",
        "    uint simd_lid",
        ") {",
    ]
    if tiled:
        out += [
            "    static_assert(NA % MSPLIT == 0, \"MSPLIT must divide NA\");",
            "    typedef vec<float, MSPLIT> VT;",
            "    constexpr int NTILE = NA / MSPLIT;",
        ]
    else:
        out.append("    typedef vec<float, NA> VF;")
    rows = ("(NA == %d) ? 2 : 4" % rows2_at) if rows2_at else "4"
    out += [
        "    constexpr int rows_per_simd = %s;" % rows,
        "    constexpr int values_per_thread = 16;",
        "    constexpr int block_size = values_per_thread * 32;",
        "    constexpr int bytes_per_lane = 8;",
        "    const int in_vec_size_w = in_vec_size / 2;",
        "    const int in_vec_size_g = in_vec_size / 64;",
        "",
    ]
    if tiled:
        out += [
            "    VT acc[rows_per_simd][NTILE];",
            "    for (int r = 0; r < rows_per_simd; r++) {",
            "        for (int t = 0; t < NTILE; t++) {",
            "            acc[r][t] = VT(0.0f);",
            "        }",
            "    }",
        ]
    else:
        out += [
            "    VF acc[rows_per_simd];",
            "    for (int r = 0; r < rows_per_simd; r++) {",
            "        acc[r] = VF(0.0f);",
            "    }",
        ]
    out += ["", "    for (int k = 0; k < in_vec_size; k += block_size) {"]
    if shell_only:
        out.append("        for (int r = 0; r < rows_per_simd; r++) {")
        if tiled:
            out.append("            for (int t = 0; t < NTILE; t++) {")
        out.append("            acc[r]%s += %s(float(k));" % (acc_t, vec))
        if tiled:
            out.append("            }")
        out += ["        }", "    }", ""]
        out += _epilogue(tiled)
        out += ["}", ""]
        return "\n".join(out) + "\n"
    out += ["        thread uint16_t packed[rows_per_simd][4];"]
    if not late_meta:
        out += [
            "        thread float scale_local[rows_per_simd];",
            "        thread float bias_local[rows_per_simd];",
        ]
    out += [
        "        for (int r = 0; r < rows_per_simd; r++) {",
        "            const int row = out_row + r;",
        "            const device uint16_t* ws =",
        "                reinterpret_cast<const device uint16_t*>(",
        "                    reinterpret_cast<const device uint8_t*>(w) +",
        "                    row * in_vec_size_w + k / 2 +",
        "                    simd_lid * bytes_per_lane);",
        "            for (int i = 0; i < 4; i++) {",
        "                packed[r][i] = ws[i];",
        "            }",
    ]
    if not late_meta:
        out += [
            "            const int group_index =",
            "                row * in_vec_size_g + k / 64 + int(simd_lid) / 4;",
            "            scale_local[r] = scales[group_index];",
            "            bias_local[r] = biases[group_index];",
        ]
    out += ["        }", ""]

    table_read = [
        "if (USE_TABLE) {",
        "    const device float* st =",
        "        xsums + ((k / block_size) * 32 + int(simd_lid)) *",
        "        sums_stride + first_m%s;" % shift,
        "    for (int m = 0; m < %s; m++) {" % span,
        "        sums[m] = st[m];",
        "    }",
        "}",
    ]
    tile: list[str] = ["%s sums = %s(0.0f);" % (vec, vec)]
    if not sink_sums:
        tile += table_read
    tile += [
        "%s partial[rows_per_simd];" % vec,
        "for (int r = 0; r < rows_per_simd; r++) {",
        "    partial[r] = %s(0.0f);" % vec,
        "}",
        "for (int i = 0; i < 4; i++) {",
        "    %s a0, a1, a2, a3;" % vec,
        "    for (int m = 0; m < %s; m++) {" % span,
        "        const device bfloat16_t* xm =",
        "            x + (first_m%s + m) * in_vec_size + k +" % shift,
        "            simd_lid * values_per_thread + 4 * i;",
        "        const vec<bfloat16_t, 4> xv =",
        "            *reinterpret_cast<const device vec<bfloat16_t, 4>*>(",
        "                xm);",
        "        a0[m] = static_cast<float>(xv[0]);",
        "        a1[m] = static_cast<float>(xv[1]);",
        "        a2[m] = static_cast<float>(xv[2]);",
        "        a3[m] = static_cast<float>(xv[3]);",
        "        if (!USE_TABLE) {",
        "            sums[m] += xv[0] + xv[1] + xv[2] + xv[3];",
        "        }",
        "    }",
        "    for (int r = 0; r < rows_per_simd; r++) {",
        "        partial[r] += (a0 * (packed[r][i] & 0x000f) +",
        "                       a1 * ((packed[r][i] >> 4) & 0x000f) +",
        "                       a2 * ((packed[r][i] >> 8) & 0x000f) +",
        "                       a3 * ((packed[r][i] >> 12) & 0x000f));",
        "    }",
        "}",
    ]
    if sink_sums:
        tile += table_read
    tile.append("for (int r = 0; r < rows_per_simd; r++) {")
    if late_meta:
        tile += [
            "    const int group_index =",
            "        (out_row + r) * in_vec_size_g + k / 64 + "
            "int(simd_lid) / 4;",
            "    const float scale_local_r = scales[group_index];",
            "    const float bias_local_r = biases[group_index];",
            "    acc[r]%s += scale_local_r * partial[r] + "
            "sums * bias_local_r;" % acc_t,
        ]
    else:
        tile.append("    acc[r]%s += scale_local[r] * partial[r] + "
                    "sums * bias_local[r];" % acc_t)
    tile.append("}")

    if tiled:
        out.append("        for (int t = 0; t < NTILE; t++) {")
        out.append("            const int base_m = t * MSPLIT;")
        out += _ind(tile, 12)
        out.append("        }")
    else:
        out += _ind(tile, 8)
    out += ["    }", ""]

    out += _epilogue(tiled)
    out += ["}", ""]
    return "\n".join(out) + "\n"


def _epilogue(tiled: bool) -> list[str]:
    """The simd_sum reduction and the store of one output element per lane."""
    if tiled:
        return [
            "    for (int r = 0; r < rows_per_simd; r++) {",
            "        for (int t = 0; t < NTILE; t++) {",
            "            for (int m = 0; m < MSPLIT; m++) {",
            "                const float reduced = simd_sum(acc[r][t][m]);",
            "                if (simd_lid == 0) {",
            "                    y[(first_m + t * MSPLIT + m) * out_vec_size +",
            "                      out_row + r] =",
            "                        static_cast<bfloat16_t>(reduced);",
            "                }",
            "            }",
            "        }",
            "    }",
        ]
    return [
        "    for (int r = 0; r < rows_per_simd; r++) {",
        "        for (int m = 0; m < NA; m++) {",
        "            const float reduced = simd_sum(acc[r][m]);",
        "            if (simd_lid == 0) {",
        "                y[(first_m + m) * out_vec_size + out_row + r] =",
        "                    static_cast<bfloat16_t>(reduced);",
        "            }",
        "        }",
        "    }",
    ]


def assert_faithful(header: str) -> None:
    """The generator with every flag off must reproduce the shipped template.

    Without this the census would measure a hand-retyped kernel rather than the
    one that ships, and every register number would be unattributable.
    """
    shipped = shipped_wide(header)
    if sha8(shipped) != SHIPPED_SHA8:
        raise SystemExit(
            "the shipped qwen_e120_qmv_wide template changed (sha8 %s, "
            "expected %s): re-derive research/e132_wide_source.py before "
            "censusing anything" % (sha8(shipped), SHIPPED_SHA8))
    generated = wide_source()
    if generated != shipped:
        for n, (a, b) in enumerate(zip(shipped.split("\n"),
                                       generated.split("\n"))):
            if a != b:
                raise SystemExit(
                    "generator diverges from the shipped template at line %d\n"
                    "  shipped:   %r\n  generated: %r" % (n + 1, a, b))
        raise SystemExit("generator length differs from the shipped template")


def patched_header(header: str, **flags) -> str:
    """The E120 header with `qwen_e120_qmv_wide` replaced by a variant."""
    assert_faithful(header)
    start, end = wide_span(header)
    return header[:start] + wide_source(**flags) + header[end:]


if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import e131_kernel_sources as ks

    text = ks.named_literal(ks.swift_text(ks.QWEN35, None),
                            "qwen35E120QMVHeader")
    assert_faithful(text)
    print("generator reproduces the shipped qwen_e120_qmv_wide byte for byte "
          "(sha8 %s)" % SHIPPED_SHA8)
    print(wide_source(msplit={8: 4}, late_meta=True))
