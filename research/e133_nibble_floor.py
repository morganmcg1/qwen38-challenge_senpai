#!/usr/bin/env python3
"""E133: the nibble-dequantisation floor of the Route B QMV cell.

F157 prices the cell at `statements per output element = 38 / IPG + 25 / RPS`.
Of the 38 weight-side statements, 28 are nibble operations: three shifts and
four masks for each of the four `uint16_t` halfwords a lane holds. At
`(IPG=6, RPS=4)` those 28 are 37 % of the whole cell, so a free dequantisation
would move the cell from 12.583 to 7.917 statements, a 37 % cut. That is the
bound on the entire remaining QMV axis, and it is only a bound: nobody has
measured what the smallest bit-exact extraction actually costs.

This module measures it. It reads `qwen_e120_qmv_wide` out of `Qwen35.swift`
READ-ONLY, substitutes ONE expression in memory, and censuses the result. It
writes nothing to a tracked source and it never touches the GPU.

RULE 92. Every variant produces `n0..n3` as the exact float value of the
nibble, and every variant then sums `a0*n0 + a1*n1 + a2*n2 + a3*n3` in that
order into the same `partial[r]`. No floating-point operation and no
accumulation order moves. `exactness` proves the value claim for all sixteen
nibble values at all four positions by simulating each idiom's bit
manipulation in Python.

RULE 89. Simdgroup residency is a MODEL OUTPUT computed from the measured
register count, so it is labelled `derived`. Registers, spill bytes and ISA
text sizes are measurements read out of the translated binary.

Three numbers are reported per variant:

  air_statements    AIR statements in the hot region, by class. This is the
                    F151 currency and needs no calibration.
  registers/spill   `metal-tt` for `applegpu_g17s`, the ranked generation.
  text_bytes        translated ISA size. AGX is variable length, so bytes are
                    NOT instructions; `research/e132-artifacts/
                    instruction-channel.json` measured 4.4 bytes per float ALU
                    instruction and 12.2 per load on this generation, and a
                    byte delta is reported only inside that bracket.

  python3 research/e133_nibble_floor.py exactness
  python3 research/e133_nibble_floor.py sources --variant bfe
  python3 research/e133_nibble_floor.py census --out research/e133-artifacts/rung0-nibble-floor.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402
import e120_g17s_census as e120  # noqa: E402
import e131_kernel_sources as ks  # noqa: E402
import e132_instruction_probe as probe  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
QWEN35 = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
INCLUDE = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx"
RANKED = agx.RANKED_ARCH
ARCHES = (agx.LOCAL_ARCH, RANKED)
SIMDGROUP_BUDGET = e120.SIMDGROUP_BUDGET

# The cell the ranked width mix actually pays for. F157 puts the balance-law
# optimum at IPG 6.08 and the g17s register ceiling at IPG 7, so `{6:6, 7:7}`
# is both. RPS is 4 everywhere in the shipped table.
CELLS = ((6, 4), (7, 4), (8, 4), (4, 4))

# `research/e132-artifacts/instruction-channel.json`, verdict field -1 on
# applegpu_g17s. No channel is an instruction counter, so a text-byte delta is
# only ever converted into an instruction delta as a bracket.
BYTES_PER_INSTRUCTION = (4.375, 12.1741)

# F157 as published, for the report. `nibble_int_ops` is the source-level count
# of three shifts and four masks for each of four halfwords.
F157_ROW_KEYED = {
    "weight_element_loads": 4,
    "weight_address_arith": 1,
    "metadata_loads": 2,
    "metadata_index_arith": 1,
    "metadata_widenings": 2,
    "nibble_int_ops": 28,
}
F157_M_KEYED = {
    "sums_table_loads": 1,
    "activation_vec4_loads": 4,
    "activation_widenings": 16,
    "activation_address_arith": 4,
}


class Refused(SystemExit):
    pass


# --------------------------------------------------------------------------
# the shipped expression, read rather than retyped
# --------------------------------------------------------------------------

PRODUCT = re.compile(
    r"^(?P<pad>[ ]*)partial\[r\] \+= \(a0 \* .*?\);$",
    re.M | re.S)


def shipped_header() -> str:
    swift = ks.swift_text(QWEN35, None)
    return ks.named_literal(swift, "qwen35E120QMVHeader")


def product_span(header: str) -> re.Match:
    hit = PRODUCT.search(header)
    if hit is None:
        raise Refused("no `partial[r] += (a0 * ...)` statement in the header")
    if PRODUCT.search(header, hit.end()) is not None:
        raise Refused("more than one product statement; the patch is ambiguous")
    return hit


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------
#
# Each entry is (statements, exactness argument, python simulator).
# `statements` defines `n0..n3` as floats from `packed[r][i]`. The product line
# is appended by `patched_header` and is identical for every variant.

SUM = "partial[r] += (a0 * n0 + a1 * n1 + a2 * n2 + a3 * n3);"

VARIANTS: dict[str, dict] = {}


def variant(name: str, note: str, sim, body: str) -> None:
    VARIANTS[name] = {"note": note, "sim": sim, "body": body.strip("\n")}


def _sim_mask(p: int) -> list[int]:
    return [(p >> (4 * j)) & 0xF for j in range(4)]


def _sim_magic_f32(p: int) -> list[float]:
    out = []
    for j in range(4):
        bits = 0x4B000000 | ((p >> (4 * j)) & 0xF)
        out.append(struct.unpack("<f", struct.pack("<I", bits))[0] - 8388608.0)
    return out


def _sim_magic_bf16_pair(p: int) -> list[float]:
    """Two nibbles per 32-bit word through the bfloat exponent trick."""
    out = [0.0] * 4
    for half in range(2):
        # nibbles 2*half and 2*half+1 lifted into the two bfloat lanes
        src = (p >> (8 * half)) & 0xFF
        word = (src & 0xF) | ((src & 0xF0) << 12) | 0x43004300
        for lane in range(2):
            bf = (word >> (16 * lane)) & 0xFFFF
            value = struct.unpack("<f", struct.pack("<I", bf << 16))[0] - 128.0
            out[2 * half + lane] = value
    return out


variant(
    "shipped_lifted",
    "the shipped idiom, written as four named floats. This is the "
    "harness-neutrality control: it must census exactly like the untouched "
    "header, otherwise the substitution itself costs something and every "
    "other row is confounded.",
    _sim_mask,
    """
const uint16_t p = packed[r][i];
const float n0 = float(p & 0x000f);
const float n1 = float((p >> 4) & 0x000f);
const float n2 = float((p >> 8) & 0x000f);
const float n3 = float((p >> 12) & 0x000f);
""")

variant(
    "bfe",
    "`extract_bits` is the Metal spelling of a bitfield extract. AGX has a "
    "bitfield-extract instruction, so if the compiler was not already folding "
    "shift-then-mask into it, this collapses two statements into one for each "
    "nibble.",
    _sim_mask,
    """
const uint p = uint(packed[r][i]);
const float n0 = float(extract_bits(p, 0, 4));
const float n1 = float(extract_bits(p, 4, 4));
const float n2 = float(extract_bits(p, 8, 4));
const float n3 = float(extract_bits(p, 12, 4));
""")

variant(
    "rolling",
    "One shift feeds the next, so the shift amounts are all 4 and the top "
    "nibble needs no mask. Fewer distinct constants, one fewer mask, at the "
    "price of a serial dependence the shipped form does not have.",
    _sim_mask,
    """
uint p = uint(packed[r][i]);
const float n0 = float(p & 0xfu);
p >>= 4;
const float n1 = float(p & 0xfu);
p >>= 4;
const float n2 = float(p & 0xfu);
const float n3 = float(p >> 4);
""")

variant(
    "bytes",
    "`as_type<uchar2>` splits the halfword into two bytes for free if the "
    "backend has a sub-register addressing mode. Each byte then needs one "
    "mask and one shift instead of two masks and two shifts.",
    _sim_mask,
    """
const uchar2 b = as_type<uchar2>(packed[r][i]);
const float n0 = float(b[0] & 0x0f);
const float n1 = float(b[0] >> 4);
const float n2 = float(b[1] & 0x0f);
const float n3 = float(b[1] >> 4);
""")

variant(
    "magic_f32",
    "Build the float bit pattern instead of converting. `0x4b000000 | v` is "
    "exactly `2^23 + v` for v in 0..15, and subtracting `2^23` is exact "
    "because both operands share a binade and the result is representable. "
    "Trades one integer-to-float conversion for one OR and one subtract, so "
    "it wins only if conversion is more expensive than two ALU statements.",
    _sim_magic_f32,
    """
const uint p = uint(packed[r][i]);
const float n0 = as_type<float>(0x4b000000u | (p & 0xfu)) - 8388608.0f;
const float n1 = as_type<float>(0x4b000000u | ((p >> 4) & 0xfu)) - 8388608.0f;
const float n2 = as_type<float>(0x4b000000u | ((p >> 8) & 0xfu)) - 8388608.0f;
const float n3 = as_type<float>(0x4b000000u | ((p >> 12) & 0xfu)) - 8388608.0f;
""")

variant(
    "magic_bf16_pair",
    "The same trick two nibbles at a time. A bfloat carries seven mantissa "
    "bits, so 128..143 are exact and `0x4300 | v` is exactly `128 + v`. Two "
    "bfloat lanes fit in one 32-bit register, so one AND, one OR and one "
    "vector subtract serve two nibbles. The bfloat-to-float widening is a "
    "shift on this hardware. This is the form the advisor priced near 24 "
    "statements for eight nibbles; here it is measured rather than estimated.",
    _sim_magic_bf16_pair,
    """
const uint p = uint(packed[r][i]);
const uint lo = (p & 0xfu) | ((p & 0xf0u) << 12) | 0x43004300u;
const uint hi = ((p >> 8) & 0xfu) | ((p >> 8 & 0xf0u) << 12) | 0x43004300u;
const bfloat2 bl = as_type<bfloat2>(lo) - bfloat2(128.0bf);
const bfloat2 bh = as_type<bfloat2>(hi) - bfloat2(128.0bf);
const float n0 = float(bl[0]);
const float n1 = float(bl[1]);
const float n2 = float(bh[0]);
const float n3 = float(bh[1]);
""")

variant(
    "magic_bf16_pair_novec",
    "`magic_bf16_pair` without the vector subtract, in case the backend "
    "scalarises `bfloat2` arithmetic and the vector spelling costs a pack.",
    _sim_magic_bf16_pair,
    """
const uint p = uint(packed[r][i]);
const uint lo = (p & 0xfu) | ((p & 0xf0u) << 12) | 0x43004300u;
const uint hi = ((p >> 8) & 0xfu) | ((p >> 8 & 0xf0u) << 12) | 0x43004300u;
const float n0 = float(as_type<bfloat2>(lo)[0]) - 128.0f;
const float n1 = float(as_type<bfloat2>(lo)[1]) - 128.0f;
const float n2 = float(as_type<bfloat2>(hi)[0]) - 128.0f;
const float n3 = float(as_type<bfloat2>(hi)[1]) - 128.0f;
""")

# The floor of the harness. Not bit-exact and never shippable: it deletes the
# extraction and keeps the load, so `shipped_lifted` minus `no_extract` is the
# cost of extraction and nothing else.
variant(
    "no_extract",
    "NOT BIT-EXACT AND NOT A CANDIDATE. The weight load stays live and every "
    "nibble becomes the whole halfword, so this row is the harness floor: the "
    "difference between any real variant and this one is the extraction cost "
    "with the load, the address arithmetic and the four products held fixed.",
    None,
    """
const float p = float(packed[r][i]);
const float n0 = p;
const float n1 = p;
const float n2 = p;
const float n3 = p;
""")


def patched_header(header: str, name: str) -> str:
    if name == "shipped":
        return header
    spec = VARIANTS[name]
    hit = product_span(header)
    pad = hit.group("pad")
    lines = [pad + "{"]
    for line in spec["body"].splitlines():
        lines.append(pad + "    " + line)
    lines.append(pad + "    " + SUM)
    lines.append(pad + "}")
    return header[:hit.start()] + "\n".join(lines) + header[hit.end():]


# --------------------------------------------------------------------------
# exactness, without a GPU
# --------------------------------------------------------------------------

def exactness() -> int:
    """Every idiom, every nibble value, every position, against `float(v)`."""
    ok = True
    print("bit-exactness of each idiom, all 16 values at all 4 positions")
    for name, spec in VARIANTS.items():
        sim = spec["sim"]
        if sim is None:
            print("  %-22s SKIPPED, declared not exact" % name)
            continue
        worst = 0.0
        for value in range(16):
            for position in range(4):
                packed = value << (4 * position)
                got = sim(packed)[position]
                worst = max(worst, abs(float(got) - float(value)))
        status = "EXACT" if worst == 0.0 else "NOT EXACT, max abs %g" % worst
        ok = ok and worst == 0.0
        print("  %-22s %s" % (name, status))
    # A positive control: the comparison must be able to fail.
    broken = [v + 0.5 for v in range(16)]
    print("  %-22s %s" % ("positive control",
                          "detected" if any(abs(broken[v] - v) > 0
                                            for v in range(16))
                          else "NOT DETECTED, the check is inert"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------

def wrapper(name: str, na: int, rps: int) -> str:
    """One entry point holding exactly `qwen_e120_qmv_wide<NA, RPS, true>`.

    `qwen_e120_qmv_m<NA, NA, RPS, true>` has `TAIL == 0`, so it inlines that
    single body and nothing else.
    """
    body = """    const int qmv_m = x_shape[x_ndim - 2];
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = qmv_m <= 8 ? 8 : 16;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * %d + int(qmv_sgid) * %d;
    const int qmv_gx = int(qmv_tid.x);
    qwen_e120_qmv_m<%d, %d, %d, true>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride,
        qmv_gx, qmv_out_row, qmv_lid);""" % (
        2 * rps, rps, na, na, rps)
    inputs = [("w", "uint32_t"), ("scales", "bfloat16_t"),
              ("biases", "bfloat16_t"), ("x", "bfloat16_t"),
              ("xsums", "float")]
    return e120.generate(name, inputs, [("y", "bfloat16_t")], body)


def library(header: str, cells) -> tuple[str, list[str]]:
    parts, names = [], []
    for na, rps in cells:
        name = "cell_na%d_rps%d" % (na, rps)
        parts.append(wrapper(name, na, rps))
        names.append(name)
    return "\n".join([e120.PRELUDE, header, ""] + parts) + "\n", names


CONVERTS = ("uitofp", "sitofp", "fptoui", "fptosi")


def reclassify(lines: list[str]) -> dict:
    """`census_lines` with the conversions split out of `other`.

    F157 counts twenty-eight nibble statements and no conversions. Whether the
    sixteen integer-to-float conversions a lane needs are real statements or
    are folded into the multiply is exactly what the weight-side constant
    depends on, so they are never left inside a residual class here.
    """
    counts = probe.census_lines(lines)
    counts["convert"] = 0
    counts["bitcast"] = 0
    for line in lines:
        head = line.split("=")[-1].strip().split(" ")[0] if "=" in line \
            else line.split(" ")[0]
        if head in CONVERTS:
            counts["convert"] += 1
            counts["other"] -= 1
        elif head == "bitcast":
            counts["bitcast"] += 1
    return counts


def hot_region(text: str, kernel: str) -> tuple[str, dict]:
    """The outermost loop of `kernel`, or the whole body when it is unrolled."""
    functions = probe.air_functions(text)
    match = [n for n in functions if n == kernel or n.endswith(kernel)]
    if not match:
        raise Refused("no AIR function named %s" % kernel)
    blocks = functions[match[0]]
    loops = probe.natural_loops(blocks)
    if loops:
        outer = loops[0]
        lines = [line for name in outer["blocks"]
                 for label, body in blocks if label == name for line in body]
        return "loop %s" % outer["header"], reclassify(lines)
    return "whole body (no loop survived unrolling)", reclassify(
        [line for _, body in blocks for line in body])


def per_output_element(counts: dict, na: int, rps: int) -> float:
    """F157 currency. Statements attributable to one output element.

    The hot region computes `NA * RPS` output elements' worth of one k block,
    so the whole census divides by `NA * RPS`. F157's split into `38 / IPG` and
    `25 / RPS` is the same number written by key; this returns the total so
    variants are comparable without assuming the split still holds.
    """
    return counts["machine"] / float(na * rps)


def census(header: str, cells, workdir: pathlib.Path, tag: str,
           translate: bool) -> dict:
    source, names = library(header, cells)
    air = probe.emit_air(source, workdir / tag)
    rows: dict[str, dict] = {}
    for (na, rps), name in zip(cells, names):
        where, counts = hot_region(air, name)
        rows[name] = {
            "na": na, "rps": rps, "region": where, "air": counts,
            "air_per_output_element": round(
                per_output_element(counts, na, rps), 4),
        }
    if translate:
        lib = agx.build_metallib(source, workdir / tag, include=INCLUDE)
        wanted = set(names)
        for arch in ARCHES:
            found = agx.translate(lib, arch, workdir / tag,
                                  select=lambda n: n in wanted)
            for name in names:
                record = found[name]
                registers = record["registers"]
                rows[name][arch] = {
                    "registers": registers,
                    "spill_bytes": record["spill_bytes"],
                    "text_bytes": record["text_bytes"],
                    "simdgroups_derived": SIMDGROUP_BUDGET[arch] // registers,
                }
    return rows


def toolchain() -> str:
    done = subprocess.run(["xcrun", "metal", "--version"],
                          capture_output=True, text=True, check=True)
    return done.stdout.strip().splitlines()[0]


def base_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def run_census(out: pathlib.Path | None, cells, translate: bool) -> int:
    header = shipped_header()
    result = {
        "schema_version": 1,
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "harness": "compile_only",
        "occupancy_label": "derived",
        "occupancy_rule": "Rule 89",
        "exactness_rule": "Rule 92",
        "tool": "research/e133_nibble_floor.py census",
        "base_sha": base_sha(),
        "toolchain": toolchain(),
        "ranked_arch": RANKED,
        "bytes_per_instruction_bracket": BYTES_PER_INSTRUCTION,
        "f157_row_keyed": F157_ROW_KEYED,
        "f157_m_keyed": F157_M_KEYED,
        "cells": [list(c) for c in cells],
        "variants": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        for name in ["shipped"] + list(VARIANTS):
            patched = patched_header(header, name)
            note = "the untouched header" if name == "shipped" \
                else VARIANTS[name]["note"]
            try:
                rows = census(patched, cells, work, name, translate)
            except subprocess.CalledProcessError as error:
                result["variants"][name] = {
                    "note": note,
                    "compiled": False,
                    "error": (error.stderr or b"").decode(
                        "utf-8", "replace")[-2000:],
                }
                continue
            result["variants"][name] = {
                "note": note, "compiled": True, "cells": rows}
    report(result)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
        print("\nwrote %s" % out)
    return 0


def report(result: dict) -> None:
    cells = [tuple(c) for c in result["cells"]]
    print("base %s" % result["base_sha"][:8])
    print("%s   harness=compile_only, no GPU, no timing"
          % result["toolchain"])
    print()
    header_row = "%-24s" % "variant"
    for na, rps in cells:
        header_row += "  NA=%d RPS=%d" % (na, rps)
    print(header_row + "     (AIR machine statements in the hot region)")
    base = {}
    for name, entry in result["variants"].items():
        if not entry["compiled"]:
            print("%-24s  DID NOT COMPILE" % name)
            continue
        line = "%-24s" % name
        for na, rps in cells:
            row = entry["cells"]["cell_na%d_rps%d" % (na, rps)]
            line += "  %10d" % row["air"]["machine"]
            base.setdefault((na, rps), {})[name] = row
        print(line)
    print()
    print("per output element, F157 currency")
    for na, rps in cells:
        rows = base.get((na, rps), {})
        if "shipped" not in rows:
            continue
        reference = rows["shipped"]["air_per_output_element"]
        print("  NA=%d RPS=%d   shipped %.4f" % (na, rps, reference))
        for name, row in rows.items():
            if name == "shipped":
                continue
            value = row["air_per_output_element"]
            print("      %-22s %8.4f   %+7.2f %%"
                  % (name, value, 100.0 * (value - reference) / reference))
    print()
    print("applegpu_g17s, translated")
    for na, rps in cells:
        rows = base.get((na, rps), {})
        if not rows or RANKED not in next(iter(rows.values())):
            continue
        print("  NA=%d RPS=%d" % (na, rps))
        for name, row in rows.items():
            cell = row[RANKED]
            print("      %-22s registers %3d  spill %3d  text %6d  "
                  "simdgroups(derived) %2d"
                  % (name, cell["registers"], cell["spill_bytes"],
                     cell["text_bytes"], cell["simdgroups_derived"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("exactness")
    show = sub.add_parser("sources")
    show.add_argument("--variant", required=True)
    run = sub.add_parser("census")
    run.add_argument("--out", type=pathlib.Path)
    run.add_argument("--air-only", action="store_true",
                     help="skip metal-tt; AIR statements only")
    run.add_argument("--cell", action="append",
                     help="NA,RPS; repeatable, defaults to the ranked set")
    args = parser.parse_args()
    if args.command == "exactness":
        return exactness()
    if args.command == "sources":
        print(patched_header(shipped_header(), args.variant))
        return 0
    cells = CELLS
    if args.cell:
        cells = tuple(tuple(int(v) for v in c.split(",")) for c in args.cell)
    return run_census(args.out, cells, not args.air_only)


if __name__ == "__main__":
    raise SystemExit(main())
