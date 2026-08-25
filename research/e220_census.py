#!/usr/bin/env python3
"""E220 step 2 (RULE 400): integer-path census of the scored QMV dequant chain.

E220 asks whether the integer/bit side of the affine-4/group-64 dequant chain
in `qwen_e120_qmv_wide` can be restructured to fewer or cheaper instructions
with the floating-point instruction sequence literally unchanged.

Before any GPU second is spent this census answers two questions:

  1. Which template instantiations does a scored decode round actually reach?
     The width plan is `qwen35QMVVariant` over `Qwen35QMVKernelVariant.pairs`,
     both read live out of `Qwen35.swift`, so the enumeration cannot drift from
     the kernel the worker builds.

  2. What integer work does each instantiation already carry after -O2? The
     census compiles each one to textual AIR, isolates the k-loop body, and
     counts the integer opcodes, the address arithmetic and the integer to
     float conversions. If the sequence is already minimal, E220 stops here
     with a "not useful" result and the enumeration is the finding.

The census also records the ORDERED floating-point opcode sequence of the loop
body and its digest. That digest is the static half of the no-reassociation
constraint: any variant that changes it has changed the FP instruction
sequence and is dead before it reaches the 0-ULP gate.

AIR is architecture-independent, so the opcode counts are a proxy for, not a
measurement of, the AGX instruction stream. Register and spill numbers come
from the real AGX backend through `agx_crossarch`, for the local g16s
generation and for the ranked g17s generation.

Zero GPU seconds.

  python3 research/e220_census.py [--variant-header SYMBOL]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agx_crossarch  # noqa: E402
import e220_variants  # noqa: E402

REPO = HERE.parent
QWEN35 = REPO / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
ARTIFACTS = HERE / "e220-artifacts"

ARCHES = ("applegpu_g16s", "applegpu_g17s")

# The seven fused decode cells of `Qwen35QMVCell`, as `(k, n)`.
SCORED_CELLS = {
    "mlp.gate_up": (5120, 34816),
    "mlp.down": (17408, 5120),
    "gdn.in_proj": (5120, 16480),
    "out_proj": (6144, 5120),
    "fa.qkv": (5120, 14336),
    "lm_head": (5120, 248_320),
}

WIDTHS = tuple(range(2, 10))

PREAMBLE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

ENTRY = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    constant int& qmv_k [[buffer(6)]],
    constant int& qmv_n [[buffer(7)]],
    constant int& qmv_stride [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    qwen_e120_qmv_m<{m}, {ipg}, {table}>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride,
        int(qmv_tid.x), qmv_out_row, qmv_lid);
}}
"""

# AIR opcode families. `sitofp`/`uitofp` sit on the boundary: they are produced
# by the integer side and consumed by the float side, so they are counted apart
# from both.
INT_OPS = ("shl", "lshr", "ashr", "and", "or", "xor", "mul", "add", "sub",
           "zext", "sext", "trunc")
FP_OPS = ("fmul", "fadd", "fsub", "fdiv", "fneg", "fpext", "fptrunc",
          "insertelement", "extractelement", "shufflevector")
CONV_OPS = ("sitofp", "uitofp")

DEF = re.compile(r"^\s*(?:%[\w.\-]+\s*=\s*)?([a-z][\w.]*)\b(.*)$")
LABEL = re.compile(r"^([\w.\-]+):")
# The nibble-to-float conversion. The block that contains it is the dequant
# inner block, whatever the compiler numbered it.
NIBBLE_CONVERT = "air.convert.f.f32.s.i32"
# `alloca`s that survive -O2 are private-memory arrays the frontend could not
# keep in registers; each one costs a private load or store in the loop that
# reads it.
ALLOCA = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*alloca\s+(.*?),\s*align")


def live_text(symbol: str) -> str:
    """The exact Metal source string the worker compiles."""
    text = QWEN35.read_text()
    match = re.search(
        r'(?:private |internal |public )?let ' + symbol
        + r' = """\n(.*?)\n    """\n', text, re.S)
    if match is None:
        raise SystemExit(f"could not find {symbol} in Qwen35.swift")
    return match.group(1)


def live_pairs() -> dict[str, dict[int, int]]:
    """`Qwen35QMVKernelVariant.pairs`, read live out of the Swift source."""
    text = QWEN35.read_text()
    out: dict[str, dict[int, int]] = {}
    for case in ("staged", "singlePass"):
        match = re.search(
            r"case \." + case + r":\s*\n\s*return \[([^\]]*)\]", text)
        if match is None:
            raise SystemExit(f"could not read the {case} pair table")
        out["staged" if case == "staged" else "singlepass"] = {
            int(m): int(ipg)
            for m, ipg in re.findall(r"\((\d+),\s*(\d+)\)", match.group(1))
        }
    return out


def live_plan_rule() -> str:
    """The body of `qwen35QMVVariant`, so the report quotes the shipped rule."""
    text = QWEN35.read_text()
    match = re.search(
        r"func qwen35QMVVariant\([^)]*\) -> Qwen35QMVKernelVariant \{\n(.*?)\n\}",
        text, re.S)
    if match is None:
        raise SystemExit("could not read qwen35QMVVariant")
    return " ".join(match.group(1).split())


def variant_for(m: int, cell: str) -> str:
    """`qwen35QMVVariant` transcribed; asserted against the live source."""
    return "singlepass" if m == 6 and cell != "mlp.down" else "staged"


def check_plan_rule(rule: str) -> None:
    if "m == 6" not in rule or "mlpDown" not in rule:
        raise SystemExit(
            f"qwen35QMVVariant no longer matches the transcribed rule: {rule}")


def instantiations(pairs: dict[str, dict[int, int]]) -> dict[tuple, dict]:
    """Every `(m, ipg)` template instance a scored decode round can reach.

    A round runs all seven cells at one width, so the reachable set is the
    union over cells and widths of the `(variant, m, ipg)` the plan selects.
    `NA` is the template's own width argument: `IPG` for a full group and the
    `TAIL` branch value for a short last group.
    """
    found: dict[tuple, dict] = {}
    for m in WIDTHS:
        for cell in SCORED_CELLS:
            variant = variant_for(m, cell)
            ipg = pairs[variant][m]
            key = (m, ipg)
            entry = found.setdefault(key, {
                "m": m, "ipg": ipg, "variant": variant, "cells": [],
                "groups": -(-m // ipg),
                "na_full": ipg,
                "na_tail": (m % ipg) if (m % ipg) else None,
            })
            entry["cells"].append(cell)
    for entry in found.values():
        tail = entry["na_tail"]
        entry["na_instantiated"] = sorted(
            {entry["na_full"]} | ({max(tail, 2)} if tail else set()))
    return found


def functions(text: str) -> dict[str, list[str]]:
    """Every AIR function body, keyed by symbol name."""
    out: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        define = re.match(r"^define\s+[^@]*@([\w.$]+)\(", line)
        if define:
            name = define.group(1)
            out[name] = []
            continue
        if line == "}":
            name = None
            continue
        if name is not None and line.strip():
            out[name].append(line)
    return out


def split_blocks(body: list[str]) -> dict[str, list[str]]:
    """Basic blocks of one function body, in source order."""
    out: dict[str, list[str]] = {"entry": []}
    name = "entry"
    for line in body:
        label = LABEL.match(line)
        if label:
            name = label.group(1)
            out[name] = []
            continue
        out[name].append(line)
    return out


def opcodes(lines: list[str]) -> list[str]:
    found = []
    for line in lines:
        match = DEF.match(line)
        if match:
            found.append(match.group(1))
    return found


def op_stats(lines: list[str]) -> dict:
    ops = opcodes(lines)
    counts = collections.Counter(ops)
    fp_sequence = [op for op in ops if op in FP_OPS] + [
        "fmuladd" for line in lines
        if re.search(r"@(llvm|air)\.(fma|fmuladd)\.", line)]
    return {
        "instructions": len(lines),
        "int_ops": {op: counts[op] for op in INT_OPS if counts[op]},
        "int_ops_total": sum(counts[op] for op in INT_OPS),
        "conv_int_to_float": sum(
            1 for line in lines if NIBBLE_CONVERT in line)
        + sum(counts[op] for op in CONV_OPS),
        "fp_ops": {op: counts[op] for op in FP_OPS if counts[op]},
        "fp_ops_total": sum(counts[op] for op in FP_OPS),
        "fp_fma": sum(1 for line in lines
                      if re.search(r"@(llvm|air)\.(fma|fmuladd)\.", line)),
        "getelementptr": sum(
            1 for line in lines if re.search(r"=\s*getelementptr", line)),
        "device_load": sum(
            1 for line in lines
            if re.search(r"=\s*load\s.*addrspace\(1\)", line)),
        "device_load_vector": sum(
            1 for line in lines
            if re.search(r"=\s*load\s+<\d+ x \w+>.*addrspace\(1\)", line)),
        "private_load": sum(
            1 for line in lines
            if re.search(r"=\s*load\s", line)
            and "addrspace(" not in line.split("!")[0]),
        "private_store": sum(
            1 for line in lines
            if re.match(r"\s*store\s", line)
            and "addrspace(" not in line.split("!")[0]),
        "fp_sequence_sha8": hashlib.sha256(
            "\n".join(fp_sequence).encode()).hexdigest()[:8],
        "fp_sequence_length": len(fp_sequence),
    }


def normalized(lines: list[str]) -> list[str]:
    """Instruction text with SSA names erased, so two builds compare directly."""
    out = []
    for line in lines:
        text = re.sub(r"%[\w.\-]+", "%", line.split("!")[0]).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            out.append(text)
    return out


def census_template(text: str, symbol: str) -> dict:
    """Per-instantiation census of one `qwen_e120_qmv_wide` AIR body.

    The frontend leaves the `k`, `r`, `i` and `m` loops rolled, so the census
    reports the whole body and, separately, the DEQUANT BLOCK: the basic block
    that holds the nibble-to-float conversions. That block is the per-4-value
    integer sequence E220 attacks, and its normalized text is the artifact a
    variant must beat.
    """
    body = functions(text)[symbol]
    blocks = split_blocks(body)
    dequant = [name for name, lines in blocks.items()
               if any(NIBBLE_CONVERT in line for line in lines)]
    if len(dequant) != 1:
        raise SystemExit(
            f"{symbol}: {len(dequant)} blocks hold {NIBBLE_CONVERT}")
    lines = blocks[dequant[0]]
    allocas = [(m.group(1), m.group(2))
               for line in body if (m := ALLOCA.match(line))]
    return {
        "whole_body": op_stats(body),
        "dequant_block": op_stats(lines),
        "dequant_block_sequence": normalized(lines),
        "surviving_allocas": [t for _, t in allocas],
        "block_count": len(blocks),
    }


def emit_air(source: str, workdir: pathlib.Path) -> str:
    src = workdir / "census.metal"
    src.write_text(source)
    out = workdir / "census.ll"
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", "-emit-llvm", str(src), "-o", str(out)],
        check=True, capture_output=True)
    return out.read_text()


def name_for(m: int, ipg: int, table: bool) -> str:
    return f"qmv_m{m}_ipg{ipg}_{'tbl' if table else 'rec'}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-header", default="qwen35E120QMVHeader",
                        help="Metal source symbol to census")
    parser.add_argument("--variant", default="base",
                        help="integer-path transform from e220_variants")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pairs = live_pairs()
    rule = live_plan_rule()
    check_plan_rule(rule)
    reachable = instantiations(pairs)
    header = e220_variants.apply(live_text(args.variant_header), args.variant)

    arms = [(m, ipg, table)
            for (m, ipg) in sorted(reachable)
            for table in (True, False)]
    source = PREAMBLE + header + "\n" + "\n".join(
        ENTRY.format(name=name_for(m, ipg, table), m=m, ipg=ipg,
                     table="true" if table else "false")
        for m, ipg, table in arms)

    result = {
        "experiment": "e220-dequant-integer-path",
        "step": "2-rule-400-census",
        "harness": "local",
        "measurement": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "variant": args.variant,
        "source_symbol": args.variant_header,
        "source_sha8": hashlib.sha256(header.encode()).hexdigest()[:8],
        "width_plan_rule": rule,
        "pair_table": pairs,
        "scored_cells": SCORED_CELLS,
        "reachable_instantiations": {
            f"m{m}_ipg{ipg}": reachable[(m, ipg)] for m, ipg in sorted(reachable)
        },
        "arches": list(ARCHES),
        "cells": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        air = emit_air(source, workdir)
        lib = agx_crossarch.build_metallib(source, workdir)
        records: dict[str, dict] = {}
        for arch in ARCHES:
            for name, rec in agx_crossarch.translate(lib, arch, workdir).items():
                records.setdefault(name, {})[arch.replace("applegpu_", "")] = rec

    # The AGX numbers are per entry point, where the template is fully inlined.
    # `text_bytes` is real backend machine-code size, so it is the one static
    # figure that sees the unrolling, LICM and load merging the AIR does not.
    for m, ipg, table in arms:
        name = name_for(m, ipg, table)
        entry = {"m": m, "ipg": ipg, "table": table,
                 "variant": reachable[(m, ipg)]["variant"],
                 "groups": reachable[(m, ipg)]["groups"]}
        for arch, rec in records.get(name, {}).items():
            entry[arch] = {"registers": rec.get("registers"),
                           "spill_bytes": rec.get("spill_bytes"),
                           "text_bytes": rec.get("text_bytes"),
                           "text_sha8": rec.get("text_sha8")}
        result["cells"][name] = entry

    # The AIR census is per template instantiation, keyed by `(NA, USE_TABLE)`.
    result["instantiations"] = {}
    for symbol in functions(air):
        match = re.match(
            r"_Z18qwen_e120_qmv_wideILi(\d+)ELb([01])E", symbol)
        if not match:
            continue
        na, use_table = int(match.group(1)), match.group(2) == "1"
        result["instantiations"][f"NA{na}_{'tbl' if use_table else 'rec'}"] = {
            "na": na, "use_table": use_table, "symbol": symbol,
            **census_template(air, symbol),
        }

    out = (pathlib.Path(args.out) if args.out
           else ARTIFACTS / f"e220_census_{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"variant: {args.variant}  source_sha8: {result['source_sha8']}")
    print(f"width plan: {rule}")
    print("reachable (m, ipg): "
          f"{sorted((m, ipg) for m, ipg in reachable)}")
    print("\nAGX backend, per entry point (text_bytes is real machine code):")
    print(f"  {'cell':22s} {'g16 reg/sp':>11s} {'g16 text':>9s} {'g16 sha':>8s}"
          f" {'g17 reg/sp':>11s} {'g17 text':>9s} {'g17 sha':>8s}")
    for m, ipg, table in arms:
        e = result["cells"][name_for(m, ipg, table)]
        g16, g17 = e.get("g16s", {}), e.get("g17s", {})
        print(
            f"  {name_for(m, ipg, table):22s}"
            f" {str(g16.get('registers')) + '/' + str(g16.get('spill_bytes')):>11s}"
            f" {g16.get('text_bytes', -1):9d} {str(g16.get('text_sha8')):>8s}"
            f" {str(g17.get('registers')) + '/' + str(g17.get('spill_bytes')):>11s}"
            f" {g17.get('text_bytes', -1):9d} {str(g17.get('text_sha8')):>8s}")
    print("\nAIR, per template instantiation:")
    print(f"  {'inst':12s} {'blkI':>5s} {'int':>4s} {'conv':>5s} {'fp':>4s}"
          f" {'fma':>4s} {'gep':>4s} {'dld':>4s} {'pld':>4s} {'pst':>4s}"
          f" {'fpsha':>8s}")
    for key, inst in result["instantiations"].items():
        d = inst["dequant_block"]
        print(f"  {key:12s} {d['instructions']:5d} {d['int_ops_total']:4d}"
              f" {d['conv_int_to_float']:5d} {d['fp_ops_total']:4d}"
              f" {d['fp_fma']:4d} {d['getelementptr']:4d} {d['device_load']:4d}"
              f" {d['private_load']:4d} {d['private_store']:4d}"
              f" {d['fp_sequence_sha8']:>8s}")
    print("\nwhole template body (all loops rolled at AIR level):")
    for key, inst in result["instantiations"].items():
        b = inst["whole_body"]
        print(f"  {key:12s} instr={b['instructions']:4d} int={b['int_ops']}"
              f" mul={b['int_ops'].get('mul', 0)}"
              f" dld={b['device_load']} dldvec={b['device_load_vector']}"
              f" pld={b['private_load']} pst={b['private_store']}"
              f" allocas={inst['surviving_allocas']}")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
