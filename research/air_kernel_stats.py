#!/usr/bin/env python3
"""Summarize per-kernel AIR statistics from `xcrun metal -S` textual output.

Private-memory `alloca`s that survive -O2 mean the compiler could not keep an
array in registers, which is the cheapest static signal that a wider row-packing
factor has fallen off the register-allocation cliff.
"""

from __future__ import annotations

import argparse
import pathlib
import re

ALLOCA = re.compile(r"alloca\s+(\[[^\]]*\]|<[^>]*>|[%\w.]+)")
# Two traps here, either of which reports fma=0 for a fully contracted build:
# `metal::fma` lowers to AIR's own `@air.fma.*` and never to `@llvm.fma.*`, and
# the crossrow inner loop is vectorized over NA, so the intrinsic is `v4f32`
# rather than `f32`. Match the whole family.
FMA = re.compile(r"@(llvm|air)\.fma\.(f32|v\d+f32)\b")
VECTOR_OP = re.compile(r"<\d+ x float>")
# Device-memory operands live in addrspace(1); `= load` without it is a private
# or threadgroup access. Counting the two separately is what distinguishes "the
# arithmetic was cut" from "the loads were cut", which is the whole DCE check.
DEVICE_LOAD = re.compile(r"=\s*load\s.*addrspace\(1\)")
ANY_LOAD = re.compile(r"=\s*load\s")

DEF = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*(.*)$")
SSA_REF = re.compile(r"%[\w.\-]+")
VEC_TYPE = re.compile(r"<(\d+) x (\w+)>")
SCALAR_TYPE = re.compile(r"\b(float|half|bfloat|double|i1|i8|i16|i32|i64|ptr)\b")
# A 32-bit lane register holds one float/i32/ptr-sized value; i64/double take
# two. Sub-word types still occupy a full register on AGX, so they cost 1.
SCALAR_WIDTH = {"double": 2, "i64": 2}


def value_width(rhs: str) -> int:
    """Registers a defined SSA value occupies, from the first type in its RHS."""
    vec = VEC_TYPE.search(rhs)
    if vec:
        return int(vec.group(1)) * SCALAR_WIDTH.get(vec.group(2), 1)
    scalar = SCALAR_TYPE.search(rhs)
    return SCALAR_WIDTH.get(scalar.group(1), 1) if scalar else 1


def peak_live_registers(body: list[str]) -> tuple[int, int]:
    """Lane-weighted peak live SSA values, as a register-pressure proxy.

    This is a textual linear scan, not a real CFG liveness: a value defined
    before a loop and used after it counts as live across the whole span. That
    over-counts, but the bias is identical across builds that differ only in a
    packing factor, so the NA-to-NA *shape* of the curve is the usable signal
    and the absolute number is not.
    """
    defs: dict[str, tuple[int, int]] = {}
    last_use: dict[str, int] = {}
    for i, line in enumerate(body):
        match = DEF.match(line)
        name, rhs = (match.group(1), match.group(2)) if match else (None, line)
        for ref in SSA_REF.findall(rhs):
            last_use[ref] = i
        if name and name not in defs:
            defs[name] = (i, value_width(rhs))

    peak = peak_count = 0
    for i in range(len(body)):
        live = [
            width
            for name, (start, width) in defs.items()
            if start <= i < last_use.get(name, start)
        ]
        if sum(live) > peak:
            peak, peak_count = sum(live), len(live)
    return peak, peak_count


def kernels(path: pathlib.Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith("define "):
            match = re.search(r"@([\w.]+)\(", line)
            name = match.group(1) if match else None
            if name:
                out[name] = []
        elif line == "}":
            name = None
        elif name is not None:
            out[name].append(line)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("air_ll")
    ap.add_argument("--match", default="")
    args = ap.parse_args()

    for name, body in kernels(pathlib.Path(args.air_ll)).items():
        if args.match and args.match not in name:
            continue
        allocas = [ALLOCA.search(line).group(1) for line in body if ALLOCA.search(line)]
        fma = sum(1 for line in body if FMA.search(line))
        fmul = sum(1 for line in body if re.search(r"=\s*fmul\s", line))
        fadd = sum(1 for line in body if re.search(r"=\s*fadd\s", line))
        # An AIR op on <NA x float> costs NA lane-issues on a scalar-lane GPU,
        # so the scalar/vector split is what makes op counts comparable.
        vec = sum(
            1
            for line in body
            if VECTOR_OP.search(line)
            and (FMA.search(line) or re.search(r"=\s*f(mul|add)\s", line))
        )
        dev_loads = sum(1 for line in body if DEVICE_LOAD.search(line))
        loads = sum(1 for line in body if ANY_LOAD.search(line))
        # Metal emits AIR with loops still rolled, so every count above is per
        # loop body. They are only comparable across builds at equal trip
        # counts, which this back-edge count is the machine check for.
        backedges = sum(1 for line in body if "!llvm.loop" in line)
        peak_regs, peak_vals = peak_live_registers(body)
        print(f"{name}: lines={len(body)} allocas={len(allocas)} fma_f32={fma} "
              f"fmul={fmul} fadd={fadd} float_ops={fma + fmul + fadd} "
              f"vector_float_ops={vec} flops={fma * 2 + fmul + fadd} "
              f"device_loads={dev_loads} loads={loads} loop_backedges={backedges} "
              f"peak_live_regs={peak_regs} peak_live_values={peak_vals} "
              f"types={sorted(set(allocas))}")


if __name__ == "__main__":
    main()
