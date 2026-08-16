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
# Device-memory operands live in addrspace(1); `= load` without it is a private
# or threadgroup access. Counting the two separately is what distinguishes "the
# arithmetic was cut" from "the loads were cut", which is the whole DCE check.
DEVICE_LOAD = re.compile(r"=\s*load\s.*addrspace\(1\)")
ANY_LOAD = re.compile(r"=\s*load\s")


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
        fma = sum(line.count("call float @llvm.fma.f32") for line in body)
        fmul = sum(1 for line in body if re.search(r"=\s*fmul\s", line))
        fadd = sum(1 for line in body if re.search(r"=\s*fadd\s", line))
        dev_loads = sum(1 for line in body if DEVICE_LOAD.search(line))
        loads = sum(1 for line in body if ANY_LOAD.search(line))
        # Metal emits AIR with loops still rolled, so every count above is per
        # loop body. They are only comparable across builds at equal trip
        # counts, which this back-edge count is the machine check for.
        backedges = sum(1 for line in body if "!llvm.loop" in line)
        print(f"{name}: lines={len(body)} allocas={len(allocas)} fma_f32={fma} "
              f"fmul={fmul} fadd={fadd} flops={fma * 2 + fmul + fadd} "
              f"device_loads={dev_loads} loads={loads} loop_backedges={backedges} "
              f"types={sorted(set(allocas))}")


if __name__ == "__main__":
    main()
