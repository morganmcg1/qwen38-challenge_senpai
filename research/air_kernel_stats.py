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
        print(f"{name}: lines={len(body)} allocas={len(allocas)} fma_f32={fma} "
              f"types={sorted(set(allocas))}")


if __name__ == "__main__":
    main()
