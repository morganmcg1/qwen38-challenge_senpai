#!/usr/bin/env python3
"""Per-function AIR opcode census for the E107 affine-2 arms.

`research/air_kernel_stats.py` reports aggregate float and load counts. The
E107 question is narrower: how many operations does ONE packed 2-bit value
cost, and how many of them are 64-bit. So this tool counts the exact
instructions the extraction chain emits, per kernel, and normalises them to
the 32 values that one lane handles per k-block.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

DEFINE = re.compile(r"define .*?@([A-Za-z0-9_]+)\(")

PATTERNS = {
    "fmuladd": r"@llvm\.fmuladd\.f32",
    "cvt_u32_f32": r"@air\.convert\.f\.f32\.u\.i32",
    "cvt_u64_f32": r"@air\.convert\.f\.f32\.u\.i64",
    "extract_bits": r"@air\.extract_bits",
    "lshr_i64": r"= lshr[^\n]*i64",
    "lshr_i32": r"= lshr[^\n]*i32",
    "and_i64": r"= and i64",
    "and_i32": r"= and i32",
    "fmul_f32": r"= fmul float",
    "fadd_f32": r"= fadd float",
    "fadd_bf16": r"= fadd bfloat",
    "fpext": r"= fpext",
    "alloca": r"= alloca",
    "device_load": r"= load[^\n]*addrspace\(1\)",
}

COLUMNS = [
    "kernel",
    "lines",
    "fmuladd",
    "cvt_u32_f32",
    "cvt_u64_f32",
    "extract_bits",
    "lshr_i64",
    "lshr_i32",
    "and_i64",
    "and_i32",
    "fmul_f32",
    "fadd_f32",
    "fadd_bf16",
    "fpext",
    "alloca",
    "device_load",
]

# Instructions attributable to the value-extraction chain, i.e. the work that
# scales with the 32 packed values a lane handles per k-block.
EXTRACT_KEYS = (
    "cvt_u32_f32",
    "cvt_u64_f32",
    "extract_bits",
    "lshr_i64",
    "lshr_i32",
    "and_i64",
    "and_i32",
)


def split_functions(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    name = None
    body: list[str] = []
    for line in text.split("\n"):
        m = DEFINE.match(line)
        if m:
            name = m.group(1)
            body = []
        elif name is not None:
            if line.startswith("}"):
                out[name] = "\n".join(body)
                name = None
            else:
                body.append(line)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("air_ll")
    ap.add_argument("--json", default=None)
    ap.add_argument("--values-per-lane", type=int, default=32)
    args = ap.parse_args()

    text = pathlib.Path(args.air_ll).read_text()
    rows = []
    for name, body in sorted(split_functions(text).items()):
        row = {"kernel": name, "lines": len(body.split("\n"))}
        for key, pat in PATTERNS.items():
            row[key] = len(re.findall(pat, body))
        # i64 shifts and i64 conversions are the operations the ranked GPU has
        # to split into two 32-bit instructions.
        row["extract_ops"] = sum(row[k] for k in EXTRACT_KEYS)
        row["extract_ops_i64"] = row["lshr_i64"] + row["and_i64"] + row["cvt_u64_f32"]
        row["ops_per_value"] = round(
            (row["extract_ops"] + row["fmuladd"]) / args.values_per_lane, 3
        )
        rows.append(row)

    widths = {c: max(len(c), 12) for c in COLUMNS}
    print(" ".join(c.rjust(widths[c]) for c in COLUMNS))
    for row in rows:
        print(" ".join(str(row[c]).rjust(widths[c]) for c in COLUMNS))
    print()
    print(
        f"{'kernel'.rjust(14)} {'extract_ops'.rjust(12)} "
        f"{'of which i64'.rjust(13)} {'fmuladd'.rjust(8)} "
        f"{'ops/value'.rjust(10)}"
    )
    for row in rows:
        print(
            f"{row['kernel'].rjust(14)} {str(row['extract_ops']).rjust(12)} "
            f"{str(row['extract_ops_i64']).rjust(13)} "
            f"{str(row['fmuladd']).rjust(8)} "
            f"{str(row['ops_per_value']).rjust(10)}"
        )

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
