#!/usr/bin/env python3
"""Assemble ONE self-contained source string holding every E77 sweep arm.

The scored worker does not compile the readable `kernels/*.metal`; for the
`quantized` family it concatenates the checked-in `mlx-generated/*.cpp`
preambles and hands the string to `newLibrary` with no include path.
`research/jit_string_compile.py` already reproduces that concatenation, so this
reuses it and appends one `[[kernel]]` entry point per sweep arm.

One source string for all arms means no arm can differ by preamble text or
compiler options. Each arm is still its own entry point, so each gets its own
register allocation, which is the variation the sweep needs.

Every arm calls the SHIPPED wrapper `qmv_fast_crossrow_affine4_g64_m`
unchanged, with `DIRECT_NIBBLES = true` exactly as the scored switch does.

  python3 research/e77_emit_sweep.py --out /tmp/e77/sweep.metal
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import PREAMBLES, preamble  # noqa: E402
from e77_probe import arms, entry  # noqa: E402


def assemble() -> str:
    parts = [preamble(stem, None) for stem in PREAMBLES]
    parts.extend(entry(spec) for spec in arms())
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = assemble()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    print(f"{args.out} {len(source)} bytes arms={len(arms())} "
          f"source_sha256={hashlib.sha256(source.encode()).hexdigest()}",
          file=sys.stderr)
    print(",".join(spec["arm"] for spec in arms()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
