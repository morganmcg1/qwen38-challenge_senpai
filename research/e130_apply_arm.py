#!/usr/bin/env python3
"""E130: apply one censused arm to the real kernel source and its generated twin.

`research/e130_census.py` builds every arm as a text transform of the emitted
JIT source. The same transforms apply to the two files that ship the kernel,
because the generated twin is a verbatim string copy of the readable header:

  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp

Both must move together or `python3 research/twin_audit.py` fails, because the
`quantized` family is JIT-compiled from the twin at runtime.

    python3 research/e130_apply_arm.py prune_na5_pair
    python3 research/e130_apply_arm.py --check prune_na5_pair
"""

from __future__ import annotations

import argparse
import sys

from e130_census import VARIANTS

HDR = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(VARIANTS))
    ap.add_argument("--check", action="store_true",
                    help="report the byte delta without writing")
    args = ap.parse_args()

    transform = VARIANTS[args.arm]
    for path in (HDR, TWIN):
        before = open(path).read()
        after = transform(before)
        if after == before:
            sys.exit(f"{path}: arm '{args.arm}' changed nothing")
        print(f"{path}: {len(before)} -> {len(after)} bytes "
              f"({len(after) - len(before):+d})")
        if not args.check:
            with open(path, "w") as fh:
                fh.write(after)


if __name__ == "__main__":
    main()
