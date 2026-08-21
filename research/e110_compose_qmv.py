#!/usr/bin/env python3
"""E110 rung 2 step 1: build the COMPOSED qmv JIT source the scored worker
actually compiles, so a driver can report its threadgroup allocation.

`get_quantized_kernel` (jit_kernels.cpp:915) concatenates
utils() + gemm() + quantized_utils() + quantized() + one template
instantiation, then hands the string to `newLibraryWithSource`. The four
preambles live in Vendor/mlx-swift/Source/Cmlx/mlx-generated/*.cpp, so this
script reproduces that string from the runtime-effective twins, not from the
readable .metal tree.

`--tile` inserts an allocation probe: a `threadgroup T xtile[NA * 512]` plus a
use that cannot be eliminated. It measures ONLY the static threadgroup
allocation of the composed entry point. It is not a functional `xs_stage` and
never runs.

Research-only. Nothing here is on the scored path.
"""

import argparse
import pathlib
import re
import sys

GEN = pathlib.Path("Vendor/mlx-swift/Source/Cmlx/mlx-generated")

WIDE_ANCHOR = "  static_assert(NA >= 2 && NA <= 5,"
# `if (batched) {` opens several kernels, so reach it through the one
# `affine_qmv_fast` signature instead.
ENTRY_SIGNATURE = "[[kernel]] void affine_qmv_fast("
ENTRY_ANCHOR = "  if (batched) {"

# NA * 512 bfloat16 = NA * 1024 B. The store, the barrier and the guarded
# device write keep the array live at every optimisation level.
WIDE_PROBE = """  threadgroup T xtile[NA * 512];
  xtile[simd_lid] = x[first_m * in_vec_size + simd_lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (xtile[0] == T(-1.0e30f)) {
    y[0] = xtile[NA * 512 - 1];
  }
"""

ENTRY_PROBE = """  threadgroup T xtile[5 * 512];
  xtile[simd_lid] = x[simd_lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (xtile[0] == T(-1.0e30f)) {
    y[0] = xtile[5 * 512 - 1];
  }
"""


def preamble(name: str) -> str:
    text = (GEN / f"{name}.cpp").read_text()
    m = re.search(r'R"preamble\((.*)\)preamble";', text, re.S)
    if not m:
        sys.exit(f"e110_compose_qmv: no preamble string in {name}.cpp")
    return m.group(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", choices=("none", "wide", "entry"), default="none")
    ap.add_argument("--type", default="bfloat16_t")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    quantized = preamble("quantized")
    if args.tile == "wide":
        if quantized.count(WIDE_ANCHOR) != 1:
            sys.exit("e110_compose_qmv: wide anchor is not unique")
        quantized = quantized.replace(WIDE_ANCHOR, WIDE_PROBE + WIDE_ANCHOR, 1)
    elif args.tile == "entry":
        if quantized.count(ENTRY_SIGNATURE) != 1:
            sys.exit("e110_compose_qmv: entry signature is not unique")
        start = quantized.index(ENTRY_SIGNATURE)
        body = quantized.index(ENTRY_ANCHOR, start)
        quantized = quantized[:body] + ENTRY_PROBE + quantized[body:]

    src = [preamble("utils"), preamble("gemm"), preamble("quantized_utils"), quantized]
    t, gs, b = args.type, args.group_size, args.bits
    for batched in (1, 0):
        name = f"affine_qmv_fast_{t}_gs_{gs}_b_{b}_batch_{batched}"
        src.append(
            f'\ntemplate [[host_name("{name}")]] [[kernel]] decltype('
            f"affine_qmv_fast<{t}, {gs}, {b}, {batched}>) "
            f"affine_qmv_fast<{t}, {gs}, {b}, {batched}>;\n"
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(src))
    print(f"e110_compose_qmv: wrote {out} tile={args.tile} bytes={out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
