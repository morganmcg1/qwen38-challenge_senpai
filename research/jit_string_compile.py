#!/usr/bin/env python3
"""Compile the RUNTIME-EFFECTIVE JIT source for a quantized kernel.

`tools/build-mlx-metallib.sh` proves the AOT path: it compiles the readable
`kernels/*.metal` with the vendored include tree available. The scored worker can
instead take the JIT path, where MLX builds a SELF-CONTAINED source string by
concatenating checked-in `mlx-generated/*.cpp` preambles and appending one
`[[host_name]]` instantiation, then hands it to `newLibrary` with NO include
path at all (mlx/backend/metal/jit_kernels.cpp get_quantized_kernel,
mlx/backend/metal/device.cpp build_library_from_source).

The two paths therefore fail differently. A quoted include added to a readable
header is expanded INTO the twin by MLX's generator, which can duplicate a
definition the concatenation already supplied; and a symbol resolved in the AOT
path by an include that quantized.h does not itself carry has to be supplied by
an earlier preamble in the JIT concatenation instead. Neither is visible from an
AOT build, so this reproduces the JIT string byte-for-byte and compiles it.

Usage:
    research/jit_string_compile.py                       # default qmv_fast cells
    research/jit_string_compile.py 'affine_qmv_fast<bfloat16_t, 64, 4, false>'
    research/jit_string_compile.py --emit /tmp/arm.metal --rev efff400c -- CELL

`--emit` writes the assembled string instead of only compiling it, so an A/B
harness can hand the same bytes to `newLibraryWithSource:`. `--rev` reads the
generated twins from a git revision, which is what makes a base arm and a
candidate arm available to one process in one session.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx-generated"
# get_quantized_kernel, mode == "affine".
PREAMBLES = ("utils", "gemm", "quantized_utils", "quantized")
PREAMBLE_BODY = re.compile(r'R"preamble\(\n(.*)\n\)preamble"', re.DOTALL)
DEFAULT_CELLS = (
    "affine_qmv_fast<bfloat16_t, 64, 4, false>",
    "affine_qmv_fast<bfloat16_t, 64, 4, true>",
    "affine_qmv_fast<bfloat16_t, 64, 2, false>",
    "affine_qmv<bfloat16_t, 64, 4, false>",
    # The steel BlockMMA consumer in the same header: it is what proves
    # quantized.h still gets mlx::steel from the gemm preamble and not from a
    # duplicate copy expanded into the quantized twin.
    "affine_qmm_t<bfloat16_t, 64, 4, true, false>",
)


def preamble(stem: str, rev: str | None) -> str:
    relative = f"Vendor/mlx-swift/Source/Cmlx/mlx-generated/{stem}.cpp"
    if rev is None:
        text = (GEN_DIR / f"{stem}.cpp").read_text()
    else:
        text = subprocess.run(
            ["git", "show", f"{rev}:{relative}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    match = PREAMBLE_BODY.search(text)
    if not match:
        raise SystemExit(f"{stem}.cpp: no R\"preamble(...)\" body")
    return match.group(1) + "\n"


def host_name(cell: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", cell).strip("_")


def template_def(cell: str) -> str:
    # jit_kernels.cpp get_template_definition, with the AOT host name.
    return (
        f'\ntemplate [[host_name("{host_name(cell)}")]] [[kernel]] '
        f"decltype({cell}) {cell};\n"
    )


def assemble(cells: tuple[str, ...], rev: str | None) -> str:
    source = "".join(preamble(stem, rev) for stem in PREAMBLES)
    return source + "".join(template_def(cell) for cell in cells)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--emit", type=pathlib.Path)
    parser.add_argument("--rev")
    parser.add_argument("cells", nargs="*")
    args = parser.parse_args()

    cells = tuple(args.cells) or DEFAULT_CELLS
    source = assemble(cells, args.rev)

    if args.emit is not None:
        args.emit.write_text(source)
        print(f"{args.emit} {len(source)} "
              f"{args.rev or 'worktree'} {' '.join(host_name(c) for c in cells)}")
        return 0

    with tempfile.TemporaryDirectory(prefix="qwen38-jit-") as directory:
        path = pathlib.Path(directory) / "jit.metal"
        path.write_text(source)
        # device.cpp: setLanguageVersion(LanguageVersion4_0) on macOS 26 and
        # setFastMathEnabled(false). No -I: the string must be self-contained.
        command = [
            "xcrun", "-sdk", "macosx", "metal", "-x", "metal",
            "-std=metal4.0", "-fno-fast-math", "-c", str(path),
            "-o", str(path.with_suffix(".air")),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"JIT source: {len(source)} bytes from "
              f"{' + '.join(PREAMBLES)} + {len(cells)} instantiation(s)")
        for cell in cells:
            print(f"  cell: {cell}")
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode:
            print("JIT STRING COMPILE FAILED")
            return 1
    print("JIT STRING COMPILE OK (no include path, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
