#!/usr/bin/env python3
"""Emit the three E98 rung-1b kernel arms as self-contained Metal sources.

    research/e98_variant_sources.py --outdir /tmp/e98-arms

Arm (a) is the unmodified runtime-effective JIT string for
`affine_qmv_fast<bfloat16_t, 64, 4, false>`, assembled by
`research/jit_string_compile.py --emit`. Arms (b) and (c) are that same string
with the metadata fetch replaced:

  (a) shipped   scale = scales[g], bias = biases[g]        36 B per 64 elements
  (b) indexed   idx = ushort(scales[g]); scale = lut[2*idx], bias = lut[2*idx+1]
                where lut is the front of the biases buffer  34 B per 64 elements
  (c) constant  scale and bias are literals, no metadata read  32 B per 64
                elements. Numerically wrong on purpose: a timing-only upper
                bound that must never reach a candidate path.

Every substitution count is asserted, so a source drift fails loudly instead of
silently producing an arm that is identical to another arm.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"

# Arm (c) literals. Any finite value works; these match the magnitude the
# harness writes so the arm does not denormal-stall on a value the shipped arm
# never sees.
CONST_SCALE = "0.006f"
CONST_BIAS = "-0.045f"

# `qmv_impl` offsets `biases` in two mutually exclusive branches, so the
# capture has to happen in both or the second branch loses the name.
LUT_CAPTURE = re.compile(
    r"^( *)biases \+= (used_out_row|out_row) \* in_vec_size_g", re.M
)
SL_PAIR = re.compile(r"( *)U s = sl\[0\];\n( *)U b = bl\[0\];")
LOCAL_PAIR = re.compile(
    r"( *)scale_local\[r\] = scales\[group_index\];\n"
    r"( *)bias_local\[r\] = biases\[group_index\];"
)

EXPECT = {"lut_capture": 4, "sl_pair": 6, "local_pair": 3}


def emit_base(path: pathlib.Path) -> str:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "research/jit_string_compile.py"),
            "--emit",
            str(path),
            "--",
            CELL,
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return path.read_text()


def substitute(text: str, pattern: re.Pattern, repl, label: str) -> str:
    out, count = pattern.subn(repl, text)
    if count != EXPECT[label]:
        raise SystemExit(
            f"e98_variant_sources: {label} matched {count} sites, expected "
            f"{EXPECT[label]}. The generated twin moved; re-derive the patch."
        )
    return out


def arm_b(text: str) -> str:
    text = substitute(
        text,
        LUT_CAPTURE,
        lambda m: (
            f"{m.group(1)}const device T* e98_lut = biases;\n"
            f"{m.group(1)}biases += {m.group(2)} * in_vec_size_g"
        ),
        "lut_capture",
    )
    text = substitute(
        text,
        SL_PAIR,
        lambda m: (
            f"{m.group(1)}const ushort e98_idx =\n"
            f"{m.group(1)}    reinterpret_cast<const device ushort*>(sl)[0];\n"
            f"{m.group(1)}U s = e98_lut[2 * e98_idx];\n"
            f"{m.group(2)}U b = e98_lut[2 * e98_idx + 1];"
        ),
        "sl_pair",
    )
    return substitute(
        text,
        LOCAL_PAIR,
        lambda m: (
            f"{m.group(1)}const ushort e98_idx =\n"
            f"{m.group(1)}    reinterpret_cast<const device ushort*>(scales)[group_index];\n"
            f"{m.group(1)}scale_local[r] = biases[2 * e98_idx];\n"
            f"{m.group(2)}bias_local[r] = biases[2 * e98_idx + 1];"
        ),
        "local_pair",
    )


def arm_c(text: str) -> str:
    text = substitute(
        text,
        SL_PAIR,
        lambda m: (
            f"{m.group(1)}U s = {CONST_SCALE};\n{m.group(2)}U b = {CONST_BIAS};"
        ),
        "sl_pair",
    )
    return substitute(
        text,
        LOCAL_PAIR,
        lambda m: (
            f"{m.group(1)}scale_local[r] = {CONST_SCALE};\n"
            f"{m.group(2)}bias_local[r] = {CONST_BIAS};"
        ),
        "local_pair",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/e98-arms")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = outdir / "arm_a.metal"
    text = emit_base(base)
    (outdir / "arm_b.metal").write_text(arm_b(text))
    (outdir / "arm_c.metal").write_text(arm_c(text))

    for name in ("arm_a", "arm_b", "arm_c"):
        path = outdir / f"{name}.metal"
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
