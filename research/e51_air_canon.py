#!/usr/bin/env python3
"""E51 step 0b: is the BF16 affine-bias expression tree a real dose, or does the
compiler already discard it?

The scored path compiles the JIT string with `setFastMathEnabled(false)`
(Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp:631), so the
question has two answers and only one of them matters. This emits AIR for the
runtime-effective JIT source under BOTH math modes and diffs the arms.

Arms, applied to the ONE line the campaign invariants pin
(`sums[m] += xm[0] + xm[1] + xm[2] + xm[3];`, quantized.h / quantized.cpp):

    r0  unchanged control
    r1  `(xm[0] + xm[1]) + (xm[2] + xm[3])`   pure reassociation, same operands
    r2  `xc[0] + xc[1] + xc[2] + xc[3]`       the fp32 values already in registers

CANONICALISATION IS THE TRAP. AIR names are position-dependent, so a diff of raw
text is all false positives; but a canonicaliser that maps every `%name` to one
token erases operand dataflow and then reports left-associated and balanced trees
as IDENTICAL. This renames SSA values in first-occurrence order, which is
injective and therefore dataflow-preserving, and it refuses to report anything
until its own positive control (an isolated R0/R1 tree pair that MUST differ
under safe math) has fired.

Research-only. Nothing here is on the scored path.

    research/e51_air_canon.py                 # full readout
    research/e51_air_canon.py --arms r0 r1    # subset
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from jit_string_compile import assemble  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTRY_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"

R0_LINE = "sums[m] += xm[0] + xm[1] + xm[2] + xm[3];"
ARM_LINES = {
    "r0": R0_LINE,
    "r1": "sums[m] += (xm[0] + xm[1]) + (xm[2] + xm[3]);",
    "r2": "sums[m] += xc[0] + xc[1] + xc[2] + xc[3];",
}

# The canonicaliser's own positive control: same operands, same operator, same
# count, association only. Under safe math these MUST differ.
# `typedef bfloat bfloat16_t` in the JIT preamble, so the tree is native BF16.
TREE_PROBE = """#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

[[kernel]] void e51_tree(
    const device bfloat16_t* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  const device bfloat16_t* xm = x + 4 * gid;
  float sums = 0.0f;
  __TREE__
  y[gid] = sums;
}
"""
TREE_ARMS = {
    "r0": "sums += xm[0] + xm[1] + xm[2] + xm[3];",
    "r1": "sums += (xm[0] + xm[1]) + (xm[2] + xm[3]);",
    "r2": ("float xc[4] = {float(xm[0]), float(xm[1]), float(xm[2]), float(xm[3])};"
           " sums += xc[0] + xc[1] + xc[2] + xc[3];"),
}

SSA = re.compile(r"%[-A-Za-z0-9_.$]+")
META = re.compile(r"![-A-Za-z0-9_.]*\d+")
FP_OP = re.compile(r"\b(fadd|fmul|fsub|fdiv|fpext|fptrunc|fma)\b\s+(?:\w+\s+)*?(bfloat|float|half)")


# "safe" is the scored mode (device.cpp:631). "default" is what an ordinary
# `xcrun metal` invocation does, which is what research/e40_cell_air.sh and
# research/e44_sgmm_air.sh used; it is reported only to show that difference.
MATH_MODES = {"safe": ["-fno-fast-math"], "default": [], "fast": ["-ffast-math"]}


def emit_air(source: str, mode: str) -> str:
    with tempfile.TemporaryDirectory(prefix="e51-air-") as directory:
        path = pathlib.Path(directory) / "arm.metal"
        path.write_text(source)
        command = [
            "xcrun", "-sdk", "macosx", "metal", "-x", "metal", "-std=metal4.0",
            *MATH_MODES[mode], "-S", "-o", "-", str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"AIR emit failed ({mode} math):\n{result.stderr}")
        return result.stdout


def canonicalise(air: str) -> list[str]:
    """Order-preserving alpha-renaming: injective on names, so dataflow survives."""
    names: dict[str, str] = {}

    def rename(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in names:
            names[token] = f"%v{len(names)}"
        return names[token]

    metas: dict[str, str] = {}

    def rename_meta(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in metas:
            metas[token] = f"!m{len(metas)}"
        return metas[token]

    lines = []
    for raw in air.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("."):
            continue
        line = SSA.sub(rename, line)
        line = META.sub(rename_meta, line)
        lines.append(line)
    return lines


def fp_ops(air: str) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for line in air.splitlines():
        for op, ty in FP_OP.findall(line):
            counts[f"{op} {ty}"] += 1
    return counts


def diff_lines(a: list[str], b: list[str]) -> int:
    import difflib
    return sum(1 for d in difflib.unified_diff(a, b, n=0, lineterm="")
               if d.startswith(("+", "-")) and not d.startswith(("+++", "---")))


def arm_source(arm: str) -> str:
    source = assemble((ENTRY_CELL,), None)
    if source.count(R0_LINE) != 1:
        raise SystemExit(f"expected exactly one pinned bias line in the JIT string, "
                         f"found {source.count(R0_LINE)}")
    return source.replace(R0_LINE, ARM_LINES[arm])


def report(name: str, arms: dict[str, str], reference: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ref_air = {mode: emit_air(arms[reference], mode) for mode in MATH_MODES}
    ref_canon = {mode: canonicalise(air) for mode, air in ref_air.items()}
    print(f"\n=== {name} ===")
    for mode in MATH_MODES:
        print(f"  {reference} @{mode:<8}: {len(ref_canon[mode])} canonical lines, "
              f"fp ops {dict(sorted(fp_ops(ref_air[mode]).items()))}")
    for arm, source in arms.items():
        if arm == reference:
            continue
        verdicts = {}
        for mode in MATH_MODES:
            air = emit_air(source, mode)
            canon = canonicalise(air)
            changed = diff_lines(ref_canon[mode], canon)
            verdicts[mode] = {
                "verdict": "IDENTICAL" if changed == 0 else "DIFFERS",
                "changed_canonical_lines": changed,
                "fp_ops": dict(sorted(fp_ops(air).items())),
            }
        print(f"  {arm} vs {reference}: " + " | ".join(
            f"{mode}{'(SCORED)' if mode == 'safe' else ''} {verdicts[mode]['verdict']}"
            f" ({verdicts[mode]['changed_canonical_lines']} lines)"
            for mode in MATH_MODES))
        for mode in MATH_MODES:
            print(f"      {arm} @{mode:<8} fp ops: {verdicts[mode]['fp_ops']}")
        out[arm] = verdicts
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=["r0", "r1", "r2"])
    args = parser.parse_args()

    toolchain = subprocess.run(["xcrun", "metal", "--version"], capture_output=True,
                               text=True).stdout.splitlines()[0]
    print(f"toolchain: {toolchain}")
    print("scored compile mode: fastMath DISABLED (device.cpp:631)")

    probes = {arm: TREE_PROBE.replace("__TREE__", TREE_ARMS[arm]) for arm in args.arms}
    control = report("isolated tree probe (canonicaliser positive control)",
                     probes, "r0")
    if "r1" in control and control["r1"]["safe"]["verdict"] != "DIFFERS":
        print("\nCANONICALISER CONTROL DID NOT FIRE: r0 vs r1 read IDENTICAL on the "
              "isolated tree under safe math. The instrument cannot see association, "
              "so every verdict below is VOID.")
        return 1
    print("  control: r0 vs r1 DIFFERS on the isolated tree under safe math -> the "
          "canonicaliser can see association.")

    entries = {arm: arm_source(arm) for arm in args.arms}
    report(f"runtime-effective JIT string, {ENTRY_CELL}", entries, "r0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
