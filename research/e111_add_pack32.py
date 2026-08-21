#!/usr/bin/env python3
"""Derive the g_pack32 arm from a_shipped by transcription, not by hand.

The arm must differ from a_shipped in exactly one place: where the pair
`scales[g], biases[g]` is read. Copying the source and applying one checked
substitution keeps that guarantee visible, so the arm cannot silently pick up
an unrelated change.
"""

from __future__ import annotations

import pathlib
import re
import sys

RULE = "// " + "-" * 75

BANNER = f"""{RULE}
// g_pack32: the shipped values, read as one 32-bit interleaved record.
//
// a_shipped reads scales[g] and biases[g] from two buffers at the same index,
// so it issues two 16-bit loads for every group of every row. This arm reads
// the identical pair from one interleaved uint32 array. Bytes streamed,
// values and summation order do not change, so the output must stay
// bit-identical and the arm prices the second load instruction on its own.
{RULE}
"""

OLD_LOAD = ("      scale_local[r] = scales[group_index];\n"
            "      bias_local[r] = biases[group_index];\n")

NEW_LOAD = ("      const uint32_t sb = packed_sb[group_index];\n"
            "      scale_local[r] = float(as_type<T>(uint16_t(sb & 0xFFFFu)));\n"
            "      bias_local[r] = float(as_type<T>(uint16_t(sb >> 16)));\n")

OLD_SIG = "    const constant ulong& wseed [[buffer(8)]],\n"
NEW_SIG = OLD_SIG + "    const device uint32_t* packed_sb [[buffer(9)]],\n"


def main() -> None:
    path = pathlib.Path("research/e111_bias6_arms.metal")
    src = path.read_text()
    if "e111_g_pack32" in src:
        print("g_pack32 already present")
        return

    lines = src.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("// a_shipped")) - 1
    end = next(i for i, l in enumerate(lines)
               if l.startswith("// n_nobias:")) - 1
    arm = "".join(lines[start:end])

    for token, count in ((OLD_LOAD, 1), (OLD_SIG, 1),
                         ("e111_a_shipped", 1)):
        if arm.count(token) != count:
            sys.exit(f"expected {count} of {token!r}, found {arm.count(token)}")

    arm = arm.replace("e111_a_shipped", "e111_g_pack32")
    arm = arm.replace(OLD_SIG, NEW_SIG)
    arm = arm.replace(OLD_LOAD, NEW_LOAD)
    arm = re.sub(r"\A// -+\n// a_shipped\n// -+\n", BANNER, arm)
    if not arm.startswith(RULE):
        sys.exit("banner substitution failed")

    src = src.replace("#define instantiate_kernel",
                      arm + "\n#define instantiate_kernel")
    tail = 'instantiate_kernel("c_loadonly", e111_c_loadonly, bfloat)\n'
    src = src.replace(
        tail, tail + 'instantiate_kernel("g_pack32", e111_g_pack32, bfloat)\n')
    path.write_text(src)
    print("added g_pack32")


if __name__ == "__main__":
    main()
