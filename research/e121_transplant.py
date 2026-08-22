#!/usr/bin/env python3
"""E121 rung 3: put the gated cross-simdgroup chunk-sum share into the tree.

    python3 research/e121_transplant.py
    python3 research/twin_audit.py

Both simdgroups of the wide-QMV threadgroup recompute the identical activation
chunk sums, because `sums[m]` depends only on m, k, i and the lane. At NA <= 4
each simdgroup computes one half and the halves are exchanged once per k-block.
Lane l reads lane l's value, produced from the same addresses in the same order
by the same bf16 expression tree, so every `sums[m]` is bit-identical to the
redundant form. At NA == 5 `SHARE_SUMS` is false, `owns_m` folds to a constant
true, and the whole shared path compiles out: the rung-0 census proves the NA=5
machine text is byte-identical to the base on both architectures.

The readable header and the JIT twin carry the same Metal source, so the same
edits are applied to both, each asserted to match exactly once. No comment line
is added to either file: comments de-pin the allowlisted `twin_audit` waiver
(HARNESS DEFECT 3), and E121 feedback 2 records a second, empirical reason.
The rationale lives in `research/e121-results.md` instead.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e121_arms import patch_dispatch  # noqa: E402

TARGETS = [
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
]

# Depth 1 buffer sized for NA <= 4 only, because NA=5 never reaches it:
# 4 * 32 floats = 512 B, against a 32768 B threadgroup budget.
BUFFER_DEPTH = 1
BUFFER_MAX_NA = 4

EDITS = [
    (
        "    int first_m,\n"
        "    int out_row,\n"
        "    uint simd_lid) {\n"
        '  static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");',
        "    int first_m,\n"
        "    int out_row,\n"
        "    uint simd_gid,\n"
        "    uint simd_lid,\n"
        "    threadgroup float* sums_xchg) {\n"
        '  static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");',
    ),
    (
        "  constexpr int bytes_per_lane = 8;\n"
        "  const int in_vec_size_w = in_vec_size / 2;",
        "  constexpr int bytes_per_lane = 8;\n"
        "  constexpr bool SHARE_SUMS = NA <= 4;\n"
        "  constexpr int H = NA / 2;\n"
        "  const bool own_lo = simd_gid == 0;\n"
        "  const int in_vec_size_w = in_vec_size / 2;",
    ),
    (
        "        thread float xc[4];\n"
        "        if (DIRECT_NIBBLES) {",
        "        thread float xc[4];\n"
        "        const bool owns_m = !SHARE_SUMS || ((m < H) == own_lo);\n"
        "        if (DIRECT_NIBBLES) {",
    ),
    (
        "          sums[m] += xv[0] + xv[1] + xv[2] + xv[3];\n"
        "        } else {\n"
        "          sums[m] += load_vector<T, float, 4, 4>(xm, xc);\n"
        "        }",
        "          if (owns_m) {\n"
        "            sums[m] += xv[0] + xv[1] + xv[2] + xv[3];\n"
        "          }\n"
        "        } else {\n"
        "          const float xsum = load_vector<T, float, 4, 4>(xm, xc);\n"
        "          if (owns_m) {\n"
        "            sums[m] += xsum;\n"
        "          }\n"
        "        }",
    ),
    (
        "    for (int r = 0; r < rows_per_simd; r++) {\n"
        "      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];\n"
        "    }\n"
        "  }",
        "    if constexpr (SHARE_SUMS) {\n"
        "      for (int m = 0; m < NA; m++) {\n"
        "        if ((m < H) == own_lo) {\n"
        "          sums_xchg[m * SIMD_SIZE + simd_lid] = sums[m];\n"
        "        }\n"
        "      }\n"
        "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "      for (int m = 0; m < NA; m++) {\n"
        "        if ((m < H) != own_lo) {\n"
        "          sums[m] = sums_xchg[m * SIMD_SIZE + simd_lid];\n"
        "        }\n"
        "      }\n"
        "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    for (int r = 0; r < rows_per_simd; r++) {\n"
        "      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];\n"
        "    }\n"
        "  }",
    ),
]


def main() -> int:
    for name in TARGETS:
        path = pathlib.Path(name)
        text = path.read_text()
        before = len(text)
        for old, new in EDITS:
            seen = text.count(old)
            if seen != 1:
                raise SystemExit(
                    "%s: pattern matched %d times, expected 1: %r"
                    % (name, seen, old.splitlines()[0]))
        for old, new in EDITS:
            text = text.replace(old, new)
        text = patch_dispatch(text, BUFFER_DEPTH, BUFFER_MAX_NA)
        path.write_text(text)
        print("patched %-70s %+d bytes" % (name, len(text) - before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
