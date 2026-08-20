#!/usr/bin/env python3
"""E87 arm G: generalise the promoted 2-bit single-row QMV over the group size.

The readable Metal header and its runtime-effective JIT twin carry the same
kernel text, so both get exactly the same edit from one table. Running the
script twice is a no-op check: every pattern must match exactly once.
"""

from pathlib import Path

EDITS = [
    (
        "// Single-row (M == 1) affine2/g64 fast QMV for the coarse compact draft",
        "// Single-row (M == 1) affine-2 fast QMV for the coarse compact draft",
    ),
    (
        "// qmv_fast_impl<T, 64, 2> value; the wider lane coverage reassociates the",
        "// qmv_fast_impl<T, group_size, 2> value; the wider lane coverage\n"
        "// reassociates the",
    ),
    (
        "template <typename T>\nMETAL_FUNC void qmv_fast_singlerow_affine2_g64(",
        "// The group size is a template parameter, not a constant. A lane covers 32\n"
        "// contiguous values, so any group size that is a multiple of 32 keeps one\n"
        "// lane inside one scale group and leaves the arithmetic above unchanged.\n"
        "// Only the two scale/bias divisors depend on it, so instantiating at 64\n"
        "// reproduces the promoted kernel value for value.\n"
        "template <typename T, int group_size>\n"
        "METAL_FUNC void qmv_fast_singlerow_affine2(",
    ),
    (
        "  const int in_vec_size_w = in_vec_size / 4;   // weight bytes per output row\n"
        "  const int in_vec_size_g = in_vec_size / 64;  // scale groups per output row",
        "  static_assert(\n"
        "      group_size % values_per_thread == 0,\n"
        "      \"a lane's 32 values must lie inside one scale group\");\n"
        "  const int in_vec_size_w = in_vec_size / 4;  // weight bytes per output row\n"
        "  const int in_vec_size_g =\n"
        "      in_vec_size / group_size;  // scale groups per output row",
    ),
    (
        "      // 32 values per lane = half of one 64-value group.\n"
        "      const int group_index =\n"
        "          row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;",
        "      const int group_index = row * in_vec_size_g + k / group_size +\n"
        "          (simd_lid * values_per_thread) / group_size;",
    ),
    (
        "  if (!batched && group_size == 64 && bits == 2 && out_vec_size == 98336 &&\n"
        "      ntg.x == 1) {",
        "  if (!batched && (group_size == 64 || group_size == 128) && bits == 2 &&\n"
        "      out_vec_size == 98336 && ntg.x == 1) {",
    ),
    (
        "    qmv_fast_singlerow_affine2_g64<T>(",
        "    qmv_fast_singlerow_affine2<T, group_size>(",
    ),
]

TARGETS = [
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
]


def main() -> None:
    for name in TARGETS:
        path = Path(name)
        text = path.read_text()
        for old, new in EDITS:
            count = text.count(old)
            if count != 1:
                raise SystemExit(f"{name}: pattern matched {count} times: {old[:70]!r}")
            text = text.replace(old, new)
        path.write_text(text)
        print("patched", name)


if __name__ == "__main__":
    main()
