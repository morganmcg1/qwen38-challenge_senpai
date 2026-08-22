#!/usr/bin/env python3
"""Check the advisor's per-simdgroup byte accounting against the arm source.

Advisor f1 asks for independent confirmation of the weight-side denominator
before any more GPU time is spent. The numbers below are read off the loop
structure of the transcribed `_wide` body, not copied from the brief.
"""

from __future__ import annotations

SIMD_SIZE = 32
ROWS_PER_SIMD = 4
VALUES_PER_THREAD = 16
BYTES_PER_LANE = 8          # 4 x uint16 of packed nibbles
BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE
GROUP = 64


def main() -> None:
    # Packed nibbles: every lane reads BYTES_PER_LANE for each of ROWS_PER_SIMD
    # rows, over the whole simdgroup.
    nibbles = ROWS_PER_SIMD * SIMD_SIZE * BYTES_PER_LANE
    assert nibbles == ROWS_PER_SIMD * BLOCK_SIZE // 2, nibbles

    # group_index advances as simd_lid / 4, so the 32 lanes of one row touch
    # BLOCK_SIZE / GROUP == 8 distinct groups, and each of the 4 rows has its
    # own groups.
    groups_per_row = BLOCK_SIZE // GROUP
    groups = ROWS_PER_SIMD * groups_per_row
    scales = groups * 2
    biases = groups * 2
    weight_side = nibbles + scales + biases

    rows = [
        ("packed nibbles", nibbles),
        ("scales", scales),
        ("biases", biases),
        ("weight-side total", weight_side),
    ]
    print(f"per simdgroup per k-block, rows_per_simd={ROWS_PER_SIMD} "
          f"block_size={BLOCK_SIZE} group={GROUP}")
    print(f"distinct groups touched: {groups} "
          f"({groups_per_row} per row x {ROWS_PER_SIMD} rows)")
    print(f"\n{'stream':<20}{'bytes':>8}{'share':>10}")
    for name, value in rows:
        print(f"{name:<20}{value:>8}{100.0 * value / weight_side:>9.3f} %")

    cut = biases // 2
    print(f"\nBias6 removes {cut} of {weight_side} weight-side bytes "
          f"= {100.0 * cut / weight_side:.4f} %")
    print(f"per-group view: 1 of {weight_side // groups} B "
          f"= {100.0 / (weight_side // groups):.4f} %")

    advisor = {"packed nibbles": 1024, "scales": 64, "biases": 64,
               "weight-side total": 1152}
    ok = all(advisor[name] == value for name, value in rows)
    print(f"\nmatches advisor f1 accounting: {ok}")
    if not ok:
        for name, value in rows:
            if advisor[name] != value:
                print(f"  MISMATCH {name}: advisor {advisor[name]}, "
                      f"source {value}")


if __name__ == "__main__":
    main()
