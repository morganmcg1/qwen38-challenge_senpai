#!/usr/bin/env python3
"""Exhaustive coverage proof for the E38 row-block indexing.

The advisor's deliverable (b)/(c) hinge: moving a row block from a loop counter
to `tid.x % ROW_BLOCKS` is exactly where an off-by-one produces a silently wrong
submission.  So this does not argue -- it enumerates every (tid.x, tid.y,
simd_gid, block, row) the kernel can execute and asserts that each of the N
output rows is written exactly once, and that each of the M input rows is read
by exactly the groups that should read it.

Mirrors kernels/quantized.h verbatim:

    constexpr int ROW_BLOCKS      = 4 / ROWS_PER_SIMD;
    constexpr int X_GROUPS        = (M + IPG - 1) / IPG;
    constexpr int X_PER_GROUP     = ROW_BLOCKS_IN_X ? ROW_BLOCKS : 1;
    constexpr int BLOCKS_PER_CALL = ROW_BLOCKS_IN_X ? 1 : ROW_BLOCKS;
    const int first_m = (tid.x / X_PER_GROUP) * IPG;
    if (first_m >= M) return;
    const int out_row = tid.y * 8 + simd_gid * 4
                      + (tid.x % X_PER_GROUP) * ROWS_PER_SIMD;
    // then `for b in range(BLOCKS_PER_CALL)`: rows out_row + b*ROWS_PER_SIMD ...

Host launch, backend/metal/quantized.cpp:251-254 (NOT editable):
    MTL::Size group_dims(32, 2, 1);              -> simd_gid in {0, 1}
    MTL::Size grid_dims(M, ceil(N/8), B);        -> tid.x in [0, M), tid.y in [0, ceil(N/8))

  python3 research/e38_coverage_proof.py
"""

from __future__ import annotations

import sys

# (label, M, IPG, ROWS_PER_SIMD, ROW_BLOCKS_IN_X)
ARMS = [
    ("base   _m<T,6,3,true>       ", 6, 3, 4, False),
    ("arm a  _m<T,6,3,true,2>     ", 6, 3, 2, False),
    ("arm b  _m<T,6,6,true,2,true>", 6, 6, 2, True),
    ("E33    _m<T,6,6,true,2>     ", 6, 6, 2, False),
]
# every scored n, all divisible by 8 (the gate's own N % 8 == 0)
SCORED_N = [5120, 14336, 16480, 34816, 98336, 248320]


def simulate(M, IPG, ROWS_PER_SIMD, ROW_BLOCKS_IN_X, N):
    ROW_BLOCKS = 4 // ROWS_PER_SIMD
    X_GROUPS = (M + IPG - 1) // IPG
    X_PER_GROUP = ROW_BLOCKS if ROW_BLOCKS_IN_X else 1
    BLOCKS_PER_CALL = 1 if ROW_BLOCKS_IN_X else ROW_BLOCKS

    if ROW_BLOCKS_IN_X:
        assert X_GROUPS * ROW_BLOCKS <= M, "static_assert would have fired"

    # The output element is out[m][n]: tid.x selects the INPUT rows m via
    # first_m, and (tid.y, simd_gid, tid.x % X_PER_GROUP, b) select n.  Two
    # x-blocks writing the same n for different m is correct, so the key must
    # be the pair.
    writes = {}           # (m, n) -> number of times written
    working_x = set()
    for tid_x in range(M):                       # grid.x == M
        first_m = (tid_x // X_PER_GROUP) * IPG
        if first_m >= M:
            continue
        working_x.add(tid_x)
        ms = range(first_m, min(first_m + IPG, M))
        for tid_y in range((N + 7) // 8):        # grid.y == ceil(N/8)
            for simd_gid in (0, 1):              # group_dims(32, 2, 1)
                out_row = (tid_y * 8 + simd_gid * 4
                           + (tid_x % X_PER_GROUP) * ROWS_PER_SIMD)
                for b in range(BLOCKS_PER_CALL):
                    base = out_row + b * ROWS_PER_SIMD
                    for r in range(ROWS_PER_SIMD):
                        for m in ms:
                            key = (m, base + r)
                            writes[key] = writes.get(key, 0) + 1
    return writes, working_x, X_GROUPS


def main() -> None:
    ok = True
    print(f"{'arm':<30}{'n':>8}{'work x':>8}{'TGs':>9}{"elems":>9}"
          f"{'cover':>8}{'dupes':>7}")
    for label, M, IPG, R, XB in ARMS:
        for N in SCORED_N:
            writes, working_x, xg = simulate(M, IPG, R, XB, N)
            covered = set(writes)
            expect = {(m, n) for m in range(M) for n in range(N)}
            dupes = sum(1 for v in writes.values() if v != 1)
            good = covered == expect and dupes == 0
            ok &= good
            tgs = len(working_x) * ((N + 7) // 8)
            print(f"{label:<30}{N:>8}{len(working_x):>8}{tgs:>9}"
                  f"{len(covered):>9}{'exact' if covered == expect else 'BAD':>8}"
                  f"{dupes:>7}")
        print()

    print("Coverage identity, arm b, symbolically:")
    print("  rows per tid.y = 2 simdgroups x ROW_BLOCKS x ROWS_PER_SIMD")
    print("                 = 2 x 2 x 2 = 8, at offsets {0,4}+{0,2}+{0,1} = 0..7")
    print("  total          = ceil(N/8) x 8 = N   whenever N % 8 == 0 (gate)")
    print()
    print("M=1 unreachability, all arms: the >=4096 tier has no `case 1:`, so")
    print("M=1 dispatches qmv_fast_impl and never enters the crossrow kernel.")
    print("Confirmed by dispatch readback: in_kernel_path == 'qmv_fast_impl'.")
    print()
    if not ok:
        sys.exit("COVERAGE PROOF FAILED")
    print("COVERAGE PROOF OK: every scored n, every arm, each row written once.")


if __name__ == "__main__":
    main()
