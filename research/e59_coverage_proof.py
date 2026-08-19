#!/usr/bin/env python3
"""E59 rung 1 gate: exhaustive output-row coverage of every row-block mapping.

The host grid is IPG-blind and frozen (`backend/metal/quantized.cpp`):

    bn = 8;  bk = 32;  group_dims(bk, 2, 1);  grid_dims(M, (N + bn - 1)/bn, B)

so each `tid.y` owns eight output rows and exactly two simdgroups, whatever
`rows_per_simd` the kernel uses. At `rows_per_simd = 2` a mapping must therefore
cover eight rows with four two-row blocks, and item 99's wall is that the naive
`out_row = tid.y*8 + simd_gid*2` covers only four of them.

This enumerates the write set of each mapping over EVERY thread coordinate the
frozen grid produces, as `(input_row, output_row)` pairs, and requires the
multiset to be the full product exactly once. The two coverage-defect controls
must fail, otherwise the check has no power.

No GPU, no build. Run it before spending a GPU hour:

    python3 research/e59_coverage_proof.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter


def shipped(M: int, IPG: int, N: int) -> Counter:
    """`..._m`: rows_per_simd = 4, one block per simdgroup."""
    seen: Counter = Counter()
    for tx in range(M):
        first_m = tx * IPG
        if first_m >= M:
            continue
        na = IPG if (M % IPG == 0 or M - first_m >= IPG) else max(M % IPG, 2)
        for ty in range(N // 8):
            for sg in range(2):
                out_row = ty * 8 + sg * 4
                for r in range(4):
                    for m in range(na):
                        seen[(first_m + m, out_row + r)] += 1
    return seen


def rb2(M: int, IPG: int, N: int, blocks: int = 2) -> Counter:
    """`..._m_rb2`: rows_per_simd = 2, two SEQUENTIAL blocks in one x-group."""
    seen: Counter = Counter()
    for tx in range(M):
        first_m = tx * IPG
        if first_m >= M:
            continue
        for ty in range(N // 8):
            for sg in range(2):
                for rb in range(blocks):
                    out_row = ty * 8 + sg * 4 + rb * 2
                    for r in range(2):
                        for m in range(IPG):
                            seen[(first_m + m, out_row + r)] += 1
    return seen


def rbx(M: int, IPG: int, N: int, groups: int = 2) -> Counter:
    """`..._m_rbx`: rows_per_simd = 2, two PARALLEL blocks in two x-groups."""
    seen: Counter = Counter()
    for tx in range(M):
        if tx >= groups:
            continue
        for ty in range(N // 8):
            for sg in range(2):
                out_row = ty * 8 + tx * 4 + sg * 2
                for r in range(2):
                    for m in range(IPG):
                        seen[(m, out_row + r)] += 1
    return seen


def verdict(seen: Counter, M: int, N: int) -> dict:
    want = {(m, n) for m in range(M) for n in range(N)}
    got = set(seen)
    return {
        "pairs_expected": len(want),
        "pairs_written": len(got),
        "never_written": sorted(want - got)[:8],
        "never_written_count": len(want - got),
        "written_twice": sorted(k for k, v in seen.items() if v > 1)[:8],
        "written_twice_count": sum(1 for v in seen.values() if v > 1),
        "out_of_range_count": len(got - want),
        "exact_cover": got == want and all(v == 1 for v in seen.values()),
    }


# N is any multiple of 8; three tiles is enough to expose a tile-relative bug,
# and the mapping is affine in tid.y so more tiles add nothing.
CASES = [
    ("shipped <T,5,3>", lambda N: shipped(5, 3, N), 5, True),
    ("shipped <T,5,5> r=4", lambda N: shipped(5, 5, N), 5, True),
    ("shipped <T,7,4>", lambda N: shipped(7, 4, N), 7, True),
    ("shipped <T,9,3>", lambda N: shipped(9, 3, N), 9, True),
    ("rb2 <T,5,5> r=2", lambda N: rb2(5, 5, N), 5, True),
    ("rbx <T,5,5> r=2", lambda N: rbx(5, 5, N), 5, True),
    ("CONTROL rb2 one block", lambda N: rb2(5, 5, N, blocks=1), 5, False),
    ("CONTROL rbx one x-group", lambda N: rbx(5, 5, N, groups=1), 5, False),
    ("CONTROL rb2 three blocks", lambda N: rb2(5, 5, N, blocks=3), 5, False),
    ("CONTROL rbx three x-groups", lambda N: rbx(5, 5, N, groups=3), 5, False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e59-artifacts/e59-coverage.json")
    ap.add_argument("--n", type=int, default=24, help="output rows, multiple of 8")
    args = ap.parse_args()

    rows = []
    ok = True
    for label, build, M, expect_cover in CASES:
        res = verdict(build(args.n), M, args.n)
        passed = res["exact_cover"] == expect_cover
        ok &= passed
        rows.append(dict(res, mapping=label, expect_exact_cover=expect_cover,
                         passed=passed))
        print("  %-28s cover=%-5s expected=%-5s  missing=%-4d twice=%-4d  %s"
              % (label, res["exact_cover"], expect_cover,
                 res["never_written_count"], res["written_twice_count"],
                 "ok" if passed else "FAILED"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"out_vec_size": args.n, "cases": rows, "all_passed": bool(ok)},
        indent=2, sort_keys=True) + "\n")
    print("\nall_passed=%s   wrote %s" % (ok, args.out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
