#!/usr/bin/env python3
"""Harness-native bitwise gate for the E38 arms.

Every cost-curve row carries `row0_bitwise_matches_m1`: the arm's row 0 at width
M compared against the same shape evaluated at M=1, which dispatches
`qmv_fast_impl` and never enters the crossrow tier.  So this is a direct
candidate-vs-serial comparison on the real scored shapes, produced by the same
run that produced the timing.

  python3 research/e38_bitwise_check.py e38-base-r1 e38-armb-r1 e38-arma-r1
"""

from __future__ import annotations

import json
import os
import sys

CURVE = ".mlxfast-private/qmv-curve/%s/vendored.json"


def check(tag: str) -> int:
    d = json.load(open(CURVE % tag))
    bad = 0
    checked = 0
    worst = 0.0
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if r["m"] == 1:
                continue
            m = r.get("row0_bitwise_matches_m1")
            delta = r.get("row0_max_abs_delta_vs_m1")
            if m is None:
                continue
            checked += 1
            if delta is not None:
                worst = max(worst, abs(delta))
            if not m:
                bad += 1
                print(f"    MISMATCH {sh['name']} M={r['m']} "
                      f"max_abs_delta={delta}")
    verdict = "BIT-IDENTICAL" if bad == 0 else f"{bad} MISMATCHES"
    print(f"  {tag:<16} {checked:>3} rows checked  worst |delta| = {worst:g}"
          f"   {verdict}")
    return bad


def main() -> None:
    tags = sys.argv[1:] or ["e38-base-r1", "e38-armb-r1", "e38-arma-r1"]
    print("row0 vs the M=1 serial reference, per shape x width:")
    total = 0
    for tag in tags:
        if not os.path.exists(CURVE % tag):
            print(f"  {tag:<16} (not present)")
            continue
        total += check(tag)
    print()
    if total:
        sys.exit(f"FAIL: {total} bitwise mismatch(es)")
    print("PASS: every arm reproduces the serial reference bit-for-bit.")


if __name__ == "__main__":
    main()
