#!/usr/bin/env python3
"""Explain how two E51 row traces differ.

The whole-trace digest is a coarse signal: the number of target rows a run
evaluates depends on the drafting schedule, so two runs of identical source can
emit different row counts. This script separates the two failure modes.

  schedule difference: the set or multiplicity of evaluated positions changed
  numeric difference:  a position that both runs evaluated reports different
                       top-two token evidence

Only the second mode is an exactness failure.

usage: research/e51_row_diff.py ARM_A ARM_B
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

ROW = re.compile(r"^mtp-row: pos=(\d+) ids=(\d+),(\d+) v=(\S+)")


def rows(arm: pathlib.Path) -> list[tuple[int, str, str, str]]:
    text = (arm / "trace.txt").read_text(errors="replace")
    parsed = [ROW.match(line) for line in text.splitlines()]
    return [(int(m.group(1)), m.group(2), m.group(3), m.group(4))
            for m in parsed if m]


def value_map(data) -> dict[int, set]:
    out = collections.defaultdict(set)
    for pos, id0, id1, value in data:
        out[pos].add((id0, id1, value))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm_a", type=pathlib.Path)
    parser.add_argument("arm_b", type=pathlib.Path)
    args = parser.parse_args()

    a = rows(args.arm_a)
    b = rows(args.arm_b)
    print(f"rows: {args.arm_a.name}={len(a)}  {args.arm_b.name}={len(b)}")

    mult_a = collections.Counter(r[0] for r in a)
    mult_b = collections.Counter(r[0] for r in b)
    print(f"distinct positions: {len(mult_a)} vs {len(mult_b)}")
    print(f"position sets equal: {set(mult_a) == set(mult_b)}")
    only_a = sorted(set(mult_a) - set(mult_b))
    only_b = sorted(set(mult_b) - set(mult_a))
    if only_a:
        print(f"  only in {args.arm_a.name}: {only_a[:12]}")
    if only_b:
        print(f"  only in {args.arm_b.name}: {only_b[:12]}")
    shared = sorted(set(mult_a) & set(mult_b))
    remult = [p for p in shared if mult_a[p] != mult_b[p]]
    print(f"shared positions with different evaluation count: {len(remult)}"
          f" {remult[:12]}")

    va = value_map(a)
    vb = value_map(b)
    numeric = [p for p in shared if va[p] != vb[p]]
    print()
    print(f"SHARED POSITIONS WITH DIFFERENT ROW EVIDENCE: {len(numeric)}")
    if numeric:
        print(f"  positions: {numeric[:12]}")
        for pos in numeric[:3]:
            print(f"  pos={pos}")
            print(f"    {args.arm_a.name}: {sorted(va[pos])}")
            print(f"    {args.arm_b.name}: {sorted(vb[pos])}")
    print()
    print("VERDICT: " + ("NUMERIC DIVERGENCE" if numeric else
                         "no numeric divergence on shared positions"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
