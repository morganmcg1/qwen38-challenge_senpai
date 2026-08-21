#!/usr/bin/env python3
"""E101 row-digest check for the composed arm C + selected-rerank tree.

`mtp-row:` is the exact per-row evidence the trusted parent checks: the target
row position, its top-two token ids, and their top-two values as hex float
literals, so the digest is over exact bits and not a rounded print. Comparing
the fused arm against the `MLX_E101_ROW_TOP32=0` legacy arm inside one binary
is an arm-relative claim, so it needs no assumption about the reference source
or about how the parent cuts the decode window.

    usage: research/e101_row_digest.py TAG_A TAG_B [TAG_A TAG_B ...]

Exits non-zero when any pair disagrees.
"""

import hashlib
import json
import pathlib
import sys


def rows(tag):
    path = pathlib.Path("research/out") / tag / "trace.txt"
    return [line.strip() for line in path.read_text().splitlines()
            if line.startswith("mtp-row:")]


def digest(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main(argv):
    if len(argv) < 2 or len(argv) % 2:
        sys.exit(__doc__)
    report, failures = [], 0
    for a, b in zip(argv[0::2], argv[1::2]):
        ra, rb = rows(a), rows(b)
        da, db = digest(ra), digest(rb)
        first_bad = next(
            (i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), -1)
        ok = da == db and len(ra) == len(rb) and ra
        failures += not ok
        report.append({
            "a": a, "b": b, "rows_a": len(ra), "rows_b": len(rb),
            "sha256_a": da, "sha256_b": db,
            "first_mismatch": first_bad, "identical": bool(ok),
        })
        print(f"{a} vs {b}: rows {len(ra)}/{len(rb)} "
              f"sha256 {da[:16]}/{db[:16]} "
              f"first_mismatch {first_bad} "
              f"{'IDENTICAL' if ok else 'DIFFERENT'}")
    out = pathlib.Path("research/out/e101-composed/row-digest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"e101_row_digest: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
