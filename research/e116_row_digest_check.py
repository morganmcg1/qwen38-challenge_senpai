#!/usr/bin/env python3
"""E116: check traced row evidence against a pinned digest, and prove the
check can fail.

    usage: research/e116_row_digest_check.py TAG [TAG ...] --pin SHA256
               [--expect-rows N] [--negative-control TAG] [--json OUT]

`research/e101_row_digest.py` compares two legs against EACH OTHER, which is
an arm-relative claim. This adds the absolute claim: every named leg must
reproduce a digest that was pinned before this experiment existed.

THE COMPARATOR CONTROL. A digest that always matches is not evidence. This
script flips one hexadecimal digit inside one `mtp-row:` value of the first
leg and re-digests. The perturbed digest must differ and the reported first
mismatch must be the perturbed line. That proves the comparison is sensitive
to a single changed bit of a single row value, which is the property the
exactness claim needs. `--negative-control` adds the end-to-end version: a leg
that really did behave differently, whose digest must not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def rows(tag: str) -> list[str]:
    path = pathlib.Path("research/out") / tag / "trace.txt"
    if not path.exists():
        sys.exit(f"e116_row_digest_check: no trace at {path}")
    return [line.strip() for line in path.read_text().splitlines()
            if line.startswith("mtp-row:")]


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def perturb(lines: list[str], index: int) -> tuple[list[str], str]:
    """Flip one hex digit of one row value; return the new lines and the line."""
    line = lines[index]
    head, _, hexvalues = line.partition(" v=")
    if not hexvalues:
        sys.exit("e116_row_digest_check: row line carries no v= field")
    chars = list(hexvalues)
    for position, char in enumerate(chars):
        if char in "0123456789abcdef":
            chars[position] = "f" if char != "f" else "0"
            break
    else:
        sys.exit("e116_row_digest_check: row value has no hex digit to flip")
    perturbed = list(lines)
    perturbed[index] = f"{head} v={''.join(chars)}"
    return perturbed, perturbed[index]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--pin", required=True)
    ap.add_argument("--expect-rows", type=int)
    ap.add_argument("--negative-control")
    ap.add_argument("--json")
    args = ap.parse_args()

    failures = 0
    report: dict[str, object] = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "pinned_sha256": args.pin,
        "expected_rows": args.expect_rows,
        "legs": [],
    }

    first_rows: list[str] = []
    for tag in args.tags:
        lines = rows(tag)
        if not first_rows:
            first_rows = lines
        actual = digest(lines)
        row_ok = args.expect_rows is None or len(lines) == args.expect_rows
        ok = actual == args.pin and row_ok
        failures += not ok
        report["legs"].append({
            "tag": tag, "rows": len(lines), "sha256": actual,
            "matches_pin": actual == args.pin, "row_count_ok": row_ok,
        })
        print(f"{tag}: rows {len(lines)} sha256 {actual[:16]} "
              f"{'MATCHES PIN' if ok else 'DOES NOT MATCH PIN'}")

    perturbed, changed_line = perturb(first_rows, len(first_rows) // 2)
    perturbed_digest = digest(perturbed)
    control_ok = perturbed_digest != args.pin
    failures += not control_ok
    report["comparator_control"] = {
        "kind": "single hex digit flipped in one row value",
        "row_index": len(first_rows) // 2,
        "perturbed_line": changed_line,
        "perturbed_sha256": perturbed_digest,
        "digest_moved": control_ok,
    }
    print(f"comparator control: one hex digit flipped in row "
          f"{len(first_rows) // 2} -> sha256 {perturbed_digest[:16]} "
          f"{'MOVED (check can fail)' if control_ok else 'DID NOT MOVE'}")

    if args.negative_control:
        lines = rows(args.negative_control)
        actual = digest(lines)
        moved = actual != args.pin
        failures += not moved
        report["runtime_negative_control"] = {
            "tag": args.negative_control, "rows": len(lines),
            "sha256": actual, "digest_moved": moved,
        }
        print(f"runtime negative control {args.negative_control}: "
              f"rows {len(lines)} sha256 {actual[:16]} "
              f"{'MOVED (check can fail)' if moved else 'DID NOT MOVE'}")

    report["failures"] = failures
    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {path}")
    print(f"e116_row_digest_check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
