#!/usr/bin/env python3
"""Research-only (qwen38-r1-e215): reduce the two suite logs to one verdict.

usage: research/e215_suite_report.py research/out/e215-suite

Writes `research/e215-artifacts/pin-test.json`: the issue count of each run,
the failing-name diff between them, and the pin tests the branch run added.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ARTIFACTS = pathlib.Path("research/e215-artifacts")
PIN_SUITE = "E215 shipped depth-price table pin"
PINNED_THRESHOLDS_HEX = [
    "0x1.eb65abab49018p-4",
    "0x1.3cd94a284f571p-3",
    "0x1.0d1de16a67c83p-1",
    "0x1.4d3a8b9cf3e83p-1",
    "0x1.a836c532de064p-1",
    "0x1.a836c532de064p-1",
    "0x1.a836c532de064p-1",
    "0x1.a836c532de064p-1",
]

RUN_SUMMARY = re.compile(
    r"Test run with (\d+) tests (passed|failed) after [\d.]+ seconds(?: with (\d+) issues?)?"
)
FAILED_TEST = re.compile(r'Test (?:case )?"?([^"\n]+?)"? failed after')
XCTEST_FAILURE = re.compile(r"Test Case '(.+?)' failed")
PASSED_TEST = re.compile(r'Test "?([^"\n]+?)"? passed after')


def parse(path: pathlib.Path) -> dict:
    text = path.read_text(errors="replace")
    tests, issues = None, 0
    for match in RUN_SUMMARY.finditer(text):
        tests = int(match.group(1))
        issues = int(match.group(3) or 0)
    failing = set()
    for match in FAILED_TEST.finditer(text):
        failing.add(match.group(1).strip())
    for match in XCTEST_FAILURE.finditer(text):
        failing.add(match.group(1).strip())
    passing = {match.group(1).strip() for match in PASSED_TEST.finditer(text)}
    return {
        "tests": tests,
        "issues": issues,
        "failing_names": sorted(failing),
        "passing_names": passing,
    }


def main() -> int:
    out_dir = pathlib.Path(sys.argv[1])
    branch = parse(out_dir / "branch.log")
    base = parse(out_dir / "base.log")

    branch_failing = set(branch["failing_names"])
    base_failing = set(base["failing_names"])
    added = sorted(branch_failing - base_failing)
    removed = sorted(base_failing - branch_failing)

    pin_tests = sorted(
        name
        for name in branch["passing_names"] | branch_failing
        if "depth-price" in name
        or "stepq" in name
        or "shipped table" in name
        or "one ulp" in name
    )
    pin_failing = sorted(name for name in pin_tests if name in branch_failing)

    payload = {
        "harness": "local",
        "suite": PIN_SUITE,
        "suite_tests_branch": branch["tests"],
        "suite_tests_base": base["tests"],
        "suite_issues_branch": branch["issues"],
        "suite_issues_base": base["issues"],
        "issue_count_matches_base": branch["issues"] == base["issues"],
        "failing_names_added": added,
        "failing_names_removed": removed,
        "failing_name_diff_empty": not added and not removed,
        "pin_test_names": pin_tests,
        "pin_test_count": len(pin_tests),
        "pin_tests_passed": bool(pin_tests) and not pin_failing,
        "pin_tests_failing": pin_failing,
        "pinned_arm": "stepq",
        "pinned_thresholds_hex": PINNED_THRESHOLDS_HEX,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "pin-test.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "pinned_thresholds_hex"},
                     indent=1, sort_keys=True))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
