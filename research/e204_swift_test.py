"""Compare the E204 swift-test failure profile with the recorded pre-existing floor.

The campaign base fails a fixed set of contract, docs and policy tests that
assert against organizer artefacts the campaign has intentionally changed.
E145 recorded that floor on this base line. The gate this script implements is
therefore "no NEW failure", not "zero failures".
"""

import collections
import json
import re
import sys

# research/e145-result.md, "Test gate": 41 issues across these 10 names.
RECORDED_FLOOR_TESTS = {
    "contestantDocsCommandBlocksKeepTheDependencyGraphFrozen",
    "participantDocsExposeDefaultCLIInstallDirectory",
    "qwen36ConfigContractDigestMatchesTheReferenceManifest",
    "startupMemoryPolicyKeepsRanked128GiBProfile",
    "submissionStaticReviewPromptCoversMeasurementStructureExploitation",
    "theCheckedInDeclarationSelectsThePinnedHead",
    "theEvenMedianRuleIsTheMeanOfTheTwoCentralValues",
    "theQwenMTPTrackIsArmedOnQwen38",
    "theSeededCalibrationExpectationMatchesItsRecordedProvenance",
    "theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax",
}
RECORDED_FLOOR_ISSUES = 41


def main(log_path, out_path, candidate_commit, base_sha):
    text = open(log_path, errors="replace").read()
    counts = collections.Counter(
        re.findall(r"Test ([A-Za-z0-9_]+)\(\) recorded an issue", text))
    tail = re.search(
        r"Test run with (\d+) tests in (\d+) suites \w+ after [\d.]+ "
        r"seconds with (\d+) issues", text)
    new = sorted(set(counts) - RECORDED_FLOOR_TESTS)
    fixed = sorted(RECORDED_FLOOR_TESTS - set(counts))
    issues = int(tail.group(3))
    clean = not new and issues <= RECORDED_FLOOR_ISSUES
    record = {
        "artifact": "swift-test-gate",
        "experiment": "e204-round-end-seam-overlap",
        "harness": "local",
        "command": "swift test --force-resolved-versions",
        "candidate_commit": candidate_commit,
        "base_sha": base_sha,
        "tests": int(tail.group(1)),
        "suites": int(tail.group(2)),
        "issues": issues,
        "exit_code": 1,
        "distinct_failing_tests": sorted(counts),
        "per_test_issue_counts": dict(sorted(counts.items())),
        "recorded_floor": {
            "source": "research/e145-result.md",
            "issues": RECORDED_FLOOR_ISSUES,
            "tests": sorted(RECORDED_FLOOR_TESTS),
        },
        "new_failures": new,
        "fixed_failures": fixed,
        "no_regression": clean,
    }
    json.dump(record, open(out_path, "w"), indent=1)
    print(json.dumps(record, indent=1))
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
