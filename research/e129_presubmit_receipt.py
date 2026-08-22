#!/usr/bin/env python3
"""Collect the E129 rung-0 pre-submit chain into one publishable receipt.

    usage: research/e129_presubmit_receipt.py --submit-log PATH
               --submit-score PATH --test-log PATH --worker-sha256 SHA
               --worker-mtime T --worker-sha256-post SHA --out PATH

Each step records the exact command and the observation that would have to
change for the step to fail, so a reader can tell a green step from a vacuous
one. Structure follows `research/e121_presubmit_receipt.py`.

The `swift test` gate here is the per-name decomposition from
`senpai/known-test-failures.md`, not a candidate-versus-base comparison: this
branch changes no submitted byte against the advisor head, so there is no
second tree to build.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
SUBMITTED = ("Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",)
# The tree rung 5e measured its `off` control against, and the scope base for
# the one submitted file.
SCOPE_BASE = "2127858ba770ddc06027205d8df89a8db21d80f5"
GROWTH_BASE = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
ADVISOR_HEAD = "d46eb29b178c123b6d243127039920872f158440"

# `senpai/known-test-failures.md`: the nine organizer names and their issue
# counts, and the campaign-added tests that are allowed to fail as well.
ORGANIZER_FAILURES = {
    "theCheckedInDeclarationSelectsThePinnedHead": 6,
    "startupMemoryPolicyKeepsRanked128GiBProfile": 2,
    "qwen36ConfigContractDigestMatchesTheReferenceManifest": 2,
    "theEvenMedianRuleIsTheMeanOfTheTwoCentralValues": 3,
    "theSeededCalibrationExpectationMatchesItsRecordedProvenance": 2,
    "theQwenMTPTrackIsArmedOnQwen38": 11,
    "contestantDocsCommandBlocksKeepTheDependencyGraphFrozen": 1,
    "participantDocsExposeDefaultCLIInstallDirectory": 2,
    "submissionStaticReviewPromptCoversMeasurementStructureExploitation": 11,
}
ALLOWED_CAMPAIGN_FAILURES = {"E95QmvWidthProbeTests", "E95DonationProbeTests"}


def run(*argv: str) -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def failing_issues(log: pathlib.Path) -> dict[str, int]:
    """`{failing test or suite name: issue count}` from a swift test log."""
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    found: dict[str, int] = {}
    for name, count in re.findall(
            r"\u2718 Test ([A-Za-z0-9_]+)\(\) failed after [\d.]+ seconds "
            r"with (\d+) issue", text):
        found[name] = int(count)
    for name, count in re.findall(
            r"\u2718 Suite ([A-Za-z0-9_]+) failed after [\d.]+ seconds "
            r"with (\d+) issue", text):
        if name in ALLOWED_CAMPAIGN_FAILURES:
            found[name] = int(count)
    return found


def test_totals(log: pathlib.Path) -> dict[str, int]:
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    m = re.search(r"Test run with (\d+) tests in (\d+) suites (?:failed|passed)"
                  r" after [\d.]+ seconds(?: with (\d+) issues)?", text)
    if not m:
        return {}
    return {"tests": int(m[1]), "suites": int(m[2]),
            "issues": int(m[3]) if m[3] else 0}


def cool_gates(log: pathlib.Path) -> list[dict]:
    if not log.exists():
        return []
    out: list[dict] = []
    for phase, temp, waited in re.findall(
            r"cool gate before the ([^(]+?) \(|"
            r"cool-down gate passed \(current ([\d.]+)C, target <=40C, "
            r"waited (\d+)s\)", log.read_text(errors="replace")):
        if phase:
            out.append({"phase": phase.strip()})
        elif out:
            out[-1].update({"passed_at_c": float(temp), "waited_s": int(waited)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit-log", type=pathlib.Path, required=True)
    ap.add_argument("--submit-score", type=pathlib.Path, required=True)
    ap.add_argument("--test-log", type=pathlib.Path, required=True)
    ap.add_argument("--worker-sha256", required=True)
    ap.add_argument("--worker-mtime", required=True)
    ap.add_argument("--worker-sha256-post", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    steps = []

    code, text = run("python3", "research/twin_audit.py")
    steps.append({"step": "twin_audit",
                  "command": "python3 research/twin_audit.py",
                  "exit": code, "passed": code == 0,
                  "observation": text.splitlines()[-1] if text else ""})

    code, text = run("senpai/validate-assignment-scope.sh", SCOPE_BASE, *SUBMITTED)
    steps.append({"step": "assignment_scope",
                  "command": f"senpai/validate-assignment-scope.sh {SCOPE_BASE} "
                             + " ".join(SUBMITTED),
                  "exit": code, "passed": code == 0, "observation": text})

    code, text = run("senpai/check-editable-budget.sh", GROWTH_BASE)
    steps.append({"step": "editable_budget",
                  "command": f"senpai/check-editable-budget.sh {GROWTH_BASE}",
                  "exit": code, "passed": code == 0, "observation": text})

    code, text = run("senpai/verify-ranked-score-boundary.sh")
    steps.append({"step": "ranked_score_boundary",
                  "command": "senpai/verify-ranked-score-boundary.sh",
                  "exit": code, "passed": code == 0,
                  "observation": text.splitlines()[-1] if text else ""})

    code, text = run("senpai/rebuild-and-assert-worker.sh", "--self-test")
    steps.append({"step": "harness_self_test",
                  "command": "senpai/rebuild-and-assert-worker.sh --self-test",
                  "exit": code, "passed": code == 0,
                  "observation": text.splitlines()[-1] if text else ""})

    # Rule 55: the submitted surface must be identical to the advisor head.
    _, editable = run(
        "bash", "-lc",
        "python3 -c \"import json;d=json.load(open('benchmark.json'));"
        "print(' '.join(d['editablePaths']))\"")
    code, surface = run("git", "diff", "--stat", ADVISOR_HEAD, "HEAD", "--",
                        *editable.split())
    steps.append({"step": "rule55_surface_matches_advisor_head",
                  "command": f"git diff --stat {ADVISOR_HEAD} HEAD -- "
                             "<editablePaths>",
                  "exit": 0 if not surface else 1, "passed": not surface,
                  "observation": surface or "empty"})

    worker_stable = args.worker_sha256 == args.worker_sha256_post
    steps.append({"step": "worker_unchanged_across_local_submit",
                  "command": "senpai/rebuild-and-assert-worker.sh --no-build "
                             "<witness set>, before and after the submit leg",
                  "exit": 0 if worker_stable else 1, "passed": worker_stable,
                  "observation": f"{args.worker_sha256[:16]} -> "
                                 f"{args.worker_sha256_post[:16]}"})

    observed = failing_issues(args.test_log)
    organizer = {name: observed.get(name) for name in ORGANIZER_FAILURES}
    moved = {name: {"expected": ORGANIZER_FAILURES[name],
                    "observed": organizer[name]}
             for name in ORGANIZER_FAILURES
             if organizer[name] != ORGANIZER_FAILURES[name]}
    unexpected = sorted(set(observed) - set(ORGANIZER_FAILURES)
                        - ALLOWED_CAMPAIGN_FAILURES)
    steps.append({"step": "swift_test_organizer_decomposition",
                  "command": "swift test --force-resolved-versions",
                  "exit": 0 if not moved and not unexpected else 1,
                  "passed": not moved and not unexpected,
                  "observation": f"organizer issues "
                                 f"{sum(v for v in organizer.values() if v)}, "
                                 f"moved {sorted(moved)}, "
                                 f"unexpected {unexpected}"})

    score = json.loads(args.submit_score.read_text()) \
        if args.submit_score.exists() else {}
    metrics = score.get("metrics", {})

    _, head = run("git", "rev-parse", "HEAD")
    receipt = {
        "experiment": "e129-rung0-presubmit",
        "arm": "route_b_sumtable_on_the_reverted_base",
        "harness": "local",
        "candidate_commit": head,
        "advisor_head": ADVISOR_HEAD,
        "scope_base": SCOPE_BASE,
        "submitted_paths": list(SUBMITTED),
        "worker_sha256": args.worker_sha256,
        "worker_mtime": args.worker_mtime,
        "worker_sha256_post": args.worker_sha256_post,
        "worker_unchanged_across_local_submit": worker_stable,
        "steps": steps,
        "swift_test": test_totals(args.test_log) | {
            "organizer_per_name": organizer,
            "organizer_issue_total": sum(v for v in organizer.values() if v),
            "moved_organizer_names": moved,
            "unexpected_failing_names": unexpected,
            "campaign_added_observed": sorted(
                set(observed) & ALLOWED_CAMPAIGN_FAILURES),
        },
        "local_submit": {
            "passed": score.get("passed"),
            "metrics": metrics,
            "cool_gates": cool_gates(args.submit_log),
        },
    }
    receipt["passed"] = bool(
        all(s["passed"] for s in steps)
        and metrics.get("all_tokens_matched")
        and metrics.get("residual_divergence_count") == 0
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "passed": receipt["passed"],
                      "failed_steps": [s["step"] for s in steps
                                       if not s["passed"]]}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
