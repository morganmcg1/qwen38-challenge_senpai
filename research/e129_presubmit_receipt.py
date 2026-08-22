#!/usr/bin/env python3
"""Collect the E129 rung-0 pre-submit chain into one publishable receipt.

    usage: research/e129_presubmit_receipt.py --submit-log PATH
               --submit-score PATH --test-log PATH --worker-sha256 SHA
               --worker-mtime T --worker-sha256-post SHA --out PATH

Each step records the exact command and the observation that would have to
change for the step to fail, so a reader can tell a green step from a vacuous
one. Structure follows `research/e121_presubmit_receipt.py`.

The `swift test` gate here is the per-name decomposition from
`senpai/known-test-failures.md`, not a candidate-versus-base comparison. The
gate cannot be a candidate-versus-base build because the base moves under the
branch whenever the advisor merges, and a moving control is worse than a fixed
per-name census.
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
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"

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


def resolve_advisor_head() -> str:
    """The advisor head, read from the ref rather than pinned in this file.

    A pinned sha silently ages into a comparison against a tree nobody is on.
    That is what happened here: the constant predated an accepted merge, so the
    surface check reported the merged work as this branch's own change.
    """
    for ref in ("origin/" + ADVISOR_BRANCH, ADVISOR_BRANCH):
        code, sha = run("git", "rev-parse", "--verify", "--quiet", ref)
        if code == 0 and sha.strip():
            return sha.strip()
    raise SystemExit("cannot resolve the advisor branch %r" % ADVISOR_BRANCH)


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
    ap.add_argument("--timed-commit", default="",
                    help="commit whose build produced the submit log, when the "
                         "receipt is regenerated after a research-only edit")
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

    # The submitted surface, in two halves.
    #
    # This branch used to be a pure revert that changed no submitted byte, and
    # the old single check asserted exactly that. E129 changes the surface on
    # purpose, so that check now demands the experiment not exist. Splitting it
    # keeps both things it was actually protecting and drops the part that only
    # described the old arm.
    _, editable = run(
        "bash", "-lc",
        "python3 -c \"import json;d=json.load(open('benchmark.json'));"
        "print(' '.join(d['editablePaths']))\"")
    editable_paths = editable.split()
    advisor_head = resolve_advisor_head()

    # (a) We changed the files this experiment owns, and no others. This is
    # what catches a submitted byte picked up from another student's branch.
    _, touched = run("git", "diff", "--name-only", advisor_head, "HEAD", "--",
                     *editable_paths)
    touched_set = sorted(p for p in touched.split() if p)
    owns_only = touched_set == sorted(SUBMITTED)
    steps.append({"step": "surface_touches_only_owned_paths",
                  "command": f"git diff --name-only {advisor_head[:12]} HEAD "
                             "-- <editablePaths>",
                  "exit": 0 if owns_only else 1, "passed": owns_only,
                  "observation": "%s (owned: %s)"
                                 % (touched_set or "empty", sorted(SUBMITTED))})

    # (b) The advisor merged no submitted byte since we branched, so this
    # candidate drops nobody's promoted work. A non-empty diff here means the
    # base moved under us and the arm must be replayed on the new base.
    _, mergebase = run("git", "merge-base", "HEAD", advisor_head)
    mergebase = mergebase.strip()
    _, drift = run("git", "diff", "--stat", mergebase, advisor_head, "--",
                   *editable_paths)
    steps.append({"step": "advisor_surface_contained_since_branch_point",
                  "command": f"git diff --stat {mergebase[:12]} "
                             f"{advisor_head[:12]} -- <editablePaths>",
                  "exit": 0 if not drift else 1, "passed": not drift,
                  "observation": drift or
                                 f"empty; advisor added no submitted byte "
                                 f"since {mergebase[:12]}"})

    # (c) The tree that was timed and the tree being certified must carry the
    # same submitted bytes. They can be different commits: fixing a research
    # script after the timed leg is normal and must not force a rerun. What
    # must never differ is a submitted byte.
    if args.timed_commit:
        _, moved = run("git", "diff", "--stat", args.timed_commit, "HEAD", "--",
                       *editable_paths)
        steps.append({"step": "submitted_surface_unchanged_since_timed_commit",
                      "command": f"git diff --stat {args.timed_commit[:12]} "
                                 "HEAD -- <editablePaths>",
                      "exit": 0 if not moved else 1, "passed": not moved,
                      "observation": moved or
                                     f"empty; {args.timed_commit[:12]} and HEAD "
                                     "carry identical submitted bytes"})

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
        "advisor_head": advisor_head,
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
