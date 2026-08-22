#!/usr/bin/env python3
"""Collect the E121 rung-3 pre-submit chain into one publishable receipt.

    usage: research/e121_presubmit_receipt.py --submit-log PATH
               --submit-score PATH --test-log-candidate PATH
               --test-log-base PATH --worker-sha256 SHA --worker-mtime T
               --worker-sha256-post SHA --out PATH

Each step records the exact command and the observation that would have to
change for the step to fail, so a reader can tell a green step from a vacuous
one. Structure follows `research/e110_rung3_receipt.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
PATCH = "research/e121-artifacts/e121-share.patch"
SUBMITTED = (
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
)
BASE_SHA = "2127858ba770ddc06027205d8df89a8db21d80f5"
GROWTH_BASE = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"


def run(*argv: str) -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def failing_names(log: pathlib.Path) -> list[str]:
    if not log.exists():
        return []
    text = log.read_text(errors="replace")
    return sorted(set(re.findall(r"\u2718 Test ([A-Za-z0-9_]+)\(\) failed", text)))


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
    ap.add_argument("--test-log-candidate", type=pathlib.Path, required=True)
    ap.add_argument("--test-log-base", type=pathlib.Path, required=True)
    ap.add_argument("--worker-sha256", required=True)
    ap.add_argument("--worker-mtime", required=True)
    ap.add_argument("--worker-sha256-post", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    cand = failing_names(args.test_log_candidate)
    base = failing_names(args.test_log_base)

    steps = []

    code, text = run("python3", "research/twin_audit.py")
    steps.append({"step": "twin_audit",
                  "command": "python3 research/twin_audit.py",
                  "exit": code, "passed": code == 0,
                  "observation": text.splitlines()[-1] if text else ""})

    code, text = run("senpai/validate-assignment-scope.sh", BASE_SHA, *SUBMITTED)
    steps.append({"step": "assignment_scope",
                  "command": f"senpai/validate-assignment-scope.sh {BASE_SHA} "
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

    worker_stable = args.worker_sha256 == args.worker_sha256_post
    steps.append({"step": "worker_unchanged_across_local_submit",
                  "command": "senpai/rebuild-and-assert-worker.sh --no-build "
                             "<witness set>, before and after the submit leg",
                  "exit": 0 if worker_stable else 1, "passed": worker_stable,
                  "observation": f"{args.worker_sha256[:16]} -> "
                                 f"{args.worker_sha256_post[:16]}"})

    score = json.loads(args.submit_score.read_text()) \
        if args.submit_score.exists() else {}
    metrics = score.get("metrics", {})

    _, diff = run("git", "diff", "--stat", f"{BASE_SHA}..HEAD", "--", *SUBMITTED)
    _, head = run("git", "rev-parse", "HEAD")

    receipt = {
        "experiment": "e121-rung3-presubmit",
        "arm": "share",
        "harness": "local",
        "candidate_commit": head,
        "base_sha": BASE_SHA,
        "submitted_paths": list(SUBMITTED),
        "submitted_diff_stat": diff,
        "worker_sha256": args.worker_sha256,
        "worker_mtime": args.worker_mtime,
        "worker_sha256_post": args.worker_sha256_post,
        "worker_unchanged_across_local_submit": worker_stable,
        "steps": steps,
        "swift_test": {
            "candidate": test_totals(args.test_log_candidate) | {"names": cand},
            "base": test_totals(args.test_log_base) | {"names": base},
            "added_by_candidate": sorted(set(cand) - set(base)),
            "removed_by_candidate": sorted(set(base) - set(cand)),
            "base_tree_method": f"git apply -R {PATCH}, quantized.cpp recompiled",
        },
        "local_submit": {
            "passed": score.get("passed"),
            "metrics": metrics,
            "cool_gates": cool_gates(args.submit_log),
        },
    }
    receipt["passed"] = bool(
        all(s["passed"] for s in steps)
        and not receipt["swift_test"]["added_by_candidate"]
        and metrics.get("all_tokens_matched")
        and metrics.get("residual_divergence_count") == 0
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "passed": receipt["passed"],
                      "added_test_failures":
                          receipt["swift_test"]["added_by_candidate"]}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
