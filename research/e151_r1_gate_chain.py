#!/usr/bin/env python3
"""E151: run the cheap gates, collect the expensive ones, emit one verdict.

Written for R1 and generalized to any rung by `--rung` and `--label`. The
filename keeps its R1 name because the R1 record cites it.

The cheap gates below run here. The expensive gates (the arm-on `mlx.metallib`
build, the worker rebuild-and-assert, `swift test`, the 512-token traced legs
and `--local-submit`) already ran under `run_job`; this script reads their
persisted artifacts instead of paying for them twice, and records the digest
that ties each artifact to the commit that produced it.

harness=offline for every static gate. harness=local for the artifact-backed
runtime gates. Nothing here is an effect estimate: this host reports
`is_nax_available() == false`, so the retiled kernel never executes locally.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

# The campaign base an experiment must beat. It moves when the advisor branch
# moves, so `--base-sha` overrides it rather than the constant being edited.
BASE_SHA = "14247cce11216639a04ecfc2798cf0798091ff92"
GROWTH_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
CANDIDATE_PATHS = [
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
]
GROWTH_LIMIT = 262144

# The `swift test` floor recorded on this branch before any E151 edit, at base
# de8ce44c: 41 issues over 10 distinct failing tests, 780 tests, 74 suites.
SWIFT_TEST_FLOOR_ISSUES_DE8CE44C = 41
SWIFT_TEST_FLOOR_NAMES_DE8CE44C = 10

# Base 14247cce adds Tests/MLXFastTests/E145WidthPinTests.swift, which does not
# exist at de8ce44c. Its single assertion pins `depthPriceArm == .pb6`, and the
# campaign retired the pb6 arm, so the shipped default is `.ship` and the pin is
# stale. That one stale pin is the whole 41 -> 42 move. E151 cannot have caused
# it: this branch changes zero files under Sources/ and Tests/.
SWIFT_TEST_FLOOR_ISSUES = 42
SWIFT_TEST_FLOOR_NAMES = 11


def run(*args: str) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def load_json(path: str):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def static_gates(base_sha: str) -> dict:
    gates: dict = {}

    code, out = run("python3", "research/twin_audit.py")
    gates["twin_audit"] = {"exit": code, "tail": out.strip().splitlines()[-3:]}

    code, out = run(
        "senpai/validate-assignment-scope.sh", base_sha, *CANDIDATE_PATHS
    )
    gates["validate_assignment_scope"] = {
        "exit": code,
        "base": base_sha,
        "paths": CANDIDATE_PATHS,
        "tail": out.strip().splitlines()[-3:],
    }

    code, out = run("senpai/check-editable-budget.sh", GROWTH_BASE_SHA)
    growth = re.search(r"growth[^0-9-]*(-?\d+)", out)
    gates["check_editable_budget_enforced"] = {
        "exit": code,
        "base": GROWTH_BASE_SHA,
        "growth_bytes": int(growth.group(1)) if growth else None,
        "limit_bytes": GROWTH_LIMIT,
    }

    code, out = run("senpai/check-editable-budget.sh", base_sha)
    growth = re.search(r"growth[^0-9-]*(-?\d+)", out)
    gates["check_editable_budget_attributable"] = {
        "exit": code,
        "base": base_sha,
        "growth_bytes": int(growth.group(1)) if growth else None,
    }

    code, out = run("senpai/verify-ranked-score-boundary.sh")
    gates["verify_ranked_score_boundary"] = {
        "exit": code,
        "tail": out.strip().splitlines()[-3:],
    }

    return gates


def swift_test_gate(log_path: str | None) -> dict:
    if not log_path or not pathlib.Path(log_path).is_file():
        return {"available": False}
    text = pathlib.Path(log_path).read_text()
    # Swift Testing prints an identifier name bare and a `@Test("display name")`
    # in quotes. Matching only the identifier form undercounts a display-named
    # failure and silently reports a floor the run did not hit.
    names = sorted(set(
        re.findall(r"Test ([A-Za-z0-9_]+)\(\) recorded an issue", text)
        + re.findall(r'Test "([^"\n]+)" recorded an issue', text)
    ))
    issues = len(re.findall(r"recorded an issue", text))
    suite = re.search(r"Test run with (\d+) tests? in (\d+) suites?", text)
    return {
        "available": True,
        "log": log_path,
        "issue_count": issues,
        "failing_name_count": len(names),
        "failing_names": names,
        "tests": int(suite.group(1)) if suite else None,
        "suites": int(suite.group(2)) if suite else None,
        "floor_issue_count": SWIFT_TEST_FLOOR_ISSUES,
        "floor_failing_name_count": SWIFT_TEST_FLOOR_NAMES,
        "floor_issue_count_de8ce44c": SWIFT_TEST_FLOOR_ISSUES_DE8CE44C,
        "floor_failing_name_count_de8ce44c": SWIFT_TEST_FLOOR_NAMES_DE8CE44C,
        "floor_moved_by_base_not_by_experiment": True,
        "floor_move_cause": "Tests/MLXFastTests/E145WidthPinTests.swift:65 "
                            "pins depthPriceArm == .pb6; the campaign retired "
                            "the pb6 arm so the shipped default is .ship. The "
                            "file does not exist at de8ce44c.",
        "at_known_floor": issues == SWIFT_TEST_FLOOR_ISSUES
        and len(names) == SWIFT_TEST_FLOOR_NAMES,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swift-test-log")
    ap.add_argument("--rung", default="R1")
    ap.add_argument("--label", default="", help="artifact suffix, e.g. r2")
    ap.add_argument(
        "--base-control-digest",
        help="row digest of the matched arm-off base control leg. When given, "
        "the exactness verdict is equality with it rather than with the stale "
        "E121 pin.",
    )
    ap.add_argument("--base-sha", default=BASE_SHA)
    ap.add_argument("--out")
    args = ap.parse_args()
    rung = args.rung.lower()
    suffix = f"-{args.label}" if args.label else ""
    out_path = args.out or f"research/e151-{rung}-gate-chain.json"

    doc: dict = {
        "experiment": "E151",
        "rung": args.rung,
        "harness": "offline+local",
        "commit": run("git", "rev-parse", "HEAD")[1].strip(),
        "worktree_clean": run("git", "status", "--porcelain")[1].strip() == "",
        "base_sha": args.base_sha,
        "growth_enforced_base": GROWTH_BASE_SHA,
    }

    doc["static_gates"] = static_gates(args.base_sha)
    doc["swift_test"] = swift_test_gate(args.swift_test_log)
    doc["row_digest_512"] = load_json(
        f"research/e151-artifacts/row-digest-512{suffix}.json"
    )
    doc["local_submit_512"] = load_json(
        f"research/e151-artifacts/local-submit-512{suffix}.json"
    )
    doc["arm_attribution"] = load_json("research/e151-arm-attribution.json")

    static_ok = all(g["exit"] == 0 for g in doc["static_gates"].values())
    attrib = doc["arm_attribution"] or {}
    submit = (doc["local_submit_512"] or {}).get("metrics", {})

    # The 512-token exactness verdict is the matched same-host base control,
    # not the E121 pin. The pin was recorded on E89 base f18400c4 over 78 trace
    # rounds; base de8ce44c now emits 1024 rows over 82. `arm_attribution`
    # shows the arm-off base and two independent arm-on rebuilds all emit the
    # same digest, so the drift is the base's and the arm is output neutral.
    exact_ok = bool(
        attrib.get("e151_arm_is_local_output_neutral")
        and attrib.get("e151_leg_is_deterministic")
        and attrib.get("e151_attribution_complete")
    )
    # A later rung inherits R1's attribution of the base drift, but it must
    # still land its own leg on the same matched base-control digest. Without
    # this the verdict would carry R1's evidence for a tree R1 never built.
    leg_matches_base_control = None
    if args.base_control_digest:
        legs = (doc["row_digest_512"] or {}).get("legs") or []
        leg_matches_base_control = bool(legs) and all(
            leg.get("sha256") == args.base_control_digest for leg in legs
        )
        doc["base_control_digest"] = args.base_control_digest
        doc["leg_matches_base_control"] = leg_matches_base_control
        exact_ok = exact_ok and leg_matches_base_control

    metrics = {
        "e151_static_gates_all_green": float(static_ok),
        "e151_growth_enforced": float(
            doc["static_gates"]["check_editable_budget_enforced"]["growth_bytes"] or 0
        ),
        "e151_growth_enforced_limit": float(GROWTH_LIMIT),
        "e151_growth_attributable": float(
            doc["static_gates"]["check_editable_budget_attributable"]["growth_bytes"]
            or 0
        ),
        "e151_arm_is_local_output_neutral": float(
            bool(attrib.get("e151_arm_is_local_output_neutral"))
        ),
        "e151_leg_is_deterministic": float(
            bool(attrib.get("e151_leg_is_deterministic"))
        ),
        "e151_base_also_misses_pin": float(
            bool(attrib.get("e151_base_also_misses_pin"))
        ),
        "e151_exact512_verdict_ok": float(exact_ok),
        "e151_local_submit_passed": float(
            bool((doc["local_submit_512"] or {}).get("passed"))
        ),
        "e151_local_submit_all_tokens_matched": float(
            bool(submit.get("all_tokens_matched"))
        ),
        "e151_local_submit_decode_tokens": float(submit.get("decode_tokens") or 0),
        "e151_local_submit_residual_divergence_count": float(
            submit.get("residual_divergence_count") or 0
        ),
        "e151_local_submit_mtp_decode_speedup": float(
            submit.get("mtp_decode_speedup") or 0.0
        ),
    }
    if doc["swift_test"]["available"]:
        metrics["e151_swift_test_issue_count"] = float(
            doc["swift_test"]["issue_count"]
        )
        metrics["e151_swift_test_failing_name_count"] = float(
            doc["swift_test"]["failing_name_count"]
        )
        metrics["e151_swift_test_at_known_floor"] = float(
            doc["swift_test"]["at_known_floor"]
        )

    all_green = (
        static_ok
        and exact_ok
        and bool((doc["local_submit_512"] or {}).get("passed"))
        and bool(submit.get("all_tokens_matched"))
        and doc["swift_test"].get("at_known_floor", False)
        and doc["worktree_clean"]
    )
    metrics["e151_gate_chain_all_green"] = float(all_green)
    metrics[f"e151_{rung}_submission_ready"] = float(all_green)
    if leg_matches_base_control is not None:
        metrics[f"e151_{rung}_leg_matches_base_control"] = float(
            leg_matches_base_control
        )

    doc["metrics"] = metrics
    pathlib.Path(out_path).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps(metrics, indent=1, sort_keys=True))
    print(f"wrote {out_path}")
    raise SystemExit(0 if all_green else 1)


if __name__ == "__main__":
    main()
