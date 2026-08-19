#!/usr/bin/env python3
"""Research-only (qwen38-r1-e55): dispose of the 40 `swift test` issues by
measurement instead of by argument.

Three arms, all on one host and toolchain:

  base_twins_no_runtime   base twins, plain `swift test`
  candidate_no_runtime    candidate twins, plain `swift test`
  candidate_with_runtime  candidate twins, MLXFAST_RUN_MLX_RUNTIME_TESTS=1

The opt-in gate needs its own positive control. Several gated tests use
`guard ... else { return }`, which PASSES TRIVIALLY when the variable is unset,
so a green run is not evidence that the gated paths ran. The repository supplies
the control: `theMetallibIsBesideTheTestBundleWhenGatedTestsAreRequested` is
`.enabled(if:)` on that variable, so it is skipped without it and runs with it.
Its outcome is what proves the gate changed state.
"""
import json
import pathlib
import re
import sys

PRIVATE = pathlib.Path(".mlxfast-private/e55")
OUT = pathlib.Path("research/e55-swift-test-disposition.json")
GATED_PROBE = "theMetallibIsBesideTheTestBundleWhenGatedTestsAreRequested"

ARMS = {
    "base_twins_no_runtime": "base-swift-test.log",
    "candidate_no_runtime": "cand-swift-test.log",
    "candidate_with_runtime": "cand-swift-test-runtime.log",
}


def parse(name: str) -> dict:
    text = (PRIVATE / name).read_text(errors="replace")
    m = re.search(
        r"Test run with (\d+) tests in (\d+) suites \w+ after [\d.]+ seconds "
        r"with (\d+) issues", text)
    if not m:
        raise SystemExit(f"no test-run summary line in {name}")
    return {
        "log": name,
        "tests": int(m.group(1)),
        "suites": int(m.group(2)),
        "issues": int(m.group(3)),
        # Match on "recorded an issue" and require the parentheses. The looser
        # "\u2718 Test <word>" also matches the run summary line, which yields a
        # phantom test named `run` and inflates the count by one.
        "distinct_failing_tests": sorted(set(re.findall(
            r"Test ([A-Za-z0-9_]+)\(\) recorded an issue", text))),
        "gated_probe_outcomes": re.findall(
            rf"Test {GATED_PROBE}\(\) (\w+)", text),
    }


def main() -> int:
    arms = {k: parse(v) for k, v in ARMS.items()}
    base = arms["base_twins_no_runtime"]["distinct_failing_tests"]
    cand = arms["candidate_no_runtime"]["distinct_failing_tests"]
    runtime = arms["candidate_with_runtime"]["distinct_failing_tests"]
    took_effect = (
        "skipped" in arms["candidate_no_runtime"]["gated_probe_outcomes"]
        and "passed" in arms["candidate_with_runtime"]["gated_probe_outcomes"])
    payload = {
        "question": (
            "Are the swift test issues pre-existing, and does enabling the "
            "opt-in MLX runtime tests add any failure?"),
        "arms": arms,
        "base_vs_candidate_failing_sets_identical": base == cand,
        "runtime_gate_adds_no_failure": base == runtime,
        "runtime_gate_provably_took_effect": took_effect,
        "positive_control": (
            f"{GATED_PROBE} is .enabled(if: MLXFAST_RUN_MLX_RUNTIME_TESTS == 1). "
            "It is skipped without the variable and passes with it, so its "
            "outcome proves the gate changed state and the metallib was loadable."),
        "limitation": (
            "One host and one toolchain. This shows the candidate twins do not "
            "change the failing set; it does not audit why those tests fail on "
            "the campaign base."),
        "no_phantom_run_entry": all(
            "run" not in rec["distinct_failing_tests"] for rec in arms.values()),
        "verdict_ok": base == cand and base == runtime and took_effect,
    }
    payload["verdict_ok"] = payload["verdict_ok"] and payload["no_phantom_run_entry"]
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("arm                       tests suites issues failing  gated probe")
    for name, rec in arms.items():
        print("  %-24s %5d %6d %6d %7d  %s"
              % (name, rec["tests"], rec["suites"], rec["issues"],
                 len(rec["distinct_failing_tests"]),
                 rec["gated_probe_outcomes"] or "absent"))
    print()
    print("  base vs candidate failing sets identical : %s"
          % payload["base_vs_candidate_failing_sets_identical"])
    print("  runtime gate adds no failure            : %s"
          % payload["runtime_gate_adds_no_failure"])
    print("  runtime gate provably took effect       : %s" % took_effect)
    print("  VERDICT                                 : %s"
          % ("OK" if payload["verdict_ok"] else "NOT ESTABLISHED"))
    return 0 if payload["verdict_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
