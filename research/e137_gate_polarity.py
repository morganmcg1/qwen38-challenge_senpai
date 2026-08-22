#!/usr/bin/env python3
"""Both polarities of `senpai/entry-point-cliff-census.sh`, on a real commit.

Rule 101: a gate whose failure has never been observed is not a gate. The
census passed for every submission after E129 landed, and it passed because it
could not read the Route B QMV surface at all. This script shows the repaired
gate saying PASS on an unchanged base and FAIL on two perturbations of that
same base:

  blind       the Swift QMV source generator carries a signature the reader
              does not know. This is the E129 defect itself. Before the repair
              the census answered PASS with zero Route B entry points censused;
              a gate that cannot see its subject must fail.
  residency   the compiled-default width table moves to `.onePass678`, so M=8
              leaves the ipg-4 entry point for the ipg-8 one. M=8 carries 240
              of 308 rounds in the gate histogram, so this is the exact shape
              of the E121 loss the gate exists to stop.

Both perturbations are applied to the Swift TEXT inside this process through
`gate.side_sources(swift_patch=...)`. No tracked file is written, so this runs
against a file another student owns without touching it.

Registers, spill bytes and text sizes are measurements. Every simdgroup count
is DERIVED from registers under Rule 89.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e131_cliff_gate as gate  # noqa: E402
import e131_kernel_sources as ks  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def blind_patch(text: str) -> str:
    """Give the QMV source generator a signature the reader does not know."""
    old = "func qwen35E120QMVSource(table: Bool, tier: Int?) -> String {"
    if old not in text:
        raise SystemExit("blind_patch: the pinned signature is not in Qwen35.swift")
    return text.replace(
        old, "func qwen35E120QMVSource(table: Bool, tier: Int?, lane: Int) "
             "-> String {")


def residency_patch(text: str) -> str:
    """Move the compiled-default width table to the one-pass M=8 plan."""
    old = "public static let compiledDefault = Table.onePass67\n"
    if old not in text:
        raise SystemExit("residency_patch: the compiled default table moved")
    return text.replace(
        old, "public static let compiledDefault = Table.onePass678\n")


PERTURBATIONS = {"blind": blind_patch, "residency": residency_patch}


def run(base: str, patch, tag: str) -> dict:
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        base_rows = gate.census(gate.side_sources(base), workdir, "%s_b" % tag)
        cand_rows = gate.census(
            gate.side_sources(base, swift_patch=patch), workdir, "%s_c" % tag)
    return gate.evaluate(base_rows, cand_rows, base,
                         "%s + %s perturbation" % (base, tag),
                         gate.git_sha(base), None, started)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    arms = {}
    unchanged = run(args.base, lambda text: text, "unchanged")
    arms["unchanged"] = unchanged
    print("=" * 78)
    print("ARM unchanged: the base against itself")
    print("=" * 78)
    gate.report(unchanged)

    for name, patch in PERTURBATIONS.items():
        receipt = run(args.base, patch, name)
        arms[name] = receipt
        print()
        print("=" * 78)
        print("ARM %s" % name)
        print("=" * 78)
        gate.report(receipt)

    expected = {"unchanged": "pass", "blind": "fail", "residency": "fail"}
    got = {name: receipt["verdict"] for name, receipt in arms.items()}
    ok = got == expected
    summary = {
        "tool": "research/e137_gate_polarity.py",
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "occupancy_label": "derived",
        "base_ref": args.base,
        "base_sha": gate.git_sha(args.base),
        "expected_verdicts": expected,
        "verdicts": got,
        "failing_polarity_demonstrated": ok,
        "arms": arms,
    }
    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 78)
    for name in ("unchanged", "blind", "residency"):
        print("%-12s expected %-5s got %s" % (name, expected[name], got[name]))
    print("failing polarity demonstrated: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
