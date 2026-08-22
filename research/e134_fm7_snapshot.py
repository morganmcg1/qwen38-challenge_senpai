#!/usr/bin/env python3
"""E134 FM7 -- assert the PACKAGED snapshot ships the pb6 arm, not the worktree.

    usage: research/e134_fm7_snapshot.py [--ref HEAD] [--control REF] [--json PATH]

Run from the repository ROOT. Zero GPU.

FM7 is the failure this check exists to prevent: a submission cut without the
arm flip measures the null and burns a validation slot. Reading the working
tree cannot see that failure, because Yukon packages the COMMITTED tree. So
every assertion here reads blobs out of `git cat-file`, never the checkout.

A gate that cannot fail is worse than no gate, so the same assertions run
against a control ref that predates the flip and are REQUIRED to fail there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

SESSION_FILE = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

ARM_DEFAULT = re.compile(
    r"internal static let depthPriceArm: DepthPriceArm = \{.*?"
    r"DepthPriceArm\(rawValue: requested\) \?\? \.(\w+)", re.S)
ARM_PLAIN = re.compile(
    r"internal static let depthPriceArm: DepthPriceArm = \.(\w+)")
WIDTH = re.compile(r"let passBoundaryVerifyWidth\s*(?::\s*Int)?\s*=\s*(\d+)")
TIER = re.compile(
    r"let passBoundaryTierFactor\s*(?::\s*Double)?\s*=\s*([\d.]+)")


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, check=True, capture_output=True,
                          text=True).stdout


def tracked(ref: str) -> list[str]:
    return git("ls-tree", "-r", "--name-only", ref).splitlines()


def blob(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")


def shipped_arm(source: str) -> str | None:
    match = ARM_DEFAULT.search(source) or ARM_PLAIN.search(source)
    return match.group(1) if match else None


def assertions(ref: str) -> dict:
    """Every value the packaged archive must carry, read out of the ref."""
    source = blob(ref, SESSION_FILE)
    width = WIDTH.search(source)
    tier = TIER.search(source)
    return {
        "ref": git("rev-parse", ref).strip(),
        "arm_when_env_unset": shipped_arm(source),
        "pass_boundary_verify_width": int(width.group(1)) if width else None,
        "pass_boundary_tier_factor": float(tier.group(1)) if tier else None,
        "session_bytes": len(source.encode()),
    }


def verdict(found: dict) -> tuple[bool, list[str]]:
    want = {
        "arm_when_env_unset": "pb6",
        "pass_boundary_verify_width": 6,
        "pass_boundary_tier_factor": 1.45,
    }
    bad = [f"{key}: want {value!r}, packaged {found[key]!r}"
           for key, value in want.items() if found[key] != value]
    return not bad, bad


def coverage(ref: str, required: list[str], optional: set[str]) -> dict:
    """Which editablePaths the archive would carry out of this ref."""
    files = tracked(ref)
    rows, missing, skipped = [], [], []
    for path in required:
        under = [f for f in files if f == path or f.startswith(path + "/")]
        if not under:
            (skipped if path in optional else missing).append(path)
            continue
        size = sum(int(git("cat-file", "-s", f"{ref}:{f}")) for f in under)
        rows.append({"path": path, "files": len(under), "bytes": size})
    return {"paths": rows, "missing_required": missing,
            "absent_optional": skipped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--control", default="89ea923c",
                    help="a ref that predates the arm flip; the same "
                         "assertions must FAIL there")
    ap.add_argument("--json")
    args = ap.parse_args()

    contract = json.loads(pathlib.Path("benchmark.json").read_text())
    required = contract["editablePaths"]
    optional = set(contract["optionalEditablePaths"])

    found = assertions(args.ref)
    ok, bad = verdict(found)
    control = assertions(args.control)
    control_ok, _ = verdict(control)

    print("E134 FM7 -- packaged snapshot assertions, read from git blobs")
    print()
    print("packaged ref     %s" % found["ref"])
    print("arm when unset   %s" % found["arm_when_env_unset"])
    print("verify width     %s" % found["pass_boundary_verify_width"])
    print("tier factor      %s" % found["pass_boundary_tier_factor"])
    print("session bytes    %d" % found["session_bytes"])
    print()

    # The worktree may hold uncommitted research files. It must not hold an
    # uncommitted difference in the file the archive carries.
    dirty = git("status", "--porcelain", "--", SESSION_FILE).strip()
    print("uncommitted change in the packaged session file: %s"
          % (dirty or "none"))

    cover = coverage(args.ref, required, optional)
    print("editablePaths required %d, carried %d, absent optional %d, "
          "missing required %d"
          % (len(required), len(cover["paths"]), len(cover["absent_optional"]),
             len(cover["missing_required"])))
    for path in cover["missing_required"]:
        print("  MISSING REQUIRED %s" % path)
    for path in cover["absent_optional"]:
        print("  absent optional  %s (pinned head applies)" % path)
    print()

    print("negative control %s" % control["ref"])
    print("  arm when unset %s -> assertions pass: %s"
          % (control["arm_when_env_unset"], control_ok))
    control_proves_it_can_fail = not control_ok
    print("  the gate can fail: %s" % control_proves_it_can_fail)
    print()

    passed = (ok and not dirty and not cover["missing_required"]
              and control_proves_it_can_fail)
    for line in bad:
        print("FAIL %s" % line)
    print("FM7: %s" % ("PASS" if passed else "FAIL"))

    digest = hashlib.sha256(
        json.dumps({"found": found, "coverage": cover}, sort_keys=True)
        .encode()).hexdigest()

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "harness": "static",
            "packaged": found,
            "packaged_assertion_failures": bad,
            "uncommitted_change_in_session_file": bool(dirty),
            "coverage": cover,
            "control": control,
            "control_assertions_pass": control_ok,
            "gate_can_fail": control_proves_it_can_fail,
            "passed": passed,
            "snapshot_digest": digest,
        }, indent=2, sort_keys=True) + "\n")
        print("wrote %s" % out)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
