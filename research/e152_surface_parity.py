#!/usr/bin/env python3
"""E152 F8 acceptance criterion: is our submitted surface byte-identical to a
reference commit?

    usage: python3 research/e152_surface_parity.py [--ref SHA] [--json OUT]
                                                   [--show all|differing]

WHY THIS EXISTS. F8 FINDING 273 prices the import by parity, not by local
timing: Yukon builds the candidate from the submitted archive, and the
non-editable code, the head artifact and the runner are all fixed. So a
byte-identical submitted surface plus an identical head must score what the
reference scored. That argument is only as good as the digest table under it.

WHAT IT COMPARES. Every path in `benchmark.json` `editablePaths` and
`optionalEditablePaths`, expanded to files. A listed path may be a file or a
directory; a directory is expanded on BOTH sides and the union of the two file
sets is compared, so a file that exists on one side only is a difference, not a
silent omission.

WHICH TREE. The working tree, because that is what gets committed and
submitted. A dirty file therefore shows up here rather than hiding behind a
clean commit digest. The reference side is read from git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MISSING = "-"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ref_files(ref: str, path: str) -> dict[str, str]:
    """Digest every file the reference holds under `path`."""
    out = git("ls-tree", "-r", "--format=%(objectname) %(path)", ref, "--", path)
    files: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        blob, name = line.split(" ", 1)
        files[name] = sha256_bytes(
            subprocess.run(
                ["git", "-C", str(REPO), "cat-file", "blob", blob],
                check=True,
                capture_output=True,
            ).stdout
        )
    return files


def worktree_files(path: str) -> dict[str, str]:
    target = REPO / path
    if target.is_file():
        return {path: sha256_bytes(target.read_bytes())}
    if not target.exists():
        return {}
    files: dict[str, str] = {}
    for child in sorted(target.rglob("*")):
        if child.is_file():
            files[str(child.relative_to(REPO))] = sha256_bytes(child.read_bytes())
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="0863b06ac16e26e48fc06e97444095b00feb66d4")
    ap.add_argument("--json")
    ap.add_argument("--show", choices=["all", "differing"], default="differing")
    args = ap.parse_args()

    contract = json.loads((REPO / "benchmark.json").read_text())
    listed = list(
        dict.fromkeys(contract["editablePaths"] + contract["optionalEditablePaths"])
    )

    ref_sha = git("rev-parse", "--verify", f"{args.ref}^{{commit}}").strip()

    ours: dict[str, str] = {}
    theirs: dict[str, str] = {}
    for path in listed:
        ours.update(worktree_files(path))
        theirs.update(ref_files(ref_sha, path))

    rows = []
    for name in sorted(set(ours) | set(theirs)):
        a = ours.get(name, MISSING)
        b = theirs.get(name, MISSING)
        rows.append({"path": name, "ours": a, "ref": b, "identical": a == b})

    differing = [r for r in rows if not r["identical"]]
    identical = len(rows) - len(differing)

    print(f"reference          {ref_sha}")
    print(f"listed paths       {len(listed)}")
    print(f"files compared     {len(rows)}")
    print(f"identical          {identical}")
    print(f"differing          {len(differing)}")
    print(f"e152_surface_identical_to_frontier = {str(not differing).lower()}")

    shown = rows if args.show == "all" else differing
    if shown:
        print()
        print(f"{'ours':<18} {'reference':<18} path")
        for r in shown:
            print(f"{r['ours'][:16]:<18} {r['ref'][:16]:<18} {r['path']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "reference": ref_sha,
                    "listed_paths": len(listed),
                    "files_compared": len(rows),
                    "identical": identical,
                    "differing": len(differing),
                    "e152_surface_identical_to_frontier": not differing,
                    "files": rows,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")

    return 0 if not differing else 1


if __name__ == "__main__":
    sys.exit(main())
