#!/usr/bin/env python3
"""Prove that the E149 branch changes no submitted editable path.

Compares the working tree against the assignment base and intersects the
changed-file set with `editablePaths` plus `optionalEditablePaths` from
benchmark.json. Exits non-zero if the intersection is non-empty.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: e149_editable_surface_diff.py BASE_SHA", file=sys.stderr)
        return 2
    base = sys.argv[1]

    spec = json.loads((ROOT / "benchmark.json").read_text())
    editable = list(spec["editablePaths"]) + list(spec.get("optionalEditablePaths", []))

    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.split()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.split()
    changed = sorted(set(changed) | set(untracked))

    def in_surface(path: str) -> str | None:
        for entry in editable:
            e = entry.rstrip("/")
            if path == e or path.startswith(e + "/"):
                return entry
        return None

    hits = [(p, in_surface(p)) for p in changed]
    offenders = [(p, e) for p, e in hits if e is not None]

    print(f"base {base}")
    print(f"editable entries: {len(editable)}")
    print(f"changed or untracked files: {len(changed)}")
    for p, e in hits:
        mark = f"IN SURFACE via {e}" if e else "outside surface"
        print(f"  {p}  ->  {mark}")
    print(f"submitted-surface changes: {len(offenders)}")
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
