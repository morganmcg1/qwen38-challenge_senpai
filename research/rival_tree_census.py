#!/usr/bin/env python3
"""Rank board submissions and report which high scorers we can read locally.

`git fetch upstream` brings down one `upstream/submissions/<uuid>` ref per
organizer submission, so the exact submitted source of every scored rival tree
is readable offline. `research/e53-board-facts.json` supplies the score, solver
and commit for each uuid. Joining the two turns the public board into a
readable corpus of mechanisms that were measured on the ranked M5.

This script only reports. It does not check out, merge, or copy anything.

Run: python3 research/rival_tree_census.py [--top N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FACTS = HERE / "e53-board-facts.json"
OURS = "morganmcg1"

# Scored-surface roots. Anything outside these is research or campaign noise.
SCORED_PREFIXES = (
    "Sources/",
    "Vendor/",
    "benchmark.json",
    "mtp-head.manifest.json",
    "mtp-head/",
    "Package.swift",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def local_refs() -> set[str]:
    out = git("for-each-ref", "--format=%(refname)", "refs/remotes/upstream/submissions")
    return {line.rsplit("/", 1)[-1] for line in out.splitlines()}


def scored_diff(ref: str, base: str) -> tuple[int, list[str]]:
    """Churn and files this submission changed on the scored surface.

    A submission commit's first parent is the organizer main of its day, so
    `ref^..ref` is exactly what that submission proposed. A three-dot diff
    against current main would show nothing for any ancestor of main.
    """
    spec = f"{ref}^..{ref}" if base == "parent" else f"{base}..{ref}"
    try:
        out = git("diff", "--numstat", spec)
    except subprocess.CalledProcessError:
        return -1, []
    files = []
    churn = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add, dele, path = parts
        if not path.startswith(SCORED_PREFIXES):
            continue
        files.append(path)
        if add.isdigit():
            churn += int(add)
        if dele.isdigit():
            churn += int(dele)
    return churn, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--base",
        default="parent",
        help="reference to diff each submission against",
    )
    args = ap.parse_args()

    facts = json.loads(FACTS.read_text())
    tele = facts["telemetry"]
    first = next(iter(tele))
    rows = {r["submission"]: r for r in tele[first]}

    refs = local_refs()
    print(f"local upstream/submissions refs: {len(refs)}")
    print(f"board rows with telemetry:       {len(rows)}")
    joined = [r for s, r in rows.items() if s in refs]
    print(f"joinable (readable + scored):    {len(joined)}")
    print()

    ranked = sorted(joined, key=lambda r: -r["score"])
    print(f"=== top {args.top} readable scored trees, vs {args.base} ===")
    print(
        f"{'#':>3s} {'score':>18s} {'solver':22s} {'churn':>6s} "
        f"{'files':>5s}  submission"
    )
    for i, r in enumerate(ranked[: args.top], 1):
        churn, files = scored_diff(f"upstream/submissions/{r['submission']}", args.base)
        mark = " <== OURS" if r["solver"] == OURS else ""
        print(
            f"{i:3d} {r['score']:18.14f} {r['solver']:22s} {churn:6d} "
            f"{len(files):5d}  {r['submission'][:8]}{mark}"
        )
        for f in files[:6]:
            print(f"      {f}")
        if len(files) > 6:
            print(f"      ... and {len(files) - 6} more")
    print()

    ours = [r for r in joined if r["solver"] == OURS]
    print(f"=== our own readable submissions ({len(ours)}) ===")
    for r in sorted(ours, key=lambda r: -r["score"]):
        print(f"  {r['score']:18.14f}  {r['submission'][:8]}  commit {r['commit'][:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
