#!/usr/bin/env python3
"""Materialise one E84 arm of `Qwen35.swift` into the worktree.

E84 removes two independent pieces of dead work in one file, so the four arms
of the design are subsets of one diff:

  base  the assignment base, byte for byte
  a     mechanism A only, the precision-island K/V removal
  b     mechanism B only, the state-only Gated DeltaNet prefix replay
  ab    both, which must reproduce the branch tip byte for byte

The two mechanisms touch disjoint regions of the file, so each hunk of
`git diff BASE TIP` belongs to exactly one of them. Classification is by
content, not by line number, so a later edit that moves a region cannot
silently mislabel an arm.

  research/e84_arm.py ab --out /dev/stdout

The script refuses to write an arm it cannot prove: `base` must reproduce the
base blob and `ab` must reproduce the tip blob, both by sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
FILE = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"

# Markers that put a hunk on one mechanism. Checked against the hunk body.
B_MARKERS = (
    "qwen35GatedDeltaReplayState",
    "boundarySsm",
)

ARMS = {
    "base": frozenset(),
    "a": frozenset({"a"}),
    "b": frozenset({"b"}),
    "ab": frozenset({"a", "b"}),
}


def git(*args, **kw):
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                         text=True, check=True, **kw)
    return out.stdout


def classify(hunk: str) -> str:
    if any(m in hunk for m in B_MARKERS):
        return "b"
    return "a"


def split_hunks(patch: str):
    lines = patch.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("@@")]
    if not starts:
        raise SystemExit("e84_arm: the diff has no hunks")
    header = "".join(lines[: starts[0]])
    bounds = starts + [len(lines)]
    hunks = ["".join(lines[bounds[i]: bounds[i + 1]])
             for i in range(len(starts))]
    return header, hunks


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--base", default="5ea174c50b98407bc463c463cc7c7a85d32960a7")
    ap.add_argument("--tip", default="HEAD")
    ap.add_argument("--out", default=str(REPO / FILE))
    ap.add_argument("--print-plan", action="store_true")
    args = ap.parse_args()

    base_text = git("show", "%s:%s" % (args.base, FILE))
    tip_text = git("show", "%s:%s" % (args.tip, FILE))
    patch = git("diff", "-U6", args.base, args.tip, "--", FILE)
    header, hunks = split_hunks(patch)

    wanted = ARMS[args.arm]
    plan = [(classify(h), h) for h in hunks]
    if args.print_plan:
        for i, (kind, hunk) in enumerate(plan, 1):
            head = re.match(r"@@[^@]*@@", hunk).group(0)
            print("%2d %-6s %s" % (i, kind, head))
        return 0

    selected = [h for kind, h in plan if kind in wanted]
    if args.arm != "base" and not selected:
        raise SystemExit("e84_arm: arm %s selected no hunks" % args.arm)

    text = base_text
    if selected:
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp) / "work"
            work.mkdir()
            target = work / FILE
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(base_text)
            patch_path = pathlib.Path(tmp) / "arm.patch"
            patch_path.write_text(header + "".join(selected))
            subprocess.run(
                ["git", "apply", "--unsafe-paths", "--directory", str(work),
                 "-p1", str(patch_path)],
                cwd=work, check=True)
            text = target.read_text()

    expect = {"base": base_text, "ab": tip_text}.get(args.arm)
    if expect is not None and sha256(text) != sha256(expect):
        raise SystemExit(
            "e84_arm: arm %s did not reproduce its reference blob "
            "(%s != %s)" % (args.arm, sha256(text)[:16], sha256(expect)[:16]))

    out = pathlib.Path(args.out)
    out.write_text(text)
    print("e84_arm: arm=%s hunks=%d/%d sha256=%s -> %s"
          % (args.arm, len(selected), len(hunks), sha256(text), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
