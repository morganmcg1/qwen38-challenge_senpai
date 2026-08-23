#!/usr/bin/env python3
"""Port an edited readable Metal header into its runtime-effective JIT twin.

``mlx-generated/*.cpp`` is the source string MLX JIT-compiles at runtime, so an
edit that lands only in the readable header changes nothing the GPU executes.
The canonical repair is to regenerate the twin, but ``research/twin_audit.py``
carries a pinned comment-stream waiver for the ``quantized`` stem, and a full
regeneration would silently resolve that recorded divergence. This script
instead transplants the exact hunks: it diffs the header against a git
reference, then applies each ``old -> new`` block to the twin verbatim.

A hunk may legitimately repeat when several instantiations share one body, so
the rule is that a hunk must occur in the twin exactly as many times as it
occurs in the header reference. A drifted twin therefore still fails loudly
instead of being partially patched.

Usage:
    research/e147_port_twin.py <header-path> <twin-path> <git-ref>
"""

import pathlib
import subprocess
import sys


def git_show(ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def hunks(before: str, after: str, context: int = 3):
    """Yield (old, new) block pairs with enough context to be unique."""
    import difflib

    a = before.splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for group in matcher.get_grouped_opcodes(context):
        i1 = group[0][1]
        i2 = group[-1][2]
        j1 = group[0][3]
        j2 = group[-1][4]
        yield "".join(a[i1:i2]), "".join(b[j1:j2])


def main() -> int:
    header_path, twin_path, ref = sys.argv[1:4]
    header_before = git_show(ref, header_path)
    header_after = pathlib.Path(header_path).read_text()
    twin = pathlib.Path(twin_path)
    text = twin.read_text()

    pairs = {}
    for old, new in hunks(header_before, header_after):
        if pairs.setdefault(old, new) != new:
            print(
                f"FAIL: one hunk maps to two different replacements:\n"
                f"----- hunk -----\n{old}",
                file=sys.stderr,
            )
            return 1

    applied = 0
    for old, new in pairs.items():
        expected = header_before.count(old)
        count = text.count(old)
        if count == 0 or count != expected:
            print(
                f"FAIL: hunk occurs {expected} time(s) in the header but "
                f"{count} time(s) in {twin_path}:\n"
                f"----- hunk -----\n{old}",
                file=sys.stderr,
            )
            return 1
        text = text.replace(old, new)
        applied += count

    twin.write_text(text)
    print(f"ported {applied} hunk(s) from {header_path} into {twin_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
