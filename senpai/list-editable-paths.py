#!/usr/bin/env python3
"""Print the editablePaths declared by benchmark.json at a given git ref.

Used by senpai/diff-promoted-surface.sh (RULE 162). Exits non-zero rather than
printing an empty list, so a caller can never mistake a parse failure for an
empty scope.
"""
import json
import subprocess
import sys


def walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "editablePaths" and isinstance(value, list):
                for entry in value:
                    if isinstance(entry, str):
                        yield entry
                    elif isinstance(entry, dict) and "path" in entry:
                        yield entry["path"]
            else:
                yield from walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: list-editable-paths.py <git-ref>", file=sys.stderr)
        return 2
    ref = sys.argv[1]
    proc = subprocess.run(
        ["git", "show", f"{ref}:benchmark.json"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"cannot read benchmark.json at {ref}: {proc.stderr.strip()}",
              file=sys.stderr)
        return 2

    seen = []
    for path in walk(json.loads(proc.stdout)):
        if path not in seen:
            seen.append(path)

    if not seen:
        print(f"no editablePaths found in benchmark.json at {ref}",
              file=sys.stderr)
        return 1

    print("\n".join(seen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
