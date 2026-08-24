#!/usr/bin/env python3
"""Print the exact submitted-surface diffstat of a ref against organizer main.

The submitted surface is `editablePaths` from `benchmark.json`, which is the
only authority for what a Yukon archive replaces. Anything outside it, such as
`research/` or `senpai/`, cannot ride along with the candidate.

Run: python3 research/e175_packaged_diff.py [ref] [organizer_ref]
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    organizer = sys.argv[2] if len(sys.argv) > 2 else "upstream/main"
    with open("benchmark.json") as fh:
        bench = json.load(fh)
    tracks = bench.get("tracks", [bench])
    paths = sorted(
        {p for t in tracks for p in t.get("editablePaths", [])}
    )
    out = subprocess.run(
        ["git", "--no-pager", "diff", "--stat", organizer, ref, "--", *paths],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    print(f"submitted surface: {len(paths)} declared paths")
    print(f"{organizer} -> {ref}")
    print(out or "  (no difference)")


if __name__ == "__main__":
    main()
