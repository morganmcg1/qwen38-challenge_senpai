#!/usr/bin/env python3
"""E130 rung 0b: reduce the E80 dispatch-census patch to the two hunks E130 needs.

The full `research/e80-artifacts/gputime-census.patch` also edits
`Qwen36MTPBlockSession.swift` (round-window labels) and a test file that no
longer exists on this base. Edward owns the session file this round, so E130
takes only the self-contained census instrument and its startup hook.

    python3 research/e130_split_census_patch.py /tmp/e130-census.patch
"""

from __future__ import annotations

import re
import sys

SRC = "research/e80-artifacts/gputime-census.patch"
DROP = "diff --git a/Tests/MLXFastTests/E71WidthTaxCensusTests.swift"


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/e130-census.patch"
    parts = re.split(r"(?m)^(?=diff --git )", open(SRC).read())
    keep = [p for p in parts if p.startswith("diff --git") and not p.startswith(DROP)]
    if len(keep) != 3:
        raise SystemExit(f"expected 3 hunks, found {len(keep)}")
    with open(out, "w") as fh:
        fh.write("".join(keep))
    for p in keep:
        print(p.split("\n")[0])


if __name__ == "__main__":
    main()
