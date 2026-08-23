#!/usr/bin/env python3
"""Run the remaining E150 offline rungs back to back in one supervised job.

`harness=local`. Zero GPU. Zero Swift. Rule 79 is not engaged: every number
these rungs produce is an offline replay price, not a timing contrast.

Both rungs take about the same wall clock as R1 did on its own, and neither
touches the other's state, so running them in one process keeps the job slot
free for the R4 Swift build.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

RUNGS = [
    ("curve_bracket", ["e150_curve_bracket.py", "--json", "curve_bracket.json"]),
    ("r1", ["e150_r1.py", "--json", "r1.json"]),
]


def main() -> int:
    common = ["--windows", "200", "--seeds", "6"]
    failures = []
    for name, argv in RUNGS:
        started = time.time()
        print("\n" + "=" * 72, flush=True)
        print("== %s" % name, flush=True)
        print("=" * 72, flush=True)
        proc = subprocess.run(
            [sys.executable, str(HERE / argv[0])] + common + argv[1:],
            cwd=str(HERE.parent))
        print("== %s exit %d after %.1f s"
              % (name, proc.returncode, time.time() - started), flush=True)
        if proc.returncode != 0:
            failures.append(name)
    if failures:
        print("\nFAILED rungs: %s" % ", ".join(failures), flush=True)
        return 1
    print("\nall rungs finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
