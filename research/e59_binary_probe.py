#!/usr/bin/env python3
"""Prove by content that a linked binary carries exactly one arm's QMV routing.

  research/e59_binary_probe.py ARM_JSON BINARY [BINARY ...]

The crossrow QMV family has an `mlx-generated/quantized.cpp` twin, so it is
JIT-compiled at run time from a source string held inside that C++ translation
unit. The string is linked into the image verbatim, which means each width's
dispatch call — `qmv_fast_crossrow_affine4_g64_m_rbx<T, 5, 5, true>` and its
siblings — appears as literal bytes in the binary that will be timed.

That makes the image itself the evidence. An mtime comparison is not: llbuild
signs C and C++ inputs by content, so rewriting a file with identical bytes
correctly relinks nothing and leaves a product older than its sources. The
first E59 rung 2b leg failed that way, on the `shipped` arm, whose bytes equal
the base bytes by construction.

Only `mlxfast-runtime-worker` embeds the source. `mlxfast-swift` is a driver
that spawns the worker and contains no Metal text at all, so probing it would
always fail. Pass the worker.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Every wrapper an E59 arm can route a width to. The probe rejects any of these
# that the arm did not ask for, so a leftover build cannot pass as the arm.
WRAPPERS = (
    "qmv_fast_crossrow_affine4_g64_m",
    "qmv_fast_crossrow_affine4_g64_m_rb2",
    "qmv_fast_crossrow_affine4_g64_m_rb2t",
    "qmv_fast_crossrow_affine4_g64_m_rbx",
    "qmv_fast_crossrow_affine4_g64_m_rbx4",
)
# Widened to 9 for E68. The exclusivity half of the probe can only reject a
# stray instantiation it enumerates, and the live table already routes NA=6
# while E68's probe arms reach NA=9, so a range that stops at 5 is a guard that
# cannot fail for exactly the widths under test.
IPG_RANGE = range(2, 10)


def token(wrapper: str, m: int, ipg: int) -> bytes:
    return f"{wrapper}<T, {m}, {ipg}, true>".encode()


def probe(arm: dict, path: pathlib.Path) -> list[str]:
    blob = path.read_bytes()
    faults: list[str] = []
    for width, spec in sorted(arm["routing"].items(), key=lambda kv: int(kv[0])):
        m = int(width)
        wrapper = spec["wrapper"]
        if wrapper not in WRAPPERS:
            continue  # width 2 uses the pair kernel, which takes no <T, M, IPG>
        want = token(wrapper, m, spec["ipg"])
        if want not in blob:
            faults.append(f"missing M={m} dispatch {want.decode()}")
        for other_wrapper in WRAPPERS:
            for ipg in IPG_RANGE:
                other = token(other_wrapper, m, ipg)
                if other != want and other in blob:
                    faults.append(f"unexpected M={m} dispatch {other.decode()}")
    return faults


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_json", type=pathlib.Path)
    ap.add_argument("binaries", nargs="+", type=pathlib.Path)
    args = ap.parse_args()

    arm = json.loads(args.arm_json.read_text())
    rc = 0
    for path in args.binaries:
        if not path.exists():
            print(f"BINARY PROBE FAIL {path}: not built")
            rc = 1
            continue
        faults = probe(arm, path)
        if faults:
            rc = 1
            print(f"BINARY PROBE FAIL {path} (arm {arm['arm']}):")
            for fault in faults:
                print(f"  {fault}")
        else:
            widths = ",".join(sorted(arm["routing"], key=int))
            print(f"BINARY PROBE OK {path}: arm {arm['arm']} routing at "
                  f"widths {widths} is present and exclusive")
    return rc


if __name__ == "__main__":
    sys.exit(main())
