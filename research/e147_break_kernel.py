#!/usr/bin/env python3
"""Apply one named E147 positive-control defect to the readable Metal header.

Each edit is asserted to hit all four pipelined k-loops exactly, so a drifted
kernel refuses the edit instead of producing a half-broken control build.

Usage:
    research/e147_break_kernel.py <header-path> <barrier0|nobarrier|wronghalf>
"""

import pathlib
import sys

LOOP_TOP = ("      for (int k = 0; k < K_eff; k += BK) {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n")

MMA = "        mma_op.mma(Xs + cur * Xs_tile, Ws + cur * Ws_tile);\n"

DEFECTS = {
    "barrier0": (
        LOOP_TOP,
        "      for (int k = 0; k < K_eff; k += BK) {\n"
        "        if (k > 0) { threadgroup_barrier(mem_flags::mem_threadgroup); }\n",
    ),
    "nobarrier": (
        LOOP_TOP,
        "      for (int k = 0; k < K_eff; k += BK) {\n"
        "        /* e147 control: loop-top barrier deliberately removed */\n",
    ),
    "wronghalf": (
        MMA,
        "        mma_op.mma(Xs + (cur ^ 1) * Xs_tile, Ws + (cur ^ 1) * Ws_tile);\n",
    ),
}

EXPECTED_SITES = 4


def main():
    path = pathlib.Path(sys.argv[1])
    mode = sys.argv[2]
    if mode not in DEFECTS:
        raise SystemExit(f"e147_break_kernel: unknown mode {mode}")

    old, new = DEFECTS[mode]
    text = path.read_text()
    count = text.count(old)
    if count != EXPECTED_SITES:
        raise SystemExit(
            f"e147_break_kernel: {mode} expected {EXPECTED_SITES} sites, "
            f"found {count}")
    path.write_text(text.replace(old, new))
    print(f"e147_break_kernel: applied {mode} to {count} pipelined k-loops")


if __name__ == "__main__":
    main()
