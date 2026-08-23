#!/usr/bin/env python3
"""E151 rung R2: is the R2 edit invisible when its arm is off?

Compares the AIR of the R2 tree with `kE151NaxDoubleBufferOn` forced to false
against the R1 tree at a git revision, on cells the arm does and does not
reach. A byte-identical pair says the edit adds no cost where it is not armed;
a differing pair says something leaked and has to be explained before the
submission is trusted.

Not a gate. A reconnaissance probe run once, to find out whether digest
identity is even achievable before a gate is written around it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e147_rungE1c as e1c  # noqa: E402

R2_FLAG_ON = "constexpr bool kE151NaxDoubleBufferOn = true;"
R2_FLAG_OFF = "constexpr bool kE151NaxDoubleBufferOn = false;"

CELLS = {
    "scored_bf16_g64_b4": e1c.SCORED_INST,
    "aot_bf16_g32_b4": "affine_qmm_t_nax<bfloat16_t, 32, 4, 1, 0, 64, 64, 64, 2, 2>",
    "aot_float_g32_b4": "affine_qmm_t_nax<float, 32, 4, 1, 0, 64, 64, 64, 2, 2>",
    "shipped_qmm_n": "affine_qmm_n_nax<bfloat16_t, 64, 4, 1, 64, 64, 64, 2, 2>",
    "shipped_gather_rhs": (
        "affine_gather_qmm_rhs_nax<bfloat16_t, 64, 4, 64, 64, 64, 2, 2, true>"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1-rev", required=True)
    args = ap.parse_args()

    tree = e1c.preamble(e1c.worktree, e1c.NAX_TWINS)
    r1 = e1c.preamble(lambda p: e1c.show(args.r1_rev, p), e1c.NAX_TWINS)
    forced_off = e1c.substitute(tree, R2_FLAG_ON, R2_FLAG_OFF, "r2_forced_off")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp) / "probe"
        for label, inst in CELLS.items():
            a = e1c.compile_probe(r1, inst, "e151_r2_neutral", workdir)
            b = e1c.compile_probe(forced_off, inst, "e151_r2_neutral", workdir)
            c = e1c.compile_probe(tree, inst, "e151_r2_neutral", workdir)
            same = (
                a["compiled"]
                and b["compiled"]
                and a["air_sha256"] == b["air_sha256"]
            )
            print(
                f"{label:<24} r1={a.get('air_bytes')} "
                f"r2_off={b.get('air_bytes')} r2_on={c.get('air_bytes')} "
                f"armoff_identical={same}"
            )
            if not same and a["compiled"] and b["compiled"]:
                print(f"    r1     {a['air_sha256'][:32]}")
                print(f"    r2_off {b['air_sha256'][:32]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
