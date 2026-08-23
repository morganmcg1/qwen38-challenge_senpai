#!/usr/bin/env python3
"""E151 rung R2 smoke probe: does the pipelined tree compile at the cells that matter?

Not a gate. `research/e151_r2_compile_gate.py` is the gate. This exists so the
first compile of a new kernel edit is cheap to read.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e147_rungE1c as e1c  # noqa: E402

CELLS = {
    "scored_bf16_g64_b4": e1c.SCORED_INST,
    "aot_bf16_g32_b4": "affine_qmm_t_nax<bfloat16_t, 32, 4, 1, 0, 64, 64, 64, 2, 2>",
    "aot_float_g32_b4": "affine_qmm_t_nax<float, 32, 4, 1, 0, 64, 64, 64, 2, 2>",
    "aot_float_g64_b4": "affine_qmm_t_nax<float, 64, 4, 1, 0, 64, 64, 64, 2, 2>",
    "shipped_qmm_n": "affine_qmm_n_nax<bfloat16_t, 64, 4, 1, 64, 64, 64, 2, 2>",
    "shipped_gather_rhs": (
        "affine_gather_qmm_rhs_nax<bfloat16_t, 64, 4, 64, 64, 64, 2, 2, true>"
    ),
}


def main() -> int:
    src = e1c.preamble(e1c.worktree, e1c.NAX_TWINS)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp) / "probe"
        for label, inst in CELLS.items():
            rec = e1c.compile_probe(src, inst, "e151_r2_probe", workdir)
            digest = (rec.get("air_sha256") or "")[:16]
            print(f"{label:<24} compiled={rec['compiled']} "
                  f"air_bytes={rec.get('air_bytes')} sha={digest}")
            if not rec["compiled"]:
                failures += 1
                print(rec["error"][-1000:])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
