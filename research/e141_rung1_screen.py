#!/usr/bin/env python3
"""E141 rung 1 screen: can we derive the declared coarse table from lm_head?

`Qwen35.swift:5870-5872` states that the shipped `draft_lm_head.*` is exactly
`quantize(dequantize(exact compact lm_head), 64, 2)`. If that is true bit for
bit, the coarse shortlist table is a pure function of the target's own fixed
lm_head, so it can be derived in-process for ANY compact prefix at zero ship
bytes. That is the licensing gate for widening the compact draft vocabulary.

This is the offline screen. It reads the same two artifacts the process loads
and applies the same op chain, so it answers the data question before any Swift
build. The in-process Swift test is the real gate; a screen failure here stops
the assignment immediately.

Bit-for-bit only. Rule 101: a positive control perturbs one input row and must
make the comparison fail.

  python3 research/e141_rung1_screen.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

PREFIX_COUNT = 98_304
CONTROL_START = 248_044
CONTROL_END = 248_070
REAL_COUNT = PREFIX_COUNT + CONTROL_END - CONTROL_START
PADDED_COUNT = 98_336

TARGET = Path("weights/model-00003-of-00003.safetensors")
HEAD = Path.home() / (
    ".cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared/model.safetensors"
)
OUT = Path("research/e141-rung1-screen.json")


def compact_rows(a: mx.array) -> mx.array:
    """`makeCompactDraftHead`'s row selection (`Qwen35.swift:6153`)."""
    padding = PADDED_COUNT - REAL_COUNT
    return mx.concatenate(
        [a[:PREFIX_COUNT], a[CONTROL_START:CONTROL_END], a[:padding]], axis=0
    )


def compare(name: str, got: mx.array, want: mx.array) -> dict:
    if got.shape != want.shape or got.dtype != want.dtype:
        return {
            "tensor": name,
            "equal": False,
            "reason": "shape or dtype",
            "got": [list(got.shape), str(got.dtype)],
            "want": [list(want.shape), str(want.dtype)],
        }
    # Bit-for-bit: compare the raw bytes, so bf16 -0.0 vs 0.0 and any NaN
    # payload difference still counts as a mismatch.
    gb = got.view(mx.uint16) if got.dtype == mx.bfloat16 else got.view(mx.uint32)
    wb = want.view(mx.uint16) if want.dtype == mx.bfloat16 else want.view(mx.uint32)
    diff = gb != wb
    n = int(mx.sum(diff).item())
    row = {
        "tensor": name,
        "equal": n == 0,
        "elements": int(gb.size),
        "differing_elements": n,
    }
    if n:
        g = got.astype(mx.float32)
        w = want.astype(mx.float32)
        row["max_abs_difference"] = float(mx.max(mx.abs(g - w)).item())
        row["differing_rows"] = int(mx.sum(mx.any(diff, axis=1)).item())
    return row


def derive(exact_w, exact_s, exact_z):
    rows = mx.dequantize(exact_w, exact_s, exact_z, group_size=64, bits=4, mode="affine")
    wq, s, z = mx.quantize(rows, group_size=64, bits=2, mode="affine")
    mx.eval(wq, s, z)
    return wq, s, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    target = mx.load(str(TARGET))
    head = mx.load(str(HEAD))

    exact_w = compact_rows(target["language_model.lm_head.weight"])
    exact_s = compact_rows(target["language_model.lm_head.scales"])
    exact_z = compact_rows(target["language_model.lm_head.biases"])
    mx.eval(exact_w, exact_s, exact_z)

    declared = {
        "weight": head["draft_lm_head.weight"],
        "scales": head["draft_lm_head.scales"],
        "biases": head["draft_lm_head.biases"],
    }

    wq, s, z = derive(exact_w, exact_s, exact_z)
    checks = [
        compare("draft_lm_head.weight", wq, declared["weight"]),
        compare("draft_lm_head.scales", s, declared["scales"]),
        compare("draft_lm_head.biases", z, declared["biases"]),
    ]
    reproduced = all(c["equal"] for c in checks)

    # Rule 101 positive control: perturb one 4-bit input row and require the
    # derivation to disagree with the declared table.
    bad_s = mx.array(exact_s)
    bad_s[7] = bad_s[7] * mx.array(1.5, dtype=bad_s.dtype)
    mx.eval(bad_s)
    cwq, cs, cz = derive(exact_w, bad_s, exact_z)
    control = [
        compare("control.weight", cwq, declared["weight"]),
        compare("control.scales", cs, declared["scales"]),
        compare("control.biases", cz, declared["biases"]),
    ]
    control_failed = any(not c["equal"] for c in control)

    report = {
        "mlx_version": mx.__version__,
        "prefix_count": PREFIX_COUNT,
        "padded_count": PADDED_COUNT,
        "target": str(TARGET),
        "declared_head": str(HEAD),
        "checks": checks,
        "e141_coarse_table_bit_reproduction": 1.0 if reproduced else 0.0,
        "positive_control": control,
        "positive_control_failed_as_required": control_failed,
        "gate_passed": reproduced and control_failed,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for c in checks + control:
        print(
            f"  {c['tensor']:<24} equal={int(c['equal'])} "
            f"differing={c.get('differing_elements', '-')}"
            + (
                f" max_abs={c['max_abs_difference']:.6g} rows={c['differing_rows']}"
                if not c["equal"] and "max_abs_difference" in c
                else ""
            )
        )
    print(f"\ne141_coarse_table_bit_reproduction = {report['e141_coarse_table_bit_reproduction']}")
    print(f"positive control failed as required = {control_failed}")
    print(f"gate passed = {report['gate_passed']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
