#!/usr/bin/env python3
"""Research-only (qwen38-r1-e55): falsification test for research/e55_twin_gate.py.

Per ledger 178(E), an instrument that cannot fail is not an instrument. This
applies the exact faults the gate exists to catch and asserts it catches each
one, then restores the tree. It must be run on a clean worktree for the two
twins and it restores them from disk, not from git, so an uncommitted candidate
edit survives.

    python3 research/e55_twin_gate_negative_control.py
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
GATE = [sys.executable, "research/e55_twin_gate.py"]


def gate():
    proc = subprocess.run(GATE, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


FAULTS = [
    (
        "code desync: twin only moves case 7 to NA=5",
        TWIN,
        "qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>(",
        "qmv_fast_crossrow_affine4_g64_m<T, 7, 5, true>(",
        "NON-COMMENT drift",
    ),
    (
        "code desync: header only relaxes the NA bound further",
        HEADER,
        'static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");',
        'static_assert(NA >= 2 && NA <= 6, "wide multi-row QMV supports NA in [2, 6]");',
        "NON-COMMENT drift",
    ),
    (
        "new comment divergence: extra comment in the twin only",
        TWIN,
        "        case 9:\n",
        "        case 9:\n          // a new, unpinned comment\n",
        "does not match the pinned block",
    ),
]

ok = True
rc, out = gate()
if rc != 0:
    print("PRECONDITION FAIL: the gate does not pass on the current tree:")
    print(out)
    sys.exit(1)
print("[PASS] precondition: gate passes on the current tree")

for label, path, find, replace, expect in FAULTS:
    original = path.read_text(encoding="utf-8")
    if find not in original:
        print(f"[SKIP] {label}: anchor not present")
        continue
    try:
        path.write_text(original.replace(find, replace, 1), encoding="utf-8")
        rc, out = gate()
        caught = rc != 0 and expect in out
        ok = ok and caught
        print(f"[{'PASS' if caught else 'FAIL'}] {label}: rc={rc} expected={expect!r}")
        if not caught:
            print("    " + out.strip().replace("\n", "\n    "))
    finally:
        path.write_text(original, encoding="utf-8")

rc, out = gate()
restored = rc == 0
ok = ok and restored
print(f"[{'PASS' if restored else 'FAIL'}] tree restored and gate passes again")
print(f"\nE55_TWIN_GATE_NEGATIVE_CONTROL={'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
