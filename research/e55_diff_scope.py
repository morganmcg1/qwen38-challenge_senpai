#!/usr/bin/env python3
"""Research-only (qwen38-r1-e55): prove the candidate diff is exactly one cell.

The twin audit reports STALE for this section once either twin is edited,
because the case-8 comment-only waiver is pinned to both bodies' sha256. That is
designed fail-closed behaviour, not new drift. This script separates the two:

  1. every line of the audit's reported drift block is a COMMENT line;
  2. the two edited lines are byte-identical in the readable header and in the
     runtime-effective JIT twin;
  3. no other dispatch cell moved, and the JIT source grew by 0 bytes.
"""
import subprocess
import sys

BASE = "a35bb006fd47785dc916241df63ec8780bda8e5c"
HEADER = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

ok = True


def check(label, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")


proc = subprocess.run(
    [sys.executable, "research/twin_audit.py"], capture_output=True, text=True
)
audit = proc.stdout + proc.stderr
block = [
    ln for ln in audit.splitlines()
    if ln[:1] in ("+", "-") and not ln.startswith(("+++", "---")) and ln[1:].strip()
]
noncomment = [ln for ln in block if not ln[1:].strip().startswith("//")]
check(
    "audit drift block is comment-only",
    bool(block) and not noncomment,
    f"{len(block)} drift lines, {len(noncomment)} non-comment",
)

for name, path in (("header", HEADER), ("twin", TWIN)):
    text = open(path).read()
    asserts = sorted({l.strip() for l in text.splitlines() if "static_assert(NA >= 2" in l})
    case9 = sorted({l.strip() for l in text.splitlines() if "crossrow_affine4_g64_m<T, 9," in l})
    print(f"    {name}: {asserts} | {case9}")

hdr, twn = open(HEADER).read(), open(TWIN).read()
check(
    "both twins carry NA<=5 identically",
    hdr.count('NA <= 5, "wide multi-row QMV supports NA in [2, 5]"') == 1
    and twn.count('NA <= 5, "wide multi-row QMV supports NA in [2, 5]"') == 1,
)
check(
    "both twins carry <T, 9, 5, true> identically",
    hdr.count("<T, 9, 5, true>") == 1 and twn.count("<T, 9, 5, true>") == 1,
)
check("no <T, 9, 3, true> left", hdr.count("<T, 9, 3, true>") == 0 and twn.count("<T, 9, 3, true>") == 0)

base_twin = subprocess.run(
    ["git", "show", f"{BASE}:{TWIN}"], capture_output=True, text=True
).stdout
base_hdr = subprocess.run(
    ["git", "show", f"{BASE}:{HEADER}"], capture_output=True, text=True
).stdout

print("    untouched dispatch cells (base -> candidate):")
untouched = True
for tag in ("<T, 2>", "<T, 3, 3, true>", "<T, 4, 4, true>", "<T, 5, 3, true>",
            "<T, 6, 3, true>", "<T, 7, 4, true>", "<T, 8, 4, true>"):
    b, c = base_twin.count(tag), twn.count(tag)
    print(f"      {tag:22s} {b} -> {c}")
    untouched = untouched and b == c
check("every other dispatch cell unchanged (incl. case 5 and case 8)", untouched)

check(
    "JIT source byte delta is zero",
    len(twn) == len(base_twin),
    f"base={len(base_twin)} candidate={len(twn)} delta={len(twn) - len(base_twin)}",
)
check(
    "readable header byte delta is zero",
    len(hdr) == len(base_hdr),
    f"base={len(base_hdr)} candidate={len(hdr)} delta={len(hdr) - len(base_hdr)}",
)

print(f"\nE55_DIFF_SCOPE={'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
