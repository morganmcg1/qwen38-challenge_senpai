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
import json
import subprocess
import sys

# Revision e55-r2 rebased the branch onto the advisor head that carries ledger
# 189. Pinning the old base made the advisor's own ledger, fixture and test
# commits read as unsubmitted changes of mine.
BASE = "989596895b7c8f889443dac0c87e024a428e6e9e"
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
# The advisor's fixed audit waives this section outright, so an empty drift block
# is the strongest outcome, not a missing one. Only non-comment drift may fail.
check(
    "audit passes with no non-comment drift",
    proc.returncode == 0 and not noncomment,
    f"rc={proc.returncode}, {len(block)} drift lines, {len(noncomment)} non-comment",
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

def blob_bytes(ref: str, path: str) -> int:
    sha = subprocess.run(["git", "rev-parse", f"{ref}:{path}"],
                         capture_output=True, text=True).stdout.strip()
    return int(subprocess.run(["git", "cat-file", "-s", sha],
                              capture_output=True, text=True).stdout.strip())


# len() on a decoded str counts characters, and both files carry multi-byte UTF-8,
# so the submitted byte budget must be read from the blob sizes instead.
for label, path in (("JIT source", TWIN), ("readable header", HEADER)):
    b = blob_bytes(BASE, path)
    c = len(open(path, "rb").read())
    check(f"{label} byte delta is zero", b == c,
          f"base={b} candidate={c} delta={c - b}")


# Step 7 of the promotion chain: a local win that depends on an unsubmitted file
# is not a candidate, so classify every changed file against benchmark.json.
def editable_paths() -> set:
    spec = json.load(open("benchmark.json"))
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "editablePaths":
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return {p for group in found for p in (group if isinstance(group, list) else [group])}


submitted = editable_paths()
changed = subprocess.run(
    ["git", "diff", "--name-only", BASE, "HEAD"], capture_output=True, text=True
).stdout.split()
in_set = [f for f in changed if f in submitted]
research_only = [f for f in changed if f not in submitted]
stray = [f for f in research_only if not f.startswith("research/")]

print("    submitted paths changed:")
for f in in_set:
    print(f"      {f}")
check("both twins are in benchmark.json editablePaths",
      sorted(in_set) == sorted([HEADER, TWIN]),
      f"{len(in_set)} submitted, expected exactly the 2 twins")
check("the runtime-effective JIT twin is submitted", TWIN in in_set)
check("every unsubmitted change is research-only",
      not stray,
      f"{len(research_only)} research-only, {len(stray)} outside research/"
      + (f": {stray}" if stray else ""))
check("research/ is compiled by no target",
      "research" not in open("Package.swift").read(),
      "Package.swift does not reference research/")

print(f"\nE55_DIFF_SCOPE={'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
