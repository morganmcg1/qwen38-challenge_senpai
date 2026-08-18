#!/usr/bin/env python3
"""Prove research/twin_waiver_digests.py exercises its divergent branch, and
that twin_audit.py now fails CLOSED on a comment-only divergence (empty table).

Fully reversible: the perturbed twin is restored from its original bytes and the
blob hash is re-verified at the end.
"""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
TWIN = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"


def blob(path):
    return subprocess.run(
        ["git", "hash-object", str(path)], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()


def run(argv):
    p = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


original = TWIN.read_bytes()
before_blob = blob(TWIN)
print(f"=== twin blob BEFORE : {before_blob}")

text = original.decode("utf-8")
lines = text.split("\n")
# Find the first whole-line // comment INSIDE the SECTION BODY (i.e. after the
# `// Contents from "..."` banner), not in the preamble metadata -- a preamble
# edit trips twin_audit's coarser "generated preamble metadata changed" gate
# instead of the per-section comparison the digests tool derives rows for.
start = next(
    i
    for i, l in enumerate(lines)
    if l.startswith('// Contents from "mlx/backend/metal/kernels/quantized.h"')
)
idx = next(
    i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("//")
)
print(f"=== perturbing checked-in comment at file line {idx + 1}: {lines[idx][:70]!r}")
lines[idx] = lines[idx] + " SELFTEST-MARKER"
TWIN.write_bytes("\n".join(lines).encode("utf-8"))
print(f"=== twin blob PERTURBED: {blob(TWIN)}")

rc = 0
try:
    rc_d, out_d = run(["python3", "research/twin_waiver_digests.py", "quantized"])
    print(f"\n=== twin_waiver_digests.py quantized  rc={rc_d}")
    print(out_d.rstrip())
    if "COMMENT-ONLY" not in out_d:
        print("SELFTEST FAIL: digests tool did not label the section COMMENT-ONLY")
        rc = 1
    if "checked_in_sha256" not in out_d:
        print("SELFTEST FAIL: digests tool emitted no paste-ready row")
        rc = 1

    rc_a, out_a = run(["python3", "research/twin_audit.py"])
    tail = [l for l in out_a.rstrip().split("\n") if l.strip()][-6:]
    print(f"\n=== twin_audit.py  rc={rc_a} (expected NONZERO: fail-closed)")
    print("\n".join(tail))
    if rc_a == 0:
        print("SELFTEST FAIL: twin_audit.py passed on a comment-only divergence")
        rc = 1
finally:
    TWIN.write_bytes(original)
    after = blob(TWIN)
    print(f"\n=== twin blob RESTORED: {after}")
    if after != before_blob:
        print(f"SELFTEST FAIL: restore did not reproduce {before_blob}")
        rc = 1

rc_a2, out_a2 = run(["python3", "research/twin_audit.py"])
print(f"=== twin_audit.py after restore  rc={rc_a2} (expected 0)")
print(out_a2.rstrip().split("\n")[-1])
if rc_a2 != 0:
    rc = 1

print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
sys.exit(rc)
