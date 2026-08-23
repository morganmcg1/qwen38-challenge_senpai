#!/usr/bin/env python3
"""E156 F4 item 2: is the DECODE JIT source string byte-identical?

FINDING 293 says four attempts on this kernel family all died in the decode
channel, and the leading hypothesis is that editing a file which lands inside
the decode module's JIT source string perturbs the decode kernels even when no
changed kernel is ever dispatched.

MLX builds two different source strings from two different module lists
(backend/metal/jit_kernels.cpp):

    decode / non-NAX : utils + gemm     + quantized_utils + quantized
    NAX              : utils + gemm_nax + quantized_utils + quantized_nax

This script does not take that composition on trust. It PARSES the two
`concatenate(...)` call sites out of jit_kernels.cpp, extracts the raw-string
payload each named module returns from mlx-generated/<module>.cpp, and digests
the composed strings at several revisions.

Two things must hold, and the second is what makes the first meaningful:

1. The decode source string is byte-identical across every revision.
2. The NAX source string DOES move at the candidate. Rule 101: without this
   positive control, an all-identical result would be indistinguishable from a
   script that cannot see the edit at all.

harness=offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
JIT = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp"
GEN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated"

# The two builder functions whose source strings we care about. Names are the
# enclosing C++ function in jit_kernels.cpp.
DECODE_BUILDER = "get_quantized_kernel"
NAX_BUILDER = "get_qmm_nax_kernel"


def show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:{path}"],
        capture_output=True, text=True, check=True,
    ).stdout


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def parse_module_list(jit_src: str, builder: str) -> list[str]:
    """Extract the metal::X() modules concatenated inside one builder."""
    start = jit_src.find(f"MTL::ComputePipelineState* {builder}(")
    if start < 0:
        raise SystemExit(f"builder {builder} not found in {JIT}")
    body = jit_src[start:]
    cat = body.find("concatenate(")
    if cat < 0:
        raise SystemExit(f"no concatenate() inside {builder}")
    depth, i = 0, cat + len("concatenate(") - 1
    while i < len(body):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    args = body[cat + len("concatenate("):i]
    # Modules appear as metal::name(); a ternary may offer two alternatives,
    # and both branches are recorded so a mode switch cannot hide one.
    return re.findall(r"metal::(\w+)\(\)", args)


def module_payload(rev: str, module: str) -> str:
    """Return the exact raw-string payload metal::<module>() emits."""
    src = show(rev, f"{GEN}/{module}.cpp")
    m = re.search(r'const char\* ' + module + r'\(\) \{\s*return R"preamble\(',
                  src)
    if not m:
        raise SystemExit(f"no raw string for metal::{module}() at {rev}")
    body = src[m.end():]
    end = body.find(')preamble"')
    if end < 0:
        raise SystemExit(f"unterminated raw string for {module} at {rev}")
    return body[:end]


def compose(rev: str, modules: list[str]) -> tuple[str, dict[str, str]]:
    parts, per = [], {}
    for mod in modules:
        payload = module_payload(rev, mod)
        parts.append(payload)
        per[mod] = sha(payload)
    return "".join(parts), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", action="append", required=True,
                    metavar="LABEL=SHA",
                    help="revision to census, repeatable")
    args = ap.parse_args()

    revs = []
    for spec in args.rev:
        label, _, sha_ = spec.partition("=")
        if not sha_:
            raise SystemExit(f"--rev needs LABEL=SHA, got {spec!r}")
        revs.append((label, sha_))

    # Discover the composition from the candidate tree, then verify that the
    # composition itself did not change at any revision.
    compositions = {}
    for label, rev in revs:
        jit_src = show(rev, JIT)
        compositions[label] = {
            "decode": parse_module_list(jit_src, DECODE_BUILDER),
            "nax": parse_module_list(jit_src, NAX_BUILDER),
        }
    first = compositions[revs[0][0]]
    composition_stable = all(c == first for c in compositions.values())

    # The affine mode is the scored one. A ternary yields both branches, so
    # keep the affine member and drop the fp alternative.
    def affine_only(mods: list[str]) -> list[str]:
        return [m for m in mods if not m.startswith("fp_")]

    decode_mods = affine_only(first["decode"])
    nax_mods = affine_only(first["nax"])

    rows = []
    for label, rev in revs:
        dec, dec_per = compose(rev, decode_mods)
        nax, nax_per = compose(rev, nax_mods)
        rows.append({
            "label": label,
            "rev": rev,
            "decode_source_bytes": len(dec),
            "decode_source_sha256": sha(dec),
            "nax_source_bytes": len(nax),
            "nax_source_sha256": sha(nax),
            "decode_per_module_sha256": dec_per,
            "nax_per_module_sha256": nax_per,
            "nax_text_in_decode_modules": sum(
                module_payload(rev, m).count("_nax") for m in decode_mods),
        })

    decode_digests = {r["decode_source_sha256"] for r in rows}
    nax_digests = {r["nax_source_sha256"] for r in rows}
    decode_identical = len(decode_digests) == 1
    nax_moved = len(nax_digests) > 1
    no_leak = all(r["nax_text_in_decode_modules"] == 0 for r in rows)

    ok = decode_identical and nax_moved and no_leak and composition_stable
    out = {
        "experiment": "E156",
        "rung": "R1",
        "harness": "offline",
        "question": (
            "Does this candidate perturb the decode JIT source string, which "
            "is the shared cause of death in FINDING 293?"
        ),
        "e156_decode_jit_source_byte_identical": decode_identical,
        "e156_decode_jit_positive_control_nax_moved": nax_moved,
        "e156_decode_jit_no_nax_text_in_decode_modules": no_leak,
        "e156_decode_jit_composition_stable": composition_stable,
        "e156_decode_jit_gate_pass": ok,
        "decode_modules": decode_mods,
        "nax_modules": nax_mods,
        "decode_builder": DECODE_BUILDER,
        "nax_builder": NAX_BUILDER,
        "source_of_composition": (
            f"{JIT}, parsed from the concatenate() call inside each builder; "
            "not hardcoded here"
        ),
        "revisions": rows,
        "interpretation": (
            "The decode qmv kernels are JIT-compiled from the decode source "
            "string alone. If that string is byte-identical, the candidate "
            "cannot change a decode kernel through the JIT path, because the "
            "compiler input is the same bytes. The NAX digest moving at the "
            "candidate is the positive control that this script can see the "
            "edit at all."
        ),
    }
    (HERE / "e156-decode-jit-source-identity.json").write_text(
        json.dumps(out, indent=2) + "\n")

    print("decode modules:", " + ".join(decode_mods))
    print("nax    modules:", " + ".join(nax_mods))
    print()
    print("%-12s %-10s %-12s %-10s %-12s %s"
          % ("label", "dec bytes", "decode sha8", "nax bytes", "nax sha8",
             "nax text in decode"))
    for r in rows:
        print("%-12s %-10d %-12s %-10d %-12s %d"
              % (r["label"], r["decode_source_bytes"],
                 r["decode_source_sha256"][:8], r["nax_source_bytes"],
                 r["nax_source_sha256"][:8],
                 r["nax_text_in_decode_modules"]))
    print()
    print("e156_decode_jit_source_byte_identical      %s" % decode_identical)
    print("positive control, nax digest moved         %s" % nax_moved)
    print("no _nax text inside the decode modules     %s" % no_leak)
    print("composition stable across revisions        %s" % composition_stable)
    print("gate                                       %s"
          % ("PASS" if ok else "FAIL"))

    if not ok:
        raise SystemExit("decode JIT source identity gate failed")


if __name__ == "__main__":
    main()
