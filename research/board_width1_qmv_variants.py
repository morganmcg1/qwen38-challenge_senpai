#!/usr/bin/env python3
"""E50 Q1b control. Did the field ever edit the WIDTH-1 quantized matvec?

Ledger 173(B) records that width 1 dispatches `qmv_fast_impl` while widths 2..9
dispatch the crossrow `qmv_fast_crossrow_affine4_g64_m<...>` family, and that
they are different code paths. That fact is a confounder for the E50 Q1 result:
if every board solver only ever touched the crossrow helpers, the serial leg
would sit still even if it DID execute candidate code, and 'the serial leg never
moves' would prove nothing about whether it can.

So count distinct implementations of each path separately across the board.

  serial leg  -> qmv_fast_impl              (width 1)
  candidate   -> qmv_fast_crossrow_...      (widths 2..9)

Reads blob oids from /tmp/tree_ids.json's sibling scan; fetches each DISTINCT
blob once via one `git cat-file --batch`.
"""
import hashlib
import json
import re
import subprocess
import sys

FILES = ["Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
         "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"]
ROWS = "/tmp/rows_live.json"
TREES = "/tmp/tree_ids.json"
OUT = "/tmp/width1_variants.json"


def run(args, **kw):
    p = subprocess.run(args, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        sys.exit(f"git failed: {' '.join(args[:3])}: {p.stderr[:400]}")
    return p.stdout


def extract(text, name):
    """Body of a function/template by brace matching from its first definition."""
    i = text.find(name)
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    body = text[i:k + 1]
    return re.sub(r"\s+", " ", body).strip()


def self_test():
    """The headline here is a NEGATIVE (one width-1 implementation board-wide),
    so the extractor has to be shown capable of (a) finding a real body and
    (b) discriminating bodies that genuinely differ."""
    fails = 0
    for f in FILES:
        t = open(f).read()
        for name in ("qmv_fast_impl(", "qmv_fast_crossrow_affine4_g64_m"):
            b = extract(t, name)
            ok = b is not None and len(b) > 400 and b.count("{") == b.count("}")
            print(f"  {'ok  ' if ok else 'FAIL'} {f.split('/')[-1]:<16s} {name:<34s} "
                  f"len={len(b) if b else 0}")
            if not ok:
                fails += 1
            elif name == "qmv_fast_impl(":
                assert b.startswith("qmv_fast_impl("), b[:60]

    # discrimination control: a one-token edit must change the signature
    t = open(FILES[0]).read()
    a = extract(t, "qmv_fast_impl(")
    mutated = t.replace("qmv_fast_impl(", "qmv_fast_impl( /*x*/ ", 1)
    b = extract(mutated, "qmv_fast_impl(")
    if a == b:
        print("  FAIL extractor is blind to an edit inside the body")
        fails += 1
    else:
        print("  ok   extractor detects a one-token edit inside the body")

    # negative control: a name that does not exist must yield None
    if extract(t, "qmv_fast_nonexistent_zzz(") is not None:
        print("  FAIL extractor invents a body for a missing symbol")
        fails += 1
    else:
        print("  ok   missing symbol yields None")
    print(f"self-test: {fails} failures")
    return fails


def main():
    if "--self-test" in sys.argv:
        sys.exit(1 if self_test() else 0)
    ids = list(json.load(open(TREES))["build"])
    specs = [f"upstream/submissions/{s}:{f}" for s in ids for f in FILES]
    out = run(["git", "cat-file", "--batch-check=%(objectname) %(rest)"],
              input="".join(f"{s} {s}\n" for s in specs))
    oid = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] not in ("missing", "ambiguous"):
            oid[p[1]] = p[0]

    blobs = sorted(set(oid.values()))
    print(f"{len(ids)} submissions x {len(FILES)} files -> {len(blobs)} distinct blobs")
    text = {}
    raw = run(["git", "cat-file", "--batch"], input="".join(b + "\n" for b in blobs))
    pos = 0
    for b in blobs:
        nl = raw.index("\n", pos)
        hdr = raw[pos:nl].split()
        size = int(hdr[2])
        body = raw[nl + 1:nl + 1 + size]
        text[b] = body
        pos = nl + 1 + size + 1

    TARGETS = {"width1_qmv_fast_impl": "qmv_fast_impl(",
               "crossrow_m_family": "qmv_fast_crossrow_affine4_g64_m"}
    per_blob = {}
    for b in blobs:
        per_blob[b] = {k: (hashlib.sha256(v.encode()).hexdigest()[:12]
                           if (v := extract(text[b], t)) else None)
                       for k, t in TARGETS.items()}

    result = {}
    for s in ids:
        acc = {}
        for k in TARGETS:
            sigs = [per_blob[oid[f"upstream/submissions/{s}:{f}"]][k]
                    for f in FILES if f"upstream/submissions/{s}:{f}" in oid]
            sigs = [x for x in sigs if x]
            acc[k] = hashlib.sha256("|".join(sigs).encode()).hexdigest()[:12] if sigs else None
        result[s] = acc

    print()
    for k in TARGETS:
        vals = [v[k] for v in result.values()]
        got = [x for x in vals if x]
        print(f"{k:<24s}: {len(set(got)):4d} distinct implementations "
              f"across {len(got)} submissions ({len(vals)-len(got)} not found)")
    json.dump(result, open(OUT, "w"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
