#!/usr/bin/env python3
"""E124 stage 0.1 and 0.2: prove the island premise at the live pin.

Two questions, both answerable with zero GPU:

1. Do the six `precision_islands.*` tensors exist in the head the harness will
   actually load, and does that file's sha256 equal the manifest pin?
2. Is `Qwen35Attention.isCompletePermutation` true for K and for V at this pin?
   If yes, `installExactQKVRows` takes the fast branch, K and V are dense BF16
   today, and only 1,024 of 12,288 Q rows carry a scatter. The whole byte model
   for this experiment depends on that branch.

The permutation predicate is reimplemented here exactly as the Swift source
states it (`Qwen35.swift:2426-2448`): length equals `count`, every value in
`0 ..< count`, no repeats. Sortedness is reported but is not part of it.

  python3 research/e124_head_census.py --out research/out/e124-head-census.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e82_st import SafeTensors  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
MANIFEST = Path(__file__).resolve().parent.parent / "mtp-head.manifest.json"

# The counts `sanitize` passes to `installExactQKVRows` at Qwen35.swift:4342.
OUTPUT_COUNT = {"q": 12_288, "k": 1_024, "v": 1_024}


def is_complete_permutation(idx: np.ndarray, count: int) -> bool:
    if idx.ndim != 1 or idx.size != count or count <= 0:
        return False
    seen = np.zeros(count, dtype=bool)
    for value in idx.astype(np.int64):
        if value < 0 or value >= count or seen[value]:
            return False
        seen[value] = True
    return True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root: Path) -> tuple[str, int]:
    """The manifest pin is a TREE digest, not a file digest.

    `QwenMTPHeadDeclaration.swift:186-230` and `research/fetch-declared-head.sh`
    hash `"<file sha256>  <relative path>\\n"` for every regular file except
    `README.md`, in `LC_ALL=C` path order. Comparing the manifest value against
    a bare `model.safetensors` digest is a category error.
    """
    lines: list[str] = []
    total = 0
    for path in sorted(
            (p for p in root.rglob("*") if p.is_file() and p.name != "README.md"),
            key=lambda p: str(p.relative_to(root)).encode()):
        lines.append(f"{sha256(path)}  {path.relative_to(root)}\n")
        total += path.stat().st_size
    return hashlib.sha256("".join(lines).encode()).hexdigest(), total


def census(path: Path, manifest: dict) -> dict:
    st = SafeTensors(path)
    digest = sha256(path)
    tree_root = path.parent
    tree_sha, tree_bytes = tree_digest(tree_root)
    out: dict = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": digest,
        "tree_root": str(tree_root),
        "tree_sha256": tree_sha,
        "tree_bytes": tree_bytes,
        "tree_sha256_matches_manifest": tree_sha == manifest["sha256"],
        "tree_bytes_match_manifest": tree_bytes == manifest["bytes"],
        "tensor_count": len(st.entries),
        "islands": {},
        "island_bytes_total": 0,
    }
    for proj, count in OUTPUT_COUNT.items():
        wkey = f"precision_islands.{proj}.weight"
        ikey = f"precision_islands.{proj}.indices"
        if wkey not in st.entries or ikey not in st.entries:
            out["islands"][proj] = {"present": False}
            continue
        w, i = st.entries[wkey], st.entries[ikey]
        idx = np.asarray(st.array(ikey)).ravel()
        out["islands"][proj] = {
            "present": True,
            "weight_dtype": w.dtype,
            "weight_shape": list(w.shape),
            "weight_bytes": w.nbytes,
            "indices_dtype": i.dtype,
            "indices_shape": list(i.shape),
            "indices_bytes": i.nbytes,
            "output_count": count,
            "unique": int(np.unique(idx).size),
            "min": int(idx.min()),
            "max": int(idx.max()),
            "is_complete_permutation": is_complete_permutation(idx, count),
            "is_sorted_strictly_increasing": bool(np.all(np.diff(idx) > 0)),
        }
        out["island_bytes_total"] += w.nbytes + i.nbytes

    k = out["islands"].get("k", {})
    v = out["islands"].get("v", {})
    out["fast_branch_live"] = bool(
        k.get("is_complete_permutation") and v.get("is_complete_permutation"))

    # Projection weights the plain (no-island) path must read instead.
    proj_keys = {
        name: e for name, e in st.entries.items()
        if name.startswith("mtp.") is False
        and any(name.startswith(p) for p in ("self_attn.",))
    }
    out["self_attn_tensors"] = {
        name: {"dtype": e.dtype, "shape": list(e.shape), "bytes": e.nbytes}
        for name, e in sorted(proj_keys.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/out/e124-head-census.json")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    heads = {
        "declared (verifiable tree)": CACHE / "mtp-head-declared/model.safetensors",
        "declared-run (E124 arms load this)": CACHE / "mtp-head-declared-run/model.safetensors",
        "setup default mtp-head": CACHE / "mtp-head/model.safetensors",
    }
    report = {"manifest": manifest, "heads": {}}
    for label, path in heads.items():
        if not path.exists():
            report["heads"][label] = {"path": str(path), "present": False}
            continue
        report["heads"][label] = census(path, manifest)

    for label, entry in report["heads"].items():
        print(f"=== {label}")
        if not entry.get("file_sha256"):
            print("    absent")
            continue
        print(f"    file bytes       {entry['bytes']:,}")
        print(f"    file sha256      {entry['file_sha256']}")
        print(f"    tree bytes       {entry['tree_bytes']:,}  "
              f"manifest_match={entry['tree_bytes_match_manifest']}")
        print(f"    tree sha256      {entry['tree_sha256']}")
        print(f"    manifest sha256  {manifest['sha256']}  "
              f"match={entry['tree_sha256_matches_manifest']}")
        print(f"    tensors          {entry['tensor_count']}")
        print("    proj  dtype  shape              bytes        outCount  unique  min  max  complete-perm")
        for proj, isl in entry["islands"].items():
            if not isl.get("present"):
                print(f"    {proj:>4}  ABSENT")
                continue
            print(f"    {proj:>4}  {isl['weight_dtype']:>5}  {str(isl['weight_shape']):<18}"
                  f"{isl['weight_bytes']:>11,}  {isl['output_count']:>8}  {isl['unique']:>6}"
                  f"{isl['min']:>5}{isl['max']:>6}   {isl['is_complete_permutation']}")
            print(f"          {isl['indices_dtype']:>5}  {str(isl['indices_shape']):<18}"
                  f"{isl['indices_bytes']:>11,}  (indices)")
        print(f"    island bytes     {entry['island_bytes_total']:,}")
        print(f"    fast branch live {entry['fast_branch_live']}")
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
