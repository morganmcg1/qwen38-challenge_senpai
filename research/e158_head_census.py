#!/usr/bin/env python3
"""E158 R0 -- census both MTP head artifacts.

Enumerates every tensor in the organizer-pinned bf16 head and in the head that
`mtp-head.manifest.json` declares, recomputes the workflow's tree-digest rule
independently of the shell implementation, and writes one artifact.

No model is loaded and no GPU work is done: this reads safetensors headers.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

CACHE = Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1"
PINNED_DIR = CACHE / "mtp-head"
DECLARED_DIR = CACHE / "mtp-head-declared"
DECLARED_RUN_DIR = CACHE / "mtp-head-declared-run"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research/e158-artifacts/head-census.json"

# safetensors dtype -> bits per stored element
DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "U32": 32, "I32": 32, "U8": 8, "I8": 8}
VOCAB = 248_320
COMPACT_ROWS = 98_336
HIDDEN = 5_120


def read_header(path: Path) -> tuple[dict, int, dict | None]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    metadata = header.pop("__metadata__", None)
    return header, header_len + 8, metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(directory: Path) -> dict:
    """The rule the ranked workflow and the trusted CLI both use.

    sha256 over "<file sha256>  <relative path>\\n" lines, LC_ALL=C sorted by
    relative path, README.md excluded.
    """
    files = sorted(
        (p for p in directory.rglob("*") if p.is_file() and p.name != "README.md"),
        key=lambda p: str(p.relative_to(directory)).encode(),
    )
    lines = []
    records = []
    for path in files:
        rel = str(path.relative_to(directory))
        sha = file_sha256(path)
        lines.append(f"{sha}  {rel}\n")
        records.append({"path": rel, "sha256": sha, "bytes": path.stat().st_size})
    return {
        "dir": str(directory),
        "sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "bytes": sum(r["bytes"] for r in records),
        "file_count": len(records),
        "files": records,
    }


def census(label: str, path: Path) -> dict:
    header, header_bytes, metadata = read_header(path)
    tensors = []
    for name, spec in header.items():
        start, end = spec["data_offsets"]
        shape = spec["shape"]
        stored = end - start
        bits = DTYPE_BITS[spec["dtype"]]
        elements = 1
        for dim in shape:
            elements *= dim
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "dtype": spec["dtype"],
                "bytes": stored,
                "stored_elements": elements,
                "bits_per_stored_element": bits,
            }
        )
    tensors.sort(key=lambda t: -t["bytes"])
    vocab_like = [
        t
        for t in tensors
        if VOCAB in t["shape"] or COMPACT_ROWS in t["shape"]
    ]
    return {
        "label": label,
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "header_bytes": header_bytes,
        "tensor_bytes": sum(t["bytes"] for t in tensors),
        "tensor_count": len(tensors),
        "metadata": metadata,
        "tensors": tensors,
        "vocabulary_sized_tensors": [t["name"] for t in vocab_like],
        "has_full_vocab_projection": any(VOCAB in t["shape"] for t in tensors),
        "has_compact_vocab_projection": any(COMPACT_ROWS in t["shape"] for t in tensors),
    }


def main() -> int:
    pinned = census("pinned_bf16", PINNED_DIR / "model.safetensors")
    declared = census("declared_q2_q4", DECLARED_DIR / "model.safetensors")

    manifest = json.loads((ROOT / "mtp-head.manifest.json").read_text())
    declared_tree = tree_digest(DECLARED_DIR)
    pinned_tree = tree_digest(PINNED_DIR)
    run_tree = tree_digest(DECLARED_RUN_DIR) if DECLARED_RUN_DIR.is_dir() else None

    result = {
        "pinned_head": pinned,
        "declared_head": declared,
        "manifest": {
            "source": manifest["source"],
            "source_url": manifest["source_url"],
            "sha256": manifest["sha256"],
            "bytes": manifest["bytes"],
            "max_bytes": manifest["max_bytes"],
        },
        "declared_tree_digest": declared_tree,
        "pinned_tree_digest": pinned_tree,
        "declared_run_tree_digest": run_tree,
        "tree_digest_recipe_verified": (
            declared_tree["sha256"] == manifest["sha256"]
            and declared_tree["bytes"] == manifest["bytes"]
        ),
        "heads_differ_local_default_vs_ranked": (
            pinned["file"] != declared["file"]
            and file_sha256(PINNED_DIR / "model.safetensors")
            != file_sha256(DECLARED_DIR / "model.safetensors")
        ),
        "declared_capacity_unused_bytes": manifest["max_bytes"] - manifest["bytes"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"pinned_head", "declared_head", "declared_tree_digest", "pinned_tree_digest", "declared_run_tree_digest"}}, indent=2))
    print(f"pinned:   {pinned['tensor_count']} tensors, {pinned['tensor_bytes']:,} tensor bytes")
    print(f"declared: {declared['tensor_count']} tensors, {declared['tensor_bytes']:,} tensor bytes")
    print(f"declared tree sha256 {declared_tree['sha256']} ({declared_tree['bytes']:,} bytes)")
    print(f"pinned   tree sha256 {pinned_tree['sha256']} ({pinned_tree['bytes']:,} bytes)")
    if run_tree:
        print(f"run tree sha256 {run_tree['sha256']} ({run_tree['bytes']:,} bytes, {run_tree['file_count']} files)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
