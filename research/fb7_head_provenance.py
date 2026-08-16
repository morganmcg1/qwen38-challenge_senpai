#!/usr/bin/env python3
"""Identify the proposal head the local candidate leg actually loaded.

FB7 asks every result to record `head_provenance_sha256`, the resident head
directory, its byte count and its dtype, because `mtp-head.manifest.json`
declares a 4-bit/group-64 head while `setup-qwen-mtp.sh` hardcodes the bf16
head and never reads that manifest.

Three independent numbers describe the same tree and are routinely confused:

  file digest  -- sha256 of `model.safetensors` alone, which is what
                  `fixtures/qwen3_8_27b_mtp_head.sha256` pins.
  tree digest  -- the number the sealed report calls
                  `head_provenance_sha256`, defined by
                  `computeQwenMTPHeadProvenance` as sha256 over
                  `"<file sha256>  <relative path>\\n"` joined in sorted
                  relative-path order across every regular file except a
                  top-level README.md.
  payload size -- safetensors tensor bytes, which excludes the 8-byte length
                  prefix and the JSON header, so it is always smaller than
                  the file on disk.

This reproduces all three from the resident tree so the report can state which
head ran without asking the reader to trust a remembered digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

DEFAULT_HEAD_DIR = os.path.expanduser(
    "~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head"
)
CHUNK = 4 * 1024 * 1024


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def walk_regular_files(root: str) -> list[str]:
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            absolute = os.path.join(dirpath, name)
            if not os.path.isfile(absolute) or os.path.islink(absolute):
                continue
            found.append(os.path.relpath(absolute, root))
    return sorted(found)


def safetensors_header(path: str) -> dict:
    with open(path, "rb") as handle:
        raw_length = handle.read(8)
        header_bytes = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_bytes))
    metadata = header.pop("__metadata__", None)
    dtypes: dict[str, int] = {}
    payload_end = 0
    quant_keys: list[str] = []
    for name, spec in header.items():
        dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
        payload_end = max(payload_end, spec["data_offsets"][1])
        if name.endswith(".scales") or name.endswith(".biases"):
            quant_keys.append(name)
    return {
        "header_bytes": header_bytes,
        "tensor_count": len(header),
        "dtypes": dtypes,
        "metadata": metadata,
        "payload_bytes": payload_end,
        "prefix_and_header_bytes": 8 + header_bytes,
        "quantization_keys": sorted(quant_keys),
        "tensor_names": sorted(header.keys()),
    }


def read_pinned_fixture(path: str) -> dict[str, dict]:
    pinned: dict[str, dict] = {}
    if not os.path.exists(path):
        return pinned
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            digest, size, name = parts
            pinned[name] = {"sha256": digest, "bytes": int(size)}
    return pinned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dir", default=DEFAULT_HEAD_DIR)
    parser.add_argument(
        "--manifest", default="mtp-head.manifest.json"
    )
    parser.add_argument(
        "--pinned-fixture", default="fixtures/qwen3_8_27b_mtp_head.sha256"
    )
    parser.add_argument(
        "--reported-provenance",
        default=None,
        help="head_provenance_sha256 copied from a score report, to confirm "
        "the tree-digest rule reproduces it",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    head_dir = os.path.expanduser(args.head_dir)
    relatives = walk_regular_files(head_dir)

    files = []
    tree_hasher = hashlib.sha256()
    total_bytes = 0
    for relative in relatives:
        absolute = os.path.join(head_dir, relative)
        digest = sha256_of_file(absolute)
        size = os.path.getsize(absolute)
        # Mirrors computeQwenMTPHeadProvenance: a top-level README.md is the
        # only file excluded from the tree digest.
        counted = relative != "README.md"
        if counted:
            tree_hasher.update(f"{digest}  {relative}\n".encode())
            total_bytes += size
        files.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": size,
                "counted_in_tree_digest": counted,
            }
        )

    tree_digest = tree_hasher.hexdigest()

    weights = os.path.join(head_dir, "model.safetensors")
    header = safetensors_header(weights) if os.path.exists(weights) else None

    manifest = None
    if os.path.exists(args.manifest):
        with open(args.manifest, "r") as handle:
            manifest = json.load(handle)

    pinned = read_pinned_fixture(args.pinned_fixture)

    weights_entry = next(
        (f for f in files if f["path"] == "model.safetensors"), None
    )
    pinned_weights = pinned.get("model.safetensors")

    verdict = {
        "resident_matches_pinned_fixture": bool(
            weights_entry
            and pinned_weights
            and weights_entry["sha256"] == pinned_weights["sha256"]
            and weights_entry["bytes"] == pinned_weights["bytes"]
        ),
        "resident_matches_declared_manifest": bool(
            manifest
            and weights_entry
            and weights_entry["sha256"] == manifest.get("sha256")
        ),
        "resident_is_quantized": bool(
            header and header["quantization_keys"]
        ),
    }
    if manifest and weights_entry:
        declared = manifest.get("bytes")
        payload = header["payload_bytes"] if header else None
        if declared and payload:
            verdict["payload_over_declared_ratio"] = payload / declared
            verdict["per_forward_extra_bytes"] = payload - declared

    if args.reported_provenance:
        verdict["reported_provenance_reproduced"] = (
            args.reported_provenance == tree_digest
        )
        verdict["reported_provenance"] = args.reported_provenance

    result = {
        "head_directory": head_dir,
        "file_count": len(files),
        "files": files,
        "tree_digest_sha256": tree_digest,
        "tree_digest_bytes": total_bytes,
        "safetensors": header,
        "declared_manifest": manifest,
        "pinned_fixture": pinned,
        "verdict": verdict,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
