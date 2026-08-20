#!/usr/bin/env python3
"""E82: where every byte of each MTP head goes, and what the pinned head is.

The advisor asked for a per-head byte split across `fc`, the decoder trunk,
`draft_lm_head`, and the precision islands, because head cost is now a
first-class outcome: 1 % of candidate time is worth 1.000 % of official score.
Bytes are not time, but on this head the once-per-round flush is dominated by
weight traffic, so bytes per group is the cheapest honest proxy to publish
next to a measured step time.

This also settles a provenance question the screen raised. The organizer pins
the head to `EigenLabs/Qwen3.8-27B-MTP-bf16 @ 26a328e0` and ships a byte
manifest for it in `fixtures/qwen3_8_27b_mtp_head.sha256`, so the identity
claim is checkable without downloading anything: verify the resident pinned
tree against that manifest file by file.

  python3 research/e82_head_bytes.py --out research/e82-head-bytes.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from e82_st import SafeTensors, file_sha256, tree_digest

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
ARMS = {
    "pinned": CACHE / "mtp-head",
    "declared": CACHE / "mtp-head-declared-run",
    "soup-q4": CACHE / "e82/built/e82-soup-q4-run",
    "qat-q4": CACHE / "e82/built/e82-qat-q4-run",
    "master-bf16": CACHE / "e82/built/e82-master-bf16-run",
    "kamciosz": CACHE / "e82/built/e82-kamciosz-run",
}
MANIFEST = Path("fixtures/qwen3_8_27b_mtp_head.sha256")

# `fc` is the once-per-round join of embedding and hidden state; the trunk is
# the single decoder layer that follows it; `draft_lm_head` is the readout over
# the 248,320-entry vocabulary; islands are the BF16 rows kept exact.
GROUPS = (
    ("draft_lm_head", re.compile(r"(^|\.)draft_lm_head\b")),
    ("precision_islands", re.compile(r"(^|\.)precision_islands\b")),
    ("fc", re.compile(r"(^|\.)fc\b")),
    ("trunk", re.compile(r".")),
)


def group_of(name: str) -> str:
    for label, pattern in GROUPS:
        if pattern.search(name):
            return label
    raise AssertionError(name)


def split_tree(directory: Path) -> dict:
    groups: dict[str, dict] = {}
    tensors = {}
    for path in sorted(directory.rglob("*.safetensors")):
        st = SafeTensors(path)
        for name in st.names():
            e = st.entries[name]
            g = groups.setdefault(group_of(name), {"bytes": 0, "tensors": 0, "dtypes": set()})
            g["bytes"] += e.nbytes
            g["tensors"] += 1
            g["dtypes"].add(e.dtype)
            tensors[name] = {"dtype": e.dtype, "shape": list(e.shape), "bytes": e.nbytes}
    for g in groups.values():
        g["dtypes"] = sorted(g["dtypes"])

    files = sorted(p for p in directory.rglob("*") if p.is_file())
    # The lock file is runtime state the fetcher writes, not part of the head.
    payload = [p for p in files if p.suffix == ".safetensors"]
    digest, total = tree_digest(directory)
    return {
        "path": str(directory),
        "tree_sha256": digest,
        "tree_bytes": total,
        "tensor_bytes": sum(t["bytes"] for t in tensors.values()),
        "safetensors_bytes": sum(p.stat().st_size for p in payload),
        "file_count": len(files),
        "tensor_count": len(tensors),
        "groups": groups,
        "tensors": tensors,
    }


def verify_pinned_against_fixture(directory: Path) -> dict:
    """The fixture manifest is the organizer's own record of the EigenLabs
    master. Matching it byte for byte is the identity proof; nothing else in
    this repository asserts what the pinned head actually contains."""
    records = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, size, rel = line.split(None, 2)
        records.append((sha, int(size), rel))
    checked = []
    for sha, size, rel in records:
        p = directory / rel
        if not p.exists():
            checked.append({"path": rel, "present": False})
            continue
        checked.append({
            "path": rel,
            "present": True,
            "bytes_match": p.stat().st_size == size,
            "sha256_match": file_sha256(p) == sha,
            "expected_bytes": size,
        })
    return {
        "manifest": str(MANIFEST),
        "manifest_sha256": file_sha256(MANIFEST),
        "records": checked,
        "all_match": all(c.get("sha256_match") and c.get("bytes_match") for c in checked),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e82-head-bytes.json")
    args = ap.parse_args()

    report = {"arms": {}, "pinned_vs_fixture_manifest": verify_pinned_against_fixture(ARMS["pinned"])}
    for arm, directory in ARMS.items():
        if not directory.exists():
            raise SystemExit(f"missing arm tree: {directory}")
        report["arms"][arm] = split_tree(directory)

    v = report["pinned_vs_fixture_manifest"]
    print(f"pinned tree vs {v['manifest']}: all_match={v['all_match']}")
    for c in v["records"]:
        print(f"  {c['path']}: {c}")

    order = ["fc", "trunk", "draft_lm_head", "precision_islands"]
    print("\n=== head byte split (safetensors tensor payload) ===")
    print("arm".ljust(13) + "tree bytes".rjust(13) + "".join(g.rjust(20) for g in order) + "  tensors")
    for arm, e in report["arms"].items():
        cells = ""
        for g in order:
            got = e["groups"].get(g)
            cells += (f"{got['bytes']:>13,} " + f"{'/'.join(d.lower() for d in got['dtypes']):>5}" if got
                      else "                   -")
        print(f"{arm.ljust(13)}{e['tree_bytes']:>13,}{cells}  {e['tensor_count']:>3}")

    print("\n=== share of tensor payload ===")
    print("arm".ljust(13) + "".join(g.rjust(20) for g in order))
    for arm, e in report["arms"].items():
        tot = e["tensor_bytes"]
        print(arm.ljust(13) + "".join(
            f"{100 * e['groups'][g]['bytes'] / tot:19.1f}%" if g in e["groups"] else "                   -"
            for g in order))

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
