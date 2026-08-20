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

# A head that ships no `draft_lm_head` does NOT skip the readout: the runtime
# derives a compact affine-4 g64 trim of the exact lm_head at warm and reads
# all of it on every draft step (`Qwen35.swift:3126-3153`,
# `compactDraftPaddedCount = 98_336`, hidden 5120). Those bytes are absent
# from the artifact but present in the traffic, so a bytes-per-millisecond
# fit that ignores them mis-prices the pinned head by a factor of 1.33.
COMPACT_ROWS = 98_336
COMPACT_BYTES = (COMPACT_ROWS * (5120 * 4 // 32) * 4   # u32 4-bit weight
                 + 2 * COMPACT_ROWS * (5120 // 64) * 2)  # bf16 scales + biases

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


QUANT_PARTS = (".weight", ".scales", ".biases")


def stream_bytes_by_precision(tensors: dict, derived: int) -> dict:
    """Split the per-draft read into bandwidth classes, not one flat total.

    A single bytes-per-millisecond law cannot fit these six heads: the
    controlled pinned/master-bf16 pair differs only in the readout and implies
    856 GB/s, which is above the machine's measured 226 GB/s peak. Bytes of
    2-bit weight cost more milliseconds than bytes of BF16 weight because each
    byte carries four times as many values to unpack, so the classes have to be
    priced separately.

    `precision_islands` is INCLUDED, and the enforcing source says why. The
    side-channel in `Qwen35TextModel.sanitize` installs the island rows on
    `mtp?.layers.first` and its own comment states "The target model never sees
    or consumes this artifact". `qkv(x)` then calls
    `replaceExactRows(y, input: x, kvOnly: false)`, which multiplies the whole
    concatenated 3072x5120 BF16 correction against the draft hidden state on
    every draft step. These are head bytes, read at head cadence.
    """
    quant: dict[str, dict[str, dict]] = {}
    for name, t in tensors.items():
        for part in QUANT_PARTS:
            if name.endswith(part):
                quant.setdefault(name[: -len(part)], {})[part[1:]] = t
                break

    classes: dict[str, int] = {}
    for name, t in tensors.items():
        module = next((name[: -len(p)] for p in QUANT_PARTS if name.endswith(p)), None)
        parts = quant.get(module, {}) if module else {}
        w, s = parts.get("weight"), parts.get("scales")
        if w is not None and s is not None and w["dtype"] == "U32":
            bits = w["shape"][1] * 32 // (s["shape"][1] * 64)
            label = f"q{bits}"
        else:
            # Unpacked reads are one class: what separates the classes is
            # how many values a delivered byte must be expanded into, and a
            # BF16 weight and an I32 index are both expanded into one.
            label = "dense"
        classes[label] = classes.get(label, 0) + t["bytes"]

    if derived:
        # The derived compact head is an affine-4 g64 trim built at warm time,
        # so it streams in the same class as a shipped 4-bit readout.
        classes["q4"] = classes.get("q4", 0) + derived
    return dict(sorted(classes.items()))


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
    tensor_bytes = sum(t["bytes"] for t in tensors.values())
    ships_readout = "draft_lm_head" in groups
    derived = 0 if ships_readout else COMPACT_BYTES
    streamed = stream_bytes_by_precision(tensors, derived)
    return {
        "path": str(directory),
        "tree_sha256": digest,
        "tree_bytes": total,
        "tensor_bytes": tensor_bytes,
        "safetensors_bytes": sum(p.stat().st_size for p in payload),
        "file_count": len(files),
        "tensor_count": len(tensors),
        "ships_draft_lm_head": ships_readout,
        "derived_compact_draft_head_bytes": derived,
        "traffic_bytes_per_draft": tensor_bytes + derived,
        "head_stream_bytes_by_precision": streamed,
        "head_stream_bytes": sum(streamed.values()),
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

    print("\n=== traffic read per draft step (artifact + derived readout) ===")
    print("arm".ljust(13) + "artifact tensors".rjust(18)
          + "derived compact".rjust(18) + "traffic".rjust(18) + "  ships readout")
    for arm, e in report["arms"].items():
        print(f"{arm.ljust(13)}{e['tensor_bytes']:>18,}"
              f"{e['derived_compact_draft_head_bytes']:>18,}"
              f"{e['traffic_bytes_per_draft']:>18,}  {e['ships_draft_lm_head']}")

    classes = sorted({c for e in report["arms"].values() for c in e["head_stream_bytes_by_precision"]})
    print("\n=== head stream bytes per draft by precision class (islands included) ===")
    print("arm".ljust(13) + "".join(c.rjust(18) for c in classes) + "total".rjust(18))
    for arm, e in report["arms"].items():
        s = e["head_stream_bytes_by_precision"]
        print(arm.ljust(13) + "".join(f"{s.get(c, 0):>18,}" for c in classes)
              + f"{e['head_stream_bytes']:>18,}")

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
