#!/usr/bin/env python3
"""E87 rung 2: build the proposal heads that carry the coarse-readout arms.

Everything the arm does not own is copied from the declared head as raw bytes
and asserted identical, so the only difference the runtime can see is the
coarse retrieval index.

`research/e87_head.py` proves the shipped coarse readout is exactly
`mx.quantize(dequantize(compact affine-4 lm_head rows), 64, 2)`.

Arm G applies that same rule at group 128, which halves the scale and bias
tensors and removes 15,733,760 bytes from every draft step:

  /opt/homebrew/bin/python3 research/e87_build_head.py --group-size 128

Arm C keeps `draft_lm_head.*` byte-identical and ADDS a cluster index under
`draft_cluster.*`: a 2-bit centroid table, the same coarse rows permuted into
contiguous per-cluster blocks, and the permuted-to-compact row map. The runtime
scores the centroids, probes the top `C` clusters and reads only those blocks:

  /opt/homebrew/bin/python3 research/e87_build_head.py \
      --cluster plain-k12292 --probe-fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e87_head as H  # noqa: E402
from e82_st import SafeTensors, file_sha256, tree_digest, write_safetensors  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
DECLARED_RUN = CACHE / "mtp-head-declared-run"
CLUSTER_DIR = CACHE / "e87/clusters"
COARSE = ("draft_lm_head.weight", "draft_lm_head.scales", "draft_lm_head.biases")


def _u16(a: mx.array) -> np.ndarray:
    return np.asarray(a.view(mx.uint16))


def cluster_tensors(rule_k: str, probe_fraction: float, centroid_bits: int) -> dict:
    """Arm C's `draft_cluster.*` block, plus the accounting the report needs.

    The permuted row table is the SAME coarse readout the declared head ships,
    reordered so cluster `c` owns rows `[c*rpc, (c+1)*rpc)`. `draft_cluster.perm`
    maps a permuted row back to its compact row. The six padding rows of the
    compact head duplicate compact rows 0..5, so their map entries point at
    those originals and a probe can never propose an out-of-range row.
    """
    table = np.load(CLUSTER_DIR / f"{rule_k}.npz")
    assign = table["assign"].astype(np.int32)
    probe = table["probe"].astype(np.float32)
    k, rpc = int(probe.shape[0]), int(table["rows_per_cluster"])
    if assign.shape[0] != k * rpc:
        raise SystemExit(f"{rule_k}: {assign.shape[0]} rows for {k}x{rpc}")
    counts = np.bincount(assign, minlength=k)
    if not np.all(counts == rpc):
        raise SystemExit(f"{rule_k}: ragged clusters, sizes {counts.min()}..{counts.max()}")

    order = np.argsort(assign, kind="stable").astype(np.int32)
    coarse = H.requantize(H.dequantized(H.load_exact()), 64, 2)
    idx = mx.array(order)
    wp = mx.take(coarse["weight"], idx, axis=0).reshape(k, rpc, 320)
    sp = mx.take(coarse["scales"], idx, axis=0).reshape(k, rpc, 80)
    bp = mx.take(coarse["biases"], idx, axis=0).reshape(k, rpc, 80)

    cent = H.requantize(mx.array(probe).astype(mx.bfloat16), 64, centroid_bits)
    mx.eval(wp, sp, bp, cent["weight"], cent["scales"], cent["biases"])

    perm = order.copy()
    pad = perm >= H.REAL_COUNT
    perm[pad] -= H.REAL_COUNT

    probed = max(1, math.ceil(probe_fraction * k))
    per_row = 320 * 4 + 80 * 2 * 2
    centroid_bytes = (
        int(np.asarray(cent["weight"]).nbytes)
        + _u16(cent["scales"]).nbytes
        + _u16(cent["biases"]).nbytes
    )
    return {
        "tensors": {
            "draft_cluster.rows.weight": np.asarray(wp),
            "draft_cluster.rows.scales": _u16(sp),
            "draft_cluster.rows.biases": _u16(bp),
            "draft_cluster.centroids.weight": np.asarray(cent["weight"]),
            "draft_cluster.centroids.scales": _u16(cent["scales"]),
            "draft_cluster.centroids.biases": _u16(cent["biases"]),
            "draft_cluster.perm": perm.astype(np.int32),
            "draft_cluster.shape": np.array([k, rpc, probed], dtype=np.int32),
        },
        "report": {
            "cluster_table": rule_k,
            "clusters": k,
            "rows_per_cluster": rpc,
            "probe_fraction": probe_fraction,
            "clusters_probed": probed,
            "rows_probed": probed * rpc,
            "centroid_bits": centroid_bits,
            "centroid_bytes": centroid_bytes,
            "stage2_bytes": probed * rpc * per_row,
            "dense_stage_bytes": H.COARSE_STAGE_BYTES,
            "padding_rows_remapped": int(pad.sum()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--cluster", default=None,
                    help="arm C: cluster table stem, e.g. plain-k12292")
    ap.add_argument("--probe-fraction", type=float, default=0.25)
    ap.add_argument("--centroid-bits", type=int, default=2)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-root", default=str(CACHE / "e87/built"))
    args = ap.parse_args()

    if args.cluster:
        built = cluster_tensors(args.cluster, args.probe_fraction, args.centroid_bits)
        new = built["tensors"]
        pct = f"{args.probe_fraction:g}".replace("0.", "")
        tag = args.tag or f"e87-armC-{args.cluster}-p{pct}"
        extra = built["report"]
    else:
        rows = H.dequantized(H.load_exact())
        weight, scales, biases = mx.quantize(rows, group_size=args.group_size, bits=2)
        mx.eval(weight, scales, biases)
        new = {
            "draft_lm_head.weight": np.asarray(weight),
            "draft_lm_head.scales": _u16(scales),
            "draft_lm_head.biases": _u16(biases),
        }
        tag = args.tag or f"e87-coarse-g{args.group_size}"
        extra = {"group_size": args.group_size}

    out_dir = Path(args.out_root) / f"{tag}-run"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = SafeTensors(DECLARED_RUN / "model.safetensors")
    tensors: dict[str, np.ndarray] = {}
    for name in src.names():
        tensors[name] = new[name] if name in COARSE else src.array(name)
    for name, value in new.items():
        if name not in tensors:
            tensors[name] = value

    dest = out_dir / "model.safetensors"
    size = write_safetensors(dest, tensors, metadata=src.metadata)
    shutil.copyfile(DECLARED_RUN / "config.json", out_dir / "config.json")

    # Every declared tensor the arm does not own must be byte-identical.
    check = SafeTensors(dest)
    for name in src.names():
        if name in new:
            continue
        assert check.sha256(name) == src.sha256(name), name

    digest, tree_bytes = tree_digest(out_dir)
    src_bytes = sum(e.nbytes for e in src.entries.values())
    dst_bytes = sum(e.nbytes for e in check.entries.values())
    report = {
        "tag": tag,
        "dir": str(out_dir),
        "file_bytes": size,
        "file_sha256": file_sha256(dest),
        "head_provenance_sha256": digest,
        "tree_bytes": tree_bytes,
        "declared_tensor_bytes": src_bytes,
        "tensor_bytes": dst_bytes,
        "tensor_bytes_delta": dst_bytes - src_bytes,
        "new_shapes": {k: list(v.shape) for k, v in new.items()},
        "declared_tensors_byte_identical": True,
        **extra,
    }
    print(json.dumps(report, indent=2))
    Path(f"research/e87-build-{tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
