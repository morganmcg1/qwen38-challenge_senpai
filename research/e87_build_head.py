#!/usr/bin/env python3
"""E87 rung 2: build a proposal head whose coarse draft readout uses group 128.

Everything except `draft_lm_head.{weight,scales,biases}` is copied from the
declared head as raw bytes and asserted identical, so the only difference the
runtime can see is the coarse retrieval index.

`research/e87_head.py` proves the shipped coarse readout is exactly
`mx.quantize(dequantize(compact affine-4 lm_head rows), 64, 2)`. This script
applies the same rule at group 128, which halves the scale and bias tensors and
removes 15,733,760 bytes from every draft step.

  /opt/homebrew/bin/python3 research/e87_build_head.py --group-size 128
"""

from __future__ import annotations

import argparse
import json
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
COARSE = ("draft_lm_head.weight", "draft_lm_head.scales", "draft_lm_head.biases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-root", default=str(CACHE / "e87/built"))
    args = ap.parse_args()

    tag = args.tag or f"e87-coarse-g{args.group_size}"
    out_dir = Path(args.out_root) / f"{tag}-run"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = H.dequantized(H.load_exact())
    weight, scales, biases = mx.quantize(rows, group_size=args.group_size, bits=2)
    mx.eval(weight, scales, biases)
    new = {
        "draft_lm_head.weight": np.asarray(weight),
        "draft_lm_head.scales": np.asarray(scales.view(mx.uint16)),
        "draft_lm_head.biases": np.asarray(biases.view(mx.uint16)),
    }

    src = SafeTensors(DECLARED_RUN / "model.safetensors")
    tensors: dict[str, np.ndarray] = {}
    for name in src.names():
        tensors[name] = new[name] if name in COARSE else src.array(name)

    dest = out_dir / "model.safetensors"
    size = write_safetensors(dest, tensors, metadata=src.metadata)
    shutil.copyfile(DECLARED_RUN / "config.json", out_dir / "config.json")

    # Everything outside the coarse readout must be byte-identical.
    check = SafeTensors(dest)
    for name in src.names():
        if name in COARSE:
            continue
        assert check.sha256(name) == src.sha256(name), name

    digest, tree_bytes = tree_digest(out_dir)
    src_bytes = sum(e.nbytes for e in src.entries.values())
    dst_bytes = sum(e.nbytes for e in check.entries.values())
    report = {
        "tag": tag,
        "group_size": args.group_size,
        "dir": str(out_dir),
        "file_bytes": size,
        "file_sha256": file_sha256(dest),
        "head_provenance_sha256": digest,
        "tree_bytes": tree_bytes,
        "declared_tensor_bytes": src_bytes,
        "tensor_bytes": dst_bytes,
        "tensor_bytes_delta": dst_bytes - src_bytes,
        "coarse_shapes": {k: list(v.shape) for k, v in new.items()},
        "non_coarse_tensors_byte_identical": True,
    }
    print(json.dumps(report, indent=2))
    Path(f"research/e87-build-{tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
