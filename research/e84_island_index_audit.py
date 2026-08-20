#!/usr/bin/env python3
"""E84 rung 0: verify the precision-island index claim from the declared head.

Reads the safetensors header and the raw index tensors out of the locally
verified declared-head tree, then answers, per tensor: is it a complete
permutation of 0 ..< n, and is it already in natural order?
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

DEFAULT = os.path.expanduser(
    "~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared/model.safetensors"
)
DTYPES = {"I32": np.int32, "I64": np.int64, "BF16": np.uint16, "F32": np.float32}


def load_header(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)), 8 + n


def read_tensor(path, meta, base):
    start, end = meta["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + start)
        raw = fh.read(end - start)
    return np.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]), raw


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    header, base = load_header(path)
    print(f"file: {path}")
    print(f"file sha256: {hashlib.sha256(open(path,'rb').read()).hexdigest()}")

    island = sorted(k for k in header if "precision_islands" in k)
    for key in island:
        meta = header[key]
        print(f"  {key}: dtype={meta['dtype']} shape={meta['shape']}")

    report = {}
    for name in ("q", "k", "v"):
        ikey = f"precision_islands.{name}.indices"
        wkey = f"precision_islands.{name}.weight"
        idx, raw = read_tensor(path, header[ikey], base)
        idx = np.asarray(idx, dtype=np.int64)
        n = idx.size
        distinct = np.unique(idx).size
        complete = bool(distinct == n and idx.min() == 0 and idx.max() == n - 1)
        natural = bool(np.array_equal(idx, np.arange(n)))
        wshape = header[wkey]["shape"]
        report[name] = dict(
            count=n,
            min=int(idx.min()),
            max=int(idx.max()),
            distinct=int(distinct),
            complete_permutation=complete,
            natural_order=natural,
            weight_shape=wshape,
            weight_dtype=header[wkey]["dtype"],
            first12=[int(x) for x in idx[:12]],
            sha256_indices=hashlib.sha256(raw).hexdigest(),
        )
        print(json.dumps({name: report[name]}, indent=2))

    print("\nSummary")
    for name, r in report.items():
        print(
            f"  {name}: count={r['count']} range=[{r['min']},{r['max']}] "
            f"distinct={r['distinct']} complete={r['complete_permutation']} "
            f"natural={r['natural_order']} weight={r['weight_shape']} "
            f"{r['weight_dtype']}"
        )


if __name__ == "__main__":
    main()
