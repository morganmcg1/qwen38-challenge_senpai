#!/usr/bin/env python3
"""E82 positive control: is `declared` exactly `mx.quantize(master, 64, 4)`?

Rung 0 measured `master-bf16` at 93.13 % pooled acceptance and `declared` at
92.31 %, and attributed the 0.82 pt gap to requantization damage. That
attribution only holds if the two heads really are parent and child. This
script tests it the strongest way available: quantize every BF16 trunk tensor
of the master with MLX's own affine estimator and compare the packed `uint32`
payload, the scales and the biases byte for byte against `declared`.

A bit-exact match proves two things at once. First, `declared` is the naive
requantization of these exact weights, so the 0.82 pt is the estimator's and
nothing else's. Second, the read, quantize and pack path in this repository
reproduces the shipped baseline before any of it is changed, which is the
control every rung-6 quantizer result will be measured against.

  python3 research/e82_quant_control.py --out research/e82-quant-control.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from e82_st import SafeTensors

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
MASTER = CACHE / "mtp-head"
DECLARED = CACHE / "mtp-head-declared-run"
GROUP_SIZE = 64
BITS = 4


def load_tree(directory: Path) -> dict[str, tuple[SafeTensors, str]]:
    out = {}
    for path in sorted(directory.rglob("*.safetensors")):
        st = SafeTensors(path)
        for name in st.names():
            out[name] = (st, name)
    return out


def bf16_to_mx(st: SafeTensors, name: str) -> mx.array:
    """Widen stored BF16 to float32 losslessly, then hand MLX a BF16 array.

    `e82_st` deliberately refuses to upcast so that byte-level provenance work
    stays honest, so the widening happens here and only here.
    """
    e = st.entries[name]
    assert e.dtype == "BF16", (name, e.dtype)
    u16 = np.asarray(st.raw(name)).view(np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(e.shape)
    return mx.array(f32).astype(mx.bfloat16)


def raw_bytes(a: mx.array) -> bytes:
    """Exact stored bytes, with no dtype conversion anywhere on the path."""
    return np.array(a.view(mx.uint8)).tobytes()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e82-quant-control.json")
    args = ap.parse_args()

    master = load_tree(MASTER)
    declared = load_tree(DECLARED)

    records = []
    for name in sorted(master):
        if not name.endswith(".weight"):
            continue
        stem = name[: -len(".weight")]
        if f"{stem}.scales" not in declared:
            continue  # norms and any bare-BF16 tensor stay unquantized
        st, _ = master[name]
        if st.entries[name].dtype != "BF16":
            continue

        w = bf16_to_mx(st, name)
        wq, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode="affine")
        mx.eval(wq, s, b)

        got = {
            "weight": raw_bytes(wq),
            "scales": raw_bytes(s),
            "biases": raw_bytes(b),
        }
        rec = {"tensor": stem, "shape": list(st.entries[name].shape), "parts": {}}
        for part, produced in got.items():
            dst, dname = declared[f"{stem}.{part}"]
            shipped = np.asarray(dst.raw(dname)).tobytes()
            rec["parts"][part] = {
                "bytes_ours": len(produced),
                "bytes_shipped": len(shipped),
                "sha256_ours": sha(produced),
                "sha256_shipped": sha(shipped),
                "identical": produced == shipped,
            }
        rec["identical"] = all(p["identical"] for p in rec["parts"].values())
        records.append(rec)
        flags = "".join("." if rec["parts"][p]["identical"] else "X"
                        for p in ("weight", "scales", "biases"))
        print(f"{stem.ljust(42)}{str(rec['shape']).ljust(16)}{flags}")

    verdict = bool(records) and all(r["identical"] for r in records)
    report = {
        "master_dir": str(MASTER),
        "declared_dir": str(DECLARED),
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "mode": "affine",
        "mlx_version": mx.__version__,
        "tensors_compared": len(records),
        "tensors_identical": sum(r["identical"] for r in records),
        "verdict_declared_is_mlx_quantize_of_master": verdict,
        "records": records,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n{report['tensors_identical']}/{len(records)} tensors bit-exact")
    print(f"VERDICT declared == mx.quantize(master, 64, 4, affine): {verdict}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
