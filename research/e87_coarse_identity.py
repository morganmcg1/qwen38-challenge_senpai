#!/usr/bin/env python3
"""Check whether the shipped 2-bit g64 coarse table is bit-identical to a
recomputation from the exact 4-bit head.

`research/e87_screen.py` scores arm C from the shipped coarse tensors
(`H.load_coarse()`), while `research/e87_build_head.py` writes arm C rows from
`H.requantize(H.dequantized(H.load_exact()), 64, 2)`. The screen result
transfers to the built head only if those two tables agree. This script
measures the agreement instead of assuming it.

Run with the mlx-capable interpreter:

    /opt/homebrew/bin/python3 research/e87_coarse_identity.py \
        --out research/e87-coarse-identity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e87_head as H  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _u8(array: mx.array) -> np.ndarray:
    # numpy has no bfloat16, so reinterpret on the mlx side first.
    return np.asarray(array.view(mx.uint8)).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e87-coarse-identity.json")
    args = ap.parse_args()

    shipped = H.load_coarse()
    rebuilt = H.requantize(H.dequantized(H.load_exact()), 64, 2)

    report: dict[str, object] = {}
    all_identical = True
    for key in ("weight", "scales", "biases"):
        a, b = shipped[key], rebuilt[key]
        same_shape = tuple(a.shape) == tuple(b.shape)
        same_dtype = a.dtype == b.dtype
        entry: dict[str, object] = {
            "shipped_shape": list(a.shape),
            "rebuilt_shape": list(b.shape),
            "shipped_dtype": str(a.dtype),
            "rebuilt_dtype": str(b.dtype),
            "shape_match": same_shape,
            "dtype_match": same_dtype,
        }
        if same_shape and same_dtype:
            ba, bb = _u8(a), _u8(b)
            differing = int(np.count_nonzero(ba != bb))
            entry["byte_count"] = int(ba.size)
            entry["differing_bytes"] = differing
            entry["bit_identical"] = differing == 0
            if differing:
                fa = np.asarray(a.astype(mx.float32)).reshape(-1)
                fb = np.asarray(b.astype(mx.float32)).reshape(-1)
                entry["max_abs_diff"] = float(np.max(np.abs(fa - fb)))
            all_identical &= differing == 0
        else:
            entry["bit_identical"] = False
            all_identical = False
        report[key] = entry

    deq_shipped = H.dequantized(shipped)
    deq_rebuilt = H.dequantized(rebuilt)
    diff = np.asarray((deq_shipped - deq_rebuilt).astype(mx.float32))
    report["dequantized"] = {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rows_with_any_diff": int(np.count_nonzero(np.any(diff != 0, axis=1))),
        "row_count": int(diff.shape[0]),
    }
    report["all_bit_identical"] = bool(all_identical)

    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
