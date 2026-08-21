#!/usr/bin/env python3
"""Ledger 199E -- the lossless (scale, bias) metadata cardinality census.

Affine-4 group-64 metadata is a bf16 scale and a bf16 bias for every 64 packed
weights, which is 4 bytes per 32 bytes of nibbles, about 10.5 % of the weight
stream. An 8-bit lookup table over the DISTINCT (scale, bias) pairs of a tensor
would shrink that metadata by 4x and would be LOSSLESS only if the per-tensor
pair cardinality is at most 256.

This script answers that question exactly. It reads the safetensors headers of
the transformed checkpoint directly, so it needs no MLX, no GPU and no model
load, and it counts BIT PATTERNS rather than decoded floats: two bf16 values are
the same value exactly when their 16 bits agree, apart from the two zeros, which
are folded together here.

  usage: research/e97_metadata_census.py [WEIGHTS_DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

NEG_ZERO = np.uint16(0x8000)
# Above this many (scale, bias) cells the dense-flag counter would allocate more
# than about 64 MB, so the sort-based counter is used instead.
DENSE_FLAG_LIMIT = 1 << 29


def read_header(path: pathlib.Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return header, 8 + length


def read_u16(path: pathlib.Path, entry: dict, data_start: int) -> np.ndarray:
    start, end = entry["data_offsets"]
    count = (end - start) // 2
    return np.memmap(
        path, dtype=np.uint16, mode="r", offset=data_start + start, shape=(count,)
    )


def fold_zero(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.uint16)
    out[out == NEG_ZERO] = 0
    return out


def distinct_u16(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.bincount(values, minlength=1 << 16)))


def distinct_pairs(scales: np.ndarray, biases: np.ndarray) -> int:
    """Exact number of distinct (scale, bias) bit-pattern pairs."""
    scale_map = np.bincount(scales, minlength=1 << 16) > 0
    bias_map = np.bincount(biases, minlength=1 << 16) > 0
    n_scales = int(scale_map.sum())
    n_biases = int(bias_map.sum())
    cells = n_scales * n_biases
    scale_index = np.cumsum(scale_map) - 1
    bias_index = np.cumsum(bias_map) - 1
    si = scale_index[scales].astype(np.int64)
    bi = bias_index[biases].astype(np.int64)
    flat = si * n_biases + bi
    if cells <= DENSE_FLAG_LIMIT:
        seen = np.zeros(cells, dtype=bool)
        seen[flat] = True
        return int(seen.sum())
    return int(np.unique(flat).size)


def census(weights_dir: pathlib.Path) -> dict:
    index_path = weights_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted({weights_dir / name for name in weight_map.values()})
    else:
        shards = sorted(weights_dir.glob("*.safetensors"))

    tensors: list[dict] = []
    for shard in shards:
        header, data_start = read_header(shard)
        prefixes = sorted(
            key[: -len(".scales")] for key in header if key.endswith(".scales")
        )
        for prefix in prefixes:
            scale_entry = header[f"{prefix}.scales"]
            bias_key = f"{prefix}.biases"
            if bias_key not in header:
                continue
            bias_entry = header[bias_key]
            scales = fold_zero(read_u16(shard, scale_entry, data_start))
            biases = fold_zero(read_u16(shard, bias_entry, data_start))
            if scales.size != biases.size:
                raise SystemExit(f"{prefix}: scale and bias counts differ")
            record = {
                "tensor": prefix,
                "shard": shard.name,
                "dtype": scale_entry["dtype"],
                "shape": scale_entry["shape"],
                "groups": int(scales.size),
                "distinct_scales": distinct_u16(scales),
                "distinct_biases": distinct_u16(biases),
                "distinct_pairs": distinct_pairs(scales, biases),
            }
            record["metadata_bytes"] = 4 * record["groups"]
            record["pairs_fit_u8"] = record["distinct_pairs"] <= 256
            tensors.append(record)
            print(
                f"{record['tensor']:<52} groups={record['groups']:>9} "
                f"scales={record['distinct_scales']:>6} "
                f"biases={record['distinct_biases']:>6} "
                f"pairs={record['distinct_pairs']:>9} "
                f"u8={'yes' if record['pairs_fit_u8'] else 'no'}",
                flush=True,
            )

    if not tensors:
        raise SystemExit(f"no quantized tensors under {weights_dir}")

    pairs = np.array([t["distinct_pairs"] for t in tensors])
    metadata_bytes = int(sum(t["metadata_bytes"] for t in tensors))
    fit_bytes = int(sum(t["metadata_bytes"] for t in tensors if t["pairs_fit_u8"]))
    summary = {
        "weights_dir": str(weights_dir),
        "tensor_count": len(tensors),
        "total_groups": int(sum(t["groups"] for t in tensors)),
        "metadata_bytes": metadata_bytes,
        "max_distinct_pairs": int(pairs.max()),
        "min_distinct_pairs": int(pairs.min()),
        "median_distinct_pairs": float(np.median(pairs)),
        "tensors_pairs_le_256": int((pairs <= 256).sum()),
        "fraction_tensors_le_256": float((pairs <= 256).mean()),
        "metadata_bytes_in_tensors_le_256": fit_bytes,
        "fraction_metadata_bytes_le_256": fit_bytes / metadata_bytes,
        "max_distinct_scales": int(max(t["distinct_scales"] for t in tensors)),
        "max_distinct_biases": int(max(t["distinct_biases"] for t in tensors)),
        # A byte-wide code replaces a 4-byte (scale, bias) pair, so a tensor
        # that fits saves three quarters of its metadata.
        "lossless_saving_bytes": int(0.75 * fit_bytes),
    }
    return {"summary": summary, "tensors": tensors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="?", default="weights")
    parser.add_argument("--json", default="research/out/e97-metadata-census.json")
    args = parser.parse_args()

    result = census(pathlib.Path(args.weights))
    out = pathlib.Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    summary = result["summary"]
    print("\n--- summary ---")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
