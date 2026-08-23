#!/usr/bin/env python3
"""E143 C2 rung 0: what does an affine-4 K/V island cost, and does an arm exist?

C2 replaces the dense BF16 `_exactKVDenseW` with an affine-4 group-64 pack:
20.97 MB of resident traffic per proposal step becomes 5.90 MB. The cost is
proposal quality, and Rule 107 says that cost must be measured live. Before
spending a GPU day on that, two cheap questions decide how to run it.

Q1. PROVENANCE. Are the declared head's own affine-4 `layers.0.{k,v}_proj`
    rows bit-identical to `quantize(precision_islands.{k,v}.weight, 64, 4)`?
    If yes, then C2's numerics already exist behind the shipped research
    selector `DARKBLOOM_QWEN_MTP_ISLAND_ARM=q`, and the acceptance half of C2
    can be measured with an environment variable and no source edit at all.

Q2. SIZE. What does the affine-4 pack actually weigh against the dense BF16
    form, and how large is the numerical perturbation it introduces at the
    K and V projection outputs?

Neither question is a substitute for the live acceptance measurement. They
only decide which instrument runs it.

Usage: research/e143_c2_head.py [--out research/e143-c2-head.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

GROUP = 64
BITS = 4
HEAD = Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run/model.safetensors"
K_OUT = 1024
V_OUT = 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def natural_order(weight: mx.array, indices: mx.array) -> mx.array:
    """Island rows put back in output order, exactly as `installExactQKVRows`.

    `argSort` of a permutation is its inverse, which is what the Swift does.
    """
    return mx.take(weight, mx.argsort(indices.astype(mx.int32)), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="research/e143-c2-head.json")
    args = parser.parse_args()

    head = mx.load(str(HEAD))
    prefix = "precision_islands."
    state = {
        "head_path": str(HEAD),
        "head_sha256": digest(HEAD),
        "island_keys": sorted(k for k in head if k.startswith(prefix)),
    }

    per_projection = {}
    for name, out_count in (("k", K_OUT), ("v", V_OUT)):
        island = head[f"{prefix}{name}.weight"]
        indices = head[f"{prefix}{name}.indices"]
        natural = natural_order(island, indices).astype(mx.bfloat16)
        mx.eval(natural)

        perm = np.sort(np.array(indices).astype(np.int64))
        complete = bool(perm.size == out_count
                        and np.array_equal(perm, np.arange(out_count)))

        qw, qs, qz = mx.quantize(natural, group_size=GROUP, bits=BITS)
        mx.eval(qw, qs, qz)
        back = mx.dequantize(qw, qs, qz, group_size=GROUP, bits=BITS)
        mx.eval(back)

        entry = {
            "island_shape": list(island.shape),
            "island_dtype": str(island.dtype),
            "indices_are_complete_permutation": complete,
            "dense_bf16_bytes": int(np.prod(natural.shape)) * 2,
            "affine4_bytes": (int(qw.size) * 4 + int(qs.size) * 2
                              + int(qz.size) * 2),
            "quantization_abs_error_max": float(
                mx.max(mx.abs(back.astype(mx.float32)
                              - natural.astype(mx.float32))).item()),
            "quantization_rel_error_rms": float(
                (mx.sqrt(mx.mean((back.astype(mx.float32)
                                  - natural.astype(mx.float32)) ** 2))
                 / mx.sqrt(mx.mean(natural.astype(mx.float32) ** 2))).item()),
        }

        # Q1: does the head already ship these rows in affine-4?
        shipped_key = f"layers.0.self_attn.{name}_proj"
        available = [k for k in head if k.startswith(shipped_key)]
        entry["shipped_projection_keys"] = sorted(available)
        if f"{shipped_key}.weight" in head:
            ship_w = head[f"{shipped_key}.weight"]
            entry["shipped_weight_shape"] = list(ship_w.shape)
            entry["shipped_weight_dtype"] = str(ship_w.dtype)
            if (f"{shipped_key}.scales" in head
                    and ship_w.shape == qw.shape and ship_w.dtype == qw.dtype):
                ship_s = head[f"{shipped_key}.scales"]
                ship_z = head[f"{shipped_key}.biases"]
                entry["shipped_is_quantized_island"] = {
                    "weight_bit_identical": bool(mx.all(ship_w == qw).item()),
                    "scales_bit_identical": bool(mx.all(ship_s == qs).item()),
                    "biases_bit_identical": bool(mx.all(ship_z == qz).item()),
                    "weight_mismatch_words": int(mx.sum(ship_w != qw).item()),
                }
            else:
                entry["shipped_is_quantized_island"] = None
        per_projection[name] = entry
        del island, indices, natural, qw, qs, qz, back

    state["per_projection"] = per_projection
    dense = sum(e["dense_bf16_bytes"] for e in per_projection.values())
    affine = sum(e["affine4_bytes"] for e in per_projection.values())
    state["kv_dense_bf16_bytes"] = dense
    state["kv_affine4_bytes"] = affine
    state["kv_saved_bytes"] = dense - affine
    state["kv_saved_mb"] = (dense - affine) / 1e6
    state["draft_step_budget_mb"] = 323.59
    state["kv_saved_share_of_draft_step"] = (dense - affine) / 1e6 / 323.59

    identity = [e.get("shipped_is_quantized_island") for e in per_projection.values()]
    state["env_arm_has_c2_numerics"] = bool(
        identity and all(i and i["weight_bit_identical"] and i["scales_bit_identical"]
                         and i["biases_bit_identical"] for i in identity))
    state["reading"] = (
        "env_arm_has_c2_numerics true means DARKBLOOM_QWEN_MTP_ISLAND_ARM=q "
        "already runs C2's exact K/V numerics, so the live acceptance delta "
        "needs no source edit; false means C2 must be implemented and gated")

    Path(args.out).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
