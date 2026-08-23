"""E144 R-A: reproduce the declared MTP head from master-bf16 bit-exactly.

Establishes the ancestry of all 40 declared tensors against the 15 master
tensors, and proves the recipe by re-deriving every quantized core tensor with
MLX's stock round-to-nearest affine quantizer and comparing packed words,
scales and biases bit-for-bit.
"""

import argparse
import json
import os

import numpy as np

from e144_quant import dequantize, mlx_rtn
from e144_st import SafeTensors

CACHE = os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1")
MASTER = f"{CACHE}/mtp-head/model.safetensors"
DECLARED = f"{CACHE}/mtp-head-declared-run/model.safetensors"

# The eight affine-4 group-64 core projections the head actually quantizes.
CORE = [
    "fc",
    "layers.0.mlp.down_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.v_proj",
]


def rel_l2(reference, reconstruction):
    reference = reference.astype(np.float64)
    difference = reference - reconstruction.astype(np.float64)
    return float(np.sqrt((difference**2).sum()) / np.sqrt((reference**2).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="e144-ra.json")
    arguments = parser.parse_args()

    master = SafeTensors(MASTER)
    declared = SafeTensors(DECLARED)

    report = {
        "experiment": "e144",
        "rung": "R-A",
        "harness": "local",
        "master": {"path": MASTER, "bytes": os.path.getsize(MASTER), "tensors": len(master.names)},
        "declared": {
            "path": DECLARED,
            "bytes": os.path.getsize(DECLARED),
            "tensors": len(declared.names),
            "metadata": declared.metadata,
        },
        "tensors": {},
    }

    exact_words = 0
    total_words = 0

    for name in CORE:
        weight = master.float32(f"{name}.weight")
        rows, columns = weight.shape
        packed, scales, biases, _ = mlx_rtn(weight)

        declared_packed = declared.raw(f"{name}.weight")
        declared_scales = declared.raw(f"{name}.scales")
        declared_biases = declared.raw(f"{name}.biases")

        word_match = float((packed == declared_packed).mean())
        exact_words += int((packed == declared_packed).sum())
        total_words += packed.size

        mine = dequantize(packed, scales, biases, 4, 64, columns)
        theirs = dequantize(declared_packed, declared_scales, declared_biases, 4, 64, columns)

        report["tensors"][name] = {
            "kind": "core-affine4-g64",
            "ancestor": f"{name}.weight (master-bf16)",
            "shape": [rows, columns],
            "packed_word_match": word_match,
            "scales_match": float((scales == declared_scales).mean()),
            "biases_match": float((biases == declared_biases).mean()),
            "rel_l2_reproduction_vs_master": rel_l2(weight, mine),
            "rel_l2_declared_vs_master": rel_l2(weight, theirs),
            "negative_scale_fraction": float((declared_scales.astype(np.uint16) >> 15).mean()),
        }
        print(
            f"{name:32s} words {word_match:.8f} "
            f"scales {report['tensors'][name]['scales_match']:.8f} "
            f"biases {report['tensors'][name]['biases_match']:.8f} "
            f"relL2 {report['tensors'][name]['rel_l2_declared_vs_master']:.4e}"
        )
        del weight, packed, mine, theirs

    # Pass-through BF16 norms: expected byte-identical to master.
    passthrough = [
        n
        for n in declared.names
        if n.endswith(".weight")
        and not n.startswith("precision_islands")
        and n.rsplit(".weight", 1)[0] not in CORE
        and n != "draft_lm_head.weight"
    ]
    for name in passthrough:
        same = bool(np.array_equal(declared.raw(name), master.raw(name)))
        report["tensors"][name] = {
            "kind": "passthrough-bf16",
            "ancestor": f"{name} (master-bf16)",
            "byte_identical_to_master": same,
        }
        print(f"{name:32s} passthrough byte-identical={same}")

    # Precision islands: BF16 rows of the master projection at inherited indices.
    for axis in ("q", "k", "v"):
        indices = declared.raw(f"precision_islands.{axis}.indices").astype(np.int64)
        island = declared.raw(f"precision_islands.{axis}.weight")
        source = master.raw(f"layers.0.self_attn.{axis}_proj.weight")
        same = bool(np.array_equal(island, source[indices]))
        report["tensors"][f"precision_islands.{axis}"] = {
            "kind": "island-bf16-rows",
            "ancestor": f"layers.0.self_attn.{axis}_proj.weight rows (master-bf16)",
            "rows": int(indices.size),
            "source_rows": int(source.shape[0]),
            "covers_all_rows": bool(indices.size == source.shape[0]),
            "rows_byte_identical_to_master": same,
        }
        print(
            f"precision_islands.{axis:22s} rows {indices.size}/{source.shape[0]} "
            f"byte-identical={same}"
        )

    # draft_lm_head: affine-2 group-64 compact readout, absent from master.
    scales_shape = declared.info("draft_lm_head.scales")[1]
    packed_shape = declared.info("draft_lm_head.weight")[1]
    logical_columns = scales_shape[1] * 64
    report["tensors"]["draft_lm_head"] = {
        "kind": "affine2-g64-compact-readout",
        "ancestor": None,
        "no_master_ancestor": True,
        "rows": packed_shape[0],
        "logical_columns": logical_columns,
        "bits": int(32 * packed_shape[1] / logical_columns),
        "note": (
            "master-bf16 carries no readout tensor at all, so this triplet cannot be "
            "derived from it. It comes from the metadata's second ancestor, "
            "'base': dwsdubey/qwen3.8-27b-mtp-4bit."
        ),
    }
    print(
        f"draft_lm_head                    affine-{report['tensors']['draft_lm_head']['bits']} "
        f"g64 rows={packed_shape[0]} cols={logical_columns} NO master ancestor"
    )

    core_exact = all(
        report["tensors"][n]["packed_word_match"] == 1.0
        and report["tensors"][n]["scales_match"] == 1.0
        and report["tensors"][n]["biases_match"] == 1.0
        for n in CORE
    )
    passthrough_exact = all(report["tensors"][n]["byte_identical_to_master"] for n in passthrough)
    islands_exact = all(
        report["tensors"][f"precision_islands.{a}"]["rows_byte_identical_to_master"]
        for a in ("q", "k", "v")
    )

    report["summary"] = {
        "e144_declared_reproduction_exact": 1.0 if core_exact else exact_words / total_words,
        "core_tensors_bit_exact": core_exact,
        "passthrough_tensors_bit_exact": passthrough_exact,
        "island_tensors_bit_exact": islands_exact,
        "core_packed_words_matched": exact_words,
        "core_packed_words_total": total_words,
        "tensors_with_master_ancestor": len(CORE) * 3 + len(passthrough) + 6,
        "tensors_without_master_ancestor": 3,
        "rel_l2_declared_range": [
            min(report["tensors"][n]["rel_l2_declared_vs_master"] for n in CORE),
            max(report["tensors"][n]["rel_l2_declared_vs_master"] for n in CORE),
        ],
    }

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("\n" + json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
