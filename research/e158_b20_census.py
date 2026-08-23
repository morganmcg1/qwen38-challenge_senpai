"""E158 B2.0: per-matrix precision census of the MTP head trunk.

For each of the eight quantized trunk projections, report the reconstruction
error of MLX's stock round-to-nearest affine quantizer at every candidate bit
width against the BF16 master, together with the bytes that width costs on
disk and the bytes it costs per draft step.

The census is offline numerics. It reads the two head artifacts and nothing
else, and it writes research/e158-artifacts/b20-census.json.

Two facts bound how the ranking may be used, and both are recorded in the
artifact:

  * The loader derives a submodule's quantization from the archive itself
    (`quantize(model:)` promotes a submodule to `QuantizedLinear` if and only
    if `<path>.scales` exists), but the bit width and group size come from the
    backbone `config.json`, which is `bits 4, group_size 64`. A trunk matrix
    therefore has exactly two legal forms: affine-4 group-64, or plain BF16.
    The 2, 3, 6 and 8-bit columns are numerics, not loadable options.
  * The whole-trunk BF16 upgrade -- every matrix at rel_l2 = 0 at once -- was
    measured end to end in E158 R1 and moved conditional exact-readout
    acceptance by -0.0073 pp. Any per-matrix subset of that upgrade is bounded
    by the whole.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e144_quant import _groups, metal_round, rtn_params  # noqa: E402
from e144_st import SafeTensors, bf16_to_f32, f32_to_bf16  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1")
MASTER = f"{CACHE}/mtp-head/model.safetensors"
DECLARED = f"{CACHE}/mtp-head-declared-run/model.safetensors"
GROUP = 64
WIDTHS = (2, 3, 4, 6, 8)

# Master tensor name -> declared core name.
CORE = {
    "fc.weight": "fc",
    "layers.0.self_attn.q_proj.weight": "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj.weight": "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj.weight": "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj.weight": "layers.0.self_attn.o_proj",
    "layers.0.mlp.gate_proj.weight": "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj.weight": "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj.weight": "layers.0.mlp.down_proj",
}


def quantized_bytes(rows, columns, bits, group_size=GROUP):
    """Packed codes plus BF16 scales and biases, as the archive stores them."""
    packed = rows * columns * bits // 8
    groups = rows * (columns // group_size)
    return {
        "packed": packed,
        "scales": groups * 2,
        "biases": groups * 2,
        "total": packed + 2 * groups * 2,
    }


def reconstruction(weight, bits, group_size=GROUP):
    """Dequantized weight under MLX's deployed (BF16 scale/bias) RTN affine."""
    rows, columns = weight.shape
    grouped = _groups(weight, group_size)
    n_bins = np.float32((1 << bits) - 1)
    scale, bias = rtn_params(grouped, bits)
    codes = np.minimum(metal_round((grouped - bias) / scale), n_bins)
    deployed_scale = bf16_to_f32(f32_to_bf16(scale)).reshape(-1, 1)
    deployed_bias = bf16_to_f32(f32_to_bf16(bias)).reshape(-1, 1)
    return (codes * deployed_scale + deployed_bias).reshape(rows, columns)


def rel_l2(reference, approximation):
    reference = reference.astype(np.float64)
    difference = reference - approximation.astype(np.float64)
    return float(np.sqrt((difference**2).sum()) / np.sqrt((reference**2).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "e158-artifacts",
            "b20-census.json",
        ),
    )
    parser.add_argument("--widths", default=",".join(str(b) for b in WIDTHS))
    args = parser.parse_args()
    widths = tuple(int(b) for b in args.widths.split(","))

    steps = json.load(
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "e158-artifacts",
                "draft-step-bytes.json",
            )
        )
    )
    read_now = steps["declared_head"]["trunk_reads_per_draft_step"]

    out = {
        "experiment": "e158",
        "rung": "B2.0",
        "harness": "local",
        "quantizer": "MLX stock round-to-nearest affine, group_size 64, BF16 scales and biases",
        "loader_legality": {
            "trunk_bit_width_source": "backbone weights/config.json (bits 4, group_size 64)",
            "legal_trunk_forms": ["affine-4 group-64", "bf16"],
            "note": (
                "2, 3, 6 and 8-bit columns are numerics only. The loader reads one "
                "bit width for every quantized submodule, so a per-matrix width "
                "other than 4 cannot be declared without a backbone config change, "
                "which is not an editable path."
            ),
        },
        "measured_bound_on_the_whole_upgrade": {
            "arm": "organizer-pinned BF16 head, every trunk matrix at rel_l2 = 0",
            "e158_head_precision_recoverable_pp": -0.0073,
            "source": "E158 R1, research/e158-artifacts/recall-{pinned,declared}.json",
        },
        "matrices": {},
    }

    master = SafeTensors(MASTER)
    declared = SafeTensors(DECLARED)
    declared_names = set(declared.names)

    for name, core in CORE.items():
        weight = master.float32(name)
        rows, columns = weight.shape
        bf16_bytes = rows * columns * 2
        entry = {
            "shape": [rows, columns],
            "elements": rows * columns,
            "bf16_bytes": bf16_bytes,
            "widths": {},
            "read_bytes_per_draft_step_now": sum(
                v
                for k, v in read_now.items()
                if k.startswith(core + ".") or k == core + ".weight"
            ),
        }
        for bits in widths:
            approximation = reconstruction(weight, bits)
            byte_model = quantized_bytes(rows, columns, bits)
            entry["widths"][f"a{bits}"] = {
                "rel_l2": rel_l2(weight, approximation),
                "bytes": byte_model["total"],
                "bytes_vs_bf16": byte_model["total"] / bf16_bytes,
            }
            del approximation
        entry["widths"]["bf16"] = {
            "rel_l2": 0.0,
            "bytes": bf16_bytes,
            "bytes_vs_bf16": 1.0,
        }
        # The only legal upgrade for a trunk matrix.
        a4 = entry["widths"]["a4"]
        entry["a4_to_bf16"] = {
            "rel_l2_removed": a4["rel_l2"],
            "extra_bytes_on_disk": bf16_bytes - a4["bytes"],
        }
        # Confirm the declared archive really stores this matrix at a4.
        entry["declared_stores_scales"] = (core + ".scales") in declared_names
        out["matrices"][core] = entry
        print(
            f"{core:34s} {rows:6d}x{columns:<6d} "
            + " ".join(
                f"a{b}={entry['widths'][f'a{b}']['rel_l2']:.5f}" for b in widths
            ),
            flush=True,
        )
        del weight

    # Rank the legal upgrade by error removed per byte added per draft step.
    upgrade = json.load(
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "e158-artifacts",
                "draft-step-bytes.json",
            )
        )
    )["bf16_trunk_upgrade"]["per_matrix"]
    ranking = []
    for core, entry in out["matrices"].items():
        short = core.replace("layers.0.", "")
        step_delta = upgrade.get(short, {}).get("delta_bytes")
        if step_delta is None:
            continue
        ranking.append(
            {
                "matrix": short,
                "rel_l2_at_a4": entry["widths"]["a4"]["rel_l2"],
                "delta_bytes_per_draft_step_to_bf16": step_delta,
                "rel_l2_removed_per_MB_per_step": (
                    entry["widths"]["a4"]["rel_l2"] / (step_delta / 1e6)
                    if step_delta
                    else None
                ),
            }
        )
    ranking.sort(
        key=lambda r: (r["rel_l2_removed_per_MB_per_step"] is None,
                       -(r["rel_l2_removed_per_MB_per_step"] or 0.0))
    )
    out["upgrade_ranking_error_removed_per_byte"] = ranking

    master.close()
    declared.close()
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"e158_b20_census: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
