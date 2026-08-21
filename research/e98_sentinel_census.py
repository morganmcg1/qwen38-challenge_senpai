#!/usr/bin/env python3
"""E98 rung 0: can a partly converted checkpoint be recognised by the kernel?

One compiled kernel serves every quantized tensor, so an MLP-only rollout has
to let the kernel tell a converted tensor from an untouched one at run time.
Two discriminators are possible and this census measures both exactly.

  1. MAGIC. Reserve `biases[0]` and `biases[1]` for a two-word magic value.
     This is only safe if no shipped tensor already begins with that pair.

  2. RANGE. A converted tensor stores a uint16 pair index in the `scales` slot.
     Indices run from 0 to `max(distinct_pairs) - 1`, which the E97 census puts
     at 7845 = 0x1EA5. Read as bf16 those bit patterns are denormal or tiny
     positives. Every shipped affine scale is a normal positive float near
     1e-2. If the smallest shipped scale bit pattern across the whole
     checkpoint is far above the largest index, then `scales[0]` alone
     identifies a converted tensor and no reserved magic word is needed.

The census reads the safetensors shards directly, so it needs no MLX, no GPU
and no model load, and it compares BIT PATTERNS rather than decoded floats.

  usage: research/e98_sentinel_census.py [WEIGHTS_DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e97_metadata_census import read_header, read_u16  # noqa: E402

# E97 census maximum distinct (scale, bias) pairs over 498 tensors.
MAX_PAIRS = 7846
MAX_INDEX_BITS = MAX_PAIRS - 1


def bf16(bits: int) -> float:
    return float(np.frombuffer(
        np.uint32(bits << 16).tobytes(), dtype=np.float32)[0])


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
            key[: -len(".scales")] for key in header if key.endswith(".scales"))
        for prefix in prefixes:
            if f"{prefix}.biases" not in header:
                continue
            scales = read_u16(shard, header[f"{prefix}.scales"], data_start)
            biases = read_u16(shard, header[f"{prefix}.biases"], data_start)
            record = {
                "tensor": prefix,
                "shard": shard.name,
                "groups": int(scales.size),
                "scale_bits_min": int(scales.min()),
                "scale_bits_max": int(scales.max()),
                "scale_negative_count": int(np.count_nonzero(scales >= 0x8000)),
                "scale_zero_count": int(np.count_nonzero(
                    (scales == 0) | (scales == 0x8000))),
                "bias_bits_min": int(biases.min()),
                "bias_bits_max": int(biases.max()),
                "bias_word0": int(biases[0]),
                "bias_word1": int(biases[1]),
                "scales_at_or_below_max_index": int(
                    np.count_nonzero(scales <= MAX_INDEX_BITS)),
            }
            tensors.append(record)
            print("%-52s groups=%9d scale_bits=[0x%04x,0x%04x] "
                  "bias0=0x%04x bias1=0x%04x collide=%d" % (
                      record["tensor"], record["groups"],
                      record["scale_bits_min"], record["scale_bits_max"],
                      record["bias_word0"], record["bias_word1"],
                      record["scales_at_or_below_max_index"]))
    return {"tensors": tensors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="weights")
    ap.add_argument("--json", dest="out")
    args = ap.parse_args()

    result = census(pathlib.Path(args.weights))
    tensors = result["tensors"]
    if not tensors:
        raise SystemExit("e98_sentinel_census: no quantized tensors found")

    global_scale_min = min(t["scale_bits_min"] for t in tensors)
    global_scale_max = max(t["scale_bits_max"] for t in tensors)
    range_collisions = sum(t["scales_at_or_below_max_index"] for t in tensors)
    first_words = {(t["bias_word0"], t["bias_word1"]) for t in tensors}

    # A magic value only has to avoid the observed first-word pairs. 0x7fc1 is
    # a bf16 quiet NaN payload; the transform writes it, no quantizer produces
    # it, and the pair is checked here against every shipped tensor.
    magic = (0x7FC1, 0xE98E)
    summary = {
        "tensors": len(tensors),
        "max_index_bits": MAX_INDEX_BITS,
        "max_index_as_bf16": bf16(MAX_INDEX_BITS),
        "global_scale_bits_min": global_scale_min,
        "global_scale_bits_min_as_bf16": bf16(global_scale_min),
        "global_scale_bits_max": global_scale_max,
        "global_scale_bits_max_as_bf16": bf16(global_scale_max),
        "scales_negative_total": sum(t["scale_negative_count"] for t in tensors),
        "scales_zero_total": sum(t["scale_zero_count"] for t in tensors),
        "range_discriminator_collisions": range_collisions,
        "range_discriminator_safe": range_collisions == 0,
        "range_margin_bits": global_scale_min - MAX_INDEX_BITS,
        "distinct_first_bias_word_pairs": len(first_words),
        "magic": list(magic),
        "magic_collides": magic in first_words,
    }
    result["summary"] = summary

    print()
    for key, value in summary.items():
        print("%-38s %s" % (key, value))

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
