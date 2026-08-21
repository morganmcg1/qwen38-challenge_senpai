#!/usr/bin/env python3
"""E98 rung 0 item 5 -- can the (scale, bias) index be avoided entirely?

The advisor asks whether the bias is a function of the scale on a tensor. If
`distinct_pairs == distinct_scales`, every scale bit pattern implies exactly one
bias, so the kernel can key a table with the scale's own 16 bits and stop reading
`biases`. That removes the index, the bit-cast and the non-finite question.

The mirror question is `distinct_pairs == distinct_biases`, which would let the
kernel key on the bias and stop reading `scales`.

This reads the E97 census, which already counted all three cardinalities per
tensor, so no new pass over the checkpoint is needed.

  usage: research/e98_census_analysis.py [CENSUS_JSON] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st


def summarize(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": st.median(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "census", nargs="?", default="research/out/e97-census/census.json")
    parser.add_argument("--json", dest="out")
    args = parser.parse_args()

    census = json.loads(pathlib.Path(args.census).read_text())
    tensors = census["tensors"]

    bias_of_scale = [t for t in tensors if t["distinct_pairs"] == t["distinct_scales"]]
    scale_of_bias = [t for t in tensors if t["distinct_pairs"] == t["distinct_biases"]]

    # The LUT lives in the front of the same tensor's `biases` array, which holds
    # `groups` bf16 elements. It needs two magic words plus two words per entry.
    lut_overflow = [
        t for t in tensors if t["groups"] < 2 + 2 * t["distinct_pairs"]]
    index_overflow = [t for t in tensors if t["distinct_pairs"] > 65536]

    def family(name: str) -> str:
        if ".mlp." in name:
            return "mlp"
        if name.endswith("embed_tokens") or name.endswith("lm_head"):
            return "embed_or_head"
        return "other"

    families: dict[str, dict[str, int]] = {}
    for t in tensors:
        slot = families.setdefault(
            family(t["tensor"]), {"tensors": 0, "metadata_bytes": 0, "groups": 0})
        slot["tensors"] += 1
        slot["metadata_bytes"] += t["metadata_bytes"]
        slot["groups"] += t["groups"]

    report = {
        "tensors": len(tensors),
        "bias_is_function_of_scale": len(bias_of_scale),
        "scale_is_function_of_bias": len(scale_of_bias),
        "distinct_scales": summarize([t["distinct_scales"] for t in tensors]),
        "distinct_biases": summarize([t["distinct_biases"] for t in tensors]),
        "distinct_pairs": summarize([t["distinct_pairs"] for t in tensors]),
        "pairs_over_scales": summarize(
            [t["distinct_pairs"] / t["distinct_scales"] for t in tensors]),
        "pairs_over_biases": summarize(
            [t["distinct_pairs"] / t["distinct_biases"] for t in tensors]),
        "lut_overflow_tensors": len(lut_overflow),
        "index_overflow_tensors": len(index_overflow),
        "min_groups": min(t["groups"] for t in tensors),
        "max_lut_bytes": max((2 + 2 * t["distinct_pairs"]) * 2 for t in tensors),
        "total_lut_bytes": sum((2 + 2 * t["distinct_pairs"]) * 2 for t in tensors),
        "total_metadata_bytes": sum(t["metadata_bytes"] for t in tensors),
        "families": families,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
