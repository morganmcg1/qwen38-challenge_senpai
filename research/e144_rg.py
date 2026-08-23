"""E144 R-G: is metadata-free column permutation admissible on this artifact?

Advisor F4 names one lever that E82 rung 6 could not have tested, because that
census swept estimators with the grouping held FIXED. A permutation does not
change the estimator; it changes which 64 weights share a group. The CafeQ
identity `W1 @ W2 == (W1 @ Minv) @ (M @ W2)` makes it exactly invertible, free
in metadata and free at inference, provided the permuted dimension is private
to the head.

The advisor's gate is the statistic `std(log(per-column max-abs))` on the
grouped axis. This file reports that statistic, the supporting artifact facts
F4 asked for, and -- because it costs about a minute of CPU -- the DIRECTLY
MEASURED permutation gain, which settles the gate without relying on the
simulated exchange-rate table.

Data-free by construction: every quantity is a pure function of the master
weights. No calibration data, no activations, no training, no corpus.
"""

import argparse
import json
import os

import numpy as np

from e144_quant import dequantize, mlx_rtn
from e144_ra import CORE, DECLARED, MASTER, rel_l2
from e144_st import SafeTensors

GROUP = 64


def column_scale_spread(weight):
    """`std(log(per-column max-abs))` on the grouped axis, F4's deciding number."""
    magnitude = np.abs(weight.astype(np.float64)).max(axis=0)
    positive = magnitude[magnitude > 0]
    if positive.size == 0:
        return 0.0, 0
    return float(np.log(positive).std()), int(magnitude.size - positive.size)


def group_kurtosis(weight):
    """Median excess kurtosis over the 64-element groups actually quantized."""
    rows, columns = weight.shape
    groups = weight.astype(np.float64).reshape(rows * columns // GROUP, GROUP)
    centred = groups - groups.mean(axis=1, keepdims=True)
    variance = (centred**2).mean(axis=1)
    fourth = (centred**4).mean(axis=1)
    usable = variance > 0
    excess = fourth[usable] / variance[usable] ** 2 - 3.0
    return float(np.median(excess)), float(np.percentile(excess, 99))


def permutation_order(weight):
    """Sort columns by max-abs. A pure function of the weights."""
    return np.argsort(np.abs(weight.astype(np.float32)).max(axis=0), kind="stable")


def measured_permutation_gain(weight):
    """relL2 under the incumbent quantizer, natural order against sorted order."""
    natural = rel_l2(weight, dequantize(*mlx_rtn(weight, GROUP)))
    order = permutation_order(weight)
    permuted = np.ascontiguousarray(weight[:, order])
    sorted_error = rel_l2(permuted, dequantize(*mlx_rtn(permuted, GROUP)))
    return natural, sorted_error, natural / sorted_error


def coupled_pairs(shapes):
    """Which pairs share an intermediate dimension private to the head.

    A permutation only changes grouping when it moves the GROUPED axis, which
    is the last axis. In every pair below the shared dimension is the last axis
    of exactly one member, so the lever reaches that member only; its partner
    takes a row permutation, which leaves its grouping untouched.
    """
    pairs = []
    for label, produces, consumes, shared in (
        (
            "ffn intermediate",
            ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"],
            "layers.0.mlp.down_proj",
            "intermediate",
        ),
        (
            "attention value",
            ["layers.0.self_attn.v_proj"],
            "layers.0.self_attn.o_proj",
            "value",
        ),
    ):
        dimension = shapes[consumes][1]
        pairs.append(
            {
                "pair": label,
                "shared_dimension": shared,
                "size": dimension,
                "produced_by": produces,
                "consumed_by": consumes,
                "produced_by_axis": "rows (axis 0) — NOT the grouped axis",
                "consumed_by_axis": "columns (axis 1) — the grouped axis",
                "grouping_changes_on": [consumes],
                "grouping_unchanged_on": produces,
                "private_to_head": True,
                "shapes_agree": all(shapes[name][0] == dimension for name in produces),
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="e144-rg.json")
    parser.add_argument("--gate-low", type=float, default=0.15)
    parser.add_argument("--gate-high", type=float, default=0.40)
    arguments = parser.parse_args()

    master = SafeTensors(MASTER)
    declared = SafeTensors(DECLARED)

    shapes = {}
    tensors = {}
    for name in CORE:
        weight = np.ascontiguousarray(master.float32(f"{name}.weight"))
        shapes[name] = list(weight.shape)
        spread, zero_columns = column_scale_spread(weight)
        kurtosis_median, kurtosis_p99 = group_kurtosis(weight)
        natural, sorted_error, factor = measured_permutation_gain(weight)
        tensors[name] = {
            "shape": list(weight.shape),
            "e144_col_scale_log_std": spread,
            "zero_columns": zero_columns,
            "group_kurtosis_median": kurtosis_median,
            "group_kurtosis_p99": kurtosis_p99,
            "rel_l2_natural_order": natural,
            "rel_l2_sorted_columns": sorted_error,
            "e144_permutation_rel_l2_factor": factor,
            "groups_per_row": weight.shape[1] // GROUP,
        }
        print(
            "%-32s shape %5d x %5d  col_scale_log_std %.4f  kurt %7.3f  "
            "relL2 %.6e -> %.6e  x%.5f"
            % (
                name,
                weight.shape[0],
                weight.shape[1],
                spread,
                kurtosis_median,
                natural,
                sorted_error,
                factor,
            )
        )

    # F4 item 5: the metadata dtypes. If they are BF16 the group already owns a
    # free independent (s, z) and per-row rescaling is representationally
    # vacuous.
    metadata = {}
    for name in CORE:
        for field in ("scales", "biases"):
            key = f"{name}.{field}"
            dtype, shape, _ = declared.info(key)
            metadata[key] = {"dtype": dtype, "shape": list(shape)}
    grouped_axis_confirmed = all(
        declared.info(f"{name}.scales")[1][1] == shapes[name][1] // GROUP for name in CORE
    )

    pairs = coupled_pairs(shapes)
    admissible = sorted({name for pair in pairs for name in pair["grouping_changes_on"]})
    carrying = {
        name: tensors[name]["e144_col_scale_log_std"]
        for name in tensors
        if tensors[name]["rel_l2_natural_order"] > 0.05
    }
    worst_spread = max(carrying.values())
    admissible_spread = max(tensors[name]["e144_col_scale_log_std"] for name in admissible)
    admissible_factor = max(tensors[name]["e144_permutation_rel_l2_factor"] for name in admissible)

    report = {
        "experiment": "e144",
        "rung": "R-G",
        "harness": "local",
        "data_free": True,
        "group_size": GROUP,
        "tensors": tensors,
        "metadata_dtypes": metadata,
        "grouping_axis": {
            "axis": "last (input) dimension, blocks of 64",
            "confirmed_from_scales_shape": grouped_axis_confirmed,
        },
        "rounding": {
            "mode": "half away from zero (Metal round()), not half to even",
            "clamp": "[0, 15], asymmetric",
            "grid_anchor": "re-anchored on the larger-magnitude edge so 0.0 stays "
            "representable, which makes the stored scale negative on ~49.7 % of groups",
            "verified_by": "R-A bit-exact reproduction, e144_declared_reproduction_exact = 1.0",
        },
        "coupled_pairs": pairs,
        "admissible_tensors": admissible,
        "gate": {
            "statistic": "std(log(per-column max-abs)) on the grouped axis",
            "low": arguments.gate_low,
            "high": arguments.gate_high,
            "max_spread_on_error_carrying_tensors": worst_spread,
            "max_spread_on_admissible_tensors": admissible_spread,
            "max_measured_permutation_factor_on_admissible_tensors": admissible_factor,
            "rg_dead": admissible_spread < arguments.gate_low,
            "rg_reopens_axis": admissible_spread > arguments.gate_high,
        },
    }

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print()
    print(json.dumps(report["gate"], indent=2))
    print()
    print("metadata dtypes: " + ", ".join(sorted({m["dtype"] for m in metadata.values()})))
    print("grouped axis confirmed from scales shape: %s" % grouped_axis_confirmed)
    for pair in pairs:
        print(
            "coupled pair %-18s size %5d  grouping moves on %s"
            % (pair["pair"], pair["size"], ", ".join(pair["grouping_changes_on"]))
        )


if __name__ == "__main__":
    main()
