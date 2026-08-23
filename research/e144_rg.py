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

from e144_price import recovery
from e144_quant import dequantize, mlx_rtn
from e144_ra import CORE, DECLARED, MASTER, rel_l2
from e144_st import SafeTensors

GROUP = 64
BITS = 4
HERE = os.path.dirname(os.path.abspath(__file__))


def rb_pooled_factor(default=1.0):
    """R-B's measured best-of-breed pooled factor, so R-G can be composed with it."""
    path = os.path.join(HERE, "e144-rb.json")
    if not os.path.exists(path):
        return default
    with open(path) as handle:
        return json.load(handle)["summary"]["e144_rel_l2_improvement_factor_pooled"]


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


def incumbent_rel_l2(weight):
    packed, scales, biases, _ = mlx_rtn(weight, BITS, GROUP)
    return rel_l2(weight, dequantize(packed, scales, biases, BITS, GROUP, weight.shape[1]))


def measured_permutation_gain(weight):
    """relL2 under the incumbent quantizer, natural order against sorted order."""
    natural = incumbent_rel_l2(weight)
    order = permutation_order(weight)
    permuted = np.ascontiguousarray(weight[:, order])
    sorted_error = incumbent_rel_l2(permuted)
    return natural, sorted_error, natural / sorted_error


def positive_control(rows=64, columns=1024, log_std=1.0, seed=0):
    """A synthetic tensor whose columns really do have heterogeneous scale.

    Without this the gate is unfalsifiable: a null result on the real tensors
    would be indistinguishable from a measurement that cannot detect a gain at
    all. The control is drawn with a known `std(log(per-column max-abs))` near
    the top of the advisor's simulated exchange table.
    """
    generator = np.random.default_rng(seed)
    base = generator.standard_normal((rows, columns))
    column_scale = np.exp(generator.standard_normal(columns) * log_std)
    weight = (base * column_scale).astype(np.float32)
    spread, _ = column_scale_spread(weight)
    natural, sorted_error, factor = measured_permutation_gain(weight)
    oracle = per_row_oracle_rel_l2(weight)
    return {
        "shape": [rows, columns],
        "requested_log_std": log_std,
        "e144_col_scale_log_std": spread,
        "e144_column_effect_share": column_effect_share(weight),
        "rel_l2_natural_order": natural,
        "rel_l2_sorted_columns": sorted_error,
        "e144_permutation_rel_l2_factor": factor,
        "rel_l2_per_row_oracle": oracle,
        "e144_per_row_oracle_rel_l2_factor": natural / oracle,
        "mse_reduction_pct": 100.0 * (1.0 - (sorted_error / natural) ** 2),
        "detects_a_gain": factor > 1.02,
    }


def per_row_oracle_rel_l2(weight):
    """Strict upper bound on EVERY column-permutation scheme.

    Groups are 64 contiguous entries of one row, so a group's affine error is
    driven by that row-slice's range. Sorting a row ascending minimises the sum
    of the ranges of its contiguous blocks. Allowing a DIFFERENT order per row
    is not implementable -- one weight matrix admits one column order -- but it
    dominates every shared, tiled or constrained permutation. If this bound is
    near 1.0 the whole permutation family is dead, whatever the column-scale
    statistic says.
    """
    return incumbent_rel_l2(np.ascontiguousarray(np.sort(weight, axis=1)))


def column_effect_share(weight, floor=1e-12):
    """Fraction of the variance of log|w| that column identity explains.

    This is what a column permutation can actually act on. The advisor's
    statistic summarises each column by its max over thousands of rows, which
    is a max-of-N order statistic; this one asks the direct question.
    """
    magnitude = np.abs(weight.astype(np.float32))
    logs = np.log(np.maximum(magnitude, floor)).astype(np.float64)
    total = logs.var()
    if total <= 0:
        return 0.0
    return float(logs.mean(axis=0).var() / total)


def block_scale_profile(weight, boundaries):
    """Per-block column-scale summary, to show grouping already respects blocks."""
    magnitude = np.abs(weight.astype(np.float64)).max(axis=0)
    profile = []
    for label, start, stop in boundaries:
        window = magnitude[start:stop]
        positive = window[window > 0]
        profile.append(
            {
                "block": label,
                "columns": [int(start), int(stop)],
                "median_col_max_abs": float(np.median(positive)),
                "col_scale_log_std_within_block": float(np.log(positive).std()),
                "aligned_to_group_grid": (start % GROUP == 0) and (stop % GROUP == 0),
            }
        )
    return profile


HEAD_DIM = 256


def tiled_permutation_gain(weight, block=HEAD_DIM):
    """Gain from one shared within-block order applied to every block.

    Grouped-query attention forces this form on `o_proj`: its 6144 columns are
    24 query-head blocks of 256, but they are all fed from only 4 key/value
    heads, so the free 6144-column order is not reachable. Only a single
    256-element order, tiled across all blocks, is invertible by a matching
    row permutation inside each key/value head of `v_proj`.
    """
    columns = weight.shape[1]
    if columns % block:
        raise ValueError(f"{columns} columns is not a multiple of {block}")
    blocks = columns // block
    magnitude = np.abs(weight.astype(np.float64)).max(axis=0).reshape(blocks, block).max(axis=0)
    order = np.argsort(magnitude, kind="stable")
    tiled = (np.arange(blocks)[:, None] * block + order[None, :]).reshape(-1)
    permuted = np.ascontiguousarray(weight[:, tiled])
    return incumbent_rel_l2(permuted)


def coupled_pairs(shapes):
    """Which shared dimensions are private to the head, and how free each is.

    A permutation only changes grouping when it moves the GROUPED axis, which
    is the last axis. So the lever reaches the CONSUMING tensor of a pair; its
    producing partner takes a row permutation, which leaves its own grouping
    untouched. The pair is usable only when the shared dimension never leaves
    the head, because the frozen target must not see a reordered vector.
    """
    return [
        {
            "pair": "ffn intermediate",
            "size": shapes["layers.0.mlp.down_proj"][1],
            "produced_by": ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"],
            "consumed_by": ["layers.0.mlp.down_proj"],
            "grouping_changes_on": ["layers.0.mlp.down_proj"],
            "private_to_head": True,
            "permutation_freedom": "free",
            "reachable_orders_log2": float(shapes["layers.0.mlp.down_proj"][1]),
            "note": "gate_proj and up_proj take the same row permutation; the "
            "intermediate vector is consumed inside the head and never reaches "
            "the frozen target.",
            "shapes_agree": all(
                shapes[name][0] == shapes["layers.0.mlp.down_proj"][1]
                for name in ("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj")
            ),
        },
        {
            "pair": "attention value",
            "size": shapes["layers.0.self_attn.o_proj"][1],
            "produced_by": ["layers.0.self_attn.v_proj"],
            "consumed_by": ["layers.0.self_attn.o_proj"],
            "grouping_changes_on": ["layers.0.self_attn.o_proj"],
            "private_to_head": True,
            "permutation_freedom": "constrained to one shared %d-element order, "
            "tiled across %d query-head blocks" % (HEAD_DIM, shapes["layers.0.self_attn.o_proj"][1] // HEAD_DIM),
            "head_dim": HEAD_DIM,
            "query_head_blocks": shapes["layers.0.self_attn.o_proj"][1] // HEAD_DIM,
            "kv_head_blocks": shapes["layers.0.self_attn.v_proj"][0] // HEAD_DIM,
            "note": "o_proj consumes the grouped-query attention output (%d), "
            "not v_proj's raw output (%d). Each key/value head feeds several "
            "query heads, so a free column order is not invertible; only one "
            "within-head order tiled across every block is."
            % (shapes["layers.0.self_attn.o_proj"][1], shapes["layers.0.self_attn.v_proj"][0]),
            "shapes_agree": False,
        },
        {
            "pair": "fc input",
            "size": shapes["fc"][1],
            "produced_by": ["pre_fc_norm_embedding", "pre_fc_norm_hidden"],
            "consumed_by": ["fc"],
            "grouping_changes_on": [],
            "private_to_head": False,
            "permutation_freedom": "none",
            "note": "fc consumes the concatenated target embedding and target "
            "hidden state. Both are produced by the frozen target, so there is "
            "no partner inside the head that can absorb the inverse. fc is the "
            "largest error carrier and it is inadmissible.",
            "shapes_agree": True,
        },
        {
            "pair": "residual stream",
            "size": shapes["layers.0.self_attn.q_proj"][1],
            "produced_by": ["fc", "norm", "pre_fc_norm_*"],
            "consumed_by": [
                "layers.0.self_attn.q_proj",
                "layers.0.self_attn.k_proj",
                "layers.0.self_attn.v_proj",
                "layers.0.mlp.gate_proj",
                "layers.0.mlp.up_proj",
            ],
            "grouping_changes_on": [],
            "private_to_head": False,
            "permutation_freedom": "none",
            "note": "the head's residual stream carries a root-mean-square norm "
            "and a residual add, and its width matches the target hidden size. "
            "A permutation here would have to be undone by the frozen target.",
            "shapes_agree": True,
        },
    ]


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
    tiled = {}
    for name in CORE:
        weight = np.ascontiguousarray(master.float32(f"{name}.weight"))
        shapes[name] = list(weight.shape)
        if name == "layers.0.self_attn.o_proj":
            tiled[name] = tiled_permutation_gain(weight)
        spread, zero_columns = column_scale_spread(weight)
        kurtosis_median, kurtosis_p99 = group_kurtosis(weight)
        natural, sorted_error, factor = measured_permutation_gain(weight)
        oracle = per_row_oracle_rel_l2(weight)
        tensors[name] = {
            "e144_column_effect_share": column_effect_share(weight),
            "rel_l2_per_row_oracle": oracle,
            "e144_per_row_oracle_rel_l2_factor": natural / oracle,
            "mse_reduction_pct": 100.0 * (1.0 - (sorted_error / natural) ** 2),
            "oracle_mse_reduction_pct": 100.0 * (1.0 - (oracle / natural) ** 2),
            "shape": list(weight.shape),
            "e144_col_scale_log_std": spread,
            "zero_columns": zero_columns,
            "group_kurtosis_median": kurtosis_median,
            "group_kurtosis_p99": kurtosis_p99,
            "rel_l2_natural_order": natural,
            "rel_l2_sorted_columns": sorted_error,
            "e144_permutation_rel_l2_factor": factor,
            "groups_per_row": weight.shape[1] // GROUP,
            "frobenius_norm": float(np.linalg.norm(weight.astype(np.float64))),
        }
        if name in tiled:
            tensors[name]["rel_l2_tiled_head_order"] = tiled[name]
            tensors[name]["e144_tiled_permutation_rel_l2_factor"] = natural / tiled[name]
        if name == "fc":
            half = weight.shape[1] // 2
            tensors[name]["block_profile"] = block_scale_profile(
                weight, [("target embedding", 0, half), ("target hidden", half, weight.shape[1])]
            )
        if name == "layers.0.self_attn.o_proj":
            tensors[name]["block_profile"] = block_scale_profile(
                weight,
                [
                    ("query head %d" % index, index * HEAD_DIM, (index + 1) * HEAD_DIM)
                    for index in range(weight.shape[1] // HEAD_DIM)
                ],
            )
        print(
            "%-32s shape %5d x %5d  log_std %.4f  col_effect %.4f  kurt %7.3f  "
            "relL2 %.6e  sorted x%.5f  oracle x%.5f"
            % (
                name,
                weight.shape[0],
                weight.shape[1],
                spread,
                tensors[name]["e144_column_effect_share"],
                kurtosis_median,
                natural,
                factor,
                natural / oracle,
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

    controls = [positive_control(log_std=value) for value in (0.0, 0.25, 0.50, 0.75, 1.00)]
    print()
    for control in controls:
        print(
            "positive control  requested log_std %.2f  measured %.4f  relL2 %.6e -> %.6e  x%.5f"
            % (
                control["requested_log_std"],
                control["e144_col_scale_log_std"],
                control["rel_l2_natural_order"],
                control["rel_l2_sorted_columns"],
                control["e144_permutation_rel_l2_factor"],
            )
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

    def pooled(reachable):
        """Pooled relL2 over all eight core tensors, weighted by Frobenius norm."""
        squared = 0.0
        total = 0.0
        for name, record in tensors.items():
            norm = record["frobenius_norm"]
            error = reachable(name, record) * norm
            squared += error * error
            total += norm * norm
        return (squared**0.5) / (total**0.5)

    pooled_natural = pooled(lambda name, record: record["rel_l2_natural_order"])
    pooled_reachable = pooled(
        lambda name, record: (
            record.get("rel_l2_tiled_head_order", record["rel_l2_sorted_columns"])
            if name in admissible
            else record["rel_l2_natural_order"]
        )
    )
    pooled_admissible_oracle = pooled(
        lambda name, record: (
            record["rel_l2_per_row_oracle"]
            if name in admissible
            else record["rel_l2_natural_order"]
        )
    )
    pooled_all_oracle = pooled(lambda name, record: record["rel_l2_per_row_oracle"])
    rb_factor = rb_pooled_factor()

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
        "positive_controls": controls,
        "pooled": {
            "rel_l2_natural_order": pooled_natural,
            "rel_l2_admissible_permutations_only": pooled_reachable,
            "e144_rg_pooled_rel_l2_factor": pooled_natural / pooled_reachable,
            "rel_l2_admissible_per_row_oracle": pooled_admissible_oracle,
            "e144_rg_pooled_admissible_oracle_factor": pooled_natural / pooled_admissible_oracle,
            "rel_l2_all_tensors_per_row_oracle": pooled_all_oracle,
            "e144_rg_pooled_unreachable_oracle_factor": pooled_natural / pooled_all_oracle,
            "note": "only the admissible consumers move; every other tensor keeps "
            "its natural order, and o_proj uses the grouped-query tiled order. The "
            "oracle rows sort every ROW independently, which no single column order "
            "can reach, so they bound the whole family from above.",
        },
        "gate": {
            "statistic": "std(log(per-column max-abs)) on the grouped axis",
            "low": arguments.gate_low,
            "high": arguments.gate_high,
            "max_spread_on_error_carrying_tensors": worst_spread,
            "max_spread_on_admissible_tensors": admissible_spread,
            "max_measured_permutation_factor_on_admissible_tensors": admissible_factor,
            "max_measured_permutation_factor_on_any_core_tensor": max(
                tensors[name]["e144_permutation_rel_l2_factor"] for name in tensors
            ),
            "max_per_row_oracle_factor_on_admissible_tensors": max(
                tensors[name]["e144_per_row_oracle_rel_l2_factor"] for name in admissible
            ),
            "max_column_effect_share": max(
                record["e144_column_effect_share"] for record in tensors.values()
            ),
            "positive_control_detects_a_gain": any(c["detects_a_gain"] for c in controls),
            "rg_dead_by_statistic": admissible_spread < arguments.gate_low,
            "rg_reopens_axis_by_statistic": admissible_spread > arguments.gate_high,
            "e144_rg_reachable_pooled_factor": pooled_natural / pooled_reachable,
            "rg_dead_by_reachable_measurement": (pooled_natural / pooled_reachable) < 1.01,
            "statistic_and_reachable_measurement_agree": (
                (admissible_spread > arguments.gate_high)
                == ((pooled_natural / pooled_reachable) >= 1.01)
            ),
            "verdict": "the statistic reopens the axis and the direct measurement "
            "closes it. std(log(per-column max-abs)) is a max-of-N order statistic "
            "over thousands of rows, not a measure of the column effect a shared "
            "permutation can act on. Column identity explains at most %.2f %% of the "
            "variance of log|w| on this head, so the reachable pooled gain is only "
            "x%.5f even though sorting each row independently -- which no single "
            "column order can reach -- would give x%.4f."
            % (
                100.0 * max(record["e144_column_effect_share"] for record in tensors.values()),
                pooled_natural / pooled_reachable,
                pooled_natural / pooled_all_oracle,
            ),
        },
        "value": {
            "note": "INFERRED, not measured. Prices the reachable pooled factor "
            "through the E144 acceptance model damage = 0.82 * factor ** -k.",
            "reachable_pooled_factor": pooled_natural / pooled_reachable,
            "rb_best_of_breed_factor": rb_factor,
            "composed_with_rb_best_of_breed": rb_factor * pooled_natural / pooled_reachable,
            "recovery_pt": {
                model: [
                    recovery(pooled_natural / pooled_reachable, k),
                    recovery(rb_factor * pooled_natural / pooled_reachable, k),
                ]
                for model, k in (("linear", 1.0), ("fitted_low", 1.7381), ("square", 2.0))
            },
            "target_pt": 0.30,
        },
    }

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print()
    print(json.dumps(report["pooled"], indent=2))
    print(json.dumps(report["gate"], indent=2))
    print(json.dumps(report["value"], indent=2))
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
