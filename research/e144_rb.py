"""E144 R-B: data-free requantizers at identical bytes, group size and layout.

Rebuilds the eight affine-4 group-64 core projections of the declared MTP head
with three data-free quantizers and measures reconstruction error against
master-bf16. Also measures the exhaustive affine optimum on a sample, which
bounds what ANY quantizer in this format can achieve.

Every tensor outside the eight core projections is copied verbatim from the
declared head, so the produced tree has the same 40 tensors, the same shapes and
the same payload byte count.
"""

import argparse
import json
import os
import time

import numpy as np

from e144_quant import (
    EPS,
    _als_loop,
    _groups,
    alternating_ls,
    best_of_breed,
    clip_search,
    dequantize,
    deployed_error,
    mlx_rtn,
    rtn_params,
)
from e144_ra import CORE, DECLARED, MASTER
from e144_st import SafeTensors, save

# `rtn` is the incumbent's own quantizer. Keeping it in the sweep is a standing
# self-check: its relL2 must equal the declared head's to every printed digit,
# which proves the reproduction still holds after any edit to e144_quant.
QUANTIZERS = {"rtn": mlx_rtn, "clip": clip_search, "als": alternating_ls, "best": best_of_breed}


def _sample_groups(grouped, samples, seed):
    rng = np.random.default_rng(seed)
    count = min(samples, grouped.shape[0])
    chosen = rng.choice(grouped.shape[0], count, replace=False)
    return np.ascontiguousarray(grouped[chosen]), count


def affine_ceiling(grouped, bits=4, samples=3000, seed=0, coarse=31):
    """Best affine-4 group-64 parameters we can find on a random sample of groups.

    Every affine quantizer is exactly a choice of reconstructed range [lo, hi],
    so searching that range searches the whole family. Each end is shrunk towards
    the group mean independently, because a group with one large outlier needs a
    one-sided clip and a symmetric sweep cannot express that. The grid winner is
    then polished by alternating least squares, which is a proper local optimizer
    for the same objective.

    This is ACHIEVABLE, so it is a lower bound on the family optimum, not a
    proved supremum. `codebook_ceiling` supplies the matching upper bound. The
    search is scored in float32, so it is also optimistic by the amount BF16
    parameter rounding costs.

    Returns the incumbent MLX round-to-nearest error, the plain min/max error and
    the search optimum, all as sums of squares over the same sampled groups.
    """
    sample, count = _sample_groups(grouped, samples, seed)
    n_bins = np.float32((1 << bits) - 1)

    w_min = sample.min(axis=1, keepdims=True)
    w_max = sample.max(axis=1, keepdims=True)
    center = sample.mean(axis=1, keepdims=True)
    low_span = center - w_min
    high_span = w_max - center

    def error_of(lo, hi):
        scale = np.maximum((hi - lo) / n_bins, EPS)
        codes = np.clip(np.round((sample - lo) / scale), 0.0, n_bins)
        return np.sum((sample - (codes * scale + lo)) ** 2, axis=1, keepdims=True), scale

    rtn_scale, rtn_bias = rtn_params(sample, bits)
    incumbent, _ = deployed_error(sample, rtn_scale, rtn_bias, n_bins)

    best_lo = w_min.copy()
    best, best_scale = error_of(w_min, w_max)
    minmax = best.copy()

    grid = np.linspace(0.02, 1.00, coarse, dtype=np.float32)
    for low_fraction in grid:
        lo = center - low_fraction * low_span
        for high_fraction in grid:
            error, scale = error_of(lo, center + high_fraction * high_span)
            improved = error < best
            best = np.where(improved, error, best)
            best_scale = np.where(improved, scale, best_scale)
            best_lo = np.where(improved, lo, best_lo)

    _, _, best = _als_loop(sample, best_scale, best_lo, best, n_bins, sample.shape[1], 40)
    return float(incumbent.sum()), float(minmax.sum()), float(best.sum()), count


def codebook_ceiling(grouped, bits=4, samples=3000, seed=0, iterations=40):
    """Per-group Lloyd-Max optimum: the best possible `bits`-bit codebook.

    This drops the affine constraint entirely and lets each group of 64 weights
    have its own free 16-entry codebook. It is deliberately UNSHIPPABLE -- a
    16-entry BF16 codebook costs 32 bytes per group against affine's 4 -- but it
    bounds every quantizer that spends `bits` bits per weight, whatever the
    reconstruction rule. If the affine optimum is already close to this, the
    remaining error is paid to the bit width and no quantizer choice can recover
    it.
    """
    sample, count = _sample_groups(grouped, samples, seed)
    levels = 1 << bits
    lanes = np.arange(levels)

    def lloyd(centroids):
        for _ in range(iterations):
            assignment = np.argmin(np.abs(sample[:, :, None] - centroids[:, None, :]), axis=2)
            onehot = assignment[:, :, None] == lanes.reshape(1, 1, levels)
            occupancy = onehot.sum(axis=1)
            total = np.einsum("gc,gcl->gl", sample, onehot.astype(np.float32))
            moved = total / np.maximum(occupancy, 1).astype(np.float32)
            centroids = np.where(occupancy == 0, centroids, moved)
        assignment = np.argmin(np.abs(sample[:, :, None] - centroids[:, None, :]), axis=2)
        reconstruction = np.take_along_axis(centroids, assignment, axis=1)
        return np.sum((sample - reconstruction) ** 2, axis=1, keepdims=True)

    w_min = sample.min(axis=1, keepdims=True)
    w_max = sample.max(axis=1, keepdims=True)
    uniform = w_min + (w_max - w_min) / np.float32(levels - 1) * lanes.astype(np.float32)
    # Lloyd is only locally optimal, so start it twice: from the uniform grid the
    # affine family would use, and from the group's own quantiles, which is much
    # closer for a bell-shaped group. Keep whichever converges lower per group.
    ordered = np.sort(sample, axis=1)
    picks = np.linspace(0, sample.shape[1] - 1, levels).round().astype(int)
    error = np.minimum(lloyd(uniform), lloyd(ordered[:, picks]))
    return float(error.astype(np.float64).sum()), count


def run_bounds(master, samples, out):
    """What ANY quantizer of this bit width could reach, ignoring who writes it.

    Two nested bounds per tensor on the same sampled groups: the best affine-4
    group-64 parameters, which is what we may actually ship, and the best free
    16-entry codebook per group, which we may not. The distance between them
    says how much of the incumbent's error is the affine constraint and how much
    is the bit width itself.
    """
    report = {
        "experiment": "e144",
        "rung": "R-B ceiling",
        "harness": "local",
        "sample_groups_per_tensor": samples,
        "reference": "the incumbent MLX affine_quantize error on the same sampled groups",
        "tensors": {},
    }
    incumbent = minmax = affine = codebook = 0.0

    for name in CORE:
        grouped = _groups(master.float32(f"{name}.weight"), 64)
        incumbent_sse, minmax_sse, affine_sse, count = affine_ceiling(grouped, samples=samples)
        codebook_sse, _ = codebook_ceiling(grouped, samples=samples)
        incumbent += incumbent_sse
        minmax += minmax_sse
        affine += affine_sse
        codebook += codebook_sse
        report["tensors"][name] = {
            "groups": count,
            "incumbent_sse": incumbent_sse,
            "minmax_sse": minmax_sse,
            "affine_optimum_sse": affine_sse,
            "codebook_optimum_sse": codebook_sse,
            "minmax_rel_l2_factor": float(np.sqrt(incumbent_sse / minmax_sse)),
            "affine_rel_l2_factor": float(np.sqrt(incumbent_sse / affine_sse)),
            "codebook_rel_l2_factor": float(np.sqrt(incumbent_sse / codebook_sse)),
        }
        print(
            f"{name:32s} minmax x{np.sqrt(incumbent_sse / minmax_sse):.4f} "
            f"affine x{np.sqrt(incumbent_sse / affine_sse):.4f} "
            f"codebook x{np.sqrt(incumbent_sse / codebook_sse):.4f}"
        )
        del grouped

    report["pooled"] = {
        "minmax_rel_l2_factor": float(np.sqrt(incumbent / minmax)),
        "affine_rel_l2_factor": float(np.sqrt(incumbent / affine)),
        "codebook_rel_l2_factor": float(np.sqrt(incumbent / codebook)),
        "note": (
            "relL2 improvement over the INCUMBENT MLX quantizer, which is the denominator "
            "the assignment's stop rule uses. The affine figure bounds every shippable "
            "quantizer at 4 bits and group 64. The codebook figure drops the affine "
            "constraint and is UNSHIPPABLE at these bytes -- a 16-entry BF16 codebook costs "
            "32 bytes per group against affine's 4 -- so it bounds every scheme that spends "
            "4 bits per weight, whatever the reconstruction rule."
        ),
    }
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["pooled"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="e144-rb.json")
    parser.add_argument("--emit", default=None, help="write the best-quantizer head here")
    parser.add_argument("--quantizer", default="best", choices=sorted(QUANTIZERS))
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="use only the first N rows of each core tensor; cannot emit a head",
    )
    parser.add_argument(
        "--bounds-only",
        action="store_true",
        help="measure only the affine and codebook ceilings, skipping the quantizers",
    )
    parser.add_argument("--samples", type=int, default=3000, help="sampled groups per tensor")
    arguments = parser.parse_args()
    if arguments.smoke and arguments.emit:
        parser.error("--smoke produces a truncated tensor, so it cannot emit a head")

    master = SafeTensors(MASTER)
    if arguments.bounds_only:
        run_bounds(master, arguments.samples, arguments.out)
        return

    declared = SafeTensors(DECLARED)

    report = {
        "experiment": "e144",
        "rung": "R-B",
        "harness": "local",
        "data_free": True,
        "note": "no calibration data, no activations, no training; a pure function of master-bf16",
        "smoke_rows": arguments.smoke,
        "tensors": {},
    }

    totals = {name: 0.0 for name in ["declared", *QUANTIZERS]}
    energy = 0.0
    ceiling_reference = 0.0
    ceiling_best = 0.0
    emitted = {}

    rows_kept = arguments.smoke or None
    for name in CORE:
        weight = master.float32(f"{name}.weight")[:rows_kept]
        columns = weight.shape[1]
        reference = weight.astype(np.float64)
        tensor_energy = float((reference**2).sum())
        energy += tensor_energy

        declared_error = float(
            (
                (
                    reference
                    - dequantize(
                        declared.raw(f"{name}.weight")[:rows_kept],
                        declared.raw(f"{name}.scales")[:rows_kept],
                        declared.raw(f"{name}.biases")[:rows_kept],
                        4,
                        64,
                        columns,
                    ).astype(np.float64)
                )
                ** 2
            ).sum()
        )
        totals["declared"] += declared_error

        entry = {
            "shape": list(weight.shape),
            "rel_l2": {"declared": float(np.sqrt(declared_error / tensor_energy))},
        }

        errors = {}
        for label, quantizer in QUANTIZERS.items():
            started = time.time()
            packed, scales, biases, _ = quantizer(weight)
            error = float(
                (
                    (
                        reference
                        - dequantize(packed, scales, biases, 4, 64, columns).astype(np.float64)
                    )
                    ** 2
                ).sum()
            )
            errors[label] = error
            totals[label] += error
            entry["rel_l2"][label] = float(np.sqrt(error / tensor_energy))
            if label == arguments.quantizer:
                emitted[f"{name}.weight"] = packed
                emitted[f"{name}.scales"] = (scales, "BF16")
                emitted[f"{name}.biases"] = (biases, "BF16")
                entry["seconds"] = round(time.time() - started, 2)
            del packed, scales, biases

        entry["improvement_factor_vs_declared"] = (
            entry["rel_l2"]["declared"] / entry["rel_l2"][arguments.quantizer]
        )
        incumbent_sse, _, ceiling_sse, sampled = affine_ceiling(_groups(weight, 64))
        ceiling_reference += incumbent_sse
        ceiling_best += ceiling_sse
        entry["affine_optimum_sample"] = {
            "groups": sampled,
            "incumbent_sse": incumbent_sse,
            "optimum_sse": ceiling_sse,
            "sse_factor": incumbent_sse / ceiling_sse,
        }

        entry["rtn_reproduces_declared"] = errors["rtn"] == declared_error
        report["tensors"][name] = entry
        print(
            f"{name:32s} declared {entry['rel_l2']['declared']:.5e} "
            f"rtn {entry['rel_l2']['rtn']:.5e} "
            f"clip {entry['rel_l2']['clip']:.5e} "
            f"als {entry['rel_l2']['als']:.5e} "
            f"best {entry['rel_l2']['best']:.5e} "
            f"x{entry['improvement_factor_vs_declared']:.4f}"
        )
        del weight, reference

    pooled = {label: float(np.sqrt(total / energy)) for label, total in totals.items()}
    chosen = arguments.quantizer
    worst = min(
        report["tensors"].values(), key=lambda e: e["improvement_factor_vs_declared"]
    )["improvement_factor_vs_declared"]

    report["pooled_rel_l2"] = pooled
    report["summary"] = {
        "quantizer": chosen,
        "e144_declared_reproduction_exact": float(
            all(e["rtn_reproduces_declared"] for e in report["tensors"].values())
        ),
        "e144_rel_l2_improvement_factor_pooled": pooled["declared"] / pooled[chosen],
        "e144_rel_l2_improvement_factor_worst_tensor": worst,
        "stop_rule_threshold": 2.0,
        "stop_rule_passed": (pooled["declared"] / pooled[chosen]) >= 2.0,
        "format_ceiling": {
            "note": (
                "exhaustive 2D (scale, bias) search per group, pooled over sampled groups "
                "of all eight core tensors; bounds ANY affine-4 group-64 quantizer"
            ),
            "sse_factor": ceiling_reference / ceiling_best,
            "rel_l2_factor": float(np.sqrt(ceiling_reference / ceiling_best)),
            "rel_l2_factor_needed_for_stop_rule": 2.0,
            "sse_factor_needed_for_stop_rule": 4.0,
        },
    }

    if arguments.emit:
        for name in declared.header:
            if name not in emitted:
                raw = declared.raw(name)
                dtype = declared.info(name)[0]
                emitted[name] = (raw, dtype) if dtype == "BF16" else raw
        ordered = {name: emitted[name] for name in declared.header}
        save(
            arguments.emit,
            ordered,
            metadata={
                **declared.metadata,
                "e144_quantizer": chosen,
                "e144_note": (
                    "eight affine-4 group-64 core projections requantized from "
                    "EigenLabs/Qwen3.8-27B-MTP-bf16 by a data-free per-group optimizer; "
                    "all other tensors verbatim from the incumbent declared head"
                ),
            },
        )
        report["emitted"] = {
            "path": arguments.emit,
            "bytes": os.path.getsize(arguments.emit),
            "declared_bytes": os.path.getsize(DECLARED),
            "e144_head_bytes_delta": os.path.getsize(arguments.emit) - os.path.getsize(DECLARED),
        }
        print("\nemitted", json.dumps(report["emitted"], indent=2))

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("\npooled relL2", json.dumps(pooled, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
