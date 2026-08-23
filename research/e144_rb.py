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

from e144_quant import _groups, alternating_ls, best_of_breed, clip_search, dequantize, mlx_rtn
from e144_ra import CORE, DECLARED, MASTER
from e144_st import SafeTensors, save

# `rtn` is the incumbent's own quantizer. Keeping it in the sweep is a standing
# self-check: its relL2 must equal the declared head's to every printed digit,
# which proves the reproduction still holds after any edit to e144_quant.
QUANTIZERS = {"rtn": mlx_rtn, "clip": clip_search, "als": alternating_ls, "best": best_of_breed}


def affine_ceiling(grouped, bits=4, samples=3000, seed=0):
    """Exhaustive 2D (scale, bias) optimum on a random sample of groups.

    Upper bound on what any affine quantizer of this bit width and group size can
    reach, used to test whether the assignment's 2x relL2 stop rule is reachable
    at all.
    """
    rng = np.random.default_rng(seed)
    count = min(samples, grouped.shape[0])
    sample = np.ascontiguousarray(grouped[rng.choice(grouped.shape[0], count, replace=False)])
    n_bins = np.float32((1 << bits) - 1)

    w_min = sample.min(axis=1, keepdims=True)
    w_max = sample.max(axis=1, keepdims=True)
    base_scale = (w_max - w_min) / n_bins
    base_bias = w_min

    def error_of(scale, bias):
        codes = np.clip(np.round((sample - bias) / scale), 0.0, n_bins)
        return np.sum((sample - (codes * scale + bias)) ** 2, axis=1, keepdims=True)

    reference = error_of(base_scale, base_bias).sum()
    best = np.full((count, 1), np.inf, dtype=np.float32)
    for fraction in np.linspace(0.55, 1.05, 101, dtype=np.float32):
        for offset in np.linspace(-0.5, 0.5, 81, dtype=np.float32):
            best = np.minimum(best, error_of(base_scale * fraction, base_bias + offset * base_scale))
    return float(reference), float(best.sum()), count


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
    arguments = parser.parse_args()
    if arguments.smoke and arguments.emit:
        parser.error("--smoke produces a truncated tensor, so it cannot emit a head")

    master = SafeTensors(MASTER)
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
        reference_sse, ceiling_sse, sampled = affine_ceiling(_groups(weight, 64))
        ceiling_reference += reference_sse
        ceiling_best += ceiling_sse
        entry["affine_optimum_sample"] = {
            "groups": sampled,
            "minmax_sse": reference_sse,
            "exhaustive_sse": ceiling_sse,
            "sse_factor": reference_sse / ceiling_sse,
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
