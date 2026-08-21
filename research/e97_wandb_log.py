#!/usr/bin/env python3
"""Publish the E97 per-row verify cost attribution to W&B.

    usage: research/e97_wandb_log.py [--out DIR]

Four runs, one per rung, plus the metadata census:

  `e97-peak`    rung 0: the arithmetic ceiling this GPU actually reached,
                measured through the same MLX build the scored path uses.
  `e97-row`     rung 1: the marginal cost of one verify row at two scored
                widths, quantized against bf16, with the band fits, the
                per-increment table and the session null.
  `e97-shape`   rung 2: the same band slope swept over the reduction length,
                which splits the per-row cost into a K-proportional term and a
                K-independent term, plus the matched NA contrast.
  `e97-census`  the lossless (scale, bias) cardinality census of the
                transformed checkpoint.

Every leg is a within-session relative measurement with the cool gate off, so
`timing_valid`, `cool_gate_passed_real_gate` and `gate_qualified_for_timing`
are logged false verbatim on every run and no leg is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e97_row_cost_analysis as rowcost  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e97-ranked-per-row-verify-cost"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out")

# The ranked cost curve the advisor fitted on M5, for transfer only. The local
# host is not the ranked host, so these are recorded as configuration, never as
# a measured value of this session.
RANKED = {
    "ranked_slope_us_per_row": 7232.4,
    "ranked_slope_sd_pct": 2.76,
    "ranked_within_run_se_us": 205.4,
    "ranked_implied_tflop_per_s": 4.74,
}


def gate_flags() -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def identity(meta: dict[str, str]) -> dict[str, object]:
    return {
        "host": HOST,
        "hostname": meta.get("host"),
        "chip": meta.get("chip"),
        "memory_gib": int(meta["memory_gib"]) if meta.get("memory_gib") else None,
        "toolchain": meta.get("toolchain"),
        "git_head": meta.get("git_head"),
        "git_dirty": meta.get("git_dirty") != "0",
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
        if meta.get("gpu_temp_entry_c")
        else None,
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
        if meta.get("gpu_temp_exit_c")
        else None,
        "started_utc": meta.get("started_utc"),
        "finished_utc": meta.get("finished_utc"),
    }


def log_peak(directory: pathlib.Path) -> dict:
    """Rung 0. Without this run every later fraction is a fraction of a
    specification number nobody measured."""
    payload = json.loads((directory / "peak.json").read_text())
    ceilings = rowcost.peak_ceilings(payload)
    meta = read_meta(directory / "meta.txt")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="achievable-peak",
        name="e97-peak",
        config={
            "experiment": GROUP,
            "rung": 0,
            "question": "what arithmetic rate can this GPU actually reach",
            "eval_overhead_us": payload["eval_overhead_us"],
            **identity(meta),
            **gate_flags(),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=["label", "dtype", "m", "k", "n", "replicates", "net_us", "tflop_per_s"]
    )
    for record in payload["records"]:
        m, k, n = record["shape"]
        table.add_data(
            record["label"],
            record["dtype"],
            m,
            k,
            n,
            record["replicates"],
            record["net_us"],
            record["tflop_per_s"],
        )

    run.log(
        {
            "peak/records": table,
            "peak/affine4_tflop_per_s": ceilings["affine4"]["tflop_per_s"],
            "peak/dense_tflop_per_s": ceilings["dense"]["tflop_per_s"],
            "peak/affine4_over_dense": ceilings["affine4"]["tflop_per_s"]
            / ceilings["dense"]["tflop_per_s"],
            "transfer/ranked_slope_over_affine4_peak": RANKED[
                "ranked_implied_tflop_per_s"
            ]
            / ceilings["affine4"]["tflop_per_s"],
            "transfer/ranked_slope_over_dense_peak": RANKED[
                "ranked_implied_tflop_per_s"
            ]
            / ceilings["dense"]["tflop_per_s"],
        }
    )
    run.summary.update(RANKED)
    run.finish()
    return ceilings


def log_row(directory: pathlib.Path, ceilings: dict) -> dict:
    """Rung 1. The marginal verify row, quantized against bf16."""
    payload = json.loads((directory / "row-cost.json").read_text())
    result = rowcost.analyse(payload)
    result["against_peak"] = rowcost.against_peak(result["band_fits"], ceilings)
    meta = read_meta(directory / "meta.txt")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="per-row-slope",
        name="e97-row",
        config={
            "experiment": GROUP,
            "rung": 1,
            "question": "what does a marginal verify row buy, and is it dequant",
            "kernel": "qmv_fast_crossrow_affine4_g64_wide",
            "hidden": payload["hidden"],
            "group_size": payload["group_size"],
            "bits": payload["bits"],
            "vector_limit": payload["vector_limit"],
            "blocks": payload["blocks"],
            "eval_overhead_us": payload["eval_overhead_us"],
            "affine4_peak_tflop_per_s": ceilings["affine4"]["tflop_per_s"],
            "dense_peak_tflop_per_s": ceilings["dense"]["tflop_per_s"],
            **identity(meta),
            **gate_flags(),
        },
        reinit=True,
    )

    widths = wandb.Table(
        columns=["kernel", "outputs", "m", "groups", "inputs_per_group",
                 "mean_us", "net_us", "range_pct"]
    )
    for entry in result["per_width_mean_us"].values():
        widths.add_data(
            entry["kernel"], entry["outputs"], entry["m"], entry["groups"],
            entry["inputs_per_group"], entry["mean_us"], entry["net_us"],
            entry["range_pct"],
        )

    bands = wandb.Table(
        columns=["arm", "band", "widths", "slope_us_per_row", "se_us",
                 "tflop_per_s", "fraction_of_affine4_peak", "intercept_us",
                 "r_squared"]
    )
    for key, entry in sorted(result["band_fits"].items()):
        against = result["against_peak"].get(key, {})
        bands.add_data(
            f"{entry['kernel']}/{entry['outputs']}", entry["band"],
            str(entry["widths"]), entry["slope_us_per_row"],
            entry["se_slope_us_per_row"], entry["tflop_per_s"],
            against.get("fraction_of_affine4_peak"), entry["intercept_us"],
            entry["r_squared"],
        )

    steps = wandb.Table(
        columns=["arm", "step", "step_us", "crosses_group", "ipg_from",
                 "ipg_to", "tflop_per_s_if_arithmetic",
                 "fraction_of_m1_weight_read"]
    )
    for entry in result["increments"].values():
        steps.add_data(
            f"{entry['kernel']}/{entry['outputs']}",
            f"{entry['from_m']}->{entry['to_m']}", entry["step_us"],
            entry["crosses_group"], entry["ipg_from"], entry["ipg_to"],
            entry["tflop_per_s_if_arithmetic"],
            entry.get("fraction_of_m1_weight_read"),
        )

    reads = wandb.Table(columns=["kernel", "outputs", "bytes", "net_us", "gb_per_s"])
    for entry in result["reads"]:
        reads.add_data(
            entry["kernel"], entry["outputs"], entry["bytes"], entry["net_us"],
            entry["gb_per_s"],
        )

    metrics: dict[str, object] = {
        "row/per_width": widths,
        "row/band_fits": bands,
        "row/increments": steps,
        "row/read_rate": reads,
    }
    for key, entry in result["band_fits"].items():
        safe = key.replace("/", "_")
        metrics[f"slope/{safe}_us_per_row"] = entry["slope_us_per_row"]
        metrics[f"slope/{safe}_se_us"] = entry["se_slope_us_per_row"]
        metrics[f"slope/{safe}_tflop_per_s"] = entry["tflop_per_s"]
    for key, entry in result["against_peak"].items():
        safe = key.replace("/", "_")
        metrics[f"peakfrac/{safe}"] = entry["fraction_of_affine4_peak"]
        metrics[f"headroom_pct/{safe}"] = entry["headroom_pct_vs_affine4_peak"]
    for key, entry in result["ratios"].items():
        safe = key.replace("/", "_")
        metrics[f"quant_over_bf16/{safe}"] = entry["ratio"]
        metrics[f"quant_over_bf16/{safe}_gap_pct"] = entry["gap_pct"]
    for key, entry in result["regime_step"].items():
        safe = key.replace("/", "_")
        metrics[f"regime/{safe}_matrix_over_vector"] = entry["matrix_over_vector"]
    for key, entry in result["session_null"].items():
        metrics[f"null/o{key}_drift_pct"] = entry["drift_pct"]

    run.log(metrics)
    run.summary.update(
        {
            "eval_overhead_open_us": result["eval_overhead_us"],
            "eval_overhead_close_us": result["eval_overhead_close_us"],
            **RANKED,
        }
    )
    run.finish()
    return result


def log_shape(directory: pathlib.Path, ceilings: dict) -> dict:
    """Rung 2. The same slope against the reduction length."""
    payload = json.loads((directory / "shape.json").read_text())
    result = rowcost.analyse_shape(payload)
    meta = read_meta(directory / "meta.txt")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="cost-shape",
        name="e97-shape",
        config={
            "experiment": GROUP,
            "rung": 2,
            "question": "is the per-row cost FMA throughput or occupancy",
            "kernel": "qmv_fast_crossrow_affine4_g64_wide",
            "group_size": payload["group_size"],
            "bits": payload["bits"],
            "blocks": payload["blocks"],
            "widths": payload["widths"],
            "eval_overhead_us": payload["eval_overhead_us"],
            "affine4_peak_tflop_per_s": ceilings["affine4"]["tflop_per_s"],
            **identity(meta),
            **gate_flags(),
        },
        reinit=True,
    )

    sweep = wandb.Table(
        columns=["outputs", "k", "slope_us_per_row", "se_us", "tflop_per_s",
                 "fraction_of_affine4_peak", "intercept_us", "r_squared"]
    )
    for entry in result["band_slope_per_k"].values():
        sweep.add_data(
            entry["outputs"], entry["k"], entry["slope_us_per_row"],
            entry["se_slope_us_per_row"], entry["tflop_per_s"],
            entry["tflop_per_s"] / ceilings["affine4"]["tflop_per_s"],
            entry["intercept_us"], entry["r_squared"],
        )

    contrast = wandb.Table(
        columns=["outputs", "k", "step_na2_to_3_us", "step_na3_to_4_us",
                 "excess_pct"]
    )
    for entry in result["na_contrast"].values():
        contrast.add_data(
            entry["outputs"], entry["k"], entry["step_na2_to_3_us"],
            entry["step_na3_to_4_us"], entry["excess_pct"],
        )

    metrics: dict[str, object] = {
        "shape/slope_per_k": sweep,
        "shape/na_contrast": contrast,
    }
    for key, entry in result["slope_in_k"].items():
        metrics[f"split/o{key}_ns_per_row_per_k"] = (
            1e3 * entry["us_per_row_per_k"]
        )
        metrics[f"split/o{key}_se_ns_per_row_per_k"] = (
            1e3 * entry["se_us_per_row_per_k"]
        )
        metrics[f"split/o{key}_k_independent_us_per_row"] = entry[
            "k_independent_us_per_row"
        ]
        metrics[f"split/o{key}_k_independent_share"] = entry[
            "k_independent_share_at_reference_k"
        ]
        metrics[f"split/o{key}_r_squared"] = entry["r_squared"]
        # The rate of the reduction-scaled term alone, once the K-independent
        # per-row overhead is taken out. This is the number that decides (B).
        rate = (
            2.0 * int(key) / (entry["us_per_row_per_k"] * 1e-6) / 1e12
        )
        metrics[f"split/o{key}_reduction_tflop_per_s"] = rate
        metrics[f"split/o{key}_reduction_fraction_of_peak"] = (
            rate / ceilings["affine4"]["tflop_per_s"]
        )
    for key, entry in result["session_null"].items():
        metrics[f"null/{key.replace('/', '_')}_drift_pct"] = entry["drift_pct"]

    run.log(metrics)
    run.summary.update(
        {
            "eval_overhead_open_us": result["eval_overhead_us"],
            "eval_overhead_close_us": result["eval_overhead_close_us"],
        }
    )
    run.finish()
    return result


def index_width(distinct: int) -> int:
    """Bits needed to index that many distinct pairs, exactly and losslessly."""
    return max(1, (distinct - 1).bit_length())


def log_census(directory: pathlib.Path) -> dict:
    """Item 5. Can an 8-bit table index the distinct (scale, bias) pairs?"""
    payload = json.loads((directory / "census.json").read_text())
    summary = payload["summary"]
    tensors = payload["tensors"]

    # An affine-4 group-64 record is 32 bytes of nibbles plus a bf16 scale and
    # a bf16 bias, so metadata is 4 of every 36 weight-stream bytes. Replacing
    # the pair with an aligned uint16 index into a per-tensor table is lossless
    # for every tensor whose pair cardinality is below 65536.
    bits = max(index_width(t["distinct_pairs"]) for t in tensors)
    aligned_bytes = 2 if bits <= 16 else 4
    stream_cut = (4 - aligned_bytes) / 36.0
    table_bytes = sum(4 * t["distinct_pairs"] for t in tensors)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="metadata-census",
        name="e97-census",
        config={
            "experiment": GROUP,
            "rung": "item5",
            "question": "is a lossless 8-bit (scale, bias) table possible",
            "weights_dir": summary.get("weights_dir"),
            "tensor_count": summary["tensor_count"],
            "cpu_only": True,
            **gate_flags(),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=["tensor", "shard", "shape", "groups", "distinct_scales",
                 "distinct_biases", "distinct_pairs", "pairs_fit_u8",
                 "index_bits", "metadata_bytes"]
    )
    for entry in tensors:
        table.add_data(
            entry["tensor"], entry["shard"], str(entry["shape"]),
            entry["groups"], entry["distinct_scales"], entry["distinct_biases"],
            entry["distinct_pairs"], entry["pairs_fit_u8"],
            index_width(entry["distinct_pairs"]), entry["metadata_bytes"],
        )

    run.log(
        {
            "census/tensors": table,
            **{
                f"census/{key}": value
                for key, value in summary.items()
                if isinstance(value, (int, float))
            },
            "census/min_lossless_index_bits": bits,
            "census/aligned_index_bytes": aligned_bytes,
            "census/weight_stream_cut_fraction": stream_cut,
            "census/pair_table_bytes_total": table_bytes,
        }
    )
    run.summary.update(summary)
    run.finish()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-dir", default=str(OUT / "e97-peak-r0"))
    parser.add_argument("--row-dir", default=str(OUT / "e97-row-cost-r1"))
    parser.add_argument("--shape-dir", default=str(OUT / "e97-shape-r2"))
    parser.add_argument("--census-dir", default=str(OUT / "e97-census"))
    args = parser.parse_args()

    ceilings = log_peak(pathlib.Path(args.peak_dir))
    log_row(pathlib.Path(args.row_dir), ceilings)
    log_shape(pathlib.Path(args.shape_dir), ceilings)
    census_dir = pathlib.Path(args.census_dir)
    if (census_dir / "census.json").exists():
        log_census(census_dir)
    else:
        print(f"no census at {census_dir}, skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
