#!/usr/bin/env python3
"""Publish the E98 transform-owned metadata index evidence to W&B.

    usage: research/e98_wandb_log.py [--only RUN]

  `e98-census`  rung 0 item 5: per-tensor distinct scale, bias and pair counts
                over the transformed checkpoint, which decides whether the
                uint16 pair index is required or a scale-keyed table suffices.
  `e98-bytes`   rung 1a: the group-size ladder. Group sizes 32, 64 and 128 read
                40, 36 and 34 metadata bytes per 64 quantized elements through
                the same kernel at M = 1, so the ladder prices a metadata byte
                without writing a new kernel.

Every leg is a within-session relative measurement with the cool gate off, so
`timing_valid`, `cool_gate_passed_real_gate` and `gate_qualified_for_timing`
are logged false verbatim and no leg is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import defaultdict

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e98_bytes_analysis import DRAM_PEAK_GB_S, weight_streams  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e98-transform-owned-weight-metadata-index"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out")

# Advisor f2, from alphonse's E96 in-situ dispatch census (W&B 8m8d3mnr).
# Recorded as configuration for cross-instrument comparison, never as a value
# measured by this session.
E96 = {
    "e96_streaming_share_of_round": 0.886,
    "e96_streaming_gb_per_s": 249.55,
    "e96_lm_head_gb_per_s": 271.9,
    "e96_mlp_gate_up_gb_per_s": 265.8,
    "e96_weight_streams_per_round": 2.005,
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


def log_census(directory: pathlib.Path) -> None:
    payload = json.loads((directory / "dependence.json").read_text())
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="metadata-cardinality",
        name="e98-census",
        config={
            "experiment": GROUP,
            "rung": "0-item-5",
            "question": "is the bias a function of the scale, so that no index is needed",
            "source_census": "research/out/e97-census/census.json",
            "host": HOST,
            **gate_flags(),
        },
        reinit=True,
    )
    table = wandb.Table(columns=["family", "tensors", "groups", "metadata_bytes"])
    for name, fam in sorted(payload["families"].items()):
        table.add_data(name, fam["tensors"], fam["groups"], fam["metadata_bytes"])
    run.log({"census/families": table})
    flat = {
        f"census/{key}": value
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }
    for key in ("distinct_scales", "distinct_biases", "distinct_pairs",
                "pairs_over_scales", "pairs_over_biases"):
        for stat, value in payload[key].items():
            flat[f"census/{key}_{stat}"] = value
    run.log(flat)
    run.summary.update(
        {
            "index_required": payload["bias_is_function_of_scale"] == 0
            and payload["scale_is_function_of_bias"] == 0,
            "uint16_index_sufficient": payload["index_overflow_tensors"] == 0,
            "lut_fits_in_own_biases_array": payload["lut_overflow_tensors"] == 0,
            "max_lut_bytes": payload["max_lut_bytes"],
            "metadata_byte_cut_fraction": 2.0 / 36.0,
        }
    )
    run.finish()


def log_bytes(directory: pathlib.Path) -> None:
    payload = json.loads((directory / "bytes.json").read_text())
    meta = read_meta(directory / "meta.txt")
    cells = payload["cells"]

    by = defaultdict(list)
    for cell in cells:
        by[(cell["shape"], cell["width"], cell["group_size"])].append(cell)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="metadata-byte-ladder",
        name="e98-bytes",
        config={
            "experiment": GROUP,
            "rung": "1a",
            "question": "does a metadata byte convert to time at the achieved read rate",
            "instrument": "group-size ladder through the shipped kernel",
            "metadata_bytes_per_64_elements": {"32": 40, "64": 36, "128": 34},
            "blocks_abba": payload["blocks"],
            "eval_overhead_us": payload["eval_overhead_us"],
            "dram_peak_gb_per_s": DRAM_PEAK_GB_S,
            **E96,
            **identity(meta),
            **gate_flags(),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=[
            "shape",
            "width",
            "group_size",
            "kernel_family",
            "weight_streams",
            "net_us",
            "spread_pct",
            "read_bytes",
            "read_gb_per_s",
            "pct_dram_peak",
            "dram_floor_us",
            "below_logical_byte_floor",
        ]
    )
    summary: dict[str, object] = {}
    for (shape, width, gs), group in sorted(by.items()):
        net = statistics.mean(c["net_us"] for c in group)
        spread = 100 * (
            max(c["net_us"] for c in group) - min(c["net_us"] for c in group)
        ) / net
        family = sorted({c["kernel_family"] for c in group})[0]
        streams = weight_streams(family, width)
        read = group[0]["read_bytes"] * streams
        gbs = read / (net * 1e-6) / 1e9
        floor = (read + group[0]["write_bytes"]) / (DRAM_PEAK_GB_S * 1e9) * 1e6
        table.add_data(
            shape,
            width,
            gs,
            family,
            streams,
            net,
            spread,
            read,
            gbs,
            100 * gbs / DRAM_PEAK_GB_S,
            floor,
            net < floor,
        )
        summary[f"cell/{shape}/m{width}/gs{gs}/net_us"] = net
        summary[f"cell/{shape}/m{width}/gs{gs}/read_gb_per_s"] = gbs

    # The only same-kernel byte contrast the ladder can reach is M = 1, where
    # quantized.h:1917 sends every group size to qmv_fast_impl.
    contrast = wandb.Table(
        columns=["shape", "contrast", "measured_pct", "predicted_pct", "conversion"]
    )
    for shape in sorted({c["shape"] for c in cells}):
        base = statistics.mean(c["net_us"] for c in by[(shape, 1, 64)])
        base_b = by[(shape, 1, 64)][0]["read_bytes"]
        for other, label in ((128, "g64_to_g128"), (32, "g32_to_g64")):
            cand = statistics.mean(c["net_us"] for c in by[(shape, 1, other)])
            cand_b = by[(shape, 1, other)][0]["read_bytes"]
            measured = 100 * (cand - base) / base
            predicted = 100 * (cand_b - base_b) / base_b
            contrast.add_data(shape, label, measured, predicted, measured / predicted)
            summary[f"m1_contrast/{shape}/{label}_measured_pct"] = measured
            summary[f"m1_contrast/{shape}/{label}_predicted_pct"] = predicted
            summary[f"m1_contrast/{shape}/{label}_conversion"] = measured / predicted

    nulls = wandb.Table(columns=["shape", "tag", "group_size", "width", "us"])
    for null in payload["nulls"]:
        nulls.add_data(
            null["shape"], null["tag"], null["group_size"], null["width"],
            null["microseconds"],
        )

    run.log({"bytes/cells": table, "bytes/m1_contrasts": contrast, "bytes/nulls": nulls})
    run.summary.update(summary)
    run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["census", "bytes"])
    args = ap.parse_args()
    if args.only in (None, "census"):
        log_census(OUT / "e98-census")
    if args.only in (None, "bytes"):
        log_bytes(OUT / "e98-bytes-r1a")


if __name__ == "__main__":
    main()
