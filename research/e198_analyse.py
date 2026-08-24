#!/usr/bin/env python3
"""E198 -- reduce the fused row-amortized SDPA gates and price the dispatch.

Reads the three Stage artefacts and writes one analysis file:

  Stage 1  research/e198-pipeline.jsonl   static pipeline limits per width
  Stage 3  research/e198-exactness.json   bf16 value gate + positive control
  Stage 2  research/e198-timing.json      ABBA dispatch pricing

harness=local. Every number here is a within-session relative measurement on
an ungated GPU; `cool_gate_passed_real_gate` and `gate_qualified_for_timing`
are carried through verbatim.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

# Current-tree census (E196, 81 rounds): rows m = depth + 1.
DEPTH_HISTOGRAM = {3: 1, 4: 29, 5: 2, 6: 3, 7: 46}
ROUNDS_BY_M = {d + 1: n for d, n in DEPTH_HISTOGRAM.items()}
TOTAL_ROUNDS = sum(ROUNDS_BY_M.values())
FULL_ATTENTION_LAYERS = 16
MUE_US_PER_ROUND = 567.0
# FINDING 507: floor identity F(m, kL) = T(1, kL) + (m - 1) * b0, with the
# KV-independent per-row term b0 measured at 3.40 us.
FLOOR_B0_US = 3.40
REQUIRED_THREADS = 1024


def load(path):
    return json.loads(Path(path).read_text())


def load_lines(path):
    text = Path(path).read_text().strip()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def register_curve(records):
    curve = {}
    for record in records:
        name = record["kernel"]
        width = None
        for part in name.replace("<", "_").replace(">", "_").split("_"):
            if part.isdigit():
                width = int(part)
        curve[name] = {
            "kernel": name,
            "width_from_name": width,
            "max_total_threads_per_threadgroup": record[
                "max_total_threads_per_threadgroup"
            ],
            "static_threadgroup_memory_bytes": record[
                "static_threadgroup_memory_bytes"
            ],
            "thread_execution_width": record["thread_execution_width"],
            "fits_required_threadgroup": record["max_total_threads_per_threadgroup"]
            >= REQUIRED_THREADS,
        }
    return curve


def reduce_samples(timing):
    """Median microseconds per dispatch for every (arm, mode, m, kv) cell."""
    buckets = {}
    for sample in timing["samples"]:
        key = (sample["arm"], sample["mode"], sample["m"], sample["kv"])
        buckets.setdefault(key, []).append(sample["microseconds"])
    reduced = {}
    for key, values in buckets.items():
        reduced[key] = {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
            "samples": len(values),
        }
    return reduced


def thermal_record(timing):
    """Entry-temperature spread, which an ungated ABBA session must report."""
    entries = [
        sample["entry_gpu_temperature_c"]
        for sample in timing["samples"]
        if sample.get("entry_gpu_temperature_c", -1) > 0
    ]
    exits = [
        sample["exit_gpu_temperature_c"]
        for sample in timing["samples"]
        if sample.get("exit_gpu_temperature_c", -1) > 0
    ]
    if not entries:
        return {"available": False}
    return {
        "available": True,
        "entry_min_c": min(entries),
        "entry_max_c": max(entries),
        "entry_spread_c": max(entries) - min(entries),
        "entry_median_c": statistics.median(entries),
        "exit_min_c": min(exits) if exits else None,
        "exit_max_c": max(exits) if exits else None,
        "samples": len(entries),
    }


def price(reduced, mode):
    """Per-cell fused-vs-shipped pricing against the FINDING 507 floor."""
    cells = []
    keys = {(m, kv) for (arm, md, m, kv) in reduced if md == mode}
    for m, kv in sorted(keys):
        pair = reduced.get(("today_pair", mode, m, kv))
        shipped = reduced.get(("today_shipped", mode, m, kv))
        fused = reduced.get(("fused", mode, m, kv))
        one_row = reduced.get(("one_row_floor", mode, m, kv))
        if not (pair and shipped and fused and one_row):
            continue
        # Every arm divides its elapsed time by the chain length, and one chain
        # step of the pair arm already issues BOTH split dispatches. So every
        # arm below is one full layer's SDPA work for this cell.
        # `today_pair` is the two SDPA calls alone; `today_shipped` adds the
        # query slicing and the output concatenation the branch also issues,
        # so it is the arm the kernel actually replaces and the gate baseline.
        pair_us = pair["median_us"]
        shipped_us = shipped["median_us"]
        fused_us = fused["median_us"]
        one_row_us = one_row["median_us"]
        floor_us = one_row_us + (m - 1) * FLOOR_B0_US
        head_room = shipped_us - floor_us
        recovered = shipped_us - fused_us
        # 1/m means the m rows rode along on one set of KV loads; 1.0 means
        # they cost the same as m separate one-row calls.
        serialized_us = m * one_row_us
        cells.append(
            {
                "m": m,
                "kv": kv,
                "pair_us": pair_us,
                "shipped_us": shipped_us,
                "slice_concat_us": shipped_us - pair_us,
                "fused_us": fused_us,
                "one_row_us": one_row_us,
                "serialized_us": serialized_us,
                "amortization_efficiency": fused_us / serialized_us,
                "floor_us": floor_us,
                "headroom_us": head_room,
                "recovered_us": recovered,
                "recovered_fraction_of_headroom": (
                    recovered / head_room if head_room > 0 else None
                ),
                "gate_pass_half_headroom": (
                    head_room > 0 and recovered >= 0.5 * head_room
                ),
                "pair_stdev_us": pair["stdev_us"],
                "fused_stdev_us": fused["stdev_us"],
                "samples": fused["samples"],
            }
        )
    return cells


def project(cells, enabled_widths):
    """Directional per-round projection from the current-tree census.

    A scored 512-token leg sweeps the cache offset from 512 to 1024, so the
    two measured cells bracket the leg. Averaging them is an interpolation
    WITHIN the one-pass family; the kL = 1024 family boundary is never
    crossed, and the fused path declines above it.
    """
    by_m = {}
    for cell in cells:
        if cell["m"] not in enabled_widths:
            continue
        by_m.setdefault(cell["m"], []).append(cell["recovered_us"])
    per_m = {}
    total_us_per_round = 0.0
    for m, recovered in sorted(by_m.items()):
        rounds = ROUNDS_BY_M.get(m, 0)
        mean_recovered = statistics.fmean(recovered)
        contribution = (
            rounds / TOTAL_ROUNDS * FULL_ATTENTION_LAYERS * mean_recovered
        )
        per_m[str(m)] = {
            "rounds": rounds,
            "mean_recovered_us_per_layer": mean_recovered,
            "us_per_round": contribution,
        }
        total_us_per_round += contribution
    return {
        "enabled_widths": sorted(enabled_widths),
        "per_m": per_m,
        "us_per_round": total_us_per_round,
        "ms_per_round": total_us_per_round / 1000.0,
        "mue": total_us_per_round / MUE_US_PER_ROUND,
        "total_rounds": TOTAL_ROUNDS,
        "full_attention_layers": FULL_ATTENTION_LAYERS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="research/e198-pipeline.jsonl")
    parser.add_argument("--exactness", default="research/e198-exactness.json")
    parser.add_argument("--timing", default="research/e198-timing.json")
    parser.add_argument("--out", default="research/e198-analysis.json")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    report = {
        "probe": "e198-route1-row-amortized-kernel",
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "depth_histogram": {str(k): v for k, v in DEPTH_HISTOGRAM.items()},
        "rounds_by_m": {str(k): v for k, v in ROUNDS_BY_M.items()},
        "mue_us_per_round": MUE_US_PER_ROUND,
        "full_attention_layers": FULL_ATTENTION_LAYERS,
    }

    if Path(args.pipeline).exists():
        report["register_curve"] = register_curve(load_lines(args.pipeline))

    if Path(args.exactness).exists():
        exactness = load(args.exactness)
        report["exactness"] = exactness
        all_cells = exactness.get("cells", [])
        served = [cell for cell in all_cells if cell.get("served")]
        declined = [cell for cell in all_cells if not cell.get("served")]
        report["exactness_verdict"] = {
            "cells": len(all_cells),
            "served_cells": len(served),
            "declined_cells": len(declined),
            "all_zero_differing": all(
                cell["differing_elements"] == 0 for cell in served
            ),
            "control_fired_everywhere": all(
                cell["control_differing_elements"] > 1000 for cell in served
            ),
            "declines_only_at_two_pass_boundary": all(
                cell.get("declined_at_two_pass_boundary") for cell in declined
            ),
            "served_key_lengths": sorted(cell["kL"] for cell in served),
        }

    if Path(args.timing).exists():
        timing = load(args.timing)
        reduced = reduce_samples(timing)
        report["probe_needle"] = timing.get("probe_needle")
        report["blocks"] = timing.get("blocks")
        report["reps"] = timing.get("reps")
        report["chain"] = timing.get("chain")
        report["thermal_record"] = thermal_record(timing)
        report["modes"] = {}
        for mode in ("serial", "indep"):
            cells = price(reduced, mode)
            if not cells:
                continue
            passing = {
                cell["m"] for cell in cells if cell["gate_pass_half_headroom"]
            }
            report["modes"][mode] = {
                "cells": cells,
                "stage2_gate_passing_widths": sorted(passing),
                "projection_all_measured": project(
                    cells, {cell["m"] for cell in cells}
                ),
                "projection_gate_passing": project(cells, passing),
            }

    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "exactness"}, indent=1)[:4000])

    if args.wandb:
        import wandb

        run = wandb.init(
            project="qwen38-mlx-challenge-senpai",
            entity="wandb-applied-ai-team",
            name="e198-route1-row-amortized-kernel",
            job_type="probe",
            config={
                "experiment": "E198",
                "harness": "local",
                "mechanism": "fused row-amortized one-pass SDPA",
                "depth_histogram": report["depth_histogram"],
                "mue_us_per_round": MUE_US_PER_ROUND,
                "full_attention_layers": FULL_ATTENTION_LAYERS,
                "cool_gate_passed_real_gate": False,
                "gate_qualified_for_timing": False,
                "blocks": report.get("blocks"),
                "reps": report.get("reps"),
            },
        )
        summary = {}
        for name, record in report.get("register_curve", {}).items():
            width = record["width_from_name"]
            summary[f"register/m{width}/max_threads"] = record[
                "max_total_threads_per_threadgroup"
            ]
            summary[f"register/m{width}/threadgroup_memory_bytes"] = record[
                "static_threadgroup_memory_bytes"
            ]
            summary[f"register/m{width}/fits_1024"] = record[
                "fits_required_threadgroup"
            ]
        verdict = report.get("exactness_verdict")
        if verdict:
            summary["exactness/all_zero_differing"] = verdict["all_zero_differing"]
            summary["exactness/control_fired"] = verdict["control_fired_everywhere"]
            summary["exactness/declines_only_at_two_pass_boundary"] = verdict[
                "declines_only_at_two_pass_boundary"
            ]
        thermal = report.get("thermal_record", {})
        if thermal.get("available"):
            summary["thermal/entry_spread_c"] = thermal["entry_spread_c"]
            summary["thermal/entry_min_c"] = thermal["entry_min_c"]
            summary["thermal/entry_max_c"] = thermal["entry_max_c"]
        for mode, payload in report.get("modes", {}).items():
            for cell in payload["cells"]:
                tag = f"{mode}/m{cell['m']}/kv{cell['kv']}"
                summary[f"{tag}/pair_us"] = cell["pair_us"]
                summary[f"{tag}/shipped_us"] = cell["shipped_us"]
                summary[f"{tag}/slice_concat_us"] = cell["slice_concat_us"]
                summary[f"{tag}/fused_us"] = cell["fused_us"]
                summary[f"{tag}/floor_us"] = cell["floor_us"]
                summary[f"{tag}/recovered_us"] = cell["recovered_us"]
                summary[f"{tag}/recovered_fraction"] = cell[
                    "recovered_fraction_of_headroom"
                ]
                summary[f"{tag}/amortization_efficiency"] = cell[
                    "amortization_efficiency"
                ]
            summary[f"{mode}/stage2_gate_passing_widths"] = str(
                payload["stage2_gate_passing_widths"]
            )
            summary[f"{mode}/projection/ms_per_round"] = payload[
                "projection_gate_passing"
            ]["ms_per_round"]
            summary[f"{mode}/projection/mue"] = payload["projection_gate_passing"][
                "mue"
            ]
        run.summary.update(summary)
        print("wandb run:", run.url, run.id)
        run.finish()


if __name__ == "__main__":
    main()
