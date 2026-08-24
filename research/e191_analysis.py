#!/usr/bin/env python3
"""E191 Stage 0: reduce the dispatch census and the ABBA timing probe.

Answers three questions from the two probe JSON files:

  1. Does the shipped m >= 6 split path materialise a KV copy (mechanism M1)?
     Evidence: copy_* dispatches per form in the census.
  2. Does the recorded two-append candidateFix remove any cost?
     Evidence: today - twoAppend absolute microseconds, ABBA-counterbalanced.
  3. What is the PRODUCTION m = 5 -> 6 step, and does FINDING 489's
     +87.7 / +173.3 us transfer to it?
     Evidence: today(m) - unsplit(5) versus unsplit(m) - unsplit(5).

harness=local. Within-session relative measurement, no thermal gate, no score.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict

FULL_ATTENTION_LAYERS = 16
MUE_US_PER_ROUND = 567.0  # minimum useful effect, senpai/frontier-state.json


def load(path):
    with open(path) as handle:
        return json.load(handle)


def reduce_census(census):
    rows = []
    for cell in census["cells"]:
        rows.append(
            {
                "form": cell["form"],
                "query_layout": cell["query_layout"],
                "qL": cell["qL"],
                "kv": cell["kv"],
                "dispatches": cell["dispatches"],
                "copy_dispatches": cell["copy_dispatches"],
                "kernels": cell["kernel_counts"],
                "sequence": cell["kernel_sequence"],
            }
        )
    rows.sort(key=lambda r: (r["query_layout"], r["kv"], r["qL"], r["form"]))
    return rows


def block_stats(samples):
    """Per-cell mean over ABBA blocks, plus the between-block spread."""
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["form"], sample["qL"], sample["kv"])].append(
            sample["microseconds"]
        )
    stats = {}
    for key, values in grouped.items():
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[key] = {
            "mean_us": mean,
            "median_us": statistics.median(values),
            "sd_us": sd,
            "sem_us": sd / math.sqrt(len(values)) if values else 0.0,
            "n_blocks": len(values),
            "min_us": min(values),
        }
    return stats


def per_round_ms(delta_us):
    """One full-attention layer microsecond delta -> whole-round milliseconds."""
    return delta_us * FULL_ATTENTION_LAYERS / 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--census")
    parser.add_argument("--timing")
    parser.add_argument("--out")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="e191-stage0")
    parser.add_argument("--wandb-notes", default="")
    args = parser.parse_args()

    report = {"probe": "e191-stage0", "harness": "local"}

    if args.census:
        census = load(args.census)
        rows = reduce_census(census)
        report["host_architecture"] = census["host_architecture"]
        report["census"] = rows
        scored = [r for r in rows if r["query_layout"] == "contiguous"]

        def pick(form, kv, qL, field):
            for row in scored:
                if row["form"] == form and row["kv"] == kv and row["qL"] == qL:
                    return row[field]
            return None

        report["census_verdict"] = {
            "scored_query_layout": "contiguous",
            "extra_dispatches_two_append_minus_today": {
                f"kv{r['kv']}_m{r['qL']}": pick(
                    "twoAppend", r["kv"], r["qL"], "dispatches"
                )
                - r["dispatches"]
                for r in scored
                if r["form"] == "today"
            },
            "extra_copies_two_append_minus_today": {
                f"kv{r['kv']}_m{r['qL']}": pick(
                    "twoAppend", r["kv"], r["qL"], "copy_dispatches"
                )
                - r["copy_dispatches"]
                for r in scored
                if r["form"] == "today"
            },
        }
        print("\n== dispatch census ==")
        print(f"host: {census['host_architecture']}")
        header = (
            f"{'layout':<15} {'kv':>5} {'qL':>3} {'form':<10} {'disp':>5} "
            f"{'copy':>5}  kernels"
        )
        print(header)
        for row in rows:
            kernels = ", ".join(
                f"{k}x{v}" for k, v in sorted(row["kernels"].items())
            )
            print(
                f"{row['query_layout']:<15} {row['kv']:>5} {row['qL']:>3} "
                f"{row['form']:<10} {row['dispatches']:>5} "
                f"{row['copy_dispatches']:>5}  {kernels}"
            )

    if args.timing:
        timing = load(args.timing)
        stats = block_stats(timing["samples"])
        report["gpu_temperature_c"] = timing["gpu_temperature_c"]
        report["cool_gate_passed_real_gate"] = timing["cool_gate_passed_real_gate"]
        report["gate_qualified_for_timing"] = timing["gate_qualified_for_timing"]
        report["abba_counterbalanced"] = timing["abba_counterbalanced"]
        report["agreement"] = timing["agreement"]
        report["blocks"] = timing["blocks"]
        report["reps"] = timing["reps"]

        cells = []
        for (form, qL, kv), value in sorted(stats.items(), key=lambda i: (i[0][2], i[0][1], i[0][0])):
            cells.append({"form": form, "qL": qL, "kv": kv, **value})
        report["timing_cells"] = cells

        print("\n== absolute time per call (us), ABBA blocks ==")
        print(f"{'kv':>5} {'qL':>3} {'form':<10} {'mean':>9} {'sd':>7} {'sem':>7}")
        for cell in cells:
            print(
                f"{cell['kv']:>5} {cell['qL']:>3} {cell['form']:<10} "
                f"{cell['mean_us']:>9.2f} {cell['sd_us']:>7.2f} {cell['sem_us']:>7.2f}"
            )

        derived = []
        for kv in sorted({c["kv"] for c in cells}):
            control = stats.get(("unsplit", 5, kv))
            for qL in [6, 7, 8, 9]:
                today = stats.get(("today", qL, kv))
                two = stats.get(("twoAppend", qL, kv))
                unsplit = stats.get(("unsplit", qL, kv))
                if not (today and two and unsplit and control):
                    continue
                fix_gain_us = today["mean_us"] - two["mean_us"]
                prod_step_us = today["mean_us"] - control["mean_us"]
                f489_step_us = unsplit["mean_us"] - control["mean_us"]
                derived.append(
                    {
                        "kv": kv,
                        "qL": qL,
                        "two_append_gain_us_per_layer": fix_gain_us,
                        "two_append_gain_ms_per_round": per_round_ms(fix_gain_us),
                        "two_append_gain_mue": per_round_ms(fix_gain_us)
                        * 1000.0
                        / MUE_US_PER_ROUND,
                        "production_step_us_per_layer": prod_step_us,
                        "production_step_ms_per_round": per_round_ms(prod_step_us),
                        "finding489_step_us_per_layer": f489_step_us,
                        "finding489_step_ms_per_round": per_round_ms(f489_step_us),
                        "split_saves_vs_unsplit_us_per_layer": unsplit["mean_us"]
                        - today["mean_us"],
                        "split_saves_vs_unsplit_ms_per_round": per_round_ms(
                            unsplit["mean_us"] - today["mean_us"]
                        ),
                    }
                )
        report["derived"] = derived

        print("\n== derived, per full-attention layer and per round (x16) ==")
        print(
            f"{'kv':>5} {'qL':>3} {'fix gain us':>12} {'fix ms/rnd':>11} {'fix MUE':>8} "
            f"{'prod step us':>13} {'prod ms/rnd':>12} {'489 step us':>12} "
            f"{'split saves us':>15}"
        )
        for row in derived:
            print(
                f"{row['kv']:>5} {row['qL']:>3} "
                f"{row['two_append_gain_us_per_layer']:>12.2f} "
                f"{row['two_append_gain_ms_per_round']:>11.3f} "
                f"{row['two_append_gain_mue']:>8.2f} "
                f"{row['production_step_us_per_layer']:>13.2f} "
                f"{row['production_step_ms_per_round']:>12.3f} "
                f"{row['finding489_step_us_per_layer']:>12.2f} "
                f"{row['split_saves_vs_unsplit_us_per_layer']:>15.2f}"
            )

        print("\n== today vs twoAppend agreement (max abs delta) ==")
        for row in timing["agreement"]:
            print(
                f"  kv={row['kv']:>5} qL={row['qL']} "
                f"max_abs_delta={row['max_abs_delta']:.3e} "
                f"offsets today={row['today_offset_after']} "
                f"twoAppend={row['two_append_offset_after']}"
            )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
        print(f"\nwrote {args.out}")

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            job_type="probe",
            tags=["e191", "stage0", "widthSixWall", "harness=local"],
            config={
                "experiment": "E191",
                "stage": "stage0-mechanism",
                "harness": "local",
                "full_attention_layers": FULL_ATTENTION_LAYERS,
                "mue_us_per_round": MUE_US_PER_ROUND,
                "host_architecture": report.get("host_architecture"),
                "cool_gate_passed_real_gate": report.get(
                    "cool_gate_passed_real_gate"
                ),
                "gate_qualified_for_timing": report.get("gate_qualified_for_timing"),
                "abba_counterbalanced": report.get("abba_counterbalanced"),
                "blocks": report.get("blocks"),
                "reps": report.get("reps"),
                "gpu_temperature_c": report.get("gpu_temperature_c"),
            },
        )
        if "census" in report:
            table = wandb.Table(
                columns=[
                    "query_layout",
                    "kv",
                    "qL",
                    "form",
                    "dispatches",
                    "copy_dispatches",
                    "kernels",
                ]
            )
            for row in report["census"]:
                table.add_data(
                    row["query_layout"],
                    row["kv"],
                    row["qL"],
                    row["form"],
                    row["dispatches"],
                    row["copy_dispatches"],
                    json.dumps(row["kernels"], sort_keys=True),
                )
            run.log({"dispatch_census": table})
        if "timing_cells" in report:
            table = wandb.Table(
                columns=["kv", "qL", "form", "mean_us", "median_us", "sd_us", "sem_us"]
            )
            for cell in report["timing_cells"]:
                table.add_data(
                    cell["kv"],
                    cell["qL"],
                    cell["form"],
                    cell["mean_us"],
                    cell["median_us"],
                    cell["sd_us"],
                    cell["sem_us"],
                )
            run.log({"timing_cells": table})
        if "derived" in report:
            columns = list(report["derived"][0].keys()) if report["derived"] else []
            table = wandb.Table(columns=columns)
            for row in report["derived"]:
                table.add_data(*[row[c] for c in columns])
            run.log({"derived": table})
            summary = {}
            for row in report["derived"]:
                tag = f"kv{row['kv']}_m{row['qL']}"
                for key in (
                    "two_append_gain_us_per_layer",
                    "two_append_gain_ms_per_round",
                    "two_append_gain_mue",
                    "production_step_us_per_layer",
                    "production_step_ms_per_round",
                    "finding489_step_us_per_layer",
                    "split_saves_vs_unsplit_ms_per_round",
                ):
                    summary[f"{tag}/{key}"] = row[key]
            run.log(summary)
            run.summary.update(summary)
        if "agreement" in report:
            table = wandb.Table(columns=["kv", "qL", "max_abs_delta"])
            worst = 0.0
            for row in report["agreement"]:
                table.add_data(row["kv"], row["qL"], row["max_abs_delta"])
                worst = max(worst, row["max_abs_delta"])
            run.log({"today_vs_twoappend_agreement": table})
            run.summary["today_vs_twoappend_max_abs_delta"] = worst
        if "census_verdict" in report:
            for group in (
                "extra_dispatches_two_append_minus_today",
                "extra_copies_two_append_minus_today",
            ):
                for key, value in report["census_verdict"][group].items():
                    run.summary[f"{group}/{key}"] = value
        print(f"wandb run: {run.url}  id={run.id}")
        run.finish()


if __name__ == "__main__":
    main()
