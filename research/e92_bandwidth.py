#!/usr/bin/env python3
"""E92 rung 1: in-session read bandwidth against buffer size.

Joins `research/out/TAG/e92-bandwidth.jsonl`, one record per probe repetition,
with `gpu-intervals.jsonl`, the GPU execution interval of every command buffer
the same process submitted. Each repetition therefore reports DEVICE time, not
wall time: `wall_us` is kept beside it only as a bound.

    usage: research/e92_bandwidth.py TAG [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from e90_intervals import OUT, busy_between, read_intervals, read_meta, union

GB = 1_000_000_000.0


def read_probes(tag: str) -> list[dict]:
    path = OUT / tag / "e92-bandwidth.jsonl"
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "e92_bw":
            records.append(record)
    return records


def analyse(tag: str) -> dict:
    probes = read_probes(tag)
    if not probes:
        raise SystemExit("e92_bandwidth: %s has no probe records" % tag)
    by_pid, buffer_stats = read_intervals(tag)

    # The probe fires in every worker the leg spawns. Analyse the process that
    # recorded the most repetitions; a reference worker that fired the probe
    # before dying leaves a shorter record.
    counts: dict[int, int] = {}
    for record in probes:
        counts[record["pid"]] = counts.get(record["pid"], 0) + 1
    pid = max(counts, key=lambda key: counts[key])
    probes = [record for record in probes if record["pid"] == pid]
    starts, ends, prefix = union(by_pid.get(pid, []))

    cells: dict[tuple[str, int], dict] = {}
    for record in probes:
        key = (record["kind"], record["megabytes"])
        cell = cells.setdefault(
            key,
            {
                "kind": record["kind"],
                "megabytes": record["megabytes"],
                "bytes": record["bytes"],
                "gpu_busy_us": [],
                "wall_us": [],
            },
        )
        gpu_ns = busy_between(starts, ends, prefix, record["t0_ns"], record["t1_ns"])
        cell["gpu_busy_us"].append(gpu_ns / 1000.0)
        cell["wall_us"].append(float(record["wall_us"]))

    rows = []
    for key in sorted(cells, key=lambda k: (k[0], k[1])):
        cell = cells[key]
        gpu = statistics.median(cell["gpu_busy_us"])
        wall = statistics.median(cell["wall_us"])
        rows.append(
            {
                "kind": cell["kind"],
                "megabytes": cell["megabytes"],
                "bytes": cell["bytes"],
                "reps": len(cell["gpu_busy_us"]),
                "gpu_busy_us_median": gpu,
                "gpu_busy_us_min": min(cell["gpu_busy_us"]),
                "gpu_busy_us_max": max(cell["gpu_busy_us"]),
                "wall_us_median": wall,
                "achieved_bandwidth_gbs": cell["bytes"] / (gpu * 1000.0) if gpu else None,
                "wall_bandwidth_gbs": cell["bytes"] / (wall * 1000.0) if wall else None,
                "gpu_fraction_of_wall": gpu / wall if wall else None,
            }
        )

    # Marginal rate between adjacent sizes of the same dtype: the bytes added
    # divided by the device time added. A flat curve makes this equal to the
    # average rate; a curve that bends does not.
    for kind in sorted({row["kind"] for row in rows}):
        ladder = [row for row in rows if row["kind"] == kind]
        for previous, current in zip(ladder, ladder[1:]):
            delta_bytes = current["bytes"] - previous["bytes"]
            delta_us = current["gpu_busy_us_median"] - previous["gpu_busy_us_median"]
            current["marginal_from_mb"] = previous["megabytes"]
            current["marginal_bytes"] = delta_bytes
            current["marginal_gpu_busy_us"] = delta_us
            current["marginal_bandwidth_gbs"] = (
                delta_bytes / (delta_us * 1000.0) if delta_us > 0 else None
            )

    streamed = [row for row in rows if row["kind"] != "bfloat16"]
    rates = [row["achieved_bandwidth_gbs"] for row in streamed]
    spread = (max(rates) - min(rates)) / max(rates) if rates else None

    return {
        "tag": tag,
        "pid": pid,
        "meta": read_meta(tag),
        "buffer_stats": buffer_stats.get(pid),
        "rows": rows,
        "peak_achieved_bandwidth_gbs": max(rates) if rates else None,
        "min_achieved_bandwidth_gbs": min(rates) if rates else None,
        "relative_spread": spread,
        "flat_within_10_percent": (spread is not None and spread <= 0.10),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    result = analyse(arguments.tag)
    print("=== %s (pid %d) ===" % (result["tag"], result["pid"]))
    print(
        "%-9s %7s %13s %13s %10s %10s %10s"
        % ("kind", "MB", "gpu_busy_us", "wall_us", "GB/s", "marg GB/s", "gpu/wall")
    )
    for row in result["rows"]:
        print(
            "%-9s %7d %13.1f %13.1f %10.1f %10s %10.3f"
            % (
                row["kind"],
                row["megabytes"],
                row["gpu_busy_us_median"],
                row["wall_us_median"],
                row["achieved_bandwidth_gbs"],
                (
                    "%.1f" % row["marginal_bandwidth_gbs"]
                    if row.get("marginal_bandwidth_gbs")
                    else "-"
                ),
                row["gpu_fraction_of_wall"],
            )
        )
    print(
        "peak %.1f GB/s   min %.1f GB/s   spread %.1f %%   flat_within_10_percent=%s"
        % (
            result["peak_achieved_bandwidth_gbs"],
            result["min_achieved_bandwidth_gbs"],
            100.0 * result["relative_spread"],
            result["flat_within_10_percent"],
        )
    )
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
