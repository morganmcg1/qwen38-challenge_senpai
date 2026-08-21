#!/usr/bin/env python3
"""E92 rung 2: device time and achieved bandwidth of the target verify pass.

Reads one width-pinned leg per tag and reports, per verify width `M`:

    verify GPU busy   `verify_graph` + `eval_wall`, head-free under --sync-head
    head GPU busy     the five head-chain windows, and `d_submit2` alone
    round GPU busy    and round idle, from the E90 ledger
    implied bytes     verifyGPUbusy(M) / verifyGPUbusy(1) x 14.4123 GB

The implied-bytes column is model-free. It needs no byte accounting to be
right; it only needs achieved bandwidth to be roughly constant across widths,
which the rung 1 residency sweep tests directly.

    usage: research/e92_widths.py TAG [TAG ...] [--skip-rounds N] [--output P]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from e90_intervals import (
    ANCHORS,
    HOST_PHASES,
    HOST_STUCK_US,
    busy_between,
    read_intervals,
    read_meta,
    read_rounds,
    union,
)

# The full 4-bit affine backbone plus head weight stream of one target forward,
# as the campaign has priced it since E68.
WEIGHT_STREAM_BYTES = 14_412_349_440

# Live input-group count per verify width, from the `switch (ntg.x)` dispatch
# table in Vendor/mlx-swift/.../kernels/quantized.h and confirmed against the
# persisted live table in research/e88-artifacts/rung01.json.
INPUT_GROUPS = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3}

HEAD_WINDOWS = ("d_flush", "d_head1", "d_submit1", "d_chain", "d_submit2")
VERIFY_WINDOWS = ("verify_graph", "eval_wall")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def analyse_leg(tag: str, skip_rounds: int) -> dict:
    rounds = read_rounds(tag)
    by_pid, buffer_stats = read_intervals(tag)
    meta = read_meta(tag)

    # A pinned leg holds one width for every round, including width 1, so the
    # E90 rule of "the pid with the most drafting rounds" cannot be used here.
    counts: dict[int, int] = {}
    for record in rounds:
        counts[record["pid"]] = counts.get(record["pid"], 0) + 1
    if not counts:
        raise SystemExit("e92_widths: %s has no traced rounds" % tag)
    pid = max(counts, key=lambda key: counts[key])

    starts, ends, prefix = union(by_pid.get(pid, []))
    selected = [r for r in rounds if r["pid"] == pid and r["round"] > skip_rounds]
    if not selected:
        raise SystemExit("e92_widths: %s has no post-warmup rounds" % tag)

    per_round = []
    for record in selected:
        anchors = record["anchors"]
        row = {
            "round": record["round"],
            "d": record["d"],
            "acc": record["acc"],
            "round_us": (anchors["t_tail_done"] - anchors["t_round0"]) / 1000.0,
            "host_thread_cpu_ns": record.get("trace", {}).get("host_thread_cpu_ns"),
        }
        window_sum = 0
        busy_sum = 0
        for lo_name, hi_name, label in ANCHORS:
            lo, hi = anchors[lo_name], anchors[hi_name]
            span = max(0, hi - lo)
            busy = busy_between(starts, ends, prefix, lo, hi)
            row["%s_us" % label] = span / 1000.0
            row["%s_gpu_busy_us" % label] = busy / 1000.0
            window_sum += span
            busy_sum += busy
        row["gpu_busy_us_total"] = busy_sum / 1000.0
        row["gpu_idle_us_total"] = (window_sum - busy_sum) / 1000.0
        row["tiling_error_us"] = (window_sum / 1000.0) - row["round_us"]
        row["verify_gpu_busy_us"] = sum(
            row["%s_gpu_busy_us" % label] for label in VERIFY_WINDOWS)
        row["head_gpu_busy_us"] = sum(
            row["%s_gpu_busy_us" % label] for label in HEAD_WINDOWS)
        row["host_phase_sum_us"] = sum(row["%s_us" % p] for p in HOST_PHASES)
        row["host_stuck"] = row["host_phase_sum_us"] > HOST_STUCK_US
        per_round.append(row)

    histogram: dict[int, int] = {}
    for row in per_round:
        histogram[row["d"]] = histogram.get(row["d"], 0) + 1
    dominant = max(histogram, key=lambda key: histogram[key])
    on_pin = [row for row in per_round if row["d"] == dominant]
    stuck = [row for row in on_pin if row["host_stuck"]]
    clean = [row for row in on_pin if not row["host_stuck"]]

    width = dominant + 1
    verify = median([row["verify_gpu_busy_us"] for row in on_pin])
    head = median([row["head_gpu_busy_us"] for row in on_pin])
    return {
        "tag": tag,
        "pid": pid,
        "meta": meta,
        "buffer_stats": buffer_stats.get(pid, {}),
        "skip_rounds": skip_rounds,
        "rounds_analysed": len(per_round),
        "width_histogram": {str(k + 1): v for k, v in sorted(histogram.items())},
        "width_is_delta": len(histogram) == 1,
        "pin_purity": len(on_pin) / len(per_round),
        "M": width,
        "rounds_on_pin": len(on_pin),
        "frac_rounds_host_stuck": len(stuck) / len(on_pin) if on_pin else None,
        "accepted_mean": statistics.mean([row["acc"] for row in on_pin]),
        "round_us": median([row["round_us"] for row in on_pin]),
        "round_gpu_busy_us": median([row["gpu_busy_us_total"] for row in on_pin]),
        "round_gpu_idle_us": median([row["gpu_idle_us_total"] for row in on_pin]),
        "tiling_error_us": median([row["tiling_error_us"] for row in on_pin]),
        "verify_gpu_busy_us": verify,
        "head_gpu_busy_us": head,
        "d_submit2_gpu_busy_us": median(
            [row["d_submit2_gpu_busy_us"] for row in on_pin]),
        "snapshot_gpu_busy_us": median(
            [row["snapshot_gpu_busy_us"] for row in on_pin]),
        "verify_gpu_busy_us_clean": median(
            [row["verify_gpu_busy_us"] for row in clean]),
        "verify_gpu_busy_us_stuck": median(
            [row["verify_gpu_busy_us"] for row in stuck]),
        "host_thread_cpu_ns": median(
            [row["host_thread_cpu_ns"] for row in on_pin
             if row["host_thread_cpu_ns"] is not None]),
    }


def combine(legs: list[dict], peak_gbs: float | None) -> dict:
    by_width: dict[int, list[dict]] = {}
    for leg in legs:
        by_width.setdefault(leg["M"], []).append(leg)

    anchor = None
    if 1 in by_width:
        anchor = median([leg["verify_gpu_busy_us"] for leg in by_width[1]])

    table = []
    for width in sorted(by_width):
        group = by_width[width]
        verify = median([leg["verify_gpu_busy_us"] for leg in group])
        head = median([leg["head_gpu_busy_us"] for leg in group])
        round_busy = median([leg["round_gpu_busy_us"] for leg in group])
        groups = INPUT_GROUPS.get(width)
        modelled_weight = WEIGHT_STREAM_BYTES * groups if groups else None
        row = {
            "M": width,
            "legs": [leg["tag"] for leg in group],
            "verify_gpu_busy_us": verify,
            "head_gpu_busy_us": head,
            "round_gpu_busy_us": round_busy,
            "round_gpu_idle_us": median(
                [leg["round_gpu_idle_us"] for leg in group]),
            "round_us": median([leg["round_us"] for leg in group]),
            "verify_share_of_round_busy": verify / round_busy if round_busy else None,
            "head_share_of_round_busy": head / round_busy if round_busy else None,
            "G": groups,
            "modelled_weight_bytes": modelled_weight,
            "achieved_bandwidth_gbs": (
                modelled_weight / (verify * 1000.0) if modelled_weight and verify
                else None),
        }
        if peak_gbs and row["achieved_bandwidth_gbs"]:
            row["ratio_to_peak"] = row["achieved_bandwidth_gbs"] / peak_gbs
        if anchor:
            implied = verify / anchor * WEIGHT_STREAM_BYTES
            row["implied_bytes"] = implied
            row["implied_over_weight_stream"] = implied / WEIGHT_STREAM_BYTES
            if groups:
                row["implied_over_G_times_stream"] = implied / (
                    WEIGHT_STREAM_BYTES * groups)
        table.append(row)
    return {"anchor_verify_gpu_busy_us": anchor, "table": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--skip-rounds", type=int, default=8)
    parser.add_argument("--peak-gbs", type=float)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    legs = [analyse_leg(tag, arguments.skip_rounds) for tag in arguments.tags]
    combined = combine(legs, arguments.peak_gbs)

    print("%-10s %3s %6s %8s %11s %9s %10s %8s %9s %9s"
          % ("tag", "M", "rounds", "pin%", "verify_us", "head_us", "round_us",
             "busy_us", "idle_us", "stuck"))
    for leg in legs:
        print("%-10s %3d %6d %8.3f %11.1f %9.1f %10.1f %8.1f %9.1f %9.3f"
              % (leg["tag"], leg["M"], leg["rounds_on_pin"], leg["pin_purity"],
                 leg["verify_gpu_busy_us"], leg["head_gpu_busy_us"],
                 leg["round_us"], leg["round_gpu_busy_us"],
                 leg["round_gpu_idle_us"], leg["frac_rounds_host_stuck"]))
    print()
    print("%3s %11s %9s %3s %8s %10s %9s %12s %10s"
          % ("M", "verify_us", "head_us", "G", "GB/s", "ratio_peak",
             "verify%", "implied_GB", "impl/G"))
    for row in combined["table"]:
        print("%3d %11.1f %9.1f %3s %8s %10s %9.3f %12s %10s"
              % (row["M"], row["verify_gpu_busy_us"], row["head_gpu_busy_us"],
                 row["G"],
                 "%.1f" % row["achieved_bandwidth_gbs"]
                 if row["achieved_bandwidth_gbs"] else "-",
                 "%.3f" % row["ratio_to_peak"] if row.get("ratio_to_peak") else "-",
                 row["verify_share_of_round_busy"],
                 "%.2f" % (row["implied_bytes"] / 1e9)
                 if row.get("implied_bytes") else "-",
                 "%.3f" % row["implied_over_G_times_stream"]
                 if row.get("implied_over_G_times_stream") else "-"))

    result = {"legs": legs, **combined}
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
