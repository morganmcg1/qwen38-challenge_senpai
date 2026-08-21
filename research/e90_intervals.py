#!/usr/bin/env python3
"""E90 rung 0b: place GPU busy and GPU idle inside the anchors of one round.

usage: research/e90_intervals.py TAG [TAG ...] [--skip-rounds N] [--output PATH]

Two files per leg, both written on the same mach uptime clock:

  research/out/TAG/trace.txt            `mtp-anchor:` lines, one per round
  research/out/TAG/gpu-intervals.jsonl  one GPU execution interval per
                                        completed command buffer

The device intervals are merged into a union first, so two command buffers that
overlap on the GPU count once. Each round's inter-anchor window is then
intersected with that union. `gpu_busy_us` plus `gpu_idle_us` equals the window
by construction, and the windows tile `round_us` exactly, which is the
instrument's own consistency check.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

TRACE_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
ANCHOR_RE = re.compile(r"^mtp-anchor: (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")

# Ordered anchor names. Consecutive pairs are the intervals of the round.
ANCHORS = [
    ("t_round0", "t_draft0", "d_pre"),
    ("t_draft0", "t_flush_built", "d_flush"),
    ("t_flush_built", "t_head1_built", "d_head1"),
    ("t_head1_built", "t_submit1", "d_submit1"),
    ("t_submit1", "t_chain_built", "d_chain"),
    ("t_chain_built", "t_draft_built", "d_submit2"),
    ("t_draft_built", "t_snapshot_done", "snapshot"),
    ("t_snapshot_done", "t_verify_built", "verify_graph"),
    ("t_verify_built", "t_eval_done", "eval_wall"),
    ("t_eval_done", "t_read_done", "readout"),
    ("t_read_done", "t_commit_done", "commit"),
    ("t_commit_done", "t_tail_done", "upkeep"),
]


def read_meta(tag: str) -> dict:
    path = OUT / tag / "meta.txt"
    if not path.exists():
        return {}
    return dict(
        line.partition("=")[::2] for line in path.read_text().splitlines() if "=" in line
    )


def read_rounds(tag: str) -> list[dict]:
    """Pair each `mtp-trace` line with the `mtp-anchor` line that follows it."""
    rounds = []
    pending = None
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        trace = TRACE_RE.match(line)
        if trace:
            fields = {k: float(v) for k, v in KV_RE.findall(trace.group(4))}
            pending = {
                "round": int(trace.group(1)),
                "d": int(trace.group(2)),
                "acc": int(trace.group(3)),
                "trace": fields,
            }
            continue
        anchor = ANCHOR_RE.match(line)
        if not anchor:
            continue
        fields = {k: int(float(v)) for k, v in KV_RE.findall(anchor.group(1))}
        record = {
            "round": fields.pop("round"),
            "d": fields.pop("d"),
            "acc": fields.pop("acc"),
            "pid": fields.pop("pid"),
            "anchors": fields,
        }
        if pending and pending["round"] == record["round"]:
            record["trace"] = pending["trace"]
        pending = None
        rounds.append(record)
    return rounds


def read_intervals(tag: str) -> dict[int, list[tuple[int, int]]]:
    path = OUT / tag / "gpu-intervals.jsonl"
    by_pid: dict[int, list[tuple[int, int]]] = {}
    stats: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "e90_intervals":
            continue
        pid = record["pid"]
        by_pid.setdefault(pid, []).extend(
            zip(record["gpu_start_ns"], record["gpu_end_ns"])
        )
        stats[pid] = {
            "committed_total": record["committed_total"],
            "completed_total": record["completed_total"],
            "invalid_total": record["invalid_total"],
        }
    return by_pid, stats


def union(intervals: list[tuple[int, int]]) -> tuple[list[int], list[int], list[int]]:
    """Merge overlaps and return starts, ends and a prefix sum of busy time."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    starts = [m[0] for m in merged]
    ends = [m[1] for m in merged]
    prefix = [0]
    for start, end in merged:
        prefix.append(prefix[-1] + end - start)
    return starts, ends, prefix


def busy_between(starts, ends, prefix, lo: int, hi: int) -> int:
    """Union busy nanoseconds inside [lo, hi)."""
    if hi <= lo or not starts:
        return 0
    index = bisect_left(ends, lo)
    total = 0
    while index < len(starts) and starts[index] < hi:
        total += min(ends[index], hi) - max(starts[index], lo)
        index += 1
    return max(0, total)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": st.median(values),
        "mean": st.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def analyse(tag: str, skip_rounds: int) -> dict:
    rounds = read_rounds(tag)
    by_pid, stats = read_intervals(tag)
    meta = read_meta(tag)

    # The production leg is the drafting worker: the pid with the most rounds
    # that proposed drafts. A depth-0 serial control leg in the same file has
    # d=0 for every round and is reported separately.
    counts: dict[tuple[int, bool], int] = {}
    for record in rounds:
        key = (record["pid"], record["d"] > 0)
        counts[key] = counts.get(key, 0) + 1
    drafting = [(count, pid) for (pid, drafts), count in counts.items() if drafts]
    if not drafting:
        raise SystemExit("e90_intervals: %s has no drafting rounds" % tag)
    pid = max(drafting)[1]

    starts, ends, prefix = union(by_pid.get(pid, []))
    selected = [r for r in rounds if r["pid"] == pid and r["round"] > skip_rounds]
    if not selected:
        raise SystemExit("e90_intervals: %s has no post-warmup rounds" % tag)

    per_round = []
    previous_tail = None
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
            row["%s_gpu_idle_us" % label] = (span - busy) / 1000.0
            window_sum += span
            busy_sum += busy
        row["gpu_busy_us_total"] = busy_sum / 1000.0
        row["gpu_idle_us_total"] = (window_sum - busy_sum) / 1000.0
        row["gpu_idle_us_within_d_submit2"] = row["d_submit2_gpu_idle_us"]
        row["interval_sum_us"] = window_sum / 1000.0
        row["tiling_error_us"] = row["interval_sum_us"] - row["round_us"]
        if previous_tail is not None:
            gap_lo, gap_hi = previous_tail, anchors["t_round0"]
            gap = max(0, gap_hi - gap_lo)
            gap_busy = busy_between(starts, ends, prefix, gap_lo, gap_hi)
            row["inter_round_gap_us"] = gap / 1000.0
            row["inter_round_gap_gpu_busy_us"] = gap_busy / 1000.0
            row["inter_round_gap_gpu_idle_us"] = (gap - gap_busy) / 1000.0
        previous_tail = anchors["t_tail_done"]
        per_round.append(row)

    keys = [k for k in per_round[0] if k not in ("round", "d", "acc")]
    aggregate = {}
    for key in keys:
        values = [r[key] for r in per_round if r.get(key) is not None]
        aggregate[key] = summarize(values)

    return {
        "tag": tag,
        "pid": pid,
        "skip_rounds": skip_rounds,
        "rounds_analysed": len(per_round),
        "rounds_total": len([r for r in rounds if r["pid"] == pid]),
        "buffer_stats": stats.get(pid, {}),
        "union_intervals": len(starts),
        "union_busy_us": prefix[-1] / 1000.0 if prefix else 0.0,
        "meta": {
            k: meta.get(k)
            for k in (
                "tag",
                "tokens",
                "sync_head",
                "cool_gate_passed_real_gate",
                "gate_qualified_for_timing",
                "base_sha",
                "worker_sha256",
                "head_dir",
                "gpu_temp_entry_c",
                "gpu_temp_exit_c",
                "arm",
            )
            if k in meta
        },
        "aggregate": aggregate,
        "per_round": per_round,
    }


def print_table(result: dict) -> None:
    agg = result["aggregate"]
    print("=== %s (pid %d, %d post-warmup rounds of %d) ==="
          % (result["tag"], result["pid"], result["rounds_analysed"],
             result["rounds_total"]))
    print("%-14s %12s %12s %12s %7s" % ("interval", "median_us", "gpu_busy_us",
                                        "gpu_idle_us", "idle_%"))
    for _, _, label in ANCHORS:
        span = agg["%s_us" % label]["median"]
        busy = agg["%s_gpu_busy_us" % label]["median"]
        idle = agg["%s_gpu_idle_us" % label]["median"]
        share = 100.0 * idle / span if span else 0.0
        print("%-14s %12.1f %12.1f %12.1f %6.1f%%" % (label, span, busy, idle, share))
    if "inter_round_gap_us" in agg:
        span = agg["inter_round_gap_us"]["median"]
        busy = agg["inter_round_gap_gpu_busy_us"]["median"]
        idle = agg["inter_round_gap_gpu_idle_us"]["median"]
        print("%-14s %12.1f %12.1f %12.1f %6.1f%%"
              % ("(gap)", span, busy, idle, 100.0 * idle / span if span else 0.0))
    print("%-14s %12.1f %12.1f %12.1f %6.1f%%"
          % ("ROUND", agg["round_us"]["median"], agg["gpu_busy_us_total"]["median"],
             agg["gpu_idle_us_total"]["median"],
             100.0 * agg["gpu_idle_us_total"]["median"] / agg["round_us"]["median"]))
    print("tiling error median %.3f us (must be 0)" % agg["tiling_error_us"]["median"])
    if agg.get("host_thread_cpu_ns", {}).get("n"):
        print("host_thread_cpu_ns median %.0f (%.1f us)"
              % (agg["host_thread_cpu_ns"]["median"],
                 agg["host_thread_cpu_ns"]["median"] / 1000.0))
    print("buffers: %s, union intervals %d"
          % (result["buffer_stats"], result["union_intervals"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--skip-rounds", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()

    results = []
    for tag in args.tags:
        result = analyse(tag, args.skip_rounds)
        print_table(result)
        print()
        results.append(result)

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
