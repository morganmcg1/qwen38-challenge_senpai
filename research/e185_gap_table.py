#!/usr/bin/env python3
"""E185 stage 2: where the decode round's GPU idle is, and how coherent it is.

    usage: research/e185_gap_table.py TAG [--skip-rounds N] [--json OUT]

`research/e90_intervals.py` already reports GPU busy and idle per inter-anchor
window. This reader adds the two things the coherent-slice verdict needs and
that the per-phase medians cannot give:

1. **Individual gaps.** A phase's idle total can be one closeable 400 us gap or
   forty 10 us gaps that no mechanism can address. The complement of the
   command-buffer union inside the round is enumerated gap by gap, each tagged
   with the phase it starts in, so the size distribution is measured instead of
   assumed.
2. **The instrument's own share.** `upkeep` spans `t_commit_done` to
   `t_tail_done` and therefore contains the research row dump and the trace
   write, which exist only because the leg is traced. They are split out as
   `upkeep_pre`, `row_dump` and `trace_tail`, so the production-equivalent idle
   is reported beside the traced one rather than inferred from it.

harness=local. Observation only.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# Consecutive anchor pairs tiling the round. The first twelve reproduce E90's
# windows exactly, except that E90's `upkeep` is split into its production part
# and the two instrument parts.
PHASES = [
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
    ("t_commit_done", "t_row_trace0", "upkeep_pre"),
    ("t_row_trace0", "t_row_trace_done", "row_dump"),
    ("t_row_trace_done", "t_tail_done", "trace_tail"),
]
INSTRUMENT_PHASES = {"row_dump", "trace_tail"}


def read_anchors(path: Path) -> list[dict]:
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-anchor: "):
            continue
        fields = {}
        for token in line[len("mtp-anchor: ") :].split():
            key, _, value = token.partition("=")
            try:
                fields[key] = int(value)
            except ValueError:
                pass
        if "t_round0" in fields:
            rounds.append(fields)
    return rounds


def read_intervals(path: Path, pid: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if doc.get("event") != "e90_intervals" or doc.get("pid") != pid:
            continue
        for start, end in zip(doc["gpu_start_ns"], doc["gpu_end_ns"]):
            if end > start:
                spans.append((int(start), int(end)))
    spans.sort()
    return spans


def union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def busy_in(merged: list[tuple[int, int]], t0: int, t1: int) -> int:
    total = 0
    for a, b in merged:
        if b <= t0:
            continue
        if a >= t1:
            break
        total += min(b, t1) - max(a, t0)
    return total


def gaps_in(merged: list[tuple[int, int]], t0: int, t1: int) -> list[tuple[int, int]]:
    """Idle sub-intervals of [t0, t1) — the complement of the busy union."""
    out = []
    cursor = t0
    for a, b in merged:
        if b <= t0:
            continue
        if a >= t1:
            break
        if a > cursor:
            out.append((cursor, min(a, t1)))
        cursor = max(cursor, min(b, t1))
    if cursor < t1:
        out.append((cursor, t1))
    return [(a, b) for a, b in out if b > a]


def phase_of(anchors: dict, t: int) -> str:
    for lo, hi, name in PHASES:
        if anchors.get(lo, 0) <= t < anchors.get(hi, 0):
            return name
    return "inter_round_gap"


def summarise(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "median": st.median(values),
        "mean": st.fmean(values),
        "p90": sorted(values)[int(0.9 * (len(values) - 1))],
        "max": max(values),
        "total": sum(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--skip-rounds", type=int, default=8)
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--min-gap-us", type=float, default=1.0)
    args = ap.parse_args()

    out = Path("research/out") / args.tag
    meta = dict(
        line.split("=", 1)
        for line in (out / "meta.txt").read_text().splitlines()
        if "=" in line
    )
    rounds = read_anchors(out / "trace.txt")
    if not rounds:
        print("e185_gap_table: no anchors", file=__import__("sys").stderr)
        return 2

    pid = st.mode(r["pid"] for r in rounds)
    merged = union(read_intervals(out / "gpu-intervals.jsonl", pid))
    if not merged:
        print("e185_gap_table: no gpu intervals for pid %d" % pid,
              file=__import__("sys").stderr)
        return 2

    rounds = [r for r in rounds if r["pid"] == pid][args.skip_rounds :]

    per_phase: dict[str, dict[str, list[float]]] = {
        name: {"span": [], "busy": [], "idle": []} for _, _, name in PHASES
    }
    per_phase["inter_round_gap"] = {"span": [], "busy": [], "idle": []}
    round_span, round_busy, round_idle, tiling = [], [], [], []
    prod_idle = []
    all_gaps: list[dict] = []
    per_round_gap_count = []

    for i, r in enumerate(rounds):
        span_total = 0
        idle_total = 0
        instrument_idle = 0
        for lo, hi, name in PHASES:
            t0, t1 = r.get(lo, 0), r.get(hi, 0)
            if t1 < t0:
                t1 = t0
            span = t1 - t0
            busy = busy_in(merged, t0, t1)
            idle = span - busy
            per_phase[name]["span"].append(span / 1000.0)
            per_phase[name]["busy"].append(busy / 1000.0)
            per_phase[name]["idle"].append(idle / 1000.0)
            span_total += span
            idle_total += idle
            if name in INSTRUMENT_PHASES:
                instrument_idle += idle

        # The window between one round's tail and the next round's start is
        # outside every phase, so it is reported separately rather than folded
        # into the round.
        if i + 1 < len(rounds):
            t0, t1 = r["t_tail_done"], rounds[i + 1]["t_round0"]
            if t1 > t0:
                busy = busy_in(merged, t0, t1)
                per_phase["inter_round_gap"]["span"].append((t1 - t0) / 1000.0)
                per_phase["inter_round_gap"]["busy"].append(busy / 1000.0)
                per_phase["inter_round_gap"]["idle"].append((t1 - t0 - busy) / 1000.0)

        r_span = r["t_tail_done"] - r["t_round0"]
        r_busy = busy_in(merged, r["t_round0"], r["t_tail_done"])
        round_span.append(r_span / 1000.0)
        round_busy.append(r_busy / 1000.0)
        round_idle.append((r_span - r_busy) / 1000.0)
        prod_idle.append((r_span - r_busy - instrument_idle) / 1000.0)
        tiling.append((r_span - span_total) / 1000.0)

        gaps = gaps_in(merged, r["t_round0"], r["t_tail_done"])
        kept = 0
        for a, b in gaps:
            us = (b - a) / 1000.0
            if us < args.min_gap_us:
                continue
            kept += 1
            all_gaps.append(
                {"round": r.get("round"), "start_phase": phase_of(r, a), "us": us}
            )
        per_round_gap_count.append(kept)

    by_phase_gap: dict[str, list[float]] = {}
    for gap in all_gaps:
        by_phase_gap.setdefault(gap["start_phase"], []).append(gap["us"])

    n = len(rounds)
    report = {
        "tag": args.tag,
        "harness": "local",
        "official_or_ranked_score": False,
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "leg_role": "gpu-interval-ledger-observation",
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "tokens": meta.get("tokens"),
        "pid": pid,
        "rounds_analysed": n,
        "rounds_skipped": args.skip_rounds,
        "union_intervals": len(merged),
        "tiling_error_us_median": st.median(tiling),
        "tiling_error_us_max_abs": max(abs(t) for t in tiling),
        "round_us": summarise(round_span),
        "round_gpu_busy_us": summarise(round_busy),
        "round_gpu_idle_us": summarise(round_idle),
        "round_gpu_idle_us_production": summarise(prod_idle),
        "round_gpu_idle_pct_median": 100.0 * st.median(round_idle) / st.median(round_span),
        "gaps_per_round": summarise([float(c) for c in per_round_gap_count]),
        "gap_us": summarise([g["us"] for g in all_gaps]),
        "min_gap_us": args.min_gap_us,
        "phases": [
            {
                "phase": name,
                "span_us_median": st.median(per_phase[name]["span"]),
                "gpu_busy_us_median": st.median(per_phase[name]["busy"]),
                "gpu_idle_us_median": st.median(per_phase[name]["idle"]),
                "gpu_idle_us_mean": st.fmean(per_phase[name]["idle"]),
                "idle_pct": (
                    100.0 * st.median(per_phase[name]["idle"])
                    / st.median(per_phase[name]["span"])
                    if st.median(per_phase[name]["span"]) > 0
                    else 0.0
                ),
                "instrument_only": name in INSTRUMENT_PHASES,
            }
            for name in list(dict.fromkeys([p[2] for p in PHASES]))
            + ["inter_round_gap"]
            if per_phase[name]["span"]
        ],
        "gap_size_by_start_phase": {
            phase: {
                "count_total": len(values),
                "count_per_round": len(values) / n,
                "us_per_round": sum(values) / n,
                "median_us": st.median(values),
                "max_us": max(values),
            }
            for phase, values in sorted(
                by_phase_gap.items(), key=lambda kv: -sum(kv[1])
            )
        },
    }

    score_path = out / "score.json"
    if score_path.exists():
        metrics = json.loads(score_path.read_text()).get("metrics", {})
        for key in (
            "mtp_seconds_per_token",
            "serial_seconds_per_token",
            "effective_mean_draft_len",
            "accepted_draft_rate",
            "all_tokens_matched",
        ):
            if key in metrics:
                report[key] = metrics[key]

    print("%-16s %12s %12s %12s %8s" % ("phase", "span_us", "busy_us", "idle_us", "idle_%"))
    for row in report["phases"]:
        print(
            "%-16s %12.1f %12.1f %12.1f %7.2f%%%s"
            % (
                row["phase"],
                row["span_us_median"],
                row["gpu_busy_us_median"],
                row["gpu_idle_us_median"],
                row["idle_pct"],
                "  [instrument]" if row["instrument_only"] else "",
            )
        )
    print(
        "%-16s %12.1f %12.1f %12.1f %7.2f%%"
        % (
            "ROUND",
            report["round_us"]["median"],
            report["round_gpu_busy_us"]["median"],
            report["round_gpu_idle_us"]["median"],
            report["round_gpu_idle_pct_median"],
        )
    )
    print(
        "production-equivalent idle (instrument phases removed): %.1f us/round median"
        % report["round_gpu_idle_us_production"]["median"]
    )
    print("tiling error median %.3f us, max abs %.3f us (must be 0)"
          % (report["tiling_error_us_median"], report["tiling_error_us_max_abs"]))
    print("gaps >= %.1f us: %.1f per round, median %.1f us, p90 %.1f us, max %.1f us"
          % (args.min_gap_us, report["gaps_per_round"]["median"],
             report["gap_us"]["median"], report["gap_us"]["p90"],
             report["gap_us"]["max"]))
    print()
    print("%-16s %9s %9s %10s %10s" % ("start_phase", "gaps/rnd", "us/round", "median_us", "max_us"))
    for phase, stats in report["gap_size_by_start_phase"].items():
        print("%-16s %9.2f %9.1f %10.1f %10.1f"
              % (phase, stats["count_per_round"], stats["us_per_round"],
                 stats["median_us"], stats["max_us"]))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
