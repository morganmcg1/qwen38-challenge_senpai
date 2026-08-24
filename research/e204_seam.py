#!/usr/bin/env python3
"""E204 stage 0: the round-end seam on the composed tree, by acceptance outcome.

    usage: research/e204_seam.py TAG [--skip-rounds N] [--json OUT]

E185 measured the round-end seam as four contiguous phases (commit 455.8 +
inter_round_gap 451.6 + readout 44.8 + upkeep_pre 7.1 = 959.3 us/round) on a
base that did NOT carry the E165/E193 head prefetch. The prefetch now submits
the next round's FIRST head step inside the commit phase, so part of that
window is already covered by GPU work and the E185 numbers cannot be reused.

This reader answers three questions the phase medians alone cannot:

1. **How much of the round-end window is still GPU idle**, per phase and in
   total, split into full-acceptance rounds (acc == d) and rejection rounds
   (acc < d), which run different bookkeeping.
2. **The largest COHERENT idle slice.** The window is walked as one interval
   from `t_eval_done` of round i to `t_round0` of round i+1, crossing the round
   boundary, so a slice that spans several phases is measured as one slice
   rather than split by an anchor that no mechanism respects.
3. **Where the coherent slice starts**, so the report names the host work that
   would have to be reordered to cover it.

The seam is reported ONCE, whole. FINDING 528 closed the E192 removal question:
the 397 us `clearRecurrentRollback` release cost is relocation, not removable
work, so no other mechanism will take that slice out of the commit phase and
there is nothing to double-count.

harness=local. Observation only, no timed contrast.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

# Consecutive anchor pairs tiling the round (E185's PHASES, unchanged).
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
# The round-end seam: everything after the round's single blocking eval, plus
# the protocol turnaround before the next round starts.
SEAM_PHASES = ["readout", "commit", "upkeep_pre", "inter_round_gap"]


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
        if not line.startswith("{"):
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


def phase_of(anchors: dict, next_round0: int, t: int) -> str:
    for lo, hi, name in PHASES:
        if anchors.get(lo, 0) <= t < anchors.get(hi, 0):
            return name
    if anchors.get("t_tail_done", 0) <= t < next_round0:
        return "inter_round_gap"
    return "outside"


def summarise(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": st.median(values),
        "mean": st.fmean(values),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": max(values),
        "total": sum(values),
    }


def seam_report(rounds: list[dict], merged: list[tuple[int, int]],
                pairs: list[tuple[dict, dict]], label: str,
                round_us_median: float) -> dict:
    """Decompose the round-end seam over the given (round, next round) pairs."""
    per_phase: dict[str, list[float]] = {name: [] for name in SEAM_PHASES}
    per_phase_span: dict[str, list[float]] = {name: [] for name in SEAM_PHASES}
    seam_span, seam_idle, seam_idle_prod = [], [], []
    coherent_max, coherent_start = [], []
    slices: list[dict] = []

    for r, nxt in pairs:
        # Production-equivalent seam: the two instrument phases (row_dump,
        # trace_tail) exist only because the leg is traced, so their span is
        # removed from the production figure rather than assumed small.
        instrument_span = (r.get("t_row_trace_done", 0) - r.get("t_row_trace0", 0))
        t0, t1 = r["t_eval_done"], nxt["t_round0"]
        if t1 <= t0:
            continue
        span = t1 - t0
        busy = busy_in(merged, t0, t1)
        idle = span - busy
        seam_span.append(span / 1000.0)
        seam_idle.append(idle / 1000.0)
        instrument_idle = (
            instrument_span - busy_in(merged, r.get("t_row_trace0", 0),
                                      r.get("t_row_trace_done", 0)))
        seam_idle_prod.append((idle - instrument_idle) / 1000.0)

        for lo, hi, name in PHASES:
            if name not in per_phase:
                continue
            a, b = r.get(lo, 0), r.get(hi, 0)
            if b > a:
                per_phase[name].append((b - a - busy_in(merged, a, b)) / 1000.0)
                per_phase_span[name].append((b - a) / 1000.0)
        a, b = r["t_tail_done"], nxt["t_round0"]
        if b > a:
            per_phase["inter_round_gap"].append(
                (b - a - busy_in(merged, a, b)) / 1000.0)
            per_phase_span["inter_round_gap"].append((b - a) / 1000.0)

        # Coherent slices: maximal contiguous idle intervals inside the seam,
        # crossing every anchor boundary including the round boundary.
        window_gaps = gaps_in(merged, t0, t1)
        best_us, best_phase = 0.0, "none"
        for ga, gb in window_gaps:
            us = (gb - ga) / 1000.0
            start_phase = phase_of(r, nxt["t_round0"], ga)
            end_phase = phase_of(r, nxt["t_round0"], gb - 1)
            slices.append({
                "round": r.get("round"), "us": us,
                "start_phase": start_phase, "end_phase": end_phase,
            })
            if us > best_us:
                best_us, best_phase = us, start_phase
        coherent_max.append(best_us)
        coherent_start.append(best_phase)

    coherent_median = st.median(coherent_max) if coherent_max else 0.0
    start_hist: dict[str, int] = {}
    for phase in coherent_start:
        start_hist[phase] = start_hist.get(phase, 0) + 1
    return {
        "label": label,
        "rounds": len(coherent_max),
        "seam_span_us": summarise(seam_span),
        "seam_idle_us": summarise(seam_idle),
        "seam_idle_us_production": summarise(seam_idle_prod),
        "seam_idle_pct_of_round": (
            100.0 * st.median(seam_idle_prod) / round_us_median
            if seam_idle_prod and round_us_median else 0.0),
        "phase_idle_us_median": {
            name: (st.median(values) if values else 0.0)
            for name, values in per_phase.items()
        },
        "phase_span_us_median": {
            name: (st.median(values) if values else 0.0)
            for name, values in per_phase_span.items()
        },
        "largest_coherent_slice_us": summarise(coherent_max),
        "largest_coherent_slice_pct_of_round": (
            100.0 * coherent_median / round_us_median if round_us_median else 0.0),
        "largest_coherent_slice_start_phase": start_hist,
        "coherent_slice_us": summarise([s["us"] for s in slices]),
        "slices_per_round": (len(slices) / len(coherent_max)) if coherent_max else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--skip-rounds", type=int, default=8)
    ap.add_argument("--json", dest="json_path")
    args = ap.parse_args()

    out = Path("research/out") / args.tag
    meta = dict(
        line.split("=", 1)
        for line in (out / "meta.txt").read_text().splitlines()
        if "=" in line
    )
    rounds = read_anchors(out / "trace.txt")
    if not rounds:
        print("e204_seam: no anchors", file=sys.stderr)
        return 2
    pid = st.mode(r["pid"] for r in rounds)
    merged = union(read_intervals(out / "gpu-intervals.jsonl", pid))
    if not merged:
        print("e204_seam: no gpu intervals for pid %d" % pid, file=sys.stderr)
        return 2
    rounds = [r for r in rounds if r["pid"] == pid][args.skip_rounds :]

    round_us = [
        (r["t_tail_done"] - r["t_round0"]) / 1000.0 for r in rounds
    ]
    round_us_median = st.median(round_us)

    pairs_all = [(rounds[i], rounds[i + 1]) for i in range(len(rounds) - 1)]
    pairs_full = [(a, b) for a, b in pairs_all if a.get("acc") == a.get("d")]
    pairs_reject = [(a, b) for a, b in pairs_all if a.get("acc") != a.get("d")]

    report = {
        "tag": args.tag,
        "experiment": "e204-round-end-seam-overlap",
        "stage": 0,
        "harness": "local",
        "official_or_ranked_score": False,
        "leg_role": "gpu-interval-ledger-observation",
        "timed_contrast": False,
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "tokens": meta.get("tokens"),
        "head_dir": meta.get("head_dir"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "pid": pid,
        "rounds_analysed": len(rounds),
        "rounds_skipped": args.skip_rounds,
        "round_us": summarise(round_us),
        "depth_histogram": {},
        "accept_histogram": {},
        "seam": {
            "all": seam_report(rounds, merged, pairs_all, "all", round_us_median),
            "full_acceptance": seam_report(
                rounds, merged, pairs_full, "full_acceptance", round_us_median),
            "rejection": seam_report(
                rounds, merged, pairs_reject, "rejection", round_us_median),
        },
    }
    for r in rounds:
        key = str(r.get("d"))
        report["depth_histogram"][key] = report["depth_histogram"].get(key, 0) + 1
        key = "%s/%s" % (r.get("acc"), r.get("d"))
        report["accept_histogram"][key] = report["accept_histogram"].get(key, 0) + 1

    score_path = out / "score.json"
    if score_path.exists():
        metrics = json.loads(score_path.read_text()).get("metrics", {})
        for key in (
            "mtp_seconds_per_token", "serial_seconds_per_token",
            "effective_mean_draft_len", "accepted_draft_rate",
            "all_tokens_matched",
        ):
            if key in metrics:
                report[key] = metrics[key]

    print("round median %.3f ms over %d rounds (%s, %s)"
          % (round_us_median / 1000.0, len(rounds), report["chip"], report["tokens"]))
    print("accept histogram acc/d: %s" % report["accept_histogram"])
    print()
    for key in ("all", "full_acceptance", "rejection"):
        block = report["seam"][key]
        if not block["rounds"]:
            continue
        print("== %s (%d rounds) ==" % (key, block["rounds"]))
        print("%-18s %12s %12s" % ("phase", "span_us", "idle_us"))
        for name in SEAM_PHASES:
            print("%-18s %12.1f %12.1f"
                  % (name, block["phase_span_us_median"][name],
                     block["phase_idle_us_median"][name]))
        print("%-18s %12.1f %12.1f  (%.3f%% of round)"
              % ("SEAM total", block["seam_span_us"]["median"],
                 block["seam_idle_us_production"]["median"],
                 block["seam_idle_pct_of_round"]))
        print("largest coherent idle slice: median %.1f us (%.3f%% of round), "
              "p90 %.1f, max %.1f"
              % (block["largest_coherent_slice_us"]["median"],
                 block["largest_coherent_slice_pct_of_round"],
                 block["largest_coherent_slice_us"]["p90"],
                 block["largest_coherent_slice_us"]["max"]))
        print("slice start phase histogram: %s"
              % block["largest_coherent_slice_start_phase"])
        print()

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
