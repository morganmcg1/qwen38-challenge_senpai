#!/usr/bin/env python3
"""E210: the GPU busy/idle census of a whole decode leg.

    usage: research/e210_census.py TAG [--skip-rounds N] [--json OUT]

E185 measured GPU idle inside a drafting round and E204 measured it inside the
round-end seam. Neither placed the WHOLE leg on the device axis, so the leg's
idle fraction -- the number that decides whether any CPU-side overlap family
can still pay -- was never measured. This reader does that.

The leg is tiled with no gaps and no overlaps:

    seed        [t_begin0, t_begin_done)      the charged 512-token prefill
    prologue    [t_begin_done, t_round0#1)    parent protocol before round 1
    round r     [t_round0, t_tail_done)       the anchored round
    turnaround  [t_tail_done, next t_round0)  trace emission + parent protocol

Every window is intersected with the union of command-buffer GPU execution
intervals from the E90 ledger, which shares the mach uptime clock with the
anchors. Idle is the complement.

`row_dump`, `trace_tail` and `trace_emit` exist only because the leg is traced,
so their idle is reported separately and subtracted for the
production-equivalent figures.

harness=local. Observation only; never a timing or promotion statistic.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e185_gap_table import (  # noqa: E402
    INSTRUMENT_PHASES,
    PHASES,
    busy_in,
    gaps_in,
    read_intervals,
    union,
)


def parse_fields(line: str, prefix: str) -> dict:
    fields = {}
    for token in line[len(prefix):].split():
        key, _, value = token.partition("=")
        try:
            fields[key] = int(value)
        except ValueError:
            pass
    return fields


def read_trace(path: Path) -> tuple[list[dict], list[dict]]:
    begins, anchors = [], []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("mtp-begin-anchor: "):
            begins.append(parse_fields(line, "mtp-begin-anchor: "))
        elif line.startswith("mtp-anchor: "):
            anchors.append(parse_fields(line, "mtp-anchor: "))
    return begins, anchors


def summarise(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "total": sum(values),
        "mean": st.fmean(values),
        "median": st.median(values),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def window_report(merged, windows: list[tuple[int, int]]) -> dict:
    """Busy/idle of a set of windows, plus the per-window idle distribution."""
    span = busy = 0
    idles = []
    for t0, t1 in windows:
        if t1 <= t0:
            continue
        w_busy = busy_in(merged, t0, t1)
        span += t1 - t0
        busy += w_busy
        idles.append((t1 - t0 - w_busy) / 1000.0)
    return {
        "windows": len(windows),
        "span_us": span / 1000.0,
        "gpu_busy_us": busy / 1000.0,
        "gpu_idle_us": (span - busy) / 1000.0,
        "gpu_idle_pct_of_segment": (100.0 * (span - busy) / span) if span else 0.0,
        "idle_us_per_window": summarise(idles),
    }


def leg_census(merged, begin: dict, anchors: list[dict], skip: int,
               min_slice_us: float) -> dict:
    """One decode leg, tiled from its seed anchor to its last round anchor."""
    kept = anchors[skip:] if skip < len(anchors) else []
    if not kept:
        return {}

    leg_t0 = begin["t_begin0"]
    leg_t1 = anchors[-1]["t_tail_done"]
    leg = window_report(merged, [(leg_t0, leg_t1)])

    seed = window_report(merged, [(begin["t_begin0"], begin["t_begin_done"])])
    prologue = window_report(
        merged, [(begin["t_begin_done"], anchors[0]["t_round0"])])

    round_windows = [(a["t_round0"], a["t_tail_done"]) for a in kept]
    rounds = window_report(merged, round_windows)

    # The turnaround splits into the instrument's own trace emission and the
    # parent protocol that follows it. `t_prev_trace_done` is recorded by the
    # NEXT round, which is the only ordering that can carry it.
    emit_windows, gap_windows, turn_windows = [], [], []
    for a, b in zip(kept, kept[1:]):
        turn_windows.append((a["t_tail_done"], b["t_round0"]))
        emit_end = b.get("t_prev_trace_done", 0)
        if a["t_tail_done"] <= emit_end <= b["t_round0"]:
            emit_windows.append((a["t_tail_done"], emit_end))
            gap_windows.append((emit_end, b["t_round0"]))
        else:
            gap_windows.append((a["t_tail_done"], b["t_round0"]))
    turnaround = window_report(merged, turn_windows)
    trace_emit = window_report(merged, emit_windows)
    protocol_gap = window_report(merged, gap_windows)

    per_phase = {}
    instrument_idle = 0.0
    # The serial body defines no drafting anchors, so its compute window has no
    # name in the drafting phase list.
    for lo, hi, name in PHASES + [("t_round0", "t_eval_done", "serial_body")]:
        windows = []
        for a in kept:
            t0, t1 = a.get(lo, 0), a.get(hi, 0)
            if t0 and t1 and t1 > t0:
                windows.append((t0, t1))
        if not windows:
            continue
        per_phase[name] = window_report(merged, windows)
        if name in INSTRUMENT_PHASES:
            instrument_idle += per_phase[name]["gpu_idle_us"]
    instrument_idle += trace_emit["gpu_idle_us"]

    # Coherent idle slices across the whole censused span: an idle total made
    # of one 5 ms slice and one made of 5000 one-microsecond slices are not the
    # same opportunity.
    slices = []
    for t0, t1 in gaps_in(merged, kept[0]["t_round0"], leg_t1):
        slices.append((t1 - t0) / 1000.0)
    big = sorted((s for s in slices if s >= min_slice_us), reverse=True)

    censused_span = (leg_t1 - kept[0]["t_round0"]) / 1000.0
    censused_idle = window_report(merged, [(kept[0]["t_round0"], leg_t1)])

    return {
        "rounds_analysed": len(kept),
        "rounds_skipped": skip,
        "leg_span_us": leg["span_us"],
        "leg_gpu_busy_us": leg["gpu_busy_us"],
        "leg_gpu_idle_us": leg["gpu_idle_us"],
        "leg_gpu_idle_pct": leg["gpu_idle_pct_of_segment"],
        "leg_gpu_idle_pct_production": 100.0
        * (leg["gpu_idle_us"] - instrument_idle) / leg["span_us"],
        "instrument_idle_us": instrument_idle,
        "segments": {
            "seed": seed,
            "prologue": prologue,
            "rounds": rounds,
            "turnaround": turnaround,
            "turnaround_trace_emit": trace_emit,
            "turnaround_protocol_gap": protocol_gap,
        },
        "segment_share_of_leg_pct": {
            "seed": 100.0 * seed["span_us"] / leg["span_us"],
            "prologue": 100.0 * prologue["span_us"] / leg["span_us"],
            "rounds": 100.0 * rounds["span_us"] / leg["span_us"],
            "turnaround": 100.0 * turnaround["span_us"] / leg["span_us"],
        },
        "idle_share_of_leg_idle_pct": {
            "seed": 100.0 * seed["gpu_idle_us"] / leg["gpu_idle_us"],
            "prologue": 100.0 * prologue["gpu_idle_us"] / leg["gpu_idle_us"],
            "rounds": 100.0 * rounds["gpu_idle_us"] / leg["gpu_idle_us"],
            "turnaround": 100.0 * turnaround["gpu_idle_us"] / leg["gpu_idle_us"],
        },
        "phases": per_phase,
        "censused_span_us": censused_span,
        "censused_idle_us": censused_idle["gpu_idle_us"],
        "coherent_slices": {
            "min_slice_us": min_slice_us,
            "count": len(big),
            "count_per_round": len(big) / len(kept),
            "total_us": sum(big),
            "us_per_round": sum(big) / len(kept),
            "pct_of_censused_span": 100.0 * sum(big) / censused_span,
            "largest_us": big[:10],
        },
    }


def budget_crosscheck(leg_us: float, seed_us: float, round_us: list[float],
                      seed_eval_wall_us: float) -> dict:
    """FINDING 539's leg budget, recomputed both ways.

    E204's reader matched `wall_us` with a non-greedy prefix, so it captured
    `eval_wall_us` of the same `begin` line. The seed segment then lost its
    graph-build share and that share reappeared as an outside-anchor pool.
    """
    anchored = sum(round_us)
    return {
        "leg_us": leg_us,
        "anchored_us": anchored,
        "anchored_share_pct": 100.0 * anchored / leg_us,
        "seed_us": seed_us,
        "seed_share_pct": 100.0 * seed_us / leg_us,
        "outside_us": leg_us - anchored - seed_us,
        "outside_share_pct": 100.0 * (leg_us - anchored - seed_us) / leg_us,
        "e204_reader_seed_us": seed_eval_wall_us,
        "e204_reader_seed_share_pct": 100.0 * seed_eval_wall_us / leg_us,
        "e204_reader_outside_us": leg_us - anchored - seed_eval_wall_us,
        "e204_reader_outside_share_pct": 100.0
        * (leg_us - anchored - seed_eval_wall_us) / leg_us,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--skip-rounds", type=int, default=8)
    ap.add_argument("--min-slice-us", type=float, default=100.0)
    ap.add_argument("--json", dest="json_path")
    args = ap.parse_args()

    out = Path("research/out") / args.tag
    meta = dict(
        line.split("=", 1)
        for line in (out / "meta.txt").read_text().splitlines()
        if "=" in line
    )
    score = json.loads((out / "score.json").read_text())["metrics"]
    begins, anchors = read_trace(out / "trace.txt")
    if not begins or not anchors:
        print("e210_census: no leg anchors", file=sys.stderr)
        return 2

    pid = st.mode(a["pid"] for a in anchors)
    merged = union(read_intervals(out / "gpu-intervals.jsonl", pid))
    if not merged:
        print("e210_census: no gpu intervals for pid %d" % pid, file=sys.stderr)
        return 2

    # One trace file collects every worker the run spawns -- the reference pass
    # writes anchors too -- and `leg` counts begins per process, so both records
    # are scoped to the timed worker before anything is joined (RULE 386).
    anchors = [a for a in anchors if a["pid"] == pid]
    begins = [b for b in begins if b["pid"] == pid]

    decode_tokens = score["decode_tokens"]
    legs = {}
    for leg_id in sorted({a["leg"] for a in anchors if "leg" in a}):
        leg_anchors = [a for a in anchors if a.get("leg") == leg_id]
        begin = next((b for b in begins if b.get("leg") == leg_id), None)
        if not begin or not leg_anchors:
            continue
        serial = all(a.get("serial", 0) == 1 for a in leg_anchors)
        spt = (score["serial_seconds_per_token"] if serial
               else score["mtp_seconds_per_token"])
        report = leg_census(merged, begin, leg_anchors, args.skip_rounds,
                            args.min_slice_us)
        report["leg_kind"] = "serial-k1" if serial else "mtp"
        report["parent_leg_us"] = spt * decode_tokens * 1e6
        report["parent_seconds_per_token"] = spt
        report["leg_span_minus_parent_leg_us"] = (
            report["leg_span_us"] - report["parent_leg_us"])
        report["budget_crosscheck"] = budget_crosscheck(
            leg_us=report["parent_leg_us"],
            seed_us=(begin["t_begin_done"] - begin["t_begin0"]) / 1000.0,
            round_us=[(a["t_tail_done"] - a["t_round0"]) / 1000.0
                      for a in leg_anchors],
            seed_eval_wall_us=(
                begin["t_begin_done"] - begin["t_begin_built"]) / 1000.0,
        )
        legs[f"leg{leg_id}-{report['leg_kind']}"] = report

    report = {
        "tag": args.tag,
        "experiment": "e210-gpu-idle-census",
        "harness": "local",
        "official_or_ranked_score": False,
        "leg_role": "gpu-interval-ledger-observation",
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "memory_bytes": meta.get("memory_bytes"),
        "sandbox": meta.get("sandbox"),
        "tokens": meta.get("tokens"),
        "stall_us": meta.get("stall_us", "0"),
        "head_provenance_sha256": score.get("head_provenance_sha256"),
        "all_tokens_matched": score.get("all_tokens_matched"),
        "effective_mean_draft_len": score.get("effective_mean_draft_len"),
        "accepted_draft_rate": score.get("accepted_draft_rate"),
        "mtp_seconds_per_token": score.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": score.get("serial_seconds_per_token"),
        "pid": pid,
        "union_intervals": len(merged),
        "legs": legs,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
