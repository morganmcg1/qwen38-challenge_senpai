#!/usr/bin/env python3
"""E165 Stage 0: decompose one MTP round and find the GPU-idle window.

usage: research/e165_census.py TAG [TAG ...] [--json PATH]

Every number comes from the `mtp-anchor:` lines, which carry absolute mach
uptime nanoseconds at fourteen boundaries of the round. The round period is
therefore reconstructed ACROSS the round boundary, which is where the window
this experiment targets actually lives: the trusted parent owns the protocol
turnaround between `t_tail_done` of round N and `t_round0` of round N+1, and
the GPU is idle for all of it.

THE GPU-IDLE WINDOW.

The round's single blocking `eval` returns at `t_eval_done` with every
submitted command buffer complete, so the device owns no work from that
instant. The next submission is the head chain's first `asyncEval`, issued
immediately after `t_head1_built`. The window between them is

    idle_raw = t_head1_built(N+1) - t_eval_done(N)

and it contains, in order: the host readout, the accept walk and commit, the
head-history upkeep, the trace instrument, the protocol turnaround, and the
next round's pre-draft and head-1 graph build.

INSTRUMENT COST IS SUBTRACTED, NOT ASSUMED SMALL. The row dump and the two
trace writes sit inside that window and are worth tens of microseconds against
a target of hundreds, so both are bracketed by their own anchors and removed:

    idle = idle_raw - row_trace(N) - trace_emit(N)

`row_trace` is bracketed in-round; `trace_emit` is measured after the write and
reported in the NEXT round's line as `t_prev_trace_done`, which is the only
ordering that can carry it. A trace without those fields reports idle_raw and
says so.

The reject path submits a recurrent-state prefetch inside `commit`, so its
idle window is shorter than the anchors show. Accept and reject rounds are
reported separately and the fit uses full-accept rounds only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"


def read_meta(tag: str) -> dict:
    path = OUT / tag / "meta.txt"
    if not path.exists():
        return {}
    return dict(
        line.partition("=")[::2]
        for line in path.read_text().splitlines()
        if "=" in line
    )


def read_score(tag: str) -> dict:
    path = OUT / tag / "score.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def anchors(tag: str) -> list[dict]:
    """Every `mtp-anchor` round of one leg, in file order, per pid."""
    path = OUT / tag / "trace.txt"
    if not path.exists():
        return []
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-anchor: "):
            continue
        fields = {}
        ok = True
        for token in line[len("mtp-anchor: "):].split():
            key, sep, value = token.partition("=")
            if not sep:
                ok = False
                break
            try:
                fields[key] = int(value)
            except ValueError:
                ok = False
                break
        if ok and "t_round0" in fields:
            rounds.append(fields)
    return rounds


def components(rounds: list[dict]) -> list[dict]:
    """Per-round component microseconds, including the cross-boundary window.

    The last round of a leg has no successor and therefore no idle window, so
    it contributes every in-round component and no window.
    """
    out = []
    for index, cur in enumerate(rounds):
        nxt = rounds[index + 1] if index + 1 < len(rounds) else None
        if nxt is not None and nxt["pid"] != cur["pid"]:
            nxt = None
        if nxt is not None and nxt["t_round0"] < cur["t_tail_done"]:
            nxt = None

        us = lambda a, b: (b - a) / 1000.0  # noqa: E731

        row_trace = 0.0
        if cur.get("t_row_trace_done", 0) and cur.get("t_row_trace0", 0):
            row_trace = us(cur["t_row_trace0"], cur["t_row_trace_done"])

        record = {
            "round": cur["round"],
            "d": cur["d"],
            "acc": cur["acc"],
            "full_accept": cur["acc"] == cur["d"],
            # Host graph build for the head chain, both submits included.
            "host_draft_build": us(cur["t_round0"], cur["t_draft_built"]),
            "host_draft_pre": us(cur["t_round0"], cur["t_flush_built"]),
            "host_head1_build": us(cur["t_flush_built"], cur["t_head1_built"]),
            "submit1": us(cur["t_head1_built"], cur["t_submit1"]),
            "host_chain_build": us(cur["t_submit1"], cur["t_chain_built"]),
            "submit2": us(cur["t_chain_built"], cur["t_draft_built"]),
            "snapshot": us(cur["t_draft_built"], cur["t_snapshot_done"]),
            # ~97 % GPU wait under the shipped ladder (E86); never read as
            # host op-count evidence.
            "verify_window": us(cur["t_snapshot_done"], cur["t_verify_built"]),
            "gpu_verify_eval": us(cur["t_verify_built"], cur["t_eval_done"]),
            "readout": us(cur["t_eval_done"], cur["t_read_done"]),
            "commit": us(cur["t_read_done"], cur["t_commit_done"]),
            "upkeep": us(cur["t_commit_done"], cur["t_tail_done"]),
            "row_trace": row_trace,
            "round_total": us(cur["t_round0"], cur["t_tail_done"]),
        }
        record["host_post_eval_tail"] = (
            record["readout"] + record["commit"] + record["upkeep"] - row_trace
        )
        if nxt is not None:
            trace_emit = 0.0
            if nxt.get("t_prev_trace_done", 0):
                trace_emit = us(cur["t_tail_done"], nxt["t_prev_trace_done"])
            gap = us(cur["t_tail_done"], nxt["t_round0"]) - trace_emit
            idle_raw = us(cur["t_eval_done"], nxt["t_head1_built"])
            record["trace_emit"] = trace_emit
            record["protocol_gap"] = gap
            record["next_pre_submit"] = (
                us(nxt["t_round0"], nxt["t_head1_built"])
            )
            record["gpu_idle_window_raw"] = idle_raw
            record["gpu_idle_window"] = idle_raw - row_trace - trace_emit
            record["round_period"] = (
                us(cur["t_round0"], nxt["t_round0"]) - trace_emit - row_trace
            )
        out.append(record)
    return out


FIT_KEYS = [
    "gpu_idle_window",
    "host_post_eval_tail",
    "readout",
    "commit",
    "upkeep",
    "protocol_gap",
    "next_pre_submit",
    "host_draft_build",
    "host_chain_build",
    "submit1",
    "submit2",
    "snapshot",
    "verify_window",
    "gpu_verify_eval",
    "round_period",
    "round_total",
]


def median_of(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if key in r]
    return st.median(values) if values else None


def fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares intercept and slope of y against d."""
    n = len(points)
    if n < 2:
        return (points[0][1], 0.0) if n else (float("nan"), float("nan"))
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return (sy / n, 0.0)
    slope = (n * sxy - sx * sy) / denom
    return ((sy - slope * sx) / n, slope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    ap.add_argument(
        "--accept-only",
        action="store_true",
        help="fit on full-accept rounds only (the reject path submits a "
             "recurrent prefetch inside the idle window)",
    )
    args = ap.parse_args()

    report = {"legs": [], "fit": {}}
    per_depth: dict[int, dict[str, float]] = {}

    for tag in args.tags:
        meta = read_meta(tag)
        score = read_score(tag)
        recs = components(anchors(tag))
        if not recs:
            print(f"{tag}: no anchor rounds")
            continue
        use = [r for r in recs if r["full_accept"]] if args.accept_only else recs
        depths = sorted({r["d"] for r in recs})
        leg = {
            "tag": tag,
            "rounds": len(recs),
            "rounds_used": len(use),
            "depths": depths,
            "sync_head": meta.get("sync_head"),
            "trace": meta.get("trace"),
            "cool_gate": meta.get("cool_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "worker_sha256": meta.get("worker_sha256"),
            "base_sha": meta.get("base_sha"),
            "full_accept_rounds": sum(1 for r in recs if r["full_accept"]),
            "mtp_seconds_per_token": score.get(
                "candidate_mtp_seconds_per_token_mean"),
            "effective_mean_draft_len": score.get("effective_mean_draft_len"),
            "accepted_draft_rate": score.get("accepted_draft_rate"),
            "median": {k: median_of(use, k) for k in FIT_KEYS},
        }
        report["legs"].append(leg)

        print(f"\n=== {tag}  rounds={len(recs)} used={len(use)} "
              f"depths={depths} sync_head={meta.get('sync_head')} "
              f"gate={meta.get('cool_gate')} ===")
        print(f"  s/tok={leg['mtp_seconds_per_token']} "
              f"edl={leg['effective_mean_draft_len']} "
              f"full_accept={leg['full_accept_rounds']}/{len(recs)}")
        for key in FIT_KEYS:
            value = leg["median"][key]
            if value is not None:
                print(f"  {key:24s} {value:12.1f} us")

        if len(depths) == 1:
            per_depth.setdefault(depths[0], {})
            if meta.get("sync_head") != "1" and meta.get("trace") != "0":
                per_depth[depths[0]] = leg["median"]

    fit_depths = sorted(d for d, m in per_depth.items() if m)
    if len(fit_depths) >= 2:
        print(f"\n=== component fit against pinned depth d in {fit_depths} ===")
        print(f"  {'component':24s} {'intercept us':>14s} {'per-draft us':>14s}")
        for key in FIT_KEYS:
            points = [
                (float(d), per_depth[d][key])
                for d in fit_depths
                if per_depth[d].get(key) is not None
            ]
            if len(points) < 2:
                continue
            intercept, slope = fit(points)
            report["fit"][key] = {
                "intercept_us": intercept,
                "per_draft_us": slope,
                "points": points,
            }
            print(f"  {key:24s} {intercept:14.1f} {slope:14.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
