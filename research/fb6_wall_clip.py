#!/usr/bin/env python3
"""FB6 deliverables from an existing MLX_QWEN_MTP_TRACE run.

1. Chosen-depth histogram with and without `sdpaWidthWallDepthCap`, and the
   rate at which the greedy walk in `costModelDepth` wanted to extend past the
   wall and was clipped.
2. Per-width top-2 logit deviation of the MTP rows against the serial
   trajectory, with top-2 ordering and identity preservation.

The depth replay is a counterfactual on the RECORDED (streak, EMA) trajectory,
not a re-simulation: opening the wall would change acceptance and therefore the
trajectory itself. It answers "how often did the shipped walk want more depth
than the wall allowed", which is the clip rate the advisor asked for.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?round_us=(\d+) "
    r"streak_in=(-?\d+) cap=(\d+) ema_in=([0-9.,]+)"
)
ROW_RE = re.compile(r"mtp-row: pos=(\d+) ids=(\d+),(\d+) v=(\S+),(\S+)")
PHASE_RE = re.compile(
    r"benchmark-qwen-mtp\.sh: (generating the MTP reference rows|"
    r"measuring the TRUE serial control|measuring native-MTP decode)"
)
PHASE_KEY = {
    "generating the MTP reference rows": "reference",
    "measuring the TRUE serial control": "serial",
    "measuring native-MTP decode": "mtp",
}

SDPA_WALL = 4
HEAD_STEP_COST_RATIO = 0.20
MAX_DEPTH = 8


def parse(path):
    phases = {"reference": {}, "serial": {}, "mtp": {}}
    rounds = []
    phase = None
    pending_rows = []
    with open(path) as handle:
        for line in handle:
            marker = PHASE_RE.search(line)
            if marker:
                phase = PHASE_KEY[marker.group(1)]
                pending_rows = []
                continue
            row = ROW_RE.search(line)
            if row and phase:
                pos = int(row.group(1))
                rec = (
                    (int(row.group(2)), int(row.group(3))),
                    (float.fromhex(row.group(4)), float.fromhex(row.group(5))),
                )
                phases[phase][pos] = rec
                if phase == "mtp":
                    pending_rows.append((pos, rec))
                continue
            rnd = ROUND_RE.search(line)
            if rnd and phase == "mtp":
                rounds.append(
                    {
                        "round": int(rnd.group(1)),
                        "d": int(rnd.group(2)),
                        "acc": int(rnd.group(3)),
                        "round_us": int(rnd.group(4)),
                        "streak_in": int(rnd.group(5)),
                        "cap": int(rnd.group(6)),
                        "ema_in": [float(x) for x in rnd.group(7).split(",")],
                        "rows": pending_rows,
                    }
                )
                pending_rows = []
    return phases, rounds


def walk(ema, width_cap, offered, conf):
    """Exact transcription of costModelDepth's greedy marginal rule."""
    cap = min(min(offered, MAX_DEPTH), width_cap)
    if cap <= 0:
        return 0
    h = HEAD_STEP_COST_RATIO
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        p = ema[depth]
        if depth == 0 and conf is not None:
            p = min(p, conf)
        reach *= p
        threshold = h * (1.0 + expected) / (1.0 + depth * h)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def conf_of(rec):
    margin = rec[1][0] - rec[1][1]
    return 1.0 / (1.0 + math.exp(-margin / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--label", required=True)
    ap.add_argument("--verify-cap", type=int, required=True)
    ap.add_argument("--streak-gate", type=int, default=3)
    ap.add_argument("--offered", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    phases, rounds = parse(args.trace)
    if not rounds:
        sys.exit(f"no traced rounds in {args.trace}")

    # pendingTop2 entering round n is the last row emitted by round n-1.
    confs = [None]
    for prev in rounds[:-1]:
        confs.append(conf_of(prev["rows"][-1][1]) if prev["rows"] else None)

    replay_ok = 0
    per_round = []
    hist_shipped, hist_open = Counter(), Counter()
    clipped = 0
    wanted_deeper_total = 0
    for rnd, conf in zip(rounds, confs):
        ema = rnd["ema_in"]
        shipped_cap = (
            args.verify_cap if rnd["streak_in"] >= args.streak_gate else SDPA_WALL
        )
        d_shipped = walk(ema, shipped_cap, args.offered, conf)
        d_open = walk(ema, args.verify_cap, args.offered, conf)
        d_unbounded = walk(ema, MAX_DEPTH, args.offered, conf)
        ok = d_shipped == rnd["d"]
        replay_ok += ok
        wall_bound = shipped_cap == SDPA_WALL and d_open > d_shipped
        clipped += wall_bound
        wanted_deeper_total += max(0, d_open - d_shipped)
        hist_shipped[rnd["d"]] += 1
        hist_open[d_open] += 1
        per_round.append(
            {
                "round": rnd["round"],
                "observed_d": rnd["d"],
                "replay_d": d_shipped,
                "replay_matches": ok,
                "streak_in": rnd["streak_in"],
                "shipped_width_cap": shipped_cap,
                "d_wall_open": d_open,
                "d_unbounded": d_unbounded,
                "clipped_by_wall": wall_bound,
                "conf_in": conf,
            }
        )

    n = len(rounds)
    # Per-width top-2 deviation of MTP rows against the serial trajectory.
    by_width = {}
    for rnd in rounds:
        width = rnd["d"] + 1
        bucket = by_width.setdefault(
            width,
            {
                "rows": 0,
                "max_abs_dev_top1": 0.0,
                "max_abs_dev_top2": 0.0,
                "max_rel_dev_top1": 0.0,
                "max_rel_dev_top2": 0.0,
                "top1_id_mismatches": 0,
                "top2_id_mismatches": 0,
                "ordering_swaps": 0,
                "unmatched_positions": 0,
            },
        )
        for pos, rec in rnd["rows"]:
            ref = phases["serial"].get(pos)
            if ref is None:
                bucket["unmatched_positions"] += 1
                continue
            bucket["rows"] += 1
            for k, tag in ((0, "top1"), (1, "top2")):
                dev = abs(rec[1][k] - ref[1][k])
                bucket[f"max_abs_dev_{tag}"] = max(
                    bucket[f"max_abs_dev_{tag}"], dev
                )
                denom = abs(ref[1][k])
                if denom > 0:
                    bucket[f"max_rel_dev_{tag}"] = max(
                        bucket[f"max_rel_dev_{tag}"], dev / denom
                    )
            if rec[0][0] != ref[0][0]:
                bucket["top1_id_mismatches"] += 1
            if rec[0][1] != ref[0][1]:
                bucket["top2_id_mismatches"] += 1
            if set(rec[0]) == set(ref[0]) and rec[0] != ref[0]:
                bucket["ordering_swaps"] += 1

    report = {
        "label": args.label,
        "trace": args.trace,
        "verify_depth_cap": args.verify_cap,
        "streak_gate": args.streak_gate,
        "sdpa_width_wall_depth_cap": SDPA_WALL,
        "head_step_cost_ratio": HEAD_STEP_COST_RATIO,
        "rounds": n,
        "replay_matches_observed": replay_ok,
        "replay_match_rate": replay_ok / n,
        "clip_rate_rounds": clipped / n,
        "clipped_rounds": clipped,
        "clipped_depth_total": wanted_deeper_total,
        "mean_depth_shipped": sum(r["d"] for r in rounds) / n,
        "mean_depth_wall_open": sum(p["d_wall_open"] for p in per_round) / n,
        "depth_histogram_shipped": dict(sorted(hist_shipped.items())),
        "depth_histogram_wall_open": dict(sorted(hist_open.items())),
        "serial_rows_available": len(phases["serial"]),
        "reference_rows_available": len(phases["reference"]),
        "per_width_top2_deviation": {
            str(w): by_width[w] for w in sorted(by_width)
        },
        "per_round": per_round,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    summary = dict(report)
    summary.pop("per_round")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
