#!/usr/bin/env python3
"""Research-only: turn forced-depth phase traces into a per-depth cost curve.

Reads research/out/<arm>/trace.txt.<pid> files written by the
MLX_QWEN_MTP_TRACE=1 instrumentation and reports, for each observed per-round
draft count d:

    C(d)         mean round_us over steady full-accept rounds at depth d
    marginal(d)  C(d) - C(d-1)
    h(d)         marginal(d) / C(0)  -- per-step head cost in serial-round units

C(0) is free in every arm: benchmark-qwen-mtp.sh always runs a
`mtp-timed --mtp-depth 0` serial control through the same session, so a
depth-0 reference is measured on the same host at the same temperature as the
drafting leg it is compared against. `headStepCostRatio` in the cost model is
expressed in exactly these units, so h(d) drops straight into a per-depth
vector.
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=(-?\d+)")
BEGIN_RE = re.compile(r"^mtp-trace: begin seed=(\d+) build_us=(\d+) eval_wall_us=(\d+)$")

PHASES = [
    "draft_build_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]


def parse_trace(path):
    begin, rounds = None, []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        m = BEGIN_RE.match(line)
        if m:
            begin = {"seed": int(m.group(1)), "build_us": int(m.group(2)),
                     "eval_wall_us": int(m.group(3))}
            continue
        m = ROUND_RE.match(line)
        if m:
            row = {"round": int(m.group(1)), "d": int(m.group(2)),
                   "acc": int(m.group(3))}
            row.update({k: int(v) for k, v in KV_RE.findall(m.group(4))})
            rounds.append(row)
    return begin, rounds


def summarize(rows, key="round_us"):
    vals = [r[key] for r in rows]
    if not vals:
        return None
    return {"n": len(vals), "mean_us": statistics.fmean(vals),
            "median_us": statistics.median(vals),
            "stddev_us": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min_us": min(vals), "max_us": max(vals)}


def load_legs(arm_dir, warmup):
    """One entry per worker process that emitted decode rounds.

    `warmup` leading decode rounds are dropped per leg: the seed prologue is
    already its own `begin` line, but the first decode rounds still pay
    first-touch costs (lazy recurrent roots installed by the prologue, first
    wide SDPA shape, allocator growth), so they are not steady state.

    A round with acc < d rejected a draft, which changes the work in that same
    round (rollback / repair). Only acc == d rounds measure the clean cost of
    proposing and verifying d drafts.
    """
    legs = []
    for path in sorted(arm_dir.glob("trace.txt*")):
        begin, rounds = parse_trace(path)
        if not rounds:
            continue
        tail = [r for r in rounds if r["round"] > warmup]
        steady = [r for r in tail if r["acc"] == r["d"]]
        legs.append({
            "trace": path.name, "begin": begin,
            "rounds_total": len(rounds),
            "dropped_warmup": len(rounds) - len(tail),
            "dropped_partial": len(tail) - len(steady),
            "depths_seen": sorted({r["d"] for r in rounds}),
            "acc_mean": statistics.fmean([r["acc"] for r in rounds]),
            "steady_rows": steady,
        })
    return legs


def read_meta(arm_dir):
    meta = {}
    path = arm_dir / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm dir names to include (default: every dir)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    arm_dirs = ([args.out_dir / a for a in args.arms] if args.arms
                else sorted(p for p in args.out_dir.iterdir() if p.is_dir()))

    by_depth = defaultdict(list)
    arms = {}
    print(f"{'arm':<14} {'trace':<22} {'N':>4} {'depths':<10} {'acc':>5} "
          f"{'drop_w':>6} {'drop_p':>6}")
    for arm_dir in arm_dirs:
        if not arm_dir.is_dir():
            print(f"skip missing {arm_dir}", file=sys.stderr)
            continue
        legs = load_legs(arm_dir, args.warmup)
        score_path = arm_dir / "score.json"
        arms[arm_dir.name] = {
            "meta": read_meta(arm_dir),
            "score": json.loads(score_path.read_text()) if score_path.exists() else None,
            "legs": [{k: v for k, v in leg.items() if k != "steady_rows"}
                     for leg in legs],
        }
        for leg in legs:
            print(f"{arm_dir.name:<14} {leg['trace']:<22} {leg['rounds_total']:>4} "
                  f"{str(leg['depths_seen']):<10} {leg['acc_mean']:>5.2f} "
                  f"{leg['dropped_warmup']:>6} {leg['dropped_partial']:>6}")
            for row in leg["steady_rows"]:
                by_depth[row["d"]].append(row)

    c0_rows = by_depth.get(0, [])
    c0 = summarize(c0_rows)["mean_us"] if c0_rows else None
    if c0 is None:
        print("\nwarning: no depth-0 rounds; h(d) cannot be normalised",
              file=sys.stderr)

    print(f"\n{'d':>2} {'N':>4} {'C(d) us':>10} {'median':>9} {'sd':>8} "
          f"{'marg':>9} {'h(d)':>7} {'C/C0':>7} {'us/tok':>8} {'eval':>8} {'host':>8}")
    curve, prev = {}, None
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        s = summarize(rows)
        marginal = None if prev is None else s["mean_us"] - prev
        h = None if (marginal is None or not c0) else marginal / c0
        phases = {p: summarize(rows, p) for p in PHASES}
        host = sum(phases[p]["mean_us"] for p in PHASES if p != "eval_wall_us")
        ratio = s["mean_us"] / c0 if c0 else None
        curve[depth] = {"depth": depth, "steady": s, "marginal_us": marginal,
                        "h": h, "c_over_c0": ratio,
                        "us_per_token": s["mean_us"] / (depth + 1),
                        "phases": phases, "host_us": host}
        marg_s = "-" if marginal is None else "%.1f" % marginal
        h_s = "-" if h is None else "%.4f" % h
        ratio_s = "-" if ratio is None else "%.3f" % ratio
        print(f"{depth:>2} {s['n']:>4} {s['mean_us']:>10.1f} {s['median_us']:>9.1f} "
              f"{s['stddev_us']:>8.1f} {marg_s:>9} {h_s:>7} {ratio_s:>7} "
              f"{s['mean_us'] / (depth + 1):>8.1f} "
              f"{phases['eval_wall_us']['mean_us']:>8.1f} {host:>8.1f}")
        prev = s["mean_us"]

    if args.json:
        args.json.write_text(json.dumps(
            {"c0_us": c0, "warmup": args.warmup, "arms": arms,
             "curve": {str(k): v for k, v in curve.items()}},
            indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
