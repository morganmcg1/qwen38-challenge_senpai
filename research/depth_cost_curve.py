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
BEGIN_RE = re.compile(
    r"^mtp-trace: begin seed=(\d+) build_us=(\d+) eval_wall_us=(\d+)"
    r"(?: h=(\S+))?$")

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
                     "eval_wall_us": int(m.group(3)), "h": m.group(4)}
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
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"n": len(vals), "mean_us": mean,
            "median_us": statistics.median(vals),
            "stddev_us": sd, "sd_pct": 100.0 * sd / mean if mean else 0.0,
            "sem_us": sd / len(vals) ** 0.5,
            "min_us": min(vals), "max_us": max(vals)}


def trace_pid(path):
    """Numeric PID suffix of a `trace.txt.<pid>` file.

    Sorting on this rather than the filename keeps leg order equal to process
    start order once a PID crosses a digit boundary.
    """
    return int(str(path).rsplit(".", 1)[-1])


def select_mtp_leg(legs, meta, arm):
    """The one leg whose rounds came from the candidate MTP session.

    Each arm starts two model-holding workers: a serial reference leg, whose
    rounds are all `d=0`, and the MTP leg. Reading the wrong file silently
    turns a real MTP arm into a serial control, so name the leg explicitly and
    fail loudly rather than guessing.

    Exactly one leg with a nonzero depth is the normal case. Zero such legs is
    only legal when the arm forced depth 0, and both legs are then genuinely
    indistinguishable by depth alone; the MTP leg is the later-started process.
    Two nonzero-depth legs mean the arm layout changed and every downstream
    per-arm number would be a pool of two schedules.
    """
    scoring = [leg for leg in legs if leg["rounds_total"]]
    nonzero = [leg for leg in scoring if any(leg["depths_seen"])]
    if len(nonzero) == 1:
        return nonzero[0]
    if len(nonzero) > 1:
        raise SystemExit(
            f"{arm}: {len(nonzero)} legs carry nonzero depths "
            f"({[leg['trace'] for leg in nonzero]}); cannot name one MTP leg")
    if meta.get("force_depth") != "0":
        raise SystemExit(
            f"{arm}: no leg carries a nonzero depth but force_depth="
            f"{meta.get('force_depth')!r}; the MTP leg is missing or drafting "
            f"never ran")
    if not scoring:
        raise SystemExit(f"{arm}: no leg emitted any round")
    return scoring[-1]


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
    for path in sorted(arm_dir.glob("trace.txt*"), key=trace_pid):
        begin, rounds = parse_trace(path)
        if not rounds:
            continue
        tail = [r for r in rounds if r["round"] > warmup]
        steady = [r for r in tail if r["acc"] == r["d"]]
        legs.append({
            "trace": path.name, "pid": trace_pid(path), "begin": begin,
            "rounds_total": len(rounds),
            "dropped_warmup": len(rounds) - len(tail),
            "dropped_partial": len(tail) - len(steady),
            "depths_seen": sorted({r["d"] for r in rounds}),
            "acc_hist": {str(k): sum(1 for r in rounds if r["acc"] == k)
                         for k in sorted({r["acc"] for r in rounds})},
            "depth_hist": {str(k): sum(1 for r in rounds if r["d"] == k)
                           for k in sorted({r["d"] for r in rounds})},
            "acc_mean": statistics.fmean([r["acc"] for r in rounds]),
            "steady_rows": steady, "tail_rows": tail,
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
    ap.add_argument("--c0-arm", default=None,
                    help="normalise h(d) against this arm's depth-0 rounds "
                         "only; every leg contributes a serial control, and "
                         "an out-of-session control is not the base the cost "
                         "model means by `1`")
    args = ap.parse_args()

    arm_dirs = ([args.out_dir / a for a in args.arms] if args.arms
                else sorted(p for p in args.out_dir.iterdir() if p.is_dir()))

    by_depth = defaultdict(list)
    by_depth_all = defaultdict(list)
    arms = {}
    print(f"{'arm':<14} {'trace':<22} {'N':>4} {'depths':<10} {'acc':>5} "
          f"{'drop_w':>6} {'drop_p':>6}  {'leg':<4}")
    for arm_dir in arm_dirs:
        if not arm_dir.is_dir():
            print(f"skip missing {arm_dir}", file=sys.stderr)
            continue
        legs = load_legs(arm_dir, args.warmup)
        if not legs:
            continue
        meta = read_meta(arm_dir)
        mtp_leg = select_mtp_leg(legs, meta, arm_dir.name)
        score_path = arm_dir / "score.json"
        arms[arm_dir.name] = {
            "meta": meta,
            "mtp_leg_pid": mtp_leg["pid"],
            "mtp_leg_trace": mtp_leg["trace"],
            "score": json.loads(score_path.read_text()) if score_path.exists() else None,
            "legs": [{k: v for k, v in leg.items()
                      if k not in ("steady_rows", "tail_rows")}
                     for leg in legs],
        }
        for leg in legs:
            tag = "MTP" if leg is mtp_leg else "ref"
            print(f"{arm_dir.name:<14} {leg['trace']:<22} {leg['rounds_total']:>4} "
                  f"{str(leg['depths_seen']):<10} {leg['acc_mean']:>5.2f} "
                  f"{leg['dropped_warmup']:>6} {leg['dropped_partial']:>6}  {tag}")
            for row in leg["tail_rows"]:
                row["arm"] = arm_dir.name
                row["trace"] = leg["trace"]
                by_depth_all[row["d"]].append(row)
            for row in leg["steady_rows"]:
                by_depth[row["d"]].append(row)

    all_c0 = by_depth.get(0, [])
    print(f"\ndepth-0 control by arm ({'pooled' if not args.c0_arm else args.c0_arm + ' selected'})")
    c0_by_arm = defaultdict(list)
    c0_by_leg = defaultdict(list)
    for row in all_c0:
        c0_by_arm[row["arm"]].append(row)
        c0_by_leg[(row["arm"], row["trace"])].append(row)
    for arm in sorted(c0_by_arm):
        s = summarize(c0_by_arm[arm])
        arms[arm]["c0"] = s
        print(f"  {arm:<14} {'(all legs)':<22} N={s['n']:>4} mean={s['mean_us']:>9.1f} "
              f"median={s['median_us']:>9.1f} sd={s['sd_pct']:>5.1f}%")
        for (a, trace) in sorted(k for k in c0_by_leg if k[0] == arm):
            ls = summarize(c0_by_leg[(a, trace)])
            print(f"  {'':<14} {trace:<22} N={ls['n']:>4} mean={ls['mean_us']:>9.1f} "
                  f"median={ls['median_us']:>9.1f} sd={ls['sd_pct']:>5.1f}%")
    if all_c0:
        pooled = summarize(all_c0)
        print(f"  {'POOLED':<14} N={pooled['n']:>4} mean={pooled['mean_us']:>9.1f} "
              f"median={pooled['median_us']:>9.1f} sd={pooled['sd_pct']:>5.1f}%")

    c0_rows = c0_by_arm.get(args.c0_arm, []) if args.c0_arm else all_c0
    if args.c0_arm and not c0_rows:
        print(f"\nerror: --c0-arm {args.c0_arm} has no steady depth-0 rounds",
              file=sys.stderr)
        return 2
    c0 = summarize(c0_rows)["mean_us"] if c0_rows else None
    if c0 is None:
        print("\nwarning: no depth-0 rounds; h(d) cannot be normalised",
              file=sys.stderr)

    print(f"\nfull-accept rounds only (acc == d)")
    print(f"{'d':>2} {'N':>4} {'C(d) us':>10} {'median':>9} {'sd%':>6} "
          f"{'marg':>9} {'h(d)':>7} {'C/C0':>7} {'us/tok':>8} {'eval':>8} {'host':>8} "
          f"{'Nall':>5} {'C_all':>9}")
    curve, prev = {}, None
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        s = summarize(rows)
        s_all = summarize(by_depth_all[depth])
        marginal = None if prev is None else s["mean_us"] - prev
        h = None if (marginal is None or not c0) else marginal / c0
        phases = {p: summarize(rows, p) for p in PHASES}
        host = sum(phases[p]["mean_us"] for p in PHASES if p != "eval_wall_us")
        ratio = s["mean_us"] / c0 if c0 else None
        curve[depth] = {"depth": depth, "steady": s, "all_rounds": s_all,
                        "marginal_us": marginal,
                        "h": h, "c_over_c0": ratio,
                        "us_per_token": s["mean_us"] / (depth + 1),
                        "phases": phases, "host_us": host,
                        "arms": sorted({r["arm"] for r in rows})}
        marg_s = "-" if marginal is None else "%.1f" % marginal
        h_s = "-" if h is None else "%.4f" % h
        ratio_s = "-" if ratio is None else "%.3f" % ratio
        print(f"{depth:>2} {s['n']:>4} {s['mean_us']:>10.1f} {s['median_us']:>9.1f} "
              f"{s['sd_pct']:>6.1f} {marg_s:>9} {h_s:>7} {ratio_s:>7} "
              f"{s['mean_us'] / (depth + 1):>8.1f} "
              f"{phases['eval_wall_us']['mean_us']:>8.1f} {host:>8.1f} "
              f"{s_all['n']:>5} {s_all['mean_us']:>9.1f}")
        prev = s["mean_us"]

    # Every arm carries its own depth-0 control, so each C(d) can be divided by
    # the C(0) measured in the same arm. That cancels between-arm drift (clock,
    # temperature, allocator age) which the pooled table above leaves in the
    # marginal, since C(d) and C(d-1) come from different runs there.
    self_norm = {}
    c0_arm_mean = {a: summarize(rows)["mean_us"]
                   for a, rows in c0_by_arm.items()}
    if c0_arm_mean:
        print("\nself-normalised: each round divided by its own arm's C(0)")
        print(f"{'d':>2} {'N':>4} {'arms':>4} {'C/C0':>8} {'sd%':>6} {'h(d)':>8}")
        prev_r = None
        for depth in sorted(by_depth):
            vals = [r["round_us"] / c0_arm_mean[r["arm"]]
                    for r in by_depth[depth] if r["arm"] in c0_arm_mean]
            if not vals:
                continue
            mean = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            h = None if prev_r is None else mean - prev_r
            self_norm[depth] = {"n": len(vals), "c_over_c0": mean,
                                "sd_pct": 100.0 * sd / mean if mean else 0.0,
                                "h": h,
                                "arms": sorted({r["arm"] for r in by_depth[depth]})}
            print(f"{depth:>2} {len(vals):>4} {len(self_norm[depth]['arms']):>4} "
                  f"{mean:>8.4f} {100.0 * sd / mean if mean else 0:>6.1f} "
                  f"{'-' if h is None else '%.4f' % h:>8}")
            prev_r = mean

    if curve:
        vec = [curve[d]["h"] for d in sorted(curve) if d and curve[d]["h"] is not None]
        print("\npooled     headStepCostRatioByDepth = [" +
              ", ".join("%.4f" % v for v in vec) + "]")
    if self_norm:
        vec = [self_norm[d]["h"] for d in sorted(self_norm)
               if d and self_norm[d]["h"] is not None]
        print("self-norm  headStepCostRatioByDepth = [" +
              ", ".join("%.4f" % v for v in vec) + "]")

    if args.json:
        args.json.write_text(json.dumps(
            {"c0_us": c0, "c0_arm": args.c0_arm, "warmup": args.warmup,
             "c0_by_arm_us": c0_arm_mean,
             "self_normalised": {str(k): v for k, v in self_norm.items()},
             "arms": arms, "curve": {str(k): v for k, v in curve.items()}},
            indent=2, sort_keys=True))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
