#!/usr/bin/env python3
"""Research-only: recover per-depth repair-path counters from existing traces.

Answers "how often does a prefix reject take the cheap restore vs. the full
re-forward?" without rebuilding the binary.

Qwen36MTPBlockSession.swift stamps tReadDone at :1219, *before* the
accept/reject branch, and tCommitDone at :1287, *after* the whole branch --
including the rollbackAfterVerify + model.callWithHidden re-forward at :1267.
commit_us = (tCommitDone - tReadDone)/1000 therefore brackets the repair, and
upkeep_us (tTailDone - tCommitDone) does not.

That makes commit_us a direct per-round classifier:

    acc == d                      full accept, branch never entered
    acc <  d, commit_us small     restoreAfterPrefixReject returned true
                                  (cache trim + row slicing, no blocking eval)
    acc <  d, commit_us large     it returned false: rollbackAfterVerify plus a
                                  full target forward over the committed block

The two classes are separated by orders of magnitude, not by a tuned cutoff:
any target forward on this host is memory-bound at >=60 ms because it reads the
whole 4-bit backbone, while the cheap path allocates no kernel launches.
--full-repair-us defaults to 10000 (10 ms), ~6x below the cheapest possible
forward and ~40x above the observed cheap-path cost.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=(-?\d+)")


def parse_trace(path):
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        m = ROUND_RE.match(line.strip())
        if not m:
            continue
        row = {"round": int(m.group(1)), "d": int(m.group(2)),
               "acc": int(m.group(3))}
        row.update({k: int(v) for k, v in KV_RE.findall(m.group(4))})
        rounds.append(row)
    return rounds


def regime(d):
    """Checkpoint regime a depth-d round runs under.

    The verify window is S = d + 1 rows. Qwen35.swift takes a single-launch
    free-checkpoint path at S == 2 and records the cheap replay tape only when
    nConfirmed == 1 && S >= 3 && mask == nil (:977, written :1112, consumed by
    replayPrefix :889); otherwise it falls back to the eager-checkpoint kernel.
    """
    s = d + 1
    if d == 0:
        return "S=1 no drafts, reject impossible"
    if s == 2:
        return "S=2 single-launch free checkpoint"
    return f"S={s} replay tape"


def stats(vals):
    if not vals:
        return None
    out = {"n": len(vals), "mean": statistics.fmean(vals),
           "median": statistics.median(vals),
           "min": min(vals), "max": max(vals)}
    out["sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--warmup", type=int, default=2,
                    help="leading rounds dropped per leg")
    ap.add_argument("--full-repair-us", type=float, default=10000.0)
    ap.add_argument("--serial-round-us", type=float, default=65115.0,
                    help="C(0); cost of one full target forward")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    arms = args.arms or sorted(
        p.name for p in args.out_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_"))

    by_depth = defaultdict(lambda: {"full": [], "prefix": [], "reforward": []})
    legs = 0
    # confusion between the commit_us classifier and the literal counters, for
    # legs produced by a binary that emits prefix_repair=/full_repair=
    xcheck = {"legs": 0, "rounds": 0, "agree_prefix": 0, "agree_full": 0,
              "classifier_prefix_literal_full": 0,
              "classifier_full_literal_prefix": 0, "literal_neither": 0}
    for arm in arms:
        for tf in sorted((args.out_dir / arm).glob("trace.txt.*")):
            rows = parse_trace(tf)
            if not rows:
                continue
            legs += 1
            # counters are cumulative per leg; recover per-round increments
            literal = "prefix_repair" in rows[0]
            if literal:
                xcheck["legs"] += 1
                prev_p = prev_f = 0
                for row in rows:
                    p, f = row["prefix_repair"], row["full_repair"]
                    row["d_prefix_repair"] = p - prev_p
                    row["d_full_repair"] = f - prev_f
                    prev_p, prev_f = p, f
            for row in rows[args.warmup:]:
                d = row["d"]
                if d == 0:
                    continue
                row["arm"] = arm
                if row["acc"] == d:
                    by_depth[d]["full"].append(row)
                    continue
                if row.get("commit_us", 0) > args.full_repair_us:
                    by_depth[d]["reforward"].append(row)
                    said_full = True
                else:
                    by_depth[d]["prefix"].append(row)
                    said_full = False
                if not literal:
                    continue
                xcheck["rounds"] += 1
                lp, lf = row["d_prefix_repair"], row["d_full_repair"]
                if lf:
                    key = "agree_full" if said_full else "classifier_prefix_literal_full"
                elif lp:
                    key = "classifier_full_literal_prefix" if said_full else "agree_prefix"
                else:
                    key = "literal_neither"
                xcheck[key] += 1

    report = {"legs": legs, "arms": arms, "warmup": args.warmup,
              "full_repair_us_threshold": args.full_repair_us,
              "serial_round_us": args.serial_round_us, "depths": {}}

    hdr = (f"{'d':>2} {'regime':<34} {'N_full':>7} {'N_rej':>6} "
           f"{'prefixRepair':>12} {'fullRepair':>10} "
           f"{'commit_acc':>10} {'commit_rej':>10} {'commit_max':>10} "
           f"{'dRound_mean':>11} {'dRound_med':>10} {'bound_med':>9}")
    print(hdr)
    print("-" * len(hdr))

    for d in sorted(by_depth):
        cls = by_depth[d]
        full, prefix, refwd = cls["full"], cls["prefix"], cls["reforward"]
        rej = prefix + refwd
        c_acc = stats([r.get("commit_us", 0) for r in full])
        c_rej = stats([r.get("commit_us", 0) for r in rej])
        r_acc = stats([r["round_us"] for r in full])
        r_rej = stats([r["round_us"] for r in rej])

        both = r_acc and r_rej
        delta = (r_rej["mean"] - r_acc["mean"]) if both else None
        # verify_build_us absorbs command-queue backpressure, so a single
        # stalled round can dominate the mean at small N; the median delta is
        # the statistic that actually bounds a systematic repair term.
        delta_med = (r_rej["median"] - r_acc["median"]) if both else None
        # Upper bound on the fraction of reject rounds that could have hidden a
        # full re-forward inside the round-time budget, independent of commit_us.
        bound = (delta / args.serial_round_us) if delta is not None else None
        bound_med = (delta_med / args.serial_round_us) if both else None

        entry = {
            "regime": regime(d),
            "n_full_accept": len(full),
            "n_reject": len(rej),
            "prefixRepairCount": len(prefix),
            "fullRepairCount": len(refwd),
            "commit_us_full_accept": c_acc,
            "commit_us_reject": c_rej,
            "round_us_full_accept": r_acc,
            "round_us_reject": r_rej,
            "round_us_delta_mean": delta,
            "round_us_delta_median": delta_med,
            "reforward_fraction_upper_bound_mean": bound,
            "reforward_fraction_upper_bound_median": bound_med,
        }
        report["depths"][d] = entry

        nan = float("nan")
        print(f"{d:>2} {regime(d):<34} {len(full):>7} {len(rej):>6} "
              f"{len(prefix):>12} {len(refwd):>10} "
              f"{(c_acc['mean'] if c_acc else nan):>10.1f} "
              f"{(c_rej['mean'] if c_rej else nan):>10.1f} "
              f"{(c_rej['max'] if c_rej else nan):>10.0f} "
              f"{(delta if delta is not None else nan):>11.1f} "
              f"{(delta_med if delta_med is not None else nan):>10.1f} "
              f"{(bound_med if bound_med is not None else nan):>9.4f}")

    tot_prefix = sum(len(v["prefix"]) for v in by_depth.values())
    tot_refwd = sum(len(v["reforward"]) for v in by_depth.values())
    report["total_prefixRepairCount"] = tot_prefix
    report["total_fullRepairCount"] = tot_refwd
    print(f"\nlegs={legs}  prefixRepairCount={tot_prefix}  "
          f"fullRepairCount={tot_refwd}")

    all_commit_rej = [r.get("commit_us", 0)
                      for v in by_depth.values() for r in v["prefix"] + v["reforward"]]
    if all_commit_rej:
        s = stats(all_commit_rej)
        report["commit_us_reject_pooled"] = s
        print(f"pooled reject commit_us: n={s['n']} mean={s['mean']:.1f} "
              f"median={s['median']:.1f} max={s['max']} "
              f"(full re-forward floor ~{args.serial_round_us:.0f})")

    report["counter_crosscheck"] = xcheck
    if xcheck["rounds"]:
        agree = xcheck["agree_prefix"] + xcheck["agree_full"]
        print(f"\ncounter cross-check ({xcheck['legs']} legs from a binary that "
              f"emits the literal counters, {xcheck['rounds']} reject rounds):")
        print(f"  classifier prefix & literal prefix : {xcheck['agree_prefix']}")
        print(f"  classifier full   & literal full   : {xcheck['agree_full']}")
        print(f"  classifier prefix & literal full   : "
              f"{xcheck['classifier_prefix_literal_full']}")
        print(f"  classifier full   & literal prefix : "
              f"{xcheck['classifier_full_literal_prefix']}")
        print(f"  literal incremented neither        : {xcheck['literal_neither']}")
        print(f"  agreement: {agree}/{xcheck['rounds']} = "
              f"{agree / xcheck['rounds']:.4f}")
    else:
        print("\ncounter cross-check: no legs carried the literal counters")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
