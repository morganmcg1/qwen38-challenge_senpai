#!/usr/bin/env python3
"""Replay the shipped acceptance ratchet over a real arm's rounds.

`positionAcceptEMA` is a deterministic function of the `(offered_depth,
accepted_count)` sequence, so the converged EMA and the per-round depth each
rule *would* have selected can both be recovered offline from a trace, without
re-running the model.

Two questions it answers:

1. The converged `positionAcceptEMA[0..7]` per arm (advisor comment 18 item 4).
2. Whether the shipped greedy walk and the argmin scan diverge on the states a
   real run actually visits -- as opposed to the flat-`q` idealisation, where
   they provably do.

The one input the trace does not carry is the depth-0 `pendingTop2` margin
clamp. It lowers `p[0]` for both rules identically, so it cannot by itself
create a divergence; it can only move a round into or out of one. Rounds where
the clamp could bind are reported separately.

Usage:
  research/ema_replay.py research/out/cand512 --h measured
  research/ema_replay.py research/out/base512 --h flat
"""
import argparse
import glob
import json
import os
import re
import sys

ALPHA = 0.15
MAX_DEPTH = 8
PRIOR = [0.85 * 0.98**i for i in range(MAX_DEPTH)]
SDPA_CAP = 4
SEGMENTED_CAP = 8
STREAK_GATE = 3

MEASURED_H = [0.0902, 0.0680, 0.2435, 0.3804, 0.2778, 0.2981, 0.2715, 0.4250]
FLAT_H = [0.20] * MAX_DEPTH

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")


def load_rounds(arm_dir):
    """Return the (d, acc) sequence of the trace with the most drafting rounds.

    The wrapper spawns a worker per leg; only the MTP leg drafts.
    """
    best = []
    for path in sorted(glob.glob(os.path.join(arm_dir, "trace.txt.*"))):
        rounds = []
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = ROUND_RE.search(line)
                if match:
                    rounds.append(
                        (int(match.group(2)), int(match.group(3))))
        if sum(1 for d, _ in rounds if d > 0) > sum(
                1 for d, _ in best if d > 0):
            best = rounds
    return best


def record(ema, accepted, drafted):
    for i in range(min(accepted, MAX_DEPTH)):
        ema[i] += ALPHA * (1.0 - ema[i])
    if accepted < drafted and accepted < MAX_DEPTH:
        ema[accepted] += ALPHA * (0.0 - ema[accepted])
    elif accepted == drafted and drafted > 0 and accepted < MAX_DEPTH:
        if ema[accepted] < 0.95:
            ema[accepted] += ALPHA * (0.95 - ema[accepted])


def greedy(h, ema, cap):
    """First local minimum of f(d) = (1 + sum h) / T(d).

    The shipped test `reach > h*(1+expected)/(1+depth*h)` is that comparison
    specialised to a constant marginal; the cumulative sum is its only correct
    de-specialisation to a vector.
    """
    reach, expected, cum, depth = 1.0, 0.0, 0.0, 0
    while depth < cap:
        reach *= ema[depth]
        if not reach > h[depth] * (1.0 + expected) / (1.0 + cum):
            break
        expected += reach
        cum += h[depth]
        depth += 1
    return depth


def argmin(h, ema, cap):
    reach, expected, cost = 1.0, 1.0, 1.0
    best, best_cpt = 0, 1.0
    for depth in range(cap):
        reach *= ema[depth]
        expected += reach
        cost += h[depth]
        cpt = cost / expected
        if cpt < best_cpt:
            best_cpt, best = cpt, depth + 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--h", default="measured",
                    choices=["measured", "flat"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--json", dest="json_path")
    args = ap.parse_args()

    h = MEASURED_H if args.h == "measured" else FLAT_H
    rounds = load_rounds(args.arm_dir)
    if not rounds:
        sys.exit(f"no round records under {args.arm_dir}")

    ema = list(PRIOR)
    streak = 0
    mismatches = []
    agree = 0
    clamp_sensitive = 0
    chosen = {}
    for index, (drafted, accepted) in enumerate(rounds):
        cap = min(SEGMENTED_CAP if streak >= STREAK_GATE else SDPA_CAP,
                  MAX_DEPTH)
        if index >= args.warmup:
            g = greedy(h, ema, cap)
            a = argmin(h, ema, cap)
            if g == a:
                agree += 1
            else:
                mismatches.append((index + 1, g, a, drafted))
            chosen[a] = chosen.get(a, 0) + 1
            # A clamp on p[0] can only matter where depth 0 is contested.
            if min(g, a) <= 1:
                clamp_sensitive += 1
        record(ema, accepted, drafted)
        streak = streak + 1 if accepted == drafted else 0

    total = agree + len(mismatches)
    print(f"arm={os.path.basename(args.arm_dir.rstrip('/'))} h={args.h} "
          f"rounds={len(rounds)} scored={total}")
    print("converged positionAcceptEMA[0..7] = ["
          + ", ".join(f"{v:.4f}" for v in ema) + "]")
    print("prior                             = ["
          + ", ".join(f"{v:.4f}" for v in PRIOR) + "]")
    print(f"greedy == argmin on {agree}/{total} realised rounds "
          f"({100.0 * agree / total:.2f}%)")
    if mismatches:
        print(f"  first 10 mismatches (round, greedy, argmin, actual d): "
              f"{mismatches[:10]}")
    print(f"argmin depth histogram: "
          f"{dict(sorted(chosen.items()))}")
    print(f"rounds where the depth-0 margin clamp could bind: "
          f"{clamp_sensitive}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump({
                "arm": os.path.basename(args.arm_dir.rstrip("/")),
                "h": args.h,
                "converged_ema": ema,
                "prior": PRIOR,
                "scored_rounds": total,
                "greedy_argmin_agree": agree,
                "mismatches": mismatches,
                "argmin_histogram": chosen,
                "clamp_sensitive_rounds": clamp_sensitive,
            }, handle, indent=2)


if __name__ == "__main__":
    main()
