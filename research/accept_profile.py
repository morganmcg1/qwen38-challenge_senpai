#!/usr/bin/env python3
"""Realised draft-depth histogram and per-position acceptance from round traces.

The schedule is an OFFER-taker: the parent offers a per-round ceiling and the
session picks `d`. Two things the cost curve cannot tell you on its own:

  * how deep the schedule actually goes when it is free to choose, and
  * the realised acceptance at each draft position.

Position `i` is OBSERVED in a round when a draft existed there (`d > i`) and
every earlier draft was accepted (`acc >= i`); it is a success when
`acc > i`. Positions past the first rejection are never reached and contribute
nothing, which is why a naive `acc/d` average is not the same estimator.

usage: research/accept_profile.py OUT_DIR [--arms a b ...] [--warmup N]
"""
import argparse
import re
from collections import Counter
from pathlib import Path

ROUND = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")


def load(arm_dir, warmup):
    rounds = []
    for path in sorted(arm_dir.glob("trace.txt.*")):
        per_leg = []
        for line in path.read_text(errors="replace").splitlines():
            m = ROUND.search(line)
            if not m:
                continue
            idx, d, acc = (int(x) for x in m.groups())
            if idx == 0 and per_leg:
                rounds.extend(per_leg[warmup:])
                per_leg = []
            per_leg.append((idx, d, acc))
        rounds.extend(per_leg[warmup:])
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    arms = args.arms or sorted(
        p.name for p in args.out_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_"))

    for arm in arms:
        arm_dir = args.out_dir / arm
        rounds = load(arm_dir, args.warmup)
        if not rounds:
            continue
        depth_hist = Counter(d for _, d, _ in rounds)
        acc_hist = Counter(a for _, _, a in rounds)
        total_d = sum(d for _, d, _ in rounds)
        total_a = sum(a for _, _, a in rounds)
        print(f"\n=== {arm}: {len(rounds)} scored rounds "
              f"(warmup {args.warmup} dropped per leg) ===")
        print("  chosen depth histogram: " + ", ".join(
            f"d={d}:{n}" for d, n in sorted(depth_hist.items())))
        print("  accepted-count histogram: " + ", ".join(
            f"{a}:{n}" for a, n in sorted(acc_hist.items())))
        print(f"  mean chosen depth = {total_d / len(rounds):.3f}   "
              f"mean accepted drafts = {total_a / len(rounds):.3f}   "
              f"tokens/round = {1 + total_a / len(rounds):.3f}")
        print(f"  {'pos':>4} {'reached':>8} {'accepted':>9} {'p_i':>7} "
              f"{'shipped prior':>14}")
        for i in range(8):
            reached = sum(1 for _, d, a in rounds if d > i and a >= i)
            ok = sum(1 for _, d, a in rounds if d > i and a > i)
            if not reached:
                continue
            prior = 0.85 * 0.98 ** i
            print(f"  {i + 1:>4} {reached:>8} {ok:>9} {ok / reached:>7.4f} "
                  f"{prior:>14.4f}")


if __name__ == "__main__":
    main()
