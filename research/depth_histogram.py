#!/usr/bin/env python3
"""Chosen-depth histogram and offered-vs-realised depth for adaptive arms.

Splits each arm's trace into the serial reference leg and the MTP leg on the
round-counter reset, then reports the distribution of depths the policy actually
chose. Answers the advisor's comment-6 demand for an explicit statement of
whether the policy is sitting on the width cap at 4.
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")


def legs(path):
    out, cur, last = [], [], -1
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ROUND_RE.search(line)
            if not m:
                continue
            r, d, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if r <= last and cur:
                out.append(cur)
                cur = []
            last = r
            cur.append((r, d, a))
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    for arm in args.arms:
        paths = sorted(glob.glob(os.path.join(args.out_dir, arm, "trace.txt.*")),
                       key=lambda p: int(p.rsplit(".", 1)[-1]))
        # Legs may share one process (forced-depth arms) or be split across
        # worker PIDs (adaptive arms), so collect from every file.
        all_legs = [(int(p.rsplit(".", 1)[-1]), lg)
                    for p in paths for lg in legs(p) if lg]
        if not all_legs:
            print(f"{arm}: no rounds found", file=sys.stderr)
            continue
        # The serial reference leg is all d=0, so a nonzero depth names the MTP
        # leg outright. Reading the reference leg by accident would silently
        # report a real MTP arm as a serial control, so say which PID was used.
        drafting = [(pid, lg) for pid, lg in all_legs if any(d for _, d, _ in lg)]
        if len(drafting) > 1:
            sys.exit(f"{arm}: {len(drafting)} legs carry nonzero depths "
                     f"{[pid for pid, _ in drafting]}; cannot name one MTP leg")
        pid, rounds = drafting[0] if drafting else all_legs[-1]
        rounds = rounds[args.warmup:]
        hist = Counter(d for _, d, _ in rounds)
        acc_by_d = {}
        for _, d, a in rounds:
            acc_by_d.setdefault(d, []).append(a)
        n = len(rounds)
        tok = sum(a for _, _, a in rounds) + n
        print(f"\n=== {arm}  mtp_leg_pid={pid}  rounds={n}  "
              f"committed_tokens={tok} ===")
        print(f"{'d':>3} {'rounds':>7} {'share':>8} {'mean_acc':>9} {'accept_rate':>12}")
        for d in sorted(hist):
            accs = acc_by_d[d]
            ma = sum(accs) / len(accs)
            rate = ma / d if d else float("nan")
            print(f"{d:>3} {hist[d]:>7} {hist[d]/n:>8.4f} {ma:>9.4f} {rate:>12.4f}")
        mean_d = sum(d for _, d, _ in rounds) / n
        mean_a = sum(a for _, _, a in rounds) / n
        at_cap4 = sum(v for k, v in hist.items() if k == 4)
        above4 = sum(v for k, v in hist.items() if k > 4)
        print(f"mean_offered_depth={mean_d:.4f} mean_accepted={mean_a:.4f} "
              f"tokens_per_round={tok/n:.4f}")
        print(f"rounds_at_d==4: {at_cap4} ({at_cap4/n:.4%})   "
              f"rounds_at_d>4: {above4} ({above4/n:.4%})")


if __name__ == "__main__":
    main()
