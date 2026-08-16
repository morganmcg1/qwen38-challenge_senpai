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
from collections import Counter, defaultdict
from pathlib import Path

ROUND = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
ROUND_US = re.compile(r"round_us=(\d+)")
REPAIR = re.compile(r"prefix_repair=(\d+) full_repair=(\d+)")
BEGIN_H = re.compile(r"mtp-trace: begin.* h=([0-9.,eE+-]+)")


def load_legs(arm_dir, warmup):
    """Split every trace file into legs; a leg restarts at `round=0`.

    Returns (legs, h_texts) where a leg is a list of (idx, d, acc, round_us)
    and the leading `warmup` rounds have already been dropped.
    """
    legs, h_texts = [], []
    for path in sorted(arm_dir.glob("trace.txt.*")):
        per_leg = []
        for line in path.read_text(errors="replace").splitlines():
            hm = BEGIN_H.search(line)
            if hm:
                h_texts.append(hm.group(1))
            m = ROUND.search(line)
            if not m:
                continue
            idx, d, acc = (int(x) for x in m.groups())
            us = ROUND_US.search(line)
            if idx == 0 and per_leg:
                legs.append(per_leg[warmup:])
                per_leg = []
            per_leg.append((idx, d, acc, int(us.group(1)) if us else 0))
        if per_leg:
            legs.append(per_leg[warmup:])
    return legs, h_texts


def load(arm_dir, warmup):
    legs, _ = load_legs(arm_dir, warmup)
    return [r for leg in legs for r in leg]


def repair_split(arm_dir):
    """Count rounds taking each repair branch, keyed by chosen depth.

    The session exports cumulative counters, so a round took a branch when
    that counter advanced since the previous round of the same leg. Legs are
    detected by the counters resetting, which also makes the pre-counter
    traces degrade to "no rows" rather than to wrong rows.
    """
    prefix, full, rejects = Counter(), Counter(), Counter()
    for path in sorted(arm_dir.glob("trace.txt.*")):
        prev_p = prev_f = 0
        for line in path.read_text(errors="replace").splitlines():
            m = ROUND.search(line)
            r = REPAIR.search(line)
            if not m or not r:
                continue
            d = int(m.group(2))
            p, f = int(r.group(1)), int(r.group(2))
            if p < prev_p or f < prev_f:
                prev_p = prev_f = 0
            if p > prev_p:
                prefix[d] += 1
            if f > prev_f:
                full[d] += 1
            if p > prev_p or f > prev_f:
                rejects[d] += 1
            prev_p, prev_f = p, f
    return prefix, full, rejects


def split_at_eos(legs, eos_index):
    """Partition rounds by whether they commit before or after `eos_index`.

    A round commits `acc + 1` tokens, so the decode index of its first
    committed token is the running total over earlier rounds of that leg.
    Rounds straddling the boundary are dropped from both sides rather than
    assigned to one, so neither segment mixes the two regimes.
    """
    pre, post = [], []
    for leg in legs:
        if not any(d for _, d, _, _ in leg):
            continue  # serial reference leg carries no acceptance signal
        cursor = 0
        for r in leg:
            emitted = r[2] + 1
            if cursor + emitted <= eos_index:
                pre.append(r)
            elif cursor >= eos_index:
                post.append(r)
            cursor += emitted
    return pre, post


def binned_profile(legs, width):
    """Mean depth/acceptance by decode-index bucket, to LOCATE regime changes.

    The EOS boundary is not recorded in the trace, so the split index must be
    justified from the data rather than assumed. A copy task that runs out of
    source text and continues degenerately shows up here as a step.
    """
    buckets = defaultdict(list)
    for leg in legs:
        if not any(d for _, d, _, _ in leg):
            continue  # serial reference leg carries no acceptance signal
        cursor = 0
        for r in leg:
            buckets[cursor // width].append(r)
            cursor += r[2] + 1
    return sorted(buckets.items())


def acceptance_table(rounds):
    """Per-position (reached, accepted, p_i) over a round list."""
    rows = []
    for i in range(8):
        reached = sum(1 for _, d, a, _ in rounds if d > i and a >= i)
        ok = sum(1 for _, d, a, _ in rounds if d > i and a > i)
        if reached:
            rows.append((i + 1, reached, ok, ok / reached))
    return rows


def print_segment(label, rounds):
    if not rounds:
        print(f"  [{label}] no rounds")
        return
    total_d = sum(d for _, d, _, _ in rounds)
    total_a = sum(a for _, _, a, _ in rounds)
    print(f"  [{label}] N={len(rounds)} "
          f"mean_depth={total_d / len(rounds):.3f} "
          f"mean_acc={total_a / len(rounds):.3f} "
          f"tokens/round={1 + total_a / len(rounds):.3f}")
    for pos, reached, ok, p in acceptance_table(rounds):
        print(f"  [{label}]   pos {pos}: reached {reached:>5} "
              f"accepted {ok:>5}  p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--eos-index", type=int, default=301,
                    help="decode index where the public trajectory emits EOS")
    ap.add_argument("--bin", type=int, default=64,
                    help="decode-index bucket width for the regime scan")
    args = ap.parse_args()

    arms = args.arms or sorted(
        p.name for p in args.out_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_"))

    for arm in arms:
        arm_dir = args.out_dir / arm
        legs, h_texts = load_legs(arm_dir, args.warmup)
        rounds = [r for leg in legs for r in leg]
        if not rounds:
            continue
        depth_hist = Counter(d for _, d, _, _ in rounds)
        acc_hist = Counter(a for _, _, a, _ in rounds)
        total_d = sum(d for _, d, _, _ in rounds)
        total_a = sum(a for _, _, a, _ in rounds)
        print(f"\n=== {arm}: {len(rounds)} scored rounds in {len(legs)} leg(s) "
              f"(warmup {args.warmup} dropped per leg) ===")
        for h in dict.fromkeys(h_texts):
            print(f"  active h vector: {h}")
        print("  chosen depth histogram: " + ", ".join(
            f"d={d}:{n}" for d, n in sorted(depth_hist.items())))
        print("  accepted-count histogram: " + ", ".join(
            f"{a}:{n}" for a, n in sorted(acc_hist.items())))
        print(f"  mean chosen depth = {total_d / len(rounds):.3f}   "
              f"mean accepted drafts = {total_a / len(rounds):.3f}   "
              f"tokens/round = {1 + total_a / len(rounds):.3f}")

        prefix, full, rejects = repair_split(arm_dir)
        if rejects:
            print("  --- repair branch split per chosen depth ---")
            for d in sorted(rejects):
                n = rejects[d]
                print(f"  d={d}: rejecting_rounds={n:>4} "
                      f"prefixRepairCount={prefix[d]:>4} "
                      f"fullRepairCount={full[d]:>4} "
                      f"full_share={full[d] / n:.4f}")
            print(f"  totals: prefixRepairCount={sum(prefix.values())} "
                  f"fullRepairCount={sum(full.values())}")
        else:
            print("  repair split: unavailable (trace predates the counters)")

        drafting = [r for r in rounds if r[1] > 0]
        if drafting:
            above = sum(1 for _, d, _, _ in drafting if d > 4)
            at = sum(1 for _, d, _, _ in drafting if d == 4)
            top = max(d for _, d, _, _ in drafting)
            print(f"  width-wall check: max chosen depth = {top}, "
                  f"d==4 {at}/{len(drafting)} ({100 * at / len(drafting):.1f}%), "
                  f"d>4 {above}/{len(drafting)} "
                  f"({100 * above / len(drafting):.1f}%) -> "
                  + ("CLIPPED AT 4" if above == 0 else "not clipped at 4"))

        print(f"  realised T(d): {'d':>3} {'N':>6} {'acc/round':>10} "
              f"{'tokens/round':>13} {'accept rate':>12}")
        by_depth = defaultdict(list)
        for _, d, a, _ in rounds:
            by_depth[d].append(a)
        for d in sorted(by_depth):
            accs = by_depth[d]
            mean_a = sum(accs) / len(accs)
            rate = mean_a / d if d else float("nan")
            print(f"  {'':>16} {d:>3} {len(accs):>6} {mean_a:>10.3f} "
                  f"{1 + mean_a:>13.3f} {rate:>12.4f}")

        for i, leg in enumerate(legs):
            if len(leg) < 2:
                continue
            tail = leg[1:]
            tok = sum(a for _, _, a, _ in leg) + len(leg)
            tok_t = sum(a for _, _, a, _ in tail) + len(tail)
            us = sum(u for _, _, _, u in leg)
            us_t = sum(u for _, _, _, u in tail)
            print(f"  leg {i}: rounds={len(leg)} "
                  f"round_index {leg[0][0]}..{leg[-1][0]} tokens={tok} "
                  f"s/token={us / tok / 1e6:.6f} "
                  f"after_first_block s/token={us_t / tok_t / 1e6:.6f}")

        print(f"  {'pos':>4} {'reached':>8} {'accepted':>9} {'p_i':>7} "
              f"{'shipped prior':>14}")
        for pos, reached, ok, p in acceptance_table(rounds):
            prior = 0.85 * 0.98 ** (pos - 1)
            print(f"  {pos:>4} {reached:>8} {ok:>9} {p:>7.4f} "
                  f"{prior:>14.4f}")

        print(f"  --- decode-index bins of {args.bin} tokens ---")
        for b, rs in binned_profile(legs, args.bin):
            md = sum(d for _, d, _, _ in rs) / len(rs)
            ma = sum(a for _, _, a, _ in rs) / len(rs)
            us = sum(u for _, _, _, u in rs) / len(rs)
            print(f"  [{b * args.bin:>4}-{(b + 1) * args.bin - 1:>4}] "
                  f"N={len(rs):>4} mean_depth={md:.3f} mean_acc={ma:.3f} "
                  f"tokens/round={1 + ma:.3f} round_us={us:.0f}")

        pre, post = split_at_eos(legs, args.eos_index)
        print(f"  --- EOS split at decode index {args.eos_index} "
              f"(straddling rounds dropped) ---")
        print_segment("pre-EOS", pre)
        print_segment("post-EOS", post)


if __name__ == "__main__":
    main()
