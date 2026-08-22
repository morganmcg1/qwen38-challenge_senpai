"""Measure the per-prompt candidate-leg null from byte-identical replicate pairs.

Two submissions that carry the same candidate source commit differ only by the
runner draw. The spread of their per-prompt candidate-leg ratio is therefore a
directly measured null distribution, and it is the correct scale against which
to judge a per-prompt effect. The cross-prompt spread of one receipt pair is NOT
that scale: under a real dose-response it contains the signal.

harness=ranked. Read-only over the cached Yukon board.
"""

import itertools
import json
import math
import sys

from e129_schedule_invariance import NAMES, per_prompt

BOARD = "/tmp/yukon-board/full.json"


def rows_with_metrics():
    out = []
    for r in json.load(open(BOARD)):
        m = r.get("officialMetrics") or {}
        if not m.get("per_prompt"):
            continue
        pp = per_prompt(r)
        if len(pp) != len(NAMES):
            continue
        out.append(r)
    return out


def source_key(r):
    m = r.get("officialMetrics") or {}
    return r.get("submissionCommitSha")


def cand_deltas(a, b):
    pa, pb = per_prompt(a), per_prompt(b)
    return [(1.0 - pa[n]["mtp_seconds_per_token_mean"]
             / pb[n]["mtp_seconds_per_token_mean"]) * 100
            for n in NAMES.values()]


def main():
    rows = rows_with_metrics()
    groups = {}
    for r in rows:
        k = source_key(r)
        if k:
            groups.setdefault(k, []).append(r)

    pairs = [(k, a, b) for k, v in groups.items() if len(v) > 1
             for a, b in itertools.combinations(v, 2)]
    print(f"harness=ranked   {len(rows)} receipts with all eight prompts, "
          f"{len(pairs)} same-source replicate pairs")
    print()

    allabs = []
    print(f"{'source':<12}{'a':<10}{'b':<10}{'max|d|%':>9}{'sd%':>8}"
          f"{'mean%':>8}   per-prompt candidate d%")
    for k, a, b in sorted(pairs, key=lambda t: t[0]):
        d = cand_deltas(a, b)
        allabs.extend(abs(x) for x in d)
        mx = max(abs(x) for x in d)
        mean = sum(d) / len(d)
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1))
        body = " ".join(f"{x:+.3f}" for x in d)
        print(f"{k[:10]:<12}{a['id'][:8]:<10}{b['id'][:8]:<10}"
              f"{mx:>9.3f}{sd:>8.3f}{mean:>8.3f}   {body}")

    if allabs:
        allabs.sort()
        n = len(allabs)
        print()
        print(f"pooled |per-prompt candidate delta| over {n} observations")
        print(f"  max      {allabs[-1]:.3f} %")
        print(f"  p95      {allabs[int(0.95 * n) - 1]:.3f} %")
        print(f"  median   {allabs[n // 2]:.3f} %")
        rms = math.sqrt(sum(x * x for x in allabs) / n)
        print(f"  rms      {rms:.3f} %")
        print()
        print("prompt order: " + " ".join(NAMES.values()))


if __name__ == "__main__":
    sys.exit(main())
