#!/usr/bin/env python3
"""Realised decode-only seconds/token per arm, excluding the seed prologue.

The idealised C(d)/(d+1) in depth_cost_curve.py conditions on full acceptance.
This reports what the arm actually achieved: every post-warmup round's wall time
divided by every token those rounds actually committed.
"""
import json
import re
import sys
import glob
import os

ROUND = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?round_us=(\d+)"
)


def legs(arm_dir):
    """Split an arm's rounds into legs. A leg boundary is a round-counter reset,
    which is how the serial reference leg and the MTP leg appear in one trace."""
    out = []
    for path in sorted(glob.glob(os.path.join(arm_dir, "trace.txt.*"))):
        cur = []
        with open(path, errors="replace") as fh:
            for line in fh:
                m = ROUND.search(line)
                if not m:
                    continue
                row = tuple(int(g) for g in m.groups())
                if cur and row[0] <= cur[-1][0]:
                    out.append(cur)
                    cur = []
                cur.append(row)
        if cur:
            out.append(cur)
    return out


def summarise(rows, warmup=2):
    rows = rows[warmup:]
    if not rows:
        return None
    total_us = sum(r[3] for r in rows)
    tokens = sum(r[2] + 1 for r in rows)
    return {
        "rounds": len(rows),
        "tokens": tokens,
        "s_per_token": total_us / tokens / 1e6,
        "mean_round_us": total_us / len(rows),
        "mean_acc": sum(r[2] for r in rows) / len(rows),
        "mean_d": sum(r[1] for r in rows) / len(rows),
    }


def main():
    out = sys.argv[1]
    arms = sys.argv[2:]
    print(f"{'arm':6} {'leg':>4} {'rounds':>7} {'tokens':>7} {'mean d':>7} "
          f"{'s/token':>10} {'mean round us':>14} {'mean acc':>9}")
    per_arm = {}
    for a in arms:
        summaries = [summarise(rows) for rows in legs(os.path.join(out, a))]
        summaries = [s for s in summaries if s is not None]
        for i, r in enumerate(summaries):
            print(f"{a:6} {i:4d} {r['rounds']:7d} {r['tokens']:7d} "
                  f"{r['mean_d']:7.3f} {r['s_per_token']:10.6f} "
                  f"{r['mean_round_us']:14.1f} {r['mean_acc']:9.3f}")
        if len(summaries) == 2:
            per_arm[a] = summaries

    if not per_arm:
        return
    # Each arm carries its own serial leg, so the ratio cancels the per-arm
    # host and thermal drift that a cross-arm comparison would absorb.
    print(f"\n{'arm':6} {'serial s/tok':>13} {'mtp s/tok':>11} "
          f"{'speedup':>9} {'mean d':>7} {'mean acc':>9}")
    for a, (ref, mtp) in sorted(
        per_arm.items(), key=lambda kv: kv[1][1]["s_per_token"]
    ):
        print(f"{a:6} {ref['s_per_token']:13.6f} {mtp['s_per_token']:11.6f} "
              f"{ref['s_per_token'] / mtp['s_per_token']:9.4f} "
              f"{mtp['mean_d']:7.3f} {mtp['mean_acc']:9.3f}")

    prefill(out, arms)


def prefill(out, arms):
    """Recover the seed prologue from each leg independently.

    The harness clock covers seed processing plus decode; the round trace covers
    decode only. Their difference must be the same seed constant in the serial
    and MTP leg of every arm, which is what makes the trace safe to reason with.
    """
    rows = []
    for a in arms:
        path = os.path.join(out, a, "score.json")
        if not os.path.exists(path):
            continue
        m = json.load(open(path))["metrics"]
        n = m["decode_tokens"]
        ls = [summarise(r, warmup=0) for r in legs(os.path.join(out, a))]
        ls = [s for s in ls if s is not None]
        if len(ls) != 2:
            continue
        rows.append((
            a,
            n * m["serial_seconds_per_token"] - ls[0]["s_per_token"] * ls[0]["tokens"],
            ls[0]["rounds"],
            n * m["mtp_seconds_per_token"] - ls[1]["s_per_token"] * ls[1]["tokens"],
            ls[1]["rounds"],
        ))
    if not rows:
        return
    print(f"\n{'arm':6} {'P serial s':>11} {'n':>5} {'P mtp s':>9} {'n':>5} "
          f"{'delta s':>9}")
    for a, ps, ns, pm, nm in rows:
        print(f"{a:6} {ps:11.4f} {ns:5d} {pm:9.4f} {nm:5d} {pm - ps:9.4f}")
    pts = [(n, p) for _, ps, ns, pm, nm in rows for n, p in ((ns, ps), (nm, pm))]
    allp = [p for _, p in pts]
    mean = sum(allp) / len(allp)
    sd = (sum((p - mean) ** 2 for p in allp) / len(allp)) ** 0.5
    print(f"pooled seed prologue: n={len(allp)} mean={mean:.4f}s "
          f"sd={sd:.4f}s ({100 * sd / mean:.2f}%) "
          f"range={min(allp):.4f}..{max(allp):.4f}")

    # The residual is not noise: it scales with round count, so the harness
    # clock carries a per-round parent cost that in-session round_us omits.
    nbar = sum(n for n, _ in pts) / len(pts)
    pbar = mean
    sxx = sum((n - nbar) ** 2 for n, _ in pts)
    c = sum((n - nbar) * (p - pbar) for n, p in pts) / sxx
    p0 = pbar - c * nbar
    resid = [p - (p0 + c * n) for n, p in pts]
    rsd = (sum(r * r for r in resid) / len(resid)) ** 0.5
    print(f"parent-clock fit P_est = P0 + c*rounds: P0={p0:.4f}s "
          f"c={1e6 * c:.1f}us/round resid_sd={1e3 * rsd:.1f}ms "
          f"({100 * rsd / p0:.2f}% of P0)")


if __name__ == "__main__":
    main()
