#!/usr/bin/env python3
"""E86: resolve the paired ladder effect by draft width.

usage: research/e86_bywidth.py PREFIX [--ref default] [--refleg TAG]

The paired MEDIAN and the paired MEAN disagree by ~5x on this experiment. That
is not a contradiction and not an outlier problem: it is the signature of an
effect that varies with the round's work.

The score integrates TOTAL decode time, so the mean of paired differences is
the estimator that maps to the score (total = n x mean). The median is the
estimator that resists outliers. When they disagree the honest move is to show
the effect as a function of the round's draft width and let the reader see
which rounds carry it, rather than picking the statistic that flatters the
result.

`--refleg` pins one reference leg by tag, for a session where one reference
leg is anomalous and pooling both would corrupt every comparison.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")


def rounds(tag: str) -> list[dict]:
    out = []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        rec = {"round": int(m.group(1)), "d": int(m.group(2)), "acc": int(m.group(3))}
        rec.update({k: float(v) for k, v in KV_RE.findall(m.group(4))})
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--ref", default="default")
    ap.add_argument("--refleg", default=None)
    args = ap.parse_args()

    tags = sorted(p.name for p in OUT.iterdir()
                  if p.name.startswith(args.prefix + "-") and (p / "trace.txt").exists())
    legs = {}
    for t in tags:
        legs.setdefault(t[len(args.prefix) + 1:].rpartition("-")[0], []).append((t, rounds(t)))

    refs = legs[args.ref]
    if args.refleg:
        refs = [rl for rl in refs if rl[0] == args.refleg]
        if not refs:
            raise SystemExit(f"no reference leg named {args.refleg}")
    print(f"reference legs: {[t for t, _ in refs]}")

    widths = sorted({r["d"] for _, rs in refs for r in rs})
    counts = {w: sum(1 for r in refs[0][1] if r["d"] == w) for w in widths}
    print(f"round count by draft width: "
          + ", ".join(f"d={w}:{counts[w]}" for w in widths))

    print(f"\npaired Δ round_us vs `{args.ref}`, per draft width "
          f"(mean; negative is faster)")
    print(f"{'arm':<10}" + "".join(f"{'d=' + str(w):>9}" for w in widths)
          + f"{'ALL mean':>10}{'ALL med':>9}{'total ms':>10}")
    for arm in sorted(legs):
        if arm == args.ref:
            continue
        byw = {w: [] for w in widths}
        allv = []
        for _, la in legs[arm]:
            for _, lr in refs:
                for a, b in zip(la, lr):
                    d = a["round_us"] - b["round_us"]
                    byw[a["d"]].append(d)
                    allv.append(d)
        n_legs = len(legs[arm]) * len(refs)
        print(f"{arm:<10}"
              + "".join(f"{st.mean(byw[w]):>+9.0f}" if byw[w] else f"{'-':>9}"
                        for w in widths)
              + f"{st.mean(allv):>+10.0f}{st.median(allv):>+9.0f}"
              + f"{sum(allv) / n_legs / 1000:>+10.1f}")

    print("\nsame, but for the verify pipeline only (verify_build + eval_wall)")
    print(f"{'arm':<10}" + "".join(f"{'d=' + str(w):>9}" for w in widths)
          + f"{'ALL mean':>10}")
    for arm in sorted(legs):
        if arm == args.ref:
            continue
        byw = {w: [] for w in widths}
        allv = []
        for _, la in legs[arm]:
            for _, lr in refs:
                for a, b in zip(la, lr):
                    d = ((a["verify_build_us"] + a["eval_wall_us"])
                         - (b["verify_build_us"] + b["eval_wall_us"]))
                    byw[a["d"]].append(d)
                    allv.append(d)
        print(f"{arm:<10}"
              + "".join(f"{st.mean(byw[w]):>+9.0f}" if byw[w] else f"{'-':>9}"
                        for w in widths)
              + f"{st.mean(allv):>+10.0f}")


if __name__ == "__main__":
    main()
