#!/usr/bin/env python3
"""E86: decide whether the ladder effect lives in the median round or the tail.

usage: research/e86_tail.py PREFIX [PREFIX ...] [--ref default]

The paired median and the leg total disagree by ~4x on this experiment. Only
one of them can drive a promotion decision, because the score integrates TOTAL
decode time, not a robust central estimate.

Two readings are possible:

  a. the tail is random OS jitter that happened to land on the reference legs,
     in which case the median is right and the total is contaminated;
  b. the tail is caused by the rung set, in which case the total is right and
     the median discards the real effect.

They are distinguishable. Random jitter does not reproduce across independent
sessions; a caused effect does. This script prints, per arm, the median round,
the trimmed mean, the full mean, the excess mass (mean - median) and the
fraction of rounds above a fixed slow threshold, for every session given. If
the excess mass reproduces per arm across sessions, reading (b) holds.
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


def trimmed(xs: list[float], frac: float = 0.10) -> float:
    s = sorted(xs)
    k = int(len(s) * frac)
    return st.mean(s[k:len(s) - k]) if len(s) - 2 * k > 0 else st.mean(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--ref", default="default")
    args = ap.parse_args()

    for prefix in args.prefixes:
        tags = sorted(p.name for p in OUT.iterdir()
                      if p.name.startswith(prefix + "-") and (p / "trace.txt").exists())
        legs = {}
        for t in tags:
            legs.setdefault(t[len(prefix) + 1:].rpartition("-")[0], []).append((t, rounds(t)))

        print(f"\n=== {prefix} ===")
        print(f"{'leg':<24}{'median':>9}{'trim10':>9}{'mean':>9}{'excess':>9}"
              f"{'p90':>9}{'max':>9}{'>+5ms':>7}{'total s':>9}")
        per_arm = {}
        for arm in sorted(legs):
            for tag, rs in legs[arm]:
                v = [r["round_us"] for r in rs]
                med, mn = st.median(v), st.mean(v)
                slow = sum(1 for x in v if x > med + 5000)
                print(f"{tag:<24}{med:>9.0f}{trimmed(v):>9.0f}{mn:>9.0f}"
                      f"{mn - med:>+9.0f}{sorted(v)[int(.9 * len(v))]:>9.0f}"
                      f"{max(v):>9.0f}{slow:>7}{sum(v) / 1e6:>9.4f}")
                per_arm.setdefault(arm, []).append(mn - med)
        print(f"\n{'arm':<12}{'excess mass (mean-median), us/round':>38}")
        for arm in sorted(per_arm):
            print(f"{arm:<12}{str([round(x) for x in per_arm[arm]]):>38}")


if __name__ == "__main__":
    main()
