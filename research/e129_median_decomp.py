#!/usr/bin/env python3
"""Decompose the published median into the part our code controls and the rest.

    python3 research/e129_median_decomp.py --ours 0c6191b7 --crown 48423d09

The published score is the median of eight per-prompt ratios
``serial_p / mtp_p``. Only ``mtp_p`` responds to candidate code. This script
asks how much of a published gap between two receipts comes from the candidate
leg and how much comes from the serial leg and from which prompts land in the
central pair.

harness=ranked.
"""

from __future__ import annotations

import argparse
import statistics

from e129_prereg import ORDER, leg_means, load, pick
from e129_state_scan import serial_means


def ratios(row: dict) -> dict[str, float]:
    return {
        e_name: s / m
        for e_name, (s, m) in (
            (n, (serial_means(row)[n], leg_means(row)[n])) for n in ORDER
        )
    }


def published(rs: dict[str, float]) -> tuple[float, list[tuple[str, float]]]:
    order = sorted(rs.items(), key=lambda kv: kv[1])
    return (order[3][1] + order[4][1]) / 2.0, order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--crown", required=True)
    args = ap.parse_args()

    by = load()
    aid, a = pick(by, args.ours)
    bid, b = pick(by, args.crown)

    print("harness=ranked")
    for tag, sid, row in (("ours", aid, a), ("crown", bid, b)):
        rs = ratios(row)
        med, order = published(rs)
        print("%-5s %-9s %-14s published %.6f  reconstructed median %.6f"
              % (tag, sid, (row.get("solverUsername") or "")[:14],
                 row["officialScore"], med))
        print("      central pair: %s %.4f and %s %.4f"
              % (order[3][0], order[3][1], order[4][0], order[4][1]))

    ra, rb = ratios(a), ratios(b)
    ma, mb = leg_means(a), leg_means(b)
    sa, sb = serial_means(a), serial_means(b)

    print()
    print("per prompt, ours against the crown, positive means ours is better")
    print("%-9s %9s %9s %9s %9s %9s"
          % ("prompt", "ratio us", "ratio cr", "ratio %", "cand %", "serial %"))
    for n in sorted(ORDER, key=lambda n: ra[n]):
        print("%-9s %9.4f %9.4f %+9.3f %+9.3f %+9.3f"
              % (n, ra[n], rb[n], (ra[n] - rb[n]) / rb[n] * 100,
                 (mb[n] - ma[n]) / mb[n] * 100,
                 (sa[n] - sb[n]) / sb[n] * 100))

    print()
    ca = sum(ma[n] for n in ORDER) / 8
    cb = sum(mb[n] for n in ORDER) / 8
    print("candidate leg, unweighted: ours %.6f  crown %.6f  ours is %+.3f %% faster"
          % (ca, cb, (cb - ca) / cb * 100))
    print("published:                 ours %.6f  crown %.6f  ours is %+.3f %%"
          % (a["officialScore"], b["officialScore"],
             (a["officialScore"] - b["officialScore"]) / b["officialScore"] * 100))
    print()
    print("counterfactual: hold the serial leg and the prompt set fixed and give")
    print("our candidate leg the crown's per-prompt shape.")
    scale = ca / cb
    cf = {n: sa[n] / (mb[n] * scale) for n in ORDER}
    med_cf, _ = published(cf)
    print("  our published median if our leg were the crown's shape: %.6f" % med_cf)
    print("  actually published:                                     %.6f"
          % a["officialScore"])
    print()
    print("spread of the eight per-prompt ratios")
    for tag, rs in (("ours", ra), ("crown", rb)):
        vals = [rs[n] for n in ORDER]
        print("  %-5s min %.4f  median %.4f  max %.4f  sd %.4f"
              % (tag, min(vals), statistics.median(vals), max(vals),
                 statistics.stdev(vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
