#!/usr/bin/env python3
"""Attribute the scored round-1 cold cost to a named draft sub-phase.

The r1probe arm splits draft_build into six sub-phases that tile the interval
exactly. Round 1 is the only systematic cold event the census found, and the
other rounds at the same draft depth are its matched control, so the excess per
sub-phase names the statement that pays the cost.
"""

from __future__ import annotations

import argparse
import json
import statistics as st

SUB = ["d_pre", "d_flush", "d_head1", "d_submit1", "d_chain", "d_submit2"]


def report(path):
    session = json.load(open(path))["sessions"][0]
    rows = session["round_table"]
    if "d_submit2" not in rows[0]:
        raise SystemExit(f"{path}: no sub-phase fields; not an r1probe trace")

    depth = rows[0]["d"]
    peers = [r for r in rows if r["d"] == depth and r["round"] != rows[0]["round"]]
    first = rows[0]

    print(f"=== {path}")
    print(f"    round 1 is d={depth}; control is the other {len(peers)} "
          f"d={depth} rounds")
    print(f"    {'sub-phase':<12}{'round 1':>12}{'median peer':>14}{'excess':>12}")
    total = 0
    for key in SUB:
        peer = st.median([r[key] for r in peers])
        excess = first[key] - peer
        total += excess
        print(f"    {key:<12}{first[key] / 1000:>10.2f}ms"
              f"{peer / 1000:>12.2f}ms{excess / 1000:>10.2f}ms")
    peer_total = st.median([r["draft_build"] for r in peers])
    print(f"    {'draft_build':<12}{first['draft_build'] / 1000:>10.2f}ms"
          f"{peer_total / 1000:>12.2f}ms{total / 1000:>10.2f}ms")

    print("    steady state over all rounds:")
    for key in SUB:
        later = [r[key] for r in rows[1:]]
        print(f"      {key:<12} median {st.median(later) / 1000:7.3f}ms  "
              f"max {max(later) / 1000:8.3f}ms  round1 {first[key] / 1000:8.3f}ms")

    print("    top 5 rounds by d_submit2:")
    for r in sorted(rows, key=lambda r: -r["d_submit2"])[:5]:
        print(f"      rnd {r['round']:>3} d={r['d']} "
              f"d_submit2={r['d_submit2'] / 1000:8.3f}ms "
              f"d_submit1={r['d_submit1'] / 1000:7.3f}ms "
              f"d_chain={r['d_chain'] / 1000:7.3f}ms")

    leg = session["timed_leg_us"]
    rounds_us = sum(r["round_us"] for r in rows)
    totals = dict(session["segment_totals_us"])
    # begin() prefill and the tail are inside the timed leg but outside every
    # round, so they only appear as the residual.
    totals["begin+tail"] = leg - rounds_us
    print(f"    leg budget (timed leg {leg / 1e6:.3f}s, rounds "
          f"{100 * rounds_us / leg:.1f}% of it):")
    for key, value in sorted(totals.items(), key=lambda kv: -kv[1]):
        if 100 * value / leg < 0.05:
            continue
        share = f"{100 * value / leg:6.2f}% of leg"
        if key == "d_submit2":
            share += (f"  = {100 * value / totals['draft_build']:.1f}% of "
                      f"draft_build")
        print(f"      {key:<14}{value / 1e6:8.3f}s  {share}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census", nargs="+")
    args = ap.parse_args()
    for path in args.census:
        report(path)
        print()


if __name__ == "__main__":
    main()
