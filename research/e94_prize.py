#!/usr/bin/env python3
"""E94: the session null, and the prize a depth-4 guard can win in each leg.

usage:
  research/e94_prize.py research/e94-artifacts/rung1.json

The null comes from same-cap leg pairs of the SAME arm, which differ only in
their position in the session. The prize comes from each leg's own measured
depth-4 mass, the five-constant round-cost model, and the flat acceptance rate
that reproduces the leg's measured depth-4 token yield.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e94_model import round_us  # noqa: E402


def flat_q(mean_accepted: float, positions: int) -> float:
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(mid ** i for i in range(1, positions + 1)) < mean_accepted:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    doc = json.loads(Path(sys.argv[1]).read_text())
    legs = {leg["tag"]: leg for leg in doc["legs"]}
    caps = sorted({int(leg["meta"]["e94_cap"]) for leg in doc["legs"]})

    print("cap | a s/tok | b s/tok | pair % | a round us/tok | b round us/tok |"
          " round pair % | non-round overhead ms/tok")
    spreads = []
    for cap in caps:
        a, b = legs.get(f"e94c{cap}a"), legs.get(f"e94c{cap}b")
        if not (a and b):
            continue
        sa = a["score"]["mtp_seconds_per_token"]
        sb = b["score"]["mtp_seconds_per_token"]
        ra, rb = a["round_us_per_token"], b["round_us_per_token"]
        spread = abs(sa - sb) / ((sa + sb) / 2) * 100
        spreads.append(spread)
        print(f"{cap:3d} | {sa:.6f} | {sb:.6f} | {spread:5.2f} | {ra:10.0f} | "
              f"{rb:10.0f} | {abs(ra - rb) / ((ra + rb) / 2) * 100:5.2f} | "
              f"{(sa * 1e6 - ra) / 1000:.2f}, {(sb * 1e6 - rb) / 1000:.2f}")
    print(f"\nsame-arm pair spreads %: "
          f"{[round(s, 2) for s in spreads]}  max={max(spreads):.2f}  "
          f"rms={statistics.fmean(s * s for s in spreads) ** 0.5:.2f}")

    print("\nprize a depth-4 guard can win, from each leg's own depth-4 mass")
    print("cap | d4 rounds | d4 tokens | token share | q | us/token d4 -> d3 |"
          " gain on those tokens % | round-busy % | leg s/tok %")
    for cap in caps:
        leg = legs.get(f"e94c{cap}a")
        hist = {int(k): v for k, v in leg["depth_histogram"].items()}
        cell = hist.get(4)
        if not cell:
            continue
        q = flat_q(cell["mean_accepted"], 4)
        yield4 = cell["mean_accepted"] + 1.0
        yield3 = sum(q ** i for i in range(0, 4))
        per4, per3 = round_us(4) / yield4, round_us(3) / yield3
        gain = (per4 - per3) / per4
        share = cell["tokens_emitted"] / leg["tokens_emitted"]
        round_gain = share * gain * 100
        overhead = (leg["score"]["mtp_seconds_per_token"] * 1e6
                    - leg["round_us_per_token"])
        leg_gain = round_gain * leg["round_us_per_token"] / (
            leg["round_us_per_token"] + overhead)
        print(f"{cap:3d} | {cell['rounds']:9d} | {cell['tokens_emitted']:9d} | "
              f"{share:11.4f} | {q:.4f} | {per4:.0f} -> {per3:.0f} | "
              f"{gain * 100:22.1f} | {round_gain:12.2f} | {leg_gain:11.2f}")


if __name__ == "__main__":
    main()
