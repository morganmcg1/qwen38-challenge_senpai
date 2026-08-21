#!/usr/bin/env python3
"""E94: the five-constant round-cost model, and what it says about a leg.

usage:
  research/e94_model.py [--legs research/e94-artifacts/rung1.json]

Model, from E95 rung 2 (verify pass) and E93 (head chain):

    verify_us(M) = 10_920 + 27_377 * G + 10_268 * M
    G            = ceil(M / IPG),  IPG = ceil(M / ceil(M / 4))
    head_us(d)   = 2_560 + 2_226.5 * (d - 1)      for d >= 1
    round_us(d)  = verify_us(d + 1) + head_us(d)

With `--legs` the script also prints, for every leg, the modelled round cost
beside the measured median round cost at each chosen depth, and the modelled
cost per token of the leg's own depth mixture against the depth-3 and depth-7
alternatives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

VERIFY_CONST = 10_920.0
VERIFY_PER_GROUP = 27_377.0
VERIFY_PER_ROW = 10_268.0
HEAD_CONST = 2_560.0
HEAD_PER_STEP = 2_226.5


def groups(width: int) -> int:
    ipg = math.ceil(width / math.ceil(width / 4))
    return math.ceil(width / ipg)


def verify_us(width: int) -> float:
    return VERIFY_CONST + VERIFY_PER_GROUP * groups(width) \
        + VERIFY_PER_ROW * width


def head_us(depth: int) -> float:
    return 0.0 if depth <= 0 else HEAD_CONST + HEAD_PER_STEP * (depth - 1)


def round_us(depth: int) -> float:
    return verify_us(depth + 1) + head_us(depth)


def expected_tokens(depth: int, q: float) -> float:
    return sum(q ** i for i in range(depth + 1))


def cost_per_token(depth: int, q: float) -> float:
    return round_us(depth) / expected_tokens(depth, q)


def print_model() -> None:
    print("depth | width | G | verify_us | head_us | round_us")
    for depth in range(0, 9):
        width = depth + 1
        print(f"{depth:5d} | {width:5d} | {groups(width)} | "
              f"{verify_us(width):9.0f} | {head_us(depth):7.0f} | "
              f"{round_us(depth):8.0f}")
    print()
    print("marginal round cost of the step into verify width M")
    for width in range(2, 10):
        print(f"  M={width}: {round_us(width - 1) - round_us(width - 2):9.0f} us")
    print()
    print("modelled cost per accepted token, flat q, cap 7")
    header = "    q   | " + " | ".join(f"d={d}" for d in range(1, 8)) + " | argmin"
    print(header)
    for q in [0.90, 0.94, 0.9551, 0.966, 0.9728, 0.98, 1.0]:
        costs = {d: cost_per_token(d, q) for d in range(1, 8)}
        best = min(costs, key=costs.get)
        print(f"  {q:.4f} | " + " | ".join(f"{costs[d]:6.0f}" for d in range(1, 8))
              + f" | {best}")


def print_legs(path: str) -> None:
    doc = json.loads(Path(path).read_text())
    for leg in doc["legs"]:
        meta = leg["meta"]
        print()
        print(f"=== {leg['tag']}  arm={meta.get('e94_arm')} "
              f"cap={meta.get('e94_cap')} ===")
        print("depth | rounds | frac  | measured_median_us | modelled_us | "
              "ratio | mean_acc")
        for depth, cell in sorted(leg["depth_histogram"].items(),
                                  key=lambda kv: int(kv[0])):
            depth = int(depth)
            modelled = round_us(depth)
            print(f"{depth:5d} | {cell['rounds']:6d} | {cell['fraction']:.3f} | "
                  f"{cell['median_round_us']:18.0f} | {modelled:11.0f} | "
                  f"{cell['median_round_us'] / modelled:5.3f} | "
                  f"{cell['mean_accepted']:.3f}")
        # What the same acceptance mixture would cost if every round at depth 4
        # ran at depth 3 instead, holding the per-round accepted counts of the
        # measured depth-3 rounds.
        hist = {int(d): c for d, c in leg["depth_histogram"].items()}
        total_rounds = sum(c["rounds"] for c in hist.values())
        tokens = sum(c["tokens_emitted"] for c in hist.values())
        modelled_us = sum(round_us(d) * c["rounds"] for d, c in hist.items())
        print(f"rounds={total_rounds} tokens={tokens} "
              f"measured_us_per_token={leg['round_us_per_token']:.0f} "
              f"modelled_us_per_token={modelled_us / tokens:.0f}")
        share4 = hist.get(4, {}).get("fraction", 0.0)
        print(f"depth-4 share of rounds = {share4:.4f}, "
              f"of round time = {hist.get(4, {}).get('round_us_share', 0.0):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs")
    args = ap.parse_args()
    print_model()
    if args.legs:
        print_legs(args.legs)


if __name__ == "__main__":
    main()
