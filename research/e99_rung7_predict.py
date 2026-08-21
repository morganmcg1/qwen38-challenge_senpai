#!/usr/bin/env python3
"""E99 rung 7: pre-registered replay prediction for the clamp depth.

Replays recorded arm-off rounds under the margin gate at a fixed threshold and
a varying clamp depth, and prices each depth on the ranked M5 curve. The round
sequence is held fixed, so every number here is a REPLAY BOUND and not a
measurement. It exists to fix an ordering prediction in writing before the
measured legs land.

usage:
  research/e99_rung7_predict.py --legs e99r5c8a1 e99r5c8a2 --threshold 9.4375
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e99_oracle import load_leg, ranked_round_us  # noqa: E402


def streams_per_round(width: int) -> int:
    groups = -(-width // 4)
    per_group = -(-width // groups)
    return -(-width // per_group)


def no_step_round_us(width: int) -> float:
    """The ranked G=2 line extrapolated over every width.

    Not a machine. It is the counterfactual cost curve in which drafting is
    priced but the weight-stream boundary does not exist, so it answers
    whether the measured depth optimum NEEDS the boundary to sit where it
    sits.
    """
    return 16_943.2 + 7_233.0 * width


CURVES = dict(ranked=ranked_round_us, nostep=no_step_round_us)


def replay(rounds, threshold: float, clamp_depth: int, curve) -> dict:
    cost = tokens = 0.0
    clamped = lost = 0
    for record in rounds:
        depth = record.depth
        if record.margin <= threshold:
            depth = min(depth, clamp_depth)
            if depth < record.depth:
                clamped += 1
                lost += min(record.accepted, record.depth) - \
                    min(record.accepted, depth)
        cost += curve(depth + 1)
        tokens += min(record.accepted, depth) + 1
    return dict(clamp_depth=clamp_depth, us_per_token=cost / tokens,
                clamped=clamped, tokens_given_up=lost, rounds=len(rounds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--legs', nargs='+', required=True)
    parser.add_argument('--threshold', type=float, default=9.4375)
    parser.add_argument('--depths', nargs='*', type=int,
                        default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--curve', default='ranked', choices=sorted(CURVES))
    parser.add_argument('--out-root', default='research/out')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    curve = CURVES[args.curve]
    root = Path(args.out_root)
    legs = []
    for tag in args.legs:
        info, rounds = load_leg(tag, root)
        actual_cost = sum(curve(r.depth + 1) for r in rounds)
        actual_tokens = sum(r.accepted + 1 for r in rounds)
        legs.append(dict(
            leg=tag, offered_cap=info['offered_cap'], rounds=len(rounds),
            mean_width=statistics.fmean([r.depth + 1 for r in rounds]),
            actual_us_per_token=actual_cost / actual_tokens,
            eligible=sum(1 for r in rounds if r.margin <= args.threshold),
            depths=[replay(rounds, args.threshold, d, curve)
                    for d in args.depths]))

    print(f"{args.curve}-curve replay bound, threshold {args.threshold}")
    print(f'{"leg":12s}{"cap":>4s}{"rounds":>7s}{"width":>7s}'
          f'{"actual":>10s}' + ''.join(f'{"d" + str(d):>10s}'
                                       for d in args.depths))
    for leg in legs:
        print(f'{leg["leg"]:12s}{leg["offered_cap"]:4d}{leg["rounds"]:7d}'
              f'{leg["mean_width"]:7.3f}{leg["actual_us_per_token"]:10.1f}'
              + ''.join(f'{row["us_per_token"]:10.1f}'
                        for row in leg['depths']))
        print(f'{"":12s}{"":4s}{"":7s}{"gain %":>7s}{"":10s}'
              + ''.join(
                  f'{100.0 * (leg["actual_us_per_token"] - row["us_per_token"]) / leg["actual_us_per_token"]:+10.3f}'
                  for row in leg['depths']))
        print(f'{"":12s}{"clamped":>18s}{"":7s}{"":10s}'
              + ''.join(f'{row["clamped"]:10d}' for row in leg['depths']))
        print(f'{"":12s}{"tokens lost":>18s}{"":7s}{"":10s}'
              + ''.join(f'{row["tokens_given_up"]:10d}'
                        for row in leg['depths']))

    pooled = {}
    for index, depth in enumerate(args.depths):
        gains = [100.0 * (leg['actual_us_per_token']
                          - leg['depths'][index]['us_per_token'])
                 / leg['actual_us_per_token'] for leg in legs]
        pooled[depth] = statistics.fmean(gains)
    print()
    print('pooled replay gain by clamp depth, % on the ranked curve')
    for depth, gain in pooled.items():
        width = depth + 1
        print(f'  depth {depth}  M {width}  G {streams_per_round(width)}'
              f'  {gain:+7.3f} %')

    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(threshold=args.threshold, legs=legs, pooled=pooled),
            indent=2) + '\n')


if __name__ == '__main__':
    main()
