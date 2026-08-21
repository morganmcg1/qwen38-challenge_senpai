#!/usr/bin/env python3
"""E99 rungs 6 and 7: report the session null, the clamp depth and the width
dependence of the threshold.

Three questions, three blocks.

  block N  What is the wall-clock null, and does this build reproduce the
           submitted build's shipped round sequence? The ranked figure is a
           deterministic function of that sequence, so its own spread must be
           exactly zero and the wall clock's spread is the whole error bar.
  block D  Rung 7. Measured gain by clamp depth against the PRE-REGISTERED
           replay prediction, on the real ranked curve and on a counterfactual
           curve with no weight-stream boundary.
  block W  Does the threshold optimum move with the verify width?

usage:
  research/e99_rung67_report.py [--out research/e99-artifacts/rung67.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e99_rung5_price import leg_row  # noqa: E402

# Pre-registered in PR #101 comment 5371093526, before any leg of this session
# ran. Replay bound over the recorded cap-8 arm-off rounds at t = 9.4375.
PREDICTION = {
    1: dict(ranked=1.116, nostep=2.741),
    2: dict(ranked=2.501, nostep=2.893),
    3: dict(ranked=3.228, nostep=2.450),
    4: dict(ranked=1.985, nostep=1.981),
}

# The submitted build's sequences, for the cross-build reproduction check.
REFERENCE_SEQUENCE = {
    'shipped': 'e99r5c8b1',
    'off': 'e99r5c8a1',
}

BLOCK_N_OFF = ['e99r6n1o', 'e99r6n4o', 'e99r6n5o', 'e99r6n8o']
BLOCK_N_SHIPPED = ['e99r6n2s', 'e99r6n3s', 'e99r6n6s', 'e99r6n7s']
BLOCK_D = [('e99r7d1', 1), ('e99r7d2', 2), ('e99r7d4', 4)]
BLOCK_W = [('e99r6w8a', 8, 12.5), ('e99r6w8b', 8, 14.0),
           ('e99r6w5a', 5, 12.5), ('e99r6w5b', 5, 14.0)]


ROLE = dict(
    [(tag, 'null-off') for tag in BLOCK_N_OFF]
    + [(tag, 'null-shipped') for tag in BLOCK_N_SHIPPED]
    + [(tag, 'clamp-depth') for tag, _ in BLOCK_D]
    + [(tag, 'width-threshold') for tag, _, _ in BLOCK_W]
    + [('e99r6w5o', 'width-baseline')])


def meta_field(tag: str, root: Path, key: str) -> str:
    for line in (root / tag / 'meta.txt').read_text().splitlines():
        if line.startswith(f'{key}='):
            return line.split('=', 1)[1]
    return ''


def sequence_sha(tag: str, root: Path) -> str:
    return meta_field(tag, root, 'e99_round_sequence_sha256')


def reference_sequence_sha(tag: str, root: Path) -> str:
    import hashlib
    import re
    text = (root / tag / 'trace.txt').read_text()
    found = re.findall(r'round=\d+ d=\d+ acc=\d+', text)
    return hashlib.sha256(
        ''.join(line + '\n' for line in found).encode()).hexdigest()


def spread(rows, field: str) -> dict:
    values = [row[field] for row in rows]
    mean = statistics.fmean(values)
    return dict(mean=mean, lo=min(values), hi=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
                range_pct=100.0 * (max(values) - min(values)) / mean,
                distinct=len(set(round(v, 9) for v in values)))


def gain(base: float, value: float) -> float:
    return 100.0 * (base - value) / base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', default='research/out')
    parser.add_argument('--out', default='')
    args = parser.parse_args()
    root = Path(args.out_root)

    have = [tag for tag in
            BLOCK_N_OFF + BLOCK_N_SHIPPED + [t for t, _ in BLOCK_D]
            + [t for t, _, _ in BLOCK_W] + ['e99r6w5o']
            if (root / tag / 'score.json').exists()]
    rows = {tag: leg_row(tag, root) for tag in have}
    for tag in have:
        rows[tag]['sequence_sha256'] = sequence_sha(tag, root)
        rows[tag]['gate_depth'] = int(meta_field(tag, root, 'e99_gate_depth'))
        rows[tag]['role'] = ROLE[tag]

    print('=' * 110)
    print('ALL LEGS')
    print('=' * 110)
    print(f'{"leg":10s}{"gate":8s}{"t":>9s}{"cap":>4s}{"rounds":>7s}'
          f'{"fire":>5s}{"width":>7s}{"s/token":>10s}{"ratio":>8s}'
          f'{"ranked":>9s}{"exact":>7s}{"inC":>6s}{"outC":>6s}{"seq":>10s}')
    for tag in have:
        row = rows[tag]
        print(f'{tag:10s}{row["gate"]:8s}{row["threshold"]:9.4f}'
              f'{row["offered_cap"]:4d}{row["rounds"]:7d}'
              f'{row["fired_rounds"]:5d}{row["mean_width"]:7.3f}'
              f'{row["mtp_seconds_per_token"]:10.6f}{row["local_ratio"]:8.4f}'
              f'{row["ranked_us_per_token"]:9.1f}'
              f'{str(row["all_tokens_matched"]):>7s}'
              f'{row["entry_c"]:6.1f}{row["exit_c"]:6.1f}'
              f'{row["sequence_sha256"][:8]:>10s}')

    report = dict(legs={t: rows[t] for t in have})

    off = [rows[t] for t in BLOCK_N_OFF if t in rows]
    ship = [rows[t] for t in BLOCK_N_SHIPPED if t in rows]
    if off and ship:
        print()
        print('=' * 110)
        print('BLOCK N - THE SESSION NULL')
        print('=' * 110)
        block_n = {}
        for name, group in (('off', off), ('shipped', ship)):
            for field in ('mtp_seconds_per_token', 'ranked_us_per_token',
                          'local_ratio', 'serial_seconds_per_token'):
                info = spread(group, field)
                block_n[f'{name}.{field}'] = info
                print(f'{name:8s}{field:26s}mean {info["mean"]:12.6f}  '
                      f'range {info["range_pct"]:6.3f} %  '
                      f'distinct values {info["distinct"]} of {len(group)}')
        contrast = {}
        for field in ('mtp_seconds_per_token', 'ranked_us_per_token',
                      'local_us_per_token', 'effective_mean_draft_len',
                      'accepted_draft_rate', 'serial_seconds_per_token'):
            base = statistics.fmean([r[field] for r in off])
            cand = statistics.fmean([r[field] for r in ship])
            contrast[field] = dict(off=base, on=cand,
                                   delta_pct=gain(base, cand))
            print(f'{"contrast":8s}{field:26s}off {base:12.6f}  '
                  f'on {cand:12.6f}  {gain(base, cand):+7.3f} %')
        report['block_n'] = dict(spread=block_n, contrast=contrast)

        print()
        print('cross-build reproduction of the submitted round sequence')
        for name, group in (('off', off), ('shipped', ship)):
            reference = reference_sequence_sha(
                REFERENCE_SEQUENCE[name], root)
            same = all(row['sequence_sha256'] == reference for row in group)
            print(f'  {name:8s}reference {REFERENCE_SEQUENCE[name]} '
                  f'{reference[:16]}  all {len(group)} legs identical: {same}')
            report.setdefault('reproduction', {})[name] = dict(
                reference_leg=REFERENCE_SEQUENCE[name],
                reference_sha256=reference, identical=same,
                legs=[row['leg'] for row in group])

    if off:
        base_ranked = statistics.fmean([r['ranked_us_per_token'] for r in off])
        base_wall = statistics.fmean(
            [r['mtp_seconds_per_token'] for r in off])

        depth_rows = [(rows[t], d) for t, d in BLOCK_D if t in rows]
        if ship:
            depth_rows.append((ship[0], 3))
        if depth_rows:
            print()
            print('=' * 110)
            print('BLOCK D - RUNG 7, CLAMP DEPTH AGAINST THE PRE-REGISTERED '
                  'PREDICTION')
            print('=' * 110)
            print(f'{"depth":>6s}{"M":>3s}{"G":>3s}{"fired":>7s}{"width":>8s}'
                  f'{"ranked":>10s}{"measured":>10s}{"pred G":>9s}'
                  f'{"pred noG":>10s}{"err G":>8s}{"err noG":>9s}')
            block_d = {}
            for row, depth in sorted(depth_rows, key=lambda x: x[1]):
                measured = gain(base_ranked, row['ranked_us_per_token'])
                pred = PREDICTION.get(depth, {})
                width = depth + 1
                groups = -(-width // 4)
                per_group = -(-width // groups)
                streams = -(-width // per_group)
                block_d[depth] = dict(
                    leg=row['leg'], measured_gain_pct=measured,
                    predicted_ranked=pred.get('ranked'),
                    predicted_nostep=pred.get('nostep'),
                    wall_gain_pct=gain(base_wall,
                                       row['mtp_seconds_per_token']),
                    fired=row['fired_rounds'], width=row['mean_width'])
                print(f'{depth:6d}{width:3d}{streams:3d}'
                      f'{row["fired_rounds"]:7d}{row["mean_width"]:8.3f}'
                      f'{row["ranked_us_per_token"]:10.1f}{measured:+10.3f}'
                      f'{pred.get("ranked", float("nan")):+9.3f}'
                      f'{pred.get("nostep", float("nan")):+10.3f}'
                      f'{measured - pred.get("ranked", float("nan")):+8.3f}'
                      f'{measured - pred.get("nostep", float("nan")):+9.3f}')
            if 2 in block_d and 3 in block_d:
                statistic = block_d[2]['measured_gain_pct'] \
                    - block_d[3]['measured_gain_pct']
                verdict = ('stream boundary' if statistic < 0
                           else 'drafting cost only')
                print()
                print(f'DISCRIMINATING STATISTIC  gain(d2) - gain(d3) = '
                      f'{statistic:+.3f} pp')
                print(f'  pre-registered: stream boundary -0.727 pp, '
                      f'drafting cost only +0.443 pp')
                print(f'  VERDICT: {verdict}')
                report['discriminator'] = dict(
                    statistic_pp=statistic, verdict=verdict,
                    predicted_boundary_pp=-0.727,
                    predicted_nostep_pp=+0.443)
            report['block_d'] = block_d

        width_rows = [(rows[t], cap, t_) for t, cap, t_ in BLOCK_W
                      if t in rows]
        if width_rows:
            print()
            print('=' * 110)
            print('BLOCK W - DOES THE THRESHOLD OPTIMUM MOVE WITH WIDTH?')
            print('=' * 110)
            base5 = (rows['e99r6w5o']['ranked_us_per_token']
                     if 'e99r6w5o' in rows else None)
            bases = {8: base_ranked, 5: base5}
            block_w = {}
            print(f'{"cap":>4s}{"t":>10s}{"fired":>7s}{"width":>8s}'
                  f'{"ranked":>10s}{"gain %":>9s}')
            for row, cap, threshold in width_rows:
                reference = bases.get(cap)
                value = (gain(reference, row['ranked_us_per_token'])
                         if reference else float('nan'))
                block_w[f'c{cap}t{threshold}'] = dict(
                    leg=row['leg'], cap=cap, threshold=threshold,
                    ranked_us_per_token=row['ranked_us_per_token'],
                    gain_pct=value, fired=row['fired_rounds'])
                print(f'{cap:4d}{threshold:10.4f}{row["fired_rounds"]:7d}'
                      f'{row["mean_width"]:8.3f}'
                      f'{row["ranked_us_per_token"]:10.1f}{value:+9.3f}')
            report['block_w'] = block_w

    exact = all(rows[t]['all_tokens_matched']
                and rows[t]['residual_divergence_count'] == 0 for t in have)
    floors = sum(rows[t]['dram_floor_violations'] for t in have)
    print()
    print(f'legs analysed: {len(have)}')
    print(f'exactness green on every leg: {exact}')
    print(f'rounds under their own DRAM floor: {floors}')
    report['exact'] = exact
    report['dram_floor_violations'] = floors

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
