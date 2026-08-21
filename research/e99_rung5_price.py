#!/usr/bin/env python3
"""E99 rung 5: price the REALISED round sequence of the margin-gate legs.

The rung-2 to rung-4 numbers replay a fixed recorded round sequence, so they
are bounds: a policy that commits fewer tokens in one round changes the state
every later round sees. These legs run the policy, so their traces carry the
true sequence and the ranked price computed here is a measurement of the
mechanism rather than a bound on it.

Three things are reported for every leg:

  measured     seconds per token from the harness, plus the local
               serial-to-MTP ratio. A schedule change lives entirely in the
               candidate leg, so the local ratio is direct evidence here.
  ranked       sum_r C(d_r + 1) / sum_r (a_r + 1) on the ranked M5 curve,
               which is the number the campaign acts on.
  DRAM floor   every round's measured time against G * 52,792 us, the time
               14,412,349,440 B needs at the M4 Pro's 273 GB/s peak. A round
               under its own floor means the instrument is wrong.

usage:
  research/e99_rung5_price.py --off e99r5c8a1 e99r5c8a2 \
                              --on  e99r5c8b1 e99r5c8b2 \
                              [--sweep e99r5c8t1 e99r5c8t2 ...] \
                              [--replay-thresholds 8.25 9.4375 11.5625] \
                              [--out research/e99-artifacts/rung5.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e99_oracle import (  # noqa: E402
    MAX_DEPTH, load_leg, local_round_us, ranked_round_us)

TRANSFORMED_BYTES = 14_412_349_440
DRAM_PEAK_BYTES_PER_S = 273e9
MARGIN_GATE_DEPTH = 3


def streams_per_round(width: int) -> int:
    groups = -(-width // 4)
    per_group = -(-width // groups)
    return -(-width // per_group)


def dram_floor_us(width: int) -> float:
    return 1e6 * streams_per_round(width) * TRANSFORMED_BYTES \
        / DRAM_PEAK_BYTES_PER_S


def price(rounds, curve) -> float:
    cost = sum(curve(r.depth + 1) for r in rounds)
    tokens = sum(r.accepted + 1 for r in rounds)
    return cost / tokens


def replay(rounds, curve, threshold: float) -> dict:
    """Clamp on the recorded margin, holding the round sequence fixed."""
    cost = tokens = 0.0
    fired = 0
    for record in rounds:
        depth = record.depth
        if record.margin <= threshold:
            depth = min(depth, MARGIN_GATE_DEPTH)
            fired += depth < record.depth
        cost += curve(depth + 1)
        tokens += min(record.accepted, depth) + 1
    return dict(threshold=threshold, us_per_token=cost / tokens,
                clamped=fired, rounds=len(rounds))


def leg_row(tag: str, out_root: Path) -> dict:
    info, rounds = load_leg(tag, out_root)
    meta, score = info['meta'], info['score']
    widths = [r.depth + 1 for r in rounds]
    floor_violations = [
        dict(round=r.index, width=r.depth + 1, round_us=r.round_us,
             floor_us=dram_floor_us(r.depth + 1))
        for r in rounds if r.round_us < dram_floor_us(r.depth + 1)]
    return dict(
        leg=tag, gate=meta.get('e99_gate', 'off'),
        threshold=float(meta.get('e99_gate_threshold', 'nan')),
        offered_cap=info['offered_cap'], rounds=len(rounds),
        fired_rounds=sum(1 for r in rounds if r.fired),
        fired_share=statistics.fmean([1.0 if r.fired else 0.0
                                      for r in rounds]),
        clamped_rounds=sum(1 for r in rounds
                           if r.fired and r.depth <= MARGIN_GATE_DEPTH),
        mean_width=statistics.fmean(widths),
        g1_share=statistics.fmean([1.0 if streams_per_round(w) == 1 else 0.0
                                   for w in widths]),
        effective_mean_draft_len=score['effective_mean_draft_len'],
        accepted_draft_rate=score['accepted_draft_rate'],
        all_tokens_matched=score['all_tokens_matched'],
        residual_divergence_count=score['residual_divergence_count'],
        mtp_seconds_per_token=score['mtp_seconds_per_token'],
        serial_seconds_per_token=score['serial_seconds_per_token'],
        local_ratio=score['mtp_decode_speedup'],
        head_provenance_sha256=score['head_provenance_sha256'],
        ranked_us_per_token=price(rounds, ranked_round_us),
        local_us_per_token=price(rounds, local_round_us),
        measured_round_us=statistics.fmean([r.round_us for r in rounds]),
        min_round_us=min(r.round_us for r in rounds),
        dram_floor_violations=len(floor_violations),
        first_floor_violation=json.dumps(floor_violations[0])
        if floor_violations else '',
        entry_c=float(meta.get('gpu_temp_entry_c', 'nan')),
        exit_c=float(meta.get('gpu_temp_exit_c', 'nan')),
        worker_sha256=meta.get('worker_sha256', ''),
        base_sha=meta.get('base_sha', ''),
        cool_gate_passed_real_gate=meta.get('cool_gate_passed_real_gate'),
        gate_qualified_for_timing=meta.get('gate_qualified_for_timing'))


def contrast(off_rows, on_rows, field: str) -> dict:
    off = statistics.fmean([row[field] for row in off_rows])
    on = statistics.fmean([row[field] for row in on_rows])
    return dict(off=off, on=on, delta_pct=100.0 * (off - on) / off)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--off', nargs='+', required=True)
    parser.add_argument('--on', nargs='+', required=True)
    parser.add_argument('--sweep', nargs='*', default=[])
    parser.add_argument('--replay-thresholds', nargs='*', type=float,
                        default=[8.25, 9.1875, 9.4375, 10.4375, 11.5625])
    parser.add_argument('--out-root', default='research/out')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    root = Path(args.out_root)
    off_rows = [leg_row(tag, root) for tag in args.off]
    on_rows = [leg_row(tag, root) for tag in args.on]
    sweep_rows = [leg_row(tag, root) for tag in args.sweep]

    print('=' * 96)
    print('MEASURED LEGS')
    print('=' * 96)
    print(f'{"leg":12s}{"gate":6s}{"t":>9s}{"rounds":>7s}{"fire":>6s}'
          f'{"width":>7s}{"G1":>6s}{"s/token":>10s}{"ratio":>8s}'
          f'{"rankedus":>10s}{"exact":>7s}{"in C":>6s}{"out C":>7s}')
    for row in off_rows + on_rows + sweep_rows:
        print(f'{row["leg"]:12s}{row["gate"]:6s}{row["threshold"]:9.4f}'
              f'{row["rounds"]:7d}{row["fired_rounds"]:6d}'
              f'{row["mean_width"]:7.3f}{row["g1_share"]:6.2f}'
              f'{row["mtp_seconds_per_token"]:10.6f}{row["local_ratio"]:8.4f}'
              f'{row["ranked_us_per_token"]:10.1f}'
              f'{str(row["all_tokens_matched"]):>7s}'
              f'{row["entry_c"]:6.1f}{row["exit_c"]:7.1f}')

    exact = all(row['all_tokens_matched'] and
                row['residual_divergence_count'] == 0
                for row in off_rows + on_rows + sweep_rows)
    floors = sum(row['dram_floor_violations']
                 for row in off_rows + on_rows + sweep_rows)

    print()
    print('=' * 96)
    print('ABBA CONTRAST, ARM ON AGAINST ARM OFF')
    print('=' * 96)
    fields = ('ranked_us_per_token', 'mtp_seconds_per_token',
              'local_us_per_token', 'measured_round_us',
              'effective_mean_draft_len', 'accepted_draft_rate', 'local_ratio')
    contrasts = {}
    for field in fields:
        values = contrast(off_rows, on_rows, field)
        contrasts[field] = values
        print(f'{field:28s}off {values["off"]:12.6f}   '
              f'on {values["on"]:12.6f}   '
              f'{"lower is better" if "ratio" not in field else "higher is better"}'
              f'   {values["delta_pct"]:+7.3f} %')
    print()
    print(f'exactness green on every leg: {exact}')
    print(f'rounds under their own DRAM floor: {floors}')
    print(f'entry temperature spread off '
          f'{max(r["entry_c"] for r in off_rows) - min(r["entry_c"] for r in off_rows):.1f} C'
          f'   on '
          f'{max(r["entry_c"] for r in on_rows) - min(r["entry_c"] for r in on_rows):.1f} C')

    print()
    print('=' * 96)
    print('THRESHOLD SENSITIVITY')
    print('=' * 96)
    if sweep_rows:
        print('measured legs')
        base = statistics.fmean([r['ranked_us_per_token'] for r in off_rows])
        for row in sweep_rows:
            print(f'  t={row["threshold"]:8.4f}  fired {row["fired_rounds"]:4d}'
                  f'  ranked {row["ranked_us_per_token"]:9.1f}'
                  f'  {100.0 * (base - row["ranked_us_per_token"]) / base:+7.3f} %'
                  f'  s/token {row["mtp_seconds_per_token"]:.6f}')
    print('replay over the arm-off rounds, which holds the sequence fixed and '
          'is therefore a bound, not a measurement')
    _, off_rounds = load_leg(args.off[0], root)
    reference = price(off_rounds, ranked_round_us)
    replay_rows = []
    for threshold in args.replay_thresholds:
        row = replay(off_rounds, ranked_round_us, threshold)
        row['gain_pct'] = 100.0 * (reference - row['us_per_token']) / reference
        replay_rows.append(row)
        print(f'  t={threshold:8.4f}  clamped {row["clamped"]:4d}'
              f'  ranked {row["us_per_token"]:9.1f}  {row["gain_pct"]:+7.3f} %')

    report = dict(off=off_rows, on=on_rows, sweep=sweep_rows,
                  contrasts=contrasts, replay=replay_rows,
                  exactness_green=exact, dram_floor_violations=floors,
                  reference_leg=args.off[0])
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
