#!/usr/bin/env python3
"""E99 rung 1: the two identity checks that license the offline replay.

  1. RECORDER NEUTRALITY. The cap-5 legs run with `MLX_QWEN_MTP_TRACE=1` and
     without it. `effective_mean_draft_len` and `accepted_draft_rate` must be
     digit-identical, because the recorder only formats strings.
  2. TRACE REUSE. The E94 rung-1 traces were recorded at base `c4e849a8`. This
     base adds the E95 q-live-rows rider, which claims bit-exactness. If that
     claim holds, the traced legs here must reproduce the E94 legs round for
     round in (d, acc), and the whole recorded cap sweep stays valid evidence.

usage:
  research/e99_rung1_check.py --pairs e99r1c5t1=e94c5a e99r1c7t1=e94c7a \
                              --identical e99r1c5t1 e99r1c5u1 e99r1c5u2 e99r1c5t2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROUND_RE = re.compile(r'^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) ')


def read_meta(directory: Path) -> dict:
    return dict(line.split('=', 1)
                for line in (directory / 'meta.txt').read_text().splitlines()
                if '=' in line)


def read_score(directory: Path) -> dict:
    return json.loads((directory / 'score.json').read_text())['metrics']


def read_rounds(directory: Path) -> list[tuple[int, int, int]]:
    path = directory / 'trace.txt'
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        found = ROUND_RE.match(line)
        if found:
            out.append(tuple(int(found.group(i)) for i in (1, 2, 3)))
    return out


def arm_witness(directory: Path) -> tuple[int, int]:
    path = directory / 'trace.txt'
    if not path.exists():
        return (0, 0)
    lines = [line for line in path.read_text().splitlines()
             if line.startswith('mtp-trace: round=')]
    return (sum(1 for line in lines if ' arm=ship ' in line), len(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', default='research/out')
    parser.add_argument('--identical', nargs='+', required=True)
    parser.add_argument('--pairs', nargs='+', default=[])
    args = parser.parse_args()

    root = Path(args.out_root)
    status = 0

    print('== leg record ==')
    print(f'{"leg":12s}{"trace":>6s}{"rounds":>8s}{"draft_len":>22s}'
          f'{"accept_rate":>22s}{"s/token":>11s}{"entry C":>9s}{"exit C":>8s}')
    for tag in args.identical + [p.split('=')[0] for p in args.pairs]:
        directory = root / tag
        meta, score = read_meta(directory), read_score(directory)
        accepted, total = arm_witness(directory)
        print(f'{tag:12s}{meta.get("trace", "?"):>6s}{total:8d}'
              f'{score["effective_mean_draft_len"]!r:>22s}'
              f'{score["accepted_draft_rate"]!r:>22s}'
              f'{score["mtp_seconds_per_token"]:11.6f}'
              f'{float(meta.get("gpu_temp_entry_c", "nan")):9.1f}'
              f'{float(meta.get("gpu_temp_exit_c", "nan")):8.1f}')
        if total and accepted != total:
            print(f'FAIL {tag}: arm witness on {accepted} of {total} rounds')
            status = 1
        if not score['all_tokens_matched']:
            print(f'FAIL {tag}: all_tokens_matched is false')
            status = 1

    print()
    print('== recorder neutrality ==')
    reference = read_score(root / args.identical[0])
    for tag in args.identical[1:]:
        score = read_score(root / tag)
        for field in ('effective_mean_draft_len', 'accepted_draft_rate',
                      'mtp_depth'):
            same = repr(score[field]) == repr(reference[field])
            print(f'{"ok  " if same else "FAIL"} {tag} {field}: '
                  f'{score[field]!r} vs {reference[field]!r}')
            status = status if same else 1

    print()
    print('== trace cost, ABBA counterbalanced (T U U T) ==')
    traced, untraced = [], []
    for tag in args.identical:
        score = read_score(root / tag)
        target = traced if read_meta(root / tag).get('trace') == '1' \
            else untraced
        target.append(score['mtp_seconds_per_token'])
    if traced and untraced:
        on = sum(traced) / len(traced)
        off = sum(untraced) / len(untraced)
        print(f'trace on  {on:.6f} s/token  (n={len(traced)})')
        print(f'trace off {off:.6f} s/token  (n={len(untraced)})')
        print(f'recorder cost {100.0 * (on - off) / off:+.3f} %'
              f'   within-arm spread on {max(traced) - min(traced):.6f}'
              f'  off {max(untraced) - min(untraced):.6f}')

    print()
    print('== trace reuse: this base against the E94 recording ==')
    for pair in args.pairs:
        new, old = pair.split('=')
        left, right = read_rounds(root / new), read_rounds(root / old)
        same = left == right
        print(f'{"ok  " if same else "FAIL"} {new} vs {old}: '
              f'{len(left)} and {len(right)} rounds, '
              f'(round, d, acc) sequences {"identical" if same else "DIFFER"}')
        if not same:
            status = 1
            for index, (a, b) in enumerate(zip(left, right)):
                if a != b:
                    print(f'     first difference at index {index}: '
                          f'{a} vs {b}')
                    break

    print()
    print('PASS' if status == 0 else 'FAIL')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
