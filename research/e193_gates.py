#!/usr/bin/env python3
"""E193 arm-witness and exactness gates, corrected for a two-leg trace file.

WHY THIS EXISTS
---------------
`research/e165_prefetch.py witness` and `... rows` cannot validate a trace
produced by `research/e79_trace_leg.sh` on this tree. One leg script points
MLX_QWEN_MTP_TRACE_PATH at a single file and BOTH decode legs of the benchmark
write into it: the serial depth-0 leg and the candidate MTP leg. Measured on
the `on` witness leg of this experiment (256 decode tokens):

    total traced rounds            291
      serial leg, d=0 acc=0        256   <- no pf fields
      MTP leg, d in {2,4,5,6,7}     35   <- carries the pf instrument
    round numbers restart at 1 in both legs

Two consequences, both of which make the original gates useless here rather
than merely noisy:

1. `witness` requires EVERY traced round to carry `pf`, so it reports "this
   build predates the instrument" and exits 1 on a correct build that is
   running the arm perfectly. Its `hit_rate` is also computed over all 291
   rounds (34/291 = 0.117) instead of over the 35 drafting rounds
   (34/35 = 0.971), so it would fail the default 0.75 threshold as well.

2. `compare_rows` keys rows by token position in a dict, so the two legs
   collide at every shared position and one silently overwrites the other.
   The surviving comparison depends on which leg happens to write last.

The fix keeps every protection the original gates provided and adds one.
`witness` still fails a build that lacks the instrument, and it now also
fails if any DRAFTING round is missing `pf` -- which is the real corruption
the original blanket check was reaching for. `rows` now compares the complete
ordered row sequence rather than a deduplicated position map, which is
strictly stronger: both arms run the same schedule under RULE 179, so the
sequences must be identical element for element.

Usage:
  python3 research/e193_gates.py witness TRACE --want on|off
  python3 research/e193_gates.py rows LEFT RIGHT [--positive-control]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from e165_prefetch import next_after, read_rows, rounds

PF_FIELDS = ("pf", "pf_hit", "pf_made", "pf_hits", "pf_undo")


def split_legs(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split traced rounds into pf-carrying rounds and pf-less rounds."""
    carry = [r for r in recs if all(k in r for k in PF_FIELDS)]
    bare = [r for r in recs if not all(k in r for k in PF_FIELDS)]
    return carry, bare


def witness(args) -> int:
    recs = rounds(Path(args.trace))
    if not recs:
        print("witness: no traced rounds")
        return 1

    carry, bare = split_legs(recs)
    print(f"traced_rounds={len(recs)} pf_rounds={len(carry)} "
          f"pf_less_rounds={len(bare)}")

    # The instrument must be present. This is the original "build predates the
    # instrument" protection, now expressed as "no round carries pf at all".
    if not carry:
        print("witness: no round carries a pf field; this build predates the "
              "instrument")
        return 1

    # Every pf-less round must be a zero-draft round. Those are the serial
    # leg's rounds and the adaptive-skip rounds, which return before the
    # drafting trace emit. A DRAFTING round without pf is real corruption.
    drafting_without_pf = [r for r in bare if r["d"] != 0 or r["acc"] != 0]
    if drafting_without_pf:
        print(f"witness: {len(drafting_without_pf)} drafting rounds lack a pf "
              "field; the instrument is broken")
        return 1
    print(f"  [ok] all {len(bare)} pf-less rounds are zero-draft rounds")

    for name in ("pf_made", "pf_hits", "pf_undo"):
        series = [r[name] for r in carry]
        if any(b < a for a, b in zip(series, series[1:])):
            print(f"witness: {name} is not monotone; it is not a counter")
            return 1

    last = carry[-1]
    made, hits, undo = (
        int(last["pf_made"]), int(last["pf_hits"]), int(last["pf_undo"]))
    arm_on = sum(1 for r in carry if r["pf"] == 1)
    hit_rounds = sum(1 for r in carry if r["pf_hit"] == 1)
    hit_rate = hit_rounds / len(carry)
    print(f"pf_on={arm_on}/{len(carry)} pf_made={made} pf_hits={hits} "
          f"pf_undo={undo} hit_rounds={hit_rounds} hit_rate={hit_rate:.4f}")

    if args.want == "on":
        checks = [
            ("arm compiled on in every drafting round", arm_on == len(carry)),
            ("counter agrees with per-round flag", hits == hit_rounds),
            ("made == hits + undo + pending<=1", 0 <= made - hits - undo <= 1),
            ("steps were actually made", made >= 1),
            (f"hit_rate >= {args.min_hit_rate}", hit_rate >= args.min_hit_rate),
        ]
    else:
        checks = [
            ("arm compiled off in every drafting round", arm_on == 0),
            ("no step made", made == 0),
            ("no step consumed", hits == 0 and hit_rounds == 0),
            ("no step undone", undo == 0),
        ]
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


def compare_rows(args) -> int:
    """Compare the COMPLETE ordered row sequence of two traces.

    Ordered comparison, not a position map: both legs of a run write into one
    trace file and revisit the same token positions, so a dict keyed by
    position discards evidence and its result depends on write order.
    """
    left = read_rows(Path(args.left))
    right = read_rows(Path(args.right))
    if not left or not right:
        print("rows: one of the traces carries no row dump")
        return 1

    if args.positive_control:
        index = len(right) // 2
        pos, a, b, values = right[index]
        right = list(right)
        right[index] = (pos, a, b, (next_after(values[0]),) + values[1:])
        print(f"positive control: perturbed row {index} (pos={pos}) by one ulp")

    print(f"rows_left={len(left)} rows_right={len(right)}")
    if len(left) != len(right):
        print("rows: the two arms emitted a different number of rows")
        return 1

    bad = [i for i, (x, y) in enumerate(zip(left, right)) if x != y]
    print(f"compared={len(left)} mismatched={len(bad)}")
    if bad:
        i = bad[0]
        print(f"  first mismatch at row {i}: left={left[i]} right={right[i]}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("witness")
    w.add_argument("trace")
    w.add_argument("--want", choices=["on", "off"], required=True)
    w.add_argument("--min-hit-rate", type=float, default=0.75)
    w.set_defaults(func=witness)

    r = sub.add_parser("rows")
    r.add_argument("left")
    r.add_argument("right")
    r.add_argument("--positive-control", action="store_true")
    r.set_defaults(func=compare_rows)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
