#!/usr/bin/env python3
"""Recover the exact decode round count from a published receipt.

The receipt publishes no round count, so every microsecond-per-round figure in
this campaign so far divided by an ASSUMED acceptance rate. It does not have to.

`effective_mean_draft_len` is a ratio of two integers: total proposed drafts
over total decode rounds. A float carries that ratio exactly enough that
`Fraction.limit_denominator` returns the reduced denominator q, and the true
round count is a multiple of q. Three constraints pin the multiple:

  1. a round emits one primary token, so rounds <= decode_tokens;
  2. a round emits at most 1 + edl tokens, so rounds >= tokens / (1 + edl);
  3. `non_drafting_round_count` counts rounds, so rounds >= that.

On this fixture the surviving multiple is unique. Given rounds, the acceptance
rate follows: accepted = tokens - rounds and proposed = edl * rounds.

Validated against a local 512-token leg whose trace counted 78 rounds: the
recovery returns 78 from the published edl alone.

  research/e135_round_recovery.py [--board PATH] [--receipt ID ...]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_receipt_read as rr  # noqa: E402

MAX_DENOMINATOR = 600


def recover_rounds(edl: float, tokens: int, non_drafting: int = 0) -> dict:
    """Exact round count behind one published `effective_mean_draft_len`."""
    if edl <= 0.0:
        return {"rounds": tokens, "period": tokens, "candidates": [tokens],
                "acceptance": float("nan")}
    q = Fraction(edl).limit_denominator(MAX_DENOMINATOR).denominator
    lo = max(int(math.ceil(tokens / (1.0 + edl))), non_drafting, 1)
    candidates = [n for n in range(q, tokens + 1, q) if n >= lo]
    rounds = candidates[0] if candidates else tokens
    proposed = edl * rounds
    return {
        "rounds": rounds,
        "period": q,
        "candidates": candidates,
        "proposed": proposed,
        "acceptance": (tokens - rounds) / proposed if proposed else float("nan"),
    }


def _affine(points: list) -> tuple:
    """Least-squares a + b*width over (width, us_per_round) points."""
    n = len(points)
    sw = sum(w for w, _ in points)
    sy = sum(y for _, y in points)
    sww = sum(w * w for w, _ in points)
    swy = sum(w * y for w, y in points)
    den = n * sww - sw * sw
    b = (n * swy - sw * sy) / den
    return (sy - b * sw) / n, b


def select_multiples(row: dict, tokens: int, rounds_of: dict) -> dict:
    """Pick each prompt's round multiple, and say which picks are anchored.

    A decode round costs about `a + b * verify_width`, and the verify width is
    `1 + edl`. That law alone cannot fix the multiples: scaling every round
    count by one common factor rescales `a` and `b` and leaves the percentage
    residuals untouched. The anchor is the SCHEDULER. It drafts deeper when it
    sees higher acceptance, so a prompt whose smallest admissible multiple
    implies MORE acceptance than a strictly deeper-drafting prompt contradicts
    the policy that produced it. Those prompts are the ambiguous ones.

    The consistent prompts, plus any prompt with a unique candidate, fix the
    law. The ambiguous ones are then assigned to the candidate closest to it,
    and reported as fitted rather than recovered.
    """
    entries = rr.per_prompt(row)
    acc = {}
    for name, e in entries.items():
        edl = e["effective_mean_draft_len"]
        acc[name] = ((tokens - rounds_of[name]) / (edl * rounds_of[name])
                     if edl > 0 else 0.0)
    ambiguous = {
        name for name, e in entries.items()
        if any(acc[name] > acc[o] and
               entries[o]["effective_mean_draft_len"] >
               e["effective_mean_draft_len"] for o in entries)}
    anchored = [n for n in entries if n not in ambiguous]

    for _ in range(20):
        a, b = _affine([(1.0 + entries[n]["effective_mean_draft_len"],
                         entries[n]["mtp_seconds_per_token_mean"] * tokens
                         * 1e6 / rounds_of[n]) for n in anchored])
        moved = False
        for name in ambiguous:
            e = entries[name]
            want = a + b * (1.0 + e["effective_mean_draft_len"])
            total = e["mtp_seconds_per_token_mean"] * tokens * 1e6
            best = min(recover_rounds(e["effective_mean_draft_len"], tokens,
                                      e.get("non_drafting_round_count", 0)
                                      )["candidates"],
                       key=lambda c: abs(total / c - want))
            if best != rounds_of[name]:
                rounds_of[name] = best
                moved = True
        if not moved:
            break
    return {"a": a, "b": b, "rounds": rounds_of, "ambiguous": ambiguous}


def report(row: dict, tokens: int) -> None:
    print(f"\n{row['id'][:8]}  {row.get('officialScore', float('nan')):.8f}  "
          f"{(row.get('solverUsername') or '?')}")
    entries = rr.per_prompt(row)
    smallest = {n: recover_rounds(e["effective_mean_draft_len"], tokens,
                                  e.get("non_drafting_round_count", 0))["rounds"]
                for n, e in entries.items()}
    fit = select_multiples(row, tokens, dict(smallest))
    print(f"  verify-cost law  a {fit['a']:.1f} us  b {fit['b']:.1f} us/width"
          f"  ambiguous {sorted(fit['ambiguous']) or 'none'}")
    print("  prompt      edl      period  small  fitted  alts  accept  "
          "us/round   resid %")
    for name, e in sorted(entries.items(),
                          key=lambda kv: kv[1]["effective_mean_draft_len"]):
        edl = e["effective_mean_draft_len"]
        r = recover_rounds(edl, tokens, e.get("non_drafting_round_count", 0))
        n = fit["rounds"][name]
        us = e["mtp_seconds_per_token_mean"] * tokens * 1e6 / n
        want = fit["a"] + fit["b"] * (1.0 + edl)
        acc = (tokens - n) / (edl * n) if edl > 0 else float("nan")
        print(f"  {name:<9} {edl:9.6f} {r['period']:6d} {smallest[name]:6d}"
              f" {n:7d} {len(r['candidates']):5d}  {acc:6.4f} {us:9.1f}"
              f"  {(us / want - 1) * 100:+8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="/tmp/yukon-board/read.json")
    ap.add_argument("--receipt", action="append",
                    default=None, help="repeatable")
    args = ap.parse_args()

    board = rr.load_board(args.board)
    for ident in (args.receipt or ["0cf1637e", "684821ed", "572b2cc4"]):
        row = rr.find(board, ident)
        report(row, row["officialMetrics"].get("decode_tokens", 512))


if __name__ == "__main__":
    main()
