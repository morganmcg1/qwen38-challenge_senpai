#!/usr/bin/env python3
"""Price the E55-by-E56 interaction from the two E56 sessions.

  python3 research/e56_e55_interaction.py

The advisor asked for the 8-to-9 width component measured on the old base and
on the post-E55 base with the same arms, and called it the most valuable single
number in the revision. This computes it.

The two sessions share a host, a fixture, a token window and a real cool gate,
so absolute candidate seconds per token are comparable between them. Each
session also carries its own unchanged base pair, so E55's own effect is
measured rather than assumed.

WHICH ARM IS "E56 AT ITS BEST" DIFFERS BY BASE, AND THAT IS THE POINT. Pre-E55
the dispatch table has two weight-stream boundaries, at 4->5 and 8->9, so the
arm that prices the full staircase is `sfull` (priced 5 and 9). E55 deleted the
8->9 boundary, so post-E55 the same policy is `s45`. Comparing `sfull` against
`s45` is therefore comparing one mechanism against itself on two tables, not
comparing two different mechanisms.
"""
from __future__ import annotations

S3_BASE = (0.035747471, 0.035747012)      # pre-E55, aded0f5
S3_SFULL = (0.032719596, 0.032816973)     # prices both boundaries: 5 and 9
S4_BASE = (0.03409490, 0.03402046)        # post-E55, 7040406
S4_S45 = (0.03271110, 0.03272873)         # prices the only boundary left: 5


def mean(pair: tuple[float, ...]) -> float:
    return sum(pair) / len(pair)


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def main() -> None:
    s3_base, s3_best = mean(S3_BASE), mean(S3_SFULL)
    s4_base, s4_best = mean(S4_BASE), mean(S4_S45)

    e55_alone = pct(s4_base, s3_base)
    e56_alone = pct(s3_best, s3_base)
    combined = pct(s4_best, s3_base)
    e56_on_new_base = pct(s4_best, s4_base)

    # Multiplicative because both are proportional speedups of the same total.
    additive = 100.0 * ((1 + e55_alone / 100.0) * (1 + e56_alone / 100.0) - 1)
    interaction = combined - additive

    print("E55 x E56 interaction, absolute candidate seconds per token")
    print(f"  pre-E55  base   {s3_base:.8f}")
    print(f"  pre-E55  sfull  {s3_best:.8f}   (prices boundaries 5 and 9)")
    print(f"  post-E55 base   {s4_base:.8f}")
    print(f"  post-E55 s45    {s4_best:.8f}   (prices boundary 5, the only one)")
    print()
    print(f"  E55 alone, base to base            {e55_alone:+8.4f} %")
    print(f"  E56 alone, on the pre-E55 base     {e56_alone:+8.4f} %")
    print(f"  E56 alone, on the post-E55 base    {e56_on_new_base:+8.4f} %")
    print(f"  both, pre-E55 base to post-E55 s45 {combined:+8.4f} %")
    print()
    print(f"  independent prediction             {additive:+8.4f} %")
    print(f"  INTERACTION                        {interaction:+8.4f} pp")
    print(f"  fraction of the joint gain that the two mechanisms share: "
          f"{100.0 * interaction / -additive:.1f} %")
    print()
    print("  Reading: the two changes attack the same inefficiency. E55 makes")
    print("  the wide dispatch cheaper in the kernel; E56 routes the schedule")
    print("  around it. Doing both does not pay twice. The best candidate on")
    print(f"  the new base is only {pct(s4_best, s3_best):+.4f} % faster than the best")
    print("  candidate on the old one, even though E55 moved the base "
          f"{e55_alone:+.4f} %.")


if __name__ == "__main__":
    main()
