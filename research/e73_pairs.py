#!/usr/bin/env python3
"""The legal `(M, IPG)` group partitions and what each one implies statically.

One source of truth for the census, the arm emitter and the fit, so a pair
cannot be legal in one instrument and missing from another.

Legality is the two shipped static asserts, `3 <= M <= 9` (the wrapper),
`2 <= IPG <= 6` (the helper) and `M % IPG != 1` (no one-row tail group), plus
`IPG <= M`. The last one is not asserted in the source but is implied: at
`IPG > M` the wrapper's only active group takes the tail branch and runs at
`NA = M`, so the pair is a slower spelling of `IPG = M` and not a partition.
"""

from __future__ import annotations

import math

M_RANGE = range(3, 10)
IPG_RANGE = range(2, 7)

# The partition the campaign base ships today, and the crown's partition.
SHIPPED = {3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 5}
CROWN = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# Ranked verify-width time shares, from the assignment's public receipts.
RANKED_WIDTH_SHARE = {4: 14.2, 5: 24.1, 6: 33.4, 7: 12.2, 8: 7.35, 9: 5.75}

# K to N, every distinct affine-4 group-64 shape on the scored path.
SCORED_SHAPES = [
    ("linear_attn.in_proj_fused_qkvzba", 5120, 16480),
    ("full_attn.qkv_proj_fused", 5120, 14336),
    ("mlp.gate_up_fused", 5120, 34816),
    ("head.lm_head", 5120, 248320),
    ("linear_attn.out_proj", 6144, 5120),
    ("mlp.down", 17408, 5120),
]
# `full_attn.o_proj` is the same (6144, 5120) shape as `linear_attn.out_proj`,
# so one entry covers both. Seven scored linear shapes, six distinct cells.


def legal(m: int, ipg: int) -> bool:
    return m in M_RANGE and ipg in IPG_RANGE and ipg <= m and m % ipg != 1


def pairs() -> list[tuple[int, int]]:
    return [(m, ipg) for m in M_RANGE for ipg in IPG_RANGE if legal(m, ipg)]


def groups(m: int, ipg: int) -> int:
    """Active x-slots, which is also the number of weight streams."""
    return math.ceil(m / ipg)


def tail(m: int, ipg: int) -> int:
    return m % ipg


def bodies(m: int, ipg: int) -> list[int]:
    """Helper instantiations the wrapper inlines, main body first."""
    t = tail(m, ipg)
    return [ipg] if t == 0 else [ipg, max(t, 2)]


def live_floats(ipg: int) -> int:
    """`acc[4]`, `partial[4]` and `sums`, each `vec<float, IPG>`."""
    return 9 * ipg


def weight_bytes(k: int, n: int) -> float:
    """Bytes ONE weight stream reads: 4-bit packed data plus g64 scale/bias."""
    return n * k / 2.0 + 2.0 * 2.0 * n * k / 64.0


def name(m: int, ipg: int) -> str:
    return f"m{m}_ipg{ipg}"


if __name__ == "__main__":
    print(f"{'M':>2} {'IPG':>3} {'groups':>6} {'tail':>4} {'bodies':>8} "
          f"{'live_floats':>11} shipped crown")
    for m, ipg in pairs():
        print(f"{m:2d} {ipg:3d} {groups(m, ipg):6d} {tail(m, ipg):4d} "
              f"{','.join(str(b) for b in bodies(m, ipg)):>8} "
              f"{live_floats(ipg):11d} "
              f"{'*' if SHIPPED.get(m) == ipg else ' ':>7} "
              f"{'*' if CROWN.get(m) == ipg else ' ':>5}")
    print(f"{len(pairs())} legal pairs")
