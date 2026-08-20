#!/usr/bin/env python3
"""One source of truth for the E77 occupancy sweep arms.

E73 fitted a per-IPG rate level `q(IPG)` on a host whose register allocator
clamps every legal `(M, IPG)` cell into 93 to 96 registers. E72 then showed the
ranked generation spreads the same six shipped cells over 90 to 111. A cost
model cannot carry a register term that was fitted where registers do not move,
so E77 creates the variation deliberately: it holds one `(M, IPG)` cell and one
shape fixed and adds inert live state around the SHIPPED wrapper call.

Two arm families, both wrapping `qmv_fast_crossrow_affine4_g64_m` unchanged.

The MAIN contrast is natural, not synthetic. At M = 8 the cells IPG 4, 5 and 6
all split into exactly two groups, so M, the group count, the weight traffic,
the working threadgroup count and the grid are all identical and only the
register count moves. M = 6 (IPG 3 vs 4), M = 7 (IPG 4 vs 5) and M = 9
(IPG 5 vs 6) give the same fixed-group-count contrast as replicates, and
M = 4 IPG 2 gives a 70-register low anchor at two groups. Nothing in these arms
is inert, so the optimizer has nothing to delete.

The synthetic ladders are SECONDARY. Their job is to locate the occupancy tier
boundaries at or below 96 registers, not to price a ranked cell:

  `p{P}`  P extra named live floats, loaded before the call and consumed after
          it under a runtime-false predicate. The register allocator must keep
          them across the whole inlined body, including the streaming loop, so
          the arm raises register pressure. The loop's own arithmetic is
          emitted from the same template instantiation and is unchanged.

  `q{P}`  the SAME P loads and the SAME consumption, both placed BEFORE the
          call. The memory traffic, the load count and the arithmetic are
          identical to `p{P}`; only the live range is moved off the loop,
          because a conditional store to `y` cannot sink past the stores the
          helper makes. This is the traffic control that separates the cost of
          the loads from the cost of the pressure.

`p0` is the unmodified cell and is the shared reference for both families.
"""

from __future__ import annotations

import math

# Carrier cells, each a legal shipped partition; see research/e73_pairs.py.
#
# `m6_ipg2` carries the sweep. It is the lowest-pressure legal cell in the
# family, 70 registers unpadded, and the local allocator stops at 96, so it is
# the only carrier on this host that offers more than a few registers of clean
# range. Its pressure ladder is dense enough to resolve single registers.
#
# `m6_ipg3` is the crown's M=6 choice and the shipped-cell carrier. It starts at
# 93 and reaches the ceiling after two live floats, so it contributes three
# clean points and then a frame-bytes ladder at a fixed 96 registers, which is
# the local spill regime rather than the occupancy regime.
CARRIERS: dict[tuple[int, int], tuple[int, ...]] = {
    (6, 2): tuple(range(0, 29)) + (32,),
    (6, 3): (0, 1, 2, 4, 8, 16, 32),
}

# Traffic controls per carrier: same loads, live range off the loop.
CONTROL: dict[tuple[int, int], tuple[int, ...]] = {
    (6, 2): (8, 16, 24),
    (6, 3): (2, 8, 32),
}

# Natural register contrasts, all at pressure 0: real shipped-capable cells that
# hold M, the group count, the weight traffic, the working threadgroup count and
# the grid fixed, and move ONLY the register count. The advisor's g17s census
# supplies the ranked side, where the same contrasts span 91 to 111 registers.
#
#   M=8, IPG 4/5/6  -> [4,4] [5,3] [6,2], 2 groups, 94/95/96 local, 91/98/111 ranked
#   M=6, IPG 3/4    -> [3,3] [4,2], 2 groups, 93/94 local, 90/91 ranked
#   M=7, IPG 4/5    -> [4,3] [5,2], 2 groups, 94/95 local, 91/98 ranked
#   M=9, IPG 5/6    -> [5,4] [6,3], 2 groups, 95/96 local, 98/111 ranked
#   M=4, IPG 2      -> [2,2],       2 groups, 70 local, 83 ranked: the low anchor
#
# These carry the main contrast. The synthetic ladders above are secondary and
# locate the tier boundaries at or below 96 registers.
NATURAL: tuple[tuple[int, int], ...] = (
    (8, 4), (8, 5), (8, 6),
    (6, 3), (6, 4),
    (7, 4), (7, 5),
    (9, 5), (9, 6),
    (4, 2),
)

PROBE_ARGS = """    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const device float* pad [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]"""

CALL = """  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, {m}, {ipg}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
"""


def name(m: int, ipg: int, kind: str, pressure: int) -> str:
    return f"m{m}_ipg{ipg}_{kind}{pressure}"


def arms() -> list[dict]:
    found, seen = [], set()

    def add(m: int, ipg: int, kind: str, pressure: int, family: str) -> None:
        arm = name(m, ipg, kind, pressure)
        if arm in seen:
            return
        seen.add(arm)
        found.append(dict(arm=arm, m=m, ipg=ipg, kind=kind, pressure=pressure,
                          groups=math.ceil(m / ipg), family=family))

    for m, ipg in NATURAL:
        add(m, ipg, "p", 0, "natural")
    for (m, ipg), ladder in CARRIERS.items():
        for kind, levels in (("p", ladder), ("q", CONTROL[(m, ipg)])):
            for pressure in levels:
                add(m, ipg, kind, pressure, "synthetic")
    return found


def body(m: int, ipg: int, kind: str, pressure: int) -> str:
    """The kernel body for one arm.

    Every pad element is read at `simd_lid + 32 * j`, so a simdgroup reads whole
    128-byte lines and every threadgroup reads the SAME `32 * P` floats. The
    unique footprint is at most 4 KiB, which the cache serves, while the arm
    still gets one distinct per-thread value per named float.

    The consumption is guarded by `in_vec_size < 0`. `in_vec_size` is a runtime
    constant buffer value and is always positive, so the store never runs, but
    the compiler cannot prove it and must keep the values live to that point.
    """
    call = CALL.format(m=m, ipg=ipg)
    if pressure == 0:
        return call
    load = "".join(f"  float v{j} = pad[simd_lid + {32 * j}];\n"
                   for j in range(pressure))
    consume = ("  if (in_vec_size < 0) {\n    y[0] = static_cast<bfloat16_t>("
               + " + ".join(f"v{j}" for j in range(pressure)) + ");\n  }\n")
    if kind == "p":
        return f"{load}{call}{consume}"
    return f"{load}{consume}{call}"


def entry(spec: dict) -> str:
    return (f"\n[[kernel]] void e77_{spec['arm']}(\n{PROBE_ARGS}) {{\n"
            f"{body(spec['m'], spec['ipg'], spec['kind'], spec['pressure'])}}}\n")


if __name__ == "__main__":
    for spec in arms():
        print(spec["arm"], spec["m"], spec["ipg"], spec["kind"],
              spec["pressure"], spec["groups"], spec["family"])
    print(f"{len(arms())} arms")
