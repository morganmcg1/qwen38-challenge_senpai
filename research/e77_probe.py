#!/usr/bin/env python3
"""One source of truth for the E77 occupancy sweep arms.

E73 fitted a per-IPG rate level `q(IPG)` on a host whose register allocator
clamps every legal `(M, IPG)` cell into 93 to 96 registers. E72 then showed the
ranked generation spreads the same six shipped cells over 90 to 111. A cost
model cannot carry a register term that was fitted where registers do not move,
so E77 creates the variation deliberately: it holds one `(M, IPG)` cell and one
shape fixed and adds inert live state around the SHIPPED wrapper call.

Two arm families, both wrapping `qmv_fast_crossrow_affine4_g64_m` unchanged:

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
    found = []
    for (m, ipg), ladder in CARRIERS.items():
        for kind, levels in (("p", ladder), ("q", CONTROL[(m, ipg)])):
            for pressure in levels:
                found.append(dict(arm=name(m, ipg, kind, pressure), m=m,
                                  ipg=ipg, kind=kind, pressure=pressure,
                                  groups=math.ceil(m / ipg)))
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
              spec["pressure"], spec["groups"])
    print(f"{len(arms())} arms")
