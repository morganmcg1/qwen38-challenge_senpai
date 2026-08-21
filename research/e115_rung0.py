#!/usr/bin/env python3
"""E115 rung 0 -- reconcile the three group-scaling `A` values, no GPU.

Every number here is either read from `research/group_scaling.py` or derived
from it in this file. Nothing is measured here.

The one result that matters: `A_local` is an identity, not a measurement of
concurrency.

    r2 = G * W / t2      with G = 2 and t2 the measured [3+2] round
    r1 =     W / t1      with t1 = t2 * (1 - c), c the measured collapse
    A  = r2 / r1 = 2 * t1 / t2 = 2 * (1 - c)

so `A_local` carries exactly one measured input, the E100 round-level collapse
fraction c = 0.180, and `W` cancels. The script that publishes it prints
"matches by construction" out loud.
"""

from __future__ import annotations

W = 14.41235e9  # logical weight bytes in one pass over the backbone
LOCAL = {1: 64445, 2: 69776, 3: 74778, 4: 86237, 5: 126103}  # us per round
RANKED = {1: 31177, 2: 35172, 3: 39167, 4: 43162, 5: 53108}  # us per round
GOF_PRE_E100 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2}  # partition map assumed
COLLAPSE_MEASURED = 0.180
DW_MEASURED_PP, DW_SD_PP, G2_SHARE = -0.070, 0.360, 0.24

# Finding 44, harness=local: load-only, ALU-only and the shipped kernel.
ROOFLINE = {4: (203.0, 33.0, 245.9), 5: (207.0, 39.1, 292.2)}
# Finding 47 / E106 refit, harness=local: isolated ONE-group rate, GB/s.
ISOLATED_RATE = {2: 253.6, 3: 245.6, 4: 211.7, 5: 178.8}
# Standing local round weights over NA.
ROUND_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

FRAMES = {
    "E96 anchor decode round": 127_533,
    "current-tree decode-only M=5 round": 102_864,
}
QUALIFYING_DISPATCHES_PER_ROUND = 258  # Finding 36: 128 + 96 + 32 + 2
DISPATCH_BOUNDARY_US = 1.8  # E106 corrected boundary

SHAPES = {
    "mlp.gate_up": (34816, 5120),
    "lm_head": (248320, 5120),
    "gdn.in_proj": (16480, 5120),
    "fa.qkv": (14336, 5120),
}


def rate(times: dict[int, int], m: int, gof: dict[int, int]) -> float:
    return gof[m] * W / (times[m] * 1e-6) / 1e9


def packed_bytes(outputs: int, hidden: int) -> float:
    """affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias."""
    return outputs * hidden / 2 + 4 * (outputs * hidden / 64)


def main() -> None:
    print("=" * 78)
    print("1. A_local is an identity on ONE measured input")
    print("=" * 78)
    r2 = rate(LOCAL, 5, GOF_PRE_E100)
    t1 = LOCAL[5] * (1 - COLLAPSE_MEASURED)
    r1 = W / (t1 * 1e-6) / 1e9
    a_local = r2 / r1
    print(f"  measured input   E100 collapse c = {COLLAPSE_MEASURED:.3f} "
          f"(round level, -17.5..-18.6 % per M=5 round)")
    print(f"  r2 = 2W/t2 = {r2:.1f} GB/s   r1 = W/t1 = {r1:.1f} GB/s")
    print(f"  A_local = r2/r1 = {a_local:.3f}")
    print(f"  identity 2*(1-c) = {2 * (1 - COLLAPSE_MEASURED):.3f}   "
          f"difference {abs(a_local - 2 * (1 - COLLAPSE_MEASURED)):.1e}")
    print("  => W, the byte convention and the rate units all cancel. A_local")
    print("     is a re-parameterisation of one round-level time ratio.")

    print()
    print("=" * 78)
    print("2. A_local is a ROUND ratio, so non-QMV round time dilutes it")
    print("=" * 78)
    print("  t = F + q with F the non-QMV part of the round. Collapsing groups")
    print("  changes q only, so A_round = 2(F+q1)/(F+q2) > A_kernel = 2q1/q2")
    print("  for every F > 0 and q1 < q2. Worked illustration only:")
    for f_share in (0.0, 0.2, 0.4):
        t2 = LOCAL[5]
        f = f_share * t2
        q2 = t2 - f
        q1 = t1 - f
        print(f"    F = {100 * f_share:2.0f} % of the round -> "
              f"A_kernel = {2 * q1 / q2:.3f}")
    print("  => the kernel-level two-group advantage is at most A_local, and is")
    print("     smaller than it whenever the round holds any non-QMV work.")

    print()
    print("=" * 78)
    print("3. A does not isolate concurrency: it mixes four terms")
    print("=" * 78)
    print("  (a) overlap between two concurrent dispatches")
    print("  (b) the per-dispatch rate(NA) curve, Finding 47 harness=local:")
    for na, gbs in ISOLATED_RATE.items():
        print(f"        NA={na}  isolated one-group rate {gbs:6.1f} GB/s")
    print("      a [3+2] partition runs its groups at NA=3 and NA=2, which are")
    print("      intrinsically faster per byte than one NA=5 group, and that")
    print("      difference is inside A with no way to remove it")
    print("  (c) total logical bytes: 2 passes against 1")
    print("  (d) the extra dispatch boundary")
    print("  E115 rung 1 holds (b), (c) and NA fixed and measures (a) alone.")

    print()
    print("=" * 78)
    print("4. The two ranked routes")
    print("=" * 78)
    sc_loc = rate(LOCAL, 5, GOF_PRE_E100) / rate(LOCAL, 3, GOF_PRE_E100)
    sc_rnk = rate(RANKED, 5, GOF_PRE_E100) / rate(RANKED, 3, GOF_PRE_E100)
    a_route1 = a_local * (sc_rnk / sc_loc)
    gain = (DW_MEASURED_PP / 100.0) / G2_SHARE
    gain_sd = (DW_SD_PP / 100.0) / G2_SHARE
    a_route2 = 2 * (1 + gain)
    print(f"  route 1  A_ranked = A_local x {sc_rnk / sc_loc:.3f} = "
          f"{a_route1:.3f}   INFERRED, inherits every A_local input")
    print(f"  route 2  A_ranked = 2(1 + dW/(100*share)) = {a_route2:.3f} "
          f"[{2 * (1 + gain + gain_sd):.3f}, {2 * (1 + gain - gain_sd):.3f}]")
    print(f"           dW = {DW_MEASURED_PP:+.3f} +/- {DW_SD_PP:.3f} pp, so the")
    print(f"           standard deviation is {DW_SD_PP / abs(DW_MEASURED_PP):.1f}x "
          f"the point estimate; share = {G2_SHARE} is assumed, not measured")

    print()
    print("=" * 78)
    print("5. Roofline headroom, the H1 upper bound, harness=local")
    print("=" * 78)
    for na, (load, alu, base) in ROOFLINE.items():
        best = max(load, alu)
        print(f"  NA={na}  a_base {base:6.1f} us  load {load:6.1f} us  "
              f"ALU {alu:5.1f} us  perfect overlap would give {best:6.1f} us "
              f"= {100 * (1 - best / base):+.1f} %")
    print("  An N-split cannot beat perfect overlap, so those are the ceilings")
    print("  for c_nsplit under H1 at those two widths.")

    print()
    print("=" * 78)
    print("6. Break-even: the split doubles dispatches on the four tensors")
    print("=" * 78)
    extra = QUALIFYING_DISPATCHES_PER_ROUND * DISPATCH_BOUNDARY_US
    print(f"  {QUALIFYING_DISPATCHES_PER_ROUND} extra dispatches x "
          f"{DISPATCH_BOUNDARY_US} us = {extra:.0f} us per round")
    for name, frame in FRAMES.items():
        print(f"    {100 * extra / frame:.3f} % of the {name} ({frame} us)")
    print("  Rule 34: the same absolute cost is 0.36 % or 0.45 % depending on")
    print("  the frame. Every table below names its frame.")

    print()
    print("=" * 78)
    print("7. Shapes that qualify for an N-split, and the boundary tax on each")
    print("=" * 78)
    print(f"  {'shape':14s} {'N':>7s} {'half':>7s} {'quarter':>8s} "
          f"{'MB':>7s} {'us@200GB/s':>11s} {'+1 boundary':>12s}")
    for name, (outputs, hidden) in SHAPES.items():
        b = packed_bytes(outputs, hidden)
        us = b / 200e9 * 1e6
        half_ok = "ok" if outputs // 2 >= 4096 else "NARROW"
        quarter_ok = "ok" if outputs // 4 >= 4096 else "NARROW"
        print(f"  {name:14s} {outputs:7d} {outputs // 2:7d} "
              f"{outputs // 4:8d} {b / 1e6:7.1f} {us:11.1f} "
              f"{100 * DISPATCH_BOUNDARY_US / us:11.2f} %   "
              f"half={half_ok} quarter={quarter_ok}")

    print()
    print("=" * 78)
    print("8. Round weights used for the per-NA aggregate, harness=local")
    print("=" * 78)
    for na, weight in ROUND_WEIGHTS.items():
        print(f"  NA={na}  weight {weight:.3f}")
    print(f"  sum {sum(ROUND_WEIGHTS.values()):.3f}. askeladd's E114 is "
          f"re-deriving the scoring-correct weights, so the per-NA table must")
    print("  survive being reweighted later.")


if __name__ == "__main__":
    main()
