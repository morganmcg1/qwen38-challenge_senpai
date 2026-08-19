#!/usr/bin/env python3
"""E41 pre-registration: resolve R2 into re-read locality versus register tile.

Committed BEFORE the kernel exists and before anything is compiled or timed.
Nothing here is re-derived later; research/e41_analyze.py only compares against
these numbers.

    python3 research/e41_prereg.py              # print the registered table
    python3 research/e41_prereg.py --self-test  # check the internal arithmetic

WHAT R2 IS
----------
E38 measured, at M=6 on the eight scored shapes, that covering a simdgroup's
frozen 4 output rows as two sequential 2-row blocks costs +10.54 % against the
shipped one-block cell:

    base   <T,6,3,true>     r=4, one row block,  83 AIR regs, vec_float_ops 48
    arm(a) <T,6,3,true,2>   r=2, two row blocks, 66 AIR regs, vec_float_ops 24

That single change bundles TWO mechanisms and E38 could not separate them:

    MEM  the activation tile is read twice instead of once, and the second read
         happens a full K away from the first (`for b { for k }`), so it cannot
         hit any cache that a 104 KB (NA=3) working set has already evicted.
    ILP  the register tile halves, so the number of independent accumulator
         chains per loaded x tile halves (4 -> 2 at NA=3), and one loop is added
         (loop_backedges 2 -> 3).

The decision this experiment gates: K-tiled activation staging removes the
re-read and keeps r=2, so it recovers R2 under MEM and recovers nothing under
ILP.

WHY NOT THE UNROLLED ARM
------------------------
The advisor proposed unrolling arm(a)'s row-block loop to restore ILP at
constant loads. That arm is not sound as stated, and the reason is worth more
than the arm: once the two blocks are adjacent in program order they read the
SAME device addresses with no intervening store, so common-subexpression
elimination is legal and likely. An unrolled arm that recovers R2 cannot
distinguish "ILP restored" from "the compiler deleted the second load", i.e. it
silently changes BOTH mechanisms in the direction that makes ILP look guilty.
peak_live_regs is the detector: if CSE fires, a0..a3 stay live across both
blocks and the cell's register footprint climbs toward the r=4 cell's.

So the arms below vary re-read DISTANCE at fixed instruction mix instead.

THE LADDER (one build, one M per rung, ratio taken against the same M in base)
-----------------------------------------------------------------------------
A new kernel keeps r=2 and BLOCKS=2 but tiles K, holding both blocks'
accumulators live so the k order per output row is unchanged:

    for kt in 0..K step KT*512:  for b in 0..1:  for k in kt..kt+KT*512

    KT = all  both blocks run the whole K.  re-read distance = K.   no adjacency
    KT = 4    re-read distance = 4 k-blocks (~12 KB at NA=4).      no adjacency
    KT = 1    inner k loop collapses; blocks adjacent per k-block. adjacency

KT=all -> KT=4 changes ONLY the re-read distance: identical trip counts,
identical loads, identical accumulator count, identical chain count per loop
body. That step alone is the discriminator. KT=1 additionally allows the
scheduler (or CSE) to interleave the two blocks, so it measures the total
recoverable by loop structure and is NOT the discriminating step.
"""

from __future__ import annotations

import argparse

# --- what E38 measured, on the same host and instrument -----------------------
CONTROL_BAND = 0.0046  # untreated-width ratio band, E38 measured
E38_ARM_A_M6 = 1.1054  # R2 at M=6, NA=3, sequential full-K row blocks
E38_ANCHOR_TOL = 0.010  # cross-session anchor tolerance (E38 held to 0.118 %)
REG_WALL = 128

# AIR peak_live_regs, measured in E38 with research/air_kernel_stats.py.
# Replication of these three is the instrument check: they must come back
# EXACTLY, or the census is not comparable with E38's and the +21/NA law is not
# the same law.
E38_REGS = {
    "xship_na3": 83,
    "xship_na4": 104,
    "xship_na5": 125,
    "xrb_na3_r2": 66,
    "xrb_na6_r2": 117,
}
# The law holds to na5 and BREAKS at na6: steps are +21, +21, +21, +19 against
# an extrapolated 146, and na6 is the only r=4 cell with allocas=2 and the new
# [4 x <6 x float>] type. 144 is already post-spill, so na6's true demand is
# >= 144 and the deviation is the wall becoming visible, not noise in the law.
E38_REGS_R4_LAW = {2: 62, 3: 83, 4: 104, 5: 125, 6: 144}

# --- registered register predictions, before compiling ------------------------
# Only the accumulator array grows: the K-tiled form holds all 4 rows'
# accumulators live across the kt loop instead of 2, and each accumulator is NA
# floats wide, so the delta over the sequential r=2 cell is +2*NA. Everything
# else -- packed, scale/bias, partial, sums, a0..a3 -- stays at one block's
# worth because only one block runs inside a kt iteration.
REG_R2_SEQ = {3: 66, 4: 83, 5: 100, 6: 117}  # 66 and 117 measured; +17/NA between


def predicted_regs_central(na: int, r: int = 2) -> int:
    """Central registered prediction: the r=2 cell plus the second row pair.

    r=1 additionally drops one row's worth of packed/scale/bias/partial from the
    live set, which the r=4-to-r=2 measurements put at ~10 registers.
    """
    return REG_R2_SEQ[na] + 2 * na - (10 if r == 1 else 0)


def predicted_regs(na: int, kt: str, r: int = 2) -> tuple[int, int]:
    """(low, high) registered prediction for a K-tiled cell."""
    base = predicted_regs_central(na, r)
    if kt == "1":
        # CSE or scheduler interleave may hold a0..a3 (4*NA) live across both
        # blocks. Upper bound is the r=4 cell plus the second accumulator pair
        # the K-tiled form needs anyway.
        return base, max(base, E38_REGS_R4_LAW[na] + 6)
    return base - 3, base + 3


REGISTERED_CELLS = {
    # cell name -> (na, r, KT, used in a timed arm?)
    "xkt_na3_r2_kt1": (3, 2, "1", True),
    "xkt_na4_r2_ktall": (4, 2, "all", True),
    "xkt_na4_r2_kt4": (4, 2, "4", True),
    "xkt_na4_r2_kt1": (4, 2, "1", True),
    # priced in advance so deliverable (b) is not a surprise: this is the cell
    # the crown arithmetic needs, one weight pass at M=6.
    "xkt_na6_r2_ktall": (6, 2, "all", False),
    "xkt_na6_r2_kt1": (6, 2, "1", False),
    "xkt_na6_r1_ktall": (6, 1, "all", False),
}

# --- the arm map: which cell each width dispatches in the arm build -----------
ARM_MAP = {
    3: ("xkt_na3_r2_kt1", "NA=3 max locality + adjacency"),
    4: ("xkt_na4_r2_ktall", "NA=4 no locality, no adjacency"),
    6: ("xrb_na3_r2", "E38 arm(a) replication anchor, NA=3 sequential full-K"),
    7: ("xkt_na4_r2_kt4", "NA=4 locality, no adjacency  <-- DISCRIMINATOR"),
    8: ("xkt_na4_r2_kt1", "NA=4 locality + adjacency"),
}
UNTREATED = [1, 2, 5, 9]  # must stay inside CONTROL_BAND; 1 and 2 are the global null

# --- registered timing predictions -------------------------------------------
# Ratios are arm/base at the same M, so anything not in ARM_MAP is a control.
PRED_M6_ANCHOR = (E38_ARM_A_M6 - E38_ANCHOR_TOL, E38_ARM_A_M6 + E38_ANCHOR_TOL)
PRED_KTALL_NA4 = (1.07, 1.12)  # same mechanism as arm(a), one NA up

# The two hypotheses, stated as predictions on the ONE discriminating step.
LOCALITY_STEP_MEM = 0.50  # >= this fraction of R2 recovered by KT=all -> KT=4
LOCALITY_STEP_ILP = 0.10  # <= this fraction recovered, or step inside the band


def locality_recovery(r_ktall: float, r_kt4: float) -> float:
    """Fraction of the row-blocking tax removed by shortening the re-read."""
    tax = r_ktall - 1.0
    return (r_ktall - r_kt4) / tax if tax > 0 else float("nan")


def total_recovery(r_ktall: float, r_kt1: float) -> float:
    tax = r_ktall - 1.0
    return (r_ktall - r_kt1) / tax if tax > 0 else float("nan")


def verdict(r_ktall: float, r_kt4: float, r_kt1: float) -> str:
    """The pre-registered decision rule, applied to the NA=4 ladder."""
    step = r_ktall - r_kt4
    if (r_ktall - r_kt1) <= CONTROL_BAND:
        return "ILP_OR_TILE: no loop-structural recovery at all; K-tiling dead"
    frac = locality_recovery(r_ktall, r_kt4)
    if step <= CONTROL_BAND or frac <= LOCALITY_STEP_ILP:
        return "ILP_OR_TILE: locality step is null; K-tiling dead"
    if frac >= LOCALITY_STEP_MEM:
        return "MEM: locality step carries R2; K-tiling licensed"
    return "DISTRIBUTED: partial ceiling by construction; report the fraction"


# --- value chain, kernel level only ------------------------------------------
# Corrected constants from the E38 merge comment. sensitivity 1.00 for a uniform
# both-leg MTP speedup, valid up to a -0.635 % leg move (medicine saturates
# against essays at raw_p = 3.366118 beyond that).
SCORE_SENSITIVITY = 1.00
CROWN_PCT = 0.5193
GAP_PCT = 0.2586
SIGMA_SCORE_PCT = 0.0978
# psi (QMV share of the candidate leg) is NOT measured and is being measured by
# askeladd. phi (M=6 share of QMV cost) = 0.201 on this fixture. E38's implied
# psi*phi = 0.0459 is back-solved, not measured, so every score number below is
# reported as a conditional, never as a result.
PSI_PHI_BACKSOLVED = 0.0459


def score_pct(m6_ratio: float, psi_phi: float = PSI_PHI_BACKSOLVED) -> float:
    """Conditional score gain (%) for an M=6-only per-row cost ratio."""
    return (1.0 - m6_ratio) * psi_phi * SCORE_SENSITIVITY * 100.0


def self_test() -> None:
    assert abs(locality_recovery(1.1054, 1.0527) - 0.5) < 1e-9
    assert abs(total_recovery(1.10, 1.00) - 1.0) < 1e-9
    assert verdict(1.10, 1.10, 1.10).startswith("ILP_OR_TILE")
    assert verdict(1.10, 1.099, 1.02).startswith("ILP_OR_TILE")
    assert verdict(1.10, 1.02, 1.00).startswith("MEM")
    assert verdict(1.10, 1.075, 1.00).startswith("DISTRIBUTED")
    # a null total recovery outranks a nominal locality step
    assert verdict(1.004, 1.0, 1.0).startswith("ILP_OR_TILE")
    for na in (3, 4):
        lo, hi = predicted_regs(na, "all")
        assert lo <= predicted_regs_central(na) <= hi
        assert hi < REG_WALL, na
    # registered in advance: the cell the crown arithmetic needs sits AT the wall
    assert predicted_regs_central(6) == 129
    assert predicted_regs(6, "all")[1] > REG_WALL
    # and the r=1 fallback is registered as the one that fits
    assert predicted_regs(6, "all", r=1)[1] < REG_WALL
    # E38's own arithmetic, reproduced so a drift in the constants is visible
    assert abs(score_pct(0.9858) - 0.0652) < 0.002
    assert abs(score_pct(0.8804) - 0.549) < 0.01
    print("e41_prereg self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    print("E41 PRE-REGISTRATION\n")
    print("Question: is R2 (+10.54 % at M=6) the activation re-read (MEM) or the")
    print("          halved register tile and extra loop (ILP)?\n")
    print("Discriminating step: KT=all -> KT=4 at NA=4, which changes ONLY the")
    print("re-read distance. KT=1 is the total-recovery bound, not the test.\n")

    print("registered register predictions (AIR peak_live_regs, wall = 128)")
    for name, (na, r, kt, timed) in REGISTERED_CELLS.items():
        lo, hi = predicted_regs(na, kt, r)
        flag = "TIMED" if timed else "priced only"
        wall = "OVER WALL" if lo >= REG_WALL else ("at wall" if hi >= REG_WALL else "fits")
        print(f"  {name:22s} na={na} r={r} KT={kt:3s} predict {lo}-{hi} regs  {wall}  ({flag})")
    print("  hard gate: a timed arm must be <= 128 with no accumulator alloca.")
    print("  a cell that spills is not a discriminator and will not be timed.\n")

    print("arm map (one build; ratio = arm/base at the same M)")
    for m, (cell, why) in sorted(ARM_MAP.items()):
        print(f"  M={m}: {cell:18s} {why}")
    print(f"  untreated controls: M={UNTREATED}, all within +-{CONTROL_BAND*100:.2f} %\n")

    print("registered timing predictions")
    print(f"  M=6 anchor        rho in [{PRED_M6_ANCHOR[0]:.4f}, {PRED_M6_ANCHOR[1]:.4f}]"
          "   (E38 replication; a miss invalidates the session, not the arms)")
    print(f"  M=4 KT=all        rho in [{PRED_KTALL_NA4[0]:.2f}, {PRED_KTALL_NA4[1]:.2f}]")
    print(f"  MEM  predicts     M=7 KT=4 recovers >= {LOCALITY_STEP_MEM:.0%} of the tax")
    print(f"  ILP  predicts     M=7 KT=4 recovers <= {LOCALITY_STEP_ILP:.0%}, or a step"
          f" inside +-{CONTROL_BAND*100:.2f} %")
    print("  both predict M=8 KT=1 recovers; that is why KT=1 cannot be the test\n")

    print("stop rules")
    print("  - any timed cell over 128 regs or with an accumulator alloca: do not time it")
    print("  - any arm changing one parity digest or one emitted token: stop and report")
    print("  - total recovery inside the control band: informative null, K-tiling dead")
    print("  - locality step inside the control band: R2 is not the re-read, K-tiling dead")
    print("  - no E2E leg: predicted move is ~0.07-0.5 %, MDE at n=4 is 0.417 %/0.632 %\n")

    print("value, conditional and kernel-level only")
    print(f"  score % = (1 - rho_M6) * psi*phi * {SCORE_SENSITIVITY:.2f};"
          f" psi*phi = {PSI_PHI_BACKSOLVED} is BACK-SOLVED, not measured")
    print(f"  crown {CROWN_PCT:.4f} %, engineerable gap {GAP_PCT:.4f} %,"
          f" sigma_score {SIGMA_SCORE_PCT:.4f} %")


if __name__ == "__main__":
    main()
