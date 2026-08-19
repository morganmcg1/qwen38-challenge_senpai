#!/usr/bin/env python3
"""What a QMV change is WORTH IN SCORE, and what the register ceiling costs.

WHY THIS FILE EXISTS
--------------------
I computed both of these in `/tmp` and nearly left them there. A number that
decides what students work on does not belong in `/tmp`: it will be lost, or
worse, silently re-derived differently next turn. `research/noise_floors.py`
made the floors a single authority; this does the same for the two leverage
constants, and it imports its floors from there rather than restating them.

THE ONE FACT THAT INVERTS EVERYTHING
------------------------------------
Item 103, verified with 0 mismatches over the corpus:

    raw_p = serial / mtp            <-- SERIAL IS THE NUMERATOR

So making the shared quantized matvec faster speeds up BOTH legs, and because
the serial leg is MORE QMV-bound than the candidate leg, the ratio goes DOWN.
A uniform QMV speedup is a board-visible LOSS. This is not intuitive and it is
why the calculation is written down instead of done in anyone's head.

    d ln(raw_p) / dx = psi_mtp - psi_serial

Usage:
    research/qmv_score_leverage.py            # report
    research/qmv_score_leverage.py selftest   # assert the signs and targets
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from noise_floors import SCORE_BETWEEN_SUBMISSION  # noqa: E402

# --------------------------------------------------------------------------
# MEASURED INPUTS. Each carries its provenance, because a psi from a different
# tree is a different psi.
# --------------------------------------------------------------------------

# askeladd E42, merged 65a73455, measured on tree 04ad6bf1 (E27 present, NA<=5,
# sole stream boundary 5->6). Method: inject a large BIT-EXACT regression (a
# rolled pass loop recomputing an identical accumulation) and divide it out --
# effect size is chosen rather than fought for, which is how it escapes the MDE
# trap. Linearity checked at two doses: p2 ratio 1.0012, p6 0.9980, so one psi
# to 0.12 %. Bit-exactness 6/6 arms, 1152 cells, 0 differing.
PSI_MTP = 0.6736        # QMV share of the CANDIDATE leg, dispatched widths 2..9
PSI_MTP_FLOOR = 0.604   # his conservative floor
PSI_SERIAL = 0.8525     # QMV share of the SERIAL leg, width 1 only

# thorfinn E46, merged 512359f4. Refit T = 16.757 + 27.532*ceil(M/IPG) +
# 9.624*M, max|resid| 0.770 ms against 11.348 ms for an [M>=6] indicator whose
# coefficient came out NEGATIVE. So the cost is stream count, causally: the
# streams contrast is +18.72 % over 8/8 shapes while the group-width contrast
# at fixed streams is null.
STREAM_MS = 27.532
T_BY_WIDTH = {3: 73.455, 4: 82.620, 5: 120.338, 6: 128.794,
              7: 138.848, 8: 149.263, 9: 186.098}

# askeladd E42, dispatched-width histogram on the candidate leg, 78 dispatches.
# 🔴 CORPUS-WIDE. The score is the mean of the 4th and 5th ORDER STATISTICS =
# beagle and medicine only; the other six prompts are worth 0.0000. beagle's
# mean M is 5.533 against this histogram's 7.269, so every weighting below is
# provisional until the per-prompt histogram exists. See `caveats()`.
HIST = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}

# E27, measured. Kernel-wide register max 108 (<T,7,4>) -> 129 (<T,9,5>).
# There is exactly one [[kernel]] and every helper is METAL_FUNC inline
# (alphonse E40), so this allocation is SHARED by every width.
E27_SCORE_PCT = -0.3321
E27_REG_DELTA = 21
E27_WIDTHS_CHANGED = (5, 9)


def leverage(gated):
    """Score % per 1 % QMV cost reduction. `gated` = does it skip M=1?"""
    return PSI_MTP - (0.0 if gated else PSI_SERIAL)


def qmv_share(m):
    """Width m's share of candidate-leg QMV time, corpus-wide."""
    total = sum(HIST[w] * T_BY_WIDTH[w] for w in HIST if w in T_BY_WIDTH)
    return HIST[m] * T_BY_WIDTH[m] / total


def stream_win(m):
    """One stream removed at width m, as a fraction of TOTAL QMV cost."""
    return (STREAM_MS / T_BY_WIDTH[m]) * qmv_share(m)


def target_for(score_pct, gated=True, psi=None):
    """QMV cost reduction needed to move the score by score_pct."""
    lev = (psi if psi is not None else PSI_MTP) - (0.0 if gated else PSI_SERIAL)
    return None if lev <= 0 else score_pct / lev


def e27_residual():
    """Score % left over after crediting E27's local stream wins."""
    wins = sum(stream_win(m) for m in E27_WIDTHS_CHANGED)
    return E27_SCORE_PCT - 100 * wins * PSI_MTP, 100 * wins


def report():
    sd = SCORE_BETWEEN_SUBMISSION.pct
    print("QMV -> SCORE LEVERAGE     (raw_p = serial/mtp; serial is the NUMERATOR)")
    print("  psi_mtp    = %.4f  candidate leg, verify widths 2..9" % PSI_MTP)
    print("  psi_serial = %.4f  serial leg, width 1 only" % PSI_SERIAL)
    print()
    print("  UNIFORM QMV speedup : %+.4f %% of score per 1 %%   <-- NEGATIVE"
          % leverage(False))
    print("  GATED off M=1       : %+.4f %% of score per 1 %%" % leverage(True))
    print()
    print("  A uniform 10 %% QMV win COSTS %.3f %% of score = %.2f sd."
          % (-10 * leverage(False), -10 * leverage(False) / sd))
    print()
    print("  THE GATE IS FREE. Width 1 dispatches qmv_fast_impl; widths 2..9")
    print("  dispatch the crossrow _m family. Different code paths -- so any")
    print("  optimisation confined to _m is ALREADY gated and earns +%.4f %%/%%."
          % leverage(True))
    print()
    print("GATED TARGETS (QMV cost reduction needed)")
    for label, tgt in (("crown gap", 0.5193), ("1 sd", sd), ("2 sd", 2 * sd)):
        print("  %-10s %.4f %% of score : %6.3f %%   (%.3f %% at psi floor %.3f)"
              % (label, tgt, target_for(tgt), target_for(tgt, psi=PSI_MTP_FLOOR),
                 PSI_MTP_FLOOR))
    print()
    print("WHERE THE QMV COST IS (corpus-wide -- see caveats)")
    for m in sorted(HIST):
        if m not in T_BY_WIDTH:
            print("  M=%d  n=%2d  never measured" % (m, HIST[m]))
            continue
        print("  M=%d  n=%2d  T=%7.3f ms  share=%5.1f %%"
              % (m, HIST[m], T_BY_WIDTH[m], 100 * qmv_share(m)))
    print()
    print("E27 DECOMPOSED: two local stream wins minus one shared ceiling step")
    residual, wins_pct = e27_residual()
    for m in E27_WIDTHS_CHANGED:
        print("  local win M=%d : %6.3f %% of QMV cost" % (m, 100 * stream_win(m)))
    print("  total         : %6.3f %% of QMV cost -> %+.3f %% of score expected"
          % (wins_pct, wins_pct * PSI_MTP))
    print("  observed      : %+.4f %% of score" % E27_SCORE_PCT)
    print("  residual      : %+.3f %% of score = %.1f sd  <-- price of +%d registers"
          % (residual, abs(residual) / sd, E27_REG_DELTA))
    print()
    prize = 100 * stream_win(9) * PSI_MTP
    print("  THE PRIZE: a 2-stream M=9 held at <= 108 registers is worth")
    print("  %+.2f %% of score = %.1f sd. Nothing else on the roadmap is within"
          % (prize, prize / sd))
    print("  an order of magnitude; every other lever is BELOW the %.4f %% floor."
          % sd)
    print()
    caveats()


def caveats():
    print("🔴 CAVEATS THAT GATE THE ABOVE")
    print("  (a) HIST is CORPUS-WIDE. Score = mean of the 4th and 5th order")
    print("      statistics = beagle + medicine ONLY (other six worth 0.0000).")
    print("      beagle's mean M is 5.533 vs the corpus 7.269. If their mixes")
    print("      are less M=9-heavy, the M=9 prize is overstated FOR THE ONLY")
    print("      TWO PROMPTS THAT SCORE.")
    print("      FALSIFIER: measure the per-width histogram for beagle and")
    print("      medicine separately. Cheap, needs no timing precision, and it")
    print("      must run before anyone hunts 21 registers.")
    print("  (b) T_BY_WIDTH and STREAM_MS are microbenchmark numbers. Only the")
    print("      RATIO transfers, and only if QMV time per dispatch really is")
    print("      proportional to T(M).")
    print("  (c) The E27 residual assumes the ENTIRE shortfall is the register")
    print("      step. E27 shipped more than two IPG cells.")
    print("  (d) Occupancy is a STEP function. '%.3f %% of score per register'"
          % (abs(e27_residual()[0]) / E27_REG_DELTA))
    print("      is arithmetic, not physics -- meaningful only if 108 and 129")
    print("      straddle exactly one boundary. thorfinn found")
    print("      maxTotalThreadsPerThreadgroup saturated at 1024 in both arms,")
    print("      so our occupancy instrument cannot currently see the boundary.")


def selftest():
    bad = []

    def ck(name, cond, detail=""):
        print("%-4s %s%s" % ("PASS" if cond else "FAIL", name,
                             "" if cond else "   " + detail))
        if not cond:
            bad.append(name)

    # THE headline sign. If this ever flips, the campaign's direction flips.
    ck("uniform QMV leverage is NEGATIVE", leverage(False) < 0,
       "got %+.4f" % leverage(False))
    ck("gated QMV leverage is POSITIVE", leverage(True) > 0)
    ck("gated leverage equals psi_mtp exactly", leverage(True) == PSI_MTP)
    ck("serial leg is MORE QMV-bound than candidate", PSI_SERIAL > PSI_MTP)

    # Targets, pinned so a silent edit to psi is caught.
    ck("1 sd target is 1.140 %", abs(target_for(SCORE_BETWEEN_SUBMISSION.pct) - 1.1398) < 2e-3,
       "got %.4f" % target_for(SCORE_BETWEEN_SUBMISSION.pct))
    ck("crown-gap target is 0.771 %", abs(target_for(0.5193) - 0.7709) < 2e-3)

    # A lower psi must make the target HARDER, never easier.
    ck("a smaller psi raises the required reduction",
       target_for(1.0, psi=PSI_MTP_FLOOR) > target_for(1.0))

    # The floor must come from the between-submission component. Dividing an
    # effect by the within-run floor is the error of item 172.
    ck("floor used is between-submission",
       SCORE_BETWEEN_SUBMISSION.component == "between-submission",
       SCORE_BETWEEN_SUBMISSION.component)

    # Shares must be a partition of the measured widths.
    total = sum(qmv_share(m) for m in HIST if m in T_BY_WIDTH)
    ck("measured-width shares sum to 1", abs(total - 1.0) < 1e-9,
       "got %.6f" % total)

    # M=9 must dominate, or the prize framing is wrong.
    ck("M=9 is the largest single share",
       max((qmv_share(m), m) for m in HIST if m in T_BY_WIDTH)[1] == 9)

    # E27 residual must be negative and large, or item 173(C) is misstated.
    residual, _ = e27_residual()
    ck("E27 residual is a large NEGATIVE", residual < -1.0, "got %+.3f" % residual)

    # CONSTRUCTED INPUT: a zero-cost change must be worth exactly zero, and a
    # win at a width that is never dispatched must be worth exactly zero.
    ck("zero QMV reduction is worth zero score", target_for(0.0) == 0.0)
    saved = HIST.get(9)
    try:
        HIST[9] = 0
        ck("a width dispatched zero times has zero share", qmv_share(9) == 0.0)
        ck("... and removing its stream is worth nothing", stream_win(9) == 0.0)
    finally:
        HIST[9] = saved
    ck("the constructed mutation was reverted", HIST[9] == 34)

    print()
    if bad:
        print("SELFTEST FAILED: %d case(s): %s" % (len(bad), bad))
        return 1
    print("SELFTEST PASSED: uniform sign negative, gated targets 1.140 %% / "
          "0.771 %%, M=9 dominant, E27 residual large and negative.")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "selftest":
        sys.exit(selftest())
    if mode == "report":
        report()
        sys.exit(0)
    print("usage: %s [report|selftest]" % sys.argv[0])
    sys.exit(2)
