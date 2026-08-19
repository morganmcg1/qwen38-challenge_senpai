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

# 🔴🔴🔴 PSI_SERIAL IS A LOCAL-HARNESS QUANTITY AND IT DOES NOT ENTER THE RANKED
# SCORE. Ledger 176 / edward E50 (merged 26fd0ac). It is retained here ONLY so
# the local two-leg ratio can still be predicted; every ranked pricing path must
# ignore it. If you find yourself subtracting it to price a submission, stop.
#
# WHY. The ranked harness times TWO DIFFERENT BINARIES:
#   .github/workflows/qwen-mtp-ranked-benchmark.yml, above the timed step --
#     "baseline:  pinned baseline tree, serial K=1 target decode ...
#      candidate: this workspace, native-MTP speculative decode"
#   the invocation --  --candidate "${MLXFAST_JOB_WS}"
#                      --baseline  "${MLXFAST_QWEN_MTP_BASELINE_RESOLVED}"
#   the precondition -- test -d "${BASELINE_RESOLVED}/.build/release" || exit 1
#   the scorer       -- .aggregate.baseline_serial_seconds_per_token_mean
#                     / .aggregate.candidate_mtp_seconds_per_token_mean
#   MLXFAST_QWEN_MTP_BASELINE_WS = /opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current
#   and benchmark.json's editablePaths has NO .github entry.
# So d ln(serial)/dx == 0 for every x we can edit, BY CONSTRUCTION.
#
# HOW IT GOT INTO THE MODEL. askeladd's E42 injection was run on the local
# harness, where senpai/program.md:156 says: "Both local legs also use the same
# candidate build. ... a general target or kernel improvement may speed both legs
# and cancel in that ratio." The 0.8525 is that cancellation. The measurement was
# correct; promoting it to a ranked score model was not. Item 103 verified the
# IDENTITY raw_p = serial/mtp with 0 mismatches, and the identity was then
# differentiated without checking which term is a function of x. An identity
# between two measured quantities says nothing about which one your edits move.
PSI_SERIAL = 0.8525     # LOCAL-ONLY. QMV share of the local serial leg, width 1.

# The two harnesses price the SAME physical change differently, so every pricing
# entry point takes this explicitly rather than defaulting silently to a model.
HARNESS_RANKED = "ranked"   # scored submissions: serial leg is a pinned binary
HARNESS_LOCAL = "local"     # --local-iterate: both legs from the candidate build
_HARNESSES = (HARNESS_RANKED, HARNESS_LOCAL)


def _leg_coupling(harness):
    """d ln(serial)/d ln(qmv cost) for a change that DOES reach width 1.

    Ranked: 0, because the serial leg is a separately built pinned tree.
    Local:  PSI_SERIAL, because both legs are the same binary.
    """
    if harness not in _HARNESSES:
        raise ValueError("harness must be one of %r, got %r" % (_HARNESSES, harness))
    return 0.0 if harness == HARNESS_RANKED else PSI_SERIAL

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

# ---------------------------------------------------------------------------
# ORDER STATISTICS AND THE SUBSTITUTION KINK                     (ledger 177)
# ---------------------------------------------------------------------------
# Everything above converts %QMV-cost into %score at a CONSTANT rate. That is
# only true while the scored pair keeps its membership. The score is the mean of
# the 4th and 5th order statistics of eight per-prompt raw speedups, so a
# mechanism that lifts only the scored prompts eventually pushes one of them out
# of the pair, and from that point its marginal value COLLAPSES.
#
# Measured, not assumed. Source: `.mlxfast-private/ranked-telemetry.json`,
# crown submission solverUsername=ofou, officialScore 3.24929398547457,
# createdAt 2026-08-18T21:48:43, submissionCommitSha ef42e0432727.
#
# The scored field is the RAW median. Verified across all 411 metrics-bearing
# board rows: officialScore == officialMetrics.mtp_decode_speedup_raw_median
# with ZERO mismatches. `decode_speedup_ceiling` (which the organizers raised
# 3 -> 5 at 2026-08-17T11:10:46Z, and which 136 submissions touched from below)
# clips only the cosmetic `mtp_decode_speedup_median` field. It never clips the
# score, so it is not a confound for any board-derived noise floor.
# `raw_ratio_of_means` is published rounded to 10 decimals while officialScore
# carries full precision, so the reconstruction agrees only to ~2.5e-11. That
# residual is the board's own rounding, not a modelling error -- do not "fix" it
# by fitting a fudge term.
CROWN_ORDER_STATS = (
    ("plutarch", 1.2560334838),   # rank 1   nd = 449 -- see NON_DRAFTING note
    ("drama",    1.9231089575),   # rank 2
    ("travel",   2.1895159531),   # rank 3
    ("beagle",   3.1433255794),   # rank 4  <-- SCORED
    ("medicine", 3.3552623916),   # rank 5  <-- SCORED
    ("essays",   3.3906635754),   # rank 6  <-- THE SUBSTITUTE, +1.055 % over medicine
    ("botany",   3.4143725007),   # rank 7
    ("republic", 3.4490615187),   # rank 8
)
CROWN_SCORE = 3.24929398547457
SCORED_PROMPTS = ("beagle", "medicine")

# 🔴 `non_drafting_round_count` is NOT 0 on the candidate. It is 0 on the seven
# high-acceptance prompts and 449 on plutarch, on BOTH the crown tree and our
# own 3.23250848263467 tree, and nonzero on plutarch in 320 of 371 healthy board
# rows (mode 449, n=303). The ledger's flat "nd = 0" was a generalisation of a
# measurement taken on beagle/medicine. The CONCLUSION it supported -- that the
# 4-bit width-1 verifier kernel is worth about zero -- survives, but for a
# different and stronger reason: plutarch's raw speedup is 1.2528 against a
# rank-4 value of 3.1202, so it would have to improve by +149 % to acquire any
# marginal weight at all. The reason is the ORDER STATISTICS, which are pinned
# in the non-editable workflow, not a property of the candidate binary.
NON_DRAFTING_ROUNDS = {"plutarch": 449}


def _median_of_eight(values):
    """Mean of the two central order statistics. The organizers' rule, verbatim:
    officialMetrics.median_rule == 'even_n_mean_of_two_central_order_statistics'
    (distinct=1 across all 371 healthy board rows)."""
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def score_from_leg_gains(gains_pct, stats=CROWN_ORDER_STATS):
    """Score after applying per-prompt percentage speedups to the raw ratios.

    The kink is COMPUTED, by re-sorting, not modelled by a piecewise formula.
    That is deliberate: ledger 176(D) burned eight days on an analytic
    coefficient that was never stress-tested at its sign extremes. A re-sort
    cannot have the wrong sign anywhere, so there is nothing to get wrong.
    """
    for name in gains_pct:
        if name not in dict(stats):
            raise KeyError("unknown prompt %r; expected one of %r"
                           % (name, tuple(n for n, _ in stats)))
    return _median_of_eight([r * (1.0 + (gains_pct[n] if n in gains_pct else 0.0) / 100.0)
                             for n, r in stats])


def score_pct_from_leg_gains(gains_pct, stats=CROWN_ORDER_STATS):
    """Percentage score change from per-prompt percentage leg speedups."""
    base = score_from_leg_gains({}, stats)
    return 100.0 * (score_from_leg_gains(gains_pct, stats) / base - 1.0)


def marginal_weights(stats=CROWN_ORDER_STATS):
    """{prompt: % of score per 1 % leg speedup} for the two scored prompts.

    d score / d ratio = 1/2 for each member of the pair, so the weight is
    (ratio / 2) / score -- LARGER for the faster member. These come out near
    0.484 and 0.516, i.e. very nearly equal.

    🔴 Do not confuse these with the ledger's "beagle 79 % / medicine 21 %".
    That pair is the split of ONE mechanism's value (E40's width-deficit
    closure, whose per-prompt LEG effects were +0.363 % and +0.088 %, a 4.1x
    difference). It is an effect split, not a score weight. Multiplying by it
    as if it were a weight double-counts the heterogeneity.
    """
    base = score_from_leg_gains({}, stats)
    ranked = sorted(stats, key=lambda t: t[1])
    return {name: 0.5 * ratio / base for name, ratio in (ranked[3], ranked[4])}


def substitution_headroom(prompt, stats=CROWN_ORDER_STATS):
    """% leg speedup `prompt` can absorb before it stops paying at full rate.

    A scored prompt keeps paying while it stays inside the pair. Passing the
    OTHER member is harmless -- they simply swap ranks 4 and 5. Passing rank 6
    ejects it, and from there its marginal value is exactly zero.
    """
    ranked = sorted(stats, key=lambda t: t[1])
    names = [n for n, _ in ranked]
    if prompt not in names:
        raise KeyError("unknown prompt %r" % (prompt,))
    i = names.index(prompt)
    if i not in (3, 4):
        return 0.0        # not in the scored pair: no rate to lose
    return 100.0 * (ranked[5][1] / ranked[i][1] - 1.0)


def saturation_cap_pct(stats=CROWN_ORDER_STATS):
    """Max % score obtainable from an arbitrarily large gain on the scored pair.

    Once both scored prompts overtake ranks 6 and 7, the pair becomes those two
    and further gains are worth nothing. This is a HARD ceiling on every
    beagle/medicine-only mechanism in the campaign, however good it is.
    """
    ranked = sorted(stats, key=lambda t: t[1])
    base = score_from_leg_gains({}, stats)
    return 100.0 * (0.5 * (ranked[5][1] + ranked[6][1]) / base - 1.0)


def kink_pct(stats=CROWN_ORDER_STATS):
    """Uniform scored-pair leg gain at which the marginal rate first drops.

    Below it, a uniform gain on the pair converts 1:1 into score. Above it, only
    the rank-4 prompt still pays, at roughly half the rate.
    """
    return min(substitution_headroom(p, stats) for p in SCORED_PROMPTS)


def leverage(gated, psi_mtp_w1=0.0, harness=HARNESS_RANKED):
    """Score % per 1 % QMV cost reduction. `gated` = does it skip M=1?

    🔴 RANKED (what a submission scores). The serial leg is a pinned separate
    binary, so it cannot respond to anything we edit:

      GATED   (widths 2..9) = PSI_MTP
      UNIFORM (all widths)  = PSI_MTP + psi_mtp_w1        <-- POSITIVE, always

    Both are positive and they differ only by the candidate leg's OWN width-1
    share. Gating therefore buys **exactly zero** on ranked when psi_mtp_w1 = 0,
    and is strictly WORSE than not gating when psi_mtp_w1 > 0. Item 173(B)'s
    "free gate" was free because it was empty. Retired in ledger 176.

    🔴 LOCAL (what --local-iterate's serial-to-MTP ratio shows). Both legs are
    the same build, so a change reaching width 1 moves the numerator too:

      GATED   (widths 2..9) = PSI_MTP                     -- same as ranked
      UNIFORM (all widths)  = PSI_MTP + psi_mtp_w1 - PSI_SERIAL

    GATED IS HARNESS-INVARIANT. That is why every gated price in this campaign
    survived the E50 correction, including alphonse's merged E44 r2. Only the
    uniform family was mispriced, and it was mispriced in both sign and size.

    `psi_mtp_w1` is the CANDIDATE leg's own width-1 QMV share and is still NOT
    MEASURED. PSI_MTP = 0.6736 was injected into widths 2..9 only, so the
    candidate leg's TOTAL QMV share is PSI_MTP + psi_mtp_w1.

    Verified from source (kernels/quantized.h): the 4-bit switch at :1917 has
    cases 2..9 and NO case 1, so width-1 4-bit falls through to qmv_fast_impl at
    :2026. But :1908 dispatches a width-1 2-bit coarse DRAFT readout
    (out_vec_size == 98336) on the candidate leg, so "width 1 implies serial leg"
    is not a theorem. What HAS since been pinned is askeladd's E42
    non_drafting_round_count = 0 (research/e42_width_census.py:16,
    research/e42-results.md:749): the candidate runs no verifier-side width-1
    rounds at all, so psi_mtp_w1 is carried entirely by that 2-bit draft readout,
    which fires mtp_depth = 8 times per round. Still needs measuring; E48 (PR 52)
    carries it, no longer as a sign correction but as the entire score value of
    the width-1 path.
    """
    # Validate UNCONDITIONALLY, before the branch. A typo'd harness on the gated
    # path must not sail through just because the coupling term happens to be
    # unused there -- a checker that only fires on one branch is not a checker.
    coupling = _leg_coupling(harness)
    if gated:
        # A gated change cannot reach width 1 in EITHER harness, so the serial
        # leg is untouched either way and the coupling term does not apply.
        return PSI_MTP
    return PSI_MTP + psi_mtp_w1 - coupling


def qmv_share(m):
    """Width m's share of candidate-leg QMV time, corpus-wide."""
    total = sum(HIST[w] * T_BY_WIDTH[w] for w in HIST if w in T_BY_WIDTH)
    return HIST[m] * T_BY_WIDTH[m] / total


def stream_win(m):
    """One stream removed at width m, as a fraction of TOTAL QMV cost."""
    return (STREAM_MS / T_BY_WIDTH[m]) * qmv_share(m)


def width_set_share(widths):
    """Share of candidate-leg QMV cost carried by a SET of widths.

    This is alphonse's `f`: the quantity that converts a per-width speedup into
    a score. He pre-registered a sensitivity table over it and correctly
    refused to predict it, because E43 left the depth mixture unidentified.
    Corpus-wide it is computable from HIST; per-prompt it is not, which is
    caveat (a) and the reason E48 exists.
    """
    return sum(qmv_share(m) for m in widths if m in T_BY_WIDTH)


def _lev(gated, psi, harness, psi_mtp_w1=0.0):
    """Shared %score-per-%QMV coefficient for the pricing entry points.

    One definition so the ranked/local distinction cannot drift between them.
    """
    base = psi if psi is not None else PSI_MTP
    coupling = _leg_coupling(harness)        # validates harness on BOTH branches
    if gated:
        return base
    return base + psi_mtp_w1 - coupling


def mechanism_value(widths, win_pct, gated=True, psi=None,
                    harness=HARNESS_RANKED, psi_mtp_w1=0.0):
    """Score % from removing win_pct of QMV cost AT `widths` only.

    Gated by construction whenever 1 not in widths, and gated pricing is
    IDENTICAL in both harnesses: a change confined to widths 2..9 cannot reach
    the serial leg whether that leg is our own build or the pinned baseline.

    🔴 An UNGATED change is no longer automatically worse. On ranked it is worth
    `psi + psi_mtp_w1` -- strictly MORE than gated. Only on the local harness does
    it pay the PSI_SERIAL cancellation. Ledger 176; see `leverage.__doc__`.
    """
    if 1 in widths and gated:
        raise ValueError("a change reaching width 1 is NOT gated; pass gated=False")
    return _lev(gated, psi, harness, psi_mtp_w1) * width_set_share(widths) * win_pct


def mechanism_value_per_width(wins, gated=True, psi=None,
                              harness=HARNESS_RANKED, psi_mtp_w1=0.0):
    """Score % from a per-width map {M: win_pct}. WEIGHT FIRST, THEN SUM.

    This is the CORRECT conversion and the only one this module will offer for a
    non-uniform win. `mechanism_value(widths, pooled_win)` is a special case that
    happens to be right only when the win is genuinely flat across `widths`.
    """
    if 1 in wins and gated:
        raise ValueError("a change reaching width 1 is NOT gated; pass gated=False")
    lev = _lev(gated, psi, harness, psi_mtp_w1)
    return lev * sum(qmv_share(m) * w for m, w in wins.items() if m in T_BY_WIDTH)


def pooling_bias(wins, gated=True, psi=None,
                 harness=HARNESS_RANKED, psi_mtp_w1=0.0):
    """POOL-THEN-WEIGHT minus WEIGHT-THEN-SUM. alphonse, E44 r2.

    I asked him to report per width to preserve USEFULNESS after a census
    landed. He showed it preserves CORRECTNESS: pooling first gives the wrong
    answer, not a coarser one.

        pool then weight:   psi * mean(win) * sum(share)
        weight then sum:    psi * sum(share * win)
        difference        =  -psi * n * Cov_n(share, win)

    where Cov_n is the population covariance over the pooled widths. So the sign
    of the bias is the sign of -Cov(share, win): if the widths that win MORE also
    cost MORE, pooling UNDERSTATES. That is the normal case, because both the win
    and the cost tend to grow with the work done per dispatch -- E44's M=8 wins
    +14.85 % against M=7's +7.99 % while carrying 7.55 % of census cost against
    4.71 %.

    Returns (pooled, per_width, bias). `bias < 0` means pooling understates.

    This is not an E44 fact. It applies to EVERY %cost -> %score conversion in
    this campaign, and it is the reason `mechanism_value_per_width` exists.
    """
    widths = [m for m in wins if m in T_BY_WIDTH]
    if not widths:
        return 0.0, 0.0, 0.0
    pooled_win = sum(wins[m] for m in widths) / len(widths)
    pooled = mechanism_value(widths, pooled_win, gated=gated, psi=psi,
                             harness=harness, psi_mtp_w1=psi_mtp_w1)
    per_width = mechanism_value_per_width(
        {m: wins[m] for m in widths}, gated=gated, psi=psi,
        harness=harness, psi_mtp_w1=psi_mtp_w1
    )
    return pooled, per_width, pooled - per_width


def target_for(score_pct, gated=True, psi=None,
               harness=HARNESS_RANKED, psi_mtp_w1=0.0):
    """QMV cost reduction needed to move the score by score_pct.

    Returns None when the target is UNREACHABLE, for either of two reasons:

      1. the leverage is non-positive -- under the LOCAL harness an ungated
         mechanism with psi_mtp_w1 < PSI_SERIAL - PSI_MTP has negative leverage.
         Under RANKED this case does not arise, leverage is always positive.
      2. 🔴 the target exceeds `saturation_cap_pct()`. A QMV mechanism acts on
         the scored prompts, and once they leave the 4th/5th order statistics no
         further speedup is worth anything. Returning a finite cost reduction
         for an unreachable score would be a tool that fails open toward an
         encouraging number, which is the most dangerous kind (ledger 175).
    """
    lev = _lev(gated, psi, harness, psi_mtp_w1)
    if lev <= 0:
        return None
    if score_pct > saturation_cap_pct():
        return None
    return score_pct / lev


def e27_residual():
    """Score % left over after crediting E27's local stream wins."""
    wins = sum(stream_win(m) for m in E27_WIDTHS_CHANGED)
    return E27_SCORE_PCT - 100 * wins * PSI_MTP, 100 * wins


def report():
    sd = SCORE_BETWEEN_SUBMISSION.pct
    print("QMV -> SCORE LEVERAGE     (raw_p = serial/mtp; serial is the NUMERATOR)")
    print("  psi_mtp    = %.4f  candidate leg, verify widths 2..9" % PSI_MTP)
    print("  psi_serial = %.4f  LOCAL ONLY -- not in the ranked score" % PSI_SERIAL)
    print()
    print("  🔴 THE RANKED SERIAL LEG IS A PINNED SEPARATE BINARY (ledger 176).")
    print("     .github/.../qwen-mtp-ranked-benchmark.yml times --baseline (a")
    print("     prebuilt tree at /opt/bench-runner/baseline/...) against")
    print("     --candidate (this workspace), and scores")
    print("       baseline_serial_seconds_per_token_mean")
    print("     / candidate_mtp_seconds_per_token_mean.")
    print("     editablePaths has no .github entry. So d ln(serial)/dx == 0 for")
    print("     everything we can edit, and psi_serial cannot appear in a price.")
    print()
    print("                        RANKED (scored)    LOCAL (--local-iterate)")
    print("  UNIFORM QMV speedup : %+.4f %%/%%         %+.4f %%/%%"
          % (leverage(False), leverage(False, harness=HARNESS_LOCAL)))
    print("  GATED off M=1       : %+.4f %%/%%         %+.4f %%/%%"
          % (leverage(True), leverage(True, harness=HARNESS_LOCAL)))
    print()
    print("  A uniform 10 %% QMV win EARNS %.3f %% of score = %.2f sd on ranked,"
          % (10 * leverage(False), 10 * leverage(False) / sd))
    print("  while APPEARING to cost %.3f %% in the local ratio. If you optimise"
          % (-10 * leverage(False, harness=HARNESS_LOCAL)))
    print("  against the local ratio you will reject your best changes.")
    print("  senpai/program.md:156 -- 'a general target or kernel improvement may")
    print("  speed both legs and cancel in that ratio'. Compare ABSOLUTE candidate")
    print("  seconds per token against a fresh unchanged BASE_SHA run instead.")
    print()
    print("  🔴 THE GATE IS FREE BUT IT IS ALSO EMPTY (item 173(B) retired).")
    print("  Width 1 dispatches qmv_fast_impl; widths 2..9 dispatch the crossrow")
    print("  _m family, so _m work is already gated and earns +%.4f %%/%%."
          % leverage(True))
    print("  But gating buys NOTHING extra on ranked: the numerator is pinned")
    print("  either way. Gate for risk containment if you like -- never for score.")
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
    f78, e44 = e44_narrow()
    print()
    print("E44 r2 NARROW VARIANT, M in {7,8}   (alphonse, PR 49) -- GATED")
    print("  f{7,8} = %.4f of candidate-leg QMV cost, corpus-wide" % f78)
    for k in ("mlp_down only", "equal shape mix", "attn_out only"):
        print("    %-16s  %+.4f %% of score   %5.2f sd   %4.2fx crown gap"
              % (k, e44[k], e44[k] / sd, e44[k] / 0.5193))
    print("  🔴 The FIRST buildable item this campaign has priced above the")
    print("     %.4f %% board floor. Already measured faster at exactly these" % sd)
    print("     widths, and bit-exact. Its whole value rides on f, which is")
    print("     CORPUS-WIDE here -- see caveat (a), and E48 Part 1.")
    print()
    print("ORDER STATISTICS: THE SUBSTITUTION KINK   (measured, ledger 177)")
    print("  rank  prompt      raw_ratio   nd     marginal %/1% leg gain")
    mw = marginal_weights()
    for i, (name, ratio) in enumerate(sorted(CROWN_ORDER_STATS, key=lambda t: t[1]), 1):
        nd = NON_DRAFTING_ROUNDS[name] if name in NON_DRAFTING_ROUNDS else 0
        tag = "  <== SCORED  %+.4f" % mw[name] if name in mw else ""
        print("   %d    %-10s %10.6f  %4d%s" % (i, name, ratio, nd, tag))
    print("  KINK      +%.4f %% uniform gain on the scored pair. Below it the"
          % kink_pct())
    print("            conversion is exactly 1:1. Above it %s leaves the pair,"
          % min(SCORED_PROMPTS, key=lambda p: substitution_headroom(p)))
    print("            essays substitutes, and the marginal rate roughly HALVES.")
    print("  CAP       +%.4f %% -- the most any beagle/medicine-only mechanism"
          % saturation_cap_pct())
    print("            can ever be worth, at any size. target_for() returns None")
    print("            above this rather than a comforting finite number.")
    print("  🔴 The crown gap is 0.5193 %, BELOW the kink, so closing it is")
    print("     still 1:1. But any dScore claim above +%.3f %% must be" % kink_pct())
    print("     re-derived piecewise with score_pct_from_leg_gains(), including")
    print("     the top of E44 r2's +0.789..+1.228 % range and E49's ceiling.")
    print()
    caveats()


def e44_narrow():
    """Price alphonse's E44 r2 narrow simdgroup-matrix variant, M in {7,8}.

    r1 (023a3fcf) measured the SAME cell body faster at exactly these widths and
    bit-exact (20/20 lines, worst_abs 0.0 over 778,567,680 elements/arm):
        attn_out  M7 +10.46 %   M8 +16.65 %
        mlp_down  M7  +4.46 %   M8 +13.05 %
    It was catastrophic at M=4 (-41.7 / -52.4 %), which is why only the narrow
    range survived. Reported per (width, shape) so this can be re-weighted once
    a per-(width, shape) census exists; pooling would destroy that.
    """
    wins = {("attn_out", 7): 10.46, ("attn_out", 8): 16.65,
            ("mlp_down", 7): 4.46, ("mlp_down", 8): 13.05}
    f = width_set_share((7, 8))
    out = {}
    for label, sel in (("attn_out only", lambda k: k[0] == "attn_out"),
                       ("mlp_down only", lambda k: k[0] == "mlp_down"),
                       ("equal shape mix", lambda k: True)):
        v = [wins[k] for k in wins if sel(k)]
        out[label] = mechanism_value((7, 8), sum(v) / len(v))
    return f, out


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
    print("  (e) E44's win % was measured on a tree whose kernel-wide max was")
    print("      89; the bankable variant leaves _m<T,4,4>=104 instantiated, so")
    print("      the SAME cell runs under a 16.9 % larger shared allocation.")
    print("      alphonse pre-registered that transfer loss at ~+0.16 % of cell")
    print("      time, falsified if any of the four cells flips sign or falls")
    print("      below +2 %. Until r2 lands, e44_narrow() prices a cell body")
    print("      measured in a DIFFERENT allocation regime -- caveat, not a")
    print("      correction, but it is the reason r2 exists.")
    print("      Note the shape spread (0.7279 -> 1.1270) is nearly as wide as")
    print("      the board floor itself, so the SHAPE census matters almost as")
    print("      much as the width census. Neither exists yet.")


def selftest():
    bad = []

    def ck(name, cond, detail=""):
        print("%-4s %s%s" % ("PASS" if cond else "FAIL", name,
                             "" if cond else "   " + detail))
        if not cond:
            bad.append(name)

    def _raises(fn, exc):
        """True iff fn() raises exc. A tool that silently accepts a typo'd
        prompt name and returns 0.0 is a tool that fails open toward a null."""
        try:
            fn()
        except exc:
            return True
        return False

    # 🔴 THE headline sign, CORRECTED in ledger 176 (edward E50, merged 26fd0ac).
    # These three assertions previously read "uniform QMV leverage is NEGATIVE",
    # "uniform sign flips once width-1 candidate QMV reaches 0.1789" and "ungated
    # is worth less than gated". All three were consequences of pricing a ranked
    # submission with PSI_SERIAL, which does not enter the ranked score at all.
    # They are kept here INVERTED rather than deleted, so that anyone who
    # reintroduces the old model gets a red gate instead of a plausible number.
    ck("RANKED uniform QMV leverage is POSITIVE", leverage(False) > 0,
       "got %+.4f" % leverage(False))
    ck("RANKED uniform equals psi_mtp when psi_mtp_w1 = 0",
       leverage(False) == PSI_MTP, "got %+.4f" % leverage(False))
    ck("LOCAL uniform QMV leverage is NEGATIVE",
       leverage(False, harness=HARNESS_LOCAL) < 0,
       "got %+.4f" % leverage(False, harness=HARNESS_LOCAL))
    ck("gated QMV leverage is POSITIVE", leverage(True) > 0)
    ck("gated leverage equals psi_mtp exactly", leverage(True) == PSI_MTP)
    # 🔴 THE VACUITY OF THE FREE GATE, encoded. On ranked, gating a change that
    # would otherwise reach width 1 buys exactly nothing when psi_mtp_w1 = 0, and
    # is strictly WORSE than not gating when it is positive. Item 173(B) retired.
    ck("RANKED gating buys exactly zero at psi_mtp_w1 = 0",
       leverage(True) == leverage(False))
    ck("RANKED gating is a LOSS when the candidate has width-1 QMV",
       leverage(False, psi_mtp_w1=0.30) > leverage(True))
    # GATED IS HARNESS-INVARIANT. This is why every gated price in the campaign
    # survived the correction, including alphonse's merged E44 r2.
    ck("gated leverage is identical in both harnesses",
       leverage(True, harness=HARNESS_RANKED)
       == leverage(True, harness=HARNESS_LOCAL))
    ck("serial leg is MORE QMV-bound than candidate LOCALLY", PSI_SERIAL > PSI_MTP)
    # A typo'd harness must raise on BOTH branches, not just the one that reads
    # the coupling term. A checker that fires on one branch is not a checker.
    for _g in (True, False):
        _raised = False
        try:
            leverage(_g, harness="rankd")
        except ValueError:
            _raised = True
        ck("an unknown harness raises (gated=%s)" % _g, _raised)

    # 🔴 The unmeasured quantity. GATED must be invariant to it; UNIFORM must
    # not be. If someone ever "simplifies" leverage() so the gated branch reads
    # psi_mtp_w1, every target in this file silently acquires a dependency on a
    # number nobody has measured.
    ck("gated leverage is INVARIANT to the candidate leg's width-1 QMV share",
       leverage(True, 0.0) == leverage(True, 0.30) == PSI_MTP)
    ck("uniform leverage DOES depend on it",
       leverage(False, 0.30) > leverage(False, 0.0))
    # 🔴 The sign flip is now a LOCAL-harness phenomenon only. On ranked there is
    # no sign to flip: uniform leverage is PSI_MTP + psi_mtp_w1, positive for every
    # non-negative psi_mtp_w1, so no threshold exists.
    ck("LOCAL uniform sign flips once width-1 candidate QMV reaches 0.1789",
       leverage(False, 0.0, HARNESS_LOCAL) < 0
       <= leverage(False, 0.1789, HARNESS_LOCAL),
       "%.4f -> %.4f" % (leverage(False, 0.0, HARNESS_LOCAL),
                         leverage(False, 0.1789, HARNESS_LOCAL)))
    ck("the LOCAL flip threshold is exactly the uniform gap",
       abs((PSI_SERIAL - PSI_MTP) - 0.1789) < 1e-9)
    ck("RANKED uniform leverage has NO sign flip to find",
       all(leverage(False, w / 100.0) > 0 for w in range(0, 101)))

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

    # ------------------------------------------------------------------
    # POOLING BIAS -- alphonse, E44 r2. Pooling before weighting is BIASED,
    # not merely coarse, and the sign is the sign of -Cov(share, win).
    # ------------------------------------------------------------------
    e44 = {7: 7.9925, 8: 14.8495}          # his per-M means over both shapes
    pooled, per_width, bias = pooling_bias(e44)
    ck("E44: pooling UNDERSTATES because the bigger win is the costlier width",
       bias < 0, "bias %+.6f" % bias)
    ck("E44: per-width value exceeds the pooled value",
       per_width > pooled, "%.6f vs %.6f" % (per_width, pooled))
    # M=8 wins more AND costs more => positive covariance => negative bias.
    ck("E44: the win and the cost share are positively correlated",
       (e44[8] - e44[7]) * (qmv_share(8) - qmv_share(7)) > 0)

    # A FLAT win must have exactly zero bias. If pooling ever disagrees with
    # weighting on a flat win, one of the two conversions is wrong.
    flat = {7: 10.0, 8: 10.0}
    _, _, flat_bias = pooling_bias(flat)
    ck("a flat win has ZERO pooling bias", abs(flat_bias) < 1e-12,
       "got %+.3e" % flat_bias)

    # And the sign must INVERT when the cheap width is the big winner. This is
    # the case that proves the bias is covariance and not an artefact of always
    # rounding one way.
    inverted = {7: 14.8495, 8: 7.9925}
    _, _, inv_bias = pooling_bias(inverted)
    ck("swapping the wins inverts the bias sign (it is a covariance)",
       inv_bias > 0 > bias, "%+.6f vs %+.6f" % (inv_bias, bias))
    ck("the two biases are equal and opposite", abs(inv_bias + bias) < 1e-12)

    # The per-width conversion must inherit the width-1 guard. A per-width map
    # that touches the serial leg is not a gated mechanism.
    try:
        mechanism_value_per_width({1: 10.0, 7: 10.0})
        ck("per-width conversion refuses to price width 1 as gated", False,
           "no ValueError raised")
    except ValueError:
        ck("per-width conversion refuses to price width 1 as gated", True)

    # Cross-check against alphonse's independent script: equal shape mix over
    # M in {7,8} at his census gives +1.009 %, pooled gives +0.942 %.
    ck("E44 per-width lands near his published +1.009 %",
       abs(per_width - 1.009) < 0.06, "got %+.4f" % per_width)
    ck("E44 pooled lands near his published +0.942 %",
       abs(pooled - 0.942) < 0.06, "got %+.4f" % pooled)

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

    # Width-set shares must compose and must never exceed the whole.
    ck("share of all measured widths is 1",
       abs(width_set_share([w for w in HIST if w in T_BY_WIDTH]) - 1.0) < 1e-9)
    ck("share of the empty set is 0", width_set_share(()) == 0.0)
    ck("{7,8} share equals its parts",
       abs(width_set_share((7, 8)) - (qmv_share(7) + qmv_share(8))) < 1e-12)

    # 🔴 A change that touches width 1 is NOT gated, and asking for it to be
    # priced as gated must RAISE, not silently return a wrong positive number.
    raised = False
    try:
        mechanism_value((1, 7), 10.0)
    except ValueError:
        raised = True
    ck("pricing width 1 as gated raises", raised)
    # 🔴 CORRECTED. Was "ungated is worth less than gated" unconditionally. That
    # holds only on the LOCAL harness. On RANKED an ungated change is worth at
    # least as much as a gated one, because the serial leg cannot respond.
    ck("LOCAL ungated is worth less than gated",
       mechanism_value((7, 8), 10.0, gated=False, harness=HARNESS_LOCAL)
       < mechanism_value((7, 8), 10.0, gated=True, harness=HARNESS_LOCAL))
    ck("RANKED ungated is worth NO LESS than gated",
       mechanism_value((7, 8), 10.0, gated=False, psi_mtp_w1=0.30)
       >= mechanism_value((7, 8), 10.0, gated=True))
    # The two harnesses must actually DISAGREE on an ungated price, otherwise the
    # harness argument is decorative and the correction never landed.
    ck("the harnesses disagree on an ungated price",
       mechanism_value((7, 8), 10.0, gated=False, harness=HARNESS_RANKED)
       != mechanism_value((7, 8), 10.0, gated=False, harness=HARNESS_LOCAL))
    # ...and must AGREE on a gated one, to the bit.
    ck("the harnesses agree exactly on a gated price",
       mechanism_value((7, 8), 10.0, gated=True, harness=HARNESS_RANKED)
       == mechanism_value((7, 8), 10.0, gated=True, harness=HARNESS_LOCAL))

    # E44 r2 narrow variant. Pinned so a silent edit to HIST or psi is caught.
    f78, e44 = e44_narrow()
    ck("f{7,8} is 0.1234", abs(f78 - 0.12343) < 5e-5, "got %.5f" % f78)
    ck("E44 narrow clears the board floor on an equal shape mix",
       e44["equal shape mix"] > SCORE_BETWEEN_SUBMISSION.pct,
       "got %+.4f vs floor %.4f" % (e44["equal shape mix"],
                                    SCORE_BETWEEN_SUBMISSION.pct))
    ck("E44 narrow clears the crown gap even on its WORST shape mix",
       e44["mlp_down only"] > 0.5193, "got %+.4f" % e44["mlp_down only"])
    ck("E44 shape mixes are ordered mlp < equal < attn",
       e44["mlp_down only"] < e44["equal shape mix"] < e44["attn_out only"])

    # ------------------------------------------------------------------
    # ORDER STATISTICS AND THE SUBSTITUTION KINK (ledger 177).
    # Stress-tested at zero, at both signs, and past saturation, because a
    # formula evaluated only at plausible inputs is not tested (ledger 176(D)).
    # ------------------------------------------------------------------
    ck("crown order stats reproduce the official score to 1e-9",
       abs(score_from_leg_gains({}) - CROWN_SCORE) < 1e-9,
       "got %.12f vs %.12f" % (score_from_leg_gains({}), CROWN_SCORE))
    ck("no leg gain is no score change (the null)",
       score_pct_from_leg_gains({}) == 0.0)
    ck("a UNIFORM gain on all eight prompts converts exactly 1:1",
       abs(score_pct_from_leg_gains({n: 3.0 for n, _ in CROWN_ORDER_STATS}) - 3.0) < 1e-9,
       "got %+.6f" % score_pct_from_leg_gains({n: 3.0 for n, _ in CROWN_ORDER_STATS}))
    ck("a gain on an UNSCORED prompt is worth exactly zero",
       score_pct_from_leg_gains({"republic": 5.0, "botany": 5.0,
                                 "essays": 5.0, "travel": 5.0}) == 0.0)
    # 🔴 THE SLOWDOWN CASE. 176(D) shipped a model that paid you to slow down.
    # Every sign path here is checked, not just the encouraging one.
    ck("a SLOWDOWN of the scored pair is a LOSS",
       score_pct_from_leg_gains({p: -5.0 for p in SCORED_PROMPTS}) < 0.0,
       "got %+.6f" % score_pct_from_leg_gains({p: -5.0 for p in SCORED_PROMPTS}))
    ck("a slowdown of an UNSCORED prompt above the pair is worth zero",
       score_pct_from_leg_gains({"republic": -1.0}) == 0.0)
    ck("score is monotone non-decreasing in a scored-pair gain",
       all(score_from_leg_gains({p: x for p in SCORED_PROMPTS})
           <= score_from_leg_gains({p: x + 0.25 for p in SCORED_PROMPTS}) + 1e-12
           for x in [i * 0.25 for i in range(-8, 60)]))
    # The kink itself: below it the rate is 1:1, above it the rate has dropped.
    k = kink_pct()
    ck("kink is where medicine meets essays, +1.055 %",
       abs(k - 1.0551) < 0.002, "got %+.4f" % k)
    ck("below the kink the scored-pair rate is 1:1",
       abs(score_pct_from_leg_gains({p: k * 0.5 for p in SCORED_PROMPTS})
           - k * 0.5) < 1e-9)
    ck("above the kink the MARGINAL rate is strictly less than 1:1",
       (score_pct_from_leg_gains({p: 2 * k for p in SCORED_PROMPTS})
        - score_pct_from_leg_gains({p: k for p in SCORED_PROMPTS})) < 0.75 * k,
       "marginal %+.4f over %+.4f of dose" % (
           score_pct_from_leg_gains({p: 2 * k for p in SCORED_PROMPTS})
           - score_pct_from_leg_gains({p: k for p in SCORED_PROMPTS}), k))
    ck("the scored pair SATURATES: a 1000 % gain buys the cap, not 1000 %",
       abs(score_pct_from_leg_gains({p: 1000.0 for p in SCORED_PROMPTS})
           - saturation_cap_pct()) < 1e-9)
    ck("saturation cap is finite and near +4.72 %",
       abs(saturation_cap_pct() - 4.7156) < 0.01,
       "got %+.4f" % saturation_cap_pct())
    ck("saturation cap comfortably exceeds the 0.5193 % crown gap",
       saturation_cap_pct() > 0.5193)
    ck("the crown gap sits BELOW the kink, so closing it is still 1:1",
       0.5193 < k, "gap 0.5193 vs kink %+.4f" % k)
    # target_for must refuse an unreachable target instead of returning a number.
    ck("target_for refuses a target above the saturation cap",
       target_for(saturation_cap_pct() + 0.01) is None)
    ck("target_for still answers for the crown gap", target_for(0.5193) is not None)
    # Marginal weights: near-equal, and NOT the 79/21 effect split.
    mw = marginal_weights()
    ck("marginal weights are near 0.484 / 0.516 and sum to ~1",
       abs(mw["beagle"] - 0.4837) < 0.002 and abs(mw["medicine"] - 0.5163) < 0.002
       and abs(sum(mw.values()) - 1.0) < 1e-9,
       "got %r" % {k2: round(v, 4) for k2, v in mw.items()})
    ck("medicine's marginal weight EXCEEDS beagle's (it is the faster member)",
       mw["medicine"] > mw["beagle"])
    ck("marginal weights are NOT the ledger's 79/21 effect split",
       abs(mw["beagle"] - 0.79) > 0.25)
    ck("beagle has far more headroom than medicine",
       substitution_headroom("beagle") > 7.0 > 2.0 > substitution_headroom("medicine"))
    ck("an unscored prompt has zero headroom by definition",
       substitution_headroom("republic") == 0.0)
    ck("an unknown prompt raises rather than defaulting",
       _raises(lambda: substitution_headroom("nonesuch"), KeyError))
    ck("an unknown prompt in a gain map raises rather than being ignored",
       _raises(lambda: score_from_leg_gains({"nonesuch": 1.0}), KeyError))

    print()
    if bad:
        print("SELFTEST FAILED: %d case(s): %s" % (len(bad), bad))
        return 1
    print("SELFTEST PASSED: RANKED uniform sign POSITIVE (+psi_mtp) and LOCAL "
          "uniform sign negative, gating vacuous on ranked, gated pricing "
          "harness-invariant at 1.140 %% / 0.771 %% targets, M=9 dominant, E27 "
          "residual large and negative, E44 narrow above the board floor on "
          "every shape mix; order statistics reproduce the crown score, the "
          "scored pair kinks at +%.3f %% and saturates at +%.3f %%, a slowdown "
          "is a loss at every dose, and target_for refuses the unreachable."
          % (kink_pct(), saturation_cap_pct()))
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
