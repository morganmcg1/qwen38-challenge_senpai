#!/usr/bin/env python3
"""Advisor desk analysis. Three results, one file, no new measurements.

1. SETTLES PR #3's `C(8) = 160.959 ms` anchor against PR #1's direct
   `C(8) = 198.237 ms`. The anchor is not a stale binary and not a bad
   measurement: the `d = 8` label names `segmentedVerifyDepthCap`, not attempted
   depth. Inverting the anchor through PR #1's measured cost curve recovers a
   mean attempted depth that matches the shipped policy's independently measured
   mean offered depth.

2. Shows that PR #1's measured per-depth `h` vector STRUCTURALLY CAPS the
   drafting policy at depth 3, for every acceptance profile, on every prompt.
   This is an analytic property of the vector, not a property of the fixture it
   was measured on.

3. Draws the consequence for the campaign: the three measured depth levers
   (h-curve, cap-7, fixed cap-3) are substitutes, not complements, so the
   published stacked-candidate arithmetic was wrong.

Every input below is a published student measurement. Nothing here is fitted.

    python3 research/depth_lever_reconciliation.py
"""

# ---------------------------------------------------------------------------
# Inputs, all from merged student reports.
# ---------------------------------------------------------------------------

# PR #1 (Edward), research/results/qwen38-r1-e1-depth-cost-curve.md, "Measured
# curve": pooled forced-depth arms, DECLARED 4-BIT head, C(0) from N = 1778.
C_DECL_US = {
    0: 65009.4,
    1: 70482.4,
    2: 75519.2,
    3: 91287.8,
    4: 115690.9,
    5: 134668.0,
    6: 154169.1,
    7: 172827.0,
    8: 198236.5,
}
# Same report, fitted per-depth marginal in units of C(0).
H_MEASURED = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]
H_FLAT = [0.20] * 8  # shipped `headStepCostRatio`

# PR #1 pair #3, live rebuilt binary, 512 decode tokens, declared head.
ARM_A512_HIST = {3: 2, 4: 38, 5: 2, 6: 3, 7: 7, 8: 30}  # flat 0.20, 82 rounds
ARM_A512_MEAN_OFFERED = 5.7927
ARM_A512_SPT = 0.03292571287602186
ARM_B512_SPT = 0.03229108382947743  # measured curve
ARM_B512_HIST = {1: 1, 2: 2, 3: 129}  # 132 rounds

# PR #3 (Thorfinn), research/results/qwen38-r1-e3-seed-prefill-amdahl.md.
# Produced by research/run-amdahl-measurement.sh, which contains NO depth
# control of any kind, on the default setup-qwen-mtp.sh stack => PINNED BF16
# head. The "d=8" column label is the cap constant.
PR3_C0_MS = 66.975
PR3_C8_MS = 160.959

# ESTABLISHED_FACTS.md "The re-basing rule": bf16 head is more expensive than
# the declared 4-bit head by this much PER HEAD FORWARD, and a depth-d round
# does exactly d head forwards.
DELTA_HEAD_MS = 2.689

# PR #2 (Alphonse), 512 decode tokens, default stack (pinned bf16 head).
ALPH_CONTROL_SPT = 0.035103861  # cap 8 / gate 3
ALPH_CAP7_SPT = 0.034020964     # cap 7 / gate 3, two-repeat mean
ALPH_EFF_DRAFT_LEN = 5.890

# PR #7 (Askeladd) control, 512 decode tokens, default stack.
ASKE_CONTROL_SPT = 0.035119320498779416

# PR #1, 256 decode tokens, SAME session, adaptive policy, heads swapped.
# This is the only head A/B that controls for session.
PR1_256_BF16_SPT = 0.041100      # arm `base-1`,    run h1t8073f, pinned bf16
PR1_256_DECL_SPT = 0.039478      # arm `base-decl`, run w8aocl64, declared 4-bit


def pct(new, old):
    return 100.0 * (new - old) / old


# ---------------------------------------------------------------------------
# 1. The PR #3 anchor
# ---------------------------------------------------------------------------

def c_curve(head, session_scale=1.0):
    """C(d) in ms for a given head, optionally rescaled to another session."""
    bump = DELTA_HEAD_MS if head == "bf16" else 0.0
    return {d: (us / 1000.0 + bump * d) * session_scale
            for d, us in C_DECL_US.items()}


def invert(curve, target_ms):
    """Mean attempted depth implied by a pooled round cost."""
    depths = sorted(curve)
    for lo, hi in zip(depths, depths[1:]):
        if curve[lo] <= target_ms <= curve[hi]:
            span = curve[hi] - curve[lo]
            return lo + (target_ms - curve[lo]) / span
    return float("nan")


def expected_cost(curve, hist):
    n = sum(hist.values())
    return sum(curve[d] * k for d, k in hist.items()) / n


print("=" * 74)
print("1. PR #3's C(8) = %.3f ms is a LABEL, not a depth" % PR3_C8_MS)
print("=" * 74)
print()
print("Two facts that need no new measurement:")
print("  - `git show 51d7dbb9:Sources/MLXFastModel/Qwen36MTPBlockSession.swift`")
print("    contains ZERO occurrences of FORCE_DEPTH. The forced-depth")
print("    instrument did not exist in the tree at PR #3's head. It first")
print("    appears at PR #1's 58cf0197, ~6 h later, and its own doc-comment")
print("    names it `qwen38-r1-e1-depth-cost-curve`.")
print("  - research/run-amdahl-measurement.sh, the producer of those anchors,")
print("    has no depth control of any kind (65 lines, read in full).")
print()
print("So `d=8` names segmentedVerifyDepthCap. The measured quantity is")
print("E[C(d_attempted)] under the natural adaptive policy. Invert it:")
print()

scale = PR3_C0_MS / (C_DECL_US[0] / 1000.0)
rows = [
    ("declared 4-bit head, no session correction", c_curve("decl")),
    ("pinned bf16 head (PR #3's actual stack)", c_curve("bf16")),
    ("pinned bf16 + PR #3 session offset x%.4f" % scale,
     c_curve("bf16", scale)),
]
print("  %-46s  implied mean depth" % "assumption")
for label, curve in rows:
    print("  %-46s        %.2f" % (label, invert(curve, PR3_C8_MS)))
print()
print("  Independently measured mean offered depth of the SHIPPED policy at")
print("  512 decode tokens:")
print("    PR #1 armA512  mean offered depth      %.4f" % ARM_A512_MEAN_OFFERED)
print("    PR #2 control  effective_mean_draft_len %.3f" % ALPH_EFF_DRAFT_LEN)
print()
print("  The anchor implies 5.4-6.4. The policy measures 5.79-5.89. The rival")
print("  hypothesis -- that PR #3 really ran pinned depth 8 -- predicts:")
for label, curve in rows:
    print("    %-46s  C(8) = %6.1f ms  (%+.1f%% vs anchor)"
          % (label, curve[8], pct(curve[8], PR3_C8_MS)))
print()
print("  Model comparison: the labelling explanation lands inside the measured")
print("  depth spread; the pinned-depth-8 explanation is 23-41% away. The")
print("  stale-binary hypothesis is not needed for THIS discrepancy.")
print()
print("  Cross-check via the pooled-cost route (PR #1 armA512 histogram,")
print("  which is the same shipped policy at the same window):")
for label, curve in rows:
    e = expected_cost(curve, ARM_A512_HIST)
    print("    %-46s  E[C] = %6.1f ms  (%+.1f%% vs anchor)"
          % (label, e, pct(e, PR3_C8_MS)))
print()
print("  CONSEQUENCE: C(0) = 66.975 vs 65.009 (+3.0%) was always fine; the")
print("  C(8) row was never a depth-8 measurement. Every number derived from")
print("  `h_avg = (161.0-67.0)/8/67.0 = 0.1754` is void. The measured mean is")
print("  %.4f (PR #1's own fit)." % (sum(H_MEASURED) / len(H_MEASURED)))
print()
print("  SIGN AUDIT: ESTABLISHED_FACTS.md ':865-867' concluded that the shipped")
print("  headStepCostRatio = 0.20 'overestimates the true marginal ratio by")
print("  1.39x locally and 1.92x against the ranked configuration'. That block")
print("  is doubly wrong -- it is built on the void C(8) = 161.0 anchor AND it")
print("  re-bases arms that already ran the declared head. Corrected: 0.20")
print("  UNDER-prices a draft step by %.0f%%, and the true h is strongly"
      % (100.0 * (sum(H_MEASURED) / len(H_MEASURED) - 0.20) / 0.20))
print("  increasing (%.4f -> %.4f), which no scalar can represent."
      % (H_MEASURED[0], H_MEASURED[-1]))
print()


# ---------------------------------------------------------------------------
# 2. The measured curve structurally caps depth at 3
# ---------------------------------------------------------------------------

def walk(h, q, cap=8):
    """Exact replica of Qwen36MTPBlockSession.instrumentedCostModelDepth with a
    constant per-position acceptance q. The shipped scalar rule is the same walk
    with a flat vector, term for term."""
    reach, expected, cum_h, depth = 1.0, 0.0, 0.0, 0
    trace = []
    for _ in range(cap):
        reach *= q
        thr = h[depth] * (1.0 + expected) / (1.0 + cum_h)
        take = reach > thr
        trace.append((depth, reach, thr, take))
        if not take:
            break
        expected += reach
        cum_h += h[depth]
        depth += 1
    return depth, trace


print("=" * 74)
print("2. The measured h vector is a HARD CAP AT DEPTH 3")
print("=" * 74)
print()
print("  %-8s %-22s %s" % ("q", "measured-curve depth", "flat-0.20 depth"))
for q in [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.00]:
    print("  %-8.3f %-22d %d" % (q, walk(H_MEASURED, q)[0], walk(H_FLAT, q)[0]))
print()
_, tr = walk(H_MEASURED, 1.0)
print("  Best possible case for going deeper, q = 1.0:")
for depth, reach, thr, ok in tr:
    print("    step %d -> %d : reach=%.4f threshold=%.4f take=%s"
          % (depth, depth + 1, reach, thr, ok))
print()
print("  `reach` is a product of probabilities and cannot exceed 1. The")
print("  threshold for the fourth draft step is %.4f. But constant q is only"
      % tr[-1][2])
print("  a sweep, so here is the general bound, for ARBITRARY per-position")
print("  acceptance p0..p3 in [0,1] (which covers the depth-0 confidence-margin")
print("  clamp, since that only ever LOWERS p0):")
print()
_cum3 = H_MEASURED[0] + H_MEASURED[1] + H_MEASURED[2]
print("    At the 3 -> 4 test the walk has set")
print("      reach    = p0*p1*p2*p3            =: r3*p3,  r3 := p0*p1*p2")
print("      expected = p0 + p0*p1 + p0*p1*p2  =  r1+r2+r3  >=  3*r3")
print("        (because r1 >= r2 >= r3, each factor being <= 1)")
print("      cumH     = h0+h1+h2 = %.4f" % _cum3)
print("    so the test `reach > h3*(1+expected)/(1+cumH)` implies")
print("      r3*p3 > %.4f*(1+3*r3)/%.4f = %.4f + %.4f*r3"
      % (H_MEASURED[3], 1.0 + _cum3,
         H_MEASURED[3] / (1.0 + _cum3),
         3.0 * H_MEASURED[3] / (1.0 + _cum3)))
_a = H_MEASURED[3] / (1.0 + _cum3)
_b = 3.0 * H_MEASURED[3] / (1.0 + _cum3)
print("    and since p3 <= 1 the left side is at most r3, giving")
print("      r3 > %.4f / (1 - %.4f) = %.4f" % (_a, _b, _a / (1.0 - _b)))
print("    which is impossible: r3 is a product of probabilities, r3 <= 1.")
print()
print("  Depth 4 is therefore UNREACHABLE under the measured curve for every")
print("  acceptance profile on every prompt -- not merely unlikely. armB512's")
print("  histogram %s is not a property of that fixture; it is a"
      % ARM_B512_HIST)
print("  property of the vector.")
print()

# Independent numerical attempt to falsify the algebra above. If the proof is
# wrong, a random search over non-constant acceptance vectors should find a
# counterexample; a deterministic seed keeps this reproducible.
import random as _random


def walk_vec(h, ps, cap=8):
    """Same walk, but with a per-position acceptance vector instead of a scalar."""
    reach, expected, cum_h, depth = 1.0, 0.0, 0.0, 0
    for _ in range(cap):
        reach *= ps[depth]
        if not reach > h[depth] * (1.0 + expected) / (1.0 + cum_h):
            break
        expected += reach
        cum_h += h[depth]
        depth += 1
    return depth


_random.seed(20260816)
_worst_curve, _worst_flat = 0, 0
_TRIALS = 400000
for _ in range(_TRIALS):
    ps = [_random.random() ** _random.choice([0.05, 0.25, 1.0]) for _ in range(8)]
    _worst_curve = max(_worst_curve, walk_vec(H_MEASURED, ps))
    _worst_flat = max(_worst_flat, walk_vec(H_FLAT, ps))
# The all-ones corner is the analytic optimum; include it explicitly.
_worst_curve = max(_worst_curve, walk_vec(H_MEASURED, [1.0] * 8))
_worst_flat = max(_worst_flat, walk_vec(H_FLAT, [1.0] * 8))
print("  Falsification attempt: %d random NON-constant acceptance vectors"
      % _TRIALS)
print("  (biased hard toward 1.0), plus the all-ones corner, run through the")
print("  same walk with widthCap released to 8:")
print("    deepest depth ever reached, measured curve : %d   <- the cap"
      % _worst_curve)
print("    deepest depth ever reached, flat 0.20      : %d" % _worst_flat)
if _worst_curve != 3:
    raise SystemExit("PROOF FALSIFIED: measured curve reached depth %d"
                     % _worst_curve)
print()
print("  Corollaries, all analytic:")
print("    - depth <= 3 => verify width <= 4 => sdpaWidthWallDepthCap = 4")
print("      never binds, segmentedStreakGate never matters, and")
print("      segmentedVerifyDepthCap (7 or 8) is unreachable dead code.")
print("    - PR #2's cap-7 win is therefore INERT on top of the h-curve.")
print("    - a fitted 8-vector that provably collapses to 'at most 3' should")
print("      be considered against simply shipping 'at most 3', which has no")
print("      fitted parameters. They differ only in the low-acceptance tail:")
for q in [0.60, 0.65, 0.70, 0.75]:
    print("        q=%.2f  h-curve picks %d,  flat-0.20 under a hard cap 3 picks %d"
          % (q, walk(H_MEASURED, q)[0], walk(H_FLAT, q, cap=3)[0]))
print()


# ---------------------------------------------------------------------------
# 3. Campaign consequence
# ---------------------------------------------------------------------------

print("=" * 74)
print("3. The three depth levers are SUBSTITUTES, and are not comparable yet")
print("=" * 74)
print()
print("  lever                      d(s/token)   head             source")
print("  h-curve as default         %+8.4f%%   declared 4-bit   PR #1 armB512"
      % pct(ARM_B512_SPT, ARM_A512_SPT))
print("  cap 7 / gate 3             %+8.4f%%   pinned bf16      PR #2"
      % pct(ALPH_CAP7_SPT, ALPH_CONTROL_SPT))
print("  fixed cap 3                   unmeasured   -                -")
print()
print("  The two measured numbers ran on different heads, so they cannot be")
print("  ranked. To compare them one must be converted to the other's head.")
print("  Define the head offset as the multiplier that turns a declared-4-bit")
print("  s/token into a pinned-bf16 s/token. Three independent estimates:")
HEAD_OFFSETS = [
    ("256 tok, one session, base-1 / base-decl",
     pct(PR1_256_BF16_SPT, PR1_256_DECL_SPT)),
    ("512 tok, PR #2 control / PR #1 armA512",
     pct(ALPH_CONTROL_SPT, ARM_A512_SPT)),
    ("512 tok, PR #7 control / PR #1 armA512",
     pct(ASKE_CONTROL_SPT, ARM_A512_SPT)),
]
for label, off in HEAD_OFFSETS:
    print("    %-42s : %+.2f%%" % (label, off))
print("  (PR #2 and PR #7 controls agree with each other to %.2f%%, which is"
      % abs(pct(ASKE_CONTROL_SPT, ALPH_CONTROL_SPT)))
print("   why I read the 512-token gap as the head and not the session. The")
print("   256-token estimate is the only one that also controls for session,")
print("   and it is the SMALLEST. Both readings stay on the table.)")
print()
for label, off in HEAD_OFFSETS:
    corrected = ARM_B512_SPT * (1.0 + off / 100.0)
    winner = "cap-7" if ALPH_CAP7_SPT < corrected else "h-curve"
    print("  h-curve arm + %+.2f%% -> %.6f  vs cap-7 %.6f  => %-7s wins"
          % (off, corrected, ALPH_CAP7_SPT, winner))
print()
print("  The ranking FLIPS inside the uncertainty of the head correction.")
print("  One host, one head, one session is the only way to settle it, which")
print("  is what PR #13 (E11) does.")
print()
print("  RETRACTED: the stacked-candidate arithmetic published in")
print("  ESTABLISHED_FACTS.md section 9 added cap-7 (-3.085%) to the 3-bit")
print("  draft head (-1.90%) and projected ~3.03. The depth levers do not add")
print("  to each other; only ONE of them can be in the candidate, and the")
print("  correct form is max(depth lever) + 3-bit draft head. On today's")
print("  numbers that is -3.085% + -1.90% ~= -4.9% only because cap-7 happens")
print("  to be the depth lever chosen -- but if the h-curve or a fixed cap-3")
print("  wins on a matched head, the depth term is smaller and the candidate")
print("  does not clear 3.0 on this stack alone.")
