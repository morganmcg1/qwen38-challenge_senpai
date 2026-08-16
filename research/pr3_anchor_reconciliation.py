#!/usr/bin/env python3
"""Reconcile PR #3's `C(8) = 161.0 ms` anchor with PR #1 r3's direct curve.

Question: PR #3 reports the "MTP d=8" leg at 161.0 ms/round. PR #1 r3 measures
C(8) = 198.683 ms directly (forced depth, acc == d, 512-token window, 4-bit
head). The two disagree by -23.4% at d=8 while agreeing to 2.3% at d=0.

Hypothesis under test: **the PR #3 leg never ran at depth 8.** "d=8" is the
*configured cap*; the adaptive policy (acceptEMAAlpha = 0.15) chose a smaller
depth on most rounds, so 161.0 ms is E[C(d_attempted)], not C(8).

Two independent readings of the same leg are compared:
  (A) cost channel  -- round cost / C(0), inverted through PR #1's curve
  (B) yield channel -- decode tokens / rounds, which fixes accepted drafts

If the hypothesis is right, (A) and (B) must imply a mutually consistent and
physically plausible acceptance rate. They are not fitted to each other.

Run: python3 research/pr3_anchor_reconciliation.py
Dependency-free.
"""

# ---- PR #3 parent-clock leg data (research/results/qwen38-r1-e3-seed-prefill-amdahl.md:150)
MTP_SUM_BLOCK_LATENCY = 1.6095870733261108  # seconds, "MTP d=8" leg
MTP_ROUNDS = 10
SERIAL_SUM_BLOCK_LATENCY = 4.286431789398193
SERIAL_ROUNDS = 64
DECODE_TOKENS = 64  # MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS for PR #3's legs

# ---- PR #1 r3 measured curve (research/results/qwen38-r1-e1-depth-cost-curve.md)
C0_EDWARD_MS = 65.469
RATIO = {0: 1.0, 1: 1.0902, 2: 1.1582, 3: 1.4017, 4: 1.7821,
         5: 2.0599, 6: 2.3580, 7: 2.6295, 8: 3.0545}


def invert_ratio(r):
    """Piecewise-linear inverse of the measured C(d)/C(0) curve."""
    if r < RATIO[0]:
        return None
    for d in range(8):
        if RATIO[d] <= r <= RATIO[d + 1]:
            return d + (r - RATIO[d]) / (RATIO[d + 1] - RATIO[d])
    return None


def expected_accepted(q, depth):
    """E[accepted drafts] for i.i.d. per-draft acceptance q, first miss stops."""
    total = 0.0
    p = 1.0
    for _ in range(depth):
        p *= q
        total += p
    return total


def solve_q(target_accepted, depth, lo=0.0, hi=1.0, iters=200):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if expected_accepted(mid, depth) < target_accepted:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    mtp_ms = MTP_SUM_BLOCK_LATENCY / MTP_ROUNDS * 1000.0
    serial_ms = SERIAL_SUM_BLOCK_LATENCY / SERIAL_ROUNDS * 1000.0

    print("=" * 74)
    print("PR #3 leg, recomputed from raw totals")
    print("=" * 74)
    print("  serial d=0 : %8.3f ms/round  (%d rounds)" % (serial_ms, SERIAL_ROUNDS))
    print("  'MTP d=8'  : %8.3f ms/round  (%d rounds)" % (mtp_ms, MTP_ROUNDS))
    print("  PR #1 r3   : C(0) = %.3f ms, C(8) = %.3f ms"
          % (C0_EDWARD_MS, C0_EDWARD_MS * RATIO[8]))
    print("  gap at d=8 : %+.1f%%"
          % (100.0 * (mtp_ms - C0_EDWARD_MS * RATIO[8]) / (C0_EDWARD_MS * RATIO[8])))
    print("  gap at d=0 : %+.1f%%" % (100.0 * (serial_ms - C0_EDWARD_MS) / C0_EDWARD_MS))

    print()
    print("=" * 74)
    print("CHANNEL A -- cost. Invert the measured curve at the observed cost.")
    print("=" * 74)
    depths = {}
    for label, c0 in (("PR #3 own C(0) = %.3f" % serial_ms, serial_ms),
                      ("PR #1 C(0)     = %.3f" % C0_EDWARD_MS, C0_EDWARD_MS)):
        r = mtp_ms / c0
        d = invert_ratio(r)
        depths[label] = d
        print("  %s -> ratio %.4f -> implied mean attempted depth %.2f" % (label, r, d))

    print()
    print("=" * 74)
    print("CHANNEL B -- yield. Tokens per round fixes accepted drafts.")
    print("=" * 74)
    tpr = DECODE_TOKENS / MTP_ROUNDS
    accepted = tpr - 1.0
    print("  %d decode tokens / %d rounds = %.2f tokens per round"
          % (DECODE_TOKENS, MTP_ROUNDS, tpr))
    print("  => %.2f accepted drafts per round" % accepted)
    print("  rounds required if depth really were 8 with full acceptance: %.1f"
          % (DECODE_TOKENS / 9.0))
    print("  ...%d were observed, so acceptance was NOT full." % MTP_ROUNDS)

    print()
    print("=" * 74)
    print("CONSISTENCY -- do A and B agree on a plausible acceptance rate?")
    print("=" * 74)
    for label, d in depths.items():
        q_flat = accepted / d
        q_geom = solve_q(accepted, max(1, int(round(d))))
        print("  attempted depth %.2f (%s)" % (d, label.split("=")[0].strip()))
        print("      q if every draft verified independently : %.3f" % q_flat)
        print("      q if first-miss-stops (geometric)        : %.3f" % q_geom)

    print()
    print("  Cross-check: does depth 8 with the geometric q reproduce the yield?")
    for q in (0.85, 0.875, 0.90):
        print("      q = %.3f at depth 8 -> %.2f accepted drafts (observed %.2f)"
              % (q, expected_accepted(q, 8), accepted))

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("""  THE YIELD CHANNEL DOES NOT DISCRIMINATE. I set this analysis up expecting
  two independent channels to converge. They do not. 5.40 accepted drafts per
  round is reproduced by depth 6.2 at q ~ 0.97 AND by depth 8 at q ~ 0.93, and
  both acceptance rates are physically plausible. Channel B therefore only
  rules out *full* acceptance; it does not choose between the hypotheses, and
  reporting it as confirmation would be double-counting.

  ALL the discriminating power is in the cost channel, which is one number
  (160.959 ms) from one 10-round leg.

  => LEADING HYPOTHESIS, NOT A RESULT: `161.0 ms` is E[C(d_attempted)] at
     d_bar ~ 6.2, not C(8) -- a LABELLING error in PR #3 rather than a
     measurement error, since the leg was configured with cap 8 while the
     adaptive policy (acceptEMAAlpha = 0.15) chose smaller depths.
     Competing explanations NOT excluded: partial scope of the latency counter,
     and a 10-round sample.

  What is nevertheless settled, because it does not depend on the explanation:
    - `C(8) = 161.0 ms` is REFUTED as a depth-8 round cost. PR #1's direct
      forced-depth measurement is 198.683 ms, and the two legs agree at d=0 to
      2.3%, so the disagreement is real and depth-dependent.
    - PR #3's P = 4.0086 s seed prefill SURVIVES. The two-leg algebra treats
      sum(block_latency) as an observed total and never interprets it per-depth.
    - Every downstream use of `C(8) = 161.0` as a depth-8 round cost is WRONG:
      h_avg = 0.176, "residual tax 1.71-1.91x", "67-77 ms per round
      unexplained", and the endpoint test that produced my false retraction of
      the depth cost curve.

  TO SETTLE IT: read realised attempted depth per round directly off the leg.
  PR #1's merged instrument (MLX_QWEN_MTP_FORCE_DEPTH / the depth histogram)
  is exactly the tool for this. One run, not a research programme.""")

    print()
    print("  Sign convention, stated because I got it wrong once:")
    hi, lo = C0_EDWARD_MS * RATIO[8], mtp_ms
    print("    PR #1 measurement is %+.1f%% relative to the anchor" % (100.0 * (hi - lo) / lo))
    print("    the anchor is        %+.1f%% relative to the PR #1 measurement"
          % (100.0 * (lo - hi) / hi))
    print("    Using the report's stated C(8) = 198.683 ms instead: %+.1f%% / %+.1f%%"
          % (100.0 * (198.683 - lo) / lo, 100.0 * (lo - 198.683) / 198.683))


if __name__ == "__main__":
    main()
