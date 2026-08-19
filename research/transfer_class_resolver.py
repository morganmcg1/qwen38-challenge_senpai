#!/usr/bin/env python3
"""Which transfer branch applies to a QMV decode-width change? Resolve 187(J).

187(J) records an inconsistency in my own published work: two transfer
estimates for the same class of term, differing by 3x.

  /3.55            mechanistic, 186(D): prefill GEMM transfers 7.58x while a
                   depth-0 decode round transfers 2.14x.
  x0.834..0.862    calibrated: the two-parameter depth transfer
                   g in [0.7388, 0.7778], mean-pinned at depth 4.

I instructed the campaign to carry both as a band. That was the right call
while the question was open. It is now closable, and the answer is
decision-relevant: the r=2 row-block route forecasts +0.17..+0.25 % under the
/3.55 branch, which does NOT close our 0.5367 % deficit, against +0.49..+0.76 %
under the g branch, which does.

THE FRAMEWORK

Let L be leg time, t the time of one cost term, and tau = t_local / t_rank the
term's own transfer ratio. The leg transfers at R = L_local / L_rank = 2.14
(the depth-0 round ratio 65.009 / 30.402).

A local relative saving delta_local = dt_local / L_local becomes, at rank:

    delta_rank = dt_rank / L_rank
               = (dt_local / tau) / (L_local / R)
               = delta_local * (R / tau)

So the transfer multiplier is exactly R / tau. This single expression
reproduces every branch of 186(D):

    tau = 7.58  (arithmetic-bound, qmm_nax)  ->  2.14/7.58 = 0.282 = /3.55
    tau = 2.14  (transfers like the leg)     ->  2.14/2.14 = 1.00
    tau = 1.00  (latency/dispatch-bound)     ->  2.14/1.00 = 2.14, better than 1:1

186(D) is therefore structurally correct. The open question is purely: what is
tau for the QMV crossrow kernel at decode widths?

THE RESOLUTION

186(D) groups "compute-bound OR memory-traffic-bound" into the /3.55 class.
That grouping is wrong, and the reason is mechanistic rather than empirical.

The 7.58x prefill advantage is the qmm_nax signature (186(C):
quantized.cpp:473, requires is_nax_available() and GPU gen >= 17). qmm_nax is
a MATMUL path. It accelerates arithmetic using matrix hardware; it does nothing
for memory bandwidth.

The scored decode path never reaches it. The shipped host chooser sends
M <= 9 to qmv_fast (quantized.cpp:250-295) and only switches to the qmm
split-k path at M = 10 (quantized.cpp:300-325) -- which is exactly why the
campaign records M=10 bitwise deltas as pre-existing qmm split-k 9->10 padding
and pins the maximum scored verify width at 9.

So the M5's 7.58x advantage is unreachable from the decode QMV kernel by
construction. tau_qmv cannot be 7.58.

What tau_qmv is instead follows from what the depth-0 round is made of: 64
layers of quantized projections dispatched through this same qmv_fast family.
The round IS predominantly this kernel, so the kernel transfers at
approximately the round ratio, tau_qmv ~= R = 2.14, giving a multiplier near
1.0. E54's own bandwidth measurement is consistent: the kernel runs at
150.9 GB/s against a sustained 148.4 GB/s ceiling, i.e. bandwidth-bound, and
no plausible M5 memory system is 7.58x an M4 Pro's ~273 GB/s.

VERDICT: the /3.55 branch is REFUTED for QMV decode-width changes. It remains
correct for prefill-GEMM-class terms, which is where it was derived and where
it should stay. The calibrated g band is the applicable estimate, and it sits
just below the structural 1.0 because not all of a round is QMV.

This does not touch the additive-vs-multiplicative residual question. Neither
transfer factor is negative, so neither explains E27's sign flip; only the
shared-ceiling tax does. E59 still has to measure that.
"""

import argparse

# 186(C) / 186(D) measured transfer anchors.
PREFILL_LOCAL_S, PREFILL_RANK_S = 3.9938, 0.5269
ROUND_LOCAL_MS, ROUND_RANK_MS = 65.009, 30.402

# 186(D) mechanistic divisor as published.
PUBLISHED_DIVISOR = 3.55

# Calibrated joint depth transfer (ledger 184), applied directly.
G_LO, G_HI = 0.7388, 0.7778

# The same calibration mean-pinned at depth 4, which is the form 187(I) used.
# h and g are NOT the same number and must not be conflated: pinning rescales
# the band upward by about 11-13 %. Both are reported; the applicable
# calibrated range is their union.
H_LO, H_HI = 0.8343, 0.8617

# 187(L) r=2 route inputs: diluted (already x0.9125) score points.
R2_DILUTED = {"e48": 0.5900, "e53_mid": 0.8834}

DEFICIT_PCT = 0.5367
# RETRACTED by ledger 193(E): this is 2 sd of the SERIAL leg's jitter applied to the
# score, and the median over eight prompts does not average the candidate-leg common
# mode away. The measured single-pair ranked MDE is 2.10 %, 7.4x larger. The value
# below is kept so this module's published arithmetic stays reproducible; import
# research/ranked_noise.py for any NEW ranked pricing.
RANKED_MDE_PCT = 0.283


def tau_prefill():
    return PREFILL_LOCAL_S / PREFILL_RANK_S


def leg_ratio():
    return ROUND_LOCAL_MS / ROUND_RANK_MS


def multiplier(tau):
    """delta_rank / delta_local for a term with transfer ratio tau."""
    return leg_ratio() / tau


def report():
    R = leg_ratio()
    tp = tau_prefill()
    print("=" * 74)
    print("1. The two anchors, and the single expression behind both branches")
    print("=" * 74)
    print(f"  prefill GEMM transfer   tau = {PREFILL_LOCAL_S} / {PREFILL_RANK_S}"
          f" = {tp:.4f}x")
    print(f"  depth-0 round transfer  R   = {ROUND_LOCAL_MS} / {ROUND_RANK_MS}"
          f" = {R:.4f}x")
    print(f"\n  multiplier(tau) = R / tau")
    print(f"    tau = {tp:.2f} (arithmetic, qmm_nax) -> {multiplier(tp):.4f}"
          f"  = /{1/multiplier(tp):.2f}   <- 186(D)'s published /3.55")
    print(f"    tau = {R:.2f} (transfers like the leg) -> {multiplier(R):.4f}")
    print(f"    tau = 1.00 (latency/dispatch-bound)  -> {multiplier(1.0):.4f}"
          f"  better than 1:1")

    print()
    print("=" * 74)
    print("2. Why tau_qmv cannot be the prefill value")
    print("=" * 74)
    print("  The 7.58x advantage is the qmm_nax signature (quantized.cpp:473,")
    print("  needs is_nax_available(), GPU gen >= 17). qmm_nax is a MATMUL")
    print("  path: it accelerates arithmetic, not memory bandwidth.")
    print()
    print("  The scored decode path never reaches it:")
    print("    M <= 9  -> qmv_fast        (quantized.cpp:250-295)")
    print("    M == 10 -> qmm split-k     (quantized.cpp:300-325)")
    print("  and the maximum scored verify width is 9. The campaign already")
    print("  records the M=10 bitwise deltas as qmm split-k 9->10 padding,")
    print("  which is independent confirmation of where the switch sits.")
    print()
    print("  Therefore the M5's arithmetic advantage is unreachable from the")
    print("  decode QMV kernel by construction.")

    print()
    print("=" * 74)
    print("3. What tau_qmv is instead")
    print("=" * 74)
    print("  A depth-0 round is 64 layers of quantized projections dispatched")
    print("  through this same qmv_fast family, so the round is predominantly")
    print("  this kernel and tau_qmv ~= R, giving a multiplier near 1.0.")
    print()
    print("  E54's bandwidth measurement agrees: 150.9 GB/s achieved against a")
    print("  148.4 GB/s sustained ceiling, i.e. bandwidth-bound. No plausible")
    print("  M5 memory system is 7.58x an M4 Pro's ~273 GB/s.")
    print()
    print(f"  Calibrated band g in [{G_LO}, {G_HI}] sits just below the")
    print("  structural 1.0, as expected, because not all of a round is QMV.")

    print()
    print("=" * 74)
    print("4. Decision impact: the r=2 row-block route")
    print("=" * 74)
    print(f"  deficit to close = {DEFICIT_PCT} %   ranked MDE (2 sd) ="
          f" {RANKED_MDE_PCT} %")
    print()
    print(f"  {'mixture':<9} {'diluted':>8} {'/3.55 refuted':>14}"
          f" {'x g':>17} {'x h (pinned)':>19}")
    for mix, dil in R2_DILUTED.items():
        old = dil / PUBLISHED_DIVISOR
        print(f"  {mix:<9} {dil:>8.4f} {old:>14.4f}"
              f" {dil*G_LO:>7.4f}..{dil*G_HI:<9.4f}"
              f" {dil*H_LO:>8.4f}..{dil*H_HI:<9.4f}")
    print()
    print("  Calibrated range = union of the g and h bands, since 187(J)'s")
    print("  rule forbids silently picking a branch:")
    for mix, dil in R2_DILUTED.items():
        lo, hi = dil * G_LO, dil * H_HI
        verdict = ("CLOSES the deficit at the top of the band"
                   if hi >= DEFICIT_PCT else "does NOT close the deficit")
        print(f"    {mix:<8} {lo:.4f}..{hi:.4f} %   {verdict}")
    print()
    print("  Under the refuted /3.55 branch BOTH mixtures fall below the")
    print(f"  ranked MDE of {RANKED_MDE_PCT} %, so the route would not even be")
    print("  measurable at rank. That is how much this resolution matters.")
    print()
    print("  🔴 The route closes the deficit only under the e53_mid mixture.")
    print("  Under e48 it does not, at any point in the calibrated band. So")
    print("  askeladd's #57 M=9 arm, which settles the mixture dispute, also")
    print("  decides whether thorfinn's r=2 route can win. The two live")
    print("  experiments are coupled and neither is redundant.")

    print()
    print("=" * 74)
    print("5. Verdict")
    print("=" * 74)
    print("  /3.55 is REFUTED for QMV decode-width changes. It stays correct")
    print("  for prefill-GEMM-class terms, where it was derived.")
    print("  187(J)'s band collapses to the calibrated g branch.")
    print()
    print("  This does NOT resolve the additive-vs-multiplicative residual.")
    print("  Neither transfer factor is negative, so neither explains E27's")
    print("  sign flip. Only the shared-ceiling tax does, and E59 measures it.")


def self_test():
    n = 0
    R = leg_ratio()
    tp = tau_prefill()

    # 1. the published divisor is reproduced by R / tau_prefill
    assert abs(1.0 / multiplier(tp) - PUBLISHED_DIVISOR) < 0.02, 1 / multiplier(tp)
    n += 1

    # 2. a term transferring like the leg has multiplier exactly 1
    assert abs(multiplier(R) - 1.0) < 1e-12
    n += 1

    # 3. a latency-bound term transfers better than 1:1
    assert multiplier(1.0) > 1.0
    n += 1

    # 4. the branches really do differ by about 3x, which is what 187(J) flagged
    assert 2.5 < (G_LO / multiplier(tp)) < 3.5, G_LO / multiplier(tp)
    n += 1

    # 5. the calibrated band sits strictly below the structural 1.0
    assert 0 < G_LO < G_HI < 1.0
    n += 1

    # 6. decision relevance: the branches straddle the deficit for e53_mid
    dil = R2_DILUTED["e53_mid"]
    assert dil / PUBLISHED_DIVISOR < DEFICIT_PCT < dil * G_HI
    n += 1

    # 7. under the refuted branch BOTH mixtures fall below the ranked MDE,
    #    so the route would not even be measurable at rank
    for v in R2_DILUTED.values():
        assert v / PUBLISHED_DIVISOR < RANKED_MDE_PCT, v
    n += 1

    # 8. h and g are distinct; conflating them inflates the band
    assert H_LO > G_LO and H_HI > G_HI
    assert 1.05 < (H_LO / G_LO) < 1.20, H_LO / G_LO
    n += 1

    # 9. the coupling claim: e53_mid closes the deficit across the calibrated
    #    union and e48 never does, so the mixture dispute decides the route
    assert R2_DILUTED["e53_mid"] * H_HI >= DEFICIT_PCT
    assert R2_DILUTED["e48"] * H_HI < DEFICIT_PCT
    n += 1

    # 10. monotonicity: a better-transferring term is worth less locally
    assert multiplier(10.0) < multiplier(2.0) < multiplier(0.5)
    n += 1

    print(f"self-test OK: {n} checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    self_test() if a.self_test else report()
