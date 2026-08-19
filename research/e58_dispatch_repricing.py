#!/usr/bin/env python3
"""Re-derive the SDPA-dispatch value proposition that ledger 186(D) priced at
`2.2 % of the ranked leg / 2.0 % of score / 7.1 sd` and assigned as E58.

Ledger 186(D)'s chain is:

    22 us/dispatch            (from E57 Arm B: +2226 dispatches, +0.27 % s/tok)
    6163 dispatches -> 136 ms (Arm A local SDPA total)
    136 ms / 18.35 s = 0.74 % (local leg share)
    136 ms /  6.23 s = 2.2 %  (asserted ranked beagle leg share)
    2.2 % x 0.9125  = 2.0 %   ("after dilution")

This module reconstructs each step and tests it. Three defects are checked:

  D1  The 22 us constant's denominator is never written down anywhere in the
      ledger.  It IS reconstructible (512 tokens), so this is a presentation
      gap, not an arithmetic error -- but the constant is an UPPER BOUND, not a
      point estimate, because Arm B's added dispatches perform real work.

  D2  `136 ms / 6.23 s` carries the LOCAL dispatch COUNT onto the RANKED leg.
      The ranked beagle leg runs 107 rounds at mean draft 4.5327; our local
      Arm A ran 76 rounds at mean draft 6.5132.  Dispatch count per round is a
      step function of width, so the count must be recomputed, not carried.

  D3  `2.2 % x 0.9125` multiplies a quantity that is ALREADY leg-basis by the
      round-basis -> leg-basis dilution factor.  This is a third instance of the
      double-dilution that ledger 189(J) forbids (item 122 hypothesis B and
      item 189/E55 are the two already named).

Run `python3 research/e58_dispatch_repricing.py --self-test` for the checks and
with no argument for the report.  Exit 0 means every check passed.

Sources
-------
ledger :10327-10331   E57 arm results (dispatch counts, s/tok, +0.27 %)
ledger :10341         "of 76 rounds" -- Arm A round count
ledger :10492-10496   the 186(D) pricing chain under audit
ledger :10425-10440   186(B) per-prompt leg / dilution table
ledger :11041-11060   188(A) transfer multiplier R/tau, R = 2.1383
research/e53-board-facts.json  ranked per-prompt rounds and mean draft
"""

import sys

# ---------------------------------------------------------------- E57 arm data
# ledger :10327-10331
ARM_A_DISPATCHES = 6163
ARM_B_DISPATCHES = 8389
ARM_C_DISPATCHES = 10957
ARM_A_SPT = 0.035845          # seconds per token, local
ARM_B_REGRESSION_PCT = 0.27   # % of local s/token
ARM_A_ROUNDS = 76             # ledger :10341
ARM_A_MEAN_DRAFT = 6.5132
DECODE_TOKENS = 512           # the E57 window

# ------------------------------------------------------------- ranked receipts
# research/e53-board-facts.json, reconstructed in ledger item 184
RANKED = {
    # prompt:     (rounds, mean_draft, leg_ms)
    "plutarch":   (487, 0.1540, 15516.8),
    "drama":      (252, 2.2976, 10125.7),
    "travel":     (212, 2.6557,  8903.0),
    "beagle":     (107, 4.5327,  6233.1),
    "medicine":   ( 99, 4.7677,  5820.7),
    "republic":   ( 89, 5.2697,  5726.1),
    "essays":     ( 87, 5.4253,  5763.7),
    "botany":     ( 85, 5.7765,  5673.2),
}

# 186(B): dilution_p = 1 - K/leg_p
RANKED_DILUTION = {
    "plutarch": 0.96606, "drama": 0.94799, "travel": 0.94085,
    "beagle": 0.91552, "medicine": 0.90953, "republic": 0.90803,
    "essays": 0.90863, "botany": 0.90724,
}
DILUTION_MEDIAN_PAIR = 0.9125          # mean(beagle, medicine)

# ------------------------------------------------------------------- constants
FULL_ATTENTION_LAYERS = 16             # 64 layers = 48 GDN + 16 full attention
LOCAL_PREFILL_SHARE = 0.23389          # E55, MTP leg
TRANSFER_R = 65.009 / 30.402           # 188(A), = 2.1383
RANKED_JITTER_PCT = 0.2257             # per prompt per leg, n=408
RANKED_MDE_PCT = 0.283                 # 2 sd
MAX_DRAFT = 8                          # segmentedVerifyDepthCap

# SDPA dispatches per full-attention layer, by round width qL (185(B)):
#   qL <= 5 : fused vector, 1 dispatch (kL < 1024) or 2 (kL >= 1024)
#   qL 6..9 : chunked into a 5-row and a (qL-5)-row call, both fused, 4 or 6
NARROW_PER_LAYER = (1, 2)
WIDE_PER_LAYER = (4, 6)
CHUNK_WIDTH_THRESHOLD = 6              # qL >= 6 takes the chunked path


def per_dispatch_seconds_upper_bound():
    """Reconstruct the 22 us constant, showing the denominator 186(D) omits."""
    leg_seconds = ARM_A_SPT * DECODE_TOKENS
    added_seconds = leg_seconds * (ARM_B_REGRESSION_PCT / 100.0)
    added_dispatches = ARM_B_DISPATCHES - ARM_A_DISPATCHES
    return added_seconds / added_dispatches, leg_seconds, added_dispatches


def local_share():
    per, leg, _ = per_dispatch_seconds_upper_bound()
    total = ARM_A_DISPATCHES * per
    return total, total / leg * 100.0, leg


def dispatches_per_round(width, narrow_idx, wide_idx):
    """SDPA dispatches in one round of target width `width` (= draft + 1)."""
    if width >= CHUNK_WIDTH_THRESHOLD:
        return FULL_ATTENTION_LAYERS * WIDE_PER_LAYER[wide_idx]
    return FULL_ATTENTION_LAYERS * NARROW_PER_LAYER[narrow_idx]


def wide_round_fraction_bounds(mean_draft):
    """Bound the fraction of rounds with draft >= 5 given only the mean draft.

    184(D) proved the ranked width histogram is not identifiable from the
    receipt.  Only the mean is observed, so the wide fraction is an interval.

    Lower bound: put all narrow mass at draft 4 and all wide mass at draft 8.
    Upper bound: put all narrow mass at draft 0 and all wide mass at draft 5.
    """
    # A round of draft d has target width M = d + 1, so the chunked path starts
    # at draft CHUNK_WIDTH_THRESHOLD - 1 = 5 and the widest narrow draft is 4.
    narrow_max = CHUNK_WIDTH_THRESHOLD - 2
    if mean_draft <= narrow_max:
        lo = 0.0
    else:
        lo = (mean_draft - narrow_max) / (MAX_DRAFT - narrow_max)
    hi = min(1.0, mean_draft / float(narrow_max + 1))
    return lo, min(hi, 1.0)


def ranked_dispatch_bounds(prompt):
    rounds, mean_draft, _leg = RANKED[prompt]
    lo_w, hi_w = wide_round_fraction_bounds(mean_draft)
    narrow_lo = FULL_ATTENTION_LAYERS * NARROW_PER_LAYER[0]
    narrow_hi = FULL_ATTENTION_LAYERS * NARROW_PER_LAYER[1]
    wide_lo = FULL_ATTENTION_LAYERS * WIDE_PER_LAYER[0]
    wide_hi = FULL_ATTENTION_LAYERS * WIDE_PER_LAYER[1]
    lo = rounds * ((1 - lo_w) * narrow_lo + lo_w * wide_lo)
    hi = rounds * ((1 - hi_w) * narrow_hi + hi_w * wide_hi)
    return lo, hi, (lo_w, hi_w)


def ranked_leg_share_bounds(prompt):
    """Dispatch overhead as a share of the ranked leg, at tau = 1."""
    per, _, _ = per_dispatch_seconds_upper_bound()
    lo_d, hi_d = ranked_dispatch_bounds(prompt)[:2]
    leg_s = RANKED[prompt][2] / 1000.0
    return lo_d * per / leg_s * 100.0, hi_d * per / leg_s * 100.0


def ledger_186d_chain():
    """Reproduce the published chain exactly, defects included."""
    per, leg, _ = per_dispatch_seconds_upper_bound()
    total_ms = ARM_A_DISPATCHES * per * 1000.0
    local_pct = total_ms / (leg * 1000.0) * 100.0
    ranked_pct = total_ms / RANKED["beagle"][2] * 100.0
    diluted_pct = ranked_pct * DILUTION_MEDIAN_PAIR
    return total_ms, local_pct, ranked_pct, diluted_pct


def transfer_route():
    """The 188(A) route: convert to round basis, apply R/tau, convert back."""
    _total, local_leg_pct, _leg = local_share()
    local_round_pct = local_leg_pct / (1.0 - LOCAL_PREFILL_SHARE)
    ranked_round_pct = local_round_pct * TRANSFER_R      # tau = 1
    ranked_leg_pct = ranked_round_pct * DILUTION_MEDIAN_PAIR
    return local_round_pct, ranked_round_pct, ranked_leg_pct


# ------------------------------------------------------------------ self-tests

def self_test():
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    per, leg, added = per_dispatch_seconds_upper_bound()
    ck("D1 denominator is 512 tokens: leg reproduces the ledger's 18.35 s",
       abs(leg - 18.35) < 0.01, f"leg={leg:.4f} s")
    ck("D1 per-dispatch bound reproduces the published 22 us",
       21.5e-6 < per < 22.9e-6, f"per={per*1e6:.2f} us")
    ck("D1 added dispatch count is 2226 as the ledger states",
       added == 2226, f"added={added}")

    total_ms, local_pct, ranked_pct, diluted_pct = ledger_186d_chain()
    ck("published 136 ms local SDPA total reproduces",
       134.0 < total_ms < 139.0, f"{total_ms:.1f} ms")
    ck("published 0.74 % local leg share reproduces",
       0.73 < local_pct < 0.76, f"{local_pct:.3f} %")
    ck("published 2.2 % ranked share reproduces (defect D2 present)",
       2.15 < ranked_pct < 2.25, f"{ranked_pct:.3f} %")
    ck("published 2.0 % reproduces only via a second dilution (defect D3)",
       1.96 < diluted_pct < 2.02, f"{diluted_pct:.3f} %")

    # D3 is a double dilution: the input was already a leg-basis fraction.
    ck("D3: the 2.2 % input is a leg-basis fraction, so x0.9125 is a second "
       "dilution of the same prefill",
       diluted_pct < ranked_pct and
       abs(diluted_pct / ranked_pct - DILUTION_MEDIAN_PAIR) < 1e-9)

    # D2: the implied leg-time ratio contradicts the formal transfer constant.
    implied = leg * 1000.0 / RANKED["beagle"][2]
    ck("D2: implied leg-time multiplier 2.945 differs from 188(A)'s R=2.1383",
       abs(implied - 2.945) < 0.01 and abs(implied - TRANSFER_R) > 0.5,
       f"implied={implied:.4f} vs R={TRANSFER_R:.4f}")

    # D2 core: the ranked dispatch count is not the local dispatch count.
    lo_d, hi_d, (lo_w, hi_w) = ranked_dispatch_bounds("beagle")
    ck("D2: ranked beagle dispatch count is an interval, not 6163",
       lo_d < ARM_A_DISPATCHES < hi_d and hi_d / lo_d > 3.0,
       f"[{lo_d:.0f}, {hi_d:.0f}], ratio {hi_d/lo_d:.2f}x")
    ck("D2: the wide-round fraction is genuinely unidentified (184(D))",
       0.0 < lo_w < hi_w < 1.0, f"w in [{lo_w:.4f}, {hi_w:.4f}]")

    lo_p, hi_p = ranked_leg_share_bounds("beagle")
    ck("corrected ranked beagle band brackets the published 2.2 % point",
       lo_p < ranked_pct < hi_p, f"[{lo_p:.3f} %, {hi_p:.3f} %]")
    ck("corrected band's LOW end still clears the +0.283 % ranked MDE",
       lo_p > RANKED_MDE_PCT, f"lo={lo_p:.3f} % = {lo_p/RANKED_MDE_PCT:.2f}x MDE")

    lr, rr, rl = transfer_route()
    ck("188(A) R/tau route lands inside the recomputed-count band",
       lo_p < rl < hi_p, f"R/tau route = {rl:.3f} %")
    ck("188(A) route disagrees with the published 2.0 % by more than 5 %",
       abs(rl - diluted_pct) / diluted_pct > 0.05,
       f"{rl:.3f} % vs {diluted_pct:.3f} %")

    # POSITIVE CONTROL 1: a harness that cannot fail is not a harness.
    saved = RANKED["beagle"]
    RANKED["beagle"] = (saved[0], 0.0, saved[2])       # force every round narrow
    lo_z, hi_z, (lo_wz, hi_wz) = ranked_dispatch_bounds("beagle")
    RANKED["beagle"] = saved
    ck("POSITIVE CONTROL: mean draft 0 forces the wide fraction to 0",
       lo_wz == 0.0 and hi_wz == 0.0, f"w=[{lo_wz}, {hi_wz}]")

    # POSITIVE CONTROL 2: at mean draft 8 every round must be wide.
    saved = RANKED["beagle"]
    RANKED["beagle"] = (saved[0], float(MAX_DRAFT), saved[2])
    _, _, (lo_w8, hi_w8) = ranked_dispatch_bounds("beagle")
    RANKED["beagle"] = saved
    ck("POSITIVE CONTROL: mean draft 8 forces the wide fraction to 1",
       abs(lo_w8 - 1.0) < 1e-9 and abs(hi_w8 - 1.0) < 1e-9,
       f"w=[{lo_w8}, {hi_w8}]")

    # NEGATIVE CONTROL: if the ranked round count equalled the local one AND
    # the width mixture matched, D2 would vanish and the ledger would be right.
    per_round_local = ARM_A_DISPATCHES / ARM_A_ROUNDS
    ck("NEGATIVE CONTROL: local per-round dispatch count sits in the legal "
       "range implied by its own mean draft",
       FULL_ATTENTION_LAYERS * WIDE_PER_LAYER[0] <= per_round_local
       <= FULL_ATTENTION_LAYERS * WIDE_PER_LAYER[1],
       f"{per_round_local:.2f} per round, mean draft {ARM_A_MEAN_DRAFT}")

    ck("ranked beagle runs MORE rounds than the local arm (41 % more)",
       RANKED["beagle"][0] > ARM_A_ROUNDS,
       f"{RANKED['beagle'][0]} vs {ARM_A_ROUNDS}")
    ck("ranked beagle runs NARROWER rounds than the local arm",
       RANKED["beagle"][1] < ARM_A_MEAN_DRAFT,
       f"{RANKED['beagle'][1]} vs {ARM_A_MEAN_DRAFT}")

    width = max(len(n) for n, _, _ in checks)
    bad = 0
    for name, ok, detail in checks:
        flag = "PASS" if ok else "FAIL"
        bad += 0 if ok else 1
        print(f"[{flag}] {name.ljust(width)}  {detail}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def report():
    per, leg, added = per_dispatch_seconds_upper_bound()
    total_ms, local_pct, ranked_pct, diluted_pct = ledger_186d_chain()

    print("E58 dispatch value proposition -- re-derivation of ledger 186(D)")
    print("=" * 78)
    print("\n1. THE PUBLISHED CHAIN, REPRODUCED\n")
    print(f"   E57 Arm A leg          {leg:.4f} s  "
          f"(= {ARM_A_SPT} s/tok x {DECODE_TOKENS} tokens)")
    print(f"   Arm B regression       +{ARM_B_REGRESSION_PCT} % = "
          f"{leg * ARM_B_REGRESSION_PCT / 100 * 1000:.2f} ms")
    print(f"   Arm B added dispatches {added}")
    print(f"   => per dispatch        {per * 1e6:.2f} us"
          "   <- the ledger's 22 us; denominator was never written down")
    print(f"   Arm A SDPA total       {ARM_A_DISPATCHES} x {per*1e6:.2f} us = "
          f"{total_ms:.1f} ms   (ledger: 'about 136 ms')")
    print(f"   local leg share        {local_pct:.3f} %          "
          f"(ledger: 0.74 %)")
    print(f"   / ranked beagle leg    {ranked_pct:.3f} %          "
          f"(ledger: 2.2 %)")
    print(f"   x {DILUTION_MEDIAN_PAIR} dilution      {diluted_pct:.3f} %"
          f"          (ledger: 2.0 %, '7.1 sd')")
    print(f"\n   '7.1 sd' is {diluted_pct:.3f} / {RANKED_MDE_PCT} = "
          f"{diluted_pct / RANKED_MDE_PCT:.2f}, i.e. multiples of the MDE, not "
          "standard\n   deviations. The MDE is itself 2 sd, so the label "
          "understates by 2x.")

    print("\n2. DEFECT D1 -- the 22 us constant is an UPPER BOUND\n")
    print("   The arithmetic is sound once the 512-token denominator is "
          "supplied.\n"
          "   But E57 Arm B did not add EMPTY dispatches. Narrowing the chunk "
          "predicate\n"
          "   to qL=9 pushed widths 6, 7 and 8 off the fused vector path onto "
          "the\n"
          "   composed fallback: arangeint32 x2, sv_Multiply, g2_GreaterEqual,\n"
          "   steel_gemm_fused_nt, g2_Select, block_softmax_precise,\n"
          "   steel_gemm_fused_nn (ledger :10320-10324). Those dispatches "
          "carry real\n"
          "   arithmetic and real intermediate traffic. Attributing 100 % of "
          "the\n"
          f"   +0.27 % to launch overhead makes {per*1e6:.2f} us a CEILING on "
          "per-dispatch cost.")

    print("\n3. DEFECT D2 -- the ranked dispatch COUNT was carried, not "
          "recomputed\n")
    implied = leg * 1000.0 / RANKED["beagle"][2]
    print(f"   `{total_ms:.0f} ms / 6.23 s` asserts the ranked beagle leg runs "
          f"the SAME {ARM_A_DISPATCHES}\n   dispatches our local arm ran. It "
          "does not:\n")
    print(f"     local  E57 Arm A : {ARM_A_ROUNDS} rounds, mean draft "
          f"{ARM_A_MEAN_DRAFT}")
    print(f"     ranked beagle    : {RANKED['beagle'][0]} rounds, mean draft "
          f"{RANKED['beagle'][1]}")
    print(f"\n   {RANKED['beagle'][0] / ARM_A_ROUNDS - 1:+.1%} rounds and "
          f"{RANKED['beagle'][1] / ARM_A_MEAN_DRAFT - 1:+.1%} mean draft. "
          "Dispatches per round are a STEP\n   function of width (185(B)): "
          f"qL<=5 -> {NARROW_PER_LAYER} per layer, qL 6..9 -> "
          f"{WIDE_PER_LAYER}.\n")
    print(f"   The implied leg-time multiplier is {implied:.4f}x. 188(A) "
          f"derives the formal\n   transfer multiplier as R/tau with R = "
          f"{TRANSFER_R:.4f}. The {implied:.3f} vs {TRANSFER_R:.3f} gap is "
          "exactly\n   the leg-versus-round prefill mismatch, and was never "
          "reconciled.\n")
    print("   Recomputing the count under 184(D)'s unidentifiable width "
          "histogram:\n")
    print(f"   {'prompt':<10} {'rounds':>6} {'draft':>6} {'wide frac':>16} "
          f"{'dispatches':>16} {'leg share %':>16}")
    for p in ("plutarch", "drama", "travel", "beagle", "medicine", "republic",
              "essays", "botany"):
        lo_d, hi_d, (lo_w, hi_w) = ranked_dispatch_bounds(p)
        lo_p, hi_p = ranked_leg_share_bounds(p)
        r, md, _ = RANKED[p]
        print(f"   {p:<10} {r:>6} {md:>6.3f} "
              f"{f'{lo_w:.2f}-{hi_w:.2f}':>16} "
              f"{f'{lo_d:.0f}-{hi_d:.0f}':>16} "
              f"{f'{lo_p:.2f}-{hi_p:.2f}':>16}")

    lo_p, hi_p = ranked_leg_share_bounds("beagle")
    print(f"\n   => ranked beagle: [{lo_p:.2f} %, {hi_p:.2f} %] of the leg, "
          f"a {hi_p/lo_p:.1f}x band.\n      The published 2.2 % is a point "
          "inside a band it never acknowledged.")

    print("\n4. DEFECT D3 -- a THIRD uncaught double dilution\n")
    print(f"   `{ranked_pct:.2f} %` was computed as {total_ms:.0f} ms divided "
          "by the ranked beagle LEG of\n   6233 ms. The ranked leg is "
          "prefill-INCLUSIVE (186(B): K/leg = 8.44 % on\n   beagle). So "
          f"{ranked_pct:.2f} % is already a leg-basis fraction, already diluted "
          "once.\n")
    print(f"   Multiplying by {DILUTION_MEDIAN_PAIR} to reach "
          f"{diluted_pct:.2f} % charges the same prefill a SECOND time.\n"
          "   That is the exact failure ledger 189(J) forbids. The ledger "
          "names two\n   instances -- item 122 hypothesis B, and item 189 / "
          "E55. This is the THIRD,\n   and it is the one that was ASSIGNED as "
          "E58 (ledger :10498).")

    lr, rr, rl = transfer_route()
    print("\n5. THE CORRECTED VALUE PROPOSITION\n")
    print(f"   route A -- recompute the count directly:      "
          f"[{lo_p:.2f} %, {hi_p:.2f} %] of ranked leg")
    print(f"   route B -- 188(A) R/tau, tau = 1:             {rl:.2f} % of "
          "ranked leg")
    print(f"              local leg {local_pct:.3f} % / "
          f"{1 - LOCAL_PREFILL_SHARE:.5f} = {lr:.3f} % round basis")
    print(f"              x R = {TRANSFER_R:.4f}            -> {rr:.3f} % "
          "ranked round basis")
    print(f"              x {DILUTION_MEDIAN_PAIR}                 -> "
          f"{rl:.3f} % ranked leg basis")
    print(f"\n   Route B lands inside route A's band. Both are UPPER BOUNDS, "
          "because D1's\n   per-dispatch constant is a ceiling and E58 can "
          "only remove SOME dispatches.")
    print(f"\n   Against the +{RANKED_MDE_PCT} % ranked MDE: the band's low end "
          f"is {lo_p / RANKED_MDE_PCT:.1f}x MDE and\n   route B is "
          f"{rl / RANKED_MDE_PCT:.1f}x MDE. **E58 survives the correction.** "
          "The headline number\n   was wrong; the direction was not.")

    print("\n6. WHAT E58 MUST NOW REPORT TO CLOSE D2 PERMANENTLY\n")
    print("   A single leg-total dispatch count cannot be transferred, because "
          "the\n   ranked width histogram is unidentifiable (184(D)). A "
          "dispatch census\n   reported PER ROUND WIDTH can:\n")
    print("     d(M) for M = 1..9  =  SDPA (and total) dispatches in one round "
          "of width M\n")
    print("   Then the ranked count is sum_M n_M * d(M) for ANY candidate "
          "histogram\n   {n_M}, which turns an unidentifiable point into a "
          "function of a quantity\n   the campaign already brackets. That "
          "single table is worth more than the\n   leg total E58 was "
          "originally asked for.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    report()
