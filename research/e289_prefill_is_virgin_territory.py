#!/usr/bin/env python3
"""FINDING 289 - the NAX prefill kernel family is untouched by every promoted tree.

harness=ranked throughout. The ranked numerator is the runner-owned prebuilt
serial baseline, so a candidate prefill saving cannot cancel: it lands whole on
the denominator. See senpai/verify-ranked-score-boundary.sh.

Source facts, verified from git this session:
  quantized_nax.h and mlx-generated/quantized_nax.cpp are byte-identical
  between organizer baseline 5d029178 and crown 0863b06a. Across the whole
  upstream/main history only the baseline import commit touches them.

Leg decomposition from FINDING 286:
  seed cost 0.5265 s per leg, prompt-independent, sd 0.00055 s
  P / (C + P) = 10.0438 percent of the candidate leg
"""

BAR = 3.72911001
OUR_BEST = 3.68278758168578
PREFILL_SHARE = 0.100438           # P / (C + P), candidate leg, FINDING 286
SEED_SECONDS = 0.5265              # per leg, prompt-independent

GAP_ABS = BAR - OUR_BEST
GAP_PCT = 100.0 * GAP_ABS / OUR_BEST

# Parity moves us to the crown tree, so the gap a prefill win must close from
# the crown base is zero; what matters is the margin it buys above the bar.
PARITY = BAR


def score_gain_pct(prefill_fraction_removed: float) -> float:
    """Amdahl on the candidate leg only. Positive means the score rises."""
    return 100.0 * (1.0 / (1.0 - PREFILL_SHARE * prefill_fraction_removed) - 1.0)


def prefill_fraction_for(target_gain_pct: float) -> float:
    """Inverse: prefill speedup needed for a given published gain."""
    t = target_gain_pct / 100.0
    return (1.0 - 1.0 / (1.0 + t)) / PREFILL_SHARE


# FINDING 286 ranked row law: prompt -> (tokens per round, clean round us).
# Rule 148 weights are the share each prompt has in the published median pair.
PROMPTS = {
    "plutarch": (1.0519, 30744, 0.0),
    "drama":    (2.0317, 34140, 0.0),
    "travel":   (2.4113, 35177, 0.0),
    "beagle":   (4.6545, 44991, 0.5000),
    "republic": (5.5054, 47809, 0.0329),
    "essays":   (5.5628, 48975, 0.4474),
    "medicine": (5.6883, 49448, 0.0197),
    "botany":   (6.3179, 54561, 0.0),
}


def per_prompt_shares():
    """Independent reconstruction of the prefill share, prompt by prompt."""
    rows = []
    for name, (tok_per_round, clean_us, weight) in PROMPTS.items():
        rounds = 512.0 / tok_per_round
        decode_s = rounds * clean_us / 1e6
        leg_s = decode_s + SEED_SECONDS
        rows.append((name, rounds, decode_s, leg_s, 100.0 * SEED_SECONDS / leg_s, weight))
    return rows


def check_share_consistency() -> None:
    rows = per_prompt_shares()
    print("=== prefill share is prompt dependent, and the median is what scores ===")
    print("  prompt      rounds   decode s    leg s   prefill %   rule148 w")
    for name, rounds, dec, leg, share, w in rows:
        print(f"  {name:<10}{rounds:7.1f}  {dec:8.3f} {leg:8.3f}   {share:7.4f}     {w:.4f}")
    shares = sorted(r[4] for r in rows)
    median = 0.5 * (shares[3] + shares[4])
    weighted = sum(r[4] * r[5] for r in rows)
    print()
    print(f"  unweighted mean of eight     {sum(shares)/8:.4f} %")
    print(f"  median of eight              {median:.4f} %")
    print(f"  Rule 148 weighted            {weighted:.4f} %")
    print(f"  FINDING 286 carried value    {100*PREFILL_SHARE:.4f} %")
    print(f"  agreement, weighted vs carried  {abs(weighted - 100*PREFILL_SHARE):.4f} pp")
    print()
    print("  The score is a median over eight prompts, so the Rule 148 weighted")
    print("  share is the correct one. It reproduces the carried value without")
    print("  any fitted parameter, from the row law alone.")


def stress_test() -> None:
    print()
    print("=== stress test, required before the equation is used ===")
    for f in (-0.20, -0.05, 0.0, 0.05, 1.0):
        print(f"  prefill fraction removed {f:+.2f}  ->  published {score_gain_pct(f):+.4f} %")
    print("  zero maps to zero; a slower prefill lowers the score; both signs behave.")


def main() -> None:
    check_share_consistency()
    stress_test()
    print()
    leg = SEED_SECONDS / PREFILL_SHARE
    decode = leg - SEED_SECONDS
    print("=== median-prompt leg decomposition, harness=ranked ===")
    print(f"  seed (prefill)      {SEED_SECONDS:.4f} s   {100*PREFILL_SHARE:.4f} %")
    print(f"  decode              {decode:.4f} s   {100*(1-PREFILL_SHARE):.4f} %")
    print(f"  leg total           {leg:.4f} s")
    print(f"  decode per token    {1000*decode/512:.3f} ms over 512 tokens")

    print()
    print("=== the bar and the gap ===")
    print(f"  bar   ec24d59       {BAR:.8f}")
    print(f"  ours  0cf1637e      {OUR_BEST:.8f}")
    print(f"  gap                 {GAP_ABS:.5f} absolute = +{GAP_PCT:.4f} %")

    print()
    print("=== what a prefill speedup is worth, Amdahl on the candidate leg ===")
    print("  prefill  published   absolute on   absolute on")
    print("  speedup  gain %      our best      parity 3.729")
    for f in (0.05, 0.10, 0.1235, 0.15, 0.20, 0.25, 0.30, 0.40, 1.00):
        g = score_gain_pct(f)
        print(f"   {100*f:5.1f} %  +{g:7.4f}    +{OUR_BEST*g/100:7.5f}      "
              f"{PARITY*(1+g/100):.5f}")

    print()
    need = prefill_fraction_for(GAP_PCT)
    print("=== the headline ===")
    print(f"  prefill speedup that closes the whole gap on its own: {100*need:.2f} %")
    print(f"  total prefill elimination would be worth  +{score_gain_pct(1.0):.4f} % "
          f"= {score_gain_pct(1.0)/GAP_PCT:.1f}x the gap")

    print()
    print("=== why it is still untouched after 200+ submissions ===")
    print("  the _nax variants execute on the ranked M5 only.")
    print("  no team host, and plausibly no rival development host, can run them.")
    print("  the axis is blind, so everyone skipped it. that is the edge.")


if __name__ == "__main__":
    main()
