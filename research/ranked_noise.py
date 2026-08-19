"""Authoritative noise model for the official ranked instrument (ledger 193).

WHY THIS MODULE EXISTS
----------------------
Nine campaign modules define `RANKED_MDE_PCT = 0.283`. That number is wrong by
a factor of 7.4 and every one of them inherited it. It was derived as two
standard deviations of a 0.2257 % per-prompt-per-leg jitter figure, under the
assumption that the median over eight prompts averages that jitter down.

Both inputs fail:

  * 0.2257 % describes the PINNED SERIAL leg. Measured directly on identical
    submitted surfaces, that leg's median pair delta is 0.181 %, so the old
    figure was a good estimate -- of the wrong leg.
  * The published score is dominated by the CANDIDATE MTP leg, which is 3.62x
    noisier, and its noise is a per-run common mode that moves all eight
    prompts together. The median over eight prompts therefore averages away
    almost nothing.

This module carries the measured replacement. It does not rewrite the nine
historical call sites: their arithmetic was internally correct at the time and
their self-tests pin their own published numbers. Import from here for any new
ranked pricing.

The names below deliberately carry their DOMAIN, which is the lesson of the
`PSI_MTP` collision recorded in ledger 192(R). `RANKED_*` means the official M5
ranked instrument. `LOCAL_*` means a matched-identity-tuple ABBA measurement on
a student Mac.

MEASUREMENT
-----------
Replicates are keyed on the git tree of the submitted surface, never on the
announced `submissionCommitSha` (512 scored rows carry 417 distinct values of
it and recover zero replicate groups). Source data: the full Yukon dump, 767
rows, 512 scored; 18 identical-surface groups covering 44 rows and 37
within-group pairs, 26 degrees of freedom.

Reproduce with `_advisor_scratch/verify193b.py` and `verify193c.py`.

Self-test:  python3 research/ranked_noise.py
"""
from __future__ import annotations

import math

# --- measured, ledger 193(B) -------------------------------------------------

RANKED_SD_ONE_RUN_PCT = 0.756
"""Pooled sd of ONE official ranked run, pct of score. 18 groups, dof 26."""

RANKED_SD_DIFFERENCE_PCT = RANKED_SD_ONE_RUN_PCT * math.sqrt(2.0)
"""sd of the difference of two ranked runs = 1.069 %."""

RANKED_MDE_ONE_PAIR_PCT = 1.96 * RANKED_SD_DIFFERENCE_PCT
"""95 % two-sided detection threshold for a single (S, S^) pair = 2.10 %."""

RANKED_MEDIAN_REPLICATE_DELTA_PCT = 1.113
"""Median disagreement of two ranked runs of a BYTE-IDENTICAL surface."""

RANKED_MAX_REPLICATE_DELTA_PCT = 2.126
RANKED_REPLICATE_FRAC_OVER_1PCT = 0.514
"""Over half of identical-surface replicate pairs disagree by more than 1 %."""

# --- per-leg decomposition, ledger 193(D) ------------------------------------

RANKED_SERIAL_LEG_MEDIAN_PAIR_PCT = 0.181
"""Pinned, runner-owned, prebuilt baseline leg. n = 296 prompt-level pairs."""

RANKED_CANDIDATE_LEG_MEDIAN_PAIR_PCT = 0.589
"""Candidate MTP leg, built fresh per run. n = 296. This dominates the score."""

RANKED_SERIAL_BASELINE_DRIFT_PCT_PER_HOUR = 0.000058
"""Regressed over 109.8 h, t = 0.48. The pinned baseline does not drift."""

# --- empty-diff null pairs, ledger 193(C) ------------------------------------

EMPTY_DIFF_NULL_DELTAS_PCT = (+0.0081, +0.1556, -0.6106, -0.6786, -1.2737)
"""Five (S, S^) pairs whose `git diff S^..S` is empty. True effect is exactly
zero, so this is the instrument measuring a known null."""

# --- local instrument, unchanged and still valid -----------------------------

LOCAL_NULL_FLOOR_PCT = 0.0629
"""Local end-to-end null floor, matched identity tuple, ABBA. Measured on this
fleet. Ledger 193 does NOT touch this number."""

# --- retracted ---------------------------------------------------------------

RETRACTED_RANKED_MDE_PCT = 0.283
"""RETRACTED by ledger 193(E). Serial-leg jitter applied to the score, and an
independence assumption that the candidate-leg common mode violates. Present
only so the nine historical call sites remain traceable to their correction."""

RETRACTED_RANKED_JITTER_PCT = 0.2257
"""RETRACTED as a SCORE jitter. Still a good estimate of the SERIAL leg."""


def ranked_over_local_coarseness() -> float:
    """How much coarser the ranked instrument is than a local ABBA pair."""
    return RANKED_SD_DIFFERENCE_PCT / LOCAL_NULL_FLOOR_PCT


def mde_correction_factor() -> float:
    """Divide any published `N x MDE` claim by this before believing it."""
    return RANKED_MDE_ONE_PAIR_PCT / RETRACTED_RANKED_MDE_PCT


def winners_curse_bias_pct() -> float:
    """E[observed | true effect 0, observed > 0] for one ranked run."""
    return RANKED_SD_ONE_RUN_PCT * math.sqrt(2.0 / math.pi)


def prob_outscoring(target_score: float, candidate_true_score: float) -> float:
    """Probability one ranked run of `candidate_true_score` beats `target_score`.

    Normal approximation with sd = RANKED_SD_ONE_RUN_PCT on the candidate only,
    treating the already-published target as fixed.
    """
    z = (target_score / candidate_true_score - 1.0) * 100.0 / RANKED_SD_ONE_RUN_PCT
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def pairs_needed_for(effect_pct: float, power: float = 0.80) -> int:
    """Independent (S, S^) pairs needed to detect `effect_pct` at 95 %/power."""
    z_beta = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}[power]
    n = ((1.96 + z_beta) * RANKED_SD_DIFFERENCE_PCT / effect_pct) ** 2
    return math.ceil(n)


def _self_test() -> None:
    checks: list[tuple[str, float, float, float]] = []

    def ck(name: str, got: float, want: float, tol: float) -> None:
        checks.append((name, got, want, tol))
        assert abs(got - want) <= tol, f"{name}: got {got!r} want {want!r}"

    ck("sd of a difference", RANKED_SD_DIFFERENCE_PCT, 1.069, 5e-4)
    ck("single-pair MDE", RANKED_MDE_ONE_PAIR_PCT, 2.096, 1e-3)
    ck("MDE correction factor", mde_correction_factor(), 7.4047, 1e-3)
    ck("candidate/serial leg noise ratio",
       RANKED_CANDIDATE_LEG_MEDIAN_PAIR_PCT / RANKED_SERIAL_LEG_MEDIAN_PAIR_PCT,
       3.254, 1e-3)
    ck("ranked is this many times coarser than local",
       ranked_over_local_coarseness(), 17.00, 0.05)
    ck("winner's curse bias", winners_curse_bias_pct(), 0.6032, 1e-3)

    # The empty-diff nulls must be consistent with the pooled sd.
    rms = math.sqrt(sum(d * d for d in EMPTY_DIFF_NULL_DELTAS_PCT)
                    / len(EMPTY_DIFF_NULL_DELTAS_PCT))
    ck("empty-diff rms about zero", rms, 0.7043, 1e-3)
    assert rms < RANKED_SD_DIFFERENCE_PCT, (
        "a known null must not exceed the sd of a difference")

    # Ledger 193(H): our deficit against the frontier is under one sd.
    frontier, ours = 3.24985583421771, 3.23250848263467
    deficit_pct = (frontier / ours - 1.0) * 100.0
    ck("published deficit", deficit_pct, 0.5366, 5e-4)
    ck("deficit in sd of one run", deficit_pct / RANKED_SD_ONE_RUN_PCT,
       0.7098, 1e-3)
    assert deficit_pct < RANKED_SD_ONE_RUN_PCT, (
        "ledger 193(H): the deficit is inside one run's sd")

    # Organizer main's own identical-tree replicate is worse than our deficit.
    main_a, main_b = 3.24929398547457, 3.22945266
    main_spread = (main_a / main_b - 1.0) * 100.0
    ck("organizer main's own replicate spread", main_spread, 0.6144, 5e-3)
    assert main_spread > deficit_pct, (
        "ledger 193(H): main's own replicate spread exceeds our whole deficit")

    # Ledger 193(J): a candidate equal to organizer main is a coin flip.
    p = prob_outscoring(frontier, main_a)
    ck("P(beating the frontier | true score == organizer main)", p, 0.4907, 2e-3)

    # A local ABBA pair resolves what 17 ranked pairs cannot.
    assert pairs_needed_for(0.5) >= 35, "power arithmetic drifted"

    width = max(len(c[0]) for c in checks)
    for name, got, want, _ in checks:
        print(f"  ok  {name:<{width}}  {got:.6g}")
    print(f"\n{len(checks)} checks passed.")
    print(f"\nRANKED single-pair MDE  {RANKED_MDE_ONE_PAIR_PCT:.3f} %"
          f"   (was {RETRACTED_RANKED_MDE_PCT} %, wrong by "
          f"{mde_correction_factor():.1f}x)")
    print(f"LOCAL  ABBA null floor  {LOCAL_NULL_FLOOR_PCT:.4f} %"
          f"   ({ranked_over_local_coarseness():.0f}x more sensitive)")
    print(f"\nIndependent (S, S^) pairs needed at 80 % power:"
          f"  +0.5 % -> {pairs_needed_for(0.5)},"
          f"  +1 % -> {pairs_needed_for(1.0)},"
          f"  +2 % -> {pairs_needed_for(2.0)}")


if __name__ == "__main__":
    _self_test()
