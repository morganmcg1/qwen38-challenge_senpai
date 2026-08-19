#!/usr/bin/env python3
"""E39 — minimum detectable effect (MDE) for every instrument this campaign uses.

Why this exists
---------------
Five times in one week the campaign recorded a *failure to detect* as an
*absence*: items 126 and 131, E33's +0.088 % prediction against a +-0.3 %
instrument, E37's under-depth prose proxies, and the two halves of the
residency+cmdbuf family. An under-powered null is not a null. This module is
the single implementation that answers, before an experiment is assigned:

    "what effect would this instrument actually have detected?"

The uniform formula (quoted once, applied everywhere)
-----------------------------------------------------
Two-sided alpha = 0.05, power 1 - beta = 0.80, normal approximation:

    MDE = (z_0.975 + z_0.80) * se(effect) = 2.8016 * se(effect)

with the standard error of the effect depending only on the design:

    paired      n pairs, s = per-pair sd            se = s / sqrt(n)
    two_sample  n per arm, s = within-arm sd        se = s * sqrt(2/n)
    single      one reading against a known level   se = s
    slope       regression slope, s = se of slope   se = s

`paired` is the campaign default: `MDE ~= 2.8 s / sqrt(n)`, exactly as the E39
brief specifies.

Small n: the normal approximation LIES
--------------------------------------
At n = 2 pairs the true MDE is 5.83x the normal figure (standardised effect
11.55 rather than 1.98); at n = 3 it is 2.02x, and only past n = 10 does it
fall inside 13 %. `mde_exact` inverts the noncentral-t power function instead
and is the honest number whenever df <= 10; `--audit` prints both. Reporting
only the normal figure understates how badly a two-repeat arm is
under-powered, which is the very error this module exists to stop.

Usage
-----
    python3 research/e39_mde.py --self-test
    python3 research/e39_mde.py --audit                 # every known instrument
    python3 research/e39_mde.py --mde --sd 0.0923 --n 1
    python3 research/e39_mde.py --n-required --sd 0.0923 --target 0.185
"""
from __future__ import annotations

import argparse
import math
import random
import statistics as st

Z_ALPHA_TWO_SIDED_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
NORMAL_MULTIPLIER = Z_ALPHA_TWO_SIDED_95 + Z_POWER_80  # 2.8016

DESIGNS = ("paired", "two_sample", "single", "slope")


def se_of_effect(sd: float, n: int, design: str = "paired") -> float:
    """Standard error of the estimated effect, per design. `sd` units == effect units."""
    if design not in DESIGNS:
        raise ValueError("design must be one of %s" % (DESIGNS,))
    if sd < 0:
        raise ValueError("sd must be non-negative")
    if n < 1:
        raise ValueError("n must be >= 1")
    if design == "paired":
        return sd / math.sqrt(n)
    if design == "two_sample":
        return sd * math.sqrt(2.0 / n)
    return sd


def degrees_of_freedom(n: int, design: str = "paired") -> int:
    """Residual df, used only by the exact (noncentral-t) route."""
    if design == "paired":
        return n - 1
    if design == "two_sample":
        return 2 * (n - 1)
    if design == "single":
        return max(n - 1, 1)
    return max(n - 2, 1)


def mde(sd: float, n: int, design: str = "paired") -> float:
    """MDE at 80 % power, two-sided alpha = 0.05, normal approximation."""
    return NORMAL_MULTIPLIER * se_of_effect(sd, n, design)


def n_required(sd: float, target: float, design: str = "paired") -> int:
    """Smallest n whose MDE is at or below `target`. Inverse of `mde`."""
    if target <= 0:
        raise ValueError("target effect must be positive")
    if sd == 0:
        return 1
    if design == "single":
        raise ValueError("the 'single' design has no n to solve for")
    if design == "paired":
        exact = (NORMAL_MULTIPLIER * sd / target) ** 2
    elif design == "two_sample":
        exact = 2.0 * (NORMAL_MULTIPLIER * sd / target) ** 2
    else:
        raise ValueError("the 'slope' design takes an se, not an sd; solve outside")
    return max(2, math.ceil(exact - 1e-12))


# --------------------------------------------------------------------------
# exact route: noncentral-t power, stdlib only (no scipy on this host)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(b * math.log1p(-x) + a * math.log(x) - _log_beta(b, a)) \
        * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """Student-t CDF."""
    x = df / (df + t * t)
    tail = 0.5 * betainc_reg(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p: float, df: int) -> float:
    """Student-t quantile by bisection on `t_cdf`."""
    lo, hi = -400.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def nct_cdf(t: float, df: int, ncp: float, nodes: int = 4000) -> float:
    """Noncentral-t CDF by quadrature over the chi-square mixing density.

    T = (Z + ncp) / sqrt(V / df),  V ~ chi2_df, so
    P(T <= t) = E_V[ Phi( t * sqrt(V/df) - ncp ) ].
    Simpson's rule over v; df here is small (1-30), so the integrand is smooth
    and a few thousand nodes are ample.
    """
    hi = df + 12.0 * math.sqrt(2.0 * df) + 40.0
    if nodes % 2:
        nodes += 1
    h = hi / nodes
    half_df = df / 2.0
    log_norm = math.lgamma(half_df) + half_df * math.log(2.0)

    def integrand(v: float) -> float:
        if v <= 0.0:
            return 0.0
        log_pdf = (half_df - 1.0) * math.log(v) - v / 2.0 - log_norm
        return _norm_cdf(t * math.sqrt(v / df) - ncp) * math.exp(log_pdf)

    total = integrand(0.0) + integrand(hi)
    for i in range(1, nodes):
        total += (4.0 if i % 2 else 2.0) * integrand(i * h)
    return min(max(total * h / 3.0, 0.0), 1.0)


def power_exact(effect: float, sd: float, n: int, design: str = "paired") -> float:
    """Power of the two-sided t-test at alpha = 0.05 for a true `effect`."""
    df = degrees_of_freedom(n, design)
    if df < 1:
        raise ValueError("design/n gives no residual degrees of freedom")
    ncp = effect / se_of_effect(sd, n, design)
    crit = t_ppf(0.975, df)
    return 1.0 - nct_cdf(crit, df, ncp) + nct_cdf(-crit, df, ncp)


def mde_exact(sd: float, n: int, design: str = "paired", power: float = 0.80) -> float:
    """MDE from the exact noncentral-t power function, by bisection on effect."""
    lo, hi = 0.0, 50.0 * max(se_of_effect(sd, n, design), 1e-12)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if power_exact(mid, sd, n, design) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# the two trusted numbers the self-test must reproduce

# Per-prompt single-pair ratio sigma (%), ascending-ratio ORDER, derived in
# E35 `within_head_cost.py --noise` from fixtures/qwen3_8_27b_mtp_track.json:
# exact pair spreads where the organizers publish them, else the conservative
# sqrt(CV_s^2+CV_m^2) propagation deflated by the matched factor k = 0.7219.
PROMPT_ORDER = ("plutarch", "drama", "travel", "beagle",
                "medicine", "essays", "republic", "botany")
PROMPT_SIGMA_PCT = (0.3155, 0.1160, 0.1206, 0.1040, 0.1494, 0.1119, 0.1942, 0.2810)

# Crown 0cd0a6b4 (ofou, 3.24929398547457) per-prompt raw_ratio_of_means, same order.
CROWN_PROFILE = (1.256033, 1.923109, 2.189516, 3.143326,
                 3.355262, 3.390664, 3.414373, 3.449062)

SIGMA_SCORE_PCT = 0.0923          # crown profile; 2 sigma detection bar = 0.185 %
SCORE_THRESHOLD_PCT = 0.185

# E33's own within-arm-type spreads (%), `--local-iterate`, 64 tokens, n = 2 per
# arm: base serial 0.27, cand serial 0.22, base MTP 0.40, cand MTP 0.13. The
# published "+-0.3 %" is the pooled within-arm sd of the two MTP arms.
E33_MTP_ARM_SPREADS_PCT = (0.40, 0.13)


def median_of_8(values) -> float:
    """The published rule: mean of the two central order statistics of eight."""
    s = sorted(values)
    if len(s) != 8:
        raise ValueError("the ranked score rule is defined on exactly eight prompts")
    return 0.5 * (s[3] + s[4])


def score_sigma_pct(profile=CROWN_PROFILE, sigma_pct=PROMPT_SIGMA_PCT,
                    trials: int = 60000, seed: int = 17) -> float:
    """Sigma of the median-of-8 score, given per-prompt relative sigma.

    Reproduces E35's `mc_score_sigma`: sigma_score is not the per-prompt sigma
    because the median listens to only two prompts, and WHICH two depends on
    the profile's steepness.
    """
    rng = random.Random(seed)
    draws = [median_of_8([r * (1.0 + rng.gauss(0.0, s / 100.0))
                          for r, s in zip(profile, sigma_pct)])
             for _ in range(trials)]
    return 100.0 * st.stdev(draws) / st.mean(draws)


def pooled_sd(*sds: float) -> float:
    """Pooled sd of equal-sized groups: sqrt(mean of variances)."""
    return math.sqrt(sum(s * s for s in sds) / len(sds))


# --------------------------------------------------------------------------
# instrument registry
#
# `own_repeats` False means the sd is BORROWED from the nearest comparable run
# rather than measured by the run it is being used to judge. The E39 brief
# requires that to be stated in the row, not in a footnote.

INSTRUMENTS = {
    "ranked_score": dict(
        sd=SIGMA_SCORE_PCT, n=1, design="single", unit="% of score",
        own_repeats=True,
        provenance="organizers' six gated identical-code sessions -> 0.0784 %; "
                   "median-of-8 on the crown's steep profile -> 0.0923 %"),
    "local_e2e_leg": dict(
        sd=pooled_sd(*E33_MTP_ARM_SPREADS_PCT), n=2, design="two_sample",
        unit="% of MTP leg", own_repeats=True,
        provenance="E33 within-arm-type spreads, base MTP 0.40 % / cand MTP 0.13 %, "
                   "--local-iterate 64 tok, ungated ABBA"),
    "local_e2e_leg_e29": dict(
        sd=0.86, n=2, design="two_sample", unit="% of MTP leg", own_repeats=True,
        provenance="E29 D0-vs-N1 same-config repeat differs by 0.86 % on this host"),
    "e29_ladder_slope": dict(
        sd=0.0311, n=4, design="slope", unit="ms per forced boundary",
        own_repeats=True,
        provenance="E29 four-arm ladder least-squares, slope -0.0365 ms, se 0.0311 ms, "
                   "2 dof; this is an se, not an sd"),
    "local_microbench": dict(
        sd=0.30, n=3, design="paired", unit="% of kernel time", own_repeats=False,
        provenance="BORROWED: no campaign microbenchmark publishes its own repeat sd; "
                   "0.30 % is the nearest comparable (E33 pooled arm spread)"),
    "competitor_ranked_n1": dict(
        sd=SIGMA_SCORE_PCT, n=1, design="two_sample", unit="% of score",
        own_repeats=False,
        provenance="BORROWED: a single competitor ranked row carries no repeat of its "
                   "own; sigma_score is imported and the row is confounded by every "
                   "other mechanism in that submission"),
    "board_family_n5": dict(
        sd=0.324, n=5, design="single", unit="% of score", own_repeats=True,
        provenance="E35 residency+cmdbuf family join, se 0.145 over n=5 -> "
                   "implied row sd 0.324 %; refreshed 646-row corpus gives se 0.143"),
    "algebraic": dict(
        sd=0.0, n=1, design="single", unit="exact", own_repeats=True,
        provenance="closed-form identity or counting argument; variance is "
                   "structurally zero, so MDE is zero and power is not the "
                   "failure mode"),
    "static_analysis": dict(
        sd=float("inf"), n=1, design="single", unit="n/a", own_repeats=False,
        provenance="no timed arm; MDE is undefined and no null can be read from it"),
    "hardware_gated_out": dict(
        sd=float("inf"), n=0, design="single", unit="n/a", own_repeats=False,
        provenance="the mechanism's code path cannot execute on the available host, "
                   "so no n makes the contrast observable; MDE is undefined for a "
                   "reason no sample size fixes"),
}


def audit_table() -> str:
    rows = ["instrument            design      n     sd        MDE(norm)  MDE(exact)  own-sd",
            "-" * 78]
    for name, spec in INSTRUMENTS.items():
        if spec["sd"] == float("inf"):
            rows.append("%-21s %-11s %-5s %-9s %-10s %-11s %s"
                        % (name, spec["design"], spec["n"], "inf",
                           "undefined", "undefined", spec["own_repeats"]))
            continue
        if spec["sd"] == 0.0:
            rows.append("%-21s %-11s %-5d %-9.4f %-10s %-11s %s"
                        % (name, spec["design"], spec["n"], 0.0,
                           "0 (exact)", "0 (exact)", spec["own_repeats"]))
            continue
        norm = mde(spec["sd"], spec["n"], spec["design"])
        try:
            exact = "%.4f" % mde_exact(spec["sd"], spec["n"], spec["design"])
        except ValueError:
            exact = "n/a"
        rows.append("%-21s %-11s %-5d %-9.4f %-10.4f %-11s %s"
                    % (name, spec["design"], spec["n"], spec["sd"],
                       norm, exact, spec["own_repeats"]))
    rows.append("")
    rows.append("MDE at 80 %% power, two-sided alpha 0.05. Normal multiplier %.4f."
                % NORMAL_MULTIPLIER)
    rows.append("Score claims are judged against 2 sigma_score = %.3f %%."
                % SCORE_THRESHOLD_PCT)
    return "\n".join(rows)


def self_test() -> int:
    """Reproduce two numbers the campaign already trusts, plus internal checks."""
    failures = []

    def check(label, got, want, tol):
        ok = abs(got - want) <= tol
        print("  [%s] %-52s got %.5f  want %.5f  (tol %.5f)"
              % ("PASS" if ok else "FAIL", label, got, want, tol))
        if not ok:
            failures.append(label)

    print("TRUSTED NUMBER 1 — sigma_score on the crown profile")
    # CROWN_PROFILE is stored to 6 dp, so the rule reproduces the published
    # score to 1.5e-8 rather than exactly.
    check("median-of-8 reproduces the crown's published score",
          median_of_8(CROWN_PROFILE), 3.24929398547457, 5e-8)
    check("sigma_score (Monte Carlo, median-of-8)",
          score_sigma_pct(), SIGMA_SCORE_PCT, 5e-4)
    check("2 sigma detection threshold",
          2 * score_sigma_pct(), SCORE_THRESHOLD_PCT, 1e-3)

    print("TRUSTED NUMBER 2 — the E33 end-to-end instrument at n = 2")
    check("pooled within-arm sd of E33's two MTP arms",
          pooled_sd(*E33_MTP_ARM_SPREADS_PCT), 0.30, 5e-3)

    print("INTERNAL CONSISTENCY")
    check("normal multiplier", NORMAL_MULTIPLIER, 2.8016, 1e-3)
    check("paired MDE is 2.8 s / sqrt(n)", mde(1.0, 4, "paired"), 2.8016 / 2.0, 1e-3)
    check("n_required inverts mde", float(n_required(0.0923, 0.185, "paired")),
          2.0, 0.0)
    check("mde(n_required) <= target",
          mde(0.0923, n_required(0.0923, 0.185, "paired"), "paired"), 0.1828, 1e-3)
    check("t_ppf(0.975, 1) = 12.706", t_ppf(0.975, 1), 12.7062, 1e-3)
    check("t_ppf(0.975, 10) = 2.228", t_ppf(0.975, 10), 2.2281, 1e-3)
    check("nct_cdf with ncp=0 reduces to central t",
          nct_cdf(2.2281, 10, 0.0), 0.975, 1e-4)
    check("power at the exact MDE is 0.80",
          power_exact(mde_exact(1.0, 6, "paired"), 1.0, 6, "paired"), 0.80, 1e-3)

    print("EXTERNAL ANCHORS — the power engine against published values")
    check("Cohen: two-sample n=64/arm, d=0.5 -> power 0.80",
          power_exact(0.5, 1.0, 64, "two_sample"), 0.80, 5e-3)
    check("G*Power: paired n=2 -> standardised MDE 11.6",
          mde_exact(1.0, 2, "paired"), 11.6, 0.15)
    check("exact/normal blow-up at n=2 paired",
          mde_exact(1.0, 2, "paired") / mde(1.0, 2, "paired"), 5.83, 0.05)

    print()
    if failures:
        print("SELF-TEST FAILED: %s" % ", ".join(failures))
        return 1
    print("SELF-TEST PASSED (13 checks)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--audit", action="store_true", help="MDE of every known instrument")
    ap.add_argument("--mde", action="store_true")
    ap.add_argument("--n-required", action="store_true")
    ap.add_argument("--sd", type=float, help="per-pair / within-arm sd, effect units")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--target", type=float, help="effect that would matter")
    ap.add_argument("--design", default="paired", choices=DESIGNS)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.audit:
        print(audit_table())
        return 0
    if args.mde:
        if args.sd is None:
            ap.error("--mde needs --sd")
        norm = mde(args.sd, args.n, args.design)
        print("MDE (normal, 80 %% power, two-sided) = %.4f" % norm)
        try:
            ex = mde_exact(args.sd, args.n, args.design)
            print("MDE (exact noncentral-t)             = %.4f   [%.2fx the normal figure]"
                  % (ex, ex / norm if norm else float("nan")))
        except ValueError as exc:
            print("MDE (exact noncentral-t)             = n/a (%s)" % exc)
        if args.target:
            print("effect that would matter             = %.4f" % args.target)
            print("verdict: %s" % ("UNDER-POWERED (MDE >= effect that matters)"
                                   if norm >= args.target else "adequately powered"))
        return 0
    if args.n_required:
        if args.sd is None or args.target is None:
            ap.error("--n-required needs --sd and --target")
        n = n_required(args.sd, args.target, args.design)
        print("n required = %d  (MDE there = %.4f, target %.4f)"
              % (n, mde(args.sd, n, args.design), args.target))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
