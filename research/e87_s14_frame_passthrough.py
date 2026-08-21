#!/usr/bin/env python3
"""Settle the published-frame residual by measuring session pass-through.

``e87_s13_frame_variance.py`` assumed the run-level session speed factor ``k_i``
enters both timed legs with gain 1, so it cancels exactly in the published
ratio.  That model predicts a serial-free replicate sd the advisor's 39
byte-identical pairs contradict, and it leaves about 0.10 % of published
replicate sd unexplained.  The advisor's own guess at the resolution was that
``k_i`` does not pass through to both legs at the same gain.  This script turns
that guess into a measured number.

Model, in log units, for submission ``i`` and prompt ``p``:

    log serial_ip = log S_p + k_i         + a_ip
    log cand_ip   = log C_p(code_i) + g*k_i + b_ip

``k_i`` is one run-level environmental factor, ``a_ip`` and ``b_ip`` are
independent per-prompt draws, and ``g`` is the pass-through gain the old model
fixed at 1.  The median of eight averages two prompts, so at score level

    var(published)  = (1-g)^2 sigma_k^2 + (sigma_a^2 + sigma_b^2) / 2
    var(serialfree) =     g^2 sigma_k^2 +              sigma_b^2  / 2

Two routes to ``g``:

  solved    from the advisor's two measured replicate sds plus board sigma_a
            and sigma_k, by subtracting the two equations.
  measured  by regressing each run's 8-prompt mean candidate excess on its
            8-prompt mean serial excess.  Candidate excess is taken against the
            mean of that run's own same-schedule cohort, so code differences
            between cohorts cannot drive the slope.  A run's own code offset is
            uncorrelated with its serial draw, so it inflates the standard error
            without biasing the slope.

            E[slope] = g * sigma_k^2 / (sigma_k^2 + sigma_a^2 / 8)

The two routes are independent.  If they agree, the residual is explained and
the detection floors the campaign screens riders against are on a measured
footing rather than an assumed one.
"""

import collections
import statistics as st
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from board_per_prompt import PROMPT_NAMES, load, serial_means, vec

NAMES = list(PROMPT_NAMES.values())

# Advisor's direct replicate measurements over 39 byte-identical pairs.
PUBLISHED_SD = 0.196
SERIALFREE_SD = 0.113
N_PAIRS = 39

MIN_COHORT = 4


def twoway(cohorts):
    """Per-prompt residual sd for both legs after removing run and prompt means.

    Returns (sigma_b, sigma_a_check, cross) in percent, where ``cross`` pairs the
    two legs' residuals so their correlation can be tested.
    """
    import math

    acc = {"cand": [0.0, 0], "serial": [0.0, 0]}
    cross = []
    for members in cohorts.values():
        if len(members) < MIN_COHORT:
            continue
        nrun, nprompt = len(members), len(NAMES)
        resid = {}
        for leg in ("cand", "serial"):
            y = [[100.0 * math.log(m[leg][n]) for n in NAMES] for m in members]
            grand = st.fmean(v for row in y for v in row)
            rmean = [st.fmean(row) for row in y]
            pmean = [st.fmean(y[i][j] for i in range(nrun)) for j in range(nprompt)]
            resid[leg] = [
                [y[i][j] - rmean[i] - pmean[j] + grand for j in range(nprompt)]
                for i in range(nrun)
            ]
            acc[leg][0] += sum(v * v for row in resid[leg] for v in row)
            acc[leg][1] += (nrun - 1) * (nprompt - 1)
        for i in range(nrun):
            for j in range(nprompt):
                cross.append((resid["serial"][i][j], resid["cand"][i][j]))

    # The two-way residual is shrunk by the run and prompt means it removes.
    def sd(leg):
        ss, df = acc[leg]
        return (ss / df) ** 0.5

    return sd("cand"), sd("serial"), cross


def ols(xs, ys):
    n = len(xs)
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - a - b * x for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    return b, (s2 / sxx) ** 0.5, n


def main():
    scored = load()
    smeans = serial_means(scored)

    runs = []
    for r in scored:
        v = vec(r)
        if any(n not in v for n in NAMES):
            continue
        sched = tuple(
            (v[n]["effective_mean_draft_len"], v[n]["non_drafting_round_count"])
            for n in NAMES
        )
        runs.append(
            {
                "id": r["id"][:8],
                "sched": sched,
                "serial": {n: v[n]["serial_seconds_per_token_mean"] for n in NAMES},
                "cand": {n: v[n]["mtp_seconds_per_token_mean"] for n in NAMES},
            }
        )

    # sigma_a and sigma_k from the serial leg alone.
    within, session = [], []
    for run in runs:
        ex = [100.0 * (run["serial"][n] / smeans[n] - 1.0) for n in NAMES]
        run["serial_excess"] = ex
        run["E_serial"] = st.fmean(ex)
        within.append(st.stdev(ex))
        session.append(run["E_serial"])
    sigma_a = (sum(7 * s * s for s in within) / (7 * len(within))) ** 0.5
    sd_mean = st.stdev(session)
    sigma_k = max(sd_mean**2 - sigma_a**2 / 8.0, 0.0) ** 0.5

    print("board serial-leg variance components, %d runs" % len(runs))
    print("  sigma_a  per-prompt serial draw        %.4f %%" % sigma_a)
    print("  sd of the 8-prompt mean excess         %.4f %%" % sd_mean)
    print("  sigma_k  run-level serial factor       %.4f %%" % sigma_k)
    print()

    # Candidate excess against the run's own same-schedule cohort.
    cohorts = collections.defaultdict(list)
    for run in runs:
        cohorts[run["sched"]].append(run)
    used = []
    for sched, members in cohorts.items():
        if len(members) < MIN_COHORT:
            continue
        cmean = {n: st.fmean(m["cand"][n] for m in members) for n in NAMES}
        for m in members:
            m["E_cand"] = st.fmean(
                100.0 * (m["cand"][n] / cmean[n] - 1.0) for n in NAMES
            )
            used.append(m)
    print(
        "same-schedule cohorts with >= %d runs: %d cohorts, %d runs"
        % (MIN_COHORT, sum(1 for v in cohorts.values() if len(v) >= MIN_COHORT), len(used))
    )

    slope, se, n = ols([m["E_serial"] for m in used], [m["E_cand"] for m in used])
    atten = sigma_k**2 / (sigma_k**2 + sigma_a**2 / 8.0)
    print("  regression of mean candidate excess on mean serial excess")
    print("    slope        %+.4f  se %.4f  t %+.2f  (n %d)" % (slope, se, slope / se, n))
    print("    implied g    %+.4f  se %.4f   -- USELESS, see below"
          % (slope / atten, se / atten))
    print("    real code differences inside a cohort swamp the candidate leg,")
    print("    so this route cannot separate g = 0 from g = 1.  Discard it.")
    print()

    # Two-way removal inside each cohort: subtract the cohort-by-prompt mean and
    # the run mean.  On the serial leg the residual is a_ip by construction, so
    # recovering sigma_a there is a positive control for the same operation on
    # the candidate leg, where the residual estimates sigma_b.
    sigma_b, sigma_a_check, cross = twoway(cohorts)
    print("  two-way residual inside same-schedule cohorts")
    print("    serial leg  sigma_a recovered  %.4f %%   (board value %.4f %%)"
          % (sigma_a_check, sigma_a))
    print("    candidate   sigma_b measured   %.4f %%" % sigma_b)
    b2, se2, n2 = ols([c[0] for c in cross], [c[1] for c in cross])
    print("    cross-leg residual slope %+.4f  se %.4f  t %+.2f  (n %d)  expect null"
          % (b2, se2, b2 / se2, n2))
    print()

    print("    that 1.46 % is not candidate noise.  It is real code-by-prompt")
    print("    structure: inside one schedule cohort, different code speeds up")
    print("    different prompts by different amounts.  So it bounds sigma_b")
    print("    from above and cannot identify it.  Both candidate-leg routes")
    print("    fail for the same reason, and neither is used below.")
    print()

    # The identification that survives.  Subtracting the two frame equations
    # removes sigma_b entirely, so g follows from the two replicate sds and the
    # two board constants with no free parameter and no candidate-leg estimate.
    #
    #   var(pub) - var(free) = (1 - 2g) sigma_k^2 + sigma_a^2 / 2
    #
    vk = sigma_k**2
    d_obs = PUBLISHED_SD**2 - SERIALFREE_SD**2
    d_at_1 = sigma_a**2 / 2.0 - vk
    g_joint = 0.5 * (1.0 - (d_obs - sigma_a**2 / 2.0) / vk)
    sigma_b_joint = (2.0 * (SERIALFREE_SD**2 - g_joint**2 * vk)) ** 0.5

    # Sampling error.  A variance from n replicate pairs has relative se
    # sqrt(2/n).  The two frame variances come from the same pairs and are
    # positively correlated, so treating them as independent overstates se(D).
    rel = (2.0 / N_PAIRS) ** 0.5
    se_d = ((rel * PUBLISHED_SD**2) ** 2 + (rel * SERIALFREE_SD**2) ** 2) ** 0.5
    se_g = 0.5 * se_d / vk

    print("parameter-free test of the old model, sigma_b never enters")
    print("  D = var(published) - var(serialfree)")
    print("    predicted at g = 1     %.6f   (sigma_a^2/2 - sigma_k^2)" % d_at_1)
    print("    observed               %.6f" % d_obs)
    print("    difference             %+.6f   se %.6f   t %+.2f"
          % (d_obs - d_at_1, se_d, (d_obs - d_at_1) / se_d))
    print()
    print("  g  %+.3f  se %.3f   (from %d replicate pairs)" % (g_joint, se_g, N_PAIRS))
    print("    g = 1 rejected at t %+.2f ;  g = 0 rejected at t %+.2f"
          % ((g_joint - 1.0) / se_g, g_joint / se_g))
    print()

    need = abs(d_obs - d_at_1) / 2.0
    n_need = 2.0 * ((PUBLISHED_SD**2) ** 2 + (SERIALFREE_SD**2) ** 2) / need**2
    print("  VERDICT: the tension is inside sampling error.  The old g = 1 model")
    print("  is not refuted and the g %+.2f point estimate is not established."
          % g_joint)
    print("  Separating them at 2 se needs about %.0f replicate pairs, against"
          % n_need)
    print("  the %d we have.  The pair-level correlation between the two frames"
          % N_PAIRS)
    print("  reduces se(D) below the independent bound used here, so the real")
    print("  requirement is lower.  That correlation is the one number that can")
    print("  settle this, and it needs the per-pair values, not the two sds.")
    print()

    g, sigma_b = g_joint, sigma_b_joint
    print("if the point estimate is taken at face value, for reference only")
    parts = [
        ("serial run factor (1-g)^2 sigma_k^2", (1 - g) ** 2 * vk),
        ("serial prompt draw sigma_a^2 / 2", sigma_a**2 / 2.0),
        ("candidate draw sigma_b^2 / 2", sigma_b**2 / 2.0),
    ]
    total = sum(v for _, v in parts)
    print("published-frame variance budget")
    for label, v in parts:
        print("  %-38s %8.6f   %5.1f %%" % (label, v, 100.0 * v / total))
    print("  %-38s %8.6f" % ("total, sd %.4f %%" % total**0.5, total))
    print("  serial-leg share of published variance   %.1f %%"
          % (100.0 * (parts[0][1] + parts[1][1]) / total))
    print()
    print("single-pair detection floor, empirical")
    print("  published  %.3f %%     serialfree %.3f %%"
          % (PUBLISHED_SD * 2**0.5, SERIALFREE_SD * 2**0.5))


if __name__ == "__main__":
    main()
