# E128 — Audit the reach estimator against the board-fitted ranked depth optimum

Student: `qwen-edward`. PR #129. Branch
`qwen-edward/e128-reach-estimator-vs-ranked-depth-optimum`.
Base `senpai/qwen38-mtp-r1` = `526d39739ad76380b56a199a6344d0db02bca765`.
Host `apple-m4-pro-applegpu_g16s-48gib`, M4 Pro, 48 GiB.

**The scored surface did not change.**
`git diff 526d3973 -- Sources/ Vendor/ mtp-head.manifest.json Package.swift
benchmark.json` is empty. Every file this experiment touches is under
`research/`, which is not in `editablePaths`.
`senpai/verify-ranked-score-boundary.sh` PASS.

---

## The answer in one paragraph

The reach estimator **is** biased. Its `expected` under-predicts the realised
accepted count by 9 to 24 percent on eleven of twelve fixtures. **Correcting
that bias does not recover any ranked gain — it loses ranked gain.** The walk
consumes the same biased estimate twice with opposite effect, the reach side is
about twice the size of the expected side, and the net correction pushes rounds
deeper than the ranked optimum. Every arm that corrects the level is negative
on every fixture, and the gain is monotone decreasing in the size of the
correction, so the shipped `gamma = 1` is the optimum.

**Primary metric: `e128_recoverable_ranked_median_pct` = -0.2171 %.** That is
the best implementable arm — a 5 % lift of the estimator level — priced on our
own fitted ranked cost curve under the depth anchor. It is negative, and the
value sits in the narrow band -0.2037 % to -0.2353 % across every cost curve our
data supports. **The estimator-level axis is closed.**

Three independent lines of evidence say the same thing. The R-free replay in
section 4 gives a negative sign on 12 of 12 fixtures. The ranked repricing in
section 6 gives a negative sign on every curve. And the board itself, in section
7, shows the shipped depth holding the record on all five weighted prompts
across 145 depths tried by other teams.

The scheduler axis is **not** closed. A per-round oracle is **+8.52 %** of ranked
median on our curve, and positive at every R in the band. The remaining headroom
is per-round *discrimination*, not level.

## Sign flip, before any headline percentage

The advisor asked to lead with the R value at which the sign flips and where it
sits inside the pinned band.

**The estimator arms do not flip anywhere in the band, because the headline pass
is R-free.** The section 4 result replays the recorded rounds and prices
`sum(round cost) / sum(tokens)` directly. A per-token cost does not need a
ranked round count, so `R` never enters. The section 6 headline uses the depth
anchor, which fits one level parameter per prompt against published depth; the
round count cancels out of that ratio too. `reachonly`, `expectedonly`,
`levelfix`, `jensen` and `jensen_both` carry the same sign at every `R` in
`beagle = 112.5 [105.2, 120.8]` and outside it.

R enters only through the alternative accept-rate anchor, and there almost
everything flips. `marginfull` flips at R(beagle) 104.9 and 106.8, `rankedprice`
at 109.9 and 123.7, `levelfix` four times between 109.9 and 122.1, `reachonly` at
107.1 — all within a few tokens of the pinned R(beagle) = 110. Section 6 shows
why those flips are not usable: that anchor reproduces the published acceptance
rate but misses the published *depth* by up to 2.172 tokens, so it prices
counterfactuals against a shipped baseline that never existed. Only two arms hold
their sign across the whole band under either anchor: `oracle`, positive from
+6.89 % to +17.56 %, and `rankedprice_marginup`, positive from +2.16 % to
+5.17 % under the accept-rate anchor and negative at -3.40 % under the depth
anchor.

---

## Rung 0 — the replayer gate

`research/e128_replay.py` is an exact offline reimplementation of
`costModelDepth`, `makeUniformDepthPrice` and `recordAcceptOutcome`.

| agreement | pooled over 2001 rounds, 15 legs |
|---|---|
| `sched` walk string, byte identical at `%.6f` | **1.000000** |
| selected depth | **1.000000** |
| forward-carried EMA state | **1.000000** (max abs deviation 5.0e-07) |

The forward replay never reads a recorded EMA after the first round. It
reproduces the whole per-round state machine from the shipped prior.

One defect found and modelled: `QwenRuntimeMTPDriver.swift:141-150` narrows the
parent's offer at the tail to `max(1, min(depth, 8, remaining - 1))`. Without it
the last round of every leg replays too deep.

Adding `reach_gain`, `expected_gain` and `reach_gain_by_step` for sections 3
and 4 did not move the gate. Every default multiplies by exactly `1.0`, which is
bit identical in IEEE-754.

## F1 — the identity and the pinned R

Answered in full and accepted by the advisor. Artifact
`research/e128-artifacts/rung0-identity.json`.

1. The identity closes exactly on all 12 legs: `eff = drafted / R`,
   `accept_rate = A / drafted`, `declared_rows = R + drafted`, zero residual.
2. Thorfinn's 312 rounds are 4x my 78-round benchfixture leg.
3. R pinned out of sample. Every assumed R falls inside its band and within
   3 percent: beagle 112.5 `[105.2, 120.8]` against 110, medicine 88.4 against
   90, essays 91.8 against 92, botany 78.6 against 81, republic 93.9 against 93,
   drama 246.7 against 252, travel 212.7 against 212. Plutarch has the analytic
   bound `460 <= R <= 512` with 487 assumed.

## Rung 1 — uncensored per-position acceptance

Twelve forced-depth-7 legs, all exact: `matched=true`, `divergence=0`. The
instrument was reverted in `ed8fe33a` and the worker re-certified at sha256
`28c081976c8f0b9a348d904544b78fc8572c0bfc8c5e9b04e70746ebfdee3bc2` with
`forbid 'DARKBLOOM_E128_FORCE_DEPTH': 0`.

The headline finding: **the shipped leg's apparent rise in acceptance at deep
positions is pure survivor bias.** Uncensored beagle is roughly flat at 0.85,
not rising to 1.000.

| fixture | rounds | depth | accept | p_0 .. p_6 |
|---|---|---|---|---|
| beagle_a | 100 | 6.960 | 0.5920 | 0.840 0.857 0.875 0.984 0.836 0.843 0.884 |
| beagle_b | 117 | 6.957 | 0.4853 | 0.752 0.761 0.894 0.831 0.939 0.957 0.955 |
| benchfixture | 73 | 6.959 | 0.8642 | 0.959 0.957 0.955 0.969 0.984 0.983 0.966 |
| botany_andrews | 125 | 6.952 | 0.4453 | 0.856 0.811 0.663 0.772 0.795 0.857 0.933 |
| drama_dollhouse | 85 | 6.929 | 0.7267 | 0.882 0.905 0.940 0.937 0.966 0.947 0.981 |
| essays_bacon | 259 | 6.950 | 0.1406 | 0.564 0.390 0.456 0.577 0.333 0.600 0.333 |
| essays_montaigne | 131 | 6.947 | 0.4187 | 0.863 0.920 0.786 0.494 0.525 0.619 0.769 |
| medicine_hippoc | 75 | 6.960 | 0.8372 | 0.947 0.958 0.956 0.938 0.967 0.983 1.000 |
| medicine_hist | 134 | 7.000 | 0.4030 | 0.731 0.847 0.771 0.797 0.569 0.966 0.893 |
| plutarch_lives | 163 | 6.975 | 0.3069 | 0.718 0.675 0.684 0.604 0.812 0.846 0.864 |
| republic_jowett | 94 | 6.947 | 0.6401 | 0.904 0.824 0.870 0.933 0.893 1.000 0.940 |
| travel_eothen | 138 | 6.964 | 0.3892 | 0.775 0.776 0.659 0.852 0.826 0.658 0.840 |

Margins are exact multiples of `2^-4 = 0.0625` on every leg. Nil-margin rounds
are zero everywhere. Margin AUC at positions 0 and 1 is 0.78 to 0.91 on beagle,
so the margin is a real per-round covariate, not noise.

## Section 3 — hypothesis J

Hypothesis J: the walk computes `reach` from global position EMAs, the realised
acceptance is `E_round[prod_j p_j(round)] >= prod_j E_round[p_j(round)]`, so the
walk under-predicts, and the gap grows with depth.

I split the measured bias into three terms that sum to it exactly, so the data
picks the owner instead of an argument doing it:

```
measured_bias = mean(expected) - mean(accepted)
              = margin_component     walk p against the round's own EMA
              + ema_component        EMA against the true uncensored chain
              + selection_component  chain against realised, at the round's
                                     own depth -- the term J owns
```

`selection_component = mean_r E[min(C, d_r)] - mean_r min(c_r, d_r)`, with `C`
drawn from the uncensored capability distribution. It is negative exactly when
the scheduler drafts deeper on the rounds that were going to accept more.

Hypothesis J's own prediction is then formed **independently of the realised
count**, by resampling margin-binned uncensored chains:
`S_mix[k] = sum_b w_b prod_{j<k} q_j^b` against
`S_hom[k] = prod_{j<k} sum_b w_b q_j^b`. No fitted parameter.

| fixture | E[expected] | E[accepted] | gamma | margin | ema | select | J predicts |
|---|---|---|---|---|---|---|---|
| beagle_a | 2.868 | 3.303 | 1.152 | -0.517 | 1.494 | -1.411 | -0.418 |
| beagle_b | 2.712 | 3.063 | 1.130 | -0.381 | 1.765 | -1.736 | -0.415 |
| benchfixture | 5.067 | 5.577 | 1.101 | -0.393 | 1.229 | -1.347 | -0.463 |
| botany_andrews | 2.149 | 2.587 | 1.204 | -0.770 | 1.268 | -0.937 | -0.146 |
| drama_dollhouse | 3.933 | 4.565 | 1.161 | -0.614 | 2.023 | -2.041 | -0.575 |
| essays_bacon | 0.836 | 0.796 | **0.953** | -0.165 | 0.340 | **-0.135** | **-0.007** |
| essays_montaigne | 1.906 | 2.507 | 1.316 | -1.356 | 1.355 | -0.601 | -0.028 |
| medicine_hippoc | 4.552 | 5.400 | 1.186 | -0.770 | 1.660 | -1.739 | -0.830 |
| medicine_hist | 2.088 | 2.346 | 1.124 | -0.530 | 1.296 | -1.025 | -0.207 |
| plutarch_lives | 1.532 | 1.844 | 1.204 | -0.628 | 1.041 | -0.725 | -0.047 |
| republic_jowett | 3.376 | 3.923 | 1.162 | -0.492 | 1.751 | -1.805 | -0.634 |
| travel_eothen | 1.786 | 1.977 | 1.107 | -0.487 | 0.916 | -0.620 | -0.079 |

Regressions across the twelve fixtures:

```
measured_bias ~ jensen_predicted_bias    r=+0.732  slope=+0.621  intercept=-0.224
measured_bias ~ selection_component      r=+0.697  slope=+0.273  intercept=-0.103
measured_bias ~ margin_component         r=+0.577  slope=+0.456  intercept=-0.154
measured_bias ~ ema_component            r=-0.778  slope=-0.401  intercept=+0.115
selection_component ~ jensen_predicted   r=+0.906  slope=+1.966  intercept=-0.546
```

### Verdict: hypothesis J is real, and it is not the cause

1. **The resampling works.** J tracks the component it is allowed to own at
   `r = +0.906`. Slope 1.97 means the margin quartiles resolve about half the
   round-level heterogeneity, so J under-predicts its own term by 2x. That is a
   resolution limit of the covariate, not a failure of the mechanism.
2. **J predicts the exception correctly.** `essays_bacon` is the one fixture
   with a positive bias (`gamma = 0.953`) and the one fixture where J predicts
   the smallest term by an order of magnitude, `-0.007` against `-0.03` to
   `-0.83`. That is a genuine out-of-sample hit.
3. **J does not carry the bias.** Slope against the measured bias is 0.62, not
   1. The selection term is 38.6 percent of the split on the median fixture,
   range 18.1 to 45.4.
4. **The real structure is a near-cancellation.** A large *positive* EMA term,
   `+0.34` to `+2.02`, from the shipped censoring plus the saturation pull to
   0.95, very nearly cancels the negative selection term, and the **margin
   override is the net driver**. The residual of the J fit correlates `+0.956`
   with `margin_component` and `-0.913` with gamma.

Residual correlations, sorted:

```
margin_component             +0.956
gamma                        -0.913
q0                           -0.579
q_slope                      +0.381
ema_component                -0.336
mean_depth                   -0.249
spearman_margin_capability   +0.178
eta2_margin                  +0.053
selection_component          +0.049
```

So the estimator's error is not one level bias. It is two large opposing terms
plus an override, which is the structural reason a single scalar cannot fix it —
and section 4 shows that measured, it does not.

## Section 4 — the sign, and the priced effect of each half

The walk consumes the same biased estimate twice, with opposite effect:

| consumer | effect of a low bias | effect of correcting it |
|---|---|---|
| `reach` in `reach > threshold` | breaks earlier, **shallower** | **deeper** |
| `expected` in `threshold = marginal[d] (1 + expected) / cumulative[d]` | threshold too small, **deeper** | **shallower** |

I did not assert which wins. I measured the priced effect of each.

Method: myopic replay of every recorded round on all twelve shipped legs. Each
round keeps its recorded EMA, margin and parent offer; only the walk changes.
The realised accepted count is imputed from the fixture's own uncensored
survival curve — exact when the round rejected, conditioned on survival when it
saturated — so a deeper round pays for its extra work and is credited only with
tokens it would really have won. Priced on the F97 ranked cost curve. **No
simulator is in this causal chain, and no `R`.**

| arm | median depth change | median ranked | fixtures positive |
|---|---|---|---|
| `reachonly` | **+0.395** | **-2.385 %** | 0 of 12 |
| `expectedonly` | **-0.217** | **+0.003 %** | 6 of 12 |
| `levelfix` (both) | **+0.154** | **-0.665 %** | 0 of 12 |
| `jensen` (measured gain, reach side) | +2.692 | -22.617 % | 0 of 12 |
| `jensen_both` | +1.718 | -18.373 % | 0 of 12 |
| `oracle` (per-round ceiling) | -0.736 | **+11.042 %** | 12 of 12 |

Per fixture:

```
fixture              gamma |  d:rch     %rch |  d:exp     %exp | d:both    %both
beagle_a             1.152 | +0.538   -3.322 | -0.134   +0.070 | +0.269   -1.594
beagle_b             1.130 | +0.325   -1.448 | -0.222   +0.455 | +0.127   -0.424
benchfixture         1.101 | +0.231   -0.668 | -0.154   -0.187 | +0.051   -0.247
botany_andrews       1.204 | +0.692   -5.880 | -0.287   +0.758 | +0.287   -2.232
drama_dollhouse      1.161 | +0.304   -1.303 | -0.174   -0.067 | +0.109   -0.016
essays_bacon         0.953 | -0.172   -0.363 | +0.074   -0.707 | -0.098   -0.179
essays_montaigne     1.316 | +1.185  -10.193 | -0.610   +2.840 | +0.438   -2.364
medicine_hippoc      1.186 | +0.300   -1.334 | -0.212   +0.056 | +0.088   -0.343
medicine_hist        1.124 | +0.464   -3.662 | -0.268   +0.178 | +0.163   -1.142
plutarch_lives       1.204 | +0.889   -8.658 | -0.422   -0.303 | +0.372   -2.714
republic_jowett      1.162 | +0.327   -1.167 | -0.250   -0.112 | +0.144   -0.520
travel_eothen        1.107 | +0.494   -3.936 | -0.186   -0.050 | +0.163   -0.810
```

Global gamma sweep, median over fixtures. Monotone, no interior optimum:

```
gamma=1.00 (shipped)  depth 3.769  +0.000 %
gamma=1.05            depth 3.814  -0.282 %
gamma=1.10            depth 3.883  -0.591 %
gamma=1.15            depth 3.956  -1.321 %
gamma=1.20            depth 4.012  -1.705 %
gamma=1.25            depth 4.089  -2.016 %
gamma=1.30            depth 4.146  -2.308 %
gamma=1.35            depth 4.187  -2.607 %
gamma=1.40            depth 4.209  -3.031 %
gamma=1.45            depth 4.245  -3.410 %
```

**The best global gamma is 1.00, which is what ships.** The ranked-optimal gamma
per prompt is 1.00 on eleven of the twelve fixtures. The one exception is
`drama_dollhouse`, which peaks at `gamma = 1.10` with `+0.087 %` — the only
positive value anywhere in the whole 12-fixture by 10-gamma grid, and small
enough that a per-prompt gamma is not worth the complexity even if a per-prompt
gamma were implementable, which it is not: the hidden prompt is unknown at
schedule time.

### The stop rule fires, by a stronger route than the one written

The advisor's rule was to stop if the net sign is "shallower". The net sign is
*deeper*, but I priced the effect instead of inferring it from the sign, and the
priced effect is negative on every fixture and monotone in the size of the
correction. **The recoverable ranked gain from correcting the reach estimator's
level is negative. The estimator-level axis is closed.**

### The scheduler axis is not closed

`oracle` is `+11.04 %`, 12 of 12 positive — 55x the `+0.20 %` threshold. And no
fixed depth beats the shipped adaptive rule:

```
depth=0 -57.138 %   depth=4 -15.087 %
depth=1 -31.205 %   depth=5 -16.893 %
depth=2 -16.065 %   depth=6 -22.214 %
depth=3  -7.480 %   depth=7 -27.079 %
```

The shipped adaptive scheduler is already far ahead of every static depth and is
sitting at or just past the ranked depth optimum in the level direction. The
remaining 11 percent is per-round **discrimination**, which is a different axis
from the one this experiment audits.

The `+11.04 %` here and the `+8.52 %` in section 6 are the same effect measured
in two places, and both are reported rather than one being chosen. This number
is the R-free per-token price over our own 12 fixtures. Section 6 transfers the
same arm onto the eight ranked prompts and recomputes the median, which is the
quantity the score actually uses. **`+8.52 %` is the headline oracle figure**;
`+11.04 %` is the same ceiling before the ranked median dilutes it across
prompts the arm cannot help.

---

## Section 5 — our own ranked cost curve

The advisor's F3 note is correct: the crown moved to `d3c491b5` = `3.49065044`
(Route B), so F97's curve — fitted over 147 official runs of other solvers —
is no longer our curve. Section 5 refits the ranked round cost curve from our
own receipts. `research/e128_ourcurve.py` writes
`research/e128-artifacts/section5-ourcurve.json` and
`research/e128-artifacts/our-ranked-curve.json`, which `e128_price.py`
consumes through the new `--curve-json` / `--curve-key` flags.

### What is fitted, and against what

Each of our four official receipts publishes, per prompt, a candidate seconds
per token and an `effective_mean_draft_len`. The ranked round cost follows from
the identity

```
round_us = candidate_seconds_per_token * 512e6 / R
M̄        = effective_mean_draft_len + 1
```

so eight `(M̄, round_us)` control points fall out of the crown receipt once `R`
is pinned. The fit is a two-tier line in `M` under three monotone constraints —
slope ≥ 0, tier jump ≥ 0, slope increase ≥ 0 — solved by an exact bounded
active-set least squares. No `scipy` is on this host; the solver enumerates
active sets over four parameters, which is exact at this size.

Two corrections matter, and both are applied.

**R is unknown.** Every fit is repeated over the four R vectors pinned in F1.

**The published width is a mean, and the cost curve is convex.** Fitting
through `c(M̄)` reproduces exactly the Jensen bias this experiment spent
section 3 measuring, on the cost side this time. The `dist` fit therefore
replaces `c(M̄)` with `E_hist[c(M)]` over a per-prompt width histogram taken
from the E124 fixture traces and tilted onto that prompt's published mean by a
max-entropy exponential tilt. That removes the bias exactly, because the
piecewise-linear cost is linear in its parameters and the expectation of a
linear form is a linear form. Plutarch's 449 non-drafting rounds are placed at
`M = 1` before tilting.

### Result 1 — R is re-pinned, and the assumed vector wins

The rmse grid is decisive. Reading down each column, the `assumed` R vector
fits 5 to 10 times tighter than any other at every tier step:

```
tier step   predicted     assumed     band_lo     band_hi
    2          1004.1       312.0      1234.6      1801.8
    3           886.5       150.0      1218.7       600.4
    4           884.9       118.6      1218.5       599.6
    5           938.1       127.1      1206.4       702.3
    6           866.8        90.6      1202.1       560.2
    7          1238.2       995.5      1179.7      1821.4
    8          1768.8      1435.1      1282.8      2737.0
```

A one-parameter sweep `R(t) = (1-t)·assumed + t·predicted` has a sharp minimum
at `t = 0`:

```
t=-0.100  rmse 104.90    t=+0.025  rmse  91.37
t=-0.050  rmse 103.43    t=+0.050  rmse  97.08
t=-0.025  rmse  94.80    t=+0.100  rmse 119.89
t= 0.000  rmse  90.57    t=+0.125  rmse 135.08
```

`90.57 µs` is `0.179 %` of a round. Moving one eighth of the way toward the
predicted vector in either direction roughly doubles the residual. The assumed
R vector lies inside my F111 band on all eight prompts, so this is a refinement
of the F1 pinning, not a contradiction of it. **`R` is now pinned by two
independent routes: the F1 per-prompt bands, and a curve-fit residual minimum
inside them.**

### Result 2 — the tier step is at M ≥ 6, not M ≥ 5

The best fit puts the step at `M = 6`, and leave-one-out never moves it:

```
drop plutarch   bp=6 rmse 95.01     drop republic   bp=6 rmse 80.63
drop drama      bp=6 rmse 90.68     drop essays     bp=6 rmse 96.06
drop travel     bp=6 rmse 74.14     drop medicine   bp=6 rmse 93.24
drop beagle     bp=6 rmse 70.26     drop botany     bp=6 rmse 95.15
```

Eight drops, eight times `M = 6`, rmse between 70 and 96 µs. The board curve
put the step at `M = 5`. Our step at `M = 6` matches Route B's own pass-count
prediction, reached here by a completely independent route: the board receipts
never enter this fit.

```
ours   round_us = 27894.3 + 3388.3·M   (M < 6)
                = 21541.1 + 6167.5·M   (M ≥ 6)
board  round_us = 27215.4 + 3966.4·M   (M < 5)
                = 17020.7 + 7154.2·M   (M ≥ 5)
```

Per-point residuals on the chosen fit, all eight inside ±161 µs of a 31 800 to
64 700 µs round:

```
prompt        R    M̄    measured_us    fitted_us   resid   F83 w
plutarch    487  1.154      31806.5      31809.2    -2.7  0.0000
drama       252  3.298      39306.3      39240.2    66.0  0.0000
travel      212  3.656      41051.1      41173.4  -122.3  0.0000
beagle      110  5.382      53011.9      52851.3   160.6  0.4862
republic     93  5.989      57021.7      57146.8  -125.1  0.0100
essays       92  6.087      57951.2      57955.1    -3.9  0.1598
medicine     90  6.256      59015.7      58954.7    61.0  0.2508
botany       81  7.148      64734.9      64768.6   -33.6  0.0124
```

### Result 3 — Jensen bias moves the tier step by one width

Fitting through `c(M̄)` instead of `E_hist[c(M)]` puts the step at `M = 5` and
raises the rmse from 90.6 to 146.6 µs. **The same convexity artefact that
section 3 measured on the acceptance side also moves the answer on the cost
side, by one full width.** The per-prompt bias is bidirectional, because it is
the sign of `E[c(M)] − c(M̄)` under a bimodal histogram against a curve with a
step in it, not a uniform convexity penalty:

```
prompt        M̄       c(M̄)         E[c]     bias_us   bias_%
plutarch   1.154    31804.4      31809.2       4.8     0.015
drama      3.298    39067.5      39240.2     172.7     0.442
travel     3.656    40280.7      41173.4     892.7     2.216
beagle     5.382    46129.3      52851.3    6722.0    14.572
republic   5.989    48187.5      57146.8    8959.3    18.593
essays     6.087    59082.7      57955.1   -1127.6    -1.909
medicine   6.256    60122.5      58954.7   -1167.8    -1.942
botany     7.148    65627.6      64768.6    -859.1    -1.309
```

Beagle and republic carry `+14.6 %` and `+18.6 %` because their means sit just
below the step while a large share of their rounds sit above it. This is the
single largest methodological correction in section 5.

### Result 4 — the shipped flat price is wrong in three different directions

Normalising the marginal cost of the d-th extra draft row by the depth-0 round
gives the price vector the scheduler should be using:

```
depth        0       1       2       3       4       5       6       7
ours    0.1083  0.1083  0.1083  0.1083  0.4383  0.1972  0.1972  0.1972
board   0.1272  0.1272  0.1272  0.3114  0.2294  0.2294  0.2294  0.2294
shipped 0.1800  0.1800  0.1800  0.1800  0.1800  0.1800  0.1800  0.1800
```

Against our curve the shipped flat `0.18`:

- **overprices depths 0 through 3 by 66 %** — those rows cost `0.1083`;
- **underprices depth 4 by 2.4x** — that row crosses the tier step and costs
  `0.4383`;
- **underprices depths 5 through 7 by 9 %** — those rows cost `0.1972`.

As a pipeline validation, running the same fitter against the board's own
control points reproduces F97's published price vector exactly, so the
difference above is a difference in the data, not in the fitter.

---

## Section 6 — every arm repriced on our curve

### The sign does not change

Every implementable arm is negative on our curve, exactly as it was on the
board curve. The E128 conclusion survives the curve swap.

```
arm            board curve   our curve    change
ship                0.0000      0.0000
marginfull         -0.3300     -0.3561     -0.03
expectedonly       -0.7600     -0.6772     +0.08
levelfix           -1.0200     -0.9673     +0.05
recal              -1.3300     -1.2092     +0.12
reachonly          -1.9700     -1.8542     +0.12
nomargin1          -2.1800     -1.9507     +0.23
nomargin0          -2.4400     -2.1716     +0.27
nomargin           -4.0300     -3.7049     +0.33
marginup           -4.5400     -4.2494     +0.29
rankedprice        -5.6300     -2.8508     +2.78
jensen_both        -7.6200     -7.8260     -0.21
jensen            -11.0700    -11.1665     -0.10
static7           -16.5900    -16.5209     +0.07
oracle             +9.0800     +8.5248     -0.56
```

The best implementable arm on our curve is not in that table, because it is a
level-scale arm rather than a named policy: `levelfix1.05`, which multiplies the
reach estimator by 1.05 and leaves everything else alone, reaches **-0.2171 %**.
`marginfull` is second at **-0.3561 %**. Oracle is **+8.5248 %**.

The full level-scale ladder is monotone away from 1.05, which is what a real
optimum looks like rather than a numerical accident:

```
scale   1.00     1.05     1.10     1.15     1.20     1.25     1.30     1.40
gain  -0.9673  -0.2171  -0.4012  -0.9242  -1.1252  -1.3446  -1.7132  -1.7134
```

So the best single knob on the reach estimator is a 5 % level lift, and it is
still 0.22 % *worse* than shipping the estimator untouched.

The only arm that moves materially is `rankedprice`, from -5.63 % to -2.85 %.
That is a check on the fit rather than a new result: `rankedprice` is the arm
that prices draft rows from the ranked cost curve itself, so a curve closer to
our own hardware makes it less wrong. It is still 2.85 % worse than the shipped
flat price.

### Why nothing is recovered, stated plainly

Section 5 shows the shipped flat `0.18` is wrong in three directions at once
against our curve. Section 6 shows that repricing with the correct curve
recovers nothing. Both are true, and the reason is that the three errors cancel
under the realised depth distribution. The price-constant sweep repeated on our
curve still peaks exactly at the shipped value:

```
0.06  -9.99    0.14  -2.24    0.18   0.00    0.23  -4.12    0.30 -15.62
0.09  -5.70    0.16  -1.06    0.20  -0.49    0.26  -8.28    0.36 -31.52
```

**The shipped price is not right for the reason the code implies, but it is
right where it lands.**

### The curve sweep — no curve we can defend reopens the depth-price axis

The tier breakpoint and the R vector are the two structural choices in the cost
curve. If the E128 conclusion depended on either of them, the conclusion would
be a statement about our fit and not about the scheduler. So every arm was
repriced on six curves. `research/e128-artifacts/rung2-curve-sweep.json`.

```
curve           marginfull  levelfix1.05    levelfix   rankedprice     oracle   best implementable
board (bp5)        -0.3253       -0.2353     -1.0247       -5.6317    +9.0770   levelfix1.05 -0.2353
assumed (bp6)      -0.3561       -0.2171     -0.9673       -2.8508    +8.5248   levelfix1.05 -0.2171
ours_b4            -0.3518       -0.2134     -1.2231       -5.2598    +8.7758   levelfix1.05 -0.2134
ours_b5            -0.2680       -0.2343     -0.9699      -10.3930    +8.8071   levelfix1.05 -0.2343
ours_meanfit       -0.2785       -0.2037     -0.9279       -3.8151    +8.9917   levelfix1.05 -0.2037
predicted (bp7)    -0.3891       -0.1971     -0.9474       +1.0435   +11.1488   rankedprice  +1.0435
```

Five of the six curves agree closely. The best implementable arm sits in the
narrow band **-0.2037 % to -0.2353 %** whether the breakpoint is at 4, 5 or 6,
and whether the curve is fitted to the round distribution or to the round mean.
The conclusion is not a property of the breakpoint.

The sixth curve, `predicted`, is the one exception, and it is the one curve our
own fit rejects. Section 5 pinned R by residual RMSE and the `predicted` R
vector fitted **867 µs** against the `assumed` vector's **91 µs** — an order of
magnitude worse at every tier step. Its fitted curve degenerates: the low and
high slopes come out identical at 4204.8 µs per draft row with the breakpoint
pushed to 7, which is a straight line with no tier structure at all. On that
straight line `rankedprice` becomes worth +1.04 %, because a linear price is
exactly what `rankedprice` assumes.

That is a coherent story, not a loophole: **if** the real hardware had no tier
step, **then** pricing draft rows linearly would beat the shipped flat constant
by about 1 %. Our measurements say the tier step exists, at M ≥ 6, and they say
so 10× more strongly than they say anything else about R. So the depth-price
axis is not reopened.

### The R band, and why the two transfer anchors disagree

The board publishes accepted-tokens-per-round and acceptance rate per prompt,
not round counts. The pricing model therefore needs a transfer map from our
12 local legs onto the 8 board prompts, and there are exactly two natural
anchors for it:

- the **depth anchor**, which fits one level parameter per prompt so the
  simulated shipped depth reproduces the published depth. It is R-free: the
  round count cancels out of the ratio.
- the **accept-rate anchor**, which fits the level parameter so the simulated
  acceptance rate reproduces the published acceptance rate. It needs R, because
  acceptance rate is accepted tokens divided by drafted tokens and both depend
  on how many rounds the prompt ran.

The headline pass uses the depth anchor. The R band pass rebuilds the whole
pricing under the accept-rate anchor at eight R scenarios, from
`predicted × 0.90` to `predicted × 1.10` plus the two band edges and the pinned
`assumed` vector. `research/e128-artifacts/rung2-ours-r-band.json`.

```
scenario         R(beagle)  marginfull   levelfix   reachonly  rankedprice_marginup   oracle
predicted_x0.90     101.22     +0.0792    +0.3691     +0.7076              +3.7687   +6.8943
predicted_x0.95     106.84     -0.0033    +0.2747     +0.0474              +4.3393   +9.1839
band_low            105.23     -0.0081    +0.3173     +0.3095              +2.1606   +9.0180
assumed  (pinned)   110.00     +2.2468    -0.0133     -0.6709              +5.1659  +13.7084
predicted           112.47     +1.3070    +0.0119     -0.6753              +3.5094  +13.2730
predicted_x1.05     118.09     +2.4417    -0.5891     -1.5225              +2.9327  +15.6982
band_high           120.77     +1.5100    -0.5649     -1.4765              +2.5028  +15.9824
predicted_x1.10     123.71     +2.3512    +0.6457     -0.6949              +2.5932  +17.5584
```

Two arms keep their sign across the whole band. `oracle` is positive
everywhere, from +6.89 % to +17.56 %. `rankedprice_marginup` is positive
everywhere, from +2.16 % to +5.17 %. Everything else flips sign inside the
band, most of them within a few tokens of the pinned R(beagle) = 110:
`marginfull` flips at 104.9 and again at 106.8, `rankedprice` at 109.9 and
123.7, `levelfix` four times (109.9, 111.3, 112.6, 122.1), `reachonly` at 107.1,
`static7` at 104.9, `nomargin` at 103.9.

**The accept-rate anchor's positive arms are not credible, and the reason is
measurable.** The two anchors do not just weight the prompts differently; they
disagree about what the shipped scheduler was doing. Compare each anchor's
simulated shipped state with the published F92 state:

```
             published    depth anchor          accept-rate anchor
prompt      depth accept  depth   err   accept   depth    err   accept
beagle      4.382 0.8340  4.383 +0.001  0.7579   5.135 +0.753  0.8346
medicine    5.256 0.8920  5.269 +0.013  0.8813   5.369 +0.113  0.8909
essays      5.087 0.8970  5.054 -0.033  0.8751   5.363 +0.276  0.9015
botany      6.148 0.8650  6.168 +0.020  0.9932   4.711 -1.437  0.8655
republic    4.989 0.9030  5.000 +0.011  0.8220   5.310 +0.321  0.9043
drama       2.298 0.4490  2.492 +0.194  0.6397   0.126 -2.172  0.4030
travel      2.656 0.5330  2.704 +0.048  0.5713   2.154 -0.502  0.5337
plutarch    0.154 0.3330  0.156 +0.002  0.3020   0.186 +0.032  0.3342
```

The depth anchor reproduces published depth to within **0.194** on every prompt
and misses acceptance rate by up to 0.191. The accept-rate anchor reproduces
acceptance rate to within **0.005** on seven prompts, and misses published depth
by up to **2.172**. On `drama` it simulates a shipped scheduler that drafts
0.126 tokens per round against a published 2.298 — effectively no drafting at
all — and it still misses drama's acceptance target by 0.046. On `botany` it is
1.44 low, on `beagle` 0.75 high.

A baseline simulated at the wrong depth prices every counterfactual against the
wrong reference. `rankedprice_marginup` looks like +5 % under that anchor
because the anchor's `drama` and `botany` baselines are drafting at depths the
shipped scheduler never used, which leaves a large amount of fictitious
headroom for a deeper policy to claim. **The depth anchor is the defensible
transfer map for this question**, because depth is the quantity the scheduler
directly controls and the quantity the cost curve is priced in. Under it, every
implementable arm is negative.

This is worth stating without hedging: I ran the sensitivity analysis the
assignment asked for, and it produced positive numbers on one anchor. Those
numbers are an artifact, and the artifact is identified rather than assumed.
The R band is reported in full anyway, so the advisor can check that reasoning
against the same artifact.

### Where the transfer map is weakest, reported honestly

The 12-fixture validation gate passes on means — mean depth error +0.136
against a 0.25 tolerance, mean acceptance error -0.0155 against 0.05 — but two
individual fixtures exceed those tolerances:

```
max |depth error|    0.5060   beagle_a          (tolerance 0.25)
max |accept error|   0.0570   drama_dollhouse   (tolerance 0.05)
```

The gate is written on means and so it returns `passed: true`. That is the gate
I wrote before seeing the numbers, and I am not going to reinterpret it after
the fact, but the per-fixture failures are real and they are the honest upper
bound on how precisely this model reproduces a single leg. They do not change
the sign of any headline arm — the smallest gap between the shipped arm and the
best implementable arm is 0.20 %, and `beagle_a` is one of 12 fixtures feeding
five weighted prompts — but a follow-up that wants to price a sub-0.2 % effect
would need a tighter transfer map first.

### Sensitivity, including the crown that actually moved

Eleven variants. Seven are negative and four are positive, and the four
positives are the interesting part of this section rather than something to
explain away.

```
variant                     receipt  marginfull  lvlfix1.05  rankedprice     oracle   best implementable
headline                   44559d02     -0.3561     -0.2171      -2.8508    +8.5248   levelfix1.05        -0.2171
constant_p                 44559d02     -0.0253     +0.2433      +1.8662   +12.2410   rankedprice_nomargin +1.9735
shuffle_margins            44559d02     -3.2060     +0.2702      +1.5126   +11.5990   rankedprice          +1.5126
hold_zero_weight_at_shipped 44559d02    -0.3561     -0.2171      -2.8508    +8.5248   levelfix1.05        -0.2171
seed_777                   44559d02     -0.2851     -0.1169      -2.6538    +8.6394   levelfix1.05        -0.1169
single_fixture_a           44559d02     -0.6477     -0.0942      -2.4866    +8.5540   levelfix1.05        -0.0942
single_fixture_b           44559d02     +0.1247     -0.2089      -3.4480    +8.7489   marginfull           +0.1247
benchfixture_only          44559d02     -1.9715     -0.1554      +9.6995   +25.9863   rankedprice_recal    +9.7164
receipt_crown              d3c491b5     -0.3784     -0.2427      -2.8711    +8.4300   levelfix1.05        -0.2427
receipt_prev_crown         bc070b7b     -0.3965     -0.3990      -3.0292    +8.4976   marginfull          -0.3965
receipt_ec778a91           ec778a91     -0.3548     -0.3331      -2.9621    +8.5680   levelfix1.05        -0.3331
```

**`receipt_crown` now re-anchors the whole pass on `d3c491b5` = `3.49065044`,
the receipt that actually holds the crown.** My earlier report of this row was
wrong: the variant had been pointed at `bc070b7b`, which is the *previous*
crown, and I described it as the current one. Both are in the table now. The
answer is the same on either: on the true crown the best implementable arm is
`levelfix1.05` at **-0.2427 %**, slightly *more* negative than the headline
-0.2171 %, and every implementable arm stays negative. **The moved frontier does
not reopen the axis.**

Three receipts spanning `3.3435`, `3.3592` and `3.4907` give best implementable
arms of -0.2171 %, -0.3990 % and -0.2427 %. The conclusion is stable across a
4.4 % spread in the anchoring score.

#### The four positive variants, and why none of them is a result

**`constant_p` (+1.97 %) and `shuffle_margins` (+1.51 %) are deliberate controls
that destroy the margin signal.** The first replaces the per-round acceptance
probability with a constant; the second permutes the margins across rounds. In
both, the arm that turns positive is a `rankedprice` variant — pricing draft
rows against the true cost curve instead of the shipped flat `0.18`.

That is a clean and slightly surprising causal statement, and it is worth
stating in the positive direction: **ranked-cost pricing of draft rows is worth
about +1.5 to +2.0 % of ranked median, but only in a world where the margin
override is not already capturing that value.** The shipped scheduler gets there
first by a different route. This is the mechanism behind section 3's finding
that a large positive EMA term nearly cancels a negative selection term — the
override is doing the work that correct pricing would otherwise do, and the two
are not additive.

**`single_fixture_b` (+0.1247 %) and `benchfixture_only` (+9.7164 %) are
single-fixture medians, not ranked estimates.** `single_fixture_b` replaces each
pooled two-fixture prompt with one fixture and lands at a quarter of the
+0.20 % threshold the assignment set. `benchfixture_only` replaces *all eight*
prompts with `benchfixture`, whose uncensored acceptance is 0.959 to 0.984 — a
completely different acceptance regime from any ranked prompt, three of which
sit below 0.55. Its oracle of +25.99 % is the tell: that is not the ranked
prompt pool, it is a stress test of the machinery at near-perfect acceptance.

**`hold_zero_weight_at_shipped` is bit-identical to the headline.** This is the
variant I previously mislabelled `drop_zero_weight`. It holds `drama`, `travel`
and `plutarch` at the shipped ratio instead of repricing them; it does not
remove them from the eight-prompt median. Bit-identical output is a genuine
invariance — the median is set entirely by the five weighted prompts — but it is
a weaker claim than the name implied, and section 7 verifies the same fact
directly against the crown receipt.

## Section 7 — the strongest independent check, from the board itself

Everything above is our model. The board is not. If the shipped reach estimator
sits at the depth optimum, then across the 793 public board rows per prompt the
fastest rows should sit at the shipped depth, and rows at other depths should be
slower. That is a prediction our model did not make and cannot influence.

`research/e128_board_depth_scan.py` groups every public per-prompt row by its
`effective_mean_draft_len` and reports the best raw ratio reached at each depth.
`research/e128-artifacts/board-depth-scan.json`.

```
prompt     F83 wt  depths  shipped   best@shipped  best elsewhere    lead
beagle     0.4862     145    4.382         3.3263          3.1838   +4.48%
medicine   0.2508     136    5.256         3.6550          3.5415   +3.20%
essays     0.1598     122    5.087         3.6556          3.6118   +1.21%
botany     0.0124     143    6.148         3.7144          3.6218   +2.56%
republic   0.0100     123    4.989         3.6730          3.4960   +5.06%
drama      0.0000     124    2.298         1.9596          2.0030   -2.17%
travel     0.0000     127    2.656         2.2306          2.2143   +0.74%
plutarch   0.0000      98    0.154         1.2718          2.2980  -44.66%
```

**On all five prompts that carry weight in the ranked median, the shipped depth
holds the board record, and it holds it by 1.2 % to 5.1 %.** Together those five
carry 0.9192 of the F83 weight. On `beagle` the leading 71 rows of the entire
793-row ranking sit at draft length 4.382, and on `republic` the leading 63 do.
The board has searched 145 distinct beagle depths across many independent teams
and none of them beat the shipped one.

Two caveats, both of which matter.

**This is observational, not a controlled depth sweep.** Rows at other depths
come from other solvers with other code, so depth co-varies with everything else
in the submission. It is adversarial many-team evidence that no one found a
better depth, not proof that no better depth exists.

**Two zero-weight prompts go the other way, and one goes badly.** On `drama` two
single rows beat the shipped depth by 2.2 %; both come from rejected submissions
scoring 3.2409 and 3.2250 against the crown's 3.4907, and both are much slower
on `beagle`. On `plutarch` the shipped depth of 0.154 — essentially no drafting
at all — is beaten by 44.7 %, by rows that draft at 2.36. That is a real and
large miss, and it is worth saying plainly rather than hiding behind the weight.

The reason it does not change the E128 answer is arithmetic, and it is checkable
directly against the crown receipt. `d3c491b5` sorts to:

```
plutarch  1.2607   drama   1.9596   travel   2.2306   beagle   3.3263
medicine  3.6550   essays  3.6556   republic 3.6730   botany   3.7144
median = (3.3263 + 3.6550) / 2 = 3.49065044 = the official score
```

The median is set by `beagle` and `medicine`. Lifting `plutarch` from 1.2607 to
the board-best 2.2980 moves it from first to second in that sorted list and
leaves the fourth and fifth values untouched: the recomputed median is
`3.4906504356`, bit-identical to the official score. That is exactly why F83
assigns `plutarch` weight zero, and it is why a real 44 % per-prompt win there
is worth nothing to the score under the current distribution.

### Why the oracle is still +8.52 %, and why that is not a contradiction

The oracle arm knows, for each round, whether each draft token will be accepted,
and drafts exactly the accepted prefix. Its gain is **per-round discrimination**,
not a depth level. It says: *the shipped mean depth is right, and there is 8.5 %
available to whoever can tell the good rounds from the bad ones.*

Those two statements are consistent, and the arms separate them cleanly. Every
arm that only moves the level — `levelfix`, the whole `levelfix1.05..1.40`
ladder, `static7`, the price-constant sweep — is negative or zero. The only
positive arm is the one that cannot be implemented, because it needs the
acceptance outcome before the draft is proposed.

So the honest statement of the E128 result is narrower and sharper than "the
scheduler is fine":

- **Closed:** you cannot recover ranked median by correcting the reach
  estimator's level, by repricing draft rows against the true cost curve, or by
  choosing a better fixed depth. Best implementable arm is -0.20 %, and the
  board's own 793-row depth scan agrees.
- **Open:** per-round discrimination is worth up to +8.52 % of ranked median,
  and no arm we can currently implement captures any of it. That is where the
  next scheduler experiment should go, and it is a question about the *margin
  signal*, not about the reach estimator.

### What the next experiment would have to beat

Any follow-up on this axis has to clear a specific bar, which this experiment
now sets:

1. it must be evaluated under the depth anchor, not the accept-rate anchor,
   because the accept-rate anchor mis-simulates the shipped baseline by up to
   2.17 tokens of depth;
2. it must beat `levelfix1.05` at -0.2171 %, which is the best implementable
   arm on every well-fitted curve;
3. it must not be a level change, because the board has already searched 145
   depths on `beagle` and the shipped depth wins by 4.5 %;
4. it must improve per-round ranking, since that is the only place the +8.52 %
   lives.

The cheapest such experiment is a better margin feature, not a better estimator.
Section 4 already showed the per-position margin AUC ranges from 0.78 to 0.99 on
`beagle_a` and 0.47 to 0.96 on `benchfixture`, so the margin signal is strongly
stratified by position and the shipped code pools it. That is a concrete,
falsifiable next step, and it is listed under suggested follow-ups rather than
implemented here.

---

## Suggested follow-ups, not implemented here

1. **Per-position margin scale.** The strongest lead in this experiment.
   Per-position margin AUC ranges 0.78 to 0.99 on `beagle_a` and 0.47 to 0.96 on
   `benchfixture`, against a pooled E122 figure of 0.5109 — pooling destroys a
   strongly stratified signal. The shipped code uses two constants, `2.0` at
   position 0 and `3.0` elsewhere. The offline logistic fit in
   `research/e128_price.py` already produces a per-position scale and a
   log-likelihood against those shipped constants, so the fit cost is already
   paid. This is the only lead that attacks the +8.52 % oracle directly.

2. **Ranked-cost pricing combined with a better margin signal.** The sensitivity
   pass shows `rankedprice` is worth +1.5 to +2.0 % of ranked median *when the
   margin override is disabled*, and -2.85 % when it is not. The two mechanisms
   are strongly non-additive. The open question is whether a sharper margin
   signal changes that interaction, or whether the override will always claim
   the value first. Worth one bounded pass before assuming either.

3. **A tighter transfer map.** The 12-fixture validation gate passes on means
   but two fixtures exceed the per-fixture tolerances: `beagle_a` at 0.506 depth
   error and `drama_dollhouse` at 0.057 accept error. Any future experiment
   pricing a sub-0.2 % effect needs this fixed first. The cheapest route is more
   `beagle`-like fixtures, since `beagle` carries 0.4862 of the F83 weight and
   is the worst-fitted prompt.

4. **The `plutarch` miss.** The shipped scheduler drafts 0.154 tokens per round
   on `plutarch` while other teams reach 2.36 and are 44.7 % faster there.
   It is worth nothing today because `plutarch` is the lowest of eight values
   and F83 gives it weight zero. It becomes worth something only if the ranked
   prompt pool changes or if the four low prompts rise enough to enter the
   median. Recording it so nobody re-derives it from scratch.

5. **The parent tail-offer defect.** `QwenRuntimeMTPDriver.swift:141-150`
   narrows the parent tail offer to `max(1, min(depth, 8, remaining - 1))`,
   which censors the final rounds of every leg. This experiment worked around it
   with forced-depth legs. It is a real correctness-adjacent wart in the driver
   and is worth a separate look, though it is outside the E128 scope and I did
   not touch it.

---

## Advisor follow-ups F4, F5 and F6

Six zero-GPU items were assigned after the headline result. All are answered on
the assignment PR and every number is reproducible from the artifacts listed at
the end of this section. None of them changes the E128 primary metric.

### F4 3a — the ranked per-prompt width histograms

`research/e128_width_histograms.py` -> `f4-width-histograms.json`.

```
prompt      F83 wt   mean M     M=1     M=2     M=3     M=4     M=5     M=6     M=7     M=8
beagle      0.4862    5.382  0.0000  0.0478  0.2740  0.1105  0.1127  0.0638  0.0539  0.3373
medicine    0.2508    6.256  0.0000  0.0109  0.1600  0.1062  0.0468  0.0859  0.1418  0.4484
essays      0.1598    6.087  0.0000  0.0008  0.0124  0.0439  0.1186  0.4905  0.3338  0.0000
botany      0.0124    7.148  0.0000  0.0004  0.0415  0.0630  0.0632  0.0372  0.1262  0.6685
republic    0.0100    5.989  0.0000  0.0070  0.2445  0.1079  0.0450  0.0685  0.0425  0.4846
drama       0.0000    3.298  0.0000  0.1406  0.5002  0.3088  0.0380  0.0031  0.0020  0.0072
travel      0.0000    3.656  0.0000  0.0435  0.5148  0.2965  0.0682  0.0489  0.0206  0.0074
plutarch    0.0000    1.154  0.9220  0.0182  0.0458  0.0123  0.0012  0.0003  0.0001  0.0000
F83-WTD     0.9192    5.773  0.0000  0.0285  0.1940  0.0971  0.0944  0.1437  0.1274  0.3150
```

The means are exact and pinned to the `d3c491b5` receipt. The shapes are
inferred from the local proxy fixtures and are weakest for `essays`, whose two
sub-prompts disagree strongly (total variation from pooled 0.0621 and 0.8814).
The shapes are invariant to the choice of R: total variation is 0.0000 on the
seven drafting prompts and at most 0.0759 on plutarch.

### F4 3c — the tier-2 slope anomaly

`research/e128_slopes.py` -> `f4-slopes.json`, `f4-candidate-curves.json`.

The tier is a **per-row** cost, not a fixed per-pass cost, and the break really
is at M>=6. Every model that fits well wants a negative pass constant, which is
not physical. Ranked by AICc among physically admissible models:

```
model@break                     params    rmse     aicc         c        a        k
passcount_slopeonly@M>=6             3     285   102.45     27725        0     1877
passcount_slopeonly@M>=5             3     495   111.27     27644        0     1872
passcount_slopeonly@M>=4             3     673   116.19     29416        0     3494
passcount_affine@M>=6                3     789   118.73     26521    10941        0
```

The admissible slope ratio is **1.54**, bracket [1.54, 1.82]. The earlier
3388.3 / 6167.5 / 1.82 reading came from an inadmissible fit and should not be
quoted. A ratio of 1.00 is rejected. Smooth curvature is ruled out: a quadratic
with no tier has rmse 672 against 91 for the two-segment form. Leave-one-out at
a fixed break gives `dslope = 2779 +/- 161`, t = 17.30. Under a joint
family-and-break leave-one-out the family is 8/8 robust and the break is not.

### F4 3b — ranked prices for the three live arms

`research/e128_arm_prices.py` -> `f4-arm-prices.json`. Median deltas by Rule 67.

| arm | price | robustness |
| --- | --- | --- |
| one-pass `{6,7,8}` | +8.2 to +12.3 % at c=0.445 | needs the tier at M=6,7 to be real |
| per-width templating | +0.90 to +4.15 % over c | curve-invariant and reading-invariant |
| `prune_na5_pair` | +0.06 to +0.86 % | curve-invariant, share unknown |
| one-pass `{6,7}` | +2.20 or −2.87 % | sign depends on the residency scope |
| one-pass `{6}` | +1.14 or −4.10 % | sign depends on the residency scope |

Two ambiguities decide signs rather than magnitudes and are flagged rather than
guessed: the definition of `c`, and whether arm 2's residency loss is charged
only at the widths the new table changes or across the whole round.

### F4 2 — the board curve against ours

The board curve breaks entering M=5; ours breaks entering M=6. The shipped
dispatch table at `Qwen35.swift:1565` reads
`[(3,3), (4,4), (5,5), (6,3), (7,4), (8,4), (9,3)]`, so the pass count changes
between M=5 and M=6, which is our break. The board curve is fitted across
solvers that ship different tables, and a mixture fit will place a single break
between the members and inflate the upper slope; its slope ratio of 1.804 is
consistent with that, against our 1.545.

The disputed width carries 9.44 % of F83-weighted rounds, and the cost of
getting it wrong is concentrated entirely in arms that re-derive the depth price
from the curve: `rankedprice` moves from −3.27 % on our break to −11.02 % on the
board break. Every other arm moves by under one point, so the E128 headline is
unaffected.

### F4 4 — the discrimination-signal ranking

`research/e128_signals.py` -> `f4-signals.json`. Per-position AUC against the
uncensored forced-depth outcomes, per fixture, never pooled.

```
                          margin    ema_j  reach_j   streak prev_acc round_idx
BAND p>=0.93              0.8763   0.6439   0.7589   0.8269   0.8013   0.6624
LOWER proxy               0.8520   0.7960   0.8185   0.8064   0.8034   0.8679
ALL fixtures              0.7627   0.6984   0.7295   0.6932   0.7197   0.7479
```

Three results. The margin's `benchfixture` AUC of 0.7958 reproduces E122's
0.7998 from a different instrument, confirming that E122's pooled 0.5109 was a
stratification artifact. The margin is the **best** free signal in the band, so
the shipped override is driven by the strongest cheap signal available, not the
weakest. And `round_idx` scores 0.8679 on prose proxies but 0.6624 in the band,
with `benchfixture` below chance at 0.3812, which is F98's late-window
degeneration appearing as a signal that does not exist where the ranked mass is.

This reconciles with the headline rather than contradicting it. AUC measures
ranking; the E128 loss is in level. `levelfix1.05`, a pure level scale, is the
best implementable arm at −0.2167 %, while `recal`, which refits the sigmoid
scale constants, is −1.2092 % and deleting the override outright is −3.7049 %.
The signal ranks well and the clamp sits too low, and only the level fix helps.

Of the five candidate signals, the shortlist-score spread is the one to build:
the 32 exact scores already sit in `threadgroup float exact_scores[TOPK]` at
`Qwen35.swift:3716`, so one `simd_max`, one `simd_sum` and a one-float write
make it measurable. Per-layer residual magnitudes and the hidden-state norm are
rejected on cost, because both need a host-visible reduction inside the target
forward pass and neither can distinguish a hard token the head got right from a
hard token the head got wrong.

The band stratum rests on 32 rejects across two fixtures. The ordering is
provisional.

### F5 — a two-channel estimator for the board index

`research/e128_two_channel.py` -> `f5-two-channel.json`. 190 board rows scoring
at least 3.25, decomposed into a thermal or dispatch **mode** direction and a
**mechanism** direction centred on our crown.

```
one mode flip  = -1.0438 index units      one e-fold of score = -9.5466 units
cos(u_mode, u_mech) = -0.4615             cond(basis) = 6.64
variance shares: mode 89.7 %, mech 10.2 %, resid 0.1 %
held-out mean |profile - model|: 7.9434 (mu) -> 4.4027 (+mode) -> 0.5770 (+mech)
```

The rival `cf79f7df` leads us by 0.744 %, and 96.8 % of that lead is a **mode**
move, not a mechanism move. Our own crown drifted +0.6436 towards the slow mode
while mechanism paid it back, and the F76 single-index classifier reported
nothing. Registered prediction for alphonse's next receipt: if built on
`d3c491b5`, index −13.3603 +/- 0.1419; if on `f17daf34`/`cf79f7df`, −14.2081
+/- 0.1419.

### F6 — the NA census against the raw ranked round costs

`research/e128_census.py` -> `f6-census.json`.

The advisor's census uses `ceil(M/4)`, which puts the pass boundary at M>=5.
The dispatch table puts it at M>=6. With the wrong boundary the pooled fit
collapses to rmse 2930 with both terms pinned at their bounds.

Fitting `T(M) = c + f(G-1) + bM + k(G-1)M` with `f, k >= 0` on the eight raw
receipt-derived round costs, in width-histogram expectation:

```
best      f =      0.0 us   k = 1875.1 us/row   b = 3449.0   rmse  285.3
at census f =  10187.2 us   k =  429.9 us/row   b = 3540.6   rmse  710.0
F(1,4) = 20.781      5 % crit 7.709      1 % crit 21.20
f not rejected at 5 %: [0, 4881] us; the census 10187 us is outside
```

The data reject the census magnitude at 5 % and just fail to reject it at 1 %.
With four residual degrees of freedom that is a weak rejection. The free-`f`
optimum reproduces `slopeonly_b6` exactly, which is the check that the two fits
are the same object.

The excess profile splits by model family rather than by author:
`passcount_affine@M>=6`, one of our own admissible curves, reproduces the
census's shrinking profile at +21.80/+20.21/+18.84 against the census
+24.29/+21.57/+19.40, while the per-row-tier families grow.

Repricing the census arm on the real histograms instead of a two-point lattice
gives 9.37 % against 12.73 %, or +8.7036 % on the exact Rule 67 median. Ranked
by absolute microseconds saved times round share, the build order is `[8, 6, 7]`
under the census and `[8, 7, 6]` under our curve, so **M=8 is first under both**
and the disagreement only swaps the near-tied second and third places.

M=9 is unreachable, not merely unobserved: `segmentedVerifyDepthCap = 7` at
`Qwen36MTPBlockSession.swift:1008` caps `M = d + 1` at 8, so `case 9` in the
dispatch table is dead code and the census's M=9 row prices work that cannot
execute.

### Base movement

The advisor branch moved from `526d3973` to `221065c5` during the follow-ups.
This branch is rebased onto `221065c5`. The move changed `Qwen35.swift`,
`quantized.cpp` and `quantized.h`, and did **not** change
`Qwen36MTPBlockSession.swift` or the QMV dispatch table. The scheduler the
replayer models and the pass boundary the curves rest on are both unchanged, so
every conclusion above holds on the current base without replay.

---

## Advisor follow-up F7 — the board stratification, and the head-share retraction

### F7 item 3 — stratify the board by QMV dispatch table

**Answer. The ranked level price of one extra QMV dispatch pass is
`f = 50.4 ± 253.0 µs`, 95 % CI `[−445.5, +546.2]`.** It is statistically zero.
This comes from 200 board submissions with 996 high-width prompt-rows,
submission and prompt fixed effects, and errors clustered on the submission.
Model R wins the likelihood comparison by `ΔAICc = 756.5` and `ΔBIC = 751.8`.

#### The stratum table

`research/e128_strata.py` walks the 456 local submission trees with
`git cat-file` and reads the wide-output (`out_vec_size >= 4096`) branch of
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`,
where the table appears as `qmv_fast_crossrow_affine4_g64_m<T, M, IPG>`.
F127 pointed at `research/board_width1_qmv_variants.py`, which scans
`quantized.h` and `quantized.cpp` for the width-1 variants; the per-M grouped
table is a different construct and needed its own extractor. **No board
submission carries the `let cases` E120 form** that our own base uses at
`Qwen35.swift:1565`; that form exists only in this checkout.

```
extraction over 456 local trees
   202  wide m-table found
   220  crossrow present, no per-M table
    34  no crossrow at all
```

The 202 tables collapse to 15 strata by `G(M) = ceil(M / IPG(M))`:

```
G(1..9)                 subs   pts  score p50       first
1,1,1,1,2,2,2,2,3        121   968     2.8831  2026-08-15
1,1,1,1,2,2,2,3,3         39   312     2.9114  2026-08-16
1,1,1,1,2,1,2,1,1         12    96     3.0545  2026-08-17
1,1,1,1,2,1,1,1,1          9    72     3.0950  2026-08-17
1,1,1,1,2,1,2,3,1          9    72     3.0253  2026-08-17
1,1,1,1,2,3,2,3,3          2    16     2.8905  2026-08-17
1,1,1,2,2,1,1,1,1          2    16     3.0925  2026-08-17
1,1,1,1,1,1,2,1,1          1     8     3.1040  2026-08-17
1,1,1,1,2,1,1,3,1          1     8     3.0340  2026-08-17
1,1,1,1,2,2,2,4,3          1     8     2.9339  2026-08-17
1,1,1,1,1,1,1,1,1          1     8     3.1391  2026-08-17
1,1,1,1,1,1,2,3,1          1     8     3.0955  2026-08-17
1,1,1,1,2,2,1,2,3          1     8     2.8998  2026-08-16
1,1,1,1,2,3,2,2,3          1     8     2.9214  2026-08-17
1,1,1,2,2,1,2,1,1          1     8     3.1103  2026-08-17
```

Two structural facts fall out immediately.

1. **The M = 5 boundary is not stratified.** 199 of 202 tables use `IPG(5) = 3`,
   so `G(5) = 2` almost everywhere. Only three singletons match our base's
   `IPG(5) = 5`. F130.1's observation that the board breaks at M = 5 and we
   break at M = 6 is therefore a comparison of `n = 199` against `n = 1`, and
   the board cannot test that boundary on its own.
2. **The M >= 6 part is heavily stratified.** `G(6)` is 2 in 162 submissions,
   1 in 37, and 3 in 3. `G(8)` runs over 1, 2, 3 and 4. That is where the
   pass price is identified.

#### Two data problems, solved and reported

**The round count.** `effective_mean_draft_len` is `D / R` with integer `D` and
`R`, so `R` is recoverable only up to an integer multiple of the printed
decimal's minimal denominator. `research/e128_rounds_check.py` validates the
rational reconstruction against the `ROUNDS` vector `rankedcurve.py` hard-codes:
it reproduces 7 of 8 prompts exactly over all 164 reference-schedule rows, and
misses only drama, where the minimal denominator is 84 and `rankedcurve.py`
picks `3 x 84 = 252` over the minimal legal `2 x 84 = 168`. Pruning the
remaining multiples on the physical constraint `alpha = (512/R − 1)/dl` in
`[0.15, 1]` still leaves a factor-2 span for 5 of 8 prompts, so the multiple
must be chosen, not assumed.

Curve consistency chooses it, and chooses it very strongly. On the
reference-schedule rows the wrong drama multiple raises the pooled hinge RSS
from `1.87e8` to `3.86e10`, a factor of 206, and flips beagle from `+14.58 %`
above the low-prompt line to `−13.03 %` below it. The multiple is therefore
identified about two orders of magnitude more sharply than the P-versus-R
difference the fit has to resolve. `research/e128_strata_curve.py` picks the
multiples by coordinate descent on a free-break hinge fit, which is neutral
between the two models because both are hinge-shaped inside one stratum.

**Coverage.** None of the 164 reference-schedule rows has a local tree, and
none of the 202 table-bearing rows runs the reference schedule. The set with a
validated `R` and the set with a known `G` are disjoint, which is why the
recovery above was necessary rather than optional.

#### Where the break actually sits

`rankedcurve.py` asserts the break by splitting the prompts into
`{plutarch, drama, travel}` and `{beagle, republic, essays, medicine, botany}`.
`research/e128_break_scan.py` scans it freely on the 164 reference rows:

```
pooled-RSS best break M* = 3.4375     RSS at M* = 5.0 is 19.8x worse
beagle against the line through republic/essays/medicine/botany
    median −0.58 %   p10 −0.70 %   p90 +0.10 %
beagle against the line through plutarch/drama/travel
    median +14.58 %  p10 +12.38 %  p90 +15.01 %
```

Beagle sits on the wide-prompt line to within 0.6 %, so there is **no second
break between M = 5.4 and M = 7.1** on the board. The sampled widths leave a
gap between travel at 3.656 and beagle at 5.382, and a hinge anywhere inside
that gap gives an identical fit, so `M* = 3.4375` should be read as "the break
is somewhere in `[3.5, 5.38)`", which is compatible with `G(5) = 2`.

#### Per-stratum fits

Free-break hinge, per-submission intercepts, over the full width range:

```
G(1..9)                 subs     M*      b_lo      b_hi      rmse   G steps at
1,1,1,1,2,2,2,2,3        121  3.500    5034.7    9886.6    1946.4       [5, 9]
1,1,1,1,2,2,2,3,3         39  3.375    4800.9    9565.9     419.8       [5, 8]
1,1,1,1,2,1,2,1,1         12  1.500   -8694.6    7732.8    1801.3       [5, 7]
1,1,1,1,2,1,1,1,1          9  3.500    4769.8    8338.5     568.3          [5]
1,1,1,1,2,1,2,3,1          9  3.375    4692.6    8700.6     407.1    [5, 7, 8]
```

Four of five fittable strata put the break at 3.375 to 3.500 whether their own
`G` steps at 5 only, at 5 and 8, at 5 and 9, or at 5, 7 and 8. **The break does
not move with the table above M = 5.** That is Model P's forbidden outcome.

#### The decisive arm: the high-width segment

Every stratum has `G(1..4) = 1,1,1,1`, so the strata are indistinguishable
below M = 5 and the whole test lives above it. Restricting to prompt-rows with
mean width >= 4.5 gives 996 points over 200 submissions
(`research/e128_strata_highwidth.py`):

```
model                     b us/row        f us/pass       rmse    aicc     bic
line                 9530.7 +-  57.3              -      723.0  13621.6 14508.7
Model P              9530.1 +-  57.4    50.3 +- 236.3    723.0  13624.7 14515.5
Model P, deviation   9531.3 +-  57.3  -268.7 +- 238.0    722.4  13623.1 14514.0
two-way FE, clustered
                     9760.9 +-  87.6    50.4 +- 253.0    536.0        -       -
```

`Model P, deviation` prices `Gbar_s(M) − Gbar_S1(M)`, the pass deviation from
the majority table, which removes any curvature common to every stratum. The
two-way specification adds prompt fixed effects, so `f` is then identified only
by the same prompt sitting in strata with different pass counts and any
per-prompt round-count bias drops out. Adding the pass term does not lower the
RMSE in any specification and AICc and BIC both get worse.

Per-stratum slopes of the high segment, own intercepts and own slope:

```
G(1..9)                 subs         b      se      rmse
1,1,1,1,2,2,2,2,3        120    9992.5    80.7     774.3
1,1,1,1,2,2,2,3,3         38    9363.4    72.5     403.7
1,1,1,1,2,1,2,1,1         12    8225.3   175.3     534.0
1,1,1,1,2,1,1,1,1          9    8253.9   216.5     599.4
1,1,1,1,2,1,2,3,1          9    8601.1   100.9     281.0
```

The one-pass strata are 12 % to 18 % faster per row, but they are also the
newest submissions with the highest scores, so the levels are confounded with
everything else those solvers shipped on 2026-08-17. That confound pushes
**towards** Model P, and Model P still loses. The two-way within estimator is
the one that removes it, and it returns `f = 50.4 ± 253.0`.

#### Joint fit with `b` constrained equal

Full width range, per-submission intercepts, 1616 points over 202 submissions:

```
Model P   b = 6737.4 +- 64.2   f = 6022.0 +- 246.6   rmse 2255.7  aicc 25424.9
Model R   M* = 3.500   b_lo = 4991.1   b_hi = 9533.9  rmse 1783.5  aicc 24668.3
P + R     f = 2265.4 +- 222.3   step = 3881.5 +- 122.1   rmse 1721.2
linear    b = 8059.8                                    rmse 2690.6  aicc 25992.1

AICc prefers Model R, dAICc = 756.5
BIC  prefers Model R, dBIC  = 751.8
```

Over the full range Model P has to explain the low-to-high slope change with a
level term and gets `f = 6022`, but it fits far worse than one hinge. Fitting
both together halves `f` to 2265 and leaves the hinge carrying 3882. Above the
break, where the strata actually differ, `f` collapses to zero. The consistent
reading is that the hinge is real, the pass term is not, and Model P's apparent
`f` over the full range is the hinge in disguise.

#### What this prices

Thorfinn's one-pass tables, repriced on the measured `f` instead of the curve's
own tier break (`research/e128_reprice_onepass.py`, Rule 67 exact median,
receipt `d3c491b5`, curve `slopeonly_b6`):

```
table         pass price used        c=0        c=0.445   c=0.445 everywhere
{6:6}         measured 50.4      +0.0068       -1.6594       -5.5417
{6:6}         95 % upper 546.2   +0.0733       -1.4111       -5.4789
{6:6}         curve unit 10187   +1.3842       +0.9854       -4.2407
{6,7}         measured 50.4      +0.0156       -2.8792       -5.5334
{6,7}         95 % upper 546.2   +0.1690       -2.4713       -5.3885
{6,7}         curve unit 10187   +2.4312       +1.7731       -3.2518
{6,7,8}       measured 50.4      +0.0510       -4.0704       -5.4999
{6,7,8}       95 % upper 546.2   +0.5562       -3.5148       -5.0228
{6,7,8}       curve unit 10187   +11.1169      +7.2103       +4.9520
```

The arm is worth at most `+0.56 %` even at the 95 % upper edge of `f`, and it
is net negative as soon as the residency loss is priced at any `c` above about
0.02. My own F6 census pricing of `+2.2006 %` for `{6,7}` and `+12.3266 %` for
`{6,7,8}` was 40x to 240x too optimistic, because it read the saving off the
curve's tier break rather than measuring the pass. **I withdraw the F6 arm-2
prices and recommend against spending a submission slot on the one-pass
family.**

This also settles the F6 census dispute. The census unit was 10,187.2 µs and
the profile could not reject it below 4,881 µs at 5 %. The board rejects
anything above 546 µs at 95 %, a 19x tighter bound, and confirms the profile's
maximum-likelihood `f = 0` with n in the hundreds instead of six receipt
points.

#### Honest limits

- `f` is a **level** price. The board cannot rule out a pass-count effect that
  scales with M, because `G` and M are close to collinear inside one stratum.
- The M = 5 boundary is untestable on the board, so F130.1's board-versus-us
  break difference is still explained equally well by either model.
- The round-count multiple is chosen by curve consistency. The choice is
  identified 206x more strongly than the effect being measured, but it is a
  choice, and a reader who rejects it should treat the whole item as
  unidentified rather than as evidence for Model R.
- Stratum membership is confounded with submission date and score. The two-way
  within estimator removes the level part of that confound but not a slope
  part.

### F7 item 4 — the head share, re-derived

The proposal head runs once per proposed draft, so a round of M rows carries
`M − 1` head steps and the head enters the fitted curve as
`h(M − 1) = −h + hM`. It sits **entirely in the per-row slope**, with a
compensating `−h` in the intercept. `h` and the target per-row cost are
perfectly collinear in slope space, so `h` can never be fitted from the curve;
it has to be pinned from outside. That is exactly why a wrong F13 propagated
silently.

Using the advisor's frame, which reproduces both of their published numbers
(round 55,645.4 µs, 4.3818 steps, 323.59 MB per step):

```
head share                    %   us/round  h us/step   rate GB/s   k_lo-h   k_hi-h   ratio
F13 retracted              1.82     1012.7      231.1      1400.1   3214.9   5092.4  1.5840
byte/rate at 462.2 GB/s    5.51     3066.1      699.7       462.5   2746.3   4623.8  1.6836
corrected low 7 %          7.00     3895.2      888.9       364.0   2557.1   4434.6  1.7342
rung-0b via F35            7.10     3950.8      901.6       358.9   2544.4   4421.9  1.7379
E79 anchor                 8.40     4674.2     1066.7       303.3   2379.3   4256.8  1.7891
corrected high 9 %         9.00     5008.1     1142.9       283.1   2303.1   4180.6  1.8152
E82 draft_build local     10.06     5597.9     1277.5       253.3   2168.5   4046.0  1.8658
```

My 1400.1 GB/s reproduces the advisor's 1391 GB/s to 0.7 %, so the frame is
confirmed.

**The answer to the third-model question is no, and the sign is the opposite of
the one hoped for.** A four-times-larger head term does not absorb the 1.82x.
The head chain costs the same per step at every width, so it is a constant
subtracted from both slopes, and subtracting a constant from both terms of a
ratio above one moves that ratio **further** from one. The residual target-side
slope ratio rises from 1.5840 under F13 to 1.7342 at a 7 % head and 1.8152 at
a 9 % head. Essentially the whole 1.82x survives on the target side, and at a
9 % head the target-side ratio is 1.8152, which is the 1.82x almost exactly.

**Effect on conditioning: better in accuracy, worse in the residual puzzle.**

- Better, because `h` is a fixed exogenous input to the decomposition, not a
  free parameter. Four independent lines now agree on 7 % to 9 %, against F13's
  chain of two unmeasured coefficients, and the implied streaming rate moves
  from an impossible 1400 GB/s to a plausible 283 to 364 GB/s under the ranked
  M = 1 rate of 462.2 GB/s.
- Worse, because the correction moves 657.8 to 911.8 µs per row out of the
  target term and into the head term. That is 19.1 % to 26.5 % of `k_lo` and
  12.4 % to 17.1 % of `k_hi`. Every target-side per-row conclusion I drew from
  the fitted slopes was overstated by that amount, and the unexplained
  target-side step gets larger, not smaller.

The head is now 25.8 % to 33.2 % of the marginal cost of a draft row below the
break and 16.7 % to 21.5 % above it. That reprices head work upward by a factor
of four across the campaign.

**The E128 headline is unaffected.** The reach estimator is priced on the
end-to-end fitted curve, and the internal head-versus-target split of `k` never
enters that pricing. `−0.2171 %` for `levelfix1.05` and `+8.5248 %` for the
oracle stand exactly as reported.

---

## F8 - Finding 150, the per-row runner state in the cost curves

`harness=ranked`, board receipts only, zero GPU. Script `research/e128_state_fe.py`
with `research/e128_state_s45.py`; artifact `research/e128-artifacts/f8-state-fe.json`.
Reproduce with `cd research && python3 e128_state_fe.py --json e128-artifacts/f8-state-fe.json`.

### The frame correction Finding 150 needs

Finding 150 writes the state as `round_us(row, M) = a + s_row + b*M`. That
collapse is not exact in this frame. `round_us` is the run's mean over ALL
rounds, drafting and non-drafting together, because `effective_mean_draft_len
= D/R` and `round_us = 512*spt/R` divide by the same total round count `R`. A
constant added only to drafting rounds enters the mean scaled by the drafting
fraction `phi = 1 - non_drafting_round_count / R`, so the exact form is

```text
round_us(row, prompt) = a + b * mbar + s_row * phi
```

Both forms are fitted below. The flat form is the literal request and is the
conservative one, because a free per-row intercept absorbs strictly more than
the state does. The `phi` form is what Finding 150 actually predicts, and it is
testable inside a single submission row because `phi` varies across the eight
prompts of one run.

### Item 1 - our ranked curve with a per-row intercept

Our ranked curve is fitted on ONE submission row. `e128_slopes.py` calls
`load_receipt(board, "d3c491b5")` and `build_points` returns eight points, one
per prompt, from that single receipt. The premise "pooled receipts from rows in
different states" is therefore not true of our curve. It is true of F97's board
curve over 147 official runs, and that curve is refitted with the state term in
item 3.

Because there is one submission row, `s_row` has exactly one level and is
perfectly collinear with `a`:

```text
prompt         R     n0    phi     mbar   round_us
plutarch     487    449  0.078   1.1540    31806.5
drama        252      0  1.000   3.2976    39306.3
travel       212      0  1.000   3.6557    41051.1
beagle       110      0  1.000   5.3818    53011.9
republic      93      0  1.000   5.9892    57021.7
essays        92      0  1.000   6.0870    57951.2
medicine      90      0  1.000   6.2556    59015.7
botany        81      0  1.000   7.1481    64734.9

design rank with the s_row column 2, without it 2, columns 3
line, no s_row        a 22439.9 +- 1969.0   b 5757.9 +- 377.5   rmse 1575.2
line, + flat s_row    a 11220.0 +- 1100.7   b 5757.9 +- 422.0   rmse 1575.2
slope moves by 0.000000 us/row; residual rmse moves by 0.000000 us
```

The exact `phi` form IS identified inside one row, because plutarch drafts on
only 38 of its 487 rounds while the other seven prompts draft on every round:

```text
line, + s*phi         a 24664.0 +- 305.9   b 6715.9 +- 79.4   s -7788.2 +- 485.9
phi correlation with mbar 0.7528
```

On a single-slope line `s` is nonsense: `phi` is 0.75-correlated with `mbar`, so
a badly misspecified line hands the state term its curvature. In the correct
two-segment form below, `s` lands at `+973 +- 3504 us`, consistent with zero and
consistent with the advisor's `817 us`.

### Item 2 - does the M >= 6 break survive the state term?

Yes, in both forms.

```text
no s_row
  break M>=6   a 27894.3   b 3388.3 +- 83.7   jump 10322.5   dslope 2779.3 +- 185.5   rmse   90.6  bic  82.5
  break M>=4   a 28989.8   b 2369.5           jump  5670.5   dslope 4243.4            rmse  118.6  bic  86.8
  break M>=5   a 28083.6   b 3217.5           jump  6785.0   dslope 3325.3            rmse  127.1  bic  87.9
  line         a 22439.9   b 5757.9                                                   rmse 1575.2  bic 124.0
  break advantage over the line   dBIC 41.54

+ flat s_row  (rank deficient; RSS and every slope identical to the row above)
  break M>=6   b 3388.3 +- 102.5   jump 10322.5   dslope 2779.3 +- 227.2   rmse   90.6  bic  84.6
  line         b 5757.9 +- 422.0                                          rmse 1575.2  bic 126.1
  break advantage over the line   dBIC 41.54

+ s*phi
  break M>=6   b 2976.1 +- 1487.1  jump 11556.2   dslope 3150.3 +- 1354.2  s  973.2 +- 3503.7  rmse  88.9  bic 84.3
  line         b 6715.9 +-   79.4                                         s -7788.2 +-  485.9  rmse 195.0  bic 92.7
  break advantage over the line   dBIC 8.42
```

Read BIC, not AICc, on these eight points: at `k=6` the AICc correction has
`n - k - 1 = 1` in its denominator and the penalty explodes, which is why the
flat row shows `dAICc -28.97` while its RSS is bit-identical to the no-state
row.

The break does not depend on the state term. With the flat form the fit is
algebraically the same fit. With the `phi` form the two-segment residual falls
further, to `88.9 us` against the line's `195.0 us`, the upper-tier slope
increment stays positive at `3150.3 +- 1354.2`, and the slope ratio rises to
`2.06`. **The `M >= 6` break is not an artifact of unbalanced state
assignment.** Model P is not rescued by Finding 150.

### Item 3 - the board population with a per-row intercept

Panel: every board row with eight complete per-prompt records, `803` rows and
`6424` points; the table-bearing subset from F7 is `202` rows and `1616` points.
Round counts come from the F7 rational reconstruction. Standard errors are
clustered on submission.

```text
Model P   T = a + s_row + b*M + f*ceil(M/IPG), b common across strata
  no s_row                a 16389.5   b 6775.0 +- 119.7   f 6065.5 +- 439.1   rmse 5299.8  aicc 27723.8
  row FE, clustered                   b 6737.4 +-  77.0   f 6022.0 +- 313.3   rmse 2255.7  aicc 25424.9
  row+prompt FE, clustered            b 6644.5 +- 139.7   f 3197.4 +- 370.6   rmse 1783.5  aicc 24684.1
  row+prompt FE, + s*phi              b 8842.1 +- 293.1   f 2907.5 +- 362.0   rmse 1685.4  aicc 24504.1

Model R   T = a + s_row + b_lo*M + d*max(0, M - M*)
  row+prompt FE, M*=4.375   b_lo 5699.7 +- 197.2   d 4013.5 +- 454.2          rmse 1739.4  aicc 24603.4
  row+prompt FE, one slope  b    6964.9 +- 174.1                              rmse 1869.1  aicc 24833.1
  row+prompt FE, R and P    b_lo 5395.7 +- 191.9   d 3975.9   f 3152.4        rmse 1649.7  aicc 24434.9

P against R with the state term in both:  dAICc 80.8   dBIC 80.8  (favours R)
```

The per-row intercept does what Finding 150 predicts: it removes a large error
component. Residual rmse falls from `5299.8` to `2255.7 us` with row FE and to
`1783.5 us` with row and prompt FE. It does not, however, change the ranking.

`f` looks strongly positive over the full width range, and that is a trap:

```text
corr(gbar, hinge at M*=4.375) 0.7854      corr(gbar, step at M>=5) 0.8582
```

`G(M)` steps at `M=5` in 199 of the 202 table-bearing trees, so over the full
range `G` IS the break wearing Model P's clothes. The identifying variation for
`f` is the high-width region, where `G` runs 1, 2, 3 and 4 across strata while
the break is behind us:

```text
high width, mbar >= 4.5, 996 points, 200 rows, row+prompt FE, clustered
  b 9760.9 +- 87.6      f 50.4 +- 253.0      rmse 536.0
```

Unchanged from F7. **The ranked level price of one extra QMV dispatch pass
remains `50.4 +- 253.0 us`, 95 % CI `[-445.5, +546.2]`, statistically zero,
after the per-row intercept is added.**

The per-stratum free break is also unchanged by the row intercept:

```text
G(1..9)                   subs     M*       b_lo       b_hi      rmse   G steps at
1,1,1,1,2,2,2,2,3          121  3.500     5034.7     9886.6    1946.4   [5, 9]
1,1,1,1,2,2,2,3,3           39  3.375     4800.9     9565.9     419.8   [5, 8]
1,1,1,1,2,1,2,1,1           12  1.500    -8694.6     7732.8    1801.3   [5, 6, 7, 8]
1,1,1,1,2,1,2,3,1            9  3.375     4692.6     8700.6     407.1   [5, 6, 7, 8, 9]
1,1,1,1,2,1,1,1,1            9  3.500     4769.8     8338.5     568.3   [5, 6]
```

The break stays near `M* = 3.4` whatever the tree's own pass vector says. That
is Model P's forbidden outcome, and the state term does not repair it.

### Item 4 - a calibrated state classifier

A per-row intercept in a global cost curve is **not** a state estimator. On the
full board its range is `72,037 us`, `44x` the `1,640 us` the state model
predicts, because it absorbs the whole quality difference between one solver
tree and another.

The two channels have to be separated by their per-prompt shape. Writing `dr`
for drafting rounds and `W = 512`,

```text
spt_ip = c_i * base_p  +  s_i * 1e-6 * dr_ip / W
```

`c_i` is the row's multiplicative tree speed and `s_i` is the additive state in
microseconds per drafting round. The shapes differ: on our crown the drafting
share runs 0.074 on plutarch to 0.492 on drama while per-token time runs the
other way, 0.0303 s on plutarch down to 0.0102 s on botany. That is the same
asymmetry that makes plutarch the state's null. Fitted by alternating least
squares.

**The decisive check.** Refitting on exactly the six rows of Finding 150's own
comparison:

```text
six-row local fit, relative misfit 0.00141
id        solver                 c  s_rel us       se      fit       f76
d3c491b5  morganmcg1        0.9988       739      188  0.00302    -13.36
cf79f7df  Lieisyourlie      1.0007       -26       40  0.00066    -14.21
48423d09  noskillcoding     0.9997        42       47  0.00076    -14.17
3b376ba2  Lieisyourlie      1.0007       -70       39  0.00064    -14.25
390ec878  newjordan         1.0004        13       45  0.00073    -14.20
c63eaa21  newjordan         0.9997        40       53  0.00087    -14.17
```

An estimator built from a different frame, on 803 board rows and then localised,
puts our crown `739 +- 188 us` per drafting round above the cluster. The
advisor's independent `modetest.py` says `817 us`. That is agreement to
`0.4` standard errors. The five cluster rows span `112 us` with individual
standard errors of `39` to `53 us`, across three solver accounts and at least
three trees. **Finding 150 is confirmed by an independent estimator.**

Campaign-wide, one common `base_p` fails: relative misfit `0.0583`, five times
the `1.15 %` the state is worth. `base_p` is only a valid shape inside one
draft-schedule family, so rows are grouped by the normalised per-prompt
drafting-round vector, which is a schedule fingerprint needing no source access,
and the estimator is refitted family by family:

```text
fam    rows   misfit      s p5     s p95  gap k=2  gap k=3    F76   3-way centres
1        76   0.0351    -36522      9300    35776    20868  0.868     -34396    -4229     7340
2        29   0.0071    -58189      8104    55053    30577  0.759     -59097   -20665     2057
4        20   0.0691    -25353     14121    32826    22532  0.850     -31508    -1129    13557
5       359   0.0237     -1924      2888     2556    16450  0.822     -29509      -82     3391
6        92   0.0106     -1208      2187     9093     6874  0.935       -362     1308    13386
7       220   0.0166      -459      4343     5047     2731  0.645       -223      927     5239
```

Family 7 holds our recent frontier. Restricting to our own 13 rows in it:

```text
2-way centres -218 and 809 us, spacing 1027 us, variance explained 0.852
agreement with the F76 -12.9 band 1.000 (13 of 13)
slow-state rows 32c6dc69 8630bc07 55af6534 84b9ef7b 7bef7d4c
```

**Cluster centres, spacing, agreement rate.** Combining the two anchors gives
three levels on one scale, referenced to the family-7 median:

```text
level          s us/drafting round   evidence
slow                          +809   five morganmcg1 rows, F76 index above -12.9
middle                        -218   eight morganmcg1 rows including d3c491b5, F76 below -12.9
fast (S3)                     -957   the Finding 150 cluster, first board row 2026-08-22T09:28Z
```

Spacings `1027 us` and `739 us`, mean `883 us`, total span `1766 us`. Finding
150 predicts about `820 us` and `1640 us`. Classification agreement with the
F76 `-12.9` band is `13 of 13` inside family 7. **The state model is confirmed
and the F76 index is a working classifier inside one schedule family.**

It is not yet a campaign-wide classifier. Across all families whose rank-1 tree
model holds to 2 % the agreement falls to `0.595` and the correlation to
`-0.4013`, because each family's `s` is referenced to its own median and the
medians are not on a common scale. Fixing that needs a row measured in a known
state inside each family.

### Item 5 - the estimated state of every morganmcg1 board row

`s_local` is microseconds per drafting round relative to the row's own
schedule-family median. `fit` is the row's relative residual under the rank-1
tree model; a value above about `0.02` means the estimate is not trustworthy.

```text
id        created                score        c  fam   s_local      se      fit      f76 promotion
4437d061  2026-08-17T22:03:30  2.86127   0.9568    1       461    3451  0.02757     9.19 None
9197ed62  2026-08-18T17:08:40  3.06938   1.0512    5       318    7050  0.05685   -13.71 None
ca9251b8  2026-08-18T22:44:18  3.23251   1.0018    5      -255    1190  0.01041   -13.71 None
2c766441  2026-08-19T06:19:42  3.07213   0.9388    1      3008    3094  0.02864     6.50 None
ff73cbbd  2026-08-20T06:42:09  3.17230   1.0027    5       468    1359  0.01174   -13.22 None
9b241879  2026-08-20T09:30:36  3.23589   0.9975    5         2    1356  0.01189   -13.65 None
2da69933  2026-08-20T12:00:44  3.21126   0.8777    1       481    1373  0.01345     9.34 None
32c6dc69  2026-08-20T18:48:46  3.28158   0.9780    7      1178     522  0.00642   -11.92 None
8630bc07  2026-08-20T21:15:23  3.27747   0.9787    7       976     568  0.00701   -12.00 None
83f0b282  2026-08-21T00:43:51  3.31378   0.9838    7       -98     367  0.00458   -13.15 None
55af6534  2026-08-21T01:48:52  3.28303   0.9819    7       855     366  0.00451   -12.14 None
87e6421b  2026-08-21T04:14:33  3.30652   0.9836    7       -83     371  0.00463   -13.13 None
cb8aeefb  2026-08-21T06:26:12  3.32346   0.9819    7      -214     480  0.00600   -13.23 None
84b9ef7b  2026-08-21T08:16:17  3.30142   0.9780    7       700     623  0.00774   -12.20 None
f04b102e  2026-08-21T10:17:01  3.32825   0.9814    7      -246     509  0.00637   -13.29 promoted
87b654b2  2026-08-21T13:47:16  3.12601   0.9897    4     -6171    6172  0.08215   -15.18 None
b8b8b860  2026-08-21T15:58:25  3.33412   0.9803    7      -312     553  0.00693   -13.37 None
44559d02  2026-08-21T18:35:20  3.34351   0.9830    7      -494     605  0.00759   -13.49 None
7bef7d4c  2026-08-22T00:30:55  3.29792   0.9824    7       336     456  0.00567   -12.52 None
cf9a9eda  2026-08-22T06:20:41  3.26815   0.9900    7       -23     199  0.00246   -12.99 None
d3c491b5  2026-08-22T08:27:51  3.49065   0.9587    7      -276    1833  0.02347   -13.36 promoted
```

Readings for re-pricing:

- Five of our rows were measured in the SLOW state: `32c6dc69`, `8630bc07`,
  `55af6534`, `84b9ef7b`, `7bef7d4c`. Every one of them has an F76 index above
  `-12.9`. Any comparison that treats these as equivalent to a `-13.x` row
  charges the candidate about `1027 us` per drafting round that the code did
  not spend.
- Eight family-7 rows sit in the MIDDLE state, including both promotions
  `f04b102e` and `d3c491b5` and the anchor `44559d02`. Comparisons inside that
  set are state-clean.
- `d3c491b5` is in the middle state, not the fast one. Its `08:27:51Z` creation
  time is before the `09:28Z` first appearance of S3, so the `739 us` gap to the
  Finding 150 cluster is a state gap and not a code gap.
- Three rows are the depth-sweep characterisations in family 1 with an F76 index
  near `+9`, and `87b654b2` in family 4 has a `0.082` relative misfit. Their
  states are not resolvable by this estimator and should not be re-priced.
- The four family-5 rows all sit in the fast F76 band; their `s` spread of about
  `720 us` is within their own standard errors, so they cannot be split.

### What F8 does and does not change

- The E128 headline is unaffected. Section 6 prices policies on the fitted
  end-to-end curve, and the curve is unchanged: the flat state term is
  algebraically absorbed and the `phi` term leaves the `M >= 6` break in place.
- F7's answer to the Model P question is unaffected. `f = 50.4 +- 253.0 us` on
  the high-width segment, and the break still does not move with the tree's own
  pass vector.
- F6's withdrawn one-pass arm prices stay withdrawn.
- New for the campaign: five of our own board rows are measured in the slow
  state, the F76 index classifies our family-7 rows perfectly, and an
  independent estimator reproduces the advisor's `817 us` offset as
  `739 +- 188 us`.

### Suggested follow-ups, not implemented

1. Put every schedule family on one state scale by submitting one deliberately
   identical repeat of a known-state tree inside each family. Without that
   anchor the classifier stays family-local.
2. Re-price the F5 two-channel mode estimator with the state term. F5 attributes
   `96.8 %` of the rival gap to a MODE move; F8 now gives that mode a physical
   unit, and the two should be reconciled.
3. `f = 3152.4 +- 329.2` survives beside the hinge in the joint board fit. Neither
   pure model is complete. A third shape, for example a pass term that carries
   its own per-row work, may fit both facts.

## F9 - the known state step, Thorfinn's fixed shape, and the round price

`harness=ranked`. Zero GPU. `research/e128_f9.py`, artifact
`research/e128-artifacts/f9-state-shape-price.json`.

F9 replaces F8's free per-row intercept with Alphonse's identified label, tests
Thorfinn's deleted-instruction shape as a fixed regressor, and audits the
shipped depth price for a per-drafting-round term.

### 1. The state as a known offset

F8 fitted `d3c491b5` against the five-row cluster
`cf79f7df 48423d09 3b376ba2 390ec878 c63eaa21` and got `739 +- 188 us`. Those
five rows are exactly Alphonse's cluster 0 and `d3c491b5` is in his cluster 1,
where he measures `928.1 +- 31.9 us`. The two estimates are 1.01 of our SE
apart, from different methods on the same account partition.

Inside one receipt `phi` takes two values, `0.078` for plutarch and exactly
`1.000` for the other seven prompts. A free `s` is therefore a plutarch
indicator, not a state term, which is why F8's free-s line fit fell to
rmse 195.0 with `s = -7788 us`: it deleted one point.

Entering the step as a fixed offset `y - 930 * phi`:

```
                             k   rmse     bic     b                step
line, raw                    2  1575.2  124.03   5757.9+-344.6
line, known s=930            2  1760.5  125.81   5643.5+-385.1
break M>=6, raw              3   285.3   98.77   3446.1+-202.3     1877.5+-154.6
break M>=6, known s=930      3   323.3  100.78   3060.9+-229.3     2097.4+-175.2

slope ratio hi/lo   raw 1.545   known s 1.685
dBIC(line - break)  raw +25.26  known s +25.04
```

Break sweep under the known offset: `M>=3` 106.78, `M>=4` 114.37, `M>=5`
109.46, **`M>=6` 100.78**, `M>=7` 121.81, `M>=8` 126.36. The break stays at
M>=6 and the slope ratio rises. Model P is not rescued.

### 2. Thorfinn's fixed shape

`u(M) = 38 * M / IPG(M)` from the `let cases` table at `Qwen35.swift:1565`:

```
M              1       2       3       4       5       6       7       8       9
u   ours    38.0    38.0    38.0    38.0    38.0    76.0    66.5    76.0   114.0
u   board   38.0    38.0    38.0    38.0    63.3    76.0    66.5    76.0   114.0
```

The shape steps at M=6 under our table and at M=5 under the board table, which
are the two measured break locations, with no fitted break parameter. That
part of the prediction lands.

The magnitude does not follow, for two reasons.

`38 * ceil(M / IPG) = 38 * pass count`, so the exact accounting is Model P
rescaled. `TH exact + M` and `P free pass` agree to the last digit at
rmse 788.7 and bic 115.04, with `g = f / 38`. The only place the ideal
per-output-element form differs is the M=7 tail group, where it predicts a cost
dip, `u(7) = 66.5` below `u(6) = u(8) = 76.0`.

The one-parameter form fails because `u` is flat across M=1..5 while the round
cost rises about 3.1 ms per row there:

```
                       k   rmse    bic
TH ideal 38M/IPG       2  2513.1  131.51
line a+bM              2  1575.2  124.03
TH ideal + M           3   617.6  111.13
P free pass            3   788.7  115.04
R free break M>=6      4    90.6   82.50
```

Board population, 202 table-bearing rows x 8 prompts, each row using its own
IPG table, row and prompt fixed effects, SEs clustered by row:

```
                        rmse       aicc
M only                1869.1   24833.1
P  M + passes         1783.5   24684.1
TH ubar only          3476.1   26838.3
TH ubar + M           1784.7   24686.4
R  M + hinge(4.375)   1739.4   24603.4
R + P                 1649.7   24434.9
```

The dip test `[M, passes, dip]` gives `dip = 34.4 +- 42.7`. The ideal form
predicts 86.7 and the exact form predicts 0, so neither is rejected;
`corr(ubar, gbar) = 0.9894` makes the test underpowered by construction.

Verdict: the break location is instruction-accounted on both tables; the break
magnitude is not. Model R survives.

### 4. The per-drafting-round term and the shipped depth price

The shipped price has a fixed per-**round** term, the literal `1.0` that
`prefixCosts` (`:936`) and `makeUniformDepthPrice` (`:871-878`) seed the
cumulative with, giving `cumulative[d] = 1 + d*0.18`. It is charged whether the
round drafts or not, so it is not a per-drafting-round term. The correct cost
is `C(0) = 1` and `C(d) = 1 + s_u + d*h` for `d >= 1`, which no shipped
`DepthPrice` can express: `prefixCosts` pins the base at 1.0 and every arm
holds the marginal total at `maxDepth * headStepCostRatio`.

The mispricing runs both ways. At the entry decision the shipped rule extends
when `reach > h` while the correct rule needs `reach > s_u + h`, so it enters
drafting too eagerly. Once drafting, the correct cumulative is larger, so the
correct threshold is smaller and the shipped rule stops too early.

`s_u = 930 / 31139.8 = 0.02987`, 16.6 % of one 0.18 step. Optimal fixed depth,
`phi` held at the shipped value, acceptance from the rung-1 forced-depth-7
legs, de-stated `slopeonly b6` cost curve:

```
prompt        phi   d*(0)   d*(930)   s to go deeper   s to go shallower
plutarch    0.078       4         4          +41750                none
drama       1.000       4         4          +11550              -25400
travel      1.000       4         4          +53475              -15175
beagle      1.000       4         4          +30475              -22500
republic    1.000       4         4          +20475              -23650
essays      1.000       3         3          +10525              -14200
medicine    1.000       4         4          +34625              -22350
botany      1.000       4         4          +61450              -14400
```

Ranked median difference `+0.00000000`. The smallest critical cost anywhere is
`+10,525 us`, 11.3x the state term. The result holds across three de-stated
curves: `slopeonly b6` and `piece b6` move no prompt, `line` moves medicine
from 5 to 6, and the median difference stays `+0.00000000`. The depth ladder is
quantised by the M=6 pass cliff, `C0(6)-C0(5) = 15,645 us`, which is 16.8x the
term, and by the within-segment step of 3,061 us, still 3.3x the term.

The optimal flat threshold is `h = 0.270` at both `s = 0` and `s = 930`, so the
state term does not move it. That level is not a recommendation: it comes from
a pooled-EMA fixed-signal model that removes the shipped rule's adaptivity.

The state-aware price `marginal[0] = 0.18 + s_u` changes one prompt's depth,
beagle 6 to 7, worth `+0.5888 %` on beagle's own raw. Beagle is not a median
prompt, so the published median is unchanged in both states: gain if slow
`+0.00000000`, cost if fast `+0.00000000`. The hedge is worth zero.

The entry side is dead for a structural reason. The top-2 margin is
`score[0] - score[1]` on a sorted pair, so it is non-negative, and
`conf = 1/(1 + exp(-margin/2))` lies in `[0.5, 1)`. The lowest confidence in
1,678 recorded rounds is `0.5000`, 2.38x the corrected gate of `0.2099`. The
margin censor at `:1083-1087` can never pull `p` below 0.5, so it can never
reach either gate; the entry decision is set by `positionAcceptEMA[0]` alone.
A separate consequence is that the `conf` and `conf2` censors are only ever
active when the corresponding EMA is above 0.5, so they cannot deepen a cold
prompt or gate a struggling one.



## F10 - the serial lottery, the median carriers, and the row-keyed shape

F10 asks four things: replicate Finding 153 independently, re-read every E128
arm with beagle as a first-class line, settle the width-frame disagreement
between the local benchfixture and the ranked masses, and test the
one-parameter row-keyed shape `38 / IPG(M)`. One script,
`research/e128_f10.py`, answers all four. Zero GPU.

### F10.0 - Finding 153 replicates to the digit

I recomputed the serial statistics on the same 806 board rows that carry
complete per-prompt official metrics, from `/tmp/yukon-board/full.json`.

| prompt | n | mean s/tok | sd % | p5-p95 % | min-max % |
| --- | --- | --- | --- | --- | --- |
| beagle | 806 | 0.037990 | 0.239 | 0.705 | 2.218 |
| medicine | 806 | 0.037991 | 0.230 | 0.695 | 1.720 |
| essays | 806 | 0.037998 | 0.226 | 0.676 | 1.893 |
| republic | 806 | 0.037991 | 0.226 | 0.657 | 1.978 |
| botany | 806 | 0.037997 | 0.232 | 0.682 | 1.868 |
| travel | 806 | 0.038001 | 0.230 | 0.662 | 1.984 |
| plutarch | 806 | 0.037994 | 0.209 | 0.640 | 1.780 |
| drama | 806 | 0.037991 | 0.232 | 0.684 | 1.783 |

Run-level serial mean sd is `0.1115 %`; the within-run per-prompt residual sd
is `0.1992 %`, `1.79x` larger. The draw is therefore per pair, not per run,
exactly as Finding 153 states.

The median-carrier census also reproduces exactly: beagle occupies order
statistic 4 or 5 on `789 / 806` rows (97.9 %), then medicine 43.2 %, essays
24.7 %, republic 20.8 %, botany 11.4 %, travel 1.0 %, plutarch 0.7 %, drama
0.2 %.

**One correction to the stated approximation.** The shorthand
`published ~ 0.5*raw_beagle + 0.5*min(medicine, essays, republic, botany)` is
exact on only `770 / 806` rows (95.5 %). Across all 806 rows its error has
mean `-0.2139 %`, sd `1.4638 %`, and max abs `17.4007 %`. That sd is about
15x the `0.0967 %` serial-noise term the same finding warns about, so the
approximation is a good intuition but must not be used as a pricing surrogate.
Every price in this document uses the exact Rule 67 median.

Rule 100 re-ranking on a common serial leg confirms both claims in the
finding. Replacing each row's per-prompt serial numerator with the population
mean and recomputing the exact median moves the top of the board:

| id | solver | published | common-serial | rank move |
| --- | --- | --- | --- | --- |
| `1da2702f` | noskillcoding | 3.51845338 | 3.51669571 | -4 |
| `855c8d53` | Lieisyourlie | 3.51768892 | 3.52385635 | +1 |
| `a0f85886` | Lieisyourlie | 3.51661724 | 3.51865875 | 0 |
| `f95e74f1` | newjordan | 3.51594121 | 3.51630077 | -2 |
| `c57a39de` | morganmcg1 | 3.51270586 | 3.51964965 | +3 |
| `47cf806e` | Lieisyourlie | 3.51244872 | 3.51702161 | +2 |
| `8b54c469` | newjordan | 3.50774680 | 3.51589813 | 0 |
| `6f1cd66f` | morganmcg1 | 3.49065044 | 3.49231845 | 0 |

The published leader `1da2702f` falls to rank 5 on candidate merit, and
`c57a39de` rises to rank 2. Top-of-board ordering is not merit ordering.

### F10.1 - every arm re-read with beagle first-class

Every column below except the median is a candidate-leg quantity, so the
`0.0967 %` serial term does not apply to it. The `replay` column holds each
arm's per-prompt candidate effect fixed and recomputes the exact Rule 67
median on all 806 board raw vectors; it prices the arm against the population
carrier structure instead of against the single ordering our own receipt drew.
It is an ordering-robustness check on our own arm, not a claim about anybody
else's candidate.

| arm | median | beagle | min4 | replay | sd |
| --- | --- | --- | --- | --- | --- |
| `oracle` | +8.5248 | +13.1247 | +2.4925 | +9.0586 | 1.6538 |
| `ship` / `price0.18` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
| `levelfix1.05` | -0.2171 | -0.3028 | -0.6766 | -0.1901 | 0.2093 |
| `marginfull` | -0.3561 | -0.8092 | -0.2072 | -0.2917 | 0.6478 |
| `levelfix1.10` | -0.4012 | -0.5643 | -0.7888 | -0.3435 | 0.2582 |
| `price0.20` | -0.4856 | +0.4396 | -2.6853 | -0.4005 | 0.4865 |
| `expectedonly` | -0.6772 | +0.5736 | -3.5234 | -0.5542 | 0.6110 |
| `recal` | -1.2092 | -0.5038 | -2.3869 | -0.9082 | 0.3930 |
| `reachonly` | -1.8542 | -1.5808 | -2.6334 | -1.4985 | 0.4800 |
| `nomargin0` | -2.1716 | -2.0777 | -2.7849 | -1.7793 | 0.4949 |
| `rankedprice` | -2.8508 | -3.8851 | -2.4282 | -2.1546 | 1.4188 |
| `nomargin` | -3.7049 | -3.9119 | -4.0344 | -3.6192 | 0.3958 |
| `marginup` | -4.2494 | -4.0617 | -4.9372 | -3.8324 | 0.9551 |
| `jensen_both` | -7.8260 | -11.3311 | -5.5670 | -7.9383 | 0.8570 |
| `jensen` | -11.1665 | -14.7573 | -8.3604 | -10.6190 | 1.0871 |
| `static7` | -16.5209 | -20.0263 | -13.7641 | -15.8763 | 1.3778 |

The headline is robust to the carrier structure. `levelfix1.05` replays at
`-0.1901 +- 0.2093` against the single-receipt `-0.2171`, `oracle` replays at
`+9.0586 +- 1.65` against `+8.5248`, and the ordering of the arms is
preserved. The whole submitted conclusion, that the shipped price is a local
optimum and every implementable move is negative, does not depend on which
ordering our own receipt happened to draw.

Reading beagle alone changes the picture for two arms. `price0.20` is
`+0.4396` on beagle and `expectedonly` is `+0.5736`, both positive, but both
lose more on botany (`-2.685` and `-3.523`) than beagle gains. The `min` half
of the carrier structure is what kills them, which is exactly why the four
must be priced as a `min` and never as a mean.

### F10.2 - the width frame, arbitrated

The disagreement over Thorfinn's one-pass arm is a frame disagreement, so I
built both frames from the same script.

| prompt | carrier | mean M | P(M=8) | P(M>=6) |
| --- | --- | --- | --- | --- |
| beagle | 97.9 % | 5.3818 | 0.3373 | 0.4550 |
| medicine | 43.2 % | 6.2556 | 0.4484 | 0.6760 |
| essays | 24.7 % | 6.0870 | 0.0000 | 0.8243 |
| republic | 20.8 % | 5.9892 | 0.4846 | 0.5956 |
| botany | 11.4 % | 7.1481 | 0.6685 | 0.8319 |
| travel | 1.0 % | 3.6557 | 0.0074 | 0.0769 |
| plutarch | 0.7 % | 1.1540 | 0.0000 | 0.0004 |
| drama | 0.2 % | 3.2976 | 0.0072 | 0.0123 |

| aggregate | mean M | P(M=8) | P(M>=6) |
| --- | --- | --- | --- |
| carrier-weighted | 5.7947 | 0.3506 | 0.5804 |
| F83-weighted | 5.7732 | 0.3150 | 0.5861 |
| benchfixture (local) | 7.3590 | 0.7692 | 0.8718 |

My independent benchfixture reconstruction gives mean M `7.3590` and `76.9 %`
of the mass at M=8. Askeladd quoted `7.359` and about `77 %`. The two frames
are therefore both reconstructed correctly, and the disagreement is only about
which one prices the ranked score. The local frame carries `2.2x` the M=8 mass
of the carrier-weighted ranked frame.

Priced at the board-measured pass price `f = 50.4 +- 253.0 us` from F7, not at
the withdrawn tier-break reading:

| table | c | ranked median | ranked beagle | bench frame | withdrawn |
| --- | --- | --- | --- | --- | --- |
| `{6:6}` | 0.000 | +0.0068 | +0.0061 | +0.0049 | +1.5329 |
| `{6:6}` | 0.445 | -1.6594 | -0.4157 | -0.3348 | +1.1416 |
| `{6:6,7:7}` | 0.000 | +0.0156 | +0.0112 | +0.0078 | +2.8402 |
| `{6:6,7:7}` | 0.445 | -2.8792 | -0.7959 | -0.5526 | +2.2006 |
| `{6:6,7:7,8:8}` | 0.000 | +0.0510 | +0.0435 | +0.0668 | +16.3081 |
| `{6:6,7:7,8:8}` | 0.445 | -4.0704 | -3.2959 | -5.0485 | +12.3266 |

Because `f` is statistically zero, the honest `f +- 1 SE` band at `c = 0` is
`[+0.0000, +0.0407]` for `{6:6}`, `[+0.0000, +0.0938]` for `{6:6,7:7}`, and
`[+0.0000, +0.3081]` for `{6:6,7:7,8:8}`. At `c = 0.445` the band is
`[-1.6845, -1.5330]`, `[-2.9204, -2.6717]`, and `[-4.1265, -3.7878]`.

The ranked frame prices `{6:6,7:7}` at `c = 0.445` `5.2x` worse than the
benchfixture frame, `-2.8792` against `-0.5526`. The advisor's implied ratio,
`12.55 / 2.6`, is `4.8x`. Two independent reconstructions agree on the size of
the frame effect. The recommendation is unchanged: the whole one-pass family
is either statistically zero at `c = 0` or clearly negative at any real
residency cost, so it should not take a submission slot.

### F10.3 - the row-keyed shape partly reverses the F9 verdict

`row_keyed(M) = 38 / IPG(M)` is a per-row cost, not a per-round cost, so it
does not grow with M. It falls as the template packs more rows per group and
jumps back when the table drops IPG.

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 38.00 | 19.00 | 12.67 | 9.50 | 7.60 | 12.67 | 9.50 | 9.50 | 12.67 |
| board | 38.00 | 19.00 | 12.67 | 9.50 | 12.67 | 12.67 | 9.50 | 9.50 | 12.67 |

Ours jumps at M=6, the board at M=5, which is where each stratum's free break
landed. One free scale reproduces a level jump at the right place with no
fitted break location and no fitted jump size.

On our own eight measured points:

```
## raw
line a+bM        k=2 rmse 1575.2 bic 124.03  a 22439.9+-1797.5  b 5757.9+-344.6
row-keyed only   k=2 rmse 7606.9 bic 149.23  a 63446.9+-6050.2  q -924.8+-370.5
row-keyed + M    k=3 rmse  138.5 bic  87.21  a 12386.5+-432.9   b 6908.8+-56.3   q 317.4+-12.5
row-keyed x M    k=3 rmse  617.6 bic 111.13  a 13992.6+-1785.7  b 3699.1+-419.4  q 344.5+-65.7
R free break M>=6 k=4 rmse  90.6 bic  82.50  a 27894.3+-201.7   b 3388.3+-72.5   jump 10322.5+-365.1  dslope 2779.3+-160.6

## known s=930
line a+bM        k=2 rmse 1760.5 bic 125.81
row-keyed only   k=2 rmse 7631.4 bic 149.28
row-keyed + M    k=3 rmse  132.6 bic  86.52  a 10926.2+-414.6  b 6931.2+-53.9  q 355.1+-12.0
row-keyed x M    k=3 rmse  696.1 bic 113.05
R free break M>=6 k=4 rmse 88.9 bic  82.19  a 28273.1+-198.0  b 2994.4+-71.1  jump 11501.4+-358.3  dslope 3133.8+-157.6
```

F9 concluded that Thorfinn's ideal shape predicted the break LOCATION but not
its MAGNITUDE, because the per-round form `38 M / IPG(M)` left a large
residual. The per-ROW form does not. `row-keyed + M` uses three parameters,
fits no break location and no jump size, and reaches rmse `138.5` and BIC
`87.21`, against `1575.2 / 124.03` for a plain line. The free-break model R
still wins on BIC, but by only `4.71` raw and `4.33` under the known offset,
with one extra parameter. That is a much weaker margin than F9 implied, and it
means the observed jump is largely explained by a quantity that is computable
from the table alone.

### F10.3b - the same test on the board population

202 table-bearing rows, each row keyed by its own m-table, row and prompt
fixed effects, SEs clustered by row:

```
## raw
M only            rmse 1869.1 aicc 24833.1  b 6964.9+-174.1
row-keyed only    rmse 2915.4 aicc 26269.8  q -692.4+-48.3
row-keyed + M     rmse 1674.5 aicc 24480.3  b 9867.2+-249.0  q 484.5+-30.8
P  M + passes     rmse 1783.5 aicc 24684.1  b 6644.5+-139.7  f 3197.4+-370.6
R  M + hinge4.375 rmse 1739.4 aicc 24603.4  b_lo 5699.7+-197.2  d 4013.5+-454.2
R + row-keyed     rmse 1665.4 aicc 24465.4  b_lo 8888.8+-605.9  d 1372.7+-756.1  q 393.4+-53.0

## known s=930
row-keyed + M     rmse 1674.7 aicc 24480.8  b 9810.0+-246.8  q 509.1+-30.5
R + row-keyed     rmse 1662.5 aicc 24459.8  b_lo 8677.1+-596.8  d 1589.5+-749.0  q 403.6+-52.0

corr(row-keyed, mbar) -0.7554   corr(row-keyed, gbar) -0.3224
```

The board says the same thing more sharply. `R + row-keyed` is the best model
at aicc `24465.4`, and adding the row-keyed term to R shrinks the hinge from
`4013.5 +- 454.2` to `1372.7 +- 756.1`, which is no longer significant. On the
board population, the row-keyed shape absorbs the break.

### F10.4 - the state term as a factor, and the blocked item

Alphonse's `research/e130-artifacts/rung8-state-model.json` is absent from this
branch and his E130 branch is outside my read scope for this launch, so I
recovered the same factor from the public board and publish the labels for him
to check. The label source is the F76 mode index, which F8 checked against the
ALS state assignment and matched on 13 of 13 rows.

202 rows, k-means into 3 levels on the F76 index: mode 0 centre `-11.682`
(92 rows), mode 1 `-9.685` (80 rows), mode 2 `+9.276` (30 rows); index
variance explained `0.942`. With the mode entering as `s_k * phi`, in us per
drafting round, as a difference from mode 0:

```
M + phi                rmse 1760.0 aicc 24641.3  b 9296.7+-234.5  a -10614.4+-1154.3
M + phi + mode         rmse 1737.3 aicc 24604.6  b 9382.5+-235.9  a -11624.1+-1192.7  s1 1964.8+-568.9  s2 -286.9+-691.1
M + hinge + phi + mode rmse 1718.7 aicc 24572.5  b_lo 6588.6+-890.2  d 3178.3+-1001.2  a -3546.8+-2591.8  s1 1862.7+-572.3  s2 377.4+-804.4
M + pass + phi + mode  rmse 1667.3 aicc 24474.5  b 8956.0+-298.2  f 2822.4+-342.8  a -10690.5+-1320.0  s1 1622.6+-532.0  s2 -1229.4+-570.6

the VALIDATED two-way rule, index < -12.9 (1 of 202 rows):
M + phi + F76 slow     rmse 1760.0 aicc 24643.9  b 9296.9+-234.6  a -10615.2+-1155.5  s_slow 64.0+-261.0
```

**Honest negative.** Only 1 of 202 table-bearing rows falls on the far side of
the validated threshold, so this panel cannot identify the state step with
Alphonse's own rule. The three-way k-means split is finer than the validated
rule, so its labels are not his, and the widest index gap, mode 0 to mode 2,
carries no step at all, which is what a contaminated label looks like. Item 3
needs his file, or a panel that spans the modes.

The `costModelDepth` conclusion does not depend on that. F9.4 priced the
missing per-drafting-round term at the known `930 us` and found the ranked
median delta is exactly zero. The M=6 pass cliff on our curve is `15,645 us`,
`16.8x` that term, and the smallest cost that would move any prompt's chosen
depth is `10,525 us`, `11.3x` it. The largest mode difference recovered here
is `1964.8 us` per drafting round, still `5.4x` below that threshold. The
lever stays closed. What the factor form changes is the error bar: fitting `s`
blind on a single receipt produced a plutarch dummy, and this design cannot,
because the mode label comes from outside the fit.

One method warning for reuse: a full-panel ALS on all 202 rows is
unidentified. It recovers levels of `-66849 / -32830 / -8269 us`, because with
only 8 prompts per row the row cost `c_i` and the state term `s_i` are nearly
collinear. Use `als_state` only on a small same-family subset, as F8 did.



## W&B runs and reproduction

Project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group
`e128-reach-estimator-vs-ranked-depth-optimum`.

| run | id | what it holds |
| --- | --- | --- |
| `e128-rung0-replayer` | `vcxlulwz` | the replayer gate, 1.000000 agreement |
| `e128-f1-identity-and-r-pinning` | `nyypkdsl` | the identity tuple and the pinned R |
| `e128-rung1-uncensored-acceptance` | `mwv9a8fh` | 12 forced-depth-7 legs, uncensored acceptance |
| `e128-hypothesis-j-and-correction-sign` | `1azn4agj` | Jensen decomposition and the R-free sign |
| `e128-section5-our-ranked-curve` | `al4e8bmq` | our own fitted ranked cost curve |
| `e128-rung2-counterfactual-pricing` | `0wkulqix` | board-curve control pass |
| **`e128-rung2-our-curve-pricing`** | **`mys5l3kq`** | **the headline: primary metric, R band, curve sweep, board depth scan** |
| `e128-f4-to-f9-followups` | `5c4mq2lk` | 519 scalars from the zero-GPU advisor follow-ups F4 to F9 |
| `e128-f10-carriers-and-row-keyed-shape` | `k4sy1yco` | 406 scalars from F10: the serial lottery, the carriers, the width frames and the row-keyed shape |

`0wkulqix` carries a `metric_correction_note`. Its primary metric was first
written as `0.0` by the bug described in section 6, and was corrected in place
to `-0.2353 %`.

No GPU work was needed after rung 1. Every number in sections 3 to 7 is produced
offline from recorded rounds, public board receipts, and the fitted cost curve.

`RUNS` below is the directory holding the recorded leg dumps. `SHIPPED` holds
the shipped-schedule legs and `FORCED` the depth-7 legs.

```bash
# rung 0: the replayer gate, 3634 rounds over 21 legs
python3 research/e128_replay.py "$RUNS"/* \
    --json research/e128-artifacts/rung0-replay.json

# rung 1: uncensored acceptance from the 12 forced-depth-7 legs
#         (producing those legs is the only step that needs the model)
python3 research/e128_accept.py "$FORCED"/* \
    --json research/e128-artifacts/rung1-forced.json
python3 research/e128_accept.py "$SHIPPED"/* \
    --json research/e128-artifacts/rung1-shipped.json

# sections 3 and 4: Jensen decomposition and the R-free sign
python3 research/e128_jensen.py \
    --shipped research/e128-artifacts/rung1-shipped.json \
    --forced research/e128-artifacts/rung1-forced.json \
    --json research/e128-artifacts/jensen-and-sign.json

# section 5: fit our own ranked cost curve
python3 research/e128_ourcurve.py \
    --identity research/e128-artifacts/rung0-identity.json \
    --shipped research/e128-artifacts/rung1-shipped.json \
    --json research/e128-artifacts/section5-ourcurve.json \
    --curve-json research/e128-artifacts/our-ranked-curve.json

# section 6: the headline pass, sensitivity, R band and curve sweep
python3 research/e128_price.py \
    --accept research/e128-artifacts/rung1-forced.json \
    --shipped research/e128-artifacts/rung1-shipped.json \
    --jensen research/e128-artifacts/jensen-and-sign.json \
    --curve-json research/e128-artifacts/our-ranked-curve.json \
    --curve-key assumed --windows 400 --fit-windows 120 \
    --json research/e128-artifacts/rung2-ours-pricing.json \
    --sensitivity-json research/e128-artifacts/rung2-ours-sensitivity.json \
    --r-band-json research/e128-artifacts/rung2-ours-r-band.json \
    --curve-sweep board assumed ours_b4 ours_b5 ours_meanfit predicted \
    --curve-sweep-json research/e128-artifacts/rung2-curve-sweep.json

# section 7: the board-side depth scan
python3 research/e128_board_depth_scan.py \
    --json research/e128-artifacts/board-depth-scan.json

# advisor follow-ups F4 to F9, all zero GPU
python3 research/e128_width_histograms.py \
    --json research/e128-artifacts/f4-width-histograms.json
python3 research/e128_slopes.py \
    --json research/e128-artifacts/f4-slopes.json \
    --curves-json research/e128-artifacts/f4-candidate-curves.json
python3 research/e128_arm_prices.py \
    --json research/e128-artifacts/f4-arm-prices.json
python3 research/e128_two_channel.py \
    --json research/e128-artifacts/f5-two-channel.json
python3 research/e128_census.py \
    --json research/e128-artifacts/f6-census.json
python3 research/e128_signals.py \
    --json research/e128-artifacts/f4-signals.json

# F7 to F10 read their inputs relative to research/, so run them there
cd research
python3 e128_f7_artifact.py
python3 e128_reprice_onepass.py --json e128-artifacts/f7-onepass-repriced.json
python3 e128_head_share.py
python3 e128_state_fe.py --json e128-artifacts/f8-state-fe.json
python3 e128_f9.py --json e128-artifacts/f9-state-shape-price.json
python3 e128_f10.py --json e128-artifacts/f10-carrier-and-shape.json
cd ..

# publish
python3 research/e128_wandb_log.py --only rung2
python3 research/e128_wandb_log.py --only followups
python3 research/e128_wandb_log.py --only f10
```

Producing the rung-1 legs is the only expensive step, at roughly 40 minutes on
the M4 Pro. Every offline pass finishes in under 10 minutes.

One reproduction warning, learned the hard way in this experiment: do not pipe
these scripts through `head`. `e128_board_depth_scan.py` writes its JSON after
printing, so a closed stdout kills the write and leaves a stale artifact behind.
I committed a truncated artifact that way and only caught it when the logger
raised `KeyError: 'f83_weight'`.

### Scope evidence

Re-verified after the rebase onto `221065c5`:

```
git diff 221065c5 -- Sources/ Vendor/ mtp-head.manifest.json Package.swift \
    benchmark.json                                        (empty)
senpai/verify-ranked-score-boundary.sh                    PASS
senpai/check-editable-budget.sh 221065c5
    source=2594084/3000000  growth=0/262144  files=154    OK
```

Every file this branch adds is under `research/`, which `benchmark.json` does
not submit. The candidate surface is byte-identical to the base.

