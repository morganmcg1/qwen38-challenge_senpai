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
correction, so the shipped `gamma = 1` is the optimum. **The estimator-level
axis is closed.** The scheduler axis is not: a per-round oracle is +11.0 %,
and the remaining headroom is per-round *discrimination*, not level.

## Sign flip, before any headline percentage

The advisor asked to lead with the R value at which the sign flips and where it
sits inside the pinned band.

**The estimator arms do not flip anywhere in the band, because they are R-free.**
The section 4 result is measured by replaying the recorded rounds and pricing
`sum(round cost) / sum(tokens)` directly. A per-token cost does not need a
ranked round count, so `R` never enters. `reachonly`, `expectedonly`, `levelfix`,
`jensen` and `jensen_both` carry the same sign at every `R` in
`beagle = 112.5 [105.2, 120.8]` and outside it.

The R-dependent statements are in rung 2 only, where the transfer to the hidden
ranked prompts is anchored on the R-derived accept rate. Those are reported as a
curve over four R values per prompt in section 5, and any arm that flips inside
the band is labelled sign indeterminate there.

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

Best implementable arm is `marginfull` at **-0.356 %**. Oracle is **+8.52 %**.

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

### Sensitivity, including the moved crown

Ten variants. The headline conclusion holds in nine of them:

```
variant             marginfull   levelfix  reachonly     oracle
headline               -0.3561    -0.9673    -1.8542    +8.5248
constant_p             -0.0253    +0.2467    -0.2462   +12.2410
shuffle_margins        -3.2060    +0.3434    -0.0654   +11.5990
drop_zero_weight       -0.3561    -0.9673    -1.8542    +8.5248
seed_777               -0.2851    -0.8657    -1.8106    +8.6394
single_fixture_a       -0.6477    -0.9280    -1.8689    +8.5540
single_fixture_b       +0.1247    -0.9599    -1.8024    +8.7489
benchfixture_only      -1.9715    -0.4679    -1.6349   +25.9863
receipt_crown          -0.3965    -1.1473    -2.0326    +8.4976
receipt_ec778a91       -0.3548    -1.0830    -1.9689    +8.5680
```

Three of these deserve comment.

**`receipt_crown` re-anchors the whole pass on `d3c491b5` = `3.49065044`,** the
receipt that took the crown. This is the direct answer to the F3 question of
whether the moved frontier changes the conclusion. It does not: `marginfull`
goes from -0.3561 to -0.3965 and every other implementable arm moves further
negative, not less.

**`constant_p` and `shuffle_margins` are the only variants that turn `levelfix`
positive,** at +0.2467 and +0.3434. Both are the deliberate controls that
destroy the margin signal — the first replaces the per-round acceptance
probability with a constant, the second permutes the margins across rounds.
Correcting the estimator level helps only once the margin override has been
stripped of its information. That is a confirmation that the override is doing
real work, not an escape route for the hypothesis.

**`single_fixture_b` is the one variant with a positive `marginfull`,** at
+0.1247. It is a single-fixture median, so it is one prompt's arithmetic rather
than a ranked estimate, and it is a quarter the size of the +0.20 % threshold
the assignment set.
