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
