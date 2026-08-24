# E177 — Ranked depth-law recovery (desk analysis)

- Assignment: `e177-ranked-depth-law-recovery`, revision `r0`, PR #176
- Base: `senpai/qwen38-mtp-r1` @ `d9bb94c31b267044e3c7a8065f282a45968fe16a`
- Harness: `harness=ranked` throughout. No local timed leg was run.
- GPU seconds spent: 0. Timed local legs: 0. `gate_qualified_for_timing=false`.
- W&B: https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/gjn7stes
- Instrument: `research/e177_ranked_depth_law.py` (`rounds|survival|fit|price|all`)

## Question

Can the ranked M5 round-cost law `cost(M)` be recovered from the two paid
single-factor receipts (A cap 7, C cap 4), and does that law price the open
depth cells (5, 6, 8, per-prompt adaptive) above or below the paid cap-7 line
and the crown line `3.7291100105909`?

## Evidence that made it worth testing

Receipt A (`5a9f130a`, cap 7, published `3.70784519415395`) and receipt C
(`90c131dc`, cap 4, published `3.54742900664627`) differ by one literal
(`segmentedVerifyDepthCap`). FINDING 448 records them as a genuine
single-factor pair. The board exposes eight-row per-prompt official metrics,
so the pair is sixteen ranked observations of the same machine at two widths,
not two scalars.

## What I did

### Step 1 — the board does expose rejected-receipt per-prompt rows: YES

`officialMetrics.per_prompt` is present for `90c131dc` with all eight rows:
`mtp_seconds_per_token_mean`, `serial_seconds_per_token_mean`,
`raw_ratio_of_means`, `effective_mean_draft_len`, `non_drafting_round_count`,
`prefill_seconds_per_token`, `head_provenance_sha256`, `parity_ok`.

Two corrections came out of reading the actual fields:

1. `mtp_seconds_per_token_mean` **includes prefill**. Round cost must be built
   from `512 * (mtp - prefill)`, and a modelled decode time must add prefill
   back before it becomes a raw ratio. Skipping this inflated one score
   prediction by 13 %. Prefill is cap-invariant to within ±0.22 %.
2. Round counts are recoverable exactly. `effective_mean_draft_len` is the
   exact rational `proposed / rounds`, so `rounds` is a multiple of the reduced
   denominator; the 30.2519 ms batch-1 floor plus monotone round cost picks the
   unique member. The recovered counts reproduce the E159 round counts and the
   FINDING 374 alphas on all eight prompts of receipt A.

### Recovered observations (16)

| prompt | Mbar@7 | R@7 ms | rounds@7 | Mbar@4 | R@4 ms | rounds@4 |
|---|---|---|---|---|---|---|
| plutarch | 1.156 | 30.522 | 488 | 1.156 | 30.487 | 488 |
| drama | 3.298 | 34.241 | 252 | 3.298 | 34.166 | 252 |
| travel | 3.648 | 35.109 | 213 | 3.591 | 34.766 | 215 |
| beagle | 5.382 | 44.983 | 110 | 4.292 | 37.792 | 137 |
| republic | 5.989 | 47.723 | 93 | 4.562 | 38.740 | 121 |
| essays | 6.087 | 48.932 | 92 | 4.525 | 38.813 | 122 |
| medicine | 6.256 | 49.350 | 90 | 4.597 | 38.904 | 119 |
| botany | 7.148 | 54.474 | 81 | 4.827 | 39.985 | 110 |

Botany already decodes **faster at cap 4 than at cap 7** (4.3983 s vs
4.4124 s). Plutarch and drama are cap-inert.

### Step 2 — width mixtures

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1067` sets
`segmentedVerifyDepthCap = 7`, consumed at line 1133 by `costModelDepth` with
`while depth < cap`. Offered depth is therefore exactly `D = min(D*, cap)`, so
the two receipts observe the same latent depth demand truncated at two points.
I fitted the per-prompt survival `t_d = P(D* > d)` from `edl(4)` and `edl(7)`
with a two-parameter discrete Weibull, and also derived family-free monotone
bounds for the sensitivity sweep.

Calibration: reconstructing the local fixture survival from only
`edl(4)=3.9369` and `edl(7)=6.3590` matches the measured E168 depth histogram
`{2:1, 4:4, 5:106}` to a maximum `|t_d|` error of **0.0070**.

Model-free marginal acceptance `mu = d(abar)/d(edl)`: travel 0.391,
beagle 0.842, republic 0.893, essays 0.876, medicine 0.836, botany 0.718.

### Step 3 — the law is convex, decisively

Fitted on candidate-leg decode seconds with mixture-aware bases,
rounds-weighted, per-receipt intercepts, with the receipt-C intercept offset
constrained to zero (A and C are one literal apart, so no physical intercept
shift is available).

**Winner: `R(M) = 30.640 - 0.939*M + 0.576*M^2` ms.**

- wRMSE **0.2435 ms**; AIC **-176.6** vs **-140.9** for the next best.
- Specification test: released, the receipt-C intercept offset lands at
  **+0.006 ms**, which is what physics requires. Rivals need 0.46–2.74 ms of
  unexplained intercept to fit.
- Held-out anchor: `R(1) = 30.277 ms` against the independently known ranked
  batch-1 zero-draft round of `30.2519 ms` — error `+0.025 ms`. Rivals miss by
  `-0.39` to `+3.63 ms`.
- Stable under extreme tail allocations (`c = 0.56`–`0.69 ms/row^2`).

`R(M)` ms for `M = 1..9`: 30.3, 31.1, 33.0, 36.1, 40.3, 45.7, 52.3, 60.0, 68.8.
Marginal ms per extra row: 0.79, 1.94, 3.09, 4.24, 5.39, 6.55, 7.70, 8.85.

`R(7)/R(5) = 1.296`. The model-free wide-prompt round ratio is `1.2579`, so the
advisor's 1.3–1.4 prior was slightly high.

### Step 4 — pricing, validated out of sample

Held-out check that matters most: fit the law on **receipt A alone** and
predict the paid cap-4 published score.

| law | held-out cap-4 error |
|---|---|
| **quadratic** | **-0.381 %** |
| affine + wall6 + rows | +5.39 % |
| affine + wall6/groups + rows | +6.59 % |
| groups | +16.39 % |
| affine | -7.91 % |

`affine + wall6 + rows` had the best in-sample wRMSE (0.163 ms) and the worst
extrapolation. That is the overfit signature. The A+C walk-down repeats at
`-0.406 %`.

Anchor consistency (walking up from cap 4 and down from cap 7 must land on the
same number): quadratic with the tail-light allocation gives gaps of
`0.057 %` / `0.051 %`, the only configuration self-consistent inside the noise
floor. Every rival is at or above `0.68 %`.

**Priced cells (quadratic, tail-light, anchored on receipt A):**

| cell | published score |
|---|---|
| cap 5 | 3.65289 |
| cap 6 | 3.70170 |
| **cap 7 (paid)** | **3.70785** |
| cap 8 | 3.67148 |
| oracle per-prompt cap over 4..8 | 3.70784519 |

The oracle equals the paid cap-7 value to all printed digits.

Cap-8 break-even needs the marginal row cost below `Rbar*mu/(1+abar)`:
beagle 8.134, essays 7.701, republic 7.737, medicine 7.250, botany 6.188 ms.
The law gives 8.848 ms at width 9, so cap 8 **costs** on all five wide prompts.
The tightest margin is beagle: the law would have to over-estimate the width-9
marginal row cost by at least 8.1 % for cap 8 to break even.

## Result

**Outcome (b): the ranked depth axis closes with data.**

Cap 7 is the ranked depth optimum. Every open cell prices below it (5: -1.49 %,
6: -0.17 %, 8: -0.98 %). None reaches the crown line `3.7291100105909`; the
best cell is short by `+0.5735 %`. Per-prompt depth adaptivity buys exactly
zero, because the median window is set by beagle (rank 4) and essays (rank 5)
and both already sit at their own per-prompt optimum at cap 7.

Cap-8 DEAD is re-confirmed, this time on a recovered cost law rather than on
the step model that E159 falsified.

## Honest caveats

- `R(9)` is the one pure extrapolation. No observation reaches width 9, so the
  cap-8 verdict rests on the quadratic continuing one row past the data. The
  8.1 % beagle margin is the size of the error that would overturn it.
- The end-to-end pipeline reproduces a paid published score to about 0.4 %.
  The gap from the best priced cell to the crown is 0.5735 %. These are the
  same order. The pricing is strong enough to rank cells against each other and
  to say no cell is a crown candidate, but it is not precise enough to certify
  a 0.5 % crown gap by itself.
- The survival reconstruction assumes a two-parameter discrete Weibull family.
  The family-free monotone bounds move the priced cells by less than the noise
  floor, but the point estimates carry the family.
- Beagle's marginal acceptance `0.8417` slightly exceeds its cap-4 average
  `0.8315`. A strictly cap-invariant monotone per-position acceptance profile
  forbids that, so the depth controller shows mild positive selection: rounds
  that reach deeper are slightly better than average. The effect is small and
  does not change any ranking here, but it means the mixture model is an
  approximation, not an identity.
- Receipts B (`180db842`, 3.70465399) and D (`2c885d64`, 3.65820901) are cap-7
  replicates and set the candidate-leg 1-sigma at 0.189 %. All the cell spreads
  above except cap 5 are inside two sigma of each other, so the ordering of
  caps 6, 7 and 8 is a ranking claim, not a separation claim.

## Suggested follow-ups (not implemented)

- The cheapest ranked measurement that separates the surviving law from its
  rivals is **one more single-factor receipt at cap 5 or cap 6**. That pins the
  interior marginal row cost, where the surviving and rival laws disagree most
  while both still fit A and C. A cap-8 receipt would instead test the single
  extrapolated point `R(9)`, which is the only place the cap-8 verdict can
  break.
- The convex law implies the lever is **width reduction, not width extension**.
  Marginal row cost rises from 0.79 ms at width 2 to 8.85 ms at width 9. Any
  mechanism that keeps accepted tokens while narrowing the verified row count
  (for example verifying a pruned draft set, or splitting one wide verify into
  two cheaper ones) is priced far more favourably than any mechanism that adds
  depth. Cap 7 being optimal is a statement about this cost curve, not about
  draft quality.
- Plutarch and drama are cap-inert at both 4 and 7. Their round cost is near
  the batch-1 floor, so they contribute almost nothing to depth-axis variance
  and can be excluded from future depth screens to reduce ranked cost.
