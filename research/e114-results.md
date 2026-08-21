# E114 — the standing kernel weights are local, and the ranked ones are not identified

Student `qwen-askeladd`. PR #116. Assignment
`qwen38-r1-e114-every-kernel-arm-is-weighted-at-the-wrong-operating-point`, r1.
Base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`, rebased onto advisor base
`3eef86eafd38eae6a59c24d6117751e169cfcd38`.

**Verdict, as agreed in advisor feedback f2: non-identification is the result,
not a caveat on the result. The ranked NA weight vector is not point identified
by anything Yukon publishes. The 0.05 pp kill rule is AMENDED, not applied. E114
delivers a proven identified-set bound for every arm the ledger can price,
twelve arms whose SIGN is identified across that whole set, one arm whose sign
is not, and the exact artefact that would close the remaining four degrees of
freedom.**

The guaranteed lower bound on every arm's re-weighting is `0.0000 pp`. The
standing vector lies inside the identified set, so no arm's move can be proven
non-zero from the receipts alone. Every point estimate in this document is
conditional on a width shape that the pre-registered gate REJECTED.

This is an analysis-only result. No Swift source, kernel, test, or fixture file
was changed. `senpai/verify-ranked-score-boundary.sh` passes at this base.

## Identity tuple

| field | value |
|---|---|
| base | `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` |
| rebased onto | `3eef86eafd38eae6a59c24d6117751e169cfcd38` |
| candidate diff | none; `research/` only |
| harness | **every table below is labelled**; two rate frames are reported side by side |
| ranked receipt priced | `b8b8b860` (official median 3.33412148) |
| ranked receipts cross-checked | `44559d02`, promoted crown `51b9bf85` — per-prompt fields identical to `b8b8b860`, so the schedule is deterministic |
| GT1 | E106 census, W&B `19kgn6xi`, 19 rounds, mean width 6.9474 |
| GT2 | `research/out/e109-witness-w512/report.json`, 77 rounds, 512 tokens, mean width 7.3766, byte-identical across 17 replicates |
| GT3 | `research/e99-artifacts/rung67.json` null-off legs, 78 rounds, `g1_share` 0.0641 |
| host | `apple-m4-pro-applegpu_g16s-20core-48gib` |
| GPU time | none. No timed leg was run |
| gate flags | not applicable; no timed measurement. `timing_valid=false`, `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false` |

Artifacts: `research/scoring_weights.py` (the rung-2 instrument),
`research/scoring_weights_selftest.py` (17/17 pass),
`research/e114_width_recovery.py` (rung 0),
`research/e114_policy_sim.py` (rung 1b, the shipped-schedule generator),
`research/e114_rerank.py` (rung 1),
`research/e114_wandb_log.py` and `research/e114_wandb_refs.py` (publication),
`research/e114-artifacts/rung0.{json,txt}`,
`research/e114-artifacts/rung1b.{json,txt}`,
`research/e114-artifacts/rung1.{json,txt}`,
`research/e114-artifacts/selftest.txt`.

Campaign rule 40: every committed script's own input trace is committed beside
it under `research/e114-artifacts/`. `rung1.txt` reads `rung1b.json`, and
`rung1b.json` records the exact policy constants it transcribed, so the whole
chain replays from this checkout plus the public board.

W&B run `4y8mqcav`,
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>.
Run `cua0ege5` in the same project published the r1 tables before advisor
feedback f2 supplied the missing ledger cells; it is superseded and its arm
numbers must not be quoted. `mwr9kshr`, `x02hdj4v` and `u67arif0` are failed
logging attempts that crashed on W&B table typing before any deliverable table
was written.

Reproduce:

```bash
python3 research/board_per_prompt.py fetch
python3 research/scoring_weights_selftest.py
PYTHONPATH=research python3 research/e114_width_recovery.py
PYTHONPATH=research python3 research/e114_policy_sim.py
PYTHONPATH=research python3 research/e114_rerank.py
python3 research/e114_wandb_log.py          # prints the new run id
python3 research/e114_wandb_refs.py <run>   # re-points every table reference
```

## Report shape

**Question.** Every kernel arm in the ledger is summarised with one weighted
percentage built from the NA weights `0.024/0.275/0.667/0.034`. Those weights
were derived from a 19-round LOCAL trace. Is the published operating point
different enough to change which arm wins?

**Evidence that made it worth testing.** The provenance claim is exactly true
and is now proven rather than asserted. Section "Where the standing weights
come from" below reproduces `0.024/0.275/0.667/0.034` to four decimals from the
E106 local histogram alone. Nothing ranked enters it.

**Expected result.** The assignment predicted the published weights would put
much less mass on NA=4 and much more on NA=2 and NA=5, and that at least one
arm would move by more than 0.05 pp.

**Smallest decisive test.** Recover the ranked per-prompt verify-width
distribution from the published per-prompt receipt fields, validate the
recovery against three traced histograms already in the repository, then
re-price every arm whose per-NA cells are recorded. Zero GPU cost.

**Stop or promotion rule.** Pre-registered before any recovery was run, in
<https://github.com/morganmcg1/qwen38-challenge_senpai/pull/116#issuecomment-5375970810>.
Rung 0 ends as a negative result if the recovery misses GT2 by more than 0.05,
GT1 by more than 0.10, or GT3 by more than 0.08 in any NA cell. Calibration may
use only `(mean draft length, accept, R0/R)`. The shape is never fitted to the
ground truth. Rung 1 stops if no arm moves by more than 0.05 pp.

**The rung-1 stop rule was AMENDED by advisor feedback f2 before this result was
submitted.** Rung 0 showed the weight vector is a set, not a point, so a scalar
"did an arm move" test is not well defined: it can only be evaluated at a shape
that the rung-0 gate rejected. The amended deliverable is the identified-set
bound per arm plus the sign-invariant list. The 0.05 pp scalar is still reported
so the original pre-registration closes on the record.

## What was withdrawn, and what replaced it

Two planned routes died and their failures are part of the result.

1. **Route A is gone.** The assignment expected per-round width sequences in
   `research/e99-artifacts/*.json`. Those files hold aggregated tables only.
   The advisor confirmed this in feedback f1 and withdrew the route.
2. **The iid substitute failed.** The pre-registered fallback modelled
   acceptance as iid per position and predicted the width distribution from
   `(mean draft length, accept)`. It cannot reproduce the ranked pairs: beagle
   sits at `(4.382, 0.834)`, which is far off the iid curve for any single
   acceptance probability. The pre-registration allowed this model and it
   failed, so the width recovery below uses only the linear constraints that
   the receipts imply, plus the ranked cost curve.

The consequence is the central finding: without a traced ranked width sequence
the weight vector is a SET, not a point.

## Where the standing weights come from — `harness=local`

The E106 traced local histogram over verify width `M` is
`{2:1, 5:1, 6:4, 7:3, 8:10}`, 19 rounds, mean width 6.947. Mapping each width
through the live QMV partition table and weighting each resulting group by
`1 / rate(NA)` with the LOCAL one-group rates gives:

| source | NA2 | NA3 | NA4 | NA5 |
|---|---|---|---|---|
| standing rule as quoted in the ledger | 0.024 | 0.275 | 0.667 | 0.034 |
| reproduced from E106 alone, `harness=local` | 0.0242 | 0.2748 | 0.6667 | 0.0343 |
| the same histogram, group COUNTS only (rate free) | 0.0278 | 0.3056 | 0.6389 | 0.0278 |

The standing rule is therefore the local operating point exactly, and the
hypothesis's provenance claim is confirmed. `scoring_weights_selftest.py` test 1
fails if this stops reproducing.

## Rung 0 — recovering the ranked operating point

### The board does not publish a histogram

The Yukon per-prompt record contains `effective_mean_draft_len`,
`non_drafting_round_count`, `mtp_seconds_per_token_mean`, `raw_ratio_of_means`,
`serial_seconds_per_token_mean`, `prefill_seconds_per_token`, `parity_ok`,
`prompt_sha256`, `head_provenance_sha256`, `accepted_pair_count` and
`noop_reference_decode_speedup`. There is no width histogram in any of the 1039
rows on the board.

### Round counts are independently confirmed — `harness=ranked`

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung0/round_count_admissibility`._

Token conservation alone leaves several admissible round counts per prompt.
Pricing each candidate against the E113 route-B ranked cost curve selects the
Finding 18b value on **all eight** prompts, with residuals from −0.06 % to
+0.68 %. Drama is the informative case: conservation allows 168 as well as 252,
and the cost curve picks 252.

| prompt | chosen R | admissible R | curve pick | residual |
|---|---|---|---|---|
| plutarch | 487 | 487 | 487 | −0.06 % |
| drama | 252 | 168, 252, 336, 420, 504 | 252 | −0.57 % |
| travel | 212 | 212, 424 | 212 | +0.47 % |
| beagle | 110 | 110, 165, 220, … | 110 | −0.25 % |
| republic | 93 | 93, 186, 279, … | 93 | −0.21 % |
| essays | 92 | 92, 115, 138, … | 92 | +0.68 % |
| medicine | 90 | 90, 180, 270, … | 90 | −0.11 % |
| botany | 81 | 81, 108, 135, … | 81 | −0.11 % |

### Ranked mean verify widths — `harness=ranked`

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung0/ranked_board_inputs`._

| prompt | mean width | P(M=1) |
|---|---|---|
| plutarch | 1.154 | 0.922 |
| drama | 3.298 | 0 |
| travel | 3.656 | 0 |
| beagle | 5.382 | 0 |
| republic | 5.989 | 0 |
| essays | 6.087 | 0 |
| medicine | 6.256 | 0 |
| botany | 7.148 | 0 |

Only plutarch spends round time at the narrow width, and it spends almost all of
it there.

### The pre-registered validation gate — FAILED for both point estimators

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung0/validation_against_traced_truth`._

Two point estimators were tried: a maximum-entropy distribution on the mean
constraint, and an exponential tilt (I-projection) of a traced local histogram
onto the ranked mean. Both are legal under the pre-registration because neither
fits the shape to the ground truth it is scored against.

`harness=ranked` rate table, ranked widths:

| ground truth | estimator | max abs err | tol | verdict |
|---|---|---|---|---|
| GT1 | maxent | 0.0214 | 0.10 | PASS |
| GT1 | tilt transport | 0.1315 | 0.10 | **FAIL** |
| GT2 | maxent | 0.0899 | 0.05 | **FAIL** |
| GT2 | tilt transport | 0.1030 | 0.05 | **FAIL** |
| GT3 | maxent | 0.0041 | 0.08 | PASS |
| GT3 | tilt transport | 0.0194 | 0.08 | PASS |
| GT1, GT2, GT3 | identified BOUND | — | coverage | **PASS on all three** |

The `harness=local` rate frame gives the same verdicts (GT1 maxent 0.0218, GT2
maxent 0.0868, GT1 transport 0.1274, GT2 transport 0.0994).

**The pre-registered rule is honoured: both point estimators are reported as
failures and no headline is built on either of them.** The bound passes its
coverage check on all three traced histograms, so the bound is the object E114
can defend.

### Why the vector is not identified

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung0/recovered_na_weights_by_prompt`._

The unknown is a distribution over `M = 1..8`, which is seven free parameters
after normalisation. The receipts impose three linear equalities per prompt:
normalisation, the mean draft length, and the non-drafting round share. Four
dimensions are left free. Intersecting with the ranked cost curve at its own
0.68 % fit residual barely helps, because the cost curve is nearly linear in `M`
over the relevant range and so is almost the same functional as the mean.

The unconstrained per-NA bands at the score-carrying prompts are 0.95 to 1.00
wide. Beagle is the worst: its band is exactly 1.000, and the distribution
`{2: 0.227, 4: 0.314, 8: 0.459}` matches its published mean AND its published
round cost while putting no mass at all on the widths the point estimators
prefer.

Per-prompt recovered weights are in
`research/e114-artifacts/rung0.json` under `recovery`, in four frames
(`local`/`ranked` rate table × `mean_only`/`with_cost` constraints).

## Rung 1b — can the SHIPPED SCHEDULER close the four free dimensions?

Advisor feedback f2, item 4. Rung 0 leaves the weight vector with four free
dimensions because the receipts imply only three equalities. The scheduler that
produced those receipts is in this checkout, so it is a source of information
the equalities do not contain. If simulating `costModelDepth` reproduces the
ranked per-prompt moments, its emitted width histogram is a fourth shape derived
from the mechanism rather than from a maximum-entropy convention.

The policy is transcribed verbatim from
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`: max depth 8,
`segmentedVerifyDepthCap` 7 (SRC:1009), `headStepCostRatio` 0.18 (SRC:841),
EMA α 0.15, EMA prior `0.85 · 0.98^i`, optimism cap 0.95 (SRC:1167), uniform
price so cumulative cost is `1 + 0.18 i`. Acceptance uses the E92 measured
per-position profile, `harness=local`. 300 Monte Carlo windows per evaluation,
seed 20260817.

### Model A, the advisor's one-parameter prescription, is FALSIFIED

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung1b/policy_fit_per_prompt`._

Model A scales the acceptance profile by one per-prompt level λ and switches
both confidence overrides off, then fits λ to the board's
`effective_mean_draft_len` alone. It matches that moment by construction and
then fails everything else:

| prompt | board accept | model A accept | round-count residual |
|---|---|---|---|
| beagle | 0.8340 | 0.5558 | +36.01 % |
| essays | 0.8974 | 0.6062 | +36.86 % |
| medicine | 0.8922 | 0.6173 | +34.64 % |
| republic | 0.9030 | 0.5971 | +38.80 % |
| botany | 0.8655 | 0.6983 | +20.03 % |

One parameter cannot carry both moments. Fitting the mean draft length forces an
acceptance level far below the board's, and the round count is then wrong by a
third. **The one-parameter model is rejected.**

### Model B adds one parameter and fits both moments

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung1b/policy_fit_per_prompt`._

Model B restores the two confidence overrides through a single fitted
exponential mean top-2 margin µ, so it has two parameters (λ, µ) against two
moments. It fits:

| prompt | λ | µ | board d̄ | sim d̄ | board accept | sim accept | round resid |
|---|---|---|---|---|---|---|---|
| beagle | 0.9761 | 0.902 | 4.3818 | 4.3865 | 0.8340 | 0.8331 | +0.45 % |
| botany | 1.0001 | 4.432 | 6.1481 | 6.1428 | 0.8655 | 0.8641 | +0.83 % |
| essays | 1.0069 | 0.946 | 5.0870 | 5.0929 | 0.8974 | 0.8973 | +0.55 % |
| medicine | 1.0056 | 1.166 | 5.2556 | 5.2417 | 0.8922 | 0.8922 | +0.75 % |
| republic | 1.0087 | 0.829 | 4.9892 | 4.9902 | 0.9030 | 0.9023 | +0.65 % |

Round counts land within ±0.86 % on every scored prompt, and the fitted λ sits
within 2.4 % of 1.0 on four of the five. That is a genuine result: **the E92
local acceptance profile, at essentially its measured level, plus one margin
parameter, reproduces the ranked schedule's two published moments and its round
count.** The ranked prompts are not accepting at a different rate from the local
trace; they are accepting at almost exactly the same rate.

### But model B FAILS its own pre-registered held-out gate

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, tables `rung1b/held_out_width_census_validation`, `rung1b/policy_shape_against_rung0_band`._

Fitting two moments per prompt is in-sample by construction, so it cannot
license a published weight vector. The gate, registered before the fits were
run, is held-out: reproduce the width censuses this experiment did NOT fit, at a
total-variation distance of at most 0.10, and land every scored prompt inside
the rung-0 cost band.

| target | model A TVD | model B TVD |
|---|---|---|
| E106 census (19 rounds) | 0.2058 | 0.1603 |
| E109 witness (77 rounds) | 0.5151 | 0.5435 |
| v2 pair census (31 rounds) | 0.1188 | — |

| model | worst OOS TVD | scored prompts out of band | verdict |
|---|---|---|---|
| A | 0.5028 (E109 witness) | essays | **FAIL** |
| B | 0.5494 (E109 witness) | beagle, essays | **FAIL** |

Both models miss the held-out histograms badly. The E109 witness trace puts 73 %
of its rounds at exactly M=7 and none at M=8; model B spreads that mass from M=3
to M=8. Two prompts also leave the rung-0 cost band: essays by −1.28 % and
beagle by −0.72 % against a ±0.68 % band.

**The essays miss is a genuine contradiction, not noise.** The policy shape and
the cost band are both derived from the same receipt fields, so a shape that
reproduces the board's mean draft length and accept rate should price inside the
band that reproduces the board's round time. It does not. Either the shipped
policy is not what generated those receipts, or the cost curve `ROUTE_B` is
mis-specified at high width, or both. This is now the most informative open
question E114 leaves behind.

### What rung 1b is worth

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung1b/structural_gates`._

It does NOT close the four free dimensions. The `policy` column in every table
below is a **diagnostic**: it demonstrates that a shape built from the actual
mechanism lands inside the identified set the equalities imply, which is a
useful consistency check on rung 0, and nothing more. It is never used to narrow
a bound or to overturn a sign.

Two structural gates limit any conclusion from this rung. Both confidence
overrides are bounded below by 0.5 because a top-2 margin cannot be negative, so
neither model can produce a non-drafting round except by driving
`positionAcceptEMA[0]` under 0.18. Plutarch, whose board mean width is 1.154
with `P(M=1) = 0.922`, is therefore out of scope by construction, and its round
count misses by −4.86 %. Acceptance is also drawn independently per position
within a round, which rung 0 already showed the board rejects as a complete
model.

## Rung 1 — re-ranking the arms

### The published operating point weights prompts, not rounds

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung1/published_score_sensitivity_mix`._

Finding 16: `published = 0.5·raw_beagle + 0.5·min(essays, medicine, republic,
botany)`, and rank-5 occupancy over 81 strong runs is essays 0.667, medicine
0.198, republic 0.074, botany 0.062. Because `raw_p` is inversely proportional
to prompt `p`'s candidate seconds per token, a relative change `dp` moves the
published median by `−0.5·raw_p·dp / published`. Converting to absolute
microseconds gives the score-sensitivity mix:

| prompt | share of published sensitivity |
|---|---|
| beagle | 0.457 |
| essays | 0.359 |
| medicine | 0.109 |
| republic | 0.041 |
| botany | 0.035 |

Plutarch, drama and travel carry **zero** published weight. They never occupy
rank 4 or rank 5. Every arm summary that averaged over all eight prompts was
spending effort on prompts that cannot move the score.

### Provenance check — the instrument reproduces the ledger's own column

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung1/provenance_reproduce_ledger_column`._

Advisor feedback f2 supplied the authoritative per-NA cells for the ledger's
259.8 arm table. Before any re-weighting is meaningful, the instrument must
reproduce the ledger's own `weighted %` column from those cells and the standing
weights. It does, on all nine arms that carry a recorded reduction:

| arm | recorded `weighted %` | rebuilt | residual pp |
|---|---|---|---|
| `xs_stage` | −1.458 | −1.454 | +0.0041 |
| `xv4` | −0.673 | −0.670 | +0.0035 |
| `mo_swap` | +153.341 | +153.344 | +0.0033 |
| `mo_stage` | +151.277 | +151.275 | −0.0022 |
| `b_barrier` | +0.274 | +0.276 | +0.0020 |
| `b_constw` (E110) | −86.868 | −86.870 | −0.0015 |
| `xv8` | +7.975 | +7.974 | −0.0013 |
| `xv4_stage` | −1.859 | −1.858 | +0.0008 |
| `l_loadonly` | −14.578 | −14.577 | +0.0005 |

Worst residual 0.0041 pp against a 0.005 pp bar — **PASS**. The residual is
rounding in the quoted cells, not a different weighting rule.

### The weight vectors — `harness=ranked` widths, both rate frames

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `deliverable/na_weight_vectors`._

Rate table `harness=ranked`:

| vector | NA2 | NA3 | NA4 | NA5 |
|---|---|---|---|---|
| STANDING (`harness=local`) | 0.0240 | 0.2750 | 0.6670 | 0.0340 |
| published / maxent | 0.0398 | 0.3483 | 0.5107 | 0.1011 |
| published / gt1 tilt | 0.1158 | 0.3421 | 0.4792 | 0.0628 |
| published / gt2 tilt | 0.0000 | 0.2478 | 0.5696 | 0.1826 |
| published / policy — DIAGNOSTIC, gate FAILED | 0.0000 | 0.3346 | 0.4999 | 0.1655 |

Rate table `harness=local`, same ranked widths:

| vector | NA2 | NA3 | NA4 | NA5 |
|---|---|---|---|---|
| published / maxent | 0.0416 | 0.3377 | 0.5211 | 0.0996 |
| published / gt1 tilt | 0.1206 | 0.3305 | 0.4872 | 0.0617 |
| published / gt2 tilt | 0.0000 | 0.2399 | 0.5805 | 0.1796 |
| published / policy — DIAGNOSTIC, gate FAILED | 0.0000 | 0.3252 | 0.5114 | 0.1634 |

The direction the assignment predicted is right in every shape: NA=4 loses 9 to
19 points of weight and NA=5 gains. The magnitude is not identified. The
`policy` row comes from rung 1b below; it is shown because a shape derived from
the shipped scheduler independently lands inside the same set, but it FAILED its
own held-out gate and must not be used to narrow anything.

### The deliverable an arm owner can apply without this script

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `deliverable/na_weight_vectors`._

`move_pp = Σ_NA dW[NA] · arm_pct[NA]`, `harness=ranked`:

| shape | dW NA2 | dW NA3 | dW NA4 | dW NA5 |
|---|---|---|---|---|
| maxent | +0.0158 | +0.0733 | −0.1563 | +0.0671 |
| gt1 tilt | +0.0918 | +0.0671 | −0.1878 | +0.0288 |
| gt2 tilt | −0.0240 | −0.0272 | −0.0974 | +0.1486 |
| policy (diagnostic) | −0.0240 | +0.0596 | −0.1671 | +0.1315 |

**Applying `dW` gives a point, not a bound.** An arm owner who wants the honest
answer must use the identified range in the next table.

### The arm table — `harness=ranked` rate frame

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `deliverable/arm_rerank`._

Sign convention as recorded: positive is SLOWER than `a_base`. The range is the
exact extremum of the published-weighted value over every width distribution
consistent with the receipts, computed by Dinkelbach iteration over the vertex
set of the identified polytope. A flat arm is used as a positive control and
prices at exactly `[7.000000, 7.000000]`.

| arm | source | standing % | identified range | maxent / gt1 / gt2 / policy | max move pp | sign identified |
|---|---|---|---|---|---|---|
| `mo_swap` | E110 | +153.344 | [+128.458, +175.299] | +149.847 / +141.514 / +159.098 / +155.517 | 24.886 | yes |
| `mo_stage` | E110 | +151.275 | [+127.861, +171.774] | +147.905 / +139.825 / +156.711 / +153.348 | 23.414 | yes |
| `xv8` | E110 | +7.974 | [+4.845, +29.174] | +10.735 / +8.908 / +13.472 / +13.256 | 21.200 | yes |
| `f44_roofline_gap` | Finding 44 | +17.272 | [+6.732, +33.900] | +17.212 / +15.201 / +21.066 / +19.408 | 16.628 | yes |
| `mu_swap` | E110 | +78.619 | [+69.914, +93.280] | +77.520 / +70.753 / +83.239 / +81.960 | 14.660 | yes |
| `mo_hoist` | E110 | +21.752 | [+11.480, +124.932] | +32.153 / +24.571 / +47.126 / +43.651 | 103.180 | yes |
| `l_loadonly` | E110 | −14.577 | [−25.020, −6.857] | −14.304 / −12.867 / −16.941 / −15.762 | 10.443 | yes |
| `b_constw_e111` | E111 | −0.602 | [−1.418, +1.864] | +0.105 / +1.113 / −0.675 / −0.373 | 2.466 | **no — SIGN FLIPS** |
| `xv4_stage` | E110 | −1.858 | [−2.399, −0.940] | −1.721 / −1.524 / −1.981 / −1.855 | 0.919 | yes |
| `xs_stage` | E110 | −1.454 | [−2.374, −0.861] | −1.427 / −1.236 / −1.682 / −1.588 | 0.920 | yes |
| `b_constw_e110` | E110 | −86.870 | [−87.776, −86.242] | −87.038 / −87.412 / −86.683 / −86.809 | 0.907 | yes |
| `xv4` | E110 | −0.670 | [−1.557, −0.404] | −0.718 / −0.615 / −0.887 / −0.835 | 0.887 | yes |
| `b_barrier` | E110 | +0.276 | [+0.105, +0.345] | +0.233 / +0.151 / +0.300 / +0.278 | 0.171 | yes |

`mo_hoist`'s range is asymmetric because its recorded NA=5 cell is `+185.29`
against `+2.07` at NA=2. Any shape that moves mass to NA=5 moves that arm a very
long way. It is the clearest single demonstration that the operating point
matters: `mo_hoist` is the arm whose ranked cost is least knowable from the
ledger's one number.

### The primary metric

`e114_max_abs_arm_reweight_pp` = **10.4011 pp**, the largest absolute difference
between the published-weighted and the standing-weighted value over the E110 arm
table at the maxent shape, `harness=ranked` rate frame. The 0.05 pp kill rule is
**amended, not applied**: a scalar computed at a rejected shape cannot kill or
save the hypothesis. It is reported so the r1 pre-registration closes cleanly.

| quantity | value |
|---|---|
| primary metric, maxent point | 10.4011 pp |
| shape ensemble spread | −11.8306 to +25.3743 pp |
| identified-set upper bound | 103.1804 pp |
| **guaranteed lower bound** | **0.0000 pp** |

The guaranteed lower bound is zero and this is the honest headline. The standing
vector lies INSIDE the identified set for every one of the thirteen arms, so no
single arm's re-weighting can be proven non-zero from the receipts alone. Read
the metric as "plausibly around ten points, provably at most 103, provably at
least 0".

### What IS decisive — the sign-invariant list

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `deliverable/sign_invariance_verdicts`._

Twelve of the thirteen arms keep their sign at every admissible operating point.
For these the ledger's qualitative verdict survives re-weighting even though its
magnitude does not:

| arm | identified move band pp | invariant conclusion |
|---|---|---|
| `mo_swap` | [−24.887, +21.954] | slower, large |
| `mo_stage` | [−23.414, +20.500] | slower, large |
| `xv8` | [−3.129, +21.200] | slower |
| `f44_roofline_gap` | [−10.539, +16.628] | slower |
| `mu_swap` | [−8.705, +14.660] | slower, large |
| `mo_hoist` | [−10.272, +103.180] | slower |
| `l_loadonly` | [−10.443, +7.720] | faster, large |
| `xs_stage` | [−0.920, +0.593] | faster |
| `xv4_stage` | [−0.541, +0.919] | faster |
| `b_constw_e110` | [−0.907, +0.627] | faster, very large |
| **`xv4`** | **[−0.887, +0.266]** | **faster** |
| `b_barrier` | [−0.171, +0.069] | slower, small |

`xv4` is on this list at the advisor's explicit request in f2. It prices in
`[−1.557, −0.404]` and never crosses zero, so the recorded `−0.673` keeps its
sign at every admissible operating point. Its maximum re-weighting is 0.887 pp,
which is more than the 0.1 pp threshold the advisor set for an immediate report;
that report was posted at
<https://github.com/morganmcg1/qwen38-challenge_senpai/pull/116#issuecomment-5376158714>
when the cells were still missing, and it is now superseded by this row.

One arm is NOT sign identified: `b_constw_e111` prices in `[−1.418, +1.864]`, so
it changes sign inside the identified set. No conclusion about it can survive
without the missing trace, and its withdrawal was correct on these grounds too.

### Reconciliation against the advisor's own reweighting

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung1/reconciliation_against_advisor`._

Feedback f2 included the advisor's independent reweighting of seven arms. The
point estimates agree to **0.0207 pp worst case against a 0.05 pp bar — PASS**.
Two instruments built from the same cells now give the same three points.

| arm | Δ chk | Δ maxent | Δ gt1 | Δ gt2 | advisor 3-shape spread | my identified set |
|---|---|---|---|---|---|---|
| `b_barrier` | +0.000 | +0.002 | +0.002 | +0.002 | 0.125 | 0.171 |
| `xs_stage` | +0.000 | +0.004 | +0.004 | +0.004 | 0.228 | 0.920 |
| `xv4` | +0.000 | +0.003 | +0.003 | +0.003 | 0.217 | 0.887 |
| `xv4_stage` | −0.000 | +0.001 | +0.000 | +0.001 | 0.335 | 0.919 |
| `xv8` | −0.000 | −0.001 | +0.001 | −0.002 | 5.499 | 21.200 |
| `l_loadonly` | −0.000 | −0.001 | −0.001 | +0.001 | 2.364 | 10.443 |
| `mo_swap` | +0.000 | +0.017 | +0.021 | +0.001 | 11.848 | 24.886 |

The last two columns measure **different quantities** and must not be read as a
disagreement. The advisor's `spread` is the maximum over three named shapes; the
identified set is the maximum over every shape consistent with the receipts. The
set is up to **4.4× wider**. This is exactly why the deliverable is the set and
not the spread: three shapes chosen for being reasonable understate the true
uncertainty by a factor of four.

### The `weighted % → round %` factor moves too

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `deliverable/weighted_to_round_factor`._

The ledger converts a weighted percentage to a round percentage with the fixed
scalar `0.7788`. That scalar is not a constant — it is the M=5 POINT of a
function of the width distribution. Rebuilding it from the E106 local round
times and the local rate table, `harness=local` on both, with `harness=ranked`
widths:

| basis | factor | extrapolated mass |
|---|---|---|
| ledger scalar as quoted | 0.7788 | — |
| M=5 point, rebuilt | 0.7836 (+0.62 % vs ledger) | 0 % |
| E106 realised widths | 0.8260 | 89 % |
| published / maxent | 0.8176 | 61 % |
| published / gt1 | 0.8215 | 70 % |
| published / gt2 | 0.8125 | 50 % |
| published / policy (diagnostic) | 0.8146 | 54 % |
| **identified set** | **[0.7994, 0.8292]** | — |

Rebuilding the ledger's own scalar at its own M=5 point reproduces it to 0.62 %,
which confirms the model. At the published operating point the factor is 4 to
6 % HIGHER than 0.7788, so every ledger round percentage is slightly
conservative in magnitude. The direction is uniform across the whole identified
set, so this correction is sign invariant even though its size is not.

`extrapolated mass` is the share of the width distribution priced with a round
time that E106 did NOT measure. Only M=2..5 were measured; M=6, 7 and 8 use a
least-squares overhead fit (intercept 9780.52 µs, slope 2330.63 µs per width).
Between half and nine tenths of the mass at the published operating point sits
on that extrapolation, so treat the factor as directional.

### Arms that cannot be re-weighted at all

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `rung1/arms_without_per_na_cells`._

Eight arms the assignment named still have no recorded per-NA cells anywhere in
this checkout: `c_loadonly`, `n_nobias`, `n_nosums`, `g_pack32`, `d_bias1` and
`e_bias6` (E111 reduced them to one weighted percentage; the per-NA cells live
in a host-local `research/out/<tag>/arms.json` on thorfinn's Mac),
`h_prunenarrow` and `i_pruneall` (E108 reported one pooled percentage). For
`n_nosums` the single cell NA=5 = `+9.93 %` is recorded, which is not enough to
weight it.

`xv4`, `xv4_stage`, `xv8`, `mu_swap` and `mo_hoist` were on this list in r1 and
came off it when advisor feedback f2 supplied their cells.

For `e_bias6` the recorded relation `round % = 0.877 × weighted %` fixes the
ratio but not the four cells, so it does not unblock a re-weighting either.

## Item 5 — is E100's M=5 collapse worth its register tax?

Requested by advisor feedback f1 and exempt from the kill rule.

### Share of published-weighted candidate TIME at exactly verify width 5

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `item5/m5_share_of_candidate_time`._

`harness=ranked`:

| prompt | maxent | gt1 | gt2 | policy (diag.) | identified set |
|---|---|---|---|---|---|
| beagle | 0.1331 | 0.0821 | 0.2348 | 0.2271 | [0.0000, 0.8298] |
| essays | 0.1058 | 0.0684 | 0.1835 | 0.1604 | [0.0000, 0.5559] |
| medicine | 0.0975 | 0.0642 | 0.1690 | 0.1450 | [0.0000, 0.4970] |
| republic | 0.1104 | 0.0707 | 0.1916 | 0.1726 | [0.0000, 0.5911] |
| botany | 0.0429 | 0.0330 | 0.0830 | 0.0621 | [0.0000, 0.2200] |
| **COMBINED, published weighting** | **0.1154** | **0.0731** | **0.2022** | **0.1863** | **[0.0000, 0.6645]** |

The identified set spans nearly the whole unit interval. The share of ranked
candidate time spent at exactly verify width 5 is one of the least identified
quantities in this document, and item 5 multiplies by it.

### The arithmetic

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `item5/collapse_net_of_register_tax`._

The register tax of keeping the `<T,5,5,true>` instantiation: E108's
`i_minus_case5` moved g17s registers 98 → 91 and `S_ranked` 40 → 43. The E77 law
predicts +0.0974 %; E102 measured +0.1068 % but only on the G=1 prompts, and all
five score-carrying prompts here are G=2, so the measured figure is transferred
outside the population it was measured in. Both are reported.

Net = `gain × share − tax`, as a percentage of candidate time:

| collapse price | kind | tax | net point | net band |
|---|---|---|---|---|
| E104 Finding 33, +1.9 % | per round | E77 law | **+0.1218** | [−0.0974, +1.1651] |
| E104 Finding 33, +1.9 % | per round | E102 measured | +0.1124 | [−0.1068, +1.1557] |
| Finding 32 route 1, −2.0 % | per round | E77 law | **−0.3282** | **[−1.4264, −0.0974]** |
| Finding 32 route 1, −2.0 % | per round | E102 measured | −0.3376 | [−1.4358, −0.1068] |
| Finding 32 route 2, −0.070 ± 0.360 % leg | **leg effect** | E77 law | **−0.1674** | [−0.5274, +0.1926] |
| Finding 32 route 2, −0.070 ± 0.360 % leg | **leg effect** | E102 measured | −0.1768 | [−0.5368, +0.1832] |

Route 2 is handled differently from the other two, and this corrects an error in
the first version of this table. `research/group_scaling.py` computes route 2 as
`gain = (measured dW / 100) / share` with `dW = −0.070 ± 0.360` and an assumed
`share = 0.24`. The MEASURED quantity is therefore the leg effect
`gain × share = −0.070 ± 0.360 %`, and the per-round price only exists after
dividing by the assumed share. Multiplying our own recovered share back in would
double count it and inflate the band roughly fourfold. The corrected route-2 band
is `[−0.527, +0.193]`, not `[−1.294, +0.700]`.

The assumed share can now be checked rather than inherited. Route 2's `share` is
the M=5 fraction of PRE-E100 two-group rounds, that is `P(M=5 | M ≥ 5)` counted
in rounds. Recovered here at the published weighting: 0.1771 / 0.1068 / 0.3143 /
0.2785 for maxent / gt1 / gt2 / policy, identified set `[0.0000, 0.7281]`. **The
0.24 assumption lies inside the identified set**, so route 2's own share is not
the problem.

### The three prices are one coefficient, and its measured range straddles zero

The three "irreconcilable" prices are not three measurements that disagree.
`research/group_scaling.py` produces every per-round price through one transform:

```text
A_ranked    = A_local × adv,   adv = 1.24369
ranked gain = 100 × (1 − A_ranked / 2) = 100 × (1 − 0.62185 × A_local)
```

They differ only in the value of `A_local` fed in:

| `A_local` | ranked gain | which price this is |
|---|---|---|
| 1.552 | +3.49 % | E104 Finding 33 lower edge of its own measured range |
| **1.577** | **+1.94 %** | **E104 Finding 33 point, quoted as +1.9 %** |
| **1.6081** | **0.00 %** | **BREAK-EVEN** |
| **1.640** | **−1.98 %** | **Finding 32 route 1, quoted as −2.0 %** |
| 1.641 | −2.04 % | E104 Finding 33 upper edge of its own measured range |

**E104's own measured range for `A_local` at M=5 is `[1.552, 1.641]`, and that
range CONTAINS the break-even 1.6081.** Route 1's `A_local = 1.640` sits one
thousandth below E104's own upper edge. The two are not in conflict at all; they
are the same coefficient read at two points of one interval, and the gain
`1 − 0.62185·A_local` is a small difference of large numbers, so a ±3 % range on
`A_local` becomes a 5.5-point swing in the gain that crosses zero.

Route 1 is also not independent evidence. Its `A_local = 1.640` is algebra on
`COLLAPSE_MEASURED = 0.180`, as `research/group_scaling.py` states in its own
output string, so it is a restatement of one local collapse measurement rather
than a second opinion about it. Ledger 260.7 records the same point.

**Answer to the advisor's question, as asked.** Under Finding 32 route 1 the net
is negative AND its band excludes zero: `−0.3282 %` in `[−1.4264, −0.0974]`. The
advisor asked to be told if that happened, and it has — but the finding is not
decisive, because route 1 is algebra on a local measurement whose own range
crosses break-even. The corrected route-2 leg effect straddles zero at
`−0.1674 %` in `[−0.5274, +0.1926]`, and E104 straddles zero at `+0.1218 %` in
`[−0.0974, +1.1651]`.

**The honest answer is that the sign of the net is undetermined, and the share
arithmetic is not what determines it.** The share is the well-measured half of
this product: it is `0.1154` with an identified set of `[0, 0.6645]`, and even
its widest value cannot flip a sign that the price already flips on its own. The
decisive missing measurement is `A_local` at M=5 with an error bar narrower than
`±0.032`, which is the distance from E104's point estimate to break-even. No
further share arithmetic can substitute for it.

### The fourth price is rejected — placebo test

_W&B run `4y8mqcav`, <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4y8mqcav>, table `item5/curve_difference_placebo`._

A fourth price is available from our own receipts: the pre-E100 and post-E100
ranked two-line refits evaluated at M=5 give 53,108 µs and 52,522 µs, a
difference of **+1.103 %**. That looks like an end-to-end price for E100.

It is not one. E100 changed the partition only at M=5. At M=6, 7 and 8 the
partition is identical across the two fits, so the same difference taken there
is a placebo:

| M | partition changed by E100 | pre-E100 µs | post-E100 µs | difference |
|---|---|---|---|---|
| **5** | **yes (treatment)** | 53,108 | 52,522 | **+1.103 %** |
| 6 | no (placebo) | 60,341 | 59,631 | +1.178 % |
| 7 | no (placebo) | 67,574 | 66,739 | +1.236 % |
| 8 | no (placebo) | 74,807 | 73,847 | +1.284 % |

The placebo is LARGER than the treatment at every unchanged width. The
difference is global drift between the two receipt populations, not the
collapse. A second check agrees: the register tax should make the post-E100 fit
about 0.10 % SLOWER at every M ≥ 6, and instead it is 1.2 % faster there. The
fit is therefore reported as `usable_as_price=false` and contributes nothing to
the item-5 answer.

Fit constants, `harness=ranked`: pre-E100 `a1=27181.5, c1=3995.1, a2=16943.2,
c2=7233.0`, break at 5. Post-E100 `a1=27439.9, c1=3799.2, a2=16982.0,
c2=7108.1`, break at 5, max absolute residual 0.68 %.

## Rung 2 — the instrument

`research/scoring_weights.py` is now the single source for every weighted
kernel number. It holds one `PARTITION` table, both rate tables labelled by
harness, the E106 histogram, the standing weights, the Finding 16 occupancy
mixture, and the functions `na_weights`, `local_weights`, `published_weights`,
`weighted`, `reweigh` and `rerank`. It has a CLI:

```bash
python3 research/scoring_weights.py --widths 4:1,6:2 --arms a=2:-1,3:0,4:+2,5:0
```

It also holds the round-time model that rebuilds the ledger's `weighted % →
round %` scalar: `WEIGHT_STREAM_GB`, the measured E106 per-width round times,
and the functions `qmv_us`, `round_us` and `qmv_share_of_round`.

`research/scoring_weights_selftest.py` passes 17/17. The tests that matter:

- **Test 1** reproduces the standing weights from their own source data. It
  fails if the rate table, the histogram or the partition table moves, which
  would make every standing-weighted number in the ledger unreproducible.
- **Test 4 is the stale-partition-table case the assignment required.** Before
  E100 the group boundary sat one width lower, so `M=5` was `[3+2]` and not
  `[5]`. A distribution concentrated at width 5 puts all its weight on NA=5
  under the live table and none at all under the stale one. The test asserts
  the two tables disagree, so it cannot silently pass if the live table is ever
  left behind by an edit to `quantized.h:1918-1979`.
- **Test 5** proves an unknown NA raises instead of being silently dropped. A
  dropped cell renormalises the other three and quietly changes every headline.
- **Test 10** proves the ranked rate table is not a rescaled copy of the local
  one. An NA=5 group costs 1.51× an NA=2 group on the ranked host against 1.42×
  locally, so the `harness=` label carries real information. This test caught a
  genuine error: the comment in `scoring_weights.py` had that comparison
  inverted, and the test failed until it was corrected.
- **Tests 11-14** cover the recovery machinery: `tilt` keeps its support, hits
  its target mean, and is the identity on its own mean; every emitted vertex is
  feasible; the maxent point lies inside the vertex hull, which is what makes
  the rung-1 range a bound at all; and both cost fits are monotone with the same
  break width, which is what makes the placebo comparison well posed.
- **Tests 15-17** cover the round-time model added for the `weighted → round`
  factor. Test 15 asserts that every width E106 actually measured comes back
  from the measurement and that M=6, 7 and 8 are marked extrapolated; if a
  measured width silently fell through to the fit, the whole factor would be an
  extrapolation and the ledger comparison would mean nothing. Test 16 asserts
  the M=5 point stays within 0.01 of the ledger's 0.7788, stays a fraction, and
  is a proper mixture of its per-width points. Test 17 asserts both rate tables
  are strictly decreasing in width, because `qmv_us` divides by them and a
  non-monotone table would make a wider round look cheaper and invert every arm
  sign downstream.

`research/e114_rerank.py` also runs a positive control on every invocation: a
flat arm must price identically under every distribution, so its range has to
collapse onto the flat value. It does, at `[7.000000, 7.000000]`, and the run
asserts on it.

## Honest limitations

1. **Both point estimators failed the pre-registered gate.** The maxent shape
   used for the primary metric missed GT2 by 0.0899 against a 0.05 tolerance,
   and the tilt estimator missed GT1 by 0.1315 and GT2 by 0.1030. Every point
   number in this document inherits that failure. Only the ranges and the
   sign-invariant list are defensible.
2. **The third estimator also failed, on a harder gate.** The rung-1b policy
   generator was the attempt to replace a convention with a mechanism. It fits
   the two ranked moments, then misses the held-out E109 witness histogram by a
   TVD of 0.5494 against a 0.10 bar and leaves two scored prompts outside the
   rung-0 cost band. Three independent shape estimators have now been tried and
   all three have been rejected by evidence already in this repository. That is
   the strongest support for the non-identification headline: the problem is not
   that the right estimator has not been found, it is that the receipts do not
   contain the information.
3. **The model class changed after the pre-registration.** The iid acceptance
   model was pre-registered and failed. Maxent and tilt were chosen afterwards.
   They were still scored against the fixed pre-registered tolerances rather
   than tolerances chosen to fit them, but they are post-hoc and are labelled
   as such here.
4. **One receipt was priced.** `b8b8b860` was rejected at submission. Its
   per-prompt fields are identical to `44559d02` and to the promoted crown
   `51b9bf85`, which is what licenses its use, but no independent ranked
   population was available.
5. **The E102 register tax is transferred across populations.** It was measured
   on G=1 prompts and all five score-carrying prompts are G=2. The E77 law
   figure is reported beside it for that reason.
6. **The arm tables are local measurements.** Only the WEIGHTS are ranked. A
   per-NA percentage measured on g16s is not a ranked per-NA percentage, and
   re-weighting it does not make it one. This document re-prices the recorded
   arms at the ranked operating point; it does not claim they would measure the
   same on M5.
7. **GT1 is 19 rounds.** It is small, and it is the ground truth the maxent
   estimator passes most easily.

## Suggested follow-ups, not implemented

1. **The one artefact that closes this.** A single ranked-shaped 512-token trace
   that records the verify width of every round would collapse four free
   dimensions to zero and turn every range in this document into a number. The
   cheapest version is a local 512-token run at the ranked mean widths with the
   per-round width logged; it would not be ranked evidence, but it would fix
   the SHAPE, which is the only thing missing. Thorfinn's `e101ctl512` trace is
   exactly this object and it is host-local to his Mac; committing such a trace
   as a small artefact would unblock this and any future weighting question.
2. **Settle the essays cost-band contradiction.** Rung 1b produces a width
   shape that matches essays' published mean draft length and accept rate to
   four decimals, yet prices 1.28 % away from the round time the same receipt
   publishes, against a ±0.68 % band. Both objects come from the same receipt,
   so one of the two models is wrong. The cheapest discriminator is to extend
   the `ROUTE_B` cost fit with a measured M=6, 7 and 8 round time; the current
   fit is a two-line extrapolation above M=5 and carries between half and nine
   tenths of the published mass. If the cost curve is the error, the rung-0
   cost band widens and several identified ranges shrink; if the policy is the
   error, the shipped scheduler is not what produced these receipts, which is a
   much more important finding.
3. **Record per-NA cells in the ledger for E111 and E108.** Eight arms are
   unrankable purely because their per-NA cells were reduced to a single
   weighted percentage before being recorded. The cost of keeping four numbers
   instead of one is nil, and advisor feedback f2 proved the point by supplying
   the E110 cells and immediately unblocking five arms that r1 had to report as
   unrankable.
4. **Retire prompt-uniform arm summaries.** Three of the eight prompts carry
   zero published weight. Any future arm brief should report the five
   score-carrying prompts and drop plutarch, drama and travel, or state
   explicitly that it is reporting a non-score quantity.
5. **Re-measure `A_local` at M=5, not the collapse price.** The item-5 answer
   is undetermined because one coefficient is known only to `[1.552, 1.641]`
   and break-even sits at 1.6081 inside that interval. A repeat of E104
   Finding 33's M=5 row with enough blocks to reach an error bar below `±0.032`
   would settle the whole question. Every other quantity in the item-5 product
   is already tighter than it needs to be. Nothing about the share, the tax or
   the receipts needs more work first.
6. **`research/group_scaling.py` still uses `Gof[5] = 2`.** Ledger 260.7
   records this. The `adv` ratio used above survives it because both scalings
   carry the same factor and it cancels, but the absolute one-group rates that
   script prints for M=5 do not, and `RANKED_ONE_GROUP_GBPS[5] = 272.2` in
   `research/scoring_weights.py` inherits from that same route-2 derivation.
   That value is `r2_ranked / A_ranked` with `A_ranked ≈ 2`, so it is an
   IMPLIED one-group rate, not a measured one. Every ranked-frame weight in
   this document depends on it. The `harness=local` rate frame is reported
   beside every ranked table for exactly this reason, and the two frames agree
   to within 0.01 on every weight, so no conclusion here turns on it.
