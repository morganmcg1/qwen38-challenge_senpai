# E114 — the standing kernel weights are local, and the ranked ones are not identified

Student `qwen-askeladd`. PR #116. Assignment
`qwen38-r1-e114-every-kernel-arm-is-weighted-at-the-wrong-operating-point`, r1.
Base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`, rebased onto advisor base
`91b51ec3c5c3eb86b917de1efb3de7219dc3eecb`.

**Verdict: the kill rule DOES NOT fire — but the headline the assignment asked
for cannot be delivered as a number. The ranked NA weight vector is not point
identified by anything Yukon publishes. What E114 delivers instead is a proven
RANGE per arm, two arms whose sign is identified inside that range, and the
exact artefact that would close the remaining four degrees of freedom.**

This is an analysis-only result. No Swift source, kernel, test, or fixture file
was changed. `senpai/verify-ranked-score-boundary.sh` passes at this base.

## Identity tuple

| field | value |
|---|---|
| base | `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` |
| rebased onto | `91b51ec3c5c3eb86b917de1efb3de7219dc3eecb` |
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
`research/scoring_weights_selftest.py` (14/14 pass),
`research/e114_width_recovery.py` (rung 0),
`research/e114_rerank.py` (rung 1),
`research/e114-artifacts/rung0.{json,txt}`,
`research/e114-artifacts/rung1.{json,txt}`,
`research/e114-artifacts/selftest.txt`.

W&B run `cua0ege5`,
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cua0ege5>.
Two earlier runs in the same project, `mwr9kshr` and `x02hdj4v`, are failed
logging attempts. They crashed on W&B table typing before any deliverable table
was written and carry no evidence.

Reproduce:

```bash
python3 research/board_per_prompt.py fetch
python3 research/scoring_weights_selftest.py
PYTHONPATH=research python3 research/e114_width_recovery.py
PYTHONPATH=research python3 research/e114_rerank.py
python3 research/e114_wandb_log.py
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

## Rung 1 — re-ranking the arms

### The published operating point weights prompts, not rounds

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

### The weight vectors — `harness=ranked` widths, both rate frames

Rate table `harness=ranked`:

| vector | NA2 | NA3 | NA4 | NA5 |
|---|---|---|---|---|
| STANDING (`harness=local`) | 0.0240 | 0.2750 | 0.6670 | 0.0340 |
| published / maxent | 0.0398 | 0.3483 | 0.5107 | 0.1011 |
| published / gt1 tilt | 0.1158 | 0.3421 | 0.4792 | 0.0628 |
| published / gt2 tilt | 0.0000 | 0.2478 | 0.5696 | 0.1826 |

Rate table `harness=local`, same ranked widths:

| vector | NA2 | NA3 | NA4 | NA5 |
|---|---|---|---|---|
| published / maxent | 0.0416 | 0.3377 | 0.5211 | 0.0996 |
| published / gt1 tilt | 0.1206 | 0.3305 | 0.4872 | 0.0617 |
| published / gt2 tilt | 0.0000 | 0.2399 | 0.5805 | 0.1796 |

The direction the assignment predicted is right in every shape: NA=4 loses 9 to
19 points of weight and NA=5 gains. The magnitude is not identified.

### The deliverable an arm owner can apply without this script

`move_pp = Σ_NA dW[NA] · arm_pct[NA]`, `harness=ranked`:

| shape | dW NA2 | dW NA3 | dW NA4 | dW NA5 |
|---|---|---|---|---|
| maxent | +0.0158 | +0.0733 | −0.1563 | +0.0671 |
| gt1 tilt | +0.0918 | +0.0671 | −0.1878 | +0.0288 |
| gt2 tilt | −0.0240 | −0.0272 | −0.0974 | +0.1486 |

### The arm table — `harness=ranked` rate frame

Sign convention as recorded: positive is SLOWER than `a_base`. The range is the
exact extremum of the published-weighted value over every width distribution
consistent with the receipts, computed by Dinkelbach iteration over the vertex
set of the identified polytope. A flat arm is used as a positive control and
prices at exactly `[7.000000, 7.000000]`.

| arm | source | standing % | identified range | maxent / gt1 / gt2 | max move pp | sign identified |
|---|---|---|---|---|---|---|
| `mo_swap` | E110 A | +153.344 | [+128.458, +175.299] | +149.847 / +141.514 / +159.098 | 24.886 | yes |
| `mo_stage` | E110 A | +151.275 | [+127.861, +171.774] | +147.905 / +139.825 / +156.711 | 23.414 | yes |
| `f44_roofline_gap` | Finding 44 | +17.272 | [+6.732, +33.900] | +17.212 / +15.201 / +21.066 | 16.628 | yes |
| `l_loadonly` | E110 A | −14.577 | [−25.020, −6.857] | −14.304 / −12.867 / −16.941 | 10.443 | yes |
| `b_constw_e111` | E111 | −0.602 | [−1.418, +1.864] | +0.105 / +1.113 / −0.675 | 2.466 | **no — SIGN FLIPS** |
| `xs_stage` | E110 A | −1.273 | [−2.302, −0.827] | −1.291 / −1.117 / −1.530 | 1.029 | yes |
| `b_constw_e110` | E110 A | −86.870 | [−87.776, −86.242] | −87.038 / −87.412 / −86.683 | 0.907 | yes |
| `b_barrier` | E110 A | +0.305 | [+0.120, +0.405] | +0.267 / +0.190 / +0.336 | 0.185 | yes |

### The primary metric

`e114_max_abs_arm_reweight_pp` = **3.4972 pp** at the maxent shape over the E110
arm table, against a kill rule of 0.05 pp. **The kill rule does not fire.**

| quantity | value |
|---|---|
| primary metric, maxent point | 3.4972 pp |
| shape ensemble spread | −11.8306 to +5.7539 pp |
| identified-set upper bound | 24.8865 pp |
| **guaranteed lower bound** | **0.0000 pp** |

The guaranteed lower bound is zero and this is the honest headline. The standing
vector lies INSIDE the identified set for every one of the eight arms, so no
single arm's re-weighting can be proven non-zero from the receipts alone. The
3.4972 pp headline is a point estimate at a shape that failed the GT2 gate. It
should be read as "plausibly several points, provably at most 24.9, provably at
least 0".

### What IS decisive

Two arms have their sign identified across the whole set:

- `xs_stage` prices in `[−2.302, −0.827]`. It is a win at every admissible
  operating point.
- `b_barrier` prices in `[+0.120, +0.405]`. It is a small regression at every
  admissible operating point.

One arm is proven fragile: `b_constw_e111` prices in `[−1.418, +1.864]`, so it
changes sign inside the identified set. No conclusion about it can survive
without the missing trace, and its withdrawal was correct on these grounds too.

### Arms that cannot be re-weighted at all

Eleven of the arms the assignment named have no recorded per-NA cells anywhere
in this checkout: `xv4`, `xv4_stage`, `xv8` (E110 session B, still `wip` at the
recorded base), `c_loadonly`, `n_nobias`, `n_nosums`, `g_pack32`, `d_bias1`,
`e_bias6` (E111 reduced them to one weighted percentage; the per-NA cells live
in a host-local `research/out/<tag>/arms.json` on another Mac), `h_prunenarrow`
and `i_pruneall` (E108 reported one pooled percentage). The advisor asked for an
immediate report if `xv4` moved by more than 0.1 pp; `xv4` cannot be moved at
all, and this was reported at
<https://github.com/morganmcg1/qwen38-challenge_senpai/pull/116#issuecomment-5376158714>.

For `e_bias6` the recorded relation `round % = 0.877 × weighted %` fixes the
ratio but not the four cells, so it does not unblock a re-weighting either.

## Item 5 — is E100's M=5 collapse worth its register tax?

Requested by advisor feedback f1 and exempt from the kill rule.

### Share of published-weighted candidate TIME at exactly verify width 5

`harness=ranked`:

| prompt | maxent | gt1 | gt2 | identified set |
|---|---|---|---|---|
| beagle | 0.1331 | 0.0821 | 0.2348 | [0.0000, 0.8298] |
| essays | 0.1058 | 0.0684 | 0.1835 | [0.0000, 0.5559] |
| medicine | 0.0975 | 0.0642 | 0.1690 | [0.0000, 0.4970] |
| republic | 0.1104 | 0.0707 | 0.1916 | [0.0000, 0.5911] |
| botany | 0.0429 | 0.0330 | 0.0830 | [0.0000, 0.2200] |
| **COMBINED, published weighting** | **0.1154** | **0.0731** | **0.2022** | **[0.0000, 0.6645]** |

### The arithmetic

The register tax of keeping the `<T,5,5,true>` instantiation: E108's
`i_minus_case5` moved g17s registers 98 → 91 and `S_ranked` 40 → 43. The E77 law
predicts +0.0974 %; E102 measured +0.1068 % but only on the G=1 prompts, and all
five score-carrying prompts here are G=2, so the measured figure is transferred
outside the population it was measured in. Both are reported.

Net = `gain × share − tax`, as a percentage of candidate time:

| collapse price | tax | net point | net band |
|---|---|---|---|
| E104 Finding 33, +1.9 % | E77 law | **+0.1218** | [−0.0974, +1.1651] |
| E104 Finding 33, +1.9 % | E102 measured | +0.1124 | [−0.1068, +1.1557] |
| Finding 32 route 1, −2.0 % | E77 law | **−0.3282** | **[−1.4264, −0.0974]** |
| Finding 32 route 1, −2.0 % | E102 measured | −0.3376 | [−1.4358, −0.1068] |
| Finding 32 route 2, −0.3 ± 1.5 % | E77 law | −0.1320 | [−1.2935, +0.7000] |
| Finding 32 route 2, −0.3 ± 1.5 % | E102 measured | −0.1414 | [−1.3029, +0.6906] |

**Answer to the advisor's question, as asked.** Under Finding 32 route 1 the net
is negative AND the band excludes zero: `−0.3282 %` in `[−1.4264, −0.0974]`. The
advisor asked to be told if that happened, and it has. Under E104 Finding 33 the
net is positive but its band straddles zero. The three recorded collapse prices
disagree by 3.9 percentage points, which is fourteen times the tax, so **the
answer is determined entirely by which collapse price is believed, not by the
share arithmetic.** The share is the well-measured half of this product and the
price is the badly-measured half.

### The fourth price is rejected — placebo test

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

`research/scoring_weights_selftest.py` passes 14/14. The tests that matter:

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

`research/e114_rerank.py` also runs a positive control on every invocation: a
flat arm must price identically under every distribution, so its range has to
collapse onto the flat value. It does, at `[7.000000, 7.000000]`, and the run
asserts on it.

## Honest limitations

1. **Both point estimators failed the pre-registered gate.** The maxent shape
   used for the primary metric missed GT2 by 0.0899 against a 0.05 tolerance.
   Every point number in this document inherits that failure. Only the ranges
   and the two sign-identified arms are defensible.
2. **The model class changed after the pre-registration.** The iid acceptance
   model was pre-registered and failed. Maxent and tilt were chosen afterwards.
   They were still scored against the fixed pre-registered tolerances rather
   than tolerances chosen to fit them, but they are post-hoc and are labelled
   as such here.
3. **One receipt was priced.** `b8b8b860` was rejected at submission. Its
   per-prompt fields are identical to `44559d02` and to the promoted crown
   `51b9bf85`, which is what licenses its use, but no independent ranked
   population was available.
4. **The E102 register tax is transferred across populations.** It was measured
   on G=1 prompts and all five score-carrying prompts are G=2. The E77 law
   figure is reported beside it for that reason.
5. **The arm tables are local measurements.** Only the WEIGHTS are ranked. A
   per-NA percentage measured on g16s is not a ranked per-NA percentage, and
   re-weighting it does not make it one. This document re-prices the recorded
   arms at the ranked operating point; it does not claim they would measure the
   same on M5.
6. **GT1 is 19 rounds.** It is small, and it is the ground truth the maxent
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
2. **Re-run E110 session B and E111 with per-NA cells written to the ledger.**
   Nine arms are unrankable purely because their per-NA cells were reduced to a
   single weighted percentage before being recorded. The cost of keeping four
   numbers instead of one is nil.
3. **Retire prompt-uniform arm summaries.** Three of the eight prompts carry
   zero published weight. Any future arm brief should report the five
   score-carrying prompts and drop plutarch, drama and travel, or state
   explicitly that it is reporting a non-score quantity.
4. **Settle the collapse price before spending anything on the M=5 decision.**
   The item-5 answer is dominated by a 3.9-point disagreement between three
   recorded prices for the same effect. A single matched measurement of the
   `<T,5,5,true>` instantiation against its `[3+2]` alternative, at the ranked
   width mix, would be worth more than any further share arithmetic.
