# E53 results — scored verify-width mixture, the two walls, and re-pricing

Student `qwen-edward`, PR 56, branch `qwen-edward/scored-width-mixture-and-policy-map`,
base `45b7c6a4e4a94e4c6389d8a7e9d76ccd47d4a239`.

**Zero GPU work, by assignment design.** No W&B run exists for E53 and none
should. Every number below is derived from repository source at the base
commit, from the public Yukon board, or from constants the advisor supplied.
Reproduce with:

```bash
python3 research/e53_board_facts.py            # board census -> e53-board-facts.json
python3 research/e53_width_mixture.py --plutarch-mean 0.154004
python3 research/e53_repricing.py              # Part 3 table
python3 research/e53_scored_repricing.py       # Parts 1+3 synthesis
```

## Headline

1. **The scored width mixture is not the corpus histogram, and the difference
   inverts the campaign's mechanism ranking.** On the two prompts that carry the
   score, width 9 holds **4.6–9.3 %** of candidate-leg QMV cost, not the
   corpus-wide **53.8 %**. Widths {7,8} hold **21.2–25.1 %**, not **12.3 %**.
2. Consequently **E49's `<T,9,5>` stream prize falls from +3.2155 % to
   +0.47…+0.92 %**, straddling the 0.7678 % board floor, while **E44 r2's narrow
   M∈{7,8} variant rises from +1.0289 % to +1.13…+1.30 %** and clears the floor
   at every fit and every shape mix. The narrow mechanism is now worth more than
   the M=9 mechanism on the scored prompts.
3. **The brief's depth model is refuted at the source.** The shipped schedule is
   a greedy marginal-cost walk, not a truncated geometric in the acceptance
   rate, and `M = drafts + 1`, not `accepted + 1`.
4. **IID acceptance is refuted.** No per-token acceptance probability reproduces
   the published pair of telemetry constraints on either scored prompt, and the
   refutation does not depend on ledger 153's monotonicity assumption.
5. **The parity wall is survivorship, exactly as the advisor said.** Item 102
   does not refute; the join is degenerate.
6. **The policy wall needs re-framing.** The quantization clause is appended
   unconditionally, but the static review is diff-only against the trusted
   ancestor, and our own live default already sits outside the clause's literal
   envelope. The clause is a wall for new edits only.
7. **The scored pair is beagle+medicine in only 199 of 408 healthy board
   trees.** The marginal weights are crown-tree-specific and substitution is an
   observed board fact, not a modelling worry.
8. **I must report that my own pre-registered prediction 2 is refuted by my own
   model.** Detail in the scorecard below.

## What is and is not an interval here

Every interval in this report is an **identification** interval, not a standard
error. `research/e53_board_facts.py` establishes why:

| board quantity | count |
|---|---|
| rows live at 2026-08-19T15:04Z | 717 |
| rows carrying `officialMetrics` | 476 |
| healthy rows (`officialScore >= 2.0`) | 435 |
| content-unique healthy rows | 408 (19 lacked a content id and are flagged) |
| **content-distinct trees at >= 8 solvers, scores 2.3256–3.2499, sharing beagle 485/107 and medicine 472/99 to 16 digits** | **152** |
| distinct `(beagle, medicine)` behavioural pairs among the 408 | 127 |
| head-provenance shas across the 152 | 3 (559b24eb 116, 477ba726 35, 15add858 1) |

152 trees reproduce the identical telemetry pair to sixteen digits. The width
mixture is therefore a **deterministic function of the shipped schedule**, and
the board supplies roughly **one** relevant observation of it, not 371 draws.
Nothing below may be quoted with a sampling standard error. Where I give a
range, it is the set of parameter values that reproduce the published
constraints.

### The scored pair is tree-dependent, and substitution is common

Not asked for, but it falls out of the same census and it bears directly on the
pricing framework. Counting which two prompts actually occupy ranks 4 and 5
across the 408 content-unique healthy rows:

| central pair | rows |
|---|---|
| beagle + medicine | 199 |
| beagle + republic | 128 |
| beagle + botany | 66 |
| beagle + essays | 8 |
| travel + republic | 2 |
| six other pairs | 1 each |

**beagle + medicine is the scored pair in only 199 of 408 healthy trees.**
beagle itself is in the pair in 402 of 408, but its partner is not stable.

This is the substitution kink observed directly rather than modelled: the pair
membership that `CROWN_ORDER_STATS` encodes is a property of the crown tree
`ef42e043`, not a fixed property of the benchmark. Two consequences:

* the marginal weights 0.483694 / 0.516306 are **crown-tree-specific**, and any
  mechanism large enough to move ranks re-derives them;
* the kink is not a theoretical worry. 209 of 408 trees already sit at a
  different pair, so a mechanism priced past +1.0551 % on the crown tree is
  being priced on an order-statistic configuration it is itself dissolving.

The module handles this correctly by re-sorting, which is why Part 3b prices
through `score_pct_from_leg_gains()` rather than by multiplication. I record the
census because it turns that design choice from prudence into a measured
requirement.

## Part 1 — the scored verify-width mixture

### 1.1 The brief's model is wrong at the source

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift` at base `45b7c6a4`:

| fact | location |
|---|---|
| depth chosen by a greedy marginal-cost walk, `reach > h*(1+expected)/(1+depth*h)` | `costModelDepth`, line 738 |
| `h = 0.18` | line 668 |
| per-position acceptance EMAs seeded `0.85 * 0.98^i` | line 635 |
| `acceptEMAAlpha = 0.15` | line 637 |
| optimism transfer capped at 0.95 | line 837 |
| `widthCap = fullAcceptStreak >= 2 ? 8 : 5` | lines 700, 707, 735, 744 |
| parent offer `max(1, min(options.depth, 8, remaining - 1))` | `QwenRuntimeMTPDriver.swift:120-167` |
| `rowsPerRound(depth) = depth + 1` | `Qwen36MTPBlockSession.swift` |

So the verify width is **`M = drafts proposed + 1`**. `effective_mean_draft_len`
counts drafts **proposed**, not accepted (ledger 153, MT1). The brief's
`M = k + 1` over the accepted count gives beagle mean M = 4.785 against the
published 5.533, so it is not the published quantity.

The schedule constants are pinned to a real tree: commit
`cbdc3a8d5fa9d4ffe035e4847b629fb296234cc6` (a validated submission, ancestor of
this base, score 3.004, beagle n = 4.5327) carries **byte-identical** schedule
constants. Our own submitted commits `2b0c36a0`, `e277c57b` and `dbf91c6c` are
not local objects, so I could not diff them; I name that gap rather than assume
they match.

### 1.2 IID acceptance cannot reproduce the telemetry

Published per-prompt constraints: beagle `n = 4.532710`, `A/D = 0.8351`,
`R = 107`; medicine `n = 4.767677`, `A/D = 0.8750`, `R = 99`.

Sweeping a single per-token acceptance probability `q` from 0.05 to 0.95 traces
a one-dimensional frontier in `(n, A/D)` that **never passes through either
published point**:

* at `n = 4.5327` the IID frontier gives `A/D ~= 0.59` (published 0.8351);
* at `A/D = 0.835` it gives `n ~= 6.2` (published 4.5327).

To remove any dependence on ledger 153's monotonicity assumption I also
enumerated the legal integer round counts from exact rationals under `R + A = 512`:

| prompt | legal `(R, A, A/D)` |
|---|---|
| beagle | (107, 405, 0.8351); (214, 298, 0.3072); (321, 191, 0.1313); (428, 84, 0.0433) |
| medicine | (99, 413, 0.8750); (198, 314, 0.3326); (297, 215, 0.1518); (396, 116, 0.0614); (495, 17, 0.0072) |

**No** legal pair on either prompt is consistent with the published `n` under
IID. The refutation is therefore structural.

### 1.3 A two-state burst model reconciles both constraints

`BurstAcceptance` in `research/e53_width_mixture.py` runs the *actual* shipped
schedule — the same cost-model walk, EMA seeds, alpha, optimism cap and width
cap — over a two-state easy/hard token stream with fixed margins 10.0 and 0.5.
Feasibility is narrow: `q_easy` 0.94–0.97, `share_easy` 0.10–0.24, `q_hard`
0.93–0.96.

Feasible fits reproduce `n`, `A/D`, the round count and zero non-drafting
rounds. Cost-weighted width shares, using thorfinn's E46 refit
`T(M) = 16.757 + 27.532*ceil(M/IPG) + 9.624*M`:

| prompt | fits | f{4,5,6} | f{7,8} | f{9} | f{1,2,3} | predicted R (published) |
|---|---|---|---|---|---|---|
| beagle | 8 of 15 | 0.6827–0.7428 | 0.1659–0.2160 | 0.0246–0.0923 | 0.0360–0.0440 | 106.5–109.5 (107) |
| medicine | 6 of 15 | 0.6208–0.6403 | 0.2530–0.2829 | 0.0656–0.0864 | 0.0236–0.0301 | 98.5–99.3 (99) |

Composite at the advisor's marginal score weights **beagle 0.483694 /
medicine 0.516306**, over the two `(persistence, q_easy)` points feasible on
both prompts:

| point | f{4,5,6} | f{7,8} | f{9} | f{1,2,3} |
|---|---|---|---|---|
| persistence 0.00, q_easy 0.96 | 0.6688 | 0.2361 | 0.0613 | 0.0337 |
| persistence 0.50, q_easy 0.96 | 0.6727 | 0.2199 | 0.0773 | 0.0301 |

Relaxing the requirement that both prompts share the same nuisance parameters
gives the full identification envelope over all 48 cross-product fit pairs:

| block | envelope |
|---|---|
| f{4,5,6} | **0.6502 – 0.6887** |
| f{7,8} | **0.2117 – 0.2510** |
| f{9} | **0.0462 – 0.0892** |
| f{1,2,3} | 0.0295 – 0.0366 |

### 1.4 Comparison with askeladd's independent estimate

| block | askeladd | E53 (this work) | verdict |
|---|---|---|---|
| M∈{4,5,6} | 65.01 % | 65.02–68.87 % | **agree** |
| M∈{7,8} | 9.158 % | 21.17–25.10 % | **disagree, factor ~2.5** |
| M=9 | 20.48 % | 4.62–8.92 % | **disagree, factor ~2.9** |

We agree on the large low-width block and we **roughly swap** the two narrow
blocks. The boundary-sensitivity table explains why: moving `q_hard` by ±0.02
around the beagle fit moves `f{7,8} + f{9}` from 0.2175 to 0.3693, but the
split *inside* it moves much less predictably. The published telemetry pins the
**sum** far better than the split.

| d q_hard | n | A/D | f{4,5,6} | f{7,8} | f{9} | f{7,8}+f{9} |
|---|---|---|---|---|---|---|
| -0.02 | 4.2961 | 0.8063 | 0.7311 | 0.1668 | 0.0507 | 0.2175 |
| -0.01 | 4.4381 | 0.8204 | 0.7073 | 0.1903 | 0.0547 | 0.2450 |
| 0.00 | 4.5397 | 0.8357 | 0.6827 | 0.2094 | 0.0647 | 0.2741 |
| +0.01 | 4.7125 | 0.8449 | 0.6669 | 0.2232 | 0.0770 | 0.3002 |
| +0.02 | 4.9112 | 0.8585 | 0.6071 | 0.2750 | 0.0943 | 0.3693 |

I therefore state the disagreement honestly and do **not** claim to have
falsified askeladd. What both estimates agree on, and what actually drives the
re-pricing, is that **`f{9}` is nowhere near the corpus 53.8 %**. His 20.48 % and
my 4.62–8.92 % are 2.6x and 6–12x below it respectively. The re-pricing
conclusion in Part 3 survives either estimate; only its magnitude differs.

### 1.5 Plutarch control — the absorbing state reproduces itself

Plutarch publishes `n = 0.154004` with **449 non-drafting rounds**, on 151 of
the 152 trees in the cohort. Fitting the same machinery gives `q = 0.3150` and
predicts **462.4** non-drafting rounds and `R = 498.1` against a published 449.
The mechanism is the absorbing state: a non-drafting round updates no EMA, so
once the walk stops proposing it cannot learn its way back. The model slightly
over-predicts the trap, which is the expected direction of error for a
two-state approximation of a heterogeneous prompt. This is a control, not a
scored claim; plutarch is rank 1 and carries zero marginal weight.

## Part 2 — the two walls

### 2a. The parity wall is survivorship. CONFIRMED.

The advisor's claim is exactly right and item 102 does not refute anything.
Over all 717 rows:

| | metrics present | metrics absent |
|---|---|---|
| score present | 476 | 0 |
| score absent | **0** | 241 |

The join is degenerate in both directions. `parity_all_ok` is true on all 476
metrics-bearing rows and `parity_ok` is true on all 3,808 per-prompt rows. A run
that fails parity never publishes metrics, so "parity is always true among rows
that have metrics" is a tautology, not evidence of parity invariance.

Failure taxonomy over the 717 rows (advisor's older dump had 635):

| category | count |
|---|---|
| score did not improve | 406 |
| verify surface | 85 |
| null / no reason | 78 |
| parity | 29 |
| bypass review | 23 |
| head resolution | 12 |
| timed | 11 |
| build | 9 |
| public behavior | 9 |
| workflow timeout | 6 |
| floor | 3 |
| setup | 3 |
| correctness | 1 |

### 2b. The policy wall — re-framing required

Full detail in `research/e53_policy_wall.md`. The three findings that change the
advisor's reading:

1. **The clause is unconditional.** The "Controlling quantization rule" at line
   453 of `.github/scripts/run-submission-static-review.sh` is appended after
   the per-track dispatch `fi` at line 452. It applies to every track. Its
   stated premise is NVFP4 group-16, but our target is affine 4-bit group-64, so
   the clause's premise does not describe the model it is judging. It rejects
   "any bit width other than 4 or 8" "even when the re-quantized path passes the
   correctness gates".
2. **The decisive mitigation is that the review is diff-only.** The workflow
   (lines 1190–1203) and the script (lines 299–395) judge only the diff against
   the trusted ancestor base. Base content is never re-judged. The clause is a
   wall for **new** edits, not a standing property of the tree.
3. **The q2/q4 question resolves against the advisor's framing.** Our live
   default **is already** 2-bit coarse plus 4-bit rerank.
   `mtp-head.manifest.json` declares `qwen38-mtp-head-q2-q4-rerank-v1`;
   `Qwen35.swift:3178-3181` calls `quantizedMM(..., groupSize: 64, bits: 2, mode: .affine)`
   over 98,336 rows, then reranks 32 rows at 4 bits at lines 3202–3207. This
   matches `quantized.h:1908`. The crown tree therefore already sits outside the
   clause's literal envelope, which it survives only because of finding 2.

The clause-attribution table for all 23 bypass failures includes our own two
3-bit readout failures (board items #11 and #13) and, decisively, a **promoted**
affine-2 singlerow submission (#14, +1.84 %). A promoted affine-2 row proves the
clause is not enforced against base content. All probed run and tree URLs
returned HTTP 200 unauthenticated, so this is reproducible from the public
board.

## Part 3 — re-pricing at `psi_mtp = +0.693391`

The full table is `research/e53_repricing.md`. Gate-26 selftest passed with 71
PASS lines before any number was quoted; every conversion calls
`research/qmv_score_leverage.py` public functions with `psi=` passed explicitly,
and nothing above the +1.0551 % kink was priced by multiplication.

Ten mechanisms re-priced, five refused with reasons, five retracted levers
listed unpriced. The result that **changed** rather than merely moved:

* the shared-template `load_vector`/`qdot` family flips from **-0.1789 %/%
  "harmful"** to **+0.6934 [+0.6923, +0.6945] % score per 1 % kernel-wide QMV
  win**. A kernel win of **>= 1.107 %** clears the 0.7678 % floor. This is the
  highest-leverage revived family: it needs no shape gate, and the pinned serial
  binary cannot follow it.

Advisor QMV cost-reduction targets move to **0.749 / 1.107 / 2.954 %** (crown
gap / 1 sd / 2 sd). The 2-sd target sits above the kink, so the old linear
2.280 % understated the true requirement by about 33 %.

Two items flagged for the advisor: `research/noise_floors.py` still carries the
stale `EFFECTS["alphonse E44 predicted"] = -0.17`, and ledger 174/175's E44
headline range was produced by multiplying above the kink.

### 3b. Closing caveat C2 — the scored mixture changes two verdicts

Part 3 kept the corpus-wide histogram as its width weighting and flagged that as
caveat C2. `research/e53_scored_repricing.py` closes C2 for the two mechanisms
whose price is dominated by a width share. It prices beagle and medicine
**separately** and combines them with `score_pct_from_leg_gains()`, so the order
statistics and the kink are applied by re-sorting; the marginal weights are
never multiplied in, only recovered as a check (0.483694 / 0.516306, exact).

Validity check: at the corpus histogram this path returns **+3.2155 %** for the
M=9 mechanism, reproducing Part 3's independent number exactly. A positive
control confirms the module price is linear in both the win and the share, which
is what makes the re-weighting exact rather than approximate.

| mechanism | corpus HIST price | **scored-mixture price** | clears 0.7678 % floor? |
|---|---|---|---|
| `<T,9,5>` stream prize at M=9 (E49) | +3.2155 % order-stat | **+0.4719 … +0.9193 %** | **no at the worst fit, yes at the best** |
| E44 r2 narrow, attn_out only | +1.1359 % order-stat | **+1.2659 … +1.4840 %** | yes, always |
| E44 r2 narrow, mlp_down only | +0.8355 % | **+0.9737 … +1.1176 %** | yes, always |
| E44 r2 narrow, equal shape mix | +1.0289 % | **+1.1251 … +1.3008 %** | yes, always |

The `psi_mtp` interval contributes almost nothing next to the mixture
uncertainty: for the M=9 mechanism it moves the envelope from
[+0.4712, +0.9178] to [+0.4726, +0.9207], about 0.3 % of the width that the
mixture itself contributes.

**This inverts the campaign's mechanism ranking.** The M=9 stream prize was
priced at 4.2x the board floor and is the load-bearing justification for E49's
register budget; on the scored prompts it is a coin-flip against the floor. The
narrow M∈{7,8} mechanism was "marginal at the mlp end" and is now above the
floor everywhere. Its blocker is unchanged and is not a pricing question: the
bit-exactness bar of ledger 175(A), which E51's reassociation dose ladder owns.

## Pre-registration scorecard

Pre-registered in `research/e53_prereg.md`, commit `7d771ce` at
2026-08-19T15:11:23Z, **before** any of the above was computed.

| # | prediction | confidence | outcome |
|---|---|---|---|
| 1 | `f{7,8}` above askeladd's 9.158 % | 55 % | **supported** — 21.2–25.1 %, and for a stronger reason than the weight tilt |
| 2 | `rho(7) + rho(8) < rho(9)` | 75 % | **REFUTED by my own model, in 14 of 14 feasible fits** — `rho(7)+rho(8)` 0.138–0.244 against `rho(9)` 0.016–0.059 |
| 3 | `f{4,5,6}` largest and above 50 % | 70 % | **supported** — 65.0–68.9 % |
| 4 | the brief's truncated geometric fails at least one constraint | 90 % | **supported** — it fails at the source and on the telemetry |

Prediction 2 is the one I got wrong, and it is worth stating plainly because it
was my highest-confidence quantitative claim about the shape of the mixture. I
expected the width cap to pile probability at M=9 through the
`fullAcceptStreak >= 2` path. It does not: the greedy walk's marginal-cost test
`reach > h*(1+expected)/(1+depth*h)` stops the walk well before the cap on most
rounds, so the cap is rarely the binding constraint. The direction of my error
is the same direction that makes the Part 3b re-pricing bite, so I have an
incentive to be sceptical of it — which is precisely why the scored-mixture
conclusion is stated as an envelope and why I show that it survives askeladd's
20.48 % estimate too.

## Caveats

* **T(M) is an Apple M4 Pro fit** (thorfinn E46, `nax_available false`, max
  residual 0.770 ms). Absolute levels do not transfer to the ranked M5. The
  cost-share *ratios* transfer better than the levels, but this is an
  assumption, not a measurement.
* **T(1) and T(2) are extrapolations.** M=1 falls to `qmv_fast_impl` and M=2 to
  `qmv_fast_crossrow_affine4_g64<T,2>`. The f{1,2,3} block is the least
  trustworthy row in every table, and it is also the smallest.
* **Stop tokens are not modelled.** The parent continues the trajectory through
  EOS for the full window; the burst model does not represent that boundary.
* **The composite weighting is prompt-QMV-cost x score weight.** Part 3b avoids
  it entirely by pricing per prompt, which is the better path; the Part 1
  composite is retained only for comparison with askeladd's number.
* **19 of the 435 healthy board rows lacked a content id** and are flagged
  rather than silently dropped. Content ids were reused from E50's
  `/tmp/tree_ids.json`.
* **`psi_mtp` is an ALU-injection measurement.** At M∈{2,3,4} injected ALU work
  converts to time at only 54–93 %, so `psi` applies cleanly to
  weight-traffic-shaped costs and conservatively at narrow widths.

## Suggested follow-ups, not implemented

1. **Settle the f{7,8} vs f{9} split with one cheap instrumented run.** Both
   askeladd and I identify the *sum* well and the *split* poorly, and the split
   is what re-prices E49. A per-prompt dispatched-width histogram on beagle and
   medicine — the E48 instrument — would collapse both envelopes to a
   measurement and is far cheaper than the register work E49 is currently
   justifying with the corpus number.
2. **Re-examine E49's register budget before spending more on it.** Its headline
   price depends on `qmv_share(9) = 0.5383`, which no scored-prompt evidence
   supports.
3. **Update `research/noise_floors.py`.** The stale
   `EFFECTS["alphonse E44 predicted"] = -0.17` is now known wrong; the gate owner
   should make that edit, because updating it fires check 7 by design.
4. **Re-state ledger 174/175's E44 headline range.** Its top end was produced by
   multiplying above the kink.
5. **Ask the organizers to correct the quantization clause's premise.** It cites
   NVFP4 group-16 while judging an affine 4-bit group-64 model, and a promoted
   affine-2 submission already sits inside the tree it would reject.
