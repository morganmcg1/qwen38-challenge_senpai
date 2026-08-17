# E17 — curve transfer and refit

Assignment `qwen38-r1-e17-curve-transfer-and-refit`, revision r1, PR #19.
Base `e6e6f81767e84cc8c39b48c09a4f5cac597cdbca` (PR #13 merged; the per-depth
`headStepCostRatioByDepth` curve is the compiled default).

---

## 1. Pre-registration (deliverable 14)

**Signed and committed before any E17 timing arm was launched.** The commit that
introduces this section contains no E17 run artifacts; `.mlxfast-private/e17/`
does not yet exist. That ordering is the only thing that makes the rest of this
document worth reading, because the question E17 asks — does a curve fitted on
one prompt survive a median over eight? — is exactly the question that
prompt-level or convention-level freedom can fake.

### 1.1 What is being compared

Two binaries differing in eight `Double` literals and nothing else:

| arm | `headStepCostRatioByDepth` |
| --- | --- |
| `CURVE` | `[0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]` (this branch's HEAD) |
| `FLAT18` | `[0.18 × 8]` |

`FLAT18` is the counterfactual rather than a revert of the pre-merge file
because the shipped extend test is

```text
extend at depth d  iff  reach_d > h[d] · (1 + expected_d) / (1 + cumH_d)
```

and with a flat vector `cumH_d = d·h`, so a flat `0.18` reproduces the retired
scalar rule `h(1+expected)/(1+d·h)` term for term. The contrast is therefore the
constants alone, not a partial revert of everything else that moved between
bases. `research/e17-build.sh` asserts, per arm, that the vector declaration is
present, that no scalar `headStepCostRatio` is declared, that the arm's own
literals are installed, and that `sdpaWidthWallDepthCap = 5`,
`segmentedVerifyDepthCap = 8` and `segmentedStreakGate = 3` are untouched. The
`FLAT18` patch is keyed on the exact shipped literal, so a base that re-fitted
the curve makes the build fail loudly instead of silently producing a third arm.

### 1.2 Prompt set and order — fixed here

Eight 512-token prose seeds, 512 decode steps, one golden each:

| # | id | register | prompt file | seed tokens |
| --- | --- | --- | --- | --- |
| 0 | `english` | expository history | `research/e11_prose_gate_english_512.txt` | 870 |
| 1 | `narrative` | literary narrative | `research/e17_prose_narrative_512.txt` | 981 |
| 2 | `technical` | technical exposition | `research/e17_prose_technical_512.txt` | 903 |
| 3 | `dramatic` | stage dialogue | `research/e17_prose_dramatic_512.txt` | 1146 |
| 4 | `travel` | first-person travel | `research/e17_prose_travel_512.txt` | 985 |
| 5 | `philosophy` | philosophical argument | `research/e17_prose_philosophy_512.txt` | 909 |
| 6 | `natural_history` | 19th-c. natural history | `research/e17_prose_natural_history_512.txt` | 1005 |
| 7 | `medicine` | clinical review | `research/e17_prose_medicine_512.txt` | 939 |

Seed-token counts are from `research/e17_token_check.py` against the target
tokenizer; `generate-golden` uses the first 512 tokens, so every prompt has a
full seed window with margin.

Commitments:

* **No prompt is added, dropped or reordered after timing starts**, and every
  prompt that completes is reported, including any that embarrasses the curve.
* `english` is E11's prompt and therefore **in-sample**: the shipped curve was
  fitted against marginals measured on it. It is retained as an anchor that ties
  E17 back to the r3 pair, and it is reported, but the **headline is the
  held-out 7**. The all-8 median is reported alongside it because 8 prompts
  mirror the ranked pool's shape and exercise the even-`n` rule.
* The other seven are original prose written for this experiment, in registers
  chosen to spread acceptance behaviour (dialogue and clinical prose should
  draft differently from narrative and travel). None is a copy or repetition
  task.
* `correctness_prompts/public_longcopy_gate_english_512*` — the copy fixture
  whose 0.89–0.95 accept rate flatters deep drafting — **is not used**, and
  nothing under `fixtures/` or `correctness_prompts/` is touched. Goldens land
  outside Git in `.mlxfast-private/e17/goldens/`.

### 1.3 Interleaving and order-effect control

* Arms are interleaved **within** prompt, never blocked by arm, so session-scale
  thermal drift cannot correlate with arm.
* The within-prompt order **alternates with the prompt's index**: even index →
  `CURVE, FLAT18`; odd index → `FLAT18, CURVE`. A systematic first-arm/second-arm
  effect (cold cache, cool-gate history) therefore cannot correlate with arm
  either.
* Every run is serialized through `research/await-lock-then-run.sh`, so one
  model-holding process is resident at a time and each arm passes the unmodified
  40 °C cool gate, drift tripwire and orphan scan.
* GPU/CPU temperature and power are sampled before and after every arm into that
  arm's `meta.txt`.
* Timed arms run with every `MLX_QWEN_MTP_*` name cleared (`research/e11-run.sh`
  unsets them), so no trace or research override can leak into a headline; the
  verbatim surviving list is recorded per arm. Both arms use the pinned/declared
  head at `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared`.

### 1.4 Metric convention — fixed here

For each prompt `p`:

```text
raw_p = serial_seconds_per_token / mtp_seconds_per_token
```

Both terms are read **verbatim** from `.parent_measured_seconds_per_token` in
that run's own reports — `03-mtp-timed.json` for the depth-0 serial control leg,
`04-mtp-timed.json` for the depth-8 arm leg. **Nothing is subtracted.** This is
prefill-inclusive by construction, and §2 below establishes that from source.
Each `--local-iterate` invocation measures its own byte-identical serial control,
so each prompt yields two independent serial legs and their spread is that
prompt's noise floor.

Reported statistics:

* `median(raw_p | CURVE)` and `median(raw_p | FLAT18)`, ranked median (with even
  `n`, the mean of the two middle values), over the held-out 7 and over all 8.
* `g_p = (mtp_spt_FLAT18 − mtp_spt_CURVE) / mtp_spt_FLAT18` — the curve's
  per-prompt gain on the MTP leg — and `g_median`. This is the same quantity r3
  published as `g = 6.378%`, so E17's `g_median` is directly comparable to it.
* Per-prompt full table, min/max spreads, win count, per-prompt serial noise
  floor, temperatures.
* Per arm per prompt: depth histogram (from `effective_draft_lengths`), round
  count, mean/max effective depth, accepted/rejected totals, accept rate,
  replayed-round count, row accounting, `all_tokens_matched`, `parity_all_ok`,
  `residual_divergence_count`, head sha256, binary sha256.

### 1.5 Stop rules — fixed here

| outcome | reading | action |
| --- | --- | --- |
| `g_median ≥ +4%` | the curve's r3 win is real at ranked scale | report a transferring win; the curve is the right lever to keep refining |
| `+1%` ≤ `g_median` < `+4%` | real but much smaller than the single-prompt 6.378% | publish the shrinkage honestly; the curve stays, but priority should shift to levers with more headroom |
| `g_median < +1%`, or the sign flips across prompts | **the r3 headline was substantially a one-prompt artifact** | publish as the most valuable negative available and stop refining `h`; a refit fitted on one prompt inherits the same defect |

### 1.6 Falsification statement

This experiment is designed so that the curve can lose, and the following would
falsify "the merged per-depth curve is a genuine ranked-scale improvement":

1. `g_median` over the held-out 7 falls below `+1%`, i.e. an order of magnitude
   below the `+6.378%` r3 published from `english` alone.
2. `g_p` changes sign across prompts — the curve helps some registers and hurts
   others — so a median over eight unseen prompts is a coin flip and the
   *published median* rather than the mean is what decides the score.
3. The curve's advantage on the held-out set is smaller than the per-prompt
   serial noise floor, in which case r3's win was inside measurement error all
   along and only looked clean because a single prompt was measured twice.
4. `CURVE`'s depth histogram on held-out prompts collapses to the same depths
   `FLAT18` chooses, which would mean the eight fitted constants are not
   actually changing the schedule off the fitting prompt and the r3 delta came
   from something else.

If any of 1–3 holds I will report the curve as a one-prompt artifact, in those
words, and will say so in the result summary rather than foregrounding the
in-sample anchor. I will not recover a headline by dropping prompts, switching to
the all-8 or decode-only variant after the fact, substituting the mean for the
median, or re-timing a losing prompt until it wins. Any repeat run will be
reported as a repeat, with both values shown.

The in-sample anchor supplies a fifth, diagnostic falsification: if `english`
itself no longer reproduces something close to r3's `g = 6.378%` on this base and
host, then the E17 harness — not the curve — is what changed, and no held-out
conclusion may be drawn until that is resolved.

---

## 2. The metric convention, from source (deliverable 12 support)

The advisor's r1 feedback is **right** that the ranked score is prefill-inclusive
and that E11's r3 prose claiming seed prefill is "outside the ranked score" was
wrong. The chain, in the trusted (non-editable) driver:

| # | source | what it establishes |
| --- | --- | --- |
| 1 | `Sources/MLXFastRuntime/QwenRuntimeMTPDriver.swift:94` — `let started = Date()` | the clock starts **before** any decode work |
| 2 | `…:95` — `beginMTPDecode(...)` | seed prefill happens **after** the clock started, so it is inside the span |
| 3 | `…:197` — `decodeSeconds = Date().timeIntervalSince(started)` | the span runs to the last emitted token and contains prefill |
| 4 | `Sources/MLXFastRuntime/QwenRuntimeMTP.swift:442` — `decodeSecondsPerToken = decodeSeconds / max(decodeTokenCount, 1)` | the whole span, prefill included, is divided by decode tokens only |
| 5 | `Sources/mlxfast-swift/main.swift:~2013-2028` | emits `parent_measured_seconds_per_token` from that value, and emits `seed_prefill_seconds` / `prefill_seconds_per_token` beside it marked *"Observability only — nothing above subtracts it — and omitted (never zero)"* |
| 6 | `benchmark-qwen-mtp.sh:651-652,726` | the reported ratio is built from `parent_measured_seconds_per_token`, unmodified |

So a 512-token seed is charged to a 512-token decode window, and both legs of
every pair carry it. There is no subtraction anywhere on the scored path, and
`seed_prefill_seconds` exists purely to be looked at.

**Consequence for prompt design, which is the reason this matters:** prefill is a
fixed per-leg cost that speculation cannot reduce, so it dilutes every ratio
toward `1.0`. Measured here, `P ≈ 3.995 s` against a serial leg of `38.2 s` and
an MTP leg of `23.7–25.3 s` — i.e. prefill is ~10 % of the serial leg and ~16 %
of the MTP leg. The prefill-inclusive ratio is therefore *lower* than the
decode-only ratio (`1.609` vs `1.733` for the curve arm), and any candidate's
apparent speedup is capped by how much of its leg is unspeculatable prefill.

## 3. Where the r1 arithmetic restatement went wrong

The convention correction is right; the numbers attached to it are not, and the
correction must not be recorded with them. `research/e17_analyse.py --r3`
recomputes both r3 arms from their own committed reports:

```text
Sp3 (scalar h=0.18)   serial 0.074569517513737082  mtp 0.049472851445898414
Hp3 (merged curve)    serial 0.074528021505102515  mtp 0.046317363390699029
measured seed_prefill_seconds: 3.994360 .. 3.998045 s (all four legs, n = 512)

                    Sp3        Hp3      pair delta
prefill-inclusive  1.507282   1.609073   0.101791   <-- equals what r3 published
decode-only        1.602116   1.732572   0.130456
```

The published r3 pair **already was** the prefill-inclusive pair, to six
decimals; likewise `g = 6.378%` already was the prefill-inclusive MTP-leg gain
(decode-only would have been `7.577%`). Applying the advisor's convention to r3
therefore changes no published number — only the surrounding prose was wrong.

The r1 feedback instead quotes `Sp3 = 1.437971`, `Hp3 = 1.521771`, "scored delta
17.67% smaller". Those come from adding `P/n` to values that already contained
it — charging seed prefill twice:

```text
(serial_spt + P/n) / (mtp_spt + P/n):  Sp3 = 1.438285   Hp3 = 1.521191
advisor r1 quoted                   :  Sp3 = 1.437971   Hp3 = 1.521771
pair delta 0.101791 -> 0.082906, i.e. 18.55% smaller (r1 quoted 17.67%)
```

The reconstruction lands within 0.03 % of both quoted ratios and reproduces the
"~18 % smaller delta" claim, which identifies the operation. The residual is
consistent with a single rounded `P` instead of each leg's own measured value.

Note also the direction: moving from decode-only to prefill-inclusive *shrinks* a
ratio, so had r3 really published decode-only numbers, the corrected values would
have been `1.602116 / 1.732572` — **larger**, not smaller. E17 uses the
prefill-inclusive convention throughout regardless, per §1.4, because that is
what the driver measures.

## 4. Q2 — the shipped fit is wrong, and being wrong is why it wins

### 4.1 The gate, transcribed exactly from source

`costModelDepth` (`Qwen36MTPBlockSession.swift` l.621) decides, for each round,
how many drafts to request. Written out, the loop extends from depth `d` to
`d+1` if and only if

```text
    r_d * (1 + H_d)  >  (1 + E_d) * h[d]
```

where

```text
    r_d  = reach(d)   = optimism-capped estimate of P(all d+1 drafts accepted)
    E_d  = sum_{j<d}  prod_{k<=j} p_k        expected extra tokens already bought
    H_d  = sum_{j<d}  h[j]                   head cost already committed
    h[]  = headStepCostRatioByDepth          head step cost / verify forward cost
```

Dividing through gives the form used everywhere below:

```text
    extend  iff  reach(d) > h[d] * (1 + E_d) / (1 + H_d)
                            \___________  ___________/
                                        \/
                                    threshold(d) = h[d] * f(d)
```

`f(d) = (1 + E_d)/(1 + H_d)` is the *leverage* term. For a flat `h`, `f(d)`
collapses and the rule reduces exactly to the retired scalar gate, which is a
useful sanity check that the transcription is faithful: `flat` arms are not a
different mechanism, they are this same mechanism with a degenerate `h`.

`research/e17_gate_sim.py` is a term-for-term Python port of `costModelDepth`,
`recordAcceptOutcome` (l.669), the `fullAcceptStreak` update (l.1126), the EMA
prior `0.85 * 0.98^i` (l.496), `acceptEMAAlpha = 0.15`, the 5/8/3 caps and the
0.95 optimism cap. It is the instrument for this section, because **the runtime
emits no reach or threshold telemetry** — `MLX_QWEN_MTP_TRACE=1` dumps only
hexfloat top-2 rows. Q2 and Q3 therefore cannot be answered by reading a trace;
they are answered by a faithful port plus the pricing test in section 4.4.

### 4.2 Defect 1 — the shipped curve understates the cheap depths

`headStepCostRatioByDepth` as shipped, against the marginals actually measured
for depths 0..3:

| depth | shipped h[d] | measured marginal | error |
|---|---|---|---|
| 0 | 0.0842 | 0.0971 | **-13.3 %** |
| 1 | 0.0775 | 0.1152 | **-32.7 %** |
| 2 | 0.2482 | 0.2482 | (fit anchor) |
| 3 | 0.3761 | 0.3761 | (fit anchor) |

The two entries that every round pays are the two that are wrong, and both are
wrong in the same direction: the fit makes the first two head steps look cheaper
than they are.

### 4.3 Defect 2 — the shipped curve is not monotone

A head step at depth `d+1` cannot cost less than one at depth `d`: the history
it attends to is strictly longer. The shipped fit violates this three times:

```text
  d=1  0.0775 < 0.0842 = h[0]
  d=4  0.2919 < 0.3754 = h[3]
  d=6  0.2870 < 0.3000 = h[5]
```

These are artefacts of an unconstrained least-squares fit through noisy
per-depth timings, not physical measurements.

**The monotone-by-construction refit** (the deliverable) takes the measured
marginals where they exist and then applies a running maximum over the tail, so
monotonicity holds by construction rather than by luck:

```text
  refit = [0.0971, 0.1152, 0.2482, 0.3761, 0.3761, 0.3761, 0.3761, 0.3909]
                                           \_______________________/
                                            running max of the tail
```

### 4.4 The pricing test — which h vector actually explains the clock?

This is the load-bearing measurement of the whole section, and it needs no
acceptance model at all. Each measured arm already tells us the exact depth it
chose on every round. Price that *observed* round sequence under a candidate
`h`, using the gate's own cost model (a round drafting `d` costs
`1 + sum(h[:d])` verify-forward units), and compare the predicted cost ratio
between two arms with their measured decode seconds/token ratio.

Measured r3 prose arms, 512 decode tokens, same host and golden:

| label | curve | rounds | mean depth | accept | s/tok incl. prefill | s/tok decode-only | depth histogram |
|---|---|---|---|---|---|---|---|
| Hp3 | shipped | 246 | 2.020 | 0.535 | 0.046317 | 0.038513 | d1:2 d2:237 d3:7 |
| Sp3 | flat 0.18 | 245 | 2.367 | 0.460 | 0.049473 | 0.041670 | d1:19 d2:138 d3:67 d4:21 |
| S20p | flat 0.20 | 250 | 2.152 | — | 0.049149 | 0.041347 | d1:45 d2:138 d3:51 d4:16 |

Priced Hp3 -> Sp3 (measured decode cost **+7.577 %**):

| pricing vector | predicted | error vs measured |
|---|---|---|
| **refit (monotone, measured)** | **+7.651 %** | **+0.074 pp** |
| shipped fit | +8.021 % | +0.444 pp |
| flat 0.20 | +4.321 % | -3.256 pp |
| flat 0.18 | +3.990 % | -3.587 pp |

Three conclusions, in order of importance:

1. The measured monotone marginals price real timing to **0.074 pp**, about
   **6x more accurately** than the shipped fit. The cost model itself is sound;
   the shipped *coefficients* are what carry the error.
2. Both flat vectors are wrong as pricing vectors by 3-4 pp. Head-step cost is
   genuinely depth-dependent. A scalar `h` is not a rival theory of cost, it is
   a rival *policy* that happens to be priced badly.
3. The refit is therefore the better description of the machine. Section 4.5 is
   why that does not make it the better policy.

Cross-check on the second pair, Hp3 -> S20p (measured decode **+6.855 %**):
shipped predicts +7.075 %, refit +6.491 %. Same ordering of magnitude, no sign
error.

### 4.5 Why correcting the fit is predicted to *lose* — the accidental conservatism

Look again at the threshold, `threshold(d) = h[d] * (1 + E_d)/(1 + H_d)`. `H_d`
is the cumulative head cost *already committed*, and it sits in the
**denominator**. So understating `h[0]` and `h[1]` shrinks `H_d` at every deeper
`d`, which **raises** `threshold(d)` there. The fit's error at the cheap depths
does not make the gate greedier; it makes the gate *more conservative deeper in*.

Simulated against a prior accept process `q[i] = 0.86 * 0.94^i`:

| curve | opens to depth | slack at the deciding depth |
|---|---|---|
| shipped | 2 | **-0.0053** at d=2 (closes, only just) |
| refit | 3 | **+0.0052** at d=2 (opens, only just) |
| flat 0.18 | 4 | +0.0170 at d=3 |
| flat 0.20 | 3 | -0.0081 at d=3 |

The shipped curve sits on a **knife edge at depth 2**, roughly half a percent of
reach away from flipping. Correcting the fit moves it across that edge and the
gate starts asking for a third draft.

And on prose, asking for more drafts is the losing move. The measured arms say
so directly: the shipped curve wins with mean depth **2.020** against flat
0.18's **2.367** at a statistically identical round count (246 vs 245) and a
*higher* accept rate (0.535 vs 0.460). **The curve wins by drafting more
shallowly, not more deeply.**

Predicted effect of shipping the refit: **neutral to slightly worse**, about
**-0.12 %** in cost units.

> The honest reading is uncomfortable and should be recorded as such: the
> merged curve's advantage is not that it prices head steps correctly. It is
> that two fitting errors in the cheap depths accidentally purchased extra
> conservatism at depth 2, and on prose that conservatism is worth more than
> the pricing accuracy it cost. The refit is more correct and predicted no
> better. If the advisor wants the *mechanism* rather than the *coefficients*,
> the thing to ship is an explicit depth-2 conservatism term on top of the
> refit — not the raw refit, and not the shipped fit's bug.

### 4.6 Predicted depth histograms before timing (pre-registration requirement)

Simulated round mix under the prior accept process, stated here before the Q1
arms were read:

```text
  shipped   overwhelmingly d2, thin d1/d3 tails      (matches Hp3: d2 237/246)
  refit     mass shifts d2 -> d3, small round-count drop
  flat 0.18 broad d1..d4 spread                      (matches Sp3: 19/138/67/21)
  flat 0.20 broad, shifted shallower than 0.18       (matches S20p: 45/138/51/16)
```

The simulator reproduces the qualitative shape of all three *measured*
histograms it can be checked against, which is the evidence that its refit
prediction is worth anything.

## 5. Q3 — 0.18 vs 0.20 settled analytically, then checked against the clock

### 5.1 The prediction, made from the gate algebra alone

For a flat `h`, `threshold(d) = h * (1 + E_d)/(1 + d*h)`. Raising `h` from 0.18
to 0.20 raises the threshold at every depth, but not uniformly — the `1 + d*h`
denominator grows with depth, so the *relative* tightening is largest at the
depths where `E_d` is still small. Evaluating the gate under the prior accept
process:

| h | opens to depth | slack at d=3 |
|---|---|---|
| 0.18 | 4 | **+0.0170** (opens) |
| 0.20 | 3 | **-0.0081** (closes) |

So the whole difference between the two scalars is predicted to be **a single
gate flip at depth 3**. There is no continuous re-tuning story: rounds whose
reach lands in the narrow band between the two thresholds stop asking for a
fourth draft, and — because `reach` is a product of per-position accept
estimates and the streak counter resets — those rounds mostly fall all the way
back to depth 1 rather than settling at depth 3.

Predicted direction: **0.20 is faster on prose**, because prose reach rarely
justifies depth 4.

### 5.2 The measurement agrees, in sign and in size

Sp3 (flat 0.18) vs S20p (flat 0.20), same host, same golden, 512 decode tokens:

```text
  measured decode s/tok:  0.041670  ->  0.041347     = -0.782 %   (0.20 faster)
```

Predicted under each pricing vector:

| pricing vector | predicted | measured | sign |
|---|---|---|---|
| shipped fit | -1.028 % | -0.782 % | correct |
| refit | -1.256 % | -0.782 % | correct |
| flat 0.18 | -0.738 % | -0.782 % | correct |
| flat 0.20 | -0.951 % | -0.782 % | correct |

Every pricing vector gets the sign right and lands within ~0.5 pp of a 0.78 %
effect. The analytic prediction was made from the gate rule, not fitted to this
outcome.

### 5.3 The mechanism is visible in the histograms

The predicted "single flip at d=3, falling back to d=1" is exactly what the
measured round mix shows:

| depth | flat 0.18 (Sp3) | flat 0.20 (S20p) | change |
|---|---|---|---|
| d1 | 19 | 45 | **+26** |
| d2 | 138 | 138 | 0 |
| d3 | 67 | 51 | -16 |
| d4 | 21 | 16 | -5 |
| **rounds** | 245 | 250 | +5 |

26 rounds move out of d3/d4 and land in d1, d2 is untouched, and the price is 5
extra rounds. That is a gate flip, not a re-tuning — a continuous parameter
change would have smeared mass between adjacent depths and moved d2.

**Q3 is settled: 0.20 beats 0.18 on prose, by ~0.8 % of decode cost, via one
gate flip at depth 3.** Note the practical consequence for Q1: the scalar arm in
this experiment is pinned at `h = 0.18` because that is the retired shipped
scalar the assignment names as the comparison point. The evidence here says
0.18 is *not* the best scalar, so the Q1 headline is measured against a scalar
baseline that is itself about 0.8 % of decode cost off its own optimum. Any
curve-vs-scalar margin below ~1 % should be read with that in mind, and I report
it rather than quietly switching the arm after pre-registration.

## 6. Two corrections to prior campaign arithmetic

### 6.1 Prefill is inside the score (deliverable 12)

`research/e11-notes.md` r3 claimed the seed prefill sits outside the ranked
score. That is wrong and is now retracted in a dated block in that file. Both
timed legs of every pair include seed processing, and
`.parent_measured_seconds_per_token` is total leg seconds over decode tokens.

The size of the mistake, from these arms: prefill is 3.9944-3.9980 s per leg,
which is ~15.8 % of the slower leg and ~16.9 % of the MTP leg. So a decode-only
claim is **diluted** when it reaches the score:

```text
  dilution = spt_decode / spt_inclusive = 0.041670 / 0.049473 = 0.84228

  Hp3 -> Sp3:  +7.577 % decode  x 0.84228  =  +6.382 %  inclusive
  measured inclusive                        =  +6.378 %      (checks out)
```

Practical rule adopted for this experiment: cost-model work is done in
decode currency because that is where the mechanism lives, but **every number
promoted to a score claim is multiplied by ~0.842 first**, and the Q1 headline
is taken verbatim from `.parent_measured_seconds_per_token` with no subtraction
at all.

### 6.2 The r1 restatement double-charged the head

The r1 assignment's arithmetic restatement of the gate charged the head step at
depth `d` both inside the cumulative term and again in the marginal comparison.
Corrected form is the one transcribed in section 4.1: `H_d` covers strictly
`j < d`, and `h[d]` appears once, on the right-hand side. With the double
charge, the gate looks far more conservative than it is and the depth-2 knife
edge in section 4.5 disappears entirely — which would have hidden the actual
mechanism behind the merged curve's win.

## 7. Q1 — the curve transfers, and it transfers for a reason

### 7.1 Result on the marker base `e6e6f81`

Four prompt pairs completed before the GPU was released (see §7.5 for why the
run stopped at four). Every arm is clean on all of contract item 5: 512/512
tokens emitted, `all_tokens_matched=True`, `residual_divergence_count=0`,
`parity_all_ok=True`, `declared_rows == reference_checked_rows`,
`mlx_qwen_env` empty, `dirty=0`, pinned head on every leg, public-drift
tripwire passed, stall ratio 1.23-1.74x against the 4x guardrail.

| prompt | serial C | serial F | floor % | mtp CURVE | mtp FLAT18 | raw C | raw F | d_raw | g % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `english` (anchor) | 0.074378 | 0.074425 | 0.063 | 0.046269 | 0.049585 | 1.6075 | 1.5010 | +0.1066 | **+6.688** |
| `narrative` | 0.074399 | 0.074464 | 0.087 | 0.045837 | 0.048394 | 1.6231 | 1.5387 | +0.0844 | **+5.284** |
| `technical` | 0.074456 | 0.074553 | 0.130 | 0.044686 | 0.046511 | 1.6662 | 1.6029 | +0.0633 | **+3.923** |
| `dramatic` | 0.074301 | 0.074554 | 0.340 | 0.043463 | 0.047777 | 1.7095 | 1.5605 | +0.1490 | **+9.028** |

All ratios are 512-decode-token, prefill-inclusive, straight from
`.parent_measured_seconds_per_token` with no subtraction, per §1.4.

Held-out only (`narrative`, `technical`, `dramatic`; n=3, odd, so the median is
the single central order statistic):

```text
median(raw_p | CURVE)  = 1.666207    spread 1.6231..1.7095
median(raw_p | FLAT18) = 1.560481    spread 1.5387..1.6029
headline delta         = +0.105726   (+6.775% of FLAT18)
g_median               = +5.284%     spread +3.923..+9.028%
curve wins on          = 3/3 prompts
```

Including the in-sample anchor (n=4, even, so the median is the mean of the two
central order statistics — the even-n rule the board uses). The two arms sort
differently, so their central pairs are different prompts:

```text
CURVE  sorted: 1.607516 english < 1.623115 narrative < 1.666207 technical < 1.709516 dramatic
               median = (narrative + technical)/2 = (1.623115 + 1.666207)/2 = 1.644661
FLAT18 sorted: 1.500951 english < 1.538689 narrative < 1.560481 dramatic  < 1.602927 technical
               median = (narrative + dramatic)/2  = (1.538689 + 1.560481)/2 = 1.549585

headline delta = +0.095076   (+6.136% of FLAT18)
g_median       = (5.2843 + 6.6880)/2 = +5.986%
curve wins on  = 4/4 prompts
```

Two things to be explicit about, because they are easy to misread:

- `median(raw_p | CURVE)` and `median(raw_p | FLAT18)` are medians taken
  *independently per arm*, so they are contributed by different prompts. That is
  correct — it is exactly what the ranked aggregation does — but it means the
  headline delta is **not** any single prompt's `d_raw`.
- `g_median` is the median of the per-prompt `g` values, a different aggregation
  from the ratio of the two medians. Both are reported; they differ slightly
  (+5.986% vs +6.133%) because median does not commute with the ratio.

### 7.2 Against the pre-registered stop rules

`g_median` on the held-out set is **+5.284%**, which is **>= 4%**: by §1.5 the
lever is real at scale.

On the anchor's influence, precisely. Sorted, the four `g` values are
`3.923 (technical) < 5.284 (narrative) < 6.688 (english) < 9.028 (dramatic)`, so
the in-sample anchor is the *second-strongest* prompt and it does inflate the
all-4 median: dropping it moves `g_median` from +5.986% down to +5.284%.
**The headline I am reporting is therefore the lower, held-out one**, and the
anchor is excluded from it by the pre-registered design in §1.2 rather than
after the fact. The all-4 figure is published alongside only for completeness.

The weakest prompt, `technical` at +3.923%, is on its own below the 4% line. It
is still 30x its own serial-leg noise floor (0.130%), so the *sign* is not in
question anywhere; only the magnitude varies by register.

### 7.3 Advisor predictions

| # | prediction | outcome at n=4 |
|--:|:--|:--|
| 1 | `g_median` between 1.5% and 4% | **falsified** — +5.284% held-out, +5.986% all-4, and every single prompt is above 3.9% |
| 2 | at least one prompt shows no benefit or slight regression | **falsified so far** — 4/4 wins, min +3.923% at 30x its own noise floor |
| 3 | non-monotonicity is an identification failure, not forcing | **confirmed** (§4.2) |
| 4 | monotone refit goes shallower, mass M=4 -> M=3, not opening M=5 | **refuted in the specific claim** (§4.3): the refit opens d3, it does not close to M=3. The *spirit* — shallower — is what the shipped curve already does |
| 5 | 0.18-vs-0.20 gap reproduces analytically within 2x | **confirmed** (§5): measured -0.782%, predicted -0.738..-1.256% |
| 6 | at least one of the above is wrong | **confirmed** — three are |

Prediction 1 is falsified in the *favourable* direction, which is the outcome
that most deserves scepticism. §7.4 is the reason to believe it anyway.

### 7.4 Mechanism: the win is saved verify rows, not accepted tokens

The naive story — "the curve accepts more, so it is faster" — is false here, and
measurably so. On **every** prompt the scalar-0.18 arm accepts **more** draft
tokens than the curve and is **slower**:

| prompt | g % | extra rows F-C | extra rows % | extra accepts F-C | extra rounds F-C |
|:--|--:|--:|--:|--:|--:|
| `technical` | +3.923 | 56 | +7.80 | +11 | -11 |
| `narrative` | +5.284 | 58 | +7.86 | +3 | -3 |
| `english` | +6.688 | 82 | +11.04 | +1 | -1 |
| `dramatic` | +9.028 | 86 | +11.78 | +15 | -16 |

Sorted by `g`, the extra-verify-row column is **monotone**:
`Spearman(g, extra rows %) = +1.000` on n=4. Sorted by acceptance advantage it
is not (`+0.400` for rows-per-accept).

So the curve's advantage is the *redundant target verification work it declines
to do*. It spends **more rounds** (each cheap: one extra round is one extra
`M=2..3` verify plus round overhead) to spend **far fewer rows** (each
expensive: rows at `M=4,5` hit `_m<T,5,3>` with two weight streams). The scalar
buys a handful of extra accepted tokens at a price of 56-86 extra verify rows,
and that trade loses on all four prompts.

This also predicts the register sensitivity correctly. The curve's depth
histogram is nearly invariant across prompts — `max_d=3` on all four, mean depth
2.020/2.037/2.055/2.093 — because the gate refuses depth 4 structurally. The
scalar's histogram is not: mean depth 2.367/2.317/2.455/**2.709**, and on
`dramatic` it reaches d5 nine times. **The curve's win is large exactly where
the flat threshold over-drafts most.** That is a causal story with a
one-parameter summary, not a per-prompt coincidence, and it is why I do not
expect the sign to flip on the remaining four prompts.

Note the corollary for Q4/§6 of the brief: the depth-4 gate being "closed by
only 6.9%" is not obviously a defect to fix. On this evidence, rounds that reach
`M=5` are exactly the rounds that cost the scalar its lead.

### 7.5 Why the run stopped at four prompts

Not a stop-rule trigger and not a failure. The advisor's r1 feedback
(comment `5316027151`, 2026-08-17T12:27:41Z) established two things mid-run:

1. The live advisor base moved `e6e6f81` -> `b85e782`, and the move landed on
   `segmentedStreakGate = 3 -> 2` plus `qmv_fast_crossrow_affine4_g64_m<T,8,4>
   -> <T,8,3>`. The advisor's instruction: the blinded within-session contrast
   stays valid on a single fixed base and should be reported as measured, but
   **any headline proposed for submission must be measured on `b85e782`.**
2. askeladd is first in the GPU queue with a 512-token four-phase ABBA, and I
   was told to release the GPU when the current *timed block* completed.

The `dramatic` block completed at 12:37:35Z. Spending a further ~71-76 min of
exclusive GPU to take prompts 5-8 to n=8 **on a base that cannot carry a
submission headline** is worse campaign economics than releasing the GPU and
re-measuring on `b85e782`, where the same GPU-minutes satisfy both the n>=6
contract and the submission requirement. So the run stopped at the block
boundary, as instructed.

### 7.6 Salvage and re-measure cost for the `b85e782` revision

Measured wall clock on this host, from each arm's `meta.txt` (`started` ->
`finished`, inclusive of the harness's separate 40 C cool gate before *each*
leg):

| prompt | span | note |
|:--|--:|:--|
| `english` | 981 s (16.35 min) | |
| `narrative` | 1016 s (16.93 min) | |
| `technical` | 1105 s (18.42 min) | |
| `dramatic` | 1143 s (19.05 min) | |
| **mean** | **1061 s (17.69 min)** | 2 arms + 2 cool gates |
| **max** | **1143 s (19.05 min)** | budget against this |

One prompt pair does not fit a single 30-min `run_job` twice over, so the
practical unit is **one prompt per job**, `timeout_seconds = 1740`.

| scope on `b85e782` | GPU wall time | jobs |
|:--|--:|--:|
| 4 remaining prompts only (invalid — mixes bases) | — | — |
| 6 prompts (contract minimum) | 106-114 min | 6 |
| 8 prompts (contract preference) | 142-152 min | 8 |
| + release-build both arms once | +~10-14 min | — |
| **8-prompt re-measure, total** | **~152-166 min** | 8 + build |

**Salvageable without re-measurement** (base-independent):

- §4 in full: the gate algebra, the non-monotonicity diagnosis, the monotone
  refit vector, the per-candidate predicted histograms, the pricing test. These
  are algebra over a logged reach distribution plus a cost model; they do not
  depend on the streak gate.
- §5 in full: the 0.18-vs-0.20 gate-flip mechanism and its analytic prediction.
- §6 in full: both arithmetic corrections.
- §7.4's mechanism: the row-economy story is a statement about what the two
  gates do to verify width, which the streak-gate change modulates but does not
  invert.
- The whole instrument: prompt set, goldens, `e17-run.sh`, `e17_analyse.py`,
  `e17_wandb.py`, and the ABBA order-control design all replay unchanged.

**Needs re-measurement on `b85e782`:**

- Every number in §7.1 and §7.2, because both halves of a deliberately-paired
  kernel+gate change moved.
- Anything about depth >= 4 firing rates, `M >= 5`, or `M = 8`. My
  `M >= 5 = 0.00%` figure and the `{1:2, 2:231, 3:13}` histogram were taken at
  gate 3 and now understate firing. The `dramatic` FLAT18 arm already reaches d5
  nine times at gate 3, so gate 2 will move this further.
- The projection table `R'`, since it is parameterised by `p = P/D` and the
  decode leg changes.

**One caveat that survives the base move** (from §5): the pinned 0.18 scalar is
~0.8% off its own optimum, because 0.20 is faster. The counterfactual in §7.1 is
therefore the arm the curve actually replaced, not the best available scalar. A
fair statement of the curve's advantage over a *well-chosen* scalar is roughly
`g_median - 0.8pp`, i.e. ~4.5% held-out rather than ~5.3%. That still clears the
4% stop rule, but the caveat must travel with the number.

### 7.7 Order-of-information disclosure (brief §7)

The brief quarantined third-party timing evidence about my own curve and asked
me to state whether my pre-timing falsification statement was signed at the
moment I read it. **It was.** §1.6 of this file was committed in `82ce6ce`,
and the first timed arm (`english-CURVE`) started at 2026-08-17T11:19:41Z; the
quarantined section arrived in the advisor comment timestamped 12:27:41Z, i.e.
after three of four prompt pairs were already on disk. The prompt set, its
generation procedure, and the arm order were all fixed in §1.2 in the same
commit. No arm was added, dropped, reordered or re-run after reading it.

Unavoidable disclosure: the quarantined content arrived inside a single
`get_prs` payload together with the rest of the comment, so it was not possible
to read §1-§6 of the feedback without also receiving §7. It changed nothing —
by then the design was frozen and the timing was three-quarters done.

The observation itself (alphonse, E16/PR #18: the curve may cost ~1% of the MTP
leg on `e6e6f81`, n=1, 64 tokens, M4 Pro) is **not** reproduced by this
experiment: at 512 tokens, n=4, on the M5, the curve is faster on 4/4 prompts by
+3.9% to +9.0%. The most likely reconciliation is window length — at 64 tokens
his MTP leg is ~5.6 s against a ~4.0 s prefill, so his ratio is prefill-dominated
and 10-16 rounds is far too few rounds for the gate's steady-state behaviour to
appear. That is a hypothesis, not a finding; the brief asks for the fold-in at
n=3 and 512 tokens after Q1, which is now a follow-up. PR #18 is outside my
launch isolation scope, so I treat it strictly as advisor-relayed context and
have not inspected it.
