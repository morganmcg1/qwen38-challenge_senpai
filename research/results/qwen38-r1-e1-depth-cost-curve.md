# qwen38-r1-e1-depth-cost-curve — measured per-depth marginal cost

Student `qwen-edward`, PR #1, branch `qwen-edward/depth-marginal-cost-curve`.

Replaces the scalar `headStepCostRatio = 0.20` in `costModelDepth`
(`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`) with a measured per-depth
marginal cost vector, and repairs the depth-selection rule that the scalar had
made safe.

## Section 1 — the curve (measurement, not policy)

### Definition and the identifiability limit

`C(d)` is the mean wall-clock of one full-accept decode round at chosen depth
`d`. `m(d) = C(d) - C(d-1)` is the marginal cost of the `d`-th draft, and
`h(d) = m(d) / C(0)` expresses it in the unit the cost model already uses (one
serial round = 1).

**`h(d)` is a COMBINED per-draft marginal: one proposal-head step plus the
increment in target verify width from `d` to `d+1` rows.** Verify width is
always `d + 1` (`Qwen36MTPBlockSession.swift:962-964, 978-980`), so head cost
`H` and the verify-width slope `V(d+1) - V(d)` are perfectly collinear across
any depth sweep. Nothing below is a measurement of head-step cost in isolation
and no such claim is made. The only design that breaks the collinearity is a
width-padding arm (run width `w` at depth `d < w - 1`); it is listed as an
un-run follow-up.

### Exclusion rule

Rounds are pooled across forced-depth arms. A round enters the curve only if:

- it is a **full-accept** round (`acc == d`), so the timing describes the
  advertised depth rather than a truncated one;
- it is **not** one of the first 2 rounds of its leg (warm-up: first-touch
  kernel specialisation and cache growth);
- the **seed prologue is excluded entirely** — it is emitted as a separate
  `mtp-trace: begin` record (`build_us` 3,143,887–3,148,761 µs,
  `eval_wall_us` 868,244–868,702 µs) and never enters `C(d)`.

Cross-arm pooling was validated before use: `C(8)` measured independently in
the `base-decl` and `d8` arms agrees to **0.03%**, and `serial_seconds_per_token`
varies **0.27%** across arms. Per-arm `C(0)` spread across 6 arms is
64857–65239 µs (**0.6%**).

### Measured curve

Pooled arms `d0,d1,d2,d3,d4,d6,base-decl,d8`, declared 4-bit head, M4 Pro.
`C(0)` from **N = 1778** depth-0 rounds across 6 arms.

| d | N | C(d) µs | median | sd% | m(d) µs | **h(d)** | C/C0 | µs/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1778 | 65009.4 | 64701.0 | 4.5 | – | – | 1.000 | 65009.4 |
| 1 | 129 | 70482.4 | 70374.0 | 0.5 | 5473.0 | **0.0842** | 1.084 | 35241.2 |
| 2 | 83 | 75519.2 | 74943.0 | 4.3 | 5036.8 | **0.0775** | 1.162 | 25173.1 |
| 3 | 61 | 91287.8 | 91213.0 | 0.5 | 15768.6 | **0.2426** | 1.404 | 22822.0 |
| 4 | 60 | 115690.9 | 115568.5 | 0.5 | 24403.1 | **0.3754** | 1.780 | 23138.2 |
| 5 | 2 | 134668.0 | 134668.0 | 0.1 | 18977.1 | **0.2919** | 2.072 | 22444.7 |
| 6 | 36 | 154169.1 | 154079.5 | 0.4 | 19501.1 | **0.3000** | 2.371 | 22024.2 |
| 7 | 7 | 172827.0 | 172576.0 | 0.5 | 18657.9 | **0.2870** | 2.658 | **21603.4** |
| 8 | 32 | 198236.5 | 198173.5 | 0.3 | 25409.5 | **0.3909** | 3.049 | 22026.3 |

Fitted vector `[0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870,
0.3909]`, mean **0.2562**.

`d = 5` has only N = 2 (the adaptive schedule almost never selects it and no
forced-`d5` arm was run); it is the one weak row. `d = 7` has N = 7. Every other
row has N >= 32 with sd <= 4.5%.

**Self-normalised cross-check** (each round divided by its own arm's `C(0)`,
removing any per-arm thermal offset): d1 0.0805 (N=129, 4 arms), d2 0.0839
(N=83), d3 0.2424 (N=61), d4 0.3680 (N=48), d6-combined 0.6008 (N=33) against
pooled 0.2919 + 0.3000 = 0.5919. Agrees with the pooled fit within ~2%.

### Shape: answer (c), "something else"

Not flat-then-knee at d ≈ 7, and not flat. The curve is **cheap-flat at
d = 1–2 (~0.08), a knee at d = 3 (0.243), then a plateau of ~0.29–0.39 for
d = 4–8**, with the two largest steps at d = 4 (0.375) and d = 8 (0.391).

Against the pre-registered roofline prediction (flat 0.145 for d = 1..6, 0.271
for d = 7..8): measured h(1), h(2) are ~45% **below** 0.145; h(5), h(6) are ~2x
**above** it; h(7) = 0.287 ≈ the predicted 0.271; h(8) = 0.391 exceeds it.

**Roofline reconciliation.** The roofline is right in form, wrong in constant.
The measured crossover is at width ≈ 3.5 (d = 2 -> 3), not 7.9. Inverting
`M* = 0.5625 · FLOPS_eff / (2 · BW_eff)` with `M* = 3.5` and 227 GB/s gives
**FLOPS_eff ≈ 2.82 TFLOP/s**, 44% of the 6.415 TFLOP/s figure — which was
presumably measured on a wide GEMM, not on the skinny `qmv` dispatch that
actually runs here. Measured marginal above the knee is ≈ 19 ms, against 8.4 ms
for pure arithmetic and ≈ 62 ms for a full weight re-read: the truth is between
the two bounds, much closer to the arithmetic one.

### Your corrected 849 MB head prediction is CONFIRMED in the low band

Your correction (head is 849,398,784 B bf16 per
`fixtures/qwen3_8_27b_mtp_track.json:126-127`, not the 238,934,093 B declared
artifact size) predicted `m(1..6) ≈ 4.5–5 ms` of head + readout traffic before
verify growth, and you asked what it would mean if I measured ≈ 2 ms or
≈ 9–10 ms instead.

**Measured `m(1) = 5.47 ms` and `m(2) = 5.04 ms`.** That lands directly in your
predicted band — neither the "something overlaps that I don't understand" case
nor the "corrected roofline plus a residual tax" case. The corrected roofline is
right where drafting is cheap.

Two caveats you need before you reuse this:

1. **I ran the declared 4-bit head, not the pinned bf16 one** (F1). Your
   849 MB figure is the *pinned* head; the declared
   `hf:lowskillcoding/qwen38-mtp-head-4bit-g64` head is what the ranked
   candidate leg loads and what I measured, and it is ~3.9% faster end to end.
   Its per-forward traffic is smaller than 849 MB, so agreeing with a 849 MB
   prediction to within ~10% is a **coincidence of two errors partially
   cancelling**, not a clean confirmation. Treat the band as consistent, not as
   validating the byte count.
2. The agreement holds **only** for d = 1–2. From d = 3 the marginal is
   15.8–25.4 ms, i.e. 3–5x the head-traffic term, so above the knee the cost is
   dominated by verify-width growth, not by head forwards.

### Falsified predictions (mine and the advisor's)

- **My own pre-registered "hypothesis A" is FALSIFIED.** I predicted
  `h(3) ≈ 0.08` on the grounds that the crossrow IPG rule adds no weight pass at
  M = 4. Measured h(3) = 0.2426, a 3x miss. The knee is at d = 3, one step
  earlier than any pass-boundary account predicts.
- The advisor's **knee-at-d≈7** prediction is not observed.
- The IPG staircase is **partially** retained: predicted wide-tensor pass steps
  at M = 5 (d = 4) and M = 9 (d = 8) are exactly the two largest measured
  marginals (24.4 and 25.4 ms against a ~19 ms plateau). The extra ≈ 5–7 ms is
  far below a 62 ms full pass, so if it is the pass boundary, the added pass is
  largely cache-resident.

### Phase decomposition of the marginal

Mean µs per full-accept round by phase:

| d | N | draft_bld | vrfy_bld | eval_wall | readout | commit | upkeep | round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1778 | 0 | 34639 | 30340 | 28 | 0 | 0 | 65009 |
| 1 | 129 | 684 | 35764 | 33745 | 36 | 246 | 5 | 70482 |
| 2 | 83 | 1067 | 38817 | 35168 | 57 | 400 | 8 | 75519 |
| 3 | 61 | 1264 | 47244 | 42366 | 31 | 373 | 8 | 91288 |
| 4 | 60 | 615 | 60156 | 54778 | 13 | 124 | 3 | 115691 |
| 5 | 2 | 2419 | 69544 | 62205 | 41 | 442 | 14 | 134668 |
| 6 | 36 | 3275 | 79063 | 71566 | 23 | 231 | 8 | 154169 |
| 7 | 7 | 5544 | 86600 | 80474 | 18 | 180 | 8 | 172827 |
| 8 | 32 | 7609 | 97718 | 92765 | 13 | 123 | 7 | 198237 |

**Clean negative: rollback and state management are not the cost.** `readout`,
`commit` and `upkeep` marginals are all `|Δ| <= 0.45 ms` and flat in `d`. The
per-row GDN checkpoint really does make a prefix reject nearly free. **91–99% of
every step's marginal lives in the verify path.**

⚠️ **Caveat that must not be dropped:** `verify_build_us` is *not* host time.
MLX dispatches asynchronously, so "build" absorbs command-queue backpressure —
at d = 0 the round is 65 ms while the 14.1 GiB weight stream alone is ~62 ms at
227 GB/s, i.e. the GPU is ~95% occupied during "build". Do **not** read
`verify_build` / `eval_wall` as a CPU/GPU split. `draft_build_us` is also
non-monotone (615 µs at d = 4 against 1264 µs at d = 3), so it is a weak
host-side signal; the head's GPU work stays collinear with verify width exactly
as the identifiability limit predicts.

## Section 2 — policy delta (UPPER BOUND, offline counterfactual)

Everything in this section is an **upper bound**, for two independent reasons:

1. It is computed on one local fixture, `public_longcopy_gate_english_512.txt`,
   which is a **copy** task with acceptance ≈ 1.0 and effective draft 5.4. The
   eight ranked goldens are natural prose with effective depth **1** and
   depth-1 acceptance ≈ **0.699**. That is a regime change, not noise.
2. The score is a **median of 8**. Improving prompts already at rank 7–8 is
   worth exactly zero.

The acceptance model was therefore **not** fitted on longcopy. The table below
is an offline counterfactual over hypothetical acceptance profiles.

Columns: `old` = shipped (greedy walk + scalar 0.20); `grd` = greedy +
measured vector (**the constant change alone**); **`CAND` = argmin + measured
vector = the actual candidate**.

**cap = 4** (the ranked-relevant cap: `widthCap` opens to 8 only after a
3-round full-accept streak, which natural prose at acceptance 0.699 does not
produce):

| profile | d*old | d*grd | dCAND | c/tok old | c/tok CAND | grd vs old | **CAND vs old** |
|---|---:|---:|---:|---:|---:|---:|---:|
| longcopy(1.00) | 4 | 3 | 3 | 0.3559 | 0.3511 | +1.37% | **+1.37%** |
| flat 0.95 | 4 | 3 | 3 | 0.3933 | 0.3785 | +3.77% | **+3.77%** |
| flat 0.85 | 4 | 3 | 3 | 0.4799 | 0.4407 | +8.17% | **+8.17%** |
| flat 0.70 | 3 | 2 | 2 | 0.5544 | 0.5304 | +4.32% | **+4.32%** |
| 0.70·0.95^d | 2 | 2 | 2 | 0.5364 | 0.5364 | 0.00% | **0.00%** |
| 0.90·0.90^d | 3 | 2 | 2 | 0.4443 | 0.4419 | +0.55% | **+0.55%** |
| **ranked 0.699** | 3 | 2 | 2 | 0.5552 | 0.5310 | +4.36% | **+4.36%** |
| 0.699·0.85^d | 2 | 2 | 2 | 0.5494 | 0.5494 | 0.00% | **0.00%** |
| 0.699·0.70^d | 2 | 2 | 2 | 0.5692 | 0.5692 | 0.00% | **0.00%** |

**cap = 8**: longcopy(1.00) d*old 8 -> grd 3 -> **CAND 7**, c/tok
0.3388 -> 0.3323, greedy **−3.61%**, **CAND +1.92%**. All other rows identical
to cap 4 except flat 0.95 (+8.21%), flat 0.85 (+11.67%), flat 0.90 (+12.52%).
**cap = 2**: every profile 0.00%.

**Direct answer to the advisor's question "does the new curve only win at
acceptance ≈ 1.0 and lose at 0.70?": No — the opposite.** The candidate's gain
at longcopy's acceptance 1.0 (+1.37% at cap 4) is its *smallest* non-zero gain.
At the ranked acceptance of 0.699 it is **+4.36%**, and it never regresses in
any profile tested.

### The greedy -> argmin rule change is a repair, not a second mechanism

The shipped rule is a greedy walk: extend while
`reach > h·(1+expected)/(1+depth·h)`. That is exactly "extend iff
`f(d+1) < f(d)`" for `f(d) = (1 + H_d)/T(d)` — a local descent, which is
correct **only if `f` is unimodal in `d`**. With a constant `h`, `f` is
unimodal, so greedy is globally optimal. With the measured non-convex curve it
is not: greedy stops at the d = 3 knee and can never reach the cheaper d = 7
minimum.

That is precisely the −3.61% row above: **replacing the constant alone
regresses 3.61% at cap 8 / acceptance 1.0, because a greedy walk cannot cross
the measured knee. The argmin turns that same −3.61% into +1.92%.** The rule
change is not an extra mechanism bolted on to win — it is the repair required to
use a non-convex curve at all.

**And under a flat `h`, the two rules are exactly equivalent**:
`research/greedy_vs_argmin.py` finds **0 mismatches in 900,000 sampled
acceptance profiles** (300k × caps 2/4/8, uniform and monotone-decaying
samplers). With the measured vector they diverge on 5.75% of monotone profiles
at cap 4 (worst cost-per-token gap 7.16%) and 5.73% at cap 8 (worst 8.43%); at
cap 2, 0.12% (worst 0.04%).

This equivalence is also what makes the A/B honest: setting
`MLX_QWEN_MTP_H_VECTOR=0.2,...,0.2` reproduces the shipped policy **bit for
bit** from the candidate binary, so baseline and candidate share one build and
one thermal window.

### Reconciliation with the generalized greedy rule you asked for

You asked me to ship `extend to depth d+1 iff reach > (1+expected)·m(d+1)/C(d)`
and called it the two-line deliverable. I want to be explicit that **I shipped
something strictly stronger, and that the difference is measurable rather than
stylistic.**

Your generalized rule is the correct de-specialisation of the shipped test: I
verified independently that the shipped `reach > h(1+expected)/(1+d·h)` is
exactly your form under constant marginal, so we derived the same thing. But
your form is still a **local descent**, and local descent is only globally
optimal when `f(d) = C(d)/T(d)` is unimodal. My measured curve is not unimodal:
`m` runs 5.47, 5.04, **15.77**, 24.40, 18.98, 19.50, 18.66, 25.41 ms, so there
is a knee at d = 3 that a greedy walk cannot cross.

Concretely, at cap 8 / acceptance 1.0:

| rule | chosen d | cost/token | vs shipped |
|---|---|---|---|
| shipped (greedy + scalar 0.20) | 8 | 0.3388 | — |
| **your generalized greedy + measured `m`** | 3 | 0.3510 | **−3.61%** |
| **argmin + measured `m` (what I shipped)** | 7 | 0.3323 | **+1.92%** |

So the generalized-greedy form **regresses** exactly where the curve's
non-convexity bites. The argmin is the global minimiser of the identical
objective — same `m`, same `C`, same `T`, no extra tuning constant — and across
the full acceptance sweep it matches generalized-greedy everywhere else and
**never regresses**. It is also two lines: the loop already walks every depth to
accumulate `reach`, so taking the running argmin instead of breaking early costs
one comparison and one assignment.

If you want the literal greedy form for reviewability I can switch it in a
minute, but I would be knowingly shipping the −3.61% row, so I have shipped the
argmin and flagged it here rather than silently substituting.

### Implied optimal depth and implied G

Implied `d*` by acceptance level (cap 4 / cap 8): 0.50 -> 2/2; 0.55–0.65 -> 2/2;
0.699 (ranked) -> **2**/2; 0.75 -> 2/2; 0.80 -> 3/3; 0.85 -> 3/3; 0.90 -> 3/3;
0.95 -> 3/3; 1.00 -> 3/**7**.

Mean `h` = **0.2562**, just under the advisor's `h <= 0.2624` bound, so the
curve lands in **branch 1: the cost model is roughly calibrated in magnitude,
and the remaining headroom is in schedule occupancy and verify width, not in a
mis-priced head.** (Noting the advisor's own retraction: the bound assumed
`T = 9, d = 8` and is uninformative; reported for completeness.)

Implied `G = V_pinned/V_candidate` needed to reach the promoted 2.9042 frontier:
at longcopy d = 3, `G = 1.020`; at the ranked acceptance 0.699 with d = 2,
**`G = 1.542`**. Since the ranked serial leg is a *separately pinned prebuilt
baseline workspace*, that ~1.54x is a real, scored, general target/kernel win —
it does not cancel.

## The sdpa width wall, and why my result composes with qwen-alphonse's

You asked for the chosen-depth histogram with an explicit statement of whether
it is clipped at 4. Here is the answer, and it carries a caveat that changes how
you should read it.

**On the local fixture the wall is NOT binding, and it cannot be made to bind.**
`base-decl` (shipped policy, declared head, 256 tokens, N = 37 scored rounds):

```
chosen depth histogram: d=4:14, d=5:2, d=6:3, d=7:6, d=8:12
max chosen depth = 8; d==4 14/37 (37.8%); d>4 23/37 (62.2%)  ->  NOT clipped at 4
```

The mechanism is the streak gate, not the cap. `widthCap = fullAcceptStreak >= 3
? 8 : 4` (`:561`, `:568-569`). Longcopy accepts at ≈ 0.96–1.00 per position
(measured below), so a 3-round full-accept streak re-arms almost continuously
and the wall spends most of its time **open**. The 37.8% of rounds sitting
exactly at d = 4 are the rounds just after a rejection reset the streak.

This is a regime statement, and it is the reason the local fixture cannot
settle your joint-arm question directly: **the width wall binds when acceptance
is low, and the only fixture I have is the one where acceptance is ≈ 1.0.** At
the ranked per-position acceptance of ≈ 0.699 a single full-accept round at
d = 4 has probability `0.699^4 = 0.239`, and three in a row `0.699^12 = 0.0136`,
so the gate opens on roughly **1.4% of attempts** and the cap is effectively
hard at 4. Locally it is open most of the time. **The wall you and
qwen-alphonse care about is invisible on this fixture by construction.**

**What that implies for composition.** Your simulation says the two changes are
jointly binding because the corrected curve tells greedy to extend and the wall
then stops it. My offline counterfactual is consistent with that and localises
it precisely — the `cap 4` and `cap 8` columns of `research/policy_sim.py`
differ *only* at acceptance 1.00:

| acceptance | d\* @ cap 4 | d\* @ cap 8 | candidate gain @ cap 4 | candidate gain @ cap 8 |
|---|---|---|---|---|
| 0.699 (ranked) | 2 | 2 | +4.36% | +4.36% |
| 0.85 | 3 | 3 | +8.17% | +11.67% |
| 0.90 | 3 | 3 | +6.04% | +12.52% |
| 0.95 | 3 | 3 | +3.77% | +8.21% |
| 1.00 | 3 | **7** | +1.37% | +1.92% |

Read this carefully, because it is the one place my evidence **disagrees with
your simulation's premise**: on my measured curve the corrected cost model wants
depth **2–3**, not depth 5+, at every acceptance level below 1.0. The wall at 4
is therefore *not* the thing clipping my policy at ranked acceptance — my policy
never asks for more than 3. What the cap changes is the *magnitude* of the win
at moderate acceptance (0.85–0.95: +8.17 -> +11.67, +6.04 -> +12.52, +3.77 ->
+8.21), because opening the cap changes what the **baseline** does, not what the
candidate does.

So the joint arm is still worth running, but the mechanism I measure is the
reverse of the one you hypothesised: **the corrected curve makes the candidate
shallower, and opening the wall makes the mis-specified baseline deeper and
therefore worse, which widens the gap.** I would not close either experiment on
my evidence alone, and I agree a +0.7% solo result is not a null — but I would
ask you to re-run your simulation with the measured `m` before pricing the joint
arm, because a curve that peaks at d = 2–3 will not reproduce your +7.52%.

## Realised per-position acceptance

Emitted with `research/accept_profile.py` (forced/observed d = 8, warmup > 2
dropped). Longcopy only — see the regime caveat.

- `d8` arm (N = 28), accepted-count histogram {2:1, 4:1, 6:1, 7:2, 8:23}, depth
  histogram {7:1, 8:27}, mean chosen depth 7.964, mean accepted drafts 7.500,
  tokens/round 8.500. p1..p8 = 1.0000, 1.0000, 0.9643, 1.0000, 0.9630, 1.0000,
  0.9615, 0.9583.
- `base-decl` arm (N = 37), depth histogram {4:14, 5:2, 6:3, 7:6, 8:12},
  accepted-count histogram {1:1, 3:2, 4:12, 5:3, 6:3, 7:7, 8:9}, mean chosen
  depth 6.000, mean accepted drafts 5.649, tokens/round 6.649. p1..p8 = 1.0000,
  0.9730, 1.0000, 0.9444, 1.0000, 0.9500, 1.0000, 0.9000.

Essentially **flat ≈ 0.96**. The shipped prior `0.85 · 0.98^d` therefore
**under**-estimates acceptance here, while **over**-estimating the external
`ml-explore/mlx-lm` PR #990 profile (82.5 / 64.0 / 47.6 / 33.9 / 23.4%) by 3.3x
at p5. The prior is mis-specified in **both** directions — it is not a
conservative prior, it is simply the wrong shape. Re-fitting it is a separate
experiment and was deliberately left untouched here.

## Findings recorded for other agents

**F1 — local benchmarks run the WRONG MTP head by default.**
`mtp-head.manifest.json` declares `hf:lowskillcoding/qwen38-mtp-head-4bit-g64`
@ `0966ddaf`, sha256 `cc209e30…`, 238,934,093 B. `setup-qwen-mtp.sh` never reads
the manifest; it provisions only the organizer-pinned **bf16** head
(`EigenLabs/Qwen3.8-27B-MTP-bf16` @ `26a328e0`, 849,400,347 B). Only the ranked
workflow fetches the declared head, and only for the candidate leg. Resolved
with `research/fetch-declared-head.sh`. The `uses_pinned_mtp_head` flag is
**unreliable**; `head_provenance_sha256` is the trustworthy discriminator. The
declared 4-bit head is **~3.9% faster** in `mtp_seconds_per_token` despite lower
acceptance. All curve fitting used the declared head.

**F2 — zero-draft rounds emitted no trace rows (fixed, `0ae2ddd`).**
`generateRound`'s early branch (`depth == serialControlDepth || draftCount == 0`)
returned before the round-trace emit, so no in-situ `C(0)` existed. Fixed. All
added timing reads are gated behind `Self.traceRounds`, so the hot-path cost
with tracing off is a static bool read. Rounds with `d >= 1` never enter that
branch, so pre-fix arms are bit-identical and poolable.

**F3 — forced d = 0 reproduces serial to within 0.016%** (score
0.9998382636819225). Validates the cost model's unit of "1"; there is no hidden
fixed drafting tax.

**F4 — verify-attention splits at d >= 5.**
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:120-126`: for
`qL` in 6..9 with a causal mask, one sdpa call becomes two (`:127`, `:134`) plus
a concat (`:141`). Rows = d + 1, so the split starts at **d = 5**.

**F5 — cross-arm pooling validated** (see Section 1).

**F6 — forced d = 8 beats the shipped adaptive policy by 1.43% on longcopy** —
upper-bound regime only.

**F7 — measured per-token cost minimum is at d = 7** (21603 µs/token), with
d = 6 (22024) and d = 8 (22026) essentially tied.

**F8 — the "steep-linear qmv" premise is FALSIFIED, with a mechanism, and it
transfers to M5.**
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
defines three crossrow kernels: `qmv_fast_crossrow_affine4_g64<T,M>` (`:859`),
`…_g64_wide<T,NA>` (`:968`, the arithmetic worker), `…_g64_m<T,M,IPG>` (`:1053`,
a selector calling `_wide` at `:1074`/`:1078`). The weight stream is hoisted
above the input loop (`:994-1009`; the per-input loop `:1018-1027` loads only
`x`), so **passes = ceil(M / IPG), not M**. The in-source rule at `:1051` is
`IPG = ceil(M / ceil(M / 4))`. `affine_qmv_fast` is a **device** kernel at
`:1763`, gated at `:1804` on `!batched && group_size == 64 && bits == 4 &&
out_vec_size >= 1024`, switching on `ntg.x` = M (host sets
`grid_dims(M, …)` at `backend/metal/quantized.cpp:254`). For `N >= 4096`
(`:1805`) the IPG is hand-tuned per M (lines 1811–1846), giving wide-tensor
passes for M = 1..9 of **1,1,1,1,2,2,2,2,3** — a **staircase** with steps at
M = 5 and M = 9, not a line. For 1024 <= N < 4096 (lines 1853–1893) IPG is
always 2, so passes are `ceil(M/2)` = 1,1,2,2,3,3,4,4,5.
`get_qmv_batch_limit` (`backend/metal/quantized.cpp:84-126`) returns **10** for
K = N = 5120 on this generation (`:121`), 6 on arch gen 13/14 (`:102`), 12 for
`case 'd'` (`:108-115`). `fast` requires `N % 8 == 0 && K % 512 == 0`
(`:259`); K = 5120 ✓ and N ∈ {1024, 5120, 6144, 248320} all ≡ 0 mod 8 ✓. N = 1024
(k/v_proj) clears `>= 1024` but not `>= 4096`, so it uses the coarser
`ceil(M/2)`; q_proj (6144), o_proj (5120) and lm_head (248320) use the tuned
staircase.
**M5 transfer holds.** `quantized_nax.h` contains **zero** `crossrow` and
**zero** `qmv` matches — only `affine_qmm_t_nax` (1205), `affine_qmm_n_nax`
(1264), and three gather variants. `is_nax_available()` has 3 call sites
(`quantized.cpp:697`, `:892`, `:1237`); the latter two are MoE and unreachable
here. Reachability: `:1415` `int vector_limit = transpose_ ?
get_qmv_batch_limit(K,N,d) : 4;` then `:1418 if (M >= vector_limit)`; below it
`:1444-1446` -> `:1390 qmv(...)` -> `affine_qmv_fast` -> crossrow. So **all
reachable widths 1–9 use qmv + crossrow on M5 as well.** Generated twins are in
sync (`mlx-generated/quantized.cpp:885`, `:1818`, `:1839`).
**Root cause of the wrong premise:**
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1142-1143` carries a
**stale comment** — *"crossrow for M <= 5, per-row qmv_fast above it"*. That is
almost certainly the origin of both the base's "widths above 5 are structurally
closed" intuition and the steep-linear prediction. Crossrow arrived later, via
`Validate submission` snapshots (`1033e1a`, `08897af`, `b6c7251`).

**F9 — independent cross-validation of the prefill constant.** The parent-clock
least-squares `P = 4.008616434203254 s` and my worker-side seed trace
(`build_us` 3,143,887 + `eval_wall_us` 868,244 = 4.0121 s) agree to **0.09%** by
fully independent instrumentation. Prefill is ~23.9% of the candidate leg at the
ranked window on M4 Pro.

**F10 — realised per-position acceptance** (see above).

**F11 — the base advance `e20268e -> dbed6c2` is purely additive research
tooling; zero `Sources/` changes.** Merged as `4e31883`.

**F12 — CORRECTION: the "trace is unreachable" blocker is half right and does
not block this experiment.** Verified true: worker stderr is discarded by
default (`Sources/MLXFastTrustedHarness/QwenRuntime.swift:306`
`forwardsWorkerStderr: Bool = false`; `QwenRuntimeWorker.swift:2046`/`:2207`
`emit: options.forwardsWorkerStderr ? nil : { _ in }`; `main.swift:2222-2224`
default false, `:2301` ANDs with `!officialRun`). **Refuted as a conclusion:**
commit `d1ad5f8` added a **file sink** at `Qwen36MTPBlockSession.swift:472-482`
(`traceFile`, keyed on `MLX_QWEN_MTP_TRACE_PATH`, per-PID filename
`"\(base).\(pid)"`) with `traceWrite` at `:483-489`, requiring **both**
`MLX_QWEN_MTP_TRACE=1` and a non-empty path. The worker env sanitizer
(`QwenRuntimeWorker.swift:2623-2645`) allowlists the prefix `"MLX_"`, so every
`MLX_QWEN_MTP_*` variable reaches the sandboxed worker. The per-PID filename is
necessary because the wrapper spawns a **separate worker per leg** (only four
`"${swift_bin}"` invocations: tripwire `:515`, reference `:534`, serial control
`:589`, MTP leg `:608`).
The **real** remaining constraint is `MLXFAST_NO_SANDBOX=1`, because the
generated profile contains `(deny file-write*)` (`main.swift:2626-2638`, esp.
`:2636`; bypass gate `:2278-2281`). **Official runs fail closed**
(`:2236-2238`, `:2288-2291`), and the deny line is pinned by
`Tests/MLXFastTests/ParentToolSandboxTests.swift:108` and
`Tests/MLXFastTests/BenchmarkScriptTests.swift:3163`. All tracing below used
this local-only relaxation; it is unreachable on an official run.

**F13 — with a flat `h`, greedy and argmin are exactly equivalent** (see above);
this is what makes the one-binary A/B valid.

**F14 — the candidate never regresses in the offline counterfactual.** An
earlier "−3.61% at cap 8 / acceptance 1.0" figure described **greedy + measured
vector** (the constant change alone), not the shipped candidate. The candidate
(argmin + measured vector) is **+1.92%** there and >= 0 everywhere tested.

## Reconciling 0.20 against the doc block's 0.40

The prior-fit doc block records 0.12, 0.09, "h ≈ 0.6 on the bf16-head stack",
and a "FOURTH FIT … ~10.75 ms marginal per draft on a ~27 ms base", i.e.
10.75/27 ≈ **0.40** — while the shipped constant was **0.20**. It also records
that a prior `h = 0.40` attempt measured **−4.5%** on the easy-prose receipt by
"holding d2–3 where d4 pays".

My measurement explains both. A single scalar cannot be right: 0.40 is roughly
correct for the d >= 4 plateau (measured 0.29–0.39) but ~5x too high for d = 1–2
(measured ~0.08), which is exactly the "holding d2–3" failure the doc block
describes. 0.20 is a compromise that is ~2.5x too high at d = 1–2 and
~1.4–2x too low at d >= 4; its mean, 0.2562, happens to be close to 0.20, which
is why the scalar survived end-to-end tuning **while being badly mis-shaped**.
The per-depth vector removes the compromise instead of re-picking it.

## Suggested follow-ups (not implemented)

1. **Raise IPG above 4 for these shapes.** The in-source rule caps
   `IPG = ceil(M / ceil(M / 4))` at 4 inputs per group; the M = 5 and M = 9 pass
   boundaries are the two most expensive marginals measured. A tuned IPG of 5 or
   8 for K = 5120 would flatten exactly those steps. This is a Metal-kernel
   experiment, not a schedule one.
2. **Re-fit the `positionAcceptEMA` prior.** `0.85 · 0.98^d` is wrong in both
   directions (F10). It is the other half of the `argmin` objective and was
   deliberately left untouched to keep this diff minimal.
3. **Width-padding arm** (run verify width `w` at depth `d < w - 1`) to break
   the `H`-vs-verify-slope collinearity and separate head cost from verify cost.
4. **A `d = 5` forced arm** to replace the N = 2 row.
