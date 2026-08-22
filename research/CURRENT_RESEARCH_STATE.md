# SENPAI Research State

- 2026-08-22 06:10 UTC
- Track `qwen3.8-27b-mtp-v1`. Advisor branch `senpai/qwen38-mtp-r1` at
  `3f40d9b0`. Campaign base `origin/main` at `770a3ff2`. Organizer
  `upstream/main` at `fac135f2`. Crown `bc070b7b` at 3.35922017, unchanged.

## Most recent research direction from the human researcher team

No new direction since the standing instruction: keep every Mac productive,
compose mechanisms rather than resampling, and submit autonomously.

## 🔴🔴🔴🔴🔴 CURRENT RESEARCH FOCUS — SHIP THE FIRST MODE-PROOF CANDIDATE, AND FIRST FIND OUT WHETHER WE HAVE ONE

The target has not changed. The confidence in reaching it has.

```
crown bc070b7b                                          3.35922
our best receipt 7bef7d4c, published on a slow draw     3.29792
the same receipt corrected to a fast draw               3.34136

parity with the crown                                   +0.53 % ranked
beating the crown on either draw                        +1.86 % ranked
```

Only one live mechanism reaches +1.86 % on its own price, and that price is now
in doubt.

### 🔴 The dominant uncertainty: isolated measurements over-price by about two

Three independent observations inside two hours, all in the same direction.

| source | prediction | in-situ measurement | ratio |
|---|---|---|--:|
| E118 injection ladder against a real deletion | — | — | **1.66x** |
| E123 census time reconstruction at NA=4 | 55 to 85 % pre-registered | 117.2 % | **2.16x** |
| E121 isolated rung 2 to in-situ rung 3 | -0.888 % leg | **-0.436 % leg** | **2.04x** |

The prediction has two terms:

```
predicted leg effect = (per-cell effect measured in isolation) x (that cell's share of the leg)
```

E116 validated the share term directly at `alpha x beta` = 1.000 [0.963, 1.038].
**The entire error therefore lives in the per-cell isolated effect.** The factor
is not one number: E123 gives 2.164, 1.116 and 1.420 for the same add tree at
NA=4 depending on which half is deleted, so it has structure worth finding.

**Route B's +2.5 % ranked headline is priced entirely from an isolated
per-matvec probe.** Corrected by two it reads about +1.2 %, which clears parity
but sits below the mode-proof line. **This single factor decides whether the
campaign has a winner or a coin flip.** Thorfinn's rung 5e is the in-situ
measurement that settles it and is the most important outstanding measurement in
the campaign. Askeladd's E125 must put a corrected prediction on the record
before 5e lands.

### 🔴 E121 is merged, not submitted, and it turned out to overlap Route B

Alphonse's stop rule fired: predicted -0.888 % against a measured pooled
-0.436 % absolute candidate MTP seconds per token, sd 0.093, n=2, ranked frame
+0.415 %. Merged as `3f40d9b0` because the mechanism is exact and the
kernel-frame effect is well separated from zero; not submitted, because the
arithmetic below says a marginal win is a certain loss on a slow draw.

```
xv4 + E121, fast draw    3.3552   =  -0.12 % under the crown
xv4 + E121, slow draw    3.3116
probability of promotion              about 25 %
```

**No single mechanism in the queue reaches the mode-proof line alone.**
Composition is the only path. Which makes the next two findings the most
important thing on this page.

### 🔴🔴 FINDING F90 — E121 AND ROUTE B REMOVE THE SAME WORK

The shipped wide QMV now carries `constexpr bool SHARE_SUMS = NA <= 4;`. Each
simdgroup accumulates `sums[m]` for its own half of the m range, then the two
simdgroups exchange through `sums_xchg` with two threadgroup barriers per
k-block. **That is the same activation chunk-sum tree Route B hoists.** E121
halves it; Route B removes all of it.

Thorfinn's 7x7 grid was measured against a base that did not contain E121, so
his gross numbers price Route B against a control that no longer exists.

The overlap is not uniform. `SHARE_SUMS` keys off IPG, not M:

| M | IPG (= NA) | shares today | thorfinn ranked % |
| --: | --: | :-: | --: |
| 3 | 3 | yes | 0.01 |
| 4 | 4 | yes | 2.36 |
| **5** | **5** | **no** | **2.78** |
| 6 | 3 | yes | 1.32 |
| 7 | 4 | yes | 2.37 |
| 8 | 4 | yes | 3.28 |
| 9 | 3 | yes | 1.44 |

M=5 is the only wide width that does not share, so Route B's 6.52 % at M=5
survives whole — and M=5 sits at beagle's ranked mean width of 5.382 on the
prompt carrying weight 0.484 of the published median.

If the overlap is total at the sharing widths, gross drops by the full 1.463 %:
M=4 from 5.88 to about 4.4, ranked 2.36 to about 1.77. Round-weighted, Route B
falls from about +2.5 % to about +1.9 % before the transfer correction, and to
roughly +1.0 to +1.2 % after it.

**Alphonse measures the overlap directly in E126, PR #127**, with three isolated
arms: `share_off`, `share_on`, `sums_free`. The `sums_free vs share_off`
contrast is also a cross-instrument replication of thorfinn's 5.88 %.

### 🔴🔴 FINDING F91 — THE FIRST TRANSFER FACTOR ABOVE 1, AND IT BELONGS TO ROUTE B

From E123's entry-point census, which askeladd computed and correctly declined
to headline once his kill rule had fired:

| arm | arch | registers | spill | text | resident simdgroups |
| --- | --- | ---: | ---: | ---: | ---: |
| `a_base` | g16s | 94 | 0 | 24942 | 32 |
| `n_nosums` | g16s | 95 | 0 | 22710 | 32 |
| `a_base` | g17s | 101 | 0 | 25898 | 39 |
| `n_nosums` | g17s | 99 | 0 | 23608 | **40** |

Deleting the **whole** sums tree crosses a resident-simdgroup step on the
**ranked** architecture and crosses nothing on the local one. Deleting half
crosses nothing anywhere, consistent with E121 measuring 39 -> 38 for its gated
arm.

Every factor in F87 runs against us. This one runs for us, and Route B is the
only mechanism that deletes the whole tree. A second and probably larger version
of the same effect: the shipped entry point inlines every width and pays the F66
entry-point occupancy tax, while thorfinn's replica is specialised to one width
and does not pay it at all.

Requested from thorfinn for zero GPU time: an entry-point census of the replica
and of the consuming QMV body on both `applegpu_g16s` and `applegpu_g17s`.
Added to askeladd's correction table as a fourth index axis beside mechanism
class, roofline distance and launched volume.

### 🔴 `quantized.h` is close to exhausted for in-place gains

Every remaining group in E123's priced census is either not deletable, already a
template parameter (`DIRECT_NIBBLES`), numerics changing, or on the stop list.
The one apparently unexploited bit-exact group, activation register moves at
6.781 % weighted, is the extraction-hoist axis that `mo_hoist` closed at +2 to
+185 %. What remains is a bit-exact reduction of the E121 exchange, worth about
+0.11 to +0.17 % ranked and subsumed if Route B lands. **That is why alphonse's
slot goes to composition science and not to another kernel mechanism.**

### Route B, and why it is the largest thing we have measured

Thorfinn built a candidate-owned QMV replica dispatched from `Qwen35.swift`
through `MLXFast.metalKernel`, then added a standalone in-stream dispatch that
precomputes the activation chunk sums the wide kernel currently recomputes
4,352 times.

- **Rung 1.** Bit exact against `quantizedMM` over 1,633,536 elements, 14 cells,
  positive control fires on every cell. End to end the replica is **faster on
  average, -0.188 %**, worst width +0.401 %. Per-dispatch host cost is
  **+1.674 us**, of which the per-dispatch source-string compare I flagged as
  the main risk is **1.6 %**. Making a custom kernel's source longer is nearly
  free; issuing the dispatch at all costs 1.67 us of CPU.
- **Rung 2.** He pre-registered 25-40 us for the fill with a falsifier below
  22.7 us and **measured 4.85 us**, falsifying his own prediction by five. My
  22.7 us break-even, derived after ruling askeladd's 117-452 us fill table a
  thousand-fold overstatement, holds with **4.7x of margin**. A 3.4x larger
  table is not slower, so the fill is dispatch and barrier overhead, not
  bandwidth.

```
  width   wide-QMV round   Route B net        Route C ceiling
  M=4        76.8 ms       3177 us = 4.14 %   4434 us = 5.77 %
  M=8       149.6 ms       8111 us = 5.42 %   9412 us = 6.29 %
```

Route C's ceiling reproduces askeladd's independent E118 `x_sumshoist` figure of
+5.376 %. Thorfinn's consumer gains reproduce askeladd's on both shared shapes
with baselines agreeing to within 1 %. My own ledger-265 prediction for this
mechanism was **+2.84 % ranked** against the measured **+2.70 %**. Four
independent routes to the same number.

**Every activation tensor has exactly one wide-QMV consumer**, because q/k/v and
gate/up are fused. So Route B buys 257 matvec savings with 257 extra dispatches,
one for one, which is exactly what the 22.7 us break-even prices. Route C is now
a +1.63 pp increment that recovers 1257 us of dispatch overhead, not the prize.

### What is left to prove before Route B ships

Bit exactness on **every routed shape**, not the two tested: K=6144 is 12
k-blocks, K=17408 is 34, and `lm_head` runs grid.y = 31040, seven times anything
tested. A build-time-only routing gate. A host-side row-stride precondition,
because `ensureRowContiguous: false` makes a non-contiguous input silently
wrong. Direct measurement of the three modelled rows, which carry 815 of the
3177 us. Then 512-token exactness with post-EOS continuation and row-ledger
closure, and absolute candidate seconds per token as the headline.

### The NA=5 mystery is solved, and it was never spill

Thorfinn hit askeladd's exact failure signature and removed it. Passing `K` and
`N` as compile-time template arguments fully unrolls the ten-iteration k-loop at
K=5120, and **at NA=5 only** the output is wrong: 174,072 of 174,080 elements
differ, positive control still fires. K=512 is one k-block and is unaffected.

This explains the E118 retraction. A clean register census is not an exactness
proof **because the census reads the compiled function and the fault is in what
the compiler kept live across the unrolled copies**. Suspect compile-time
`in_vec_size` and `#pragma unroll` at NA=5 before suspecting arithmetic.

### The entry-point tax nobody had priced

`_wideN` is a `METAL_FUNC`, so the entry point allocates registers for the
**widest inlined body**. On the ranked g17s an ungated arm costs 39 to 32-33
concurrent simdgroups, about -15 %; a gated arm costs 39 to 38.

Two consequences. A register-hungry arm at one width **taxes every width**. And
**every per-cell probe in this campaign measures the body alone and misses this
tax**, so all of those numbers may be optimistic. This is an additional
unpriced argument for Route A: owning the dispatch lets us instantiate one
kernel per width.

Related: **the NA=5 spill is a g16s artifact.** On g17s neither ungated form
spills. Askeladd's -31 % at NA=5 is local only, and any ranked conclusion drawn
from a g16s spill is unsupported.

## 🔴🔴🔴🔴 THE DRAFT PATH — CORRECTED, AND STILL THE LARGEST UNTAKEN AREA

I priced this wrong in ledger 265 and the correction matters. The compact-vocab
row is **1,600 B**, not 960: 5,120 weights at 2 bits is 1,280 B, plus 80 groups
of bf16 scales and 80 of bf16 biases at 160 B each.

```
  centroid pass   12,292 rows x 1,600 B = 19.67 MB   (I wrote 11.8)
  gather pass     24,584 rows x 1,600 B = 39.33 MB   (I wrote 23.6)
  total per draft step                    59.00 MB   (I wrote 35.4)
```

The measured selection matmuls cost **264.8 us per draft step**, not the 21-41 us
I quoted, and the correct census was already in this campaign's own ledger at
lines 33446-33447 the whole time. The prize is **3.7x** the crown's probe-select
edit, not fifteen times.

### The acceptance exchange law, and the coefficient we have been using wrong

`raw_p = S_p x A_p / T_p`, so break-even is `dA/A = dT/T`: **exactly one percent
of accepted tokens per round per percent of round time.** Converting to
per-draft acceptance, the tolerable loss is `1/E` percent and the penalty of a
miss rate `m` is `100 x E x m`.

```
  prompt      d      alpha     E     1/E    break-even m   100E
  plutarch  0.154   0.182   0.035  28.88     2.9e-1         3.5
  beagle    4.382   0.934   2.031   0.492    4.9e-3       203.1
  republic  4.989   0.966   2.395   0.418    4.2e-3       239.5
  essays    5.087   0.965   2.435   0.411    4.1e-3       243.5
  medicine  5.256   0.964   2.510   0.398    4.0e-3       251.0
  botany    6.148   0.960   2.903   0.345    3.5e-3       290.3
```

This reproduces the campaign's 206.6 from first principles — it is a
**beagle-calibrated** elasticity, not an arbitrary number — and then corrects it:
the published score is a **median over prompts**, so the penalty applies per
prompt before the median. **Use 240.** Re-ranking every past `gain - 206.6 x m`
decision costs zero GPU time; all 37 cells are already computed.

Bytes saved scale with `d`; the penalty scales with `E`, which grows faster.
Break-even `m` therefore falls with depth, to **3.5e-3 on botany**. Today's
measured miss rate is **2.266e-4**. **There is roughly 15x of unspent acceptance
headroom on the draft path and nothing in the tree is spending it.**

### The queue, priced and legality-checked

**Every number in this table was repriced in ledger 268. Two of the old prices
were overstated by roughly four times because they applied a head-byte law that
is stated in the LOCAL candidate frame as though it were ranked (advisor error
87), and the acceptance coefficient was botany's 240 rather than beagle's 203
(advisor error 88).**

| id | mechanism | ranked | risk | state |
|---|---|---|---|---|
| **`noislands`** | delete the `installExactQKVRows` call at `Qwen35.swift:4342-4346`; removes about 26.2 MB **and three dispatches** per proposal step | **+0.41 to +0.45 %** | moves acceptance; kill line 0.21 pt against a 0.36 to 0.39 pt break-even | 🔴 **ASSIGNED as E124, PR #125, edward.** Island tensors verified present at the live head pin |
| **C1** | sign-sketch or low-rank first pass over gathered rows and centroids; 1,600 B row to about 130 B, removes 53.31 MB/draft | **+0.23 to +0.34 %** | changes proposals; kill rule at `Δm` 1.0e-3 vs today 2.266e-4 | designed end to end, `_advisor_scratch/c1design/REPORT.md`. Offline screen is **cheap but not free and not runnable on the advisor host** |
| ~~C2~~ | quantize the head's bf16 precision islands | ~~+0.10 to +0.14 %~~ | — | 🔴 **DO NOT ASSIGN. Dominated by `noislands` about four to one; with round-to-nearest it is numerically identical to `noislands` while still paying both dispatches.** |
| C4 | probe fraction 0.25 to 0.15 | +0.05 to +0.20 % at coefficient 203 | measured -0.3806 % round time, effect/se 15.8 | ship as a rider — **but C1 inverts its sign**, see below |
| C5 | pad centroid table 12,292 to 12,296 so `12292 % 8 == 4` stops forcing the slow `affine_qmv` | +0.03 % | **bit exact**, three lines | free rider on any draft-path PR; useless if C1b lands, because C1b replaces that pass |
| C6 | reuse the probed leaf set across drafts in one round | +0.27 to +0.31 % | **legality ambiguous — flagged, needs a second reading** | do not assign yet |

C1 is the "SlimSpec-style low-rank factorisation of the readout" this campaign
called the durable answer at ledger line 20309 and never assigned. It is now
designed: build the sketch inside `buildDerivedClusterIndex`
(`Qwen35.swift:4630-4696`), which already materialises the dequantized 98,336 by
5,120 matrix at `:4651` and discards it; hand-write the scoring kernel because
MLX rejects 1-bit quantization at `ops.cpp:4818-4823`; feed survivors into the
**existing** affine-4 rerank at `:3062-3165` rather than adding an affine-2
rescore. On paper the best estimator is **low rank k=128 int8 at 130 B/row**,
because the task is maximum inner product search and a sign sketch estimates the
cosine instead. Requantization is structurally dominated: its floor is 640 B/row
because it cuts bits per dimension and keeps D = 5,120, while sketching cuts the
dimension itself.

🔴 **C1 inverts C4.** Once centroids are sketched, raising the probe fraction
from 0.25 to 0.40 costs 0.038 pp of local round time and buys back recall. C4's
downward move is only correct while the centroid pass still streams 1,600 B rows.

Counter-evidence to respect: a static prefix trim to 49,152 rows regressed
acceptance 1.00 to 0.877. That was **row deletion**, where the miss probability
is one and unbounded. A sketch is **bit reduction**, bounded and measurable
offline. Different failure mode.

Closed and not to be re-proposed: killing the centroid argsort (+0.061 %, on the
stop list); `rowsPerLeaf` 8 to 32; coarser group size, which lost achieved
bandwidth 191.8 to 179.1 GB/s; C2 in every form.

## 🔴🔴🔴 THE MODE — SOLVED WELL ENOUGH TO STOP PAYING FOR IT

**Ledger 268 changed this from a nuisance into an instrument.** Five things are
now established, and one lever is established and forbidden.

- 🔴 **The mode is a three-slot round robin, not a coin flip.** 564 absolute
  labels in completion order give lag-1 **-0.204**, lag-3 **+0.191**, lag-6
  +0.127, lag-9 +0.127, and a runs test at **z = +4.83**. P(fast) = **2/3**, not
  1/2. Three artefact controls pass, the strongest being that the **candidate
  free serial leg over all 770 rows shows the same signature more sharply**
  (lag-1 -0.236, lag-3 +0.327) — no candidate can cause that. This is runner
  infrastructure, and a submitter cannot see or choose its completion
  neighbours: the queue runs 26 to 224 minutes deep with up to 20 rows open.
- 🔴 **There is an absolute per-row mode classifier, 96.0 % accurate.** A fixed
  weight vector over the eight per-prompt log candidate times, with weights
  summing to zero, separates the two modes by **3.39 sd** and agrees with 72 of
  75 pairwise ground-truth verdicts. Per-run noise is 0.0817 mode units against a
  step of 1.000. Valid only with head provenance held fixed.
- 🔴 **This does not contradict the "no linear contrast" result below.** That
  result says no fixed contrast can *cancel* the mode, because its magnitude
  varies run to run. Classification only has to *detect* a binary step, then add
  the mean back. Detection and cancellation are different problems.
- 🔴 **Nothing at submission time predicts the mode.** 54 tests, permutation p,
  Benjamini-Hochberg at 5 %: solver, files changed, lines added, surface bytes,
  schedule stratum, prefill, hour and day are all null. The two survivors are
  classifier aliasing by one proposal head, not real predictors.
- 🔴 **The mode costs about 0.82 ms per drafting round and prefill is untouched
  at 90 sigma.** R-squared through the origin is 0.856 on drafting rounds and
  **0.000 on non-drafting rounds**. This independently reproduces the 0.803 +/-
  0.025 ms per drafting round from the pair fit, by a different route.
- 🔴 **Post-hoc correction is free and worth 5.8x.** A slow run publishes
  **1.3170 +/- 0.0589 %** below the same candidate run fast. Correction cuts the
  residual to 0.108 %, a 99.0 % variance reduction. **Operational rule: never
  conclude a candidate regressed from a drop below 1.4 % in one uncorrected
  ranked run.**
- 🔴 **Applied to our own six receipts, `7bef7d4c` is our only slow draw.**
  Corrected it reads **3.34136**, which beats `b8b8b860` by +0.217 % — and the
  independent two-parameter pairwise decomposition gave c = +0.217 as well. Two
  methods with no shared machinery agree to three decimals. **`xv4` was never a
  regression.**
- 🔴 **The re-roll lever is real and forbidden.** P(at least one fast in a
  consecutive triple) is 0.994 against 0.971 under independence. Re-rolling is a
  duplicate submission and slot timing is measurement-structure exploitation.
  **The mode is to be measured and corrected for, never steered.** Recorded here
  so nobody re-derives it and thinks it is an opportunity.

- **It is not a wide-dispatch cost.** `g2_rounds` has r-squared **0.0005**. The
  best regressor is gated drafting duty, r-squared 0.9364, fitted at 0.803 +/-
  0.025 ms per drafting round. That rules out every explanation making the mode
  a property of the kernel we keep optimising.
- **No fixed linear contrast can remove it.** The min-variance mode-orthogonal
  contrast is 88-91 % plutarch and does not beat plutarch alone. The reason is
  structural: the mode's **magnitude** varies run to run (`k` sd 0.166, range
  0.684 to 1.427), a multiplicative nuisance a linear contrast cannot cancel.
  **Stop searching for a better mode-orthogonal estimator from board data.**
- 🔴 **Plutarch is the WORST instrument for a wide-QMV mechanism.** Finding 56
  already said plutarch reaches the wide QMV on 9.6 % of its leg; I failed to
  multiply that by the MDE. Dividing the leg-frame MDE by the reachable fraction
  `phi` (0.0564 plutarch, 0.6068 the rest):

```
  scenario       estimator                     MDE on a wide-QMV mechanism
  mode shared    wide-optimal, mode matched          0.30 %
  mode shared    candidate mean of 8                 0.35 %
  mode flipped   wide-optimal, mode orthogonal       1.95 %
  mode flipped   gated 2-parameter decomposition     2.42 %
  mode flipped   plutarch alone                      5.82 %   <- worst
```

  **Mode robustness and mechanism sensitivity are anti-correlated.** A
  mode-matched reference is worth about 6x in resolving power, which makes
  finding one more valuable than improving the estimator.
- The plutarch anomaly survives every ungated model at 3.7 to 5.5 sigma. The
  best zero-free-parameter account is that the cost is charged only when the
  **previous** round also drafted — lost pipeline or state-reuse overlap. A
  DVFS or residency hysteresis story fits equally well and cannot be separated,
  because no prompt sits at intermediate drafting duty. Cheapest resolver: a
  synthetic schedule at duty near 0.5.

## 🔴🔴 CROSS-CUTTING RULES ADDED THIS ROUND

- **Rule 65.** Re-derive a mechanism on the current base. Never port an arm's
  diff. Alphonse measured the penalty at **2.2x**: askeladd's arm ported
  unchanged reads +0.568 % where his re-derived form reads +1.506 %.
- **Rule 66.** An MDE belongs to the estimator *and* the mechanism. Divide by
  the mechanism's reachable leg fraction `phi` before comparing with a
  prediction.
- **Rule 67, amended twice.** Apply the acceptance penalty per prompt, then
  recompute the median **exactly**. If a single coefficient is unavoidable use
  **beagle's 203**, not botany's 240 and not 206.6. Botany is not median
  eligible; beagle is the fourth order statistic in 326 of 326 frontier rows.
- **Rule 68.** Before pricing from a byte count, `grep` the ledger for an
  existing measured census of the same cell.
- **Rule 69.** Never price a byte, dispatch or acceptance mechanism without
  naming its transfer **class**. The classes are: draft-path head **bytes** at
  0.24 (band 0.237 to 0.33); draft-path **dispatch** deletions at 0.95 in per
  cent and 0.52 in absolute microseconds; general per-round **DRAM streaming** at
  1.0 in per cent; **acceptance** loss at 1.0 by identity. 🔴 **The coefficient
  0.87 is void** — it was never measured, and it rests on a bandwidth assumption
  that is about two times wrong. Never mix classes in one price. State whether a
  local gain is at local or ranked draft depth before choosing 0.237 (depth
  corrected) or 0.327 (raw local depth).
- **Rule 70.** **Plutarch, drama and travel carry exactly zero weight on the
  published median.** Beagle carries **0.484**, hard and stable. Any single fast
  prompt is worth about 0.526 but saturates at the fifth-to-sixth gap, typically
  1.15 %. Never count a gain on a zero-weight prompt toward a ranked prediction.
  They remain legitimate *instruments*.
- **Rule 71.** Before calling any single ranked run a regression, classify its
  mode and correct it. Never conclude a regression from a drop below 1.4 % in one
  uncorrected ranked run.
- **Rule 72.** Timing a submission to land on a fast slot, and re-rolling to
  catch one, are **both forbidden**. Measure the mode; never steer it.
- **Rule 73.** A register or spill census is a **cost** observation, never
  correctness evidence. It reads whole-function summaries and cannot see what the
  compiler keeps live across the unrolled copies of a loop. Correctness comes
  only from an exactness sweep with positive controls. Thorfinn wrote this rule
  against his own passing result.
- **Rule 74.** Before describing an offline screen as free, verify on a **named
  reachable host** that the interpreter, the corpus and every auxiliary table
  exist. A screen needing a 370 MB corpus copy, or a GPU re-capture, is not free.
- **Rule 75.** Before adding two mechanisms in a plan, name the instructions or
  bytes each one removes and show the two sets are disjoint. A composition sum
  is a claim about independence. If the sets intersect, carry the intersection
  as a subtraction and schedule the experiment that measures it **before**
  either mechanism is promoted. Merging one of an overlapping pair also moves
  the control the other one measures against, so the in-flight experiment must
  re-derive on the new base (Rule 65). Earned by Advisor Error 91.
- **Rule 56 extended.** Census the **entry point** on g17s, not only the body,
  and apply the build threshold to the arm including its entry-point cost.

### 🔴 The exact marginal value of each prompt, and why draft work is safe

A one per cent candidate speedup on one prompt, with the median recomputed
exactly over 326 frontier rows:

```
  prompt     mean gain   P(gain > 0)
  beagle      +0.4862 %     1.000     <- the only reliable single lever
  medicine    +0.2508 %     0.512
  essays      +0.1598 %     0.411
  botany      +0.0124 %     0.043
  republic    +0.0100 %     0.034
  plutarch     0.0000 %     0.000
  drama        0.0000 %     0.000
  travel       0.0000 %     0.000
  all eight   +1.0101 %     1.000
```

The published score is exactly `0.5 x raw_beagle + 0.5 x min(essays, medicine,
republic, botany)` in 326 of 326 frontier rows above 3.2. Single-prompt weights
are sub-additive: they sum to 0.919 against 1.010 for a uniform change.

🔴 **This does not penalise draft-path work.** The three prompts a draft
mechanism cannot help are exactly the three worth zero. A draft-scaled one per
cent saving converts at **0.92 to 0.94** of a uniform one per cent.

🔴 **Beagle sits at ranked mean width M = 5.382 and carries weight 0.484.** Any
mechanism whose gain varies with width must be priced at M = 5 and M = 6, not at
the local mean width.

## 🔴🔴 HARNESS DEFECTS 22 TO 25

**25.** 🔴 **The ABBA rebuild gap is an arm-attached thermal confound**, found by
alphonse. In an `A B B A` schedule where each arm change triggers a rebuild, the
position with no rebuild in front of it gets no cooling gap. Measured entry
temperatures across his eight legs were 40.18, 44.87, 44.78 and 44.76 C on the
base arm, mean 43.65, against 44.81, 53.00, 45.09 and 52.95 C on the candidate
arm, mean 48.96. **The candidate arm ran 5.31 C hotter on every replicate, by
construction.** The confound attaches to the arm, not to the position, so
position balancing cannot remove it. Here it ran against the candidate by about
0.05 pp. Fix: **insert a matched idle gap wherever a rebuild does not occur.** An
arm pair selected by an environment variable needs no rebuild at all and removes
the confound by construction; prefer that design where the mechanism allows it.

**24.** 🔴 **A general-purpose subagent that spawns helpers fails on exit** with
`child agent exited with uncollected descendants`. Two of three general-purpose
children failed this way in one round. **Both had already written a complete
report to scratch, so nothing was lost.** Mitigations: prefer leaf agents where
helpers are unnecessary; instruct general-purpose children explicitly to await or
cancel every spawned task before returning; require the child to write its report
to scratch as it goes. 🔴 **Always list a failed agent's scratch directory before
treating its work as lost.**

**22.** An out-of-bounds device write in an isolated probe kernel **faults the
command buffer while the harness still exits 0**. Every later dispatch retires
in 1-3 us and the table reads as an enormous speedup. Alphonse caught it from an
implied read rate of 17-32 TB/s against a 273 GB/s peak. `if (first_m >= NA)
{ return; }` is mandatory with `research/e104_rate_probe.m`, and three validity
gates now void a session before any speedup prints: implied bandwidth above
`1.2 x 273` GB/s, the null scaffold moving more than 0.50 %, or any positive
control reporting `detected=False`.

**23.** 🔴 **`rg` is not installed on the advisor host**, yet the harness
documentation instructs agents to search with it. An `rg` call whose stderr is
discarded looks exactly like "no matches". **Every negative conclusion resting on
an `rg` search must be re-run with `grep`.** This is one plausible cause of the
draft-path byte error above: the correct census was in the ledger all along.

## 🔴🔴🔴🔴🔴 THE `7bef7d4c` RECEIPT — REJECTED, AND IT REFUTED NOTHING

`xv4` scored 3.29792432850592 and was rejected for score, not for any gate.
`parity_all_ok` is true on all eight prompts, so **`xv4` is exactly correct on
the hidden set**.

The apparent local-to-ranked sign flip is not a mechanism signal. The
two-parameter decomposition, `slower_p = 100*m*rounds_p/leg_p + c`:

```
pair                                  m (ms/round)        c (%)         R2
xv4 vs b8b8b860                    +0.6841 +/- 0.1949  +0.2154 +/- 0.3504  0.673
xv4 vs 44559d02                    +0.8221 +/- 0.2223  -0.0546 +/- 0.3998  0.695
CALIBRATION 51b9bf85 vs 097991a0   -0.8086 +/- 0.1227  +0.2255 +/- 0.2171  0.879
```

The calibration pair is **byte-identical code**, so its true `c` is exactly
zero, and the fit still returns +0.2255. `xv4`'s `c` is inside that offset on
both references, its `m` matches the calibration magnitude, and its TARGET
effect has **opposite signs** against the two references, +0.1605 and -0.1716.
A mechanism does not change sign when you change which run of the same code you
compare against.

**Ruling: `xv4` is ranked-unresolved, not refuted. It stays on the base. It will
not be resubmitted alone.**

## 🔴🔴🔴🔴 THE FOUR EXPERIMENTS IN FLIGHT

All four Macs are busy. Three experiments closed this cycle and three opened.

| PR | student | experiment | state |
|---|---|---|---|
| #121 | thorfinn | E120 own the QMV dispatch, hoisted activation sums | **rungs 1, 2 and 5d passed. Gate settled. Rung 5a exactness in flight, then 5e, the decisive in-situ measurement. Must merge `3f40d9b0` and re-derive: E121 moved his control** |
| #125 | edward | E124 delete the MTP head precision islands and price the acceptance exchange | Stage 0 zero GPU |
| #126 | askeladd | E125 explain and correct the isolated-to-in-situ over-prediction | Stage 0 pre-registered prediction due before 5e |
| #127 | alphonse | E126 price Route B on the shipped base, settle the in-situ transfer | **new this cycle**, rung 0 zero GPU, rung 1 is what thorfinn needs to read 5e |

Closed this cycle: **#122 E121 merged**, exact, +0.415 % ranked, not submitted;
**#123 E122 merged as a decisive null**, pooled stratified AUC 0.5109 [0.4850,
0.5364]; **#124 E123 merged despite its own kill rule firing**, primary metric
0.66 to 0.2908 pp with five practice-changing findings.

### E120 — the ship path, and the measurement that decides it

Rung 5d produced the most valuable table of the round: a measured 7x7 grid of net
microseconds saved per matvec across seven shapes and seven widths, with no
modelled rows. Two of my claims died in the same comment.

- **Advisor error 90.** I priced a gate that never existed. The shipped
  predicate tests the **total row count**, not the accumulator group width, so it
  takes the table at M=6 and M=9 and never discarded the microseconds I charged
  to it.
- **F74 corrected.** Absolute per-group additivity is falsified across 28 cells,
  mean 1.510, range 0.509 to 4.270, failing by shape not by noise. **Fractional
  additivity holds**: the percentage gain is flat across all seven shapes at
  every width from 4 to 9, with a standard deviation of 0.10 to 0.77 pp over a
  bandwidth span of 74 to 246 GB/s.

**My NA=3 bandwidth regression is refuted and that is good news.** It fitted the
M=3 row only; M=6 and M=9 are also pure NA=3 groups at lower bandwidth and do not
follow the line. M=3 is special, NA=3 is not bandwidth-sensitive. A mechanism
that is flat from 74 to 246 GB/s transfers far more safely to the ranked M5 host,
which streams at 542.8 GB/s.

**Gate settled and endorsed.** Over all 257 calls every gate form is identical at
M >= 4 and all 42 cells pay. The whole gate question lives at M=3 and is worth
+62 us of 68,410, or 0.09 %. He shipped `tablePays(m: Int) -> Bool { m >= 4 }`,
dropped the volume term as unsupported dead complexity, and recommends against a
three-entry M=3 exception. Endorsed: 0.09 % is below every measurement floor we
own.

**Ranked conversion** `ranked % = wide-QMV round % x 0.577` gives M=3 0.01,
M=4 2.36, M=5 2.78, M=6 1.32, M=7 2.37, M=8 3.28, M=9 1.44. **Route B is
mode-proof at M=4, 5, 7 and 8 but not at M=6 or M=9.** Beagle's ranked mean width
is 5.382 and it carries median weight 0.484, so the realised width histogram is a
first-order uncertainty.

Rung 5e must deliver four things: the headline in **absolute candidate seconds
per token against a fresh unchanged base**, not the local ratio; the
isolated-to-in-situ transfer ratio as its own named number; the realised width
histogram from the timed legs themselves under harness defect 20; and a worker
assertion before and after each timed leg.

### E121 — measured, halted, not submitted

Eight legs, two ABBA quads, 512 tokens, one ungated session. Pooled -0.436 %,
sd 0.093, ranked -0.415 %. Schedule invariant to six decimal places, exactness
4/4, all eight legs matched. The wide interval is a `df=1` artefact, not noise.

He caught his own optional-stopping error unprompted and withdrew his own
preferred option. His occupancy-transfer explanation is probably wrong because
F66 makes occupancy this arm's penalty rather than its gain; two better
hypotheses were issued.

**Harness defect 25, his finding, now a campaign rule of practice.** In an ABBA
schedule where each arm change triggers a rebuild, the position with no rebuild
in front of it gets no cooling gap, and the confound attaches to the **arm**, not
the position, so counterbalancing cannot remove it. His candidate arm ran 5.31 C
hotter on every replicate by construction. Fix: insert a matched idle gap
wherever a rebuild does not occur. An env-selected arm pair needs no rebuild at
all and removes the confound by construction.

### E124 — the top unassigned draft-path mechanism, now de-risked

I read the declared head artifact the harness actually loads. The six
`precision_islands` tensors are present at the live pin, 31,469,568 bytes in
total. E82 measured them on a different base against a different head and rule 65
forbids porting an old diff; the artifact is present now, so E124 re-derives
rather than ports.

The byte model changes as a result. `kIndices` and `vIndices` are complete
permutations, so `installExactQKVRows` takes its fast branch and **the head's K
and V projections are computed entirely in bf16 today**. Removing the islands
forces the affine-4 K and V rows to be read, so the net saving is about **26.2 MB
per proposal step, not 31.47 MB**. Repriced under rule 69 the mechanism is
**+0.41 to +0.45 % ranked**.

Exactness cannot break, because the target verifies every emitted token and the
target is unchanged. Only acceptance and time move, which makes this a clean
one-to-one exchange under F69. Break-even acceptance loss is 0.36 to 0.39 pt
absolute; the kill line is set at **0.21 pt**, about half of break-even, because
the time gain is a prediction and the acceptance loss will be a measurement.

Four env-selected arms on one binary: `all`, `none`, `q`, `kv`. The ladder
matters because K and V cost 20.97 MB while Q is 1,024 corrected rows of 12,288
at 10.49 MB plus a scatter, so the acceptance cost and the time saving may not sit
in the same place.

### E125 — turn the over-prediction warning into a number

Askeladd owns F87. He produced two of its three observations and wrote the
warning that rung-1 rankings are ceilings. `research/` only: `quantized.h`
belongs to alphonse and `Qwen35.swift` belongs to thorfinn and edward, and his
instruments are standalone `research/*.m` programs that never touch shipped
source.

He cannot run the in-situ frame and does not need to. E121, E118 and E123 supply
three held-out in-situ points, and E116 supplies a required null control: any
correction that moves `alpha x beta` outside [0.963, 1.038] is wrong, because the
share term is not the problem.

Stage 0 is mandatory and due within his first hour: a pre-registered prediction
of thorfinn's rung 5e, on the record before 5e lands. Stage 1 runs the
three-frame discriminator — standard isolated, isolated with the weight buffer
cycled to force DRAM streaming, and isolated with a co-scheduled bandwidth
consumer. If either modified frame reproduces the in-situ effect we gain both the
explanation and **a corrected screening instrument that costs minutes instead of
hours**, which is worth more than the explanation.

## 🔴🔴🔴 POTENTIAL NEXT RESEARCH DIRECTIONS

Ordered by expected value, to be assigned as students free.

1. **Compose a submission.** Campaign Rule 59 forbids another sub-0.5-percent
   single-mechanism submission. The next slot carries `xv4` plus whatever E120
   and E121 return, measured independently and then measured together. If all
   four converging mechanisms land, the composed candidate is about +3.95 %
   ranked over the current base.
2. **The draft-path gather, which is fifteen times the crown's prize.** The
   `gatherQuantizedMM` over 24,584 rows plus the centroid QMM read about 35.4 MB
   per draft step, which is 1.10 % to 1.29 % of the ranked round. Nobody on the
   board has touched it. It is a different class of experiment because changing
   what is gathered changes proposals, so acceptance rate becomes an outcome to
   measure rather than an invariant to preserve. E122's margin fit and E99's
   2.88 to 3.95 % acceptance headroom are the instruments for pricing that trade.
3. **Adopt the crown's radix probe-select.** Read at source and fully specified:
   three hunks in `Qwen35.swift`, minus eight dispatches per draft step, exact by
   construction, proposal-side only. Worth +0.074 % ranked central. Queued behind
   E120 because thorfinn owns the file. Must be screened with an isolated
   micro-benchmark; the local effect of 0.03 to 0.06 % is below our end-to-end
   noise floor. Optionally also collapse our own `Qwen35RowTop32` from two
   dispatches to one using their 16-bit key packing, taking it to +0.083 %.
4. **Locate the launched-grid-volume trough on M5.** E117 found that the IPG=4
   rate curve collapses onto one function of `V = M * grid.y`, with a trough
   near -11 percent over V in 16384 to 18432, and that
   `1 - rate(V)/rate(V/2)` predicts the split sign in all seven testable N with
   no free parameters. We have no M5 access. This needs a transfer-probe design
   that can be run inside a submission.
5. **Route C, the free sums producer.** Emit the sums table as a second output
   of our own editable `qwen35_dual_rms_norm_concat_bf16_v1`. Near-free, and it
   removes the precompute dispatch for 59.98 percent of the round. Now folded
   into E120 as rung 3.
6. **Settle the 27-dispatch proposal-side census.** `campaign-ledger.md:27133`
   records 27 proposal-side dispatches per draft step, which cannot be reconciled
   with the incumbent selection chain alone needing 24 to 27. Either that census
   predates the E87 cluster index, or it counts command-buffer flushes rather
   than kernel enqueues. Settle it before that constant prices anything again.
7. Also live, unassigned: N-selective stream collapse; a width-aware Q-row
   narrowed pack; one traced per-round verify-width sequence from a
   ranked-representative prompt; and resolving harness defect 20 with a matched
   dual-harness census.

**Closed this cycle.** The AIR-instruction cost model is delivered as Finding 58
and is now an injection ladder, not a regression. The crown mechanism is read.
The byte-identical-resample resolution study is in flight as a subagent.

## Measurement discipline that now governs every assignment

- **Finding 56. A probe's resolution is not its sensitivity.** Plutarch has 449
  non-drafting rounds of 487 and a mean draft length of 0.154, so about 92
  percent of its rounds dispatch at M=1 and never enter the wide cross-row QMV.
  The sharpest instrument on the board is structurally blind to the hottest
  kernel we work on. Always check that a probe's traffic reaches the changed
  code path before naming it a discriminator.
- **Finding 57 and Campaign Rule 59.** The only probe that sees the wide QMV is
  the DRAFT probe, which is exactly where the FACT-2 mode lands at 1.0 to 1.5
  percent. One receipt cannot separate a half-percent wide-QMV mechanism from
  the mode. Submit only candidates with a predicted ranked effect above about
  +0.5 percent, or compose.
- **Campaign Rule 60.** Before attributing any per-prompt move on a receipt to a
  mechanism, run the two-parameter mode and mechanism decomposition and read `c`
  against the byte-identical calibration pair, never against zero. Mechanised in
  `research/board_prompt_instrument.py --read`.
- **Campaign Rule 58, amended.** Every `A` carries a frame label and BOTH
  endpoint launched grid volumes. It does not carry an IPG label.
  `A_tensor(IPG=4)` ranges 1.7172 to 2.0407 over ten clean N. Live pricing keeps
  `A_ranked` at about 2.0.
- **Campaign Rule 55.** Before any official submission, diff the submitted
  surface against the exact measured tree over `editablePaths` only, from the
  submitting branch's own merge base. Merge for ancestry only after the receipt
  lands.
- **Campaign Rule 56.** A register-channel arm must be censused on
  `applegpu_g17s` before any local timing is booked.
- **Campaign Rule 57.** An isolated per-group cell may not be weighted by a
  realised-width histogram unless the dispatch grouping is the same in both
  frames.
- **Finding 28, the stale-build defect.** For the `quantized` family the
  metallib is dead; only the worker binary carries the arm. No timed kernel leg
  may be reported unless `senpai/rebuild-and-assert-worker.sh` asserted the arm
  string inside the binary immediately before the leg. That script now uses
  `grep -F` by default, with `--regex` to opt out and `--self-test` to prove a
  bracketed needle can fire.

## 🔴🔴 THE MEASUREMENT RESOLUTIONS, RE-VERIFIED

Re-measured on the 764-row board: 711 groups, 36 replicated, 79 code-identical
pairs. All four constants reproduce within 5 percent.

```
constant             in file   measured
target_same_mode      0.0945     0.0916
draft_same_mode       0.0952     0.0931
target_all            0.1100     0.1047
draft_all             0.6687     0.6393
```

`D all` is about 0.64 percent because it **contains the mode**. It is not a
resolution. Read the `same` columns, and do not transpose the two probes.
Harness defect 21, which claimed these constants were six times optimistic, is
**withdrawn**.

The real defect was next door. The mode classifier requires DRAFT above
`MODE_DRAFT_SHIFT` and TARGET below `MODE_TARGET_SHIFT`. The same-mode TARGET
pair floor is 0.0945 * sqrt(2) = 0.1336 percent, so the old 0.15 cut sat at 1.12
pair sigma and misclassified about one mode flip in four. It is now 0.30, which
is 2.25 pair sigma, with `MODE_TARGET_AMBIGUOUS = 0.15` marking the band
between.

## 🔴🔴🔴🔴🔴 WE HOLD THE BEST CANDIDATE ON THE BOARD. WE LOST ONLY THE DRAW.

```
TOP PROMOTED
  bc070b7b francip      3.35922017  src=fac135f2   <- crown
  7358c89f newjordan    3.35206897  src=1ec3625b
  51b9bf85 vibecodooor  3.35025879  src=41bad1c6
  276aa2c2 hadakang     3.33849825  src=ca061247
  f04b102e morganmcg1   3.32824629  src=23ef7556   <- ours
  8819b108 audreyt      3.32794961  src=b40c28e9

SERIAL-FREE, 764 scored rows, median-of-8 reproducing every published
score to 3.98e-11
  3.34789703  pub 3.33412148  b8b8b860  rejected   <- OURS, RANK 1 of 764
  3.34767209  pub 3.35922017  bc070b7b  promoted   <- the crown, rank 2
  3.34723355  pub 3.35206897  7358c89f  promoted   rank 3
  3.34722609  pub 3.34351272  44559d02  rejected   <- OURS, rank 4
  3.34573143  pub 3.34792207  1422606f  rejected   rank 5
  3.34549718  pub 3.33872765  9612d3ba  rejected   rank 6
  3.29427211  pub 3.29792433  7bef7d4c  rejected   <- xv4, rank 116
```

The crown's engineering is 0.007 percent behind ours. Draw ratios, published
over serial-free: ours 0.99588 and 0.99889, the crown 1.00345. Spread about 0.85
percent.

**At serial-free 3.3479 even the best plausible draw yields about 3.3595, level
with the crown and not past it. We need 0.3 to 0.5 percent of real serial-free
gain.** `x_sumshoist` at a predicted +4.11 percent ranked is an order of
magnitude more than that. That is the whole reason this cycle looks the way it
does.

Our five used slots: `f04b102e` accepted 3.32824629; `87b654b2` rejected
3.12600524, a real mechanism loss; `b8b8b860` rejected 3.33412148, mode A;
`44559d02` rejected 3.34351272, mode A; `7bef7d4c` rejected 3.29792433, mode B.
Two of five were pure resamples. Rule 59 exists to stop that repeating.

---
## 🔴 FINDING 22. THE TRANSFER LAW HAS TWO CLASSES. PRICE EVERY MECHANISM WITH THE RIGHT ONE

Source: thorfinn's E87 terminal result on PR #89, self-corrected against his own
receipt, plus my reprice in `research/finding22_reprice.py`. Ledger 248.

**The law:**

```
ranked delta_us / local delta_us  =  (local achieved rate) / (ranked achieved rate)
```

For DRAM-bound work both rates are the machine's streaming bandwidth, so the
ratio is 249.55 / 542.8 = 0.460 and the PERCENTAGE is preserved. For
latency-bound work neither rate scales with DRAM bandwidth, the ratio is about
1.0 (measured 0.98), and the PERCENTAGE is amplified by
`local_round / ranked_round` = 2.401 at M=5.

```
STREAM  work            ranked % = local % x 1.0     (0.460 x 2.401 = 1.104)
LATENCY work            ranked % = local % x 2.40    (0.980 x 2.401 = 2.353)
HEAD BYTE removal       x 0.236                       MEASURED, E87 arm C
ACCEPTANCE loss         x 1.0                         accounting identity
```

The sanity check the law must pass is that a DRAM-bound saving keeps its
percentage, because the item and the round it divides into scale together. It
does, at 1.104. That is why the latency branch is credible.

**Evidence.** Section 8 removes fixed dispatch latency, not bytes. Priced with
the 0.236 byte factor it was +0.0095 %; measured in the serial-free frame it was
**+0.1117 %**, an understatement of about **12x**. Thorfinn's forward prediction
with no fitted parameter, from the isolated census rate 12.84 us/draft and
public ranked round times, gives +0.1036 % on the median pair, 93 % agreement.
A board regression concurs at 12.53 us/draft (se 5.73, t 2.19).

**RETIRED: Finding 13's derived transfer factors.** The "fixed / launch"
transfer of 0.670 is wrong: a fixed-class local cost of 65,674 us transferring
at 0.98 would need 64,361 us of a 55,870 us ranked round. Finding 13's "fixed"
bucket is streaming work that the marginal-per-row model failed to attribute,
because that model counted only marginal per-row cost and never the G=2 base
streams. Finding 21's direct census supersedes the split. **Keep only the
measured head factor 0.236 and the acceptance factor 1.0. Delete the derived
verify factor 1.532 and the derived fixed factor 0.670 from all pricing.**

**The corrected closure threshold.** Compare a LOCAL cost against:

```
STREAM-class item is dead below    0.160 % local
LATENCY-class item is dead below   0.067 % local   (0.115 % on the published floor)
```

Every item closed between those two bounds was closed on the wrong test.

**The reprice of the E96 census** (local M=5 round 127,533 us, ranked M=5 round
53,108 us, DRAM peak 273 GB/s, DRAM-bound cut at 60 % of peak):

| family | us/rnd | GB/s | %peak | class | local % | ranked % |
|---|---:|---:|---:|---|---:|---:|
| MLP gate_up | 48381.86 | 265.8 | 97.4 | stream | 37.937 | 41.883 |
| out_proj + down_proj | 36559.21 | 238.1 | 87.2 | stream | 28.666 | 31.649 |
| GDN in_proj | 17675.04 | 258.4 | 94.7 | stream | 13.859 | 15.301 |
| lm_head | 5269.31 | 271.9 | 99.6 | stream | 4.132 | 4.562 |
| attn fused QKV + gate | 5163.37 | 256.5 | 94.0 | stream | 4.049 | 4.470 |
| GDN recurrent step | 1421.13 | 212.5 | 77.8 | stream | 1.114 | 1.230 |
| **SDPA over FA history** | 1267.00 | ~53 | ~19 | **latency** | 0.993 | **2.386** |
| **fused residual + RMSNorm** | 771.54 | 27.0 | 9.9 | **latency** | 0.605 | **1.453** |
| **GDN prework** | 543.39 | 32.6 | 11.9 | **latency** | 0.426 | **1.023** |
| q/k norm + RoPE | 149.85 | - | - | latency | 0.117 | 0.282 |
| KV cache write | 89.10 | - | - | latency | 0.070 | 0.168 |
| MTP top-2 | 56.13 | - | - | latency | 0.044 | 0.106 |
| STREAM subtotal | 114469.92 | | | | 89.757 | 99.094 |
| **LATENCY subtotal** | **2877.01** | | | | **2.256** | **5.417** |

After the measured isolation discount (calibrated by the two dose ladders: GDN
step 1421.13 isolated against 861.0 dose = 1.65x; fused norm 771.54 against
298.0 = 2.59x), the latency pool is **2.09 % to 3.28 % of the ranked round**,
not the 0.87 % to 1.37 % I had been pricing.

**REVIVED by the reprice:**

- **fused residual + RMSNorm.** Dose 298.01 us/pass/round, R2 0.9506. Local
  0.234 % is below the 0.277 % published floor, which is why E96 rung 3a closed
  it. Ranked **0.561 %**, which is 2.0x the published floor.
- **SDPA over the full-attention history.** Carried at "0.4 % to 0.6 %
  corrected"; discounted ranked **0.92 % to 1.45 %**. Largest single latency
  item. 79.19 us per dispatch is far above launch overhead, so this is
  inefficiency, not launch cost: about 4.2 MB per layer per round at 79.19 us
  implies 53 GB/s, 19 % of peak. Its true factor sits between 0.46 and 0.98; at
  a conservative 0.7 it is still 0.64 % to 1.01 % ranked.
- **GDN prework.** Ranked 1.023 % isolated, **0.40 % to 0.62 %** discounted.

**STAYS CLOSED: the GDN recurrent step.** Stream-class at 212.5 GB/s, 77.8 % of
peak, so its percentage does not amplify: 0.675 % local, 0.745 % ranked, with
little headroom, and the scored path reaches the non-editable `GatedDelta.swift`.

**No live assignment's stop rule moves.** E98, E99 and E100 are stream-class or
schedule-class, and E99 is already priced on the ranked curve. The head-side
affine-2 metadata idea is a genuine byte change, so 0.236 stays correct there
and its 0.17 % shelving stands.

**ADVISOR ERROR 52.** I accepted a byte factor for a latency mechanism and then
retired the E87 selection chain on it. Repriced, that chain is +0.918 % on the
median pair as an f16 bound, and the realizable part is +0.32 % to +0.72 %.
E101 (thorfinn, PR #103) reopens it.

## 🔴 FINDING 21. THE ROUND IS AT LEAST 82 % DRAM WEIGHT STREAMING, AND THE TRANSFORM THAT WRITES THOSE WEIGHTS IS OURS

This finding reorganises the whole campaign. Read it before pricing anything.

**21a. The floor.** The transformed target weights total **14.4123 GB**. The
student M4 Pro has a DRAM peak of about **273 GB/s**, so one full weight stream
cannot take less than **52,792 us**.

| M | G | measured round busy | minimum streaming time | streaming share |
|--:|--:|---:|---:|---:|
| 1 | 1 | 64,445 us | 52,792 us | **>= 81.9 %** |
| 5 | 2 | 126,103 us | 105,584 us | **>= 83.7 %** |
| 9 | 3 | 204,029 us | 158,376 us | **>= 77.6 %** |

The achieved rate implied by `G * 14.4123 GB / round` is 223.6, 232.2 and
219.7 GB/s, which is **82 to 85 % of the DRAM ceiling**. The ranked M5 at M = 5
implies about **542 GB/s** on the same accounting.

🔴 **Only two quantities can move the score by a large amount: the number of
bytes per weight, and the number of full weight streams per round, `G`.** One
extra stream costs about 52,800 us, which is **42 % of the M = 5 round**.
Everything inside the width-independent term `a` = 10,919.5 us lives inside
**8.7 %** of the round, and no single item in it clears the detection floor.

Caveat on method: a single-`S` fitted model `cost = G*S + k*M` does not fit both
dispatch bands, because `G` varies per tensor. **Use the floor argument — total
weight bytes divided by DRAM peak, against the measured round — and never a
fitted `S`.**

**21b. The transform is candidate-owned and the whole field has left it
untouched.** The ranked workflow step `Transform Qwen target in bench sandbox`
at `.github/workflows/qwen-mtp-ranked-benchmark.yml:1669-1700` runs, inside the
submission sandbox, with the log line `running submitted transform`:

```
.build/release/mlxfast-swift transform --reference "${MLXFAST_QWEN_MTP_TARGET_DIR}" --output weights
```

The pinned artifact is the **source** checkpoint. The `weights/` directory the
ranked target loads is produced by **our** code, and
`Sources/MLXFastTransform/` is editable. `qwen_mtp_weights_hash` is a **TOCTOU
guard, not a pin**: the workflow hashes the transform output at `:1703` and
re-checks it at `:2791-2821` to detect a change **during** the run. It is never
compared with a repository constant.

🔴 **690 of 690 scored board runs report the identical hash
`b53e4991737cdf50827e518e7559628874d3ff6d5f63bebc057ddbb16a89e2cd`.** No
submission from any solver has ever changed a byte of the transformed weight
representation.

**21c. The mechanism is already written and has no reader.**
`Sources/MLXFastTransform/AffineMetadataCoding.swift`, 438 lines, already builds
the uint16 (scale, bias) index: `pairToIndex: [UInt32: UInt16]`, a 65,536-entry
lookup table, the pair packed as `UInt32(scale) | (UInt32(bias) << 16)` over two
bf16 halves, emitting `<stem>.metadata_indices` and `<stem>.metadata_lut` into
the shard `mlxfast-projection-metadata.safetensors`. It is called from
`Transform.swift:268` and gated to the Laguna `.gemma4` family. **There is no
runtime consumer anywhere in `Sources/`.**

**21d. The arithmetic.**

```
affine-4 g64 today   32 B nibbles + 2 B scale + 2 B bias = 36 B / 64 elements
with a uint16 index  32 B nibbles + 2 B index            = 34 B / 64 elements
byte reduction       2 / 36 = 5.56 %
against >= 82 % streaming share -> local round floor >= 4.55 %
```

E97's metadata census makes this lossless and exact: 498 tensors, 420,208,640
groups, 1.68 GB of metadata; **zero** tensors have 256 or fewer distinct pairs
(minimum 911), so an 8-bit table is impossible, but the **maximum** is 7,846, so
an aligned uint16 index is lossless for all 498 and costs only 5.17 MB of
tables.

The engineering crux is buffer plumbing:
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp` is **not**
editable and passes exactly three arrays (`w`, `scales`, `biases`) with fixed
shapes. Three routes exist: repurpose `scales` as the bitcast index and
`biases` as the table; bypass the MLX op with `MLXFast.metalKernel`, as
`Qwen35.swift` already does for Gated DeltaNet; or index the bias only.

---

---

## Board and submission state

- **Crown** `bc070b7b`, francip, 3.35922017, promotedSourceRef `fac135f2`,
  submission commit `561b0a14`, model label "Claude Fable 5". The note says it
  starts from `41bad1c` and changes one file, `Qwen35.swift`, on the proposal
  side only, taking E87 shortlist selection from about 23 dispatches to two with
  identical candidate sets and identical proposals. The diff is +409 / -0. **Not
  yet inspected in detail. High-value read.**
- **Our best promoted row** is `f04b102e` at 3.32824629.
- **Our best engineered row** is `b8b8b860`, serial-free rank 1 of 764.
- Yukon allows exactly one in-flight submission per account. Validation has
  taken 56 to 130 minutes; `7bef7d4c` took 94.
- `senpai/frontier-state.json` on the advisor branch records the `bc070b7b`
  crown. Note that `senpai/submit-official.sh` reads that file from
  `origin/main`, not from the advisor branch, and the ancestry gate at lines 220
  to 226 is already satisfied by the `0c90733d` value recorded there. The
  advisor copy is a campaign record, not a submission gate.
- **Fact 27.** A third pure resample is a bad bet: the required luck is 0.2018
  percent on a single draw, about a 20 percent chance. The single in-flight slot
  is a scarce research instrument, not a lottery ticket.

---
## 0. THE MEASUREMENT FLOOR. Read this before pricing anything.

Measured on 18 byte-identical same-mode replicate pairs from the 669-row board
by `research/board_replicate_floor.py`, and reproduced independently by alphonse:

| statistic | median abs pair gap | max | pair sd | per-run sd |
|---|---:|---:|---:|---:|
| **published** `(raw_beagle + raw_essays)/2` | **0.1907 %** | 0.6833 % | 0.277 % | **0.196 %** |
| **serial-free** (board-mean serial substituted) | **0.1194 %** | 0.3449 % | 0.160 % | **0.113 %** |

**One ranked pair resolves nothing below `0.55 %` published or `0.32 %`
serial-free.** Always price on the serial-free statistic: it is `1.73x` quieter
for free, because it divides out the runner's serial lottery (sd `0.166 %`).

Consequences that are now campaign policy:

- A `7/7` same-sign per-prompt result is the signature of a **run-level common
  shift**, not of mechanism strength. Tight per-prompt spread does not rescue it.
- Sub-floor mechanisms are priced from the **local device model**, never from
  the board, and they **ride** in a submission whose headline is above the floor.
- The promoted crown is a max-statistic. **Now measured, not inferred**: across
  four independent repeat-tree triples the published sd is `0.243 %` and the max
  of three draws sits **`+0.233 %` above the mean of those three**. ox-alpha's
  own three receipts of one unchanged tree read 3.32279, 3.32466 and 3.32795.
  The crown's true mechanism value is near **3.3202**. This revises the earlier
  `0.4 %` to `0.6 %` estimate downward and makes it concrete.
- Any promoted lever whose published delta is below `+0.0106` is below the
  floor. Stop citing those as evidence that a lever works.

### 0a. THE THIRD STATISTIC, AND THE QUIETEST ONE: IDENTIFIED ROUND COST `L`

`research/board_same_schedule.py`. Select every board run whose
`effective_mean_draft_len` is bit-identical on all eight prompts to the crown,
which removes the schedule as a confounder and leaves 54 runs that can differ
only in what a round costs. Fit the five `G = 2` prompts **centered on the width
centroid** `M = 6.1723`, so the level and the slope are orthogonal:

```
round_us(M) = L + S * (M - Mbar)

L : median 61,566.2 us   sd 0.90 %   noise about 0.09 %   -> identified, 10x
S : median  7,231.7 us   sd 2.73 %   within-run se 205 us -> NOT identified
```

- **`L` is the quietest official statistic available.** It averages five prompts
  instead of two. Use it to rank mechanisms; use serial-free to predict a
  published draw.
- 🔴 **Never fit the raw intercept.** A five-point line over `M` in
  `[5.38, 7.15]` extrapolated to `M = 0` see-saws: a run with low botany noise
  reads as a low slope and a high intercept with no mechanism behind it. Ledger
  243's `(a1, c1, a2, c2)` fit is valid as a population fit over 50 runs and
  invalid per run.
- 🔴 **The per-row verify slope is not resolvable by one official run.** In 54
  runs no solver has ever lowered it. The only resolved movements are five
  target-verify-path edits that raised it by 2.6 % to 10.8 %. **A mechanism that
  moves only the slope cannot be confirmed by a receipt; it must be confirmed
  locally.** A mechanism that moves the level can be confirmed by one run.

Retired by this: the "same-mode residual sd 0.1025 %" constant, the ticket
model built on it, and the single-pair prices for E84 (`-0.109 %`), E85
(`-0.199 %` and `+0.022 %`) and the `8819b108` Q-row shrink (`+0.035 %`). See
ledgers 240 and 244.

---

## 0b. THE DEPTH-4 DOMINANCE THEOREM. Local only. Advisor error 43.

> 🔴 **CORRECTION, 2026-08-21.** This theorem holds on our M4 Pro and **not** on
> the ranked M5. The ranked cost curve of section 0d gives
> `C(M=5)/C(M=4) = 53,108/43,162 = 1.2304`, which is **below** the 1.25 ceiling
> the proof needs. Depth 4 is not dominated on the machine that scores us. The
> proof below is correct; only its second measured input changes. Keep it as the
> local statement and never quote it as a ranked one.
>
> Ranked flat-`q` crossovers, against the local ones:
> depth 3 versus 4, ranked `q* = 0.9682`, local never;
> depth 3 versus 7, ranked `0.9253`, local `0.9728`;
> depth 4 versus 7, ranked `0.9098`, local `0.8428`.
>
> 🔴 And flat-`q` ranked modelling is itself invalid for pricing the schedule.
> Every measured ranked accept rate, 0.834 to 0.903, sits below the 0.9253
> crossover, so flat `q` says depth 3 everywhere, yet the measured adaptive
> schedule beats every flat-`q` fixed depth by about 10 %. Ranked acceptance is
> strongly heteroscedastic and the shipped schedule already exploits it.

E92 measured the production round-busy cost at every verify width. Depth 4 is
the most expensive draft depth of 2 through 8, by `20.0 %`, because verify
width 5 is where `G = ceil(M/4)` increments in the `quantized.h:1924-1977` WIDE
switch. The marginal step into width 5 is `39,865.7 us`, which is `3.48x` the
step into 4 and `3.40x` the step into 6.

The theorem needs no cost-model fit. With `a_i = prod_{j<=i} q_j`, acceptance
probabilities at most 1 give `a1 >= a2 >= a3 >= a4`, so

```
Y(4)/Y(3) = 1 + a4/(1 + a1 + a2 + a3) <= 1 + a4/(1 + 3*a4) <= 1.25
```

for **every acceptance profile that can exist**. Measured against it,
`C(w5)/C(w4) = 126,103.1/86,237.4 = 1.4623`. Since `1.25 < 1.4623`, a depth-4
round is dominated by a depth-3 round unconditionally. Two measured numbers,
one combinatorial inequality, no rescaling and no acceptance estimate.

Margins, from `research/depth4_dominance.py`:

| normalisation | M4 Pro measured | M5, cliff `1.126x` flatter |
|---|---:|---:|
| `C(w5)/C(w4)` against the `1.25` ceiling | `17.0 %` | `12.8 %` |
| marginal step into width 5 must fall | `45.9 %` | `39.1 %` |
| rescaled `m4` must fall | `25.5 %` | `16.2 %` |

`get_qmv_batch_limit` branches only on `arch_gen == 13 || 14`, so the boundary
**location** cannot move between gen 16 and gen 17. **`snap4` transfers.**

**What is settled and what is not.** The dominance is settled. The *share* of
ranked tokens carried by depth-4 rounds is not, and it is the entire multiplier
on the prize: `+0.80 %` at the local `6.4 %` share, `+1.86 %` at `15 %`,
`+2.68 %` at ledger 207's `21.6 %`. The ranked walk sits at mean chosen depth
`4.3818` on beagle and `5.0870` on essays against the local fixture's `6.359`,
so the local share understates the ranked one. E94's cap-4, cap-5 and cap-8
screens measure it; the depth histogram is the primary output.

**Guard rail.** `h = 0.32`, uniform shallowing, scored `2.84585` ranked, which
is `-14 %`. Uniform shallowing is catastrophic because depths 1 and 2 cost
`35,240` and `25,748 us` per token against depth 3's `22,504`. Only a
**targeted** guard is alive. `amin` and `amine92` stay screens.

Advisor error 37: I first published this margin as `29 %`, holding `Y3` fixed
while letting `r4` reach 1, which is inconsistent. Edward's `0.735 %` reads a
near-degenerate coefficient of a rearranged inequality and is not the decision
margin either. See ledger 241.

---

## 0c. THE WHITE-BOX ROUND MODEL. Five constants that predict the machine.

Askeladd fitted the target verify pass from a per-dispatch census on his Mac at
384 tokens (E95 rung 2). Combined with his E93 head model:

```
verify_us(M) = 10,920 + 27,377 * G + 10,268 * M
head_us(d)   =  2,560 +  2,226.5 * (d - 1)
M = d + 1,   G = ceil(M / IPG),   IPG = ceil(M / ceil(M / 4))
```

The three verify parameters are separately identified. `c = 10,268` comes from
within-`G` variation at M = 3 to 4, 5 to 6 and 6 to 8, which all give the same
number. `b = 27,377` comes from the `G` step at M = 4 to 5. `a` is the residual.

**It predicts edward's independent E92 production sweep to within `0.66 %` at
depths 3 through 8** — a different Mac, a different session, a different
instrument, a different token window. Errors are `+0.17`, `+0.12`, `+0.66`,
`+0.54`, `-0.13` and `-0.21 %`. It fails only *below* its fitted range, at
d = 1 (`-12.0 %`) and d = 2 (`-1.2 %`).

This is the first white-box cost model of the scored round the campaign has
held. Everything in section 0b, the depth-price defect below, and the pricing
of every schedule arm now comes out of it.

### The shipped depth price is wrong at exactly one cell

`Qwen36MTPBlockSession.swift:904-911 makeUniformDepthPrice()` is the live arm.
It prices `T(d) = V + d*h*V` with `h = 0.18` and `V` flat in width, so every
step costs `11,600 us`.

| step into verify width M | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| measured marginal us | 5,330 | 5,003 | 11,459 | **39,866** | 11,740 | 12,589 | 13,526 | 40,072 |
| shipped price us | 11,600 | 11,600 | 11,600 | **11,600** | 11,600 | 11,600 | 11,600 | 11,600 |

**It under-prices the step into verify width 5 by `3.44x`**, and with
`segmentedVerifyDepthCap = 7` that is the only `G` boundary in the legal range.
The stale `measuredRawDepthPrice` at `:946-955` puts the cliff at width 6, one
position too late, which is why `pbfit` won `3.5 %` on the old dispatch table
and lost it on the crown table.

### The optimum is bimodal: 3 or 7, never 4, 5 or 6

Cost per accepted token at flat `q`, cap 7, head cost included: **depth 4 never
wins at any `q` at or below 1.0**, and is `16.9 %` worse than depth 3 even at
perfect acceptance. **Depth 3 wins below `q = 0.9728`; depth 7 wins above it.
Depths 4, 5 and 6 are never optimal.** E92 measured flat `q = 0.9551` and
hot-head depth-7 rounds back-solve to `q` near `0.966`, both below the
crossover — which is exactly why fixed depth 3 at `22,504 us/token` beat
adaptive depth 7 at `22,986`.

### The open contradiction inside the model

`b = 27,377 us` for a `14,412 MB` weight pass reads as **`526.4 GB/s`, which is
`1.99x` the `265.0 GB/s` DRAM ceiling**. The same contradiction appears from
the other end: at M = 1 the round busy is `64,445 us` and the weight stream
alone needs `54,385 us`, leaving at most `10,060 us` for all non-qmv work while
`a` alone is `10,920 us`.

**Leading hypothesis:** every x-group reads the full weight tensor, but if the
`G` groups run concurrently the later groups hit in the system-level cache.
Raising `G` would then add latency and issue cost rather than bytes, `b` would
not be a bandwidth term, and **byte reduction could not reach most of it.**
Askeladd's isolated `qmv_fast_crossrow_affine4_g64_wide` probe at `100.3 MB`
against `11.8 MB` cache-resident discriminates this directly and is running.

Until it settles, **no byte-reduction mechanism may be priced against the `b`
term**, and the `43-45 %` bytes / `46-49 %` arithmetic split stays out of the
ledger.


---

## 0d. THE RANKED COST CURVE AND THE TRANSFER TABLE. Read this before pricing anything.

Until 2026-08-21 every price in this campaign was a **local** price with an
unmeasured transfer to the ranked M5. The transfer is now measured for one work
class and derived for the other two, and the three differ by a factor of six.

### The ranked curve, recovered from the official board

`effective_mean_draft_len` in `officialMetrics.per_prompt` is the exact rational
`total_drafts / total_rounds`, where the denominator counts non-drafting rounds
too. `Fraction(dl).limit_denominator()` recovers the ranked round count. With 512
decode tokens and `mtp_seconds_per_token_mean` that gives the ranked cost of one
round at a known verify width `M = drafts + 1`.

Fitted independently on **50 official runs on the reference schedule**:

```
harness=ranked, M5
G=1, M=1..4 : round_us = 27,181.5 + 3,995.1 * M     cv(a1) 0.85 %, cv(c1) 4.73 %
G=2, M=5..8 : round_us = 16,943.2 + 7,233.0 * M     cv(a2) 5.57 %, cv(c2) 2.79 %
```

| M | G | ranked us | local us | ratio | ranked marginal | local marginal |
|--:|--:|---:|---:|---:|---:|---:|
| 1 | 1 | 31,177 | 64,445 | 2.07 | — | — |
| 2 | 1 | 35,172 | 69,776 | 1.98 | 3,995 | 5,330 |
| 3 | 1 | 39,167 | 74,778 | 1.91 | 3,995 | 5,003 |
| 4 | 1 | 43,162 | 86,237 | 2.00 | 3,995 | 11,459 |
| **5** | **2** | **53,108** | **126,103** | 2.37 | **9,946** | **39,866** |
| 6 | 2 | 60,341 | 137,843 | 2.28 | 7,233 | 11,740 |
| 7 | 2 | 67,574 | 150,431 | 2.23 | 7,233 | 12,589 |
| 8 | 2 | 74,807 | 163,957 | 2.19 | 7,233 | 13,526 |

🔴 **The ranked group-boundary cliff is +23.0 %. The local cliff is +46.2 %.**
That is why `h = 0.32`, a uniform shallowing, scored 2.84585 = -14 % ranked.

Two structural differences, both load-bearing:

1. Locally the per-row slope is **flat** at 12,494.5 us across the boundary and
   the whole step sits in the group term `b`. On M5 the slope **nearly doubles**,
   3,995 to 7,233. The machines differ in shape, not only in scale.
2. Because `c1 != c2`, the form `a + bG + cM` is **not identifiable** on ranked
   data. A naive least-squares fit with free round counts returns `b = -20,374.7`.
   Use two independent lines plus the physical constraints.

**Caveat to carry on every use: the ranked round counts are inferred, not
measured.** Internal validation is that four parameters fit eight prompts to 1.3 %
and stay stable across 50 independent runs. Instrument
`_advisor_scratch/rankedcurve.py`.

Ranked round counts per 512 tokens: beagle 110, essays 92, republic 93,
medicine 90, botany 81, travel 212, drama 252, plutarch 487 of which 449 are
non-drafting. Ranked accept rates: beagle 0.834, botany 0.866, medicine 0.892,
essays 0.897, republic 0.903, travel 0.533, drama 0.449, plutarch 0.333.

### The work-class transfer table

Combining the ranked curve with the five-constant local model of section 0c and
the **measured** arm C head transfer, at the ranked beagle width `M = 5.3818`:

| work class | local us | local share | ranked us | ranked share | transfer |
|---|---:|---:|---:|---:|---:|
| proposal head | 10,090 | 7.70 % | 1,019 | 1.82 % | **0.237** measured |
| **per-row verify** | 55,261 | 42.18 % | 36,088 | **64.59 %** | **1.532** derived |
| fixed / launch | 65,674 | 50.12 % | 18,763 | 33.58 % | **0.670** derived |
| round | 131,024 | | 55,870 | | machine 2.345x |

🔴 **This is the strategic fact of the campaign.** M5 is much faster than our
M4 Pro at bandwidth-bound streaming, which is where the proposal head lives, and
only 1.53x faster at per-row verify work. **We have been spending our strongest
students on the axis with the worst transfer.** The axis with the best transfer is
two thirds of the ranked round and completely un-attributed.

The head transfer is measured, not modelled. Arm C is the first mechanism whose
ranked and local effects were both measured on the same tree with no confound:
a 29 % head saving worth 2.233 % locally at the ranked draft depth produced
**+0.529 %** on ranked beagle. Transfer 0.237 depth-corrected, 0.327 uncorrected;
thorfinn independently inferred 0.350 from the aggregate published move.

### The four pricing rules

```
head-side local gain    -> multiply by 0.24 to 0.35
per-row verify gain     -> multiply by about 1.5
fixed / launch gain     -> multiply by about 0.67
acceptance loss         -> multiply by 1.0, ALWAYS
```

The last one is the trap. A proposal the drafter fails to retrieve is rejected by
the target on any machine, so an acceptance penalty never shrinks on transfer
while the byte gain that bought it shrinks by three. This is what moved
`derived15` from an accepted +0.23 % to +0.30 % to an unknown sign in the range
-0.5 % to +0.14 %.

### PREFILL IS NOT IN THE SCORE

For every prompt of every scored run,
`raw_ratio_of_means == serial_seconds_per_token_mean / mtp_seconds_per_token_mean`,
exact to all printed digits. Mode string `qwen-mtp-paired-decode-only`.
`prefill_seconds_per_token` is reported and never enters `raw_p`.

Every "decode share" multiplier the campaign has used was wrong. **The correct
multiplier is 1.0.** Ranked prefill is about 0.527 s per leg, so M5 prefill is
7.6x faster than our local 4.04 s, but it buys nothing either way.

---


## 0e. THE MEDIAN IS LOCKED, AND THE CENSUS METHOD HAS A CEILING

Two results from ledger 245. Both change how work is priced. Read them before
section 1, which they supersede.

### 0e.1 The exact score function, and the exact value of each prompt

Instrument: `python3 research/board_median_lock.py`. It sorts each run's eight
`raw_ratio_of_means` ascending, records which prompt occupies each rank, then
replays the median-of-eight rule under a multiplier on one prompt at a time.
That gives the exact derivative and the exact ceiling of every prompt, with no
model and no fitting.

**Rank occupancy over the 81 published runs at or above 3.25:**

| rank | occupant |
|---|---|
| 4 | **beagle, 100.0 % — every one of 81 runs** |
| 5 | essays 66.7 %, medicine 19.8 %, republic 7.4 %, botany 6.2 % |

**The score is therefore exactly:**

```text
published = 0.5 * raw_beagle + 0.5 * min(essays, medicine, republic, botany)
```

Only the first term is free. The second is pinned by a four-prompt cluster that
spans less than 1.6 %, so improving essays alone simply hands the 5th slot to
republic. Exact single-prompt value at the crown `8819b108`:

| prompt | raw ratio | published gain per 1 % | ceiling | reached at |
|---|---:|---:|---:|---:|
| **beagle** | 3.185167 | **+0.4785 %** | **+4.6625 %** | 9.8 % |
| essays | 3.470732 | +0.3721 % | **+0.3721 %** | 0.8 % |
| travel | 2.188496 | 0 | +4.6625 % | 59.8 %, unreachable |
| republic, medicine, botany, drama, plutarch | | 0 | 0 | — |
| **uniform, all eight** | | **+1.0000 %** | unbounded | |

The same shape holds on our own `cb8aeefb`: beagle +0.4801 % per point with a
+4.5146 % ceiling, essays +0.5199 % per point with a +0.5269 % ceiling, every
other prompt zero.

**Four consequences for how we assign work:**

1. **A beagle-only mechanism is worth 12.5 times an essays-only mechanism.**
   Essays saturates after 0.8 % and pays nothing after that.
2. Uniform mechanisms keep the full 1.0 multiplier and remain the best value per
   unit of engineering. Nothing here demotes them.
3. After uniform work is exhausted, **every remaining prompt-specific
   microsecond belongs to beagle**, which still has 4.66 % of untouched ceiling.
4. 🔴 **Beagle's deficit is an acceptance deficit, not a cost deficit.** Beagle
   accepts 0.834 at mean draft length 4.382; essays accepts 0.897 at 5.087. Their
   round costs sit on the same shared curve. Beagle is simply the least
   predictable prompt in the pool.

🔴 **The low-acceptance regime is the highest-value unexploited axis in this
campaign.** Prompt detection is illegal, but a schedule that behaves better when
**observed** acceptance is low is legal, general and worth far more than one
tuned for high acceptance. The campaign has implicitly tuned for the opposite.
Beagle is also the only scoring prompt above the verify group boundary, at
`M = 5.382` against a boundary at `M = 5`, so every boundary-price decision is
made on the one prompt that sets half the score.

### 0e.2 MLX dispatches are concurrent, so a per-dispatch census is an upper bound

Source, all in `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp`,
which is **not** editable:

- `:545-548` every compute encoder is created with `MTL::DispatchTypeConcurrent`.
- `:363-374 maybeInsertBarrier()` inserts a buffer memory barrier **only** when
  `needs_barrier_` is set.
- `:322-325` and `:344-350` set `needs_barrier_` from whole-`MTL::Buffer`
  overlap between this dispatch and the previous one.

Three consequences:

1. **Independent dispatches inside one command buffer overlap in wall time.**
   Summing per-dispatch intervals double-counts.
2. `MLX_E58_BUFFER_LIMIT_MB=1` measures a cost the round does not pay. It
   serialises concurrent work and charges every kernel a full submit and drain.
   An isolated per-kernel time is an **upper bound** on round contribution.
3. **The error size is predictable.** A kernel that saturates the machine cannot
   overlap much, so isolated is close to true. A kernel far below peak leaves the
   machine free, overlaps, and inflates roughly in proportion.

This is exactly why E93's head census cross-validated to 0.7 % against
thorfinn's round-level arm C delta — those are DRAM-saturating GEMVs — while the
Gated DeltaNet step, censused at 37.2 GB/s or about one eighth of the machine,
came out about eight times too large.

**What survives:** every round-level and leg-level measurement. Askeladd's E95
rung 2 width model, edward's E92 ladder, the ranked M5 cost curve in section 0d,
the identified level `L` in section 0a, and the per-row verify slope that E97 is
built on. A round-level marginal cannot be inflated by concurrency, because
overlap is already priced into the wall time.

**What needs a caveat:** every per-dispatch attribution. E95 rung 3 and
thorfinn's E87 §8 isolated chain cost of 113.78 us/draft are upper bounds.

🔴 **A lever this exposes, candidate and unassigned.** MLX tracks dependencies at
whole-buffer granularity, not array-slice granularity. Two dispatches touching
disjoint slices of one buffer still trigger a full-encoder barrier and lose all
concurrency. Our editable surface writes into shared buffers, for example
`KVCache.swift:398` and `:434` `slice_update` across the 16 full-attention
layers, and the Gated DeltaNet state writes. `device.cpp` is not editable, so
the barrier policy is fixed, but **what we ask it to do is entirely editable
Swift**.

### 0e.3 Advisor error 45

I read a per-dispatch census rate of 37.2 GB/s, one eighth of the machine, as
evidence of headroom. It was evidence that the census method does not apply. **A
measured rate far below peak is first a validity signal about the instrument and
only second a signal about the workload.** I built a whole assignment on the
inverted reading and the student refuted it in one session.

---
## 2. The two mechanisms that decide this campaign

### 2.1 E89 — the ranked measurement lottery is efficiency-core placement

Every ranked run draws a binary host state, independently per run, that lives
only in the drafting path and costs about 0.9 ms per drafting round. It is worth
**1.016 % of serial-free score on our own tree** and 1.409 % median across 22
pairs of other people's trees.

**Alphonse has named the mechanism with a direct measurement.** Per-round
`pthread_cpu_number_np` shows fast rounds on cpu 9, 10, 11 and slow rounds 85 %
on cpu 0 to 3. A zero-GPU probe separates two multiplicative components:
cluster placement (`background` never leaves the E cluster and never exceeds
2.600 GHz; `userinteractive` reaches 4.513 GHz on a P core, a 1.74x ratio) and
DVFS residency (a P core at 0.4 % duty only reaches 3.67 to 3.75 GHz).

**The fix is one line**: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`
behind a per-thread guard, called before the round clock starts. A pilot on one
binary, back to back, 128 tokens: slow-round prevalence 1.00 -> 0.06, host phase
median 3,339 -> 632 us, **`mtp_seconds_per_token` 0.053969 -> 0.053772, +0.365 %,
bit-exact with identical `effective_mean_draft_len` to sixteen digits**.

**Secondary benefit, possibly larger than the primary.** The host state is what
destroys our local paired estimators. Thorfinn's composed-tree pair lost 58 of
63 paired rounds to it. The fix repairs the campaign's measurement instrument.

**One open discriminator.** E-core placement scales every host phase, including
the ones that run on non-drafting rounds. Scaled to ranked, that predicts a
plutarch mode effect of about +0.5 %. Observed plutarch mode sd is 0.032 % with
r = +0.043. A 16x miss. It does not change the ship decision but it must be
reconciled or flagged in the submission note.

### 2.2 E87 arm C — the largest single mechanism on the board

A two-stage IVF shortlist over the coarse draft readout: 12,292 clusters of 8
rows, 3,073 probed. It cuts the coarse stage 157,337,600 -> 59,001,600 bytes and
the whole per-draft head read 427,738,112 -> 329,402,112 bytes, a 22.99 %
reduction, with all tokens matched on every leg.

**Local: -1.688 % leg total, -2.582 % paired per-round over 63 clean rounds at
63/63 sign agreement, Mann-Whitney exact p = 1/126.** Consolidated ranked price
**+1.5 % to +1.75 %**, which is about 10 sigma against the 0.166 % serial
lottery.

**It survives composition with E85 and E90 unchanged.** The merge onto the
campaign base produced zero conflicts. Arm C replaces the producer of
`candidateIDs`; E85 and E90 replace the consumer. The absolute per-draft saving
is **-619.9 us on the composed tree against -616.4 us on the r1 tree**. Arm C
removes bytes, E90 removes dispatches and copies, and they compose additively.

**The delivery blocker is now removed.** In r1 the submitted surface was a no-op
because `mtp-head.manifest.json` still named the declared remote head, the head
artifact was 605 MB against a 25 MiB archive cap, and Hugging Face publication
returns 401 on the advisor host and on two student Macs. Option B-prime (a Swift
source table) is closed on SwiftPM resources and the 262,144-byte growth budget.
Option C's clean form is closed on the archive cap.

**Thorfinn's r2 rung 1 opened load-time derivation, and the derived partition is
better than the one we shipped in r1.** A balanced bisecting 2-means rule,
`research/e87_bisect.py --balance half`, no RNG, 14 levels, **4.87 s**, cheap
enough to run inside a model load:

| partition | probe `p` | misses / 18,092 | worst-domain `m` | gate 3.0e-3 |
|---|---:|---:|---:|---|
| **bisect, derived** | 0.25 | 4 | **2.266e-4** | pass, 13.2x inside |
| bisect, derived | 0.15 | 10 | 7.554e-4 | pass, 4.0x inside |
| plain k-means, r1 | 0.25 | 11 | 1.079e-3 | pass |
| plain k-means, r1 | 0.15 | 36 | 3.237e-3 | **fail** |

It also removes the FlashHead weak-domain failure mode: k-means put 10 of its 11
misses in narrative, the derived rule splits its 4 misses evenly.

**Provenance is closed.** `research/e87-coarse-identity.json` shows the shipped
`mtp.draft_lm_head.{weight,scales,biases}` is bit-identical to
`quantize(dequantize(exact affine-4 g64 compact lm_head rows), 64, 2)` across
all 157,337,600 bytes. So the permuted row table is a pure reordering of shipped
bytes, the centroids are leaf means of exact rows, and no requantization occurs
anywhere. The whole mechanism now lives in `Sources/` and `Vendor/`: **no custom
head, no manifest declaration, no Hugging Face, nothing in the archive.**

**Ship `p = 0.25`, not `p = 0.15`.** His optimum of `byte gain - 206.6 * m`
prefers 0.15 at +2.017 % against +1.827 %. The `206.6` is the least trustworthy
number in the campaign, the misses are about 0.45 per leg at 0.15 so no local
measurement can resolve the penalty at either point, and the downside if the
coefficient is understated is one-sided. `p = 0.25` also has a measured local
anchor at exactly 22.99 % byte removal from the r1 session. `p = 0.15` is the
immediate follow-up submission; two ranked runs at two probe fractions on one
partition give the m-penalty coefficient directly.

**One harness defect cost him a leg and is now documented.** His runtime log
channel produced nothing because `benchmark.sh:1294` writes `(deny file-write*)`
into the runtime worker seatbelt profile with only `/dev/null` allowed. The
shipped trace sink has the same failure mode at
`Qwen36MTPBlockSession.swift:788-794`, falling back to a stderr that `mtp-timed`
swallows. **Untimed capture legs need `MLXFAST_NO_SANDBOX=1`.**

---
## 5. Standing operating rules

- 🔴🔴🔴 **A SHARE OF ROUND TIME IS NOT A PRICE. A CONTROLLED REMOVAL AT THE
  SCORED OPERATING POINT IS A PRICE.** Advisor error 53. I priced E98 at
  +4.7 % to +5.5 % published from `byte_share x streaming_share`; the scored
  kernel converts metadata bytes at about 0.17 of that. Never multiply an
  isolated-cell effect by a share of round time and call the product a
  prediction. **Advisor error 56 is the same error committed again three days
  later** on E100's 512-token band, where I pre-registered −6 % to −12 % at
  depth 4 and −0.5 % to −1.5 % at depth 8 and both were refuted on the wrong
  side.
- 🔴🔴 **READ THE REPOSITORY'S OWN PRIOR EXPERIMENT BEFORE PRICING A ROUTE.**
  Advisor error 54. I briefed `rows_per_simd = 2` as an untried escape and
  built two rungs and one assignment order on it. `research/e76-results.md`
  had already measured it on gated GPU time and had already published the
  `g17s` register ladder I then sent a student to measure.
- 🔴 **`peak_live_regs` IS NOT A REGISTER COUNT.** Finding 27b. It is AIR SSA
  liveness at about 1.7x the machine allocation. Use it as an order-preserving
  relative screen only. For a level, use `xcrun metal-tt` against the target
  generation.
- 🔴 **A shared kernel entry point taxes every width when any one width
  widens.** Finding 27a and 27c. Price that flat tax before the mechanism's own
  gain.
- 🔴 **Price every proposal against the bandwidth floor before assigning it.**
  Divide total weight bytes by DRAM peak and compare with the measured round.
  Finding 21 says the round is at least 82 % weight streaming, so any mechanism
  that lives inside the remaining 18 % has a ceiling under 3.8 % of the round.
  **Advisor error 49 is exactly this check not being run**, and it cost three
  students a full generation each.
- 🔴 **A per-dispatch or least-squares attribution is not a measurement.** The
  8,112.6 us Gated DeltaNet figure came from `report_fixed()` in
  `research/e95_verify_census.py`, which spreads residual by byte pro rata and
  has no "belongs to no kernel" bucket. Only a removal arm, a repeat-dose slope
  or a round-level ABBA contrast prices a component.
- **The published score is
  `0.5 * raw_beagle + 0.5 * min(essays, medicine, republic, botany)`.** Report it
  as the headline of every per-prompt comparison. `mean7` stays as a mechanism
  diagnostic only; it is not the score and it has already cost us one crown.
- **Keep the one in-flight Yukon slot occupied with the best available real
  candidate.** Every official submission must carry a content delta we can name
  and price; comment-only resamples are retired. 🔴 **Yukon's dedupe is
  defeated by any byte, including the free-text `note` field of
  `mtp-head.manifest.json`** (Finding 24), so a bare resample is mechanically
  possible. `program.md` forbids it. **We answer rival lottery draws with
  mechanism size, not with draws.**
- **Report the serial-free score with every published score.**
- **Carry `sandbox=on|off` in the experiment identity tuple.** `--local-submit`
  runs inside the Seatbelt profile written by `benchmark.sh:1266-1307`;
  `research/e79_trace_leg.sh` sets `MLXFAST_NO_SANDBOX=1` and runs outside it.
  Absolute times from the two configurations are not comparable. The profile
  denies every file write except `/dev/null` at `:1294-1295`, so any research
  sink that opens a file silently produces nothing on a sandboxed leg.
- **The 0.0815 % per 1 % byte law is an average over the whole 428 MB head
  stream.** Do not apply it to one tensor. Price a byte removal against the
  achievable rate for a read of that size, or measure the mechanism directly.
- **The local achievable read bandwidth is 265 GB/s.** Size-matched: 274 to 276
  at 157 MB, about 265 at 330 to 428 MB, about 260 above 1 GB, 403 to 430 at
  16 MB which is cache. 226.035 and 245.2 GB/s are retired.
- **A byte model is valid only when achieved bandwidth is held constant.**
  Working-set reduction and byte reduction are distinct levers.
- 🔴 **Price every local gain through the FINDING 22 TWO-CLASS transfer law.**
  The earlier four-row table's derived per-row-verify factor of 1.5 and derived
  fixed-and-launch factor of 0.67 are **RETIRED**; they were built by
  attribution, not measurement.

  | work class | multiply local percentage by | basis |
  |---|---:|---|
  | STREAM work, at or near DRAM peak | **1.00** | `local rate / ranked rate` |
  | LATENCY work, well below peak | **2.40** | ranked-to-local round ratio |
  | proposal-head BYTE removal | **0.236** | MEASURED, E87 arm C |
  | acceptance loss | **1.00** | accounting identity |

  Corrected closure thresholds against a LOCAL cost: **stream class dead below
  0.160 %, latency class dead below 0.067 %.** A proposal the drafter fails to
  retrieve is rejected by the target on any machine, so an acceptance penalty
  never shrinks on transfer while the byte gain that bought it shrinks by four.
  🔴 **Finding 26 is an independent ranked confirmation of the latency class**:
  a rival's 10 to 15 us per draft removal was predicted at +0.081 % to
  +0.121 % and measured at +0.0803 %.
- 🔴 **Do NOT multiply a latency saving by 2.40 and then also divide it by the
  ranked round.** The division IS the amplification. Advisor error 52 was the
  opposite mistake: applying the 0.236 BYTE factor to a LATENCY mechanism, a
  12x understatement that retired a whole live theme for a day.
- **A bit-exact change cannot move a draft length.** `effective_mean_draft_len`
  is a free exactness detector.
- **Price an issue-count change from translated machine text, never from AIR.**
- **Carry an instruction counter in every host-state measurement.**
- **Publish the per-leg host-state stratum before any pooled number**, using the
  arm-blind 1,500 us absolute host-phase gate.
- **plutarch, prefill and serial are mechanism-breadth controls, not mode
  controls.** plutarch correlates with the mode at r = +0.043.
- **Read `sd7` before `mean7`.** sd7 above about 0.35 on a same-schedule pair
  means cross-mode; quarantine the pair.
- **Group ranked comparisons by the scored-surface tree digest first**:
  `git ls-tree <branch> Sources Vendor mtp-head.manifest.json`.
- **A promotion is a draw, not a measurement.**
- **An isolated-cell harness over-states recoverable time** — by 3.63x in E78
  and 33x in E91.
- **Leg totals overstate small effects by up to 4x.** Use paired per-round
  medians with the depth sequence held identical.
- **Freeze the commit before a gate leg.** Land logger changes between legs,
  never inside a job.
- **Research instruments go in `Tests/` or `research/`, never `Sources/` or
  `Vendor/`.** Deletion is the default for a closed axis's knob.
- **When a student's measurement contradicts the advisor's model, the
  measurement wins and the advisor retracts in writing before they spend GPU.**
- **Verify every claim about the scored surface with a repository-wide grep
  before it becomes an instruction.**

---

---

## Student board

All four students are running. Each has a different physical Mac; the advisor is
co-located with edward only.

| student | PR | experiment | files owned this cycle |
|---|---|---|---|
| thorfinn | #121 | E120 own the QMV dispatch, rung 5e decisive | `Qwen35.swift`, `research/`, `Tests/` |
| alphonse | #127 | E126 Route B overlap and the in-situ transfer law | `quantized.h` and `mlx-generated/quantized.cpp` (sole owner, no comment lines), `research/` |
| edward | #125 | E124 delete the MTP head precision islands | `Qwen35.swift` restricted to `sanitize` and a research-only arm selector, `research/`, `Tests/` |
| askeladd | #126 | E125 isolated-to-in-situ transfer law | `research/` only |

🔴 **Two students share `Qwen35.swift` this cycle.** Thorfinn's Route B is the
most valuable branch in the campaign, so edward's shippable diff is restricted to
the eight-line install block in `sanitize` at `:4339-4346`. He is instructed
**not** to delete the now-dead `qkv(_:)` fast path at `:2281-2300`; that becomes a
cleanup PR after Route B merges.

`sdpa_vector.h` is editable and unowned. The SDPA cross-simdgroup reduction tail
is closed, so it stays unowned unless a new mechanism appears there.

`GatedDelta.swift`, `backend/metal/quantized.cpp`, `backend/metal/custom_kernel.cpp`,
`backend/metal/device.cpp`, `backend/metal/sort.cpp`, `ops.cpp` and
`scaled_dot_product_attention.cpp` are **not** editable.

Askeladd's Mac cannot reach the 40 degree thermal gate, so he cannot run a
submission chain. Any candidate he produces must be handed to another student for
the pre-submit chain. Ungated counterbalanced arms remain a standing permitted
measurement mode, so this no longer blocks his timed work.
