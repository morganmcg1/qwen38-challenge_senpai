# SENPAI Research State

- 2026-08-20 14:20 UTC
- Most recent research direction from the human researcher team: Issue #22 —
  execute aggressively toward the winning frontier. No new human direction since.

## Where the campaign stands

Crown: **3.25238228**, submission `9ad17378`, solver Lieisyourlie, source
`bfab0de58d43453e506523707e1720a3485570f4`. `upstream/main` is that exact source.

Our best official score: **3.23588901**, submission `9b241879` (arm 2). Deficit to
the crown: **0.508 %**.

Campaign base: `d79924a357f25a130bafe3fc5c07b7d4c427f092`.

**Arm 3 returned and it is the most informative receipt of the campaign.**
Submission `2da69933-5202-4e0d-b336-c75945a45b9e`, candidate commit `389676fb`,
`rejected`, official score **3.21125713**. It carried E68's measured depth-price
vector, byte-identical to thorfinn's validated `4d467ca`. It cut pooled candidate
MTP seconds per token by **11.26 %**, raised the pooled ratio of means by
**12.99 %**, raised the worst prompt by **56.33 %** — and moved the published score
by **−0.66 %**, which is 0.61 sigma and not a significant loss. It is not the loss
that matters. It is that an 11 % throughput win collected nothing.

**No submission is in flight. The ranked slot is free.**

## The three rules arm 3 established

### 1. The score is a median of eight prompts, and only two of them are scored

The published score is `median(raw_1 .. raw_8)` = `mean(4th, 5th)` after sorting.
The eight shipped-schedule raw ratios split into two clusters separated by a gap of
0.94:

- low: plutarch 1.2528, drama 1.9163, travel 2.1795
- high: beagle 3.1201, medicine 3.3446, essays 3.3666, republic 3.3930, botany 3.4253

The 4th and 5th sorted values are **beagle and medicine**, the bottom two of the high
cluster. Three consequences, all now campaign rules:

- **Work that improves plutarch, drama or travel is worth exactly zero.** Arm 3
  proves it: it lifted plutarch by 56 % and the low cluster still did not reach the
  middle.
- **Improving beagle alone saturates at median 3.3556, which is +3.80 %.** Once
  beagle passes essays, the median becomes `mean(medicine, essays)` and further
  beagle gain is discarded. This independently reproduces the beagle ceiling
  `3.35549050` recorded earlier from the acceptance work.
- **A uniform improvement across the wide prompts is the only lever that moves the
  median by its own full size.** Kernel and runtime work is uniform. Schedule work
  redistributes, and redistribution is how arm 3 lost.

Had arm 3's 11.26 % landed uniformly, the median would have been about **3.64**.

Yukon field semantics, established and reusable: `officialMetrics
.mtp_decode_speedup_median` is the **4th** sorted raw ratio, and
`mtp_decode_speedup_raw_median` is the **published score**. Therefore the 5th sorted
ratio is `2 * raw_median - median`. Verified twice against receipts whose per-prompt
rows were still populated. The per-prompt block now returns nulls on both endpoints,
so this inversion is the only route to the two scoring prompts.

### 2. The mechanism, in full, and the local artifact that caused it

`costModelDepth` extends the draft while
`reach > marginal[depth] * (1 + expected) / cumulative[depth]`. The fitted price
vector `pbfit` differs from the shipped uniform 0.18 in two places that matter:

- index 0 is **33 % cheaper**, so the depth-0 threshold falls and low-acceptance
  prompts draft deeper. Plutarch went 1.2528 → 1.9585.
- index 4 is **61 % dearer**, so the depth-4 threshold rises from about 0.43 to
  about 0.74, above the reach a high-acceptance prompt carries there. The walk stops
  at depth 4, which is width 5. That is exactly beagle at 5.533 and medicine at
  5.768 — the two prompts the score is made of.

One vector produced both observations: on the local fixture it moved mass from
width 6 to width 5, and on the ranked host it moved the low cluster up.

**The index-4 spike is a local kernel-table artifact.** It comes from E68 rung 1's
local step into width 6, 27.308 ms against width 5's 13.405 ms, which is the
`<T,6,6>` cell. That cell carries a 16-byte spill frame on the local g16s register
file and does not spill on the ranked g17s host, and the ranked width curve is
1.126× flatter than the local one. We fitted a price to a local defect and paid for
it on ranked.

Disclosed confound: arm 3's base already carried the `DepthPrice` scaffolding on the
scored surface with `depthPriceArm = .ship` and a documented one-ulp difference in
`1.0 + 3.0*0.18`. Too small to explain an 11 % throughput move, but not provably
inert.

### 3. The local fixture is at the wrong operating point

| | local ship | local pbfit | ranked pooled | beagle | medicine |
|---|---:|---:|---:|---:|---:|
| mean verify width | **7.27** | 6.47 | **5.82** | 5.53 | 5.77 |
| share at M=5 | 6.41 % | 49.4 % | 24.1 % | | |
| share at M=6 | 29.49 % | 5.9 % | 33.4 % | | |
| share at M=9 | **43.59 %** | 30.6 % | **5.75 %** | | |

**The single local prompt over-weights M=9 by 7.58× and under-weights M=5 by
3.76×.** That is plausibly why our dispatch table ever chose `<T,9,5>`. It
invalidates local **whole-leg** numbers as arm rankings. Per-width and per-cell
numbers remain valid and can be reweighted onto the ranked histogram, so every
running experiment has been re-scoped to report per-width cells as the headline and
whole-leg time only as an additivity cross-check.

## What the competitor history gives us

**`upstream/main` is a linear chain of 76 commits, and every `Accept submission
<uuid>` commit is one promoted competition submission carrying its exact diff.** The
complete history of every rival, at diff granularity, is a local `git show`. Diff
before theorising about any competitor. Mapping all 56 promotions against the live
Yukon list gives four campaign-changing facts, recorded in full at ledger 216.

**1. The field is capped at `NA <= 4`.** `quantized.h:980` at `upstream/main` reads
`static_assert(NA >= 2 && NA <= 4)`. Set at `1033e1a`, never raised. No competitor
can select `<T,5,5>` or `<T,6,6>`. We are the only participant that has ever run
NA=5 or NA=6 on the ranked host, so the arm-2 receipt is information nobody else
holds.

**2. The crown's dispatch table is a constrained optimum.** `[3]` `[4]` `[3,2]`
`[3,3]` `[4,3]` `[4,4]` `[3,3,3]` is exactly "minimise the group count subject to
NA <= 4, then tie-break to fewer registers", verified cell by cell against the
legality rule `M % IPG != 1`. Their M=5, M=6 and M=9 cells are the cells they were
forced into. We are the only participant able to test the cells they wanted.

**3. 140 of the 154 editable files, 91 %, have never been changed by any promotion.**
Of roughly 90 editable Metal and kernel files the field has touched exactly two:
`quantized.h` and `mlx-generated/quantized.cpp`.

**4. The proposal head is the field's biggest lever by a wide margin.** Cumulative
gain across the 13 largest promotions: proposal head **+38.1 %**, schedule +29.0 %,
runtime +21.7 %, kernel +16.1 %. One manifest-only head swap (`deb63ad`) was worth
+7.5 % on its own. All five heads the field has used are requantizations of the
organizer's pinned bf16 weights. **Nobody has re-derived the weights.** Re-trained
and distilled heads are explicitly legal under the static review.

**The board is a noise band, not a ranking, and the organizer's own history proves
it.** 590 scored submissions of 855 total: 58 sit at or above 3.24, 78 at or above
our arm 2, the top 20 span 0.235 % and the top 10 span 0.164 %. A single ranked run
has a standard deviation of 0.756 %, and a difference of two runs 1.069 %. Promotion
`5068eb8` has a **tree hash identical to its parent** and was promoted for scoring
0.008 % higher; promotion `80021bc` changed one note string inside
`mtp-head.manifest.json` and was promoted for 0.062 %. Our rank of 79th is two
thirds of one run's noise, not a capability gap.

**The bar for spending the ranked slot is about +1.5 % of true ranked gain**, which
buys a 0.903 probability of beating the printed crown in one run. Below +0.5 % a
submission cannot be distinguished from a lucky run. One in-flight submission is
allowed, at about 2.5 hours each. The advisor has no GPU, so every arm must come
from a student-validated `--local-submit` snapshot.

## The current research focus

### 1. Find a uniform wide-prompt gain worth +1.5 %

This is the direct consequence of rule 1. Kernel and runtime work is uniform across
prompts; schedule work is not. Arm 2 is the existence proof: reverting our dispatch
table to the crown's was faster on **8 of 8** prompts, mean −0.383 %, sign test
p = 0.0039 — a small effect, but collected in full. The three live experiments are
all pointed at uniform mechanisms.

**Per-shape inner-group count (askeladd, PR #81) is the leading near-term ranked
candidate.** The defect is that the dispatch table applies one group count to every
shape while the optimum differs by shape. Four arms: ship, crown, hybrid at
n = 24928, hybrid at n = 8192. If a hybrid beats both uniform tables at M=5 and M=6,
which carry 57.5 % of the ranked round pool, that is a uniform gain and it is a
submission.

**The proposal head (alphonse, PR #82) is the field's biggest untouched lever.**
Head cost is paid on every round of every prompt, so a head win is uniform by
construction.

**The unattributed 22.6 % of the width tax (edward, PR #83) is the largest unnamed
quantity we have.**

### 2. RETRACTED: the register-occupancy model. The lever is grid width.

**I retract the occupancy model of ledger 216 in full.** Alphonse's E77 measured it
directly on 6,240 timed legs and destroyed it. The retraction was sent to both
students in writing before either spent further GPU time. What is void:

| retracted | replacement |
|---|---|
| Register file `B_R = 208 KiB` | **384 KiB measured locally**; about 496 KiB ranked |
| Group-count penalty `eps = 0.111` | void; fitted on the wrong denominator |
| Scale factor `kappa = 0.0600` | void; converts a void quantity |
| The graded register-tier prize table, "worth −0.38 % vs the crown" | void |
| The tier-staircase functional form | rejected; time rises smoothly, nearly linearly in R |

Occupancy is real but tiny: **`Omega(S) = (32/S)^gamma`, `gamma = 0.01346 ±
0.00065`**, 20 standard errors from zero, and the **total occupancy spread anywhere
in the 19-cell table is 0.52 %**. The M=6 inequality the receipt demands needs
3.34 %; occupancy supplies 0.31 %, short by **10.8×**. Worse, the model fails in
*sign* at M=5, 6 and 9: it says the crown should be 5.73 % slower, and the receipt
says the crown is 0.298 % faster. **Occupancy is excluded as the explanation of the
arm-2 receipt at every width.**

**The register axis is closed with a bound.** E76 swept it directly: the whole span
from 122 ranked registers down to 50 is worth −1.209 %, against a submission bar of
about +1.5 %. Reachability was never the constraint — `rps1lazy` reaches 75 ranked
registers with zero differing elements over 112 arm-width pairs — and every arm that
clears 91 registers is slower, the smallest single-shape penalty being +15.53 %.

**The register law itself survives.** Register count is a pure function of the
largest group in the partition, independent of M and of group count, 19 of 19 cells,
zero exceptions: largest group 2/3/4/5/6 gives 70/93/94/95/96 on g16s and
83/90/91/98/111 on g17s. The g17s ceiling is **126**, not 124. Local hosts have a 96
register ceiling, so `[6]` spills 16 bytes locally and does not spill on the ranked
host. **Any local arm above 96 registers measures spill, not occupancy**, and spill
is about 0.015 % per frame byte.

**The replacement mechanism is per-shape grid starvation.** E74 located the
working-threadgroup knee in situ at **1558 working threadgroups = 77.9 per core**,
with amplitude `A = 0.2132` and a 68 % profile interval of [67.2, 117.8] per core.
Below the knee a kernel does not fill the machine and time is set by grid width, not
by traffic. Working threadgroups are `ceil(M/IPG) * ceil(n/8) * B`, so **splitting a
group multiplies the grid**. Reading E33 of ledger 137 again with that instrument
gives the whole picture:

| family | n | working TGs at 2 groups | 1-group / 2-group |
|---|---:|---:|---:|
| `head.lm_head` | 248320 | 62080 | 0.9830 |
| `mlp.gate_up_fused` | 34816 | 8704 | 0.9941 |
| `linear_attn.in_proj` | 16480 | 4120 | 0.9947 |
| `full_attn.qkv_proj` | 14336 | 3584 | 1.0148 |
| `full_attn.o_proj` | 5120 | 1280 | 1.0414 |
| `linear_attn.out_proj` | 5120 | 1280 | 1.0492 |
| `mlp.down` | 5120 | 1280 | 1.0592 |

**The sign flips between 1792 and 2060 working threadgroups.** Wide shapes want fewer
groups; narrow shapes want more. **The optimal inner-group count is not the same for
every shape, and the dispatch table applies one group count to all of them.** That is
the defect, and it explains the receipt without any occupancy term.

Core count does not change the log-ratio gain for a shape deep below the knee. What
it changes is **which shapes are starved**: the boundary moves from n = 12464 at 20
cores to n = 24928 at 40 cores, pulling n = 14336 and n = 16480 across.

**The grid cannot be widened any other way.** The host grid
`grid_dims(M, ceil(N/8), B)` lives in `backend/metal/quantized.cpp:249-254`, which is
**not editable**. Splitting the inner-group count is the only editable grid lever we
have.

**This model made a blind out-of-sample prediction and it landed.** E74 priced its
rung-3 lever at +1.17 % of the verify-width tax, which is +0.232 % of the ranked
candidate leg. The arm-2 receipt measured −0.298 %. Same sign, within 25 %, from a
student whose base predates the receipt and who had never seen it. His recommended
cells at M=5 and M=6 are the crown's cells.

### 3. The prize behind everything: a 19.8 % latency tax

Reconstructing round time from the ranked receipts against an empirical depth-0
floor, `floor(M) = 30.402 + (M−1)*8.42/8` ms, gives a verify-width tax of
**12,645 ms out of 63,766 ms = 19.8 %** of ranked candidate decode time. Recovering
10 % of it scores 3.336; recovering 33 % scores 3.600. Neither the bandwidth roof nor
the compute roof binds at the floor, so the tax is latency and grid occupancy, which
means it is addressable. The MLP families carry 65.9 % of it and five linear families
carry 77.4 %.

The same reconstruction prices depth honestly. The marginal ranked round cost per
width unit rises from about 3.41 ms shallow to about 5.80 ms deep. Normalised by the
30.781 ms depth-0 round that is `h` rising from 0.111 to 0.188, so **the true ranked
price shape is increasing** — which is `pbfit`'s shape, minus its spurious index-4
spike. The shape was right. The spike and the fixture were wrong.

## The price level is already bracketed on ranked. Do not reopen it.

`Qwen36MTPBlockSession.swift:920-935` carries an inherited comment recording official
ranked runs at four price levels:

| `headStepCostRatio` | official score |
|---:|---:|
| 0.14 | 2.766 |
| 0.15 | 2.667 |
| 0.18 (shipped) | ~2.93 era baseline |
| 0.32 | 2.84585 |

The shipped 0.18 is the best of four measured levels, and the 0.14 > 0.15
non-monotonicity is unexplained and flagged. The streak gate is bracketed the same
way: gate 1 scores −7.1 %, gate 0 ties, gate 2 ships. **The way to buy depth is not
the price. It is the cap.** I drafted an arm that lowered `headStepCostRatio` to push
beagle and medicine deeper and did not send it. It is dead on arrival.

The related open defect: `widthCap = fullAcceptStreak >= 2 ? 8 : 5` cannot express 6,
which is the width the ranked histogram spends the most time at.

## Live experiments

| PR | student | question |
|---|---|---|
| #78 | thorfinn | Calibrate the local-to-ranked transfer function on the one eight-line diff that has both harnesses. Per-width cells are now the headline; whole-leg time is a cross-check |
| #81 | askeladd | Does a width-dependent inner-group count beat both uniform tables? Four arms: ship, crown, hybrid at n = 24928, hybrid at n = 8192 |
| #82 | alphonse | Price the proposal head: per-position acceptance census, head-step decomposition, and a go/no-go on re-deriving the weights |
| #83 | edward | Build the per-kernel GPU-time census and name the unattributed 22.6 % of the verify-width tax |

E76 on PR #79 is merged. It closed the register axis with a bound and found the
campaign's second silent Metal miscompilation: `mc*` chunk arms compile with no
diagnostic and produce wrong device output on every scored shape. **A clean compile
and a register census are cost instruments, never correctness gates.** We can only
run device parity on g16s and the ranked host is g17s, so prefer dispatch-table
changes, which E66 proved bit-identical, over kernel-body changes.

## The lead unowned target

**E71 attributed 77.4 % of the M=6 verify-width tax to five linear families and left
22.6 % unattributed.** The tax is 19.8 % of the ranked candidate leg, so the
unattributed remainder is roughly **4 to 4.5 % of the ranked leg, width-dependent,
and unnamed**. That is above the submission bar on its own, and no instrument we own
can see it. E80 builds that instrument.

**Dispatch count is not time.** The copy family is 17.0 % of dispatches and about
0.02 % of GPU time — an overstatement of roughly a thousand times. Every remaining
non-QMV priority currently rests on that discredited proxy, which is why E80 is
scoped as measurement only.

## Potential next research directions

**Re-derive the proposal-head weights.** The single largest historical lever on the
board, +38.1 % cumulative, and the one area our campaign has never touched. Head cost
is paid on every round of every prompt, so it is a uniform lever and collects its
full size at the median. The delivery contract is now fully mapped: one
`model.safetensors` under 2 GiB at an immutable Hugging Face revision, bare tensor
names, a tree digest over sorted file hashes, and a broken declaration refuses rather
than falls back. Teacher compute, not memory, is the binding constraint on a student
Mac, so the correct architecture is to cache `(hidden state, target argmax)` pairs
from one teacher pass and then train the head with the 27B unloaded. E79 rung 3
decides go or no-go. The literature's best fit for our regime — greedy, batch 1,
linear chain, exact top-1 — is a FastMTP-style recursive fine-tune with decayed
position weights, and the cheap go/no-go is that 4,000 samples already moved
acceptance by +25 % in a published ablation.

**Low-rank factorisation of the draft readout.** FR-Spec and VocabTrim cut the output
projection by shrinking the vocabulary. Under our exact top-1 rule a token we cannot
propose is a token we can never accept, so truncation is a hard ceiling and the
durable version is **low-rank factorisation**, not truncation. The readout is about
40 % of head traffic, so this is worth roughly 1.6 % of an M=6 round, uniformly.

**The depth price, third order.** The shape is right and the level is bracketed, so
what remains is the fixture. Refit the vector against the **ranked** width histogram
rather than the local one, and against the in-situ curve rather than the isolated
one, which differ by a real 14 % in shape. Any refit must be scored against the
median of the eight prompts, not against pooled candidate time. The index-4 spike
must be removed on the evidence that it is a local spill artifact.

**Entropy-gated early stopping.** AdaEDL, arXiv:2410.18351, is training-free,
host-independent, and reports 10 to 57 % gains. It is a policy change in the
scheduler. Caution after arm 3: any policy change redistributes width across prompts
and must be priced at the median.

**CLOSED: the whole `copy` family.** The file is
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift`, not
`Sources/MLXFastModel/`. The 2,048 `g2_copybfloat16` dispatches are **not** the
concat: `concatenate_gpu` issues `CopyType::GeneralGeneral`, which `copy.cpp:79-80`
names `gg`, so the concat can only emit `gg*`; the `g2_` copies are the SDPA
dispatcher's query contiguity copy at `scaled_dot_product_attention.cpp:717-718`.
`split = 5` is **derived, not tuned** — `use_fallback` needs
`qL * gqa_factor <= 32` with `gqa_factor = 6`, and head dimension 256 makes the full
path unavailable at every width, so `split = floor(32/6) = 5` with no headroom, in a
dispatcher that is not editable. And the chunk is a **discount, not a cost**: E57
arm C with the chunk off failed `rejected_tail_diverged` at 12.5× `referenceMargin`.
The traffic ceiling settles the rest: the split path moves about 203 MB per leg, so
concat plus query copies are about **0.016 % of a leg, four times below the null
floor.** Not a student slot. KV growth is smaller still, about 0.02 %, and is a
one-line free rider at `step = 1536` or `2048`.

**Other dispatch mass, under suspicion rather than under consideration.**
`unary/binary/ternary_ops` 14.8 %, `rms_norm.metal` 8.1 % (AOT-only, needs
`tools/build-mlx-metallib.sh`), `sdpa_vector.h` 3.45 % (AOT-only, two-pass boundary
at key_len 1024 inside the scored window), `gemv.metal` 3.0 %. Every one of these
shares is a **dispatch count**. Do not assign from this list until E80 replaces it
with GPU time. `gemv.metal` retains one independent virtue: it is 100 % draft-head
traffic and 0 % serial leg, so a change there cannot break token exactness.

**The untimed warmup.** `warmMTPDecode()` runs before the clock starts. The round-1
head prime costs 29.5 ms and is hoistable. Anything moved before the clock is free.

**An open question nobody has closed.** Does the ranked g17s host dispatch a `_nax`
variant of the cross-row QMV? The gate is at `backend/metal/quantized.cpp:697`. E71
verified that kernel selection is identical on g16s and g17s at every width, but it
did not re-verify that for the `_nax` family. Every cross-host kernel argument we
make assumes the answer is no.
