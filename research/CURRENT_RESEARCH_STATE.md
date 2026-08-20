# SENPAI Research State

- 2026-08-20 13:20 UTC
- Most recent research direction from the human researcher team: Issue #22 —
  execute aggressively toward the winning frontier. No new human direction since.

## Where the campaign stands

Crown: **3.25238228**, submission `9ad17378`, solver Lieisyourlie, source
`bfab0de58d43453e506523707e1720a3485570f4`. `upstream/main` is that exact source.

Our best official score: **3.23588901**, submission `9b241879` (arm 2). Deficit to
the crown: **0.508 %**.

**Arm 3 is in flight.** Submission `2da69933-5202-4e0d-b336-c75945a45b9e`, created
2026-08-20T12:00:44Z, status `validating`, candidate commit `389676fb`, scored
surface byte-identical to thorfinn's validated `4d467ca`. One variable: the measured
depth-price vector. Projected median 3.2824 to 3.2875.

Campaign base: `8d938c911df52b6a324f259a55dbaa75e508c822`.

**The board is a noise band, not a ranking, and the organizer's own history proves
it.** 587 scored submissions of 854 total: 58 sit at or above 3.24, 78 at or above
our arm 2, the top 20 span 0.235 % and the top 10 span 0.164 %. A single ranked run
has a standard deviation of 0.756 %. Promotion `5068eb8` has a **tree hash identical
to its parent** and was promoted for scoring 0.008 % higher; promotion `80021bc`
changed one note string inside `mtp-head.manifest.json` and was promoted for
0.062 %. Two of the last six promotions therefore changed nothing functional. Our
rank of 79th is two thirds of one run's noise, not a capability gap.

**The bar for spending the ranked slot is about +1.5 % of true ranked gain.** Below
+0.5 % a submission cannot be distinguished from a lucky run. One in-flight
submission is allowed, at about 2.5 hours each.

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

## The current research focus

### 1. Ship the depth-price vector, and take the crown

E68 measured the true marginal cost of each verify width and refitted the drafting
scheduler to it. Result: **−3.500 % candidate MTP seconds per token** against a
0.143 % null, over a nine-leg gated palindrome at 512 tokens, byte-identical output
on every leg. It is the largest single-mechanism gain of the campaign. E75 rung A
banked it at `4d467ca` with a gated 512-token exactness leg that matched all tokens,
closed the row ledger at 550 rows, passed parity, and emitted a digest byte-identical
to all fifteen E68 legs. Arm 3 carries it and is validating now.

Two things rung A found that were not asked for and that changed the record.

**The vector published in our own experiment report was never executed by anything.**
Swift rescales as `raw * (total / sum)`; the Python behind the report computed
`raw * total / sum`. Those differ by one ulp at three of eight positions. A committed
test now pins all eight doubles bit for bit. General rule: any constant crossing from
an analysis script into a scored Swift file must be pinned by a committed test
against the value the timed build evaluated.

**My interaction estimate for `crown table + pbfit` was wrong and the student's is
better founded.** I priced the rounds the schedule *moves*; he priced where the round
mass *sits*, which is correct when the cost table underneath changes. `pbfit` parks
42 of its 85 rounds at width 5, the one cell the crown's table charges +26.746 ms
more for. His prediction is a sign flip, +0.77 % rather than my −2.4 %. The two
mechanisms look close to mutually exclusive, so declining the crown's 0.298 % this
round costs nothing.

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

**The register law itself survives.** Register count is a pure function of the
largest group in the partition, independent of M and of group count, 19 of 19 cells,
zero exceptions: largest group 2/3/4/5/6 gives 70/93/94/95/96 on g16s and
83/90/91/98/111 on g17s. Local hosts have a 96-register ceiling, so `[6]` spills
16 bytes locally and does not spill on the ranked host. **Any local arm above 96
registers measures spill, not occupancy**, and spill is about 0.015 % per frame byte,
three times steeper than the spill-free ladder.

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
the defect, and it explains the receipt without any occupancy term: the crown's
higher group count helps the three n=5120 families and hurts the wide ones, and on
the ranked host the balance falls the other way from ours.

Core count does not change the log-ratio gain for a shape deep below the knee. What
it changes is **which shapes are starved**: the boundary moves from n = 12464 at 20
cores to n = 24928 at 40 cores, pulling n = 14336 and n = 16480 across. The
dispatcher's own `out_vec_size >= 4096` gate is therefore three times too low
locally and six times too low on the ranked host, and no scored shape sits below it.

**The grid cannot be widened any other way.** The host grid
`grid_dims(M, ceil(N/8), B)` lives in `backend/metal/quantized.cpp:249-254`, which is
**not editable**. Splitting the inner-group count is the only editable grid lever we
have.

**This model made a blind out-of-sample prediction and it landed.** E74 priced its
rung-3 lever at +1.17 % of the verify-width tax, which is +0.232 % of the ranked
candidate leg. The arm-2 receipt measured −0.298 %. Same sign, within 25 %, from a
student whose base predates the receipt and who had never seen it. His recommended
cells at M=5 and M=6 are the crown's cells. That is the first time a local cost model
predicted an unseen ranked receipt, and it is why per-shape IPG is now the lead
kernel hypothesis.

### 3. The prize behind everything: a 19.8 % latency tax

Reconstructing round time from the ranked receipts against an empirical depth-0
floor, `floor(M) = 30.402 + (M−1)*8.42/8` ms, gives a verify-width tax of
**12,645 ms out of 63,766 ms = 19.8 %** of ranked candidate decode time. Recovering
10 % of it scores 3.336; recovering 33 % scores 3.600. Neither the bandwidth roof nor
the compute roof binds at the floor, so the tax is latency and grid occupancy, which
means it is addressable. The MLP families carry 65.9 % of it and five linear families
carry 77.4 %.

## Live experiments

| PR | student | question |
|---|---|---|
| #78 | thorfinn | Calibrate the local-to-ranked transfer function on the one eight-line diff that has both harnesses |
| #79 | edward | Compile-only close-out of the register question, and remove the 16-byte local `<T,6,6>` spill that biases every local M=6 measurement |
| #81 | askeladd | Does a width-dependent inner-group count beat both uniform tables? Four arms: ship, crown, hybrid at n = 24928, hybrid at n = 8192 |
| #82 | alphonse | Price the proposal head: per-position acceptance census, head-step decomposition, and a go/no-go on re-deriving the weights |

## Potential next research directions

**Re-derive the proposal-head weights.** The single largest historical lever on the
board, +38.1 % cumulative, and the one area our campaign has never touched. The
delivery contract is now fully mapped: one `model.safetensors` under 2 GiB at an
immutable Hugging Face revision, bare tensor names, a tree digest over sorted file
hashes, and a broken declaration refuses rather than falls back. Teacher compute, not
memory, is the binding constraint on a student Mac, so the correct architecture is to
cache `(hidden state, target argmax)` pairs from one teacher pass and then train the
head with the 27B unloaded. E79 rung 3 decides go or no-go. The literature's best fit
for our regime — greedy, batch 1, linear chain, exact top-1 — is a FastMTP-style
recursive fine-tune with decayed position weights, and the cheap go/no-go is that
4,000 samples already moved acceptance by +25 % in a published ablation.

**The exactness-chunk concat.** Correcting the `copy`-family analysis moved the
target. KV growth is **not** per-round: `KVCache.swift:388` already sets `step = 256`,
so growth fires three times per candidate leg, 288 of 10,235 copies, worth about
0.02 % of a leg — roughly 40 times below our measurement floor. Do not spend a
student slot on it; fold it in as a one-line free rider at 1536 or 2048. The real
target is the exactness-chunk `concatenated` at `AttentionUtils.swift:141`: **2,048
dispatches per leg, ten times the KV-growth prize, editable, and never touched by any
promotion or any campaign experiment.** It exists to preserve exactness above
width 5, which is exactly where our round mass sits. Caveat that still holds:
dispatch count is not time, and no per-family GPU-time census exists.

**A truncated draft vocabulary is a hard ceiling under exact top-1 verification.**
FR-Spec and VocabTrim cut the output projection by shrinking the vocabulary, and
FastMTP measured 152K→32K costing only 0.068 tokens of accepted length. Under our
exact top-1 rule a token we cannot propose is a token we can never accept, so the
durable version is **low-rank factorisation of the readout**, not truncation. The
readout is about 40 % of head traffic, so this is worth roughly 1.6 % of an M=6
round.

**The depth price, second order.** E68's own follow-ups: fit the vector to the
in-situ curve rather than the isolated one, which differ by a real 14 % in shape;
bisect between `pbfit` and `ship`, because the optimum is between them; raise the
level now that the shape is right. Each is a five-leg experiment.

**Other never-examined dispatch mass.** `unary/binary/ternary_ops` 14.8 %,
`rms_norm.metal` 8.1 % (AOT-only, needs `tools/build-mlx-metallib.sh`),
`sdpa_vector.h` 3.45 % (AOT-only, two-pass boundary at key_len 1024 inside the scored
window), `gemv.metal` 3.0 %. `gemv.metal` is 100 % draft-head traffic and 0 % serial
leg, so a change there cannot break token exactness — the cheapest risk-adjusted
surface we have.

**The untimed warmup.** `warmMTPDecode()` runs before the clock starts. The round-1
head prime costs 29.5 ms and is hoistable. Anything moved before the clock is free.

**Entropy-gated early stopping.** AdaEDL, arXiv:2410.18351, is training-free,
host-independent, and reports 10 to 57 % gains. It is a policy change in the
scheduler, the same surface that produced our largest win.

**A defect worth fixing.** `widthCap = fullAcceptStreak >= 2 ? 8 : 5` cannot express
6, which is exactly the width the measured price says the scheduler should choose
most often.

**An open question nobody has closed.** Does the ranked g17s host dispatch a `_nax`
variant of the cross-row QMV? The gate is at `backend/metal/quantized.cpp:697`. E71
verified that kernel selection is identical on g16s and g17s at every width, but it
did not re-verify that for the `_nax` family. Every cross-host kernel argument we
make assumes the answer is no.
