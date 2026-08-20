# SENPAI Research State

- 2026-08-20 13:10 UTC
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

Campaign base: `5f12720dc0187f0720ca6e35d89fe5e5b27b21ea`.

**The board is a noise band, not a ranking, and the organizer's own history proves
it.** 579 scored submissions: 58 sit at or above 3.24, 78 at or above our arm 2, the
top 20 span 0.235 % and the top 10 span 0.164 %. A single ranked run has a standard
deviation of 0.756 %. Promotion `5068eb8` has a **tree hash identical to its parent**
and was promoted for scoring 0.008 % higher; promotion `80021bc` changed one note
string inside `mtp-head.manifest.json` and was promoted for 0.062 %. Two of the last
six promotions therefore changed nothing functional. Our rank of 79th is two thirds
of one run's noise, not a capability gap.

**The bar for spending the ranked slot is about +1.5 % of true ranked gain.** Below
+0.5 % a submission cannot be distinguished from a lucky run. One in-flight
submission is allowed, at about 2.5 hours each.

## The single largest change in how we work

**`upstream/main` is a linear chain of 76 commits, and every `Accept submission
<uuid>` commit is one promoted competition submission carrying its exact diff.** The
complete history of every rival, at diff granularity, is a local `git show`. I had
been reasoning about competitors from a list of scores for many rounds. Diff before
theorising about any competitor.

Mapping all 56 promotions against the live Yukon list gives four campaign-changing
facts, recorded in full at ledger 216.

**1. The field is capped at `NA <= 4`.** `quantized.h:980` at `upstream/main` reads
`static_assert(NA >= 2 && NA <= 4)`. Set at `1033e1a`, never raised. No competitor
can select `<T,5,5>` or `<T,6,6>`. We are the only participant that has ever run
NA=5 or NA=6 on the ranked host, so the arm-2 receipt is information nobody else
holds.

**2. The crown's dispatch table is a constrained optimum.** `[3]` `[4]` `[3,2]`
`[3,3]` `[4,3]` `[4,4]` `[3,3,3]` is exactly "minimise the group count subject to
NA <= 4, then tie-break to fewer registers", verified cell by cell against the
legality rule `M % IPG != 1`. The field's revealed preference is fewer groups,
whenever the cap allows. Their M=5, M=6 and M=9 cells are the cells they were forced
into. We are the only participant able to test the cells they wanted.

**3. 140 of the 154 editable files, 91 %, have never been changed by any promotion.**
Of roughly 90 editable Metal and kernel files the field has touched exactly two:
`quantized.h` and `mlx-generated/quantized.cpp`.

**4. The proposal head is the field's biggest lever by a wide margin.** Cumulative
gain across the 13 largest promotions: proposal head **+38.1 %**, schedule +29.0 %,
runtime +21.7 %, kernel +16.1 %. One manifest-only head swap (`deb63ad`) was worth
+7.5 % on its own. All five heads the field has used are requantizations of the
organizer's pinned bf16 weights. **Nobody has re-derived the weights.**

## The current research focus

### 1. Ship the depth-price vector, and take the crown

E68 measured the true marginal cost of each verify width and refitted the drafting
scheduler to it. Result: **−3.500 % candidate MTP seconds per token** against a
0.143 % null, over a nine-leg gated palindrome at 512 tokens, byte-identical output
on every leg. It is the largest single-mechanism gain of the campaign. E75 rung A
banked it at `4d467ca` with a gated 512-token exactness leg that matched all tokens,
closed the row ledger at 550 rows, passed parity, and emitted a digest byte-identical
to all fifteen E68 legs. Arm 3 carries it and is now validating.

Two things rung A found that were not asked for and that changed the record.

**The vector published in our own experiment report was never executed by anything.**
Swift rescales as `raw * (total / sum)`; the Python behind the report computed
`raw * total / sum`. Those differ by one ulp at three of eight positions. A committed
test now pins all eight doubles bit for bit. General rule: any constant crossing from
an analysis script into a scored Swift file must be pinned by a committed test
against the value the timed build evaluated.

**My interaction estimate for `frontier table + pbfit` was wrong and the student's is
better founded.** I priced the rounds the schedule *moves*; he priced where the round
mass *sits*, which is correct when the cost table underneath changes. `pbfit` parks
42 of its 85 rounds at width 5, the one cell the frontier's table charges +26.746 ms
more for. His prediction is a sign flip, +0.77 % rather than my −2.4 %. The two
mechanisms look close to mutually exclusive, so declining the frontier's 0.298 % this
round costs nothing.

### 2. The register law, the missing term, and the graded prize

The arm-2 receipt reverted our cross-row QMV dispatch table to the frontier's and
changed nothing else. All 8 prompts were faster. Corrected by plutarch, the
speculation-free control, the frontier's table is **0.298 % faster on the scoring
prompts** — while our table is about 1.6 % faster locally.

A full 19-cell census on both architectures, read by `metal-tt` on our own hardware,
gives a clean law: **the register count is a pure function of the largest group in
the partition**, independent of M and of group count, 19 of 19 cells, no spill on
g17s.

| largest group | g16s | g17s | g17s occupancy |
|---:|---:|---:|---:|
| 2 | 70 | 83 | 20 |
| 3 | 93 | 90 | 18 |
| 4 | 94 | **91** | 18 |
| 5 | 95 | 98 | 16 |
| 6 | 96 + 16 B spill | 111 | 14 |

Our hosts have a 96-register ceiling and clamp all six cells into 93 to 96, so the
effect that decides the ranked score is structurally invisible where we develop.
**A local kernel measurement is not transferable until the cell's g17s register count
has been read.**

**The missing term is now identified, and my earlier "minimise the largest group"
reading is falsified.** Two solvers flipped the M=4 cell in opposite directions and
both were promoted: `1abe636` went `[4]` at 91 registers to `[2,2]` at 83, and
`d1530a4` reverted it. Both deltas sit inside the single-run sd, so M=4 `[2,2]` at 83
is **tied** with `[4]` at 91 on the ranked host. Fewer registers is not automatically
better.

That tie fixes one parameter. With `occ(r) = floor(208 KiB / (128 r))`,

```
cell time  ~  (1 + eps (groups - 1)) / occ,   eps = 0.111
```

**A second weight-stream group costs 11 %, not 100 %.** That is the first
quantitative support for the cache-reuse hypothesis: two groups share a 22.5 KiB
output-row tile at K=5120, so the second read is nearly free in traffic and pays only
in scheduling. It also resolves why occupancy alone could not produce the receipt —
`occ(90)/occ(111)` maximised over all budgets is exactly 2.0000, which can cancel a
doubling but not beat it. The extra stream never was a doubling.

With `eps` fixed by M=4, the model predicts the arm-2 receipt with no further
fitting, and a single scale factor `kappa = 0.298 / 4.97 = 0.0600` converts modelled
cell cost into leg time.

**The prize is graded.** g17s tiers: 92 registers or fewer gives occupancy 18, 93 to
97 gives 17, 98 to 104 gives 16. Against the crown's cells:

| our registers | M=5 `[5]` | M=6 `[6]` | M=9 `[5,4]` | leg value |
|---|---|---|---|---|
| today, 98 and 111 | +1.3 % | +15.7 % | +2.3 % | +0.30 % (measured) |
| 93 to 97 | −4.7 % | −4.7 % | −3.7 % | about −0.18 % |
| 92 or fewer | −10.0 % | −10.0 % | −9.1 % | about **−0.38 %** |

The field's last six promotions total +0.28 % between them. NA=6 is worth more than
NA=5, because M=6 carries 33.4 % of ranked width time and sits two occupancy tiers
above the crown. `[6]` at 98 registers is still worse than `[3,3]`, so 98 is not a
partial win at M=6.

### 3. The prize behind everything: a 19.8 % latency tax

Reconstructing round time from the ranked receipts against an empirical depth-0
floor, `floor(M) = 30.402 + (M−1)*8.42/8` ms, gives a verify-width tax of
**12,645 ms out of 63,766 ms = 19.8 %** of ranked candidate decode time. Recovering
10 % of it scores 3.336; recovering 33 % scores 3.600. Neither the bandwidth roof nor
the compute roof binds at the floor, so the tax is latency and occupancy, which means
it is addressable. Cost is ordered by working threadgroups, `ceil(M/IPG)*ceil(n/8)`;
the local knee is near 1,900 working threadgroups, about 95 per core, with two
independent instruments at Kendall tau = −1.0.

## Live experiments

| PR | student | question |
|---|---|---|
| #77 | askeladd | Locate the working-threadgroup knee in situ, with zero source edits |
| #78 | thorfinn | Calibrate the local-to-ranked transfer function on the one eight-line diff that has both harnesses |
| #79 | edward | Can a bit-identical `_wide<T,5>` or `_wide<T,6>` body reach 92 registers or fewer on `applegpu_g17s`? |
| #80 | alphonse | Measure the occupancy coefficient directly on a fixed cell, and predict the ranked-optimal table |

## Potential next research directions

**Re-derive the proposal-head weights.** The single largest historical lever on the
board, +38.1 % cumulative, and the one area our campaign has never touched. Every
head in the field's history is a requantization of the organizer's pinned bf16
weights; nobody has distilled or finetuned a head against the fixed target for the
committed-history regime, where acceptance at deep positions is what pays. The
current head's design constants — a 32-row shortlist, a 2-bit first stage, a
1024-row island budget, SSE selection — were each measured once and never swept. Long
lead time, largest expected value, needs an owner and a plan.

**The `copy` family and KV growth.** The largest untouched scored surface in the
competition: 17.0 % of in-round dispatches, 10,235 of the 13,554 inside
`target_verify`, driven by `KVCacheSimple.update` growing its buffer with `zeros()`
plus `concatenated()` every round. Zero promotions and zero campaign experiments.
The fix is available on the Swift side alone, by over-allocating KV growth so the
concat copies disappear. `gather_front.h` at 2.15 % bundles with it. **Highest-value
unowned experiment; assign as soon as a slot frees.** Caveat: dispatch count is not
time, and no per-family GPU-time census exists yet.

**Calibrate the local-to-ranked transfer function.** E75 rung D measures the
frontier's dispatch table locally. The identical eight-line diff already has a ranked
number, −0.298 % plutarch-corrected. If the local prediction of +6.44 % holds, one
diff moves 6.7 points between harnesses. No other arm has both sides, and every
future kernel transfer argument rests on this coefficient.

**Ranked-aware kernel table.** The immediate follow-on from #79 and #80. If a
one-group cell fits in 92 registers on g17s we keep the local win and the ranked win
at once, worth about −0.38 % against the crown. If not, the correct table is the
frontier's at M=5, 6 and 9 and ours nowhere, and we take the 0.298 % as a separate
clean arm.

**Per-shape IPG.** The dispatcher already gates the wide tier on `out_vec_size`, so a
two-dimensional table over (M, output width band) is available, bit-identical, and
untried. `mlp.down` at n = 5120 is the single shape that loses under every uniform
choice.

**The depth price, second order.** E68's own follow-ups: fit the vector to the
in-situ curve rather than the isolated one, which differ by a real 14 % in shape;
bisect between `pbfit` and `ship`, because the optimum is between them; raise the
level now that the shape is right. Each is a five-leg experiment.

**Other never-examined dispatch mass.** `unary/binary/ternary_ops` 13.4 %,
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
