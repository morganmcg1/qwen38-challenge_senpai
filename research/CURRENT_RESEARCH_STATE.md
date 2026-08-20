# SENPAI Research State

- 2026-08-20 12:05 UTC
- Most recent research direction from the human researcher team: Issue #22 —
  execute aggressively toward the winning frontier. No new human direction since.

## Where the campaign stands

Crown: **3.25238228**, submission `9ad17378`, solver Lieisyourlie, source
`bfab0de58d43453e506523707e1720a3485570f4`. `upstream/main` is that exact source,
so any question about what the leader does is a local diff, not a guess.

Our best official score: **3.23588901**, submission `9b241879` (arm 2). Deficit to
the crown: **0.508 %**.

Campaign base: `e93d2127c5faf8f0ef1e8f3a2a9584ddaf2ff268`.

**The board is a noise band, not a ranking.** 579 scored submissions: 58 sit at or
above 3.24, 78 at or above our arm 2, the top 20 span 0.235 % and the top 10 span
0.164 %. A single ranked run has a standard deviation of 0.756 %. Fifty-eight
submissions are packed inside a third of one run's standard deviation. Our rank of
79th is not a capability gap; it is two thirds of one run's noise. The corollary is
that the printed crown is the maximum of many draws in that band and is above its own
true mean, so a run that lands just under it is not evidence that we are slower.

**The bar for spending the ranked slot is therefore about +1.5 % of true ranked
gain.** Below +0.5 % a submission cannot be distinguished from a lucky run.

The ranked slot allows exactly one in-flight submission at about 2.5 hours each. It
is free and I am holding it for arm 3.

## The current research focus

### 1. Ship the depth-price vector, and take the crown

E68 measured the true marginal cost of each verify width and refitted the drafting
scheduler to it. Result: **−3.500 % candidate MTP seconds per token** against a
0.143 % null, over a nine-leg gated palindrome at 512 tokens, byte-identical output
on every leg. It is the largest single-mechanism gain of the campaign.

The merge shipped the instrument but not the winner: `measuredRawDepthPrice` was
empty and `depthPriceArm` was `.ship`. **E75 rung A closed on 2026-08-20 at 11:38 UTC
and the vector is banked** at commit `4d467ca`, with a gated 512-token exactness leg
that matched all tokens, closed the row ledger at 550 rows, passed parity, and
emitted a 513-token digest byte-identical to all fifteen E68 legs. The round
histogram is bit-for-bit the E68 histogram, so the banked constant selects exactly
the schedule that was timed. Arm 3 now waits only on that commit reaching the remote.

Projected ranked median with the vector on our own kernel table, using the measured
ranked round counts and the 1.126 width-curve flattening factor: **3.2824 to 3.2875,
which is +0.92 % to +1.08 % over the crown**, or +1.60 % to +1.70 % over our arm 2.
At a single-run sd of 0.756 %, one run beats the printed crown with probability about
**0.92 to 0.94**.

Two things rung A found that were not asked for and that changed the record.

**The vector published in our own experiment report was never executed by anything.**
Swift rescales as `raw * (total / sum)`; the Python that produced the report computed
`raw * total / sum`. Those differ by one ulp at three of eight positions. The timed
legs ran Swift, so the report was wrong and the banked vector is the executed one. A
committed test now pins all eight doubles bit for bit. The general rule: any constant
crossing from an analysis script into a scored Swift file must be pinned by a
committed test against the value the timed build evaluated, not the value the report
printed.

**My interaction estimate for `frontier table + pbfit` was wrong and the student's is
better founded.** I priced the rounds the schedule *moves*; he priced where the round
mass *sits*, which is the correct weighting when the cost table underneath changes.
`pbfit` parks 42 of its 85 rounds at width 5, the one cell the frontier's table
charges +26.746 ms more for. His prediction is a **sign flip**, +0.77 % rather than
my −2.4 %, with a +4.29 point interaction. His two-layer model validated out of
sample at width 1 to −0.3 %. The two mechanisms therefore look close to mutually
exclusive rather than merely sub-additive, which makes declining the frontier's
0.298 % this round cost nothing.

### 2. The kernel table inverts between our hosts and the ranked host, and we now know why

The arm-2 receipt reverted our cross-row QMV dispatch table to the frontier's table
and changed nothing else. All 8 prompts were faster. Correcting by plutarch, the
speculation-free control, the frontier's table is **0.298 % faster on the scoring
prompts** — while our table is about 1.6 % faster locally.

The mechanism is registers. Edward's `metal-tt -arch applegpu_g17s` oracle reads the
ranked host's real allocation on our development hardware. Our one-group cells need
**98 and 111 registers on g17s against 90** for the frontier's cells. Our hosts have
a 96-register ceiling and clamp all six cells into 93 to 96, so the effect that
decides the ranked score is structurally invisible where we develop.

Alphonse independently proved that `argmin` over IPG is invariant to any pure
rescaling of the per-byte rate, which kills the bandwidth-headroom story outright,
and measured a cost surface in which live state is paid through occupancy rather than
through arithmetic.

**This is now the campaign's most important standing constraint: a local kernel
measurement is not transferable until the cell's g17s register count has been read.**

A full 19-cell census of the legal `(M, IPG)` grid on both architectures gives a
clean law: **the register count is a pure function of the largest group in the
partition.** It does not depend on M and it does not depend on the group count. 19 of
19 cells, no exceptions, no spill anywhere on g17s.

| largest group | g16s | g17s |
|---:|---:|---:|
| 2 | 70 | 83 |
| 3 | 93 | 90 |
| 4 | 94 | **91** |
| 5 | 95 | 98 |
| 6 | 96 + 16 B spill | 111 |

The cliff is between 4 and 5, not between 3 and 4. Going from largest group 3 to 4
costs **one** register on the ranked host, and 90 and 91 sit in the same occupancy
tier. That is why edward's search target is 91 rather than 90, and why a one-group
cell at 91 would strictly dominate the frontier's two-group cell at 90: half the
weight traffic for one register. Dominance analysis also closes M = 7 and M = 8,
which are already at their unique undominated two-group choice in both tables. Only
M = 4, 5, 6 and 9 are open.

**But the register law alone cannot explain the receipt, and that is a proof rather
than a doubt.** At M = 6 the frontier runs two weight streams at 90 registers and we
run one at 111. Under `time proportional to streams / floor(R / regs)`, the frontier
wins only if `occ(90) > 2 * occ(111)`. Maximised over every possible register budget,
that ratio is exactly 2.0000. Doubling the streams can at best be exactly cancelled,
never beaten. **A term is missing from the model.**

The leading candidate is **cache reuse making group count nearly free**. Two
co-scheduled threadgroups in a two-group partition read the same 22.5 KiB output-row
weight tile, so the second read hits L2 or SLC rather than DRAM. If that holds, the
ranked-optimal table is simply "minimise the largest group", which is a *different*
table from the frontier's and a stronger claim. The counter-evidence that must be
weighed is E73's best local fit, which used a linear `groups * W` traffic term and
reached 4.94 % rms with three parameters. This question has no owner yet and it is
the most valuable open kernel question in the campaign.

### 3. The prize is a 19.8 % latency tax, not a bandwidth wall

Reconstructing round time from the ranked receipts against an empirical depth-0
floor, `floor(M) = 30.402 + (M−1)*8.42/8` ms, gives a verify-width tax of
**12,645 ms out of 63,766 ms = 19.8 %** of ranked candidate decode time. Recovering
10 % of it scores 3.336; recovering 33 % scores 3.600.

Neither the bandwidth roof nor the compute roof binds at the floor. The tax is
latency and occupancy, which means it is addressable.

Cost is ordered by working threadgroups, `ceil(M/IPG) * ceil(n/8)`. The local knee is
near 1,900 working threadgroups, about 95 per core on 20 cores. Two independent
instruments agree at Kendall tau = −1.0.

## Live experiments

| PR | student | question |
|---|---|---|
| #78 | thorfinn | Bank the depth-price vector, then measure whether it survives the frontier's kernel table, as a full two-by-two with the interaction term |
| #79 | edward | Is there a bit-identical `_wide` variant that compiles to 90 or fewer registers on `applegpu_g17s`? |
| #80 | alphonse | Deliberately create register variation on a fixed cell and measure the occupancy coefficient directly, then predict the ranked-optimal table |
| #77 | askeladd | Locate the working-threadgroup knee in situ, with zero source edits |

## Potential next research directions

**Is group count nearly free because the groups share a weight tile?** The highest
value open question and the only one with a proof behind it: occupancy alone
mathematically cannot produce the arm-2 receipt, so a term is missing, and cache
reuse across co-scheduled groups is the leading candidate. If it holds, the
ranked-optimal table is "minimise the largest group", which is neither our table nor
the frontier's. Needs an owner.

**Calibrate the local-to-ranked transfer function on the one diff that has both
sides.** E75 rung D measures the frontier's dispatch table locally. The identical
eight-line diff already has a ranked number from receipt `9b241879`, −0.298 %
plutarch-corrected. If the local prediction of +6.44 % holds, one diff moves 6.7
points between harnesses. No other arm in the campaign has both sides, and every
future kernel transfer argument rests on this coefficient.

**Ranked-aware kernel table.** The immediate follow-on from #79 and #80. If a
one-group cell can be made to fit in 91 registers on g17s, we keep the local win and
the ranked win at once. If not, the correct table is the frontier's at M = 5, 6 and 9
and ours nowhere, and we should take the 0.298 % as a separate clean arm.

**Per-shape IPG.** The dispatcher already gates the wide tier on `out_vec_size`, so a
two-dimensional table over (M, output width band) is available, bit-identical, and
untried. E66 proved group partitions are unordered and the change is exact, 12 of 12.
`mlp.down` at n = 5120 is the single shape that loses under every uniform choice.

**The depth price, second order.** E68's own follow-ups: fit the vector to the
in-situ curve rather than the isolated one, which differ by a real 14 % in shape;
bisect between `pbfit` and `ship`, because the optimum is between them; raise the
level now that the shape is right. Each is a five-leg experiment.

**The 45.5 % of dispatches never examined.** `copy.metal` is 17.0 % of in-round
dispatches with zero experiments against it. `unary/binary/ternary_ops` is 13.4 %,
`rms_norm.metal` 8.1 %, `gemv.metal` 3.0 %. `gemv.metal` is 100 % draft-head traffic
and 0 % serial leg, so a change there cannot break token exactness — the cheapest
risk-adjusted surface we have.

**The untimed warmup.** `warmMTPDecode()` runs before the clock starts. The round-1
head prime costs 29.5 ms and is hoistable. Anything that can be moved before the
clock is free.

**Entropy-gated early stopping.** AdaEDL, arXiv:2410.18351, is training-free,
host-independent, and reports 10 to 57 % gains. It is a policy change in the
scheduler, the same surface that just produced our largest win.

**A defect worth fixing.** `widthCap = fullAcceptStreak >= 2 ? 8 : 5` cannot express
6, which is exactly the width the measured price says the scheduler should be
choosing most often.
