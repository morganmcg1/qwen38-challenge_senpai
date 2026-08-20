# SENPAI Research State

- 2026-08-20 11:15 UTC
- Most recent research direction from the human researcher team: Issue #22 —
  execute aggressively toward the winning frontier. No new human direction since.

## Where the campaign stands

Crown: **3.25238228**, submission `9ad17378`, solver Lieisyourlie, source
`bfab0de58d43453e506523707e1720a3485570f4`.

Our best official score: **3.23588901**, submission `9b241879` (arm 2). Deficit to
the crown: **0.508 %**.

Campaign base: `41ddc183817979be8d2f0817d79f98b2ddefb984`.

The ranked slot allows exactly one in-flight submission at about 2.5 hours each. It
is free and I am holding it for arm 3.

## The current research focus

### 1. Ship the depth-price vector, and take the crown

E68 measured the true marginal cost of each verify width and refitted the drafting
scheduler to it. Result: **−3.500 % candidate MTP seconds per token** against a
0.143 % null, over a nine-leg gated palindrome at 512 tokens, byte-identical output
on every leg. It is the largest single-mechanism gain of the campaign.

The merge shipped the instrument but not the winner: `measuredRawDepthPrice` is empty
and `depthPriceArm` is `.ship`. E75 rung A on PR #78 is banking the vector now. It is
the campaign critical path.

Projected ranked median with the vector on our own kernel table, using the measured
ranked round counts and the 1.126 width-curve flattening factor: **3.2824 to 3.2875,
which is +0.92 % to +1.08 % over the crown.** With a single-run sd of 0.756 %, one
run wins with probability about 0.89 to 0.92.

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

**Ranked-aware kernel table.** The immediate follow-on from #79 and #80. If a
one-group cell can be made to fit in 90 registers on g17s, we keep the local win and
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
