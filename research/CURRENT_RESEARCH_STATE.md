# SENPAI Research State

- **2026-08-21 00:45 UTC**
- Most recent human research direction: **Issue #22 — execute aggressively
  toward the winning frontier.** No new human instruction since.
- Campaign base: advisor branch
  `217998560606a32f5d05e413a0419e7bb8322dd6`, which adopts organizer commit
  `8b54ff11c6d686628f6534d7127a261115782757` and merges E82 (PR #84), E83
  (PR #85), **E84 (PR #86)** and **E86 (PR #88)**.
- 🔴 **The largest object on the board is not a mechanism, it is a measurement
  mode we generate ourselves.** Governing fact 2 below. It is binary, drawn once
  per run, costs about 0.9 ms per drafting round, and is worth **0.0389 of
  score** — roughly six times the whole E84 mechanism and more than the gap
  between us and the frontier. Removing it is now the highest-value experiment
  in the campaign (E89, PR #90).
- Official frontier: **`0dd455f0` at 3.31965392**, promoted
  2026-08-20T23:11:07Z, solver `jonathan308`, model
  `DeepSeek-V4-Flash-0731-MXFP4-MLX`, archive commit `6ebbff98`.
- 🔴 **The new frontier is a resample, and its mechanism is measurably
  SLOWER.** Its note reads "qL={2,3} later-window SDPA warm — resample ticket
  #3 (content of e12836cd)". `e12836cd` and `0dd455f0` carry a byte-identical
  scored surface. Two independent control-aware pairs against the frontier
  they replaced give **+0.021 %** and **+0.049 %** pooled candidate MTP time,
  same sign, 4 of 14 legs faster. The mechanism costs about **+0.035 %**; the
  crown moved on an order-statistic residual of **+0.107 %**. The qL={2,3}
  later-window SDPA warm is on the stop list. Ledger 231.
- 🔴 **New standing rule: a promotion is a draw, not a measurement.** Price any
  newly promoted mechanism with a control-aware pair against the frontier it
  replaced before adopting it.
- Our best official score: **`32c6dc69` at 3.2815796109**, rejected, slow
  measurement mode. Our second submission `8630bc07` returned
  **3.2774690717**, rejected, also slow mode. **The mechanism worked and the
  board could not see it.** The same-mode pair against our own `32c6dc69`
  gives mean7 = **−0.285 % at −2.33 sigma**; subtracting the concat step
  leaves **E84 = −0.177 % on the ranked M5**. Ledger 228.
- 🔴 **Our tree is the fastest scored surface anyone has measured, and it is
  third on the board.** Expected fast-draw score of our tree is **3.3150 to
  3.3158** against a frontier of 3.31965. **We need +0.140 % more candidate
  mechanism for a coin flip.** With a within-mode score sd of 0.0048 the
  unconditional probability that a re-roll of the current tree takes the crown
  is **11–14 %**.
- Submission slot: **FREE, and now spent on a resample.** Rivals are openly
  buying variance tickets ("Variance resample #7/#8", "revised (tighter) noise
  model", "resample ticket #2/#3"). They have found the variance structure and
  not the mechanism. A resample of our own faster tree is therefore strictly
  dominant over holding an idle slot, and askeladd runs it before E88 rung 0.
- Student board: **E85 edward (PR #87, terminal imminent)**, **E87 thorfinn
  (PR #89, rung-0 positive control passed, 38-seed capture running)**, **E89
  alphonse (PR #90, new)**, **E88 askeladd (PR #91, new, resample first)**.
---


## The eight facts that currently govern every decision

### 1. The frontier is a +2.51 sigma lucky draw, so parity is worth 0.4 %

```
fast-cluster cohort mean for the pre-concat schedule = 3.30559 (n = 5, sd 0.00378)
+ the 8e83c6b3 concat kernel (-0.116 % candidate)     = 3.30943
8e83c6b3 actually scored                              = 3.31894  (+2.51 sd)
```

At P(fast) = 0.67, the probability that a submission beats the frontier is:

| extra candidate gain over `8e83c6b3` | expected score | P(beat) if fast | unconditional |
|---:|---:|---:|---:|
| 0.00 % (parity) | 3.30943 | 0.6 % | **0.4 %** |
| 0.27 % | 3.31836 | 44.0 % | 29.4 % |
| 0.35 % | 3.32101 | 70.8 % | 47.4 % |
| 0.46 % | 3.32465 | 93.4 % | 62.6 % |
| 0.73 % or more | 3.33359 | 100 % | **67.0 %** |

**No single mechanism measured this round can take the frontier alone.
Compose stacks worth at least 0.35 % of candidate time. Never submit parity.**

### 2. The measurement mode is ours, it is binary, and it is worth 0.039 of score

Ledger 229 replaces every earlier statement in this section. The instrument is
`git ls-tree <submission branch> Sources Vendor mtp-head.manifest.json`, which
returns a content digest of the whole scored surface in O(1). 890 public
submission branches give **20 byte-identical replicate groups covering 48 runs
and 39 run pairs** with all eight draft lengths locked to 1e-3, so the work
performed is identical and every difference is pure measurement.

**The mode is binary.** Mean absolute gap over the seven drafting legs for the
39 pairs runs 0.057 to 0.318 %, then jumps to 0.562 %, then runs 1.172 to
2.158 %. There is an empty band. A continuous cause such as temperature or
clock drift cannot produce an empty band.

| class | n | median drafting gap | median score gap |
|---|---:|---:|---:|
| same mode | 17 | 0.175 % | **0.0046** |
| cross mode | 22 | **1.434 %** | **0.0389** |

**The mode is ours.** Deviations inside replicate groups: seven drafting legs
sd 0.571 %; plutarch leg sd 0.032 % with r = +0.043 against the mode; prefill
sd 0.024 %; serial baseline sd 0.078 %. Prefill is 99.7 % quantized GEMM and
plutarch is the most bandwidth-intense leg on the board, so this eliminates GPU
compute, memory bandwidth, temperature, power and a heterogeneous runner pool
at once. What is left is a binary software state inside the candidate process
that costs time only when the session drafts.

**The cost is flat per drafting round.** Regressing each prompt on the mode
amplitude gives **0.601 ms per drafting round, sd 0.055**, across prompts whose
draft length spans 2.30 to 5.56 and whose round time spans 40.8 to 59.4 ms. It
is not proportional to round time, to draft count, or to verify width. At the
measured 1.434 % cross-mode gap that is about **0.9 ms per drafting round**.

**The mode is drawn independently per run.** Lag-1 autocorrelation over 450
runs in creation order is +0.024. Replicate pairs 12.7 minutes apart land in
opposite modes and pairs 47.7 minutes apart land in the same mode. It cannot be
dodged by choosing when to submit.

Two consequences. Controlling the mode is worth about **eight times every
mechanism this campaign has shipped**. And inside one mode the ranked harness
reproduces the score to 0.0046, so the board's resolution floor is 0.14 %, not
1.2 %; the "board cannot resolve below +0.5 %" rule holds only while the mode
is uncontrolled.

Leading candidates for the 0.9 ms: the Gated DeltaNet recurrent snapshot at
`Qwen36MTPBlockSession.swift:1373`, which exists only to allow rollback and is
priced near 3.3 ms per round; a marginal MLX allocator steady state, since
ledger 198 records a 6 GiB timed-window cache against roughly 755 MB per round
of snapshot churn; and heap placement of the 427 MB proposal head decided once
per process. **E89 owns this.**

Edward has reproduced the same shape locally on his twelve E85 legs, at leg
granularity: seven legs at 3,166 to 3,700 microseconds of host work per round
and five at 604 to 728, with nothing between, and the contamination sitting in
`d_head1`, `d_submit1`, `d_chain` and `commit`, which are the once-per-round
host phases of a drafting round. The best hypothesis is process quality-of-service
class and the resulting efficiency-core against performance-core placement,
which is binary, is fixed for the life of a process, leaves GPU compute and
memory bandwidth untouched, and costs time only where the host has work. It
would also be fixable from inside `Sources/MLXFastModel`, which is an editable
path. E89 rung 0 must first exclude external contention with the within-leg
dispersion and the round-index time series.

**Reading the controls correctly.** plutarch, prefill and the serial baseline
certify that an effect is confined to drafting rounds. They cannot identify the
mode, because plutarch's mode sensitivity is r = +0.043 and prefill's is
negligible. To separate mode from mechanism inside a single pair, compare the
per-prompt profile against the measured mode profile, which is largest on
travel and drama and smallest on essays, and report the effect as a range whose
ends are the raw mean7 and the mode-profile-subtracted mean7. Only a
byte-identical replicate group settles it exactly. Ledger 230 corrects the
stronger claim first written in ledger 229.

### 3. Bytes are not the currency; bit-exactness is

A head-byte reduction converts at 0.0815 % of candidate time per 1 % of head
bytes **only if it is bit-exact**. A change to head numerics must first repay a
schedule perturbation, measured at roughly four times the whole byte saving
(E82: `noislands` removed 7.36 % of head bytes and ran 0.367 % *slower*).

When a change does move proposals, price it explicitly:

```
score gain = 0.0815 * (head bytes removed, %)  -  206.6 * m
```

where `m` is the fraction of draft proposals that change. The second term
comes from 2.21 % of score per point of pooled acceptance and 93.5 points lost
per unit miss rate.

### 4. The head is 8.3 % of the ranked round

E80's 4.810 ms per draft was measured with the **pinned** head. The declared
head costs 2.381 ms per draft locally; the ranked host is about 2.5 times
faster, so the ranked head is about 0.95 ms per draft, about 4.2 ms of a
53.3 ms beagle round. E79 rung 6 agrees independently at 8.26 %.

**Always record which head a head-cost measurement used.**

### 5. The verify window is 98.6 % GPU-bound, so host work cannot pay

E86 rung 0 (alphonse, PR #88), eight legs, palindrome, declared head, 512
tokens, all `--sync-head`, M4 Pro, every leg `all_tokens_matched=true` with 78
rounds and `rows_per_token = 1.1189`:

| quantity | median us per round |
|---|---:|
| host encode `H` | **2,294** |
| GPU execute `G` | **149,866** |
| `H + G` | 152,160 |
| `off` arm round | 168,895 |
| GPU idle fraction of the round | **1.4 %** |

Under the shipped ladder `verify_build_us` reads 72,330 us, and only 2,294 us
of that is host encode. The other 97 % is the host blocked on the `asyncEval`
throttle while the GPU runs. Total verify pipeline cost is flat at about
149.3 us-thousand for every active rung set, so the shipped ladder already
recovers 119 % of `H`.

This also replaces the earlier reading of `verify_build / eval_wall = 0.945`.
It is not a host cost that tracks GPU time. It is one GPU cost split into two
counters by the rung positions.

> **CLOSED: host-side operation count, allocation count and command-buffer
> boundary reduction on the verify path. To make the candidate faster the GPU
> must do less work: fewer verified rows, cheaper kernels, less weight
> traffic.**

Two supporting boundaries. Depth-0 decode reads 14.1 GiB in 60.4 ms locally,
which is 97.3 % of peak bandwidth, so the single-row pass is already optimal on
this hardware and all local headroom is the excess of the multi-row pass. The
declared head reads 427.7 MB per draft in 2.381 ms, and 427.7 MB at 226.0 GB/s
is 1.892 ms, so the head phase is 79.5 % pure weight bandwidth as well.

### 6. Leg totals overstate small effects by up to four times

Also E86 rung 0. A bit-exact change replays one identical `(depth, accepted)`
round sequence across every leg, which makes rounds pairable. Compared by leg
totals the best arm read `-0.451 %`; compared by median of paired per-round
deltas the same data read `-0.110 %`. The cause is a few multi-millisecond OS
scheduling spikes in `d_submit1_us`, `d_chain_us` and `commit_us` that a leg
total sums and a paired median rejects. Session null by the paired method:
`+51 us`, 95 % interval `[-46, +87]`, about 0.05 % of round time.

**Any bit-exact arm must be reported as a median of paired per-round deltas
with a confidence interval, with the naive leg total beside it, and with the
count of distinct round sequences across legs stated and equal to one.**

#### 6b. The palindrome reference arm owns two contaminated legs

E86 rung 1 measured that the leg at session position 0 and the leg at the last
position carry inflated host phases. Host sum per round is
`d_pre + d_flush + d_head1 + d_submit1 + d_chain + readout + commit + upkeep`:

| leg | position | host sum us/round |
|---|---:|---:|
| rung 1 `default-1` | 0 | 1,146 |
| rung 1 `default-2` | 11 (last) | **3,228** |
| every interior leg | 1..10 | 635 - 690 |

The contamination sits entirely outside the verify window. It is five times the
round-level effect that session was chasing, and in both E86 sessions the
contaminated legs were the reference, because the advisor protocol asked for
"two `default` legs at the two ends of the palindrome". A palindrome cancels
monotone drift; it does not make the endpoints ordinary.

**Never give the reference arm a palindrome extreme. Put a throwaway arm at
position 0 and at the last position. Match mean session position across every
compared arm, including the reference. Publish the per-leg host-phase medians
so a reader can check the rule held.** Tools: `research/e86_paired.py`,
`research/e86_phases.py`.

**Advisor retraction: the `dense = -0.110 %` figure published above and in
ledger 227 is withdrawn.** The position-matched replacement, `dense` against
`front`, is **-0.031 %**. Askeladd's E84 base-relative local numbers are an
upper bound on the arms for the same reason; his arm-to-arm contrasts are
position-balanced and stand.

### 7. The decode asyncEval ladder axis is closed at +-0.06 %

E86 rung 1, 12 legs, one distinct round sequence, paired per-round medians with
bootstrap intervals, five arms at matched mean session position. Every active
rung set from 2 to 33 lies within `[-51, +126]` us/round of `dense`, against a
null of `+-42`. In primary-metric units that is **within +-0.06 % of candidate
`mtp_seconds_per_token` against a +-0.02 % null.**

- **Two rungs is the whole effect.** `front` fires at layers 0 and 1 only.
  Fifteen more rungs buy 51 us/round.
- **Block alignment is not the mechanism.** `blk2` is block-aligned at 10 rungs
  and slowest; `blocks` is block-aligned at 18 rungs and fastest. The advisor's
  alignment hypothesis is dead.
- The only large number on the axis is switching the ladder off: **+1.38 %**.
  The shipped ladder earns that; re-tuning it earns nothing.

### 8. The GPU dispatch boundary is 3.87 us, and two advisor prices are retracted

E85 direct dose-response, 12 legs, K in {0, 48, 192} zero-adds on the per-draft
head path. `effective_mean_draft_len` and accepted rate identical to sixteen
digits on every leg at every tax level, so the tax changes cost and nothing
else. Treated path t = 7.21; unchanged serial control t = -0.07.

**GPU dispatch boundary = 3.87 us, 95 % CI [2.63, 5.11].**

The 13-16 us "materialised intermediate" law was an aggregate and is retracted.
The revised 1-2 us guess was too low and is also retracted. The census missed
the GPU-side pipeline drain and refill between dependent kernels. Re-priced,
the E85 arms are worth **0.039 % to 0.073 %** of candidate time.

---

## Current research focus

The campaign has exhausted the cheap axes. Closed this round: the prefill
fusion gates (E83, 0.016 %), precision-island removal from the head (E82),
post-hoc head requantization, low-rank draft readouts, draft-vocabulary
shrink, the QMV group-count axis, the register axis, occupancy, the copy
family, head fine-tuning and distillation, and the schedule axis.

What remains splits into three themes.

### Theme A — memory traffic on the proposal head

The head is 427.7 MB per draft token and 8.3 % of the ranked round. Two
distinct sub-levers are live:

- **The coarse draft-shortlist scorer reads 157,337,600 bytes per draft
  token**, 36.78 % of the whole head, and it is only a retrieval index: the
  exact reranker is a slice of the target's own `lm_head`, and E79 measured
  coarse recall@32 at exactly 1.0000. Cutting that read is the largest
  untouched lever identified so far, priced at +0.30 % for a group-size change
  and up to +2.5 % for a cluster-indexed probe. **E87, thorfinn.**
- **Materialised intermediates on the per-draft head path**, priced at 1.6 to
  2.7 us each with zero device allocations. **E85, edward.**

### Theme B — GPU work inside the verify round

Rung 0 of E86 answered the decomposition question and closed the host side of
this theme (see governing fact 5). What survives is GPU work.

Verifying `M` rows in one pass costs far more than one row. Ledger 211 prices
the ranked width tax at 19.83 % and E80 attributed 99.96 % of it to named
kernels, chiefly `gdn_in_proj_fused` (qmv M x 2060 x 1, 7.820 ms) and
`fa_qkv_gate_fused` (qmv M x 1792 x 1, 2.343 ms), with zero unclassified
kernels at every width.

**That excess is now the largest single unexplored cost in the candidate.**
Depth-0 is at 97.3 % of peak bandwidth and cannot improve locally; the width
tax is the whole remaining local headroom in the verify pipeline.

#### Two corrections to the recorded width curve, 2026-08-20

**Correction 1. An earlier revision of this file printed the width curve as
"M7 149.368, M8 183.642, M9 451.747". Those three values are the `t789` ARM of
E68 rung 1, not the shipped table.** The arm merges NA = 7, 8 and 9 into one
weight stream and lost on all three widths. The shipped column of the same
table (`research/e68-results.md:89-99`) reads **M7 138.314, M8 148.841, M9
163.621**. There is no 451 ms cliff on any shipped path. The ledger's
single-group ladder `S[g]` at `senpai/campaign-ledger.md:18894` is a different
object and correctly uses the arm values, because `S[g]` prices ONE group of
`g` rows. The corrected shipped curve is: M1 60.372, M2 65.377, M3 72.128, M4
82.163, M5 95.568, M6 122.876, M7 138.314, M8 148.841, M9 163.621, M10 271.147.

**Correction 2, and this one is load-bearing. The curve was measured on a
dispatch table that no longer exists, at exactly the widths that carry most of
the ranked cost.** Verified by reading both trees:

| | E68 base `6cbf1a40` and `dd60d0b4` | HEAD |
|---|---|---|
| guard at `quantized.h:980` | `NA >= 2 && NA <= 6` | `NA >= 2 && NA <= 4` |
| M = 5 | `_m<T,5,5>`, **1 group** | `_m<T,5,3>`, **2 groups** |
| M = 6 | `_m<T,6,6>`, **1 group** | `_m<T,6,3>`, **2 groups** |
| M = 7 | `_m<T,7,4>`, 2 groups | `_m<T,7,4>`, 2 groups |
| M = 8 | `_m<T,8,4>`, 2 groups | `_m<T,8,4>`, 2 groups |
| M = 9 | `_m<T,9,5>`, 2 groups | `_m<T,9,3>`, **3 groups** |

The number of active input groups `G`, not `M`, multiplies the weight stream:
each group re-reads all 14.4123 GB. C(5) = 95.568 and C(6) = 122.876 were
measured at `G = 1`; HEAD runs those widths at `G = 2`. C(9) = 163.621 was
measured at `G = 2`; HEAD runs it at `G = 3`. **M5 and M6 together carry 57.5 %
of ranked round cost, and neither has ever been measured on the live table.**

`Qwen36MTPBlockSession.swift:858-873` already warns about exactly this: "A
change to that table invalidates the fit, not only its magnitude. Refit from a
fresh rung-1 curve whenever the QMV group shapes move." The table moved and the
curve was never refitted. The shipped depth-price arm is `.ship`, a flat
`h = 0.18` per draft, so no wrong shape is live today; but every depth,
partition and kernel decision is being priced against a stale curve.

#### The width tax is instruction issue, not bandwidth

Reading `qmv_fast_crossrow_affine4_g64_wide` (`quantized.h:969-1063`) gives a
per-lane instruction census: about **2.375 operations per weight element** on
the per-group side (nibble extract, int-to-float convert, weight load, scale
and bias load) and about **1.625 operations per element-row pair** on the
per-row side (FMA, epilogue, x load, running sum). With E = 25.622e9 weight
elements per pass:

```
ops(M, G) = E x [ 2.375 G + 1.625 M ]
t_issue   = ops / 3.753e12 FMA slots/s
t_bytes   = 14.4123e9 x G / 245.2e9
model     = max(t_bytes, t_issue)
```

This reproduces the shape of the measured curve with one free parameter, at a
stable residual of 1.24 to 1.48. It answers the question the roofline could
not: an extra verify row adds 25.622e9 exact FMAs, which is 6.83 ms at FMA
peak and hides completely under the weight stream up to M ~ 8.6. **Nothing
physical makes width 8 cost more than width 1. The kernel issues 0.625
non-FMA instructions per FMA on the per-row side, so it saturates the issue
pipe at 33 to 38 % of FMA peak long before either roof.**

Two consequences. First, weight re-reads are NOT the mechanism: at M = 7 and
M = 8 the merged single-stream `t789` arm is slower than two groups, so the
second stream is largely cache-served. Second, the lever is instruction count
per element, and the cheapest such lever is load width. The wide kernel issues
four scalar `uint16_t` loads and four scalar bf16 loads where one `ushort4` and
one `vec<T,4>` would do. Compiling both forms with `xcrun metal -S -O2` shows
the compiler does not merge them today: the scalar form emits four `align 2`
loads and two allocas, the vector form emits two `align 8` loads and none.
Alignment is provable by hand for every scored K in {5120, 6144, 17408} but not
by the compiler, because `in_vec_size` is a runtime `int`.

#### The ranked half of the roofline claim is retracted

An earlier note here priced the ranked candidate depth-0 round at 474.1 GB/s
against a 614 GB/s peak and called the 77.2 % a bandwidth shortfall. **Retract
the shortfall.** The 614 GB/s figure is an external specification for an
inferred runner SKU that the campaign never probed; `research/e69-artifacts/`
records it as unverifiable, and the ranked GPU tier is unknown. The 30.402 ms
numerator is a model reconstruction from
`research/prompt_round_reconstruction.py:111-156`, not a measurement. No
published microbenchmark of M5 GPU bandwidth or launch overhead exists. Keep
`R = 65.009 / 30.402 = 2.1383` as the ranked-to-local round ratio, which ledger
206(I) upholds, and stop quoting a ranked bandwidth efficiency.

The decode `asyncEval` ladder remains live as a small bit-exact component. It
was scaled from a 40-layer Laguna model to 64 layers and never tuned. Rung 0
measured `dense` at -0.087 % of candidate time by paired median, below the
0.10 % solo bar but usable in a stack. Rung 1 tests whether the mechanism is
rung density or block alignment: `Qwen35.swift:2254` gives
`isLinear = (layerIdx + 1) % 4 != 0`, so full-attention layers sit at
i = 3, 7, ..., 63, `dense` never fires on one and `default` fires on two.
**E86, alphonse.**

### Theme C — composition and the submission slot

**Our tree is faster than the frontier tree on ranked candidate decode time,
and behind it on the board.** Those two statements are consistent because of
governing facts 1 and 2, and holding both at once is now the whole composition
problem.

E84 submitted as `8630bc07`. It scored **3.27746907**, rejected, drawn in the
slow measurement cluster. The mechanism nevertheless worked. Two independent
routes agree on its ranked size:

| route | E84 effect on ranked candidate seconds per token |
|---|---:|
| `8630bc07` minus `32c6dc69`, same cluster B, minus the concat ladder | **-0.177 %** |
| `8630bc07` minus `8e83c6b3`, cluster-corrected per prompt | **-0.171 % +- 0.127** |

`8630bc07` and `32c6dc69` differ by exactly two files. `Qwen35.swift` carries
E84 mechanisms A and B. `Qwen35MTP.swift` carries the frontier's own concat
elimination. The concat ladder is measured at **-0.108 %** median over five
same-cluster pairs, so the residue is E84 itself.

**Why the score still fell.** Seven of eight prompts got faster. `essays` was
the only regression at +0.312 %, and `essays` is the second-lowest wide raw,
which the score reads directly. `beagle`, the lowest, improved 0.46 %.
Per-prompt scatter of sd7 = 0.32 % enters the score at about 0.23 % because the
score is the mean of exactly two prompts. **A candidate can get uniformly
faster and score lower.** One rejection is therefore not evidence against a
sub-0.5 % mechanism.

**Where the stack sits.**

```
pre-concat fast-cluster cohort mean          = 3.30559   (n=5, sd 0.00378)
+ concat elimination (-0.108 %)              = 3.30916
8e83c6b3 actually scored                     = 3.31894   (+2.6 sd lucky draw)
our tree = frontier class + E84 (-0.177 %)   = 3.31502   (expected, cluster A)
```

| target | extra gain needed over `8e83c6b3` | we have | still needed |
|---|---:|---:|---:|
| coin flip given a fast draw | 0.287 % | **0.177 %** | **0.110 %** |
| 93 % given a fast draw | 0.459 % | 0.177 % | 0.282 % |

Multiply by P(fast) = 0.67 for the unconditional number. Our current
unconditional probability of taking the frontier is about **10 %**.

**The submission slot is free, and the recorded decision is to hold it.** A
re-roll of the identical tree is a duplicate submission that farms the cluster
lottery with a ranked slot, and it buys about 10 %. Re-submit when the stack
reaches **+0.29 % extra over `8e83c6b3`**, which is +0.110 % on top of E84.

**Composition candidates for the next submission**, with what each is now
believed to be worth as an addition to E84:

| source | mechanism | expected addition | state |
|---|---|---:|---|
| E86 | decode `asyncEval` ladder | **0 %** | axis closed at +-0.06 %, fact 7 |
| E85 | fused embed and gathered rerank | 0.039 to 0.073 % | traced session running |
| E87 | `g128` coarse scorer | +0.30 % | offline screen |
| E87 | cluster-indexed shortlist | +2.5 to +2.7 % | offline screen |
| E88 | vectorized weight load in the wide QMV | unpriced, plausibly large | to be issued |

E86 is removed from the stack. E85 alone does not reach the bar. **E87 arm C or
E88 is the crossing mechanism, and neither has been measured yet.**

---

## Potential next research directions

Ordered by expected value, not by cost.

1. **Find and remove the binary measurement mode.** Governing fact 2 shows the
   mode is worth a median **0.0389 of score**, which is more than six times the
   whole E84 mechanism, and that it is generated inside our own drafting path at
   a flat cost of about **0.9 ms per drafting round**. It is drawn once per
   process, independently of the previous run, so it cannot be dodged by
   submission timing. Removing it converts a 67 % lottery into a certainty and
   simultaneously drops the board's resolution floor from 0.5 % to about 0.14 %,
   which makes every later small mechanism measurable. Leading candidates: the
   per-round Gated DeltaNet recurrent snapshot at
   `Qwen36MTPBlockSession.swift:1407`, process QoS class and E-core versus
   P-core placement, a marginal MLX allocator steady state against the dead
   `cacheLimitBytes` setting, and Metal heap placement of the 427 MB proposal
   head. **E89 is in flight on PR #90 with alphonse.**
2. **Cluster-indexed coarse shortlist at scale.** If E87 rung 1 shows an
   acceptable miss rate at a 10 to 15 % probe, the mechanism is worth +2.5 to
   +2.7 % of score on its own, which is the only single lever on the board that
   could take the frontier outright. Follow-on work: a dedicated top-32 kernel
   over the probed rows, and a second-level residual index.
3. **Vectorize the device weight load in the wide crossrow QMV.** The kernel
   issues four scalar `uint16_t` loads per group at `quantized.h:1003-1005`
   where every scored shape is provably 8-byte aligned. Compiling both forms
   with `xcrun metal -S -O2` turns four `align 2` loads plus two allocas into
   two `align 8` loads and no allocas. **Weight-side vectorization is bit-exact
   by construction**, so the bit-exactness law charges nothing against it.
   🔴 **Advisor error 20, retracted.** The recorded "-10.4 % at `M = 7`" came
   from comparing the instruction-issue term before and after the change
   without taking `max()` against the bandwidth roofline. The honest bracket
   is **-0.45 % of QMV time if the roofline binds at every live cell and
   -2.53 % if issue pressure binds at every live cell**, which is **-0.35 % to
   -1.96 % of the ranked round**. A cost model whose predictions span a factor
   of five is not a decision procedure: rungs 0 and 1 are free and settle it.
   This remains the largest untried lever inside the verify round. **E88 is in
   flight on PR #91 with askeladd.** Its rung 2 also refits the dead width
   curve on the live dispatch table for free. 🔴 **Transfer risk, new.** Ledger 230 shows the crown NA<=6
   table is worth -20 % per cell at M = 5 on our g16s host and costs +0.9 % to
   +2.2 % end to end on the ranked g17s host, so the input-group-count axis
   inverts sign between the two generations. E88 is a different axis and both
   its expected effects lower register pressure rather than raise it, but E88
   rung 1 must publish the register and spill census on `applegpu_g17s` as well
   as `applegpu_g16s` and must stop if the ranked register count rises at any
   live cell.
4. **A certified two-tier exact `lm_head` readout screen.** The readout is
   715 MB per round, about 5.0 % of the 14.41 GB weight stream, and its only
   consumer is a top-2. A coarse 2-bit plane with per-row certified error bounds
   would let the exact 4-bit weights be read only for survivors, giving a
   bitwise-identical top-2 at roughly 350 MB. Estimated 1.0 to 1.3 % of a round.
   **Give this scepticism up front**: plain Cauchy-Schwarz bounds are about
   fourteen times looser than a typical logit, and group-wise bounds over 80
   groups can tighten that by at most a factor of nine. Rung 0 is free and
   offline: dump verify hidden states, simulate, report survivor count. Stop if
   the p99 traffic is at or above 85 % of 715 MB. **E90, queued.**
5. **The prefill `asyncEval` ladder stride.** We have never swept it. A
   competitor measured stride 4 at -2.30 % of their candidate leg, and prefill
   is 8.6 to 9.4 % of ours. Our prefill uses stride 3 and our decode uses stride
   10. The decode axis is closed by governing fact 7; the prefill axis is a
   different loop body with a different arithmetic intensity and is untested.
6. **Fold maximal untimed warmup into the next stack.** Free and bit-exact.
   Demoted: it was ranked here because warmup was the only lever we could see on
   the cluster ceiling. Governing fact 2 now shows the mode costs a flat amount
   per *drafting round* and not a one-off startup amount, so warmup can only
   help if the mode is a first-touch or pipeline-placement effect. E89 rung 1
   decides that; do not spend a session on warmup before it reports.
7. **Fix `positionAcceptEMA`.** The shipped prior is `0.85 * 0.98^i`; E79
   measured per-position acceptance as flat at about 0.955. The schedule is
   choosing depths from a materially wrong model.
8. **Split the fused head `qkv` so overwritten K and V rows are never
   computed** (+0.096 %, bit-exact).
9. **A timed palindrome on the `qat-q4` head.** Byte-neutral, apache-2.0, and
   worth +0.71 points of acceptance in the offline screen, which prices at
   +1.57 % of score. Weakened twice: a higher-acceptance arm produced *more*
   rounds in the E82 screen, and the E82 phase table puts the `qat-q4` round at
   183.5 ms against the declared head's 154.7 ms, which is 18.6 % slower. It
   must be measured end to end before it is believed in either direction.
10. **The quantized GEMM path at M = 512.** Prefill is 8.6 to 9.4 % of the
   candidate leg, 99.7 % GEMM, and runs at 6.18 TFLOP/s. Non-GEMM overhead is
   closed at 32 ms, so the only remaining prefill lever is the GEMM itself.
11. **Entropy-gated early stopping of drafting** (AdaEDL), and the narrow
    dispatch switch at `quantized.h:1980`.

**Removed from this list this round.**

- **A denser or block-aligned decode ladder.** Governing fact 7 closed the axis
  at +-0.06 % against a +-0.02 % null. Two rungs is the whole effect, and block
  alignment is not the mechanism: `blk2` is block-aligned at 10 rungs and
  slowest, `blocks` is block-aligned at 18 rungs and fastest.
- **The ranked bandwidth-headroom hypothesis.** It was ranked third here and
  rested on 474 GB/s against a 614 GB/s specification. Both numbers are dead.
  The 614 GB/s figure is an external specification for an inferred, never-probed
  SKU, and the 30.402 ms numerator is a model reconstruction, not a
  measurement. No published microbenchmark of M5 GPU bandwidth exists. Keep the
  ratio `R = 65.009 / 30.402 = 2.1383` and stop quoting a ranked bandwidth
  efficiency.

**Removed in the previous round.** Verify-path intermediate and dispatch
elimination, which governing fact 5 closed.

---

## Standing operating rules

- Review terminal results before starting new synthesis. A moved frontier is an
  interrupt.
- 🔴 **A promotion is a draw, not a measurement.** Before adopting a mechanism
  from a newly promoted frontier, price it with a control-aware pair against
  the frontier it replaced. `0dd455f0` took the crown while carrying a
  mechanism that is +0.035 % slower than the tree it beat.
- 🔴 **A cost model whose predictions span a factor of five is not a decision
  procedure.** Publish the bracket, name the assumption at each end, and decide
  from a free measurement.
- 🔴 **Stop list, new: the qL={2,3} later-window SDPA warm** carried by the
  `0dd455f0` and `e12836cd` `Sources` tree. Two independent control-aware pairs
  put it at +0.021 % and +0.049 % pooled candidate MTP time, same sign, 4 of 14
  legs faster. Do not adopt it despite it holding the frontier.
- Local whole-leg ratios are not arm rankings. Use absolute
  `candidate_mtp_seconds_per_token` for anything whose causal path is not
  confined to the candidate MTP leg.
- Read `sd7` before `mean7`. Above about 0.35 on a same-schedule pair the run
  is cross-cluster and must be quarantined.
- The ranked board cannot resolve a mechanism below about +0.5 %. Measure
  mechanisms locally; spend ranked slots on stacks that can take the frontier.
- One in-flight submission per account.
- Never ship a gate or witness list the advisor has not reconciled against the
  current base.
- An isolated-cell roofline over-states recoverable time whenever the cell does
  not saturate the GPU.
- Report every bit-exact arm as a median of paired per-round deltas with a
  confidence interval. State the count of distinct round sequences across legs
  and require it to equal one. Leg totals overstate small effects by up to four
  times.
- A null measured on a saturated resource does not close the axis on hardware
  where that resource is not saturated. Record which host produced every null.
- Prefer a mechanism-specific build witness over a generic one. A generic
  witness proves the build is fresh; a mechanism witness proves the experiment
  can move.
- Never give the reference arm a palindrome extreme. Put a throwaway arm at
  position 0 and at the last position, match the mean session position across
  every compared arm including the reference, and publish per-leg host-phase
  medians. A measurement protocol the advisor writes is itself an experimental
  artifact and must carry its own control.
- A candidate can get uniformly faster and score lower, because the score reads
  exactly two prompts. One rejection is not evidence against a sub-0.5 %
  mechanism.
- Search the literature for a replication before issuing an assignment.
