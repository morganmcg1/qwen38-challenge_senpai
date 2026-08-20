# SENPAI Research State

- 2026-08-20 18:55 UTC
- Most recent human direction: issue #22 — execute aggressively toward the
  winning frontier. No new human instruction since.

## Where the campaign stands

Official frontier `c6af1e24` = **3.30955573** (organizer `88578f9295523af1`,
accepted 2026-08-20T14:40). Two other trees landed just under it within the
hour: `94d14b4d` 3.30855 and `dda78bfa` 3.30344. The top pack is three trees
inside 0.19 %.

Our best official run `9b241879` = 3.23588901, a deficit of **2.29 %**. That
deficit is almost entirely staleness: every submission we have made was built
on a base that is now several promotions old. **Submitting the composed
advisor branch is the single highest-value action available and it is
dispatched.**

Three distinct candidates are now queued behind it, in order: the frontier
composition (thorfinn), the two dead-work eliminations (askeladd), and the
head requantization recovery (alphonse). Each student holds autonomous submit
authority once its own gates pass, and each checks `yukon submissions --all`
immediately before calling `senpai/submit-official.sh`.

One process rule came out of the first attempt. `swift test
--force-resolved-versions` on this checkout has a **standing floor of 40 issues
across 9 test functions in 7 files** — six suites of organizer staleness that
fail on `upstream/main` too, plus the campaign's `AGENTS.md` overlay. Every
failing input is byte-identical to `BASE_SHA`. The bare exit code is not a
gate; the gate is that no failing test name falls outside the recorded nine and
that the count does not exceed 40. The list lives in
`senpai/known-test-failures.md`.

## The measurement model — rewritten today, supersedes everything before it

Four claims were retracted today. Read this section before quoting any ranked
number.

### 1. The ranked host is a constant. There is no host factor.

Across 602 scored runs the runner-owned serial leg varies by cv **0.21–0.24 %**
per prompt, and the host index (serial mean over eight prompts) has sd
**0.115 %**. Correlation between the host index and any candidate prompt time
is **+0.02**. There is no host-speed variation to correct for, and every
"build factor", "host factor", and "host index" correction in this campaign's
history is withdrawn. plutarch is **not** a host-speed control. It is a control
for the non-drafting S=1 path only, because it is the one prompt with
non-drafting rounds (449 of 487; every other prompt has zero).

### 2. The ranked host became about 2.4× quieter on 2026-08-19.

Same-schedule pairs with a small effect, by day:

| day | n | median sd7 | sd(mean7) |
| --- | ---: | ---: | ---: |
| 08-15 | 15 | 0.157 | 0.254 |
| 08-16 | 19 | 0.197 | 0.113 |
| 08-17 | 36 | 0.170 | 0.214 |
| 08-18 | 46 | 0.174 | 0.224 |
| **08-19** | 61 | **0.071** | **0.128** |
| **08-20** | 41 | **0.081** | **0.088** |

Evidence taken before 2026-08-19 carries roughly 2.4× the noise of evidence
taken after it. Do not pool across that boundary.

### 3. The current ranked instrument resolves 0.09 %.

The clean null is six **note-only** submissions — the manifest note string is
the only change, so the tree is functionally identical — each measured against
its own base:

```
742cdf67 vs 9ad17378   mean7 -0.152  sd7 0.147   7/7 faster   ACCEPTED
bbc1622d vs 9ad17378   mean7 -0.091  sd7 0.159   6/7
abd41069 vs 9ad17378   mean7 -0.041  sd7 0.148   4/7
da9d4a20 vs 9d5569bb   mean7 -0.008  sd7 0.042   4/7
ad37b4b9 vs 59b321ee   mean7 +0.020  sd7 0.070   2/7
9d5569bb vs 59b321ee   mean7 +0.053  sd7 0.036   0/7          ACCEPTED
```

**Null: mean −0.036 %, sd 0.075 %. All six inside ±0.25 %.**

So for a same-schedule change, one ranked pair measures mean7 with sd ≈ 0.09 %.
A mechanism worth 0.15 % is measurable in one shot at about 2σ, and in three
shots at 3σ. This is a far better instrument than we believed this morning.

### 4. Read sd7 before you read mean7.

`sd7` is the scatter of the per-prompt deltas inside one pair.

- **Same-schedule pairs, small effect: median sd7 = 0.117, p90 = 0.369.**
- **Schedule-changing pairs, small effect: median sd7 = 1.350, p90 = 2.268.**

A pair with sd7 above about 0.35 is either a schedule change or a disturbed
run. Quarantine it. Do not report its mean7 as a uniform effect.

This kills two of my own claims. The frontier chain
`742cdf67 → 89cbdc02 → ead84bba → c6af1e24` is **three schedule-changing pairs
in a row**, with sd7 of 1.676, 1.117 and 0.642. The "fixed cost of 0.86 ms per
drafting round" was read off that chain and is **withdrawn**. **H-221** — the
claim that the head path pays about 0.35 ms per MLX op boundary — rested on
attributing `89cbdc02 → c6af1e24` entirely to `qwen35DualRMSNorm`, but that
pair also changed the verify width cap.

**H-221 is now DEAD in the per-MLX-op form.** Three independent measurements
killed it on the same day, in three different regimes:

- **Edward, decode GPU-time census.** Prefill carries 512 rows per dispatch and
  is throughput bound; the head at decode width is launch bound. A per-dispatch
  tax cannot be one constant across both.
- **Thorfinn, prefill.** 2265 dispatches at 0.35 ms would be 793 ms, 19.6 % of a
  4046.5 ms `begin()` whose isolated GEMM sum already accounts for 100.2 % of
  it. There is no room for the tax.
- **Alphonse, head path.** The declared readout runs 10 MLX ops and 6 dispatches
  against the pinned path's 3 and 2, so it pays seven extra boundaries per
  draft. At 0.35 ms those seven alone would cost 2.45 ms, more than the entire
  measured declared head step of 2.381 ms.

The one surviving variant is a cost per **host synchronisation point**, a
different quantity with a different count — 23 forced eval points per round,
not 2265 dispatches — and it exists only in the decode regime. Edward owns it.

### 5. The score is a 3× noisier instrument than mean7.

The six note-only nulls moved the published score by +0.07, −0.47, −0.32,
−0.35, −0.45 and +0.06 percent: **mean −0.24 %, sd 0.24 %**. The score is the
mean of the 4th and 5th sorted raw ratios, so it inherits order-statistic
noise that averaging over eight prompts removes. Judge every change by mean7
and by sd7. Never by the score.

The −0.24 % mean is base-selection regression: a base is promoted partly
because it drew well, so a re-measurement of the same tree regresses.

### 6. The official frontier advances partly on measurement noise.

Three of the last twelve promotions carry **zero functional change**:

- `Accept submission 11863aa9` (`5068eb8`) has a **completely empty diff**
  against its parent. It is a bit-identical resubmission of `4f76de6e`.
- `742cdf67` and `9d5569bb` were both promoted on **note-string-only** diffs.

This is not an accusation of misconduct; resubmission is permitted and the
board records what it measures. It is a calibration fact. It means the recorded
frontier sits above the top pack's true level by a lucky draw of order +0.2 %,
and a candidate at true parity is recorded above the frontier roughly one time
in five.

## Priced and ready: three levers with ranked or matched-control evidence

Both are absent from the organizer tree, both are pure removals of computed
work that is then discarded, and both now have clean same-schedule ranked
evidence in the quiet regime.

**A. Island K/V full-precision elimination.** The pinned head declares
precision islands with `K=all, V=all`, so the quantized K/V projection result
is 100 % overwritten by the BF16 island scatter. Detect that the island index
set is a complete permutation of the K/V rows, pack **Q only** into the
quantized path, and slice K and V straight out of the exact BF16 matmul. The
K/V-only committed-history flush skips the quantized pack entirely.

```
c37b4f67 vs 9d5569bb   mean7 -0.190  sd7 0.063   7/7
9383f9a4 vs 9d5569bb   mean7 -0.168  sd7 0.046   7/7
11a9412a vs 9ad17378   mean7 -0.157  sd7 0.196   6/7
POOLED -0.172 %, n=3, SE 0.052  ->  3.3 sigma; -0.136 % net of the null
```

**B. State-only Gated DeltaNet prefix replay.** `replayPrefix` needs only the
recurrent boundary state, but the shared `gated_delta_step` kernel also forms
the output accumulation, its simd reduction and the `y` store, all of which are
discarded. A twin kernel with those three deleted, same grid and threadgroup,
same state statements in the same order.

```
a6661c80 vs 9ad17378   mean7 -0.183  sd7 0.162   7/7
04cd6f95 vs 9ad17378   mean7 -0.154  sd7 0.206   7/7
POOLED -0.169 %, n=2, SE 0.064  ->  2.6 sigma; -0.133 % net of the null
```

Combined expectation about **−0.27 % of candidate time**, which maps roughly
1:1 into the median. Both must be reimplemented from the mechanism, and the
prior art must be credited in the submission note. Assigned as E84, PR #86.

**C. The head's requantization loss — the largest single unclaimed number on
the board.** The declared head is the naive affine-4 g64 round-to-nearest
requantization of the untrained EigenLabs master. Alphonse measured both, on
identical seeds, against the target's own greedy chain:

```
master-bf16   93.13 %  pooled acceptance, depths 3-6
declared      92.31 %  the SAME weights after round-to-nearest
                       -0.82 pt paid to quantization damage and nothing else
```

The damage is recoverable, not intrinsic. xkm's quantization-aware parent
requantizes with relL2 **2.89e-2 … 3.52e-2** against the declared head's
**9.18e-2 … 9.97e-2**, a 3.2× reduction, and recovers **0.71 of the 0.82 pt at
identical bytes**. Quantization-aware training is one route; a better quantizer
applied to the master weights we already hold is the direct one. It needs no
training, adds no bytes, and changes only `mtp-head.manifest.json`.

Priced against the frontier's own per-prompt rows, +0.71 pt is about **+1.57 %
of median (+0.049 absolute)** and the full +0.82 pt is about **+1.81 %
(+0.057)**. The promotion that currently leads the board was worth +0.0073.

**How to recover it (ledger 224).** MLX's affine quantizer is not min-max. It
flips the scale sign and snaps the dominant-magnitude edge onto an exact
integer code, spending half the group's range on an endpoint that carries no
special weight (`quantized.h:2973-2986`, verified in the checkout). That design
choice is the mechanism behind the 0.82 pt. Three tiers recover it, cheapest
first: a shrink-grid plus closed-form least-squares refit of `(s, b)` with the
codes held fixed, which cannot regress; then HQQ, whose published defaults are
already `nbits=4, group_size=64, axis=1`; then gradient descent on `(scales,
biases)` alone against the BF16 master, which MLX supports and which is **not**
the closed head-training axis because the integer payload never moves. There is
a free positive control: reproduce `declared` bit-exactly from `master-bf16`
with MLX's own quantizer before changing anything.

A second, independent head lever rides on the same finding. Head time is linear
in head bytes: 179 MB/ms on two arms, agreeing to 0.75 %. At E79's anchor of
0.0844 % of score per 1 % of head cost, coarsening the 2-bit sweep's scales and
biases from g64 to g128 removes 15.7 MB (3.68 % of the head) for about
**+0.31 % (+0.0098)**. A worse coarse shortlist cannot break exactness — it
feeds an exact affine-4 rerank and the target verify rejects wrong proposals —
so acceptance is the only currency this axis spends, and arm `qat-q4` has
already banked some of it.

🔴 **The matching head-trunk lever is withdrawn.** One global
`(groupSize, bits)` tuple governs every quantized submodule including the head,
and `Qwen35Config.swift:266` asserts group size 64. The earlier `+0.0072`
trunk-metadata line was wrong. The draft-readout lever survives but is **not**
a manifest-only change: group size is hard-coded at
`MLXLLM/Models/Qwen35.swift:3232` and `:3313`, and the shape guard at
`:3294-3295` fails silently into a wrong-bits path. It needs that editable file
too. Assigned as E82 rungs 4-8, PR #84.

## Closed axes — do not reopen without a named new reason

- **Schedule**: +0.46 % total headroom from the new frontier.
- **Registers** (bound −1.209 %), **occupancy** (0.52 %), **the copy family**
  (ceiling 0.016 %), **low-rank draft readout** (full rank is load-bearing),
  **head fine-tuning and distillation on prose corpora** (six ranked
  negatives), **the depth-price level** (bracketed both sides on rank).
- **Narrow-N M=1 crossrow routing**: measured null (−0.074, 4/7).
- **The decode asyncEval ladder at S=1**: worth +7.29 % of the plutarch leg and
  **exactly zero** on all seven drafting prompts (`763e6f6f`, mean7 +0.005,
  sd7 0.063). plutarch can never enter the median, so this is score-free.
- **The decode ladder at S=3..9**: already shipped, worth about +0.49 % mean7,
  benefit monotone decreasing in draft width (`74db80ab`).

## Next research directions, in priority order

1. **Recover the head's requantization loss.** 0.82 pt of acceptance measured
   lying on the floor, worth about +0.057 absolute, obtainable with a better
   affine-4 quantizer on weights we already hold. No training, no extra bytes,
   two lines of `mtp-head.manifest.json`. This is the largest single priced
   number in the campaign. E82 rungs 5-8.
2. **Ship the two dead-work eliminations.** Priced on rank at about −0.27 % of
   candidate time. E84.
3. **The two virgin prefill fast-path gates.** A census of every diff in all
   846 board trees found that the decode-width bound in
   `if S <= 9, let fused = fusedInProjections(inputs)` at `Qwen35.swift:1003`
   and in `if x.dim(-2) <= 16, let y = fusedGateUp(x)` at `Qwen35.swift:1292`
   has **never been raised above 9 or 16 by anybody**. Five and eight
   submissions respectively touched those lines, every one of them to revert,
   rename, or restore the same bound. At the 512-token seed width the GDN layer
   therefore issues four quantized GEMMs per layer instead of one, priced by
   direct measurement at 94.6 ms of a 4046.5 ms `begin()`, and the SwiGLU
   intermediate is materialised instead of fused. Prefill is 8.6-9.4 % of the
   candidate leg and is the one place a local fraction transfers exactly,
   because every prompt on every host seeds exactly 512 tokens. E83.
4. **Head bytes.** Head time is linear in head bytes at 179 MB/ms. The 2-bit
   sweep spends 31.47 MB, 7.36 % of the whole head, on scales and biases for a
   stage whose only job is a 32-row shortlist, and whose recall@32 measures
   exactly 1.0000. Correctness has a hard floor on this axis. E82 rung 7.
5. **The host synchronisation point.** The only surviving form of H-221. 23
   forced eval points per round, not 2265 dispatches, decode regime only.
   Closure gap 2.254 ms/round at width 1 and 3.177 at width 6. E80.
6. **Delete the pre-fc concat.** `qwen35DualRMSNorm` already computes both
   halves; write them into one buffer of width 10240 at column offsets 0 and
   5120 so `fc` consumes it directly. Bit-identical by construction. Scope it
   narrowly: a larger fused pre-FC embed/RMS/concat was a local negative at
   +23 % MTP time.
7. **Tune the decode ladder rung set.** The shipped `[0,1,9,19,29,39,49,57]`
   was scaled from a 40-layer to a 64-layer stack and has never been tuned for
   this model. It is env-selectable, so measuring it costs no code change, and
   the board supplies a calibrated positive control: `off` at S=1 must cost
   about 7 % on plutarch. Nobody has added a rung in the 9 < S < 512 dead band,
   and the `>= 512` prefill gate has never been changed in 861 trees.
8. **The narrow dispatch switch** at `quantized.h:1980`. Every promoted kernel
   change has been on the wide switch behind `out_vec_size >= 4096`. The narrow
   switch serves proposal-head shapes only and cannot touch the serial
   numerator. M=1 there is a null; the rest of the table is unexamined. Price
   the call path before spending GPU on it.

## Live experiments

| PR | student | question |
| --- | --- | --- |
| #85 | thorfinn | E83 — prefill decomposition, plus the two virgin prefill gates |
| #83 | edward | E80 — per-kernel GPU-time census; the 22.6 % is now named |
| #84 | alphonse | E82 — head economics; rung 0 no-go, redirected to the requantization loss |
| #86 | askeladd | E84 — the two ranked-measured dead-work eliminations |

## The board-wide instrument

Every submission the organizer has ever received is a fetchable branch:

```
git fetch upstream 'refs/heads/submissions/*:refs/remotes/upstream/submissions/*'
```

859 branches, 8 MB, 1.3 s. Each is exactly one commit on top of a commit in the
linear `upstream/main` chain, so `merge-base` yields the exact base tree and the
diff is the complete change. **Step back one commit when the merge-base commit's
subject names the submission itself** — accepted submissions fast-forward into
`main`, so a naive merge-base returns the branch's own tip and the submission
pairs with itself.

Joined against `officialMetrics.per_prompt`, which is populated for every scored
submission, this gives a per-prompt candidate-time A/B for 543 submissions
against their exact bases. Diff before theorising about any competitor.

## Standing measurement rules

- **Score = (beagle_raw + slowest other wide prompt) / 2.** The 4th slot is
  beagle in 100 % of strong trees; the 5th is currently essays.
- **Judge by mean7 and sd7, never by the score.**
- **Read sd7 first.** sd7 > 0.35 on a same-schedule pair means the run was
  disturbed; sd7 > 0.35 with a schedule change means the effect is not uniform
  and cannot be summarised by a mean.
- A uniform candidate-side speedup maps into the median roughly 1:1. Never
  subtract a locally measured serial-path share when pricing official value.
- The local fixture sits at mean verify width 7.27 against 5.82 on rank. Local
  whole-leg numbers are not arm rankings; per-width and per-cell numbers are.
  Prefill is the exception — one fixed 512-token cell on every prompt and host.
- Local total marginal ratio 0.4023 against 0.2136 on rank, so local pricing of
  draft depth is about 2× too dear.
- **Verify every `path:line` citation in the checkout before it becomes an
  instruction.** A delegated agent's source claim is a lead, not a fact. Three
  of four re-verified cleanly this cycle; the fourth named a file that does not
  exist and would have cost a student an hour.
- **Never ship a gate the advisor has not run.** `swift test
  --force-resolved-versions` exits 1 on a clean base and has a standing floor of
  40 issues across 9 named test functions in 7 files. Gate on the name set and
  the count, never the exit code.
