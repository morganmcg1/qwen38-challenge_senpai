# SENPAI Research State

- 2026-08-20 17:20 UTC
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
pair also changed the verify width cap. H-221 is **unproven, not disproven**,
and needs a same-schedule test.

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

## Priced and ready: two dead-work eliminations

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
prior art must be credited in the submission note.

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

1. **Ship the two dead-work eliminations** above as one composed candidate.
   Highest expected value and already priced on rank.
2. **Prefill.** 8.6–9.4 % of the scored candidate leg. Across 470 scored runs
   the `prefill_seconds_per_token` p10–p90 band is 3.4 % wide — the whole field
   treats it as a constant and **no submission has ever changed the
   `inputs.dim(1) >= 512` prefill ladder gate**. Worth +0.903 % per 10 % cut.
   Under test as E83.
3. **Retest H-221 with a same-schedule pair.** If one removed op boundary on
   the head flush really is worth about 0.5 % of median, it is the largest
   remaining lever and it generalises. If it is worth 0.05 %, a whole class of
   fusion work is dead. Either answer is valuable.
4. **Delete the pre-fc concat.** `qwen35DualRMSNorm` already computes both
   halves; write them into one buffer of width 10240 at column offsets 0 and
   5120 so `fc` consumes it directly. Bit-identical by construction. Scope it
   narrowly: a larger fused pre-FC embed/RMS/concat was a local negative at
   +23 % MTP time.
5. **Tune the decode ladder rung set.** The shipped `[0,1,9,19,29,39,49,57]`
   was scaled from a 40-layer to a 64-layer stack and has never been tuned for
   this model. It is now env-selectable, so measuring it costs no code change,
   and the board supplies a calibrated positive control: `off` at S=1 must cost
   about 7 % on plutarch. Nobody has added a rung in the 9 < S < 512 dead band.
6. **The narrow dispatch switch** at `quantized.h:1980`. Every promoted kernel
   change has been on the wide switch behind `out_vec_size >= 4096`. The narrow
   switch serves proposal-head shapes only and cannot touch the serial
   numerator. M=1 there is a null; the rest of the table is unexamined.
7. **Head bytes, not head quality.** Head cost ×0.75 is +2.11 % of median and
   the head step is bandwidth-scaling (2.05× cost for 1.99× traffic). No custom
   head has beaten the pinned head on beagle across ~40 board digests, so the
   lever is bytes moved per draft step, not acceptance.

## Live experiments

| PR | student | question |
| --- | --- | --- |
| #85 | thorfinn | E83 — decompose the untouched prefill leg |
| #83 | edward | E80 — per-kernel GPU-time census, name the unattributed 22.6 % |
| #84 | alphonse | E82 — requantize the one genuinely retrained published head |
| #81 | askeladd | E78 — width-dependent QMV inner-group count |

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
