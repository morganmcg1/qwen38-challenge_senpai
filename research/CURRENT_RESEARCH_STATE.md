# SENPAI Research State

- **2026-08-20 10:50 UTC**
- Track `qwen3.8-27b-mtp-v1`. Advisor branch `senpai/qwen38-mtp-r1`, base
  `d19d6f5c9612da313785eb32038d9e3781fcc9a4`.
- Crown: Yukon `9ad17378`, **3.25238228**, solver Lieisyourlie, source
  `bfab0de58d43453e506523707e1720a3485570f4`.
- Our best ranked: `ca9251b8`, 3.23250848, rejected. Our last ranked:
  `ff73cbbd`, 3.17229699, rejected, `parity_all_ok: true`. Deficit **0.61 %**.
- **In flight: arm 2, Yukon `9b241879`, workflow run `32357580843`, entered the
  M5 runner 10:19:13Z, step 31 of 67, bypass review passed.**

## Most recent human research direction

Issue #22, standing: execute aggressively toward the winning frontier. No new
human message this cycle. All experiment selection is autonomous under
`senpai/program.md`.

## The two facts that now govern every decision

### 1. The published score is the mean of beagle and medicine

The eight `raw_p`, sorted: 1.2528, 1.9163, 2.1795, **3.1201**, **3.3446**,
3.3666, 3.3930, 3.4253. The median of eight is the mean of the fourth and
fifth: beagle at M = 5.5327 and medicine at M = 5.7677.

**The score is set by the speed of an M ~ 5.5 to 5.8 round and by nothing
else.** Price every mechanism through that round.

### 2. The verify-width tax is 19.8 % of ranked candidate decode time

The ranked candidate already runs a 30.402 ms depth-0 round while streaming the
identical 14.412 GB. A wider round streams the same bytes. So an empirical
floor for a width-M round, invoking no roofline, is
`30.402 + (M-1) * 8.42/8` ms. Across the eight legs that leaves **12,645 ms of
tax out of 63,766 ms**, and the reconstruction reproduces our measured
`ca9251b8` score to 0.004 %.

| tax recovered | published median | vs crown |
|---:|---:|---:|
| 0 % | 3.2324 | -0.6 % |
| 5 % | 3.2832 | +0.9 % |
| 10 % | 3.3356 | +2.6 % |
| 20 % | 3.4456 | +5.9 % |
| 50 % | 3.8239 | +17.6 % |

**Five percent of the tax beats the crown.** Neither roof binds at the floor:
a width-5.53 round at the floor would sit at 18 % of ranked compute and the
measured round sits at 10.5 %. The tax is latency, not physics.

## Current research focus: the working-threadgroup knee

**The law.** Cost is rank-ordered by working threadgroups,
`ceil(M/IPG) * ceil(n/8)`. Above the knee, saving a weight stream wins. Below
it, concurrency wins. The local knee is near **1900 working threadgroups, about
95 per core on our 20 cores**.

Two instruments, 150 ledger items apart, two students, two methods, Kendall
tau = -1.0 in both:

- **E33** measured eight scored shapes at M = 6 under a grid-halving arm. Ratios
  run 0.9830 at 62080 threadgroups to 1.0592 at 1280. The traffic ratio is
  exactly 1.3571 on all eight, so a traffic model is blind to the effect by
  construction. Sign flips between 1792 and 2060.
- **E71** (merged today) measured the in-situ per-byte rate at real occupancy.
  ms/GB at M = 6: `lm_head` 2.995 at 31040 threadgroups, `mlp_gate_up` 3.259 at
  4352, `gdn_out_proj` 3.901 at 640, `fa_o_proj` 4.099 at 640, `mlp_down` 5.263
  at 640. Its shape-invariance control settles that the per-byte rate is a
  property of the shape and the grid, not the byte count.

The vendored source states the mechanism at `quantized.h:1917-1921` and puts one
threshold at `out_vec_size >= 4096`. **`mlp.down` at n = 5120 is the smallest
scored shape above that gate**, so the threshold is one shape too low.

**Why the trade inverts between hosts.** Both host terms move the same way.
Bandwidth ratio 614 / 227.9 = 2.69x makes the stream saving worth less at rank.
Core ratio, inferred at about 2x, makes the occupancy loss cost more at rank.
The exchange rate moves by roughly 5.4x, which is why `t6` won -4.199 %
isolated locally and the composed candidate lost 2.46 % at rank. **The core
count is an inference and every use of it must be labelled.**

## The leading candidate mechanism: per-shape IPG

`out_vec_size` is a live kernel parameter and already gates the wide tier.
Nothing prevents the IPG choice from depending on both `ntg.x` and
`out_vec_size`. A two-dimensional table keeps the one-group path above the knee
and uses two groups below it, **winning on both hosts at once** — the property
every previous kernel arm lacked.

Exactness is inherited rather than argued: E66 proved 12 of 12 exact that
changing the group partition is bit-identical with `max_abs_ulp_top2_logits = 0`,
because partitions are unordered and per-element accumulation order is
untouched. Small diff, bit-width neutral, no manifest touch, no FP
reassociation.

## Live assignments

| PR | student | experiment | role in the programme |
|---|---|---|---|
| #71 | thorfinn | E68 draft-depth retune | rung 3 relaunch in flight; the schedule arm for a future submission |
| #75 | edward | E72 register census and NA=6 de-spill | can reopen the M = 6 one-group axis by clearing the 128-register wall |
| #76 | alphonse | E73 IPG exchange rate | fits the occupancy model and owns the wrapper and switch implementation |
| #77 | askeladd | E74 in-situ threadgroup knee | locates the knee across M = 4..9 with zero source edits and recommends the bands |

Surfaces are disjoint: edward owns the `_wide` body `quantized.h:975-1065`,
alphonse owns the wrapper `:1157-1186` and both `switch (ntg.x)` blocks at
`:1922` and `:1980`, askeladd changes no candidate file.

## The 128-register wall

E38 read `peak_live_regs` at `rows_per_simd = 4`: na2 62, na3 83, na4 104, na5
125, na6 **144**. Steps +21, +21, +21, +19. na6 is the only cell with
`allocas = 2` of type `[4 x <6 x float>]`, so 144 is post-spill and true demand
is at least 146. NA = 6 at r = 4 does not fit in 128 registers, r = 2 is forced,
and r = 2 was measured as a +10.54 % tax against an +11.96 % prize.

**edward's rung 2 is the first thing that can reopen this.** Target: recover 16
to 18 registers with element-wise FP order unchanged. The live float count does
not change; the fix is allocatability, not demand.

## Potential next research directions

1. **Per-shape IPG bands**, once E74 places the knee and E73 fits the exchange
   rate. Highest expected value and the safest exactness posture available.
2. **De-spilled NA = 6 composed with the bands.** Faster one-group path above
   the knee, two groups below it. The two halves need each other.
3. **The 22.6 % census residual.** `gdn.in_proj` at 2.28 GB and `fa.qkv` at
   0.66 GB are unreached by the E71 harness and owe an estimated 78 % of the
   gap. They also straddle the local knee at 2060 and 1792 threadgroups, so
   they are the most informative shapes in the model.
4. **`gemv.metal`, 2406 dispatches per round, 100 % draft head and 0 % serial
   leg.** A change there cannot break token exactness because the target
   re-verifies. The cheapest exactness posture on the campaign and never
   examined.
5. **`rms_norm.metal`, 6418 dispatches**, of which 3648 are exactly one stock
   RMSNorm per GDN layer per round. The campaign fused the attention norms and
   never touched the GDN-side stock kernel. AOT only, so it needs
   `tools/build-mlx-metallib.sh`.
6. **`copy.metal`, 13554 dispatches, 17.0 % of all in-round work**, driven by
   `KVCacheSimple.update` growing by `step = 256` with a full `concatenated`
   copy. Pre-reserving to 1280 rows is prompt-independent. Sized at only 0.3 to
   1 ms of 6233 ms, so it is a cleanup rather than a lever.
7. **Round-1 head prime**, +29.5 ms with only 3 to 6 ms attributable to compute.
   `asyncEval` of the prime chain at the end of `begin()` is worth 5 to 15 ms,
   bounded at 30.
8. **Adaptive draft depth on entropy** (AdaEDL), training-free and
   host-independent, as a hedge against cost-curve transfer error. The depth
   optimum is empirically flat, so the shape class matters far more than the
   exact constant.

## Closed with arithmetic, do not reopen without new measured cost

- Dequantized weight caches. 4-bit g64 traffic is 1.125P against 4P for a bf16
  dequant, a 3.556x inflation that pays only below 28 % of the bandwidth roof.
  Break-even at width 5 to 6 needs 960 GB/s, 156 % of the roof.
- GDN snapshot elimination. `snapshotRecurrent` captures a lazy slice view and
  does no GPU work.
- Tree verification for the delta-rule target. Published work leaves efficient
  single-pass tree verification for non-commuting recurrences open. Our linear
  0..8 chain is the right structure.
- Acceptance-side scheduling levers, the head-weight replacement, and
  `headStepCostRatio` retuning. All measured, all closed.
