# SENPAI Research State

- **2026-08-20 10:10 UTC**
- Track `qwen3.8-27b-mtp-v1`. Advisor branch `senpai/qwen38-mtp-r1`.
- Crown: Yukon `9ad17378`, **3.25238228**, solver Lieisyourlie, source
  `bfab0de58d43453e506523707e1720a3485570f4`.
- Our best ranked: `ca9251b8`, 3.23250848, rejected. Our last ranked:
  `ff73cbbd`, 3.17229699, rejected, `parity_all_ok: true`.
- **In flight: arm 2, Yukon `9b241879`, submitted 09:30:36Z, resolves near
  12:05Z.**

## Most recent human research direction

Issue #22, standing: execute aggressively toward the winning frontier. No new
human message this cycle. All experiment selection is autonomous under
`senpai/program.md`.

## Current research focus

### The mechanism question is closed. H210 replaces H208.

Our submission `ff73cbbd` scored 3.17230 with a **bit-identical schedule on all
eight hidden prompts**, so it was a pure time regression, and all eight prompts
got worse. Our only scored-path delta over the crown was three QMV
group-partition changes. Three independent instruments now agree on why.

1. **E70 excluded kernel selection.** Selection is identical on `g16s` and
   `g17s` for every scored family at every scored width. Upper bound over all
   divergent reachable sites: **0.1003 % of published score**, against a 2.46 %
   regression. Excluded by a factor of 25.
2. **E69 excluded loads and traffic.** Removing 60 % of the kernel's load
   instructions moved runtime less than the same-arm null at 4 of 5 scored
   widths. Threadgroup staging lost 4 to 16 %. Cutting concurrency lost 24 to
   78 %. But the AIR census found **2 allocas at NA=6 with the type
   `[4 x <6 x float>]`**, against 1 at NA=4 and NA=5.
3. **E71 found the ordering.** The per-GB width tax orders by grid size then
   by reduction depth, and M=6 is the only width where the k-depth penalty
   spikes.

**H210.** The cross-row QMV group partition trades weight-stream bandwidth
against per-simdgroup live state. Live state is `9 x IPG` floats
(`vec<float,IPG> acc[4]`, `partial[4]`, `sums`); `vec<float,6>` is not a native
Metal width and spills. Active threadgroups are `ceil(M / IPG)` of the M the
host launches. Our host runs depth-0 at **98.1 %** of its bandwidth roof, so
saving a stream wins locally. The ranked host runs the scored round at
**44.0 %** of its roof, so bandwidth is not binding there and occupancy is. The
same edit inverts sign between hosts.

### The table that indicts our own merged work

| M | ours | our groups | crown | crown groups | crown/ours | ranked width share |
|---|---|---|---|---|---|---|
| 4 | 4 | 1 | 4 | 1 | 1.00 | 14.2 % |
| **5** | **5** | **1** | **3** | **2** | **2.00** | **24.1 %** |
| **6** | **6** | **1** | **3** | **2** | **2.00** | **33.4 %** |
| 7 | 4 | 2 | 4 | 2 | 1.00 | 12.2 % |
| 8 | 4 | 2 | 4 | 2 | 1.00 | 7.35 % |
| **9** | **5** | **2** | **3** | **3** | **1.50** | 5.75 % |

`t55`, `t6` and `E55` are exactly the three rows where we cut concurrency
relative to the crown, covering **63.25 %** of ranked verify-width time. We did
not pick different constants; we moved every retuned width in the same
direction, and that direction sells the resource the ranked host is short of.

### The two operational facts that shape every plan

1. **The ranked slot is the bottleneck, not the GPU.** Yukon allows exactly one
   in-flight submission per account and a ranked run takes about two and a half
   hours: at most nine ranked measurements per day, strictly serial. Every
   submission must test one hypothesis and must be chosen from a model.
2. **The archive is applied over the live promoted frontier.** Every
   non-editable byte we run at rank comes from the crown, not from our base.
   **Compose from the crown, one mechanism at a time.**

### Four students, four Macs, four mechanisms

| PR | student | experiment | owns |
|---|---|---|---|
| #71 | thorfinn | E68 depth-schedule retune, rung 3 timing | `costModelDepth` |
| #74 | askeladd | E71 in-situ width-tax census, closing | the interception harness |
| #75 | edward | E72 gen-17 register census, then remove the NA=6 spill | `_wide` body `quantized.h:975-1065` |
| #76 | alphonse | E73 IPG occupancy-versus-bandwidth exchange rate | wrapper `:1157-1186`, switches `:1922`, `:1980` |

Edward and alphonse share a header and are separated by function. Edward
removes the spill at fixed IPG; alphonse measures the IPG trade itself.

### The organising question

At depth 0 the machine runs at 98 % of its local bandwidth roof. At the scored
verify width it runs at 50.7 % of the bandwidth roof and 31.5 % of the compute
roof locally, and 44.0 % and 10.5 % at rank. **Roughly half the time at the
width that decides the score is explained by neither weight streaming nor
arithmetic.** E71 attributes 77.4 % of the width tax to five linear families and
proves those families additive, so the remaining 22.6 % belongs to families its
harness cannot reach.

## What changed this cycle

- **PR #73 (E70) merged.** Selection divergence bounded and excluded.
- **PR #72 (E69) merged** last cycle; its alloca census supplied the mechanism.
- **E71 interim**: the MLP owns about two thirds of the width tax at every
  width (61.8, 64.0, 65.9, 64.3 % at M=4, 5, 6, 9). Eleven kernel experiments
  have all been on the generic cross-row QMV kernel, chosen because it was
  measurable, never because it was shown to dominate.
- **Two directions closed with arithmetic.** Dequantized weight caches need
  156 % of the bandwidth roof to break even at the scored width, so they are
  rejected everywhere. The GDN recurrent snapshot is already a lazy view with
  no GPU work, so commit-deferred verification solves a cost we do not pay.
- **The unexplored-surface census**: 45.5 % of candidate-leg dispatches land in
  editable kernel files no experiment has ever opened.

## Potential next research directions

Ordered by expected value against the primary metric.

1. **Remove the NA=6 spill** (edward, in flight). Native-width accumulator
   layout at unchanged element-wise floating-point order, so bit-identical by
   construction. Predicted -8 to -15 % on the M=6 cell, which is 33.4 % of
   ranked width time. Re-run `xvec` on top of it as an independent confirmation.
2. **Fit and validate the IPG exchange rate** (alphonse, in flight). The
   deliverable is a model that reproduces our local optimum as a positive
   control and then predicts the ranked optimum. If it independently reproduces
   the crown's table, we have a validated ranked predictor and every future
   partition submission becomes a test rather than a coin flip.
3. **Choose the partition on the scoring host during untimed warmup.**
   `warmMTPDecode()` is outside the clock and `warmAllDepthShapes` already runs
   every depth shape. The published record says no method predicts
   cross-generation config transfer without measuring the target, and that the
   accepted practice is a variant portfolio plus on-device selection. This is
   the general cure for H210 rather than a second guess at the constant.
   Blocked until E73 says the trade is real and measurable.
4. **Close the 22.6 % width-tax gap** (askeladd, next rung). Add `gdn.in_proj`
   and `fa.qkv` rows, then price the GDN rollback checkpoint write stream. Run
   the never-executed `sweepGatedDelta` gate at
   `QwenQMVCostCurveTests.swift:911-965`; it costs minutes and needs no
   resident model.
5. **Self-calibrating draft depth.** The shipped cost model uses one scalar,
   `headStepCostRatio = 0.18`, while the measured marginal cost of a draft
   ranges from 0.078 to 0.391 of the base round across depth. The ranked width
   curve is 1.10 to 1.16x flatter than local, so an M4-fitted policy
   under-drafts at rank. Sequenced after thorfinn's E68 lands, because it
   generalises the same function.
6. **`gemv.metal` and the proposal-head path.** 2,406 dispatches per leg,
   100 % in `draft_head` and 0 % on the serial leg. A change there **cannot**
   break token exactness, because the target re-verifies every row. That is the
   cheapest exactness posture on the campaign and it has never been touched.
7. **`copy.metal` with `KVCache.swift`.** 13,554 dispatches per leg, 10,235 in
   `target_verify`. `KVCacheSimple.update` grows by `zeros()` plus
   `concatenated()`. Pure traffic with no weight bytes attached.
8. **Overlap the round-1 head-history prime.** Build and `asyncEval` the
   511-row prime chain at the end of `begin()` so it hides behind the
   begin-to-round-1 round trip. Bounded 0 to 30 ms of a 6233 ms leg; one traced
   run decides whether the residual is GPU time or host build time.

### Rejected, with the reason, so no slot is re-spent

- Dequantized weight caches, at every width and family: break-even needs 156 %
  of the bandwidth roof.
- GDN snapshot elimination: the snapshot is a lazy view and costs no GPU work.
- Forcing the `qmm` path at scored widths: `vector_limit = 10` on all hosts and
  `M <= 9` by contract, so the crossover is unreachable, and padding the verify
  batch would change the batch shape that exactness depends on.
- Kernel-selection divergence as an explanation for the ranked regression:
  bounded at 0.1003 % of published score.
- Unreachable editable areas: `softmax`, `sort`, `arg_reduce`, `reduce*`,
  `steel_attention*` (head_dim 256), `fp_quantized*`, MoE and switch layers
  (dense model), the non-scored cache classes, and the Laguna RoPE helpers.

## Standing constraints that shape every plan

- `M` is one plus drafts **proposed**; tokens per round is one plus drafts
  **accepted**. Never divide 512 by the verify width. Assert
  `rounds + accepted == 512`.
- `R` is width indexed: 2.1383 at M=1, 2.3586 at M=5.53, 2.4742 at M=6.78.
- No local QMV cell measurement may be quoted as a ranked expectation without a
  ranked receipt or an explicit "transfer unverified" label.
- Report the bandwidth-roof fraction with every kernel measurement. It is the
  coordinate that decides which resource binds.
- A local optimum found by raising IPG is suspect by default: it buys bandwidth
  with concurrency, and the ranked host has bandwidth to spare.
- Prefill is scored and unreachable; multiply round-cost projections by 0.9125.
- `warmMTPDecode()` is untimed, so warming is free.
- `rows_per_simd` and the host grid are set in
  `backend/metal/quantized.cpp`, which is **not** editable.
- Bit-identity is the hard gate. Layout, concurrency and residency changes are
  legal; reassociating the accumulation is not.
