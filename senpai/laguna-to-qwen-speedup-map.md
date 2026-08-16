# Laguna XS to Qwen 3.8 speedup transfer audit

**Snapshot:** 2026-08-16 10:41 UTC

**Purpose:** record what the final Laguna XS frontier actually optimized, map
those mechanisms onto the current Qwen 3.8 frontier, and separate quick
experiments from already-landed, model-specific, or previously negative work.

This is a research map, not a claim that a Laguna patch can be copied verbatim.
The two models share MLX, Swift orchestration, Metal kernels, RMSNorm, quantized
linear layers, attention, KV state, and vocabulary selection, but their dominant
work is different. Measure each transfer on the current Qwen frontier and keep
the exact-value gates authoritative.

## Executive conclusion

The final Laguna tree is valuable prior art, but most of its largest wins are
not copy-ready for Qwen:

- Laguna is a 40-layer, hidden-2048, group-16 NVFP4 MoE with 256 routed experts,
  head dimension 128, and vocabulary 100,352.
- Qwen is a 64-layer, hidden-5120, dense affine-4/group-64 hybrid with 48
  recurrent Gated DeltaNet layers, 16 full-attention layers, head dimension 256,
  vocabulary 248,320, and a native speculative MTP head.
- Therefore Laguna's router, expert-bound, GatherQMM, shared-expert, rotating
  sliding-window, and E2M1/E4M3 scale tricks do not transfer directly.

There are nevertheless several worthwhile seams. In priority order:

1. **Port the exact-fill `KVCacheSimple` first-update fast path.** It is a small,
   exact change that avoids constructing zero-backed buffers and replacing
   roughly 32 MiB of target K/V slices at the charged 512-token seed (up to
   roughly 64 MiB of logical write traffic), plus the MTP-head cache where
   applicable. Physical traffic still needs profiling under MLX laziness.
2. **Do not duplicate the currently validating public composition.** A source
   stack containing wider decode/seed `asyncEval`, a seed-safe Q/K-prep gate,
   a target-margin draft prior, and compiled MLP SiLU/product scored
   `2.90354559365115`; its first promotion failed on stale harness ancestry,
   not correctness. Adopt it if a rebased receipt promotes, or isolate its
   components only if it does not.
3. **Extend the packed Gated DeltaNet prework kernel to widths 1 and 2.** The
   current fused mixer covers 3...9, leaving the common serial/narrow cases on
   a less fused path. This is Qwen-specific rather than a literal Laguna port,
   but it applies Laguna's most durable lesson: eliminate a producer-consumer
   intermediate, not merely a launch.
4. **Try the generic RMSNorm input-cache hunk.** It is a direct, exact Laguna
   carryover, but the easy hunk accelerates `rms_single_row`; Qwen's main
   hidden-5120 norms use `rms_looped`, so expected whole-model impact is limited
   unless a separate looped or residual-plus-RMS design proves safe.
5. **Treat certified target LM-head screening as the largest research bet.**
   Laguna avoided most reads of a 100,352-row BF16 head. Qwen still evaluates
   the exact 248,320-row target head for every verification row; its compact
   vocabulary is proposal-only. A Qwen certificate could be valuable, but it
   must preserve both exact top-two IDs and values.

The current promoted Qwen score is already `2.87642940762738` and the hard
plausibility ceiling is 3.0. That leaves only about 4.3% multiplicative headroom
before a faster result can fail rather than receive a clamped score. Prefer
small, attributable candidates and inspect all eight prompt results before
stacking wins.

## Reproducible endpoints

### Final Laguna authority

| Item | Value |
| --- | --- |
| Final organizer commit | [`4ea72c3b28873fca23b12b6f33193a2eeb5042f8`](https://github.com/Layr-Labs/mlxfast-challenge/commit/4ea72c3b28873fca23b12b6f33193a2eeb5042f8) |
| Senpai integration commit | [`f52be6aa9cefabc74f0f369d29d54e0686284aac`](https://github.com/morganmcg1/mlxfast-challenge_senpai/commit/f52be6aa9cefabc74f0f369d29d54e0686284aac) |
| Final receipt | `cdcd0918-0002-45b0-a14b-81f34c40a398` |
| Final score | `2.6195531094824` |
| Published throughput | 203.937 decode TPS; 5,314.295 prefill TPS |
| Official pinned baseline | `15852ee52858def42ddd4f32bca7e59d275e020e` |
| Scoring | `decode_speedup^0.75 * prefill_speedup^0.25` |
| Durable campaign synthesis | [final research-frontier briefing](https://github.com/morganmcg1/mlxfast-challenge_senpai/blob/29250d98d2886dbc11040a466ef758ff490f2ae3/senpai/research-frontier-briefing.md) |

The final editable surface differs from the official pinned baseline in 68
files (`+20,205/-17,564`). Most implementation mass is in
`LagunaRuntimeModel.swift`, `LagunaLmHeadPrune.swift`, the NAX NVFP4 kernels,
`quantized.cpp`, `LagunaRuntimeWeights.swift`, `sdpa_vector.h`, and
`SwitchLayers.swift`. The final result was overwhelmingly a runtime/kernel
campaign: no offline-transform mechanism drove the frontier.

The final executable was effectively identical to an earlier rejected receipt
apart from an ordinary comment, while cross-receipt decode and prefill moved in
opposite directions. The source is authoritative; individual leaderboard deltas
are noisy evidence, not clean causal attribution.

### Current Qwen authority

| Item | Value |
| --- | --- |
| Campaign commit at audit | `80c26b679198f83cff05df0c75c66b8959fa9b37` |
| Organizer contract commit | `26ae2bf6326de93e7f1b1b0aaf94a7667aca797b` |
| Promoted editable snapshot | `df404e08fee2ef8681f5bf2d68fe841969788eaf` |
| Promoted receipt | `aa7c3e0c-20d1-4b27-a80c-e622e7880999` |
| Official score | `2.87642940762738` |
| Score | median of eight per-prompt serial-relative speedups; floor 0.90; ceiling 3.0 |

The campaign commit overlays the promoted editable snapshot onto the current
organizer contract. [`frontier-state.json`](frontier-state.json) and the live
Yukon receipt are the operational authorities; this report is a timestamped
comparison and can become stale quickly.

## Architecture and cost-center comparison

| Property | Final Laguna XS | Current Qwen 3.8 | Transfer consequence |
| --- | --- | --- | --- |
| Layers / hidden | 40 / 2,048 | 64 / 5,120 | Fixed-width kernels require new geometry. |
| Block type | One dense MLP, then 39 sparse MoE blocks | Dense MLP throughout; 48 GDN + 16 full-attention blocks | Laguna router/expert machinery is irrelevant; recurrent-state work is new. |
| Quantization | NVFP4 group 16 for expert/attention weights, BF16 elsewhere | affine 4-bit group 64 backbone and declared MTP head | Carry byte-layout ideas, not codebooks or scale formats. |
| Attention | 30 sliding + 10 full; 8 KV heads; head dim 128 | 16 full-attention layers; 4 KV heads; head dim 256; GQA factor 6 | GQA K/V reuse may generalize, but Laguna kernels cannot be pasted in. |
| Position encoding | YaRN partial RoPE on full attention | Qwen partial RoPE, already fused with Q/K RMSNorm/layout | The old YaRN scale trick is not a remaining seam. |
| Vocabulary | 100,352 | 248,320 | Exact target-head screening has a larger possible byte prize in Qwen. |
| Generation | Serial target inference | Adaptive native-MTP speculation, widths 0...8, rollback/replay | Qwen has proposal cost, multi-row exact values, and state rollback absent from Laguna. |
| Timed seed | Separately measured 512-token prefill carried exponent 0.25 in the geometric score | 512-token seed is charged inside every per-prompt decode ratio | Seed-only traffic reductions still count materially. |

## What the final Laguna frontier had already optimized

The table deliberately distinguishes mechanisms from implementation details.
“Qwen status” means whether the same cost is already addressed in the promoted
Qwen source, not whether filenames happen to overlap.

| Laguna mechanism | What it removed | Evidence in the final tree | Qwen status |
| --- | --- | --- | --- |
| Attention weights moved from BF16 through affine INT8 to native group-16 NVFP4 | Large decode-side weight reads | Native-affine QKV/`o_proj` preparation and decode kernels in `LagunaRuntimeModel.swift` | **Already analogous.** Qwen is affine4/g64. Audit layouts/conversions; do not port NVFP4 code. |
| One RMSNorm producer feeding separate QKV and gate consumers | Repeated hidden-state reductions inside output tiles | The earlier giant RMSNorm+QKV+gate fusion was disabled after it duplicated the reduction | **Durable warning.** Any Qwen GDN/attention fusion must retain one producer owner. |
| Fused Q/K RMSNorm, layout transform, and RoPE | Q/K intermediates and dependency stages | Sliding and full-attention fused kernels | **Already analogous.** Qwen has a full-attention Q/K RMSNorm + partial-RoPE + transpose Metal path. |
| Fused attention, cache write/advance, and short-query work | K/V materialization and cache dependency stages | `lagunaSlidingFusedAttention` and `lagunaFullFusedAttention` | **Partly open.** Qwen prepares Q/K jointly but still uses generic cache-update + SDPA. A D256 implementation is substantial. |
| Adjacent GQA query-head K/V reuse | Duplicate K/V row reads while preserving independent accumulators | A pair-head path ships in final `sdpa_vector.h`; wider grouped-SDPA claims remained unvalidated | **Open adaptation.** Qwen's GQA factor 6 is attractive, but D256 raises register pressure. |
| Wider SDPA exchange planes | Repeated large threadgroup rendezvous | One exchange replaced eight exposed barriers in Laguna geometry | **Profile first.** Qwen may dispatch a different SDPA family and has a width-6...9 exactness bridge. |
| M5 NAX Steel-attention partial loop unroll | Poor load/MMA interleaving | `unroll_count(4)` in both source and generated twin | **Check ancestry first.** It may already exist in the vendored MLX version. |
| Exact attention scale-plane compression, pairwise layout, and conversion hoisting | 27,698,336 metadata bytes per Laguna decode step | Narrow/pairwise/lane-major scale banks; later pairwise NAX conversion | **Concept only.** Qwen's g64 metadata and cross-row QMV differ; establish a current byte census before proposing a format. |
| Gate activation owned once by the producer | Softplus recomputed across many `o_proj` consumer groups | Custom affine gate activation consumed by NVFP4 `o_proj` | **Generic lesson.** Search Qwen GDN/attention gates for tiled recomputation; plain full-attention sigmoid fusion has negative public evidence. |
| Exact two-tier LM-head screening | Most full BF16 vocabulary-head reads | Planar INT5 coarse values + conservative bounds + exact BF16 survivor refinement | **Largest unclaimed concept.** Qwen's compact head is proposal-only; target top-two remains full-vocabulary and exact. |
| Fixed-shape BF16 argmax and hidden-2048 RMSNorm | Generic indexing and synchronization at ubiquitous terminal shapes | `argmax_bfloat16_100352`; specialized single-row RMSNorm | **No direct portability.** Qwen already has a custom exact top-two reducer; hidden 5,120 selects a different RMS path. |
| Streaming `asyncEval` ladder | Swift graph-build latency before GPU work begins | Enqueues at selected layer boundaries | **Confirmed transfer.** Qwen's current layer loop explicitly cites the Laguna result and uses a ladder for widths <=2. |
| Exact-shape warmup and residency setup | First-use compilation and driver/cache stalls inside timing | Greedy argmax and runtime shapes warmed before scoring | **Largely transferred.** Qwen warms legal verify widths, replay, seed, and proposal selection. Keep an inventory, not a generic “warm more” task. |
| Fused routed/shared gate-up, SwiGLU, down, weighting, and residual | Huge MoE intermediates and rereads | Custom routed/shared QMV and down-residual kernels | **Model-specific.** Qwen is dense; only residual/activation epilogue principles remain relevant. |
| Prefill RUNSKIP | Gather-GEMM MMA whose outputs were outside a consumer's store band | Exact SIMD-row range elision | **No direct transfer.** Use only as a template for proving work is later discarded. |
| Exact active-64 router tournament and packed ordinal/index exchange | Router scratch, redundant sorts, serialized selector work | Local top-eight from eight groups, one exact 64-finalist sort | **Not applicable.** Qwen has no routed experts. |
| Exact expert-bounds sidecar reuse | Repeated lower-bound searches, bounds scratch, and barriers in gate/up and down GatherQMM | Final `4ea72c3b` source across sorter, Swift dispatch, NAX header/twin | **Instance irrelevant.** The reusable idea is to carry exact metadata to multiple consumers rather than recompute it. |
| Pairwise/shared scale layouts and certified zero-stride views | Repeated scale conversion and copies in routed prefill | Pairwise NAX loaders and zero-copy scale exposure | **Concept only.** Qwen needs a g64-specific proof and byte count. |
| Residual/normalization and projection epilogue fusions | Hidden-size intermediates and rereads | Multiple Laguna runtime-local fused boundaries | **Open adaptation.** Qwen's decoder still materializes `h = x + r`, normalizes it, then materializes `h + mlp(...)`. |

## What Qwen has already optimized

Do not assign these as fresh Laguna transfers. The promoted Qwen source already
contains Qwen-native versions or stronger mechanisms:

- **Laguna-derived host/GPU overlap.** The layer loop in
  [`Qwen35.swift`](../Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift)
  runs an `asyncEval` ladder for serial and width-two forwards.
- **Quantized weight sharing across verification rows.** The affine4/g64 QMV
  kernels share weight/code loads across rows 2...9 while preserving the exact
  per-row arithmetic order; source and `mlx-generated/quantized.cpp` twin are
  synchronized.
- **Projection and preprocessing fusion.** Qwen has packed GDN projections,
  compiled pre/post expressions, checkpoint/replay support, fused dense MLP
  gate/up for short widths, fused full-attention QKV, and fused Q/K
  RMSNorm/layout/partial-RoPE.
- **Adaptive speculation, not a fixed K=1 policy.** The session constructor
  installs a measured cost-model schedule over 0...8 drafts, per-position
  acceptance EMAs, a width-five exactness wall, and a guarded deeper SDPA
  bridge. Some nearby comments still describe historical defaults.
- **No repair target forward after an ordinary rejection.** Per-row recurrent
  checkpoints, exact prefix restoration, and lazy replay handle GDN state.
- **Device-resident proposal and one batched target verification.** Draft IDs
  stay on device and the target verifies `[primary] + drafts` together.
- **Exact two-stage top-two reduction over all 248,320 target logits.** The
  result is reused as verifier argmax; redundant target argmax work is gone.
- **Candidate-only compact vocabulary.** Proposal selection uses a 98,304-row
  prefix plus required control tokens with an on-device mapping. This does not
  prune the exact target head.
- **A declared 4-bit/g64 MTP head.** [`mtp-head.manifest.json`](../mtp-head.manifest.json)
  pins a 238,934,093-byte remote artifact rather than the older roughly
  849-MiB head.
- **Persistent committed MTP history.** Lazy seed priming, accepted-row
  backlog, K/V-only history flush, and draft chaining avoid recomputing dead
  leading-row head work.
- **Scored-shape warmups.** Legal verify widths, proposal selection, long-KV,
  recurrent replay, and the 512-row seed shapes have explicit warm paths.
- **Decode mask construction is mostly absent.** Recurrent layers receive no
  attention mask and full-attention layers use a symbolic causal mode, so the
  Laguna/Cedar “skip both mask builders” win is not an obvious seam here.

All runtime-effective generated Metal twins currently pass
`python3 research/twin_audit.py`. Preserve header/generated pairs for every JIT
kernel edit.

## Ranked remaining opportunities

Expected payoff below is qualitative unless a receipt is cited. The falsifier
is part of the experiment: stop when the mechanism misses it rather than
keeping a large speculative patch alive.

| Rank | Experiment | Why it remains | Effort / risk | Decisive test |
| ---: | --- | --- | --- | --- |
| 0 | **Adopt the rebased 2.903545 stack if it promotes** | Public receipt `9ff11c5` passed all eight prompt checks and exceeded the frontier, but promotion failed on stale harness ancestry. Byte-identical rebases `e6c5ef3` and `1c2787b` were validating at this snapshot. | Low adoption risk after source review; mixed causal attribution | Require a promoted rebased receipt, compare its exact editable snapshot with current frontier, then import it as one unit before opening overlapping work. |
| 1 | **Exact-fill first `KVCacheSimple.update`** | Current code allocates zero buffers and slice-copies a first update even when 512 tokens exactly fill two 256-token steps. | Very low / low | Bitwise compare returned K/V, offset, next-growth behavior, snapshots, and rollback after 512 + 1...9 tokens. Measure seed/window time on all eight prompts. |
| 2 | **Packed GDN prework for S=1...2** | The packed mixer covers S=3...9; serial, adaptive-skip, and narrow verify calls retain separate conv/QKV/g paths. | Medium / medium | Reproduce every output and cache byte at S=1,2. Kill if a target call saves less than about 0.25 ms or end-to-end movement is noise. |
| 3 | **Complete packed GDN `beta` with one certified BF16 exception** | Existing comments identify one finite BF16 input where fused sigmoid rounds differently; the excluded beta step costs a small but repeated dispatch/intermediate. | Low-medium / medium | Exhaust all 65,536 BF16 encodings, including NaNs/infinities, with an explicit correction for `0xC0DB`; require exact output/cache values. |
| 4 | **Generic `rms_single_row` input reuse** | Current kernel rereads each input for the output pass; Laguna caches the already promoted floats in registers. | Low / low-medium | Port only the generic hunk, rebuild `mlx.metallib`, run raw-array equality and official-path tests. Inspect occupancy; stop if affected calls are not material. |
| 5 | **Residual-add plus following RMSNorm** | Qwen materializes `h = x + r`, rereads it for post-attention RMSNorm, then repeats the pattern after MLP. Laguna proved the class can win. | Medium / medium-high | First emit both exact BF16 residual and normalized output for hidden 5,120 at S=1...9 and 512. Preserve stock `rms_looped` reduction/cast order; require exact hidden, top-two values, and caches. |
| 6 | **Certified target LM-head screening** | Qwen still computes every exact target logit before reducing top two; its larger vocabulary makes avoided rows valuable. | High / high | Build a conservative coarse plane and per-row bound; exact-evaluate every possible top-two survivor. Stop if p90/p99 survivor density makes coarse+tail bytes approach stock affine4 reads or any ID/value differs. |
| 7 | **Pair GQA query heads to reuse K/V reads** | Laguna's shipped pair-head path attacked a real duplicate-byte seam; Qwen has six query heads per KV head. Its causal benefit was not isolated in the final campaign evidence. | High / high | Reimplement for D256 with independent accumulation order. Compile/occupancy gate first; a recent wider grouped-SDPA attempt failed pipeline creation from register pressure. |
| 8 | **Fuse full-attention K/V append with short-query SDPA** | Q/K preparation is fused, but new K/V are written through generic cache update then reread by SDPA. | Very high / high | Preserve unbounded cache semantics and the qL=6...9 split that protects exact values. Require compile/resource evidence before full correctness work. |
| 9 | **Dense affine4 MLP down projection plus residual epilogue** | SiLU/product can be fused cheaply, but `downProj` output and residual remain a hidden-size intermediate in every layer. | High / medium-high | A custom editable kernel must reproduce the current QMV arithmetic and BF16 cast-before-add boundary. Try residual+RMS first; stop if the epilogue changes exact top-two values. |
| 10 | **Current affine4 scale/code byte census** | Laguna's largest late win came from scale-plane bytes, but Qwen's format and cross-row sharing are different and may already be near the floor. | Low audit / high implementation | Trace runtime-effective layouts and count unique code/scale bytes by shape. Open a format experiment only with a concrete removable-byte construction. |

### Pending public stack: overlap warning

At this snapshot, public receipt
`9ff11c51-5c82-4f67-86de-bc23fd61f786` at commit
`6e5c10d50709f317c169c640ee20d76bd5e8b7bf` reported
`2.90354559365115`, all eight prompts exact, and “promotion failed” because the
trusted branch changed outside `editablePaths` during validation. Byte-identical
rebases `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd` and
`1c2787b0-600b-4be2-8cd7-a8e25f0fd249` were validating on organizer commit
`26ae2bf` when observed. Its public source description contains:

1. decode `asyncEval` ladder widened from S<=2 through S<=9;
2. a denser ladder for the charged 512-row seed;
3. fused Q/K preparation restricted to L<=16 so the seed uses the faster stock
   path while short verify widths retain the fused path;
4. a target top-one/top-two-margin cap on the first-position acceptance prior;
5. shape-specialized compiled dense-MLP SiLU/product.

Do not independently implement these while a rebased copy is validating. The
generic shapeless `compiledSiluProduct` helper already exists in
[`SwitchLayers.swift`](../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift),
but the public composition used a shape-specialized closure after a shapeless
half-slice could not infer its output shape. If the stack does not promote,
isolate mechanisms 1, 2, and 5 first; mechanisms 3 and 4 had mixed standalone
evidence and should not inherit the composite score.

## Two first experiment cards

### A. Exact-fill KV-cache retention

Target:
[`KVCacheSimple.update`](../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift)

The final Laguna code adds one guarded main path before allocation:

```swift
if self.keys == nil, previous == 0, tokenCount > 0,
    tokenCount.isMultiple(of: step)
{
    self.keys = keys
    self.values = values
    self.offset = tokenCount
    return (keys, values)
}
```

Why it is exact: in this case the stock allocation has no slack. It creates
arrays with exactly the incoming sequence length, zero-fills them, then replaces
the complete slices with the same inputs. Retaining those arrays preserves
shape, visible values, offset, and the next growth boundary.

Required tests:

1. First updates of 0, 1, 255, 256, 511, 512, and 513 tokens.
2. Exact-boundary 512 followed by each width 1...9.
3. Snapshot/restore and MTP prefix rollback after direct retention.
4. Contiguity/layout inspection before the following attention call.
5. Same-host baseline/candidate runs with seed and steady phases reported
   separately.

### B. Generic single-row RMSNorm input reuse

Target:
[`rms_norm.metal`](../Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/rms_norm.metal)

Cache each `float xi` loaded for the sum in `float xcache[N_READS]`, then use
`xcache[i]` in the output loop instead of rereading `x[i]`. This preserves the
square order, SIMD reductions, precise reciprocal square root, cast point,
weight multiply, and barriers. It is the generic part of the final Laguna hunk,
not Laguna's separate hidden-2048 specialization.

Required tests:

1. Rebuild the AOT library with `tools/build-mlx-metallib.sh`.
2. Compare raw outputs bit-for-bit for BF16/FP16/FP32, weighted/unweighted,
   tail and non-tail shapes.
3. Confirm which Qwen calls actually select `rms_single_row`; hidden 5,120 is
   expected to use `rms_looped`.
4. Inspect register count/occupancy and measure only after proving reachability.
5. Treat a looped input cache or residual-plus-RMS kernel as a separate
   experiment; do not expand this easy patch mid-measurement.

## Closed, negative, or misleading transfer lines

- **Do not copy Laguna MoE machinery.** Router tournaments, packed route
  results, expert bounds, GatherQMM, RUNSKIP, routed/shared fusion, and expert
  scale planes have no Qwen consumer.
- **Do not reopen broad RMSNorm-to-QKV fusion.** Laguna's unbiased result was
  effectively null, and the larger all-consumer fusion was disabled because it
  recomputed the same norm reduction in every output tile.
- **Do not count launches or barriers as savings.** Mature Laguna measurements
  put an added dispatch near 0.45 microseconds on M4 and barrier-only changes
  near zero. A useful fusion removes bytes, arithmetic, or a dependency edge.
- **Do not assume an isolated kernel ratio survives Amdahl's law.** Several
  Cedar experiments won by 2...23% in a microbenchmark and moved the whole
  model by less than 0.1% or regressed it.
- **Do not use M4 NAX-prefill timing as M5 evidence.** Path selection and wave
  quantization changed signs between machines.
- **Do not port Laguna's fixed vocabulary or hidden-width kernels.** Qwen's
  exact top-two reducer already supersedes fixed argmax, and hidden 5,120 uses
  different RMS geometry.
- **Do not copy the YaRN sentinel or rotating-cache accessors.** Qwen has
  different RoPE semantics and recurrent state; its full-attention cache is not
  Laguna's 512-token sliding ring.
- **Do not assume full-attention sigmoid fusion wins.** A current-Qwen public
  attempt was about 0.42% slower. Reopen only with a different byte/ownership
  mechanism, not a compiled wrapper around the same work.
- **Do not widen grouped SDPA without a resource gate.** A recent Qwen attempt
  failed M5 pipeline creation from register pressure.
- **Do not shrink the proposal head blindly.** A two-bit proposal head was
  roughly 14% worse, and reducing the compact proposal vocabulary to 49,152
  lowered acceptance and increased latency.
- **Do not dismantle Qwen's rollback machinery.** Recurrent snapshot deletion,
  target-hidden reuse, and several compiled MTP-head chains have recent negative
  receipts.
- **Do not attribute the public 2.903545 composite to every component.** It is
  a combined result and has not yet become the promoted source authority.

## Research discipline for transfers

1. Start from the current campaign `BASE_SHA`, not a downloaded Laguna file.
2. Name one causal mechanism and the exact Qwen cost center it removes.
3. Preserve Qwen's arithmetic order, BF16 cast boundaries, top-two values,
   cache state, and generated Metal twin.
4. Run source-level reachability and byte/resource analysis before spending a
   ranked receipt on an M5-only hypothesis.
5. Compare a fresh same-host unchanged baseline and candidate with seed versus
   steady time separated.
6. Use all eight prompt results and the 3.0 ceiling when deciding whether to
   compose independent wins.
7. Record negative results in the campaign novelty ledger so future agents do
   not rediscover Laguna-shaped dead ends.

## Source index

- Current Qwen model:
  [`Qwen35.swift`](../Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift)
- Current speculative session:
  [`Qwen36MTPBlockSession.swift`](../Sources/MLXFastModel/Qwen36MTPBlockSession.swift)
- Current KV cache:
  [`KVCache.swift`](../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift)
- Current attention/cache dispatch:
  [`AttentionUtils.swift`](../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift)
- Current generic RMSNorm Metal:
  [`rms_norm.metal`](../Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/rms_norm.metal)
- Current affine quantized Metal:
  [`quantized.h`](../Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h)
- Current declared MTP head:
  [`mtp-head.manifest.json`](../mtp-head.manifest.json)
- Qwen editable-surface contract:
  [`qwen-mtp-editable-surface.md`](../docs/qwen-mtp-editable-surface.md)
- Final Laguna fork and briefing:
  [`morganmcg1/mlxfast-challenge_senpai`](https://github.com/morganmcg1/mlxfast-challenge_senpai/),
  [research-frontier briefing](https://github.com/morganmcg1/mlxfast-challenge_senpai/blob/29250d98d2886dbc11040a466ef758ff490f2ae3/senpai/research-frontier-briefing.md)
