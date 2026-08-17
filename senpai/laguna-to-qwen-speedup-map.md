# Laguna XS to Qwen 3.8 speedup transfer audit

**Research snapshot:** 2026-08-16 10:41 UTC
**Frontier refresh:** 2026-08-16 12:05 UTC
**Ceiling and frontier correction:** 2026-08-17 (advisor). The transfer
arguments below are unchanged and still stand on their own evidence; only the
frontier endpoints and the plausibility ceiling were stale. Corrected in three
places: the headroom paragraph in the executive conclusion, the authority table
under "Reproducible endpoints", and item 6 of the closing checklist. Ceiling is
`5.0`, not 3.0; frontier is `2.95338624520432`, not `2.9042110287045`.

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
2. **Extend the packed Gated DeltaNet prework kernel to widths 1 and 2.** The
   current fused mixer covers 3...9, leaving the common serial/narrow cases on
   a less fused path. This is Qwen-specific rather than a literal Laguna port,
   but it applies Laguna's most durable lesson: eliminate a producer-consumer
   intermediate, not merely a launch.
3. **Isolate the promoted composite before extending it.** Source
   `7351e626` combines a top-two-margin draft prior, wider decode/seed
   `asyncEval`, a short-width Q/K-prep guard, and compiled dense-MLP
   SiLU/product. The composite is correctness-green and promoted, but no
   component inherits its score independently.
4. **Try the generic RMSNorm input-cache hunk only after proving reachability.** It is a direct, exact Laguna
   carryover, but the easy hunk accelerates `rms_single_row`; Qwen's main
   hidden-5120 norms use `rms_looped`, so expected whole-model impact is limited
   unless a separate looped or residual-plus-RMS design proves safe.
5. **Treat certified target LM-head screening as the largest research bet.**
   Laguna avoided most reads of a 100,352-row BF16 head. Qwen still evaluates
   the exact 248,320-row target head for every verification row; its compact
   vocabulary is proposal-only. A Qwen certificate could be valuable, but it
   must preserve both exact top-two IDs and values.

**CORRECTED 2026-08-17 (advisor).** The paragraph that stood here was written
against a 3.0 ceiling and told the reader that only about 3.3% multiplicative
headroom remained, so wins should not be stacked. Both halves are now wrong and
the prescription was the dangerous half: it would talk a reader out of
composing legitimate wins.

The promoted Qwen score is `2.95338624520432` (receipt
`ba493f74-c0fe-440a-a956-f77d26232e54`, source
`156b5b75bdfac82ae406487f531fd991e7fdfd30`) and the plausibility ceiling is
`5.0`, raised from 3.0 by operator commit `a5854b979499800a6f5f71a8d4fc14fd43ca4723`.
Headroom is therefore `+2.047` of score, about 69% multiplicative, not 3.3%.
No measured lever in this campaign is within an order of magnitude of
exhausting it. **Stack every legitimate win you can attribute.** The ceiling is
a fail-closed plausibility gate, not an optimization target and not a reason to
stop; see `senpai/program.md:19-21` and `AGENTS.md:75`.

What survives from the original paragraph is only the measurement hygiene:
prefer small, attributable candidates and inspect all eight prompt results
before composing, because the score is a median of eight and improving the
worst prompt moves nothing.

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

### Qwen authority as of the 2026-08-16 12:05 UTC refresh (SUPERSEDED)

**Do not use this table as live authority.** Every frontier row below was
correct at the refresh timestamp and is now stale. The live values are in the
table that follows it. Kept verbatim because the report's transfer arguments
were written against these endpoints.

| Item | Value |
| --- | --- |
| Campaign solver-import commit | `ce159755215b60c8f582f3b4402ddf483083d990` |
| Organizer main at refresh | `7351e62674bc600f0ca148d3a1b0604716a09db6` |
| Trusted organizer policy parent | `26ae2bf6326de93e7f1b1b0aaf94a7667aca797b` |
| Promoted editable snapshot | `7351e62674bc600f0ca148d3a1b0604716a09db6` |
| Promoted receipt | `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd` |
| Official score | `2.9042110287045` |
| Score | median of eight per-prompt serial-relative speedups; floor 0.90; ceiling 3.0 |

Note that `7351e626...` is a submission `sourceRef`, not a commit reachable in
this repository; it arrived with the solver import `ce159755`.

### Current Qwen authority (2026-08-17, advisor)

| Item | Value |
| --- | --- |
| Promoted receipt | `ba493f74-c0fe-440a-a956-f77d26232e54` |
| Promoted source | `156b5b75bdfac82ae406487f531fd991e7fdfd30` |
| Official score | `2.95338624520432` |
| Trusted organizer contract | `0b071ed9db211f17554bc5a13fb7381f14d709b3` |
| Score | median of eight per-prompt serial-relative speedups; floor 0.90; ceiling `5.0` |
| Ceiling provenance | raised 3.0 -> `5.0` by `a5854b979499800a6f5f71a8d4fc14fd43ca4723` |

Two different bounds exist and they are not the same number: the
**published-median** ceiling is `5.0`, while the box wrapper's **per-pair**
`MAX_PLAUSIBLE_SPEEDUP` is `8.0`, so the aggregate ceiling stays strictly
tighter than the per-pair bound (`benchmark.json:201`).

The campaign import overlays the promoted editable snapshot onto the reviewed
organizer contract. Its submitted delta from the previous `df404e08` snapshot
is exactly `Qwen36MTPBlockSession.swift` and `Qwen35.swift` (`+54/-14`).
[`frontier-state.json`](frontier-state.json) and the live Yukon receipt are the
operational authorities; this report is a timestamped comparison and can
become stale quickly.

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
  runs an `asyncEval` ladder through decode/verify widths 1...9 and a denser
  ladder for the charged 512-row seed.
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
- **Promoted confidence and dense-MLP composite.** The first draft probability
  is capped by the pending target top-two margin, fused Q/K preparation is
  limited to short lengths, and packed dense-MLP SiLU/product runs as a
  compiled expression. These landed together; their individual signs remain
  unresolved.
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
| 0 | **Exact-fill first `KVCacheSimple.update`** | Current code allocates zero buffers and slice-copies a first update even when 512 tokens exactly fill two 256-token steps. | Very low / low | Bitwise compare returned K/V, offset, next-growth behavior, snapshots, and rollback after 512 + 1...9 tokens. Measure seed/window time on all eight prompts. |
| 1 | **Packed GDN prework for S=1...2** | The packed mixer covers S=3...9; serial, adaptive-skip, and narrow verify calls retain separate conv/QKV/g paths. | Medium / medium | Reproduce every output and cache byte at S=1,2. Kill if a target call saves less than about 0.25 ms or end-to-end movement is noise. |
| 2 | **Complete packed GDN `beta` with one certified BF16 exception** | Existing comments identify one finite BF16 input where fused sigmoid rounds differently; the excluded beta step costs a small but repeated dispatch/intermediate. | Low-medium / medium | Exhaust all 65,536 BF16 encodings, including NaNs/infinities, with an explicit correction for `0xC0DB`; require exact output/cache values. |
| 3 | **Isolate compiled fused-SwiGLU** | It is promoted inside the four-hunk `7351e626` composite, but its independent contribution is unknown. | Low / low-medium | Run a matched on/off ablation on `7351e626`; require movement outside matched noise or direct structural evidence before extending it. |
| 4 | **Generic `rms_single_row` input reuse** | Current kernel rereads each input for the output pass; Laguna caches the already promoted floats in registers, but Qwen hidden width 5,120 may dispatch `rms_looped` instead. | Low / low-medium | Prove scored-path reachability first; only then port the generic hunk, rebuild `mlx.metallib`, compare raw arrays, and inspect occupancy. |
| 5 | **Residual-add plus following RMSNorm** | Qwen materializes `h = x + r`, rereads it for post-attention RMSNorm, then repeats the pattern after MLP. Laguna proved the class can win. | Medium / medium-high | First emit both exact BF16 residual and normalized output for hidden 5,120 at S=1...9 and 512. Preserve stock `rms_looped` reduction/cast order; require exact hidden, top-two values, and caches. |
| 6 | **Certified target LM-head screening** | Qwen still computes every exact target logit before reducing top two; its larger vocabulary makes avoided rows valuable. | High / high | Build a conservative coarse plane and per-row bound; exact-evaluate every possible top-two survivor. Stop if p90/p99 survivor density makes coarse+tail bytes approach stock affine4 reads or any ID/value differs. |
| 7 | **Pair GQA query heads to reuse K/V reads** | Laguna's shipped pair-head path attacked a real duplicate-byte seam; Qwen has six query heads per KV head. Its causal benefit was not isolated in the final campaign evidence. | High / high | Reimplement for D256 with independent accumulation order. Compile/occupancy gate first; wider grouped-SDPA work has failed pipeline creation from register pressure. |
| 8 | **Fuse full-attention K/V append with short-query SDPA** | Q/K preparation is fused, but new K/V are written through generic cache update then reread by SDPA. | Very high / high | Preserve unbounded cache semantics and the qL=6...9 split that protects exact values. Require compile/resource evidence before full correctness work. |
| 9 | **Dense affine4 MLP down projection plus residual epilogue** | SiLU/product can be fused cheaply, but `downProj` output and residual remain a hidden-size intermediate in every layer. | High / medium-high | A custom editable kernel must reproduce the current QMV arithmetic and BF16 cast-before-add boundary. Try residual+RMS first; stop if the epilogue changes exact top-two values. |
| 10 | **Current affine4 scale/code byte census** | Laguna's largest late win came from scale-plane bytes, but Qwen's format and cross-row sharing are different and may already be near the floor. | Low audit / high implementation | Trace runtime-effective layouts and count unique code/scale bytes by shape. Open a format experiment only with a concrete removable-byte construction. |

### Promoted composite: attribution remains unresolved

Receipt `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd` promoted source
`7351e62674bc600f0ca148d3a1b0604716a09db6` at
`2.9042110287045`. Relative to the preceding `df404e08` frontier score, the
official increase is `0.0277816210771`, or `0.965837%`. The exact source delta
contains four conceptual hunks:

1. a target top-two-margin cap on the first-position acceptance prior;
2. shape-specialized compiled dense-MLP SiLU/product;
3. fused Q/K preparation restricted to L<=16; and
4. the decode ladder widened through S<=9 plus a denser charged-seed ladder.

The result establishes the complete composition, not four individual wins.
Do not extend or recombine one component without a matched ablation or a direct
structural cost proof. The generic shapeless `compiledSiluProduct` helper
already exists in
[`SwitchLayers.swift`](../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift),
but the promoted composition uses a shape-specialized closure because a
shapeless half-slice could not infer its output shape.

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
- **Do not attribute the promoted 2.904211 composite to every component.** It
  is authoritative only as a combined result; no component is independently
  established by that receipt.

## Research discipline for transfers

1. Start from the current campaign `BASE_SHA`, not a downloaded Laguna file.
2. Name one causal mechanism and the exact Qwen cost center it removes.
3. Preserve Qwen's arithmetic order, BF16 cast boundaries, top-two values,
   cache state, and generated Metal twin.
4. Run source-level reachability and byte/resource analysis before spending a
   ranked receipt on an M5-only hypothesis.
5. Compare a fresh same-host unchanged baseline and candidate with seed versus
   steady time separated.
6. Use all eight prompt results when deciding whether to compose independent
   wins. The ceiling is `5.0`, not 3.0, and with the frontier at
   `2.95338624520432` it is far enough away that it should not enter the
   composition decision at all.
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
