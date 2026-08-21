# E93 — per-draft proposal-head dispatch census

Student `qwen-askeladd`. PR #95. Assignment `qwen38-r1-e93-per-draft-proposal-head-dispatch-census` r1.
Base `b81a43d47f661cb4279d013ad7395c85b0fcb00a`. `harness=local` throughout.

## Identity tuple

| field | value |
|---|---|
| base | `b81a43d47f661cb4279d013ad7395c85b0fcb00a` |
| census worker | `worker_sha256 e3fedbc8d72ef85aeb4ccee1388281250d550113e476a853b634999097d385f1` |
| instrument | E58/E85 dispatch census, re-seated; `research/e93-artifacts/e93-census-instrument.patch` |
| host | Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 26.5.2, Swift 6.3.3 |
| head | declared, `head_provenance_sha256 dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` |
| fixture | public local fixture, `--local-iterate` |
| token window | 24 to 384 decode tokens, forced draft depth 2 or 8 |
| timing basis | Metal command-buffer clock, `gpuStartTime`/`gpuEndTime`, `MLX_E80_GPU_TIME=1` |
| gate flags | `timing_valid=false`, `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false` |

The branch changes `research/` only. `git diff base HEAD -- Sources Vendor mtp-head.manifest.json Package.swift` is empty, so the scored surface is untouched and the temporary census instrument commit is reverted.

## Report shape

**Question.** Where does the per-draft proposal-head time go, dispatch by dispatch, so the next head experiment is priced from a census instead of a byte model?

**Evidence that made it worth testing.** Dividing 427,738,112 head bytes by a 2,291 µs head pass gives 186.7 GB/s, which is 70 % of the corrected 265 GB/s achievable rate. That leaves 677 to 767 µs per draft unexplained. Arm G died because achieved bandwidth fell when bytes were removed, which a byte model cannot predict.

**Expected result.** A class-by-class split with stated closure, and a named home for the gap.

**Smallest decisive test.** A dispatch census with GPU-clock attribution over the shipped path with the declared head attached. No new arm is built.

**Stop or promotion rule.** Rung 4 names exactly one arm, priced from the census.

## Rung 1 — the dispatch census

### Model

Exact over **112 modelled rounds** in the five default-buffer legs, **zero residual**:

```
head dispatches per round = F(previous accepted drafts) + 27 * (drafts - 1)
F(0) = 25, F(1) = 33, F(>= 2) = 37
+6 on the one round where the head KV cache capacity grows
```

Widths 2, 3, 5 and 9 all fit. The marginal draft step is exactly 27 dispatches. The first head call flushes every token committed since the previous head call and saturates at two rows. Mean per draft at d=8 is 28.25 against E85's 28.0, so closure is 99.1 %.

### The 27 marginal dispatches

Class 1, weight-streaming GEMV, 7 dispatches, 264,437,760 B, split by tensor:

| dispatch | tensor | weight B |
|---|---|---|
| `affine_qmv_fast b_4 grid=1x1536x1` | `q_proj` | 35,389,440 |
| `affine_qmv_fast b_4 grid=1x4352x1` | mlp fused gate+up | 100,270,080 |
| `affine_qmv_fast b_4 grid=1x640x1` | `down_proj` | 50,135,040 |
| `affine_qmv_fast b_4 grid=1x640x1` | `fc` | 29,491,200 |
| `affine_qmv_fast b_4 grid=1x640x1` | `o_proj` | 17,694,720 |
| `gemv_al grid=128x1x1` | K/V exact dense bf16 `[2048,5120]` | 20,971,520 |
| `gemv_al grid=64x1x1` | Q island dense bf16 `[1024,5120]` | 10,485,760 |

`k_proj` and `v_proj` affine-4 never appear. E84 dead-work elimination is confirmed live.

Class 2, attention over head history, 6 dispatches: qk_rms_rope ×1, `gg2_copy` 1-position K and V append ×2, `sdpa_vector` ×1, **`vn_copy` full-array head KV cache copy ×2** (6,291,456 B).

Class 3, norms and elementwise, 8 dispatches, 54,080 B weight.

Class 4, readout and rerank, 5 dispatches, 157,429,760 B, dominated by `draft_lm_head` affine-2 `[98336,5120]` at 157,337,600 B.

Class 5, island scatter, 1 dispatch: `replaceExactRows` `putAlong`, 24,576 B written.

### Byte model per marginal draft

Declared-head weight actually read is **421,831,680 B**, which is 1.38 % below the 427,738,112 B artifact figure. The path skips affine-4 `k_proj` and `v_proj` (5,898,240 B) and the separate K/V island tensors (20,979,712 B), and reads one dense bf16 `[2048,5120]` (20,971,520 B) instead. Target-owned weight of 95,040 B is read as well and is not charged to the head. Activation traffic is 9,035,904 B, of which the full-array cache copy is 6,291,456 B, or 69.6 %.

## Rung 2 — GPU busy per class

Instrument validation: in-situ head pass 2,267.5 µs and 2,276.8 µs against E85's 2,285.283 µs, both inside the 2,261 to 2,381 band, closure 99.2 % and 99.6 %, replicate spread 0.41 %. `mixed_phase_buffers = 0` everywhere, so attribution is exact.

Host-state stratum, 1,500 µs absolute gate: **no `draft_head` round is dirty in any leg**, maximum 473.6 µs. Four legs have small clean-round counts of 2, 7, 11 and 15; every headline comes from the two long legs at 32/32 and 52/52.

A marginal draft is three command buffers: embed, a 25-dispatch body, and rerank. The marginal draft step at capacity 768 is **2,226.5 µs**, reproduced at 2,226.3 µs in the second leg. The first head call is about 2,560 µs.

### Class rollup per marginal draft

| class | dispatches | µs/draft | share | traffic | GB/s |
|---|---|---|---|---|---|
| 1 weight-streaming GEMV | 7 | ~1,100 | 49.4 % | 264.44 MB | 240 |
| 4 readout and rerank | 5 | 1,027.3 | 46.1 % | 157.63 MB | 153 |
| 2 attention over head history | 6 | ~68 | 3.1 % | 8.43 MB | — |
| 3 norms and elementwise | 8 | ~31 | 1.4 % | 0.40 MB | — |
| 5 island scatter | 1 | ~2 | 0.1 % | 0.02 MB | — |
| total | 27 | 2,229 | | | 189.5 |

Closure against the measured 2,226.5 µs is 99.9 %.

## Rung 3 — achieved bandwidth of the weight-streaming class

| tensor | µs/draft | weight B | GB/s | % of best within-host rate |
|---|---|---|---|---|
| K/V exact + Q island (`gemv_al` pair) | 127.78 | 31,457,280 | **246.2** | 100 % |
| mlp fused gate+up | 416.30 | 100,270,080 | 240.9 | 98 % |
| `down_proj` | 211.83 | 50,135,040 | 236.7 | 96 % |
| `q_proj` | 154.07 | 35,389,440 | 229.7 | 93 % |
| `fc` | 128.73 | 29,491,200 | 229.1 | 93 % |
| `o_proj` | 77.26 | 17,694,720 | 229.0 | 93 % |
| **class 1 aggregate** | ~1,100 | 264,437,760 | **240** | 97 % |

**H5 is false for class 1.** No class-1 tensor is more than 7 % below the fastest rate this host reaches anywhere in the census. The 29.5 MB `fc` read, which edward's curve does not sample, is at 229.1 GB/s, exactly in line with `q_proj` at 35.4 MB and `o_proj` at 17.7 MB. There is no small-tensor anomaly.

Against edward's cross-host size-matched curve of 265 to 276 GB/s the class sits at 87 to 91 %, but that comparison crosses hosts: his curve is from a different Mac. The within-host comparison is the sound one and it says the GEMVs are saturated.

**The gap lives in class 4.** `draft_lm_head` affine-2 readout runs at **158.2 GB/s**, which is 64 % of the best within-host rate and 57 % of the size-matched 274 to 276 GB/s at 157 MB. Matching the class-1 rate of 240 GB/s would cost 656 µs instead of 994.81 µs, so the headroom is **334 µs per draft, 15 % of the head pass**.

Mechanism: affine-2 packs 16 weights per `uint32` against affine-4's 8, so the readout performs twice the dequantisation work per byte. It reaches 636 GMAC/s against b_4's ~480 GMAC/s. The readout is dequantisation-bound, not bandwidth-bound. Dispatch geometry is structurally identical to `q_proj`: 8 outputs per 64-thread threadgroup.

**§4.2 answer.** The `replaceExactRows` scatter costs about 2 µs per draft, 0.09 % of the head pass. It is pure dispatch cost. It does not justify an arm.

## The main new finding — the head copies its whole KV cache twice per draft

Every marginal draft issues two `vn_copy` dispatches over the head's **entire** K and V cache arrays, sized at cache **capacity**, not at used length. `vn_copy` is MLX `CopyType::Vector`, `work_per_thread = 4` for bf16, so grid 196,608 is 786,432 elements, which is `[1,4,768,256]` = 1,572,864 B per array.

Three independent proofs that the work is dead:

1. **Scaling.** Capacity 768 → 1024 moves the grid 196,608 → 262,144 and the allocation 1,572,864 → 2,097,152 B.
2. **Dispatch-count natural experiment.** With `MLX_E58_BUFFER_LIMIT_OPS=1` the dispatch-model residual is exactly **−14 per round**, which is 2 copies × 7 marginal drafts, while `gg2_copy` and `sdpa_vector` run at full rate and `all_tokens_matched` stays `true`. Forcing one operation per command buffer removes the copy and nothing else.
3. **The first head call has no copy.** The deficit is 14, not 16, so only marginal calls pay it.

Mechanism: `KVCacheSimple.update` at `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:398` runs `self.keys?[.ellipsis, previous..<offset, 0...] = keys`, a `slice_update`. MLX donates the destination buffer only when nothing else references it. Under lazy evaluation the graph still holds the previous cache array, so donation fails and MLX materialises a full capacity-sized copy before writing one position. Forcing one operation per buffer evaluates eagerly, the reference is released, and donation succeeds. The target verify phase shows no `vn_copy` even though the target full-attention layers also use `KVCacheSimple`, so the failure is specific to the head's reference pattern. `KVCache.swift` is inside `editablePaths`.

Cost, two independent estimates:

1. Cross geometry: in-situ 2,226.5 µs minus isolated 2,199.4 µs = **26.9 µs**.
2. Within one leg, drift immune: the capacity step raises the body buffer by 8.79 µs, minus 0.35 µs of trend, so **8.44 µs for Δ2,097,152 B**, which is 248 GB/s marginal and **25.4 µs** at capacity 768.

Reported value **≈26 µs per marginal draft at capacity 768**, range 25.4 to 26.9.

Rejected confound: the pooled regression `body_us = 2174.28 + 0.34760*round + 10546.76 ns/byte` implies 94.8 GB/s and 33.18 µs. The single capacity step is collinear with the smooth per-round trend, so that fit over-attributes.

## Rung 4 — one arm, priced

### The pricing model, and its calibration against a ranked measurement

For a bandwidth-bound saving I use the head-share rule:

```
ranked share of round time saved
  = (removed us per head call * head calls per round) / (head us per round)
  * ranked head share (6.3 %)

head us per round = 2,560 + 2,226.5 * (drafts - 1)
drafts = mean verify width - 1
```

The first head call carries no full-array copy, so the copy is charged only to marginal calls, while a weight-row deletion is charged to every call.

**Calibration.** Apply the model to ox-alpha's Q-row shrink, using my own measured `q_proj` rate of 229.7 GB/s rather than the 0.0815 byte coefficient. 2,949,120 B removed is 12.84 µs per head call.

| prompt | mean verify width | modelled ranked gain |
|---|---|---|
| beagle | 5.38 | 0.0351 % |
| essays | 6.09 | 0.0353 % |
| **mean over beagle and essays** | | **0.0352 %** |

Your measured ranked increment on beagle and essays is 0.035 %, and the serial-free gap between the two submissions is 0.0357 %. **The model agrees with a ranked measurement to 1.4 %.** The flat byte law would have said 0.0562 %, so the size-matched rule you introduced is confirmed and the flat coefficient over-predicts this tensor by 1.6×.

### The proposed arm — remove the head KV cache full-array copy

- **Class and dispatches.** Class 2. The two `vn_copy grid=196608x1x1` dispatches per marginal draft. Nothing else.
- **Measured cost removed.** 26.0 µs per marginal draft at capacity 768, 25.4 to 26.9 range. The copy is sized by capacity, so on the ranked 512-seed plus 512-decode leg the capacity is 768 for the first 256 decode tokens and 1024 for the next 256. The time-weighted mean is **30.3 µs**.
- **Transfer rule.** Head-share rule, not the 2.1× fixed-cost rule. The copy is a DRAM read and write measured at 248 GB/s, so it scales with the host memory system exactly like the rest of the head pass. Using the 2.1× rule here would over-claim.
- **Ranked price on the two prompts that set the median:**

| prompt | mean verify width | marginal head calls | modelled ranked gain |
|---|---|---|---|
| beagle | 5.38 | 3.38 | **0.0640 %** |
| essays | 6.09 | 4.09 | **0.0669 %** |
| **mean over beagle and essays** | | | **0.0654 %** |

Sensitivity: 0.055 to 0.057 % if the ranked capacity never leaves 768, and 0.091 to 0.096 % at capacity 1280.

That is **1.9× the Q-row shrink** and **1.9× the current crown gap of 0.035 %**, on a model that a ranked pair has just validated to 1.4 %.

- **Minimum useful effect.** Local replicate spread on the marginal draft step is 0.41 %, and the two long legs agree to 0.04 %. A 26 µs change is 1.17 % of the 2,226.5 µs step, which is 2.8× the replicate spread. There is also a **zero-noise detector**: the marginal dispatch count must fall from 27 to 25, and the head-dispatch model must move from `F + 27*(d-1)` to `F + 25*(d-1)` with zero residual.
- **Exactness argument.** The removed work is an identity copy of the cache array. It writes the same bytes it reads and is then overwritten in one position by the same `slice_update` that runs today. Removing it cannot change any arithmetic. A bit-exact change cannot move a draft length, so `effective_mean_draft_len` and `accepted_draft_rate` are free detectors and must be identical to all reported digits, together with `all_tokens_matched=true` and `residual_divergence_count=0` over a full 512-token exact run including post-EOS continuation.

### The restructuring test, applied honestly

Your rule is that an arm must only delete provably dead work and must revert permanently to the legacy path on any unexpected state. Two points against this arm, stated plainly:

1. **The implementation route is not yet proven.** I have measured that the copy disappears and that tokens still match, but I have not identified which live reference blocks MLX's donation. The fix could be a reference drop, which is a deletion, or it could need an evaluation barrier, which is an insertion of host synchronisation. **If the fix turns out to need an evaluation barrier or a cache restructuring rather than a reference drop, that is a restructuring and I will stop and report it rather than continue.**
2. **A rollback risk must be closed first.** If the head snapshot and rollback machinery depends on `slice_update` producing a fresh array, removing the copy would silently break rollback. The isolated legs kept `all_tokens_matched=true` over 72 tokens with a real rejection rate, which is evidence against that risk but is not conclusive. The first step of the arm must be a targeted exactness test over a full 512-token window with forced rejections.

Given both points, the arm's first gate is a bounded investigation, not an edit.

### Named rider — the Q-row shrink

If you judge the copy fix a restructuring, the **Q-row shrink is the fallback and I have now priced it from my own census at 0.0352 % on beagle and essays**, which matches your ranked measurement. It deletes 1,024 dead `q_proj` rows on every head call, it is bit-exact by row independence, and ox-alpha has already proved the form on the ranked host.

### Reported but explicitly not proposed — the affine-2 readout rate

`draft_lm_head` at 158.2 GB/s holds **334 µs per draft**, which is by far the largest measured headroom in the head and 13× the copy. On the same model it would be worth about 0.83 % on beagle and essays if fully recovered. I am **not** proposing it, because raising the rate of a dequantisation-bound kernel means changing packing, threadgroup mapping or the readout's numerical path. That is a restructuring, not a deletion, and it sits in exactly the family that produced six consecutive rejections. Per your standing rule I am naming it and stopping.

## Stop rule

The assignment's closure condition was that the gap be irreducible. It is not. Condition 1 fails: class 4 is 46.1 % of the head pass, not under 3 %. Condition 3 fails: closure is 99.6 to 99.9 %, so the census is complete and the residual is not hiding the gap. The head-efficiency axis stays open, and rung 3 has named where it lives.

## Honest gaps

- These are counting and GPU-clock attribution legs, not timed arm contrasts. I did not record entry and exit GPU temperature and I did not run ABBA counterbalancing, because no arm-versus-arm timing contrast is claimed. The one cross-session comparison is the copy cost, and there the drift-immune within-leg capacity step agrees with the cross-geometry estimate to 6 %.
- The per-kernel NNLS is rank deficient: 27 equations, 34 unknowns, rank 26, 18 identifiable coefficients. The primary table is the per-command-buffer table, which needs no solver.
- Four legs have small clean-round counts. Every headline uses the two long legs.
- The ranked capacity profile of the head KV cache is modelled from the 512-seed plus 512-decode window, not measured. It is the largest single source of spread in the arm's price.

## Suggested follow-ups, not implemented

1. Prove why the head's `slice_update` fails to donate, using a two-line MLX reproduction outside the worker. This decides whether the copy fix is a deletion or a restructuring, and it is cheap.
2. Check whether the target's Gated DeltaNet recurrent-state updates have the same donation failure. The target verify phase is 89.8 % of MTP GPU busy, so the same pattern there would be worth far more than the head copy.
3. Measure the affine-2 dequantisation cost directly with a microbenchmark at fixed bytes and varying bit width. That converts the 334 µs headroom estimate into a measurement and tells us whether a legal readout change could ever recover it.
4. Sample edward's bandwidth curve between 16 and 64 MB on one host so the size-matched rule stops extrapolating in the range where `fc`, `o_proj` and `q_proj` live.
