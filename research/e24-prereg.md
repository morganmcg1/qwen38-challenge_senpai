# E24 pre-registration — constant scalar dispatch tax

Assignment `qwen38-r1-e24-constant-scalar-dispatch-tax` rev `r1`, PR #28, student `qwen-alphonse`.

Committed **before any GPU is used**. Everything in Phase 0 below is zero-GPU source reading and is
already resolved; Phases 1–3 are unresolved and their thresholds are fixed here so they cannot be
moved after seeing a number.

| field | value |
| --- | --- |
| base ref / SHA | `senpai/qwen38-mtp-r1` / `55c727e959e26cf24333d3e8c0896f7d97ab1224` |
| head at commit time | `3ff131b2ee19f899a3c90e537c0c5957dd598045` |
| host | Apple **M4 Pro**, 48 GiB, macOS 26.5.2 — **directional only**, ranked host is M5 |
| toolchain | Apple Swift 6.3.3 (swiftlang-6.3.3.1.3), target `arm64-apple-macosx26.0` |
| promoted frontier | submission `bd007bc7-e8ab-4919-baf4-d5e90068dd83`, sourceRef `d1530a40…`, score 3.13098700135133 |
| budget | 4 GPU-hours |
| written | 2026-08-18T02:58:04Z |

## Hypothesis

The GDN layers construct two `invScale` scalars per layer per forward and cast each to the working
dtype. Each cast is a real 1-thread Metal kernel launch that computes an input-independent constant.
At 48 GDN layers that is a fixed **96 launches per verify forward** of pure launch overhead with
zero useful bandwidth. Hoisting them to a cached constant should be free of correctness risk. The
question is only whether the saving is measurable.

---

# Phase 0 — resolved, zero-GPU

## 0a — the dispatch trace is CONFIRMED; no short-circuit exists

Every link verified at source in this checkout:

| step | citation | finding |
| --- | --- | --- |
| Swift `asType` guard | `Vendor/mlx-swift/Source/MLX/MLXArray.swift:495-501` | guard is `type != self.dtype`; f32→bf16 passes through |
| `astype` op | `…/Cmlx/mlx/mlx/ops.cpp:270-284` | short-circuits only on **equal** dtype |
| eval routing | `…/mlx/transforms.cpp:193-195`, `:264-268` | schedules any `unscheduled` node; GPU nodes go to `gpu::eval`. **No size threshold, no scalar special-case, no CPU redirect** |
| `AsType::eval_gpu` | `…/backend/gpu/primitives.cpp:28-33` | contiguous input → `CopyType::Vector` → `copy_gpu` |
| `copy_gpu` | `…/backend/metal/copy.cpp:13-24` | early-returns only if `donated && in.dtype()==out.dtype()`; f32→bf16 fails both, and donation is structurally impossible because `is_donatable` needs equal itemsize (4 ≠ 2) |
| `copy_gpu_inplace` | `…/backend/metal/copy.cpp:26,38-40` | early-returns only if `out.size()==0`; a size-1 array proceeds |
| the launch | `…/backend/metal/copy.cpp:178` | `compute_encoder.dispatch_threads(grid_dims, group_dims)`, kernel `v_copy_float32_bfloat16`, **1 thread** |

No compile-time escape either. `compile.cpp:576-578` `compile_simplify` is scalar **CSE, not constant
folding**, and a `Compiled` node is only produced by `compile_fuse` (`compile.cpp:1150`) inside
`compile_impl` — and the Qwen forward is **not inside any `compile()`**. The only `compile(` calls in
this repo's sources are `Sources/MLXFastModel/LagunaRuntimeModel.swift:453,465,478,491`, which are
unrelated prior art.

**P1 → CONFIRMED.**

## 0b — the prize is EXACTLY 96, and it is width-independent

The advisor asked me to widen the enumeration in case the true number is larger than 96. It is not.
The widened search found **no additional dispatching constant site**, and it corrected an
overcount I had to adjudicate between two conflicting sub-analyses.

**Decisive fact: `MLXArray(Float)` costs ZERO dispatches.**
`Vendor/mlx-swift/Source/MLX/MLXArray+Init.swift:196-209` — `init<T: HasDType>(_ value: T)` calls
`mlx_array_new_data(ptr, [], 0, dtype)`. `…/mlx/array.cpp:258-262` — the `ArrayDesc(Shape, Dtype)`
constructor sets `status = Status::available`, and Metal's `allocator::malloc` returns
`MTL::ResourceStorageModeShared`, so the CPU writes 4 bytes straight into unified memory. No blit,
no command buffer, no dispatch. Therefore each `invScale` site costs **1** dispatch (the `.asType`
alone), not 2 → **2 × 48 = 96**, not 192.

**The attention scalars are also zero-dispatch.** `Qwen35.swift:1435` passes `eps` / `log2Base` /
`offset` through `MLXFastKernel.swift:150` `inputs.map { $0.asMLXArray(dtype: nil) }`.
`DType.swift:240-243` sends `HasDType.asMLXArray(dtype: nil)` to `MLXArray(self, dtype: .float32)`;
`init<T: HasDType>(_ value: T, dtype: DType)` takes the `T.dtype == dtype` branch →
`case is Float32.Type: self.init(value as! Float32)` → `mlx_array_new_data` → **0 dispatches**. The
`Int` offset resolves through `.int32` and the `BinaryInteger` branch to `self.init(Int32(v))` →
likewise **0**. So there is no 32-dispatch attention contribution.

**The four `invScale` sites** (`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`):

| lines | branch | dtype | on scored path? |
| --- | --- | --- | --- |
| 754 / 756 / 759 | `processChunk` | dynamic | no — not reached from the `:1008` dispatch |
| **811 / 814-815** | `processChunkStashingPrefix`, **`mixerHit` fast branch** | hardcoded `.bfloat16` | **yes — the live shipped-default branch at M ≥ 3** |
| 853 / 855 / 858 | `processChunkStashingPrefix` else branch | dynamic `q.dtype` | yes, when `mixerHit` is false |
| 1043 / 1045 / 1048 | `S == 2` midKernel path | dynamic `q.dtype` | yes at M = 2 |

Exactly one branch fires per GDN layer per forward, so the count is **96 at M = 3 and 96 at M = 6 —
flat in M**. This is the single most useful simplification in the whole experiment: only the *round
count* matters, never the width histogram.

The `mixerHit` predicate is `Qwen35.swift:797-803`
(`isCompiledDecodeSupported && B==1 && 3<=S<=9 && nKeep==3 && numKHeads==16 && numVHeads==48 &&
headKDim==128 && headVDim==128 && qkv.dim(2)==16*128*2+48*128 && qkv/convState/a/b all .bfloat16`).
Footnote: `MLX_COMPILED_DECODE` can override `isCompiledDecodeSupported`, and `MLX_` **is** in the
worker env allowlist, so that branch is externally reachable — irrelevant to the fix, which covers
all three live branches, but worth stating.

**Two paths that do NOT add to the count**, both confirmed this phase:

1. `Sources/MLXFastModel/Qwen35GatedDelta.swift:316-322` holds a fifth `invScale` pair, but it
   belongs to `Qwen35FastEngine`, reached only via `Sources/MLXFastModel/Qwen35Model.swift:91`
   `fastPathLogits` ← `Qwen35RuntimeWeights.swift:59/195`. `Qwen36MTPBlockSession.swift` contains
   **zero** references to `Qwen35FastEngine`, `fastPathLogits`, `Qwen35RuntimeWeightCache`, or
   `executeQwen35CachedForward`. The scored worker
   (`Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift:153` `LLMModelFactory.shared.load`,
   `:194,197` `Qwen36MTPBlockSession`) uses the MLXLLM `Qwen35Model`. **Off the scored path — do not
   edit it, do not count it.**
2. **The MTP draft head contains no GDN layers at all.** `weights/config.json` has
   `mtp_num_hidden_layers: 1`, and `Qwen35MTP.swift:23,33` states the layer "always uses full
   attention (never SSM/linear)" and instantiates `Qwen35Attention`. `grep invScale` on that file
   returns nothing. So **drafting adds zero invScale dispatches per round**; the 96 is purely a
   per-verify-forward cost and rounds = verify forwards.

**P2 → CONFIRMED** (prize is 96, not 192). **P3 → CONFIRMED** (no further sites; flat in M).

Live verify path, re-verified: `Qwen36MTPBlockSession.swift:930-932` builds `S = M = draftCount+1`
verify tokens → `:951-953` `model.callWithHiddenAndNormed(…, nConfirmed: 1)` → `Qwen35.swift:2757-2762`
→ `:2341-2354` → `Qwen35TextModelInner.callAsFunction :1930-1980` → `Qwen35DecoderLayer :1866-1888`
→ GDN `:964-1160`. Dispatch decision at `Qwen35.swift:1008`
(`nConfirmed == 1 && S >= 3 && mask == nil` → `processChunkStashingPrefix` at `:778`; `S == 2` →
midKernel at `:1021-1023`). Model layout `weights/config.json:8,91`: 64 layers,
`full_attention_interval: 4` → 48 GDN + 16 full-attention.

`python3 research/verify_forward_dispatch_inventory.py --verify-citations` exits 0 at this head
("checked 47 citations across 6 files … OK"), so the E23 line references above are current.

**A loud finding that constrains Phase 2.** The naive fix — "just build the constant as bf16" —
does not work. `MLXArray(x, dtype: .bfloat16)` routes to `init(bfloat16:)`
(`MLXArray+Init.swift:219-227`), which itself calls `mlx_astype` — **the same single dispatch**.
There is no `BFloat16: HasDType` conformance anywhere in `DType.swift:246-353`, so the zero-dispatch
`mlx_array_new_data` route is unreachable for bf16 through the stock Swift API. **Caching a computed
`MLXArray` is therefore the only correct fix**, which is exactly what the assignment prescribes. I
note this because it would be easy to "fix" this and measure nothing.

## 0c — cost model

Anchors from `research/results/qwen38-r1-e17-curve-transfer-and-refit.md`, prompt `english`,
512 decode tokens, shipped-default **CURVE** policy (E17's `S18`/`FLAT18` is the *retired* scalar,
not the default):

| quantity | value |
| --- | --- |
| MTP s/tok | 0.046275 → `T_m` = 23.693 s |
| decode share `(T_m − P_m)/T_m` | 0.83136 |
| **MTP true decode** | **19.698 s** |
| rounds | **246** |
| accepted-depth histogram | `{1:2, 2:237, 3:7}` |
| ranked candidate leg (E18 §11.2) | 8.557915 s |

Caveat I will repeat in the writeup: E17's histogram is *accepted* depth, so `M = depth + 1` is a
proxy that undercounts width. It does not matter here, because 0b proved the 96 is flat in M.

**Dispatches removed per 512-token decode run: 246 × 96 = 23,616** (plus a negligible ~96 per
prefill forward).

Per-dispatch prior for a 1-thread Metal kernel inside an already-open encoder: **2.5 µs** central,
**1.0–6.0 µs** 80 % range.

| per-dispatch | saved | % of local MTP true decode (19.698 s) |
| --- | ---: | ---: |
| 1.0 µs | 23.6 ms | 0.120 % |
| **2.5 µs** | **59.0 ms** | **0.300 %** |
| 6.0 µs | 141.7 ms | 0.719 % |

Ranked transfer at the central value: if absolute µs transfer, 59.0 ms / 8557.9 ms = **0.69 %**; if
the cost scales with host speed (M4 Pro / M5 ≈ 23.693 / 8.5579 = 2.77×), **≈ 0.30 %**.

### Pre-registered stopping threshold

The Phase 3 instrument (MTP true decode wall time) reproduced to **0.0722 %** across a provably inert
change, per the advisor's null control. I want an expected effect of at least **3×** the instrument's
own reproducibility, i.e. **≥ 0.217 %**. An isolated microbenchmark systematically overstates what a
real forward realizes; I budget a realization fraction of **≈ 0.5**. So the Phase 1 upper bound must
be **U ≥ 0.43 %**. I round to the advisor's suggested **U ≥ 0.50 %** — it is within rounding of my
own independent derivation and using their number removes any appearance of moving goalposts.

**There is no hope zone.** If `U` lands in [0.35 %, 0.50 %) I still stop and report the bound as the
result. I am writing that sentence down now precisely because I expect to be tempted later.

### Falsifiable predictions

| # | prediction | status |
| --- | --- | --- |
| P1 | dispatch trace confirmed end-to-end, no short-circuit | **CONFIRMED** (0a) |
| P2 | `MLXArray(Float)` is zero-dispatch → prize is exactly 96, not 192 | **CONFIRMED** (0b) — refutes "may be larger than 96" |
| P3 | no additional dispatching constant site; 96 at both M=3 and M=6, flat in M | **CONFIRMED** (0b) |
| P4 | Phase 1 per-dispatch slope = **2.5 µs** (80 % CI 1.0–6.0 µs) | open |
| P5 | Phase 1 upper bound = **0.30 %** of local MTP true decode (80 % CI 0.12–0.72 %) | open |
| P6 | **Phase 1 fails my own 0.50 % threshold and this ends as a bounded negative — I put this at ~65 %** | open |
| P7 | if Phase 3 runs, measured effect ≤ Phase 1 upper bound (realization fraction < 1) | open |

P6 is the honest headline. I am pre-registering that I expect my own hypothesis to fail.

---

# Phase 1 — microbenchmark (≤ 20 min GPU)

Measures the **per-dispatch slope** of a size-1 `float32 → bfloat16` cast by sweeping `N` and reading
the slope, not by timing one cast. This yields an **upper bound**, not a point estimate: a real
forward may hide some of this cost behind other work.

**Design deviation I am pre-registering (with justification): two arms, not one.** A homogeneous
loop of identical casts amortizes away the Metal pipeline-state switch that the real forward pays,
because in the real GDN layer each `asType` sits between unrelated kernels. One arm alone would
mis-state the bound in a direction I cannot sign.

- **Arm A (homogeneous)** — `N` identical size-1 casts back-to-back. PSO bound once. This is the
  pure launch-overhead floor and the cleanest slope.
- **Arm B (interleaved)** — `N` size-1 casts each separated by a *different* tiny kernel, forcing a
  PSO switch per cast, as in the real layer.

Report both slopes. Apply the threshold to **Arm B** (the realistic one) and report Arm A as the
floor; if they disagree materially that is itself a reportable finding.

`N` sweep: 0, 1k, 2k, 4k, 8k, 16k, 24k (24k brackets the modelled 23,616). Fit slope by least
squares on the linear region; report intercept as launch-independent overhead. Repeat each point
enough times to state a spread, and report spread next to the slope.

## Phase 2 — implementation (minimal GPU)

Hoist to `MLXArray(<float>).asType(dtype)` computed **once and cached**. Explicitly **not** computing
a bf16 constant on the Swift host — that would change the rounding path and therefore the tokens.
Cache keyed by dtype. `headKDim = 128` for all 48 layers, so one cache entry serves every layer and
every branch.

Required evidence, all of which I will produce rather than argue:

1. a **bitwise equality** test, asserted in code, that the cached constant is bit-identical to the
   per-call one for every dtype used;
2. a correctness run reporting `all_tokens_matched`, `residual_divergence_count == 0`,
   `parity_all_ok`, and declared rows == reference-checked rows;
3. an explicit written statement that I checked the cached `MLXArray`'s lifetime against the
   `asyncEval` ladder (a cached array captured across evals must not be mutated or freed mid-flight).

Scope and budget checked with `senpai/validate-assignment-scope.sh` and
`senpai/check-editable-budget.sh` before implementation. No Metal source changes are expected, so
`research/twin_audit.py` should be a no-op — I will run it anyway and say so.

## Phase 3 — end-to-end measurement (≤ 2.5 GPU-h, only if 1–2 are clear)

Metric: **MTP true decode `(T_m − P_m)`**, not the score. Per the advisor's calibration, any sub-1 %
local `--local-submit` *score* difference is noise (null control: score moved −0.5530 %; three scores
spanned 0.9636 %), whereas MTP true decode reproduced to +0.0722 %. `decode_seconds` is
prefill-**inclusive**, so prefill is reported separately and subtracted; dilution is ~1.19× at 512
tokens.

- 512 decode tokens, shipped default policy, MTP leg carries the effect; serial leg is a drift
  control only.
- **Prose prompts** `research/e17_prose_*_512.txt` — **not** the public golden, which is structurally
  capped near 300 decode tokens by a stop-token defect.
- ABBA, one session, interleaved; entry/exit GPU temperature recorded per arm.
- With `MLXFAST_LOCAL_COOL_GATE=0` I must carry `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false` **verbatim**, and report ABBA spread next to the effect. Idle GPU
  here is ~42.9 °C against a 40 °C gate, so the real gate is unsatisfiable on this host; this is the
  authorized path, not a shortcut.
- `macmon` lives at student-qwen-edward's home `bin/macmon`; set `MLXFAST_MACMON_BIN`.
- Report **per-prompt spread**, not just a pooled mean; plus round count and the derived width
  histogram at `M = depth + 1`.

## Stop rules

1. Phase 0 refutes the trace → stop. *(Did not fire; trace confirmed.)*
2. Phase 1 Arm-B upper bound < **0.50 %** → stop and report the bound.
3. Phase 2 cannot show bit-exactness → stop; the change is not free.
4. Phase 3 effect falls inside ABBA spread → report as unclear/negative, do not promote.
5. 4 GPU-hours elapsed → stop wherever I am and report.

## Out of scope

The `M = 2`-avoidance / depth-policy lever is explicitly **not** mine — it collides with E21 and E22.
I will not touch draft-depth policy even if Phase 3 makes it tempting.

## Queue discipline

The GPU is fully loaded and I am fourth. Every timed run goes through
`research/await-lock-then-run.sh MAX_WAIT_SECONDS CMD …`, and every GPU process goes through
`run_job`. I will not run git-mutating commands while another student's timed job is live.
