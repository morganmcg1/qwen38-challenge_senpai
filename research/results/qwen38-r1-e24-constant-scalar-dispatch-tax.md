# E24 — Are the per-layer scalar `asType` casts a real dispatch tax?

- **Assignment** `qwen38-r1-e24-constant-scalar-dispatch-tax` (r1), PR #28
- **Base** `55c727e959e26cf24333d3e8c0896f7d97ab1224` on `senpai/qwen38-mtp-r1`
- **Host** Apple M4 Pro, 48 GiB, macOS 26.5.2, Swift 6.3.3 — **not** the ranked
  M5, so every wall-clock number here is **directional only**.
- **Submitted surface touched** exactly one file:
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`.

**Question.** Two `invScale` constants are rebuilt in every one of the 48 Gated
DeltaNet layers on every target forward. Each is
`MLXArray(Float).asType(dtype)`. Is that a real per-forward GPU dispatch cost,
and is removing it worth more than measurement noise end-to-end?

Phases 0–2 were **pre-registered before any measurement**
(`research/e24-prereg.md`, commit `dccc2af`, posted to PR #28 as comment
`5322977006`) and are scored honestly against that pre-registration below,
including the three predictions I got wrong.

---

## Phase 0 — pre-registration

### 0a. The trace survives to a real kernel launch (verified at source)

Every link was read, not assumed:

| step | source | finding |
|---|---|---|
| Swift guard | `Vendor/mlx-swift/Source/MLX/MLXArray.swift:495-501` | short-circuits only when `type != self.dtype` fails; f32→bf16 passes |
| C++ op | `…/Cmlx/mlx/mlx/ops.cpp:270-284` | `astype` short-circuits only on **equal** dtype |
| eval | `…/mlx/transforms.cpp:193-195, 264-268` | no size threshold, no scalar special-case, no CPU redirect |
| GPU | `…/backend/gpu/primitives.cpp:28-33` | `AsType` → `copy_gpu` |
| copy | `…/backend/metal/copy.cpp:13-24` | early return needs `donated && equal dtype`; donation is **structurally impossible** (itemsize 4 ≠ 2) |
| copy | `…/backend/metal/copy.cpp:26, 38-40` | returns early only when `out.size() == 0` |
| dispatch | `…/backend/metal/copy.cpp:178` | `dispatch_threads`, kernel `v_copy_float32_bfloat16`, **1 thread** |

`compile.cpp:576-578` looked like a threat to the whole premise — it is
scalar **CSE, not constant folding**, and in any case the Qwen forward is not
executed inside a `compile()`. The trace therefore survives: each site is one
real, 1-thread kernel launch per forward.

### 0b. The prize is exactly 96 dispatches per forward, and it is width-independent

The load-bearing fact is that **`MLXArray(Float)` costs zero dispatches**:
`MLXArray+Init.swift:196-209` calls `mlx_array_new_data`, and
`array.cpp:258-262` marks the result `Status::available` with no primitive. So
the cast, not the construction, is the entire cost, and each site is exactly
one dispatch → **2 × 48 = 96 per target forward**.

Two corrections I found against my own initial framing, both of which **shrink**
the claim:

- The fifth `invScale` pair at `Sources/MLXFastModel/Qwen35GatedDelta.swift:316-322`
  belongs to `Qwen35FastEngine`, which is **off the scored path**. Not edited,
  not counted.
- The MTP draft head has **no GDN layers** (`mtp_num_hidden_layers: 1`;
  `Qwen35MTP.swift:23,33`). Drafting adds **zero** dispatches, so the MTP-leg
  prize is set by **rounds**, not by emitted tokens.

Also decisive for the implementation: `MLXArray(x, dtype: .bfloat16)` routes to
`init(bfloat16:)`, which **itself calls `mlx_astype`**, and there is no
`BFloat16: HasDType` conformance. The "obvious" one-line fix does not remove
the dispatch. **Caching is the only fix that works.**

### 0c. Cost model and the threshold, fixed before measuring

Anchors from `research/results/qwen38-r1-e17-curve-transfer-and-refit.md`
(prompt `english`, 512 tokens, shipped CURVE default): MTP `s/tok` 0.046275 →
leg 23.693 s; decode share 0.83136 → **true decode 19.698 s**; **246 rounds**;
depth histogram `{1:2, 2:237, 3:7}`.

246 rounds × 96 = **23,616 dispatches removed** on the MTP leg.

**Threshold adopted before measuring: U ≥ 0.50 %, with no "hope zone."**

My pre-registered predictions, and how they scored:

| # | prediction | outcome |
|---|---|---|
| P1 | `asType` reaches a kernel launch | **confirmed** |
| P2 | `MLXArray(Float)` is free | **confirmed** |
| P3 | draft head adds no GDN dispatches | **confirmed** |
| P4 | per-cast slope ≈ 2.5 µs (80 % CI 1.0–6.0) | **refuted** — measured 9.711 µs |
| P5 | end-to-end bound ≈ 0.30 % (CI 0.120–0.719) | **refuted** — measured 1.164 % |
| P6 | Phase 1 will **fail** the 0.50 % threshold (~65 % confident) | **refuted** — it passed |

I was wrong in the same direction three times: I underestimated MLX's
per-operation overhead by roughly **4×**.

---

## Phase 1 — microbenchmark of one scalar cast

`Tests/MLXFastTests/E24ScalarCastDispatchCostTests.swift`, gated on
`MLXFAST_RUN_MLX_RUNTIME_TESTS=1` **and** `E24_MICROBENCH=1`. Counts
`[0, 256, 512, 1024, 2048, 4096]`, 21 reps, 5 warmup. Three arms so the answer
is a **marginal** cost in a mixed stream, not a best-case isolated launch:

| arm | what it measures | slope |
|---|---|---|
| A `homogeneous` | pure casts (launch floor) | 9.465 µs/cast |
| C `filler_only` | cycling filler alone (subtrahend) | 7.646 µs/filler |
| B `cast_plus_filler` | cast interleaved with filler | 17.357 µs/pair |
| **B − C** | **marginal cost of one cast** | **9.711 µs** |

All fits r² ≥ 0.99997, inter-quartile range < 1 %.

Projected onto the anchors: 23,616 × 9.711 µs = 0.2293 s against 19.698 s of
local MTP true decode → **U = 1.164 %** (eval-only 0.937 %). **Passes the
0.50 % threshold.**

**The caveat that matters, stated before the result is used.** A
microbenchmark that does nothing but launch tiny kernels **starves the GPU**,
so launch overhead is fully exposed and 1.164 % is an **upper bound**, not a
forecast. Coherence check:

- a real round is 19.698 / 246 = **80.1 ms** carrying ~1016 dispatches at M = 3
  → ~78.8 µs/dispatch on average;
- a ~13.5 GB/round working set at ~273 GB/s is a **~49 ms bandwidth floor**,
  leaving ~31 ms of non-bandwidth time in which ~1016 × 7.5 µs ≈ **7.6 ms** of
  launch bubble could hide;
- at a realization factor of ~0.5 the end-to-end effect should be **~0.58 %** —
  still ~8× the **+0.0722 %** reproducibility floor the advisor measured.

*Inferred, not measured:* `backend/metal/device.cpp:574-596` sizes command
buffers by arch letter; M4 Pro takes the `'g'` branch with
`max_ops_per_buffer_ = 40`. I did not verify this at runtime and no conclusion
rests on it.

**W&B** (group `qwen38-r1-e24-constant-scalar-dispatch-tax`):
`1n5e4erm` (A), `zzrhnzgq` (B), `44zqoc4o` (C), `vwx3a3lf` (analysis).

---

## Phase 2 — implementation and bit-exactness

A per-layer, dtype-keyed memo on `Qwen35GatedDeltaNet`, placed **beside the
existing `negExpALog` memo** (`Qwen35.swift:573-587`) — the same class and the
same pattern, so this reuses an established precedent rather than inventing
one:

```swift
private var _invScaleMemo: [DType: (sq: MLXArray, lin: MLXArray)] = [:]
private func invScalePair(_ dtype: DType) -> (sq: MLXArray, lin: MLXArray) {
    if let cached = _invScaleMemo[dtype] { return cached }
    let invScale = pow(Float(headKDim), -0.5)
    let value = (sq: MLXArray(pow(invScale, 2)).asType(dtype),
                 lin: MLXArray(invScale).asType(dtype))
    _invScaleMemo[dtype] = value
    return value
}
```

Keyed by dtype because the packed-prework branch pins `.bfloat16` while the
other three follow the activation dtype. Four call sites converted (≈779
`processChunk`, ≈834 `mixerHit` packed prework, ≈874 else branch, ≈1061
`S == 2` midKernel).

- **Bit-exact by construction:** the cached value *is* the result of the same
  expression, evaluated once rather than per call.
- **Aliasing/lifetime:** `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/array.h:294-296`
  makes `is_donatable()` require `use_count() == 1` on both the descriptor and
  its data. An instance-variable memo permanently holds an extra reference, so
  the cached constant is **structurally non-donatable** — a *stronger*
  guarantee than the status quo, not a new hazard.
- **Tests** `Tests/MLXFastTests/E24InvScaleMemoTests.swift`: bit-identical to
  the inline expression, and unperturbed across 64 reuse rounds. Real arrays
  and real kernels; no mocks.
- **Gates:** scope `OK, 1 submitted path`; budget
  `source=2423251/3000000 growth=988/262144`.

**Limitation I should volunteer:** `invScalePair` is `private`, so the unit
test proves the memo pattern and MLX's reuse semantics but **not** the four
call-site rewirings. Those are proved only by Phase 3's correctness evidence.

---

## Phase 3 — paired BASE/MEMO measurement over eight prose prompts

### A correction to the assignment's framing

The assignment specified "MTP leg carries the effect, serial leg as drift
control." **That is wrong for this change**, and scoring it that way would
have under-reported the result by construction. The constants live in the
**target** GDN layers, which the depth-0 serial leg executes exactly as the
MTP leg does:

- the serial leg pays **512 × 96 = 49,152** dispatches per prompt;
- the MTP leg pays only **~246 × 96 = 23,616**;
- so the "control" carries the **larger** prize, and since both legs speed up,
  the local serial-to-MTP **ratio partly cancels the effect**.

I therefore promoted the serial leg from control to **second independent
witness** and scored on **absolute, prefill-subtracted decode wall seconds**,
which is the only valid instrument here. Goldens are pinned to a single named
build (`E24_GOLDEN_PIN=BASE`) so both arms check against **identical**
reference rows.

<!-- PHASE3_RESULTS -->

---

## Reproduction

```bash
research/e24-build.sh BASE MEMO          # arms from named git refs
research/e24-run.sh --goldens            # 512-step reference rows, pinned to BASE
research/e24-run.sh --arms BASE,MEMO english narrative   # ABBA, 2 prompts/job
research/e24_analyse.py --json --logs <job logs>
research/e24_wandb_phase3.py
```

Phase 1 alone:

```bash
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 E24_MICROBENCH=1 \
  swift test --force-resolved-versions --filter E24ScalarCastDispatchCost
```
