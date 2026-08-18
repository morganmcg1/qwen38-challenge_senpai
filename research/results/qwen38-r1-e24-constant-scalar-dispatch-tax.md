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

## Phase 3 — paired BASE/MEMO measurement over prose prompts (4 of 8 registered)

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

### Result: the mechanism fires, the speedup does not arrive

Four prompts, balanced ABBA (2 BASE-first, 2 MEMO-first), 512 decode tokens,
prefill subtracted, absolute wall seconds. `effect% > 0` means MEMO is faster.

| prompt | MTP base | MTP memo | MTP % | SER base | SER memo | SER % | rounds |
|---|---|---|---|---|---|---|---|
| english | 20.8113 | 20.8041 | +0.034 | 34.1244 | 34.1215 | +0.008 | 251/251 |
| narrative | 20.8679 | 20.8387 | +0.140 | 34.1620 | 34.1305 | +0.092 | 245/245 |
| technical | 19.5293 | 19.5050 | +0.125 | 34.1500 | 34.1000 | +0.146 | 231/231 |
| dramatic | 19.7223 | 19.7526 | **−0.154** | 34.1485 | 34.0339 | +0.335 | 220/220 |

- **MTP leg (the scored leg): median +0.080 %, mean +0.036 %, range −0.154 %
  to +0.140 %, MEMO faster on 3/4 prompts.**
- Serial leg: median +0.119 %, mean +0.146 %, range +0.008 % to +0.335 %,
  MEMO faster on 4/4.

Pre-registered threshold was **0.50 %**, fixed before any measurement. The MTP
result is **6–14× below it**, sits at or under the +0.0722 % MTP
reproducibility floor the advisor established, and **changes sign on one of
four prompts**. Stop rule 4 (effect inside the ABBA spread) is met. This is a
**null on the scored path**.

### The null is not a botched experiment

Four independent checks confirm the change was actually present and active:

1. **Static**: no raw `MLXArray(invScale)` or `MLXArray(pow(invScale, 2))`
   remain in `Qwen35.swift`; only the memo constructs them, and all four call
   sites (779, 834, 874, 1062) route through `invScalePair`.
2. **Caching**: the Phase 2 unit test shows `invScalePair` returns the
   identical cached array on the second call.
3. **Persistence**: `Qwen35GatedDeltaNet` is a `final class` (a reference type)
   built once in `Qwen35DecoderLayer.init`, so `_invScaleMemo` survives every
   forward rather than being rebuilt per token.
4. **Binaries differ**: `worker_sha256` BASE `ef296fd8…` vs MEMO `4aa78316…`;
   `source_sha256` `43fcfddb…` vs `f3e1f655…`, stamped per arm at run time.

### Why the predicted saving evaporates

Phase 1 measured a real marginal cost of **9.711 µs per cast**, which is
**2.15× the advisor's 4.521 µs break-even** for a 0.5 % effect. That number is
not wrong — it just does not reach the wall clock.

| leg | dispatches/forward | forward wall | encode share | predicted saving | measured | realization |
|---|---|---|---|---|---|---|
| MTP (M≈3) | 1016 | 85.6 ms | 11.5 % | 0.2207 s | +7.62 ms | **0.035** |
| serial (M=1) | 1480 | 66.7 ms | 21.6 % | 0.4773 s | +49.74 ms | **0.104** |

**96 % of the removed encode time never becomes wall time.** The MTP round is
GPU-bound: against a ~49.5 ms bandwidth floor (~13.5 GB at ~273 GB/s), the CPU
encode thread has tens of milliseconds of slack per round, so deleting 0.93 ms
of encode work is absorbed by CPU/GPU overlap instead of shortening the round.

The forward-count scaling check is consistent with this and worth stating
carefully. Casts are paid once per **target forward** and are width-independent,
so an equal-exposure model predicts the serial leg should save 512/236.8 =
**2.16×** the MTP leg; measured is **6.53×**. That gap is *not* a refutation —
equal exposure is the wrong model. An M=1 serial forward issues **more**
dispatches (1480 vs 1016) over **less** GPU work (66.7 ms vs 85.6 ms), so
encode sits closer to the serial critical path. Its encode share is 1.87×
higher, and it realizes 3.02× more of the tax: same direction, same order of
magnitude. **Exposure, not forward count, governs how much of a CPU-side
saving survives.**

**This serial-leg gain cannot help the score.** The official score is
pinned-serial ÷ candidate, and the ranked serial leg runs the *organizer's
pinned build*, which does not contain this change. Only my local harness runs
the candidate build on both legs. So the serial result is a valuable second
witness for the mechanism, but scoring-irrelevant. Applied to the scored MTP
leg alone, +0.038 % would move the 3.13099 frontier to ≈3.13217.

### Arm effect vs run-position effect

| leg | arm effect | position effect |
|---|---|---|
| MTP | +7.62 ms | −8.13 ms |
| SERIAL | +49.74 ms | +23.29 ms |

On the MTP leg the run-position (residual-heat) effect is **larger in magnitude
than the arm effect and points the other way** — precisely the confound ABBA
exists to cancel, and precisely why a 3/4 sign count should not be read as a
result. Per-prompt MTP arm deltas (+7.2, +29.3, +24.3, −30.3 ms) disagree in
sign and vary ~4× in magnitude.

### Correctness: exact, and behaviourally inert

- Every timed leg, both arms: `matched=True parity=True divergence=0`,
  declared rows == checked rows (817, 829, 787, 800), `tripwire=True`.
- **Cross-arm reference-row identity: bit-identical on all four prompts**,
  513/513 rows, `worst |Δtop2| = 0.0`, 0 mismatches. This closes the gap the
  Phase 2 unit test could not: `invScalePair` is private, so the unit test
  proved the memo but not the four call-site rewirings. These rows carry exact
  top-two logits from all 64 target layers across the full window.
- Identical round counts and identical width histograms per prompt (e.g.
  dramatic `M={2:14, 3:94, 4:77, 5:28, 6:7}`, mean depth 2.6364 in both arms),
  so the change is behaviourally inert — it alters no acceptance decision.

### Thermal and gate disclosure

Entry-temperature spread across all eight timed legs was **0.297 °C**
(42.860–43.157 °C). Mean entry: BASE 43.018 °C, MEMO 42.954 °C — a +0.065 °C
bias, i.e. BASE started marginally *warmer*, which flatters MEMO. ABBA cancels
this to first order, but it is one more reason not to promote a sub-0.1 %
residual.

Carried verbatim, not softened: every leg recorded
`cool_gate=stalled_above_40.0C`, `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false`. This host's idle GPU floor (~42.8–43.2 °C,
confirmed repeatedly) sits above `COOL_GATE_TEMP_C=40`, so the wrapper gate is
unsatisfiable here. Timing ran under the E15-authorized
`MLXFAST_LOCAL_COOL_GATE=0` policy, applied identically to both arms, with a
settle-to-plateau protocol (target 40 °C, max 240 s, 0.25 °C stall epsilon)
before every arm.

Two further disclosures:

1. `english-MEMO` stamped `dirty=2`: two research-only Python files were
   uncommitted at launch. They are not compiled into the worker, and that arm's
   `worker_sha256`/`source_sha256` match the pre-built MEMO binary exactly.
   Fixed from `narrative` onward (`dirty=0` on all six later legs).
2. `mlx.metallib` is stale relative to the vendored Metal sources (recorded
   `6639cc59…`, current `3e2818f1…`). This is pre-existing and **identical
   across arms**, so it cannot confound BASE vs MEMO, but it rides in the
   absolute numbers.

### Scope honesty

I pre-committed to running prompts in registered order and stopping only at an
**even** prefix, reporting every completed prompt and never dropping one after
seeing its number. I stopped at 4 of 8 under stop rule 4 rather than spend a
further ~1.6 GPU-h re-confirming a null that is 6–14× below threshold. All four
completed prompts are reported above.

### Result label: **Not useful**

The target cost is real and was measured (9.711 µs/cast, 2.15× break-even), the
implementation is correct and bit-exact, but the valid implementation has **no
meaningful end-to-end gain** on the scored path. Not *Invalid* — correctness is
perfect. Not *Unclear* — the balanced decomposition and the exposure analysis
both explain the null rather than leaving it to noise.

### Transferable finding for the cost model

The advisor's "everything else = 7.384 ms/round" residual is real as wall time,
but its **dispatch-encode component is largely off the critical path**. The
conversion factor from saved encode time to saved wall time on this workload is
**~0.035 on the scored MTP leg** (~0.10 on an M=1 serial forward), not ~1. Any
proposal of the form "remove N CPU-side dispatches from the target forward"
should be discounted by that factor before it is costed. On the MTP leg, a
0.5 % wall-clock win would require removing ~29 % of all encode work, not 12.6 %
of the non-QMV residual. **Dispatch-count reduction is only promising where
encode is exposed** — narrow widths, short kernels, or paths with forced
evaluation barriers — and the M≈3 speculative round is not such a path.

### W&B evidence

| run | id | url |
|---|---|---|
| phase3-BASE | `og68oxqa` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/og68oxqa |
| phase3-MEMO | `9swz7m3m` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/9swz7m3m |
| phase3-analysis | `eaow2hx9` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/eaow2hx9 |

Phase 1: `1n5e4erm` (A), `zzrhnzgq` (B), `44zqoc4o` (C), `vwx3a3lf` (analysis).

### Suggested follow-ups (not implemented)

- The 1480-dispatch M=1 path has 21.6 % encode share and is where this class of
  optimization would actually pay. It is not the scored width, but the
  **M=2 round (1544 dispatches, E23) is** — and it is 9.9 % of shipped-default
  rounds. Worth a targeted look at whether encode is exposed at M=2.
- Encode exposure should be measured directly (e.g. a deliberately inflated
  dispatch count vs wall time) to calibrate the 0.035 factor, rather than
  inferred from a null.


---

## Reproduction

```bash
research/e24-build.sh BASE MEMO          # arms from named git refs
research/e24-run.sh --goldens            # 512-step reference rows, pinned to BASE
research/e24-run.sh --arms BASE,MEMO english      # ABBA order fixed by registered index
research/e24-run.sh --arms BASE,MEMO narrative   # one prompt per job (30-min job cap)
research/e24-run.sh --arms BASE,MEMO technical
research/e24-run.sh --arms BASE,MEMO dramatic
research/e24_analyse.py --json --logs <job logs>
research/e24_wandb_phase3.py
```

Phase 1 alone:

```bash
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 E24_MICROBENCH=1 \
  swift test --force-resolved-versions --filter E24ScalarCastDispatchCost
```
