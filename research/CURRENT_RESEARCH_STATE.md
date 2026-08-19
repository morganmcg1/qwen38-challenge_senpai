# SENPAI Research State

- **2026-08-19 17:20 UTC**
- Track `qwen3.8-27b-mtp-v1`; advisor branch `senpai/qwen38-mtp-r1`;
  `BASE_SHA = daa1d0185d4d3fa1383539d07242e3654a587d6d`;
  `UPSTREAM_SHA = 0c90733d383f6b987a29682bf9eb9458a6172bfa` — the value actually synced into
  this tree. The organizer tip is `9e1ff9ec7152a04b753f2efb91c3e559909ea4b9` and the trusted
  delta between the two is **empty**, so a future sync has no contract work to do, only
  editable cherry-picks (ledger 179(G)).
- Most recent human research direction: issue #31 asked for a maintenance checkpoint, which is
  complete and closed. **Issue #22 — execute aggressively toward the winning frontier — is the
  standing directive.**

This is a **living document**. The per-experiment record of authority is
[`../senpai/campaign-ledger.md`](../senpai/campaign-ledger.md); durable measurements,
source-line citations and closed questions live in
[`ESTABLISHED_FACTS.md`](ESTABLISHED_FACTS.md). The 2955-line predecessor of this file is
retained verbatim as [`RESEARCH_STATE_ARCHIVE_2026-08-19.md`](RESEARCH_STATE_ARCHIVE_2026-08-19.md).
Keep this file to current hypotheses, live slots, and where we go next. Prune it every round.

---

## Where we stand

| quantity | value |
|---|---|
| live promoted frontier | **3.24985583421771** — submission `59b321e`, solver fkiene, commit `9e1ff9ec…` |
| our best official submission | **3.23250848263467** — receipt `ca9251b`, candidate `2b0c36a`, rejected |
| **our deficit** | **0.01734735158304 = 0.534 %** |
| our official submissions | six, under solver `morganmcg1`; four scored, none promoted |
| board floor (between-submission) | 0.7678 % |
| within-tree resubmission agreement | **≤ 0.0693 % per prompt** (item 148, six submissions of one tree) |
| local end-to-end null floor | **0.0629 %** — askeladd's E48 `base2` arm |
| ranked score leverage `psi_mtp` | **0.693391** [0.692292, 0.694490] |
| the leader's last step | **+0.0173 %** |

The entire 3.24986 promotion is 70 inserted lines, zero deletions, one file — an untimed
warm-up. Our deficit is 31× that step.

Confirmed causal boundary, re-verified (`senpai/verify-ranked-score-boundary.sh` PASSES):
candidate-editable code cannot move the ranked serial numerator, so any compliant edit that
lowers candidate MTP seconds per token improves every affected `raw_p`. Never subtract a
locally measured serial share such as `psi_serial` when pricing official value.

### 🔴 Our noise model is internally inconsistent, and it is load-bearing

Item 148 says six submissions of one identical tree agree to **≤ 0.0693 % per prompt**. Items
166/172 say the between-submission floor is **0.7678 %** with a 17× per-set spread. **Both
cannot describe one homoscedastic instrument**, and we currently use the large floor for MDE
arithmetic while implicitly using the small one to credit the frontier's +0.0173 % step to a
warm-up.

This decides what a submission slot is worth. If the large floor governs, our 0.534 % deficit
is about **0.7 sd of a redraw**, the leader's last two steps are indistinguishable from
resubmission lottery draws, and any legitimate new mechanism composed on our best tree has
order-25 % promotion probability per submission from noise alone — which argues for submitting
**more often**, not for hoarding slots. If the small floor governs, sub-floor composition is
rational and the leader is genuinely stepping. **Resolving this is zero-GPU** and it is the
highest-leverage open analysis question we have.

---

## Current research focus and themes

### 1. The QMV dispatch table is the main attack surface, and its cells are provably live

`quantized.h` dispatches `qmv_fast_crossrow_affine4_g64_m<T,M,IPG,true>` on verify width M.
The design rule is `IPG = ceil(M / ceil(M/4))`; the binding constraint is
`static_assert(NA >= 2 && NA <= 4)` at `:980`, accumulator `typedef vec<float, NA> VF`. Raising
a cell to NA=5 is the mechanism under test.

**Settled from source (ledger 179(D)):** a width-M verify dispatches QMV **once at the full M**
for every M in 1…9. Only the SDPA chunks. So `case 7`, `case 8` and `case 9` are live scored
cells.

Measured: thorfinn's `<T,9,5>` isolated cell win is **−12.255 %**. The register-tax objection
is **refuted**. The **bandwidth** objection from PR #8 is **still open**: one NA=5 group
sustained 95.5 GB/s against 165.6 for NA ≤ 4, while M=9 already runs at 88 % of peak.

🔴 `vec<float,5>` remains an unresolved hard gate: no local `vec` alias exists in the kernel
headers, so it resolves to `metal::vec`, which MSL specifies only for N ∈ {2,3,4}. Every NA=5
experiment must report `sizeof(VF)`, `alignof(VF)`, per-lane correctness, and a positive
control that fails on lane perturbation.

### 2. 🔴 THE CENTRAL OPEN DISAGREEMENT — the narrow-width cost split

askeladd (E48) and edward (E53) roughly **swap** the split and **agree** on its total:

| share of candidate-leg QMV cost | askeladd E48 | edward E53 |
|---|---|---|
| M ∈ {4,5,6} | 64.025 % | 65.0–68.9 % — agree |
| M ∈ {7,8} | **9.391 %** | **21.2–25.1 %** |
| M = 9 | **21.630 %** | **4.6–8.9 %** |
| {7,8} + 9 | 31.02 % | 25.8–34.0 % — agree |

Neither is a GPU measurement. **This decides which mechanism we ship**: under edward's mixture
the ranking *inverts*. **PR #57 settles it by direct measurement** — the hypotheses are 2.4–4.7×
apart and all three predictions are ≥ 6× the 0.0629 % null floor.

### 3. 🔴 The SDPA chunk predicate is WIDER than the constraint that motivated it

Full source proof in [`SDPA_ROUTE_MAP.md`](SDPA_ROUTE_MAP.md). Three routes exist, and the
trusted host dispatcher `scaled_dot_product_attention.cpp` is **not** in `editablePaths`:

| condition | route | threads / threadgroup |
|---|---|---|
| `qL >= 9` | `steel_attention` full attention | — |
| `qL <= 8`, `kL < 1024` | `sdpa_vector`, one pass | fixed **1024**, no `qL` term |
| `qL <= 8`, `kL >= 1024`, arch `'d'`/`'s'` | `sdpa_vector_2pass` | `32 * gqa * qL` |

`qL * gqa <= 32` governs **only** the two-pass route, and that route needs `kL >= 1024`. Our
window is `kL = 512 + tokensCommitted + M`, so `kL >= 1024` arrives only in the **final round or
two**. For essentially the whole scored window, widths 6, 7 and 8 are legal single calls and the
`qL >= 6` chunk splits them for no kernel-family reason — paying two query copies, one extra
SDPA dispatch and one `concatenated`, per full-attention layer, 16 layers deep.

The chunk is still load-bearing at `qL = 9` (steel avoidance) and at `kL >= 1024`
(`utils.h:84-96` **throws** when the thread cap is violated). Correct predicate:
`qL >= 6 && (qL >= 9 || kL >= 1024)`. Bit-exactness of the narrowed form is provable from
`sdpa_vector.h:15-176`: no reduction crosses query rows, the causal predicate is bottom-right
aligned, and chunked and unchunked see identical contributing keys in identical per-thread order.

**Second-order prize, larger than the first.** Deleting this surcharge removes the width 5→6
cost step that E56 (#59) is currently trying to price around, so `costModelDepth` can buy the
sixth row at its true marginal cost — on a pool whose own source comment says it **rewards
depth** (`:723`).

### 4. Warm-up and compile placement is the frontier's own current lever

Both the leader's last promotion (+0.0173 %) and the mechanism we hold that they lack
(+0.0283 %) are untimed warm-ups that move first-touch Metal compilation out of the scored
window. `SDPA_ROUTE_MAP.md` calibrates the unit: a pipeline-creation miss inside the scored
window is worth roughly **0.02 %**. This class is real, free and token-neutral but sits **below
our 0.0629 % local null floor**, so it is composed into a candidate, never screened as a local
A/B.

Two independent gaps in the frontier's own warm set: it never warms `qL = 2` or `qL = 3` (chunk
B of widths 7 and 8), and on an `'s'`-class host `blocks` takes **two** values in our range
(64 at `kL == 1024`, 128 above), so padding to exactly 1024 warms one pipeline and misses the
other.

### 5. Gated DeltaNet: the scan is NOT the cost, and the published fix does not apply

Verified geometry: `Hv=48, Hk=16, Dk=Dv=128`; SSM state 3 MiB fp32 per GDN layer; one
recurrence launch reads and writes 6.29 MB, so **302 MB per forward across 48 layers**. Against
a 14,413 MB forward that is **2.1 % of bytes**, and ~91 % of GDN bytes are its three quantized
projections, not the scan.

🔴 **State traffic is FIXED in draft width S.** The kernel loads state into registers once
before the t-loop and stores once after; grid `(32,128,48)` and threadgroup `(32,4,1)` are
independent of `T` (`GatedDelta.swift:54-58, 92-95, 162-163`). **This refutes the published
KVBuffer-style "deferred commit removes `O(m·d²)` per-row state traffic" recommendation for our
tree — our kernel already has the property those papers add.** Do not spend a slot on it.

What actually costs, in priority order:

- **A rejecting round pays three state passes** (verify, replay, next verify) where full
  attention pays two cache writes and one integer decrement: **302 MB + 48 dispatches per
  rejecting round**. Rejecting rounds are the **common case on prose** — per-draft accept is
  0.4685/0.4398 on the two prose proxies against 0.8875 on the copy task.
- **At S=2 the mid-state is written unconditionally and discarded on full accept**:
  3.15 MB/layer, **151 MB/round**. M=2 is 15.8 % of rounds on `natural_history`.
- **`q`/`k`/`g`/`beta` are re-read `Dv = 128×` per head per timestep** — the kernel's indexing
  contains `hk_idx`/`hv_idx` but never `dv_idx`. Order 340 MB/forward at S=9 of cache traffic
  for 8 KB of unique data. A `VPT` (values-per-thread) template is bit-identical by
  construction because each output keeps its own `simd_sum` over the same 32 lanes.
- `snapshotRecurrent` costs **zero** — `arrays[0]?[.ellipsis]` hits `ops.cpp:811-813`
  (`if (!has_neg_strides && out_shape == a.shape()) return a;`) and returns the same array. The
  protection is real but the doc comment's stated mechanism is wrong. Nobody should "optimize"
  this.

**The recurrence kernel's absolute cost has never been measured, and the microbenchmark already
exists**: `sweepGatedDelta` over widths 1…12 with `traffic_bytes` and `flops` at
`Tests/MLXFastTests/QwenQMVCostCurveTests.swift:898-966`, skipped whenever
`MLXFAST_QMV_COST_CURVE_SHAPES_ONLY=1`. That is the cheapest gate in the campaign right now.

Caution: E20 measures the forward at 14,413 MB / 197.45 ms ≈ **73 GB/s effective**, so the
forward is *not* bandwidth-bound and byte counts must not be priced at that average rate.

### 6. The proposal head is NOT unexplored, and its open lever is the shortlist

Correcting this file's previous claim. A non-organizer head **is already declared and in use**
(`mtp-head.manifest.json`, remote `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827`,
427,742,600 bytes). Replacing head **weights** is closed by measurement: two scored
submissions did it and both were rejected (`4437d06` at 2.86127, `9197ed6` at 3.06938).

What survives is head **runtime**:

- **Shortlist containment is the cheapest untried lever.** The proposal is `argmax over exact
  affine-4 logits restricted to the coarse affine-2 top-32` (`Qwen35.swift:3155-3216`). Nobody
  has measured `P(exact argmax ∈ coarse top-32)`. If it is below ~98 %, raising `K` to 64 costs
  ~82 KB of gather (about 0.05 % of the readout's 157 MB) and buys acceptance. The bitmask
  `static_assert`s at `:2506-2508` and `:2594-2596` already admit K=64. **Zero-GPU to falsify.**
- **A flat vocabulary crop is already dead**: halving the compact prefix to 49,152 regressed
  acceptance 1.00 → 0.877 (`Qwen35.swift:2757-2768`). The published FR-Spec / VocabTrim lever is
  therefore already partly harvested by our 98,336-row compact readout; only a *hierarchical*
  shortlist generator remains, and it must be declared and digest-pinned, never derived at load
  time.
- `h = 0.18` decomposes as **~22 % head step, ~78 % extra verify row** (E1: isolated head step
  2.590 ms against a 65.009 ms depth-0 round; 84.4 % of the depth-8 marginal is verify width).
  Measured per-depth `h` is `[0.084, 0.078, 0.243, 0.375, 0.292, 0.300, 0.287, 0.391]` —
  over-priced at d ≤ 1, under-priced at d ≥ 2. E56 (#59) is testing exactly this.
- **Our head does not collapse with depth.** Published vanilla MTP heads reused recursively go
  70 % → 10 % → ~0 % at k=1/2/3; our pooled tape is 0.693 / 0.584 / 0.508 / 0.419 — monotone,
  no cliff. The shipped prior `0.85 * 0.98^i` is the wrong shape in both directions, but the
  head itself is not the depth blocker.
- A free A/B already exists for the head's bf16 precision islands:
  `MLXFAST_QWEN_MTP_EXACT_QKV_ROWS` at `Qwen35.swift:2882`. Its dose has never been swept.

### 7. Measurement discipline that now gates every claim

- 🔴 **A green `--local-iterate` parity line is NOT exactness evidence.** E51 measured it: an
  arm reporting `all_tokens_matched=true`, `residual_divergence_count=0`,
  `public_drift_tripwire_passed=true` had moved declared top-two row evidence at **52 of 64
  positions** with two top-2 identity flips. Every brief touching precision, reduction order,
  packing, recurrence, cache layout or replay must gate on **declared per-position row
  evidence** with a positive control, and must state that the local parity line was not the gate.
- The **0.0629 % end-to-end null floor** (E48 `base2`) is the unit for every effect claim.
- Board intervals are **identification intervals, not standard errors** — 152 content-distinct
  trees reproduce the published telemetry to 16 digits, so the board gives ~1 observation.
- `psi_serial` is **NOT IDENTIFIABLE** locally (four treatments imply 0.7966/0.8694/1.0414/
  1.2470, two exceeding 1.0). Unidentified, not refuted.
- Ungated timing (`MLXFAST_LOCAL_COOL_GATE=0`) only ABBA-counterbalanced, entry/exit
  temperatures recorded, `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
  preserved verbatim.
- Log W&B **per leg while timing**, never at session end.

---

## Live experiment slots

| PR | student | experiment | state |
|---|---|---|---|
| #57 | askeladd | E55 — compose `<T,9,5>` onto the real shipped table to a **submittable candidate**; settles theme 2 | `status:wip` r1 |
| #58 | thorfinn | E54 — lone-versus-sibling NA=5 law across M=5/7/8/9; I predicted **Law C** on the record | `status:wip` r1 |
| #59 | edward | E56 — draft-depth schedule staircases; theme 3 changed what he should conclude | `status:wip` r1 |
| #60 | alphonse | E57 — narrow the SDPA chunk predicate (theme 3) | assigning |

Merged: **#55** (alphonse E51 — refuted at rung one, instruments retained), **#53** (thorfinn
E49), **#52** (askeladd E48), **#56** (edward E53). Base advanced
`981e69a` → `1247c57f` → `a35bb006` → `a2c3dbc4` → `67b8547` → `daa1d018`, each with
base-change inertness verified rather than assumed.

---

## Potential next research directions

Ordered by expected value, not by convenience.

1. **Submit whatever #57 produces, if it survives exactness.** A 0.39–1.84 % MTP-leg win on a
   0.534 % deficit is a submission, not a screen. Official evaluation is part of the research
   loop.
2. **🔴 Compile the MTP head step.** E29 measured a *genuinely removable host cost of 4.35 % of
   decode* and the campaign never acted on it — **69× the local null floor**, candidate-leg
   only, uniform across prompts, so it moves the 4th/5th order statistics. The session's own
   comment says the ~2.4 ms/step is host graph **build**, not GPU work to overlap
   (`Qwen36MTPBlockSession.swift:1048-1052`, `:649`). `CompilableKVCache.swift` and
   `CompiledDecode.swift` are in `editablePaths` and have **zero ledger mentions** — the
   facility was never tried. This is the largest recorded-but-unexploited lever we have.
   Hazard: the M1/M2/M4 Tahoe JIT crash class (`MLXHardwareInfo.swift:11-21`) may block local
   measurement; ranked M5 is outside the reported class.
3. **Resolve the noise-model inconsistency** (see "Where we stand"). Zero GPU: regress per-set
   MTP-leg variance on tree content, specifically warm coverage. It decides whether submission
   cadence or mechanism size dominates near the frontier, and it plausibly *upgrades* warm-up
   composition from a mean effect to a variance reduction that sharpens every future ranked
   observation.
4. **Land the latch release valve.** `positionAcceptEMA[0] <= 0.18` is an absorbing state:
   it is written only inside `recordAcceptOutcome`, whose single call site is unreachable at
   depth 0. Simulated at **−14.55 % to −18.02 %** when it hits a bankable prompt, observed at
   3/94 ≈ 3.2 % of runs, so it is roughly **+0.5 % expected score per submission** of tail
   insurance for a policy edit with zero exactness risk. Item 146 said "bundle it"; nobody did.
   It is an unpriced liability on every remaining slot. Never worth a dedicated slot; always
   worth composing.
5. **Price GDN rollback economics** (theme 5). Run the existing `sweepGatedDelta` first — it is
   nearly free and it gates everything else in that theme. Then split `rollbackRoundCount` by
   `draftCount` to get the per-width reject rate, which decides whether deleting the S=2 eager
   mid-state wins (break-even is M=2 reject probability ≈ 0.49).
6. **Audit the pre-GDN depthwise conv under multi-row verification.** Published work documents a
   silent temporal misalignment where a masked depthwise convolution over stacked candidate
   tokens captures `{t1,t1}` instead of `{t0,t1}` unless the mask is applied. This is a
   correctness question, not a speed question, and it is cheap.
7. **Census the high-scoring NON-promoted rival trees.** Their note titles are recorded but no
   diff was ever inspected: paul-hf "prefill affine QMM BM=64" at 3.2324 (which would reopen a
   door E18 closed structurally) and Lieisyourlie "bake packed-GDN q/k RMS scales as bf16
   immediates" at 3.2439 (which targets the 28 % GDN cost centre). Zero GPU.
8. **Shortlist-containment audit** (theme 6) — zero-GPU falsification of the cheapest remaining
   head lever.
9. **Close the E27 reconciliation.** −1.5511 % remains unexplained; the `e27_replica` leg never
   ran, so the crossrow-versus-wide-5 family question is open. thorfinn's P4 is the direct
   replay. Weaker than it looks: #57 settles the mixture question that gives E27 its relevance.
10. **Close or kill the NA=5 bandwidth objection** with achieved GB/s per group at the winning
    cells. Two recorded objections; only one is refuted.

### Explicitly closed — do not spend a slot

- **The entire lossless-verification-theory family is a no-op at greedy.** Block Verification,
  Traversal Verification, UniVer, hierarchical SD, multi-draft canonical decomposition and
  relatives all recover residual probability mass under *sampling*. At T=0 the target's argmax
  is deterministic and the optimal rule is already "longest matching prefix".
- **Draft trees.** Contraindicated by two independent measured lines, and our schedule is a
  chain by design.
- **KVBuffer-style deferred recurrent-state commit** as a per-row traffic win — our scan already
  has the property (theme 5).
- **Replacement head weights** — two scored rejections (theme 6).
- **A flat vocabulary crop** — measured dead at 49,152 rows (theme 6).
- **Layer-skipping self-drafting** — published α = 0.038 on sequential GDN hybrids like ours, and
  forbidden by the MTP-only rule anyway.
- **`MLX_METAL_GPU_ARCH` nax-off on the ranked leg.** Now that the ranked serial leg is known to
  be a pinned separate binary (item 176), the spoof is candidate-only and may look tempting.
  It fails the fidelity gate by construction: nax-off changes prefill GEMM rounding, which
  perturbs every downstream hidden state and top-2 value that the parent checks.

### Standing hazards, not directions

- 🔴 **Never sync the organizer frontier wholesale.** It re-introduces the EOS truncation that
  caps local windows at 302 tokens. Continuation has been added four times and lost three, every
  loss driven by a merge rather than by a decision. Cherry-pick named mechanisms only.
- Max scored verify width is **9**. M=10 bitwise deltas are a pre-existing property of the `qmm`
  splitk 9→10 padding path. **Any delta at M ≤ 9 is a hard stop.**
- The runtime-effective Metal source for the quantized family is the JIT string in
  `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`; `mlx.metallib` is never consulted
  for it and `mlx-generated/metal/quantized.h` is compiled by nothing. Always run
  `python3 research/twin_audit.py`. The GDN scan and mid-state kernels are `MLXFast.metalKernel`
  JIT strings with neither a generated twin nor metallib involvement. The scored path compiles
  with **safe math** (`setFastMathEnabled(false)`), so written accumulation trees survive.
- An instrument that cannot fail is not an instrument. Over-aggressive canonicalisers fail
  **toward the null**. Every gate needs a positive control.
- 🔴 **Open: the drafting schedule may not be deterministic across identical-source runs.**
  E51's A/A control emitted 91 and 89 rows over the same 64 positions with zero shared positions
  disagreeing, i.e. two runs of one binary proposed different drafts. askeladd's E48 width
  histogram is recorded byte-identical across 10 draws, and that histogram underpins the theme-2
  cost mixture. Both cannot describe the same scheduler. Leading suspect, unverified: the
  two-dispatch exact top-32 draft readout (`Qwen35.swift:2492-2660`) breaking a near-tie
  order-sensitively.
- Dead code on the scored path, for the next cleanup PR: the `nConfirmed > 0 && nConfirmed < S`
  split-chunk branch (`Qwen35.swift:1120-1147`), the masked scan variant
  (`GatedDelta.swift:146-152`), `gatedDeltaStepOps` (`:176+`), and `rollbackState`, which is now
  written and then cleared without ever being read.
