# Research ideas, 2026-08-22 00:45 UTC

Synthesis of four parallel research agents run against the merged experiment
ledger through E118. Written while official submission `7bef7d4c` (`xv4`) is
validating and all four students hold open assignments.

Every item below names its causal path, its measured evidence, its smallest
decisive test and its kill rule. Items are ranked by expected ranked-score value
per GPU hour, not by novelty.

---

## 1. `xr_split2` — split the row loop to buy registers back

**Status: routed to qwen-alphonse as a zero-GPU compile screen (E110 f8).**

### The observation that forces this

Finding 44 measured, on `mlp.gate_up` at NA=2/3/4/5:

| NA | load-only | ALU-only | serial sum | shipped `a_base` | shipped vs serial sum |
|--:|--:|--:|--:|--:|--:|
| 2 | 200.3 | 17.0 | 217.3 | 207.4 | **−4.6 %** |
| 3 | 202.2 | 24.7 | 226.9 | 214.4 | **−5.5 %** |
| 4 | 203.0 | 33.0 | 236.0 | 245.9 | **+4.2 %** |
| 5 | 207.0 | 39.1 | 246.1 | 292.2 | **+18.7 %** |

At NA=2/3 the kernel beats the zero-overlap serial sum, so load/ALU overlap
works. At NA=5 it is **19 % slower than running the two halves back to back**.
No throughput mechanism — issue slots, conversion units, bandwidth, MSHR depth —
can exceed the serial sum. Only stall bubbles can. This narrows the +17.3 %
round-weighted residue to two candidates.

### The two survivors

**(a) Register-slack collapse of latency hiding.** Essential live 32-bit values
in the k-loop steady state estimate to roughly `13·NA + 30` = 56 / 69 / 82 / 95,
against measured g16s allocation 70 / 93 / 94 / 95 at a 96 budget. Slack is
therefore about **40 / 27 / 14 / 1** — monotone-inverse to the gap 3.5 / 6.0 /
21.2 / 41.1. With no slack the scheduler can hold zero spills only by placing
each load just before its consumer, so the FMA chain eats exposed load latency.
This explains Finding 41's 96 % interaction term exactly: the reduced body frees
registers, loads re-pipeline, and the load price collapses to 0.93 %.

The kernel's own dispatch code already records the same effect independently.
`quantized.h:1953-1958` notes M = 7/8/9 at 319/437/216 µs and calls it "a
register cliff, not work scaling", with NA=3 vectors cheaper than NA=4 despite
strictly more work.

**(b) Instruction-cache / fetch pressure.** Unrolled body text grows with NA and
may cross a capacity boundary between NA=3 and NA=4. Supported by Finding 36
("ISA text size and spill predict time; AIR op counts do not"). The capacity
crossing is inference — no Apple i-cache size is established in this repository.

### Ruled out, with the reason

- **Occupancy from allocated registers** — allocation is 93/94/95 at NA=3/4/5,
  near-identical, while the gap runs 6/21/41.
- **Load-issue slot saturation** and **memory-queue depth** — `l_loadonly` keeps
  every load and runs 29 % *faster*; and the shape is a cliff, not linear.
- **bf16→f32 conversion throughput** — lives entirely inside the 17–39 µs
  ALU-only arm and cannot produce a 45–85 µs residue.
- **Threadgroup scheduling granularity** and the **`simd_sum` tail** — both
  cancel in paired medians and neither interacts with the arithmetic body.

### The arm

Inside each k-block of `qmv_fast_crossrow_affine4_g64_wide`, process rows
`{0,1}` then `{2,3}` as two sequential half-passes, each with its own `i`-loop,
each re-reading `x` and recomputing `sums`. `sums` does not depend on `r` and
each `acc[r]` chain keeps its own order, so the arm is **bit exact by
construction** on the same argument that carried `xv4`. Live
`packed`/`scale_local`/`bias_local`/`partial` roughly halve; text grows about
30 % and activation loads double.

Predictions diverge cleanly. Register mechanism: NA=5 recovers a large share of
the 41 %. I-cache mechanism: it *loses*, because text and loads both grow. The
sign alone discriminates.

### Kill rule for the screen

If g16s registers at NA=5 do not fall by at least 8 (95 → ≤ 87) at zero spill,
the register-relief premise is falsified; do not spend GPU time. **CAMPAIGN RULE
38 is the live trap** — if splitting the `r`-loop converts constant-index access
into indexed-array access the unroll dies and the arm is dead regardless of
register count.

### Two source questions this raised

- The scored `ntg.x == 2` path dispatches the older pair kernel
  `qmv_fast_crossrow_affine4_g64<T,2>` at `quantized.h:1923-1927`, **not**
  `wide<...,2>`. Finding 44's NA=2 row is therefore a bench instantiation, not
  the shipped path. NA=2 carries weight 0.024 so no conclusion moves, but the
  ledger should say so.
- `quantized.h:981` declares `typedef vec<float, NA> VF`, and MSL's built-in
  `vec` is documented for 1–4 lanes. How `NA=5` lowers is unresolved and the
  NA=5 liveness estimate depends on it.

### Not promotable

A `#pragma clang loop unroll(disable)` rolled-loop arm, and any text-padding arm
built to test the i-cache branch, are **diagnostic only**. Padding only adds
cost, and rolling the loop is the E110 rule-38 loser.

---

## 2. The SDPA cross-simdgroup reduction tail — strongest unassigned experiment

**Owner: none. `sdpa_vector.h` is held by no current student.**

### Why this is now ranked above every other unassigned item

The ceiling is already measured end to end. E103's ablation arm `h_tailfree_c`
deletes the tail and returns a deliberately wrong answer, giving
(W&B [`bj9zpvtw`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/bj9zpvtw)):

```
us saved / round:  N=512 362.11 | 576 360.15 | 768 341.14 | 1024 319.92
window mean 341.90 us/round  =  0.2688 % local  =  0.249 - 0.391 % ranked
```

That is above my earlier 0.239 % estimate, and the saving *falls* with N, so
there is no adverse window surprise. The tail is a fixed per-dispatch cost that
survives every future draft-depth policy change.

### The live call path, proven not assumed

The scored path executes the **single-pass** `sdpa_vector` at `qL ∈ {1,4,5}`,
N ∈ [512, 1023], 32 simdgroups live, 24·qL threadgroups per dispatch.

1. `scaled_dot_product_attention.cpp:634-637` requires `qL*gqa <= 32`; gqa = 6,
   so qL ≤ 5 and the fallback never fires.
2. `:747-753` routes two-pass only at `k.shape(2) >= 1024`; a 512-token seed
   plus a 512-token window reaches that only in the final rounds.
3. E103's own census names the live kernels
   `sdpa_vector_bfloat16_t_256_256_nomask_qnt_c_nosinks grid=24x5x1 tg=1024x1x1`
   — the single-pass instantiation. Two-pass would be named `sdpa_vector_2pass_1_*`.

`sdpa_vector.h` is `editablePaths[76]`. It has **no** `mlx-generated` twin, so it
is AOT and a metallib rebuild is mandatory. `kernels/CMakeLists.txt:18` passes
`-fno-fast-math`, so the compiler will not reassociate — an addressing-only
change is safe by construction.

### The mechanism

At `:164` each thread writes `outputs[l*32 + g]`. Within one simdgroup's store,
`g` is fixed and `l` runs 0..31, so addresses are stride-32 words — every lane in
the same bank. The read at `:166` is stride-1 and conflict-free. Executed 8
times by each of 32 simdgroups, plus 17 full-threadgroup barriers over 1024
threads and 8 dependent divides on the critical path.

### Arms, all bit exact

| arm | change | threadgroup mem | plausible share of the 25.3 % tail |
|---|---|--:|---|
| **A `k_pad`** | `outputs[BN*(BD+1)]`; index `l*(BD+1)+g` and `g*(BD+1)+l`. Store bank index becomes `(l+g) mod 32`. | 4352 → 4480 B | 30–60 % |
| **C `m_padchunk2/4`** | stage C components per barrier pair, padded; 16 barriers → 8 or 4 | 8704 / 17152 B | 50–70 % |
| **D `l_divhoist`** | move the `:167` divide into the `:172-176` epilogue | unchanged | 2–12 % |

Arm A is a five-character diff with zero arithmetic risk and it composes with
every other win. Arms rejected: reducing participating simdgroups (geometry is
owned by the non-editable `:358`, and redistribution changes summation order);
`simd_shuffle` replacement (Metal shuffles are intra-simdgroup only and this data
must cross 32 simdgroups); tree-reduce and threadgroup atomics (change reduction
order — diagnostic only).

Apply the same edit to `sdpa_vector_2pass_2` at `:338`, `:381`, `:383`; it costs
nothing and covers the kL ≥ 1024 tail rounds.

### Rungs

- **Rung 0, zero GPU, ~30 min.** Add the arms to `research/e103_sdpa_arms.metal`.
  The harness already holds a byte-faithful reference of the exact tail at
  `:121-137` and a bit-exact comparator with positive controls that scored 16/16
  in E103. Gate: every new arm bit-identical to `a_shipped_c` in all cells with
  the positive control detected.
- **Rung 1, ~1 GPU hour, no metallib rebuild and no model load.** Standalone
  `research/e103_sdpa_ab.m`, E103 palindrome protocol, cells (N,M) ∈
  {512,576,768,1024} × {5,1}.
  **Headline: best bit-exact arm's µs saved per round as a fraction of 341.90.**
  **Kill below 0.35.**
- **Rung 2, only on advance.** Edit the header, rebuild with
  `tools/build-mlx-metallib.sh --all-build-roots`, full test suite, then the
  512-token exactness run with post-EOS continuation and row-ledger closure.

### The honest risk

Against a 0.20 % ranked bar this needs f ≥ 0.51 (optimistic discount) to f ≥ 0.80
(pessimistic) — demanding for a bit-exact change that cannot delete the
reduction. Against the **target-path receipt bar of 0.1281 %** it needs only
f ≥ 0.11–0.17. This is a target-path change, so Finding 37's TARGET probe is the
correct instrument and the receipt bar is the correct comparison.

---

## 3. Beagle: per-position head-side confidence, the E99 named reopener

**Owner: none. Requires `Qwen36MTPBlockSession.swift`, held by qwen-edward until
E116 closes.**

### Why beagle and why now

Beagle carries 0.457 of published sensitivity through the rank-4 anchor
`published = 0.5·raw_beagle + 0.5·min(essays, medicine, republic, botany)`. It is
worth 12.5× essays. Its board fingerprint is R=110, mean verify width 5.382,
mean draft length 4.3818, acceptance 0.8340 — the lowest acceptance and shortest
drafts of the five drafting prompts.

Two facts must be held together:

- **The unconditional mean is already at a local optimum.** Fifteen of fifteen
  declared-head board rows that ran beagle at any mean draft length other than
  4.5327 have lower beagle `raw_p`, monotone in distance on both sides.
- **The conditional allocation still has a measured pool.** E113 rung 0 puts
  beagle's fixed-depth optimum at d\*=6 against a realised 5.382, a 2.06 % gap;
  E99 rung 8's gated oracle gap is 2.88–3.95 %.

So the lever is **which rounds go deep**, not how deep on average. Every recorded
negative constrains the mean; none constrains the allocation.

### What the scheduler cannot currently see

`qwen35DraftSelectKernel` at `Qwen35.swift:2972-3060` computes a full readout over
the 98,336-row compact draft vocabulary and reduces it to a bare argmax,
discarding `best_value` and every runner-up. **No head-side confidence signal
reaches the host at all.** The two existing margin overrides at `:1104-1111` use
the *target's* top-2 margin and stop at depth 1; positions 2–7 are priced only by
position-indexed EMAs, which are prompt-level averages.

Beagle is heterogeneous, not uniformly hard: no constant-q model reproduces its
(draft length, acceptance) pair, and E114's two-parameter fit needs margin scale
µ = 0.902 against botany's 4.432. Beagle text is dense in near-ties. Only a
context signal can capture that.

### P1 — margin into the EMA update, zero extra sync

Track `margin = best_value − second_value` through the identical shuffle
reduction (registers only; the logits are already streamed, so zero extra memory
traffic). The comparison order that picks the token id is untouched, so draft ids
stay bit-identical. Append `d` scalars to the round's **existing single eval
bundle** at `:1491-1493` and read them in the same `.asArray`. Then in
`recordAcceptOutcome` at `:1156-1186`, scale the reject penalty and the optimism
transfer by a bounded monotone function of that position's head margin.

**Causal story.** At acceptance 0.834 beagle rejects roughly one round in six,
each reject craters one EMA position by 15 %, and reach then under-drafts for
several rounds even when the underlying stretch is hot. Margin-conditioned
updates hold depth on hot stretches and concede it on cold ones — moving the
depth↔acceptance correlation without moving the mean.

**Rung 0 falsifier, zero GPU-timed cost.** Emit per-position head margins as a
trace field only, run local sessions on the reject-rich `research/e17_prose_*_512.txt`
proxies, and measure per-position AUC of head margin → acceptance using the E21
methodology at `research/e21_fit.py:421-495`.
**AUC ≤ 0.55 at positions 2–5 kills the entire reopener. AUC ≥ 0.65 licenses the
timed arm.** If the signal is dead the instrument never ships and the ranked path
is untouched.

**Safety.** Emitted tokens are always target argmaxes from the verify/serial
forwards at `:1490-1544`, so this class cannot break bit-exactness. The live
hazard is ledger item 125: `positionAcceptEMA[0] ≤ 0.18` is an absorbing state
and a depth-0 perturbation risks a ~29 % cliff. P1 does not touch depth 0.

### Lower-ranked companions

- **P2, mid-round extension on head confidence.** Largest ceiling, gated behind
  P1's AUC result, carries a host-visible sync tax on firing rounds.
- **P3, recalibrate the `/2.0` and `/3.0` sigmoid maps at `:1104-1111`.** These
  are uncalibrated Platt maps applied to a bimodal target-margin distribution
  (E99: q0.10 = 1.25 vs q0.50 = 14.25). Refit as measured calibration curves,
  never as a tuned constant — the E99 fixed-constant graveyard is adjacent.
- **P5, stale-suffix recycling after a reject.** Interesting cost profile.
  **Blocked pending a legality ruling** — the tokens are head-produced within the
  same request, but it is close enough to "token-history shortcut" that I will
  not assign it without deciding first.

Literature support, flagged as literature: AdaEDL (2410.18351) and SVIP (EMNLP
2025) are training-free draft-confidence acceptance predictors; SpecDec++
(2405.19715) proves the optimal stopping policy for candidate length is a
threshold on predicted rejection probability — exactly the marginal-rule shape
this scheduler already has, but driven by per-position confidence instead of
position averages.

---

## 4. Literature findings that change how we read our own data

### 4.1 The step-shaped verify curve may be a dispatch-plan switch, not physics

**This is the most decision-relevant literature result of the round.**

No published source reports a step-shaped `C(k)`. Every source that plots it
describes flat-then-linear. Our local step at M=5 is 3.48× and the ranked step is
1.23×, and nobody has explained either.

The batch-invariance literature (Thinking Machines' `batch_invariant_ops`; Yuan
et al. 2506.09501; TBIK 2511.17826) establishes that split-K choice, tile size and
reduction tree are **selected as a function of operand shape**. The hypothesis
that follows: **our sharp step in verify cost against row count is a dispatch-plan
switch, not a memory-system property.**

This is directly testable and it bears on live work. E117 rung 0b is already
sweeping N at fixed K to discriminate resonance from a working-set threshold; the
plan-switch hypothesis is a third arm for the same probe. The falsifiable form:
force a single fixed reduction plan and threadgroup tiling across rows 1..8 and
re-measure `C(k)`. If the step moves or flattens, it was the plan. Either answer
is decision-changing, and a fixed plan additionally buys row-count-invariant
numerics.

Caveat: reported batch-invariance overheads elsewhere are 34–62 %, but those are
general-purpose serving numbers across all shapes. We need invariance across rows
1..8 of one kernel family only.

### 4.2 Apple inverts the standard threadgroup heuristic — this validates E118

An Apple-specific FFT study (2603.27569) reports that on Apple GPUs **threadgroup
barriers are cheap (~2 cycles) while scattered threadgroup access is expensive**,
against a ~208 KiB register file with `simd_shuffle` at 1–2 cycles and only 32 KiB
of threadgroup memory drawn from the same TBDR tile pool as imageblocks.

Two consequences:

- It independently predicts **why `xs_stage` and `xv4_stage` lost**: we staged
  into the expensive resource.
- It independently predicts that **E118's `s_bcast` is the right primitive** —
  one lane loads the group-64 scale/bias and `simd_broadcast`s it, avoiding both
  the threadgroup round trip and 63 redundant global loads. The transferable-
  looking CUDA recipe (shared memory + `__syncthreads`) is the wrong recipe here.

This raises my prior on E118 arm 1 materially.

### 4.3 A measurement hazard: bimodal GPU clock state

The same body of Apple microbenchmark work (2606.12765, M4 Max) reports explicit
clock-state bimodality — a rare low-clock tail at roughly **2.7 % incidence
running near half speed**. At our effect sizes an uncaught low-clock leg is a
2× outlier in a 6-replicate ABBA. This sits alongside harness defect 16 (the
macmon DVFS ramp) as a second, independent clock-state hazard.

**Action:** every timed arm should report per-leg dispersion, not just the mean
and CI, and any leg more than ~1.5× the arm median should be flagged in the
result rather than silently pooled. Cheap, and it would have caught this class in
any of the last six timed experiments.

### 4.4 One genuine M5 datapoint

`Kernel Contracts` (2604.22032) reports incidental **Apple M5** measurements:
batching k matvecs into one matmul is 3–17× faster, above 10× median, with the
crossover to sub-1 % overhead at n = 16,384. Weak, but it is the only M5 GPU
number in the indexed literature and it argues that the batched-rows path is
strongly favoured on our ranked host.

**Gap to record: no published GPU microbenchmark of the M5 generation exists.**
Every Apple GPU measurement found is M1–M4. Every cross-host inference we make
from g16s to g17s remains unvalidated by anything external.

### 4.5 Structural GDN ideas, parked with their reasons

- **Weaver (2607.06763)** — rollback-free delta-rule tree verification on a 27B
  GDN target: verification becomes a *read-only* masked triangular solve against
  the pre-round state, and the accepted prefix is committed by a short recurrence
  replay at the start of the next round. This deletes the snapshot cost **and**
  the rollback cost together, across 48 GDN layers. Largest structural lever
  found. Parked because E106 measured the GDN tail at ~0.03 % and E96 put the
  whole recurrent step at 0.67 % of the round — the prize may be small even if
  the mechanism is sound. **Reopener: a measured snapshot+rollback cost above
  0.30 % of the round.**
- **STree (2505.14969)** — a useful **negative**. Its accumulated-state-transition
  trick works because Mamba transitions are diagonal. Gated DeltaNet's
  `I − βkkᵀ` does not commute and admits no cumulative-product form. **Do not
  attempt to port it.**
- **Traversal Verification (2505.12398)** — gains come from leaf-to-root joint
  acceptance under *sampling*. Under greedy exact-match against a fixed serial
  trajectory the accepted set collapses to the longest correct prefix and the
  advantage disappears. Non-transferable to our contract.

### 4.6 Explicitly excluded, do not spend GPU time

Everything batched (Speculative Verification's scheduler, 2510.22876, 2505.07858,
2310.18813) — conclusions frequently invert at batch 1. Everything needing a
trained depth predictor (DISCO 2405.04304, Yggdrasil's MLP, 2603.01639) —
distribution-shift bets against 8 hidden prompts. LUT dequantisation (LUT-GEMM,
FLUTE, T-MAC) — affine 4-bit g64 already dequantises in ~2 ops/element and a LUT
would change the transform contract for an ALU saving on a bandwidth-bound loop.
All FPGA/NPU/ASIC results — dataflow framings only. Any M4 tensor-path assumption
carried to M5.

One flagged warning with low confidence: 2605.01106 claims that for *sequential*
hybrids an SSM/linear-only self-draft achieves α = 0.038 at k = 2 versus 0.68 for
parallel hybrids. If true it would mean "skip the 16 full-attention layers to get
a free drafter" fails catastrophically on our architecture family. The source is
weak, but the claim is cheap to test locally and expensive to discover late.

---

## Assignment order when a student frees

1. **SDPA tail, arm A `k_pad`** — highest measured ceiling among unassigned work
   (0.249–0.391 % ranked), a five-character bit-exact diff, an existing harness
   with existing positive controls, a zero-GPU rung 0, and an unowned file.
2. **Beagle P1 rung 0** — the AUC gate is trace-only and cannot regress the
   ranked path. Needs `Qwen36MTPBlockSession.swift`, so it waits for E116.
3. **`xr_split2` timed arm** — only if alphonse's compile screen shows the
   register drop.
4. **The bf16→f32 conversion count in the wide-QMV inner loop** — alphonse's
   follow-up 1; the only quantity that scales with NA inside the interaction
   term. Now ranked below `xr_split2` because the roofline analysis puts
   conversions inside the 17–39 µs ALU-only arm, which cannot produce the
   residue.
5. **The fixed-dispatch-plan arm for `C(k)`** — folds into E117's N sweep at
   near-zero marginal cost if thorfinn's rung 0b is still open.
