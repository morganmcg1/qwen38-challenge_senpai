# E42 results — ψ and φ by injected bit-exact regression

Assignment: PR #47, `qwen38-r1-e42-psi-phi-by-injected-regression`, revision `r1`.
Pre-registration: `research/e42-prereg.md` (commit `dc379d5`, before any arm ran).
W&B: [`bitem8ak`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/bitem8ak)
(4 tables, 90 summary keys).

**Nothing in E42 ships.** The final commit reverts every kernel edit and proves
`git diff 04ad6bf11437c269df85a47e91faa769c74fe6da HEAD -- Sources Vendor benchmark.json`
is empty. E42 is a measurement instrument, not a candidate.

## 0. Read this first: the order statistic

The published score is the **median** of eight per-prompt ratios, so it is an
order statistic, not a mean. On the reference ranked row only **beagle (79 %
weight) and medicine (21 %)** move it; essays contributes **+0.0000 %**. Every
number below is a *mechanism* measurement on a local M4 Pro against one public
fixture. None of it is a score claim, and none of it is gate-qualified. The
sensitivity of the score to a uniform both-leg MTP speedup is **1.00**; the
0.4827 figure is beagle-alone.

## 1. What was asked and what was delivered

`score gain = sensitivity × ψ × φ × x`. ψ — the share of the candidate leg that
the quantised-matvec (QMV) family accounts for — had never been measured. Its
only prior number was a **time attribution of 59 %**, which is an *upper bound*
on ψ, not a measured sensitivity. Nothing here is tuned toward 59 %.

Direct measurement of ψ fails on the minimum detectable effect (MDE): a
realistic kernel win is smaller than the local two-arm MDE of **0.4315 %**. The
design escapes that trap by *injecting* a large, **bit-exact** regression of
known kernel magnitude `x` and dividing:

```
psi_eff = (measured leg slowdown) / (measured kernel slowdown)
```

Both numerator and denominator are large, so both clear the MDE by two orders of
magnitude. The regression is redundant recomputation — a rolled pass loop that
recomputes the identical accumulation `E42_PASSES+1` times — so the emitted
tokens are unchanged by construction.

| deliverable | status |
|---|---|
| p2 arm (M≥2 → ψ) at ≥2 magnitudes | **done**, L1 and L2 |
| p6 arm (M≥6 → ψ·φ) at ≥2 magnitudes | **done**, L1 and L2 |
| linearity across magnitudes | **done** for both p2 and p6 (the scientific core) |
| m1 sign-flip control | **done**, raw_p flipped up |
| m6 single-width arm (priority B) | **done** |
| bit-exactness, end to end on the scored path | **done**, every arm |
| bit-exactness, 192-cell parity rig | **done**, 6/6 arms BIT-IDENTICAL |
| denominator measured per width, `--shapes-only` | **done**, no ranked-geometry replay |
| ψ and φ reported separately, with intervals | **done** |
| requirement #4: marginal vs occupancy | **done**, `asyncEval` ladder |
| MDE for every null, unmodified `research/e39_mde.py` | **done** |
| fixture pinned, `effectiveDraftLengths` element-wise equal | **done** |
| revert proof: empty diff against base | **done** at `c8e41e9` |

Every deliverable is measured. The sections below state exactly which arm each
number comes from.

## 2. Provenance

| field | value |
|---|---|
| base | `04ad6bf11437c269df85a47e91faa769c74fe6da` |
| host | local aws-mac **M4 Pro**, 48 GiB (ranked runner is **M5**) |
| fixture | `correctness_prompts/public_longcopy_gate_english_512_256.json`, sha256 `3d922b1a0ada04d9827b905c881232bf50fb697d4be9ab3ee21346f7e0b8ae9c` |
| head | declared `mtp-head-declared-run`, safetensors sha256 `d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1` |
| depth | `MLXFAST_QWEN_MTP_DEPTH=8` |
| decode window | 512 tokens, both legs |
| thermal | `MLXFAST_LOCAL_COOL_GATE=0`, ABBA-counterbalanced within session |
| gate flags | `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, `official_or_ranked_score=false` |

Every arm reports entry/exit GPU temperature in
`.mlxfast-private/e42/runs/<tag>/meta.txt`. The ungated mode is the standing
permitted local measurement mode; these are directional causal results within a
counterbalanced session, **not** comparable to gated historical runs.

Register counts in this report were measured on tree **`04ad6bf1`**. On the
rebased advisor base the same ladder's max is 108 and the production entry
`affine_qmv_fast<bfloat16_t,64,4,false>` is 163 — different tree, different
numbers. **E27 is present in base `04ad6bf1`** (M=5 and M=9 both IPG=5); the
advisor reverted it on the newer base, so any comparison across those two trees
is not like-for-like. The retired-family hunks guarded on
`physicalMemory >= 96 GiB` were **inactive in every measurement here**, because
this box is 48 GiB.

### 2.1 Which table these numbers were measured on

The advisor asked me to choose between staying on `04ad6bf1` and rebasing onto
the tip, and declined to choose for me. **I chose to stay on `04ad6bf1`.**

Reason: all seven arms, the base curve, the ladder slopes, the cross-arm
intercept check, and the 1152 parity comparisons are one internally consistent
session on one tree. A rebase would invalidate the base arm and every ratio
taken against it, and would cost seven arm re-runs plus two parity groups —
roughly five hours of exclusive GPU time — to reproduce numbers whose scientific
content is a *decomposition*, not a candidate. Nothing here ships, so the value
of these numbers is the structure they expose, and that structure is strictly
richer on this tree: `04ad6bf1` has exactly one stream boundary in the
dispatched range, so a single ladder separates "crosses a boundary" from "does
not" on both sides. On the tip the one boundary I could measure at M=6 is
structurally empty.

Accordingly, every per-width number in this document carries this label:

> measured on the E27 table `04ad6bf1`, boundary 5→6

What transfers to the tip and what does not:

| quantity | transfers? | why |
|---|---|---|
| ψ = QMV share of the MTP leg | **yes, approximately** | a share of leg time; the QMV family and its call counts are identical on both trees |
| ψ falls with width | **yes** | driven by non-QMV growth (6.273×) outpacing QMV growth (2.240×), neither of which is an IPG property |
| φ_local(M≥6), φ_local(M=6) | **yes, as shares** | width shares of *this fixture's* histogram; independent of the IPG table |
| Q(M) absolute ms/round | **no** | register ceiling differs at every width (129 here vs 108 shipped) |
| ΔQ(M) increments | **no** | the boundary is at 5→6 here and at 4→5 and 8→9 on the tip |
| "the quadratic is refuted" | **yes** | a sign-change argument on measured second differences; it needs *a* boundary, not this boundary |
| decode-time share of M∈{5,9} | **no** | width 5 is single-stream here and two-stream on the tip |
| the 5.70 %/3.74 % threshold straddles | **no** | they are built from this tree's C(M) |

The one thing I would most like and do not have is the tip-side ladder. Together
the two rows would form a 2×2 in which the bend moves with a template argument,
which no smooth function of M can do. I did not measure it and do not claim it.

## 3. Scope and the injection

Edited files, both submitted paths, twin-locked at offset +13:

- `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
- `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`

This family is **JIT-only**: the runtime-effective source is the kernel text
inside the C++ translation unit, so both twins must move together.
`python3 research/twin_audit.py quantized` passes on every arm commit.
`tools/build-mlx-metallib.sh` contains no `quantized` reference, confirming the
metallib path is not the one that matters here.

Untouched, deliberately: host dispatch (`backend/metal/quantized.cpp`, **not in
`editablePaths`**), `mtp-head.manifest.json`, depth-policy constants,
`AttentionUtils.swift`, `sdpaWidthWallDepthCap`, `segmentedStreakGate`.

Budget: `source=2457141/3000000`, `growth=1852/262144`;
`validate-assignment-scope.sh` reports `assignment scope OK: 2 submitted path(s)`.

`M` reaches the kernel as `ntg.x` (host sets `grid_dims(M, ceil(N/8), B)`), and
all M-dependent selection happens **in-kernel** in the `affine_qmv_fast` ladder.
There is one PSO per `(mode, fast, dtype, group_size, bits, B>1)`, so a
width-selective injection needs no host change — which is exactly why this
experiment fits inside the editable surface.

| arm | treated widths | mechanism |
|---|---|---|
| `p2` | 2..9 | every drafting width → ψ |
| `p6` | 6..9 | the deep-width set → ψ·φ(M≥6) |
| `m6` | 6 only | the single width the ranked corpus cannot isolate |
| `m1` | 1 only, via the `ntg.x == 1` fall-through | control; the serial leg is *entirely* width 1 |

`m1` treats a **distinct template instantiation** of `qmv_fast_impl`, so the
untreated widths are provably untouched rather than argued to be.

## 4. Static gates: the redundant work provably executes

`research/e42-artifacts/air-dce-gate.txt`. The risk is that the compiler deletes
the redundant pass, making `x` a fiction. Evidence it does not:

- the pass loop **stays rolled**: `loop_backedges` increases by exactly the
  treated-cell count, and per-body `float_ops` / `device_loads` are unchanged;
- `peak_live_regs` is **flat** versus base, so the injection does not perturb
  occupancy as a side effect.

Real scored PSO `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` on tree
`04ad6bf1`:

| metric | base | p2L2 | p6L1 |
|---|---|---|---|
| `peak_live_regs` | 208 | 208 | 208 |
| allocas | 110 | 110 | 110 |
| fmul | 225 | 225 | 225 |
| fadd | 280 | 280 | 280 |
| `float_ops` | 505 | 505 | 505 |
| `device_loads` | 169 | 169 | 169 |
| loads | 472 | 472 | 472 |
| **`loop_backedges`** | **212** | **230 (+18)** | **222 (+10)** |

Only the backedge count moves, and it moves by exactly the number of treated
cells. That is the signature of "same body, executed more times".

A cheap pre-flight Metal compile
(`xcrun -sdk macosx metal -std=metal3.1 -O2 -c ... quantized.metal`) compiles
every width case the JIT builds, in ~24 s, and runs before every arm.

## 5. ψ — the QMV share of the candidate leg

All arms share the identical decode trajectory: `all_tokens_matched=true`,
`residual_divergence_count=0`, row ledger closes at `declared_rows_total=567`,
accept rate 0.8875, mean draft 6.2692, 78 rounds, and
`effective_draft_lengths` **element-wise equal** across every arm. Width
histogram identical everywhere: `M2:1 M4:5 M5:5 M6:23 M7:4 M8:6 M9:34`. M=3 is
never dispatched on this fixture.

| arm | commit | treated | MTP s | serial s | raw_p | x̄ (kernel) | MTP leg Δ | ψ_eff | α |
|---|---|---|---|---|---|---|---|---|---|
| base | `8ec793f` | — | 16.266 | 37.662 | 2.3154 | — | — | — | — |
| p2L1 | `bf64ead` | 2..9 | 26.149 | 37.621 | 1.4387 | +0.9030 | +60.757 % | **0.6729** | 1.0015 |
| p2L2 | `afc8916` | 2..9 | 36.188 | 37.583 | 1.0386 | +1.8180 | +122.470 % | **0.6736** | 1.0027 |
| p6L1 | `6b8ae93` | 6..9 | 25.393 | 37.583 | 1.4801 | +0.9129 | +56.105 % | **0.6145** | 1.0023 |
| p6L2 | `78cd88d` | 6..9 | 34.545 | 37.583 | 1.0879 | +1.8321 | +112.372 % | **0.6133** | 1.0004 |
| m6L2 | `04f28ac` | 6 only | 21.684 | 37.603 | 1.7341 | +1.8261 | +33.308 % | **0.1824** | 1.0053 |
| m1L1 | `d984b45` | 1 only | 16.555 | **51.376** | **3.1033** | +0.4271 | +1.774 % | 0.0415 † | — |

† m1's primary leg is the **serial** one; see §5.1. Its MTP-leg figure is the
draft head's width-1 share, not a verify-width share.

Serial-leg drift across the five arms that do not treat width 1: −0.111 %,
−0.210 %, −0.210 %, −0.211 %, −0.159 % — all well inside the 0.4315 % MDE, as
required, since the serial leg is entirely width 1 and therefore untreated in
every one of them. m1 is the exception by construction and moves it +36.411 %.

The spread of `x` over the treated widths is the gate-tightness signature and it
tracks the gate exactly: p2 treats seven dispatched widths and reads 0.4260 /
0.7210, p6 treats four and reads 0.0063 / 0.0104, and m6 treats exactly one and
reads **0.0000** because a single-width arm has nothing to spread over.

`α` is the additivity check: predicted-over-observed leg time under the
assumption that the injection adds cost without changing anything else. α ≈ 1 to
0.3 % on every arm, so the injection behaves as pure added work.

### Linearity — the scientific core

ψ must not depend on the injection magnitude. Measured at two magnitudes a
factor ~2 apart in kernel cost, on **both** gated families:

| family | ψ_eff(L1) | ψ_eff(L2) | ratio |
|---|---|---|---|
| p2 (M≥2) | 0.6729 | 0.6736 | **1.0012** |
| p6 (M≥6) | 0.6145 | 0.6133 | **0.9980** |

0.12 % and 0.20 %, versus a 0.4315 % MDE. If the instrument were mis-specified —
DCE, occupancy change, cache-pressure nonlinearity, or a leg-time term I had not
accounted for — ψ would drift with magnitude. It does not, to a fifth of a
percent, and it does not on either gate. That the two families' ratios bracket
1.0 from opposite sides is what a noise-limited estimate of a true constant looks
like; a systematic magnitude effect would push both the same way.

### An independent ψ that predicts no absolute cost

The ratio estimator still leans on the curve's absolute calibration. A second
estimator avoids that entirely: regress leg time on the *ladder slope* only.

| family | Q_treated from slope | pairwise spread | ψ from slope | non-QMV intercept | curve prediction |
|---|---|---|---|---|---|
| p2 | **10.9577 s** | 0.230 % | **0.6736** | 5.3087 s = **68.06 ms/round** | 10.9284 s (**+0.268 %**) |
| p6 | **9.9770 s** | 0.391 % | **0.6133** | 6.2894 s = **80.63 ms/round** | 9.9731 s (**+0.039 %**) |

Two estimators with different failure modes agree to a quarter of a percent on
p2 and to four hundredths of a percent on p6.

#### The cross-arm intercept check

The two intercepts above are the strongest validation in this experiment,
because they were not fitted to the quantity they predict. Each intercept comes
from **leg timing alone**; the gates are compile-time template selections; so the
*difference* of the two intercepts must equal the **curve-measured** QMV cost of
exactly the widths that separate the two gates — dispatched widths {2, 4, 5}:

| quantity | value |
|---|---|
| curve prediction, 1×Q(2) + 5×Q(4) + 5×Q(5) over 78 rounds | **12.248 ms/round** |
| measured difference of two independently built arms' intercepts | **12.574 ms/round** |
| ratio | **1.0266** |

A leg-only quantity predicts a curve-only quantity to 2.7 % with no shared code
path and no shared fit. Reproduced by
`research/e42_analyze.py` as `cross_arm_intercept_check`.

### ψ interval over three denominators

The denominator is measured per width on `--shapes-only` curves; no ranked
geometry is replayed. Three defensible denominators give:

| arm | as measured | drift corrected | stable shapes only | interval |
|---|---|---|---|---|
| p2L1 | 0.6729 | 0.6040 | 0.6691 | [0.6040, 0.6729] |
| p2L2 | 0.6736 | 0.6592 | 0.6739 | [0.6592, 0.6739] |
| p6L1 | 0.6145 | 0.5541 | 0.6125 | [0.5541, 0.6145] |
| p6L2 | 0.6133 | 0.5665 | 0.6029 | [0.5665, 0.6133] |
| m6L2 | 0.1824 | 0.1722 | 0.1942 | [0.1722, 0.1942] |

**Preferred ψ ≈ 0.672**, interval **[0.659, 0.674]** from the tightest arm.

The drift term is worth naming honestly, and it is not pure noise. Every arm
carries the same *structural* template change (the extra `E42_PASSES` parameter
and the surrounding loop) at **all** widths, including untreated ones, where it
is inert because `E42_PASSES = 0` gives exactly one pass. So a small non-zero
reading at an untreated width is expected: it is the cost of the inert
restructuring, and the calibration cell is there to price it.

What is *not* explicable that way is the sign. The untreated M=1 calibration cell
reads −5.13 % (p2L1), −1.39 % (p2L2), −4.95 % (p6L1), −5.08 % (p6L2),
−3.70 % (m6L2): an inert
restructuring cannot make the kernel meaningfully *faster*. It is **one shape**:
`linear_attn.in_proj_fused_qkvzba` at −24.61 %, whose *within-curve* spread is
32.6 % (base) to 146.2 % (arm) — i.e. that cell is not a stable measurement at
all. Restricted to reproducing shapes the same calibration reads **+0.38 %,
+0.40 %, +0.73 %** — small, positive, and exactly the sign the inert
restructuring predicts. I report the interval over all three denominators rather
than silently dropping the unstable shape.

### 5.1 The m1 control, predicted before it was run

Every arm so far slows the MTP leg and leaves the serial leg alone. That is a
one-directional check: it shows the injection reaches the widths it targets, but
not that it *misses* the widths it does not target. The `m1` arm inverts the
test. It treats width 1 only, and the serial leg runs at depth 0, so **every one
of its rounds is a single M=1 target row**. Predictions, recorded here before the
arm was measured:

1. **raw_p must go up, not down.** `raw_p = serial / MTP`, so slowing the
   numerator raises it. p2, p6 and m6 all pushed raw_p down; m1 must push it up.
   The analyzer now asserts this direction per family rather than leaving it to
   prose (`raw_p_sign_expected` vs `raw_p_sign_observed`).
2. **ψ(serial) should be high, near 0.87.** The base serial leg is 37.66 s for
   512 tokens = 73.6 ms per round, and the curve puts Q(1) = 64.433 ms of QMV
   work in a width-1 forward. That is 64.433 / 73.6 ≈ **0.875**, i.e. a depth-0
   decode is almost entirely quantised matvec — which is what a memory-bound
   4-bit single-token decode should look like.
3. **So the serial leg should slow by roughly 0.875 × x(1) ≈ +79 % at L1**, an
   effect ~180× the 0.4315 % MDE.
4. **The MTP leg should move only a little**, and for a specific reason: width 1
   never appears as a *target verify* width there (`nd = 0`), so the only
   width-1 QMV work in the MTP leg is the draft head's autoregressive calls.
   Whatever the MTP leg does move by is therefore a measurement of the
   **draft-head width-1 share** — a quantity nothing else in E42 prices.
5. **MTP-leg occupancy is not identified for m1**, because the round histogram
   contains no width-1 rounds to weight. The analyzer says so explicitly
   (`mtp_leg_has_treated_verify_width: false`) instead of emitting a number it
   cannot support.

Prediction 2 is the interesting one to be wrong about: if ψ(serial) came out
well below 0.87, it would mean the curve's absolute calibration is off in a way
the MTP-leg arms happen not to expose.

#### Scored, after the fact

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | raw_p rises | 2.3154 → **3.1033**, +0.7880 | ✅ |
| 2 | ψ(serial) ≈ 0.875 | **0.8525** | ✅ within 2.6 % |
| 3 | serial slows ≈ +79 % | **+36.411 %** | ❌ **point value wrong** |
| 4 | MTP leg barely moves | **+1.774 %** | ✅ |
| 5 | MTP occupancy not identified | reported as unidentified | ✅ |

**Prediction 3 is a real miss and stays on the record as one.** I assumed x(1)
would land near the ≈0.90 the other arms measured. It did not: width 1 runs the
**non-crossrow `qmv_fast_impl` fallback**, a different kernel with a different
work profile, and it measured x(1) = **+0.4271**. The prediction's *form* is
intact — 0.875 × 0.4271 = **+37.4 %** against **+36.411 %** measured, agreeing to
2.6 % — so the coefficient was right and the denominator guess was wrong. Had I
written the prediction as a coefficient rather than as a leg percentage it would
have been correct; that is the lesson, not the number.

The leg asymmetry is the result that matters: **+36.411 % serial against +1.774 %
MTP, a factor of 20.5, from an injection confined to one template
instantiation.** Six arms moved raw_p down and this one moved it up. ψ is an
attribution, not a global slowdown artefact — and nothing else in this experiment
could have established that.

α(serial) = **0.9733** is the only α below 1 anywhere here, i.e. the serial leg
absorbs ~2.7 % of an injected width-1 slowdown where the MTP leg absorbs none.
That is the small overlap effect requirement #4 anticipated, appearing on the leg
with the *least* work to overlap and 512 rounds to hide it in.

Trajectory unchanged (`trajectory_identical_to_base: true`) and
`all_tokens_matched: true`, so the 36 % serial slowdown did not perturb decoding.

## 6. φ — but per width, not pooled

Pooling over M≥6 hides the question. Ranked telemetry brackets φ(M≥6) to
0.2351–0.9380 on beagle, but **φ(M=6) alone only to 0.0000–0.9701** — it cannot
be bounded away from zero from ranked data at all.

**The m6 arm answers this.** It injects into width 6 *alone* — a distinct kernel
instantiation, `x̄ = +1.8261` with **spread 0.0000** because a single width has no
spread — and produces a signed end-to-end leg effect of **+33.308 %** with the
trajectory unchanged.

| quantity | measured | ranked bracket | verdict |
|---|---|---|---|
| φ_local(M≥6) | **0.9105** (L2), 0.9132 (L1) | 0.2351–0.9380 | inside, near the top |
| **φ_local(M=6)** | **0.2708**, interval [0.2612, 0.2882] | 0.0000–0.9701 | **bounded away from zero** |
| φ_local(M∈{7,8,9}) | **0.6397** by subtraction | — | — |

φ_local(M=6) = ψ_eff(m6L2) / ψ_eff(p2L2) = 0.1824 / 0.6736. The ranked corpus
cannot bound φ(M=6) away from zero at all; a single injected width pins it to
±5 % of its own value.

φ_local(M≥6) is now measured at both magnitudes — 0.9132 at L1 and 0.9105 at L2,
agreeing to **0.30 %** — so the pooled φ carries the same linearity guarantee as
ψ. Cross-level use of the L1 and L2 arms is licensed by that result and by §5.

The interval on φ(M=6) applies the drift correction **consistently to numerator
and denominator**, because it is a common-mode correction on a shared calibration
cell: as-measured 0.1824/0.6736 = 0.2708, drift-corrected 0.1722/0.6592 = 0.2612,
stable-shapes 0.1942/0.6739 = 0.2882. Pairing the extreme numerator with the
opposite-extreme denominator — which double-counts a correction that cancels in
the ratio — would give the wider [0.2555, 0.2946]; that bound is also above zero,
so the conclusion does not depend on the choice.

φ_local sits near the top of the pooled ranked bracket because this fixture is a
width **ceiling**: its 0.8875 accept rate saturates drafting at depth 8, giving
mean M 7.27 against beagle's 5.53. This is *local* φ. Ranked ρ(M) is edward's.

Per-width shares from the isolated `--shapes-only` curve. These are **ψ·φ(M)**,
width M's QMV share of the *whole leg* — not ψ at width M. Measured on the E27
table `04ad6bf1`, boundary 5→6.

| width | ψ·φ(M) from curve | ψ_eff measured end-to-end | agreement |
|---|---|---|---|
| M=5 | 0.0294 | — | — |
| **M=6** | **0.1814** | **0.1824** (m6L2) | **0.55 %** |
| M=9 | 0.3427 | — | — |

The curve and the end-to-end injection are independent instruments — one times
isolated kernel dispatches with a raised op-per-buffer fence, the other times a
512-token decode leg — and they agree on the width-6 share to **0.55 %**. That
is the strongest internal validation in this experiment.

🔴 **A labelling error worth recording, because it inverted a direction.** An
earlier draft called this column "per-width ψ" and concluded that ψ *rises* with
width. The numbers are right; the label and the conclusion were not. ψ·φ(M)
rises with width because the round count and per-round cost at M rise. ψ within
a round does the opposite — see §8.1: non-QMV grows 6.273× from M=1 to mean
M 7.27 while QMV grows 2.240×, so ψ **falls** as width rises, from 0.8525 at
M=1 to 0.6736 at mean 7.27. Both of those are measured, from different arms
(m1 and p2), not modelled. The consequence is favourable and is why §12 states
ψ = 0.672 as a *lower* bound for beagle, whose mean M is 5.53.

### 6.1 The mechanism of the M=6 step

In the `_m` kernel:

```
const int first_m = int(tid.x) * IPG;
if (first_m >= M) return;
```

Working groups = `ceil(M / IPG)`, and **each group streams the whole weight
matrix**. With the base IPG table `{3:3, 4:4, 5:5, 6:3, 7:4, 8:4, 9:5}`:

**weight streams = 1 for M ≤ 5, and 2 for M ≥ 6 — on this base.**

That is a structural discontinuity, not a smooth cost curve. It is independently
corroborated by the committed `research/e34-ranked-operating-point.json`
(`/dispatch/weight_passes/*`), which also records `weight_passes_pre_e27` of
M=5→2 and M=9→3 and `single_pass_top_width_now = 5`: E27's IPG 3→5 bought
exactly one weight stream at each of M ∈ {5, 9}.

#### 🔴 The step location is a property of the IPG table, not of the width

This is the single most transferable thing in this document and it is easy to
get wrong. `streams(M) = ⌈M / IPG(M)⌉` is read straight out of the `case M:`
switch, so **changing a template argument moves the boundary**. My base
`04ad6bf1` still carries E27; the live tip has it reverted, and the two trees
therefore have different boundaries:

| case | base `04ad6bf1` | streams | live tip (E27 reverted) | streams |
|---|---|---|---|---|
| 3 | `<T,3,3>` | 1 | `<T,3,3>` | 1 |
| 4 | `<T,4,4>` | 1 | `<T,4,4>` | 1 |
| 5 | **`<T,5,5>`** | **1** | `<T,5,3>` | **2** |
| 6 | `<T,6,3>` | **2** | `<T,6,3>` | 2 |
| 7 | `<T,7,4>` | 2 | `<T,7,4>` | 2 |
| 8 | `<T,8,4>` | 2 | `<T,8,4>` | 2 |
| 9 | **`<T,9,5>`** | **2** | `<T,9,3>` | **3** |

Read from source with `git show 04ad6bf1:…/kernels/quantized.h`, not from notes.

So on this base `streams(M) = 1,1,1,1,2,2,2,2` for M = 2..9: the **only** stream
boundary in the dispatched range is **5→6**, and there is no 2→3 boundary at all.
On the tip the boundaries are 4→5 and 8→9 and the 5→6 increment is structurally
empty. **Every Q(M) figure in this document is an E27-present `04ad6bf1` number
and does not transfer to the tip.**

The useful consequence is that the model becomes two-sided on this tree, which is
a sharper test than a same-tree replicate: the mechanism predicts a step at 5→6
*and no step* at 4→5 or 8→9, whereas any smooth function of M predicts no such
asymmetry.

| increment | streams here | **measured ΔQ (ms)** | streams on tip |
|---|---|---|---|
| 2→3 | 1→1 | +5.54 | 1→1 |
| 3→4 | 1→1 | +9.89 | 1→1 |
| 4→5 | 1→1 | +13.31 | **1→2** |
| **5→6** | **1→2** | **+32.80** | 2→2 |
| 6→7 | 2→2 | +9.86 | 2→2 |
| 7→8 | 2→2 | +10.91 | 2→2 |
| 8→9 | 2→2 | +14.87 | **2→3** |

The one increment that crosses a boundary here is 2.2×–5.9× every increment that
does not. The two increments that cross on the *tip* but not here are ordinary.

Only the `04ad6bf1` row of that design is mine. A tip-side per-width measurement
would complete a 2×2 in which the bend moves with a template argument — and no
smooth function of M can do that — but I have not measured the tip and do not
claim it.

#### What this does *not* say

The m6 injection arm measures **φ(M=6), a share**, and M=6 carries 23 of 78
rounds here, so that share is non-zero whatever the stream count is. It is the
**ladder increments above**, not the injection arm, that locate the boundary.
Those are separate instruments answering separate questions and conflating them
would make a share look like structural evidence.

#### The arms alone reproduce the boundary ratio, with no curve and no model

The strongest form of the boundary result does not use the isolated curve at all.
p2 treats widths 2..9 and p6 treats widths 6..9, and both are measured against
the same base leg, so the **difference of their ψ_eff is exactly the widths that
separate the two gates** — dispatched widths {2, 4, 5}, which on this tree are
precisely the single-stream widths (M=3 is never dispatched).

| quantity | value | source |
|---|---|---|
| ψ_eff(p2), widths 2..9 | 0.6736 | leg timing |
| ψ_eff(p6), widths 6..9 (two-stream) | 0.6133 | leg timing |
| difference = widths {2,4,5} (single-stream) | **0.0603** | subtraction only |
| two-stream rounds | **67 / 78** | fixture histogram |
| single-stream rounds | **11 / 78** | fixture histogram |

Per-round cost ratio of a two-stream width to a single-stream width:

```
(0.6133 / 67) / (0.0603 / 11) = 1.670×
```

The isolated curve, which shares no timing code with the legs, gives the same
ratio as a round-count-weighted mean of Q(M):

```
two-stream   (23·128.316 + 4·138.173 + 6·149.087 + 34·163.958) / 67 = 148.852 ms
single-stream           (1·66.771 + 5·82.205 + 5·95.514) / 11      =  86.851 ms
ratio = 1.714×
```

**Agreement 2.6 %.** This is the third independent instrument to land on the
boundary — ladder increments, the m6 injection arm, and now a pure subtraction of
two leg measurements — and the only one that needs neither a fitted model nor the
`--shapes-only` harness. It also fixes the weight of the headline ψ: the
cost-weighted two-stream share is 9973.054 / 10928.420 = **0.9126**, so
**ψ = 0.672 is a two-stream ψ carrying about 91 % of the weight**, and any
single-stream-heavy prompt sits outside it.

### 6.2 Priority B: the step is real, and a quadratic is refuted

`research/e42_width_census.py`, artifact
`research/e42-artifacts/width-census.json`.

Measured base curve Q(M), ms per round over the 7 verify shapes:

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Q(M) | 64.433 | 66.771 | 72.316 | 82.205 | 95.514 | **128.316** | 138.173 | 149.087 | 163.958 |

First differences: +2.34, +5.54, +9.89, +13.31, **+32.80**, +9.86, +10.91, +14.87.
Second differences: +3.21, +4.34, +3.42, **+19.49**, **−22.95**, +1.06, +3.96.

**A quadratic has constant second differences by construction. The measured
second differences change sign twice. The quadratic is therefore refuted
arithmetically, independently of any fit residual.**

Fitting both families to the nine measured widths:

| model | k | rms (ms) | max\|res\| | predicted T(6)−T(5) |
|---|---|---|---|---|
| linear | 2 | 7.716 | 12.539 | 13.715 |
| quadratic | 3 | 6.165 | 12.344 | 14.508 |
| cubic | 4 | 3.856 | 8.216 | 18.635 |
| **step(M≥6)+linear** | 3 | **3.498** | 6.387 | **36.782** |
| **step(M≥6)+quadratic** | 4 | **1.822** | 3.176 | **34.671** |
| **passes×linear (mechanistic)** | 4 | **2.497** | 3.932 | **35.441** |

At equal parameter count the step form fits **1.76× tighter (k=3)** and **2.12×
tighter (k=4)**. And the decisive number:

**measured Q(6) − Q(5) = 32.802 ms.**

The ranked corpus could not separate these families: both fit it with zero slack
while disagreeing 4.4× on `T(6)−T(5)` (37.730 ms vs 8.575 ms). The measured
32.802 ms sits with the **step** family and is 3.8× the quadratic's prediction.
Since Q is a *component* of T and non-QMV work cannot decrease with M,
`T(6) − T(5) ≥ 32.8 ms`.

That is the discrimination the ranked corpus cannot touch, and it is consistent
with edward's derived pins (`beagle=107, medicine=99, essays=87, republic=89,
botany=85`), which I treat as solid.

## 7. Priority A side-product: the M∈{5,9} decode-time share

Full detail in `research/e42-artifacts/width-census.json`. Summary:

- **Measured half** — cost weights C(M) = 68.06 ms + Q(M) on this host, table in §6.2.
- **Fully measured census** on the public fixture: numerator 8706.5 ms,
  denominator 16237.1 ms, **decode-time share 53.621 %** (round share 50.000 %,
  amplification 1.0724×). But mean M 7.27 ≠ beagle 5.53, so this is not a beagle
  number.
- **The ranked absolute share is not identified.** Exact vertex enumeration of
  `{ρ≥0, Σρ=1, ΣMρ=1+n}` on support 2..9 gives **[0 %, 100 %]** for both beagle
  and medicine. `ρ₂=0.116822, ρ₆=0.883178` and `ρ₅=0.866822, ρ₉=0.133178` both
  reproduce beagle's published mean exactly, with shares 0 and 1.
- **The amplification ratio *is* identified**, because mean-M pins C̄ to ±4.8 %:
  beagle **[0.8646×, 1.3436×]**, medicine **[0.8484×, 1.3130×]**. This is the
  reusable instrument that converts any round share into a decode-time share.
- Applied to alphonse's thresholds: beagle 5.70 % → **[4.93 %, 7.66 %]**,
  medicine 3.74 % → **[3.17 %, 4.91 %]**. Both **straddle**. I cannot place
  "E27 done right" above or below the line.

Fields used: `effective_mean_draft_len`, `non_drafting_round_count` (nd=0 pins
ρ₁=0 — the correction that moved my earlier `.90654` to **`.883175`**). Round
counts enter only an integrality remark. Integrality would only *tighten*, so
the brackets are conservative in the safe direction. One extra published field —
a per-round width histogram or any second width moment — would close it.

## 8. Requirement #4: marginal cost vs occupancy

The concern behind this requirement is the `asyncEval` ladder at
`Sources/MLXFastModel/Qwen35.swift:2187-2197`: if QMV work overlaps with other
scheduled work, then the *occupancy* share of QMV (how much of the timeline it
sits on) can exceed its *marginal* share (how much leg time a change to it
actually costs). Optimising against occupancy when marginal cost is lower is
exactly how a kernel win evaporates end to end.

The injection measures the marginal share directly; the isolated curve measures
occupancy. They can be compared arm by arm:

| arm | treated | marginal (ψ_eff) | occupancy (curve) | marginal / occupancy |
|---|---|---|---|---|
| p2L1 | 2..9 | 0.6729 | 0.6718 | 1.0016 |
| p2L2 | 2..9 | 0.6736 | 0.6718 | 1.0027 |
| p6L1 | 6..9 | 0.6145 | 0.6131 | 1.0023 |
| p6L2 | 6..9 | 0.6133 | 0.6131 | **1.0004** |
| m6L2 | 6 only | 0.1824 | 0.1814 | 1.0055 |

**Marginal cost equals occupancy to within 0.6 % at every width set tested**, and
the ratio is slightly *above* 1 rather than below, so there is no hidden
overlap discount. The practical consequence: for this family, on this fixture and
host, an occupancy-based cost model is a sound basis for deciding what to
optimise. That is not a general licence — it is a measured fact about the QMV
family at these widths, and the same check should be repeated for any family
whose dispatches are expected to overlap.

## 8.1 Both legs decomposed, with both shares measured

ψ(MTP) comes from the p2 arms and ψ(serial) from m1, so neither term is a curve
prediction. `research/e42_leg_decomposition.py`, artifact
`research/e42-artifacts/leg-decomposition.json`:

| per round | serial (M = 1) | MTP (mean M 7.27) | growth |
|---|---|---|---|
| QMV | **62.711 ms** | **140.484 ms** | **2.240×** |
| non-QMV | **10.849 ms** | **68.060 ms** | **6.273×** |
| total | 73.559 ms | 208.543 ms | 2.835× |
| non-QMV share | 14.7 % | **32.6 %** | — |

**Non-QMV work scales 2.800× more steeply with row count than QMV does**, and the
mechanism is the same weight-stream amortisation as §6.1: one stream serves a
whole group of rows, so QMV is sublinear in M, while per-row attention,
recurrence and normalisation work plus the speculation machinery are not.

Two consequences worth carrying forward:

- **One third of the candidate leg — 5.309 s of 16.266 s — is not QMV at all**,
  which is a larger absolute pool than the stream boundary, and its share grows
  with depth. It is a different axis from anything E42 was asked about; flagged,
  not implemented.
- **ψ falls with width, so 0.672 is the pessimistic end.** ψ(M=1) = 0.8525 and
  ψ(mean M 7.27) = 0.6736. Any prompt with a lower mean width than this fixture
  should read a *higher* ψ, so 0.672 is a lower bound for beagle (mean M 5.53) —
  subject to a monotonicity for which I have two points and not a curve.

Independent check: the isolated `--shapes-only` curve puts Q(1) = **64.433 ms**
against the injection-measured serial QMV of **62.711 ms**, **+2.75 %** — curve
against leg, no shared code path.

One check I wrote and then deleted, because it was not one: the p2 ladder
intercept **is** `mtp.non_qmv_ms_per_round` by construction, since
`psi_from_slope ≡ q_mean/t0`. Its "0.00 % agreement" was an identity. The script
now records `mtp_non_qmv_intercept_is_identical_by_construction: true` so nobody
re-reads it as corroboration.

## 9. Bit-exactness

Two independent layers:

1. **End-to-end, on the scored path**: every arm produced
   `all_tokens_matched=true` and `residual_divergence_count=0` over the full
   512-token window, with `effective_draft_lengths` element-wise equal to base.
   That is the contract the trusted parent actually checks.
2. **Per-width, at kernel level**: the 192-cell parity rig
   (8 shapes × widths 1..12 × bits {4,3}) via `research/run-qmv-parity.sh`.

`research/qmv_parity_compare.py` never exits non-zero, so the verdict line is
parsed explicitly. Any differing cell is a hard stop, not a tolerance.
`research/e42-parity.sh` wraps the runner with a group-private output directory
and exits 1 if any arm reports `DIVERGES`.

Both layers are complete. All six treated arms are bit-identical to the
reference arm on all 192 cells:

| arm | group | cells compared | differing | bits=4 | bits=3 | verdict |
|---|---|---|---|---|---|---|
| p2L1 | A | 192 | **0** | 96 / 0 | 96 / 0 | BIT-IDENTICAL |
| p2L2 | A | 192 | **0** | 96 / 0 | 96 / 0 | BIT-IDENTICAL |
| p6L1 | A | 192 | **0** | 96 / 0 | 96 / 0 | BIT-IDENTICAL |
| p6L2 | B | 192 | **0** | 96 / 0 | 96 / 0 | BIT-IDENTICAL |
| m6L2 | B | 192 | **0** | 96 / 0 | 96 / 0 | BIT-IDENTICAL |
| m1L1 | B | 192 | **0** | 96 / 0 | 96 / 0 | BIT-IDENTICAL |

1152 cell comparisons, zero differing. Group A ran as one supervised job in
844 s and group B in 930 s, each exiting 0.

**Cross-group reference determinism: IDENTICAL, 192 cells compared, 0
differing.** Group A and group B built the test binary independently from the
same base twins, so this is evidence that the rig is reproducible across builds
rather than only self-consistent inside one build. Without it, six
`BIT-IDENTICAL` verdicts would be consistent with a rig that silently compares
each arm against itself.

The result is what the construction predicts and is therefore weak evidence
about the *design* and strong evidence about the *implementation*: a rolled loop
that recomputes an identical accumulation cannot change the result, so the value
of this rig is that it would have caught an implementation slip — a
reassociated accumulator, a hoisted load, an off-by-one in the pass bound —
which is exactly the failure mode that would have invalidated every ψ number
here.

### 9.1 What the coverage number actually covers

The suite reports `covering_cells_by_bits`, which counts cells whose
`in_kernel_path` is anything other than `qmv_fast_impl`. On this base that is
`{4: 64}` and nothing at bits=3, because the crossrow gate hardcodes
`bits == 4`. That metric answers "did this cell reach a crossrow body". It is
the right question for p2/p6/m6 and the wrong question for m1.

m1's edit is an `ntg.x == 1` dispatch inserted immediately *before* the generic
fall-through, so every cell m1 treats is a `qmv_fast_impl` cell — precisely the
cells the shared metric excludes. Quoting only the suite number would report m1
as zero coverage; quoting only my number would claim crossrow coverage m1 does
not have. `research/e42_covering_cells.py` emits both from the reference arm's
own cell table:

| arm | mechanism | crossrow-covering | treated | by bits | controls |
|---|---|---|---|---|---|
| p2L1 | crossrow | 64 | 64 | {4: 64} | 128 |
| p2L2 | crossrow | 64 | 64 | {4: 64} | 128 |
| p6L1 | crossrow | 32 | 32 | {4: 32} | 160 |
| p6L2 | crossrow | 32 | 32 | {4: 32} | 160 |
| m6L2 | crossrow | 8 | 8 | {4: 8} | 184 |
| m1L1 | generic | **0** | **16** | {3: 8, 4: 8} | 176 |

Treated union 80 / 192. The 112 untreated cells are 24 bits=4 generic cells
(widths 10..12, which have no switch case and so run with the default
`E42_PASSES = 0`) plus all 88 bits=3 cells at widths 2..12. bits=3 is a pure
control for p2/p6/m6 and is *treated* by m1 at width 1, because m1's insert
sits downstream of the bit-width gate. Both facts are load-bearing: the bits=3
column is the evidence that the crossrow arms cannot perturb an excluded bit
width, and m1's bits=3 cells are the evidence that its insert really is at the
generic fall-through rather than inside the gate.

The script additionally compares the reference arm's digests between the two
parity groups. The groups are separate test-binary builds of the same base
twins, so a match is independent evidence that the rig is deterministic and not
merely self-consistent within one build.

## 10. MDE for every null

Computed with **unmodified** `research/e39_mde.py`.

- within-arm MTP sd: 0.0404 % (base), 0.0763 % (p2L1)
- exact two-sample n=2 MDE: **0.4315 %**
- the p2 effects are **140×** and **284×** that

`--audit` reference values: `local_e2e_leg two_sample n=2 sd=0.2974 → 0.8332 %
(exact 1.6814 %)`; `ranked_score single n=1 sd=0.0923 → 0.2586 %`.

## 11. Reproduction

```bash
python3 research/e42_perturb.py --self-test          # 191 checks
python3 research/e42_perturb.py --arm p2 --level 2   # or p6 / m6 / m1
python3 research/twin_audit.py quantized
bash research/e42-run.sh p2L2 --curve --legs 2
python3 research/e42_perturb.py --revert

# bit-exactness, one supervised job per group (each takes the run lock)
bash research/e42-parity-prebuild.sh
bash research/e42-parity.sh A p2L1=bf64ead p2L2=afc8916 p6L1=6b8ae93
bash research/e42-parity.sh B p6L2=78cd88d m6L2=04f28ac m1L1=d984b45
python3 research/e42_covering_cells.py

python3 research/e42_width_census.py
python3 research/e42_leg_decomposition.py
python3 research/e42_analyze.py \
  --arms base p2L1 p2L2 p6L1 p6L2 m6L2 m1L1 --parity-groups A B --wandb
```

Every arm's reference is `04ad6bf11437c269df85a47e91faa769c74fe6da`; the arm
commits are `p2L1=bf64ead p2L2=afc8916 p6L1=6b8ae93 p6L2=78cd88d m6L2=04f28ac
m1L1=d984b45`. The parity runner deliberately leaves one arm's twins in the
worktree while it runs and restores base twins on exit, so do not run it
concurrently with anything that inspects the tree.

`--local-iterate` never rebuilds Swift, so `research/e42-run.sh` rebuilds both
SwiftPM roots explicitly and asserts product freshness before timing. The curve
uses `--shapes-only` with a raised `MLX_MAX_OPS_PER_BUFFER` fence so per-width
cost is isolated from ranked geometry.

## 12. What this does and does not license

**Does**: ψ ≈ 0.672 for the QMV family on this fixture and host, interval
[0.659, 0.674] with a conservative floor of **0.604** that must be quoted
alongside it wherever ψ is load-bearing; a 0.12 %-linear instrument; three
agreeing estimators for the M=6 boundary and a smooth-in-M cost model refuted
arithmetically; per-width ψ·φ(M) available as a cost table; and — because
non-QMV cost grows 2.800× steeper in width than QMV cost — the statement that
**0.672 is a lower bound for a narrower-mean prompt such as beagle** (mean M
5.53 against 7.27 here). That is a two-point monotonicity argument, not an
extrapolation, and it holds only in direction.

**Does not**: any score claim. This is one public fixture at a width ceiling on
an M4 Pro, ungated, with the ranked runner being an M5, and it is a **two-stream
ψ at ~91 % cost weight** on the E27 table `04ad6bf1`. ψ is a *ceiling on the
prize* from QMV work, not a prediction that any particular kernel change earns
it, and no per-width absolute time here transfers to the shipped table.

## 13. Suggested follow-ups (not implemented)

1. **Ask for one telemetry field.** A per-round width histogram in
   `officialMetrics.per_prompt` collapses the partial identification in §7 to
   arithmetic. This is the highest value-per-effort item in this whole report.
2. **Price the M=6 step directly.** The step is 32.8 ms/round of pure weight
   re-streaming at the width that carries the most ranked mass. An IPG table
   that keeps M=6 at one weight pass is the obvious candidate, and §6.1 gives
   the exact predicate to change. Note this is *not* E27 (which bought M=5 and
   M=9); it is the untouched boundary.
3. **Re-run the ψ instrument on an M5.** ψ is a host property. The instrument is
   cheap, bit-exact and reverts cleanly, so it can be pointed at the ranked host
   without risking a candidate.
4. **Use the injection as a general sensitivity probe.** Any family with an
   editable kernel can be measured this way; the MDE trap is not specific to
   QMV.
5. **Run the same `--shapes-only` ladder on the tip.** This is the cheapest
   remaining experiment and the only one that would turn §6.1 from a one-sided
   observation into a 2×2. The tip's boundaries are 4→5 and 8→9, so the
   mechanism predicts large increments *there* and an ordinary increment at 5→6,
   which is the exact reverse of my row. No smooth function of M can produce
   that reversal, and no timed leg is needed — the ladder alone settles it.
   `senpai/verify-kernel-table.sh` should be run first so the row is labelled
   with the table it was measured on.
