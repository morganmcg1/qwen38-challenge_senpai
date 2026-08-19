# E42 results — ψ and φ by injected bit-exact regression

Assignment: PR #47, `qwen38-r1-e42-psi-phi-by-injected-regression`, revision `r1`.
Pre-registration: `research/e42-prereg.md` (commit `dc379d5`, before any arm ran).

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
| p6 arm (M≥6 → ψ·φ) at ≥2 magnitudes | L1 done, L2 running |
| linearity across magnitudes | **done** for p2 (the scientific core) |
| m1 sign-flip control | pending |
| m6 single-width arm (priority B) | **done** |
| bit-exactness, end to end on the scored path | **done**, every arm |
| bit-exactness, 192-cell parity rig | pending |
| denominator measured per width, `--shapes-only` | **done**, no ranked-geometry replay |
| ψ and φ reported separately, with intervals | **done** |
| requirement #4: marginal vs occupancy | **done**, `asyncEval` ladder |
| MDE for every null, unmodified `research/e39_mde.py` | **done** |
| fixture pinned, `effectiveDraftLengths` element-wise equal | **done** |
| revert proof: empty diff against base | pending, final commit |

Status is as of this revision of the document; the sections below state exactly
which arms are measured.

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
| m6L2 | `04f28ac` | 6 only | 21.684 | 37.603 | 1.7341 | +1.8261 | +33.308 % | **0.1824** | 1.0053 |

Serial-leg drift across arms: −0.111 %, −0.210 %, −0.210 %, −0.159 % — all well
inside the 0.4315 % MDE, as required, since the serial leg is entirely width 1
and therefore untreated in every arm above.

`α` is the additivity check: predicted-over-observed leg time under the
assumption that the injection adds cost without changing anything else. α ≈ 1 to
0.3 % on every arm, so the injection behaves as pure added work.

### Linearity — the scientific core

ψ must not depend on the injection magnitude. Measured at two magnitudes a
factor 2.013 apart in kernel cost:

**ψ_eff(L2) / ψ_eff(L1) = 1.0012** — 0.12 %, versus a 0.4315 % MDE.

If the instrument were mis-specified — DCE, occupancy change, cache-pressure
nonlinearity, or a leg-time term I had not accounted for — ψ would drift with
magnitude. It does not, to a tenth of a percent.

### An independent ψ that predicts no absolute cost

The ratio estimator still leans on the curve's absolute calibration. A second
estimator avoids that entirely: regress leg time on the *ladder slope* only.

- treated QMV work `Q_treated` = **10.9577 s**, pairwise spread **0.230 %**
- ⇒ **ψ = 0.6736**
- non-QMV intercept **5.3087 s = 68.06 ms/round**
- the isolated `--shapes-only` curve independently predicts 10.9284 s, agreeing
  to **+0.268 %**

Two estimators with different failure modes agree to a quarter of a percent.

### ψ interval over three denominators

The denominator is measured per width on `--shapes-only` curves; no ranked
geometry is replayed. Three defensible denominators give:

| arm | as measured | drift corrected | stable shapes only | interval |
|---|---|---|---|---|
| p2L1 | 0.6729 | 0.6040 | 0.6691 | [0.6040, 0.6729] |
| p2L2 | 0.6736 | 0.6592 | 0.6739 | [0.6592, 0.6739] |
| p6L1 | 0.6145 | 0.5541 | 0.6125 | [0.5541, 0.6145] |
| m6L2 | 0.1824 | 0.1722 | 0.1942 | [0.1722, 0.1942] |

**Preferred ψ ≈ 0.672**, interval **[0.659, 0.674]** from the tightest arm.

The drift term is worth naming honestly. The untreated M=1 calibration cell
reads −5.13 % (p2L1), −1.39 % (p2L2), −4.95 % (p6L1) — apparently impossible,
since M=1 is untreated. It is **one shape**:
`linear_attn.in_proj_fused_qkvzba` at −24.61 %, whose *within-curve* spread is
32.6 % (base) to 146.2 % (arm) — i.e. that cell is not a stable measurement at
all. Restricted to reproducing shapes the same calibration reads **+0.38 %,
+0.40 %, +0.73 %**, consistent with zero. I report the interval rather than
silently dropping the unstable shape.

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
| φ_local(M≥6) | **0.9132** (p6L1) | 0.2351–0.9380 | inside, near the top |
| **φ_local(M=6)** | **0.2708**, interval [0.2557, 0.2883] | 0.0000–0.9701 | **bounded away from zero** |
| φ_local(M∈{7,8,9}) | **0.6424** by subtraction | — | — |

φ_local(M=6) = ψ_eff(m6L2) / ψ_eff(p2L2) = 0.1824 / 0.6736. The ranked corpus
cannot bound φ(M=6) away from zero at all; a single injected width pins it to
±6 % of its own value. Cross-level use of L1 and L2 arms is licensed by the
0.12 % linearity result in §5.

φ_local sits near the top of the pooled ranked bracket because this fixture is a
width **ceiling**: its 0.8875 accept rate saturates drafting at depth 8, giving
mean M 7.27 against beagle's 5.53. This is *local* φ. Ranked ρ(M) is edward's.

Per-width shares from the isolated `--shapes-only` curve:

| width | ψ(M) from curve | ψ_eff measured end-to-end | agreement |
|---|---|---|---|
| M=5 | 0.0294 | — | — |
| **M=6** | **0.1814** | **0.1824** (m6L2) | **0.55 %** |
| M=9 | 0.3427 | — | — |

The curve and the end-to-end injection are independent instruments — one times
isolated kernel dispatches with a raised op-per-buffer fence, the other times a
512-token decode leg — and they agree on the width-6 share to **0.55 %**. That
is the strongest internal validation in this experiment.

### 6.1 The mechanism of the M=6 step

In the `_m` kernel:

```
const int first_m = int(tid.x) * IPG;
if (first_m >= M) return;
```

Working groups = `ceil(M / IPG)`, and **each group streams the whole weight
matrix**. With the base IPG table `{3:3, 4:4, 5:5, 6:3, 7:4, 8:4, 9:5}`:

**weight passes = 1 for M ≤ 5, and 2 for M ≥ 6.**

That is a structural discontinuity at exactly M=6, not a smooth cost curve. It
is independently corroborated by the committed
`research/e34-ranked-operating-point.json` (`/dispatch/weight_passes/*`), which
also records `weight_passes_pre_e27` of M=5→2 and M=9→3 and
`single_pass_top_width_now = 5`: E27's IPG 3→5 bought exactly one weight pass at
each of M ∈ {5, 9}.

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
| m6L2 | 6 only | 0.1824 | 0.1814 | 1.0055 |

**Marginal cost equals occupancy to within 0.6 % at every width set tested**, and
the ratio is slightly *above* 1 rather than below, so there is no hidden
overlap discount. The practical consequence: for this family, on this fixture and
host, an occupancy-based cost model is a sound basis for deciding what to
optimise. That is not a general licence — it is a measured fact about the QMV
family at these widths, and the same check should be repeated for any family
whose dispatches are expected to overlap.

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
python3 research/e42_analyze.py --arms base p2L1 p2L2 p6L1 p6L2 m6L2 m1L1 --wandb
python3 research/e42_width_census.py
python3 research/e42_perturb.py --revert
```

`--local-iterate` never rebuilds Swift, so `research/e42-run.sh` rebuilds both
SwiftPM roots explicitly and asserts product freshness before timing. The curve
uses `--shapes-only` with a raised `MLX_MAX_OPS_PER_BUFFER` fence so per-width
cost is isolated from ranked geometry.

## 12. What this does and does not license

**Does**: ψ ≈ 0.672 for the QMV family on this fixture and host, with a
0.12 %-linear instrument and two agreeing estimators; the M=6 weight-pass step
is real and a smooth-in-M cost model is refuted; per-width ψ(M) is available as
a cost table.

**Does not**: any score claim. This is one public fixture at a width ceiling on
an M4 Pro, ungated, with the ranked runner being an M5. ψ is a *ceiling on the
prize* from QMV work, not a prediction that any particular kernel change earns
it.

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
