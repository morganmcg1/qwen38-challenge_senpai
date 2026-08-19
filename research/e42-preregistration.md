# E42 pre-registration — ψ and φ by injected bit-exact regression

Committed **before** any timed run. Everything below is a prediction or a rule,
not a result. `research/e42-results.md` reports the outcome against this text.

- assignment `qwen38-r1-e42-psi-phi-by-injected-regression`, revision `r1`
- base `04ad6bf11437c269df85a47e91faa769c74fe6da`, branch
  `qwen-askeladd/psi-phi-by-injected-regression`
- host: local M4 Pro (**not** the ranked M5). Nothing here ships.

## Question

`score gain = sensitivity × ψ × φ × x`.

`x` (kernel-level cost reduction) has been measured repeatedly. `φ` (treated
widths' share of QMV cost) is ~0.201 local / ≥0.217 ranked at M=6. **ψ — the QMV
share of the candidate leg — has never been measured.** thorfinn back-solved
`ψ·φ = 0.0459` from an end-to-end delta, implying `ψ ≈ 0.228`; our own forward
time attribution puts the quantised matvec at **0.59** of forward time. The
factor-2.6 gap decides whether K-tiled activation staging reaches the crown
(`ψ ≈ 0.59`) or the whole axis is dead (`ψ ≈ 0.1`). This has blocked the
campaign since ledger 137.

## Why the obvious design cannot answer it

A candidate optimisation worth `x` at the kernel level moves the leg by
`ψ·φ·x`, which for any plausible `x` sits under the paired-E2E floor
(0.417 % normal / 0.632 % exact at n=4, `research/e39_mde.py`). Every attempt so
far has been a null that cannot distinguish `ψ = 0.59` from `ψ = 0.1`.

## Design: divide a large regression by its own kernel cost

Inject a **deliberate, large, bit-exact slowdown** into the QMV kernel at
selected widths and divide:

```
ψ_effective = (measured leg slowdown) / (measured kernel-level slowdown)
```

The regression is sized to land 10–100× above the MDE, so the *ratio* is precise
even though each factor is crude.

Mechanism (`research/e42_perturb.py`): wrap accumulator re-init **plus the
unchanged K loop** in `for (e42_pass = 0; e42_pass <= E42_PASSES; e42_pass++)`.
Each pass re-zeroes the accumulators and repeats the same loads in the same
order, so **the surviving final pass is bit-identical to base by construction** —
no reassociation, unchanged lane membership, unchanged `simd_sum`, one store
after the last pass. `E42_PASSES` is a template parameter, so untreated widths
keep base instantiations verbatim within the same build.

### Arms

| arm | gate | treated widths | what the ratio yields |
|---|---|---|---|
| `p2` | `ntg.x ∈ 2..9`, both `out_vec_size` tiers | 2–9 | **ψ** for the verify pass |
| `p6` | `ntg.x ∈ 6..9` | 6–9 | **ψ·φ** directly |
| `m1` | `qmv_fast_impl` under `ntg.x == 1` | 1 | sign-flip control + serial-leg ψ |

`p2`/`p6` leave the M=1 draft-head readout and the whole depth-0 serial leg
untouched, so **the serial leg of the same `--local-iterate` invocation is a
within-run control** and `raw_p` must *decrease*. The `m1` arm slows the serial
leg's entire matvec workload, so `raw_p` must *increase*. That sign flip is the
measurement-chain test the assignment requires; `m1` additionally measures the
QMV share of the **pinned serial leg**, which is the most direct possible test of
the "matvec is 0.59 of forward time" attribution.

### Magnitude ladder (the scientific core, not optional)

Levels `L1 = 1` and `L2 = 2` extra passes in both `p2` and `p6`. If a single ψ
exists, `ψ_eff(L1) = ψ_eff(L2)`. Reported as a first-class result: **non-linearity
means no single ψ exists and linear pricing of this axis is invalid.**

## Static gate already passed (`research/e42-artifacts/air-dce-gate.txt`)

Compiled from `research/e42_air_probe.metal` at `-O2`, `_wide` at NA=3 (the
shipped M=6 cell) and NA=5 (the M=5/M=9 register high-water), plus the pair
kernel at M=2:

- **The pass loop stays rolled** (`loop_backedges` +1 at L1; per-body `float_ops`
  and `device_loads` unchanged, as `air_kernel_stats.py` documents for rolled
  loops). A rolled loop cannot have a dead iteration peeled, so the redundant
  work provably executes `E42_PASSES+1` times.
- **`peak_live_regs` is flat and equal to base at every level** (50 / 70 / 42).
  No spill and no occupancy change, so untreated widths remain a valid
  within-build control.
- Level 0 is identical to base on every semantic counter (allocas, fmul, fadd,
  float_ops, device_loads, loads, backedges, peak_live_regs, types). The `_wide`
  AIR text is 3 lines shorter (291 → 288); a line-level diff shows only
  instruction scheduling and metadata — hoisted `bitcast`/`shl`/`lshr`
  placement, renumbered `!tbaa`/`!alias.scope` IDs, and one extra
  `icmp eq i32 %N, 0` + `br` from the collapsed level-0 pass loop. No arithmetic
  or load added or removed.

## Predictions

**Kernel level `x(L)`.** The treated kernels run at 211 GB/s = 77 % of this
host's 273 GB/s peak, so they are weight-bandwidth bound and repeating the K
loop should cost near-linearly: `x(L1) ∈ [0.5, 1.0]`, `x(L2) ∈ [1.0, 2.0]`. The
lower end is what L2 cache reuse across passes would buy. **This is why `x` is
measured per width rather than assumed** — it is the denominator.

**Leg level, `p2` L1**, under the three live hypotheses (using `x = 1.0`):

| hypothesis | ψ | predicted MTP-leg slowdown |
|---|---|---|
| forward time attribution | 0.59 | +59 % |
| thorfinn back-solve | 0.228 | +23 % |
| axis is dead | 0.10 | +10 % |

`p6` L1 at `ψ·φ = 0.0459` predicts **+4.6 %**; at `ψ = 0.59, φ = 0.2` it predicts
+11.8 %. On the local longcopy fixture `φ_local` is plausibly 0.6–0.9, so the
observed `p6` move may be much larger than the ranked-φ figure — which is why
**`p6`/`φ_local` is reported as a local quantity and `p2`/ψ is the headline.**

Every prediction is far above the two-sample n=2 exact MDE of **1.68 %**
(`sd = 0.2974 %`, `research/e39_mde.py --audit`), so n=2 legs per arm suffice:
leg noise contributes only ±0.003 to ψ. The binding uncertainty is the
denominator and the width weighting, not the leg.

## Measurement plan

- **Denominator, per width.** `research/run-qmv-curve.sh TAG --widths 1..9
  --shapes-only --reps 21 --inner 10 --skip-stock`, one curve per arm.
  Ranked-geometry replay is deliberately skipped: thorfinn's geometry-invariance
  result makes it redundant and it costs ~9.5 min/curve.
  **Fence:** a `--shapes-only` probe issues one op per call, so it is
  insensitive to `MLX_MAX_OPS_PER_BUFFER` *by construction* and must never be
  cited for end-to-end command-buffer behaviour.
- **Numerator.** Two `--local-iterate` invocations per arm (each = one serial
  depth-0 leg + one MTP depth-8 leg), 512 decode tokens, fixture pinned to
  `correctness_prompts/public_longcopy_gate_english_512_256.json`, declared
  proposal head pinned by digest. `MLXFAST_LOCAL_COOL_GATE=0` with
  `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
  preserved verbatim; entry/exit GPU temperature recorded per arm. Base is
  bracketed at session start and end to bound drift.
- **Fixture / trajectory invariance.** `effective_draft_lengths` is recorded per
  round in every arm and asserted **element-wise equal** across arms.
  `costModelDepth` (`Qwen36MTPBlockSession.swift:700-744`) is a pure function of
  `positionAcceptEMA`, top-2 margins and `h = 0.18` with no wall-clock input, so
  equality is expected — but it is verified, not trusted. Movement is a stop.
  The longcopy window is a *ceiling* on width, not a matched proxy for ranked
  prompts; its accept rate is reported next to every histogram.
- **Bit-exactness.** thorfinn's 192-cell cross-build parity rig
  (`research/run-qmv-parity.sh`), arms passed as **commit SHAs** so each cell is
  built from committed source with recorded twin digests. Job A =
  `ref + p2L1 + p2L2` (headline arm at both levels, i.e. level-independence);
  job B = `ref + p6L2 + m1L1` (the other two mechanisms at their top level).
  Reported honestly: of 192 cells per arm, the **treated** cells are 64 for
  `p2` (bits=4 × widths 2–9), 32 for `p6`, and 8 for `m1`. The rig's own
  `covering_cells_by_bits` counts crossrow cells, so it reads 64 for `p2`/`p6`
  and **0 for `m1`** — for `m1` the treated cells are exactly the cells the rig
  labels non-covering, and that will be stated rather than papered over. bits=3
  cells reach no treated path and act as a control that the rest of the file did
  not move.

## Analysis quantities

1. `x(M)` per width from the curve; `x̄` = QMV-time-weighted mean over treated
   widths. If `x(M)` is flat, `ψ_eff` is essentially weighting-free.
2. `ψ_eff ≡ (ΔT/T) / x̄` on the MTP leg. **Primary answer.**
3. `φ_local ≡ ψ_eff(p6) / ψ_eff(p2)`.
4. **Absorption `α ≡ ΔT_measured / ΔQ_predicted`**, with `ΔQ` predicted from
   curve `seconds_per_call` × per-shape `calls_per_verify` × the round-width
   histogram. Since `ψ_eff = α × (Q_treated/T)`, this turns assignment
   requirement #4 (marginal vs occupancy share) from a caveat into a
   measurement: `α ≈ 1` means QMV time is fully on the critical path;
   `α < 1` means part of it hides behind the `asyncEval` ladder at `S ≤ 9`
   (`Qwen35.swift:2187-2197`), and the **marginal** share is the correct
   coefficient for pricing an optimisation. `α` also absorbs any
   isolated-vs-in-situ dispatch difference, so `ψ_eff` (which needs only the
   *ratio* `x`) is the robust primary and `Q/T` is the auxiliary.
5. Linearity: `ψ_eff(L1)` vs `ψ_eff(L2)` per arm. Sub-linear ⇒ inspect (4)
   first; super-linear ⇒ thermal/DVFS, since AIR excludes spill.
6. MDE for every null via `research/e39_mde.py`, **unmodified**. An
   under-powered leg is reported as under-powered; no aggregate is substituted.

## Stop rules

Report when `p2` and `p6` are each measured at ≥2 magnitudes, bit-exactness is
proven, per-width denominators are measured, and linearity is reported.

Stop early on:

- **parity failure on any cell** — hard stop, the mechanism is not bit-exact;
- **`effective_draft_lengths` moving between arms** — the instrument moved;
- **`m1` failing to raise `raw_p`** — the measurement chain is broken;
- serial-leg time moving in a `p2`/`p6` arm beyond the drift envelope — the
  width gate leaked.

## Scope

`Vendor/mlx-swift/.../kernels/quantized.h` and
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` only, twin-locked at
`+13` (`research/twin_audit.py` must print `TWIN AUDIT OK`). Host dispatch
(`backend/metal/quantized.cpp`), `mtp-head.manifest.json`, every depth-policy
constant, `AttentionUtils.swift`, `sdpaWidthWallDepthCap` and
`segmentedStreakGate` are untouched.

**Nothing ships.** The final commit restores both twins to base, and
`git diff 04ad6bf11437c269df85a47e91faa769c74fe6da HEAD -- Sources Vendor benchmark.json`
must be empty.
