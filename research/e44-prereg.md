# E44 Gate 0 pre-registration — simdgroup-matrix QMV, register gate before any timing

Written and committed **before** the candidate cell was compiled. Every number
below is from `research/e44_sgmm_air.sh` on the tree named in §1.

## 1. Tree every register number is measured on

| item | value |
| --- | --- |
| assignment base | `efff400c1b5554be2e8993b01856653d55de7664` (PR #48 merge) |
| worktree | `d3e498a` (assignment commit), branch `qwen-alphonse/simdgroup-qmv-register-gate` |
| `quantized.h` sha256 | `75d45143959eb3bd…` (identical at base and worktree) |
| toolchain | Apple metal 32023.883 (metalfe-32023.883), `-std=metal3.1 -O2` then `metal-opt -passes=default<O3>` |
| host | local `applegpu_g16s`; register readout is compile-only so the GPU is not involved |

## 2. Baseline, re-derived rather than cited

Post-E27-revert `>=4096` dispatch table `<T,2> <T,3,3> <T,4,4> <T,5,3> <T,6,3>
<T,7,4> <T,8,4> <T,9,3>`:

| M | cell | regs | allocas |
| --- | --- | --- | --- |
| 2 | `qmv_fast_crossrow_affine4_g64<T,2>` | 89 | 5 |
| 3 | `<T,3,3>` | 83 | 1 |
| 4 | `<T,4,4>` | 104 | 1 |
| 5 | `<T,5,3>` | 87 | 2 |
| 6 | `<T,6,3>` | 83 | 1 |
| 7 | `<T,7,4>` | **108** | 2 |
| 8 | `<T,8,4>` | 104 | 1 |
| 9 | `<T,9,3>` | 83 | 1 |

* **Kernel-wide max = 108**, binding cell `<T,7,4>`. Matches the advisor's 108.
* Production entry `affine_qmv_fast<bfloat16_t,64,4,false>` = **163**, 55
  allocas. Matches the advisor's 163. `batch1` = 31, 2-bit control = 57.
* `_wide` NA=2/3/4 anchor = **62 / 83 / 104**, reproducing E13/E27/E32/E40 and
  thorfinn's independent `crossrow` r=4 numbers on this toolchain and tree.
* Static threadgroup memory in the module: **0 bytes**.

🔴 **A floor the campaign has not stated.** The `<4096` cells
`qmv_fast_crossrow_affine4_g64<T,M>`, which no arm touches, read **89** — above
`<T,3,3>`=83, `<T,5,3>`=87, `<T,6,3>`=83 and `<T,9,3>`=83. They are inlined into
the same `[[kernel]]`, so **the kernel-wide max cannot fall below 89 no matter
what replaces the wide cells.** The whole available ceiling movement from this
mechanism is therefore `108 -> 89`, i.e. **-17.6 %**, and nothing larger can be
claimed for it.

## 3. Instrument, and the artifact that must be corrected before the gate is read

There is no true register readout on this box (E40 §E). `peak_live_regs` is a
lane-weighted peak-live-SSA textual heuristic: shape usable, absolute number
not.

🔴 **The uncorrected heuristic is guaranteed to report a false ceiling raise for
this specific mechanism, and I verified the mechanism of the artifact rather
than assuming it.** AIR models an 8×8 fp32 simdgroup matrix as **one
simdgroup-wide `<64 x float>` value**:

```llvm
%6  = call fast <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float 0.0)
%23 = call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(...)
%13 = insertelement <64 x float> %8, float %12, i64 1     ; thread_elements()[1]
```

Its 64 elements are distributed across the 32 lanes of the simdgroup, so the
per-lane footprint is `64/32 = 2` registers, and `thread_elements()` indexes
exactly 2 slots — which is why `mlx::steel::BaseMMAFrag<T,8,8>` carries state as
`vec<T,2>`. Lane-weighting a `<64 x float>` like an ordinary `<4 x float>`
over-reports it by **exactly 32×**.

`research/air_kernel_stats.py` therefore gains an **opt-in**
`--simdgroup-distributed` correction (default off, so every previously published
number from the tool is byte-identical), self-tested in
`research/e44_air_summary.py --selftest`: naive 68 / corrected 6 on a synthetic
body holding one `<64 x float>` plus one `<4 x float>`.

**Measurement precision:** the readout is deterministic. Repeat runs on a fixed
(tree, toolchain) are bit-identical, so there is no sampling MDE on this gate —
the only risk is instrument *validity*, which is what §3 addresses and §4
gate (b) cross-checks.

## 4. Pre-registered decision rule

**Primary (a): kernel-wide max of the candidate dispatch table, lane-corrected,
against the base 108.**

* `<= 108` → register gate **PASSES**; proceed to §7.3.
* `> 108` → **FAILS**: the mechanism has inherited E27's tax and is *not
  bankable* without an end-to-end leg proving otherwise. Stop and report.

I accept the advisor's 108 bound and his reasoning: at 108 the mechanism is
register-neutral and its case rests entirely on weight traffic; above 108 every
untouched width pays. I differ only in *which readout* the bound is applied to,
for the reason in §3 — applying it to the uncorrected number would decide the
experiment on a known 32× artifact.

**(b) Spill allocas — independent and first-class.** E40 §E's own conclusion is
that spill-alloca detection, not the register heuristic, is the genuine compiler
outcome here. Base cells carry 1–2 allocas, all of type `[4 x [4 x i16]]` (the
`packed[4][4]` weight staging). **Any alloca type in the candidate cell that the
base cells do not have — e.g. an accumulator array that failed to promote to
registers — FAILS the gate on its own, even if (a) passes.**

**(c) Production entry against 163**, reported in the same direction. Treated as
corroboration, not as the primary: it inlines `qmv_fast_impl`,
`adjust_matrix_offsets` and the 2-bit cell too, so it moves for reasons the cell
table can attribute and it cannot.

**(d) Static threadgroup bytes, base = 0 bytes.** `affine_qmv_fast` is one
kernel, so a `threadgroup` array declared for one width cell is allocated on
**every** dispatch of **every** width — the same shared-allocation channel as
registers, through a resource no previous experiment in this campaign has had to
measure. Non-zero does not auto-fail, but it must be reported with its exact
byte count and priced; the design targets **0** by giving each of the two
simdgroups its own output tile instead of splitting K between them.

**(e) Readout disagreement.** If naive and lane-corrected verdicts differ, I
report the exact count of live `<64 x float>` values and show that it accounts
for the whole gap. If any **non**-distributed value type contributes to the gap,
the gate is reported **inconclusive**, not passed.

## 5. Predictions, so the gate is falsifiable

1. Candidate cell, lane-corrected: **60–95**, hence `<= 108`. Persistent
   per-lane state is 1–2 accumulator frags (2 regs each), one A-frag, one
   B-frag, one `uint4` of packed weights, ~8 pointers/scalars.
2. Candidate cell, **naive**: **> 108**, by roughly +62 per live tile, purely as
   the §3 artifact. This is the number the advisor's premise would have been read
   on, and it would have failed the mechanism for a reason that is not physical.
3. Binding cell of the candidate tree becomes an **untouched `<4096` narrow cell
   at 89** (§2 floor), so the ceiling moves `108 -> 89`, **-17.6 %**.
4. If prediction 3 holds, the H1 mechanism predicts the untouched widths get
   *slightly faster*. E40 priced a **+19.4 %** ceiling raise at **+0.186 %** mean
   tax per untouched width, so a **-17.6 %** drop scales to roughly **-0.17 %**.
   🔴 That is **below** the §7.3 paired MDE of **0.5040 %** (exact, df=4, n=5):
   the `M ∈ {1,2,3}` arm is powered to catch a **regression > 0.5 %**, and is
   **not** powered to confirm a +0.17 % gain. It is a guard, not a confirmation.

## 6. What is NOT being claimed at Gate 0

* No speed claim. Gate 0 spends zero GPU seconds and runs no timed leg.
* No exactness claim. The simdgroup MMA performs an 8-wide hardware reduction
  per step, so the K reduction order inside each quant group cannot be
  bit-identical to the incumbent per-lane sequential fp32 chain. Element
  products are identical (`(x/16^j) * (16^j q)` is exact), and the two-term
  affine algebra `scale·Σxq + bias·Σx` is preserved, but reduction order is not.
  Exactness against the public golden is deliberately **after** the ≥5 % bar, as
  the advisor kept from §7.3 ordering.
* No import of E43's `16–47 %` excess or `+1.15 %`: those are
  step-family-conditional and say nothing about a per-pass change across
  `M >= 4`. This mechanism is attractive *because* it is a per-pass weight-traffic
  change at every width `M >= 4`, not an `M = 6` boundary change.

---

## Gate 0 RESULT (reproduce with `research/e44_sgmm_air.sh`)

Toolchain `Apple metal version 32023.883`; probe compiled `-std=metal3.1 -O2`
then `metal-opt -passes=default<O3>`. Base `quantized.h` @ `efff400c`
sha256 `75d45143959eb3bd...`; candidate sha256 `e1594483af795dfc...`.

### VERDICT: **PASS** (all five pre-registered conditions)

| # | Pre-registered condition | Bound | Measured | Result |
|---|--------------------------|-------|----------|--------|
| a | lane-corrected kernel-wide max register allocation | `<= 108` | **89** (`e44_narrow_m2`) | **PASS** (-17.6 %) |
| b | new alloca *type* in the production entry (fails alone) | none new | entry type set unchanged; new cell has **0 allocas** | **PASS** |
| c | production entry lane-corrected peak | `<= 163` | **143** (allocas 55 -> 47) | **PASS** (-12.3 %) |
| d | static threadgroup bytes | no increase | 0 -> 0 | **PASS** |
| e | naive/lane split fully explained | must be | yes, see below | **PASS** |

### Kernel-wide maxima, by tree

The number that binds the kernel-wide max is *not* the new cell. Every register
figure below names the tree it came from.

```
tree                        naive  lane  allocas  loads
e44_sgmm_runtime (NEW)        344   34      0      28
e44_sgmm_m8      (NEW)        343   33      0      28
e44_sgmm_m9      (NEW)        343   33      0      28
e44_narrow_m2..m9 (kept)       89   89      5      80   <- binds kernel-wide max
e44_m3_ipg3  (base, replaced)  83   83
e44_m4_ipg4  (base, replaced) 104  104
e44_m5_ipg3  (base, replaced)  87   87
e44_m6_ipg3  (base, replaced)  83   83
e44_m7_ipg4  (base, replaced) 108  108   <- bound the BASE kernel-wide max
e44_m8_ipg4  (base, replaced) 104  104
e44_m9_ipg3  (base, replaced)  83   83
NA anchor NA=2/3/4             62/83/104 (reproduced exactly)
```

Per-M, lane-corrected, base -> candidate:
`2 89->89 · 3 83->83 · 4 104->34 · 5 87->34 · 6 83->34 · 7 108->34 · 8 104->34 · 9 83->34`.

Production entry AIR shrank 8047 -> 5869 lines (post-O3 14488 -> 9885).
The `bits==2` entry (57) and the `batch_1` entry (31) are byte-for-byte
unchanged, confirming the change is confined to the affine-4/g64 wide path.

### Naive vs lane-corrected split (condition e)

`simdgroup_matrix<float,8,8>` is a *distributed* type: one SSA value occupies
2 registers per lane, not 64. The naive per-value cost of 64 is a counting
artifact, so both peaks are published and fully decomposed:

* new cell: naive peak 344 = 5 distributed values (5x64 = 320) + 24 ordinary;
  lane peak 34 = 5x2 + 24. **Identical live set**, only the per-value cost differs.
* production entry: naive peak 395 = 5 distributed (320) + 75 ordinary;
  lane peak 143 with **0 distributed values live** at that program point
  (the entry's peak is in scalar prologue/epilogue code, not inside the cell).

### Feasibility through *both* compile paths

* **AOT**: `tools/build-mlx-metallib.sh` wrote `.build-worker/release/mlx.metallib`
  cleanly. Only warning is the pre-existing `quantized.h:1091 unused parameter
  'out_vec_size'` in the 2-bit singlerow cell.
* **JIT (runtime-effective)**: `Vendor/mlx-swift/Package.swift:25,284` compiles
  `jit_kernels.cpp` and *excludes* `nojit_kernels.cpp`, so the string in
  `mlx-generated/quantized.cpp` is what the scored worker actually compiles.
  `research/jit_string_compile.py` reassembles that exact 211,288-byte
  concatenation (utils + gemm + quantized_utils + quantized preambles plus
  `[[host_name]]` instantiations) and compiles it with
  `xcrun metal -std=metal4.0 -fno-fast-math -c` and **no `-I`**. It passes.
* **The trap this proves we avoided**: adding
  `#include ".../steel/gemm/mma.h"` to `quantized.h` would make MLX's generator
  expand that quoted include *into* the twin, producing duplicate definitions in
  the JIT string that an AOT-only build cannot see. The cell instead reuses
  `mlx::steel::BaseMMAFrag`, already reachable because `qmm_t_impl` requires
  `mlx::steel::BlockMMA`.

### Prediction scorecard

1. cell peak in 60-95: **MISS -- measured 34.** The band assumed a persistent
   `uint4` weight staging plus separate fragment storage. The shipped cell loads
   one `uint32` per step and keeps `float2` accumulators, so ordinary pressure is
   24, not 60-95.
2. naive count exceeds 108: **HIT** (344) -- and it is a pure counting artifact.
3. binding cell migrates to the narrow `M<=3` cell at 89: **HIT.**
4. the `M in {1,2,3}` arm stays a guard, not a confirmation: predicted effect
   ~ -0.17 %, below the 0.5040 % MDE, so it cannot confirm anything. **HELD.**

### Compliance gates

* `research/twin_audit.py` -> `TWIN AUDIT OK: 29 runtime-effective twin(s)`
* `senpai/validate-assignment-scope.sh efff400c... quantized.h quantized.cpp` -> OK
* `senpai/check-editable-budget.sh efff400c...` -> source 2,466,538 / 3,000,000;
  candidate growth 7,589 / 262,144; exempt 2,410; files 154

### Consequence

Gate 0 passes, so section 7.3 (the paired local microbenchmark) is authorised.
Exactness work stays deferred behind the >= 5 % bar: the matrix unit fixes its
own 8-wide summation order, so this cell is *not* bit-equal to the M=1 readout
and no exactness claim is made or implied here.
