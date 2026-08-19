# E33 pre-registration — row-blocked single-pass M=6

Written **before** any E33 code exists. Base `4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549`,
host Apple M4 Pro `Mac16,11` (20 GPU cores, 48 GiB, macOS 26.5.2, Swift 6.3.3,
`applegpu_g16s`, NAX off). Branch `qwen-thorfinn/m6-single-pass-rowblocked`.

## 0. Coverage arithmetic, stated before implementation

The host dispatch is frozen and not editable
(`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:251-254`):

```text
bn = 8;  bk = 32;  group_dims(bk, 2, 1);  grid_dims(M, (N + bn - 1) / bn, B)
```

`dispatch_threadgroups` takes threadgroup counts, so for the MLP `N = 17408`:

```text
threadgroups launched in y = (17408 + 7) / 8            = 2176
simdgroups per threadgroup = group_dims.y               = 2
rows a simdgroup must cover (shipped out_row expression) = 4
rows written per threadgroup = 2 simdgroups x 4 rows     = 8
TOTAL rows written = 8 x 2176                            = 17408   == N   OK
```

The candidate keeps every one of those numbers. It only re-orders *how* one
simdgroup covers its 4 rows: `4 / ROWS_PER_SIMD` sequential blocks of
`ROWS_PER_SIMD` rows, at `out_row + b * ROWS_PER_SIMD`. At `ROWS_PER_SIMD = 2`
that is blocks `{out_row, out_row+1}` and `{out_row+2, out_row+3}` — a partition
of the same 4 rows, so `rows written per threadgroup` stays 8 and the product
stays 17408. Nothing in the launch changes; `rows_per_simd < 4` **without** the
`4/r` blocks would write only `2 x 2 x 2176 = 8704` of 17408 rows, which is the
failure mode this arm must not ship.

## 1. Primary metric prediction

`e33/m6_per_row_cost_ratio` = `C_round(6)_candidate / C_round(6)_shipped`,
where `C_round(M) = Σ_shapes calls_per_verify × seconds_per_call` recomputed
identically from each arm's `vendored.json` (the E27 estimator, unchanged).
Baseline 1.0 = shipped `<T,6,3,true>` at 2 weight passes. Direction: minimize.

**My number is 0.85.** Advisor's is 0.82. Derivation, from the current shipped
curve measured in E27 (`.mlxfast-private/qmv-curve/e27-na5-r1`, the arm whose
dispatch table is what ships today):

| estimator | fit | `T₁(6)` | ratio |
|---|---|---:|---:|
| `T = F·M + passes·S` on the 1-pass segment M=3..5 | F=11.6435, S=38.2055 | 108.07 | 0.836 |
| same on the 2-pass segment M=6..9 | F=11.8733, S=29.0200 | 100.26 | 0.776 |
| joint LS over all nine widths | F=8.3385, S=43.7393 | 93.77 | 0.725 |

I take the 1-pass segment fit (0.836) because the candidate M=6 cell **is** a
1-pass cell, then correct it upward for two reasons the fit cannot see:

1. **E27 calibration.** The identical estimator, run on the then-base to predict
   the M=5 2→1 transition, said 0.7726; the transition actually measured
   **0.7990**. This estimator was 3.4 % optimistic on the one transition of this
   exact kind we have ever measured.
2. **Row-blocking is not free on the activation side.** E32's ALU/tile for
   NA=6/r=2-blocked is 1064 against 1000 for shipped 3+3 (+6.4 %), and the x
   vector is loaded and converted once per block instead of once per tile. E27's
   NA=5 paid neither.

Part of E27's 3.4 % miss is plausibly the NA=5 register cost (125 regs) which
NA=6/r=2 (117) does not pay, so I apply roughly two thirds of it:
`0.836 × 1.021 ≈ 0.85`.

**Pre-registered band: [0.78, 0.92].** Outside that band the fixed+stream
decomposition is wrong about this cell and I will say so.

## 2. Register count of the actual production cell

Production cell is `qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true, 2>`
(IPG = NA = 6, `TAIL = 6 % 6 = 0`, so the tail branch is compile-time dead and
one body is emitted per row block).

- Affine model from E32, `r = 2: regs = 16 + 15·NA` → **106** for a single
  un-blocked body.
- E32 measured the *coverage-preserving* (2-block) form `xrb_na6_r2` at
  **117/1 clean**, and my production cell has that same structure plus the `_m`
  wrapper's `first_m >= M` early return.

**Prediction: 117 peak live registers, 1 alloca, no accumulator-typed spill.**

How far it may differ before I call the model broken:

- `regs ∈ [102, 125]` → model intact. 125 is not an arbitrary bound: it is the
  shipped `<T,5,5,true>` high-water mark already in production and already
  measured fast, so anything at or below it is inside proven territory.
- `regs ∈ (125, 140]` and no spill alloca → model degraded but the cell is still
  shippable; I report the deviation and let timing decide.
- an alloca whose type is the accumulator array (`[2 x <6 x float>]`) at any
  register count → **model broken and cell rejected**; I stop before timing.

## 3. Untreated-width control

M = 3, 4, 5 change no dispatch line and must be flat. M = 7, 8, 9 are also
untreated in rung 1 and are additional controls.

**Prediction: |ratio − 1| ≤ 0.01 for every one of M = 3, 4, 5, 7, 8, 9.** E27's
five mid-sweep untouched widths all landed within ±0.5 %, so ±1 % is a
two-fold-generous noise floor. M = 1 is excluded from the control set: it is the
first width measured on the coldest GPU and drifted −7.1 % in E27 for that
reason. It is still measured (E27's own lesson: **always include width 1**, or
`qmv_cost_curve_summary.py` normalisation fails).

## 4. Kill criteria

**Hard stop — report immediately, ship nothing** (any one of):

- the coverage arithmetic in §0 fails to close at 17408;
- QMV parity digests differ in any (shape, bits, M) cell between shipped and
  candidate;
- `all_tokens_matched` false, `residual_divergence_count > 0`, or
  `parity_all_ok` false in either leg;
- `effective_draft_lengths` is not element-wise identical to the shipped build.

A weight-pass regrouping is a pure re-association of the same per-element
arithmetic — each output element keeps its own k-order, its own `simd_sum`, and
its own scale/bias — so any of these firing means the regrouping is **not**
arithmetically neutral and nothing else in the experiment matters.

**Static stop:** accumulator-typed spill alloca at the production cell → report
the static negative, do not spend a timing allocation.

**Kill (negative result, do not ship):** `e33/m6_per_row_cost_ratio ≥ 0.97`.
Three times the ±1 % control floor; below that the mechanism has not repaid the
+6.4 % ALU and the extra activation pass and I report it as a negative.

**Ambiguity stop:** if the M=6 ratio lands in [0.94, 0.97] — measurably better
than the controls but under the kill line — I report rung 1 as unclear and do
**not** start rung 2.

## 5. Scope

Submitted paths touched: `kernels/quantized.h` and its runtime-effective twin
`mlx-generated/quantized.cpp` only, edited identically (twin-locked, offset +13).
Not touched: `costModelDepth`, `headStepCostRatio`, `sdpaWidthWallDepthCap`,
`segmentedVerifyDepthCap`, the host dispatch, `mtp-head.manifest.json`.
`static_assert(M % IPG != 1, ...)` stays. IPG=5 at M=6 is illegal (`6 % 5 == 1`)
and would still be 2 passes, which is why the M=6 answer is NA=6, not NA=5.
