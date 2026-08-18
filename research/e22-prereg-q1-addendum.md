# E22 pre-registration addendum — Question #1, the C(M) staircase

Written and committed **before** the first timed C(M) run, on the advisor's
redirect of the full remaining 4 GPU-hour budget away from the withdrawn A3X
arm (`research/e22-prereg.md` §7 is superseded for that arm only; §8's
measurement protocol still governs).

Host: Apple M4 Pro, `mem=51539607552`. **Not the ranked M5.** Everything below
is directional per-call kernel evidence, never a ranked score claim.

## 1. Question

For `bits == 4` on the real scored shapes, what is the absolute per-call cost
`C(M)` of the quantized mat-vec at verify width `M`, and — the column the depth
policy actually needs — what is `C(M)/M`, the cost per verified row?

`Qwen36MTPBlockSession`'s depth policy uses

```text
threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)
```

with `headStepCostRatio = 0.18`. That form treats the verify cost as **constant
in depth**, but the verify round runs at `M = depth + 1`. If `C(M)/M` is not
flat, the policy is pricing the wrong thing. E17's S18 tops out at depth 4
(M=5) and CURVE at depth 3 (M=4), which straddles the predicted step.

## 2. Where the width actually goes (live source, re-read at HEAD)

`backend/metal/quantized.cpp`: `QuantizedMatmul::eval_gpu` `:1393`, `M` `:1412`,
`vector_limit = get_qmv_batch_limit(K,N,d)` `:1415` (definition `:84-125`),
branch `if (M >= vector_limit)` `:1418`. Every scored shape has `K >= 5120 >
4096`, so `vector_limit == 10` and `M = 1..9` stays on `qmv`. `qmv` `:235`
launches `grid_dims(M, ceil(N/8), B)` `:254`, so `ntg.x == M`; the kernel name
built at `:261-269` contains **no `M` term**. The width therefore selects a
branch *inside* one kernel, not a different kernel.

That branch is the crossrow `switch (ntg.x)` at
`backend/metal/kernels/quantized.h:1822-1955`, gated at `:1822` by
`!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024`, with a
nested `out_vec_size >= 4096` tier at `:1823`. All eight `scoredShapes` have
`n >= 5120`, so all eight land in the `>= 4096` tier.

## 3. Correction to the advisor's P2: the live IPG table is not `ceil(M/ceil(M/4))`

The brief predicted stream boundaries from `IPG = ceil(M/ceil(M/4))`,
`streams = ceil(M/IPG)`, which gives steps at M=4→5 and M=8→9. Reading the
`>= 4096` tier cases verbatim, the shipped template arguments are

| M | callee | IPG | `streams = ceil(M/IPG)` |
|---|---|---|---|
| 2 | `qmv_fast_crossrow_affine4_g64<T, 2>` | 2 (fixed) | 1 |
| 3 | `..._m<T, 3, 3, true>` | 3 | 1 |
| 4 | `..._m<T, 4, 4, true>` | 4 | 1 |
| 5 | `..._m<T, 5, 3, true>` | 3 | 2 |
| 6 | `..._m<T, 6, 3, true>` | 3 | 2 |
| 7 | `..._m<T, 7, 4, true>` | 4 | 2 |
| 8 | `..._m<T, 8, 3, true>` | 3 | **3** |
| 9 | `..._m<T, 9, 3, true>` | 3 | 3 |

`ceil(M/ceil(M/4))` says IPG=4 at M=8; the tree ships 3. So the second stream
step is at **M=7→8, not M=8→9**. This is registered as **P2′** and the sweep
tests it directly. The IPG column is not assumed by the harness either: it is
parsed out of the live header's own template arguments by
`CrossrowGate.weightStreamModel`, so a future retune of the switch cannot
silently invalidate the labels.

## 4. Pre-registered predictions

- **P1 (positive control, not a result).** `C(2)/C(1) ≈ 1.00`. Both are one
  weight stream. Already satisfied by the banked bits-grid data at
  `n = 98336`: `0.99506`, inside the ±1.2% cell spread. Re-measured here on
  every scored shape as an instrument check. If P1 fails on a shape, that
  shape's whole row is suspect and is reported as such rather than interpreted.
- **P2′ (the staircase).** `C(M)` steps up at **M=4→5** and at **M=7→8**, and
  is flat-to-gently-rising within each stream plateau `{2,3,4}`, `{5,6,7}`,
  `{8,9}`.
- **Magnitude is reported, not predicted.** The advisor's standing warning
  applies: the staircase *shape* is expected, its *magnitude* is a live open
  question, and the campaign has already refuted one over-prediction of it. No
  step size is pre-registered. If both steps come back at or below the ~1% cell
  spread, that is a real result and it retires the "stream count drives verify
  cost" family for this shape class.
- **P3 (open check, deliberately not a prediction).** The in-source comment at
  `case 8` asserts non-monotonicity — "M=9 uses three-lane vectors and profiles
  CHEAPER despite more work (319 / 437 / 216 µs for M = 7 / 8 / 9)" — a
  register cliff rather than work scaling. The pass model in §3 predicts the
  opposite ordering: monotone non-decreasing with a step at 7→8 and `C(9) >=
  C(8)`. These cannot both hold. The 319/437/216 figures most likely describe
  the pre-fix `4+4` configuration at `case 8`. **Which one holds is reported as
  measured.** A genuine `C(9) < C(8)` dip would matter a great deal to any
  depth policy that treats verify cost as monotone in depth.
- **P4 (banked, restated for completeness).** The advisor's free falsifiable
  prediction that adding `DIRECT_NIBBLES` to the `<T,2>` pair kernel is worth
  ≈ 0 is **already confirmed** by the M=2 cell: `C(2)/C(1) = 0.99506` puts M=2
  at 243.7 GB/s single-pass-equivalent, +0.5% over the M=1 roofline, i.e. the
  pair kernel is bandwidth-saturated and has no ALU headroom to recover.

## 5. What is measured

`QwenQMVCostCurveTests.sweepQuantizedMatmulOverVerifyWidth`, `bits == 4`,
widths `M = 1..12` over all eight `scoredShapes`. Headline shapes are
`head.lm_head` (k=5120, n=248320) and `mlp.gate_up_fused` (k=5120, n=34816) as
requested. `M = 10,11,12` is past `vector_limit`, so those rows are labelled
`qmm` and their stream model is emitted as null rather than extrapolated —
they are boundary evidence, not part of the contract range.

Reported per cell: `C(M)` absolute, `C(M)/C(1)`, **`C(M)/M`**, the selected
kernel name, the crossrow boolean, IPG, and stream count.

## 6. Stop rule

- The budget is the advisor's 4 GPU-hours, and the run stops earlier if the
  staircase resolves cleanly in both directions — that is, if r1 and r2 agree
  on the presence-or-absence and the sign of both steps on both headline
  shapes.
- Two independent processes (r1, r2). Both are reported; neither is averaged
  away. The banked bits grid reproduced to <= 0.06% across processes at
  `bits == 4`, so a step that does not clear ~1% is reported as "not resolved
  above cell spread", not as a small step.
- Nothing here ships. The depth-policy consequence is written as a proposal
  with its evidence; the policy change itself is the advisor's call.
