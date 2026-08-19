# E36: the two axes compose perfectly in registers — and `values_per_thread` is still dead, for a reason nobody costed

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"max_spill_free_NA_at_values_per_thread_32","available":true,"value":9},"test_metric":{"name":"gate_control_cells_matching_expected_verdict","available":true,"value":1}}
```

**Read the primary metric with its attribution.** `9` is the max spill-free `NA`
at `values_per_thread=32` in thorfinn's `rows_per_simd=2` row-blocked form. It is
also `9` at `values_per_thread=16`. The `+4` over the shipped ceiling of `5` is
**entirely E32's row blocking**; `values_per_thread` contributes **exactly 0**
registers at every cell in this grid. `e36/vpt_attributable_delta_in_max_spill_free_NA = 0`.

- Student / branch: `qwen-askeladd` / `qwen-askeladd/values-per-thread-composes-with-na`
- Hypothesis and target cost: the advisor's model said the two width axes
  contend — that `values_per_thread` scales the x-side register term, so
  `slope = 8.36*(vpt/16) + 3.19*r`, predicting `regs(NA=6, r=2, v=32) ≈ 196` and
  therefore that thorfinn must choose between `NA` and `vpt`. Target cost is the
  same one E32 chased: the second weight pass at `M = 6..9` on the verify path.
- Decision: **complete terminal answer — and the mechanism is not the one that
  was asked about.** The axes compose (register cost of `vpt` is zero, not
  additive). The advisor's register model is **falsified**. But
  `values_per_thread` is independently blocked, first by a hard `K`-coverage wall
  at `v=64` and then, at `v=32`, by the crossrow kernel's own written exactness
  claim. Rung 1 of E33 needs no change.
- `BASE_SHA` `4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549` / `UPSTREAM_SHA`
  unchanged / candidate commit: this branch (research-only).
- Yukon promoted submission / source ref used as frontier: unchanged; nothing
  submittable was produced or proposed.
- Submitted candidate files: **none.** Zero shipped-surface bytes changed.
- Supporting test, tooling, or documentation files (all under `research/`):
  `research/crossrow_vpt_gen.py`, `research/generated/crossrow_vpt_wide.h`,
  `research/crossrow_vpt_probe.metal`, `research/crossrow_vpt_sweep.py`,
  `research/e36-vpt-grid.json`, `research/e36_analysis.py`,
  `research/e36-vpt-analysis.txt`, `research/e36_wandb_log.py`, this report.
- MTP head provenance and draft policy: unchanged — organizer-pinned head, no
  draft policy touched. This experiment never ran a model.
- Assignment-scope preflight: the diff against `BASE_SHA` plus untracked files is
  9 files, all under `research/`. A script that expands all 89 `editablePaths`
  entries from `benchmark.json` and matches them against that file list reports
  `touching editablePaths: NONE`, `all under research/: True`. `quantized.h`,
  `mlx-generated/`, `Sources/` and the shipped
  `static_assert(NA >= 2 && NA <= 5, ...)` are untouched.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `senpai/check-editable-budget.sh 4e5dc2bd…` →
  `source=2455289/3000000 headroom=544711 growth=0/262144 exempt=2410/2147483648 files=154`.
- Scored-path reachability evidence: the sweep's `xship_*` anchor arm calls the
  **shipped** `qmv_fast_crossrow_affine4_g64_wide<T, NA, true>` exactly as
  `quantized.h:1177` calls it, and reproduces E27's measured `NA=2..5` ladder
  digit-for-digit (`62 / 83 / 104 / 125`). The `(c)` section separately walks the
  frozen host dispatch in `backend/metal/quantized.cpp` and enumerates the eight
  scored `(N, K)` shapes from `weights/config.json`.

## Evidence

- Host: Apple M4 Pro, `metal 32023.883`, `air64-apple-darwin25.5.0`.
  **Zero GPU timing. No benchmark, no timing lock, no seconds-per-token, no
  model was loaded.** The entire experiment is `xcrun metal` compilation plus AIR
  register/alloca accounting, exactly as the assignment required.
- W&B: [`cumiyz2s`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cumiyz2s)
  (`finished`) — carries the full 262-cell grid, the 14 gate-control verdicts,
  the `vpt`-invariance table, the `K`-coverage table and the decision table as
  five logged tables.
  Honesty note: an earlier attempt at the same logging, run `fwwozbts`, crashed
  because a W&B `Table` column mixed `bool` and `int` for the control-cell
  `expected`/`observed` fields. The logger now stringifies those two columns.
  `fwwozbts` carries no evidence and should not be cited.
- Exact commands (no GPU, ~3 min wall on 8 jobs):
  ```bash
  python3 research/crossrow_vpt_gen.py                       # derive the probe body
  python3 research/crossrow_vpt_gen.py --check               # content assertion vs quantized.h
  python3 research/crossrow_vpt_sweep.py --out research/e36-vpt-grid.json \
      --jobs 8 --keep-air /tmp/e36-air
  python3 research/e36_analysis.py --grid research/e36-vpt-grid.json \
      > research/e36-vpt-analysis.txt
  python3 research/e36_wandb_log.py research/e36-vpt-grid.json
  ```
- Pipeline: reused from E32 unchanged —
  `xcrun metal -std=metal3.1 -O2 -S` → `xcrun metal-opt -passes='default<O3>'`,
  peak live registers and alloca bytes read from AIR.
  `research/crossrow_vpt_gen.py` **imports** `extract`, `SHIPPED`,
  `TEMPLATE_LINE` and `SIGNATURE` from E32's `research/crossrow_rps_gen.py`
  rather than copying them, so the two experiments cannot drift. It applies 10
  closed rewrites (E32's 4, plus 6 that lift `values_per_thread` out of the
  shipped body). Extracted body sha256 prefix `1dd8c516008c7e01`,
  `quantized.h` lines 968–1066.
- Tests and risk-based checks — **probe validity gates, all passing**:
  - E27's independently measured ladder reproduces digit-for-digit:
    `xship_na2=62`, `na3=83`, `na4=104`, `na5=125`.
  - E27's known-BAD cell reproduces: `NA=6, r=4, v=16` → `144` regs,
    accumulator spill `True` (E27 also measured 144, spilling).
  - The **generated** body at `r=4, v=16` equals the shipped anchors at
    `NA=2..5`: `[62, 83, 104, 125]` — i.e. the rewrites are identity at the
    shipped point.
  - Two forced-spill canaries are compiled every run:
    `PROBE_CELL_FORCED_ACC_SPILL` (gate 1, inherited from E32) and
    `PROBE_CELL_FORCED_STAGE_SPILL` (gate 2, new — it also exercises the
    alloca parser).
  - **14 / 14 control cells matched their pre-declared verdict**;
    `gate_validation_failures: none`. This is the reported test metric.
  - Parser fix that mattered: E32's `ALLOCA` regex truncates nested types like
    `[4 x [4 x i16]]`. E36 uses a balanced-bracket parser and reports
    `private_bytes`, `stage_bytes` and `threadgroup_refs` separately.
    Threadgroup memory is `0` in **all 262 cells** — the crossrow family
    declares none at any `values_per_thread`.
- Exact-token and row-ledger verdict: **not applicable.** No generation was run.
  This is a compile-time resource study; it produces no candidate and makes no
  token-level claim.
- Divergent tokens or failure category: none — see above.
- Generated-twin audit: not applicable. No kernel source was modified, so
  `mlx-generated/quantized.cpp` is untouched. (Recorded for whoever does edit it:
  it is the runtime-effective JIT twin, `+13` line offset, 22 crossrow
  occurrences; a real edit needs both files plus `research/twin_audit.py`.)
- Peak RAM / artifact size: not applicable; the grid JSON is 262 cells.

| Metric | Baseline (`v=16`) | Candidate (`v=32`) | Delta |
| --- | ---: | ---: | ---: |
| `max_spill_free_NA`, `r=2` row-blocked | 9 | 9 | **0** |
| `max_spill_free_NA`, `r=4` shipped form | 5 | 5 | **0** |
| `max_spill_free_NA`, `r=2` grid-relaxed | 12 | 12 | **0** |
| regs at `NA=6, r=2` (M=6 target cell) | 117 | 117 | **0** |
| regs at `NA=9, r=2` (M=9 target cell) | 168 | 168 | **0** |
| private (staging) bytes at `r=2` | 16 | 32 | **+16** |
| register-model slope at `r=2` | 15.00 | 15.00 | **0.000 spread** |

No timing column is reported and none may be inferred: nothing here was timed.

## Results

### (a)+(d) `values_per_thread` costs zero registers — the advisor's model is falsified

Hold `(arm, NA, rows_per_simd)` fixed and sweep only `values_per_thread ∈
{8,16,32,64}`. The register count is *the same number*. Max / mean absolute delta
against the `v=16` column over spill-free cells:

| arm | `NA` range | `v=8` | `v=32` | `v=64` |
|---|---|---|---|---|
| grid_relaxed | 2..9 | 10 / 1.08 | 10 / 0.54 | 27 / 2.77 |
| grid_relaxed | 4..9 | 4 / 0.50 | **0 / 0.00** | 5 / 0.50 |
| coverage_preserving | 2..9 | 10 / 1.40 | 10 / 0.50 | 13 / 1.20 |
| coverage_preserving | 4..9 | 4 / 0.64 | **0 / 0.00** | 2 / 0.29 |

`NA=2,3` is where the allocator does small-cell things and carries the whole max.
From `NA=4` up, every spill-free cell at `r ≤ 2` is **bit-identical** across
`v=8..64`.

Max spill-free `NA` is likewise identical at every `values_per_thread`:

| arm | `r` | `v=8` | `v=16` | `v=32` | `v=64` |
|---|---|---|---|---|---|
| grid_relaxed | 1 | 12 | 12 | 12 | 12 |
| grid_relaxed | 2 | 12 | 12 | 12 | 12 |
| grid_relaxed | 3 | 7 | 7 | 7 | 7 |
| grid_relaxed | 4 | 5 | 5 | 5 | 5 |
| coverage_preserving | 1 | 9 | 9 | 9 | 9 |
| coverage_preserving | 2 | 9 | 9 | 9 | 9 |
| coverage_preserving | 4 | 5 | 5 | 5 | 5 |

(`12` is the largest `NA` compiled; those cells read `≥12, untested above`.)

Per-`(r, v)` line fits over spill-free `NA ≥ 4` give slopes
`r=1 → 11.0`, `r=2 → 15.0`, `r=3 → 18.0`, `r=4 → 21.0`, **unchanged across `v`**
(`r=2` spread `0.000`). Refit: `slope(r) = 8.00 + 3.30·r`, `max|resid| = 0.40`.
Two-dimensional form, with `values_per_thread` carrying zero weight:

```text
regs(NA, r, v) = intercept(r) + (8.00 + 3.30*r)*NA + 0.00*v
```

Scoring the advisor's prediction:

```text
'slope becomes 8.36*(vpt/16) + 3.19*r'  ->  slope(r=2, v=32) = 23.10
predicted regs(NA=6, r=2, v=32) = 16.0 + 23.10*6 = 155   (advisor quoted ~196)
MEASURED                                              = 106
E32's measured regs(NA=6, r=2, v=16)                  = 106
```

The x-side term does **not** scale with `values_per_thread`. **Falsified.**

**Where the cost actually goes:** private (alloca) bytes, and exactly
`stage_bytes = rows_per_simd * values_per_thread / 2` in **every** cell — the
`uint16 packed[r][v/4]` staging array, `r * v/4` words × 2 bytes,
**`NA`-independent**. The two resources are separable with no cross term:

```text
registers      = f(NA, rows_per_simd)         -- values_per_thread free
private bytes  = rows_per_simd * vpt / 2      -- NA free
```

Mechanism: `packed[r][v/4]` is already an alloca in *every* cell including the
shipped clean ones, so growing it spends private memory rather than registers. At
`r=4, v=64` the compiler additionally demotes `scale_local[4]` / `bias_local[4]`
(two `[4 x float]`, `+32` bytes).

The refit's small shift from E32's `8.36 + 3.19·r` is explained, not hand-waved:
E32's `r=1` fit included `NA=11,12`, where the allocator leaves the linear regime
(`156` regs at `NA=12, v=16` with **zero** allocas; `126` at `v=32`). That is what
gave E32's `r=1` row its `5.45` residual. The mechanism reading is unchanged and
slightly better — `8.00/3.30 = 2.42` against the `5:2 = 2.50` ratio of x-side
floats to per-row floats per `NA` (E32 read `2.62`).

### (c) Host-side legality: the grid is safe, `K` coverage is not

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp` is **not** in
`editablePaths` (the 89-entry filter returns `[]` for it), so it is frozen:

```text
:246-259  bn=8, bk=32, group_dims(32,2,1), grid_dims(M, ceil(N/8), B)
:259      bool fast = N % bn == 0 && K % 512 == 0;
:992      the same gate again for gather_qmv
```

A grep of `backend/metal/*.cpp|*.h` (excluding `kernels/`) finds **nothing**
host-side that derives a size, offset or thread count from `values_per_thread` or
`bytes_per_lane`. The grid is a function of `(M, N, B)` only, so raising
`values_per_thread` does **not** disturb the output geometry that killed
`rows_per_simd` in E32. The advisor's expectation here is confirmed.

But `K % 512 == 0` is the **only** alignment the frozen host guarantees, and
`512 == 16 * SIMD_SIZE` is exactly the `vpt=16` block size. The crossrow k-loop
`for (k = 0; k < in_vec_size; k += block_size)` has **no K tail and no bounds
check**; `_m` is a tail over `M`, never over `K`. Scored shapes, from
`weights/config.json` (hidden 5120, intermediate 17408, vocab 248320, 64 layers,
24 Q / 4 KV heads, head_dim 256, affine 4-bit g64):

| N | K | projection | K%1024 | K%2048 | v=8 | v=16 | v=32 | v=64 |
|---|---|---|---|---|---|---|---|---|---|
| 34816 | 5120 | fused gate+up | 0 | 1024 | ok | ok | ok | **OVERRUN** |
| 16480 | 5120 | GDN fused in_proj | 0 | 1024 | ok | ok | ok | **OVERRUN** |
| 14336 | 5120 | fused q+k+v | 0 | 1024 | ok | ok | ok | **OVERRUN** |
| 248320 | 5120 | lm_head | 0 | 1024 | ok | ok | ok | **OVERRUN** |
| 5120 | 6144 | o_proj / GDN out_proj | 0 | 0 | ok | ok | ok | ok |
| 5120 | 17408 | down_proj | 0 | 1024 | ok | ok | ok | **OVERRUN** |
| 5120 | 10240 | MTP head fc | 0 | 0 | ok | ok | ok | ok |
| 2048 | 5120 | KV-only pack (narrow pair kernel) | 0 | 1024 | ok | ok | ok | **OVERRUN** |

`K=5120` and `K=17408` are 1024-aligned but **not** 2048-aligned, and they carry
gate+up, QKV, GDN in_proj, lm_head and down_proj. **`values_per_thread=64` is a
hard coverage wall on the highest-traffic shapes in the model.** `v=32` is
shape-safe *empirically* but is **not guaranteed by the frozen gate**, so a
1024-block kernel would have to self-guard.

An independent cap exists regardless of `K`: one scale/bias is fetched per lane
per k-block at `simd_lid / (64/values_per_thread)`, so a lane must stay inside one
64-element affine group. Upstream writes the same constraint as
`scale_step_per_thread = group_size / values_per_thread` (`quantized.h:768`),
which is `0` — a division by zero — above 64.

### (c2) The real blocker is not in the register file: FP32 lane→K repartition

`values_per_thread` does not merely widen loads. It **repartitions `K` across
lanes**, which reassociates the FP32 sum. The shipped kernel states order
preservation as its own safety case, in its own header:

```text
quantized.h:966  '...the K accumulation order and simd_sum are unchanged for
                  every output element.'
quantized.h:821  the same sentence for the narrow pair kernel.
```

That claim is true today only because both paths partition `K` identically at
`vpt=16`:

```text
quantized.h:786   qmv_fast_impl   x += tid.x*in_vec_size + simd_lid*values_per_thread
quantized.h:1020  crossrow wide   xm = x + ... + k + simd_lid*values_per_thread + 4*i
```

Lane `L` owns `[k+16L, k+16L+16)` in both. Raising `values_per_thread` in the
crossrow kernel alone breaks it:

- `M==1` and `M>9` fall through to `qmv_fast_impl` (`quantized.h:2016`); the
  switch at `:1920` covers `ntg.x` in `2..9` only. So within **one** candidate run
  the same `(x, W)` product would be summed in two different orders.
- The narrow pair kernel (`:873-877`) and the `N<1024` path stay at `vpt=16`.
- **The pinned serial leg is `M==1` throughout** — entirely `vpt=16` — and it is
  the token stream the candidate must match.

scarletbright's shipped `vpt=32` has none of this exposure: it is the 2-bit draft
readout, which `quantized.h:1068-1082` documents as proposal-only and exactly
reranked afterwards, so reassociation there cannot change a token. On the wide
crossrow **verify** path there is no such escape.

Row-blocked `NA` does not have this problem. For a given output element the
row-blocked form runs the identical `k` sequence into the identical `acc[r]`; only
which simdgroup pass computes it moves. **E32's axis is order-preserving by
construction. This one is not.**

### (b) Composition verdict for thorfinn

| M | shipped IPG/passes | best legal NA | passes | widest K-legal vpt | bytes/lane | k-blocks K=5120 | regs | spill | recommended vpt |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 3 / 2 | 6 | 1 | 32 | 16 | 5 | 117 | clean | **16 (unchanged)** |
| 9 | 5 / 2 | 9 | 1 | 32 | 16 | 5 | 168 | clean | **16 (unchanged)** |

Register headroom at those cells is not the binding constraint, and the grid says
so four times each: `M=6, NA=6, r=2` is **117 regs at all of `v=8/16/32/64`**;
`M=9, NA=9, r=2` is **168 at all four**. Only the private staging bytes move
(`8/16/32/64`).

**Plain answer: thorfinn can have `NA`; he cannot usefully have
`values_per_thread`; and the reason is NOT contention.**

- axis 1, row-blocked `NA` — free in registers, order-preserving → ship it;
- axis 2, `values_per_thread` — free in registers (zero cost at every cell in
  this grid), capped at 32 by `K` coverage, and blocked at 32 by the kernel's own
  exactness claim.

The axes compose perfectly in the resource he was worried about, and do not
compose at all in the one nobody costed. **Rung 1 of E33 needs no change.**

### (e) Why did the shipped kernel choose 16? A real reason exists

This had to be answered, because `v=32` *is* spill-free and *is* `K`-legal on
every scored shape. Four candidate reasons were checked; two are real, one is real
but does not transfer, one is dead.

1. **REAL — and it is upstream's whole reason — but it does not transfer.** The
   generic `qmv_fast_impl` materialises `thread U x_thread[values_per_thread]`
   (`quantized.h:774`): `values_per_thread` FP32 registers per lane, live across
   the whole row loop. 16 there costs 16 registers/lane; 32 costs 32. The single
   literal that sets it is `packs_per_thread = bits == 2 ? 1 : 2`
   (`quantized.h:761`), uncommented, and the `bits==2` special case exists
   precisely to hold `vpt` at 16 when `pack_factor` doubles. So the upstream
   invariant is "16 values per lane", and it is a register argument — for a kernel
   that stages all 16. The crossrow kernel **never materialises `x_thread`**: it
   re-reads 4 activations per staged word into `a0..a3`
   (`quantized.h:1014-1035`). That is exactly why this grid measures zero register
   cost. It inherited 16 by copy, not by analysis:
   `git log -S 'constexpr int values_per_thread = 16'` on `quantized.h` returns
   only Validate/Accept submission snapshots, and
   `git log -S packs_per_thread -- Vendor/` returns only the initial squashed MLX
   import.
2. **REAL and binding here:** `512 == 16 * SIMD_SIZE` is exactly the block size
   the frozen host gate `K % 512 == 0` guarantees. 16 is the largest lane width
   that needs no self-guard. See (c).
3. **REAL but not about 16:** `qdot` for `bits ∈ {3,5,6}` advances its weight
   pointer cumulatively inside the loop and is correct only for `vpt ≤ 16`
   (`quantized.h:216-218, 247-248, 267-269`). `bits==4` — our path — uses the
   generic form and needs only `values_per_thread % 4 == 0`.
4. **DEAD:** no reduction assumes a lane covers a quarter of a group. `simd_sum`
   (`:1059`) reduces one scalar per `(row, input)` over 32 lanes and is
   `vpt`-independent; `quad_sum` (`:742`) is reachable only at `K ∈ {64,128}`
   (`quantized.cpp:1385-1387`), which no scored shape hits. Coalescing is not a
   reason either: `bytes_per_lane` goes `8 → 16` contiguous per lane, a better
   burst, not a worse one.

So a real reason exists, it is reason 1, and it is a reason for the **generic**
kernel that was never re-derived for the crossrow one. Reason 2 is a real reason
for the crossrow kernel and survives. Nothing had to be invented, and no
coalescing, shuffle-width or group-size-16 interaction was found, because there is
none.

## Conclusion

- **What happened and why:** the assignment asked whether raising
  `values_per_thread` contends with row-blocked `NA` for registers. It does not —
  it costs **zero** registers at every cell, because the crossrow kernel stages
  weights into an alloca (`packed[r][v/4]`) and re-reads activations rather than
  materialising `x_thread`. The advisor's `8.36*(vpt/16) + 3.19*r` model predicted
  155 registers (quoted ~196) at `NA=6, r=2, v=32`; the measurement is 106,
  identical to `v=16`. The cost of `vpt` is private memory, exactly
  `r*vpt/2` bytes, `NA`-independent.
- **Evidence for or against the mechanism:** 262 compiled cells across two arms ×
  `r ∈ {1,2,3,4}` × `v ∈ {8,16,32,64}` × `NA ∈ 2..12`; probe validity anchored on
  E27's four independently measured points reproducing digit-for-digit, plus 14/14
  control cells and two forced-spill canaries. Separability is exact, not fitted:
  `stage_bytes == r*v/2` in every one of the 262 cells.
- **Prompt or M5 transfer risk:** the register/alloca numbers are compiler output
  for `air64-apple-darwin25.5.0` on an M4 Pro and could shift on the M5 toolchain;
  E27's ladder reproducing exactly is the only cross-host anchor available without
  GPU time. The `(c)`/`(c2)` conclusions are host-independent — they follow from
  frozen source and from `weights/config.json` shapes, not from this compiler.
  **Neither conclusion depends on a timing measurement, so neither carries
  thermal or prompt-sensitivity risk.**
- **Smallest useful next action:** none for this axis. Close it. Rung 1 of E33
  should proceed at `values_per_thread=16` exactly as designed; no register
  re-budgeting is needed.
- **Recommendation: close.** `values_per_thread` on the wide crossrow verify path
  is dead — not because it contends with `NA`, but because at `v=64` it overruns
  `K=5120`/`K=17408` and at `v=32` it reassociates the FP32 sum against the
  `M==1` serial leg the candidate must match. The composition question is answered
  green and needs no follow-up run.

## Suggested follow-ups (not implemented)

1. **`vector_limit` is a material, unverified risk to E33 — worth its own
   assignment.** `get_qmv_batch_limit` (`quantized.cpp:84-126`) returns `10` on
   this M4 Pro (arch_gen is neither 13 nor 14, non-`'d'`, `D,O > 4096`), so
   `M = 1..9` all reach `qmv` and the crossrow switch. But for `arch_gen ∈
   {13,14}`, non-`'d'`, `D,O > 4096` it returns **6** — and `M = 6..9` would leave
   the `qmv` path entirely, taking thorfinn's whole `NA` win with them. **No M5
   arch string exists anywhere in this repository**, so which branch the ranked
   runner takes is unknown. This should be resolved before E33 spends M5 time.
2. **A vpt=32 variant that keeps the lane→K partition.** Unroll the crossrow
   k-loop by 2 — two 512-element blocks per iteration, each summed into its own
   per-512 partial and combined in the original order — to buy wider in-flight
   loads **without** re-association. That sidesteps (c2) entirely and would be
   worth pricing before anyone spends an exactness argument on plain `vpt=32`.
   It does not sidestep the `K % 512` gate for a *2048*-block form, but a
   2×512 form needs no new alignment at all.
3. If a kernel edit ever does land here, `mlx-generated/quantized.cpp` is the
   runtime-effective JIT twin (`+13` line offset, 22 crossrow occurrences); both
   files plus `research/twin_audit.py` are required.
