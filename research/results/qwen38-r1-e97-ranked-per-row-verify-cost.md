# E97 — what a marginal verify row buys: the per-row axis is closed

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e97/g2_per_row_slope_fraction_of_achievable_affine4_peak","available":true,"value":0.827},"test_metric":{"name":"session_null_max_abs_drift_pct","available":true,"value":0.23}}
```

**Decision: dead as a lever, green as measurement.** The marginal verify row is
already arithmetic running at 83–86 % of the arithmetic ceiling this GPU
actually reaches. Hypothesis (B), FMA throughput, is confirmed. (A), per-row
dequantisation, is refuted three ways. (D), activation re-loads, is refuted.
(C), occupancy and register pressure, survives only as the residual: a
K-independent 7–10 % per-row overhead plus a ~20 % penalty for adding a row
inside an `IPG = 4` kernel instead of an `IPG = 3` kernel.

No candidate change is proposed. Nothing on the per-row axis is worth
+1 % ranked.

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e97-ranked-per-row-verify-cost`, PR #98, revision `r1`
- Assignment: `qwen38-r1-e97-ranked-per-row-verify-cost`
- Hypothesis and target cost: the ranked cost curve prices a verify row at
  7,232.4 us on M5 and the E95 width model prices it at 10,268 us over the whole
  trunk. Neither says what the row buys. Attribute that cost between (A)
  per-row dequantisation, (B) FMA throughput, (C) occupancy and register
  pressure, and (D) activation re-loads.
- Decision: **dead** for a candidate change; **green** as attribution
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: branch base
  `1d5445176559a58ccc3cfe7aefdac9ef3d879acc`; budget contract base
  `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`; **no candidate commit** — this is
  a research-only result
- Yukon promoted submission / source ref used as frontier: unchanged; this
  experiment proposes no submission
- Candidate build fingerprint: not applicable; no submitted file changed
- Submitted-surface / generated-twin / metallib digests: unchanged.
  `python3 research/twin_audit.py` reports
  `TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)`
- **Submitted candidate files: none.** The complete diff against the assignment
  head `0e3ff4c8` touches 9 files: 3 under `Tests/` and 6 under `research/`. A
  programmatic check against every `editablePaths` entry in `benchmark.json`
  (89 entries) reports `SUBMITTED SURFACE TOUCHED: NONE`.
- Supporting test, tooling, or documentation files:
  `Tests/MLXFastTests/E97VerifyRowCostTests.swift` (new, 477 lines, three
  opt-in rungs), `research/e97_row_cost_probe.sh`,
  `research/e97_row_cost_analysis.py`, `research/e97_metadata_census.py`,
  `research/e97_wandb_log.py`, `research/e97_round_reconstruction.py`, and the
  operational fix in
  `Tests/MLXFastTests/E95QmvWidthProbeTests.swift` and
  `Tests/MLXFastTests/E95DonationProbeTests.swift`
- MTP head provenance, digest, and draft policy: not applicable. No model, no
  fixture, no worker and no proposal head is loaded by any rung. Every rung
  times one isolated MLX operation.
- Token window, fixture, reference source, and harness: **no token window and
  no fixture.** `harness=local` on every leg. No leg is a score.
- Exact cell: `qmv_fast_crossrow_affine4_g64_wide<T, NA, DIRECT_NIBBLES>`
  reached through `qmv_fast_crossrow_affine4_g64_m<T, M, IPG, true>` at
  `K = 5120`, `N ∈ {34816, 248320}`, `M = 1..9`; and `qmm_splitk` at
  `M = 10..32`. Source form: JIT from `mlx-generated/quantized.cpp`. M5 variant:
  not measured; this host is `applegpu_g16s`, arch gen 16.
- Official causal path and score equation: **none claimed.** Every number here
  is `harness=local`. This experiment measures a cost shape, not a score, and
  no local ratio is presented as a ranked term.
- Assignment-scope preflight: no submitted path changed, so
  `senpai/validate-assignment-scope.sh` has no submitted path to check.
  `senpai/verify-ranked-score-boundary.sh` reports
  `PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only`
- Editable source bytes / headroom / growth / exempt-head bytes:
  `editable budget OK: source=2515544/3000000 bytes headroom=484456 growth=60709/262144 exempt=2410/2147483648 files=154`
- Scored-path reachability evidence: the probe calls the same
  `quantizedMM(transpose: true, groupSize: 64, bits: 4, mode: .affine)` entry
  point the model uses. `get_qmv_batch_limit(5120, N) == 10` for both widths on
  arch gen 16, and the WIDE table at
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:1917`
  is entered for `bits == 4 && group_size == 64 && out_vec_size >= 4096`, which
  both widths satisfy.

---

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy:
  Apple M4 Pro, 20 GPU cores, 48 GiB, `ip-10-231-2-227.ec2.internal`,
  `swift-driver 1.148.6 / Apple Swift 6.3.3 (swiftlang-6.3.3.1.3 clang-2100.1.1.101)`.
  **Thermal policy: cool gate off.** Every rung is a within-session
  counterbalanced comparison, so `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false` and `timing_valid=false` are recorded
  verbatim in every `meta.txt` and on every W&B run. No leg is gate-qualified
  and no leg is an official or ranked score.

  | rung | entry GPU C | exit GPU C |
  | --- | ---: | ---: |
  | 0 peak | 35.08 | 64.49 |
  | 1 row | 34.9 | 40.1 |
  | 2 shape | 35.00 | 53.74 |

- `head_provenance_sha256`: not applicable. No proposal head and no worker
  artifact is loaded by any leg of this experiment.

- Exact commands:

  ```bash
  bash research/e97_row_cost_probe.sh e97-peak-r0    peak      # rung 0
  bash research/e97_row_cost_probe.sh e97-row-cost-r1 row      # rung 1
  bash research/e97_row_cost_probe.sh e97-shape-r2   shape     # rung 2
  python3 research/e97_metadata_census.py --json research/out/e97-census/census.json
  python3 research/e97_row_cost_analysis.py \
      --peak  research/out/e97-peak-r0/peak.json \
      --shape research/out/e97-shape-r2/shape.json \
      --json  research/out/e97-row-cost-r1/analysis.json
  python3 research/e97_round_reconstruction.py \
      > research/out/e97-reconstruction/reconstruction.json
  python3 research/e97_wandb_log.py
  ```

- W&B runs, all `finished`, group `e97-ranked-per-row-verify-cost`:

  | run | id | URL |
  | --- | --- | --- |
  | `e97-peak` | `tjl6xim7` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/tjl6xim7 |
  | `e97-row` | `ch0owi0d` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ch0owi0d |
  | `e97-shape` | `pflg8on2` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/pflg8on2 |
  | `e97-census` | `ywzutsuy` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ywzutsuy |
  | `e97-recon` | `vykpmmlf` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/vykpmmlf |

- Cheapest real falsification gate and positive-control verdict: the **session
  null**. One identical cell is repeated at the open and the close of every
  session. If drift could manufacture a slope, the null moves.

  | rung | null | drift |
  | --- | --- | ---: |
  | 1 | `affine4 M=4 O=34816` | **−0.23 %** |
  | 1 | `affine4 M=4 O=248320` | **+0.20 %** |
  | 2 | `affine4 M=6`, 8 of 10 cells | −0.43 % to +0.70 % |
  | 2 | `affine4 M=6 O=34816 K=1024` | +6.18 % |
  | 2 | `affine4 M=6 O=34816 K=2048` | +2.56 % |

  The two large rung-2 nulls are the two shortest cells, where the fixed
  per-eval cost is a large share of the measurement. Their band slopes carry
  the largest standard errors in the sweep and the K-fit weights them like any
  other point; excluding them changes `b` by less than its standard error.

  A second control is the **per-eval overhead**, measured independently at the
  open and close of each session: 139.28 → 134.60 us (rung 1) and
  139.53 → 134.12 us (rung 2). It is a constant in M, so it cannot bias a
  slope, but it is subtracted before every rate.

  A third control is the **bf16 arm** of rung 1, which does the same
  multiply-accumulates on the same shapes and unpacks nothing. It is the
  positive control for (A).

- Tests and risk-based checks, in execution order:
  1. `senpai/verify-ranked-score-boundary.sh` — PASS
  2. `python3 research/twin_audit.py` — OK, 29 runtime-effective twins
  3. `senpai/check-editable-budget.sh 770a3ff2` — OK, growth 60,709 of 262,144
  4. programmatic `editablePaths` intersection of my diff — empty
  5. rung 0, rung 1, rung 2, census, round reconstruction, W&B publish — all
     exit 0

- Exact-token and row-ledger verdict: **not applicable.** No generation runs in
  this experiment and no submitted file changed, so there is no token stream to
  match and no row ledger to close.

- Divergent tokens or failure category: none.

- Generated-twin audit: OK; nothing under `Vendor/` was touched.

- Peak RAM or head/artifact size: the largest working set is the rung-2
  `N = 248320, K = 5120` pair, about 3.2 GiB of dense plus packed weights.
  `Memory.clearCache()` runs between reduction lengths.

---

## 1. Rung 0 — the ceiling this GPU actually reaches

Every later fraction needs a measured denominator, not a specification number.
Rung 0 drives the same MLX build the scored path uses at shapes large enough to
saturate it.

| form | best shape | TFLOP/s |
| --- | --- | ---: |
| dense square matmul, f16 and bf16 | 8192³ | **7.586** |
| dense batched, bf16, scored widths | 1024 × 5120 × 248320 | 7.478 |
| **affine-4 group-64 batched, the scored weight form** | 1024 × 5120 × 248320 | **6.568** |
| affine-4 group-64 batched | 256 × 5120 × 34816 | 6.482 |
| dense square matmul, f32 | 8192³ | 6.698 |

The affine-4 ceiling is stable at 6.48–6.57 TFLOP/s across four shapes and two
widths. Dequantisation costs this machine a fixed 12.2 % of its dense ceiling,
at every batch size that saturates it. **That 12.2 % is the total price of
unpacking when the kernel is arithmetic-bound**, and it is the right
denominator for a kernel that must unpack.

## 2. Rung 1 — the marginal row, and the refutation of (A)

`us(M) = intercept + slope · M`, fitted inside each group band, 8 ABBA
counterbalanced blocks, replicate counts chosen per cell so every cell
integrates about 120 ms.

| arm | band | widths | slope us/row | se | TFLOP/s | of affine-4 peak | of dense peak | R² |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affine4 / 248320 | G1 | 2,3,4 | 273.55 | 24.18 | 9.30 | *141.5 %* | *122.5 %* | 0.853 |
| **affine4 / 248320** | **G2** | 5,6,7,8 | **468.08** | **3.96** | **5.432** | **82.7 %** | **71.6 %** | 0.998 |
| affine4 / 34816 | G1 | 2,3,4 | 81.21 | 7.40 | 4.39 | 66.8 % | 57.9 % | 0.845 |
| **affine4 / 34816** | **G2** | 5,6,7,8 | **63.34** | **1.19** | **5.628** | **85.7 %** | **74.2 %** | 0.989 |
| bf16 / 248320 | G2 | 5,6,7,8 | 18.82 | 5.27 | — | — | — | 0.298 |
| bf16 / 34816 | G2 | 5,6,7,8 | 1.20 | 0.82 | — | — | — | 0.067 |

M = 1 is excluded from every fit. `ntg.x == 1` has no case in the WIDE switch,
so `M = 1` runs the generic `qmv`, and the bf16 arm leaves `gemv` for the steel
GEMM at `M = 2`. A fit that spans `M = 1` straddles a kernel change in both
arms and its slope is meaningless. The rung-1 analysis was corrected for this
before any conclusion was drawn.

### The decision rule fires

The advisor's rule: a marginal FLOP rate at or above 70 % of achieved peak
closes the axis.

- Measured, local, same host, same build: **82.7 %** at the `lm_head` width and
  **85.7 %** at the `mlp.gate_up` width, against the affine-4 ceiling. Even
  against the dense ceiling, which the scored kernel cannot legally reach
  because it must unpack, the fractions are 71.6 % and 74.2 %.
- Transfer of the advisor's ranked figure, **labelled as a cross-host
  extrapolation and not a measurement of this session**: the ranked
  4.74 TFLOP/s divided by this host's affine-4 ceiling is 72.2 %, and divided
  by this host's dense ceiling is 62.5 %. The ranked host is M5 and this host
  is M4 Pro, so this ratio is directional only. The local same-host ratio above
  is the defensible one, and it is higher, not lower.

Every reading clears 70 %. **The per-row axis is closed.** Total remaining
headroom, assuming a perfect kernel that reaches the affine-4 GEMM ceiling in a
vector dispatch, is 17.3 % of the row at `O = 248320` and 14.3 % at
`O = 34816`.

### (A) per-row dequantisation is refuted three ways

1. **Source.** In `qmv_fast_crossrow_affine4_g64_wide`, the nibble mask, the
   integer-to-float conversion and the scale and bias loads are done once per
   k-block and shared across all `NA` lanes. Per k-block a thread issues 16
   weight `uint16` loads and 8 scale and bias loads **independent of NA**; only
   the activation loads and the FMAs scale with NA. Dequantisation is in the
   intercept by construction, not in the slope.
2. **The split-K control.** The quantized `qmm_splitk` kernel at `M = 10..32`
   unpacks exactly the same nibbles and has a per-row slope of **5.82 us**
   (`O = 248320`) and **0.83 us** (`O = 34816`) — 1.2 % and 1.3 % of the vector
   slope, flat all the way to `M = 32`. A kernel that dequantises just as hard
   and reaches 6.49 TFLOP/s at `M = 32`, which is 98.8 % of the affine-4
   ceiling, cannot leave a large per-row dequantisation cost in the vector
   kernel.
3. **The rate itself.** 5.43 TFLOP/s is 82.7 % of a ceiling that *already
   includes* dequantisation. There is no room left for a second dequantisation
   charge.

The bf16 arm of rung 1 was designed as the direct control but **does not
adjudicate (A)**, and I report that honestly: MLX routes bf16 at `M ≥ 2` to the
steel GEMM, which tiles the activations and re-reads nothing per row. Its slope
is 1.2–18.8 us/row with `R² ≤ 0.30`, that is, statistically flat. That is a
different algorithm, not a dequant-free version of the same one. Its measured
`M = 1 → 2` step of 1.18–1.20 × the full weight read is the gemv-to-GEMM
transition and its transpose, which also explains why its intercept is not
comparable. The three arguments above stand without it.

## 3. Rung 2 — splitting (B) from (C), and refuting (D)

`get_qmv_batch_limit` (`backend/metal/quantized.cpp:84`) returns 10 for **every
K in this sweep**, because `out_vec_size > 4096` puts both widths in the final
`else` arm for any architecture that is not `'d'`-suffixed and is not
generation 13 or 14. This host is `applegpu_g16s`, so that arm is the live one.
The WIDE table then switches on `ntg.x` alone. Sweeping the reduction length K
therefore holds the kernel, the launch geometry, the row-group split and the
register allocation fixed while it scales the multiply-accumulates and the
activation traffic linearly. The `G == 2` band slope was refitted at eight
reduction lengths.

| N | K | slope us/row | se | TFLOP/s | of affine-4 peak | R² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 34816 | 1024 | 20.14 | 0.97 | 3.54 | 53.9 % | 0.951 |
| 34816 | 2048 | 31.37 | 1.63 | 4.55 | 69.3 % | 0.944 |
| 34816 | 3072 | 37.65 | 2.29 | 5.68 | 86.5 % | 0.925 |
| 34816 | 4096 | 52.00 | 0.62 | 5.48 | 83.4 % | 0.997 |
| 34816 | 5120 | 61.53 | 7.42 | 5.79 | 88.2 % | 0.757 |
| 34816 | 6144 | 79.74 | 0.83 | 5.36 | 81.6 % | 0.998 |
| 34816 | 7168 | 89.30 | 1.44 | 5.59 | 85.1 % | 0.994 |
| 34816 | 8192 | 99.17 | 3.34 | 5.75 | 87.5 % | 0.976 |
| 248320 | 2560 | 250.01 | 2.22 | 5.09 | 77.5 % | 0.998 |
| 248320 | 5120 | 467.09 | 4.77 | 5.44 | 82.8 % | 0.998 |

Fitting `s(K) = a + b·K`:

| N | b (ns/row/K) | a (us/row) | R² | a as a share of the slope at K = 5120 | reduction-only rate | of affine-4 peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 34816 | 11.378 ± 0.415 | **+6.43 ± 2.14** | 0.9921 | **9.95 %** | 6.120 TFLOP/s | **93.2 %** |
| 248320 | 84.795 (2 points) | **+32.94** | — | **7.05 %** | 5.857 TFLOP/s | **89.2 %** |

Two conclusions, both sharp.

**(B) is confirmed and there is nothing left in the inner loop.** Once the
K-independent per-row overhead is removed, the reduction-scaled part of the
marginal row runs at **89–93 % of the affine-4 arithmetic ceiling of this
machine**. Its own remaining headroom is 7–11 %, and that is the headroom of a
vector kernel measured against a fully tiled GEMM at batch 1024.

**(C) exists but is small, and it splits in two.**

- *The K-independent part* is `a`: 6.43 us/row at `O = 34816` and 32.94 us/row
  at `O = 248320`, which is 7.0–10.0 % of the row at the scored `K = 5120`.
  This is launch, group setup and the idle x-group the host always dispatches
  (`grid.x = M`, of which only `G = ceil(M/IPG)` groups do work). It is the
  only part of the per-row cost that does **not** scale with the reduction.
- *The throughput part* is a matched contrast inside the same band. `M 5 → 6`
  adds a row at `IPG = 3` (NA 2 → 3); `M 7 → 8` adds a row at `IPG = 4`
  (NA 3 → 4). Both steps add one row, one idle x-group and the same number of
  multiply-accumulates.

  | N | K | step NA 2→3 (us) | step NA 3→4 (us) | excess |
  | ---: | ---: | ---: | ---: | ---: |
  | 34816 | 4096 | 46.87 | 55.75 | +19.0 % |
  | 34816 | 6144 | 71.59 | 82.62 | +15.4 % |
  | 34816 | 7168 | 80.55 | 93.50 | +16.1 % |
  | 34816 | 8192 | 84.66 | 107.44 | +26.9 % |
  | 248320 | 2560 | 222.17 | 264.00 | +18.8 % |
  | 248320 | 5120 | 407.68 | 500.07 | +22.7 % |

  Mean excess over these six cells: **+19.8 %**. The three shortest reduction
  lengths (`K ≤ 3072`) are noisy — their absolute steps are 9 to 28 us — and
  are excluded from that mean but are reported in the W&B table.

  **Adding a row inside an `IPG = 4` kernel costs about 20 % more than adding a
  row inside an `IPG = 3` kernel.** The excess grows with K, so it is a
  throughput effect, not a launch effect: a wider accumulator vector, more live
  registers, lower occupancy. This is the E13 register cliff, measured here at
  the margin instead of at the level.

**(D) activation re-loads is refuted.** Per k-step the WIDE kernel loads one
bf16 activation per row and issues 8 multiply-accumulates against it, so the
activation arithmetic intensity is 8 FLOP per byte. Sustaining the measured
5.43 TFLOP/s therefore needs 679 GB/s of activation traffic, which is
34 GB/s per GPU core on this 20-core part — well inside L1 and never a DRAM
cost, because the whole activation vector is 10.2 KB. The measurement agrees:
if activation traffic were the binding term, the reduction-scaled part could
not sit at 89–93 % of a ceiling produced by a kernel that tiles activations
into threadgroup memory and pays no such cost.

## 4. Rung 3 — the slope step at the group boundary is not arithmetic

A band fit averages over the template changes inside the band, so I report the
model-free per-increment table. Each single added row is priced two ways: as
arithmetic, and as a fraction of the single-group weight read measured at
`M = 1`.

`O = 248320`, `K = 5120`, net microseconds:

| step | us | new group | IPG | TFLOP/s if arithmetic | × the M=1 weight read |
| --- | ---: | :---: | --- | ---: | ---: |
| 1 → 2 | −24.5 | | 1→2 | — | −0.008 |
| 2 → 3 | +77.7 | | 2→3 | 32.73 | 0.027 |
| 3 → 4 | +469.4 | | 3→4 | 5.42 | 0.160 |
| **4 → 5** | **+1907.1** | **NEW** | 4→3 | **1.33** | **0.651** |
| 5 → 6 | +412.4 | | 3→3 | 6.17 | 0.141 |
| 6 → 7 | +485.9 | | 3→4 | 5.23 | 0.166 |
| 7 → 8 | +500.0 | | 4→4 | 5.09 | 0.171 |
| **8 → 9** | **+1915.0** | **NEW** | 4→3 | **1.33** | **0.654** |
| 9 → 10 | +3858.1 | leaves the vector kernel | — | 0.66 | 1.317 |

`O = 34816`, same session: the two group-crossing steps are +271.2 us
(**0.640** of the M=1 read) and +258.1 us (**0.609**); the five in-band steps
are 24.5 to 137.9 us (0.058 to 0.325).

### The mechanism, proved from source

`qmv_fast_crossrow_affine4_g64_m` computes

```c++
const int first_m  = int(tid.x) * IPG;   // which input rows
const int out_row  = int(tid.y) * 8 + int(simd_gid) * 4;   // which weight rows
```

The weight rows a threadgroup reads depend on `tid.y` **only**. Every one of the
`G = ceil(M / IPG)` x-groups with the same `tid.y` reads exactly the same
weights, scales and biases. **The weight tensor is streamed `G` times.**

### The two readings, and which one survives

The identical unit of work — one added input row — is priced at 5.09 to
6.42 TFLOP/s inside a band, that is 78 to 98 % of this machine's affine-4
ceiling, and at 1.31 to 1.38 TFLOP/s when it crosses a group boundary, that is
20 to 21 %. No arithmetic mechanism makes the same row 4.7 × slower.

Read as bandwidth instead: 715 MB in 1,907 us is 375 GB/s, which is 1.25 × the
measured single-stream read rate of this tensor. That is what two threadgroups
reading the same cache lines look like when part of the traffic is served
without a second DRAM trip. The bandwidth reading is physically consistent; the
arithmetic reading is not.

The fraction is **0.609 to 0.654 of a full weight read at both boundaries and
at both widths**, across a 7.1 × change in tensor size. About 37 % of the
second stream is absorbed; 63 % is paid.

A whole-curve check: with a single effective read rate of 250 GB/s,
`time ≈ G × (bytes / rate)` predicts 2,860 / 5,720 / 8,580 us for
`G = 1 / 2 / 3` at `O = 248320`, against measured 2,905 (M=2), 5,359 (M=5) and
8,672 (M=9) — within 2 %, 7 % and 1 %. The vector-regime cost curve is
dominated by the weight stream, and the per-row arithmetic rides on top of it
inside each band.

### This replicates E27's staircase and prices the term that stopped E33 and E38

This is an **independent replication** of E27's `weight_streams = ceil(M/IPG)`
staircase, from a different instrument: no model, no fixture, no worker, one
isolated `quantizedMM`. E27 measured in-model round-weighted steps of 1.4559 at
`M 4 → 5` and 1.2759 at `M 7 → 8`. Those are the boundaries of the WIDE table
that E27's own base shipped. The current table moves the second boundary to
`M 8 → 9`, so the two experiments do not agree cell by cell — they agree that
the staircase exists, that each riser is a large fraction of a full weight
read, and that the risers sit exactly where `ceil(M/IPG)` increments in
whichever table is shipped. Same staircase, boundary positions moved with the
table.

It is **not a new prize.** The campaign has already falsified the two obvious
levers that remove a stream:

- **E33** (row-blocked single pass at `M = 6`): the mechanism engaged, `M = 6`
  really became one weight pass, and the candidate was **1.50 % slower**.
- **E38** (row blocks in the idle x-blocks at `M = 6`): drift-adjusted ratio
  **0.9858** (raw 0.9891) against a registered prediction of 0.84, so
  **falsified**. Its decomposition is the useful part: the removed second
  weight pass is worth **−11.96 %**, the row-blocking tax is **+10.54 %**, grid
  thinning is **+2.93 %**, and the net is **−1.09 % raw / −1.42 %
  drift-adjusted** — a wash.

Rung 2 supplies an independent price for the term that eats the prize. Removing
one stream at `M = 6`, `O = 248320` returns about 0.65 × 2,929 = **1,904 us**,
which agrees in size with E38's −11.96 %. But the replacement must contract all
6 rows in one group, that is `NA = 6` instead of `NA = 3`, and the matched NA
contrast prices each widening at about **+20 % per row** in isolation. So the
saving and the penalty are the same order, and the measured net is the small
residual between two large opposed terms.

E38's compiled-AIR register counts are the mechanism behind that price, and
they are independent evidence for it: `peak_live_regs` for the `crossrow`
kernel at `r = 4` is `na2 = 62`, `na3 = 83`, `na4 = 104`, `na5 = 125`,
`na6 = 144`, i.e. **+21 registers for every NA step against a 128-register
wall**. `NA = 6` does not fit, so `r = 2` is forced, and that forcing *is*
E38's +10.54 %. My +19.8 % per-IPG-step is a time-domain measurement of the
same ladder taken with no model and no row blocking; E38's is a static
register-domain measurement. They are not the same arm and I do not claim my
number reproduces his, but they agree that widening the per-group accumulator
is expensive and that the cost grows step by step rather than at one cliff.

**Prediction, falsifiable:** any future lever that reduces `G` will fail unless
it reduces `G` **without widening `NA`** — for example by giving each group more
output columns instead of more input rows, so that the extra concurrency comes
from `tid.y` rather than from a wider register vector. Every step of `NA` is
about +20 % of the row in time and +21 registers in the compiled AIR, and the
shipped `<T,5,5>` cell already sits at 125 of 128. That is the only direction on
this axis that rung 2 does not already price as self-cancelling.

## 5. Item 5 — the lossless (scale, bias) cardinality census

CPU only. Reads the safetensors headers of the transformed checkpoint directly,
loads no model and touches no GPU. Counts **bit patterns**, not decoded floats,
so two bf16 values are the same exactly when their 16 bits agree, with the two
zeros folded together.

| quantity | value |
| --- | ---: |
| quantized tensors | 498 |
| affine-4 group-64 groups | 420,208,640 |
| metadata bytes in the checkpoint | 1,680,834,560 (1.68 GB) |
| **tensors with ≤ 256 distinct (scale, bias) pairs** | **0 of 498** |
| minimum distinct pairs in any tensor | 911 |
| median distinct pairs | 2,658 |
| maximum distinct pairs | 7,846 |
| maximum distinct scales alone | 2,382 |
| maximum distinct biases alone | 2,496 |
| lossless saving from an 8-bit pair table | **0 bytes** |

**A lossless 8-bit (scale, bias) table is impossible for every tensor in this
checkpoint**, by a factor of 3.6 at the easiest tensor and 30.6 at the hardest.
Separate 8-bit tables for scale and for bias are also impossible: both need
12 bits.

The census does bound the smallest lossless form, which is a new number and not
part of the assignment's question: 13 bits index every tensor exactly
(7,846 < 8,192), and an aligned `uint16` index is lossless for all 498 tensors
with room to spare. An affine-4 group-64 record is 32 bytes of nibbles plus
4 bytes of metadata, so replacing the pair with a `uint16` index removes 2 of
every 36 weight-stream bytes: a **5.56 % cut in weight traffic**, for
5.17 MB of pair tables across the whole checkpoint. Given section 4, where the
vector-regime cost curve is weight-stream dominated, that is the one number in
this report that could matter. It needs an offline transform change plus one
extra indirection per group in the kernel, and it is not implemented here.

---

## 6. Advisor feedback f2 — the concurrent encoder

Feedback f2 arrived after every rung had run. It adds one threat to this
report, asks one method question, and repeats one housekeeping item.

### 6.1 Reading 3 is dead: the marginal row overlaps nothing

The threat is reading 3. MLX opens every compute encoder with
`MTL::DispatchTypeConcurrent` (`backend/metal/device.cpp:545-548`), so
dispatches inside one command buffer are unordered unless a buffer dependency
forces a barrier. If the marginal verify row already overlapped other round
work, its effective rate would not be a rate and comparing it with a peak would
be meaningless.

Two facts settle this without a GPU leg.

**First, the E97 probe is already the serialised geometry.** `timed()` calls a
blocking `eval` inside the loop
(`Tests/MLXFastTests/E97VerifyRowCostTests.swift:94-101`), so every replicate is
its own submit and drain and no two replicates overlap. By the advisor's own
rule (b), an isolated time is an **upper bound** on in-situ contribution. The
measured `eval` floor is subtracted from every cell, and a slope is a
difference, so the floor cancels twice over.

**Second, that upper bound lands below the round-level marginal.**
`research/e97_round_reconstruction.py` splits the rung-2 fit into three terms:
a per-dispatch constant `p = 2.107 us/row`, an output-column term
`q_a = 0.12417 ns/row/column`, and a per-multiply-accumulate term
`q_b = 0.33414 ps/row/MAC`. It then extrapolates that fit over the six
quantized-projection classes of one decode round, taken from `QMV_CLASSES` in
`research/e95_verify_census.py`.

| class | K | N | dispatches | us/row each | us/row total |
| --- | ---: | ---: | ---: | ---: | ---: |
| full-attention fused QKV+gate | 5,120 | 14,336 | 16 | 28.41 | 454.6 |
| GDN `in_proj` | 5,120 | 16,480 | 48 | 32.35 | 1,552.7 |
| MLP `gate_up` fused | 5,120 | 34,816 | 64 | 65.99 | 4,223.5 |
| `out_proj` | 6,144 | 5,120 | 64 | 13.25 | 848.2 |
| MLP `down_proj` | 17,408 | 5,120 | 64 | 32.52 | 2,081.6 |
| `lm_head` | 5,120 | 248,320 | 1 | 457.76 | 457.8 |
| **isolated sum** | | | **257** | | **9,618.4** |
| reduction term only | | | | | 8,561.3 |
| output-column term | | | | | 515.6 |
| per-dispatch launch term | | | | | 541.5 |

```text
isolated sum          9,618.4 us/row   (E97, isolated quantizedMM, this session)
round-level marginal 10,268.0 us/row   (E95 rung 2, confirmed by E92)
ratio                    0.9367
```

An upper bound that reconstructs **93.7 %** of an independently fitted
round-level marginal leaves no room for a concurrency discount. If the marginal
row overlapped other work, the isolated sum would have to *exceed* the
round-level marginal, not fall 6.3 % short of it. The 650 us/row shortfall is
the per-row work that is not a quantized projection at all: the recurrent step
loop over `T`, the norms, SwiGLU, the attention row, the cache writes, and the
extra encode and barrier cost of a wider dispatch. **Reading 3 is refuted, and
reading 4 with it** — a missed traffic term would have to appear as unexplained
*extra* round-level cost, and the arithmetic-only reconstruction already
accounts for nearly all of it.

Four calibration classes are extrapolated below the two measured widths
(`N = 5,120`, `14,336`, `16,480` against calibration at `34,816` and
`248,320`), and the two-point split of `a_N` into `p + q_a·N` has no residual
and no error bar. That is why 93.7 % should be read as "the same number", not
as a measured 6.3 % gap.

The reconstruction also re-prices the decision rule on a consistent
multiply-accumulate count. The six classes contain 25.622 G MACs per row, not
the 24.35 G trunk-parameter count in the brief; the difference is the fused
gate and conv columns and the vocabulary readout. On that count the round-level
marginal is **4.991 TFLOP/s = 76.0 % of the measured affine-4 ceiling**, and
the isolated sum is 5.328 TFLOP/s = 81.1 %. Both are above the 70 % threshold,
so the rule fires on either accounting.

**Why I did not spend the two legs.** f2 asks for a normal round and a round
under `MLX_E58_BUFFER_LIMIT_OPS=1` together with
`MLX_E58_BUFFER_LIMIT_MB=1`. That request is conditional on the marginal row's
rate being *far below* peak; rung 0 says it is at 72–76 % of peak, so the
condition does not hold. The measurement is also not available at the stated
cost: neither variable exists in this tree. `MLX_E58_BUFFER_LIMIT_OPS` lives
only in `research/e95-artifacts/e95-census-instrument.patch`, and
`MLX_E58_BUFFER_LIMIT_MB` has not been written yet, so the two legs are
preceded by an instrument change to `device.cpp` plus a full worker rebuild,
and every one of those edits must be unwound before a scope check. The
campaign-wide concurrency-discount question is real and still open; it is a
separate assignment with its own instrument, and it no longer blocks E97.

### 6.2 How 8,112.6 us/round was obtained

It was neither an isolated-buffer measurement nor a GPU-timeline interval. It
is the output of `report_fixed()` in `research/e95_verify_census.py`, run on an
**in-situ** census leg at the default 50 MLX ops per command buffer, through
three stacked modelling steps: (1) least squares fits `a + b·G(M) + c·M` to the
measured per-round verify time over several widths; (2) the modelled variable
cost `b·G + c·M` is allocated to each individual `qmv` dispatch **pro rata by
multiply-accumulate count** and subtracted from the measured GPU interval of
the command buffer that contains it; (3) the remaining per-buffer residual is
distributed over the non-`qmv` shapes in that buffer **pro rata by modelled
bytes**. The residuals sum to `a` by construction, which is an identity and not
a check.

The number is therefore a **residual allocation, not a measurement**, and I
labelled it that way at the time: `research/e95-artifacts/e95-gdn-census-handover.md`
opens with "That number came from `fixed` mode, which is a **model output**, not
a measurement. Do not start from it", lists it under "Model outputs", and its
section 4 records the falsifying evidence — the modelled series *falls* with
width, 9,493.8 us at M=3 down to 6,768.3 us at M=9, slope −428.3 us per row,
while the kernel body loops `for (int t = 0; t < T; ++t)` over exactly those
rows and therefore cannot get cheaper as rows are added.

So the correction rule f2 derives for isolation does **not** apply to this
number, and there is no second unexplained effect. The applicable failure is
the byte-weighted residual allocation in step (3). With a concurrent encoder a
round's wall time contains barrier stalls, encode gaps, submit and drain, and
non-overlapped tails, and none of that belongs to any kernel. Step (3) has no
"belongs to no kernel" bucket, so all of it is forced onto the non-`qmv` shapes
present, in proportion to their bytes. The GDN recurrent state is by a wide
margin the largest non-`qmv` byte mover in those buffers, so it absorbs almost
all of `a` whatever actually caused `a`. Step (2) compounds it: giving the
whole `M`-linear term to `qmv` pulls the genuinely per-row part of the GDN step
out of the fixed term, which is exactly the negative width slope the handover
flagged. The general rule is that any residual-allocation split of a fixed term
is invalid without a no-kernel bucket, and its error is largest for the shape
with the largest allocator weight. Alphonse's repeat arm is the right
instrument, and 854 us/round landing on my own ruling-4 cache-resident rate of
371.1 GB/s is the confirmation.

### 6.3 Housekeeping

Both probe suites already skip instead of failing. Commit `370ddc2` replaced
`try #require(Self.enabled)` with `@Suite(.enabled(if:))` in
`Tests/MLXFastTests/E95QmvWidthProbeTests.swift` and
`Tests/MLXFastTests/E95DonationProbeTests.swift`, so the campaign `swift test`
gate returns to 40 issues across 9 names on a host with the probe variables
unset.

---

## Metric table

The standard score table does not apply: no model, no fixture, no token window
and no score. The measured quantities are these.

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| per-row slope, `O=248320`, G2 (us/row) | — | 468.08 ± 3.96 | — |
| per-row slope, `O=34816`, G2 (us/row) | — | 63.34 ± 1.19 | — |
| **fraction of achievable affine-4 peak, `O=248320` G2** | 0.70 (decision rule) | **0.827** | **+0.127** |
| fraction of achievable affine-4 peak, `O=34816` G2 | 0.70 | 0.857 | +0.157 |
| reduction-only rate as a fraction of peak, `O=34816` | — | 0.932 | — |
| K-independent share of the row at K=5120 | — | 7.0–10.0 % | — |
| `IPG 4` row against `IPG 3` row | 1.000 | 1.198 | +19.8 % |
| group-boundary step, × the M=1 weight read | — | 0.609–0.654 | — |
| tensors admitting a lossless 8-bit pair table | 498 | **0** | −498 |
| isolated per-row sum ÷ round-level marginal | 1.000 | **0.9367** | −6.3 % |
| round-level marginal as a fraction of affine-4 peak | 0.70 | 0.760 | +0.060 |
| session null, worst rung-1 drift | 0 % | −0.23 % | — |

Every compared identity field matched within each session: same host, same
build, same commit, same MLX, same process, same power state, back-to-back
cells, ABBA block order. Across rungs the commit differs (`370ddc2` for rung 0,
`4599edb` for rungs 1 and 2) but no rung-1 or rung-2 number is compared against
a rung-0 *time*; rung 0 contributes only a TFLOP/s ceiling, which is a property
of the machine and the MLX build, both identical.

**Labelled inferences.** The 4.74 TFLOP/s ranked figure and the ranked slope of
7,232.4 us/row come from the advisor's M5 fit and are used only as a
cross-host transfer, marked as such. The `O = 248320` K-fit has two points, so
its `a` and `b` are exact through two points with no residual and no standard
error; the eight-point `O = 34816` fit is the one with an error bar. The
whole-curve `G × bytes / rate` check uses a single rounded 250 GB/s rate and is
a consistency check, not a fit.

---

## Conclusion

- **What happened and why.** The marginal verify row is arithmetic, and the
  arithmetic is already close to this machine's ceiling. At the two scored
  widths the `G = 2` per-row slope runs at 82.7 % and 85.7 % of the affine-4
  arithmetic peak that the same MLX build actually reaches on the same host.
  Strip the K-independent per-row overhead and the reduction-scaled remainder
  runs at 89–93 %. (B) is the answer.
- **Evidence for or against the mechanism.** (A) is refuted by the source, by
  the split-K control whose per-row slope is 1.2 % of the vector slope while
  dequantising just as hard, and by the rate itself, which is a fraction of a
  ceiling that already includes unpacking. (D) is refuted by an activation
  intensity of 8 FLOP per byte on a 10.2 KB vector that never leaves L1. (C)
  survives as the residual and is now sized: a K-independent 7–10 % of the row,
  plus about +20 % for each `IPG` widening.
- **The isolated instrument reconstructs the round.** Extrapolating the rung-2
  fit over the 257 quantized-projection dispatches of one decode round gives
  9,618.4 us per row against the 10,268 us per row that E95 rung 2 and E92
  fitted from round-level times, a ratio of 0.9367. Two instruments that share
  no code, no fixture and no geometry agree to 6.3 %. That also refutes advisor
  reading 3: an isolated time is an upper bound, so an upper bound below the
  round-level marginal leaves no room for a concurrency discount on the
  marginal row.
- **The one thing the row cost does *not* explain**, and the largest single
  term in the vector-regime curve, is the group-boundary step: 0.61–0.65 of a
  full weight read every time `ceil(M/IPG)` increments, replicating E27's
  staircase from an independent instrument. Rung 2 prices the term that stopped
  E33 and E38 from cashing it: the `NA` widening needed to remove a stream
  costs about +20 % of the row per step, which is the same order as the stream
  it removes, and E38's compiled-AIR ladder (+21 registers per NA against a
  128-register wall) is the static mechanism behind that price.
- **Prompt or M5 transfer risk.** High for absolute times, low for the
  conclusion. This is an M4 Pro (`applegpu_g16s`, gen 16, 20 cores); the ranked
  host is M5. The conclusion is a *ratio to that same host's own measured
  ceiling*, which is the transfer-robust form: it says the kernel is near the
  machine's limit, not that the machine is fast. The kernel selection is
  identical on both hosts because `get_qmv_batch_limit` returns 10 at these
  widths for any architecture that is not `'d'`-suffixed and is not generation
  13 or 14, ledger item 121 records that the ranked box resolves to 10, and the
  WIDE table switches on `ntg.x` only.
  The one thing that would not transfer is the *location* of the
  bandwidth-to-compute crossover, which on this host lands at `M ≈ 6`; on a
  part with a different FLOP-to-bandwidth ratio it moves, and with it the size
  of the `G1` band anomaly (the `G1` slope at `O = 248320` implies
  9.30 TFLOP/s, which is 141 % of the machine's arithmetic ceiling and
  therefore proves that band is still hiding rows under the memory stream
  rather than paying for them).
- **Smallest useful next action.** Do not spend another run on the per-row
  axis. If the advisor wants one more measurement here, the cheapest decisive
  one is the `uint16` (scale, bias) index from section 5: it is the only
  mechanism this experiment surfaced with a quantified, lossless, double-digit
  effect on the term that actually dominates the vector-regime curve, and it is
  orthogonal to everything E33 and E38 closed.
- **Recommendation: close.** The assignment asked whether a mechanism worth
  more than +1 % ranked lives on the per-row verify axis. Measured answer: no.
  The axis has at most 14–17 % of the row left, of which at most 10 points are
  launch overhead and 6 points are inner-loop inefficiency, and the campaign
  has already falsified the levers that would reach them.

## Suggested follow-ups, not implemented

1. **`uint16` (scale, bias) index.** Lossless for all 498 tensors, 5.56 % off
   the weight stream, 5.17 MB of tables. Needs
   `Sources/MLXFastTransform/` plus one indirection per group in the kernel.
   Section 4 shows the weight stream is what the vector regime is made of.
2. **More `tid.y`, not more `IPG`.** The only untested direction that reduces
   `G` without widening the accumulator. Rung 2 prices `IPG` widening at
   +20 % per row per step; giving a group more output columns instead should
   not pay that.
3. **Re-price the `G1` band on M5.** Its local slope implies 141 % of the
   arithmetic ceiling, which means `M = 2..4` rows are still free under the
   memory stream on this host. If M5's FLOP-to-bandwidth ratio moves the
   crossover, the ranked `G1` slope should move with it, and the advisor's
   band model would need a refit rather than a transfer.
4. **The stale table comment.** `kernels/quantized.h` case 8 carries a long
   comment arguing for a `3+3+2` split, but the code instantiates
   `qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>`, that is `4+4`. The comment
   and the code disagree. I did not change either — rung 2's brief forbids a
   table change — but a reader could be misled.
5. **The campaign concurrency discount, as its own assignment.** Feedback f2's
   two-leg ratio is still worth measuring for the ledger, but it needs an
   instrument that does not exist yet: `MLX_E58_BUFFER_LIMIT_OPS` lives only in
   `research/e95-artifacts/e95-census-instrument.patch` and
   `MLX_E58_BUFFER_LIMIT_MB` is unwritten. Whoever takes it should write both,
   run a normal round and a one-op-per-buffer round at one fixed width, and
   publish the ratio per kernel family, because f2 predicts the discount is
   largest for families that run furthest below peak.
6. **Whole-buffer false barriers (f2 section 6).** Rung 0 satisfies the
   condition f2 attached to this idea: the verify path is near peak, so the
   remaining round cost is more likely to be serialisation than arithmetic.
   `prev_inputs_` and `prev_outputs_` hold `MTL::Resource*`, so two dispatches
   that touch disjoint slices of one buffer still force a full
   `memoryBarrier(BarrierScopeBuffers)`. `KVCache.swift:398` and `:434` write
   slices into one K buffer and one V buffer across 16 layers. The fix is
   editable Swift, not `device.cpp`. I did not start it; it is the single most
   promising direction this experiment leaves open.
