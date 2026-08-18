# E22 pre-registration — narrow-width dispatch cost curve + 3-bit crossrow adjudication

Committed and pushed **before** the first timed run of this experiment, per the
assignment's pre-registration requirement.

- Assignment: `qwen38-r1-e22-narrow-width-dispatch-cost-curve` (revision `r1`)
- Branch: `qwen-thorfinn/narrow-width-dispatch-cost-curve`
- Base: `senpai/qwen38-mtp-r1` @ `c0f7e370921a14f348fa1872f2176b1b43028752`
- Assignment head at pre-registration: `ec36259030c2a10ff461e49ebc5c09c5850fcd7a`
- Host: Apple M4 Pro (**not** the ranked M5 `m5-qwen38-27b-mtp`). All numbers here
  are directional micro-benchmark evidence about a dispatch decision, not a
  ranked-score claim.

---

## 0. Structural correction to the assignment premise

The assignment asks whether the M=2 3-bit penalty is fixed by "instantiating a
crossrow `affine3` kernel". Reading the live source at HEAD `ec36259`, **there is
no crossrow kernel symbol to instantiate**, so the experiment has to be
restated before it can be run:

- The dispatcher builds the kernel name in
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:261-269`
  (inside `qmv`, entered from `dispatch_qmv` at `:1371`). The name only encodes
  `gs`, `b`, and `batch`. `affine_qmv_fast_bfloat16_t_gs_64_b_3_batch_0`
  **already exists and is already dispatched today** for the 3-bit arms.
- "Crossrow" is not a separate kernel. It is an **in-kernel `switch (ntg.x)`**
  inside `affine_qmv_fast`
  (`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:1783`),
  guarded at `:1822` by
  `if (!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024)`.
  The 3-bit instantiation therefore compiles, enters `affine_qmv_fast`, fails
  that `bits == 4` clause, and falls through to `qmv_fast_impl` at `:1959`.
- Because `qmv` launches `grid_dims(M, ceil(N/8), B)` at `:254`, `ntg.x == M`
  exactly, so the switch is literally a switch on the row count M.

So the real fork is: **add a 3-bit device-helper family + widen the `bits == 4`
clause at `quantized.h:1822`**, not "instantiate a missing symbol". The
falsifiable question the assignment actually poses survives this correction
unchanged, and that is what is pre-registered below.

Related bookkeeping: `Tests/MLXFastTests/QwenQMVCostCurveTests.swift:92-101`
cites the crossrow span as `quantized.h:1804-1908`. The live span at `ec36259`
is `1822-1955`. That stale citation will be corrected as part of this work.

## 1. Hypotheses

- **H_MISSING (missing-kernel):** the M=2 3-bit penalty exists only because no
  crossrow path is compiled for `bits == 3`. Supplying one recovers most of the
  gap, because the penalty is entirely "two weight passes instead of one".
- **H_FUNDAMENTAL (narrow-width):** the penalty is intrinsic to 3-bit narrow-M
  work — unaligned 3-bit lane geometry, extra unpack ALU, register pressure, or
  reduced occupancy — so a crossrow `affine3` recovers little and the only real
  fix is gating bit width on M.

## 2. Predicted outcome

**I predict H_MISSING.**

Reasoning, from askeladd's E15 numbers on the scored draft-readout shape
(`k=5120`, `n=98336`, affine group-64, bf16 scale+bias, so bytes/value =
`bits/8 + 0.0625`, total values `503,480,320`):

| arm | s/call | weight bytes touched | effective GB/s |
|---|---|---|---|
| M=1 b=2 | 0.0006700 | 157,337,600 | 234.8 |
| M=1 b=3 | 0.0008827 | 220,272,640 | 249.5 |
| M=1 b=4 | 0.0011664 | 283,207,680 | 242.8 |
| M=2 b=4 (crossrow) | 0.0011653 | 283,207,680 (1 pass) | 243.0 |
| M=2 b=3 (`qmv_fast_impl`) | 0.0015329 | 2 passes = 440,545,280 | 287.4 |

Three things follow:

1. Every M=1 arm lands on the **same ~235–250 GB/s roofline regardless of bit
   width**. If 3-bit unpack were ALU-limited, the 3-bit arm would fall *below*
   that roofline; it does not (it is in fact the highest M=1 point). So the
   3-bit unpack has spare ALU headroom on this host.
2. M=2 b=4 crossrow costs 0.9991× the M=1 b=4 call: weight-pass amortization at
   4 bits is essentially **perfect**, i.e. the second row is nearly free once
   the weight bytes are already in registers.
3. M=2 b=3 moves 287 GB/s of *two-pass* traffic against a ~250 GB/s roofline,
   i.e. only ~15% of the second pass is being served from cache. The 220 MB
   working set is far larger than L2, so L2 is not amortizing the second pass.
   The gap is real DRAM traffic, and single-passing it is exactly what a
   crossrow kernel does.

If the mechanism is "one weight pass instead of two", then the 3-bit crossrow
should approach the M=1 b=3 cost the same way the 4-bit crossrow approaches the
M=1 b=4 cost.

**Predicted M=2 b=3 crossrow seconds/call: 0.00095 s** (plausible range
0.00090–0.00110 s), against 0.0015329 s for today's `qmv_fast_impl` fallback and
an ideal single-pass floor of `220,272,640 / 249.5e9 = 0.000883 s`.

That prediction is deliberately a little above the ideal floor: my adjudication
kernel mirrors the 4-bit **pair** helper (`inputs_per_group = 2`) rather than
the tuned wide/`_m` variants, and 3-bit lane geometry costs 6 bytes/lane instead
of a clean 8, so I expect to leave a few percent on the table.

## 3. Decision rule (fixed in advance)

Let `t_x` be the measured M=2 bits=3 seconds/call with the crossrow path live,
measured on this host under the same harness as the baseline arms.

Absolute rule:

- `t_x <= 0.00120` → **H_MISSING supported**.
- `t_x >= 0.00145` → **H_FUNDAMENTAL supported**.
- `0.00120 < t_x < 0.00145` → **partial / ALU-limited**; report as such, do not
  round it to either hypothesis.

Because absolute thresholds are host- and thermal-sensitive, the *primary*
adjudication is the normalized recovery fraction, computed from arms measured in
this experiment rather than from askeladd's table:

```
R = (t_impl - t_crossrow) / (t_impl - t_M1b3)
```

where `t_impl` is this experiment's own measured M=2 bits=3 fallback cost and
`t_M1b3` is this experiment's own measured M=1 bits=3 cost.

- `R >= 0.7` → **H_MISSING supported**.
- `R <= 0.2` → **H_FUNDAMENTAL supported**.
- otherwise → **partial**.

Using askeladd's numbers as the reference point, my predicted `t_x = 0.00095`
corresponds to `R = (0.0015329 - 0.00095) / (0.0015329 - 0.000883) = 0.897`.

Hypotheses are considered "separated" only when the distance between `t_x` and
the nearer threshold exceeds the measured cell-to-cell spread (max−min across
the r1/r2 repeats of the arms involved).

## 4. Design to be measured

Adjudication kernel, added to **both** twins
(`kernels/quantized.h` and `Cmlx/mlx-generated/quantized.cpp`, constant +13 line
offset):

1. `qdot_affine3_loaded<U, values_per_thread>` — thread-address-space twin of the
   `bits == 3` arm of `qdot` (`quantized.h:~211-228`), reproducing its expression
   sequence byte-for-byte, **including** the cumulative `x_thread += 8 * i;
   w += 3 * i;` pointer arithmetic inside the loop. That arithmetic is
   accidentally correct only for `values_per_thread <= 16`; any deviation is a
   silent numerical change, so it is replicated verbatim rather than "fixed".
2. `qdot_affine3_loaded_pair` — `float2` analogue mirroring
   `qdot_affine4_loaded_pair` (`:841`): the same 10 sequential `accum +=` steps,
   masks `0x07,0x38,0xc0 | 0x01,0x0e,0x70,0x80 | 0x03,0x1c,0xe0`, including the
   two `* 256.0f` straddle terms.
3. `template <typename T, int M> qmv_fast_crossrow_affine3_g64` — structural
   mirror of the 4-bit pair kernel at `:860` (`inputs_per_group = 2`,
   `rows_per_simd = 4`, `values_per_thread = 16`, `block_size = 512`, the
   `has_pair` two-path structure, `first_m = tid.x * 2`, early return when
   `first_m >= M`, `out_row = tid.y * 8 + simd_gid * 4`) with 3-bit geometry:
   `bytes_per_lane = 16 * 3 / 8 = 6`, `in_vec_size_w = in_vec_size * 3 / 8`,
   weight byte offset `k * 3 / 8` (integral because `block_size == 512`), and
   scale/bias index `row * in_vec_size_g + k / 64 + simd_lid / 4`.
   `IPG = 2` covers M=2 (2), M=3 (2+1), M=4 (2+2), M=5 (2+2+1), so NA=1 tails
   degenerate to the single path and the `M % IPG != 1` restriction never binds.
   This is an **adjudication** kernel, not a tuned one.
4. Widen the gate at `quantized.h:1822` with a sibling `bits == 3` branch
   (switch cases 2..5). The existing `bits == 4` branch is left byte-for-byte
   unchanged.

## 5. Parity gate, and its power

The gate is mandatory because this touches the exactness-critical readout path.

The existing parity suite `QwenQMVParityTests`
(`Tests/MLXFastTests/QwenQMVCostCurveTests.swift:230-276`) sweeps
`widths = 1...12` × 8 `scoredShapes` = 96 cells, but **hardcodes `bits: 4` at
`:250`**. It therefore has **zero power** over a `bits == 3` change: not one of
its 96 cells would execute a single line of the new code.

So the gate is extended to 3-bit cells using `lowBitQuantWeight` (`:288-309`,
which already handles non-power-of-two bit widths), and the covering cell count
is reported explicitly:

- cells that execute the new `affine3` crossrow path **at M=2**: 8 scored shapes
  × 1 width = **8**;
- cells over the full new switch range M ∈ {2,3,4,5}: 8 × 4 = **32**;
- M=1 and M ≥ 10 3-bit cells are controls that must remain identical.

A gate result is only reported as meaningful together with that count.

## 6. Stopping rule

- Stop as soon as the hypotheses separate by more than the measured cell-to-cell
  spread.
- Hard stop at **4 GPU-hours** of timed work regardless of separation.
- **If the `affine3` path fails parity, STOP and report the failure.** Exactness
  is not to be repaired under time pressure; a diverging kernel is a reportable
  result, not a bug to patch out before the deadline.

## 7. Scope discipline

This experiment **does not ship a bit-width policy change**. If the answer is
H_FUNDAMENTAL, the correct follow-up (gate bit width on M) is *proposed* in the
report, not implemented here. If the answer is H_MISSING, the crossrow `affine3`
is reported as an adjudication result with its parity evidence; whether to
promote it is the advisor's call on a later assignment.

## 8. Measurement protocol

- Arm **BASE** (unmodified tree): M ∈ {1,2,3,4,5} × bits ∈ {3,4}, plus the
  M=1 bits=2 continuity cell that cross-checks askeladd's E15 table.
- Arm **A3X** (3-bit crossrow live): the same cells, so bits=4 doubles as a
  no-regression control and bits=3 is the adjudication.
- Every cell repeated at least twice in separate processes; both values are
  reported as r1/r2, never averaged away.
- Per arm: entry/exit GPU temperature, host, toolchain, `head_provenance_sha256`,
  dirty flag, selected kernel name and crossrow boolean per cell, and the
  dispatch decision cited to file:line at live HEAD.
- The crossrow boolean is to be derived from the **real** predicate including the
  new `bits == 3` path. Today the harness infers it arithmetically at `:175`
  (`spec.bits == 4 && (2...9).contains(spec.m)`) and hardcodes the kernel name at
  `:182`; that inference would silently mislabel every new cell, so it is
  corrected.
