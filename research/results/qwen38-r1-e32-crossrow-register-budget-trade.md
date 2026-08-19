# E32: the width lever is not closed, but `rows_per_simd` is not the axis that opens it

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"max_spill_free_IPG_under_frozen_grid","available":true,"value":9},"test_metric":{"name":"gate_control_cells_matching_expected_verdict","available":true,"value":1}}
```

- Student / branch: `qwen-askeladd` / `qwen-askeladd/crossrow-register-budget-trade`
- Hypothesis and target cost: spending `rows_per_simd` buys back the registers that
  raising `NA` costs, so some `NA >= 6` cell is spill-free. Target cost is the
  second weight pass at `M = 6..9` (E20: MLP is 59 % of verify time and 65.1 % of
  byte traffic).
- Decision: **green gate, with the mechanism relocated.** `NA >= 6` is reachable
  spill-free, but *not* by changing `rows_per_simd`, and the advisor's register
  model is falsified.
- `BASE_SHA` `d2fcebb0796926962016b87060f9580b9bca89d4` / `UPSTREAM_SHA` unchanged /
  candidate commit: this branch (research-only).
- Submitted candidate files: **none.** Zero shipped-surface bytes changed.
- Supporting files: `research/crossrow_rps_gen.py`,
  `research/generated/crossrow_rps_wide.h`, `research/crossrow_rps_probe.metal`,
  `research/crossrow_rps_sweep.py`, `research/e32_analysis.py`,
  `research/e32-rps-grid.json`, `research/e32-rps-analysis.txt`,
  `research/e32-occupancy.txt`, `research/e32_wandb_log.py`.
- Assignment-scope preflight: `git diff --stat d2fcebb -- .` lists 10 files (the
  9 above plus this report), all under `research/`. No `quantized.h`, no
  `mlx-generated/`, no `Sources/`, and the shipped
  `static_assert(NA >= 2 && NA <= 5, ...)` is untouched.
- Scored-path reachability: the sweep's anchor arm calls the **shipped**
  `qmv_fast_crossrow_affine4_g64_wide<T, NA, true>` exactly as `quantized.h:1177`
  calls it, and reproduces E27's ladder digit-for-digit.

## Evidence

- Host: Apple M4 Pro, `metal 32023.883`, `air64-apple-darwin25.5.0`.
  **Zero GPU timing. No benchmark, no timing lock, no seconds-per-token.**
- W&B: [`qsky4vvs`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/qsky4vvs)
  — carries the full 77-cell grid, the gate-control verdicts, and the decision
  table as logged tables.
- Commands:
  ```bash
  python3 research/crossrow_rps_gen.py            # derive the probe body
  python3 research/crossrow_rps_gen.py --check    # content assertion vs quantized.h
  python3 research/crossrow_rps_sweep.py --out research/e32-rps-grid.json \
      --jobs 4 --keep-air /tmp/e32_air
  python3 research/e32_analysis.py research/e32-rps-grid.json

  # pipeline reflection for (b), over the same 77 AIR objects
  xcrun metallib /tmp/e32_air/*.air -o /tmp/e32_air/probe.metallib
  swiftc -O research/crossrow_na_occupancy.swift -o /tmp/e32_air/na_occupancy
  /tmp/e32_air/na_occupancy /tmp/e32_air/probe.metallib   # -> research/e32-occupancy.txt
  ```
  Per cell: `metal -std=metal3.1 -O2 -S` then
  `metal-opt -passes=default<O3>`, which is E27's pipeline. Dropping the
  `metal-opt` stage changes the numbers, so it is not optional for comparability.
- Reproduction: the sweep was re-run from scratch after the fact and its 77
  cells match the committed `e32-rps-grid.json` field for field (`regs`,
  `allocas`, `acc_spill`, `status`), with `gate validation: 10/10` again and an
  empty failure list. `e32_analysis.py` over the committed grid reproduces
  `e32-rps-analysis.txt` byte for byte, and `crossrow_rps_gen.py --check` still
  passes against `quantized.h`. The pipeline reflection was likewise re-run from
  fresh AIR and is now committed as `research/e32-occupancy.txt` rather than
  quoted from a scratch directory; `e32_wandb_log.py` now *parses* the two
  reflection scalars out of that file instead of carrying them as literals, and
  it reads back the same `1024` / `0` the W&B run already carries.
  `senpai/check-editable-budget.sh` reports `growth=0/262144` — zero
  shipped-surface bytes.

### How the probe stays honest

`rows_per_simd` is a `constexpr` in the body, so sweeping it needs a second copy
of that body — and a hand-copied body stops measuring the shipped kernel the
moment either file moves. `crossrow_rps_gen.py` therefore *extracts* the body
from `quantized.h` (lines 968–1066) and applies **exactly four** substitutions,
each of which must match exactly once or the generator aborts:

1. add the `ROWS_PER_SIMD` template parameter,
2. rename the symbol so it cannot shadow the shipped one,
3. widen the `NA` bound **inside the probe only** (`<= 16`),
4. `constexpr int rows_per_simd = ROWS_PER_SIMD;`.

Nothing else differs. `--check` re-derives and fails on drift. Per the carried-
forward warning this is a content assertion over the extracted template text,
not a hash of `quantized.h`.

**Independent confirmation that this worked:** the generated body at `r = 4`
produces AIR statistics *identical* to the shipped template — 62 / 83 / 104 / 125
registers at `NA = 2/3/4/5`, one `[4 x [4 x i16]]` alloca each. Those are also
E27's numbers to the digit, measured on a different day at a different base.

### Gate validation on known-GOOD and known-BAD

The spill gate matches allocas by **accumulator type** `[r x <NA x float>]`, not
by `alloca > 0` (every cell has a `packed` alloca; that is not a spill).

**My first negative control was broken and I am reporting it rather than
quietly fixing it.** It allocated `float ballast[NA*R]`, produced
`[16 x float]`, and the gate said "no spill" — correctly, because that is not
the accumulator type. The control exercised nothing. It is now a
`vec<float,NA> acc[R]` array indexed by runtime data, which is the real
signature, and the sweep exits non-zero if any control disagrees with its
expected verdict.

| control | kind | expected | observed |
|---|---|---|---|
| `xctl_spill_na4_r4` | forced runtime-indexed accumulator | SPILL | SPILL |
| `xctl_spill_na9_r2` | forced runtime-indexed accumulator | SPILL | SPILL |
| `xctl_spill_na16_r4` | forced runtime-indexed accumulator | SPILL | SPILL |
| `xctl_e27_spill_na6_r4` | E27's historically measured spill cell | SPILL | SPILL |
| `xctl_clean_na4_r4`, `xctl_clean_na5_r4` | shipped-equivalent cells | clean | clean |
| `xship_na2..na5` | the shipped template itself | clean | clean |

`gate validation: 10/10 control cells matched their expected verdict`.

## (a) The grid

`peak_live_regs / allocas`, `SPILL` = an accumulator-typed alloca survived `-O2`
plus `default<O3>`. Threadgroup memory is **0 bytes in every cell** (no
`addrspace(3)` reference anywhere in the grid, confirmed independently by
`staticThreadgroupMemoryLength == 0` from pipeline reflection —
`research/e32-occupancy.txt`, 77/77 cells).

| NA | r=1 | r=2 | r=3 | r=4 |
|---|---|---|---|---|
| 2 | 38/1 | 46/1 | 54/1 | 62/1 |
| 3 | 49/1 | 61/1 | 72/1 | 83/1 |
| 4 | 60/1 | 76/1 | 90/1 | 104/1 |
| 5 | 71/1 | 91/1 | 108/1 | 125/1 |
| 6 | 82/1 | 106/1 | 126/1 | **144/2 SPILL** |
| 7 | 93/1 | 121/1 | 144/1 | **157/2 SPILL** |
| 8 | 104/1 | 136/1 | **168/2 SPILL** | **177/2 SPILL** |
| 9 | 115/1 | **151/1** | **187/2 SPILL** | **197/2 SPILL** |
| 10 | 126/1 | 166/1 | **195/2 SPILL** | **217/2 SPILL** |
| 11 | 137/1 | 181/1 | – | – |
| 12 | 156/0 | 196/1 | – | – |

Spill-free ceiling: **r=4 → NA ≤ 5; r=3 → NA ≤ 7; r=2 → NA ≤ 12 (no spill found);
r=1 → NA ≤ 12 (no spill found).** `NA = 11, 12` are model-probe cells only; no
legal `M` can use them.

## (b) The occupancy consequence — and why it is the whole answer

`num_simdgroups = 2` is **verified**, not inferred, and from the host rather than
from `qmv_impl`: `backend/metal/quantized.cpp:251-254` sets `bn = 8`,
`group_dims(bk=32, 2, 1)` — 64 threads = 2 simdgroups — and
`grid_dims(M, (N + bn - 1)/bn, B)`. The kernel's
`out_row = tid.y * 8 + simd_gid * 4` agrees.

**That file is not in `benchmark.json /editablePaths`.** I re-verified this
directly rather than inheriting it. So the grid is frozen at 2176 threadgroups
for the MLP's `N = 17408`, each covering 8 rows:

| r | rows/threadgroup | threadgroups needed | frozen grid launches | legal? |
|---|---|---|---|---|
| 1 | 2 | 8704 | 2176 | **NO** — computes 4352 of 17408 rows |
| 2 | 4 | 4352 | 2176 | **NO** — computes 8704 of 17408 rows |
| 3 | 6 | 2901.33 | 2176 | **NO** — and 17408 is not divisible by 6 |
| 4 | 8 | 2176 | 2176 | yes |

Lowering `rows_per_simd` in the plain sense does not cost occupancy. It produces
**wrong answers**: the rows nobody covers are never written. This is a
correctness wall, not a performance trade, and it applies to every `NA >= 6`
cell in the r ≤ 3 columns above.

**On theoretical occupancy I have to report a tool failure rather than a
number.** `maxTotalThreadsPerThreadgroup` is **1024 for all 77 cells**
(`research/e32-occupancy.txt`, from E27's own
`crossrow_na_occupancy.swift` run unmodified over this grid's metallib;
`threadExecutionWidth` is 32 everywhere too),
including the deliberately spilled controls with 64-float private arrays. It
does not discriminate register pressure here, and the reason is structural: the
dispatch uses 64 threads per threadgroup, so a 1024-thread ceiling was never
going to bind. Residency per shader core is the quantity that matters and Metal
does not expose it. E27's comment calling this "the cheapest direct readout of
the register cliff" does not hold for this dispatch shape. I did not manufacture
an occupancy estimate to fill the gap.

### The form that *is* legal

Cover the same 4 rows per simdgroup as `4/r` sequential row blocks. `out_row`
stays `tid.y * 8 + simd_gid * 4`, the grid is untouched, every row is written,
and each row's weights are still read exactly once.

| NA | r | regs/allocas | spill | ALU/tile |
|---|---|---|---|---|
| 6 | 2 | 117/1 | clean | 1064 |
| 7 | 2 | 134/1 | clean | 1220 |
| 8 | 2 | 151/1 | clean | 1376 |
| 9 | 2 | **168/1** | **clean** | 1532 |

`xrb_na9_r2` has `loop_backedges = 3` against `xrps_na9_r2`'s 2 — exactly one
extra **rolled** loop. The compiler did not unroll the two blocks back into a
4-row accumulator (which would have re-spilled) and did not eliminate one, and
the alloca list is `[2 x [4 x i16]]` only.

Bit-exactness is preserved by construction: rows are independent, each row's
k-loop accumulation order is untouched, and only the order in which one
simdgroup visits its own 4 rows changes. This is the same argument that
`crossrow-closure.md` §3.5 already accepted for row/lane reassignment.

## (c) THE DECISION TABLE

ALU/tile is per lane per k-block for one 8-row tile, from the op accounting in
`crossrow-closure.md` §3 — **analysis, not measurement**. Only the x-side term
repeats per row block.

| M | shipped IPG/passes | best legal (NA=IPG, r) | `M%IPG != 1` | passes | spill | ALU/tile vs shipped |
|---|---|---|---|---|---|---|
| 3 | 3 / 1 | NA=3, r=4 (**keep shipped**) | true | 1 | clean | 500 vs 500 (+0.0 %) |
| 4 | 4 / 1 | NA=4, r=4 (**keep shipped**) | true | 1 | clean | 624 vs 624 (+0.0 %) |
| 5 | 5 / 1 | NA=5, r=4 (**keep shipped**) | true | 1 | clean | 748 vs 748 (+0.0 %) |
| 6 | 3 / 2 | **NA=6, r=2 row-blocked** | true | **1** | clean | 1064 vs 1000 (+6.4 % ALU, **−50 % weight passes**) |
| 7 | 4 / 2 | **NA=7, r=2 row-blocked** | true | **1** | clean | 1220 vs 1124 (+8.5 % ALU, **−50 % weight passes**) |
| 8 | 4 / 2 | **NA=8, r=2 row-blocked** | true | **1** | clean | 1376 vs 1248 (+10.3 % ALU, **−50 % weight passes**) |
| 9 | 5 / 2 | **NA=9, r=2 row-blocked** | true | **1** | clean | 1532 vs 1372 (+11.7 % ALU, **−50 % weight passes**) |

Every `M` in 3..9 reaches a single weight pass. On the depth histogram quoted in
the assignment that is **67 of 78 rounds (86 %)**. The trade is roughly +6 % to
+12 % ALU against halved weight-stream DRAM traffic, in a kernel E15 measured as
bandwidth-bound (`C(2)/C(1) = 0.9982`).

**Provenance caveat on the 86 %.** The histogram `{1:1, 3:5, 4:5, 5:23, 6:4,
7:6, 8:34}` is not recorded verbatim anywhere in the repository. Ledger item 73
records only the two endpoints (23/78 at M=6, 34/78 at M=9), which are
consistent with it. Anyone acting on the 86 % should re-derive the histogram
from the current dispatch table first; the *table above* does not depend on it.

## (d) The falsification I was asked to attempt

**Why did the shipped kernel choose `NA <= 5` with `rows_per_simd = 4`? Because
it never chose `rows_per_simd` at all.** `rows_per_simd = 4` is forced by
`bn = 8` and 2 simdgroups in the non-editable host dispatch. Given `r = 4`,
`NA = 6` spills — E27 measured it, and this grid reproduces it. So `NA <= 5` is
*exactly* the spill boundary at the only `r` the contract allows. The shipped
constants are the unique legal optimum of the axis the kernel was allowed to
tune, and there is no unexplained free lunch to explain away.

I looked for the other candidate reasons the advisor listed and they are real
but secondary: the host's fast-path predicate `N % bn == 0` is also tied to
`bn = 8`, so a 6-row tile would fail to tile `N = 17408` at all.
`bytes_per_lane = 8` and `values_per_thread = 16` are per-row-per-lane and do
**not** constrain the row count. No simdgroup-reduction assumption blocks it:
`simd_sum` is applied per `(r, m)` independently.

What *was* wrong is narrower and worth stating plainly: `crossrow-closure.md`
§3.5 recorded the geometry constraint as blocking `rows_per_simd = 8` (raising
it). It is symmetric — it blocks lowering it too — and nobody had written that
down, which is exactly why the width lever looked closed.

## Scoring the pre-registered predictions

1. **NA=6 at r=2 is spill-free — CORRECT.** 106 registers, 1 alloca.
2. **NA=9 at r=2 is spill-free or marginal — CORRECT, and not marginal.** 151
   registers, 1 alloca, with at least three further NA steps of headroom
   (NA=12/r=2 is still clean). This is the cell carrying the 86 %.
3. **Registers approximately affine in the product `rows_per_simd × NA` —
   FALSIFIED.** The advisor asked for this result over the lever, so here it is
   in detail.

   The product fit is `regs = 35.5 + 5.96·(r·NA)` with **max residual 49
   registers** — a third of the value at the cells that matter. It predicts
   `regs(9, 2) ≈ 143` (the advisor predicted ~125); the measurement is **151**.
   The decisive counterexample is that the product does not even determine the
   *verdict*: `NA=6, r=4` and `NA=12, r=2` are both 24 accumulator floats, and
   the first spills while the second is clean.

   The correct model is affine in `NA` at fixed `r`, with an `r`-dependent slope:

   | r | fit | max residual |
   |---|---|---|
   | 1 | `regs = 14.2 + 11.36·NA` | 5.45 |
   | 2 | `regs = 16.0 + 15.00·NA` | 0.00 |
   | 3 | `regs = 18.0 + 18.00·NA` | 0.00 |
   | 4 | `regs = 20.0 + 21.00·NA` | 0.00 |

   and `slope(r) = 8.36 + 3.19·r`, max residual 0.25. The split is mechanistic,
   and it matches this campaign's own op accounting: **8.36 registers per NA are
   independent of `r`** (`a0..a3` and `sums`, 5 live floats per NA) and only
   **3.19 per NA scale with `r`** (`acc` and `partial`, 2 floats per NA per row).
   The measured ratio 2.62 against the predicted 5:2 = 2.50.

   So halving `rows_per_simd` buys back only the `acc`/`partial` half of the
   cost and none of the x-side half. That is precisely why the r=1 column is
   *not* twice as good as r=2, and why r=2 row-blocking — not r=1 — is the right
   operating point. (The single r=1 outlier, `NA=12` at 156 registers with
   `allocas = 0`, is the one cell where the compiler also promoted the `packed`
   array; it is the entire source of that column's 5.45 residual.)

## Conclusion

- **What happened:** the width lever is open, but the advisor's axis is closed.
  `rows_per_simd` cannot be lowered — the frozen host grid makes any `r < 4` a
  correctness failure, not a trade. The same register relief is available for
  free by covering the tile in `4/r` sequential row blocks; the row-blocked form
  at `r = 2` is clean through `NA = 10` (the whole legal `M` range), and the
  single-block body at `r = 2` is clean through `NA = 12`.
- **Evidence for the mechanism:** anchors reproduce E27 exactly; 10/10 gate
  controls behave; the row-blocked form keeps one extra rolled loop and no
  accumulator alloca.
- **Transfer risk:** this is an AIR-level signal measured on M4 Pro, and final
  register allocation happens in the M5 driver back end. E27's ladder was
  measured the same way and its NA=5 promotion did transfer, which is the best
  available evidence that this gate transfers — but it is not proof.
- **Smallest useful next action:** hand `NA=9 / r=2 row-blocked` to whoever owns
  the dispatch table, as a *single* `M = 9` arm first. It is the largest single
  histogram bucket (34/78) and the highest-ALU cell, so it is simultaneously the
  best case for the win and the worst case for the ALU cost. If M=9 wins, M=6..8
  follow at strictly better ALU ratios.
- **Recommendation:** promote the gate result; assign the productionization and
  its timing to one student. I did not touch the dispatch table, per the
  assignment.

## Suggested follow-ups I did not implement

- The `r = 2` column never spilled inside the tested range, so the real ceiling
  is unknown. If anything ever wants `IPG > 12` it needs a fresh probe, not an
  extrapolation of my fit.
- `NA = 7` at `r = 3` is spill-free at 144 registers, the same register count as
  the *spilling* `NA = 6, r = 4`. If a future contract change ever unfreezes
  `bn`, that cell is worth revisiting; today it cannot tile `N = 17408`.
- `crossrow-closure.md` §3.5 should record that the geometry constraint is
  symmetric. As written it reads as blocking only `rows_per_simd = 8`.
