# E41 — Resolving E38's R2 into activation re-read (MEM) vs register tile (ILP)

PR #46 · branch `qwen-thorfinn/r2-confound-before-ktiling` · base
`senpai/qwen38-mtp-r1` @ `04ad6bf11437c269df85a47e91faa769c74fe6da`

<!-- VERDICT AND NUMBERS ARE FILLED IN AFTER THE ARM CURVE LANDS. -->

## Question

E38 measured a **+10.54 % kernel cost** at M=6 for halving the row tile of the
wide crossrow affine-4 g64 QMV kernel (its "arm (a)"). That number is the price
any future NA=6 single-weight-pass scheme must pay, so it gates deliverable (b),
K-tiled activation staging. But E38's arm (a) changed two things at once:

1. **MEM** — each activation element `x[k]` is now read by two row blocks
   instead of one, and the two reads are separated by a full pass over K.
2. **ILP / register tile** — the register tile halved (4 rows/SIMD → 2), the
   accumulator count halved, and one extra sequential loop appeared.

K-tiled staging can only recover cause 1. If R2 is actually cause 2, building
(b) is wasted work. E41 attributes R2 before any staging is written.

## Why the advisor's suggested arm (a) would have answered the wrong question

The assignment sketched an **unrolled** arm: emit the two row blocks' loads
adjacently in one loop body and see whether the tax disappears. That arm is
unsound here, and it fails in the direction that *falsely kills* K-tiling.

Adjacent row blocks read **identical device addresses** for `x[k]` — the
activation is shared across output rows; only the weight stream differs. With
no intervening store between them, **common-subexpression elimination is legal**,
and the compiler is entitled to delete the second load outright. If the tax then
vanished, the recovery could be attributed either to instruction-level
parallelism or to a deleted load, and those are exactly the two hypotheses under
test. Worse, a CSE'd arm makes the MEM cost unobservable by construction, so a
real re-read cost would read as "no MEM component" → ILP → K-tiling dead.

## Registered alternative: a K-tile distance ladder

Instead of removing a load, hold the **instruction mix, trip counts and
accumulator count fixed** and vary **only the reuse distance** of `x[k]`:

```
for kt in K-tiles:            # k_tile = KT * 512 values
  for b in row blocks:        # BLOCKS_PER_CALL
    for k in this tile:       # the verbatim E38 k-block body
```

Both row blocks stay live, both loads are still issued, and the loop nest is the
same shape at every rung. `KT` alone moves the second block's read of `x[k]`
from "one full pass over K later" (`KT=64`) to "512 values later" (`KT=1`).

| rung | reuse distance | role |
|---|---|---|
| `KT=64` | spans K in one tile (32 768 ≥ widest scored K = 17 408) | no locality |
| `KT=4` | 2 048 values | **discriminator** |
| `KT=1` | 512 values | total-recovery bound |

`KT=1` is *not* the discriminator: both hypotheses predict some recovery there
(it is also the shortest loop). The single-mechanism signal is `KT=64 → KT=4`.

## Pre-registration and the compile-only gate

`research/e41_prereg.py` was committed (`1e8bc95`) **before any kernel compile**
and carries the advisor-corrected constants verbatim: score sensitivity 1.00,
crown 0.5193 %, engineerable gap 0.2586 %, σ_score 0.0978 %, control band
±0.46 %, E38 arm (a) M=6 ratio 1.1054, and ψ·φ = 0.0459 flagged
**BACK-SOLVED, not measured**.

Before spending GPU time, `research/e41_ktile_census.py` compiled every ladder
cell to AIR and checked the design's core claim. Results in
`research/e41-ktile-census.json` (**CENSUS OK**):

- **Non-perturbation.** The shipped kernels are untouched by the refactor:
  `xship_na2/3/4/5` = **62 / 83 / 104 / 125** registers with **32 / 36 / 40 / 44**
  device loads — identical to the E32/E36 census.
- **E32's row-blocked grid reproduces exactly** from the new template
  (`xrb_na3_r1/r2/r4` = 51/66/83; `xrb_na4_*` = 63/83/104; `xrb_na6_*` =
  87/117/144), so the template is a faithful generalisation, not a new kernel.
- **The ladder is one mechanism.** At every (NA, r), `KT ∈ {1, 2, 4, 64}` gives
  **identical** `peak_live_regs`, `device_loads`, `vector_float_ops`,
  `loop_backedges` (=4) and `allocas` (=2). This is the census evidence that the
  `KT=64 → KT=4` step changes reuse distance and nothing else.
- **No timed cell spills.** `xkt_na3_r2` = 77 regs, `xkt_na4_r2` = 97,
  `xkt_na6_r1` = **105** (fits under the 128 wall), `xkt_na6_r2` = 135 (over —
  excluded from timing). Spill-gate controls fired correctly.

`xkt_na6_r1` fitting at 105 registers is the concrete shape deliverable (b)
would use, and it needs **no threadgroup memory and no barrier**.

Registered-vs-measured register predictions were scored honestly: **4 of 7** in
band. Misses: `xkt_na4_*` 97 vs predicted 88–94 (by 3), `xkt_na6_r2` 135 vs
126–132 (by 3), `xkt_na6_r1` 105 vs 116–122 (by 11, in the favourable
direction).

### Two amendments recorded before the GPU ran

1. **"KT = all K" is `KT=64`, not `KT=0`.** `KT=0` sets `k_tile = in_vec_size`,
   which the compiler can prove is trip count 1, so it deletes the loop
   (`loop_backedges` 4 → 3). That would have confounded the top rung with a loop
   removal. `KT=64` spans K in one tile but keeps the loop.
2. **Discriminator M=7 → M=8, bound M=8 → M=7.** M=7 dispatches IPG=4 with a
   TAIL=3 group, mixing NA=4 and NA=3 work. M=4 and M=8 are both **pure NA=4**,
   so `M=4 → M=8` is a one-constant step. M=7 is retained only as the adjacency
   bound, where impurity is tolerable.

## Method

One base build and one arm build, same host, same session, same
`--reps 21 --inner 10` shapes-only configuration; the ratio is taken per M
between the two builds. The arm build carries the whole ladder at once so that
widths **1, 2, 5 and 9 are byte-for-byte the base** and serve as untreated
controls (must land within ±0.46 %).

| M | base | arm | rung |
|---|---|---|---|
| 3 | `<T,3,3,true>` | `<T,3,3,true,2,2,1>` | NA=3, KT=1 |
| 4 | `<T,4,4,true>` | `<T,4,4,true,2,2,64>` | NA=4, no locality |
| 6 | `<T,6,3,true>` | `<T,6,3,true,2,1>` | E38 arm (a) tax anchor |
| 7 | `<T,7,4,true>` | `<T,7,4,true,2,2,1>` | NA=4, adjacency bound |
| 8 | `<T,8,4,true>` | `<T,8,4,true,2,2,4>` | **NA=4, discriminator** |

The M=6 anchor is deliberately `KT=0`, i.e. the K-tile loop folded away, which
makes it the sequential row-blocked kernel E38 actually timed. Replicating
1.1054 there is the check that this harness reproduces E38 before any new claim
is read off it.

**Pre-registered decision rule.** With the M=4 rung defining the tax:

- M=8 recovers **≥ 50 %** of the tax → **MEM** → build deliverable (b).
- M=8 recovers **≤ 10 %**, or the M=4 → M=8 step falls inside the ±0.46 %
  control band → **ILP / register tile** → **K-tiled staging is dead; stop and
  report** without building (b).
- Total recovery already inside the control band at M=4 → informative null.

**No E2E leg was run**, per the assignment: the predicted end-to-end move is
0.07–0.5 %, far inside the n=4 minimum detectable effect (0.417 % / 0.632 %).
All ratios below are **kernel-level only**.

## Provenance and gates

| item | value |
|---|---|
| base commit measured | `5d97fe3` (kernel template landed, dispatch table untouched) |
| arm commit measured | `dfe39af` (dispatch table selects the ladder) |
| host | Apple M4 Pro, 48 GiB |
| scope | `quantized.h` + `mlx-generated/quantized.cpp` only (2 paths) |
| twin lock | `cpp = cpp[0:13] + h + cpp[-6:]`; `twin_audit.py` OK, 29 runtime-effective twins |
| editable budget | source 2 467 227 / 3 000 000; growth 11 938 / 262 144 |

Thermal honesty, as required for an ungated local arm: this host's real 40 °C
cool gate cannot be reached (it stalls at ~43.4 °C), so both curves record
`cool_gate_vendored=stalled_above_40C`. Entry and exit GPU temperatures are
logged per arm, and `cool_gate_passed_real_gate=false` /
`gate_qualified_for_timing=false` are preserved verbatim in the W&B record. E38
measured under the same condition, which is what makes the anchor comparable.
These are **directional causal kernel measurements within one counterbalanced
session**, not gate-qualified numbers and not any kind of score.
