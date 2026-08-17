# E13 — Is `NA <= 4` in `qmv_fast_crossrow_affine4_g64_wide` a register cliff?

- **Assignment:** `qwen38-r1-e13-na4-register-cliff` (PR #15, rev `r1`), student `qwen-thorfinn`
- **Base:** `fe38ecc21e4084e4d17dac3aa76264bb5897a614` · **Result commit:** see branch head
- **Host:** Mac mini, Apple M4 Pro, 14 cores, 48 GB, macOS 26.5.2,
  `Apple metal version 32023.883 (metalfe-32023.883)`, target `air64-apple-darwin25.5.0`
- **Scope:** static only. Phase 3 (timing) was explicitly forbidden; no timing was run.
- **Arch-transfer caveat:** the ranked runner is M5. Every number here is M4 Pro / host-LLVM.

## Verdict

**Hypothesis REFUTED, and the brief's motivating model is wrong in two independent ways.**

1. There is no register cliff at `NA=4`. The `static_assert(NA >= 2 && NA <= 4)` at
   `quantized.h:980` is a **pure policy bound with nothing behind it** — `NA=5`, `6` and `8`
   all compile cleanly once the assert is widened. The first real compiler discontinuity
   is at **`NA=6`**.
2. Lifting the assert alone would change nothing on the scored path. The assert binds
   **`IPG`** (the packing factor), and production only ever instantiates `IPG ∈ {3,4}` from a
   hardcoded `M → IPG` table (`quantized.h:1809-1852`). **The cap is not the binding
   constraint; the dispatch table is.**

## Q1 — Is `NA <= 4` a register cliff?

No. Raw frontend `-O2` AIR is structurally **identical** for every `NA ∈ {2,3,4,5,6,8}`:
`allocas=5`, `loop_backedges=11`, `loads=16`, `device_loads=7`, `float_ops=19`,
`vector_float_ops=12`, `br=24`, `gep=26`, `store=8`. The only difference is the vector width
inside `[4 x <NA x float>]`. The `m` loop is still rolled, so raw AIR is structurally
NA-blind and cannot answer the question by itself.

Alloca alignment does partition, but not the way the hypothesis needs (datalayout has
`v96:128:128`):

| NA | alignment | private footprint |
|---:|---|---|
| 2 | `align 8` | ~128 B |
| 3, 4 | `align 16` | ~192 B (identical) |
| 5, 6, 8 | `align 32` | ~320 B |

That is `{2} | {3,4} | {5,6,8}` — which does **not** match the 3-vs-4 timing partition the
cap was supposedly protecting.

### The instrument that actually resolves it: forced unroll

Beyond the brief, the modules were re-run through
`xcrun -sdk macosx metal-opt -passes='default<O3>' -S` to force unroll + SROA, and a
lane-weighted peak-live-SSA estimator was added to `research/air_kernel_stats.py`.

| NA | lines | allocas | surviving private types | float_ops | dev_loads | loads | backedges | peak_live_regs |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | 513 | 1 | `[4 x [4 x i16]]` | 62 | 32 | 38 | 2 | 62 |
| 3 | 602 | 1 | `[4 x [4 x i16]]` | 69 | 36 | 42 | 2 | 83 |
| 4 | 691 | 1 | `[4 x [4 x i16]]` | 76 | 40 | 46 | 2 | 104 |
| 5 | 780 | 1 | `[4 x [4 x i16]]` | 83 | 44 | 50 | 2 | 125 |
| 6 | 863 | **2** | + `[4 x <6 x float>]` ← **spill** | 90 | 48 | 62 | 2 | 144 |
| 8 | 719 | **2** | + `[4 x <8 x float>]` ← **spill** | 104 | 56 | 67 | **3** | 177 |

`peak_live_regs` is exactly linear at **+21/NA** for `NA=2→5` (62, 83, 104, 125). At `NA=6`
a second alloca appears, the accumulator array `[4 x <6 x float>]` is spilled to private
memory, loads jump +12 in one step, and the trend breaks (+19 instead of +21, because the
spilled value stops being live). `NA=8` adds a third backedge.

**The accumulator register cliff is real, but it sits at `NA=6`. `NA=5` is free.**

Rolled-AIR `peak_live_regs` is likewise perfectly linear — 41, 50, 60, 70, 80, 100 for
`NA=2,3,4,5,6,8` — with no knee at 4.

> Caveat, stated plainly: this is host Apple LLVM 32023.883 at O3, **not** the AGX back end.

## Q2 — What is the cap actually made of?

Policy, not compiler enforcement. With the assert widened to `NA <= 8`, `crossrow_na5`,
`crossrow_na6` and `crossrow_na8` all compile and link into a metallib with no diagnostics.

`vec<float,5>` and `vec<float,6>` are legal Metal (clang `ext_vector_type`). A standalone
probe confirms `vec<float,N>` lowers to a true `<N x float>` for `N = 2,3,4,5,6,8,16`:

| N | 2 | 3 | 4 | 5 | 6 | 8 | 16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `sizeof` | 8 | 16 | 16 | 32 | 32 | 32 | 64 |

## Q3 — What is `M`, and whose kernel is this?

**Part 1 — CONFIRMED.** `M` is the verify width = `draftCount + 1`:

| Step | Citation |
|---|---|
| depth chosen | `Qwen36MTPBlockSession.swift:711` `draftCount = draftPolicy(depth, roundCount)` |
| ceiling | `maxDepth` → `Constants.swift:331/337` = `8` ⇒ width ≤ 9 |
| width built | `:904-906` `verifyTokens = concatenated([primary] + draftIdArrays, axis: 1)` → `[1, draftCount+1]` |
| one target call | `:923-927` `callWithHiddenAndNormed(..., nConfirmed: 1)` |
| `M` computed | `quantized.cpp:1412` `int M = non_batched ? x.size()/K : x.shape(-2);` |
| `M` → grid.x | `quantized.cpp:254` `grid_dims(M, (N+bn-1)/bn, B)` |
| kernel switches | `quantized.h:1809` `switch (ntg.x)` |

`maxDepth = 8` ⇒ `M ≤ 9`, which is exactly why the switch covers `M ∈ [2,9]`, and why
`vector_limit = 10` for these shapes (`quantized.cpp:84-124`, `D>4096` branch) so `M ≥ 10`
escapes to `qmm`.

**Part 2 — REFUTED.** This is predominantly the **target's** projection kernel, not the
proposal head's. The head drafts strictly at `M=1` (autoregressive), so it only reaches the
crossrow path on a data-dependent history flush. All seven scored affine-4/g64 target
families have `N ≥ 4096` and `K % 512 == 0`, so they **all** take the wide `_m` branch at
every verify call:

| family | shape | count |
|---|---|--:|
| `in_proj_fused_qkvzba` | 5120→16480 | 48 |
| `linear_attn.out_proj` | 6144→5120 | 48 |
| `full_attn.qkv_proj_fused` | 5120→14336 | 16 |
| `full_attn.o_proj` | 6144→5120 | 16 |
| `mlp.gate_up_fused` | 5120→34816 | 64 |
| `mlp.down` | 17408→5120 | 64 |
| `head.lm_head` | 5120→248320 | 1 |

Two further corrections: (a) crossrow starts at `M ≥ 2`, not `M ≥ 3` — `M=2` uses a separate
pair-form kernel `qmv_fast_crossrow_affine4_g64<T,M>` with `inputs_per_group` hardcoded to 2
and its own `static_assert(M >= 2 && M <= 9)`; (b) in the `1024 ≤ N < 4096` band that pair
form handles all of `M ∈ [2,9]` and the `NA` cap is irrelevant there.

## The finding that matters: the cap costs weight passes, not registers

`_m<T,M,IPG>` (`quantized.h:1054-1082`) splits `M` rows into `ceil(M/IPG)` working groups;
each group re-reads and re-dequantizes the whole weight tile.

| M | IPG (prod) | working groups | best if NA uncapped | saving |
|--:|--:|--:|---|--:|
| 2 | 2 (pair) | 1 | 1 | — |
| 3 | 3 | 1 | 1 | — |
| 4 | 4 | 1 | 1 | — |
| **5** | 3 | **2** | 1 (`IPG=5`) | **50%** |
| **6** | 3 | **2** | 1 (`IPG=6`) | **50%** |
| 7 | 4 | 2 | 2 | — |
| 8 | 4 | 2 | 2 | — |
| **9** | 3 | **3** | 2 (`IPG=5`) | **33%** |

`static_assert(M % IPG != 1)` forbids a one-input tail, which blocks `M=6/IPG=5` and
`M=7/IPG=6`. So the reachable prize is: **`NA=5` unlocks `M=5` (2→1 group) and `M=9` (3→2);
`NA=6` additionally unlocks `M=6` (2→1)**. Since `NA=5` is spill-free, `M=5` is the free win.

**Honest caveat on magnitude.** Both working groups for a given `tid.y` read the *same*
weight rows (`out_row = tid.y*8 + simd_gid*4` is independent of `tid.x`), so the duplicated
work is certainly fetch instructions + dequant ALU, but the duplicated **DRAM** traffic may
be absorbed by L2 if the tiles stay resident. That is precisely why this needs a timing slot
and cannot be settled statically. Note also `sdpaWidthWallDepthCap = 5` implies `M ≤ 6` is
the operating range — so `M=5` and `M=6`, the two widths the cap penalizes, are exactly the
widths in play.

## Occupancy readout: saturated, uninformative

All six kernels report `maxTotalThreadsPerThreadgroup=1024`, `threadExecutionWidth=32`,
`staticThreadgroupMemoryLength=0` on Apple M4 Pro — **including `NA=8`, which by AIR is
visibly spilling**. The API cannot resolve register pressure here and must not be used as
evidence either way.

```
=== /tmp/probe_base.metallib ===   crossrow_na2/na3/na4        → 1024 / 32 / 0
=== /tmp/probe_wide.metallib ===   crossrow_na2/3/4/5/6/8      → 1024 / 32 / 0 (all)
```

## Back-end ISA route: attempted, blocked

`applegpu-nt` / `metal-nt` (AIR Native Translator, supports `-S`) rejects the module: the
frontend emits **AIR 2.8** while every offline target (`applegpu_g16p`, `g17p`, `g17g`,
`g17s`, `g18p`) caps at **AIR 2.5**. `-mmacos-version-min=13.0/14.0` does not lower the
emitted AIR version (still `i32 26, i32 5`); `-std=metal3.0` fails for lack of native
`bfloat`; `.air`/`.metallib` inputs are unparseable by the translator. No offline AGX
register/ISA readout is available on this host.

## Integrity — scaffolding reverted, submitted surface untouched

| state | git-blob | sha256 |
|---|---|---|
| `quantized.h` before | `24d88c20699a6af74f60047f262356cbf08ed3a0` | `b99146e9…2fe1245` |
| `quantized.h` scaffolded (`NA<=8`) | `a4c2a7b378f7ec2bd9524b209e1438ff7a585536` | `a36af5f3…8f118c4` |
| `quantized.h` **after revert** | `24d88c20699a6af74f60047f262356cbf08ed3a0` | `b99146e9…2fe1245` |

Reverted with `git checkout HEAD --`; digest is an **exact match** to base and
`static_assert(NA >= 2 && NA <= 4, ...)` is verified back at line 980. The re-compiled base
probe reproduces byte-identically (`/tmp/probe_base.ll` and `/tmp/probe_recheck.ll` both
sha256 `8859c1ad…f637abe86`).

```
$ senpai/check-editable-budget.sh fe38ecc21e4084e4d17dac3aa76264bb5897a614
editable budget OK: source=2402203/3000000 bytes headroom=597797 growth=0/262144
  exempt=2410/2147483648 files=154
```

**Growth = 0 bytes.** The two committed files (`research/air_kernel_stats.py` +50,
`research/crossrow_na_probe.metal` +9) are research-only and outside `benchmark.json`
`editablePaths`. Note: `check-editable-budget.sh` against `32b94cb6` (organizer promoted
frontier) errors with *"not a commit in this repository"* — that SHA is not fetched here.

## Scorecard vs the 6 preregistered predictions

| # | Prediction | Outcome |
|--:|---|---|
| P1 | alloca/backedge discontinuity at `NA=4` | **REFUTED** — none in raw AIR; forced-unroll discontinuity is at `NA=6` |
| P2 | `maxTotalThreadsPerThreadgroup` saturates at 1024 | **CONFIRMED**, and stronger than predicted: still 1024 at `NA=8` while spilling |
| P3 | `NA=5` fails to compile (`vec<float,5>` illegal) | **REFUTED** — legal, `sizeof=32`; `NA=5`, `6`, `8` all compile |
| P4 | Q3 confirms `M` is verify width | **HALF CONFIRMED** — `M = draftCount+1` confirmed; "proposal head's kernel" REFUTED |
| P5 | lifting the cliff buys < 0.010 ranked score | **NOT MEASURED** (no timing slot). Static analysis suggests this may be an underestimate — see caveat above |
| P6 | student finds ≥ 1 thing wrong in the brief | **CONFIRMED** — P1, P3, P4-part-2, plus the IPG-vs-M and `M≥2` corrections |

## Reproduction

```bash
xcrun -sdk macosx metal -std=metal3.1 -S -O2 \
  research/crossrow_na_probe.metal -o /tmp/probe_base.ll

# needs the NA<=8 assert widening, then revert with `git checkout HEAD --`
xcrun -sdk macosx metal -std=metal3.1 -S -O2 -DCROSSROW_NA_PROBE_WIDE \
  research/crossrow_na_probe.metal -o /tmp/probe_wide.ll

xcrun -sdk macosx metal-opt -passes='default<O3>' -S \
  /tmp/probe_wide.ll -o /tmp/probe_wide_o3.ll
python3 research/air_kernel_stats.py /tmp/probe_wide_o3.ll --match crossrow_na
```

Occupancy: `run_job` `9df5f538-2ebe-4a7c-b5f6-0f20a3f1e5de`, argv
`[research/await-lock-then-run.sh, 600, /tmp/na_occupancy_all.sh]`, finished exit 0 in
0.56 s (lock respected, no model-holding process).

**W&B:** no runs. Phase 3 was forbidden and no GPU timing or training was performed. The
only GPU-adjacent job was the 0.56 s metallib reflection query above, which emits no metrics.

## Suggested follow-ups (not implemented)

1. **Highest value, needs a timing slot.** Widen the assert to `NA<=5` *and* change the
   dispatch table entry `M=5` from `IPG=3` to `IPG=5` (and `M=9` from 3 to 5). Both are
   spill-free. Measure whether collapsing 2 working groups to 1 at `M=5` is real or absorbed
   by L2. **Without the table change the assert edit is a no-op.**
2. **Cheap and decisive for prioritization.** Log the verify-width histogram
   (`draftCount+1` per round) in one `--local-iterate`. If `M=5/6` are rare the whole
   mechanism is worth < 0.005 and should be closed; if depth pins at 4–5 under
   `sdpaWidthWallDepthCap=5` it is worth a real slot.
3. Also log `flushTokens.count` per round to bound how often the head itself reaches
   `M ≥ 2` — statically unresolvable.
4. **Do not** spend further effort on `NA ≥ 6`: it spills the accumulator array, and the tail
   rule blocks the `M=6/IPG=5` and `M=7/IPG=6` combinations that would be the cheap wins.
5. AGX register counts, if ever needed, require a GPU-side experiment or a newer offline
   translator; the AIR 2.8 vs 2.5 gap blocks the offline route on this host.
