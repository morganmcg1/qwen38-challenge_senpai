# E14 — IPG weight passes: is the second weight pass real, and is width or streaming the bigger tax?

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":false,"value":null},"test_metric":{"name":"qmv_parity_cells_differing","available":true,"value":8}}

- Student / branch: `qwen-thorfinn` / `qwen-thorfinn/ipg-weight-passes` (PR #16)
- Hypothesis and target cost: the `qmv_fast_crossrow_affine4_g64_m<T,M,IPG>` dispatch table
  makes weight passes `ceil(M/IPG)` jump from 1 to 2 exactly at `M=5`; if that second pass is
  a real DRAM re-read it should be worth ~1 depth-0 round, and removing it at `M=5`
  (`<5,3>` -> `<5,5>`) should make depth 4 much cheaper.
- Decision: **dead as a speedup, green as measurement.** Both directions of the 2x2 were
  measured. The second weight pass is real but ~89% absorbed by cache, and the only lever
  that removes it costs 3.5x more than it saves.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: measured on
  `ef16dea4ab2cb1023eb96f3740e0dfdbd88a3bda`, branch rebased onto
  `b85e7827158eb8c29b6b290a9e2971812f7e70b4` for merge (see *Base moved after the
  measurements* below) / see `senpai/frontier-state.json` / no candidate proposed
  (research-only result)
- Yukon promoted submission / source ref used as frontier: unchanged; this experiment
  proposes no submission.
- Submitted candidate files: **no submitted runtime change; one test file extended.** All arm
  patches were applied transiently by `research/run-ipg-arms.sh` and restored by its exit
  trap, so nothing inside `editablePaths` differs from the base. The branch also restores and
  extends `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` (722 -> 792 lines), which Yukon does
  not submit: `Tests/` is not in `benchmark.json editablePaths` at either base, and
  `senpai/validate-assignment-scope.sh` rejects that path by name (receipt below).
- Recovered test suite: `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` is **absent at
  `b85e782`** — the frontier-sync merge resolved in favour of the sync side and silently
  dropped it (`Tests/` holds 58 files at `e6e6f81`, 57 at `b85e782`, and that is the single
  lost file). This PR is the recovery vehicle for it: the rebase hits a modify/delete conflict
  on exactly that path, resolved by **keeping** the 792-line version, which carries
  `sweepQuantizedMatmulOverVerifyWidth`, `scoredShapesStayOnTheQMVFastPath`,
  `nOnlyDimsAreNotSafeAsReductionDims`, `sweepCompactDraftReadoutOverBits`, and
  `QwenQMVParityTests.digestQuantizedMatmulOverVerifyWidth` — the bit-exactness gate this
  experiment's parity verdict depends on.
- Supporting test, tooling, or documentation files: `research/run-ipg-arms.sh`,
  `research/roofline_arm_patch.py` (arms `ipg-a`, `ipg-b`, `ipg-d`, `ipg-e`, `perturb`),
  `research/ipg_h_from_curve.py`, `research/ipg_shape_breakdown.py`,
  `research/ipg_depth_frequency.py`, `research/run-qmv-parity.sh`,
  `research/qmv_parity_compare.py`, `research/depth_histogram.py`,
  `research/ipg_wandb_log.py`, this file.
- MTP head provenance and draft policy: organizer-pinned head, unchanged. No
  `mtp-head.manifest.json` declaration was added.
- Assignment-scope preflight, re-run against the rebase base `b85e782`:

  ```text
  $ senpai/validate-assignment-scope.sh b85e7827158eb8c29b6b290a9e2971812f7e70b4 \
      Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
      Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
  assignment scope OK: 2 submitted path(s) against BASE_SHA=b85e7827158eb8c29b6b290a9e2971812f7e70b4
  ```

  (The same run against `ef16dea4…` gave `assignment scope OK: 2 submitted path(s)`.)
  The recovered test file is confirmed non-submitted by the same script:

  ```text
  $ senpai/validate-assignment-scope.sh b85e7827158eb8c29b6b290a9e2971812f7e70b4 \
      Tests/MLXFastTests/QwenQMVCostCurveTests.swift
  assignment scope: 'Tests/MLXFastTests/QwenQMVCostCurveTests.swift' is outside
  b85e7827158eb8c29b6b290a9e2971812f7e70b4:benchmark.json editablePaths
  ```

- Editable source bytes / headroom / growth / exempt-head bytes, re-run against `b85e782`:

  ```text
  $ senpai/check-editable-budget.sh b85e7827158eb8c29b6b290a9e2971812f7e70b4
  editable budget OK: source=2403812/3000000 bytes headroom=596188 growth=0/262144
  exempt=2410/2147483648 files=154 (growth base=b85e782…; contract=b85e782…;
  base source=2403812, exempt=2410, files=154)
  ```

  `growth=0` because this branch changes no submitted path at all. The base source figure
  moved `2402203 -> 2403812` bytes purely because the organizer's own base advanced; it is
  not growth attributable to this experiment. (The same run against `ef16dea4…` gave
  `source=2402203/3000000 headroom=597797 growth=0/262144 exempt=2410 files=154`.)
- Scored-path reachability evidence: the device-side dispatch `switch (ntg.x)` at
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:1809` is guarded by
  `!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024`; the wide `_m` branch
  additionally needs `out_vec_size >= 4096`. All eight projection shapes measured here have
  `n >= 4096`, so every one takes the wide `_m` branch on the scored path.

## W&B runs

Group `qwen38-r1-e14-ipg-weight-passes`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`. One `analysis` run per measured arm
carrying its full cost curve, plus one comparison run carrying the cross-arm tables and one
run carrying the Q4 end-to-end dispatch-frequency screen. Eleven runs, all `finished`.

| run | arm | NA_max scored | run id |
| --- | --- | ---: | --- |
| `qmv-cost-curve-e14-ref1` | reference | 4 | [`88khsek3`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/88khsek3) |
| `qmv-cost-curve-e14-ref2` | reference repeat | 4 | [`bfk6o414`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/bfk6o414) |
| `qmv-cost-curve-e14-ref3` | reference repeat | 4 | [`97ieuck5`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/97ieuck5) |
| `qmv-cost-curve-e14-armB` | `<4,4>` -> `<4,2>` | 4 | [`tu839z8z`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/tu839z8z) |
| `qmv-cost-curve-e14-armA` | `<5,3>` -> `<5,5>` | 5 | [`e62r389y`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e62r389y) |
| `qmv-cost-curve-e14-armA3` | arm A repeat | 5 | [`a10cxpfs`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/a10cxpfs) |
| `qmv-cost-curve-e14-armD` | arm A + packed `acc` | 5 | [`qnwqdh03`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/qnwqdh03) |
| `qmv-cost-curve-e14-armE` | arm A + packed all 13 | 5 | [`md5dlsm0`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/md5dlsm0) |
| `qmv-cost-curve-e14-armE2` | arm E repeat | 5 | [`fdus9cxa`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/fdus9cxa) |
| `e14-ipg-weight-passes` | cross-arm comparison | — | [`2qvqo4z8`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2qvqo4z8) |
| `e14-q4-dispatch-frequency` | Q4 end-to-end screen | — | [`sxau0sjl`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/sxau0sjl) |

The comparison run carries tables `e14/arms`, `e14/per_shape_excess`, `e14/h_by_depth` and
`e14/weighted_verify_seconds`, and summary keys `ref/h4_pass_spike`,
`ref/structural_pass_cost_h_units` and `ref/row_tax_over_stream_tax`.

The Q4 run is the only end-to-end run in the group. It carries tables
`q4/dispatch_occupancy` and `q4/policies`, and summary keys `q4/m5_round_share`,
`q4/m5_best_case_end_to_end_pct`, `q4/multi_pass_round_share` and
`q4/three_pass_m9_round_share`, alongside the `e2e/*` decode metrics. Its config records
`ranked_equivalent: false` and `screen_kind: directional-policy-screen` because it runs a
256-token window rather than the ranked 512; its dispatch *frequencies* are the transferable
result, not its absolute timings.

`NA_max scored` is the stream law `ceil(M/NA_max)` each curve's staircase test is scored
against, so it tracks the arm rather than the shipped kernel. It is a labelling choice for
the staircase diagnostic only and does not touch any measured time.

## Base moved after the measurements (r2 bookkeeping)

Every timed arm below was measured against `ef16dea4`. This branch is now rebased onto
`b85e782`, and the organizer's base changed one line of the dispatch table in the window
between them. Nothing here is a re-measurement; it is arithmetic over the already-measured
numbers plus a source diff, and it exists so the next agent does not read a stale claim as
current.

`git diff ef16dea4..b85e782 -- Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
touches exactly one `case`:

```text
case 8: qmv_fast_crossrow_affine4_g64_m<T,8,4>  ->  qmv_fast_crossrow_affine4_g64_m<T,8,3>
```

with an organizer comment citing 319 / 437 / 216 us for `M = 7 / 8 / 9` — the same register
cliff this experiment measured independently. That is a useful outside confirmation of the
mechanism: two agents found the same `<T,8,4>` outlier from opposite directions, one from a
cost curve and one from a dispatch retune.

What that change does and does not invalidate:

| claim | status on `b85e782` |
| --- | --- |
| arm A (`<T,5,3>` -> `<T,5,5>`) | **stands** — `case 5` is byte-identical across the diff |
| arm B (`<T,4,4>` -> `<T,4,2>`) | **stands** — `case 4` is byte-identical across the diff |
| arm E (arm A + packed 13 vectors) | **stands** — built on `case 5`, untouched |
| the `h8 = 0.3816` pass spike | **superseded** — measured against `<T,8,4>`, which no longer ships |
| the histogram frequency model | **superseded** — `segmentedStreakGate` moved 3 -> 2 |
| Q1 / Q2 / Q3 pass-vs-row conclusion | **stands** — it rests on arms A, B and the `M=3..7` curve |

The pass vector for `M = 3..9` was `1,1,2,2,2,2,3` when measured and is `1,1,2,2,2,3,3` on
`b85e782`. The pass increment therefore moved from `M=8` to `M=7`. That gives a falsifiable
prediction rather than a dead claim: on the new base the structural pass spike should appear
at `h7`, not `h8`, and `h8` should fall back onto the `NA=3` two-pass line. This experiment
did not run that check and should not be read as having run it.

`segmentedStreakGate` also moved `3 -> 2` in `Qwen36MTPBlockSession.swift` between the two
bases. Every frequency number in Q4 is pinned to gate 3 and therefore *understates* how often
`M >= 5` fires on the new base. Re-indexing the already-measured histogram against the new
dispatch table — arithmetic only, no new run — moves the three-pass round share from
**48.57%** (`M=9` alone under `<8,4>`) to **57.14%** (`M=8` at 8.57% plus `M=9` at 48.57%,
both three-pass under `<8,3>`). The direction of follow-up #8 is unchanged and its
motivation is slightly stronger, but its exact numbers need the new base.

## Question

Does verify width `M` cost what it costs because of the *number of weight passes*
(`ceil(M/IPG)`), or because of the *number of rows / accumulator width* (`NA`)? And if the
second weight pass at `M=5` can be removed, does depth 4 become worth offering?

## Evidence that made it worth testing

- Host: Mac mini, Apple M4 Pro, 14 cores, 48 GB. **The ranked runner is M5; every number
  here carries architecture-transfer risk and is directional only.**
- Toolchain / thermal policy: repository-frozen Swift dependency graph
  (`--force-resolved-versions`); every resident measurement went through the wrapper's run
  lock and 40C cooling gate. Per-leg temperatures are recorded in
  `.mlxfast-private/qmv-curve/<TAG>/start-temps.txt`.
- The shipped dispatch table **as measured, pinned to `ef16dea4`** (`quantized.h:1809`) is:

  | M | template | passes = ceil(M/IPG) | NA = IPG (or TAIL) |
  | ---: | --- | ---: | ---: |
  | 2 | `qmv_fast_crossrow_affine4_g64<T,2>` | — (different family) | — |
  | 3 | `_m<T,3,3>` | 1 | 3 |
  | 4 | `_m<T,4,4>` | 1 | 4 |
  | 5 | `_m<T,5,3>` | 2 | 3 |
  | 6 | `_m<T,6,3>` | 2 | 3 |
  | 7 | `_m<T,7,4>` | 2 | 4 |
  | 8 | `_m<T,8,4>` | 2 | 4 |
  | 9 | `_m<T,9,3>` | 3 | 3 |

  On the rebase base `b85e782` the `M=8` row is `_m<T,8,3>`, 3 passes, `NA=3`. Every other
  row is unchanged. See *Base moved after the measurements* above.

## Smallest decisive test

A **2x2**: add a pass / remove a pass, each at a known `NA` change, measured on the same
host in one session against a bracketing reference.

| arm | edit | passes | NA | direction |
| --- | --- | --- | --- | --- |
| ref1 / ref2 | unpatched | — | — | bracketing drift + noise floor |
| **A** | `_m<T,5,3>` -> `_m<T,5,5>` (+ `static_assert` widened to `NA<=5`) | 2 -> 1 | 3 -> 5 | remove a pass, widen accumulator |
| **B** | `_m<T,4,4>` -> `_m<T,4,2>` | 1 -> 2 | 4 -> 2 | add a pass, narrow accumulator |
| **D** | A + exact `float acc[rows][NA]` packing | 2 -> 1 | 3 -> 5 (unpadded) | isolate the padding tax inside A |

Each arm is one width's table entry, so **every other width in the same binary is an
in-session control**. That is what makes a 1% effect measurable on a shared machine.

### Measurement method

`MLX_QWEN_MTP_FORCE_DEPTH` / `MLX_QWEN_MTP_H_VECTOR` **do not exist in `Sources/` at
`ef16dea4`** (only `MLX_QWEN_MTP_TRACE`, `Qwen36MTPBlockSession.swift:462`), so forced-depth
end-to-end arms were impossible without an out-of-scope source change. Instead h is derived
from the QMV cost curve:

```text
wvs[M] = sum over the 8 scored shapes of seconds_per_call[M] * calls_per_verify
h(d)   = (wvs[d+1] - wvs[d]) / wvs[1]
```

This reproduces edward's shipped vector closely (see Q3), which validates the derivation.

### Estimator correction (a real methodology finding)

The natural `mean`-of-15-reps estimator showed `M=1` — an **untouched control width** —
moving **+7.3%** between two builds. Per-shape decomposition traced all of it to
`linear_attn.in_proj_fused_qkvzba` at `M=1`, which is the **first timed row of the first
shape of the sweep**, with intra-run `max/min` = **1.730** in armB vs **1.065** in ref1.
That is a clock-ramp transient, not a kernel effect.

Switching to `seconds_per_call_min` (best-of-15) is now the standard estimator for this
tooling:

| | mean | min |
| --- | ---: | ---: |
| control widths M=3,5,6,7,8,9 | +/-0.5% | **+/-0.27%** |
| M=1 (untouched control) | +7.4% | +3.2% |
| M=4 (the changed entry, arm B) | +7.75% | **+8.15%** |

The signal on the changed width *grows* while the controls tighten, which is what a real
steady-state cost looks like. The min estimator also reproduces edward's `h4` to 1.6%
(0.3816 vs 0.3754 shipped) and recovers the `h1 > h2` ordering that the mean estimator
inverts.

## Results

### Reference (ref1, min estimator)

`wvs` (seconds per verify, all 8 shapes weighted by calls):

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wvs | 0.058462 | 0.064497 | 0.069949 | 0.084780 | 0.107088 | 0.123076 | 0.139277 | 0.155408 | 0.177720 |

`h_ref = [0.1032, 0.0933, 0.2537, 0.3816, 0.2735, 0.2771, 0.2759, 0.3816]`
(edward's shipped vector: `[.0842, .0775, .2426, .3754, .2919, .3000, .2870, .3909]`)

The reference was remeasured from scratch three times across the session, so `h4` — the
quantity every Q3 conclusion turns on — has an independent spread rather than a single
reading:

| reference | started | h4 |
| --- | --- | ---: |
| e14-ref1 | 10:04:23Z | 0.3816 |
| e14-ref2 | 10:33:50Z | 0.3764 |
| e14-ref3 | 10:49:23Z | 0.3880 |

Mean **0.3820, spread +/-1.5%**. Edward's shipped 0.3754 sits 1.7% below that mean, i.e.
just outside the spread and in the direction that makes depth 4 look *cheaper* on his
vector than on mine. The Q3 threshold below is quoted on `h4 = 0.3816`; recomputing it on
0.3764 or 0.3880 moves the required cut between 3.6% and 6.5% and changes no conclusion,
because depth 4 loses to depth 3 and depth 7 on cost-per-token at every value in that range.

`ref3 h = [0.0931, 0.0874, 0.2407, 0.3880, 0.2624, 0.2730, 0.2665, 0.3846]`

Round-cost model fitted on the same session:
`C(d) = G(d+1) + 4.20 ms + 3.96 ms/draft`, max |residual| 2.13 ms, `C(0) = 65.07 ms`.

### Q1 — is the second weight pass real, or absorbed by L2?

**It is real, and it is ~89% absorbed.**

Two independent routes agree:

*Structural* — line the h-steps up against the dispatch table:

| step | M -> M+1 | passes | NA | h (min) |
| --- | --- | --- | --- | ---: |
| h3 | 3 -> 4 | 1 -> 1 | 3 -> 4 | 0.2537 |
| **h4** | 4 -> 5 | **1 -> 2** | 4 -> 3 | **0.3816** |
| h5 | 5 -> 6 | 2 -> 2 | 3 -> 3 | 0.2735 |
| h6 | 6 -> 7 | 2 -> 2 | 3 -> 4 | 0.2771 |
| h7 | 7 -> 8 | 2 -> 2 | 4 -> 4 | 0.2759 |
| **h8** | 8 -> 9 | **2 -> 3** | 4 -> 3 | **0.3816** |

The two spikes land **exactly** on the two pass-count increments, and they are the **same
height**. Spike minus the mean of the four flat steps (0.2701) = **0.1115 h-units**.

*Scoped to `ef16dea4`:* the `h8` row is a property of the `<T,8,4>` entry that shipped when
these curves were taken. `b85e782` moved `M=8` to `<T,8,3>`, so on the current base the
second increment sits at `h7` and `h8` should fall onto the flat `NA=3` line. The `h4` row
and the Q1 conclusion are unaffected — `case 4` and `case 5` did not move. Treat the `h8`
number as superseded evidence and the `h7` shift as an untested prediction.

*Interventional* — arm B forces a second pass at `M=4`, where the table ships one:
**+0.1161 h-units** drift-adjusted (ratio 1.0801 -> 1.0816; control widths [3,5,6,7,8,9]
spanned [0.9977, 1.0013], median drift 0.9986, noise floor +/-0.27%).

The two routes agree within 4% (and to 4 decimal places — 0.1084 both ways — under the mean
estimator). A genuine full re-read of the weights would cost about one whole depth-0 round,
so **~89% of the second pass is served from cache.**

Per-shape excess in arm B scales cleanly with weight footprint, which is the signature of
weight traffic rather than scheduling:

| shape | n | k | W MiB | calls/verify | M4 excess | share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head.lm_head | 248320 | 5120 | 682.03 | 1 | 12.07% | 6.3% |
| head.compact_draft_vocab | 98336 | 5120 | 270.09 | 0 | 11.29% | 0% |
| mlp.gate_up_fused | 34816 | 5120 | 95.62 | 64 | 11.03% | 56.0% |
| full_attn.qkv_proj_fused | 14336 | 5120 | 39.38 | 16 | 9.89% | 5.7% |
| linear_attn.in_proj_fused_qkvzba | 16480 | 5120 | 45.26 | 48 | 8.73% | 17.3% |
| linear_attn.out_proj | 5120 | 6144 | 16.88 | 48 | 4.51% | 4.3% |
| mlp.down | 5120 | 17408 | 47.81 | 64 | 3.25% | 9.9% |
| full_attn.o_proj | 5120 | 6144 | 16.88 | 16 | 1.32% | 0.4% |

682 MiB -> 12.1% excess down to 16.9 MiB -> 1.3% excess, monotone in `W`.

*Honest caveat:* arm B changes two things at once (adds a pass **and** narrows `NA` 4 -> 2).
The +8.2% is therefore a net figure with an `NA` **credit** inside it, so the isolated
stream tax is **at least** that large. Arms A and D are the mirror image and close the 2x2.

### Q2 — which tax is bigger, NA width or weight streams?

**One-sentence verdict:** *Adding one verify row at constant pass count costs ~0.27 depth-0
rounds while adding a second weight pass costs ~0.11, so the row/NA-width tax is ~2.4x the
weight-stream tax — width, not weight streaming, is what makes deep verification
expensive.*

Arms A and D confirm this from the other direction, and add a second mechanism:

- **Arm A** (`<5,3>` -> `<5,5>`: pass removed, `NA` 3 -> 5) made `M=5`
  **+39.3%** slower (ratio 1.3932, drift-adjusted 1.3929, +0.7202 h-units; control widths
  [3,4,6,7,8,9] spanned [0.9944, 1.0016], median drift 1.0002, noise floor +/-0.58%).
  Removing a whole weight pass is worth ~0.11 h-units; widening `NA` from 3 to 5 costs
  ~0.83. The lever loses by **3.5x** in ratio terms.
- The per-shape profile of arm A is **flat** (34%–54%) and does *not* track weight
  footprint — the opposite of arm B — which is the signature of a register/occupancy effect
  rather than memory traffic. The largest excess is `mlp.down` (53.7%), the shape with the
  longest reduction (`k = 17408`), exactly where accumulator pressure bites hardest.

### Arm D — is the arm A penalty register pressure?

`_wide` keeps its accumulator as `VF acc[4]`. On this target `vec<float,5>` occupies eight
lanes, so at `NA = 5` three lanes per element are dead. Arm D is arm A with `acc` packed
exactly, and it recovered **nothing**:

| arm | ratio vs ref1 | drift-adjusted | h-units | noise floor | control span |
| --- | ---: | ---: | ---: | ---: | --- |
| ref2 (repeat) | 1.0002 | 0.9998 | +0.0003 | +/-0.39% | [0.9965, 1.0027] |
| A (`<5,3>` -> `<5,5>`) | 1.3932 | 1.3929 | +0.7202 | +/-0.58% | [0.9944, 1.0016] |
| D (A + packed `acc`) | 1.3969 | **1.3996** | +0.7270 | +/-0.49% | [0.9949, 1.0029] |

Per-shape excess is statistically indistinguishable from arm A (D vs A: 34.58/54.00/35.74/
37.70/36.14/34.25/39.75/34.50 against 34.53/53.74/35.59/33.62/36.22/34.05/40.97/34.36).

I first read this as "the penalty is not register pressure." **That read was wrong, and arm
D was simply too weak a probe.** `_wide` holds *thirteen* `NA`-wide vectors per thread —
`acc[4]`, `sums`, `partial[4]`, `a0..a3`. At `NA = 5` that is 13 x 8 = 104 lanes against 65
useful, so 39 lanes are wasted. Arm D packed four of thirteen vectors, removing 12 of those
39 lanes. Occupancy is a step function of registers per thread, so removing 31% of the waste
can plausibly cross no step at all and return exactly zero — which is what happened. The
correct conclusion from arm D alone is *"one vector is not enough to move occupancy"*, not
*"registers are not the mechanism"*. Arm E is the strong form of the same probe.

**Arm D is closed.** The advisor dropped it at r2: arm E already answers the register
question in its strong form, and re-running D would spend GPU time to re-confirm a null. The
measurement above stands as recorded; there is no further work on this arm.

### Arm E — packing all thirteen vectors

Arm E is arm A plus exact packing of all thirteen `NA`-wide vectors, with identical
elementwise arithmetic in identical order. It was run twice against two different
references:

| leg | started | reference | ref measured | control median | M=5 raw |
| --- | --- | --- | ---: | ---: | ---: |
| armE | 10:40:38Z | e14-ref1 | 36 min earlier | 1.1215 | 1.0762 |
| armE2 | 10:56:18Z | e14-ref3 | 7 min earlier | **1.1241** | **1.0739** |
| armA | 10:22:01Z | e14-ref1 | 18 min earlier | 0.9985 | 1.3932 |
| armA3 | 11:02:30Z | e14-ref3 | 13 min earlier | **0.9985** | **1.3921** |

Control span for armE2 is [1.0989, 1.1259] and for armA3 [0.9880, 1.0030]; start temps
39.73 / 39.59 / 39.48 / 39.80 C against references at 39.51 (ref1) and 39.54 C (ref3).

`armA3` is the decisive control on the drift hypothesis. It ran in the *same session and the
same thermal window* as `armE2`, four minutes later and 0.2 C warmer, against the same
reference — and its controls came back flat while `armE2`'s came back +12%. Two arms cannot
drift in opposite directions in the same session. The elevated widths belong to the arm E
patch, not to the machine. Independently of that, `armA`'s `M=5` ratio reproduces across
sessions to **0.08%** (1.3932 vs 1.3921) and its control median to four decimal places,
which fixes the run-to-run reproducibility of this fixture well below every effect reported
here.

#### The controls were never controls

On the first run I read the elevated control widths as thermal drift and reported a
drift-adjusted `0.9596`. **That was a measurement-design error, and the repeat exposes it:**
two control medians measured in different sessions against references taken 36 and 7 minutes
prior agree to **0.23%**. Thermal drift does not reproduce to 0.23%.

The cause is in the patch structure. `VF` is `vec<float, NA>` declared inside the *shared*
`_wide` body, not inside a per-`M` instantiation. Arms D and E rewrite that body, so every
`_m<T,M,IPG>` instantiation compiled from it changes — all of `M = 3..9`. Widths
[3,4,6,7,8,9] receive the intervention. Only arms A and B, which rewrite a dispatch-table
line and an assert, have genuine controls.

| arm | edits | controls valid | control span |
| --- | --- | --- | --- |
| A | dispatch `<5,3>` -> `<5,5>` | yes | [0.9944, 1.0016] |
| B | dispatch `<4,4>` -> `<4,2>` | yes | [0.9977, 1.0013] |
| D | dispatch + pack `acc` (4/13) | **no** | [0.9949, 1.0029] |
| E | dispatch + pack all 13 | **no** | [1.0989, 1.1259] |

Arm D's flat controls were luck, not design: packing four vectors happens to be free at
`NA = 3` and `NA = 4`. That coincidence is what let me misread arm D as a clean null.

**The drift-adjusted `0.9596` / `0.9553` figures are retracted.** Dividing the `M=5` ratio by
the `NA=3/4` penalty presumes packing costs the same at every `NA`, which is precisely false:
packing *helps* at `NA = 5` by removing padding and *hurts* at `NA = 3/4`. The quotient is
meaningless in both directions.

#### What arm E does establish

Two statements survive, and both are strong:

1. **Full packing is a reproducible global regression** of +12.4% median (+9.9% to +12.6%)
   at every unchanged wide width.
2. **Full packing removes the `NA = 5` penalty.** At `M = 5`, arm A costs **+39.3%** against
   shipped and arm E costs **+7.4%** — packing recovers **about 32 points**.

So the register-pressure diagnosis is confirmed, and the arm D null is now *explained*
rather than merely recorded: 12 of 39 wasted lanes crosses no occupancy step, 39 of 39 does.
The two arms together are a clean pair, which is more than either is alone.

#### Why it is still not a win

Arm E taxes the widths that actually execute — `M = 3` carries 231 of 246 verifies in
edward's histogram and `M = 4` carries 13 — by roughly 11%, in order to cheapen an `M = 5`
that is dispatched 0.00% of the time on `e6e6f81`. No schedule trades that positively.

This makes the conclusion **stronger**, not weaker. The earlier draft said to reopen if a
future toolchain removed the `vec<float,5>` padding penalty. The accurate version is that
the padding penalty is removable in source **today**, and removing it costs more elsewhere
than it saves. That is a closed door rather than a pending one.

Arm E does not touch Q2, which rests on arms A and B and on the structural h-step spikes.

## Q3 — if M=5 gets cheap, does depth 4 open?

Threshold arithmetic on the reference curve: `cumH(h1..h3) = 0.4502`, `h4 = 0.3816`,
breakeven `<= 0.3625`, so depth 4 needs a **4.99% cut in `h4`** (mean estimator: 3.65%;
edward's shipped vector: 6.48%). In absolute terms **`wvs[5]` only has to drop ~1.0%** —
which is why this arm looked promising.

**But depth 4 is never optimal anyway.** Sweeping cost-per-token
`(1 + sum(h[:d])) / sum(q^i, i=0..d)` over acceptance `q`:

| q | best d | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 7 | 0.3988 | 0.3625 | 0.3664 | 0.3509 | 0.3403 | **0.3323** | 0.3378 |

Optimal depth is 2 for `q <= 0.80`, 3 for `q = 0.85–0.95`, and 7 at `q = 1.00`. Depth 4 is a
**local bump**, not a plateau: it is the first width that pays the pass tax, while depths
5–8 amortise that *same single* tax over more accepted tokens. This holds under the min
estimator, the mean estimator, and edward's shipped vector.

So the honest answer to Q3 is: making `M=5` cheap does not "open depth 4" as a stable
operating point — it would at best flatten a bump that the scheduler already routes around.

## Q4 — what would it buy on the scored fixture?

Q1–Q3 are all measured on the isolated kernel fixture. Q4 asks the only question that
decides the assignment: **how often does the live scored path actually dispatch `M=5`, and
what is the largest end-to-end gain removing its second weight pass could possibly buy?**

### Directional policy screen, 256 decode tokens

This is a **directional policy screen, not a ranked-equivalent measurement.** The ranked
contract is 512 decode tokens per leg, and the base defect documented in the next section
makes a 512-token local run impossible on `ef16dea4`. 256 tokens is the longest window the
public fixture supports, and it is used here only to count dispatch frequencies — no timing
claim in this report rests on it.

Traced run, job `3e52f4a3-3dd2-464d-8fa3-5ffe395b2847`, `2026-08-17T11:49:46Z`–`11:56:20Z`,
exit 0, `research/out/e14-trace/`. Twins pristine
(`metallib_source_fingerprint=6639cc59d6fb84ff…`, `dirty=0`), zero stale-metallib warnings,
`all_tokens_matched=true`, `residual_divergence_count=0`, `mtp_decode_speedup=1.9265`,
`accepted_draft_rate=0.9609`, `effective_mean_draft_len=6.571`.

Round accounting closes exactly: 35 rounds, 35 primaries + 221 accepted drafts = **256
committed tokens**, matching `decode_tokens`. `implied_d0=0` — no zero-draft round was
hidden by the trace (the depth-0 branch returns before emitting its `round=` line, so
`depth_histogram.py` now reconstructs those from gaps in the round counter; there were none).

### Measured dispatch occupancy — verify width is `M = d + 1`

| offered `d` | `M` | rounds | share | dispatch | passes | `NA` |
|---|---|---|---|---|---|---|
| 1 | 2 | 1 | 2.86% | `qmv_fast_crossrow_affine4_g64<T,2>` | — | — |
| **4** | **5** | **1** | **2.86%** | **`_m<5,3>`** | **2** | **3** |
| 5 | 6 | 10 | 28.57% | `_m<6,3>` | 2 | 3 |
| 6 | 7 | 3 | 8.57% | `_m<7,4>` | 2 | 4 |
| 7 | 8 | 3 | 8.57% | `_m<8,4>` | 2 | 4 |
| 8 | 9 | **17** | **48.57%** | `_m<9,3>` | **3** | 3 |

`mean_offered_depth=6.571`, `tokens_per_round=7.314`. The schedule on this base is a **ramp**:
it opens at `d=4` and climbs to `d=8` within ten rounds as it observes near-perfect
acceptance, then saturates. Dropping the two clock-ramp warmup rounds leaves **0 of 33**
rounds at `d=4`.

*Scoped to `ef16dea4`, and superseded in two ways.* The `dispatch` / `passes` / `NA` columns
are the table that shipped when the trace was taken; on `b85e782` the `M=8` row is
`_m<8,3>`, 3 passes, `NA=3`. Re-indexing these **same measured rounds** against the new table
— arithmetic, not a new run — moves the three-pass share from **48.57%** (`M=9` only) to
**57.14%** (`M=8` at 8.57% plus `M=9` at 48.57%). Separately, `segmentedStreakGate` moved
`3 -> 2` on the new base, which changes how fast the ramp climbs, so the `rounds` and `share`
columns themselves need a fresh trace before anyone quotes them as current. Both changes push
in the same direction as the finding below.

### Validation of the recovered suite (r2, no GPU)

One `swift build --build-tests --force-resolved-versions` on the rebased tree, to check that
the recovered 792-line file still type-checks against the new base:

- `QwenQMVCostCurveTests.swift` **compiles clean** — it appears once in the compile log and
  contributes **zero** diagnostics, and `Emitting module MLXFastTests` is reached. It imports
  only `CryptoKit`, `Foundation`, `MLX`, `MLXLLM` and `Testing`, with no `@testable import`
  of the session or kernel modules the base changed, which is why the base move cannot reach
  it.
- The build nevertheless **fails**, on exactly one error in exactly one file:

  ```text
  Tests/MLXFastTests/QwenMTPVerbTests.swift:755:21: error:
  cannot convert value of type 'String' to expected argument type 'Comment?'
  ```

  That file is **byte-identical to `b85e782`** (`git diff b85e782 HEAD -- <path>` is empty)
  and is not among the 14 files this branch changes. It is a **pre-existing base defect, not
  a rebase artifact**: the offending expression is present unchanged at `e6e6f81`,
  `ef16dea4`, `d098212` and `b85e782`, introduced by `ee977ae Restore organizer test source
  snapshot`. See follow-up #9.

So the recovery is sound and the suite is ready to run, but nobody can run it until #9 lands.

### The answer

| policy | provenance | rounds | `d=4` rounds | round share | cost-weighted time share | best-case end-to-end at −7150 µs per `M=5` verify |
|---|---|---|---|---|---|---|
| PR #13 control | Edward, `e6e6f81` | 253 | 18 | 7.11% | 10.07% | **−0.622%** |
| PR #13 best `Hp` arm | Edward, `e6e6f81` | 246 | 0 | 0.00% | 0.00% | **0** |
| measured `e14-trace` | this base, live | 35 | 1 | 2.86% | 1.97% | **−0.122%** |

**`M=5` is a cold dispatch slot under every scheduler measured, on two different bases.**
On this base the best-case end-to-end value of making the second `M=5` weight pass free is
**0.12%** — an order of magnitude inside `--local-iterate` noise. The −7150 µs input is
itself the *upper bound* from Q1 (0.11 depth-0 rounds × `C(0)=65009.4` µs), i.e. the value if
the second pass were removed at **zero** cost.

No arm comes near that bound. Arm A makes `M=5` **+39.2%** more expensive and arm E **+7.4%**
more expensive, so the realised end-to-end value of the E14 mechanism is **negative under
every arm tested**. Q4 confirms the Q3 verdict from a completely independent direction: Q3
argued analytically that the scheduler routes around depth 4; Q4 shows a live traced run in
which it does exactly that.

### The finding Q4 adds that Q1–Q3 could not

E14 targeted the wrong width. **97.1% of live rounds take a multi-pass `_m` dispatch, and
48.6% take `_m<9,3>` — the only three-pass slot in the whole table.** The two-pass slot at
`M=5` that this assignment was scoped to is 2.86%. If the weight-pass hypothesis is worth
retesting at all, `M=9` is where the tax actually is, and it is roughly *twice* the tax
(two redundant passes rather than one) at seventeen times the frequency.

E14's own results already predict how that retest would go, which is why I am not claiming it
as a likely win. Collapsing `_m<9,3>` to two passes means `_m<9,5>`, which lands on exactly
the `vec<float,5>`→`vec<float,8>` padding trap that made arm A both **+39.2%** slower and
**bit-divergent**; one pass would mean `NA=9` padded to 16, which is worse. Avoiding the trap
requires arm E's `float[NA]` packing — and arm E's **+12.4%** penalty falls on the `NA=3` and
`NA=4` widths, which this table shows are `M=6, 7, 8, 9`, i.e. **94.3% of live rounds**.

That last point also sharpens the arm E verdict. I previously wrote that arm E taxes `M=3`
and `M=4` to cheapen a cold `M=5`. On this base the taxed hot slots are `M=6` and `M=9`
instead — but the shape of the trade is identical, so **arm E taxes the hot widths to cheapen
a cold one under both schedulers**, which is a materially stronger basis for rejecting it than
either histogram alone.

Artifacts: `.mlxfast-private/ipg-arms/e14-trace-hist-w0.json` (full 35 rounds),
`e14-trace-hist.json` (warmup 2), `e14-q4-frequency.json`.

Reproduce:

```bash
research/run-arm.sh e14-trace --trace --tokens 256
/usr/bin/python3 research/depth_histogram.py research/out e14-trace --warmup 0 \
  --json-out .mlxfast-private/ipg-arms/e14-trace-hist-w0.json
/usr/bin/python3 research/ipg_depth_frequency.py \
  --measured-hist .mlxfast-private/ipg-arms/e14-trace-hist-w0.json --delta-m5-us 7150
```

## Out-of-scope base defect found while collecting Q4

Collecting the depth histogram needs a traced `--local-iterate` run, and I first asked for
the ranked-equivalent window of 512 decode tokens. **That run cannot complete on the current
base.** This is not an E14 arm — the twins were pristine (`metallib_source_fingerprint=6639cc59d6fb84ff…`,
`dirty=0`) and no patch was applied — so it is a property of `ef16dea4`'s editable session,
reported here because `program.md` names this exact failure mode a solver defect.

Job `502e563f-12e3-4caf-90dc-8a888a708850`, `2026-08-17T11:36:13Z`–`11:40:48Z`, exit 1,
`research/run-arm.sh e14-trace --trace --tokens 512`.

What the trace shows, in order:

1. The reference pass **succeeds completely**:
   `mtp-verify: rows=513 seed_tokens=512 reference_seed_token=271 self_consistent=true
   (replayed 1 row bit-identically) chain_contradictions=0`.
2. The true serial control (`mtp-timed --mtp-depth 0`) starts, `mtp-trace: begin seed=512
   build_us=2947560 eval_wall_us=1047001`, and decodes **301 tokens**, `mtp-row: pos=513`
   through `pos=813`.
3. The last row is `mtp-row: pos=813 ids=248044,271`. **248044 is the EOS token** —
   `weights/config.json` has `eos_token_id: 248044` and `generation_config.json` has
   `[248046, 248044]`.
4. The next round dies: `mlxfast-swift: runtime worker mtp_decode_round failed: MTP round
   requested before the seed prefill`.

Root cause, exactly as written in the source:

- `Qwen36MTPBlockSession.swift:723` — `if stopTokens.contains(primary)` sets
  `reachedStopToken = true` and then **nils out `pendingPrimary`, `pendingTop2` and
  `pendingHidden`** (`:726-728`) before returning its 1-row result.
- `Qwen36MTPBlockSession.swift:672-674` — the next `generateRound` guard is
  `guard began, let primaryPending = pendingPrimary, let tailPending = pendingTop2,
  let hidden = pendingHidden else { throw .notBegun }`. `began` is still `true`; the three
  pendings are `nil`. `.notBegun`'s description is `"MTP round requested before the seed
  prefill"` (`:90`), which is why the message misleadingly blames the prefill.
- The parent never asked the session to stop: `reachedStopToken` **is not read anywhere**
  in `Sources/MLXFastTrustedHarness/` (grep over `QwenRuntimeBenchmark.swift`,
  `QwenRuntimeWorker.swift`, `QwenRuntimeMTPWorker.swift` returns no consumer). The trusted
  parent owns the window and keeps requesting rounds to 512, exactly as `program.md` says
  it should.
- `Qwen36MTPReferenceSession` has no such short-circuit, which is why the 513-row reference
  pass in step 1 sails past the same EOS. The two sessions disagree about what "the window"
  means, and the timed side is the one that is wrong.

Consequences worth the advisor's attention:

- It kills the **denominator**, not just the candidate: the depth-0 serial control is the
  leg that failed. On the public fixture **no arm — E14's or anyone's — can produce a
  512-token local measurement on this base.** Every local number in this campaign, mine
  included, is from a window short enough to stay inside the model's own completion.
- The public fixture's natural completion is at decoded token **302**, so 64, 128 and 256
  are all safe and 512 is not. This is fixture-specific, not a fixed 302-token limit.
- The ranked contract is 512 decode tokens per leg. If any hidden prompt emits EOS before
  its 512th decoded token, the candidate leg hits this same throw. I cannot see the hidden
  pool, so I cannot say whether it does — but the failure is one `if` away from the ranked
  path and costs a whole prompt if it fires.
- The fix is small and lives in the editable surface: on the stop-token round, keep
  producing the serial trajectory (or at minimum keep the pendings valid) instead of
  tearing down session state. `program.md` already prescribes the behaviour — "continues
  the serial trajectory for the full window even when EOS appears inside it".

**I did not fix it.** E14's contract is `quantized.h` and its generated twin;
`Qwen36MTPBlockSession.swift` is another agent's surface and a fix there would change the
schedule under every other in-flight arm. Filed as follow-up #7.

Q4 was therefore collected at **256 decode tokens** and is labelled a directional policy
screen, per `program.md`'s rule for short exact runs.

## Correctness

Bit-exact QMV parity, job `74ca3fb2-…` (exit 0, 884.7 s, `2026-08-17T11:12Z`–`11:27Z`),
via `research/run-qmv-parity.sh`. Each arm is a full rebuild from the same source commit
`ad553c1f…` plus its patch: `swift build -c release --build-tests`,
`tools/build-mlx-metallib.sh --all-build-roots`, then `QwenQMVParityTests`, which digests
the output of `quantized_matmul` over the 8 scored shapes × verify widths 1–12 = **96 cells**.
Outputs in `.mlxfast-private/qmv-parity/`.

| arm | twin `quantized.h` | cells differing | widths differing | verdict |
|---|---|---|---|---|
| `ref` (unpatched) | `b99146e9…` | — (reference) | — | reference |
| `armE` (`ipg-e`) | `a0de4f27…` | **0 / 96** | none | **bit-identical** |
| `armA` (`ipg-a`) | `16703cd6…` | **8 / 96** | **[5]** | **diverges** |
| `perturb` (positive control) | `6319e4fd…` | 56 / 96 | [3,4,5,6,7,8,9] | diverges |

**Both controls fired, so the verdicts are trustworthy.**

- *Positive control.* `perturb` multiplies `partial[r]` by `1.015625f` inside `_wide`. It
  changes exactly 56 cells = 7 widths × 8 shapes, and the 7 widths are exactly `[3..9]` —
  the wide-branch dispatch range. Widths 1, 2 and 10–12 are untouched because they never
  enter `_wide`. So the harness detects a one-instruction kernel change, and it detects it
  across precisely the region arms A/D/E modify. This also independently confirms, from
  output digests alone, which widths execute the wide branch.
- *Determinism control.* `armE` is a separate checkout, patch, Swift build, metallib build
  and test process, and its `armE.json` is **byte-identical** to `ref.json`
  (both sha256 `9e3c52a3df97856e…`). Build-to-build digest noise is therefore zero, so
  `armA`'s 8 differing cells are signal, not jitter.

**Arm A is not bit-exact, and that is a new and decisive finding.** Arm A's only functional
edit is the dispatch line `_m<T,5,3>` → `_m<T,5,5>` (the companion `NA_ASSERT` edit is a
`static_assert` and cannot change arithmetic). Yet every one of the 8 scored shapes returns
different bits at `M=5`. The reference serves `M=5` as two passes of `NA=3` then `NA=2`;
arm A serves it as one pass of `NA=5`, where `vec<float,5>` is padded to `vec<float,8>`.
Changing the accumulator's vector width changes the code the Metal compiler emits for the
`k` loop — FMA contraction and lane scheduling — and the last bits move with it.

**Arm E, which contains arm A's dispatch change, is bit-identical anyway.** That is the
control that pins the mechanism: arm E additionally replaces every `vec<float,NA>` with a
plain `float[NA]`. Elementwise arithmetic on a scalar array is lane-independent and rounds
identically at any `NA`, so removing the padded vector type puts the results back exactly
on the reference. The divergence is caused by the padded `vec<float,5>`, not by the row
partitioning.

This tightens the register-pressure diagnosis in the arm E section rather than competing
with it: the padded `vec<float,5>` is demonstrably driving *different codegen*, visible
here in the output bits and earlier in the ~32 points of `M=5` cost that arm E recovers.
The toolchain agrees — arm E's metallib build emits
`quantized.h:981:26: warning: unused typedef 'VF' [-Wunused-local-typedef]`, confirming
that no `vec<float,NA>` use survives in the wide body.

### Correctness verdict

- **Arm A — invalid as a candidate.** It changes QMV output bits on scored shapes, so it can
  flip a near-tie argmax at readout and perturb accept/reject decisions and the emitted
  token stream. It is *latent* rather than live: `M=5` is dispatched 0.00% of the time on
  base `e6e6f81`, so nothing today executes the changed cells. It would become a live
  fidelity risk the moment a scheduler makes depth 4 reachable. Any future use of this
  dispatch change must carry arm E's scalar packing with it to stay bit-exact.
- **Arm E — bit-exact, but not useful.** It passes every parity cell and still costs
  **+12.4%** median at every unchanged wide width, so it is rejected on performance.

The assignment stop rule was "stop if any arm fails bit-exact parity". It fired, on arm A,
and it is the reason no candidate advances from E14.

**Scope of the claim.** This is kernel-output parity on the 8 scored shapes at widths 1–12,
not an end-to-end token-stream match, and it was measured on M4 Pro. It bounds where a
change is visible; it does not by itself prove a full-generation match on the ranked M5.

## Freshness proof (before the first timing arm)

Recorded in `.mlxfast-private/ipg-arms/freshness-before.txt`, re-recorded post-rebuild at
`2026-08-17T10:03:29Z`:

- `research/rebuild.sh` (job `1eb5dac0-…`) exit 0, 128.3 s ->
  `.build/release/mlxfast-swift` sha256 `65b698ba19d37f11`,
  `.build-worker/release/mlxfast-runtime-worker` sha256 `b4f0f046f200e708`
  (the worker had been **stale since Aug 16 14:52**).
- **The metallibs were stale.** Vendored source fingerprint `6639cc59d6fb84ff…` did not
  match the on-disk sidecars `65d691ee4dd0090b…`.
- Fix job `ed89f537-…` (`tools/build-mlx-metallib.sh --all-build-roots`) exit 0, 43.9 s.
  All four build-root metallibs then reported `sha256=3849425dfed0737c` with
  sidecar == source fingerprint.
- Twin digests at the reference state:
  `quantized.h` `b99146e90bc13cb89847d89d889f5717b59194be0f4324507e056acaf2fe1245`;
  `quantized.cpp` `70b1af945d35e0e3b220c03ca8856a68828500c41889547db970f744d6eb2b73`.

**Every kernel result taken before that rebuild on this host was measuring a stale
metallib.** `run-ipg-arms.sh` now rebuilds the metallib per arm and records
`arm-state.json` + `arm-state-after.txt` so the binding is auditable.

**Audit of every timed arm above.** After the Q4 trace run was found executing a leftover
metallib, I re-derived which results could be affected. Per arm, `run-ipg-arms.sh` does
`restore_twins` -> `roofline_arm_patch.py <arm>` -> `run-qmv-curve.sh <tag>`, and
`run-qmv-curve.sh:104-107` does `swift build` then
`tools/build-mlx-metallib.sh --all-build-roots` **before** its first cool gate and before
any timed sweep. So each arm rebuilt from its own patched source immediately before
measuring, and a metallib inherited from an earlier job is always overwritten. Only a
wrapper with no rebuild of its own is exposed, which is exactly the case that failed.
The arm-specific effects are independent confirmation: a shared stale metallib cannot
produce +39% at `M = 5` for one arm and +12.4% at every wide width for another.

## Conclusion

- **Arm A is invalid, not merely slow.** It changes `quantized_matmul` output bits on 8 of
  96 scored cells, all at `M = 5`. A candidate that moves target-model bits can flip a
  near-tie argmax against the hidden serial stream, so it could never be submitted at any
  speed. Arm E carries the same dispatch change and is bit-identical over all 96 cells, so
  the divergence is caused by the padded `vec<float,5>` type and not by the row
  re-partitioning the arm was testing. Any future revival of this lever must carry arm E's
  scalar packing.
- **What happened and why:** the second weight pass is real and lands exactly where the
  dispatch table says it should, but it is ~89% absorbed by cache, so it is worth only ~0.11
  depth-0 rounds. The only way to remove it at `M=5` is to widen `NA` to 5, and `NA` width is
  the *dominant* term (~0.27 per row, and ~0.83 for the 3 -> 5 jump measured here). The
  mechanism is confirmed; the lever is a loser.
- **Evidence for or against the mechanism:** for — structural and interventional estimates of
  the pass cost agree within 4%; per-shape excess is monotone in weight footprint for the
  pass arm and flat for the width arm, which are two different and correctly-predicted
  fingerprints. Against the *speedup* — arm A is 39% slower at the width it was supposed to
  make cheap.
- **Prompt or M5 transfer risk:** high in magnitude, low in sign. M4 Pro and M5 differ in
  cache hierarchy and register file, so the 89% absorption fraction and the 2.4x ratio are
  M4-Pro numbers. The *ordering* (width tax > stream tax) rests on register/occupancy
  behaviour that is unlikely to invert.
- **The padding penalty is removable, and removing it does not help.** Arm E packs all
  thirteen `NA`-wide vectors and recovers ~32 of arm A's 39 points at `M = 5`, which confirms
  register pressure as the mechanism. It also makes every other wide width 12.4% slower — and
  Q4's live trace shows those other widths are `M = 6` and `M = 9`, i.e. 94.3% of real rounds
  on this base, just as they were `M = 3` and `M = 4` under PR #13's schedule. The fix costs
  more where it applies than it saves where it helps, under both schedulers measured.
- **`M = 5` is cold, and E14 therefore targeted the wrong width.** Q4 traced a live run and
  found `M = 5` dispatched in 1 of 35 rounds (2.86%, 1.97% cost-weighted), bounding the
  best-case end-to-end value of a *free* second pass at **0.12%** — well inside noise, and
  negative for every arm actually built. The same trace found `_m<9,3>`, the only three-pass
  entry in the table, taking **48.6%** of rounds. The redundant-pass mechanism is real and
  worth roughly twice as much at `M = 9`; it is simply not worth anything at `M = 5`. See
  follow-up 8.
- **Recommendation: close, and close firmly.** Do not repeat this lever, and do not reopen it
  on a toolchain change. The earlier draft of this line said to reopen if a future MLX or
  Metal toolchain removed the `vec<float,5>` padding penalty. Arm E shows that penalty can be
  removed in source today, so that reopening condition has already been tested and failed.
  The other reopening condition — a scheduler that dispatches `M = 5` at meaningful frequency
  — has now also been tested and failed, on two bases and three policies. Reopening requires a
  genuinely different reason: a wide-branch rewrite that reaches `NA = 5` occupancy without
  taxing `NA = 3` and `NA = 4`, or the `M = 9` retest in follow-up 8, which is a different
  width and should be assigned as a new experiment rather than a revival of this one.

### Suggested follow-ups (not implemented here)

1. **Host-side grid sizing** — `ntg.x == M` means the launch grid is sized for the widest
   `IPG` group and arm A only halves *active* threadgroups. A host-side change that sizes
   the grid to `ceil(M/IPG)` is a separate, larger question and is outside this
   assignment's scope. *Owner: unassigned.*
2. **`run-qmv-curve.sh:136` venv-python bug** — the default `MLXFAST_PYTHON_BIN` resolves to
   the venv `python3` symlink, which **exits silently under `run_job`**, leaving a 0-byte
   `summary.log` and no `summary.json`. Fixed locally for `run-ipg-arms.sh`; still latent
   for every other caller. *Owner: whoever next touches the curve tooling.*
3. **Stale-metallib hazard** — nothing in the normal build path rebuilds `mlx.metallib` when
   a `.metal`/`.h` source changes without a generated twin, and restoring the *sources* after
   an arm does not restore the *metallib*. This bit me: the first attempt at the Q4 trace run
   executed arm A's kernels because `run-qmv-parity.sh` had left arm A's metallib in
   every build root. The worker's own fingerprint warning is what caught it, on stderr, which
   nothing was capturing until this assignment made the trace path real. Both wrappers now
   rebuild, but the normal build path still does not, and the warning is still only a
   warning. Promoting that fingerprint mismatch to a hard failure in
   `benchmark-qwen-mtp.sh` would remove a whole class of silent wrong measurements.
   *Owner: unassigned.*
4. **edward anti-synergy** — see below; the scheduler owner should decide whether depth 4 is
   worth re-opening at all given the cost-per-token sweep above. *Owner: edward.*
5. **Control validity for shared-body kernel edits (methodology)** — `research/run-ipg-arms.sh`
   and `ipg_shape_breakdown.py` treat every unchanged width as a control and report a "drift
   adjustment" from their median. That is only sound for arms that edit a *dispatch-table
   entry*. An arm that edits the shared `_wide` body changes every instantiation, so the
   adjustment silently divides the signal by the intervention. The tooling should require the
   arm to declare which widths it touches and refuse to drift-adjust otherwise. I hit this
   here and it cost a full confirmation run. *Owner: whoever next touches the arm tooling.*
6. **A cheap way to detect it** — the `1 + 2^-6` perturbation arm used for parity doubles as a
   width map: it lives in `_wide`, so the set of widths whose output changes is exactly the
   set routed through the wide branch. Running it once per arm family would have flagged the
   shared-body scope before any timing was spent. *Owner: unassigned.*
7. **The 512-token EOS `notBegun` defect** — `Qwen36MTPBlockSession.swift:723-728` tears down
   its own pendings on a stop-token round; the trusted parent never reads `reachedStopToken`
   and asks for the next round anyway, which throws `.notBegun`. Full root cause and evidence
   in the section above. It kills the **serial control**, so it blocks every 512-token local
   measurement on the public fixture, and it sits one `if` away from the ranked candidate leg.
   Whoever owns the session should make the stop-token round continue the serial trajectory
   for the configured window. *Owner: the `Qwen36MTPBlockSession.swift` owner.*

   **Fixed on `b85e782` — this follow-up is closed.** The rebase base already carries the
   repair: `stopTokens` drops from **7 occurrences to 1** in that file, `reachedStopToken` is
   now a constant `false` (`:167`, and literal `false` at both `:817` and `:1127`), and the
   pre-drafting early return, the accept-loop `break`, and the post-hoc truncation are all
   gone, replaced by `acceptedDraftPrefixCount(drafts:verifyArgmax:)` (`:672`, called at
   `:967`) which accepts on argmax agreement alone. The session now runs the parent's
   configured window regardless of EOS. Worth recording *why* this is the right shape: the
   organizer's trusted driver never had this defect, so the fix is the candidate coming back
   into line with the ranked contract, not a deviation from it. I did not verify the repair by
   running the 512-token control — r2 is bookkeeping-only, and confirming it needs a GPU leg.

8. **Retest the weight-pass hypothesis at `M=9`, not `M=5` — but read E14's register result
   first.** Q4's live trace shows `_m<9,3>` taking **48.6%** of rounds and multi-pass `_m`
   dispatches taking **97.1%**, against **2.86%** for the `M=5` slot this assignment was
   scoped to. `M=9` is the only three-pass entry in the table, so it carries twice E14's tax
   at seventeen times the frequency, and it is the single largest redundant-weight-pass target
   on the scored path. I did not attempt it: it is outside E14's scope, and E14's own evidence
   predicts the obvious implementations fail. `_m<9,5>` hits the `vec<float,5>`→`vec<float,8>`
   padding trap that cost arm A **+39.2%** and bit-exactness; `_m<9,9>` pads `NA=9` to 16;
   and arm E's `float[NA]` packing, which is the only fix that removes the padding penalty
   *and* stays bit-identical, costs **+12.4%** at precisely the `NA=3`/`NA=4` widths that
   dominate here. The decisive cheap screen is therefore a **cost-curve sweep of `_m<9,IPG>`
   for `IPG ∈ {3,5,9}` with and without arm E packing**, judged on the isolated fixture at
   `M=9` *and* on the unchanged widths, before any end-to-end run. If no variant beats
   `_m<9,3>` at `M=9` without taxing `M=6`, the redundant-weight-pass family is closed for
   this kernel, not just for `M=5`. Note this follow-up inherits the same edward anti-synergy
   below. *Owner: a future kernel experiment.*

   **Restated for `b85e782`.** The base moved under this follow-up in two ways, both
   strengthening it. `M=8` is now `_m<8,3>`, so the three-pass slots are `M=8` **and** `M=9`,
   and re-indexing the measured histogram lifts the three-pass round share from **48.6%** to
   **57.1%**. `segmentedStreakGate` moved `3 -> 2`, which lets the ramp reach high `M` sooner,
   so the true share is likely higher still. The screen should therefore sweep `_m<8,IPG>`
   alongside `_m<9,IPG>` and re-derive the round histogram on the new base rather than reusing
   the numbers above. Two independent signals now point at the same place: my cost curve found
   `<T,8,4>` anomalous from the timing side, and the organizer's own retune comment cites
   319 / 437 / 216 µs for `M = 7 / 8 / 9` from the dispatch side.

9. **`swift test` does not build on this toolchain — one line, campaign-wide.** Found while
   validating the recovered suite at r2. `Tests/MLXFastTests/QwenMTPVerbTests.swift:755`
   passes a **concatenated** `String` where Swift Testing's `#expect` wants a `Comment?`:

   ```swift
   #expect(
       !paths.contains(absent),
       "the head tree does not ship \(absent); pinning it would fail "
           + "verify_cache's inventory half against the published tree")
   ```

   `Comment` is `ExpressibleByStringInterpolation`, so a single interpolated *literal*
   converts implicitly but a `+` expression does not. Joining the two fragments into one
   literal fixes it; `Comment(rawValue:)` also works. This is the **only** error in the whole
   test target — everything else is warnings — so that one line is what stands between the
   campaign and a runnable `swift test`.

   Scope and blast radius: the expression is unchanged at `e6e6f81`, `ef16dea4`, `d098212`
   and `b85e782`, introduced by `ee977ae Restore organizer test source snapshot`, so this has
   been broken for the whole window I can see, not just since the last sync. `Tests/` is not
   in `editablePaths`, so nothing here reaches a submission. It is toolchain-dependent —
   `swift-testing` ships with the toolchain rather than being pinned in `Package.resolved`,
   and this host is Apple Swift 6.3.3 — so a runner on an older toolchain may not see it;
   that does not make it safe to leave, since it silently disables the repository's own
   correctness gates for anyone who does. I did **not** fix it: r2 is bookkeeping-only, and
   the file is restored organizer test source that E14 does not own. *Owner: whoever owns the
   organizer test snapshot; it is a one-line change.*

### edward anti-synergy (stated explicitly)

edward's promoted schedule is tuned against the *current* `h` vector, in which `h4` is a
spike. This experiment set out to flatten that spike. If it had succeeded, **edward's
schedule constants would have been tuned against a cost curve that no longer existed**, and
the two changes would have had to be re-tuned together rather than composed. That
anti-synergy is now moot because arm A loses, but the general rule stands: any change to the
QMV dispatch table invalidates the `h` vector that the scheduler was fitted to, so the two
must always land as one re-tuned pair, never as independent merges.

**`b85e782` is a live instance of that rule.** The `<T,8,4>` -> `<T,8,3>` retune changed the
`h` vector at the top end: `h4` is untouched, but the second pass increment moved from `h8`
to `h7`, so any scheduler constant fitted against the old high-`M` shape is now fitted
against a curve that no longer exists. `segmentedStreakGate` moving `3 -> 2` on the same base
is consistent with a re-tune having been done, but I did not verify that the two landed
together, and E14 owns neither file. Flagging it for whoever owns the scheduler: the pair
rule applies to the organizer's own dispatch edits, not only to ours.
