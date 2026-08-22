# E138: sweep the plan surface and find what flattens the width-6 cliff

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e138_best_plan_isolated_step_reduction_pct","available":true,"value":2.4593},"test_metric":{"name":"e138_cliff_is_plan_invariant","available":true,"value":1}}

- **Student / branch:** `qwen-alphonse` / `qwen-alphonse/e138-plan-surface-at-the-width-6-cliff`
- **Hypothesis and target cost:** E137 measured a 39,134.9 us in-situ M=5 -> M=6 verify step and found no single dispatch family owns it. E138 asks whether the step is an artefact of the *shipped plan choice*. If some other legal `(M, IPG, RPS)` cell were cheaper at width 6, the cliff would be a tuning defect rather than a hardware boundary.
- **Decision:** **negative on the stated hypothesis, positive as a plan-surface map.** The cliff is plan-invariant: no legal cell flattens the 5 -> 6 step, and the best available reduction is **2.46 %** of the isolated step. The sweep did find one clean, unrelated win one width up, and it found that the plan surface is exhausted exactly where the ranked mass sits.
- **`BASE_SHA`:** `328c4b9eac1b386f0c0913afcf0c7a64c232e5c0`
- **`UPSTREAM_SHA`:** unchanged this experiment; no organizer sync performed.
- **Candidate commit:** `645dedf2` (see the terminal submission for the exact head).
- **Yukon promoted submission / frontier:** live crown `08b67f1` (`jungjipdo`) at `3.69071882618532`, previous `ed608e6` at `3.68172016051458`. Checked at the end of this experiment. Neither is affected by this work, and **this experiment proposes no submission.**
- **Candidate build fingerprint:** not applicable. No worker was built and no decode leg was run. The instrument is a Swift test that dispatches Metal kernels directly.
- **Submitted-surface / generated-twin / metallib digests:** not applicable. **No submitted path changed.** The complete non-research diff against base is one new test file, `Tests/MLXFastTests/E138PlanSurfaceTests.swift`. `python3 research/twin_audit.py` -> `TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)`. No Metal source and no generated twin was touched.
- **Submitted candidate files:** **none.** This is a measurement experiment.
- **Supporting test, tooling, and evidence files:**
  - `Tests/MLXFastTests/E138PlanSurfaceTests.swift` (new instrument)
  - `research/e138_plan_census.py`, `research/e138_plan_surface.sh` (new)
  - `research/e138_plan_analysis.py`, `research/e138_grid_control.py` (new)
  - `research/e138_wandb_log.py` (new)
  - `research/e131_kernel_sources.py` (reused, minor repair)
  - `research/e138-artifacts/*.json` (evidence, 11 timing sessions plus the offline census)
- **MTP head provenance, digest, and draft policy:** not applicable. The sweep loads no checkpoint and attaches no proposal head. It builds synthetic operands at the seven scored shapes and times the quantized matvec directly.
- **Token window, fixture, reference source, harness:** not applicable in the decode sense. This is an **isolated kernel sweep**, not an end-to-end decode. `harness = local` throughout. **No ranked measurement was taken by me**, and no number in this report is a score.
- **Exact cell:** the 257 linear cells of one target verify forward - `linear_attn.in_proj_fused_qkvzba` x48, `linear_attn.out_proj` x48, `full_attn.qkv_proj_fused` x16, `full_attn.o_proj` x16, `mlp.gate_up_fused` x64, `mlp.down` x64, `head.lm_head` x1 - at widths M = 5..9. Dispatch family: affine 4-bit group-64 quantized matvec via `Qwen35CustomQMV.generatedSource(table:true, tier:nil)`. Source form: JIT from the generator string. M5 `_nax` variant: **not measured**, see transfer risk.
- **Official causal path and score equation:** `harness=ranked`. This experiment measures no candidate edit and therefore changes no ranked term. `senpai/verify-ranked-score-boundary.sh` -> `PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only`. No local cancellation term appears in any statement below.
- **Assignment-scope preflight:** `senpai/entry-point-cliff-census.sh --base 328c4b9e` -> `verdict: PASS`; `senpai/verify-ranked-score-boundary.sh` -> `PASS`.
- **Editable source bytes / headroom / growth / exempt-head bytes:** `source=2613813/3000000`, `headroom=386187`, `growth=158978/262144`, `exempt=2410/2147483648`, `files=154` against growth base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
- **Scored-path reachability evidence:** the seven shapes and their 48/48/16/16/64/64/1 dispatch counts are the cells E137's live 512-token routing census proved are dispatched at every routed width. This experiment reuses that census rather than repeating it.

---

## Evidence

- **Host, chip, memory, toolchain, thermal policy:** `ip-10-231-2-22.ec2.internal`, Apple M4 Pro (`applegpu_g16s`, 20-core, 48 GiB), `memory_bytes=51539607552`, macOS 26.5.2, Apple Swift 6.3.3, Apple metal version 32023.883. `MLXFAST_LOCAL_COOL_GATE=0` on every timing session: `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, `official_or_ranked_score=false`, preserved verbatim in each `.session.json` sidecar. **No result in this report is a gate-qualified timing measurement or a score.**

- **Entry and exit GPU temperature, every session:**

  | session | entry degC | exit degC |
  |---|---|---|
  | `item1-decisive-cells` | 37.85 | 58.49 |
  | `item1a-m5-m7-rows` | 37.86 | 62.62 |
  | `item1c-spill-control-and-stock` | 38.61 | 64.45 |
  | `item1d-grid-control-tight` | 38.70 | 53.65 |
  | `item1d-grid-control-wide` | 39.27 | 54.70 |
  | `item5-m8-m9-and-m7-replicate` | 38.88 | 63.95 |
  | `item6-factorial-m6-wide` | 38.09 | 59.02 |
  | `item6-factorial-m6-tight` | 40.54 | 58.80 |
  | `item7-interleaved-factorial-rep1` | 38.16 | 64.94 |
  | `item7-interleaved-factorial-rep2` | 42.55 | 65.01 |
  | `item7-interleaved-factorial-rep3` | 45.18 | 66.50 |

  Entry spread across the three pooled headline sessions (`item1`, `item1a`, `item5`) is **1.03 degC**. Across the three factorial replicates it is **7.02 degC**; those three are interleaved within each session, so the drift cancels to first order inside each replicate, and the residual is reported as replicate spread rather than hidden.

- **Exact commands:**

  ```bash
  # offline register/spill census of the whole legal plan surface
  python3 research/e138_plan_census.py

  # one timing session: OUT CELLS [SHAPES] [REPS] [INNER] [GRID] [REFERENCE]
  research/e138_plan_surface.sh research/e138-artifacts/item1-decisive-cells.json \
      "6:6:4,6:3:4,6:2:4,5:5:4,6:stock" "" 31 12 tight 6:6:4

  # the interleaved factorial, both grids in one block
  research/e138_plan_surface.sh research/e138-artifacts/item7-interleaved-factorial-rep1.json \
      "5:5:4@wide,5:5:4@tight,6:1:4@wide,6:1:4@tight,6:2:4@wide,6:2:4@tight,\
6:3:4@wide,6:3:4@tight,6:4:4@wide,6:4:4@tight,6:6:4@wide,6:6:4@tight" "" 31 12 tight 6:stock

  # analysis
  python3 research/e138_plan_analysis.py research/e138-artifacts/item1-decisive-cells.json \
      research/e138-artifacts/item1a-m5-m7-rows.json \
      research/e138-artifacts/item5-m8-m9-and-m7-replicate.json
  python3 research/e138_grid_control.py research/e138-artifacts/item7-interleaved-factorial-rep{1,2,3}.json
  ```

- **W&B runs:**

  | rung | contents | run |
  |---|---|---|
  | a | offline plan/register census | [`lea7ukkw`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/lea7ukkw) |
  | b | pooled widths 5-9, primary metric, RULE 116 ladder | [`mm4p2t99`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/mm4p2t99) |
  | c | M=7 spill dose-response and the stock reference arm | [`yd64ass7`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/yd64ass7) |
  | d | interleaved plan x grid factorial and its null control | [`4pu920le`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/4pu920le) |

  Group `e138-plan-surface-at-the-width-6-cliff`, project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`.

- **Exactness:** every timed cell is compared bitwise against the incumbent kernel on the same operands. **161 rows in the pooled headline set, `matches_incumbent_bitwise = true` on all of them, `max_abs_delta_vs_incumbent = 0`.** Every shape also carries a positive control that perturbs one operand and must fail the comparison; `exactness_positive_control_rejects` is true everywhere, and the analysis refuses to load an artifact where any positive control passed.

---

## Item 1 - the plan surface at each width

Isolated dispatch-weighted microseconds of one verify forward. Pooled across `item1`, `item1a`, `item5` by median over replicates, all anchored on `6:6:4`, `REPS=31 INNER=12`.

| width | shipped cell | shipped us | best single plan | shape-keyed us |
|---|---|---|---|---|
| 5 | `5:5:4` | 104184.0 | `5:5:4` (shipped) | 104184.0 |
| 6 | `6:6:4` | 131861.2 | `6:6:4` (shipped) | 131180.6 |
| 7 | `7:7:4` | 155620.8 | **`7:4:4`, 149704.4** | 149565.6 |
| 8 | `8:4:4` | 155910.2 | `8:4:4` (shipped) | 155910.2 |
| 9 | `9:3:4` | 191429.2 | `9:3:4` (shipped) | 182962.3 |

The shipped cell is the single best plan at four of the five widths. At width 8 it is optimal on **every one of the seven shapes** individually, so no shape-keyed table can improve it either.

**Primary metric.** The isolated M=5 -> M=6 step under the shipped plan is 27,677.3 us. Under the best shape-keyed table it is 26,996.6 us. That is a reduction of **2.4593 %** of the step. `harness=local`, isolated, and explicitly not a ranked figure.

**Test metric.** `e138_cliff_is_plan_invariant = 1`. No legal cell in the 120-cell surface removes the step; the best case shaves 2.46 % off it.

## Item 2 - the one real win, and its mechanism

`(7,7,4) -> (7,4,4)` saves 5916.4 isolated us per round at width 7, with **one** pipeline and no table. Per shape:

| shape | calls | us/call | us/round |
|---|---|---|---|
| `mlp.down` | 64 | 87.104 | 5574.7 |
| `full_attn.qkv_proj_fused` | 16 | 19.212 | 307.4 |
| `mlp.gate_up_fused` | 64 | 2.417 | 154.7 |
| `linear_attn.in_proj_fused_qkvzba` | 48 | 0.385 | 18.5 |
| `full_attn.o_proj` | 16 | −1.861 | −29.8 |
| `head.lm_head` | 1 | −40.403 | −40.4 |
| `linear_attn.out_proj` | 48 | −1.429 | −68.6 |
| **total** | | | **5916.4** |

94 % of the saving is one shape. The offline census explains it: at `RPS=4`, `7:7:4` uses 96 registers and **spills 32 B** on g16s, while `7:4:4` uses 92 registers and spills nothing. Spill cost scales with K, and `mlp.down` has the largest K in the model at 17,408. The effect reproduced in two independent sessions at −3.65 % and −3.93 % on that shape.

Register census at `RPS=4` (g16s reg/spill | g17s reg/spill): `5:5:4` 96/0 | 98/0; `6:6:4` 96/0 | 105/0; `6:3:4` 91/0 | 94/0; **`7:7:4` 96/32 | 118/0**; `7:4:4` 92/0 | 96/0; `8:8:4` 96/80 | 126/16; `8:4:4` 92/0 | 96/0; `9:9:4` 96/144 | 126/64; `9:3:4` 91/0 | 96/0.

Of 120 legal cells built offline, 31 spill on g16s and 22 on g17s, and **9 disagree between the two GPUs**. Per RULE 83 that closure is void as g17s evidence: `7:7:4` spills on my host but not on g17s, so the mechanism itself may not transfer.

## Item 3 - N-separation at width 9

Width 9 is the only width where a shape-keyed table beats every single plan, and the split is clean along N:

- N = 5120 shapes (`out_proj`, `o_proj`, `mlp.down`) prefer `9:3:4`, which makes 3 passes.
- N >= 14336 shapes (`in_proj`, `qkv_proj`, `gate_up`, `lm_head`) prefer `9:6:4`, which makes 2 passes, by 4.6-8.8 %.

Total 8467.0 isolated us. This supports the advisor's N-keyed table hypothesis. It is currently unpayable: width 9 carries zero mass on both scoring prompts.

## Item 4 - the plan x launch-grid factorial, and its null control

The first attempt (`item6`) ran the two grids in separate sessions and was invalid. An identical `6:6:4` cell measured in two otherwise-matched sessions moved by up to **10.40 %**, so any across-session interaction estimate reads session drift. The instrument was rebuilt with a per-cell `@wide`/`@tight` suffix so both arms sit in one interleaved session anchored on the grid-independent `6:stock` cell, sharing one pipeline.

Three replicates. Global best plan is `6:6:4` under **both** grids.

The estimator carries two null controls that fall out of the launch rule at `Qwen35.swift:1957-1961`, where launched columns are `640 * M` wide and `640 * ceil(M/IPG)` tight:

1. **Exact.** At `IPG=1`, `ceil(M/1) == M`, so the tight column count collapses onto the wide one. `6:1:4@wide` and `6:1:4@tight` dispatch the identical grid over the identical pipeline; both rows record `launched_columns: 6` and the same `threadgroups_per_column`. True interaction is exactly 0.00 pp. Measured: −1.82 to **−23.18** pp across the seven shapes.
2. **Matched pair.** `ceil(6/3) == ceil(6/4) == 2`, so `6:3:4` and `6:4:4` share a grid pair on every shape and must share an interaction. Measured difference: +0.05 to +8.21 pp.

```
measured interaction resolution floor:                    23.18 pp
largest interaction where the grid really differs:         7.62 pp
interaction resolved above the null control:              NO
```

**No interaction on any shippable plan is resolved.** The plan axis and the grid axis are separable on this evidence, and the advisor's proposed plan x grid mechanism for the `0b2f0014` sign reversal is not supported.

The three replicates agree to **0.02 %** on `head.lm_head` (4856.1, 4856.7, 4857.1 us) despite a 7.02 degC entry-temperature spread, which is the evidence that anchor normalisation is doing its job and that replicate 3 is not an outlier.

## Item 5 - RULE 116 reporting: where the saving lands

Per CAMPAIGN RULE 116 and FINDING 196, the result is reported as absolute microseconds per round at each width, with the dispatch count applied. `in-situ` applies the E137 factor 0.7858; `ranked-host` also applies 0.65.

| width | shipped | arm | plans | isolated us | in-situ us | ranked-host us |
|---|---|---|---|---|---|---|
| 5 | `5:5:4` | best global | `5:5:4` | 0.0 | 0.0 | 0.0 |
| 5 | | shape keyed | `5:5:4` | 0.0 | 0.0 | 0.0 |
| 6 | `6:6:4` | best global | `6:6:4` | 0.0 | 0.0 | 0.0 |
| 6 | | shape keyed | `6:3:4`,`6:6:4` | 680.7 | 534.8 | 347.6 |
| 7 | `7:7:4` | best global | `7:4:4` | 5916.4 | 4648.9 | 3021.8 |
| 7 | | shape keyed | `7:4:4`,`7:7:4` | 6055.2 | 4758.0 | 3092.7 |
| 8 | `8:4:4` | best global | `8:4:4` | 0.0 | 0.0 | 0.0 |
| 8 | | shape keyed | `8:4:4` | 0.0 | 0.0 | 0.0 |
| 9 | `9:3:4` | best global | `9:3:4` | 0.0 | 0.0 | 0.0 |
| 9 | | shape keyed | `9:3:4`,`9:6:4` | 8467.0 | 6653.0 | 4324.5 |

Rebuilt for the only two prompts that can move the published median, using their own width masses:

| prompt | m5 | m6 | m7 | m8 | best global | shape keyed |
|---|---|---|---|---|---|---|
| beagle | 0.0920 | 0.0739 | 0.0589 | 0.3388 | 273.8 us | 319.8 us |
| essays | 0.1995 | 0.1558 | 0.1408 | 0.2958 | 654.6 us | 753.2 us |

**No headline ranked percentage is formed here, and the instrument can no longer produce one.** `RANKED_ROUND_US` and the withdrawn F83 `RANKED_WIDTH_MASS` table are deleted from `research/e138_plan_analysis.py`. Rebuilding the median needs all eight prompts' width masses and each prompt's own baseline round time; I have two prompts' masses and no baseline round times.

**The strategic finding.** The gain is anti-correlated with the ranked mass:

| prompt | mass at widths that pay (6,7) | mass at widths that pay nothing (5,8) | share paying |
|---|---|---|---|
| beagle | 0.1328 | 0.4308 | 23.6 % |
| essays | 0.2966 | 0.4953 | 37.5 % |

Width 8 is the largest single width mass on both scoring prompts, and the plan surface yields exactly zero there. Width 8 alone is 60.1 % of beagle's covered mass. Beagle pays 0.478 per unit across 9.10 % of headroom and is concentrated where nothing is available; essays, which saturates after 1.37 %, is where the gain lands.

## Conclusion

The width-6 cliff is **not** a plan-tuning defect. It survives the whole legal `(M, IPG, RPS)` surface, and the best any plan choice can do is remove 2.46 % of the step. E137's conclusion stands unchanged: the step is a property of the dispatch, not of the schedule that reaches it.

The sweep produced three things worth keeping:

1. `(7,7,4) -> (7,4,4)`, a one-pipeline win worth 5916.4 isolated / 4648.9 in-situ / 3021.8 ranked-host us per round at width 7, mechanistically explained by a 32 B register spill on the largest-K shape, and carrying a RULE 83 transfer risk because the spill does not occur on g17s.
2. An N-keyed split at width 9, currently unpayable.
3. A negative that closes a line of enquiry: the plan x launch-grid interaction is below a 23.18 pp instrument floor, so it cannot explain the `0b2f0014` sign reversal.

The most useful output may be the fourth thing, which is not a gain at all: **the plan surface is exhausted at width 8, which is where the ranked mass actually is.**

## Risks and honest limits

- **Isolated, not in-situ.** Every microsecond here is a kernel-level measurement multiplied by a dispatch count. The 0.7858 factor comes from E137 and is applied where stated, but no end-to-end decode was run in this experiment.
- **Ungated.** No timing here is gate-qualified. Entry temperatures are recorded and the headline sessions span 1.03 degC.
- **g16s, not g17s.** The `7:4:4` mechanism depends on a spill that the offline census says does not occur on g17s. Nine of 120 cells disagree between the two GPUs. This needs an M5 replay before anyone ships it.
- **`_nax` not measured.** The ranked M5 runs `_nax` variants. This sweep measured the non-`_nax` generator path.
- **Session noise is large on four shapes.** `mlp.down`, `qkv_proj_fused`, `in_proj` and `o_proj` show 6-17 % replicate spread. The two per-shape plan-order changes in the factorial both sit inside that spread.
- **Driver dirty on replicate 3.** `item7-interleaved-factorial-rep3` recorded `measured_source_dirty=true` for `research/e138_plan_surface.sh`. The edit was confined to the post-measurement sidecar writer and cannot affect a recorded time; the three replicates agree to 0.02 % on the most stable shape. Replicates 1 and 2 were dirty only in offline analysis files, which the flag now classifies separately.
- **Full test suite.** `swift test --force-resolved-versions` reports 41 issues across 10 tests, all pre-existing on this checkout and unrelated to this work: campaign documentation blocks, the artifact contract digest, the 128 GiB startup memory profile on a 48 GiB host, and the remote MTP head declaration. **`E138PlanSurfaceTests` passed.** My only non-research change is that new test file.

## Suggested follow-ups, not implemented

1. **Attack width 8 with something other than plan selection.** It carries the largest mass on both scoring prompts and the plan surface is provably exhausted there: shipped `8:4:4` wins on all seven shapes. This is the highest-value open question in this area.
2. **Replay `(7,4,4)` on g17s.** The mechanism is a g16s spill that the census says is absent on g17s. Either it transfers, in which case it is free, or it does not, in which case the census has told us something important about RULE 83.
3. **Give me each prompt's baseline round time** and I will finish the FINDING 196 rebuild properly, sort the eight ratios, and name the two central prompts instead of reporting microseconds.
4. **Tighten the interaction floor** if the plan x grid question is worth reopening. The 23.18 pp floor is dominated by `qkv_proj_fused`, which has the lowest dispatch count of the noisy shapes. More replicates or a lower-variance tap would help; a different analysis would not.
