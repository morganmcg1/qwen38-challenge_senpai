# E137: attribute the M=5 → M=6 verify cost step to a dispatch family

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e137_isolated_to_insitu_transfer_at_boundary_4","available":true,"value":0.7858},"test_metric":{"name":"e137_cliff_gate_failing_polarity_demonstrated","available":true,"value":1}}

- **Student / branch:** `qwen-alphonse` / `qwen-alphonse/e137-cliff-family-attribution`
- **Hypothesis and target cost:** the M=5 → M=6 verify step (39,134.9 us per round locally, 36.7 % of the round at M=5) is carried by one identifiable dispatch family. Finding that family would give the campaign a single kernel to attack.
- **Decision:** **green as an attribution, negative as a lever.** The step is real, it transfers to the ranked host, and it is 79–91 % inside the 4-bit affine quantized matvec. But no single family owns it: all seven QMV shapes step by 24.9–47.5 %, roughly in proportion to their existing cost.
- **`BASE_SHA`:** `33ce6a3f478043d168dc74e7322e754b3021d620`
- **`UPSTREAM_SHA`:** unchanged this experiment; no organizer sync performed.
- **Candidate commit:** `4feef53d` (see the terminal submission for the exact head).
- **Yukon promoted submission / frontier:** promoted `623e77af` at 3.52085227; crown `02742bf0` at 3.52686512. Neither moved during this experiment, and this experiment does not propose a submission.
- **Candidate build fingerprint:** `worker_sha256=e3bfdf904fc9eeed1a078f7dd55011fa14d3634e641ce304e49aeff11e47b3c9`, `cli_sha256=693c7e13d628a7492546a0e370b4f5a036b740ec56610dc37a8f198ac5d8debf`, `metallib_source_fingerprint=2050ebf1c1cf091ebbf35fceb7e9c1a9b399b7ed371b133d10b8546734efe7a7` (census leg `e137pipe512`).
- **Submitted-surface / generated-twin / metallib digests:** not applicable. **No submitted path changed.** `senpai/validate-assignment-scope.sh 33ce6a3f… <89 editable paths>` returns `assignment scope OK`. No Metal source and no generated twin was touched, so `research/twin_audit.py` is not in scope for this result.
- **Submitted candidate files:** **none.** This is a measurement experiment.
- **Supporting test, tooling, or documentation files:**
  - `Tests/MLXFastTests/E137RouteBCostCurveTests.swift` (new)
  - `research/e137_routeb_curve.sh`, `research/e137_cost_curve.sh` (new)
  - `research/e137_width_table.py`, `research/e137_cliff_attribution.py` (new)
  - `research/e137_pipeline_census.sh`, `research/e137_pipeline_census.py` (new)
  - `research/e137_gate_polarity.py` (new)
  - `research/e137_wandb_log.py` (new)
  - `research/e131_cliff_gate.py`, `research/e131_kernel_sources.py` (item 0 repair)
  - `research/e137-artifacts/*.json` (evidence)
- **MTP head provenance, digest, and draft policy:**
  - census leg `e137pipe512`: `head_provenance_sha256 = dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`, `head_dir = …/mtp-head-declared-run`
  - item 1 source legs (E130 rung 11): `head_provenance_sha256 = 62516c6f3799b66c91171ee13aa6816db5af197aa8c527cec0f6bb4026f0c7b7`
  - draft policy: setup default ladder, unchanged by this experiment.
- **Token window, fixture, reference source, harness:** 512 decode tokens, the single public `--local-iterate` fixture, candidate-generated reference rows, `harness = local` throughout. **No ranked measurement was taken by me.** Every ranked number quoted below is the advisor's F2 refit of the `623e77af` / `d3c491b5` receipt pair and is labelled as quoted.
- **Exact cell:** the 257 linear cells of one target verify forward — `linear_attn.in_proj_fused_qkvzba` ×48, `linear_attn.out_proj` ×48, `full_attn.qkv_proj_fused` ×16, `full_attn.o_proj` ×16, `mlp.gate_up_fused` ×64, `mlp.down` ×64, `head.lm_head` ×1 — at widths M = 3…9. Dispatch family: affine 4-bit group-64 quantized matvec, Route B (`qmv_sums_na*_v2`, `qmv_wide_na3_v2`) and the incumbent MLX gate. Source form: JIT from `mlx-generated/*.cpp`. M5 variant: `_nax` — **not measured here**, see transfer risk.
- **Official causal path and score equation:** `harness=ranked`. This experiment measures no candidate edit and therefore changes no ranked term. `senpai/verify-ranked-score-boundary.sh` → `PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only`. No local cancellation term appears in any ranked statement below.
- **Assignment-scope preflight:** `senpai/validate-assignment-scope.sh` OK (89 paths); `senpai/verify-ranked-score-boundary.sh` PASS.
- **Editable source bytes / headroom / growth / exempt-head bytes:** `source=2613813/3000000`, `headroom=386187`, `growth=158978/262144`, `exempt=2410/2147483648`, `files=154` against growth base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
- **Scored-path reachability evidence:** the census leg proves from a live 512-token decode that all 257 cells are dispatched at every routed width. See item 2.

---

## Evidence

- **Host, chip, memory, toolchain, thermal policy:** `ip-10-231-2-22.ec2.internal`, Apple M4 Pro (`applegpu_g16s`, 20-core, 48 GiB), `memory_bytes=51539607552`, Apple metal version 32023.883 (metalfe-32023.883). `MLXFAST_LOCAL_COOL_GATE=0` on the census leg: `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, entry 37.56 °C, exit 61.49 °C. **No result in this report is a gate-qualified timing measurement or a score.**
- **Exact commands:**
  - item 0: `senpai/entry-point-cliff-census.sh --json`; `python3 research/e137_gate_polarity.py`
  - item 1: `python3 research/e137_width_table.py`
  - item 2R: `research/e137_routeb_curve.sh` (env `MLXFAST_RUN_E137_ROUTEB_CURVE=1`, `MLXFAST_E137_ROUTEB_CURVE_OUT=…`, `_REPS=41`, `_INNER=15`)
  - census: `research/e137_pipeline_census.sh e137pipe512 512`; `python3 research/e137_pipeline_census.py --tag e137pipe512`
  - attribution: `python3 research/e137_cliff_attribution.py`
  - W&B: `python3 research/e137_wandb_log.py --rung {0,1,2,3,4}`
- **Cheapest real falsification gate and positive-control verdict:** item 0. The repaired entry-point cliff census fails closed, and three deliberately damaged polarities each make it fail: `unchanged` → pass, `blind` → fail, `residency` → fail. `e137_cliff_gate_failing_polarity_demonstrated = true`. A gate that cannot fail proves nothing, so this is reported before any measurement that depends on it.
- **Tests and risk-based checks, in execution order:**
  1. `senpai/verify-ranked-score-boundary.sh` → PASS
  2. `senpai/check-editable-budget.sh` → OK
  3. gate polarity control (item 0) → 3/3 agree with expectation
  4. Route B bitwise-equality check against the fallback at every measured cell → max absolute delta **0**
  5. `senpai/validate-assignment-scope.sh` over all 89 editable paths → OK
- **Exact-token and row-ledger verdict:** not applicable and not claimed. This experiment changes no submitted code, so there is no candidate token stream to check. The census leg ran the unmodified base and completed with `exit=0`, 78 rounds, 513 tokens.
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not relevant; no Metal source or twin changed.
- **Peak RAM / head size:** not measured; no memory-relevant change.
- **Official status and score:** not submitted. No official run.

### W&B runs

| rung | what | run id | URL |
|---|---|---|---|
| 0 | cliff-gate repair and polarity control | `r8w8145h` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/r8w8145h |
| 1 | width-keyed round parts (FINDING 187) | `ij0s6pxx` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ij0s6pxx |
| 2 | Route B coverage census | `vn6rw1ft` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/vn6rw1ft |
| 3 | Route B cost curve and attribution | `fdeo0a6a` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/fdeo0a6a |
| 4 | dispatch bracket and two falsifications | `2v96a2w2` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2v96a2w2 |

### Assignment metrics

| metric | value | 95 % CI |
|---|--:|---|
| `e137_non_qmv_share_of_m5_to_m6_step` | **0.2142** | [0.1885, 0.2809] |
| `e137_isolated_to_insitu_transfer_at_boundary_4` | **0.7858** | [0.7191, 0.8115] |
| `e137_sdpa_dispatch_step_us_bracket_low` | **8.39 us** | — |
| `e137_sdpa_dispatch_step_us_bracket_high` | **46.65 us** | — |
| `e137_cliff_gate_failing_polarity_demonstrated` | **true** | — |

### Standard metric table

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | not measured | not measured | — |
| MTP seconds/token | not measured | not measured | — |
| local serial-relative speedup | not measured | not measured | — |
| effective mean draft length | not measured | not measured | — |
| accepted draft rate | not measured | not measured | — |

**No candidate was built and no speedup was measured.** E137 is an attribution experiment. Filling these rows would require inventing a comparison that was never run.

### Identity-field match statement

Item 1 (E130 legs, base `cbf87ee8`) and item 2R (isolated curve on base `33ce6a3f`) are **not** the same base. I use item 1 only for the in-situ step denominator and item 2R for the isolated numerator, and I state that mismatch rather than hide it. The E92 ledger (item 3, base `b5cff751`) is a third base and is used only for a qualitative cliff-position argument, never pooled. All three share host, chip, memory profile, token window and harness. Every ranked figure is **quoted from F2**, is not my measurement, and is labelled at each use.

---

## Item 0 — cliff-gate repair (accepted in full by F2)

The gate now fails closed. Per-width g17s registers and derived simdgroups: M=3 `na3` 94/42; M=4 and M=8 `sums_na4` 96/41; M=5 `sums_na5` 98/40; M=6 `sums_na6` 105/37; M=7 `sums_na7` 118/33; M=9 `sums_na3` 94/42. Verdict on the current base: **pass**, 0 failures, 1 warning, 12 cells, 5.18 s.

Rule 89 applies: simdgroups is a model output computed from the register count, not a measurement.

## Item 1 — where in the round the step lands, and the retraction

**FINDING 187 is the headline, and it corrects my own first answer.**

12 E130 rung-11 legs, 924 rounds, 512 tokens per leg, Apple M4 Pro, leg base `cbf87ee8`. Width histogram `{3:12, 4:24, 5:72, 6:60, 7:84, 8:672}`.

| M | round us (mean) | step from M−1 | step ÷ round(M−1) |
|--:|--:|--:|--:|
| 3 | 81,548 | — | — |
| 4 | 91,717 | 10,169 | 12.5 % |
| 5 | 106,606 | 14,889 | 16.2 % |
| 6 | 145,741 | **39,135** | **36.7 %** |
| 7 | 158,440 | 12,699 | 8.7 % |
| 8 | 171,809 | 13,369 | 8.4 % |

**Retraction.** My first headline claimed 43 % of the step was non-QMV, derived from a 13-segment host-side split. That was wrong and I withdrew it. `host_thread_cpu_ns` shows the host thread runs for only 4.9–9.2 % of a round, and the host CPU step at 5→6 is **152.3 us — 0.39 % of the 39,135 us step**. The step is **99.6 % GPU**. A host-side segment table cannot attribute a GPU step, so the 13-segment split is retired for width attribution. Stop rule 4 fired, not stop rule 1.

**Local versus ranked, normalised by the round at the lower width** (ranked column quoted from F2, Apple M5 `applegpu_g17s`, not measured by me):

| step | local (M4 Pro) | ranked (M5, quoted) | gap |
|---|--:|--:|--:|
| 3→4 | 12.5 % | 9.1 % | +3.4 pp |
| 4→5 | 16.2 % | 8.3 % | +7.9 pp |
| **5→6** | **36.7 %** | **36.1 %** | **+0.6 pp** |
| 6→7 | 8.7 % | 2.7 % | +6.0 pp |
| 7→8 | 8.4 % | 11.9 % | −3.5 pp |

Every shallow step disagrees between hosts by 3.4–7.9 pp. The cliff agrees to 0.6 pp. **The cliff is a property of the model at verify width 6, not a g16s register artefact.** This is the single most transfer-relevant fact in the experiment.

## Item 2 — the routing census, and what the instrument cannot do

F2 set a stop condition: post immediately if width 6 shows fewer than 257 routed dispatches. **Width 6 shows exactly 257, identical to every other fully warmed routed width. The stop condition did not fire.**

But the number is not a scored-window count, and I say so rather than let it read as one. `notePipeline` (`Qwen35.swift:1999-2013`) increments `pipelineWidths[width]` on every routed dispatch, so the in-memory counter is a true per-dispatch count. The **file** is not: `flushPipelineLog` runs only when a key or width is seen for the first time, plus once from the `atexit` handler at `Qwen35.swift:1977`. `QwenRuntimeWorker.swift:1905` sends the worker SIGTERM and `:1911` escalates to SIGKILL, and neither runs `atexit`.

The arithmetic confirms the freeze exactly:

- snapshot ordinal = **1543**
- warm-up-only prediction = 257 × 6 + 1 = **1543** — match
- 78 scored rounds would have added at least 20,000 further dispatches

**Every scored dispatch increments a counter that is never written to disk.** A width-6 shortfall inside the scored window would be invisible here, so I did not raise the alarm.

**What the leg does prove** — the warm-up gate, which passes:

- first-dispatch ordinals `0 / 257 / 514 / 771 / 1028 / 1285 / 1542`
- gaps `[257, 257, 257, 257, 257, 257]`, a perfect arithmetic progression of one target forward
- every routed width 3…9, including 6, first reached during warm-up, before any scored token
- `no_pipeline_first_compiled_in_timed_window = true`

This independently confirms the 257-cells-per-forward weighting used in item 2R, from a live leg rather than from counting layers.

Two incidental facts: the sum-table route costs **two** dispatches per cell (`xsums_v1` = 1286 = 257 × 5 + 1, matching widths 4…9; width 3 uses `qmv_wide_na3_v2` and needs no table), and widths 1 and 2 do not route at all. The xsums cost is uniform across the 5→6 boundary, so it does not create the step.

**Tooling correction.** My first `e137_pipeline_census.py` assumed the file was cumulative and printed a false `route_b_covered_every_scored_cell = False`. That was an artefact of my wrong model of the counter. I rewrote the script around the real semantics before reporting.

## Item 2R — the Route B cost curve and the attribution

Isolated curve, reps 41, inner 15, bootstrap 10,000 draws, seed 20260822. Route B claims **every** cell at M=3…9 and is **bitwise identical** to the fallback everywhere — max absolute delta 0. Config `arm=sumtable`, `entry=tiered_switch`; ipg 3,4,5,6,7,4,3 → weight passes 1,1,1,1,1,2,3.

| quantity | value |
|---|--:|
| in-situ round step 5→6 (item 1) | 39,134.9 us |
| isolated Route B step, dispatch-weighted | **30,750.8 us** [28,143.0, 31,756.8] |
| `e137_isolated_to_insitu_transfer_at_boundary_4` | **0.7858** [0.7191, 0.8115] |
| `e137_non_qmv_share_of_m5_to_m6_step` | **0.2142** [0.1885, 0.2809] |
| in-situ ÷ isolated | 1.2727 |

**F2 pre-registered decision rule: ≥ 0.60 → "QMV is the carrier; attack it."**

This **falsifies the advisor's F3 prediction of 0.12–0.32.** Per F3 section 3, all three QMV cost models are therefore wrong by 3–5x: Model A (activation traffic) predicted 0.117, Model B (instruction issue) 0.219, Model C (occupancy, the advisor's own disbelieved upper bound) 0.197. The measurement is 3.6x the best model and 2.5x the top of the double-counted bracket.

### Per-shape steps, as fractions

| shape | disp | M=5 us/call | M=6 us/call | step us/call | **rel** | weighted step us |
|---|--:|--:|--:|--:|--:|--:|
| `mlp.down` | 64 | 352.0 | 519.2 | 167.2 | **47.5 %** | 10,702.0 |
| `mlp.gate_up_fused` | 64 | 565.7 | 727.6 | 161.9 | **28.6 %** | 10,363.5 |
| `linear_attn.in_proj_fused_qkvzba` | 48 | 303.1 | 407.7 | 104.6 | **34.5 %** | 5,018.5 |
| `linear_attn.out_proj` | 48 | 147.1 | 183.6 | 36.6 | **24.9 %** | 1,755.5 |
| `full_attn.qkv_proj_fused` | 16 | 269.6 | 340.7 | 71.1 | **26.4 %** | 1,137.7 |
| `head.lm_head` | 1 | 3,642.2 | 4,788.2 | 1,145.9 | **31.5 %** | 1,145.9 |
| `full_attn.o_proj` | 16 | 145.8 | 185.0 | 39.2 | **26.9 %** | 627.6 |

Sum = 30,750.7 us ✓

F3 named `mlp.gate_up` as the shape to watch and quoted a 3.1x step from FINDING 52's pre-Route-B curve. **Route B does not reproduce that shape.** `mlp.gate_up` steps by 28.6 %, and it is not even the largest relative step — `mlp.down` is, at 47.5 %.

**All seven shapes step, roughly proportionally, across a narrow 24.9–47.5 % band.** By F3 section 4's own criterion this is "a property of the whole weight stream; no family owns it". The two MLP shapes carry 69 % of the weighted step only because they have the most dispatches (64 each), not because they step harder per call.

### Robustness to the unresolved routing question

Because the census cannot show scored-window routing, I re-derived the decision rule against the incumbent MLX gate using the fallback arm of the same curve:

| arm | 5→6 step (dispatch-weighted) | ÷ in-situ step |
|---|--:|--:|
| Route B routed | 30,750.8 us | **0.786** |
| incumbent MLX fallback | 35,488.3 us | **0.907** |

**The incumbent's step is the larger of the two.** Both clear the 0.60 threshold, so `verdict_identical_under_either_routing = true`, worst case 0.786, best case 0.907. The step is a property of the 4-bit affine matvec work at M=6, **not** an artefact of the Route B specialization ladder. This is a stronger statement than the census would have supplied.

## Item 3 — the SDPA dispatch bracket, recorded only

Per F2 section 7 this is a recorded bracket, not an investigation. 109 extra dispatches at the boundary, priced with the E58 corrected in-situ tax:

- 109 × 77 ns = **8.39 us**; 109 × 428 ns = **46.65 us**
- share of the local 39,135 us step: **0.0214–0.1192 %**
- share of the ranked 16,567 us step (ranked step quoted from F2): **0.0507–0.2816 %**
- SDPA-chunk dispatches alone (64): 4.93–27.39 us

**Dispatch count is eliminated as the carrier**, by three orders of magnitude.

### Free falsification of chunked SDPA, independent of the bandwidth bracket

F2 section 4 closed FINDING 184 by arithmetic (270–420 us = 0.7–1.1 % of the step). I add a second, independent line that needs no bandwidth constant and no dispatch tax:

1. `git diff b5cff751 33ce6a3f -- Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift` is **empty**. The file, the `qL >= 6` guard and `let split = 5` are byte-identical on both bases.
2. The E92 pinned-width GPU-interval ledger (base `b5cff751`, Apple M4 Pro, 512 tokens, `gpu_intervals=1`, widths **pinned** by `e92_verify_width`, ungated) puts the largest `verify_gpu_busy_us` step at **4→5** (39,304.7 us). The current base puts it at **5→6**.
3. A guard that fires only at `qL >= 6` cannot produce a 39,305 us step at the 4→5 boundary.

**Chunked SDPA is eliminated as the carrier of the step.** E92 verify_gpu_busy steps: 1→2 5,163; 2→3 4,312; 3→4 10,936; **4→5 39,305**; 5→6 12,202; 6→7 10,902; 7→8 11,447; **8→9 38,125**. GPU idle never steps (0.6–2.7 ms at all widths).

Because E92 pinned the widths, width and decode position are not confounded there. That the cliff sits at a *different* width on a different base is itself informative: the cliff tracks something the base changed, not a fixed guard.

---

## Conclusion

**What happened and why.** The M=5 → M=6 step is real, large (36.7 % of the round locally), 99.6 % GPU, and it transfers to the ranked host almost exactly (36.1 % ranked, quoted). It is 79 % reproduced by an isolated, dispatch-count-weighted Route B QMV curve and 91 % by the incumbent MLX gate. So the answer to "which dispatch family carries it" is: **the quantized matvec carries nearly all of it, but no individual family owns it.** All seven QMV shapes step within a narrow 24.9–47.5 % band, in rough proportion to their existing cost.

**Evidence for the mechanism.** The step survives every structural explanation I could test and every one the advisor proposed:

- not host CPU (0.39 % of the step) — FINDING 187
- not dispatch count (≤ 0.12 % of the step)
- not chunked SDPA (guard byte-identical while the cliff moved one width between bases)
- not GDN recurrence (FINDING 188: `gatedDeltaKernel` grid contains no T, registers independent of T)
- not a Route B specialization artefact (the incumbent gate steps harder)
- not one shape (F3's `mlp.gate_up` 3x prediction is not reproduced)

**Evidence against a lever.** That list is the problem. Three quantitative cost models built by the advisor — activation traffic, instruction issue, occupancy — predict 0.117, 0.219 and 0.197 against a measured 0.786. A 3–5x modelling error across three independent models suggests the true mechanism is not in any of them. The streaming-efficiency identity the advisor derived (implied GB/s falls 7–11 % per row everywhere except 26.5 % at the boundary) is consistent with a whole-weight-stream effect, and it is a discriminator only, not a lever (Rule 94).

**Prompt or M5 transfer risk.** **High for every microsecond, low for the cliff itself.** FINDING 181 governs: this host is `applegpu_g16s` with a 96-register ceiling, ranked is g17s at 126. `qwen_e120_qmv_wide<6>` clamps here and `<7>` spills 32 B here, so the routed QMV curve is host-specific and `transfer_safe = false`. The `_nax` M5 variants were not measured. What does transfer is the normalised cliff: 36.7 % local against 36.1 % ranked. Everything here rests on one public fixture and candidate-generated reference rows, so it is directional local evidence only.

**Smallest useful next action.** I recommend **not** spending a leg on the remaining 21 % non-QMV residual: its CI is [0.189, 0.281] on a base that differs from the in-situ denominator's base, and shrinking it would not produce a lever.

The one question I would fund is different and cheap: **why does the cliff sit at 4→5 on `b5cff751` and 5→6 on `33ce6a3f`?** A cliff that moves between bases is a cliff with a cause somebody already changed. Bisecting the two bases for the commit that moved it would name the mechanism directly, and it needs no new instrument — E92's pinned-width ledger already exists and the pinning removes the width/position confound. That is a far better use of GPU time than another attribution digit.

**Recommendation: close.** The assigned question is answered, with a negative practical verdict: QMV carries the step, but the step is diffuse across all seven shapes, so there is no single kernel to attack. Record FINDING 187 (step is 99.6 % GPU), the transfer factor 0.786, the routing-robustness result, and the two falsifications (chunked SDPA, dispatch count). Open the cliff-position bisect as a separate, distinct experiment.

## Suggested follow-ups, not implemented

1. **Bisect `b5cff751` → `33ce6a3f` for the commit that moved the cliff.** Highest value per GPU hour of anything I saw.
2. **Add an end-of-leg flush to `MLX_E120_QMV_PIPELINE_LOG`.** One line in `flushPipelineLog`'s trigger, outside my scope. It would make scored-window routing observable for every future experiment. Low value for E137 specifically, real value as shared instrument repair.
3. **Measure the `_nax` M5 variants of the seven shapes.** Everything quantitative here is g16s-only, and the campaign will eventually need the g17s curve.
4. **Check whether the 26.5 % boundary drop in implied GB/s appears in the incumbent gate too.** If it does, it is a memory-system property and the search should move off the kernels entirely.
