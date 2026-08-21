SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"g17s_entry_point_registers_wideN","available":true,"value":120},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}

# E102 — g17s occupancy and rows per SIMD

- **Student / branch:** `qwen-askeladd` / `qwen-askeladd/e102-g17s-occupancy-rows-per-simd`
- **Hypothesis and target cost:** A wide-row `qmv_fast_crossrow_affine4_g64` case
  raises the register count of the scored `affine_qmv_fast` entry point, which
  lowers SIMD-group occupancy and taxes every dispatch through that entry point,
  including the low-width prompts that never take the wide case. The assignment
  pre-registered the opposite prediction for `3ff80e86`: that routing wide rows
  through a separate `_wideN` helper keeps the low widths at 91 registers.
- **Decision:** green locally, as a **negative-and-corroborating static
  result**. The pre-registered premise is falsified. The E77 occupancy law is
  corroborated a second time with zero fitted parameters. No candidate change is
  proposed and none is possible inside the editable surface.
- **`BASE_SHA`:** `ad8403f1fcca4c3cc5b2f6aa7239ea7e40c81d1a`
- **`UPSTREAM_SHA`:** `8b54ff11c6d686628f6534d7127a261115782757`
- **Candidate commit:** none. This experiment ships research files only.
- **Yukon promoted submission used as frontier:** live top promoted row at
  write time is `51b9bf85`, `officialScore` 3.35025879204714, source ref
  `41bad1c6f124`, created 2026-08-21T11:41:47Z. `senpai/frontier-state.json`
  still records `8819b108` / 3.32794960796967, so the advisor record is three
  promotions behind. No number in this report depends on the frontier value.
- **Candidate build fingerprint:** none. No candidate worker was built.
- **Submitted-surface / generated-twin / metallib digests:** unchanged. The
  diff from `BASE_SHA` touches only `research/*`.
- **Submitted candidate files:** none.
- **Supporting test, tooling, or documentation files:**
  `research/e102_register_reconcile.py`, `research/e102_report.py`,
  `research/e102_reachability.py`, `research/e102_fixed_cost_split.py`,
  `research/e102_wide_row_pricing.py`, `research/e102_dispatch_census.py`,
  `research/e102_na_census.py`, `research/e102_sibling_scan.py`,
  `research/e102_round_check.py`, `research/e102_kernel_fingerprint.sh`,
  `research/e102_pipeline_probe.m`, `research/e102_wandb_log.py`,
  `research/e102-results.md`.
- **MTP head provenance, digest, and draft policy:** not applicable. No
  proposal head was loaded and no generation was run.
- **Token window, fixture, reference source, and harness:** not applicable for
  the static arms. The ranked per-prompt deltas reused in rung 0a come from
  published Yukon receipts at the fixed 512-token ranked window, eight hidden
  prompts. Every number in this report is labelled **`harness=local`** because
  the analysis ran on the local M4 Pro host; no leg of this experiment is a
  ranked measurement made by me.
- **Exact cell:** entry point
  `affine_qmv_fast<bfloat16_t, 64, 4, false>`, affine 4-bit group-64, dispatch
  family `qmv_fast` cross-row, source form **JIT from the
  `mlx-generated/quantized.cpp` twin string**, cross-compiled for both the local
  `applegpu_g16s` variant and the ranked `applegpu_g17s` M5 variant.
- **Official causal path and score equation:** `harness=ranked`. Registers per
  thread set concurrent SIMD groups per core through
  `S = floor(496 KiB / (128 R))`, and E77 prices occupancy as
  `Omega = (32 / S) ** 0.01346`. A rise in `R` at the shared entry point raises
  candidate MTP seconds per token on **every** prompt, so it lowers every
  affected ranked `raw_p`. This is a candidate-leg-only effect; the ranked
  serial numerator is produced by the runner-owned prebuilt baseline workspace
  and cannot move. No local cancellation term is used anywhere in this report.
- **Assignment-scope preflight:** `git diff --stat BASE_SHA..HEAD` lists 12
  files, all under `research/`. No `benchmark.json` editable path is touched.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  `senpai/check-editable-budget.sh ad8403f1…` reports
  `source=2539338/3000000 headroom=460662 growth=0/262144 exempt=2410 files=154`.
  Growth is exactly zero.
- **Scored-path reachability evidence:** see rung 2 below. Published as W&B run
  `j6zdgx8e`.

## Evidence

- **Host, instance, chip, memory profile, toolchain, thermal policy:** Apple M4
  Pro, `applegpu_g16s`, 20 GPU cores, 48 GiB, macOS 26.5.2, Metal 32023.883.
  The ranked `applegpu_g17s` M5 variant is reached only through
  `xcrun metal-tt` cross-compilation, never by execution. **No GPU work of any
  kind ran.** The advisor refused GPU time for this experiment, so
  `cool_gate_passed_real_gate` and `gate_qualified_for_timing` are **not
  applicable**: there is no timed leg to gate. Every W&B run logs
  `timing_valid=false`, `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false` verbatim, and `official_or_ranked_score=false`.
- **`head_provenance_sha256` for every leg:** not applicable. No leg was run.
- **Exact baseline and candidate commands:**

  ```bash
  python3 research/e102_register_reconcile.py --out research/out/e102/regs.json --keep /tmp/e102-arms
  python3 research/e102_report.py research/out/e102/regs.json
  python3 research/e102_reachability.py research/out/e102/reachability.json
  YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
  bash research/e102_kernel_fingerprint.sh
  python3 research/e102_wide_row_pricing.py all
  python3 research/e102_fixed_cost_split.py ca9251b8 B --regs 98
  python3 research/e102_fixed_cost_split.py 3ff80e86 A --regs 120
  python3 research/e102_fixed_cost_split.py 3ff80e86 A2 --regs 120
  python3 research/e102_fixed_cost_split.py ff73cbbd B --regs 111
  python3 research/e102_wandb_log.py
  ```

- **Cheapest real falsification gate and positive-control verdict:** the
  patched-JIT method had to be proved able to reproduce a real submission tree
  before any patched arm could stand for a tree. Arms `K_ctrlA_59b321ee` and
  `L_ca9251b8_real` build the real submission trees from
  `upstream/submissions/<uuid>`. `K` reproduces patched arm `A_shipped` with a
  byte-identical `applegpu_g17s` AIR text sha `846d5999`, and `L` reproduces
  patched arm `B_ca9251b8` with `f7e64a2a`. The **positive control that proves
  the comparison can fail** is arm `J_3ff80e86_wideN`, built by the same real-tree
  path, which produces a different sha `5d5fd459` and a different register
  count. The instrument therefore separates trees that differ and matches trees
  that do not.
- **Tests and risk-based checks, in execution order:**
  1. bare-body cell census, which must reproduce the E76 and E97 numbers;
  2. patched-JIT dispatcher arms A through I;
  3. real-tree arms K, L, J and the widened control N;
  4. cross-arch translation of every arm to `applegpu_g17s`;
  5. Metal pipeline-state probe on every arm and cell;
  6. reachability derivation from the live editable source;
  7. `benchmark.json` editable-path census over all 972 submission branches;
  8. fixed-versus-proportional fit on the published ranked receipts.
- **Exact-token and row-ledger verdict:** not applicable. No generation was run
  and no candidate surface changed, so token fidelity is unchanged by
  construction.
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not applicable. No twin was modified. Every arm is
  a patched JIT string compiled outside the shipped tree, so
  `Vendor/.../kernels/quantized.h` and `mlx-generated/quantized.cpp`, which
  alphonse owns, were never edited.
- **Peak RAM or head/artifact size:** not relevant. Peak working set of the
  census is a few hundred MiB of toolchain output under `/tmp/e102-arms`.
- **Official status and score, if submitted:** not submitted.

### Rung 1 — the entry-point register census

Entry cell `affine_qmv_fast<bfloat16_t, 64, 4, false>`. `AIRpeak` is peak live
values in the AIR entry scope. `reg` is the allocated register count reported
by the Metal toolchain, `spill` is spill bytes.

| arm | source | AIRpeak | AIRlines | g16s reg/spill | g16s bytes | g17s reg/spill | g17s bytes | g17s sha |
|---|---|---:|---:|---|---:|---|---:|---|
| A_shipped | patched JIT | 163 | 13038 | 94/0 | 121,072 | 91/0 | 126,984 | 846d5999 |
| B_ca9251b8 | patched JIT | 183 | 13554 | 95/0 | 128,668 | 98/0 | 134,730 | f7e64a2a |
| C_m5_only | patched JIT | 182 | 12703 | 95/0 | 119,344 | 98/0 | 125,096 | 983a6bc7 |
| D_fact2b (NA<=6) | patched JIT | 201 | 12958 | 96/16 | 118,650 | 111/0 | 124,280 | 576e3432 |
| E_dead_m9_body | patched JIT | 162 | 12434 | 94/0 | 115,476 | 91/0 | 121,174 | 44096e70 |
| F_dead_m9_case | patched JIT | 162 | 12434 | 94/0 | 115,476 | 91/0 | 121,174 | 44096e70 |
| G_prune_both_m9 | patched JIT | 156 | 11640 | 94/0 | 109,404 | 91/0 | 114,736 | e83845f6 |
| H_prune_narrow | patched JIT | 121 | 7479 | 94/0 | 75,948 | 91/0 | 79,190 | 53e07912 |
| I_prune_all_dead | patched JIT | 120 | 6875 | 94/0 | 70,352 | 91/0 | 73,380 | f178d05e |
| **K_ctrlA_59b321ee** | **real tree** | 163 | 13038 | 94/0 | 121,072 | 91/0 | 126,984 | **846d5999 = A** |
| **J_3ff80e86_wideN** | **real tree** | 164 | 12725 | 96/**48** | 115,622 | **120**/0 | 121,016 | 5d5fd459 |
| **L_ca9251b8_real** | **real tree** | 183 | 13554 | 95/0 | 128,668 | 98/0 | 134,730 | **f7e64a2a = B** |
| N_wideN_all_widths | patched JIT | 246 | 12948 | 96/**320** | 109,508 | **126/272** | 113,634 | c8f6a80a |

Bare-body cells, which reproduce E76 and E97 digit for digit (g16s/g17s
registers): NA2 70/83, NA3 93/90, NA4 94/91, NA5 95/98, NA6 96 with 16 B spill
/ 111.

**Rule.** The dispatcher entry point allocates for the **widest inlined body**,
not for the body the current call takes. This held in **11 of 11** arms of the
`_wide` family.

**The Metal pipeline-state probe is a null instrument here.** Every arm and
every cell reported `maxTotalThreadsPerThreadgroup=1024`,
`threadExecutionWidth=32`, `staticThreadgroupMemoryBytes=0`. The probe cannot
see the register-driven occupancy limit, so it neither supports nor contradicts
the register numbers. I report it so nobody spends time on it again.

E77 pricing applied to these register counts, `harness=ranked` model:

| R | S | predicted flat tax vs R=91 |
|---:|---:|---:|
| 91 | 43 | reference |
| 98 | 40 | +0.0974 % |
| 111 | 35 | +0.2775 % |
| 120 | 33 | **+0.3569 %** |
| 126 | 31 | +0.4414 % |

### Rung 2 — scored-path reachability, complete and negative

`M = min(qwenMTPMaxDraftDepth 8, segmentedVerifyDepthCap 7) + 1 = 8`, and
`ntg.x == M` through `grid_dims(M, ...)` at
`mlx/backend/metal/quantized.cpp:254`, consumed by `[[threadgroups_per_grid]]`
at `kernels/quantized.h:1886`.

The smallest scored `out_vec_size` is 5120, confirmed against E70 section 2.2
and E74 rung 3. The full scored set is
{5120, 14336, 16480, 34816, 98336, 248320}. The narrow branch
`1024 <= out_vec_size < 4096` is therefore **entirely dead**.

Eight instantiations are dead: `qmv_fast_crossrow_affine4_g64<T,3..9>` and
`_m<T,9,3,true>`. Narrow case 2 is not dead code in the linker sense because it
shares the live wide case-2 instantiation.

Pruning all of them changes registers by **exactly zero** (arm `I`: 91/0 on
g17s, identical to shipped). It removes text only: 73,380 bytes against 126,984,
which is **−42.21 %** on g17s. **Text size is not a register lever.**

### Rung 3 — not started, as instructed

The advisor explicitly refused GPU time for the M=5 dispatcher cell. Rung 3
required a timed dispatcher measurement, so it was not started. No partial rung-3
evidence exists and none is claimed.

### Rung 0a — why `ca9251b8` looked like a rows-per-simd win

`9b241879` (2026-08-20) against `ca9251b8` (2026-08-18) touch six editable
files. The session diff is a **pure addition** by `9b241879`: 173 lines added,
**0 removed**.

The addition is `warmTargetLaterWindowSDPA`, defined at line 501 and called at
line 494. It pads every full-attention KV cache to 1024 and runs SDPA at query
lengths `[1, 5, 4]` twice, plus `concatenated` at every draft depth. `ca9251b8`
does not have it, so `ca9251b8` pays those first-use costs **inside the timed
leg**. The schedule stays bit-identical, matching E87 section 13.

A ranked leg decodes a fixed 512 tokens, but wall time varies **2.7×** across
the eight prompts, from 5.6 s to 15.5 s. A fixed one-time cost therefore appears
as a **larger percentage on the short, high-width prompts**, which is the same
sign and ordering as a genuine per-row cost. `research/e102_fixed_cost_split.py`
separates the two by fitting one constant each and comparing percent-space RMSE.

| tree | tier | n_ctrl | R | flat tax | net G=2 | net G=1 | fitted fixed c | RMSE fixed | RMSE prop | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ca9251b8` | B | 1 | 98 | +0.0974 % | +0.4542 % | **+0.0094 pp** | +16.9 ms/leg (sd 15.6) | 0.2112 | 0.2572 | **fixed, 1.22×** |
| `3ff80e86` | A | 2 | 120 | +0.3569 % | −0.1643 % | **−0.0635 pp** | −11.5 ms (sd 24.5) | 0.1839 | 0.1728 | prop, 1.06× |
| `3ff80e86` | A2 | 18 | 120 | +0.3569 % | −0.5001 % | −0.5929 pp | −47.5 ms (sd 48.5) | 0.3843 | 0.2245 | prop, 1.71× |
| `ff73cbbd` | B | 1 | 111 | +0.2775 % | +1.9844 % | +1.0200 pp | +105.8 ms (sd 82.8) | 1.0228 | 1.1827 | fixed, 1.16× |

`NA <= 6` still genuinely loses. `ff73cbbd` keeps a +1.02 pp to +1.98 pp
residual after the flat tax is removed, and that residual is far larger than any
plausible warmup difference.

### The second corroboration of the E77 occupancy law

The law was fitted in E77 on a different dataset. It is applied here with **no
refitting and zero free parameters**. The low-width group is the clean test,
because those prompts never take the wide case, so the only thing they can feel
is the shared entry-point allocation.

| tree | R | S | predicted tax | measured raw G=1 | residual |
|---|---:|---:|---:|---:|---:|
| `ca9251b8` | 98 | 40 | +0.0974 % | +0.1068 % | **+0.0094 pp** |
| `3ff80e86` | 120 | 33 | +0.3569 % | +0.2934 % | **−0.0635 pp** |

Both residuals lie inside the 0.113 % within-mode standard deviation, and they
have **opposite signs**, so this is not a shared bias in my instrument.

### There is no escape inside the editable surface

The kernel name is built by trusted code at
`mlx/backend/metal/quantized.cpp:263-273`, and the constructed name does not
encode `M`. That file is **not** in `benchmark.json .editablePaths`; I checked
all 89 entries and only `kernels/*` and `mlx-generated/*` appear. A census of
all **972** submission branches shows every one of them carries the identical
blob `5bdea16c` for that file, so no submission has ever changed it and none
can.

Consequence: every `M` must share one entry point, and that entry point pays the
widest-body register cost. **There is no way to give M=5 its own register
allocation from the editable surface.**

### Net arithmetic for the M=5 cell alone

For `<T,5,5,true>` alone at 98 registers: `+0.0974 %` tax minus `0.1643 pp`
gain, which is about **−0.067 % net**, a small win.

**INFERENCE, not measurement.** This assumes the `_wideN` form and the `_wide`
form deliver the same round-level M=5 gain. Source tree of the gain term is
`3ff80e86` tier A, domain is the five high-width ranked prompts. I did not
measure it and this session had no GPU budget to measure it.

### Metrics table

No timed metric exists. Every timed field is honestly unmeasured.

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | not measured | not measured | not measured |
| MTP seconds/token | not measured | not measured | not measured |
| local serial-relative speedup | not measured | not measured | not measured |
| effective mean draft length | not measured | not measured | not measured |
| accepted draft rate | not measured | not measured | not measured |
| **g17s entry-point registers** | **91** (shipped `59b321ee`) | **120** (`3ff80e86` `_wideN`) | **+29** |
| g17s entry-point registers | 91 (shipped) | 98 (`ca9251b8`) | +7 |
| g17s entry-point spill bytes | 0 | 0 (`J`), 272 (`N`) | +272 at M=8 |
| g17s entry-point text bytes | 126,984 | 73,380 (`I`, all dead pruned) | −42.21 % |
| E77 flat tax, R=120 | 0 % | +0.3569 % | +0.3569 pp |
| E77 residual, R=98 | 0 pp | +0.0094 pp | inside 0.113 % sd |
| E77 residual, R=120 | 0 pp | −0.0635 pp | inside 0.113 % sd |

**Identity fields.** Every arm shares the same base, the same twin source
revision for the unpatched text, the same host, the same toolchain, the same
entry cell, and the same cross-arch tool. The single varied dimension is the
patched dispatcher body or the real submission tree. The ranked per-prompt
deltas in rung 0a all come from the same eight hidden prompts at the same fixed
512-token window, and control tiers are formed only from same-schedule siblings
by `research/e102_wide_row_pricing.py`.

### W&B runs

| run | name | what it holds |
|---|---|---|
| `tk1myykr` | `e102-registers` | 13 arms, 5 bare-body cells, both architectures, E77 tax per arm |
| `j6zdgx8e` | `e102-reachability` | 16 dispatcher cases with reachability and reason |
| `lstdr1e3` | `e102-fixed-split` | 4 fits, 32 per-prompt rows, both model RMSEs |

A first attempt crashed on a missing key and was deleted from the project.

## Retractions

I must withdraw part of my own earlier reporting on this PR.

1. **Interim 2 headline, "the occupancy hypothesis is falsified": WITHDRAWN.**
   The comparison was invalid. I compared two tier-C population levels against
   each other. The tier-C against tier-A control level differs by 0.75 %
   (63,030 µs against 62,563 µs), which is 2× to 8× the effect I was trying to
   resolve. That contrast cannot resolve a 0.1 % to 0.4 % tax.
2. **Interim 2 section 2: WITHDRAWN.** It implied `3ff80e86` avoids the shared
   register tax at low widths. It does not. It pays 120 registers.
3. **Interim 1 "suspect 2": WITHDRAWN.** There is no evidence for a non-register
   `g17s` execution effect at NA=5.

Interim 2 sections 1, 5, 6 and 7 stand unchanged.

## Conclusion

- **What happened and why:** The assignment predicted that `3ff80e86` keeps low
  widths at 91 registers because it routes wide rows through a separate `_wideN`
  helper. That premise is false. `_wideN` is a `METAL_FUNC`, so the compiler
  inlines it into the same `[[kernel]]`, and the entry point allocates for the
  widest inlined body. A separate helper is not a separate allocation. The
  measured cost is 120 registers, not 91, and the in-source comment in
  `3ff80e86` estimating "~96 registers" for M=8 is wrong by 30.
- **Evidence for or against the mechanism:** For. Two independent trees, at 98
  and at 120 registers, match the unrefitted E77 prediction on the low-width
  prompts to +0.0094 pp and −0.0635 pp, with opposite signs and both inside the
  0.113 % within-mode standard deviation. The `ca9251b8` receipt, which
  previously looked like evidence for a rows-per-simd win, is fully explained by
  three things it really has: the flat register tax it does pay, the
  `warmTargetLaterWindowSDPA` warmup it lacks, and the fixed-cost-as-percentage
  artefact created by 2.7× variation in leg duration.
- **Link to Finding 14:** `S` is fitted on the five high-width prompts, and
  those are the shortest legs. Any fixed one-time cost inflates the measured `S`
  with no per-row mechanism behind it. Finding 14 and this result share that
  artefact, so `S` should be refitted after the fixed component is removed.
- **Prompt or M5 transfer risk:** low for the register numbers, because they are
  produced directly for the ranked `applegpu_g17s` variant by the same toolchain
  the runner uses, and because the local `g16s` and ranked `g17s` counts differ
  (94 against 91 for shipped, 96 against 120 for `J`), which shows the
  cross-arch step is doing real work rather than echoing the local target. The
  risk is concentrated in the E77 conversion from registers to time, which is a
  fitted law; this result adds two out-of-sample confirmations of that law but
  does not re-derive it.
- **Smallest useful next action:** none inside this experiment. If the campaign
  later wants the M=5 gain, the only remaining question is whether the `_wideN`
  form and the `_wide` form give the same round-level M=5 gain, which needs one
  timed matched pair on the dispatcher cell. That is exactly the rung-3 work the
  advisor declined, and it stays declined.
- **For alphonse:** his `case 6` → `<T,6,2,true>` control is IPG-only. The
  widest inlined body does not change, so the entry point stays at 91 registers.
  That makes his control register-neutral and a clean dose test. I can confirm
  this statically in minutes on request.
- **Warning for anyone widening further:** `N_wideN_all_widths`, which widens
  M=5 through M=8, spills 272 B on g17s and 320 B on g16s at 126 registers. Any
  proposal that widens all four high-M cases at once will hit spill, not just
  occupancy.
- **Recommendation: close.** The pre-registered question is answered
  definitively and negatively, the mechanism behind it is confirmed twice, and
  the escape route it implied is proved impossible inside the editable surface.
