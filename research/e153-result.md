# E153 result: ship leaf16, close the merged wide-decode SDPA kernel

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e153_leaf16_ranked_pct_total_leg","available":true,"value":0.24109},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- **Student / branch:** `qwen-askeladd` / `qwen-askeladd/e153-merged-sdpa-kernel-and-leaf16`
- **Hypothesis and target cost:** Two independent mechanisms.
  - **R1 leaf16.** Widening the derived cluster index leaf from 8 rows to 16
    halves the leaf count, so the coarse centroid pass moves half the bytes at
    a fixed probe fraction, while `probes * rowsPerLeaf` keeps the refinement
    row count effectively constant. Target cost: the coarse centroid pass, once
    per draft step.
  - **R2 merged SDPA.** The wide-decode path issues two separate SDPA calls per
    full-attention layer. Merging them into one kernel removes one dispatch and
    one pass over the key and value tensors per layer for widths at or above 6.
    Target cost: dispatch and streaming overhead across 16 full-attention
    layers.
- **Decision:** R1 **green locally** and shipped. R2 **not useful**, closed,
  and stripped from the submitted surface.
- **`BASE_SHA` / `UPSTREAM_SHA` / candidate commit:**
  `b27c004afd515d0998ef86f252ce0cf048b4c197` /
  `c0dbec051c58bccf5435ee1e1e5b01271dc7e179` / see the submitted head of this
  PR. The gate-chain session commit is
  `a37efb2450ba3eca43b461255efbc641a6d51cc2`.
- **Yukon promoted submission / source ref used as frontier:**
  `684821ed-f7b5-48f5-9ce1-df99b59e19b6`, source ref
  `eb5eadc7a165047d4321ce883b9ff30894d8bd19`, score `3.71959723`, as recorded
  in `senpai/frontier-state.json` at `2026-08-23T08:55:00Z`. I did not query
  Yukon. The advisor owns the receipt watcher.
- **Candidate build fingerprint:** gate chain worker
  `18090ae3e012535b4a1c17329688d3c6691f2159c7653a77de4f2e0f0affbd98`. The R2
  ABBA session used one single binary for both arms,
  `4f3abde77ca30e5858268dc91a304da0d7c03a3fc9a362fd3873889671f0fd49`.
- **Submitted-surface / generated-twin / metallib digests:** no Metal source
  changed in the submitted surface. `python3 research/twin_audit.py` reports OK
  across 29 runtime-effective twins with one allowlisted comment-only waiver.
- **Submitted candidate files:** exactly one,
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`, 41 insertions and
  5 deletions.
- **Supporting test, tooling, or documentation files:** `research/e153_*.sh`,
  `research/e153_*.py`, `research/e153-r2-abba.json`,
  `research/e153-r2-prediction.json`, and this document. None is submitted.
- **MTP head provenance, digest, and draft policy:**
  `head_provenance_sha256 = dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`
  on every leg, baseline and candidate.
  `head_manifest_tree_sha256 = 559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`.
  Organizer-pinned head. This branch declares no candidate head. Draft policy
  is the shipped adaptive schedule at offered depth 8;
  `depth_price_arm_requested = unset`.
- **Token window, fixture, reference source, and harness:** 512 decode tokens
  behind a 512-token seed. Public fixtures `beagle_a`
  (`research/e124_prose_hi_beagle_a_512.txt`, sha256 `70b3dca1...`),
  `essays_montaigne`, and `benchfixture`. Reference rows are the checked-in
  E128 goldens, for example `beagle_a-rows-513.json` sha256 `8ed47038...`.
  **`harness=local` on every measured number in this report.**
- **Exact cell:**
  - R1: the derived cluster index over the `[98336, 640]` affine 4-bit
    group-64 compact draft head. Leaf width 8 to 16, so 12,292 leaves to 6,146.
    Dispatch family `qwen35ClusterCentroidQMVKernel` at `grid: (32, tiles*2, 1)`
    and `qwen35ClusterRowQMVKernel` at `grid: (32, probes*2, 1)`. Swift-side
    custom kernel, no `.metal` source form, no `_nax` variant.
  - R2: `MLXFast.scaledDotProductAttention` over the 16 full-attention layers,
    4 KV heads, 24 query heads, head dim 256, `qL` 6 to 9, `kL` 513 to 1023,
    both `contiguous` and `headTransposed` query layouts. Refuses at
    `kL >= 1024`.
- **Official causal path and score equation:** `harness=ranked`. Both
  mechanisms live only in the candidate workspace, so
  `d ln(ranked baseline serial time) / dx = 0` and any reduction in candidate
  MTP seconds per token raises every affected `raw_p`.
  `senpai/verify-ranked-score-boundary.sh` reports
  `PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only`.
  No local cancellation term appears in the ranked equation.
- **Assignment-scope preflight:**
  `senpai/validate-assignment-scope.sh b27c004a Vendor/.../Qwen35.swift` reports
  `assignment scope OK: 1 submitted path(s)`.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  `source=2652191/3000000`, `headroom=347809`, `growth=197356/262144` against
  `770a3ff2`, `exempt=2410/2147483648`, `files=154`. Growth attributable to
  this PR against its own base is **1,860 bytes**. Stripping R2 returned
  **15,842 bytes** to the team, leaving **64,788 bytes** of shared headroom.
- **Scored-path reachability evidence:**
  - R1: the derived cluster readout fires once per draft step, from
    `Qwen36MTPBlockSession.swift:1633` once and `:1649` `(draftCount - 1)`
    times. The 512-token legs recorded 119 and 146 rounds at `edl` 4.2437 and
    3.4384, so the path executed 505 and 502 times per leg.
  - R2: `merged_ran = true` in all 24 exactness cells, and the built worker
    carried the `qwen35_merged_sdpa_vector` symbol. Phase 0 of the shipping
    gate chain then proved the symbol is **absent** from the submitted build.
- **Written promotion rule and verdict:**
  - R1 promotion rule: ship when the exactness gate passes at 512 tokens on two
    prompts with zero divergences, the override positive control fires,
    `swift test` stays at the documented floor, and the surface is
    self-contained. **All four met. Verdict: ship.**
  - R2 stop rule: stop if the gated ABBA local percent fails to reach the
    predicted sign and magnitude outside the 0.052 percent gated-leg floor.
    **Fired against the mechanism at 3.14σ in the wrong direction. Verdict:
    close.**
- **Pre-official evidence budget / timed legs used:** R2 used 12 timed ABBA
  legs plus an ungated isolated probe. R1 used 2 exactness legs in this
  assignment and carries its effect size from the 12-leg E149 arm A gated ABBA.
  **26 timed legs in total across both rounds.** This exceeds the default
  budget of four to six. The named uncertainty that justified it was whether
  the merged kernel's isolated microbenchmark transferred end to end. It did
  not, and the overspend is exactly what caught that.
- **Frozen candidate SHA, if promoted for submission:** the advisor owns
  submission. This branch is ready to freeze at its submitted head.
- **Specific evidence that invalidated the frozen SHA, if any:** none.
- **Submission owner / read-only receipt-watcher job ID:** advisor. I launched
  no watcher and did not call Yukon.

## Evidence

- **Host, instance, chip, memory profile, toolchain, and thermal policy:**
  Apple M4 Pro, Mac mini `Mac16,11`, 48 GB. The worker engages the low-memory
  startup profile because 48 GB is below the 64 GiB full-profile minimum, so
  this host does **not** reproduce the ranked 128 GiB profile. The ranked host
  is `m5-qwen38-27b-mtp` and is not this machine. Thermal policy: the real
  40 °C cool gate, never bypassed, for every leg in this report.
- **`head_provenance_sha256` for every leg, baseline and candidate:**
  `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`,
  identical on both arms of the R2 ABBA and on both R1 exactness legs.
  `score.json.uses_pinned_mtp_head` was not used.
- **Exact baseline and candidate commands:**

  ```bash
  # R1 shipping gate chain, job 0234b74c, exit 0
  research/e153_r1_gate_chain.sh

  # R2 exactness gate, 24 cells
  research/e153_exactness.sh

  # R2 isolated kernel microbenchmark
  research/e153_timing.sh

  # R2 pre-registered prediction, written before the ABBA session
  python3 research/e153_r2_predict.py

  # R2 thermally gated ABBA, 12 legs, one binary
  research/e153_r2_abba.sh
  python3 research/e153_r2_report.py

  # R2 cold-fallback confound test, no new GPU time
  python3 research/e153_r2_warm_confound.py

  # Frames, mechanism classes, and ranked projections
  python3 research/e153_frame_check.py
  ```

- **Cheapest real falsification gate and positive-control verdict:**
  - R2 exactness: 24 cells, two query layouts by `kL` in {513, 769, 1023} by
    `qL` in {6, 7, 8, 9}. `merged_vs_split_max_abs = 0` and
    `merged_vs_split_bit_mismatch_bytes = 0` in every cell. The comparison is
    against the **split** arm, never against a composed fallback. The positive
    control fires at a minimum absolute difference of **0.0107421875**, which
    is 2.7 to 6.0 times the 0.00195 to 0.00391 split-versus-composed-fallback
    difference, so the comparison can fail and does fail when it should.
  - R1 override positive control: `MLX_E141_ROWS_PER_LEAF=12` does not divide
    98,336, and the guard raises
    `MLX_E141_ROWS_PER_LEAF=12 does not divide the padded draft row count 98336`.
    `e153_leaf16_override_positive_control_passed = true`.
- **Tests and risk-based checks, in execution order:**
  1. Phase 0, two-sided symbol assertion on the rebuilt worker. The leaf16
     selectors are present and `qwen35_merged_sdpa_vector` is absent. **PASS.**
  2. Phase 1, `swift test --force-resolved-versions`. **41 issues under the
     documented 10 names.** `diff` of the per-test issue locations against
     `research/out/e149-swift-test.log` is empty, so the floor is matched
     name for name and count for count. The run total prints 42 because of one
     extra **pre-existing base-drift** issue at `E145WidthPinTests.swift:65`,
     where the test expects `depthPriceArm == .pb6` and the base compiles
     `.ship`. This branch touches neither that test nor
     `Qwen36MTPBlockSession.swift`. **Zero issues added by E153.**
  3. Phase 2, `python3 research/twin_audit.py`. **PASS**, 29 twins.
  4. Phase 3, leaf-override positive control. **PASS.**
  5. Phase 4, two 512-token exactness legs. **PASS.**
  6. `senpai/validate-assignment-scope.sh`. **PASS.**
  7. `senpai/check-editable-budget.sh`. **PASS.**
  8. `senpai/verify-ranked-score-boundary.sh`. **PASS.**
- **Exact-token and row-ledger verdict:**

  | leg | tokens | `all_tokens_matched` | residual divergences | rounds | `edl` | accept | entry °C | exit °C |
  | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `beagle_a` | 512 | true | 0 | 119 | 4.2437 | 0.7782 | 40.24 | 61.80 |
  | `essays_montaigne` | 512 | true | 0 | 146 | 3.4384 | 0.7291 | 41.58 | 61.96 |

  **Gate provenance, stated plainly.** The chain ran the real 40 °C gate in a
  separate process immediately before each leg, through
  `benchmark.sh --local-cool-gate-only`, and recorded
  `e153_cool_gate_status=passed` at 39.9 °C and 39.4 °C. Each leg's own
  `meta.txt` therefore carries `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false`, because the leg itself ran no gate. These
  are exactness legs. **No timing claim in this report uses them.** The flags
  are preserved as written rather than relabelled.
- **Divergent tokens or failure category:** none, on either round. The R2 ABBA
  also reported `e153_merged_sdpa_divergences = 0` and
  `schedule_identical_all_blocks = true`, meaning identical round counts,
  identical `edl`, and identical width histograms across arms.
- **Generated-twin audit:** OK, 29 runtime-effective twins, 1 allowlisted
  comment-only waiver, non-comment lines byte-identical.
- **Peak RAM or head/artifact size:** unchanged. The declared head tree is
  427,742,600 bytes and is organizer-pinned. leaf16 halves the centroid table
  and leaves the row tables the same size.
- **Official status and score:** not submitted by me. The advisor owns
  submission.

### R1 leaf16, the shipped candidate

The underlying effect size is the **E149 arm A gated ABBA**: 12 timed legs, real
40 °C gate, entry-temperature spread 0.596 °C, `all_gate_qualified = true`,
`all_tokens_matched = true`, zero divergences.

| Metric | Baseline, leaf 8 | Candidate, leaf 16 | Ratio / delta |
| --- | ---: | ---: | ---: |
| local total-leg percent (`harness=local`) | 0 | −0.10248 % | merged-pair weighted, negative is faster |
| local decode-frame percent (`harness=local`) | 0 | −0.09821 % | |
| local round cost, total-leg frame | 0 | −169.84 µs/round | |
| local round cost, decode frame | 0 | −132.55 µs/round | |
| leaves | 12,292 | 6,146 | ×0.5 |
| probes at the shipped fraction | 3,073 | 1,537 | ×0.5 |
| probed rows in the refinement pass | 24,584 | 24,592 | +0.0325 % |
| effective mean draft length, `beagle_a` | — | 4.2437 | 512-token leg |
| accepted draft rate, `beagle_a` | — | 0.7782 | 512-token leg |
| effective mean draft length, `essays_montaigne` | — | 3.4384 | 512-token leg |
| accepted draft rate, `essays_montaigne` | — | 0.7291 | 512-token leg |
| divergent tokens | 0 | 0 | |

Ranked projection. **This is a model, not a measurement.** Two modelling steps
apply: the depth-exposure discount, and the Rule 134 conversion.

```text
ranked_steps_per_round / local_edl = 4.7345 / 6.3590 = 0.74453   (depth discount)
Rule 134: 524.5 us/round per 1 percent, fitted_on=0cf1637e
```

| frame | local µs/round | ranked µs/round | ranked percent |
| --- | ---: | ---: | ---: |
| **total-leg (frame-consistent)** | 169.84 | 126.45 | **+0.2411 %** |
| decode (mixed frame) | 132.55 | 98.69 | +0.1882 % |

**Frame note, which resolves the two readings the advisor asked for.** I
re-derived the Rule 134 constant from the same board data. It reproduces at
**521.0 µs/round per 1 percent in the total-leg frame** and **468.8 in the
decode frame**, against the recorded 524.5. Rule 134 is therefore a
**total-leg** constant. Pushing a decode-frame delta through it mixes frames
and understates the result. So `+0.2411 %` is the frame-consistent reading and
`+0.1882 %` is a conservative mixed-frame reading. Neither is a ceiling. Every
Rule 134 conversion in this report and in the artifacts is tagged
`fitted_on=0cf1637e`.

**On the Rule 160 isolated-probe discount.** It does not apply here, and the
reason is simpler than an exception class. **The leaf16 number is not an
isolated probe.** It is a 12-leg thermally gated end-to-end ABBA. The only
modelling on top of it is the depth discount and the Rule 134 conversion. There
is no isolated-probe hot factor to remove.

**Mechanism identity, verified this session.**

- The submitted diff matches `outputs_per_thread`, `outputsPerThread`, and
  `OUTPUTS_PER_THREAD` **zero times**.
- leaf16 is **not** the wide-QMV output row tile. That term is `n / entry.rps`
  at `Qwen35.swift:2141`, with `entry.rps == 4` at every routed width. leaf16
  is `derivedClusterRowsPerLeaf`, an ANN index leaf width in a different pair
  of kernels.
- `grep -in nax Qwen35.swift` returns **nothing**. leaf16 has **no dependency**
  on the `quantized_nax` retile scaffold.
- The x-side grid trim is **already on the base and already the compiled
  default**: `Grid.compiledDefault = .tight` gives `columns = ceil(m / ipg)`,
  and the built worker witnesses `e135_default_grid/tight`.

### R2 merged wide-decode SDPA, closed

Pre-registered before the session, in `research/e153-r2-prediction.json`: the
isolated microbenchmark measured 30.77 µs saved per layer at widths 6 to 8, so
492.3 µs per eligible round, predicting −0.1596 % local and +0.550 % ranked.

Measured, `harness=local`, sign convention **positive means the merged arm is
slower**:

| Metric | Split arm | Merged arm | Delta |
| --- | ---: | ---: | ---: |
| `e153_merged_sdpa_local_pct` (total-leg) | 0 | — | **+0.11570 %** |
| `e153_merged_sdpa_local_pct_decode_frame` | 0 | — | **+0.14507 %** |
| round cost, decode frame | 0 | — | +189.56 µs/round |
| round cost, total-leg frame | 0 | — | +186.22 µs/round |
| σ against the 0.052 % gated-leg floor | — | — | **3.14** |
| divergent tokens | 0 | 0 | 0 |
| exactness cells with `max_abs = 0` | — | 24 of 24 | |

Dose response per eligible round, which is what makes this causal rather than
drift:

| prompt | rounds | eligible | eligible fraction | µs/round | µs per eligible round | merged-arm kernels |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `beagle_a` | 119 | 47 | 0.395 | 248.86 | 630.1 | `merged,split` |
| `benchfixture` | 78 | 67 | 0.859 | 459.33 | 534.7 | `merged,split` |
| `essays_montaigne` | 146 | 31 | 0.212 | 135.26 | 637.0 | `merged` |

**The `ARM_MISMATCH` witness flag is a false alarm and does not void the
contrast.** The merged arm witnessed `merged,split` on two prompts because the
documented `kL >= 1024` refusal engages near the end of a 512-token leg and
hands those rounds back to the split path, exactly as designed. The witness
compared an exact string instead of a set.

**The advisor's cold-fallback confound is rejected on three independent
grounds** (`research/e153_r2_warm_confound.py`,
`research/out/e153-r2-warm-confound.json`):

1. The merged warm was **additive**. `warmQwen35MergedSdpaVector` was appended
   after the existing `kL = 1024` and `kL = 1025` warms; none was removed or
   narrowed. The merged arm warmed a strict superset of the split arm.
2. **The leg that never refused carries the largest penalty.**
   `essays_montaigne` witnessed `merged` alone, so no fallback compile was
   possible, yet it is the worst leg at 637.0 µs per eligible round.
3. A compile miss is fixed per leg; the mechanism cost scales with eligible
   rounds. Fitting `extra_leg_us = F + V * n_eligible` gives `F = 7,078 µs` and
   `V = 441.1 µs`. **Deleting `F` entirely still leaves +0.1342 % decode-frame
   at 2.58σ**, merged still slower.

JIT compilation inside the timed region is separately ruled out:
`warmQwen35MergedSdpaVector` is wired at `Qwen36MTPBlockSession.swift:657` and
ran during warmup.

## Conclusion

- **What happened and why.** leaf16 is a clean, small, self-contained win that
  passes every gate, so it ships. The merged SDPA kernel is bit-exact and
  slower, so it is closed and stripped, returning 15,842 bytes of shared budget.
- **Evidence for or against the mechanism.** leaf16: a 12-leg gated ABBA plus
  two 512-token exactness legs with zero divergences, an override positive
  control, and a `swift test` run at the documented floor. Merged SDPA: 24
  bit-exact cells with a firing positive control, and a 12-leg gated ABBA at
  3.14σ **against** it, with a clean per-eligible-round dose response and the
  confound rejected.
- **The methodological result is the most transferable finding here.** The
  isolated kernel microbenchmark predicted −0.1596 % local. The gated ABBA
  measured +0.1157 %. The probe was wrong by more than 1,000 µs per eligible
  round **and got the sign wrong**. When an isolated probe and a gated ABBA
  disagree, **the ABBA wins.** Treat the +0.550 % ranked figure that probe
  produced as an upper bound that was never approached. This is direct support
  for Rule 160 and, for this kernel family, something stronger than a 4.7×
  discount: a sign flip.
- **Prompt or M5 transfer risk.** Real and unquantified for leaf16. This host
  runs the 48 GB low-memory startup profile, not the ranked 128 GiB profile.
  The ranked projection also assumes the Rule 148 median-pair occupancy weights
  hold. leaf16 fires **per draft step**, so a prompt with a low effective draft
  length gains almost nothing: on `plutarch`, at `edl = 0.1557` with 449
  non-drafting rounds of 488, the modelled gain is **+0.0106 %**. The Rule 148
  weighting already excludes `plutarch` from the median pair, but if the median
  pair moves toward low-`edl` prompts, the projection falls sharply.
- **Smallest useful next action.** Two, both cheap, neither implemented here:
  1. **Later-window SDPA warm coverage.**
     `Qwen36MTPBlockSession.warmTargetLaterWindowSDPA` warms `kL = 1024` and
     `kL = 1025` for **`qL` in {1, 5, 4} only**. Widths 2, 3, 6, 7, 8 and 9 are
     warmed only by `warmAllDepthShapes`, behind the 512-row seed, so they
     compile in the short-context dispatch family. With `P(M >= 6) = 0.5861`
     and the later window covering the back half of a 512-token decode, this is
     a large uncovered area in the same family as four promoted frontier rows.
  2. **The wide-QMV output row tile**, `n / entry.rps` with `rps == 4` at every
     routed width. It is a real lever, it is the `y` term the advisor assigned
     to me, and no measurement of mine covers it. It is untouched work.
- **Recommendation:** **promote leaf16**, **close merged SDPA**. The merged
  kernel is preserved complete at commit `e4a9ea2a` if E156 wants to reuse the
  exactness harness rather than the kernel.

## W&B

Run `e153askeladdr1`,
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e153askeladdr1>.
It carries every metric above, four tables (R2 exactness cells, R2 ABBA blocks,
R1 exactness legs, R2 width histogram), and the `e153-terminal` artifact with
every script and JSON.
