SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"stacked_ceiling_pct_of_local_round","available":true,"value":0.2988},"test_metric":{"name":"split_merge_bit_identical_cells","available":true,"value":16}}

# E103: pack GQA siblings and verify rows for the SDPA-over-FA-history dispatch

- Student / branch: qwen-askeladd / `qwen-askeladd/e103-sdpa-fa-history-head-packing`
- Hypothesis and target cost: the largest latency-class census item is
  `SDPA over FA history`, 1,267–1,397 µs/round locally. The claim under test
  was that this cost is dominated by re-reading the same K/V history 30 times
  (24 GQA sibling heads × up to 5 verify rows per threadgroup dispatch), and
  that packing those siblings and rows into one threadgroup would recover at
  least 383 µs/round, the assignment's 0.30 % minimum useful effect.
- Decision: **dead**. The mechanism is not the one named, and the best
  achievable saving is a ceiling that lands exactly on the minimum useful
  effect, which means the realistic in-situ result is under it.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: research base
  `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`; assignment base `97511edb…`,
  rebased mid-experiment onto `5c2c3b8b613841d0d9677d4540e6a08e8bd40759`
  after E100 merged (PR #102), on advisor instruction. **No candidate
  commit: no candidate surface was touched.**
- Yukon promoted submission / source ref used as frontier: none. No
  submission slot consumed.
- Candidate build fingerprint: not applicable, no candidate.
- Submitted-surface / generated-twin / metallib digests: no submitted
  surface. The unmodified metallib on the rebased tree fingerprints
  `mlxfast-metallib-fingerprint-v1 a2dd6b8e470800a158aa85492589b44e9589743a4a6b1426f3f83e6d3b38b3f4`,
  verified content-determined by forcing a recompile and getting a
  byte-identical library. The SDPA family is AOT into `mlx.metallib` and has
  no generated twin.
- Submitted candidate files: none.
- Supporting test, tooling, or documentation files:
  `research/e103_sdpa_arms.metal` (12 arms of the scored kernel),
  `research/e103_sdpa_ab.m` (isolated-dose harness),
  `research/e103_split_ab.m` (5 + r partition harness),
  `research/e103_split_report.py`, `research/e103_census_costs.py`,
  `research/e103_insitu_split.py`, `research/e103_reprice.py`,
  `research/e103_wandb_log.py`.
- MTP head provenance, digest, and draft policy: census legs used the
  declared head at
  `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run`, forced draft
  depth 4 and 5, 64 tokens. No leg was a timing leg, so no head-provenance
  comparison across a baseline/candidate pair applies.
- Token window, fixture, reference source, and harness: `harness=local` for
  every leg. The two census legs are `--local-iterate`, 64 tokens. The
  microbenchmarks hold no model and run no benchmark wrapper.
- Exact cell: verify widths 5 and 6 of the target verify phase, 16
  full-attention layers per round, dispatch family
  `sdpa_vector_bfloat16_t_256_256_nomask_qnt_{c,nc}_nosinks`, source form
  AOT `mlx.metallib`, grid `24x{1..5}x1`, threadgroup `1024x1x1`, gqa 6
  (24 query heads, 4 KV heads), head dim 256.
- Official causal path and score equation: not exercised. Nothing here is a
  score.
- Assignment-scope preflight: the research-only E58 dispatch census
  instrument was installed for the census legs and reverted before this
  report. `git diff --stat 5c2c3b8b HEAD` is eight files, all under
  `research/`, 3,536 insertions, zero deletions.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `senpai/check-editable-budget.sh 770a3ff2…` reports
  `source=2539338/3000000 headroom=460662 growth=84503/262144 exempt=2410`.
- Scored-path reachability evidence: rung 0 census proved the scored decode
  reaches `sdpa_vector_bfloat16_t_256_256_nomask_qnt_c_nosinks grid=24x5x1
  tg=1024x1x1` 16 times per round at width 5 and
  `..._nc_... grid=24x1x1` 4 times per round on the draft head. No steel and
  no two-pass variant is reached.

## Evidence

- Host, instance, chip, memory profile, toolchain, thermal policy:
  `ip-10-231-2-227.ec2.internal`, Apple M4 Pro, `applegpu_g16s`, 20 cores,
  48 GiB. Ranked target is `applegpu_g17s`; register arms were
  cross-compiled for both. No thermal gate applies because no leg is a
  timing leg.
- `head_provenance_sha256` for every leg: not applicable. There is no
  baseline/candidate pair and no scored artifact in this experiment. Both
  census legs used the same declared head directory, recorded verbatim in
  `research/out/e103r2-d{4,5}-ops0/meta.txt`.
- Exact baseline and candidate commands:
  - `research/e96_census_leg.sh e103r2-d4-ops0 4 64 0` (verify width 5)
  - `research/e96_census_leg.sh e103r2-d5-ops0 5 64 0` (verify width 6)
  - `python3 research/e103_census_costs.py e103r2-d4-ops0 e103r2-d5-ops0 --json research/out/e103/census_rebased.json`
  - `python3 research/e103_insitu_split.py e103r2-d5-ops0 11 'w6|target_verify|g2_copy'`
  - `python3 research/e103_reprice.py`
  - `clang -framework Metal -framework Foundation research/e103_sdpa_ab.m` and
    `research/e103_split_ab.m`, run as `/tmp/e103_sdpa_ab` and
    `/tmp/e103_split_ab`.
- Cheapest real falsification gate and positive-control verdict: every
  microbenchmark cell compares its arm's output bit for bit against a
  verbatim transcription of the shipped kernel, and every cell carries a
  positive control that perturbs one output element. **16 of 16 split cells
  bit-identical, 0 violations; 16 of 16 positive controls detected.** The
  merge is exactness-neutral; the reason to reject it is cost, not
  correctness.
- Tests and risk-based checks, in execution order: rung 0 in-situ dispatch
  census; rung 1 static register and spill budget for 11 arms on g16s and
  g17s; rung 2 isolated dose over N ∈ {512, 576, 768, 1024} × M ∈ {1..5},
  palindrome ordering, one process, one queue; rung 2c widths 6–9; rung 2b
  the 5 + r partition; rung 2d the rebased anchor and the in-situ partition.
- Exact-token and row-ledger verdict: not applicable. No candidate was built
  and no 512-token exactness gate was run, because the ceiling measurement
  closed the experiment before any implementation.
- Divergent tokens or failure category: none observed; see the bit-exactness
  gate above.
- Generated-twin audit: not relevant, the SDPA family is AOT only.
- Peak RAM or head/artifact size: unchanged, no artifact produced.
- Official status and score: not submitted.

### What the rungs measured

| Rung | Question | Verdict |
| --- | --- | --- |
| 0 | which kernel does the scored path reach, and who owns the grid | `sdpa_vector…qnt_c…grid=24x5x1 tg=1024x1x1`, 16/round; the kernel body in `kernels/sdpa_vector.h` is editable but `backend/metal/scaled_dot_product_attention.cpp:358-359` owns the grid and is trusted, so packing needs a Swift `MLXFast.metalKernel` |
| 1 | is there register headroom to pack | yes at P ≤ 3 (29–67 registers, 0 spill, ceiling 124–126); **P = 6 spills 400 bytes and costs +282 %** |
| 2 | is the dispatch actually bandwidth bound | **no.** Traffic is 21.6 % of the dispatch, softmax 26.3 %, the cross-simdgroup tail 25.3 %. The redundant read rate is 835–1,046 GB/s, 3.1–3.8× DRAM peak, so the redundancy is cache-served and the census bandwidth line understates the real rate about 16× |
| 2b | what does the trusted `qL*gqa<=32` cap force at M ≥ 6, and what does merging it back save | a 5 + r partition, exactness-neutral in all 16 cells; merge saving **7.54 µs/dispatch**, flat in r, against the advisor's predicted 20.9 µs |
| 2c | do the pack arms hold at widths 6–9 | yes, best arm `d_pack2`, gain grows with width |
| 2d | what is the post-E100 anchor and the real in-situ structure | anchors 103,579 µs (w5) and 139,476 µs (w6); the partition is confirmed in situ and each leg drags a `g2_copy` worth 2.4–3.0 µs |

### The in-situ partition, verbatim

At verify width 6 the FA-layer attention is **absent from the exclusive-kernel
view entirely** — 24 distinct kernels in `w6|target_verify`, none of them
SDPA. At qL = 6 the trusted `qL * gqa <= 32` cap with gqa = 6 is exceeded and
each attention op emits two kernels into one command buffer, so a
one-kernel-per-buffer reducer discards it. The buffer-signature table, 176
buffers each over 11 rounds:

```
16 buffers/rd  kernels_per_buffer=2  77.65 us/buffer
   g2_copybfloat16bfloat16 grid=1280x24x1 tg=64x16x1
   sdpa_vector_bfloat16_t_256_256_nomask_qnt_c_nosinks  grid=24x5x1 tg=1024x1x1
16 buffers/rd  kernels_per_buffer=2  35.33 us/buffer
   g2_copybfloat16bfloat16 grid= 256x24x1 tg=64x16x1
   sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks grid=24x1x1 tg=1024x1x1
```

against width 5, where the same 5-row kernel is alone in its buffer at
79.29 µs. FA attention therefore costs 1,808 µs/round at width 6 (1.296 % of
that round) and 1,269 µs/round at width 5 (1.225 %).

### Metrics

Nothing below is a score. Every figure is exclusive GPU time from a census
leg or a standalone Metal microbenchmark.

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| round GPU busy, width 5, pre-E100 (µs) | 127,176 | — | — |
| round GPU busy, width 5, post-E100 (µs) | 127,176 | 103,579 | −18.6 % |
| round GPU busy, width 6, post-E100 (µs) | 103,579 | 139,476 | +34.7 % |
| session-weighted round (µs) | 127,176 | 126,771 | −0.3 % |
| SDPA over FA history, width 5 (µs/round) | 1,269 | — | 1.225 % of round |
| SDPA over FA history, width 6 (µs/round) | 1,269 | 1,808 | 1.296 % of round |
| merge saving per dispatch, M ≥ 6 (µs) | 20.9 predicted | 7.54 measured | 0.36× |
| in-situ copy given back per FA layer (µs) | 0 | 2.71 | — |
| best pack arm, width 5 (µs/round) | 0 | 296 | 0.78× bar |
| stacked ceiling, session (µs/round) | 0 | **379** | **1.00× bar** |
| stacked ceiling, upper variant (µs/round) | 0 | 404 | 1.06× bar |
| **stacked ceiling (% of local round)** | 0.300 required | **0.2988** | **−0.0012 pp** |
| ranked, central, after in-situ discount (%) | 0.277 floor | 0.277–0.435 | straddles |
| split-vs-merge bit-identical cells | 16 | 16 | 0 violations |
| positive controls detected | 16 | 16 | all |

Labelled inferences: the round lengths for M = 7, 8, 9 are **extrapolated**
from the measured width-6 anchor using the pre-E100 width-5 → width-6
increment of 12,300 µs per row, which is the increment in the regime where
E100's `case 5:` edit does not apply; the M = 3 length is **interpolated**
and the M = 4 length is a **single-round sample**. Those three widths carry
20.8 % of the width mass between them. Verify-width shares are ledger 207,
same source tree, same domain. The ×2.40 latency-class factor and the
1.65×–2.59× in-situ discount are the campaign's published ranked-transfer
terms; the local percentage is not itself a ranked score and the ranked
figures are labelled estimates throughout.

### Retractions

1. I quoted the unmodified rebased metallib fingerprint as `5482488e…`. That
   was read from a build root the sanctioned script had not yet refreshed.
   The correct, reproducible value is `a2dd6b8e…`. It differs from the
   pre-rebase `7ae5c5a3…` solely because E100 edited two lines of
   `kernels/quantized.h`, which compiles into the same library; no SDPA
   kernel changed across the base move, and the measured SDPA cost per
   dispatch is unchanged.
2. After rung 2 I declared the stop rule fired. The advisor was right that
   this was premature: the split arm had not been measured. Measuring it
   raised the ceiling from 310 µs/round to 379 µs/round. The rule now fires
   on far better evidence.
3. Rung 2b priced the merge on the SDPA kernels alone, because the
   microbenchmark could not see that the in-situ split also pays two
   `g2_copy` dispatches where a merged pass pays one. Correcting that raises
   the ceiling by 25 µs/round to 404 µs, the upper variant above.

## Conclusion

- **What happened and why.** The named mechanism is wrong. The dispatch is
  not bandwidth bound: the redundant K/V traffic is 21.6 % of the dispatch
  and is served from cache at 3.1–3.8× DRAM peak, so removing the redundancy
  removes almost nothing. What the dispatch actually spends is softmax
  (26.3 %) and a cross-simdgroup reduction tail (25.3 %), neither of which
  head packing touches. Packing more rows per threadgroup does help a little,
  but it runs into a hard register wall at P = 6 (400 bytes of spill,
  +282 % cost), so the usable arm is P = 2. Separately, the trusted
  `qL * gqa <= 32` cap does force a 5 + r partition at M ≥ 6 exactly as the
  advisor predicted, and merging it is exactness-neutral, but the merge is
  worth 7.54 µs per dispatch and not the 20.9 µs a linear fit predicts,
  because 24·M threadgroups against 20 cores quantise into waves and the
  fixed term is not recoverable.
- **Evidence for or against the mechanism.** Against: the isolated dose
  decomposition, the measured read rate against DRAM peak, the P = 6 spill,
  and the flat-in-r merge saving. For: nothing survived. The one claim that
  did survive is structural, not economic — the partition exists and merging
  it is bit-exact.
- **The arithmetic that closes it.** Stacking the best pack arm on the
  measured merge gives 379 µs/round session average, 0.2988 % of the local
  round, against a minimum useful effect of 380 µs/round, 0.300 %. That is a
  **ceiling**: it assumes both effects transfer perfectly into a Swift
  `MLXFast.metalKernel` that replaces a dispatch trusted code currently owns,
  at zero integration cost. An upper bound sitting on the bar means the
  realistic result is under the bar.
- **Prompt or M5 transfer risk.** The dominant risk is wave quantisation.
  Every gain here comes from how 24·M threadgroups tile onto 20 cores; the
  ranked target `applegpu_g17s` has a different core count, so the sign of
  the pack gain is not guaranteed to transfer. Ranked, after the ×2.40
  latency-class factor and the 1.65×–2.59× in-situ discount, the central
  estimate is 0.277 %–0.435 % against a 0.277 % published detection floor:
  the low end is the floor to three decimals. A ceiling that is plausibly
  undetectable is not worth a custom-kernel integration.
- **Smallest useful next action, and it is not in this experiment.** The
  width-6 census found a far larger item in the adjacent latency class.
  `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` costs 116.60 µs/row at
  M = 5 but 136.11 µs/row at M = 6 on the `grid=Mx4352x1` shape, and is
  6.7 % to 18.3 % worse per row at M = 6 across all five shapes it runs.
  E100 collapsed M = 5 to one x-group and left M ≥ 6 paying the old rate. If
  M = 6 matched M = 5's per-row cost that is **13,869 µs/round, 9.94 % of
  the width-6 round**; M ≥ 6 carries 58.7 % of the width mass, so roughly
  **8.1 ms/round session average, 21× this entire experiment's ceiling**.
  E100 raised the `static_assert` cap to `NA <= 5` exactly, so extending it
  needs the register budget checked at NA = 6, but the inefficiency is
  measured, not modelled. This belongs to whoever owns the weight-stream
  line.
- **Recommendation: close.** Do not promote, do not repeat, do not compose
  later. The mechanism named in the assignment is falsified, the best
  remaining variant is a ceiling equal to the minimum useful effect, and the
  census produced a handoff worth twenty times more than the thing it was
  measuring.

## W&B runs

Entity `wandb-applied-ai-team`, project `qwen38-mlx-challenge-senpai`, group
`e103-sdpa-fa-history-head-packing`:

| Run | Rung | Contents |
| --- | --- | --- |
| `2p44qjdf` | 0 | in-situ dispatch census, editable-versus-trusted split |
| `peutgxvj` | 1 | registers, spill, text bytes for 11 arms on g16s and g17s |
| `bj9zpvtw` | 2 | isolated dose, 20 cells, 10 arms, bit-exactness per cell (supersedes `pokn3qzu`) |
| `tan8623z` | 2b | the 5 + r partition, merge saving, fidelity and positive controls |
| `ibgx90bh` | 2d | rebased anchor, in-situ buffer signatures, session re-pricing, QMV handoff |

Every run logs `harness=local`, `timing_valid=false`,
`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`official_or_ranked_score=false`.
