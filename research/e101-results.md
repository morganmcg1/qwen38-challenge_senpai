SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":"b8b8b860-18cd-42b3-a828-2c223aaa2173","primary_metric":{"name":"round_us_delta_per_draft_us","available":true,"value":-39.16},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

# E101 — fused row top-32 selection for the arm C cluster shortlist

- **Student / branch:** `qwen-thorfinn` / `qwen-thorfinn/e101-selection-chain-topk`
- **Hypothesis and target cost:** replacing the `argPartition` merge-sort chain
  in the arm C draft-token selection path with single-dispatch threadgroup
  top-K kernels removes a fixed per-draft latency. Under the latency-class
  transfer frame (Finding 22) that latency transfers to the ranked M5 host at
  about 1:1 in absolute microseconds, so it pays in proportion to the ranked
  round time rather than to a byte share. Target: at least 40 us/draft removed
  and at least +0.32 % on the ranked median pair.
- **Decision:** green locally, submitted officially.
- **`BASE_SHA`:** `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`
- **`UPSTREAM_SHA` at submit:** `41bad1c6f124f8f0c7f324cf60e95cd2c4de2ca6`
  ("Accept submission 51b9bf85")
- **Advisor base:** `88d5037d107a2e3a0a48b56736e5dd7d2a77f0d2`
- **Candidate commit:** `75db0c712cc69302a8746ef26b5b431057e4d773`
- **Yukon promoted frontier used:** `51b9bf85` at `3.35025879` (live board).
  `senpai/frontier-state.json` still records `8819b108` at `3.32794960796967`
  and is two promotions stale; that file is advisor-owned and was not edited.
  Our own promoted row is `f04b102e` at `3.32824628683457`.
- **Candidate build fingerprint:** worker
  `3be0383accbd22c93a83e2b42d3725f5c696bb68da5e5835863f855dba9cf7a9`,
  built 2026-08-21T15:26:52Z. Earlier chain-C-only evidence was taken on
  `7572bc01b3f3f83c30591a6854cf768b99b0554d084fd2c67db7cec5c2f7431e`.
- **Generated-twin digests:** twin audit reports 29 runtime-effective twins and
  1 allowlisted comment-only waiver, non-comment lines byte-identical with both
  comment streams sha256-pinned. `mlx.metallib` is not the source form for the
  `quantized` family; the worker binary carries the JIT source string.
- **Submitted candidate files:**
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`,
  `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`.
- **Supporting test, tooling, documentation files:**
  `Tests/MLXFastTests/QwenDraftReadoutExactnessTests.swift`,
  `research/e101_abba.sh`, `research/e101_abba_analyse.py`,
  `research/e101_bare_gate.sh`, `research/e101_head_census.py`,
  `research/e101_wandb_log.py`, `research/e101_row_digest.py`, this file.
- **MTP head provenance, digest, draft policy:** declared run head,
  `head_provenance_sha256`
  `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`,
  identical on every leg reported here. Depth cap 8, `arm=ship` EMA schedule
  unchanged by this experiment.
- **Token window, fixture, reference source, harness:** 512 decode tokens for
  the exactness and ABBA work and 128 for `--local-submit`, public fixture,
  candidate-generated reference rows, `harness=local` throughout. The single
  `harness=ranked` datum is the submission itself, which is still validating.
- **Exact cell:** proposal-side draft-token selection. Live arm C geometry is
  12,292 clusters, 8 rows per cluster, 3,073 probes, so selection runs over
  24,584 rows and emits 32 candidate ids. Source form: `MLXFast.metalKernel`
  JIT strings inside the worker binary, kernels `qwen_mtp_probe_sort`,
  `qwen_mtp_row_top32_partial`, `qwen_mtp_row_top32_finalize`. The composed
  tree also carries `qwen_mtp_draft_selected_affine4_rerank_g64_v1` (imported)
  and the E100 `qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>` verify variant.
- **Official causal path and score equation (`harness=ranked`):** the ranked
  numerator is the runner-owned prebuilt baseline, so
  `d ln(ranked baseline serial time) / dx = 0` for every candidate edit. Any
  microsecond removed from the candidate MTP leg lowers
  `candidate_mtp_seconds_per_token_mean` and raises every affected `raw_p`.
  Ranked percentage gain = microseconds removed per draft x drafts per round /
  ranked round microseconds. No `psi_serial` term and no byte-share factor is
  applied. `senpai/verify-ranked-score-boundary.sh` PASS.
- **Assignment-scope preflight:** `assignment scope OK: 2 submitted path(s)
  against BASE_SHA=770a3ff2...`
- **Editable bytes:** `source=2554622/3000000 headroom=445378
  growth=99787/262144 exempt=2410/2147483648 files=154`
- **Scored-path reachability:** proven by a compiled-default witness rather than
  by reading the source. With `MLX_E101_ROW_TOP32` absent from the environment,
  the per-leg witness reads `sel_env=unset sel_fused=500 sel_argpart=0` at 512
  tokens, so every draft goes through the fused kernels on a bare worker. A
  malformed value calls `fatalError` and the leg exits 1. Caveat kept from the
  earlier report: `fatalError` fires lazily on the first draft, not at process
  start.

## Evidence

- **Host, chip, memory, toolchain, thermal policy:** Apple M4 Pro `Mac16,11`,
  20 GPU cores, 51,539,607,552 bytes, macOS 26.5.2 build 25F84, Apple Swift
  6.3.3. The `--local-submit` leg passed the **real** cool gate: 57.1 C down to
  39.5 C after 50 s. The rung-4 ABBA legs ran ungated and counterbalanced and
  are labelled `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false` verbatim on every W&B run.
- **`head_provenance_sha256` for every leg:**
  `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` on all 16
  rung-4 legs, all 5 bare-gate legs, and the `--local-submit` leg.
  `uses_pinned_mtp_head` is `true` for both artifacts and was not used.
- **Exact commands:**
  - `senpai/rebuild-and-assert-worker.sh --require qwen35_dual_rms_norm_concat_bf16_v1 --require-symbol snapshotScheduleSignal --require-symbol buildDerivedClusterIndex --require qwen_mtp_probe_sort --require qwen_mtp_row_top32_partial --require qwen_mtp_row_top32_finalize --require qwen_mtp_draft_selected_affine4_rerank_g64_v1 --forbid qwen35_dual_rms_norm_bf16_v1`
  - `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions --filter QwenRowTop32SelectionTests`
  - `research/e101_bare_gate.sh`
  - `python3 research/e101_row_digest.py e101bare32 e101ctl32 e101bare512 e101ctl512`
  - `swift test --force-resolved-versions`
  - `python3 research/twin_audit.py`
  - `senpai/validate-assignment-scope.sh 770a3ff2... Sources/MLXFastModel/Qwen36MTPBlockSession.swift Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
  - `senpai/check-editable-budget.sh 770a3ff2...`
  - `senpai/verify-ranked-score-boundary.sh`
  - `MLXFAST_QWEN_MTP_HEAD_DIR=... ./benchmark-qwen-mtp.sh --local-submit`
  - `LC_ALL=C senpai/submit-official.sh 770a3ff2... --model "senpai" --note-file senpai/submission-note-e101-composed.md`
- **Cheapest real falsification gate and positive controls:** three gates, each
  with a control that can fail.

  | gate | trials | result | positive control |
  | --- | ---: | --- | --- |
  | fused selection vs `argPartition` | 256 | 0 mismatches | one rejected row promoted above every selected row: `detected true` |
  | rerank order invariance | 256 | 0 mismatches, 0 set mismatches | one shortlist member replaced from outside the shortlist: fired on 15 of 256, against a pre-stated prediction of about 16 |
  | compiled-default arm identity | 4 legs | ledgers identical | `MLX_E101_ROW_TOP32=0` control leg shows the mirror witness, proving the witness discriminates |

- **Tests and risk-based checks, in execution order:** build; runtime exactness
  suite (4 tests, 1 suite, 0 failures); bare compiled-default gate (0 failures);
  row-digest check (0 failures); full `swift test` (732 tests, 65 suites, 40
  issues across the same 9 organizer names, **zero added**); twin audit; scope;
  budget; ranked-boundary; `--local-submit`.
- **Exact-token and row-ledger verdict:** at 512 tokens the recovered ledger is
  identical tuple for tuple between the fused and legacy arms, including
  per-round `(drafts, accepted)`: 78 rounds, 574 target rows, 513 emitted, 496
  drafts, 435 accepted, 61 rejected, `all_tokens_matched true`,
  `residual_divergence_count 0`, and exact agreement with `score.json` on both
  ratios. Row-evidence digests over the ordered `mtp-row` lines, whose values
  are hex float literals so the comparison is over exact bits:

  ```
  e101bare32  vs e101ctl32 :   64 rows  sha256 c556822abdd850b6fefadd0ebb26dce0750c55eb0362235b6054752bb7afeb3a  IDENTICAL
  e101bare512 vs e101ctl512: 1025 rows  sha256 719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e  IDENTICAL
  ```

- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** `TWIN AUDIT OK: 29 runtime-effective twin(s), 1
  allowlisted comment-only waiver`.
- **Official status and score:** submission
  `b8b8b860-18cd-42b3-a828-2c223aaa2173`, benchmark
  `5d1ee4d7-80bd-4555-b182-6505f26ef495`, status `validating` at
  2026-08-21T15:58Z. Note 10.4 KiB, model attribution `senpai`. No ranked score
  exists yet and none is inferred here.

### Isolated per-draft measurement, rung 4, `harness=local`

STRICT tier 3v3 contrast, base `6699f86b`, worker `7572bc01...`, arm-blind
pooled round-level contamination filter, round 1 always dropped, `p8k` excluded
by the pre-registered kept-fraction rule.

| counter | off | on | delta / round | delta / draft (us) | ranked % on median pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| `round_us` | 152388.8 | 152137.2 | -251.6 | **-39.16** | **+0.3160** |
| `draft_build_us` | 3280.7 | 3218.3 | -62.4 | -9.71 | +0.0784 |
| `d_head1_us` | 56.7 | 58.0 | +1.3 | +0.21 | -0.0017 |
| `d_submit1_us` | 179.1 | 146.1 | -33.0 | -5.13 | +0.0414 |
| `d_chain_us` | 239.1 | 241.2 | +2.0 | +0.32 | -0.0026 |
| `d_submit2_us` | 2773.9 | 2741.5 | -32.4 | -5.04 | +0.0406 |
| `verify_build_us` | 75827.2 | 75693.5 | -133.7 | -20.81 | +0.1679 |
| `eval_wall_us` | 73060.8 | 73000.5 | -60.3 | -9.38 | +0.0757 |

`d_chain_us` is reported but is **not** the decision statistic. `tSubmit1` and
`tChainBuilt` bracket host graph construction with no `eval` between them, so
it cannot see an 11-dispatch GPU removal. This was accepted as advisor error 58.

Whole-leg score: `mtp_s/tok` 0.031001228 off, 0.030955231 on; raw -0.1484 %,
drift-corrected -0.1504 %, session null 0.0263 %, about 5.7 sigma, Welch
`t = -6.37`, `df = 2.8`, `p = 0.01`, 6.4254 drafts per round.

Four-instrument bracket, all `harness=local`:

```
OPS=0 dispatch census (upper bound)   -104.21 us/draft
dispatch-count lower bound             -42.60 us/draft
production round_us (strict)           -39.16 us/draft
whole-leg mtp_s/tok score              -35.70 us/draft
sync-head GPU drain             -28.41 to -30.93 us/draft
```

### `--local-submit`, composed tree, `harness=local`

| Metric | Baseline (chain C alone, worker `7572bc01`) | Candidate (composed, worker `3be0383a`) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.09724727272987366 (DERIVED, see below) | 0.097288226708769798 | +0.042 % |
| MTP seconds/token | 0.054076562635600567 | 0.053877444937825203 | **-0.368 %** |
| local serial-relative speedup | 1.7983257069274636 | 1.80573200568514 | +0.412 % |
| effective mean draft length | 5.9473684210526319 | 5.9473684210526319 | identical |
| accepted draft rate | 0.97345132743362828 | 0.97345132743362828 | identical |

The chain C serial figure is **DERIVED**, not read from a field: it is
`mtp_seconds_per_token x mtp_decode_speedup` on that leg's own `score.json`,
computed inside this campaign's local harness on this host. It is labelled so
that no reader treats it as a recorded measurement.

Identity fields that matched: host, chip, memory profile, toolchain, thermal
policy, fixture, token window, decode depth, proposal head digest, reference
source. Identity field that did **not** match: the worker binary and the base
commit, because the composed tree also carries the merged E100 change. That
comparison is therefore labelled directional, not matched. The 512-token
schedule fields required by the assignment are unmoved in both trees:
`effective_mean_draft_len 6.358974358974359` and
`accepted_draft_rate 0.8770161290322581`.

The local score is a one-prompt directional measurement. It is not the ranked
median across eight hidden prompts, and both of its legs run on the candidate
build, so a change that speeds the whole model cancels in the ratio.

### External ranked evidence that arrived during the run, `harness=ranked`

Yukon row `9612d3b` (`newjordan`, "Exact top-32 over the IVF shortlist, in two
dispatches instead of eight") scored **3.33872764713944** and was rejected only
for not beating the live frontier at `3.35025879`. Against our promoted
`f04b102e` at `3.32824628683457` that is **+0.3149 %**, against a chain C
prediction of +0.32 %. This is an independent tree with an independent serial
draw at sigma about 0.150 %, so it is one observation and not a replication of
our candidate, but it is the first ranked-box datum consistent with a
latency-class saving transferring at roughly 1:1 in absolute microseconds.

## Conclusion

- **What happened and why:** the vendored `ArgPartition::eval_gpu` is a stub
  that fully argsorts the row through `multi_block_sort`, so selecting 32
  elements from 24,584 cost fourteen dependent dispatches and five device
  temporaries. Two custom kernels answer the same question. The saving is
  launch-and-serialization latency, not bandwidth, which is why it prices
  against ranked round time rather than against a byte share.
- **Evidence for or against the mechanism:** four independent instruments
  bracket the effect between -28.41 and -104.21 us/draft, with the two
  end-to-end instruments agreeing to 9 % at -39.16 and -35.70. Exactness is
  established three ways: kernel-level selection identity with a positive
  control, order invariance of the downstream rerank with a control that fires
  at the predicted rate, and bit-exact row-evidence digests over 1,025 target
  rows at 512 tokens. The speculative schedule does not move.
- **Prompt or M5 transfer risk:** the local ratio understates the effect
  because both local legs use the candidate build. The ranked transfer rests on
  the latency-class frame, which had never been checked on the ranked box until
  `9612d3b` landed at +0.3149 % for the same mechanism family. Remaining risk
  is the serial lottery, sigma about 0.150 % on the published number and a
  0.277 % two-sigma single-pair floor, which is large relative to a +0.37 %
  composed expectation.
- **Smallest useful next action:** chain A, the 12,292-to-3,073 radix select
  plus bitmap emit, priced at about 40.42 us/draft and pairing with chain C to
  roughly 145 us/draft and about +1.1 % on the median pair. It is the largest
  unclaimed number on the head path.
- **Recommendation:** promote. The candidate is submitted as
  `b8b8b860-18cd-42b3-a828-2c223aaa2173` and is validating. If the receipt
  rejects on the serial draw rather than on a gate, keep the mechanism and
  resubmit only when chain A or another distinct mechanism changes the archive,
  because Yukon deduplicates on archive content.
