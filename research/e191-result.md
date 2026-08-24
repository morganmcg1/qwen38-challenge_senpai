# E191 — widthSixWall two-append fix: Stage 0 terminal result

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"two_append_gain_ms_per_round","available":true,"value":-1.9029},"test_metric":{"name":"today_vs_twoappend_max_abs_delta","available":true,"value":0}}

- Student / branch: qwen-edward / `qwen-edward/e191-widthsix-two-append`
- Hypothesis and target cost: FINDING 489 recorded that the shipped m >= 6
  split-SDPA path pays 1.403-2.772 ms/round (2.5-4.9 MUE) because the axis-2
  cache SLICES it feeds to `MLXFast.scaledDotProductAttention` force a KV copy
  (mechanism M1). The assigned fix replaces those slices with a two-append
  protocol so that no slice ever reaches SDPA.
- Decision: **dead**. M1 is refuted, and the assigned fix is a measured,
  replicated regression.
- `BASE_SHA`: `abe68f58fb474ee4eda6377b4fa90dfc62b6b2f0`
- `UPSTREAM_SHA`: unchanged from the base; this experiment did not sync.
- Yukon promoted submission used as frontier: Receipt A, score
  3.70784519415395, cap-7, commit `8ba6e738`. Crown 3.7291100105909. Neither is
  compared against here, because Stage 0 never ran the wrapper.
- Candidate build fingerprint: not applicable. No submitted-surface source was
  changed, so no worker was rebuilt and RULE 384
  (`senpai/rebuild-and-assert-worker.sh`) does not apply. Both probes run under
  `swift test`.
- Submitted-surface / generated-twin / metallib digests: unchanged. No Metal
  source, generated twin, or `mlx.metallib` was touched.
- Submitted candidate files: **none**.
- Supporting test, tooling, and documentation files:
  `Tests/MLXFastTests/E191TwoAppendProbeTests.swift`,
  `research/e191_analysis.py`, `research/e191-stage0.json`,
  `research/e191-result.md`.
- MTP head provenance, digest, and draft policy: not applicable. Stage 0 never
  loads the target model or a proposal head. It drives arms shaped like
  `attentionWithCacheUpdate` directly, so `head_provenance_sha256` has no leg to
  describe.
- Token window, fixture, reference source, and harness: not applicable / not
  applicable / arm-versus-arm array comparison / **`harness=local`**.
- Exact cell: queries `[1, 24, m, 256]`, KV `[1, 4, kL, 256]`, m in 5..9,
  kL in {512, 1024}, `KVCacheSimple`, causal mask, bf16, full-attention layers
  only (16 of 64). Dispatch families observed:
  `sdpa_vector_2pass_1_bfloat16_t_256_256_nomask_qt_c_nosinks_128` and
  `sdpa_vector_2pass_2_bfloat16_t_256`; the `steel_gemm_fused` plus
  `block_softmax_precise` composed fallback; and the `copy` family
  (`g2_`, `g3_`, `gg2_`, `vn_`). Source form: `mlx.metallib` plus JIT-compiled
  `copy`. Host `applegpu_g16s`, **not** the ranked M5 runner.
- Official causal path and score equation: `harness=ranked`. Full-attention
  SDPA runs inside the candidate MTP leg, so a change there moves
  `candidate_mtp_seconds_per_token_mean` and therefore every affected `raw_p`.
  For a candidate edit `x`, `d ln(ranked baseline serial time) / dx = 0`, so no
  `psi_serial` term is subtracted. `senpai/verify-ranked-score-boundary.sh`
  PASS. Every measurement below is `harness=local` and is NOT a ranked score.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh` reports
  `Tests/MLXFastTests/E191TwoAppendProbeTests.swift` outside `editablePaths`.
  That is the intended research-only placement. Yukon never packages `Tests/`
  or `research/`.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `source=2651144/3000000 headroom=348856 growth=0/262144 files=154`. Growth is
  zero because nothing on the submitted surface changed.
- Scored-path reachability evidence: `AttentionUtils.swift:59-146`. The split
  runs only when `queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL` and the mask
  is `.causal`. `Qwen35.swift:3730` is the single scored call site that reaches
  it.
- Written promotion rule and verdict: the assignment stop rule was
  "Stage 0 refutes M1 -> terminal report (Not useful / repriced), no build."
  Stage 0 refutes M1. **No Stage 1 build was made, and nothing was submitted.**
- Pre-official evidence budget / timed legs used: one screen session plus one
  replication session. Zero wrapper legs, zero `--local-submit` legs, zero
  official submissions. Well inside the default budget.
- Frozen candidate SHA: none.
- Specific evidence that invalidated the frozen SHA: not applicable.
- Submission owner / read-only receipt-watcher job ID: none.

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy: AWS Mac,
  GPU family `applegpu_g16s`, 48 GiB, Swift toolchain pinned by
  `--force-resolved-versions`. `MLXFAST_LOCAL_COOL_GATE` does not apply, because
  no wrapper leg ran. Recorded verbatim in `research/e191-stage0.json`:
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `abba_counterbalanced=true`.
- GPU temperature, `temps` session (macmon, degrees C): `session_entry` 37.167,
  `after_warmup` 37.040, per-block entry 36.937 / 37.028 / 37.361 / 37.128 /
  37.350 / 37.220 / 37.483 / 37.299 / 37.145 / 37.350, `session_exit` 37.499.
  The entry-temperature spread across blocks is **0.55 C** and the session drift
  is **0.33 C**. Both are far below any level that could produce the effects
  reported here. The first (`clean`) session ran before macmon was located, so
  it carries no temperature record. The two sessions agree to 1.49% worst case,
  which is the purpose of the replication.
- `head_provenance_sha256` for every leg: not applicable. No leg loads a
  proposal head, and Stage 0 has no baseline/candidate model artifact pair.
- Exact baseline and candidate commands:

  ```bash
  # 1. dispatch census — deterministic, noise-free
  MLXFAST_E191_DISPATCH=1 \
    MLXFAST_E191_DISPATCH_OUT=$OUT/e191-dispatch-census.json \
    swift test --force-resolved-versions --filter E191TwoAppendDispatchTests

  # 2. ABBA-counterbalanced timing, 10 blocks x 50 reps  (run twice)
  MLXFAST_E191_TIMING=1 MLXFAST_E191_BLOCKS=10 MLXFAST_E191_REPS=50 \
    MLXFAST_MACMON_BIN=$HOME/bin/macmon \
    MLXFAST_E191_TIMING_OUT=$OUT/e191-timing-temps.json \
    swift test --force-resolved-versions --filter E191TwoAppendTimingTests

  # 3. reduction + W&B
  python3 research/e191_analysis.py \
    --census $OUT/e191-dispatch-census.json \
    --timing $OUT/e191-timing.json $OUT/e191-timing-temps.json \
    --timing-labels clean temps \
    --out research/e191-stage0.json \
    --wandb-project qwen38-mlx-challenge-senpai \
    --wandb-entity wandb-applied-ai-team --wandb-run-name e191-stage0
  ```

- W&B run: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/jf8cin0o>
  (`jf8cin0o`, state `finished`). It contains the `dispatch_census`,
  `timing_cells`, `cross_session_replication`, `derived`, and
  `today_vs_twoappend_agreement` tables.
- Cheapest real falsification gate and positive-control verdict: the dispatch
  census counts every kernel that Metal actually receives per arm, so a wrong M1
  prediction cannot hide in timing noise. **Positive control:** the `twoAppend`
  arm DOES emit `vn_copy` dispatches, which proves that the census can see a
  KV-sized copy when one exists. The shipped `today` arm emits none.
- Tests and risk-based checks, in execution order: source read of the MLX slice
  and SDPA path; dispatch census; array-agreement check; ABBA timing;
  replication session; reduction. Full `swift test` was not run, because no
  product source changed.
- Exact-token and row-ledger verdict: not applicable at this stage, because no
  decode ran. The equivalent array check is reported below and is exact.
- Divergent tokens or failure category: none. The failure is performance, not
  correctness.
- Generated-twin audit: not relevant. No `.metal`, `.h`, or `mlx-generated/*`
  file was touched.
- Peak RAM or head/artifact size: not relevant.
- Official status and score: not submitted.

### 1. Mechanism M1 is refuted by source and by dispatch count

Line-cited source facts:

| Fact | Source |
| --- | --- |
| A unit-step axis-2 slice is a zero-kernel VIEW, not a copy | `gpu/primitives.cpp:208-218` -> `gpu/slicing.cpp:8-17` -> `common/slicing.cpp:37-69` (`shared_buffer_slice` / `copy_shared_buffer`) |
| SDPA accepts strided KV without copying when `strides.back()==1 && shape[0]==1` | `metal/scaled_dot_product_attention.cpp:702-714` (`kv_copy_unless`); the kernel takes `k_head_stride` and `k_seq_stride` as arguments at `:352-355` and `:690-693` |
| Our KV is `[1, 4, kL, 256]`, so it satisfies that condition | shape above |
| "No slice reaches SDPA" is unachievable anyway, because `KVCacheSimple.update` RETURNS a slice by construction | `KVCache.swift:435-437` |

Dispatch census, scored contiguous query layout, per call
(`research/e191-stage0.json`, table `dispatch_census`):

| kv | m | form | dispatches | copy dispatches |
| ---: | ---: | --- | ---: | ---: |
| 512 | 5 | unsplit | 3 | 2 |
| 512 | 6-9 | **today (shipped)** | **8** | **6** |
| 512 | 6-9 | twoAppend | 12 | 10 |
| 512 | 6-9 | unsplit | 10 | 2 |
| 1024 | 5 | unsplit | 4 | 2 |
| 1024 | 6-9 | **today (shipped)** | **10** | **6** |
| 1024 | 6-9 | twoAppend | 14 | 10 |
| 1024 | 6-9 | unsplit | 10 | 2 |

Grid attribution for the shipped `today` arm at kv=512, m=6. Every copy is
small, and none is KV-sized:

- `gg2_copy(1536, 4)` x2 — the cache write of the 6 new rows. Both arms pay it.
- `g2_copy(1280, 24)` and `g2_copy(256, 24)` — the two QUERY-chunk slices
  (5x256 and 1x256 per head).
- `gg2_copy(1280, 24)` and `gg2_copy(256, 24)` — the final
  `concatenated([outA, outB])`.

**Zero KV-sized copies.** The predicted M1 copy does not exist.

The `twoAppend` arm adds `vn_copy(196608, 1, 1)` x2. Grid x 4 elements per
thread = 786,432 elements = 4 heads x 768-row capacity x 256 — the ENTIRE KV
buffer. At kv=1024 it becomes `vn_copy(327680)` x2 = 1,310,720 elements. The
cause is buffer donation: the second `cache.update` slice-assign cannot donate
its buffer, because the `keysA` and `valuesA` returned by the first update are
still live, so MLX copies the whole buffer and then writes the slice into the
copy. **The assigned fix manufactures the very KV copy it was designed to
remove, and the penalty grows with KV length** — about 48 MB per round extra at
kv=512, and about 80 MB per round at kv=1024.

### 2. Provenance: FINDING 489 measured a different call

FINDING 489 came from
`Tests/MLXFastTests/E186DecodeWidthProbeTests.swift:439-462`, family `sdpa`,
which calls `MLXFast.scaledDotProductAttention` DIRECTLY on contiguous keys —
the **unsplit** form. At m >= 6 that call fails `use_fallback`
(`metal/scaled_dot_product_attention.cpp:591-639`: `supports_sdpa_full` needs
head_dim in {64, 80, 128} and ours is 256; `supports_sdpa_vector` needs
`qL * gqa <= 32` and gqa is 6, so qL <= 5) and runs the `steel_gemm_fused` plus
`block_softmax_precise` composed fallback. FINDING 489 therefore measured what
the split SAVES, not what the split COSTS. Its 1.403-2.772 ms/round is an
accurate number attached to the wrong arm.

### 3. Advisor decision-tree item 2: `today(5) == unsplit(5)` is true by construction

`AttentionUtils.swift:124` gates the split on `qL >= 6, qL <= 9`. At m=5 the
shipped path is exactly the single unsplit `MLXFast.scaledDotProductAttention`
on the cache-updated keys and values. The census confirms it: at kv=512, m=5 the
arm issues 3 dispatches with no split-related copies. `unsplit(5)` is therefore
a valid production control, and the m5 -> m6 production step below is a
like-for-like comparison.

### 4. Timing

Estimator: **minimum over the 10 ABBA blocks**. External system contention only
ever ADDS time to a GPU dispatch block, so the per-cell minimum is the
contention-robust estimator of the cell's true cost. The `temps` session picked
up external load in some blocks, which is visible as a large `sd` in
`timing_cells`. The robust minimum is unaffected, and the two sessions replicate
to **1.49% worst case, with most cells below 1.0%**.

Absolute microseconds per call, `clean` session, robust minimum,
`harness=local`:

| kv | m=5 unsplit | m | today | twoAppend | unsplit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 271.30 | 6 | 370.04 | 475.05 | 473.21 |
| 512 | | 7 | 382.85 | 487.81 | 474.91 |
| 512 | | 8 | 392.01 | 497.66 | 476.17 |
| 512 | | 9 | 405.47 | 507.69 | 476.48 |
| 1024 | 321.78 | 6 | 442.80 | 563.61 | 587.16 |
| 1024 | | 7 | 461.46 | 596.36 | 591.74 |
| 1024 | | 8 | 485.55 | 619.76 | 591.36 |
| 1024 | | 9 | 511.58 | 655.31 | 594.50 |

Derived, multiplied by the 16 full-attention layers, `clean` session,
`harness=local`. The MUE is 567 us/round:

| kv | m | fix gain ms/round | fix MUE | production m5->m step ms/round | 489-form step ms/round | split saves ms/round |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 6 | **-1.680** | -2.96 | +1.580 | +3.231 | 1.651 |
| 512 | 7 | **-1.679** | -2.96 | +1.785 | +3.258 | 1.473 |
| 512 | 8 | **-1.690** | -2.98 | +1.931 | +3.278 | 1.347 |
| 512 | 9 | **-1.636** | -2.88 | +2.147 | +3.283 | 1.136 |
| 1024 | 6 | **-1.933** | -3.41 | +1.936 | +4.246 | 2.310 |
| 1024 | 7 | **-2.158** | -3.81 | +2.235 | +4.319 | 2.084 |
| 1024 | 8 | **-2.147** | -3.79 | +2.620 | +4.313 | 1.693 |
| 1024 | 9 | **-2.300** | -4.06 | +3.037 | +4.364 | 1.327 |

The `temps` session reproduces every row within 1.49%
(`cross_session_replication` in W&B). The sign of `fix gain` is negative in
**16 of 16** session-by-cell rows.

| Metric | Baseline (shipped `today`) | Candidate (`twoAppend`) | Ratio / delta |
| --- | ---: | ---: | ---: |
| fix gain, mean over the 8 clean cells (ms/round) | 0.0 | -1.903 | -1.903 |
| fix gain, best cell for the fix (kv512 m9) | 0.0 | -1.636 | -1.636 |
| minimum needed to clear the assignment target | 0.0 | >= +1.403 | not met |
| `today` vs `twoAppend` max abs array delta | — | 0.000e+00 | exact |
| final cache offset agreement, all 8 cells | — | exact | exact |

The fix needed at least +1.403 ms/round. Its BEST cell is -1.636 ms/round. The
gap is about 3.0 ms/round in the wrong direction, which is roughly 5.4 MUE, and
the result replicates across two sessions.

Identity fields: every compared cell matches on host, GPU family, toolchain,
shapes, dtype, cache class, mask mode, block and rep counts, and arm order. The
only varying dimension inside a comparison is the arm. The two sessions differ
only in wall-clock time and in whether macmon ran. No value above is
interpolated or extrapolated; every number is measured. No local cancellation
term is used, and none of these numbers is a ranked score.

## Conclusion

- **What happened and why.** The recorded widthSixWall mechanism M1 is wrong.
  MLX does not copy KV for an axis-2 unit-step slice: the slice is a zero-kernel
  view, and the SDPA kernel accepts strided KV directly at B == 1. The shipped
  m >= 6 split therefore issues zero KV-sized copies. The assigned two-append
  fix, applied to a problem that does not exist, creates a real full-KV-buffer
  copy per call, because the second slice-assign cannot donate its buffer while
  the first update's result is still live. The fix is bit-exact and slower by
  1.64-2.30 ms/round.
- **Evidence for or against the mechanism.** Against M1: four line-cited MLX
  source facts, a deterministic dispatch census with a working positive control,
  and a two-session ABBA timing replication that agrees to 1.49%.
- **Repricing the widthSixWall record.** The production m5 -> m6 step is REAL,
  but it is smaller than recorded and mechanically different: **+1.58 to +3.04
  ms/round (2.8-5.4 MUE)**, rising with both m and KV length. Its mechanism is
  **dispatch and launch count, not copy bandwidth**: 4 extra small copies (2
  query-chunk `g2_copy`, 2 concatenate `gg2_copy`) plus 1 extra SDPA call, which
  at kv >= 1024 means 2 extra 2-pass kernels. The total copied bytes are a few
  tens of KB per layer, which is far too small to explain the time.
- **The split is a WIN, not a wall** (advisor decision-tree item 3 confirmed).
  It saves **1.14-2.31 ms/round** against the unsplit composed fallback, and it
  is also the exactness fix for the top-2 value drift documented at
  `AttentionUtils.swift:106-122`. Removing it would be worse on both axes. The
  real boundary is the vector-mode threshold `qL * gqa <= 32`, and FINDING 489's
  1.403-2.772 ms/round should be re-labelled as the cost of the composed
  fallback that the split already avoids.
- **Prompt or M5 transfer risk.** Moderate, and stated. The host is
  `applegpu_g16s`, not the ranked M5 runner. The census counts dispatches, which
  is a property of MLX source logic, so it should transfer exactly. The timing
  gap is a dispatch-latency effect, and its magnitude may differ on M5. Each
  probe call also includes one blocking `eval()`, which is about a 250 us floor.
  Differences are unaffected, because every arm pays exactly one eval, but
  in-path decode has no per-layer eval barrier, so the dispatch-latency portion
  of the production step is an **upper bound**. The `vn_copy` bandwidth penalty
  of the fix is real regardless of eval placement. The probe passes the same
  array for keys and values in `cache.update`, so the cache-write cost is
  identical across arms and the verdict is unaffected.
- **Smallest useful next action.** Reprice the widthSixWall ledger entry from
  "cache-slice copy" to "extra dispatch count", then attack the dispatch count
  instead of the copies. Follow-up 1 below is the cheapest.
- **Recommendation: close.** Label **Not useful / repriced**. Do not build the
  two-append fix.

## Suggested follow-ups (NOT implemented)

1. Emit the two query chunks directly from the fused
   `qwen35AttentionQKRMSRoPE` kernel as two row-contiguous outputs. That removes
   `g2_copy` x2 per full-attention layer with no change to the SDPA calls. It
   touches `Qwen35.swift`, which is Askeladd's E189 file, so sequencing matters.
2. Remove `concatenated([outA, outB])` by writing both SDPA outputs into one
   preallocated buffer, or by keeping the chunks separate through the row-wise
   `o_proj`. That removes `gg2_copy` x2 per full-attention layer.
3. Overlap chunk A and chunk B on two MLX streams. They are independent given
   the already-committed cache, and both are heavily under-occupied (24x5 and
   24x(m-5) threadgroups). This is the only candidate that attacks the
   KV-proportional part of the step.
4. Investigate reshaping `[1, 24, m, 256]` into a batch-major form to lower the
   effective `gqa` in `supports_sdpa_vector` (`qL * gqa <= 32`), so that one
   fused vector call could serve m = 6..9. This needs care, because broadcast
   keys would probably fail `kv_copy_unless` and reintroduce a real KV copy.
   This is the "vector-mode variant for qL*gqa in (32, 48]" question that the
   advisor raised as a SEPARATE experiment.
