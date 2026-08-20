SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mtp_seconds_per_token_prefill_removed","available":true,"value":0.023942},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

# E59 — route M=5 QMV to one NA=5 group (`<T,5,3>` to `<T,5,5>`)

## Headline: two independent instruments agree on the conversion chain

The most useful thing this experiment produced is not `t55`. It is a measured
conversion from a synthetic kernel cell to a real decode leg.

```
leg instrument    -21.821 +/- 1.488 %      (512-token decode, converted to cell units)
cell instrument   -20.209 +/- 0.165 %      (whole-table QMV probe at M=5)
difference         -1.612 +/- 1.497 %,  t = -1.08,  consistent at 2 sigma
```

The two instruments share no code. One times a synthetic QMV probe table; the
other times a 512-token decode through the full MTP session and subtracts
prefill. They were run in separate sessions on separate days of campaign time.
Agreeing at two sigma turns `cell -> round cost -> leg` from an assumption into
a measurement, so every future kernel arm can be priced from a cheap cell probe
before anyone spends a leg session on it.

The conversion needs two session-local constants, and both must come from the
session being converted, not from a remembered value:

| constant | value on this host | how it was obtained |
| --- | ---: | --- |
| M=5 share of rounds | 0.047906 | the session's own width histogram, `{2:1, 4:5, 5:5, 6:23, 7:4, 8:6, 9:34}` over 78 rounds |
| round fraction of leg | 0.7562 | the session's own 0.024195 / 0.031996 |

`round-cost % / 0.047906` gives −21.821 %. E1 puts the M=5 round at 115,691 µs
against a measured shipped cell of 120,067 µs, so at M=5 the round essentially
*is* the QMV verify and no extra QMV-share factor belongs on that basis.

### The result being converted

- **M=5 cell:** −20.209 % ± 0.165 %, t = −122.18
- **512-token leg:** −0.7689 %, t = −10.68, against a 0.2423 % same-arm bar
- **512-token round cost, prefill removed:** −1.0453 % ± 0.071 %, t = −14.67,
  against a 0.1629 % bar
- **Gate:** the preregistered divisor gives `implied_cell_pct = −0.7689 / 0.0668
  = −11.510 %`, which is past the −6.0 % advance threshold with a stable sign.
  **`t55_cell_gate = advance`.** The corrected divisor gives −21.821 %. The
  verdict is the same under every divisor in the range, so it does not depend on
  the choice.

Per the advisor's adopted rule, the **preregistered** divisor decides the gate,
because a gate that moves after the fact is not a gate, and the **corrected**
conversion is reported beside it.

## Identity

- **Student / branch:** `qwen-thorfinn` / `qwen-thorfinn/e59-m5-rowblock-r2-route`
- **Hypothesis and target cost:** at width M=5 the shipped QMV route splits five
  output rows into two sequential row blocks `{3,2}`. The second block is a cheap
  NA=2 tail that still pays a full x-group load. Routing M=5 to one NA=5 group
  makes `TAIL = M % IPG = 0`, so the route becomes a single `[5]` group and that
  second pass disappears. The measured M=5 QMV cell cost was 120.07 ms per probe,
  the largest single-width discontinuity in the shipped width curve.
- **Decision:** green locally.
- **`BASE_SHA`:** `31e67cb82c0e78c04c3d36b401ae213aa9e540e8`. The base moved
  three times during this experiment. I merged `44bb38d5` in `74dbf34a`,
  `45b4f3a8` in `391ed5f`, and `31e67cb8` in `2227a9c5`, always by merge and
  never by rebase. The assignment's original `989596895b` is retired.

  The third move landed while the candidate arm was timing. It contains
  `53d9d58`, which merges the promoted organizer frontier `80021bc0`, and
  `31e67cb8`, which adds a symbol-table witness to the worker guard. It changes
  three files: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` by +70 lines,
  `senpai/rebuild-and-assert-worker.sh`, and `senpai/experiment-runbook.md`.
  **The two scored twins are byte-identical between `45b4f3a8` and
  `31e67cb8`**, so the candidate surface against the new base is still exactly
  the two-character diff. The guard upgrade needs no change here: the new file
  documents `--require` and `--forbid` on the string table as the correct
  witness for a Metal JIT arm, which is what this is, and reserves
  `--require-symbol` for Swift arms.

  **What I replayed on `31e67cb8`, and what I did not.** I replayed both
  close-out chain arms, so the content assertion, the Swift control comparison,
  the `--local-submit` correctness receipt, and all five gates are evidence on
  the current base. I did **not** replay the 512-token matched leg session that
  produced the −0.7689 % headline. That session is about 40 minutes, the advisor
  asked for no new arms, and the new base touches neither scored file nor the
  QMV path. The leg session is an A/B inside one session, so a change to the MTP
  session code applies to both arms and cancels in the contrast. The effect size
  in this report is therefore measured on `45b4f3a8` and stated as such.
- **`UPSTREAM_SHA`:** `80021bc03e4b270f7dfef5b4425107bfc57b8d70`. My earlier
  report said this was not resolvable; that was wrong and I retract it. Two
  sources disagree, and the commit graph settles it:
  `senpai/frontier-state.json` still records
  `organizer.syncedCommit = 0c90733d383f6b987a29682bf9eb9458a6172bfa` at
  `observedAt 2026-08-20T05:15:00Z`, but the base's own merge `53d9d58` at
  06:01Z brought in organizer main `80021bc0`, which is now an ancestor of HEAD
  and is the `sourceRef` of the live promoted crown `9d5569bb` at score
  `3.25187972017987`. **`frontier-state.json` is stale by one organizer sync**
  and should be refreshed by whoever owns it; I did not edit it, because
  campaign state is advisor-owned.
- **Candidate commit:** `b757237`. Result head: the submitted head of this PR.
- **Yukon promoted submission:** the crown moved during this experiment to
  hadakang's `80021bc0`. The advisor relays that it differs from the row it
  displaced by one line in the `note` string of `mtp-head.manifest.json`, worth
  +0.0623 %, which is a resample and not a measurement. I did not query Yukon;
  the advisor owns submission.
- **Base content that is not mine.** The merged base already carries askeladd's
  `t6` (PR #64, `0040ff45`), which moves `case 6` from `<T,6,3>` to `<T,6,6>` and
  raises the wide bound to `NA <= 6`. After the merge the routing table reads
  `case 5 -> <T,5,5>` (mine) and `case 6 -> <T,6,6>` (his), and `ceil(M/IPG)` is
  `1 1 1 1 2 2 2` for M = 3..9, a single boundary between M=6 and M=7. **All
  causal measurements in this report were made on `44bb38d5`, before `t6`
  merged.** They are unaffected, because `t6` changes M=6 only and my cell
  contrast isolates M=5, but the composed `t55 + t6` leg effect is a separate
  question that askeladd is measuring in PR #69. I did not spend a leg on it.
- **Submitted candidate files (exactly two):**
  - `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
  - `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`

  The diff against the current base is one character in each file, at
  `quantized.h:1939` and `quantized.cpp:1952`:
  `qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>` to `<T, 5, 5, true>`.
  Byte growth is zero.
- **Supporting research files, not submitted:** everything under `research/`.
  This experiment added or changed `research/e59_ols.py`,
  `research/e59_worker_digest.py`, `research/e59_rung4_cells.py`,
  `research/e59_e2e_run.sh`, `research/e59_e2e_analyze.py`,
  `research/e59_wandb_log.py`, `research/e59_gates.sh`,
  `research/e59_final_chain.sh`, `research/e49_run_leg.sh`, and the artifacts
  under `research/e59-artifacts/`.
- **MTP head provenance and draft policy:** declared head
  `mtp-head-declared-run/`, tree digest
  `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`, reported
  as `uses_pinned_mtp_head=true`. The head is unchanged. This experiment changes
  no drafting policy, no depth schedule, and no acceptance rule. Depth 8
  throughout.
- **Token window, fixture, reference source, harness:** `local`. 512-token leg
  session on `correctness_prompts/public_longcopy_gate_english_512_256.json`;
  pre-submit `--local-submit` at its default 128 decode tokens with
  candidate-generated reference rows.
- **Exact cell:** affine 4-bit, group size 64, bfloat16 activations. Kernel
  family `qmv_fast_crossrow_affine4_g64_m`, entry point
  `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0`. Host grid `bn=8`, `bk=32`,
  `group_dims(32,2,1)`, `grid_dims(M,(N+7)/8,B)`, so `ntg.x == M`, two
  simdgroups, eight output rows per `tid.y`. Source form: JIT-compiled from the
  string inside the `mlx-generated/quantized.cpp` twin, so both twins move
  together and the runtime-effective artifact is the worker binary. Runs on the
  ranked M5.
- **Official causal path:** the change affects the MTP denominator only.
  `senpai/verify-ranked-score-boundary.sh` passes with "ranked numerator is
  pinned baseline; candidate edits affect the MTP denominator only". The serial
  leg decodes at depth 0, therefore at M=1, and never reaches `case 5`.
- **Scored-path reachability:** the rung-4 parity gate fired three negative
  controls against the live scored worker. `t55_lane_perturb` changed 16 rows at
  widths [5,9], `t55_row_drop` changed 8 rows at M=5, and
  `m5_rowsplit2_coverage_drop` changed 8 rows at M=5, while `t55` itself changed
  0/192. A control that perturbs only the M=5 route can move scored output only
  if the scored worker executes that route.

## Naming: my `m5_rowsplit2` is not askeladd's `shipped_rbx`

Both arms were called `rbx`. They are different mechanisms and the ledger must
not conflate them. In this report I use **`m5_rowsplit2`** for mine. Its
arm-table key in `research/e59_arms.py` remains `m5_rbx`, and every artifact
under `research/e59-artifacts/` uses that key, so the mapping is:

| ledger name | arm-table key | mechanism |
| --- | --- | --- |
| `m5_rowsplit2` | `m5_rbx` | IPG=5, **one** NA=5 group split across two threadgroups in the **output-row** dimension at `rows_per_simd = 2`, via a fourth template parameter on `_wide`. Carries `static_assert(M % IPG == 0 && M / IPG == 1)` and returns early for `tid.x >= 2`. |
| `shipped_rbx` (askeladd) | — | Keeps the shipped IPGs and only literalises the first input row, branching on `tid.x` before passing a literal into `_wide`. No divisibility constraint. |

Consequences I accept from the advisor's review: my −13.431 % at M=5 is not
comparable to his −2.7288 % at M=9; the earlier "rbx at M=5 is −0.658 % of
ranked QMV" was about 5x low for my arm and is nearer −3.2 %, so `t55` still
dominates but by 1.5x rather than 4.0x; and my wrapper cannot build the
mixed-group ladder while his can.

## Evidence

- **Host, chip, memory profile, toolchain, thermal policy:** Apple M4 Pro,
  48 GiB (`hw.memsize = 51539607552`), macOS 26.5.2, Swift 6.3.3
  (swiftlang-6.3.3.1.3), target `arm64-apple-macosx26.0`. This host is below the
  64 GiB full-profile minimum, so every run forced
  `DARKBLOOM_STARTUP_MEMORY_PROFILE=full`. The real 40 °C cool gate was enforced
  on every timed phase reported here. No arm used `MLXFAST_LOCAL_COOL_GATE=0`.
- **Exact commands:**

  ```bash
  # whole-table cell palindrome
  research/e59_e2e_session.sh shipped:e59-r4c-a1 t55:e59-r4c-a2 m5_rbx:e59-r4c-a3 \
    m5_rbx:e59-r4c-a4 t55:e59-r4c-a5 shipped:e59-r4c-a6 \
    --widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10

  # 512-token leg session
  research/e59_e2e_session.sh shipped:e59-r4-w0 shipped:e59-r4-l1 t55:e59-r4-l2 \
    t55:e59-r4-l3 shipped:e59-r4-l4 t55:e59-r4-l5 shipped:e59-r4-l6 \
    shipped:e59-r4-l7 t55:e59-r4-l8 --tokens 512 --warmup-first

  # close-out chain, one arm at a time
  research/e59_final_chain.sh base
  research/e59_final_chain.sh candidate --local-submit
  ```

- **Cheapest real falsification gate:** the rung-4 parity gate,
  `all_passed=True controls_fired=3/3`. Every leg also ran a binary probe
  reporting "routing at widths 2..9 present and exclusive", which fails loudly
  if the arm never reached the binary.
- **Exact-token and row-ledger verdict:** every arm matched. In the 512-token
  session `all_arms_token_exact` and `all_arms_row_ledger_closes` are both
  `True`, and both arms produced the identical width histogram
  `{2:1, 4:5, 5:5, 6:23, 7:4, 8:6, 9:34}`.
- **Divergent tokens or failure category:** none.
- **Peak RAM or artifact size:** not separately instrumented. The candidate adds
  no allocation; it selects a different template instantiation of an existing
  kernel. `mtp-head/` is unchanged, so exempt-head bytes stay at 2410.
- **Official status and score:** not submitted. The advisor owns
  `senpai/submit-official.sh`.

### Primary causal measurement — 512-token matched leg session

Eight timed legs, palindrome-balanced, arm-position sums 18/18, so arm and
position are exactly orthogonal. Order:
`shipped, t55, t55, shipped, t55, shipped, shipped, t55`, after one discarded
warm-up leg. Every leg passed the real 40 °C gate; entry temperatures spanned
43.290 to 44.085 °C.

| Metric | Baseline (`shipped`, n=4) | Candidate (`t55`, n=4) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.074346 | 0.074331 | −0.0196 % |
| MTP seconds/token | 0.031996 | 0.031750 | **−0.7689 %** |
| MTP seconds/token, prefill removed | 0.024195 | 0.023942 | **−1.0453 %** |
| prefill seconds | 3.9938 | 3.9974 | +0.09 % |

Regression on `mtp_seconds_per_token` (n=8, dof=5, residual sd 3.259e−05):
`arm[t55] = −2.460049e−04 ± 2.304e−05`, t = **−10.68**; position term
+4.141076e−06 ± 5.028e−06, t = +0.82, so there is no detectable drift.
Regression on the prefill-removed metric (residual sd 2.439e−05):
`arm[t55] = −2.529218e−04 ± 1.724e−05`, t = **−14.67**; position t = +0.06.

Per-leg `t55` deltas: −0.8184 %, −0.8591 %, −0.8198 %, −0.5781 %. Sign stable.

The serial channel is the null channel: it moved −0.0196 % against a 0.0676 %
bar, so the instrument is quiet where it should be quiet.

**Same-arm session null.** Two legs of the same arm from the same binary bound
the noise. The measured effect exceeds the widest same-arm gap by more than 3x
on the leg metric and 6x on the round-cost metric.

| field | sep1 | sep2 | sep3 | sep5 | sep6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP leg | 0.0410 | 0.1086 | 0.2438 | 0.2835 | 0.2423 |
| MTP round-cost | 0.0509 | 0.1502 | 0.1950 | 0.2133 | 0.1629 |
| serial | 0.1194 | 0.1548 | 0.1073 | 0.1169 | 0.0676 |

### Whole-table cell measurement

Six whole-table legs, palindrome, widths 1..10, 21 reps, inner 10. All six
passed the real 40 °C gate; entry spread 0.493 °C. All six reported
`BINARY PROBE OK`.

| tag | arm | M1 | M2 | M3 | M4 | **M5** | M6 | M7 | M8 | M9 | M10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | shipped | 60.770 | 65.073 | 71.961 | 82.081 | **119.947** | 128.496 | 138.116 | 148.621 | 163.741 | 271.156 |
| a2 | t55 | 59.977 | 64.339 | 72.200 | 82.343 | **95.950** | 128.443 | 138.288 | 149.153 | 164.193 | 271.086 |
| a3 | `m5_rowsplit2` | 60.710 | 65.030 | 72.043 | 82.095 | **104.001** | 128.220 | 138.132 | 149.113 | 164.004 | 271.025 |
| a4 | `m5_rowsplit2` | 60.302 | 64.991 | 72.214 | 83.431 | **103.880** | 128.106 | 138.075 | 148.936 | 163.850 | 271.158 |
| a5 | t55 | 60.047 | 65.759 | 73.062 | 82.324 | **95.655** | 128.227 | 138.236 | 149.012 | 164.266 | 271.115 |
| a6 | shipped | 60.963 | 65.166 | 72.220 | 82.240 | **120.186** | 128.630 | 138.143 | 148.997 | 164.844 | 271.358 |

Cell OLS at M=5 (`t_ms ~ arm + leg_position`, n=6, dof=2, residual sd 0.1654 %
of base, drift +0.0045 % per leg):

| contrast | effect | t | worst untreated width | verdict |
| --- | ---: | ---: | ---: | --- |
| `t55` vs `shipped` | **−20.209 %** | −122.18 | 1.404 % (M=1) | clears |
| `m5_rowsplit2` vs `shipped` | **−13.431 %** | −81.20 | 0.733 % (M=4) | clears |
| `m5_rowsplit2` − `t55` | +6.778 % | +40.98 | 0.816 % (M=1) | clears |

Every untreated width moved by less than 1.5 %, so the effect is confined to the
treated cell.

**Derived `r=2` tax at NA=5.** `t55` 95.8025 ms to `m5_rowsplit2` 103.9405 ms is
**+8.495 %** for a parallel x-group. Rung 3 measured +14.351 % for `_rb2` and
+8.318 % ± 1.007 % for `_rbx`. This replaces the value E44 inherited at NA=4
(10.54 %).

## Why the effect beat the preregistered band

The advisor preregistered −8 % to −14 % at the cell and I measured −20.209 %.
The band was computed from a pure model that prices a removed group at the
**average** group rate. `t55` does not remove an average group; it removes the
**cheapest** one, the NA=2 tail running at 218.5 GB/s. Any change that deletes a
narrow tail is therefore under-credited by that model.

The M=5 bandwidth read is a second, independent proof of the same defect. The
shipped `[3,2]` route measures 120.036 GB/s single-pass but models at
240.6 GB/s, which is 105.6 % of the 227.769 GB/s stream peak. A route cannot
exceed the stream peak, so the second x-group must be partly cache-served. That
reproduces E1's finding that the extra x-group costs 64–94 % of a full pass,
mean about 0.80. `t55` `[5]` measures 150.438 GB/s and models at 151.1 GB/s,
66 % of peak, with no second pass to explain away.

The direct evidence that the mechanism is deletion rather than overlap is the
`m5_rowsplit2` contrast: keeping two blocks and only running them in parallel
gets −13.431 %, comfortably inside the preregistered band, while deleting the
block gets −20.209 %. The 6.778 pp gap is the price of the second x-group.

A third supporting fact comes from edward's E63: the register law `22 + 20·NA`
is exact through NA=5 and breaks at NA=6, where a second alloca appears and peak
live registers cross the AGX 128 boundary. `t55` runs M=5 at NA=5, the last cell
below that step, which is part of why it over-performs.

## Preregistration and where I was wrong

Frozen prereg: `research/e59-artifacts/e59-rung4-prereg.json`,
`2026-08-19T23:21:32Z`, sha256
`22a3e9a717f1971bd0aff93d2548273081abb45778cf5fc95e9e2f5d059c11c7`.

| prediction | predicted | measured | error |
| --- | ---: | ---: | ---: |
| frozen prereg, leg | −0.7564 % | −0.7689 % | 1.7 % |
| frozen prereg, round-cost | −0.9875 % | −1.0453 % | 6 % |
| my PR comment c11 restatement, leg | −1.1088 % | −0.7689 % | 44 % |
| corrected forward model, leg | −0.7322 % | −0.7689 % | 5 % |
| corrected forward model, round-cost | −0.9682 % | −1.0453 % | 8 % |

My c11 restatement missed by 44 % and the fault is mine in two places. I used
0.0668 for the M=5 share when this session ran 0.047906, a factor of 1.394. And
I used `QMV_SHARE_OF_LEG = 0.82127` as if it were the round-fraction-of-leg,
when the measured fraction is 0.024195/0.031996 = 0.7562, a factor of 1.086.
With both corrected the model gives −20.209 × 0.047906 × 0.7562 = −0.7322 %.

The frozen prereg, written before stage A, was already closer than my later
restatement. That is a useful warning: my "improved" mid-flight model was worse
than the one I had committed to in advance.

**`psi_mtp_ranked_leg = 0.82127` is retracted as a leg-conversion factor.**
Nothing should use it again.

The amendment `e59-rung4-prereg-amendment.json` predicted `t55` −0.751 %,
`m5_rowsplit2` −0.501 %, gap −0.250 %, and `predict_tie=false`. The measured
cell gap of −7.829 % vindicates the refusal to predict a tie.

## Arm certificate: assert content, not layout

An earlier version of this report certified the arms with the worker
`__TEXT,__text` digest. **I withdraw that as a content witness.** Ledger 202(I)
shows the digest tracks link-time layout: two builds of the same tree produced
different digests, and two builds of different trees produced the same one.

Two things replace it.

1. **The paired digest still stands as a falsifiable check**, because it can
   fail in a way a single digest cannot: across arms `__TEXT,__text` was
   *identical* (`5ea9a670…`) while `__TEXT,__cstring` *differed* (`shipped
   95b28239…`, `t55 b7818020…`). One half must stay the same and the other must
   change, which is the correct signature for a JIT-string-only edit.
2. **String assertion inside the built worker**, which is the real witness. This
   matters because ledger 202(H) shows `--local-submit` can silently time a
   stale worker and still report `passed: true`: the wrapper's `METALLIB-GUARD`
   at `benchmark-qwen-mtp.sh:200-204` extracts only
   `metallib_rebuild_required()` and not the sibling `swift_build_required()`
   that guards `.build-worker/release/mlxfast-runtime-worker`. For the
   `quantized` family the runtime-effective source is the JIT string compiled
   *into* the worker, so the half the wrapper refreshes is exactly the half that
   does not govern.

   My first `--local-submit` run predates that finding and I therefore treat it
   as void. The result reported below comes from `research/e59_final_chain.sh`,
   which rebuilds the worker and asserts by content before timing, and asserts
   again afterwards.

## Close-out chain

Both arms run through `research/e59_final_chain.sh`, which swaps the two scored
files to the arm, rebuilds the metallib into every build root, rebuilds the
worker, asserts the JIT kernel string by content inside the built binary, and
then runs the full Swift suite with runtime tests enabled.

### Base arm, the control

```
worker_mtime  2026-08-20T05:46:30Z
worker_sha256 34205e1bfe580f1edfe3b145088d9d3d93f2b22f526258a2d6613a86d106ac3a
ok require '<T, 5, 3, true>': 1
ok require '<T, 6, 6, true>': 1
ok forbid  '<T, 5, 5, true>': 0
ok extraction: 80399 strings
rebuild-and-assert-worker: PASS
```

Swift suite: 688 tests in 49 suites, 41 issues, **10 failing tests**.

### Candidate arm

```
worker_mtime  2026-08-20T05:50:27Z   (assert before timing)
worker_sha256 fab69831ed1d7f55bb6b56f0fbe6b5a379d77202bed63fee8fb7585a0927a8a8
ok require '<T, 5, 5, true>': 1
ok require '<T, 6, 6, true>': 1
ok forbid  '<T, 5, 3, true>': 0
ok extraction: 80399 strings
rebuild-and-assert-worker: PASS

worker_mtime  2026-08-20T05:50:27Z   (assert after timing, --no-build)
worker_sha256 fab69831ed1d7f55bb6b56f0fbe6b5a379d77202bed63fee8fb7585a0927a8a8
worker_unchanged_across_timing: true
```

The two worker digests are equal, so the binary that produced the timings below
is the same binary whose JIT kernel string was asserted by content. That closes
the ledger 202(H) stale-worker trap on both sides of the measurement.

Swift suite: 688 tests in 49 suites, 40 issues, **9 failing tests**.

The candidate failing set is a strict subset of the base control's 10. The one
test that differs is
`phaseStartAllocatorResetLeavesExactlyEmptyCacheWhenRuntimeTestsAreEnabled`,
which failed in the base arm with `runtime worker failed to clear the MLX
allocator cache at phase start` (`RuntimeWorkerSupportTests.swift:179`) and
passed in the candidate arm after 9.024 s. I read this as host memory pressure
on a 48 GiB machine, not as an effect of the change: a one-character template
width cannot decide whether the allocator cache drains at phase start. I claim
no credit for it. The honest statement is **no new failure**, not a fix. The
other 9 failures are the same environment and provenance failures both arms
share.

`--local-submit`, run between the two assertions, exit code 0:

| field | value |
| --- | --- |
| `passed` | `true` |
| `all_tokens_matched` | `true` |
| `residual_divergence_count` | 0 |
| `public_drift_tripwire_passed` | `true` |
| `uses_pinned_mtp_head` | `true` |
| `head_provenance_sha256` | `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` |
| serial rows checked | 128/128, depth 0, 0.097499 s/token |
| MTP rows checked | 134/134, depth 8, 20 rounds, 0.055101 s/token |
| `accepted_draft_rate` | 0.9561 |
| `effective_mean_draft_len` | 5.7 |
| local speedup | 1.7694 |

Reference rows: `rows=129 seed_tokens=512 reference_seed_token=271
self_consistent=true (replayed 1 row bit-identically) chain_contradictions=0`.

All three cool gates were the real 40 °C gate, not a bypass: 39.8 °C before
reference generation, 40.0 °C before the serial control, 39.8 °C before the MTP
leg.

That 1.7694 is not a score. `rankable` is `false` and the recorded reason is
`candidate-generated reference rows; official scoring disabled; ranked run is
the only authority`. Both legs also use the candidate build, so this number is a
correctness receipt and a sanity check, not evidence for the effect size. The
effect size comes from the matched leg session above.

### Submission gates

All five now pass, recorded in `research/e59-artifacts/e59-gates.json` with
`all_passed=true`:

| gate | verdict |
| --- | --- |
| `assignment_scope` | PASS, 2 submitted paths against `BASE_SHA=45b4f3a8` |
| `editable_budget` | PASS, `source=2458949/3000000 headroom=541051 growth=0/262144 exempt=2410 files=154` |
| `twin_audit` | PASS, 29 runtime-effective twins, 1 allowlisted comment-only waiver |
| `scored_surface` | PASS, every unscored shipped delta acknowledged |
| `ranked_score_boundary` | PASS, candidate edits affect the MTP denominator only |

`scored_surface` was blocked in the previous report and is now unblocked. See
the blocker section for how, and for the stale pin it exposed.

## Negative and null results

- **Command-buffer geometry dose: null.** Both dose pairs fail the 3.0 % floor.
  Moderate (`ops=8` against `ops=50`) gives −0.251 %; extreme (`ops=1` against
  `ops=50`) gives +0.534 %. All four profile probes pass: `=bogus` rc=133 with 0
  notices, `=low` rc=1 with 1 notice, `=full` rc=1 with 0 notices, unset rc=1
  with 1 notice, so this 48 GiB host defaults to the low profile and `=full` is
  load-bearing. Honest label: **unproven by dose, proven by source reading.**
  alphonse's E62 independently supports this: he measured the MLX per-commit
  cost at 11.24 µs, 35 % of the ledger 199(A) ceiling, and showed the OPS term
  binds while the byte cap is inert, which is exactly why my cap-1 dose moved
  nothing.
- **Occupancy hypothesis: retracted.** The E61 ballast dose shows +15 registers
  at fixed routing costs only +0.38 %. Registers are not the width cliff. My own
  census agrees: `shipped` M5 uses 87 registers and `t55` uses 125, both under a
  table maximum of 129 that E55's `case 9` already pins. The `ceil_only` arm is
  scientifically void and I did not run it.
- **`worker_low_memory_notices = 0` for rungs 2b and 3: retracted.** The field
  was absent, not zero, so the evidence was vacuous.
- **`SHIPPED_IPG[9] = 3` was wrong.** The base ships `<T,9,5>`, which is a
  `{5,4}` mapping. The M=9 lane-perturb control firing at width 9 confirms it
  independently.

## Blockers and defects found

1. **`--local-submit` failed three times before I found the lever.** Attempts at
   04:58:24Z (job `2780bf6c`), 05:01Z (job `7576f2d7`), and a `--local-iterate`
   control at 05:06Z (job `35458761`) all died at model load with
   `mlxfast-worker: low-memory startup profile engaged (physical memory 48 GiB
   is below the 64 GiB full-profile minimum)`. No `mtp-verify: generate` line
   ever appeared. Setting `MLX_MAX_MB_PER_BUFFER` and `MLX_MAX_OPS_PER_BUFFER`
   cannot help, because `applyQwenMTPStartupMemoryProfile`
   (`QwenRuntimeMTPWorker.swift:479`) force-sets 128 MiB and 64 ops with
   `overwrite=1` whenever the low-memory profile resolves.
   `DARKBLOOM_STARTUP_MEMORY_PROFILE=full` is the only working lever on this
   host. The `exit_status=15` in the error text is an artifact:
   `workerExitDiagnostic()` calls `stopRuntimeWorkerProcess` while the process
   is still running, so the SIGTERM is the harness's own.
2. **The scored-surface gate is unblocked, and it caught a stale pin on the
   base.** The gate needs two objects that exist only in the organizer
   repository: the scored submission commit
   `2b0c36a078b7660c9215adee933336ff46da25af` and the promoted frontier tip.
   `git fetch origin` answers `not our ref` for both because `origin` is our
   fork, this checkout has no `upstream` remote, and terminal policy blocks `git
   remote`. The fix needs no remote: fetch by URL into refs we own.

   ```bash
   git fetch https://github.com/Layr-Labs/qwen-3.8-mtp-challenge \
     +2b0c36a078b7660c9215adee933336ff46da25af:refs/e59/scored-commit \
     +main:refs/e59/frontier-main
   export SCORED_GATE_FRONTIER_REF=refs/e59/frontier-main
   ```

   `research/e59_gates.sh` now does this itself, so the gate is reproducible in
   any student checkout. `refs/e59/frontier-main` resolves to
   `80021bc03e4b270f7dfef5b4425107bfc57b8d70`, the current crown. This is a
   fetch only. Nothing merges organizer work into the experiment.

   With the gate finally reading the diff, both `FRONTIER-TAKEN` entries verify
   byte-identical, and both `FRONTIER-PLUS-PINNED-DIFF` entries failed against
   their pinned digest `08c42cf7…`. **That pin was already wrong on the base,
   before my change.** Digests of the `-U0` content lines against the frontier:

   | subject | digest |
   | --- | --- |
   | pinned in `ACK_UNSCORED` | `08c42cf7891adc91…` |
   | base `45b4f3a8` | `76659e5e5db34baa…` |
   | this candidate | `d1c64484de7b1821…` |

   t6 merged its `case 6` and `NA <= 6` hunks without re-pinning, so the
   tripwire was already red and would have stayed red no matter what I did. The
   base declares three content-line pairs against the crown: the assert widened
   from `NA <= 4` to `NA <= 6`, `case 6` moved from `<T,6,3>` to `<T,6,6>`, and
   `case 9` moved from `<T,9,3>` to `<T,9,5>`. My change adds a fourth, `case 5`
   from `<T,5,3>` to `<T,5,5>`.

   I re-pinned to the observed `d1c64484…` in the same commit that introduces
   my hunk, which is what the gate instructs, and I rewrote both rationale
   strings to declare all four pairs and to attribute t6's hunks to t6 rather
   than to me. The rewritten text also drops two claims that went stale: the old
   text said the assert was relaxed to `NA in [2, 5]`, and it did not mention
   `case 6` at all.

   One fact in that rationale deserves to be read directly rather than buried.
   The crown reverted E27 because whole-table register widening cost 0.3321 % of
   score, and `t55` raises the production entry
   `affine_qmv_fast<bfloat16_t,64,4,false>` from 181 to 183 registers, which is
   exactly E27's entry count. So this candidate does reopen the channel E27 was
   reverted for. The evidence that it does not cost anything here is direct and
   local: the matched serial control moved −0.0196 % against a 0.0676 % bar, and
   the isolated `M=1` cell is 1.404 % *faster* under `t55`, not slower. The
   kernel-wide maximum stays 129, set by `case 9`, not by me. I state this as a
   named residual risk that the ranked M5 run should settle, not as a solved
   question.
3. **Stale source that I did not fix, to stay in scope.**
   - `quantized.h:1154` still says "IPG = ceil(M / ceil(M / 4)) … at NA <= 4",
     which E55 contradicts and which the `NA <= 6` bound now contradicts twice.
   - The `case 8` comment block at `quantized.h:1953-1966` contradicts the code
     it sits on. It argues for "3+3+2, not 4+4", claims "IPG 3", and states the
     wide helper's range is "[2,4]". The shipped line is
     `qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>`, which is IPG=4 and
     therefore exactly the `{4,4}` split the comment argues against, and the
     helper's range is now `[2,6]`. Anyone reading that block to plan an M=8
     experiment would start from three wrong facts. I found this while checking
     the feasibility of the proposed `t8` arm. I did not fix it, to stay in
     scope, but it is more misleading than the other stale comments because it
     reads as a deliberate design rationale rather than an outdated aside.
   - `SHIPPED_IPG[9] = 3` in `research/e54_arms.py` is stale, so every `m9_*`
     and `iso_m9_*` arm fails loudly.
   - The retracted `0.0629 %` null floor is still hard-coded in
     `research/e54_gap_decomposition.py:92`,
     `research/e58_buffer_analysis.py:28`, `research/e58_tax_analysis.py:33`,
     `research/e58_wandb_log.py:58`, and `research/ranked_noise.py:83`.
   - `research/twin_audit.py` waives comment-only divergence, so a pure comment
     addition to one twin would pass unnoticed.
4. **Two stale base defaults that I did fix.** `research/e49_run_leg.sh` and
   `research/e59_gates.sh` both defaulted to the retired base `9895968…` and now
   resolve `origin/senpai/qwen38-mtp-r1`. A `log_geometry` KeyError was also
   silently suppressing the geometry summary; that is fixed.
5. **Ledger 193 noise, corrected.** One ranked run has sd 0.756 %, a difference
   of two has sd 1.069 %, and the 95 % single-pair threshold is 2.10 %. The
   candidate-to-serial noise ratio is 3.62x. My earlier +0.283 % MDE claim was
   wrong by a factor of 7.4 and I retract it. The crown moving +0.0623 % on a
   byte-identical resample is a live demonstration of the same point.
6. **Rung 2b race artifact, for the record.** In that earlier session leg 1 was
   built from a different commit than legs 2 and 3, which is why the leg runners
   now guard on scored-file identity and whole-worktree cleanliness. Those rung
   2b numbers are directional only.

## Conclusion

- **What happened and why:** routing M=5 to a single NA=5 group removes the
  second row block entirely. The M=5 QMV cell gets 20.209 % ± 0.165 % cheaper,
  and a real 512-token decode gets 1.0453 % ± 0.071 % cheaper per token once
  prefill is removed, at t = −14.67, with exact tokens and a closed row ledger.
- **Evidence for or against the mechanism:** four independent lines agree. The
  `m5_rowsplit2` contrast separates deletion from parallelisation and prices the
  second x-group at 6.778 pp. The bandwidth model shows the shipped route
  exceeding the stream peak, which is only possible with a partly cache-served
  second pass. The two instruments agree within 2σ. The paired section digests
  plus the in-binary string assertion show that nothing but the kernel source
  string changed.
- **Prompt and M5 transfer risk:** moderate, and it favours the candidate. This
  host ran an M=5 share of 0.047906, while the ranked shares are 24.1 % at M=5
  and 33.4 % at M=6, so the local session under-weights M=5 by about 2.04x. The
  cell effect transfers directly; the leg effect measured here is a **floor**
  for the ranked leg, not a ceiling. Against that, this is one public fixture on
  an M4 Pro, and the ranked box is an M5 whose QMV cell timings I cannot
  reproduce locally.
- **Value:** `t55` alone prices at −0.639 % of ranked QMV against a deficit
  needing −0.646 %, so it is a hair short on its own and clears comfortably when
  composed. With `t6` the advisor's pricer gives −3.40 % ranked QMV, −2.808 %
  candidate leg, and +1.942 % published score at sd 2.57.
- **Close-out chain: green.** Both arms rebuilt and asserted the JIT kernel
  string inside the worker binary by content. The candidate asserted again after
  timing with the same worker digest, so no stale binary was timed. The
  candidate Swift failing set is a strict subset of the base control's.
  `--local-submit` passed with exact tokens, a closed row ledger, and three real
  40 °C gates. All five submission gates pass, including the scored-surface gate
  that could not run before.
- **Named residual risk:** `t55` restores E27's 183 entry registers on
  `affine_qmv_fast<bfloat16_t,64,4,false>`, and E27 lost 0.3321 % of score. The
  local evidence says this costs nothing here — serial −0.0196 % against a
  0.0676 % bar, and the isolated `M=1` cell 1.404 % faster — but the local
  serial leg is not the ranked numerator, so only an M5 run settles it.
- **Recommendation: compose, do not submit alone.** The mechanism is real, the
  evidence is reproducible, and the diff is two characters with zero byte
  growth. Compose with `t6`, which is already in the base, and let askeladd's
  PR #69 additivity test certify the pair.

## Proposed next assignment: test the cost model, not another candidate

The advisor asked for this and I recommend it as the single highest-value next
step. It is one cell palindrome and it tests the pricing model that every
remaining idea depends on.

**Question.** Does the "removing a narrow tail beats the average-rate model"
rule predict the sign and size of the miss at other widths?

**Design.** One whole-table cell palindrome, widths 1..10, 21 reps, inner 10,
real 40 °C gate, arms `shipped`, `t9`, `t8`.

- `t9`: M=9 `{5,4}` to `[9]`. Two distinct group sizes, so a tail exists.
  **Predicted: over-performs the pure model**, as `t55` did.
- `t8`: M=8 `{4,4}` to `[8]`. Both groups run at the same rate, so there is no
  cheap tail to delete. **Predicted: matches the pure model**, no over-performance.

**Why it decides something.** The two arms differ only in whether a cheap tail
exists. If `t9` over-performs and `t8` does not, the tail rule is confirmed and
the campaign gains a corrected pricer. If both over-perform, the defect is not
about tails and the model is wrong in a different way. If neither does, `t55` is
a one-off and should not be generalised.

**Run a feasibility step first; it may answer the question for free.** Both arms
need widths the wide helper does not currently admit. `quantized.h:980` asserts
`NA >= 2 && NA <= 6`, and the helper builds `vec<float, NA>` accumulators with
`rows_per_simd = 4`. So `t9` needs NA=9 and `t8` needs NA=8, both above the
bound, and both above edward's NA=6 register step where a second
`[4 x <6 x float>]` alloca appears and peak live registers cross the AGX 128
boundary (122 fits, 142 does not).

The cheap first step is therefore a compile and register census at NA=7, 8 and 9,
with no timing at all:

- If NA=8 and NA=9 do not fit the register budget, that **is** the result. The
  tail-deletion rule would then be bounded by the register step rather than by
  bandwidth, which is a sharper and more useful statement than any timing.
- If they fit, run the palindrome and test the prediction as designed.

Raising the bound for a probe carries no submission risk, because this is a cell
measurement and not a candidate.

**A cleaner control exists at M=7.** M=7 ships `<T,7,4>` = `{4,3}`, which has two
distinct group sizes and therefore a cheap tail, and NA=7 is the smallest width
above the current bound. If only one width can be made to fit, test M=7 rather
than M=9: it needs the smallest bound increase, and it carries 12.2 % of ranked
QMV against 5.75 % at M=9.

## Other suggested follow-ups, not implemented

1. **Re-measure the M=5 share on a fixture whose width histogram is closer to
   ranked.** My 0.047906 against the advisor's 0.066816 is a 28 % spread on the
   single most important conversion constant in this report.
2. **Extend the in-binary string assertion to the leg runners**, not just
   `--local-submit`. My leg sessions used a binary probe, but that probe checks
   routing presence rather than the exact template arguments.
3. **Re-pin `ACK_UNSCORED` in the same commit as every future routing change.**
   The pin went stale on t6 and nobody noticed, because the gate could not run
   here. It runs now, so the next routing merge should carry its own digest.
4. **Clear the five hard-coded `0.0629 %` null floors** listed above so no later
   analysis silently inherits a retracted constant.
