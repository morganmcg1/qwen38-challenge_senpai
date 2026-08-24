# E196 — Price the second SDPA call of the qL 6..9 exactness split

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"route1_recoverable_ms_per_round","available":true,"value":1.212},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}

- Student / branch: qwen-edward / `qwen-edward/e196-second-sdpa-pass-pricing`
- Hypothesis and target cost: the second SDPA call that `AttentionUtils.swift:103-141`
  issues for query rows 6..9 is a large, removable cost. Price it directly and
  decide Route-1 (a fused one-pass kernel) go/no-go.
- Decision: **CONTINUE**, with one binding qualification (see "Decision").
  This is a pricing probe, not a merge candidate. Close unmerged.
- `BASE_SHA`: `a31e48e2be3bb878dbdeaf8cd9a4eeaa00062d3c`
- `UPSTREAM_SHA`: `80021bc03e4b270f7dfef5b4425107bfc57b8d70`
- Candidate commit: this branch head. No submitted-surface file changes.
- Yukon promoted submission used as frontier: crown `ec24d591`, score
  3.7291100105909, source ref `0863b06ac16e`. Our best receipt: `5a9f130a`,
  score 3.70784519415395, source ref `8ba6e738`.
- Candidate build fingerprint: not applicable. Every arm runs under
  `swift test`; no `.build-worker` runtime worker takes part, so RULE 384 and
  RULE 387 worker-freshness do not apply.
- Submitted-surface / generated-twin / metallib digests: unchanged. This branch
  touches no path in `benchmark.json` `editablePaths`.
- Submitted candidate files: none.
- Supporting test, tooling, or documentation files:
  `Tests/MLXFastTests/E196SecondPassPricingTests.swift`,
  `research/e196_census.sh`, `research/e196_session.sh`,
  `research/e196_analyse.py`, `research/e196-census.json`,
  `research/e196-timing.json`, `research/e196-analysis.json`,
  `research/e196-result.md`.
- MTP head provenance, digest, and draft policy: not applicable. No decode leg
  runs. The probe drives `MLXFast.scaledDotProductAttention` directly.
- Token window, fixture, reference source, and harness: not applicable to a
  dispatch probe. **`harness=local`.**
- Exact cell: `sdpa_vector` and `sdpa_vector_2pass` bfloat16 head-dim-256
  GQA-6 family, `mlx.metallib` source form, no `_nax` variant. See "Stage 0".
- Official causal path and score equation: the split runs inside the candidate
  MTP leg only. Lower candidate MTP seconds per token raises every affected
  ranked `raw_p`. `senpai/verify-ranked-score-boundary.sh` passes. No local
  serial-path share is subtracted anywhere in this report.
- Assignment-scope preflight: `git diff --name-only BASE..HEAD` lists only
  `Tests/MLXFastTests/` and `research/` files. Zero `editablePaths` entries, so
  `senpai/validate-assignment-scope.sh` has no submitted path to check.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `editable budget OK: source=2651144/3000000 headroom=348856 growth=0/262144
  exempt=2410/2147483648 files=154`.
- Scored-path reachability evidence: `AttentionUtils.attentionWithCacheUpdate`
  applies the `qL > 5` split on every full-attention layer of every round with
  m >= 6 rows. `Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1060` sets
  `sdpaWidthWallDepthCap = 5`, and line 1078 sets `segmentedVerifyDepthCap = 7`.
  The shipped cap-7 tree therefore reaches m = 8 rows.
- Written promotion rule and verdict: see "Decision".
- Pre-official evidence budget / timed legs used: **zero timed decode legs.**
  One Stage 0 dispatch census plus one Stage 1 timing session of record (six
  ABBA-counterbalanced blocks, 4092 samples, 98 s wall clock), and two earlier
  sessions of the same probe that are reported only as replicates.
- Frozen candidate SHA: none. Nothing to submit.
- Submission owner / receipt-watcher job ID: none.

## Question, evidence, expectation, test, and rule

**Question.** How much GPU time does the second SDPA call of the exactness
split cost per round on the current tree, and is a fused one-pass Route-1
kernel worth building?

**Evidence that made it worth testing.** FINDING 497 measured a large step in
`today`-form attention cost between m = 5 and m >= 6, but its instrument put a
blocking `eval()` behind every call, so the published step mixed real GPU work
with an eval barrier that in-path decode does not pay.

**Expected result.** A recoverable price of order 1 ms/round, against a
median unit of effort (MUE) of 567 µs/round.

**Smallest decisive test.** Price the dispatches directly instead of timing a
decode leg: put K SDPA calls behind ONE `eval`, regress total time on K, and
take the slope as the pipelined per-call cost.

**Stop or promotion rule (pre-declared).**
current-tree-weighted recoverable time, ms/round:
`>= 1.0` CONTINUE, `< 0.5` CLOSE, `0.5..1.0` unclear plus one named
uncertainty and the smallest check that settles it.

## Stage 0 — kernel identity, with no timing

`research/e196_census.sh` swizzles
`newComputePipelineStateWithFunction:error:` and records which Metal pipeline
each call creates, with its grid and threadgroup dimensions
(`research/e196-census.json`).

1. **Kernel selection.** For `kL < 1024` both split calls run the one-pass
   `sdpa_vector_bfloat16_t_256_256_nomask_qnt_{c,nc}_nosinks`, grid
   `(24 query heads, R rows, 1)`, threadgroups of 1024 threads: **one
   threadgroup per (query head, query row)**. For `kL >= 1024` MLX switches to
   `sdpa_vector_2pass_1_..._128` (grid `(4 kv heads, 1, 128)`, group dims
   `(32, gqa, qL)`) plus `sdpa_vector_2pass_2_bfloat16_t_256`. The census
   covers kv 512, 768, 1024, and 2048. The timing session adds kv 896 and
   1008, which the `kL < 1024` rule places in the one-pass family with kv 512
   and 768. Call B at m = 6
   has R = 1 and therefore selects the `_nc` (non-causal) variant. Each
   shipped-slice call adds exactly one `g2_copy` query copy; the `today` form
   adds two more `gg2_copy` concatenation copies.
2. **M5 transfer.** This host reports `applegpu_g16s` (M4 Pro); the ranked
   runner reports `applegpu_g17s`. MLX's SDPA dispatch selection branches on
   `devc == 's'`, which holds on both, so **M5 selects the same kernels**. The
   vector family is loaded from `mlx.metallib` and has **no `_nax` variant**
   (only the full `steel_attention` family has one). Absolute times scale;
   kernel identity does not change.
3. **No KV sharing exists to lose.** In both `sdpa_vector` and
   `sdpa_vector_2pass_1`, every (query row, query head) simdgroup reads K and V
   straight from device memory. There is no threadgroup staging and no
   cross-row reuse. Splitting m rows into 5 + (m - 5) therefore duplicates no
   load instructions: the split costs one extra dispatch, one extra window
   traversal, and one extra per-call fixed cost.
4. **Why a Route-1 kernel must block rows in registers.**
   `sdpa_vector_2pass_1` uses group dims `(32, gqa, qL)` = 192·qL threads. At
   gqa = 6, qL >= 6 needs at least 1152 threads, above the 1024-thread
   hardware limit. That is the structural origin of the qL·gqa <= 32 rule. A
   genuine fused kernel must loop rows inside each thread, which is also why
   it can stay bit-exact: the per-row reduction order does not change.
5. **KV weighting correction.** `program.md` fixes a 512-token seed plus 512
   decoded tokens, so the cache offset runs 512 -> 1024 and `kL = offset + m`.
   `sdpa_vector_2pass` needs `kL >= 1024`, so **about 98 % of scored rounds run
   the one-pass family**, kv = 1024 carries 1-2 % weight, and **kv = 2048 has
   zero current-tree weight**. The analyser never fits or interpolates across
   the `kL = 1024` family boundary.

## Stage 1 — chain-slope pricing

`research/e196_session.sh` runs `E196ChainSlopePricingTests`. K SDPA calls sit
behind one `eval`, K in {1, 2, 4, 8, 16}. Two dependency modes bracket MLX's
scheduling freedom: `indep` uses distinct query tensors, and `serial` feeds
each output back as the next query. Arms: `ladder` (R = 1..5 at the m = 5
window), `callA`, `callB`, `pair`, `occupancy`, and `today` (FINDING 497's own
one-call-per-eval form). Six ABBA-counterbalanced blocks, 20 reps per cell,
per-cell robust minimum over blocks.

Model fitted on the ladder, per KV window N:

```text
T(R, N) = a(N) + R * b(N)        a(N) = a0 + a1*N     b(N) = b0 + b1*N
```

`b1*N` is the row's own traversal of the KV window, and it is the only term a
row-amortized kernel can remove.

Ladder fit, `serial` mode (max absolute residual in the last column). A scored
512-token leg runs the cache offset from 512 to 1024, so four of the six
windows sit inside the one-pass family and cover the whole leg. Nothing in the
one-pass price is extrapolated beyond the measured range.

| kv (kL) | family | a, per-call fixed µs | b, per-row µs | residual |
| --- | --- | ---: | ---: | ---: |
| 512 (517) | 1-pass | 30.8 | 11.33 | 1.01 |
| 768 (773) | 1-pass | 35.5 | 15.17 | 0.94 |
| 896 (901) | 1-pass | 33.2 | 17.62 | 2.21 |
| 1008 (1013) | 1-pass | 37.0 | 18.76 | 2.39 |
| 1024 (1029) | 2-pass | 37.1 | 22.61 | 4.81 |
| 2048 (2053) | 2-pass | 38.4 | 35.36 | 5.69 |

One-pass KV scaling over four windows: `a0 = 25.8 µs`, `a1 = 0.0104 µs/key`,
`b0 = 3.40 µs`, `b1 = 0.01538 µs/key`. At the scored window kL = 517,
`b1*kL = 7.95 µs` of the 11.33 µs per-row cost, so **70 % of the per-row cost
is KV traversal** and the per-call fixed cost `a` is nearly KV-independent.

Per-layer prices at the highest-weight cell (kv 512, m 8, `serial`), and at the
end-of-leg window (kv 1008, m 8) for comparison:

| quantity | kv 512, µs/layer | kv 1008, µs/layer |
| --- | ---: | ---: |
| call A (5 rows over the older keys) | 91.6 | 136.6 |
| call B (3 rows over the whole window) | 62.2 | 100.9 |
| A + B, timed separately | 153.8 | 237.5 |
| A + B, timed as one pair | 147.7 | 219.4 |
| overlap between the two calls | 6.1 | 18.1 |
| m = 5 control | 87.4 | 130.8 |
| Route-1 fused floor | 66.0 | 79.8 |
| recover by fusion alone | 26.3 | 32.4 |
| recover by fusion + row amortization | 81.7 | 139.6 |

Current-tree weighted aggregates (16 full-attention layers, depth histogram
`{3:1, 4:29, 5:2, 6:3, 7:46}` from `research/analysis-runP-512-confirm.json`,
so m in `{4:1, 5:29, 6:2, 7:3, 8:46}`; the split fires in **51/81 = 63 % of
rounds** and m = 8 alone is 56.8 %):

| quantity | serial ms/round (MUE) | indep ms/round (MUE) |
| --- | ---: | ---: |
| gross price of the second call | 0.899 (1.59) | 0.779 (1.37) |
| overlap with call A | 0.176 (0.31) | 0.120 (0.21) |
| step against the m = 5 control | 0.956 (1.69) | 0.823 (1.45) |
| recover by fusion alone | **0.291 (0.51)** | **0.111 (0.20)** |
| recover by fusion + row amortization (Route-1) | **1.212 (2.14)** | **1.161 (2.05)** |

Two earlier sessions of the same probe replicate the headline. Both measured
only two one-pass windows, and the first recorded no temperature because the
probe looked for `macmon` under the passwd home instead of the role home. Their
Route-1 figures were 1.158 / 1.220 and 1.214 / 1.231 ms/round (serial /
indep), and their fusion-only figures were 0.304 / 0.119 and 0.325 / 0.132.
Their artifacts were overwritten by the session of record, so they count as
consistency checks only. The decision does not depend on which session is used.

**FINDING 497 bridge.** The `today` arm reproduces FINDING 497's instrument.
Its m = 8 step over the m = 5 control at kv 512 is 123.3 µs/layer = 1.97
ms/round, against the chain-slope step of 0.956 ms/round. About **half of the
published step is the blocking per-call `eval()` barrier**, which in-path
decode does not pay. The eval-free number is the one to price against.

## The deciding measurement: occupancy and roofline

The two brackets straddle both thresholds, so one question decides the
experiment: does a fused kernel that loops m rows inside one threadgroup
actually recover the KV traversal, or is the machine already saturated?

The `occupancy` arm holds per-threadgroup work fixed (GQA 6, head dim 256, one
KV window) and scales the query-head count, which scales the threadgroup count
and the total load traffic together. Per-call µs, `serial` mode:

| kL | rows | 6 heads | 12 heads | 24 heads | 48 heads | elasticity at 24 heads |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 517 | 1 | 27.0 | 26.9 | 37.3 | 48.4 | 0.35 |
| 517 | 5 | 37.3 | 48.9 | 81.3 | 157.1 | 0.86 |
| 1029 | 1 | 34.0 | 36.9 | 57.6 | 87.1 | 0.55 |
| 1029 | 5 | 62.0 | 91.1 | 158.2 | 288.1 | 0.82 |

**I must correct my own pre-declared reading of this arm.** I planned to read
"time scales with head count" as "throughput bound, therefore close". That
reading is wrong, because the arm scales the threadgroup count and the load
traffic together and so cannot separate them. The elasticity proves only that
a per-work resource binds. A roofline names which one, at the scored cell
(kL = 517, 5 rows, 24 heads, 87.5 µs from the ladder fit):

| quantity | value | fraction of the M4 Pro limit |
| --- | ---: | ---: |
| unique KV bytes read | 2.12 MB | 24.2 GB/s = 8.9 % of 273 GB/s |
| KV bytes actually requested | 63.5 MB (redundancy factor GQA·rows = 30) | 727 GB/s |
| arithmetic rate | 727 GFLOP/s | 9.0 % of the 8.06 TFLOP/s FMA peak |

The dispatch sits at 9 % of DRAM bandwidth and 9 % of the ALU peak while
issuing 727 GB/s of load traffic that is redundant by a factor of 30. Neither
DRAM nor ALU binds. **The binding resource is redundant load throughput
through the cache hierarchy, and that is exactly the resource a row-amortized
kernel removes.** The R = 1 rows of the table add the second fact: at 24
threadgroups the elasticity falls to 0.35-0.57, so the machine is not full at
the parallelism a fused kernel would run at, and part of the extra per-thread
arithmetic is free.

The Route-1 floor is not an extrapolation. The identity
`F(m, kL) = a + m*b0 + b1*kL = T(1, kL) + (m-1)*b0` makes it **the measured
single-row call at the same window, plus the small non-KV per-row term**. At
kv 512, m = 8: `T(1, 520) = 42.1 µs` and `F = 42.1 + 7*3.40 = 66.0 µs`.

## Decision

**CONTINUE — build the Route-1 kernel, but only in its row-amortized form.**

The pre-declared rule reads 1.212 ms/round (serial) and 1.161 ms/round
(indep), both above the 1.0 CONTINUE threshold, in both dependency modes and
in all three sessions.

The qualification is the most useful part of this result. The price splits
into two different mechanisms:

- **Removing the second dispatch alone is worth 0.29 ms/round (serial) or 0.11
  (indep), below the 0.5 CLOSE threshold.** A fused kernel that merely
  concatenates the two calls into one dispatch, and keeps today's per-row
  cost, fails its own decision rule. Do not build that kernel.
- **Removing the redundant per-row KV traversal is worth the remaining 0.92
  ms/round (serial) or 1.05 (indep).** Route-1 pays only if each threadgroup
  loads a KV tile once and applies it to all m rows from registers.

A row-amortized kernel also pays outside the split. The m <= 5 rounds issue one
call whose rows still re-traverse the KV window, worth a further **0.220
ms/round (serial) / 0.251 (indep)** across the 30/81 non-split rounds. That is
a separate prize for a separate experiment, and it is deliberately excluded
from the 1.212 headline.

### Named uncertainty and the smallest check that settles it

**Register pressure at m rows.** A fused kernel holds q[m][8] and o[m][8] per
thread at head dim 256 across 32 lanes, which is 16·m vector registers. At
m = 8 that is 128 registers per thread, which can cut occupancy or spill, and
the R = 1 elasticity already shows the machine is not full at 24 threadgroups.

Smallest check, and it is cheap: write the fused kernel for **m = 6 only**
(96 registers) and time one `occupancy`-style dispatch against today's pair at
kv 512. If the m = 6 fused call lands near its 59.2 µs floor, the register
model holds and m = 7 and 8 follow. If it lands near today's 120.0 µs pair,
spilling has eaten the prize and Route-1 closes without a full implementation.
Compile the kernel with `-Rpass-analysis=kernel-resource-usage` first: the
register count per thread is a free static answer before any timing.

## Compliance and honesty statements

- **`harness=local`.** Every number here is a within-session relative dispatch
  measurement on this M4 Pro host. None of it is a score, a ranked estimate,
  or a decode-leg measurement.
- **Ungated timing mode.** `MLXFAST_LOCAL_COOL_GATE` is not involved because no
  decode leg runs, but the same standing conditions are met and reported:
  arms are ABBA-counterbalanced inside one session; GPU temperature is
  recorded at session entry (39.2 °C), after warmup (41.0 °C), at each block
  entry (39.9, 44.6, 45.9, 47.8, 48.0, 49.2 °C), and at session exit
  (49.7 °C); and the report preserves `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false` verbatim. The 9.3 °C monotone entry-
  temperature drift across blocks cancels to first order under the ABBA order,
  and the per-cell robust minimum takes the coolest, least contended block.
- **RULE 388** does not bind. It forbids leg-level ABBA for effects below 0.5
  ms/round. This probe measures dispatches directly, and the deciding effect is
  62-101 µs per call.
- **RULE 386** per-leg trace paths: not applicable, no legs.
- The probe needle `E196-SECOND-SDPA-PASS-PRICING-LIVE-PATH-2026-08-24` is
  stamped into every report file.
- Interpolation: cell prices are interpolated in the KV window only inside one
  kernel family, never across the `kL = 1024` boundary. 49 of the 51 split
  rounds fall between measured windows; only 2 rounds, at the very end of the
  leg, are extrapolated. Both counts are recorded in
  `weighted[*].interpolated_cells` and `weighted[*].extrapolated_cells`.
- No fidelity claim is made or needed: no token was generated.

## Regression test run

`swift test --force-resolved-versions` at branch head `957c4dad`:
751 tests in 78 suites, 41 issues, exit code 1.

Both new suites pass and skip their bodies, because the probe is opt-in:

```text
✔ Suite E196KernelIdentityCensusTests passed after 10.310 seconds.
✔ Suite E196ChainSlopePricingTests passed after 10.310 seconds.
➜ Test namesTheKernelsOfCallAAndCallB() skipped: set MLX_E196_CENSUS=1 ...
➜ Test pricesTheSecondSdpaCall() skipped: set MLX_E196_TIMING=1 ...
```

The 41 issues sit in `QwenMTPTrackNamingTests`, `QwenMTPHeadDeclarationTests`,
`QwenMTPScoringSemanticsTests`, `Qwen35ArtifactContractTests`,
`RuntimeStartupMemoryPolicyTests`, `E130WiredResidencySlackTests`,
`BenchmarkScriptTests`, and `SetupScriptTests`. They check pinned artifact
digests, head declaration provenance, campaign document text, calibration
provenance, and the startup memory profile. This branch changes no source
file at all: `git diff a31e48e2..957c4dad --stat` lists only
`Tests/MLXFastTests/E196SecondPassPricingTests.swift` and `research/`, and
zero `editablePaths`. The failures are therefore pre-existing on the
assignment base `a31e48e2` and are not caused by this experiment. I did not
repair them, because they are outside the assigned question.

## W&B

- Run of record: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/0xkga2ov>
  (id `0xkga2ov`), `harness=local`, job type `probe`. It carries the ladder
  fits, the per-cell price table, the occupancy table, the roofline, the
  FINDING 497 bridge, and the weighted decision.
- Earlier logging passes of the replicate sessions: `or7xrw2g`, `k2f7njb3`,
  `c5wmxffo`.

## Reproduction

```bash
swift build --build-tests --force-resolved-versions
research/e196_census.sh                      # Stage 0, no timing
research/e196_session.sh 6 20 6              # Stage 1, six ABBA blocks
python3 research/e196_analyse.py --wandb
```

Host: AWS EC2 mac, M4 Pro `applegpu_g16s`, 48 GiB unified memory,
macOS 26.5.2 (25F84), Apple Swift 6.3.3, frozen dependency graph.
Session wall clock 98 s, 4092 samples. No model is loaded, so peak RAM stays
at test-process level and no model-holding lock is taken.

## Conclusion

- **What happened.** The second SDPA call costs 0.899 ms/round of GPU time on
  the current tree, and only 0.176 ms/round of that overlaps with the first
  call. An eval-free instrument shows about half of FINDING 497's published
  m >= 6 step is a measurement barrier, not decode work.
- **Evidence for the mechanism.** The kernel gives every (row, head) its own
  threadgroup and its own full KV traversal, the achieved rates sit at 9 % of
  both the DRAM and the ALU limit while issuing 30x redundant load traffic, and
  the per-work elasticity is 0.82-0.86. Redundant load throughput binds, and
  row amortization removes it.
- **Evidence against.** Fusion by itself is worth only 0.29 ms/round, so the
  cheap version of Route-1 is already refuted. Register pressure at m = 8 is
  the one unquantified risk.
- **Prompt or M5 transfer risk.** Low for kernel identity: the M5 selects the
  same kernels and the vector family has no `_nax` variant. Absolute times
  will differ, so the fused kernel must be re-measured on M5 before promotion.
  Round composition comes from one 512-token cap-7 trace, so the 63 % split
  share is a single-trace estimate.
- **Smallest useful next action.** Build the m = 6 register-blocked fused
  kernel, read its static register usage, and time one dispatch against
  today's pair.
- **Recommendation.** Close E196 unmerged as a completed pricing probe.
  Open the Route-1 experiment with the row-amortization requirement written
  into its assignment.

## Suggested follow-ups, not implemented

1. **Route-1 fused kernel, register-blocked over rows** — the direct successor,
   priced here at 1.21 ms/round on split rounds.
2. **Row-amortized kernel for the m <= 5 rounds** — a separate 0.22-0.25
   ms/round prize that needs no split at all, and a safer first target because
   m = 5 needs only 80 registers per thread.
3. **Raise the depth cap once the width wall is gone.** The qL·gqa <= 32 limit
   is why `sdpaWidthWallDepthCap` is 5. A row-blocked kernel removes the
   hardware reason for that constant, which may reopen deeper drafting.
4. **Re-measure the round composition** on more than one trace before anyone
   prices a width-dependent change again; the 63 % split share drives every
   weighted number in this report.
