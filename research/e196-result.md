# E196 — Price the second SDPA call of the qL 6..9 exactness split

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"route1_recoverable_ms_per_round","available":true,"value":1.214},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}

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
  One Stage 0 dispatch census plus one Stage 1 timing session (six
  ABBA-counterbalanced blocks, 2856 samples, 74 s wall clock).
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
   `(32, gqa, qL)`) plus `sdpa_vector_2pass_2_bfloat16_t_256`. Call B at m = 6
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

Ladder fit, `serial` mode (max absolute residual in brackets):

| kv (kL) | a, per-call fixed µs | b, per-row µs | residual |
| --- | ---: | ---: | ---: |
| 512 (517) | 30.8 | 11.31 | 0.48 |
| 768 (773) | 35.5 | 15.11 | 0.42 |
| 1024 (1029) | 39.1 | 22.09 | 2.96 |
| 2048 (2053) | 38.7 | 35.40 | 6.05 |

One-pass KV scaling: `a0 = 21.2 µs`, `a1 = 0.0185 µs/key`, `b0 = 3.62 µs`,
`b1 = 0.01486 µs/key`. At the scored window kL = 517, `b1*kL = 7.7 µs` of the
11.3 µs per-row cost, so **68 % of the per-row cost is KV traversal** and the
per-call fixed cost `a` is nearly KV-independent.

Per-layer prices at the highest-weight cell (kv 512, m 8, `serial`):

| quantity | µs / layer |
| --- | ---: |
| call A (5 rows over 512 keys) | 92.1 |
| call B (3 rows over 520 keys) | 62.1 |
| A + B, timed separately | 154.2 |
| A + B, timed as one pair | 145.7 |
| overlap between the two calls | 8.5 |
| m = 5 control | 87.3 |
| Route-1 fused floor | 67.5 |

Current-tree weighted aggregates (16 full-attention layers, depth histogram
`{3:1, 4:29, 5:2, 6:3, 7:46}` from `research/analysis-runP-512-confirm.json`,
so m in `{4:1, 5:29, 6:2, 7:3, 8:46}`; the split fires in **51/81 = 63 % of
rounds** and m = 8 alone is 56.8 %):

| quantity | serial ms/round (MUE) | indep ms/round (MUE) |
| --- | ---: | ---: |
| gross price of the second call | 0.950 (1.68) | 0.813 (1.43) |
| overlap with call A | 0.203 (0.36) | 0.165 (0.29) |
| step against the m = 5 control | 1.008 (1.78) | 0.890 (1.57) |
| recover by fusion alone | **0.325 (0.57)** | **0.132 (0.23)** |
| recover by fusion + row amortization (Route-1) | **1.214 (2.14)** | **1.231 (2.17)** |

An earlier identical session, run before the temperature-probe fix, produced
serial 1.158 and indep 1.220 ms/round for the Route-1 figure and serial 0.325
-> 0.304 for the fusion-only figure. Its artifact was overwritten by the
retained session, so the retained session is the evidence of record, and the
earlier session counts only as a consistency check.

**FINDING 497 bridge.** The `today` arm reproduces FINDING 497's instrument.
Its m = 8 step over the m = 5 control at kv 512 is 123.4 µs/layer = 1.97
ms/round, against the chain-slope step of 1.008 ms/round. About **half of the
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
| 517 | 1 | 27.3 | 25.2 | 38.0 | 49.4 | 0.37 |
| 517 | 5 | 37.3 | 49.6 | 91.0 | 155.4 | 0.75 |
| 1029 | 1 | 34.1 | 37.2 | 56.6 | 87.1 | 0.55 |
| 1029 | 5 | 57.9 | 95.4 | 157.7 | 287.3 | 0.82 |

**I must correct my own pre-declared reading of this arm.** I planned to read
"time scales with head count" as "throughput bound, therefore close". That
reading is wrong, because the arm scales the threadgroup count and the load
traffic together and so cannot separate them. The elasticity proves only that
a per-work resource binds. A roofline names which one, at the scored cell
(kL = 517, 5 rows, 24 heads, 87.3 µs):

| quantity | value | fraction of the M4 Pro limit |
| --- | ---: | ---: |
| unique KV bytes read | 2.12 MB | 24.3 GB/s = 8.9 % of 273 GB/s |
| KV bytes actually requested | 63.5 MB (redundancy factor GQA·rows = 30) | 728 GB/s |
| arithmetic rate | 728 GFLOP/s | 9.0 % of the 8.06 TFLOP/s FMA peak |

The dispatch sits at 9 % of DRAM bandwidth and 9 % of the ALU peak while
issuing 728 GB/s of load traffic that is redundant by a factor of 30. Neither
DRAM nor ALU binds. **The binding resource is redundant load throughput
through the cache hierarchy, and that is exactly the resource a row-amortized
kernel removes.** The R = 1 rows of the table add the second fact: at 24
threadgroups the elasticity falls to 0.37-0.57, so the machine is not full at
the parallelism a fused kernel would run at, and part of the extra per-thread
arithmetic is free.

The Route-1 floor is not an extrapolation. The identity
`F(m, kL) = a + m*b0 + b1*kL = T(1, kL) + (m-1)*b0` makes it **the measured
single-row call at the same window, plus the small non-KV per-row term**. At
kv 512, m = 8: `T(1, 520) = 42.1 µs` and `F = 42.1 + 7*3.62 = 67.5 µs`.

## Decision

**CONTINUE — build the Route-1 kernel, but only in its row-amortized form.**

The pre-declared rule reads 1.214 ms/round (serial) and 1.231 ms/round
(indep), both above the 1.0 CONTINUE threshold, in both dependency modes.

The qualification is the most useful part of this result. The price splits
into two different mechanisms:

- **Removing the second dispatch alone is worth 0.33 ms/round (serial) or 0.13
  (indep), below the 0.5 CLOSE threshold.** A fused kernel that merely
  concatenates the two calls into one dispatch, and keeps today's per-row
  cost, fails its own decision rule. Do not build that kernel.
- **Removing the redundant per-row KV traversal is worth the remaining 0.89
  ms/round (serial) or 1.10 (indep).** Route-1 pays only if each threadgroup
  loads a KV tile once and applies it to all m rows from registers.

A row-amortized kernel also pays outside the split. The m <= 5 rounds issue one
call whose rows still re-traverse the KV window, worth a further **0.213
ms/round (serial) / 0.264 (indep)** across the 30/81 non-split rounds. That is
a separate prize for a separate experiment, and it is deliberately excluded
from the 1.214 headline.

### Named uncertainty and the smallest check that settles it

**Register pressure at m rows.** A fused kernel holds q[m][8] and o[m][8] per
thread at head dim 256 across 32 lanes, which is 16·m vector registers. At
m = 8 that is 128 registers per thread, which can cut occupancy or spill, and
the R = 1 elasticity already shows the machine is not full at 24 threadgroups.

Smallest check, and it is cheap: write the fused kernel for **m = 6 only**
(96 registers) and time one `occupancy`-style dispatch against today's pair at
kv 512. If the m = 6 fused call lands near its 60.2 µs floor, the register
model holds and m = 7 and 8 follow. If it lands near today's 123.0 µs pair,
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
  recorded at session entry (39.3 °C), after warmup (40.4 °C), at each block
  entry (40.0, 43.1, 45.8, 45.9, 47.2, 48.6 °C), and at session exit
  (48.3 °C); and the report preserves `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false` verbatim. The 8.6 °C monotone entry-
  temperature drift across blocks cancels to first order under the ABBA order,
  and the per-cell robust minimum takes the coolest, least contended block.
- **RULE 388** does not bind. It forbids leg-level ABBA for effects below 0.5
  ms/round. This probe measures dispatches directly, and the deciding effect is
  62-92 µs per call against a 0.4-2.1 % block spread.
- **RULE 386** per-leg trace paths: not applicable, no legs.
- The probe needle `E196-SECOND-SDPA-PASS-PRICING-LIVE-PATH-2026-08-24` is
  stamped into every report file.
- Interpolation: cell prices are interpolated in the KV window only inside one
  kernel family, never across the `kL = 1024` boundary. Every interpolated
  round is counted in `weighted[*].interpolated`.
- No fidelity claim is made or needed: no token was generated.

## W&B

- Run of record: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/k2f7njb3>
  (id `k2f7njb3`), `harness=local`, job type `probe`.
- An earlier logging pass of the same session data, before the per-call price
  aggregates were added, is run `c5wmxffo`.

## Reproduction

```bash
swift build --build-tests --force-resolved-versions
research/e196_census.sh                      # Stage 0, no timing
research/e196_session.sh 6 20 6              # Stage 1, six ABBA blocks
python3 research/e196_analyse.py --wandb
```

Host: AWS EC2 mac, M4 Pro `applegpu_g16s`, 48 GiB unified memory,
macOS 26.5.2 (25F84), Apple Swift 6.3.3, frozen dependency graph.
Session wall clock 74 s, 2856 samples. No model is loaded, so peak RAM stays
at test-process level and no model-holding lock is taken.

## Conclusion

- **What happened.** The second SDPA call costs 0.950 ms/round of GPU time on
  the current tree, and only 0.203 ms/round of that overlaps with the first
  call. An eval-free instrument shows about half of FINDING 497's published
  m >= 6 step is a measurement barrier, not decode work.
- **Evidence for the mechanism.** The kernel gives every (row, head) its own
  threadgroup and its own full KV traversal, the achieved rates sit at 9 % of
  both the DRAM and the ALU limit while issuing 30x redundant load traffic, and
  the per-work elasticity is 0.75-0.82. Redundant load throughput binds, and
  row amortization removes it.
- **Evidence against.** Fusion by itself is worth only 0.33 ms/round, so the
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
2. **Row-amortized kernel for the m <= 5 rounds** — a separate 0.21-0.26
   ms/round prize that needs no split at all, and a safer first target because
   m = 5 needs only 80 extra registers per thread.
3. **Raise the depth cap once the width wall is gone.** The qL·gqa <= 32 limit
   is why `sdpaWidthWallDepthCap` is 5. A row-blocked kernel removes the
   hardware reason for that constant, which may reopen deeper drafting.
4. **Re-measure the round composition** on more than one trace before anyone
   prices a width-dependent change again; the 63 % split share drives every
   weighted number in this report.
