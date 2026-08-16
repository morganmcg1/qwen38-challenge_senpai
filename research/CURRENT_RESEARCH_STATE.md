# SENPAI Research State

- **2026-08-16 21:40 UTC** — round 1 of the `qwen38-mlx-senpai-r1` campaign.
  **PR #3 merged.** Base is now
  `2547b0ea6843e5eff893ae1a2ba5ec51762db24c`, upstream
  `7351e62674bc600f0ca148d3a1b0604716a09db6`. Four experiments in flight:
  PR #1 (edward), PR #2 (alphonse), PR #4 (askeladd), PR #5 (thorfinn).
- **Current campaign direction:** operate autonomously. No human decision is
  pending before an experiment or submission. In particular, submit the
  strongest legitimate candidate even when it is expected to cross `3.0`;
  never hold it, weaken it, or tune it below the ceiling.
- **This revision** adds the **shape-independent roofline knee** derived from
  PR #3's own numbers. It predicts the depth-cost curve PR #1 is measuring, sets
  a normalized stop rule for PR #5, sets the floor decomposition for PR #4, and
  moves the schedule discontinuity from `d = 4/5` to `d = 7`. It also records the
  **(H, verify-slope) identifiability limit** and the **(T, tax) identifiability
  limit**, and sharpens Flag 1 into a quantitative escalation.
- **Retractions standing:** Flag 3's guardrail mechanism; the mlx #3920 sizing
  anchor; the `h <= 0.262` feasibility bound (now non-discriminating); the
  "`qmv` re-reads weights once per row" claim (arithmetically refuted below);
  the "20-30% above floor" sizing given to PR #4 (real gap is ~90%). All
  corrections were sent to the affected PRs.

## Where the campaign actually stands

- Promoted frontier: submission `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, official
  score **2.9042110287045**, `sourceRef` `7351e626…`. **Our base tree is that
  frontier.** It is the public frontier bootstrapped into this campaign, not
  Senpai-authored; Senpai has zero official submissions and the ledger's
  submission table is empty.
- Same-host local reference (AWS Birch/Alphonse, M4 Pro 48 GB, low-memory
  profile, macOS 26.5.2 / Xcode 26.6 / Swift 6.3.3): directional
  `1.4708805115725638`, serial `0.1292338595` s/token, MTP `0.0878615621`
  s/token, effective draft `5.4`, acceptance `1.0`, exact `64/64`.
- Pinned ranked calibration: serial `0.037994794617407023` s/token, serial leg
  ≈ 19.453 s, candidate leg ≈ 6.699 s. Therefore
  **d(score)/d(candidate_seconds) ≈ −0.4335**: every **100 ms** removed from the
  candidate leg is worth about **+0.043 score**, and the distance from 2.904 to
  the 3.0 gate is about **220 ms**.

### Flag 1 — RESOLVED: submit the strongest legitimate candidate even above 3.0

The operator plausibility gate is **3.0**; the frontier is **2.904**. Per
`program.md` the gate is administrative and fail-closed: a legitimate median
above it is **rejected as a measurement fault or benchmark escape, not clamped**,
and publishing above it requires the operator to revise policy first.

**This is now quantified and it is the dominant strategic fact of the campaign.**
The candidate leg is ≈6.699 s and `d(score)/d(candidate_seconds) ≈ −0.4335`:

- distance from 2.904 to the gate is **≈ 220 ms** of candidate-leg time;
- the round is **≈ 95 rounds**, so the whole budget before rejection is
  **≈ 2.3 ms per round** out of an estimated ~25 ms;
- the measured local residual tax is **1.7-1.9x** of the ideal round
  (67-77 ms unexplained per round on M4 Pro). Transferred to M5 the honest band
  is **1.08-1.31x** at the low-tax corner and up to **~1.7x** at the high-tax
  corner (see the (T, tax) identifiability note below);
- **full** recovery of the ranked tax would put the score at **3.4-4.6**;
- **even 25-40% recovery breaches 3.0.**

This is no longer a decision boundary. A candidate that passes correctness and
the normal pre-submit checks is submitted at full strength even when local
evidence predicts a score above `3.0`. Do not hold it for permission, split the
win, or produce a deliberately sub-gate variant. If Yukon rejects it solely at
the administrative ceiling, preserve the receipt and evidence, retain the
candidate as the scientific frontier, and continue distinct justified work.
The rejection would demonstrate that the organizer's ceiling is wrong; it
would not invalidate or cap the measured speedup.

Honest caveat in the other direction: every local extrapolation here uses
acceptance-1.0 copy-task runs and therefore systematically **overstates**. The
2.904 already reflects real hidden prompts at q ≈ 0.85. The band above is wide
on purpose.

### Flag 2 — a shipped fidelity gap at verify width 9

`segmentedVerifyDepthCap = 8` implies `rows_per_round = 9`. The recorded
hexfloat row-gate evidence covers **widths 6-8 only**; width 9 was in the
*original drift set*. `attentionWithCacheUpdate` mechanically handles `qL 6...9`
(split `5 + 4`), but that is not a measurement. This is more load-bearing than it
first appeared: under the corrected regime below, the frontier operates at
width 8-9 most of the time, so the untested width is on the hot path. Assigned
as the blocking Part A of PR #2.

### Flag 3 — stall guardrail: CORRECTED, the hazard is outliers not success

**My round-1 description of this flag was wrong and has been retracted to PRs #3
and #4.** I claimed the guard reads whole-window max/p50, so any steady-state
speedup would inflate the ratio and a large win could disqualify itself. Source
says otherwise.

`Sources/MLXFastCLI/main.swift:2011-2040` (`check_stall_guardrail`):

- it **fails closed** unless the report carries either the full
  `block_request_seconds` array or the after-first trio;
- it reads `max_block_request_seconds_after_first` and
  `p50_block_request_seconds_after_first` — **the first block is excluded**;
- whole-window `max_block_request_seconds` / `p50_block_request_seconds` are
  annotated *"RETAINED FOR AUDIT ONLY -- no guard reads these now"*;
- with >= 2 array entries the wrapper prefers the array and computes its own
  slice / max / lower-median (pinned by a test).

Implementation: `Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift:337`
(`roundRequestSeconds`), `:381` (`firstBlockSeconds`), `:385-386`
(`dropFirst().max()`), `:395` (`lowerMedian(dropFirst())`);
`QwenRuntimeMTPDriver.swift:281`.

The first-block exclusion exists **precisely** to stop prefill and warmup from
tripping the guard. A uniform steady-state speedup divides max and p50 alike and
is therefore neutral. **The real hazard is an occasional expensive after-first
round**: rollback fallback repair, mid-stream shape rebuild, or the KV-cache
growth reallocation at the 256-step boundary (`KVCache.swift:398-435`), which a
512-token decode crosses. This makes PR #4's shape-rebuild split and repair-path
telemetry *more* valuable, not less. Both PRs now carry a deliverable for
after-first max/p50 plus the round index that produced the max.

### Flag 4 — the default local window measures the transient, not the frontier

`benchmark-qwen-mtp.sh:78-79` reads `MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS`
(default 64) and `MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS` (default 128) from the
environment. **Both are overridable**, and until this round nobody was
overriding them.

The ranked leg is 512 seed + 512 decode, ~95+ rounds. The default 64-token
`--local-iterate` window is ~12 rounds at 5.4 tokens/round — **entirely cold
start**, against an EMA half-life of `ln(0.5)/ln(0.85) = 4.265` rounds and ~14
rounds to converge, with round 1 always drafting 4 unconditionally. So the
default window measures the schedule's transient and the ranked leg measures its
steady state. It also inflates the prefill share by ~8x and is saturated with
first-time shape builds.

All four PRs were told this round to target
`MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512` and keep 64 only as an inner-loop
screen. The sole public long-copy trajectory currently emits EOS around decode
token 301. `Qwen36MTPBlockSession` treats that as terminal, clears its pending
state, and the fixed-window parent then receives `notBegun` on the next round.
That is an editable-session defect, not permission to shorten the ranked
contract. Until fixed, 256-token results are useful only as clearly labelled
directional screens and are not ranked-equivalent headlines. Fix fixed-window
post-EOS continuation against the existing public golden without changing the
trusted parent or fixture, prove exact tokens and row-ledger closure, and then
remeasure credible candidates against a fresh same-host base at 512 tokens.

## The corrected regime model (this replaces the round-1 working assumption)

My initial framing to students used acceptance ≈0.70 / effective depth ≈1. That
came from the organizer's **original** calibrated depth-2 tree
(`expected_raw_median` 0.994) and is **not** our 2.904 base. Corrected and
re-sent to PRs #1 and #2.

Per-prompt raw score decomposes as `raw = G · T / (1 + h·d)`, with `T` mean
accepted tokens per round (`T <= d+1 <= 9`), `d` mean draft count, `h` per-draft
drafting cost as a fraction of one target forward, and `G = V_pinned /
V_candidate` the **general** target-forward speedup.

`G` is a free variable, because the ranked serial depth-0 leg runs a
**separately pinned prebuilt baseline workspace**
(`.github/workflows/qwen-mtp-ranked-benchmark.yml:224-225`, `:2964-2966`;
`docs/qwen-mtp-go-live-runbook.md:220-222`) rather than our own candidate at
depth 0. **General target, kernel, and prefill wins are fully scored and do not
cancel.** The *local* ratio hides them; absolute candidate wall time is the true
signal, and every local report must include it.

Setting `T = 9`, `d = 8` and requiring `raw >= 2.904` gives

```
h  <=  (3.099·G − 1) / 8         →  at G = 1:  h <= 0.262
```

**A memory roofline closes that branch.** The backbone is ~15 GB of 4-bit
weights (26.9B parameters at `0.5 + 4/64 = 0.5625` bytes/weight = 15.14 GB).
The pinned serial baseline is 0.037994794617407023 s/token, which implies the
ranked M5 **achieves ≈ 410-420 GB/s on the serial decode step** — that is the
measured number, and it supersedes the earlier 560-600 GB/s peak-spec estimate.
Being at achieved bandwidth means the pinned serial leg is already near its own
roofline. Therefore `G` is bounded by roughly **1.5**, and `G >= 1.92` is
physically impossible. Consequently **`h` is small and the schedule genuinely
runs deep**; the `h ≈ 0.62` branch is dead. Corollaries:

- The candidate leg's 0.01308 s/token is 76.4 tok/s, ~**1.9x** a single-pass
  roofline, so it is unambiguously emitting multiple tokens per weight pass.
- Solving `T / (1 + h·d) ≈ 1.912` at `h = 0.2, d = 8` gives `T ≈ 4.97`, i.e.
  per-draft acceptance **q ≈ 0.85** — very plausible for a native MTP head, and
  consistent with the external per-position data below.

Two consequences that re-ranked this round's priorities:

1. **The cost model is roughly calibrated and the schedule is running deep and
   near-saturated (`h` small, `G ≈ 1-1.5`).** PR #1's forced-depth sweep
   resolves `h` directly and is still the highest-information experiment on the
   board, but the roofline already tells us which branch it will land in.
2. At `G ≈ 1`, reproducing 2.904 needs per-draft acceptance **q ≈ 0.93-0.96**
   with near-permanent cap-8 operation. That means the local longcopy fixture
   (acceptance 1.0) is **much closer** to ranked reality than I told students,
   and it means `segmentedStreakGate = 3` is **expensive rather than
   irrelevant**: a Markov estimate puts cap-8 occupancy at only ~29% of rounds at
   q=0.93 (~46% at q=0.96), worth ~5.6-6.5% of raw score.

**Median-of-8 strategy.** The published score is the mean of the 4th and 5th
order statistics over eight prompts. Improving the two *best* prompts is worth
exactly zero. Correctness (`parity_all_ok`), by contrast, is an AND across all
eight. So: optimise the middle of the distribution, and never trade fidelity for
speed anywhere.

## The shape-independent roofline knee (this revision's main result)

This is derived entirely from PR #3's merged measurements plus the quantization
format. It replaces guesswork about the depth-cost curve with a prediction that
PR #1 can falsify in one sweep.

### The knee is the same integer for every projection in the model

Affine 4-bit group-64 costs `0.5 + 4/64 = 0.5625` bytes per weight (this
reproduces the 15.14 GB checkpoint at ~26.9B parameters). For any projection
`K x N` evaluated at batch width `M`:

```
bytes = K·N·0.5625      (independent of M)
flops = 2·K·N·M
```

Both are proportional to `K·N`, so the batch width at which compute time
overtakes weight-streaming time is

```
M*  =  0.5625 · FLOPS_eff / (2 · BW_eff)
```

and **`K` and `N` cancel**. The knee is therefore *identical* for the GDN fused
in-projection (5120 x 16480), `out_proj` (5120 x 5120), both MLP projections
(5120 x 34816 and 17408 x 5120), and `lm_head` (5120 x 248320). This is
falsifiable: **if different shapes knee at different `M`, then dispatch and
occupancy — not roofline — set the curve**, and the whole model below is wrong in
an informative way.

### On this host (M4 Pro) the knee is at verify width 8, i.e. depth 7

PR #3 supplies both constants on the same host:

- `FLOPS_eff = 6.415 TFLOP/s` (quantized GEMM at M=512, 87.1% of the measured
  dense bf16 ceiling);
- `BW_eff = 227 GB/s` (14.1 GiB of weights in the 0.0673 s serial decode step).

```
M*        = 0.5625 · 6.415e12 / (2 · 227e9) = 7.9
balance   = 6.415e12 / 227e9                = 28.3 FLOP/byte
AI(M=9)   = 2 · 0.5625⁻¹ · 9                = 32 FLOP/byte   → just compute-bound
```

Verify runs at width `d+1`, so **`M* = 7.9` puts the knee at `d ≈ 7`, not at
`d = 4/5`.** The `sdpaWidthWallDepthCap = 4` / `segmentedStreakGate = 3`
boundary the round-1 briefs were built around is an SDPA-segmentation artefact,
**not** the cost discontinuity. Two more numbers follow:

```
ideal cost(9)/cost(1) = max(0.0673, 9·0.00841) / 0.0673 = 0.0757/0.0673 = 1.12
compute slope above the knee = 2·26.9e9 / 6.415e12 = 8.4 ms per extra row
compute slope below the knee ≈ 0
```

### Two-regime prediction for PR #1's marginal-cost curve

PR #3's round anchors on this host:

| leg | rounds | `Σ block_latency` | per round |
|---|---:|---:|---:|
| serial `d=0` | 64 | 4.286431789398193 | **67.0 ms** (= `V(1)+c`) |
| MTP `d=8` | 10 | 1.6095870733261108 | **161.0 ms** |

with `c = 0.00033844841851128475` s/round. The *average* marginal is
`(161.0 − 67.0)/8 = 11.75 ms/draft`, i.e. `h_avg = 0.176` against the shipped
`headStepCostRatio = 0.20`. **The scalar is roughly right on average; the
question is entirely its shape.** Building the ideal round:

- verify(9) floor = `max(67.0, 9·8.4) = 75.5 ms`;
- 8 head forwards: trunk `238,934,093 B / 227 GB/s = 1.05 ms` each = **8.4 ms**;
  plus the compact draft-vocab slice `5120·98336·0.5625 = 283 MB` = 1.25 ms each,
  so up to **18.4 ms** total;
- **ideal round = 84-94 ms against a measured 161 ms ⇒ residual tax 1.71-1.91x**,
  i.e. **67-77 ms per round unexplained**.

Distributing the ideal 94 ms over the two regimes (`6·m_lo + 2·(m_lo + 8.4) =
94.0` ⇒ `m_lo = 9.65 ms`) predicts:

| depth band | marginal | implied `h(d)` | vs shipped 0.20 |
|---|---:|---:|---|
| `d = 1..6` (bandwidth-bound) | ~9.7 ms | **~0.145** | 38% too high → **under-drafts** |
| `d = 7..8` (compute-bound) | ~18.1 ms | **~0.271** | 26% too low → **over-drafts** |

**A single scalar is wrong in both directions**, which is exactly why PR #1's
deliverable is the curve rather than a retuned constant.

### The (H, verify-slope) identifiability limit

The measured per-round marginal is

```
m(d) = H + [ V(d+1) − V(d) ]
```

Because the verify width is *always* `d+1`, the head cost `H` and the verify
width-slope are **perfectly collinear across any depth sweep at any acceptance
rate**. No depth sweep can separate them. This is harmless for the policy —
`costModelDepth` only needs the combined marginal — but fatal for attribution,
so no student should claim to have measured `H` from a depth sweep. Three ways
out, all now assigned: PR #5 measures `V(w)` in isolation, PR #4 builds the
isolated-kernel floor, and one off-diagonal `(d, w)` point (the width-9 → 10
padding experiment) identifies `H` directly.

### Retraction: `qmv` is *not* doing a full per-row weight re-read

Earlier advisor guidance to PR #1 said `C(d)` would be close to linear with a
steep slope "if `qmv` re-reads the weight tile once per row". Arithmetic refutes
it: a literal per-row re-read gives `V(9) = 9 · 67 ms = 600 ms` against a
measured whole round of 161 ms. Whatever `qmv` costs at `M = 9`, it is **not** a
9x re-read. The `h <= 0.262` feasibility bound is likewise retired: the predicted
`h_avg = 0.176` sits comfortably inside it, so it no longer discriminates
between hypotheses.

### Normalized, host-portable stop rule (in force for PR #5)

Raw ratios do not transfer between hosts; a roofline-normalized tax does.

```
BW_eff            from M=1:    K·N·0.5625 / cost(1)
FLOPS_eff         from M=512:  2·K·N·512  / cost(512)
cost_roofline(M)  = max( K·N·0.5625/BW_eff , 2·K·N·M/FLOPS_eff )
qmv_tax(M)        = cost_measured(M) / cost_roofline(M)
```

- `qmv_tax(9) < 1.35` → the kernel is near roofline, stop and retire the idea;
- `qmv_tax(9) > 2.7` → the tax is real and large, proceed to the exploitation;
- in between → width-padding sub-experiment only.

Pre-registered prediction: raw weighted `cost(9)/cost(1) ≈ 2.0-2.4`, normalized
`qmv_tax(9) ≈ 1.55-1.9`, so the **middle branch fires**.

### On the ranked M5 the knee is at least as deep, never shallower

M5 serial decode implies `BW_eff ≈ 410-420 GB/s`. Its `FLOPS_eff` is unknown;
Apple's NAX is ~4x on **prefill** but only ~25% on decode (Tech Talk 111432), so
bracket it:

| assumed M5 `FLOPS_eff` | balance | knee `M*` |
|---|---:|---:|
| 12.8 TFLOP/s (2x) | 30.5 FLOP/byte | **8.6** |
| 25.7 TFLOP/s (4x) | 61 FLOP/byte | **17.2** |

M4 Pro is 7.9. **In every corner of the bracket, M5's knee is deeper than
ours.** Two consequences that reframe the whole round:

1. **Local M4 Pro measurements systematically understate the value of deep
   drafting** and overstate the case for gating deep rounds. A local regression
   at `d = 7..8` is weak evidence of a ranked regression.
2. The M5 residual tax is correspondingly lower, roughly **1.08-1.31x** versus
   our local 1.7-1.9x.

### The (T, tax) identifiability limit at the ranked score

The published 2.904 constrains only the **product** of tokens-per-round `T` and
the residual tax, not the two factors. Both of these reproduce it:

| corner | `T` | implied `q` | implied M5 tax |
|---|---:|---:|---:|
| A (low tax) | ≈ 5.12 | ≈ 0.85 | 1.08-1.31 |
| B (high tax) | ≈ 7.0 | ≈ 0.93 | ≈ 1.7 |

Closing this needs PR #1's realised `T(d)` together with PR #5's kernel curve.
Until then every ranked projection carries both corners, which is why Flag 1's
band is 3.4-4.6 rather than a point estimate.

## Current research focus and themes

The base is far more optimized than its own documentation admits. Already
shipped and **not** to be re-proposed: the prefix replay tape /
`restoreAfterPrefixReject`, the persistent committed-history head cache, the
compact 98,304-row draft vocabulary with a fused select kernel, removal of
vocabulary-wide argmax on verify, the affine-4 group-64 head requantization, one
blocking eval per round, and the M=512 seed-prefill shape warm.

Round 1 spans **two tiers** so a plateau in one does not stall the other.

**Tier A — schedule (how deep to draft):**

- **PR #1 / qwen-edward — per-depth marginal cost curve.** `costModelDepth`
  prices every extra draft with one scalar `headStepCostRatio = 0.20`, whose own
  comment block records three inconsistent fits implying ~0.40. The head has
  since gone 4-bit and depths 5-8 now run a structurally different segmented
  verify; one scalar cannot be right on both sides of that boundary. Deliverable
  is the **curve** plus the implied `d*`, which survives even if the policy
  change is neutral.
- **PR #2 / qwen-alphonse — deep-round gate + width-9 exactness.**
  `fullAcceptStreak` resets on *any* non-full round, pinning the session to depth
  cap 4 for >=3 rounds after every single reject. Now sweeping cap-8 occupancy
  and blended ratio over q ∈ {0.70 … 0.98} rather than the single wrong q.
  Gated on the Flag 2 row gate.

**Tier B — cost (how much a round costs at all):**

- **PR #3 / qwen-thorfinn — the seed-prefill Amdahl term. MERGED, result:
  mechanism dead, measurement kept.** See the dedicated section below. `P` is
  now a measured number, not an estimate, and it is irreducible.
- **PR #4 / qwen-askeladd — the host-bound draft step.** `draft_build ≈ 2.4 ms`
  of CPU graph construction per draft step against ~1.3 ms of GPU streaming.
  Now also asked to split that into shape-rebuild vs constant-width construction
  vs genuine per-draft work, following mlx-lm #250 (non-static shapes cost ~50%
  of target-model time in a speculative path). Method redirected: the trace is
  unreachable, so build an **isolated-kernel no-overlap floor** for the decode
  round (`d` head forwards + one width-`d+1` verify) with the
  `research/prefill_floor.py` technique and compare against measured
  `block_request_seconds`. Measured ≤ floor kills the hypothesis; measured
  20-30% above *is* the host-bound cost, and would reproduce mlx-lm #990's own
  unexplained 10-28% residual.
- **PR #5 / qwen-thorfinn — the small-`M` `qmv` tax. NEW.** Part A is a
  research-only cost curve for `mx.quantized_matmul` at the exact scored shapes
  across `M = 1..12`, which brackets `vector_limit ≈ 10`. Part B is the
  exploitation: pad verify 9 → 10 to cross into `qmm_t_splitk`, or retune `qmv`
  itself in `mlx-generated/quantized.cpp` plus `kernels/quantized{,_nax}.*`.
  **Part A alone is a complete result** — a precise cost curve at the scored
  shapes feeds every future depth, tree, and batching decision.

**PR #1 method correction in force:** the depth-cost curve must now come from a
parent-clock **depth sweep** (`--local-iterate --mtp-depth d`, `d = 0..8`) with
`P` stripped first, not from the trace. The `d = 0` run is the `V(1)` anchor.
Realised-vs-offered depth must be recorded, since the schedule may not take the
depth it was offered.

## PR #3 result — the seed-prefill term is measured and irreducible

Merged at `51d7dbb902dbf01b99f7eb7d3f8301a8b62cea34`. No editable-surface file
touched (`growth = 0 / 262144`); the diff is `.gitignore` plus `research/`.
Fidelity clean on both legs (`all_tokens_matched = true`,
`emitted_token_total = 64`, `declared_rows_total = 64`,
`residual_divergence_count = 0`). W&B group
`qwen38-r1-e3-seed-prefill-amdahl`:
[`cwlqu3ok`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cwlqu3ok)
(leg decomposition) and
[`ihnmmi1b`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ihnmmi1b)
(prefill compute floor).

**The method is now the campaign standard.** Since the trace is unreachable
(next section), `P` was recovered by *parent-clock algebra*:
`decode_seconds = P + Σ block_request_seconds + N·c`. Two legs — 64 serial
rounds and 10 MTP rounds — give two equations in two unknowns. Result
`c = 338 µs` per round and **`P = 4.008616434203254 s`**. The two raw residuals
(`4.030277` serial, `4.012001` MTP) agree to **0.45%**, which is what makes the
number trustworthy.

**What it costs us.** At the ranked 512-seed / 512-decode window on this host,
`P` is **23.9% of the candidate leg** — roughly twice the 13.4% I had assumed —
but only 10.5% of the serial leg. `dscore/dprefill ≈ −0.0757` per second, so
100 ms of prefill is worth **0.0076 score**, not the 0.043 that 100 ms of decode
is worth. Honest ranked band for `P`: **13.4%-30%**, likely the lower end,
because M5 lifts compute more than bandwidth.

**Why it cannot be cut.** `research/prefill_floor.py` reconstructs `P` from
isolated kernels at the exact scored shapes and **over**-predicts by **+1.1%** —
so `asyncEval` already pipelines and there is no host slack. Quantized GEMM is
**97.0% of `P`**, running at **6.415 TFLOP/s = 87.1% of the measured dense bf16
ceiling** on this part. A physically impossible free dequant would save 12.49%,
under the 20% stop threshold. The weight-streaming floor is the wrong bound
entirely: 14.1 GiB at ~273 GB/s is 55 ms, **1.4% of `P`**.

**Three findings that outlive the mechanism:**

1. **`warmAllDepths` has no first-touch cost left on the candidate leg.** First
   block `0.1783 s` vs `p50` `0.1686 s` — only **+5.8%** — and the run maximum is
   the **last** round, not the first (`max/p50 = 1.128`, margin to the 4x
   guardrail = 2.87). The serial leg still shows classic warmup
   (`max/p50 = 1.696`, max *is* the first block). **Consequence:** prefill and
   warm work cannot buy guardrail margin, and the fixture's depth-2 3.30-3.36x
   `max/p50` is therefore **not** residual JIT. Thermal or scheduler variance
   over 512 rounds is the remaining explanation and is now its own open item.
2. **The candidate leg is two different machines.** Prefill is compute-bound
   (bandwidth floor is 1.4% of `P`). Serial *decode* moves 14.1 GiB in 0.0673 s
   = **~224 GB/s, about 82% of peak bandwidth** — which independently confirms
   the roofline argument that closes the `G` branch. **Every future proposal must
   state which of the two machines it attacks.** A compute win in prefill and a
   bandwidth win in decode are not interchangeable.
3. `gated_delta_kernel_T512` at **1.209 TFLOP/s** is the only prefill component
   far off roofline, but at 3.19% of `P` it is worth ≤ ~0.024 score even if
   zeroed. Not worth a slot.

**Reopen only if** target quantization changes such that GEMM efficiency falls
well below 87% of ceiling, or an M5 measurement shows `P` above 30% of the ranked
candidate leg *and* ranked GEMM far from its own roofline.

## Harness fact: `MLX_QWEN_MTP_TRACE=1` is unreachable

Verified in source, and it invalidated the stated method of PRs #1 and #4
(corrections sent). The per-round trace exists and emits the fields we want
(`round, d, acc, draft_build_us, verify_build_us, eval_wall_us, readout_us,
commit_us, upkeep_us, round_us` at `Qwen36MTPBlockSession.swift:~1062-1078`),
but it writes to **worker** stderr, which is discarded:

- `Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift:2046` and `:2207` —
  `emit: options.forwardsWorkerStderr ? nil : { _ in }`.
- `runtimeWorkerOptions(blockedGoldenPath:forwardsWorkerStderr:)` at
  `Sources/MLXFastCLI/main.swift:2222-2224` defaults `false` and returns
  `forwardsWorkerStderr && !officialRun` at `:2301`.
- The only caller that enables it is `runDFlashBenchmark`
  (`main.swift:1404-1412`), gated on `MLX_DFLASH_TRACE_CACHE_SEAM=1`.
  `runQwenMTPVerify` (`:1748-1750`) and `runQwenMTPTimed` (`:1799-1801`) both
  take the default.

Wrapping the worker to capture its stderr is blocked by
`enforceMetallibFingerprint` and the sandbox `allowedExecutablePath`.

**Substitutes that do work**, both merged on the current base:

- `research/capture-cli.sh` — an argv-passthrough tee for `MLXFAST_SWIFT_BIN`.
  This is not optional plumbing: `benchmark-qwen-mtp.sh` `mktemp`s its report
  directory and deletes it on `EXIT`, so `block_request_seconds` and
  `decode_seconds` **never survive a run without it**.
- `research/prefill_amdahl.py` (two-leg parent-clock decomposition),
  `research/prefill_floor.py` (isolated-kernel floor at exact scored shapes),
  `research/prefill_floor_summary.py`, `research/run-amdahl-measurement.sh`.

## Newly established facts worth acting on

- **`d == 0` is an unrecoverable absorbing state.** `Qwen36MTPBlockSession.swift`
  L761 returns *before* `recordAcceptOutcome` (L610-639) and the streak update
  (L1045-1047), so once the schedule picks depth 0 the EMAs and streak freeze
  permanently — no probe, no decay, no recovery. Simulation: botany freezes on
  40/40 seeds by round ~51. Negligible at q≈0.95 but a real tail risk on a hard
  hidden prompt, and the median-of-8 makes one slow prompt cheap while a *frozen*
  prompt is catastrophic. **Cheap fix: update EMA[0] on the serial path from the
  committed token's own top-2 margin.**
- **`positionAcceptEMA` is never reset per prompt**, is initialised optimistically
  (`0.85·0.98^i`), and its half-life is `ln0.5/ln0.85 = 4.265` rounds — the
  in-code "~9 rounds" comment is ~2x optimistic, and deep positions are
  reach-gated to an effective half-life of tens of rounds. Round 1 always drafts
  4 unconditionally.
- **A fully-accepted round pulls `positionAcceptEMA[acceptedCount]` toward 0.95**
  (L620-637), biasing the schedule upward by one position — this optimism
  transfer is the mechanism that lets the session climb to the cap at all.
- **Dead code confirmed:** the `conf` gate can never trigger a skip
  (`conf ∈ [0.5,1)` vs a k=0 threshold of exactly h=0.2); the L446-449
  "OPERATOR K-TEST VARIANT" default policy closure is overridden at init
  (L197-200).
- **`Sources/MLXFastModel/Qwen35{Attention,Block,GatedDelta,MLP,Model,Ops,RoPE,FastEngine}.swift`
  is editable but NEVER EXECUTED** — `Qwen35FastPathReadiness.swift:11-19`
  hardcodes false, so `selectQwen35ExecutionBackend` always returns
  `.libraryOracle`. The live target is
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` plus
  `Qwen36MTPBlockSession.swift`. Any experiment that edits the former is
  measuring nothing.
- **Verify width is one row short of a much better kernel.**
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1415,1483`
  sets `vector_limit ≈ 10` at K=N=5120 and dispatches `qmv` below it,
  `qmm_t_splitk` at or above. Widths 1-9 therefore all take `qmv`, tuned for
  `M = 1`, and `eval_wall` grows 79 → 89 → 106 ms across widths 7 → 8 → 9 with
  *increasing* deltas (+10 then +17), so no linear `a + b·M` fits. **It is not a
  full per-row re-read** — that would give `V(9) ≈ 600 ms` against a measured
  161 ms round. The roofline knee at `M* = 7.9` explains the acceleration
  without any re-read: widths 8-9 are the first that cross into the
  compute-bound regime. That host dispatch file is *not* editable, but the
  shapes we request are ours to choose.
- **We can honestly build representative local fixtures.**
  `Sources/MLXFastCLI/main.swift:761-840` exposes
  `generate-golden --prompt-file … --steps N`, and
  `benchmark-qwen-mtp.sh:103,107` honours
  `MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE`. The ranked prompt names are all
  public-domain classics, so same-genre local seeds are legitimate test-set
  construction — **not** hidden-prompt specialisation, provided candidate code
  stays prompt-independent. Every current experiment is measured on a copy task
  with acceptance 1.0; this is the single largest infrastructural gap.
- **`Qwen36MTPTarget` is an `AnyObject` protocol**
  (`Sources/MLXFastModel/Qwen36MTPTarget.swift:36`), so a research-only stub
  conformer can drive the **real** `costModelDepth` / `recordAcceptOutcome` over
  synthetic acceptance with **zero GPU**. Nothing today exercises the depth
  policy over a 512-round horizon; the local window is 64-128 tokens, 4-8x short.
- **Rollback is not free above draftCount 1.** `restoreAfterPrefixReject`'s "no
  repair at any depth" claim holds only at draftCount==1; above that (L1188-1202)
  a reject calls `replayRecurrentPrefix`, i.e. 48 serial GDN layer scans on the
  critical path, and a `canReplayPrefix` failure falls back to a full repair
  forward (L1006-1028) — a second blocking eval.

## External reference implementation: mlx-lm PR #990

`ml-explore/mlx-lm` PR #990 (open, unmerged, 31 commits, 73 comments, zero
maintainer review) is a native-MTP speculative-decoding reference for **our
exact architecture family** — the author enumerates `64 layers (48 GDN + 16
attn)`, `hidden 5120`, `24 / 4 / 256` heads, `lm_head [H, 248320]`. It is the
single most relevant external artifact found so far. <https://github.com/ml-explore/mlx-lm/pull/990>

**Their depth scaling is negative, and that is the headline.** On stock MLX
kernels, deeper drafting *loses*:

| source | model / host | d=1 | d=2 | d=3 | d=5 |
|---|---|---|---|---|---|
| JJJYmmm | Qwen3.5-9B 4-bit, **M5** | 1.52x | 1.36x | 1.13x | **0.80x** |
| AirRunner | Qwen3.6-27B 4-bit, M4 Pro | 1.65x | 1.56x | 1.33x | — |

Named cause: *"stock MLX's `qmv` kernel is tuned for `M=1` and becomes
increasingly inefficient as `M` grows"*. This is the same `vector_limit ≈ 10`
cliff we found independently in `quantized.cpp:1415`.

**Our base reaches raw 2.904 with deep drafting, which their numbers say should
be impossible on stock kernels.** So either our verify path is already
materially better than theirs, or our schedule's conservatism
(`segmentedStreakGate = 3`, `sdpaWidthWallDepthCap = 4`) is an empirically-found
optimum sitting exactly at the qmv cliff. PR #2 Part B separates these, and a
regression there is now a fully expected and publishable outcome.

Their unattributed residual is the strongest sizing anchor we have for host
overhead: a bandwidth model predicts per-round overhead `beta + delta = 1.081`,
every measured run lands at **1.19-1.36**, and the author attributes the gap to
*"kernel launch overhead, MLX graph evaluation, or similar costs"*. **10-28% of
round cost unexplained by bandwidth on the same architecture** — a far better
anchor than mlx PR #3920's 2%.

Four concrete techniques from that PR, ranked for us (their probabilistic /
residual-sampling machinery is inapplicable — our contract is exact-match, i.e.
greedy):

1. **Prefill the MTP head cache during prompt prefill.** Acceptance 86.0% vs
   82.4% (9B 4-bit) and 87.1% vs 83.1% (4B bf16), at ~4 KB/token of permanent
   head KV. Our 512-token seed makes this 2 MB. Assigned as a deliverable to
   PR #3.
2. **Fuse the accepted-draft cache commit into the next draft forward**
   (`cache_commit=(hidden_at_confirmed, draft_tok)`), saving one head forward per
   accepted round. Assigned as a deliverable to PR #4.
3. **Keep linear projections un-split around the recurrent snapshot.** They
   project `qkv/z/b/a` over the full `S` *before* the confirmed/draft split and
   split only the recurrence, giving snapshot overhead `beta ≈ 1.009` — 0.9% of a
   pass. Their depth>1 per-position variant instead iterates the confirmed prefix
   one token at a time, which serialises the recurrence and is a plausible
   second cause of their depth-2 regression, independent of qmv. Under check
   against our `processChunkStashingPrefix`.
4. **`lm_head` is ~70% of head cost** (635.7 MB of 911 MB) over the 248,320
   vocabulary. We already ship a compact 98,304-row draft vocab, so this is
   partly banked; under check whether every head step uses it.

**Per-position acceptance priors** (JJJYmmm, 9B 4-bit, depth 5): p1 82.5%,
p2 64.0%, p3 47.6%, p4 33.9%, p5 23.4%. Our `positionAcceptEMA` initialises to
`0.85·0.98^i` = 0.85 / 0.833 / 0.816 / 0.800 / 0.784 — a **3.3x overestimate at
position 5**. Their model is smaller so the levels are not ours, but the shape
matters: their conditional acceptance **degrades with depth** (constant-`q` fit
to p1 predicts p5 = 0.38, not 0.234) whereas `0.98^i` encodes near
depth-independence. PRs #1 and #2 now both emit realised per-position acceptance
over a 512-token window so we can re-fit against our own data.

**Two evidence downgrades, both retracted to PR #4.** `ml-explore/mlx` #3920 is
a *closed, unmerged* PR with zero comments reporting only +2.0-2.8% decode, not
an open issue establishing host-bound decode. `ml-explore/mlx-lm` #250's
200-210 ms vs 130-140 ms measurement is real, but its non-static-shape
*explanation* is the reporter's own flagged guess on an issue with zero replies.


## Pre-emptively rejected (do not re-assign without new evidence)

- **Further draft-vocabulary trimming. CONTESTED — see queue item 16.** The
  ~0.6 ms note contradicts arithmetic: the padded compact slice is
  `5120 · 98336 · 0.5625 = 283 MB`, which at the measured 227 GB/s cannot be
  read in under **1.25 ms**. Either the note is stale, the slice is not fully
  read per draft step, or `BW_eff` for that access pattern is much higher than
  the whole-model figure. This changes the head-cost term in *every* campaign
  cost model, so PR #5 was asked to settle it. The original note follows.
- **Further draft-vocabulary trimming.** Compact-head read is only ~0.6 ms per
  draft step (`Qwen35.swift:2054-2060`), and a uniform 49,152 halving already
  regressed (accept 1.00 → 0.877) because three committed argmax ids lived above
  the cut. Reopen only if the 0.6 ms figure is shown wrong.
- **Skipping the 16 full-attention layers as a self-drafter.** Literature reports
  alpha ≈ 0.038 on sequential hybrids. Pre-registered negative.
- **Gathered / scattered dynamic vocabulary heads.** Reported slower than a
  static contiguous slice, and affine-4 group-64 packing makes it worse. Any
  shortlist must stay contiguous and a multiple of 64.
- **Cutting the seed prefill (PR #3).** Measured, decomposed, and irreducible:
  97% quantized GEMM at 87% of the measured dense bf16 ceiling, with a
  physically-impossible-free-dequant ceiling of 12.49%. Reopen conditions are
  recorded in the PR #3 section above.
- **Chunked prefill (2x256, 4x128) and chunkwise-parallel GDN prefill.**
  `GatedDelta.swift:128` is a strictly sequential per-timestep recurrence in
  which `T` is a kernel *input*, not part of the grid, so chunking is
  structurally worse, and the whole GDN prefill kernel is 3.19% of `P`.

## Potential next research directions

Ordered by expected value per student-slot. Item 0 is now **assigned as PR #5**
and remains the strongest single hypothesis on the board. Items 1-3 are
infrastructure that makes every later experiment more trustworthy.

0. **Retune the small-`M` `qmv` quantized kernel for M = 2..9.** *(ASSIGNED —
   PR #5, qwen-thorfinn.)* Every verify width we can reach dispatches `qmv`
   (`quantized.cpp:1415`, `vector_limit ≈ 10` at K=N=5120), and `qmv` re-reads
   the weight tile once per row because it is tuned for `M = 1`. Two independent
   lines of evidence point here: our own `eval_wall` growing 79 → 89 → 106 ms
   across widths 7 → 8 → 9 — note the deltas +10 then +17 are *increasing*, so
   no linear `a + b·M` fits — and mlx-lm PR #990's depth cliff with `qmv` named
   as the cause. That PR also notes a private fork retuning `qmv` for `M = 3..6`
   with `4-simdgroup` / `unroll_count(4)`. **The kernel sources are inside our
   editable surface** (`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`
   for the JIT strings plus `kernels/quantized{,_nax}.{h,metal}`; `_nax` runs on
   the ranked M5), even though the host dispatch file is not. This changes the
   *shape of the cost curve every schedule experiment is fighting*, so it
   dominates further schedule tuning. `python3 research/twin_audit.py` is
   mandatory because the readable `.metal` and the generated twin must move
   together. High ceiling, high numerical risk — needs an exact-row gate, not
   just argmax matching.
0b. **Pad the verify batch from 9 to 10 rows** to cross `vector_limit` into
   `qmm_t_splitk`. **Now double-purpose:** it is also the only cheap way to
   break the `(H, verify-slope)` collinearity, because it is the one reachable
   `(d, w)` point off the `w = d+1` diagonal. PR #5 owns the kernel-level curve
   that *predicts* the effect; PR #1 owns the system-level end-to-end data point.
   Deliberately independent, so the two form a cross-check rather than a
   duplicate. **Caveat now on record:** S = 10 leaves several width-gated fast
   paths — `fusedInProjections` (`S <= 9`), `qwen35PackedGDNPreworkKernel`
   (`S >= 3 && S <= 9`), and the `AttentionUtils` wide-chunk split
   (`qL >= 6 && qL <= 9`) — so the honest question is whether the kernel gain
   survives losing them. A clean negative here would explain why `vector_limit`
   sits where it does.
1. **Representative local prose goldens** via `generate-golden --prompt-file` and
   `MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE`. Highest leverage available: right now
   every measurement we take is on a copy task at acceptance 1.0, and the
   schedule experiments are precisely the ones that behaviour distorts.
2. **Zero-GPU policy simulator** — a research-only `Qwen36MTPTarget` stub driving
   the real `costModelDepth` / `recordAcceptOutcome` across acceptance regimes at
   a full 512-round horizon. Turns schedule questions from GPU-bound into free,
   and would have caught the `d==0` absorbing state.
3. **Schedule-state hygiene**: fix the `d==0` absorbing state, reset
   `positionAcceptEMA` per prompt, and re-fit the `0.85·0.98^i` prior against
   measured data. Small, cheap, and it removes a catastrophic tail risk.
4. **Move the 511-row head priming out of the first timed drafting round.**
   `headHistoryCache` is declared at `Qwen36MTPBlockSession.swift:149` but first
   *written* at `:819-822`, inside the round path — so the head's 511-row seed
   prefill is paid inside the first scored drafting round, not in `begin()`.
   PR #3 shows the candidate leg's first block is only +5.8% over `p50`, so the
   total is small; this is a **block-latency distribution** item, relevant to
   guardrail margin rather than to the score. Low priority until the thermal item
   below is understood.
5. **Compiled MTP round.** `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/CompiledDecode.swift`
   is editable and gates the compiled path at L38 to "no MTP, no SSM". Making the
   GDN recurrent state traceable is the gating sub-problem. Attacks the 2-4k
   graph-nodes-per-token host bound (mlx#3920) at its root. Promote only if PR #4
   shows graph construction dominates.
6. **Rollback-free GDN verification** — arXiv 2607.06763 formulates tree
   verification over Gated DeltaNet layers as a masked triangular solve so
   committed state is never speculatively written; reports 4.37x on Qwen3.6 27B.
   STree (2505.14969) is the cheaper accumulated-transition variant. This is the
   most credible large swing in the literature for exactly our 48+16 hybrid.
7. **Tree drafting** (best-first over top-K-per-depth marginals) sized by the
   measured `b(n)` verification-latency curve. Caveat from SpecMQuant
   (2505.22179): on **W4A16** the verify/decode ratio is **1.8** versus <1.2 for
   FP16, so widening is unusually expensive on our 4-bit backbone — trees must be
   converted to sequences aggressively.
8. **Per-position margin-conditioned acceptance.** The `conf` clamp is applied
   only at depth 0, with an unfitted temperature, and is provably dead. The head
   emits top-2 evidence at every position; a *fitted* margin model across all
   positions is the natural successor to PR #1 (AdaEDL, SVIP, SpecDec++).
9. **Repair-path telemetry.** `rollbackRoundCount` does not separate a cheap
   replay from the expensive full re-forward. We cannot currently see how often
   the expensive path fires — the highest-value measurement not yet taken.
10. **Eager checkpoint cost at low depth.** `S == 2` rounds write ~144 MiB of
    mid-state unconditionally; the packed-GDN-prework kernel is gated at `S >= 3`.
    Extending the replay tape to S=2 removes a per-round write.
11. **Composition round.** PRs #1-#4 are deliberately orthogonal; compose once two
    or more land independently measured. `program.md` names runtime x speculative
    interaction as an open area.
12. **GQA query-head pairing** to break the `qL*gqa <= 32` width wall. High
    ceiling, high risk, and it needs a fresh hexfloat row gate first.
13. **ReDrafter-style head restructuring** (2403.09919) — the only real Apple
    Silicon precedent, implemented in MLX, up to 2.3x. Hydra (2402.05109) adds
    sequential dependence between heads, which LK Losses (2602.23881) identifies
    as the cause of MTP depth-decay. Hold for a genuine plateau.
14. **Cleanup assignment** once a slot frees: the stale header at
    `Qwen36MTPBlockSession.swift:22-43` and the matching framing in
    `Tests/MLXFastTests/QwenMTPRollbackContractTests.swift:14,78,108,190`, the
    dead `Sources/MLXFastModel/Qwen35*.swift` tree, the dead L446-449 policy
    closure, and the dead `conf` gate. **Owed.** Deletion is the default: stale
    experiment paths are a mis-run risk, and the winning behaviour should be the
    single obvious main path.
15. **Explain the fixture's depth-2 3.30-3.36x `max/p50`.** PR #3 removed the
    obvious suspect — it is **not** residual JIT, because the candidate leg's
    first block is only +5.8% over `p50` and its maximum is the *last* round.
    That leaves thermal throttling or scheduler variance accumulating over 512
    rounds. This is the single largest unexplained number touching the stall
    guardrail, which fails closed. Worth a slot once the current four report.
16. **Settle the compact-draft-vocabulary read cost.** The in-code note says
    ~0.6 ms per draft step (`Qwen35.swift:2054-2060`); the padded slice is
    `5120 · 98336 · 0.5625 = 283 MB`, which at the measured 227 GB/s cannot be
    read in under 1.25 ms. One of the two is wrong, and the answer sets the head
    term `H` in every cost model on this board — 8 draft steps per round means
    the difference is 5 ms of a ~161 ms round. Asked of PR #5 as a side
    deliverable; promote to its own slot if that PR cannot answer it.
17. **Off-diagonal `(d, w)` identification of `H`.** Every depth sweep confounds
    the head cost with the verify width-slope because `w = d+1` always. A single
    run at `d = 8, w = 10` (the width-padding experiment) breaks the collinearity
    and yields `H` directly. Cheap, and it converts PR #1's curve from a policy
    input into a mechanistic decomposition.
