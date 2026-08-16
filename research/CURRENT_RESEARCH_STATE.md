# SENPAI Research State

- **2026-08-16 16:20 UTC** — round 1 of the `qwen38-mlx-senpai-r1` campaign, four
  experiments in flight. Base `e20268e9c2c1f35c2d75221d059e75bb95768ef6`,
  upstream `7351e62674bc600f0ca148d3a1b0604716a09db6`.
- **Most recent research direction from the human researcher team:** none
  received yet this launch. Three items are flagged below for the team; Flag 1
  needs an answer *before* a crossing submission, not after.

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

### Flag 1 — we are ~3.3% from a fail-closed rejection gate

The operator plausibility gate is **3.0**; the frontier is **2.904**. Per
`program.md` the gate is administrative and fail-closed: a legitimate median
above it is **rejected as a measurement fault or benchmark escape, not clamped**,
and publishing above it requires the operator to revise policy first. Round-1
work is plausibly large enough to cross it. **The human team needs to tell us
whether to (a) hold a crossing candidate unsubmitted pending a policy revision,
(b) submit and accept a rejection as evidence, or (c) deliberately submit a
sub-gate variant.** Until answered, no official submission should be sent.

### Flag 2 — a shipped fidelity gap at verify width 9

`segmentedVerifyDepthCap = 8` implies `rows_per_round = 9`. The recorded
hexfloat row-gate evidence covers **widths 6-8 only**; width 9 was in the
*original drift set*. `attentionWithCacheUpdate` mechanically handles `qL 6...9`
(split `5 + 4`), but that is not a measurement. This is more load-bearing than it
first appeared: under the corrected regime below, the frontier operates at
width 8-9 most of the time, so the untested width is on the hot path. Assigned
as the blocking Part A of PR #2.

### Flag 3 — a live stall-guardrail hazard that penalises success

`scoring_semantics.stall_guardrail` rejects a run whose max block latency
exceeds **4x p50**. The candidate leg currently measures **3.30-3.36x**, i.e.
~19% of margin. `max_block` is the one-time first block after prefill, so **any
steady-state decode speedup raises that ratio**. A large win can therefore
disqualify itself. Not enforced anywhere in this repository (it is box-owned),
so we cannot test it locally — PRs #3 and #4 both carry a reporting deliverable
for it.

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

Two consequences that re-ranked this round's priorities:

1. **Either the cost model is roughly calibrated and the schedule is running
   deep and near-saturated (`h` small, `G ≈ 1`), or most of our 2.904 is general
   target speed (`h ≈ 0.62` forces `G >= 1.92`).** Both branches are actionable
   and they point at different work. PR #1's forced-depth sweep resolves `h`
   directly and is now the highest-information experiment on the board.
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

- **PR #3 / qwen-thorfinn — the seed-prefill Amdahl term.** The 512-token seed
  prefill is charged *inside* the timed leg, and `begin` measured ~0.9 s. Since
  the pinned serial leg does not receive our improvements, cutting `P` is scored
  in full and gets *more* valuable as decode gets faster.
- **PR #4 / qwen-askeladd — the host-bound draft step.** `draft_build ≈ 2.4 ms`
  of CPU graph construction per draft step against ~1.3 ms of GPU streaming.
  Now also asked to split that into shape-rebuild vs constant-width construction
  vs genuine per-draft work, following mlx-lm #250 (non-static shapes cost ~50%
  of target-model time in a speculative path).

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
  `qmm_t_splitk` at or above. Widths 1-9 therefore all take `qmv`, which
  **re-reads the weights per row** — the direct cause of `eval_wall` growing
  79 → 89 → 106 ms across widths 7 → 8 → 9. That host dispatch file is *not*
  editable, but the shapes we request are ours to choose.
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

## Pre-emptively rejected (do not re-assign without new evidence)

- **Further draft-vocabulary trimming.** Compact-head read is only ~0.6 ms per
  draft step (`Qwen35.swift:2054-2060`), and a uniform 49,152 halving already
  regressed (accept 1.00 → 0.877) because three committed argmax ids lived above
  the cut. Reopen only if the 0.6 ms figure is shown wrong.
- **Skipping the 16 full-attention layers as a self-drafter.** Literature reports
  alpha ≈ 0.038 on sequential hybrids. Pre-registered negative.
- **Gathered / scattered dynamic vocabulary heads.** Reported slower than a
  static contiguous slice, and affine-4 group-64 packing makes it worse. Any
  shortlist must stay contiguous and a multiple of 64.

## Potential next research directions

Ordered by expected value per student-slot. Items 1-3 are infrastructure that
makes every later experiment more trustworthy.

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
4. **Pad verify width 9 → 10** to cross `vector_limit` into `qmm_t_splitk`
   weight reuse. Cheap to test, potentially large, directly explains the
   measured width cost curve.
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
    closure, and the dead `conf` gate.
