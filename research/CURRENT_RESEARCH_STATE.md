# SENPAI Research State

- **2026-08-16 14:45 UTC** — round 1 of the `qwen38-mlx-senpai-r1` campaign. Base
  `e20268e9c2c1f35c2d75221d059e75bb95768ef6`, upstream `7351e62674bc600f0ca148d3a1b0604716a09db6`.
- **Most recent research direction from the human researcher team:** none received
  yet this launch. Two items below are flagged for the team's attention and are
  the first things to raise when contact is made.

## Where the campaign actually stands

- Promoted frontier: submission `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, official
  score **2.9042110287045**. That result is the *public* frontier bootstrapped
  into this campaign, **not** Senpai-authored. Senpai has zero official
  submissions so far; `campaign-ledger.md`'s submission table is empty.
- Same-host local reference (AWS Birch/Alphonse, M4 Pro 48 GB, low-memory
  profile): directional `1.4708805115725638`, serial `0.1292338595` s/token,
  MTP `0.0878615621` s/token, effective draft `5.4`, acceptance `1.0`,
  exact `64/64`.

### Flag 1 — the plausibility gate is close

The operator plausibility gate is **3.0** and the frontier is **2.904**. That is
~3.3% of published headroom. Per `program.md` the gate is administrative and
fail-closed: a legitimate median above it is *rejected*, not clamped, and would
need an operator policy revision before it could publish. Round-1 work is
plausibly large enough to cross it. **This needs a decision from the human team
before, not after, a crossing submission.**

### Flag 2 — a shipped fidelity gap at verify width 9

`segmentedVerifyDepthCap = 8` implies `rows_per_round = 9`. The recorded
hexfloat row-gate evidence covers **widths 6-8 only**; width 9 was in the
*original drift set*. `attentionWithCacheUpdate` mechanically handles `qL 6...9`
(split `5 + 4`), but that is not a measurement. Assigned as the blocking Part A
of PR #2.

### Flag 3 — the file header is stale and misleads readers

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:22-43` describes a session that
no longer exists. It claims one fresh head cache per round (a persistent
committed-history cache has shipped) and an expensive repair forward (the prefix
replay tape deleted it). `Tests/MLXFastTests/QwenMTPRollbackContractTests.swift`
carries the same stale framing at lines 14, 78, 108, 190. Queued as a cleanup
assignment once a round-1 slot frees.

## Current research focus and themes

The base is far more optimized than its own documentation admits. Before
assigning, I verified that the following are **already shipped** and must not be
re-proposed: the prefix replay tape / `restoreAfterPrefixReject`, the persistent
committed-history head cache, the compact 98,336-row draft vocabulary with a
fused select kernel, removal of vocabulary-wide argmax on verify, the affine-4
group-64 head requantization, one blocking eval per round, and the M=512
seed-prefill shape warm.

Round 1 therefore deliberately spans **two different tiers** so a plateau in one
does not stall the other.

**Tier A — schedule (how deep to draft):**

- **PR #1 / qwen-edward — per-depth marginal cost curve.** `costModelDepth`
  prices every extra draft with one scalar `headStepCostRatio = 0.20`. The
  constant's own comment records four contradictory fits and instructs
  "re-fit after every head-variant change"; the head has since gone 4-bit
  (~3.6x smaller) and depths 5-8 now run a structurally different segmented
  verify. One scalar cannot be right on both sides of that boundary. Deliverable
  is the **curve**, which survives even if the policy change is neutral.
- **PR #2 / qwen-alphonse — deep-round gate + width-9 exactness.**
  `fullAcceptStreak` resets on *any* non-full round, so every single reject pins
  the session to depth cap 4 for >=3 rounds. The per-position acceptance EMAs
  already measure head health directly and recover faster than a streak.
  Gated on the width-9 row gate above.

**Tier B — cost (how much a round costs at all):**

- **PR #3 / qwen-thorfinn — the seed-prefill Amdahl term.** The fixture charges
  the 512-token seed prefill *inside* the timed decode window, and `begin` was
  measured at ~0.9 s. Since `score = (P + D_s)/(P + D_c)` and the pinned serial
  leg does not receive candidate improvements, cutting `P` is worth several
  times any decode-loop tweak — and it gets *more* valuable as decode gets
  faster. This is the largest single untouched lever on the board.
- **PR #4 / qwen-askeladd — the host-bound draft step.** `draft_build ≈ 2.4 ms`
  of *CPU graph construction* per draft step, against ~1.3 ms of GPU streaming.
  The draft step is host-bound. Deliverable is the breakdown first, optimization
  second.

## Pre-emptively rejected this round (do not re-assign without new evidence)

- **Further draft-vocabulary trimming.** The compact-head read is only ~0.6 ms
  per draft step (`Qwen35.swift:2054-2060`), and a uniform 49,152 halving already
  regressed (accept 1.00 -> 0.877, 21.1 -> 22.8 ms/token) because three committed
  argmax ids lived above the cut. Ceiling is small, downside is not. A
  position-decaying contiguous shortlist was designed and then dropped for the
  same reason. Reopen only if the 0.6 ms figure is shown to be wrong.
- **Skipping the 16 full-attention layers as a self-drafter.** Literature reports
  alpha ~= 0.038 on sequential hybrids. Pre-registered negative.
- **Gathered / scattered dynamic vocabulary heads.** DynaSpec reports these can
  be slower than a static contiguous slice, and affine-4 group-64 packing makes
  that worse here. Any shortlist must stay contiguous and a multiple of 64.

## Potential next research directions

1. **Per-position margin-conditioned acceptance.** The depth-0 clamp
   `conf = 1/(1+exp(-margin/2.0))` is applied *only* at depth 0 with an unfitted
   temperature. The head produces top-2 evidence at every position. Extending a
   *fitted* margin model to all positions is the natural successor to PR #1 and
   is well supported by AdaEDL / SVIP / SpecDec++. Held back this round purely
   to avoid three students editing `costModelDepth` at once.
2. **Repair-path telemetry.** `rollbackRoundCount` does not distinguish a cheap
   `restoreAfterPrefixReject` replay from the expensive full re-forward fallback.
   We currently cannot see how often the expensive path fires. Highest-value
   *measurement* not yet taken.
3. **Eager checkpoint cost at low depth.** K=1 rounds still pay eager
   checkpoints (~144 MiB of extra writes per round; gate is `S >= 3` at
   `Qwen35.swift:991-1062`), and the per-round snapshot looks near-dead.
4. **Composition round.** PRs #1-#4 are deliberately orthogonal. Once two or more
   land independently measured, compose them and re-measure — `program.md`
   explicitly calls out interactions between runtime and speculative wins as an
   open area.
5. **Prefill chunk geometry as a first-class target.** If PR #3 confirms prefill
   is a large share of the candidate leg, the GDN scan's T-chunking and the
   quantized gemm path at M=512 become a research area of their own.
6. **ReDrafter-style head restructuring.** The only real Apple Silicon precedent
   in the literature (1.37x on M1 Max, 2.3x on M2 Ultra). A bigger swing to hold
   for a genuine plateau, and it interacts with the editable head manifest.
