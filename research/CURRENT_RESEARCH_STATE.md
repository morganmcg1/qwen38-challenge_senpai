# SENPAI Research State

- **2026-08-23 14:25 UTC**

## Most recent research direction from the human researcher team

No new human direction since Issue #22 (2026-08-21). The standing direction is
unchanged and is being executed: keep four students productive on independent
current-frontier questions, adopt the promoted editable surface before the next
official submission, publish candidate implementations and evidence early,
submit autonomously whenever a clean candidate has credible evidence, and
continue research after every result.

One correction to item 3 of that direction, on a fact discovered today. The
student role holds **no `git push` capability**; the only path to the remote is
the guarded lease-push inside `submit_experiment_result`. The "publish early"
requirement is therefore satisfied by delivering a composable piece as a
**terminal result** and moving the follow-on work into a fresh assignment, not
by asking students to push mid-experiment.

## Where the campaign stands

- **The bar: `ec24d591` newjordan = 3.72911001**, source `0863b06a`, which is
  also `upstream/main`. Unchanged this cycle.
- **Campaign best: `0cf1637e` = 3.68278758** (tree `e09d6aa7`), rejected. Still
  our scientific frontier.
- **Last promoted senpai row: `623e77a` = 3.52085227**, 2026-08-22T14:12Z. It
  has been frozen for **24 hours** while the bar moved 3.52 → 3.729.
- **The official slot is FREE.** `749da2cf` resolved rejected at
  **3.45192370143778**. Six rival rows are validating; none of ours.
- Advisor base `14247cce`. Growth budget remains tight; thorfinn's import
  returns a large block of it.

## The two facts that dominate everything else

### 1. Our submission scatter is larger than every mechanism we build

Last six senpai receipts: 3.66219, 3.57503, 3.61655, 3.42654, 3.68279, 3.45192.
Mean 3.56917, **sd 0.10764 = 3.02 %**. Portfolio mechanisms are +0.19 % to
+0.73 % each — **4x to 16x smaller than the observed scatter**.

Part of that is genuine tree difference and part is ranked-runner nuisance; six
points cannot separate them. Until E154 R1 bounds it, the working policy is:
**stack every finished mechanism and submit rarely.** Submitting mechanisms one
at a time is close to a random walk against a bar that moves +0.25 % per rival
promotion.

### 2. FINDING 267 now has a rival hypothesis of equal standing

Our editable tree is **+0.7275 % slower on the ranked candidate leg** than
`0863b06a`, for no mechanism we can name. Fitted against `s_p = 903 us x
drafting_rounds_p / decode_seconds_p`, all four senpai rows read `k` between
+0.28 and +1.16 steps with `R^2` 0.79 to 0.90, against a rival median near
+0.02; probability of four such draws is about 6e-4.

- **Hypothesis A, residency pressure** — the extra kernel variants and resident
  tensors our tree compiles and holds. Thorfinn's E155 probe tests it.
- **Hypothesis B, dispatch cost (new this cycle)** — more Swift on the hot path
  means more MLX ops enqueued or worse instruction locality, which presents
  exactly as a small, diffuse, mechanism-free slowdown. Edward's E154 R2 tests
  it.

Hypothesis B arrives from Edward's R3 measurement: **one host readback costs
1,262.6 µs/round against a 524.5 µs total-leg round.** One synchronisation costs
2.4 entire rounds, so the CPU must run far ahead of the GPU — and the campaign
has never measured which side saturates. If the round is dispatch-bound, the
number of ops and command buffers per round matters more than the speed of any
kernel, **and every local kernel win is partly an artifact**, because a shared
CPU-side dispatch term appears in both legs of the local ratio and cancels while
a GPU-side kernel win does not. That would also explain why our local winners do
not transfer.

## Current research focus

**Adopt the promoted frontier surface, stack every finished mechanism onto it,
and submit one composite — not a sequence of single mechanisms.**

Corrected composition arithmetic, multiplicative from `0cf1637e` = 3.68278758:

| composite | score | margin vs bar 3.72911 |
| --- | ---: | ---: |
| import `0863b06a` (+0.7275 %) | 3.70958 | −0.0195 |
| ∘ E151 R1 NAX 128x32 seed retile (+0.505 %) | 3.72831 | −0.0008 |
| **∘ E153 R1 leaf16 (+0.188 %)** | **3.73532** | **+0.0062** |
| ∘ E151 R2 affine double buffer (+0.419 %) | 3.75097 | +0.0219 |
| ∘ E153 R2 merged SDPA (+0.29 %) | 3.76185 | +0.0327 |

The earlier "import ∘ E151 R1 ≈ 3.748, margin +0.019" was an arithmetic error.
**Import plus one mechanism is a tie, not a win.** Minimum viable candidate is
import ∘ R1 ∘ leaf16; the target is all five, because at sd 3.02 % a +0.0062
margin is a coin flip and +0.0327 is a real bet.

FINDING 269 removes the guesswork about what the import costs. Surviving in the
frontier: `derivedClusterRowsPerLeaf = 8`, `buildDerivedClusterIndex`,
`draftTokenIDWithDeclaredRerank`, `segmentedVerifyDepthCap`, `depthPriceArm`.
Deleted: `qwen35ClusterCentroidQMV`, `onePass67` and the one-pass width table,
`passBoundaryTierFactor`, `pb6`. The frontier NAX header carries zero `kE147`
identifiers.

## Four live experiments, one per student, one per physical Mac

1. **#154 edward — the anchor receipt and the boundedness question.** New this
   cycle. **R0** submits the unmodified base `14247cce` officially with zero
   mechanism, satisfying RULE 159 and pricing the un-receipted E147 + E135 +
   E149 merge block. It is the direct test of FINDING 267: an anchor near 3.656
   means our own merges carry the frontier deficit and the import recovers a
   regression we introduced; near 3.683 means the deficit predates them and the
   +0.7275 % import pricing needs re-deriving. Pre-registered point estimate
   **3.6694**, 80 % interval **3.6144 to 3.7244**. **R1** bounds the ranked
   nuisance floor and derives the minimum composite pp worth a slot. **R2** is
   the GPU-bound versus dispatch-bound discriminator by side-pure delay
   injection.
2. **#152 thorfinn — import the promoted editable surface.** The campaign
   critical path. Restructured this cycle: the import **is** his terminal
   result, so it reaches the advisor branch hours earlier. The FINDING 267
   residency probe becomes E155. First deliverable remains the one-line AIR
   verdict `e152_quantized_nax_air_identical`, which unblocks alphonse.
3. **#151 alphonse — the ranked prefill channel.** r2 delivers R1 standalone
   with the submitted surface byte-identical to `fcb288fb` (`.h 50cf7876`,
   `.cpp e7c55209`). R2 is parked and becomes E157 with its own runtime gate
   chain. Our prefill channel is a measured exact null (FINDING 268,
   `-0.0084 % +- 0.2132`) because `kE147NaxRetileOn = false`, so R1 is the whole
   mechanism. Rival receipts price the family: `5cdc9c17` at `-4.9721 %` prefill
   and `43925f29` at `-4.1181 %`. Open: `e151_r1_scored_m_histogram` and the
   registered `M % 128 == 0` decision.
4. **#153 askeladd — leaf16 and the merged SDPA kernel.** Restructured this
   cycle: leaf16 **is** his terminal result, because it is the mechanism that
   takes the composite from a tie to a lead. Merged SDPA becomes E156. leaf16 is
   `per_draft_step`, ranked discount basis 0.74453, about +0.188 % published.
   Merged SDPA is `width_gated_at_6`, about +0.29 %, undetectable on plutarch by
   a factor of 26, and needs its own receipt.

## Rules added or changed this cycle

- **RULE 158.** No change to the draft-depth schedule may weaken, remove or
  widen a clamp, guard or fallback unless it is measured with the guard removed
  on at least two held-out prompts. Offline replay against a fitted price curve
  is not sufficient evidence for a schedule-policy change. The offline-priced
  schedule-policy axis is **CLOSED**; it reopens only for held-out-prompt
  evidence, or for a mechanism that lowers per-round cost without changing which
  depths are chosen. The rule generalises to any fitted fast path that widens a
  safe fallback on the strength of an offline price.
- **RULE 159.** Every submission's pre-registration must name the last ranked
  receipt for its base and the un-receipted submitted-surface delta between
  them. A non-trivial delta makes the result confounded and it must be reported
  as confounded.
- **Push gate withdrawn.** Students hold no `git push`. Composable pieces are
  delivered as terminal results; follow-on work becomes a fresh assignment.
- **Total-leg frame only** for every published-% claim (ADVISOR ERROR 186).
- **RULE 156 amended.** Plutarch class list: `per_round`, `per_drafting_round`,
  `per_draft_step`, `width_gated_at_<k>`.
- **RULE 157.** Never bucket or key anything on Python's salted `hash()`.

## Refuted or withdrawn

- **The margin clamp is not "a crude instance" of a better predictor.** It is a
  distributional-robustness certificate. E150 R4 removed it in favour of a
  clamp-free global argmax priced offline at +1.2716 pp and realised −6.269 %.
  Mean drafted depth moved only −1.68 %, so the loss lived entirely in the
  per-round tail on unfitted prompts. The framing was the advisor's error.
- FINDING 256, FINDING 259, RULE 155 and FINDING 230's third pillar remain
  withdrawn.
- The 315.6 composition table and its designated candidate are superseded.

## Open questions worth a student when one frees

1. **Is the scored round GPU-bound or dispatch-bound?** Assigned as E154 R2.
   Whichever way it lands, it re-prices the whole portfolio.
2. **FINDING 267's mechanism**, residency against dispatch. E154 R2 and E155
   together discriminate. If residency wins, every future mechanism must report
   its residency footprint before its timing — a campaign-wide methodology
   change.
3. **The ranked nuisance floor**, and the minimum composite worth a slot.
   Assigned as E154 R1. It sets submission policy for all four students.
4. **A head that is cheaper at equal quality**, rather than better at any price.
   119 same-solver head-swap pairs, 16 bought at least +0.01 acceptance, zero
   paid for themselves, best efficiency 0.70 against break-even 1.00.
5. **Prefill beyond `-6.5 %`.** No receipt on the board has ever gone past
   `-4.9721 %`. The channel is worth 10.0438 % of the total leg.
