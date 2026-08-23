# SENPAI Research State

- 2026-08-23 09:25 UTC
- Most recent human research direction: none received this generation. The team
  has given no new steer since the campaign opened; the advisor is operating
  autonomously under `senpai/program.md`.

## Where the campaign stands

Crown `684821ed` at 3.71959723 on organizer source `eb5eadc7`, unmoved for nine
hours. Our best row is `572b2cc4` at 3.66218564. Advisor base `601c137c`.
Five rival rows are validating; none is ours, so our submission slot is free.

**The single most important fact of this generation: the campaign base has been
shipping a mechanism that costs about 2.4 % of the published median.** The
`pb6` depth-price arm is compiled as the default at
`Qwen36MTPBlockSession.swift:1139`. The only ranked contrast that isolates it,
`572b2cc4 -> e003a86d`, prices it at `-2.3800 %`. It starts drafting on
plutarch, which carries exactly zero median weight, and pays for that on beagle,
which carries about half. Two local instruments read it as a win with tight
intervals and the wrong sign. Retiring it is a larger move than anything the
campaign has landed since `623e77af`, and it costs one changed line.

Three further advisor errors were found by students in the same window: the
"unexplained +4.2 %" decode gap was our own base drift between two of our own
submissions; `shift_dst` was already present in the kernel I said lacked it; and
I compared two different pricing frames inside one sentence. All three are
recorded in ledger 310 with their corrections.

## Current research focus and themes

**1. Get the free 2.4 % back and re-establish the frontier.** Thorfinn holds the
submission slot and is flipping the compiled depth-price default to `.ship`.
Forecast 3.68 to 3.69 against a bar of 3.71959723. That is a checkpoint, not a
target.

**2. Per-round discrimination, `+6.2170` pp.** This is now the campaign's largest
open number by an order of magnitude. Edward's E145 split the acceptance axis in
two: perfect per-position marginal calibration is worth `-0.5144` pp and is
closed; perfect per-round realised capability is worth `+6.2170` pp and is open.
The shipped depth-0 and depth-1 margin clamps are already a crude instance of the
open half, worth `+0.5668` pp where they run, restricted to two of eight depths,
with two hand-set constants. They are a depth brake, not a draft veto. E150 asks
the only question that decides the axis: how much of `+6.2170` pp can a predictor
built from observables available before the drafts exist actually reach? The
conversion is unforgiving, `sig 0.20` buys `+2.80` pp and `sig 0.30` buys `+0.45`.

**3. The seed prefill, `8.45 %` of the candidate leg, still untouched by us.** A
rival's 128x32 NAX seed retile reached `-4.9721 %` prefill, worth about
`+0.382 %` of median. Alphonse holds the grid-stride form; rung E-1a is green
with zero surplus tiles and five rejected positive controls.

**4. Draft readout dispatch reduction.** Askeladd holds leaf 16 on the shipped
vocabulary and the finalize-plus-rerank fusion that a rival measured at
`-0.1983 %` on the median pair.

**5. Our measurement instrument is coarser than we claimed, and we now say so.**
Rule 144 fixes the reporting frame: the realised median pair leads every ship
decision, the weighted-five mean sits beside it, and no two numbers may be
compared across frames. The single-receipt `2 sigma` MDE in the lead frame is
`0.1547` pp, not the `0.1154` pp the campaign had been quoting. Rule 79 is a hard
gate: no local timing leg may publish a depth-price or schedule-policy contrast,
because two students produced tight, wrongly-signed results on that axis in one
day. The cause is width concentration, local mass 0.62 to 0.77 at width 8
against a ranked 0.5390, which is now Rule 143.

**6. A possible first-order fact awaiting a real estimate.** Over three draws of
one declared-identical tree, beagle's candidate leg is `4.7x` noisier than the
next paying prompt, at `0.1850` pp against `0.0292` to `0.0396`. beagle carries
about half the published median. If that survives an eleven-sample estimate it
means our uncertainty lives almost entirely in the lower median slot, and it
should reorder the queue toward mechanisms that pay on essays, republic,
medicine and botany.

## Potential next research directions

- **A sequential draft-stopping rule.** The scheduler picks depth once, before
  the head has produced anything, yet the head drafts sequentially and its
  confidence at position `i` exists before it proposes position `i+1`. A
  sequential rule strictly dominates a one-shot choice on information and costs
  no extra head work for the drafts produced. The only new cost is making the
  decision visible on the host, because verify width is a launched grid shape.
  Being priced inside E150.
- **Certified margin bounds, 2606.30265.** Still the principled replacement for
  the clamp's two magic constants, but E145 R7-4 narrowed its fit: a certificate
  that decides whether to draft is not a drop-in for a thing that decides how
  far. Needs adaptation before assignment.
- **First-error focal loss on the proposal head, 2606.11552.** Off the acceptance
  axis and therefore not blocked by the `-0.5144` pp closure. A Rule 125
  mechanism: price it through the scheduler's response.
- **The xsums fill fusion.** Ranked-measured at `-0.1463 %` candidate leg. The
  rival variant that fused it into the residual-plus-RMSNorm pass FAILED
  validation, so the class is unrefuted but the implementation is not free.
- **FP32-twin activations in the Route B QMV load**, `Qwen35.swift:1480-1494`,
  bit-exact by construction, `+0.3` to `+2.0 %`, unclaimed.
- **The `AttentionUtils.swift` KV re-read at qL >= 6**, about `+0.34 %`, fires on
  58.6 % of ranked rounds, editable and unowned.
- **MARLIN's four-stage pipeline and SplitK on the QMV family**, unpriced.
- **The 6 to 7 width cliff**, `25,861` us, a `2.32x` step in the measured curve
  that no source account yet explains.
- **Nibble entropy of our own checkpoint**, corrective, one hour, zero GPU.

## Standing constraint

Tree, multi-candidate and hedge-row drafting are structurally blocked:
`QwenRuntimeMTPDriver.requireStructurallySound` forces a single linear chain and
`declaredRows == rowsPerRound(draftTokens.count)`, so `M == d + 1`. Every
speculation idea must fit that shape.
