# SENPAI Research State

- **2026-08-20 09:40 UTC**
- Campaign base: `d6237abbd94fe36e38d7fa953d15869e4184b196`
- Organizer frontier: `bfab0de58d43453e506523707e1720a3485570f4` (Yukon `9ad17378`, 3.25238228, Lieisyourlie)
- Our best board score: `ca9251b8` 3.23250848 (rejected, 0.61 % behind the crown)
- Last resolved: `ff73cbbd` **3.17229699 rejected** — a 1.86 % regression on our own previous best
- In flight: `9b241879`, arm 2, the crown-kernel revert, submitted 09:33Z

## Most recent human research direction

Issue #22: execute aggressively toward the winning frontier. No new human
direction this cycle.

## Current research focus

### The two facts that dominate every decision right now

**1. The ranked slot is the scarce resource, not the GPU.** Yukon allows exactly
one in-flight submission per account. A ranked run takes about two and a half
hours. The campaign can obtain at most about nine ranked numbers per day, in
strict series. Four students on four Macs can run local experiments in parallel
all day and produce zero ranked numbers.

**2. We hold no ranked-verified advantage over the frontier.** Every experiment
we have merged is a local win on `applegpu_g16s`. The one ranked comparison that
isolates our own kernel work says that work is **negative**: `ff73cbbd` scored
3.17230 against our own 3.23251 and the crown's 3.25238, with a bit-identical
schedule on all eight prompts, so the loss was pure time inside drafting rounds.

Together these say the campaign must stop producing unpriceable local wins and
start spending its serialized ranked slots as a measuring instrument.

### The operating rule this produces

**Compose from the crown, not from our base.** Each ranked arm should be the
crown plus exactly one of our mechanisms, so the receipt prices that mechanism.
Our base is a staging area for candidate mechanisms; it is not a frontier.

**Prefer mechanisms whose transfer sign is known.** Kernel group-partition
choices trade register pressure against reuse, and occupancy boundaries differ
between generation 16 and generation 17. Schedule and policy changes do not
carry that exposure: they change which shapes run, not how a shape is compiled.

### Hypothesis H208, now measuring

The local QMV group-partition optimum is host specific, and the local M4 Pro
optimum is a ranked pessimum. The transfer inverts in **sign**, not merely in
magnitude.

Our delta over the crown contained three independent partition widenings —
width 5 (`t55`), width 6 (`t6`) and width 9 (`E55`) — covering 63.25 % of ranked
verify-width time. Arm 2 reverts all three to the crown's exact bytes. Its
pre-registered point prediction is **3.2524**; a return to about 3.17 exonerates
the partitions and moves suspicion to the gated trace plumbing.

### Four mechanisms in flight, one per student, each on its own Mac

| PR | student | experiment | state |
|---|---|---|---|
| #71 | thorfinn | E68 depth schedule against the measured cost curve | rung 3 timing |
| #72 | edward | E69 cross-row QMV x-operand traffic | rung 0, under redirection |
| #73 | alphonse | E70 local-to-ranked dispatch divergence audit | rung 1 running |
| #74 | askeladd | E71 in-situ width-tax census | rung 0 |

### The organising question

At the scored verify width the machine runs at **50.7 % of the local bandwidth
roof and 31.5 % of the local compute roof**, and at **44.0 % / 10.5 %** at rank,
while the depth-0 round runs at **98.1 %** of the local bandwidth roof and the
ranked candidate depth-0 round runs at **77.2 %** of the 614 GB/s ranked roof.
The weights a round reads are width-independent, so `T(6) - T(1) = 69.7 ms` buys
the verify width with **zero extra weight bytes**.

Roughly half the machine, at the width the score is decided at, has never been
named. Every kernel experiment this campaign has run is QMV-only, chosen because
QMV was measurable and not because it was shown to dominate. **E71 exists to
produce that attribution or refute it.**

## What changed this cycle

### Exact ranked round accounting, with no model

Every prompt's round count is fixed by two integer constraints in the public
receipt: the mean draft length times the round count must be a whole number of
proposals, and rounds plus accepted drafts must equal 512. All eight close
exactly. Beagle runs **107** rounds at **53.338 ms**, not the 92.5 rounds at
61.672 ms we had published.

### Two of my own numbers retracted

- **`M` is one plus drafts PROPOSED. Tokens per round is one plus drafts
  ACCEPTED.** Dividing 512 by the verify width is wrong. Every ranked round cost
  I published before this cycle was too high.
- **`R` is width indexed, not a scalar.** `R(1) = 2.1383`, `R(5.53) = 2.3586`,
  `R(6.78) = 2.4742`. Scalar-`R` projections at deep widths were 10.3 % to
  15.7 % too high.

### 205(G) reinstated, and its retraction retracted

The ranked width curve is **1.10 to 1.16 times flatter** than the local one, on
all five deep prompts, confirmed by a second independent instrument whose own
joint feasibility test reports the single-factor transfer refuted with an empty
intersection. The sign of this is directly actionable: **a depth policy tuned on
our host under-drafts at rank, because our host over-charges for width.**

### The `R` dispute with alphonse, adjudicated

Both sides were wrong. His refutation compared a candidate-build quantity with
the pinned-serial population — the two-build error. My `188(A)` text did say
"leg" where the arithmetic used rounds. The dispute was settled not by either
model but by a bound: plutarch runs 449 of 487 rounds at depth 0 with a mean
round of 30.781 ms, and a drafting round cannot cost less than a depth-0 round,
so `c1 <= 30.781 ms` with no transfer model at all. **A bound beats a fit.**

## Potential next research directions

Ordered by expected value under a queue of depth one.

1. **The depth schedule as the next ranked arm.** It is the only mechanism whose
   transfer sign we know: the ranked width curve is flatter, so rank tolerates
   deeper drafting than our host does. E68 rung 3 is timing now, and the shipped
   policy has a known expressiveness defect — `widthCap` can be 5 or 8 and
   cannot be 6 — that caps depth exactly where 33.4 % of ranked width time sits.
2. **Transfer calibration.** Convert local measurements into ranked-priceable
   ones by measuring the same diff both ways. Arm 2 is a ranked measurement of a
   diff whose local effect we can measure exactly; that pair is the first
   calibration point of a real transfer law.
3. **Whatever E71's census points at.** If a family other than QMV owns a large
   share of the width tax, that family becomes the next programme, and it will
   be the first chosen from evidence rather than availability.
4. **The unattributed half of the round.** If E71's closure gap is large, the
   width tax is dependency latency rather than any family's work, and the next
   programme is scheduling and evaluation boundaries rather than kernels.
5. **plutarch as a runtime-only ranked instrument.** Its `raw_p` is 92.2 %
   depth-0 rounds, so it reads the candidate build's non-speculative runtime
   almost purely. We are 0.26 % behind the crown on that channel. Every ranked
   receipt splits our deficit into a runtime part and a speculation part for
   free.
6. **A cleanup PR after the next merge.** Dead GDN paths, the masked scan
   variant, the never-executed generic repair fallback, the dead
   `cacheLimitBytes` and `clearAllocatorCacheAfterWarmup` constants, and E65's
   instrumentation lines in the submitted session file. Deletion is the default.

## Standing constraints that shape every plan

- One in-flight submission per account. Cancel with
  `POST /api/submissions/<uuid>/cancel`; cancelling before the timed step costs
  no runner time.
- The runner checks out the **live promoted frontier** and overwrites its
  editable paths with our archive. Every non-editable byte at rank is the
  crown's. Diff the packaged surface against the live frontier before every
  submission.
- `warmMTPDecode()` is untimed, so warming is free.
- Every scored verify width M = 1..9 reaches the cross-row QMV kernel on every
  plausible ranked tier, because `vector_limit = 10` whenever the input
  dimension exceeds 4096. `M <= 9` by contract.
- Prefill is scored at 8.4-9.1 % of a leg and is unreachable. Multiply every
  round-cost projection by 0.9125.
- No local QMV cell measurement may be quoted as a ranked expectation without a
  ranked receipt or an explicit unverified-transfer label.
- Label every measurement `harness=local`, `harness=ranked`, or
  `harness=arch-probe`, and label every transfer ratio `candidate:candidate` or
  `baseline:baseline`. The ranked receipt carries two builds.
- One hypothesis per **submission**, not merely per pull request.
