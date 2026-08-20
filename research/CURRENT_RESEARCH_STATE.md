# SENPAI Research State

- **2026-08-20 08:45 UTC**
- Campaign base: `4898738ef12a423212c00485aa865b8e52056974`
- Organizer frontier: `bfab0de58d43453e506523707e1720a3485570f4` (Yukon `9ad17378`, 3.25238228, Lieisyourlie)
- Our best board score: `ca9251b8` 3.23250848 (rejected, 0.61 % behind the crown)
- In flight: `ff73cbbd`, validating since 06:42Z

## Most recent human research direction

Issue #22: execute aggressively toward the winning frontier. No new human
direction this cycle.

## Current research focus

**We are 0.61 % behind the crown, which is inside the noise of a single ranked
pair.** Ledger 193 puts the standard deviation of one ranked run at 0.756 % and
of a difference at 1.069 %. The crown itself last moved by +0.0155 %, which is
0.02 sigma, on a mechanism measured at 0.0023 % of a leg. **The top of this
board is a noise ratchet.** Composition of several independently measured wins,
not micro-optimisation, is the way across.

Four mechanisms are in flight, one per student, each on its own Mac:

| PR | student | experiment | state |
|---|---|---|---|
| #71 | thorfinn | E68 depth schedule against the inverted cost curve | rung 3 timing |
| #72 | edward | E69 cross-row QMV x-operand traffic | rung 0 |
| #73 | alphonse | E70 local-to-ranked dispatch divergence audit | rung 0 delivered, rung 1 next |
| #74 | askeladd | E71 in-situ width-tax census | just assigned |

### The organising question

At the scored verify width the machine runs at **50.7 % of the local bandwidth
roof and 31.5 % of the local compute roof**, and at **38.1 % / 9.1 %** at rank,
while the depth-0 round runs at **98.1 %** of the local bandwidth roof. The
weights a round reads are width-independent, so `T(6) - T(1) = 69.7 ms` buys the
verify width with **zero extra weight bytes**.

Roughly half the machine, at the width the score is decided at, has never been
named. Every kernel experiment this campaign has run is QMV-only, chosen because
QMV was measurable and not because it was shown to dominate. **E71 exists to
produce that attribution or refute it.** The corrected roofline already predicts
which way the answer leans: byte-reduction levers have been systematically
over-priced and latency, issue-slot and cache-traffic levers under-priced.

### What closed this cycle

- **The depth-price lever is closed from four directions.** The level is
  bracketed by measurement, the shape is under test in E68 rung 3, the order is
  irrelevant because QMV groups are concurrent threadgroups, and the decision
  rule is provably equivalent to argmax at every ranked operating point.
- **The QMV dispatch table is closed** at every width 3 through 9, now with
  measured rather than extrapolated single-group costs, under a symmetric cost
  form `sum_g S[g] - 15.191 ms`.
- **`sdpa.cpp:177` is unreachable on every host** because this checkpoint is
  head_dim 256. No local SDPA measurement describes a kernel the ranked M5 does
  not run. A standing transfer risk is retired.
- **Ranked prefill is closed twice over**: `qmm_t_nax` on the quantized
  projections and `steel_gemm_fused_nax_*` on the dense attention fallback.
- **The head-prime follow-up is priced down**, because M = 511 is the one
  decode-reachable architecture divergence and it runs `qmm_t_nax` at rank.

### What was retracted this cycle

- The public local fixture is at **p = 0.8808**, between ranked medicine and
  ranked republic, not at 0.9625. **Local scheduler A/B tests transfer.**
- **`R = 2.1383` is correct**; my own 1.759 crossed two builds and is retracted,
  along with 205(G)'s "the ranked width curve is 1.16x flatter". The corrected
  reading is that the width curve transfers approximately 1:1.
- thorfinn's conservation theorem — that a cost-conserving reshape can only
  shorten the selected depth — is **false**, with a counterexample at beagle's
  acceptance rate.

## Potential next research directions

Ordered by expected value, not by ease.

1. **Whatever E71's census points at.** If a family other than QMV owns a large
   share of the width tax, that family becomes the campaign's next programme and
   it will be the first one chosen from evidence rather than from availability.
2. **Composition of the live mechanisms.** t55 and t6 are merged and each
   measured. E69's arm C, if it wins, is a third independent QMV lever. E56 and
   E55 share 33.1 % of their effect, so kernel and schedule arms are
   substitutes; kernel arms against each other may not be. Compose only winners
   that were measured independently, and re-measure the composition.
3. **The unattributed half of the round.** If E71's closure gap is large, the
   width tax is dependency latency rather than any family's work, and the next
   programme is scheduling and evaluation boundaries rather than kernels.
4. **plutarch as a runtime-only ranked instrument.** Its `raw_p` is 92.2 %
   depth-0 rounds, so it reads the candidate build's non-speculative runtime
   almost purely. We are 0.26 % behind the crown on that channel and 0.73 %
   behind on beagle. Splitting our deficit into a runtime part and a
   speculation part is now possible from public receipts alone.
5. **Deliberate ranked replication.** With a 0.756 % single-run standard
   deviation and a 0.61 % deficit, one ranked run cannot distinguish our
   candidate from the crown. A resampling ticket would buy a real comparison.
6. **A cleanup PR after the next merge.** Dead GDN paths, the masked scan
   variant, the never-executed generic repair fallback, the dead
   `cacheLimitBytes` and `clearAllocatorCacheAfterWarmup` constants, and E65's
   26 instrumentation lines in the submitted session file. Deletion is the
   default; the winning behaviour should be the only path.

## Standing constraints that shape every plan

- `warmMTPDecode()` is untimed, so warming is free. Diff the candidate surface
  against `upstream/main` before every submission: a frontier move can turn a
  no-op into a deletion.
- Every scored verify width M = 1..9 reaches the cross-row QMV kernel on every
  plausible ranked tier. `M = 1 + drafts <= 9` by contract, so the M = 10
  kernel-family cliff is not a live constraint on the schedule.
- Prefill is scored at 8.4-9.1 % of a leg and is unreachable. Multiply every
  round-cost projection by 0.9125.
- Label every measurement `harness=local`, `harness=ranked`, or
  `harness=arch-probe`, and label every transfer ratio `candidate:candidate` or
  `baseline:baseline`. The ranked receipt carries two builds.
