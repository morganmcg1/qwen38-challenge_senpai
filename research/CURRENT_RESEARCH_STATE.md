# SENPAI Research State

- 2026-08-22 10:40 UTC
- Most recent research direction from the human researcher team: none since the last brief. The
  standing direction is unchanged: take and hold the Qwen 3.8 27B native-MTP crown, submit the
  strongest legitimate candidate autonomously, and never hold a candidate back because of the
  operator plausibility ceiling.

## Where the campaign stands

WE HOLD THE CROWN. Submission `d3c491b5-902f-4f80-8d33-b7938f980d2d` promoted at
**3.49065043561149**, source `6f1cd66fc214d4281b37da335e21077c7e9d7ad5`, improving the previous
crown `bc070b7b` (francip, 3.35922017) by 3.913 %. The candidate is thorfinn's Route B: a
candidate-owned wide matvec that removes the sums add tree from the target verify path. The
advisor branch already contains the promoted scored surface.

We hold six of the eight per-prompt floors. The best composition envelope any rival can build
from all published per-prompt bests publishes at 3.492374, which is +0.049 % over us. Nobody can
reach us by composition. Two rivals are already redrawing on our promoted source `6f1cd66f`, so
the lead erodes only if they compose a new mechanism onto Route B.

Our submission slot is FREE. Nine submissions resolved, none in flight.

## Current research focus

The campaign pivoted this cycle. For most of its life the frontier moved through register and
occupancy work on the wide matvec. That axis is now measured out, and a much larger one opened.

**1. THE PASS-COUNT LEVER IS THE LARGEST OPEN ARM, BY AN ORDER OF MAGNITUDE.**

The shipped dispatch table makes one pass over the weight matrix at M=3,4,5 and TWO passes at
M=6,7,8 and THREE at M=9. Two students found independently, from a direct microbenchmark and
from a fit to our own eight ranked receipts, that the extra pass and not the width sets the
price. The excess over a one-pass line is +21.4 % at M=6, +25.4 % at M=7 and +28.9 % at M=8, and
M=8 is 76.9 % of routed rounds. A one-pass table is worth 7.6 % to 15.4 % of the leg and would
publish between 3.76 and 4.09.

A cross-solver contrast confirms the mechanism. The board's fitted cost curve breaks at M=5 and
ours breaks at M=6, and that is exactly the width where our table differs from the organizer
default.

**2. THE BLOCKER IS SPILL IN THE SHARED SWITCH, AND TEMPLATING REMOVES IT.**

The prior stop-list entry rested on E104, which priced collapsing M=6,7,8 at -16.5 %, -31.2 % and
-50.9 % ranked with a real receipt behind it. That entry is now explained. The Route B entry
point emits ONE Metal function that inlines every width body behind a switch, so the register
allocator must satisfy the widest inlined body. One-pass-everywhere in a shared switch measures
126 registers with 160 BYTES OF SPILL on the ranked architecture. E104 measured the spill step,
not the pass count.

Per-width templating is therefore a hard prerequisite, not an optimisation. It is also a
standalone bit-exact candidate worth +10.18 % residency across 88 % of the round, using only
three pipelines, and it is the only work item that is both immediately shippable and required by
the bigger arm.

**3. THE RESIDENCY COEFFICIENT IS MEASURED NULL, WHICH CHANGES HOW WE PRICE EVERYTHING.**

Two students, two independent designs, both return a residency coefficient indistinguishable from
zero and far below the kill gate. The modelled 0.445 that drove several recent priorities is dead
as a point estimate. This is advisor error 105. Consequences: register-only arms are cheap
insurance rather than levers; the entry-point cliff gate is a register instrument and not a time
instrument; and an arm that loses residency while deleting real work should usually proceed.

The local host cannot settle the question, because it is memory-saturated at M=1,2,3 and
ALU-bound at M=6,9, and a null is what both regimes predict. Three receipts at very different
leverage (12.82 % on a small share, 10.18 % on 88 % of the round, 76.92 % on the head path) will
identify the ranked coefficient far better than any local ladder.

**4. THE ROOFLINE FRAMING IS REFUTED AND REPLACED.**

The 542.8 GB/s "M5 ceiling" is a pre-E100 achieved rate computed with double-counted bytes, and
the 2.8x headroom built on it is algebraically two times a ratio of round times with the
bandwidth cancelled out. Advisor error 107. The correct instrument is within-segment inflation:
inside a fixed pass count the bytes are constant, so ranked M=1 to M=4 inflating by 38.2 % and
M=6 to M=8 by 23.9 % are pure non-byte work. Build on 19 to 28 % envelopes, never on 2.8x.

**5. `quantized.h` IS EXHAUSTED FOR IN-PLACE BIT-EXACT GAINS.**

Exactly one class of the priced instruction census survives as a candidate-owned removal, and
Route B already deletes it. Everything else is shipped, forbidden by the accumulation-order wall,
or the actual work. Future kernel value must come from the Route B custom-kernel surface, not
from editing the vendored kernel in place.

## Live assignments

| PR | student | topic | state |
|---|---|---|---|
| #128 | thorfinn | Route B per-width templating, then the one-pass table | WIP. Rung 2-lite refuted and reverted. Templating first, table second. |
| #129 | edward | ranked cost curve and depth scheduling | WIP. Estimator axis closed. Next: price the templating and one-pass arms on our own curve, and resolve the tier-2 slope anomaly. |
| #130 | alphonse | `prune_na5_pair` entry-point register diet | Revision r2. Awaiting an official submission and receipt. |
| #131 | askeladd | entry-point census and the cliff gate | Terminal, succeeded, merged. Next: Route C. |

## Potential next research directions

**Immediate, already assigned or about to be.**

1. Get `qwen_e120_qmv_wide<8>` under the ranked register boundary. It currently spills 48 bytes
   and carries 18.6 of the 20.3 points of the one-pass prize. This is the single highest-value
   engineering task open. Report the spill site from the ISA, not the byte count, and census the
   no-table arm.
2. Ship per-width templating with a proven warmup gate, measure it alone, then build the
   `{6:6,7:7}` spill-free table on top of it, then attempt `{6:6,7:7,8:8}`.
3. Route C: a narrow custom entry point for the head's `ntg.x == 1` projections, which today pay
   a 101-register allocation to run a 57-register body. +76.92 % residency on the head path, a
   real submitted-surface candidate, and a third leverage point for the residency coefficient.
4. Resolve the tier-2 slope anomaly. The fitted marginal cost per row is 1.8x higher in the
   two-pass regime on both our curve and the board curve, which the pass-count model does not
   predict. Either the model is incomplete or the piecewise fit is ill-conditioned in the slopes.
   This must be settled before the one-pass arms are priced from those slopes.

**Near term, unowned.**

5. The scheduler-discrimination axis. The oracle depth schedule is worth +8.52 % on our own curve
   with 8 of 8 prompts positive, while every implementable estimator arm is negative. The pooled
   AUC of the top-2 margin is 0.5109, so the discrimination signal is NOT in the margin.
   Enumerate where else it could be: hidden-state norms, head-chain agreement, position in the
   round, per-layer residual magnitudes, entropy of the shortlist scores.
6. Concentrated bf16 precision islands on the head tensors that carry the requantization damage.
   The declared head loses 0.82 acceptance points against the bf16 master, worth about +1.8 %
   ranked, which is 39x the whole draft-readout pool. Shipping the full master loses 0.68 %
   because it adds 1.354x the head bytes; a concentrated set wins if the recovery-per-byte curve
   beats the uniform line by 1.38x. The decisive screen is offline and needs no GPU: decompose
   the 0.82 points by tensor.
7. C1, the sign-sketch low-rank first pass on the draft readout, +0.23 % to +0.34 % ranked,
   designed and unowned.
8. C5, pad the centroid table from 12,292 to 12,296 rows. Askeladd confirmed the mechanism:
   `12292 % 8 == 4` fails the fast-path gate, and 12,296 clears it. Three lines, bit-exact on
   proposals, +0.03 %. A rider on any draft-path PR, never a standalone assignment.
9. P5, stale-suffix recycling after a reject. Legal, de-prioritised, and its reopener is a
   zero-GPU trace measurement showing stale-suffix acceptance above 0.5 at position 1.

**Structural, after the next promotion.**

10. Cleanup PR: delete the dead `qkv(_:)` island fast path, the dead E121 code, the research-only
    `Qwen35IslandArm` selector, and the now-dead `bits == 2` affine-2 readout branch, which
    censuses as never dispatched.
11. Re-rank all 37 E87 decision cells with the corrected acceptance coefficient of 203.

## Standing constraints that shape every direction

- Bit-exactness against the hidden serial token stream is absolute. One character of
  reassociation once moved declared top-two row evidence at 52 of 64 positions while the local
  parity line stayed green. A local parity line cannot clear an accumulation-order change.
- The local host is `applegpu_g16s` and the ranked host is `applegpu_g17s`. A register or
  occupancy closure decided on the local host is void as evidence about the ranked host, and the
  register delta has already been observed to invert sign across the two.
- Only five of the eight ranked prompts carry any marginal weight, and beagle alone carries
  0.4862. Weight every price per prompt, never by the local histogram.
- The operator plausibility ceiling of 5.0 is not a target and never a reason to hold a
  candidate.
