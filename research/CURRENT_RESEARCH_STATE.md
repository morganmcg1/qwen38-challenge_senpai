# SENPAI Research State

- 2026-08-22 11:20Z

## Most recent research direction from the human researcher team

None received this round. The campaign is operating autonomously under
`senpai/program.md`.

## Where we stand

We took the Qwen 3.8 27B MTP crown at 09:08Z with `d3c491b5` at **3.49065044**
and lost it twenty minutes later to `cf79f7df` (`Lieisyourlie`) at
**3.51661724**. We trail by 0.744 %.

Promotion publishes the promoted source. Within fifty minutes of ours going
public, three separate solvers cleared 3.47 by compounding on our Route B, and
five candidates are validating now. The board had been flat near 3.35 for two
days before that. Assume roughly 0.1 % per hour of frontier movement while our
source is the parent of record, and assume any mechanism we promote is
reproduced within the hour.

The rival frontier `a0f8588668c603864deffe407f5895b31858414e` is imported into
the advisor branch with proved byte parity. Exactly one editable path differed.
Our submission slot is free.

## Current research focus

**The pass count on the target-verify path is the campaign's largest open arm,
and it is the one thing on this list that rivals cannot cheaply rediscover.**

Everything the field is currently copying lives on the draft-proposal path. Our
open lever lives on the target-verify path, is disjoint from theirs under
Rule 75, and requires offline g17s register census work to even see. That
asymmetry is our only durable advantage, so the round is organised around it.

The shipped dispatch table runs two verify passes at M=6, 7 and 8 and three at
M=9. Fitted from our own ranked receipts, the excess over the one-pass line is
+21.4 %, +25.4 % and +28.9 % at M=6, 7 and 8. M=8 alone is 76.9 % of routed
rounds. Collapsing all three is worth +7.6 % to +15.4 % of the leg, which
composes onto 3.51662 as a published score between 3.78 and 4.06.

Three things gate it, and one student owns each:

1. **Per-width templating** (thorfinn, PR #128). The shared `switch (qmv_m)`
   inlines all seven bodies into one Metal function, so the register allocator
   must satisfy the widest. Changing the table inside it reproduces E104, which
   scored 3.23588901. Templating is a prerequisite, not a deliverable: with the
   residency coefficient near zero it is worth about +0.09 % on its own.
2. **The NA=8 register spill** (askeladd, PR #132). `wide<8>` is 126 registers
   with 48 bytes of spill on g17s. NA=6 and NA=7 are already spill-free. Until
   the spill goes, the arm is worth +1.8 % instead of +7.6 % to +15.4 %.
3. **The tier-2 slope anomaly** (edward, PR #129). The fitted in-segment slope
   above M=5 is 1.82 times the slope below it, which the cost model forbids.
   Every number in the bracket above depends on that extrapolation.

Alphonse (PR #130) is submitting `frontier + prune_na5_pair` now. The arm is
small, worth +0.007 % to +0.86 %, but the submission is correct for three
reasons that are not about its score: the slot is idle while five rivals
validate; it is the only way to measure the residency coefficient `c` on the
ranked architecture, which gates roughly +4 % of downstream work; and it proves
the imported frontier builds and passes exactness in our tree before we stack
the pass-count arm on it.

## A measurement instrument broke this round

The F76 mode classifier no longer separates the measurement mode from a
draft-path mechanism. Both frontier-carrying rows read 6 to 8 same-mode sigmas
below the entire historical fast cluster, driven almost entirely by travel.
The cause is structural: the mode costs about 0.82 ms per drafting round and a
draft-path mechanism saves time per drafting round, so an instrument fitted to
detect one is maximally sensitive to the other.

This voids the fast-cluster reference band for every candidate we will submit
from now on, and it makes F80's post-hoc mode correction unreliable. Edward
owns the repair. A clean negative is acceptable: if mode and mechanism are not
separable from board data, we need that stated plus the measurement that would
separate them.

The general lesson is recorded as Rule 95. A nuisance-parameter estimator must
be checked for identifiability against the class of treatment effects you
intend to measure, not only against the noise present while fitting it.

## Potential next research directions

**Near term, already owned.** Land the one-pass table. Add `(8,8)` the moment
askeladd clears the spill. Census NA=9 while he is there; if the same mechanism
clears it we can retire the last multi-pass width entirely.

**Route C, unowned and ready.** Every steady-state head projection launches at
`ntg.x == 1`, misses both switches in `affine_qmv_fast`, and runs a 57-register
body inside a function that allocated 101. That is +76.92 % g17s residency
across 31.44 dispatches per round. A narrow kernel holding only `qmv_fast_impl`,
dispatched from Swift, captures it. One flagged inconsistency must be settled
first: the 232.96 MB/step head byte count disagrees with the measured ranked
head time by about a factor of four, probably because the byte law used total
artifact bytes rather than the 59.0 MB the readout actually streams.

**Head weight quality, the largest unexploited pool.** Declared head accuracy
is 92.31 % against 93.13 % for `master-bf16`, a 0.82 point gap worth about
+1.8 % ranked. That is 39 times the entire draft-readout approximation pool,
which the acceptance-exchange law caps at +0.046 %. Shipping full bf16 loses on
bytes at -0.68 %, so the question is whether concentrated precision islands on
the damaged tensors beat the uniform line by the required 1.38 times. The
decisive first step is offline and needs no GPU: decompose the 0.82 points by
tensor on the E124 median-regime corpus. Existing islands cover q, k and v only,
and K and V are already fully bf16, so the gap lives in uncovered tensors.

**C1 sign-sketch readout**, designed and unowned, +0.23 % to +0.34 % ranked.

**The within-segment inflation instrument.** With the roofline framing refuted,
the correct measure of remaining headroom is time growth at constant bytes
inside a fixed pass count: +38.2 % from M=1 to M=4, and +23.9 % from M=6 to
M=8. Build future pricing on 19 % to 28 %, never on the old 2.8 times figure.

**Strategic.** Prefer mechanisms that are hard to rediscover. The board has
demonstrated it will reproduce a single-file mechanism within thirty minutes of
publication, so a mechanism's value to us is its gain multiplied by the time we
hold it exclusively.
