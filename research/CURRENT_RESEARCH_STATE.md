# SENPAI Research State

- 2026-08-22 ~19:35 UTC. Advisor base `33ce6a3f` (Merge PR #130, alphonse E130).
  Campaign `BASE_SHA` for every submit call is unchanged at `770a3ff2`.
- Most recent research direction from the human researcher team: none received this round.
  The standing direction is the one in `senpai/program.md`: move the official frontier,
  submit autonomously, and treat a promoted rival row as an interrupt.

## Where the board stands

Our promoted row `623e77af` at 3.52085227 is now fourth. The crown is `02742bf0`
(scarletbright) at 3.52686512, promoted 19:06:44Z. Three rows passed us in five hours.
On the common-denominator scale (`research/common_denominator.py --anchor 623e77af`)
our candidate leg is still the fastest tree measured on the board; the published
ordering above us is mostly serial-leg lottery plus one real constant.

The one in-flight submission slot is FREE. It is allocated to thorfinn's tight
launch grid, with edward's `pb6` depth price second in the queue.

## Current research focus

1. **The launched threadgroup count.** F182. `grid = .tight` deletes the empty
   columns that the wide grid launches for every routed wide-QMV dispatch.
   Measured +1.806 % faster absolute candidate seconds per token at 27 sigma over
   twelve 512-token legs, with a serial null predicted from source before the
   measurement and confirmed to 0.004 pp. Honest ranked bracket +0.5 % to +1.8 %.
   Ship commit is built and gated; one transient `--local-submit` failure is being
   retried.

2. **The pass-boundary depth price.** Edward's `pb6` raises `marginal[4]`, the entry
   into verify width 6, by a 1.45 tier factor. A refit on his own measured round-cost
   curve holds the argmax at boundary 4 in 24 of 24 leave-one-prompt-out refits and
   in 5,997 of 6,000 bootstrap draws, and reads +2.4683 % held out. The open gate is
   FM1: the replayer draws realised capability independently of round-start state, so
   it cannot price the rounds `pb6` actually stops early. The archived traces can, and
   that audit costs zero GPU.

3. **The C1 sketch readout.** Askeladd's rank-256 checkpoint-derived row PCA replaces
   the affine-2 probe scan. Rule 107 net +0.546 %, 1.8x the bar. The rung-1 risk gate
   is the best in the campaign: eleven boundaries, each with a positive control that
   was shown to fail. Rung 2 ABBA timing is running.

4. **The M=5 to M=6 cost cliff.** F185. The one mechanism that was supposed to explain
   it - a second QMV pass - is dead by construction on the current base, because
   `onePass67` makes both `plan(5)` and `plan(6)` single-pass. The step is still
   11.3 to 13.1 ms above the shallow slope, 25 to 29 % of an M=5 round, and the slope
   after the boundary is 5,324 us per row instead of 3,446. This is the largest
   unexplained quantity in the campaign and it prices the whole depth-schedule family.
   Alphonse owns the attribution (E137).

5. **Held riders, composed in order after the queue clears.** The fp32 tiebreak at
   `Qwen35.swift:4117` at +0.2166 %, then the cluster probe fraction argmax at
   +0.0992 %. Both are one-token or one-constant edits with zero bytes and zero
   dispatches.

## Potential next research directions

- **Attribute the cliff to a dispatch family.** F184 found that the exactness chunk in
  `AttentionUtils.swift` engages at exactly `qL >= 6` and adds two SDPA dispatches, one
  concat, and two hidden contiguous query copies per full-attention layer - 64 extra
  dispatches per drafting round, 32 of them uncounted. That matches the recorded
  +109 dispatch step at M=5 to M=6. It has never been priced as time. If the whole
  family moves less than 1 ms it is eliminated and the cliff is elsewhere, most likely
  in Gated DeltaNet at S=6.

- **Gated DeltaNet at the same boundary.** P4, the S=2 mid-state write, gates about
  151 MB per round on rejection and is unowned. If the cliff is not SDPA it is the
  strongest remaining structural candidate.

- **Head-history fold warm gap.** The scored flush concatenates widths 1 through 9 and
  only width 2 is warmed. Predicted signature is a latency spike at the first round that
  reaches each new width. Must clear Rule 110 first: name the pipeline cache key field
  the change moves.

- **C2 precision-island quantization.** Reopened and unowned, +0.38 % to +0.45 %.

- **Host bookkeeping overlap (P3).** Bracketed at 0.12 % to 0.32 % with a prior
  retraction against it. Not assigned; the ceiling for all in-round overlap is the
  measured GPU idle of 1,493 us at M=6.

- **Cleanup.** After the queue settles, prune the stale E120 arm flags, the dead Route B
  table paths, and the E128 price arms so the winning behaviour is the only path.

## Standing methodological state

The campaign now prices every ranked claim through the candidate 8-prompt mean, whose
pair-difference null is 0.067 % (Rule 112), never through the published median. Every
offline replay that stands in for a Metal kernel must reproduce that kernel's own
arithmetic and prove it with a bit comparison (Rule 113). Every `--require` witness must
have a demonstrated failing polarity (Rule 101); edward's own self-caught violation this
round is the worked example. The pre-submit occupancy gate fails open (HARNESS DEFECT 35)
and is excluded from the submission chain until alphonse repairs it.
