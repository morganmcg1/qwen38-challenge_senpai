# SENPAI Research State

- **2026-08-22 ~17:20Z**
- Most recent research direction from the human researcher team: none received this round. The
  campaign runs autonomously under `senpai/program.md`.

---

## Where the campaign stands

**We hold the crown.** `623e77af` promoted at **3.52085227003175** at 16:05:35Z and is the top of
the public board. Promoted source ref `60d5b34a`, submission commit `fd562cb7`. The mechanism is
thorfinn's per-width one-pass QMV table `{6:6,7:7}` on the D_S kernel body.

The margin is real but thin, and it is thinner on the published median than on merit. The nearest
rival, `e44c0ba5`, resolved 0.0107 % below us on the published median but **0.1671 % below us on the
F83-weighted candidate leg**. The published median is a lossy, high-variance projection of the only
quantity we control. Every decision this round was priced on the candidate vector.

Advisor base: `35d8cf586b8671dc3d01faf3cdbd724ec603801b`. Contract base for submission:
`770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.

## Current research focus

**1. The launched threadgroup count is the leading unexplained lever.** Three separate levers moved
hard at widths 6 and 7 — instruction issue −33.5 %, weight passes −50.0 %, resident simdgroups
−15.7 % — and the measured ranked round time moved by about zero on all three. The one quantity that
did not move is the number of threadgroups actually launched. The shipped build launches `m` columns
at every routed width while only `ceil(m/ipg)` of them ever load a weight, so at width 6 it launches
2,595,200 threadgroups per round that return immediately. Tightening the grid is bit-exact by
construction, costs one word, and is the only intervention that breaks the collinearity between
launched columns and total work. This is the campaign's central open question.

**2. Where the draft schedule stops is worth more than how deep it goes.** The shipped depth price is
flat at 0.18 per step, but the measured cost curve has a hard break at verify width 6. Replaying a
tiered price at that single boundary is worth **+2.34 % held out** across six seeds with four
controls holding. The remaining work is a falsification gate and a refit against the measured
post-arm curve rather than a predicted one.

**3. The draft readout streams 59 MB per step to pick 32 rows, and 98 % of its errors are ties.**
The exact chain's base miss rate is almost entirely an exact bfloat16 tie at the maximum, not a
ranking failure. That closes the accuracy levers on the exact chain and redirects the work to
byte reduction: an int8 query-fitted low-rank sketch removes 37.5 MB per draft step for a predicted
+0.678 % ranked, at a resident cost of 31.84 MB.

**4. Wired residency is a single first-come-first-served pool with no floor, and it is now known to
cut both ways.** Raising the slack admits post-sizing state and buys about 0.20 %. Spending the same
slack on warm-phase scratch after wiring cost a rival **5.3 %** — a per-drafting-round offset of
3,030 µs that lands squarely on the known cluster-2 state. Any mechanism that allocates resident
memory now has to report the admission ledger before it reports a time.

**5. Our own archives delete other people's promoted mechanisms, and we had not been checking.**
A Yukon archive replaces every required editable path. Our crown tree silently dropped four
mechanisms other accounts had promoted. Three rivals are in flight restoring them onto our tree
right now. Priced from our own instrument, two are measured nulls, one is +0.064 %, one is +0.03 %,
and one is unpriced and interesting. A frontier mechanism sweep is now a standing pre-submit rule.

## In flight

| PR | student | question | state |
|---|---|---|---|
| #130 | alphonse | wired slack 64 MiB -> ladder argmax | holds the free submission slot; 512-token ladder complete, awaiting readout and rebase |
| #134 | edward | the pass-boundary depth price, plus warm/scored expression parity | r2; item 0 is the two zero-allocation warm restorations |
| #135 | thorfinn | does tightening the QMV launch grid move ranked round time | new |
| #136 | askeladd | build the C1 sketch behind a dispatch microbenchmark gate | new |

## Potential next research directions

- **Threadgroup resource allocation as a first-class cost.** If the tight grid pays, the currency
  this kernel actually spends is neither instructions nor bytes nor occupancy but launch and teardown
  of register footprints. That would reframe every Route B result to date and open a family of
  grid-shaping experiments across all 257 scored dispatches.
- **fp32 accumulation in the exact affine-4 rerank.** The rerank emits bfloat16 over 32 rows per
  step. If a wider accumulator separates the tie population that FINDING 170 identified, acceptance
  moves at essentially zero byte cost. Unowned, and it needs its own experiment because of the
  accumulation-order wall.
- **The normed-verify warm as the explanation for the round-1 excess.** 20–31 ms of GPU-side
  round-1 excess with host CPU *falling* is the signature of a blocking off-thread pipeline build.
  The warm phase provably never compiles the expression the scored verify dispatches.
- **Joint sizing of resident consumers.** The sketch needs 31.84 MB, the warm refill needs far more,
  and the slack is 64 MiB today. The ladder should be read as a budget for the whole roadmap rather
  than as a single arm.
- **C2, quantizing the bf16 precision islands to affine-4 g64.** Reopened and unowned, worth
  +0.38 % to +0.45 % on the byte model.
- **Composition risk between the depth-price arm and the kernel arms.** `pb6` changes which widths
  run; the tight grid and the one-pass table change what each width costs. Neither has been measured
  in the presence of the other, and the depth price is a `static let` that reads no kernel state.
- **The plateau escape, if the tight grid returns a null.** Three levers and a launch-count lever
  would then all read zero at widths 6 and 7, which would mean the width-6 cost break is not a
  property of the matvec at all. The next tier is the verify batching contract itself: whether the
  target must be fed as one wide row block, and whether the split-5 SDPA chunk boundary can be moved.

## Standing constraints

- Price on the F83-weighted candidate vector, never on the published median. The published-median
  null spread is at least ±0.4 %.
- Every register or occupancy closure decided on a student's g16s Mac is void as evidence about the
  ranked g17s runner. The one-pass table is unconditional, so student Macs now run a differently
  clamped kernel than the runner at widths 6 and 7.
- Yukon allows exactly one in-flight submission. Validation runs 42–130 minutes.
