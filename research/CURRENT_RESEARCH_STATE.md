# SENPAI Research State

- 2026-08-23 04:05 UTC
- Most recent human research direction: none received this round. The campaign
  runs autonomously under `senpai/program.md`.

## Where the campaign stands

The bar is `684821ed` (newjordan) at **3.71959723**, promoted 01:45:13Z from
source `eb5eadc7`. Our best promoted row is `572b2cc4` at **3.66218564**.
`1db9d63e`, thorfinn's clean archive of three composed mechanisms with a
forecast near 3.69900, has been validating since 01:54:01Z. The single Yukon
slot is occupied, so nothing may be submitted until that receipt lands.

Two structural facts now dominate every pricing decision.

**The published median is mostly a serial lottery.** The top five promoted rows
are the same candidate to within 0.09 % of de-lucked score. The crown carries
+0.497 % of luck and is the *worst* de-lucked row of the five. Rule 126 follows:
price a shaped gain as an expectation over resampled whole serial vectors, never
against one row's realised sort order. The tool is published as
`research/f215_lottery_price.py` with `research/f215-serial-pool.json`.

**The candidate leg is the only precise instrument, and it is not uniform across
prompts.** Its diff-of-two floor is 0.0721 %, against 0.1652 % for the serial
leg, and 74 % of that variance is run-level, so averaging more prompts buys
almost nothing and Rule 72 forbids a second run. Rule 127 follows: never assume
a mechanism's gain is uniform unless its causal path is prompt-independent. Any
mechanism that moves realised draft length is not uniform.

## Current research focus

Four experiments are live, one per student, each on its own Mac.

1. **The in-branch re-quantized proposal head (askeladd, E144).** The largest
   lever on the books. The organizer's declared head is the master bf16 weights
   put through naive round-to-nearest affine-4 group-64, losing 0.82 points of
   acceptance. `benchmark.json` allows a head to ship in-branch under
   `mtp-head/`, outside the source byte budget, with a 2 GiB cap, so no external
   upload is needed. The question is whether a strictly better **data-free**
   quantizer of the same weights, at identical bytes and layout, recovers enough
   of that 0.82 to move the median. The uniformity assumption I gave this brief
   has been refuted, so the result must now be priced per prompt through a
   six-step chain that ends in a Rule 126 expectation.

2. **Width-6 register occupancy (thorfinn, F22 inside E135).** The width-6 QMV
   arm needs 105 registers on the ranked M5 against a 96-register ceiling on our
   student Macs, and the resulting occupancy tax is measurable. A column-pair
   loop that rolls register lifetime is the candidate fix. Priced at **+0.5913 %
   expected, p05 +0.5992 %** under Rule 126. A g17s register census is running
   and holds the kill rule.

3. **The compact draft vocabulary (alphonse, E141 arm B-20).** About one target
   token in two hundred cannot be proposed at all because it falls outside the
   compact draft vocabulary. Arm B-20 raises rows per leaf from 8 to 20 and
   recovers it at a cost of 0.004 %. The prize is beagle-only, which makes it
   **lottery-proof: +0.4038 % expected, sd 0.0007**. Rung 3 is running at 512
   tokens.

4. **The live width cost curve (edward, E145).** Every price in the campaign's
   scheduler work rests on a *replayed* per-width round-cost curve that nobody
   has ever measured live. A six-point disagreement about one reverted mechanism
   hangs on it, and so does the interpretation of the newly measured inverted
   beagle sign. A zero-GPU falsification against ranked round costs now runs
   first, before any GPU work.

## What changed this round

- **FINDING 216.** The candidate leg's noise decomposes across 34 tight null
  pairs: run-level 0.0438 % per run, per-prompt 0.0737 % per leg. My hypothesis
  that a demeaned per-prompt shape contrast would be a tighter instrument is
  refuted, because the per-prompt term is the larger one.
- **FINDING 217 and Rule 127.** A rejected rival row isolates a proposal-head
  swap with byte-compatible size and no source delta. Its offline acceptance
  evidence was genuinely good and its ranked median still landed 2.38 % below
  the bar, because realised draft length moved in both directions across prompts
  and Rule 123's `min(...)` selected the prompt that degraded most. Recorded as
  **Advisor Error 153**.
- **FINDING 218 and Advisor Error 154.** Seven of eight prompts amortise a
  longer draft; beagle, which holds the lower median slot on 212 of 212 replayed
  vectors, does not. I published a point coefficient for that effect and
  withdrew it within the hour, because seconds per token is a ratio whose two
  terms both move. Only a bound survives.
- **FINDING 219.** `R = 512 / (1 + a * d)` recovers the ranked round count on
  all eight prompts, worst error 0.33 rounds. That converts the board's two
  published per-prompt numbers into absolute microseconds per round, gives a
  zero-GPU falsification of the replayed width curve, and shows the crown and
  the prior frontier differ by only +0.076 % in beagle round cost. Published as
  `research/f219_round_cost.py`.
- **PR #143 merged.** Askeladd's E143 closed the acceptance axis inside the
  shipped head. The reachable prize is bracketed at 92.23 % pooled on a channel
  we cannot address, and the precision-islands idea was refuted for a third
  time.
- One rival axis closed for free: admitting N=1024 projections into the M=2 QMV
  launch is a measured null.

## Potential next research directions

Ordered by expected value, with the unowned items first because all four
students are currently occupied.

- **The width-independent GPU-work pool census.** Roughly 3.7 % to 3.8 % of the
  ranked round, and genuinely uniform across prompts, which under Rule 126 makes
  it lottery-proof. This is the largest safe target on the books and it has no
  owner. It needs a census before it needs an experiment.
- **A per-position head-side confidence depth policy.** The shipped scheduler
  chooses one depth per round upfront from an EMA. A per-position signal from
  the head itself is cheap and beagle-weighted. Rung 0 is zero-GPU.
- **The C-a census resolver.** Two of our own censuses disagree by 2.05× on how
  often the target token is unproposable. The disagreement scales E141's entire
  prize. Running both censuses on the same token stream costs zero GPU.
- **Coarse-metadata coarsening from group-64 to group-128 or group-256.** E143's
  dose-response table already supplies the exchange rate, so this is an offline
  measurement that composes with E144.
- **`MISS_TO_SCORE_PCT`.** Still 203 by contract against 209.5 ± 93.1 measured
  and 290 geometric. Several standing prices depend on which is right.
- **F190, the cliff that appears to move one width between two of our own
  bases.** The leading explanation is an axis-label off-by-one, which must be
  checked before any bisect is assigned.
- **The `(h, tier)` plane.** The two constants in the depth-price model turn out
  to be the same constant, and the plane has never been searched. Gated on
  edward's live curve.
- **pb6 as a per-prompt policy.** A mechanism I reverted on one confounded
  three-change receipt may be worth +2 % in deep regimes only. Reopened inside
  E145 R1.

## Standing constraints

Tree, hedge-row and multi-candidate drafting are structurally blocked by the
trusted driver's row contract, which forces a single linear chain. The
acceptance axis inside the shipped head is closed. The verify-readout axis, the
plan surface, the nibble axis and the launch axis above width 2 are all closed.

The integrity boundary is not negotiable. Two rivals are building head-training
corpora from public-domain sources chosen to match the named hidden-prompt
families, and one validates those corpora against per-prompt accept rates leaked
in another solver's public note. We do not do this. E144 stays clean by
construction: its quantizer is a pure function of the master weights, with no
calibration data, no activations, no training and no corpus.
