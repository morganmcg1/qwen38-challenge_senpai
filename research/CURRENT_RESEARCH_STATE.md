# SENPAI Research State

- **2026-08-22 ~15:20Z**
- Most recent research direction from the human researcher team: none received this round. The
  campaign runs autonomously under `senpai/program.md`.

---

## Where the campaign stands

Our best measured candidate `0c6191b7` is **at parity with the crown on candidate merit**.
Priced on the F154 instrument against the live crown `0b8602e1` (nagaral, published 3.51925374):
unweighted eight-prompt candidate mean **+0.0400 %** in our favour, F83-weighted **+0.0218 %**
against us. The common-denominator leaderboard anchored on the crown puts the crown at 3.519254
and us at 3.517402, a gap of 0.053 %. The published gap is the pinned-serial lottery, not a
mechanism.

**Finding 162, this round.** The crown's public note prices its restored probe-select mechanism
at +0.72 %, with a cited isolated pair at +0.88 %. Both are published medians. On the
eight-prompt candidate leg the cited pair is **−0.0359 % at z = −1.92, with four of eight prompts
moving the wrong way**, and its entire F83-weighted gain is one prompt. The real promotion pair
reads **+0.064 %**, which reproduces our own independent price of 0.074 % from campaign constants.
The whole field is bunched inside 0.23 % on merit, and no rival has produced a mechanism worth
more than 0.13 % in six hours.

**`623e77af` is in flight**, thorfinn's per-width templated Route B entry point with one-pass
accumulation at widths 6 and 7 on askeladd's D_S body. Pre-registered central prediction −5 % on
the candidate leg, band −3 % to −9 %, refuted if worse than −1 %. It is an order of magnitude
larger than anything the field has shown.

---

## Current research focus

**The campaign's central thesis is unchanged and every live mechanism serves it.** A ranked
M = 1 round moves 14.41 GB in 31,182 µs, which is 462.2 GB/s and already at the DRAM bound. A
beagle round at mean width 5.38 costs 55,870 µs for 4.38 tokens. If wide verification cost what
M = 1 costs for the identical weight bytes, the published ratio would be about 5.34 against
today's 3.34.

> **The entire remaining gap is the tax wide verification pays for zero extra weight bytes.
> Finding 157 names it exactly: `38/IPG + 25/RPS` statements per output element, against a
> load-only floor near `0.83 + 2.25`. Every other axis is worth 0.4 % to 1.5 %. This one is worth
> the rest.**

Four live mechanisms, each owned by one student, no two sharing a file:

| mechanism | owner | predicted | state |
|---|---|--:|---|
| one-pass verify table at widths 6 and 7 | thorfinn | −5 % candidate leg | **submitted, validating** |
| bit-exact load-width vectorisation (W1/W2/A1) | thorfinn | −1.8 % ranked for W1 alone | compile table ordered |
| wired-slack ladder to the bound-C ceiling | alphonse | −0.675 % | twelve legs ordered |
| sketch-first draft readout (C1) | askeladd | +1.15 % | offline screen, gates not yet cleared |
| depth-schedule repair at the cost cliff | edward | +0.50 % held-out to advance | estimator inverts, cause found |

### The load-width axis is the largest thing now in reach

At the shipped `IPG = 6, RPS = 4` the cell costs 12.583 statements per output element. Three
bit-exact forms, all read at source this round, all leaving accumulation order untouched:

- **W1** — the weight side does four separate scalar `uint16` loads of eight contiguous bytes
  while the activation side of the same kernel already uses one vectorised load. Alignment is
  provably safe at every scored K. Removes about 3 of the 38.
- **W2** — vectorised nibble extraction; the nibble block is 28 of those 38 statements.
- **A1** — vectorised bf16 widening; 16 of the activation side's 25.

Together they take the cell to about 7.08, which is **−44 %**. The static count conflicts with
E123's occupancy-contaminated ladder (Rule 97), so the honest range is wide, but even a third of
it dominates every other open axis. **One `xcrun metal-tt` compile on `applegpu_g17s` decides
W1**: four discrete 16-bit loads means it is real, and coalesced loads mean Finding 157's
decomposition is mislabelled. Both answers are worth having.

### Two closures this round

**Finding 163 — the wired allowance is a first-come-first-served pool with no floor.** Thirty-six
resize draws and 109 steady draws, byte-identical across all of them. The head is never excluded.
The 448 MiB step from `s64` to `s512` admits only post-sizing KV, Gated DeltaNet and scratch
state; no weight class appears in the class delta. Residency is monotone in slack with slope 1.0
and the allowance is exhausted at both 64 and 512 MiB. Bound A is withdrawn; bound C's ceiling
near 2,154 MiB is the only limit, and two doublings inside it have never been timed.

**The head-versus-backbone question is answered**, not by the serial leg — which has residual
sd 0.4788 % against the candidate's 0.0498 % and cannot resolve a 0.197 % effect — but by an exact
class-shrink argument with zero events over 109 draws.

### The measurement discipline that is doing the work

- **Finding 160**: the ranked numerator is causally fixed, so ranked value equals the **absolute**
  candidate improvement. The local serial-to-MTP ratio is 9.6× noisier on a causally wrong
  estimand. Absolute candidate seconds per token is the headline of every local arm.
- **Finding 154**: the eight-prompt candidate mean has se 0.0187 % against the published median's
  serial contamination at 0.21–0.24 % per prompt. It is a 5.17× sharper ranked instrument, and
  it is what exposed Finding 162.
- **Rule 102**: a counterbalanced palindrome cancels drift in the arm means and destroys it into
  the within-arm spreads. Never read dispersion from one. Four-arm ladders now use
  `y ~ arm + leg_index` for real degrees of freedom.
- **Rule 104**: every incremental AUC, R² or fit-based gain is reported next to its permuted null
  floor under the identical protocol.

---

## Potential next research directions

1. **Finish the load-width axis.** W1, then W2, then A1, each measured independently, then
   composed only after each has its own matched receipt. This is the highest expected value on
   the board and it is entirely bit-exact.
2. **Ladder the wired slack to the bound-C ceiling** and ship the fitted argmax rather than the
   largest value anyone happened to time.
3. **Settle C1 by extending the byte search upward.** Advisor error 121 capped the search at
   200 bytes per row where the payout is maximal; the mechanism pays to about 1,400 bytes per row.
   Whitening the projected coefficients before int8 quantization is untested and may be the whole
   difference between failing the gate by 9× and clearing it.
4. **C2, quantizing the bf16 precision islands to affine-4 g64**, +0.38 % to +0.45 %, reopened and
   unowned. It needs `Qwen35.swift`, so it queues behind E129.
5. **Restore francip's probe-select**, priced honestly at +0.064 %. A rider, not a direction, and
   it collides with both E129 and E133 in the same file.
6. **Convert the depth-schedule oracle gap.** The replayer says a perfect depth choice is worth
   +8.52 %, the shipped estimator inverts at the depth-4 cost cliff through Berkson selection,
   and the predicted fix is a live observable the scheduler already holds. Bound the prize with
   a cheating arm before building any estimator.
7. **Watch the rival board for two specific rows** under Rule 93: `48f182c6` (batched draft-id
   readout, our unassigned direction 6) and `19677283` (probe fraction 0.15, our retired C4).
   Both are free external tests of ideas we hold.
8. **If the one-pass receipt promotes**, refit the round-cost curve and replay the
   `headStepCostRatio` bracket against the post-arm curve. Zero GPU. The depth-price axis closes
   permanently if the optimum stays at 0.18.

---

## What would change the plan

A promotion of `623e77af` moves the base and makes composition order the immediate question. A
rejection leaves the base still and makes the load-width axis the next submission. Either way the
next submission is decided by measurement, not by waiting.
