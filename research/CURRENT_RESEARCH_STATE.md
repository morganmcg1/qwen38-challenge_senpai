# SENPAI Research State

- **2026-08-23 01:45 UTC**

- **Most recent research direction from the human researcher team:** none received this round. The campaign is running autonomously under `senpai/program.md`.

---

## Current research focus and themes

### The objective function changed this round, and it is now the organising fact

The published score is the median of eight per-prompt raw ratios. The median of eight is the
mean of the two middle values **after sorting**, so it is an order statistic and not a
weighted average. Two findings this round turned that from a technicality into the dominant
constraint on every decision.

**Campaign Rule 121** — price on the sorted order statistic. Predict all eight raw ratios,
sort, read positions 3 and 4, and report the predicted rank vector.

**Campaign Rule 123** — beagle sits reliably at rank 3, so:

```
published median = ( beagle + min(essays, republic, medicine, botany) ) / 2
```

The upper slot is a **minimum over four prompts**. It is a worst case, not an average. A
mechanism that helps essays enormously and hurts republic loses. Two independent solvers have
now each thrown away more than a percent of published median by ignoring this: our own
`e003a86d` and, an hour later, `09b452f3`, whose candidate leg is the fastest on the entire
board and which still scored eleventh.

### The consequence: one axis is worth 9.35 times all the others combined

On the live crown vector:

```
prompt      ceiling value of the median that this prompt alone can reach
beagle          +4.6336 %       needs +9.670 % of beagle raw ratio
essays          +0.4957 %       needs +0.955 % of essays raw ratio
all others      +0.0000 %       at any magnitude
```

Beagle's deficit is **acceptance, not round time**. Its rounds are already cheaper than
essays'; it loses by running 18 more of them. Closing beagle's per-step acceptance gap to
essays' level is worth about +4.3 % of published median, which is 92 % of the entire
remaining ceiling of this benchmark.

**Everything now points at beagle acceptance, or at uniform gains.**

### Two classes of work are worth pursuing, and only two

1. **Beagle-specific gains.** Uncapped, +4.63 % of runway. Owned by Askeladd's E143
   acceptance decomposition, whose C-d fork is a campaign-level decision: if the head is
   genuinely wrong on more than 80 % of beagle's first divergences, the axis is unreachable
   and four students get redirected.
2. **Uniform gains.** They convert exactly one to one, have no ceiling, and cannot manufacture
   a new minimum in the upper slot. The largest identified uniform target is the
   width-independent GPU-work pool: about 3.7 to 3.8 % of the ranked round with no owner.

Non-uniform gains are still worth having, but each must be priced through
`research/f209_reorder_value.py` and `research/f209_compose.py` and reported with its rank
vector. Marginal-weight products are valid only for an infinitesimal order-preserving move.

### Measurement: the noise floor split in two

**Finding 211.** Twenty true null pairs across nine solvers, found by scatter tightness rather
than by draft schedule, because the schedule is deterministic and proves nothing.

```
                                          diff-of-two sd
candidate 8-prompt mean                       0.0731 %
serial    8-prompt mean                       0.1695 %
published median delta                        0.2253 %

model  published delta = serial medpair - candidate medpair
residual across all twenty pairs                 0.0093 %
```

The candidate leg is reproducible to 0.073 %. The whole remaining published-median noise is
the serial baseline leg, which is identical code in every row and which no candidate edit can
touch. Campaign Rule 118, price on the candidate leg, is now quantitatively justified: it is a
3.08 times more precise instrument.

Campaign Rule 72 is reaffirmed and load-bearing. The serial lottery is exactly the re-roll
lever that rule forbids. We do not resubmit an unchanged candidate. The only legitimate way to
move the score is to move the candidate leg.

### In flight

| PR | student | experiment | state |
|---|---|---|---|
| #135 | thorfinn | E135 launch geometry, then F22 width-6 register occupancy | reverting pb6, then submits the clean archive as a control receipt, then the free g17s `na6` census |
| #140 | edward | E140 depth-price cliff | zero-GPU R1 and R2 remain, then terminal |
| #141 | alphonse | E141 compact draft vocabulary widening | prize confirmed +0.9495 %; the round cost now decides whether it ships |
| #143 | askeladd | E143 beagle acceptance decomposition | R0 channel census pending; **highest value in the campaign** |

The clean archive forecasts 3.69900 against a crown of 3.70355, roughly a 29 % chance. It is
submitted for the control receipt and the Finding 210 falsification, not for the crown. F22 on
top of it forecasts 3.72576, which is about 97 % at its point estimate.

### Closed this round

- `depthPriceArm = .pb6`, refuted by ranked receipt and by source (Findings 209 and 210).
- The certified verify-readout screen family, and the whole verify-readout axis.
- The round-boundary bubble census as specified: the round is 99.93 % GPU-busy.
- Lossless weight-stream recoding: the cost is the load instruction, not the bytes.
- Tree, multi-candidate and hedge-row drafting: structurally blocked by the trusted driver's
  row contract, not merely discouraged.
- Block verification: exactly zero at temperature 0.

---

## Potential next research directions and themes

Ordered by expected value under Rules 121 and 123.

1. **Beagle acceptance recovery.** Whatever E143's census says is reachable. Uncapped, and the
   only axis with more than half a percent of runway. If C-d closes it, that answer is itself
   worth having quickly, because it frees the whole team.

2. **Attribution census of the width-independent GPU-work pool.** About 3.7 to 3.8 % of the
   ranked round has no owner and is uniform by construction, so it converts one to one with no
   ceiling. This is the largest safe target on the board. Instruments are already specified,
   including the never-run `sweepGatedDelta` gate. **Currently unowned; assign to the next
   student who frees up.**

3. **Per-position head-side confidence feeding the depth policy.** Read the head's own
   per-step shortlist top-1 to top-2 gap and entropy with one-round lag, and blend it into the
   reach estimator beside the slow acceptance EMAs. Point +0.5 %, band [0, +1.5 %], and
   beagle-weighted because beagle is the lowest-p carrier with bursty hard mass while the
   9-round EMA half-life reacts slowest exactly there. Rung 0 is zero-GPU on the cached E143
   capture. Named refutation available from E134 rung 1.

4. **The two-dimensional `(h, tier)` depth-price search.** Finding 210 showed that
   `makeBoundaryDepthPrice` holds the total, so `tier` and `within` are the same constant and
   the E134 grid swept a one-dimensional diagonal. The plane has never been searched. Must use
   the min-of-four objective, must flag every reordering cell, and must carry a
   `plutarch_unlock` flag. Under Rule 122 it needs a live same-binary A/B in beagle's regime,
   not a fixed-trajectory replay.

5. **Width-6 register occupancy on g17s (F22).** In flight with Thorfinn. Point +0.5996 %
   under Rule 121 and not capped. Hard kill rule at the register census.

6. **Compact draft vocabulary widening at a lower round cost.** Alphonse's arm B, generalising
   the `rowsPerCluster == 8` guard, or a step-1-only widened probe at roughly a quarter of the
   traffic. The recall result is settled; only the cost is open.

7. **C2 precision islands to affine-4 group-64.** +0.35 %, band [+0.30, +0.42]. Held as
   E143's fallback arm. Do not double-assign.

8. **GDN S=2 mid-state eager write.** 0.2 to 0.6 %. Start from a zero-GPU reject-rate split.
   Highest correctness risk of anything queued.

9. **Composition of the two measured held riders.** +0.2 to +0.3 %, using the zero-noise live
   acceptance instrument.

10. **Finding 190, the cliff that appears to move one width between two of our own bases.**
    Check the E92 axis label before assigning any bisect.

### Standing methodological commitments

- Price every ranked contrast on the **candidate leg medpair**, report the serial medpair
  beside it, and the published median third.
- Report the **predicted rank vector** for every mechanism, and flag any order change.
- Prefer **uniform** mechanisms; price non-uniform ones by their worst outcome among essays,
  republic, medicine and botany.
- A plateau is a map of where not to look. The board has been flat for four hours with five
  rejected attempts between 3.686 and 3.699; that is the serial lottery scattering near-equal
  candidates, not five failed mechanisms.
