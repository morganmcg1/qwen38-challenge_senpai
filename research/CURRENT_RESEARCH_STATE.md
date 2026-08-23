# SENPAI Research State

- 2026-08-23 ~01:10Z. Advisor base `892dc5e1` (Merge PR #142). Campaign base `origin/main` `770a3ff2`. Board crown `1760479a` (scarletbright) `3.70355222`, source `e8f14c44`. Our best receipt `572b2cc4` at `3.66218563656629`.
- Most recent human research direction: none new this cycle. The standing direction is to keep the frontier moving, submit strong candidates autonomously, and never hold a credible candidate for approval.

## Current research focus

**The published median is a sorted order statistic, and we just paid 2.38 percent of median to learn it.**

Our composition `e003a86d` carried the tight launch grid, `depthPriceArm = .pb6` at tier 1.45, probe fraction 0.15 and `Table.compiledDefault = .shipped`. Forecast `3.7747`. Measured `3.57502547`. Holding the median pair fixed at beagle and essays, the composition is a **−0.07 % null**. The entire loss is a reorder: essays improved +3.24 % and left the median pair, republic took the upper slot 4.30 % below where essays landed, and plutarch gained 88.78 % for exactly nothing.

The cause is in source. `makeBoundaryDepthPrice` holds the price TOTAL, so raising `marginal[4]` by the tier forces every other marginal from 0.18 down to 0.1704142. **`pb6` is a boundary price plus a hidden 3.5 to 6.7 percent global depth subsidy at the seven depths it was never meant to touch.** That subsidy, not the boundary, unlocked plutarch's 449 non-drafting rounds and deepened essays. Beagle pays the boundary at full strength and is the one prompt whose value cannot be replaced.

**The value model, now exact** (`research/f209_reorder_value.py`, anchored on `572b2cc4`):

```
  prompt     dM/dx at 0    CEILING x    CEILING value
  beagle       0.4795        9.940 %      +4.7639 %
  essays       0.5205        1.260 %      +0.6548 %
  all others   0.0000        0.000 %      +0.0000 %
  uniform gain across all eight            converts exactly 1:1, no ceiling
```

Three consequences drive everything below. Broad mechanisms are the safest class because they preserve order and never saturate. Beagle is the only prompt-specific axis with real headroom. Essays gains stop paying after 1.258 percent and **do not stack**, so the first mechanism to claim that headroom gets it.

## Live work

- **thorfinn PR #135.** Held from submitting until `depthPriceArm` defaults to `.ship`. Then the clean archive ships tight grid plus the width-2 launch shrink plus probe 0.15 plus `Table.shipped`, predicted 3.69 to 3.71 against a crown of 3.70355. A coin flip worth taking. Next lever is the width-6 register occupancy tax, gated by a free offline g17s register census, point +0.6 % and band [0, +0.93 %].
- **edward PR #140.** Cell D refuted and the depth-cap axis closed with `wants_past_cap = 0.0000`. Item B replaced: decompose `pb6` into arm S, the pure subsidy, and arm P, the pure boundary, then make every replayed median reorder-aware and re-flag past conclusions. Then terminal.
- **alphonse PR #141.** Rung 3 arm A widens the compact draft vocabulary from 98,304 to 248,320 while holding the absolute probe count at 1,844. Prize +0.95 %, cost +0.70 to +0.83 %, net −0.4 % to +0.25 %. Recall at fixed probed rows is the headline scientific quantity.
- **askeladd PR #143.** The beagle acceptance decomposition, now the highest-value open experiment in the campaign. R0 is a zero-GPU census on the cached E142 capture. Kill rule: if channel C-d exceeds 80 percent the acceptance axis is unreachable and I redirect four students.

## Potential next research directions

1. **The 2-D `(h, tier)` depth-price search with a live same-binary A/B in beagle's regime.** The direct successor to Finding 210. `tier` and `within` are the same constant, so the E134 grid swept a diagonal and the plane has never been searched. Under Rule 122 this needs a live A/B on beagle and essays fixtures, not `benchfixture`.
2. **Broad round-cost mechanisms, preferred over prompt-specific ones.** They convert 1:1 with no ceiling. The width-6 register occupancy tax and the width-2 launch shrink are both in this class.
3. **The round-boundary bubble census.** About 8.7 percent of the medpair round is unattributed by the F22 part table. Unowned. Point +1.0 %, band [+0.2 %, +2.5 %], a guess.
4. **Beagle-specific acceptance.** Beagle sits at per-step acceptance 0.9341 against essays at 0.9647, has the highest unproposable-token rate at 0.4832 percent, and its top misses are South American proper nouns. Any mechanism that improves low-confidence positions lands disproportionately on the one prompt with headroom.
5. **Per-position head-side confidence**, the sole named reopening signal from E99.
6. **The P4 GDN S=2 mid-state write**, gating 151 MB per round on rejection.
7. **F190, the apparent one-width cliff move.** Check the E92 axis label before assigning any bisect.

## Closed this cycle

The certified verify-readout screen family, refuted at the arithmetic floor: the epsilon floor is 0.478 against the 0.1417 required, and a certificate needs 3.694 bits against an exact head of 4. The entire verify-readout axis, at 3.56 percent of the round with no recoverable inefficiency. The column-count ladder, decision value zero under both readings of Finding 200. The one-entry mixed table at −0.064 percent. Finding 206's probe turnover, downgraded to a probable run-state artefact by a source audit that found only one probes-shaped dispatch of about thirteen per draft step.
