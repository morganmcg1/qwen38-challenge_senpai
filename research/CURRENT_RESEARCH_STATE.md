# SENPAI Research State

- 2026-08-22 21:50 UTC
- Most recent human research direction: none received this session. The campaign runs autonomously under `senpai/program.md`.

## Where the campaign stands

Our promoted row `623e77af` at 3.52085227 is now sixth. The frontier moved to **3.69071883** in about three hours, and both leading rows are built from mechanisms this campaign generated and published: the tight QMV launch grid, and the derived cluster probe fraction. Publication is promotion under Rule 96, and the cost of that is now measured rather than theoretical.

The correct response is not to slow publication. It is to notice that we priced both mechanisms far too conservatively, by 2.4x and 3.3x, and to fix the pricing instrument.

One submission is in flight: `572b2cc4`, the tight launch grid on our own tree. A composed tree carrying three mechanisms is being built behind it.

## The current research focus

**Composition is the near-term theme.** Four mechanisms are independently measured and mutually disjoint in the source, and none of them has ever been run together:

```
tight launch grid          rival ranked receipt   +4.39 %
pb6 depth price, tier 1.45 held out, lower bound  +2.4683 %
probe fraction -> 0.10     measured zero cost     +0.5103 %
one-pass table under tight rival ranked receipt   -0.2187 %   (a cost we currently pay)
```

**Measurement fidelity is the deeper theme, and it has repaid more than any single mechanism this session.** Three pricing errors were found and corrected:

- The raw ratio hides sub-percent candidate effects, because the pinned serial leg is 9.6 times noisier than the candidate leg. Read the candidate leg. This alone repriced the probe fraction from +0.0992 to +0.3311 percent.
- A per-round absolute mechanism must be converted through absolute microseconds, never through a local percentage. The local round is about 3.6 times longer than the ranked round.
- Prefill is published but not scored, so round-side gains land undiluted. There is no 1.265 dilution factor.

## What is running now

```
PR   student   question
135  thorfinn  compose tight grid + pb6 + probe 0.10, gate, and submit when the slot frees
138  alphonse  grid x plan factorial; is the plan table keyed on N rather than (K, N, M)?
139  askeladd  where is the recall knee below p = 0.10, found from the margin distribution?
140  edward    does depth argmax beat the greedy walk on a non-convex cost curve?
```

## Potential next research directions

**Ranked now, by expected value.**

1. **The shape-keyed plan table keyed on N.** A ranked receipt and an isolated measurement agree that our one-pass table reverses sign under the tight grid, and the separator is the launched threadgroup count, not K. Floor estimate 2.7 percent, bit-exact by E137, and a single threshold rather than a cross product so the pipeline count stays bounded.

2. **Depth scheduling as an optimisation problem rather than a walk.** pb6 works because it accidentally repairs a greedy walk on a non-convex curve. If that reading is right, the parameter-free argmax should beat a hand-tuned barrier, and the whole depth-price tuning axis collapses into one correct decision rule.

3. **The probe knee.** One `Double`, zero bytes, zero dispatches, asymptote +0.85 percent. The live acceptance channel cannot find the knee because it quantises at one round; the recall margin distribution can.

4. **Beagle's acceptance deficit, still unowned.** Beagle carries F83 weight 0.4862 and is the median carrier at 98 percent, yet its realised acceptance is 0.834 against 0.85 to 0.90 for the other high-weight prompts. Closing that gap moves the median about 4.1 percent. This is the largest single untapped prize on the board and nobody is working on it.

5. **Widths 8 and 9 under a tight grid.** The `{8:8}` and `{9:9}` one-pass plans were closed on a register basis, but that measurement was taken under a wide launch where the column count was `M` in both arms, so it saw only half the trade. The closure should be reopened and re-measured.

**Held or unowned.**

- The fp32 tiebreak at the one narrowing point in the rerank kernel, +0.2166 percent, zero bytes.
- P4, the Gated DeltaNet S=2 mid-state write on rejection, 0.2 to 0.6 percent.
- C2, precision islands to affine-4 group-64, 0.38 to 0.45 percent.
- The head-history fold warm gap, unpriced, and it must clear Rule 110 before it is worth a slot.

**Closed this session.** Prefill as a target, refuted at the enforcing workflow source. The offline corpus acceptance screen is now bounded rather than discarded: it is accurate to 0.973 for byte removal at a constant retrieved set, and it was 18.4 times wrong for a change that replaced the scoring arithmetic.

## Standing method notes

- Read the candidate leg, not the published median and not the raw ratio.
- Convert per-round absolute mechanisms through microseconds.
- A rejected submission still publishes complete per-prompt evidence, so a probable rejection is not a reason to withhold or cancel a run that discriminates between live hypotheses.
- The acceptance gate is set by the crown that is live when a run finishes, not when it starts.
