# SENPAI Research State

- 2026-08-22 22:35 UTC
- No new research direction from the human researcher team since the last update. The standing direction remains the one in `senpai/program.md`: maximise the official decode score on `qwen3.8-27b-mtp-v1`, submit autonomously, and never stop at synthesis.

## Where the campaign stands

Board crown `08b67f12` (jungjipdo) at `3.69071883`. Our best promoted row is `623e77af` at `3.52085227`, now sixth. Our submission `572b2cc4` has been validating for about 100 minutes; the band is 42 to 130 minutes.

`upstream/main` has moved to `1d66bb36`, the crown's exact promoted tree. I read it at source tonight. The crown is a SIMPLER tree than ours: our editable surface is 950 lines larger and differs in only two files. Its scheduler is our own promoted code, citing E56, E68 and E75 by name, and it ships `depthPriceArm = .ship`.

**Our tree is structurally behind the crown by about 0.83 to 1.10 %, and the whole deficit is our own tiered one-pass QMV entry points (`+0.2649 %` candidate leg under a tight grid) plus our probe fraction at 0.25 rather than 0.15 (`+0.2603 %`).** Both are one-line repairs. `572b2cc4` is therefore forecast at `3.66` and is expected to be rejected.

Our edge is four mechanisms that nobody on the board has: the `pb6` boundary depth price, the probe-fraction knee below `p = 0.10`, the N-keyed QMV plan table, and the parameter-free depth argmax.

## Current research focus

**1. Ship the composed tree tonight.** Thorfinn is composing the tight launch grid, `pb6`, and probe fraction `0.10` on our base. Forecast `3.769`, against a gate of `3.69071883`. A two-line revert of `Table.compiledDefault` to `.shipped` is prepared and held behind a decision rule keyed on the `572b2cc4` receipt.

**2. The dispatch table is the campaign's centre of gravity.** Three separate experiments now aim at it. Thorfinn's launch geometry changes how many threadgroup columns are launched. Alphonse's plan table changes how many passes each shape takes, keyed on `N`. Edward's argmax changes how the scheduler reads the resulting cost curve. They interact, and CAMPAIGN RULE 117 now requires every fitted scheduler constant to name the dispatch-table state behind it.

**3. Weighting is now exact.** CAMPAIGN RULE 116: the published median is the identity `(beagle_raw + essays_raw) / 2`, confirmed on ten trees from seven solvers, weights `0.478` and `0.522`, all six other prompts exactly zero. Finding 83 is withdrawn as an instrument. Every held-out median must be computed by sorting, never as a weighted sum, because a weighted sum cannot represent the saturation cap.

**4. The beagle asymmetry is the largest unclaimed prize on the board.** Beagle must rise `9.10 %` to reach essays and pays `0.478` per unit the whole way. Essays pays `0.522` per unit for only `1.37 %`, after which republic overtakes it and further essays gains pay exactly zero. Raising beagle to essays' level moves the median `+4.35 %`; the hard ceiling is `+5.07 %`. Beagle is also the weakest of the five high-acceptance prompts: per-step acceptance `0.9341` against `0.9598` to `0.9661` for the others.

## Potential next research directions

- **Import `1d66bb36` as the campaign base.** It starts from a measured `3.69071883` instead of a forecast `3.66`, removes our two deficits by construction, and porting `pb6` into it is roughly six lines because `makeBoundaryDepthPrice`, `prefixCosts` and the arm enum are already present. Refused tonight only because three of four students are editing `Qwen35.swift` in flight. First action of the next generation.
- **Beagle's acceptance deficit.** Unowned, strongest free-slot candidate, worth `+4.35 %` to `+5.07 %`. Needs a hypothesis about why beagle's per-step acceptance is 2.6 to 3.2 points below the other four high-acceptance prompts.
- **The column-count ladder.** Thorfinn's next rung after the composition. Separates the flat, linear and logarithmic models of the launch saving completely, and reopens the `{8:8}` and `{9:9}` one-pass plans that were closed on a register basis measured under a WIDE launch where the column count was `M` in both arms.
- **The probe-fraction asymptote.** Askeladd is finding the recall knee below `p = 0.10`. The gross ceiling is `+0.85 %`, about `+0.81 %` of published median after the `0.95` conversion. The probe fraction is the campaign's only mechanism whose relative effect is uniform across prompts, so it converts to median with no weighting at all.
- **P4, the GDN S=2 mid-state write.** Unowned, `0.2` to `0.6 %`, gates 151 MB per round on rejection.
- **The head-history fold warm gap.** Widths 1 to 9 are flushed but only 2 are warmed. Unpriced and must clear Rule 110 before it is worth a slot.
- **C2 precision islands to affine-4 group-64.** Reopened, unowned, `+0.38 %` to `+0.45 %`.
- **Finding 190, the cliff that appears to move one width between two of our own bases.** Check the E92 axis label before assigning any bisect: if E92's `M` is the draft count rather than the verify width, the cliff never moved and only the label did.

## Closed tonight

- Prefill as a scored target. Finding 193 is now confirmed twice, once from the enforcing workflow source and once by a rival's `+0.5` to `+2.5 %` prediction that measured `-0.0557 %`.
- Finding 83 as a weighting instrument.
- A blanket revert of the one-pass table as a strategy, as distinct from the N-keyed form.
