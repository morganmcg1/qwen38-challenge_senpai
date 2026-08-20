# SENPAI Research State

- 2026-08-20 15:10 UTC
- Most recent research direction from the human researcher team: Issue #22 —
  execute aggressively toward the winning frontier. No new human direction since.

## Where the campaign stands

Crown: **3.25592233**, submission `89cbdc02`, solver yijunyu, source
`96f20f872932e3ce2fd36209d36bf81d107dd02d`. `upstream/main` is that exact
source. Second place `742cdf67` at 3.25464080 is a pure resample of the previous
tree.

Our best official score: **3.23588901**, submission `9b241879` (arm 2). Deficit
to the crown: **0.619 %**.

Campaign advisor branch: `9ec6e087c17963a5223454c265cd696fb1fa6228`. Campaign
`main` is still at `770a3ff2` and must be synced to `96f20f87` before any
submission.

**The ranked slot is free. No submission is in flight.**

## The single most important fact in the campaign

The published score is the mean of the 4th and 5th sorted per-prompt ratios, and
the board now tells us exactly which prompts those are. Over the 248 scored runs
above 3.15, the 4th is **beagle in 100 % of runs** and the 5th is **medicine in
91.5 %**.

So the score is `(beagle_raw + slowest of the other four wide prompts) / 2`.
At the crown tree, republic, medicine and botany all sit above the 5th slot and
are worth **exactly zero**. plutarch, drama and travel have been worth zero
throughout. beagle carries a fixed 50 % of the score and sits 9.6 % below the
next-lowest per-prompt floor on the board.

**Every experiment must state what it does to beagle.** An experiment that
cannot move beagle or the current 5th prompt is worth nothing, however large its
local effect.

## Two public instruments we only just started using

1. **`officialMetrics.per_prompt` is populated for all 596 scored submissions.**
   One API call returns eight rows per run with candidate and serial seconds per
   token, raw ratio, effective mean draft length, non-drafting round count and
   the declared head provenance digest. The serial denominator varies by only
   0.21–0.24 %, so candidate times are comparable across every seat on the
   board. Tool: `research/board_per_prompt.py`.
2. **`git fetch upstream` creates a branch for every submission**, rejected and
   validating ones included. Joining source to per-prompt timing gives a full
   factorial over any mechanism a competitor has tried.

Ledger 220 records both, with the tables.

## Current research focus

### 1. The width cap and the streak gate — the largest cheap prize

The factorial is solved from the board. `cap 7 + streak gate` produces three of
the four fastest beagle legs ever recorded, from three independent solvers, and
it is the only cell that reaches beagle above 3.16. It wins beagle, essays and
republic against the crown and loses only medicine. Repairing medicine inside
that cell is worth **+1.43 %**, to a median of 3.302636.

The diagnosis is that `fullAcceptStreak` is a single-round signal that cannot
tell a hard prompt from an unlucky round. `positionAcceptEMA` is a multi-round
signal that can. The arm is a one-parameter family that nests both board
champions as its endpoints. This is E81, going to thorfinn when his E75 r3
lands. The `cap 8 flat` cell has never been run by anyone and is one line.

### 2. The proposal head — highest ceiling, worst track record

The pinned head has never been trained: its `precision_islands` are bit-exact
against the `EigenLabs` master. 19 of 25 island-bearing published heads are
likewise pure re-quantization artifacts. Only `xkm` produced a substantial
chain-faithful, quantization-aware retrain, and it ships no `draft_lm_head` and
a BF16 trunk at twice the weight traffic.

The board's verdict is blunt: across ~40 custom head digests, none has ever
beaten the pinned head on beagle, the best falling short by 3.44 %. The most
controlled case is hadakang losing 3.0 % of score to a custom head on the same
day as their own promoted pinned-head tree.

E82 with alphonse holds the readout and the footprint fixed and varies only the
trunk weights — the arm nobody has run. Its rung-0 stop rule now requires a
gain on the hardest tercile of the screen corpus, because gaining on easy text
and losing on hard text is the exact failure signature of every distill on the
stop list, ours included.

### 3. The kernel table — small, real, and ours alone

The ranked field is capped at `NA <= 4`; we are the only participant that has
ever run `NA = 5` or `6` on the ranked host. A competitor's own closure states
that removing the structural weight re-read across m-groups requires `NA > 4`.
Arm 2 measured our table at −0.383 % on candidate time across 8 of 8 prompts,
p = 0.0039, but only +0.105 % of median because the gain was not concentrated on
beagle. E78 with askeladd tests an `out_vec_size`-gated table; the pre-registered
prediction is −0.62 % candidate time, +0.17 % median.

The register axis and the occupancy axis are both closed with bounds. The
leading live explanation is per-shape grid starvation with a knee at 77.9
threadgroups per core, and the knee is per core, which is what transfers from
our 20-core hosts to the ranked 40-core host.

### 4. Where the time actually goes

E80 with edward replaces dispatch counts with GPU time. 22.6 % of the verify
width tax is still unattributed, and dispatch count has already misled us by
about three orders of magnitude once, on the copy family. No new kernel
assignment should be made from the dispatch census until E80 returns.

## Potential next research directions

- **Prompt-adaptive depth by acceptance state.** The generalisation of E81.
  Every one of the 596 board runs uses either a flat cap or the same binary
  streak gate. Nobody has shipped a graded, acceptance-conditioned width policy.
- **Entropy-gated early stop of the draft chain.** Untried at rank by anyone.
  Adjacent work exists: four top-2 margin policies lost locally, and the
  depth-0/1 tempering ladder is in-tree and extending it set a beagle minimum.
  Nobody has used the head distribution's entropy as the gate. Matches AdaEDL
  (arXiv:2410.18351), which reports +10–57 % training-free.
- **Objective-matched head training.** Train against the schedule's marginal
  value objective rather than top-32 cross-entropy. Every failed head on the
  board optimised the wrong objective.
- **The prefill QMM tile.** Charged inside the timed leg and, by a competitor's
  own note, never moved by anyone.
- **Composition risk.** Kernel and schedule arms are substitutes and share
  33.1 % of their effect. Anything composed must be re-measured, not summed.

## Rules that currently bind hardest

- The local fixture runs at mean verify width 7.27 against a ranked 5.82, and
  the local total marginal cost ratio is 0.4023 against a ranked 0.2136. Local
  whole-leg numbers are not arm rankings. Local per-width and per-cell numbers
  are, after reweighting. Local prefers a shallower cap than rank does, so a
  local "deeper is better" result is one-sided in our favour.
- The candidate distribution is right-skewed with a hard floor. Compare minima,
  not means. A bad draw and a drafting-side regression are indistinguishable by
  score alone; same-tree resample sigma is 0.2 % to 0.76 %.
- The schedule is a closed loop, price to proposals to acceptance to EMAs to
  depth, and it is demonstrably bistable on plutarch at rank. Open-loop pricing
  does not survive contact with it.
- The price level is bracketed on both sides at rank. The way to buy depth is
  the cap, not the price. Do not retune `headStepCostRatio`, and do not reseat
  `positionAcceptEMA`.
