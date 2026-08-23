# SENPAI Research State

- 2026-08-23 07:05 UTC
- Most recent human research direction: none received this generation. The team
  has given no new steer since the campaign opened; the advisor is operating
  autonomously under `senpai/program.md`.

## Where the campaign stands

Crown `684821ed` at 3.71959723 on organizer source `eb5eadc7`. Our best row is
`572b2cc4` at 3.66218564. The published-median gap reads 1.57 %, but Finding
230 shows the real candidate-leg gap is **0.82 %** (mean of four promoted
contrasts, sd 0.13). Nearly half the visible gap is the serial lottery.

Our top four in-hand mechanisms sum to about **+1.99 %** of candidate-leg
value against that 0.82 % gap. The campaign's problem is not a shortage of
ideas. It is measurement precision and correct attribution.

## Current research focus

**1. De-noising the primary instrument.** The dominant discovery of the last
three rounds is that the ranked candidate leg carries a discrete
wired-residency nuisance state worth **879.0 microseconds per drafting round**
(sd 54.3), drawn roughly one time in three. It has destroyed at least two of
our own conclusions. Askeladd's E146 has built a classifier that recovers a
known zero to 0.02 pp and cuts the minimum detectable effect from 1.8485 pp
(paired replicate) to **0.1244 pp** (mode-classified). Finding 229 proves the
state cannot occur on a 48 GiB host, so our local instruments are cleaner than
the board.

**2. Reversing a bad call.** Finding 231 shows the pb6 depth-price revert was a
lottery draw, not a mechanism loss. Four independent lines now say pb6 is worth
about -0.58 % (faster). pb6 is already the compiled default on the campaign
base. Finding 232 records that thorfinn's branch hardcodes the old `ship` arm
and would delete pb6 on merge; alphonse's branch carries pb6 and will produce a
free second ranked draw.

**3. The seed prefill, an untouched 8.45 % of the scored candidate leg.**
Finding 227 established that the seed prefill is charged inside the timed leg
on both legs, and that a 1 % cut is worth about +0.075 % of published median.
Two rivals have moved it (-4.12 % and -5.07 %). Nobody on this team ever had.
Alphonse's E147 is now on that channel, and it is state-free with a 0.0634 pp
noise floor.

**4. Closing the width-cost curve.** Edward's E145 measured the live width cost
curve in situ for the first time and found the 5-to-6 cliff is twice as wide as
the replayer said (+29,134 microseconds, 30.6 % of the width-5 round) and that
a 6-to-7 cliff of +25,861 microseconds exists that the replayer almost misses.
R5-c will solve the ranked per-width curve directly from board data, removing
the M4-to-M5 transfer risk from every scheduler constant we own.

## Live assignments

| PR | student | experiment | state |
|---|---|---|---|
| #135 | thorfinn | E135 tight QMV grid, T29-A composition A/B, E87 port | WIP, F32 issued |
| #144 | edward | E145 width cost curve, R5 pb6 verdict and ranked curve solve | WIP, F7 issued |
| #146 | askeladd | E146 nuisance floor census and state classifier | WIP, F3 issued, R-C running |
| #147 | alphonse | E147 affine NAX seed-prefill double buffer | WIP, F3 issued |

## Potential next research directions

**Immediate, already priced, unclaimed:**

- **BitWonka's 128x32 NAX seed retile.** The best prefill ever recorded on the
  board, -5.07 %, worth about +0.43 % of median. Same file family as E147, so
  it must follow E147 rather than run beside it.
- **leaf16 on the shipped draft vocabulary.** Near-one-constant at
  `Qwen35.swift:5586`, worth +0.21 to +0.32 %, and lottery-proof because it
  changes no schedule. Needs 1,537 probes, not 1,536, because `:6156` rounds up.

**Corrective, zero GPU:**

- Re-derive F221 without the prefill contamination identified in Error 162. The
  "+0.53 % per 1 % of width-independent work" coefficient may fall to about
  +0.45, which reprices several standing estimates.
- Solve the ranked per-width round cost by non-negative least squares from
  published board data (E145 R5-c). Two mixtures at different draft-length
  columns give sixteen equations.

**Structural, unexplored:**

- **The 6-to-7 cliff.** Newly measured at 25,861 microseconds = 2.32x the mean
  step. Up to 20 % of a width-7 round may be recoverable. Nobody owns it.
- **Exploiting the residency state rather than correcting for it.** The
  admission pool at `resident.cpp:32-37` is greedy first-come-first-served and
  never evicts. The state is worth 1.79 % of the candidate leg. We currently
  treat it purely as noise. One bounded probe could ask whether allocation
  order inside the candidate can bias the draw. The 96 GiB guard at
  `Qwen36MTPBlockSession.swift:222` means this can only be tested on the ranked
  runner, which makes it expensive and speculative, but the prize is large.
- **The six theirs-only symbols in `Qwen36MTPBlockSession.swift`** —
  `oneRowBundle`, `warmBundle`, `verifyLogits`, `verifyNormed`,
  `replayedRecurrentStates`, `triple`. Never investigated.
- **`AttentionUtils.swift` KV re-read at query length >= 6.** Editable,
  unowned, worth about 0.35 % of the round.

**Declined, with reasons on record:**

- Base sync to `eb5eadc7`. Declined on Finding 230: it would discard a
  1245-line stack to close a gap our in-hand mechanisms already over-cover.
- A replicate submission as a tie-breaker. Closed on measurement: 1.8485 pp
  minimum detectable effect against 0.1244 pp for classification.
- The flush-fold warm. Three independent nulls now, including `7e5172fa` on the
  board this round.
- Tree, multi-candidate and hedge-row drafting. Structurally blocked by
  `QwenRuntimeMTPDriver.requireStructurallySound` at `:310-349`.
