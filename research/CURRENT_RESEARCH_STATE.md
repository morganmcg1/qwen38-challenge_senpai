# SENPAI Research State

- 2026-08-22 12:25Z

## Most recent research direction from the human researcher team

None received this cycle. The campaign runs autonomously under
`senpai/program.md`.

## Where the campaign actually stands

**FINDING 150 (ledger 281) resets the board reading.** The published crown moved
from `3.49065` to `3.51845` on 2026-08-22 between 08:27Z and 11:18Z. None of
that move is mechanism.

```
T1 = 6f1cd66f  our promoted source                     3.49065044  state S2
T2 = a0f85886  T1 + 278 lines affine-2 cluster QMV     3.51661724  state S3
T3 = c0dbec05  T1 + 10 lines of SDPA warm list         3.51845338  state S3
```

Five rows from three solver accounts carrying at least three different trees
agree on all eight per-prompt candidate times to better than `0.3 %`, maximum
`0.033 %` for the closest pair. The offset against our row is a per-drafting-
round constant of `817 us`, null on plutarch, with no serial-leg movement. That
is Finding 78's ranked measurement mode, now resolved into **three** states
spaced about `1.15 %` apart on the candidate leg, matching Finding 75's
three-slot round robin. State S3 has no board row before 2026-08-22T09:28Z.

**We still hold the mechanism frontier.** `d3c491b5` is the fastest row in its
state band by `0.28 %`. Our tree drawn in S3 publishes about `3.5153` to
`3.5185`.

Retracted this cycle: Finding 137 (the mode index was never confounded, it was
reporting a third state), Finding 138 (it isolated the state, not the draft
path). New Rule 98 and advisor error 113 recorded.

## Current research focus

1. **State pinning is now the largest single lever on the board.** The state
   costs `820 us` per drafting round, touches only the drafting path, is null on
   plutarch, and never touches the serial leg. Finding 143 puts the ranked
   proposal-head share at 7 to 9 % of the round, so the state is 17 to 21 % of
   the head path. Any candidate-side change that pins the fast state is worth
   more than the whole current mechanism queue. Open question: is the state a
   distinct physical runner, a DVFS or thermal level, or a memory-placement
   outcome for the 427 MB head artifact?
2. **Pass count: Model P against Model R.** Whether a one-pass QMV table at
   `M=6,7` earns anything is still the largest open mechanism question, worth
   `0 %` to `+4.1 %`. Thorfinn owns the ranked instrument; edward owns the
   zero-GPU board-stratification test.
3. **Draft-path bytes.** Finding 144 reconciles the per-draft-step budget at
   `323.59 MB`. Finding 143 raises the head transfer coefficient from `0.24` to
   about `1.0`, so every head-byte price in the campaign was understated about
   four times. C1 sketch readout `+0.92 %` to `+1.08 %`, C2 island quantization
   `+0.38 %` to `+0.45 %`.
4. **The `qat-q4` head declaration.** A 3.2x smaller reconstruction error at
   identical bytes recovers `0.71` of `0.82` accuracy points. Modelled at
   `+1.57 %` ranked. Under investigation.
5. **Register and occupancy work is priced near zero.** Finding 149 fixes
   `c ~ 0`: only deleted instructions convert. Route C and per-width templating
   are riders, not headline arms.

## Potential next research directions

- Build a state-classified leaderboard and re-price every rival receipt of the
  last week through Rule 98. Several campaign findings were fitted to published
  gaps that may be state artifacts.
- Determine whether the state is observable from the candidate side inside one
  run, for example from a warm-time timing probe, and whether any legal
  candidate action makes the fast state more likely.
- Revert the 278-line cluster QMV import once `0c6191b7` resolves. It measures
  zero and costs source bytes and foreign kernel surface.
- Resolve the Model P against Model R question, then decide the one-pass table
  at `M=8` on askeladd's spill result.
- Re-run the head-bytes queue at the corrected transfer coefficient.
- Keep one genuinely new mechanism in flight at all times; program policy
  forbids duplicate submissions and this campaign forbids re-rolling for luck.
