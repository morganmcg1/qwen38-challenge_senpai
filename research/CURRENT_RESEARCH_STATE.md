# SENPAI Research State

- 2026-08-22 ~21:15 UTC. Advisor base moves to the merge of PR #134 (edward, `pb6`).
  The campaign `BASE_SHA` passed to every submit call is unchanged at `770a3ff2`,
  because `senpai/submit-official.sh` reads `SOURCE_BRANCH="main"`.
- Most recent research direction from the human researcher team: none received this round.
  The standing direction is `senpai/program.md`: move the official frontier, submit
  autonomously, and treat a promoted rival row as an interrupt.

## Where the board stands

The frontier moved twice in ninety minutes and both moves were built out of mechanisms
this campaign already owns.

```
08b67f12 jungjipdo 3.69071883 21:02:29Z   tight launch grid + probe fraction 0.15
ed608e64 jungjipdo 3.68172016 19:47:14Z   tight launch grid alone
02742bf0 scarlet   3.52686512 19:06:44Z   probe fraction 0.15 alone
623e77af morganmcg1 3.52085227 16:05:35Z  ours
```

`ed608e64` published the same active-group launch arithmetic that thorfinn measured
independently and shipped in `572b2cc4`. `08b67f12` then added one constant we have
held unshipped since E133. We are behind on the board and ahead on inventory: we hold
both of those mechanisms plus a third, `pb6`, that nobody on the board has.

Our submission `572b2cc4` is validating. One slot, one in-flight submission.

## Current research focus

1. **Compose, do not iterate.** The three mechanisms are in disjoint files and disjoint
   line ranges, so composition is mechanical rather than scientific:
   `Qwen35.swift` Route B launch geometry (thorfinn), `Qwen36MTPBlockSession.swift`
   depth price (edward), and one `Double` literal at `Qwen35.swift:4880` (askeladd).
   The composed tree is the next official submission the moment the slot frees.

2. **The pass-boundary depth price is ready.** `pb6` reads +2.4683 % held out on
   edward's own measured round-cost curve, with the FM1 audit refuted in its favour:
   the rounds it suppresses accept at 0.5057 against a population 0.7796, so the
   replayed price is a lower bound. A same-binary twelve-leg ABBA reads +2.2467 % on
   the decode basis as a falsifier only. Merged.

3. **The probe fraction is worth three times what we recorded.** Two independent ranked
   receipts on two different bases now price the same one-constant change. Read on the
   candidate leg alone, which is the only leg our source can move, it is +0.33 %, not
   the +0.0992 % we recorded from the raw ratio. The offline screen says recall stays
   exactly 1.0 down to p=0.10, so the true argmax lies below where anyone has sampled.

4. **The width-6 cost cliff is still the largest unexplained quantity.** E137 closed the
   attribution question: QMV carries 0.7858 of the step and no single shape carries it,
   all seven step 24.9 % to 47.5 %. Alphonse now sweeps the (IPG, RPS) plan surface to
   find out whether any plan flattens it.

## Potential next research directions

- **The column-count ladder.** Widths 8 and 9 still launch 2 and 3 columns under
  `onePass67`. Their one-pass plans were closed on a register basis under a wide launch
  where both arms launched `M` columns, so that closure measured only half the trade.
  The ladder also separates the three surviving models of the launch-geometry law:
  flat per round, linear in columns, or logarithmic in the column ratio.

- **Prefill is not scored, and that is now established.** The ranked scorer runs in mode
  `qwen-mtp-paired-decode-only` and the workflow states that the scoring path never
  reads `prefill_seconds_per_token`. Seed processing is a sealed sub-interval published
  for observability. Round-side gains therefore land undiluted, and prefill is not a
  target.

- **C2 precision-island quantization.** Reopened and unowned, +0.38 % to +0.45 %.

- **P4, the Gated DeltaNet S=2 mid-state write.** Gates about 151 MB per round on
  rejection, unowned, 0.2 % to 0.6 %.

- **Cleanup.** After the composition ships, prune the E120 arm flags, the dead Route B
  table paths, and the E128 price arms so the winning behaviour is the only path.

## Standing methodological state

Rule 115 is the newest and the most expensive lesson of the round: convert a per-round
absolute mechanism to a ranked percentage through absolute microseconds per round, never
through a local percentage. The local benchfixture round is about 3.6 times longer than
the F83-weighted ranked round, so the measured ranked-to-local ratio for launch geometry
is about 2.3, not the 0.95 the old transfer table predicted. Prices are read on the
candidate leg, whose pair-difference null is 0.067 % (Rule 112), never on the published
median. Every offline replay standing in for a Metal kernel must reproduce that kernel's
arithmetic and prove it with a bit comparison (Rule 113). Every same-binary A/B must
witness its arm from the run's own trace (Rule 114).
