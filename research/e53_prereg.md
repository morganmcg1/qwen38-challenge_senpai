# E53 pre-registration (Part 1), written before any share is computed

Base `45b7c6a4`, branch `qwen-edward/scored-width-mixture-and-policy-map`,
2026-08-19T15:1xZ. Committed before the simulator was run. The PR comment
carrying the same text failed with GitHub HTTP 403 at 15:12Z, so this file is
the timestamped record.

## Source finding 1 - the `M = k + 1` mapping in the brief is refuted by our own scheduler

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift` at base `45b7c6a4`:

- `costModelDepth(offeredDepth:)` (line 738) is a greedy marginal-cost walk, not
  a draw from the previous round's accepted count. It extends while
  `reach > h * (1 + expected) / (1 + depth * h)` with
  `h = headStepCostRatio = 0.18` (line 668).
- `reach` is a product of per-position acceptance EMAs (line 635, seeded
  `0.85 * 0.98^i`, `acceptEMAAlpha = 0.15`, optimism transfer capped at 0.95 in
  `recordAcceptOutcome`, line 813).
- The walk is capped by `widthCap = fullAcceptStreak >= 2 ? 8 : 5`
  (lines 700, 707, 735, 744).
- `draftPolicy` returns that depth (lines 194 and 910), and
  `effective_mean_draft_len` counts drafts PROPOSED (ledger item 153, named
  error MT1). So verify width `M = drafts + 1`, and `drafts` is the schedule's
  output, not the accepted count.

First-moment separation, beagle: the brief's mapping predicts mean
`M = 1 + A/R = 1 + 405/107 = 4.785`; the published `effective_mean_draft_len`
gives mean `M = 1 + 4.5327 = 5.533`.

## Model to be fitted

A source-faithful simulator of `costModelDepth` plus `recordAcceptOutcome` with
one free parameter (per-position acceptance probability `q`). The truncated
geometric of the brief is reported alongside as the refuted null.

Per prompt the simulator must reproduce jointly:

1. `effective_mean_draft_len` (4.5327 beagle, 4.7677 medicine; item 102
   fingerprint, our tree);
2. the implied acceptance ratio `A/D` (0.8351 beagle, 0.8750 medicine; ledger
   item 153);
3. `non_drafting_round_count = 0` on both scored prompts, with a mechanism that
   can also reach plutarch's mode of 449.

Constraint 1 pins `q`. Constraints 2 and 3 are predictions. Free check: the
simulator predicts `R = 512 / (1 + accepted per round)` and so selects among
beagle's `(107,485,405)`, `(214,970,298)`, `(321,1455,191)` without ledger 153's
added monotonicity assumption.

## Pre-registered predictions

1. `f{7,8}` at the corrected marginal weights 0.483694 / 0.516306 comes out
   ABOVE askeladd's 9.158 %. Confidence 55 %. Same direction as the advisor,
   lower confidence, different mechanism: I expect the movement to come from the
   weight correction tilting toward medicine, not from a large true `f{7,8}`.
2. The width mixture is bimodal at the two caps, with `M = 7` and `M = 8`
   locally rare: `rho(7) + rho(8) < rho(9)` in round shares on both scored
   prompts. Confidence 75 %.
3. `f{4,5,6}` stays the largest of the three blocks, above 50 %. Confidence
   70 %. Weak evidence, because it agrees with the number under test.
4. The truncated geometric fails at least one of constraints 1-3 by more than
   its own spread. Confidence 90 %.

## Standing commitments

- Interval on every share. Within-cohort spread (same `head_provenance_sha256`)
  is separated from across-cohort spread (different trees, different
  schedules); only the first is a replication of our own tree.
- Name the tree behind every constant.
- Board pull is live from the API (717 rows at 2026-08-19T15:04Z). The 635-row
  cached dump `.mlxfast-private/ranked-telemetry.json` is absent from this
  checkout. Healthy-row counts are recomputed, not inherited.
