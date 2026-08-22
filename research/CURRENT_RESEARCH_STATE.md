# SENPAI Research State

- 2026-08-23 00:15 UTC
- Most recent human research direction: none received this generation. The standing direction is `senpai/program.md`: maximise the official decode score on the `qwen3.8-27b-mtp-v1` track, submit autonomously, and never stop at synthesis.

## Where the campaign stands

Board crown `1760479a` (scarletbright) at `3.70355222`. Our best measured receipt is `572b2cc4` at `3.66218564`, rejected only because the crown moved while it validated. `e003a86d` has been validating since 22:50:23Z and carries the tight launch grid, the `pb6` boundary tier, probe fraction 0.15 and the shipped QMV plan table; the forecast is `3.7747`.

The next archive, already in preparation, adds the width-2 launch shrink and forecasts `3.78295`, about `+2.14 %` over the live crown.

`pb6`, our boundary depth-price tier worth `+2.4683 %` held out, exists nowhere else on the board. It is the campaign's only unique mechanism and it is the reason our forecast clears the crown.

## Current research focus

**1. Launch geometry.** Finding 200 established that the cost of a launched threadgroup column on the ranked M5 is logarithmic and one-parameter: `1296.8 * ln(columns)` microseconds per round, replicated to 2.4 percent across two independent solvers' receipts. No such coefficient has ever been published for any Apple GPU. The width-2 shrink is the last no-op column available; the open question is whether the law prices launched columns or only no-op columns, which a column-count ladder settles.

**2. The verify readout.** Every decode round streams a 715 MB `lm_head` and reduces it to two token ids and two values per row. This is 4.5 to 7.5 percent of the round and it has exactly one scored consumer. A screened readout with an on-device certificate and a dense fallback is bit-identical by construction. This is the largest single lever now in flight.

**3. The beagle prize.** The published median is an exact identity in two prompts, beagle and essays. Beagle is the laggard, with the worst per-step acceptance of the deep prompts at 0.9341 against 0.96 elsewhere. Raising beagle to equal essays moves the median `+4.35 %`; raising it without limit gives a hard ceiling of `+5.07 %`. This is the largest identified prize in the campaign and only one experiment currently attacks it.

**4. Acceptance, not speed.** 60.4 percent of the vocabulary, 149,990 token ids, can never be proposed by the draft head at any depth for any prompt, because the compact draft vocabulary is a contiguous prefix of 98,304 ids plus 26 control tokens. Prior art in the opposite direction implies a large exchange coefficient.

**5. Measurement discipline.** Finding 207 remeasured the run-level noise floor at large N on the serial leg, which runs identical code on every board row. The diff-of-two floor is `0.15 %` on the eight-prompt mean and `0.30 %` on the medpair, not the `0.067 %` that Rule 112 claimed. Rule 120 now requires an eight-prompt sign test alongside any sub-0.3 percent medpair claim.

## Open threads and next directions

- **The probe fraction turns over.** Finding 206: a rival's clean one-line receipt shows probe 0.12 is `+1.05 %` SLOWER on the medpair at pinned round count and pinned draft length, while 0.15 is confirmed faster by two receipts. Fewer streamed bytes cannot make a round slower, so there is an unidentified mechanism in the probe path. The optimum may lie above 0.15, not below it. A same-binary absolute-time ladder at `{0.25, 0.20, 0.17, 0.15, 0.12, 0.10}` settles it and may expose a fixable cliff.
- **Re-fit every scheduler constant to the shipped launch table.** Rule 117. The `pb6` tier 1.45 was fitted to a pre-tight cost curve where the width-6 step was 16,241 microseconds; under the table that ships it is 16,903, which is 4.1 percent steeper. A one-constant, zero-byte, zero-risk gain.
- **Precision islands to affine-4 group-64.** About 31 MB of dense bf16 in the proposal head, reopened, unowned, priced `+0.38 %` to `+0.45 %`.
- **Gated DeltaNet mid-state write on rejection.** Gate 151 MB per round, unowned, `0.2 %` to `0.6 %`.
- **Per-position head-side confidence.** The sole named reopening signal from E99, unowned, speculative at `+0.3 %` to `+0.8 %`.
- **The head-history fold warm gap.** Widths 1 to 9 are flushed but only 2 are warmed; must clear Rule 110 before it is worth anything.
- **Composition risk.** Our tiered one-pass QMV entry points reverse sign under a tight launch grid, confirmed by two rival receipts. Any mechanism whose value was measured under the wide grid must be re-measured.

## What would change the plan

A promoted receipt for `e003a86d` that confirms `pb6` on the ranked host would make the boundary-tier family the campaign's main line and justify a full re-fit of the depth-price curve. A refuted `pb6` would make the launch-geometry and readout families the whole campaign. Either way the next assignments are already runnable and no student is idle.
