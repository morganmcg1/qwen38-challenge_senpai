# E204 — round-end seam overlap

**Result: Not useful. The mechanism works, and it relocates the recovered wait
instead of removing it from the leg.**

Assignment: PR #202, `qwen-alphonse/e204-round-end-seam-overlap`, revision
`e204-r0`, base `136ab8a480e877f67d28264887162067a4d321f2`.

## Question

Does round-end overlap — submitting the next round's whole head chain before
the accept and rollback bookkeeping completes — recover at least 0.25 % of the
round?

## Answer

At the round anchors, yes: it recovers 2532.5 ± 792.5 µs, which is 1.74 % of a
145.2 ms round, at 3.20σ. Across the whole timed leg, no: the ON arm is
0.0975 % **slower** in absolute `mtp_seconds_per_token`, and both ABBA pair
deltas agree on that sign. The recovered device idle is refilled by an equal
wait that lands outside the round anchors but inside the timed leg.

## Experiment identity

| field | value |
|---|---|
| base | `136ab8a480e877f67d28264887162067a4d321f2` |
| candidate (stripped, shipped form) | `09da15ff` |
| armed research build | `809ac668`, worker `11c6793590a426208b24c72490fa3f5ce8d62bcfd5e4bf35c308270cd7e8ce1c` |
| stage-1 gate worker | `ae8730a52917c6be42746ff59019e1bc8b61d41f60582b6eb0992c001d99eeac` |
| stage-2 worker | `bfd7916b417b629a5dd35a373bbfee89ed3e00bf72376dcfd4e8c32fc60ea73e` |
| host | `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 48 GiB |
| head | declared run tree, `head_provenance_sha256` `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` |
| token window | 512 decode tokens, depth 8, 78 rounds |
| harness | `local` throughout. No result here is an official or ranked score. |

## Mechanism

The base already prefetches head step 1 before the bookkeeping (E165, E193).
Steps 2..d were still built at the start of the next round. E204 builds and
submits them in the same round-end seam.

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift` is the only submitted file
touched:

- `PendingHeadChain` holds the prefetched draft ids and their head hidden rows.
- `prefetchChain(after:offeredDepth:)` runs from `prefetchHeadStep`, chains
  `predicted - 1` further head steps on the same head cache, and `asyncEval`s
  the last id. It adds no synchronisation; the commit phase never waits for the
  device.
- `predictedNextDraftCount(offeredDepth:)` asks the shipped schedule what the
  next round would propose, under a flag that suppresses the schedule's trace
  side effect. The count is a prediction, never a commitment.
- The next round takes `min(chain.count, draftCount - 1)` steps, chains any
  shortfall from the last prefetched hidden, and lets the existing
  `runHeadUpkeep` trim any surplus speculative rows.
- `undoHeadPrefetch` drops the chain handle; the rows themselves are already
  discarded by the pre-flush trim.
- The accept-outcome and full-accept-streak updates move above the prefetch so
  the look-ahead reads exactly what the next round will read. Both are pure
  host arithmetic over the accept walk.

## Stage 0 — is there a seam at all?

`research/e204-s0-seam.json`, leg `e204-s0-seam`, all rounds, medians:

| window | span µs | GPU-idle µs | % of round |
|---|---|---|---|
| protocol seam | 2369.0 | 1052.4 | 0.725 % |
| overlappable | 3327.2 | 1574.7 | 1.084 % |

Largest coherent idle slice 687.0 µs, 0.473 % of the round, which cleared the
0.25 % stop rule by 1.9×. Host chain-build cost 886.4 µs, 0.610 % of the round,
was flagged at the time as work that **relocates** rather than disappears.

Full-acceptance rounds (53) put the slice at `readout`; rejection rounds (16)
put a longer 903.3 µs slice inside `commit`.

## Step-7 falsification gate — 11 / 11 PASS

`research/e204-s1-gate.json`, legs `e204-s1-off` and `e204-s1-on`.

513 of 513 positions compared: **0 id mismatches and 0 hexfloat value
mismatches**. Accept ledgers identical. The positive control detected a planted
perturbation, so the comparison was able to fail. `all_tokens_matched=true` and
`residual_divergence_count=0` on both legs. OFF consumed 0 chain steps; ON
consumed 415 with 6 overshoot, 1.4 %.

Expected draft length 6.358974358974359 and accept rate 0.8770161290322581 are
identical across every leg in this experiment.

## Stage 2 — round-anchored attribution, PASS

`research/e204-s2-decision.json`, complementary alternating legs `e204-s2-pA`
(`alt2`) and `e204-s2-pB` (`alt2i`), 70 usable pairs, statistic `round_us` per
RULE 394.

| statistic | value |
|---|---|
| delta (off − on), mean | 2532.5 µs/round |
| standard error | 792.5 µs |
| significance | 3.20σ |
| 2σ interval | [947.5, 4117.5] µs |
| median | 1338.5 µs |

RULE 396 validity evidence: 0 transition-round disagreements of 77 on both
legs, symmetric assignment, entry temperatures 36.73 °C and 46.85 °C for a
10.12 °C spread, and the ON arm faster in both legs so the session offsets
cancel rather than load onto one arm.

By outcome: full acceptance n=54, 2346.5 ± 1006.1 µs; rejection n=16,
3160.2 ± 732.2 µs.

This stage remains valid **as attribution**. The stall does leave the round.

## The 512-token confirmation and the base twin

`research/out/e204-confirm` and `research/out/e204-base-confirm`, both gated at
the real 40 °C gate, untraced, 512 tokens.

| leg | serial s/token | MTP s/token | ratio |
|---|---|---|---|
| base `136ab8a4` | 0.07320833 | 0.02850291 | 2.568451 |
| candidate `09da15ff` | 0.07303509 | 0.02852488 | 2.560400 |

The candidate confirmation passed every correctness gate: `all_tokens_matched`
true, `residual_divergence_count` 0, row-ledger closure 574 of 574 target rows
on the MTP leg and 512 of 512 on the serial leg, full fixed 512-token window
with no EOS shortening, `public_drift_tripwire_passed` true.

It also showed the candidate 0.077 % **slower** than the base in absolute
terms. That is the opposite sign to Stage 2, and it is what triggered Stage 3.

## Stage 3 — whole-leg ABBA, the promotion-grade statistic

RULE 397: for a mechanism that can move work across the round anchors,
`round_us` is attribution only and the whole-leg absolute
`mtp_seconds_per_token` decides promotion.

`research/e204-s3-decision.json`, `research/e204_leg_abba.sh`. One armed build
serves all four legs, so no rebuild sits between the arms. Order OFF, ON, ON,
OFF. Every leg is a full gated untraced 512-token `--local-submit` at the real
40 °C gate with the declared head named explicitly.

| leg | arm | serial s/token | MTP s/token | ratio |
|---|---|---|---|---|
| `e204-abba-1-off` | OFF | 0.07321563 | 0.02852072 | 2.567103 |
| `e204-abba-2-on` | ON | 0.07325678 | 0.02853294 | 2.567446 |
| `e204-abba-3-on` | ON | 0.07308230 | 0.02856749 | 2.558233 |
| `e204-abba-4-off` | OFF | 0.07312428 | 0.02852408 | 2.563598 |

| statistic | value |
|---|---|
| mean OFF | 0.0285224043764174 |
| mean ON | 0.02855021576397121 |
| delta (ON − OFF) | +2.7811387553811073e-05 s/token |
| delta as a fraction of OFF | **+0.0975 %**, ON slower |
| pair deltas (ON − OFF) | +1.221848651766777e-05 and +4.3404288589954376e-05 |
| pair deltas agree in sign | yes, both ON slower |
| OFF pair spread | 0.0118 % |
| ON pair spread | 0.1210 % |

**Serial K=1 internal control:** OFF mean 0.07316995330620557, ON mean
0.07316954096313566, difference −0.00056 %. At depth 0 there is no drafting
round, so the chain prefetch never executes. A null this tight shows the ABBA
order cancelled the session's thermal ramp and that the arm does not touch the
serial path.

**Fidelity across all four legs, identical:** expected draft length
6.358974358974359, accept rate 0.8770161290322581, 78 rounds, 574 of 574 rows,
`all_tokens_matched` true, `residual_divergence_count` 0.

Predeclared band: `|delta| < 0.1 %` with both pair deltas agreeing in sign →
**relocation, no end-to-end value**. `promote=false`.

## Why the two statistics disagree — the relocation, measured

`round_us` spans a round's host anchors. It does not span the seed processing
or the time between the anchors. `research/e204-leg-budget.json` decomposes the
traced MTP legs:

| leg part | `e204-s1-off` | share of leg |
|---|---|---|
| seed processing | 1024.2 ms | 7.0 % |
| inside the round anchors | 10656.1 ms | 72.9 % |
| **outside the round anchors** | **2936.7 ms** | **20.1 %** |
| whole leg | 14617.0 ms | 100 % |

That leaves **37.7 ms per round of leg time outside the round anchors**, which
is 26 % of a 145 ms round and about 15× the 2.5 ms the mechanism moves. There
is ample room for the recovered wait to reappear there, and the measurements
say it does:

- round-anchored improvement, Stage 2: 2532.5 µs × 78 rounds ≈ **197.5 ms per
  leg**, 1.35 % of the leg;
- whole-leg change, Stage 3: **+14.2 ms per leg**, ON slower.

A 197.5 ms round-anchored gain produces no leg gain at all. The device does not
run out of work earlier; it waits in a different place. This also explains the
Stage-1 surprise where the measured round gain, about 1.16 ms per round, was
larger than the 484 µs GPU-idle slice Stage 0 predicted: what moved was a
blocking `asyncEval` submission stall against the MLX command-buffer throttle,
and a throttle stall that is pushed out of the round is not removed from the
leg — the next submission simply waits instead.

**Honest limitation.** The 197.5 ms round-anchored figure and the flat leg time
come from different sessions. A single traced ABBA session would measure the
per-leg `round_us` total and the leg time together and close that gap in one
measurement. The advisor's Stage-3 ruling arrived after this session had
started and set a one-session budget with exhaustive bands, so I did not run
it.

## Correctness

Every leg in this experiment reports `all_tokens_matched=true` and
`residual_divergence_count=0` with full row-ledger closure. The step-7 gate
compared actual hexfloat values at 513 of 513 positions with a working positive
control and found no divergence. The emitted trajectory, the expected draft
length and the accept rate are identical between the arms, so the mechanism
changes the submission schedule only.

## Test gate

`research/e204-swift-test.json`. `swift test --force-resolved-versions` reports
754 tests in 79 suites with 41 issues across 10 distinct names, which equals the
pre-existing failure floor recorded for this base line in
`research/e145-result.md`, with identical per-test issue counts. No new failure
and no fixed failure.

## Scope

Submitted surface touched: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
only. Everything else is research-only under `research/`.

Per RULE 198 the shipped candidate `09da15ff` carries no arm: the
`DARKBLOOM_E204_CHAIN_ARM` environment variable, the `ChainPrefetchArm` enum,
the per-round gate and the arm witness are removed, and `pf_chain` is a build
witness like `pf`. Per the E92 precedent the E90 GPU interval ledger, which
stage 0 used, is removed from the candidate as well, so
`Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift` is byte-identical to
the base again.

## Recommendation

Do not merge. Close the round-end seam axis.

The seam is real and the mechanism fills it, but the leg has 20 % of its time
outside the round anchors and the recovered wait moves there. Any future
mechanism in this class needs a whole-leg endpoint before it is priced.

## Suggested follow-ups, not implemented

1. **Measure the 37.7 ms per round outside the round anchors.** It is 20 % of
   the MTP leg and is currently unattributed. If any part of it is candidate
   work rather than trusted parent work, it is a far larger target than the
   seam and no one has looked at it.
2. **One traced gated ABBA session** that records the per-leg `round_us` total
   and the leg time together. It would turn the relocation inference in this
   report into a single-session direct measurement.
3. **Ask whether the MLX command-buffer throttle stall can be removed rather
   than moved.** Stage 1 showed roughly 2.2 ms per round of blocking
   `asyncEval` submission time. Moving it does nothing. Reducing the number or
   size of submitted command buffers per round might reduce it.
