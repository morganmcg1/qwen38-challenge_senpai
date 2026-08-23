# E145 — the width cost curve, measured

The depth price that ships in `makeUniformDepthPrice` and every tuned variant
of it rests on a **replayed** width cost curve. Nobody had decoded at each
width and timed it. A six-point disagreement between two research lines hung
on that curve. This experiment measures it live.

Base `2cd0d459c651de53cc4ccebb160a19fb00ed87c4`. Host `Mac16,11`, Apple M4
Pro, 48 GiB. Worker
`9fe1bf8076dd2832fcbc011bb81a20f8e6af49b5352c744eeb823db2552ee307`, the same
binary in every session reported here. All timing is local, never ranked.

## Headline

1. **The replayed curve has the wrong shape.** Level transfers well
   (`k = 2.1034`), but the worst per-width shape residual is **+13.73 % at
   width 7**. The disagreement is real and it is not a level offset.
2. **There are two cliffs, not one.** The measured curve steps hard at 5->6
   *and* at 6->7, then 7->8 is nearly free. The replayed curve has a single
   cliff at 5->6 and calls 6->7 almost free. The two curves disagree about
   *which* boundary is expensive.
3. **Both of those facts change almost nothing about the policy.** Re-fitting
   the price plane on the measured curve moves the objective by **+0.070 pp**,
   and paying for the newly discovered second cliff adds a further **+0.049
   pp**. The price surface is a plateau, not a peak.
4. **The regime hypothesis is dead.** On `beagle_a` the `pb6` arm is
   **2.57 % faster**, not slower. Both preregistered directional intervals
   miss low.
5. **A mean-draft-length model of this system is wrong, and I can now show it
   on measured data.** The advisor's Error 156 reproduces locally: predicting
   from the curve evaluated at the mean draft length is off by up to
   **10.3 %**, while predicting from the full width histogram is off by
   **0.37 % at worst**. The Jensen gap changes sign between fixtures.

Taken together: the curve everybody was using was wrong, and correcting it
does not buy a meaningful gain, because the objective is flat in the
directions the correction moves. That is a negative result about the *lever*,
established on a positive result about the *measurement*.

## What was measured

`MLX_E145_PIN_DEPTH` pins the offered draft count inside the width envelope,
after `guard cap > 0`, so a pinned session decodes at one width and stays
exactly on the serial token trajectory. Pinning is not a fidelity change: every
timed leg below matched its 512-token golden with zero divergence.

### R2 — the curve, widths 2 to 8

14 timed legs plus 7 traced legs, one session, palindrome-ordered arms, every
leg through the real 40 C cool gate. Entry temperatures 37.41 C to 39.50 C,
interior spread 1.15 C. Pin leakage never exceeded one round per leg and was
always the final budget-clamped round.

| width | measured us | repeat spread % | replayed us | ratio | shape residual % | rounds |
|-------|-------------|-----------------|-------------|-------|------------------|--------|
| 2 | 70,429.4 | 0.129 | 34,619 | 2.0344 | -3.28 | 272 |
| 3 | 75,197.3 | 0.054 | 38,065 | 1.9755 | -6.08 | 193 |
| 4 | 83,188.0 | 0.096 | 41,511 | 2.0040 | -4.73 | 155 |
| 5 | 95,301.5 | 0.001 | 44,958 | 2.1198 | +0.78 | 136 |
| 6 | 124,436.0 | 1.433 | 61,199 | 2.0333 | -3.33 | 118 |
| 7 | 150,297.5 | 0.004 | 62,825 | 2.3923 | **+13.73** | 105 |
| 8 | 154,243.9 | 0.157 | 70,315 | 2.1936 | +4.29 | 100 |

Repeat spread is the palindrome spread on the per-leg median basis. It is
below 0.2 % everywhere except width 6.

### R2b — the width-1 anchor

R2 pins widths 2 to 8. Width 1 needs its own session because pin 0 produces a
non-drafting round, which is a verify at width 1. R2b runs four legs on
`beagle_a` in the palindrome `0 4 4 0`: two width-1 legs and two width-5 legs.

The width-5 legs are the tie point. They let the width-1 measurement be
rescaled onto R2's thermal level instead of importing R2b's own level into the
curve. They landed at 95,471.3 and 95,865.4 us against R2's width-5 point of
95,820.3 us, so the session scale is 1.0016 and the correction is almost
nothing. All four legs matched with zero divergence through the real gate, and
the two width-5 legs were bit-identical in rounds, mean draft length and
acceptance, which is a clean determinism check on the pin itself.

### The boundary shape

Each step expressed as a multiple of its own curve's mean step, which removes
the level factor entirely:

```
width step   2->3  3->4  4->5  5->6  6->7  7->8
replayed     0.62  0.62  0.62  2.98  0.26  1.32
measured     0.46  0.71  1.00  2.49  2.32  0.28
```

The replayed curve says: flat, flat, flat, **cliff**, free, moderate.
The measured curve says: cheap, cheap, moderate, **cliff, cliff**, free.

This is the core finding. The replayed model treats 6->7 as the cheapest step
on the curve. It is actually the second most expensive.

## The closure test — the strongest result

If the measured curve is right, then the curve plus a leg's own width
histogram should predict that leg's round cost. Tested on four held-out legs
that were never used to build the curve:

| fixture | arm | predicted us | observed us | error % | interp-at-mean error % | Jensen gap us |
|---------|-----|--------------|-------------|---------|------------------------|---------------|
| beagle_a | ship | 108,768.1 | 108,793.2 | **-0.023** | -5.554 | +6,018 |
| beagle_a | pb6 | 105,963.7 | 105,947.5 | **+0.015** | -5.464 | +5,806 |
| benchfixture | ship | 143,578.5 | 143,288.6 | **+0.202** | +6.037 | -8,360 |
| benchfixture | pb6 | 132,666.7 | 133,158.9 | **-0.370** | +10.333 | -14,252 |

The histogram model closes to better than 0.4 % on every leg. The
mean-draft-length model is off by 5 to 10 % and **changes sign** between
fixtures, so it cannot be repaired with a constant correction. This is
Advisor Error 156 reproduced on a measured curve.

The curve also predicts the *difference* between arms, which is a harder test
because the level cancels:

| fixture | predicted arm effect % | observed % | error pp |
|---------|------------------------|------------|----------|
| beagle_a | -2.5783 | -2.5934 | +0.0151 |
| benchfixture | -7.5999 | -7.5014 | -0.0985 |

## R1 — the regime hypothesis

### `beagle_a`: the hypothesis fails, decisively and in the wrong direction

| metric | ship | pb6 | delta |
|--------|------|-----|-------|
| seconds per token | 0.033137977 | 0.032286135 | **-2.5706 %** |
| round us (blocks only) | 108,962.9 | 106,137.1 | -2.5934 % |
| rounds | 119 | 118 | |
| mean draft length | 4.2437 | 4.1525 | |
| accepted draft rate | 0.7782 | 0.8041 | |

Serial reference 0.073842463 s/token, 512 rounds.

Preregistered intervals: advisor +1.0 to +3.5 % (MISS low), student +0.5 to
+2.5 % (MISS low). The mass-shift prereg of 5 to 20 pp HIT at 8.99 pp, so the
mechanism fired as designed; it simply helped instead of hurting.

Width mass moved out of 6 entirely: `w6 8.40 -> 0.00`, with the mass landing on
4 and 5 (`w4 10.92 -> 16.10`, `w5 18.49 -> 24.58`).

### R1b: the arm effect replicates in a second session

Rule 119 asks for this, because the session is the unit that drifts. R1b runs
the same `beagle_a` comparison in its own session, palindrome
`ship pb6 pb6 ship`.

| quantity | R1 | R1b | gap |
|----------|-----|-----|-----|
| pb6 against ship, seconds per token | -2.5706 % | -2.4961 % | +0.0745 pp |
| pb6 against ship, blocks only | -3.4119 % | -3.2489 % | +0.1630 pp |

Same sign, and the gap is inside the 0.122 pp noise floor on the primary
basis. The comparison is only meaningful because the **work signatures match
exactly**: both sessions produced the same round counts (119 and 118), the
same mean draft lengths (4.2437 and 4.1525) and the same acceptance rates
(0.7782 and 0.8041). Decoding is deterministic for a fixed fixture, budget,
build and arm, so identical signatures prove the two sessions ran the same
work and only the timing moved. `r1b_replicate` checks that before it
compares any timing.

### Rule 128 warm telemetry cannot be captured on the timed path

R1b was also meant to supply the `wired-zh` and `warm` residency telemetry
that Rule 128 asks for in every leg. **It failed, and the failure is
structural.** Two separate faults were found, and the second one cannot be
fixed from inside this assignment.

1. The probe read the wrong path. `e145_leg` passed its own slot directory,
   but `e128_session.sh` writes one level deeper under the fixture name, so
   `stderr.log` never existed where the probe looked. Fixed.
2. The telemetry is absent even at the corrected path. Both lines are written
   by the **worker** to its own stderr
   (`Qwen36MTPBlockSession.swift:271` and `:354`). The `mtp-timed` parent
   calls `runtimeWorkerOptions` **without** `forwardsWorkerStderr`, so
   `QwenRuntimeWorker.swift:2046` installs a swallowing emitter and the stream
   reaches no file at all. `MLX_DFLASH_TRACE_CACHE_SEAM` cannot rescue it,
   because that variable is read only inside the DFlash subcommand
   (`MLXFastCLI/main.swift:1409`), which `mtp-timed` never enters. The source
   states this directly at `Qwen36MTPBlockSession.swift:800-806`.

Capturing it needs a change in the trusted parent, which is outside the E145
scope. So **every leg in this experiment reports
`warm_telemetry_present=false`**, and that is recorded rather than hidden: a
leg with no telemetry cannot support an arm attribution claim on its own.

What can be said without the telemetry, from the source and from the data:

- `wired_gate_fired` is driven by `residencySizingGateFires`, and the source
  notes that a 48 GiB host fails the 96 GiB guard. On this host the residency
  sizing path is inert for **both** arms, so it cannot separate them.
- The warm phase warms shapes up to `maxDepth`, and both arms run
  `E128_DEPTH=8`, so the warmed shape set is identical by construction.
- The arm changes only `depthPriceArm`, a price table consumed at
  draft-decision time, which runs after the warm phase.
- Empirically, R1 and R1b reproduce each other to 0.0745 pp with identical
  work signatures, which is what an unchanged warm state predicts.

That is a source-level argument plus a replication, not the direct
measurement Rule 128 asks for. I am not claiming it is equivalent.

### `benchfixture`: the positive control passes

`pb6` is -2.0086 % on seconds per token against E134 item 5b's -2.2467 %, a
gap of 0.238 pp. The stop rule was to abort if this control missed by more
than 1 pp. It did not, so the `beagle_a` result stands as a real effect and
not a harness fault.

## R0 — is the detector sound?

The F219 identity reproduces to +/-0.005 %. Against F219 at `1760479a` the
mean gap is **+6.946 %** and systematic; against its own anchor `623e77af` it
is +1.878 % and not systematic.

The F4 null control (`684821ed` against `1760479a`) reproduces exactly, giving
a **median-eligible noise floor of 0.122 pp**. Mass-weighting repairs the two
residuals that had broken the shortcut: republic +7.393 -> **-0.348 pp**,
plutarch -21.122 -> **-0.277 pp**.

Required metrics: `e145_predicted_beagle_shift_us = 514.19`,
`e145_f218_bound_consistent = True`.

## R0b — the width-1 anchor and the prefill correction

Three independent width-1 anchors agree to 0.75 %: replayed 31,173.2 us,
F219 plutarch at `1760479a` 31,090.4 us, `c91581eb` 31,323.8 us.

A separate finding fell out of this. **Ranked `mtp_seconds_per_token_mean` is
seed-inclusive.** The fixture's `proposed_scoring.prefill_component`,
`program.md`, and the local identity
`decode_seconds = seed_prefill + sum(blocks)` (residual 0.0027 %) all agree.
So prefill enters any fitted width curve as a **spurious slope**: at the
estimated ranked prefill `P ~ 1.710 s`, the prefill slope is 2,772 us per
width, which is **80.4 % of the replayed 3,446 us slope**. At `P = 2.126 s`
the implied work curve is flat over widths 1 to 5. Any width curve fitted to
ranked per-token numbers without removing prefill is mostly fitting prefill.

Local-to-ranked level factor: 2.098 at width 1, 2.122 at width 5. The caveat
the advisor asked for: the depth-0 row is verify-only, so the width-1 point is
not a drafting round and the factor at width 1 is not measuring the same work
as the factor at width 5.

## F217 and F218

On the measured curve, with realised shift 0.0953:

| quantity | measured cost-only | same-p | replayed-rescaled cost-only | replayed-rescaled same-p |
|----------|--------------------|--------|-----------------------------|--------------------------|
| shift % | +1.4057 | +0.5343 | +1.2387 | +0.3688 |
| shift us | **+699.2** | +265.7 | +616.1 | +183.4 |
| within F218 bound | True | True | True | True |

`brackets_ranked_measurement` is False for both, and unpriced width mass is
0.0000. Ranked measured shift +0.2391 %.

`e145_predicted_beagle_shift_us_measured = 699.2`,
`e145_f218_bound_consistent_measured = True`.

## R3 — does the shape difference change the price?

6 seeds, 200 windows, measured width-1 anchor, zero GPU. Every curve is
normalised by its own width-1 point, so the level factor cannot leak into the
comparison.

Worst shape disagreement **+13.539 % at width 7**. Agrees-within-5 %:
**False**, so the R3 trigger written into the assignment fired correctly.

Both curves place the cliff at the same boundary, but they ask for different
prices there:

| curve | cliff step us | mean other step us | implied tier | argmax boundary |
|-------|---------------|--------------------|--------------|-----------------|
| replayed | 16,566.7 | 3,978.5 | 4.1641 | 4 |
| measured | 28,436.9 | 8,987.4 | **3.1641** | 4 |

The measured curve asks for a **lower** tier. That is counter-intuitive until
you look at the denominator: the measured cliff is larger in absolute terms,
but its non-cliff steps are larger too, so the cliff stands out *less* against
its own curve. The whole disagreement reduces to that one ratio.

Cells, replayed median gain in percent against the shipped arm:

| cell | cost=rep price=rep | cost=meas price=rep | cost=meas price=meas |
|------|--------------------|---------------------|----------------------|
| A_ship | +0.0000 | +0.0000 | +0.0000 |
| B_rankedprice | -3.1234 | -1.4478 | +0.1338 |
| C_flatlook | +0.0000 | +0.0000 | +0.0000 |
| D_curvelook | -3.1234 | -1.4478 | +0.1338 |
| E_pb6 | +2.4880 | +2.8201 | +2.8201 |
| F_pb6look | +1.9715 | +1.9132 | +1.9132 |
| G_pb68 | +2.4880 | +2.8201 | +2.8201 |
| H_pb68look | +1.9715 | +1.9132 | +1.9132 |

Two controls worth recording:

1. At `cost=replayed price=replayed`, `E_pb6` scores **+2.4880** against
   E140's published **+2.489**. The replay path reproduces the earlier
   published number, so these results do not come from a changed replayer.
2. `G_pb68` and `H_pb68look` are exact duplicates of `E_pb6` and `F_pb6look`
   because `best_tier8 = 1.0` on this base. They are degenerate, not
   independent evidence.

The attachment gate passed: 12 legs, 1,494 attached rounds, zero accept
mismatches, zero margin mismatches, zero unmatched.

### The width-1 anchor is not a free parameter

Three independent anchors agree to **0.75 %**: measured 65,778.9 us,
extrapolated 65,614.1 us, local serial control 66,017.0 us. Every R3 and R4
cell is identical between the measured and the extrapolated anchor, so the
anchor is a robustness control here and not a result.

## What could not be identified

Two quantities are **not identified** by this data, and I am withholding
coefficients rather than reporting fitted numbers that mean nothing:

1. **Repair cost per replay.** Fitted from shipped legs it straddles zero:
   beagle ship -498 us/replay, beagle pb6 +251, benchfixture ship +820,
   benchfixture pb6 -1,477. The between-width least-squares fit is degenerate
   because width and replay share are collinear at `corr = 0.9843`.
2. **Ranked prefill from two curves.** The two-curve identification failed. The
   R0b level-factor estimate `P ~ 1.71 s` remains the only estimate.

Both have explicit guards in the analysis code so a later run cannot silently
consume the unidentified coefficients.

A third limitation is structural, not statistical. `Qwen36MTPBlockSession`
declares `declaredRows: draftCount + 1` and `QwenRuntimeMTPDriver` forces
`declaredRows == rowsPerRound(...)`, so `M == d + 1` always. **A width ladder
cannot separate per-row verify cost from proposal-head cost.** Every number in
the curve above is the sum of the two.

## Reproduction

```bash
research/e145_r1_session.sh 512 beagle_a
research/e145_r1_session.sh 512 benchfixture
research/e145_r1b_session.sh 512 beagle_a
research/e145_r2_session.sh 512 beagle_a
research/e145_r2b_session.sh 512 beagle_a
python3 research/e145_read.py
python3 research/e145_r0.py
python3 research/e145_anchors.py
python3 research/e145_r1.py
python3 research/e145_curve.py
python3 research/e145_r3.py --seeds 6 --windows 200
research/e145_r4_both.sh 6 200
python3 research/e145_wandb_log.py
```

Timed sessions hold the GPU and take the real cool gate; the analysis steps
use no GPU. `e145_r4_both.sh` runs the measured and replayed cost models and
then the cross-evaluation, which needs both blobs.

## Test gate

`swift test --force-resolved-versions` reports 774 tests in 73 suites with
**41 issues across 10 distinct test names**, which is exactly the recorded
pre-existing floor on this base. E145 adds no regression. The 10 names are
`contestantDocsCommandBlocksKeepTheDependencyGraphFrozen`,
`participantDocsExposeDefaultCLIInstallDirectory`,
`qwen36ConfigContractDigestMatchesTheReferenceManifest`,
`startupMemoryPolicyKeepsRanked128GiBProfile`,
`submissionStaticReviewPromptCoversMeasurementStructureExploitation`,
`theCheckedInDeclarationSelectsThePinnedHead`,
`theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`,
`theQwenMTPTrackIsArmedOnQwen38`,
`theSeededCalibrationExpectationMatchesItsRecordedProvenance` and
`theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax`.

## Scope and budget

Submitted surface touched: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
only, and only outside the reserved line ranges. Everything else is
research-only under `research/` and `Tests/MLXFastTests/E145*`.

`senpai/validate-assignment-scope.sh` passes against
`BASE_SHA=2cd0d459c651de53cc4ccebb160a19fb00ed87c4`.
`senpai/check-editable-budget.sh` reports source 2,627,254 of 3,000,000 bytes
and growth 172,419 of 262,144. `senpai/verify-ranked-score-boundary.sh`
passes.

**No Yukon submission was made.** The assignment forbids it, and the
submission slot is held by another candidate.
