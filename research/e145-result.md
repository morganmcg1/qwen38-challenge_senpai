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
3. **The replayed curve is wrong in shape, wrong by a whole extra cliff, and
   the entire consequence for the policy is 0.3305 pp** — locally resolvable,
   ranked-unresolvable. The cross-evaluation is the decisive test: let each
   curve pick its own best price cell, then price both cells under the
   measured costs. The replayed curve moves `h` from 0.18 to 0.16 and believes
   it gains +0.1367 pp; that cell in truth **loses 0.0993 pp**. The measured
   curve instead moves the tier from 1.45 to 1.60 and gains **+0.2313 pp**.
   The regret from having used the wrong curve is **0.3305 pp**, which is
   **2.713x** the F4 median-eligible noise floor of 0.1218 pp and **0.19x**
   the smallest nuisance a single ranked receipt carries, one 879 us state
   step at 1.78 pp. Local instruments can see this; the ranked board cannot.
4. **The regime hypothesis is dead.** On `beagle_a` the `pb6` arm is
   **2.57 % faster**, not slower. Both preregistered directional intervals
   miss low. Advisor Error 151 resolves in `pb6`'s favour on the local
   instrument.
5. **A mean-draft-length model of this system is wrong, and I can now show it
   on measured data.** The advisor's Error 156 reproduces locally: predicting
   from the curve evaluated at the mean draft length is off by up to
   **10.3 %**, while predicting from the full width histogram is off by
   **0.37 % at worst**. The Jensen gap changes sign between fixtures.
6. **The ranked receipt cannot decide `pb6` either, and the campaign base
   ships it.** R5 reads the two ranked rows directly. After removing the one
   per-drafting-round state draw the crown family measured at 879 us, the
   published median pair moves by a fraction of a percent, and the sign of
   that move depends on a state-step count that three independent readings
   place between 0.94 and 1.45. See "R5-b" for the exact interval and verdict.
7. **The ranked cliff is not where the local cliff is, and it is not
   identified either.** Solving the ranked per-width round cost from the board
   by monotone non-negative least squares puts the ranked cliff at **7->8**,
   with 4->5 the runner-up. R2 measured the local cliff at 5->6. Board-row
   noise leaves the 7->8 answer at 100 %, but per-seed **design** jitter splits
   it 50/50 between 7->8 and 4->5, and design noise beats board noise at all
   eight widths. So the ranked cliff has moved, and where it moved to is not
   identified. This is the concrete form of the M5 transfer risk that F6 named
   through the `g16s`/`g17s` register-spill tables.
8. **RULE 138, an exact and permanent pruning rule.** A draft width is
   admissible if and only if its measured cost per token is a new running
   minimum over every smaller width. Admissibility is a property of the cost
   curve alone. It holds for every monotone nonnegative acceptance model,
   because the best case puts the whole extra width into accepted tokens, so
   the reachable token ratio between widths `M' > M` is capped at exactly
   `M'/M`. The bound is attained, not conservative. **Never argue for a width
   from acceptance evidence without clearing this test first.** On the
   measured curve the admissible set is `{1,2,3,4,5}`. **Width 7 can never be
   optimal under any acceptance model**, which closes a question that has been
   open since E128.
9. **The measured cost table is an enabler, not a lever. Verdict:
   `enabler, standalone +0.1338 %, unlocks +6.2170 pp`.** Both halves of that
   sentence must always travel together, because either half alone misleads:
   - the shipped walk spends **26.75 % of its rounds at an inadmissible width,
     16.37 % of them at width 8**; and
   - **the argmax that removes every one of them is worth only +0.1338 %**,
     because width 8 misses width 5 by just **0.43 %** on cost per token, so
     the rounds being removed were barely wrong.
10. **The finding is that the shipped rule is already unimodal-optimal, and
    only the price table was wrong.** Under the shipped flat price the full
    argmax reproduces the shipped walk **round for round**, with identical
    width histograms. Under the measured price the argmax and the shipped
    threshold rule return **identical** results in all three acceptance states
    R7-2 tested. The decision rule was never the defect. The cost table it
    reads was, and correcting it is what makes the next number visible at all.
11. **+6.2170 pp of headroom, and R7-2 says every bit of it is per-round
    discrimination.** On the same curve and the same rule, an oracle that
    knows this round's realised acceptance scores **+6.3508 %** against the
    EMA's **+0.1338 %**. An oracle that knows the true per-position **marginal
    acceptance distribution** perfectly scores **-0.3806 %**, that is
    **-0.5144 pp**, worse than the shipped estimator. So calibrating the
    marginal estimator is dead even at perfect accuracy, and the open axis is
    a **per-round discriminator**. The one-line mechanism:
    **the shipped EMA is not a bad estimator of the marginal; it is a good
    accidental estimator of the round.** The true marginal vector is
    optimistic about any single round, so a policy reading it never declines
    to draft (width-1 share **0.00 %**) and pays on every bad round. The EMA's
    pessimism is accidentally protective.
12. **The two campaign "oracle" numbers reconcile.
    `e145_r7_oracle_reconciled = 1.0`.** E140's -4.5296 and R7's +8.9390 are
    different arms sharing one overloaded word, and the 13.4539 pp between
    them is a name collision, not an error. E140's number is one policy, not
    an upper bound, so it could not have closed an axis.
    **Advisor Error 167 is confirmed, and narrowed to the distributional
    half.** **RULE 140**: an acceptance-estimator arm must declare which
    information state it manipulates, the per-position marginal distribution
    or the per-round realised outcome. "Oracle" alone is not a declaration.
13. **A weak per-round predictor is already worth a lot.** R7-3 walks a
    predictor from the shipped estimator to the per-round truth. Ten per cent
    of the way collects **22.64 %** of the gap
    (`e145_r7_lam_for_half_the_gap = 0.4` buys half of it). Additive noise of
    **sigma = 0.10** on the true indicator still collects **81.29 %**. Capture
    falls below a tenth only at
    **`e145_r7_noise_sigma_below_floor = 0.3`**, and the arm turns actively
    harmful by **sigma = 0.50**, where it loses 6.67 pp against the shipped
    rule. That ladder is a pre-registered acceptance specification: any future
    per-round discriminator can be priced from its error rate before a GPU leg
    is spent.
14. **The campaign is already running one per-round discriminator, nobody
    called it that, and it is worth +0.5668 pp.** The shipped depth-0/1 margin
    clamp `p = min(p, 1/(1 + exp(-(top1 - top2)/scale)))` reads `pendingTop2`,
    which is a property of **this round**, not of the marginal distribution.
    It is the only per-round signal anywhere in the scheduler, it is applied
    at 2 of 8 depths, it is one-sided so it can only lower `p`, and its two
    scale constants have no recorded derivation. R7-4 prices it where it runs,
    not under perfect information, and it is **4.24x the entire measured
    cost-table correction** and **4.65x the F4 noise floor**. Three sub-results
    matter and two of them refute a pre-registered prior:
    - **Removal costs +0.5668 pp**, so the clamp earns its place.
    - **Extending it to all eight depths costs -3.9910 pp** at scale 2.0 and
      **-5.2929 pp** at scale 3.0. The restriction to 2 of 8 depths is
      load-bearing, not neglect.
    - **It does not work by skipping rounds.** Removing it moves the width-1
      share the wrong way, 19.51 % to 20.10 %, while mean depth moves 3.1269
      to 3.3830. **The clamp does not skip rounds, it shortens them.**

    Retuning the two constants is worth only **+0.0588 pp**, below the
    0.1218 pp noise floor, so the surface has real structure and no useful
    peak. What is unexploited is not the clamp's reach but its **quality**: a
    single scalar margin through a sigmoid with an underived constant already
    pays this much, which is direct evidence that the +6.2170 pp per-round
    axis is reachable from code that already ships.
15. **E128's whole 14-arm depth-policy table is unpriced.** Every one of those
    arms was scored on the replayed curve, whose shape is wrong by up to
    13.73 %. R7-1 re-priced two of its cells and both moved far:
    `rankedprice` from -3.1234 % to **+0.1338 %**, `oracle` from +8.9390 % to
    **+6.3508 %**. E145 does not re-run the table, but it leaves a checked
    injection point that does it with no new plumbing. See
    "How to re-price any E128 arm on the measured curve".
16. **The residency-slack direction is closed by arithmetic, and then a leg
    found something else.** Slack placement priced at E130's own measured
    marginal rate is 1.009 us/round, and the KV capacity walk is 5.755
    us/round. Both are two to three orders of magnitude below the 879 us/round
    state step they were proposed to explain. R6-1 then measured wired against
    unwired residency directly, twelve legs in a counterbalanced palindrome:
    **wiring makes the candidate 1.3203 % faster on seconds per token**, which
    is 1700x larger than the placement arithmetic and of the **opposite sign**
    to the proposed mechanism. It is the resident weights, not the slack. The
    ranked two-state behaviour does not reproduce locally with wiring on.

Taken together: the curve everybody was using was wrong, and correcting it
buys 0.33 pp of direct value, which is real on this bench and invisible on the
board. That is a negative result about the *lever*, established on a positive
result about the *measurement*. The larger result is what the corrected table
made visible. Under the wrong table the depth-policy direction looked closed
and the acceptance axis looked closed with it. Under the measured table the
action set prunes exactly, the decision rule turns out to be already optimal,
and a **+6.2170 pp** gap opens between the shipped acceptance estimate and a
per-round oracle. R7-2 then shows that gap is not a calibration problem,
R7-3 shows it does not need a good predictor to start paying, and R7-4 shows
the scheduler already contains one crude instance of the right mechanism.

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

## R4 — the cross-evaluation, and what the wrong curve actually cost

R3 asked whether the two curves price the *same* cells differently. R4 asks
the decision-relevant question instead: **let each curve choose its own best
cell, then settle the bill under the measured costs.** 6 seeds, 200 windows,
zero GPU, one grid over the price plane `plane_price(h, tier)`, which
generalises `boundary_price`'s hard-wired `0.18` to a free `h`.

| cell chosen by | cell | objective under measured costs | upper-slot spread | worst slot |
|----------------|------|-------------------------------|-------------------|------------|
| uniform | h=0.18 tier=1.00 | +0.0000 | 0.000 | essays |
| shipped | h=0.18 tier=1.45 | +2.2469 | 3.7886 | botany |
| replayed curve | **h=0.16** tier=1.45 | +2.1477 | 3.5586 | botany |
| measured curve | h=0.18 **tier=1.60** | **+2.4782** | 3.9910 | botany |

Read the third and fourth rows together. The two curves do not merely disagree
about the size of the win: **they move different knobs.** The replayed curve
lowers `h`; the measured curve raises the tier and leaves `h` alone.
`same_cell = false`.

The three numbers that matter:

- `replayed_believed_gain_pp = +0.1367` — what the replayed curve thinks its
  own cell wins.
- `replayed_cell_true_gain_pp = -0.0993` — what that cell actually does. The
  replayed curve's recommendation is **worse than shipping unchanged**.
- `measured_best_over_shipped_pp = +0.2313` — what the measured curve wins.

Regret, the price of having used the wrong curve, is
`+2.4782 - 2.1477 = 0.3305 pp`.

Scale it twice, because the two instruments answer differently:

- Against the **local** F4 null-control floor of 0.1218 pp for a
  median-eligible prompt, 0.3305 pp is **2.713x the floor**. This bench can
  resolve it.
- Against the **ranked** instrument, the smallest nuisance one receipt carries
  is a single 879 us per-drafting-round state draw, worth 1.78 pp at shipped
  round counts. 0.3305 pp is **0.19 steps**. No single ranked receipt can
  resolve it.

That asymmetry, not the regret itself, is the transferable finding: the width
cost curve is a quantity that must be settled locally, because the board will
never see it.

An earlier one-seed preview of this table reported the regret as *below* the
noise floor. That preview was wrong. The six-seed run above supersedes it, and
the direction of the correction is against my earlier claim.

### Appendix — the second-cliff price

`pb67` adds a second free tier for the 6->7 boundary that R2 discovered. Run
on the replayed curve's own chosen cell (`h = 0.16`), the nested control is
exact: `tier7 = 1.00` at `tier6 = 1.45` reproduces the plane cell at +1.800.

| tier6 | tier7 | objective |
|-------|-------|-----------|
| 1.45 | 1.00 | +1.800 |
| 1.50 | 1.10 | **+1.818** |

Paying separately for the newly discovered second cliff is worth **+0.019 pp**
over the best single boundary. That is one sixth of the local noise floor. The
second cliff is real in the *cost curve* and worthless in the *price policy*,
so it gets an appendix and no further legs.

## R5 — can the ranked instrument decide anything here?

R4 says the local bench resolves 0.33 pp and the board does not. R5 turns that
around and asks what the board *can* decide, using the only two ranked receipts
that differ by the depth-price arm alone: `572b2cc4` (ship) and `e003a86d`
(pb6). No GPU. Every number below is `harness=ranked` unless it is explicitly
a replay.

### R5-0 — the local seed prefill

F5 asked for this. It needed no instrumentation: `QwenRuntimeMTP` already
writes `seed_prefill_seconds` into every leg report.

Over the 40 E145 leg reports the mean is **4.005141 s**, median 4.006380 s,
spread 0.613 %. The ranked seed prefill on `684821ed` is **0.526485 s**, so
the local-to-ranked prefill level factor is **7.6073**.

This retires two things at once. It supplies the number F5 asked for, and it
confirms Advisor Error 160 from the other side: the ranked prefill is a
published field, it is 0.5265 s, and the 1.71 s figure that R0b derived from a
level argument was an artifact of that argument. R0b's `P ~ 1.71 s` is
withdrawn.

### R5-a — the closed form, and where it breaks

Before touching the two arm rows I checked whether the ranked receipt can be
read as a round-count identity at all. For each prompt I recomputed the round
count two ways: from the published `effective_mean_draft_len` of the arm row
itself, and from the pinned serial row. The two agree to better than 0.4 of a
round on every prompt.

| prompt | dlen A | dlen B | R model | R pinned | R ratio | d accept | R at pb6 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| beagle | 4.3818 | 4.2069 | 110.00 | 110.00 | 1.0324 | +0.00944 | -0.873 |
| medicine | 5.2556 | 5.2222 | 90.01 | 90.01 | 1.0053 | +0.01858 | -1.686 |
| essays | 5.0870 | 5.2955 | 92.04 | 92.04 | 0.9675 | +0.02364 | -2.131 |
| botany | 6.1481 | 5.8118 | 81.04 | 81.04 | 1.0483 | +0.00294 | -0.283 |
| republic | 4.9892 | 4.7938 | 93.00 | 93.00 | 1.0331 | +0.03268 | -2.856 |
| drama | 2.2976 | 2.2948 | 252.01 | 252.01 | 1.0006 | +0.02301 | -2.534 |
| travel | 2.6557 | 2.6343 | 211.97 | 212.33 | 1.0047 | +0.00122 | -0.134 |
| plutarch | 0.1540 | 2.6995 | 487.02 | 486.76 | 0.5536 | -0.01610 | +2.343 |

`r5a_worst_model_minus_pinned_rounds = 0.3627`. The identity holds, so the
ranked rows can be read as round counts and the arm effect can be split into
"fewer rounds" and "cheaper rounds".

Two prompts invert: on `essays` and `plutarch` the arm that decodes in fewer
rounds is the slower one. That inversion does **not** survive the measured
acceptance shift (`r5a_inversion_survives_accept_shift = False`), so it is not
an acceptance story. The worst per-prompt width-share gap between the two
ranked rows is **0.9163**, which is a real transfer failure: the two receipts
are not running the same width mixture, so no single local curve reproduces
both.

### R5-b — the state-step lattice, and the verdict on pb6

The crown family measured one per-drafting-round state draw at **879.0 us**,
sd 54.3 over 3 readings. Every ranked comparison between `572b2cc4` and
`e003a86d` carries an unknown integer number of those draws. R5-b enumerates
the lattice: what does the published median pair become after removing `s`
state steps per drafting round, for `s` in 0 to 3?

Per-prompt effect in the decode frame, in percent:

| prompt | s=0 | s=1 | s=2 | s=3 |
| --- | --- | --- | --- | --- |
| beagle | +4.0467 | +2.0509 | +0.0551 | -1.9407 |
| medicine | +0.8408 | -0.9343 | -2.7093 | -4.4844 |
| essays | -3.2232 | -4.9413 | -6.6594 | -8.3776 |
| botany | +1.4656 | -0.2138 | -1.8932 | -3.5726 |
| republic | +2.6833 | +0.8042 | -1.0749 | -2.9539 |
| drama | +2.6799 | +0.1567 | -2.3666 | -4.8899 |
| travel | +3.5936 | +1.1090 | -1.3756 | -3.8602 |
| plutarch | -48.7519 | -50.3414 | -51.9309 | -53.5205 |
| **mean of 7** | **+1.7267** | **-0.2812** | **-2.2891** | **-4.2970** |
| **mean of 5** | **+1.1626** | **-0.6469** | **-2.4563** | **-4.2658** |
| **total of 5** | **+1.0410** | **-0.5825** | **-2.2061** | **-3.8296** |
| **median pair** | **-2.3800** | **-0.7021** | **+1.0346** | **+2.8331** |

The sign of the answer moves with `s`. That is the whole point of the rung:
the published pair cannot be read without committing to a state-step count.

Three independent readings of `s`:

1. **The `drama` control.** `drama` changes its draft length by only -0.122 %
   between the two rows, so almost all of its +2.6799 % is state steps. One
   step is +2.5233 %, so `drama` reads **1.0621 steps** with a residual of
   +0.1567 pp.
2. **The `travel` control** reads 1.4464 steps, but its draft length moves
   -0.806 % so it is contaminated. Held out.
3. **A free fit** over the seven decodable prompts gives **0.9395 steps**,
   which is 825.85 us per drafting round, **-0.98 sd** from the 879 us crown
   measurement. Rms residual 2.1158 pp; the per-prompt residuals are beagle
   +2.1716, botany -0.1123, drama +0.3092, essays -4.8374, medicine -0.8270,
   republic +0.9179, travel +1.2593.

`r5b_control_mean_steps = 1.1493`. Taking the controls and the free fit
together the interval is `s` in **[0.8815, 1.1277]**, over which the corrected
score moves from **-0.9039 %** to **-0.4837 %**. Both ends are negative, so
`r5b_verdict = loss_confirmed`: on the ranked receipt, with the state step
removed, `pb6` is a loss.

That verdict is **not robust to adding `travel`**. Including the contaminated
control raises the top of the interval to 1.5357 steps, where the corrected
score is **+0.2207 %** and the sign flips
(`r5b_verdict_robust_to_travel = False`). The honest statement is therefore:
the ranked receipt says `pb6` loses under every defensible reading of `s`, and
one indefensible reading makes it win. That is an instrument limit, not a
result about `pb6`.

A separate closure check ran the local replayer's own prediction of the arm
effect against each prompt's observed `s=1` value. The replayer over-predicts
`pb6`'s benefit by about two state steps: `r5b_closure_best_state_steps = 3`
with rms 1.5191 pp, against rms 4.36 pp at `s=1`.

### R5-c — solving the ranked width curve from the board

R5-c asks the harder question directly: can the ranked per-width round cost be
recovered from the board rows themselves? Each receipt gives one linear
equation -- the prefill-free round cost is the width histogram times the
unknown per-width cost vector -- so three receipts over eight prompts give up
to 24 equations in 9 unknowns. I solve them by monotone non-negative least
squares.

Prefill-free ranked cost per round, in microseconds, for bar / ship / pb6 /
pb6-state-corrected:

| prompt | bar | ship | pb6 | pb6 corrected |
| --- | --- | --- | --- | --- |
| beagle | 44992.8 | 45468.3 | 45825.4 | 44815.2 |
| medicine | 49450.3 | 49779.9 | 49936.0 | 48925.8 |
| essays | 48975.4 | 49496.7 | 49511.7 | 48501.5 |
| botany | 54559.1 | 54867.0 | 53107.3 | 52097.0 |
| republic | 47806.3 | 48327.2 | 48033.3 | 47023.0 |
| drama | 34139.7 | 34857.3 | 35769.3 | 34759.1 |
| travel | 35177.5 | 35545.4 | 36648.8 | 35638.6 |
| plutarch | 30744.9 | 30614.6 | 28339.8 | 27329.6 |

Our tree runs **0.8957 %** above the bar at the same level.

| solve | cliff | margin us | step us | rms us | rel rms | runner-up |
| --- | --- | --- | --- | --- | --- | --- |
| bar_only | 7 | 4897.2 | 18490.1 | 1051.4 | 2.4320 % | 3 (13592.8) |
| ours_ship_only | 7 | 2023.6 | 15602.1 | 936.7 | 2.1474 % | 3 (13578.5) |
| ours_ship_plus_pb6_raw | 4 | 4938.6 | 16131.0 | 1278.0 | 2.9374 % | 7 (11192.4) |
| **ours_ship_plus_pb6_state_corrected** | **7** | 2377.9 | 13581.6 | 1344.1 | 3.1255 % | 4 (11203.7) |

The headline solve is the state-corrected one, because the raw `pb6` row still
carries the R5-b state draw. It puts the ranked cliff at **7->8**. R2 measured
the local cliff at **5->6**. `r5c_ranked_vs_local_cliff_moved = 1.0`.

Two noise models disagree about how much to believe that:

- **Board-row noise** (400 draws at the published row cv of 0.00132) leaves
  the cliff at 7 in **100 %** of draws.
- **Per-seed design jitter** -- resampling which width histogram each receipt
  presents -- splits the answer **50/50 between 7->8 and 4->5**.

Design noise is larger than board noise at all eight widths (per-width sd,
board vs design: 29.9/154.8, 29.9/154.8, 182.4/640.5, 420.2/2063.7,
97.9/713.1, 98.2/1238.0, 98.2/1238.0, 78.0/375.1). So the ranked cliff has
moved away from the local one, and its new location is **not identified**.
Width 9 is unreachable in the design and is reported as
`r5c_unidentified_widths = [9]`.

This is the quantitative form of the transfer risk F6 raised with the
`g16s`/`g17s` register-pressure tables: those tables predict spill onset at
width 6 on `g16s`, which matches the local 5->6 cliff, and at width 8 on
`g17s`, which is the ranked part. The board solve is consistent with that
reading but does not establish it.

### R5-d — which arm the base actually ships

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1139` reads
`return DepthPriceArm(rawValue: requested) ?? .pb6`, so
`r5d_base_ships_pb6 = 1.0`. The campaign base already ships the arm that R5-b
finds is a ranked loss under every defensible state-step count. I did not
change that default; the assignment reserves those lines.

### R5 gates

The attachment gate covers 12 legs and 1494 attached rounds with 0 acceptance
mismatches and 0 margin mismatches. `design_matrix_curve_independence_max_diff`
is exactly 0, so the design matrix does not smuggle the cost curve into its own
solve.

## R6 — the residency direction, closed twice

F8 proposed that the wired-residency slack explains the 879 us/round state
step. F9 refuted the mechanism at source and asked for the arithmetic and one
clean measured arm instead. R6 supplies both.

### R6-0 — the decode state, priced

Every byte is read from the pinned checkpoint config on disk and cross-checked
against the allocation shapes in the scored session, then against the source
literals. No GPU.

| quantity | bytes | note |
| --- | --- | --- |
| full-attention KV per token | 65,536 | 16 layers x 2 x 4 kv heads x 256 dim x 2 B |
| full-attention KV at 1024 tokens | 67,108,864 | 64.0000 MiB |
| wired residency slack | 67,108,864 | equal to the KV figure by coincidence |
| GDN convolution state | 2,949,120 | 48 x `[1,3,10240]` bf16 |
| GDN recurrent state | 150,994,944 | 48 x `[1,48,128,128]` **float32** |
| persistent decode state | 221,052,928 | 210.8125 MiB, **3.2939x** slack |
| per-round GDN snapshot | 153,944,064 | 146.8125 MiB |
| round peak decode state | 374,996,992 | 357.625 MiB, **5.5879x** slack |

The slack is **30.36 %** of the persistent decode state, so it cannot hold the
decode state and the exact equality with the KV figure is arithmetic
coincidence, exactly as F9 said.

The KV capacity walk is fully determined: `KVCacheSimple.step = 256`, the
512-token seed opens capacity at exactly 512, so the first decode round already
resets, and capacity walks 512 -> 768 -> 1024. That is 2 proven growth events
per leg, 3 in the worst case, with a 128 MiB peak transient.

Six residency-contract facts were machine-verified true against the allocator
source: insert is greedy first fit, insert has no eviction, repromotion happens
only on resize, free recycles without erase, malloc inserts only fresh device
buffers, and the session never ends the ticket.

### R6-0 kill 1 — placement cannot pay for the step

E130 measured the marginal rate of slack size directly:
**3.0600e-05 %/MiB** at the point estimate, 9.4050e-05 %/MiB at the 95 %
bound. The whole 64 MiB of slack is therefore worth **0.001958 %** (point) or
**0.006019 %** (95 %) of the published median, that is **1.009** or **3.101**
us/round.

The state step is 879.0 us/round, which is **1.7061 %** of the published
median. Placement falls short by **871x** at the point estimate and **283x**
at the 95 % bound. `e145_r6_placement_can_explain_step = False`.

### R6-0 kill 2 — KV reallocation cannot pay for it either

The two proven resets copy 33,554,432 B and 50,331,648 B, so 83,886,080 B are
copied and **167,772,160 B** are moved counting the read and the write. At the
host's 265 GB/s that is **633.1 us per leg**, which is 0.0128 % of the ranked
beagle leg and **5.755 us/round**.

That is **153x** short of the state step and **90x** short of the 515.2 us/round
that one percent of the published median costs. The bandwidth sensitivity band
does not rescue it: 50 GB/s gives 30.504 us/round, 100 GB/s 15.252, 265 GB/s
5.755, 400 GB/s 3.813.

So `step=1280` -- raising `KVCacheSimple.step` to remove both resets -- has a
**maximum** saving of 5.755 us/round and was **not implemented**.
`e145_r6_residency_direction_closed = True`.

### R6-1 — the wired arm, measured

The arithmetic closed the *slack* direction. It said nothing about whether
wired residency itself matters, so R6-1 measured that directly.

Twelve timed legs in the counterbalanced palindrome `W U U W  W U U W
W U U W`, after one discarded unwired warmup leg that exists only to cut the
entry-temperature spread. Arm `pb6`, pin `none`, 512 tokens, the real 40 C
gate on every leg, one binary
`8c295074cefd167d3c76d5ef16d53d0475cd7a9329c03e7157a71ca01df9a807`, session
commit `d1ecd431`. Entry temperature spread across all twelve legs is
**0.757 C**.

Every leg reports `matched=true`, `divergence=0`, `rounds=118`,
`mean_draft_len=4.1525423728813555` and
`accepted_draft_rate=0.80408163265306121`. The work signature is identical
leg to leg, so the two arms differ in residency and in nothing else.

| pos | res | spt | round us | block us med | entry C | exit C |
|---|---|---|---|---|---|---|
| 1 | W | 0.031874 | 104501.3 | 93869.1 | 39.10 | 55.34 |
| 2 | U | 0.032272 | 106106.5 | 94847.9 | 39.38 | 55.53 |
| 3 | U | 0.032220 | 105929.0 | 94631.0 | 39.46 | 55.70 |
| 4 | W | 0.031920 | 104700.3 | 93902.0 | 39.78 | 55.21 |
| 5 | W | 0.031868 | 104480.4 | 93936.0 | 39.70 | 55.76 |
| 6 | U | 0.032335 | 106417.9 | 95218.1 | 39.67 | 55.18 |
| 7 | U | 0.032372 | 106506.3 | 94600.9 | 39.85 | 56.29 |
| 8 | W | 0.031874 | 104494.5 | 93976.0 | 39.30 | 56.13 |
| 9 | W | 0.031868 | 104463.8 | 93819.0 | 39.38 | 55.88 |
| 10 | U | 0.032330 | 106348.4 | 95006.9 | 39.25 | 55.47 |
| 11 | U | 0.032308 | 106249.1 | 95253.1 | 39.72 | 55.47 |
| 12 | W | 0.031875 | 104494.6 | 93931.0 | 39.22 | 55.67 |

Wired mean spt **0.03187965**, sd **0.0621 %**, range 0.1621 %, round cost
104,522.5 us. Unwired mean spt **0.03230619**, sd **0.1660 %**, range
0.4723 %, round cost 106,259.5 us.

**Wiring makes the candidate 1.3203 % faster on seconds per token and 1.6347 %
faster on round cost.** The three drift-free adjacent blocks agree:
(1,2,3,4) -1416.9 us and -1.3365 %, (5,6,7,8) -1974.6 us and -1.8548 %,
(9,10,11,12) -1819.5 us and -1.7117 %. That gives
`e145_r6_local_step_us = -1737.0 +/- 166.2` us per round.

Two things follow, and they point in opposite directions.

**The mechanism is not the one that was proposed.** The effect is about 1700x
larger than R6-0's slack-placement arithmetic and it has the **opposite
sign**: the proposal was that wiring costs time through slack placement, and
wiring in fact saves time. It is the 25.5 GB of wired-resident weights on a
48 GiB host, not the 64 MiB of slack.

**The ranked two-state behaviour does not reproduce here.** Against the crown
state step of 879.0 +/- 54.3 us per round, the local step is
`z = -14.96`, so `e145_r6_local_step_explains_state_step = False`. The
preregistered kill fired: the wired arm's sd is 0.0621 %, below the 0.30 %
threshold, and its widest internal gap is 0.1409 %, below the 0.80 %
threshold, so `e145_r6_kill_fired = True` and
`e145_r6_ranked_state_locally_reproducible = False`. The wired arm is a
single tight mode, not two states.

The residency witness is machine-checked rather than asserted. The probe log
under `.mlxfast-private/e128/e145/r6-probe/` carries 12 lines reading
`request=25545645176 applied=25545645176 active=25478536312 slack_mb=64
fraction=1.0 maxrec=40200896512 gate_gib=32 physmem=51539607552`, two per
wired leg, and 21 lines reading `skipped=gate gate_gib=96
physmem=51539607552`, three per unwired leg including the warmup. So the wired
legs really wired and the unwired legs really refused, at the **compiled
default**, which is the proof that the shipped gate did not move.

The transfer consequence is uncomfortable and worth stating plainly. **Every
prior local leg on this host ran unwired, and therefore about 1.63 % slower
per round than the residency state the 128 GiB ranked M5 always uses.** That
is a level effect that largely cancels in A/B contrasts, which is why the
comparisons in this report stand, but it is a real local-to-ranked caveat for
any future absolute number measured on a 48 GiB host.

### R6-2 and R6-3 — cancelled

R6-2 was the slack-size ladder. It had already run as a measured null: 12 legs
over arms s64/s512/s1024/s2048 with residual sd 0.0650 %, giving s64->s512
+0.0179 +/- 0.0531, s512->s1024 +0.0437 +/- 0.0532 and s1024->s2048
-0.0009 +/- 0.0534, with the ladder argmax at `s64`. R6-3 depended on R6-2.
Both were cancelled on F9's instruction and neither consumed GPU time.

## R7 — which widths can ever be optimal

F9 §6 asked whether the greedy walk is "trapped behind the +30.6 % wall" at
5->6 and therefore never reaches an "M=8 plateau" that the nearly free 7->8
step seems to open. R7 answers it exactly, then measures the answer.

R7 has four rungs. R7-0 proves which widths can ever be optimal, R7-1 prices
the argmax against the shipped rule, R7-2 reconciles the two campaign
"oracles" and splits the headroom into a marginal half and a per-round half,
R7-3 measures how accurate a per-round predictor must be to collect it, and
R7-4 prices the one per-round signal the scheduler already contains.

### R7-0 — the admissibility theorem

The policy maximises expected accepted tokens per unit cost, `E(M)/C(M)`. Over
all monotone non-negative reach vectors the largest possible ratio
`E(M')/E(M)` is `M'/M`, attained at the all-accept vertex. So a wider width
`M'` can beat a narrower `M` **if and only if**

```text
C(M') / M'  <  C(M) / M
```

A width is admissible **exactly when its cost per token is a new running
minimum**. The bound is attained, not conservative, and the strict inequality
matches the shallower-on-tie rule in `walk_argmax`.

On the measured curve, with the R2 basis `us_mean_from_blocks` and the R2b
measured width-1 anchor of 65,778.9 us:

| width | cost us | cost/token us | admissible | blocked by | misses by |
| --- | --- | --- | --- | --- | --- |
| 1 | 65,778.9 | 65,778.9 | yes | | |
| 2 | 70,905.3 | 35,452.7 | yes | | |
| 3 | 76,196.5 | 25,398.8 | yes | | |
| 4 | 84,374.3 | 21,093.6 | yes | | |
| 5 | 95,820.3 | **19,164.1** | yes | | minimum |
| 6 | 124,257.1 | 20,709.5 | **no** | 5 | +8.0644 % |
| 7 | 150,803.2 | 21,543.3 | **no** | 5 | +12.4152 % |
| 8 | 153,965.4 | 19,245.7 | **no** | 5 | **+0.4259 %** |

`e145_r7_admissible_set = [1,2,3,4,5]`. **Width 7 can never be optimal under
any acceptance model** (`e145_r7_m7_ever_optimal = 0`): it is beaten by width
4 with 2.0876 % to spare, by width 5 with 11.0441 %, and by width 6 with
3.8703 %.

The "M=8 plateau" is a mirage. Width 8 is cheap *per round* relative to width
7, but its cost per token is 19,245.7 us against width 5's 19,164.1 us -- it
misses admissibility by **0.43 %**. The wall at 5->6 is not a trap; it is
correct, because nothing above width 5 ever repays itself.

That 0.43 % margin is inside the measurement, so I quantified it: 2000
Gaussian draws with a per-width 1-sigma taken from the half-range of R2's two
legs at that width (width 1 gets the mean relative spread). Widths 1 to 5 are
admissible in 100 % of draws, width 6 in 0.00 %, width 7 in 0.00 %, and width
8 in **4.30 %**. The admissible set is `{1,2,3,4,5}` in 95.70 % of draws and
`{1,2,3,4,5,8}` in 4.30 %.

**On the replayed curve the admissible set is `{1,2,3,4,5,8}`** -- exactly the
set F9 expected. So F9's expectation is a property of the replayed curve. The
measured curve prunes width 8 too, by 0.43 %, and the replayed curve is the
one that keeps it.

### R7-1 — the argmax, measured

Six seeds, 200 windows, attachment gate 12 legs / 1494 attached / 0
mismatches. Percentages are the published-median change against the shipped
policy.

| arm | median % | sd | mean depth | width-8 rounds % | inadmissible % |
| --- | --- | --- | --- | --- | --- |
| A_ship (flat price, greedy) | +0.0000 | 0.0000 | 4.3823 | 16.3692 | 26.7488 |
| C_flatlook (flat price, argmax) | +0.0000 | 0.0000 | 4.3823 | 16.3692 | 26.7488 |
| argmax_full (measured price) | **+0.1338** | 0.0537 | 3.1269 | 0.0000 | 0.0000 |
| argmax_admissible | +0.1338 | 0.0537 | 3.1269 | 0.0000 | 0.0000 |
| argmax_admissible_plus8 | +0.1338 | 0.0537 | 3.1269 | 0.0000 | 0.0000 |
| oracle_full | **+6.3508** | 0.0890 | 2.6735 | 0.0000 | 0.0000 |
| oracle_admissible | +6.3508 | 0.0890 | 2.6735 | 0.0000 | 0.0000 |
| argmax_full_on_replayed | **-3.1234** | 0.0902 | 3.4411 | 0.0000 | 0.0000 |
| oracle_full_on_replayed | **+8.9390** | 0.0736 | 4.0468 | 28.5311 | 28.5311 |

Width histograms, percent of rounds at widths 1 to 8:

| arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_ship and C_flatlook | 39.62 | 5.08 | 13.58 | 8.01 | 6.96 | 5.83 | 4.55 | 16.37 |
| argmax_* (all three) | 19.51 | 2.44 | 18.49 | 25.89 | 33.68 | 0 | 0 | 0 |
| oracle_full | 28.73 | 15.61 | 9.00 | 4.75 | 41.90 | 0 | 0 | 0 |
| argmax_full_on_replayed | 24.90 | 2.97 | 10.06 | 8.60 | 53.47 | 0 | 0 | 0 |
| oracle_full_on_replayed | 32.14 | 16.82 | 9.39 | 4.70 | 8.42 | 0 | 0 | 28.53 |

Five findings, in order of how much they change the campaign:

1. **F9 §6's premise is refuted.** The greedy walk is not trapped. Restricting
   the argmax to the admissible set changes the score by exactly nothing
   (`argmax_full`, `argmax_admissible` and `argmax_admissible_plus8` are
   identical to four decimal places) because the measured cost table already
   rejects 6, 7 and 8 on its own.
2. **H140's "+0.0000 pp from lookahead" is reproduced and explained.** With
   the shipped flat price the objective is unimodal in width, so first-break
   **is** the argmax. `e145_r7_argmax_equals_greedy_on_flat_price = True`.
   That result was never evidence that lookahead is worthless; it was evidence
   that the shipped price is flat.
3. **The shipped policy over-drafts, not under-drafts.** It spends
   **26.75 %** of rounds at widths that can never be optimal, **16.37 %** of
   them at width 8. Swapping the flat price for the measured curve removes all
   of that and gains +0.1338 %.
4. **+0.1338 % is below the +0.25 % stop rule**, so the script writes
   `e145_r7_verdict = "drop: below the +0.25 % stop rule"`. **The advisor
   overrode that verdict in F10 and the override stands.** The recorded
   verdict is `enabler, standalone +0.1338 %, unlocks +6.2170 pp`. The stop
   rule was written for a lever and this turned out to be an instrument, so it
   scored the wrong quantity. Nothing further in the *depth-price* direction
   is worth a leg; the table itself is what the next direction needs.
5. **The remaining 6.2170 pp is the acceptance state.** Same curve, same
   rule, oracle acceptance instead of the EMA: +6.3508 % against +0.1338 %.
   The lever is not the width rule and not the cost curve. R7-2 then narrows
   "the estimator" to the only half of it that carries value: **per-round
   discrimination**, not marginal calibration.

The replayed curve inflates its own oracle: `oracle_full_on_replayed` scores
+8.9390 % and spends 28.53 % of rounds at width 8. That is 2.59 pp of value
invented at a width the measured curve says never pays.

R7 cross-validates exactly against R3's independent grid. `argmax_full`
+0.1338 equals R3's `B_rankedprice` and `D_curvelook` at cost=measured,
price=measured; `argmax_full_on_replayed` -3.1234 equals R3's `D_curvelook` at
cost=replayed, price=replayed; `C_flatlook` +0.0000 equals R3's `C_flatlook`.
Two independently written simulators agree to four decimal places.

### R7-2 — the two campaign "oracles", reconciled

`e145_r7_oracle_reconciled = 1.0`. Accepted by the advisor as **Finding 244**,
with **RULE 140** below.

Two research lines published an "oracle" for the same system and got
**-4.5296 %** (E140) and **+8.9390 %** (E128, reproduced by R7-1). The gap is
**13.4539 pp**. Neither number is arithmetically wrong. The word `oracle` is
overloaded, and the two arms manipulate different information states.

| name | what the policy is handed | how often | decision rule |
| --- | --- | --- | --- |
| **estimator oracle** (E140) | the true per-position **marginal** acceptance vector `p_target` | once, per prompt | unchanged shipped rule |
| **decision oracle** (E128, R7-1) | the **realised outcome of this round** | every round | replaced by the argmin of cost per realised token |

The source is unambiguous once you look. `research/e140_oracle_state.py:66-79`
substitutes `p_target` for the EMA inside `make_walker` and leaves the rule
alone; its own summary keys are `estimator_worth_pp` and
`argmax_worth_with_perfect_estimator_pp`. `research/e128-results.md:823` says
of its arm: "The oracle arm knows, for each round, whether each draft token
will be accepted."

So E140's -4.5296 is **one policy's score**, not an upper bound on anything. It
could never have closed an axis, and Advisor Error 167 is confirmed.

I reproduced both, plus the four cells nobody had run, as one factorial:
three information states x two decision rules x two cost curves. Six seeds,
200 windows, attachment gate 12 legs / 1494 attached / 0 mismatches. Artifact
`research/e145-artifacts/r7-state.json`.

| arm | median % | sd | mean depth | inadmissible % |
| --- | --- | --- | --- | --- |
| `replayed_ema_greedy` | -3.1234 | 0.0902 | 3.4411 | 0.0000 |
| `replayed_ema_argmax` | -3.1234 | 0.0902 | 3.4411 | 0.0000 |
| `replayed_truep_greedy` | **-4.5148987193991665** | 0.0288 | 3.6608 | 0.0000 |
| `replayed_truep_argmax` | -4.5149 | 0.0288 | 3.6693 | 1.5772 |
| `replayed_clairvoyant_greedy` | +4.3391 | 0.0838 | 2.6735 | 0.0000 |
| `replayed_clairvoyant_argmax` | +7.7165 | 0.0651 | 3.4458 | 15.1309 |
| `replayed_oracle_depth` | **+8.9390** | 0.0736 | 4.0468 | 28.5311 |
| `measured_ema_greedy` | +0.1338 | 0.0537 | 3.1269 | 0.0000 |
| `measured_ema_argmax` | +0.1338226619729229 | 0.0537 | 3.1269 | 0.0000 |
| `measured_truep_greedy` | -0.3806 | 0.0536 | 3.2558 | 0.0000 |
| `measured_truep_argmax` | **-0.380622197388301** | 0.0536 | 3.2558 | 0.0000 |
| `measured_clairvoyant_greedy` | +4.9541 | 0.0777 | 2.6164 | 0.0000 |
| `measured_clairvoyant_argmax` | **+4.95412243417301** | 0.0777 | 2.6164 | 0.0000 |
| `measured_oracle_depth` | **+6.3508** | 0.0890 | 2.6735 | 0.0000 |

Two exact controls hold. `replayed_truep_greedy` is **bit-identical** to
E140's published in-sample mean `-4.5148987193991665`, and
`measured_ema_argmax` reproduces R7-1's `argmax_full` to full float precision.
E140's published headline `-4.529642090803134` is that same arm under
leave-one-prompt-out; LOPO was never the source of the disagreement.

**What each information set is actually worth, on the measured curve:**

- a perfect per-position **marginal** distribution: **-0.5144 pp**, that is
  *worse* than the shipped EMA;
- a perfect **per-round realised outcome**: **+6.2170 pp**.

`e145_r7_headroom_that_is_per_round_frac = 1.0827`. More than all of the
headroom is per-round. The distributional half is not merely closed, it is
negative.

**The mechanism.** Look at the width histograms, measured curve, argmax rule,
as a share of rounds at widths 1 to 8:

| arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `measured_ema_argmax` | 19.51 | 2.44 | 18.49 | 25.89 | 33.68 | 0 | 0 | 0 |
| `measured_truep_argmax` | **0.00** | 14.64 | 19.50 | 31.38 | 34.49 | 0 | 0 | 0 |
| `measured_clairvoyant_argmax` | 28.48 | 15.50 | 8.95 | 9.67 | 37.40 | 0 | 0 | 0 |
| `measured_oracle_depth` | 28.73 | 15.61 | 9.00 | 4.75 | 41.90 | 0 | 0 | 0 |

The true marginal vector is *optimistic* relative to what an individual round
delivers. A policy that reads it therefore **never declines to draft** — its
width-1 share is 0.00 % — and it pays the drafting cost on every bad round.
The EMA's pessimism is an accident that protects it. The per-round states go
the other way and buy their gain mostly by *skipping*: 28 to 29 % of rounds at
width 1.

That single row is the whole finding. **The shipped EMA is not a bad estimator
of the marginal; it is a good accidental estimator of the round.**

Two smaller results fall out:

- On the measured curve, greedy first-break and full argmax are **identical**
  in all three information states. On the replayed curve they diverge by 3.38
  pp under perfect information. The decision rule only looks like a lever when
  the price table is wrong.
- Given perfect information, the shipped depth-0/1 margin clamps cost
  **1.3967 pp** (`e145_r7_margin_clamp_cost_with_perfect_info_pp`). See R7-4;
  that number is a property of perfect information, not a verdict on the
  clamps.

**RULE 140.** An acceptance-estimator arm must declare **which information
state it manipulates**: the per-position marginal distribution, or the
per-round realised outcome. The word "oracle" alone is not a declaration. Two
arms that both call themselves oracles differ here by 13.4539 pp, and only one
of them bounds anything.

Scalar keys: `e145_r7_two_oracles_differ_pp = 13.453948264020784`,
`e145_r7_e140_estimator_oracle_repro_pct = -4.5148987193991665`,
`e145_r7_e140_estimator_oracle_lopo_pct = -4.529642090803134`,
`e145_r7_e128_decision_oracle_repro_pct = 8.939049544621618`,
`e145_r7_e128_decision_oracle_band = [8.5248, 9.0800]`,
`e145_r7_distributional_ceiling_measured_pct = -0.380622197388301`,
`e145_r7_distributional_headroom_pp = -0.5144448593612239`,
`e145_r7_realised_headroom_pp = 6.216971808451828`.

### R7-3 — how good must a per-round predictor be?

R7-2 says the open axis is per-round discrimination worth +4.8203 pp between
the shipped estimator and a clairvoyant one under the same greedy rule. R7-3
asks the only question that matters next: **how accurate does a real predictor
have to be before it pays?**

Two degradations of the clairvoyant state, both at the argmax rule on the
measured curve:

- **shrink**: `p_hat[i] = lam * 1[i < capability] + (1 - lam) * ema[i]`. At
  `lam = 0` this is the shipped EMA; at `lam = 1` it is clairvoyant.
- **noise**: `p_hat[i] = clip(1[i < capability] + N(0, sigma), 0, 1)`.

The ladder spans **+0.1338 % to +4.9541 %**, that is
`e145_r7_ladder_span_pp = 4.820299772200087`. "Captured" is the share of that
span.

| lambda | median % | captured | mean depth |
| --- | --- | --- | --- |
| 0.00 | +0.1338 | 0.00 % | 3.1269 |
| 0.10 | +1.2251 | 22.64 % | 3.0792 |
| 0.20 | +1.8390 | 35.37 % | 3.0465 |
| 0.30 | +2.3596 | 46.17 % | 3.0096 |
| 0.40 | +2.8573 | 56.50 % | 2.9544 |
| 0.50 | +3.2205 | 64.03 % | 2.9115 |
| 0.60 | +3.4925 | 69.68 % | 2.8777 |
| 0.70 | +3.8042 | 76.14 % | 2.8154 |
| 0.80 | +4.1112 | 82.51 % | 2.7632 |
| 0.90 | +4.5182 | 90.96 % | 2.6975 |
| 1.00 | +4.9541 | 100.00 % | 2.6164 |

| sigma | median % | captured | mean depth |
| --- | --- | --- | --- |
| 0.02 | +4.7095 | 94.93 % | 2.6043 |
| 0.05 | +4.4806 | 90.18 % | 2.5978 |
| 0.10 | +4.0524 | 81.29 % | 2.6004 |
| 0.15 | +3.5721 | 71.33 % | 2.5932 |
| 0.20 | +2.7981 | 55.27 % | 2.5633 |
| 0.30 | +0.4496 | 6.55 % | 2.4519 |
| 0.50 | -6.5325 | -138.30 % | 2.1394 |

`e145_r7_noise_sigma_below_floor = 0.3`,
`e145_r7_lam_for_half_the_gap = 0.4`,
`e145_r7_ladder_base_pct = 0.1338226619729229`,
`e145_r7_ladder_top_pct = 4.95412243417301`,
`e145_r7_capture_floor = 0.1`. Every arm on both ladders has an inadmissible
share of exactly 0.0000, because the measured price table prunes widths 6, 7
and 8 whatever the estimator says.

Three things follow.

1. **The response is concave, so early accuracy is cheap value.** The first
   10 % of the way to truth returns 22.64 % of the gap. A predictor does not
   have to be good to be worth building.
2. **The tolerance for noise is wide.** Additive noise of sigma = 0.10 on the
   true accept indicator still returns 81.29 %. Break-even is near
   sigma = 0.30.
3. **There is a real floor, and past it the mechanism inverts.** At
   sigma = 0.50 the arm loses 6.6663 pp against the shipped rule — much worse
   than doing nothing. A noisy per-round signal is not a free option.

Depth falls monotonically along both ladders, from 3.1269 to 2.6164 on lambda.
Every route to the +4.82 pp drafts *less*, not more.

**This table is the acceptance specification.** Any future per-round
discriminator can be scored against it before a single GPU leg is spent:
measure the arm's error against the realised accept indicator, read across.
Profitable at `sigma <= 0.20` for +2.80 pp, +4.05 pp at `sigma = 0.10`, break
even near `sigma = 0.30`, and a 6.67 pp loss at `sigma = 0.50`.

### R7-4 — the shipped margin clamp, priced where it actually runs

R7-2 priced the depth-0/1 margin clamp under *perfect* information and got
1.3967 pp. That number cannot decide anything about the shipped system,
because under perfect information a hedge is pure cost by construction. R7-4
prices the clamp where it runs: shipped EMA state, greedy rule, measured cost
and measured price. Same 6 seeds and 200 windows.

The clamp is

```text
p = min(p, 1 / (1 + exp(-(top1 - top2) / scale)))     scale 2.0 at depth 0,
                                                      scale 3.0 at depth 1
```

`top1 - top2` comes from `pendingTop2`. It is a property of **this round**.

To price it I lifted it out of the walker into an injectable state transform.
`clamp_control` re-applies the shipped `{0: 2.0, 1: 3.0}` through the lifted
path and must land on the untouched shipped cell exactly. It does:
`e145_r7_clamp_control_error_pp = 0.0`, exactly zero, not rounded. The
reimplementation is faithful, so every other cell in this rung is comparable.

| cell | median % | sd | width-1 % | mean depth |
| --- | --- | --- | --- | --- |
| `clamp_shipped` | +0.1338 | 0.0537 | 19.5064 | 3.1269 |
| `clamp_control` | +0.1338 | 0.0537 | 19.5064 | 3.1269 |
| `clamp_none` | **-0.4330** | 0.0954 | 20.1022 | 3.3830 |
| `clamp_all8_scale2` | **-3.8572** | 0.0895 | 19.0431 | 2.7622 |
| `clamp_all8_scale3` | **-5.1591** | 0.0699 | 18.5041 | 2.6579 |
| `clamp_shipped_argmax` | +0.1338 | 0.0537 | 19.5064 | 3.1269 |

**The advisor's prior, recorded in F11 before the run, splits.**

- *"Removal hurts."* **Confirmed.** `e145_r7_clamp_cost_at_shipped_ema_pp =
  +0.5668 pp`. Deleting the clamp costs more than half a point.
- *"Extension helps."* **Refuted, and not narrowly.** Extending the same clamp
  to all eight depths costs **-3.9910 pp** at scale 2.0 and **-5.2929 pp** at
  scale 3.0.

**The stated mechanism is also refuted, while its sign survives.** F11
predicted the clamp works by carrying the EMA's 19.51 % width-1 share toward
the clairvoyant 28.48 %, that is by making the policy decline to draft.
It does not. Removing the clamp *raises* the width-1 share to 20.10 %, the
wrong way, by 0.60 pp. What moves is mean depth, 3.1269 without to 3.3830
with it removed. **The clamp does not skip rounds. It shortens them.** It
trims depth on rounds whose top-two margin is thin, and leaves the decision to
draft at all untouched.

That also explains why extension is so costly. A clamp at depth 0 or 1 acts
on the decision to *commit* to a draft chain. The same clamp at depth 4 or 5
truncates a chain that is already running and has already paid its setup.
Mean depth collapses to 2.7622 and 2.6579 in the two extension arms. **The
shipped restriction to 2 of 8 depths is load-bearing, not neglect.**

#### The two scale constants, swept jointly

`scale` controls selectivity, and the direction is worth stating because it is
the opposite of what the name suggests. For a confident round the margin is
large and positive, so a **small** scale sends the sigmoid to 1 and the clamp
does nothing; a **large** scale sends it to 0.5 and caps every round alike.
Small scale is therefore a *sharp* per-round test and large scale is an
*indiscriminate* haircut.

Median %, `scale0` down the side and `scale1` across:

| s0 \ s1 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |
| --- | --- | --- | --- | --- | --- | --- |
| **1.0** | **+0.1926** | +0.1502 | +0.1703 | +0.1755 | +0.1441 | +0.1301 |
| **1.5** | +0.1578 | +0.1601 | +0.1731 | +0.1320 | +0.1353 | +0.0800 |
| **2.0** | +0.1448 | +0.1762 | +0.1494 | +0.1338 | +0.1019 | +0.0036 |
| **3.0** | +0.1627 | +0.1548 | +0.1093 | +0.0915 | +0.0246 | -0.0996 |
| **4.0** | +0.1762 | +0.1552 | +0.1440 | +0.0491 | -0.0352 | -0.1830 |
| **6.0** | +0.1777 | +0.1102 | +0.1271 | -0.0160 | -0.1355 | **-0.3760** |

The shipped cell `(2.0, 3.0)` scores +0.1338. The argmax is `(1.0, 1.0)` at
+0.1926, a gain of `e145_r7_clamp_grid_gain_over_shipped_pp = +0.0588 pp`.
The grid spread is `e145_r7_clamp_grid_spread_pp = 0.5686 pp`.

**The Rule 128 call, made against the pre-registered 0.50 pp threshold.**
Strictly the surface is not a plateau: `e145_r7_clamp_grid_is_plateau =
False`, because 0.5686 exceeds 0.50 by 0.0686 pp. I am reporting that
literally rather than rounding it into the answer I would prefer. But the
spread is produced entirely by the *bad* corner. The **achievable** gain from
retuning is +0.0588 pp, which is less than half the F4 median-eligible noise
floor of 0.1218 pp. **Retuning the two constants is not actionable.** The
right sentence is "the surface has real structure and no useful peak", and I
am not proposing a new pair.

The structure it does have is the interesting part. Both extremes lose about
half a point: no clamp at all costs 0.5668 pp, and the most indiscriminate
corner `(6.0, 6.0)` costs 0.5098 pp against shipped. The optimum is interior
in *selectivity*, and the best cell tested is the sharpest one. Note also that
`clamp_none` is **not** the small-scale limit of the grid: as scale falls the
clamp becomes a hard sign test on the margin, refusing depth outright when
`top1 < top2`, which is a strong discriminator rather than an absent one. The
grid argmax sits on the boundary in that direction.

#### What this rung actually establishes

**The campaign's only per-round discriminator is crude, applied at 2 of 8
depths, one-sided, and untuned, and it is already worth +0.5668 pp.** Put that
next to the other two numbers in this experiment: it is **4.24x** the entire
measured-cost-table correction of +0.1338 %, and **4.65x** the F4
median-eligible noise floor.

So F11's reading is correct in substance and needs one amendment. The clamp is
not "two-thirds disabled" in the sense that more coverage would help; the
measurement says coverage is exactly where it belongs and widening it costs 4
to 5 pp. What is unexploited is not the clamp's *reach* but its *quality*. It
is a single scalar read of one margin, thresholded through a sigmoid with an
underived constant, and it already pays. R7-3 says a per-round signal only
10 % of the way to the truth is worth +1.09 pp. This is the empirical proof
that such signals pay on this system, obtained from code that already ships.

That is the argument for arXiv 2606.30265, margin-based certified bounds: it
replaces exactly this scalar and exactly these two constants with a per-round
certificate. R7-4 says the slot is real and occupied by a placeholder.

### How to re-price any E128 arm on the measured curve

**Every arm in E128's 14-arm depth-policy table was scored on the replayed
curve. R2 shows that curve is wrong in shape by up to 13.73 %. So the whole
table is unpriced until it is re-run against the measured curve.** E145 does
not re-run it — that is a separate assignment — but the entry point is built,
tested and documented here so the re-run costs no new plumbing.

Everything below is in `research/e145_r7_state.py`.

**Step 1 — build both cost environments.**

```python
from e145_r7_state import curves_and_prices
points, measured, replayed, measured_price, replayed_price = \
    curves_and_prices(anchor="measured")
```

`anchor` selects the width-1 cost and takes `"measured"`, `"extrapolated"` or
`"serial"`; R2b shows the three agree to 0.75 %. `measured` and `replayed` are
`e128_price.CURVE` dictionaries. `measured_price` and `replayed_price` are the
`(marginal, cumulative)` price tables the *chooser* reads.

**Step 2 — decide which of the two substitutions the arm needs.** They are
independent and E128 conflated them:

- the **charged** cost is `e128_price.ranked_round_us(depth + 1)`, read from
  the installed module-global `CURVE`. `run_state_arm` sets it with
  `e140_cells.install(cost_curve)`.
- the **believed** cost is the `price` argument, which reaches only the depth
  chooser.

A re-pricing that changes one and not the other answers a different question.
R3's 3 x 8 grid is the map of which cell is which.

**Step 3 — score the arms.**

```python
from e145_r7_state import score_state_arms
import e134_rung2

arms = {
    "shrinkdeep_on_measured": {
        "cost":   measured,
        "price":  measured_price,
        "state":  "shipped",                       # inject no walker
        "rule":   "greedy",
        "adjust": e134_rung2.make_arm("shrinkdeep", 0.5),
    },
}
out = score_state_arms(cache, seeds, receipt, windows, arms, admitted)
```

`state = "shipped"` is the switch that matters: it injects **no** walker, so
the shipped `walk` runs and the arm is expressed entirely through `adjust`.
Each returned row carries `median_pct_mean`, `median_pct_sd`, `per_prompt`,
`weighted_mean_depth`, `width_hist`, `frac_rounds_inadmissible` and `rounds`.

**Step 4 — pick the adapter for the arm's shape.** The campaign has two
different arm interfaces, and they enter at two different slots:

| arm family | factory | returns | slot |
| --- | --- | --- | --- |
| E134 rung-2 | `e134_rung2.make_arm(name, k)` | a per-step hook `adjust(depth, p, ctx) -> p` | `"adjust"`, straight through |
| E128 policy | `e128_price.make_policy(arm, recal, level)` | a whole rule `policy(ema, margin, offer, capability) -> depth` | needs the four-line adapter below |

`e134_rung2.make_arm` accepts exactly `ship`, `shrinkdeep`, `slopedeep`,
`kmdeep` and `capdeep`. Those five map straight through with no adapter at
all.

E128's own 14-arm table comes from `e128_price.make_policy`, whose full
inventory is `e128_price.ARMS`: `ship`, `nomargin`, `nomargin0`, `nomargin1`,
`recal`, `marginup`, `marginfull`, `rankedprice`, `rankedprice_nomargin`,
`rankedprice_recal`, `rankedprice_marginup`, `static7`, `oracle`, plus
`LEVEL_ARMS` (`reachonly`, `expectedonly`, `levelfix`, `jensen`,
`jensen_both`), `LEVEL_GRID_ARMS` and `PRICE_ARMS`. A policy replaces the
whole rule, so it enters at `simulate`'s `walker` slot, whose contract is
`chooser(ema, margin, offer, adjust, ctx, force, price) -> depth`:

```python
import e128_price

def e128_walker(arm, recal=(2.0, 3.0), level=None):
    policy = e128_price.make_policy(arm, recal=recal, level=level)
    def chooser(ema, margin, offer, adjust, ctx, force, price):
        return policy(ema, margin, offer, ctx["capability"])
    return chooser
```

`ctx` carries `capability`, `offer`, `base_rate`, `prev_slope`, `prev_rows`
and `km`, so no policy needs information the harness does not already have.
That adapter is checked, not sketched: it returns a legal depth for `ship`,
`nomargin`, `recal`, `marginup`, `marginfull`, `rankedprice`, `static7` and
`oracle` on a probe state, and `static7` correctly saturates at the
`SEGMENTED_VERIFY_DEPTH_CAP` of 7 while the rest sit at 3 or 4.

**One ordering constraint.** The `rankedprice*` arms freeze their price table
at construction time, because `make_policy` calls `ranked_price_table()`
eagerly. Call `e140_cells.install(measured)` **before** `make_policy`, or that
family silently keeps the replayed table while everything else moves. The
`levelfix*`, `reachonly`, `expectedonly`, `jensen` and `jensen_both` arms also
need a `level` dictionary from `e128_price.pooled_level`.

`state_walker(rule, state, p_target, *, lam, sigma, noise_seed, clamp)` builds
the information-state walkers R7-2 and R7-3 use, and `run_state_arm(cache,
seed, prompt, cost_curve, price, walker, windows, adjust=None)` is the single
prompt-level call underneath all of it. An arm may set `walker` or `adjust`
or both.

**The consequence to name.** E128's published table, on the replayed curve,
ran from `oracle` at +9.08 % down to `static7` at -16.59 %, with
`marginfull` -0.33, `expectedonly` -0.76, `levelfix` -1.02, `recal` -1.33,
`reachonly` -1.97, `nomargin1` -2.18, `nomargin0` -2.44, `nomargin` -4.03,
`marginup` -4.54, `rankedprice` -5.63, `jensen_both` -7.62 and `jensen`
-11.07. R7-1 already re-priced two of those cells and both moved a long way:
`rankedprice` moves from -3.1234 % to +0.1338 %, and `oracle` moves from
+8.9390 % to +6.3508 %. **No ranking inside that table should be trusted
until it is re-run.** In particular, arms that were rejected for being
slightly negative sit inside the 3.26 pp that re-pricing moved `rankedprice`.

## What could not be identified

Two quantities are **not identified** by this data, and I am withholding
coefficients rather than reporting fitted numbers that mean nothing:

1. **Repair cost per replay.** Fitted from shipped legs it straddles zero:
   beagle ship -498 us/replay, beagle pb6 +251, benchfixture ship +820,
   benchfixture pb6 -1,477. The between-width least-squares fit is degenerate
   because width and replay share are collinear at `corr = 0.9843`.
2. **Ranked prefill from two curves.** The two-curve identification failed, and
   R5-0 makes it moot: the board publishes `prefill_seconds_per_token`
   directly. Nothing in this report fits ranked prefill any more.

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
research/e145_r6_session.sh 512 beagle_a
python3 research/e145_read.py
python3 research/e145_r0.py
python3 research/e145_anchors.py
python3 research/e145_r1.py
python3 research/e145_curve.py
python3 research/e145_r3.py --seeds 6 --windows 200
research/e145_r4_both.sh 6 200
python3 research/e145_r5.py --seeds 6 --windows 200 --draws 400
python3 research/e145_r6.py
python3 research/e145_r7.py --seeds 6 --windows 200 --draws 2000
python3 research/e145_r7_state.py --seeds 6 --windows 200
python3 research/e145_wandb_log.py
```

Timed sessions hold the GPU and take the real cool gate; the analysis steps
use no GPU. `e145_r4_both.sh` runs the measured and replayed cost models and
then the cross-evaluation, which needs both blobs. `e145_r5.py`, `e145_r6.py`
and `e145_r7.py` read the board cache and the leg blob and touch no GPU.
`e145_r6_session.sh` is the only session that needs the temporary residency
probe patch described under R6-1; that patch is reverted in this branch, so
reproducing R6-1 means re-applying it first.

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
`senpai/check-editable-budget.sh` reports source 2,627,254 of 3,000,000 bytes,
headroom 372,746, and candidate growth **1,717** of 262,144 across 154 files.
`senpai/verify-ranked-score-boundary.sh` passes.

The only submitted-surface change left in this branch is a 32-line depth pin
in `Qwen36MTPBlockSession.swift`, read once at static initialisation from
`MLX_E145_PIN_DEPTH` and applied in the draft-count chooser. It is inert
unless that variable is set, and R2 is the only session that sets it. The
temporary residency probe used by R6-1 is reverted.

**No Yukon submission was made.** The assignment forbids it, and the
submission slot is held by another candidate.
