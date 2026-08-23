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
research/e145_r2_session.sh 512 beagle_a
research/e145_r2b_session.sh 512 beagle_a
python3 research/e145_read.py
python3 research/e145_r0.py
python3 research/e145_r3.py --seeds 6 --windows 200
python3 research/e145_r4.py --seeds 6 --windows 200
```

Timed sessions hold the GPU and take the real cool gate; the analysis steps
use no GPU.
