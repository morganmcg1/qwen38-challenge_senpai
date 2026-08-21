# E94 arms, written during E92 and held out of the E92 tree

The advisor asked for the E94 patch to be written while the E92 sweep held the
GPU, and asked that the mechanism stay out of PR #94. Both are satisfied here:
the code is complete and compile-verified, and it lives only as a patch under
`research/`, which is not an editable path and never ships.

Apply it in the E94 branch:

```bash
git apply research/e94-artifacts/e94-depth-price-arms.patch
```

## What the patch adds

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`

- `measuredRawDepthPriceE92` — the E92 rung-2 `pure` curve, production sweep,
  job `2694a061`. Index `i` prices the step into verify width `i + 2`. The
  group-transition cliff sits at index 3 (width 5) and repeats at index 7
  (width 9), matching `G = ceil(M / 4)`.
- `makeDepthPrice(rescaling:)` — the E68 rescale, now taking the raw curve as a
  parameter so every arm holds the shipped total and changes shape only.
  `makeMeasuredDepthPrice()` calls it, so `pbfit` stays bit-identical.
- Three arms on the existing `DepthPriceArm` enum:
  - `snap4` — the shipped greedy walk plus one guard: a walk that returns 4
    returns 3.
  - `amin` — global minimum of cost per token over the shipped uniform price.
    Implementation check, not a science arm.
  - `amine92` — global minimum over the E92 refit shape.
- `positionAcceptEstimate(_:)` — the per-position acceptance estimate, lifted
  out of the greedy walk so the argmin walk reads exactly the same inputs. The
  arithmetic is unchanged, and `QwenMTPDepthPriceTests` proves the `ship` walk
  still equals the tip's walk on a dense grid.
- `argminDepth(cap:price:)` — scores every legal depth and takes the best.

`Tests/MLXFastTests/QwenMTPDepthPriceTests.swift`

- `e92ArmHoldsTheTotalAndCliffsAtWidthFive` — the arm holds the shipped level
  and its width-5 step is more than twice every non-cliff step.
- `depthFourIsAlwaysDominated` — the theorem `snap4` implements, in the form
  that uses no price array.
- `rescaledPriceAlsoDominatesDepthFour` — the same theorem through the price
  array that `amine92` reads.

## The theorem

This section was rewritten after advisor feedback `e92-f7`. My first version
read a margin off a near-zero coefficient, which was wrong. The corrected form
below is both simpler and much stronger.

A depth-4 round beats a depth-3 round only when

```text
C(w5) / C(w4) < Y(4) / Y(3)
```

Write `a_i` for the probability that the chain reaches position `i`. Then
`Y(3) = 1 + a1 + a2 + a3` and `Y(4) - Y(3) = a4`. Reach is non-increasing, so
`a1 >= a2 >= a3 >= a4` and

```text
Y(4) / Y(3) = 1 + a4 / (1 + a1 + a2 + a3)
           <= 1 + a4 / (1 + 3 * a4)
           <= 1.25          attained only in the limit q -> 1
```

**`Y(4) / Y(3) <= 1.25` holds for every acceptance profile that can exist.**
E92 measured `C(w5) / C(w4) = 126,103.1 / 86,237.4 = 1.4623`, which is 17.0 %
above the ceiling, so **depth 4 is dominated unconditionally**.

The claim rests on two measured round-busy numbers and one combinatorial
inequality. It does not use `makeMeasuredDepthPrice`, the rescaling to
`8 * h = 1.44`, the `pure` versus `hadd` choice, or any acceptance estimate,
so it survives every objection to the price array.

### Why my first margin of 0.735 % was not a margin

I rearranged the decision to `r4 * (C3 - 3 * m4) > m4` and read the margin off
the coefficient `C3 - 3 * m4 = -0.009`, which is 0.735 % of `C3`. The
coefficient really does sit near zero, but a coefficient near zero does not
make the decision near. The inequality needs `r4 * (negative) > m4` with
`m4 = 0.411`, so the required `r4` is `-45.7`. The failure is not marginal.

The correct condition in price-array terms is `m4 / C3 > 1/4`. Measured
`m4 / C3 = 0.335783`, so the price sits **34.3 % above the ceiling**, and
equivalently `m4` would have to **fall 25.5 %** before depth 4 became viable.
Both figures describe the same margin with different denominators.

### Margins in all three normalisations

| normalisation | M4 Pro measured | M5 at a 1.126x flatter cliff |
|---|---:|---:|
| `C(w5)/C(w4)` against the 1.25 ceiling | **17.0 %** | **12.8 %** |
| marginal step into width 5 must fall from | 39,866 to 21,559 us = **45.9 %** | 35,405 to 21,559 us = **39.1 %** |
| rescaled `m4` must fall | 0.411272 to 0.306204 = **25.5 %** | 0.365250 to 0.306204 = **16.2 %** |

### Required cliff flattening, corrected

`e92-f7` states the cliff would have to flatten by `1.343x` before the theorem
fails. That value comes from `0.411272 / 0.306204`, which shrinks the rescaled
`m4` without re-rescaling the curve, so it is not a statement about the raw
cliff. Three consistent answers, computed from the E92 artefacts:

| model | required flattening of the raw width-5 step |
|---|---:|
| raw round-busy, assumption-free (the sound one) | **1.8491x** |
| rescaled price array, re-rescaled correctly to 1.44 | 1.4475x |
| rescaled `m4` alone, no re-rescale (`e92-f7`) | 1.3431x |

All three exceed the `1.126x` M5 flattening the advisor assumes, so the
conclusion that `snap4` transfers is unchanged. The correction widens the
safety margin rather than narrowing it.

## Modelled cost per token at the E92 pooled acceptance profile

`q = 0.9659, 0.9652, 0.9543, 0.9486, 0.9487, 0.9859, 0.9451, 0.8333`

| depth | C(d) | Y(d) | C/Y |
|---|---|---|---|
| 0 | 1.000000 | 1.0000 | 1.000000 |
| 1 | 1.054987 | 1.9659 | 0.536643 |
| 2 | 1.106599 | 2.8982 | 0.381825 |
| 3 | 1.224816 | 3.7879 | 0.323352 |
| 4 | 1.636087 | 4.6318 | 0.353228 |
| 5 | 1.757198 | 5.4325 | 0.323462 |
| 6 | 1.887069 | 6.2218 | 0.303297 |
| 7 | 2.026605 | 6.9679 | **0.290850** |
| 8 | 2.440000 | 7.5895 | 0.321495 |

Two consequences for the E94 run plan:

- At offered depth 8 the argmin is 7, which is where the shipped walk already
  operates. Expect `amine92` to be close to a null there, as predicted.
- At offered depth 5 the argmin is 3, but only by 0.03 % over depth 5 on the
  RESCALED price. The rescale holds the total at `8 * h`, so it compresses the
  real level: the measured `C(5) / C(1)` is 1.9567 against a modelled 1.6361.
  On the unrescaled measured curve depth 3 wins by much more. Report the
  offered-depth-5 arm against the measured curve, not against the model.

## Run plan the advisor specified

Order, from `e92-f6` section 9 and `e92-f7` section 5:

1. **The cap-4 operating-point screen first**, before any arm sweep.
   `MLXFAST_QWEN_MTP_DEPTH=4`, one ABBA pair, `ship` against `snap4`. At a cap
   of 4 the shipped walk selects 4 and `snap4` selects 3 on nearly every round,
   so this is a clean A/B at maximum mass for two legs.
2. Repeat at cap 5 and cap 8.
3. Then the full arm sweep over `ship`, `snap4`, `amin`, `amine92`.

**Primary output is the depth histogram, not the timing delta.** The dominance
result is assumption-free, so the screen is no longer testing whether depth 3
beats depth 4. It measures how much of the leg the depth-4 rounds carry, which
is the one number that sets the ranked prize. The seconds-per-token delta is
the confirmation.

Advisor prediction at cap 4: round-busy per token falls from about 26,000 to
about 22,500 us, near 13.6 %; tokens per round move 4.670 to 3.832 with the
round count rising to match; leg seconds per token fall about 12.4 %. A move
below 8 % means the model is missing something, and that finding comes first.

Every leg must report `all_tokens_matched: true`, and `effective_mean_draft_len`
must move, or the arm did not fire. Label all three caps as directional screens
at a non-ranked offered depth.

Two constraints carried from the feedback:

- **Do not bundle the acceptance-prior fix into E94.** Deliverable (a) shows
  the shipped per-position EMA seed, `0.85 * pow(0.98, i)` at
  `Qwen36MTPBlockSession.swift:804`, is wrong in both level and shape against
  the measured flat 0.955. Changing it moves the walk's inputs everywhere at
  once and would confound `snap4`. The advisor will assign it separately.
- **`amin` and `amine92` stay screens.** Uniform shallowing is ranked-
  catastrophic: `h = 0.32` scored 2.84585, which is -14 %. Neither is promoted
  to a candidate unless `amin` shows a clean local win above 1 % at cap 4 or
  cap 5.

## Measurement floor

The ranked reproducibility floor is a median 0.191 % on the published score
statistic, p75 0.35 %, max 0.68 %, per-run sd 0.196 %, from 39 byte-identical
replicate pairs. The two-sigma single-pair detection floor is 0.55 % published
and 0.32 % serial-free. `snap4` at a predicted +0.8 % to +2.7 % clears it.

## Verification already done

- `swift build --force-resolved-versions` on the merged advisor head — Build
  complete, no errors.
- `swift build --build-tests --force-resolved-versions` with the patch applied
  — Build complete.
- `swift test --force-resolved-versions --filter QwenMTPDepthPrice` with the
  patch applied — **16 tests in 1 suite passed**, including `the shipped arm is
  ship`, `every arm holds the shipped total`, and `the ship arm's walk equals
  the tip's walk on a dense grid`. The `ship` control survives the
  `positionAcceptEstimate` refactor bit for bit.
- `git apply --check` against the restored, post-merge E92 tree — clean.
- `python3 research/e94-artifacts/depth4_flattening_check.py` — reproduces
  every margin in `e92-f7` and reports the corrected flattening factors.
