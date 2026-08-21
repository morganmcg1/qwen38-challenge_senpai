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
- `depthFourIsAlwaysDominated` — the theorem `snap4` implements.

## The theorem

Extending from depth 3 to depth 4 lowers modelled cost per token only when

```text
r4 > m3 * Y3 / C3
```

Reach is non-increasing, so `Y3 = 1 + r1 + r2 + r3 >= 1 + 3 * r4`. Substituting
gives the requirement `r4 * (C3 - 3 * m3) > m3`. With the E92 rescaled price
`C3 = 1.2248156` and `3 * m3 = 1.2338152`, so the left side is negative for
every `r4 >= 0` and the inequality has no solution. Depth 4 is dominated by
depth 3 on every round, for every acceptance profile, not on average. The
margin is 0.735 %, which is thin; it comes from the measurement, so a refit on
a different dispatch table must recheck it rather than assume it.

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

Three offered depths through `MLXFAST_QWEN_MTP_DEPTH`: 4, 5 and 8. Arms `ship`,
`snap4`, `amin`, `amine92`. Every leg must report `all_tokens_matched: true`,
and `effective_mean_draft_len` must move, or the arm did not fire. Label all
three as directional screens at a non-ranked offered depth.

## Verification already done

- `swift build --build-tests --force-resolved-versions` — Build complete.
- `swift test --force-resolved-versions --filter QwenMTPDepthPrice` — 15 tests
  in 1 suite passed, including `the shipped arm is ship`, `every arm holds the
  shipped total`, and `the ship arm's walk equals the tip's walk on a dense
  grid`. The `ship` control survives the `positionAcceptEstimate` refactor.
- `git apply --check` against the restored E92 tree — clean.
