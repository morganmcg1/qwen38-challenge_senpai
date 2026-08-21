# E99 — the round-level allocation bound, and why the margin gate lost on M5

Student `qwen-edward`. PR #101. Assignment `qwen38-r1-e99-oracle-allocation-bound`, r1 measured, r2 deleted the candidate.
Base `f556bd5f9bcf0ddb3c8c6ccd33490cb0d2000e03`. `harness=local` throughout.

## Identity tuple

| field | value |
|---|---|
| base | `f556bd5f9bcf0ddb3c8c6ccd33490cb0d2000e03`, merged in at `8836058` so the branch stays a fast-forward of its published head. The r1 legs were measured on `4d937ce35854f75db70eabf00f152daf1bca0ad2`; the two bases differ in `research/` only, so no measurement changes. |
| submission base | `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` |
| host | Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 26.5.2, Swift 6.3.3 |
| fixture | public local fixture, `--local-iterate` |
| token window | 128 decode tokens per leg, offered caps 4, 5, 6, 7 and 8 |
| head | declared, organizer-pinned proposal head |
| pricing curve | ranked M5, `research/ranked_cost_curve.py` |
| gate flags | `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false` on every leg |
| timing mode | `MLXFAST_LOCAL_COOL_GATE=0`, ABBA counterbalanced, entry and exit GPU temperature per leg |

Twenty W&B runs carry the legs. The run ids are listed in the terminal result on PR #101.

## Report shape

**Question.** How much round-level allocation prize does the shipped adaptive walk leave on the ranked M5 cost curve, and how much of it is reachable from information the walk already has before the round?

**Evidence that made it worth testing.** E94 could not reproduce the ranked mean draft length from any constant-`q` model (beagle +53.79 % error at identical pooled acceptance). That failure is the measurement of a correlation between chosen depth and about-to-be-realised acceptance, which is exactly the allocation mechanism.

**Expected result.** An oracle upper bound, a realisable lower bound, and a decision on Theme G.

**Smallest decisive test.** Replay recorded rounds. The shipped trace already records every field the assignment listed, so rungs 2 to 4 cost zero GPU time.

**Stop or promotion rule.** Pre-registered: `actual - oracle` under 2.0 % kills Theme G; `best realisable - actual` under 1.0 % means the prize is unreachable; over 5.0 % means stop and report immediately.

## Rungs 1 to 4 — the bound (the assignment's question)

`MLX_QWEN_MTP_TRACE=1` already records per round the chosen depth, the accepted prefix, the full pre-round `positionAcceptEMA` vector, the pending primary's top-2 margin, the `fullAcceptStreak`, the width cap and the round index. `snapshotScheduleSignal` takes that snapshot **before** the walk, so it is pre-round state.

Headline, cap-5 leg `e99r5c5a1`, mean verify width 5.804, nearest scoring prompt `republic`, ranked M5 curve, µs per emitted token:

| treatment | actual | oracle | oracle gap | one-bit G | best fixed |
|---|---:|---:|---:|---:|---:|
| observed (`a* = a_r`) | 11,122.5 | 10,476.1 | **5.81 %** | 4.31 % | +2.58 % |
| impute (E92 `q_i`) | 11,122.5 | 10,409.7 | **6.41 %** | 4.31 % | +0.24 % |
| exclude censored | 18,925.1 | 13,207.2 | 30.21 % | 21.21 % | -21.21 % |

`best fixed` is positive when the best single constant depth is worse than the shipped walk. The three treatments agree in sign; the two unbiased ones agree to 0.6 pp. The `exclude` treatment keeps only the 19.6 % of rounds that rejected before the cap, so it is selected on the outcome and its level is not comparable.

Two curves, opposite verdicts on the same rounds: on the ranked curve the best fixed depth is 2.58 % **worse** than the shipped walk; on the local curve it is 9.46 % **better**. A local speed measurement cannot rank a schedule (ADVISOR ERROR 43, second form).

Reachability, one-bit action set `{ship, clamp into the G = 1 band}`, seven folds (parity, time-order and four leg-out), pre-round features only:

| fold | held-out gain |
|---|---:|
| parity even→odd | +1.853 % |
| parity odd→even | +2.165 % |
| time first→second | +0.992 % |
| leg-out cap 4 | +1.130 % |
| leg-out cap 5 | +2.127 % |
| leg-out cap 6 | +2.498 % |
| leg-out cap 8 | +3.044 % |

`margin` is the first split feature on 6 of 7 folds (`streak` on the seventh). The fit beats a random-clamp control matched on clamp count on 7 of 7 folds. So the bound is 5.81 to 6.41 %, of which 36 to 44 % is reachable from pre-round state.

## Rung 5 — the candidate, and the official refutation

A single margin threshold clamping low-confidence rounds into the `G = 1` band, `t = 9.4375`, clamp depth 3. Shipped on by default because the ranked worker exports no campaign environment.

| quantity | value |
|---|---|
| baseline submission | `f04b102` 3.32824629 (promoted) |
| candidate submission | `87b654b` 3.12600524 |
| delta | **-0.20224105, -6.077 %, REJECTED** |

## Rungs 6 and 7 — the durable finding: the depth price is a stream boundary

Pre-registered discriminator over clamp depth, run on an already-running session:

```
gain(d2) - gain(d3) = -1.790 pp
predicted, stream boundary   : -0.727
predicted, drafting cost only: +0.443
verdict: stream boundary
```

The depth-price structure is set by a hardware dispatch and weight-stream boundary, not by drafting cost. The advisor's depth-4 falsification arm was refuted the same way: predicted 61.5 % recovery, measured 57 %, against the claim of near zero. Depths 4 and 5 have no discriminating power because the two candidate curves are the same line at `M >= 5`, which is why depth 1 was added.

Cross-build reproduction passed exactly on both arms across three worker digests, `exact=true`, `dram_floor_violations=0`.

## Rung 8 — the gated oracle, which closes the theme

Re-running the whole bound with the gate on:

| arm | oracle gap (observed→exclude) | reachable |
|---|---|---|
| ungated | 5.81 % → 30.21 % | 36 to 44 %, `margin` first split on 6/7 folds, beats control 7/7 |
| gated | 2.88 % → 23.95 % | none reliable, held-out one-bit gains -2.41 to +0.75 %, `margin` first split on 0/7 folds, beats control 2/7 |

The oracle is arm-invariant to within 1 % (+0.48, -0.11, +0.99, +0.36 % across caps 4, 5, 6 and 8), which validates the instrument before it is used to judge the arm. **The schedule allocation theme is closed** unless a genuinely new pre-round signal appears or the cost curve moves.

## Post-mortem — the candidate failed by overfiring, not by mispricing

Cost-curve error is ruled out. The empirical local round cost by width matches the oracle table to within 3.1 %, and re-pricing the recorded sequences while resizing the assumed M5 step at `M = 4 → 5` never turns the predicted gain negative:

| assumed step, µs | 0 | 7,233 | 9,946 (assumed) | 39,866 (local measured) |
|---|---:|---:|---:|---:|
| predicted gain | +0.778 % | +2.622 % | +3.222 % | +7.777 % |

Overfiring explains the loss quantitatively. Gain collapses with firing share: **+3.222 % at 26 %, -2.999 % at 81 %, -5.662 % at 100 %**. The official -6.077 % sits at the always-fire end. The root cause is the calibration: the margin distribution is bimodal (q0.10 = 1.25, q0.25 = 9.625, q0.50 = 14.25) and the shipped constant 9.4375 sits at quantile 0.244, immediately below the dense region, so a small distribution shift on the hidden prompts sweeps the mass across at once. An absolute constant was tuned on one public fixture and shipped against eight hidden prompts.

Warning signs present in the pre-submission data and under-weighted: gain against offered cap is jagged, not smooth (+3.22 cap 8, -0.31 cap 6, +3.13 cap 5, +0.12 cap 4); the threshold optimum itself moves with cap; and rung 7 identified the mechanism as a hardware cost-step boundary, which is a reason to expect cross-host fragility.

Two independent routes reach the same answer. The advisor's receipt-side decomposition (`research/board_pair_decompose.py f04b102e 87b654b2`) implies per-prompt fire rates of 0.295 beagle, 0.435 essays, 0.506 republic, 0.729 botany and 1.000 medicine against a local 0.259 at cap 8, with candidate slowdown regressing on fire rate at -10.89 % per unit, `r = -0.958`.

## Methodological result

The ranked-curve statistic has **zero run-to-run variance**: four replicates per arm gave one distinct value on both arms, because the `(round, draft, accept)` sequence is bit-identical. One leg per cell suffices for any ranked-curve contrast. The caveat is the important half: zero variance is not accuracy. The statistic is a deterministic function of a cost model, so it repeats its own bias exactly, which is how a confidently wrong local number was produced.

## Campaign rule adopted from this experiment

Require realised-firing-share telemetry for any conditional schedule gate, and refuse promotion when a local sweep shows a sign change within a factor of two of the operating rate. The E99 cap sweep (+0.12, +3.13, -0.31, +3.22) would have blocked this submission.

## r2 — the gate is deleted from the candidate surface

The gate is a closed mechanism with a measured -6.077 % official cost, so it is removed rather than defaulted off. After `fb0be4b`, `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` and `Tests/MLXFastTests/QwenMTPDepthPriceTests.swift` are byte-identical with base `f556bd5f`, and `git diff f556bd5f HEAD -- Sources Tests Vendor mtp-head.manifest.json Package.swift` is empty.

Deleted: `enum MarginGateArm`, `marginGateArm`, `marginGateDefaultThreshold`, `marginGateThreshold`, `marginGateDefaultDepth`, `marginGateDepth`, `pendingTop2Margin()`, the three gate lines in `costModelDepth`, the `gate=`, `gt=`, `gd=` and `fire=` trace fields, and the `marginGateShipsOn` test with its `streamsPerRound` helper. Every other `costModelDepth` test still runs.

### Recovery path

The complete gate implementation is at commit **`c2d94f9f44b3b75d1f017705daf58197eb422f56`** on branch `qwen-edward/e99-oracle-allocation-bound`, which is the tip immediately before the deletion. Restore it with either line:

```bash
git checkout c2d94f9f -- Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
                         Tests/MLXFastTests/QwenMTPDepthPriceTests.swift
git revert -n fb0be4b5
```

A single `git cherry-pick -n <sha>` does not work here, because the gate was built across five commits rather than added by one: `0cb3951` (arm and clamp), `959d872` (ranked pricing of the realised sequence), `bf79839` (default on), `a656288` (threshold 9.4375) and `49ad19a` (clamp-depth override). Cherry-pick that list in order if you want the history instead of the end state.

### The research scripts stay, and they will not fire on the merged base

Every `research/e99_*.py`, `research/e99_*.sh` and `research/e99-artifacts/*` file is kept as the record of the measurement. `research/e99_rung5.sh`, `research/e99_rung67.sh`, `research/e99_rung5_price.py` and `research/out/e99_local_submit.sh` still export or read `MLX_QWEN_MTP_MARGIN_GATE`, `MLX_QWEN_MTP_MARGIN_GATE_T` and `MLX_QWEN_MTP_MARGIN_GATE_D`. **No Swift code reads those variables on the merged base**, so those legs are replay-only history: exporting the variables changes nothing until the gate is restored from `c2d94f9f`. The offline analysis tools (`e99_oracle.py`, `e99_threshold_map.py`, `e99_transfer_postmortem.py`, `e99_rung67_report.py`, `e99_rung7_predict.py`, `e99_margin_dist.py`) read recorded traces only and still run unchanged.

## Correctness

All 16 session legs matched exactly. Zero DRAM-floor violations, zero residual divergences, `dirty_candidate_paths=0`.

`swift test --force-resolved-versions` at the r2 head runs 728 tests in 64 suites and stops at the organizer floor: **40 issues under 9 test names**, none of them added by this branch.

| failing test | area |
|---|---|
| `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen` | campaign docs marker |
| `participantDocsExposeDefaultCLIInstallDirectory` | campaign docs marker |
| `submissionStaticReviewPromptCoversMeasurementStructureExploitation` | campaign docs marker |
| `qwen36ConfigContractDigestMatchesTheReferenceManifest` | artifact manifest |
| `theCheckedInDeclarationSelectsThePinnedHead` | head declaration |
| `theQwenMTPTrackIsArmedOnQwen38` | fixture |
| `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues` | fixture |
| `theSeededCalibrationExpectationMatchesItsRecordedProvenance` | fixture |
| `startupMemoryPolicyKeepsRanked128GiBProfile` | ranked memory profile on a 48 GiB host |

The scored-surface suite `E68 depth price` passes.

## Follow-ups, not implemented

1. Declined by the advisor: express the threshold as a running quantile of margins inside the same request, targeting a fixed firing share near 0.25. Correct fix for the observed failure, but rung 8 says there is no reliably reachable prize left after clamping.
2. Adopted as a campaign rule: realised-firing-share telemetry plus the sign-change promotion block described above.
3. On the stop list: do **not** retune the constant to 11.5625. Same class of error.
4. The single named reopener: per-position head-side confidence, the head logits for every drafted position that are computed and never reduced or retained. Different signal class, not a re-tuning. Not open now.
