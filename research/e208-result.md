# E208: staged QMV plan entry (9,3) -> (9,5)

`harness=local`. Apple M4 Pro. Ungated and counterbalanced.
`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`official_or_ranked_score=false`. Nothing here is an official or ranked score.

## Question

Does changing the staged QMV plan entry `(9,3)` to `(9,5)` remove one of the
three full weight passes at m = 9 on the cap-8 surface, with exact tokens and
clean numerics?

## Mechanism

At m = 9 all seven fused decode cells run `.staged` with an items-per-group of
3, so the kernel makes G = 3 weight passes. An items-per-group of 5 splits the
9 rows into a group of 5 and a tail group of 4, so G = 2. FINDINGs 534 to 536
priced the third pass at 40.051 ms per m = 9 round, or 22.13 ms per round after
weighting by the cap-8 census share of m = 9 rounds.

## Implementation

Two submitted files change against base `846a2033`:

- `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`: `segmentedVerifyDepthCap`
  7 -> 8, which is what puts m = 9 on the served surface.
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`: a new
  `Qwen35QMVKernelVariant.stagedWide9` with pairs
  `[(2,2),(3,3),(4,4),(5,5),(6,3),(7,4),(8,4),(9,5)]`, four matching kernel
  globals, and the arm switch.

The arm is selected once by `DARKBLOOM_E208_QMV_ARM`. The plan witness
`qwen35QMVWidthPlanWitness` is derived from the selected table, not from the
environment, and is emitted every round as `qmv_plan=` (RULE 391).

## Identity tuple (RULE 386)

| field | value |
| --- | --- |
| campaign base | `846a2033e032faf45efe6cedd4be4b95da7d7cd7` |
| candidate | `5fc04d00c3d06ed22a04f5fe290fec0c0c7785f9` |
| worker sha256 | `c73b4f477406d7ee8d8cf77815992df425fba079ea4e5e6af52b24be8a5933f5` |
| worker digest stable | true, identical in all four legs |
| metallib source fingerprint | `ce5e223eb6b404f16f07aca83ca44e65fc517187f3b130edaf4e7acb56789d50`, unchanged |
| host / chip | `ip-10-231-2-12.ec2.internal` / Apple M4 Pro, 48 GiB |
| dirty candidate paths | 0 |
| token window | 512 decode tokens |
| mode | `--local-iterate`, one public fixture |
| depth cap | 8 |
| head | organizer-pinned, `head_provenance_sha256=dadbfb80…` |

## Stage 0: numerics gate

Direct comparison of the `(9,3)` and `(9,5)` group partitions at m = 9 across
all seven fused decode cells, on both the table and the plain paths.

| quantity | value |
| --- | --- |
| elements compared | 5,927,616 |
| differing | **0** |
| worst max ULP | **0** |
| non-finite | 0 |
| groups, off arm | 3 in every cell |
| groups, on arm | 2 in every cell |

Positive control, injected perturbation on the same comparison: 415 differing,
max ULP 23, max absolute difference 0.25 on both paths. The comparison can
fail, so the zero above is informative.

Metal source form: this kernel family is a Swift JIT source string. It has no
`mlx-generated/*.cpp` twin and no `mlx.metallib` entry, so no twin or metallib
rebuild applies. `python3 research/twin_audit.py` reports
`TWIN AUDIT OK: 29 runtime-effective twin(s)` and the metallib fingerprint is
unchanged.

## Stage 0 screen at 64 tokens: coverage gap, plumbing correct

The 64-token screen passed exactness in all four legs with the correct witness
per arm and a closed row ledger, but it produced **zero m = 9 rounds**. The
adaptive scheduler ramps from d = 4 and only reaches d = 7 by round 9, with the
depth-8 EMA entry still at 0.770. Both arms therefore ran identical code and
the whole-leg difference was noise.

This screen paid for itself by exposing two analyzer defects before the
expensive session:

1. The trace covers the whole worker process, so it starts with the serial
   control rounds. The analyzer counted all 74 traced rounds for a 10-round MTP
   leg. It now uses only the MTP tail and cross-checks that tail against the
   trusted report's `effective_draft_lengths`.
2. The session meta recorded the candidate HEAD under the name `base_sha`. It
   now records `candidate_sha` and `campaign_base_sha` separately.

The analyzer now also refuses to pair, and reports a problem, when a session
has no m = 9 rounds, so an uncovered session can never be published as a null
result for this mechanism.

## Stage 1: one ABBA session at 512 tokens

Four legs, order `off on on off`, 512 decode tokens. Analyzer `problems` list
is empty.

### The predicted surface reproduced exactly

| quantity | assignment census | measured |
| --- | --- | --- |
| qL9 share of rounds | 55.26% | 55.263% (42 of 76) |
| accepted draft rate | 0.83685 | 0.83685 |
| accepted drafts per round | 5.7368 | 5.7368 (436 / 76) |

Served-width census: `{4:3, 5:7, 6:8, 7:4, 8:12, 9:42}`.

### Exactness and pairing

| leg | arm | witness | matched | divergence | rows | replayed | proposed/round | acc rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | off | `selective-m6+ipg9-3` | true | 0 | 597/597 | 15 | 6.8553 | 0.83685 |
| 2 | on | `selective-m6+ipg9-5` | true | 0 | 597/597 | 15 | 6.8553 | 0.83685 |
| 3 | on | `selective-m6+ipg9-5` | true | 0 | 597/597 | 15 | 6.8553 | 0.83685 |
| 4 | off | `selective-m6+ipg9-3` | true | 0 | 597/597 | 15 | 6.8553 | 0.83685 |

The `(d, acc)` trajectory is identical in all four legs, so exact round-index
pairing is valid (RULE 396(b)). The row ledger closes in every leg (RULE 179).

### Decision statistics (RULE 394)

Trusted-parent `block_request_seconds`, per round. Positive means the ON arm
recovered time.

| statistic | n | mean ms/round | 95% CI |
| --- | --- | --- | --- |
| **paired m = 9** | 42 | **+20.774** | [+20.396, +21.152] |
| **paired all rounds, round weighted** | 76 | **+11.757** | [+9.394, +14.121] |
| paired non-m = 9 control | 34 | +0.619 | [-0.804, +2.043] |
| drift floor, off vs off | 76 | -0.896 | [-2.225, +0.432] |
| drift floor, on vs on | 76 | -1.166 | [-1.592, -0.740] |

The effect is confined to m = 9. The non-m = 9 control is consistent with zero
and both same-arm drift floors are an order of magnitude below the effect.

### Whole-leg absolute candidate time

| arm | candidate MTP s/token | serial s/token | local ratio |
| --- | --- | --- | --- |
| off | 0.030999, 0.031126 (mean 0.0310624) | 0.073436, 0.073298 | 2.3690, 2.3549 |
| on | 0.029235, 0.029400 (mean 0.0293173) | 0.073263, 0.073362 | 2.5060, 2.4953 |

Relative gain in candidate MTP seconds per token: **+5.618%**, a delta of
0.0017451 s/token.

The two channels agree independently: 11.757 ms/round x 76 rounds / 512 tokens
= 0.0017451 s/token, which reproduces the whole-leg delta to every printed
digit.

The serial leg is flat across arms (spread 0.24%, no arm pattern). The serial
leg always runs m = 1 and never reaches the changed plan entry, so the causal
path is confined to the candidate MTP leg. The local ratio is therefore
admissible direct evidence here, alongside the absolute.

## Why recovery is about half the desk prediction

The desk model predicted 40.05 ms per m = 9 round; the measurement recovered
20.77 ms, or 51.9%. The candidate-side phase split, attribution only:

| phase | off (us) | on (us) | delta (us) |
| --- | --- | --- | --- |
| `round_us` | 185890 | 164884 | +21006 |
| `eval_wall_us` | 89612 | 77562 | +12049 |
| `verify_build_us` | 90913 | 80934 | +9979 |
| `commit_us` | 552 | 1365 | -813 |
| `upkeep_us` | 64 | 198 | -134 |
| `draft_build_us` | 4737 | 4792 | -55 |
| `readout_us` | 10 | 31 | -20 |

Candidate-side `round_us` recovers 21.0 ms, corroborating the trusted parent's
20.77 ms from an independent clock.

A weight pass has a fixed weight-streaming cost plus a per-row arithmetic cost.
Moving from three passes of 3 rows to two passes of 5 and 4 rows removes one
weight-streaming cost, but all 9 rows are still computed, so the per-row
arithmetic is unchanged. The recoverable part is the fixed share of a pass, not
a whole pass. Recovering 51.9% of the naive estimate is the expected signature
of that decomposition rather than a shortfall.

## Stop rule

The predeclared rule was: if the paired m = 9 effect is below 5 ms per round,
the enumeration is wrong, so stop and report without iterating arms. The
measured effect is 20.77 ms per round with the entire 95% CI above 20.3 ms. The
rule is cleared by a wide margin and the enumeration is sound.

## Thermal disclosure

Entry temperatures 37.4, 56.1, 56.3, 56.9 C. Exit temperatures 59.1, 59.7,
60.6, 59.6 C. Entry spread 19.5 C, because leg 1 started cold. Leg 1 is an
**off** leg, so the cold start favoured the control arm; the off-vs-off drift
floor confirms leg 1 ran 0.90 ms per round faster than leg 4. The bias works
against the hypothesis, and the ON arm still won by 5.6%.

## Ranked projection, EXTRAPOLATED, `harness=ranked`

Not measured. `senpai/verify-ranked-score-boundary.sh` reports
`PASS: ranked numerator is pinned baseline; candidate edits affect the MTP
denominator only`. The ranked serial numerator cannot move, so a relative
reduction `g` in candidate MTP seconds per token multiplies every affected
`raw_p` by `1 / (1 - g)`.

With `g = 0.056182` on this host, the boundary-implied factor is 1.0595. The
assignment's supplied transfer constant `k = 0.23731` discounts the local
effect for the M4 Pro to M5 move, giving `3.70785 x 0.056182 x 0.23731 = 0.0494`
published points on receipt A. Both figures are desk models on a non-ranked
host and neither is a measurement.

## Swift tests

`swift test --force-resolved-versions` on the candidate: 756 tests in 80
suites, 41 issues. Every QMV suite passes:

- `E208 width-9 staged QMV group partition` passed
- `E195 cell-selective QMV width plan` passed
- `QwenQMVCostCurveTests` passed
- `QwenQMVParityTests` passed
- `scoredShapesStayOnTheQMVFastPath()` passed

The 41 issues are pre-existing. I ran the identical command on a detached
worktree of the unchanged base `846a2033` and the failing set is identical,
test for test and issue count for issue count:

| failing test | issues, candidate | issues, base |
| --- | --- | --- |
| `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen` | 1 | 1 |
| `participantDocsExposeDefaultCLIInstallDirectory` | 2 | 2 |
| `qwen36ConfigContractDigestMatchesTheReferenceManifest` | 2 | 2 |
| `startupMemoryPolicyKeepsRanked128GiBProfile` | 2 | 2 |
| `submissionStaticReviewPromptCoversMeasurementStructureExploitation` | 11 | 11 |
| `theCheckedInDeclarationSelectsThePinnedHead` | 6 | 6 |
| `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues` | 3 | 3 |
| `theQwenMTPTrackIsArmedOnQwen38` | 11 | 11 |
| `theSeededCalibrationExpectationMatchesItsRecordedProvenance` | 2 | 2 |
| `theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax` | 1 | 1 |

The set difference in both directions is empty. These tests assert checked-in
documents, fixture digests, the pinned head declaration, track naming, the
startup memory profile and the wired residency slack. None of them reads the
QMV plan table or the depth cap, and my branch does not touch any file they
inspect. The failure lists are saved at `research/out/e208/`.

## Contract checks against `846a2033`

- `senpai/verify-ranked-score-boundary.sh`: PASS
- assignment scope: OK, 2 submitted paths
- editable budget: `source=2660373/3000000 headroom=339627 growth=3278/262144`

## Reproduction

```bash
git checkout 5fc04d00c3d06ed22a04f5fe290fec0c0c7785f9
./setup.sh && ./setup-qwen-mtp.sh
research/e208_gate.sh                        # Stage 0 numerics gate
research/e208_session.sh e208/stage1-abba 512  # Stage 1 ABBA session
python3 research/e208_analyze.py research/out/e208/stage1-abba
```

## Suggested follow-ups, not implemented

1. **Sweep the items-per-group at m = 9.** `(9,5)` was the single point this
   assignment authorised. If the recovered time is the fixed cost of one pass,
   then `(9,9)` should remove the second pass as well and roughly double the
   effect, unless register pressure or occupancy turns over first. `(9,7)`
   splits 7 and 2 and is a useful asymmetry probe.
2. **Re-price the other staged entries.** The same fixed-versus-per-row split
   should apply at m = 8, which is 12 of 76 rounds here and still runs
   `(8,4)`, that is two groups. `(8,8)` would take it to one.
3. **Confirm at 512 tokens under `--local-submit`.** This assignment budgeted
   one timed session and forbade an official submission, so the exact post-EOS
   continuation gate has not been run for this candidate.
4. **Check the m = 9 share under a different prompt.** The census here comes
   from one public fixture. The effect is proportional to the m = 9 share, so
   prompt sensitivity of that share directly scales the round-weighted figure.
