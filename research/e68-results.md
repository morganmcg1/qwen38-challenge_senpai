# E68 — the depth price is a vector, and the shipped flat price over-drafts

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"candidate_mtp_seconds_per_token","available":true,"value":0.030356223},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- Student / branch: `qwen-thorfinn` / `qwen-thorfinn/e68-schedule-against-the-new-cost-curve`
- Hypothesis and target cost: the scheduler prices every extra draft at one flat
  constant `headStepCostRatio = 0.18`. The merged dispatch table's real
  per-width cost is not flat, and `t55` + `t6` + E55 inverted its shape at the
  width-5 to width-6 step, which is where both score-setting ranked prompts
  live. Target cost is the QMV verify forward, which is 33.7 % of the measured
  candidate decode leg.
- **Decision: green locally.** `pbfit` lowers candidate MTP seconds per token by
  **3.500 %** against a **0.143 %** null bar, with a byte-identical emitted
  token stream. Rung 4 is deliberately not run; see "Deviations".
- `BASE_SHA` `6cbf1a40632ea44f4eff0406d32eddf72f50d282` /
  `UPSTREAM_SHA` `80021bc03e4b270f7dfef5b4425107bfc57b8d70` /
  candidate commit `09bdbe7ae117f50a0c4f2cafdd34d3bd20d11fbd`
- Yukon promoted frontier: `senpai/frontier-state.json` records `9d5569bb` at
  3.25187972 with source `80021bc0`. The assignment brief records a newer crown
  `9ad17378` by `Lieisyourlie` at 3.25238228 with source `bfab0de5`.
  `frontier-state.json` is stale relative to the brief. Our best official score
  is 3.23250848, with `ff73cbbd` in flight.
- Candidate build fingerprint: one worker `__TEXT` digest per arm, stable across
  repeated legs — `ship` `9b7e5fa5f3bbb866`, `pb5` `0f179f327d4541e7`,
  `pb7` `8524c5a82b426e42`, `pbfit` `9fd29c6b2ef04876`. Full Mach-O digests
  differ between repeats of the same arm, so the Swift release build is not
  byte-reproducible outside `__TEXT`.
- Submitted-surface digests: scored source
  `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` —
  `ship` `b02b19927edc462f`, `pb5` `71edb93de426e022`,
  `pb7` `6f77e40f434aa0b9`, `pbfit` `0248edae23a5f532`.
  `metallib_source_fingerprint` `f09821bdbd820b77`, identical on all 9 legs.
  No Metal source changed, so there is no generated-twin audit.
- Submitted candidate files: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
  only.
- Supporting test, tooling, and documentation files, none submitted:
  `Tests/MLXFastTests/QwenMTPDepthPriceTests.swift`, `research/e68_arms.py`,
  `research/e68_reg_census.py`, `research/e68_rung1_session.sh`,
  `research/e68_rung1_analysis.py`, `research/e68_insitu_curve.py`,
  `research/e68_apply_swift_patch.py`, `research/e68_swift_arm.py`,
  `research/e68_run_leg.sh`, `research/e68_rung3_session.sh`,
  `research/e68_wandb_log.py`, `research/e68-artifacts/*.json`,
  `research/e59_binary_probe.py` (one constant widened, see "Instrument defects").
- MTP head provenance, digest, and draft policy: organizer-pinned head,
  `uses_pinned_mtp_head = true` on all 9 legs,
  `head_run_dir_tree_sha256 = dadbfb806d80eca2...`,
  `head_safetensors_sha256 = d038fd41e2d5dab1...`. No head declaration was
  added. Offered depth 8, `mtp_depth = 8`, `non_drafting_round_count = 0`.
- Token window, fixture, reference source, harness: **512 decode tokens** after
  a 512-token seed, fixture
  `correctness_prompts/public_longcopy_gate_english_512_256.json`
  (`3d922b1a0ada04d9`), reference source
  `candidate-local-mtp-golden-rows`, **`harness=local`**. Not a ranked score.
- Exact cell: `qmv_fast_crossrow_affine4_g64_m` and its `_rbx` / `_rbx4`
  wrappers, affine 4-bit group 64, `D = 5120`, dispatch widths M = 1..10, source
  form `mlx-generated` JIT twin, `_nax` variants unchanged. Measured on M4 Pro,
  not on the ranked M5.
- Official causal path and score equation, `harness=ranked`: the scheduler runs
  only inside the candidate MTP leg. The ranked serial numerator comes from the
  runner-owned prebuilt baseline workspace, so
  `d ln(ranked baseline serial time) / dx = 0` for this edit. Any reduction in
  candidate MTP seconds per token raises every affected `raw_p`. No local
  cancellation term, including `psi_serial`, is subtracted anywhere in this
  report. `senpai/verify-ranked-score-boundary.sh` was run before rung 3.
- Assignment-scope preflight:
  `senpai/validate-assignment-scope.sh 6cbf1a40 Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
  → `assignment scope OK`.
- Editable source bytes: `source=2466476/3000000`, `headroom=533524`,
  `growth=4422/262144`, `exempt=2410/2147483648`, `files=154`.
- Scored-path reachability: four arms produced four distinct realised
  verify-width histograms in the timed candidate leg, each reproduced exactly
  across two separated legs, with four distinct worker `__TEXT` digests. `pb7`
  is 4.283 % **slower** than `ship`, so the selector can hurt and is not a
  no-op.

---

# Rung 1 — the measured marginal curve

This is the deliverable regardless of what the arms did. It is the campaign's
reference cost curve.

Whole-table QMV cell palindrome, widths 1 to 10, 21 reps, 10 inner iterations,
real 40 C gate before every measurement, session order `S T S S T S`, median of
the three `shipped` legs. Job `21ac5458`, 6/6 legs, exit 0.

| M | C(M) ms | replicate spread | step into M ms | null bar ms |
|---|---|---|---|---|
| 1 | 60.372 | 5.978 % | — | — |
| 2 | 65.377 | 3.090 % | +5.005 | 9.069 (in null) |
| 3 | 72.128 | 0.165 % | +6.751 | 3.255 |
| 4 | 82.163 | 0.431 % | +10.035 | 0.596 |
| 5 | 95.568 | 0.408 % | +13.405 | 0.839 |
| 6 | 122.876 | 0.281 % | **+27.308** | 0.688 |
| 7 | 138.314 | 0.065 % | +15.438 | — |
| 8 | 148.841 | 0.297 % | **+10.527** | — |
| 9 | 163.621 | 0.082 % | +14.780 | — |
| 10 | 271.147 | 0.166 % | +107.526 | — |

**The premise is confirmed.** The step into width 6 costs 27.308 ms; the step
into width 5 costs 13.405 ms. The difference is 13.903 ms against a 0.839 ms
null bar. The curve is non-monotone in both directions: the step into width 8
(10.527) is *cheaper* than the step into width 5.

Against the brief's modelled curve, the two decisive rows agree closely:
modelled 4→5 = 13.24 against measured 13.405 (+1.2 %); modelled 5→6 = 26.86
against measured 27.308 (+1.7 %).

## Rung 1 also closes the dispatch table with measurement

The `t789` arm routes NA = 7, 8 and 9 as single merged streams. Whole-table
cost against the shipped table:

| M | shipped ms | `t789` ms | delta | clears null |
|---|---|---|---|---|
| 7 | 138.314 | 149.368 | **+7.992 %** | yes |
| 8 | 148.841 | **183.642** | **+23.381 %** | yes |
| 9 | 163.621 | **451.747** | **+176.093 %** | yes |
| 10 | 271.147 | 271.098 | −0.018 % | no |

Widths 1 to 6 and width 10 all sit inside the null bar, maximum +0.216 %. Those
seven widths are the arm's own negative control: `t789` cannot change them and
does not.

**The shipped routing is optimal at all three widths, by measurement.** The old
table used extrapolations of 172.21 at NA 8 and 199.21 at NA 9. The measured
`S[8] = 183.642` and `S[9] = 451.747`. The +176 % at NA 9 is far outside any
extrapolation and the largest single model error this instrument has corrected.

**Per advisor ruling 3, the symmetric partition form is the correct one.**
`quantized.h:1157-1186` dispatches groups as concurrent threadgroups in `tid.x`
with the tail always `M % IPG` and always last, so a partition is an unordered
multiset and an order-dependent cost form is unphysical. Refitting
`cost = sum(S[g]) − L` on the three measured mixed widths gives
**`L = 15.191 ms`**, spread 12.3 % of the mean, with better residuals than the
asymmetric form at every width and the shipped partition selected at every
width 3 to 9.

**`C(10) = 271.147` confirms a source prediction, and it is a hard rule.**
`quantized.cpp:84` `get_qmv_batch_limit` returns 10 for `D = 5120`;
`quantized.cpp:1415` sets `vector_limit = 10`; `:1417` routes M = 10 to `qmm` or
`qmm_splitk`; and `qmv_fast_crossrow_affine4_g64_m` carries
`static_assert(M >= 3 && M <= 9)`, so no cross-row instantiation exists at 10.
Against the best QMV routing the symmetric form gives for a hypothetical width
10, `{5,5}` at 175.95 ms, `qmm` costs **+54.1 %**. The same table returns 10 on
the ranked M5 Max (gen 17, arch size `'s'`, same default branch, same
`D = 5120`). **Never widen the verify past 9.**

## Rung 1b — the NA ≥ 7 register census, compile only

`metal -O2 | metal-opt -O3`, artifact `research/e68-artifacts/e68-reg-census.json`.

| arm | entry point regs | kernel max | peak live at M7 / M8 / M9 |
|---|---|---|---|
| shipped | 202 | 144 | 108 / 104 / 129 |
| `t789` | **250** | 197 | 157 / 177 / 197 |

The advisor's predicted 162 / 182 / 202 is consistently 5 registers high, so the
law behind it is right and slightly conservative. The finding the extrapolation
did not have: routing all three raises the shipped **entry point** from 202 to
250, a +48 dose paid by every width. E27 was reverted for +2.

---

# Rung 2 — what the scheduler actually knows

Zero GPU. Line numbers are pre-patch, in
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`.

- `costModelDepth` at `:807-849`
- `headStepCostRatio = 0.18` at `:738`
- `sdpaWidthWallDepthCap = 5` at `:770`
- `segmentedVerifyDepthCap = 8` at `:777`
- `segmentedStreakGate = 2` at `:805`
- `snapshotScheduleSignal` at `:864-876`
- `declaredRows = draftCount + 1` at `:1366-1375`

The exact walk:

```
reach = 1.0; expected = 0.0; depth = 0
while depth < cap {
    p = positionAcceptEMA[depth]            // clamped at depth 0 (margin/2) and 1 (margin/3)
    reach *= p
    threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)
    guard reach > threshold else { break }
    expected += reach
    depth += 1
}
```

**Answers to the three questions.**

1. **No per-width cost exists.** The walk prices every extension at one flat
   scalar `h = 0.18`. The model is `T(d) = V + d·H` with the verify forward `V`
   assumed flat in width.
2. **The old curve's shape enters nowhere.** It enters only through the level:
   `h = 0.18` is an end-to-end fit against ranked receipts. A single scalar
   cannot carry a shape.
3. **`sdpaWidthWallDepthCap = 5` is a depth cap, not a width cap**, and it does
   not bind before the inversion. `costModelDepth` bounds the returned draft
   count, and a round declares `draftCount + 1` rows. Depth 5 is verify width 6,
   and depth 8 under the streak gate is verify width 9. The 4→5 and 5→6 verify
   steps are the depth-3 and depth-4 extensions, strictly inside the default
   cap. **This is a pricing question, not a cap question.** Ledger 201(K) item 5
   reads the cap as a verify width; it is a depth, and the shipped default cap
   expresses exactly verify width 6.

**E56 reconstruction, verified without reading the branch.** Per advisor ruling
1 I did not fetch `e2bd7e61`. The generalised per-position form reproduces every
published E56 constant: `withinTier = 0.15946667`, depth-3 marginal
`0.32373329`, `cumulative[3] = 1.47840002`, crossing coefficient `0.21897544`,
crossing `p = 0.94902841`. `pricedBoundaryWidths` element `w` prices index
`w − 2`.

**A correction to my own first rung-2 report.** Widths 6 to 9 are not a single
wide SDPA. `attentionWithCacheUpdate` splits a 6-to-9-row causal decode
attention into two SDPA calls of at most 5 rows each. Quantized projections at
those widths still ride the per-row-exact QMV dispatch as one call at width M,
so rung 1's curve does apply to the scored path. The width wall is a
correctness wall with a measured door through it, not a performance choice.

---

# Rung 3 — the arms

Job `51495aff-90d5-4491-b2b9-5a87e2563aba`, 9 legs, 0 failed, exit 0.
One discarded `ship` warm-up, then the mirrored palindrome
`ship pb5 pb7 pbfit pbfit pb7 pb5 ship`. 512 decode tokens on every leg.
Real 40 C gate on every leg; `cool_gate_passed_real_gate = true` and
`gate_qualified_for_timing = true` on all 9.

## Primary result

| arm | n | candidate MTP s/tok | same-arm spread | vs `ship` | serial s/tok | local ratio |
|---|---|---|---|---|---|---|
| `pbfit` | 2 | **0.030356223** | 0.105 % | **−3.500 %** | 0.074321 | **2.44829** |
| `pb5` | 2 | 0.030989251 | 0.009 % | −1.488 % | 0.074238 | 2.39559 |
| `ship` | 2 | 0.031457267 | 0.143 % | ref | 0.074225 | 2.35956 |
| `pb7` | 2 | 0.032804505 | 0.096 % | +4.283 % | 0.074140 | 2.26006 |

**Null bar 0.143 %**, the largest same-arm spread across the session. I use the
largest and not the nearest, per E56's non-monotone null floor. `pbfit` is 24
times the bar.

Estimator `time ~ arm + centred leg position`, ordinary least squares on the 8
timed legs, `ship` as the reference level:

```
intercept   +0.031457267
pbfit       -0.001101044   (-3.500 % of intercept)
pb5         -0.000468015   (-1.488 %)
pb7         +0.001347238   (+4.283 %)
pos_slope   -0.000003156   (-0.010 % per leg)
residual rms 0.000014092   ( 0.045 %)
```

The position slope is 0.010 % per leg against a 3.500 % effect, so there is no
drift confound. The serial leg spans 0.074049 to 0.074399 across all 8 legs, a
0.473 % spread with no arm ordering, which is the expected behaviour: the
schedule lives only in the MTP leg.

**Why the local ratio is legitimate direct evidence here**, stated explicitly as
the brief requires: a schedule change is confined to the candidate MTP leg. It
cannot touch the serial leg, and the measured serial legs confirm that. So the
local serial-to-MTP ratio does not cancel the effect. I still lead with absolute
candidate MTP seconds per token, and both are labelled `harness=local`.

## Correctness — every arm emits the identical token stream

This is the gate a schedule change must pass, because it changes which tokens
are drafted but must not change which tokens are emitted.

```
sha256(emitted 513 tokens) = da92be8a0dc02229...   identical on 9 of 9 legs
```

That includes both `pb7` legs, which lose on time. Per-leg row ledger:

| leg | arm | rounds | declared rows | reference-checked rows | accepted | rejected | rejected rows checked | divergence | max rejected-tail logit delta |
|---|---|---|---|---|---|---|---|---|---|
| warmup | ship | 78 | 567 | 567 | 434 | 55 | 55 | 0 | 0 |
| c1 | ship | 78 | 567 | 567 | 434 | 55 | 55 | 0 | 0 |
| c2 | pb5 | 87 | 558 | 558 | 425 | 46 | 46 | 0 | 0 |
| c3 | pb7 | 97 | 569 | 569 | 415 | 57 | 57 | 0 | 0 |
| c4 | pbfit | 85 | 550 | 550 | 427 | 38 | 38 | 0 | 0 |
| c5 | pbfit | 85 | 550 | 550 | 427 | 38 | 38 | 0 | 0 |
| c6 | pb7 | 97 | 569 | 569 | 415 | 57 | 57 | 0 | 0 |
| c7 | pb5 | 87 | 558 | 558 | 425 | 46 | 46 | 0 | 0 |
| c8 | ship | 78 | 567 | 567 | 434 | 55 | 55 | 0 | 0 |

`declared_rows_total == accepted + rejected + rounds` on every leg, and
`declared_rows_total == reference_checked_row_total` on every leg, so the ledger
closes. `parity_all_ok = true`, `target_cache_offset_final = 1024`
(512 seed + 512 decode), `target_tail_total == round_count`, and
`non_drafting_round_count = 0` everywhere.

## The mechanism — realised verify-width histograms

Rounds by realised verify width, one representative leg per arm. Repeats are
identical.

| arm | rounds | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | mean W | share ≥ 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ship` | 78 | 1 | 0 | 5 | 5 | 23 | 4 | 6 | 34 | 7.2692 | 56.4 % |
| `pb5` | 87 | 0 | 0 | 39 | 0 | 7 | 3 | 3 | 35 | 6.4138 | 47.1 % |
| `pb7` | 97 | 0 | 1 | 1 | 8 | 87 | 0 | 0 | 0 | 5.8660 | 0.0 % |
| `pbfit` | 85 | 0 | 0 | 5 | 42 | 5 | 0 | 7 | 26 | 6.4706 | 38.8 % |

`pbfit` does exactly what it was built to do. It moves 18 rounds off the newly
expensive step into width 6 (23 → 5) and onto width 5 (5 → 42), and it trims the
deep mode from 34 rounds to 26. `pb7` collapses the whole distribution onto
width 6 and never reaches width 7, which is why it loses.

## Rollback, repair and block latency

| arm | replayed verify blocks | share of rounds | first block s | p50 block s after first | max block s after first |
|---|---|---|---|---|---|
| `ship` | 12 | 15.4 % | 0.13256 | 0.16497 | 0.24426 |
| `pb5` | 11 | 12.6 % | 0.11712 | 0.13409 | 0.19944 |
| `pb7` | 16 | 16.5 % | 0.15964 | 0.13375 | 0.24711 |
| `pbfit` | 10 | 11.8 % | 0.13306 | 0.10438 | 0.21243 |

`pbfit` has the fewest replays, the lowest median block latency (−36.7 % against
`ship`) and a lower tail. Prefill is 0.00782 s/token for every arm, spread
0.44 %, as expected for a decode-only change.

## The finding in one number

Measured tokens per round against measured milliseconds per round, and the
marginal price of the extra tokens along the frontier:

| arm | tokens/round | ms/round | ms/token | marginal price of the extra tokens |
|---|---|---|---|---|
| `pb7` | 5.278 | 173.15 | 32.80 | — |
| `pb5` | 5.885 | 182.37 | 30.99 | 15.2 ms/token |
| `pbfit` | 6.024 | 182.85 | 30.36 | **3.5 ms/token** |
| `ship` | 6.564 | 206.49 | 31.46 | **43.7 ms/token** |

`pbfit`'s average token costs 30.36 ms. `ship` buys its last 0.540 tokens per
round at 43.7 ms each, **1.44 times that average**. `pbfit` buys its last 0.139
tokens at 3.5 ms each. **The efficient frontier crosses between `pbfit` and
`ship`, and the shipped schedule is on the wrong side of it.**

## The inversion, confirmed in situ by the strongest instrument

Pooled median round latency by realised verify width across all 8 timed legs.
This is measured inside the timed scored path.

| width | n | median ms | p25 | p75 | step |
|---|---|---|---|---|---|
| 2 | 2 | 74.849 | 74.696 | 75.001 | — |
| 3 | 2 | 76.343 | 76.277 | 76.409 | +1.495 |
| 4 | 100 | 87.919 | 87.704 | 88.453 | +11.576 |
| 5 | 110 | 104.005 | 103.614 | 104.406 | +16.086 |
| 6 | 244 | 133.871 | 133.598 | 134.355 | **+29.867** |
| 7 | 14 | 151.885 | 151.693 | 152.494 | +18.015 |
| 8 | 32 | 165.042 | 164.919 | 165.215 | +13.157 |
| 9 | 190 | 183.389 | 182.963 | 183.794 | +18.347 |

The 5→6 step is 1.86 times the 4→5 step in situ. Against rung 1's isolated cell
steps, the in-situ marginals are a near-constant **1.19×**:

| step | isolated ms | in-situ ms | ratio |
|---|---|---|---|
| 4→5 | 13.405 | 16.086 | 1.200 |
| 5→6 | 27.308 | 29.867 | 1.094 |
| 6→7 | 15.438 | 18.015 | 1.167 |
| 7→8 | 10.527 | 13.157 | 1.250 |
| 8→9 | 14.780 | 18.347 | 1.241 |

**Two instruments with no shared code agree on the shape of the curve and
differ by one scale factor on the marginals**, range 1.094 to 1.250. The
isolated cell instrument is therefore a valid, calibrated predictor of scored
marginal cost, not only of relative cell cost.

Round latency at a given width is arm-independent:

| width | ship | pb5 | pb7 | pbfit | max spread |
|---|---|---|---|---|---|
| 4 | 88.452 | 87.858 | 88.488 | 88.405 | 0.72 % |
| 5 | 104.532 | — | 104.711 | 103.881 | 0.80 % |
| 6 | 134.106 | 133.605 | 133.847 | 134.154 | 0.41 % |
| 7 | 151.768 | 152.298 | — | — | 0.35 % |
| 8 | 165.058 | 164.955 | — | 165.115 | 0.10 % |
| 9 | 183.252 | 183.334 | — | 183.717 | 0.25 % |

The arms therefore differ **only** through the width histogram. That is why the
histogram fully explains the timing, and it is the cleanest possible separation
of mechanism from noise in this harness.

## Where the win comes from

Measured round time split by width region, per leg:

| arm | rounds | sum of W ≤ 6 rounds | sum of W ≥ 7 rounds | total |
|---|---|---|---|---|
| `ship` | 78 | 4.2621 s | 7.8302 s | 12.0924 s |
| `pb5` | 87 | 4.5048 s | 7.3657 s | 11.8705 s |
| `pb7` | 97 | 12.7926 s | 0 s | 12.7926 s |
| `pbfit` | 85 | 5.6106 s | 5.9287 s | 11.5394 s |

| arm | delta W ≤ 6 | delta W ≥ 7 | net |
|---|---|---|---|
| `pb5` | +242.7 ms | −464.5 ms | −221.9 ms |
| `pb7` | +8530.5 ms | −7830.2 ms | +700.2 ms |
| `pbfit` | +1348.5 ms | **−1901.5 ms** | −553.0 ms |

`pbfit`'s win is dominated by cutting the deep mode, not by dodging the width-6
step alone. The width-6 avoidance pays for itself and the deep-mode trim is the
surplus. Note that round counts differ between arms, so this split is
directional, not a like-for-like decomposition.

## The rung-1 curve predicts the rung-3 ranking

Pricing each arm's realised histogram with the rung-1 isolated curve scaled by
the measured in-situ factor:

| arm | modelled delta vs `ship` | measured delta vs `ship` |
|---|---|---|
| `pb5` | −1.269 % | −1.488 % |
| `pb7` | +7.595 % | +4.283 % |
| `pbfit` | −4.213 % | −3.500 % |

Three signs out of three, and `pbfit`'s magnitude within 0.7 percentage points.
`pb7` is over-predicted by 1.77×.

## Projection at the two ranked prompts

Flat-`p` pricing with the pooled in-situ latency, using `p` as the per-draft
acceptance from ledger 184(B):

| operating point | `ship` depth/width | `pbfit` depth/width | round ms ship / pbfit | E[tokens] ship / pbfit | raw effect |
|---|---|---|---|---|---|
| local fixture 0.8808 | 6 / 7 | 4 / 5 | 151.885 / 104.005 | 4.939 / 3.942 | −14.204 % |
| ranked beagle 0.8351 | 5 / 6 | 4 / 5 | 133.871 / 104.005 | 4.007 / 3.601 | −13.548 % |
| ranked medicine 0.8750 | 6 / 7 | 4 / 5 | 151.885 / 104.005 | 4.858 / 3.897 | −14.625 % |
| ranked republic 0.9019 | 7 / 8 | 4 / 5 | 165.042 / 104.005 | 5.731 / 4.111 | −12.142 % |

Raw flat-`p` over-predicts, because flat-`p` fits `ship` well and `pbfit`
badly. Calibrating each arm on the one point where both the prediction and the
measurement exist:

```
local flat-p predicted decode   ship 15.7455 s   pbfit 13.5091 s
local measured decode           ship 16.1061 s   pbfit 15.5424 s
calibration factor              ship  1.0229     pbfit  1.1505
```

**Calibrated projection**, marked as an interpolation from one calibration
point:

| operating point | calibrated effect | raw |
|---|---|---|
| ranked beagle 0.8351 | **−2.763 %** | −13.548 % |
| ranked medicine 0.8750 | **−3.974 %** | −14.625 % |
| ranked republic 0.9019 | −1.182 % | −12.142 % |

Every point clears the 0.30 % minimum useful effect; beagle and medicine both
clear the 2 % target; and the calibrated pair brackets the measured local
−3.500 %. **The honest weakness is that the correction rests on a single
calibration point and assumes the `pbfit` miss factor is constant in `p`.** I
would defend the sign and the order of magnitude, not the third decimal.

**The rung-3 advance rule is satisfied on both clauses**: the best arm beats
`ship` by more than the largest same-arm spread, and the re-derived boundary
moves favourably at both `p = 0.8351` and `p = 0.8750`.

## The local fixture is in the ranked policy regime

Per advisor ruling 2, the authoritative local per-draft acceptance is
**`p = 0.8808`**, from alphonse's E65 on the shipped cap-8 gate-2
configuration: `R = 76`, `P = 495`, `A = 436`, mean depth 6.513. My `ship` legs
independently measure `R = 78`, `P = 489`, `A = 434`, `p = 0.8875` on the same
fixture at a different base, which agrees to 0.0067.

| prompt | R | P | A | mean depth | per-draft accept |
|---|---|---|---|---|---|
| ranked beagle | 107 | 485 | 405 | 4.533 | 0.8351 |
| ranked medicine | 99 | 472 | 413 | 4.768 | 0.8750 |
| **public local fixture** | **76** | **495** | **436** | **6.513** | **0.8808** |
| ranked republic | 89 | 469 | 423 | 5.270 | 0.9019 |

The fixture sits 0.0058 above medicine and 0.0211 below republic. Both the
retracted 0.9625 and the cap-7 gate-3 figure 0.9189 are wrong for the shipped
configuration. **The local A/B therefore measures the scheduler in the same
policy regime that sets the published median.** Ledger `:15480` is retracted.

The residual transfer risk is not the aggregate `p`; it is the **shape** of the
per-position EMA trajectory. The local fixture is bimodal: `ship` spends 34 of
78 rounds at width 9 and 23 at width 6. The ranked prompts, at mean depth 4.5
and 4.8, cannot have a comparable width-9 mode. Since `pbfit`'s measured win is
dominated by the W ≥ 7 region, the ranked effect is likely to be smaller than
the local one, which is what the calibrated projection also says.

---

# Corrections to my own earlier claims

## My conservation claim is false

In interim 2 I wrote that a conserving reshape can only tie or shorten the walk,
so no arm in this class can select deeper than `ship`. That is wrong, and the
advisor's algebra shows why. Writing `E(D) = sum_{j<D} reach_j`,
`H(D) = sum_{j<D} h_j` and `J(D) = (1 + E(D)) / (1 + H(D))`, the shipped
threshold is exactly the condition `J(D+1) > J(D)`, so the shipped rule is
greedy ascent on expected tokens per unit modelled round cost. Because
`1 + H(D)` sits in the **denominator**, loading cost onto early steps lowers
every later threshold. At beagle with `sum(h) = 1.44` held exactly,
`[.19,.19,.19,.19,.19,.13,.18,.18]` and `[.20,.20,.20,.20,.20,.08,.18,.18]`
both select depth 6 where flat `ship` selects 5.

The correct statement is a property of **this measured curve**, not of the arm
class: this curve does not buy depth. It is not true that no curve can. A null
rung 3 would have meant the former, not the latter — and rung 3 was not null.

## The argmax rule is a measured negative, not an open follow-up

The advisor ran argmax of `J(D)` over `D ≤ cap` on my six-leg curve. Greedy
equals argmax at beagle 0.8351, at medicine 0.8750 and at the local fixture
0.8808. The two rules differ only at `p ≥ 0.9189`, which is not an operating
point after ruling 2. The width-6 cliff is large enough that `J` never recovers
above `J(4)`, so there is nothing behind it to harvest. **Do not build the
argmax rule.**

The price lever is now closed from four directions: level bracketed above and
below (0.14, 0.15, 0.18, 0.32), order shown irrelevant by concurrency, rule
shown equivalent by argmax — and **shape shown to be a real 3.500 % lever by
rung 3**.

## The un-renormalised vector

The honest un-renormalised vector `h_d = (2.590 + ΔC_d) / C(1)` with the E1 head
step sums to 2.053 against `ship`'s 1.44, and it selects depth 3 at both ranked
prompts. That is a level statement, and the level is already measured-bracketed
with 0.18 winning, so the un-renormalised vector is not a candidate. Every E68
arm holds the total at 1.44 exactly and is therefore a pure shape test.

## What rung 3 overturns

The `segmentedStreakGate` doc comment at `Qwen36MTPBlockSession.swift:876-897`
concludes: "h is now bracketed on both sides and 0.18 is a true local optimum —
so the way to buy depth is NOT the price. It is the cap." That conclusion holds
for the **level** and is contradicted for the **shape**. A conserving reshape at
the same level buys 3.500 %.

It also resolves an apparent contradiction. `h = 0.32` shortened every draft and
lost 3 %; `pbfit` also shortens the mean width, from 7.269 to 6.471, and wins
3.5 %. Those are consistent only because what matters is **which** widths are
dropped. `h = 0.32` raises every marginal uniformly, so it drops cheap width-5
and width-9 rounds along with the dear width-6 ones. `pbfit` drops the width-6
mode specifically and keeps the cheap deep mode.

---

# Evidence

- Host: AWS Mac, Apple **M4 Pro**, 48 GiB, `startup_memory_profile = full`,
  `mlx_max_mb_per_buffer = 512`, `mlx_max_ops_per_buffer = 50`,
  `wired_residency_active = false`. **Not the ranked M5.**
- Thermal policy: real 40 C cool gate before every leg, never bypassed.
  `cool_gate_passed_real_gate = true` and `gate_qualified_for_timing = true` on
  all 9 legs. Entry temperature 40.41 to 46.85 C; across the 8 timed legs the
  entry spread is 44.95 to 46.85 C, 1.90 C. Exit 60.91 to 63.13 C.
  `MLXFAST_LOCAL_COOL_GATE` was **not** set to 0 at any point.
- Baseline and candidate commands, identical except for the arm selector:

```bash
research/e68_rung1_session.sh                       # rung 1, 6 legs
python3 research/e68_rung1_analysis.py \
  --leg shipped:e68-r1-a1 --leg t789:e68-r1-a2 --leg shipped:e68-r1-a3 \
  --leg shipped:e68-r1-a4 --leg t789:e68-r1-a5 --leg shipped:e68-r1-a6 \
  --verify-forward-s 0.0603,0.0657,0.0730 \
  --out research/e68-artifacts/e68-rung1.json

research/e68_rung3_session.sh \
  ship:e68-r3c-warmup ship:e68-r3c-c1 pb5:e68-r3c-c2 pb7:e68-r3c-c3 \
  pbfit:e68-r3c-c4 pbfit:e68-r3c-c5 pb7:e68-r3c-c6 pb5:e68-r3c-c7 \
  ship:e68-r3c-c8 --tokens 512
```

Every rung-3 number in this file is regenerated from the committed per-leg
artifacts by one command, which asserts the row ledger, the token-stream
identity and the recorded arm labels as it runs:

```bash
python3 research/e68_rung3_analysis.py \
  --leg ship:e68-r3c-warmup:discard --leg ship:e68-r3c-c1 \
  --leg pb5:e68-r3c-c2 --leg pb7:e68-r3c-c3 --leg pbfit:e68-r3c-c4 \
  --leg pbfit:e68-r3c-c5 --leg pb7:e68-r3c-c6 --leg pb5:e68-r3c-c7 \
  --leg ship:e68-r3c-c8 --out research/e68-artifacts/e68-rung3.json
```

- Cheapest real falsification gate, run before any timing:
  `Tests/MLXFastTests/QwenMTPDepthPriceTests.swift`, **8 of 8 pass**. It
  compares actual `Double` values at the touched arithmetic, not integers or an
  argmax, and it carries two positive controls that prove the comparison can
  fail: naive left-to-right accumulation of `0.18` **does** diverge from the
  tip's closed form at depths 3 to 6, and an empty `measuredRawDepthPrice`
  **does** trap instead of silently selecting a degenerate curve. The Swift file
  has not changed since that test ran (last commit touching it is `6abaf1f`;
  `e3c4a6d` and `09bdbe7` touch research scripts only).
- Positive-control verdict: **`pb5` did not lose.** I pre-registered it as the
  plumbing control that must lose, and said that if it did not, the selector was
  not reaching the scored path and every conclusion would be void. The void
  clause does not fire, because scored-path reachability is established three
  other ways (four distinct histograms, four distinct worker `__TEXT` digests,
  and `pb7` losing by 4.283 %). What failed is my flat-`p` prediction that
  `pb5` would select depth 3; it realises mean verify width 6.414. The shipped
  walk reads per-position EMAs, and on this fixture the early-position EMAs sit
  well above the aggregate 0.8808, so every arm walks deeper than flat-`p`
  predicts.
- Exact-token and row-ledger verdict: **pass on 9 of 9 legs.** Identical
  513-token emitted stream `da92be8a0dc02229`, `all_tokens_matched = true`,
  `residual_divergence_count = 0`, `max_rejected_tail_logit_delta = 0`,
  closed row ledger, `parity_all_ok = true`.
- Divergent tokens: none.
- Generated-twin audit: not relevant. No Metal source changed;
  `metallib_source_fingerprint` is identical on all 9 legs.
- Peak RAM: not separately instrumented. No head artifact was added, so the
  2 GiB `mtp-head/` cap is untouched (`exempt = 2410` bytes).
- Official status: **not submitted.** This is a local result.

| Metric | Baseline (`ship`) | Candidate (`pbfit`) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.0742252 | 0.0743208 | +0.129 % |
| MTP seconds/token | 0.0314573 | **0.0303562** | **−3.500 %** |
| local serial-relative speedup | 2.35956 | **2.44829** | +3.761 % |
| effective mean draft length | 6.26923 | 5.47059 | −12.74 % |
| mean realised verify width | 7.2692 | 6.4706 | −10.99 % |
| accepted draft rate | 0.88753 | 0.91828 | +3.465 % |
| rounds per 512 tokens | 78 | 85 | +8.97 % |
| tokens per round | 6.564 | 6.024 | −8.24 % |
| p50 block latency s | 0.16497 | 0.10438 | −36.73 % |
| replayed verify blocks | 12 (15.4 %) | 10 (11.8 %) | −2 |
| prefill seconds/token | 0.0078392 | 0.0078183 | −0.267 % |

The local score is a one-prompt directional measurement. It is not the ranked
median across eight hidden prompts.

**Identity fields.** All 9 legs share one `base_sha` `dd60d0b4`, one
`branch_commit` `09bdbe7a`, one fixture digest, one head digest, one metallib
fingerprint, one token window (512) and one offered depth (8). The only field
that varies is the arm, which maps one-to-one onto `scored_source_sha256` and
onto `worker_text_sha256`. Full Mach-O `worker_sha256` varies between repeats of
the same arm because the Swift release build is not byte-reproducible outside
`__TEXT`; this is a build-determinism observation, not an arm difference.

**Labelled inferences.** The calibrated ranked projection is an interpolation
from one calibration point on a flat-`p` surrogate of an EMA-driven walk. The
in-situ-to-isolated factor 1.19 is a measured ratio on this host only. Nothing
in this report has been transferred across hosts, widths, token windows or
harnesses without replay.

---

# Conclusion

- **What happened and why.** The scheduler prices every draft extension at one
  flat constant. The merged dispatch table's real per-width cost is strongly
  non-flat, and after `t55` + `t6` + E55 the step into verify width 6 costs 1.86
  times the step into width 5. Replacing the scalar with a measured-shape
  vector at the same total moves 18 of 78 rounds off the expensive width-6 step
  and trims the deep mode, which lowers candidate MTP time by 3.500 % while
  emitting a byte-identical token stream.
- **Evidence for the mechanism.** The realised width histogram shifts exactly
  where the curve says it should. Per-width round latency is arm-independent to
  within 0.8 %, so the histogram is the only channel. The rung-1 isolated curve
  predicts all three arm signs and `pbfit`'s magnitude to within 0.7 percentage
  points. The 5→6 inversion is now measured by three independent instruments —
  isolated cell palindrome, committed E37 trace reanalysis, and in-situ block
  latency inside the timed leg — and the two quantitative ones agree on shape
  with one 1.19× scale factor.
- **Evidence against.** `pb5` did not behave as the pre-registered control. My
  flat-`p` model mis-orders the arms and mis-predicts every realised depth,
  which means the projection machinery is the weakest link in this report, not
  the measurement.
- **Prompt and M5 transfer risk.** Moderate. The aggregate acceptance risk is
  now retired: the local fixture at 0.8808 is inside the ranked band, 0.0058
  from medicine. The remaining risk is the **shape** of the per-position EMA
  trajectory. The local fixture is bimodal with 44 % of `ship` rounds at width
  9; the ranked prompts, at mean depth 4.5 and 4.8, cannot be. Since the win is
  dominated by the W ≥ 7 region, the ranked effect should be smaller than the
  local one, which is what the calibrated projection says (−2.8 % and −4.0 %).
  M5 risk is separate and untested: the curve was measured on M4 Pro, and the
  width-6 cliff is a QMV group-shape effect whose location could differ on M5.
  `get_qmv_batch_limit` returns 10 on both hosts, so the width-10 boundary does
  transfer.
- **Smallest useful next action.** Merge `4898738e` and run rung 4 on `pbfit`:
  512-token exactness against the unchanged base on this host including
  post-EOS continuation, `--local-submit`, and the five gates. That is roughly
  30 minutes and it is the only work between this result and a submittable
  candidate.
- **Recommendation: promote to rung 4, then compose.** The mechanism is
  independent of `t55`, `t6` and E55 in code, but E56 measured schedule and
  kernel arms as 33.1 % substitutes, so the composed gain will be less than the
  sum. Measure the composition; do not add the numbers.

---

# Deviations from the brief, declared

1. **Rung 4 was not run.** The advisor's standing instruction is: "If `pbfit`
   does not lose, say so immediately and stop before rung 4; that result is
   worth more than finishing the ladder." `pbfit` won, so I stopped and
   reported.
2. **The advisor base `4898738e` is not merged.** The advisor gated that merge
   on the rung-4 chain. The branch stays byte-identical to the tree that
   produced this evidence.
3. **The `nm -a` witness with `--require-symbol` and `--forbid-symbol` did not
   run.** My leg runner records `scored_source_sha256`, `worker_text_sha256` and
   `worker_sha256` instead, giving four distinct sources mapping one-to-one onto
   four distinct worker `__TEXT` digests, stable across repeats. I argue that is
   a stronger arm witness because it covers the whole code section rather than
   one symbol, but it is my substitution and not the witness that was specified.
4. **`e2bd7e61` was not read**, per advisor ruling 1 and the launch isolation
   rule. The `pricedBoundaryWidths` arithmetic is an independent
   reconstruction, verified against all five published E56 constants.
5. **The first rung-3 session lost its control arm** to a bug in my own guard,
   which compared the `ship` arm against `base_sha` when the E68 patch installs
   the vector machinery in every arm. Six legs from that session are valid and
   agree with the relaunch to within 0.090 % on every arm. Fixed in `09bdbe7`.
6. **`arm.json` artifacts carry `predicted_depth` blocks computed at the
   retracted local `p = 0.9189`.** Every number in this report uses `p = 0.8808`.

---

# The change

The scored diff inside `costModelDepth` is two lines:

```swift
let price = Self.depthPrice
let threshold = price.marginal[depth] * (1.0 + expected) /
    price.cumulative[depth]
```

Everything else is the supporting `DepthPrice` struct, the arm constructors, and
the arm selector `internal static let depthPriceArm: DepthPriceArm = .ship`,
which is the one line an arm patches. `depthPrice` is a `let`, so no allocation
happens per round. `snapshotScheduleSignal` gains an `arm=<raw>` prefix.

Setting every `marginal[d] = 0.18` reduces the arithmetic to the shipped form
exactly, so `ship` and the priced arms run one code path.
`makeUniformDepthPrice()` reproduces the tip's **closed form**
`1.0 + Double($0) * headStepCostRatio` rather than accumulating `0.18`, because
the two differ by one ulp at depths 3 to 6 and a control that is not
bit-identical is not a control.

The winning vector, which sums to 1.44 exactly:

```
pbfit.marginal = [0.12014290579688386, 0.13336973691819140, 0.15825051194819845,
                  0.18378135596082668, 0.28910578332644965, 0.19917881598825601,
                  0.16197661758144877, 0.19419427247974499]
```

derived from the rung-1 curve as `marginal[d] ∝ H + C(d+2) − C(d+1)` with
`H = E68_VERIFY_FORWARD_KEY = 0.060300` s, then scaled so the total is
`maxDepth * headStepCostRatio = 1.44`.

**Which association produced it.** These are the values Swift's
`makeMeasuredDepthPrice` computes, `raw * (total / sum)`: form the scale factor
once, then multiply. An earlier revision of this table printed
`raw * total / sum`, the other association, which differs by one ulp at depths
0, 3 and 7. The timed legs ran Swift, so the vector above is the arm. E75 rung
A found the discrepancy while asserting bit-identity, and
`QwenMTPDepthPriceTests.measuredDepthPriceIsBitIdenticalToTimedArm` now pins
these eight doubles so a report and an arm cannot drift apart again.

---

# Suggested follow-ups, not implemented

1. **Fit the vector to the in-situ curve instead of the isolated one.** `pbfit`
   was derived from the isolated cell curve. The in-situ marginals differ from
   it by 1.09 to 1.25, which is a real shape difference of about 14 %, not only
   a level factor. A vector fitted to the pooled in-situ latency table in this
   report is free to derive and is the obvious next arm. Because `pbfit` sits at
   a 3.5 ms/token marginal against a 30.36 ms average, there is room to buy back
   some depth cheaply, which the isolated fit may be leaving on the table.
2. **Search the vector directly rather than deriving it.** The four arms are
   four points in an eight-dimensional simplex, and the measured frontier says
   the optimum is between `pbfit` and `ship`. A 5-leg bisection between those
   two vectors is one session and would place the optimum rather than bracket
   it.
3. **Raise the level now that the shape is right.** `h = 0.32` was measured
   against the flat shape. The level and shape interact, so the 0.14 / 0.15 /
   0.18 / 0.32 bracket may not hold under `pbfit`'s shape. One arm at
   `sum = 1.60` with `pbfit`'s shape would test this and is the cheapest
   remaining question.
4. **Instrument rollback and repair into the JSON payload.**
   `rollbackRoundCount` at `Qwen36MTPBlockSession.swift:156-162, 1246` is
   maintained and never read, so I had to use
   `verify_block_replayed_round_count` as a proxy. One line exposes it.
5. **Re-measure the curve on the ranked M5 before promotion of any
   shape-derived vector.** The width-6 cliff is a QMV group-shape effect. Its
   location on M5 is untested and the vector is derived directly from it.

---

# Instrument defects found, for the ledger

- `research/e54_arms.py` `SHIPPED_IPG` says `5: 3`; the live table routes M5 to
  ipg 5.
- `e54_arms.NA_ASSERT` and `NA_ASSERT_RELAXED` are stale; the live assert is
  `NA in [2, 6]`.
- `run-qmv-curve.sh:223` passes `--na-max ${MLXFAST_QMV_NA_MAX:-4}`, so
  `summary.json` records `crossrow_na_max: 4` while the live table reaches
  NA = 6.
- `research/arm_summary.py` `ROUND_RE` expects `streak_in=` and `cap=`; the live
  trace emits `m=… streak=… cap=…`, so `position_acceptance()` returns zeros.
- `research/e56_repair_census.py` is absent despite ledger 201(K).
- No rollback or repair counter reaches any JSON payload.
- `research/e59_e2e_run.sh` hard-wires `SCORED_FILES` to the two kernel files,
  which is why E68 needed its own `e68_run_leg.sh`.
- `research/e59_wandb_log.py` is hard-wired to the E59 identity, which is why
  E68 needed `e68_wandb_log.py`.
- `research/e59-results.md:186-188` names the wrong script for its artifacts.
- `research/e59_binary_probe.py` had `IPG_RANGE = range(2, 6)`. The exclusivity
  half of the probe can only reject an instantiation it enumerates, and the live
  table already routes NA = 6, so the guard could not fail for the widths under
  test. Widened to `range(2, 10)` in this branch.

---

# W&B

Project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group
`e68-schedule-against-the-new-cost-curve`. One run per leg, logged during the
session.

## Rung 1 — widths 1 to 10, 21 reps, 10 inner, order `S T S S T S`

| leg | arm | run |
|---|---|---|
| e68-r1-a1 | shipped | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/3twf8cke |
| e68-r1-a2 | t789 | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/v08x7fvh |
| e68-r1-a3 | shipped | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/6vx39ue9 |
| e68-r1-a4 | shipped | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/85g638ll |
| e68-r1-a5 | t789 | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ewfyovn5 |
| e68-r1-a6 | shipped | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ritrokbd |

## Rung 3 — 512 decode tokens, mirrored palindrome

| leg | arm | run |
|---|---|---|
| e68-r3c-warmup | ship, discarded | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/xlchw9fx |
| e68-r3c-c1 | ship | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ttkcwihh |
| e68-r3c-c2 | pb5 | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/y5a9zub5 |
| e68-r3c-c3 | pb7 | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/5t2lbul6 |
| e68-r3c-c4 | pbfit | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/jfkx6yb2 |
| e68-r3c-c5 | pbfit | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/3x6542d8 |
| e68-r3c-c6 | pb7 | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2udyghu9 |
| e68-r3c-c7 | pb5 | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/gq3cx5m6 |
| e68-r3c-c8 | ship | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/dnuigwn6 |

A superseded first rung-3 session lost its three `ship` legs to the guard bug in
deviation 5. Its six valid legs agree with the relaunch to within 0.090 % on
every arm: `pb5` 0.030997334 against 0.030989251, `pb7` 0.032834190 against
0.032804505, `pbfit` 0.030340782 against 0.030356223.

_This results file was written by an AI research agent (OpenHands) on behalf of
the Senpai campaign._
