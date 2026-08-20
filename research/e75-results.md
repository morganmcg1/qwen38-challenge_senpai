# E75 — bank `pbfit`, and price it on the crown table

`harness=local` unless a row says `harness=ranked`. Nothing here is a ranked
score and none of it may be converted into one.

**Outcome: `pbfit` does not ship.** The shipped arm is `ship`, the uniform
price. `pbfit` and the whole `DepthPrice` machinery stay in the tree as a
selectable research arm.

## Identity

| field | value |
|---|---|
| base (r3 pin) | `9ec6e087c17963a5223454c265cd696fb1fa6228` |
| branch | `qwen-thorfinn/e75-bank-pbfit-and-price-it-on-the-crown-table` |
| rung B / rung D measured at | `74c262513cfc6d6b7a6694f09b21bc4eda687b5e` |
| host | Apple M4 Pro, GPU `applegpu_g16s`, 48 GiB |
| ranked host | M5, `applegpu_g17s` — a different GPU generation |
| W&B group | `e75-bank-pbfit-and-price-it-on-the-crown-table` |

The candidate surface diff against the r3 base is:

```
$ git diff --stat 9ec6e087 HEAD -- Sources/ Vendor/ mtp-head/ mtp-head.manifest.json
 Sources/MLXFastModel/Qwen36MTPBlockSession.swift | 52 ++++++++++++++++++++++++++++++++++++++++++++--------
 1 file changed, 44 insertions(+), 8 deletions(-)
```

It is **not empty**. Filtering the diff down to non-comment lines leaves exactly
one change, the vector literal:

```
-    internal static let measuredRawDepthPrice: [Double] = []
+    internal static let measuredRawDepthPrice: [Double] = [
+        0.26300121724709807, 0.29195567495854047, 0.34642143034825884,
+        0.40231023217247086, 0.63287276451077956, 0.43601634825870655,
+        0.35457813598673293, 0.42510483416251998,
+    ]
```

Every other added or removed line is a doc comment. The arm enum line reads
`.ship` and does not appear in the diff at all: base `9ec6e087` already ships
`.ship`, so the line matches the base byte for byte and stays unchanged context.

The executed schedule on this branch is therefore identical to the base. No
behaviour reaches the scored worker from this branch, because
`measuredRawDepthPrice` is read only by `makeMeasuredDepthPrice()`, which only
the `pbfit` arm calls.

## Rung A — the shipped arm is `ship`

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:894` is
`internal static let depthPriceArm: DepthPriceArm = .ship`.

`Tests/MLXFastTests/QwenMTPDepthPriceTests.swift` pins it. Thirteen tests in the
`E68 depth price` suite pass, including the six that pin the executed vector and
the one-ulp association. Two tests changed:

- `the shipped arm is pbfit` → `the shipped arm is ship`. It now pins
  `.ship` and compares `depthPrice` against `makeUniformDepthPrice()`.
- `the shipped arm shortens the walk against the flat price` →
  `the pbfit arm shortens the walk against the flat price`. The body is
  unchanged; it always built both price objects directly and never read the
  shipped arm. Only the name was wrong once `ship` became the default.

`swift test --force-resolved-versions`: 677 passed, 9 failed. All 9 failures are
the known pre-existing set (`theQwenMTPTrackIsArmedOnQwen38`,
`theCheckedInDeclarationSelectsThePinnedHead`,
`qwen36ConfigContractDigestMatchesTheReferenceManifest`,
`startupMemoryPolicyKeepsRanked128GiBProfile`,
`theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`,
`theSeededCalibrationExpectationMatchesItsRecordedProvenance`,
`participantDocsExposeDefaultCLIInstallDirectory`,
`contestantDocsCommandBlocksKeepTheDependencyGraphFrozen`,
`submissionStaticReviewPromptCoversMeasurementStructureExploitation`). None sits
in the depth-price suite.

Gates against base `9ec6e087`:

```
assignment scope OK: 1 submitted path(s)
editable budget OK: source=2471340/3000000 growth=1969/262144 files=154
PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only
```

## Why this vector does not ship

`pbfit` is a real, reproduced −3.5 % on this host. It is still the wrong thing
to ship, for three independent reasons, and any one of them is sufficient.

**It is fitted to one kernel dispatch table, and that table is not the ranked
one.** The vector's shape comes from the measured QMV width curve: the step into
verify width 6 costs 27.3 ms against 13.4 ms for the step into width 5, so the
price is steep at position 5 and the scheduler parks rounds on width 5 instead.
Rung B swapped in the crown table — an eight-line dispatch diff, nothing else —
and the cliff did not flatten. It **relocated** to the 4→5 boundary and grew.
Rung D then measured the same swap end to end: `pbfit` is −3.457 % on our table
and **+0.327 %** on the crown table, an interaction of **+3.784 pp** against a
0.119 % session null. The mechanism the vector exploits is a property of one
dispatch table, not of the schedule. The shape does not survive a table change,
and the shipped constant must not depend on a table the campaign is actively
trying to change.

**The arm-3 receipt came back rejected, and an independent competitor
reproduced the level without it.** Submission `2da69933-5202-4e0d-b336-
c75945a45b9e` carried this vector and scored **3.21125713**; it was rejected.
Receipt `ac6ec0c7` — an independent competitor replication that does not carry
this vector — scored **3.20293**. The two are 0.26 % apart, which is inside the
spread this pool shows between ranked runs. The ranked evidence therefore does
not distinguish "the vector helped" from "the vector did nothing", and the
campaign has no ranked receipt that isolates a `pbfit` win. Shipping a fitted
eight-number constant on that basis buys no measured ranked value and costs a
permanent maintenance obligation: every future kernel-table change silently
invalidates it.

**The schedule and the table form a closed loop, and this vector closes it the
wrong way.** The depth price decides which verify widths the scheduler visits;
the dispatch table decides what those widths cost. Fitting the price to the
current table locks the scheduler onto the cells that table happens to make
cheap, which then makes any table change look worse than it is — exactly the
+3.784 pp interaction rung D measured. Alphonse's ranked-versus-local
comparison shows the loop is already mistuned across hosts: the marginal
cost ratio per draft is **0.2136 ranked** against **0.4023 local**, so this
host charges about **1.9×** more per extra draft than the ranked host does. A
price fitted here over-prices depth there. The uniform `ship` price does not
couple to the table at all, so a table experiment measures the table. Keeping
`pbfit` selectable preserves every measurement it enables, without letting a
host-specific fit ride into the ranked candidate.

## Rung B — the crown dispatch table on the local host

Reproduce:

```bash
research/e75_rungB_session.sh \
  ours:e75-rB-a1 crown:e75-rB-a2 crown:e75-rB-a3 ours:e75-rB-a4 \
  ours:e75-rB-a5 crown:e75-rB-a6 crown:e75-rB-a7 ours:e75-rB-a8 \
  --widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10 --skip-stock
python3 research/e75_rungB_analyze.py
```

| field | value |
|---|---|
| instrument | E68 rung-1, unchanged |
| design | balanced mirrored palindrome, 4 legs per arm |
| `ours` cell | `<T,5,5>+<T,6,6>+<T,9,5>`, `NA <= 6` |
| `crown` cell | `<T,5,3>+<T,6,3>+<T,9,3>`, `NA <= 4` |

The `crown` arm asserts both patched files hash to the exact upstream `bfab0de`
sha256 before it is allowed to build, at `research/e75_arms.py:59-60`:

```
quantized.h    75d45143959eb3bd7223875da4dbe15ce5be3d1cf45871e010817b1e5249f281
quantized.cpp  350de46828265271e504c93d009a3b3e8b05c83047666be7fc0de51ded29b6bb
```

All four crown legs recorded `crown_bytes_verified: true`. Each leg's binary
probe confirmed its arm's routing is "present and exclusive".

### The nine-row measured width curve

The two tables differ at exactly three dispatch cells, so the other six measured
widths are a negative control. They behave:

| M | ours (ms) | crown (ms) | crown − ours | | role |
|---:|---:|---:|---:|---:|---|
| 1 | 60.699 ±0.289 | 60.559 ±0.924 | −0.140 | −0.23 % | null |
| 2 | 65.318 ±0.191 | 65.273 ±1.033 | −0.045 | −0.07 % | null |
| 3 | 72.266 ±0.197 | 72.059 ±0.082 | −0.207 | −0.29 % | null |
| 4 | 82.133 ±0.130 | 82.101 ±0.068 | −0.032 | −0.04 % | null |
| **5** | **95.539 ±0.099** | **119.836 ±0.048** | **+24.297** | **+25.43 %** | **CHANGED** |
| **6** | **122.868 ±0.145** | **128.328 ±0.112** | **+5.460** | **+4.44 %** | **CHANGED** |
| 7 | 138.354 ±0.135 | 138.353 ±0.078 | −0.001 | −0.00 % | null |
| 8 | 148.852 ±0.089 | 148.784 ±0.030 | −0.068 | −0.05 % | null |
| **9** | **163.663 ±0.170** | **185.563 ±0.103** | **+21.900** | **+13.38 %** | **CHANGED** |

Measured session null, from the six unchanged cells: **max 0.286 %, mean
0.112 %**, and **0.045 %** at the wide cells 7 and 8. Width 7 agrees to
**1 microsecond** between arms. The changed cells move 4.4 % to 25.4 %, so the
signal-to-null ratio at width 5 is about 90.

### The per-cell step table: the cliff moves, and it grows

| M | ours step (ms) | crown step (ms) | delta |
|---:|---:|---:|---:|
| 5 | 13.406 | **37.735** | **+24.329** |
| 6 | **27.329** | **8.492** | **−18.837** |
| 7 | 15.486 | 10.025 | −5.461 |
| 9 | 14.811 | 36.779 | +21.968 |

The advisor's rung B prior was that `<T,6,3>` "should flatten or remove the
inversion", with the width-6 step falling "well below 27.308 ms" and landing
"near or below the width-5 step".

**The literal prediction is correct.** The width-6 step collapses from 27.329 ms
to 8.492 ms and is far below the width-5 step.

**The inference from it is wrong.** The cliff did not flatten; it relocated from
the 5→6 boundary to the 4→5 boundary and grew. Reaching width 6 from width 4
costs 40.735 ms on our table and **46.227 ms** on the crown table. The crown
table is 5.5 ms *more* expensive to reach width 6, not less.

On this host the crown table is uniformly worse at all three cells it changes.

### Pre-registered versus measured

Committed in `research/e75-artifacts/e75-predictions.json` before the session
ran.

| M | predicted | measured | error |
|---:|---:|---:|---:|
| 5 | 122.314 | 119.836 | −2.478 (−2.03 %) |
| 6 | 129.065 | 128.328 | −0.737 (−0.57 %) |
| 9 | 186.003 | 185.563 | −0.440 (−0.24 %) |

Three cells, three correct signs, all within 2.1 %. The pre-registered
falsification threshold was "crown width 5 below about 110 ms falsifies the sign
flip". Measured 119.836 ms, so the sign flip is **not** falsified.

### The re-priced 2x2

Same two-layer model, `decode = Σ n(w)·S[w] + a·rounds + b·rows`, with
`a = 4.942` ms/round and `b = 8.697` ms/row, and the deterministic E68 per-arm
histograms. Only `S` changes: modelled crown cells are replaced by measured
ones.

| cell | pre-registered (s) | re-priced (s) | shift |
|---|---:|---:|---:|
| O-ship | 16.106 | 16.107 | +0.01 % |
| O-pbfit | 15.538 | 15.538 | +0.00 % |
| C-ship | 17.143 | 17.098 | −0.26 % |
| C-pbfit | 17.274 | 17.155 | −0.69 % |

| quantity | pre-registered | re-priced |
|---|---:|---:|
| pbfit on our table | −3.525 % | **−3.533 %** |
| pbfit on crown table | +0.77 % | **+0.330 %** |
| table effect at ship | +6.44 % | **+6.153 %** |
| interaction | +4.29 pp | **+3.863 pp** |

The crown table costs `ship` an extra **991 ms** over the 512-token window, but
costs `pbfit` an extra **1617 ms**. The 626 ms difference is the interaction.
`pbfit` is hurt **63 % more** by the crown table than `ship` is, because it
parks 42 of its 85 rounds on width 5 — the single cell the crown table charges
+24.297 ms more for.

The re-priced `+0.330 %` is smaller than the pre-registered `+0.77 %`, and it is
the same size as the two-layer model's own out-of-sample error (−0.32 % at width
1). The model therefore **cannot** separate "neutral" from "slightly harmful" on
the crown table. What it *can* say, far outside any error bar, is that `pbfit`'s
−3.5 % win is **destroyed** on the crown table: a swing of +3.86 pp.

### Harness divergence, both sides measured

One identical eight-line diff:

| harness | measurement |
|---|---|
| `harness=ranked`, receipt `9b241879` | crown table **0.298 % faster** (plutarch-corrected scoring-prompt mean; 8/8 prompts faster, sign test p = 0.0039) |
| `harness=local`, rung B | crown table **+25.43 % / +4.44 % / +13.38 %** slower at widths 5 / 6 / 9; modelled e2e **+6.15 %** slower |

The register-cliff table gives a coherent mechanism.

| IPG | g16s registers (local M4 Pro) | g17s registers (ranked M5) |
|---:|---|---|
| 2 | 70 | 83 |
| 3 | 93 | 90 |
| 4 | 94 | 91 |
| 5 | 95 | 98 |
| 6 | 96 + 16 B spill | 111 |

At M = 5 our table runs one group of 5 and the crown runs `[3, 2]`. On g16s that
trades 95 registers for 93 + 70 — almost no occupancy gain — while paying for a
second dispatch, so the split loses badly. On g17s it trades 98 for 90 + 83,
near the occupancy cliff, so the split can pay.

At M = 6 our table runs one group of 6, which on g16s costs a **16 byte** spill
at 96 registers. That spill is cheap, and it is still cheaper than the crown's
second dispatch, so `[3, 3]` loses here too. On g17s IPG 6 needs 111 registers
against 90 for IPG 3, which is a real cliff, so `[3, 3]` should win there.

This is a local measurement of a local host, reported beside the ranked number
and not converted into one.

## Appendix — rung D, measured after the rung was cancelled

**Status, stated plainly.** The advisor cancelled rung D at 14:48:55Z. The
session job had started at 13:25:17Z and finished at about 14:57Z, exit 0, zero
failed legs; its last leg began at 14:47:00Z. The GPU cost was already spent
before the cancellation was visible to me, so the evidence is banked here as an
appendix. **No further rung-D work was done after the cancellation**, and
nothing in this appendix is required by the r3 decision — the decision rests on
rung B, the rejected arm-3 receipt, and the shipped-arm change.

It is reported because it supplies the one thing rung B could only model: the
measured end-to-end `C-ship − O-ship` local calibration that sits beside ranked
receipt `9b241879`, plus a positive control with a known answer.

Reproduce:

```bash
python3 research/e75_rungD_analysis.py \
  --leg ours-ship:e75-rD-warmup:discard \
  --leg ours-ship:e75-rD-d1  --leg ours-pbfit:e75-rD-d2 \
  --leg crown-ship:e75-rD-d3 --leg crown-pbfit:e75-rD-d4 \
  --leg crown-pbfit:e75-rD-d5 --leg crown-ship:e75-rD-d6 \
  --leg ours-pbfit:e75-rD-d7 --leg ours-ship:e75-rD-d8
```

2x2 over {kernel table} × {depth price}, mirrored palindrome, one discarded
warmup leg, four prebuilt cells, all inside one thermal session.

### Cell means

| cell | n | s/token | spread | ratio | decode s |
|---|---:|---:|---:|---:|---:|
| O-ship | 2 | 0.031449083 | 0.087 % | 2.36277 | 16.102 |
| O-pbfit | 2 | 0.030361750 | 0.119 % | 2.44243 | 15.545 |
| C-ship | 2 | 0.033356674 | 0.078 % | 2.22234 | 17.079 |
| C-pbfit | 2 | 0.033465600 | 0.047 % | 2.21515 | 17.134 |

Measured session null, worst within-cell spread: **0.119 %** (E68 session null
0.143 %).

| quantity | value |
|---|---:|
| main effect of kernel table | **+8.108 %** |
| main effect of `pbfit` | −1.510 % |
| **interaction** | **+3.784 pp** |
| `pbfit` on our table (positive control) | **−3.457 %** |
| `pbfit` on crown table | **+0.327 %** |
| table at ship | +6.066 % |
| table at pbfit | +10.223 % |

The positive control has a known answer from E68: **−3.500 %**. Measured
−3.457 %, an error of **+0.043 pp**. The design could have silently succeeded
and did not.

### Prediction, committed before the session

`research/e75-artifacts/e75-rungD-prereg.json`.

| cell | predicted | measured | error |
|---|---:|---:|---:|
| O-ship | 0.031459758 | 0.031449083 | −0.03 % |
| O-pbfit | 0.030348178 | 0.030361750 | +0.04 % |
| C-ship | 0.033395414 | 0.033356674 | −0.12 % |
| C-pbfit | 0.033505513 | 0.033465600 | −0.12 % |

| effect | predicted | measured | error |
|---|---:|---:|---:|
| pbfit on ours | −3.533 | −3.457 | +0.076 pp |
| pbfit on crown | +0.330 | +0.327 | −0.003 pp |
| table at ship | +6.153 | +6.066 | −0.087 pp |
| interaction | +3.863 | +3.784 | −0.079 pp |

Four cells and four effects, every sign correct, every cell within 0.12 %. The
rung B model predicted the rung D measurement.

### The harness pair

| harness | measurement | value |
|---|---|---:|
| `harness=ranked` | receipt `9b241879`, plutarch-corrected scoring mean | **−0.298 %** |
| `harness=local` | this session, C-ship against O-ship, n=2 per cell | **+6.066 %** |

Divergence **+6.364 points**, from one identical eight-line dispatch diff. The
two are reported side by side and are **not** converted into one another.

### Why this pair is a clean calibration

The verify-width histograms are identical between C-ship and O-ship: largest
per-width round difference is **0.0 rounds**. The schedule does not react to the
table swap, so the whole local table effect is per-cell cost and the pair
isolates a single mechanism.

Fixed-width round latency is arm-independent, which is the predictor's
assumption: worst `|delta|` at a shared width with n ≥ 3 is **0.57 %** on our
table and **0.44 %** on the crown table.

### Exactness and gates

Every timed leg recorded `cool_gate_passed_real_gate=true`,
`gate_qualified_for_timing=true`, `stale_metallib_warnings=0`,
`parity_all_ok=True`, `residual_divergence_count=0`, `all_tokens_matched=True`,
and full row closure (567/567 for `ship`, 550/550 for `pbfit`).

**One emitted-stream digest across all four cells and all nine legs:**
`87ce831f40d82144`. Neither the kernel table nor the depth price changes the
tokens.

### Binary witness matrix

Four cells must produce exactly two distinct `__text` images and two distinct
`__cstring` images, because the crown patch is byte-length preserving and edits
only a kernel source string, so it cannot change host machine code.

| cell | `__text` | `__cstring` |
|---|---|---|
| ours-ship | `ef5171917447d347` | `0a3d8809e94f078e` |
| ours-pbfit | `f0f2674fe36d53b9` | `0a3d8809e94f078e` |
| crown-ship | `ef5171917447d347` | `756cf5c9435aafdf` |
| crown-pbfit | `f0f2674fe36d53b9` | `756cf5c9435aafdf` |

Distinct `__text` 2 (want 2), distinct `__cstring` 2 (want 2), distinct pairs 4
(want 4). Read back from the leg that actually ran, not from the build log.

The first four-cell prebuild **failed** this witness assertion (exit 6,
`__text takes 3 values`): `crown-ship` was a stale incremental link whose
`__TEXT,__text` was 36 bytes shorter than `ours-ship` despite identical Swift
source. All four cells were rebuilt in reverse order; three reproduced their
first-build digests exactly and only `crown-ship` moved, to the required value.
The rejected digest was `d568e9ec…`. Full record in
`research/e75-artifacts/e75-rungD-cell-provenance.json`.

## Known issue — `crossrow_na_max` is not build state

`summary.json` reports `crossrow_na_max: 4` on **all eight** rung B legs,
including the four `ours` legs whose build has `NA <= 6`. The field is the
hardcoded constant `CROSSROW_MAX_INPUTS_PER_GROUP` at
`research/qmv_cost_curve_summary.py:761`; it is a model-law parameter, not build
state, and `qmv_na_compare.py:145` labels it "law NA_max". It now disagrees with
the build on half the legs and will mislead anyone who reads it as provenance.

Not fixed here: `qmv_cost_curve_summary.py` is shared by E49, E54, E59 and E68,
and silently changing a field other agents parse mid-campaign is riskier than
recording it. The authoritative per-leg record is correct and is what this
report uses: `arm_patch.na_max` is 6 for `ours` and 4 for `crown`, with the full
dispatch map and both file digests.

## Banked artifacts

| path | content |
|---|---|
| `research/e75-artifacts/e75-rungB-legs.json` | raw per-leg rung B: arm patch, dispatch map, source digests, binary probe, GPU gate, per-width table cost, W&B run |
| `research/e75-artifacts/e75-rungD-legs.json` | raw per-leg rung D: score metrics, gate flags verbatim, stream digest, width histogram, per-width round latency |
| `research/e75-artifacts/e75-rungD-cell-provenance.json` | four-cell binary witness matrix and the rejected stale link |
| `research/e75-artifacts/e75-predictions.json` | rung B pre-registration |
| `research/e75-artifacts/e75-rungD-prereg.json` | rung D pre-registration |
| `research/e75-artifacts/e75-rungB-curve.txt` | rung B analyzer output |
| `research/e75-artifacts/e75-rungA-shipped-arm.json` | rung A exactness leg |
| `research/e75_crown_table.patch` | the eight-line crown dispatch diff, kept for E78 |
| `research/e75_arms.py` | arm definitions and the upstream digest assertion |
| `research/e75_bank_artifacts.py` | rebuilds both banked JSON bundles from the run directories |

Bulky run directories stay outside Git under `.mlxfast-private/e75-legs/`,
`.mlxfast-private/qmv-curve/` and `.mlxfast-private/e75-e2e/runs/`.

## Suggested follow-ups, not implemented

1. **Cap 8 without the streak gate.** The promoted frontier removed the streak
   gate and set a flat width cap of 7. Cap 8 without the gate is untested and is
   one flag away.
2. **Refit the price on the promoted table, if a non-uniform price is ever
   wanted again.** Rung D shows the fit transfers as a −3.5 % → +0.3 % swing
   across one eight-line table diff, so any refit must be re-measured on the
   table it will ship against, and ideally on the ranked host.
3. **Resolve the marginal-ratio gap.** 0.2136 ranked against 0.4023 local is a
   1.9× discrepancy in what a draft costs. Until it is explained, every local
   depth-schedule result carries a known transfer risk that this experiment
   could only bound, not remove.
4. **Give `crossrow_na_max` a build-derived sibling field** rather than editing
   the shared law constant, so provenance and model law stop sharing one name.
