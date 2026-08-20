# E75 rung B — the crown dispatch table on the local host

`harness=local`. This is not a ranked score and none of it may be converted to one.

Reproduce with:

```bash
research/e75_rungB_session.sh \
  ours:e75-rB-a1 crown:e75-rB-a2 crown:e75-rB-a3 ours:e75-rB-a4 \
  ours:e75-rB-a5 crown:e75-rB-a6 crown:e75-rB-a7 ours:e75-rB-a8 \
  --widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10 --skip-stock
python3 research/e75_rungB_analyze.py
```

## Identity

| field | value |
|---|---|
| base | `432eba00db0b194731a68202059ce5bfb158c1e8` (assignment pin) |
| branch head | `74c262513cfc6d6b7a6694f09b21bc4eda687b5e` |
| instrument | E68 rung-1, unchanged: `--widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10 --skip-stock` |
| host | Apple M4 Pro, `g16s` |
| design | balanced mirrored palindrome, 4 legs per arm |
| `ours` cell | `<T,5,5>+<T,6,6>+<T,9,5>`, `NA <= 6` |
| `crown` cell | `<T,5,3>+<T,6,3>+<T,9,3>`, `NA <= 4` |

The `crown` arm asserts both patched files hash to the exact upstream `bfab0de`
sha256 before it is allowed to build:

```
quantized.h    75d45143959eb3bd7223875da4dbe15ce5be3d1cf45871e010817b1e5249f281
quantized.cpp  350de46828265271e504c93d009a3b3e8b05c83047666be7fc0de51ded29b6bb
```

All four crown legs recorded `crown_bytes_verified: true`. Each leg's binary
probe confirmed its arm's routing is "present and exclusive".

## The negative control makes the null a measurement, not an assumption

The two tables differ at exactly three dispatch cells. The other six measured
widths must therefore show no arm effect. They do:

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

## The inversion does not flatten. It moves, and it gets worse.

Marginal cost of the step into each width:

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
the 5→6 boundary to the 4→5 boundary and grew. Cost of reaching width 6 from
width 4 is 40.735 ms on our table and **46.227 ms** on the crown table. The
crown table is 5.5 ms *more* expensive to reach width 6, not less.

On this host the crown table is uniformly worse at all three cells it changes.

## Pre-registered predictions versus measurement

Committed in `research/e75-artifacts/e75-predictions.json` before the session ran.

| M | predicted | measured | error |
|---:|---:|---:|---:|
| 5 | 122.314 | 119.836 | −2.478 (−2.03 %) |
| 6 | 129.065 | 128.328 | −0.737 (−0.57 %) |
| 9 | 186.003 | 185.563 | −0.440 (−0.24 %) |

Three cells, three correct signs, all within 2.1 %. The pre-registered
falsification threshold was "crown width 5 below about 110 ms falsifies the sign
flip". Measured 119.836 ms, so the sign flip is **not** falsified.

## Re-pricing the rung C 2x2 with the measured curve

Same two-layer model, `decode = Σ n(w)·S[w] + a·rounds + b·rows`, with
`a = 4.942` ms/round and `b = 8.697` ms/row, and the deterministic E68 per-arm
histograms. Only `S` changes: modelled crown cells are replaced by measured ones.

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

### Mechanism, in absolute milliseconds

The crown table costs `ship` an extra **991 ms** over the 512-token window, but
costs `pbfit` an extra **1617 ms**. The 626 ms difference is the interaction.
`pbfit` is hurt **63 % more** by the crown table than `ship` is, because it
parks 42 of its 85 rounds on width 5 — the single cell the crown table charges
+24.297 ms more for.

### Honest limit on this number

The re-priced `+0.330 %` is smaller than the pre-registered `+0.77 %`, and it is
the same size as the two-layer model's own out-of-sample error (−0.32 % at width
1). The model therefore **cannot** separate "neutral" from "slightly harmful" on
the crown table.

What it *can* say, far outside any error bar, is that `pbfit`'s −3.5 % win is
**destroyed** on the crown table: a swing of +3.86 pp. That is the
decision-relevant fact and it does not depend on the residual's sign.

## Harness divergence, both sides now measured

One identical eight-line diff:

| harness | measurement |
|---|---|
| `harness=ranked`, receipt `9b241879` | crown table **0.298 % faster** (plutarch-corrected scoring-prompt mean; 8/8 prompts faster, sign test p = 0.0039) |
| `harness=local`, this session | crown table **+25.43 % / +4.44 % / +13.38 % slower** at widths 5 / 6 / 9; modelled e2e **+6.15 % slower** |

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

This is a local measurement of a local host. It is reported beside the ranked
number, not converted into one.

## Instrumentation defect found, not fixed

`summary.json` reports `crossrow_na_max: 4` on **all eight legs**, including the
four `ours` legs whose build has `NA <= 6`. The field is the hardcoded constant
`CROSSROW_MAX_INPUTS_PER_GROUP` in `research/qmv_cost_curve_summary.py:761`; it
is a model-law parameter, not build state, and `qmv_na_compare.py:145` labels it
"law NA_max". It now disagrees with the build on half the legs and will mislead
anyone who reads it as provenance.

The authoritative per-leg record is correct and is what this report uses:
`e75-rB-aN-leg.json` carries `arm_patch.na_max` = 6 for `ours` and 4 for
`crown`, with the full dispatch map and both file digests.

Not fixed here: `qmv_cost_curve_summary.py` is shared by E49, E54, E59 and E68,
and silently changing a field other agents parse mid-campaign is riskier than
reporting it.
