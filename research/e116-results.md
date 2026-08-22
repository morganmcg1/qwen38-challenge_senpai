# E116: the measured transfer from wide-QMV kernel percent to leg seconds

`harness=local` everywhere in this file. No number here is a ranked or official
score. Every timed leg carries `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false`, and
records entry and exit GPU temperature. Timing is ungated under the standing
counterbalanced exception.

Host `ip-10-231-2-12.ec2.internal`, Apple M4 Pro, `AGXG16SDevice`, 48 GiB.
Base `67fedb4adb4cb0ec757f870ec8093617ca1e5620`.
Worker `4b90ff2251d93714bf95699c4d67540663492ed8c99b5acc31558ef0bc445e3c`.

## The question

Every kernel arm in this campaign is published as a percent of wide-QMV
**kernel** time. To decide whether an arm ships we convert it to a percent of
**absolute candidate MTP seconds per token** by multiplying two coefficients
that were never measured:

```
arm % of wide-QMV kernel time
  x (wide-QMV share of the round)     assumed 0.7786
  x (round -> leg absolute transfer)  assumed 0.79
  = arm % of absolute candidate seconds per token
```

The composed coefficient is `0.615`. E116 measures it by injecting a known,
bit-exact wide-QMV dose into the round and reading the leg.

## Headline

**One microsecond of isolated wide-QMV kernel time added to a decode round adds
exactly one microsecond of that round's contribution to the leg.**

```
alpha x beta = 1.000   95 % CI [0.963, 1.038]
```

Nothing is absorbed and nothing is amplified. The round has no overlap slack,
and the leg loses nothing on the way out.

## Method

A per-round GPU dose, gated by the integer environment variable
`MLX_E116_DOSE`, read once outside every timed window.

- The dose weight is allocated once inside `begin(seedTokens:)`, which is
  untimed. It is a synthetic affine 4-bit group-64 weight at the exact
  `mlp.gate_up` scored shape, `K = 5120 -> N = 34816`, with randomised
  contents. Resident bytes **100,270,080 B = 100.27 MB**, made of 89,128,960 B
  of nibbles and 11,141,120 B of scales and biases. This matches the assignment
  prediction exactly.
- **The environment variable's presence arms the allocation, not its value.**
  `MLX_E116_DOSE=0` therefore allocates and holds the same 100.27 MB while
  applying zero units. This is what makes the null arm a true null: it removes
  the residency confound by construction. Without it, `alpha > 1` would have
  had a boring third explanation, that the dose weight evicts part of the
  14.4 GB working set.
- In `generateRound`, after the round's normal work has completed and every
  token, row-ledger entry and draft decision is fixed, the dose runs
  `k = MLX_E116_DOSE` extra `quantizedMM` passes at `M = 1` against that
  tensor, forces evaluation, and discards the result. It is registered as a
  `defer` after the census `endRound` defer, so it runs inside the round.

## Rung 0a. The dead E112 arm switch is deleted

`MLX_E112_SKIP_1025_WARM`, its `Self.traceRounds` trace line and its comment
block are removed from `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`, and
the kL=1025 warm gate is restored to `if extK.dim(2) == 1024 {`. The reopener
statement stays in `research/e112-artifacts/README.md`, not in the scored
source.

## Rung 0b. The round-loop arm switch, and a resolved harness defect

The dose switch is read once per round in `generateRound`, after the round's
token and row decisions are fixed.

**Resolved harness defect: E109 v2's `round_alignment_verified = false` at
w512.** askeladd's E109 dose flag lived inside the model forward, so at 512
tokens the instrumented path saw 401 forwards of which 380 were width 1, and
only 12 of 77 rounds passed the instrumented boundary. That is exactly the
`verify_block_replayed_round_count`, and it forced E109's recovered +529.6 us
to be reported as a lower bound with only its sign trusted.

The one-line reason a future reader needs: **a flag inside the model forward is
read once per forward, and a round contains many forwards at many widths, most
of them width 1; a flag in the round loop is read once per round at the
round's realised verify width.**

Evidence, all four rung-2 legs: `round_alignment_verified = True`, 77 witness
lines for 77 rounds, **zero width-1 lines**, width fingerprint matched against
the parent's own `effective_draft_lengths`.

Artifact: `research/e116-artifacts/rung0b-round-switch-witness.json`.

## Harness defect: a realised width histogram is meaningless without its harness

Two harnesses over the same 512-token window, the same build, the same fixture
and the same head produce different realised width histograms:

| harness | rounds | realised width histogram | mean width |
|---|--:|---|--:|
| `--local-iterate` wrapper | 78 | `{2:1, 4:4, 5:5, 6:5, 7:3, 8:60}` | 7.359 |
| `mtp-timed` | 77 | `{3:1, 4:2, 5:6, 6:5, 7:7, 8:56}` | 7.377 |

**Cause.** `--local-iterate` runs a serial control leg and an MTP leg in one
process. The witness therefore contains two sessions: 512 rounds at width 1
followed by the MTP leg's rounds. `mtp-timed` runs the MTP leg alone. The
schedule is not adaptive between them in any interesting sense; the mean widths
agree to 0.02. What differs is the round count and the exact bucket assignment,
and any weight derived from one is not a weight for the other.

**Reproduction.**

```bash
research/e116_exactness_leg.sh e116x512k0 512 0        # --local-iterate
research/e116_rung2.sh                                 # mtp-timed
python3 research/e116_round_switch_witness.py e116x512k0 --json /dev/stdout
```

**Why it matters campaign-wide.** Every realised-width weight this campaign
uses came from a trace, and none of them records which harness produced it.
Finding 47's local realised distribution, E109 v2's `{3:1, 4:1, 5:2, 6:1, 7:26}`
and the standing NA weights `{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}` all sit
downstream of that choice. Rule 34 makes us name the round frame. This says we
must also name the **harness** that produced any width histogram.

I withdraw the "adaptive schedule" reading in my own first interim comment. It
was a harness confusion.

## Rung 1. The dose is real, measured and bit exact

### It dispatches, and the rate agrees with E107

Kernel `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0`, `grid=1x4352x1`,
`tg=32x2x1`, exactly `k` dispatches per round in every width bucket.

| estimator | us per dose unit |
|---|--:|
| isolated `M=1` leg, `exclusive_kernels`, 74 dispatches, min 404.75, max 430.50 | **411.86** |
| the dosed leg's own `e116_dose` census phase | 405.00 |
| in situ, `k=4` minus `k=0` round GPU busy, round weighted | 397.44 |

The isolated rate is **+0.23 %** against E107's 410.93 us at `M = 1`, well
inside the 10 % agreement band, so the injected cell is the E107 cell. Every
number in this experiment uses 411.86, the rate of the binary that ran. The in
situ over isolated ratio is 0.965, so the dose barely self-overlaps.

**Rule 37.** One dose unit is one `mlp.gate_up`-shaped affine 4-bit group-64
QMV at `M = 1`, by design, because that fixes the byte count exactly. 411.86 us
is an `M = 1` rate. It is not a scored-width rate and must not be read as one.

Artifact: `research/e116-artifacts/rung1-dose-rate.json`.

### It changes no token

Two 512-token legs, `MLX_E116_DOSE=0` and `MLX_E116_DOSE=12`, both reproduce
the pinned row digest
`719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e` over 1025
rows, with `all_tokens_matched=true`.

Three controls move the digest, so the check can fail:

| control | result |
|---|---|
| one hex digit flipped in one row value at row 512 | digest moved |
| rows 512 and 513 swapped | digest moved, so the comparison is ordered |
| a leg over a different decode window, 128 rows | digest moved |

**`MLX_E80_FORCE_DRAFTS` cannot serve as the runtime control, and that is a
property of the digest rather than a defect.** An `mtp-row:` line is emitted
once per emitted token position and twice per wrapper leg, so 64 tokens gives
128 rows and 512 tokens gives 1025 rows. The row set depends on the decode
window, not on draft depth. Forcing depth 1 leaves the digest unchanged while
the trace shows `d=1` in place of `d=4` and `d=5`. **No compliant candidate
knob can move this digest**, which is exactly the property the exactness gate
needs.

Artifact: `research/e116-artifacts/row-digest-512.json`.

## Rung 2. `alpha`: the round absorbs the whole dose and slightly more

askeladd's E109 v2 within-leg alternating design, in one process, dose
alternating round by round, endpoint `block_request_seconds[i]`. Arms `k = 0`
and `k = 4`, two legs per arm after block 0 is excluded as conditioning. Six
legs, 8.7 minutes, 86.9 s per leg.

```
alpha = d(round us) / d(injected dose us)
```

| estimator | alpha | 95 % CI |
|---|--:|---|
| 1, mean over consecutive equal-width round pairs | **1.177** | [1.114, 1.241] |
| 2, equal-width DUD/UDU triples, drift cancelled | **1.118** | [0.999, 1.237] |

`df = 2` for both. Pair difference SEM 104.2 us, triple difference SEM
195.8 us.

**`alpha < 0.90` is refused decisively.** The round is at least perfectly
serial. There is no overlap slack at the round level.

**The null arm.** Armed identically, holding the same 100.27 MB resident,
applying zero units. Its own paired difference is `+27.1 us` against the dosed
arm's `+1966.6 us`, so the round's intrinsic period-2 structure is 1.4 % of the
effect, and `alpha` is built from the difference of the two arms so even that
is removed. The identical residency also removes the eviction explanation for
`alpha > 1` by construction.

**Why `alpha > 1`.** Two readings survive. Either the marginal in-situ cost of
a wide-QMV microsecond at the realised widths exceeds the isolated `M = 1`
rate, because the resident dose weight competes with the round's own wide-QMV
traffic for bandwidth and cache; or the isolated rate understates the marginal
rate at w8. Rung 3 shows the distinction does not matter for the transfer,
because the round frame cancels.

Frame: every percent here divides `e109_v2_control_round_us = 171,384 us`. The
`k = 4` dose is 1,647.4 us, which is 0.961 % of it.

Artifact: `research/e116-artifacts/rung2-absorption.json`.

## Rung 3. `beta` and the transfer: the leg receives exactly the dose

A `k = 0, 4, 8, 12` ladder with no alternation, so every round of a leg carries
the same dose and the leg endpoint moves. Four blocks with rotation plus mirror
ordering; block 0 is conditioning. Sixteen legs, twelve in the estimate, three
per arm, 11.6 minutes.

Endpoint: **absolute candidate MTP microseconds per token**, the trusted
parent's own `parent_measured_seconds_per_token`. This is confined to the
candidate MTP leg. It is not `leg_wall_seconds`, which would have been
contaminated, because in the `--local-iterate` wrapper the dose also runs in
the serial control leg.

| arm | us/token | sd | n | vs `k=0` |
|---|--:|--:|--:|--:|
| `k=0` | 33,474.2 | 5.5 | 3 | +0.000 % |
| `k=4` | 33,719.4 | 4.4 | 3 | +0.732 % |
| `k=8` | 33,990.5 | 21.8 | 3 | +1.543 % |
| `k=12` | 34,209.9 | 8.6 | 3 | +2.198 % |

**The ladder is linear.** OLS slope `+61.96 +- 2.32` us/token per dose unit,
`R^2 = 0.9972`, residual sd 16.13 us/token. Residuals, in leg order:
`+1.2 -19.5 -7.5 -2.9 -0.0 -2.5 -8.0 +40.6 -8.9 -9.1 -0.2 +16.7`. Two
independent estimators agree: the `k=12` minus `k=0` arm-mean slope is
`+61.31 +- 1.36`, and the neighbour-averaged contrast against the bracketing
`k = 0` legs gives `+61.45`. Because the response is linear over the whole
range, the composed coefficient is not a local derivative and does not need to
be labelled as one.

**The kill rule is satisfied with room to spare.** The `k = 12` arm resolves at
`t = 2.198 / (0.0164 x sqrt(...))`; concretely its half width is 0.131 % of the
round against an effect of 2.798 %, which is over 20 sigma.

### The product is frame free

`alpha` is measured against the parent round frame and `beta` divides by the
same frame, so the round frame cancels in the product:

```
alpha x beta = slope / (dose_unit_us x R / tokens)
             = 61.96 / (411.86 x 77 / 512)
             = 1.000   95 % CI [0.963, 1.038]
```

The two factors separately, both against this ladder's `k = 0` parent
`block_request` frame of 170,881 us:

| coefficient | value | 95 % CI | prediction it is tested against |
|---|--:|---|---|
| `alpha` | 1.177 | [1.114, 1.241] | 1.00 if the round is perfectly serial |
| `beta` | 0.850 | [0.818, 0.881] | 1.00 algebraic, ~0.75 from E94 |
| `alpha x beta` | **1.000** | [0.963, 1.038] | this is the measurement |

**`beta = 0.850` sits between the algebraic 1.00 and E94's 0.75, and it is
closer to the algebraic prediction.** E94's 0.75 came from a constant residual
of 7.82 to 8.00 ms per token, which is a level, not a slope. `beta` here is a
marginal coefficient and there is no reason for a constant residual to appear
in it. The 0.150 shortfall against 1.00 is not lost leg time; it is the parent
round frame over-reporting the marginal cost by the same 17.7 % that `alpha`
reports. That is why only the product is a measurement, and why either factor
alone is a property of the frame it divides.

Artifact: `research/e116-artifacts/rung3-transfer.json`.

## Rule 34 frame register

Every round frame this experiment touches or is compared against, with the
harness that produced it.

| frame | us | harness | why it differs |
|---|--:|---|---|
| E112 512-token traced local MTP round | 160,590 to 161,800 | `--local-iterate`, traced | traced wrapper leg; the trace write costs host time per round |
| E116 rung 3 `k=0` parent block request | 170,881 | `mtp-timed`, untraced | the ladder's own frame; no serial control leg in the process |
| E116 rung 2 parent block request | 171,384 | `mtp-timed`, traced | +6.0 % on the E112 frame; both arms pay the trace so it cancels in the contrast |
| E109 v2 control print | 177,088 | `mtp-timed`, traced | the same legs as rung 2, but retains round 0; I drop it |
| **E116 tenth frame, leg-amortised round** | **222,582** | `mtp-timed`, untraced | `leg us/token x tokens / R`; parent block requests are 76.8 % of it, the rest is seed prefill and parent overhead |
| E96 anchor | 127,533 | recorded without a harness label | pre-dates the harness distinction |

A reader who quotes a percent against the wrong one of these is off by up to
74 % across this table.

**The tenth frame is the one the composed transfer coefficient must divide**,
because it is the only frame in which a percent of the round and a percent of
the leg are the same number.

## Rung 4. The measured share, the composition and the cross-check

_pending_

## Cleanup

_pending_

## Suggested follow-ups, not implemented

_pending_
