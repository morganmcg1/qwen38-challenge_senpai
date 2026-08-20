# E78 rung 2b pre-registration

Written after rung 2a and before the rung 2b session starts. Committed so the
predictions cannot be adjusted after the legs run.

`harness=local`. Nothing here is an official or ranked score.

## What rung 2b tests

Rung 2a measured absolute time for every `(shape, M, IPG)` cell in one session.
`research/e78_price_arms.py` then priced each arm by lookup, with no further GPU
time. That pricing assumes **additivity**: that a verify round costs the sum of
its dispatches, and that an isolated cell time predicts the same cell inside a
live block.

Rung 2b measures the four arms end to end. It therefore tests additivity, and
it does so against a quantitative prediction rather than a direction.

## The derived predictions

Round counts come from rung 1, which proved the four arms produce byte identical
row ledgers, so every arm runs the same 78 verify rounds and emits the same 512
tokens. The conversion is `delta_s_per_token = 78 * delta_ms_per_round / 1000 /
512`.

| arm | derived delta ms per round, local mix | predicted delta s/token | predicted % of a_ship |
|---|---:|---:|---:|
| `a_ship` | +0.0000 | 0 | 0 |
| `d_hybrid8192` | +2.5421 | +3.873e-4 | +1.23 % |
| `c_hybrid24928` | +5.4235 | +8.262e-4 | +2.62 % |
| `b_crown` | +12.6354 | +1.925e-3 | +6.09 % |

The `a_ship` reference is the rung 1 golden leg, `mtp_seconds_per_token =
0.031591226579621434`.

## Pre-registered pass conditions

1. **Rank order.** `a_ship` is the fastest arm, and the arms order
   `a_ship < d_hybrid8192 < c_hybrid24928 < b_crown` on absolute
   `mtp_seconds_per_token`. Every gap is far above the E66 within-arm
   cross-position session null of about 0.1 %, so the order is resolvable.
2. **Sign.** Every candidate arm is slower than `a_ship`.
3. **Magnitude.** Each measured delta lands within a factor of two of the
   derived prediction.

Condition 1 alone decides the experiment. Conditions 2 and 3 decide whether the
rung 2a cell instrument may be used to price future arms without a whole-leg
session.

## What falsifies additivity

If the measured rank order disagrees with the derived order, or if a measured
delta differs from its prediction by more than a factor of two, then an isolated
cell time does not predict the same cell inside a live block. In that case the
rung 2a table stays valid as a kernel measurement, but it must not be used to
price arms, and the whole-leg session becomes the only instrument that can rank
a dispatch table.

The most likely cause of such a disagreement is residency. The rung 2a harness
streams a 12 GB working set to defeat the cache, while a live block re-reads
resident weights. If cache reuse matters, the crown's extra weight stream is
cheaper in situ than the harness reports, and every candidate arm moves toward
`a_ship`.

## What rung 2b cannot decide

Rung 2b runs on the local public fixture, whose width mix is not the ranked mix.
It cannot rank the tables for the ranked host, and its leg number is not a score
proxy. The per-cell table from rung 2a remains the transferable measurement.

## Thermal protocol

Ungated. The real 40 C gate provably aborts on this host: a probe from 42.9 C
asymptoted at 40.1 C and aborted after 340 s, and the measured idle GPU floor is
39.49 C. The session runs `a b c d d c b a` at 2 legs per group after one
discarded warm group, which gives every arm the same mean leg position, and it
records entry and exit GPU temperature for every group.
`cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false` are
preserved verbatim in every artifact.
