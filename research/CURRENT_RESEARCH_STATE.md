# SENPAI Research State

- 2026-08-23 07:25 UTC
- Most recent human research direction: none received this generation. The team
  has given no new steer since the campaign opened; the advisor is operating
  autonomously under `senpai/program.md`.

## Where the campaign stands

Crown `684821ed` at 3.71959723 on organizer source `eb5eadc7`. Our best row is
`572b2cc4` at 3.66218564.

**Finding 237 changes how that gap should be read.** `f7d59543` is a zero-delta
resample of `eb5eadc7`, the crown's own tree, and it printed 3.69864608. Same
tree, zero editable bytes changed, `-0.5633 %` of published median. On the five
weighted prompts the candidate leg moved `+0.0011 %` with `sd 0.0297 %`, while
the serial leg moved with `sd 0.7338 %`. One prompt, essays, ran 1.4968 % faster
on the serial side and flipped the upper median slot from medicine to essays.

Two consequences:

1. The crown's honest ranked level is about 3.701 to 3.709, not 3.7196. Our
   real candidate-leg gap remains the Finding 230 figure of **0.82 %**.
2. The candidate leg is a precision instrument at `se 0.0133 %` on the weighted
   five, while the published median carries a `+/- 0.56 %` lottery. Every
   decision must be priced on the candidate leg (Rule 118), and every
   submission needs candidate-leg margin as lottery insurance.

Our in-hand mechanisms sum to well over the 0.82 % gap. The campaign's problem
is not a shortage of ideas. It is measurement precision, correct attribution,
and submission-slot throughput.

## Current research focus

**1. Land the crown attempt.** Thorfinn's `composed67 + pb6` is measured at
`+2.1305 %` locally against his branch base (T44, W&B `8c5y1abg`), and Rule 130
is lifted because the composition proved additive once each term was priced
under the grid it ships with. He is adding the pre-registered `onePass678`
(`+1.434 %` local forecast) while the submission slot is blocked.

**2. Own the seed prefill.** Alphonse's E147 software-pipelined affine NAX
transposed GEMM is in flight as `7226dc9a`. Local component effect `-1.0657 %`
with `t = -11.12` and complete rank separation on 16 legs. The ranked forecast
is `-3 %` to `-4.5 %` on the prefill component, worth `+0.22 %` to `+0.34 %` of
published median. Rung E composes the 128x32 rectangular retile, whose isolated
board evidence is `-5.07 %`, the best seed prefill ever measured here.

**3. Fix the draft-length objective, not the acceptance predictor.** Our own
oracle result says a perfect acceptance estimator makes the schedule worse. The
literature explains why: the measured cost curve has a `+30.6 %` wall at
5 to 6 and a near-flat plateau at 8, so a first-order marginal rule is provably
trapped at M=5 and can never reach M=8. Edward is replaying a full argmax over
the admissible set `{0..5, 8}`, zero GPU.

**4. Mine the public board for unclaimed mechanisms.** Askeladd is repricing
every eligible rejected row through the E146 state corrector, gated on
bit-identical controls. Two of our best queued mechanisms already came from
reading one rejected rival row.

## Standing measurement discipline

- Price value against the contaminated round cost `leg_us / R`; state share
  against the clean cost `decode_us / R` (Finding 235, Rule 134). One percent of
  published median costs **515.2 us/round**.
- An overhead column costs `2.6815 +- 0.0355` us; a working column costs
  `13.8715 +- 1.8355` us. Never regress them together (Rule 135).
- A guard needle must be unique to the arm, not merely long, and both polarities
  must be asserted on real builds (Rule 136).
- Run one warmup leg before any timed ABBA session; it cut entry-temperature
  spread from 23.27 C to 1.757 C (Rule 137).
- Yukon allows one in-flight submission. Every authorisation now names the slot
  holder and the queue position.

## Closed this generation

- The wired-residency slack ladder. E130 measured s64 through s2048 across 12
  legs and the argmax is s64, the shipped value. The slack is 98.3 to 99.9 %
  consumed at every rung, so raising it admits no extra byte. The whole prize is
  capped at `+/- 0.13 %`. Finding 8's KV-cache mechanism is refuted at source and
  the KV growth path is separately dead by arithmetic at about 5.8 us/round.
- Rule 130's non-additivity claim, refuted by T44.
- All tree and multi-candidate drafting, structurally, by
  `QwenRuntimeMTPDriver.requireStructurallySound:310-349`.
- All pure acceptance predictors, by our own oracle result.

## Potential next research directions

1. **FP32-twin activations, bit-exact.** Pre-widen the bf16 activation tensor to
   fp32 once. Because bf16 to fp32 is exact, everything downstream stays
   bit-identical while 64 convert instructions per lane per k-block disappear at
   NA=4. E123 closed activation widenings on the assumption that deletion needs
   bf16 arithmetic; pre-widening makes that closure leak. Range `+0.3 %` to
   `+2.0 %`. Route B, so thorfinn after the crown attempt resolves.
2. **Fold the 257 per-round chunk-sum fill dispatches into their producers.**
   About 1.25 ms of a 53 ms round. `qwen35FusedResidualRMSNormKernel` already
   reads every element it emits. Rung 0 is a ten-minute zero-GPU bound. Range
   `+0.3 %` to `+1.2 %`.
3. **First-error focal loss on the proposal head.** We own `mtp-head/`. The
   published method reports `+21 %` to `+76 %` accepted draft length with no
   extra forward passes and no exactness change. This is the one head-side
   direction our oracle result does not kill, because it improves the thing
   predicted rather than the predictor.
4. **Bit-plane draft pass.** Read 2 of 4 planes during drafting and all 4 during
   target verification. Roughly halves draft weight bytes and stays token-exact,
   because only draft quality moves. Budget the bit-transposition cost first.
5. **MARLIN's four-stage pipeline and SplitK on the QMV family.** The published
   evidence says W4A16 can stay memory-bound to M=16 to 32 and that reducing
   register and shared-memory pressure gives nearly 4x occupancy. If our cliff is
   a register-ceiling cliff, SplitK is the direct antidote.
6. **The 6 to 7 cliff itself.** 25,861 us, 2.32 times the mean step, still
   unattributed and unowned.
7. **The GDN S=2 mid-state write gate.** A 151 MB fp32 snapshot per S=2 round.
   Skip-write and replay-on-reject wins if `P(reject | M=2) < 0.49`, and nobody
   has ever censused that number. Highest correctness risk in the queue, so it
   needs a strong exactness gate before any timing.
8. **Corrective measurements.** The empirical entropy of our own 4-bit nibble
   stream bounds every lossless-coding direction in one offline hour. A
   STREAM-calibrated Metal read-bandwidth microbenchmark resolves the 1.69x
   discrepancy between our implied 462.2 GB/s at M=1 and the 273 GB/s nominal.
