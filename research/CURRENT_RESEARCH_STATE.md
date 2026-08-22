# SENPAI Research State

- **2026-08-22, ~14:20 UTC.** Advisor base
  `b28900c808aa5c9c6c02b3d40b978bd7599a18e4` (ledger 283 + research state).
  Campaign contract base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`. Organizer
  `upstream/main` `c0dbec051c58bccf5435ee1e1e5b01271dc7e179`. Ledger through
  `## 284`.

- **Most recent human research direction.** None new this generation. The
  standing direction remains the one in `senpai/program.md`: maximise the
  official decode score, treat the plausibility ceiling as an administrative
  gate rather than a target, and submit the strongest legitimate candidate
  autonomously.

---

## Where the campaign stands

We hold one promoted row, `d3c491b5` at **3.49065044**. The live crown is
`48423d09` (noskillcoding) at **3.51845338**.

The most important thing we learned this generation is that **the published
median is a bad instrument and we were reading our own position wrong**.

- **FINDING 153.** On a common denominator, our rejected `0c6191b7` is the
  second-fastest candidate in the seven-row frontier cluster, and it beat the
  crown's candidate leg by +0.0839 %. We lost the crown to the pinned serial
  NUMERATOR, not to a rival mechanism. The crown is rank 5 of 7 on merit.
  Instrument: `research/common_denominator.py --anchor <live crown>`.
- **FINDING 154.** Within one classified runner state, the 8-prompt mean of
  `mtp_seconds_per_token_mean` is a **5.17x sharper** instrument than the
  published median: noise se 0.0187 % against 0.0967 %. On that instrument our
  row beats the crown at **z = +3.88** where the published median read
  **z = -1.65**. Same two runs, opposite signs. Rule 59's ranked bar drops to
  about **0.053 %**.
- **FINDING 152.** A two-state runner lottery worth about 930 us per drafting
  round, roughly 1.15 % of the published median, sits on top of every receipt.
  It is not the GPU clock (alphonse rung 8b, ICC +0.8242, p = 0.00005). It is
  the wired residency ticket, under-sized by construction.

The practical consequence: **we do not need luck, we need one of the three
in-flight mechanisms to land, and we can now read the result at one twentieth of
the old noise floor.**

---

## Current focus: three mechanisms in flight, one axis closed by proof

### 1. Pass-count collapse in the wide QMV — the campaign's biggest lever
`{6:6, 7:7}` at `rows_per_simd = 4`, on askeladd's D_S body, behind per-width
templating. Thorfinn, PR #128, **authorised to submit and minutes away**.
Bit-exact by F148 and confirmed by an eight-leg width gate at 84 rows per leg
with `max_abs_diff = 0`. Five pre-registered predictions span **-8.6 % to
+0.05 %**; my band is **-3 % to -9 %, central -5 %**.

- **FINDING 156 removes the 25 sigma conflict.** The predicted one-pass round
  time at M = 6, 7, 8 sits 17.63 %, 20.25 % and 22.40 % below the measured
  ranked curve, against a census that claims 29.25 %, 22.58 % and 26.34 %.
  The three ratios are 1.659, 1.115 and 1.176, all inside or just below the F87
  isolated-to-in-situ over-prediction band of 1.66x to 2.16x. The census and the
  ranked curve agree. Only edward's fitted cross-sectional `f` is the outlier.
  That yields a third, parameter-free prediction: **-7.59 %**.
- **The shipped depth price expires with this receipt.** F155's identity holds
  only while the cost curve holds, and this arm moves the curve. After the
  receipt the depth-4 marginal is over-charged **4.05x** and the depth-6
  marginal is under-charged **3.12x**, exactly where beagle's mean width of 5.38
  sits while carrying 97.9 % of the ranked median. Thorfinn's receipt is
  therefore a **lower bound on its own mechanism**, and the depth-price axis
  reopens as a constant-vector change with zero kernel work and zero bytes.

### 1b. FINDING 157 — the wide-QMV cost law, and what bounds this axis
`statements per output element = 38 / IPG + 25 / RPS` reproduces every entry of
thorfinn's F151 table to three decimals. The `38` is the weight side (4 halfword
loads, 1 address, 2 metadata loads, 1 group index, 2 widenings, **28 nibble
ops**); the `25` is the activation side (1 sum-table load, 4 vec4 loads, 16
widenings, 4 addresses).

- **Balance law.** Registers scale with `IPG x RPS`; minimising `38/I + 25/R`
  subject to `I x R = C` gives `I / R = 1.52`, so at `R = 4` the optimum is
  `I = 6.08`. Thorfinn's `{6:6}` sits on the theoretical optimum.
- **The `rows_per_simd` axis is now closed in both directions.** Askeladd's E132
  closed `RPS = 2`; the balance law closes `RPS = 8` at every equal-slot
  comparison. Reopen only if the 38 or the 25 changes.
- **What is left.** At `(6,4)` the 28 nibble ops are 73.7 % of the weight
  constant and 37 % of the cell. Free dequant would take 12.583 to 7.917, a
  **-37 %** bound on the whole axis. Honest negative already recorded: the
  tinygemm/any4 magic-constant form is op-neutral or adverse in float, and
  bit-plane decomposition violates Rule 92. The narrow open question is whether
  any bit-exact form extracts 8 four-bit values and converts to float in fewer
  than about 24 ops on g17s. Compile-only, no GPU.
- **The central thesis this produces.** A ranked M = 1 round is 31,182 us for
  14.41 GB, which is 462.2 GB/s and already at the DRAM bound. A beagle round at
  mean width 5.38 costs 55,870 us for 4.38 tokens. If wide verification cost
  what M = 1 costs, for the identical weight bytes, the published ratio would be
  about **5.34**; today it is about 3.34. **The entire remaining gap is the tax
  wide verification pays for zero extra weight bytes.** Every other axis on the
  board is worth 0.4 % to 1.5 %. This one is worth the rest.

### 2. The wired residency ticket — eliminate the runner-state lottery
`wiredZHDefaultSlackMB` 64 to 512. Alphonse, PR #130. Measured post-sizing
growth **218.71 MiB** against 64 MiB of slack, exceeded in all seven worker
processes by 3.07x to 3.42x, plus a page-rounding tax of 34.8 to 69.6 MiB over
4,454 live buffers. True floor **253.51 to 288.30 MiB**. Expected value
**+0.38 %**, realised as variance elimination rather than mean speedup, so the
ranked test is the state classification of the next three receipts. Open
mechanism question: his greedy-hash-order admission story predicts a 2.4-2.6 %
slow rate against an observed 33-50 %, a 13-21x gap. A zero-timing local probe
of `unwired_set_` at 20 starts per arm settles it today.

### 3. Discrimination, not price, in the depth scheduler
Edward, PR #134. **FINDING 155 closes the depth-PRICE axis with a proof**: the
shipped `price.marginal` vector equals our fitted ranked cost curve, normalised,
to four significant figures on all eight entries. That explains why all nineteen
E128 price arms lost and why the F9 corrected price moved zero prompts. What
remains is the **+8.52 % oracle discrimination gap**. The missing thing is
information, not functional form: `Qwen36MTPBlockSession.swift:1490-1495`
already computes per-row top-2 evidence for every verified row and the scheduler
throws all of it away except the tail. Zero GPU cost, already in his file, same
legality class as the shipped `pendingTop2` use. Score every arm **per
boundary**, because the depth-4 marginal is 4.05x the base.

### 4. Sketch-first draft readout (C1) — the largest single unexploited number
Askeladd, PR #133, offline screen only, no scored-surface change. Removes
**53.06 MB of the 323.59 MB per-draft-step byte budget**, worth **+0.90 % to
+1.47 %, central +1.15 %**, with 13x to 25x headroom against the break-even miss
rate. Kill line is a worst-stratum net miss rate above 3.0e-3.

---

## Potential next research directions

**Ready to assign to the next free student.**

1. **C2, precision-island quantization.** Quantize the bf16 precision islands to
   affine-4 g64; removes 22.61 MB per draft step, worth **+0.38 % to +0.45 %**.
   Reopened and unowned. Needs `Qwen35.swift`, so it schedules after E129 lands.
2. **A cleanup PR after receipt 1 merges.** Prune stale experiment flags and
   dead code paths from the Route B dispatch table and the E120 arm switches.
   Deletion is the default; the winning behaviour becomes the only main path.
3. **The `qwen_e120_qmv_wide<8,8>` rider.** Askeladd's compile-only model says
   one pass of `wide<8>` beats two passes of `wide<4>` by 9.25-16.62 % on the
   shipped RPS=4 body even at the harshest spill charge, and D_S removes the
   spill sensitivity. It is the rider behind `{6:6, 7:7}`, not the prize, because
   the ranked mass that pays sits at M = 5-7.

**Themes worth opening once a slot frees.**

3b. **The nibble-dequant residual, the last big number on the QMV axis.** At
   `(IPG = 6, RPS = 4)` the 28 nibble operations are 37 % of the cell and 73.7 %
   of the weight constant. A free dequant would cut the cell by 37 %, which
   bounds the axis. The question is narrow and compile-only: is there a
   bit-exact form that extracts 8 four-bit values and converts them to float in
   fewer than about 24 operations on g17s? The magic-constant and any4 forms are
   already refuted in float, and bit-plane decomposition violates Rule 92.

3c. **Refit the depth price onto the post-receipt-1 cost curve.** Assigned to
   edward as rung 0b. Constant-vector change, zero kernel work, zero submitted
   bytes, zero correctness risk, and the already-validated zero-GPU replayer
   prices it before the receipt lands. It becomes the next submission if the
   stale-price loss clears +0.30 % F83-weighted on the central curve.

4. **Shortlist-score entropy as a scheduler input.** Already resident in
   threadgroup memory at `Qwen35.swift:3716` at approximately zero cost. Edward's
   per-position AUC ranking in the paying acceptance band puts margin first at
   0.8763; entropy has never been measured and is the only cheap signal left
   that is not a transform of the ones already used.
5. **A censoring-aware reach estimator.** F112 shows the shipped reach estimator
   is biased low by 9-24 % and no E128 arm tested the censoring correction.
6. **The head's own dispatch count.** Rung 0b measured 31.44 head projections
   per round at `ntg.x == 1`, all falling through to `qmv_fast_impl<T,64,4>` at
   57 registers. Nobody has asked whether that dispatch count can be reduced by
   batching, which the `Kernel Contracts` result (2604.22032) claims is 3-17x
   faster on M5 for exactly this shape class.
7. **A second, independent read of the runner state.** If alphonse's residency
   story is refuted by his own 20-start probe, the state has no explanation and
   it is worth 1.15 % of every receipt. That would become the top priority.

**Standing methodological commitments.**

- Read every ranked receipt with the FINDING 154 instrument, after classifying
  its state with `research/common_denominator.py` and `research/cluster3.py`.
  Never decide from the published median (Rule 63, Rule 100).
- Every build or worker witness needs a demonstrated failing polarity on a real
  commit (Rule 101). A witness that cannot fail is inventory, not a gate.
- Price in instructions removed per output element, never in bandwidth, for
  anything in the QMV family (Rule 94).
- Keep the submission slot moving. FACT 10 allows exactly one in-flight
  submission and validation runs 42-130 minutes, so the queue is the scarcest
  resource in the campaign after GPU time.
