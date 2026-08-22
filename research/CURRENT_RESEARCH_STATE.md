# SENPAI Research State

- **2026-08-22, ~14:45 UTC.** Advisor base
  `0eed7b40d6c3b87d50b888fee2f6ec993dc7ea88` (ledger 284 + research state).
  Campaign contract base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`. Organizer
  `upstream/main` `c0dbec051c58bccf5435ee1e1e5b01271dc7e179`. Ledger through
  `## 285`.

- **Most recent human research direction.** None new this generation. The
  standing direction remains the one in `senpai/program.md`: maximise the
  official decode score, treat the plausibility ceiling as an administrative
  gate rather than a target, and submit the strongest legitimate candidate
  autonomously.

- **One submission is in flight.** `623e77af`, queued 14:12:57Z, `validating`.
  It carries thorfinn's `{6:6, 7:7}` one-pass wide-QMV tables behind per-width
  templating on askeladd's D_S body. FACT 10 allows exactly one in-flight
  submission, so the queue is closed until it resolves (expect 14:55Z-16:23Z).

---

## Where the campaign stands

We hold one promoted row, `d3c491b5` at **3.49065044**. The live crown is
`48423d09` (noskillcoding) at **3.51845338**. Nobody has crossed 3.5185 in the
last four and a half hours and the whole frontier cluster is bunched inside
0.23 % on a common denominator, which means the rivals are drawing lottery
tickets rather than landing mechanisms.

The most important thing we learned this generation is that **the published
median is a bad instrument and we were reading our own position wrong**.

- **FINDING 153.** On a common denominator, our rejected `0c6191b7` is the
  second-fastest candidate in the frontier cluster, and it beat the crown's
  candidate leg by +0.0839 %. We lost the crown to the pinned serial NUMERATOR,
  not to a rival mechanism. The crown is rank 7 of 9 on merit. Instrument:
  `research/common_denominator.py --anchor <live crown>`.
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

The practical consequence: **we do not need luck, we need one of the four
in-flight mechanisms to land, and we can now read the result at one twentieth of
the old noise floor.**

---

## Current focus: four mechanisms in flight, two axes closed by proof

### 1. Pass-count collapse in the wide QMV — the campaign's biggest lever
`{6:6, 7:7}` at `rows_per_simd = 4`, on askeladd's D_S body, behind per-width
templating. Thorfinn, PR #128, **submitted as `623e77af` and validating**.
Bit-exact by F148 and confirmed by an eight-leg width gate at 84 rows per leg
with `max_abs_diff = 0`. Five pre-registered predictions span **-8.6 % to
+0.05 %**; my band is **-3 % to -9 %, central -5 %**, refuted above -1 %.

- **FINDING 156 removes the 25 sigma conflict.** The predicted one-pass round
  time at M = 6, 7, 8 sits 17.63 %, 20.25 % and 22.40 % below the measured
  ranked curve, against a census that claims 29.25 %, 22.58 % and 26.34 %.
  The three ratios are 1.659, 1.115 and 1.176, all inside or just below the F87
  isolated-to-in-situ over-prediction band of 1.66x to 2.16x. The census and the
  ranked curve agree. Only edward's fitted cross-sectional `f` was the outlier,
  and FINDING 159 has since found its cause.
- **The arm makes the shipped depth price more accurate, not less.** This
  reverses what I told thorfinn in E129 F20. The shipped price is a flat
  marginal of 0.18 at every depth (`makeUniformDepthPrice`). Against the true
  ranked cost curve that price is about 1.67x too expensive at depths 0-3 and
  about 2.9x too cheap at depth 4, which is the step into verify width 6. The
  one-pass arm collapses the depth-4 true marginal from 0.4383 to 0.1083 and
  moves the cliff out to depth 6, where very little ranked mass sits. The
  receipt may therefore carry a small incidental scheduling **gain**. It is not
  a lower bound spoiled by a stale price.
- **The board is monotone in one-pass-ness.** Across 202 rows and 18 dispatch
  tables, the 121 modal two-pass rows average 2.8246 official, the 7 rows
  carrying thorfinn's exact `{6:6, 7:7}` table average 3.0637, and the single
  full-identity row is the board maximum at 3.1391.
- **The register model closes on measured hardware.** Vector-register floats per
  thread in `qwen_e120_qmv_wide<NA>` are `13 * NA`; adding a fixed 27 reproduces
  askeladd's measured g17s D_S census exactly at NA = 6, 7 and 8. That makes
  `{9:9}` dead by arithmetic and caps the reachable one-pass IPG at RPS = 4 at
  **7**. FINDING 157's balance law independently gives an optimum of 6.08. So
  `{6:6, 7:7}` is simultaneously the theoretical optimum and the register-ceiling
  maximum. It is a third independent closure of `rows_per_simd = 8`.

### 1b. FINDING 157 — the wide-QMV cost law, and what bounds this axis
`statements per output element = 38 / IPG + 25 / RPS` reproduces every entry of
thorfinn's F151 table to three decimals. The `38` is the weight side (4 halfword
loads, 1 address, 2 metadata loads, 1 group index, 2 widenings, **28 nibble
ops**); the `25` is the activation side (1 sum-table load, 4 vec4 loads, 16
widenings, 4 addresses).

- **Balance law.** Registers scale with `IPG x RPS` (confirmed exactly by
  `(3 x RPS + 1) x IPG + 27`); minimising `38/I + 25/R` subject to `I x R = C`
  gives `I / R = 1.52`, so at `R = 4` the optimum is `I = 6.08`.
- **The `rows_per_simd` axis is closed in both directions**, now by three
  independent arguments: askeladd's E132 occupancy result closes `RPS = 2`, the
  balance law closes `RPS = 8`, and the 126-register g17s ceiling closes it
  again. Reopen only if the 38 or the 25 changes.
- **What is left.** At `(6,4)` the 28 nibble ops are 73.7 % of the weight
  constant and 37 % of the cell. Free dequant would take 12.583 to 7.917, a
  **-37 %** bound on the whole axis. Honest negative already recorded: the
  tinygemm/any4 magic-constant form is op-neutral or adverse in float, `half2`
  two-at-a-time lands back near 24, and bit-plane decomposition violates
  Rule 92. The narrow open question is whether any bit-exact form extracts 8
  four-bit values and converts to float in fewer than about 24 ops on g17s.
  Compile-only, no GPU. Assigned to thorfinn for his validation window.
- **The central thesis this produces.** A ranked M = 1 round is 31,182 us for
  14.41 GB, which is 462.2 GB/s and already at the DRAM bound. A beagle round at
  mean width 5.38 costs 55,870 us for 4.38 tokens. If wide verification cost
  what M = 1 costs, for the identical weight bytes, the published ratio would be
  about **5.34**; today it is about 3.34. **The entire remaining gap is the tax
  wide verification pays for zero extra weight bytes.** Every other axis on the
  board is worth 0.4 % to 1.5 %. This one is worth the rest.

### 2. The wired residency ticket — a deterministic mean, not a lottery ticket
`wiredZHDefaultSlackMB` 64 to 512. Alphonse, PR #130. Measured post-sizing
growth **218.71 MiB** against 64 MiB of slack, exceeded in all seven worker
processes. The true floor is now **283.33 to 347.94 MiB** after the buffer
census was re-measured at steady state (max 8,271 live buffers, not the 4,454
seen at sizing), so 256 MiB fails even the expected case and 512 MiB clears the
floor with 164 MiB spare.

- **FINDING 158 — the value is a measured mean, and it is large enough to settle
  in one receipt.** His rung-10a palindrome gives, after removing one linear
  session trend that explains 97.10 % of the residual variance, an absolute
  candidate improvement of **-0.4839 %** for `none -> s512` and **-0.1968 %** for
  the shipped step `s64 -> s512`, with a residual sd of 0.0498 % on 2 df. Against
  FINDING 154's 0.0187 % ranked se that is **z ~ 10.5**. 64 MiB already buys
  59 % of the wiring benefit and the fix buys the remaining 41 %.
- **RULE 102 — a palindrome is not a dispersion instrument.** A counterbalanced
  palindrome cancels drift in the arm *means* and destroys it into the
  *within-arm spreads*, because arm identity is collinear with distance from the
  centre of the session. Alphonse's claimed two-state bimodality in the `s64` arm
  was that artefact and is withdrawn. Use repeated randomised starts instead.
- **FINDING 160 — the local ratio and the ranked value can have opposite signs.**
  For `none -> s512` the absolute candidate time improves by 0.4839 % while the
  local `mtp_decode_speedup` gets 0.3846 % **worse**, because locally both legs
  run the candidate binary and wiring speeds the co-timed serial leg by 0.8665 %,
  about 1.8x the candidate leg. Had alphonse read the local ratio he would have
  thrown the fix away. This is the first measured instance of the failure mode
  `senpai/verify-ranked-score-boundary.sh` exists to prevent. **Standing rule:
  report absolute candidate seconds per token as the headline of every local
  arm.**
- **Open mechanism question, now the only bimodality work left.** His
  greedy-hash-order admission story predicts a 2.4-2.6 % slow rate against an
  observed 33-50 %, a 13-21x gap. It also does not explain why wiring speeds the
  serial leg, which never touches the head. The adopted resolution is that local
  wiring measures a **backbone** effect while the ranked lottery is a **head
  admission** effect: two mechanisms behind one integer. A zero-timing probe of
  `unwired_set_` at 20 starts per arm is the only instrument that can settle it.

### 3. Discrimination, not price, in the depth scheduler
Edward, PR #134.

- **ADVISOR ERROR 117 — FINDING 155 is struck.** I published that the shipped
  `price.marginal` vector equals our fitted ranked cost curve normalised. It does
  not. `depthPriceArm = .ship` selects `makeUniformDepthPrice()`, which is a
  **flat 0.18 at every depth**. The vector I published is the `rankedprice` arm,
  and the contradiction was inside my own arm table: if the shipped price were
  the ranked curve then `rankedprice` would have scored exactly 0.0000, and it
  scored -2.8508 %. Edward found this at source. Ledger 284.2 is void and his
  rung 0b price refit is cancelled.
- **The depth-price axis is nevertheless closed, permanently, for a better
  reason.** 19 of 19 price arms lost; `rankedprice` -2.8508 %; `levelfix`
  -0.9673 %; `h = 0.32` scored 2.84585. A correct price only helps if the
  quantity it multiplies is trustworthy, and sharpening a price on a 0.70-AUC
  reach signal only moves more rounds across a boundary that is 2.9x more
  expensive to get wrong. **The price is not the bottleneck. The information
  is.** `cumulative` carries no independent degree of freedom either: it is a
  prefix sum from 1.0, so `cumulative[d] == round_us(d+1) / round_us(1)` for any
  curve.
- **What remains is the +8.52 % oracle discrimination gap**, and it is the whole
  of E134. `Qwen36MTPBlockSession.swift:1490-1495` already computes per-row top-2
  evidence for every verified row and the scheduler throws all of it away except
  the tail. Zero GPU cost, already in his file, same legality class as the
  shipped `pendingTop2` use. Score every arm **per boundary** and report
  **incremental** AUC over the shipped input set, starting at depth 4, which is
  the step into width 6. Headline must be held out.

### 4. Sketch-first draft readout (C1) — the largest single unexploited number
Askeladd, PR #133, offline screen only, no scored-surface change.

- **Full C1** removes **53.06 MB of the 323.59 MB per-draft-step byte budget**,
  worth **+0.90 % to +1.47 %, central +1.15 %**, with 16x headroom against the
  break-even miss rate.
- **`hybridA` is a second candidate, not a control.** Sketching only the
  24,584-row leaf stage and leaving the 12,292-row centroid stage exact removes
  35.00 MB, which is 65.96 % of full C1, and is worth **+0.59 % to +0.97 %,
  central +0.76 %**. It has 16x break-even headroom of its own and much lower
  recall risk. T0 and T0b must be evaluated separately for the two candidates.
- **An alignment defect was found and fixed before it cost the sweep.**
  `E87HiddenDump.record` sits inside `draftTokenID`, so it emits one row per
  draft-head call rather than per emitted token. The planned index shift would
  have silently mis-aligned the accepted-or-not label in every one of the 240
  sweep cells. Askeladd repaired it against the parent's `row_ledger` with a
  hard raise on mismatch and verified zero mismatches on all 11 seeds.

---

## Potential next research directions

**Ready to assign to the next free student.**

1. **C2, precision-island quantization.** Quantize the bf16 precision islands to
   affine-4 g64; removes 22.61 MB per draft step, worth **+0.38 % to +0.45 %**.
   Reopened and unowned. Needs `Qwen35.swift`, so it schedules after E129 lands.
2. **A cleanup PR after receipt 1 merges.** Prune stale experiment flags and
   dead code paths from the Route B dispatch table and the E120 arm switches,
   and delete the E128 price arms now that the axis is permanently closed.
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
   fewer than about 24 operations on g17s? The magic-constant, any4 and `half2`
   forms are already refuted in float, and bit-plane decomposition violates
   Rule 92.

4. **Shortlist-score entropy as a scheduler input.** Already resident in
   threadgroup memory at `Qwen35.swift:3716` at approximately zero cost. Edward's
   per-position AUC ranking in the paying acceptance band puts margin first at
   0.8763; entropy has never been measured and is the only cheap signal left
   that is not a transform of the ones already used.
5. **A censoring-aware reach estimator.** F112 shows the shipped reach estimator
   is biased low by 9-24 % and no E128 arm tested the censoring correction. This
   matters more than it did: FINDING 159 says the boundary the reach signal
   guards is the expensive one.
6. **The head's own dispatch count.** Rung 0b measured 31.44 head projections
   per round at `ntg.x == 1`, all falling through to `qmv_fast_impl<T,64,4>` at
   57 registers. Nobody has asked whether that dispatch count can be reduced by
   batching, which the `Kernel Contracts` result (2604.22032) claims is 3-17x
   faster on M5 for exactly this shape class.
7. **A second, independent read of the runner state.** FINDING 158 shows the
   wiring fix has a deterministic mean worth -0.1968 % on the candidate leg, so
   it now pays for itself without the lottery story. But if alphonse's admission
   model is refuted by his own 20-start probe, the state itself still has no
   explanation and it is worth 1.15 % of every receipt. That would become the
   top priority.
8. **Re-measure whatever the receipt-1 arm invalidates.** If `623e77af` lands,
   the ranked cost curve moves at M = 6 and 7. Edward's oracle gap must then be
   reported against both the pre-arm and post-arm curves, and FINDING 156's
   parameter-free prediction becomes a measured calibration rather than a
   forecast.

**Standing methodological commitments.**

- Read every ranked receipt with the FINDING 154 instrument, after classifying
  its state with `research/common_denominator.py` and `research/cluster3.py`.
  Never decide from the published median (Rule 63, Rule 100).
- Report absolute candidate seconds per token as the headline of every local
  arm. The local ratio can have the opposite sign to the ranked value
  (FINDING 160).
- Never read a within-arm spread from a counterbalanced palindrome as a
  dispersion, variance or bimodality instrument (Rule 102).
- Every build or worker witness needs a demonstrated failing polarity on a real
  commit (Rule 101). A witness that cannot fail is inventory, not a gate.
- Price in instructions removed per output element, never in bandwidth, for
  anything in the QMV family (Rule 94).
- When two channels move in opposite directions, a pooled cross-sectional fit
  estimates their difference and is a lower bound on whichever part you named
  (FINDING 159d).
- Keep the submission slot moving. FACT 10 allows exactly one in-flight
  submission and validation runs 42-130 minutes, so the queue is the scarcest
  resource in the campaign after GPU time.
