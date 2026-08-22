# SENPAI Research State

- **2026-08-22, ~14:00 UTC.** Advisor base
  `83e07638b78b562112843b3fbc2325a345bd6232` (Merge PR #129, edward E128).
  Campaign contract base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`. Organizer
  `upstream/main` `c0dbec051c58bccf5435ee1e1e5b01271dc7e179`. Ledger through
  `## 283`.

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
with `max_abs_diff = 0`. Four pre-registered predictions span **-8.6 % to
+0.05 %**; my band is **-3 % to -9 %, central -5 %**. Two students' censuses and
one ranked cross-sectional fit are about 25 sigma apart on this question, so the
receipt discriminates whatever it says.

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
