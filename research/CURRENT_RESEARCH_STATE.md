# SENPAI Research State

- 2026-08-22 18:25 UTC — campaign round 291.
- Most recent research direction from the human researcher team: none received this round.
  The last standing human instruction remains "keep the frontier moving and submit the
  strongest legitimate candidate"; the advisor decides all experiment selection autonomously.

## Where the campaign stands

The published crown is `dacf7005` (newjordan, `3.52326653`, source `166e0cad`). Our
`623e77af` sits second at `3.52085227`, a gap of 0.0686 %. That ordering is a serial-leg
lottery result, not a merit ordering.

Two measurements this round settle the competitive question:

1. **FINDING 176.** Under the common-denominator instrument (827 rows, 18 in the 0.30 %
   cluster) our candidate is the *fastest candidate on the board*. The crown holder's
   candidate is 0.034 % slower than ours. Candidate-vector-only spread inside the cluster is
   0.1049 %; serial-vector-only spread is 0.0987 %. A candidate gain must be large compared
   with 0.1 % before it is visible in a published median.
2. **FINDING 177/178.** The crown tree `166e0cad` is our own `60d5b34a` plus one change: an
   SDPA warm loop widened from `qL in [1,5,4]` to `[1,2,3,4,5]`. That change is *provably
   inert*: the SDPA pipeline cache key holds dtype, two head dims, five booleans and
   `blocks`, and no `qL`. The ranked A/B measures `+0.0039 %` candidate mean, `z = +0.33`.
   We therefore delete nothing of value when we submit over the crown, and our next receipt
   is already an isolated A/B.

The consequence is that the campaign's job is now purely to find real candidate time, and to
find enough of it that a ~0.1 % publication noise floor cannot hide it.

## Current research focus

**Theme A — the M=5→6 verify cost cliff.** This is the largest open prize. The shipped
price table is wrong at exactly one boundary (FINDING 164); replaying the correct boundary
price is worth `+2.34 %` held out (edward's `pb6`). Three levers that should have moved the
cliff — instruction issue, weight passes, resident occupancy — all measured at or near zero,
and FINDING 174 closed the occupancy axis with a ranked receipt. The surviving explanation is
that the step is carried by some *other* dispatch family (SDPA split-5, the GDN packed mixer
at S=6, or host-side round bookkeeping), and that has never been attributed by direct time
measurement.

**Theme B — schedule and price, not kernels.** `pb6` is the campaign's strongest single
candidate and holds the free submission slot. It is a two-constant change to the depth-price
table, gated by a measured-curve refit that is the current critical path.

**Theme C — draft-path byte removal.** The C1 low-rank sketch readout
(`qlowrank256-N4096-p0.35`, net `+0.573 %` after E136 rung 0 priced the widened selection at
`18.24 µs/draft step`) removes 37.5 MB of the 323.6 MB per-draft-step budget.

**Theme D — launched threadgroups.** The shipped QMV grid launches `m` columns at every
routed width while only `ceil(m/ipg)` load a weight. The tight grid removes 5.68 empty
columns per round, bit-exactly, in one word. Rung 1 is running.

**Theme E — measurement discipline.** Three new rules this round. Rule 110: a warm-phase
change is worth zero unless it changes a pipeline cache key. Rule 111: a part table finds
where the time went; it never decides a gate. Rule 109 (last round): prove the guard passes
at runtime on the measuring host before pricing anything.

## In flight

| PR | student | experiment | state |
|---|---|---|---|
| #134 | edward | E134 depth-price cliff, `pb6` | Gate A passed; measured-curve refit is the critical path; holds the submission slot |
| #135 | thorfinn | E135 tight QMV launch grid | rung 1 running, 12 counterbalanced 512-token legs |
| #136 | askeladd | E136 C1 sketch readout + fp32 rerank tiebreak | rung 0 cleared (+0.573 %); C1 build behind a gate |
| #130 | alphonse | E130 entry-point occupancy tax | closing as a clean negative; F175 resolved the guard question |

## Potential next research directions

1. **P1 — cliff family attribution (next assignment, alphonse).** Re-key 12 archived traced
   512-token legs by realized verify width using the eleven-way per-round segment split, then
   run the never-executed `sweepGatedDelta(widths:1...12)` cost curve (<5 s GPU). Stop rule:
   ≥30 % non-QMV redirects the whole theme; <10 % closes it.
2. **P2 — the 254,279,680-byte permanently unwired scratch buffer.** Present in every
   residency draw. The resize path skips oversized entries and never evicts, so it can never
   be wired at any slack rung. Admission-side only. Bracket 0.3–1.5 % if decode touches it.
3. **P3 — serialized host-side round bookkeeping.** Sum `readout_us + commit_us + upkeep_us`
   width-matched and decide whether the tail can overlap the next round's submit.
4. **P4 — the GDN S=2 mid-state write.** An unconditional 151 MB per-round write whose
   break-even needs `P(reject | M=2) < 0.49`, never measured.
5. **H179 — the `query_transposed` warm gap.** The scored SDPA path passes slices; the warm
   path passes contiguous zeros. If the boolean differs, up to three pipeline
   specialisations compile inside the timed window. Zero-GPU gate, edward rung 6.
6. **C2 — precision-island quantization to affine-4 g64.** Reopened, unowned, +0.38–0.45 %.
7. **The head-history fold warm gap.** Widths 1..9 are flushed but only width 2 is warmed.
   Must be re-checked under Rule 110 before it is priced at all.

## Closed this round

- The qL warm ladders, closed at source as well as by three receipts.
- The `_nax` / MPP tensor path: no `qmv_nax` exists, `get_qmv_batch_limit` keeps M ≤ 9 on
  matvec, and every M ≥ 4 quantized matmul on the critical path is the bit-exact target
  verify. Two named reopeners recorded.
- The wired-slack ladder above 64 MiB (null at 512 tokens, FINDING 173) and the warm refill
  after wiring (a measured −5.3 % ranked, FINDING 172).
- Occupancy and residency as a pricing axis (FINDING 174, closed by a ranked receipt).
