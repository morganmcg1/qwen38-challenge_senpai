# SENPAI Research State

- 2026-08-19, after merging E54 (PR #58), reviewing E55 (PR #57), and auditing
  my own pricing chain twice (ledger 190, 191). Campaign base `6870bdb4`.
- Most recent human research direction: issue #22 -- execute aggressively
  toward the winning frontier. Issue #31 is complete and closed. No new human
  direction is outstanding.

## Board

| quantity | value |
|---|---|
| live promoted frontier | **3.24985583421771** (submission `59b321ee`, solver fkiene, commit `9e1ff9ec`) |
| our best official submission | **3.23250848263467** (receipt `ca9251b8`, candidate `2b0c36a0`, rejected) |
| our deficit | **0.01734735 = 0.5367 %** |
| ranked MDE at 2 sd | **+0.283 %** (worst case +0.527 %) |
| local end-to-end null floor | **0.0629 %** |

Our deficit is candidate-leg overhead at a frozen accept trajectory: `ca9251b8`
and the frontier report a byte-identical `mean_draft_len` 8-tuple and the same
head digest, and we are slower on 8 of 8 prompts by a median 0.372 %. `beagle`
carries 79 % of the deficit and still has 7.9 % headroom; `medicine` saturates
after 0.64 %.

## Current research focus

**The campaign's binding unknown moved this round.** E54 measured every
crossrow QMV cell in isolation on the real shipped table, with bitwise parity,
a working positive control, and binary-verified routing. Cell timing is now
well characterised. The unknown is the **step from cell timing to score**: P4
priced E27's exact composite at +2.21 to +2.53 % `harness=ranked` against a
board result of **-0.3321 %**. The sign differs under every mixture and the gap
survives all of my published corrections, leaving +0.90 points (traffic branch)
or +2.02..+2.13 (h-ratio branch).

🔴 **A second binding correction landed with E55 (189(D)), and it is mine.**
`psi_mtp = 0.693391` is a share of the **local** candidate leg, and the local
leg carries a **23.4 %** seed prefill against the ranked **8.75 %**. Every
ranked price published through `psi_mtp x ... x 0.9125` therefore charged the
prefill twice and is **low by 1.29x to 1.31x**. As a ranked leg elasticity
`psi_mtp` is **0.8167..0.8259**. Instrument `research/dilution_basis.py`,
12 self-tests including two positive and one negative control, exits 0. Two
calibrations sharing no input agree on the underlying round-basis share to
0.36 %. **This flips the `r=2` route from closing the deficit under one mixture
to closing it under both.**

🔴🔴 **A THIRD and a FOURTH pricing error landed this round, both mine, both
found by delegated audit, both compounding with 189(D).**

- **190 / E58.** The dispatch prize published as "2.2 % of the ranked leg /
  2.0 % of score / 7.1 sd" carried the **local** dispatch count onto the ranked
  leg. Ranked `beagle` runs 107 rounds at mean draft 4.5327; our local arm ran
  76 at 6.5132, and dispatches per round are a step function of width.
  Recomputing gives **0.86 .. 3.44 %** of the ranked leg, and the `x0.9125` that
  produced "2.0 %" was a third double dilution. **E58 survives**: the low end is
  3.0x MDE, and the corrected per-prompt table shows the band is near-uniform
  across all eight prompts, so it moves the median directly rather than through
  the central-pair weight. Instrument `research/e58_dispatch_repricing.py`,
  20 checks, exits 0.
- **191 / the `r=2` route.** I multiplied a **score** by a shrinkage factor.
  The ranked pricer is **concave** — kink at `+1.0551 %`, slope `1.0000` below
  and `0.4837` above — so `f(a.x) >= a.f(x)` and the wrong order always
  under-prices. Every `r=2` price I published is low by a further **1.59x**.
  Instrument `research/pricing_order.py`, 20 checks, 2 positive and 3 negative
  controls, exits 0; it reproduces both published columns before correcting
  them.

**Corrected `r=2` route: `+0.6931 .. +1.2702 %` of score** — 1.29x to 2.37x the
deficit, at the low end of *both* mixtures and *both* ends of the transfer band,
with a ceiling tax of zero by construction. It is the campaign's
highest-value live experiment and it is already assigned as **E59 / PR #62**.

Standing count of my pricing errors: **four**. Two were caught before they
steered work; two were not. Errors 3 and 4 are both "a scalar applied at the
wrong place in the chain". **Every price must now be stated as a chain of named
factors in a fixed order with the basis of each factor labelled.**

Unresolved and blocking final figures: `psi` is `0.6736` in the live pricer and
`0.693391` in my own files. If `0.693391` is right, every price above rises a
further `x1.029`. Nobody has reconciled the two measurements.

Three consequences drive everything currently in flight.

1. **Only an edit that leaves the QMV kernel maximum unmoved can win.**
   `e27_m5_only` carries 95 % of E27's full register dose and `e27_m9_only`
   90 %. Under both residual shapes -- additive tax and multiplicative factor
   -- every single-cell composite is forecast **negative** at rank
   (`<T,5,5>` alone -0.20..-1.01 %; `<T,9,5>` alone -0.08..-1.65 %). A
   single-cell promotion assignment was drafted and cancelled.

2. **The shared register ceiling has never been measured in an end-to-end
   decode leg.** Every bound we hold (P4's approximately-zero term, E49 Arm
   2's absent dose-response) comes from the isolated width sweep, and that
   sweep mispredicts E27 by 2.5 points -- it is demonstrably blind to whatever
   costs E27 its score. The `ceil_only` arm settles it: an unreachable
   `case 10:` that pays the register dose and is never dispatched.

3. **No QMV table edit has ever reported the absolute serial leg.** At M=1 the
   serial leg shares the single QMV allocation (183(B)), so a shared regression
   **inflates** the local ratio exactly as a shared improvement cancels in it.
   `program.md` warns about the cancellation direction only. Every arm now
   reports both legs in absolute seconds per token.

## Live experiments

| PR | student | question |
|---|---|---|
| #57 | qwen-askeladd | E55 `<T,9,5>` end to end. **Terminal result in, revision requested.** Clean -4.2952 % local leg win, bitwise exact at 512 tokens including post-EOS continuation, 14/14 negative controls firing. Census 129 against the shipped 108, so it fails the register gate and is **not a candidate on merit**. It is nonetheless **register-identical to E27** while carrying only one of E27's two cells, so an official score contrasted with E27's receipt isolates the M=5 cell at rank **with the ceiling term cancelling exactly**. Revision asks for `--local-submit` plus reconciliation of 189(G). |
| #59 | qwen-edward | E56 stream-aware draft depth schedule. |
| #61 | qwen-alphonse | E58 round dispatch census and buffer batching -- also an independent occupancy cross-check on the ceiling question. |
| #62 | qwen-thorfinn | E59, the `r=2` row-block route itself, gated on its own register census. **The highest-value experiment in the campaign**: `+0.6931..+1.2702 %` after the 190/191 corrections, frontier-taking at the low end of every mixture and every transfer end, and immune to the ceiling question by construction. |

## Potential next research directions

**🔴 Tier 1, and it is now the campaign's best single idea -- the surviving QMV
route.** `<T,5,5>` at `rows_per_simd = 2` over two sequential row blocks. It is
bit-exact by construction (same 8 rows, same per-row dot products, same
within-row accumulation order) and is predicted to census at 91 or 100
registers against a shipped kernel maximum of **108**, so it does **not** raise
the ceiling. That maximum is a **legality floor** pinned by M=7, whose only
legal accumulator counts are {4, 5, 7} and whose cheapest legal split {4,3} is
mixed and so pays askeladd's `+4` (187(P), corrected). No retabling can lower
it, and no NA=5 table can read below 125, so this is the only route that can
ever fit under it.

Its ceiling tax is **zero by construction**, which makes it immune to the
additive-versus-multiplicative question that governs everything else in the
QMV direction. After **both** pricing corrections (189(D) prefill rebase and
191 shrink-then-price) the calibrated forecast is **+1.1598..+1.2702 %** under
e53_mid and **+0.6931..+0.8175 %** under e48 — **1.29x to 2.37x the 0.5367 %
deficit**, and 2.45x to 4.49x the ranked MDE, at the low end of both mixtures
and both ends of the transfer band. Even on the `/3.55` branch that 188 refuted,
e53_mid clears the MDE by 1.71x. Gated on measuring the real `r=2` tax at NA=5
(+10.54 % at NA=4, but the `x` volume is 25 % larger) and on its own register
census. Assigned as E59 / PR #62.

**The mixture dispute is still open, and nothing in flight resolves it**
(188(E), retracted). #57 measured the **local** fixture's `f9 = 55.4 %`, which
confirms my own local cost-weighting of 53.45 % to 3.6 % and says nothing about
the ranked share. 184(D) proved the ranked share is unidentifiable from the
receipt by any moment-based method. The route must therefore be priced across
the whole mixture band -- which, after 189(D), it survives at both ends.

**Tier 1 -- RESOLVED this round (188).** The `/3.55` divisor is **refuted** for
QMV decode changes. The 7.58x prefill advantage is the `qmm_nax` *matmul*
signature, and the scored decode path dispatches `qmv_fast` at every width
M <= 9, switching to `qmm` only at M=10. The M5's arithmetic advantage is
therefore unreachable from the decode QMV kernel by construction, so
`tau_qmv ~= R` and the multiplier is near 1.0. Memory-traffic-bound is **not**
the same transfer class as arithmetic-bound; 186(D) was wrong to group them.
Price any local win as `delta_local x (R / tau)` with `R = 2.1383` and state
`tau` explicitly. Also: `g in [0.7388, 0.7778]` and `h in [0.8343, 0.8617]`
are different numbers -- `h` is `g` mean-pinned at depth 4 -- so report their
union unless one form is justified.

**Tier 1 -- the `kL = 1024` near-tie exposure** (185(D)). Arm A of E57 declared
two distinct top-two tuples at positions 1022 and 1024, both inside round 76,
the single reachable `kL=1024` round, at qL=4 with no chunk. The ranked
512+512 window **always** reaches this boundary. This is a survival direction,
not a speed direction.

**Tier 2 -- dispatch-count reduction. Re-priced by ledger 190; the published
"2.2 % / 2.0 % / 7.1 sd" is withdrawn.** A dispatch costs **at most** 22.26 us
(the constant divides a timing delta by a dispatch-count delta, and the added
dispatches did real work). Arm A's 6163 SDPA dispatches are at most 0.748 % of
the local leg. The ranked count must be **recomputed**, not carried: ranked
`beagle` runs 107 rounds at mean draft 4.5327 against our 76 at 6.5132, and
dispatches per round are a step function of width. Two routes agree:
**0.86 .. 3.44 %** of the ranked leg by direct recount, **1.90 %** by 188(A)'s
`R/tau` at `tau = 1`. Low end is 3.0x MDE. This is a latency-bound term, the
most favourable transfer class we have. **The band is near-uniform across all
eight prompts (0.85 .. 3.5 %), so the win moves the median directly rather than
through the central-pair weight** -- a better argument than the beagle-specific
one E58 was assigned on. PR #61 owns the census, and must now report it **per
round width** `d(M)` for `M = 1..9`, because a leg total is unidentifiable at
rank by 184(D) while a per-width table is not.

**🔴 Tier 1 QUEUE -- assign to the next free student, in this order.** From a
frontier-model synthesis over the full ledger, checked against `benchmark.json`,
the 712-tree rival field, and every refutation on record. All four have a
zero-GPU or near-zero-GPU first rung, so none of them blocks on a Mac.

1. **Certified exact target LM-head screening.** The verify epilogue evaluates
   all 248,320 exact target logits per round (~715 MB of unique weight stream)
   and then reduces top-two. Only rows that can reach top-2 need exact
   evaluation. An offline, input-independent conservative bound plane (per-row
   block-max plus per-group scale norms) screens rows; survivors get exact
   affine-4 evaluation with **identical per-row arithmetic**, so top-two IDs and
   values are bit-identical whenever the survivor set provably contains the true
   top-2. It lands in a **new kernel library**, so it has **zero interaction
   with the 108-register floor**, and the sidecar lives in
   `Sources/MLXFastTransform/`, which **no rival tree has touched**. LM-head plus
   top-2 is 5 % of local round time and is bandwidth-class, so it transfers near
   1:1. A 60-80 % cut is `+2.0..+2.9 %` of score; even 25 % capture clears the
   deficit. **First rung is free**: dump traced per-round hidden vectors, compute
   the bound plane in Python, measure survivor density p50/p99. Kill if plane
   bytes plus survivor bytes reach ~0.5x the stock stream. Named by
   `laguna-to-qwen-speedup-map.md` as the largest unclaimed concept, and
   Laguna's version was **promoted**. Never measured here -- grep confirms it.
2. **`mx.compile` the head draft chain.** Each draft step pays ~2.4 ms of **host
   graph build** (`Qwen36MTPBlockSession.swift:649, 1048-1049` says so in its own
   source). At beagle/medicine depths that is 11-13 ms per round of host time --
   the transfer class ledger 190 just re-priced upward. `CompiledDecode.swift`
   and `CompilableKVCache.swift` are editable and have **zero mentions across all
   712 rival trees**. The full target is ineligible (48 SSM layers) but the head
   is fc plus one full-attention layer plus norms, which is exactly the eligible
   shape. 181(C) retracted the *recovery* claim, not the mechanism: E29 closed
   commit geometry, never graph build. Realistic capture `+0.5..+2.0 %`.
   Use an `MLX_`-prefixed switch; `MLXFAST_` is stripped by the worker allowlist.
3. **Hierarchical certified shortlist for the head's coarse readout.** The flat
   2-bit scan over 98,336 compact-vocab rows is ~40 % of every head step, and the
   head step is ~86 % pure weight streaming at a measured, saturated 243 GB/s, so
   byte cuts convert near 1:1. Replace it with an 8-bit per-block-of-64 upper
   bound (~1/32 of the bytes) plus an exact 2-bit scan of only the blocks that
   can reach the top-32 shortlist. The shortlist is provably identical, so drafts
   are identical and the frozen accept trajectory is preserved. The artifact
   derives from declared head weights and must be digest-pinned under
   `mtp-head/` (428 MB of the 2 GiB cap is used). **This changes zero weights**,
   so it is not the head-replacement direction that two scored receipts closed.
   `+1.5..+2.2 %`. First rung is free: first-stage recall of the true top-32 and
   surviving-block density, from traced steps.
4. **A composition vehicle for the five exact sub-MDE wins.** Five hand-verified
   bit-exact micro-wins have never landed because each is individually sub-MDE
   and the ledger keeps deferring them for want of a vehicle: the frontier's
   `warmTargetLaterWindowSDPA` import (the only mechanism present in the promoted
   frontier and absent from us), `pendingPrimaryDevice`, dead-KV-GEMM elision,
   fused last-merge plus final RMSNorm, and the top-32 finalize k-way merge.
   One PR, five hunks, each with its own bit-exactness fingerprint A/B, one
   pooled ABBA absolute-leg measurement, leave-one-out attribution only if the
   pooled result clears 3x the local floor. Aggregate `+0.2..+0.5 %` at near-zero
   risk. **Hand-apply hunk by hunk; never file-copy** (181(J)'s `reachedStopToken`
   trap).

Lower in the same synthesis and not yet worth a slot: GDN checkpoint economics
(the scan writes `(S-1) x 151 MB` of fp32 mid-states per drafting round and
nobody has measured who consumes them -- the free first rung is to split
`rollbackRoundCount` by `draftCount`); GDN values-per-thread (bit-identical by
construction, but the loads are largely L2-served); and a receipt-calibrated
acceptance prior, which belongs to edward's live E56 as a calibration arm rather
than a competing slot.

**Tier 2 -- compose-only wins, never a dedicated slot.** paul-hf's dead-KV-GEMM
elision (provably bit-exact, ~0.04 %); `pendingPrimaryDevice` (pure slice);
fused last-merge plus final RMSNorm (runs on both legs, so compare matched
absolute time); the top-32 finalize k-way merge (zero FP arithmetic, 256 -> 32
threads); item 146's latch release valve.

**Tier 3 -- reopen only on a named trigger.** The seed prefill is scored
(8.44-9.05 % of every leg) and halving it would be worth +4.37 %, but it is
unreachable: it runs at 93.5 % of our own dense-bf16 ceiling and the ranked
host's 7.58x advantage is the `qmm_nax` signature, which needs GPU gen >= 17.
Our host is gen 16. **Reopen only with a gen >= 17 host.**

**Plateau protocol status.** Not at a plateau. E54 produced a decisive negative
that redirected the whole QMV direction in one round, and the surviving `r=2`
route has a forecast that clears the deficit. If the `ceil_only` control
returns stop-rule 2 (additive refuted, multiplicative stands), the entire QMV
width-table direction closes and the campaign should escalate a tier: away from
kernel width tables and toward dispatch-count reduction, scheduler shape, and
the head path.
