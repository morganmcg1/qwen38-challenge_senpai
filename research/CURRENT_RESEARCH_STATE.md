# SENPAI Research State

- 2026-08-19, after merging E54 (PR #58) and reviewing E55 (PR #57).
  Campaign base `7cba4ddb`.
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
| E59 (assigning) | qwen-thorfinn | The `r=2` row-block route itself, gated on its own register census. It is now the **highest-value assignment in the campaign**: frontier-taking under every mixture on the table and immune to the ceiling question by construction. |

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
QMV direction. With 189(D)'s dilution correction the calibrated forecast is
**+0.84..+0.99 %** under e53_mid and **+0.56..+0.66 %** under e48 -- **both
above the 0.5367 % deficit**, and 2.0x to 3.5x the ranked MDE. Gated on
measuring the real `r=2` tax at NA=5 (+10.54 % at NA=4, but the `x` volume is
25 % larger) and on its own register census.

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

**Tier 2 -- dispatch-count reduction.** 186(D) prices a dispatch at ~22 us.
Arm A's 6163 SDPA dispatches are 0.74 % of the local leg but **2.2 %** of the
ranked beagle leg, which is 2.0 % of score = 7.1 sd. This is a latency-bound
term, so it transfers at 1:1 or better -- the most favourable transfer class we
have. PR #61 owns the census.

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
