# SENPAI Research State

- 2026-08-22 07:00 UTC
- Track `qwen3.8-27b-mtp-v1`. Advisor branch `senpai/qwen38-mtp-r1` at `3f40d9b0`.
  Campaign base `origin/main` at `770a3ff2`. Organizer `upstream/main` at `fac135f2`.
  Crown `bc070b7b` at 3.35922017, unchanged since 01:48Z.
- One official submission in flight: `cf9a9eda`, sent 06:20:41Z, `validating`.

## Most recent research direction from the human researcher team

No new direction. The standing instruction holds: keep every Mac productive, compose
mechanisms rather than resample, and submit autonomously.

---

## The number to beat, and what it really is

```
crown bc070b7b, published                     3.35922017
the same content, byte-identical redraw       3.34664074   <- the reproducible level
our own per-prompt floor envelope             3.34784
our best full fast-mode receipt, 44559d02     3.34351272
absolute per-prompt floor over all solvers    3.36504      <- the composition ceiling
```

**Finding 96, this cycle.** A declared byte-identical resample of the crown content
published 3.34664 with a mode index of -13.4240, against the crown's -13.4103. Same
content, same measurement mode, **0.374 % apart**. Twelve rows now descend from that
frontier and every one is below the crown. The crown is a favourable draw of content
whose reproducible level is about 3.347, and our own per-prompt floors already sit
0.036 % above that level.

**Finding 93, this cycle.** Taking every solver's best published time on every prompt
gives 3.36504, only 0.173 % above the crown, which is inside single-receipt noise.
**Composition is exhausted.** Every remaining gain has to be manufactured.

Two places can manufacture it: the **kernel**, where Route B is working, and the
**schedule**, where nobody has worked since the organizer.

---

## Current research focus

### 1. Land Route B (thorfinn, PR #121) — the strongest measurement in the campaign

Rung 5e, W&B [`zkcfcaxr`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/zkcfcaxr).
Eight gated legs at 512 tokens, ABBA, control pre-E121:

```
off       n=4  median 0.030754 s/tok  sd 0.000024
sumtable  n=4  median 0.029447 s/tok  sd 0.000028   = +4.249 % leg
```

Arms do not overlap; the effect is 37 times the two-sigma resolution; the candidate arm
ran colder, so drift works against it; every leg exact with a closed row ledger.

Remaining path: merge `3f40d9b0`, then rung 5g against the E121 control, then rung 5f
`--local-submit`, then submit. Pre-registered 5g prediction **+3.83 % leg**; below
+3.5 % means the F90 overlap with E121 is larger than priced.

Ranked estimate is +1.8 to +2.2 %, and it is the open question. Beagle's mean width
5.382 straddles the best width above 4 (M=5, 4.82 %) and the worst (M=6, 2.29 %), a
factor of 2.1, and beagle alone carries 0.4862 of the marginal weight. Thorfinn owes
three labelled models: A measured local histogram, B ranked mean-width point estimate,
C adverse bracket.

### 2. E127, the depth-price refit — the largest unassigned lever (edward, next)

**Finding 94, this cycle.** `Qwen36MTPBlockSession.costModelDepth` prices every draft
step at a uniform `headStepCostRatio = 0.18`. The `.pbfit` arm beside it was fitted in
E68 to a dispatch table whose width-5 step cost 13.4 ms and whose width-6 step cost
27.3 ms. The code comment demands a refit whenever the QMV group shapes move. **They
have moved three times since:** E100, E110, E121.

Reconstructed live marginal steps `0.004 0.045 0.266 0.143 0.447 0.139 0.126 0.499`
against the shipped flat `0.180`. Simulated against the oracle depth over the acceptance
range that carries the median, the shipped arm is **3.9 % to 19.1 % of candidate time**
away from optimal, because it over-drafts past a width-6 cliff that costs a full extra
weight-stream pass.

Two independent supports. E68 measured a correctly fitted shape at **-3.500 %**
end to end over nine legs. And the only pure schedule change ever made at the board
frontier, the organizer's P1 to P2 move, was worth 1.8 to 4.3 % of candidate time on
exactly the five prompts that carry the median, while moving plutarch only 0.076 %.

About ten lines of diff. Bit-exactness is structural: the schedule changes only how many
drafts are proposed, and the target verifies every emitted token.
`Qwen36MTPBlockSession.swift` is unowned.

### 3. Settle the transfer law (askeladd PR #126, alphonse PR #127)

The two in-situ anchors now **bracket unity from both sides** on the same host, fixture,
window and width histogram, for two mechanisms that touch the same activation add tree:

| anchor | mechanism | isolated-to-in-situ |
| --- | --- | --: |
| E121 rung 2 to rung 3 | **adds** a threadgroup exchange | **2.04** (1.55 re-weighted) |
| E120 rung 5e | **deletes** per-round arithmetic | **0.763** |

No scalar fits both. E125's Stage 2 deliverable is now a **class-by-regime table**, not a
factor, validated against both anchors rather than fitted to them.

### 4. Build a median-regime acceptance fixture (edward, Stage 0.5 of E124)

**Finding 92, this cycle.** 100.0 % of the published median's marginal weight sits on
prompts accepting 0.83 to 0.90 at depth 4.4 to 6.1. Every local prose fixture the
campaign owns accepts 0.44 to 0.52. `benchfixture` at 0.877 is the only local fixture in
the right regime, and it is a long-copy gate. Timing work is safe; acceptance work is
not. Eleven Gutenberg seeds in the eight published domain labels are being screened now.

---

## What closed this cycle

- **`noislands` / E124.** Closed as a net negative. E82 already measured the same
  mechanism at 512 tokens with clean arm separation: +0.366 % **slower**, one extra
  round. Edward's corrected byte model reproduces that to 0.049 pp. The acceptance
  repayment is 1.8 times the byte saving. Arm `kv`, never measured, has a ceiling of
  +0.048 to +0.065 % ranked at zero acceptance cost.
- **"Draft deeper on prose."** E122 forced depth 7: accepted tokens per round rise 11 %
  for a 77 % round cost. Closed on prose. Not closed in general.
- **Composition of rival mechanisms.** Finding 93. The envelope is the crown.

---

## Potential next research directions

1. **Refit the depth price level as well as its shape.** `makeMeasuredDepthPrice`
   rescales every arm to `maxDepth * headStepCostRatio = 1.44`, so no arm can correct a
   wrong level. F13 puts the true head cost near 0.026 local and 0.0075 ranked against
   the shipped 0.18. The rescale constraint is itself a candidate defect.
2. **Re-open margin-conditioned depth in the median regime.** E122's pooled AUC of 0.5109
   is a null on zero-weight fixtures; `benchfixture` read 0.7998. Separately, every one
   of E122's 1,761 margins is an exact multiple of 2^-4, which is bf16 catastrophic
   cancellation. A higher-precision margin is untested.
3. **Measure the wide-QMV leg share as a function of realised width.** Finding 95 voids
   the 0.6070 constant at the level. An independent width sweep would either confirm
   width dependence and collapse the E120 frame anomaly to 1.000, or leave a real frame
   effect to explain.
4. **The beagle acceptance reopener.** `qwen35DraftSelectKernel` discards every runner-up.
   Finding 69 leaves 21.6x unspent acceptance headroom against today's exchange rate, and
   beagle is the binding prompt at coefficient 203.
5. **C1, the sign-sketch first pass.** Priced at +0.23 to +0.34 % ranked. Unowned. Its
   kill rule needs re-derivation against the Finding 92 regime before it can be assigned.
6. **A schedule experiment that is not a depth price.** The organizer's P1 to P2 move
   changed per-prompt depth targets directly. Nobody has revisited the depth **cap** of 7
   or the two margin overrides at d=0 and d=1 since they were set.

---

## Standing campaign constraints added this cycle

- **Rule 76.** No acceptance measurement from a fixture outside the median-carrying
  regime, accept >= 0.80 and depth >= 4.4. Stratified, never pooled.
- **Rule 77.** Every leg-share coefficient carries the realised mean width it was
  measured at.
- **Rule 78.** No single-receipt comparison between two solvers without both F76 mode
  indices.
- **Harness defects 26 to 30.** ANSI colour in `yukon submissions`; short-window
  dilution measured at 3.8; `MLXFAST_*` never reaches a worker leg; `e123_arms.py` no
  longer builds; the arm palindrome cannot remove the clock ramp on short arm sets.
