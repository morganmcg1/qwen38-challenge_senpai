# SENPAI Research State

- **2026-08-22 07:40Z** — track `qwen3.8-27b-mtp-v1`, advisor branch `senpai/qwen38-mtp-r1`, base `770a3ff2`.
- **Most recent human research direction:** none received this round. The campaign is
  running autonomously under `senpai/program.md`.

---

## Where the score stands

```
crown bc070b7b (francip), published           3.35922017
the same content, byte-identical redraw       3.34664074   <- the reproducible level
absolute per-prompt floor, all solvers        3.36504      <- the composition ceiling
our best full fast-mode receipt 44559d02      3.34351272
post-revert base (== 7bef7d4c tree), fast     ~3.34136
current base WITH E121                        3.26815      <- a -2.10 % regression, being reverted
```

The board has **saturated**. Composing every solver's best published per-prompt time beats
the crown by only 0.173 %, which is inside single-receipt noise. Nothing is left to compose.
Every remaining gain has to be manufactured, and there are only two places it can come from:
the kernel, and the schedule.

The Yukon slot is free: 8 submissions used, 0 in flight.

---

## The dominant fact of this round

**E121 published 3.26815344, a +2.10 % ranked regression, after reading −0.436 % in the
local leg frame.** The transfer coefficient from the local leg frame to the ranked runner
was approximately **−4.8**. Not attenuated — inverted.

The *shape* transferred perfectly. Regressing the per-prompt ranked change on candidate
shapes gives R2 0.9492 for cost per verified row, through the origin, at 0.2274 ms per row.
The mechanism was real; our frame mis-priced it.

This reframes the campaign. The binding constraint is no longer "find a faster kernel". It
is **"can any local measurement price a ranked kernel change at all, and under what
conditions"**. Two of the four students are now on that question directly.

---

## Current research focus

### 1. Ship Route B and take the frontier — the critical path

Route B (thorfinn's custom cross-row QMV kernel with a hoisted activation-sum table) is
merged. It measured **+4.249 % leg** against the pre-E121 control in a gate-qualified ABBA
with an effect 37 times the resolution, and **+3.813 %** against the post-E121 base at 44
times the resolution. The 0.436 pp difference between the two sessions is exactly the
independently measured in-situ cost of E121, which is the overlap F90 predicts.

Ranked repricing matters here. Local mean verify width was 7.359; the ranked F83-weighted
mean width is **5.308**. Route B's per-width gain drops by a factor of 2.1 across the
M=5 / M=6 boundary, and beagle sits on that boundary carrying 0.4862 of the marginal weight.
So model B lands at **+1.918 %** and the adverse bracket C at **+1.321 %**, not at the
+4.036 % the leg reading implies. Only model C on a slow draw misses the crown.

Sequence: alphonse reverts E121 → I merge → thorfinn rebases, runs the proven pre-submit
chain, and submits. That receipt is also the Rule 81 confirmation Route B is entitled to,
and its attribution is clean because the bare post-revert tree is scored-surface-identical
to a tree that already has a receipt at a mode-corrected 3.34136.

### 2. The schedule splits into price and estimator, and only the price is dead

A cost curve fitted over **147 official runs** settles the price question. At the inverted
per-step acceptance of the median-carrying prompts (p in 0.934 to 0.966), the shipped
uniform 0.18 depth price already lands on the ranked optimum, the cap at d=7, under every
non-degenerate acceptance shape. F83-weighted ranked loss is **0.000 %**. The refit axis is
closed and E127 was cancelled before assignment.

But our realised ranked depth is beagle 4.382, medicine 5.256, essays 5.087 — well short of
the cap. The price is right and the walk still stops early, so the gap is in the **reach
estimator**. Pricing that gap on the fitted ranked curve gives about **+2.6 % on the
published median** under a constant-p reading, with no kernel work at all.

The suspect is a strictly-downward `min(p, conf)` margin override at depths 0 and 1, driven
by a signal E122 measured at pooled AUC 0.5109 — indistinguishable from random — but at
0.7998 on the one in-regime fixture. Pooling destroyed the stratification.

The +2.6 % is an upper bound. How much survives depends on the uncensored per-position
acceptance decline, which the shipped adaptive policy is structurally unable to observe.
A forced-depth-7 arm measures it directly, and the clairvoyant per-round `oracle` bounds
every possible estimator improvement, so this experiment retires the area either way.

### 3. Can a local frame price a ranked change? The transfer table

Five anchors now exist and no scalar fits them:

| anchor | from | to | host change | value |
| --- | --- | --- | --- | --: |
| E116 | share term | share term | none | 1.000 [0.963, 1.038] |
| E118 | isolated | in situ | none | 1.66x |
| E121 | isolated | in situ | none | 2.04x |
| Route B 5e | kernel frame | leg frame | none | 0.763 |
| **E121 ranked** | local leg, g16s | ranked leg, g17s | **architecture** | **−4.8** |

The first four are within-host **frame** transfers. The last is a cross-architecture
**regime** transfer. Collapsing the two axes is what produced the merge error.

**F100** gives the regime axis a mechanism: the host register budgets are g16s 96 and
g17s 124, and the kernel reads 94 registers on g16s and 101 on g17s. On g16s it sits two
registers under a binding budget, so register relief buys occupancy; on g17s it has 23
registers of slack, so relief buys nothing and a barrier is pure cost. That predicts
E121's local gain was an occupancy gain that only exists where the budget binds — and it is
falsifiable by census with no GPU time, because at NA=5 the shared-sums path is compiled out.

Askeladd's own pre-submission residency census already showed the g17s loss concentrated at
the dominant widths (weighted g16s −1.80 %, g17s −6.79 %). The instrument worked; the
decision did not use it.

### 4. Acceptance work needs a regime test, and the old one was wrong

No local prose seed reaches accept 0.83 on the first 128 decoded tokens; the highest is
0.791, and every seed that clears 0.83 over a full window does so through late-window
greedy-decode degeneration. That voided every acceptance experiment screened on accept rate.

Rule 76 is amended: the regime variable is the **per-step conditional p**, not the accept
rate, because the rate falls with depth even at constant p. Re-inverted, `benchfixture` sits
at p 0.9645 — inside the median-carrying band and a good stand-in for medicine, essays,
botany and republic, which carry 0.4330 of the marginal weight together. The corpus is
weaker than we thought, but it is not empty.

---

## Potential next research directions

**Near term, already gated**

- **Template the Route B entry point on M.** A switch entry point's register count is the
  maximum over its inlined branches, so the M=5 branch alone costs 4 resident simdgroups at
  every other width. Templating projects M=8 to 94 registers and 42 simdgroups with text
  falling from ~50 KB to 6–12 KB per pipeline. Gated behind a g17s census, because F99 has
  now shown this exact channel inverting.
- **Compose Route B with the crown's kernel-1 probe select.** We already own the crown's
  kernel 2 as E101; we lack kernel 1, worth about +0.074 % ranked. Single-threadgroup, so
  expect poor M5 transfer. Cheap, but small.
- **C1, the sign-sketch or low-rank first pass on the draft path.** 1,600 B per row down to
  about 130 B, worth +0.23 to +0.34 % ranked. Its kill rule was blocked on a prose corpus
  that F98 says does not exist; the amended Rule 76 may unblock it via `benchfixture`.

**If the estimator audit pays**

- Recalibrate or remove the margin override, then confirm on a ranked receipt rather than a
  local session, because Rule 79 forbids local timing as validation for a schedule change.
- Censoring-aware per-position acceptance estimation as a standing instrument, since every
  adaptive-policy measurement on this campaign has the same selection bias.

**Bigger swings, not yet started**

- **A zero-GPU cross-architecture screen as a standing gate.** If F100 holds, a per-width
  register and residency census on g17s can veto a class of changes before we spend a
  submission. That is worth more than any single mechanism, because we have now lost one
  submission and 2.1 % to its absence.
- **Verification batching at wider row counts**, where the ranked marginal cost per row is
  7.154 ms and the ranked tier break at width 5 is 22.54 %. The ranked and local boundaries
  sit at different widths, which is an unexploited asymmetry.
- **Gated DeltaNet recurrence, snapshots and rollback.** 48 of 64 layers are recurrent and
  the recurrence is only 1.114 % of the round, but snapshot and replay cost on rejected
  drafts has never been isolated.
- **Weight loading and transformed layout.** The round is 88.6 % DRAM weight streaming over
  14.41 GB, and M5 streams at 542.8 GB/s against the local 273 GB/s. Any change to the
  bytes actually touched per round converts at 1.0 in the transfer table, the only class
  that does.

**Standing discipline**

- Rule 81 is now the merge gate: name the ranked receipt before merging a scored-surface
  change on a local read, and do not stack a second mechanism until it lands.
- Rule 79 for anything touching the schedule: price on the board-fitted ranked curve.
- Rule 56 for anything touching registers or text: census g17s, not just g16s.
