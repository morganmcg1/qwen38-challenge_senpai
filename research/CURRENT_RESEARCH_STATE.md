# SENPAI Research State

- 2026-08-23 04:35 UTC
- Most recent human research direction: none new this round. Standing direction
  is the campaign objective in `senpai/program.md` — maximise the official
  decode score on track `qwen3.8-27b-mtp-v1`.

## Where the campaign stands

The bar is `684821ed` at **3.71959722580154**, promoted 01:45Z from source
`eb5eadc7`. Our own candidate frontier is `1760479a` at 3.70355222. One of our
submissions, `1db9d63e`, has been validating since 01:54Z with an additive
forecast near 3.69900.

The dominant fact about the top of the board is still FINDING 215: **the top
five rows are the same candidate to within 0.09 %**, and the 0.56 % published
spread between them is entirely serial-leg lottery. De-lucked, the crown is the
**worst** of the five. Rule 124 separates the bar from the frontier; Rule 126
prices any shaped gain as an expectation over that lottery.

## Current research focus

**1. Composition is the newly discovered risk, and it is large.** FINDING 220:
the same 25-line untimed warm measures −0.0086 %, +0.1046 % and **+1.5492 %**
on three different bases, with draft lengths digit-identical across all eight
prompts in all three. Two mechanisms that share no bytes can still interact
through the JIT pipeline library cache, the FCFS wired-residency pool, the
allocator reuse window, or the command-buffer scheduler. CAMPAIGN RULE 128 now
requires every composed submission to carry either a measured isolation of the
composition or an explicit ±1.5 pp interaction band. Our in-flight `1db9d63e`
is the campaign's first measurement of this on our own stack.

**2. Two large queue entries were deleted this round after re-reading our own
ledger.** The "unowned width-independent pool at 3.7 to 3.8 %" does not exist
(FINDING 221) — it was computed with a transfer constant the campaign retired
in Advisor Error 52, it double-counted a scope boundary, and the round closes
once the proposal head is added back. And E144's better-quantizer arm was
closed by E82 rung 6 and its metadata-coarsening replacement by E87 arm G
(ADVISOR ERROR 155, a Rule 68 violation by me).

**3. What replaced the phantom pool is bigger, better founded, and already
closed.** The genuinely width-independent part of the round is **27,727 us,
about 53 % of the median-pair round**, so a 1 % cut there is worth about
+0.53 % of published median. But it is dominated by DRAM weight streaming
already at 82 to 85 % of peak, and every route to fewer bytes has a receipt
against it. Naming this precisely matters more than reopening it: it stops us
spending rounds on a term we cannot move.

**4. The head-artifact axis is closing cleanly, with one lever left.** The
literature confirms our own measurement from theory: round-to-nearest is
provably optimal at fixed scale and zero point, and at group 64 min-max already
sits on the analytic optimum. Production evidence puts a correct INT4 draft
quantization at 0 to 1 % acceptance cost, which makes our recorded 0.82 pt gap
an **outlier** — so the remaining question is whether the artifact has a defect
or a distributional pathology, not whether our rounding rule is weak. One
mechanism sits outside everything we closed: **metadata-free column permutation
of a coupled pair**, which changes which 64 weights share a group at zero byte
and zero time cost. Its entire value is decided by one statistic that takes
minutes to compute.

## Live experiments

| student | experiment | question |
|---|---|---|
| thorfinn | E135 | width-6 register occupancy on g17s (F22, +0.5913 % expected, p05 +0.5992); then the `1db9d63e` additivity residual |
| alphonse | E141 | compact-draft-vocabulary arm B-20 (+0.4038 %, sd 0.0007, lottery-proof, beagle-weighted) |
| edward | E145 | the live width cost curve — R0 is now the campaign's highest-value zero-GPU item |
| askeladd | E144 | the head artifact, rescoped to zero-GPU: submission mechanics, bit-exact reproduction as a defect hunt, the in-branch rehearsal, and the permutation statistic |

## Potential next research directions

- **H220 — warm and residency allocation order.** The strongest unowned lever:
  up to ±1.5 %, uniform across prompts, lottery-proof, and **zero correctness
  risk**, because allocation order cannot change an emitted token. Should be
  combined with a **pipeline-construction census** — the current crown's only
  declared mechanism is skipping one unused JIT construction, and our tree
  lazily builds several kernel families. Both run through the same two shared
  resources. This is the intended E146.
- **Per-position head-side confidence as a depth policy.** Point +0.5 % of
  median, band [0, +1.5 %], beagle-weighted and therefore lottery-proof. Rung 0
  is zero-GPU on the cached E143 capture. Must carry
  `zero_weight_gain_share` and `beagle_cost_pct` from rung 0, because Rule 125
  and E140 both show this family converts accuracy into charged beagle depth
  unless the objective is explicitly the min-of-four plus beagle.
- **The C-a census resolver.** Zero GPU. Two of our own censuses disagree by
  2.05× on a quantity that scales E141's whole prize; the three candidate error
  terms are already named.
- **A data-free ranking-preservation objective for the head**, if the
  permutation statistic clears its gate. Our verifier needs only top-1 order,
  and weight MSE correlates with downstream accuracy at only about −0.65.
- **`MISS_TO_SCORE_PCT`**, still 203 by contract against 209.5 ± 93.1 measured
  and 290 geometric. Several standing prices depend on which is right.

## Standing constraints that shape all of the above

- Rule 72: one submission in flight, no re-rolling, no timing.
- Rule 121/123: predict eight raw ratios, sort them, read positions 3 and 4.
  The upper slot is a **minimum** over essays, republic, medicine and botany;
  beagle holds the lower slot on 212 of 212 replayed vectors.
- Rule 125: price an acceptance gain through the scheduler's response, never at
  fixed depth. A cell that unlocks plutarch is a warning sign — three
  independent confirmations now.
- Rule 127: no gain is uniform unless its causal path is prompt-independent.
  Anything that moves realised draft length is not uniform.
- Rule 128: a composition must be measured, not added.
- The integrity boundary: rivals are building head training corpora matched to
  the named hidden-prompt families. We do not. Every head-side method we run
  must be a pure function of the master weights.
