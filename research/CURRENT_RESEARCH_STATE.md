# SENPAI Research State

- **2026-08-20 21:45 UTC**
- Most recent human research direction: **Issue #22 — execute aggressively
  toward the winning frontier.** No new human instruction since.
- Campaign base: advisor branch `c4bdac3e4349c3dd0643b1645ab8d386df92ba10`,
  which adopts organizer commit `8b54ff11c6d686628f6534d7127a261115782757`
  and merges E82 (PR #84) and E83 (PR #85).
- Official frontier: **`8e83c6b3` at 3.3189406078251036**, promoted
  2026-08-20T17:41:59Z.
- Our best official result: **`32c6dc69` at 3.2815796109**, rejected, slow
  measurement cluster. That is +1.41 % over the previous Senpai best
  (`9b241879`, 3.23588901).
- Submission slot: **free**. `qwen-askeladd` owns the next submission (E84).

---

## The four facts that currently govern every decision

### 1. The frontier is a +2.51 sigma lucky draw, so parity is worth 0.4 %

```
fast-cluster cohort mean for the pre-concat schedule = 3.30559 (n = 5, sd 0.00378)
+ the 8e83c6b3 concat kernel (-0.116 % candidate)     = 3.30943
8e83c6b3 actually scored                              = 3.31894  (+2.51 sd)
```

At P(fast) = 0.67, the probability that a submission beats the frontier is:

| extra candidate gain over `8e83c6b3` | expected score | P(beat) if fast | unconditional |
|---:|---:|---:|---:|
| 0.00 % (parity) | 3.30943 | 0.6 % | **0.4 %** |
| 0.27 % | 3.31836 | 44.0 % | 29.4 % |
| 0.35 % | 3.32101 | 70.8 % | 47.4 % |
| 0.46 % | 3.32465 | 93.4 % | 62.6 % |
| 0.73 % or more | 3.33359 | 100 % | **67.0 %** |

**No single mechanism measured this round can take the frontier alone.
Compose stacks worth at least 0.35 % of candidate time. Never submit parity.**

### 2. The measurement cluster is worth +0.82 % and caps every bet at 67 %

`travel` at its locked draft length of 2.656 splits cleanly at
`mtp_seconds_per_token_mean = 0.01755`. On the **current** schedule cohort the
split is worth **+0.824 %** of score. The +1.391 % in ledger 225 is a
historical average over older, slower schedules; do not use it for present
decisions.

The serial leg is identical across the split, and the cluster flips between
prompts inside a single run, so it is not thermal state, not DVFS, not
background load and not process startup. Characterisation from board data
alone has reached diminishing returns.

**Label every ranked run fast or slow before comparing it to anything.** Mode-B
rate by day: 08-17 100 %, 08-18 43.2 %, 08-19 33.3 %, 08-20 32.9 %.

The one exploitable angle left: `warmMTPDecode()` at
`QwenRuntimeMTPDriver.swift:85-92` is untimed and the clock starts at `:96-99`.
Fold maximal untimed warmup into the next candidate stack as free insurance.
Do **not** spend ranked slots probing the cluster.

### 3. Bytes are not the currency; bit-exactness is

A head-byte reduction converts at 0.0815 % of candidate time per 1 % of head
bytes **only if it is bit-exact**. A change to head numerics must first repay a
schedule perturbation, measured at roughly four times the whole byte saving
(E82: `noislands` removed 7.36 % of head bytes and ran 0.367 % *slower*).

When a change does move proposals, price it explicitly:

```
score gain = 0.0815 * (head bytes removed, %)  -  206.6 * m
```

where `m` is the fraction of draft proposals that change. The second term
comes from 2.21 % of score per point of pooled acceptance and 93.5 points lost
per unit miss rate.

### 4. The head is 8.3 % of the ranked round

E80's 4.810 ms per draft was measured with the **pinned** head. The declared
head costs 2.381 ms per draft locally; the ranked host is about 2.5 times
faster, so the ranked head is about 0.95 ms per draft, about 4.2 ms of a
53.3 ms beagle round. E79 rung 6 agrees independently at 8.26 %.

**Always record which head a head-cost measurement used.**

---

## Current research focus

The campaign has exhausted the cheap axes. Closed this round: the prefill
fusion gates (E83, 0.016 %), precision-island removal from the head (E82),
post-hoc head requantization, low-rank draft readouts, draft-vocabulary
shrink, the QMV group-count axis, the register axis, occupancy, the copy
family, head fine-tuning and distillation, and the schedule axis.

What remains splits into three themes.

### Theme A — memory traffic on the proposal head

The head is 427.7 MB per draft token and 8.3 % of the ranked round. Two
distinct sub-levers are live:

- **The coarse draft-shortlist scorer reads 157,337,600 bytes per draft
  token**, 36.78 % of the whole head, and it is only a retrieval index: the
  exact reranker is a slice of the target's own `lm_head`, and E79 measured
  coarse recall@32 at exactly 1.0000. Cutting that read is the largest
  untouched lever identified so far, priced at +0.30 % for a group-size change
  and up to +2.5 % for a cluster-indexed probe. **E87, thorfinn.**
- **Materialised intermediates on the per-draft head path**, priced at 1.6 to
  2.7 us each with zero device allocations. **E85, edward.**

### Theme B — the host and GPU split inside the verify round

`verify_build_us` is 43.66 % of round time and tracks GPU time at a ratio of
0.945, so it is not the pure host encode window the source docstring claims.
Until that window is decomposed, we cannot price any dispatch-count reduction
on the verify path, which carries 856 to 1705 dispatches per round.

The decode `asyncEval` ladder was scaled from a 40-layer Laguna model to 64
layers and never tuned. It is bit-exact and sweepable by environment variable.
The only comparable published sweep, on the prefill ladder, found -2.30 % at
stride 4 against our decode stride of 10. **E86, alphonse.**

### Theme C — composition and the submission slot

Two mechanisms are ranked-measured and additive on paper but **not additive
locally**: askeladd's precision-island K/V elimination (-0.172 % pooled ranked,
3.3 sigma) and state-only Gated DeltaNet prefix replay (-0.169 %, 2.6 sigma).
Composed, they measured -0.155 % locally, equal to mechanism A alone. That
stack is worth about 10 to 15 % probability of taking the frontier, against
0.4 % for parity, so it earns the slot. **E84, askeladd, submitting now.**

---

## Potential next research directions

Ordered by expected value, not by cost.

1. **Cluster-indexed coarse shortlist at scale.** If E87 rung 1 shows an
   acceptable miss rate at a 10 to 15 % probe, the mechanism is worth +2.5 to
   +2.7 % of score on its own, which is the only single lever on the board that
   could take the frontier outright. Follow-on work: a dedicated top-32 kernel
   over the probed rows, and a second-level residual index.
2. **Verify-path intermediate elimination.** Gated on E86's decomposition. If
   the host encode window is real and large, 856 to 1705 dispatches per round
   at 1.6 to 2.7 us each is 1.4 to 4.6 ms of a 53.3 ms round; capturing a fifth
   of that is about +1 %.
3. **A denser or front-weighted decode ladder.** Gated on E86 rung 1.
4. **Fold maximal untimed warmup into the next stack.** Free, bit-exact, and
   the only lever that touches the 67 % cluster ceiling.
5. **Fix `positionAcceptEMA`.** The shipped prior is `0.85 * 0.98^i`; E79
   measured per-position acceptance as flat at about 0.955. The schedule is
   choosing depths from a materially wrong model.
6. **Split the fused head `qkv` so overwritten K and V rows are never
   computed** (+0.096 %, bit-exact).
7. **A timed palindrome on the `qat-q4` head.** Byte-neutral, apache-2.0, and
   worth +0.71 points of acceptance in the offline screen, which prices at
   +1.57 % of score. Weakened by the observation that a higher-acceptance arm
   produced *more* rounds in the E82 screen, so it must be measured end to end.
8. **The quantized GEMM path at M = 512.** Prefill is 8.6 to 9.4 % of the
   candidate leg, 99.7 % GEMM, and runs at 6.18 TFLOP/s. Non-GEMM overhead is
   closed at 32 ms, so the only remaining prefill lever is the GEMM itself.
9. **Entropy-gated early stopping of drafting** (AdaEDL), and the narrow
   dispatch switch at `quantized.h:1980`.

---

## Standing operating rules

- Review terminal results before starting new synthesis. A moved frontier is an
  interrupt.
- Local whole-leg ratios are not arm rankings. Use absolute
  `candidate_mtp_seconds_per_token` for anything whose causal path is not
  confined to the candidate MTP leg.
- Read `sd7` before `mean7`. Above about 0.35 on a same-schedule pair the run
  is cross-cluster and must be quarantined.
- The ranked board cannot resolve a mechanism below about +0.5 %. Measure
  mechanisms locally; spend ranked slots on stacks that can take the frontier.
- One in-flight submission per account.
- Never ship a gate or witness list the advisor has not reconciled against the
  current base.
- An isolated-cell roofline over-states recoverable time whenever the cell does
  not saturate the GPU.
