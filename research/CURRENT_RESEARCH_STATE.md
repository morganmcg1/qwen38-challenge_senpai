# SENPAI Research State

- **2026-08-20 22:10 UTC**
- Most recent human research direction: **Issue #22 — execute aggressively
  toward the winning frontier.** No new human instruction since.
- Campaign base: advisor branch `c05403c438caf66dd9fa8bcc53ab795a1a722b44`,
  which adopts organizer commit `8b54ff11c6d686628f6534d7127a261115782757`
  and merges E82 (PR #84) and E83 (PR #85).
- Official frontier: **`8e83c6b3` at 3.3189406078251036**, promoted
  2026-08-20T17:41:59Z.
- Our best official result: **`32c6dc69` at 3.2815796109**, rejected, slow
  measurement cluster. That is +1.41 % over the previous Senpai best
  (`9b241879`, 3.23588901).
- Submission slot: **taken**. `8630bc07-4b39-483a-a058-2003862732e5`, the E84
  stack composed on the promoted frontier, submitted 2026-08-20T21:15Z from
  `4b6119698ac4fef269054247693d26556946dca6`, validating.
- All four students hold live assignments: E84 askeladd (PR #86), E85 edward
  (PR #87), E86 alphonse (PR #88), E87 thorfinn (PR #89).

---

## The six facts that currently govern every decision

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

### 5. The verify window is 98.6 % GPU-bound, so host work cannot pay

E86 rung 0 (alphonse, PR #88), eight legs, palindrome, declared head, 512
tokens, all `--sync-head`, M4 Pro, every leg `all_tokens_matched=true` with 78
rounds and `rows_per_token = 1.1189`:

| quantity | median us per round |
|---|---:|
| host encode `H` | **2,294** |
| GPU execute `G` | **149,866** |
| `H + G` | 152,160 |
| `off` arm round | 168,895 |
| GPU idle fraction of the round | **1.4 %** |

Under the shipped ladder `verify_build_us` reads 72,330 us, and only 2,294 us
of that is host encode. The other 97 % is the host blocked on the `asyncEval`
throttle while the GPU runs. Total verify pipeline cost is flat at about
149.3 us-thousand for every active rung set, so the shipped ladder already
recovers 119 % of `H`.

This also replaces the earlier reading of `verify_build / eval_wall = 0.945`.
It is not a host cost that tracks GPU time. It is one GPU cost split into two
counters by the rung positions.

> **CLOSED: host-side operation count, allocation count and command-buffer
> boundary reduction on the verify path. To make the candidate faster the GPU
> must do less work: fewer verified rows, cheaper kernels, less weight
> traffic.**

Two supporting boundaries. Depth-0 decode reads 14.1 GiB in 60.4 ms locally,
which is 97.3 % of peak bandwidth, so the single-row pass is already optimal on
this hardware and all local headroom is the excess of the multi-row pass. The
declared head reads 427.7 MB per draft in 2.381 ms, and 427.7 MB at 226.0 GB/s
is 1.892 ms, so the head phase is 79.5 % pure weight bandwidth as well.

### 6. Leg totals overstate small effects by up to four times

Also E86 rung 0. A bit-exact change replays one identical `(depth, accepted)`
round sequence across every leg, which makes rounds pairable. Compared by leg
totals the best arm read `-0.451 %`; compared by median of paired per-round
deltas the same data read `-0.110 %`. The cause is a few multi-millisecond OS
scheduling spikes in `d_submit1_us`, `d_chain_us` and `commit_us` that a leg
total sums and a paired median rejects. Session null by the paired method:
`+51 us`, 95 % interval `[-46, +87]`, about 0.05 % of round time.

**Any bit-exact arm must be reported as a median of paired per-round deltas
with a confidence interval, with the naive leg total beside it, and with the
count of distinct round sequences across legs stated and equal to one.**

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

### Theme B — GPU work inside the verify round

Rung 0 of E86 answered the decomposition question and closed the host side of
this theme (see governing fact 5). What survives is GPU work.

Verifying `M` rows in one pass costs far more than one row, even though the
weights only need to be read once. Measured locally, in milliseconds: M1
60.372, M2 65.377, M3 72.128, M4 82.163, M5 95.568, M6 122.876, M7 149.368, M8
183.642, M9 451.747. Ledger 211 prices the ranked width tax at 19.83 % and E80
attributed 99.96 % of it to named kernels, chiefly `gdn_in_proj_fused` (qmv
M x 2060 x 1, 7.820 ms) and `fa_qkv_gate_fused` (qmv M x 1792 x 1, 2.343 ms),
with zero unclassified kernels at every width.

**That excess is now the largest single unexplored cost in the candidate.**
Depth-0 is at 97.3 % of peak bandwidth and cannot improve locally; the width
tax is the whole remaining local headroom in the verify pipeline.

The decode `asyncEval` ladder remains live as a small bit-exact component. It
was scaled from a 40-layer Laguna model to 64 layers and never tuned. Rung 0
measured `dense` at -0.087 % of candidate time by paired median, below the
0.10 % solo bar but usable in a stack. Rung 1 tests whether the mechanism is
rung density or block alignment: `Qwen35.swift:2254` gives
`isLinear = (layerIdx + 1) % 4 != 0`, so full-attention layers sit at
i = 3, 7, ..., 63, `dense` never fires on one and `default` fires on two.
**E86, alphonse.**

### Theme C — composition and the submission slot

Two mechanisms are ranked-measured and additive on paper but **not additive
locally**: askeladd's precision-island K/V elimination (-0.172 % pooled ranked,
3.3 sigma) and state-only Gated DeltaNet prefix replay (-0.169 %, 2.6 sigma).
Composed, they measured -0.155 % locally, equal to mechanism A alone. That
stack is worth about 10 to 15 % probability of taking the frontier, against
0.4 % for parity, so it earns the slot. **E84, askeladd. Submitted as
`8630bc07` at 21:15Z, validating.**

The non-additivity claim rests on a leg-total comparison whose noise floor is
0.061 % against effects of 0.077 %, so governing fact 6 applies to it. Askeladd
is re-analysing the same palindrome by paired per-round median during the
validation window. If mechanism B really adds nothing, the next stack drops it.

The composition target for the following submission is E84 plus E86 plus, if
they land, E85 and E87. Current expected sum: -0.155 % from E84, about -0.09 %
from E86, 0.03 to 0.12 % from E85, and 0 to 2.7 % from E87. Anything at or
beyond -0.35 % crosses into a better than even chance of taking the frontier
when the fast measurement cluster is drawn.

---

## Potential next research directions

Ordered by expected value, not by cost.

1. **Cluster-indexed coarse shortlist at scale.** If E87 rung 1 shows an
   acceptable miss rate at a 10 to 15 % probe, the mechanism is worth +2.5 to
   +2.7 % of score on its own, which is the only single lever on the board that
   could take the frontier outright. Follow-on work: a dedicated top-32 kernel
   over the probed rows, and a second-level residual index.
2. **Attack the width tax directly.** Verifying seven rows costs 2.47 times
   verifying one, on a path whose weights need reading once. Ledger 211 prices
   the ranked tax at 19.83 % of the round. The first job is a byte-level model
   of what one verify pass at width `M` actually reads and writes, checked
   against the crown table, followed by one concrete mechanism that reduces GPU
   traffic or GPU work per marginal row. Research in progress.
3. **Test whether the ranked host has bandwidth headroom the local host does
   not.** Depth-0 costs 60.4 ms locally at 97.3 % of a 226.0 GB/s peak, and
   30.402 ms on the ranked M5 host, which is about 474 GB/s against a 614 GB/s
   specification, or 77.2 %. Occupancy was closed at 0.52 % **on the local host,
   where the machine is already at 97 % of bandwidth and no such change could
   show anything**. That null may not transfer. A faster memory system needs
   proportionally more outstanding loads in flight to saturate. Research in
   progress. Weak point: the only ranked measurement channel is a submission,
   and the board cannot resolve anything below about 0.5 %.
4. **A denser or block-aligned decode ladder.** Gated on E86 rung 1. Worth
   about -0.09 % as a stack component.
5. **Fold maximal untimed warmup into the next stack.** Free, bit-exact, and
   the only lever that touches the 67 % cluster ceiling.
6. **Fix `positionAcceptEMA`.** The shipped prior is `0.85 * 0.98^i`; E79
   measured per-position acceptance as flat at about 0.955. The schedule is
   choosing depths from a materially wrong model.
7. **Split the fused head `qkv` so overwritten K and V rows are never
   computed** (+0.096 %, bit-exact).
8. **A timed palindrome on the `qat-q4` head.** Byte-neutral, apache-2.0, and
   worth +0.71 points of acceptance in the offline screen, which prices at
   +1.57 % of score. Weakened twice: a higher-acceptance arm produced *more*
   rounds in the E82 screen, and the E82 phase table puts the `qat-q4` round at
   183.5 ms against the declared head's 154.7 ms, which is 18.6 % slower. It
   must be measured end to end before it is believed in either direction.
9. **The quantized GEMM path at M = 512.** Prefill is 8.6 to 9.4 % of the
   candidate leg, 99.7 % GEMM, and runs at 6.18 TFLOP/s. Non-GEMM overhead is
   closed at 32 ms, so the only remaining prefill lever is the GEMM itself.
10. **Entropy-gated early stopping of drafting** (AdaEDL), and the narrow
    dispatch switch at `quantized.h:1980`.

**Removed from this list this round.** Verify-path intermediate and dispatch
elimination, which governing fact 5 closed. It was ranked second here an hour
ago and is now known to be worth approximately zero.

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
- Report every bit-exact arm as a median of paired per-round deltas with a
  confidence interval. State the count of distinct round sequences across legs
  and require it to equal one. Leg totals overstate small effects by up to four
  times.
- A null measured on a saturated resource does not close the axis on hardware
  where that resource is not saturated. Record which host produced every null.
- Prefer a mechanism-specific build witness over a generic one. A generic
  witness proves the build is fresh; a mechanism witness proves the experiment
  can move.
