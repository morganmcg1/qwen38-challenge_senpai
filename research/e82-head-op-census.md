# E82: MLX ops and forced evaluations on the proposal-head path, per drafting round

A static reading of the shipped source at base `9ec6e087`, requested as an E82
deliverable. It names every adjacent op pair on the head path that computes
independent results from the same inputs, with how often each pair runs.

**Read the measurement before the list.** `research/e82-head-cost*.json` prices
the head step at about 179 MB of head tensor per millisecond on two heads whose
readout paths differ by seven op boundaries per draft, and puts host graph
build at 0.04 ms per draft, under 2 % of the step. On this path the pairs below
are real but their fusion value is bounded by traffic, not by boundary count.

## Structure of one drafting round with draft count `d`

`Qwen36MTPBlockSession.swift:1256-1292`:

| stage | calls per round | source |
|---|---|---|
| history flush, `n` rows | 1 | `Qwen35MTP.swift:140-161` `lastHiddenWithKVOnlyHistory` |
| single-row head forward | `d - 1` | `Qwen35MTP.swift:107-132` `callAsFunction` |
| readout `draftTokenID` | `d` | `Qwen35.swift:3120-3219` |

## Forced evaluation points

Two per round, and they do not depend on the head:

1. `asyncEval(draftId)` after head step 1, which carries the history flush
   (`Qwen36MTPBlockSession.swift:1275`).
2. `asyncEval` of the last draft id after the chain is built (`:1288`).

`MLX_QWEN_MTP_TRACE_SYNC_HEAD=1` adds one `eval` at `:1290`. That is an
attribution instrument, not a shipped cost.

## Adjacent independent pairs, ranked by frequency

| # | pair | runs per round | note |
|---:|---|---:|---|
| 1 | `preFcNormEmbedding(embeds)` and `preFcNormHidden(hidden)` | `d` | Independent RMSNorms of independent inputs. This is the pair `qwen35DualRMSNorm` fuses on the current frontier. It runs once per head call, so `d` times per round, not once. |
| 2 | `concatenated([e, h], axis: -1)` feeding `fc(...)` | `d` | The concat materialises a `rows x 10240` bf16 buffer whose only consumer is the matmul. A quantized matmul that reads two operands writes no such buffer. Advisor's named lead. |
| 3 | `MLX.take` on `exact.weight`, `exact.scales`, `exactBiases` | `d` | Three gathers from three arrays with one shared index array (`Qwen35.swift:3200-3202`). Declared-readout heads only. One fused gather would do. |
| 4 | slice `coarse[..., 0 ..< realCount]` then `.reshaped(...)` | `d` | Two shape ops before `qwen35DraftTop32` (`:3190-3191`). Declared-readout heads only. |
| 5 | `exactLogits.reshaped([candidateCount])` | `d` | Shape op feeding `qwen35DraftRerankKernel` (`:3208`); the kernel could index the 3-D form. Declared-readout heads only. |
| 6 | `headHidden[0..., (dim-1) ..< dim, 0...]` | `d` | Last-row slice after every head forward (`Qwen36MTPBlockSession.swift:1264-1265`, `:1281-1282`). |
| 7 | `embedTokens(nextTokenIds)` and `preFcNormHidden(hidden)` | `d` | The embedding gather and the hidden norm are independent of each other. Only pair 1 is inside the fused kernel today. |

Pairs 1, 2, 6 and 7 run on every head. Pairs 3, 4 and 5 exist only when the
head ships `draft_lm_head.*`, so the pinned head does not pay them.

## Readout op count by head profile

| profile | ops | dispatches | source |
|---|---:|---:|---|
| no shipped `draft_lm_head` (pinned) | 3 | 2 | `Qwen35.swift:3126-3153`, compact head then `qwen35DraftSelectKernel` |
| shipped 2-bit `draft_lm_head` (declared, soup-q4, qat-q4, master-bf16, kamciosz) | 10 | 6 | `Qwen35.swift:3155-3219` |

The declared profile pays seven more op boundaries per draft than the pinned
profile and is still measured on the same bytes-per-millisecond line, so a
per-boundary tax of the size once attributed to `qwen35DualRMSNorm` is excluded
by this path's own numbers: seven boundaries at 0.35 ms would be 2.45 ms per
draft, more than the whole measured declared head step of 2.381 ms.

## What this predicts

Fusing any pair above should move the head step by roughly the traffic it
removes and by no more. Pair 2 is the only one that removes real traffic: a
`rows x 10240` bf16 write plus read, which is 20 KB per row per call, or about
0.9 MB per round at `d = 6.35` and a 6-row flush. Against 427.7 MB of head
weight traffic per draft that is under 0.04 % of the step. Pair 1 moves about
50 KB per call.

Nothing on this list is worth implementing on bandwidth grounds. If the
campaign still wants pair 2, justify it by host graph build time, which this
instrument measures at 0.04 ms per draft in total.
