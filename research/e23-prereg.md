# E23 pre-registration — verify-forward dispatch inventory

Committed BEFORE writing the enumerator, from structural reading only (layer counts, class
boundaries, the UNFUSED fact from E18). No dispatch has been counted at the time of writing.

Base: `c0f7e370921a14f348fa1872f2176b1b43028752`. Unit of account: Metal kernel launches issued by
one call of `Qwen36MTPBlockSession.swift:951` (`model.callWithHiddenAndNormed`) with `M` verify rows,
plus the top-two reducer at `:963`.

## Prediction 1 — which family dominates at M = 8

The **48 GDN / linear-attention layers** dominate the dispatch count.

Point estimate **65 %** of all dispatches at M = 8; I will call the prediction confirmed if the
derived share is **≥ 55 %** and refuted if it is **< 45 %**.

Reasoning: 48 of 64 layers are GDN; the in-projection is a 5-way concatenated layout that E18 showed
is UNFUSED, so it should cost several launches rather than one; and a recurrent layer must sequence
state over rows, which is the only structure in the model that can multiply launches by `M` rather
than amortize over it.

## Prediction 2 — rough magnitudes at M = 8

| family | predicted share | predicted count |
| --- | ---: | ---: |
| 48 GDN layers | 65 % | 1400–3000 |
| 64 MLP blocks | 15 % | ~400 |
| 16 full-attention layers | 10 % | ~200 |
| LM head + top-two reducer | < 1 % | < 15 |

Predicted total: **2000–4000** launches per verify forward at M = 8.

## Prediction 3 — how the count moves with M

- GDN is the only family whose count I expect to be **superconstant in M** (a per-row or per-chunk
  loop).
- Full attention gains **+16** launches when `M > 5` (one extra SDPA per full-attention layer from
  the `let split = 5` chunking) and is otherwise flat in M.
- MLP and LM head are flat in M in launch count; only shapes grow.
- If GDN turns out flat in M as well, then the **total is nearly flat in M**, and the marginal cost
  of a wider verify is bandwidth, not launches — that would be the most decision-relevant outcome
  and I will say so.

## Prediction 4 — the width where quantized routing changes

`get_qmv_batch_limit` = 10 on this host, so I predict **no** family switches out of the QMV family
anywhere in M ∈ {1..9}: every projection in the verify forward stays on a QMV-class kernel at every
shipped width. If that is wrong, the crossrow-QMV promotion story generalizes differently than the
leaderboard suggests.
