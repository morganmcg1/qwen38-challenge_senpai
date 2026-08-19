# E48 — Does a uniform QMV speedup lower the score?

Assignment `qwen38-r1-e48-score-weighted-qmv-and-uniform-sign` (PR #52, revision `r1`).
Base `fb0a09d3912477d94ed631bdb90fd04172d7b4cf`. Host: local M4 Pro (**not** the
ranked M5). Nothing here is gate-qualified, ranked-equivalent, or an official score.

> Every timed leg ran with `MLXFAST_LOCAL_COOL_GATE=0`.
> `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
> `official_or_ranked_score=false`.

---

## Part 2 — the uniform sign (priority)

*(filled in after the arms complete)*

---

## Part 1 — score-weighted per-width cost shares

### What was asked, and what is actually identifiable

The brief asked for a per-width dispatch histogram **for beagle and medicine
separately**, then score-weighted cost shares.

**The histogram is not measurable.** `beagle` and `medicine` are R2-only hidden
prompts in `fixtures/qwen3_8_27b_mtp_track.json`; the prompt text never reaches a
solver checkout, and `officialMetrics` exposes no per-round width histogram. E42
already recorded this as `absolute_ranked_share_is_identified: false`. One scalar
per prompt is observable — the mean dispatched width `mean_m`, recoverable from
the published accepted-draft rate.

So the deliverable below is an **extrapolation, not a measurement**, and it is
labelled that way in the artifact's `identification` field.

### Method

`research/e48_score_weighted_shares.py` takes the measured corpus histogram and
applies the **maximum-entropy exponential tilt** that reproduces each hidden
prompt's known `mean_m`:

```text
p_i(M) ∝ p_corpus(M) · exp(−λ_i · M),   λ_i chosen so Σ M·p_i(M) = mean_m_i
```

This is the least-committal distribution consistent with the one statistic that
is actually known: any other reweighting injects an assumption the data does not
support. Dispatch shares are then converted to **cost** shares with thorfinn's
E46 refit and this base's own instructions-per-group table
(`IPG = {2:2, 3:3, 4:4, 5:3, 6:3, 7:4, 8:4, 9:3}`, read off
`kernels/quantized.h:1924-1977`):

```text
T(M) = 16.757 + 27.532·ceil(M / IPG[M]) + 9.624·M
```

Fed the corpus histogram the script reproduces the advisor's published corpus
figures exactly, which is the check that the pricing model was transcribed
correctly.

### Result

| slice | corpus (measured) | beagle (pred.) | medicine (pred.) | **score-weighted 0.79/0.21** |
|---|---|---|---|---|
| M = 9 cost share | 53.45 % | 19.69 % | 23.44 % | **20.48 %** |
| M ∈ {7,8} cost share | 12.25 % | 8.998 % | 9.759 % | **9.158 %** |
| M ∈ {4,5,6} cost share | — | 65.69 % | 62.47 % | **65.01 %** |

beagle's predicted M = 9 **dispatch** share is 12.7 %.

### Why this matters to the ledger

The hidden prompts decode at much lower mean width than the public corpus
(`mean_m` 5.53 / 5.77), so ranked width mass sits in M ∈ {4,5,6}, not at M = 9.
Re-pricing two live ledger items against the score-weighted column rather than
the corpus column:

| ledger item | priced on corpus | priced score-weighted |
|---|---|---|
| 173(C) M = 9 prize | as published | ≈ **+2.06 %**, ≈ 2.7 sd |
| alphonse's M ∈ {7,8} arm | as published | ≈ **+0.69 %**, ≈ 0.90 sd |

The M ∈ {7,8} arm falls **below the 0.7678 % board-visible floor** under this
weighting. That is a prediction, not a measurement, and it should be treated as
a prioritisation signal rather than a reason to cancel work outright — but the
M ∈ {4,5,6} band carrying ~65 % of ranked QMV cost is the more valuable target
on the same reasoning.

Artifact: `research/e48-artifacts/score-weighted-shares.json`.
