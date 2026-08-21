# E113 — the group boundary moved, but the ranked cost tier did not

Student `qwen-thorfinn`. PR #115. Assignment
`qwen38-r1-e113-the-group-boundary-moved-and-the-depth-rule-was-never-told`, r1.
Base `25c33465efe6fc45123b69b2accd547bb56e0642`.

**Verdict: KILL RULE 1 fires. Every assigned arm loses. No GPU work is
justified. No candidate change is proposed.**

This is an analysis-only result. No Swift source, kernel, test, or fixture file
was changed. `senpai/verify-ranked-score-boundary.sh` passes at this base.

## Identity tuple

| field | value |
|---|---|
| base | `25c33465efe6fc45123b69b2accd547bb56e0642` |
| candidate diff | none; `research/` only |
| harness | **both frames are labelled**; see each table |
| ranked receipts used | `b8b8b860` (official 3.33412148) and `44559d02` (official 3.34351272), both post-E100 |
| replay trace | `research/out/e101ctl512`, base `9d837fc2`, 512 decode tokens, 78 rounds, offered cap 8 |
| local cost census | W&B `19kgn6xi` (E106), serialized command buffers, `timing_valid=false` |
| GPU time | none. No timed leg was run |
| gate flags | not applicable; no timed measurement. `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false` for the reused E106 census |

Artifacts: `research/e113_depth_boundary.py` (the analysis),
`research/e113-artifacts/rung01.txt` (full console record),
`research/e113-artifacts/rung01.json` (structured record).

W&B run `ytddnf7p`,
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ytddnf7p>.

Reproduce:

```bash
python3 research/board_per_prompt.py fetch
python3 research/e113_depth_boundary.py
python3 research/e113_wandb_log.py
```

## Report shape

**Question.** E100 moved the QMV dispatch group boundary from verify width 5 to
width 6. `costModelDepth()` still charges one uniform price per draft. Does
refitting that price to the post-E100 boundary buy published score?

**Evidence that made it worth testing.** The source fact is real. At this base,
`Vendor/mlx-swift/.../kernels/quantized.h` instantiates `<T,5,5,true>` at M=5
(one group) and `<T,6,3,true>` at M=6 (two groups). Before E100 the two-group
split started at M=5. The shipped depth rule was never refit.

**Expected result.** The assignment predicted a marginal-cost bump at the fifth
draft and a positive gain from teaching the walk about it.

**Smallest decisive test.** Price every arm on the two available cost frames,
then replay the arms against recorded current-tree rounds and against each
ranked prompt's realised width. Zero GPU cost.

**Stop or promotion rule.** Pre-registered. RUNG 0 stops if the realised
schedule is within 0.20 % of the per-prompt optimum on 6 of 8 prompts. RUNG 1
stops if the best arm's predicted published gain is below 0.20 %.

## Deliverable — the two-frame marginal-cost table

The advisor asked for this table regardless of the verdict.

### harness=local, E106 census `19kgn6xi`, current tree

| M | partition | round µs | marginal µs | vs G=1 step | group | provenance |
|---|---|---|---|---|---|---|
| 1 | [1] | 66,374.3 | — | — | G=1 | INFERRED |
| 2 | [2] | 78,208.2 | 11,833.9 | 1.00 | G=1 | measured, n=1 |
| 3 | [3] | 90,042.1 | 11,833.9 | 1.00 | G=1 | INFERRED |
| 4 | [4] | 101,876.0 | 11,833.9 | 1.00 | G=1 | INFERRED |
| 5 | [5] | 113,709.8 | 11,833.9 | 1.00 | G=1 | measured, n=1 |
| 6 | [3+3] | 133,563.9 | **19,854.1** | **1.68** | G=2 | measured, n=4 |
| 7 | [4+3] | 144,138.8 | 10,574.9 | 0.89 | G=2 | measured, n=3 |
| 8 | [4+4] | 156,950.0 | 12,811.1 | 1.08 | G=2 | measured, n=10 |

In the local frame the bump is where the assignment predicted it: entering
width 6, ratio 1.678.

### harness=ranked route A, Finding 31 identity with the ranked rate table

| M | partition | round µs | marginal µs | vs previous | group |
|---|---|---|---|---|---|
| 1 | [1] | 31,175.3 | — | — | G=1 |
| 2 | [2] | 35,169.2 | 3,993.9 | — | G=1 |
| 3 | [3] | 39,164.0 | 3,994.8 | 1.00 | G=1 |
| 4 | [4] | 43,163.7 | 3,999.7 | 1.00 | G=1 |
| 5 | [5] | 52,947.6 | **9,784.0** | **2.45** | G=1 |
| 6 | [3+3] | 60,340.6 | 7,392.9 | 1.85 | G=2 |
| 7 | [4+3] | 67,568.4 | 7,227.9 | 1.81 | G=2 |
| 8 | [4+4] | 74,811.1 | 7,242.6 | 1.81 | G=2 |

Ranked tier ratio into width 6 is **0.756**: the step gets *cheaper*, not more
expensive. The ranked bump is one width earlier, at width 5.

### harness=ranked route B, refit from our own post-E100 receipts

Two-line fits of round µs against verify width M over the 16 per-prompt rows of
receipts `b8b8b860` and `44559d02`:

| break | a1 | c1 | a2 | c2 | step µs | max abs residual |
|---|---|---|---|---|---|---|
| M≥4 / M≥5 (same partition of the data) | 27,439.9 | 3,799.2 | 16,982.0 | 7,108.1 | 6,086.6 | **0.68 %** |
| M≥6 | 22,669.7 | 5,868.1 | 19,179.4 | 6,783.3 | 2,001.1 | 7.73 % |
| M≥7 | 21,844.0 | 6,184.2 | 67,703.4 | 0.0 | 2,570.0 | 9.18 % |
| single line | 21,392.3 + 6,319.4·M | | | | | 10.10 % |

The break at M=5 fits our post-E100 receipts to 0.68 %. A break at M=6 fits to
7.73 %, eleven times worse.

Route B also reproduces the **pre-E100** ranked curve already recorded in
`research/e99_oracle.py` (`RANKED_A1, C1 = 27181.5, 3995.1`;
`A2, C2 = 16943.2, 7233.0`) to within 0.2 % and 1.7 %.

## Finding 1 — the section-5 transfer argument is falsified

The assignment argued that because the dispatch group count changed at M=6, the
ranked cost tier must also change at M=6. It does not.

The ranked cost curve is unchanged by E100. Its break stays at M=5, exactly
where it was before E100, and route B measures that on our own post-E100
official receipts.

What is structural is the **dispatch group count**. What is not structural is
the **cost tier**. On the ranked M5 host, per-group throughput scaling is
A ≈ 2.0, so splitting [5] into [3+3] nearly pays for itself. On the local M4 Pro
host, A = 1.64, so the same split costs 1.68× a G=1 step. The group boundary and
the cost boundary are the same object only when A is large enough, and on the
ranked host it is not.

This also confirms Finding 32 consequence 1: the ranked G boundary does not move
to M=6.

## RUNG 0 — the realised schedule against the per-prompt optimum

`cpt(d) = round_us(M = d+1) / expected emitted tokens(d)`, with the per-prompt
acceptance profile calibrated from receipt `b8b8b860`.

harness=ranked route A:

| prompt | accept | realised M | d\* | cpt(d\*) | cpt(d realised) | gap |
|---|---|---|---|---|---|---|
| plutarch | 0.3333 | 1.154 | 1 | 26,377 | 31,175 | 18.19 % |
| drama | 0.4491 | 3.298 | 2 | 20,633 | 20,633 | 0.00 % |
| travel | 0.5329 | 3.656 | 3 | 16,610 | 16,610 | 0.00 % |
| beagle | 0.8340 | 5.382 | 6 | 11,964 | 12,211 | 2.06 % |
| republic | 0.9030 | 5.989 | 7 | 10,515 | 10,941 | 4.05 % |
| essays | 0.8974 | 6.087 | 7 | 10,589 | 10,997 | 3.85 % |
| medicine | 0.8922 | 6.256 | 7 | 10,659 | 11,050 | 3.67 % |
| botany | 0.8655 | 7.148 | 7 | 10,795 | 10,911 | 1.07 % |

2 of 8 within 0.20 %. **KILL RULE 0 does not fire.** The local frame gives the
same count, 2 of 8.

But the shape of the answer already refutes the hypothesis:

- On **both** frames, `d*` for every median-carrying prompt is at or near the
  cap: `d* = 7` for republic, essays, medicine and botany; `d* = 6` for beagle
  on ranked and 7 on local.
- There is **no interior optimum** for those prompts. A boundary price can only
  make one step more expensive, so it can only pull depth *down*, away from
  `d*`. It has no mechanism to help.
- The remaining gap on those prompts is the walk stopping *short* of the cap,
  not overshooting it.

Caveat recorded in the artifact: `cpt(d realised)` prices the realised mean
depth as if it were a fixed depth. The shipped walk is adaptive, and E99
measured the adaptive walk 2.58 % **better** than the best fixed depth on the
ranked curve. These gaps therefore overstate the available prize.

## RUNG 1 — trace replay on recorded current-tree rounds

The E99 traces named in the assignment **do not exist**. `research/e99-artifacts/*.json`
hold only aggregated tables with no per-round records, and `research/out/e99*`
and `research/out/e94*` are gone because `research/out/` is gitignored. Rung 1 as
literally specified cannot be run.

Substitute: the surviving current-tree trace `research/out/e101ctl512`, 512
decode tokens, 78 rounds, offered cap 8, mean recorded width 7.359, accept
0.8770. The replay reuses `parse_trace`, `token_row` and `evaluate_policy` from
`research/e99_oracle.py`.

**Positive control: the ship walk reproduces the recorded depth on 78 of 78
rounds.** The control required two corrections that are themselves worth
recording: the emission budget is `tokens + 1` (the prefill primary is committed
by the first round), and the final rounds are capped by the remaining decode
window rather than by the schedule.

treatment=observed:

| arm | ranked µs/token | vs ship | local µs/token | vs ship |
|---|---|---|---|---|
| ship | 10,655.4 | — | 22,556.5 | — |
| pb6 | 11,159.0 | +4.73 % | 23,817.6 | +5.59 % |
| pb6fit | 10,717.9 | +0.59 % | 22,705.7 | +0.66 % |
| pb5 | 10,557.9 | −0.91 % | 23,022.0 | +2.06 % |
| pb5fit | 11,541.7 | +8.32 % | 27,225.6 | +20.70 % |
| look6 (`lookahead`) | 10,596.1 | −0.56 % | 22,376.8 | −0.80 % |
| look5 | 10,514.5 | −1.32 % | 22,478.9 | −0.34 % |

Positive is slower. This trace is one easy prompt at width 7.359 and is not
representative of the published median, so it is a screen, not the verdict.

## RUNG 1b — per-prompt replay at the realised ranked widths

Each ranked prompt's realised draft length is reproduced by scaling the recorded
EMA vectors by a single per-prompt factor λ until the ship walk lands on that
prompt's realised mean depth. This preserves round-to-round EMA dispersion and
gives a width sweep. Token yield comes from the prompt's calibrated static E92
acceptance profile, so the counterfactual does not inherit the trajectory the
ship policy happened to take.

The ship arm reproduces the official published median of receipt `b8b8b860`
exactly (3.33412148), which is the instrument's calibration check.

| arm | predicted published median | vs ship |
|---|---|---|
| ship | 3.33412148 | +0.0000 % |
| pb6 | 3.27111497 | **−1.8897 %** |
| pb6fit | 3.26889283 | **−1.9564 %** |
| pb5 | 3.29880418 | −1.0593 % |
| look6 (the assignment's `lookahead`) | 3.29186287 | **−1.2675 %** |
| look5 | 3.36833193 | +1.0261 % |
| e99gate (negative control) | 3.33313779 | −0.0295 % |

**KILL RULE 1 fires.** Every assigned arm — `pb6`, `pb6fit` and `lookahead` — is
negative and far below the +0.20 % threshold.

The mechanism is visible per prompt: on republic, essays, medicine and botany
the boundary price clamps depth from about 5–6 down to about 3.9, with a clamp
share of 0.65–0.82, and costs +2.3 % to +5.0 % of candidate time on each. Those
four prompts sit either side of the median, so the loss lands directly on the
published score.

### Why `look5` is not a promotable positive

`look5` is the only positive arm, and it is not promotable.

It wins by clamping beagle from mean depth 4.38 to 3.00 with a clamp share of
**0.78** — that is, clamping down into the ranked G=1 band. That is precisely the
E99 rung-5 action, which was measured **officially** at **−6.077 %**. E99's
regression on official data prices candidate slowdown at −10.89 % per unit fire
rate with r = −0.958, which puts a 0.78 fire rate on beagle at roughly −8.5 %,
not the +1.5 % this surrogate predicts.

The `e99gate` negative control confirms the instrument's limit directly: it
under-fires here (clamp share 0.06–0.18) against E99's official per-prompt fire
rates of 0.295–1.000. The surrogate cannot predict the sign of the one
officially measured action in this class, so its one positive arm carries no
weight.

## Weaknesses in the assignment's stated inputs

1. **The E99 traces do not exist** (see RUNG 1). Substituted with
   `e101ctl512`, which is current-tree and passes a 78/78 positive control.
2. **The E106 census tier ratio is weakly supported.** The per-width census has
   19 rounds in total, with **n=1 at M=2 and n=1 at M=5** — the two points that
   set the entire G=1 step. It is a serialized-command-buffer census with
   `timing_valid=false`. Sibling forced legs put M=5 at 102,650 and 102,864 µs
   against 113,710 µs unforced, an 11 % spread, so the local ratio 1.68 is not
   robust either.
3. **The `S µs/GB` refit values quoted in the assignment could not be
   reconciled** with the E106 census; they imply about 7,924 µs per step inside
   G=1 against the census's 11,834. They were not used. M=1, 3 and 4 were
   interpolated on the census line and are labelled INFERRED in every table.

## Conclusion

The source observation behind E113 is correct and the deliverable table is
produced, but the inference from it is wrong. E100 moved the dispatch group
boundary without moving the ranked cost boundary, which stays at M=5 and fits
our own post-E100 receipts to 0.68 %. Teaching `costModelDepth()` about a
width-6 boundary therefore teaches it a cost that the ranked host does not
charge, and every assigned arm loses 1.3 % to 2.0 % of published score.

Result label: **not useful**. The mechanism is absent on the scored host.

## Suggested follow-ups (not implemented)

1. **Do not reprice the schedule downward again.** Both frames agree that `d*`
   for every median-carrying prompt sits at or near the cap. The schedule axis is
   exhausted in the direction of reducing depth. Any remaining allocation prize
   is in *raising* depth on the prompts that stop short, and it is bounded by
   E99's 2.58 % adaptive-over-fixed advantage, most of which the walk already
   captures.
2. **Price any future schedule work on the ranked curve, not the local one.**
   The two frames disagree about the sign of the width-6 step (ranked 0.756 vs
   local 1.678). Route B in `research/e113_depth_boundary.py` refits the ranked
   curve from official receipts and should be the default pricing route for
   schedule experiments.
3. **Re-measure the local per-width census properly if it is ever load-bearing
   again.** n=1 at M=2 and M=5 is not enough to set a marginal price, and the
   forced-leg spread is 11 %.
4. **Consider the group-scaling coefficient A itself as a target.** Ranked
   A ≈ 2.0 against local A = 1.64 is the whole reason this hypothesis failed.
   Whether the ranked host's two-group QMV path can be pushed above A = 2.0 is a
   kernel question, not a schedule question, and it was not tested here.
