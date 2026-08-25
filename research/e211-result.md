# E211 result — optimal step-aware marginal depth price

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"loo_honest_published_median_delta_pct_worst_law","available":true,"value":2.312},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}
```

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e211-step-aware-depth-price`
- Hypothesis and target cost: the shipped uniform marginal depth price
  (`makeUniformDepthPrice`, h = 0.18 on every row) is not the optimal price
  over the greedy `costModelDepth` family. The target cost is the round work
  the scheduler buys at each row of the current tree, including the disputed
  9-row cell.
- Decision: **green on the desk instrument**. The assigned stop rule fires
  IMPLEMENT under every cost law. This is a desk experiment: it predicts a
  published-median move, it does not measure one.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit:
  `d9460346662bc8229696550550e9211aa40fdb6a` / unchanged by this experiment /
  no candidate commit — the diff is research-only.
- Yukon promoted submission / source ref used as frontier: receipt A
  `3.70784519` (cap 7) and crown `3.72911001`, read through the pinned
  constants in `research/e200_desk_price.py`. This experiment did not query
  Yukon and did not submit. Advisor Entry 424 confirms that E208 r1 is merged
  and that the frozen candidate behind the in-flight receipt is cap-8 + (9,5)
  (`9a91ba76`, FINDING 545, W&B run `hrzezfsk`). That candidate is not a
  comparison point for this desk experiment; it fixes which cost law prices a
  forward implementation.
- Candidate build fingerprint: none. No Swift build, no worker, no GPU.
- Submitted-surface / generated-twin / metallib digests: unchanged. The
  editable-budget gate reports `growth=0/262144` bytes against the assignment
  base, which is the direct evidence that no submitted byte moved.
- Submitted candidate files: **none**.
- Supporting test, tooling, or documentation files:
  - `research/e211_step_price.py` — the experiment.
  - `research/e211_wandb.py` — the W&B publisher.
  - `research/e211-artifacts/step-price.json`, `.txt`, `wandb.json`.
- MTP head provenance, digest, and draft policy: unchanged, organizer-pinned.
  This experiment changes the depth *price*, never the head.
- Token window, fixture, reference source, and harness: no token window and no
  fixture. Every number is `harness=ranked` and comes from the FINDING 520
  survival-pinned latent-q instrument, imported from `research/e201_online_cap.py`
  without modification.
- Exact cell: the greedy `costModelDepth` marginal price table over rows
  `d = 0..7`, evaluated against three readings of the 9-row cell.
- Official causal path and score equation: the price table changes only the
  candidate MTP leg's chosen draft depth, so it moves
  `candidate_mtp_seconds_per_token_mean` and leaves the pinned serial numerator
  untouched. `senpai/verify-ranked-score-boundary.sh` PASSes. No local
  serial-to-MTP ratio and no `psi_serial` term appears anywhere in this report.
- Assignment-scope preflight: the diff against
  `d9460346662bc8229696550550e9211aa40fdb6a` touches four files, all under
  `research/`. `Sources/`, `Vendor/`, `mtp-head.manifest.json`, `mtp-head/`,
  and `Package.swift` are untouched.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `source=2657095/3000000 headroom=342905 growth=0/262144 exempt=2410`.
- Scored-path reachability evidence: not applicable — nothing was added to the
  scored path. The *family* studied is the one the scored worker already runs;
  the recommended table is a constant vector inside it.
- Written promotion rule and verdict: the assignment's rule is LOO-honest
  published-median delta. < 0.5 % under every law closes the family; >= 1 %
  under the law the receipt lands on triggers an implementation assignment.
  **Verdict: >= 1 % under all three laws. The family does not close.**
- Pre-official evidence budget / timed legs used: **zero timed legs, zero GPU
  seconds**. This was a desk-only assignment.
- Frozen candidate SHA: none.
- Submission owner / receipt-watcher job ID: none.

## Evidence

- Host, instance, chip, memory profile, toolchain, thermal policy:
  `desk-replay`. No model-holding process ran, so no thermal gate applies and
  no timing claim is made. The W&B config records
  `coolGatePassedRealGate=false`, `gateQualifiedForTiming=false`,
  `timingClaimsPermitted=false`, `officialOrRankedScore=false`.
- `head_provenance_sha256` for every leg: not applicable. No leg ran.
- Exact reproduction commands:

  ```bash
  git checkout 9bdbe9b7          # or any later commit on this branch
  python3 research/e211_step_price.py          # ~32 s, seed 211
  python3 research/e211_wandb.py               # publishes the four runs
  ```

- Cheapest real falsification gate and positive control, in execution order:

  1. **Reproduction gate.** The prefix-sum grid form of the published median
     is compared with the merged `O.cap_grid` evaluator on every cap.
     Max relative error `2.318e-14` → PASS. The medians reproduce E203
     exactly: smooth caps 4/5/6/7/8 = `3.535224 / 3.550300 / 3.594083 /
     3.707845 / 3.829386`; step cap 8 = `3.681378`.
  2. **FINDING 520 instrument validation.** Cap 7 in-sample `-9.1e-13 %`
     against paid receipt A; cap 4 out-of-sample `-0.438 %`; cap 5
     out-of-sample `+0.243 %`; receipt-channel 1σ `0.689 %` → PASS. The
     out-of-sample caps are the positive control: they are free to disagree
     with the paid receipts and do, by a bounded amount.
  3. **Greedy-table verification.** Every reported price table is fed back
     through the unmodified `e200_desk_price.greedy_depth` walk on the full
     grid. Agreement `1.0000`, `0` mismatching grid points, for every table in
     this report. This is what makes the search exact rather than a sweep: any
     monotone threshold vector `Q_1 <= ... <= Q_8` is realised by
     `marginal[d] = Q_{d+1}^{d+1} * cumulative[d] / (1 + sum_{k=1..d} Q_{d+1}^k)`,
     so the free optimum over the price family is an exact eight-cut search.
- Exact-token and row-ledger verdict: not applicable. No tokens were generated.
- Generated-twin audit: not relevant; no Metal source changed.
- Peak RAM / head size: not relevant.
- Official status and score: not submitted.

### The three live readings of the 9-row cell

Transfer ratio (FINDING 505 over E186) `0.23731`. Local dR9 `39.827 ms`; the
E208 (9,5) correction removes `20.774 ms` locally, leaving `19.053 ms`
residual. The transferred value is marked INFERRED.

| law | ranked dR9 | R(9) ms | shipped cap-8 published median |
| --- | ---: | ---: | ---: |
| smooth | +1.962 ms | 60.76 | 3.829386 |
| step | +9.451 ms | 68.25 | 3.681378 |
| step_e208 | +4.521 ms | 63.32 | 3.777460 |

`R(1..8) ms = 30.26 31.67 33.28 35.68 40.44 48.36 55.23 58.80` under all three.

### Headline: the free optimum over the price family (`harness=ranked`)

| law | shipped | optimum | in-sample | **LOO-honest** | oracle_q ceiling | captured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth | 3.829386 | 3.962218 | +3.469 % | **+3.377 %** | 3.963760 | 98.9 % |
| step | 3.681378 | 3.780722 | +2.699 % | **+2.642 %** | 3.781543 | 99.2 % |
| step_e208 | 3.777460 | 3.866967 | +2.369 % | **+2.312 %** | 3.869460 | 97.3 % |

LOO-honest scores each held-out prompt with a table refitted on the other
seven, so it carries no in-sample fit. The gap to in-sample is under 0.1 pp
under every law: the eight-cut family is far too rigid to overfit eight
prompts. The free optimum also captures 97–99 % of the round-level `oracle_q`
ceiling, so almost nothing is left inside this family.

The worst single LOO-drop fold is `republic` at `+2.440 %` (step) and
`+2.276 %` (step_e208). No fold is negative under any law.

### Level versus structure — this corrects FINDING 519

The price has a *level* (how deep the scheduler drafts overall) and a *shape*
(how rows are priced against each other). Only the shape carries step
awareness. Against an **honestly refitted uniform-level control**:

| law | LOO-honest level-only | LOO-honest structure premium | best SHAPE at the *shipped* level |
| --- | ---: | ---: | ---: |
| smooth | +1.387 % | **+1.963 %** | +2.677 % |
| step | +0.096 % | **+2.544 %** | +2.700 % |
| step_e208 | +0.660 % | **+1.641 %** | +1.707 % |

The gain is therefore not "draft less". Under the step law the best uniform
level is worth only `+0.096 %` LOO-honest while the shape is worth `+2.544 %`.

FINDING 519 recorded that the price-shape axis was dead, from E200's `+0.17 %`.
That negative is an artefact of E200's restricted family: E200 blended the
uniform price toward the measured cost curve along a single axis. With the
shape left free at exactly E200's shipped price level, the same axis pays
`+2.677 % / +2.700 % / +1.707 %` in-sample. **FINDING 519 should be narrowed
to the uniform↔measured-cost blend; it does not hold for the price-shape
family.**

### Cross-law transfer risk

A table fitted under one reading of the 9-row cell, paid under another
(in-sample published-median delta):

| paid law \ fitted | smooth | step | step_e208 |
| --- | ---: | ---: | ---: |
| smooth | +3.469 % | **−1.271 %** | +3.204 % |
| step | −0.529 % | +2.699 % | +0.303 % |
| step_e208 | +2.115 % | +0.086 % | +2.369 % |

Worst transfer `−1.271 %`. The `aff4ad64` receipt names the true 9-row cell
only after a table must be chosen, so a per-law fitted table is a bet that can
lose. This is the reason the recommendation below is a minimax table, not the
best table under any single law.

### Recommended branch reading for both receipt outcomes

Advisor Entry 424 fixes which law prices which decision. E208 r1 is merged, so
the maintained base carries the (9,5) correction and law (c) `step_e208` is the
default **forward-pricing** law. The `aff4ad64` receipt itself reads on laws
(a)/(b), because its candidate `c47c7284` predates that merge and pays the
uncorrected m=9 cost. The receipt therefore cannot read on `step_e208` at all.

| receipt outcome | what it settles | forward law for an implementation | free optimum, LOO-honest |
| --- | --- | --- | ---: |
| lands SMOOTH | no step at m=9, so the (9,5) correction removes a cost that was not there | smooth | +3.377 % |
| lands STEP | the E186 step is real, and (9,5) is already merged | step_e208 | +2.312 % |

Both branches say IMPLEMENT. The branch only changes the size of the prize,
from `+2.312 %` to `+3.377 %` LOO-honest.

### Recommended table: no-regression guarded FORWARD minimax

Because no future implementation is ever priced under the uncorrected step
law, the minimax table should be chosen over the **forward** set
{`smooth`, `step_e208`} rather than over all three readings. This matters:

| arm | law set | worst-reading in-sample | worst-reading LOO-honest | worst single prompt | envelope |
| --- | --- | ---: | ---: | ---: | --- |
| unguarded minimax | all three | +1.678 % | +1.249 % | **−5.151 %** | 1.04× bar, OUTSIDE |
| guarded minimax | all three | +1.638 % | +1.321 % | +0.015 % | **0.85× bar, inside** |
| unguarded forward | forward | +2.376 % | +2.372 % | **−3.116 %** | 1.06× bar, OUTSIDE |
| **guarded forward** | forward | **+2.242 %** | **+2.252 %** | **+0.000 %** | 1.06× bar, OUTSIDE |

Dropping the uncorrected step law is worth **+0.604 pp** of worst-reading
delta (`+2.242 %` versus `+1.638 %`). That law was the whole source of the
transfer risk: every negative or near-zero cell in the cross-law matrix above
involves it, while the two forward laws transfer benignly in both directions
(`+2.115 %` and `+3.204 %`).

**Recommended table** — guarded forward minimax:

```text
d                0       1       2       3       4       5       6       7
marginal     0.120   0.023   0.092   0.094   0.146   0.121   0.100   0.083
Q_(d+1)     0.1200  0.1547  0.5256  0.6508  0.8285  0.8285  0.8285  0.8285
```

Greedy-table agreement `1.0000`, `0` mismatches. It pays `+3.196 %` under
smooth and `+2.242 %` under step_e208 in-sample (`+3.200 %` / `+2.252 %`
LOO-honest), regresses no prompt, and needs no receipt branch. If the dropped
uncorrected step law were somehow paid it would give `−0.146 %` — that
exposure is reported, not hidden, and it is the price of the +0.604 pp.

The guard is not optional. The unguarded forward table gains 0.135 pp more but
loses `−3.116 %` on one prompt, and the unguarded three-law table loses
`−5.151 %` on `plutarch`. `plutarch` does not set the proxy median, but the
hidden pool is not the proxy pool and it may set the hidden one. The guard
binds exactly, at `+0.000 %`.

**Conservative fallback.** The guarded three-law table is the only arm inside
the validated extrapolation envelope (0.85× the bar). If the advisor prefers
to stay inside what a paid receipt has covered, that table pays `+1.716 %`
LOO-honest under `step_e208` instead of `+2.252 %`:

```text
d                0       1       2       3       4       5       6       7
marginal     0.180   0.032   0.032   0.105   0.176   0.153   0.134   0.145
Q_(d+1)     0.1800  0.1800  0.3367  0.6728  0.8723  0.8723  0.8723  0.9105
```

Its row 0 keeps the shipped `0.180` exactly, so the first-draft decision is
unchanged and only rows 1–7 are re-priced.

### Validated extrapolation envelope

The instrument was checked against a paid receipt at caps 4 and 5 only, so the
round mass those schedules relocate away from the paid cap-7 anchor is the
largest relocation the instrument is known to survive.

| schedule | round mass relocated vs cap 7 | |
| --- | ---: | --- |
| cap 4 | 0.890 | validated |
| cap 5 | 0.757 | validated |
| cap 8 | 0.547 | shipped |
| validated bar | 0.890 | |
| free step_e208 optimum | 0.940 | 1.06× — OUTSIDE |
| unguarded minimax, all three | 0.926 | 1.04× — OUTSIDE |
| **guarded minimax, all three** | **0.752** | **0.85× — inside** |
| forward minimax, free | 0.940 | 1.06× — OUTSIDE |
| forward minimax, guarded | 0.940 | 1.06× — OUTSIDE |

This is the one place where the Entry 424 restriction costs something. Freeing
the table from the uncorrected step law lets it push depth further, and the
resulting schedule extrapolates 1.06× past anything a paid receipt has
covered. The guarded three-law table is the only arm that stays inside.

The excursion is small relative to the effect. The instrument's own
out-of-sample error at the two validated caps is `−0.44 %` and `+0.24 %`
against a receipt-channel 1σ of `0.689 %`, while the guarded forward table
predicts `+2.242 %` — roughly 3σ. A 1.06× envelope excursion is therefore a
modest extrapolation for a large predicted effect, but it is an extrapolation,
and only a paid measurement closes it.

### Metrics

There is no timed leg in this experiment, so the template's timing table has
no measured row. The reported quantities are desk predictions of the ranked
published median.

| Metric | Baseline (shipped h=0.18, cap 8) | Candidate (guarded forward minimax) | Delta |
| --- | ---: | ---: | ---: |
| published median, smooth | 3.829386 | 3.951756 | +3.196 % |
| published median, step_e208 (forward law) | 3.777460 | 3.862135 | +2.242 % |
| worst forward-reading LOO-honest delta | — | — | **+2.252 %** |
| published median, step (receipt-only law) | 3.681378 | 3.676004 | −0.146 % |
| worst single prompt, any forward law | — | — | +0.000 % |
| serial seconds/token | not measured | not measured | not measured |
| MTP seconds/token | not measured | not measured | not measured |

Every identity field matched across compared arms because every arm is
evaluated by one instrument, on one prompt table, in one process, from one
seed. The only dimension that varies between arms is the price table. Values
transferred from local measurement to the ranked reading — the 9-row dR9
column — are marked INFERRED above.

## W&B runs

Group `qwen38-r1-e211-step-aware-depth-price`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`:

| run | ID | URL |
| --- | --- | --- |
| smooth law | `qeu6gi0o` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/qeu6gi0o |
| step law | `6qc91jfy` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/6qc91jfy |
| step_e208 law | `v97tauzu` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/v97tauzu |
| receipt-proof / guarded | `re2brh5y` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/re2brh5y |

## Conclusion

- **What happened.** The shipped uniform depth price is not optimal in its own
  family. Under every live reading of the 9-row cell, a free eight-cut price
  table gains `+2.3 %` to `+3.4 %` of published median LOO-honest. A single
  table that needs no receipt branch and regresses no prompt gains `+2.252 %`
  LOO-honest under its weakest **forward** reading, or `+1.321 %` if it must
  also survive the uncorrected step law that only the `aff4ad64` receipt reads
  on.
- **Evidence for the mechanism.** The gain survives an honestly refitted
  uniform-level control by `+1.6 %` to `+2.5 %` LOO-honest, so it is the price
  *shape*, not the price level. Under the step law the level control is worth
  almost nothing (`+0.096 %`) and the shape carries the whole effect. The
  recovered tables reproduce their own depth maps through the unmodified
  greedy walk with zero mismatches, so this is an exact optimum over the
  family and not a fitted approximation of one.
- **Evidence against, and the two named risks.**
  1. **Anchor distance.** The free optima and the forward minimax tables
     relocate up to 94 % of round mass off the paid schedule and sit 1.06×
     outside the paid cap-4/cap-5 extrapolation envelope. The instrument's own
     out-of-sample error at caps 4 and 5 is `−0.44 %` and `+0.24 %`, against a
     receipt-channel 1σ of `0.689 %`. The predicted `+2.242 %` is about 3σ, so
     the extrapolation is modest relative to the effect — but it is real, and
     the guarded three-law table is the only in-envelope alternative.
  2. **Prompt risk.** Every unguarded table buys its median with a single-prompt
     loss: `−5.151 %` on `plutarch` over all three laws, `−3.116 %` over the
     forward set. Those prompts do not set the proxy median, but the hidden
     pool is not the proxy pool and they may set the hidden one. The guard
     removes this for 0.04–0.135 pp and should not be dropped.
- **Transfer risk.** Local-to-M5 transfer is not in question here because
  nothing was timed. The live risk is reading-to-reading: a per-law fitted
  table can lose `−1.271 %` under a different reading of the same cell.
  Entry 424 shrinks this risk sharply, because the uncorrected step law that
  produces every bad transfer cell is no longer a forward-pricing surface. The
  two forward laws transfer benignly in both directions, and the guarded
  forward table is immune by construction.
- **Correction to the record.** FINDING 519 ("the price-shape axis is dead")
  should be narrowed to E200's uniform↔measured-cost blend. The shape axis
  itself pays `+1.7 %` to `+2.7 %` in-sample at the shipped price level.
- **Smallest useful next action.** Implement the guarded **forward** minimax
  table as a constant vector replacing `makeUniformDepthPrice`, then run one
  thermally gated 512-token `--local-submit` pair against a fresh same-host
  base. The change is eight constants; the whole risk is whether the desk
  instrument's prediction survives contact with a paid measurement. That is
  exactly the question a receipt answers and a desk cannot.
- **Recommendation: implement in a separate PR.** The assigned stop rule fires
  IMPLEMENT under every law (weakest `+2.312 %` LOO-honest). Per the
  assignment, the implementation is a separate assignment, so this PR stays
  research-only.

## Suggested follow-ups, not implemented

1. **Per-prompt or online price adaptation.** The free optimum's depth map
   skips depths 5–7 entirely under the smooth law — it stops at 4 or pushes to
   8. A two-regime price selected online from the observed acceptance signal
   may capture more than any single constant table, but it adds state and must
   repay it.
2. **Re-run E200's blend with the shape free.** E200's negative result is now
   known to be family-restricted. Re-deriving FINDING 519's boundary would tell
   the campaign which other "dead axis" findings were family artefacts.
3. **Pay a cap-6 receipt.** The validated extrapolation envelope is currently
   set by caps 4 and 5. One more paid point between cap 5 and cap 7 would widen
   the envelope enough to cover the free optima, at the price of one official
   slot.
