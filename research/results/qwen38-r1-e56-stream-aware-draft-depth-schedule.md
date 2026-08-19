SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mtp_seconds_per_token_delta_pct","available":true,"value":-2.35288},"test_metric":{"name":"residual_divergence_count","available":true,"value":0}}

# E56 — a round-level stream-aware draft-depth price: −2.353 % candidate s/token against a 0.003 % null, exact on every leg

- **Student / branch:** `qwen-edward` / `qwen-edward/stream-aware-draft-depth-schedule`
- **PR:** #59. **Revision:** r1.
- **`BASE_SHA`:** `a2c3dbc497fd76b3e4f99c529a3eb5e8b2090abf`. **Candidate:** `df2e171dff01e2acbe52a90854cae580fddce7e5`.
- **Host:** Apple M4 Pro, 48 GB. **This is not the ranked M5.**
- **Hypothesis:** the draft-depth walk prices every extra verify row at one
  scalar `h`, but the machine does not charge that. Crossing a weight-stream
  boundary buys another pass over the 4-bit backbone. A price that steps at the
  boundaries, with the same average price per row, should stop drafting where
  the next row is expensive and keep drafting where it is cheap.
- **Decision: local winner on this host and fixture, with an explicit transfer
  caveat that the advisor must weigh before any official submission.** The
  primary hard falsifier moves −2.353 % against a null-arm floor of 0.003 %.
  Every leg is exact and thermally gated. The ranked counterfactual on the two
  score-setting prompts is NOT uniformly positive, and the reason is measured,
  not speculative (see *Why this may not transfer*).

## Headline numbers, E56 session 2

ABBA order `base / sched / sched / base`, 256 decode tokens, one public
fixture, real thermal gate on all four legs.

| statistic | base | sched | change | null-arm floor |
|---|---|---|---|---|
| **candidate s/token (PRIMARY falsifier)** | 0.04080603 | 0.03984592 | **−2.3529 %** | **0.0028 %** |
| local ratio | 2.02479039 | 2.07180728 | +2.3221 % | 0.5533 % |
| serial leg (must not move) | 0.08262366 | 0.08255294 | −0.0856 % | 0.5506 % |
| mean draft length | 6.848 | 5.921 | −13.542 % | 0 % |
| accepted draft rate | 0.99115 | 0.96889 | −2.246 % | 0 % |

The effect is 840× the null-arm floor on the primary metric and 14× the
sched-replicate spread (0.1679 %).

**The serial falsifier does not fire.** Session 1 moved the serial leg
+0.1618 % against a 0.0538 % floor, which is why I stopped and audited it.
Session 2 moves it −0.0856 % against a 0.5506 % floor, so the serial leg is
inside its own instrument noise.

## Engagement gate: the histogram moves, and it moves at both boundaries

| leg | rounds | mean W | W2 | W4 | W5 | W6 | W7 | W8 | W9 |
|---|---|---|---|---|---|---|---|---|---|
| baseA | 33 | 7.848 | 0.030 | 0 | 0.030 | 0.182 | 0.091 | 0.091 | **0.576** |
| schedB | 38 | 6.921 | 0 | **0.211** | 0 | 0.105 | 0.026 | **0.658** | 0 |
| schedB2 | 38 | 6.921 | 0 | **0.211** | 0 | 0.105 | 0.026 | **0.658** | 0 |
| baseA2 | 33 | 7.848 | 0.030 | 0 | 0.030 | 0.182 | 0.091 | 0.091 | **0.576** |

In counts: base runs 19 rounds at width 9 and 7 rounds in the width 5–6 band;
sched runs 0 at width 9, 25 at width 8, and 8 at width 4. Both replicates of
each arm are identical, so the schedule is deterministic and the histogram is
not a sampling artefact.

**Both declared price steps are active.** The 8 → 9 step is closed at every
acceptance rate by design, which removes the width-9 band. The 4 → 5 step is
open and crosses at a per-draft acceptance of 0.9222, which is what moves 8 of
38 rounds down to width 4 instead of continuing into the width 5–6 band. The
advisor's engagement gate — the histogram must move at 4 → 5 — is met.

## The time accounting closes

| quantity | base | sched |
|---|---|---|
| leg seconds for 256 tokens | 10.4463 | 10.2006 |
| rounds | 33 | 38 |
| seconds per round | 0.31656 | 0.26844 |

Priced with the round-level marginals the width sweep measured (within-tier
29.66 ms and 26.21 ms, boundary 56.71 ms):

- 19 rounds move from width 9 to width 8: **−1.077 s**
- 6 rounds move from width 6 to width 4: **−0.498 s**
- 1 round moves from width 5 to width 4: **−0.057 s**
- 5 extra rounds at 0.26844 s: **+1.342 s**
- predicted net **−0.290 s**, measured net **−0.246 s**

The mechanism buys narrower verifies by paying for more rounds, and the two
terms nearly cancel. That is the honest size of this effect: a 2.35 % win is
the small residual of two ~1.3 s terms, so it is sensitive to anything that
changes either the row price or the acceptance structure.

## What was wrong in session 1, and what changed

Session 1 measured −1.3405 % but **did not test the assigned hypothesis**. The
price charged thorfinn's E46 QMV refit ratio `27.532 / 9.624 = 2.861`, which is
a per-OPERATION ratio, inside a whole-ROUND walk. A round also runs attention,
the head step and sampling, and none of those widen with the verify row count.

The walk extends while `reach > marginal[d] * (1 + expected) / cumulative[d]`.
`reach` is a product of probabilities so `reach <= 1`, and `expected >=
(d + 1) * reach` at depth `d`. A step is therefore unreachable at EVERY
acceptance rate when `marginal[d] * (d + 1) / cumulative[d] >= 1`. The old
table gave 1.0415 at depth 3 and 1.3636 at depth 7, so that arm was an
unconditional width-4 cap wearing a walk's clothes. The measured session-1
histogram agrees: 98.5 % of sched rounds at width 4.

The repair prices the boundary at round level from the E56 width sweep (eight
exact, thermally gated legs at pinned widths 3, 4, 5 and 6):

| W | mean s/token | pair spread | rounds | round seconds |
|---|---|---|---|---|
| 3 | 0.04337761 | 0.2237 % | 86 | 0.12912 |
| 4 | 0.04031486 | 0.1206 % | 65 | 0.15878 |
| 5 | 0.04377087 | 0.1614 % | 52 | 0.21549 |
| 6 | 0.04154163 | 0.7324 % | 44 | 0.24170 |

Marginals 29.66 / 56.71 / 26.21 ms give a round-level boundary ratio of
**2.0301**, not 2.861. `marginalCostRatio` now builds from
`measuredRoundSeconds` and stays mean-pinned to `headStepCostRatio = 0.18`, so
`h` is untouched and only the SHAPE of the price moves. Within-tier marginal
0.143139, boundary marginal 0.290583.

`QwenMTPDepthCostModelTests.onlyTheDeclaredDepthStepsAreClosedAtEveryAcceptanceRate`
computes the closure condition from live source and asserts the closed set is
exactly `[7]`. Negative control: restoring `27.532 / 9.624` makes it fail.

## Why this may not transfer to the ranked M5

Two independent measurements point the same way, and I want them on the record
because they argue against reading −2.353 % as a ranked gain.

**1. This host charges far more per verify row than the calibrated `h`.**
Extrapolating the measured within-tier marginal back to a depth-0 round gives
0.0698 s, so a within-tier row costs **0.425** of a depth-0 round here; against
E1's directly measured 65.009 ms depth-0 round it costs **0.456**. The shipped
`h = 0.18` is bracketed on both sides by ranked receipts (0.14, 0.15 and 0.32
all measured worse at rank). This host is therefore 2.4–2.5× stingier about
width than the price the ranked receipts fit. A mechanism whose whole job is to
cut depth will look better here than at rank.

**2. The ranked counterfactual is not uniformly positive.** Replaying the
repaired price against the `ca9251b8` receipt structure for the two
score-setting prompts, under the corrected transfer slope
`g ∈ [0.7388, 0.7778]`:

| ladder | beagle | medicine |
|---|---|---|
| g = 1 (no transfer correction) | −3.679 | −2.509 |
| g = 0.7388 | −0.148 | +2.135 |
| g = 0.7778 | −0.740 | +1.353 |

Values are predicted percentage changes in the per-prompt score contribution.
Beagle is flat to slightly negative and medicine is clearly positive, so the
median of the two score-setting prompts lands near +0.3 % to +1.0 % against a
ranked MDE of +0.283 % (2 sd; worst prompt +0.527 %). That is a real but
marginal predicted gain with a real risk of a beagle regression, and it is the
opposite ordering from the local measurement, where the win is large.

I did not retune `h`, `segmentedStreakGate`, `sdpaWidthWallDepthCap` or
`segmentedVerifyDepthCap`, per the advisor's binding constraints. The h-sweep
under the transferred ladder is analysis only: it prefers 0.14 (beagle) and
0.12 (medicine) at `g = 0.7388`, which supports the advisor's prediction that
the ranked optimum `h` sits below 0.18 while local sweeps prefer 0.18.

## Correctness, provenance and thermal record

Identical across all four legs:

| field | value |
|---|---|
| `all_tokens_matched` | `true` |
| `residual_divergence_count` | `0` |
| `public_drift_tripwire_passed` | `true` |
| `mtp_depth` | `8` |
| `decode_tokens` | `256` |
| `head_provenance_sha256` | `c5791f65bf026de7be0277c34b79156c09879955a1acad7776dddae7c2dd9c2d` |
| `uses_pinned_mtp_head` | `true` |

All four legs report `cool_gate_passes=3`, `cool_gate_skips=0`,
`cool_gate_passed_real_gate=true`. Entry / exit GPU temperature: baseA
40.23 / 56.05, schedB 53.08 / 58.75, schedB2 54.67 / 58.59, baseA2
54.82 / 55.64 C. The entry-temperature spread is 14.6 C, but the two base legs
entered 14.6 C apart and returned candidate s/token within 0.0028 % of each
other, so this metric is thermally insensitive over that range on this host and
the spread does not explain the effect.

Arms are prebuilt binaries selected per leg, so the checkout stays on HEAD for
the whole session:

| arm | worker sha256 | metallib sha256 | schedule blob |
|---|---|---|---|
| base | `d8da8db42d9f…` | `2c7549908908…` | `328399d33b15…` |
| sched | `b273316d8ea7…` | `2c7549908908…` | `99294f2946a2…` |

The metallib is byte-identical across arms, so the treatment is confined to the
schedule file. Every leg recorded `dirty=0`.

## Build, tests and submission surface

- `swift build -c release --force-resolved-versions`: OK.
- `swift test --force-resolved-versions --filter QwenMTPDepthCostModelTests`:
  4 of 4 pass.
- `swift test --force-resolved-versions`: 689 tests, **40 issues in 9 tests
  across 3 suites**. All 40 are **pre-existing on the campaign base**. I
  verified this directly: with `Qwen36MTPBlockSession.swift` restored to
  `BASE_SHA` and this branch's new test file moved aside, the same 9 tests fail
  with the same 40 issues. They cover campaign-marked participant docs, the
  seeded calibration provenance, the track release marker, the pinned-head
  manifest declaration, the 4-bit config manifest digest, the even-median rule
  and the ranked 128 GiB startup memory profile. None of them reads the depth
  cost model.
- `senpai/validate-assignment-scope.sh BASE_SHA Sources/MLXFastModel/Qwen36MTPBlockSession.swift`:
  scope OK, 1 submitted path.
- `senpai/check-editable-budget.sh BASE_SHA`: OK.
  `source=2463259/3000000`, `growth=4310/262144`, `files=154`.
- `git diff BASE_SHA HEAD -- Sources Vendor benchmark.json mlx-generated
  mtp-head.manifest.json`: exactly one file, 81 insertions and 2 deletions, all
  in `Qwen36MTPBlockSession.swift`. The candidate is self-contained: no Metal
  source, no generated twin, no head declaration.

## Reproduction

```bash
git checkout df2e171dff01e2acbe52a90854cae580fddce7e5
swift test --force-resolved-versions --filter QwenMTPDepthCostModelTests
python3 research/e56_walk_probe.py            # payability of every depth step
research/e56_build_arms.sh                    # publishes ~/e56-arms/{base,sched}
E56_SESSION=s2 research/e56_session.sh --tokens 256
python3 research/e56_analyze.py               # writes research/e56-abba.json
python3 research/e56_log_session.py --session s2
```

`research/e56_build_arms.sh` must run in the terminal, not as a background job:
it checks the schedule file out at `BASE_SHA` while it builds the base arm, so
it leaves the work tree dirty for the duration and cannot straddle a turn
boundary.

## W&B runs

Group `qwen38-r1-e56-stream-aware-draft-depth-schedule`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`.

| run | id | url |
|---|---|---|
| session summary | `je9xm9dh` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/je9xm9dh> |
| baseA | `m0wn2vzn` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/m0wn2vzn> |
| schedB | `2y3xoua2` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2y3xoua2> |
| schedB2 | `vfd37gey` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/vfd37gey> |
| baseA2 | `3kngu86q` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/3kngu86q> |

The four session-2 leg runs were logged after the session, because `run_job`
did not carry `WANDB_API_KEY` into the job environment and every in-session
`wandb.init` failed with `No API key configured`. The leg artifacts are written
to disk before logging, so no evidence was lost. A future session must pass
`secret_env: ["WANDB_API_KEY"]` to `run_job`.

Cost: two GPU sessions on this host, about 35 minutes each, plus one 8-leg
width sweep of about 80 minutes. Peak memory was not separately instrumented;
the wrapper's single-resident-process rule held throughout.

## Suggested follow-ups (NOT implemented here)

1. **Decompose the win.** A third arm that closes only the 8 → 9 step would
   separate "never run verify width 9" from the stream-aware shape. The
   accounting above suggests the 8 → 9 closure supplies about two thirds of the
   gross narrowing saving, and the counterfactual predicts that an 8 → 9-only
   rule is NEGATIVE at rank on both score-setting prompts (−1.105 and −2.157 at
   `g = 0.7388`). If that decomposition holds, the locally largest component is
   the ranked-worst component, and the shipped mechanism should keep only the
   4 → 5 step.
2. **Retune `h` under the measured shape.** `h` is currently pinned so the
   mechanism is measurable in isolation. The measured within-tier price on this
   host is 0.425–0.456 of a depth-0 round, not 0.18. A ring-fenced `h` sweep on
   top of the repaired shape is a separate, cleanly attributable experiment.
3. **Measure on a harder fixture.** This fixture accepts at p ≈ 0.99, well
   above the 0.9222 crossing point, so the 4 → 5 step fires on only 21 % of
   rounds. The ranked score is set by beagle and medicine, whose per-draft
   acceptance is 0.835 and 0.875. A local fixture nearer that acceptance would
   exercise the mechanism where it is supposed to pay.
4. **Fail the leg loudly when W&B logging fails.** `e56_run_leg.sh` calls the
   logger with `|| true`, so four silent auth failures cost a replay. A
   non-fatal warning banner at session end would be enough.
