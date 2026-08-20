SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mtp_seconds_per_token","available":true,"value":0.03271992},"test_metric":{"name":"residual_divergence_count","available":true,"value":0}}

# E56 r2 — the stream-aware price replicates at −3.93 %, but E55 already took a third of its gain, and the `repair` arm is dead on arrival

- **Student / branch:** `qwen-edward` / `qwen-edward/stream-aware-draft-depth-schedule`
- **PR:** #59. **Revision:** `e56-r2`.
- **`BASE_SHA`:** `7040406e0383e8f9e7e2a44781f5829a9d2d762a` (post-E55). **Candidate:** `3ce87c6`.
- **Host:** Apple M4 Pro, 48 GB. **This is not the ranked M5.**
- **Sessions:** two complete ABBA sessions, 18 timed legs, **512 decode tokens each**, real cool gate on every leg, `all_tokens_matched` true on all 18.

## The four things worth your time

1. **The stream-aware price replicates exactly across a base change.** `s45` gives **−3.9259 %** pre-E55 and **−3.9279 %** post-E55. Two different bases, two different binaries, agreement to **0.002 pp**.
2. 🔴 **R6 answered: E55 and E56 are substitutes. The interaction is +4.20 pp, one third of the joint gain.** E55 alone −4.73 %, E56 alone −8.33 % on the old base, both together only −8.47 %. The best candidate on the new base is **0.15 % faster than the best candidate on the old one**, even though E55 moved the base 4.73 %.
3. 🔴 **The `repair` arm should not be run. I bounded it at 0.06 %, below my own null floor.** The repair costs **0.482 ms**, not the ~2.5 % of decode time estimated from dispatch counts — an **8x** overestimate.
4. **The `s89` sign flip is explained mechanically and is not noise.** E55 changed exactly one table cell; the marginal it governs halved from 41.205 ms to 19.834 ms; every other marginal stayed flat.

---

## 1. Session 4 — the arm set on the post-E55 base

Ten legs, ABBA-counterbalanced: `base s45 s89 h224 mix mix h224 s89 s45 base`. Job `b48f602b`, 00:18:33Z → 01:58:15Z, exit 0.

| arm | mtp s/tok | vs base | own spread | mean draft length |
|---|---|---|---|---|
| `base` | 0.03405768 | — | **0.2185 %** (null) | 6.513 |
| **`s45`** | **0.03271992** | **−3.9279 %** | 0.0539 % | 5.627 |
| `s89` | 0.03484947 | **+2.3248 %** | 0.9729 % | 6.037 |
| `h224` | 0.03461337 | +1.6316 % | 0.0093 % | 6.462 |
| `s45h224` | 0.03306136 | −2.9254 % | 0.2108 % | **2.985** |

**Serial falsifier does not fire.** Every arm's serial leg sits within **0.104 %** of the base pair mean (0.07476092), so the confinement argument that licenses the local ratio holds.

**On the null floor.** You retracted `0.0629 %` and asked me to use my own same-arm spread at matching separation. In my session, separation does **not** predict spread: adjacent 0.2108 %, three apart 0.0093 %, five apart 0.9729 %, seven apart 0.0539 %, nine apart 0.2185 %. That is non-monotone and it contradicts alphonse's separation model on my data. I therefore use the **largest same-arm spread in the session, 0.9729 %**, as the conservative floor. `s45` clears it by 4.0x and clears the base null by 18x.

**The width-4 cap prediction is confirmed.** `s45h224` collapses mean draft length from 6.513 to **2.985**, i.e. mean verify width about 3.99. Pricing the 4→5 boundary while raising `h` makes the walk stop at width 4 almost every round, exactly as the offline model says.

### The `s45` x `h224` interaction refutes your preregistered direction

You predicted substitution: "naive summation of the two main effects will overstate the combination". Measured:

| quantity | value |
|---|---|
| `s45` alone | −3.9279 % |
| `h224` alone | +1.6316 % |
| additive prediction | −2.2963 % |
| **`s45h224` measured** | **−2.9254 %** |
| **interaction** | **−0.629 pp, super-additive** |

The combination is **better** than the sum, not worse. Both arms push the walk toward the same stopping width, and once it lands there the two changes reinforce rather than compete. I record this as a refutation of the preregistered direction.

**`h224` is a loss and I am not shipping it.** `headStepCostRatio` stays at **0.18**.

---

## 2. 🔴 R6 — the E55 by E56 interaction

You called this the most valuable single number in the revision. Instrument: `research/e56_e55_interaction.py`.

Both sessions share host, fixture, token window and real cool gate, and each carries its own unchanged base pair, so E55's own effect is measured rather than assumed.

**Which arm is "E56 at its best" differs by base, and that is the point.** Pre-E55 the table has two weight-stream boundaries, 4→5 and 8→9, so the full-staircase arm is `sfull` (`priced_boundary_widths=5,9`). E55 deleted the 8→9 boundary, so post-E55 the same policy is `s45`. This compares one mechanism against itself on two tables.

| quantity | absolute s/tok | change |
|---|---|---|
| pre-E55 base | 0.03574724 | — |
| pre-E55 `sfull` | 0.03276828 | −8.3334 % |
| post-E55 base | 0.03405768 | −4.7264 % (E55 alone) |
| post-E55 `s45` | **0.03271992** | −3.9279 % vs its own base |
| | | |
| **both, pre-E55 base to post-E55 `s45`** | | **−8.4687 %** |
| independent prediction | | −12.6659 % |
| **INTERACTION** | | **+4.1972 pp** |

**33.1 % of the joint gain is shared.** The two changes attack the same inefficiency from opposite sides: E55 makes the wide dispatch cheaper in the kernel, E56 routes the schedule around it. Doing both does not pay twice.

The blunt version: **the best candidate available on the new base is only −0.1476 % faster than the best candidate that was available on the old one.** E55 moved the base 4.73 %, and E56 gave back almost exactly the headroom E55 took.

**Decision-relevant consequence.** Any future kernel change that cheapens a boundary width will erode this schedule's gain by roughly the amount it saves. Price them jointly, never additively.

---

## 3. Why `s89` flipped sign — R6's mechanism

E55's diff to the dispatch table is **exactly one cell**: `case 9` inputs-per-group `3 → 5`, so width 9 drops from **3 weight streams to 2**.

Measured clean-round marginals:

| step | pre-E55 | post-E55 | change |
|---|---|---|---|
| m(4→5) — a boundary on both bases | 42.495 ms | 41.821 ms | −1.6 % |
| **m(8→9)** — a boundary **only** pre-E55 | **41.205 ms** | **19.834 ms** | **−51.9 %** |
| m(5→6) | 13.306 | 13.931 | flat |
| m(6→7) | 15.137 | 15.386 | flat |
| m(7→8) | 16.566 | 15.946 | flat |

The targeted step halved. The untargeted boundary did not move. Every non-boundary step stayed flat.

`s89` prices a boundary at width 9. Pre-E55 that boundary was real and the arm won (−3.1180 %). Post-E55 it does not exist and the arm charges a premium for an ordinary step, so it loses (+2.3248 %). **Erosion 5.44 pp.** The arm's model went stale; it did not go wrong.

**Arm provenance pins this to E55 and nothing else.** The base arm's schedule blob is byte-identical across both sessions (`328399d33b15…`) while its `__TEXT,__text` digest differs (`18401bee…` to `5466d761…`). All five session-4 arms have distinct `__text` digests and share one metallib (`9e271c6b…`), rebuilt after the merge as instructed.

**This also means a campaign gate is stale.** The stream-optimality selftest currently fails with `shipped boundaries are [(4, 5)], expected [(4, 5), (8, 9)]` and `HEAD NA ceiling is 5, expected 4`. Those expectations encode the pre-E55 table. The gate reports the same fact I measured in seconds. Its quoted break-even tax at M=9 has also moved from 12.43 % to 14.20 %.

---

## 4. Steps 0b, 0c and 1 — zero GPU, and they close the `repair` arm

Instrument: `research/e56_repair_census.py`. Every timed leg already emits one `mtp-trace:` line per round carrying `round_us` and its components, so all three answers came from traces already on disk.

### Step 1 — repair cost in seconds

| session | base | widths with both cells | repair cost | share of a clean round |
|---|---|---|---|---|
| s4 | `7040406` | 3, 4, 5, 6 | **0.482 ms** | **0.29 %** |
| s3 | `aded0f5` | 4, 5, 6 | **0.472 ms** | **0.26 %** |

Two bases agree to 2 %. Your first-cut estimate was **+2.5 % of decode time**; the measurement is **8x smaller**. The 48 extra recurrent-layer dispatches are real but cheap. **Dispatch count is not time.**

🔴 **Applying your own stop rule.** The repair fires on 19.3 % of pooled rounds. A *perfect* repair-aware walk that avoided every repair saves at most `0.29 % x 19.3 % = 0.06 %`, which is below every candidate null floor in this session. I did not build the `repair` or `stream+repair` arms. Say the word and I will run them, but I would be spending an allocation on a quantity I have already bounded.

### Step 0b — the confound is total above width 6

| M | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| repaired share | 50.0 % | 16.0 % | 73.3 % | 53.0 % | **0 %** | **0 %** | **0 %** |
| n | 8 | 350 | 30 | 132 | 24 | 112 | 238 |

**Zero repaired rounds across 374 rounds at M = 7, 8, 9**, on both bases.

The mixture bites exactly as you warned. Naive per-arm clean-against-repaired medians **invert** — repaired rounds look *faster* (base 202.7 ms clean against 151.7 ms repaired) — purely because repaired rounds are narrow rounds.

🟢 **Re-derived from clean rounds only, the staircase grows: boundary ratio 2.0301 to 2.6596.** It does not shrink. The thesis is stronger than it was.

Caveat I will not bury: **m(4→5) rests on 8 clean rounds at M=5 in s4 and 6 in s3.** The crossing step is the thinnest cell in the table. It is consistent across two bases (42.495 / 41.821 ms), which is the best evidence I have, but it is not a large sample.

### Step 0c — `P(any reject | d)`

| d | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| P(reject) | 0.2500 | 0.1143 | 0.4000 | 0.3788 | 0.0000 | 0.1071 | 0.1429 |
| n | 8 | 350 | 30 | 132 | 24 | 112 | 238 |

**This is not monotone in `d`**, so `1 − prod(g)` is the wrong shape here. The curve is dominated by which schedule reaches which depth, not by depth itself — the same confound in another guise. I would not put this vector into a cost model as written.

---

## 5. Your cost-model functional form needs one more term

Fitted `m(M−1→M) = c + s · Δ(Σ_g 1/bw(NA_g))` on session 4's clean marginals, with `W = 14.412` GB and your `bw(NA)` ladder held fixed.

- fixed cost of one added verify row: **9.573 ms**
- bandwidth slope 8.6031, implying **b = 0.597** at round level against your 0.276 at cell level

**The purely bandwidth-driven form cannot fit the within-boundary steps.** Without the intercept, every within-step is under-predicted by 6–8 ms while the crossing step is over-predicted by 6 ms. That is a systematic pattern, not scatter. One fixed per-row term removes it.

**Out-of-sample test — the fit predicts a different table on a different base:**

| step | measured | predicted | error |
|---|---|---|---|
| 4→5 | 42.495 ms | 42.004 ms | −1.15 % |
| 5→6 | 13.306 ms | 14.211 ms | +6.80 % |
| 6→7 | 15.137 ms | 15.585 ms | +2.96 % |
| 7→8 | 16.566 ms | 15.585 ms | −5.92 % |
| **8→9** | **41.205 ms** | **40.630 ms** | **−1.40 %** |

Worst error **6.80 %**. It predicts the 41.2 ms step that exists **only** on the other base, from a fit that never saw it.

This answers the collinearity concern constructively. `M` and `ceil(M/IPG)` are collinear, but there really are two additive channels — per-row and per-stream — and the dispatch table breaks the collinearity precisely at the crossing widths, which is where the information lives.

---

## 6. 🔴 Preregistered width cap

Run offline against both tables at the clean-round ratio 2.6596, `h` pinned to 0.18.

| table | streams, M=3..9 | boundaries | walk stops at |
|---|---|---|---|
| today | `1 1 2 2 2 2 2` | `4→5` | width **4** |
| after `t55` + `t6` | `1 1 1 1 2 2 2` | `6→7` | width **6** |

**End-state cap is 6 at every acceptance from 0.80 to 0.99**, and 5 at 0.75. That band covers ranked beagle (0.8351), ranked medicine (0.8750) and the local fixture (0.9625). The prediction is acceptance-insensitive, which makes it a clean falsifier.

**`widthCap = fullAcceptStreak >= 2 ? 8 : 5` can express 5 or 8 and never 6.** If this holds, the shipped rule is structurally unable to reach the optimum once either arm lands.

**Scope limit, stated honestly.** This models the **marginal walk only**. The shipped session also has the full-accept-streak escape, which jumps past the walk's stop without pricing the skipped steps. That escape is why R5 saw rounds at width 4 and width 9 and **none between**: the walk stops at 4, the escape lands on the cap, and no schedule ever prices the widths between.

---

## 7. IPG identity digest — now a mandatory per-leg field

| session | base | canonical table | streams | digest |
|---|---|---|---|---|
| s3 | `aded0f5` | `3:3,4:4,5:3,6:3,7:4,8:4,9:3` | `1 1 2 2 2 2 3` | `7f01b6e187a386757dc8b7f87a24e25576c165e57b2a77beee99e54b16a7db94` |
| s4 | `7040406` | `3:3,4:4,5:3,6:3,7:4,8:4,9:5` | `1 1 2 2 2 2 2` | `364ec9f86d07a46ffe4882797701ab2cde2d85264bfd78ded0e2bd7f46a460b8` |

Each session is scored against the table **its own binaries were built from**, read via `git show`, never from the worktree. Scoring s3 with the live table misclassified its 8→9 step and produced a wrong ratio of 1.9716; with the correct table the two independent sessions agree at **2.7894 and 2.6596**.

⚠️ **Hazard for anyone reading that table.** The prose comment above `case 8` in `quantized.h` describes a 3+3+2 split at inputs-per-group 3. The live template argument is **4**. Read the template, not the comment. My parser now raises if the case number and the template width disagree.

---

## 8. Session 3 — the pre-E55 arm set, for completeness

Eight legs, `base s45 s89 sfull sfull s89 s45 base`, 512 tokens, real cool gate, all exact.

| arm | mtp s/tok | vs base |
|---|---|---|
| `base` | 0.03574724 | — (null spread **0.0013 %**) |
| `s45` | 0.03434384 | −3.9259 % |
| `s89` | 0.03463265 | −3.1180 % |
| **`sfull`** | **0.03276828** | **−8.3334 %** |

`s3s45a` failed and was retried: the real cool gate refused to release at `41.0C, min seen 40.9C, waited 200s`. That is an instrument stall, not a defect, and the retry is the leg reported.

Entry temperatures were 39.9 °C for `base1` and about 52 °C for every later leg, a 12 °C spread — yet the base null spread was **0.0013 %**. On this host a 12 °C entry spread moved the primary metric by less than 0.002 %.

---

## 9. Gates, and an honest reading of them

| gate | result |
|---|---|
| `senpai/validate-assignment-scope.sh` | **OK**, 1 submitted path |
| `senpai/check-editable-budget.sh` | **OK**, source 2,465,123 / 3,000,000; growth **6,174 / 262,144** |
| `swift test --force-resolved-versions` | 40 issues in 3 suites — **all pre-existing** |
| `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test` | 40 issues — same set, after isolating one flaky test |
| `QwenMTPDepthCostModelTests` | **PASS**, including the live dispatch-switch re-parse |
| `senpai/run-all-gates.sh` | 10 failures, 3 usage-exits — **all pre-existing** |

**I verified "pre-existing" rather than asserting it, three times.**

1. I ran the full gate suite at my HEAD and at the unmodified base `7040406` in a detached worktree. **Identical results**: the same 10 failures and 3 usage-exits.
2. For the plain test suite I reverted my two changed files to their base state in place, re-ran the three failing suites, and got the same failures. My diff causes none of them.
3. The runtime suite first showed **41** issues at my HEAD against **40** at base. I did not accept that difference as pre-existing. I ran the extra test, `phaseStartAllocatorResetLeavesExactlyEmptyCacheWhenRuntimeTestsAreEnabled`, in isolation: it **passes at both my HEAD and the base**. I then re-ran the full runtime suite twice at my HEAD, unchanged: **attempt 1 failed with 41, attempt 2 passed with 40**. The test is non-deterministic under parallel execution because it asserts that a **process-global** MLX allocator cache is exactly empty while other tests allocate concurrently. My HEAD reaches the same 40-issue baseline as the base.

The 40 failures are campaign-state: doc campaign markers, the config manifest digest, the 128 GiB startup memory profile, the MTP head declaration source, and track naming. None reads the scheduler.

`upstream` was not configured in this student checkout, which made `run-all-gates.sh` refuse to run at all. I configured it with the sanctioned `senpai/bootstrap-checkout.sh`, which sets the organizer push URL to `DISABLED` and leaves `origin` untouched.

---

## 10. Deviations and caveats, declared

- **No declared discarded warm-up leg in session 4.** Your instruction arrived after the session started. Re-running to add it would have changed the base the arms were measured against and destroyed the R6 comparison.
- **Geometry exports are unverifiable on this host** by the mechanisms you listed. I held them identical across both sessions instead. That keeps the R6 contrast unconfounded but does not prove the ranked geometry. `wired_residency_active=false`, as expected below 96 GiB.
- **`Memory.peakMemory` is unreadable on MTP legs**, per your correction. Not reported.
- **M4 Pro, not M5.** Levels do not transfer. I use the ratio between within-tier and boundary steps, which is more robust than levels, and I report absolute candidate seconds per token beside every ratio.
- **m(4→5) rests on 6–8 clean rounds.** Thin, consistent across two bases, but thin.
- **`post_assignment_comment` returned HTTP 403 for several hours.** Read access recovered and the interim comment is posted. No evidence was lost.

## 11. Ranked transfer — the reason I am not calling this submittable

The counterfactual against the two score-setting prompts does **not** agree with the local fixture:

| arm | this host | ranked M5 (g_lo / g_hi) |
|---|---|---|
| `s45` | **+2.6748** | **−1.5631 / −0.9644** |
| `s89` | −0.1816 | **+0.6930 / +0.5699** |
| `h224` | +0.9078 | −3.8683 / −3.2589 |
| `s45h224` | +2.5183 | −3.3133 / −2.4549 |

**The signs invert between local and ranked.** The cause is measured, not speculative: under `s45` the 4→5 step is bought only when sequential acceptance is **at or above 0.9491**. The local fixture sits at **0.9625** and crosses; ranked beagle (0.8351) and medicine (0.8750) do not. **The local fixture sits on the opposite side of the price's own decision boundary from both ranked prompts.**

So `s45` is a genuine, replicated local winner whose mechanism I understand, and which my best model says would **lose** on the ranked pool. I am not going to dress that up. The honest label is **local winner, ranked-negative by counterfactual**.

## 12. Reproduction

```bash
git checkout 3ce87c6
tools/build-mlx-metallib.sh
bash research/e56_build_arms.sh                 # 5 arms, __text digests, arm.txt
bash research/e56_session.sh s4                 # 10 legs, ABBA, 512 tokens
python3 research/e56_analyze.py --session s4
python3 research/e56_repair_census.py --session s4
python3 research/e56_repair_census.py --session s3
python3 research/e56_e55_interaction.py
```

## 13. W&B runs — session 4

| tag | arm | run ID | URL |
|---|---|---|---|
| s4base1 | base | `9jdz48re` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/9jdz48re |
| s4s45a | s45 | `xbgr93bj` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/xbgr93bj |
| s4s89a | s89 | `5wzwasmj` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/5wzwasmj |
| s4h224a | h224 | `6w0u9bls` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/6w0u9bls |
| s4mixa | s45h224 | `1a9dacfm` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/1a9dacfm |
| s4mixb | s45h224 | `l07p4fzw` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/l07p4fzw |
| s4h224b | h224 | `7ugpekz4` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/7ugpekz4 |
| s4s89b | s89 | `j6pweice` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/j6pweice |
| s4s45b | s45 | `fx5tt1bv` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/fx5tt1bv |
| s4base2 | base | `bjnlxbal` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/bjnlxbal |

All ten `finished`, all `all_tokens_matched=true`, all `residual_divergence_count=0`.

Session 3: `l35ee9fo` `3bbx1wip` (failed cool gate, retried) `j4eybpm0` `aw3inbpw` `pb89dpcd` `dzj75564` `s3nngw77` `fq59bwtr`.

## 14. Suggested follow-ups — not implemented

1. **Update the stream-optimality selftest.** Its expectations encode the pre-E55 table and it has been red since E55 landed. It is currently reporting a true fact as a failure, which trains people to ignore it.
2. **Re-run `s45` after `t55` and `t6` land, to settle the width-6 preregistration.** This is the single cheapest decisive test in my area and it is already specified.
3. **Give the walk a cap it can express.** If the width-6 prediction holds, `fullAcceptStreak >= 2 ? 8 : 5` needs to become a value the cost model chooses, not a literal. That is a separate assignment and I have not touched it.
4. **Price kernel arms and schedule arms jointly.** The +4.20 pp E55 interaction says any future boundary-cheapening kernel change will erode this schedule's gain by roughly what it saves. Two arms that each look positive alone may sum to much less.
5. **Find out why local and ranked acceptance straddle 0.9491.** The local fixture is on the wrong side of the decision boundary for every price I can build. A fixture that brackets both ranked prompts would make this whole line of work measurable locally.
