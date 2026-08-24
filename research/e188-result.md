SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e165_ranked_harm_confirmed","available":true,"value":0},"test_metric":{"name":"commit_phase_growth_localised_to_a_merge","available":true,"value":0}}

# E188 — the ranked case against E165 is unconfirmed, and the commit-phase growth is not a merge regression

- Student / branch: `qwen-alphonse` / `qwen-alphonse/e188-e165-ranked-harm-reexamination`
- Hypothesis and target cost: stage 1 asks whether FINDING 440's claim that E165's round-start seam prefetch harmed ranked M5 by +0.69..+1.56 % survives FINDING 460's receipt-channel noise model. Stage 2 asks which campaign merge grew the per-round commit phase about 2.5× since `59b67f50` (E90 183.9 µs → E165 era 433.1 µs → current base 455.8 µs).
- Decision: stage 1 **UNCONFIRMED**. Stage 2 **not localisable to any merge**, with two mechanism findings that correct FINDING 482.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: base `9bf5bd6d` (`senpai/qwen38-mtp-r1`); measured on local research commit `6a253d9e`; no candidate, no scored-surface change.
- Yukon promoted submission / source ref used as frontier: crown `3.72911`; the four E165-era receipts are `5a9f130a`, `180db842`, `fda590bb`, `2c885d64`, all rejected, all head `559b24ebca35`.
- Candidate build fingerprint: see the per-leg table; every leg records `worker_sha256`.
- Submitted-surface / generated-twin / metallib digests: not applicable. No submission. Metallib source fingerprint `ce5e223eb6b404f1` on every leg.
- Submitted candidate files: **none**. This experiment changes no scored file.
- Supporting test, tooling, or documentation files: `research/e188_finding440_audit.py`, `research/e188-stage1.json`, `research/e188_commit_phase_bisect.py`, `research/e188_bisect_leg.sh`, `research/e188_bisect_ladder.sh`, `research/e188_commit_phase_read.py`, `research/e188_round_table.py`, `research/e188_stage2_wandb.py`, `research/e188-result.md`.
- MTP head provenance, digest, and draft policy: declared head, dir `mtp-head-declared-run`, `arm=ship`, `leaf=8`, `cap=7`, realized depth ramps 4 → 7.
- Token window, fixture, reference source, and harness: 128 decode tokens, `--local-iterate`, one public fixture, candidate-generated local reference rows, `harness=local`. Stage 1 uses only published Yukon board values, `harness=ranked`.
- Exact cell: not a kernel experiment. The measured cell is the host-side commit-phase window between `tReadDone` and `tCommitDone` in `Qwen36MTPBlockSession.swift`.
- Official causal path and score equation: none is claimed. Stage 1 prices published `harness=ranked` receipts only. Stage 2 is `harness=local` observation with no candidate edit, so it has no ranked score equation.
- Assignment-scope preflight: the diff against base is `research/` only.
- Editable source bytes / headroom / growth / exempt-head bytes: unchanged. Candidate growth is 0 bytes.
- Scored-path reachability evidence: the commit window is executed on every decode round; `mtp-trace` emits `commit_us` for all 19 rounds of every leg.
- Written promotion rule and verdict: this is a measurement-only assignment. There is nothing to promote. Verdict is recorded per stage below.
- Pre-official evidence budget / timed legs used: 8 timed legs, all `harness=local` screens. No 512-token confirmation and no submission, because there is no candidate.
- Frozen candidate SHA: none.
- Specific evidence that invalidated the frozen SHA: not applicable.
- Submission owner / read-only receipt-watcher job ID: not applicable.

---

## Stage 1 — does FINDING 440's ranked harm claim survive FINDING 460?

Verdict: **UNCONFIRMED**. FINDING 483's reopen condition fires.

Full working is in the PR comment `e188-stage1-verdict` and in `research/e188-stage1.json`. All stage 1 numbers are `harness=ranked`, read from the Yukon board field `officialMetrics.per_prompt`.

### 1.1 The arithmetic reproduces

The board reproduces FINDING 440's factor table to 4 decimal places: D−A `+0.8899`, C−A `+0.8765`, B−C `−0.6691`, D−B `+0.6878`. FINDING 440 made no arithmetic error.

### 1.2 The null was 4.58× too small

FINDING 440 priced its contrasts against `0.189 %`, which is `√2 ×` the serial-leg standard deviation of `0.1338 %`. All four contrasts are candidate-leg contrasts. FINDING 460 measures the candidate channel directly: serial σ `0.114 %`, MTP σ `0.612 %`, decode σ `0.626 %`.

| contrast | value | FINDING 440 z | corrected z |
|---|---:|---:|---:|
| D−A | +0.8899 % | 4.71 | 1.03 |
| C−A | +0.8765 % | 4.64 | 1.01 |
| B−C | −0.6691 % | −3.54 | −0.77 |
| D−B | +0.6878 % | 3.64 | 0.79 |

E165 alone is `(D−B) + (C−A) = +1.5644 %` with `σ = 2 × 0.612 = 1.224 %`, so `z = +1.28`. Nothing reaches 2σ.

### 1.3 The "plutarch nulls" defence has no statistical power

I measured the per-prompt within-group channel σ on E178's 35 byte-identical receipt groups (88 receipts, 53 degrees of freedom):

| prompt | within-group σ (%) |
|---|---:|
| travel | 0.954 |
| drama | 0.921 |
| beagle | 0.753 |
| medicine | 0.647 |
| botany | 0.630 |
| essays | 0.621 |
| republic | 0.583 |
| **plutarch** | **0.105** |

Plutarch's σ equals the serial-leg σ (`0.114 %`) because 449 of its 488 rounds do not draft. The receipt channel lives in drafting work, so plutarch is a near-noiseless control that carries almost no signal either. Worse, `corr(mean-of-7 deviation, plutarch deviation) = +0.141`, so differencing against plutarch does not cancel the channel: the paired σ is `0.978 %` against `0.865 %` unpaired. On the paired statistic D−B gives `z = +0.76`.

### 1.4 The observed shape contradicts the proposed mechanism

E165's prefetch is fixed host work per round, so its relative harm must scale as `1/R_p`, where `R_p` is the per-prompt round count. Cheap-round prompts must be hurt most. FINDING 440's own `+917 µs/round` predicts drama `+2.683 %` and botany `+1.683 %`. The observed D−B decode deltas are drama `+0.184 %`, the **smallest** of the eight, and botany `+0.813 %`. The ordering is inverted.

Fitting the 7 drafting prompts:

- a flat multiplicative offset of `c = +0.758 %` beats the `1/R_p` E165 shape by `1.69×` in sum of squared error;
- the additive component recovered from the `1/R_p` tilt is `−566 ± 550 µs/round`, `z = −1.03`, and its sign is a **gain**, not a harm;
- `+917 µs/round` is excluded at `2.7σ`.

Correction to my own earlier working: a first version of this fit included plutarch and reported `−680 µs/round` at `z = −2.77`. That was a leverage artifact of plutarch's near-zero σ. Restricting the fit to the 7 drafting prompts is the correct treatment and the honest number is `−566 ± 550 µs/round`.

### 1.5 Tree and confound audit

I fetched all four submission snapshots from `refs/heads/submissions/*` on `https://github.com/Layr-Labs/qwen-3.8-mtp-challenge` into local refs `refs/e188/{A,B,C,D,ORG}`.

- A is byte-identical to organizer `0863b06a` (empty diff).
- C is the organizer tree plus 2 files (`Qwen36MTPBlockSession.swift` +30, `Qwen35.swift` +88/−10).
- B is C's two files plus four Q files.
- D is the organizer tree plus the four Q files plus `Qwen36MTPBlockSession.swift` (444 lines).
- **The four Q files are byte-identical between B and D** (matching blob SHAs), so Q cancels exactly in D−B.
- A subagent audit of D found no third behavioural change: the prefetch, a value-identical pending-row hoist, and env-gated tracing (`MLX_QWEN_MTP_TRACE`, organizer code present in all four trees).
- The prefetch is ON by default in D (`!= "0"`); the maintained base has it OFF (`== "1"`).

**Correction for the record:** receipt C (`080d4cd3`) IS fetchable. FINDING 443 records it as unfetchable and is out of date.

### 1.6 Reopen pricing

Crown `3.72911` against drift-adjusted A `3.705` gives a gap of `0.6507 %` with published σ `0.689 %`.

| assumed true candidate gain | P(beat crown) |
|---|---:|
| 0.00 % | 0.172 |
| 0.38 % | 0.347 |
| 0.50 % | 0.413 |
| 0.66 % | 0.505 |
| 0.73 % | 0.546 |
| 1.00 % | 0.694 |

The `0.50 %` row reproduces the independently recorded `40.9 %`. One receipt cannot **confirm** a 0.38–0.73 % effect, but under FINDING 467 every distinct content gets one draw, so the decision is expected value, not statistical power.

### 1.7 Honest caveat

This shows only that the ranked evidence **against** E165 is absent. It does not show that E165 helps on M5. E165's local win is M4 Pro evidence and its M5 transfer remains untested.

### 1.8 Proposed, not built: the smallest ranked-decisive test

The assignment says propose without building. The smallest ranked-decisive test is **one receipt of the current maintained base with the E165 prefetch enabled, and nothing else changed**, submitted as a distinct content so FINDING 467 grants it a draw.

Rationale. The four E165-era receipts differ in two or three files at once and sit on an abandoned organizer tree, so every contrast among them mixes the prefetch with Q and with the pending-row hoist. A single-variable receipt on the live base makes the prefetch the only difference from the current promoted frontier, and its `raw_p` per-prompt vector is directly comparable with the frontier's. That is one submission slot, not a matrix. Do not build a matrix: with σ `0.689 %` on the published median, no affordable number of receipts can resolve a 0.5 % effect by power alone, so the only rational use of a slot is a positive-expected-value draw on a single clean variable.

---

## Stage 2 — bisect the commit-phase growth

Verdict: **the growth is not attributable to any merge's edit of the commit-phase window.** Two mechanism findings correct FINDING 482.

### 2.1 Free source bisection: 15 merges reduced to 3 candidates

`research/e188_commit_phase_bisect.py` hashes the window between `tReadDone` and `tCommitDone` plus every function it calls, across the 15 first-parent merges that touch `Qwen36MTPBlockSession.swift` since `59b67f50`.

- The window body is byte-identical at 13 of the 15 merges. It changes only at `b51f893a`, the E165 merge.
- `restoreAfterPrefixReject` changes at `7a427dfa` and `eec2c14b`.
- `hiddenRow`, `clearRecurrentRollback`, `rollbackAfterVerify`, `normedRows` and `snapshotRecurrent` never change.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift`, which declares `ArraysCache` and the three fields `clearRecurrentRollback` releases, is byte-identical (blob `4145e5aa`) across the whole range.
- `Package.resolved` is byte-identical across the whole range, so the MLX pin never moved.

That leaves four distinct window contents to time: `59b67f50` (blob `b946c17c`), `7a427dfa` (`da7606c7`), `806181de` (`f8482bbd`, the last pre-E165 content) and HEAD (`2457b231`).

Honest limit: `Qwen35.swift` changes at 9 of the 15 merges, so "provably inert" is too strong a phrase for the rest of the tree. The claim is confined to the window body and its named callees.

### 2.2 Timed ladder: no merge grew the ordinary commit phase

8 legs, 4 arms, 2 legs per arm, in the palindromic order `head, pre165, e90, e134, e134, e90, pre165, head`. A palindrome is the ABBA generalisation for four arms: every arm sits as far before the session midpoint as after it, so monotone thermal drift cancels to first order in each arm mean.

| leg | arm | rev | n | commit | readout | upkeep | eval_wall | round |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `e188-head-a` | post-E165 | HEAD | 18 | 440.5 | 42.0 | 268.0 | 67503.5 | 145219.0 |
| `e188-pre165-a` | pre-E165 | `806181de` | 18 | 461.0 | 43.0 | 264.0 | 67567.5 | 145108.5 |
| `e188-e90-a` | E90 anchor | `59b67f50` | 18 | 462.0 | 43.0 | 256.0 | 65395.0 | 145059.0 |
| `e188-e134-a` | E134 restore v1 | `7a427dfa` | 19 | 461.0 | 43.0 | 251.0 | 67568.0 | 145181.0 |
| `e188-e134-b` | E134 restore v1 | `7a427dfa` | 19 | 444.0 | 42.0 | 245.0 | 67554.0 | 145212.0 |
| `e188-e90-b` | E90 anchor | `59b67f50` | 18 | 446.0 | 43.0 | 258.5 | 67504.5 | 145142.0 |
| `e188-pre165-b` | pre-E165 | `806181de` | 18 | 444.5 | 44.0 | 263.5 | 67379.5 | 145137.5 |
| `e188-head-b` | post-E165 | HEAD | 18 | 450.0 | 42.0 | 267.0 | 67471.0 | 145054.0 |

All values are µs, medians over the leg with round 1 dropped.

The leg medians are flat: arm means span `445.2` to `454.0 µs`, and the within-arm leg-to-leg spread is `9.5` to `17.0 µs`, which is as large as the between-arm spread. **Swapping the commit-phase window body across four campaign revisions does not move the commit phase, and no arm approaches E90's 183.9 µs.**

### 2.3 The E90 comparison is common-mode, not commit-specific

| phase | E90 (ledger 234.5) | E188 mean over 8 legs | ratio |
|---|---:|---:|---:|
| `readout` | 14.8 µs | 42.8 µs | 2.89 |
| `commit` | 183.9 µs | 451.1 µs | 2.45 |
| `upkeep` | 66.7 µs | 259.1 µs | 3.88 |
| `eval_wall` | 76434.2 µs | 67242.9 µs | 0.88 |
| ROUND | 165183.9 µs | 145139.1 µs | 0.88 |

Every host-only, ~100 % idle phase grew 2.4× to 3.9×. Both GPU-bound phases shrank by the same 0.88×. FINDING 482 names only the commit phase, but `readout` and `upkeep` grew as much or more, and section 2.2 shows the window body does not produce any of it.

The token window is not the cause. E185's 455.8 µs came from a ~70-round leg and E90's 183.9 µs from a 78-round leg, both about 512 tokens, and my 128-token legs reproduce E185's value to within 3 %.

**Two instrument definitions, kept explicit as the advisor asked.** E90 and E185 use the GPU interval ledger: MTLCommandBuffer `gpuStart`/`gpuEnd` unions intersected with round windows, which attributes an *interval* to a phase and splits it into GPU busy and GPU idle. `mtp-trace` uses host `DispatchTime.now().uptimeNanoseconds` deltas between named points on the decode thread; `commit_us` is exactly `tCommitDone − tReadDone`. The two agree on the current base (455.8 against 440.5–462.0), so they measure the same window there. Whether they agreed on the E90 tree cannot be checked from this side, because I can only run the current tree. That is the open half of the reconciliation.

### 2.4 The multi-millisecond spikes are the prefix-reject repair, and it regressed at `eec2c14b`

The per-round trace is deterministic and repeats exactly across legs. In every 19-round leg, rounds 3, 11 and 18 are the only rounds where `acc < d`, and they are the only rounds with a multi-millisecond commit phase.

`restoreAfterPrefixReject` is the only commit-window callee the bisection moves, and it runs **only** when `acc < d`. A median over all rounds is dominated by full-acceptance rounds and cannot see it. Splitting by acceptance outcome:

| arm | rev | full-accept n | full-accept commit | reject n | reject commit | repair cost | round excess |
|---|---|---:|---:|---:|---:|---:|---:|
| E90 anchor | `59b67f50` | 30 | 450.0 µs | 6 | 1659.5 µs | +1209.5 µs | +1331.0 µs |
| E134 restore v1 | `7a427dfa` | 34 | 446.5 µs | 4 | 1664.0 µs | +1217.5 µs | +1609.0 µs |
| pre-E165 | `806181de` | 30 | 444.0 µs | 6 | 2694.5 µs | +2250.5 µs | +2317.5 µs |
| post-E165 | HEAD | 30 | 441.0 µs | 6 | 2820.0 µs | +2379.0 µs | +2885.0 µs |

Individual rejection-round values:

- `59b67f50`: 1577, 1643, 1648, 1671, 1673, 1771
- `7a427dfa`: 1645, 1662, 1666, 1856
- `806181de`: 1631, 2508, 2606, 2783, 2842, 3025
- HEAD: 2111, 2560, 2759, 2881, 2886, 3051

Pooling the two arms whose window content predates merge `eec2c14b` against the two that follow it gives a Mann-Whitney `U = 111/120` with `n_pre = 10` and `n_post = 12`, median `1664.0 → 2771.0 µs`, step **`+1107 µs`**, `z = 3.33`. Only one of the twelve post-merge rounds falls inside the pre-merge range.

The headline uses only the eight counterbalanced ladder legs. The ninth leg, the §2.6 release probe, is also HEAD source and therefore also a post-merge sample, but it ran outside the palindromic order, so folding it in would break the counterbalancing. As a sensitivity check only: adding it gives `U = 141/150`, `n_post = 15`, step `+1095 µs`, `z = 3.63`. It moves the estimate by `12 µs` and strengthens the test, so the choice of pool does not carry the conclusion.

The step lands at the merge the free source bisection had already named. `eec2c14b` (PR #152) replaces `prefetchRecurrentBoundary(cache)` in the K=1 restore path with an inline `asyncEval` over the 48 Gated DeltaNet boundary states, and removes `prefetchRecurrentBoundary(cache)` from the generic repair fallback:

```swift
-            prefetchRecurrentBoundary(cache)
+            let replayedRecurrentStates = cache.compactMap { entry -> MLXArray? in
+                guard let arrays = entry as? ArraysCache else { return nil }
+                return arrays[1]
+            }
+            asyncEval(replayedRecurrentStates)
```

The in-source comment attributes the change to E020 replay-prefetch and records a promoted `+0.039 %`. HEAD still carries it.

**The extra cost is not hidden.** The round excess on rejection rounds tracks the commit excess almost one for one: repair rises `+1041 µs` from the E90 anchor to `806181de` while the round excess rises `+986 µs`. The asynchronous submission does not buy the time back inside the same round at this token window.

Priced honestly: rejections are 3 of 18 scored rounds here, so `986 µs × 0.167 = 165 µs/round` on a `145.1 ms` round, which is **`0.113 %`**. That is below the `0.39 %` MUE as a single mechanism. It is a real, localised, 100 %-GPU-idle seam cost with a named cause, not a candidate that pays for itself alone.

**Two honest limits.** First, the `7a427dfa` arm is not workload-matched: it diverges from the shared trajectory at round 2 and produces 20 rounds instead of 19, so it yields 4 rejection rounds rather than 6. The `59b67f50` arm reproduces the current trajectory exactly, so the matched contrast is `59b67f50` against `806181de` and HEAD; `7a427dfa` only corroborates. Second, a rejection rate of 3/18 comes from one public fixture at 128 tokens and will differ on the hidden prompts, so the `0.113 %` figure is a local point estimate, not a ranked price.

### 2.5 A regime shift that draft depth does not explain

Rounds 1 and 2 cost 173–184 µs, which reproduces E90's 183.9 µs on the current base and the current host. Round 3 is the first rejection. From round 4 onward the full-acceptance commit phase costs about 440 µs and never returns to the cheap level.

Realized draft depth ramps 4 → 5 → 6 → 7 over the same rounds, so depth and round index are confounded. Pooling all 8 legs:

| d | n | commit | readout | upkeep |
|---:|---:|---:|---:|---:|
| 1 | 8 | 231.5 | 34.5 | 108.0 |
| 4 | 18 | 181.5 | 34.5 | 93.5 |
| 5 | 18 | 432.0 | 40.5 | 222.5 |
| 6 | 22 | 452.5 | 43.0 | 260.0 |
| 7 | 65 | 450.0 | 43.0 | 273.0 |

Depth alone does not explain the step. `d = 5` appears both before and after the transition: round 2 runs `d = 5` at 181 µs, and rounds 4 and 5 run `d = 5` at 428 and 436 µs. The break tracks the first rejection, not the depth. `d = 4` occurs only in rounds 1–2, so its cell is not an independent depth contrast.

### 2.6 The advisor's discriminator: is the step the cost of releasing populated state?

The advisor asked whether the step is the cost of releasing populated recurrent state. I built the smallest probe that can answer it. `research/e188_clear_timer_leg.sh` wraps a `DispatchTime` pair around the single `clearRecurrentRollback` call site and makes that function return the number of `MLXArray` references it drops. It publishes `clear_release_us` and `clear_release_count` in the trace. It changes no behaviour: `e188_arm_behaviour_change=none`, and the build certificate required both new trace keys to be present in the binary. Job `5d396519`, tag `e188-clear-timer`, worker `f864af0b83b5…`, exit 0, 19 rounds.

| round | d | acc | commit_us | clear_release_us | clear_release_count | commit − clear |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 178 | 144 | 48 | 34 |
| 2 | 5 | 5 | 183 | 170 | 48 | 13 |
| 3 | 5 | 4 | 2564 | 0 | 0 | 2564 |
| 4 | 5 | 5 | 448 | 397 | 48 | 51 |
| 5 | 5 | 5 | 448 | 397 | 48 | 51 |
| 6 | 6 | 6 | 417 | 372 | 48 | 45 |
| 7 | 6 | 6 | 442 | 393 | 48 | 49 |
| 8 | 6 | 6 | 444 | 394 | 48 | 50 |
| 9 | 7 | 7 | 443 | 395 | 48 | 48 |
| 10 | 7 | 7 | 447 | 398 | 48 | 49 |
| 11 | 7 | 6 | 2580 | 0 | 0 | 2580 |
| 12 | 7 | 7 | 435 | 389 | 48 | 46 |
| 13 | 7 | 7 | 600 | 534 | 48 | 66 |
| 14 | 7 | 7 | 443 | 391 | 48 | 52 |
| 15 | 7 | 7 | 443 | 401 | 48 | 42 |
| 16 | 7 | 7 | 449 | 402 | 48 | 47 |
| 17 | 7 | 7 | 597 | 533 | 48 | 64 |
| 18 | 7 | 6 | 2878 | 0 | 0 | 2878 |
| 19 | 1 | 1 | 244 | 132 | 96 | 112 |

**Result: the release hypothesis is confirmed as the location, and the advisor's stated count prediction is falsified.**

`clearRecurrentRollback` is the commit phase. Over the 13 full-acceptance rounds after the transition (`n = 13`, rounds 4–10 and 12–17) the medians are `commit 444.0 µs`, `clear_release 397.0 µs`, residual `47.0 µs`. The release is **89.4 %** of the ordinary commit phase. Over the two rounds before the first rejection (`n = 2`) the medians are `commit 180.5 µs`, `clear_release 157.0 µs`, residual `23.5 µs`. So the step is `+263.5 µs` in commit and `+240.0 µs` in the release, and the release carries **91.1 %** of it. Nothing else in the commit phase moves: the residual rises only `+23.5 µs`.

The advisor predicted `clear_release_count = 0` in rounds 1–2 and positive from round 4. That is wrong. The count is **48 in every round that calls the function**, before and after the transition, and `0` in the three rejection rounds because the rejection path takes `restoreAfterPrefixReject` and never reaches the clear. So the step is not "more references to release". It is **the same 48 references costing 2.53× more each to release**, from `3.27 µs` to `8.27 µs` per reference. 48 is one array per Gated DeltaNet layer, which matches the 48 recurrent layers in the target.

Two limits on this cell. `n_pre = 2` is only rounds 1 and 2, because round 3 is the first rejection and there is no third pre-transition round to have. Round 1 also pays the cold caches, but its commit phase, `178 µs`, sits inside round 2's, `183 µs`, so the cold start does not appear to reach this phase. A longer pre-transition window would need a fixture whose first rejection comes later, which I did not have.

This also confirms the §2.4 split independently: the multi-millisecond rejection rounds contain no release work at all, so their cost is entirely `restoreAfterPrefixReject`. The two mechanisms are disjoint.

**What the probe rules out.** A monotone allocator-fragmentation account predicts a cost that drifts upward across the leg. It does not. The release cost jumps once between round 2 and round 4 and then stays within `372–402 µs` for eleven of the thirteen, with two `533–534 µs` excursions at rounds 13 and 17. That is a one-time regime change, not accumulation, so fragmentation is disfavoured.

**What the probe does not separate.** Two accounts survive, and this probe cannot tell them apart:

- **In-flight GPU references.** The restore path submits asynchronous work over the recurrent boundary. Once that has happened, dropping the last host reference to a boundary array may have to interact with a command buffer that still holds it. Rounds 1–2 precede any rejection, so no such submission has occurred yet.
- **Buffer identity or size.** Before the first restore the snapshot arrays may alias or be smaller; after a restore they are freshly materialised full recurrent states, so freeing them returns larger blocks.

**One conclusion that constrains the cause.** It is tempting to blame `eec2c14b`, since §2.4 already convicts it. That is not available here. The `59b67f50` arm predates `eec2c14b` and still shows a `450.0 µs` full-acceptance commit median, indistinguishable from HEAD's `441.0 µs`. The elevated steady-state release cost therefore already existed before that merge. The older code submitted `prefetchRecurrentBoundary(cache)` on the same path, so the trigger is the restore path's asynchronous submission over the recurrent boundary **in general**, which is present across the whole bisection range. That is exactly why the eight-leg ladder is flat. `eec2c14b` is guilty of the `+1107 µs` on the rejection rounds themselves and not of this.

### 2.7 Proposed fix — proposed only, not built

Per the assignment I propose and do not build. Ranked by expected value:

1. **Make the rollback slot persistent and stop releasing it.** Preallocate the 48 recurrent snapshot arrays once and overwrite them in place at snapshot time, instead of allocating a fresh snapshot each round and dropping the previous one on the commit path. If the release is removable, the ordinary commit phase falls from `444 µs` toward the `47 µs` residual, which is `397 µs` on a `145.1 ms` round, or **up to `0.27 %`**. Two things make this the better target than the §2.4 seam: it is paid on 15 of 18 rounds rather than 3 of 18, and `0.27 %` is a per-round saving rather than a rejection-rate-weighted one. Treat `0.27 %` as an upper bound. It assumes the release is fully removable and that the cost does not reappear as an in-place-write or synchronisation cost elsewhere.
2. **Defer the release off the commit phase.** Move the 48 references to a queue drained after the round boundary. This relocates the work rather than removing it, so it wins only where it overlaps GPU time that is currently idle. It is strictly weaker than 1 and is the fallback if in-place snapshotting turns out to break rollback exactness.
3. **Reorder the restore-path submission.** If the in-flight-reference account is the true one, issuing the boundary `asyncEval` on separately held copies, or after the release, may restore the cheap `157 µs` regime without changing the rollback contract.

**The cheapest next diagnostic is one leg of about 3.5 minutes** and it decides between 1, 2 and 3: time the release with and without a preceding explicit evaluation barrier. If a barrier immediately before the clear absorbs the `240 µs`, the cost is waiting on in-flight command buffers and option 3 is the fix. If the clear still costs `397 µs` after a barrier, the cost is the allocator free itself and option 1 is the fix. I did not run it because the assignment stops at proposal.

**Ranked-boundary note.** This is a broad candidate-runtime change, not a schedule or head-policy change. Its causal path is not confined to the candidate MTP leg, so per `program.md` it must be screened on matched absolute candidate MTP seconds per token against a fresh unchanged base, and **not** on the local serial-to-MTP ratio, where a general improvement can cancel. `harness=local` for every number in this section.

---

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy: `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 51539607552 bytes (48 GiB), sandbox off, `MLXFAST_LOCAL_COOL_GATE=0`. Every leg records `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`. These are ungated counterbalanced local arms, permitted by the program for directional causal evidence. They are not gate-qualified and are not comparable with gated historical runs as though they were.
- `head_provenance_sha256` for every leg: all nine legs used the same declared proposal head directory, `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run`, holding `config.json` and `model.safetensors`. The digest of the sorted per-file SHA-256 list is `140df0056bf2055ec0b766ba29431a8999740270a774eaeccb92b4acd5e1e838`. The head is therefore constant across every arm and is not a source of arm-to-arm difference.
- Exact baseline and candidate commands:

```bash
# free source bisection
python3 research/e188_commit_phase_bisect.py

# one bisection point (swaps one file, rebuilds with a symbol certificate,
# restores the file on every exit path)
research/e188_bisect_leg.sh REV TAG 128

# the remaining six legs, palindromic arm order
research/e188_bisect_ladder.sh

# the release-timing probe (zero behaviour change, restores the file on exit)
research/e188_clear_timer_leg.sh

# per-round table for any leg
python3 research/e188_round_table.py e188-clear-timer

# analysis and W&B publication
python3 research/e188_stage2_wandb.py --legs e188-head-a e188-pre165-a \
  e188-e90-a e188-e134-a e188-e134-b e188-e90-b e188-pre165-b e188-head-b \
  e188-clear-timer

# stage 1
python3 research/board_per_prompt.py fetch
python3 research/e188_finding440_audit.py --json research/e188-stage1.json
```

- Cheapest real falsification gate and positive-control verdict: every bisection leg rebuilds the worker and asserts the arm by symbol. Post-E165 arms require `prefetchHeadStep` to be present; pre-E165 arms require it to be absent. `restoreAfterPrefixReject` is the positive control that proves the symbol table was read at all. Every leg passed its certificate, and the pre-E165 and HEAD arms produced distinct `worker_sha256` values, so the two arms are provably different binaries.
- Tests and risk-based checks, in execution order: symbol certificate per leg; `worker_sha256` recorded before and after each leg; `dirty_candidate_paths=0` recorded per leg; tree restored to HEAD content after every leg.
- Exact-token and row-ledger verdict: not applicable. No scored-surface change was made, so there is no fidelity claim to check. Every leg ran the standard `--local-iterate` path and exited 0.
- Divergent tokens or failure category: none.
- Generated-twin audit: not relevant. No Metal source changed.
- Peak RAM or head/artifact size: unchanged.
- Official status and score: no submission.
- W&B project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group `qwen38-r1-e188-commit-phase-bisection`. Every leg run carries its full identity tuple in `config` and its per-round series as a step series, so the round-3 regime transition is visible in the UI.

| run | id | URL |
|---|---|---|
| `e188-head-a` | `j2wslp2a` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/j2wslp2a |
| `e188-pre165-a` | `0i23qzt9` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/0i23qzt9 |
| `e188-e90-a` | `reid2oxh` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/reid2oxh |
| `e188-e134-a` | `qo51eveb` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/qo51eveb |
| `e188-e134-b` | `uswipt1m` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/uswipt1m |
| `e188-e90-b` | `yidza6d0` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/yidza6d0 |
| `e188-pre165-b` | `bybbp96e` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/bybbp96e |
| `e188-head-b` | `892px8u9` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/892px8u9 |
| `e188-clear-timer` | `6nt69geq` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/6nt69geq |
| rollup | `odw531zt` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/odw531zt |

The rollup run carries the arm table, the `eec2c14b` rank test, the depth profile and the release-probe metrics under the `arm/`, `rankTest/`, `depth/` and `clearProbe/` prefixes. Its `ladderLegs` config field records the eight counterbalanced legs that produce every pooled statistic, and `publishedLegs` records all nine.
- Runtime and peak memory: each bisection leg is about 46 s of rebuild plus 154 s of timed leg, so about 3.5 min. Nine legs plus rebuilds is about 32 min of host time. Peak memory is unchanged from the base; no arm alters allocation behaviour.

### Identity-field match statement

All nine legs match on every identity field except the one dimension each leg is meant to vary.

| field | value |
|---|---|
| `BASE_SHA` | `9bf5bd6d` assignment base; advisor ruled no replay for the move to `e0c0dec7` |
| host / instance / chip | `ip-10-231-2-22.ec2.internal`, Apple M4 Pro |
| memory profile | `51539607552` bytes, sandbox off |
| thermal mode | `MLXFAST_LOCAL_COOL_GATE=0`, ungated, ABBA counterbalanced |
| token window / fixture | 128 tokens, `--local-iterate`, one public fixture |
| proposal head | `mtp-head-declared-run`, content digest `140df005…`, identical in all nine legs |
| metallib fingerprint | `ce5e223eb6b404f16f07aca83ca44e65fc517187f3b130edaf4e7acb56789d50`, identical in all nine legs |
| harness / reference source | `harness=local`; candidate-generated local reference rows |
| varied dimension | one swapped `Qwen36MTP*.swift` revision per leg (legs 1–8), or the release timer patch (leg 9) |

Entry GPU temperature spans `34.9 °C` to `50.5 °C` across the eight ladder legs, and exit temperature spans `60.3 °C` to `63.5 °C`. The spread is why the ladder is palindromic: `head, pre165, e90, e134 | e134, e90, pre165, head` makes monotone thermal drift cancel to first order between the A and B halves of each arm. The A-half and B-half commit medians for the same arm agree within `17 µs` for every arm, which is the practical bound on residual thermal contamination at this measurement.

Two honest identity notes. First, `dirty_candidate_paths=1` is recorded for every leg that swaps a file, including the release probe, because the swap is by construction a dirty worktree; each leg restores the file on every exit path and `git status --short` is empty after the run. Second, `e188-head-a` and `e188-head-b` build the same source but produce different `worker_sha256` values, so the Swift build is not bit-reproducible here. Arm identity is therefore established by the symbol certificate, not by binary digest equality.

---

## Conclusion

**Stage 1 verdict: UNCONFIRMED.** FINDING 440's claim that E165's prefetch harmed the ranked M5 score by `+0.69 %` to `+1.56 %` does not survive FINDING 460's receipt-channel model. The board reproduces FINDING 440's factors to four decimal places, so the arithmetic was never the problem; the null was. FINDING 440 priced candidate-leg contrasts against a serial-leg σ that is `4.58×` too small, and the corrected `z` values are all `≤ 1.28`. The independent shape test agrees: a `+917 µs/round` additive harm predicts a per-prompt ordering that is inverted relative to the board, a flat multiplicative fit beats the additive fit by `1.69×` in SSE, and the fitted additive component is `−566 ± 550 µs/round`, which excludes `+917` at `2.7σ`. The plutarch-null defence has no power at `σ = 0.105 %` and is recorded as VOID. Because the harm is unconfirmed rather than disproved, the seam axis reopens, and the single-variable official E165-prefetch retest in §1.8 is the way to settle it. The advisor has adopted and queued that retest; it is not built here. One correction to the record: receipt C `080d4cd3` is fetchable, so FINDING 443 is out of date.

**Stage 2 verdict: the `~2.5×` commit-phase growth since `59b67f50` is not in the source.** The eight-leg ABBA ladder is flat. Full-acceptance commit medians run `440.5, 461.0, 462.0, 461.0, 444.0, 446.0, 444.5, 450.0 µs` across HEAD, pre-E165, E90 and E134 arms in palindromic order. No merge in the range grew the ordinary commit phase, and the free source bisection agrees: the window between `tReadDone` and `tCommitDone` is byte-identical at 13 of 15 merges, and `KVCache.swift` and `Package.resolved` are byte-identical across the whole range.

Two real mechanisms came out of it instead.

1. **A rejection-round seam at `eec2c14b` (PR #152), `+1107 µs`, `z = 3.33`, `U = 111/120`.** `restoreAfterPrefixReject` runs only when `acc < d`. Rejection rounds are the only multi-millisecond commit rounds, and the repair cost steps from a `1664 µs` median before the merge to `2771 µs` after it. The merge replaced `prefetchRecurrentBoundary(cache)` with an inline `asyncEval` over the 48 GDN boundary states in the K=1 restore path. The round excess tracks the commit excess roughly one for one, so the asynchronous submission does not pay for itself at this token window. Priced at the local rejection rate of 3/18 it is `0.113 %`, which is below the `0.39 %` MUE as a standalone candidate.
2. **A larger, previously unnamed cost: `clearRecurrentRollback` is 89.4 % of the ordinary commit phase.** The release probe shows `397 µs` of a `444 µs` commit phase is the release of 48 recurrent snapshot references, one per GDN layer. The cost jumps once, from `157 µs` to `397 µs`, at the first rejection round, at a constant reference count of 48, so the per-reference cost rises `2.53×` rather than the count rising. The advisor's count prediction is falsified; the release location is confirmed. The jump is a one-time regime change and not monotone drift, which disfavours allocator fragmentation. It predates `eec2c14b`, so the trigger is the restore path's asynchronous submission over the recurrent boundary in general, not that merge.

**The larger target is the second one, and it is proposed only.** A persistent, in-place rollback slot would remove up to `397 µs` on 15 of 18 rounds, up to `0.27 %` per round, against `0.113 %` weighted by a 3/18 rejection rate for the seam. One further leg of about 3.5 minutes — timing the release with and without a preceding evaluation barrier — separates the in-flight-reference account from the allocator-free account and decides which of the three proposed fixes is correct.

**Honest limits on the whole of Stage 2.** Every number is `harness=local`, ungated, at 128 tokens, on one public fixture, on an M4 Pro rather than the ranked M5. The `3/18` rejection rate is a property of this fixture and will differ on the hidden prompts, so `0.113 %` and `0.27 %` are local point estimates and not ranked prices. `Qwen35.swift` changed at nine merges in the range and the free bisection cannot exonerate it by inspection, though the flat ladder covers it empirically. The `7a427dfa` arm is not workload-matched and only corroborates. No scored-surface change was made and no submission was produced.

---

## Suggested follow-ups

I did not implement any of these. They are ordered by expected value.

1. **The barrier diagnostic, one leg, about 3.5 minutes.** Time `clearRecurrentRollback` with and without a preceding explicit evaluation barrier. It separates the in-flight-reference account from the allocator-free account and decides which of the three §2.7 fixes is correct. This is the cheapest high-information measurement left on this axis.
2. **The persistent in-place rollback slot, §2.7 option 1.** Up to `397 µs` on 15 of 18 rounds, up to `0.27 %` per round, `harness=local`. It is the largest single local cost this experiment found. It must be screened on matched absolute candidate MTP seconds per token, not on the local ratio.
3. **Measure the rejection rate at 512 tokens and across prompts.** Both mechanisms are priced by that rate, and `3/18` comes from one public fixture at 128 tokens. If the hidden prompts reject more often, the `eec2c14b` seam is worth more than the `0.113 %` recorded here. If they reject less, it is worth less. The same measurement also bounds how much of the `0.27 %` release cost is really paid.
4. **The upkeep phase grew more than the commit phase, and no one has profiled it.** Against the E90 record, commit grew `2.45×` but upkeep grew `3.88×`, from `66.7 µs` to `259.1 µs`. At `259 µs` on a `145.1 ms` round that is `0.18 %`, which is larger than the `eec2c14b` seam. It was outside this assignment and it has no phase-resolved instrument yet. A probe of the same shape as `research/e188_clear_timer_leg.sh` would locate it in one leg.
5. **`Qwen35.swift` is not exonerated by inspection.** It changed at nine of the fifteen merges in the range. The flat ladder covers it empirically for the commit window. If a future phase regression appears outside that window, it is the first place to bisect.
6. **The §1.8 single-variable official E165-prefetch retest.** The advisor has adopted and queued it. It is the only instrument that can settle the reopened seam axis, because no local harness can reproduce the ranked numerator.

