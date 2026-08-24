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

### 2.2 Timed ladder: PLACEHOLDER

### 2.3 The E90 comparison is common-mode

PLACEHOLDER

### 2.4 The multi-millisecond commit spikes are the prefix-reject repair

PLACEHOLDER

### 2.5 A second regime shift that draft depth does not explain

PLACEHOLDER

---

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy: `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 51539607552 bytes (48 GiB), sandbox off, `MLXFAST_LOCAL_COOL_GATE=0`. Every leg records `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`. These are ungated counterbalanced local arms, permitted by the program for directional causal evidence. They are not gate-qualified and are not comparable with gated historical runs as though they were.
- `head_provenance_sha256` for every leg: PLACEHOLDER
- Exact baseline and candidate commands:

```bash
# free source bisection
python3 research/e188_commit_phase_bisect.py

# one bisection point (swaps one file, rebuilds with a symbol certificate,
# restores the file on every exit path)
research/e188_bisect_leg.sh REV TAG 128

# the remaining six legs, palindromic arm order
research/e188_bisect_ladder.sh

# analysis and W&B publication
python3 research/e188_stage2_wandb.py --legs e188-head-a e188-pre165-a \
  e188-e90-a e188-e134-a e188-e134-b e188-e90-b e188-pre165-b e188-head-b

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

### Identity-field match statement

PLACEHOLDER

---

## Conclusion

PLACEHOLDER
