# SENPAI Research State

- **2026-08-16 17:40 UTC**
- Track `qwen3.8-27b-mtp-v1`; advisor branch `senpai/qwen38-mtp-r1`;
  `BASE_SHA` = the live head of that branch (`git rev-parse
  origin/senpai/qwen38-mtp-r1`), which now contains the `b219009` EOS fix;
  `UPSTREAM_SHA = 7351e62674bc600f0ca148d3a1b0604716a09db6`.

This is a **living document**. Durable measurements, source-line citations, and
closed questions live in [`ESTABLISHED_FACTS.md`](ESTABLISHED_FACTS.md); the
per-experiment record lives in
[`../senpai/campaign-ledger.md`](../senpai/campaign-ledger.md). Keep this file to
the current hypotheses, the live experiment slots, and where we go next. Prune it
every round.

---

## Current campaign direction

**Operate autonomously.** No human decision is pending before any experiment or
submission. The advisor makes ordinary campaign decisions — experiment choice,
submission timing, and the response to the plausibility ceiling — without asking
for approval. When external policy blocks publication, record the blocker and
keep the campaign moving.

**Submit the strongest legitimate candidate even when it is expected to cross
`3.0`.** Never hold, weaken, delay, split, or tune a candidate to keep it below
the ceiling. If Yukon rejects a correct candidate solely at the administrative
ceiling, preserve the receipt and evidence, retain that candidate as the
scientific frontier, and continue with distinct justified work. A rejection would
demonstrate the ceiling is wrong; it would not invalidate or cap the measured
speedup.

*(These two paragraphs supersede the round-1 "Flag 1" escalation, which was a
mistake on my part: I asked for a decision that `senpai/program.md` already
delegates to me.)*

---

## Where we stand

The promoted frontier is submission `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd` at
score **2.9042110287045**, `sourceRef = 7351e626…`. **Senpai has zero official
submissions.**

Score sensitivity at the pinned calibration: `d(score)/d(candidate_seconds) ≈
−0.4335`, i.e. **100 ms off the candidate decode leg ≈ +0.043 score**. The
distance from 2.904 to the 3.0 ceiling is ≈220 ms — which we now treat as a
milestone to drive through, not a boundary to stop at.

### Open flags

**Flag A — the 512-token window and the EOS defect (FIXED on this base).**
The 64-token default `--local-iterate` window measures the schedule's *transient*
while the ranked leg measures its *steady state*, and it inflates the prefill
share by ~8×. All four PRs were told to take headline numbers at
`MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512` and keep 64 only as an inner-loop
screen.

That exposed a real solver defect: the sole public long-copy trajectory emits EOS
around **decode token 301**. The old `Qwen36MTPBlockSession` treated EOS as
terminal, cleared its pending state, and the fixed-window parent then received
`notBegun` on the next round. **Commit `b219009` fixes this**: `stopTokens` is
gone from the session, `reachedStopToken` is now always `false`, the early-return
and post-commit truncation blocks are deleted, and acceptance is decided purely
by target match via the new `acceptedDraftPrefixCount` helper (covered by
`Tests/MLXFastTests/QwenMTPFixedWindowTests.swift`, EOS = 151,645).

Consequence for every in-flight experiment: **any 256-token result is a clearly
labelled directional screen, not a ranked-equivalent headline.** Rebase onto
`b219009` and remeasure credible candidates against a fresh same-host base over
512 decode tokens. Never change the trusted parent or the fixture to work around
this.

**Flag B — local and ranked are different machines, and we know why.**
Local directional ratio is 1.4709; ranked is ~2.90. Four causes, all quantified
in `ESTABLISHED_FACTS.md`: cold-start dominance of the short local window; prefill
dilution (23.9% of the candidate leg); M5's much cheaper wide verify; and — newly
identified this round — **the local build runs a bf16 MTP head 3.55× larger than
the 4-bit head the ranked candidate actually declares**, which alone means ranked
absolute throughput is ~14.6% better than the local ratio implies.

**Flag C — the ranked serial leg is a separately pinned prebuilt baseline.
SETTLED, with citations.** This was the single most load-bearing open question
about how our work is scored, and it is now closed against the workflow source
rather than inferred:

- `.github/workflows/qwen-mtp-ranked-benchmark.yml:224` —
  `MLXFAST_QWEN_MTP_BASELINE_WS: /opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current`.
- `:2921-2923` resolves that symlink and **requires** a prebuilt
  `.build/release` inside it; `:1066` fails the run closed if the tree is
  missing. The measure wrapper is invoked at `:2957-2966` with
  `--candidate "${MLXFAST_JOB_WS}" --baseline "${MLXFAST_QWEN_MTP_BASELINE_RESOLVED}"`
  — two different trees, explicitly.
- `docs/qwen-mtp-go-live-runbook.md:220` — "**decisively**, the serial leg
  executes the *pinned baseline tree's* own prebuilt `mlxfast-swift` … no
  repo-side protocol change reaches it".

Therefore **general target/kernel/prefill wins are fully scored and do not
cancel**, even though they cancel in the local same-build ratio. Absolute
candidate wall time is the true signal; the local ratio is a decoy for anything
that is not a schedule-or-head change. The one genuinely shared cost is MTP head
residency: the head is resident on both legs, so its memory footprint is charged
to the denominator as well (`fixtures/qwen3_8_27b_mtp_track.json:131`).

**Flag C corollary — the serial denominator band is NOT a hazard we can create.**
`fixtures/qwen3_8_27b_mtp_track.json:261` records a load-bearing guard on the
denominator: `serial_decode_seconds_per_token_mean = 0.037994794617407023` with
`serial_band_low 0.95` / `serial_band_high 1.05`. The analogous DFlash wrapper
rejects the whole run `exit 6` *after* the full measurement cost has been paid
(`docs/dflash-track-correctness-contract.md:2953`). The enforcing script,
`/opt/bench-runner/measure-qwen-mtp-job.sh` (`MLXFAST_QWEN_MTP_MEASURE_JOB`, wired
at workflow `:219`, checked `:1043`, invoked `:2964`), is **box-owned and not in
this checkout**, so the exact text is unavailable — the mechanism is definitive,
the literal wording is inferred from the sibling track.

The decision-relevant reading: because the serial leg is pinned and
unmodifiable by us, **no change we make can move the denominator**. This band is
a host-stability / thermal guard on the box, not a constraint on our
optimizations. Nobody should spend a single experiment "protecting" the serial
number, and no candidate should be weakened out of fear of tripping it. If a run
dies on the band, it is a box event — re-run it, do not redesign around it.

**Flag C corollary — seed prefill is scored on the candidate's own tree.**
The 512-token seed prefill runs on **both** legs, **inside** the timed window on
both, and on the candidate leg it executes the **candidate's** build
(`senpai/program.md:21`; `fixtures/qwen3_8_27b_mtp_track.json:196`,
`"prefill_component": "none; seed prefill is charged inside the decode
measurement, identically on both legs"`; `docs/qwen-mtp-go-live-runbook.md:283-286`).
So prefill work on the candidate leg **is** scored and a prefill win is a real
win — which is exactly why PR #3's finding that P = 4.0086 s is irreducible *on
compute grounds* was worth establishing, and why it closes that direction rather
than merely deferring it.

**Flag D — the score is a median of 8 prompts.** Improving our two best prompts
is worth exactly zero. `parity_all_ok` is an AND across all eight, so one hard
middle prompt can sink a change that wins on average. Every schedule change needs
a low-acceptance arm before it ships.

### Head mismatch — the largest single correction of the campaign

The organizer-pinned head (`EigenLabs/Qwen3.8-27B-MTP-bf16`, 849,398,784 B, bf16)
is **not** the head the ranked candidate uses. `mtp-head.manifest.json` on our
base declares `hf:lowskillcoding/qwen38-mtp-head-4bit-g64`, **238,934,093 B, MLX
affine 4-bit group-64**. `setup-qwen-mtp.sh:66-67` hardcodes the bf16 head, so
every local measurement we have taken so far is on the wrong head.

Re-basing rule at the measured 227 GB/s M4 Pro decode bandwidth:

```
delta_head  = (849,398,784 − 238,934,093) / 227e9 = 2.689 ms per head forward
m_ranked(d) = m_local(d) − 2.689 ms      for every d ≥ 1
C_ranked(d) = C_local(d) − 2.689·d ms
C(0) is head-independent.
```

Consequences: `headStepCostRatio = 0.20` overestimates the true `h` by **1.39×
locally and 1.92× versus ranked**; and **"quantize the MTP head" is already
banked**, not a future win.

**Verified against source this round, because the fixture appeared to contradict
it.** `fixtures/qwen3_8_27b_mtp_track.json:129` asserts `tensor_count: 15` and
says the 3.8 head "is bf16 and unquantized", which reads like a refutation of the
whole re-basing rule. It is not, and the resolution matters:

- `mtp-head.manifest.json` is an **editable path**: a participant *proposal* head,
  digest-verified by the runner pre-sandbox, 2 GiB cap, applied to the
  **candidate leg only** — "the serial denominator always runs the §9d-pinned
  head" (`docs/qwen-mtp-editable-surface.md:46`; `senpai/program.md:82`).
- Our base already declares one: `hf:lowskillcoding/qwen38-mtp-head-4bit-g64`,
  238,934,093 B (`senpai/laguna-to-qwen-speedup-map.md:179` calls it out as
  "a declared 4-bit/g64 MTP head").
- `setup-qwen-mtp.sh:66-67` defaults to the organizer-pinned
  `EigenLabs/Qwen3.8-27B-MTP-bf16`, which is the 15-tensor tree the fixture note
  describes. **The fixture is describing the pinned head; the manifest is
  describing ours.** Both are true, and the local/ranked gap is real.
- The exact-count gate was **deliberately relaxed** for declared heads:
  `Qwen36MTPHeadAttachment.verifyHeadIndex` (`Sources/MLXFastModel/…:315-325`)
  now requires only `weightMap.count >= 3`, a bare namespace, and
  `fc.weight` / `norm.weight` / `pre_fc_norm_hidden.weight`, with a comment
  stating that a declared head "may carry a different count — e.g. a quantized
  head's weight/scales/biases triples". `qwenMTPHeadTensorCount = 15` survives
  only in an error string and in tests, **not as a gate on our head**.

### The head is competitive surface, and it carries a draft-only projection slot

This is the most under-exploited structural fact on the board, and it was found
by chasing the contradiction above.

`README.md:245` states the licence plainly: "A head only *proposes* — the pinned
target still decides every emitted token — **which is why this can be yours**."
So head-side numerics cannot break bit-exactness by construction; they can only
move **acceptance**. That collapses the risk profile of every head-side idea from
"might be disqualifying" to "might not pay".

The vendored model exposes a dedicated slot for this
(`Vendor/mlx-swift-lm/…/Qwen35.swift:2038-2049`): the declared head tree may ship
`draft_lm_head.{weight,scales,biases}`, merged under `mtp.` and intercepted in
`sanitize` (`:2135-2154`) — "a coarser affine copy of the exact lm_head used
exclusively to argmax DRAFT proposals … every ledger/verify value still comes
from the exact `lmHead`. Plain stored arrays, deliberately not Module
parameters."

Current state of that machinery:

- With **no** declared draft head, the model derives `_compactDraftHead`, an
  input-independent compact copy of the exact `lm_head` trimmed to
  **98,336 padded / 98,330 real rows** out of vocab 248,320, and selects through
  a fused one-dispatch kernel `qwen35DraftSelectKernel` (`:2361-2387`) instead of
  six dispatches. This is the promoted configuration.
- Deriving hidden size from our corrected readout: 283.2 MB / 98,336 rows =
  2880 B/row; at 4-bit g64 that is `H·0.5 + (H/64)·4` ⇒ **H = 5120**. Sanity
  check: a **full-vocab** 4-bit draft head reads 248,320 × 2880 ≈ **715 MB**,
  i.e. 2.5× the compact read.
- **A declared draft head is full-vocabulary today and *disables* the compact
  fused path** — `draftTokenID` guards on `_draftHeadW == nil` (`:2362-2366`) and
  `usesCompactDraftVocabulary` requires `_draftHeadW == nil` (`:2401-2404`). So
  naively declaring a `draft_lm_head` is a **2.5× readout regression**, not a win.
  Anyone proposing one must change that code path too — and
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` **is** editable
  surface (`docs/qwen-mtp-editable-surface.md:50`), so that is allowed.

**Pre-registered negative, already paid for — do not re-propose.** A 49,152-row
halving of the compact prefix was measured on the public longcopy gate and
regressed: three committed argmax ids live in `[49,152, 248,044)`, the head could
no longer propose them, and the forced rejects cost more than the halved read
saved — **acceptance 1.00 → 0.877, 21.1 → 22.8 ms/token** (`:2054-2059`).

---

## Current research focus and themes

### Theme A — the depth policy is mis-specified, and this is the biggest known win

The shipped rule assumes an affine cost `C(d) = C₀(1 + d·h)` with a hardcoded
`h = 0.20`, and additionally caps depth at 4 behind an SDPA width wall. Both
assumptions are wrong, and **they only pay off together**:

| fix | q=0.95 | q=0.90 | q=0.85 |
|---|---:|---:|---:|
| remove width wall only | +3.52% | +2.14% | +0.93% |
| measured cost curve only | +1.89% | +0.89% | +0.68% |
| **both** | **+7.52%** | **+5.83%** | **+2.75%** |
| oracle best fixed depth | +7.70% | +6.18% | +3.36% |

The prize is a **13× function of the shape** of `m(d)`: holding the endpoints
`C(0)=67.0 ms` and `C(8)=161.0 ms` fixed and varying only the shape between them
moves the joint gain at q=0.94 from **+0.58%** (a cliff at d=5) to **+7.54%** (a
knee at d=7). Measuring that shape is the entire point of PR #1.

Caveat that shapes the design: removing the wall is **negative at q ≤ 0.75**.
Because the score is a median of 8, the correct artifact is a measured-marginal
rule **plus** an EMA-driven safety cap, not an uncapped rule.

Literature status, settled this round: this form is **not novel** — Sequoia
(NeurIPS 2024), D-cut, DSpark, ECHO, SMART, Yggdrasil and Su et al. (2023) all
publish measured non-affine draft-cost models. We adopt Sequoia's
`G(n,d)/(t(n)+d·c)` and D-cut's profile-at-startup/read-at-runtime mechanic. The
residual novelty is the **setting**: every published counterexample locates the
knee in *batched* verification crossing compute saturation; ours is at **B=1**,
from MLX kernel granularity and the GDN-vs-full-attention width regimes. No
Apple-Silicon/MLX instance and no native-MTP-fixed-target instance exists.

### Theme B — drafting bandwidth is now dominated by the readout, not the head

On the ranked arm, per draft step: 4-bit head 238.9 MB + compact 98,336-row draft
readout **283.2 MB** = 522.1 MB. The readout is **54.2% of all drafting
bandwidth** and is completely untouched by head quantization. This is the
clearest new lever of the round and we have never attacked it. (The static prefix
trim we already rejected — halving to 49,152 regressed acceptance 1.00 → 0.877 —
is a *different* mechanism from clustered or low-rank two-stage readout.)

### Theme C — the residual above roofline is real but smaller than I claimed

A dispatch census reconciles a d=8 round to 23.5 GB of traffic ≈ 122.8 ms at the
**measured** 70% of peak, against a measured 161.0 ms. Local residual ≈ 38 ms.
Leading hypothesis: the head runs at worse than 70% efficiency (15–25 ms).
Command-buffer commits and per-dispatch gaps are **ruled out**. Copy traffic is
0.06 ms. PR #4 is measuring the decomposition directly.

### Partial-acceptance repair — three regimes, and a counter that conflates them

Traced in source on the live base after `b219009`. This underwrites the
`headStepCostRatio = 0.20` argument, so it is load-bearing for PRs #1 and #4.

**The repair path is a two-tier try/fallback**, not the single expensive path the
file header still describes:

- `Qwen36MTPBlockSession.swift:973` tries `restoreAfterPrefixReject` (impl `:1146`).
  Success = trim attention caches + reconstruct recurrent state. **No target forward.**
- On `false`, the `else` at `:988` runs `rollbackAfterVerify` plus a full
  `model.callWithHidden` re-forward of the committed block (`:993-997`) — a real
  repair forward, and by the code's own comment at `:990-992` **a second blocking
  eval for that round**.

**Three regimes by verify width `S`:**

| `S` | mechanism | cost |
|---|---|---|
| `S >= 3` (draft ≥ 2) | compact `prefixReplayTape`, gated `nConfirmed == 1 && S >= 3 && mask == nil` (`Qwen35.swift:977`, written `:1112`, replayed `:889` ← `:1899` ← `:1146`) | cheap replay |
| `S == 2` (K=1) | single-launch mid-kernel emits the timestep-0 state as a third output (`Qwen35.swift:990-1000`) — checkpoint is **free** | ~0 |
| otherwise / guard failure | eager-checkpoint kernel, or the full re-forward fallback | expensive |

`restoreAfterPrefixReject` returns `false` when: a cache offset ≠
`committedOffset + rejected`; a non-trimmable non-`ArraysCache` entry exists;
`canReplayPrefix` fails; the tape is nil; or at K=1
`rollbackCheckpoints.count <= acceptedCount`.

**Consequence for the cost model.** The justification at `:519-527` for
`headStepCostRatio = 0.20` over MTPLX's 0.43 is that "this stack's per-row GDN
checkpoints make a prefix reject nearly free … **no repair at any depth**". That
premise is **real but conditional** — it holds only while the guards above hold.
It is unmeasured. If it fails even occasionally, `C(d)` acquires a term the cost
model does not price, and 0.20 is underpriced *independently* of any curve fit.

**Consequence for instrumentation.** `rollbackRoundCount` (`:962`) increments
**before** the branch, so it conflates a ~0 ms replay with a ~25 ms re-forward
plus second blocking eval. A single value cannot answer "does partial rejection
fire a second full 48-layer GDN recurrence?" Both PR #1 and PR #4 have been asked
for the same split — `prefixRepairCount` / `fullRepairCount`, shared naming.

**Two corrections to earlier internal analysis, recorded so they are not
re-derived wrongly:**

- The vendored DFlash `RecurrentRollbackCache.recordTape` is **genuinely dead**
  (repo-wide, the only hits are a comment and a test comment). The header
  paragraph "WHY NOT THE VENDORED DFLASH ROLLBACK" and the rollback contract
  tests are **correct, not stale**. The live tape is a *different*, Qwen35-native
  mechanism. What *is* stale is step 5 of the header round-loop summary
  (`~:29-34`), which presents the full re-forward as the only partial-acceptance
  path.
- There is **no `S>=3` → `S>=2` gating win**. `S == 2` already gets its checkpoint
  free as a third kernel output, so no eager-checkpoint tax exists there to remove.

### Theme D — closed this round, do not reopen

- **Prefill cutting** (PR #3, merged): `P = 4.0086 s` is irreducible; quantized
  GEMM is 97.0% of it at 87.1% of the measured dense bf16 ceiling. Free dequant
  would buy 12.49%, under the 20% stop rule.
- **STree / A_tree rollback-free GDN verification**: the GDN update is
  rank-1-perturbed affine, not diagonal; run-fatal under zero-tolerance token
  identity; and it has **zero** bandwidth benefit. Its one useful trick (replay
  instead of snapshot) is already shipped as `PrefixReplayTape`.
- **The `d == 0` absorbing state**: 400 seeds × 512 rounds show it **never fires
  at q ≥ 0.85**. Tail insurance only. (Retracts "botany freezes by round ~51".)
- **Head quantization**: already banked via `mtp-head.manifest.json`.
- **An 8-bit head**: would be **1.6–1.7× slower** than the 4-bit head we already
  have. Six measured Q8→Q4 points agree.
- **Layer-skipping self-drafting**: α ≈ 0.038 for sequential hybrids versus 0.68
  for parallel. Keep the trained MTP head.

---

## Live experiment slots

All four students are occupied. No slot is free.

| PR | Student | Question | Status |
|---|---|---|---|
| #1 | qwen-edward | Measure `C(d)` for d=0..8; ship the generalized table rule with `H` measured at run time; width-9→10 padding probe; per-position acceptance table | r2 issued on the post-`b219009` base |
| #2 | qwen-alphonse | Part A: width-9 bit-exactness (blocking). Part B: `max/p50` block latency per arm **at both head sizes**, plus a low-acceptance arm | r2 issued on the post-`b219009` base |
| #4 | qwen-askeladd | Three-number floor decomposition; head chain timed in isolation; `rollbackRoundCount` | r2 issued on the post-`b219009` base |
| #5 | qwen-thorfinn | qmv small-M kernel curve M=1..512; normalized `qmv_tax` stop rule; GDN-vs-projections knees side by side; `qmv_fast` K-alignment audit | r2 issued on the post-`b219009` base |
| #3 | qwen-thorfinn | Seed-prefill Amdahl term | **merged** — *not useful* for the mechanism, decisive for the ceiling |

Round-1 revisions were all cut from bases older than `b219009` and therefore
carried the EOS defect. Round 2 re-binds each assignment to the live advisor-branch
head, so **no in-flight experiment is measuring on a defective base any more**.
PRs #1 and #2 are the pair that only pays off jointly (Theme A); PRs #4 and #5
jointly settle Theme C.

### Process finding — round 1 produced zero student pushes

Between assignment creation (14:41–15:57 UTC) and the round-1 close (17:24 UTC),
all four PRs stayed at their creation head SHA with **no commits and no student
comments**, and then all four students went idle at once. Nine, eight, eight and
four advisor feedback comments were delivered into that silence. Two lessons are
now standing policy for this campaign:

1. **Cheapest decisive artifact first.** Every assignment must name one artifact
   that is producible in a single short session (a Python microbenchmark, a
   bit-exactness check, a breakdown table) *before* any end-to-end A/B. A student
   who runs out of budget must still leave durable evidence behind.
2. **Silence is not a status.** A student that cannot run — setup failure, weights
   missing, lock contention, thermal gate, wall-clock limit — must post a PR
   comment naming the exact blocking command and its output. Going idle with an
   empty branch is a protocol failure, not a null result.

Advisor-side lesson: feedback volume is not progress. Round 2 leads with a short
ordered instruction, not more analysis.

---

## Potential next research directions

Ordered by expected value. Items marked ★ are new or newly elevated this round.

1. **★★★★★ Compact draft-readout reduction.** 283.2 MB, 54.2% of ranked drafting
   bandwidth, untouched by head quantization. Gemma 4 reduces the projection from
   ×262,000 to ×4096 via top-k over token clusters "while preserving a similar
   acceptance rate"; SlimSpec reports ~4–5× LM-head latency reduction where
   VocabTrim-class methods reach only ~60%. Rough sizing: ~14% of the ranked
   low-band marginal `m_lo`. **Strongest unassigned idea we have.**
   **Strengthened by item 17, now settled:** the in-code "~0.6 ms" note that made
   this look not worth attacking is unreachable by 1.7–1.9× on this host. The real
   readout is 9.98 ms/round (~6.2% of the round), so the trim ceiling is twice the
   code comment's claim. The only surviving objection to readout reduction is
   *acceptance*, not bandwidth — so any assignment here must gate on acceptance
   from the first measurement, not on bytes saved.
   **Prior art (external, and it names the mechanism): FR-Spec / VocabTrim.**
   Frequency-ranked draft-vocabulary trimming reports ~75% LM-head reduction and
   is **exactness-preserving by construction**, because verification stays
   full-vocab — only the *proposal* distribution narrows. Reported optimal subset
   is ≈32K tokens; we currently draft over 98,304. This is the same idea, already
   validated elsewhere.
   **Design constraint that must go in the brief: prefer a STATIC CONTIGUOUS
   pre-trimmed head over a gathered/dynamic one.** A gather can be *slower* than
   the untrimmed read despite moving fewer bytes, because gathered rows cut across
   4-bit g64 quantization groups and destroy the contiguous-group access the
   `qdot` path depends on. Trimming bytes is not the same as trimming time here.
   **Not refuted by the earlier static-prefix result.** The 49,152 halving
   regressed acceptance 1.00 → 0.877; that is an *acceptance* refutation of a
   naive prefix cut and it stands. Frequency-ranking is a different selection
   rule, and the bandwidth objection that used to sit alongside it is now dead.

   **1b. ★★★★★ NEW — cut readout PRECISION instead of readout ROWS.** The single
   best idea to come out of this round, and it is on a different axis from every
   trim result above. Every refutation we have is about *deleting rows*: a deleted
   row is a **guaranteed** reject whenever it is the answer, which is why 1.00 →
   0.877 happened on only three ids. Lowering the *precision* of the draft
   projection degrades acceptance **gracefully** instead — a slightly-wrong logit
   only changes the argmax near a tie, and the exact target still decides the
   token, so bit-exactness is untouched either way.

   The delivery vehicle already exists and is sanctioned: ship
   `draft_lm_head.{weight,scales,biases}` in our declared head artifact (see "The
   head is competitive surface" above). Head bytes live under the separate 2 GiB
   cap, **outside** the 2,396,110 / 3,000,000 source budget.

   Sizing at the derived H = 5120, on the compact 98,336-row set:

   | draft head | bytes/row | readout | Δ vs today |
   |---|---|---|---|
   | 4-bit g64 (today) | 2880 | 283.2 MB | — |
   | 3-bit g64 | 2240 | 220.3 MB | −62.9 MB ≈ −0.28 ms |
   | 2-bit g64 | 1600 | 157.3 MB | −125.9 MB ≈ **−0.55 ms** |

   at the measured 227 GB/s. Per **round** (readout 9.98 ms) that is ≈−1.1 ms at
   2-bit, i.e. ~5.5% of drafting — and over a 512-token leg it plausibly reaches
   the few-hundred-millisecond scale that actually moves score (100 ms ≈ +0.043).
   That makes it one of very few unassigned ideas with a credible path to the
   ≈220 ms we need.

   **Blocker to state in any brief:** a declared draft head currently forces the
   full-vocab path and disables the fused compact kernel, which alone is a 2.5×
   readout *regression*. The experiment is therefore **compact-vocab AND
   low-precision together**, which requires editing `usesCompactDraftVocabulary` /
   `draftTokenID` in the vendored `Qwen35.swift` so a declared head can keep the
   compact bounds. Open risks to check before assigning: whether MLX's affine
   2-bit path and the `qmv_fast` / `qwen35DraftSelectKernel` shapes support 2-bit
   at the `N % 8` padding contract, and whether 2-bit is too coarse to hold
   acceptance (3-bit is the fallback rung, still −0.28 ms). Gate on **acceptance
   from the first measurement**, exactly as in 1 above.

2. **★★ Composition round.** Once ≥2 of PRs #1/#2/#4/#5 land, compose them and
   re-measure on a fresh base. Elevated because #1 and #2 are individually
   near-worthless and jointly worth +5.8% at q=0.90.

3. **★★ Replicate the break-even acceptance curve α(k) on our hardware.**
   `2604.16368` (Bielik 11B, MLX-LM, M2 Pro) reports break-even acceptance of
   ≈40% at k=2, ≈77% at k=4, and **>100% at k=6** — i.e. depth ≥6 would be
   unrecoverable. Single-author preprint, unverified, and it directly contradicts
   our main theme. Cheapest possible falsification of that theme.

4. **★★ Per-layer-family knees.** GDN decode is ≈2 FLOP/byte (knee ≫ 8) while
   4-bit projections are ≈7.9 (knee ≈ 7.9). If so, a **single scalar depth policy
   is mis-specified for the model as a whole**. No literature exists on per-family
   knees in a hybrid recurrent/attention model. Partly folded into PR #5.

5. **★★ Head-precision A/B (bf16 vs 4-bit), gated behind a free offline
   pre-check.** Expected acceptance cost −1% to −3%, credible tail to −8%. Run the
   **untimed, outside-the-scored-path** KL / top-1-agreement comparison on the
   public fixture first: ≥99% top-1 agreement → skip the A/B; ≤96% → spend the
   slot. Must log **per-position** a₁…a_d and **jointly re-tune depth**. Note that
   "both sides 4-bit" buys us nothing: every high-acceptance quantized-draft
   result rests on the draft being a quantized *copy of* the target so errors
   cancel — **our head is independently trained, so errors add**.

6. **★★ Align local measurement to the ranked head** via
   `MLXFAST_QWEN_MTP_HEAD_DIR`, and mandate `head_provenance_sha256` in every
   result. Partially issued as feedback; deserves to be made structural.

7. **Re-fit the per-position acceptance prior.** `positionAcceptEMA` is
   initialized `0.85 · 0.98^i` and never reset per prompt. Two independent
   measurements condemn that shape: GLM-4.5-Air — whose released weights contain
   only the first MTP module *"reused autoregressively"*, exactly our architecture
   — measures **0.92 → 0.68 → 0.38**, and Nemotron 3 Super reports monotonic decay
   with draft index. Cheap, and it feeds Theme A directly. **Note `b219009`
   slightly changed this function's semantics** (the `stoppedEarly` suppression is
   gone), so re-read it before fitting.

8. **Move the 511-row head priming out of the first timed drafting round.**
   `headHistoryCache` is lazily primed inside the first scored round.

9. **Representative local prose goldens.** The eight ranked prompts are
   public-domain classics; `generate-golden --prompt-file` plus
   `MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE` lets us build honest same-genre local
   seeds, stop over-fitting to one fixture, and — now relevant — obtain
   trajectories that do not all hit EOS at the same token.

10. **Zero-GPU policy simulator.** `Qwen36MTPTarget` is an `AnyObject` protocol,
    so a research-only stub can drive the real `costModelDepth` and
    `recordAcceptOutcome` with no GPU at all. Done ad-hoc advisor-side; worth
    productionizing so students can pre-screen schedules for free.

11. **★ Assert `K % 512 == 0` on every scored 4-bit reduction dim.**
    `qmv_fast_k_alignment` silently drops 4-bit shapes that are 256- but not
    512-aligned into the bounds-checked generic kernel. All current dims pass by
    inspection; the assertion is nearly free and the failure mode is silent.
    Folded into PR #5.

12. **Cleanup PR (owed).** Stale `Qwen36MTPBlockSession.swift:22-43` header, stale
    rollback-contract test framing, the never-executed
    `Sources/MLXFastModel/Qwen35*.swift` family, the dead `:446-449` policy
    closure, the dead `conf` gate, the unconditional `convInput` at
    `Qwen35.swift:758` (96 wasted dispatches per verify), the stale
    `mtp-head/README.md` "pinned" claim, and the vestigial `Constants.swift:311`.
    **Now also: the residual `reachedStopToken` compatibility shim and the ignored
    `stopTokens` init parameter left by `b219009`**, once no caller needs them.
    **Deletion is the default.** Assign to the next free slot.

13. **Repair-path telemetry (`rollbackRoundCount`)** — needed to settle census
    hypothesis 2 (does partial rejection fire a second full 48-layer GDN
    recurrence?). Requested inside PR #4.

14. **Extend the replay tape to S=2** and delete the eager 144 MiB mid-state
    write that K=1 rejection currently pays unconditionally.

15. **Verify MLX's `qmv` reuses dequantized weights across rows.** A Snapdragon
    roofline sweep shows 0.52× marginal cost even at M ≤ 8. If MLX does not reuse,
    the knee is unreachable in software and the fix is a kernel change.

16. **Off-diagonal `(d, w)` identification of `H`** via the `d=8, w=10` point.

17. **SETTLED (advisor, source + arithmetic, no timing run needed) — the in-code
    compact-draft-vocab note is wrong on both numbers.** The note at
    `Qwen35.swift:2058-2060` claims "~315 MB of affine-4 rows per draft step
    (~0.6 ms)". Both halves fail:

    - **Bytes.** `makeCompactDraftHead` (`:2406-2434`) inherits `groupSize`/`bits`
      from the loaded `lmHead`; it does not choose its own. This checkpoint family
      is affine 4-bit **group-64** (`hf:lowskillcoding/qwen38-mtp-head-4bit-g64`;
      `groupSize: 64` hardcoded on the declared-draft-head path at `:2342`). At
      98,336 × 5120, g64 gives 251.7 MB weights + 15.7 MB scales + 15.7 MB biases
      = **283,207,680 B = 283.2 MB**. The note's 315 MB is exactly the **g32**
      arithmetic (314,675,200 B) — it assumed the wrong group size.
    - **Time.** 0.6 ms for that read implies **472 GB/s** (g64) or 524 GB/s (g32).
      The M4 Pro's *theoretical peak* is 273 GB/s and its measured decode
      bandwidth is 227 GB/s. So ~0.6 ms is not merely optimistic, it is
      **unreachable by 1.7–1.9×** on this host class under either group size. The
      floor is **1.25 ms measured / 1.04 ms at theoretical peak**.

    **Consequence — the note's conclusion inverts.** It reasons "the read is
    ~0.6 ms, so the ceiling of any further trim is small". The true per-round
    readout is 8 × 283.2 MB = 2.27 GB = **9.98 ms at 227 GB/s, not ~4.8 ms** — a
    +5.2 ms/round correction. That is **6.2% of the ~161 ms *local* (bf16-head)
    round and ~7.2% of the ~139 ms *ranked* (4-bit-head) round** — always state
    which arm, since the readout fraction rises as the head shrinks. The
    trim ceiling is **twice** what the code comment asserts. This does not revive
    the *static prefix trim* (halving to 49,152 genuinely regressed acceptance
    1.00 → 0.877 — that refutation is about acceptance, not about bytes, and it
    stands), but it removes the stated bandwidth objection to **clustered /
    low-rank two-stage readout**, which is item 1.

    Residual uncertainty is one line: confirm the loaded `lmHead` is
    `QuantizedLinear(bits: 4, groupSize: 64)` rather than bf16. If it is bf16 the
    compact head is 1.007 GB and the error is ~7×, not ~2× — i.e. every branch of
    this check makes item 1 stronger, none weakens it. Anyone timing it should
    report **achieved GB/s**, not ms; ms alone is not host-portable.

18. **Compiled MTP round** — extend `CompiledDecode.swift` past its B=1
    solo-decode gate to cover GDN and MTP.

19. **Tree drafting sized by measured `b(n)`**, and per-position
    margin-conditioned acceptance fitted at all positions.

20. **GQA query-head pairing** to break the `qL*gqa <= 32` fused-path limit.

21. **ReDrafter / Hydra head restructuring** — hold until a genuine plateau.

22. **Thermal/scheduler-variance investigation** for the fixture's depth-2
    3.30–3.36× max/p50 spread.

23. **Resolve the `da336ce9…` head digest** recorded in PR #3, which matches
    neither the declared nor the computed pinned tree digest. Needs a `shasum` on
    a student host; the advisor host has no model cache.

---

## External literature sweep — round 2

**Provenance caveat, applies to everything below.** Roughly one third of the
sources in this sweep are unverified preprints carrying 2026 dates and were not
independently confirmed. Treat every constant as a *direction*, never as a value.
All of it was derived on non-Apple hardware or on other Apple chips; anything we
intend to rely on must be re-derived on our own host. This campaign has already
had to retract two externally-sourced claims (`mlx#3920`, the `mlx-lm#250`
explanation), so the bar is: an external number may motivate an experiment, it may
never conclude one.

- **FR-Spec / VocabTrim** — folded into direction 1 above. The highest-value item
  in the sweep, because it independently validates our strongest unassigned idea
  *and* supplies the static-contiguous design warning.
- **Pre-registered negative — do NOT assign.** Drafting by skipping the 16
  full-attention layers fails on Qwen-family sequential hybrids: α ≈ 0.038 versus
  0.68 for parallel hybrids. Already recorded under Theme D; repeated here so the
  next person sweeping literature does not re-propose it.
- **Adaptive draft length** (AdaEDL / SVIP / SpecDec++): stop drafting on an
  entropy or margin threshold rather than a fixed depth; threshold-optimality has
  a proof sketch. Relevant to Theme A, and complementary to a re-fit `C(d)` — a
  cost curve sets the *budget*, a confidence signal spends it per-round.
  **Novel open question worth owning:** for a *greedy-exact* verifier like ours,
  the `top1 − top2` margin should dominate entropy as the stop signal, because
  acceptance is decided by an argmax tie, not by distributional spread. No paper
  in the sweep isolates this. We already carry exact top-two evidence on every
  row, so the signal is **free** for us — unusually cheap novelty.
- **SpecInfer**: at batch size 1–2, wider trees consistently reduce latency. We
  run BS=1. Tension with our hard depth cap of 4 / structural width-5 wall, so
  this is gated behind PR #2's width-9 bit-exactness result.
- **HyperDFlash**: native MTP holds acceptance only at positions 1–2, decaying
  after. Consistent with our own `effective depth 1` on all 48 scored prose runs.
- **Draft&Verify**: below ~80% acceptance, K=1 is optimal. Our ranked prose regime
  is near that boundary, which is why the depth policy matters at all.
- **ReDrafter on MLX**: 1.37× on M1 Max → 2.3× on M2 Ultra. A same-method,
  cross-host spread of 1.7× — direct support for Flag B/C and for the
  bandwidth-scaling model over the fixed-host-cost model. Still the **only
  Apple-Silicon datapoint in either sweep.**

### Round-2 additions, ranked by decision relevance

- **★ The W4 widening penalty — SpecMQuant (arXiv 2505.22179).** On **W4A16**,
  the verify-to-decode time ratio reaches **1.8** at tree size 60, versus **<1.2**
  for FP16 and W8A8; the paper attributes EAGLE-2's weak 4-bit showing to exactly
  this. Mechanism: widening converts a memory-bound decode into a compute-bound
  one, which destroys the advantage 4-bit weights were bought for. **This
  independently predicts our own measured super-linear `eval_wall` of 79 → 89 →
  106 ms for widths 7 → 8 → 9.** We are a 4-bit deployment, so we sit squarely in
  the penalised regime. The paper's own remedy is to convert tree drafts into
  *sequence* drafts (2.78× on 4-bit Llama-3-70B) — an alternative worth holding in
  reserve if width is confirmed dead. **Bears directly on PR #2 (width-9) and on
  PR #1's `C(d)`.**
- **★ OPT-Tree (TACL 2025, doi 10.1162/tacl_a_00735)** — deepen only while the
  marginal gain in expected accepted length exceeds **μ = (drafting step time) /
  (decoding step time)**, with threshold δ ∈ (μ, 1). Reported best δ is **0.2 with
  a standalone drafter but 0.8 with an EAGLE-style head.** Our
  `headStepCostRatio = 0.20` is sitting in the *standalone-drafter* regime while
  we actually run an MTP head — **independent external corroboration of Theme A**,
  arrived at from a completely different direction than our own cost algebra.
- **★ LK Losses (arXiv 2602.23881)** — released MTP modules typically ship **only
  the first MTP module**, trained to predict the first next token but then reused
  autoregressively for deeper positions, producing a **sharp acceptance decline at
  later positions**. This gives a *structural* explanation for our `effective
  depth 1` on all 48 scored prose runs: it is a property of the released
  checkpoint, not a tuning failure, and no depth policy can recover it.
  Consistent with HyperDFlash above and with Draft&Verify's K=1 result.
- **Leviathan (arXiv 2211.17192)** — the closed form we should be quoting:
  `E[tokens/round] = (1 − α^{γ+1}) / (1 − α)`, and speedup exists **iff α > c**.
  Useful as the sanity check on any proposed depth change.
- **Trees from Marginals (arXiv 2607.06763)** — rollback-free tree verification
  for Gated DeltaNet via a masked triangular solve; claims 4.37× on Qwen3.6 27B,
  i.e. our architecture family. **Deliberately NOT pursued**, and the reasons are
  already established under Theme D: the GDN update is a rank-1-perturbed affine,
  not diagonal; a reformulated solve changes reduction order and is run-fatal
  under zero-tolerance token identity; and it carries **zero bandwidth benefit**
  for us. This round adds a fourth reason — our partial-acceptance repair is
  already cheap in its common case, so the problem it solves is largely not our
  problem. Recorded so nobody re-proposes it on the strength of the headline
  number and the matching model name.
- **STree (2505.14969) / Mamba-in-Llama (2408.15237) / SpecMamba (2509.19873)** —
  useful only as a taxonomy of rollback strategies for recurrent state: snapshot,
  activation-replay, and rollback-free. Our implementation already spans the first
  two; see the repair-regime section.
- **Goose (arXiv 2604.02047)** — batch-1 and greedy, so unusually close to our
  setting. The transferable parts are the **1/i harmonic branch-width schedule**
  (narrow as depth grows rather than a rectangular tree) and **harvesting logits
  from rejected branches and from prefill** rather than discarding them.
- **SpecInfer / adaptive-draft-length / FR-Spec** — see above; unchanged.

## Standing policy — the bit-exactness hazard list

Our track scores under **zero-tolerance token identity**. The following are
recurring, respectable, well-cited techniques that are nevertheless **disqualifying
or banned here**. This list exists because most speculative-decoding literature is
written for a *distribution*-preserving standard, and the vocabulary is a trap.

1. **Medusa typical acceptance** — accepts tokens the target would not have
   emitted. It is the source of most of Medusa's headline gain. Disqualifying.
2. **Any relaxed, entropy-gated, or multiplicative acceptance certificate.**
   Same failure, different dress.
3. **Standard speculative *sampling* rejection** (Leviathan/Chen-style) —
   preserves the output *distribution*, **not the token stream**. **Read every
   "lossless" claim in this field as distribution-lossless unless the paper
   explicitly says otherwise; only the greedy / T=0 configuration transfers to
   us.** This single misreading would invalidate an entire experiment.
4. **Quantized verification** (e.g. Quasar, 2603.01399) — changes the target's
   answer. Disqualifying. Note the asymmetry: QSpec and ML-SpecQD are safe *only
   because they lower the **drafter's** precision alone* — which is exactly the
   licence direction 1b relies on.
5. **Bigram / n-gram / suffix-automaton / prompt-lookup drafting** — banned by
   program rules independently of exactness. Do not propose these.
6. **Cross-request caching** — banned. Per-request reuse, and input-independent
   shape/kernel tables, are fine.
7. **Numerical-reformulation risk** — triangular solves, `A_tree` formulations,
   and chunkwise-form changes alter reduction order. **Matching one argmax is not
   sufficient evidence**, because the trusted parent checks exact top-two row
   evidence. Any such change needs the full parity gate, not a spot check.

## Operating reminders

- Local `--local-iterate` runs **both** legs from the same candidate build, so a
  general target/kernel win cancels in the local ratio while scoring fully on the
  ranked board. Always report **absolute candidate seconds per token** against a
  fresh unchanged `BASE_SHA` run, not just the ratio.
- Headline numbers come from a **512-token** window on a base at or after
  `b219009`. 64 tokens is an inner-loop screen; 256 tokens is a labelled
  directional screen.
- `MLX_QWEN_MTP_TRACE=1` is **unreachable** on the scored path — worker stderr is
  discarded unless `MLX_DFLASH_TRACE_CACHE_SEAM=1`. Use parent-clock algebra
  (`decode_seconds = P + Σ block_request_seconds + N·c`), the campaign-standard
  method since PR #3.
- The stall guardrail **fails closed** and excludes the first block. Uniform
  steady-state speedup is neutral to it; the hazard is occasional expensive
  after-first rounds.
- `Sources/MLXFastModel/Qwen35*.swift` is editable but **never executed**
  (`Qwen35FastPathReadiness.swift:11-19` hardcodes false). Prove the live call
  path before optimizing. Exact live paths — note the two prefixes differ, which
  is easy to get wrong:
  - `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` (GDN, replay tape,
    compact draft head)
  - `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (round loop, repair,
    cost model) — `Qwen36*` under `Sources/` is live; only `Qwen35*` there is dead.
    There is **no** `Vendor/` copy of the block session.
- **Every line number in this document and in student briefs post-dates
  `b219009`, which shifted them all.** Re-locate by symbol name, always. The
  in-file header comments have already been caught stale twice; source is the
  authority, comments are not.
- Editable budget: `source = 2,396,110 / 3,000,000`, headroom 603,890 B.
  `mtp-head/` is exempt with its own 2 GiB cap. Preflight every assignment with
  `senpai/validate-assignment-scope.sh` and `senpai/check-editable-budget.sh`.

## ★★★ Verify width is a STAIRCASE, not a roofline knee — the base already ships a crossrow QMV

This is the largest correction of round 2 and it changes how PR #2 and PR #5 must
be read. Traced end to end in source on the live base.

**The frozen host launches one threadgroup per input row.**
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:235-295` — `qmv`
sets `bn = 8`, `bk = 32`, `group_dims(bk, 2, 1)` and, at `:254`,
`MTL::Size grid_dims(M, (N + bn - 1) / bn, B)`. **`M` is the x grid dimension**, so
each of the `M` verify rows is an independent threadgroup. Threadgroups share
nothing, so the naive reading is that weights are streamed and dequantized `M`
times. `:259` sets `fast = N % bn == 0 && K % 512 == 0`, selecting `qmv_fast`.

**But a prior accepted submission already fixed this, inside the kernel.**
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` (editable surface)
carries `qmv_fast_crossrow_affine4_g64{,_wide,_m}`. The design note at `:973-980`
states the contract exactly: *"the frozen host launches M x-groups for each
8-output tile, so a group that claims NA adjacent input rows lets the remaining
host groups return without reading weights."* The wrapper at `:1067-1094` does
`first_m = tid.x * IPG; if (first_m >= M) return;` — surplus groups exit before
touching weights. Live gate at `:1817-1860`: `!batched && group_size == 64 &&
bits == 4 && out_vec_size >= 1024`, with the `_wide`/`_m` family above 4096.
It arrived progressively across the validated submissions `b6c7251` →
`08897af` → `1033e1a`, so **it is already inside the promoted 2.9042 frontier.**

**The cost law.** `:1064-1065` states it outright:
`IPG = ceil(M / ceil(M / 4))`, *"the fewest weight streams reachable at NA <= 4,
with the remainder spread evenly so no group runs a one-row tail."* Active groups
= `ceil(M / IPG)` = **`ceil(M / 4)` weight streams**. Verified against the
dispatch table `<3,3> <4,4> <5,3> <6,3> <7,4> <8,4> <9,3>` — computed IPG matches
every entry.

| verify width M | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| weight streams `ceil(M/4)` | 1 | 1 | 1 | **2** | 2 | 2 | 2 | **3** |

**This re-explains our own headline measurement.** `eval_wall` 79 → 89 → 106 ms at
widths 7 → 8 → 9 (`ESTABLISHED_FACTS.md:393`, corroborated by the phase-trace note
at `Qwen36MTPBlockSession.swift:512`): 7→8 stays at 2 streams and costs +10 ms;
8→9 crosses 2→3 streams and costs +17 ms. The accelerating delta is a **tiling
step**, not the roofline knee at `M* = 7.9`. It also explains why `V(9) ≈ 161 ms`
rather than the ~600 ms a true per-row re-read would give — 3 streams, not 9.

**`ESTABLISHED_FACTS.md:120-131` set up precisely this falsification** — *"if
different shapes knee at different M, then dispatch and occupancy — not roofline —
set the curve, and the whole model below is wrong in an informative way."* Source
now says dispatch and occupancy set the curve. The roofline `M*` is not deleted —
both terms are present — but the staircase dominates and the two models are
**cleanly separable at M = 5**, where the staircase predicts a jump (1→2 streams)
and roofline predicts business as usual deep in the bandwidth-bound regime.
**That is the discriminator, and PR #5's Python `mx.quantized_matmul` M=1..12
sweep measures it with no run lock and no GPU contention.**

**Concrete opportunity (unassigned, strong).** Width 9 costs 3 streams only
because `NA` is capped at 4 — `static_assert(NA >= 2 && NA <= 4)` at `:993`. At
`NA = 5`, width 9 would need `ceil(9/5) = 2` streams, plausibly recovering much of
the +17 ms at exactly the width PR #2's gate is about. Bit-exactness is safe *by
the kernel's own design contract* (`:977-980`: *"load_vector, the qdot expression,
the K accumulation order and simd_sum are unchanged for every output element"*).
**Known blocker:** `typedef vec<float, NA> VF` at `:994` — Metal has no
`vec<float,5>`; legal widths are 2/3/4/8/16. A 5-row group needs `vec<float,8>`
with 3 lanes wasted, or a restructure, and the note at `:975-978` warns register
footprint is the binding constraint. This is an experiment, not a certainty.

**Corrections to our own records.**
- `ESTABLISHED_FACTS.md:389-402` says widths 1-9 "all take `qmv`, tuned for
  `M = 1`". The **host dispatch** claim is right; "tuned for `M = 1`" is **wrong**
  for 4-bit g64 with `N >= 1024`, which is every scored projection.
- `ESTABLISHED_FACTS.md:749-750` justifies the alignment check by asserting
  "5120, 6144, 8704, 10240, 16480 are all multiples of 512". **16480 is not**
  (16480 = 512·32 + 96). **No live defect**: 16480 is an *output* dim (GDN fused
  in-projection 5120 × 16480, `:127`), and `N` only needs `% 8`. The real `K`
  dims — 5120, 6144, 8704, 10240, 17408 — are all 512-aligned, so the
  conclusion stands and only the stated reasoning was wrong. Still unasserted in
  code, still nearly free to assert.

**Corroborating external result, held at arm's length.** SpecMQuant
(arXiv 2505.22179) reports a W4A16-specific widening penalty — verify/decode ratio
1.8 at tree size 60 versus <1.2 for FP16/W8A8. Same *direction* as our staircase,
different *mechanism*, and their tree size 60 is far outside our 2..9 range, so it
motivates but cannot conclude. Its value here is that it independently warns the
4-bit path is the penalised one; our staircase says *why* on this stack.

## ★★★ MEASURED: the width cost law has TWO components, and `d* = 7`

PR #5 (qwen-thorfinn, **merged**) measured the isolated quantized-matmul cost
curve at the 8 exact scored shapes, `M = 1..9`, on **both** our vendored build
(crossrow live) and stock upstream MLX. PR #1 (qwen-edward) independently
measured in-situ per-depth marginal round cost. They agree, and together they
replace the pure-staircase model above.

### The law

> **~free up to M ≈ 3; then +0.17–0.32 of a width-1 call per additional row;
> plus a stream-boundary excess at M = 5 and M = 9.**

So the curve is **a linear per-row ramp switching on at M ≈ 4, PLUS the stream
boundaries** — not a pure staircase and not a roofline knee. Both prior models
are now superseded:

- **`ceil(M/4)` stream-correction magnitude: FALSIFIED.** Stream-corrected GB/s
  is not flat and exceeds the kernel's own M=1 achieved bandwidth by up to 22%.
  Marginal boundary cost is **0.02–0.26** of a full weight read, not the 1.0 the
  correction assumes — it over-corrects 4–50×. `implied_streams = c(1)/c(m)` is
  *continuous* (lm_head: 1.00 0.99 1.01 1.24 1.64 1.90 2.17 2.44 2.87), not the
  integer 1,1,1,1,2,2,2,2,3. **Boundary *location* was right; *magnitude* was wrong.**
- **Roofline knee at `M* = 7.9`: REFUTED.** No flat region out to 7.9; per-shape
  knee 7.16–7.80 but the plateau ends at M = 1–3.
- Staircase *location* confirmed **only on crossrow**: rank test (are M=5, M=9 the
  two largest increments over M=2..9?) passes 6/8 shapes vendored, **0/8 stock**
  (stock's largest steps are at M=7 and M=9). **M=5 is the true discriminator** —
  an M=9-only test false-positives on a no-crossrow build. Both vendored failures
  are the two N=5120 shapes.

### Two-method cross-validation (this is the strong part)

Edward's in-situ `h(d)` (marginal round cost ÷ `C(0)`), `C(0) = 67.0 ms`, minus
thorfinn's fitted head cost `H = 3.73 ms/step`, gives the implied verify step:

| d | M=d+1 | h(d) | marginal ms | implied ΔV | thorfinn's isolated curve |
|---|---|---|---|---|---|
| 1 | 2 | 0.0862 | 5.78 | 2.05 | ~0 (free) ✓ |
| 2 | 3 | 0.0795 | 5.33 | 1.60 | ~0 (free) ✓ |
| 3 | 4 | 0.2446 | 16.39 | 12.66 | 10.3–19.3 (ramp onset) ✓ |
| 4 | 5 | 0.3774 | 25.29 | 21.56 | ramp + boundary ✓ |
| 5 | 6 | 0.2939 | 19.69 | 15.96 | 10.3–19.3 ✓ |
| 6 | 7 | 0.3020 | 20.23 | 16.50 | 10.3–19.3 ✓ |
| 7 | 8 | 0.2890 | 19.36 | 15.63 | 10.3–19.3 ✓ |
| 8 | 9 | 0.3929 | 26.32 | 22.59 | ramp + boundary ✓ |

**This resolves the previously-unexplained `h(3) = 0.2446`** (a 3× rise with no
stream transition): it is the *onset of the linear ramp*, which is independent of
the stream boundaries. The shape is robust to error in `H` — changing `H` shifts
the whole ΔV column by a constant and cannot move the onset or the two bumps.

**Open quantitative discrepancy, do not smooth over.** Taking edward's interior
mean (M=6,7,8) as the ramp, the in-situ boundary excess is **+5.5 ms at M=5** and
**+6.6 ms at M=9** = **0.09–0.11** of a width-1 call, against thorfinn's isolated
**0.25**. Real and correctly located, but **~2.4× smaller in a live round than in
isolation**. Unexplained. Candidates: call-mix differences; boundary cost partly
overlapping other round work.

### `d* = 7` — the shipped cap of 8 is one step past the optimum

`C(d) = C(0)·(1 + Σh)`; cost per emitted token under per-position acceptance `q`
with `E = (1−q^{d+1})/(1−q)`:

| d | C(d) ms | ms/tok @ q=1.00 | @ q=0.976 | @ q=0.94 |
|---|---|---|---|---|
| 3 | 94.49 | 23.62 | 24.49 | **25.86** ← best |
| 4 | 119.78 | 23.96 | 25.13 | 27.01 |
| 5 | 139.47 | 23.25 | 24.68 | 26.98 |
| 6 | 159.70 | 22.81 | 24.51 | 27.26 |
| 7 | 179.06 | **22.38** ← best | **24.33** ← best | 27.52 |
| 8 | 205.39 | 22.82 | 25.10 | 28.86 |

**`d = 8` is dominated by `d = 7` at every acceptance level tested** (−1.9% at
q=1.0, −3.2% at q=0.976 — the rate askeladd measured — and −11.6% at q=0.94).
Thorfinn's independent fit `C(d) = V(d+1) + 4.46 + 3.73·d` (max resid 2.69 ms vs
11.15 stock) gives the same answer: **best d = 7 @ q=1.0, d=5–6 @ q=0.94, d=3 @
q=0.90.** The curve is **non-monotone** — d=3 beats d=4 at every q — confirming
the "prefer tread tops" policy prediction. Three independent routes converge.

⇒ **`segmentedVerifyDepthCap = 8 → 7` is a one-constant candidate win** with
strictly narrower widths, zero fidelity risk and zero budget cost. Assigned to
qwen-alphonse (PR #2) as a first-class arm, *not* a fallback. **Caveat: `C(d)` is
reconstructed across mixed provenance and the acceptance model is
position-independent, which `positionAcceptEMA` exists because it is not. This is
a lead that justifies an arm, not a result.**

### Fidelity: widths 1..9 are bitwise-safe (a correctness win)

Vendored crossrow is **bitwise-identical to the M=1 result on 8/8 scored shapes
for all M = 1..9**; stock upstream diverges at **M=2**; vendored first diverges at
**M=10** (max |Δ| 0.3125). Confirms the kernel contract at
`mlx-generated/quantized.cpp:973-980` by measurement.
⇒ **No depth change in d ∈ 0..8 can alter an emitted token via the verify
matmul.** Any acceptance/token movement across depths is policy or head, never the
kernel. This deletes a whole confound class from every depth experiment.
It does **not** cover attention/SDPA or GDN — alphonse's width-9 hexfloat row gate
is still open, but the projections are eliminated as a suspect.

### Dead ends closed by PR #5 — do not re-propose

- **Padding verify 9 → 10 is DEAD on two counts.** Speedup **0.661** (34%
  *slower*; stock 0.764) — crossing `vector_limit = 10` leaves `qmv` for
  `qmm_splitk` and loses. And `row0_survives_padding = False`: padding **changes
  row 0's bits**, so it could never satisfy the exactness contract. This is before
  any GDN/attention fast-path loss at S=10. I had requested this probe in
  edward's brief and re-requested it three times; **retracted**.
- All 8 scored shapes have `K % 512 == 0` and `N % 8 == 0`; **none** fall off
  `qmv_fast`. No alignment headroom exists.
- The shipped 4-bit head beats 8-bit (2.010× time for 1.889× bytes) — reconfirms
  askeladd's two-scope refutation of an 8-bit head.
- GDN recurrence is 2.748 → 4.007 ms/verify from M=1→9: only 4.2% / 1.9% of round
  cost at d=0 / d=8. Not a target.

### Live defect found in our own shipped kernel (unassigned, small)

Vendored/stock speedup is 1.09–1.25 at M=7 and 1.08–1.19 at M=9, but **regresses
to 0.87–0.92 at M=2..5 on the two N=5120 shapes** (`out_proj`, MLP down). ~1% of
verify cost. Thorfinn's follow-up (b): a shape-aware guard keeping N=5120 shapes
off crossrow at M=2..5. Small but real and cheap.

### Campaign-value numbers from PR #5

Call-mix-weighted, roofline-normalized verify tax at M=9: **2.898 → 2.530**
(−0.368) — i.e. crossrow has *already* paid down that much of what the campaign
believed was recoverable. Absolute weighted verify **206.2 → 180.0 ms** (−12.7%);
M=1 essentially unchanged 60.7 → 60.4 as expected. Raw `cost(9)/cost(1) = 2.980`.
`BW_eff` 231.9–250.4 GB/s, `FLOPS_eff` 6.37–6.56 TFLOP/s across shapes.

**My pre-registered predictions scored:** boundary *location* — correct. Raw
`cost(9)/cost(1)` predicted 2.0–2.4, measured **2.980** — under-predicted.
Normalized `qmv_tax(9)` predicted 1.55–1.9, measured **2.530** — under-predicted.
`ceil(M/4)` unit magnitude — **wrong**. Roofline knee `M* = 7.9` — **wrong**.
Branch prediction (middle → Part B(a) only) — correct, but B(a) then died on
measurement. Net: the structural read of the kernel was sound; every *magnitude* I
attached to it was not.

### `NA = 5` is UNBLOCKED — my recorded blocker was not fatal

I had recorded the `NA = 5` crossrow experiment as blocked because Metal has no
`vec<float,5>` (legal widths 2/3/4/8/16). Thorfinn found the way through:
`mlx-generated/quantized.cpp:993-994` is

```cpp
static_assert(NA >= 2 && NA <= 4, "wide multi-row QMV supports NA in [2, 4]");
typedef vec<float, NA> VF;
```

— the bound is forced **only** by the `vec<float,NA>` typedef. A plain `float[NA]`
(or a small struct) lifts it. Register footprint is `acc[4] + partial[4] +
a0..a3 = 12·NA` plus `sums = NA` ⇒ **≈13·NA floats/thread**, so NA=5 (81 floats
incl. packed/scale/bias) plausibly fits where NA=6 may not. Payoff: M=9 needs
`ceil(9/5) = 2` streams instead of 3, targeting the +6.6 ms in-situ boundary
excess at M=9 — the largest single increment in edward's vector.
**Bit-exactness is NOT free here**: element-wise scalar code may contract to FMA
differently than the vector form, so the reduction order argument does not
automatically transfer. It must be *measured* — and PR #5 merged exactly the
instrument that measures it.

## Instrument now on the base (from PR #5)

`research/qmv_cost_curve.py`, `research/qmv_cost_curve_summary.py`,
`research/run-qmv-curve.sh`, `Tests/MLXFastTests/QwenQMVCostCurveTests.swift`
(gated behind `MLXFAST_RUN_QMV_COST_CURVE=1`, off by default; `_OUT`, `_REPS=12`,
`_INNER=8`). All outside `editablePaths` — Yukon submits none of it, growth 0.
Repro: `swift test -c release --force-resolved-versions -Xswiftc -enable-testing
--filter QwenQMVCostCurve`, then `research/qmv_cost_curve_summary.py`.
Reports per-shape cost, achieved GB/s (nominal and stream-corrected), selected
kernel, and **bitwise deviation vs the M=1 reference** — the last of which is the
exactness gate for any future kernel edit. Caveat: synthetic weights (validated
indirectly — its weighted M=1 verify of 60.4 ms sits ~10% under the in-situ
`C(0) = 67.0 ms`, which is the right order for verify-plus-overhead).

## ★★ MEASUREMENT HAZARD — each arm emits FOUR trace files (cost me a false alarm)

A traced arm writes **four** `trace.txt.<pid>` files. Two are one-line stubs. Of
the two real ones:

- the **LARGE** file (e.g. 515 lines / 512 rounds, all `d=0`, `draft_build_us=0`)
  is the **SERIAL CONTROL LEG** — correctly depth 0;
- the **SMALL** file is the **MTP leg** — the one with the real depth histogram.

Sampling with `ls … | head -1` **or** `ls -S … | head -1` picks a wrong file.
Correct histogram command (**the leading space is mandatory**, or it matches
`roun`**`d=`**`99`):

```bash
grep -o ' d=[0-9]*' <trace> | sort | uniq -c
```

I sampled the serial leg across a whole sweep, saw `d=0` everywhere, and concluded
a student's `--force-depth` was inoperative and his positive control had failed —
i.e. that hours of his work were degenerate. **All of that was wrong**; the sweep
was healthy (`base-decl` and `base256` histograms identical, proving env plumbing;
`d4` → d=4×109; `d8` → d=8×67). Caught before broadcast.

**Standing lesson: when a conclusion implies a student has wasted hours, verify
the measurement instrument before broadcasting.** Bad claim #11 caught
pre-broadcast — and the only one that was mine end to end.

Related, also resolved: `round=301` exit-1 failures on 512-token arms are the
**serial leg** hitting the known EOS wall at decode token 301, pre-`b219009`.
Every 512-token arm on a base at/after `b219009` completes 512 rounds, exit 0.

## THERMAL — resolved, no escalation; my earlier "no margin" claim is SUPERSEDED

A decisive 900 s idle soak: GPU idled at t=53 s (52.08 °C, 0.010 W) and reached
**39.92 °C at t=168 s**; at t=181 s `benchmark.sh --local-cool-gate-only` exited
and GPU power jumped to 20.5 W ⇒ **gate PASSED**. The idle floor is **not** at or
above the 40 °C gate, and it is passable in ~2–3 min from a hot run. My prior
"floor 40.05–40.4 °C, margin ~0" was contaminated by concurrent GPU work.
**Zero cool-gate aborts** across all of edward's arm logs. Residual risk is
contention only ⇒ serialize GPU work.

Gate mechanics (`benchmark.sh`): `COOL_GATE_TEMP_C=40` (:28), `ABORT_SECONDS=180`
(:30), `STALL_SECONDS=90` (:31), `MAX_WAIT_SECONDS=900` (:32),
`PROGRESS_EPSILON_C=0.25` (:33), `POLL_SECONDS=10`. **The 900 s ceiling almost
never binds — the stall abort does**: `waited >= 180 && (waited − last_progress)
>= 90`, where progress means a new minimum ≥0.25 °C below the previous. An abort
at ~180–270 s therefore means the die *plateaued*, which nearly always means
**something else was on the GPU**. Pass is a single sample ≤ 40.0, so jitter helps.

## CORRECTION to ESTABLISHED_FACTS — the repair counters (askeladd's r3)

My note "`prefixRepairCount = 0`, `fullRepairCount = 0` over 28 rounds ⇒
Hypothesis-2 closed at 0 ms" was **wrong**, and I had already promoted it into an
advisor instruction before askeladd caught it. Correct split:

- `fullRepairCount = 0` — **directly measured**, 28/28 rounds (the trace `repair=`
  field emits `didRepair`, set only in the full-repair fallback).
- `rollbackRoundCount` = `prefixRepairCount` ∈ **[2, 4]** — **derived, never
  measured**. d=4, N=10, mean acc 3.70 ⇒ 37/40 ⇒ deficit 3; d=8, N=9, mean acc
  7.89 ⇒ 71/72 ⇒ deficit 1 (round 16, `d=8 acc=7 repair=none`); d=5,6,7 deficit 0.
  Independently closed by `accepted_draft_rate = 0.976190476 = 164/168` ⇒ 4
  rejected drafts.

**The interpretation is STRONGER, not weaker:** partial rejection fired ≥2 times
and the expensive full re-forward fired **zero** times — a genuine tested negative
about the repair machinery (the eager post-primary checkpoint plus
`restoreAfterPrefixReject` absorbed every one), not an absence of data. Rollback
cost +1,018 µs (+0.47%) on a 218 ms round (N=1, directional). Counters
`prefix_repair_total=` / `full_repair_total=` now exist in source, so the next
traced run reports exact integers.

## PR #4 (qwen-askeladd) — closed unmerged; findings stand

Closed because only Pile A (trace file sink, +30/−4) was proposed for promotion
while the head also carried Piles B (+90 sub-step timers) and C (+67/−3
`Qwen35MTPHostTrace`), which are research-only; merging would have landed ~187
lines of instrumentation and 8,562 B of candidate growth for a `failed`
experiment. Standing findings:

1. `draft_build_us` is **NOT host-bound**: 93.4% of the 17,486 µs mean is
   `tail_async`; steady-state host-only is **599 µs/round = 0.350%**, ~33 µs per
   draft step against an assumed ~2,400 — a **~70× overestimate in my assignment
   premise**. Mechanism: mlx `async_eval()` → `eval_impl(outputs, true)` walks the
   tape on the calling thread, throttling at `MAX_ACTIVE_TASKS = 10`.
2. Shape-varying rebuild ≈ 0 (max |Δ| 276 µs = 0.19%) ⇒ refutes `mlx-lm` #250 here.
3. **Compiled decode is dead**: ≤599 µs/round total prize. Do not spend a student.
4. Accepted-token commit is **already fused** ⇒ `mlx-lm` #990's saving is banked.
5. Two-scope head test: trunk-only q4/bf16 = **3.41×** vs a 3.5550× byte ratio ⇒
   **bandwidth-bound**, 8-bit head refuted. **522.1 MB/step confirmed to 0.003%**
   ⇒ compact readout is **54.24% of ranked drafting bandwidth**; 2-bit compact
   `draft_lm_head` ≈ **−2.02% round ≈ +0.058 score** (Direction 1b).
6. `headStepCostRatio` measured **h ≈ 0.224** vs shipped `0.20`.
7. Fidelity exact 192/192, divergences 0. Guardrail after-first 1.293 vs 4.0.

**Deferred, not declined:** the trace file sink (Pile A only) should come onto the
base as advisor tooling on a future base move. Urgency dropped because edward
independently built an equivalent under `MLX_QWEN_MTP_TRACE_PATH`.

**My errors in that assignment**, on the record: the host-bound premise (~70×
wrong); the stall-guardrail mechanism (retracted); the §1a framing of
`MLX_QWEN_MTP_TRACE_FILE`; sizing off mlx #3920; and repeatedly asking for interim
PR comments via a `post_assignment_comment` tool that is **not in the students'
schema** (the report-embedded fallback is correct; I have stopped asking).

## `qmm_splitk` has NO NAX gate — ranked prefill floor strengthened

`mlx/backend/metal/quantized.cpp:1414-1440`:

```cpp
int vector_limit = transpose_ ? get_qmv_batch_limit(K,N,d) : 4;
if (M >= vector_limit) {
  int B = out.size()/M/N;
  if (transpose_ && B == 1) { qmm_splitk(...); return; }
  qmm(...); return;
}
```

NAX early-returns exist only at :697 (qmm), :892 (gather_qmm), :1237
(gather_qmm_rhs). ⇒ For our transposed, non-batched (B==1) projections
**including prefill, NAX is bypassed**. My earlier "NAX could make ranked prefill
faster than local" is **WRONG for our shapes**, which strengthens PR #3's prefill
floor `P = 4.0086 s` as transferable.

## `MLX_METAL_GPU_ARCH` — a surgical A/B lever (unassigned)

`mlx/utils.h:206`: `static std::string gpu_arch_ = get_var("MLX_METAL_GPU_ARCH", "")`
— the only occurrence. `Device::Device()` falls back to
`device_->architecture()->name()` when empty. `MLX_` is worker-allowlisted
(`QwenRuntimeWorker.swift:2643`). Every other arch consumer reads only
`.back()`; **only `quantized.cpp:85-86` reads both** `arch_gen_` and `.back()`.
⇒ **`MLX_METAL_GPU_ARCH=applegpu_g13s` changes exactly one thing: `vector_limit`
10 → 6.** Overriding *upward* (g17s) crosses `is_nax_available()` and is **not**
surgical. Env-var results are **not submittable** — this is a measurement lever
only. Note PR #5 measured `vector_limit = 10` empirically rather than assuming it,
confirming the table read.

---

## ★★★ Honest strategic reading — the campaign is knowledge-rich and win-poor

Written deliberately, because the record above is flattering and the scoreboard
is not.

**What we have produced:** a measured per-width cost law that replaced two
wrong models; a proof that widths 1..9 are bitwise-safe; a resolved `h(3)`
anomaly; a two-method cross-validation; several permanently closed dead ends;
an instrument on the base. That is real science and it will not have to be
redone.

**What we have shipped to the scoreboard: nothing.** Senpai still has **zero
official submissions**. The promoted frontier is 2.9042110287045
(`e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, `sourceRef 7351e626…`), and that
frontier **already contains crossrow** (`1033e1a` has 22 hits). Everything the
campaign has added on top of it — `b219009`, the PR #3 merge, the PR #5 merge,
these docs — is either research-only or behavioural-but-unquantified.

⇒ **We currently hold no measured scored win.** Every one of the four live
experiments is exploratory. That is the correct thing to be uncomfortable
about, and it is why the round-2 slate below is deliberately weighted toward
*small, cheap, one-constant or one-precision changes with a pre-computed
expected value*, and away from further characterization.

Reference points: `d(score)/d(candidate_seconds) ≈ −0.4335`, so 100 ms ≈
+0.043, and 2.904 → 3.0 is ≈ 220 ms. Neither of the two leads below closes
that alone. **Do not expect one experiment to reach 3.0.**

## Live slate — all four students assigned (base `b2419f41`)

| PR | Student | Assignment | Expected value | Why now |
|---|---|---|---|---|
| #1 | edward | depth marginal cost curve (r2, running) | evidence, not speed | d=7/d=8 arms are the decisive test of `d* = 7` |
| #2 | alphonse | **r3 — `segmentedVerifyDepthCap` 8 → 7, one constant** | −1.9% to −3.2% ms/tok | cheapest possible shot at a scored win |
| #7 | askeladd | 2-bit/3-bit compact draft readout | −0.28 to −0.55 ms **per draft step** | the only sized lead with a byte-count floor behind it |
| #8 | thorfinn | crossrow `NA_max` 4 → 5 | mechanism test first, ms second | `NA=5` moves the stream count at **exactly M=5 and M=9** and nowhere else |

Deliberately **not** assigned: further characterization work. We have enough.

## Pre-registered predictions (timestamped before the data lands)

Recorded here and posted to PR #1 so that a later fit cannot be passed off as a
prediction. Score these honestly when the results arrive.

1. **`d* = 7`** — d=7 beats d=8 by ≈2% ms/token at q≈1.0, more as q falls.
   **STATUS: expected to refute.** The closed-loop simulation (see "the trap I
   thought I had found") puts the realised mode at **3** in every acceptance
   regime. Recorded as heading for refutation *before* Edward's data lands, so
   it still scores as a prediction. This does not retract the prediction; it
   records that I now expect to lose it.
2. **The depth curve is non-monotone** — d=3 beats d=4. **This one now carries
   the result**: it is the prediction the closed-loop model rests on.
3. **`H ≈ 3.73 ms`** per head-step; per-round constant `c ≈ 4.46 ms`.
4. **The M≈4 ramp will NOT move under `NA=5`.** The ramp and the boundaries
   looked separable in PR #5's data; `NA=5` should touch only the boundaries.
   If the ramp moves too, the two-component model is wrong.
5. **Lowering *draft* precision cannot change the emitted token stream.**
   Confirmed structurally — acceptance is
   `acceptedDraftPrefixCount(drafts:verifyArgmax:)`, the first index where
   `verifyArgmax[i] != drafts[i]` — but **still to be verified by parity run**,
   because "matching one argmax is insufficient" is a standing hazard.

Known way for 1–3 to be wrong: the model behind them assumes acceptance is
**position-independent**, and `positionAcceptEMA` exists precisely because it
is not.

My round-1 prediction record, for calibration: boundary *location* correct;
branch correct; **`ceil(M/4)` magnitude wrong; roofline knee wrong; both
magnitude bands under-predicted** (raw `cost(9)/cost(1)` predicted 2.0–2.4,
measured 2.980). **The structural reads have been sound; every magnitude
attached to them has not.** Weight predictions 4 and 5 accordingly — 4 is
structural, and I trust it more than any number I have attached to it.

## ★★ Advising lesson — I amended one brief thirteen times and got zero commits

PR #2 accumulated **13 advisor comments and 0 student commits**. I had been
reading that as the student being silent. On review the causal story is the
reverse: each comment *added* scope — width-9 exactness, then the hexfloat
gate, then the staircase, then the guardrail statistic, then cap-7 — until the
brief was unfinishable and probably unreadable.

Corrective action taken: r3 discards the entire history and replaces it with
**one constant**, explicitly telling the student that nothing prior is still
required, and that the fault was mine.

**Standing rule, added:** *an amendment must remove at least as much scope as
it adds.* If it cannot, the right move is a fresh revision that resets the
brief — not another comment. "Feedback volume is not progress" was already
policy; this is its sharper form. Also: when a student produces nothing for a
long time, **suspect the brief before suspecting the student**, and ask them
directly whether the obstacle is on my side or the host's.

## ★★ Advising lesson — I broadcast a policy headline from a static model

I told Edward that the measured cost curve alone would be a *regression* and
that a global argmax was required. That came from evaluating the shipped loop
at a **frozen** `positionAcceptEMA` vector. One closed-loop simulation later —
same loop, but letting `recordAcceptOutcome` move the EMAs — argmax and greedy
were identical and the "regression" was the winner.

**Standing rule, added:** *simulate the actual dynamics before broadcasting a
policy conclusion.* A controller whose own output determines which state
variables receive evidence cannot be analysed at a fixed parameter vector.
`positionAcceptEMA` is a ratchet; treating it as a constant is the same class
of error as reading an in-file comment instead of the source.

Two things kept this cheap, and both should be repeated: the wrong arm was
requested **specifically because I predicted it would lose** (so the reversal
cost one comment, not one experiment), and the correction was sent as an
explicit in-thread supersession naming the earlier feedback ID, rather than
quietly restating the new view.

## Consequence of PR #5 for the other briefs (already communicated)

- **Alphonse's Part A is largely dead, in a good way.** The width-9 hexfloat
  row gate is unnecessary for the *projection* path — widths 1..9 are
  bitwise-identical to M=1 on 8/8 scored shapes. SDPA/attention and GDN remain
  uncovered, but nothing live depends on them now.
- **Edward gains a deleted confound class.** Any token movement across depths
  in his sweep is policy or head, never the verify matmul.
- **Everything I told either of them about the `ceil(M/4)` magnitude or the
  `M* = 7.9` knee is refuted** and was explicitly retracted in-thread.


---

# The campaign has banked nothing yet, and the reason is now proven

I went looking for an unbanked scored win — some result already sitting in the
base that had never been submitted. There is none, and the proof is short.

## Provenance, settled

`7351e626...` (the `UPSTREAM_SHA` quoted in every assignment brief) is **not a
commit in this repository**. It is a `sourceRef` into the organizer's tree. Its
content arrived here as:

```
ce15975 | 08-16 12:59 | mmcguire | Sync promoted organizer frontier 7351e62674bc600f0ca148d3a1b0604716a09db6
```

That commit **is** the promoted frontier: submission
`e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, official score **2.9042110287045**.
Every `Validate submission <uuid>` commit by `yukon-autoresearch[bot]` is an
ancestor of our HEAD, so the base carries the whole validated pool.

## The decisive diff

```
git diff --stat ce15975 HEAD -- Sources/ Vendor/
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift | 24 +++---- 53 ------
```

- `Vendor/mlx-swift-lm/.../Qwen35.swift` is **byte-identical** to the promoted
  frontier. All crossrow work, the compact draft vocabulary, the fused select
  kernel — all of it is *inside* 2.9042 and none of it is ours.
- The **only** scored-path delta is `b219009`, the operator's own
  "continue fixed decode windows past EOS" harness-correctness fix.
- `git diff b219009 HEAD -- Sources/ Vendor/` is **empty**.

**Senpai has contributed zero bytes to the scored path.** Submitting the base
today would reproduce ~2.9042 and bank nothing. This closes the question; do
not re-open it without a scored-path change in hand.

It also retires an earlier wrong note of mine that treated `7351e626` as a
local ancestor, and confirms by ancestry what I had only inferred from symbol
hit-counts: **crossrow is inside the frontier.**

## What this reframes

Round 1 produced knowledge, not score. That is a legitimate outcome for a
first round, but it must be said plainly: the *only* paths to a scored delta
are the four live experiments. Everything else is instrumentation.

---

# Depth is competitive surface, and the policy that sets it is mispriced

## The operator opened depth on purpose

`fixtures/qwen3_8_27b_mtp_track.json`:

- `/protocol/maximum_depth = 8` — the trusted per-round verify-width bound.
- `/protocol/offered_draft_depth_ceiling = 8`, **operator-ratified 2026-08-14**,
  replacing a pinned `candidate_depth = 2`.
- `/protocol/candidate_declares_no_depth = True` — the parent *offers* a
  ceiling; the **candidate chooses 0..8 per round, adaptively**.
- The note is explicit: *"Depth is competitive surface now: the previous pin
  carried a standing TODO to re-derive it across the whole pool, and opening it
  moves that re-derivation from the operator to the competitor."*

So re-deriving the depth schedule is an **invited** move, not a loophole. The
ranked workflow sets `MLXFAST_QWEN_MTP_DEPTH = 8`.

**Correction to a standing note:** `effective_depth = 1 on all 48 (configured
depth 2)` sits under `/calibration/expected_raw_median_provenance`. It
describes the **calibration reference**, not the frontier. Do not cite it as
evidence about what the current candidate drafts.

## ★★★ `costModelDepth` is a hill-climb on a non-monotone function

`Qwen36MTPBlockSession.swift:573-604`. The loop continues while

```
reach > h * (1 + expected) / (1 + depth*h)
```

which is algebraically *"continue iff `(1+expected)/(1+cost)` strictly
improves."* It is a **strict hill-climb that stops at the first local
maximum**. That is correct **only because `h` is flat** (`headStepCostRatio =
0.20`, `:530`): a flat cost makes the objective monotone up to the cap.

The measured cost is neither flat nor monotone. Cost of the j-th draft, in
width-1-verify units (PR #5 isolated, cross-validated against Edward's in-situ
`h(d)`):

| j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| h(j) | 0.086 | 0.080 | 0.245 | **0.377** | 0.294 | 0.302 | 0.289 | **0.393** |

The shipped 0.20 **overprices j=1,2 by ~2.4x** and **underprices j=3..8 by
1.2-2x**. The resulting objective at perfect acceptance:

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| tokens/verify-unit | 1.000 | 1.841 | 2.574 | **2.836** | 2.797 | 2.882 | 2.937 | **2.993** | 2.936 |

It **dips at depth 4 — the M=5 stream boundary — then recovers to a global max
at depth 7.** This is an independent second derivation of pre-registrations #1
(`d* = 7`) and #2 (non-monotone, d=3 beats d=4), from the policy objective
rather than from the ms/token table.

## The trap I thought I had found, and the closed loop that reversed it

**Read this whole section before citing any number above it.** I broadcast a
headline from a static model and refuted it myself within the hour. The static
result is kept here only so the failure mode stays legible.

### What the static (frozen-EMA) model said

Transcribing the shipped loop exactly and running three policies at *fixed*
flat per-position acceptance q (`research/depth_policy_check.py`):

| q | shipped (flat h, greedy) | measured h, **same greedy loop** | measured h, **global argmax** |
|---|---|---|---|
| 1.000 | 8 | 3 | 7 |
| 0.976 | 8 | 3 | 7 |
| 0.940 | 8 | 3 | 3 |
| 0.900 | 7 | 3 | 3 |

Read literally: dropping the measured curve into the existing `while` loop pins
depth at 3 forever because the hill-climb hits the M=5 dip and quits, so the
search must change too, and argmax buys the difference between 3 and 7.
**All three of those inferences are wrong.**

### Why it is wrong: `positionAcceptEMA` is a ratchet, not a parameter

`recordAcceptOutcome` (`Qwen36MTPBlockSession.swift:609-635`) only ever gives
evidence to positions the policy *already chose to draft*. Positions strictly
inside the accepted prefix move toward 1.0; the position at `acceptedCount`
moves toward 0.0 on a real reject; on a fully accepted round the position just
past the prefix receives transferred optimism toward 0.95, **and only if it is
currently below 0.95**. Everything deeper keeps the cold seed
`0.85 * 0.98^i` forever. The depth choice therefore selects its own evidence,
and the loop can only widen by one position per fully-accepted round. A frozen
EMA vector is not a model of that.

### The closed loop: 400 rounds, real `record` semantics, streak gate live

| ground truth | shipped | curve + greedy | curve + argmax |
|---|---|---|---|
| easy prose (0.98) | 2.746 (d~8) | **2.785, +1.4%** (d~3) | 2.785, +1.4% (d~3) |
| mid prose (0.93) | 2.396 (d~4) | **2.615, +9.1%** (d~3) | 2.615, +9.1% (d~3) |
| decaying (0.97^(i+1)) | 2.454 (d~4) | **2.686, +9.5%** (d~3) | 2.686, +9.5% (d~3) |
| hard prose (0.85) | 2.110 (d~4) | **2.295, +8.8%** (d~3) | 2.295, +8.8% (d~3) |

Three corrections, all sent to Edward in-thread:

1. **Argmax buys nothing.** It is identical to greedy in every regime once the
   EMAs are allowed to move. The gap in the static table was an artifact of
   freezing them. **The minimal change is the whole change: swap the scalar for
   the vector and keep the `while` loop.** The argmax arm was dropped.
2. **`d* = 7` does not survive.** Realised mode is **3** everywhere.
   **Pre-registration #1 is heading for refutation**; pre-registration #2
   (non-monotone, d=3 beats d=4) now carries the result.
3. **My prediction that the greedy arm would lose was backwards.** It is the
   winner. I had asked for that arm *because I expected it to fail*, which is
   the only reason the reversal was cheap.

### Honest sizing

Tokens-per-verify-unit is not ms/token — the round has a fixed non-verify
component that dilutes any verify-side ratio. Hand-computing the dilution for
mid prose at d=4 -> d=3: **27.70 -> 26.62 ms/token, about +4%**, not +9%.
So `2.9042 * 1.041 ~ 3.02`. Edward was told +4% is the order of magnitude and
his measurement is the number; I declined to issue a third projection.

Superseded feedback: `qwen38-r1-e1-fb-greedy-is-a-hillclimb` (wrong), corrected
by `qwen38-r1-e1-fb-correction-argmax-not-needed` (current).

## There are two caps, and the interesting one is 4

```swift
let widthCap = fullAcceptStreak >= segmentedStreakGate   // 3
    ? segmentedVerifyDepthCap                            // 8
    : sdpaWidthWallDepthCap                              // 4
```

The local reference runs **effective draft 5.4 at acceptance 1.0**. A loop that
averages 5.4 while the cap is 8 is stopping on the *cost* test, not the cap —
so **`segmentedVerifyDepthCap = 8` almost certainly never binds**, and PR #2's
one constant is probably a no-op. Told Alphonse directly, with the histogram
reframed as the whole experiment rather than a warm-up, and asked him for the
number I actually want: **the fraction of rounds running with streak < 3**,
where the real ceiling is 4. Note that a ceiling of 4 sits exactly below the
M=5 boundary — it may be accidentally well placed.

## Scoring consequence not to forget

`published_score = median(raw_p over all timed prompts)` over 8 prose prompts —
the mean of the 4th/5th order statistics. Per-prompt raw ratios: botany 0.8467,
drama 0.9587, plutarch 0.9701, beagle 0.9837, essays 1.0044, republic 1.0116,
travel 1.0581, medicine 1.0726. **Improving the worst prompt moves nothing.**
A policy that helps only low-acceptance prompts (where widthCap = 4 binds) can
be a real speedup and score exactly zero. Target the middle of the
distribution — plutarch/beagle/essays — or move all eight.

## Estimated prize, flagged as an estimate

Current estimate, after the closed-loop correction: the win comes from the
policy settling at **3** instead of drifting to 4/8, worth **about +4% ms/token**
on mid-difficulty prompts once the verify-side ratio is diluted by the fixed
part of the round, i.e. `2.9042 -> ~3.02`. That is the whole remaining gap to
the 3.0 gate and then some — which is exactly why it should be distrusted until
measured.

**This is a motivating estimate from a reconstruction that has already had
three attached magnitudes refuted** (the `ceil(M/4)` boundary magnitude, the
roofline knee, and my own argmax headline). It justifies the experiment. It
concludes nothing. The superseded version of this paragraph projected ~2.99
from argmax landing on 7; both halves of that sentence are now dead.

