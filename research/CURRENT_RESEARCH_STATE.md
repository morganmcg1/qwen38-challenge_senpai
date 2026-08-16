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
