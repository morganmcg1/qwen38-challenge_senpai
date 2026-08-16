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

**Flag C — the ranked serial leg is a separately pinned prebuilt baseline**
(`MLXFAST_QWEN_MTP_BASELINE_WS=/opt/bench-runner/baseline/...`). Therefore
**general target/kernel/prefill wins are fully scored and do not cancel**, even
though they cancel in the local same-build ratio. Absolute candidate wall time is
the true signal; the local ratio is a decoy for anything that is not a
schedule-or-head change.

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
- `Sources/MLXFastModel/Qwen35*.swift` is editable but **never executed**. The
  live target is `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` plus
  `Qwen36MTPBlockSession.swift`. Prove the live call path before optimizing.
- Editable budget: `source = 2,396,110 / 3,000,000`, headroom 603,890 B.
  `mtp-head/` is exempt with its own 2 GiB cap. Preflight every assignment with
  `senpai/validate-assignment-scope.sh` and `senpai/check-editable-budget.sh`.
