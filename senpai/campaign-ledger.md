# Qwen 3.8 Senpai Campaign Ledger

This is the compact, versioned index for advisor handoffs. Update it with every
terminal experiment and official receipt. Keep large local score artifacts out
of Git; link only reproducible commands, commits, and concise evidence.

Machine-readable frontier pins live in
[`frontier-state.json`](frontier-state.json). If this ledger and that file
disagree, stop and repair both before assigning or submitting work.

## Current frontier

Observed from Yukon and the organizer remote at `2026-08-17T08:09:23Z`.

| Field | Value |
| --- | --- |
| Organizer source | `Layr-Labs/qwen-3.8-mtp-challenge` |
| Organizer synced commit | `32b94cb67d2f3a102a36382d2beb62eee8d99db5` |
| Best promoted submission | `03dedda8-fc70-4e3e-881f-5384a17af405` |
| Promoted source ref | `32b94cb67d2f3a102a36382d2beb62eee8d99db5` |
| Official score | `2.94661597308114` |
| Campaign `BASE_SHA` | Fetch `origin/main`, then run `git rev-parse origin/main`; the Git ref is authoritative because a file cannot contain the hash of its own commit |
| Submitted solver snapshot | `32b94cb67d2f3a102a36382d2beb62eee8d99db5` |

The promoted receipt above is the public Yukon frontier used to bootstrap this
campaign; it is not claimed as a Senpai-authored result.

`03dedda8` (solver `vibecodooor`, `+0.012329`/`+1.24%`) supersedes `5c523482`
(source `cdb06b7045622fc40c1b336af28892c073ba28a3`, `2.93428682708139`), which
held the frontier for under three hours on 2026-08-17. Any handoff still
quoting `5c523482` or `cdb06b70` is one promotion stale. `cdb06b70` is a
verified ancestor of `32b94cb`, so the newer receipt strictly contains it.

Relative to `cdb06b70`, the promoted delta is three schedule-neutral,
head-neutral edits confined to `Qwen36MTPBlockSession.swift` (`+64`),
`Qwen36MTPTarget.swift` (`+18`), and `Qwen35.swift` (`+47`): publish the
post-final-norm block the target verify forward already computes, reuse the
accepted prefix for proposal-head history as one contiguous slice, and compile
Qwen attention output `x * sigmoid(gate)` as one shapeless fused pass. The
inherited 4-bit/group-64 proposal head arrived earlier in the `033f6227`
span, not in this delta.

Scheduler constants now live at `headStepCostRatio = 0.18`,
`sdpaWidthWallDepthCap = 5`, `segmentedVerifyDepthCap = 8`, and
`segmentedStreakGate = 3`. Any candidate that was fitted against the previous
`0.20 / 4 / 7` triple must be refitted before it is trusted on this base.

`benchmark.json` is byte-identical across the sync: `editablePaths` and
`optionalEditablePaths` are unchanged, so no editable-surface transition
applies. The organizer `AGENTS.md` diff from `7351e626` to `32b94cb` is empty,
so this sync ported no new enforceable rule.

### Preservation proof for the 2026-08-17 sync

Campaign `main` now differs from organizer `32b94cb` on **no editable path at
all**: `git diff --name-status 32b94cb HEAD` lists only the campaign overlay
(`senpai/`, `.agents/skills/`, `research/twin_audit.py`, `.gitignore`,
`AGENTS.md`) plus one declared trusted repair. Editable bytes are therefore
byte-exact to the officially scored surface, which is the strongest form of
this proof and the reason a local rescore is directly comparable to
`2.94661597308114`.

Declared non-editable overlay manifest (every path outside the editable set
that may differ from the organizer, with its standing obligation):

| Path | Kind | Obligation |
| --- | --- | --- |
| `Tests/MLXFastTests/QwenMTPVerbTests.swift` | repair | Organizer blob must stay `5bab1076ac632a0b8d7f5d95f30b281490fe8886`; campaign blob `710953e68da565738b2782b8e58a134ce1e06262`. Rewrites the organizer's `#expect(cond, "a" + "b")` as a multiline literal because the concatenation fails Swift type checking. Re-review the moment the organizer blob moves. |

`senpai/`, `.agents/`, `research/`, `AGENTS.md`, and `.gitignore` are
campaign-owned by construction and are not part of this manifest. Anything
else appearing outside the editable set is undeclared drift and must block the
sync until it is explained.

Retired in this sync: `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift`,
added by campaign commit `f1a874dbb65054b9dceb941abdfd89cac6e40ce4`. The
promoted frontier restores the organizer session's stop-token early exit, so
its subject `Qwen36MTPBlockSession.acceptedDraftPrefixCount` no longer exists
and the guard was the only `swift test` build failure after the import.
Retiring the guard rather than patching the snapshot keeps editable bytes
exact. Whether EOS lands inside the scored fixed window is a property of the
prompt and the target model's greedy continuation, not of the candidate,
because exactness forces token-for-token reproduction of the serial reference;
and the frontier was promoted at `2.94661597308114` with the early exit
present, so EOS does not truncate the window on the ranked pool. It can still
appear in campaign-local held-out fixtures, so the replacement control is a
tripwire, not a behaviour change: **every local A/B must assert a full-length
token match on both arms.**

### `swift test` disposition on this base

`swift test --force-resolved-versions` now **builds clean** and runs 657 tests
in 37 suites; 38 issues are recorded. The import introduced **none** of them,
and the attribution is static rather than empirical, so it needs no control
run: every file the failing tests read is byte-identical either to organizer
`32b94cb` or to the campaign fork base.

| Failing test | Reads | Attribution |
| --- | --- | --- |
| `theQwenMTPTrackIsArmedOnQwen38()` | organizer release markers/digests | Fails at organizer `32b94cb` itself; the track is released, the test still demands `QWEN38-PENDING-RELEASE` |
| `theCheckedInDeclarationSelectsThePinnedHead()` | `mtp-head.manifest.json` | Fails at `32b94cb` itself; the promoted head is the remote 4-bit/group-64 head, the test still expects the pinned bf16 head |
| `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues()` | trusted scoring semantics | Fails at `32b94cb` itself |
| `theSeededCalibrationExpectationMatchesItsRecordedProvenance()` | calibration provenance | Fails at `32b94cb` itself; status is `measured_qwen38_cutover_2026_08_14` |
| `qwen36ConfigContractDigestMatchesTheReferenceManifest()` | `config.json` digest | Fails at `32b94cb` itself |
| `participantDocsExposeDefaultCLIInstallDirectory()`, `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen()`, `submissionStaticReviewPromptCoversMeasurementStructureExploitation()` | `AGENTS.md` | Pre-existing **campaign** defect, not from this sync: `AGENTS.md` is byte-unchanged across the sync branch, but the campaign overlay dropped organizer-required doc content (confirmed missing: ``Yukon CLI (`yukon`)``) |

Two consequences worth acting on. First, **the ranked benchmark does not run
`swift test`** — the promoted frontier fails the organizer's own checked-in
head-declaration test and was still promoted at `2.94661597308114` — so
`swift test` is a campaign hygiene gate, never a submission gate. Second, the
`AGENTS.md` doc-content omission is a real campaign defect and is queued as a
standalone fix; it is deliberately **not** folded into this sync so that the
`AGENTS.md` blob-preservation pin stays checkable.

### Scoring-semantics correction: seed prefill is charged

Organizer commit `da72848b` ("Publish the seed-prefill rate in ranked
payloads"), cherry-picked here as a trusted-path change, states that the
trusted parent **charges the seed prefill inside the decode window on both
sides of the pair**. This *corrects* the earlier standing campaign note that
prefill was excluded from scoring. So

```
raw_p = (P + D_serial) / (P + D_mtp)
```

with prefill `P` additive to both numerator and denominator, dragging every
per-prompt ratio toward `1.0`. Therefore **reducing seed-prefill wall time is a
score lever that cannot affect correctness at all**, since prefill produces no
scored tokens. Sensitivity is
`d ln(raw_p)/dP = 1/(P + D_serial) - 1/(P + D_mtp)`, about `-0.03` per second
at current timings; a long prompt spending `0.3` to `0.5` s in prefill is worth
roughly `1.5%` of score, near `+0.045` absolute — larger than the `+0.012`
step that won the current frontier. The same commit ships the instrumentation
(`seed_prefill_seconds` and `prefill_seconds_per_token` per prompt, plus a
metrics-object mean in `score.json`), and nothing in the scoring path reads
those new keys, so they are observability only.

## Same-host baselines

| Base SHA | Host / memory profile | Toolchain | Head provenance | Command | Key metrics | Evidence location |
| --- | --- | --- | --- | --- | --- | --- |
| `7351e62674bc600f0ca148d3a1b0604716a09db6` | AWS Birch/Alphonse; Apple M4 Pro, 48 GB; automatic low-memory profile | macOS 26.5.2; Xcode 26.6; Swift 6.3.3 | pinned head SHA-256 `c3f8a09b3c2ff1a9b40c2c1a5f71236e2e57be31f861270c071e7ba909e18e64` | `MLXFAST_QWEN_MTP_LOCAL_WORK_DIR="$PWD/.mlxfast-local-qwen-mtp" yukon run` | pass; directional `1.4708805115725638`; exact `64/64`; serial `0.1292338595` s/token; MTP `0.0878615621` s/token; effective draft `5.4`; acceptance `1.0`; divergences `0` | ignored `score.aws-birch-alphonse.7351e626.baseline.json` (SHA-256 `0f166cdfcf0b3e1f33a438de5012c9e865c8c33ed1b7a20cc881a859eadc3b83`) and `local-docs/baselines/aws-birch-alphonse-7351e626/` |

That baseline ran on the detached promoted source. Its complete submitted
surface is identical to campaign import commit `ce159755`.

**That baseline is now stale for A/B use.** It was taken on `7351e626`, two
promotions back, and the scheduler triple moved from `0.20 / 4 / 7` to
`0.18 / 5 / 8` in between. No candidate may be compared against it on this
base; a fresh same-host baseline on `32b94cb` is required first, and every
comparison must be a matched pair measured in the same serialized window.

**Measurement hygiene, from a 1000 s idle thermal soak on this host.** Idle GPU
settles near `38.7` to `40` °C at about `0.02` W and recovers within roughly
`50` s, but the soak also caught recurring foreign spikes to `65` to `83` °C at
`16` to `31` W (`t ~ 181-244`, `575-744`, `871-922` s). Those are *other
agents' GPU work on the same host*, and they corrupt timing. The standing rule
is therefore: **parallelize builds and analysis freely, but serialize every
timing measurement**, and sample temperature around each timed arm so a
contaminated arm can be identified and discarded rather than believed.

## Official campaign submissions

| Submission ID | Candidate SHA | Base SHA | Model | Score / status | Public note | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| _No Senpai campaign submissions yet._ | | | | | | |

## Novelty index

Use one row per causal mechanism, not per branch. `Reopen when` must name new
evidence or a changed condition; “try again” is not enough.

| Mechanism | Scored path / cost center | Hypothesis | Best evidence | Disposition | Reopen when |
| --- | --- | --- | --- | --- | --- |
| Exact-fill first K/V-cache retention | `KVCacheSimple.update`; charged 512-token seed in 16 full-attention layers | Retaining incoming K/V when an empty cache receives an exact multiple of its 256-row step removes zero-fill and full-slice replacement without changing shape, offset, or layout | Stock code allocates exact-size zero arrays and then overwrites every element; the transfer map specifies boundary and rollback tests | untested; ready | Close after exact 512+1...9, snapshot/rollback, layout, and matched seed timing; reopen only after cache/step changes or fresh profiling |
| Packed GDN prework for S=1...2 | 48 GDN layers on serial, adaptive-skip, and narrow verify calls | A small-width variant that explicitly sources required old conv-state rows can extend the promoted S=3...9 packed mixer | The current mixer passed exhaustive campaign checks; source inspection locates the S=2 boundary at old conv-state ownership | untested; medium risk | Proceed only with explicit old-state construction; stop on any cache/output mismatch or less than roughly 0.25 ms named-call saving |
| Corrected packed GDN beta | Remaining `sigmoid(b)` launch/intermediate on promoted S=3...9 mixer path | Produce beta inside the packed kernel with an exact correction for finite BF16 input `0xC0DB` | Prior exhaustive BF16 comparison found one 1-ulp exception; the remaining launch is small but repeated | untested; exactness constrained | Require all 65,536 BF16 encodings, including NaN/Inf, plus exact recurrence/cache proof; close if correction cost has no end-to-end payoff |
| Compiled fused-SwiGLU causal isolation | `Qwen35FusedMLP`, S<=9 target/head path | The compiled slice-SiLU-product expression removes one launch/intermediate and may contribute independently | Present in promoted `7351e626`; the four-hunk composite improved `2.876429` to `2.904211`, but no component was isolated | promoted in composite; individual sign unresolved | Run a matched `7351e626` on/off ablation before extending or composing it; require movement outside matched noise or direct structural evidence |
| Pair GQA query heads per K/V read | Short-query SDPA in 16 full-attention layers; six query heads share each KV head | Processing two Q heads per K/V read can reduce duplicated traffic while retaining independent per-head accumulation order | Qwen geometry exposes a 6:1 reuse seam; broader grouped-SDPA work has hit M5 register/pipeline limits | unresolved; resource-gated | Reopen only when a D256 pair-head kernel compiles on M5 with acceptable registers/occupancy and exact per-head outputs |
| Seed-prefill wall time inside the charged window | 512-token seed prefill, charged on **both** arms of every paired prompt | Because `raw_p = (P + D_s)/(P + D_m)`, cutting `P` raises every per-prompt ratio with no effect on any scored token; sensitivity is about `-0.03` per second, so `0.3` to `0.5` s of prefill is worth roughly `1.5%` of score | `da72848b` states the trusted parent charges seed prefill on both sides and publishes `seed_prefill_seconds` / `prefill_seconds_per_token`; no scoring code reads those keys | **untested; highest expected value** | Open now: first measure the per-prompt prefill share, then attack it only through the editable prefill path |
| Compile-time group width `NA = 4` cliff | Cross-row QMV in the proposal head; `mtp-head.manifest.json` now declares 4-bit/group-64 | Something about `NA = 4` specifically, most plausibly register pressure or spilling, makes cross-row contraction regress; occupancy was refuted and the student withdrew the chain-depth story | E10 partitions **exactly** on compile-time group width: every `M` whose NA set contains 4 regresses, none without NA=4 does, zero overlap; ordered variant is bit-identical to control on all 96 cells, `max_abs_delta = 0` | mechanism OPEN; magnitude ceiling about `1%` of crossrow QMV time only | Reopen by reading register and spill counts out of compiled AIR, not threadgroup size; now more relevant because the frontier head is affine4/g64 |

## Experiment receipts

| Date | Branch / candidate | Mechanism | Base SHA | Local result | Official result | Result record |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 | clean Yukon source / `7351e626` | untouched promoted-tree baseline | `7351e62674bc600f0ca148d3a1b0604716a09db6` | pass; M4-local directional score `1.4708805115725638`; public tripwire passed; exact `64/64` | not submitted; local result is non-rankable | Same-host baseline row above; submitted surface now matches campaign import `ce159755` |
| 2026-08-16 | `codex/sync-organizer-frontier-20260816` / `ce159755` | exact promoted editable-snapshot import | `eb2dc26caf48ac126e0f51df7db5130414ff1d94` | release build, overlay, budget, twin, and trusted-parity checks passed; full `swift test` reached product compilation but was blocked by unchanged organizer test-source type error at `QwenMTPVerbTests.swift:755` | adopted public promoted `e6c5ef35` at `2.9042110287045`; not a Senpai-authored submission | Source delta is exactly two editable Swift files (`+54/-14`); campaign records and novelty queue refreshed separately |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817` | exact promoted editable-snapshot import, two frontier steps | `83201aa98a71d42415e1c7e85e8bc96cf609d5cf` | overlay, editable-budget (`source 2402203/3000000`, `growth 0/262144`), twin audit (29 runtime-effective twins), and strengthened trusted-parity checks passed; `swift test` builds clean with 38 pre-existing issues attributed above, none introduced by the import | adopted public promoted `03dedda8` at `2.94661597308114`; not a Senpai-authored submission | Editable diff against organizer `32b94cb` is **empty**; trusted policy commit `da72848b` cherry-picked with `-x`; orphaned `QwenMTPFixedWindowTests.swift` retired |

## Update checklist

1. Confirm the exact base, candidate, organizer, and promoted source SHAs.
2. Add or revise one novelty row with the mechanism's disposition.
3. Add the result receipt and the public submission receipt, if any.
4. Update same-host baseline rows whenever the base, host, head, or toolchain
   changes.
5. Keep `frontier-state.json` synchronized whenever organizer or promoted
   frontier pins change.
