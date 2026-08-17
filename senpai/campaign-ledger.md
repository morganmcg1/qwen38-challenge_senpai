# Qwen 3.8 Senpai Campaign Ledger

This is the compact, versioned index for advisor handoffs. Update it with every
terminal experiment and official receipt. Keep large local score artifacts out
of Git; link only reproducible commands, commits, and concise evidence.

Machine-readable frontier pins live in
[`frontier-state.json`](frontier-state.json). If this ledger and that file
disagree, stop and repair both before assigning or submitting work.

## Current frontier

The organizer remote and the promoted Yukon row were refreshed at
`2026-08-17T21:24:25Z`.

| Field | Value |
| --- | --- |
| Organizer source | `Layr-Labs/qwen-3.8-mtp-challenge` |
| Organizer synced commit | `d1530a409848b82a0a1890141c1483875d1e0173` |
| Best promoted submission | `bd007bc7-e8ab-4919-baf4-d5e90068dd83` |
| Promoted source ref | `d1530a409848b82a0a1890141c1483875d1e0173` |
| Official score | `3.13098700135133` |
| Campaign `BASE_SHA` | Fetch `origin/main`, then run `git rev-parse origin/main`; the Git ref is authoritative because a file cannot contain the hash of its own commit |
| Submitted solver snapshot | `d1530a409848b82a0a1890141c1483875d1e0173` |

The promoted receipt above is the public Yukon frontier used to bootstrap this
campaign; it is not claimed as a Senpai-authored result.

Campaign commit `c8dceb9` imports the exact promoted submitted surface from
`d1530a409848b82a0a1890141c1483875d1e0173`. Relative to promoted source
`ed4dfd6b0e95bb1cafb26c694bc247f551d550fe`, its only executable change is
`DIRECT_NIBBLES=true` for M=7 in the readable and runtime-effective affine4
QMV twins. The intervening `0824e0e` warmup experiment was superseded and is
absent: `Qwen36MTPBlockSession.swift` has returned to `ed4dfd6` semantics.
The incumbent precision-island head remains unchanged, and no organizer
policy, contract, fixture, workflow, guide, or dependency file changed.

The exact promoted bytes reintroduced the already-known abbreviated M=8
comments in generated `quantized.cpp`. Campaign commit `08fb76a` is a separate
canonical regeneration containing only the 3-line-to-13-line comment
expansion. Removing full-line comments leaves both versions at SHA-256
`b8a68ef536608000fe3a45331797f1ac3f0f57637165ced16a5458771a07a480`;
no executable token changed. The full twin audit then passed 29/29.

## Same-host baselines

| Base SHA | Host / memory profile | Toolchain | Head provenance | Command | Key metrics | Evidence location |
| --- | --- | --- | --- | --- | --- | --- |
| `7351e62674bc600f0ca148d3a1b0604716a09db6` | AWS Birch/Alphonse; Apple M4 Pro, 48 GB; automatic low-memory profile | macOS 26.5.2; Xcode 26.6; Swift 6.3.3 | pinned head SHA-256 `c3f8a09b3c2ff1a9b40c2c1a5f71236e2e57be31f861270c071e7ba909e18e64` | `MLXFAST_QWEN_MTP_LOCAL_WORK_DIR="$PWD/.mlxfast-local-qwen-mtp" yukon run` | pass; directional `1.4708805115725638`; exact `64/64`; serial `0.1292338595` s/token; MTP `0.0878615621` s/token; effective draft `5.4`; acceptance `1.0`; divergences `0` | ignored `score.aws-birch-alphonse.7351e626.baseline.json` (SHA-256 `0f166cdfcf0b3e1f33a438de5012c9e865c8c33ed1b7a20cc881a859eadc3b83`) and `local-docs/baselines/aws-birch-alphonse-7351e626/` |

That baseline ran on the detached promoted source. Its complete submitted
surface is identical to campaign import commit `ce159755`.

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

## Experiment receipts

| Date | Branch / candidate | Mechanism | Base SHA | Local result | Official result | Result record |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 | clean Yukon source / `7351e626` | untouched promoted-tree baseline | `7351e62674bc600f0ca148d3a1b0604716a09db6` | pass; M4-local directional score `1.4708805115725638`; public tripwire passed; exact `64/64` | not submitted; local result is non-rankable | Same-host baseline row above; submitted surface now matches campaign import `ce159755` |
| 2026-08-16 | `codex/sync-organizer-frontier-20260816` / `ce159755` | exact promoted editable-snapshot import | `eb2dc26caf48ac126e0f51df7db5130414ff1d94` | release build, overlay, budget, twin, and trusted-parity checks passed; full `swift test` reached product compilation but was blocked by unchanged organizer test-source type error at `QwenMTPVerbTests.swift:755` | adopted public promoted `e6c5ef35` at `2.9042110287045`; not a Senpai-authored submission | Source delta is exactly two editable Swift files (`+54/-14`); campaign records and novelty queue refreshed separately |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-5` / `d098212` | exact promoted editable-snapshot import | `83201aa98a71d42415e1c7e85e8bc96cf609d5cf` | preservation, overlay, and budget checks passed; inherited `quantized` twin comment drift recorded without changing promoted bytes | adopted public promoted `ba493f74` at `2.95338624520432`; not a Senpai-authored submission | Exact organizer source `156b5b75`; eight editable files changed relative to the prior promoted source |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-5` / `8b85909` | fixed-window continuation after EOS on the new promoted source | `d098212` | focused `QwenMTPFixedWindowTests`: 2/2 passed; full 512-token exact replay still required | not submitted; not promoted | Restores the campaign's parent-owned fixed-window behavior without altering the trusted fixture or parent |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-6` / `29f1ee4` | exact promoted editable-snapshot import | `1c57496` | preservation, overlay, budget, and trusted-parity checks passed; both changed QMV twins are byte-identical to promoted source `79683c63`; regeneration audit is locally blocked by the missing Xcode Metal Toolchain | adopted public promoted `14b53255` at `3.02460155382533`; not a Senpai-authored submission | Exact organizer source `79683c63`; two affine4/group-64 QMV kernel twins changed relative to the previous promoted source |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-6` / `28e591f` | fixed-window continuation after EOS on promoted source `79683c63` | `29f1ee4` | source overlay and trusted-parity checks passed; Swift test compilation remains blocked by the unchanged organizer `QwenMTPVerbTests.swift:755` type error, so full 512-token exact replay is still required | not submitted; not promoted | Reapplies the campaign's parent-owned fixed-window behavior as a separate overlay |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-ed4dfd6-r2` / `f04df93` | exact promoted editable-snapshot import | `1d573f6` | preservation, campaign overlay, and editable budget passed; canonical audit exposed one comment-only stale generated twin in the promoted bytes | adopted public promoted `39fdbf62-60e4-4ab7-bf09-0d1b5a0b618a` / `ed4dfd6` at `3.07714439121787`; not a Senpai-authored submission | Exact promoted source changes four submitted paths relative to `79683c6`; the campaign import changes five paths because it also removes the unpromoted fixed-window overlay; preserves the intervening campaign-only `program.md` update at `1d573f6` |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-ed4dfd6-r2` / `7ab7376` | canonical regeneration of promoted `quantized` twin | `f04df93` | comment-stripped source SHA is identical before/after; twin audit 29/29 and release build passed; full `swift test` remains blocked by unchanged organizer `QwenMTPVerbTests.swift:755` type error | not submitted; mechanical campaign repair only | Canonical output expands three comments to thirteen; no executable token changes; no AOT Metal source changed, so `mlx.metallib` rebuild is not applicable |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-bd007bc` / `c8dceb9` | exact promoted editable-snapshot import | `1d1eeda` | preservation, campaign overlay, editable budget, trusted parity, and release build passed; full `swift test` rebuilt products but remains blocked by the unchanged organizer `QwenMTPVerbTests.swift:755` type error | adopted public promoted `bd007bc7-e8ab-4919-baf4-d5e90068dd83` / `d1530a409848b82a0a1890141c1483875d1e0173` at `3.13098700135133`; not a Senpai-authored submission | Net executable delta from `ed4dfd6` is the M=7 direct-nibbles flag in the readable/generated QMV twins; no organizer policy or dependency files changed |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-bd007bc` / `08fb76a` | canonical regeneration of promoted `quantized` twin | `c8dceb9` | comment-stripped source SHA is identical before/after; twin audit 29/29; no AOT-only Metal source changed, so `mlx.metallib` rebuild is not applicable | not submitted; mechanical campaign repair only | Canonical output expands three comments to thirteen; no executable token changes |

## Update checklist

1. Confirm the exact base, candidate, organizer, and promoted source SHAs.
2. Add or revise one novelty row with the mechanism's disposition.
3. Add the result receipt and the public submission receipt, if any.
4. Update same-host baseline rows whenever the base, host, head, or toolchain
   changes.
5. Keep `frontier-state.json` synchronized whenever organizer or promoted
   frontier pins change.
