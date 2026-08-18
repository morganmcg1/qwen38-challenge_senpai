# Qwen 3.8 Senpai Campaign Ledger

This is the compact, versioned index for advisor handoffs. Update it with every
terminal experiment and official receipt. Keep large local score artifacts out
of Git; link only reproducible commands, commits, and concise evidence.

Machine-readable frontier pins live in
[`frontier-state.json`](frontier-state.json). If this ledger and that file
disagree, stop and repair both before assigning or submitting work.

## Current frontier

The organizer remote and the promoted Yukon row were refreshed at
`2026-08-18T17:01:12.000Z`.

| Field | Value |
| --- | --- |
| Organizer source | `Layr-Labs/qwen-3.8-mtp-challenge` |
| Organizer synced commit | `474c75013f333f119bdc465d849f23917b195b20` |
| Best promoted submission | `942e5ab2-1c46-4c50-b7c3-eaf948878ed0` |
| Promoted source ref | `474c75013f333f119bdc465d849f23917b195b20` |
| Official score | `3.2341518328631` |
| Campaign `BASE_SHA` | Fetch `origin/main`, then run `git rev-parse origin/main`; the Git ref is authoritative because a file cannot contain the hash of its own commit |
| Submitted solver snapshot | `474c75013f333f119bdc465d849f23917b195b20` |

The promoted receipt above is the public Yukon frontier used to bootstrap this
campaign; it is not claimed as a Senpai-authored result.

Campaign commit `006a369` imports the exact promoted submitted surface from
`474c75013f333f119bdc465d849f23917b195b20`. Relative to `86fb1f0`, it
restores the dedicated single-row affine-2/group-64 coarse-readout kernel and
keeps the executable M=8 affine-4/group-64 4+4 split. Replace semantics also
remove the prior full-memory residency and post-wire command-buffer policy.
No organizer policy, contract, fixture, workflow, guide, dependency, head
manifest, or other trusted file changed.

The declared head is
`hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae6282749a52e052496dd5300b4aa441df7301e8`,
tree digest `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`,
and 427,742,600 bytes. It preserves the promoted 4-bit/group-64 precision-island
head and adds the affine-2 compact `draft_lm_head` ABI: weight
`[98,336, 320]`, scales/biases `[98,336, 80]`, followed by an exact affine-4
rerank of a 32-token shortlist.

The exact promoted readable header carried a contradictory M=8 3+3+2 comment
beside the executable 4+4 call. Campaign commit `b4ed293` replaces only that
comment with the checked-in generated twin's accurate 4+4 description.
Executable kernel text is unchanged.

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
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-live-1` / `7906ea1` | exact promoted editable-snapshot import | `c80c023` | preservation, campaign overlay, editable budget, trusted parity, exact promoted-blob checks, and release build passed; full `swift test` reaches the unchanged trusted `QwenMTPVerbTests.swift:755` `String`/`Comment?` compile defect | adopted public promoted `824dc272-b560-4dc6-bf6c-42f58944f4cb` / `8dabcfb` at `3.19351254799833`; not a Senpai-authored submission | Four editable files changed; current q2/q4 rerank head and both M=1 MTP-only QMV dispatches are present; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-live-1` / `7b740c7` | canonical regeneration of promoted `quantized` twin | `7906ea1` | comment-stripped source SHA is identical; `quantized` reproduces; 27/29 twins pass and both remaining NAX differences are isolated to compiler-owned system-section inventory with every vendored section byte-equal | not submitted; mechanical campaign repair only | Preserves exact promoted NAX blobs; no AOT-only Metal source changed, so `mlx.metallib` rebuild is not applicable |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-12d3756` / `d631025` | exact promoted editable-snapshot import | `23f2781` | preservation, campaign overlay, editable budget, trusted parity, exact promoted-blob checks, and release build passed; full `swift test` reaches the unchanged trusted `QwenMTPVerbTests.swift:755` `String`/`Comment?` compile defect | adopted public promoted `578535f7-95e6-4f95-a34c-281b9dbbbffc` / `12d3756` at `3.19580475139646`; not a Senpai-authored submission | Two editable QMV twins changed: M=8 affine4/group-64 moves from 3+3+2 to 4+4, while the prior M=1 affine2 coarse-readout and narrow affine4 fast paths are absent; proposal-head tree remains `559b24eb`; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-12d3756` / `cc32c73` | canonical regeneration of promoted `quantized` twin | `d631025` | generated delta is comment-only; twin audit 29/29, final release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Canonical output expands three comments to ten; executable tokens are unchanged; the fresh metallib covers the imported readable-header change |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-369cc05` / `ff5866b` | exact promoted editable-snapshot import | `d649aab` | preservation, campaign overlay, editable budget, trusted parity, exact 89-path promoted-blob checks, and release build passed; full `swift test` compiles the new Qwen35 source and reaches the byte-identical inherited `QwenMTPVerbTests.swift:755` `String`/`Comment?` defect | adopted public promoted `12864bc1-9c9e-4e3b-8964-e8b9e4da8d31` / `369cc05` at `3.21000579584503`; not a Senpai-authored submission | Current source combines the exact two-dispatch top-32 proposal shortlist with the restored M=8 3+3+2 QMV split; the replaced Qwen35 file does not retain the immediately prior `868cde8` fusion suite; proposal-head tree remains `559b24eb`; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-369cc05` / `9d04199` | canonical regeneration of promoted `quantized` twin | `ff5866b` | generated delta is comment-only; explicit-toolchain twin audit 29/29, final release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Canonical output expands the abbreviated M=8 comments; executable tokens are unchanged |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-dccba74` / `abca948` | exact promoted editable-snapshot import | `a187ec6` | preservation, campaign overlay, editable budget, trusted parity, exact promoted-blob checks, and release build pass; full `swift test` reaches the byte-identical inherited `QwenMTPVerbTests.swift:755` `String`/`Comment?` defect | adopted public promoted `72ce82dc-f751-485d-a7b3-94ab6471cf87` / `dccba745` at `3.22826053954006`; not a Senpai-authored submission | Two editable QMV twins changed: the proposal-only M=1 affine-2 fast path is restored and M=8 affine-4/group-64 moves from 3+3+2 to 4+4; proposal-head tree remains `559b24eb`; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-dccba74` / `0b19827` | reconcile promoted M=8 readable/twin comment | `abca948` | comment-only delta; explicit-toolchain twin audit 29/29, frozen release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Replaces a stale 3+3+2 narrative with the generated twin's accurate 4+4 description; executable tokens are unchanged |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-86fb1f0` / `8afb5e8` | exact promoted editable-snapshot import | `14ef8c2` | preservation, campaign overlay, editable budget, trusted parity, exact five-path import, and release build pass; full `swift test` reaches the byte-identical inherited `QwenMTPVerbTests.swift:755` `String`/`Comment?` defect | adopted public promoted `3a995c2b-3c42-48e8-b982-f36a8abda0e7` / `86fb1f0` at `3.23222998733732`; not a Senpai-authored submission | Five editable paths changed; the full-memory residency/command-buffer policy and Qwen35 fusion suite are present, the dedicated M=1 affine-2 QMV is absent, and the proposal-head manifest is unchanged; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-86fb1f0` / `76b961f` | canonical regeneration of promoted `quantized` twin | `8afb5e8` | generated delta is comment-only; explicit-toolchain twin audit 29/29, frozen release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Expands the abbreviated M=8 comment to the readable header's direct-nibble/IPG4 rationale; executable tokens are unchanged |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-474c750` / `006a369` | exact promoted editable-snapshot import | `50a5be6` | preservation, campaign overlay, editable budget, trusted parity, and exact 89-path import pass | adopted public promoted `942e5ab2-1c46-4c50-b7c3-eaf948878ed0` / `474c750` at `3.2341518328631`; not a Senpai-authored submission | Four editable paths changed; the M=1 affine-2 coarse-readout kernel is restored, executable M=8 remains 4+4, and the prior full-memory residency policy is removed; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-474c750` / `b4ed293` | reconcile promoted M=8 readable/twin comment | `006a369` | comment-only delta; executable `<T,8,4,true>` call unchanged | not submitted; mechanical campaign repair only | Replaces the stale 3+3+2 narrative with the checked-in generated twin's 4+4 description |

## Update checklist

1. Confirm the exact base, candidate, organizer, and promoted source SHAs.
2. Add or revise one novelty row with the mechanism's disposition.
3. Add the result receipt and the public submission receipt, if any.
4. Update same-host baseline rows whenever the base, host, head, or toolchain
   changes.
5. Keep `frontier-state.json` synchronized whenever organizer or promoted
   frontier pins change.
