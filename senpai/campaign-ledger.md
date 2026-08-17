# Qwen 3.8 Senpai Campaign Ledger

This is the compact, versioned index for advisor handoffs. Update it with every
terminal experiment and official receipt. Keep large local score artifacts out
of Git; link only reproducible commands, commits, and concise evidence.

Machine-readable frontier pins live in
[`frontier-state.json`](frontier-state.json). If this ledger and that file
disagree, stop and repair both before assigning or submitting work.

## Current frontier

Observed from Yukon and the organizer remote at `2026-08-17T11:46:41Z`.

| Field | Value |
| --- | --- |
| Organizer source | `Layr-Labs/qwen-3.8-mtp-challenge` |
| Organizer synced commit | `0b071ed9db211f17554bc5a13fb7381f14d709b3` |
| Best promoted submission | `ba493f74-c0fe-440a-a956-f77d26232e54` |
| Promoted source ref | `156b5b75bdfac82ae406487f531fd991e7fdfd30` |
| Official score | `2.95338624520432` |
| Campaign `BASE_SHA` | Fetch `origin/main`, then run `git rev-parse origin/main`; the Git ref is authoritative because a file cannot contain the hash of its own commit |
| Submitted solver snapshot | `156b5b75bdfac82ae406487f531fd991e7fdfd30` |

The promoted receipt above is the public Yukon frontier used to bootstrap this
campaign; it is not claimed as a Senpai-authored result.

Campaign commit `d098212` imports that exact promoted submitted surface. The
source is based on trusted organizer parent
`d077e68567827e8d926272df2245226d72b889ac`; relative to the previous promoted
`7351e62674bc600f0ca148d3a1b0604716a09db6` snapshot, its submitted delta spans
eight editable files (`+217/-41`), including the adaptive schedule, target and
head-state paths, quantized kernel twin, and proposal-head manifest. Trusted
base ancestry was reviewed separately and was not imported as solver code.

Campaign commit `8b85909` then reapplies the fixed-window post-EOS continuation
required by the current parent-owned 512-token contract. That overlay is not
part of the promoted Yukon receipt above and must pass exact 512-token replay
and official validation before it can itself be called promoted.

**Plausibility ceiling: `5.0`.** Raised from `3.0` by operator commit
`a5854b979499800a6f5f71a8d4fc14fd43ca4723` (2026-08-17, `AGENTS.md` +
`senpai/program.md` only) and readable at
`benchmark.json /scoring/decodeSpeedupCeiling` on base `b85e782`. It is a
fail-closed administrative gate, **not** a stop target and not an optimization
target (`senpai/program.md:21`). Headroom from the promoted `2.95338624520432`
is `+2.047` score — at the stale `-0.4335` calibration roughly `4.7` s off a
`~12.05` s candidate leg, i.e. about 39% of the whole MTP leg. No lever measured
this campaign is within an order of magnitude of that, so the ceiling changes
nothing operationally except that a large legitimate result must not be held
back. Docs corrected this session:
`research/ESTABLISHED_FACTS.md` and `research/CURRENT_RESEARCH_STATE.md` were
still stale at `3.0` and at the superseded `2.904` frontier.

## Base `b85e782`: what moved, and what students must re-derive

`b85e782` is the merge of the promoted-frontier sync line with the campaign
overlay line. `d098212` and `e6e6f81` are **divergent** parents, both ancestors
of `b85e782`; neither is an ancestor of the other. Measured editable deltas:

| From | Editable files changed | What they are |
| --- | --- | --- |
| `d098212` | 1 | `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (+24/-53) — the entire scored-path delta above the promoted frontier, i.e. exactly the fixed-window overlay |
| `ef16dea4` | 3 | the above, plus `Vendor/.../mlx-generated/quantized.cpp` (+5/-1) and `Vendor/.../kernels/quantized.h` (+15/-1) |
| `e6e6f81` | 3 of 7 changed files | same three; the other four are `Sources/MLXFastCLI/main.swift`, two test files, and one deleted test file |

**Two paired frontier changes ship together and must be read together.** The
promoted sync moved `segmentedStreakGate` from `3` to `2` *and* moved
`qmv_fast_crossrow_affine4_g64_m<T, 8, 4>` to `<T, 8, 3>` (a 3+3+2 split, not
4+4). The organizer's own in-tree comment says why they are one change: *gate 2
reaches the width-8 verify sooner, so that kernel fires more*. Verified gate
values: `ef16dea4` = 3, `e6e6f81` = 3, `b85e782` = 2 (gate 2 was already in
`d098212`).

Consequence, and it is load-bearing: **any result measured on `ef16dea4` or
`e6e6f81` had both halves of a deliberately-paired change at the wrong setting.**
That invalidates depth-8 / M=8 arithmetic, draft-depth histograms, and
`h(8)` estimates taken on those bases — not by making them wrong measurements,
but by making them measurements of a superseded configuration. It also
independently corroborates E14's register-cliff mechanism from the organizer's
side: `M = 9` profiles cheaper than `M = 8` (319 / 437 / 216 us for M = 7/8/9)
because the even split of 8 needs two simultaneous `vec<float,4>` accumulators.

## Cross-cutting defects found and closed this session

**1. The post-EOS `notBegun` defect — CLOSED, and the fix is alignment, not a
deviation.** The editable session at `b85e782` has no stop-token behaviour at
all: `reachedStopToken` is a constant `false` (`Qwen36MTPBlockSession.swift:167`),
the initialiser takes `stopTokens _: Set<Int>` as an ignored parameter (`:171`),
and `acceptedDraftPrefixCount` (`:672-679`) is a pure longest-common-prefix with
no EOS case. `stopTokens` occurrences dropped from 7 at `ef16dea4` to 1.

The decisive evidence is that **the trusted harness never had the defect**:
`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift` (non-editable) owns
the window by count — `:121 while emitted.count < options.totalTokenCount`,
`:141 let remaining = options.totalTokenCount - emitted.count`, overshoot
truncated at `:210-211`, and `grep -c reachedStopToken` on it is **0**. Its
`:173-176` comment calls `stopTokenInsideWindow` a defence of the *old*
normalised denominator. `benchmark.json /scoring/mtpEmptyDraftRoundsLegalNote`
states the same rule in the organizer's words. So the overlay is mandatory
alignment with the organizer's own driver. It is **insurance, not speedup**: it
enables legs that would otherwise fail, and it does not move the score. Submitting
the bare advisor base therefore banks nothing over the promoted receipt.

**2. The frontier-sync merge silently deletes our tests — a defect, twice.**
`Tests/` went from 58 files at `e6e6f81` to 57 at `b85e782`. Both losses are
delete/modify conflicts resolved the wrong way:

- `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift` — deleted by `bc552e5`
  ("Retire the orphaned fixed-window EOS guard test"). This is the **fourth**
  time this literal has been removed: `f1a874d` added, `330b44e` reverted,
  `b219009` re-added, `bc552e5` deleted. Ledger row for `8b85909` below cites
  "focused `QwenMTPFixedWindowTests`: 2/2 passed" for a file that was not in the
  tree — a ledger/tree inconsistency caused entirely by this defect.
- `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` — 722 lines at `ef16dea4`,
  `e13a6fe` and `e6e6f81`; **absent** at `d098212` and `b85e782`; 792 lines on
  thorfinn's PR #16 head. It carries `QwenQMVCostCurveTests` and
  `QwenQMVParityTests`, including the bit-exact QMV parity suite that found the
  parity false-pass hole. PR #16 is the recovery vehicle and its r2 exit
  criterion is to KEEP the 792-line file at the conflict.

Standing rule from this: **a frontier sync may not reduce the test-file count.**
Compare `git ls-files Tests | wc -l` across the merge before publishing, and
resolve every delete/modify conflict in `Tests/` in favour of keep.

**3. The QMV parity harness could false-pass.** Found by thorfinn (E14) and
worth more than his negative result: a stale metallib meant an arm could be
"proved" bit-exact without the arm's code having been built. Hardened by
`68279a7` (stale-metallib audit) and `4a4014e` (rebuild around arm switches),
with a `perturb` positive control (x1.015625f) that must fire — it fired 56/96.
Any exactness claim made before that hardening is unverified. This generalises
the existing rule that `benchmark-qwen-mtp.sh` contains no `swift build`: the
robust defence is a per-arm `sha256` of the built worker recorded in metadata.

**4. `b85e782` did not compile its own test target — FIXED.** The frontier-sync
merge reformatted a multi-line string literal in
`Tests/MLXFastTests/QwenMTPVerbTests.swift` into a two-literal `+`
concatenation. Swift Testing's `#expect(_:_:)` takes `Comment?`, which is
`ExpressibleByStringLiteral` but **not** constructible from a `String`
*expression*, so the result was a hard error — `cannot convert value of type
'String' to expected argument type 'Comment?'` — and **the whole
`MLXFastTests` target failed to build at `b85e782`**. The change was pure
collateral reformatting: it is not part of the ceiling raise that the same
merge legitimately carried in `QwenMTPTrackNamingTests.swift`. Nothing was
wrong at `e6e6f81`, where the same message was a `"""` literal with a
backslash continuation; restoring that form fixes it. Verified: `swift build
--build-tests --force-resolved-versions` exit 0 in 12.6 s afterwards.

This one was load-bearing for the whole slate. Every student had just been told
to rebase onto `b85e782` and re-run the correctness gates, and askeladd's r2
exit criteria in particular ask him to reproduce a `swift test` comparison
against the base — all of which would have failed to build for reasons having
nothing to do with their work. Standing rule: **publish no advisor base without
first running `swift build --build-tests --force-resolved-versions` on it.** A
green student `swift test` at the *previous* base is not evidence about the new
one. Note also that `Tests/` is outside `editablePaths` (`AGENTS.md:197`), so
test restoration and repair cost nothing against the 3,000,000-byte source
budget or the 262,144-byte growth budget.

## Re-verified after the merge, not re-derived

Two standing conclusions were checked against `b85e782` rather than assumed,
because the merge touched `quantized.cpp` and `quantized.h`:

- **NAX still does not reach our shapes.** `is_nax_available()`
  (`device.cpp:913-931`) requires macOS ≥ 26.2 *and* `arch_gen >= (arch ==
  'p' ? 18 : 17)`. Local `applegpu_g16s` parses to gen `16` with suffix `s`, so
  `16 >= 17` fails and NAX is off locally even though local macOS is 26.5.2;
  ranked M5 crosses it. That divergence still does not matter, because the NAX
  early-returns live *inside* `qmm()` (`quantized.cpp:697`), `gather_qmm`
  (`:892`) and `gather_qmm_rhs` (`:1237`), while our transposed non-batched
  (`B == 1`) projections take `qmm_splitk()` and `return` at `:1418-1424`
  before `qmm()` is ever called. Confirmed still true at `b85e782`. So the
  prefill dequant prize really is a `qmm_splitk` problem, and the caveat in
  thorfinn's brief stands as written.
- **The decode QMV path is non-NAX on both hosts.** `quantized_nax.h` (1681
  lines) contains zero `qmv`, zero `crossrow`, zero `affine4` and zero
  `qmm_splitk` — only `affine_qmm_{t,n}_nax`, `affine_gather_qmm_*_nax`, and
  the two `qmm_*_nax_tgp_impl` helpers. There is therefore no NAX competitor to
  `qmv_fast_crossrow_affine4_g64_m` at all, and `AGENTS.md:259-260` ("`_nax`
  variants are selected on ranked M5 hardware and are first-class targets")
  does **not** put thorfinn's cross-row results or the frontier's
  `<T,8,4>`→`<T,8,3>` change off the ranked path.

## Scoring bounds: two different numbers

The plausibility ceiling is enforced at two levels and they are not the same
value. `benchmark.json:201` `decodeSpeedupCeilingNote`: the **published-median**
ceiling moved `3.0 -> 5.0` on 2026-08-17, and the box wrapper's **per-pair**
bound `MAX_PLAUSIBLE_SPEEDUP` moved `5.0 -> 8.0` in the same decision, "so the
aggregate ceiling stays strictly tighter than the per-pair bound"
(`.github/workflows/qwen-mtp-ranked-benchmark.yml:467` carries the same note).
Since aggregation is the median of per-prompt raw ratios, a single prompt may
legally land as high as `8.0` while the median it contributes to must stay at or
below `5.0`. The organizer's own stated reason for `5.0` being generous is worth
quoting when judging a suspicious number: "the authors' own exact-greedy
envelope tops out at 1.74x at a 32-token window, kernel work on this tower has
historically bought tens of percent, and 5.0 leaves room for both compounding."
Our promoted `2.95338624520432` is already well above that 1.74x envelope, which
is the honest measure of how much of this campaign's frontier is kernel work
rather than acceptance.

## Same-host baselines

| Base SHA | Host / memory profile | Toolchain | Head provenance | Command | Key metrics | Evidence location |
| --- | --- | --- | --- | --- | --- | --- |
| `7351e62674bc600f0ca148d3a1b0604716a09db6` | AWS Birch/Alphonse; Apple M4 Pro, 48 GB; automatic low-memory profile | macOS 26.5.2; Xcode 26.6; Swift 6.3.3 | pinned head SHA-256 `c3f8a09b3c2ff1a9b40c2c1a5f71236e2e57be31f861270c071e7ba909e18e64` | `MLXFAST_QWEN_MTP_LOCAL_WORK_DIR="$PWD/.mlxfast-local-qwen-mtp" yukon run` | pass; directional `1.4708805115725638`; exact `64/64`; serial `0.1292338595` s/token; MTP `0.0878615621` s/token; effective draft `5.4`; acceptance `1.0`; divergences `0` | ignored `score.aws-birch-alphonse.7351e626.baseline.json` (SHA-256 `0f166cdfcf0b3e1f33a438de5012c9e865c8c33ed1b7a20cc881a859eadc3b83`) and `local-docs/baselines/aws-birch-alphonse-7351e626/` |

That baseline ran on the detached promoted source. Its complete submitted
surface is identical to campaign import commit `ce159755`.

**That baseline is now stale for A/B use.** It was taken on `7351e626`, several
promotions back, and the scheduler moved from `0.20 / 4 / 7 / gate 3` to
`0.18 / 5 / 8 / gate 2` in between. No candidate may be compared against it on
this base; a fresh same-host baseline on the current advisor merge is required
first, and every comparison must be a matched pair measured in the same
serialized window.

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
| Seed-prefill wall time inside the charged window | 512-token seed prefill, charged on **both** arms of every paired prompt | Because `raw_p = (P + D_s)/(P + D_m)`, cutting `P` raises every per-prompt ratio with no effect on any scored token | **CLOSED by E16 (PR #18) with `closure_error_seconds = 0`.** Prefill is 99.94% GPU: true CPU graph construction is `1.8` ms = `0.045%` of `P`, and `build_us` is enqueue back-pressure, not work. Budget: GEMM-at-ceiling `3.369302` s (84.148%) + non-GEMM `0.212714` s (5.313%) + residual `0.421984` s (10.539%); measured dense-bf16 ceiling `7.401388` TFLOP/s with GEMM achieving `6.414787` = 86.67% of it. Best interior schedule `list:0,1,2,5,11,23,47` moves serial prefill to `3.993803` s = `0.2547%` of `P`, `5.9x` below the `1.5%` bar | **closed; scheduling is dead, dequant is the residual** | Do not reopen the ladder/schedule. The one live term is **dequant overhead `0.518202` s = 12.942% of prefill = `0.090180` pts ~ 7.3 frontier steps**, minus a `0.096218` s overlap credit. That is a `quantized.h` GEMM (`qmm_splitk`) question, not the cross-row QMV decode path |
| Prefill dequant overhead | `quantized.h` GEMM path (`qmm_splitk`) inside the charged 512-token seed prefill | The 12.942% of prefill spent on dequantisation is the single largest closed-budget residual on the scored path; removing even half of it is worth ~3.6 frontier steps | E16's closed budget (`closure_error_seconds = 0`, pipelined graph reproduces worker wall to 0.2%, build cost `0.00136` s); alphonse's attribution | **untested; highest expected value now** | Open now, assigned to thorfinn after PR #16. First falsify transfer: E14's findings are all cross-row QMV (`qmv_fast_crossrow_*`, decode, M<=9), and prefill dequant is the GEMM path with a different tiling and a different reduction — assume nothing carries over |
| Compact draft-head readout precision | proposal-head readout matmul, one call per draft step | Reading the compact draft vocabulary at 3 bits instead of 4 removes 22.22% of the readout's bytes on a bandwidth-bound call | E15 (PR #17): bytes 4->3 `-22.22%`, time `-24.32%` (`1.16635` -> `0.88271` ms), bandwidth `+2.77%`, delta `283.64` us/readout, same `qmv_fast_impl` on both. End-to-end at 256 tokens: MTP steady s/token `-0.9983%`, all four legs exact, acceptance term exactly zero (byte-identical depth schedules over 35 rounds, requant argmax-lossless over all 230 proposals). Per-round attribution agrees with the microbenchmark to 1.4% | **local winner; strongest clean exact candidate; awaiting 512-token ABBA on `b85e782`** | Blocking issues are measurement, not mechanism: a `7.96` degC thermal gap between arms, a 256-token window where the contract requires 512, and a base that has since moved |
| Compile-time group width `NA = 4` cliff | Cross-row QMV in the proposal head; `mtp-head.manifest.json` now declares 4-bit/group-64 | Something about `NA = 4` specifically, most plausibly register pressure or spilling, makes cross-row contraction regress; occupancy was refuted and the student withdrew the chain-depth story | E10 partitions **exactly** on compile-time group width: every `M` whose NA set contains 4 regresses, none without NA=4 does, zero overlap; ordered variant is bit-identical to control on all 96 cells, `max_abs_delta = 0`. E14 adds the register accounting: `sizeof(vec<float,5>) == 32` vs `16` for `vec<float,4>`, 13 NA-wide vectors per thread so 39 at NA=3, 52 at NA=4, 65 at NA=5, 104 at NA=5 padded; E13 found NA=5 compiles free with first spill at NA=6. **The organizer independently confirms the cliff**: the frontier moved `<T,8,4>` to `<T,8,3>` because the even split of 8 needs two simultaneous `vec<float,4>` accumulators, and `M=9` with three-lane vectors profiles cheaper than `M=8` | mechanism OPEN but now corroborated from two independent directions; magnitude ceiling about `1%` of crossrow QMV time only | Reopen by reading register and spill counts out of compiled AIR, not threadgroup size; now more relevant because the frontier head is affine4/g64 |
| Cross-row second weight pass | `qmv_fast_crossrow_*` in the proposal head; verify widths M=2..9 | Eliminating the second weight pass, or the row/NA-width tax, should recover a large fraction of verify time | **Measured and dead (E14, PR #16).** The second weight pass is worth only `+8.16%` drift-adjusted (`0.1161` h-units) because **~89% of it is cache-served**; per-shape excess is monotone in footprint (`head.lm_head` 682 MiB -> 12.07% down to `full_attn.o_proj` 16.88 MiB -> 1.32%), and structural `0.1115` agrees with interventional `0.1161` to 4%. One verify row at constant pass count costs ~`0.27` depth-0 rounds vs ~`0.11` for the pass, so the **row/NA-width tax is ~2.4x the weight-stream tax**. Arm A (`_m<T,5,2>`) is `+39.3%` slower at M=5 *and* fails parity 8/96; arm E (scalar `float[NA]` packing) is bit-identical (0/96, byte-identical output, sha256 `9e3c52a3df97856e...`) but a reproducible `+12.4%` regression | **negative; closed as a speedup, green as measurement** | Do not reopen the weight-pass framing. The tax is rows/width, not streaming — any future cross-row work must target row count |

## Experiment receipts

| Date | Branch / candidate | Mechanism | Base SHA | Local result | Official result | Result record |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 | clean Yukon source / `7351e626` | untouched promoted-tree baseline | `7351e62674bc600f0ca148d3a1b0604716a09db6` | pass; M4-local directional score `1.4708805115725638`; public tripwire passed; exact `64/64` | not submitted; local result is non-rankable | Same-host baseline row above; submitted surface now matches campaign import `ce159755` |
| 2026-08-16 | `codex/sync-organizer-frontier-20260816` / `ce159755` | exact promoted editable-snapshot import | `eb2dc26caf48ac126e0f51df7db5130414ff1d94` | release build, overlay, budget, twin, and trusted-parity checks passed; full `swift test` reached product compilation but was blocked by unchanged organizer test-source type error at `QwenMTPVerbTests.swift:755` | adopted public promoted `e6c5ef35` at `2.9042110287045`; not a Senpai-authored submission | Source delta is exactly two editable Swift files (`+54/-14`); campaign records and novelty queue refreshed separately |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-5` / `d098212` | exact promoted editable-snapshot import | `83201aa98a71d42415e1c7e85e8bc96cf609d5cf` | preservation, overlay, and budget checks passed; inherited `quantized` twin comment drift recorded without changing promoted bytes | adopted public promoted `ba493f74` at `2.95338624520432`; not a Senpai-authored submission | Exact organizer source `156b5b75`; eight editable files changed relative to the prior promoted source |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-5` / `8b85909` | fixed-window continuation after EOS on the new promoted source | `d098212` | focused `QwenMTPFixedWindowTests`: 2/2 passed; full 512-token exact replay still required | not submitted; not promoted | Restores the campaign's parent-owned fixed-window behavior without altering the trusted fixture or parent. **The suite this row cites was deleted by the merge that produced `b85e782`; restored and extended on the advisor branch (see defect 2 above)** |
| 2026-08-17 | `qwen-thorfinn/ipg-weight-passes` / `852af07f` (PR #16, E14) | cross-row IPG weight passes: can the second weight pass or the NA-width tax be removed? | `ef16dea4` | **terminal negative.** 11 W&B runs in `qwen38-mlx-challenge-senpai` (`88khsek3` `bfk6o414` `97ieuck5` refs, `tu839z8z` armB, `e62r389y` armA, `a10cxpfs` armA3, `qnwqdh03` armD, `md5dlsm0` armE, `fdus9cxa` armE2, `2qvqo4z8` cross-arm, `sxau0sjl` Q4). Second pass `+8.16%` (~89% cache-served); row tax ~2.4x pass tax; arm A `+39.3%` at M=5 and 8/96 parity failures; arm E bit-identical but `+12.4%` slower. Measured `h`: 0.2537/0.3816/0.2735/0.2771/0.2759 for d=3..7, three independent h4 = 0.3816/0.3764/0.3880 (mean 0.3820 +-1.5%, shipped fit 1.7% low). Depth-4 break-even needs `h3 <= 0.3511` = a 6.48% cut = 1581.4 us off the 24404.5 us fourth draft step; threshold 1.0693 unreachable over 400,000 random vectors | not submitted; zero submitted-path files | `research/results/qwen38-r1-e14-ipg-weight-passes.md`. **Not useful as speedup / green as measurement.** Byproducts worth more than the result: the QMV parity false-pass hole (defect 3), the min-of-regions estimator (control noise +-0.5% -> +-0.27%), and the arm-E finding that scalar packing recovers ~32 of arm A's 39 points |
| 2026-08-17 | `qwen-askeladd/draft-readout-3bit-default` / `de82bc37` (PR #17, E15) | compact draft-head readout at 3 bits as the compiled default | `ef16dea4` | **terminal succeeded; local winner.** Phase 1 (`n2xr3wx5`, cool gate PASSED, reps=21, 98,336x5,120 g64 affine): 4-bit `1.16635` ms / 283,207,680 B / 242.81 GB/s, 3-bit `0.88271` ms / 220,272,640 B / 249.54 GB/s, 2-bit `0.67003` ms; host STREAM peak `227,787,321,075.54` B/s. Phase 2 (`hgdke1uo`, 256 tokens): MTP leg s/token `-0.7272%`, steady `-0.9983%`, `measured_local_score` `+0.8723%`, control-serial-fixed modelled `+0.8379%`, peak RSS `-56.6` MB. All 4 legs exact, divergence 0, acceptance term exactly zero, requant argmax-lossless over 230 proposals and untimed (`117.007` ms once per process). Win is host-side: `draft_build_us` +1029 and `verify_build_us` +881 = 1910 of 1915 us, `eval_wall_us` unchanged (+0.008%) | not yet submitted | `research/results/qwen38-r1-e15-*`. **Advisor prediction 1 refuted** (both bit widths take the same `qmv_fast_impl`). Blockers before submission: `7.96` degC arm-to-arm thermal gap, 256-token window vs the contract's 512, and base moved to `b85e782`. r2 requires an ABBA `4,3,3,4` at 512 on `b85e782` |
| 2026-08-17 | `qwen-alphonse/prefill-ladder-adjudication` / `9116d435` (PR #18, E16) | adjudicate the seed-prefill ladder: is prefill CPU-bound, and what is the closed budget? | `e13a6fe0` | **terminal succeeded; verdict merge.** 7 W&B runs, 14 timed phases all with `all_tokens_matched=true`, `residual_divergence_count=0`, `declared_rows_total = emitted_token_total = 64`, `accepted_draft_rate=1`, `uses_pinned_mtp_head=true`. Ladder ON `build_us=2,957,503` / `eval_wall_us=1,046,892`; ladder OFF `build_us=1,796` / `eval_wall_us=4,004,115`; total `P` moves only `+0.0015` s serial. `closure_error_seconds = 0`. Best schedule `3.993803` s vs baseline `4.004000` s (`-0.0102` s). Q5 re-measure `4.002279` s sits inside the pre-merge band `[3.993803024, 4.007063985]` | not submitted; **decision: revert the `DARKBLOOM_QWEN_PREFILL_LADDER` block out of `Vendor/.../Qwen35.swift` and merge with zero submitted-path changes** | `research/results/qwen38-r1-e16-*`, `research/floor-e16.json`. Strongest closure this campaign has produced. Retracted his own E12 73.8%/26.2% CPU/GPU split in writing; resolved the impossible `23.808` TFLOP/s rate to `6.2235` TFLOP/s = 84.5% of ceiling; corrected E3's never-probed hardcoded `"mlx_python": "0.32.0"` to the actual mlx 0.29.3 / Python 3.9.6. The submitted-path delta is worth `0.00178` pts = `0.14` frontier steps and sits behind an `MLX_`-prefixed env var, i.e. named blocker #1 — deleting it is the cheap, honest resolution |
| 2026-08-17 | `qwen-edward/curve-transfer-and-refit` / in flight (PR #19, E17) | does the merged depth curve transfer to this base, and does the h-fit need refitting? | `e6e6f81` | **r1 in flight, blinded.** Banked already: the merged curve wins **by going shallower** (mean depth `2.020` vs `2.367` for scalar-0.18, acceptance `0.535` vs `0.460`, rounds `743` vs `825`); Q3 null — the 0.20 arm is `0.655-0.898%` *faster* than the 0.18 arms against a byte-identical serial-leg floor of `0.503%` and pair noise `0.245%`. Shipped-vs-measured `h` marginals: d=1 `M2->3` fit is **1.49x low** (`.0775` vs `.1152`); d=3 `M4->5` `.3754` vs `.3761` | not submitted | Headline `R'` must be re-measured on `b85e782`: the depth histogram `{1:2, 2:231, 3:13}`, the `M>=5 = 0.00%` claim, and all depth-8/M=8 arithmetic are stale under gate 2 + `<T,8,3>`. d=3 (`_m<T,5,3>`) is unaffected. Also owes the `research/e11-notes.md` retraction of the refuted r3 "SCORING CORRECTION" — **that record is currently wrong on the advisor branch and the debt is his** |

## Update checklist

1. Confirm the exact base, candidate, organizer, and promoted source SHAs.
2. Add or revise one novelty row with the mechanism's disposition.
3. Add the result receipt and the public submission receipt, if any.
4. Update same-host baseline rows whenever the base, host, head, or toolchain
   changes.
5. Keep `frontier-state.json` synchronized whenever organizer or promoted
   frontier pins change.
6. Before publishing a new advisor base, prove it is habitable: run `swift build
   --build-tests --force-resolved-versions` to exit 0, and check that
   `git ls-files Tests | wc -l` did not fall across the merge. Resolve every
   `Tests/` delete/modify conflict in favour of keep. Both checks exist because a
   frontier sync silently broke the test target and dropped two suites; neither
   failure is visible from a student's green run at the previous base.
7. Re-verify, do not assume, any standing conclusion that depends on a file the
   sync touched. Record the re-verification explicitly so the next reader can
   tell a checked claim from an inherited one.
