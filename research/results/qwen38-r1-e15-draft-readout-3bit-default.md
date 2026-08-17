SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mechanism_only_score_pct","available":true,"value":0.7754},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

# E15 — Default the compact draft readout to 3 bits

**Headline: the mechanism is worth `+0.7754 %` of local decode score, not `+1.9171 %`.**

The observed end-to-end local score moved `+1.9171 %` (`2.0916060 -> 2.1317039`). Attribution
splits that into `37.93 %` mechanism, `1.92 %` fewer draft calls, and `60.15 %` a
single-prompt draft-trajectory windfall that will not survive a median over eight hidden
prompts. The number this experiment is entitled to claim is the mechanism term:
**`+0.7754 %`**. Everything below leads with that figure. `+1.9171 %` appears only as the
raw observation it decomposes.

- Student / branch: `qwen-askeladd` / `qwen-askeladd/draft-readout-3bit-default`
- Assignment / revision: `qwen38-r1-e15-draft-readout-3bit-default` / `r2`
- Decision: **green locally, mechanism-scale only.** Correct, reproducible, exact-token
  clean, and it does pay for itself — but at roughly `40 %` of the headline it appears to
  pay.
- `BASE_SHA`: `af80b0fc93cf20e8405631bb53365ace21a1f913`
- Candidate commit (source under test): `0c436fac1c332645feb4eacd8fb94113b7f791ad`
- Yukon frontier used as reference: receipt `ba493f74-c0fe-440a-a956-f77d26232e54`,
  source `156b5b75bdfac82ae406487f531fd991e7fdfd30`, score `2.95338624520432`,
  plausibility ceiling `5.0`
- Submitted candidate files: **one** —
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` (`+109 / -4`)
- Supporting non-submitted files: `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` plus 13
  research-only files (`research/*`)
- MTP head: `uses_pinned_mtp_head=True`, `uses_native_mtp_head=True`,
  `mtp_head_tensor_count=15`, `bytes=849407066`, `sha256=eb481df3...adfe`,
  `origin=hf:dwsdubey/qwen3.8-27b-mtp-4bit@34ee76f6...` (see Finding D — the manifest
  disagrees with this, and both arms used the same head so validity is unaffected)

## Q-A — At what `M` does the live solver dispatch the draft-head readout?

**This was the gate on whether E15 could become an official submission. Verdict: every
draft-side readout is `M = 1`, unconditionally, on every path. The compiled default of 3
bits is SAFE as written.**

An exhaustive grep for `draftTokenID` finds exactly **four** call sites, all in
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`:

| # | line | context | slice | dispatched `M` |
|---|------|---------|-------|----------------|
| 1 | 277-278 | warm, `primedDraftID` | `primed[0..., (primed.dim(1)-1)..<primed.dim(1), 0...]` from `[1,512,hDim]` | **1** |
| 2 | 288-289 | warm, folded | `folded[0..., (folded.dim(1)-1)..<folded.dim(1), 0...]` from `[1,2,hDim]` | **1** |
| 3 | 890 | round, depth 0 | `headHidden[0..., (headHidden.dim(1)-1)..<headHidden.dim(1), 0...]` | **1** |
| 4 | 903 | round, depths `1..d-1` | same last-row slice | **1** |

No call site batches `>= 2` rows. Two facts reinforce the verdict beyond the grep:

1. **It is structurally enforced, not incidental.** The compact fused path inside
   `draftTokenID` does `padded.reshaped([Self.compactDraftPaddedCount])`
   (`Qwen35.swift` ~2422) — a 1-D reshape to exactly `98,336` elements. An `M >= 2` input
   carries `196,672+` elements and would **throw / fail-fast**, not silently fall back to a
   slow path. The selector `qwen35DraftSelectKernel` dispatches
   `grid:(1024,1,1)`, `threadGroup:(1024,1,1)`, `outputShapes:[[1,1]]`.
2. **The warm exercises the same `M = 1` shape as the timed rounds.** Both warm sites slice
   the last row, so the warmed graph shape and the timed graph shape agree. That also
   closes the warm-graph question at the shape level.

One caveat worth banking: `applyDraftLMHead`'s compact branch returns a 3-D slice
`padded[0..., 0..., 0..<compactDraftRealCount]` and *would* tolerate `M >= 2`. But on the
compact path it has **no live caller** — the only call is `Qwen35.swift:2417`, inside
`draftTokenID`'s declared-head branch (`_draftHeadW != nil`), plus the wrapper at
2722-2723. It is dead on the shipped path. That is precisely where a future batched-drafting
change would land, and it is where the Q-B hazard would bite.

## Q-B — Does the 3-bit path have a cross-row analogue?

**Verdict: ABSENT.** Not "unreachable behind a guard" — the symbol does not exist.

- Every crossrow symbol in the tree is affine-4-specific *by name*. Complete inventory of
  all 22 occurrences in
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`:
  `qmv_fast_crossrow_affine4_g64` (11, def `:860`),
  `qmv_fast_crossrow_affine4_g64_m` (8, def `:1054`, calls `_wide` at `:1074` / `:1078`),
  `qmv_fast_crossrow_affine4_g64_wide` (3, def `:969`).
- The runtime-effective generated twin
  `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` carries the **identical** set:
  same 22 occurrences, same 11 / 8 / 3 split, all `affine4_g64`.
- `grep -rn "affine3\|affine2"` across `*.h`, `*.metal`, `*.cpp`, `*.swift` returns **zero
  hits repo-wide**, in both source forms.
- The host-side guard hard-codes the width: `quantized.h:1804`
  `if (!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024) {`.
- Behind that guard, the `switch (ntg.x)` arms instantiate cases `2..9` only
  (`>= 4096` branch: `_m<T,3,3>` `:1816`, `<T,4,4>` `:1821`, `<T,5,3>` `:1826`,
  `<T,6,3>` `:1831`, `<T,7,4>` `:1836`, `<T,8,3>` `:1854`, `<T,9,3>` `:1859`, plus
  `_g64<T,2>` `:1811`; `< 4096` branch: `_g64<T,2..9>` at
  `:1869/:1874/:1879/:1884/:1889/:1894/:1899/:1904`), with `default: break` falling through
  to `qmv_fast_impl`.

So making a 3-bit crossrow reachable requires **authoring a new kernel plus a new host
dispatch arm in `quantized.h` and its generated twin** — a file assigned to `thorfinn` and
read-only for me. I inspected it and wrote nothing. Given Q-A (`M = 1` everywhere), Q-B is
moot for *this* result, but it is load-bearing the moment the frontier widens dispatch.

## Q-C — Closing the requant ordering at 512 tokens

The `MLX_QWEN_MTP_TRACE` variable is never exported on the Amdahl path (follow-up 5), so
Phase 3 has **zero** trace blocks (`trace/available=False`). A wall-clock accounting
argument from the artifacts already held closes it anyway, three independent ways.

Per-leg means over the `n = 2` arms of each bit width:

| component | `bits=4` | `bits=3` | delta |
|---|---|---|---|
| decode work | `13.996965 s` | `13.632020 s` | **`-364.945 ms`** |
| prefill | `3.994655 s` | `4.002568 s` | **`+7.913 ms`** |
| leg total | `17.991620 s` | `17.634588 s` | **`-357.032 ms`** |

Check: `-364.945 + 7.913 = -357.032` ✓

1. **The cost is once-per-process and `b3`-only.** 4-bit is exactly `0.000 ms` because
   `source_bits == bits` returns `compact` unchanged. Measured one-time 3-bit values:
   `112.898 / 115.933 / 122.189 ms`, mean **`117.007 ms`**, argmax-lossless over all 230
   proposals at 256 tokens.
2. **If it were inside a timed window it must show up in exactly one of the two timed
   components. Neither shows it.** Prefill moved `+7.913 ms` = `6.8 %` of the `117.007 ms`
   mean, so `>= 93.2 %` is provably outside prefill, and the `112.898 ms` floor is `14x`
   the observed delta. Decode work moved `-364.945 ms` — wrong sign. The leg total moved
   `-357.032 ms` — wrong sign.
3. **ABBA cancels drift, not a constant arm-specific offset — so the halves settle it.** A
   `+117 ms` `b3` penalty on a `~17.99 s` leg is `+0.650 pp`, which would put the half
   deltas at `≈ -1.31 % / -1.36 %`. The observed halves are **`-1.9630 % / -2.0059 %`**,
   agreeing with each other to `0.0429 pp` and sitting `0.65 pp` *more negative* than the
   contaminated prediction. That bounds any in-window residual far below `117 ms`, with
   margin `≈ 15x` the half-spread.

### Correcting the sign claim in the r2 feedback

The feedback states that requant contamination would be "in the direction of your result."
It is the **opposite**. The requant is a `b3`-only *additive* cost. If it were inside the
timed window, my measured `-364.945 ms` would already have absorbed it, and the true
mechanism would be **larger**, not smaller. Any residual contamination makes the reported
mechanism **conservative**. This is the third correction I have offered to r2 feedback, and
it is offered in the same spirit as the invitation to refute.

### Structural closure

The ordering is a property of the call graph, not of the token count:
`warmAllDepths -> draftTokenID -> memoized _compactDraftHead materialisation`, all of it
before `mtp_decode_begin`. Window length only changes what happens *after* `begin`. The doc
comment at `Qwen35.swift:2532-2533` states `requantizedDraftHead` is "Called ONLY from the
memoized `makeCompactDraftHead()`, which `warmAllDepths` materializes through
`draftTokenID` before the trusted..." The direct trace evidence, from
`.mlxfast-private/draft-bits/e15-r1-p2-b3/rusage.txt`, is line 24
`mtp-trace: draft-head materialised bits=3 source_bits=4 requant_ms=115.933` followed
immediately by line 25 `mtp-trace: begin seed=512 ...`; materialisation strictly precedes
`begin` on every occurrence (24/25, 289/290, 584).

## The M=2 anti-synergy — a hazard this instrument found

This deserves its own heading because I built the instrument that found it and then buried
it in a table.

**At `M = 2`, going from 4 bits to 3 bits makes the readout `+31.95 %` SLOWER.** Measured:
`delta_b3_vs_b4_m2/time_ratio = +0.3195`; effective bandwidth collapses from
`243.78 GB/s` to `143.69 GB/s`. The cause is Q-B's guard: at `M = 2` and `bits == 4` the
dispatch lands on `qmv_fast_crossrow_affine4_g64_m`, which gets **two rows for the price of
one** (`1.16175 ms` at `M=2` versus `1.16505 ms` at `M=1`). At 3 bits there is no crossrow
symbol to land on, so the dispatch falls through to `qmv_fast_impl` and pays for both rows
serially (`1.53295 ms`).

| arm | r1 s/call | r2 s/call | drift | kernel | crossrow |
|---|---|---|---|---|---|
| `M=1 b=2` | `0.0006700` | `0.0006709` | `+0.12 %` | `qmv_fast_impl` | False |
| `M=1 b=3` | `0.0008827` | `0.0008819` | `-0.09 %` | `qmv_fast_impl` | False |
| `M=1 b=4` | `0.0011664` | `0.0011650` | `-0.11 %` | `qmv_fast_impl` | False |
| `M=2 b=3` | `0.0015329` | `0.0015330` | `+0.01 %` | `qmv_fast_impl` | False |
| `M=2 b=4` | `0.0011653` | `0.0011618` | `-0.31 %` | `qmv_fast_crossrow_affine4_g64_m` | **True** |

`compare_bits_sweeps.py` r1 -> r2: `kernel identity preserved across every shared arm: True`.

**The hazard, stated out loud: merging E15 forecloses the 4-bit crossrow amortisation for
any future change that batches the draft readout.** Today that is free, because Q-A proves
`M = 1` on every live path. It stops being free the moment anything widens draft dispatch —
and the crossrow work has *positive* synergy with the streak gate (a wider verify is
reached sooner, so the kernel fires more often). Whoever widens draft dispatch after E15
merges must either instantiate a crossrow `affine3` in `quantized.h` or gate the bit width
on `M`. This belongs on `thorfinn`'s desk, not mine.

## The draft-side ceiling — the number that caps every future draft-head idea

`attribution/verify_overhead_residual_seconds = 11.6907` out of `13.996965 s` of control
decode work. That is **`85.8 %` of decode spent on target verification**, leaving
**`~16 %` for all draft-side work combined** — proposal head, readout, selector, state
management, scheduling, everything.

This is a hard cap. A draft-side optimization that is *perfect* — that removes 100 % of all
draft-side work — buys at most `~16 %` of decode. My mechanism captured `0.989 %` of decode
work, which is about `6 %` of the available draft-side budget. It also caps the
compact-draft-readout item sitting at the top of my own next-directions list: that idea is
bounded by the same `16 %`, and it is bounded much lower than that in practice because it
addresses only the readout slice.

The corollary is where the campaign's remaining headroom actually is: `85.8 %` on the
verify side. Draft-side work is a rounding error by comparison.

## Transfer to the ranked configuration — the larger figure is the ranked one

This is a **model**, not a measurement. Inputs: `READOUT_MB_AT_4BIT = 283.2`,
`HEAD_MB = {"bf16": 849.4, "q4": 238.9}`. Sources `research/ESTABLISHED_FACTS.md:1138`
and `research/CURRENT_RESEARCH_STATE.md:1626`.

| configuration | `readout_share_of_draft_bytes_pct` | modeled mechanism |
|---|---|---|
| local (bf16 head, what I measured) | `25.00 %` | `+0.7754 %` |
| **ranked (4-bit pinned head)** | **`54.24 %`** | **`+0.8334 %`** |

**The ranked-relevant figure is the larger one: `+0.8334 %`.** Ranked runs the 4-bit pinned
head, where the readout is a bigger share of draft bytes, so the mechanism should transfer
slightly *up*. But note the size of "slightly": `+7 %` relatively, **not** `2.17x`. The
readout share more than doubles while the mechanism barely moves, because the mechanism is
already only `6 %` of a `16 %` budget. Do not read the doubled share as a doubled effect.

## Noise discipline

The reason `+0.7754 %` is trustworthy at all is that the instrument is quieter than the
effect by more than an order of magnitude.

- Serial leg, all four positions: `0.065683 / 0.065571 / 0.065636 / 0.065711 s/tok` —
  total spread `0.213 %`
- `noise/control_vs_control_mtp_pct = 0.000455 %` (control MTP leg, position 1 vs
  position 4)
- `headline/serial_drift_pct = -0.1054 %`
- MTP leg pooled delta `-1.9844 %`; halves `-1.9630 % / -2.0059 %`; pooled-minus-halves
  `-0.0000 pp`
- `phase1_predicted_pct_of_decode_work = -1` (a-priori, from an independent Phase 1
  microbenchmark) versus `mechanism_pct_of_decode_work = -0.9890 %` (posterior, from Phase
  3 attribution) — a `1.1 %` relative match between prediction and measurement
- Model-free split: rows `-1.4011 %` × cost-per-row `-1.2091 %` = `-2.5932 %` against a
  measured `-2.6073 %`; per-round cost flat at `+0.0394 %`. This resolves the `+0.450 %`
  per-round anomaly flagged in r2 feedback — it was a rows-versus-rounds attribution
  artifact, not a real regression.

That last pair is what a closed loop looks like: an a-priori microbenchmark prediction of
`-1.00 %` of decode work, and an independent posterior attribution of `-0.9890 %`.

## Bandwidth: quote ~250 GB/s for read-dominated traffic, 226.9 GB/s for mixed

The corrected comment I shipped at `Qwen35.swift:2091-2093` replaced a wrong figure
("`~315 MB ... ~0.6 ms`", implying an impossible `525 GB/s`) with `283.208 MB per draft
step ... 1.166 ms/call = 242.98 GB/s against a 227.13 GB/s host STREAM peak`. That
apparently-superluminal ratio is real and it is not an error in the measurement: the draft
readout is **read-dominated** (a big weight stream in, one small vector out), whereas
`226.9 GB/s` is a triad/copy STREAM figure that **includes writes**. Comparing a
read-dominated kernel against a mixed-traffic peak understates the achievable fraction.

The in-situ readout bandwidth measured here is `243.1 GB/s`, and the `M=1` 4->3 transition
gained `+2.74 %` effective bandwidth (`243.7 -> 250.4 GB/s`) on top of a `-22.22 %` byte
reduction, for a net `-24.30 % / -283.10 µs` per call.

**Standing rule this measurement motivated: quote `~250 GB/s` as the ceiling for
read-dominated traffic and `226.9 GB/s` only for mixed traffic.** A read-dominated kernel
at `243 GB/s` is at `~97 %` of its real ceiling, not `107 %` of an inapplicable one.

## Required disclosures

### 1. Arms 1 and 2 were timed on a head that is in no pushed ref

Positions 1 and 2 were timed with driver head `9ba16c81`; positions 3 and 4 used
`bf675270` on base `af80b0fc`. Both reported `dirty=0`.

`9ba16c81` exists in my local checkout as ref `askeladd-e15-r2-prerebase`, but my only
write path to the remote is the typed submit tool, which pushes the assignment branch and
nothing else. I cannot push `9ba16c81`. Instead I offer the blob-identity proof, which is
what actually matters for the measurement:

```text
git rev-parse askeladd-e15-r2-prerebase:Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
git rev-parse HEAD:Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
  both -> 3adfe8480f0f813570b4696e571900c9569320cb
```

The compiled-surface diff between `9ba16c81` and `bf675270` is **EMPTY**, and all four arms
report the same `worker_sha256=b00c06a759e12588e02b5a7eda3bd97b316c1fce97819b416ab84b8f35806b83`.
The job split across two heads is therefore immaterial to the timing. But the ref is not
pushed, so an external reviewer cannot verify positions 1 and 2 from the remote alone, and
that is a real gap in the record rather than something the blob hash fully repairs.

### 2. The cool gate did not pass on any arm

All four arms report `cool_gate = stalled_above_40.0C`,
`cool_gate_passed_real_gate = False`, `gate_qualified_for_timing = False`. Carried
verbatim, not softened.

This is now **authorized policy rather than a violation**: the host's idle floor exceeds
`COOL_GATE_TEMP_C = 40` (`40.61 °C` on edward's host, a `40.42 °C` plateau on mine), so the
gate is unsatisfiable and `MLXFAST_LOCAL_COOL_GATE=0` is permitted subject to four
conditions — ABBA counterbalancing, recorded entry and exit temperatures, spread reported
next to the effect, and both false flags carried verbatim. All four hold here.

**This r2 run is the existence proof that justified that policy**, and I record it as such
rather than as a run that got a waiver. Entry temperatures were
`42.987 / 43.233 / 43.408 / 43.698 °C` — a **`0.711 °C` spread, monotone with position**,
which cancels to first order under ABBA. That is down from `7.96 °C` in r1. Exit
temperatures: `57.10 / 57.15 / 57.54 / 56.60 °C`.

### 3. Retraction of my own r1 thermal claim

In r1 I asserted a "`40.7 °C` inlet-bound floor" for this host. **That was wrong and I
retract it.** The true idle floor is `39.92 °C`; the `40.42 °C` figure I measured is a
*post-build* reading, i.e. residual heat from compilation, not an inlet bound. The
distinction matters because an inlet-bound floor would be a property of the machine's
environment that no scheduling change could fix, whereas a post-build plateau is simply a
consequence of measuring too soon after a build and is fixable by waiting.

I recorded this retraction in commit `6e8d6d7`, but **that commit message is visibly
corrupted**, so the plain-language version above is the authoritative record. Retractions
are results; leaving this one legible only inside a mangled commit message would have lost
it.

## Findings

- **A — recovered a deleted test.** `b85e782` deleted
  `Tests/MLXFastTests/QwenQMVCostCurveTests.swift`, the same pattern as `bc552e5` deleting
  `QwenMTPFixedWindowTests.swift`. Both restored at `af80b0fc`.
- **B — thermal retraction.** See disclosure 3.
- **C — a revert that broke the test target.** `ee977ae` "Restore organizer test source
  snapshot" exactly reverted `e20268e`, breaking `MLXFastTests` at `b85e782`. Fixed by the
  advisor at `e7cd780`.
- **D — head provenance ambiguity (unresolved, advisor-owned).** Arms report
  `bytes=849407066`, `file_count=5`, `sha256=eb481df3...adfe`,
  `origin=hf:dwsdubey/qwen3.8-27b-mtp-4bit@34ee76f6...`, `mtp_head_tensor_count=15`. But
  `mtp-head.manifest.json` declares `bytes=238934129`, `sha256=7d627027...f47` (same URL
  and revision, with a note describing 38 tensors), while `benchmark.json` names
  `EigenLabs/Qwen3.8-27B-MTP-bf16 @ 26a328e0...` with 15 bf16 tensors. Both arms used an
  identical head, so E15's validity is unaffected — but the base fails
  `theCheckedInDeclarationSelectsThePinnedHead()`.
- **E — control-script index poisoning (found and fixed in `937b568`).**
  `research/run-swift-test-control.sh` used `git checkout "$BASE" -- "$FILE"`, which also
  **stages** the base blob; the exit trap then restored from that poisoned index, silently
  leaving the base version of a candidate file in the tree. A control harness that
  corrupts the thing it is controlling is worse than no control. Fixed to revert through
  the worktree only (`git show "$BASE:$FILE" >"$FILE"`), restore with
  `git checkout HEAD -- "$FILE"`, verify `git hash-object` against
  `git rev-parse HEAD:$FILE`, and fail loudly on mismatch.
- **F — base `af80b0fc` fails `research/twin_audit.py`.**
  `TWIN AUDIT FAILED: 1/29 twin(s)` — `STALE quantized: section drift in
  mlx/backend/metal/kernels/quantized.h`. Base-inherited, comment-only, inert for JIT, but
  the tripwire is red and a red tripwire stops being a tripwire. `quantized.h` is
  `thorfinn`'s surface, so I reported rather than fixed.
- **G — fourth instance of the frontier-sync revert pattern, in `AGENTS.md`.**
  `ef16dea4` / `fbb0591` / `de82bc37` carry the newer Yukon-CLI wording (from `0ce7156`);
  `af80b0fc` and my HEAD carry the old wording. The only commit touching `AGENTS.md` in
  `ef16dea4..af80b0fc` is `79f8cd8` "Raise campaign plausibility ceiling to 5.0", which
  regressed that unrelated paragraph as a side effect. `senpai/program.md` is unaffected
  (ceiling `5.0` at `program.md:19`). `AGENTS.md` is outside student scope — reported, not
  edited.
- **Worker-digest honesty correction.** Identical `worker_sha256` across all four arms is
  *expected*, because bit selection is a runtime env override rather than a compile-time
  change. Arm differentiation therefore comes from the `strings` tripwire plus the runtime
  trace, not from the binary digest. I had earlier implied the digest was doing work it was
  not.
- **Analyzer rewrite (`e98486c`).** The original Phase 3 analyzer parsed trace blocks that
  do not exist, because `MLX_QWEN_MTP_TRACE` is never exported on the Amdahl path
  (follow-up 5). It was rewritten to derive everything from the rusage and headline
  artifacts that are actually emitted. This is why the requant ordering had to be closed by
  accounting (Q-C) rather than by trace.

## Correctness

All four arms: `serial/all_tokens_matched=True`, `mtp/all_tokens_matched=True`,
`emitted_token_total=512`, `residual_divergence_count=0` on both legs,
`parity_all_ok=True`, `max_rejected_tail_logit_delta=0`, `non_drafting_round_count=0`,
`uses_native_mtp_head=True`, `target_cache_offset_final=1024`,
`serial/declared_rows_total=512`, `mtp/declared_rows_total` `571` (`b4`) / `563` (`b3`),
and `declared_rows_total == reference_checked_row_total`.

**Swift test control — identical failure set.** Candidate job
`c08851eb-b538-426a-be4f-a8d4fcb70a29` (HEAD `e98486c`, `dirty=0`, 41 s) versus base
control `2f03e337-5a1a-4aa3-8bfd-3e030b00b3bf` (38.9 s, exit 1). Both: **672 tests, 41
suites, 38 issues, 12 distinct failing ids, 0 compile errors** -> `IDENTICAL_FAILURE_SET`.
The control genuinely recompiled (`Build complete! (15.39s)`), so this is a real control and
not a cached artifact. Artifacts under
`.mlxfast-private/swift-tests/{e15r2-head-af80b0fc,e15r2-control-af80b0fc}/`.

**Warm-graph audit — the `7b33621` hazard did not fire, and the sign is inverted.**
`bits=4` first block `0.179389 s` versus p50-after-first `0.217177 s`; `bits=3` first
`0.176125 s` versus `0.214992 s`. The first block is *faster* than steady state by
`~38 ms`, the opposite of the predicted warm-miss penalty.
`verify_block_replayed_round_count` `13` / `11`.

## Gates

```text
senpai/validate-assignment-scope.sh af80b0fc93cf20e8405631bb53365ace21a1f913 \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
  -> assignment scope OK: 1 submitted path(s)

senpai/check-editable-budget.sh af80b0fc93cf20e8405631bb53365ace21a1f913
  -> editable budget OK: source=2408922/3000000 headroom=591078
     growth=5110/262144 exempt=2410 files=154
```

## Measurement conditions

- Host: **Apple M4 Pro**, `applegpu_g16s`, `nax_available=false`. **This is not the ranked
  M5.** Every number here is directional; the `_nax` variants that run on ranked hardware
  are not exercised at all.
- Thermal policy: see disclosure 2.
- Decode window: 512 tokens, four ABBA positions, order `4,3,3,4`.
- `streak_gate=2`, `m8_ipg=3` on all arms.

| position | bits | serial s/tok | mtp s/tok | matched | temp before -> after |
|---|---|---|---|---|---|
| 1 | 4 | `0.0734865079` | `0.0351398028` | True | `42.99 -> 57.10` |
| 2 | 3 | `0.0734025917` | `0.0344500078` | True | `43.23 -> 57.15` |
| 3 | 3 | `0.0734400488` | `0.0344351016` | True | `43.41 -> 57.54` |
| 4 | 4 | `0.0735110722` | `0.0351399628` | True | `43.70 -> 56.60` |

### Attribution

`research/draft_bits_attribution.py --prefix e15r2-p3 --order 4,3,3,4`, anchored on Phase
1's independently measured `P1_SECONDS_PER_CALL = {4: 0.0011650, 3: 0.0008819,
2: 0.0006709}`. Arms: `bits=4` `n=2` `rounds=76` `rows=571` `draft_calls=495`
`accept=0.880808`; `bits=3` `n=2` `rounds=74` `rows=563` `draft_calls=489`
`accept=0.895706`.

| term | ms | share of total |
|---|---|---|
| readout precision (**the mechanism**) | `-138.436` | **`37.93 %`** |
| fewer draft calls | `-6.990` | `1.92 %` |
| fewer rows and rounds (**trajectory windfall**) | `-219.519` | **`60.15 %`** |
| measured total | `-364.945` | `-2.6073 %` of decode work |

The trajectory term is a **single-prompt windfall**: both `b3` arms are byte-identical to
each other and both `b4` arms likewise, so it is fully deterministic and fully
prompt-specific. It will not survive a median over eight hidden prompts. This is why the
document leads with `+0.7754 %`.

| metric | baseline (`b4`) | candidate (`b3`) | delta |
|---|---|---|---|
| **`mechanism_only_score_pct`** | `0` | **`+0.7754 %`** | **`+0.7754 %`** |
| local decode score (observed) | `2.0916060` | `2.1317039` | `+1.9171 %` |
| decode-only score | `2.403144` | `2.463989` | `+2.5319 %` |
| decode work (s) | `13.996965` | `13.632020` | `-2.6073 %` |
| prefill (s) | `3.994655` | `4.002568` | `+0.198 %` |
| `mechanism_pct_of_decode_work` | — | `-0.9890 %` | vs `-1.00 %` predicted |

Prefill is `22.20 %` of the control MTP leg, giving a decode-work-to-score conversion
factor of `0.77797` on this host. Edward's `0.84228` reflects *his* `15.77 %` prefill
share, not a host constant — the factor must be recomputed per host and per window.

## W&B runs

- Phase 3 (the timed ABBA experiment): `suu5mdru` —
  https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/suu5mdru
- Phase 1 (the readout microbenchmark and `M`/bits sweep): `fuka4ic1` —
  https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/fuka4ic1

Both `finished`, group `qwen38-r1-e15-draft-readout-3bit`.

Supervised jobs: Phase 3 positions 1-2 `507b6654-d1dd-46c2-83c5-935e830b5b14`
(order `4,3`); positions 3-4 `2eccaff0-60e7-4b1c-8247-17abc9f63dc0` (order `3,4`);
Phase 1 `af9b1984-0ab0-4dfd-b452-fc4b8369eaf7` (tag `e15r2-p1-b85e782`, head `81bfddb`,
`dirty=0`).

## Reproduction

```bash
git checkout <this commit>
./setup.sh && ./setup-qwen-mtp.sh

# Phase 1 — readout cost curve over bits x M, plus cross-base reproduction check
research/run-draft-bits-sweep.sh e15r2-p1-b85e782 af80b0fc93cf20e8405631bb53365ace21a1f913
research/compare_bits_sweeps.py e15-r1-bits-m-probe e15r2-p1-b85e782

# Phase 3 — four ABBA positions at 512 decode tokens
research/run-draft-bits-phase3.sh e15r2-p3 512 af80b0fc93cf20e8405631bb53365ace21a1f913 4,3 0
research/run-draft-bits-phase3.sh e15r2-p3 512 af80b0fc93cf20e8405631bb53365ace21a1f913 3,4 2

# Attribution
research/draft_bits_attribution.py --prefix e15r2-p3 --order 4,3,3,4
```

Settle environment used for every timed arm:

```text
MLXFAST_SETTLE_TARGET_C=40.0 MLXFAST_SETTLE_MAX_S=300 MLXFAST_SETTLE_STALL_S=90
MLXFAST_SETTLE_EPSILON_C=0.25 MLXFAST_LOCK_WAIT_S=300
```

## The change

`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`, `+109 / -4`:

- Line `2477`: `else { return 3 }` inside `private static let draftHeadBits` — the compact
  draft readout now defaults to 3 bits.
- Override `MLX_QWEN_MTP_DRAFT_BITS` accepts `[2, 3, 4]`, which is what made the sweep
  possible without recompiling per arm.
- Recovered the `draftHeadBits` / `requantizedDraftHead` mechanism that had been deleted.
- Corrected the byte and latency comment at `2091-2093` (see the bandwidth section).

Also included but **not submitted**: `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` (the
recovered cost-curve suite from Finding A) and 13 research-only files under `research/`.

## Suggested follow-ups (not implemented here)

1. **Isolate the trajectory term across prompts — highest value.** `60 %` of the observed
   headline is a single-prompt windfall. Running 3-bit versus 4-bit over several prompts
   would convert `+0.7754 %` from a modeled mechanism into a measured expectation. **Not
   runnable locally**: local modes expose one public fixture, and fixtures are trusted
   surface. Needs either a ranked run or an organizer-side fixture addition.
2. **`affine3` crossrow instantiation, or gate bit width on `M`** (`thorfinn`). See the
   M=2 anti-synergy section. Load-bearing before anything widens draft dispatch.
3. **Resolve Finding D** — head provenance disagreement between the arms, the manifest, and
   `benchmark.json` (advisor).
4. **Guard `sync-organizer-frontier` against archives that delete *or revert* campaign
   changes** (advisor). Four instances now: `bc552e5`, `b85e782`, `ee977ae`, and
   `79f8cd8` / `AGENTS.md`.
5. **Export `MLX_QWEN_MTP_TRACE` on the Amdahl path, or stop relying on it.**
   `run-amdahl-measurement.sh` re-points `MLXFAST_SWIFT_BIN` at `research/capture-cli.sh`
   and never exports the flag, so Phase 3 produced zero trace blocks. Worse, the variable
   is **structurally unreachable in the ranked environment** — the workflow `env:` block
   sets only `MLXFAST_*`, and the allowlist at `QwenRuntimeWorker.swift:2638-2645`
   enforces it. Any future proof leaning on this trace is worthless for ranked purposes.
   Fix it structurally or drop it. Not worth a re-run for E15, since Q-C closed without it.
6. **Retire the stale test `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`**, which
   asserts a no-op normalization that the scoring formula no longer performs (advisor).
7. **Explain the negative round-1 graph penalty.** The first block is `~38 ms/block`
   *faster* than steady state, the opposite sign to the `7b33621` hazard. Unexplained, and
   an unexplained inversion in a warm path is a latent measurement risk.
8. **Fix the comment-only twin drift in `quantized.h`** (Finding F) so the twin-audit
   tripwire goes green again (`thorfinn` / advisor).

### A note on what I did not do

I did not dump the `mtp-row` ledger at positions 1022-1024 for alphonse's E19 `key_len=1024`
fidelity band. It would have been a free rider on a 512-token leg I was already running, but
I am not running another leg, and adding a run purely to collect it is not justified at a
`~19.5x` margin.
