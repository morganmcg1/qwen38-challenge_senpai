# E9 r3 — draft-head readout bit width on the promoted-frontier base

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":false,"value":null},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}

- Student / branch: `qwen-askeladd` / `qwen-askeladd/draft-bits-multiprompt-default`
- Hypothesis and target cost: r2 measured a −1.90% / −1.52% / +0.05% spread from
  setting the compact draft readout to 3 bits, and asked whether 3 bits should
  become the in-tree default. r3 asks that question again on the promoted
  frontier. Target cost was the compact draft readout: 283.2 MB of affine-4/g64
  rows streamed once per draft step.
- **Decision: dead — the lever does not exist on this base.** Part A returned
  negative at its first gate, so per the assignment's stop rule Part B was not
  run and no timing allocation was spent.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: base `bc5e15fd` (assignment
  marker) merged forward to research base `fe38ecc`; candidate head is this
  commit. Prior r2 base was `8970d775`.
- Yukon promoted submission / source ref used as frontier: not queried — no
  candidate was produced, so no submission decision arose.
- **Submitted candidate files: none.** `git diff --stat bc5e15fd HEAD -- Vendor/
  Sources/ mtp-head.manifest.json mtp-head/` is empty. Every file in this branch
  is under `research/` or `correctness_prompts/`.
- Supporting test, tooling, or documentation files: `research/e9r3_readout_share.py`
  (A3 arithmetic), `research/run-e9r3-liveness.sh` (untimed path probe),
  `research/log_e9r3_to_wandb.py` (W&B record), this report.
- W&B record: run `lirtaqkk` —
  <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/lirtaqkk>,
  group `qwen38-r1-e9-draft-bits-default`, `job_type=source-analysis`. It carries
  the A1 symbol counts, the probe counters, and the A3 byte model as config and
  summary, plus this report as an artifact. It deliberately logs **no** speed or
  acceptance metric, because none was measured validly.
- MTP head provenance and draft policy: see "The head is not where the bits are"
  below. Draft policy on this base is the greedy marginal `costModelDepth` rule
  with `headStepCostRatio = 0.18`, `sdpaWidthWallDepthCap = 5`,
  `segmentedVerifyDepthCap = 8`, `segmentedStreakGate = 3`.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh` →
  `assignment scope OK: 1 submitted path(s)` against both `bc5e15fd` and
  `fe38ecc`.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `senpai/check-editable-budget.sh` → `editable budget OK: source=2402203/3000000
  headroom=597797 growth=0/262144 exempt=2410/2147483648 files=154`, identical
  against both `bc5e15fd` and `fe38ecc`. **`growth=0`** confirms zero submitted
  bytes changed.
- Scored-path reachability evidence: measured, not inferred. An untimed stderr
  probe shows `makeCompactDraftHead vocab=248320 draftHeadW_present=false
  lmHead=QuantizedLinear bits=4 groupSize=64` and `draftTokenID total=1
  compact_fused=1 declared_head=0` — the compact readout is reached and the fused
  path is taken, but there is no longer any bit-width knob on that path to turn.
  Full probe detail under "Untimed liveness probe".

## A1 — is the lever still live? **No. It was deleted upstream.**

The promoted-frontier rebase removed the entire PR #7 mechanism from
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`. Exact symbol counts
in that one file:

| symbol | old base `8970d775` | new base `bc5e15fd` |
| --- | ---: | ---: |
| `draftHeadBits` | 3 | **0** |
| `requantizedDraftHead` | 2 | **0** |
| `MLX_QWEN_MTP_DRAFT_BITS` | 2 | **0** |
| `makeCompactDraftHead` | 3 | 3 |

`makeCompactDraftHead()` survives, but its quantized branch now returns
directly and inherits the backbone width:

```swift
if let quantized = full as? QuantizedLinear {
    return QuantizedLinear(
        weight: compactRows(quantized.weight),
        ...
        groupSize: quantized.groupSize,
        bits: quantized.bits,   // <- no lever
```

This is a stronger negative than the assignment anticipated. The r3 brief framed
A1 as "the default integer may have moved"; in fact there is no integer. Setting
3 bits here is not a one-line default change but a **re-implementation** of the
deleted `requantizedDraftHead` (dequantize the sliced 4-bit rows in 8192-row
blocks, 13 blocks with an `eval()` each, then re-quantize). That is outside the
r3 scope as written, so I flagged it rather than silently exceeding scope.

Accordingly the branch no longer carries the r2 edit: the base merge resolved
`Qwen35.swift` in favour of the new base, and the submitted-path diff is empty.

### The head is not where the bits are — the path is live, the knob is not

Worth separating, because it decides whether re-implementation *could* ever pay.
`usesCompactDraftVocabulary` requires `lmHead != nil` and vocab 248,320 — both
hold. The `_draftHeadW` guard that would bypass `makeCompactDraftHead()` is
populated only from `mtp.draft_lm_head.{weight,scales,biases}`, and **neither
available head contains that tensor**:

- pinned bf16 head (`~/.cache/mlxfast/.../mtp-head/model.safetensors`,
  849,400,347 B): 15 BF16 tensors — `fc.weight [5120,10240]`, the seven
  `layers.0` projections, four `layers.0` norms, `norm.weight`,
  `pre_fc_norm_embedding.weight`, `pre_fc_norm_hidden.weight`. No
  `draft_lm_head`.
- declared 4-bit head (238,934,129 B): 31 tensors = 8 Linear × 3 + 7 norms. No
  `draft_lm_head`.

So `_draftHeadW == nil`, `makeCompactDraftHead()` is reached, and `draftTokenID`
takes the fused kernel path (session call sites 879 and 892, warmed at 268–287).
Runtime confirmation is in "Evidence" below.

**Route B has not shipped.** The r3 brief supposed the new base introduced a
manifest-supplied draft head. It did not, on two counts. First, the declared head
was *already* 4-bit at the old base
(`lowskillcoding/qwen38-mtp-head-4bit-g64`, 238,934,093 B, added at `deb63ad`);
`bc5e15fd` only swaps the source repo to `dwsdubey/qwen3.8-27b-mtp-4bit`
(238,934,129 B), same geometry. Second, the manifest head feeds
`mtpHeadHiddenForward`, not `applyDraftLMHead` — it is a different tensor in a
different place. Route B proper would be a head that actually ships a
`draft_lm_head` tensor; none does.

## A2 — was r2 measuring double quantization? **Yes, always.**

`makeCompactDraftHead` slices the **backbone** `lmHead`, which is affine 4-bit
group-64. It never touches the MTP head. The deleted `requantizedDraftHead`
therefore dequantized already-4-bit rows and re-quantized them to 3 or 2 bits.
PR #7 and r2 were the double-quantization experiment from the start; r3's
premise that double quantization would be a *new* penalty introduced by the
rebase is not correct.

The brief's expectation that this must be "strictly worse" for acceptance is
refuted by r2's own data: 3-bit acceptance moved **+0.019** (english), **+0.010**
(technical), **−0.007** (narrative). Double quantization did not reliably harm
acceptance — which is exactly why r2's realized gain was not attributable to the
mechanism (see the coin-flip note in Conclusion).

## A3 — how big could the prize be? **Small, and unchanged by the rebase.**

From exact byte counts (`research/e9r3_readout_share.py`; row constants
`compactDraftPaddedCount = 98_336`, `COLS = 5_120`, `GROUP = 64` are **identical
on both bases**, so the attacked bytes did not move):

| width | weight | scales | biases | total | ms at 227.13 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-bit | 251.740 MB | 15.734 MB | 15.734 MB | **283.208 MB** | 1.2469 |
| 3-bit | 188.805 MB | 15.734 MB | 15.734 MB | **220.273 MB** | 0.9698 |
| 2-bit | 125.870 MB | 15.734 MB | 15.734 MB | 157.338 MB | — |

3 bits saves 62.935 MB = **22.22%** of readout bytes, ≈ **0.277 ms** per draft
step at STREAM peak.

One correction to the brief's reasoning about the ceiling. The head the MTP block
runs differs by **measurement path, not by base**: `benchmark-qwen-mtp.sh` always
passes the pinned cache dir (`:554/603/613`), while the trusted parent resolves
the manifest. `QwenMTPHeadDeclaration.resolve(contractRoot:)` is called only at
`Sources/MLXFastCLI/main.swift:1897`, inside `qwenMTPHeadProvenancePayload` —
provenance reporting, not loading. So:

| head step denominator | readout share | cap on head step from a 24.2%-faster readout |
| --- | ---: | ---: |
| pinned bf16 head — **local** path | 25.00% | −6.05% |
| declared 4-bit head — **ranked** path | 54.24% | −13.13% |

The rebase did not lower this ceiling. If anything the ranked path flatters a
readout win more than the local path does, which means **local measurement
understates the ranked effect** — a hazard worth recording for anyone who
later measures a readout change locally and reads the ratio as ranked-equivalent.

I did **not** convert these into a share-of-round figure. That needs one
measurement I deliberately did not spend once A1 came back negative, and r2's
implied `R4 = 1.048 ms` belongs to a different schedule and must be re-derived
rather than reused. The in-tree comment claiming "~315 MB of affine-4 rows per
draft step (~0.6 ms)" disagrees with the byte model above and is worth a look by
whoever owns that comment.

## Evidence

- Host, memory profile, toolchain, and thermal policy: same racked M-series host
  and toolchain as r2; `swift build -c release --force-resolved-versions`. No
  timed arm was run, so no thermal pairing was required and no GPU-temperature
  pair is reported. The host idles at a ~40.7 °C GPU floor above the 40 °C gate.
- Exact baseline and candidate commands: no baseline/candidate pair exists. The
  only GPU work was the untimed probe:
  `research/await-lock-then-run.sh 600 research/run-e9r3-liveness.sh 16`.
- Tests and risk-based checks: release builds of both products compiled clean
  (exit 0, 0 errors), and the probe run passed the public drift tripwire. Both
  preflight scripts pass, above. I did **not** run the full `swift test` suite:
  the submitted-path diff is empty, so it would only re-test the unchanged base
  and would consume a slot another student can use. Say so if you want it anyway.
- Exact-token and row-ledger verdict: **not applicable** — no candidate to
  verify. The probe build was reverted before submission.
- Generated-twin audit: not relevant; no Metal source changed.
- Peak RAM or head/artifact size: pinned head 849,400,347 B; declared head
  238,934,129 B; `mtp-head/` in-tree holds only `README.md` (exempt bytes 2,410).
- Official status and score, if submitted: not submitted.

No metric table is reported. There is no candidate build, so every cell would be
either a copy of the baseline or an invention. Per the assignment, a missing
metric is reported as missing.

### Untimed liveness probe

Ran at probe commit `aa4bd9e`, `dirty=0`, 16 decode tokens, worker
`sha256=4aa4cdf6b7910326dd0c0cc91c76467e1b3cd521b8cd964772d48dacf87a40aa`, CLI
`sha256=c8de2266b9444dd0428dd6620c1eb2c8fb15d7255df283248fa7e46e14705005`,
started `2026-08-17T08:52:26Z`, exit 0. Lock was free at 0 s, so `qwen-edward`'s
slot was never contended.

Both probes fired (twice each — once per leg, since the depth-0 and depth-8 legs
each construct the model):

```text
E9R3-TRACE makeCompactDraftHead vocab=248320 draftHeadW_present=false \
  lmHead=QuantizedLinear bits=4 groupSize=64 \
  weightShape=[248320, 640] scalesShape=[248320, 80]
E9R3-TRACE draftTokenID total=1 compact_fused=1 declared_head=0
```

This converts the static inference into measured fact:

- `draftHeadW_present=false` — `_draftHeadW == nil`, so the bypass guard is not
  taken and `makeCompactDraftHead()` is genuinely reached on the scored path.
- `bits=4 groupSize=64` with `weightShape=[248320, 640]` (640 uint32 words =
  5120 × 4 / 32) and `scalesShape=[248320, 80]` (80 = 5120 / 64) — the sliced
  head is affine 4-bit group-64, confirming the width the byte model assumes and
  confirming A2's point that the slice is of the **backbone** head.
- `compact_fused=1 declared_head=0` — every `draftTokenID` call took the fused
  `qwen35DraftSelectKernel` path; the declared-head path was never entered.

Fidelity on both legs: `all_tokens_matched=true`,
`reference_checked_rows=16/16`, `public_drift_tripwire_passed: true`.

The run also printed `seconds_per_token` (0.318164 depth-0, 0.278726 depth-8) and
`accepted_draft_rate=1.0000` at depth 8. **None of that is usable timing or
acceptance evidence** and it is recorded here only so nobody later mistakes it
for a measurement: `MLX_QWEN_MTP_TRACE=1` is required to forward worker stderr at
all, and it also switches on the session's verbose per-position row dump, so the
run is deliberately perturbed. A 16-token window on the public fixture prefix is
also far too short and too easy for acceptance to mean anything.

### Measurement-hygiene finding

On arrival `.build-worker/release/mlxfast-runtime-worker` was **stale**
(Aug 16 23:25) while `.build/release/mlxfast-runtime-worker` was fresh
(Aug 17 08:46). This is the same stale-worker class of defect I disclosed in r2,
and it is a live trap for the next agent: a plain `swift build -c release`
refreshes only `.build/release`, while the benchmark wrapper runs the
`.build-worker` twin. `research/run-e9r3-liveness.sh` builds both with
benchmark.sh's own scratch paths and asserts its probe symbol is present in the
worker binary before trusting a run, so a silent build cannot be reported as
evidence.

Note also that `research/run-draft-bits-arm.sh` now **cannot run on this base**:
its `strings` tripwire at ~:84 requires `MLX_QWEN_MTP_DRAFT_BITS` in the worker
binary, and that symbol no longer exists. It is dead tooling unless the lever is
re-implemented. I left it in place rather than deleting tooling I was not
assigned to prune.

## Conclusion

- What happened and why: the promoted-frontier rebase deleted the mechanism this
  experiment was assigned to tune. A1 failed at its first gate, and the stop rule
  ended the experiment there with zero timing budget spent.
- Evidence for or against the mechanism: the byte model caps a 3-bit readout at
  −6.05% of a head step locally / −13.13% on the ranked path, before paying for
  13 blocked dequant/requant `eval()`s. r2's measured per-round effect was
  **+0.450% slower**, with a mechanism ceiling of only −0.683%.
- **The r2 gain was luck, and the coin is re-rolled.** r2's −2.03% decomposed
  into a per-round regression plus a −2.469% round-count win driven by acceptance
  moving 0.9190 → 0.9393. Acceptance is re-rolled by any change to the schedule,
  and this base's schedule differs materially from r2's (`headStepCostRatio`,
  `sdpaWidthWallDepthCap = 5`, `segmentedVerifyDepthCap = 8`,
  `segmentedStreakGate = 3`, greedy marginal `costModelDepth` with
  `positionAcceptEMA` and top-2-margin confidence caps). r2's cap-7 floors
  (`0.911518`) are therefore **invalid here**. Directional agreement across
  bases would not be replication, and I am not claiming any.
- Prompt or M5 transfer risk: moot for this result. Recorded for reuse: the
  local-vs-ranked head asymmetry above means a locally-measured readout change
  transfers *better* than the local ratio suggests, not worse.
- Smallest useful next action: none for this mechanism. If someone wants to
  reopen it, the decisive cheap test is a standalone `qmv` micro-benchmark of
  98,336 × 5,120 at 3 vs 4 bits *plus* the blocked requant cost, compared against
  the −6.05% ceiling — before writing any session code.
- **Recommendation: close.** Do not restore the lever. It needs a
  re-implementation, its ceiling is small, its measured per-round effect was a
  regression, and the only reason it ever looked good was an acceptance coin flip
  that this base re-rolls.

## Suggested follow-ups (not implemented, not mine to take)

- `headStepCostRatio = 0.18` is hard-coded, and a cheaper readout would lower the
  true `h` it approximates — so any future readout win silently mis-calibrates the
  depth chooser. Owner: `qwen-edward`.
- Route B proper — a manifest head that actually ships a `draft_lm_head` tensor —
  would make `_draftHeadW` non-nil and bypass `makeCompactDraftHead()` entirely.
  That is a head-artifact question, not a `Qwen35.swift` question, and it is the
  only version of this idea that avoids double quantization.
- The in-tree "~315 MB / ~0.6 ms per draft step" comment does not match the
  283.2 MB / 1.2469 ms byte model here; one of the two is wrong.
