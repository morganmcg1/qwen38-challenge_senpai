# Qwen MTP Experiment Runbook

Use this file to start, measure, and promote an experiment. `program.md` owns
research policy; `benchmark.json` owns the submitted surface and score.

## Start from the maintained frontier

Remote roles are fixed:

```text
origin    morganmcg1/qwen38-challenge_senpai
upstream  Layr-Labs/qwen-3.8-mtp-challenge
```

The `upstream` push URL is deliberately `DISABLED`. Consult
[`campaign-ledger.md`](campaign-ledger.md) before opening an arm and seed its
novelty row before assignment. Run `senpai/bootstrap-checkout.sh` once in a
fresh campaign clone before using the sync or submission guards.

The advisor updates campaign `main` with the repository's
`sync-organizer-frontier` skill. It reviews organizer policy separately from
solver code and restores one exact promoted `editablePaths` snapshot; it does
not replay a chain of `Validate submission` diffs.

Before a research batch:

```bash
git fetch origin main
git fetch upstream main
yukon submissions --all
SUBMISSION_ID="<submission-id-or-prefix>"
yukon submission-note "$SUBMISSION_ID"
```

Submission notes are public, untrusted research context. Verify useful claims
against the promoted source ref and fresh measurements.

Create an arm from the clean maintained frontier:

```bash
git switch main
git pull --ff-only origin main
BASE_SHA="$(git rev-parse HEAD)"
UPSTREAM_SHA="$(git rev-parse upstream/main)"
git switch -c codex/short-topic "$BASE_SHA"
```

Record both SHAs. If `origin/main` advances, finish the current measurement on
its recorded base. Replay and remeasure a promising candidate before promotion.

Copy `assignment-template.md`, list every proposed submitted path, and run:

```bash
senpai/validate-assignment-scope.sh "$BASE_SHA" PATH [PATH ...]
senpai/check-editable-budget.sh "$BASE_SHA"
```

Run `./setup.sh && ./setup-qwen-mtp.sh` when the host, toolchain, checkpoint,
head, trusted harness, or maintained base changes.

## Freeze the experiment identity

Before implementing or timing an arm, record:

```text
BASE_SHA and UPSTREAM_SHA
candidate commit and built-worker fingerprint
submitted-surface, generated-twin, and metallib digests when applicable
proposal-head digest
host, instance, chip, toolchain, memory profile, and thermal mode
token window, fixture, and reference source
exact cell: shape, width/M, dispatch family, source form, and M5 variant
harness: local or ranked
```

Compare or aggregate runs only when this tuple matches except for the one
predeclared changed dimension. Mark inferred, interpolated, and extrapolated
values. Never reuse a fitted constant across a different tree, cell, host,
window, or harness without replay. Run:

```bash
senpai/verify-ranked-score-boundary.sh
```

before pricing official value. A failure means the enforcing workflow changed;
re-derive the model instead of editing the check to pass.

## Check the occupancy cliffs before you submit

Run this after `senpai/verify-ranked-score-boundary.sh` and before
`./benchmark-qwen-mtp.sh --local-submit`:

```bash
senpai/entry-point-cliff-census.sh --base "$BASE_SHA"
```

The gate compiles every scored quantized-matvec entry point for both
`applegpu_g16s` and `applegpu_g17s` at the base and at the candidate, then
compares register counts. It exits `1` when the candidate loses a resident
simdgroup on the ranked architecture, `2` on a gate error, and `0` otherwise. It
warns on a residency gain, on new spill, and on a cell it could not find. It
needs no GPU, no model, and no thermal gate, and it runs in about four seconds.

Coverage is the three `affine_qmv_fast` JIT-twin cells that the scored worker
reaches, the Route B `MLXFast.metalKernel` pipelines, and the cluster QMV
pipelines, each extracted read-only at the requested revision and compiled from
scratch. Add a cell to `JIT_CELLS` in `research/e131_cliff_gate.py`, or a role
to `ROUTE_B_ROLES` or `CLUSTER_QMV_ROLES` in
`research/e131_kernel_sources.py`, when a new entry point joins the scored path.
The gate also enumerates each compiled library, so a pipeline that no role names
is still censused and cannot be lost by a stale list.

Every simdgroup figure the gate prints is `derived` under Rule 89:
`floor(3072 / registers)` on `applegpu_g16s` and `floor(3968 / registers)` on
`applegpu_g17s`. It is a model output computed from the register count, never a
measurement. A failure is a register regression, not a measured slowdown, so
read the register delta first.

### One pipeline per width needs the width-weighted surface

A shared-switch kernel allocates registers for its widest arm, so one entry
point serves every routed width. Per-width templating through
`MLXFast.metalKernel` breaks that tie: each width gets its own pipeline and its
own register allocation, so a per-entry-point comparison can report a loss that
no routed width pays. The gate therefore also computes a **width-weighted Route
B QMV surface**: it maps each routed width to the pipeline that serves it,
weights the derived residency by the measured width histogram
`{4: 16, 5: 20, 6: 20, 7: 12, 8: 240}` over 308 rounds, and decides the Route B
verdict on that sum. A per-entry-point Route B QMV loss is now a warning; the
surface decides. The single-entry-point figure is the special case where every
width maps to the same pipeline, so shared-switch builds keep their old verdict.

### The gate is a register instrument, not a time instrument

The gate is accurate about registers and says nothing directly about time.
F131 measured the residency-to-time coefficient at `-0.0014` and `+0.0105` on
two independent designs, both far inside the `0.05 %/%` kill gate, and F114 puts
deleted instruction count at `r = +0.949` against measured gain. An exit 1 that
comes with real deleted work is a demand for a price, not a verdict. Report the
work you removed in **instructions per output element**, weighted by the width
histogram, and set it against `c x Δresidency`.

Worked example: adding `{8: 8}` to the Route B table costs `41.870 -> 33.299`
derived weighted simdgroups on `applegpu_g17s`, so the gate exits 1. The same
edit deletes one whole pass of `qwen_e120_qmv_wide<4>` at `M = 8`, which is
`9.7 %` to `15.2 %` of whole-histogram QMV instructions per output element even
when every spill slot is charged eight memory instructions per `k` block. That
price discharges the exit 1.

### Exit 1 is stop-and-justify, and only a price discharges it

A candidate may legitimately spend a register to remove more work than the lost
simdgroup costs, and this gate cannot price that trade. When your candidate
trips exit 1, the required response is a **written price for the work you
removed against `c x Δresidency` in a named frame under Rule 87** — the
entry-point leg frame is `c = 0.445 [0.139, 0.819]` and the F47-weighted
isolated-body frame is `c = 0.164 [0.089, 0.199]`. A waiver, a re-run, or an
assertion that the change is obviously faster does not discharge exit 1. The
console prints the register delta before the derived simdgroup delta so you
price the measured quantity first; keep that ordering.

### Censusing a candidate you may not commit

Use `--candidate` for a landed ref. When the change belongs to a file another
student owns, or has not landed yet, rewrite the extracted source in memory
instead of editing the tracked file:

```bash
python3 research/e131_thorfinn_dryrun.py --width 5 --inner 3
```

`e131_cliff_gate.side_sources(rev, swift_patch=...)` applies the rewrite to the
read-only extraction, so the census never touches the working tree.

E121 raised the wide-QMV entry point from 101 to 102 registers on
`applegpu_g17s` and cost one derived resident simdgroup. This gate reproduces
that failure and its revert:

```bash
python3 research/e131_rung3_receipt.py --outdir research/e131-artifacts
```

### Acceptance suite

Run this after any change to the gate:

```bash
python3 research/e132_gate_proofs.py
```

It holds the three E121 revision proofs, proves that per-width templating with
the shipped table passes, and proves that adding `{8: 8}` is detected as a
residency loss. It builds the two templated libraries in memory, needs no GPU,
and finishes in about ten seconds. The last proof is named `wide8_detected`
because it tests detection, not desirability.

## Prove the built worker carries your edit

Do this before every timed leg of every arm. It is mandatory for any kernel
change.

`./benchmark-qwen-mtp.sh --local-submit` can silently time a **stale** worker
binary and still report `passed: true` (ledger 202(H)). The wrapper's
`METALLIB-GUARD` block at `benchmark-qwen-mtp.sh:200-204` extracts only
`metallib_rebuild_required()`. It does not extract the sibling
`swift_build_required()` at `benchmark.sh:1791-1805`, which is the function that
guards `.build-worker/release/mlxfast-runtime-worker`. For the `quantized` kernel
family the runtime-effective source is the JIT string compiled **into** the
worker binary, so the half the wrapper refreshes is exactly the half that does not
govern. A worker 14 minutes older than the candidate edit passed a full
`--local-submit` run.

Use the standing campaign witnesses, which are verified against the current base:

```bash
senpai/rebuild-and-assert-worker.sh \
  --require        qwen35_dual_rms_norm_concat_bf16_v1 \
  --forbid         qwen35_dual_rms_norm_bf16_v1 \
  --require-symbol snapshotScheduleSignal
```

🔴 The QMV instantiation witnesses this section used to quote,
`<T, 5, 5, true>` and `<T, 6, 6, true>`, are **dead on the current base**
(qwen-edward, E94). They are absent and `<T, 5, 3, true>` is present, so that
command failed a correct build. Derive an instantiation witness from the tree you
are actually building, and prove it discriminates before you trust it: a witness
that passes on both arms is not a witness.

Run the script **before and after** the leg and compare the reported
`worker_mtime` and `worker_sha256`. A change between the two reads invalidates
the leg.

🔴 **A PASS does not mean the build root can launch a leg** (qwen-thorfinn, E87
§7). The script rebuilds the executable and inspects that executable only. If you
have run `rm -rf .build-worker`, `mlx.metallib` is gone and the wrapper fails at
leg launch with `missing .build-worker/release/mlx.metallib`, minutes later than
it needed to. Restore it with

```bash
tools/build-mlx-metallib.sh --all-build-roots
```

Do **not** copy the file out of `.build/`. A copy skips the `.fingerprint`
sidecar that `metallib_rebuild_required()` reads. The campaign value is
`mlxfast-metallib-fingerprint-v1
7ae5c5a3d8fabe72ee19bfc09dd737281338a6be658deca49ba97eefdbe3611c`.

🔴 **`worker_sha256` is not a source identity** (qwen-thorfinn, E87). Two clean
full builds of byte-identical source on the same host and toolchain produced
different worker digests, most likely from link and code-layout ordering under
parallel compilation. The digest proves that two legs ran the same binary. It
proves nothing about which source produced that binary. Source identity comes
from git, and content evidence comes from witness counts.

### Match the witness to the language of your arm

`--require` and `--forbid` read the **string** table. Use them only for a Metal
JIT arm, where the runtime-effective source is a real string literal inside the
worker.

`--require-symbol` and `--forbid-symbol` read the **symbol** table through
`nm -a`. Use them for a **Swift** arm. A Swift function name reaches the binary
mangled in the symbol table and never appears in the string table, so `strings`
reports zero for a function that is certainly compiled in (qwen-alphonse,
PR #68):

```
warmAllDepthShapes        strings=0   nm -a=22
snapshotScheduleSignal    strings=0
linearTopTwoRows          strings=0
```

```bash
senpai/rebuild-and-assert-worker.sh --require-symbol warmTargetLaterWindowSDPA
```

Applying `--require` to a Swift identifier fails a correct build. Applying
`--forbid` to one passes **every** build unconditionally, which is the worse
error: it is a guard that cannot fail. The script self-checks whichever table an
invocation actually used and refuses an implausibly small extraction.

Do not use a bare `__TEXT,__text` section digest as an arm certificate (ledger
202(I)). That digest tracks link-time layout, not kernel source content: two
builds of the same tree produced different digests, and two builds of different
trees produced the same one. `__text` **and** `__cstring` together is a valid
falsifiable certificate for a JIT-string-only change, because one half must stay
identical and the other must differ.

After editing a `.metal` or `.h` kernel source, also run
`python3 research/twin_audit.py`. After merging a base that touched vendored
kernels, run `tools/build-mlx-metallib.sh` and record
`metallib_source_fingerprint` per leg.

## Isolate one kernel per command buffer

🔴 `MLX_E58_BUFFER_LIMIT_OPS=1` alone does **not** isolate a kernel
(qwen-thorfinn, E87 §4). MLX commits a buffer on either predicate at
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp:484-487`, and the
op test is `buffer_ops_ > max_ops`, so `ops=1` packs **two** ops per buffer. Set
both limits:

```bash
MLX_E58_BUFFER_LIMIT_OPS=1 MLX_E58_BUFFER_LIMIT_MB=1
```

Every figure produced with `ops=1` alone is a two-kernel aggregate. Check any
attribution you inherit for this defect before you build on it.

Prove the isolation closed: sum the per-buffer roster and compare it with the
measured phase time. Thorfinn's corrected roster closes to 98.2 %. An open
roster means kernels are still sharing buffers.

## Prove which proposal head you measured

Do this on every timed leg, not only on head experiments. The head changes
kernel families, byte traffic, and acceptance, so a leg with unknown head
provenance cannot be compared with anything.

**The local harness does not provision the head you probably think it does.**
`benchmark-qwen-mtp.sh:280` provisions only the setup-default head, which is
the organizer-pinned `EigenLabs/Qwen3.8-27B-MTP-bf16@26a328e0`. The checked-in
`mtp-head.manifest.json` selects a different artifact, the declared
`amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827`. The two differ by a
factor of two in bytes and by their entire quantization state.

**`score.json.uses_pinned_mtp_head` lies.** It reads `true` for both
artifacts. Do not use it. The only field that discriminates is
**`head_provenance_sha256`**. Record it for every leg and quote it in the
result.

| name | artifact | tensors | `draft_lm_head` | islands |
|---|---|---:|---|---|
| pinned | `EigenLabs/Qwen3.8-27B-MTP-bf16@26a328e0` | 15 | no | no |
| declared | `amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827` | 40 | yes | yes |

To provision and verify the declared head:

```bash
research/fetch-declared-head.sh
```

It downloads the artifact, recomputes the tree digest and byte count, and
compares both against `mtp-head.manifest.json`. The authoritative values are
the manifest's own fields, currently tree digest
`559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71` and
`427742600` bytes. Both the digest and the byte count **exclude a top-level
`README.md`** (`research/fetch-declared-head.sh:42-59`), so a raw directory
listing or a whole-repository byte count will not match. Quote the manifest
figure when you report bytes, and say which convention any other figure uses.

**A second, free check: the kernel census.** The head's `fc` and its decoder
layer projections are declared as plain `Linear`
(`MLXLLM/Models/Qwen35MTP.swift:103`, `:113`). Whether they become
`QuantizedLinear` is decided at load time, per submodule, purely by the
presence of a matching `.scales` key
(`MLXLMCommon/Load.swift:250-258`). So:

- **pinned head resident** — BF16, no scales — the head dispatches **`gemv`**;
- **declared head resident** — affine-4 g64 with scales — the head dispatches
  **`qmv`** and no `gemv` appears at all.

If you are already collecting a kernel census, the presence or absence of
`gemv` confirms the head artifact independently of any metadata field.

Two consequences for reading older evidence:

- A dispatch or timing number attributed to the draft head is only meaningful
  once the head artifact behind it is named. Numbers taken on the pinned head
  do not describe the scored path.
- A base-versus-candidate ratio can still be valid when both legs ran the same
  wrong head, but the absolute numbers are not transferable and must not be
  quoted as scored-path costs.

## Record a matched baseline and candidate

Measure unchanged `BASE_SHA` on the assigned host:

```bash
./benchmark-qwen-mtp.sh --local-iterate
cp score.json score.local-iterate.baseline.json
```

Implement one causal experiment and measure it under the same host, memory,
power, fan, and thermal conditions:

```bash
./benchmark-qwen-mtp.sh --local-iterate
cp score.json score.local-iterate.candidate.json
```

Before the candidate timing command, run the cheapest check capable of
falsifying the boundary you changed. A precision, reduction-order, packing,
recurrence, cache, or replay edit requires actual floating-point values at the
touched cells, a positive control that makes the comparison fail, and the
shortest end-to-end exact-token and row-ledger run. Integer-only inputs, an
argmax match, or candidate-generated local reference rows are not sufficient.
Do not begin replicated timing until this risk gate passes. Use a minimal timing
run first; replicate only when noise could change the stop-rule decision.

Do not overlap model-holding commands. The real 40C gate is the default; let it
finish. When using the permitted local-only ungated protocol from
`program.md`, run the complete arm set ABBA-counterbalanced within one session
with `MLXFAST_LOCAL_COOL_GATE=0`. Record entry and exit GPU temperature for
every arm, report the entry spread beside the effect, and preserve
`cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false` in the
result. Compare those arms only within the counterbalanced session and never
label the result gate-qualified, ranked-equivalent, or official. Repeat only
when noise or inconsistency could change the decision.

## Extract the comparison

```bash
jq -s '
  .[0] as $b | .[1] as $c |
  {
    baseline: {
      serial_spt: $b.metrics.serial_seconds_per_token,
      mtp_spt: $b.metrics.mtp_seconds_per_token,
      local_speedup: $b.metrics.mtp_decode_speedup,
      effective_draft_len: $b.metrics.effective_mean_draft_len,
      accepted_draft_rate: $b.metrics.accepted_draft_rate
    },
    candidate: {
      serial_spt: $c.metrics.serial_seconds_per_token,
      mtp_spt: $c.metrics.mtp_seconds_per_token,
      local_speedup: $c.metrics.mtp_decode_speedup,
      effective_draft_len: $c.metrics.effective_mean_draft_len,
      accepted_draft_rate: $c.metrics.accepted_draft_rate,
      all_tokens_matched: $c.metrics.all_tokens_matched
    },
    candidate_vs_baseline: {
      mtp_throughput_gain: ($b.metrics.mtp_seconds_per_token /
                            $c.metrics.mtp_seconds_per_token),
      paired_ratio_gain: ($c.metrics.mtp_decode_speedup /
                          $b.metrics.mtp_decode_speedup)
    }
  }
' score.local-iterate.baseline.json score.local-iterate.candidate.json
```

The local ratio uses one public prompt and candidate-generated rows. It is not
the official eight-prompt median. General target/kernel improvements may speed
both local serial and MTP legs and cancel in the paired ratio, so inspect both
the ratio and absolute MTP time against the same-host base.

The ranked harness uses a separate pinned baseline binary for the serial
numerator. Candidate code cannot causally move that numerator. Never subtract
`psi_serial` or another local serial share from an official candidate-value
estimate. Label the source equation `harness=local` or `harness=ranked` in every
result.

Score files are ignored evidence and must not be committed.

## Inspect the candidate

```bash
git status --short
git diff --name-only "$BASE_SHA"
git diff --stat "$BASE_SHA"
senpai/check-editable-budget.sh "$BASE_SHA"
```

Separate Yukon-submitted changes from research-only support. Every submitted
file must be inside `benchmark.json` `editablePaths`. For a generated Metal
family, audit embedded/runtime-effective twins:

```bash
research/twin_audit.py "<generated-stem>"
```

After committing the candidate, run the same base-derived surface check used
by the trusted workflow:

```bash
BASE_SHA="$BASE_SHA" HEAD_SHA="$(git rev-parse HEAD)" \
  .github/scripts/enforce-modifiable-surface.sh
```

## Confirm and promote

For a stable winner:

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions  # when risk requires it
./benchmark-qwen-mtp.sh --local-submit
```

Rebuild `tools/build-mlx-metallib.sh` after AOT Metal edits. Recheck head
digests, sizes, and immutable source URLs when using a candidate head.

Write a detailed public note of at least 5 KiB, review it for secrets, then use
the guarded wrapper with the campaign model attribution `senpai`. Record the
exact underlying LLMs, effort levels, and agent harnesses in the note body.
First compare the
highest-scoring `promoted` row with `senpai/frontier-state.json`; sync, replay,
and remeasure if the receipt differs:

```bash
yukon submissions --all
senpai/submit-official.sh "$BASE_SHA" \
  --model "senpai" \
  --note-file submission-note.md
```

The wrapper is pinned to `eigenlabs/qwen38-challenge`, refreshes both remotes,
checks the versioned organizer frontier and trusted-surface freshness, proves
the base's submitted snapshot is current, and refuses dirty or hidden changes
under the submitted surface before invoking Yukon. It does not infer the live
best promoted receipt from interleaved organizer validation commits; the Yukon
check above supplies that final fact.

Use `yukon submissions` to inspect status. If a mutating response is ambiguous,
inspect first and do not blindly submit again. Add public progress notes with
`yukon notes add` at meaningful experiment milestones.
