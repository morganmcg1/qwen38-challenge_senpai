<!-- SENPAI-CAMPAIGN-BEGIN -->
# Qwen 3.8 Native-MTP Challenge Agent Guide

This repository has one ranked task: make Qwen 3.8 27B native-MTP decode
faster on Apple Silicon without changing a single emitted token. The ranked
track is `qwen3.8-27b-mtp-v1`.

Retained files from earlier tracks are historical prior art only. Their
commands, model geometry, editable surface, correctness rules, scoring, and
runner configuration do not apply here. Operate this repository only as the
Qwen track named above.

## Senpai Campaign

For coordinated research in `morganmcg1/qwen38-challenge_senpai`, read
[`senpai/program.md`](senpai/program.md) and
[`senpai/experiment-runbook.md`](senpai/experiment-runbook.md) before assigning
or running experiments. The campaign ledger and exact frontier pins are in
[`senpai/campaign-ledger.md`](senpai/campaign-ledger.md) and
[`senpai/frontier-state.json`](senpai/frontier-state.json).

The intended remotes are:

```text
origin    https://github.com/morganmcg1/qwen38-challenge_senpai.git
upstream  https://github.com/Layr-Labs/qwen-3.8-mtp-challenge
```

`upstream` must have push URL `DISABLED`. Never run `yukon sync`,
`yukon sync --harness-only`, or `yukon reset` in this maintained checkout;
those commands are for plain solver checkouts and can reset campaign history or
repoint `origin`. Use the repository-local `sync-organizer-frontier` skill for
organizer policy and promoted-snapshot refreshes.

Record the exact campaign `BASE_SHA` and organizer `UPSTREAM_SHA` for every
experiment. Preserve unrelated user work, campaign files, and dirty-worktree
changes. Generated weights, caches, score files, downloaded artifacts, and
hidden material never belong in Git.

## Authority Order

This repository still contains historical comments and bring-up prose. Resolve
contract questions from the live enforcing sources in this order:

1. [`benchmark.json`](benchmark.json): track identity, commands, editable
   paths, optional paths, budget policy, and scoring fields.
2. [`fixtures/qwen3_8_27b_mtp_track.json`](fixtures/qwen3_8_27b_mtp_track.json):
   artifact pins, timed prompt pool, fidelity semantics, and calibration.
3. [`.github/workflows/qwen-mtp-ranked-benchmark.yml`](.github/workflows/qwen-mtp-ranked-benchmark.yml):
   ranked execution, runner label, and enforced environment values.
4. [`docs/qwen-mtp-editable-surface.md`](docs/qwen-mtp-editable-surface.md):
   rationale for the editable/trusted boundary.
5. [`TASK.md`](TASK.md): concise participant-facing summary.

For mutable solver behavior, inspect the current `BASE_SHA` source and
`mtp-head.manifest.json`. Narrative descriptions of the original starting tree
do not override the current implementation. If authorities disagree, stop and
identify the exact enforcing site before editing or assigning work.

## Mission And Score

The official score is a decode-only paired speedup anchored at serial `1.0`:

```text
for each hidden prompt p:
  raw_p = mean(pinned-baseline depth-0 serial seconds/token)
          / mean(candidate seconds/token)

published score = median(raw_1 ... raw_8)
```

All eight hidden prompts are measured. Because eight is even, the median is the
mean of the two central ordered values. There is no no-op normalization and no
prompt lottery. The published median must be at least `0.90` and no greater
than the `5.0` plausibility ceiling.

Each ranked leg processes a 512-token seed followed by 512 parent-counted
decode tokens, and the seed work is charged in the timed leg. Prefill is not a
separate score, but work removed from the charged seed still improves the
result. The ranked run uses one thermally gated serial/candidate pair per
prompt unless the trusted contract changes.

The trusted parent offers a per-round ceiling. Candidate policy may choose
from 0 through 8 drafts independently on every round. Drafting nothing is
legal and degenerates to serial decode; it is a control, not the optimization
goal. Acceptance rate alone is not the objective: deeper drafting wins only
when extra committed tokens repay head, target-verification, rejected-tail,
and state-management costs across varied prompts.

The organizer's original calibrated tree scored about `0.994`. That is a
historical starting-line measurement, not the current campaign frontier. Read
the live Yukon promoted row and `senpai/frontier-state.json` before comparing or
submitting work.

## Target And Current Scored Path

The fixed target is the organizer's MLX 4-bit affine/group-64 conversion of
Qwen 3.8 27B:

```text
backbone  EigenLabs/Qwen3.8-27B-4bit
          @ eda45ab47f465d08d6558f0353a2346e2eb9d5b3
upstream  Qwen/Qwen3.8-27B
          @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

The organizer-pinned default proposal head is the separate bf16 MTP
artifact. A candidate may instead declare an immutable remote or in-branch head
through `mtp-head.manifest.json`; inspect that file at `BASE_SHA` for current
head provenance. The target checkpoint remains fixed regardless of the head.

The Qwen tower is a dense hybrid architecture:

- 64 layers: 48 Gated DeltaNet recurrent layers and 16 full-attention layers;
- hidden size 5,120 and vocabulary 248,320;
- full attention with 24 query heads, 4 KV heads, and head dimension 256;
- affine 4-bit/group-64 backbone weights and dense MLPs.

The `Qwen35*` source names and `qwen3_5_text` configuration label describe the
architecture family embedded in the Qwen 3.8 artifact; they do not mean this
repository targets Qwen 3.5 or 3.6.

The scored worker loads the transformed checkpoint through `LLMModelFactory`,
attaches the selected head, warms legal shapes, and serves rounds through
[`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`](Sources/MLXFastModel/Qwen36MTPBlockSession.swift).
A round conceptually:

1. commits the pending primary token;
2. lets candidate policy choose 0...8 drafts within the offered maximum;
3. runs the MTP head chain to propose them;
4. verifies `[primary] + drafts` with the fixed target in a batched forward;
5. accepts the longest correct prefix;
6. keeps, restores, replays, or repairs target and head state as required;
7. reports actual rows and exact top-two evidence to the trusted parent.

The timed target model is the vendored `Qwen35TextModel`/`Qwen35Model` in
[`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift),
together with the editable MTP session. Directory-wide editability does not
prove reachability: for example, the separate `Qwen35FastEngine` path is not
the MTP worker's current target path. Trace the call path before optimizing.

### Current frontier is already optimized

Do not treat historical defaults as current behavior or rediscover mechanisms
already in the promoted snapshot. At this guide's 2026-08-16 rewrite:

- `Qwen36MTPBlockSession.init` installs an adaptive cost-model policy; the
  nearby K=1 property initializer and constant-depth comments are historical
  fallback/bring-up text, not the instantiated policy;
- the campaign `mtp-head.manifest.json` declares a 4-bit/group-64 proposal
  head rather than using the organizer-pinned bf16 default;
- persistent head history, K/V-only history flush, device-resident drafting,
  exact rollback/replay, and scored-shape warmups are present;
- target verification has exact multi-row affine4 QMV sharing, fused GDN and
  full-attention preparation, and an exact two-stage top-two reducer;
- proposal selection already uses a compact vocabulary and a custom selector;
- the layer loop already carries an `asyncEval` ladder for narrow widths.

These are mutable solver facts. Recheck source after every frontier sync. See
[`senpai/laguna-to-qwen-speedup-map.md`](senpai/laguna-to-qwen-speedup-map.md)
and the campaign novelty ledger before assigning overlapping work.

## Editable And Trusted Boundary

The ownership rule is:

> Candidate code may propose and execute the model, but the trusted parent owns
> correctness, measurement, and final ledger authority.

The editable session necessarily assembles target verify forwards, walks the
candidate-side accepted prefix, rolls state back, and produces evidence for the
parent. None of those self-reports is authoritative: the parent independently
checks the serial token stream, round structure, rows, and accounting.

`benchmark.json` `editablePaths` is the only machine-readable path authority.
Its current 89 entries form five practical groups:

- optional `mtp-head.manifest.json` and `mtp-head/` candidate-head content;
- `Sources/MLXFastModel/`, including the runtime and full MTP apparatus;
- `Sources/MLXFastTransform/` offline checkpoint transform;
- the exact listed Qwen model and `MLXLMCommon` support files;
- the exact listed MLX/Metal kernel sources and generated twins.

Do not infer that an adjacent file is editable. Check the manifest at
`BASE_SHA` before implementation. Submission archives use replace semantics
over `editablePaths`; only the two MTP-head paths are optional. A support-file
change outside the surface can make a local tree faster while uploading
nothing useful.

Trusted and not submitted include:

- target weights, transform contract, tokenizer, fixtures, and goldens;
- trusted driver, accept audit, row accounting, score emission, and worker
  protocol;
- timing, telemetry, thermal gate, workflows, and `.github/scripts/`;
- `benchmark.json`, `Package.swift`, and `Package.resolved`;
- tests, docs, local benchmark scripts, and other paths not named by the
  manifest.

`Sources/MLXFastModel/Qwen36MTPReferenceSession.swift` is an important mixed
case: it is editable through the directory-wide allowance even though it serves
rejected-tail reference replay. It runs after timing and cannot define emitted
tokens. Never route its reference work or any golden-derived information into
the timed path, proposal policy, or emitted stream.

The source budget is 3,000,000 bytes total, 524,288 bytes per file, and
262,144 bytes of candidate growth. `mtp-head/` is exempt from the source budget
and instead has a declared digest, byte count, and 2 GiB cap. Run the campaign
scope and budget checks before expensive validation.

## Correctness And Prohibited Strategies

Correctness is a hard gate. Every emitted token must equal the hidden serial
trajectory. The round journal must be structurally sound, effective draft
counts must match what occurred, and the reference-checked row ledger must
close. There is no near-tie allowance on this track.

The ranked pipeline also runs the public drift tripwire, hidden teacher-forced
base, anchor, free-run, behavior and GPQA gates, GPQA TTFT guardrail, semantic
judge, transform verification, static review, and integrity checks. Any failed
gate produces no promotable score.

The native MTP path inside the declared editable surface is the only sanctioned
future-token mechanism. Do not add:

- prompt lookup, n-gram, suffix, or token-history drafting;
- same-target lookahead or unsanctioned future-logit/KV computation;
- request-keyed replay caches whose only useful hit is a repeated benchmark
  request;
- cross-request future state, hidden-prompt specialization, or GPQA shortcuts;
- degraded or partial target forwards, unevaluated verify rows, omitted exact
  final-logit/top-two work for emitted rows, or fabricated top-two, row,
  acceptance, or ledger evidence;
- an undeclared auxiliary proposal model/head or candidate-selected target;
- phase, reference-leg, prompt-pool, or baseline detection used to change the
  work performed;
- logical rollback that leaves rejected cache/state rows reachable, or merely
  decrements an offset without trimming or overwriting rejected physical rows;
- timing shortcuts, protocol injection, network access, or filesystem
  exfiltration.

Input-independent weight/kernel/shape tables and ordinary within-request cache
reuse are legal when they preserve the contract. A candidate head may improve
proposal quality and cost; it cannot redefine the target answer.

Public fixtures were generated on M5. On another Apple GPU, compare an
unchanged `BASE_SHA` at the same token position before diagnosing a near-tie
local mismatch. That diagnostic does not relax official exactness.

## Kernel And Swift Rules

The vendored MLX package has two effective source forms:

- A family with an `mlx-generated/*.cpp` twin is JIT-compiled from the source
  string embedded in that generated C++ file. Update the readable
  `.metal`/`.h` source and the runtime-effective twin together.
- AOT-only families are loaded from `mlx.metallib`; rebuild it with
  `tools/build-mlx-metallib.sh` after relevant edits.
- `_nax` variants are selected on ranked M5 hardware and are first-class
  targets, not optional follow-up work.

Run `python3 research/twin_audit.py` after JIT-source edits. Before any kernel
experiment, prove the scored shape, dispatch family, source form, and M5
variant. Reduction order, precision, BF16 cast boundaries, packing, recurrent
state, cache layout, and width-dependent dispatch are correctness-sensitive.

For direct Swift commands, preserve the frozen dependency graph:

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
swift build -c release --force-resolved-versions
tools/build-mlx-metallib.sh
```

A bare Swift build may write `.build/release` without rebuilding the scored
`.build-worker/release/mlxfast-runtime-worker`. Use the Qwen benchmark wrapper
for end-to-end work. Never run `swift package resolve` or `swift package
update`; if `Package.resolved` changes, restore it rather than accepting a new
dependency graph.

## Local Workflow And Measurement

Use only the Qwen entrypoints:

```bash
./setup.sh && ./setup-qwen-mtp.sh
./benchmark-qwen-mtp.sh --local-iterate
./benchmark-qwen-mtp.sh --local-submit
```

`--local-iterate` defaults to 64 decode tokens and `--local-submit` to 128.
They use one public fixture and cannot reproduce the ranked eight-prompt,
512-token result. Treat local scores as directional.

The local serial and candidate legs use the same local build. A general target
runtime or kernel improvement can speed both sides and cancel in the local
ratio, even though it counts officially because the ranked serial denominator
comes from a pinned baseline workspace. Compare absolute candidate
seconds/token against a fresh, same-host `BASE_SHA` run as well as the local
serial/MTP ratio.

Run only one model-holding process at a time. The wrapper takes a per-user lock
and checks for orphaned workers; direct model commands do not. Inspect reported
PIDs before killing anything, and never disable the guard to work around a
legitimate concurrent run.

The wrapper cools the GPU before each resident measurement. Waiting at the 40C
gate is normal, not a hang, and the gate remains the default. For local timed
arms only, `MLXFAST_LOCAL_COOL_GATE=0` is permitted when all arms are
ABBA-counterbalanced within one session, every arm records entry and exit GPU
temperature with the entry spread reported beside the effect, and the result
retains `cool_gate_passed_real_gate=false` plus
`gate_qualified_for_timing=false`. Such evidence is explicitly ungated and
directional; compare it within that counterbalanced session, not as though it
were a gate-qualified or official result. Never fabricate a temperature source,
edit the gate constant, or bypass the process lock. Keep host, power, thermal
state, memory profile, head provenance, toolchain, token window, and frontier
SHA matched between base and candidate. Ranked authority is the self-hosted M5
runner labeled `m5-qwen38-27b-mtp`; non-M5 timing is directional only after
confirming the same kernel family executes.

## Experiment Discipline

One experiment should answer one causal question. For each arm:

1. Record `BASE_SHA`, `UPSTREAM_SHA`, host, toolchain, memory profile, and head
   provenance.
2. Prove reachability from the native-MTP worker and name the cost removed.
3. Check editable scope and byte budget.
4. Measure an unchanged same-host baseline and save its artifact outside the
   submission surface.
5. Implement one mechanism and run the cheapest decisive correctness check.
6. Measure the candidate under matched conditions.
7. Compare absolute candidate time, paired ratio, draft/accept telemetry,
   rollback, and block latency where relevant.
8. Reject, revise, or advance using a predeclared stop rule.
9. Record positive, negative, invalid, and ambiguous results in the campaign
   ledger so later agents do not repeat them.

A microbenchmark is useful only when it bounds a named end-to-end cost.
Dispatch-count reduction is not a result unless it removes bytes, arithmetic,
or a dependency. A local pass on one easy prompt does not prove an adaptive
policy is robust across the hidden pool. Compose only independently measured
winners.

Promising current cost centers include draft scheduling, proposal-head cost and
quality, persistent head history, target verify width, GDN recurrence and
rollback, full-attention/GQA cache work, affine4 projection and vocabulary
readout, MLX scheduling/materialization, warmup, and transformed weight
metadata. Start from current profiles and the novelty ledger, not this list.

## Submission And Promotion

Yukon manages accounts and official submissions. It does not run the local
pre-submit command for you. In the maintained campaign checkout, use the guard:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
yukon submissions --all
senpai/submit-official.sh "$BASE_SHA" \
  --model "senpai" \
  --note-file submission-note.md
yukon submissions
```

Use the exact lowercase campaign attribution `senpai`; record the underlying
models, effort levels, and agent harnesses in the public note body.

Immediately before submission, compare the highest-scoring live `promoted` row
with `senpai/frontier-state.json`. If the receipt, source ref, or score differs,
sync the organizer/promoted frontier, replay the candidate, and remeasure it.
Do not infer promotion from a `Validate submission` commit or a branch name.

Submit from a clean, committed candidate after `--local-submit`, exact diff
inspection, scope/budget checks, and generated-twin audit where relevant. Never
put credentials in notes, logs, commits, or agent messages. The ranked resource
is serialized; do not dispatch duplicate submissions to mine timing noise.

Only the official M5 result can promote a candidate. A green rejected receipt
may simply have failed to beat the current frontier; read its gates and metrics
before classifying it as incorrect.

## Historical Prior Art

Laguna work is useful only as a source of hypotheses about shared MLX concepts.
The curated transfer audit is
[`senpai/laguna-to-qwen-speedup-map.md`](senpai/laguna-to-qwen-speedup-map.md).
Do not run Laguna/DFlash setup or benchmark scripts, use their score formulas,
or copy their MoE/NVFP4 kernels without a Qwen-specific call-path and geometry
proof.
<!-- SENPAI-CAMPAIGN-END -->
