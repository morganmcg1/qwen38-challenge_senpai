# Qwen 3.8 27B Native-MTP Research Program

This document explains how advisor and research agents should run the Qwen speedup campaign in this repository. The goal is to make Qwen 3.8 27B native-MTP decoding faster on Apple Silicon without changing the tokens produced by the serial model or breaking the benchmark rules.

The only ranked track here is `qwen3.8-27b-mtp-v1`. Retained Laguna and DFlash files are prior art, not instructions for this challenge.

## Goal And Score

Maximize the official decode score:

```text
for each hidden prompt p:
  raw_p = mean(pinned serial depth-0 seconds/token)
          / mean(candidate seconds/token)

published score = median(raw_1 ... raw_8)
```

The ranked run measures all eight hidden prompts. With eight values, the median is the mean of the two middle values after sorting. There is no no-op normalization. Serial decoding is `1.0`, the minimum published score is `0.90`, and the plausibility ceiling is `3.0`.

Each leg starts with a 512-token seed and then generates 512 tokens that the trusted parent counts. Both seed processing and decoding are included in the same timed leg, even though prefill has no separate score.

For each prompt, the pinned serial build and candidate run as a thermally gated pair in alternating order. The current ranked setup accepts one pair per prompt, for eight pairs and sixteen timed phases in total.

On each round, the candidate may choose any draft count from zero up to the limit offered by the parent, with an absolute maximum of eight. Choosing zero is a useful serial control, but the goal is a real score above `1.0`: any drafting and state-management work must repay its own cost.

The organizer's original calibrated depth-2 tree scored about `0.994`. That is a historical starting point, not the current campaign frontier. A research round or one promoted result is a checkpoint, not a reason to stop. Continue until the operator stops the campaign or no safe, distinct, runnable experiment remains.

## Sources Of Truth

Use the live enforcing sources in this order:

1. [`benchmark.json`](../benchmark.json) defines the track, commands, editable paths, optional paths, budget policy, and scoring fields. `.github/scripts/run-submission-static-review.sh` enforces the source limits, and the trusted worker enforces the limits that also apply at runtime.
2. [`fixtures/qwen3_8_27b_mtp_track.json`](../fixtures/qwen3_8_27b_mtp_track.json) defines artifact pins, the prompt pool, fidelity rules, and calibration evidence.
3. [`.github/workflows/qwen-mtp-ranked-benchmark.yml`](../.github/workflows/qwen-mtp-ranked-benchmark.yml) defines the ranked job, runner, and environment.
4. [`docs/qwen-mtp-editable-surface.md`](../docs/qwen-mtp-editable-surface.md) explains why each area is editable or trusted.
5. [`TASK.md`](../TASK.md) is the short participant-facing summary.

[`AGENTS.md`](../AGENTS.md) contains repository and machine rules. This file adds campaign research practice. If prose disagrees with enforcement, stop and inspect the enforcing source before changing code or assigning work. Never import a Laguna rule because it appears in retained files.

The main campaign records are [`frontier-state.json`](frontier-state.json) and [`campaign-ledger.md`](campaign-ledger.md). Commands and report formats live in [`experiment-runbook.md`](experiment-runbook.md), [`assignment-template.md`](assignment-template.md), and [`result-template.md`](result-template.md). Use [`laguna-to-qwen-speedup-map.md`](laguna-to-qwen-speedup-map.md) as a reviewed source of hypotheses, not as a Qwen contract.

For historical background on the competition's progress, see the timestamped [`Qwen 3.8 MLX.fast submission audit`](qwen38-yukon-submissions-2026-08-16.md). It ties every public receipt at its cutoff to inspected source changes and records promoted, rejected, failed, and pending work. Treat it as a point-in-time research record; Yukon and `frontier-state.json` remain the authorities for live status.

For broader LLM optimization guidance, see W&B Senpai's [Large Language Model Inference Optimization Guide](https://github.com/wandb/senpai/blob/main/literature_and_guidance/LLM-INFERENCE-OPTIMIZATION-SENPAI-GUIDE.md). Treat it as a source of ideas and research methods, not as a substitute for this challenge's live contract or measurements.

## Campaign State And Git Safety

Use these remotes:

```text
origin    https://github.com/morganmcg1/qwen38-challenge_senpai.git
upstream  https://github.com/Layr-Labs/qwen-3.8-mtp-challenge
```

Set the `upstream` push URL to `DISABLED`. A fresh clone can configure the remotes and Yukon link by running:

```bash
senpai/bootstrap-checkout.sh
```

`upstream/main` is the organizer's latest source and policy. `BASE_SHA` is the exact campaign commit an experiment must beat. `UPSTREAM_SHA` is the organizer commit that supplied the trusted contract for that base.

The advisor owns updates to the campaign base. Use the repository-local `sync-organizer-frontier` skill to review organizer changes and import the exact promoted editable snapshot. Research agents should branch from the recorded base and must not merge upstream independently into their experiments.

Never run `yukon sync`, `yukon sync --harness-only`, or `yukon reset` in this maintained checkout. Those commands are designed for ordinary solver checkouts and may reset campaign history or repoint `origin`.

Before timing, record `BASE_SHA`, `UPSTREAM_SHA`, the host, the toolchain, the memory profile, and the exact proposal head. Work based on an older frontier can remain useful evidence, but it must be replayed on the current base and measured again before promotion. A branch name or `Validate submission` commit does not prove which submission Yukon promoted.

Do not commit generated weights, downloaded models, caches, local scores, hidden material, credentials, or large measurement artifacts. Preserve unrelated user changes and never use destructive Git commands to force a clean tree.

## What The Candidate May Change

The candidate may propose drafts and run the editable target path, but it may not change the target checkpoint, target weights, or transform contract, and it may not decide correctness for itself. The trusted parent measures the run, verifies the output, checks the row accounting, and computes the score.

`benchmark.json` is the only authority for submitted paths. The practical editable groups are:

- `mtp-head.manifest.json` and `mtp-head/`, which may declare a proposal head for the candidate leg only. Without a declaration, the candidate uses the organizer-pinned head. An invalid declaration fails instead of falling back, and the pinned serial leg never uses the candidate head.
- `Sources/MLXFastModel/Qwen36MTP*.swift`, which contains drafting, code that builds the target-model checks, acceptance, cache snapshots, rollback, repair, and scheduling.
- The other Qwen runtime and loading code under `Sources/MLXFastModel/`, the offline transform under `Sources/MLXFastTransform/`, and the exact Qwen, MLXLMCommon, MLX, and Metal paths listed in the manifest.

Do not change target weights, the tokenizer, goldens, the trusted driver, timing, telemetry, workflow, fixtures, package graph, or submission manifest. Tests, research notes, and campaign tools may live in this repository, but Yukon does not submit them.

Before implementation, list every file the experiment will change and run [`validate-assignment-scope.sh`](validate-assignment-scope.sh) and [`check-editable-budget.sh`](check-editable-budget.sh) against the experiment base. The current source limits are 3,000,000 bytes total, 524,288 bytes per file, and 262,144 bytes of candidate growth. `mtp-head/` is outside that source budget and instead has a declared byte count, digest, and 2 GiB cap. The live enforcing scripts remain authoritative if these values change.

Submission archives replace every required path in `editablePaths`. Only `mtp-head.manifest.json` and `mtp-head/` are optional. Inspect the packaged candidate: a local improvement that depends on an unsubmitted file is not a valid candidate.

## Scored Path

The worker loads the transformed checkpoint through `LLMModelFactory`, attaches the selected proposal head, warms legal shapes before timing, and serves rounds through `Qwen36MTPBlockSession`.

Each round does this:

1. Commit the pending primary token.
2. Choose zero through eight draft tokens within the limit offered by the parent.
3. Use the MTP head to propose those drafts.
4. Check the primary token and drafts with the fixed target.
5. Accept the longest correct prefix.
6. Keep or repair target and head state by rollback or replay.
7. Report the rows actually evaluated and the exact top-two token evidence to the parent.

The timed target is the vendored `Qwen35TextModel`/`Qwen35Model` together with the editable MTP session. The `Qwen35*` names describe the architecture family used by the Qwen 3.8 checkpoint; they do not make this a Qwen 3.5 challenge. A file being editable does not mean the scored worker executes it, so prove the live call path before optimizing it.

The target has 64 layers: 48 Gated DeltaNet recurrent layers and 16 full-attention layers. It has hidden size 5,120, vocabulary size 248,320, 24 query heads, 4 KV heads, head dimension 256, and affine 4-bit group-64 backbone weights. It is a dense hybrid model, not Laguna's NVFP4 mixture-of-experts model.

Acceptance rate alone is not the goal. Deeper drafting helps only when its extra accepted tokens outweigh proposal-head work, target checking, rejected work, and state management across varied prompts.

### Metal Source Forms

- A kernel family with an `mlx-generated/*.cpp` twin is JIT-compiled from the source string inside that C++ file. Update the readable `.metal` or `.h` source and its runtime-effective generated twin together, then run `python3 research/twin_audit.py`.
- A family without a generated twin is loaded from `mlx.metallib`. Rebuild it with `tools/build-mlx-metallib.sh` after relevant edits.
- `_nax` variants run on the ranked M5 and are first-class targets.

Before changing a kernel, prove the scored shape, dispatch family, source form, and M5 variant. Changes to precision, reduction order, packing, width-dependent dispatch, cache layout, or recurrent state need focused numerical checks. Matching one local argmax is not enough when the trusted parent also checks exact row evidence.

## Local Measurement

Use the Qwen setup and benchmark commands:

```bash
./setup.sh && ./setup-qwen-mtp.sh
./benchmark-qwen-mtp.sh --local-iterate
./benchmark-qwen-mtp.sh --local-submit
```

The official runner is the M5 host labeled `m5-qwen38-27b-mtp`. Other Apple GPUs can provide directional evidence only after confirming that they execute the same kernel family and layout.

Run one model-holding process at a time. The wrapper locks the run, checks for orphaned workers, and cools the GPU before each resident measurement. A wait at the 40C gate is normal. Do not bypass the lock or cooling gate, and inspect reported PIDs before killing a process. Compare base and candidate with the same host, power state, temperature, toolchain, memory profile, token window, and proposal head.

The local modes use one public fixture and default to 64 decode tokens for `--local-iterate` and 128 for `--local-submit`. They generate their own reference rows from the candidate, so matching those rows locally does not prove a match against the organizer's hidden reference and cannot reproduce the hidden eight-prompt result. Both local legs also use the same candidate build. Schedule and head changes therefore show up directly in the local serial-to-MTP ratio, while a general target or kernel improvement may speed both legs and cancel in that ratio. Always compare absolute candidate seconds per token with a fresh, unchanged `BASE_SHA` run as well as comparing the ratio.

When they explain a result, also inspect draft depth, accepted and rejected tokens, round count, rollback behavior, and block latency. Do not compare with an old result from another commit, machine, temperature, token window, memory profile, or head.

Use the frozen dependency graph for direct Swift commands:

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
swift build -c release --force-resolved-versions
tools/build-mlx-metallib.sh
```

Do not run `swift package resolve` or `swift package update`. A plain `swift build` may not rebuild `.build-worker/release/mlxfast-runtime-worker`, so use the Qwen wrapper for end-to-end tests.

## How Research Is Run

The advisor keeps four short records:

1. The current `UPSTREAM_SHA` and promoted `BASE_SHA`.
2. Fresh baseline measurements for each host and memory profile in use.
3. Official submissions, scores, candidate commits, and notes.
4. An experiment history that records the mechanism, scored path, evidence, result, and what new evidence would justify trying it again.

Before assigning work, read the experiment history and inspect the current source. Run different mechanisms in parallel. Duplicate an idea only to answer a named uncertainty. If an experiment is blocked by hardware or external state, preserve its commit and evidence and give the agent another runnable question.

Each research agent owns one clear question at a time. Record positive, negative, invalid, and inconclusive results so another agent does not repeat the same work blindly.

### Experiment Workflow

1. Record the base, upstream commit, host, toolchain, memory profile, and exact head.
2. State one question, the measured cost or source evidence behind it, the result you expect, and a stop rule.
3. Prove that the scored worker reaches the code and list candidate files separately from research-only files.
4. Check submission scope and byte budget before expensive work.
5. Measure a fresh, unchanged base under the local mode you will use for the candidate.
6. Implement one mechanism.
7. Run the cheapest compile, test, or numerical check that protects the boundary you changed.
8. Run one matched `--local-iterate` measurement and compare absolute time, the local ratio, and relevant counters with the fresh base.
9. Stop, revise, or advance using the written stop rule. Repeat only when noise or one specific uncertainty could change the decision.
10. For a credible winner:
    - Run the full Swift tests once and add opt-in runtime tests when the changed boundary needs them.
    - Run `--local-submit`.
    - Inspect the exact submitted diff and any generated Metal twins.
    - Recheck submission scope and byte budget.
    - Report evidence that another agent can reproduce.

Do not run every expensive check in every edit loop. Reach an end-to-end signal early, stop weak experiments quickly, and compose only winners that were measured independently.

Use this short report shape:

```text
Question:
  What does this experiment test?

Evidence:
  What measured cost or source observation makes it worth testing?

Expected result:
  What result would matter relative to noise and hardware-transfer risk?

Smallest decisive test:
  What is the shortest valid path to a decision?

Stop or promotion rule:
  What evidence ends, revises, or advances the experiment?
```

The runbook and templates contain the exact commands and reporting fields. Keep large local artifacts outside Git and label them with the host and commit.

## Official Submission And Promotion

Immediately before an official submission, query Yukon and compare the highest-scoring live `promoted` row with [`frontier-state.json`](frontier-state.json). If the submission ID, source reference, or score differs, sync the organizer and promoted frontier, replay the candidate on the new base, and measure it again.

Submit only a clean, committed candidate that passed `--local-submit` and whose exact submitted snapshot was inspected:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
yukon submissions --all
senpai/submit-official.sh "$BASE_SHA" \
  --model "<exact model name>" \
  --note-file submission-note.md
yukon submissions
```

The guard refreshes the campaign and organizer refs, checks the recorded base and trusted surface, and rejects dirty submitted paths or hidden Git index state such as `skip-worktree` and `assume-unchanged` before it calls Yukon. Yukon does not run the local pre-submit command for you.

Do not send duplicate official submissions. If a response is unclear, inspect Yukon before retrying so you do not create the same run twice. Never expose credentials in logs, notes, commits, or agent messages.

Promotion means that the official M5 run passed every gate and improved the campaign score. After a promotion, update the campaign state and replay later experiments only where the new base affects them.

## Correctness And Work Honesty

Correctness is a hard gate, not a speed tradeoff. The candidate must match the hidden serial token stream, produce a valid record for every round, account for every target row, and pass every gate in the current workflow and fixture.

The candidate may run target-model checks, decide which draft tokens were accepted, roll back after rejected drafts, and report what happened. The parent runner independently checks those claims. For every emitted token, run a real target-model evaluation and report its exact top-two scores. Never skip target work, invent results, use an undeclared model or head, change behavior based on the benchmark phase, or leave rejected cache state accessible.

Although `Qwen36MTPReferenceSession.swift` is editable, use it only to generate or replay reference rows outside timed candidate work. Never let reference results or known correct outputs influence timed generation or draft selection.

Only native MTP speculation inside the declared editable surface is allowed:

- Do not add prompt lookup, n-gram or suffix drafting, token-history shortcuts, caches that only help repeated benchmark requests, hidden-prompt specialization, or future state shared across requests.
- Do not detect the reference run, prompt pool, baseline, or benchmark phase to change the work performed.
- Do not use timing shortcuts, inject data into the worker protocol, access the network, or read hidden data from the filesystem.

Input-independent kernel, weight, and shape tables and normal cache reuse within one request are allowed when they preserve the contract. The target checkpoint is fixed even though the proposal head is editable: a new head may change proposal quality and cost, but it may not redefine the target answer.

If a public fixture generated on M5 shows a near-tie mismatch on another Apple GPU, compare the unchanged base at the same token position before blaming the candidate. That diagnostic never relaxes official exactness.

## Research Areas

Choose work from measured costs and current evidence; this list is not a backlog:

- adaptive draft depth, decisions to skip drafting, and schedules that work across prompts;
- proposal-head speed and quality, including compact or quantized representations;
- proposal-head history, cache policy, and redundant work;
- target verification batching and exact behavior at wider row counts;
- Gated DeltaNet recurrence, snapshots, replay, and rollback;
- full-attention and GQA cache updates and width-specific SDPA paths;
- affine 4-bit group-64 projections, normalization, reductions, and vocabulary readout;
- MLX scheduling, fusion, evaluation boundaries, command buffers, memory layout, and warmup misses;
- weight loading, transformed layout, and reusable metadata;
- interactions between runtime changes and speculative changes that have each won on their own.

Check [`campaign-ledger.md`](campaign-ledger.md) and [`laguna-to-qwen-speedup-map.md`](laguna-to-qwen-speedup-map.md) before starting.

Reject an experiment when the path is not used, the largest plausible gain is below measurement noise, it works only on one public prompt, it changes trusted code, or it combines several unmeasured ideas. Try it again only when the recorded reason for reopening has changed.

## Result Labels

- **Invalid:** It fails token matching, accounting, build, memory, integrity, or submission-surface checks. Repair it only if a compliant version of the same idea remains.
- **Not useful:** The target cost is absent or too small, or the valid implementation has no meaningful end-to-end gain.
- **Unclear:** Noise, prompt sensitivity, local-to-M5 differences, or cancellation in the local ratio could change the decision. Run the smallest check that settles it.
- **Local winner:** It shows a clear same-host improvement, correct behavior and counters, no observed fidelity problem, and a self-contained submitted snapshot.
- **Promoted:** An official ranked run passes every gate and improves the campaign frontier.
