# Qwen 3.8 27B Native-MTP MLX Inference Autoresearch

This document defines how advisor and student agents run competition research
in this repository. The campaign goal is to make Qwen 3.8 27B native-MTP
decode materially faster on Apple Silicon and win the `mlx.fast` competition
without changing the serial token stream or violating the trusted benchmark
contract.

The only ranked track in this repository is `qwen3.8-27b-mtp-v1`. The retained
Poolside Laguna and DFlash sources, scripts, fixtures, and prose belong to a
different repository and track. They are useful historical references, not
instructions for this campaign.

## Mission

Maximize the official decode-only score:

```text
for each hidden prompt p:
  raw_p = mean(pinned serial depth-0 seconds/token)
          / mean(candidate seconds/token)

published score = median(raw_1 ... raw_8)
```

The ranked run measures all eight hidden prompts. Because eight is even, the
median is the mean of the two central order statistics. There is no no-op
normalization and no separately scored prefill component. The 512-token seed
prefill is charged inside each leg's 512-token decode window.

Serial decode is therefore exactly `1.0`. The published median must be at least
`0.90` and no greater than the `3.0` plausibility ceiling. The organizer's
calibrated stock depth-2 tree measured about `0.994`; that number describes the
historical starting line, not necessarily the current promoted `main`.

Drafting nothing is legal and degenerates to serial decode. It is a useful
control, but the objective is a real score above `1.0`: speculation and runtime
work must together repay their own cost.

Research continues while the campaign is active. A numbered round, an
experiment batch, or one promoted result is a checkpoint rather than a stopping
condition. Stop only on explicit operator instruction or when no safe,
runnable, non-duplicative arm remains.

## Contract First

This repository was split from a Laguna/DFlash repository and still contains a
large amount of inherited prose. Some comments also preserve superseded Qwen
3.6 bring-up or pre-2026-08-14 scoring language. Do not resolve ambiguity by
majority vote among documents. Use this authority order:

1. `benchmark.json` owns the track identity, commands, editable paths,
   exempt-path budget policy, and current scoring fields.
   `.github/scripts/run-submission-static-review.sh` owns the total, per-file,
   and growth caps, with the trusted Swift worker re-enforcing its applicable
   limits. `benchmark.qwen-mtp.json` must remain byte-identical to the manifest,
   but neither file is participant-editable.
2. `fixtures/qwen3_8_27b_mtp_track.json` owns artifact pins, timed-pool
   membership, fidelity semantics, and calibration evidence.
3. `.github/workflows/qwen-mtp-ranked-benchmark.yml` owns what the ranked M5
   job actually enforces, including runner label and environment values.
4. `docs/qwen-mtp-editable-surface.md` is the definitive prose explanation of
   the editable/trusted boundary.
5. `TASK.md` is the concise participant-facing contract summary.

When those authorities disagree, stop and identify the exact live enforcing
site before changing code or assigning an experiment. Never import a Laguna
rule merely because it appears in a later or longer section of `AGENTS.md` or
`README.md`.

### Contract map

| Topic | Authoritative detail |
| --- | --- |
| Track, commands, surface, score | [`benchmark.json`](../benchmark.json) |
| Pins, pool, calibration, fidelity | [`fixtures/qwen3_8_27b_mtp_track.json`](../fixtures/qwen3_8_27b_mtp_track.json) |
| Ranked execution | [`.github/workflows/qwen-mtp-ranked-benchmark.yml`](../.github/workflows/qwen-mtp-ranked-benchmark.yml) |
| Editable versus trusted | [`docs/qwen-mtp-editable-surface.md`](../docs/qwen-mtp-editable-surface.md) |
| Participant task and commands | [`TASK.md`](../TASK.md) |
| Agent and machine operating rules | [`AGENTS.md`](../AGENTS.md) |
| Local Qwen runner | [`benchmark-qwen-mtp.sh`](../benchmark-qwen-mtp.sh) |
| MTP round hot path | [`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`](../Sources/MLXFastModel/Qwen36MTPBlockSession.swift) |
| Baseline, candidate, and promotion commands | [`experiment-runbook.md`](experiment-runbook.md) |
| Assignment and terminal result formats | [`assignment-template.md`](assignment-template.md), [`result-template.md`](result-template.md) |
| Scope and byte-budget preflights | [`validate-assignment-scope.sh`](validate-assignment-scope.sh), [`check-editable-budget.sh`](check-editable-budget.sh) |
| Generated Metal twin consistency | [`research/twin_audit.py`](../research/twin_audit.py) |
| Frontier pins, baselines, receipts, novelty | [`frontier-state.json`](frontier-state.json), [`campaign-ledger.md`](campaign-ledger.md) |

## Repository And Frontier Discipline

This campaign repository is intended to live at
`morganmcg1/qwen38-challenge_senpai`. Its canonical challenge source is
`Layr-Labs/qwen-3.8-mtp-challenge`. The campaign repository is not a GitHub
fork, so remote tracking and upstream refresh are explicit responsibilities.

Use conventional remote names when configuring a worktree:

```text
origin    https://github.com/morganmcg1/qwen38-challenge_senpai
upstream  https://github.com/Layr-Labs/qwen-3.8-mtp-challenge
```

Configure `upstream` with push URL `DISABLED`; the repository-local sync skill
and submission guard verify both fetch and push directions. A fresh clone can
configure the remotes and pinned Yukon link with:

```bash
senpai/bootstrap-checkout.sh
```

Always verify remote URLs rather than assuming those names were configured
correctly. Upstream refresh and research-frontier promotion are different
operations:

- `upstream/main` is the latest organizer source and contract state.
- `BASE_SHA` is the exact campaign frontier commit an experiment must beat.
- `UPSTREAM_SHA` is the organizer commit whose trusted surface and submission
  contract that frontier is based on.

The advisor owns upstream integration. Before opening a new research batch,
fetch the canonical upstream, inspect its commits and full diff, and determine
whether it changes trusted contracts, the editable surface, accepted code, or
only validation history. Integrate into a clean campaign branch, run cheap
contract/build checks, and record a fresh same-host baseline before declaring a
new frontier. Do not merge upstream into every in-flight student branch.

Never run `yukon sync`, `yukon sync --harness-only`, or `yukon reset` in the
maintained Senpai checkout. Yukon implements those operations for a plain solver
checkout and may reset history and repoint `origin` to the organizer. Use the
repository-local `sync-organizer-frontier` skill, which imports reviewed policy
and one exact promoted editable snapshot while preserving campaign files.

Students must record `BASE_SHA`, `UPSTREAM_SHA`, host identity, and relevant
toolchain version before timing. An experiment based on an older frontier can
remain useful evidence, but it must be rebased or replayed and remeasured before
promotion. Never infer the best frontier from a branch name, a `Validate
submission` commit message, or an old calibration number.

Generated weights, caches, local scores, hidden material, and downloaded model
artifacts never belong in the campaign history. Preserve unrelated user work in
dirty worktrees, and do not use destructive Git commands to force a refresh.

## Editable And Trusted Boundaries

The central safety rule is:

> Anything that only proposes tokens is editable. Anything that verifies,
> measures, or ledgers is trusted.

The competitive surface is the union of:

1. The MLX runner and kernel path: Qwen model execution, offline transform,
   vendored Qwen/MLXLMCommon plumbing, and the listed MLX Metal kernels.
2. The whole speculative apparatus: drafting code, per-round schedule, and MTP
   head weights.

The practical groups are:

- `mtp-head.manifest.json` and `mtp-head/`: an optional candidate head. An
  absent declaration selects the organizer-pinned head; a present declaration
  must resolve and digest-verify. The head applies to the candidate leg only.
- `Sources/MLXFastModel/Qwen36MTP*.swift`: drafting, verify-block assembly,
  accept walk, cache snapshot/rollback/repair, and the schedule.
- `Sources/MLXFastModel/`: other editable Qwen runtime and loading code.
- `Sources/MLXFastTransform/`: offline transform.
- The exact Qwen model, MLXLMCommon, and MLX/Metal paths listed in
  `benchmark.json`.

The trusted parent offers a per-round maximum. Candidate policy may propose
from 0 through 8 drafts, adaptively on every round. The candidate declares no
fixed depth to the scorer; the trusted journal derives effective draft lengths
and closes the ledger over what actually happened.

Do not change target weights, tokenizer, goldens, trusted driver, row
accounting, timing, telemetry, workflow, contract fixtures, package graph, or
submission manifest. Support tests, research notes, and campaign tooling may be
committed to this repository, but they are not part of a Yukon submission.

Before implementation, enumerate the exact candidate files and check them
against `benchmark.json` at `BASE_SHA`. The editable source budget is 3,000,000
bytes total, 524,288 bytes per file, and 262,144 bytes of growth. `mtp-head/`
is excluded from that source budget and instead has its own declared digest,
byte count, and 2 GiB cap.

Submission archives use replace semantics over `editablePaths`. The two MTP
head paths are explicitly optional; other missing source paths are not. Inspect
the packaged candidate rather than assuming a locally working support-file
change will upload.

## Scored Path

The native-MTP worker loads the transformed Qwen checkpoint through
`LLMModelFactory`, attaches the selected head, warms legal round shapes outside
the timed protocol, and serves rounds through `Qwen36MTPBlockSession`.

Each speculative round conceptually does this:

1. Commit the pending primary token.
2. Let the candidate policy choose 0...8 drafts within the offered width.
3. Run the MTP head chain to propose those drafts.
4. Verify `[primary] + drafts` with the fixed target.
5. Accept the longest correct prefix.
6. Keep, roll back, replay, or repair target and head state as required.
7. Declare the actual rows and top-2 evidence to the trusted parent.

The timed target is the vendored `Qwen35TextModel`/`Qwen35Model` reached through
that factory, together with the editable Qwen MTP session. Do not assume every
file under the directory-wide `Sources/MLXFastModel` allowance is hot. In
particular, the separate `Qwen35FastEngine` path is compile-checked and its
production selector is currently the library oracle; the MTP worker does not
construct `Qwen35RuntimeWeightCache`. Prove a call path into the MTP worker
before timing a change.

The Qwen 3.8 tower is not Laguna and is not an NVFP4 MoE. Its verified geometry
is 64 layers on a four-layer hybrid repeat: three gated-delta linear-attention
layers followed by one full-attention layer, with hidden size 5120, 24 query
heads, 4 KV heads, head dimension 256, vocabulary 248320, and affine 4-bit
group-64 backbone weights. The pinned MTP head is a separate bf16 artifact;
the current campaign frontier may declare another legal candidate head.

Promising cost centers include:

- draft policy, accepted tokens per round, and non-drafting decisions;
- MTP head history priming, cache reuse, per-draft launches, projection, and
  on-device token selection;
- target verify width, Gated DeltaNet recurrent state, full-attention cache
  work, rollback/replay, and rejected work;
- quantized projections, vocabulary projection, normalization, reductions,
  and other kernels actually reached by MTP decode;
- MLX graph construction, materialization, synchronization, command-buffer
  boundaries, warmup, and memory layout;
- transformed weight metadata and loading that reduce candidate hot-path work.

Acceptance rate alone is not the objective. A deeper round is useful only when
its extra committed tokens outweigh head, verify, rejected-tail, and state
management cost over varied prompts. The official median is deliberately
multi-prompt, so a policy tuned to the public long-copy fixture or an easy
self-continuation is weak evidence.

### Kernel source forms

The vendored MLX package uses two source forms:

- Families with an `mlx-generated/*.cpp` twin are JIT-compiled from the C++
  embedded source string. The generated twin is runtime-effective; keep the
  readable `.metal`/`.h` source in sync.
- Families without a generated twin are loaded from `mlx.metallib`, built from
  the vendored Metal sources by `tools/build-mlx-metallib.sh`.
- `_nax` variants are selected on the ranked M5 generation and must be treated
  as first-class, not as optional cleanup after tuning a plain variant.

Before a kernel arm, prove the scored shape, dispatch family, source form, and
M5 variant. Changes to reduction order, width-dependent dispatch, precision,
packing, or state layout require targeted numerical evidence. MTP correctness
includes exact emitted tokens and a closed reference-checked row ledger; a
locally stable argmax is not enough to declare a numerically different wide
path safe.

## Hardware And Local Measurement

The ranked workflow selects self-hosted Apple M5 hardware with runner label
`m5-qwen38-27b-mtp`. The ranked run is the only authority for score and hidden
fidelity. Results from M4 or another Apple GPU are directional after confirming
that the same kernel family and layout execute; `_nax` dispatch and geometry can
make a local win reverse on M5.

Run only one model-holding process at a time. The Qwen target plus head and
working state are fully resident. The local wrapper takes a run lock and scans
for orphaned workers; direct model commands do not. Inspect reported PIDs before
killing anything, and never disable the guard to work around legitimate
contention.

The local wrapper cools the GPU before each model-resident leg. A wait at the
40C gate is normal. Do not terminate it merely because cooling takes minutes,
and do not disable the gate for comparable performance measurements. Keep host,
power, thermal, memory-profile, and toolchain conditions matched between base
and candidate.

Machines below the full-profile memory threshold use a low-memory startup
profile. It changes allocator and command-buffer management rather than the
ranked code path, but performance measurements should still compare identical
profiles. An out-of-memory failure is a host limitation, not permission to skip
the ranked path.

### What the local score can and cannot show

`./benchmark-qwen-mtp.sh --local-iterate` defaults to 64 decode tokens;
`--local-submit` defaults to 128. Both use one public fixture, candidate-built
reference rows, and a local serial/MTP pair. They cannot reproduce the ranked
eight-prompt 512-token median or hidden reference.

The local serial leg and MTP leg use the same local build. Consequently:

- local serial-over-MTP ratio is useful for schedule, head, and speculative
  overhead changes;
- a general target-runtime or kernel speedup can improve both local legs and
  largely cancel in that ratio, even though it counts officially because the
  ranked serial denominator comes from the pinned baseline workspace;
- compare absolute candidate seconds/token against a fresh same-host
  `BASE_SHA` run as well as reading the local ratio;
- inspect effective draft lengths, acceptance/rejection, round count, rollback,
  first/max/p50 block latency, and parity when they explain the result.

Do not compare a candidate with an old score file from another frontier,
machine, temperature, token window, or head. Store baseline artifacts outside
the submitted surface and label them with SHA and host.

## Research Team

The advisor maintains four compact campaign records:

1. Current `UPSTREAM_SHA` and promoted `BASE_SHA`.
2. Same-host baseline measurements by hardware and memory profile.
3. Official submission receipts, scores, candidate SHAs, and notes.
4. A novelty index with one row per mechanism: scored path, hypothesis,
   evidence, disposition, and condition for reopening.

The versioned index and receipt tables are
[`campaign-ledger.md`](campaign-ledger.md); exact machine-readable organizer
and promoted pins are [`frontier-state.json`](frontier-state.json). Large local
artifacts remain outside Git and are referenced by labeled evidence location.

Before assigning work, consult the novelty index and inspect relevant current
source. Prefer parallel arms in distinct mechanisms or cost centers. Duplicate
nearby ideas only to resolve a named uncertainty. If one arm becomes blocked on
hardware or external state, preserve its exact branch and evidence and give the
student another runnable arm.

Students own one causal question at a time. They may refine the same mechanism
until it wins or is exhausted, but must not bundle unrelated ideas merely to
produce a larger diff. Every result, including a negative one, updates the
novelty index so later agents do not repeat it blindly.

## Student Workflow

For each experiment:

1. Record `BASE_SHA`, `UPSTREAM_SHA`, host, head provenance, toolchain, and
   memory profile.
2. Identify the scored call path and list candidate versus research-only files.
3. Check editable scope and byte budget before doing expensive work.
4. Measure the unchanged frontier once under the intended local mode and save
   its artifact outside the submission surface.
5. Implement one causal mechanism.
6. Run the cheapest targeted compile, test, or numerical check that guards the
   changed boundary.
7. Run the candidate under the same local mode and host conditions.
8. Compare absolute MTP time, local serial/MTP ratio, and mechanism telemetry
   against the fresh base.
9. Reject, revise, or advance the arm using its predeclared stop rule.
10. For a promotion candidate, run the stronger validation ladder, inspect the
    exact submitted diff, and report reproducible evidence to the advisor.

The concrete preflight, comparison, and reporting commands live in
[`experiment-runbook.md`](experiment-runbook.md),
[`assignment-template.md`](assignment-template.md), and
[`result-template.md`](result-template.md).

Use these direct Swift commands only with the frozen dependency flag:

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
swift build -c release --force-resolved-versions
tools/build-mlx-metallib.sh
```

A bare `swift build` does not necessarily rebuild the scored worker at
`.build-worker/release/mlxfast-runtime-worker`. Use the Qwen benchmark wrapper
for end-to-end work, and rebuild the metallib explicitly after relevant AOT
Metal edits.

## Experiment Ladder

The ladder is selective, not cumulative. Climb only when the next step can
change the decision.

### 1. Orient

- State the causal question and expected end-to-end signal.
- Prove reachability from the native-MTP worker.
- Identify the candidate surface, trusted boundary, kernel source form, and
  head provenance involved.
- Reject prompt-, token-, fixture-, timing-, or hidden-data-specific behavior.

Static inspection may close an idea without a build. A microbenchmark is useful
only when it bounds a named scored cost.

### 2. Fast screen

Run a matched `--local-iterate` candidate after one unchanged base. Add a
focused unit or runtime test only where the edit changes a contract boundary.

Do not put a full Swift suite, repeated smoke runs, every opt-in parity test,
and `--local-submit` into every edit loop. Reach an end-to-end signal early and
stop an unpromising arm.

### 3. Resolve uncertainty

Use the smallest discriminator:

- repeat once when a likely gain is near noise;
- trace one round phase or dispatch to locate a named cost, then remove or
  disable instrumentation before comparable timing;
- test multiple legal widths when a mechanism is width-sensitive;
- use real-checkpoint/runtime parity for representation, recurrence, cache,
  precision, reduction-order, or layout changes;
- compare stock/pinned and candidate head provenance when a head change is
  involved.

If no available check is likely to alter the decision, close the arm and record
the remaining uncertainty.

### 4. Confirm for promotion

For a stable winner:

- run the full Swift tests once;
- run opt-in MLX runtime tests when the changed boundary requires them;
- run `./benchmark-qwen-mtp.sh --local-submit`;
- recheck the diff and byte budget against `benchmark.json` at `BASE_SHA`;
- confirm no generated artifacts, reference weights, scores, or support-only
  dependencies are required by the submission;
- ensure a remote or in-branch head declaration is complete, immutable,
  digest-correct, and within its cap;
- write a concise submission note with mechanism, base SHA, candidate SHA,
  local evidence, and known transfer risks.

### 5. Official promotion

Only the ranked M5 result can promote a candidate. Account and submission work
uses the Yukon CLI, never an inherited `mlxfast` upload command:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
yukon submissions --all
senpai/submit-official.sh "$BASE_SHA" \
  --model "<exact model name>" \
  --note-file submission-note.md
yukon submissions
```

Immediately before submission, compare the highest-scoring `promoted` Yukon
row with `senpai/frontier-state.json`. If its submission ID, promoted source
ref, or score differs, stop and run the organizer-frontier sync workflow before
rebasing and remeasuring the candidate. Validation commits alone do not reveal
which candidate Yukon promoted, so the guarded wrapper proves campaign-base
and trusted-surface freshness but does not replace this live Yukon check.

The guard invokes `yukon submit` only after refreshing the campaign and
organizer remotes, checking that the recorded base's submitted snapshot is
current, and rejecting hidden or dirty submitted files. Yukon does not run the
local pre-submit command. Submit from a clean, committed candidate whose
editable snapshot was inspected. Do not expose API keys in logs, notes,
commits, or agent messages.

The ranked resource is serialized. Do not dispatch duplicate submissions to
try to outrun the queue. If a response is ambiguous, inspect submissions before
retrying so an already-created run is not duplicated. Waiting on one official
candidate must not stop safe work on independent follow-up arms; keep every
candidate identified by exact commit.

Promotion means the official candidate passed every gate and beat the current
campaign frontier under the published score. After promotion, update campaign
state and rebase or replay later work only where the new frontier affects it.

## Correctness And Validity

Correctness is a hard gate, never a speed/quality tradeoff. The ranked run
requires exact agreement with the hidden serial trajectory, a structurally
sound round journal, and a closed reference-checked row ledger. It also runs the
public drift tripwire, hidden teacher-forced, anchor, free-run, behavior, GPQA,
TTFT, and semantic gates. Any failure produces no useful ranked score.

The checked-in public fixtures were generated on M5. If a non-M5 host shows a
near-tie mismatch, compare unmodified `BASE_SHA` at the same position before
calling it a regression. This local diagnostic never relaxes the official M5
gate.

Allowed speculation is the native MTP mechanism inside the declared editable
surface. Do not add prompt lookup, n-gram or suffix drafting, token-history
shortcuts, request-keyed memoization, hidden-prompt specialization, cross-request
future state, timing shortcuts, protocol injection, network access, or
filesystem exfiltration. Input-independent kernel/weight/shape caches and
ordinary within-request state reuse remain legitimate when they preserve the
contract.

The target checkpoint is fixed even though the proposal head is editable. A
new head can change acceptance and cost; it cannot redefine the answer.
Representation, transform, or kernel changes must still satisfy target
identity, transform validation, output fidelity, and static review. Never apply
the inherited Laguna NVFP4 quantization envelope to Qwen by analogy.

A fallback that restores correctness but consumes the measured gain is not a
winner. A local pass on one easy prompt is not evidence of pool-wide schedule
robustness. A microbenchmark gain without an end-to-end gain is useful negative
or diagnostic evidence, not a promotable result.

## Research Method

Each arm starts with the weakest sufficient causal hypothesis: strong enough to
predict a decisive observation, but no broader than the evidence supports. As
a useful heuristic, see Bennett, [*The Optimal Choice of Hypothesis Is the
Weakest, Not the Shortest*](https://arxiv.org/pdf/2301.12987v4).

Every experiment report should contain:

```text
Causal question:
  What mechanism or uncertainty does this arm test?

Target evidence:
  Which measured cost, trace, receipt, or source observation motivates it?

Expected signal:
  What outcome would be meaningful relative to local noise and transfer risk?

Cheapest decisive test:
  What is the shortest valid path to a decision?

Stop rule:
  What evidence promotes, revises, or ends the arm?
```

The operating loop is:

1. Locate and, when useful, measure the target cost.
2. Bound the plausible end-to-end effect.
3. Test one causal mechanism.
4. Reject quickly on validity, reachability, feasibility, or speed.
5. Confirm only credible winners.
6. Compose only independently measured winners.
7. Preserve negative results and reopening conditions.

When ideas cluster around one family, use source review, current official
diffs, public submission evidence, relevant MLX/Qwen literature, and an
independent critical agent to generate alternatives. Treat every suggestion as
a hypothesis to verify on the live scored path.

## Research Map

Choose work from current profiles and evidence rather than treating this list
as a queue:

- adaptive draft depth, skip decisions, and prompt-robust scheduling;
- better proposal heads, quantized or compact head representations, and the
  acceptance-versus-cost frontier;
- head history/cache policy and removal of redundant head work;
- target verify batching and exact wide-shape behavior;
- Gated DeltaNet recurrence, snapshot, replay, and rollback;
- full-attention/GQA cache updates and width-specific SDPA dispatch;
- 4-bit group-64 projection and vocabulary-readout kernels;
- MLX scheduling, fusion, evaluation boundaries, and warmup misses;
- weight loading, transform layout, and reusable metadata;
- interactions among individually measured runtime and speculative wins.

Reject an arm when its path is dormant, its maximum plausible gain is below
noise, it relies on one public prompt, it changes a trusted surface, or it
bundles several unmeasured ideas. Reopen it only when the novelty index's named
condition changes.

## Decision Rules

- **Invalid:** correctness, fidelity, ledger, build, memory, integrity, or
  submission-surface failure. Repair only if a compliant mechanism remains.
- **Dead:** the cost is absent or too small, or a valid implementation has no
  meaningful end-to-end gain.
- **Ambiguous:** noise, local/ranked hardware transfer, prompt sensitivity, or
  cancellation in the local paired ratio could change the decision. Run the
  smallest check that resolves it.
- **Green locally:** a clear matched same-host improvement, correct telemetry,
  no observed fidelity regression, and a self-contained editable candidate.
- **Promoted:** an official ranked pass that improves the campaign frontier.

## Final Principle

Speedup is both the target and the operating principle. Explore boldly, test
decisively, keep the Qwen contract in view, and spend verification time where it
can change a research or promotion decision. We are here to secure the top of
the leaderboard with a real, reproducible M5 speedup.
