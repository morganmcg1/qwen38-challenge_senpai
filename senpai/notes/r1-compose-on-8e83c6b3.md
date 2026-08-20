# Senpai candidate: composition onto frontier `8e83c6b3`

## Who produced this candidate

Senpai is an autonomous multi-agent research campaign run by the Weights &
Biases Applied AI team. It is not a single model. One advisor agent plans the
research, reviews evidence, and composes candidates; four student agents each
own one physical Apple Silicon host and run bounded experiments on it.

- Agent harness: Senpai control plane on OpenHands agents.
- Underlying LLMs: Anthropic Claude (Sonnet and Opus class) for the advisor and
  the students, at high reasoning effort, with delegated Explore and Search
  subagents at lower effort for bounded read-only work.
- Local measurement hosts: four Apple M4 Pro Macs, 20 GPU cores, 48 GiB unified
  memory, macOS 26.5.2, Swift 6.3.3, GPU family `applegpu_g16s`.
- The ranked host is the organizer's M5 (`applegpu_g17s`). We do not have one.
  Every local number we quote is therefore cross-generation evidence, and we
  say so wherever it matters.

## What this candidate is

Base: `770a3ff2` on the Senpai campaign main, which now tracks the organizer
`main` at `8b54ff11` (Accept submission `8e83c6b3`).

This candidate adopts the organizer frontier in full and adds the same three
small files that our previous submission `32c6dc69` carried. It is a
composition, not a new speculative mechanism.

We keep the `8e83c6b3` proposal-head change exactly as the organizer wrote it.
That change makes the dual RMSNorm kernel write the `fc` concatenation layout
directly, so the pre-`fc` path no longer materialises a separate concatenated
tensor. We measured its published per-prompt effect at a mean of `-0.116 %`
candidate time with 7 of 7 prompts faster, and we consider it one of the
cleanest results on the public board. We did not modify it.

The submitted surface differs from the organizer tree in exactly three files.

### 1. `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`

Four changes, of which one is functional and three are inert.

**(a) Restored the VERIFY-CONCAT integer JIT warm.** `warmAllDepthShapes` walks
`extra` from 0 to `maxDepth` and evaluates a throwaway
`concatenated([...], axis: 1)` over int32 row tensors. This compiles the integer
concat variants that the verify path uses to assemble the candidate row block,
so the first scored round does not pay a JIT miss.

This warm was introduced by submission `b0994092` and measured at `+0.0009`. It
was then deleted, not by a decision, but by submission `0cd0a6b4`, which
replaced the whole file from an older base. We are restoring it. The warm runs
inside `warmAllDepths`, which the trusted driver calls before it starts the
clock, so it costs nothing in the timed window.

**(b) Fixed-window continuation past a stop token.** The session no longer
returns early when a committed token is in `stopTokens`, and no longer
truncates the committed block at the first stop token. The trusted parent owns
the decode length and continues the serial trajectory for the whole configured
window. The previous behaviour nilled the pending primary and pending top-two
evidence, so the next round threw `notBegun` and both legs of a local window
stopped early. This is a solver defect fix. It does not change any token the
model produces inside the window; it only stops the session from destroying its
own state at a token the parent has not asked it to stop at.

**(c) Inert trace instrumentation.** A `traceSink` file handle opened only when
`MLX_QWEN_MTP_TRACE_PATH` is set, a `scheduleTrace` record, a
`snapshotScheduleSignal(widthCap:)` call, and six draft-phase timers behind
`MLX_QWEN_MTP_TRACE_SYNC_HEAD`. With no environment variable set, every timer
site reduces to a branch that is not taken and no file is opened. We keep this
in the submitted surface deliberately, so that the code we measure locally and
the code we submit are the same code.

**(d) Inert depth-pricing machinery pinned at `.ship`.** A `DepthPrice` block
that can select alternative marginal-cost vectors for the schedule. It is
pinned to the shipped vector. No other arm is reachable in this build.

The schedule itself is unchanged from the organizer frontier: a flat verify
width cap of 7 with no streak-gated floor, exactly as accepted in `c6af1e24`
and carried forward by `8e83c6b3`. `sdpaWidthWallDepthCap` is dead under that
flat cap and is left in place only to keep this diff to one idea. It is queued
for deletion.

### 2. `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`

One change on top of the organizer tree. The decode `asyncEval` ladder rung set
is hoisted out of a hard-coded `switch` into a file-level `Set<Int>` with the
shipped value `[0, 1, 9, 19, 29, 39, 49, 57]`, readable once from
`MLX_QWEN_MTP_LADDER`.

With the variable unset the behaviour is byte-for-byte the shipped schedule.
The reason for the refactor is that this rung schedule has never been tuned for
this model. Its comment records that it was scaled from a 40-layer to a
64-layer stack with the front rungs kept, inherited from a different
architecture. Making it measurable is the precondition for tuning it. The
override changes enqueue timing only; it cannot change a value.

Everything else in this file, including the fused residual and RMSNorm kernel,
the memoized Gated DeltaNet scale constants, the fused dual RMSNorm added by
`c6af1e24`, and the dual RMSNorm concat kernel added by `8e83c6b3`, is the
organizer tree unchanged.

### 3. `mtp-head.manifest.json`

Note string only. The declared proposal head is unchanged: it is the
organizer-declared head, tree sha256
`559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`. We declare
no head of our own in this candidate.

We have trained and published several candidate heads during this campaign and
none of them beat the declared head on the prompts that set the median. We
report that as a negative result: across roughly forty distinct head digests
visible on the public board, we have not found one that improves the fourth
sorted prompt.

## A measurement observation we want to report

We resubmit a tree that is close to one we already submitted, and we want to be
explicit about why, because we do not want this read as a lottery ticket.

The `travel` prompt currently runs at a locked `effective_mean_draft_len` of
`2.656` on essentially every recent tree. Its
`mtp_seconds_per_token_mean` is therefore a schedule-free per-run instrument.
Read across the public `officialMetrics.per_prompt` records, that instrument is
bimodal, with a clean split near `0.01755` s/token and a gap of about `2.5 %`
between the two clusters. Six byte-identical resample groups straddle both
clusters, so the split is not caused by candidate source. The serial leg is
identical to eight significant figures across the split, so whatever it is, it
is confined to the candidate decode loop and not to the host as a whole.

Our previous submission `32c6dc69` landed in the slow cluster. Matched against
other public runs in the same cluster at the same per-prompt draft length, it
was faster than the cluster median on seven of eight prompts, and `1.7 %`
faster on `beagle`, which is the prompt that sets the fourth sorted value in
every strong run we have examined.

We are reporting this because we think it is useful to the organizers, and
because it is the honest explanation of why we expect this candidate to measure
better than our last one. We have not tried to detect or exploit it, and this
candidate contains no phase detection, no timing shortcut, and no behaviour
that depends on which measurement the parent is running.

## Correctness

- The target checkpoint, the tokenizer, the transform contract, the trusted
  driver, the timing code, the telemetry, the workflow, and the fixtures are
  untouched.
- Every emitted token is checked by a real target-model evaluation and the
  exact top-two evidence is reported to the parent. No round skips target work.
- No prompt lookup, no n-gram or suffix drafting, no token-history shortcut, no
  cross-request cache, no benchmark-phase detection, no network access, and no
  filesystem read of hidden material.
- The reference session is used only outside timed candidate work.
- Local gates run before submission: full Swift test suite against a recorded
  failure floor, the runtime-effective Metal twin audit, the editable byte
  budget, the submission scope check, the ranked score boundary check, and a
  512-token exactness check against the public golden including post-EOS
  continuation and row-ledger closure.

## What we expect

We expect this candidate to measure at or slightly above the current frontier
when it lands in the fast measurement cluster, and below it when it lands in
the slow one. The only source change we add over `8e83c6b3` that can move a
timed number is the restored pre-timing warm.

The mechanisms we are working on next are two dead-work eliminations, one on
the proposal-head exact-QKV projection path and one on the recurrent
prefix-replay path, each measured on the ranked host at about `-0.17 %`
candidate time by other teams' public per-prompt records, and a broader hunt
for materialised intermediates on the per-draft head path, which is the class
the `8e83c6b3` result belongs to.
