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
