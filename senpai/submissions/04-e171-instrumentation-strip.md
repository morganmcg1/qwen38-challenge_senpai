# Remove research instrumentation from the scored surface, and keep the cross-round head-chain prefetch

This candidate makes two changes to the previous submission's submitted surface. One is a
pure deletion of research instrumentation. The other is a scheduling change inside the
speculative decode session. Nothing else moves.

```
Sources/MLXFastModel/Qwen36MTPBlockSession.swift
Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
```

`Qwen35.swift` is now byte-identical to organizer `main` at
`0863b06ac16e26e48fc06e97444095b00feb66d4`.

## Mechanism 1: delete the instrumentation, because the board says it costs about 1 %

Two of our own receipts form a matched pair whose submitted surfaces differ **only** by
research instrumentation. No mechanism separates them.

| tag | submission | commit | officialScore |
| --- | --- | --- | --- |
| A | `5a9f130a` | `8ba6e73837f525553cc99a5b71ec1a5e479ca7b0` | `3.70784519415395` |
| C | `fda590bb` | `080d4cd318c273c5dd34d8ad4429d86096e19258` | `3.667847308` |

A is organizer `main` byte for byte. C differs from A on the submitted surface by 108
inserted and 10 deleted lines, and every one of them is instrumentation: an
environment-gated pinned-draft-depth instrument that is read once in an initializer and
is `nil` on any ranked run, five extra fields on a research trace line, five counter
globals, two counter increments inside the custom quantized matrix-vector router, three
counter writes in a derived-index build that runs once during the untimed warm, an
environment override for the index leaf width that returns the compiled default when
unset, and changed default arguments on four verification and benchmark functions that no
scored path calls.

The derived-index geometry is `8` rows per leaf in both trees, and the accept ledger
agrees on all eight hidden prompts: `effective_mean_draft_len` and
`non_drafting_round_count` are equal prompt by prompt. The trees do the same speculative
work. They differ by counters.

### The common mode is measured, not assumed

Receipt C carries two probes of host speed that perform identical work in both trees. The
candidate build's own prefill runs byte-identical quantized kernels in A and C, and the
runner-owned serial leg runs the pinned baseline in both.

```
candidate prefill   -0.2315 %
runner serial       -0.2513 %
pooled              -0.2414 %      the host was slightly FASTER on receipt C
```

Two unrelated instruments agree to 0.02 percentage points. With that common mode removed,
the candidate decode leg moves the other way:

```
prompt      edl     non-draft   decode C-A   corrected
plutarch    0.1557    92 %       -0.110 %     +0.131 %
drama       2.2976     0 %       +0.687 %     +0.928 %
travel      2.6479     0 %       +1.557 %     +1.798 %
beagle      4.3818     0 %       +1.138 %     +1.379 %
essays      5.0870     0 %       +0.418 %     +0.660 %
medicine    5.2556     0 %       +1.365 %     +1.606 %
republic    4.9892     0 %       +1.181 %     +1.422 %
botany      6.1481     0 %       +0.577 %     +0.818 %

mean +1.0929 %   sd 0.5564   se 0.197   effect / se 5.5
```

### One source guard predicts the shape of that table

`Qwen35CustomQMV.widths` is `2 ... 9`, and the router's `routable` check rejects any other
width. The two counter increments sit behind that check. The instrumented path is
therefore entered if and only if the row count is between 2 and 9, which partitions a
ranked leg into three regimes that the receipt measures separately:

| regime | rows | routed | measured |
| --- | --- | --- | --- |
| prefill | 512 | no | -0.23 %, equal to the drift estimate |
| non-drafting decode round | 1 | no | +0.131 % on the 92 %-non-drafting prompt |
| drafting decode round | 2 to 9 | yes | +1.230 % over the seven drafting prompts |

Three levels, one compile-time range check, and the measurement agrees with all three.
Host drift cannot select for drafting, so the internal control is what carries this
result. We do not yet have a microarchitectural account of why two integer increments
cost about 1 % of a decode leg; a lost inlining or specialisation decision in the router
once it acquires a global side effect is the leading candidate. We are reporting the
measurement and its controls, not a mechanism story we cannot support.

The response is a deletion. `Qwen35.swift` returns to organizer bytes in full rather than
losing only the two increments, because full parity makes the prediction exact: this
candidate reproduces the A-against-C contrast with the sign reversed.

## Mechanism 2: cross-round head-chain prefetch

The proposal head's flush of committed rows and its first draft step used to be built and
submitted at the top of the round that consumes them. The GPU is then idle from the
previous round's blocking evaluation until that submission arrives. This change issues the
same work at the earliest point in the previous round at which the committed row is known,
so the device runs it during the host tail and the protocol turnaround.

It is a scheduling change. It computes the same head forward on the same inputs, proposes
the same drafts, and takes the same schedule.

Local evidence, on an M4 Pro, sixteen thermally gated 512-token legs in two
counterbalanced replicates:

```
effect        -0.2872 % on the candidate decode leg, 4.80 sigma
serial-leg negative control   -0.0209 %, 2 sigma 0.1095 %, contains zero
accept ledger identical between arms, digit for digit
513/513 hexfloat reference rows exact in both replicates
```

The mechanism was also re-measured on the tree submitted here, and it remains
ledger-neutral: `effective_mean_draft_len` and `accepted_draft_rate` are identical to the
last digit with the prefetch on and off.

Its default is now opt-out. While the mechanism was an unmerged research arm it was
opt-in, so that an unset environment reproduced the older order for earlier legs. A ranked
runner sets no such variable, so an opt-in default would have shipped the old path.

## What we measured locally on this exact tree

The local harness runs both of its legs from the candidate build and generates its own
reference rows from the candidate, so it cannot prove a match against the organizer's
hidden reference and cannot reproduce the hidden eight-prompt result. We use it as a
correctness and packaging gate and rely on the official runner for the score.

LOCAL_EVIDENCE_PLACEHOLDER

## Honest limitations

- The 1 % instrumentation tax is measured on the official runner, but its mechanism is not
  identified. If it turns out to be an artefact of one specific inlining decision, the
  deletion is still correct, but the size of the recovery may differ.
- The prefetch gain is measured on an M4 Pro. The transfer of a host-side scheduling gain
  to the M5 runner is not something we can measure from here, and we do not claim to know
  it.
- `swift test` is **not** green on this tree, and it is not green on the campaign base
  either: 41 tests fail on the base before any change in this candidate. We ran the
  correctness gates that bind this change instead, and we report exactly what they showed.
  We are not claiming a green test suite.

## Attribution

`Model: senpai` on the first line above is a campaign label, not a catalogued model. It
denotes a multi-agent research program: one advisor agent supervising student agents that
run experiments on separate machines and report results through pull requests. The exact
underlying models, effort levels and harness are:

| Component | Value |
| --- | --- |
| Frontier model | `anthropic/claude-fable-5` |
| Smart model (advisor and students) | `anthropic/claude-opus-5`, reasoning effort `max` |
| Fast model (mechanical subtasks) | `anthropic/claude-sonnet-5` |
| Default reasoning effort | `max` |
| Agent harness | Senpai (`github.com/wandb/senpai`), native backend, OpenHands-derived agent SDK |
| Roles in this launch | 1 advisor + 4 students, 1 GPU each |
| Compute backend | `aws-mac` |
| Experiment tracking | Weights & Biases |

No human wrote any of the code or analysis in this submission. Humans set the campaign up
and answered clarifying questions on GitHub issues.
