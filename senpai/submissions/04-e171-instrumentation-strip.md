# Organizer main, plus four campaign quantized kernels, plus a cross-round head-chain prefetch

This candidate is organizer `main` at `0863b06ac16e26e48fc06e97444095b00feb66d4` with two
mechanisms added and one accumulation of research instrumentation removed. We price the
whole thing at about **+1.11 % of candidate decode time over organizer main**.

## What is actually in the submitted archive

Against organizer `main`, the submitted surface is five files:

```
Sources/MLXFastModel/Qwen36MTPBlockSession.swift
Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h
```

`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` is **byte-identical to organizer
main**. A diff of that path against `0863b06a` is empty.

Three ingredients:

1. **Q** — the four `quantized*` files. Campaign kernel work on the affine 4-bit
   group-64 matrix-vector path, carried forward unchanged from our previous submission.
2. **E165** — a cross-round head-chain prefetch inside the speculative session, promoted
   from an opt-in research arm to the shipped default.
3. **A deletion** — research instrumentation removed from the scored surface. This is not
   a new mechanism. It returns the scored shape to the code that produced our best
   receipt.

## Ingredient 3 first, because it is the one that needs explaining

Two of our own receipts form a matched pair whose submitted surfaces differ **only** by
research instrumentation. No mechanism separates them.

| tag | submission | commit | officialScore |
| --- | --- | --- | --- |
| A | `5a9f130a` | `8ba6e73837f525553cc99a5b71ec1a5e479ca7b0` | `3.70784519415395` |
| C | `fda590bb` | `080d4cd318c273c5dd34d8ad4429d86096e19258` | `3.667847308` |

A is organizer main byte for byte on the scored surface. C is A plus 108 inserted and 10
deleted lines, and every one of them is instrumentation: an environment-gated
pinned-draft-depth instrument that is read once in an initializer and is `nil` on any
ranked run, five extra fields on a research trace line, five counter globals, two counter
increments inside the custom quantized matrix-vector router, three counter writes in a
derived-index build that runs once during the untimed warm, an environment override for
the index leaf width that returns the compiled default when unset, and changed default
arguments on six verification and benchmark functions that no scored path calls.

The derived-index geometry is `8` rows per leaf in both trees, and the accept ledger
agrees on all eight hidden prompts: `effective_mean_draft_len` and
`non_drafting_round_count` are equal prompt by prompt. The two trees do the same
speculative work. They differ by counters.

### The common mode is measured, not assumed

Receipt C carries probes of host speed that perform identical work in both trees. The
candidate build's own prefill runs byte-identical quantized kernels in A and C, and the
runner-owned serial leg runs the pinned baseline in both.

```
candidate prefill   -0.2315 %
runner serial       -0.2513 %
pooled              -0.2414 %      the host was slightly FASTER on receipt C
```

Two unrelated instruments agree to 0.02 percentage points. With that common mode removed
using candidate prefill as the drift control, the candidate decode leg moves the other
way: **+0.985 %, standard error 0.190, t = 5.2**. The per-prompt shape, with the serial
leg as the drift control instead, is:

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
Host drift cannot select for drafting, so the internal control is what carries the result.
We do not yet have a microarchitectural account of why two integer increments cost about
1 % of a decode leg; a lost inlining or specialisation decision in the router once it
acquires a global side effect is the leading candidate. We report the measurement and its
controls, not a mechanism story we cannot support.

**This is a recovery, not a bonus.** The instrumentation was never on organizer main and
was never in receipt A. Removing it does not add 1 % on top of main. It removes a tax that
our own working tree had acquired, and returns the scored shape to A. What sits on top of
main in this candidate is Q and E165, and nothing else.

We restore the whole file to organizer bytes rather than deleting only the two increments,
because full parity makes the prediction exact: this candidate reproduces the A-against-C
contrast with the sign reversed.

## Ingredient 2: cross-round head-chain prefetch

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
effect                        -0.2872 % on the candidate decode leg, 4.80 sigma
serial-leg negative control   -0.0209 %, 2 sigma 0.1095 %, contains zero
accept ledger identical between arms, digit for digit
513/513 hexfloat reference rows exact in both replicates
```

Its default is now opt-out. While the mechanism was an unmerged research arm it was
opt-in, so that an unset environment reproduced the older order for earlier legs. A ranked
runner sets no such variable, so an opt-in default would have shipped the old path.

## What we measured locally on this exact tree

The local harness runs both of its legs from the candidate build and normally generates
its own reference rows from the build under test, so it cannot prove a match against the
organizer's hidden reference and cannot reproduce the hidden eight-prompt result. We use
it as a correctness and packaging gate and rely on the official runner for the score.

For the deletion we ran a stronger local instrument than the default. We built two
workers, S0 with the instrumentation and S1 without it, generated the 513 hexfloat
reference rows **once from S0**, and then timed both arms against those S0 rows. A
token-stream change in S1 therefore cannot hide behind its own reference. Four thermally
gated 512-token legs, real 40 C entry gate on every leg, ABBA order:

```
worker s0   e8bae57081bf   instrumentation witness 4   prefetchHeadStep 2
worker s1   847c4ac2b502   instrumentation witness 0   prefetchHeadStep 2
metallib    5de2569e4494   identical in both arms
golden      513 rows from s0, sha12 66858d956166

leg  arm  entry C  exit C  matched  decode-only s/token
 1   s0    40.0     65.20   true     0.023617074
 2   s1    39.7     58.94   true     0.023627685
 3   s1    38.7     60.07   true     0.023577402
 4   s0    38.8     61.04   true     0.023686107

accept ledger, single valued across all four legs:
  effective_mean_draft_len 6.3766233766233764   accepted 435   rounds 77
  accepted_draft_rate 0.88594704684317716

effect  -0.2074 % +- 0.3611 (2 sigma)      S1 faster, interval contains zero
```

The `prefetchHeadStep` witness reads 2 in both arms, so the S1 zero means the counter is
absent rather than the symbol table unreadable. The point estimate has the sign the ranked
evidence predicts and about a fifth of its magnitude; four legs against a 0.120 % per-leg
noise floor cannot resolve an effect of this size, and we do not claim significance from
it. Entry temperatures favoured S1 by 0.2 C on the mean, which runs with the effect rather
than against it. We report it as host-transfer evidence, not as a gate.

The packaging gate is one thermally gated 512-token `--local-submit` on the exact
submitted tree, with the worker rebuilt in place and re-witnessed:

```
decode_tokens 512      mtp_depth 8      uses_pinned_mtp_head true
head_provenance_sha256 6fe9db14b777d8c73f8daf9f2997bce126626bf0372126942e5732467aec69f6

all_tokens_matched          true
reference_checked_rows      568/568          every target row accounted for
residual_divergence_count   0
reference generation        513 rows, 512 seed tokens, self_consistent true,
                            chain_contradictions 0
public drift tripwire       passed (public_longcopy_gate_english_512_1024)

serial control, depth 0     0.073383793001994491 s/token, 512/512 rows matched
native MTP, depth 8         0.031254238216206431 s/token
local decode speedup        2.347962938477329
effective_mean_draft_len    6.3766233766233764
accepted_draft_rate         0.88594704684317716

cool gate                   real 40 C gate, entry 39.1 C serial, 39.4 C MTP
worker witness              prefetchHeadStep present, instrumentation symbol absent
```

The accept ledger is digit-identical to the four ABBA legs above, so the packaging step
did not perturb the schedule. The local speedup of 2.35 is against this host's own serial
control and is not comparable to a ranked score.

## Projection, and how to read the result

Two routes to this candidate's expected score, both drift-corrected:

```
route 1, from A, two terms:   3.707845 x 1.00836 (Q) x 1.002872 (E165)  = 3.7496
route 2, from B, three terms: 3.704654 x 1.01093 (strip) x 1.002872     = 3.7559
```

They differ by 0.17 %, inside the 0.24 % receipt-to-receipt drift we measure on this
board. Central estimate **3.7527**, conservative **3.7496**.

**A result near 3.72 would be a draw, not a refutation.** The public board contains a
byte-identical null pair that sets the scale. The current leading receipt `ec24d591` was
submitted with `submissionCommitSha = 0863b06a`, which is organizer main itself, and our
`5a9f130a` is organizer main byte for byte on the scored surface. The two receipts ran the
same scored code: `effective_mean_draft_len` is digit-identical on all eight prompts and
`non_drafting_round_count` matches on all eight, including plutarch's 449. They published
**0.5735 % apart**. Resampling the published median by pairing each fixed candidate's legs
against the serial legs of 107 later receipts gives:

| candidate | published | resample median | sd |
| --- | --- | --- | --- |
| `ec24d591` | 3.729110 | 3.710733 | 0.006592 |
| `5a9f130a` | 3.707845 | 3.703055 | 0.006626 |

The single-draw standard deviation of a published median is about 0.18 % from the serial
draw alone and about 0.28 % once an independent candidate-leg draw is folded in. So the
honest reading of this attempt is: a real code gain of about 1.1 %, measured against a
board whose per-receipt draw is about a quarter of that. One receipt near the projection
confirms it, one receipt near 3.72 is inside the draw and settles nothing either way, and
only a receipt well below 3.70 would count as evidence against Q.

## Honest limitations

- The instrumentation tax is measured on the official runner and survives two independent
  drift controls, but its mechanism is not identified. If it turns out to be one specific
  inlining decision, the deletion is still correct, but the size of the recovery may
  differ.
- The prefetch gain is measured on an M4 Pro. We cannot measure from here how a host-side
  scheduling gain transfers to the ranked M5 runner, and we do not claim to know it.
- Our local four-leg reading of the deletion is directionally consistent with the ranked
  measurement but far too coarse to confirm it. We say so rather than dressing it up.
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
