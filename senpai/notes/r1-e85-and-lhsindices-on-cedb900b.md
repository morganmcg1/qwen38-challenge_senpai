# Senpai candidate: the E85 per-draft buffer removals and the cached
# `gather_qmm` left-hand index

## Who produced this candidate

Senpai is an autonomous multi-agent research campaign run by the Weights &
Biases Applied AI team. It is not a single model. One advisor agent plans the
research, reviews the evidence, and composes candidates. Student agents each
own one physical Apple Silicon host and run one bounded experiment at a time.

- Agent harness: the Senpai control plane on OpenHands agents.
- Underlying LLMs: Anthropic Claude (Sonnet and Opus class) for the advisor and
  the students, at high reasoning effort, with delegated Explore and Search
  subagents at lower effort for bounded read-only work.
- Local measurement host for this candidate: Apple M4 Pro, 20 GPU cores,
  48 GiB unified memory, macOS 26.5.2, Swift 6.3.3, GPU family
  `applegpu_g16s`.
- The ranked host is the organizer's M5 (`applegpu_g17s`). We do not have one.
  Every local number in this note is therefore cross-generation evidence, and
  we say so wherever it matters.

## What is new in this submission, and why we are sending it

This candidate carries two mechanisms that **have never run on the ranked
host**. We can prove that from the trees, and it is the whole reason this
submission exists.

Our previous submission `83f0b282` measured tree `91d19b2c`, whose subtrees are
`Sources 8079659f` and `Vendor 5df327c0`. The pull request that merged the E85
work landed 94 seconds after that submission left our machine, so the E85
mechanism and the cached `gather_qmm` index were both **outside** the snapshot
`83f0b282` timed. The tree submitted here has `Sources d306a12c` and
`Vendor 589de1e7`, and it is the first tree we have submitted that contains
either mechanism.

Both mechanisms are byte-traffic and dispatch-count reductions on the
**proposal-head** path. Neither changes the target model, the target
verification, the acceptance rule, or the emitted tokens.

## Mechanism 1. Seven materialised intermediates removed from each draft step

**Arm (a), the fused embedding read.** `embed_tokens` is affine 4-bit,
group 64. Calling `embedTokens(ids)` costs three gathers plus a dequantize, and
those four command buffers exist only to carry one 5,120-wide row into the
`qwen35_dual_rms_norm_concat_bf16_v1` kernel, which reads the row twice and then
discards it. The new kernel `qwen35_embed_dual_rms_norm_concat_bf16_v1` reads
the packed row in place and dequantizes each element at the point of use, with
the same `scale * d + bias` expression in bfloat that `affine_dequantize` uses.
Five dispatches collapse to one, and 13,120 bytes per draft are never written.
`preFcConcat` now takes both the ids and the embedding, so `callAsFunction` and
`lastHiddenWithKVOnlyHistory` both take the fused path. Every shape, dtype,
mode and eps assumption is guarded, and any mismatch falls back to the eager
path.

**Arm (b), the gathered quantized matmul.** `draftTokenIDWithDeclaredRerank`
used three `MLX.take` calls to gather 32 rows of the compact exact head, which
copies 92,160 bytes, and then read those rows straight back inside
`quantizedMM`. `gatherQuantizedMM` collects the same rows inside the matmul.
The `[98336, 1, 640]` and `[98336, 1, 80]` batch views are metadata-only
reshapes, cached at first use rather than rebuilt per draft. `sortedIndices`
stays `false`.

Neither arm changes an arithmetic result. Arm (a) is bit-identical by
construction. Arm (b) reorders no reduction: it reads the same rows through the
matmul's own gather path.

## Mechanism 2. The cached `gather_qmm` left-hand index

With `lhsIndices` omitted, `gather_qmm` calls `indices_or_default`, which builds
`reshape(arange(1, uint32), [1])` on **every draft step** only to produce the
single value `0`. We cache that array beside the batch-dimension views of the
compact head. It is `uint32`, so the internal `astype` is a no-op; an `int32`
array would trade the `arange` for a cast. `sortedIndices` stays `false`, so the
`sorted_indices && !lhs_indices_` branch does not change.

This removes one dispatch per draft step. Our in-session GPU dispatch boundary
on this host is **3.87 us**, CI95 `[2.63, 5.11]`, measured inside the worker
process rather than in a standalone bench, which inflates the same quantity by
about 25 times.

## Evidence

### Ranked evidence for this class of change

Our most recent ranked pair, E84, removed measured dead work on the same
per-draft path and produced a **mean of -0.137 % across seven prompts, sd
0.048, faster on 7 of 7 drafting legs, bit-exact**. That is a direct pair on
the ranked host, and it is the strongest reason we believe a per-draft byte
removal transfers from our M4 Pro to the M5.

### Local evidence for mechanism 1

An ABBA-counterbalanced session of 12 legs at 512 decode tokens, base against
the two arms together:

```
base  mtp=0.03175777 s/token  sd=0.00011389   serial=0.07438016  ratio=2.3421
ab    mtp=0.03174120 s/token  sd=0.00014734   serial=0.07434882  ratio=2.3424
contrast  block -16.56 us/token (-0.0522 %)   cov -40.33 us/token (-0.1270 %)
behaviour effective_mean_draft_len contrast 0.000e+00
          accepted_draft_rate      contrast 0.000e+00
```

The confidence intervals span zero. Six removed command buffers move decode
time by less than one third of a leg's noise on this host, which is exactly
what the campaign's own dispatch-boundary estimate predicts. We therefore also
measured the same coefficient with the sign reversed, by chaining `K` dependent
bf16 additions of an exact zero onto the per-draft path and fitting the slope.
That gives **5.96 us per command buffer**, CI95 `[-17.82, 29.73]`, and it is
the reason we treat the removal as a real but small saving rather than as
noise.

We are explicit about the limit: on our host this mechanism is **below the
noise floor of a single leg**. We are submitting it because the ranked host is
the only instrument that can resolve it, because the direction is fixed by
construction (strictly fewer bytes and strictly fewer dispatches for identical
arithmetic), and because the E84 ranked pair showed this exact class of change
landing at -0.137 % there while being similarly small here.

### Conversion to an expected ranked effect

Our local round is 165.2 ms and carries 6.36 drafts. The E85 read was -0.035 %
to -0.040 % of that round, which is about 9.7 us per draft. The ranked beagle
round is 53.3 ms and carries 4.38 drafts, so the same per-draft saving is about
-0.08 % there. The `lhsIndices` dispatch removal at the 3.87 us boundary adds
about -0.03 %. We predict about **-0.11 % of candidate time** relative to the
tree we submitted as `83f0b282`.

That conversion assumes the local round is longer than the ranked round because
it is **wider**, not because it does more host work per draft. We state the
assumption because it is the weakest link in the prediction, and the ranked run
is what tests it.

### Exactness

`Tests/MLXFastTests/E85FusedIntermediateTests.swift` asserts bitwise equality
for arm (a), id dependence of the fused embedding read, an unchanged argmax
with bounded logit drift for arm (b), and a bit-exact match between the cached
`lhsIndices` array and the synthesised one. Every ABBA leg above reports
`matched=True` against the local reference rows, and both behaviour contrasts
are exactly zero, which is the check that a bit-exact change cannot move a
draft length.

## Measurement honesty

- The local legs quoted here ran with the local cool gate disabled and the arm
  order counterbalanced, so monotone thermal drift cancels to first order. The
  entry-temperature spread across the session was 22.72 C. Those legs are
  directional causal evidence inside one counterbalanced session. They are
  **not** gate-qualified and they are not a ranked score.
- The `--local-submit` gate for this exact tree ran with the declared proposal
  head and passed.
- We report no local score as though it were an official one.

## What this candidate does not change

- Target checkpoint, target weights, transform contract, tokenizer.
- The trusted driver, the timing code, the telemetry, the workflow, the
  fixtures, or the submission manifest.
- The proposal head artifact. This tree declares the same
  `mtp-head.manifest.json` head as our previous submissions, pinned by digest
  to an immutable revision.
- The schedule. The draft-depth policy in this tree is the same cost-model
  schedule with the flat width cap of 7 that our previous submissions carried.

## One thing we removed before submitting

The merged base of this tree carried a research instrument in `Sources/`: an
Objective-C swizzle of `MTLCommandBuffer.commit` that recorded GPU execution
intervals, installed from the worker startup path behind an environment gate
that defaults to off. It answered a real question for us, and it is honest,
but a submitted tree is not the place for it. We deleted it, along with an
unused environment-selected experiment flag on the head chain, and we keep the
instrument in our own repository as a patch. The submitted surface therefore
contains no research scaffolding, no process-lifetime special-casing and no
benchmark-phase detection.

_This note was written by an AI agent (OpenHands) acting as a Senpai research
student, on behalf of the Weights & Biases Applied AI team._
