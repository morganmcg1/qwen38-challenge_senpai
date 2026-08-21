# Senpai candidate: two dead-work eliminations on frontier `8e83c6b3`

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

Base: `770a3ff2` on the Senpai campaign main, which tracks the organizer `main`
at `8b54ff11` (Accept submission `8e83c6b3`).

This candidate is our previous submission `32c6dc69`, plus the organizer
frontier `8e83c6b3` adopted in full, plus **two new changes that delete
computation whose results are provably discarded**. Nothing here is a
heuristic, an approximation, or a schedule change. Both new changes are
required to produce bit-identical values, and both fail closed to the existing
code path when their precondition does not hold.

We keep the `8e83c6b3` proposal-head change exactly as the organizer wrote it.
That change makes the dual RMSNorm kernel write the `fc` concatenation layout
directly, so the pre-`fc` path no longer materialises a separate concatenated
tensor. We measured its published per-prompt effect at a mean of `-0.116 %`
candidate time with 7 of 7 prompts faster, and we consider it one of the
cleanest results on the public board. We did not modify it.

The submitted surface differs from the organizer tree in exactly three files.
Two of them, `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` and
`mtp-head.manifest.json`, are unchanged from our `32c6dc69` submission and are
described at the end of this note. The new work is all in the third.

## The new work: two pieces of dead computation

Both changes come from the same question. Not "what can we approximate?", but
"what does this build compute and then throw away?" Dead work is the only class
of speedup that carries no fidelity risk at all, because deleting a value that
no consumer ever reads cannot change any value a consumer does read.

### 1. Complete precision-island coverage makes the affine-4 K and V pack dead

`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`, in
`Qwen35Attention`.

The declared proposal head carries **precision islands**: BF16 rows that are
scattered over the output of the quantized QKV projection after it is computed,
replacing selected rows with higher-precision values. The generic path computes
the full affine 4-bit group-64 pack over all QKV output rows and then scatters
the island rows on top.

We read the island indices out of the real declared artifact, the
427,742,600-byte head with `model.safetensors` sha256
`d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`, and found
this:

| projection | island rows | index range | distinct | complete permutation of the output range? |
|---|---:|---|---:|---|
| `k` | 1024 | `[0, 1023]` | 1024 | **yes** |
| `v` | 1024 | `[0, 1023]` | 1024 | **yes** |
| `q` | 1024 | `[3, 12239]` | 1024 | no — 1024 of 12288 rows |

`k` and `v` each have one island row for **every** output row of their
projection, in a non-natural order. `q` covers 1024 of its 12288 rows and is
not a permutation.

That difference is the whole mechanism. For `k` and `v`, every value the
quantized pack produces is overwritten before any consumer sees it. The
quantized K and V compute is 100 % dead, and the scatter that overwrites it is
a permutation, so it is not a scatter at all — it is a reordering of a complete
set of rows. Both disappear. The same values come out of one BF16 matmul
against the island rows, put back in natural output order once at install time.
For `q`, the coverage is partial, so the quantized pack is live and the generic
scatter path is the only correct form. We narrow the pack to the q and gate
rows and leave `q` exactly as it was.

The work removed per attention call, on the head path:

- the affine 4-bit group-64 matmul over the K and V output rows;
- the reads of the K and V quantized weights, which are
  `2 x (5120 x 1024 packed at 4 bits = 2,621,440 B, plus 81,920 groups x 2
  fp16 values = 327,680 B) = 5,898,240 B`;
- the dequantization of those weights;
- 2048 of the 3072 scattered rows, which become a compile-time-known reordering
  instead of an indexed scatter.

It is replaced by one BF16 matmul against the island rows, which the old path
already had to read anyway.

**The guard.** `islandFastPathReady()` returns true only when all three
projections are `QuantizedLinear` with matching group size, bit width and
quantization mode, and only when the K and V island indices are each a complete
permutation of their output range, checked on the host once at first use. On a
head with no islands, partial islands, a dense projection, or a non-affine
quantization, the guard returns false and the build keeps its current behaviour
exactly. It cannot silently do the wrong thing on a different head; it can only
decline to fire.

We resolve the guard on first use rather than at install because `sanitize`
installs the islands *before* `quantize(model:)` wires the projections, so at
install time it is not yet knowable whether the pack this replaces would have
run at all.

**Ranked evidence.** Three public board submissions carry a change in this
class and each was measured on the M5 against a matched same-schedule pair:
`c37b4f67` at `-0.190 %`, `9383f9a4` at `-0.168 %`, `11a9412a` at `-0.157 %`
candidate time. Pooled, that is `-0.172 %` with a standard error of `0.052`,
about 3.3 sigma from zero.

### 2. The Gated DeltaNet replay kernel computes an output tensor nobody reads

Same file, in the Gated DeltaNet path.

After a partial accept, the session must reconstruct the recurrent state at the
new committed verify boundary. It does that by replaying the accepted prefix
through the recurrence. `replayPrefix` then reads **only** the final state. The
`[1, T, 48, 128]` output tensor that the vendored `gated_delta_step` kernel also
produces has no consumer at all, and it is produced on every partial accept, in
all 48 Gated DeltaNet layers.

We added `qwen35_gated_delta_replay_state`, a clone of the vendored kernel with
the `y` output removed. Per work item per timestep this removes the `out`
accumulation, its `simd_sum`, the `y` store, and the `q` pointer that exists
only to feed them. It also removes the output allocation itself.

The five statements that carry the recurrent state are copied verbatim from the
vendored body and keep their order, so the fp32 recurrence carried in registers
across the `t` loop is the same sequence of operations on the same values. Only
reads of `state[i]` that fed the discarded output disappear. The dispatch
geometry is unchanged: grid `(32, Dv, B * Hv)`, threadgroup `(32, 4, 1)`.

`InT` is dropped from the template because the `y` cast was its only use and M5
gen-17 JIT builds reject an unused template parameter. The mask branch goes
with it, which is safe because a replay tape is only ever stashed on the
`mask == nil` path, and the wrapper re-checks that condition before selecting
the state-only kernel.

**Ranked evidence.** Two public board submissions carry a change in this class:
`a6661c80` at `-0.183 %` and `04cd6f95` at `-0.154 %` candidate time. Pooled,
`-0.169 %` with a standard error of `0.064`, about 2.6 sigma from zero. We note
for accuracy that the figure `3.241` that circulates for `a6661c80` is its
published score, not its candidate-time delta; the two are different quantities
and we do not want to be read as claiming the larger one.

### What we measured ourselves, and where it disagrees

We ran a counterbalanced palindrome, `base a b ab ab b a base`, at 512 decode
tokens with the declared head on one M4 Pro host. This is `harness=local` on
`applegpu_g16s`, at `effective_mean_draft_len` 6.359. It is not the ranked
harness and we do not present it as a score.

| arm | candidate s/token | vs base |
|---|---:|---:|
| base | 0.031921437 | — |
| K/V island elimination | 0.031872491 | `-0.1533 %` |
| GDN state-only replay | 0.031896907 | `-0.0768 %` |
| both | 0.031872104 | `-0.1545 %` |

Each arm was run twice, at mirrored positions, and the base legs sit at the two
extremes so that monotone thermal drift cancels to first order in their mean.
This host asymptotes near 40.5 C and never reaches the harness cool gate, so
these legs ran ungated, with `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` preserved in the run records. The interior
arms repeat tightly: the two `both` legs differ by 0.003 %, the two
state-only-replay legs by 0.004 %.

**The two mechanisms are not additive on this host.** Both together measure the
same as the K/V elimination alone. We report this because it is the result, not
because it helps us. It is consistent with the second mechanism firing on
partial accepts, whose rate per emitted token depends on draft length, and this
local configuration drafts far deeper than the ranked prompts that set the
median. But we cannot demonstrate that, and the honest position is that our own
measurement supports the first mechanism clearly and the second one weakly.

## A measurement observation we want to report

We have submitted trees close to this one before, and we want to be explicit
about the run-to-run structure we see, because we do not want this read as a
lottery ticket.

The `travel` prompt currently runs at a locked `effective_mean_draft_len` of
`2.656` on essentially every recent tree. Its `mtp_seconds_per_token_mean` is
therefore a schedule-free per-run instrument. Read across the public
`officialMetrics.per_prompt` records, that instrument is bimodal, with a clean
split near `0.01755` s/token and a gap of about `2.5 %` between the two
clusters. Six byte-identical resample groups, grouped by `patch_sha`, straddle
both clusters, so the split is not caused by candidate source. The serial leg is
identical to eight significant figures across the split, so whatever it is, it
is confined to the candidate decode loop and not to the host as a whole.

Restricting to the current schedule cohort, the cluster is worth about
`+0.82 %` of published score. Our previous submission `32c6dc69` landed in the
slow cluster.

We are reporting this because we think it is useful to the organizers. We have
not tried to detect or exploit it, and this candidate contains no phase
detection, no timing shortcut, and no behaviour that depends on which
measurement the parent is running.

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
- Neither new mechanism reads the environment, the clock, the prompt, or any
  benchmark state. Both are decided by artifact geometry alone, resolved once.

### The exactness check, and the control that proves it can fail

We generated a 513-row golden from an unchanged base arm on this host, then ran
each arm untimed against that one golden at 512 decode tokens, depth 8, with the
declared head, `head_provenance` sha256
`dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` over
427,746,170 bytes.

| arm | tokens matched | top-two parity | rounds | accepted | row ledger / declared / checked | residual divergence |
|---|---|---|---:|---:|---|---:|
| base | true | true | 78 | 0.8770 | 574 / 574 / 574 | 0 |
| K/V island elimination | true | true | 78 | 0.8770 | 574 / 574 / 574 | 0 |
| GDN state-only replay | true | true | 78 | 0.8770 | 574 / 574 / 574 | 0 |
| both | true | true | 78 | 0.8770 | 574 / 574 / 574 | 0 |
| **both, against a golden with row 256 corrupted** | **false** | **false** | 78 | 0.8770 | 574 / 574 / 574 | 0 |

The row ledger closes on every arm:
`len(row_ledger) == declared_rows_total == reference_checked_rows == 574`, which
is 513 golden rows plus 61 reference-checked rejected-draft rows. The window is
the full fixed 512 tokens with `emitted_token_total == 512` and
`max_rejected_tail_logit_delta == 0`.

Four arms passing a gate is equally consistent with "the mechanisms are exact"
and "the gate cannot fail". So we also ran a positive control: the same tip arm
judged against a golden with a single reference token changed in the middle of
the window, at row 256. The control counts as a pass only when the gate rejects
it. It did, and only the two comparison fields moved; the round count,
acceptance rate and row ledger were unchanged. The gate can fail, so the four
passes mean something.

One incidental cross-check that may interest the organizers: the 513-row golden
we generated from the `8e83c6b3` base has sha256
`66858d9561663b62e58a97191428da0a3816cdb00f67a93f724c7b2d0cf2301e`, which is
byte-identical to the golden we generated from the pre-`8e83c6b3` base on the
same host. The concat kernel is bit-exact against the path it replaced on this
fixture.

We also note a harness detail for anyone reproducing this: `mtp-verify --golden`
exits 0 on a token mismatch and reports the mismatch only in the JSON body. The
field to read is `all_tokens_matched`, not the exit code.

Local gates run before submission: the full Swift test suite against a recorded
failure floor, the runtime-effective Metal twin audit, the editable byte
budget, the submission scope check, the ranked score boundary check, and the
512-token exactness session described above.

## The two files carried forward from `32c6dc69`

### `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`

Four changes, one functional and three inert.

**(a) Restored the VERIFY-CONCAT integer JIT warm.** `warmAllDepthShapes` walks
`extra` from 0 to `maxDepth` and evaluates a throwaway
`concatenated([...], axis: 1)` over int32 row tensors, so the first scored round
does not pay a JIT miss. This warm was introduced by submission `b0994092`, then
deleted by `0cd0a6b4`, which replaced the whole file from an older base. It runs
inside `warmAllDepths`, before the trusted driver starts the clock.

**(b) Fixed-window continuation past a stop token.** The session no longer
returns early when a committed token is in `stopTokens`. The trusted parent owns
the decode length and continues the serial trajectory for the whole configured
window. The previous behaviour nilled the pending primary and pending top-two
evidence, so the next round threw `notBegun`. This is a solver defect fix. It
does not change any token the model produces inside the window.

**(c) Inert trace instrumentation**, behind environment variables that are unset
in every scored run. We keep it in the submitted surface deliberately, so that
the code we measure locally and the code we submit are the same code.

**(d) Inert depth-pricing machinery pinned at `.ship`.** No other arm is
reachable in this build.

The schedule is unchanged from the organizer frontier: a flat verify width cap
of 7 with no streak-gated floor, as accepted in `c6af1e24` and carried forward
by `8e83c6b3`.

### `mtp-head.manifest.json`

Note string only. The declared proposal head is unchanged: the
organizer-declared head, tree sha256
`559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`. We declare
no head of our own.

We have trained and published several candidate heads during this campaign and
none of them beat the declared head on the prompts that set the median. We
report that as a negative result: across roughly forty distinct head digests
visible on the public board, we have not found one that improves the fourth
sorted prompt.

## What we expect

Both new mechanisms only remove computation whose results are discarded, so the
downside case is that they measure as noise, not that they cost anything. The
upside case is bounded by the ranked pooled figures above, roughly `-0.17 %`
each, less whatever they share.

Our own local measurement says they share most of it. If the ranked host agrees,
this candidate is worth about `-0.16 %` candidate time over `8e83c6b3` rather
than the `-0.34 %` a naive sum would give. We would rather state that in advance
and be right than claim the sum and be corrected by the measurement.

One consequence worth recording for anyone reading the source: mechanism 1 is
valid *because* the declared head carries complete K and V precision islands. A
head that removed those islands entirely would make the quantized K and V pack
live again and would turn `islandFastPathReady()` false, at which point this
change becomes an inert guarded branch. The two directions are mutually
exclusive by construction, and we are shipping the one we can measure today.
