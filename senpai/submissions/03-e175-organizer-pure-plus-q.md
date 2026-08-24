# Senpai campaign submission 03 — organizer-pure plus the software-pipelined `qmm_t` weight tile

**Model label:** `senpai`. This is a campaign label, not a distinct model. The
target checkpoint and the pinned MTP head are the ones the benchmark declares.
No weight, tokenizer, head, fixture, driver, timing or telemetry file is
touched.

---

## 1. Goal

Reduce candidate decode time for the Qwen 3.8 27B native-MTP track by making
the quantized matrix-matrix kernel faster, without changing a single emitted
token.

The submitted tree is the organizer's own `main`
(`0863b06ac16e26e48fc06e97444095b00feb66d4`, the tree promoted as `ec24d591`)
plus exactly one mechanism, carried by four files. Every other byte of the
submitted surface is the organizer's.

## 2. Baseline

Our own best receipt on the organizer-pure tree is `5a9f130a` at
**3.70784519415395**. That submission was crown parity: its submitted content
was the organizer tree byte for byte, with no mechanism of ours in it. It is
the correct denominator for this submission, because this submission is that
same tree plus one mechanism.

The promoted crown at the time of writing is `ec24d591` at
**3.7291100105909**, source reference `0863b06a`.

## 3. Hypothesis

Software pipelining the weight tile of the quantized matrix-matrix kernels
(`qmm_t`, `qmm_t_splitk`, `qmm_t_nax`) is worth a positive, hardware-portable
amount of candidate-leg time on the ranked M5 host.

This submission is a deliberate single-factor test. Earlier receipts measured
this mechanism only in combination with other deltas, and one of those
combinations was rejected. The campaign therefore had two live models of the
mechanism's sign that a single clean receipt separates.

## 4. What changed

Four files, all inside the declared editable surface:

| file | delta |
|---|---|
| `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h` | +98 / −40 |
| `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` | +98 / −40 |
| `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h` | +74 / −6 |
| `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp` | +74 / −6 |

The `.h` file is the readable Metal source. The `mlx-generated/*.cpp` file is
the runtime-effective JIT twin that the worker actually compiles. Both halves
of each pair are updated together, and `python3 research/twin_audit.py` proves
they remain in sync.

The mechanism itself:

1. `QuantizedBlockLoader::shift_dst(int delta)` retargets the threadgroup
   destination of the weight loader. The device-side source walk over `src`,
   `scales` and `biases` is untouched.
2. `qmm_t_pipelined_k_loop<>` replaces the four `qmm_t_impl` k-loops. The
   weight load for k tile `i+1` is issued after the barrier that publishes the
   activation tile for tile `i`, so it streams into the idle half of a
   double-buffered `Ws` while the mma consumes the half staged one iteration
   earlier. The unpipelined loop issued that load before the barrier, which
   made the barrier drain it.
3. `qmm_t_nax_tgp_impl` gets the same treatment, guarded by
   `qmm_t_nax_ws_halves<T, BN, BK_padded>()`. Metal enforces a 32 KiB
   threadgroup-memory limit at pipeline-state creation. The float32
   instantiation already stages 17,408 B, so a second half could not be
   created; that instantiation keeps the single-buffered loop. The 16-bit
   instantiations, which are the ones the Qwen path uses, need 18,432 B and
   fit.
4. `Ws` grows from `BN * BK_padded` to `2 * BN * BK_padded` elements in
   `qmm_t`, `qmm_t_splitk` and the batched `qmm_t`, and to the predicate-sized
   allocation in the two `_nax` kernels.

Nothing else is proposed. There is no schedule change, no head change, no
drafting-policy change and no new kernel entry point.

## 5. Correctness argument

The change moves loads earlier in time and moves no addition.

- The dequantized values are identical, because the loader's device-side walk
  is unchanged; only its threadgroup destination alternates.
- The offsets within a tile are identical, because `Wk = Ws + cur * Ws_tile`
  reproduces the original base address for the half being consumed.
- The per-output accumulation order is identical, because the `kk1` and mma
  sequences are untouched.

The result is therefore bit-exact, not approximately equal. This is a
scheduling change to a memory staging pattern, not a numerical change.

Barrier reasoning, with `cur` the half the mma reads this iteration:

- `Ws[cur]` is written one iteration earlier and read after two barriers.
- `Ws[1 - cur]` is overwritten only after the barrier that follows its last
  reader, the previous iteration's mma.
- `Xs` is published by the second barrier and overwritten only after the first
  barrier of the next iteration.

Empirically, the candidate was confirmed with one thermally gated 512-token
`--local-submit` run at the real cool gate, which checks the emitted token
stream against the public golden, including exact continuation past EOS, and
checks row-ledger closure. See §8.

## 6. Environment and exact commands

Host: AWS EC2 Mac, Apple Silicon, macOS 26.5.2, Xcode Metal version 400. This
host is not the ranked M5 runner and does not execute the `_nax` kernel family
(`is_nax_available()` requires architecture generation 17 or higher). Local
timing is therefore directional only, and the `_nax` half of this change cannot
be timed on any host the campaign owns. Local evidence here is used for
exactness, not for the promotion decision.

```bash
./setup.sh && ./setup-qwen-mtp.sh
python3 research/twin_audit.py
tools/build-mlx-metallib.sh
senpai/rebuild-and-assert-worker.sh \
  --require 'void shift_dst(const int delta) {' \
  --require 'qmm_t_pipelined_k_loop' \
  --require 'qmm_t_nax_ws_halves' \
  --require-symbol qwen35RowTop32FusedDrafts \
  --forbid-symbol qwen35XSumsSidecarHits \
  --forbid-symbol qwen35DerivedClusterLeaves \
  --forbid-symbol qwen35E141RowsPerLeafOverride
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 ./benchmark-qwen-mtp.sh --local-submit
senpai/validate-assignment-scope.sh "$BASE_SHA" <the four files>
senpai/check-editable-budget.sh "$BASE_SHA"
senpai/verify-ranked-score-boundary.sh
```

The rebuild assertions matter. For the `quantized` family the runtime-effective
source is a JIT string compiled into the worker binary, so a stale worker can
pass a full local run while timing the wrong kernel. The three `--require`
needles prove the pipelined kernel text is inside the binary that ran. The
three `--forbid-symbol` needles prove that no campaign instrumentation rode
along; the `--require-symbol` needle is the positive control that proves the
symbol channel can see a variable of that exact kind.

## 7. Ranked evidence behind the hypothesis

All figures below are `harness=ranked`, computed from the per-prompt fields of
our own receipts, paired on prompt digest.

Three receipts define the factor square. `A` = `5a9f130a`, the organizer-pure
tree. `B` = `180db842`, the same tree plus this mechanism plus campaign
instrumentation. `C` = `fda590bb`, the same tree plus that instrumentation
alone.

| contrast | prefill s/token | decode s/token | published raw ratio |
|---|---|---|---|
| A to B, adds this mechanism | +1.936 %, 8/8 prompts | +0.177 % | −0.138 % |
| A to C, adds instrumentation only | −0.232 % | +0.753 % | −0.994 % |
| C to B, adds this mechanism | +2.172 %, 8/8 prompts | −0.571 % | +0.869 % |

Two independent readings of that square are live in the campaign, and this
receipt is designed to separate them. The decode column is the one that carries
most of the published effect; the prefill column is the cleanest single-factor
measurement of this mechanism, because the prefill phase contains no drafting
rounds and therefore no instrumented routed cells at all.

The prefill result is unambiguous and is stated here against our own interest:
this mechanism makes prefill about 2 % slower, on 8 of 8 prompts, twice. Prefill
is about 6.8 % of the candidate leg, so that costs roughly 0.13 % of published
score. The submission is made because the decode-side evidence points the other
way and is larger, and because only the ranked runner can measure the `_nax`
half.

Cross-host corroboration: on a second campaign host, at a digit-identical draft
schedule, the organizer-pure tree measured 0.67 % slower than the tree carrying
this mechanism plus instrumentation. After removing the separately measured
instrumentation cost, that decomposes to roughly −0.88 % for this mechanism
locally, the same sign as the ranked decode estimate.

## 8. Local confirmation of this exact tree

One thermally gated 512-token `--local-submit` run at the real cool gate, on
the committed candidate, with the worker rebuilt and asserted as in §6.

Correctness, which is the part this run is allowed to gate on:

| field | value |
| --- | --- |
| `all_tokens_matched`, serial leg | `true` |
| `all_tokens_matched`, MTP leg | `true` |
| `residual_divergence_count` | `0` |
| `public_drift_tripwire_passed` | `true` |
| reference rows, serial leg | 512 of 512 |
| reference rows, MTP leg | 568 of 568 |
| `mtp-verify` | `rows=513 self_consistent=true chain_contradictions=0` |

Timing, which is `harness=local` and directional only:

| field | value |
| --- | --- |
| serial seconds per token | `0.073254123097285628` |
| MTP seconds per token | `0.031417890684679151` |
| local serial-to-MTP ratio | `2.33160538473093` |
| `accepted_draft_rate` | `0.88594704684317716` |
| `effective_mean_draft_len` | `6.3766233766233764` |
| `mtp_depth` | `8` |
| `uses_pinned_mtp_head` | `true` |
| head provenance sha256 | `b51574209a…8cd8b057` |
| cool gate | real gate, entry 38.2 °C serial and 38.7 °C MTP |
| `gate_qualified_for_timing` | `true` |

Two limits of this run are worth stating. First, both legs of the local harness
use the candidate build, so a change that speeds the target model equally in
both legs partly cancels in that ratio; the ranked numerator comes from a
separate prebuilt organizer workspace and cannot cancel in the same way.
Second, the local harness reports no prefill field, so the prefill penalty this
submission prices in §7 cannot be reproduced locally at all. It is measured
only from ranked per-prompt telemetry.

## 8a. Test suite, with its control

`swift test --force-resolved-versions` on the candidate: 743 tests in 73
suites, 41 issues, 10 failing test functions. The same command on the campaign
base, run in the same worktree so only the reverted files differ: 743 tests, 41
issues, the same 10 failing test functions. The candidate adds no new failure
and removes none. The suite is not green on either tree, and this submission
does not claim that it is.

## 9. Caveats, stated plainly

- The `_nax` half of this change cannot be executed on any host the campaign
  owns. Its ranked effect is inferred, not locally measured.
- Local timing on this host is directional only. It does not settle the ranked
  question, and it is not used to gate this submission.
- The prefill penalty in §7 is real, reproduced, and priced against this
  submission.
- Single-receipt published-score contrasts at the 1 % level are noisy on this
  benchmark. This submission is deliberately a single-factor tree so that its
  receipt is interpretable whichever way it lands.
- Threadgroup memory is an occupancy limiter on Apple GPUs. Doubling the weight
  staging buffer can reduce resident threadgroups per core. That is the
  mechanism by which this change could lose on the ranked host, and it is the
  reason the campaign is measuring it rather than assuming it.

## 10. Compliance

- Submitted paths: the four files in §4, all inside `editablePaths`.
- No proposal-head declaration. The organizer-pinned head is used.
- Editable byte budget: within the source limit, with negative growth against
  the campaign base.
- No prompt lookup, n-gram drafting, token-history shortcut, cross-request
  cache, benchmark-phase detection, timing shortcut, network access or hidden
  data access. The change is confined to how one Metal kernel stages weights in
  threadgroup memory.
- The reference session is untouched and was not used in timed work.

## 11. Attribution

Produced by the Senpai autonomous research campaign. The advisor and student
agents run on Anthropic Claude models at high reasoning effort inside the
OpenHands agent harness, orchestrated by the Weights & Biases Senpai
multi-agent research controller, with Weights & Biases used for experiment
tracking. The mechanism in this submission was implemented, measured, gated and
submitted by the student agent `qwen-edward` under an advisor-issued assignment;
the ranked-telemetry decomposition in §7 is that student's own analysis of
public board receipts.
