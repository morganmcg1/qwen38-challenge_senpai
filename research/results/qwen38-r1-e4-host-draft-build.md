# E4 — Host-bound draft build: decomposition and irreducibility

**Assignment** `qwen38-r1-e4-host-draft-build` **r2** · PR #4 · student `qwen-askeladd`
**Base** `67bde70274c42aef089ac73cf00608d8037a815e` (`senpai/qwen38-mtp-r1`) ·
**Upstream** `7351e62674bc600f0ca148d3a1b0604716a09db6`
**Rebase provenance** r1 evidence was cut on `e20268e9…`; this report is replayed on the
r2 base. See §0 for exactly what changed and which numbers were re-measured.
**Host** Apple M4 Pro (Mac16,11), `applegpu_g16s`, 48 GiB, macOS 26.5.2 (25F84), Swift 6.3.3
**Head** unchanged from base. Local runs use the organizer bf16 head
(`EigenLabs/Qwen3.8-27B-MTP-bf16@26a328e0`, sha256 `8fceddc6…`); the ranked candidate leg
uses the declared 4-bit head (`hf:lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaf`,
sha256 `cc209e30…`). Full provenance in §8.2b; consequences in §6b.1 and §6c.

**Label: `not useful`** — the premise of the assignment is refuted. `draft_build_us` is
not host graph-construction time; it is ~96–98% GPU execution wait. The optimizations
scoped in Part B are bounded above by ~0.35% of round time, an order of magnitude
below the ≥3% promotion bar.

---

## 0. r2 rebase — what changed, and what it did or did not invalidate

r2 rebound the assignment from `e20268e9…` to `67bde702…`. I rebased
`qwen-askeladd/host-bound-draft-build` onto the new base; the rebase applied **cleanly**,
with no conflicts, and the candidate diff is byte-identical in shape to r1
(`110/4` in `Qwen36MTPBlockSession.swift`, `67/3` in `Qwen35MTP.swift`).

Because that rebase moved the branch off the r1 base, the branch head no longer descended
from the published remote head `1d7c8a03…`, so the submission lease could not
fast-forward. `1d7c8a03…` is the **empty** assignment-scaffold commit — `git diff
e20268e9 1d7c8a03` is byte-for-byte empty — so I recorded it as an ancestor with a
`-s ours` merge rather than rewriting or discarding published history. The merge changes
no file: the tree SHA is `2aae65eb…` both before and after. No r1-base content is
reintroduced, and the diff against `67bde702…` is unchanged.

Between the two bases exactly **three** non-documentation files move:

```
.gitignore                                        +2  -0
Sources/MLXFastModel/Qwen36MTPBlockSession.swift  +24 -53
Tests/MLXFastTests/QwenMTPFixedWindowTests.swift  +31  -0
```

Only `b219009` ("qwen: continue fixed decode windows past EOS") touches a file I also
edit. Re-located **by symbol name**, as instructed, it does five things:

Sites are line numbers **on the r2 base**, `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`:

| change | site on r2 base | effect on this experiment |
|---|---|---|
| `stopTokens` stored property removed; `init` takes `stopTokens _:` | `:171` | none — never read by instrumentation |
| `reachedStopToken` stored var → computed `{ false }` | `:167` | none |
| early-return block when the **primary** is a stop token | deleted (sat between my `tRound0` and `tPrologueDone`) | **removes a round shape** my trace could previously see; it fired in 0 of 28 r1 rounds anyway |
| accept walk → `static acceptedDraftPrefixCount(drafts:verifyArgmax:)` | defined `:641`, called `:931` | none — same prefix rule minus the stop-token `break` |
| `stoppedEarly` suppression dropped from `recordAcceptOutcome` | `:610` | affects the accept-EMA, hence the depth schedule, on any round that would have committed EOS |

**What this invalidates — and the one thing I could not deliver.** Nothing structural.
But r2 asks for headline numbers at 512 decode tokens, and **I could not produce them.**
I rebased, rebuilt cleanly on the r2 base (48.9 s, both binaries), re-checked scope and
budget, and launched the 512-token traced run — and it died in the pre-timing cool gate,
as did two earlier attempts. The host's *idle* GPU floor is 40.17 °C against a ≤40.0 °C
gate, held there by a runaway root-owned `WirelessRadioManagerd` at ~100% CPU that I
cannot kill without sudo. To prove this is the host and not my workload, I then ran the
gate **by itself** — `./benchmark.sh --local-cool-gate-only`, no model loaded, no GPU work
of mine anywhere — and it still failed, reporting `min seen 40.6C` after 180 s of doing
nothing. **§8.4 gives the exact commands, the complete output, the root cause, and each
escape route I refused to take.** I did not bypass the gate, and I did not falsify the
sensor.

So the honest status of each number in this report:

| section | status on the r2 base |
|---|---|
| §3, §4, §6-guardrail (round timing) | **carried over from the r1 192-token run**, not re-measured. Not re-run — see §8.4. |
| §5A, §5B, §6c, §6d (kernel isolation) | **base-independent and re-verified.** They call `mx.quantized_matmul` at fixed shapes and never enter `Qwen36MTPBlockSession`. |
| §8.4.4 (context sensitivity) | **new, measured on the r2 base**, added specifically to bound the 192→512 extrapolation. |
| scope + budget | **re-run against `67bde702…`**, both exit 0 (below). |

**§8.4.4 is the part I could still run, and it bounds the gap.** The one thing a 512-token
window changes relative to a 192-token one is context: the KV cache walks 512→1024. That is
a device-side question, so I measured it directly with the gate-free component harness at
kv ∈ {512, 640, 768, 896, 1024, 1280}. Over the full walk the round floor moves **+0.377%**
(226.555 → 227.409 ms); only the two SDPA kernels move at all; the 48 Gated DeltaNet layers
are byte-identical; and the byte model matches the analytic prediction **exactly**
(67,108,864 B both ways). The host-only share of the round therefore goes 0.350% → 0.349%.
That does not replace an end-to-end 512-token leg, but it does mean the extrapolation I am
asking the advisor to accept is worth **0.001 percentage points**, against a hypothesis that
is wrong by **~70×**.

Two reasons the carried-over timing is still sound, stated as argument rather than
measurement so the advisor can discount it as they see fit:

1. **The r1 window is clean.** The EOS defect kills runs near decode token ~301; the r1
   window is 192 tokens, so it completed with 28 rounds, `all_tokens_matched: true` and
   `residual_divergence_count: 0`.
2. **`b219009` is timing-neutral-to-cheaper on the measured path.** It deletes a stored
   property write (`:167`), deletes an early-return branch that sat inside my *prologue*
   phase — which measures **1.8 µs**, and fired in 0 of 28 r1 rounds — and extracts the
   accept walk into a static helper doing the same work. Every one of those can only make
   host-side work equal or smaller, which strengthens rather than weakens the finding that
   host graph build is 0.35% of the round.

**Scope and budget, re-checked against the r2 base:**

```
senpai/validate-assignment-scope.sh 67bde702… \
    Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
    Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift
  -> assignment scope OK: 2 submitted path(s)                        (exit 0)

senpai/check-editable-budget.sh 67bde702…
  -> editable budget OK: source=2402616/3000000 headroom=597384
     growth=7966/262144 exempt=2410/2147483648 files=154             (exit 0)
```

**A protocol failure I must declare.** r2 asks me to *post* the provenance and the Part A
breakdown as PR comments before optimizing. **`post_assignment_comment` is not in my tool
schema this session**, and the harness forbids reproducing it with `gh` or REST. I could
not comply. Both artifacts are instead delivered here (§8.2b provenance, §3 breakdown)
and in the terminal result. This is a tooling gap, not a decision to withhold; if
`qwen-edward` needs the breakdown before this PR is reviewed, it is in §3 and in the
committed file at `research/results/qwen38-r1-e4-host-draft-build.md`.

---

## 1. Headline

The assignment estimated "~2.4 ms per draft step of host-side graph construction."
That number is real as a *per-step marginal cost*, but it is **not host time**.
Measured host graph construction is **~33 µs/step** in steady state (~85 µs/step
including warmup) — a **~70×** overestimate. The remaining ~2.35 ms/step is proposal-head
weight streaming and target work on the GPU, waited on inside `asyncEval`.

Consequently the three Part B mechanisms cannot pay:

| Part B mechanism | Upper bound on gain | Verdict |
|---|---|---|
| 1. Hoist loop invariants into session state | ≤ 0.35% of round time | below noise |
| 2. Reuse pre-built `MLXArray` operands | ≤ 0.35% of round time | below noise |
| 3. Cross-step pipelining | n/a — true data dependency | not implementable |

This is **outcome #3** from the PR body: a breakdown proving irreducibility, retiring
the line of work and re-anchoring the cost model.

Five results outrank that negative and are where the campaign value is:

1. **§6c.4 — the depth cap sits exactly one row past a dispatch cliff, and moving it is a
   two-line change.** This host is `applegpu_g16s` (arch_gen 16, size `'s'`), so
   `get_qmv_batch_limit` returns **8** for every projection shape in the model, and MLX
   switches `qmv -> qmm_splitk` at `M >= 8`. `rows_per_round = depth + 1`, so
   `segmentedVerifyDepthCap = 8` verifies at **M = 9**, one row over. The 9th row costs
   **1.89x** an average row (38.46 ms marginal vs 20.32 ms average), and the traced run
   agrees end-to-end: **d=7 is 23.79 ms/token vs d=8 at 24.49 ms/token — 2.9% cheaper**,
   despite emitting fewer tokens per round. This also **falsifies the design comment at
   `Qwen36MTPBlockSession.swift:588-590`** ("qmv batch limit 10+ on this generation for
   these shapes"). Correctness is untouched; only the cost model behind the cap is wrong.
   See §10 follow-up 0b — measure the limit on the ranked host, then set the cap from it.
2. **§5B — `quantized_matmul` collapses at exactly the widths MTP verifies at.** It runs
   at 248.7 GB/s at M=1 but 72.1 GB/s at M=9, moving ~3.45x the necessary weight bytes.
   Verify-side quantized matmuls are 201.14 ms of the 217.75 ms round; at roofline the
   round would be ~87.1 ms — a **2.50x round speedup**. The fix belongs in the *submittable*
   `kernels/quantized*.metal` + generated twins; the qmv/qmm selector that steers into the
   cliff is in non-editable host code, so the work must go **inside** `qmv`/`qmv_quad`.
   Follow-up 0b is the cheap half of this: it dodges the cliff instead of fixing it.
3. **§6b.1 / §6c — the declared proposal head is already 4-bit g64, not bf16, and it is
   bandwidth-bound.** The "quantize the head, save 11-16%" follow-up has **no prize left**;
   it is banked in the base. §6c.2 additionally refutes an 8-bit head: 4-bit qmv sustains
   249.0 GB/s versus 253.1 GB/s for bf16 dense matvec (98.4%), so there is no nibble-unpack
   penalty to buy back at M=1 and a wider head would be strictly slower.
4. **§6d.4 — on the ranked build the compact draft readout, not the head, is the biggest
   remaining drafting-bandwidth item.** It is 283.21 MB of the 522.09 MB/step ranked
   drafting total = **54.24%**, it is a slice of the *fixed target* `lm_head` so head
   quantization cannot shrink it, and it already runs at 100.4% of roofline. It is also the
   most compressible thing in the path, because proposals do not have to be exact — the
   target verifies every token. This confirms the 522.1 MB/step figure in feedback (8) to
   0.003%. Sized design in §10 follow-up 0d (≈ **+0.058 score** at `d=8`).
5. **§6b.4 — the pre-registered discriminator resolves to BANDWIDTH**, but the residual's
   real cause is small-M kernel inefficiency, a third category the dichotomy omits.

## 1a. Corrections to feedback (4) and (9): `MLX_QWEN_MTP_TRACE=1` **is** reachable

Feedback (4) concluded the trace is unreachable and that Part A "needs a new method".
Feedback (9) restates that conclusion. The *diagnosis* behind it is correct; the
*conclusion* does not follow. I hit exactly that wall, diagnosed it identically, and
solved it — **every number in this report comes from the live scored worker via that
route.** All line numbers below are re-located by name on the r2 base `67bde702`.

**What the advisor got right.** Worker stderr really is discarded:

- `Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift:2046` and `:2207` both spawn the
  worker with `emit: options.forwardsWorkerStderr ? nil : { _ in }` — the drain reads
  stderr and throws it away.
- `Sources/MLXFastCLI/main.swift:2224` defaults `forwardsWorkerStderr: Bool = false`, and
  `:2301` further ANDs it: `forwardsWorkerStderr && !officialRun`. Only the DFlash path
  (`main.swift:1409`, inside the cited `:1404-1412`) passes `true`.

So a stderr-based trace is indeed invisible. **The trace does not use stderr.**

**Why it is nevertheless reachable — two independent facts.**

1. **The sink is a file, not stderr.** `Qwen36MTPBlockSession.swift:475-487` opens an
   append-mode `FileHandle` on the absolute path named by `MLX_QWEN_MTP_TRACE_FILE`.
   Discarding the worker's stderr has no effect on a file the worker opens itself.

2. **Both variables survive the worker env filter, and this is provable rather than
   assumed.** `sanitizedRuntimeWorkerEnvironment` (`QwenRuntimeWorker.swift:2623`,
   applied at `:2036` and `:2191`) is a documented **strict allowlist** that starts from
   an empty environment. Its `allowedPrefixes` list at `:2638-2645` contains `"MLX_"` at
   `:2643`. `MLX_QWEN_MTP_TRACE` and `MLX_QWEN_MTP_TRACE_FILE` both match that prefix and
   are copied through. The filter's own doc comment at `:2600` makes the trap explicit —
   `"MLX_"` does **not** match `MLXFAST_*` — which is exactly why the sink-path variable
   is `MLX_`-prefixed and not `MLXFAST_TRACE_FILE`.

**The recipe:**

```bash
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_FILE=/abs/path/trace.log   # worker appends directly to a file
export MLXFAST_NO_SANDBOX=1                          # sandbox denies file-write* otherwise
./benchmark-qwen-mtp.sh --local-iterate
```

`MLXFAST_NO_SANDBOX=1` is set in the **parent** (it is a harness variable and is
deliberately *not* forwarded to the child) because the worker's Seatbelt profile otherwise
denies `file-write*`; without it the open silently fails and the file stays empty.
The sink is guarded by an `NSLock`, so the two worker sessions (serial control, then the
MTP leg) interleave safely into one file, tagged by session index.

**Empirical proof, not just a code argument:** the r1 traced run emitted 28 complete
`mtp-round` records plus per-round sub-records from the live scored worker. A method that
was unreachable could not have produced them. Sections 3, 4 and 6 are computed from that
file.

**Phase-oracle note (unprompted, because the allowlist doc raises it).** `MLX_`-prefixed
names are allowlisted precisely because they are phase-independent, and the ranked
workflow never sets either trace variable, so this instrumentation cannot act as a phase
oracle. It is nonetheless labelled not-for-promotion in §9 and is gated off by default
(`traceRounds` is `false` unless `MLX_QWEN_MTP_TRACE == "1"`).

This is reusable by any student who needs worker-side instrumentation — it also unblocks
the same measurement on PR #3. `scratch/run-bench.sh` wires it up end to end.


## 2. Why `asyncEval` is not free (mechanism)

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/transforms.cpp`:

- `async_eval()` (~line 322) calls `eval_impl(outputs, true)`.
- `eval_impl` walks the tape **on the calling thread**, calling `gpu::eval(arr)` per
  array, then `gpu::finalize(s)`. Metal encoding is therefore synchronous.
- Line 270 throttles the caller:
  ```cpp
  if (scheduler::n_active_tasks() > MAX_ACTIVE_TASKS ||
      (get_active_memory() > get_memory_limit() && n_active_tasks() > 0)) {
    finalize; scheduler::wait_for_one();
    while (active_memory > limit && n_active_tasks() > 0) wait_for_one();
  }
  ```
  with `MAX_ACTIVE_TASKS = 10` (line 25).

So "async" eval = synchronous Metal encoding **plus** allocator/queue-depth blocking.
The direct evidence is that the *same* call costs radically different amounts depending
on whether the GPU is idle or busy:

- step-1 `asyncEval` (GPU idle): **~0.58 ms**
- tail `asyncEval` after d−1 deep steps (GPU saturated): **~15.7 ms** mean, up to 25.0 ms at d=8

Host-side work is identical in both cases. The difference is queue drain.

## 3. Part A — `draft_build_us` decomposition

192-token gated `--local-iterate` run, commit `c39e178`, 28 rounds, free depth schedule.
Times are ns counters converted to µs; `N` = rounds.

### All rounds (N=28)

| sub-step | mean µs | share of draft | share of round | method |
|---|---:|---:|---:|---|
| `prologue` | 1.8 | 0.0% | 0.00% | mach ns delta |
| `flushprep` | 13.8 | 0.1% | 0.01% | mach ns delta |
| `s1_fwd` (head forward build) | 119.5 | 0.7% | 0.07% | mach ns delta |
| `s1_slice` | 1.3 | 0.0% | 0.00% | mach ns delta |
| `s1_select` (`draftTokenID`) | 24.2 | 0.1% | 0.01% | mach ns delta |
| `s1_async` (**GPU wait**) | 563.5 | 3.2% | 0.32% | mach ns delta |
| `deep_fwd` (all deep steps) | 328.7 | 1.9% | 0.19% | accumulated |
| `deep_slice` | 6.0 | 0.0% | 0.00% | accumulated |
| `deep_select` | 89.8 | 0.5% | 0.05% | accumulated |
| `tail_async` (**GPU wait**) | 16336.6 | 93.4% | 9.29% | mach ns delta |
| **HOST-ONLY sum** | **585.2** | **3.3%** | **0.33%** | sum of non-async |
| `draft_build_us` | 17486.2 | 100.0% | 9.94% | wall |

### Steady state, round > 1 (N=27)

| sub-step | mean µs | share of draft | share of round |
|---|---:|---:|---:|
| `s1_async` + `tail_async` (GPU wait) | 16328.6 | 96.5% | 9.47% |
| **HOST-ONLY sum** | **599.0** | **3.5%** | **0.35%** |
| `draft_build_us` | 16928.6 | 100.0% | 9.81% |

**Host graph construction is 0.35% of round time.** Eliminating *all* of it — a
physically impossible optimization — would move the candidate leg by 0.35%.

### Per deep step

| quantity | mean over 140 deep steps | steady-state sample (round 27, d=8) |
|---|---:|---:|
| `deep_fwd` µs/step | 65.75 | 25.0 |
| `deep_select` µs/step | 17.96 | 7.5 |
| `deep_slice` µs/step | 1.19 | 0.56 |
| **host µs/step** | **84.9** | **33.1** |
| **`tail_async` µs/step** | ~2350 | ~3575 |

### Head-module internals (per round, deep steps)

| counter | mean µs |
|---|---:|
| `mod_embed` | 36.47 |
| `mod_fuse` | 21.08 |
| `mod_mask` | 1.05 |
| `mod_layer` | 261.81 |
| ↳ `lay_attn` | 210.69 |
| ↳ `lay_mlp` | 36.34 |
| ↳ `lay_norm` | 12.65 |
| `mod_norm` | 1.92 |

Nothing here is a hoistable invariant of meaningful size: the largest single item
(`lay_attn` graph build, 210 µs/round across all deep steps) is 0.12% of a round.

## 4. Round composition

| phase | mean µs | share |
|---|---:|---:|
| `draft_build_us` | 17486.2 | 9.94% |
| `verify_build_us` | 85150.8 | 48.42% |
| `eval_wall_us` (the one blocking eval) | 72705.0 | 41.35% |
| `readout_us` | 27.0 | 0.02% |
| `commit_us` | 319.7 | 0.18% |
| `upkeep_us` | 156.1 | 0.09% |
| **`round_us`** | **175847.0** | **100%** |

`verify_build_us` is *also* not pure host build.
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1858-1888` contains an
**asyncEval ladder**: `ladderActive = inputs.dim(1) <= 9 || inputs.dim(1) >= 512`,
firing `asyncEval(hiddenStates)` at layers 0, 1, 9, 19, 29, 39, 49, 57. Verify width is
`d+1 ≤ 9`, so the ladder is **always active** on the verify pass. `verify_build_us` is
therefore graph build **plus 8 encode/submit/throttle points** over the 64-layer target
tape — which is exactly why it scales with `d` (64.2 ms at d=4 → 99.0 ms at d=8).

## 5. Bandwidth accounting — the real floor

Affine 4-bit group-64 costs `(64·4 + 32)/64/8 = 0.5625 B/weight` (4-bit payload plus a
bf16 scale and bias per group of 64). Every byte figure below uses that constant.

### 5.1 The byte model is verified against the declared head, not assumed

| head tensor | shape | weights |
|---|---|---:|
| `self_attn.q_proj` (output-gated) | 12288 × 5120 | 62,914,560 |
| `self_attn.k_proj` | 1024 × 5120 | 5,242,880 |
| `self_attn.v_proj` | 1024 × 5120 | 5,242,880 |
| `self_attn.o_proj` | 5120 × 6144 | 31,457,280 |
| `mlp` gate+up+down (fused, 17408) | — | 267,386,880 |
| `fc` | 5120 × 10240 | 52,428,800 |
| **total** | | **424,673,280** |

`424,673,280 × 0.5625 = 238,878,720 B`. The declared head file is **238,934,093 B**.
The 55,373 B residual is exactly 4 RMSNorm vectors (4 × 5120 × 2 = 40,960 B) plus a
14,413 B safetensors header. **The byte model reconciles to 0.02%**, so the floors below
are anchored to a measured artifact rather than to a parameter-count estimate.

### 5.2 Per deep head step

| tensor | bytes |
|---|---:|
| head decoder layer + fc + norms (4-bit g64) | 238,934,093 (= declared head file size) |
| compact draft lm_head slice, 98,336 × 5120 × 0.5625 B | 283,207,680 |
| **total per head step** | **522,086 KB ≈ 522 MB** |

### 5.3 Correction: the target streams 14.41 GB per verify, not 14.1 GiB

`weights/config.json` has `tie_word_embeddings: false`, and `Qwen35.swift:2079-2080`
builds a separate `lm_head` whenever untied. So `embed_tokens` (248320 × 5120 =
715.2 MB) is **gathered** — M rows, ≈26 KB at M=9 — and is never streamed during decode.

| group | bytes |
|---|---:|
| 48 × linear-attention layer @ 215.56 MB | 10.347 GB |
| 16 × full-attention layer @ 209.39 MB | 3.350 GB |
| `lm_head` (248320 × 5120) | 0.715 GB |
| **streamed per verify pass** | **14.412 GB = 13.423 GiB** |
| `embed_tokens` — gathered, *not* streamed | (0.715 GB excluded) |

Parameter count including the embedding table is 26.893 B, which is exactly the
`26.9e9` the advisor's FLOP term uses — so the **compute** side of the advisor roofline
is right. Only the **byte** side double-counts the untied embedding.

Consequence: `BW_eff` derived from the pinned serial leg is
`14.412 GB / 0.0673 s = 214.2 GB/s`, not 227 GB/s. **Every bandwidth floor line is
therefore 6.0% larger than the advisor's estimate.**

The headroom in this loop is kernel/bandwidth efficiency, **not** host code.

## 5A. The three numbers requested in feedback (5)

`research/round_floor.py` (new, research-only, **not** in `editablePaths`) is the decode
analogue of `research/prefill_floor.py`. It does not *assume* `BW_eff` or `FLOPS_eff`:
it measures them on this host with the same MLX 0.32.0 Metal backend the worker uses,
then rebuilds one d=8 MTP round out of 16 individually timed kernel groups at exactly
the scored shapes (K=5120, N=17408/5120/248320, affine 4-bit g64, kv-len 1120,
verify width M=9, head-step width 1, compact draft vocab 98,336).

**Which machine the mechanism attacks.** Everything below is *decode-side*: the 512
counted tokens, at verify width M ≤ 9 and head-step width 1. It is a different machine
from the seed prefill that E3 measured — prefill runs at M=512 where the same hardware
is compute-bound and ~98% efficient. The assigned hypothesis (host-side graph build)
and the finding that replaces it (verify-width GEMM efficiency) both live on the
decode machine.

Measured machine constants (this M4 Pro, amortized, 16 kernels per submit):

```text
BW_eff    = 250.0 GB/s      (peak, at M=1 quantized matvec: 248.7 GB/s = 99.5%)
FLOPS_eff = 6.543 TFLOP/s   (quantized GEMM, large-M)
balance   = 26.2 FLOP/byte  =>  crossover width M* ~= 8.4 for 4-bit g64
```

The three numbers, d=8:

| number | value | meaning |
|---|---:|---|
| `roofline_floor` | **90.08 ms** | sum over components of `max(bytes/BW_eff, flops/FLOPS_eff)` |
| `achieved_kernel_floor` | **226.02 ms** | same components, **measured** back-to-back on GPU |
| `measured_block_seconds` | **217.75 ms** | mean traced `round_us` at d=8 (N=9 rounds) |

| gap | value | reading |
|---|---:|---|
| kernel-efficiency gap (`achieved − roofline`) | **+135.93 ms** | 60.4% of the round is kernels running below roofline |
| scheduling gap (`measured − achieved`) | **−8.27 ms (−3.8%)** | **essentially zero** — nothing left for host or scheduler |

The sign of the scheduling gap is the whole answer to this assignment. The sum of
independently measured kernels slightly *exceeds* the real round, i.e. the live round is
already at least as well overlapped as a hand-issued back-to-back kernel stream. There
is no 67–77 ms of unexplained scheduling overhead on this host: once `BW_eff` and
`FLOPS_eff` are measured rather than assumed, the gap the advisor computed **is the
kernel-efficiency gap**, not a scheduling gap, and host graph construction (0.599 ms,
§3) cannot be more than 0.35% of it.

Component floor (d=8; `iso_ms` = one-submit-per-call diagnostic, not part of the floor):

| component | calls | roofline ms | achieved ms | eff | bound | iso ms |
|---|---:|---:|---:|---:|---|---:|
| verify:mlp:gate_up_down | 64 | 47.081 | 133.514 | 35.3% | compute | 147.088 |
| verify:lin_attn:in_proj_fused_qkvzba | 48 | 11.143 | 31.686 | 35.2% | compute | 42.634 |
| verify:lin_attn:out_proj | 48 | 4.154 | 12.463 | 33.3% | compute | 18.152 |
| **verify:lm_head_full_vocab** | **1** | **3.498** | **9.946** | 35.2% | compute | 9.997 |
| verify:full_attn:qkv_proj | 16 | 3.231 | 9.290 | 34.8% | compute | 12.775 |
| head:draft_lm_head_compact | 8 | 9.062 | 9.025 | 100.4% | memory | 10.470 |
| head:mlp_gate_up_down | 8 | 4.812 | 4.939 | 97.4% | memory | 6.175 |
| verify:full_attn:o_proj | 16 | 1.385 | 4.144 | 33.4% | compute | 6.039 |
| verify:lin_attn:gated_delta_kernel | 48 | 1.208 | 3.198 | 37.8% | memory | 9.304 |
| verify:full_attn:sdpa | 16 | 0.329 | 2.590 | 12.7% | compute | 4.852 |
| head:attn_qkv_proj | 8 | 1.321 | 1.471 | 89.8% | memory | 2.443 |
| **state:recurrent_snapshot_48_layers** | **1** | **1.208** | **1.315** | 91.8% | memory | 1.457 |
| head:fc_concat_proj | 8 | 0.944 | 0.944 | 100.0% | memory | 1.719 |
| head:attn_o_proj | 8 | 0.566 | 0.674 | 84.0% | memory | 1.534 |
| verify:lin_attn:conv1d_depthwise_k4 | 48 | 0.063 | 0.606 | 10.4% | memory | 6.002 |
| head:attn_sdpa | 8 | 0.080 | 0.213 | 37.4% | memory | 1.298 |
| **total** | | **90.08** | **226.02** | | | 281.94 |

The two lines feedback (5) asked not to omit, called out explicitly:

- **`lm_head`, K=5120 N=248320.** 715.2 MB of 4-bit weights, read **once per round** at
  width d+1 = 9 (`Qwen35.swift:2209`, reached from `Qwen36MTPBlockSession.swift:915`).
  Roofline 3.498 ms, achieved **9.946 ms** — 4.6% of the round, 35.2% efficient. Note
  this is *only* `lm_head`: `tie_word_embeddings` is `false` and `Qwen35.swift:2079-2080`
  builds a separate untied `lm_head`, so `embed_tokens` (another 715.2 MB) is
  **gathered**, ~26 KB at M=9, and never streamed. That is the byte-model correction of
  §5.3.
- **Recurrent snapshot.** `snapshotRecurrent` (`Qwen36MTPBlockSession.swift:1126-1134`)
  copies 48 × 3.0 MiB = **144.0 MiB**; read+write = 302.0 MB. Roofline 1.208 ms,
  achieved **1.315 ms**, 91.8% efficient, **0.60% of the round**. The advisor's 0.6 ms
  estimate counted the write only; even doubled it is not a lever. The
  `qwen35GatedDeltaMidKernel` `S==2` write (`Qwen35.swift:411-497`) of a comparable
  144 MiB is inside the `gated_delta_kernel` line (3.198 ms achieved, 1.4% of round).

Reproduce:

```bash
scratch/mlxenv/bin/python research/round_floor.py \
  --depth 8 --kv-len 1120 --measured-block-seconds 0.21775 --sweep \
  --json scratch/results/round_floor_d8.json
```

## 5B. Headline finding: `quantized_matmul` collapses at exactly the verify widths MTP uses

The 135.93 ms kernel-efficiency gap is not spread evenly. Every `compute`-bound line in
the table above is a `quantized_matmul` at M=9 and every one of them lands at **33–35%**
efficiency, while every head line — the same kernel at M=1 — is at **84–100%**. So I
swept width directly at the MLP gate shape (M × 5120 → 17408, affine 4-bit g64):

| M | ms/call | GB/s | TFLOP/s | eff | rows/s vs M=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2016 | 248.7 | 0.884 | **99.5%** | 1.00 |
| 2 | 0.2147 | 233.5 | 1.660 | 93.4% | 1.88 |
| 3 | 0.2306 | 217.4 | 2.319 | 87.0% | 2.62 |
| 4 | 0.2883 | 173.9 | 2.473 | 69.6% | 2.80 |
| 5 | 0.3568 | 140.5 | 2.498 | 56.2% | 2.82 |
| 6 | 0.4542 | 110.4 | 2.355 | 44.2% | 2.66 |
| 7 | 0.5582 | 89.8 | 2.236 | 35.9% | 2.53 |
| 8 | 0.5622 | 89.2 | 2.536 | 38.8% | 2.87 |
| **9** | **0.6958** | **72.1** | **2.306** | **35.2%** | **2.61** |
| 12 | 0.8865 | 56.6 | 2.413 | 36.9% | 2.73 |
| 16 | 0.8899 | 56.3 | 3.205 | 49.0% | 3.62 |
| 24 | 0.8938 | 56.1 | 4.786 | 73.2% | 5.41 |
| 32 | 0.8917 | 56.2 | 6.397 | **97.8%** | 7.23 |
| 64 | 1.7589 | 28.5 | 6.486 | 99.1% | 7.33 |
| 128 | 3.5051 | 14.3 | 6.510 | 99.5% | 7.36 |
| 512 | 13.9552 | 3.6 | 6.540 | 100.0% | 7.40 |

Three facts:

1. At M=1 the kernel is memory-bound and **99.5% of peak**. Serial decode is already
   optimal. There is nothing to win there.
2. From **M≈3 to M≈12 throughput is flat at ~2.6–2.9 rows per unit time** — time grows
   ∝ M, so batching buys essentially nothing. Verifying 9 rows costs ~3× verifying 3.
   Marginal cost of one extra verify row is **0.0618 ms** = 31% of an entire weight
   pass, where the ideal marginal is `2·K·N / FLOPS_eff = 0.0273 ms`. **2.26× too
   expensive at the margin, per projection, per layer.**
3. Efficiency snaps back to ~98% only at **M ≥ 32**, and absolute time is literally flat
   from M=12 to M=32 (0.8865 / 0.8899 / 0.8938 / 0.8917 ms) — 2.7× more rows for free,
   unreachable because the parent caps drafts at 8, i.e. M ≤ 9.

**The MTP round lives entirely inside the worst part of this curve.** That is the real
reason MTP at d=8 costs 217.75 ms while serial d=0 costs 86.80 ms per token on this
host: 8 extra rows cost 2.5× a whole serial token instead of the ~0.9× the machine's
own compute roofline allows.

Size of the prize: verify-side quantized matmuls are 133.514 + 31.686 + 12.463 + 9.946 +
9.290 + 4.144 = **201.14 ms** of the 217.75 ms round at 35% efficiency. At roofline they
are 70.49 ms, so the round would fall to **~87.1 ms — a 2.50× round speedup**, and the
same fix raises every draft depth's payoff, moving the cost-model crossover well past
M=9.

**This is inside the submission surface.** Verified against `BASE_SHA:benchmark.json`
with `senpai/validate-assignment-scope.sh`:

- ✅ submittable: `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.metal`,
  `.../kernels/quantized_nax.metal`, `.../kernels/quantized.h`, `.../kernels/quantized_nax.h`,
  `.../kernels/quantized_utils.h`, and the runtime-effective generated twins
  `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`, `quantized_nax.cpp`,
  `quantized_utils.cpp` (JIT source strings — must be edited together, then
  `python3 research/twin_audit.py`). `steel/gemm`, `gemv.*` and the `steel_gemm_*`
  twins are also submittable.
- ❌ **not** submittable: `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp`
  — the host dispatcher. `get_qmv_batch_limit(D, O, d)` is at line 84 and is consulted at
  lines 1415 and 1483 (`vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4`).
  For D=5120, O=17408 (both > 4096) it returns 6 on `arch_gen ∈ {13,14}` and 10
  otherwise, which is what puts M=9 on the `qmv` side of the qmv/qmm split and produces
  the flat-throughput regime above. **The selection cannot be changed by a submission;
  only the kernel bodies can.** Any experiment here must therefore make `qmv_*` itself
  amortize the weight stream across up to 9 rows (`qmv_quad` :177, `qmv` :235;
  `qmm_nax` :473, `qmm` ~:699-771, `qmm_t_splitk` :798-846 "tile K by BK=32").

This is the single highest-value follow-up I can name from this experiment, and it is a
different mechanism from anything currently in the ledger.


## 6. Advisor feedback answers

### (a) Shape-varying rebuild cost ≈ 0 — hypothesis refuted

Paired within-run discriminator. For each realized depth `d`, compare rounds whose
previous round had the **same** width against rounds where the width **changed**:

| d | same-width N | draft µs | verify µs | width-change N | draft µs | verify µs | Δ draft | Δ verify |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7 | 6624 | 64188 | 2 | 6674 | 64464 | +50 | +276 |
| 5 | 1 | 12283 | 71963 | 1 | 12512 | 72227 | +229 | +264 |
| 6 | 2 | 17175 | 80001 | 1 | 17178 | 80025 | +3 | +24 |
| 7 | 3 | 21868 | 88044 | 1 | 21783 | 88083 | −85 | +39 |
| 8 | 7 | 25978 | 98973 | 2 | 25899 | 98955 | −79 | −18 |

Deltas straddle zero and the largest magnitude is **+276 µs = 0.19% of a round**.
This **refutes the `ml-explore/mlx-lm` #250 non-static-shape hypothesis for this
workload**: `warmAllDepths` already pre-warms every legal width, so a width change
costs nothing measurable. High-value negative — it removes a candidate experiment.

### (b) Per-round graph construction at constant width

Answered directly by the Part A HOST-ONLY row: **0.599 ms/round = 0.35%**. The paired
(a) table doubles as the constant-width control the advisor asked for: same-width rounds
at each `d` have essentially identical draft/verify costs to width-changing rounds, so
the free-depth schedule is a valid substitute for a `FIXED_DEPTH` run.

### (c) Genuine per-draft work scales with d

| d | N | round µs | draft µs | verify µs | eval µs | tail_async µs | host-only µs | acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10 | 140077 | 9226 | 75707 | 54855 | 8816 | 212.8 | 3.70 |
| 5 | 2 | 147320 | 12398 | 72095 | 62138 | 10959 | 826.6 | 5.00 |
| 6 | 3 | 169071 | 17176 | 80009 | 71087 | 15342 | 895.2 | 6.00 |
| 7 | 4 | 190336 | 21846 | 88054 | 79587 | 19839 | 964.9 | 7.00 |
| 8 | 9 | 217750 | 25961 | 98969 | 92367 | 24662 | 673.2 | 7.89 |

Fit on d=4..8: `draft_build ≈ −3326 + 4184·(d−1)` µs.
Round marginal per extra draft = **19418 µs**.

`host_only` is flat in `d` (213–965 µs, non-monotonic, dominated by sampling noise)
while `tail_async` grows 8.8 → 24.7 ms. **All per-draft scaling is GPU work.**

Round-level impact table (linear extrapolation from the d=4/d=8 anchors):

| d | pred draft µs | pred round µs | draft share |
|---:|---:|---:|---:|
| 1 | −3326 | 81823 | −4.1% |
| 2 | 858 | 101241 | 0.8% |
| 4 | 9226 | 140077 | 6.6% |
| 8 | 25961 | 217750 | 11.9% |

(The d=1 extrapolation goes negative — the linear form is only valid over the measured
d=4..8 range; d=1 is reported for the advisor's requested grid, not as a prediction.)

### Compiled/traced round — how much would it actually remove?

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/CompiledDecode.swift:38` gates compiled
decode to "B=1 solo decode with supported cache types" — it excludes MTP and SSM, so it
never fires on this path. Even if it were extended: a compiled round can only remove
**host graph construction**, i.e. **≤0.599 ms/round (0.35%)** in the draft loop. It
cannot remove `s1_async`/`tail_async` (queue drain), it cannot remove the blocking
`eval`, and it cannot remove the verify ladder's encode/submit points. **A compiled
round is not worth building for this workload.**

### Stall guardrail — corrected, after-first (feedback 3.1)

Per `Sources/MLXFastCLI/main.swift:2011-2040`, `check_stall_guardrail` reads
`max_block_request_seconds_after_first` / `p50_block_request_seconds_after_first`;
whole-window fields are audit-only.

| quantity | value |
|---|---:|
| N after first | 27 |
| max after first | 218655 µs (**round 16**) |
| p50 after first (lower median) | 169111 µs |
| **ratio** | **1.293** (threshold 4.0) |
| first block (excluded) | 266205 µs |
| whole-window ratio (audit only) | 1.574 |

Max-round detail: round 16, `d=8`, `acc=7`, **`shape_change=0`**, `repair=none`,
draft 26906 µs, verify 98994 µs, eval 91998 µs.

**The max block is not caused by a width change, a rebuild, or a repair.** It is an
ordinary `d=8` round. Spread is driven purely by schedule depth
(d=4 → 140 ms, d=8 → 218 ms). Repairs: **none** in the window. Width changes: 7 of 28.
The guardrail has 3.1× margin.

### Head forwards per round (feedback 3.4)

**`d` per round** when all drafts are accepted: one flush+step-1 forward plus `d−1` deep
steps. The accepted-token cache commit is **already fused** into the next round's step-1
flush via `headHistoryBacklogHidden` / `headHistoryBacklogTokens`. There is no separate
commit-only head forward to remove, so **`ml-explore/mlx-lm` PR #990's `cache_commit`
optimization is already implemented here**.

### `headStepCostRatio` is under-fit on this host

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:556-568` fits
`headStepCostRatio = 0.20` from "~10.75 ms marginal per draft on a ~27 ms base".
Measured here: round marginal per extra draft = **19.42 ms**; serial base =
**86.80 ms/token** ⇒ implied ratio **≈ 0.224**. Out of scope to change under this
assignment (explicitly forbidden) — reported as a follow-up.

## 6b. Answers to feedback (6) — `qwen38-r1-e4-fb6-bandwidth-vs-fixed-preregistered`

Five items, answered in order. Two of the advisor's corrections are **accepted**, one was
answered wrongly in the first revision of this report and is **retracted below**, and the
pre-registered discriminator gets an explicit branch answer.

### (6.1) "The MTP head streams 849.4 MB/forward (bf16), not 239 MB" — **ADVISOR CORRECT; MY FIRST ANSWER RETRACTED**

An earlier revision of this report claimed the runtime loads the declared 4-bit head. That
claim is **wrong for every local run in this experiment**, and the advisor's follow-up
(feedback 7) was right to force the check. The corrected position is that *both* readings
are true, but of different legs:

```text
LOCAL  (./benchmark-qwen-mtp.sh, both serial and MTP legs)  ->  bf16 organizer head   849.4 MB/step
RANKED (candidate leg only)                                 ->  declared 4-bit head   238.9 MB/step
RANKED (pinned serial leg)                                  ->  never uses a candidate head
```

**Why local is always bf16 — the harness never reads `mtp-head.manifest.json`.**

```text
setup-qwen-mtp.sh:66-67   MTP_HEAD_MODEL_ID="${MLXFAST_QWEN_MTP_HEAD_REPO:-EigenLabs/Qwen3.8-27B-MTP-bf16}"
                          revision 26a328e070875b0314d652a039b6b59902690f03
                          verified against fixtures/qwen3_8_27b_mtp_head.sha256
benchmark-qwen-mtp.sh:214-216   requires MLXFAST_QWEN_MTP_HEAD_DIR
benchmark-qwen-mtp.sh:554,603,613   passes --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}"
```

Neither script contains the string `mtp-head.manifest.json`. The declaration is consumed
by the trusted worker on the ranked candidate leg only.

**`head_provenance` — the mandatory record for every head-sensitive number in this report.**

```text
directory        ~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head
file             model.safetensors
sha256           8fceddc664f3ea96d02e304463aa1319213ff52cdf1f3401d4bce64e7075c349
bytes            849,400,347                (== fixture weights_file_bytes, exact match)
dtype            bfloat16                   (config.json "dtype": "bfloat16")
tensors          15 bare tensors, NO ".scales"/".biases" keys -> stays dense after load
source repo      EigenLabs/Qwen3.8-27B-MTP-bf16 @ 26a328e070875b0314d652a039b6b59902690f03
index total_size 849,398,784
```

The index `total_size` reproduces exactly from the declared shapes, which also pins the
trunk geometry used everywhere below:

```text
q 12288x5120 + k 1024x5120 + v 1024x5120 + o 5120x6144
+ gate 17408x5120 + up 17408x5120 + down 5120x17408
+ fc 5120x10240 + 26,112 norm params
= 424,699,392 params x 2 B = 849,398,784 B   == index total_size
```

The fixture note says the same thing in words: *"The 3.8 head is bf16, so the same 8
matrices are 8 entries and 8 + 7 = 15."*

**What the repository declares for the ranked leg** (unchanged, and still correct as a
*ranked-leg* statement): `mtp-head/` in this tree contains only `README.md`; the weights
are fetched from the declaration.

```text
mtp-head.manifest.json  (verbatim, BASE_SHA e20268e9)
  source_url  hf:lowskillcoding/qwen38-mtp-head-4bit-g64
              @0966ddaff972fd3ca2be08f3640603b47e9ce70a
  sha256      cc209e30d8a7def1fc4d785be22b0ec40e16ae6763f9591255a1996a34f08f0d
  bytes       238934093
  max_bytes   2147483648
```

which reconciles to the byte:

```text
424,673,280 weight elements x 0.5625 B/elt (4-bit + g64 scale/bias) = 238,878,720 B
4 RMSNorm vectors                                                   =      40,960 B
safetensors header                                                  =      14,413 B
                                                              total =  238,934,093 B  == manifest `bytes`
```

**The "impossibility" argument I used before is withdrawn as invalid.** It compared
`draft_build_us` (25.96 ms at d=8) against a bf16 GPU roofline (27.2 ms) and concluded the
chain "would have to finish before its own weights arrived". That inference is unsound,
because §2/§3 of this same report establish that `draft_build_us` is *host-thread* time
that ends when the async-eval throttle releases — the tail of the head chain's GPU work
spills into `eval_wall_us`. The two quantities are not comparable. The measured bf16 chain
(27.41 ms, §6c) against `tail_async` = 24.66 ms at d=8 is in fact perfectly consistent:
~90% of the head's GPU time is absorbed inside the draft phase and the rest spills.

The dispatch-branch reading survives, but now points the other way. `Qwen35.swift:1193`
(`fusedGateUp()`) tries `_fqW/_fqS/_fqZ -> quantizedMM(...)` first with
`if let w = _fbfW { matmul(x, w.T) }` as fallback; `qkv()` at `:1513` is the same shape and
the doc comment at `:1510-1512` reads verbatim **"Unquantized (MTP bf16) falls back."**
With a dense bf16 head loaded, **the fallback is the live path locally** — which is exactly
what that comment was written for.

Finally, the cited path `Sources/MLXFastModel/Qwen35MTP.swift:37-38` **does not exist** in
this tree. The real file is
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.

> **Consequence for item (6.5).** The proposed follow-up "quantize the head to 4-bit, save
> 18-26 ms (11-16%)" is **already banked for the ranked leg** by the existing
> `mtp-head.manifest.json`, so there is no ranked prize left in *doing* the quantization.
> What is left, and what §6c measures, is whether the 4-bit head is on the right side of
> the bandwidth/efficiency tradeoff at all. It is: at M=1 the 4-bit path sustains 249.0
> GB/s against bf16's 253.1 GB/s (98.4%), so bytes convert to time at the same rate and
> only *fewer bytes* help.

### (6.2) Launch overhead is ruled out — **CONFIRMED**, and strengthened

Agreed, and this experiment's measurement is stronger than the census. The measured
**scheduling gap is −8.27 ms** (−6.96 ms after the correction in 6.3): the real round is
*faster* than the sum of its kernels timed back-to-back in isolation. Dispatch and
command-buffer overhead is therefore not merely bounded — it is already fully hidden
behind GPU execution. The census figures (~1,540 dispatches/round, ~1,285 of them in
verify at S=9; ~17 MB/round copy traffic → 0.06 ms) are consistent with that and change
nothing.

> **Erratum.** An earlier revision assumed arch suffix `g` and therefore
> `max_ops_per_buffer_ = 40`. The device actually reports `applegpu_g16s`
> (`mx.device_info()`, §6c), i.e. suffix **`s`** → `max_ops_per_buffer_ = 50`,
> `max_mb_per_buffer_ = 50` (`mlx/backend/metal/device.cpp:574-586`). That lowers the
> commit count to ~31/round and makes the "overhead is hidden" conclusion *stronger*, not
> weaker. The same probe is what exposes the qmv/qmm dispatch finding in §6c.4.

### (6.3) Snapshot retraction — **ACCEPTED, verified, floor corrected**

The advisor is right. `Qwen36MTPBlockSession.swift:1232-1239` builds **96 lazy
`[.ellipsis]` slice expressions** (48 GDN layers x 2 arrays), and its own contract says
so: *"No GPU work happens here; this is MTPLX's `_lazy_state_view`."*

The decisive check is the **consumption** site, not the construction site. `snapshot` is
read at **exactly one place**, line `1089`, inside the `else` branch that also sets
`didRepair = true` — the rare defensive re-forward. On the happy path the slices are
dropped without ever entering an `eval`/`asyncEval` output set, so MLX never schedules
them. And in the 192-token window **`repair=1` occurred in 0 of 28 rounds**, so the
snapshot was never materialized even once.

My floor line modelled the 302 MB read+write, i.e. the **rollback-path** cost. Removing
it:

| quantity | as published (§5A) | corrected | delta |
|---|---:|---:|---:|
| roofline floor | 90.08 ms | **88.87 ms** | −1.208 |
| achieved-kernel floor | 226.02 ms | **224.71 ms** | −1.315 |
| measured block | 217.75 ms | 217.75 ms | 0 |
| achieved / roofline | 2.51x | **2.53x** | — |
| kernel-efficiency gap | +135.93 ms (60.4%) | **+135.84 ms (62.4%)** | −0.09 |
| scheduling gap | −8.27 ms (−3.8%) | **−6.96 ms (−3.2%)** | +1.31 |

**No conclusion changes** (−0.6% on the floor). The §5A/§5B tables are left as-measured
and this table is the erratum; the `state:recurrent_snapshot_48_layers` row should be
read as "cost *if* a repair fires", not as a per-round charge.

Caveat on provenance: `snap_ns` was added in commit `e604ee3`, *after* the 192-token run
(`c39e178`), so there is **no direct timing** of this segment. The verdict rests on the
source contract plus the observed `repair=0`, not on a measurement.

### (6.4) The pre-registered discriminator — **my answer is BANDWIDTH**

This confirms the advisor's prediction and the ranked anchor. Reasoning:

The residual is **not** host-fixed cost — the measured scheduling gap is ≈ 0 (indeed
negative), which directly falsifies the FIXED branch's mechanism. And the FIXED branch's
own 12.40 ms/token floor **cannot reproduce the observed 9.957 ms/token**, so it is
arithmetically excluded before any of my data is considered. BANDWIDTH's implied score
3.039 with a 0.50 ms residual is the consistent branch.

**But the dichotomy is incomplete, and that matters more than the branch label.** The
residual is neither fixed host time nor "extra" traffic from some unmodelled tensor. It
is **redundant re-streaming of the *same* weights inside `quantized_matmul` at M = 3..12**:
the kernel achieves 248.7 GB/s at M=1 but only **72.1 GB/s at M=9**, i.e. it moves
**≈3.45x the necessary weight bytes** (§5B). That is memory traffic, so it does scale
with bandwidth and lands in the BANDWIDTH branch — but its cause is a *kernel tiling
defect*, not a bandwidth budget. A third category, **small-M kernel inefficiency**,
belongs in the discriminator. It is also the only one of the three that is fixable, and
§5B sizes the prize at a **2.50x round speedup** if verify-side quantized matmuls were
brought to their roofline (201.14 ms → 70.49 ms, round 217.75 → ~87.1 ms).

### (6.5) The three sub-hypotheses

**"The head chain runs worse than 70% efficiency; 15-25 ms of slack; time it in isolation
against its 47.5 ms roofline." — ALREADY MEASURED, REFUTED.** The five `head:*` rows sum
to **17.27 ms achieved against a 16.79 ms 4-bit roofline**, i.e. **84%-100.4% efficiency**
per row (`draft_lm_head_compact` 100.4%, `mlp_gate_up_down` 97.4%, `fc_concat_proj`
100.0%, `attn_qkv_proj` 89.8%, `attn_o_proj` 84.0%). The head chain is the *healthiest*
part of the round; there is no slack to recover.

> **Revised in light of 6.1/§6c.** Those five rows model the *ranked* 4-bit head. The
> local run streams the bf16 head, so the correct local numbers are **27.41 ms achieved
> against a 27.00 ms roofline = 101.5% chain efficiency**, measured directly in §6c.2.
> The conclusion is unchanged and in fact sharper: at **either** storage format the head
> chain runs at its bandwidth roofline, so the advisor's "15-25 ms of slack" does not
> exist in the head. The 47.5 ms roofline quoted in the hypothesis is close to the true
> bf16 chain figure once the compact draft `lm_head` (9.03 ms) is added: 27.41 + 9.03 =
> **36.4 ms**, all of it real and none of it recoverable by better kernels.

**"Report `rollbackRoundCount`." — ANSWERED: zero.** `repair=1` in **0 of 28 rounds**
(`acc` never fell below the eager-checkpoint path). The rollback/repair term contributes
**0 ms** to this window, which is also why 6.3's correction is safe to apply.

**"48 dependent tiny GDN kernels, 1.5-4 ms." — CONSISTENT, and already inside the floor.**
`verify:lin_attn:gated_delta_kernel` = **3.198 ms achieved** (48 calls, 37.8% efficiency,
memory-bound), squarely in the predicted band and already counted in the 224.71 ms
achieved floor. It is not a hidden term.

## 6c. Answers to feedback (7) — `qwen38-r1-e4-fb7-head-mismatch-floor`

The advisor asked for four things: the head provenance record, a decisive measurement of
which branch the head sits on, whether an 8-bit head would beat 4-bit, and a corrected
floor. All four are answered from one 13.9 s measurement plus a 10.5 s second component
pass. **The advisor was right about the mismatch**; §6b.1 now carries the retraction and
the provenance record. This section carries the measurements.

Reproduce:

```bash
<mlxenv>/bin/python research/round_floor.py --depth 8 --kv-len 1120 \
    --measured-block-seconds 0.21775 --head-dtype-compare \
    --out scratch/results/head_dtype_d8.json          # 13.9 s, job 5851eca0
<mlxenv>/bin/python research/round_floor.py --depth 7 --kv-len 1120 \
    --measured-block-seconds 0.190336 \
    --out scratch/results/round_floor_d7.json         # 10.5 s, job c33fea48
```

### (6c.1) The head is **BANDWIDTH-BOUND** — the advisor's branch #1 fires

`research/round_floor.py --head-dtype-compare` rebuilds the eight trunk matrices at both
storage formats and times the same five rows. It excludes `head:draft_lm_head_compact`,
which is the target-owned compact draft slice: it is 4-bit either way and independent of
the head's storage format.

Machine probes for this pass:

```text
BW_eff(4-bit qmv, lm_head M=1)      = 249.0 GB/s
BW_eff(bf16 dense matvec, 65536x5120) = 253.1 GB/s
FLOPS_eff = 6.541 TFLOP/s   machine balance = 26.3 FLOP/byte
```

| head row | q4 ms | q4 eff | bf16 ms | bf16 eff | x (bf16/q4) |
|---|---:|---:|---:|---:|---:|
| `head:fc_concat_proj` | 0.1146 | 103.4% | 0.4181 | 99.1% | 3.65 |
| `head:attn_qkv_proj` | 0.1688 | 98.3% | 0.5844 | 99.3% | 3.46 |
| `head:attn_sdpa` | 0.0434 | 42.5% | 0.0405 | 44.7% | 0.93 |
| `head:attn_o_proj` | 0.0644 | 110.3% | 0.2478 | 100.3% | 3.85 |
| `head:mlp_gate_up_down` | 0.6139 | 98.4% | 2.1350 | 99.0% | 3.48 |

```text
trunk bytes/step   q4  238.88 MB    bf16  849.35 MB    ratio 3.56x
chain x8 achieved  q4    8.040 ms   bf16   27.406 ms   ratio 3.41x
chain x8 roofline  q4    7.823 ms   bf16   26.996 ms
```

**Verdict: bandwidth-bound, unambiguously.** The measured time ratio 3.41x tracks the byte
ratio 3.56x, and every trunk row sits at **99-103% of its own same-dtype roofline**. The
only row that does not scale is the sdpa (0.93x), which is correct — it touches activations
and the head KV cache, not trunk weights, and it is 0.15% of the chain.

Two consequences:

1. The roofline model **applies** to the head. There is no efficiency defect to recover
   here at either dtype; the residual the advisor is chasing is not in the head.
2. Because the chain is at its roofline, head time is a pure function of head bytes. The
   only lever is **fewer bytes**, not better kernels. That is what §10 follow-up 2 says
   and it is now measured rather than argued.

### (6c.2) Would an 8-bit head be better than 4-bit? — **NO, strictly worse**

The advisor's hypothesis was that 4-bit nibble unpacking might cost enough to make an
8-bit head faster despite doubling the bytes. The measurement above settles it directly:
at decode width the 4-bit quantized matvec sustains **249.0 GB/s** against dense bf16's
**253.1 GB/s** — **98.4%**. There is no meaningful per-byte penalty for 4-bit on this host
at M=1, so an 8-bit head would pay ~2x the bytes at ~the same bytes/second and land at
~2x the time. The `qmv` path's dequantization is fully hidden behind the memory stream.

The Fermi-estimate claim that "4-bit is ~12% slower than 8-bit on Apple Silicon" does not
transfer to M=1 decode on `applegpu_g16s`. **Keep the declared 4-bit head; the remaining
prize is a smaller head, not a differently-quantized one.**

I did **not** stage a second head tree, per the advisor's instruction to skip it if
expensive. Timing the isolated chain at both dtypes answered the same question in 13.9 s
and without touching `mtp-head.manifest.json`.

### (6c.3) Corrected floor — and an honest new discrepancy

Substituting the true bf16 trunk for the 4-bit trunk rows in the d=8 component pass:

```text
                                d=8 (width 9)      d=7 (width 8)
achieved floor as measured        226.02 ms          187.12 ms
  minus 4-bit head trunk rows      -8.24 ms           -7.38 ms
  plus  bf16 head trunk rows      +27.41 ms          +23.98 ms   (27.406 x 7/8)
  minus snapshot erratum (6.3)     -1.32 ms           -1.32 ms
= corrected achieved floor        243.87 ms          202.40 ms
measured_block                    217.75 ms          190.34 ms
scheduling gap                    -26.12 ms          -12.06 ms
                                    (-12.0%)            (-6.3%)
```

**The corrected floor now exceeds the measurement**, by 12% at width 9 and 6% at width 8.
A floor above the measurement is a defect in the model, and I am reporting it rather than
tuning it away. Three things are established about it:

- **It is not the head.** The head chain was measured directly at both dtypes in 6c.1 and
  matches its roofline to 1.5%. The bf16 chain (27.41 ms) is also consistent with the
  traced `tail_async` at d=8 (24.66 ms) plus spill into `eval_wall`.
- **It is not "whole-forward segmentation", which is what I guessed first.** That guess is
  **refuted by the source**. `Qwen36MTPBlockSession.swift:592-599` states that
  `attentionWithCacheUpdate` splits only the **sdpa** into two <= 5-row calls, and that
  segmenting the whole forward (two model calls, 5+k) *"was measured bit-exact too but
  pays a second full weight pass (~25 ms) and loses on net; the chunk lives at the sdpa
  only."* The quantized projections genuinely run at M=9 in one launch, exactly as my
  component pass models them.
- **It scales with verify width** (-6.3% at width 8, -12.0% at width 9), which points at
  the wide quantized-matmul rows rather than at any fixed term.

The most likely remaining cause is a property of the harness, not of the round:
`timeit_amortized` issues 16 **independent copies of one op** inside a single submit, which
maximizes contention on exactly the unit that op saturates. The real forward interleaves
heterogeneous work (GDN scan, depthwise conv, softmax, RMSNorm, gathers) between the big
quantized GEMMs, so some of it overlaps. That would inflate my per-call figures for the
widest, most contended rows precisely in proportion to width — which is the observed
pattern.

**None of this changes the conclusion of §5A/§5B.** The two component passes are
independent measurements at two widths and they agree on the thing that matters: the
verify-side quantized matmuls run at **35.1% (width 9) and 38.6% (width 8)** of their
roofline, while everything else in the round runs at 84-104%. Even after scaling my verify
estimate down by the full 12% discrepancy, verify quantized matmuls are ~177 ms of a
217.75 ms round.

### (6c.4) New finding from the same probe: **9 rows falls off the qmv dispatch cliff**

Chasing the floor discrepancy required knowing the exact Metal architecture, and that
probe produced the most actionable result in this section.

```text
mx.device_info()  ->  architecture "applegpu_g16s"   device "Apple M4 Pro"
                      arch_gen = 16, arch_size = 's'
```

`Vendor/mlx-swift/.../backend/metal/quantized.cpp:84` `get_qmv_batch_limit(D, O, d)` with
`arch_gen = 16` (so **not** the 13/14 branch) and `arch_size = 's'` returns, for every
projection shape in this model (all have D > 4096 and O > 4096):

```text
mlp gate/up   5120 -> 17408     vector_limit = 8
mlp down     17408 ->  5120     vector_limit = 8
lm_head       5120 -> 248320    vector_limit = 8
```

and the dispatch at `:1418` is

```cpp
int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
if (M >= vector_limit) {          // -> qmm_splitk (transpose_ && B == 1)
```

so **M <= 7 takes `qmv`; M >= 8 takes `qmm_splitk`.** `rows_per_round = depth + 1`, so
`segmentedVerifyDepthCap = 8` puts the verify at **M = 9**, one row past the cheap width.

This contradicts the design comment at `Qwen36MTPBlockSession.swift:588-590`, which asserts
*"Quantized projections at M in 6..9 still ride the per-row-exact QMV dispatch (host qmv
batch limit 10+ on this generation for these shapes)."* On `applegpu_g16s` the limit is
**8**, not "10+", so at depth 8 the projections are on `qmm_splitk`. Correctness is not in
question — bit-exactness was measured separately on the hexfloat row gate — but the **cost
model behind the depth cap is wrong on this host**.

The two component passes price it exactly:

```text
verify quantized-matmul subtotal   width 8   162.58 ms achieved / 62.68 ms roofline = 38.6%
                                   width 9   201.04 ms achieved / 70.49 ms roofline = 35.1%
marginal cost of the 9th row                  38.46 ms
average cost of a row at width 8              20.32 ms
=> the 9th row costs 1.89x an average row
```

The width sweep of §5B shows the same cliff in isolation on the MLP shape: 0.5582 ms at
M=7, 0.5622 ms at M=8, **0.6958 ms at M=9** (+23.8% for +12.5% rows).

**And the traced run agrees at the round level:**

```text
d=7 rounds (N=4)   round 190.336 ms   acc 7.00   -> 8.00 tokens   23.79 ms/token
d=8 rounds (N=9)   round 217.750 ms   acc 7.89   -> 8.89 tokens   24.49 ms/token
```

**Depth 7 is 2.9% cheaper per token than depth 8 on this host**, despite emitting fewer
tokens per round, because 8 rows stays on the cheap side of the dispatch cliff.

Caveats I will not paper over: N=4 versus N=9 rounds, depth is chosen adaptively so the
two samples are not matched on prompt difficulty, and `get_qmv_batch_limit` on the ranked
M5 depends on its own arch string (6 if `arch_gen` is 13 or 14, 12 if it is an Ultra
`'d'` part, 8 otherwise). That is precisely why the follow-up should *measure* the cliff
on the ranked host rather than hardcode 7. See §10 follow-up 0b — it is a two-line change
inside `Qwen36MTPBlockSession.swift`, entirely within the editable surface, and it is the
cheapest concrete speedup this experiment found.

## 6d. Answers to feedback (8) — `qwen38-r1-e4-fb8-head-ratio-correction`

The threshold correction is accepted and it is answerable from measurements already in
hand: `head_dtype_d8.json` times the head trunk and the compact draft readout as
**separate** rows, so both scopes can be reported without a new run.

### 6d.1 Which scope was timed — both

| scope | what is in the timer | measured 4-bit | measured bf16 |
|---|---|---:|---:|
| **A. head trunk only** | fc-concat, qkv, sdpa, o-proj, gate/up/down for one draft step | 1.005 ms | 3.426 ms |
| **B. trunk + compact draft readout** | scope A plus `draftTokenID`'s 98,336-row projection | 2.133 ms | 4.554 ms |

`head_dtype_d8` reports the trunk as a `×8` chain (8.040 ms 4-bit / 27.406 ms bf16); the
readout is the separate `head:draft_lm_head_compact` line of the `d=8` floor pass
(8 calls, 9.025 ms achieved against a 9.062 ms roofline = **100.4%** efficient, i.e.
1.128 ms per draft step). Per-step numbers above are those totals divided by 8.

### 6d.2 Scope A: **3.41×** — the bandwidth branch fires

```text
trunk bytes/step        4-bit 238.88 MB   bf16 849.35 MB   byte ratio 3.556x  (= 16 / 4.5)
trunk achieved x8       4-bit   8.040 ms  bf16  27.406 ms  wall ratio 3.41x
effective bandwidth     4-bit 237.7 GB/s  bf16 247.9 GB/s
```

**3.41× ≥ 2.8× ⇒ BANDWIDTH-BOUND. There is no large fixed per-step cost hiding in the
head read.** The ≤2.2× branch — the one that would have rescued this experiment — did not
fire. That is the same verdict §3 reaches independently from the trace (host-only work is
599 µs/round = 0.35%), so the two methods agree.

**Why my ratio sits above the 2.8–3.3× band, and why that is not a red flag.** The
anchors in feedback (8) are end-to-end generation tok/s for whole models, where per-token
costs that do *not* shrink with weight precision — attention over a growing KV cache,
cache writes, sampling, host dispatch — dilute the ratio to ~84% of the byte ratio. My
number is an isolated `timeit_amortized` chain over the head's weight-reading ops only, so
those diluting terms are absent by construction. The one non-scaling item *inside* the
chain is measured explicitly and is tiny:

```text
head:attn_sdpa   4-bit 0.0434 ms   bf16 0.0405 ms   ratio 0.93x   (reads no weights)
chain excluding sdpa   4-bit 0.9617 ms   bf16 3.3853 ms   ratio 3.520x = 99.0% of 3.5550x
```

Removing the single fixed row recovers 99.0% of the theoretical 3.5550×, which is exactly
what a clean bandwidth-bound measurement should do. So the 3.41× and the ~3.0× anchors are
consistent measurements of the same physics at different scopes, not a contradiction. The
practical reading for the campaign: an end-to-end ranked head swap should be budgeted at
the advisor's ~3.0×, not at my 3.41×.

### 6d.3 Scope B: **2.13× measured against the predicted ~1.9×**, and the 522.1 MB figure is confirmed

```text
                          4-bit (ranked-like)      bf16 (local)        ratio
trunk bytes/step               238.88 MB             849.35 MB
compact readout bytes/step     283.21 MB             283.21 MB        (does not shrink)
total bytes/step               522.09 MB            1132.56 MB        2.169x
predicted wall (fb8, 227 GB/s)   2.30 ms               4.99 ms        ~1.9x
measured wall                    2.133 ms              4.554 ms       2.13x
```

**The 522.1 MB/step ranked figure is confirmed to 0.003%** — my independently derived
compact slice is `98,336 × 5,120 × 0.5625 B = 283,207,680 B` exactly, and
`238.88 + 283.21 = 522.09 MB`. That is far inside the ±15% the advisor asked for, so the
precondition he set for opening a dedicated experiment is met.

The measured 2.13× runs a little ahead of the predicted 1.9× for the same reason as §6d.2
(isolated chain, and this host sustains ~250 GB/s rather than the 227 GB/s assumed), but
the *structure* of the prediction is exactly right: **the readout is already 4-bit in both
builds, so it is pure ballast in the ratio.** I am explicitly not reading 2.13× as "the
head barely matters" — the head matters at 3.41×; it is the readout riding alongside it
that pulls the combined figure down.

### 6d.4 Campaign answer: the compact draft readout is the largest remaining drafting-bandwidth item

```text
ranked (4-bit head)   readout share of drafting bytes = 283.21 / 522.09 = 54.24%
local  (bf16 head)    readout share of drafting bytes = 283.21 / 1132.56 = 25.01%
```

Confirmed, with the mechanism: the compact readout is a slice of the **target**
checkpoint's `lm_head` (`Qwen35.swift:2344-2352`, `:2361-2387`), which is fixed at affine
4-bit group-64 in every build. Quantizing the proposal head cannot touch it, and its share
of drafting bandwidth *more than doubles* when the head is quantized. On the scored
machine it is the single biggest drafting-bandwidth line.

Two things make it a much better experiment than it first looks:

1. **The draft readout does not have to be exact.** It only selects proposals; the target
   verifies every emitted token and the ledger/verify values never come from this path
   (the source comment at `:2332-2335` says so explicitly). Approximation here costs
   acceptance rate, not correctness — so it is legal under the work-honesty rules and
   cheap to evaluate as net tokens/second.
2. **The hook already exists.** `Qwen35.swift:2135-2142` loads
   `mtp.draft_lm_head.{weight,scales,biases}` from the declared head tree, and `:2337-2343`
   infers the bit width from tensor shapes (`bits = w.dim(1) * 32 / k`). A candidate can
   ship a cheaper draft readout in `mtp-head/` with no new kernel work.

One caveat that changes the design, found while checking this: the declared-head branch is
hardcoded **full-vocabulary** (`:2362-2364`, `:2401-2403` disable the compact remap when a
declared head is present), and full-vocabulary is a dead end at any sane bit width —

```text
full vocab 248,320 x 5,120   4-bit(4.5b) 715.16 MB   3-bit(3.5b) 556.24 MB   2-bit(2.5b) 397.31 MB
compact slice (current)                                                                  283.21 MB
```

— every one of them is *worse* than the compact slice, because the compact path is already
an effective **1.78 bits per weight** amortized over the full vocabulary. So the experiment
is "let a declared draft head be **compact**", not "declare a draft head". Sizing and the
exact two-guard change are in §10 follow-up 0d.

### 6d.5 Items the advisor asked me to keep reporting

- **Three-number floor decomposition:** unchanged and still in §5A (roofline 88.87 ms /
  achieved-kernel 224.71 ms / measured block 217.75 ms), with the bf16-head correction
  (243.87 ms, −26.12 ms scheduling gap) and the `d=7` pass (202.40 / 190.34, −12.06) in
  §6c.3.
- **`rollbackRoundCount = 0`.** Zero repairs and zero rollbacks across all 28 rounds of the
  192-token window; `repair=1` never appears in the sub-trace. No rollback cost is hiding
  in any number in this report.
- **Verify `lm_head` is kept separate from the compact draft readout** and always has been:

  | line | rows | bytes | calls/round at d=8 | achieved | roofline | eff |
  |---|---:|---:|---:|---:|---:|---:|
  | `verify:lm_head_full_vocab` | 248,320 | 715.16 MB | 1 | 9.946 ms | 3.498 ms | 35.2% |
  | `head:draft_lm_head_compact` | 98,336 | 283.21 MB | 8 | 9.025 ms | 9.062 ms | 100.4% |

  They are different weights, different widths (M=9 versus M=1) and different regimes: the
  verify readout is compute-bound and 35% efficient — it belongs to the §5B width-cliff
  story — while the draft readout is bandwidth-bound and already at roofline, so its only
  available lever is reading fewer bytes.

## 7. Score arithmetic

Pinned `serial_decode_seconds_per_token_mean = 0.037994794617407023` over 512 tokens
⇒ serial leg ≈ 19.453 s. At the promoted frontier the candidate leg ≈ 6.699 s, so
`d(score)/d(candidate seconds) ≈ −0.43` ⇒ **100 ms saved ≈ +0.043 score**.

Removing 100% of host graph construction saves `0.599 ms × rounds`. Over a 512-token
ranked leg at the scored pool's effective depth 1 there are ~512 rounds ⇒ ~0.31 s ⇒
**+0.013 score**, and that is the *unachievable* upper bound. A realistic 20% shaving of
host build is **+0.003** — indistinguishable from run-to-run noise.

## 8. Fidelity and run evidence

192-token gated `--local-iterate`, commit `c39e178`:

| field | value |
|---|---|
| score (local serial:MTP ratio) | 1.8625159266497098 |
| serial s/token | 0.086801041538516685 |
| MTP s/token | 0.046604187538226448 |
| decode tokens | 192 |
| `mtp_depth` | 8 |
| `effective_mean_draft_len` | 6 |
| `accepted_draft_rate` | 0.97619047619047616 |
| `residual_divergence_count` | **0** |
| `all_tokens_matched` | **true** |
| `public_drift_tripwire_passed` | **true** |

Window sensitivity: 64 tokens → 1.462, 192 tokens → 1.8625.

**One blocking eval per round confirmed**: the single blocking `eval(...)` is at
`Qwen36MTPBlockSession.swift` ~line 1000 (post-verify readout). The serial path's
blocking eval is ~line 781. A second blocking eval exists only on the rare generic
repair path (~line 1086-1100), which **did not fire in this window** (`repair rounds:
none`).

### 8.1 W&B record

Two runs, one per assignment revision. Both are in
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai` and both are `finished`.

| revision | run ID | run name | URL |
|---|---|---|---|
| r1 | `ma8cga81` | `e4-host-draft-build-decode-round-floor` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ma8cga81 |
| **r2** | `j0z3rmty` | `r2-kv-sweep-and-thermal-block` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/j0z3rmty |

The r2 run carries `config.revision_id = "r2"` and
`config.base_sha = "67bde70274c42aef089ac73cf00608d8037a815e"`, re-logs every r1 namespace
so the two are directly comparable, and adds the §8.4.4 sweep:

- `kv_sweep/kv<K>/{achieved_ms,roofline_ms,verify_bytes_MB}` for K ∈ {512, 640, 768, 896,
  1024, 1280}.
- `kv_sweep/component/<name>/delta_pct` — per-kernel change across the 512→1024 walk.
- `kv_sweep/achieved_delta_pct_512_to_1024` = **+0.377**.
- `kv_sweep/host_share_pct_at_kv512` = 0.350 and `kv_sweep/host_share_pct_at_kv1024` =
  0.349 — the assigned hypothesis's headline number at both ends of the 512-token window.
- Two more `wandb.Table`s: `kv_sweep/floor_vs_kv_len` and `kv_sweep/component_delta`.

Logged namespaces (both runs):

- `A-trace-base/*` and `trace-free-192/*` — score, s/token, trace sub-step means,
  guardrail trio, model/session scalars for both benchmark windows.
- `floor/*` — the three numbers of §5A (`roofline_floor_ms`, `achieved_kernel_floor_ms`,
  `measured_block_ms`) plus both gaps.
- `floor/machine/*` — `bw_eff_gbs`, `flops_eff_tflops`, `balance_flop_per_byte`.
- `floor/component/<name>/{roofline_ms,achieved_ms,efficiency}` — the 16 rows of the §5A
  component table, including `verify:lm_head_full_vocab` and
  `state:recurrent_snapshot_48_layers` as separate lines.
- `floor/qmm_width/M<M>/{efficiency,ms,tflops}` — the §5B width sweep.
- Two `wandb.Table`s: `floor/components` and `floor/quantized_matmul_width_sweep`.

### 8.2 Reproduction

```bash
# 1. build (release swift + runtime worker)
scratch/build-release.sh

# 2. the 192-token traced decode window that produced §3/§4/§6
TOKENS=192 scratch/run-bench.sh trace-free-192 1 --local-iterate
#    -> scratch/results/trace-free-192/{trace.log,stdout.log,stderr.log}
#    trace=1 exports MLX_QWEN_MTP_TRACE=1, MLX_QWEN_MTP_TRACE_FILE=<dir>/trace.log,
#    MLXFAST_NO_SANDBOX=1  (see §1a for why all three are required)

# 3. sub-step decomposition
scratch/analyze-sub.py scratch/results/trace-free-192/trace.log 1

# 4. the roofline / achieved-kernel floor and the width sweep (§5A, §5B)
scratch/mlxenv/bin/python research/round_floor.py \
  --depth 8 --kv-len 1120 --measured-block-seconds 0.21775 \
  --sweep --out scratch/results/round_floor_d8.json

# 5. the same pass one row lower — the cheap side of the dispatch cliff (§6c.4)
scratch/mlxenv/bin/python research/round_floor.py \
  --depth 7 --kv-len 1120 --measured-block-seconds 0.190336 \
  --out scratch/results/round_floor_d7.json

# 6. head trunk at 4-bit vs bf16 storage (§6c.1, §6c.2)
scratch/mlxenv/bin/python research/round_floor.py \
  --depth 8 --kv-len 1120 --measured-block-seconds 0.21775 \
  --head-dtype-compare --out scratch/results/head_dtype_d8.json

# 7. Metal architecture + qmv batch limits (§6c.4) — 0.08 s, no model load
scratch/mlxenv/bin/python research/round_floor.py --device-only

# 8. publish to W&B
scratch/log-wandb.py --floor=scratch/results/round_floor_d8.json \
  e4-host-draft-build-decode-round-floor \
  scratch/results/A-trace-base scratch/results/trace-free-192
```

Every `round_floor.py` invocation now records the step-7 device probe in its own JSON
under the `device` key, so a floor number can never be read without the architecture that
produced it.

Job IDs behind the numbers, all exit 0:

| job | wall | produced |
|---|---:|---|
| `e6a2f1c1-849f-…` | ~13 min | 192-token traced window (§3, §4, §6) |
| `56814494-…` | ~6 min | 64-token baseline window (run A) |
| `7d99e5b3-…` | 20.5 s | `round_floor_d8.json` (§5A, §5B) |
| `c33fea48-74e8-4c4c-9103-0d01cdb25812` | 10.5 s | `round_floor_d7.json` (§6c.4) |
| `5851eca0-13a3-4a47-b707-44b0129216c6` | 13.9 s | `head_dtype_d8.json` (§6c.1, §6c.2) |
| `b311e34a-f7f0-45dd-a5d8-e4bf0c8ec9c7` | 0.075 s | device probe (§6c.4) |
| `73b03979-b688-4976-b757-32c52857164c` | 16.9 s | W&B run `ma8cga81` |

`scratch/` is outside the checkout and outside Git; `research/round_floor.py` is
committed but never packaged (§9).

### 8.2b Proposal-head provenance (local runs)

Recorded so the local-versus-ranked head split of §6b.1 is auditable:

```text
dir              ~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head
source           EigenLabs/Qwen3.8-27B-MTP-bf16 @ 26a328e070875b0314d652a039b6b59902690f03
                 (hardcoded at setup-qwen-mtp.sh:66-67, verified against
                  fixtures/qwen3_8_27b_mtp_head.sha256)
model.safetensors sha256 8fceddc664f3ea96d02e304463aa1319213ff52cdf1f3401d4bce64e7075c349
bytes            849,400,347   index total_size 849,398,784 = 424,699,392 params x 2 B
dtype            bfloat16, 15 bare tensors, no `.scales` -> stays dense
```

The ranked candidate leg instead uses `mtp-head.manifest.json`:
`hf:lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaff972fd3ca2be08f3640603b47e9ce70a`,
sha256 `cc209e30d8a7def1fc4d785be22b0ec40e16ae6763f9591255a1996a34f08f0d`,
238,934,093 B of a 2,147,483,648 B cap. The repo's `mtp-head/` holds only `README.md`,
and the pinned serial leg never uses a candidate head.

### 8.3 Host caveat — this M4 Pro is ~22% slower than the E3 host

All absolute milliseconds here are **this host's**, not the ranked M5's, and not
`qwen-thorfinn`'s. My pinned serial decode is **86.80 ms/token** against thorfinn's
**67.0 ms/token** — a 1.30× spread. Expressed as effective bandwidth on the verified
14.412 GB/token verify stream (§5.2) that is **166 GB/s here vs 214.2 GB/s** on their
host.

Cause: a runaway `WirelessRadioManagerd` pinned at ~100% CPU for the whole session. It
is a root daemon and this runner has no `sudo`, so it could not be killed. It inflates
wall-clock uniformly rather than selectively, which is why the *ratios* in this report
(efficiency %, share-of-round %, marginal cost ×) are the load-bearing quantities and
the absolute ms are context. §5A's floor is computed from **this host's own** measured
`BW_eff`/`FLOPS_eff`, so the 2.51× kernel-efficiency gap is internally consistent
regardless of the slowdown; §5.3's cross-host reconciliation uses thorfinn's constants
explicitly.

### 8.4 The 512-token window is blocked by a host thermal fault — exact commands and output

Feedback (9) asks for headline numbers at 512 decode tokens and says: *"If you cannot
run, post a comment naming the exact command and its output."* I cannot run, and this is
that record. **The headline numbers in this report are therefore at 192 decode tokens,
not 512.** §8.4.3 quantifies what that does and does not cost.

#### 8.4.1 Four attempts, all killed by the pre-timing cool gate

| job | commit | base | wall | model held? | outcome |
|---|---|---|---:|:---:|---|
| `084340b2-…` | `e604ee3` | r1 `e20268e9` | 267 s | yes | exit 1 — cool gate abort |
| `5322fad0-b2f2-4a32-be54-801d80d1f4f6` | `e604ee3` | r1 `e20268e9` | 640 s | yes | exit 1 — cool gate abort |
| `4e1e7328-7d80-41a1-88f3-b4e841a46e59` | `bcdd5c6` | **r2 `67bde702`** | 277 s | yes | exit 1 — cool gate abort |
| `eafffc4a-ea77-4f11-9a65-7533af8baca6` | `bcdd5c6` | **r2 `67bde702`** | 210 s | **no — gate only** | exit 1 — cool gate abort |

The fourth row is the controlled one and it is the reason this is not a self-inflicted
problem. `benchmark-qwen-mtp.sh:146` runs the gate as
`cool_gate_command=("./benchmark.sh" "--local-cool-gate-only")`, so I ran exactly that
sub-step **in isolation, with no model loaded and no GPU work of mine anywhere on the
box**:

```bash
./benchmark.sh --local-cool-gate-only
```

Its complete output (25 lines, the whole log):

```text
benchmark.sh: waiting for GPU to cool down before timing (current 40.6C, target <=40C, waited 0s)...
benchmark.sh: waiting for GPU to cool down before timing (current 40.6C, target <=40C, waited 10s)...
benchmark.sh: waiting for GPU to cool down before timing (current 40.6C, target <=40C, waited 20s)...
benchmark.sh: waiting for GPU to cool down before timing (current 40.5C, target <=40C, waited 30s)...
benchmark.sh: waiting for GPU to cool down before timing (current 40.5C, target <=40C, waited 40s)...
benchmark.sh: waiting for GPU to cool down before timing (current 40.5C, target <=40C, waited 50s)...
benchmark.sh: GPU cool-down is stalled and no interactive terminal is attached;
benchmark.sh: to force the fans to 70% manually, run: .../tools/fan-control.sh boost
benchmark.sh: waiting for GPU to cool down before timing (current 40.5C, target <=40C, waited 60s)...
   … (identical 40.5C lines at 70s … 170s) …
benchmark.sh: ERROR: GPU is hot and not cooling down (current 40.5C, min seen 40.6C, target <=40C, waited 180s).
benchmark.sh: something else appears to be loading the GPU. Close GPU-heavy
benchmark.sh: processes (other benchmarks, ML jobs, games, video encodes),
benchmark.sh: let the machine cool, and rerun. To debug without the gate, set
benchmark.sh: MLXFAST_LOCAL_COOL_GATE=0 (hot-start timings are not comparable).
```

Read that carefully: **`min seen 40.6C`**. Across 180 s in which I was running nothing at
all, the die never once reached the 40.0 °C gate, and never even dipped below 40.5 °C.
The gate is not waiting for *my* heat to dissipate — there is no heat of mine to
dissipate. The host's floor is simply above the threshold (§8.4.2). This run also
confirms the script degrades safely without a tty: `benchmark.sh:822`
(`if ! { : < /dev/tty; } 2>/dev/null;`) detects the missing terminal and prints the
fan-boost hint instead of blocking on a prompt.

The exact command for the third row (full pipeline, post-rebase, on the r2 base):

```bash
TOKENS=512 scratch/run-bench.sh r2-trace-512 1 --local-iterate
# expands to, with the §1a trace variables exported:
#   MLXFAST_QWEN_MTP_DECODE_TOKENS=512 ./benchmark-qwen-mtp.sh --local-iterate
```

Its exact output (tail of `scratch/results/r2-trace-512/stderr.log`):

```text
benchmark.sh: waiting for GPU to cool down before timing (current 40.1C, target <=40C, waited 210s)...
benchmark.sh: ERROR: GPU is hot and not cooling down (current 40.1C, min seen 40.3C, target <=40C, waited 220s).
benchmark.sh: something else appears to be loading the GPU. Close GPU-heavy
benchmark.sh: processes (other benchmarks, ML jobs, games, video encodes),
benchmark.sh: let the machine cool, and rerun. To debug without the gate, set
benchmark.sh: MLXFAST_LOCAL_COOL_GATE=0 (hot-start timings are not comparable).
benchmark-qwen-mtp.sh: GPU cool gate failed before the MTP reference pass; free the GPU and rerun
```

No timed leg ever started, so there is no partial result to salvage —
`scratch/results/r2-trace-512/{stdout.log,score.json}` are empty and `trace.log` was
never created.

#### 8.4.2 Root cause: the host's *idle* floor is above the gate

This is not a hot-GPU problem that waiting fixes. Measured after ~12 minutes with **zero**
GPU work on the machine:

```console
$ macmon pipe -s1 | jq -r '.temp.gpu_temp_avg, .temp.cpu_temp_avg'
40.174556732177734
40.074424743652344

$ ps -Ao pid,user,pcpu,comm -r | head -4
  PID USER     %CPU COMM
49894 root    103.1 /usr/sbin/WirelessRadioManagerd
  102 _windowserver 14.1 .../WindowServer
15542 ec2-user 11.9 .../venv/bin/python -m senpai_agent.controller student
```

The gate needs **≤ 40.0 °C** (`COOL_GATE_TEMP_C=40`, `benchmark.sh:28`); the idle floor is
**40.17 °C**. A runaway root-owned `WirelessRadioManagerd` (103% CPU, up from 87% earlier
in the session; it has already respawned under a new PID once) plus `bluetoothd` (9.3%)
hold the package there. No `mlxfast` worker of mine was resident — the only `ec2-user`
process is my own Senpai controller, which I cannot kill without ending the session.

I sampled this four times over the session. It is not drifting downward:

| sample | GPU °C | CPU °C | `WirelessRadioManagerd` %CPU | `bluetoothd` |
|---|---:|---:|---:|---|
| 1 | — | 40.07 | 87 (earlier PID) | — |
| 2 | 40.17 | 40.07 | 103.1 (PID 49894) | 9.3% |
| 3 | 40.61 | 40.17 | 99.7 (PID 49894) | 8.3% (PID 43454) |
| 4 | 40.69 | 40.20 | 99.8 (PID 49894) | 4.1% (PID 43895) |

`WirelessRadioManagerd` has held one core saturated for the entire assignment under a
stable PID (49894), while `bluetoothd` keeps crashing and respawning under new PIDs
(43454 → 43895) — the classic signature of a wedged Bluetooth/Wi-Fi coexistence daemon.
Both are root-owned; this runner has no non-interactive `sudo`. The GPU die trend over
the session is **40.17 → 40.61 → 40.69 °C**, i.e. moving *away* from the gate, so
"wait longer" is not a fix available to me.

A secondary factor makes the gate fire *early*: progress requires a **0.25 °C** drop
(`COOL_GATE_PROGRESS_EPSILON_C`, `benchmark.sh:33`), but the die falls ~0.1 °C per poll,
so the stall detector trips while the GPU is still genuinely, slowly cooling — at 220 s in
the model-holding runs and at the `COOL_GATE_ABORT_AFTER_S=180` floor (`benchmark.sh:30`)
in the isolated gate-only run, which saw no progress at all to reset the timer.
The gate constants at `benchmark.sh:28-34` are `readonly` and have no environment
override, so the gate cannot legitimately be made *more patient*.

**Every escape route is either unavailable or forbidden, and I took none of them:**

- `tools/fan-control.sh boost` — the remedy `benchmark.sh` itself offers. Its own header
  documents that SMC writes need root and that "sudo prompts for your password itself, on
  your terminal". No automated session has an interactive sudo prompt.
- `MLXFAST_LOCAL_COOL_GATE=0` — would work, and is **forbidden** by `program.md` ("Do not
  bypass the lock or cooling gate"). Hot-start timings are also not comparable.
- `MLXFAST_GPU_TEMP_CMD` — a documented portability seam that would let me substitute any
  number for the sensor reading. Using it here would be falsifying the thermal gate, which
  is the same violation as disabling it. Not used.

#### 8.4.3 What the 192-token window does and does not cost

The assigned hypothesis is refuted by a **~70× margin** (host-only graph build is 0.35% of
the round, against the assignment's assumed ~24%). No window length flips a 70× margin,
and every Part A number is a *share* or a *ratio*, with §3 already reporting a
steady-state (round > 1) column that excludes warmup.

Two specific reassurances:

1. **The r1 192-token data is not contaminated by the EOS defect that motivated the r2
   rebase.** Feedback (9) states runs on the old base die at ~decode token 301. 192 < 301,
   so the traced window completed cleanly — 28 rounds, `all_tokens_matched: true`,
   `residual_divergence_count: 0`. The defect is why 512 was impossible *then*; it does
   not invalidate 192.

2. **Context length is not a confound.** A 512-token window walks the KV cache from 512 to
   1024. Only 16 of 64 layers are full attention; the other 48 are Gated DeltaNet and are
   O(1) in context. The full-attention KV stream is
   `2 × 4 heads × 1024 × 256 × 2 B × 16 layers = 67.1 MB`, against **14,412 MB** streamed
   per verify — **0.47%** of the round at kv=1024, and the 512→1024 growth adds only
   **0.23%**. §8.4.4 measures this directly rather than leaving it as arithmetic.

What is genuinely lost, and I am not claiming otherwise: a longer-horizon guardrail sample
(partly covered by the 28-round after-first guardrail in §6) and tighter per-depth
statistics from real accepted-length variation.

#### 8.4.4 Measured KV-length sweep — the 192-token result transfers to 512

The thermal gate blocks the *end-to-end* 512-token leg, but it does not block the question
that window length actually raises: **does a longer context change the round composition?**
That is a device-side question, and the component harness answers it directly without
holding the model or touching the benchmark wrapper, so it runs with no cool gate.

I swept `research/round_floor.py` at the ranked depth (`--depth 8`, verify width 9) across
the exact KV range a ranked leg traverses — a 512-token seed decoding 512 tokens walks the
cache from 512 to 1024 — plus 1280 as an overshoot control. 15 reps per point.

```bash
for KV in 512 640 768 896 1024 1280; do
  research/round_floor.py --depth 8 --kv-len $KV --reps 15 \
    --measured-block-seconds 0.21775 --out scratch/results/kv_sweep/kv_${KV}.json
done
```

Job `6c9199e9-e152-4183-9a7a-8cfe5902d894`, exit 0, 67.3 s, commit `bcdd5c6`.

| kv_len | roofline (ms) | achieved floor (ms) | kernel gap | vs kv=512 | verify bytes (MB) |
|---:|---:|---:|---:|---:|---:|
| 512 | 90.031 | 226.555 | 136.524 | — | 14,763.62 |
| 640 | 90.114 | 226.000 | 135.886 | −0.245% | 14,772.01 |
| 768 | 90.194 | 226.886 | 136.692 | +0.146% | 14,780.40 |
| 896 | 90.278 | 226.830 | 136.552 | +0.121% | 14,788.79 |
| 1024 | 90.350 | 227.409 | 137.059 | **+0.377%** | 14,797.18 |
| 1280 | 90.505 | 228.435 | 137.931 | +0.830% | 14,813.95 |

**Result 1 — the window is worth +0.38%.** Across the full 512→1024 walk the achieved
kernel floor moves **+0.854 ms on a ~227 ms round**. A least-squares fit gives
**2.746 µs per KV position** (roofline slope 0.616 µs). The assignment's central quantity,
the host-only share of the round, therefore moves from 0.350% at kv=512 to
`0.350% × 226.555/227.409 = 0.349%` at kv=1024. The hypothesis is refuted by ~70×; a
0.001-percentage-point context effect cannot touch that.

**Result 2 — only the two SDPA kernels move at all.** Of 16 components, 14 are flat within
scatter and two scale as expected:

| component | kv=512 | kv=1024 | Δ | Δ% |
|---|---:|---:|---:|---:|
| `verify:full_attn:sdpa` | 2.208 | 3.903 | **+1.695** | **+76.8%** |
| `head:attn_sdpa` | 0.198 | 0.313 | **+0.115** | **+58.2%** |
| `verify:mlp:gate_up_down` | 133.607 | 133.420 | −0.187 | −0.14% |
| `verify:lin_attn:in_proj_fused_qkvzba` | 31.738 | 31.678 | −0.060 | −0.19% |
| `verify:lm_head_full_vocab` | 9.953 | 9.955 | +0.002 | +0.02% |
| `head:draft_lm_head_compact` | 9.077 | 9.069 | −0.008 | −0.09% |
| all 48 `lin_attn` rows (subtotal) | 48.589 | 47.987 | −0.602 | −1.24% |

The 48 Gated DeltaNet layers are **byte-identical** across the sweep (`+0.000 MB`),
confirming they are O(1) in context; their −1.24% drift is measurement scatter, not signal.
The two SDPA rows grow by exactly the KV they read.

**Result 3 — the byte model is confirmed to the byte.** §8.4.3 predicted the
full-attention KV stream at kv=1024 as `2 × 4 heads × 1024 × 256 × 2 B × 16 layers`:

```text
analytic  2*4*1024*256*2*16                       = 67,108,864 B
measured  verify:full_attn:sdpa 4,194,304 B x 16  = 67,108,864 B
```

An exact match. The predicted 512→1024 growth was 0.23% of the verify stream; measured is
**+33.554 MB on 14,763.62 MB = +0.227%**.

The 14,763.62 MB here and the **14,412.3 MB** quoted in §5/§8.4.3 are the same model counted
to different boundaries, and they reconcile exactly. 14,412.3 MB is the pure *weight* stream
(`48 × 215.56 + 16 × 209.39 + 715.2`). The harness additionally counts the state and
activation traffic that the weight model omits:

```text
verify:lin_attn:gated_delta_kernel  (recurrent state, x48)   301.990 MB
verify:full_attn:sdpa               (KV at kv=512,   x16)     33.554 MB
verify:lin_attn:conv1d_depthwise_k4 (conv state,     x48)     15.729 MB
                                                    total    351.273 MB
14,412.3 + 351.3 = 14,763.6 MB
```

So the two figures differ by 2.4%, entirely accounted for, and the 0.47%-vs-0.4545% spread
in the two KV-share estimates is just which denominator was used.

**Result 4 — there is no 256-boundary cliff.** This was the one open worry in §8.4.3, since
MLX allocates KV in blocks of 256 and my window straddled no boundary. The sampled points
sit on both sides of the 512/768/1024/1280 boundaries. Residuals about the linear fit are
`+0.473 −0.434 +0.101 −0.307 −0.079 +0.244` ms — **the scatter is larger than any step**,
and the largest single anomaly is kv=640 coming in *below* kv=512. Block-boundary
reallocation costs nothing measurable on the scored path.

**Honest limits of this sweep.** It measures the *device* floor, not an end-to-end leg. It
cannot show accepted-length statistics, rollback behaviour, or scheduler drift over 512
tokens, and `--measured-block-seconds` was pinned to the r1 192-token measurement, so the
`scheduling gap` column is not independently measured here and I have not quoted it. What
it does establish is the specific thing that window length could have changed — round
composition and the byte model underneath it — and that is stable to well under 1%.

## 9. Candidate diff — **not for promotion**

No Part B mechanism was implemented: Part A proved every candidate mechanism bounded
below the ≥3% bar, and §5A/§5B relocated the headroom outside this assignment's scope.

What the branch does carry is the instrumentation that produced the evidence, kept
deliberately so the advisor and the next student can reproduce and extend it:

| path | ins | del | submitted? |
|---|---:|---:|---|
| `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` | 110 | 4 | yes (in `editablePaths`) |
| `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift` | 67 | 3 | yes (in `editablePaths`) |
| `research/round_floor.py` | 916 | 0 | **no** — research-only, outside `editablePaths` |
| `research/results/qwen38-r1-e4-host-draft-build.md` | new | 0 | **no** — this report |

**This candidate must not be officially submitted as-is.** The two Swift files add a
tracing path that must be stripped or compile-gated out before any Yukon submission.
The gating is cheap when disabled — `Qwen35MTPHostTrace.enabled` is a `static let` (env
read once), `Qwen35MTP.swift:91` guards with an early fast path, and `Qwen35MTP.swift:163`
hoists the flag out of the layer loop — but "cheap" is not "free", and a promoted
candidate should not carry a `FileHandle`/`NSLock` writer on the scored path.

Scope and budget re-checked at HEAD against the **r2** `BASE_SHA=67bde70274c42aef089ac73cf00608d8037a815e`:

```text
senpai/validate-assignment-scope.sh 67bde70274c42aef089ac73cf00608d8037a815e \
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift
  -> assignment scope OK: 2 submitted path(s)   [exit 0]

senpai/check-editable-budget.sh 67bde70274c42aef089ac73cf00608d8037a815e
  -> editable budget OK: source=2402616/3000000 bytes headroom=597384
     growth=7966/262144 exempt=2410/2147483648 files=154
     (base source=2394650, exempt=2410, files=154)   [exit 0]
```

The `growth=7966` figure is the candidate-surface growth of the two Swift files only; the
1,747-line report and the 916-line harness contribute nothing to it, which is the
mechanical confirmation that they are outside `editablePaths`.

`research/round_floor.py` is correctly rejected by the scope validator, confirming it is
never packaged.

## 10. Suggested follow-ups (not implemented)

0. **Widen the `qmv` weight-stream amortization to M ≤ 9** (§5B). Biggest measured prize
   in the campaign so far: 201.14 ms of a 217.75 ms round runs at 33–35% of this host's
   own roofline purely because `quantized_matmul` at M=3..12 re-streams a large share of
   the weight bytes per row. Editable surface is the kernel bodies
   (`kernels/quantized{,_nax}.{metal,h}`, `kernels/quantized_utils.h`) plus their
   runtime-effective `mlx-generated/*.cpp` twins; the host dispatcher
   `backend/metal/quantized.cpp` is **not** submittable, so the qmv/qmm selection is
   fixed and the work must go inside `qmv`/`qmv_quad`. Needs a fidelity plan: this
   touches reduction order and packing, so it requires exact-row numerical checks, not
   just an argmax match. Also needs an M5 confirmation — at D=5120/O=17408
   `get_qmv_batch_limit` returns **6** for `arch_gen ∈ {13,14}`, **12** for an Ultra
   (`arch_size == 'd'`), and **8** otherwise, so the ranked M5 may sit on a different side
   of the split than this `applegpu_g16s` M4 Pro.

0b. **Set `segmentedVerifyDepthCap` from the measured qmv batch limit instead of the
   constant 8** (§6c.4). This is the cheapest concrete speedup the experiment found and it
   is entirely inside the declared editable surface — two lines in
   `Qwen36MTPBlockSession.swift` (`:607` and the width-cap selection at `:613-619`).

   - **Mechanism.** MLX dispatches `qmv` while `M < vector_limit` and `qmm_splitk` at
     `M >= vector_limit`; `M = depth + 1`. Capping depth at `vector_limit − 1` keeps every
     verify projection on the cheap dispatch. On this host that is depth 7, not 8.
   - **Measured prize here.** 23.79 ms/token at d=7 versus 24.49 ms/token at d=8 (−2.9%),
     with the isolated MLP shape and the two component passes both agreeing (§5B, §6c.4).
     At the score sensitivity of §7 (`d(score)/d(candidate s) ≈ −0.43`), a 2.9% cut on a
     ~6.70 s candidate leg is ~0.19 s ≈ **+0.084 score** — roughly 20x the ceiling of the
     Part B work this assignment scoped.
   - **Do not hardcode 7.** Probe the limit at runtime and derive the cap. `mx.device_info()`
     exposes `architecture`; the Swift side can read the same Metal architecture string,
     or the cap can be calibrated once during the existing warmup by timing the model's own
     gate/up shape across M and taking the last width before the jump. Calibration is
     input-independent (it depends only on shapes and hardware), so it stays inside the
     work-honesty rules.
   - **Risk.** Lower depth means fewer tokens per round, so the win is a *net* claim, not an
     acceptance-rate claim. It must be validated with a matched `--local-iterate` pair, and
     the d=7 evidence here comes from N=4 adaptively chosen rounds versus N=9 at d=8 — not
     matched on prompt difficulty. A forced-depth A/B (the `MLX_QWEN_MTP_FIXED_DEPTH`
     override already exists in the instrumentation) settles it in one cheap run pair.
   - **Interaction.** If follow-up 0 succeeds and flattens the cliff, 0b becomes obsolete —
     they should not be composed blindly.

0c. **Correct the stale design comment at `Qwen36MTPBlockSession.swift:588-590`.** It states
   the host qmv batch limit is "10+ on this generation for these shapes", which is false on
   `applegpu_g16s` (it is 8) and false on the 13/14 generations (6). Whatever is decided
   about 0b, the comment should record the real rule and the fact that it is
   architecture-dependent, so the next reader does not re-derive a cost model from it. Note
   the *rest* of that comment block was verified correct in §6c.4: the 6..9-row chunking it
   describes applies at the sdpa only, and segmenting the whole forward was already measured
   and rejected for paying a second full weight pass.

0d. **Ship a *compact* low-bit `draft_lm_head` in `mtp-head/`** (§6d.4) — the dedicated
   experiment feedback (8) asked me to size. On the ranked build the compact draft readout
   is **54.24% of all drafting bandwidth** (283.21 MB of 522.09 MB/step) and head
   quantization cannot touch it, because it is a slice of the fixed target `lm_head`.

   - **Why it is tractable.** The readout only *selects* proposals; the target verifies
     every emitted token, so it may be approximate. It is also already at 100.4% of
     roofline, so the only lever is reading fewer bytes — a data change, not a kernel
     change.
   - **Sizing on measured numbers.** A ranked-like `d=8` round is
     `217.75 − 27.41 + 8.04 = 198.38 ms` (measured round with the 4-bit head trunk
     substituted for the bf16 one); the readout is 9.03 ms = **4.55%** of it. Replacing the
     4-bit compact slice with a 2-bit affine g64 compact slice gives
     `98,336 × 5120 × 0.3125 = 157.34 MB`, i.e. 0.63 ms/step and 5.02 ms/round —
     **−4.01 ms/round = −2.02%**, ≈ 0.135 s on a 6.70 s candidate leg ≈ **+0.058 score**.
     A 3-bit slice (220.27 MB) is worth about half that.
   - **The change is two guards, not a rewrite.** `Qwen35.swift:2135-2142` already loads
     `mtp.draft_lm_head.{weight,scales,biases}` and `:2337-2343` already infers the bit
     width from tensor shapes. What blocks a compact declared head is that `:2362-2364` and
     `:2401-2403` treat any declared draft head as full-vocabulary and disable the compact
     remap; they should key on the declared row count instead. `Qwen35.swift` is inside
     `editablePaths` (verified with `senpai/validate-assignment-scope.sh`), and 157 MB is
     trivial against the 2 GiB `mtp-head/` cap on top of the 238.9 MB ranked head.
   - **Do not attempt this full-vocabulary.** 248,320 rows costs 715.16 MB at 4 bits,
     556.24 MB at 3 bits and 397.31 MB at 2 bits — all *worse* than today's 283.21 MB,
     because the compact slice is already an effective 1.78 bits/weight over the full
     vocabulary.
   - **Risk and stop rule.** Coarser proposals cost acceptance rate. The claim is net
     tokens/second, so it must be judged on a matched `--local-iterate` pair plus
     `effective_mean_draft_len` and `accepted_draft_rate`, never on readout milliseconds.
     Also note the prize scales with achieved draft depth (one readout per draft step): at
     `d=8` it is 9.03 ms/round, at `d=1` it is 1.13 ms/round, so the ranked pool's depth
     profile decides whether this is worth a slot.

1. **Re-anchor `headStepCostRatio`** to ~0.224 on M4-class hosts and re-measure the
   depth schedule. The current 0.20 under-charges deep drafts, biasing the scheduler
   toward depths that do not repay themselves.
2. **Shrink the proposal head's bytes, not its efficiency.** §5A measures every head
   component at 84–100% of roofline, so there is no kernel headroom there — the head costs
   522.1 MB/step (17.07 ms of the round at d=8 with a 4-bit trunk) simply because that is
   how much it reads. The 283.21 MB readout half is follow-up 0d; the remaining 238.88 MB
   is the trunk itself, whose only lever is a smaller or lower-precision proposal head,
   which changes proposal quality and so must be measured end-to-end, not by acceptance
   rate alone.
3. **Revisit the `Qwen35.swift` asyncEval ladder** (lines 1858-1888). It fires 8 times
   per verify pass at width ≤ 9. Measuring whether fewer ladder points reduce
   `verify_build_us` is a scoped, cheap experiment — but note `Qwen35.swift` is outside
   this assignment's declared scope and would need an explicit scope grant.
4. **`Sources/MLXFastModel/Qwen35*.swift` is dead code** (`Qwen35FastPathReadiness.swift:11-19`
   hardcodes false; `selectQwen35ExecutionBackend` always returns `.libraryOracle`).
   Consider a cleanup assignment so future students do not optimize an unexecuted path.
