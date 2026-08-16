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
| §5A, §5B, §6c, §6d (kernel isolation) | **base-independent and re-verified**, but they measure *upstream* `mx.quantized_matmul`, which does **not** carry the base's `crossrow` kernels at M in 2..9. See §5C. Only the M=1 numbers transfer to the scored stack unchanged. |
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

Five further findings sat alongside that negative. **r3 retracts the first two**; the
remaining three are where the campaign value is:

1. ~~**§6c.4 — the depth cap sits exactly one row past a dispatch cliff.**~~ **WITHDRAWN in
   r3 — this was wrong, and the design comment it accused was right.** I mirrored
   `get_qmv_batch_limit` in `research/round_floor.py` with a transcription error (`8`
   where the C++ returns `10`). The real constant for this host — `applegpu_g16s`,
   arch_gen 16, size `'s'` — is **`vector_limit = 10`** for every projection shape in the
   model (`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:84`). The
   dispatch is `if (M >= vector_limit) -> qmm_splitk`, so `qmv` covers **M ≤ 9**.
   `segmentedVerifyDepthCap = 8` verifies at **M = 9 — the last row still on `qmv`, one
   short of the switch, not one past it.** There is no dispatch cliff at the cap. The
   design comment at `Qwen36MTPBlockSession.swift` ("host qmv batch limit 10+ on this
   generation for these shapes") is **correct as written**. Follow-up 0b is withdrawn.
   Full correction and the surviving unexplained d=7/d=8 observation: §6c.4.
2. **§5B — upstream `mx.quantized_matmul` degrades sharply with M, but this is *not* the
   scored kernel.** Measured through the Python binding it runs at 248.7 GB/s at M=1 and
   72.1 GB/s at M=9. That binding path takes the **upstream** kernel: the base's
   `crossrow` family (`Vendor/mlx-swift/.../kernels/quantized.h` + the generated twin)
   gates on `!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024` and
   dispatches `switch (ntg.x)` for **M = 2..9 only**. So §5B is a *no-crossrow* baseline
   for exactly the widths MTP verifies at, and no prize can be claimed from it — the
   scored M=2..9 curve is **unmeasured**. What survives is a smaller, explicitly
   unexplained observation: the upstream curve has a **flat step from M=7 to M=8**
   (0.5582 → 0.5622 ms, +0.7%, versus ~+14% per row elsewhere). I do not have an
   explanation for it and I am not attaching one. §5C audits every conclusion that
   depended on the old reading.
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

## 1a. Feedback (4) and (9): stderr really is discarded — so I added a file sink

**r3 correction. My r1/r2 wording here was wrong and I am replacing it.** I wrote that
the advisor's conclusion "does not follow" and that the trace "**is** reachable", as if
feedback (4) had misread a facility that already worked. That is not what happened.
The accurate statement is:

> **Worker stderr really is discarded, exactly as feedback (4) and (9) said. At base
> `67bde702` the MTP trace had no other outlet, so it was genuinely unreachable through
> the benchmark. I made it reachable by *adding a file sink that the base did not have* —
> `MLX_QWEN_MTP_TRACE_FILE`, introduced by my own commit `746a54b`.**

This matters for two reasons. First, it is a correction of attribution: the advisor was
right about the base, and I changed the base rather than found a way around it. Second, it
changes what the finding *is* — not "the trace works, you missed it", but "the trace needs
one small piece of tooling, here it is, and it should probably live in the base". That
promotion proposal is now §9a.

Provenance, checked rather than asserted:

```console
$ git grep -n "MLX_QWEN_MTP_TRACE" 67bde702 -- Sources/
67bde702:Sources/MLXFastModel/Qwen36MTPBlockSession.swift:463:  ...["MLX_QWEN_MTP_TRACE"] == "1"
                                    # ^ only the on/off flag exists at base; no sink

$ git log --oneline -S"MLX_QWEN_MTP_TRACE_FILE" -- Sources/
746a54b trace: route MTP phase trace to a file sink and add fixed-depth diagnostic
```

So at base the flag existed and wrote to stderr, and stderr went nowhere. The `MLX_`
allowlist prefix (`QwenRuntimeWorker.swift:2643`) **did** pre-exist — that part of my
argument stands and is what makes the sink variable reach the child at all — but the
file sink itself is mine.

All line numbers below are re-located by name on the r2 base `67bde702`.

**What the advisor got right (verbatim, and it is the load-bearing half).** Worker stderr
really is discarded:

- `Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift:2046` and `:2207` both spawn the
  worker with `emit: options.forwardsWorkerStderr ? nil : { _ in }` — the drain reads
  stderr and throws it away.
- `Sources/MLXFastCLI/main.swift:2224` defaults `forwardsWorkerStderr: Bool = false`, and
  `:2301` further ANDs it: `forwardsWorkerStderr && !officialRun`. Only the DFlash path
  (`main.swift:1409`, inside the cited `:1404-1412`) passes `true`.

So a stderr-based trace is indeed invisible — and at base the trace **was** stderr-based.

**What the added sink changes — two facts, one new and one pre-existing.**

1. **New (mine).** `Qwen36MTPBlockSession.swift:475-487` opens an append-mode `FileHandle`
   on the absolute path named by `MLX_QWEN_MTP_TRACE_FILE`. Discarding the worker's stderr
   has no effect on a file the worker opens itself. This code does not exist at
   `67bde702`.

2. **Pre-existing (the base's, not mine).** Both variables survive the worker env filter,
   and this is provable rather than assumed.
   `sanitizedRuntimeWorkerEnvironment` (`QwenRuntimeWorker.swift:2623`,
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
oracle. It is gated off by default (`traceRounds` is `false` unless
`MLX_QWEN_MTP_TRACE == "1"`).

This is reusable by any student who needs worker-side instrumentation.
`scratch/run-bench.sh` wires it up end to end. Because it is *added* tooling rather than a
rediscovery, r3 splits it out of the not-for-promotion pile and proposes it for the base on
its own: see **§9a**.


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

> **r3 correction.** `achieved_kernel_floor` is measured through `mx.quantized_matmul`,
> which takes the **upstream** kernel path and misses the base's `crossrow` family at
> M = 2..9. Read it as **`upstream_kernel_floor`**. §5C.4 gives the corrected
> interpretation of both gaps below; the short version is that a "floor" the real system
> beats by 3.8% is evidence the scored kernels are faster than the model, not evidence
> that scheduling is optimal.

| gap | value | reading (r1/r2) | reading (r3) |
|---|---:|---|---|
| kernel-efficiency gap (`achieved − roofline`) | **+135.93 ms** | 60.4% of the round is kernels below roofline | upstream-only; the scored gap is unmeasured |
| scheduling gap (`measured − achieved`) | **−8.27 ms (−3.8%)** | **essentially zero** — nothing left for host or scheduler | at least partly the crossrow speedup; weak evidence about scheduling |

r1/r2 treated the sign of the scheduling gap as "the whole answer to this assignment".
**It is not, and r3 demotes it to corroboration.** The sum of independently measured
*upstream* kernels slightly exceeds the real round, which is exactly what you would expect
when the real round runs faster kernels than the ones in the model. The assignment's
answer rests instead on the direct trace measurement of host time (599 µs/round =
**0.350%** of the round, §3/§4), which neither error touches. There
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

## 5B. Width sweep of **upstream `mx.quantized_matmul` (no `crossrow`)** — a baseline, not the scored kernel

> **r3 relabel.** In r1/r2 this section was titled "Headline finding: `quantized_matmul`
> collapses at exactly the verify widths MTP uses", and I priced a 2.50× round speedup
> from it. **Both are withdrawn.** The sweep drives `mx.quantized_matmul` through the
> Python binding, which takes the **upstream** kernel path. The campaign base carries an
> extra `crossrow` kernel family that the binding never reaches, and that family covers
> **exactly M = 2..9** — the whole region this table was used to indict. Details and the
> full audit of what depended on it are in §5C.
>
> What the table still is: a **legitimate upstream / no-crossrow baseline**. It is
> correct for what it measures, it is the right control to diff a future crossrow
> measurement against, and its **M=1 row is unaffected** (crossrow has no M=1 case and
> falls through to `qmv_fast_impl`, so M=1 is byte-identical to upstream). Every M ≥ 2
> row must be read as "upstream would do this", not "the scored stack does this".

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

Four readings, scoped by what the sweep can actually support:

1. **Transfers to the scored stack.** At M=1 the kernel is memory-bound and **99.5% of
   peak**. Serial decode is already optimal, and crossrow does not touch M=1, so this
   holds for the base as well as upstream. There is nothing to win there.
2. **Upstream only.** From **M≈3 to M≈12 upstream throughput is flat at ~2.6–2.9 rows per
   unit time** — time grows ∝ M, so batching buys essentially nothing *there*. Marginal
   cost of one extra verify row is **0.0618 ms** = 31% of an entire weight pass, where
   the ideal marginal is `2·K·N / FLOPS_eff = 0.0273 ms`. **2.26× too expensive at the
   margin** — for upstream. The base's crossrow kernels exist precisely to attack this
   region, and I did not measure them, so I cannot say how much of this gap the scored
   stack already closes.
3. **Upstream only, and unexplained.** There is a **flat step from M=7 to M=8**:
   0.5582 → 0.5622 ms, **+0.7%**, against ~+14% per row on either side (M=6→7 is +22.9%,
   M=8→9 is +23.8%). One extra row is very nearly free at that one point. In r1/r2 I
   attached this to a dispatch boundary; that explanation was wrong (§6c.4) and I have
   **no replacement for it**. I am recording it as an unexplained feature of the upstream
   curve rather than inventing a mechanism.
4. **Upstream only.** Efficiency snaps back to ~98% at **M ≥ 32** and absolute time is
   literally flat from M=12 to M=32 (0.8865 / 0.8899 / 0.8938 / 0.8917 ms) — 2.7× more
   rows for free, unreachable regardless because the parent caps drafts at 8, i.e. M ≤ 9.

**What I am no longer claiming.** r1/r2 concluded from this table that "the MTP round
lives entirely inside the worst part of this curve", priced verify-side quantized matmuls
at 201.14 ms of the 217.75 ms round, and derived a **2.50× round speedup** from closing
the gap, with the recommendation that the work "must go inside `qmv`/`qmv_quad`" to make
it amortize the weight stream. **All of that is withdrawn.** The round's matmuls run on
the base's crossrow kernels, not on the kernel this table measured, so neither the 35%
efficiency figure nor the prize derived from it applies to the scored path. A real
prize here may or may not exist; measuring it requires instrumenting the base's crossrow
kernels directly, which this experiment did not do.

**Scope note, kept because it is still true and still useful.** Verified against
`BASE_SHA:benchmark.json` with `senpai/validate-assignment-scope.sh`:

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
  For D=5120, O=17408 (both > 4096) it returns **6** on `arch_gen ∈ {13,14}`, **12** on
  Ultra (`arch_size == 'd'`), and **10** otherwise — including this host and the ranked
  M5. Since the dispatch is `if (M >= vector_limit) -> qmm_splitk`, `qmv` covers **M ≤ 9**
  here, so the entire reachable MTP range (M ≤ 9, drafts capped at 8) sits on the `qmv`
  side with **no boundary crossing at any legal depth**. **The selection cannot be changed
  by a submission; only the kernel bodies can** (`qmv_quad` :177, `qmv` :235;
  `qmm_nax` :473, `qmm` ~:699-771, `qmm_t_splitk` :798-846).

**r3 note on internal consistency.** This paragraph said "**10** otherwise" in r1/r2 and
was correct. The `8` appeared only in `research/round_floor.py`'s mirror of this function
and propagated from there into §6c.4, headline item 1, and follow-ups 0/0b — i.e. the
report contradicted itself and I did not notice. r3 fixes the mirror (§5C) and retracts
everything downstream of it.


## 5C. r3 audit: the two root errors and every conclusion re-checked against them

This section exists because two separate mistakes contaminated an unknown set of
downstream claims, and the honest thing is to enumerate the whole set rather than patch
the two the advisor happened to catch.

### 5C.1 Root error A — the `get_qmv_batch_limit` mirror returned 8 instead of 10

**Fix applied.** `research/round_floor.py` (the `_qmv_batch_limit` helper) returned
`12 if (D <= 4096 and O <= 4096) else 8`. The C++ returns `10` in that branch for this
host class. The literal is corrected and the docstring now names the dispatch direction
(`M >= vector_limit -> qmm_splitk`, so `qmv` covers `M <= vector_limit - 1`) so the sign
error cannot recur silently.

**Verification of the corrected mirror.** I re-implemented
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:84` faithfully — the
C++ switches on `arch_gen` first, then on `arch_size`; my mirror tests `arch_size == 'd'`
first — and compared exhaustively over
`D, O ∈ {1024, 2048, 4096, 5120, 17408, 248320}²  ×  size ∈ {d, s, g, p}  ×  gen ∈ {13..17}`:
**576 combinations, 0 mismatches.** The two orderings are outcome-identical because both
`arch_gen` branches return the same `32/18/12` triple for the `'d'` part.

The relevant values, from the source rather than from memory:

| condition | D,O ≤ 4096 | one > 4096 | both > 4096 |
|---|---:|---:|---:|
| `arch_size == 'd'` (Ultra) | 32 | 18 | **12** |
| `arch_gen ∈ {13,14}`, otherwise | 14 | 10 | **6** |
| all other (incl. gen 16 `'s'` = this host, and ranked M5) | 18 | 12 | **10** |

All three scored shapes — `5120→17408`, `17408→5120`, `5120→248320` — are "both > 4096",
so **`vector_limit = 10` and `qmv` serves M ≤ 9** on this host and on M5.

### 5C.2 Root error B — the Python sweep does not exercise the base's `crossrow` kernels

`crossrow` appears in exactly three places in the tree:

```console
$ git grep -lc crossrow
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h      22
Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp                    22
Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift                     1
```

The gate in the runtime-effective generated twin is
`if (!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024)`, then
`out_vec_size >= 4096` selects the `_m` variants via `switch (ntg.x)`:

| `ntg.x` (= M) | dispatched | | `ntg.x` | dispatched |
|---:|---|---|---:|---|
| 2 | `qmv_fast_crossrow_affine4_g64<T,2>` | | 6 | `..._m<T,6,3>` |
| 3 | `..._m<T,3,3>` | | 7 | `..._m<T,7,4>` |
| 4 | `..._m<T,4,4>` | | 8 | `..._m<T,8,4>` |
| 5 | `..._m<T,5,3>` | | 9 | `..._m<T,9,3>` |

**There is no `case 1`.** M=1 falls through to `qmv_fast_impl`, identical to upstream.
All three scored shapes have `out_vec_size ≥ 4096`, so they take the `_m` branch.

Consequence, stated plainly: **the scored stack's M=2..9 quantized-matmul curve is
unmeasured by this experiment.** §5B measures what upstream would do there.

### 5C.3 Every conclusion re-checked — what I actually looked at

I walked every claim in the report that reads either `qmv_batch_limit` / the dispatch
boundary, or a number produced by `research/round_floor.py` or the Python width sweep.
Twelve claims; here is each one and its verdict.

| # | claim | depends on | verdict in r3 |
|---|---|---|---|
| 1 | §1 headline 1 / §6c.4 — "cap is one row past a dispatch cliff" | error A | **retracted.** M=9 is the last `qmv` row; no cliff. |
| 2 | §1 headline 2 / §5B "2.50× round prize" | error B | **retracted.** Priced against upstream, not the scored kernel. |
| 3 | §10 follow-up 0b — "move the depth cap off the cliff" | error A | **withdrawn.** There is no cliff to move off. |
| 4 | §10 follow-up 0c — "the `:588-590` comment is stale" | error A | **withdrawn.** The comment is correct. |
| 5 | §10 follow-up 0 — kernel work, "8 otherwise" caveat | errors A+B | **kept, both caveats corrected.** Constant is 10/6/12; and the target must be the crossrow kernels, not upstream `qmv`. |
| 6 | §5A `achieved_kernel_floor` = 226.02 ms, gap +135.93 ms | error B | **relabelled `upstream_kernel_floor`.** Not a floor for the scored stack. See 5C.4. |
| 7 | §5A "scheduling gap −8.27 ms (−3.8%)" | error B | **reinterpreted.** See 5C.4 — this is now evidence *for* crossrow, not for scheduling quality. |
| 8 | §6.4 "3.45× redundant weight re-streaming" | error B | **retracted as a scored-path claim**; true of upstream at M=9. |
| 9 | §6c.3 "corrected kernel floor" | error B | **relabelled upstream.** |
| 10 | §6.5 / §6d.5 "verify readout is 35.2% efficient" | error B | **relabelled upstream.** |
| 11 | §6c.1, §6c.2, §6d.2, §6d.3, §6d.4, follow-up 0d | neither (M=1 only) | **unaffected — verified.** Every one of these measures the proposal head or the draft readout at **M=1**, where crossrow has no case and falls through. Head bandwidth-bound; 8-bit head refuted (249.0 vs 253.1 GB/s); Scope A 3.41×; Scope B 2.13× and 522.1 MB/step; compact readout 54.24% / 283.21 MB of 522.09 MB/step; follow-up 0d ≈ +0.058 score. |
| 12 | **§1 headline — the host-bound refutation itself** | neither | **unaffected — verified.** The 599 µs/round host cost (0.350% of the round) comes from direct `MLX_QWEN_MTP_TRACE` instrumentation inside the live scored worker, not from `round_floor.py` and not from the Python sweep. Neither error can move it. §3, §4 and §8.4.4 are likewise direct measurements. |

Claim 12 is the one that matters for the assignment's verdict, and it survives both
errors untouched. The result stays **not useful**: drafting is not host-bound.

### 5C.4 An honest new reading of the §5A "scheduling gap"

This is a consequence of error B that the advisor did not ask about and that I think is
the most interesting thing to fall out of the correction.

§5A computed a no-crossrow kernel floor of **226.02 ms** for the round and compared it
with the measured live round of **217.75 ms**, calling the −8.27 ms (−3.8%) difference a
"scheduling gap" and reading it as "the scheduler is already slightly better than the
model, so there is nothing left for the host". That reading is not available any more,
because a *floor* that the real system beats is not a floor.

The corrected reading: **the live round is faster than an upstream-kernel model of itself
because the base's crossrow kernels are faster than upstream at M=9.** The −3.8% is at
least partly a crossrow speedup that the model did not know about, not evidence of
superior overlap.

Two things follow:

- The corroboration weakens. §5A can no longer be cited as independent proof that "there
  is nothing left for the host or the scheduler". It is now consistent with a scheduler
  that has some slack which crossrow's gain happens to mask.
- **The headline refutation does not depend on it.** Host cost was measured directly at
  599 µs/round = 0.350%; even a scheduler with several ms of slack cannot make a 0.350%
  component the bottleneck. §5A was corroboration, not the argument.

The clean way to settle it is to measure the crossrow kernels at M=2..9 directly and
rebuild the floor on the real numbers. That is follow-up 0 as rewritten in §10.


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

**The max block is not caused by a width change or a rebuild.** It is an
ordinary `d=8` round. Spread is driven purely by schedule depth
(d=4 → 140 ms, d=8 → 218 ms). Width changes: 7 of 28. The guardrail has 3.1× margin.

> **Corrected in §6e.** `repair=none` here is the *full-repair* flag (`didRepair`), not
> the parent partial-rejection counter. Round 16 has `acc=7 < d=8`, so it **is** a
> partially-rejected round and it **did** take the cheap prefix-repair path. §6e measures
> that cost at **+1,018 µs (+0.47%)** against the other eight `d=8` rounds, which is far
> too small to explain the max: depth, not the repair, still drives the spread, and the
> guardrail margin is unchanged.

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

> **§6e note.** `repair=1` is `didRepair`, i.e. `fullRepairCount`, and that is exactly the
> flag this argument needs: `snapshot` is consumed *only* at line 1089 on the full-repair
> path. The 2–4 partially-rejected rounds took `restoreAfterPrefixReject`, which does not
> read `snapshot`. So the retraction below stands unchanged.

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
belongs in the discriminator.

> **r3 correction.** The `3.45x` and the `2.50x round speedup` that followed it are
> **withdrawn as statements about the scored path**. Both come from the §5B sweep, which
> runs the **upstream** kernel and never reaches the base's `crossrow` family at M = 2..9
> (§5C.2). They remain true of upstream at M=9. Two things survive intact: the BANDWIDTH
> branch label — the discriminator was pre-registered on *scaling behaviour*, and
> whatever the scored kernel's efficiency is, the residual still scales with bandwidth
> rather than being fixed host time — and the structural point that **small-M kernel
> inefficiency is a third category the dichotomy omits**. What is no longer known is
> **how much** of it the base has already removed.

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

**"Report `rollbackRoundCount`." — ~~ANSWERED: zero~~ — WRONG, corrected in §6e.**
`repair=1` in **0 of 28 rounds** is true but it is the *full-repair* flag, not
`rollbackRoundCount`. The correct answer is `fullRepairCount = 0` (measured) and
`rollbackRoundCount = prefixRepairCount ∈ [2, 4]` (derived). The **snapshot** term still
contributes **0 ms** — `snapshot` is only consumed on the full-repair path, which is the
one that measured zero — so 6.3's correction remains valid. See §6e.

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

> **r3 correction.** Both component passes call `mx.quantized_matmul`, so every
> multi-row number in this subsection — the 226.02 / 187.12 ms achieved floors, the
> corrected 243.87 / 202.40 ms floors, the 35.1% / 38.6% efficiencies and the ~177 ms
> verify estimate — is an **upstream, no-crossrow** figure (§5C.2). Read "corrected
> achieved floor" as "corrected *upstream* floor".
>
> This actually *helps* the open discrepancy above. I offered `timeit_amortized`
> contention as "the most likely remaining cause" of a floor that exceeds the
> measurement. There is now a second and simpler contributor: the model uses upstream
> kernels while the live round uses faster crossrow kernels at M = 9, so the model
> **should** overshoot, and should overshoot more at width 9 than at width 8 — which is
> the observed 12.0% / 6.3% pattern. I cannot separate the two contributions without
> measuring crossrow directly. The width-scaling argument in the third bullet is
> unaffected either way, and the head-chain bullet is an M=1 measurement and stands.

### (6c.4) **RETRACTED IN r3: there is no dispatch cliff at 9 rows, and the design comment was right**

r1/r2 titled this section "New finding from the same probe: 9 rows falls off the qmv
dispatch cliff" and called it the most actionable result in the report. **It was wrong.**
This rewrite states what is actually true, why I got it wrong, and what survives.

**The correct constant.** The probe itself was fine:

```text
mx.device_info()  ->  architecture "applegpu_g16s"   device "Apple M4 Pro"
                      arch_gen = 16, arch_size = 's'
```

The error was in my Python mirror of `get_qmv_batch_limit`, which returned `8` where the
C++ returns `10`. Reading the source instead of the mirror
(`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:84`), `arch_gen = 16`
takes the `default` branch, and `arch_size = 's'` takes *its* `default` arm, which returns
`18 / 12 / 10` for (both ≤ 4096) / (one > 4096) / (both > 4096). Every projection in this
model is "both > 4096":

```text
mlp gate/up   5120 -> 17408     vector_limit = 10
mlp down     17408 ->  5120     vector_limit = 10
lm_head       5120 -> 248320    vector_limit = 10
```

The dispatch (`:1415` and `:1483`, tested at `:1418`) is

```cpp
int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
if (M >= vector_limit) {          // -> qmm_splitk (transpose_ && B == 1)
```

so **`qmv` serves M ≤ 9 and `qmm_splitk` starts at M = 10.** `rows_per_round = depth + 1`
and the parent caps drafts at 8, so the maximum legal width is **M = 9 — the last row
still on `qmv`, one short of the switch.** The depth cap does not sit past a cliff; there
is no cliff anywhere in the reachable range.

**The design comment is correct.** `Qwen36MTPBlockSession.swift` states:

> *"Quantized projections at M in 6..9 still ride the per-row-exact QMV dispatch (host qmv
> batch limit 10+ on this generation for these shapes)."*

Limit 10, M ∈ 6..9 on `qmv`, "10+" — that is exactly right, including the choice of `10+`
rather than `10` (Ultra `'d'` parts return 12). My r1/r2 claim to have "falsified" it was
a transcription error in my own tooling, not a finding. The same claim is repeated later
in the file and is equally correct there. **Retracted with apologies to the comment.**

**What this invalidates.** The "1.89× marginal 9th row" arithmetic below was computed
from the §5B upstream sweep, and by error B (§5C.2) that sweep does not run the base's
crossrow kernels at M = 2..9 at all. So the numbers describe upstream, and even for
upstream they no longer mark a dispatch boundary:

```text
[upstream, no crossrow]
verify quantized-matmul subtotal   width 8   162.58 ms achieved / 62.68 ms roofline = 38.6%
                                   width 9   201.04 ms achieved / 70.49 ms roofline = 35.1%
marginal cost of the 9th row                  38.46 ms
average cost of a row at width 8              20.32 ms
```

M=7→8→9 in that sweep is 0.5582 → 0.5622 → 0.6958 ms. In r1/r2 I read the +23.8% at
M=8→9 as the cliff. It is not — M=9 is on the same kernel as M=8. If anything the
anomaly is the *other* step, the near-free M=7→8 (+0.7%), which §5B now records as
**explicitly unexplained**.

**What survives: one real, unexplained round-level observation.** This is a direct
measurement from the traced run and does not depend on either error:

```text
d=7 rounds (N=4)   round 190.336 ms   acc 7.00   -> 8.00 tokens   23.79 ms/token
d=8 rounds (N=9)   round 217.750 ms   acc 7.89   -> 8.89 tokens   24.49 ms/token
```

Depth 7 came out **2.9% cheaper per token** than depth 8 in this trace, despite emitting
fewer tokens per round. That happened; the *explanation* I attached to it did not. I am
recording it as an observation with no mechanism, and I am deliberately not proposing a
depth-cap change on the strength of it, because the confounds are severe enough to
account for the whole effect on their own:

- N=4 versus N=9 rounds — no error bars worth the name;
- depth is chosen **adaptively**, so the d=7 and d=8 samples are not matched on prompt
  difficulty, and the sampling is plausibly correlated with round cost;
- acceptance differs between the two groups (7.00 vs 7.89), so tokens-per-round and
  per-token cost are not independent of the depth choice.

The honest version of the follow-up is therefore not "move the cap" but "if anyone wants
to know whether depth 7 beats depth 8, run the matched fixed-depth comparison" — the
`MLX_QWEN_MTP_FIXED_DEPTH` override added in `746a54b` makes that a one-variable
experiment. I am not claiming a speedup here. **Follow-up 0b is withdrawn** and
follow-up 0c (which accused the design comment of being stale) is withdrawn with it.

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
- **~~`rollbackRoundCount = 0`~~ — WRONG; see §6e.** The measured zero is
  `fullRepairCount`, the expensive re-forward path. `rollbackRoundCount` (= partially
  rejected rounds = `prefixRepairCount` here) is **2–4**, all of them on the cheap
  tape-replay path. The corrected cost is **≈ +1,018 µs per prefix repair (+0.47% of a
  `d=8` round)**, still negligible against every number in this report.
- **Verify `lm_head` is kept separate from the compact draft readout** and always has been:

  | line | rows | bytes | calls/round at d=8 | achieved | roofline | eff |
  |---|---:|---:|---:|---:|---:|---:|
  | `verify:lm_head_full_vocab` | 248,320 | 715.16 MB | 1 | 9.946 ms | 3.498 ms | 35.2% |
  | `head:draft_lm_head_compact` | 98,336 | 283.21 MB | 8 | 9.025 ms | 9.062 ms | 100.4% |

  They are different weights, different widths (M=9 versus M=1) and different regimes: the
  verify readout is compute-bound and 35% efficient **on upstream** — r3 caveat: that
  35.2% comes from the §5B no-crossrow sweep, and the base's crossrow kernels cover M=9,
  so the scored figure is unknown and is very likely better (§5C.2) — while the draft
  readout is at **M=1**, where crossrow does not apply, so its 100.4% is a scored-path
  number: bandwidth-bound and already at roofline, leaving "read fewer bytes" as its only
  lever. The asymmetry in what I can claim about these two rows is entirely due to which
  side of the crossrow gate they sit on.

## 6e. Answers to feedback (10) and (11)

Both comments landed at 17:38:31Z and 17:45:07Z, before my 18:09:40Z submission, and I
did not reconcile them. This section is the correction. **One of my published numbers was
wrong**, and the corrected answer is stronger, not weaker.

### 6e.1 `fb10` — the overlap-vs-serialize verdict: **SERIALIZED**

The advisor called this "the single most load-bearing thing you can report", so it gets a
direct answer with the structural argument first and the arithmetic second.

**Verdict: the CPU `draft_build` leg and the GPU draft forward do NOT overlap. They
serialize, and the serialization is structural, not incidental.**

Source path, per round, per deep draft step (`Qwen36MTPBlockSession.swift`):

1. The host walks the head-step Swift code and appends lazy `MLXArray` nodes. No GPU work
   is scheduled yet — MLX is lazy, so this is pure host tape construction. This is the
   segment I named `draft_build` / `host_only`.
2. The host then calls `asyncEval(...)` on the step outputs. MLX's `async_eval` **walks
   and submits the tape on the calling thread** before it returns; the return only means
   "submitted", not "complete".
3. The host immediately needs the step's sampled token to build step *n+1*'s input, so it
   blocks. That block is the segment I named `tail_async`.

Steps 1 and 3 are therefore on the same thread, in strict order, with no producer/consumer
queue between them. There is no second host thread building the next step's tape while the
GPU drains the current one. **The host cannot hide behind the GPU here because it is the
thread that submits and then waits.**

The timing discriminator the advisor proposed (`wall ≈ sum` ⇒ serialize, `wall ≈ max` ⇒
overlap) is **degenerate on this workload**, and I want to be explicit about that rather
than claim it as confirmation:

| steady-state deep draft step | value |
|---|---:|
| host leg (`draft_build`, tape construction) | **33.1 µs** |
| GPU wait leg (`tail_async`) | **3,575 µs** |
| measured wall per step | **3,608 µs** |
| `sum(host, gpu)` | 3,608 µs |
| `max(host, gpu)` | 3,575 µs |
| `sum − max` | 33 µs (**0.9%**) |

The two predictions differ by 0.9%, which is inside sampling noise. So the timing test
alone cannot separate them; the verdict above rests on the source contract, and the timing
is merely *consistent* with it (`wall = sum` to within 1 µs).

**Why this makes the assignment's conclusion stronger, not weaker.** The advisor's concern
is that a serialized host leg might be *masked* on the local arm and only matter on the
ranked arm. The masking runs the other way:

- The host leg is **arm-independent**. It is Swift graph construction over a fixed number
  of ops; it does not touch weights. Steady-state 33.1 µs/step, ~85 µs/step including
  first-call warmup.
- The GPU leg is what shrinks on the ranked 4-bit arm: **4.554 ms → 2.133 ms** per deep
  step (§6d.3, measured on an isolated chain).
- So the host share goes from 33.1 / 3,608 = **0.92%** (local, bf16) to at worst
  33.1 / 2,133 = **1.55%** (ranked, 4-bit).

Even at the ranked arm's smaller GPU denominator, deleting **100%** of host graph
construction buys ≈1.5% of the draft step, and the draft step is itself ~12% of the round.
There is no hidden 2.4 ms/step CPU win on either arm. That is the refutation.

**GPU-side per-draft-step time, as requested:**

| arm | measured GPU per deep draft step | advisor roofline | delta |
|---|---:|---:|---:|
| local, bf16 head (isolated chain, §6d.3) | **4.554 ms** | 4.99 ms | −8.7% |
| ranked-like, 4-bit head (isolated chain) | **2.133 ms** | 2.30 ms | −7.3% |
| in-run `tail_async`/step (local, bf16) | **3.575 ms** | — | — |

Both isolated figures sit ~8% under the advisor's roofline because that roofline assumes
227 GB/s and this host sustains ≈250 GB/s on these shapes. The in-run `tail_async` (3.575
ms) is *lower* than the isolated bf16 chain (4.554 ms) because some draft GPU work spills
past the tail wait and is absorbed by the round's later blocking eval, so **the in-run
number is a lower bound on GPU draft time, not a contradiction**.

**On "no CPU-cut conclusion from a local null" — accepted, and I am not making one.** The
claim I am defending is not "a local A/B showed no difference". It is a *bound*: the host
leg is 33.1 µs/step measured directly, it is arm-independent by construction, and the
ranked GPU denominator is 2.13 ms. A bound of that shape transfers across arms in a way a
null does not. If the advisor wants the null as well, the honest statement is that I do
not have one and could not obtain one this session (§8.4 thermal fault).

### 6e.2 `fb11` — the counter split: **my submitted answer was WRONG**

The advisor asked me to split `rollbackRoundCount` into `prefixRepairCount` and
`fullRepairCount`. My submitted result said "the split counters are `prefixRepairCount`=0
and `fullRepairCount`=0", and §6.5/§6d.5 said `rollbackRoundCount = 0`. **`prefixRepairCount`
was not 0 and I never measured `rollbackRoundCount` at all.**

The bug in my reading, in source:

```swift
} else {
    rollbackRoundCount += 1            // line 1045 — fires on EVERY partial rejection
    ...
    if Self.restoreAfterPrefixReject(...) {
        prefixRepairCount += 1         // cheap tape-replay repair
    } else {
        didRepair = true               // line 1074
        fullRepairCount += 1           // expensive full re-forward
    }
}
```

My trace's `repair=` field emitted `didRepair`. So the "0 of 28 rounds" I published is an
exact measurement of **`fullRepairCount`**, and it says nothing about the parent counter.

**Corrected answer:**

| counter | value | basis |
|---|---:|---|
| `fullRepairCount` | **0** | directly measured, 28 of 28 rounds, `repair=0` |
| `prefixRepairCount` = `rollbackRoundCount` | **2 – 4** | derived from the per-depth acceptance table (§4c) |

Derivation, from the §4c aggregate and the run's `accepted_draft_rate`:

- `d=4`, N=10, mean `acc` 3.70 ⇒ 37 of 40 accepted ⇒ **deficit 3**.
- `d=8`, N=9, mean `acc` 7.89 ⇒ 71 of 72 accepted ⇒ **deficit 1**.
- `d=5,6,7`: mean `acc` exactly equals `d` ⇒ **deficit 0**.
- Total drafts 168, `accepted_draft_rate` 0.976190476 = 164/168 ⇒ **4 rejected**, which
  closes against 3 + 1 independently.

A partially-rejected round with `acc = k` at depth `d` discards `d − k` drafts, so the
deficit-3 at `d=4` is 1, 2, or 3 rounds, and the deficit-1 at `d=8` is **exactly one**
round — round 16, the guardrail max block, logged as `d=8, acc=7, repair=none`. Hence
`rollbackRoundCount ∈ [2, 4]`, with the `d=8` member identified exactly. I cannot narrow
it further: the raw per-round trace was in `scratch/` and is gone, and neither W&B run
(`ma8cga81`, `j0z3rmty`) carries a per-round table or history — only the floor and
component tables.

**This upgrades hypothesis 2 from vacuous to genuinely tested.** The question was whether a
partial rejection forces a second full 48-layer GDN recurrence. Under my wrong reading the
answer was "no partial rejection ever happened", which tests nothing. The correct reading
is: **partial rejection occurred at least twice, and the expensive path fired zero times.**
The eager post-primary checkpoint plus `restoreAfterPrefixReject` absorbed every one of
them. That is a real negative result about the repair machinery, not an absence of data.

**Measured prefix-repair cost (N=1, `d=8`):**

| quantity | µs |
|---|---:|
| round 16 (`acc=7`, the prefix repair) | 218,655 |
| other eight `d=8` rounds, mean = (9 × 217,750 − 218,655) / 8 | 217,636.9 |
| **delta** | **+1,018 (+0.47%)** |

So a prefix repair costs about **1 ms on a 218 ms round**, and it also commits one fewer
token. N=1, one depth, one prompt — directional only. It does not move any floor,
guardrail, or score number in this report, and it does not change the assignment verdict.

**Instrumentation added.** `prefixRepairCount` and `fullRepairCount` are now real
`public private(set)` counters incremented at the two branch sites above, and the sub-trace
emits `prefix_repair_total=` / `full_repair_total=` next to `repair=`, so the next traced
run reports exact integers instead of a derived interval. Research instrumentation only —
see §9, not for promotion.

### 6e.3 `fb10` — the retracted readout claim, reconciled against my measurement

The advisor retracted the "315 MB / 0.6 ms" compact-readout figure and settled on 283.2 MB
with a 1.25 ms floor. My report reached 283.21 MB independently (§6d.5) before that
retraction, so the byte count agrees exactly. On the floor we differ slightly:

| source | bytes | assumed BW | floor per call | measured per call |
|---|---:|---:|---:|---:|
| advisor (settled) | 283.2 MB | 227 GB/s | 1.25 ms | — |
| this report (§6d.5) | 283.21 MB | 214.2 GB/s (`BW_eff`) | 1.133 ms | **1.128 ms** |

Measured: `head:draft_lm_head_compact` = 9.0247 ms over 8 calls ⇒ **1.128 ms/call**,
i.e. 251 GB/s achieved. That is under the advisor's 227 GB/s-derived floor but well inside
this host's 273 GB/s peak, so it is a bandwidth-assumption difference, not a violation. The
conclusion is the same either way: the compact draft readout runs **at** its roofline
(100.4% of my `BW_eff` model), so the only lever on it is reading fewer bytes, not a better
kernel. Nothing in §6d.4's ranking changes.

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
**full**-repair path (~line 1086-1100), which **did not fire in this window**
(`fullRepairCount = 0`). The 2–4 prefix-repair rounds (§6e) do **not** add a blocking
eval — they reuse already-materialized rows — so "one blocking eval per round" holds for
every one of the 28 rounds.

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

# 5. the same pass one row lower (§6c.4 — r3: no dispatch cliff; this is just d=7)
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
| `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` | 120 | 4 | yes (in `editablePaths`) |
| `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift` | 67 | 3 | yes (in `editablePaths`) |
| `research/round_floor.py` | 916 | 0 | **no** — research-only, outside `editablePaths` |
| `research/results/qwen38-r1-e4-host-draft-build.md` | new | 0 | **no** — this report |

**This candidate must not be officially submitted as-is.** The two Swift files add a
tracing path that must be stripped or compile-gated out before any Yukon submission.
The gating is cheap when disabled — `Qwen35MTPHostTrace.enabled` is a `static let` (env
read once), `Qwen35MTP.swift:91` guards with an early fast path, and `Qwen35MTP.swift:163`
hoists the flag out of the layer loop — but "cheap" is not "free", and a promoted
candidate should not carry a `FileHandle`/`NSLock` writer on the scored path.

> **r3 refinement (advisor feedback 6).** That blanket "not for promotion" is the
> safe default, not the useful answer. §9a splits the diff into three piles and
> proposes exactly one of them — the trace **file sink** — for promotion into the
> base as campaign tooling, with a source-level argument that its disabled cost is
> zero. The sentence above remains correct about the branch *as a whole*.

Scope and budget re-checked at HEAD against the **r2** `BASE_SHA=67bde70274c42aef089ac73cf00608d8037a815e`:

```text
senpai/validate-assignment-scope.sh 67bde70274c42aef089ac73cf00608d8037a815e \
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift
  -> assignment scope OK: 2 submitted path(s)   [exit 0]

senpai/check-editable-budget.sh 67bde70274c42aef089ac73cf00608d8037a815e
  -> editable budget OK: source=2403212/3000000 bytes headroom=596788
     growth=8562/262144 exempt=2410/2147483648 files=154
     (base source=2394650, exempt=2410, files=154)   [exit 0]
```

The `growth=8562` figure is the candidate-surface growth of the two Swift files only; the
report and the 916-line harness contribute nothing to it, which is the mechanical
confirmation that they are outside `editablePaths`.

`research/round_floor.py` is correctly rejected by the scope validator, confirming it is
never packaged.

### 9a. Feedback (6): splitting the diff — one hunk I *do* propose for promotion

> **r3, in response to advisor feedback (6).** §9 above says "not for promotion"
> about the whole diff, which is the safe default but not the useful answer. The
> advisor asked me to split it and argue the promotable piece on its own merits.
> I propose exactly one hunk: **the trace file sink**. Everything else stays
> not-for-promotion, and I say why below rather than leaving it implied.

#### 9a.1 The three piles

The branch's 187 insertions across two files are not one change; they are three,
with very different promotion cases. The insertion counts below reconcile exactly
with §9's per-file table: A + B = 120 in `Qwen36MTPBlockSession.swift`, C = 67 in
`Qwen35MTP.swift`, and A's 4 deletions plus C's 3 are the branch's 7.

| # | change | file | ins/del | proposed |
|---|---|---|---:|---|
| **A** | **trace file sink** (`traceSink`, `traceLock`, `traceWrite` body) | `Qwen36MTPBlockSession.swift` | **+30 / −4** | **promote as campaign tooling** |
| B | sub-step timers, `mtp-sub:` line, `traceLastDraftCount`, `prefixRepairCount`/`fullRepairCount`, `fixedDepthOverride` | `Qwen36MTPBlockSession.swift` | +90 / 0 | keep on branch only |
| C | `Qwen35MTPHostTrace` + the traced layer/module paths | `Qwen35MTP.swift` | +67 / −3 | keep on branch only |

Pile A is the one that unblocked this experiment. Without it, `MLX_QWEN_MTP_TRACE=1`
produces a perfectly correct trace that nobody can read (§1a): the `mtp-timed` verb
builds worker options with `forwardsWorkerStderr: false`, and the drain discards the
bytes. Any future student who reaches for the base's tracing switch under the
benchmark wrapper will rediscover that dead end. Piles B and C are this
assignment's questions rendered in code, and they should die with the assignment.

#### 9a.2 The exact hunk I propose

This is pile A verbatim, as it would apply to `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
immediately after the existing `traceRounds` declaration:

```swift
    /// The `mtp-timed` verb the benchmark wrapper drives builds its worker
    /// options with `forwardsWorkerStderr: false`, so the drain reads and
    /// DISCARDS everything the worker writes to stderr. A local trace run
    /// therefore names an absolute sink with `MLX_QWEN_MTP_TRACE_FILE` and
    /// runs the worker unsandboxed (`MLXFAST_NO_SANDBOX=1`, refused on
    /// official runs) so this append is permitted.
    private static let traceSink: FileHandle? = {
        guard traceRounds,
              let path = ProcessInfo.processInfo.environment["MLX_QWEN_MTP_TRACE_FILE"],
              !path.isEmpty
        else { return nil }
        // Append, never truncate: a local benchmark spawns one worker per leg
        // and all of them share this sink.
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let handle = FileHandle(forWritingAtPath: path) else { return nil }
        handle.seekToEndOfFile()
        return handle
    }()

    private static let traceLock = NSLock()
    private static func traceWrite(_ line: String) {
        let data = Data(line.utf8)
        traceLock.lock()
        defer { traceLock.unlock() }
        if let sink = traceSink {
            sink.write(data)
        } else {
            FileHandle.standardError.write(data)
        }
    }
```

It replaces the base's four-line `traceWrite` body (three comment lines plus the
unconditional `FileHandle.standardError.write`). Nothing else in pile A touches
another symbol, so it applies cleanly without piles B or C.

#### 9a.3 Why it is zero cost when disabled

Seven checks, each verified against source on this branch rather than asserted:

1. **The gate is the base's own flag.** `traceRounds` is
   `private static let ... == "1"` (`Qwen36MTPBlockSession.swift:472`). Pile A adds
   no new gate and does not widen the existing one.
2. **Every `traceWrite` call site is already inside that gate.** Grepping the file
   gives exactly three callers: two `if Self.traceRounds { ... Self.traceWrite(...) }`
   blocks (lines 1143 and 1172, the latter being pile B) and `traceRow`, whose first
   statement is `guard traceRounds else { return }` (line 522). There is no path from
   a ranked round into `traceWrite`.
3. **Swift statics are lazy, so a disabled build never runs the closure.** `traceSink`
   and `traceLock` are `static let`s, initialized on first access via `swift_once`.
   Their only accessor is `traceWrite`. With `traceRounds == false` nothing calls
   `traceWrite`, so the closure never runs: no `ProcessInfo` read for the second
   variable, no `FileManager` stat, no `createFile`, no `FileHandle`, no file
   descriptor, no `NSLock` allocation. The disabled cost is one `swift_once` token's
   worth of BSS and zero instructions on the scored path.
4. **Belt and braces inside the closure.** Even if some future caller forgot the
   outer gate, the closure's own first condition is `guard traceRounds`, so it still
   returns `nil` and `traceWrite` still falls through to the base's stderr write.
   The degraded behaviour is *exactly* the base's behaviour, not a different one.
5. **No allowlist change is needed.** `MLX_QWEN_MTP_TRACE_FILE` inherits the `"MLX_"`
   prefix already in the worker's env allowlist
   (`Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift:2643`, mirrored at
   `Sources/MLXFastHarness/QwenRuntimeWorker.swift:2578`). Pile A does not edit the
   allowlist, and the allowlist is trusted surface I must not edit anyway.
6. **The ranked workflow never sets either variable, and could not use the sink if it
   did.** `.github/workflows/qwen-mtp-ranked-benchmark.yml` sets neither
   `MLX_QWEN_MTP_TRACE` nor `MLX_QWEN_MTP_TRACE_FILE`. And the sandbox denies
   `file-write*`, so an official run that somehow had both set would fail to open the
   handle and fall back to stderr. Lifting the sandbox is itself refused:
   `enforce_official_sandbox()` (`benchmark.sh:1252-1259`) exits 1 when
   `MLXFAST_OFFICIAL_BENCHMARK_RUN=1` and `MLXFAST_NO_SANDBOX=1` are both set.
7. **It is phase-independent, so it is not a benchmark detector.** The flag changes
   *where bytes go*, never what work is done, which drafts are proposed, or which
   rows are evaluated. Both legs of a paired local run see identical code. Nothing in
   the trace line is fed back into generation — that is the §8 fidelity rule, and
   pile A does not weaken it.

#### 9a.4 Why I am *not* proposing piles B and C

Not from caution — from the campaign's own "one obvious training path" rule.

- **Pile B** is the sub-step decomposition that answered *this* question. The
  `mtp-sub:` line has 25 fields tuned to the draft-build hypothesis; a future
  experiment on a different mechanism would want different fields and would be worse
  off starting from mine. `prefixRepairCount`/`fullRepairCount` are the one part I
  would flag as arguably reusable — they fix a real ambiguity in
  `rollbackRoundCount` (§2, feedback 10) — but they are two counters behind a
  30-field trace line, and I would rather the advisor promote them deliberately later
  than smuggle them in behind pile A.
- **Pile C** puts timing calls inside the head's per-layer path. Even gated, it
  restructures `Qwen35MTPDecoderLayer.callAsFunction` into a `guard`-plus-duplicate
  form so the untraced path stays a straight line. That duplication is exactly the
  kind of thing that rots. It earned its keep for one experiment; it should not
  become permanent furniture in a vendored model file.

If the advisor wants only the *smallest* useful thing, pile A alone is a strictly
better base for the next student than what exists today, and it is the only piece
whose disabled cost I can argue down to zero.

#### 9a.5 What promoting pile A would still not fix

Honesty about the limits, so nobody over-reads this section:

- It does **not** make the branch submittable. Piles B and C remain on the scored
  path in this candidate, and §9's "must not be officially submitted as-is" stands
  for the branch as it is.
- It does **not** validate the trace numbers. Pile A only moves bytes; every timing
  claim in this report rests on piles B and C, which I am *not* proposing.
- I have **not** run a paired measurement of base-plus-pile-A versus base to
  demonstrate the zero-cost claim empirically. The argument in §9a.3 is a source
  argument, not a measurement. Given that a disabled build executes zero added
  instructions, a paired run would be measuring noise — but the advisor should know
  the claim's evidence is static analysis, not a stopwatch.

## 10. Suggested follow-ups (not implemented)

0. **Measure the base's `crossrow` quantized-matmul kernels at M = 2..9 — then decide
   whether there is a prize at all** (§5B, §5C.2). *Rewritten in r3: this used to be
   "widen the `qmv` weight-stream amortization" with a 2.50× prize attached. Both the
   prize and the target were wrong.*

   - **Why it is now a measurement, not an optimization.** Every multi-row efficiency
     figure in this report comes from `mx.quantized_matmul`, which takes the upstream
     kernel. The base ships a `crossrow` family that covers **exactly M = 2..9** and that
     the Python binding never reaches. So "verify matmuls run at 33–35% of roofline" is a
     statement about upstream, and the scored efficiency at M = 2..9 is simply **unknown**.
     Until someone measures it, no prize can be sized and no kernel work can be justified.
   - **Smallest decisive test.** Drive the crossrow path from Swift at the three scored
     shapes (5120→17408, 17408→5120, 5120→248320), affine 4-bit g64, M = 1..9, and
     tabulate achieved GB/s against the same roofline §5B used. Diff it against the §5B
     upstream table — that table is now exactly the right control for this. Rebuilding
     the §5A floor on the result also settles the open 12% / 6.3% discrepancy in §6c.3
     and the reinterpreted scheduling gap in §5C.4.
   - **Only then**, if a real gap remains: the editable surface is the kernel bodies
     (`kernels/quantized{,_nax}.{metal,h}`, `kernels/quantized_utils.h`) plus their
     runtime-effective `mlx-generated/*.cpp` twins. The host dispatcher
     `backend/metal/quantized.cpp` is **not** submittable, so the qmv/qmm selection is
     fixed. Any such work touches reduction order and packing and therefore needs
     exact-row numerical checks, not just an argmax match.
   - **Arch note, corrected.** At D=5120/O=17408 `get_qmv_batch_limit` returns **6** for
     `arch_gen ∈ {13,14}`, **12** for an Ultra (`arch_size == 'd'`), and **10** otherwise
     — *not* 8, which was the r1/r2 error. This M4 Pro and the ranked M5 both fall in the
     "10" case unless M5 reports an Ultra or a 13/14 generation, so M ≤ 9 stays on `qmv`
     on both and the crossrow gate applies on both. Confirm the M5 arch string anyway
     before relying on it.

0b. ~~**Set `segmentedVerifyDepthCap` from the measured qmv batch limit.**~~
   **WITHDRAWN in r3.** It was premised on a dispatch cliff at M = 9 that does not exist:
   the limit is 10, so `qmv` covers M ≤ 9 and the cap at depth 8 already sits on the cheap
   side. There is nothing to move.

   What is left is the *unexplained* round-level observation (23.79 ms/token at d=7 vs
   24.49 ms/token at d=8, §6c.4) with no mechanism behind it and three serious confounds
   (N=4 vs N=9, adaptive depth selection, differing acceptance). If anyone wants to settle
   it, the cheap test is a **matched fixed-depth A/B** using the `MLX_QWEN_MTP_FIXED_DEPTH`
   override — one variable, one run pair. I am not forecasting a win from it, and the
   **+0.084 score** figure I attached to 0b in r1/r2 is withdrawn along with the rest.

0c. ~~**Correct the stale design comment.**~~ **WITHDRAWN in r3 — the comment is correct.**
   `Qwen36MTPBlockSession.swift` states the host qmv batch limit is "10+ on this generation
   for these shapes". It is 10 on `applegpu_g16s` and 12 on Ultra parts, so "10+" is
   accurate for the generations it claims to cover, and the accompanying claim that
   projections at M ∈ 6..9 ride the QMV dispatch is exactly right. My r1/r2 accusation
   came from a wrong constant in my own tooling. **No change to the comment is needed.**
   (The rest of that block was independently verified in §6c.4 and also holds: the
   6..9-row chunking applies at the sdpa only, and segmenting the whole forward was
   already measured and rejected for paying a second full weight pass.)

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
