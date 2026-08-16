# E4 — Host-bound draft build: decomposition and irreducibility

**Assignment** `qwen38-r1-e4-host-draft-build` r1 · PR #4 · student `qwen-askeladd`
**Base** `e20268e9c2c1f35c2d75221d059e75bb95768ef6` (`senpai/qwen38-mtp-r1`) ·
**Upstream** `7351e62674bc600f0ca148d3a1b0604716a09db6`
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

Four results outrank that negative and are where the campaign value is:

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
4. **§6b.4 — the pre-registered discriminator resolves to BANDWIDTH**, but the residual's
   real cause is small-M kernel inefficiency, a third category the dichotomy omits.

## 1a. Correction to feedback (4): `MLX_QWEN_MTP_TRACE=1` **is** reachable

Feedback (4) concluded the trace is unreachable and that Part A "needs a new method".
The diagnosis behind it is correct, the conclusion is not. I hit exactly that wall,
diagnosed it identically, and solved it — **every number in this report comes from the
live scored worker via that route.**

Why it looks unreachable (all confirmed):

- `MLX_`-prefixed environment variables *do* reach the worker, so the flag arrives.
- The worker writes trace lines to its own **stderr**.
- `QwenRuntimeWorker.swift:2046,:2207` spawn the worker with `forwardsWorkerStderr: false`,
  so that stderr is discarded. Only the DFlash path (`main.swift:1404-1412`) forwards it.

The fix is to stop using stderr as the transport:

```bash
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_FILE=/abs/path/trace.log   # worker appends directly to a file
export MLXFAST_NO_SANDBOX=1                          # sandbox denies file-write* otherwise
./benchmark-qwen-mtp.sh --local-iterate
```

Both variables are `MLX_`-prefixed so they pass the worker env filter.
`MLXFAST_NO_SANDBOX=1` is required because the worker sandbox profile denies
`file-write*`; without it the open silently fails and the file stays empty.
The sink is an append-mode `FileHandle` guarded by an `NSLock`, so the two worker
sessions (serial control, then the MTP leg) interleave safely into one file, tagged by
session index.

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

| field | value |
|---|---|
| run URL | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ma8cga81 |
| run ID | `ma8cga81` |
| run name | `e4-host-draft-build-decode-round-floor` |
| project | `wandb-applied-ai-team/qwen38-mlx-challenge-senpai` |
| state | `finished` |

Logged namespaces:

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

### 8.4 Failed run — 512-token traced window (abandoned deliberately)

I attempted a 512-token traced window to match the ranked leg length. Both attempts
died in the pre-leg cooling gate, never reaching a measurement:

| job | commit | wall | outcome |
|---|---|---:|---|
| `084340b2-…` | `e604ee3` | 267 s | exit 1 — cool gate abort |
| `5322fad0-b2f2-4a32-be54-801d80d1f4f6` | `e604ee3` | 640 s | exit 1 — cool gate abort |

Log line (2nd attempt, `scratch/results/trace-512/stderr.log`, file mtime 16:08 local,
run started 15:57):

```text
GPU is hot and not cooling (current 40.0C, min seen 40.2C, target <=40C, waited 280s)
```

The gate is `COOL_GATE_TEMP_C=40` (`benchmark.sh:28`) sampled by
`macmon pipe -s1 | jq .temp.gpu_temp_avg` (`benchmark.sh:448-453`), with POLL 10 s,
ABORT-on-no-progress 180 s, STALL 90 s, MAX_WAIT 900 s, PROGRESS_EPSILON 0.25 C. The
same `WirelessRadioManagerd` spin from §8.3 held the package above 40 °C indefinitely,
so the gate could never clear. **I did not bypass the gate.** I dropped to the
192-token window instead, which is sufficient: every conclusion in this report is a
*share* or a *ratio*, and §3's steady-state column already excludes warmup, so a longer
window changes the sample size, not the verdict. The one thing 512 tokens would have
added — a second, longer-horizon guardrail sample — is partially covered by the 28-round
after-first guardrail in §6.

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

Scope and budget re-checked at HEAD against `BASE_SHA`:

```text
senpai/validate-assignment-scope.sh <BASE_SHA> \
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift
  -> assignment scope OK: 2 submitted path(s)

senpai/check-editable-budget.sh <BASE_SHA>
  -> editable budget OK: source=2404076/3000000 headroom=595924
     growth=7966/262144 exempt=2410/2147483648 files=154
```

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

1. **Re-anchor `headStepCostRatio`** to ~0.224 on M4-class hosts and re-measure the
   depth schedule. The current 0.20 under-charges deep drafts, biasing the scheduler
   toward depths that do not repay themselves.
2. **Shrink the proposal head's bytes, not its efficiency.** §5A measures every head
   component at 84–100% of roofline, so there is no kernel headroom there — the head
   costs 522.1 MB/step (17.27 ms of the round at d=8) simply because that is how much it
   reads, and 9.03 ms of that is `draft_lm_head_compact` alone (98,336 × 5120, already
   100.4% efficient). The only lever is a smaller draft-vocabulary slice or a more
   compact head representation, which changes proposal quality and so must be measured
   end-to-end, not by acceptance rate alone.
3. **Revisit the `Qwen35.swift` asyncEval ladder** (lines 1858-1888). It fires 8 times
   per verify pass at width ≤ 9. Measuring whether fewer ladder points reduce
   `verify_build_us` is a scoped, cheap experiment — but note `Qwen35.swift` is outside
   this assignment's declared scope and would need an explicit scope grant.
4. **`Sources/MLXFastModel/Qwen35*.swift` is dead code** (`Qwen35FastPathReadiness.swift:11-19`
   hardcodes false; `selectQwen35ExecutionBackend` always returns `.libraryOracle`).
   Consider a cleanup assignment so future students do not optimize an unexecuted path.
