# E173 — What is inside `F`, the M-independent per-round fixed cost

`harness=local` everywhere in this document. The target is `F_local = 11.792 ms`
per round (FINDING 404 fixed term, M4 Pro class). The ranked `F = 6.899 ms` is
quoted only as a reference point. **No ranked transfer is claimed.** Ranked
pricing through LAW 377 is the advisor's job at review.

- PR: #172, branch `qwen-alphonse/e173-decompose-f`, revision `r0`
- Base: `589793323fca5d1570870b7a044355944f959fa9`
- Host: Apple M4 Pro, `Mac16,11`, `applegpu_g16s`, 14 cores, 48 GB,
  macOS 26.5.2, Swift 6.3.3
- W&B run: https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/drncx8x0
  (state `finished`; carries every table and summary scalar in this document).
  An earlier attempt, run `reikzt54`, is in state `failed`: the logging script
  raised a `KeyError` on a `tablepays` field before it wrote any table. It holds
  no evidence and I disclose it only so the run list is complete.
- Submitted surface: unchanged. `senpai/check-editable-budget.sh` growth
  `0/262144`; every new file is under `Tests/` or `research/`.
  `senpai/verify-ranked-score-boundary.sh` passes.

## 0. Build configuration is a first-order confound

The scored worker is a release build. A debug `swift test` inflates host costs
by up to **6×** on this path:

| entry-point host build, per round | debug | release | ratio |
|---|---|---|---|
| `m = 1` (not routed) | 0.968 ms | 0.108 ms | 9.0 |
| `m = 3` (routed, no table) | 2.930 ms | 0.526 ms | 5.6 |
| `m = 4` (routed, table path) | 8.224 ms | 1.376 ms | 6.0 |

Every number below is release unless labelled otherwise. The debug pass is kept
only as this ratio check. Two earlier debug-era figures of mine are **withdrawn**:
the "≈0.5 µs of MLX metadata queries before the shape guards" (release: 42 ns)
and the `Qwen35XSumsSidecar.take` cost of 0.24–0.29 ms/round (release:
0.046 ms/round).

## 1. Call census of the admission path (deliverable 1)

`Qwen35CustomQMV.routable`, `Qwen35.swift:1770-1794`, over the 257 routed cells
of one weight pass. Host only: the instrument never calls `eval`, allocates no
device buffer, holds no model, and needs no thermal gate.

| m | entries | accepted | refused | refusing guard | host ns/cell | ms/round |
|---|---|---|---|---|---|---|
| 1 | 257 | 0 | 257 | `widths.contains(m)` | 42.8 | 0.011 |
| 2 | 257 | 257 | 0 | none | 179.0 | 0.046 |
| 3 | 257 | 257 | 0 | none | 179.5 | 0.046 |
| 4 | 257 | 257 | 0 | none | 179.0 | 0.046 |
| 5 | 257 | 257 | 0 | none | 178.3 | 0.046 |
| 9 | 257 | 257 | 0 | none | 180.4 | 0.046 |
| 512 (prefill) | 257 | 0 | 257 | `widths.contains(m)` | 44.5 | 0.011 |

**A drafting round makes 257 entries, takes 257 accepts and has zero refusals.**
Every routed cell satisfies `k % 512 == 0`, `n % 8 == 0` and `n >= 4096`, so
`widths.contains(m)` is the only live gate. It refuses all 257 cells at `m = 1`
and at prefill.

**The admission path costs 0.046 ms/round, well below the 0.1 ms line.** Per the
stop rule I report that plainly: an audit of admission-path overhead has no
material target here.

The 257 cells: `gdn.in_proj` 5120→16480 ×48, `gdn.out_proj` 6144→5120 ×48,
`fa.qkv` 5120→14336 ×16, `fa.o_proj` 6144→5120 ×16, `mlp.gate_up` 5120→34816
×64, `mlp.down` 17408→5120 ×64, `lm_head` 5120→248320 ×1.

Guard-order counterfactuals (cells the target never dispatches):

| case | k | n | bits | group | host ns/call |
|---|---|---|---|---|---|
| `bits8_first_guard` | 5120 | 16480 | 8 | 64 | 1.1 |
| `group32_first_guard` | 5120 | 16480 | 4 | 32 | 1.0 |
| `n1024_narrow_kv` | 5120 | 1024 | 4 | 64 | 41.9 |
| `n2048_head_kv_pack` | 5120 | 2048 | 4 | 64 | 41.9 |

### Thread 1 of the brief is refuted

No `n = 1024` quantized cell is dispatched in the target verify forward. The 16
full-attention layers use the fused pack `n = 14336` (`Qwen35.swift:3098`, built
at `:3112-3128`); the unfused `k_proj`/`v_proj` at `n = 1024` exist only as
source weights for that pack, and the `n = 2048` `_kvW` pack is proposal-head
only. The exact-QKV-rows precision island is head-only: `installExactQKVRows`
has exactly one caller (`Qwen35.swift:5393`) bound to `mtp?.layers.first`, and
the checkpoint declares `mtp_num_hidden_layers: 1`, so it covers one attention
module and never the 16 target layers.

## 2. Inventory of `F` (deliverable 4)

`F_local = 11.792 ms`. Explained **8.368 ms (71 %)**; residual **3.424 ms (29 %)**.
Split: **host 2.971 ms, GPU 4.168 ms, mixed 1.229 ms** (deliverable 2).

Each GPU row is the intercept of a linear fit of its per-round cost over
`m ∈ {1, 3, 4, 5}`; only the intercept is M-independent and therefore
`F`-eligible. The slopes are the `pass·groups(M)` and `h·d` part of the round law
and are excluded, and recorded separately in `f-inventory.json`.

| ms/round | side | item | method |
|---|---|---|---|
| 2.087 | GPU | `gdn.recurrence` — 48 FP32 recurrent states `[1,48,128,128]`, read and written every round: 302 MB/round at ~147 GB/s | measured (E173 GPU arm, intercept) |
| 1.284 | host | `admission.kernel_record_construction` — `MLXFastKernel.callAsFunction` per routed cell; twice per cell at `m ≥ 4` | measured (E173 admission arm, residual of the entry-point build) |
| 1.229 | mixed | `head_chain_fixed` | prior-ledger (FINDING 363) |
| 0.655 | host | `post_eval_host_tail` | prior-ledger (FINDING 363, 635–675 µs) |
| 0.650 | GPU | `gpu.xsums_standalone_fills` — 130 cells with no publishing producer each launch their own fill | source-derived census × in-source measured 4–6 µs (`Qwen35.swift:1732`, `:2362`) |
| 0.587 | GPU | `envelope.rms_norm_5120` — 127 row-local norms at 4.65–4.79 µs each; at 92 KB that is 4–21 GB/s, i.e. the per-dispatch floor of this host, not arithmetic | measured (E173 GPU arm, intercept) |
| 0.433 | host | `commit` | prior-ledger (FINDING 405) |
| 0.427 | GPU | `fa.sdpa` intercept at cache length 768 | measured (E173 GPU arm, intercept) |
| 0.357 | host | `protocol_gap` | prior-ledger (FINDING 399 floor) |
| 0.284 | GPU | `mlp.swiglu_elementwise` intercept | measured (E173 GPU arm, intercept) |
| 0.150 | host | 11 command-buffer submits | prior-ledger (E80 census) |
| 0.089 | GPU | `head.island_kv_matmul` — BF16 `[2048,5120]` pack, 21 MB/round | measured (E173 GPU arm, intercept) |
| 0.046 | host | `admission.xsums_sidecar_take` | measured (E173 admission arm, miss branch) |
| 0.046 | host | `admission.routable_predicate` | measured (E173 admission arm) |
| 0.036 | GPU | `head.island_q_matmul_scatter` | measured (E173 GPU arm, intercept) |
| 0.008 | GPU | `head.top_two_readout` intercept | measured (E173 GPU arm, intercept) |
| 0.000 | host | `admission.counter_increments` — below the resolution of the instrument in release | measured (E173 admission arm) |
| **3.424** | — | **residual** | — |

Out of scope but measured, so recorded here: each **rejected draft** costs
**+0.70 to +0.80 ms** on `round_us` and **+0.48 ms** on host phases (t = 8.3).
That is the rollback and repair path, not `F`.

### The residual, honestly

3.424 ms is above the 1.5 ms line, so by the stop rule **the instrument is not
seeing all of `F`**. Named gaps, none of them measured here:

1. GPU work I did not price: the 32 KV-cache slice-update writes, the `M ≤ 2`
   conv1d and GDN prework path, the embedding lookup, the rollback tape, and
   the boundary-fused norm epilogue itself.
2. The per-draft `.item()` fan-out (FINDING 363 suspect C), still unpriced.
3. Gaps between the 11 command buffers, which no per-kernel measurement can see.
4. My GPU intercepts come from isolated kernels that do not contend with the
   weight pass for bandwidth, so they are not guaranteed to be in-round costs.
5. The prior-ledger rows were measured on earlier bases and may have drifted.

### A methodological result worth keeping

The first GPU pass used 8 inner calls per timed region and produced a 21.7 µs
RMSNorm at **0.9 GB/s** — the harness's own submit-and-wait latency, not the
kernel. Raising it to 128 inner calls gives 4.7 µs and 4–21 GB/s. Both runs are
logged (`gpu/line_items_inner8_floor` and `gpu/line_items_amortised`). **An
isolated per-call GPU microbenchmark cannot price an item inside `F` until the
submit floor is amortised**, and the sum of the un-amortised per-call costs
(≈15 ms) exceeds `F_local` on its own, which is how the error announces itself.

## 3. The `tablePays` natural experiment (advisor item F1), and F4

Two independent observational datasets, both `harness=local`, both
`gate_qualified_for_timing=false` and `trace_perturbs_timing=true`, so
within-run differentials only. The estimator is curvature,
`round(m+1) − 2·round(m) + round(m−1)`, which cancels a depth-independent trace
tax exactly.

| dataset | n | curvature(3) | curvature(4) | contrast (3 minus 4) |
|---|---|---|---|---|
| e37 traces, in repo | 485 | 5.94 ms [3.69, 6.85] | 4.22 ms [3.52, 4.78] | **+1.72 ms [−0.80, +3.21]** |
| E168 artifact, adaptive arm | 2356 | 5.05 ms [2.86, 5.77] | 5.44 ms [4.96, 6.55] | **−0.39 ms [−3.67, +0.77]** |
| E168 artifact, all arms pooled | 3209 | 7.33 ms [7.09, 7.52] | 4.31 ms [4.03, 4.59] | +3.01 ms [+2.55, +3.44] — confounded |

**Both identified estimates are null.** The pooled E168 estimate is not usable:
its `m = 3` stratum is dominated by the `p2` fixed-depth arm while `m = 4` and
`m = 5` come only from the adaptive arm, so any level difference between arms
enters the second difference directly.

**The E168 fixed-depth arms cannot give an exogenous-`m` version of this test.**
They impose depth 2 (`m = 3`, 840 single-pass rounds) and depth 7 (`m = 8`,
1723 rounds) only. At one weight pass they contribute `m = 2`: 4, `m = 4`: 6,
`m = 5`: 3 rounds, and those are truncated drafts rather than imposed widths.
The curvature triple does not exist in that artifact.

**The release measurement settles the F3 reconciliation candidate.** The host
graph-build step at the `tablePays` boundary is `1.376 − 0.526 = 0.850 ms/round`,
not the 5.3 ms the debug pass suggested. That is inside both confidence
intervals above, so it cannot explain FINDING 415's +1.43 / +6.94 asymmetry. If
that asymmetry is real it is not host admission work.

For context on what the boundary buys, the E120 rung-5d grid quoted in
`Qwen35.swift:1735-1748` sums to **3.18 ms/round of net saving at `m = 4`**
across the seven shapes, against the 0.850 ms/round of extra host build measured
here.

## 4. Line items at or above 0.5 ms/round, with the smallest change (stop rule)

I did not implement any of these.

1. **`admission.kernel_record_construction`, 1.284 ms/round, host.** Every
   routed cell builds a fresh `mlx_fast_metal_kernel_config`, adds template
   args, grid, thread group and output args, builds an input `vector_array`, and
   calls `mlx_fast_metal_kernel_apply`
   (`Vendor/mlx-swift/Source/MLX/MLXFastKernel.swift:98-257`). At `m ≥ 4` it runs
   twice per cell because a sidecar miss adds the standalone `xsumsTable` fill.
   - Smallest change (a): extend the chunk-sum epilogue to the producers of the
     other 130 activations. Today only `qwen35FusedResidualRMSNorm` publishes
     (`Qwen35.swift:2362`), which covers the 127 fused-norm outputs consumed by
     `gdn.in_proj` (47 of 48), `fa.qkv` (16) and `mlp.gate_up` (64). The
     remaining consumers — `gdn.out_proj` (48, gated RMSNorm output),
     `mlp.down` (64, SwiGLU output), `fa.o_proj` (16, sigmoid-gated SDPA output),
     `lm_head` (1, final norm) and layer 0's `gdn.in_proj` — read activations no
     publisher produces. Adding the same epilogue to those producers removes
     about 130 kernel applies and 130 fill dispatches per round: roughly
     **0.43 ms host plus 0.65 ms GPU**. It is the mechanism already shipped for
     the residual norm, so the exactness argument is the existing one.
   - Smallest change (b): cache the immutable part of the kernel config per
     `(kernel, m, n)` instead of rebuilding it per call. This is an
     input-independent shape table and reuse within one request, which the
     compliance boundary allows; it must hold no prompt-derived state and must
     not persist across requests.
2. **`gdn.recurrence`, 2.087 ms/round, GPU.** 48 FP32 states of
   `[1, 48, 128, 128]`, read and written once per round: 302 MB/round at about
   147 GB/s. The traffic is M-independent by construction, which is exactly the
   shape of `F`, and it is the largest single item in the inventory.
   - Smallest change: hold the recurrent state in BF16 or FP16 and halve the
     traffic, worth about 1 ms/round. This changes recurrent numerics, so it
     needs a focused numerical gate with a positive control at the touched cells
     plus a full exact-token and row-ledger run — not an argmax check.
3. **`gpu.xsums_standalone_fills`, 0.650 ms/round, GPU.** Same fix as 1(a); the
   two items share one mechanism, so a single change closes both.
4. **`envelope.rms_norm_5120`, 0.587 ms/round, GPU.** 127 dispatches at
   4.65–4.79 µs each while moving at most 92 KB. This is the host's
   per-dispatch floor, not arithmetic.
   - Smallest change: fuse further along the layer boundary so fewer separate
     norm dispatches exist. Boundary fusion already removed the entry-norm add;
     the remaining 64 post-attention norms and 63 entry norms are the target.
5. **`post_eval_host_tail`, 0.655 ms/round, and `head_chain_fixed`, 1.229 ms/round**
   are prior-ledger rows carried for completeness. I did not re-measure them, so
   I propose no change.

Below the bar, recorded as inventory lines only: `admission.xsums_sidecar_take`
(0.046 ms/round; an early exit on hit is the obvious shape, but `take` already
costs almost nothing in release), the refusal-ordering cost (0.011 ms/round, and
it only pays at `m = 1` and prefill), and `admission.counter_increments`, which
is below the resolution of the instrument. That last point matters for the
recorded 592 µs/round instrumentation tax: **the counter pair is not the
mechanism.**

## 5. Reproduction

```bash
tools/build-mlx-metallib.sh --all-build-roots
swift build -c release --build-tests -Xswiftc -enable-testing --force-resolved-versions
tools/build-mlx-metallib.sh --all-build-roots      # publish into the release bundle

E173_RELEASE=1 bash research/e173_run_instruments.sh admission
E173_RELEASE=1 MLXFAST_E173_GPU_INNER=128 MLXFAST_E173_GPU_REPS=11 \
  bash research/e173_run_instruments.sh gpu

python3 research/e173_admission_summary.py
python3 research/e173_tablepays_natural_experiment.py
python3 research/e173_trace_phase_split.py
python3 -c "import wandb; wandb.Api().artifact(
  'wandb-applied-ai-team/qwen38-mlx-challenge-senpai/e168-round-cost-pairs:latest'
  ).download('/tmp/e168-pairs')"
python3 research/e173_e168_curvature.py
python3 research/e173_f_inventory.py
python3 research/e173_log_wandb.py
```

The first `tools/build-mlx-metallib.sh` call is mandatory in a fresh workspace:
no `mlx.metallib` existed here, and every MLX-touching test failed at its first
`MLXArray` with "Failed to load the default metallib".

## 6. Cost and provenance

No model was loaded and no benchmark leg ran. The admission arm evaluates
nothing. The GPU arm runs small real-geometry kernels for about 26 s total; it
is not gate qualified (`cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false`) and makes no absolute leg claim. Total GPU
time across the experiment is under one minute.

## 7. Follow-ups I did not implement

1. Apply `research/e90-artifacts/gpu-interval-ledger.patch` on this branch to
   read the command-buffer intervals directly. That is the instrument that can
   close the 3.424 ms residual, because it sees the gaps between command buffers
   that no per-kernel measurement can reach.
2. Price the per-draft `.item()` fan-out (FINDING 363 suspect C) and the 32
   KV-cache slice-update writes.
3. Repair the citation ledger in `research/verify_forward_dispatch_inventory.py`:
   41 of 48 citations have drifted against the current sources and `--selftest`
   is 37/38 because of it. The dispatch model is unverified against live source
   until that is fixed.
4. Log `xs_hit` and `xs_fill` from a real leg. The trace already emits them
   (`Qwen36MTPBlockSession.swift:1889-1890`), but no in-repo artifact carries
   them, so my 127-hit / 130-miss split is a source-derived census rather than a
   measurement.
5. Re-measure `post_eval_host_tail`, `commit`, `protocol_gap` and
   `head_chain_fixed` on the current base, so the inventory stops mixing bases.
