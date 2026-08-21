# E91 — the untouched prefill block

Student `qwen-askeladd`, PR #93, branch `qwen-askeladd/e91-prefill-block`.

`BASE_SHA` `853d9853306018d25e0d51eedbc738e1eb3182fe`. Candidate commit `b09ffea`.
Host `mac-mini.local`, Apple M4 Pro, `applegpu_g16s`, 20 GPU cores, 48 GiB,
macOS 26.5.2, Swift 6.3.3,
`metallib_source_fingerprint=7ae5c5a3d8fabe72ee19bfc09dd737281338a6be658deca49ba97eefdbe3611c`.
Every number below is `harness=local`, `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false`, `official_or_ranked_score=false`.

**Verdict: not useful. The prefill block is closed on this hardware.**

The 512-token seed prefill costs 4043 ms and is 90 % one vendor quantized GEMM
kernel. On the ranked M5 that kernel is a different kernel that no host in this
campaign can execute. Everything the candidate can both change and measure
locally is worth at most 0.03 % of the candidate leg.

## Sessions

| tag | profile | W&B run | job | exit | wall |
| --- | --- | --- | --- | ---: | ---: |
| `e91-ceiling-1` | rung 2 ceiling probe | `04ezvrle` | `747815f8` | 0 | 5 min |
| `e91-smoke-1` | rung 1 smoke, 2 arms | `klbr3rx1` | `3bdd7ddd` | 0 | 197 s |
| `e91-census-1` | rung 0 dispatch census | `2cljbnbk` | `9e1e5edc` | 0 | 51 s |
| `e91-ladder-1` | rung 1 full ladder | `ggrlh2xt` | `8e489996` | 0 | 13 min |

Reproduction, as the sessions were run at commit `3c98a90`:

```bash
swift build -c release --force-resolved-versions -Xswiftc -enable-testing --build-tests
research/e91_ladder.sh e91-census-1 census
research/e91_ladder.sh e91-ladder-1 ladder
research/e91_ladder.sh e91-ceiling-1 ceiling
python3 research/e91_report.py research/out/e91-ladder-1 research/out/e91-ceiling-1
```

The `ladder` profile and its stride knob are deleted at merge; see "Files
changed". Replaying the rung-1 arms needs commit `3c98a90`. The `census` and
`ceiling` profiles still run on the merged tree.

## Rung 0 — the live dispatch census

The earlier census reported `<unmapped>` for all 2265 dispatches because E83
installed its selector swizzle inside `censusBoundaries()`, after warmup had
already built every Metal pipeline. `MLXFAST_E91_REPS=0` now selects a
census-only session that installs the swizzle before the model loads and runs
no timed arm, so the perturbation can never reach a clock. The ledger also
records grid and threadgroup per kernel.

Every quantized GEMM in one 512-row prefill, measured in situ:

| launches | grid | threadgroup | kernel | N | projections |
| ---: | --- | --- | --- | ---: | --- |
| 128 | `544,16,1` | `32,2,2` | `affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0` | 17408 | `mlp.gate_proj` 64 + `mlp.up_proj` 64 |
| 128 | `160,16,1` | `32,2,2` | `affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0` | 5120 | `mlp.down_proj` 64 + `gdn.out_proj` 48 + `fa.o_proj` 16 |
| 48 | `320,16,1` | `32,2,2` | `affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0` | 10240 | `gdn.in_proj_qkv` |
| 48 | `192,16,1` | `32,2,2` | `affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0` | 6144 | `gdn.in_proj_z` |
| 16 | `448,16,1` | `32,2,2` | `affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0` | 14336 | `fa.qkv_packed` |
| 96 | `2,16,16` | `32,2,2` | `affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_false` | 48 | `gdn.in_proj_a` 48 + `gdn.in_proj_b` 48 |
| 1 | `1,31040,1` | `32,2,1` | `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` | 248320 | tail-row `lm_head`, one row |
| 1 | `1310720,1,1` | `1024,1,1` | `affine_dequantize_bfloat16_t_gs_64_b_4` | — | one-off |

464 quantized matmuls total. Every grid `x` is `ceil(N/32)` and every grid `y`
is `ceil(512/32) = 16`: the 32x32-tile signature. Non-GEMM families present:
`custom_kernel_gated_delta_step` 48, `depthwise_conv_1d` 48,
`block_softmax_precise` 16, `rope` 32, `custom_kernel_qwen35_fused_residual_rms_norm`
127, `rms` 176, `steel_gemm_fused_{nn,nt}` 16 each,
`custom_kernel_qwen_mtp_linear_top2_{partial,finalize}` 1 each.

### The NAX gate

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:697`

```cpp
if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
    (env::enable_tf32() || x.dtype() != float32)) {
  return qmm_nax(...);
}
```

`device.cpp:913-932`: `is_nax_available()` needs macOS >= 26.2 and
`gen >= (arch == 'p' ? 18 : 17)`. `device.cpp:556-573` parses `arch_gen_` from
the two digits at `size-3, size-2`. Local `applegpu_g16s` gives gen 16 against a
threshold of 17, so the gate is false here and true on the M5. `qmm_nax`
(`quantized.cpp:473-560`) uses `bm=bn=bk=64, wm=wn=2`, `group_dims(32,2,2)` and
`grid_dims(ceil(N/64), ceil(M/64), B)`, so on ranked the grid is
`(ceil(N/64), 8, 1)` and the name is
`affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0`.
All prefill K values (5120, 6144, 17408) and all big N values are divisible by
64, so all 368 launches cross.

### The split-k correction, and a correction to E70

`quantized.cpp:1415-1425`: `vector_limit = get_qmv_batch_limit(K, N, d)` returns
at most 32, so M=512 can never take a vector kernel, and `transpose_ && B == 1`
always enters `qmm_splitk` (`:776-811`). There `bm = bn = 32`,
`m_tiles = ceil(512/32) = 16`, `split_k = max(1, 512 / (n_tiles * m_tiles))`.
`split_k >= 2` needs `n_tiles <= 16`, that is **N <= 512**. Every larger N gets
`split_k == 1` and hits `if (split_k <= 1) return qmm(...)`.

Only N=48 survives: `n_tiles = 2`, `split_k = 16`, `k_align = 64`,
`min(16, 5120/64) = 16`, `5120 % 1024 == 0`. The census confirms 96 launches at
grid `2,16,16`. **`qmm_splitk` contains no NAX check**, so those 96 launches run
the identical kernel on M4 and M5.

**`research/e70-results.md` section 2.2 is a decode-width table and inverts at
prefill width.** It records `mlp.down` (K=17408 N=5120) and
`gdn.out_proj`/`fa.o_proj` (K=6144 N=5120) as staying on `affine_qmm_t_splitk`
on both arms. That holds at M=10/12, where `m_tiles == 1`; section 3's note that
`mlp.down` walks its split count 3 -> 2 only happens there. At M=512 those same
shapes take plain `qmm`, as the live census shows: 128 launches at grid
`160,16,1`, zero `splitk` at N=5120. So at prefill width **368 of 464 quantized
matmuls cross to `_nax` on ranked**, including the 31.5 % of prefill GEMM time
E70 recorded as non-crossing.

`research/e53_repricing.md:59` calling `qmm_t_splitk` dead code is also stale: it
runs 96 times per prefill.

Share of the N=48 pair:

| measure | value |
| --- | ---: |
| prefill GEMM FLOPs | 24.2 of 24 935 GFLOP = **0.097 %** |
| isolated cell time (rung 2) | 84.9 ms of 4026 ms = 2.11 % |
| E83 measured in-situ ablation (`gdn_in_ba`) | **2.6 ms** = 0.064 % of prefill |
| share of the candidate leg, in situ | **0.0055 %** |

The isolated figure over-states by 33x, exactly as thorfinn's adopted rule
predicts for a cell that does not saturate the GPU.

### What the ladder knob does to the command stream

| arm | forced eval points | dispatches | commits | commits during graph build |
| --- | ---: | ---: | ---: | ---: |
| `ship` (stride 3) | 22 | 2265 | 97 | 94 |
| `off` | 0 | 2265 | 81 | 0 |

Identical dispatch counts: direct evidence that `asyncEval` moves an enqueue
boundary and never an op.

## Rung 1 — the ladder sweep

`e91-ladder-1`: 9 schedules, ABBA quads `[ship, A, A, ship]`, 3 reps, 108 timed
blocks, one resident model, 13 min. Entry GPU temperature 47.8–56.5 C
(spread 8.7 C, median 55.4). Ungated by design, counterbalanced within the
session, and flat: the first, middle and last thirds have medians 4043.83,
4044.74 and 4043.33 ms while entry temperature rose 54.2 -> 55.9 C.

**Bit-exact across every arm.** One tail-row fingerprint,
`271|0x1.5p+4,0x1.f6p+3` (top-2 = 21.0, 15.6875), shared by
`off, s1, s2, s4, s6, s8, s12, s16, ship`. Matches E83.

### Absolute `begin()` wall time

| arm | rungs | n | median ms | min | max | host CPU ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ship` | 22 | 60 | 4043.33 | 4028.39 | 4062.17 | 94.7 |
| `s1` | 64 | 6 | 4043.18 | 4034.89 | 4062.34 | 161.9 |
| `s2` | 33 | 6 | 4044.34 | 4035.19 | 4052.33 | 97.6 |
| `s4` | 17 | 6 | 4047.67 | 4036.74 | 4049.87 | 100.1 |
| `s6` | 11 | 6 | 4038.16 | 4037.08 | 4047.61 | 93.2 |
| `s8` | 9 | 6 | 4042.60 | 4040.26 | 4048.46 | 98.4 |
| `s12` | 6 | 6 | 4049.81 | 4040.12 | 4062.68 | 99.4 |
| `s16` | 5 | 6 | 4044.06 | 4040.83 | 4045.39 | 94.2 |
| `off` | 0 | 6 | 4046.91 | 4043.17 | 4055.99 | 94.8 |

Pooled ship reference 4043.33 ms, n=60. E83 measured 4046.5 ms on the same host
class, so this base reproduces it to **0.08 %**. Single-block standard deviation
over all 108 blocks is **6.44 ms = 0.159 %**.

### ABBA effect against ship

| arm | quads | mean ms | sd ms | mean % |
| --- | ---: | ---: | ---: | ---: |
| `s6` | 3 | **-3.67** | 2.47 | **-0.091** |
| `s16` | 3 | -0.81 | 2.26 | -0.020 |
| `ship_null` | 3 | +0.27 | 6.79 | +0.007 |
| `s8` | 3 | +1.21 | 2.73 | +0.030 |
| `s1` | 3 | +2.17 | 8.84 | +0.054 |
| `s12` | 3 | +3.01 | 13.13 | +0.074 |
| `s2` | 3 | +3.07 | 16.55 | +0.076 |
| `s4` | 3 | +4.92 | 6.27 | +0.122 |
| `off` | 3 | +6.51 | 6.77 | +0.161 |

The best arm is `s6` at -3.67 ms. Its standard error is 2.47 / sqrt(3) = 1.43 ms;
the ship-against-ship null has mean +0.27 and standard error 6.79 / sqrt(3) =
3.92 ms. The difference is 3.94 ms against a combined standard error of 4.17 ms,
that is **0.94 sigma. Not significant.** Stop rule 1 fires: the axis is closed.

### Why the knob cannot matter — the phase split

| arm | rungs | build ms | final eval ms | readback ms | total ms | host CPU ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 0 | 13.4 | 4034.6 | 0.03 | 4046.9 | 94.8 |
| `s1` | 64 | 118.7 | 3925.0 | 0.03 | 4043.2 | 161.9 |
| `s4` | 17 | 2681.9 | 1359.9 | 0.02 | 4047.7 | 100.1 |
| `s6` | 11 | 2839.3 | 1198.7 | 0.03 | 4038.2 | 93.2 |
| `ship` | 22 | 3044.4 | 998.6 | 0.02 | 4043.3 | 94.7 |
| `s2` | 33 | 2747.7 | 1296.3 | 0.02 | 4044.3 | 97.6 |

The knob moves up to 3031 ms of wall time between the build phase and the
blocking eval, a 75 % swing, and **the total is invariant to 0.2 %.** `s1` shows
the host can finish enqueueing the entire 64-layer graph in 118.7 ms while the
GPU needs 3925 ms. Prefill is GPU-throughput-bound with no host component to
recover.

## Rung 2 — the quantized GEMM ceiling (retained, but on the wrong kernel)

Machine peaks measured on this host: streaming read 101.9 GB/s, streaming copy
161.4 GB/s, bf16 GEMM 4096^3 7.447 TFLOP/s.

| family | N | L | ship ms | TFLOP/s | achieved GB/s (tiled) | bf16 ms | TFLOP/s | share | gap % | gap ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp.down_proj` | 5120 | 64 | 14.318 | 6.37 | 255.2 | 12.793 | 7.13 | 22.76 % | +10.65 | 97.6 |
| `mlp.up_proj` | 17408 | 64 | 14.288 | 6.39 | 255.8 | 12.693 | 7.19 | 22.71 % | +11.16 | 102.1 |
| `mlp.gate_proj` | 17408 | 64 | 14.276 | 6.39 | 256.0 | 12.647 | 7.22 | 22.69 % | +11.41 | 104.2 |
| `gdn.in_proj_qkv` | 10240 | 48 | 8.406 | 6.39 | 255.7 | 7.617 | 7.05 | 10.02 % | +9.39 | 37.9 |
| `gdn.out_proj` | 5120 | 48 | 5.556 | 5.80 | 232.1 | 4.698 | 6.86 | 6.62 % | +15.44 | 41.2 |
| `gdn.in_proj_z` | 6144 | 48 | 5.287 | 6.09 | 243.9 | 4.728 | 6.81 | 6.30 % | +10.57 | 26.8 |
| `fa.qkv_packed` | 14336 | 16 | 11.795 | 6.37 | 255.1 | 10.394 | 7.23 | 4.69 % | +11.88 | 22.4 |
| `fa.o_proj` | 5120 | 16 | 5.262 | 6.12 | 245.1 | 4.702 | 6.85 | 2.09 % | +10.64 | 9.0 |
| `gdn.in_proj_a` | 48 | 48 | 0.969 | 0.26 | 13.1 | 0.717 | 0.35 | 1.16 % | +26.01 | 12.1 |
| `gdn.in_proj_b` | 48 | 48 | 0.799 | 0.32 | 15.9 | 0.919 | 0.27 | 0.95 % | -15.03 | -5.8 |

Modelled prefill total 4026.2 ms, which reproduces the measured 4043.3 ms to
0.4 %. The saturating-cell gap to dense bf16 is 441.2 ms = 10.96 % of prefill =
0.94 % of the candidate leg. **That is an unreachable upper bound**: dense bf16
is not a legal candidate because the target weights are fixed affine 4-bit
group-64, and the measurement is on the 32x32-tile kernel the ranked host does
not run.

The arithmetic intensity under zero cache reuse is a constant 25 FLOP/byte
(`tiled_bytes/FLOP = 0.04` exactly) against a machine ridge point of
7.447 TFLOP/s / 226 GB/s = 33 FLOP/byte, so the shipped kernel is
bandwidth-limited at bn=32. NAX's bm=bn=bk=64 halves traffic per FLOP and moves
the cell to 50 FLOP/byte, past the ridge. This is one more reason a local
`affine_qmm_t` measurement cannot be transferred to the ranked
`affine_qmm_t_nax`.

**Rungs 2 and 3 are terminal under revised stop rule 3.** 99.90 % of prefill
GEMM FLOPs execute `affine_qmm_t` here and `affine_qmm_t_nax` on ranked, and no
host in this campaign reports gen >= 17.
`research/e70-results.md:156-165` confirms the `MLX_METAL_GPU_ARCH` probe
"changes kernel selection only … proves which kernel the M5 picks and never how
fast it runs", so a forced-`applegpu_g17s` timing arm is worthless.

## The three questions

### 1. Where does the prefill second go, to the nearest 5 %?

`begin()` = 4043 ms.

| bucket | % of prefill | source |
| --- | ---: | --- |
| MLP quantized GEMM (`gate`, `up`, `down`; 192 launches) | **60 %** | rung 2 shares x GEMM fraction |
| Gated DeltaNet quantized GEMM (`in_proj_qkv`, `in_proj_z`, `out_proj`, `a`, `b`; 240 launches) | **25 %** | same |
| Full-attention quantized GEMM (`qkv_packed`, `o_proj`; 32 launches) | **5 %** | same |
| Everything else: GDN recurrence 48, depthwise conv 48, block softmax 16, rope 32, RMS norm 303, SDPA steel GEMM 32, ~460 copies and elementwise | **10 %** | E83 ablation, non-GEMM <= 260 ms |
| host thread CPU | **0 %** (94.7 ms, fully hidden) | this run, phase split |
| forced eval-point stalls | 0.2 % (8 ms) | ledger (E) |

Two independent methods bracket the GEMM fraction: the isolated cell sum is
99.5 % of the measured wall (an over-estimate, since isolated cells pay no
overlap) and E83's in-situ ablation of the interceptable families is 88.9 %.
Best estimate 90 %.

### 2. How much of it is recoverable on ranked?

Prefill is 8.59 % of the candidate leg, so 1 ms of prefill is 0.00212 % of the
leg. Because the ranked serial numerator comes from the runner-owned baseline
workspace, `d ln(ranked baseline serial time)/dx = 0` and every prefill
millisecond removed is a pure ranked gain.

| lever | best local evidence | share of candidate leg | status |
| --- | ---: | ---: | --- |
| ladder schedule | -3.67 ms, 0.94 sigma | **0.008 %** | measured, closed |
| forced eval-point stalls, all removed | 8 ms | 0.017 % | closed |
| host CPU, all removed | 0 ms (already hidden) | 0 % | measured, closed |
| `affine_qmm_t_splitk` N=48 pair | 2.6 ms in situ | 0.006 % | closed |
| prefill fusion G1 + G2 | -7.1 ms (loses) and G1 breaks bit-exactness | negative | E83 rung 3, closed |
| non-GEMM tail, hypothetically all removed | <= 260 ms | <= 0.55 % | not attacked; no single family above ~0.1 % of the leg |
| the 368 `affine_qmm_t` launches (90 % of prefill) | 441 ms to dense bf16 | 0.94 % | **not measurable here** |

**Answer: at most 0.03 % of the candidate leg is recoverable from prefill by any
mechanism I can both change and price on an M4 Pro.** The 90 % that matters is a
vendor quantized GEMM whose ranked variant this campaign cannot execute.
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp` is inside the
editable surface, so the kernel is editable — it is just unmeasurable locally.
That is the whole finding.

### 3. The smallest next experiment

Not another prefill run. Two options, cheapest first.

**E92a, zero GPU time: a static `applegpu_g17s` register and occupancy census
for `affine_qmm_t_nax`.** E76 already established this technique in this
campaign (`research/e76-results.md:67,121,167` report g17s register counts from
`xcrun metal -arch applegpu_g17s`). Compile the NAX quantized GEMM at
`bm=bn=bk=64, wm=wn=2` for `applegpu_g17s`, read register count and threadgroup
memory, and compute occupancy at the six prefill grids
`(272,8,1), (80,8,1), (160,8,1), (96,8,1), (224,8,1)`. This is a real ranked-host
number obtained without a ranked host, and it costs no GPU time and no
submission. If occupancy is register-limited there is a concrete tile-shape arm;
if it is not, prefill is closed campaign-wide with a receipt. It also transfers
to decode, because E70 shows the NAX crossing opens at M >= 10, so the same
kernel serves every verify width at or above 10.

**E92b, one submission slot: price one `quantized_nax.cpp` change officially.**
The official runner is the only instrument that can time this kernel. Make one
minimal change, prove locally that the non-NAX path and the 512-token exactness
gate are untouched, and read the score. Program.md's "official evaluation is
part of the research loop" clause authorizes this; do it only after E92a says
there is something to find.

## Gates

| gate | command | verdict |
| --- | --- | --- |
| assignment scope | `senpai/validate-assignment-scope.sh 853d9853 Sources/MLXFastModel/Qwen36MTPBlockSession.swift Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` | `assignment scope OK: 2 submitted path(s)` |
| editable budget | `senpai/check-editable-budget.sh 853d9853` | `source=2511128/3000000 headroom=488872 growth=2250/262144 exempt=2410 files=154` |
| ranked score boundary | `senpai/verify-ranked-score-boundary.sh` | `PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only` |
| exactness across arms | 108 timed blocks, 9 schedules | one tail-row top-2 fingerprint `271|0x1.5p+4,0x1.f6p+3` for every arm |
| `swift test` regression | `swift test -c release --force-resolved-versions -Xswiftc -enable-testing` | see below |

### The `swift test` regression gate

`senpai/known-test-failures.md` defines the gate as the **failing name set** and
the **issue count**, never the exit code and never the test total.

| tree | tests | suites | failing names | issues |
| --- | ---: | ---: | ---: | ---: |
| this branch | 718 | 57 | the same 9 | **40** |
| base `853d9853`, filtered to those 9 names | 9 | 3 | the same 9 | **40** |

Both runs exit 1, which the file says carries no information on its own. The
name set is exactly the documented nine
(`contestantDocsCommandBlocksKeepTheDependencyGraphFrozen`,
`participantDocsExposeDefaultCLIInstallDirectory`,
`qwen36ConfigContractDigestMatchesTheReferenceManifest`,
`startupMemoryPolicyKeepsRanked128GiBProfile`,
`submissionStaticReviewPromptCoversMeasurementStructureExploitation`,
`theCheckedInDeclarationSelectsThePinnedHead`,
`theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`,
`theQwenMTPTrackIsArmedOnQwen38`,
`theSeededCalibrationExpectationMatchesItsRecordedProvenance`) and the issue
count is exactly 40 on both trees. **No new failure. The gate passes.** The base
figure was measured directly at `853d9853` in this session, not carried over
from the file's earlier measurement at `f7f356b2`.

The 13 extra tests on this branch are the E91 ladder suite. They all pass.

## Head provenance

The ladder harness loads the target model only. It builds
`Qwen35TextModel` through `Qwen35RuntimeWeightCache` and calls
`callWithHidden` over the 512-token seed with a fresh cache, which is the exact
GPU work inside `Qwen36MTPBlockSession.begin()`. **No proposal head is attached
in any E91 session**, so no `head_provenance_sha256` exists for these runs and
none is quoted. This branch does not change `mtp-head.manifest.json`; it still
declares `sha256`
`559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`,
427742600 bytes, exactly as the base does. `begin()` reaches the head only after
the prefill forward, so the head cannot affect any number in this report.

## One correction for the advisor

Advisor comment 4 says edward's light interval ledger is "merged at
`Sources/MLXFastModel/E90GPUIntervalLedger.swift` on `cedb900b` with a
reproducible patch at `research/e90-artifacts/gpu-interval-ledger.patch`".
**Neither path exists on `853d9853`, the base this assignment names.** They are
on `cedb900b`, a later tree. I therefore could not reuse the ledger and used
E83's command-buffer commit counter instead, in an untimed census block only.
The GPU busy and idle split asked for in rung 0 is consequently **not reported**;
the phase split answers the same question without it, because `s1` proves the
host finishes the whole 64-layer enqueue in 118.7 ms of a 4043 ms block.

## Files changed

The ladder knob that produced the rung-1 arms is **deleted** at the advisor's
request, because the axis it measures is closed. The list below is the merged
end state; the measured knob itself survives only in this report and in the
branch history at commit `3c98a90`.

Candidate surface, one behaviour change and one comment:

- `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` — the `begin()` trace line
  gains `wall_us=` and `cpu_us=` from `CLOCK_THREAD_CPUTIME_ID`, beside the
  `build_us=`, `eval_wall_us=` and `head_submit=` fields already on the base.
  The `threadCPUNanoseconds()` helper is E90's; my duplicate is gone.
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` — comment only. Both
  prefill call sites hold the shipped literal `i == 0 || i % 3 == 2`, so the
  rung set is bit-for-bit what it was before E91.

Research-only, never packaged by Yukon:

- `Tests/MLXFastTests/E91PrefillLadderTests.swift` — the untimed dispatch census
  of one `begin()` and the quantized GEMM ceiling probe. The stride sweep, the
  arm helpers and the stride unit test are deleted with the knob.
- `Tests/MLXFastTests/E83PrefillDecompositionTests.swift` — 15 declarations
  changed from `private` to internal for reuse; the dispatch ledger records grid
  and threadgroup per kernel; the swizzle installer gained the descriptor-based
  pipeline hook and a comment naming the ordering requirement.
- `research/e91_ladder.sh` — profiles `census` and `ceiling`; the `smoke` and
  `ladder` profiles are deleted and the census artifact is now `census.json`.
- `research/e91_report.py` — census and ceiling tables; the ABBA arm tables are
  deleted.
- `research/e83_wandb_stream.py` — a generic flattener for unknown block kinds.
- this file.

**Recommendation: the prefill axis is closed.** The only instrument kept on the
candidate surface is the `cpu_us` trace field, which costs one
`clock_gettime(CLOCK_THREAD_CPUTIME_ID)` per `begin()` under `traceRounds` and
answered a real question: the host burns 94.7 ms of CPU in a 4043 ms block.
