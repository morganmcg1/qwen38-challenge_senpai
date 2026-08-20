## E85 interim 2: the ABBA session cannot decide your stop rule, but the execution proof can, and it prices a buffer at about 1.3 µs

Not a request for direction. I am already running the measurement that settles it. Post this now because two facts change how you should price work of this shape.

### 1. The 512-token ABBA session finished and is underpowered

12 legs, ABBA × 3, base against (a)+(b), 512 decode tokens, declared head, ungated. `all_tokens_matched=true` on all 12. `worker_sha256` `5bbacde1…` identical before and after, so no leg ran a different binary. Entry-temperature spread 22.72 C, driven by leg 1 starting cold at 39.2 C while every later leg started near 61 C.

| arm | legs | candidate s/tok mean | sd | serial s/tok mean | local ratio | `effective_mean_draft_len` | accepted rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 6 | 0.03175777 | 0.00011389 | 0.07438016 | 2.3421 | 6.358974359 | 0.877016129 |
| (a)+(b) | 6 | 0.03174120 | 0.00014734 | 0.07434882 | 2.3424 | 6.358974359 | 0.877016129 |

Three drift-free estimators on absolute candidate seconds per token. Negative means the arms are faster. Per-buffer columns divide by **6**, not 7 — see section 2.

| estimator | µs/token | 95 % CI | % of candidate | µs/draft | **µs per eliminated buffer** | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|
| ABBA block | −16.56 | [−229.8, +196.7] | −0.0522 % | −17.13 | **−2.86** | [−39.6, +33.9] |
| OLS, leg-order term | −16.56 | [−193.0, +159.9] | −0.0522 % | −17.13 | **−2.86** | [−33.3, +27.6] |
| OLS + serial covariate | −40.33 | [−201.2, +120.6] | −0.1270 % | −41.71 | **−6.95** | [−34.7, +20.8] |

Residual sd is 135 µs per leg, which is 0.43 % of candidate time. Your stop-rule boundaries are 0.11 % (5 µs/buffer) and 0.21 % (10 µs/buffer). Both boundaries sit inside one third of a single leg's noise.

**The control decides this.** Neither arm can reach the depth-0 serial path, so `serial_seconds_per_token` times unchanged code inside the same 12 runs. Its contrast is **−31.34 µs/token, t = −0.546** — larger in magnitude than the −16.56 µs/token measured on the changed path. An unchanged path shows a bigger apparent effect than the changed one, so any verdict from this session alone would be a reading of host noise. Reaching the 27 µs/token standard error that separates 5 from 10 µs/buffer needs about 100 legs, roughly six GPU hours for one boundary.

Against §5: `effective_mean_draft_len` and accepted-draft rate are identical to sixteen digits on every leg of both arms. There is no acceptance regression, and `gather_qmm` did not flip a single near-tie in 512 tokens.

### 2. The execution proof is complete, and it is decisive

Identical draft counters are what a correct arm (a) must produce, because the fused kernel is bit-identical to the eager path. They are also what a guard that never passes would produce. Four census legs from one binary `b03f0ed9…`, arm selected by environment, forced draft widths 1 and 5, Metal command-buffer timestamps on, `all_tokens_matched=true` on all four.

Both arms execute. In the `draft_head` phase, `ab` loses all five of `gather_frontuint32_int32_int_1`, `gather_frontbfloat16_int32_int_2`, `affine_dequantize_bfloat16_t_gs_64_b_4`, `gather_frontuint32_uint32_int_1` and `gather_frontbfloat16_uint32_int_2`, and also loses `custom_kernel_qwen35_dual_rms_norm_concat_bf16_v1`. It gains `custom_kernel_qwen35_embed_dual_rms_norm_concat_bf16_v1` and `affine_gather_qmv_bfloat16_t_gs_64_b_4`, and `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` falls from 6.0 to 5.0 per head pass.

**The elimination is 6 buffers per draft, not 7.** `gatherQuantizedMM` builds its own `lhsIndices` when the argument is nil, which adds one `arangeuint32` per draft. The per-draft dispatch slope falls 34.0 → 28.0, confirming −6.

Now the part that matters for your law:

| per forced draft, `draft_head` | base | (a)+(b) | Δ |
|---|---:|---:|---:|
| GPU ns | 2292.849 µs | 2285.283 µs | **−7.566 µs** |
| driver ns | 188.024 µs | 188.618 µs | +0.594 µs |
| encode ns | 46.691 µs | 34.448 µs | −12.243 µs (census-inflated) |
| dispatches | 34.0 | 28.0 | −6.0 |
| command buffers | 2.25 | 2.25 | **0** |
| commits | 2.109 | 2.109 | **0** |

**Removing six materialised intermediates removes zero command-buffer commits.** That is the mechanism behind the failure of the law. A commit costs 13.5–17.6 µs on this host, which is exactly the size of your 13–16 µs figure. Your organizer pair `c6af1e24` → `8e83c6b3` almost certainly removed a *commit boundary* along with the buffer, and the per-buffer attribution inherited the commit's price. On the per-draft proposal path the buffers sit inside an existing command buffer, so eliminating them cannot recover that cost.

The GPU-side price is **−7.566 µs / 6 = 1.26 µs per buffer**. Adding a realistic host share puts the total near 1.9–2.8 µs. That agrees with my census prediction of 1.55–2.71 µs and with the ABBA point estimate of −2.86 µs. Three independent routes converge on about 2 µs against a claimed 13–16 µs, a 6–8× overprediction.

### 3. Running now: price a buffer by adding buffers instead of removing them

Removing 6 buffers is unmeasurable on this host. Adding 192 is not. `MLX_E85_BUFFER_TAX=K` chains K dependent bf16 `[1,1,5120]` adds of an exact zero into the per-draft path: one materialised 10,240 B buffer and one dispatch each, the same write-then-read round trip your law describes, and bit-identical output, so drafts and acceptance cannot move. A palindromic session over K ∈ {0, 48, 192}, 12 legs at 512 tokens, gives the slope in µs per materialised intermediate.

At the measured 135 µs residual, K = 192 predicts **+2.41 to +2.97 ms/token (+7.6 % to +9.4 %) if your law holds**, against **+0.29 to +0.50 ms/token (+0.9 % to +1.6 %) if the census is right**. A Monte-Carlo check of the estimators against synthetic sessions at this exact noise level returns the planted coefficient without bias at a standard error of **0.50 µs per buffer**, so the design separates 2 µs from 13–16 µs by more than twenty standard errors. One tax unit is one buffer plus one dispatch, so the slope is an upper bound on the buffer price; a slope at or under the 0.66–1.55 µs E80 dispatch cost would also mean MLX elided the tax, and I would settle that with a census leg before making any claim.

This converts "we could not detect it" into a measured value for the coefficient your law rests on.

### 4. One correction to §7

§7 predicts −0.35 % for (b) and −0.46 % for (a)+(b) from 3 and 4 buffers. The census found 7 targeted intermediates, not 4, because `embed_tokens` is quantized and `QuantizedEmbedding.callAsFunction` is three gathers plus a dequantize; the net elimination is then 6, because of the `arangeuint32` above. At 13–16 µs the prediction for (a)+(b) is −0.24 % to −0.29 % of candidate time, not −0.46 %. This does not change the stop rule, which is stated per buffer.

_This comment was created by an AI agent (OpenHands) on behalf of the qwen-edward research student._
