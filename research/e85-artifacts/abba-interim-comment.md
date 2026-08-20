## E85 interim 2: the 512-token ABBA session finished, and it cannot decide your stop rule. The control proves it.

Not a request for direction — I am already running the fix. Posting because the measurement protocol in §6 does not resolve the effect sizes in §7 on this host, and you should know that before you price another assignment this way.

### The session

12 legs, ABBA x 3, base against (a)+(b), 512 decode tokens, declared head, ungated. `all_tokens_matched=true` on all 12. `worker_sha256` `5bbacde1…` identical before and after, so no leg ran a different binary. Entry temperature spread 22.72 C, driven by leg 1 starting cold at 39.2 C while every later leg started near 61 C.

| arm | legs | candidate s/tok mean | sd | serial s/tok mean | local ratio | `effective_mean_draft_len` | accepted rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 6 | 0.03175777 | 0.00011389 | 0.07438016 | 2.3421 | 6.358974359 | 0.877016129 |
| (a)+(b) | 6 | 0.03174120 | 0.00014734 | 0.07434882 | 2.3424 | 6.358974359 | 0.877016129 |

### The contrast, and why it decides nothing

Three drift-free estimators on absolute candidate seconds per token. Negative means the arms are faster.

| estimator | µs/token | 95 % CI | % of candidate | µs/draft | **µs per eliminated buffer** | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|
| ABBA block | −16.56 | [−229.8, +196.7] | −0.0522 % | −17.13 | **−2.45** | [−33.9, +29.1] |
| OLS, leg-order term | −16.56 | [−193.0, +159.9] | −0.0522 % | −17.13 | **−2.45** | [−28.5, +23.6] |
| OLS + serial covariate | −40.33 | [−201.2, +120.6] | −0.1270 % | −41.71 | **−5.96** | [−29.7, +17.8] |

Residual sd is 135 µs per leg, which is 0.43 % of candidate time. Your stop-rule boundaries are 0.11 % (5 µs/buffer) and 0.21 % (10 µs/buffer). The boundaries sit inside one third of a single leg's noise.

**The decisive fact is the control.** Neither arm can reach the depth-0 serial path, so `serial_seconds_per_token` times unchanged code inside the same 12 runs. Its contrast is **−31.34 µs/token, t = −0.546** — larger in magnitude than the −16.56 µs/token I measured on the changed path. An unchanged code path shows a bigger apparent effect than the changed one. Any conclusion from this session would be a reading of host noise.

For completeness against §5: `effective_mean_draft_len` and accepted-draft rate are identical to sixteen digits on every leg of both arms, so there is no acceptance regression. `gather_qmm` did not flip a single near-tie in 512 tokens.

### What the point estimate is worth

The point estimate lands at −2.45 µs per buffer, inside the 1.55–2.71 µs my census predicted, and the covariate estimate at −5.96 µs is still far below 13–16 µs. That agreement is suggestive, not evidence. With a CI of ±25 µs I will not claim it.

Reaching a 27 µs/token standard error, which is what separating 5 µs/buffer from 10 µs/buffer needs, takes about 100 legs at this residual, roughly six GPU hours for one boundary. That is not a good use of the allocation.

### What I am doing instead

**1. Execution proof, running now.** The identical draft counters are what a correct arm (a) must produce, since the fused kernel is bit-identical to the eager path. They are also exactly what a guard that never passes would produce. I will not report a null until those cases are separated. Four census legs from one binary, arm selected by environment, at forced draft widths 1 and 5, with Metal command-buffer timestamps on. The `ab` legs must lose `gather_frontuint32_int32_int_1`, `gather_frontbfloat16_int32_int_2`, `affine_dequantize_bfloat16_t_gs_64_b_4`, `gather_frontuint32_uint32_int_1` and `gather_frontbfloat16_uint32_int_2` from the `draft_head` phase, and the per-draft dispatch slope must fall by about 7 from 34.07. This also prices the head phase in GPU nanoseconds, which splits the per-buffer cost into a GPU part and a host part.

**2. Measure the price by adding buffers instead of removing them.** Removing 7 buffers is unmeasurable here. Adding 128 is not. An `MLX_E85_BUFFER_TAX=K` knob chains K dependent bf16 `[1,1,5120]` adds of an exact zero into the per-draft path: one materialised 10,240 B buffer and one dispatch each, the same write-then-read round trip your law describes, and bit-identical output. A palindromic session over K in {0, 32, 128} gives the slope in µs per materialised intermediate.

At the same 135 µs residual, K = 128 predicts **+1609 to +1980 µs/token (+5.1 % to +6.2 %) if your law holds**, against **+192 to +335 µs/token (+0.6 % to +1.1 %) if my census is right**. Those differ by a factor of eight and both are far above the noise. The slope must also exceed the 0.66–1.55 µs dispatch cost, which is the guard against MLX quietly eliding the tax.

This converts "we could not detect it" into a measured number for the coefficient your law is built on, which is the thing you said you need before pricing more work this way.

### One correction to §7

§7 predicts −0.35 % for (b) and −0.46 % for (a)+(b) from 3 and 4 buffers. My census found 7 buffers, not 4, because `embed_tokens` is quantized and `QuantizedEmbedding.callAsFunction` is three gathers plus a dequantize. At 13–16 µs the prediction for (a)+(b) is therefore −0.29 % to −0.35 % of candidate time, not −0.46 %. It does not change the stop rule, which is stated per buffer.

_This comment was created by an AI agent (OpenHands) on behalf of the qwen-edward research student._
