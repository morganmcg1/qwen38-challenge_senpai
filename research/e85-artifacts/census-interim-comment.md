# E85 target (c): materialised-intermediate census on the per-draft proposal-head path — COMPLETE

Posting this before I build arm (a) or arm (b), as the assignment requires. The census contradicts the stated campaign law in one important way and confirms it in another.

## Method

I restored the E80 dispatch census and added a device-allocation hook to it (`MLX_E85_ALLOC_CENSUS=1`). The hook swizzles `newBufferWithLength:options:` and `newBufferWithBytes:length:options:` on `AGXG16SDevice`, so it counts every real Metal buffer allocation, not MLX-level array creation.

Host-thread phase brackets smear because MLX encodes asynchronously. I therefore pinned the draft width with `MLX_E80_FORCE_DRAFTS` and regressed **round totals** against width. Two legs, declared head, 128 decode tokens, `--local-iterate`:

| leg | forced drafts | head passes per round | rounds | `all_tokens_matched` |
|---|---|---|---|---|
| `research/out/e85-census-d1` | 1 | 1.00 | 64 | true |
| `research/out/e85-census-d5` | 5 | 4.75 | 23 | true |

These are census builds. Their timings are invalid and I do not report them.

## Result 1 — the per-draft head budget

Slope per additional single-row head pass (`draft_head` phase, rounds > 7):

| quantity | d=1 | d=5 | per draft |
|---|---|---|---|
| Metal dispatches | 41.0 | 168.8 | **+34.07** |
| command-buffer commits | 3.00 | 11.44 | **+2.25** |
| memory barriers | 25.0 | 120.8 | **+25.55** |
| **device buffer allocations** | 0.00 | 0.00 | **+0.00** |
| **device bytes allocated** | 0 | 0 | **+0** |

## Result 2 — there is no allocation on this path

**Zero device buffer allocations occur per draft token in steady state.** MLX's buffer cache serves 100 % of the head-phase intermediates.

The hook is not silently broken. It records 18,133 allocations totalling 64.2 GB during load and warm-up, 27 allocations of 3,145,728 B in round 1 `target_verify`, 2 of 1,572,864 B in round 2 `draft_head`, and 2 of 4,096 B in round 1 `target_forward`. After warm-up the head path allocates nothing. `target_verify` still allocates 1.69 buffers (5.3 MB) per round at d=5 and nothing at d=1, so the counter is live in the same run.

So the "allocation" half of the claimed 13–16 µs mechanism does not exist on the per-draft path. Whatever the organizer pair `c6af1e24` → `8e83c6b3` bought, it was not malloc traffic.

## Result 3 — how often a commit falls between a write and a read

Marginal commits per marginal dispatch on the head path = 2.25 / 34.07 = **0.066**, i.e. one command-buffer boundary per 15.1 dispatches. A commit therefore does *sometimes* fall between the write and the read of a head-phase intermediate, but only for about 1 buffer in 15.

Costing that with the E80 constants measured on this host (dispatch 0.66–1.55 µs, commit 13.5–17.6 µs):

```
per eliminated intermediate = dispatch cost + marginal commit share
                            = 0.66-1.55 + 0.066 x 13.5-17.6
                            = 0.66-1.55 + 0.89-1.16
                            = 1.55-2.71 us
```

**The predicted value of one eliminated materialised intermediate is 1.55–2.71 µs per draft, 5–10x below the claimed 13–16 µs law.** Under the stop rule as written ("<5 µs per draft per eliminated buffer -> terminal negative"), the census already predicts the *per-buffer* law fails.

## Result 4 — the census finds more intermediates than the assignment listed

`embed_tokens` is **quantized** (`weights/config.json`: affine, 4 bit, group 64; `tie_word_embeddings=false`). `QuantizedEmbedding.callAsFunction` is three gathers plus one dequantize, so target (a) removes 4 buffers, not 1.

Confirmed per-draft kernel evidence and byte counts:

**Target (a) — quantized embedding into `qwen35_dual_rms_norm_concat_bf16_v1`**

| kernel | per draft | output buffer | bytes |
|---|---|---|---|
| `gather_frontuint32_int32_int_1 grid=640x1x1` | 1.00 | packed row | 2,560 |
| `gather_frontbfloat16_int32_int_2 grid=40x1x1` | 2.00 | scales, biases | 320 |
| `affine_dequantize_bfloat16_t_gs_64_b_4 grid=2560x1x1` | 1.00 | bf16 row | 10,240 |
| `custom_kernel_qwen35_dual_rms_norm_concat_bf16_v1 grid=2048x1x1` | 1.00 | (kept, gains inputs) | — |

5 dispatches collapse to 1. **Net −4 buffers, 13,120 B eliminated per draft.**

**Target (b) — three `MLX.take` plus `quantizedMM` into one `gatherQuantizedMM`**

| kernel | per draft | output buffer | bytes |
|---|---|---|---|
| `gather_frontuint32_uint32_int_1 grid=640x32x1` | 1.00 | gathered packed rows | 81,920 |
| `gather_frontbfloat16_uint32_int_2 grid=40x32x1` | 2.00 | gathered scales, biases | 10,240 |
| `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x4x1` | 1.00 | 32 exact logits (kept) | — |

4 dispatches collapse to 1. **Net −3 buffers, 92,160 B eliminated per draft.**

Both targets run once per draft token, not once per round. The rerank gathers scale with candidates (32), not with draft width, so their byte cost is the larger of the two.

## Census answer, in the requested form

> count, bytes, producing dispatch, commit between write and read, allocation

**7 materialised intermediates, 105,280 B, 7 producing dispatches, 0.46 commit boundaries, 0 device allocations — per draft token, for arms (a) and (b) together.**

## What I predict, and what I will do

Predicted aggregate for (a)+(b) = 7 × 1.55–2.71 µs = **10.9–19.0 µs per draft**. Against the E80 head cost of 4.810 ms per draft that is 0.23–0.40 % of head time, and roughly 0.07–0.12 % of decode time.

That aggregate straddles the 13–16 µs per-draft figure derived from the organizer pair, so the *observable* the law was fitted to is reproducible here — but the correct unit is about 2 µs per buffer, and the organizer pair must have removed several buffers, not one.

This is right at the edge of what the ungated ABBA protocol resolved in E80 (the cool gate itself measured 0.11–0.12 % on this host). I will therefore build **both** arms and measure them **together as one candidate** as well as separately, because either arm on its own is predicted to sit below the noise floor.

Plan from here, unchanged in shape:

1. Revert the census instrument; the swizzle lock makes the process unfit for timing.
2. Finish wiring arm (a) into both `Qwen35MTP.swift` call sites, keep arm (b) behind `MLX_E85_GATHER_QMM`, and add cheap numerical checks against the eager paths.
3. Rebuild with `senpai/rebuild-and-assert-worker.sh` and assert the new symbols.
4. Run ABBA-counterbalanced ungated legs in one session against a fresh unchanged base, comparing absolute candidate MTP s/tok, with `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false` preserved and entry/exit GPU temperature recorded per arm.

If the measured per-buffer value lands under 5 µs but the combined arm still shows a real absolute win, I will report both numbers plainly rather than force the result into the law's framing.

Census artefacts: `research/out/e85-census-d1/`, `research/out/e85-census-d5/`, `research/out/e85-census-slope.json`, reader `research/e85_census_slope.py`.

_This comment was created by an AI agent (OpenHands) on behalf of the qwen-edward research student._
