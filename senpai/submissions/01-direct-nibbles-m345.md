## Summary

Take the `DIRECT_NIBBLES` path in the wide cross-row affine4 / group-64 QMV kernel at
verify widths **M = 3, 4, 5**. This is a three-line change to a dispatch table that
already used that path at M = 6, 7, 8, 9. It removes roughly 12% of the per-block ALU
work on the activation side of the kernel that runs the MTP verify pass, and it is
bit-exact: the parity gate reports zero differing cells.

## What changed

Two files, +18/-3 lines each (15 of those lines are an identical rationale comment, so
the vendored JIT twin stays byte-identical to its readable source):

* `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
* `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`

```
qmv_fast_crossrow_affine4_g64_m<T, 3, 3>  ->  <T, 3, 3, true>
qmv_fast_crossrow_affine4_g64_m<T, 4, 4>  ->  <T, 4, 4, true>
qmv_fast_crossrow_affine4_g64_m<T, 5, 3>  ->  <T, 5, 3, true>
```

All seven wide dispatches now carry `DIRECT_NIBBLES = true`. No kernel body was
touched and no new template was written. `M = 4` is the only genuinely new
instantiation: `M = 3` lowers to `wide<T, 3, true>` and `M = 5` lowers to
`wide<T, 3, true>` + `wide<T, 2, true>`, all of which the existing M = 7 and M = 8
tails already instantiate.

## Why it is exact rather than approximately exact

The default path calls `load_vector<T, float, 4, 4>`, which writes the activation lane
pre-divided as `x[i], x[i+1]/16, x[i+2]/256, x[i+3]/4096`, and then multiplies it
against weight nibbles that were left in place by masking with `0x000f, 0x00f0,
0x0f00, 0xf000`. Each product is therefore `(x_j * 2^-4j) * (n_j * 2^+4j)`.

`DIRECT_NIBBLES` cancels that pair: the activation is read unscaled and each nibble is
shifted down to `[0, 15]` before the fused multiply-add, giving `x_j * n_j` directly.
Every factor removed is a power of two, so it perturbs the exponent field only and
never the mantissa; the accumulation order is unchanged term for term; and the
`DIRECT_NIBBLES` branch reproduces `load_vector`'s own expression tree for the
scale/bias sum (`sums[m] += xm[0] + xm[1] + xm[2] + xm[3]`). The result is bit-identical
independent of M and of the group width NA.

The cost that disappears is the `12 * NA` power-of-two multiplies per 512-element
k-block per SIMD group on the activation side. The weight side trades one AND for one
bitfield-extract, which is 1:1, and the `64 * NA` FMAs are unchanged. The saving is
width-independent, which is why it was already correct to have taken it at the wider
verify widths.

## Evidence

* **Bit-exactness.** A dedicated QMV parity harness digests `quantized_matmul` over
  verify widths 1..12 for the eight shapes the scored model actually dispatches, and
  compares the incumbent kernel against the candidate. Verdict: **96 cells compared,
  0 differing, 0 widths differing - BIT-IDENTICAL**.
* **Gate power, not just a green light.** A zero-difference verdict is only worth
  something if the gate can see the region that changed. The eight scored shapes have
  `n` of 16480, 5120, 14336, 5120, 34816, 5120, 248320 and 98336, all >= 4096, so all
  eight reach the wide `_m` path; 8 shapes x 7 wide widths (M = 3..9) = 56 cells, which
  matches the harness's recorded 56/96 positive-control count exactly. The gate
  demonstrably covers M = 3, 4 and 5.
* **Non-vacuity.** The two arms were built from different sources and produced
  different binaries: reference twins `a83039b7...` / `c2200b8f...` versus candidate
  twins `6247ca18...` / `97785051...`. Both arms emitted 96 cells.
* **End-to-end local gate.** `./benchmark-qwen-mtp.sh --local-submit` (128 tokens,
  depth 8 offer, public long-copy golden fixture) passes with
  `all_tokens_matched = true` on both timed legs, `reference_checked_rows` 128/128 and
  134/134, `residual_divergence_count = 0`, and `public_drift_tripwire_passed = true`.

## What this submission does not claim

The local box is an Apple M4 Pro (20 GPU cores, `applegpu_g16s`), not the ranked M5.
No local GPU temperature reader was available on this host, so the local cool-down gate
warned and skipped; the local timings are therefore **not** thermally gate-qualified
and no local speedup number is offered as a prediction of the ranked score. The claim
being made is narrow and falsifiable: the kernel change is bit-exact and strictly
removes ALU work. Whether that ALU work is on the critical path at the ranked box's
compute-to-bandwidth ratio is exactly what the ranked run decides.

## Attribution

`Model: senpai` on the first line above is a campaign label, not a catalogued model.
It denotes a multi-agent research program - one advisor agent supervising four student
agents that run experiments on separate GPUs and report results through pull requests.
The exact underlying models, effort levels and harness are:

| Component | Value |
| --- | --- |
| Frontier model | `anthropic/claude-fable-5` |
| Smart model (advisor and students) | `anthropic/claude-opus-5`, reasoning effort `max` |
| Fast model (mechanical subtasks) | `anthropic/claude-sonnet-5` |
| Default reasoning effort | `max` |
| Agent harness | Senpai (`github.com/wandb/senpai`), native backend, OpenHands-derived agent SDK |
| Roles in this launch | 1 advisor + 4 students, 1 GPU each |
| Compute backend | `aws-mac` |
| Experiment tracking | Weights & Biases |

No human wrote any of the code or analysis in this submission; humans set the campaign
up and answered clarifying questions on GitHub issues.
