# E118 - the metadata-load instruction axis of the wide affine-4 QMV

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e118_best_bit_exact_arm_round_weighted_pct_faster_vs_a_base","available":true,"value":0.2202},"test_metric":{"name":"positive_control_failures","available":true,"value":0}}
```

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e118-wide-qmv-inner-loop-load-instruction-screen`
- Hypothesis and target cost: the eight scalar metadata reads in the
  `qmv_fast_crossrow_affine4_g64_wide` inner loop occupy the load-issue port,
  so removing or coalescing them makes the kernel faster. The target cost is
  Finding 44: `a_base` runs 15.36 % (round weighted) above its own
  load-only ceiling `l_loadonly`.
- Decision: **dead**. The primary metric is `+0.2202 %`, the kill rule is
  `+0.5 %`, so the metadata-load route is closed.
- `BASE_SHA`: `1d2320bece29cddc94b95e5f99f00331b05a5025`
- Candidate commit: this branch head. No candidate file changed.
- Yukon promoted submission / frontier: unchanged. This experiment proposes no
  submission.
- Candidate build fingerprint: not applicable. No worker was built and no
  model was held. Every arm is a standalone Metal entry point.
- Submitted-surface / twin / metallib digests: unchanged, none touched.
- Submitted candidate files: **none**.
- Supporting research files: `research/e118_arms.py`,
  `research/e118_qmv_probe.m`, `research/e118_probe.sh`,
  `research/e118_analysis.py`, `research/e118_wandb_log.py`,
  `research/e118-artifacts/`.
- MTP head provenance and draft policy: not applicable. This probe runs no
  session and proposes no draft.
- Token window, fixture, reference source, harness: not applicable to a
  microbenchmark. Operands are synthetic. The reference is an exact affine-4
  evaluation in double on the CPU. **`harness=local`** on every number below.
- Exact cell: `qmv_fast_crossrow_affine4_g64_wide`, five scored shapes, widths
  NA 2, 3, 4 and 5, entry points `e118_iso_na2..na5`, JIT source form, local
  `applegpu_g16s` measured and ranked `applegpu_g17s` statically translated.
- Official causal path and score equation: none is claimed. `harness=local`
  throughout. No local ratio is presented as a ranked term.
- Assignment-scope preflight: `senpai/verify-ranked-score-boundary.sh` **PASS**.
  `senpai/validate-assignment-scope.sh` has nothing to check, because the diff
  against `BASE_SHA` touches `research/` only. None of the four forbidden files
  is modified.
- Editable source bytes: `2607365/3000000`, headroom `392635`, growth
  **`0/262144`**, exempt `2410`.
- Scored-path reachability: the arms are transcriptions of the shipped
  `quantized.h` inner loop into private entry points. This is a **screen, not
  an end-to-end measurement**, and no arm is proposed for the scored path.

## Evidence

- Host: `ip-10-231-2-227.ec2.internal`, Apple M4 Pro, `applegpu_g16s`, 20-core
  GPU, 48 GiB. Swift 6.3.3, Metal 32023.883. Fast math **off** in both the
  probe and the census, so no arm is allowed to reassociate.
- Thermal policy: entry 37.28 C, exit 63.14 C.
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `timing_valid=false` in `meta.txt`, preserved verbatim. Arms run in
  palindrome (ABBA) order inside one session, so monotone drift cancels to
  first order. **These numbers are directional causal evidence within one
  counterbalanced session. They are not gate-qualified and are not a score.**
- Exact command:

  ```bash
  bash research/e118_probe.sh e118-full2 \
    --shapes 0,1,2,3,4 --widths 2,3,4,5 --pairs 8 --samples 24
  python3 research/e118_arms.py --emit /tmp/e118-census
  python3 research/e118_arms.py --census /tmp/e118-census \
    --out research/e118-artifacts/census.json
  python3 research/e118_analysis.py \
    --rate research/out/e118-full2/rate.json \
    --census research/e118-artifacts/census.json \
    --slice research/e118-artifacts/e114_receipt_slice.json \
    --out research/e118-artifacts/summary.json
  ```

- Runtime 221 s. Probe `git_head=b2727f2e`, `git_dirty=0`.
- Falsification gate and positive controls: every bit-exact-required arm is
  perturbed twice per shape, once on the activation and once on the whole
  metadata record (`scales`, `biases`, `packed_sb` and `bias_codes` together)
  across eight output rows, and is then restored. **0 control failures** over
  the whole session. `q_scaffold` and `ctl_a_base_via_m` compile to machine
  text byte-identical to `a_base` on both architectures, so they measure the
  harness noise floor: `-0.036 %` round weighted.
- Exact-token and row-ledger verdict: not applicable. No session ran.

### W&B runs

Group `e118-wide-qmv-metadata-load-instruction-screen`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`. Every run carries
`harness=local`, `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false`.

| Run ID | Name | Contents |
| --- | --- | --- |
| [`e118arms1`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118arms1) | `e118-arms` | the 15 timed arms by width and shape, the primary metric, the discriminator, Finding 44, the defect-16 forward-reverse gap and the E111 bias axis |
| [`e118stat1`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118stat1) | `e118-static-budget` | AIR device loads and shuffles, plus registers, spill bytes and ISA text for `applegpu_g16s` and `applegpu_g17s` at NA 2-5 |
| [`e118spil1`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118spil1) | `e118-spill-defect` | the 25 NA=5 exactness failures, each arm against the exact double reference, and the spill-to-exactness join that carries the `z_ballast` control |

### Primary metric

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `e118_best_bit_exact_arm_round_weighted_pct_faster_vs_a_base` | 0.0 | **+0.2202** | +0.2202 |

Best bit-exact metadata-load arm: `g_pack32`. Kill rule `+0.5 %`:
**NOT CLEARED**. Standing weights `{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}`.

### Round-weighted percent faster than `a_base`, `mlp.gate_up`

Positive means faster. `sem` in the by-width table below.

| arm | role | NA2 | NA3 | NA4 | NA5 | round weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `l_loadonly` | diagnostic | +1.658 | +3.394 | +16.266 | +28.442 | **+12.790** |
| `n_nobias` | diagnostic | +5.470 | +5.682 | +8.931 | +11.876 | +8.055 |
| `n_nosums` | diagnostic | +0.211 | +1.766 | +7.420 | +9.504 | +5.763 |
| `g_pack32` | promotion | +0.174 | +0.458 | +0.123 | +0.241 | **+0.220** |
| `p_split_meta` | promotion | -0.068 | +0.042 | -0.024 | +0.102 | -0.003 |
| `q_scaffold` | null control | -0.020 | -0.024 | -0.053 | +0.193 | -0.036 |
| `d_bias1` | diagnostic | +2.604 | -0.530 | -0.549 | -1.052 | -0.485 |
| `p_prefetch_w` | beside metric | +0.384 | +0.134 | -0.089 | -63.282 | -2.165 |
| `e_bias6` | beside metric | +2.219 | -4.249 | -3.123 | -1.873 | -3.262 |
| `z_ballast` | spill control | +0.428 | -1.969 | -3.589 | -69.158 | -5.277 |
| `s_bcast_all` | promotion | -0.136 | -7.931 | -7.094 | -12.726 | -7.348 |
| `s_bcast_scale` | promotion | -0.027 | -8.818 | -7.668 | -6.207 | -7.751 |
| `s_bcast_pack32` | promotion | +0.124 | -5.884 | -7.314 | -58.962 | -8.498 |
| `s_bcast` | promotion | -1.680 | -15.005 | -15.458 | -62.826 | **-16.614** |

Standard error of the median is between 0.03 and 0.25 percentage points on
every cell, so `g_pack32` at `+0.220` is a real but small effect and every
broadcast arm is a large, unambiguous loss.

Absolute `a_base` microseconds on `mlp.gate_up`: 415.4, 430.3, 499.9 and 592.8
at NA 2, 3, 4 and 5.

### Every shape agrees

Round-weighted percent, all five scored shapes:

| arm | fa_qkv | gdn_in_proj | lm_head | mlp_down | mlp_gate_up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `g_pack32` | +0.361 | +0.348 | +0.200 | +1.013 | +0.220 |
| `p_split_meta` | +0.107 | +0.093 | +0.055 | +0.262 | -0.003 |
| `n_nosums` | +5.766 | +5.728 | +5.799 | +5.851 | +5.763 |
| `s_bcast` | -16.411 | -16.465 | -16.822 | -16.539 | -16.614 |
| `l_loadonly` | +14.629 | +14.588 | +11.774 | +21.140 | +12.790 |

No shape reverses any sign. The screen is not a one-shape artefact.

### Discriminator verdict, required by the assignment

```text
s_bcast -16.614   s_bcast_all -7.348   p_split_meta -0.003   n_nosums +5.763
```

**Binding resource: total instruction issue or ALU. It is not the load-issue
port and it is not memory latency.**

The census explains why, and it is the central result of this experiment:

| arm | AIR device loads per entry point |
| --- | ---: |
| `a_base` | **7** |
| `q_scaffold`, `p_split_meta`, `n_nosums`, `l_loadonly`, `d_bias1`, `e_bias6`, `z_ballast` | 7 |
| `s_bcast`, `s_bcast_all` | 7 + 2 shuffles |
| `s_bcast_scale` | 7 + 1 shuffle |
| `g_pack32`, `n_nobias` | **6** |
| `s_bcast_pack32` | 6 + 1 shuffle |
| `p_prefetch_w` | 8 |

The shipped kernel already issues only **7** device loads. The eight scalar
metadata reads the hypothesis targeted are **already coalesced by the front
end** before any arm touches them. So:

- `s_bcast` removes **zero** loads. It only adds two shuffles and 25 registers
  at NA=2, and it loses 16.6 %.
- `g_pack32` genuinely removes one load of seven, and it gains `+0.22 %`,
  which is near the `0.036 %` noise floor and far below the kill rule.
- `p_split_meta` compiles to text byte-identical to `a_base`, so it is a true
  null by construction, and it measures `-0.003 %`. That is the second null
  control and it behaves.

Removing one seventh of the device loads buys almost nothing, while deleting
arithmetic (`n_nosums`, `+5.763 %`) buys a lot. The port is not binding.

### Finding 44 placement, `mlp.gate_up`

| NA | `a_base` us | `l_loadonly` us | gap % |
| ---: | ---: | ---: | ---: |
| 2 | 415.4 | 408.5 | 1.67 |
| 3 | 430.3 | 415.6 | 3.54 |
| 4 | 499.9 | 418.4 | 19.49 |
| 5 | 592.8 | 425.2 | 39.43 |

Round-weighted gap **+15.36 %**. The headroom above the load ceiling is real
and grows steeply with width, but this experiment shows it is not reachable by
touching the loads.

### E111 bias axis, folded in at every width

E111 measured these at NA=5 only, which carries 0.034 of the standing weight.
This session measures them at every width.

| quantity | NA2 | NA3 | NA4 | NA5 | round weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n_nobias`, whole bias axis | +5.470 | +5.682 | +8.931 | +11.876 | +8.055 |
| `n_nosums`, arithmetic only | +0.211 | +1.766 | +7.420 | +9.504 | +5.763 |
| difference, **the bias load** | +5.258 | +3.916 | +1.512 | +2.371 | +2.292 |
| `d_bias1`, Bias6 ceiling | +2.604 | -0.530 | -0.549 | -1.052 | -0.485 |
| `e_bias6`, real, bit exact | +2.219 | -4.249 | -3.123 | -1.873 | -3.262 |
| difference, reconstruction | +0.386 | +3.719 | +2.573 | +0.821 | +2.776 |

`e_bias6` is **bit exact** at every cell above. The whole bias axis is worth
`+8.06 %`, but only `+2.29 %` of that is the load; the rest is the arithmetic.
The one-byte code cannot collect even the load part: its own ceiling
`d_bias1` is already **negative** (`-0.485 %`) once the whole width range is
weighted, and the exact reconstruction costs a further `+2.776 %`. **Bias6 is
a loss on this host at every width above NA=2.** E111's NA=5-only reading
understated the cost, and reweighting over the standing distribution does not
rescue it.

### Secondary finding: an NA=5 spill defect that silently corrupts results

Twenty-five exactness failures were recorded, and they are not a bug in any
arm. Every failure is at **NA=5**, on **all five shapes**, and every failing
arm spills on `applegpu_g16s`:

| arm at NA=5 | g16s spill | verdict | rel error against the double reference |
| --- | ---: | --- | ---: |
| `q_scaffold`, `p_split_meta`, `g_pack32`, `e_bias6` | 0 B | exact | 3.88e-2 |
| `s_bcast_scale` | 16 B | exact | 3.88e-2 |
| `s_bcast_all` | 80 B | **WRONG** | 5.86e+1 |
| `s_bcast_pack32` | 144 B | **WRONG** | 4.75e+1 |
| `p_prefetch_w` | 192 B | **WRONG** | 4.75e+1 |
| `s_bcast` | 208 B | **WRONG** | 5.86e+1 |
| **`z_ballast`** | 208 B | **WRONG** | 4.75e+1 |

- Largest spill that stayed exact: **16 B**. Smallest spill that went wrong:
  **80 B**. Spill separates the two groups perfectly.
- Every arm is scored against an exact affine-4 reference evaluated in
  **double** on the CPU, so the report names which side is wrong. The exact
  arms sit at `rel = 3.9e-2`, which is ordinary bf16 accumulation noise. The
  spilling arms sit at `rel = 47` to `59`, with `max_rel = 2.0` against
  `a_base`, meaning sign flips on 99.9 % of cells. **The spilling arms produce
  garbage, not a different valid rounding.**
- `z_ballast` is the decisive control. It adds twelve dead loop-carried floats
  that are consumed only inside a branch that never executes, so no value it
  computes reaches `y`. It changes the spill budget and nothing else. **It is
  equally wrong.**

**Conclusion: on Apple M4 Pro `applegpu_g16s` with Metal 32023.883, the wide
qmv NA=5 entry point returns numerically wrong results whenever the compiler
spills 80 bytes or more. The cause is spilling, not any arm's mechanism.**
Fast math is off, so reassociation does not explain it. `a_base` itself does
not spill at NA=5 and stays correct, so nothing shipped is affected today.

Note the transfer risk: the ranked `applegpu_g17s` has a larger register
budget and only one arm spills there at all (`s_bcast`, 16 B at NA=5). The
defect may therefore be local to this generation. That does not make it safe
to ignore, because a future wide-qmv change that spills on the ranked runner
would be silently wrong rather than merely slow.

### Two harness defects found and repaired during this experiment

Both were found by disbelieving a control that passed, and both had been
weakening the exactness screen.

1. **NaN synthetic biases.** `bias_bf16_from_code` takes the negative-zero
   path when a code's low nibble is zero, and the exponent adjustment then
   lands on the bf16 pattern `0x7fff`, which is NaN. One group in 64 was
   affected, so about 72 % of output columns carried a NaN. NaN compares
   bit-equal to NaN, so the screen was passing on poisoned data. The generator
   now excludes that code, and the probe reports `base_nonfinite` per width.
   The reported session has **0 non-finite baseline elements** at every cell.
2. **A metadata positive control that could not fire.** It perturbed one group
   out of eighty, and the bf16 output rounding absorbed it, so `meta_hit` was
   0 for every arm. It now perturbs every group of eight rows spread across
   the output and returns exactly `8 columns x 4 rows = 32` hits. All controls
   pass in the reported session.

Every timing number in this report comes from the repaired probe.

## Conclusion

- **What happened:** the metadata-load hypothesis is wrong for a reason the
  static census makes unambiguous. The shipped inner loop already issues 7
  device loads, not 15, because the front end coalesces the eight scalar
  metadata reads. There is no load-issue pressure left to remove. The best
  bit-exact arm gains `+0.22 %` round weighted against a `+0.5 %` kill rule
  and a `0.036 %` noise floor, so the route is closed.
- **Evidence for the mechanism being absent:** `s_bcast` removes zero loads
  and loses 16.6 %; `g_pack32` removes one load of seven and gains 0.22 %;
  `p_split_meta` is byte-identical text and measures zero. Deleting
  arithmetic instead (`n_nosums`) gains 5.76 %. The discriminator therefore
  selects total instruction issue or ALU, on all five shapes.
- **Transfer risk:** the ranked `applegpu_g17s` allocates more registers for
  the same source, so the broadcast arms would be even more costly there and
  the conclusion strengthens rather than weakens. `g_pack32` is 42 bytes of
  ISA text cheaper than `a_base` at NA=5 on g17s, so its tiny gain probably
  survives, but it is far too small to matter.
- **Smallest useful next action:** stop working the load axis of this kernel.
  The measured headroom is on the arithmetic axis, where `n_nosums` shows
  `+5.76 %` and `l_loadonly` shows `+12.79 %`. The bias axis specifically is
  worth `+8.06 %`, but E111's one-byte recoding cannot collect it: this
  session shows `d_bias1` is already negative once every width is weighted.
  The next question should ask which arithmetic in the accumulation can be
  removed or reassociated without changing the result, not which load.

## Suggested follow-ups, not implemented

1. Ask whether the `sums` accumulation can be folded into the existing
   dot-product reduction rather than deleted. `n_nosums` prices the whole
   removal at `+5.76 %`; a fold would collect part of it while staying exact.
2. Confirm the NA=5 spill defect on the ranked `applegpu_g17s` before any
   future wide-qmv change is allowed to spill there. A ten-minute run of this
   probe on an M5 host would settle it and would protect every later kernel
   experiment.
3. Investigate why `a_base` needs 95 registers at NA=5 while `l_loadonly`
   needs 95 for a far smaller body. Register pressure, not loads, is what puts
   NA=5 within 80 bytes of the corrupting spill threshold.
