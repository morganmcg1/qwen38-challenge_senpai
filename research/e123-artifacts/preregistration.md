# E123 pre-registration

Written and committed **before** any timing measurement of this experiment.
Nothing below may be edited after the first `rate.json` exists; the result
report scores this file as written.

Everything here is `harness=local`. The local Mac cannot reach the 40 C cool
gate, so no number produced by this experiment is gate qualified and no number
is an official or ranked score.

## 0. Identity

| field | value |
| --- | --- |
| base | `senpai/qwen38-mtp-r1` at `61ed64fe02346bd1fc021f1c664a9cd2c67286c4` |
| branch | `qwen-askeladd/e123-price-ladder-completion-and-deletion-audit` |
| host | `ip-10-231-2-227.ec2.internal`, Apple M4 Pro, 48 GiB |
| timed architecture | `applegpu_g16s` |
| ranked architecture | `applegpu_g17s` (censused, never timed here) |
| probe | `research/e123_probe.sh`, built from the unchanged `research/e118_qmv_probe.m` |
| arms | 32, listed in `research/e123_arms.py` |
| threadgroup | 32 x 2 threads, so 2 simdgroups, the shipped dispatch |
| widths | NA = 2, 3, 4, 5 |
| round weights | `{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}` |
| harness | `local` |
| `cool_gate_passed_real_gate` | false |
| `gate_qualified_for_timing` | false |
| `official_or_ranked_score` | false |

## 1. What the ladder already knows

E118 measured three classes on this kernel, on this host, in percent of
`a_base` per injected instruction per k-block iteration:

| class | NA2 | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: | ---: |
| `alu` | 0.00430 | 0.06759 | **0.09398** | 0.07929 |
| `ld` | 0.10216 | 0.32853 | **0.58330** | - |
| `shuf` | 0.37763 | 1.06566 | **0.96486** | - |

E118 also showed the ladder reads **issue throughput**, not chain latency:
halving the dependency depth at constant instruction count did not reduce the
cost.

## 2. The static instrument the predictions come from

The 32-arm census in `research/e123-artifacts/census.json` gives machine text
bytes per injected operation at NA=4 on `applegpu_g16s`. AGX instructions in
this kernel average about 8 bytes, and `alu` injects one instruction per
operation at 8.2 bytes per operation, so

```text
one AGX instruction  ~ 8.25 text bytes  ~ 0.09398 % of a_base per k-block
                     => 0.01139 % per text byte
```

Measured bytes per injected operation at NA=4 on `applegpu_g16s`:

| contrast | bytes/op | instructions/op | naive % from bytes |
| --- | ---: | ---: | ---: |
| `k_alu8` -> `k_alu16` | 8.2 | 1.0 | 0.094 |
| `k_ld8` -> `k_ld16` | 33.2 | 4.0 | 0.378 |
| `k_shuf8` -> `k_shuf16` | 132.5 | 16.1 | 1.509 |
| `k_tgld4` -> `k_tgld8` | 43.5 | 5.3 | 0.496 |
| `k_tgldc4` -> `k_tgldc8` | 45.0 | 5.5 | 0.513 |
| `k_tgst4` -> `k_tgst8` | 36.0 | 4.4 | 0.410 |
| `k_cvt4` -> `k_cvt8` | 44.5 | 5.4 | 0.507 |
| `k_tgst8` -> `k_barst8` | 5.5 | 0.7 | 0.063 |
| `k_tgst16` -> `k_barst16` | 6.5 | 0.8 | 0.074 |
| `k_tgst8` -> `k_sbarst8` | -0.5 | 0.0 | -0.006 |
| `k_ssum2` -> `k_ssum4` | 8.0 | 1.0 | 0.091 |
| `a_base` -> `k_bar8` | 0.8 | 0.1 | 0.009 |

The naive byte model is known to be about a factor 1.5 optimistic for device
loads (0.378 predicted against 0.583 measured) and about a factor 1.6
pessimistic for shuffles (1.509 predicted against 0.965 measured), so it is a
prior and not a prediction. The predictions below start from it and add one
stated physical correction per class.

## 3. Point predictions, per class, at NA=4

All values are percent of `a_base` per injected instruction per k-block
iteration on `applegpu_g16s`. Positive means the instruction COSTS that much.

| class | point | band | reasoning |
| --- | ---: | --- | --- |
| `tgld` | **0.45** | [0.25, 0.70] | 5.3 instructions per operation from the byte model; threadgroup memory is on chip, so no device-memory correction |
| `tgldc` | **0.80** | [0.45, 1.40] | same instruction sequence as `tgld`; a 4-way bank conflict serialises the access itself into 4 issue slots, adding about 3 slots to 5.3 |
| `tgst` | **0.38** | [0.20, 0.60] | 4.4 instructions per operation; a store has no writeback and no consumer |
| `bar` | **0.15** | [0.05, 0.40] | 0.7 instructions per barrier statically, plus a real cross-simdgroup wait over 2 simdgroups |
| `sbar` | **0.00** | [-0.05, 0.05] | emits no machine text at all, so it is a compiler scheduling barrier and nothing else |
| `cvt` | **0.47** | [0.27, 0.72] | `tgld` plus the widening |
| `cvt` - `tgld` | **0.02** | [0.00, 0.09] | the widening adds about 1 text byte per operation, so at most one instruction |
| `ssum` | **0.20** | [0.05, 0.60] | 1.0 instruction per `simd_sum`, so it is a hardware reduction, but a cross-lane reduction may still cost more than one issue slot |

Two further predictions that are not prices:

- `k_bar8`, whose 8 adjacent barriers merge in the backend, measures
  **0.0 % +/- 0.3 pp** against `a_base` at every width. The AIR carries all 8
  barriers and the machine text carries 6 extra bytes in total, so a measured
  cost here would mean the merge is not real.
- **Ordering:** `tgldc` > `tgld`, with a ratio point prediction of **1.8** and
  a band of [1.3, 3.0]. If the ratio is below 1.1 the bank-conflict model of
  this kernel's threadgroup file is wrong and must be reported as wrong.

## 4. The held-out arms and the exact prediction formulas

The three holdout arms are fitted nowhere. Their predictions are computed from
ladder quantities measured in the same session. The formulas are fixed here.

Write `cost(A)` for the measured percent by which arm `A` is SLOWER than
`a_base` at the width in question, and `p_c` for the measured price of class
`c` at that width.

```text
pred(k_hold_alu12) = cost(k_alu8) + 4 * p_alu

S_sl               = 0.5 * ( cost(k_shuf8) - 8 * p_shuf
                           + cost(k_ld8)   - 8 * p_ld )
pred(k_hold_sl)    = S_sl + 4 * p_shuf + 4 * p_ld

pred(k_hold_mix)   = cost(k_tg0) + 8 * p_tgst + 8 * p_tgld
                                 + 2 * p_bar  + 4 * p_alu
```

`S_sl` is the injection scaffold shared by the two E118 classes: the `cal`
declaration, the address arithmetic and the sink. It is estimated twice, once
from each class, and averaged. Any disagreement between the two estimates is
reported.

`cost(k_tg0)` is the measured threadgroup scaffold, so `pred(k_hold_mix)` is an
absolute prediction and not a contrast.

## 5. The primary metric

```text
err(arm, NA) = | pred(arm, NA) - cost(arm, NA) |     in percentage points

e123_ladder_out_of_sample_prediction_error_pp
    = median over { k_hold_mix, k_hold_alu12, k_hold_sl } of err(arm, 4)
```

NA=4 is the headline width because it carries 0.667 of the round. The report
also gives, for every holdout: `err` at every non-spilling width, the
round-weighted `err` over available widths, and the maximum over the three
holdouts.

Baseline to beat: **0.66 pp**, which is E118's propagated one-standard-error
band on its own out-of-sample prediction of `n_nosums`. Direction: minimize.

**Kill rule, from the assignment.** If any held-out arm cannot be predicted to
better than +/- 1.0 pp at NA=4, rung 0 has failed. The report says so, rung 1
is not run as a priced ranking, and the ladder is reported as not fit for
pricing an exchange arm.

Cells are excluded from every price and every score wherever the census says
the arm spills on `applegpu_g16s`. At NA=4 that excludes `k_tgld16`,
`k_tgldc16` and `k_cvt16`, which is why every threadgroup class has a 4 rung.

## 6. Rung 0 secondary metrics, as required by the assignment

| metric | definition |
| --- | --- |
| `e123_threadgroup_load_pct_per_instr_per_kblock_na4` | `p_tgld` at NA=4 |
| `e123_threadgroup_store_pct_per_instr_per_kblock_na4` | `p_tgst` at NA=4 |
| `e123_barrier_pct_per_barrier_per_kblock_na4` | `p_bar` at NA=4 from `k_barst8 - k_tgst8` |
| `e123_bf16_to_float_conversion_pct_per_instr_per_kblock_na4` | `p_cvt - p_tgld` at NA=4 |
| `e123_largest_predicted_bit_exact_deletion_pct_round_weighted` | rung 1 output |

## 7. Rung 1: the priced instruction census of the scored kernel

`qmv_fast_crossrow_affine4_g64_wide`, `quantized.h:969-1065`, with
`DIRECT_NIBBLES = true`, which is the promoted setting. Counts are per lane per
k-block iteration and were read from the source before any timing.

| group | count at NA=4 | scaling |
| --- | ---: | --- |
| weight element loads | 16 | fixed |
| metadata loads | 8 | fixed |
| metadata widenings | 8 | fixed |
| activation vec4 loads | 16 | 4 * NA |
| activation widenings | 64 | 16 * NA |
| `sums` chain | 80 | 20 * NA |
| activation register moves | 64 | 16 * NA |
| nibble integer operations | 112 | fixed |
| nibble integer to float conversions | 64 | fixed |
| lane FMAs | 256 | 64 * NA |
| final accumulate | 48 | 12 * NA |
| epilogue `simd_sum`, amortised over 10 k-blocks | 1.6 | 0.4 * NA |
| **total** | **~738** | |

**Instrument test, pre-registered.** The NA=4 ALU price of 0.09398 %/instr
implies a whole-kernel budget of `100 / 0.09398 = 1064` instruction
equivalents. The census above accounts for 738, or **69 %**. The prediction is
that the reconstruction ratio lands in **[0.55, 0.85]**: below 0.55 would mean
the census misses most of the kernel and cannot rank deletions; above 1.0 would
mean the ALU price is too small and the whole ladder is mis-scaled.

Rung 1 ranks each group as (a) bit-exact deletable, (b) precision changing or
(c) not deletable, and prices it as
`count * p_class * weight(NA)` summed over NA with the standing round weights.

Rung 2 runs only if rung 1 predicts a bit-exact deletion above **+1.0 %**
round weighted.

## 8. Stop rule

- Rung 0 fails if any holdout misses by more than 1.0 pp at NA=4.
- Rung 0 also fails if `q_scaffold`, the byte-identical null, moves by more
  than 0.3 pp at NA=4, because that is the session noise floor and the
  predictions are quoted to 0.1 pp.
- A class whose measured price is inside its band is confirmed. A class outside
  its band is reported as a miss with the physical reason, and its band is not
  widened after the fact.
