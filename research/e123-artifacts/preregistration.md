# E123 pre-registration

Written and committed **before** any timing measurement of this experiment.
Sections 0 to 8 below are frozen as committed in `63604656`; the result report
scores them exactly as written. Sections 9 and 10 are disclosed amendments,
both added before the measurement session and after that commit. They add
arms, predictions and validity gates. They change no existing prediction, no
formula, no band and no kill rule.

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

## 9. Amendment 1: what the harness smoke run showed

Added after the harness smoke run `research/out/e123-smoke` (1 shape, NA=4,
2 pairs, 4 samples, at commit `63604656`) and before the measurement session.
**Sections 0 to 8 are unchanged.** No prediction, band, formula, primary
metric, kill rule or exclusion rule in them is modified.

### 9.1 Two facts the smoke run made visible

1. `k_tgld16` and `k_tgldc16` fail exactness at NA=4 on `applegpu_g16s`
   (139148 and 139185 of 139264 elements differ) and their positive controls
   fail with `restored_diff` equal to the whole output. These are exactly the
   two arms the census says spill 128 and 144 bytes. The pre-registered spill
   exclusion in section 5 already removes them. `k_cvt16` spills 16 bytes and
   stays exact.
2. On that one cell, `pred(k_hold_alu12)` missed by 0.01 pp and
   `pred(k_hold_mix)` by 0.45 pp, both inside the gate, while
   `pred(k_hold_sl)` missed by 3.63 pp.

### 9.2 The `k_hold_sl` formula does not extrapolate

Substituting section 4 collapses the formula completely:

```text
pred(k_hold_sl) = S_sl + 4 * p_shuf + 4 * p_ld
                = 0.5 * ( cost(k_shuf8) + cost(k_ld8) )
```

Both slopes cancel and no scaffold term survives. That expression is exactly
the additive prediction for any cost of the form `S + a * n_shuf + b * n_ld`,
whatever the shared scaffold `S` is. So a miss here cannot be blamed on the
`shuf` price, on the `ld` price, or on extrapolating below the rung range. It
means the two classes are not additive with each other.

The static census says the miss is not an instruction-count effect either. At
NA=4 on `applegpu_g16s`:

```text
a_base   6934 B      k_cal0   7000 B      k_ld4    7260 B     k_shuf4  7516 B
k_ld8    7492 B      k_shuf8  8056 B      k_hold_sl 7782 B

additive text prediction  = 7260 + 7516 - 7000 = 7776 B   (+6 B against 7782)
formula text prediction   = 0.5 * (8056 + 7492) = 7774 B   (+8 B against 7782)
```

Both agree with the measured machine text to within 8 bytes, or one
instruction. `k_hold_sl` therefore issues the predicted instruction count. If
its TIME is not additive, the excess is dynamic, not a miscount.

### 9.3 Three diagnostic arms, and one pre-registered dynamic hypothesis

`k_cal0`, `k_ld4` and `k_shuf4` are added. They are not rungs of any ladder, so
every price and every prediction in sections 3 to 6 is computed and scored
exactly as written, and the primary metric is untouched. `k_cal0` is the
non-threadgroup injection scaffold at zero injected operations, the direct
analogue of `k_tg0`. With it the additivity test needs no slope at all:

```text
additive:      cost(k_hold_sl) == cost(k_shuf4) + cost(k_ld4) - cost(k_cal0)
superadditive: cost(k_hold_sl) >  that
```

**Pre-registered dynamic hypothesis.** If `k_hold_sl` is superadditive, the
cause is register residency. At NA=4 on `applegpu_g16s` the census gives
`k_ld8` 92 registers, `k_shuf8` 94, `a_base` 94 and `k_hold_sl` 96, and the
fitted residency model gives 33 concurrent simdgroups at 92 registers and 32
at 94 and above. Mixing the classes costs one residency step that neither
single-class arm pays.

- **Prediction:** `k_hold_sl` is superadditive by more than 1.0 pp at NA=4, and
  the arms at 92 registers are the ones the ladder under-prices.
- **Falsified if** the additivity test above is satisfied within 1.0 pp, in
  which case the smoke-run miss was single-cell noise, or if the superadditive
  gap appears at widths where no residency step separates the arms.

## 10. Amendment 2: the advisor feedback of 2026-08-22T04:08:25Z

Added before the measurement session. Sections 0 to 8 are unchanged. This
section adds pre-registered predictions and gates; it removes none.

### 10.1 Injection price against deletion price, in one session

The ladder is calibrated by injection and the campaign uses it to price
deletions. Alphonse's cross-session accounting says these differ by 1.66x: the
Finding 59 ALU price predicts +3.760 % for `n_halfsums_free` at NA=4 against
+2.266 % measured, an implied deletion price of 0.0567 %/instruction/k-block.
Those two numbers come from two different sessions, so the ratio is confounded
with everything that differs between them.

`n_halfsums_free` is added to this session, so the deletion side becomes a
two-rung ladder measured beside the injection ladder in the same
counterbalanced run:

```text
p_alu_inject = ( cost(k_alu16) - cost(k_alu8) ) / 8
p_alu_delete = ( cost(n_nosums) - cost(n_halfsums_free) ) / D
R            = p_alu_inject / p_alu_delete
```

Both contrasts cancel their own zero point, so `R` carries no scaffold term.
`D` is the instruction count the second arm issues and the first does not.

**The count is not assumed.** AIR keeps the `m` loop rolled and cannot see the
deletion, so `D` is read from the machine-text census at 8.25 bytes per AGX
instruction. Measured before the session on `applegpu_g16s`:

| contrast | NA2 | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: | ---: |
| `a_base` -> `n_nosums`, bytes | 314 | 468 | 634 | 816 |
| implied instructions | 38.1 | 56.7 | 76.8 | 98.9 |
| source accounting, `20 * NA` | 40 | 60 | 80 | 100 |
| `n_halfsums_free` -> `n_nosums`, bytes | 192 | 338 | 354 | 514 |
| implied instructions | 23.3 | 41.0 | 42.9 | 62.3 |
| source accounting | 20 | 20 | 40 | 40 |

The whole-tree deletion agrees with the source accounting to within 5 % at
every width, so the count behind the 1.66x is right and a miscount cannot
explain the ratio. The half-tree rows do not agree as well, so **`D` is taken
from the machine text and the source-accounting value is reported beside it.**

**Point prediction: `R` = 1.35, band [1.05, 1.90].** Reasoning: two live
mechanisms pull in opposite directions. E118's Finding 60 says this loop is
issue-throughput bound, and a pure issue-bound machine gives `R` = 1.00.
Against that, the injected operations run on two independent accumulator
chains and have no consumer, so they are the cheapest possible instructions to
schedule, while the deleted add tree is loop carried through `sums`. The
advisor's cross-session 1.66 is the only measurement available and is taken as
the upper half of the band.

**Consequence, pre-registered.** If `R` lands above 1.15, every rung-1 deletion
price is divided by the measured `R` before it is compared with the +1.0 %
build bar, and the ranking reports both the undivided and the divided value. If
`R` is inside [0.90, 1.15], injection price is deletion price on this class,
the divisor is 1, and the report says the cross-session 1.66 was a
between-session artifact.

### 10.2 The free threadgroup-access prediction

From two arms E118 already measured:

```text
n_halfsums_free  +2.266 %   at NA=4     the saving with no exchange
x_sumshare_min   +1.465 %   at NA=4     the same saving, paid for by sharing
exchange cost     0.801 pp  at NA=4     2 barriers + 4 threadgroup accesses
```

If barriers cost about 2 cycles, almost all of the 0.801 pp is the four
accesses, giving **0.20 %/instruction/k-block at NA=4** for one threadgroup
access.

This is scored as a fourth held-out prediction, against the mean of the
measured `p_tgld` and `p_tgst` at NA=4, which is what "one threadgroup access"
means in the exchange arm's mix of two reads and two writes:

```text
p_tgaccess_measured = 0.5 * ( p_tgld + p_tgst )      at NA=4
err_tgaccess        = | 0.20 - p_tgaccess_measured |  in pp
```

The exchange cost is also decomposed with the measured barrier price and
scored as a whole:

```text
pred(exchange cost) = 2 * p_bar + 2 * p_tgld + 2 * p_tgst
err_exchange        = | pred - 0.801 |               in pp
```

Both are reported against the same 1.0 pp gate as the three section-4
holdouts. They are **additional** to the primary metric, which stays the
median over `k_hold_mix`, `k_hold_alu12` and `k_hold_sl` exactly as section 5
defines it, because those three are measured in this session and the 0.801 pp
is not.

**Interpretation, fixed in advance.** A measured access price near 0.20
confirms the class from a completely independent arm. Near 0.05 means barriers
are NOT cheap on this part and carry most of the 0.801 pp, which would falsify
the Apple two-cycle claim the same way E118's `simd_shuffle` result falsified
its neighbour. The measured `p_bar` decides between them directly.

### 10.3 Three mandatory validity gates

Adopted from Alphonse's `research/e121_analysis.py`. Any one of them voids the
session before a speedup number is printed:

1. implied bandwidth above `1.2 * 273` GB/s;
2. the null scaffold arm moving more than 0.50 % at any width;
3. any positive control reporting `detected=False`.

Gate 2 is stricter here than section 8's 0.3 pp NA=4 noise rule, and both
apply: `q_scaffold` must stay inside 0.3 pp at NA=4 and inside 0.50 % at every
width.

Gate 3 needs one exception, declared now rather than after the fact. The smoke
run already showed that `k_tgld16` and `k_tgldc16` fail their positive control
at NA=4, and section 5 already excludes those cells for spill. A positive
control failure **on a cell that the pre-registered spill rule excludes** voids
that cell, not the session. A positive control failure on any retained cell
voids the session.

### 10.4 Harness defect 22

`research/e118_qmv_probe.m` dispatches `MTLSizeMake(m, n/8, 1)` with `m = NA`,
so an entry point without a row-bounds guard writes rows up to `NA^2 - 1`,
faults the command buffer, and lets every later dispatch retire in 1 to 3
microseconds while the harness still exits 0.

Checked before this session: every entry point this experiment emits already
carries

```c
const int first_m = int(tid.x) * NA;
if (first_m >= NA) { return; }
```

at all four widths, in `arm_*.metal` and in the `ep_*.metal` entry-point
sources. Gate 1 of section 10.3 is the runtime check that this stays true.

### 10.5 Rung 1 prices the entry point, not only the body

`qmv_fast_crossrow_affine4_g64_wide` is a `METAL_FUNC`, so the shipped
`switch (ntg.x)` inlines every live width into one entry point that allocates
registers for the widest inlined body. A per-width body census cannot see a
change that costs registers at one width only, and every per-cell probe this
campaign has run measures the body alone.

`research/e123_arms.py --entrypoint-census` is added. It compiles the
all-widths inlined form and reports registers, spill, machine text and
concurrent simdgroups on both architectures. Residency is
`floor(budget / registers)` with `budget` = 3072 on `applegpu_g16s` and 3968 on
`applegpu_g17s`. Those two constants are fitted, and they are the unique values
that reproduce all twelve cells of Alphonse's E121 table.

The fit is confirmed out of sample by this experiment's own entry-point census:
`a_base` reports 101 registers and 39 simdgroups on `applegpu_g17s`, which is
his row for the same arm, produced by a different generator.

Every rung-1 arm that would be built is reported with body census AND
entry-point registers, spill and simdgroups on `applegpu_g17s`, and the
+1.0 % build bar is applied to the arm including its entry-point cost.

Already measured before the session, so recorded here rather than presented as
a result later:

| entry point | g16s R/spill/sg | g17s R/spill/sg |
| --- | --- | --- |
| `a_base` | 94 / 0 / 32 | 101 / 0 / 39 |
| `n_nosums` | 95 / 0 / 32 | 99 / 0 / 40 |
| `n_halfsums_free` | 95 / 0 / 32 | 101 / 0 / 39 |
| `x_cvtshift` | 96 / 192 / 32 | 114 / 0 / 34 |

`x_cvtshift` costs the ranked entry point 5 of 39 simdgroups, about -13 %, and
spills 192 bytes locally. **Pre-registered verdict: the hand-written shift form
of the bf16 widening is rejected on the entry-point census alone, whatever its
per-cell time turns out to be.**

Also recorded: on `applegpu_g17s` neither ungated `x_sumshare` form spills, at
120 and 121 registers. E118's -31 % NA=5 collapse is a `g16s` artifact and does
not transfer to ranked hardware.

### 10.6 Rung 2 build discipline for NA=5

Thorfinn traced 45 NA=5 exactness failures to compile-time `K` and `N`
template arguments, which fully unroll the ten-iteration k-loop. At NA=5 only,
the output is wrong: 174,072 of 174,080 elements differ, `max_abs_diff`
4501.3125, and the positive control fires normally. Runtime `K` and `N` fix it.
NA=5 is the only width holding `vec<float,5>` in `acc[4]`, `partial[4]`,
`a0..a3` and `sums` at once.

For any rung-2 build: suspect a compile-time `in_vec_size` or a `#pragma
unroll` on the k-loop at NA=5 before suspecting the arithmetic, and separate
the two by holding the arm fixed and changing only whether `K` is a
compile-time constant, at K=5120 and at K=512.

This also explains E118's retraction that a clean spill census is not an
exactness proof. The census reads the compiled function; the fault is in what
the compiler keeps live across the unrolled copies, so the census cannot see it
by construction.

### 10.7 Arms and artifacts after both amendments

36 arms, from 32 in section 0. Added: `n_halfsums_free` (deletion rung),
`k_cal0`, `k_ld4`, `k_shuf4` (additivity diagnostics). Static artifacts
regenerated before the session: `census.json`, `aircheck.json` and the new
`entrypoint-census.json`. Every rung contrast in the AIR check still matches
its injected count, and `q_scaffold` and `ctl_a_base_via_m` are still
byte-identical to `a_base` at every width on both architectures.

