# E123 - completing the instruction price ladder, and pricing a deletion with it

```text
SENPAI-RESULT: {"terminal":true,"status":"failed","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e123_ladder_out_of_sample_prediction_error_pp","available":true,"value":0.2908},"test_metric":{"name":"positive_control_failures","available":true,"value":0}}
```

- Student / branch: `qwen-askeladd` /
  `qwen-askeladd/e123-price-ladder-completion-and-deletion-audit`
- **Decision on the registered question: the kill rule fired. The ladder is
  reported as failing its out-of-sample test, and rung 2 was not run.** The
  primary metric improved on its baseline, 0.291 pp against 0.66 pp, but the
  kill rule is written on the maximum held-out error, not the median, and
  `k_hold_sl` missed by 2.780 pp at NA=4.
- **What the failure is worth.** The failure is bounded and diagnosed. The
  ladder predicts the four-class threadgroup mix to 0.291 pp and a pure ALU
  arm to 0.118 pp. It fails only when device loads and simd shuffles appear in
  the same block. Two free predictions made from Alphonse's E121 arms alone
  landed inside gate. Three campaign beliefs are corrected below, and one of
  them, the injection-to-deletion ratio, turns out not to be a single number
  at all.
- `BASE_SHA`: `61ed64fe02346bd1fc021f1c664a9cd2c67286c4`
- Candidate commit: this branch head. **No candidate file changed.**
- Yukon promoted submission / frontier: unchanged. This experiment proposes no
  submission and no shipped-file diff.
- Candidate build fingerprint: not applicable. No worker was built and no model
  was held. Every arm is a standalone Metal entry point compiled by the probe.
- Submitted-surface / twin / metallib digests: unchanged, none touched.
- Submitted candidate files: **none**. Alphonse owns
  `quantized.h` and `mlx-generated/quantized.cpp`; this experiment did not
  touch either.
- Supporting research files: `research/e123_arms.py`, `research/e123_probe.sh`,
  `research/e123_analysis.py`, `research/e123_wandb_log.py`,
  `research/e123-artifacts/`, and one generalisation of
  `research/e118_wandb_check.py` so it can check either logger.
- Instrument: `research/e118_qmv_probe.m`, **unchanged from E118**. Reusing the
  instrument byte for byte is what makes the E118 comparison in section 6 a
  measurement of the session and not of the harness.
- MTP head provenance and draft policy: not applicable. This probe runs no
  session and proposes no draft.
- Token window, fixture, reference source: not applicable to a microbenchmark.
  Operands are synthetic. The reference is an exact affine-4 evaluation in
  double on the CPU.
- **Rule 34 labels, which apply to every table in this report:**
  `harness=local` (standalone Metal probe on this Mac, not the benchmark
  wrapper) and `round frame = local standing campaign weights`
  `{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}`. Every percentage is a
  within-kernel percentage of `a_base`, never a round share and never a score.
- **Honesty flags, verbatim from `research/e123-artifacts/meta.txt`:**
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `timing_valid=false`, `official_or_ranked_score=false`. Entry GPU
  temperature 35.38 C, exit 70.52 C. Arms are ABBA counterbalanced inside one
  session, so monotone thermal drift cancels to first order.
- Exact cell: `qmv_fast_crossrow_affine4_g64_wide`
  (`quantized.h:969-1065`), five scored shapes, widths NA 2, 3, 4 and 5, entry
  points `e118_iso_na2..na5`, JIT source form, local `applegpu_g16s` measured
  and ranked `applegpu_g17s` statically translated.
- Official causal path and score equation: none is claimed. No local ratio here
  is presented as a ranked term.
- Scored-path reachability: the arms are transcriptions of the shipped
  `quantized.h` inner loop into private entry points. This is a screen, not an
  end-to-end measurement, and no number here is a score.

## Preflight

- `senpai/verify-ranked-score-boundary.sh` -> **PASS** ("ranked numerator is
  pinned baseline; candidate edits affect the MTP denominator only").
- `senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` ->
  **OK**. `source=2556068/3000000` bytes, headroom 443932,
  `growth=101233/262144`, 154 files.
- Scope: `git diff --stat 61ed64fe HEAD` touches `research/` only.
- W&B: four runs published and read back with
  `research/e118_wandb_check.py --verify --experiment e123` -> **VERIFY_OK**,
  all four `state=finished`.

## Reproduction

```bash
# static, both architectures, before any timing (Rule 56)
python3 research/e123_arms.py --emit /tmp/e123-arms
python3 research/e123_arms.py --census /tmp/e123-arms \
    --out research/e123-artifacts/census.json
python3 research/e123_arms.py --entrypoint-census /tmp/e123-arms \
    --out research/e123-artifacts/entrypoint-census.json
python3 research/e123_arms.py --aircheck /tmp/e123-arms \
    --out research/e123-artifacts/aircheck.json

# the timed session, 8 minutes 38 seconds wall clock
bash research/e123_probe.sh e123-full --shapes 0,1,2,3,4 \
    --widths 2,3,4,5 --pairs 8 --samples 24

python3 research/e123_analysis.py \
    --rate research/e123-artifacts/rate.json \
    --census research/e123-artifacts/census.json \
    --entrypoint research/e123-artifacts/entrypoint-census.json \
    --out research/e123-artifacts/summary.json
python3 research/e123_wandb_log.py
python3 research/e118_wandb_check.py --verify --experiment e123
```

Runtime 520.8 s for the timed leg. Peak memory is the probe's own allocation
only; no model is held. 36 arms, 5 shapes, 4 widths, 8 ABBA pairs, 24 samples.

## W&B

| run | id | url |
| --- | --- | --- |
| `e123-ladder` | `e123ladd1` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e123ladd1 |
| `e123-static-budget` | `e123stat1` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e123stat1 |
| `e123-deletion-price` | `e123del01` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e123del01 |
| `e123-rung1-audit` | `e123rng11` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e123rng11 |

Group `e123-instruction-price-ladder-and-priced-deletion-audit`.

## Validity gates

All three pre-registered gates (section 10.3) passed, so the session stands.

| gate | limit | measured | verdict |
| --- | --- | --- | --- |
| implied read bandwidth, harness defect 22 | <= 327.6 GB/s | 253.8 GB/s | pass |
| null scaffold `q_scaffold`, any width | <= 0.50 % | +0.020 / +0.210 / -0.000 / -0.071 % | pass |
| positive controls | 0 failures | 0 failures, 0 excused | pass |

`q_scaffold` is `a_base` with the injection scaffold present and zero
instructions injected. Its NA=4 value of -0.000 % is the strongest single
statement in the session: the scaffold itself is free, so every rung contrast
prices its class and nothing else.

Defect 19 per-block dispersion: 0 flagged cells at the 1.5x threshold, worst
cell spread 7.567 %. Defect 16 forward-reverse counterbalancing is retained;
the per-arm gap table is in the W&B run.

## Primary metric

`e123_ladder_out_of_sample_prediction_error_pp` = **0.291** against a baseline
of 0.66, direction minimize. It is the median of the three held-out absolute
errors at NA=4, exactly as pre-registered in section 5.

| held-out arm | mix | NA2 | NA3 | **NA4** | error NA4 |
| --- | --- | ---: | ---: | ---: | ---: |
| `k_hold_mix` predicted | 8 `tgst` + 8 `tgld` + 2 `bar` + 4 `alu` | +0.244 | +3.766 | +5.692 | |
| `k_hold_mix` measured | | +0.411 | +6.451 | +5.983 | **0.291 pp** |
| `k_hold_alu12` predicted | 12 `alu` | +0.071 | +0.929 | +1.569 | |
| `k_hold_alu12` measured | | -0.032 | +0.831 | +1.452 | **0.118 pp** |
| `k_hold_sl` predicted | 4 `shuf` + 4 `ld` | +0.143 | +4.367 | +4.773 | |
| `k_hold_sl` measured | | +0.850 | +5.251 | +7.553 | **2.780 pp** |

**The kill rule fired.** The assignment states that any held-out arm missed by
more than +/- 1.0 pp stops the experiment and is reported as a failure.
`k_hold_sl` misses by 2.780 pp at NA=4 and `k_hold_mix` misses by 2.686 pp at
NA=3. Rung 2 was therefore not run, even though the rung 1 census in section 7
nominally clears its +1.0 % bar.

Every prediction was written into
`research/e123-artifacts/preregistration.md` and committed at `63604656`,
before the timed session at `339ca4f3`. Nothing was fitted.

## Secondary metrics

All at NA=4 on `applegpu_g16s`, in percent per instruction per k-block.

| metric | value |
| --- | ---: |
| `e123_threadgroup_load_pct_per_instr_per_kblock_na4` | **0.4044** |
| `e123_threadgroup_store_pct_per_instr_per_kblock_na4` | **0.0934** |
| `e123_barrier_pct_per_barrier_per_kblock_na4` | **0.0836** |
| `e123_bf16_to_float_conversion_pct_per_instr_per_kblock_na4` | **-0.2514** |
| `e123_largest_predicted_bit_exact_deletion_pct_round_weighted` | **12.497** |

The conversion metric is negative. It is defined in section 3 as
`p_cvt - p_tgld`, the two arms differing only in access width and the
widening, and that difference came out negative. Section 4 explains why, and
why the correct reading is not "conversion is free money" but "a narrow
threadgroup access is worth more than the widening costs".

## 1. Rung 0: the completed ladder

Percent per instruction per k-block. `spill` means the pre-registered spill
rule dropped a rung on the timed architecture, and fewer than two rungs
survived.

| class | NA2 | NA3 | **NA4** | NA5 | pre-registered NA4 | verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `alu` | 0.0408 | 0.0748 | 0.1350 | spill | (E118) | |
| `ld` | 0.0196 | 0.1863 | 0.5240 | spill | (E118) | |
| `shuf` | 0.3843 | 1.1560 | 1.0209 | spill | (E118) | |
| `tgld` threadgroup load, conflict free | -0.0075 | 0.1481 | **0.4044** | spill | 0.45 [0.25, 0.70] | **in band** |
| `tgldc` threadgroup load, 4-way conflicted | -0.0126 | 0.1408 | **0.2612** | spill | 0.80 [0.45, 1.40] | MISS |
| `tgst` threadgroup store | 0.0137 | 0.0869 | **0.0934** | spill | 0.38 [0.20, 0.60] | MISS, low |
| `bar` `threadgroup_barrier`, 8 barriers | 0.1652 | 0.0851 | **0.0836** | spill | 0.15 [0.05, 0.40] | **in band** |
| `bar_hi` `threadgroup_barrier`, 16 barriers | 0.0948 | 0.1840 | **0.1642** | spill | (not registered) | |
| `sbar` `simdgroup_barrier` | 0.0371 | -0.0205 | **-0.1351** | spill | 0.00 [-0.05, 0.05] | MISS, negative |
| `cvt` bf16 threadgroup load plus widen | 0.0993 | 0.1738 | **0.1530** | spill | 0.47 [0.27, 0.72] | MISS, low |
| `ssum` `simd_sum` | 0.0427 | 0.2650 | **0.6236** | spill | 0.20 [0.05, 0.60] | MISS, high |

Two of the seven pre-registered point predictions landed inside their band.
That is a poor score for the predictions and a good outcome for the campaign:
five of the seven priors about Apple GPU threadgroup behaviour were wrong in a
direction that changes design advice.

### Shape stability

The pooled price is the median over five scored shapes of the per-shape
median. Per-shape prices at NA=4:

| class | fa_qkv | gdn_in_proj | lm_head | mlp_down | mlp_gate_up | pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `alu` | 0.1175 | 0.1307 | 0.1386 | 0.0934 | 0.1387 | 0.1350 |
| `ld` | 0.5036 | 0.5323 | 0.5409 | 0.4311 | 0.5304 | 0.5240 |
| `shuf` | 0.9880 | 1.0247 | 1.0460 | 0.9448 | 1.0090 | 1.0209 |
| `tgld` | 0.3943 | 0.4365 | 0.4169 | 0.4340 | 0.3875 | 0.4044 |
| `tgldc` | 0.2468 | 0.2505 | 0.2444 | **0.0899** | 0.2679 | 0.2612 |
| `tgst` | 0.0921 | 0.0964 | 0.1013 | 0.1382 | 0.0934 | 0.0934 |
| `cvt` | 0.1369 | 0.1494 | 0.1635 | 0.1654 | 0.1530 | 0.1530 |
| `bar` | 0.1083 | 0.1118 | 0.0761 | 0.0891 | 0.0824 | 0.0836 |
| `sbar` | -0.1103 | -0.1206 | -0.1341 | -0.0862 | -0.1521 | -0.1351 |
| `ssum` | 0.6065 | 0.5458 | 0.6011 | 0.7252 | 0.6236 | 0.6236 |

Every class holds its sign and rough magnitude across all five shapes, so no
conclusion here rests on one shape. `tgldc` on `mlp_down` is the one outlier,
0.0899 against about 0.25 elsewhere; `mlp_down` is the only shape with
`k=17408` rather than `k=5120`, so it runs many more k-blocks per output row.
That single cell is flagged and not used for any claim.

## 2. The bank-conflict prediction is wrong, and wrong the other way

`tgldc` reads `tgf[((simd_lid * 4) + n) & 127]`. Lane `i` and lane `i+8` land
on the same bank of a 32-bank file, so every access is 4-way conflicted by the
textbook model. Pre-registered `tgldc / tgld` = 1.8, band [1.3, 3.0].

Measured **0.646**. The conflicted pattern is cheaper than the conflict-free
one. The harness smoke run at a single cell gave 0.567 independently, so this
reproduces.

I cannot resolve the microarchitecture from a microbenchmark and I will not
invent one. What the measurement supports is the design instruction:
**a 4-way conflicted threadgroup read costs no more than a conflict-free read
on this hardware at this access width, so padding or swizzling a threadgroup
exchange array buys nothing and costs address arithmetic.** The two candidate
explanations worth a later test are that the threadgroup crossbar coalesces
the 8 distinct addresses of the conflicted pattern into fewer wider
transactions, and that the `* 4` shift folds into the addressing mode while
the `& 31` of the conflict-free form does not.

## 3. `simdgroup_barrier` is free, and so are barriers with nothing to order

`sbar` prices at **-0.1351** per barrier: the arm with 8 `simdgroup_barrier`
calls is *faster* than the identical arm without them. `k_bar8`, eight bare
`threadgroup_barrier` calls with no threadgroup traffic at all, measures
**-0.341 pp** against `a_base`, which is outside its pre-registered +/- 0.3 pp
band on the same side.

Both point the same way. A barrier with nothing to order is removed or merged
by the backend, and a `simdgroup_barrier` is a scheduling hint with no
execution cost that in this loop appears to help the scheduler rather than
hinder it. The practical instruction is that if only simdgroup-wide ordering
is needed, a `threadgroup_barrier` is a real cost that can be avoided.

The real barrier price comes from `k_barst8 - k_tgst8`, where the store
count cancels exactly and only the barriers remain: **0.0836** per barrier at
8 barriers and **0.1642** at 16. The barrier price rises with barrier count.

## 4. bf16 staging beats float staging

`cvt` and `tgld` differ in exactly two things: `cvt` reads 2 bytes per lane
instead of 4, and it applies `static_cast<float>`. The byte stride is held
identical by construction, so the bank pattern is identical. Measured
`p_cvt - p_tgld` = **-0.2514** at NA=4 against a pre-registered [0.00, 0.09].

A widening cannot cost less than zero, so the narrower access must be worth
more than 0.2514 %/instruction/k-block, and the widening must be cheaper than
that saving. The design instruction is: **stage activations in threadgroup
memory as bf16 and widen on read, not as float.**

`x_cvtshift` answers the other half of the conversion question and was
rejected before any timing, as pre-registered in section 10.5. It replaces all
four shipped `static_cast<float>` calls with an explicit 16-bit shift, which
is bit exact for every finite value, zero, infinity and NaN payload. It is not
byte identical to `a_base`:

| arm | arch | registers | spill | text | resident simdgroups |
| --- | --- | ---: | ---: | ---: | ---: |
| `a_base` | g16s | 94 | 0 | 24942 | 32 |
| `x_cvtshift` | g16s | 96 | **192** | 23956 | 32 |
| `a_base` | g17s | 101 | 0 | 25898 | **39** |
| `x_cvtshift` | g17s | **114** | 0 | 25868 | **34** |

The static rejection was correct: the session measured `x_cvtshift` at
**+11.4 %** slower at NA=4. The compiler's `static_cast` is already the
cheapest form of this conversion.

## 5. The two free predictions, both inside gate

Section 10.2 pre-registered two predictions built only from Alphonse's two
E121 arms, with no new GPU time, and scored them against this session's
ladder.

| prediction | source | predicted | measured | error |
| --- | --- | ---: | ---: | ---: |
| one threadgroup access at NA=4 | E121 exchange cost / 4 | 0.20 | 0.2489 | **0.049 pp** |
| the `x_sumshare_min` exchange cost | `2 p_bar + 2 p_tgld + 2 p_tgst` | 1.163 pp | 0.801 pp | **0.362 pp** |

Both are inside the 1.0 pp gate. This is the strongest evidence in the
experiment that the ladder does compose across the threadgroup and barrier
classes, which is exactly the composition the campaign needs for an exchange
arm. Of the 1.163 pp only 0.167 pp is barriers; the accesses are the cost, and
`tgld` dominates `tgst` by a factor of four.

## 6. The failure, diagnosed as far as the evidence goes

### It is not a miscount

Machine text at NA=4 on `applegpu_g16s`: `k_hold_sl` is 7782 bytes. The
additive text prediction `k_ld4 + k_shuf4 - k_cal0` is
`7260 + 7516 - 7000 = 7776` bytes, and the ladder-slope prediction
`0.5 * (k_shuf8 + k_ld8)` is `0.5 * (8056 + 7492) = 7774` bytes. All three
agree to within 8 bytes, one AGX instruction. The instruction counts add.

### It is not occupancy, and this rejects my own pre-registered explanation

Amendment 1, written after the smoke run and before this session, predicted
that register residency would explain the miss. It does not. At NA=4 on
`applegpu_g16s`:

| arm | registers | spill | resident simdgroups |
| --- | ---: | ---: | ---: |
| `a_base` | 94 | 0 | 32 |
| `k_ld4` | 95 | 0 | 32 |
| `k_shuf4` | 95 | 0 | 32 |
| `k_ld8` | 92 | 0 | 33 |
| `k_shuf8` | 94 | 0 | 32 |
| `k_hold_sl` | 96 | 0 | 32 |

`k_hold_sl` holds the same 32 resident simdgroups as `k_shuf4`, `k_shuf8` and
`k_ld4`. A 3 % occupancy difference against `k_ld8` cannot produce a 58 %
excess. **The pre-registered residency explanation is rejected.**

### It is a genuine cross-class interaction

Two independent predictions, one using the ladder slope and one using only
directly measured arms at the holdout's own operation count, both under-predict
by about the same amount:

| NA | `k_cal0` | `k_ld4` | `k_shuf4` | additive prediction | measured | excess |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | +0.177 | -0.209 | +0.108 | -0.277 | +0.850 | **+1.127 pp** |
| 3 | +0.069 | +1.011 | +3.869 | +4.811 | +5.251 | +0.440 pp |
| 4 | +0.031 | +1.477 | +3.480 | +4.926 | +7.553 | **+2.627 pp** |

The cleanest statement of the effect needs no model at all:

```text
cost(8 loads)                 = +2.901 %
cost(8 shuffles)              = +6.644 %
cost(4 loads + 4 shuffles)    = +7.553 %
```

Replacing four shuffles with four loads, which are three times cheaper on their
own, makes the block *slower*. Nonlinearity in count is the wrong sign to
explain this: the marginal price of both classes rises from 4 to 8 to 16
instructions, so a ladder slope taken from the 8-to-16 rung should
over-predict a 4-plus-4 mix, not under-predict it.

I do not have a proven mechanism. The candidate worth one cheap follow-up is
that the `shuf` template overwrites its accumulator, `cal[j] = simd_shuffle(cal[j], ...)`,
so a pure shuffle arm is a serial chain, while the `ld` template only
accumulates, `cal[j] += ...`, so its loads issue independently. Mixing them may
force the scheduler to hold four in-flight load results across a serial shuffle
chain. The named test is one arm with the same 4 plus 4 mix spread over four
accumulator chains instead of two.

### The boundary this puts on the ladder

Use the ladder for a single-class change, and for a mix of threadgroup,
barrier and ALU work, where it predicts to 0.291 pp. **Do not use it to price a
change that mixes device loads with simd shuffles** until that interaction is
measured directly.

## 7. Rung 1: the priced instruction census

Computed, but computed after the kill rule fired, and read in that light.

Per-width values are `count(NA) * price(class, NA)` as a percentage of
`a_base`. `weighted` uses the standing round weights. NA=5 has no prices on the
timed architecture, so every NA=5 cell is unpriced and the weighted column
covers 0.966 of the round weight.

| group | class | deletable | NA2 % | NA3 % | NA4 % | weighted |
| --- | --- | :-: | ---: | ---: | ---: | ---: |
| lane FMAs | `alu` | c | 5.224 | 14.359 | 34.557 | 27.124 |
| nibble integer operations | `alu` | **a** | 4.571 | 8.376 | 15.119 | **12.497** |
| activation widenings | `cvt` | b | 3.179 | 8.344 | 9.789 | 8.901 |
| sums add tree | `alu` | **a** | 1.632 | 4.487 | 10.799 | **8.476** |
| nibble integer to float | `alu` | c | 2.612 | 4.786 | 8.639 | 7.141 |
| activation register moves | `alu` | **a** | 1.306 | 3.590 | 8.639 | **6.781** |
| weight element loads | `ld` | c | 0.314 | 2.981 | 8.385 | 6.420 |
| activation vec4 loads | `ld` | c | 0.157 | 2.236 | 8.385 | 6.211 |
| final accumulate | `alu` | c | 0.979 | 2.692 | 6.479 | 5.086 |
| metadata loads | `ld` | c | 0.157 | 1.491 | 4.192 | 3.210 |
| metadata widenings | `cvt` | c | 0.795 | 1.391 | 1.224 | 1.218 |
| epilogue `simd_sum` | `ssum` | c | 0.034 | 0.318 | 0.998 | 0.754 |

Classification: **(a)** bit-exact deletable, **(b)** deletable only with a
numerical change, so a ceiling and not shippable, **(c)** not deletable.

Entry-point census, all widths inlined into one kernel as the shipped entry
point presents them:

| arm | arch | registers | spill | text | resident simdgroups |
| --- | --- | ---: | ---: | ---: | ---: |
| `a_base` | g16s | 94 | 0 | 24942 | 32 |
| `n_nosums` | g16s | 95 | 0 | 22710 | 32 |
| `n_halfsums_free` | g16s | 95 | 0 | 24108 | 32 |
| `x_cvtshift` | g16s | 96 | 192 | 23956 | 32 |
| `a_base` | g17s | 101 | 0 | 25898 | 39 |
| `n_nosums` | g17s | 99 | 0 | 23608 | **40** |
| `n_halfsums_free` | g17s | 101 | 0 | 25042 | 39 |
| `x_cvtshift` | g17s | 114 | 0 | 25868 | 34 |

Deleting the whole sums tree buys one extra resident simdgroup on the ranked
architecture. Deleting half of it buys none. The ranked architecture is more
register hungry than the local one at the same source, so a change that is
register neutral on g16s can still cost residency on g17s.

### Why the ranking must not be acted on as written

**The census over-predicts a real deletion by 2.16x.** The session measures the
same kernel actually losing the first half of its sums tree, so the census can
be checked instead of trusted:

| NA | instructions deleted | census predicts | session measures | over-prediction |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 14.8 | +0.603 % | +0.073 % | 8.22x |
| 3 | 15.8 | +1.178 % | +0.599 % | 1.97x |
| 4 | 33.9 | +4.581 % | +2.117 % | **2.16x** |

**The census over-reconstructs the whole kernel.** Summing every group gives
21.0 %, 55.1 % and 117.2 % of `a_base` at NA=2, 3 and 4. Section 7 of the
pre-registration expected [0.55, 0.85] at NA=4. The measured 1.172 is outside
that band on the high side: the priced census claims to account for 117 % of a
kernel that is 100 % of itself.

So the nominal top-ranked bit-exact deletion, 12.497 % round weighted for the
nibble integer operations, clears the +1.0 % rung 2 bar by a wide margin and is
still not a number to build on. Two independent checks say it is optimistic by
roughly a factor of two, and `DIRECT_NIBBLES` is already a template parameter,
so that group is not an unexplored opportunity. The `sums add tree` row is
worse than it looks: only half of it is bit-exact deletable, and that half is
`x_sumshare_min`, which Alphonse has already measured.

Rung 2 was not run. The kill rule forbids it, and this section is why the kill
rule is right.

## 8. Injection price against deletion price: there is no single ratio

Section 10.1 pre-registered one contrast, `n_halfsums_free -> n_nosums`, and
predicted `R` = 1.35 with band [1.05, 1.90]. Measured **1.116** at NA=4, inside
the band and inside the [0.90, 1.15] unity band, which by the pre-registered
rule sets the rung 1 divisor to 1.0.

**That headline is fragile, and I report the fragility rather than the
headline alone.** The same three arms support three contrasts. Deletion price
in percent per instruction per k-block:

| contrast | NA2 | NA3 | **NA4** | NA5 | NA4 ratio to injection |
| --- | ---: | ---: | ---: | ---: | ---: |
| first half, `a_base -> n_halfsums_free` | 0.0050 | 0.0380 | 0.0624 | 0.0782 | **2.164** |
| second half, `n_halfsums_free -> n_nosums` (pre-registered) | 0.0046 | 0.0105 | **0.1209** | 0.0955 | **1.116** |
| whole tree, `a_base -> n_nosums` | 0.0047 | 0.0182 | 0.0951 | 0.0891 | **1.420** |

The last 43 instructions of the add tree are worth twice as much as the first
34. The pre-registered contrast happens to be the one that gives unity, and I
would have reported a very different conclusion had I registered either of the
other two. The honest statement is:

**The injection-to-deletion ratio is not a property of the instruction class.
It depends on which instructions you delete. It ranges from 1.12 to 2.16 for
one add tree in one kernel at one width.**

This also revises what I told Alphonse in the interim comment on this PR. I
wrote there that his cross-session 1.66 "does not survive a within-session
measurement". That is too strong. His 1.66 came from the *first-half* contrast
priced with E118's `p_alu`; the same first-half contrast priced with this
session's `p_alu` gives 2.16, and the whole-tree contrast gives 1.42. His
number was not mainly a session artifact. It was mostly this same
nonlinearity, and the correction is stated in the follow-up comment.

The instruction count behind every deletion price is read from machine text at
8.25 bytes per AGX instruction, because AIR keeps the `m` loop rolled and the
AIR delta for `n_halfsums_free` is zero by construction. The text-derived
counts agree with source accounting: 38.1 / 56.7 / 76.8 / 98.9 against
`20 * NA` = 40 / 60 / 80 / 100 for the whole tree, and 23.3 / 41.0 / 42.9 /
62.3 against `20 * halfsums_kept(NA)` = 20 / 40 / 40 / 60 for the second half.

**Correction, disclosed.** The first draft of pre-registration section 10.1
printed that second row as `20 | 20 | 40 | 40`. The correct row is
`20 * halfsums_kept(NA)`. The slip was in the prose only; the generator, the
census and the arm code always used `halfsums_kept`. It was fixed before any
timing data existed and the fix is in this branch.

## 9. E118 does not fully reproduce on the same host

Same chip, same architecture, same entry points, same unmodified instrument,
same width, same five shapes. Only the session differs.

| class | E118 | E123 | drift |
| --- | ---: | ---: | ---: |
| `p_alu` | 0.09398 | 0.13499 | **+43.6 %** |
| `p_ld` | 0.58330 | 0.52404 | -10.2 % |
| `p_shuf` | 0.96486 | 1.02095 | +5.8 % |

The two expensive classes reproduce within about 10 %. The cheapest class
drifts by 44 %. `p_alu` carries the smallest signal, 8 instructions times
0.135 % is a little over 1 %, so it is the most exposed to session-level
differences, but a 44 % drift is far larger than this session's own dispersion
and it is the price that the whole rung 1 census leans on hardest.

**Consequence for the campaign: no cross-session instruction price comparison
in this campaign should be trusted to better than about 40 % on the ALU class.**
That alone accounts for most of the distance between Alphonse's 1.66 and the
numbers in section 8.

## 10. Fidelity, spill and exactness

155 exact-match failures, all attributable to spilling and all excluded from
pricing by the pre-registered spill rule:

- At NA=5 on `applegpu_g16s`, every arm that reaches 96 registers spills 176 to
  256 bytes and produces wrong results on all five shapes. That is the E118
  spill defect, reproduced.
- At NA=4, exactly two arms spill and exactly those two fail: `k_tgld16`
  (128 bytes) and `k_tgldc16` (144 bytes).
- `k_cvt16` spills 16 bytes at NA=4 and stays exact.

The separation is clean: **largest spill while exact 16 bytes, smallest spill
while wrong 128 bytes.** Spill bytes predict exactness failure with no overlap,
which is why the pre-registered rule drops a spilling rung rather than a whole
width. `k_hold_mix` at NA=3 spills 16 bytes and is exact, and is retained.

Positive controls: 0 failures, 0 excused. The controls can detect an injected
fault at every cell that was priced.

## 11. What I would do next, and did not do

1. **Settle the load-plus-shuffle interaction.** One arm: the same 4 `shuf` plus
   4 `ld` mix spread over four accumulator chains instead of two, plus one arm
   with the loads emitted before the shuffles. Two arms, one session, and it
   separates "serial shuffle chain holding load results" from "genuine port
   contention". Until then the ladder carries the section 6 boundary.
2. **Re-measure `p_alu` twice more.** A 44 % cross-session drift on the price
   the census leans on hardest is a bigger problem for the campaign than the
   held-out miss. Three sessions on the same host, same day, would bound it.
3. **Price the bank-conflict result properly.** `tgldc` cheaper than `tgld`
   reproduces, but at one access width and one stride. A 2-way and an 8-way
   variant, plus a 2-byte conflicted variant, would say whether the crossbar
   coalesces or the addressing mode is free.
4. **Give the exchange arm the bf16 reader.** Section 4 predicts about
   0.25 %/access at NA=4 for switching the read side of a threadgroup exchange
   from float to bf16. Alphonse owns that surface; I did not touch it.
5. **Retire the reconstruction target.** Section 7 of the pre-registration
   expected the census to reconstruct 55 to 85 % of the kernel. It reconstructs
   117 % at NA=4. A priced census built from injection prices is not a
   conserved decomposition, and a future experiment should say what it is
   before ranking anything with it.
