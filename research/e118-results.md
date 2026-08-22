# E118 - the metadata-load instruction axis of the wide affine-4 QMV

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e118_best_bit_exact_arm_round_weighted_pct_faster_vs_a_base","available":true,"value":0.3364},"test_metric":{"name":"positive_control_failures","available":true,"value":0}}
```

- Student / branch: `qwen-askeladd` /
  `qwen-askeladd/e118-wide-qmv-inner-loop-load-instruction-screen`
- Hypothesis and target cost: the eight scalar metadata reads in the
  `qmv_fast_crossrow_affine4_g64_wide` inner loop occupy the load-issue port,
  so removing or coalescing them makes the kernel faster. The target cost is
  Finding 44: `a_base` runs 15.92 % (round weighted, measured here) above its
  own load-only ceiling `l_loadonly`.
- Decision on the registered question: **dead**. The primary metric is
  `+0.3364 %`, the kill rule is `+0.5 %`, so the metadata-instruction axis is
  handed back.
- **The headline is not that null.** The headline is a predictive instruction
  cost model for this kernel, measured in the same session as the arms it
  prices, that predicts every screen arm with no free parameter and identifies
  the binding resource as **total instruction issue**. Section 1 is that model.
  The arm screen is section 2.
- `BASE_SHA`: `1d2320bece29cddc94b95e5f99f00331b05a5025`
- Candidate commit: this branch head. **No candidate file changed.**
- Yukon promoted submission / frontier: unchanged. This experiment proposes no
  submission and no shipped-file diff.
- Candidate build fingerprint: not applicable. No worker was built and no model
  was held. Every arm is a standalone Metal entry point compiled by the probe.
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
  evaluation in double on the CPU.
- **Rule 34 labels, which apply to every table in this report:**
  `harness=local` (standalone Metal probe on this Mac, not the benchmark
  wrapper) and `round frame = local standing campaign weights`
  `{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}`. Every percentage is a
  within-kernel percentage of `a_base`, never a round share and never a score.
- Exact cell: `qmv_fast_crossrow_affine4_g64_wide`, five scored shapes, widths
  NA 2, 3, 4 and 5, entry points `e118_iso_na2..na5`, JIT source form, local
  `applegpu_g16s` measured and ranked `applegpu_g17s` statically translated.
- Official causal path and score equation: none is claimed. No local ratio
  here is presented as a ranked term.
- Scored-path reachability: the arms are transcriptions of the shipped
  `quantized.h` inner loop into private entry points. **Finding 28 says that
  for the `quantized` family the metallib is dead and only the worker binary
  carries the arm. That does not apply to a probe I compile myself, and I say
  so explicitly so nobody misreads these numbers as end-to-end. This is a
  screen, not an end-to-end measurement.**

## Preflight

- `senpai/verify-ranked-score-boundary.sh` → **PASS** ("ranked numerator is
  pinned baseline; candidate edits affect the MTP denominator only").
- `senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` →
  `source=2607365/3000000 headroom=392635` **`growth=152530/262144`**
  `exempt=2410 files=155`. *(An earlier revision of this report printed growth
  as `0/262144`. That was wrong and is corrected here.)*
- `senpai/validate-assignment-scope.sh` takes `BASE_SHA` followed by the
  **submitted** candidate paths. There are none, so with no paths it exits 2
  with nothing to check. Run over the full research diff it exits 1 and names
  every `research/` path as outside `editablePaths`, which is expected:
  `research/` is not a scored surface. The diff against `BASE_SHA` is:

  ```
  research/e118-artifacts/{arms.log,census.json,e114_receipt_slice.json,
                           meta.txt,probe.log,rate.json,summary.json}
  research/e118-results.md
  research/e118_analysis.py  research/e118_arms.py
  research/e118_probe.sh     research/e118_qmv_probe.m
  research/e118_wandb_log.py
  ```

  None of the four forbidden files is modified:
  `Vendor/.../kernels/quantized.h`, `Vendor/.../mlx-generated/quantized.cpp`,
  `Vendor/.../Models/Qwen35.swift`,
  `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`.

## Evidence

- Host: `ip-10-231-2-227.ec2.internal`, Apple M4 Pro, `applegpu_g16s`, 20-core
  GPU, 48 GiB. Swift 6.3.3, Metal 32023.883. Fast math **off** in the probe,
  in the table-fill kernel and in the census, so no arm is allowed to
  reassociate.
- Thermal policy: entry 36.05 C, exit 71.78 C.
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `timing_valid=false` in `meta.txt`, preserved verbatim. Arms run in
  palindrome (ABBA) order inside one session. **These numbers are directional
  causal evidence within one counterbalanced session. They are not
  gate-qualified and they are not a score.**
- Exact command, one session, 405 s:

  ```bash
  bash research/e118_probe.sh e118-full5 \
    --shapes 0,1,2,3,4 --widths 2,3,4,5 --pairs 8 --samples 24
  python3 research/e118_arms.py --emit /tmp/e118-emitG
  python3 research/e118_arms.py --census /tmp/e118-emitG \
    --out research/e118-artifacts/census.json
  python3 research/e118_analysis.py \
    --rate research/e118-artifacts/rate.json \
    --census research/e118-artifacts/census.json \
    --slice research/e118-artifacts/e114_receipt_slice.json \
    --out research/e118-artifacts/summary.json
  ```

  Every number in this report is read from a **committed** path under
  `research/e118-artifacts/` (CAMPAIGN RULE 40). Nothing here is read from
  `research/out/`, which is host-local.
- Provenance honesty: `meta.txt` records `git_head=41c2a5e5` and
  `git_dirty=0`, and the probe writes both **after** the run. The tree at
  launch was commit `949eb59c` plus one uncommitted edit to
  `research/e118_wandb_log.py`, which the probe does not read. `meta.txt` also
  pins a SHA-256 of every arm's generated Metal source, so the measured
  sources are fixed independently of the commit.
- 30 timed arms x 5 shapes x 4 widths x 8 counterbalanced blocks. Block 0 is
  discarded, 7 kept, giving **600 cell-arm series**.
- Falsification gate and positive controls: **0 control failures** over the
  whole session. Two controls fire on every bit-exact-required arm (one
  perturbed activation, one perturbed whole metadata record covering `scales`,
  `biases`, `packed_sb` and `bias_codes` together) and a third fires on
  `x_sumshoist` only (a perturbed sums-table slab). `q_scaffold` and
  `ctl_a_base_via_m` compile to machine text byte-identical to `a_base` on
  both architectures, so they measure the harness noise floor.
- **Noise floor, stated before any result:** the byte-identical null control
  `q_scaffold` reads `+0.092 %` round weighted on `mlp.gate_up` and ranges
  `-0.291 %` (`mlp.down`) to `+0.092 %` (`mlp.gate_up`) across the five shapes.
  Any arm inside +/- 0.3 % is inside that control's own scatter.
- Exact-token and row-ledger verdict: not applicable. No session ran.

---

# 1. HEADLINE - a predictive instruction cost model for this kernel

`harness=local`, round frame = local standing weights. This answers feedback 1
section 4, and it is the most transferable thing this experiment produced.

## 1.1 The instrument

Six calibration arms inject a known number of instructions of one class into
the inner loop: `k_alu8` / `k_alu16`, `k_ld8` / `k_ld16`, `k_shuf8` /
`k_shuf16`. The **price is the 8-to-16 rung contrast**, which cancels the
injection scaffold exactly. The scaffold column shows what was cancelled, and
the `3pt` column is the biased fit that anchors on `a_base` at zero, printed
so the bias is visible rather than hidden.

| class | %/instr per k-block | sem | us/instr, median | R2 median | scaffold cancelled | biased 3pt fit | cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ld` | **0.33423** | 0.12342 | 1.4405 | 0.954 | -1.285 % | 0.24512 | 15 |
| `alu` | **0.07097** | 0.00825 | 0.1996 | 0.947 | +0.419 % | 0.10817 | 20 |
| `shuf` | **0.96486** | 0.09394 | 2.3934 | 0.994 | -1.886 % | 0.91675 | 15 |

**A `simd_shuffle` costs 2.9 device loads and 13.6 ALU operations on this
part.** That single line kills the whole broadcast family, and it is the
quantitative form of ADVISOR ERROR 78 in section 2.4.

## 1.2 The single biggest finding: the price is not constant in NA

| class | NA2 | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: | ---: |
| `ld` %/instr | 0.10216 | 0.32853 | **0.58330** | - |
| `alu` %/instr | 0.00430 | 0.06759 | **0.09398** | 0.07929 |
| `shuf` %/instr | 0.37763 | 1.06566 | **0.96486** | - |

`-` marks a width where every rung of that class spilled, so no price exists;
those cells are never pooled away. An instruction is **nearly free at NA=2 and
expensive at NA=4**: the load price rises 5.7x from NA=2 to NA=4 and the ALU
price rises 22x. Any experiment that prices an instruction at one width and
weights it at another will be wrong by a factor of several. NA=4 carries 0.667
of the round, so NA=4 is the price that matters.

## 1.3 What the ladder is reading: issue throughput, not latency

ILP control, 16 injected ALU instructions either way, only the dependency
structure differs:

```
2 chains of 8   cost 1.731 %
4 chains of 4   cost 1.950 %
difference      +0.219 pp over 20 cells
```

Halving the dependency depth at constant instruction count **did not reduce
the cost**. If the kernel were chain-latency-bound the 4-chain form would be
cheaper. It is not. **The ladder reads issue throughput.**

## 1.4 The model predicts the screen arms with no free parameter

Each screen arm's delta in load count and shuffle count is read from the AIR
census, multiplied by the per-width prices in 1.2, and compared with the
measurement. Nothing is fitted to the screen arms.

| arm | d_ld | d_shuf | NA2 pred/meas | NA3 pred/meas | NA4 pred/meas | wtd pred/meas | cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s_bcast` | 0 | +8 | -3.02 / -2.47 | spills | **-7.72 / -15.32** | -7.56 / -14.87 | 0.691 |
| `s_bcast_all` | -6 | +8 | -2.41 / -0.41 | -6.55 / -7.84 | **-4.22 / -6.90** | -4.84 / -7.00 | 0.966 |
| `s_bcast_pack32` | -4 | +4 | -1.10 / -0.36 | spills | -1.53 / -7.61 | -1.51 / -7.35 | 0.691 |
| `s_bcast_scale` | 0 | +4 | -1.51 / -0.35 | spills | spills | -1.51 / -0.35 | 0.024 |
| `p_split_meta` | 0 | 0 | -0.00 / +0.03 | -0.00 / -0.08 | **-0.00 / +0.01** | 0.00 / -0.01 | 0.966 |
| `g_pack32` | -4 | 0 | +0.41 / +0.22 | +1.31 / +0.42 | **+2.33 / +0.15** | +2.00 / +0.23 | 0.966 |

`cov` is the share of the standing round weight the weighted columns speak
for; two arms with different `cov` are not comparable on that column.

Read this honestly. The model gets **`p_split_meta` exactly right** (a true
null by construction, byte-identical text), gets **`s_bcast_all` right to
within 2.2 pp**, and gets the **sign right on all six**. It fails in two
specific, informative ways:

1. It **under-predicts the shuffle arms by roughly 2x** at NA=4. Eight
   shuffles predict -7.72 % and measure -15.32 %. The extra cost is not in the
   instruction count: `s_bcast` allocates 95 registers at NA=2 against
   `a_base`'s 70, and 117 of 124 at NA=4 on `applegpu_g17s`. A shuffle
   lengthens live ranges as well as consuming a slot, and the ladder prices
   only the slot.
2. It **over-predicts `g_pack32` by 10x**: removing four of seven device loads
   should be worth +2.33 % at NA=4 and is worth +0.15 %. This is the tension I
   flagged in the first interim comment, and section 2.3 resolves it with the
   AIR census: the loads `g_pack32` removes are already-coalesced scalar reads
   that were never the constraint.

Both failures point the same way: **the constraint is total issue and register
pressure, not the load count.**

## 1.5 The observational AIR regression that feedback 1 asked for, and it fails

Regressing microseconds on static AIR counts across every arm and every cell:

```
univariate on issue lanes : 0.1750 us/instruction (se 0.0720), R2 median 0.146 over 15 cells
multivariate              : R2 median 0.749 over 15 cells
```

| AIR category | us/instruction | se |
| --- | ---: | ---: |
| `device_loads` | **-1.6656** | 1.4197 |
| `shuffles` | +1.6464 | 0.5854 |
| `arithmetic_lanes` | +0.2837 | 0.3946 |
| `convert` | +1.2224 | 0.9742 |
| `address` | -0.9820 | 0.7924 |

**A negative coefficient on device loads is nonsense**, and it appears because
the arms do not vary the categories independently: every arm that removes
loads also removes arithmetic. Worst univariate residuals on `mlp.gate_up`
NA=4, in microseconds: `s_bcast` +77.5, `l_loadonly` -63.8, `k_shuf16` +47.3,
`s_bcast_pack32` +39.0, `x_sumshare_min` -33.8 - each many slots wide.

I report this failure rather than hiding it, because it is the quantitative
justification for building the ladder at all, and it is a direct
**confirmation of Finding 36**: ISA text size and spill predict time, AIR
operation counts do not.

## 1.6 The AIR caveat that constrains every static number in this report

The AIR the census reads has **rolled loops**. A static AIR total counts a loop
body once, not once per iteration, so it can move in the opposite direction to
executed work. The clearest case in this session is `x_sumshoist`: its static
AIR total **rises** 242 -> 249 at NA=4 while its executed work **falls** by
about 80 instructions per k-block, and it measures +6.911 % faster. Static AIR
is used here for *counting distinct instructions in the source-level body*,
never as a proxy for executed instruction volume.

---

# 2. The registered screen: the metadata-instruction axis is a null

## 2.1 Primary metric

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `e118_best_bit_exact_arm_round_weighted_pct_faster_vs_a_base` | 0.0 | **+0.3364** | +0.3364 |

Best bit-exact metadata-load arm: **`g_pack32`**. Kill rule `+0.5 %`:
**NOT CLEARED**. `harness=local`, round frame = standing weights
`{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}`, cell `mlp.gate_up` K=5120 N=34816.

The metric ranks the **metadata-load arms only**. `p_prefetch_w` and `e_bias6`
are bit exact but carry other mechanisms, so they are reported beside it and
never inside it. `x_sumshoist` (section 5) is bit exact and reads +5.376 %,
and it is deliberately **not** in this metric: it is a differently scoped,
later, unshippable ceiling arm, and folding it into a screen it was not part of
would corrupt the screen's verdict. I committed to that in the interim comment
before the full session ran and I am holding to it.

**Weighting robustness, and an honest correction.** Under the E114
identified-set bounds `g_pack32` is `[+0.240, +0.538]` locally and
`[+0.240, +0.539]` ranked. In the earlier partial session this interval was
`[+0.127, +0.441]` and I wrote that the whole identified set lay below the kill
rule. **With the full session that is no longer true**: the single most
favourable admissible weighting reaches `+0.538 %` and just clears `+0.5 %`.
The null is therefore robust under the standing weights and under the great
majority of the identified set, **but not under its most favourable corner**.
I am still reporting a null, for three reasons that do not depend on the
weighting: the effect is inside the null control's cross-shape scatter
(-0.291 % to +0.092 %); it is the **third** independent reading of this arm and
the second null; and the cost model over-predicts it by 10x, so there is no
mechanism left to bank on.

## 2.2 Discriminator verdict, in the assignment's words

```text
s_bcast -16.589   s_bcast_all -7.379   p_split_meta +0.078   n_nosums +5.861
```

`s_bcast` loses badly, `p_split_meta` is exactly null, and only `n_nosums`
wins. On the assignment's own table that reads:

> **The binding resource is total instruction issue or ALU. It is not the
> load-issue port and it is not memory latency. E110's issue-slot reading is
> right and the metadata route is dead.**

This holds on **all five shapes** with no sign reversal (section 3.3).

## 2.3 The AIR census that resolves `g_pack32` against `s_bcast_all`

This is feedback 1 section 3. Device loads per **entry point** after `-O2`,
identical at every width unless stated:

| arm | AIR device loads (+ shuffles) |
| --- | --- |
| `a_base`, `q_scaffold`, `p_split_meta`, `n_nosums`, `l_loadonly`, `d_bias1`, `e_bias6`, `z_ballast`, `k_alu*`, `y_algebra`, `y_hsum_tree`, `x_sumshare_min` | **7** |
| `g_pack32`, `n_nobias` | **6** |
| `p_prefetch_w`, `x_sumshoist` (NA<=4) | 8 |
| `x_sumshoist` (NA=5) | 9 |
| `s_bcast`, `s_bcast_all` | 7 + 2 shuffles |
| `s_bcast_scale` | 7 + 1 shuffle |
| `s_bcast_pack32` | 6 + 1 shuffle |
| `x_sumshare_owner` | 11 |
| `k_ld8`, `n_halfsums`, `x_sumshare_split` | 15 |
| `k_ld16` | 23 |

**The shipped kernel already issues 7 device loads, not 15.** The eight scalar
metadata reads the hypothesis targeted are **already coalesced by the front
end** before any arm touches them. That single fact explains the whole screen:

- `s_bcast` removes **zero** loads (still 7) and only adds two shuffles. It
  cannot win, and it loses 16.6 %.
- `s_bcast_all` also still shows 7, so the +41.4 us it recovers over `s_bcast`
  is the price of **predication**, not the saving of six loads. That settles
  the tension I flagged in the first interim comment, and it settles it against
  the load-removal reading.
- `g_pack32` genuinely removes one of seven and gains +0.34 %, an order of
  magnitude less than the ladder predicts for four removed instructions.
- `p_split_meta` compiles to byte-identical text and measures -0.01 %. Second
  null control, and it behaves.

## 2.4 ADVISOR ERROR 78, recorded as asked

The advisor raised the campaign prior on `s_bcast` from an external
microbenchmark reporting `simd_shuffle` at ~1-2 cycles against expensive
scattered threadgroup access. This session measures a shuffle at **0.96486 %
of `a_base` per instruction per k-block, 2.9x a device load and 13.6x an ALU
operation**, and `s_bcast` at **-16.589 %** round weighted. The transferred
prior is falsified on this part. Recorded here as **ADVISOR ERROR 78**, an
advisor error, not a student negative result. The standing rule stands: an
external microbenchmark may raise a prior, never a verdict.

## 2.5 `s_bcast_all`, the unrequested arm, named and kept

`s_bcast` as specified predicates eight metadata loads down to eight active
lanes, which in a SIMT machine removes lanes, not instructions. Because
`rows_per_simd * groups_per_k_block == 4 * 8 == 32 == SIMD_SIZE`, one **fully
active** load can fetch every metadata word the simdgroup needs for all four
rows at once. `s_bcast_all` does that: lane `j` takes row `out_row + j / 8` and
group offset `j % 8`; lane `L` reads row `r`'s word out of lane `r * 8 + L / 4`.
No predication, no out-of-range index, bit exact, all controls fire.

Decomposition on `mlp.gate_up` NA=4, absolute microseconds against
`a_base` = 500.3:

```
s_bcast        576.5   +76.2 us    8 shuffles + 6 predicated loads
s_bcast_all    535.1   +34.8 us    8 shuffles, fully active loads
s_bcast_scale  538.8   +38.5 us    4 shuffles, half the fields
```

Removing the six predicated loads is worth **+41.4 us**; the eight shuffles
that remain still cost **+34.8 us** against `a_base`; and halving the field
count costs **+38.5 us**, against half of `s_bcast`'s `+76.2 us` = 38.1 us.
The linearity in field count is what makes the decomposition credible, and the
ladder in section 1.4 reproduces `s_bcast_all` to within 2.2 pp.

---

# 3. Every arm, every width, every shape

## 3.1 `mlp.gate_up`, percent faster than `a_base` (positive = faster)

`harness=local`. Median over 7 kept blocks, sem in brackets.

| arm | role | NA2 | NA3 | NA4 | NA5 |
| --- | --- | ---: | ---: | ---: | ---: |
| `l_loadonly` | diagnostic | +1.554 (0.267) | +3.700 (0.037) | +16.755 (0.289) | +28.909 (0.105) |
| `n_nobias` | diagnostic | +5.630 (0.240) | +5.618 (0.020) | +9.060 (0.121) | +12.024 (0.055) |
| `n_nosums` | diagnostic | +0.099 (0.269) | +1.873 (0.045) | +7.518 (0.138) | +9.680 (0.073) |
| `x_sumshoist` | hoist ceiling | +0.502 (0.312) | +1.592 (0.028) | +6.911 (0.109) | +9.321 (0.069) |
| `n_halfsums_free` | diagnostic | +0.351 (0.301) | +1.343 (0.030) | +2.266 (0.112) | +3.726 (0.118) |
| `y_hsum_tree` | diagnostic | +0.291 (0.308) | -0.334 (0.068) | +1.480 (0.129) | +0.663 (0.111) |
| `x_sumshare_min` | rung2, bit exact | +0.785 (0.289) | +0.337 (0.059) | +1.465 (0.104) | -31.318 (0.096) |
| `g_pack32` | **promotion** | +0.194 (0.311) | +0.554 (0.110) | +0.248 (0.114) | +0.420 (0.070) |
| `q_scaffold` | null control | -0.141 (0.202) | -0.059 (0.056) | +0.155 (0.073) | +0.248 (0.077) |
| `p_split_meta` | **promotion** | -0.047 (0.382) | -0.077 (0.069) | +0.144 (0.145) | +0.103 (0.082) |
| `n_halfsums` | diagnostic | +0.039 (0.333) | +0.656 (0.049) | -0.409 (0.129) | -0.140 (0.088) |
| `d_bias1` | diagnostic | +2.409 (0.221) | -0.664 (0.045) | -0.398 (0.110) | -1.010 (0.070) |
| `k_alu8` | ladder | +0.084 (0.178) | -1.268 (0.065) | -0.956 (0.135) | -3.157 (0.102) |
| `y_algebra` | diagnostic | +0.289 (0.309) | -1.377 (0.104) | -1.619 (0.143) | -2.786 (0.078) |
| `k_alu16` | ladder | +0.172 (0.186) | -1.861 (0.050) | -1.753 (0.141) | -3.746 (0.076) |
| `x_sumshare_split` | rung2, bit exact | +0.494 (0.318) | -0.645 (0.073) | -2.372 (0.129) | -3.262 (0.108) |
| `k_alu16w` | ladder ILP | +0.546 (0.216) | -1.500 (0.066) | -2.101 (0.141) | -3.869 (0.102) |
| `p_prefetch_w` | beside metric | +0.362 (0.258) | +0.299 (0.040) | +0.029 (0.161) | -63.222 (0.152) |
| `x_sumshare_owner` | rung2, bit exact | +0.445 (0.290) | +0.004 (0.076) | -3.157 (0.117) | -5.313 (0.070) |
| `e_bias6` | beside metric | +2.040 (0.169) | -4.365 (0.066) | -3.041 (0.124) | -1.837 (0.097) |
| `z_ballast` | spill control | +0.346 (0.179) | -2.043 (0.034) | -3.467 (0.120) | -68.851 (0.117) |
| `k_ld8` | ladder | +0.608 (0.187) | -1.247 (0.040) | -4.222 (0.141) | -84.542 (0.121) |
| `k_shuf8` | ladder | +0.297 (0.184) | -7.057 (0.111) | -7.794 (0.177) | -7.075 (0.125) |
| `s_bcast_all` | **promotion** | -0.126 (0.333) | -8.070 (0.035) | -7.090 (0.072) | -12.580 (0.074) |
| `s_bcast_scale` | **promotion** | +0.050 (0.359) | -8.835 (0.018) | -7.709 (0.116) | -6.162 (0.058) |
| `s_bcast_pack32` | **promotion** | +0.236 (0.256) | -6.034 (0.025) | -7.258 (0.128) | -59.436 (0.097) |
| `k_ld16` | ladder | +0.200 (0.186) | -3.921 (0.038) | -8.855 (0.194) | -93.363 (0.123) |
| `k_shuf16` | ladder | -2.483 (0.179) | -15.582 (0.105) | -15.623 (0.166) | -14.509 (0.070) |
| `s_bcast` | **promotion** | -1.843 (0.248) | -15.250 (0.058) | -15.316 (0.069) | -62.792 (0.109) |

Absolute `a_base` microseconds per dispatch on `mlp.gate_up`: **414.5, 431.0,
500.3, 594.7** at NA 2, 3, 4, 5.

Every NA=5 cell below about -30 % is a **spilling** measurement, not a
mechanism result. Section 8 proves that with a control.

## 3.2 Round weighted, with the E114 identified-set bounds

The assignment requires the same number under the E114 identified-set bounds so
the ranking cannot depend on one chosen weight vector.
`research/e118_analysis.py` imports `research/scoring_weights.py` (unmodified)
and sweeps every admissible per-NA weight vector the four board facts per
prompt allow. The interval is the range over that whole set, **not** a
confidence interval.

| arm | role | standing | identified local | identified ranked |
| --- | --- | ---: | ---: | ---: |
| `l_loadonly` | diagnostic | +13.213 | [ +4.360, +24.412] | [ +4.329, +24.512] |
| `n_nobias` | diagnostic | +8.132 | [ +5.798, +10.927] | [ +5.789, +10.951] |
| `n_nosums` | diagnostic | +5.861 | [ +2.149, +8.880] | [ +2.137, +8.898] |
| `x_sumshoist` | hoist ceiling | **+5.376** | [ +1.858, +8.429] | [ +1.846, +8.449] |
| `n_halfsums_free` | diagnostic | +2.016 | [ +1.381, +3.186] | [ +1.380, +3.198] |
| `y_hsum_tree` | diagnostic | +0.925 | [ -0.239, +1.380] | [ -0.243, +1.382] |
| `g_pack32` | **promotion** | **+0.336** | **[ +0.240, +0.538]** | **[ +0.240, +0.539]** |
| `q_scaffold` | null control | +0.092 | [ -0.049, +0.214] | [ -0.050, +0.214] |
| `p_split_meta` | **promotion** | +0.078 | [ -0.065, +0.128] | [ -0.066, +0.129] |
| `x_sumshare_min` | rung2 | +0.024 | [-19.188, +1.407] | [-19.457, +1.409] |
| `n_halfsums` | diagnostic | -0.096 | [ -0.372, +0.601] | [ -0.373, +0.603] |
| `d_bias1` | diagnostic | -0.425 | [ -0.784, -0.016] | [ -0.789, -0.024] |
| `k_alu8` | ladder | -1.092 | [ -2.343, -0.815] | [ -2.361, -0.818] |
| `y_algebra` | diagnostic | -1.546 | [ -2.354, -1.323] | [ -2.364, -1.328] |
| `k_alu16` | ladder | -1.804 | [ -3.009, -1.491] | [ -3.025, -1.496] |
| `x_sumshare_split` | rung2 | -1.858 | [ -2.932, -0.724] | [ -2.940, -0.720] |
| `k_alu16w` | ladder ILP | -1.932 | [ -3.215, -1.485] | [ -3.229, -1.487] |
| `p_prefetch_w` | beside metric | -2.040 | [-39.817, +0.285] | [-40.337, +0.286] |
| `x_sumshare_owner` | rung2 | -2.275 | [ -4.515, -0.157] | [ -4.533, -0.149] |
| `e_bias6` | beside metric | -3.242 | [ -4.296, -2.199] | [ -4.299, -2.201] |
| `z_ballast` | spill control | -5.207 | [-44.657, -2.080] | [-45.194, -2.080] |
| `k_ld8` | ladder | -6.019 | [-54.822, -1.384] | [-55.481, -1.378] |
| `k_shuf8` | ladder | -7.373 | [ -7.665, -6.693] | [ -7.659, -6.716] |
| `s_bcast_all` | **promotion** | -7.379 | [-10.549, -6.143] | [-10.594, -6.162] |
| `s_bcast_scale` | **promotion** | -7.780 | [ -8.776, -6.545] | [ -8.779, -6.550] |
| `s_bcast_pack32` | **promotion** | -8.516 | [-40.129, -5.921] | [-40.557, -5.935] |
| `k_ld16` | ladder | -10.154 | [-62.093, -4.136] | [-62.787, -4.127] |
| `k_shuf16` | ladder | -15.258 | [-15.616, -13.835] | [-15.615, -13.872] |
| `s_bcast` | **promotion** | -16.589 | [-45.225, -13.483] | [-45.615, -13.521] |

The arm ranking is identical under the local and the ranked weight family and
no promotion arm changes sign anywhere in the set. The four E114 point
weightings (`maxent`, `gt1`, `gt2`, `policy`) are in `summary.json` and are
**diagnostic only**: every one of them failed E114's own rung-0 gate.

## 3.3 Every shape agrees

Round-weighted percent, all five scored shapes, `harness=local`:

| arm | fa.qkv | gdn.in_proj | lm_head | mlp.down | mlp.gate_up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `l_loadonly` | +14.495 | +14.040 | +12.036 | +20.955 | +13.213 |
| `n_nobias` | +7.724 | +7.845 | +8.265 | +7.422 | +8.132 |
| `n_nosums` | +5.528 | +5.576 | +5.836 | +5.649 | +5.861 |
| `x_sumshoist` | +4.886 | +5.160 | +5.547 | +4.643 | +5.376 |
| `n_halfsums_free` | +1.695 | +1.828 | +1.997 | +1.662 | +2.016 |
| `g_pack32` | +0.024 | +0.142 | +0.238 | +0.687 | +0.336 |
| `q_scaffold` (null) | -0.198 | -0.003 | +0.040 | -0.291 | +0.092 |
| `p_split_meta` | -0.027 | -0.032 | +0.059 | -0.256 | +0.078 |
| `x_sumshare_min` | -0.047 | +0.109 | -0.018 | +2.103 | +0.024 |
| `s_bcast_all` | -7.208 | -7.152 | -7.532 | -7.181 | -7.379 |
| `s_bcast` | -16.407 | -16.325 | -16.894 | -17.015 | -16.589 |

No shape reverses any sign on any arm that matters. **`g_pack32` is at or below
the null control's own scatter on two of five shapes.** The screen is not a
one-shape artefact and neither is the null.

## 3.4 Finding 44 placement, `mlp.gate_up`

| NA | `a_base` us | `l_loadonly` us | gap % |
| ---: | ---: | ---: | ---: |
| 2 | 414.5 | 407.4 | 1.73 |
| 3 | 431.0 | 414.7 | 3.92 |
| 4 | 500.3 | 416.5 | **20.13** |
| 5 | 594.7 | 422.9 | **40.60** |

Round-weighted gap **+15.92 %**, against Finding 44's **+17.3 %** on a
different host and probe. Same direction, same order, and I place it on the
advisor's table rather than quoting it as a match. The headroom above the load
ceiling is real and grows steeply with width - and this experiment shows it is
**not reachable by touching the loads**.

---

# 4. Standing contradictions resolved

## 4.1 E111 against E104 on `n_nosums`, resolved in E111's favour

E111 measured `n_nosums` at **+6.132 %**; E104 measured **-4.47 %**; the
campaign rule since then has been to price nothing from either until a third
independent reading exists. This is that reading.

**`n_nosums` = +5.861 % round weighted on `mlp.gate_up`, and +5.528 to
+5.836 % on the other four shapes, at every width, in one counterbalanced
session with a byte-identical null control at +0.092 %.** It agrees with E111
in sign and to within 0.3 pp in size, and it contradicts E104 in sign.
**E111 is right and E104's `n_nosums` reading is superseded.**

The advisor asked whether `n_nosums` is crediting itself with the bias load,
which would retro-correct both numbers. **It is not.** AIR device loads per
entry point:

| arm | NA2 | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: | ---: |
| `a_base` | 7 | 7 | 7 | 7 |
| `n_nosums` | **7** | **7** | **7** | **7** |
| `n_nobias` | 6 | 6 | 6 | 6 |

`n_nosums` keeps the load because E111 deliberately kept `bias_local` live in
its epilogue, and the count proves the optimiser did not get it. `n_nobias`
drops the epilogue term and the count falls to 6, which is the control that
makes the 7 mean something. Total AIR instructions fall 242 -> 232 at NA=4,
which is the add tree and the `sums * bias_local[r]` product and nothing else.
**Neither +5.861 % nor E111's +6.132 % needs retro-correcting.**

## 4.2 E104 arm P is closed

`p_prefetch_w` - double-buffering the `packed[rows_per_simd][4]` weight loads
across k iterations, the classic software pipeline - had never been measured on
any host.

| NA | g16s R/spill/text | g17s R/spill/text | measured vs `a_base` |
| ---: | --- | --- | ---: |
| 2 | 82/0/4404 | 83/0/4672 | +0.362 % |
| 3 | 93/0/5662 | 98/0/5978 | +0.299 % |
| 4 | 94/0/6948 | 92/0/7244 | **+0.029 %** |
| 5 | **96/192/8502** | **100/0/8506** | -63.222 % (spilling) |

At NA=2 and NA=3, where g16s has headroom, the mechanism produces **nothing**
(+0.36 %, +0.30 %, both inside the null control's scatter). At NA=4, where it
fits with no spill on either arch, it is **+0.029 %**, exactly null. The NA=5
number is a spilling measurement on g16s and is transfer-invalid: the ranked
`applegpu_g17s` allocates 100 of 124 registers there with **zero spill**, so
the local NA=5 failure does not transfer.

**E104 arm P produces no gain at any width where it fits, on either
architecture's register budget. Software pipelining of this loop is closed.**

## 4.3 `g_pack32`, third reading, stop-listed

E111 read `+0.334 +/- 0.165 %` on `mlp.gate_up` NA=4 (t = +5.73) and
`+0.04 +/- 0.91 %` on `mlp.down`. This third reading gives **+0.248 %** at
`mlp.gate_up` NA=4 and **+0.336 %** round weighted, against a null control that
ranges to -0.291 %, and the cost model over-predicts the arm by 10x.
**Null on its third independent reading. Stop-listed.**

## 4.4 The E111 bias axis, folded in at every width

E111 measured these at NA=5 only, which carries 0.034 of the standing weight.

| quantity | NA2 | NA3 | NA4 | NA5 | round weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n_nobias`, whole bias axis | +5.630 | +5.618 | +9.060 | +12.024 | **+8.132** |
| `n_nosums`, arithmetic only | +0.099 | +1.873 | +7.518 | +9.680 | +5.861 |
| difference, **the bias load** | +5.531 | +3.745 | +1.542 | +2.343 | **+2.271** |
| `d_bias1`, Bias6 ceiling | +2.409 | -0.664 | -0.398 | -1.010 | -0.425 |
| `e_bias6`, real, bit exact | +2.040 | -4.365 | -3.041 | -1.837 | -3.242 |
| difference, reconstruction | +0.368 | +3.701 | +2.643 | +0.827 | **+2.818** |

`e_bias6` is bit exact at every cell above; the two ceiling rows are
deliberately wrong and price ceilings only. The whole bias axis is worth
`+8.13 %`, but only `+2.27 %` of that is the load and the rest is arithmetic.
The one-byte code cannot collect even the load part: its own ceiling `d_bias1`
is already **negative** (-0.425 %) once the whole width range is weighted, and
the exact reconstruction costs a further +2.818 %. **Bias6 is a loss on this
host at every width above NA=2.**

## 4.5 The `sums` chain is not reassociable, and the algebraic route costs time

Static answer, from the emitted AIR. `sums[m] += xm[0] + xm[1] + xm[2] + xm[3]`
compiles to **three scalar `fadd bfloat`, strictly left-associated** (`%139`
feeds `%140` feeds `%141`), then one `fpext` and one float `fadd`. It is not
vectorised. Every horizontal or tree form reassociates, so with fast math off
**there is no bit-exact horizontal replacement for this chain.**

- `y_hsum_tree` rebalances the chain at the same instruction count and reads
  **+0.925 %** round weighted, **+1.480 %** at NA=4. So the chain is partly
  latency-bound - but it is **not bit exact** and is therefore a **diagnostic
  ceiling only**.
- `y_algebra`, the Finding-40 algebraic route, has an identical `fmul` count
  and reads **-1.546 %**. Rewriting the expression **costs** time. That axis is
  closed.
- Mechanism arithmetic: 5 instructions per (i, m) gives 4 * NA * 5 = **80 per
  k-block at NA=4**. The ladder's NA=4 ALU price (0.09398 %/instr) predicts
  **+7.52 %** for removing all 80 - and `n_nosums`, which removes exactly
  those, measures **+7.518 %**. This is the strongest validation of the cost
  model in this report: an independently calibrated instrument predicting an
  arm it never saw. **The agreement to three decimals is luck, not precision.**
  Propagating the ALU price's sem through 80 instructions gives a +/- 0.66 pp
  band, so the honest claim is agreement well inside one standard error.

---

# 5. Feedback 3: the whole-table `sums` hoist, and what it actually costs

`x_sumshoist` replaces the entire `sums` accumulation with one table load. The
advisor's arithmetic is right: the `sums` value depends only on `x`, `k`,
`simd_lid` and `i`, so for `mlp.gate_up` at N=34816 it is recomputed
`4352 threadgroups x 2 simdgroups = 8704` times when one computation would do.

**This arm is not shippable from `research/` and I am not presenting it as an
arm.** It binds a tenth buffer and the host-side binding lives in
`quantized.cpp`, which is not editable. It is a ceiling measurement for a
delivery decision, and it is deliberately outside the primary metric.

## 5.1 Bit exactness over 4.47 million outputs, with a control that reaches the table

The table is `[k_block][lane][m]` with `m` fastest and the per-lane stride
padded to 8 floats, so the layout is identical across NA and every slab is
32-byte aligned. 10,240 B at K=5120; 34,816 B for `mlp.down` at K=17408. It is
filled by a Metal kernel that writes the advisor's expression verbatim, so the
three inner adds happen in `bfloat16_t` and only the accumulation into `s` is
float. No host-side bf16 emulation is involved.

**`x_sumshoist` is bit-identical to `a_base` at all 20 cells - 5 shapes x 4
widths - over 4,467,008 outputs, `differing=0`, `max_ulp=0`, including NA=5**,
where nine other arms are wrong.

Two existing controls cannot prove the table load is live, because perturbing
an activation leaves the table holding the unperturbed value. A third control
perturbs one slab entry, for this arm only:

```
mlp.gate_up  control x_sumshoist  x_hit=24470   meta_hit=16  table_hit=34812   restored_diff=0
gdn.in_proj  control x_sumshoist  x_hit=10113   meta_hit=16  table_hit=16480   restored_diff=0
fa.qkv       control x_sumshoist  x_hit=11758   meta_hit=16  table_hit=14301   restored_diff=0
mlp.down     control x_sumshoist  x_hit=3440    meta_hit=16  table_hit=5120    restored_diff=0
lm_head      control x_sumshoist  x_hit=196907  meta_hit=16  table_hit=248073  restored_diff=0
```

All three fire on every shape and every one restores to zero.

## 5.2 The two contrasts the advisor asked for

`mlp.gate_up`, percent faster than `a_base`:

| NA | `x_sumshoist` | `n_nosums` (free ceiling) | capture | hoist - nosums = the load price |
| ---: | ---: | ---: | ---: | ---: |
| 2 | +0.502 | +0.099 | 5.05 | **+0.403** |
| 3 | +1.592 | +1.873 | 0.850 | -0.281 |
| 4 | **+6.911** | +7.518 | 0.919 | **-0.608** |
| 5 | +9.321 | +9.680 | 0.963 | -0.359 |
| **round weighted** | **+5.376** | +5.861 | **0.917** | **-0.485** |

**The load is almost free: the hoist captures 91.7 % of the free ceiling, and
the whole price of the table load is -0.485 pp round weighted.** That is the
second requested contrast, and section 4.1 shows `n_nosums` is an honest
ceiling because it keeps its bias load.

## 5.3 Static budget: it *reduces* registers, as predicted

`R / spill bytes / machine text bytes`:

| arch | arm | NA2 | NA3 | NA4 | NA5 |
| --- | --- | --- | --- | --- | --- |
| g16s | `a_base` | 70/0/4430 | 93/0/5682 | 94/0/6920 | 95/0/8228 |
| g16s | `x_sumshoist` | 78/0/4230 | 90/0/5350 | **91/0/6476** | 96/0/7654 |
| g17s | `a_base` | 83/0/4644 | 90/0/5900 | 91/0/7206 | 98/0/8492 |
| g17s | `x_sumshoist` | 94/0/4442 | 90/0/5570 | **92/0/6732** | 98/0/7882 |

No spill at any width on either arch. Registers fall 94 -> 91 on g16s at NA=4
and machine text falls 6.4 %. AIR device loads go 7 -> 8 at NA <= 4 (one
`vec<float,4>`) and 7 -> **9** at NA=5. **Form used: `vec<float,4>` plus one
scalar load**, because `vec<float,5>` has `sizeof` and `alignof` 32 and cannot
overlay a packed 5-float slab. It costs one extra load instruction at NA=5
only. Static AIR total rises 242 -> 249 at NA<=4 and 253 at NA=5, which is the
rolled-loop artefact of section 1.6, not extra executed work.

## 5.4 The number the advisor most needs: table production is **not** free

The timed +5.376 % **excludes table production**. The probe now times the fill
dispatch separately, warm, minimum of five, so it can be subtracted rather than
taken on trust. The fill is one command buffer: encode, commit,
`waitUntilCompleted`.

| shape | table B | fill us | `a_base` us NA4 | `x_sumshoist` us NA4 | saving us | **net % incl. fill** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp.gate_up` | 10240 | **117.0** | 500.3 | 465.2 | 35.1 | **-16.37** |
| `lm_head` | 10240 | 161.2 | 3385.3 | 3139.8 | 245.6 | **+2.49** |
| `mlp.down` | 34816 | 217.5 | 297.1 | 279.0 | 18.1 | -67.10 |
| `gdn.in_proj` | 10240 | 428.4 | 248.1 | 231.0 | 17.1 | -165.83 |
| `fa.qkv` | 10240 | 452.1 | 220.1 | 206.1 | 14.0 | -199.02 |

Round-weighted net on `mlp.gate_up` including the fill: **-18.98 %**.

**Read that carefully, because the honest conclusion is narrower than the table
looks.**

- The same 10,240-byte fill measures 117 us on one shape and 452 us on
  another. The fill is therefore **not measuring fill work**; it is measuring
  one command-buffer round trip. The work itself is 51.2 KB of `x` read and
  10 KB written, about **0.3 us** at this host's measured bandwidth.
- The arm timings amortise their command buffer over `reps * inner`
  dispatches; the fill does not. So `fill_us` is an **upper bound** on the
  marginal cost of a fill pipelined into the same command buffer, and my
  instrument cannot resolve how far below that bound it would land.
- What survives regardless of that noise: **a separate fill dispatch costs at
  minimum 117 us on this host, and the saving is 14 to 35 us per matvec on four
  of the five shapes.** A standalone fill kernel cannot pay for itself anywhere
  except `lm_head`, where the saving scales with N (245.6 us) and the fill does
  not.
- The layout is the reason there is still a route. `offset(kb, l)` uses only
  the k-block, the lane and `m` - **not `out_row`, not `N`, not the weight
  matrix.** So one table serves every matvec that consumes the same `x` and the
  same K, and it serves all four widths at once (the fill always writes
  `max_m = 5`). Amortising one fill over the matvecs that share an activation
  is a property of the indexing, which I state from the source; I did not
  measure it.

**Verdict for the delivery decision: the mechanism is real, bit exact, worth
+5.376 % round weighted, captures 91.7 % of its own free ceiling, and lowers
both register pressure and machine text. Whether it survives delivery turns
entirely on the per-launch cost of producing the table, which my standalone
instrument bounds at 117 us and cannot resolve below. Funding the engineering
is justified only if the fill can be fused into the kernel that already
produces `x`, so it costs one extra store and no extra dispatch.** A separate
dispatch, measured here, is a net loss on `mlp.gate_up`, `mlp.down`,
`gdn.in_proj` and `fa.qkv`.

---

# 6. Rung 2, Finding 53: the registered rule fires, and the ceiling arm is confounded

The rule Finding 53 registers is that across simdgroups you may share
**reductions**, never expansions. Every arm below shares the `sums` reduction,
which is legal under that rule; none of them expands anything.

## 6.1 The registered rule

Registered: *if `n_halfsums` < +2.5 % -> stop, report the ceiling.*
`n_halfsums` measures **-0.096 %** round weighted and **-0.409 %** at NA=4.
**The rule fires.** The advisor's priors - `n_halfsums` ~ +3.5 to +3.8 % and
`x_sumshare_split` net +2.8 to +3.4 % - are both **falsified**.

## 6.2 But the registered ceiling arm carries its own scaffolding

The uniform branch as specified duplicates the whole `i` loop. At NA=4 on g16s,
`n_halfsums` carries **1.70x the machine text of `a_base` and a 48-byte spill
where `a_base` has none**. Every rung-2 arm as registered prices the mechanism
*and* its scaffolding together. `n_halfsums_free` is the unregistered
diagnostic that removes the scaffolding: both simdgroups drop the *same* half
of the add tree at compile time, so it is equally wrong, has identical executed
instruction count per simdgroup, and has **no branch and no duplication**.

Decomposition, `mlp.gate_up`, percent faster than `a_base`:

| NA | free (mechanism) | `n_halfsums` (registered) | **dup cost** | `x_sumshare_min` | xchg cost | `x_sumshare_split` | `x_sumshare_owner` | capture |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | +0.351 | +0.039 | 0.312 | +0.785 | -0.434 | +0.494 | +0.445 | 2.236 |
| 3 | +1.343 | +0.656 | 0.687 | +0.337 | +1.006 | -0.645 | +0.004 | 0.251 |
| 4 | **+2.266** | -0.409 | **2.675** | **+1.465** | +0.801 | -2.372 | -3.157 | 0.646 |
| 5 | +3.726 | -0.140 | 3.865 | -31.318 (spills) | 35.044 | -3.262 | -5.313 | -8.406 |

**The uniform branch costs 2.675 pp at NA=4** - more than the mechanism it is
supposed to bound. That is why the registered ceiling reads negative.

Bit-exact reading of the shippable arm: `x_sumshare_min` is **+0.024 %** round
weighted under the standing weights, but that number is dominated by an NA=5
cell that **spills 160 B locally and 0 B on the ranked `applegpu_g17s`**. Over
the three widths whose local build does not spill (round-weight coverage 0.966)
the same arm is **+1.1267 %**, and at NA=4 alone it is **+1.465 % (sem
0.104)**. So the mechanism clears +1 % at the width carrying two thirds of the
round; it does not clear it under the full standing weighting, because the
local NA=5 spill drags it to zero.

## 6.3 Threadgroup memory never limits occupancy

| arm | NA | threadgroup B | threadgroups allowed by 32768 | AIR threadgroup ops |
| --- | ---: | ---: | ---: | ---: |
| `a_base`, `n_halfsums`, `n_halfsums_free` | 2-5 | 0 | unlimited | 0 |
| `x_sumshare_min`, `x_sumshare_split` | 2/3/4/5 | 512/768/1024/1280 | 64/42/32/25 | 2 |
| `x_sumshare_owner` | 2/3/4/5 | 256/384/512/640 | 128/85/64/51 | 3/2/2/2 |

Even the largest arm allows 25 concurrent threadgroups on the 32768-byte
budget. **Shared memory is never the constraint; registers and issue slots
are.**

## 6.4 Registers, spill and text for every rung-2 arm, both arches

| arm | g16s NA2/3/4/5 | g17s NA2/3/4/5 |
| --- | --- | --- |
| `a_base` | 70/0/4430, 93/0/5682, 94/0/6920, 95/0/8228 | 83/0/4644, 90/0/5900, 91/0/7206, 98/0/8492 |
| `n_halfsums` | 76/0/7242, 89/0/9276, **96/48/11760**, 96/128/14006 | 98/0/7642, 103/0/9626, **124/0/11690**, 126/48/13760 |
| `n_halfsums_free` | 71/0/4302, 82/0/5542, **94/0/6654**, 92/0/7924 | 92/0/4538, 90/0/5780, 96/0/6932, 98/0/8188 |
| `x_sumshare_min` | 72/0/4652, 85/0/5994, **90/0/7310**, 96/160/9226 | 82/0/4840, 95/0/6192, 102/0/7506, **122/0/8926** |
| `x_sumshare_split` | 74/0/7432, 92/0/9548, 96/64/12002, 96/144/14440 | 92/0/7820, 105/0/9900, 125/0/12016, 126/48/14214 |
| `x_sumshare_owner` | 72/0/7330, 90/0/9390, 96/64/11934, 96/128/14390 | 92/0/7694, 103/0/9736, 125/0/11806, 126/48/13918 |

On `applegpu_g17s` the registered arms do not spill at NA=4 but go
91 -> 124/125/125 registers of 124, so **the same defect appears there as an
occupancy hit rather than a spill**. `n_halfsums_free` and `x_sumshare_min`,
the two arms that remove the scaffolding, are the only rung-2 arms that stay
near `a_base` on both budgets.

## 6.5 Scoring my own rung-2b pre-registration, honestly

I pre-registered three predictions in comment 5377192464 before those arms ran.

1. *"`n_halfsums_free` lands at +3.5 to +4.7 % and shows no spill and text
   within ~5 % of `a_base` at NA=4."* Measured **+2.266 %** at NA=4.
   **WRONG - I over-predicted by more than a factor of 1.5.** The static half
   is **CONFIRMED**: 94/0/6654 against `a_base`'s 94/0/6920 - no spill, and
   text 3.8 % *below* `a_base`.
2. *"`n_halfsums_free` - `n_halfsums` ~ 4 to 5 pp."* Measured **2.675 pp**.
   **WRONG - over-predicted; direction right.**
3. *"`x_sumshare_min` beats `x_sumshare_split` by more than 2 pp, but I do not
   predict it clears +1.0 %."* First half **CORRECT**: +1.465 against -2.372 at
   NA=4, a gap of 3.84 pp. Second half **WRONG**: it does clear +1.0 % at NA=4.

Two of my three quantitative predictions over-shot and one of my two
directional calls was too pessimistic. The pattern is that I over-price
arithmetic removal and under-price my own scaffolding - which is exactly the
bias the ladder in section 1 was built to remove, and which it still shows
here: the ladder predicts +3.76 % for dropping 40 of 80 add-tree instructions
at NA=4 against +2.266 % measured. (It gets the *full* 80-instruction removal
right to three decimal places; see section 4.5. The half-removal is where the
scaffolding hides.)

---

# 7. `rows_per_simd = 8`: compile-only census, and the axis stays closed

Compile only, zero GPU. Both the per-width entry points and the **all-widths
inlined form** (`NA0`), which is the E102 trap the advisor flagged: `_wide` is a
`METAL_FUNC` inlined into one entry point, so the entry point allocates for the
widest inlined body and every width pays. **I censused both forms and report
both.** `c_rps4` is the control.

`R / spill bytes / text bytes`:

| arch | arm | NA0 all-widths | NA2 | NA3 | NA4 | NA5 |
| --- | --- | --- | --- | --- | --- | --- |
| g16s | `c_rps4` control | 95/0/24934 | 70/0/4476 | 93/0/5728 | 94/0/6966 | 95/0/8274 |
| g16s | `c_rps8` | 96/**224**/30318 | 96/192/8602 | 96/144/6106 | 96/**192**/7610 | 96/160/8576 |
| g17s | `c_rps4` control | 98/0/25900 | 83/0/4694 | 90/0/5950 | 91/0/7256 | 98/0/8542 |
| g17s | `c_rps8` | 126/**192**/29922 | 120/0/8640 | 126/96/6020 | 126/**176**/7474 | 126/144/8552 |

The `c_rps4` control reproduces `a_base` to within the few text bytes the
`tid.y & 1` early return costs, so the census is calibrated.

**One sentence as asked: zero spill is not reachable on `applegpu_g17s` at
NA=4 - it spills 176 bytes there, and 192 bytes in the all-widths inlined form
- so on the advisor's own stated condition the `rows_per_simd` axis stays
closed.** The only spill-free `rps8` cell anywhere is g17s at NA=2, at 120 of
124 registers. Note that `c_rps8`'s text is non-monotone in NA (8602, 6106,
7610, 8576 on g16s); that is reported as measured and I have no explanation
for it.

I made the body correct rather than merely compilable - odd `tid.y` early
returns, even ones covering 16 rows - so it could become a timing arm later. I
did not time it, because with the dispatch unhalved half the launched
threadgroups do nothing, which would price the dispatch and not the mechanism.

---

# 8. The NA=5 spill defect, and a correction to my own rung-1 claim

45 exactness failures were recorded and they are not a bug in any arm. **Every
failure is at NA=5, on all five shapes, across nine arms**: `s_bcast`,
`s_bcast_all`, `s_bcast_pack32`, `p_prefetch_w`, `z_ballast`, `k_ld8`,
`k_ld16`, `x_sumshare_split`, `x_sumshare_owner`. A typical failure is
`173877/174080` cells differing with `max_rel = 2.000`, meaning sign flips on
99.9 % of cells - **garbage, not a different valid rounding.** Fast math is
off, so reassociation does not explain it.

`z_ballast` is the decisive control. It adds twelve dead loop-carried floats
consumed only inside a branch that never executes, so **no value it computes
reaches `y`**. It changes the spill budget and nothing else. It is equally
wrong, at NA=5 only.

> **On Apple M4 Pro `applegpu_g16s` with Metal 32023.883, the wide qmv NA=5
> entry point can return numerically wrong results when the compiler spills.
> The cause is spilling, not any arm's mechanism.**

**Correction to what I reported at rung 1.** I previously wrote that spill
separates exact from wrong perfectly, with the largest exact spill at 16 B and
the smallest wrong spill at 80 B. **With the full arm set that is false:**

```
largest spill that stayed exact : 160 B  (x_sumshare_min, NA=5)
smallest spill that went wrong  :  80 B  (s_bcast_all,    NA=5)
spill separates exact from wrong: False
```

`k_shuf8` and `k_shuf16` also spill 16 B at NA=5 and stay exact, and
`s_bcast_scale` spills 16 B at NA=3, NA=4 and NA=5 and stays exact everywhere.
So spilling is **necessary but not sufficient** for the corruption on this
host: every wrong cell spills, but not every spilling cell is wrong. I am
withdrawing the clean-threshold claim and replacing it with the weaker one
above, which the `z_ballast` control still supports.

`a_base` itself does not spill at NA=5 and stays correct, so **nothing shipped
is affected today.** Transfer risk: on the ranked `applegpu_g17s` only four
arms spill at all (`s_bcast` 16 B; `n_halfsums`, `x_sumshare_split` and
`x_sumshare_owner` 48 B; all at NA=5), so the defect may be local to this
generation. That does not make it safe to ignore: a future wide-qmv change that
spills on the ranked runner would be silently wrong rather than merely slow.

---

# 9. Harness

## 9.1 Defect 16 residual: forward slot against reverse slot, per arm

Thorfinn's E115 defect inflates the first arm timed after a `macmon` sample by
a fixed 30-80 ms, which an ABBA mean does not cancel; his diagnostic was
`a_one` at **+61.6 %** while every other arm moved under 0.4 %. In this probe
the temperature sample is moved ahead of the per-arm warm-up, a 150 ms
discarded ramp burst follows every sample, and block 0 is discarded.

Residual forward-minus-reverse median gap over kept blocks, all 30 arms:

| arm | kept med % | arm | kept med % | arm | kept med % |
| --- | ---: | --- | ---: | --- | ---: |
| `a_base` | **-0.160** | `k_ld8` | -0.060 | `n_halfsums` | -0.005 |
| `e_bias6` | -0.141 | `s_bcast_all` | -0.042 | `x_sumshare_split` | -0.001 |
| `s_bcast_pack32` | -0.130 | `q_scaffold` | -0.042 | `n_nobias` | -0.000 |
| `p_split_meta` | -0.096 | `l_loadonly` | -0.039 | `k_ld16` | +0.000 |
| `k_alu8` | -0.036 | `s_bcast` | -0.031 | `n_nosums` | +0.001 |
| `g_pack32` | -0.027 | `k_alu16` | -0.022 | `x_sumshare_owner` | +0.008 |
| `k_shuf8` | -0.022 | `y_hsum_tree` | -0.022 | `x_sumshoist` | +0.014 |
| `z_ballast` | -0.022 | `p_prefetch_w` | -0.020 | `y_algebra` | +0.018 |
| `s_bcast_scale` | -0.019 | `k_shuf16` | -0.018 | `d_bias1` | +0.022 |
| `k_alu16w` | -0.008 | `n_halfsums_free` | +0.036 | `x_sumshare_min` | +0.041 |

**The largest residual on any arm is 0.160 % and it is on `a_base`**, against
thorfinn's +61.6 % before the fix - a 385x reduction - and it now runs in the
*conservative* direction: the baseline reads slightly fast, so candidates read
slightly pessimistic. The `|max|` column in `summary.json` peaks at 3.8 % on
single blocks, which is block noise, not a slot effect.

## 9.2 Defect 19 dispersion

A block is flagged when it exceeds 1.5x its cell median.

```
flagged blocks: 0 of 600 cell-arm series
worst within-cell spread over any arm: 2.94 %
```

**Zero blocks were excluded**, so no pooling decision affects any number in
this report.

## 9.3 Two harness defects found and repaired during this experiment

Both were found by disbelieving a control that passed, and both had been
weakening the exactness screen.

1. **NaN synthetic biases.** `bias_bf16_from_code` takes the negative-zero path
   when a code's low nibble is zero, and the exponent adjustment lands on the
   bf16 pattern `0x7fff`, which is NaN. One group in 64 was affected, so about
   72 % of output columns carried a NaN, and NaN compares bit-equal to NaN, so
   the screen was passing on poisoned data. The generator now excludes that
   code and the probe reports `base_nonfinite` per width. This session has
   **0 non-finite baseline elements at every one of the 20 cells**.
2. **A metadata positive control that could not fire.** It perturbed one group
   in eighty and bf16 output rounding absorbed it, so `meta_hit` was 0 for
   every arm. It now perturbs groups spread across the output and fires on
   every arm on every shape (`meta_hit=16` this session).

Every timing number in this report comes from the repaired probe.

---

# 10. W&B runs

Group `e118-wide-qmv-metadata-load-instruction-screen`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`. Every run carries
`harness=local`, `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false`.

| Run ID | Name | Contents |
| --- | --- | --- |
| [`e118arms1`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118arms1) | `e118-arms` | all 30 timed arms by width and shape, the primary metric with identified-set bounds, the discriminator, Finding 44, the defect-16 forward-reverse gap and the E111 bias axis |
| [`e118stat1`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118stat1) | `e118-static-budget` | AIR device loads and shuffles, registers, spill bytes and ISA text for `applegpu_g16s` and `applegpu_g17s` at NA 2-5, including the `rows_per_simd` census |
| [`e118spil1`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118spil1) | `e118-spill-defect` | the 45 NA=5 exactness failures, each arm against the exact double reference, and the spill-to-exactness join that carries the `z_ballast` control |
| [`e118cost1`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118cost1) | `e118-cost-model` | the instruction-price ladder, the per-width prices, the ILP control, the no-free-parameter predictions and the failed observational AIR regression |
| [`e118rng21`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118rng21) | `e118-rung2-finding53` | the Finding 53 decomposition, the registered rule's outcome, the scaffolding confound and the threadgroup-occupancy table |
| [`e118hst01`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e118hst01) | `e118-sumshoist-ceiling` | `x_sumshoist` against `a_base` and `n_nosums`, its bit-exactness ledger and controls, its static budget, and the table-production cost |

---

# 11. Conclusion

- **The binding resource is total instruction issue.** `s_bcast` removes zero
  device loads and loses 16.6 %; `p_split_meta` is byte-identical text and
  measures -0.01 %; `n_nosums` deletes arithmetic and gains +5.86 %. The
  shipped inner loop already issues 7 device loads, not 15, because the front
  end coalesces the eight scalar metadata reads. **There is no load-issue
  pressure left to remove, on all five shapes, at all four widths.**
- **The slope, which is the more useful half.** A device load costs 0.334 % of
  `a_base` per instruction per k-block, an ALU operation 0.071 %, and a
  `simd_shuffle` 0.965 % - and the price rises about 5.7x (loads) and 22x (ALU)
  from NA=2 to NA=4. The model predicts the sign of every screen arm with **no
  free parameter**, and predicts `n_nosums` - an arm it never saw - at
  +7.52 % against +7.518 % measured at NA=4, well inside its +/- 0.66 pp band.
  Its two failures, shuffle arms 2x too cheap and `g_pack32` 10x too expensive,
  both point at register pressure and already-coalesced loads.
- **The registered metric is a null and the axis goes back.** `g_pack32` at
  +0.336 % against a +0.5 % kill rule, on its third independent reading, inside
  the null control's own cross-shape scatter. Stop-listed. `p_prefetch_w` is
  null wherever it fits, closing E104 arm P. The E111-versus-E104 contradiction
  on `n_nosums` resolves **in E111's favour**.
- **The largest measured mechanism in the campaign is the `sums` hoist, and its
  cost is now priced.** `x_sumshoist` is bit exact over 4,467,008 outputs at
  every cell including NA=5, worth **+5.376 %** round weighted, captures 91.7 %
  of its own free ceiling, and *reduces* registers and machine text. It is not
  shippable from `research/`, and its timed number excludes table production,
  which as a separate dispatch costs 117-452 us against a 14-35 us saving on
  four of five shapes. **The mechanism is worth funding only if the fill can be
  fused into the kernel that already produces `x`.**
- **Two axes closed by census, with no GPU time spent guessing.**
  `rows_per_simd = 8` spills 176 B on the ranked `applegpu_g17s` at NA=4, so it
  stays closed on the advisor's own condition. Threadgroup memory allows 25
  concurrent threadgroups even for the largest rung-2 arm, so shared memory is
  never the constraint.
- **Smallest useful next action:** ask whether the `sums` table can be produced
  by the kernel that already writes `x` - the RMS-norm or the preceding
  elementwise op - so it costs one extra 10 KB store and **zero extra
  dispatches**. That is a single-question experiment, it needs `quantized.cpp`
  to unfreeze only at the very end, and it is the only route by which the
  +5.376 % becomes real.

## Suggested follow-ups, not implemented

1. **Fuse the sums fill.** Measure the marginal cost of writing the 10 KB table
   from inside an existing producer kernel, against the 117 us floor this
   session measured for a standalone dispatch. That number, not the arm,
   decides the delivery question.
2. **Price the shuffle properly.** The ladder says a `simd_shuffle` costs 2.9
   loads, but the shuffle arms measure 2x worse than the ladder predicts and
   they also allocate 20-25 more registers. Separate the slot cost from the
   live-range cost with a shuffle arm that holds register pressure constant.
3. **Confirm the NA=5 spill corruption on `applegpu_g17s`** before any future
   wide-qmv change is allowed to spill there. Ten minutes of this probe on an
   M5 host would settle it and would protect every later kernel experiment.
4. **Re-run the ladder at NA=5.** Three of the six calibration rungs spill at
   NA=5 on g16s, so there is no load or shuffle price at the width where the
   kernel is most expensive. On g17s they would not spill.
