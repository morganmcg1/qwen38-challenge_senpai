# E147 terminal result: the seed-prefill k-loop software pipeline

**Label: not useful.** The mechanism is valid, it is exact, it builds, it ships
inside the byte budget, and on the ranked M5 it makes the scored prefill
**slower by 2.1451 %**. The hypothesis is refuted by an official receipt, not by
a local screen.

- PR #147, branch `qwen-alphonse/e147-nax-seed-prefill-double-buffer`
- Base `bcc11dc6527ea4fa32be22c15b99a3a95695bfaf`
- Result commit `3e88f307`
- W&B `37lgc90s`
  <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/37lgc90s>

---

## 1. Primary metric

`e147_ranked_prefill_pct` = 100 x (8-prompt mean `prefill_seconds_per_token` /
0.001028291 - 1). Baseline 0.0. Direction: minimize. Target: <= -3.0 %.

| | value |
|---|---|
| baseline | 0.0 |
| **candidate** | **+2.1451** |
| delta | +2.1451 |
| target | <= -3.0 |

harness=ranked. Receipt `7226dc9a`, published median **3.426542432971500**,
`prefill_seconds_per_token` 0.001050348, within-row CV 0.0218 %. The receipt was
**rejected**: the mechanism costs score.

Cross-checks against the three other rows this branch owns give the same sign
and magnitude: +2.2036 %, +2.1850 %, +2.0782 %. The regression is uniform across
the prompt pool, it is steady state, and it is not a JIT compile inside the
timed window.

Forecast error: I predicted roughly -3 %; the outcome was +2.1451 %.
`e147_forecast_error_pp` = **5.18 percentage points**. The local screen and the
ranked result did not merely differ in size, they differed in sign.

## 2. Why it failed

The mechanism doubles the threadgroup staging buffer, stages tile 0 in a
prologue, keeps one barrier per iteration and alternates halves. It is the
pipelined k-loop that `fp_quantized_nax.h` already ships for the float path.

Two independent findings explain the sign.

**FINDING 250, the dispatch is host-split.** `backend/metal/quantized.cpp:697`
routes every scored prefill GEMM to `qmm_nax` whenever `is_nax_available()`.
`qmm_splitk` at `:786-816` always falls back at our shapes, because
`split_k = max(1, 512 / (n_tiles * m_tiles))` is 1 at M = 512. So:

| host | `is_nax_available()` | scored prefill entry point | rung A | rung B |
|---|---|---|---|---|
| ranked M5 `applegpu_g17s` | true | `affine_qmm_t_nax` | **dead** | live |
| local M4 Pro `applegpu_g16s` | false | `affine_qmm_t` | live | **dead** |

`e147_rungA_prefill_pct` is **-1.0489 %** entry-temperature-adjusted (t = -11.30,
raw -1.0657 %, t = -11.12) on the local host and **exactly 0 by dispatch** on the
ranked host. `e147_rungB_marginal_prefill_pct` is therefore the whole ranked
regression, **+2.1451 %**, and it is unmeasurable locally.

The two hosts do not run the same kernel on the scored path. That is the finding
this experiment actually produced, and it invalidates the screening design, not
just this arm.

**The remaining 4 % was a base difference, not an arm effect.**
`research/e147-base-diff.json` decomposes it: grid wide to tight **-3.7674 %**,
add onePass67 **+0.3280 %**, probe 0.15 **-0.1961 %**. Additive 4.2915,
multiplicative 4.4617, observed 4.3978 (se 0.3222),
`max_abs_residual_sigma` 1.635. `e147_base_arm_gap_explains_four_percent = true`.
The launch-geometry term follows a shape law: the tight-grid saving is
`1296.8 * ln(Mbar)` microseconds per round.

## 3. What ships on this branch

Nothing that changes the scored surface's behaviour.

```
senpai/check-editable-budget.sh bcc11dc6
editable budget OK: source=2635239/3000000 headroom=364761
  growth=-3809/262144 exempt=2410 files=154
```

`e147_rungE_base_composition` = clean PR base `bcc11dc6` on `quantized.h` and
its twin, both rungs reverted; `quantized_nax.h` carries only the (128, 32)
retile arm, shipped off behind `kE147NaxRetileOn`, plus the RULE 145 tile guards.

The revert is verified three ways: an empty `git diff` against `bcc11dc6`, a
green `twin_audit.py`, and a machine-code census in which every non-NAX entry
point at `head` is byte-identical to `base` in registers, spill and
machine-text digest on both arches.

Worker assertion, both Rule 136 polarities plus the guard needle:

```
worker_sha256 ff6a585e26d4bb611561ff47fcc557f9f7214df53f018d0071611626a7f52415
ok require 'constexpr bool kE147NaxRetileOn = false;'                       1
ok require 'matches no tile_matmad_nax branch and would multiply nothing'   3
ok forbid  'constexpr bool kE147NaxRetileOn = true;'                        0
ok extraction: 81232 strings
```

## 4. Secondary results

**Rung E-1a, PASS** (`e147-retile-arm.json`). Grid-stride retile dispatch for
`qmm_t`. `e147_rungE_grid_stride_max_iters` = 1 on all 28 scored rows, 2 on the
probe. Five positive controls rejected over 98 cases; 12 source assertions true.

**Rung E-1b, PASS, DOES NOT SHIP** (`e147-rungE1b.txt`).
`e147_rungE1b_ships = false`. Both 512-token legs matched with divergence 0.
The corrected positive control failed at step 0 **inside the seed prefill**:

```
qwen-mtp contract violation [seed_token_mismatch] at step 0:
seed prefill token 3792 disagreed with the reference's 9572
```

What transfers to NAX is the index algebra. What does not transfer is the
arithmetic, `tile_matmad_nax` operand packing. It is a rehearsal. The defective
rotation control is preserved in `e147-rungE1b-rotation-control.txt`. Note that
`research/e147_rungE1b.sh` patches `quantized.h`, which is now reverted, so the
harness will no longer apply without re-anchoring.

**Rung E-1c, PASS** (`e147-rungE1c.json`). The (128, 32) NAX retile arm,
compiled through the real mechanism rather than a hand-instantiated proxy.

```
arm_on_differs_from_arm_off      True    the flag is the whole switch
illegal_shape_rejected           True    (128, 48) breaks both retile asserts
failopen_compiles_unguarded      True    RULE 101: the hazard is real
failopen_shape_rejected          True    RULE 145 catches it
rule145_named_in_refusal         True
guarded_tile_sites               3
shipped_other_sites_compile      True
nax_arm_off_air_delta_bytes      2112
transfer_air_delta_bytes         2176
transfer_isa_text_delta_g17s     12
transfer_register_delta_g17s     0
transfer_air_overstates_isa_by   181.3
```

The last block is the useful measurement. The NAX family is untranslatable
offline, so the arm-off default was priced on the one member of the refactor
family the backend will translate, the non-NAX twin of the same mechanism:
**+2,176 bytes of AIR collapses to +12 bytes of ISA text and 0 registers**. AIR
overstates the cost by 181x, because `metal -O2` emits the per-tile lambda
out-of-line with a capture frame and the AGX backend inlines it again.
`always_inline` shrinks AIR 15,080 to 12,920 and leaves ISA text unchanged at
5,640, so no source change is warranted.

Derived arm geometry at (BM, BN) = (128, 32), BK 64, WM = WN = 2: SM 64, SN 16,
TM 4 (even, M-paired branch), TN 1, TK 2, `BK_padded` 72 for bf16. Loader
`QuantizedBlockLoader<T,32,64,72,1,128,64,4>` reaches max destination offset
2,296 <= 2,304. At N = 16,480: tiles_x 515, tiles_y 4, required 2,060, launched
2,064, grid-stride max iterations 1.

**Rung E-2, PASS** (`e147-rungE2.json`). Three instantiations of
`affine_qmm_t_nax<bfloat16_t,64,4,1,0,BM,64,BN,2,2>`. Shipped 64x64 (TM 2, TN 2,
N-paired, metallib 27,426, NOT TRANSLATED); retile 128x32 (TM 4, TN 1, M-paired,
28,866, NOT TRANSLATED); fail-open 96x32 (TM 3, TN 1, **no branch**, 17,523,
translated). Both `mma` overloads issue an identical
`matmul2d_descriptor(16,32,16,transpose_a,transpose_b,true,multiply_accumulate)`;
only operand packing differs. `e147_rungE_matmad_branch_bitexact` stays **open
and host-blocked**.

**A1 refuted.** `e147_rungA_decode_census_delta = none`,
`decode_entry_points_moved = false`. No rung ever moved a QMV decode entry point,
at four revisions on two arches.

**A2 refuted** (`e147-f8-followup.json`). `NAXWsStagingPlan<T,BN,BK>` requires
`2 * tile * sizeof(T) <= 32768`. `e147_max_threadgroup_bytes` = **18432**.
`e147_float32_nax_pipeline_disposition` = refuted.

**Thermal bound on rung A.** Entry spread 18.27 C, sd 4.35; sensitivity
-0.0108 %/C; ABBA imbalance +1.569 C; confound bound 0.0169 pp on a 1.0657 pp
effect. The local rung A effect is not thermal.

**Rung A positive control.** `nobarrier` caught:
`seed_token_mismatch at step 0: 99323 vs 9572`.
`e147_rungA_positive_control_failed = 1.0`.

**Rung C, `--local-submit`.** `passed = true`, `all_tokens_matched = true`,
divergence 0, 128 decode tokens, speedup 1.804258671876747, acceptance
0.98198198, head `62516c6f`.

## 5. Two corrections I owe the record

**The revert target.** F11 named `770a3ff2` as the clean base. It is
byte-identical to the PR base on `quantized_nax.h` and its twin, so it is a
valid source base for the NAX compile rungs, but it is several hundred commits
behind on `quantized.h`. Reverting there deleted promoted work in
`qmv_fast_crossrow_affine4_g64_wide`: the NA bound 4 -> 5, the `vec<T, 4>`
`DIRECT_NIBBLES` load, and the width-5 dispatch `<T,5,3,true>` -> `<T,5,5,true>`.
`twin_audit.py` caught it by reporting the pinned `quantized` comment waiver
DEAD. The revert target is now `feaa92b4`, byte-identical to the PR base.

**The byte budget.** I reported `growth=192000/262144` in my F10 reply. That was
measured against `770a3ff2` and charged this experiment for other students'
promoted work. Against the real base the growth is **-3,809 bytes**. I also said
the rung B revert freed "roughly 60 KiB"; it freed **7,686 bytes**.

**A third, smaller correction.** F11 gave the RULE 145 condition as
`TN % 2 == 0 || TM % 2 == 0`. That admits `(TM, TN) = (2, 3)`, which matches
neither `tile_matmad_nax` branch. The shipped guard is the disjunction of the
two branch predicates copied from `steel/gemm/nax.h:847` and `:864`:
`(TN == 1 && TM % 2 == 0) || (TN % 2 == 0)`.

## 6. Reproduction

```bash
python3 research/e147_dispatch_check.py                       # FINDING 250
python3 research/e147_rungE1c.py --base 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf
python3 research/e147_rungE2.py
python3 research/e147_qmv_jit_census.py \
  --rev base=feaa92b4 --rev rungA=01ed58d5 \
  --rev rungB=302a44df --rev head=d16a0c00 \
  --out research/e147-qmv-jit-census.json
python3 research/twin_audit.py
senpai/rebuild-and-assert-worker.sh \
  --require 'constexpr bool kE147NaxRetileOn = false;' \
  --forbid  'constexpr bool kE147NaxRetileOn = true;' \
  --require 'matches no tile_matmad_nax branch and would multiply nothing'
senpai/check-editable-budget.sh bcc11dc6527ea4fa32be22c15b99a3a95695bfaf
python3 research/e147_wandb_log.py
```

All rungs above are zero-GPU offline compilation and complete in seconds.
The timed rung A session is `research/e147_rungA_abba.sh`; it is
ABBA-counterbalanced with `MLXFAST_LOCAL_COOL_GATE=0`, so
`cool_gate_passed_real_gate = false` and `gate_qualified_for_timing = false`.

Host: Mac16,11 Apple M4 Pro, 48 GiB, `applegpu_g16s`, no NAX unit.

## 7. Suggested follow-ups, not implemented

1. **Price E-1c.** `5cdc9c17` (BitWonka, -4.9721 % prefill) used a generalized
   NAX seed retile with a 128x32 rectangular tile gated `M >= 128 && M % 128 == 0`;
   our seed is M = 512. The F227 value model converts -5.1 % prefill to
   **+0.3822 % of published median**, and the measured gap to the bar is 0.82 %
   (F230), so E-1c alone would close about 46 % of it. Turning
   `kE147NaxRetileOn` on is a one-line change on a gated, guarded, compiled arm.
2. **RULE 146 composition.** Decode-only time is recoverable from a ranked
   receipt as `mtp_spt - prefill_spt`, so a prefill-only mechanism and a
   decode-only mechanism can ride one submission and be read independently.
3. **The remaining two NAX tile sites.** `qmm_n_nax_tgp_impl` and
   `affine_gather_qmm_rhs_nax` are now guarded but not retiled. Neither is on
   the scored path today (`any_non_transposed_scored_gemm = false`).
4. **A NAX-capable host.** `e147_rungE_matmad_branch_bitexact` cannot be closed
   from this fleet. The prefill GEMM is the one scored kernel family no machine
   we own can execute.
5. **Screen design.** Any future local screen of a scored GEMM must first run
   `research/e147_dispatch_check.py`. A local win on `affine_qmm_t` is worth
   exactly zero on the ranked host, and this experiment paid a submission slot
   to learn it.
