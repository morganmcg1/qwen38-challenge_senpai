# E27 — M=5 weight-stream cliff

Base `d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602`, host Apple M4 Pro `Mac16,11`
(20 GPU cores, 48 GiB, macOS 26.5.2, Swift 6.3.3, `applegpu_g16s`, NAX off).

Primary metric: `e27/round_weighted_step_at_first_stream_boundary_M4_to_M5`
= `C_round(5) / C_round(4)`, 1.4559 at the E22 baseline.

## Mechanism: the width curve is a weight-stream staircase

`qmv_fast_crossrow_affine4_g64_m<T, M, IPG, DN>` splits `M` inputs into
`ceil(M / IPG)` groups and re-streams the whole weight matrix once per group, so
`weight_streams = ceil(M / IPG)`. Round-weighted cost recomputed from
`.mlxfast-private/qmv-curve/e22-cm-r1/vendored.json` as
`Σ calls_per_verify × seconds_per_call`:

| M | streams | C_round (ms) | C/M (ms) | step |
|---:|---:|---:|---:|---:|
| 1 | 1 | 59.871 | 59.871 | — |
| 2 | 1 | 63.886 | 31.943 | 1.0671 |
| 3 | 1 | 72.918 | 24.306 | 1.1414 |
| 4 | 1 | 83.031 | 20.758 | 1.1387 |
| 5 | 2 | 120.886 | 24.177 | **1.4559** |
| 6 | 2 | 129.284 | 21.547 | 1.0695 |
| 7 | 2 | 139.325 | 19.904 | 1.0777 |
| 8 | 3 | 177.758 | 22.220 | **1.2759** |
| 9 | 3 | 186.422 | 20.714 | 1.0487 |
| 10 | 4 | 272.079 | 27.208 | **1.4595** |

Every large riser sits exactly on a `ceil(M/IPG)` increment; every within-stream
step is 1.05–1.14. That run's table had M=8 at IPG=3; the current base ships
`<T,8,4,true>`, so its M=8 riser should already be gone.

## IPG legality inside NA ∈ [2, 4]

`_m` requires `M % IPG != 1` and `_wide` asserts `NA ∈ [2, 4]`:

| M | legal IPG | streams | freedom |
|---:|---|---:|---|
| 4 | 2, 4 | 2, **1** | yes |
| 5 | 3 only | 2 | none |
| 6 | 2, 3, 4 | 3, **2**, 2 | yes |
| 7 | 4 only | 2 | none |
| 8 | 2, 3, 4 | 4, 3, **2** | yes |
| 9 | 3 only | 3 | none |

M=5, M=7 and M=9 have no alternative IPG at all, and the shipped table is already
stream-optimal at M=4, 7, 8, 9. Only `NA = 5` can move the M=4→M=5 riser, because
IPG=5 is legal for M=5 (`5 % 5 = 0`), M=7 (`7 % 5 = 2`) and M=9 (`9 % 5 = 4`).

## Static register/spill evidence

`research/crossrow_na_probe.metal` instantiates one kernel per `_wide` NA and per
`_m` (M, IPG) pair, all at `DIRECT_NIBBLES=true` as the production sites do.

```bash
xcrun -sdk macosx metal -std=metal3.1 -S -O2 [-DCROSSROW_NA_PROBE_WIDE] \
  research/crossrow_na_probe.metal \
  -I Vendor/mlx-swift/Source/Cmlx/mlx -o /tmp/e27_probe.ll
xcrun -sdk macosx metal-opt -passes='default<O3>' -S /tmp/e27_probe.ll \
  -o /tmp/e27_probe_o3.ll
python3 research/air_kernel_stats.py /tmp/e27_probe_o3.ll --match crossrow_
```

The wide arms need `static_assert(NA >= 2 && NA <= 6)` in `quantized.h` for the
duration of the probe only.

| kernel | allocas | alloca types | peak_live_regs | float_ops | dev_loads/body | backedges |
|---|---:|---|---:|---:|---:|---:|
| `crossrow_dn_na2` | 1 | `[4 x [4 x i16]]` | 62 | 56 | 32 | 2 |
| `crossrow_dn_na3` | 1 | `[4 x [4 x i16]]` | 83 | 60 | 36 | 2 |
| `crossrow_dn_na4` | 1 | `[4 x [4 x i16]]` | 104 | 64 | 40 | 2 |
| `crossrow_dn_na5` | 1 | `[4 x [4 x i16]]` | 125 | 68 | 44 | 2 |
| `crossrow_dn_na6` | 2 | **`[4 x <6 x float>]`** + `[4 x [4 x i16]]` | 144 | 72 | 48 | 2 |
| `crossrow_m4_ipg2` | 1 | `[4 x [4 x i16]]` | 62 | 56 | 32 | 2 |
| `crossrow_m4_ipg4` | 1 | `[4 x [4 x i16]]` | 104 | 64 | 40 | 2 |
| `crossrow_m5_ipg3` | 2 | `[4 x [4 x i16]]` ×2 | 87 | 116 | 68 | 4 |
| `crossrow_m5_ipg5` | 1 | `[4 x [4 x i16]]` | 125 | 68 | 44 | 2 |
| `crossrow_m6_ipg2` | 1 | `[4 x [4 x i16]]` | 62 | 56 | 32 | 2 |
| `crossrow_m6_ipg3` | 1 | `[4 x [4 x i16]]` | 83 | 60 | 36 | 2 |
| `crossrow_m6_ipg4` | 2 | `[4 x [4 x i16]]` ×2 | 108 | 120 | 72 | 4 |
| `crossrow_m7_ipg4` | 2 | `[4 x [4 x i16]]` ×2 | 108 | 124 | 76 | 4 |
| `crossrow_m7_ipg5` | 2 | `[4 x [4 x i16]]` ×2 | — | — | 76 | 4 |
| `crossrow_m8_ipg2` | 1 | `[4 x [4 x i16]]` | 62 | 56 | 32 | 2 |
| `crossrow_m8_ipg3` | 2 | `[4 x [4 x i16]]` ×2 | 87 | 116 | 68 | 4 |
| `crossrow_m8_ipg4` | 1 | `[4 x [4 x i16]]` | 104 | 64 | 40 | 2 |
| `crossrow_m9_ipg3` | 1 | `[4 x [4 x i16]]` | 83 | 60 | 36 | 2 |
| `crossrow_m9_ipg5` | 2 | `[4 x [4 x i16]]` ×2 | 129 | 132 | 84 | 4 |

`allocas == 2` is two inlined `_wide` bodies (main + tail), each with the ordinary
nibble-staging array. The spill signature is the extra `[4 x <NA x float>]`
accumulator alloca, and it appears only at **NA = 6**.

Static counts are per inlined body; when a group loop survives, multiply by the
runtime group count. Dynamic device loads per output tile:

| M | shipped | dynamic loads | alternative | dynamic loads |
|---:|---|---:|---|---:|
| 5 | IPG=3 (3+2) | 36 + 32 = 68 | IPG=5 (5) | **44** |
| 7 | IPG=4 (4+3) | 40 + 36 = 76 | IPG=5 (5+2) | 76 |
| 8 | IPG=4 (4+4) | 40 × 2 = **80** | IPG=3 (3+3+2) | 36 × 2 + 32 = 104 |
| 9 | IPG=3 (3+3+3) | 36 × 3 = 108 | IPG=5 (5+4) | **84** |

### Verdicts

- **No register cliff at M=8.** Neither `<T,8,3,true>` nor `<T,8,4,true>` spills.
  4+4 costs +17 lane registers (104 vs 87) but emits one body instead of two:
  80 dynamic device loads against 104. The frontier's register-pressure rationale
  for 3+3+2 is not reproducible with this toolchain.
- **NA=5 is spill-free**, NA=6 is not, so the cap is one short of the real limit.
- **M=7 gains nothing from IPG=5** (76 loads either way, 2 streams either way).
- Caveat: `maxTotalThreadsPerThreadgroup` is pinned at 1024 for every NA on this
  host (E13), so the occupancy API cannot report achieved simdgroup residency,
  and the offline AGX ISA route is blocked (AIR 2.8 vs translator 2.5). Static AIR
  is necessary but not sufficient; timing decides.

## Frontier's "319/437/216 µs for M=7/8/9" does not reproduce

Per-call µs from `e22-cm-r1` (that run had M=8 at IPG=3, the form the comment
prefers):

| shape | ×calls | M=1 | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 | M=9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear_attn.in_proj_fused_qkvzba` | 48 | 195.8 | 239.2 | 274.4 | 399.8 | 427.4 | 457.2 | 587.5 | 615.9 |
| `linear_attn.out_proj` | 48 | 80.0 | 128.1 | 137.8 | 183.3 | 193.9 | 206.9 | 253.5 | 264.4 |
| `full_attn.qkv_proj_fused` | 16 | 166.3 | 215.4 | 246.2 | 354.0 | 378.5 | 407.8 | 519.0 | 543.2 |
| `full_attn.o_proj` | 16 | 79.4 | 125.3 | 138.5 | 183.6 | 193.7 | 206.5 | 252.4 | 264.2 |
| `mlp.gate_up_fused` | 64 | 407.9 | 450.2 | 521.7 | 787.8 | 846.5 | 915.6 | 1185.1 | 1244.9 |
| `mlp.down` | 64 | 214.6 | 282.6 | 317.1 | 446.3 | 475.3 | 512.9 | 641.6 | 671.2 |
| `head.lm_head` | 1 | 2856.4 | 2934.8 | 3403.9 | 5307.9 | 5715.9 | 6198.8 | 8132.5 | 8616.0 |
| `head.compact_draft_vocab` | 0 | 1152.1 | 1189.0 | 1379.9 | 2133.3 | 2294.6 | 2487.1 | 3252.5 | 3430.7 |

Every scored shape is strictly monotone in M. No shape shows an M=9 < M=7
inversion and none produces the triple (319, 437, 216). The claimed inversion
would require M=9 to run a cheaper configuration than M=7, which no dispatch
table in this repository does.

## Falsification arm: IPG changes at constant NA (commit `7b5183d`)

Before touching `NA_max`, one build re-pointed three dispatch sites inside the
legal `NA ∈ [2,4]` range and re-measured M = 4, 6, 7, 8. Predictions were written
down first: two slower, one neutral, one control.

| M | change | streams | peak regs | predicted | base (ms) | arm (ms) | arm/base |
|---:|---|---:|---:|---|---:|---:|---:|
| 4 | IPG 4→2 | 1→**2** | 104→62 | slower | 83.115 | 116.219 | **1.3983** |
| 6 | IPG 3→4 | 2→2 | 83→**108** | neutral | 128.865 | 128.849 | **0.9999** |
| 7 | unchanged | 2→2 | same | control | 139.078 | 139.112 | **1.0002** |
| 8 | IPG 4→3 | 2→**3** | 104→87 | slower | 149.355 | 177.759 | **1.1902** |

- **M=6 is the decisive control.** +30% lane registers at a constant stream count
  costs 0.9999×, below the M=7 drift control at 1.0002×. Register pressure has no
  measurable cost anywhere in the legal NA range on this host.
- **The "register cliff at M=8" is refuted with the wrong sign.** `<T,8,3,true>`
  uses *fewer* registers (87 vs 104) and is **19.0% slower**. The shipped
  `<T,8,4,true>` is correct, but for the stream reason, not the register reason.
  The two stories give opposite advice about widening accumulators, which is
  exactly what the NA=5 arm then settled.
- Cross-session anchor: this fresh `<T,8,3>` measured 177.759 ms against E22's
  177.758 ms, 1 µs apart (0.0006%).

One job in this arm (`42e7ac84-9190-4b62-ad1e-5bbb38d9fedd`, ~19:07 UTC) exited 1
*after* the sweep passed (25.345 s) and `vendored.json` was written: the
research-only `research/qmv_cost_curve_summary.py:266` normalises every shape
against M=1, and `--widths 4,6,7,8` omits width 1. No measurement was lost.
**Always include width 1 in `--widths`.**

## Implementation (commit `0207de6`)

Applied identically to `Vendor/mlx-swift/.../kernels/quantized.h` and its
runtime-effective twin `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`:

1. `static_assert(NA >= 2 && NA <= 5, ...)` — raise the wide cap by one.
2. `<T, 5, 3, true>` → `<T, 5, 5, true>` — 2 → **1** weight stream.
3. `<T, 9, 3, true>` → `<T, 9, 5, true>` — 3 → **2** weight streams.

M = 7 was deliberately left at `<T,7,4,true>`: IPG=5 gives 2 streams and 76
dynamic loads either way. `python3 research/twin_audit.py` → rc 0,
"TWIN AUDIT OK: 29 runtime-effective twin(s)".
`crossrowGate()` in `Tests/MLXFastTests/QwenQMVCostCurveTests.swift:555` parses
`quantized.h` at runtime, so no Swift test change was needed.

Scope and budget, checked before the expensive work:
`senpai/validate-assignment-scope.sh` → assignment scope OK, 2 submitted paths;
`senpai/check-editable-budget.sh` → source 2,448,240 / 3,000,000 B, growth
0 / 262,144 B, exempt 2,410 B / 2 GiB.

## Pre-registered gates and outcomes

All four criteria were posted publicly *before* any NA=5 timing existed
(PR #32 comments `5332537783` and `5332728160`) and were not revised.

| gate | condition | outcome |
|---|---|---|
| **K1** static | `_wide<T,5,true>` no spill alloca and `peak_live_regs <= 130`, NA=6 must still spill | **PASS** — 1 alloca, 125 regs; NA=6 spills `[4 x <6 x float>]` |
| **K2** parity | `BIT-IDENTICAL`, 0 differing cells | **PASS** — 192/192 cells (96 per bit-width, bits ∈ {3,4}), 0 differing |
| **K3** timing | `C_round(5)_NA5 <= 0.95 × 120.683 ms`, i.e. metric <= 1.383 | **PASS** — 96.423 ms, metric **1.1607** |
| **K4** budget | <= 2 NA=5 builds, 1 A/B/A session | honored — 2 builds (parity arm + curve) |

## Result: full width curve, base vs NA=5

Both columns recomputed identically from each run's `vendored.json` as
`C_round(M) = Σ_shapes calls_per_verify × seconds_per_call`.

| M | base (ms) | NA=5 (ms) | NA=5/base | base C/M | NA=5 C/M |
|---:|---:|---:|---:|---:|---:|
| 1 | 64.549 | 59.979 | 0.9292 | 64.549 | 59.979 |
| 2 | 65.628 | 64.707 | 0.9860 | 32.814 | 32.353 |
| 3 | 72.993 | 73.136 | 1.0020 | 24.331 | 24.379 |
| 4 | 83.115 | 83.072 | 0.9995 | 20.779 | 20.768 |
| 5 | 120.683 | **96.423** | **0.7990** | 24.137 | 19.285 |
| 6 | 128.865 | 129.280 | 1.0032 | 21.478 | 21.547 |
| 7 | 139.078 | 139.007 | 0.9995 | 19.868 | 19.858 |
| 8 | 149.355 | 150.110 | 1.0051 | 18.669 | 18.764 |
| 9 | 186.233 | **164.900** | **0.8854** | 20.693 | **18.322** |

**Primary metric `C_round(5)/C_round(4)`: 1.4520 → 1.1607 (−0.2913).**
The assignment's stated baseline of 1.4559 reproduced to within 0.27%.

Only the two widths that were changed moved. The five untouched widths that sit
mid-sweep (M = 3, 4, 6, 7, 8) all land within ±0.5%, which sets the noise floor.
The cheapest per-token width moved from M=8 (18.669) to **M=9 (18.322)**, as
predicted by the stream model.

### Independent corroboration from the harness summary

These fields are computed by the harness, not by the recompute above.

| field | base | NA=5 |
|---|---|---|
| `stream_boundaries` | `[5, 9]` | **`[6]`** |
| `optimal_depth_q100` | 7 | **8** |
| `optimal_speedup_q100` | 2.914284 | 2.919089 |
| `optimal_seconds_per_token_q100` | 0.02274622 | 0.02270877 |
| `weighted_qmv_tax_9` | 2.88517 | **2.74929** |
| `stop_rule_branch` | part_b_full | part_b_full |
| `scored_shapes_off_fast_count` | 0 | 0 |
| `staircase_shapes_rank_first` | 8/8 | 8/8 |
| `peak_bandwidth_gb_s` | 227.59 | 226.999 |
| `peak_tflops` | 7.4974 | 7.49258 |

`stream_boundaries` moving `[5,9] → [6]` is exactly the new stream structure:
under NA=5, M = 1..5 is one stream and M = 6..9 is two. Boundary detection reads
the shipped dispatch tree (`qmv_cost_curve_summary.py:617`), not a guess.

## Honest caveats

1. **M=1 drifted −7.1% and M=2 −1.4% on unchanged widths.** M=1 is the first
   width measured in each sweep, on the coldest GPU, so this is a warm-up/thermal
   artifact rather than a code effect. It does not touch the primary metric, whose
   numerator and denominator (M=5, M=4) sit mid-sweep with M=4 at 0.9995.
2. **K2 cell count.** I pre-registered "96 cells"; the harness reports **192**,
   because it runs 96 cells at each of bits ∈ {3,4}. My number covered one
   bit-width. The binding condition (`BIT-IDENTICAL`, 0 differing) is unchanged
   and was met at both bit-widths.
3. **`qmv_parity_compare.py` never exits nonzero.** A diverging arm would also
   exit 0, so the verdict must be parsed from its text, not from the job exit code.
4. **`crossrow_na_max` still reports 4 in both summaries — cosmetic.**
   `CROSSROW_MAX_INPUTS_PER_GROUP = args.na_max` is a CLI flag defaulting to 4 and
   `research/run-qmv-curve.sh` never passes `--na-max`. It is not a readback of the
   built kernel. The kernel is definitively NA=5 per (a) M=5 at −20.1%,
   (b) `stream_boundaries` moving to `[6]`, and (c) the source at HEAD.
5. **The end-to-end modelled win is much smaller than the headline metric.**
   `optimal_speedup_q100` moves 2.9143 → 2.9191, only +0.16%, even though the
   depth optimum moves 7 → 8, because the depth curve is flat near its optimum
   (d = 3..8 spread 14.7%). The QMV verify tax genuinely drops
   (`weighted_qmv_tax_9` 2.885 → 2.749), but decode wall time is not dominated by
   the widths that changed.
6. **Standing counter-evidence is now overturned.** Branch `crossrow-na5`
   (`704af6f`) had measured NA=5 as 1.54× *slower* at M=5 while bit-exact
   (pre-`DIRECT_NIBBLES`); `0a739c9` was 1.37×/1.13× slower and *not* bit-exact;
   `84eedac` restored `NA_max=4`. This fresh same-base A/B supersedes those.
7. **Host transfer risk.** Everything here is M4 Pro. The ranked runner is M5.

## Reproduction

```bash
# baseline arm, at HEAD f0bb949 (base kernel source)
research/run-qmv-curve.sh e27-base-r1 d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602 \
  --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock

# candidate arm, at HEAD 0207de6 (NA=5)
research/run-qmv-curve.sh e27-na5-r1 d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602 \
  --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock

# K2 bit-exactness
research/run-qmv-parity.sh base=d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602 na5=0207de6
```

`run-qmv-curve.sh` performs no git operations; its second positional argument is a
recorded label and the script measures the working tree. Both arms used identical
parameters, so only HEAD differs.

W&B project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`:

- baseline `bg0yd4g3` — `qmv-cost-curve-e27-base-r1`
- NA=5 `hy0qq9sk` — `qmv-cost-curve-e27-na5-r1`

Candidate run identity: `head=0207de6476a089902b83fb658a2acd8883126f4e dirty=0`,
`mem=51539607552`, `widths=1..9 shapes_only=1 reps=21 inner=10 skip_stock=1`,
`2026-08-18T19:19:53Z → 19:28:47Z` (8 min 54 s wall). The cool gate stalled at
42.1 C (`stalled_above_40C`), the same thermal entry state as the baseline.
`sweepQuantizedMatmulOverVerifyWidth()` passed in 39.863 s.

## Suggested follow-ups (not implemented)

- Re-measure on the ranked M5 host before treating this as promotable. The whole
  result is a bandwidth/stream story, and M5's memory system differs.
- Ask whether M = 6, 7, 8 can reach one stream. That needs NA >= 6, which spills
  (`[4 x <6 x float>]`); a split-accumulator or two-pass form might avoid the
  spill, but nothing here suggests it would pay.
- Plumb `--na-max` through `research/run-qmv-curve.sh` so `crossrow_na_max` stops
  reporting a stale default.
- Because the depth curve is flat near the optimum, the largest remaining decode
  win is probably not in the QMV width table.
