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
