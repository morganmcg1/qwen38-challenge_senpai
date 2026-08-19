# E54 in-kernel routing, read out of the built binaries

Source: `research/qmv_parity_dump` cells in `.mlxfast-private/e54-parity`. Widths swept: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]. Quantization bit widths: [4, 3].

`bits=3` never enters a crossrow kernel in any arm. The crossrow family is specialised `affine4_g64`, so only 4-bit group-64 weights reach it. The scored checkpoint is affine 4-bit group-64, so `bits=3` is a negative routing control and not a scored path.

## bits = 4

| M | iso_m5_ipg3 | iso_m5_ipg5 | iso_m7_ipg4 | iso_m7_ipg5 | iso_m8_ipg4 | iso_m8_ipg5 | shipped | e27_full | iso_m5_ipg5_lane_perturb |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 2 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64<T, 2>` | `qmv_fast_crossrow_affine4_g64<T, 2>` | `qmv_fast_impl` |
| 3 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 3, 3, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 3, 3, true>` | `qmv_fast_impl` |
| 4 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true>` | `qmv_fast_impl` |
| 5 | `qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>` |
| 6 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>` | `qmv_fast_impl` |
| 7 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 7, 5, true>` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>` | `qmv_fast_impl` |
| 8 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 8, 5, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>` | `qmv_fast_impl` |
| 9 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>` | `qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>` | `qmv_fast_impl` |
| 10 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 11 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 12 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |

## bits = 3

| M | iso_m5_ipg3 | iso_m5_ipg5 | iso_m7_ipg4 | iso_m7_ipg5 | iso_m8_ipg4 | iso_m8_ipg5 | shipped | e27_full | iso_m5_ipg5_lane_perturb |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 2 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 3 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 4 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 5 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 6 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 7 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 8 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 9 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 10 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 11 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |
| 12 | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` | `qmv_fast_impl` |

## Isolation check per pair

| pair | treated cell | cells that change kernel | control cells identical |
|---|---|---|---|
| P1 | bits=4, M=5 | [(4, 5)] | 23/24 |
| P2 | bits=4, M=7 | [(4, 7)] | 23/24 |
| P3 | bits=4, M=8 | [(4, 8)] | 23/24 |

Every non-treated cell enters a byte-identical kernel in both arms of its pair, so its timing difference measures session noise only.

## P4, the E27 composite on the real shipped table

Cells that change kernel: [(4, 5), (4, 9)]

This reproduces E27's actual edit: `case 5` and `case 9` both move to IPG=5 while every other width keeps its shipped specialisation.

