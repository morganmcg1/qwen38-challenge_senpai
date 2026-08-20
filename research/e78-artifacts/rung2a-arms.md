# E78 rung 2a derived: arm prices from the measured cells

Derived from absolute cell times. No extra GPU time. Rung 2b tests the additivity assumption.

## Verify-round cost by width and arm (ms, derived)

| M | ship IPG | crown IPG | a_ship | b_crown | c_hyb24928 | d_hyb8192 | oracle (M and k) | local rounds | ranked % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 4 | 74.575 | 74.575 | 74.575 | 74.575 | 74.575 | 5 | 14.2 |
| 5 | 5 | 3 | 87.760 | 111.330 | 99.156 | 94.257 | 87.760 | 5 | 24.1 |
| 6 | 6 | 3 | 115.758 | 120.020 | 115.383 | 113.669 | 113.320 | 23 | 33.4 |
| 9 | 5 | 3 | 154.115 | 176.753 | 165.135 | 160.405 | 154.115 | 34 | 5.75 |

## Mix-weighted delta against a_ship (ms per verify round)

Negative is faster. Only M = 5, 6 and 9 move, so every other width cancels exactly.

| arm | local fixture mix | ranked pooled mix |
|---|---:|---:|
| a_ship | +0.0000 | +0.0000 |
| b_crown | +12.6354 | +8.6655 |
| hybrid_24928 | +5.4235 | +3.3556 |
| hybrid_8192 | +2.5421 | +1.2679 |
| kdown_8192 | -0.7189 | -0.8395 |
| oracle_k_and_m | -0.7189 | -0.8395 |

## Can out_vec_size separate the cells that want IPG 3?

### M = 5

| n | k | k blocks | shipped ms | crown ms | winner |
|---:|---:|---:|---:|---:|---|
| 5120 | 17408 | 272 | 0.33437 | 0.40873 | shipped |
| 5120 | 6144 | 96 | 0.12098 | 0.14814 | shipped |
| 14336 | 5120 | 80 | 0.24981 | 0.31840 | shipped |
| 16480 | 5120 | 80 | 0.28419 | 0.36339 | shipped |
| 34816 | 5120 | 80 | 0.57776 | 0.74871 | shipped |
| 248320 | 5120 | 80 | 4.00255 | 5.23523 | shipped |

separable_by_out_vec_size = True

### M = 6

| n | k | k blocks | shipped ms | crown ms | winner |
|---:|---:|---:|---:|---:|---|
| 5120 | 17408 | 272 | 0.47801 | 0.43992 | crown |
| 5120 | 6144 | 96 | 0.15379 | 0.15924 | shipped |
| 14336 | 5120 | 80 | 0.31967 | 0.34320 | shipped |
| 16480 | 5120 | 80 | 0.36402 | 0.39188 | shipped |
| 34816 | 5120 | 80 | 0.74329 | 0.80809 | shipped |
| 248320 | 5120 | 80 | 5.16470 | 5.65392 | shipped |

separable_by_out_vec_size = False
  collision: mlp.down wants IPG 3 and linear_attn.out_proj does not, at the same or lower out_vec_size

### M = 9

| n | k | k blocks | shipped ms | crown ms | winner |
|---:|---:|---:|---:|---:|---|
| 5120 | 17408 | 272 | 0.56457 | 0.63602 | shipped |
| 5120 | 6144 | 96 | 0.20216 | 0.22898 | shipped |
| 14336 | 5120 | 80 | 0.43955 | 0.50588 | shipped |
| 16480 | 5120 | 80 | 0.50244 | 0.57888 | shipped |
| 34816 | 5120 | 80 | 1.03969 | 1.20291 | shipped |
| 248320 | 5120 | 80 | 7.35521 | 8.52685 | shipped |

separable_by_out_vec_size = True

