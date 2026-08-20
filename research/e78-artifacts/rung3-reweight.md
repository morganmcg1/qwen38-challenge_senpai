mlp.down M=6 per-dispatch delta = -0.038094 ms
arm E delta at M=6 = -2.43800 ms/round (64 calls)

| M | local rounds | local weight | ranked share % | ranked weight | arm E delta ms/round |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.0128 | - | - | +0.00000 |
| 4 | 5 | 0.0641 | 14.20 | 0.1464 | +0.00000 |
| 5 | 5 | 0.0641 | 24.10 | 0.2485 | +0.00000 |
| 6 | 23 | 0.2949 | 33.40 | 0.3443 | -2.43800 |
| 7 | 4 | 0.0513 | 12.20 | 0.1258 | +0.00000 |
| 8 | 6 | 0.0769 | 7.35 | 0.0758 | +0.00000 |
| 9 | 34 | 0.4359 | 5.75 | 0.0593 | +0.00000 |

local-mix weighted mean  = -0.71890 ms/round
ranked-mix weighted mean = -0.83948 ms/round
ranked / local           = 1.16773

predicted local delta = -1.095197e-04 s/tok (-0.3480 %)
measured  local delta = -3.017132e-05 s/tok (-0.0959 %)
prediction / measurement = 3.630 x
ranked-reweighted measured estimate = -3.523188e-05 s/tok (-0.1120 %)

session null = 2.991152e-05 s/tok (0.0950 %)
threshold    = 5.982304e-05 s/tok (0.1901 %)
|measured| / null    = 1.009
|ranked est| / null  = 1.178
passes threshold: local=False ranked=False

leg rank test: U = 32 of 36, exact one-sided p = 0.01299 (12/924)
position-matched pairs: -2.903e-05, +1.430e-05, -3.622e-05, -7.376e-05, -1.571e-05, -4.060e-05
paired mean = -3.017132e-05 s/tok, negatives 5/6, sign-test p = 0.1094

rung 2a cell sign test: 23 of 24 cells favour fewer groups; one-sided p = 1.490e-06, two-sided p = 2.980e-06
