# E69 rung 1: preregistered reading of the NA=5 cell

Committed after NA=2, NA=3 and NA=4 were reduced, and before `rung1-na5.json`
existed on disk. `git log` fixes the order.

## What the first three gated NA points show

`plain` throughput and the `fma` contrast, median over the 7 scored shapes:

| NA | plain GB/s | fma % vs plain | sign |
|---|---|---|---|
| 2 | 211-225 | +0.03 | none |
| 3 | 197-221 | +0.84 | slower |
| 4 | 171-190 | -1.55 | faster |

Same-arm null is within +/-0.08 % at every NA, so the NA=3 and NA=4 effects are
both real and both far above noise. The sign flips between them.

## Mechanism I think explains it

The cell is bandwidth-bound at low NA and issue-bound at high NA.

At NA=2 and NA=3 `plain` runs at 197-225 GB/s, which is at the M4 Pro roof, so
removing arithmetic cannot help: the machine is already waiting on memory. This
is also why arms A, X and A+X are null there despite cutting load instructions
by up to 5x.

At NA=4 register pressure rises (114 registers for `plain`, against 82 at NA=2),
occupancy falls, and measured throughput drops to 171-190 GB/s. The cell leaves
the roof. Arithmetic issue slots now sit on the critical path, so contracting
4 fmul + 4 fadd into 1 fmul + 3 fma + 1 fadd buys real time.

## Preregistered prediction for NA=5

`plain` at NA=5 uses 130 registers, between NA=4 (114) and NA=6 (182), so the
mechanism predicts NA=5 falls further from the roof than NA=4 and the `fma`
contraction helps at least as much as at NA=4.

1. `plain` GB/s at NA=5 is below the NA=4 value on most shapes, and above the
   NA=6 smoke value of 117.6 GB/s.
2. `fma` at NA=5 is faster than `plain`, that is a negative percentage.
3. The `fma` effect at NA=5 is between -1.0 % and -3.0 % on most shapes.
4. `wvec`, `xvec` and `wxvec` stay inside +/-1 % at NA=5, because cutting load
   instructions did not help at any NA so far.
5. `tgx` and `rows8` stay clearly slower, that is above +5 %.

## What would falsify the mechanism

If `fma` at NA=5 is slower than `plain`, or null, while `plain` GB/s at NA=5 is
below the NA=4 value, then throughput distance from the roof does not predict
the contraction payoff. In that case the NA=4 and NA=6 gains are more likely a
code-placement or scheduling artifact of this particular compile, and the
contraction does not deserve a 512-token leg test on its own.
