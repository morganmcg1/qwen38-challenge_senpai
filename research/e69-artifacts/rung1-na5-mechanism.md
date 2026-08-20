# E69: why `xvec` pays at NA=5 and nowhere else

Source: the AIR census in `rung0-air.json`, `cells[NA][arm]`. Zero GPU.

## The observation

`xvec` replaces four 2-byte scalar `x` loads with one 8-byte vector load. It
cuts device x-load instructions by 4x at every width, yet its measured effect
is confined to one width:

| NA | x loads, plain -> xvec | `plain` GB/s | `xvec` % | float array in memory? |
|---|---|---|---|---|
| 2 | 8 -> 2 | 211-225 | +0.03 | no |
| 3 | 12 -> 3 | 197-221 | -0.54 | no |
| 4 | 16 -> 4 | 171-190 | -0.13 | no |
| 5 | 20 -> 5 | 142-159 | **-3.56** | no |
| 6 | 24 -> 6 | 101-122 | -0.22 | **yes** |

## The mechanism

Two independent conditions must both hold before cutting load instructions can
buy time, and NA=5 is the only scored width where both do.

**Condition 1: the cell must be off the bandwidth roof.** At NA=2 and NA=3
`plain` runs at 197-225 GB/s against a measured 226.035 GB/s peak, so it is
memory-limited and no instruction saving can help. Throughput falls
monotonically with NA, so this condition is satisfied from about NA=4 onward.

**Condition 2: the inner loop must not already be doing scratch-memory
traffic.** The alloca census shows exactly where that starts:

| arm | NA=4 | NA=5 | NA=6 |
|---|---|---|---|
| `plain` | 114 regs, 1 alloca | 130 regs, 1 alloca | 182 regs, **2 allocas** |
| `wvec` | 114, 1 | 130, 1 | 182, **2** |
| `xvec` | 125, 1 | 140, 1 | 191, **2** |
| `wxvec` | 125, 1 | 140, 1 | 191, **2** |
| `tgx` | 118, 1 | 135, 1 | 188, **2** |

The single alloca at NA<=5 is `[4 x [4 x i16]]`, which is `packed`. At NA=6 a
second one appears, `[4 x <6 x float>]`: one of the two `vec<float,6>` arrays,
`acc` or `partial`, stops living in registers and becomes memory-backed. Once
the inner loop reads and writes scratch every iteration, removing 18 of 24
device load instructions returns almost nothing, because the limiter has moved.

So:

- NA=2, 3: condition 1 fails. At the roof. `xvec` null.
- NA=4: condition 1 only weakly met, 181 GB/s. `xvec` -0.13 %.
- **NA=5: both conditions hold. `xvec` -3.56 %.**
- NA=6: condition 2 fails. Spilled. `xvec` -0.22 %.

This is falsifiable. If the NA=6 spill were removed, `xvec` should pay at NA=6
as it does at NA=5. NA=6 carries the largest single share of ranked verify
time, near 35 %, so that is the follow-up worth having.

## Preregistered prediction for the replication session

Provenance, stated exactly. At the moment I committed this block,
`rung1-na4-s2.json` had **already been written** by the replication job, so
prediction 3 is not blind and does not count as evidence. I had not opened the
file, but that is a claim about me, not a checkable fact, so I do not rely on
it. `rung1-na5-s2.json` and `rung1-na6-s2.json` did **not** exist, which `ls`
confirmed in the same commit, so predictions 1, 2 and 4 are blind and those are
the ones that carry weight.

The mechanism is per-compile and per-width, not per-session, so the replication
must reproduce the shape of the curve, not merely its noise level:

1. `xvec` at NA=5 reproduces between -2.0 % and -5.0 % on at least 5 of 7
   shapes.
2. `xvec` at NA=6 stays inside +/-1.0 %.
3. `xvec` at NA=4 stays inside +/-1.0 %.
4. The alloca counts are identical to session 1, because they are a property
   of the compile and not of the run.

If prediction 1 fails and NA=5 comes back inside +/-1 %, then the session-1
spike was a session artifact, the mechanism above is not supported, and I will
report the whole of `xvec` as null.

## A defect in my own FMA arm

The same census shows my `fma` arm carries an extra float alloca at **every**
width, which the shipped body does not have:

| arm | NA=4 | NA=5 | NA=6 |
|---|---|---|---|
| `plain` | 1 alloca | 1 alloca | 2 allocas |
| `fma` | **2**, `[4 x <4 x float>]` | **2**, `<5 x float>` | **3**, `<6 x float>` |

The cause is in my own code. `fma()` has no overload for the non-native widths
`vec<float,5>` and `vec<float,6>`, so I wrote the contraction as a componentwise
loop:

```
for (int m = 0; m < NA; m++) {
  qdot[m] = fma(a3[m], n3, fma(a2[m], n2, fma(a1[m], n1, qdot[m])));
}
```

Indexing `qdot[m]` through a loop induction variable forces `qdot` to memory.
So the arm buys a 37.5 % cut in arithmetic instructions and pays for it with a
new scratch array, which is why it lands at only about -1 % instead of more.

That is an implementation defect, not a property of the idea. A version that
avoids the indexed temporary, for example by writing the four component
expressions explicitly, should do better. It would still have to triple its
effect to clear the >3 % bar at both NA=5 and NA=6, so I am reporting it as a
lead rather than running it.
