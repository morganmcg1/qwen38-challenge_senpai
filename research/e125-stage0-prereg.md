# E125 Stage 0 - pre-registered prediction for thorfinn's rung 5e

Written before rung 5e exists. Nothing in this file is fitted to a number that
has not already been published in the campaign ledger or in a merged PR.

- Branch: `qwen-askeladd/e125-isolated-to-in-situ-transfer-law`
- Base at time of writing: `3f40d9b0` (E121, E122, E123 all merged)
- `harness=local` for every measurement of mine quoted here. Every ranked
  number is labelled `harness=ranked` where it is one.
- Honesty flags for the E123 session that supplies my new evidence:
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `timing_valid=false`, `official_or_ranked_score=false`.

## 0. New evidence, produced at zero GPU cost before writing the prediction

`research/e125_bandwidth_scan.py` reads the already-published E123 session
(`research/e123-artifacts/rate.json`, 20 cells, 5 shapes x NA 2..5) and reports,
per cell, the achieved read bandwidth of the unmodified kernel and the measured
effect of a real deletion of the activation add tree (`n_nosums`).

Achieved bandwidth is `read_bytes / seconds(a_base)`, the same formula the
E123 validity gate uses. `phi` is that value over the 273 GB/s M4 Pro peak.
NA=5 is excluded from pricing: every arm spills at NA=5 on `applegpu_g16s` and
fails exactness (E123 spill rule).

| NA | median GB/s | median phi | `n_nosums` gain, % of `a_base` | instructions deleted | **% per instruction** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 231.7 | 0.849 | 0.199 to 1.012 | 38.1 | **0.0081** |
| 3 | 221.3 | 0.811 | 0.645 to 1.415 | 56.7 | **0.0162** |
| 4 | 192.3 | 0.704 | 6.913 to 7.541 | 76.8 | **0.0942** |

Correlation of the deletion gain with achieved bandwidth:

| slice | n | r | reading |
| --- | ---: | ---: | --- |
| pooled, NA 2..5 | 20 | **-0.862** [-0.944, -0.678] | matches alphonse's -0.938 |
| within NA=2 | 5 | **-0.975** | strong H6 |
| within NA=3 | 5 | **-0.950** | strong H6 |
| within NA=4 | 5 | **+0.523** | **flat, wrong sign for H6** |
| `q_scaffold` null, pooled | 20 | +0.114 [-0.346, 0.530] | null holds |

At NA=4 the gain spans 1.09x over a bandwidth span of 1.25x. At NA=3 and NA=2
the same deletion in the same session is strongly bandwidth ordered.

**This reproduces both sides of the live contradiction in one session.**
Thorfinn reads flat at NA=4 (7 shapes, mean 5.85 %, sd 0.30 pp) and bandwidth
dependent at NA=3 (`gain % = 12.880 - 0.05069 x GB/s`, `r2 = 0.9532`, ledger
269 / section 39004-39015). I read exactly the same switch with a different
mechanism, a different instrument and a different arm set.

**Registered interpretation.** H6 is real but it is not a smooth law over
`phi`. It is a regime switch. Above about `phi = 0.8` the cell is bandwidth
bound, an instruction deletion is largely hidden, and the payout is ordered by
how close the cell sits to its roofline. Below about `phi = 0.75` the cell has
left the roofline, issue binds, and every shape pays the full instruction price
regardless of its residual bandwidth differences. The per-instruction price
moves 11.6x across `phi` 0.849 to 0.704.

**Consequence for the contradiction.** Alphonse's 10 live cells are 5 shapes x
{NA=3, NA=4} with NA=2 and NA=5 pinned to zero. That set straddles the switch,
so a pooled correlation over it is dominated by the width axis rather than by a
per-shape roofline law. His two low-bandwidth shapes are also his two narrow-N
shapes, `mlp_down` (N=5120) and `gdn_out_proj` (N=5120), which launch 640
threadgroups in y against 1792 to 4352 for the other three. Bandwidth, width and
launched grid volume are collinear in his design, so his `r = -0.938` cannot
separate H6 from H4. I take thorfinn's side at NA=4 and alphonse's side at
NA<=3.

**Caveat I register against my own evidence.** The NA=3 to NA=4 step in the
per-instruction price is 5.8x, which is too large for a pure roofline effect.
E123 already found that deleting the whole add tree buys one extra resident
simdgroup on `applegpu_g17s`. An occupancy step at NA=4 would inflate that row.
Stage 1 must separate an occupancy step from a roofline effect, and until it
does, the NA=4 per-instruction price is an upper bound (rule 61).

## 1. The correction model I am registering

```text
predicted in-situ effect =
    isolated per-cell effect
  x (1 / W)      width and cell weighting term
  x (1 / F)      frame term
  x share        E116 share term, unchanged at 0.6068 x 1.000
```

`share` is untouched. The E116 null control is satisfied by construction: my
correction multiplies the per-cell effect only and never enters `alpha x beta`,
which stays at 1.000 [0.963, 1.038].

### W, the weighting term

**Fitted to nothing.** It is arithmetic on published per-width values.

The 7x7 grid gives `ranked %` by width as `0.01, 2.36, 2.78, 1.32, 2.37, 3.28,
1.44` for M=3..9. That profile is not smooth. It dips at M=6 and M=9, which is
the signature of campaign rule 58's launched-volume trough, not of the
mechanism.

Evaluating the advisor's own median construction on the **new** grid rather than
on the older smooth ledger table:

```text
median ranked % = 0.484 x f(beagle mean M = 5.382) + 0.516 x f(second median prompt)
f(5.382)  = 2.78 + 0.382 x (1.32 - 2.78) = 2.222
f(5.989)  = 1.336   republic
f(6.087)  = 1.411   essays
f(6.256)  = 1.589   medicine
f(7.148)  = 2.505   botany
```

With the mean of the four candidates for the second median prompt, 1.710:

```text
isolated ranked, recomputed on the 7x7 grid = 0.484 x 2.222 + 0.516 x 1.710
                                            = 1.96 %   band [1.76, 2.37]
```

**The carried +2.47 % headline is already 1.26x too high on thorfinn's own new
grid**, before any frame question. The old grid had `f(6.0) = 2.753`; the new
one has 1.32.

A second, separate weighting error remains: `f(E[M])` is not `E[f(M)]` when `f`
swings 2.1x between adjacent widths. A realised-width histogram with mass below
M=4 lowers the aggregate further. I set

```text
W = 1.33   band [1.00, 1.76]
```

where 1.00 is "thorfinn already integrated the realised histogram" and 1.76 is a
truncated-geometric realised histogram capped at M=9 and tuned to reproduce
beagle's mean width of 5.382, which gives `E[f] = 1.263` against
`f(E[M]) = 2.222`. At beagle's published accept rate of 0.834 the same
construction gives 1.91. The central value is the geometric
mid-point. **Thorfinn can collapse this term to a point by stating which
aggregation produced his headline.**

### F, the frame term

**Partly fitted, and I say so.** Its upper endpoint is alphonse's raw ratio, so
any prediction of his own number using the top of this band is circular.

| frame-transfer point | class | isolated | in situ | factor |
| --- | --- | ---: | ---: | ---: |
| rule 58, grouped against isolated dip | launched-volume / grouping | 22.75 % | 13.12 % | **1.734** |
| alphonse E121 rung 2 to rung 3, raw | threadgroup exchange | -0.890 % | -0.436 % | **2.04** |
| alphonse E121, after his own cell re-weighting | threadgroup exchange | -0.545 % | -0.436 % | **1.25** |

```text
F = 1.43   band [1.00, 2.04]
```

1.43 is `sqrt(1.00 x 2.04)`. The lower endpoint is the null hypothesis that
there is no frame term once weighting is right, which alphonse's own residual
of 1.25 at 1.2 sd does not reject.

### The combined correction

```text
C = W x F = 1.33 x 1.43 = 1.90        80 % band [1.30, 2.70]
                                      full envelope [1.00, 3.59]
```

### What the "three observations near a factor of two" actually are

The brief lists three observations. **Two of them are not frame transfers and
must not be pooled with the third.**

| # | what it compares | frame | error term |
| --- | --- | --- | --- |
| 1, E118 ladder 1.66 | injection price against deletion price | both isolated | additivity |
| 2, E123 census 117.2 % | sum of injection prices against the whole kernel | both isolated | additivity |
| 3, alphonse 2.04 | isolated deletion against in-situ deletion | isolated to in situ | frame + weighting |

Observations 1 and 2 measure the ladder's additive model failing inside the
isolated frame. E123 already showed that this term is not one number: the same
add tree gives 2.164, 1.116 and 1.420 depending on which half is deleted.

**This matters directly for Route B.** Route B is priced from a direct isolated
measurement of the actual change, not from an injection census, so the
additivity term does not apply to it. Only `W` and `F` apply. Applying a
factor-of-two "everything is over-priced by two" rule to Route B would double
count.

## 2. Registered prediction for thorfinn's rung 5e

**Primary: rung 5e lands at +1.03 % ranked.**

```text
1.96 % isolated, recomputed on the 7x7 grid
/ 1.90 combined correction
= 1.03 % ranked, harness=ranked frame, in-situ
```

| interval | low | high |
| --- | ---: | ---: |
| 80 % | 0.73 | 1.51 |
| full envelope | 0.49 | 2.37 |

Against the decision lines:

| line | value | verdict |
| --- | ---: | --- |
| parity with the crown | +0.53 % | **cleared**, the whole envelope is above it except its floor |
| beating the crown on either mode draw | +1.86 % | **missed**, only the extreme top of the envelope reaches it |

`P(rung 5e >= 0.53 %) ~ 0.9`. `P(rung 5e >= 1.86 %) ~ 0.05`.

### Conditional branches, labelled

| branch | assumption | F | predicted rung 5e |
| --- | --- | ---: | ---: |
| **T**, primary | thorfinn's flat NA=4 reading is right; alphonse's `r` is collinear with width and N | 1.25 | **+1.18 %** |
| **A** | alphonse's roofline law is a real per-shape law that transfers | 2.04 | **+0.72 %** |
| **N** | there is no frame term at all once weighting is right | 1.00 | **+1.47 %** |
| **R** | rule 57 branch, see below | n/a | **near 0** |

Branch T is my primary because my own 20 cells reproduce the NA=4 flatness with
a different mechanism.

### Branch R, the cheapest thing anyone can check today

Campaign rule 57 forbids weighting an isolated per-group cell by a realised
width histogram unless the dispatch grouping matches. Beagle's ledger row
carries `G 2`: its rounds dispatch two groups, so a realised width of 5.382
ships as roughly two partitions of 2.7, not as one dispatch at M=5.4. If the
7x7 grid cells are single dispatches at M=3..9, then the ranked-relevant cell is
near M=3, where thorfinn's own grid reads `f(3) = 0.01`.

I give branch R low weight because Route B is dispatched from `Qwen35.swift` and
may already be in the shipped grouping. **But thorfinn can settle it in one
sentence, and if it holds, every number above is void.** I register it now so
that nobody discovers it after rung 5e lands.

## 3. Prediction for alphonse's E121 in-situ figure, as a fitting check

Applying the **same** `W = 1.33` and `F = 1.43` to his shipped isolated
prediction, with no term tuned to him:

```text
-0.890 % / 1.33 / 1.43 = -0.468 % leg
```

Measured: **-0.436 %**, sd 0.093, n=2. My model misses by 0.032 pp, which is
0.34 sd.

**Disclosure, because this check is weaker than it looks.** `F`'s upper endpoint
is his own raw ratio of 2.04, so `F` is not independent of his point. The honest
statement is that `W` is independent of him and `F` is anchored half on him and
half on unity. The check therefore shows that a factor derived mostly from
width-weighting arithmetic plus a unity prior does not need a further free
parameter to reach his number. It does not show that the factor is a law.

Campaign rule 61 and the brief's own stop rule both apply: **there are two
independent in-situ frame-transfer points in this campaign, not three. I report
1.90 as a ratio with an interval, not as a law.**

## 4. Edward's E124 island deletion

**No correction. The correction does not apply to it.**

The ledger prices `noislands` at +0.43 to +0.49 % ranked through the Finding 85
class coefficients: 0.24 for draft-path head bytes and 0.95 for draft-path
dispatch deletions. Those coefficients are already measured transfers, not
isolated per-cell instruction prices, and rule 69 forbids mixing classes. My
`C = 1.90` is a per-cell instruction-price correction and applying it here would
double count.

One flag: the ledger marks the `+0.304 %` dispatch residual `[INFERRED]`. That
component, not the frame, is where E124's uncertainty lives.

## 5. Stage 1 arms

One session, ABBA counterbalanced, using `research/e118_qmv_probe.m` unchanged
as the measuring instrument and alphonse's `research/e121_e2e_abba.sh` idle-pad
and per-arm balance report for harness defect 25.

Two instruction classes, chosen far apart on my own price table:

- **arithmetic deletion**, `n_nosums`, the whole activation add tree;
- **device-load injection**, `k_ld16`.

Two widths, NA=3 and NA=4, because that is exactly where my scan puts the regime
switch. Plus `q_scaffold` as the null and the standard positive controls.

| frame | construction | H it tests | registered prediction |
| --- | --- | --- | --- |
| **A** | standard back-to-back isolated probe | baseline | reproduces E123 within the session floor |
| **B** | weight buffer cycled across enough distinct allocations that no launch reuses a resident one | H1 | **pre-registered null.** The weight buffers are 36.7 to 635.7 MB, far past any M4 Pro cache, so no launch can already be cache resident. If frame B moves the effect by more than the session floor, my allocation-size argument is wrong and H1 is back |
| **C** | co-scheduled bandwidth consumer that raises achieved bandwidth toward the roofline | H3, H6 | under H6 the NA=4 gain shrinks and the shrink tracks the measured change in `phi`, not the frame label. Under H3 alone it shrinks without tracking `phi` |
| **D** | the opposite push: shrink K so the weight stream fits in cache and `phi` falls | H6 | the NA=3 gain grows toward the NA=4 per-instruction price. Two directions on one covariate give a slope instead of a point |

Frames C and D together are the discriminator. H6 predicts that the effect is a
single function of `phi` in both directions. H3 predicts that only C moves. H1
predicts that B moves. H5 predicts that the two instruction classes disagree
whatever the frame.

Every arm in every frame reports achieved bandwidth as a continuous covariate,
both endpoint launched grid volumes per rule 58, and entry and exit GPU
temperature with the entry-temperature spread beside the effect.

### Registered stop rules

- Stop and report if frames A, C and D agree within the session floor. That
  sends the whole question to H5 and to the class table.
- Kill the correction if applying it to the share term moves `alpha x beta`
  outside [0.963, 1.038]. It cannot, by construction, and I will show the
  arithmetic rather than assert it.
- Report the Stage 2 correction as a function of `phi` with the class table as
  the marginal summary. Report a scalar per class only if it fits as well, and
  say so explicitly.
- Any row of the Stage 2 table with fewer than two independent points is
  written "not measured".
