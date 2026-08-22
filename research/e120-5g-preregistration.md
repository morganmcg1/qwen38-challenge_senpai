# E120 rung 5g pre-registration — Route B measured against the E121 control

Written and committed **before** the post-merge session runs, and before the
pre-merge session (`research/out/e120-5e-abba`, job `ff26787e`) had produced its
headline. Advisor feedback of 2026-08-22T05:56Z requires the 5e control arm to
be the shipped `SHARE_SUMS` body that E121 merged at `3f40d9b0`.

## The identity that makes this cheap

My candidate arm dispatches my own kernel at every width the gate routes
(`tablePays(m:) = m >= 4`). E121 changes only
`qmv_fast_crossrow_affine4_g64_wide`, which my candidate does not execute at
those widths. So the merge moves the **control** and leaves the **candidate**
where it is:

```
leg_effect(post-merge) = leg_effect(pre-merge) - E121_leg_effect(realised widths)
```

Both sessions run the same host, fixture, window, head and session design, so
differencing them measures E121's in-situ leg value at the widths this decode
actually reaches. That is the quantity alphonse is measuring in isolation in
E126.

## What E121 actually deletes, read from the merged source

```c
constexpr bool SHARE_SUMS = NA <= 4;
constexpr int  H          = NA / 2;
const bool     own_lo     = simd_gid == 0;
const bool     owns_m     = !SHARE_SUMS || ((m < H) == own_lo);
```

Two simdgroups split the `m` range at `H = NA / 2`, each accumulates only its
own `sums[m]`, and they exchange through `sums_xchg` with **two
`threadgroup_barrier` calls per k-block**.

`H = NA / 2` is integer division, so the split is uneven at odd NA and the
critical path is the larger half:

| IPG (= NA) | H | simd 0 owns | simd 1 owns | critical path | sums arithmetic deleted |
| --: | --: | --: | --: | --: | --: |
| 3 | 1 | 1 of 3 | 2 of 3 | 2/3 | **1/3** |
| 4 | 2 | 2 of 4 | 2 of 4 | 1/2 | **1/2** |
| 5 | — | `SHARE_SUMS` false | — | 1 | **0** |

Route B deletes the whole tree at every width and pays no barrier. So Route B's
residual value over E121 is the undeleted remainder **plus** the two barriers
per k-block that E121 adds and Route B does not.

At K = 5120 the wide kernel runs `5120 / (16 * 32) = 10` k-blocks, so E121 costs
20 threadgroup barriers per dispatch that Route B removes.

## Width map for this experiment

The shipped dispatch switch pairs `(M, IPG)` as
`[(3,3), (4,4), (5,5), (6,3), (7,4), (8,4), (9,3)]`.

| M | IPG | E121 shares | my gate routes | contributes to the post-merge difference |
| --: | --: | :-: | :-: | --- |
| 3 | 3 | yes | no | no — both arms run the E121 body |
| 4 | 4 | yes | yes | yes, control faster by ~1/2 the tree |
| 5 | 5 | no | yes | **no change — control identical** |
| 6 | 3 | yes | yes | yes, control faster by ~1/3 the tree |
| 7 | 4 | yes | yes | yes, control faster by ~1/2 the tree |
| 8 | 4 | yes | yes | yes, control faster by ~1/2 the tree |
| 9 | 3 | yes | yes | yes, control faster by ~1/3 the tree |

## Prediction

Askeladd's `n_nosums` diagnostic deletes the tree and returns the wrong answer
at **+7.91 %** of the wide kernel at NA=4, which is the best available estimate
of the whole tree's cost. Applying the deleted fractions above and the
64-token realised histogram {5:1, 6:4, 7:4, 8:1}:

```
M=5  weight 0.1   deleted 0     ->  0.00 %
M=6  weight 0.4   deleted 1/3   ->  2.64 %
M=7  weight 0.4   deleted 1/2   ->  3.96 %
M=8  weight 0.1   deleted 1/2   ->  3.96 %
histogram-weighted, before barrier cost      3.04 % of the wide kernel
```

Alphonse measures **+1.463 %** round-weighted in the kernel frame, about half of
the barrier-free bound, so the two barriers per k-block plausibly cost about
half of what the sharing saves.

**Point prediction.** Taking E121 at its measured 1.463 % kernel-frame value,
applying the E116 transfer of 0.6070, and applying the same isolated-to-in-situ
attenuation of 1.95 that my own mechanism showed in the 64-token validation:

```
E121 leg effect, in situ  =  1.463 x 0.6070 / 1.95  =  0.455 %
predicted post-merge leg  =  (pre-merge leg) - 0.455 %
```

If the pre-merge 512-token leg effect lands near the 64-token validation's
1.127 %, the post-merge leg effect is predicted at **+0.67 % leg, +0.64 %
ranked**, just above the advisor's +0.53 % parity line.

## Falsifiers, written before the measurement

1. **Overlap larger than predicted.** Post-merge leg effect below +0.56 %
   (ranked +0.53 %) means Route B does not clear parity once E121 is in the
   control, and Route B is not independently submittable. Report it as that.
2. **Overlap smaller than predicted.** A post-merge drop of less than 0.25 %
   leg means E121's barriers cost more in situ than its sharing saves at these
   widths, and Route B keeps most of its value. Say so, and say that it implies
   E121's own in-situ value is below its isolated 1.463 %.
3. **No drop at all, or a rise.** That would contradict the identity above and
   indicates a measurement or provenance fault, not a result. Do not report it
   as a win; re-check the worker sha256, the arm certificate and the realised
   width histogram first.

## What does not change

The candidate arm's absolute seconds per token must be **statistically
indistinguishable between the two sessions**, because the merge does not touch
any code it executes at M >= 4. That is a free provenance check on the whole
pair, and I will report it. A material move in the candidate arm invalidates the
differencing and I will say so rather than difference anyway.
