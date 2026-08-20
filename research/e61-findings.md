# E61 findings — single weight stream at M=6

Measurements only. Every prediction they are scored against was written first
in `research/e61-prereg.md` and committed before the corresponding rung ran.

Host: Apple M4 Pro, 48 GiB, macOS 26.5.2, Swift 6.3.3.
Base: `d2139c924c7a7d98ca6026eea63867c2776abbca`.

## Rung 0.2 — register census (CPU only, ~28 s)

Artefact: `research/e61-artifacts/e61-reg-census.json`.

The preregistered law
`registers = 20 + 21*max(NA) + 4*[the cell has two distinct group sizes]`
is **exact, residual 0, on all seven shipped cells**. Shipped table maximum
**129**; `entry_batch0` **181**, hitting the point prediction.

The law then **breaks above NA=5**:

| cell | law | measured | residual |
| --- | --- | --- | --- |
| `<T,6,6>` | 146 | **144** | -2 |
| `<T,7,7>` | 167 | **157** | -10 |
| `<T,8,8>` | 188 | **177** | -11 |

So the register-ceiling rise the assignment prices at +17 (129 -> 146) is
actually **+15 (129 -> 144)**, and the law over-predicts increasingly with NA.
The `+4` mixed-group term reproduces this instrument's documented over-count on
mixed cells, so the hardware ceilings of mixed cells are probably 4 lower than
reported; that offset cancels in every within-instrument comparison here.

Per-arm table maximum and `entry_batch0`:

| arm | table max | entry_batch0 |
| --- | --- | --- |
| `shipped` | 129 | 181 |
| `t6` | 144 | 202 (predicted interval 196-205, hit) |
| `iso_m6_ipg3` | 83 | 135 |
| `iso_m6_ipg6` | 144 | 197 |
| `ballast` | 144 | 197 |

`ballast` reaches the same table maximum as `t6` (144) with every scored route
unchanged, which is what rung 4 needs. Its `entry_batch0` is 197 against `t6`'s
202; that 5-register residual is reported rather than explained away.
`maxTotalThreadsPerThreadgroup` is 1024 for every arm, so no arm loses
occupancy at the threadgroup limit.

The census instrument was corrected mid-rung to scan every instantiated cell
(`instantiated_cells`) instead of only widths 3-9. All numbers above come from
the corrected instrument.

## Rung 0.1 — `vec<float,N>` layout and lane fidelity

Artefacts: `research/e61-artifacts/e61-vec-probe-{mlxmatch,fastmath40,fastmath31,nocontract31}.json`.

Layout, measured on this device and identical under all four builds:

| N | sizeof | alignof |
| --- | --- | --- |
| 2 | 8 | 8 |
| 3, 4 | 16 | 16 |
| 5, 6, 7, 8 | **32** | **32** |

`lane_bleed_mask` and `lanewise_arith_fault_mask` are 0 at every N, all lanes
stay distinct at every N, and the NA=2 against NA=4 reference cross-check is
exact. So there is no aliasing or padding hazard in the wider vectors.

Lane fidelity — lane `m` of an NA=N run against the shipped NA=2 helper — is
**build-dependent**, and that is the whole result of this rung:

| variant | metal flags | NA=5 | NA=6 | NA=7 | NA=8 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `mlxmatch` | `-std=metal4.0 -fno-fast-math` | 0 | **0** | **0** | **0** | PASS |
| `nocontract31` | `-std=metal3.1 -ffp-contract=off` | 0 | **0** | **0** | **0** | PASS |
| `fastmath40` | `-std=metal4.0` | 0 | 128 | 128 | 128 | FAIL |
| `fastmath31` | `-std=metal3.1` | 0 | 128 | 128 | 128 | FAIL |

(values are max ulp over the lanes; 12 of 12 positive controls were caught in
every variant, so the probe discriminates in all four builds.)

The first probe pass used `-std=metal3.1 -O2` with `xcrun metal`'s **default
fast-math**, and reported a 128-ulp mismatch from NA=6 up. That is an artefact
of the probe's own build, not a property of the vector layout. Under fast-math
the compiler is free to contract and reassociate differently once the vector
crosses the 32-byte form, and the per-lane pattern in `fastmath31`
(`[3, 18, 32, 10, 32, 128, 57, 1]` at NA=8) is the signature of reassociation,
not of a lane collision.

MLX's runtime JIT does neither thing the failing builds do.
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp:631-632`,
`Device::build_library_`:

```cpp
options->setFastMathEnabled(false);
options->setLanguageVersion(get_metal_version());
```

and `get_metal_version()` (same file, lines 37-49) returns
`MTL::LanguageVersion4_0` on macOS 26. The `mlxmatch` variant reproduces both,
and it is **exact at NA=6, 7 and 8**.

Conclusion for rung 0.1: the wider single-stream accumulator is numerically
sound on the build the scored kernel actually gets. This is necessary, not
sufficient — rung 2 still has to prove bitwise exactness end to end at 512
tokens, because this probe exercises the accumulator shape, not the shipped
kernel body. A practical corollary: **any local Metal check of this kernel must
pass `-fno-fast-math`**, or it will report mismatches the runtime never sees.

## Rung 1 — lone-group bandwidth ladder extended to NA=6 and NA=7

Artefact: `research/e61-artifacts/e61-bandwidth.json`. Six legs in one session,
order `shipped, t6, t7, t7, t6, shipped`, widths 1-9, 21 reps, 10 inner. That
palindrome counterbalances all three arms against monotone drift; the plain
ABBA in the assignment would not counterbalance `t7`. Measured stream peak
227.9 GB/s.

**Gate result: PROCEED. Both clauses fired and every positive control passed.**

### The ladder

| NA | measured | E54 anchor | control | predicted | miss | break-even | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 218.7 | 223.8 | PASS (-2.28 %) | | | | |
| 3 | 199.6 | 199.7 | PASS (-0.07 %) | | | | |
| 4 | 175.9 | 175.2 | PASS (+0.40 %) | | | | |
| 6 | **117.8** | | | 126.65 | **-8.9** | 114.00 | **CLEARS** |
| 7 | **97.9** | | | 102.35 | -4.4 | 106.55 | **fails** |

All three positive controls are inside the preregistered 2.5 % band, so the
ladder extension is valid.

**Linearity is refuted at NA=6.** The miss is -8.9 GB/s against a preregistered
+-5.0 GB/s tolerance. The ladder bends downward past NA=5, so the assignment's
linear model was optimistic. It did not bend far enough to close M=6: 117.8
still clears the 114.00 break-even, but with only **3.8 GB/s of headroom, 3.3
%**, where the linear model promised 12.65. At NA=7 the miss is -4.4 GB/s,
inside tolerance, so linearity is not refuted there.

### What this closes permanently

`bw(7) = 97.9 GB/s` is **below** M=7's 106.55 break-even, so single-stream M=7
does not pay. The ladder decreases monotonically, so `bw(8) <= 97.9 < 100.01`
and M=8 does not pay either. Extrapolating the measured `6->7` step of -19.85
puts `bw(9)` near 78 GB/s against a 92.56 break-even, so M=9 does not pay.
**Single-stream M=7, M=8 and M=9 are closed.** M=9 already runs two streams as
`<T,9,5>` from E55, and this result says that is the right choice.

The direct M=7 cell measurement agrees: `t7` is **+7.13 %** slower than shipped
at M=7, against a predicted +4.16 %. Both the model and the measurement reject
M=7, and the measurement rejects it harder.

### The direct M=6 cell measurement

`t6` minus `shipped`, per width:

| M | routing | delta | note |
| --- | --- | --- | --- |
| 2 | `[(2,)]` unchanged | -0.48 % | |
| 3 | `[(3,)]` unchanged | -0.20 % | |
| 4 | `[(4,)]` unchanged | +0.07 % | |
| 5 | `[(3,2)]` unchanged | -0.13 % | |
| **6** | `[(3,3)] -> [(6,)]` | **-4.20 %** | predicted -9.95 % |
| 7 | `[(4,3)]` unchanged | -0.11 % | |
| 8 | `[(4,4)]` unchanged | +0.15 % | |
| 9 | `[(5,4)]` unchanged | +0.53 % | |

The M=6 cell is **-4.20 %**, comfortably past the -2.0 % gate clause but only
**0.42x** the modelled -9.95 %. The bandwidth shortfall explains the shortfall
in the cell: the model assumed a lone NA=6 group would sustain 126.65 GB/s and
it sustains 117.8.

### Rung 4 answered without a ballast arm

The seven untreated widths give **mean -0.02 %, max |delta| 0.53 %**, inside
the preregistered +-1.0 % band. `t7`'s untreated widths agree: mean -0.06 %,
max |delta| 0.42 %.

This is the rung 4 decomposition, measured directly on the real routes. Raising
the shared table maximum from 129 to 144 registers costs **nothing measurable at
any width whose routing did not change**. The whole -4.20 % at M=6 is therefore
attributable to the algorithm, not net of a ceiling tax paid elsewhere. The
`ballast` arm was built to price this term and is **not needed**; I preregistered
exactly this substitution. `ballast` stays in `research/e61_arms.py` unused.

### Whole-leg projection

`research/e61-artifacts/e61-projection.json`, from `research/e61_project.py`.

Recomputing the local width shares from **this session's own** `shipped`
per-width costs gives `f6 = 0.2673`, against the prereg's 0.2671 derived from
E54 costs — an independent confirmation of the share, not a reuse of it. M=9
remains the largest local cell at 50.44 %.

```
QMV time              0.2673 * (-4.20 %)      = -1.122 %
MTP leg               x psi_mtp 0.693391      = -0.778 %
after E55 realisation x 0.946                 = -0.736 %
```

That is **11.7x the 0.0629 % local null floor** and 2.5x the -0.30 % promote
threshold. Using the assignment's ranked M=6 share band of 30.9-34.7 % instead
of the local share gives **-0.85 % to -0.96 %**.

This is a projection from a microbenchmark cell, not a measurement. Rung 3
measures the whole leg directly and that measurement decides.
