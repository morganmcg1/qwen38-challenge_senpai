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
