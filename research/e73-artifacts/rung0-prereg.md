# E73 rung 0 pre-registration

Written and committed BEFORE any E73 compile ran. Head at writing time:
`d09b76a1da64f4d7cab0866d74a1a384e49e9163`. Host: Apple M4 Pro, 20 GPU cores,
macOS 26.5.2, `ip-10-231-2-22`. Harness label for everything in rung 0:
`harness=compile-probe`. No GPU dispatch, no timing.

## What rung 0 compiles

One entry point per legal `(M, IPG)` pair, each calling the shipped wrapper
`qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true>` at the frozen host
geometry. Legality is `3 <= M <= 9`, `2 <= IPG <= 6`, `M % IPG != 1`.

| M | legal IPG | tail `M % IPG` | inlined helper bodies (NA) |
|---|---|---|---|
| 3 | 3 | 0 | 3 |
| 4 | 2 | 0 | 2 |
| 4 | 4 | 0 | 4 |
| 5 | 3 | 2 | 3, 2 |
| 5 | 5 | 0 | 5 |
| 6 | 2 | 0 | 2 |
| 6 | 3 | 0 | 3 |
| 6 | 4 | 2 | 4, 2 |
| 6 | 6 | 0 | 6 |
| 7 | 4 | 3 | 4, 3 |
| 7 | 5 | 2 | 5, 2 |
| 8 | 2 | 0 | 2 |
| 8 | 3 | 2 | 3, 2 |
| 8 | 4 | 0 | 4 |
| 8 | 5 | 3 | 5, 3 |
| 8 | 6 | 2 | 6, 2 |
| 9 | 3 | 0 | 3 |
| 9 | 5 | 4 | 5, 4 |
| 9 | 6 | 3 | 6, 3 |

19 pairs. The `TAIL != 0` wrapper instantiates a SECOND helper body at
`max(TAIL, 2)`, so those kernels carry two accumulator sets in their static
code even though one thread executes only one of them.

## Predictions

Prior evidence I am extending: E69 reported 1 alloca at NA=4 and NA=5 and 2
allocas at NA=6 with the type `[4 x <6 x float>]`, for a probe that inlines ONE
helper body per entry point.

**P1 - spill set.** The only IPG value that spills the accumulator is IPG=6.
Every kernel whose main body is NA=6 (`m6_ipg6`, `m8_ipg6`, `m9_ipg6`) shows at
least one alloca whose element type is an array of 6-wide float, matching
`[4 x <6 x float>]`. No kernel with main body NA <= 5 shows an array-of-vector
float alloca.

**P2 - IPG=5 does not spill.** `m5_ipg5` shows exactly 1 alloca and no
`<5 x float>` array alloca. IPG=5 is the largest non-spilling width and
therefore carries the highest peak live register count of the non-spilling
arms.

**P3 - monotone live state below the cliff.** Peak live registers across the
k loop rise monotonically with IPG for IPG in {2, 3, 4, 5}, with a slope near
the 9 floats per unit IPG the source implies (`acc[4]`, `partial[4]`, `sums`,
all `vec<float, IPG>`).

**P4 - the tail body adds its own alloca.** A kernel with `TAIL != 0` has an
alloca count equal to the sum of its two bodies' counts. Concretely I predict
`m8_ipg6 = 3`, `m9_ipg6 = 3`, `m9_ipg5 = 2`, `m7_ipg5 = 2`, `m7_ipg4 = 2`,
`m5_ipg3 = 2`, `m6_ipg4 = 2`, `m8_ipg3 = 2`, `m8_ipg5 = 2`, against 1 for every
`TAIL == 0` kernel with main body NA <= 5 and 2 for `m6_ipg6`.

**P5 - the toolchain hides the cliff.** `maxTotalThreadsPerThreadgroup` is 1024
for all 19 kernels, including the IPG=6 kernels. The Apple compiler spills to
scratch rather than lowering the threadgroup limit, so the only public
occupancy figure the toolchain exposes will NOT separate the arms. If this is
wrong and the limit drops at IPG=6, the toolchain does expose the cliff and P5
is refuted.

**P6 - resident simdgroups per core is a derived quantity, not a measured one.**
No Apple tool on this host reports register file occupancy. Any
simdgroups-per-core number in the rung-0 table is derived from the AIR peak
live count and a stated register file size, and is marked derived.

## How I will report

Hits and misses will be listed one prediction at a time against the compiled
census, before any interpretation. A refuted prediction is reported as refuted.
