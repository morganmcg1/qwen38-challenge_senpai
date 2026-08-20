# E76: fit the scored cell into the ranked occupancy tier

Base `41ddc183817979be8d2f0817d79f98b2ddefb984`. Host Apple M4 Pro,
`applegpu_g16s`. Ranked host is `applegpu_g17s`. No candidate file changed: this
experiment did not touch `quantized.h`, its generated twin, the wrapper, or
either dispatch table. Every arm lives in a generated research header.

## Answer

**The 91-register bar is reachable at both scored widths, and it is not worth a
ranked slot.**

`rps1lazy` allocates **75** g17s registers at NA = 5 and **70** at NA = 6, which
is 23 and 41 registers below the shipped cells and 15 and 20 below the crown's
90. Bit-identity holds: zero differing elements on all seven scored shapes at
both widths.

The cost is +64.71 % and +49.76 % per target-verify round on this host, measured
behind the real 40 C gate. The cheapest arm that clears the bar at either width
costs +17.14 %. The public board's top eight scores span 0.564 % and a single
ranked run has a standard deviation of 0.756 %.

The pre-registered section 6 conclusion holds, but its stated antecedent does
not. The one-group partition is **not** structurally unavailable at rank. It is
available and it is priced, and the price is not close. The ranked-optimal
dispatch table is the crown's at M = 5, 6 and 9 and ours nowhere.

## 1. Why the register steps are +7, +1, +7, +13

`metal::vec<float,N>` pads to a power-of-two lane count. Read out of the front
end with `research/e76_vec_layout.py`, no GPU and no backend:

| N | `sizeof` | `alignof` | float slots | padding lanes |
|---:|---:|---:|---:|---:|
| 2 | 8 | 8 | 2 | 0 |
| 3 | 16 | 16 | **4** | 1 |
| 4 | 16 | 16 | 4 | 0 |
| 5 | 32 | 32 | **8** | 3 |
| 6 | 32 | 32 | 8 | 2 |

The accumulator peak holds thirteen `VF` values: `acc[4]`, `partial[4]`, `sums`
and `a0..a3`. Their padded footprint is 26, 52, 52, 104, 104 slots, not the 26,
39, 52, 65, 78 a per-float model predicts.

| NA | padded slots | Δ slots | g17s regs | Δ regs |
|---:|---:|---:|---:|---:|
| 2 | 26 | — | 83 | — |
| 3 | 52 | +26 | 90 | **+7** |
| 4 | 52 | **0** | 91 | **+1** |
| 5 | 104 | +52 | 98 | **+7** |
| 6 | 104 | 0 | 111 | +13 |

NA = 3 already pays for four lanes, so NA = 4 is free. The +7 steps land exactly
where the lane class widens and the +1 lands where it does not.

The fourth step is measured on a different schedule and is not comparable. At
NA = 6 the k-loop stops unrolling fully: `plain` g17s `__TEXT` runs 4624, 5880,
7186, 8472 bytes at NA = 2..5 and then **collapses to 5064** at NA = 6.

One bound on the claim. If the allocator reserved eight lanes for each of
thirteen live vectors, NA = 5 would need at least 104 registers. It allocates
98, so the backend does scalarize part of the padded state. The padded lane
class explains where the steps are, not the absolute level.

### The layout arms

`facc` replaces `VF acc[rows_per_simd]` with `float acc[rows_per_simd][NA]`.
`fall` extends the same treatment to `acc`, `partial`, `sums` and `a0..a3`.

| arm | NA=2 | NA=3 | NA=4 | NA=5 | NA=6 |
|---|---:|---:|---:|---:|---:|
| `plain` (= shipped) | 83 | 90 | 91 | 98 | 111 |
| `facc` | 83 | 90 | **93** | 98 | 111 |
| `fall` | 95 | 98 | 101 | **100** | 108 (48 B spill) |

`facc` is inert. At NA = 5 it emits byte-identical g16s machine code to `plain`
(`0a5810b4`), because SROA rebuilds the same scalars from the flat array. At
NA = 4 it is 2 registers worse.

`fall` confirms the hypothesis and refutes the remedy. The NA = 4 -> 5 cliff
disappears, 101 -> 100, a step of -1 where `plain` steps +7. But `fall` raises
the NA = 2 floor by 12 registers and spills 48 bytes at NA = 6. The vector type
does cost the +7; removing it costs more than +7 elsewhere.

## 2. Rung 1: the register census

`research/e76-artifacts/rung1.json` and `rung1-table.md`. 26 arms x NA = 2..6 x
two architectures, in one JIT-string translation unit, from the `_wide` body
extracted verbatim from `quantized.h` lines 968-1066.

The census independently reproduces the advisor's wrapper grid on both
architectures at all five widths, from the bare `_wide` body rather than the
`M`/`IPG` wrapper:

| NA | bare `_wide` g16s / g17s | advisor wrapper census |
|---:|---|---|
| 2 | 70 / **83** | 70 / 83 |
| 3 | 93 / 90 | 93 / 90 |
| 4 | 94 / 91 | 94 / 91 |
| 5 | 95 / 98 | 95 / 98 |
| 6 | 96 / 111 | 96 / 111 |

`plain == shipped` machine-code digest passes on both architectures at all five
widths, so the control is the shipped object at every point in the table.
**NA = 2 = 83 g17s registers is recorded as the low-end anchor.** The g17s hard
ceiling is **126**, not 124; six chunk kernels clamp there and start spilling.

Arms at or below the 91 bar with no spill:

| NA | arms |
|---:|---|
| 5 | `rps1lazy` **75**, `rps1lazyfacc` 75, `rps1lazyfall` 75, `rps1lazyw` 76, `rps1fall` 80, `rps1lazysb` 88, `rps1mc4` 88 |
| 6 | `rps1lazyfall` **68**, `rps1lazy` 70, `rps1lazyw` 70, `rps1lazyfacc` 70, `rps2lazy` 86, `rps2lazyw` 88, `rps1fall` 90 |

The enabling lever is one the brief did not list. `packed[4][4]`,
`scale_local[4]` and `bias_local[4]` are loaded at the top of each k-block, but
each `packed[r][i]` is used at exactly one `i` and the scales are used only in
the block epilogue. Moving those loads to their use sites shortens the scalar
live ranges that sit across the accumulator peak. Alone it is worth 4 registers
at NA = 6. Combined with a halved row block it is worth 25.

The row-block loop unroll worry is resolved and is not the variable. `rps2nu`
and `rps1nu`, which hold the loop rolled with `#pragma clang loop
unroll(disable)`, emit machine code with the same digest as `rps2` and `rps1`.

The restructuring is not specific to the two scored widths. `rps1lazy`
allocates 55, 54, 64, 75, 70 g17s registers at NA = 2..6, so it lowers the whole
grid. Register pressure stops being the binding constraint everywhere and the
decision moves entirely onto throughput.

## 3. Rung 2: the cost, measured

One session per width, 21 reps, palindrome order, real 40 C gate, all seven
scored shapes, `target-bytes 24e9`, `harness=local`.

| | NA = 5 | NA = 6 |
|---|---|---|
| W&B run | `ag2u9pbu` | `9l4ysok9` |
| GPU temperature at the gate | 36.529 C | 61.113 C, cooled to pass |
| entry GPU temperature | 36.578 C | 39.784 C |
| exit GPU temperature | 63.609 C | 62.957 C |
| `cool_gate_passed_real_gate` | `true` | `true` |
| `gate_qualified_for_timing` | `true` | `true` |

Arms are compared within a width, in one process, in palindrome order, so the
3.2 C entry difference between the two widths never enters an arm comparison.

Priced through one target-verify round with the per-round dispatch counts from
`SCORED_SHAPES`:

| arm | g17s NA=5 | s/round NA=5 | vs `plain` | g17s NA=6 | s/round NA=6 | vs `plain` |
|---|---:|---:|---:|---:|---:|---:|
| `plain` (= shipped) | 98 | 0.084630 | — | 111 | 0.111994 | — |
| `rps2` | 96 | 0.096616 | +14.16 % | 100 | 0.117795 | +5.18 % |
| `lazy` | 97 | 0.097942 | +15.73 % | 107 | 0.118393 | +5.71 % |
| `rps2lazy` | 98 | 0.107767 | +27.34 % | **86** | 0.131192 | **+17.14 %** |
| `rps2lazyw` | 94 | 0.108823 | +28.59 % | **88** | 0.132007 | +17.87 % |
| `rps1` | 93 | 0.124884 | +47.56 % | 99 | 0.149836 | +33.79 % |
| `rps1lazyw` | **76** | 0.137363 | +62.31 % | **70** | 0.170631 | +52.36 % |
| `rps1lazy` | **75** | 0.139395 | **+64.71 %** | **70** | 0.167719 | +49.76 % |

Bold register counts clear the 91 bar.

**Correction to an earlier interim comment on this PR.** I wrote that every
scored shape lands within about 1.5 percentage points of its round figure. That
is wrong. The per-shape spread runs from 1.84 to 6.47 percentage points at
NA = 5 and from 2.38 to 15.21 at NA = 6. The claim the conclusion actually needs
is narrower and it does hold: **every arm that clears the 91 bar is slower than
the shipped cell on all seven scored shapes at both widths**, and the smallest
single-shape penalty among them is +15.53 %.

| arm | NA | per-shape min | per-shape max |
|---|---:|---:|---:|
| `rps1lazyw` | 5 | +60.26 % | +66.24 % |
| `rps1lazy` | 5 | +62.67 % | +68.86 % |
| `rps2lazy` | 6 | +15.53 % | +19.01 % |
| `rps2lazyw` | 6 | +15.57 % | +21.39 % |
| `rps1lazy` | 6 | +42.27 % | +52.59 % |
| `rps1lazyw` | 6 | +50.80 % | +53.18 % |

One arm does beat the shipped cell on one shape: `rps2` at NA = 6 is **-3.85 %**
on `mlp.down` while being +7.5 to +8.7 % on the other five priced shapes.
`mlp.down` is also where `rps1lazy` at NA = 6 is least bad, +42.27 % against
about +52 % elsewhere. `rps2` allocates 100 g17s registers and does not clear
the bar, so this does not change the recommendation, but it is the one shape
where the row block is not uniformly harmful and it is recorded as such.

**The local measurement is not a cost-without-benefit measurement.** On g16s,
`plain` at NA = 5 allocates 95 registers and `rps1lazy` allocates 64; at NA = 6,
`plain` allocates 96 with a 16-byte spill frame and `rps1lazy` allocates 72 with
none. The arm already collects a 31-register and a 24-register-plus-spill
reduction on this host, and it still loses 65 % and 50 %.

## 4. Why: the register route pays in the more expensive currency

Read straight off the shipped body with T = bfloat16, per lane per k-block, for
the whole four-row block:

- x: `4 i-steps * NA * 4 elements * 2 B`, times `4 / rows_per_simd` calls.
- packed weights: 32 B, invariant in `rows_per_simd`, once per group.
- scale and bias: 16 B, invariant in `rows_per_simd`, once per group.

The row block cuts registers by calling the body more times and each call
re-reads the whole x block. Splitting the proposal width into groups is the
mirror image: x is read once and the row side repeats per group.

| M | route | partition | rows_per_simd | g17s regs | B/lane/k-block | extra | extra B per register saved |
|---:|---|---|---:|---:|---:|---:|---:|
| 5 | shipped one group | [5] | 4 | 98 | 208 | — | — |
| 5 | crown partition | [3,2] | 4 | 90 | 256 | +48 | **6** |
| 5 | one group, `rps1lazy` | [5] | 1 | 75 | 688 | +480 | **21** |
| 6 | shipped one group | [6] | 4 | 111 | 240 | — | — |
| 6 | crown partition | [3,3] | 4 | 90 | 288 | +48 | **2** |
| 6 | one group, `rps2lazy` | [6] | 2 | 86 | 432 | +192 | **8** |
| 6 | one group, `rps1lazy` | [6] | 1 | 70 | 816 | +576 | **14** |

**The crown's group partition is 3.5x to 6x more register-efficient per byte of
extra traffic than any one-group body that clears the bar.** That ordering needs
no register-file constant and no occupancy model.

This is the direct answer to the premise behind the 91 bar. "Half the weight
traffic for one extra register" would be strict dominance if a one-group body at
91 registers were free. It is not. At NA = 5 the crown's split costs +48 bytes
per lane per k-block, which is +23.1 % of the one-group total. The cheapest
one-group body that clears the bar costs +480 bytes, which is +230.8 %. The
register route trades away 32 bytes of weight re-read and buys 480 bytes of x
re-read to do it.

## 5. Rung 3: the recommendation

**Do not spend a ranked slot on a one-group cell at M = 5 or M = 6.**

Modelled columns use the advisor's unverified
`floor(208 KiB / (128 B * regs))` occupancy model and are labelled as his.
"Conversion needed" is the fraction of the modelled occupancy gain that would
have to become throughput for the arm to pay for its measured cost.

| arm | NA | g17s regs | modelled resident (`plain` -> arm) | modelled gain | measured cost | conversion needed | can it pay? |
|---|---:|---:|:--:|---:|---:|---:|:--:|
| `rps1lazy` | 5 | 75 | 16 -> 22 | +37.5 % | +64.71 % | 173 % | **NO** |
| `rps1lazyw` | 5 | 76 | 16 -> 21 | +31.3 % | +62.31 % | 199 % | **NO** |
| `rps2lazy` | 6 | 86 | 14 -> 19 | +35.7 % | +17.14 % | 48 % | possible |
| `rps2lazyw` | 6 | 88 | 14 -> 18 | +28.6 % | +17.87 % | 63 % | possible |
| `rps1lazy` | 6 | 70 | 14 -> 23 | +64.3 % | +49.76 % | 77 % | possible |
| `rps1lazyw` | 6 | 70 | 14 -> 23 | +64.3 % | +52.36 % | 81 % | possible |

At NA = 5 the arithmetic is closed. Both qualifying arms need more than 100 %
conversion, so they cannot pay even if every extra resident simdgroup were free
throughput.

At NA = 6 the arithmetic is open but the case is still weak, for three reasons.
The required 48 to 81 % conversion is high for a kernel whose local occupancy
improvement was already large and returned nothing. The crown reaches the same
tier at 2 extra bytes per register instead of 8 to 14. And the crown's table
was measured at rank as -0.298 % against ours over eight prompts, so the
composable arm already exists and costs no restructuring.

The recommendation therefore holds at both widths, and M = 9 was not searched as
instructed.

## 6. The traffic-free route, refuted twice, and a fault in the compile oracle

If the bar can only be bought with x re-reads, the question is whether it can be
bought with nothing. One lever could: **chunk the proposal width inside the
k-loop.** The block still stages `packed`, `scale_local` and `bias_local` exactly
once and each chunk touches a disjoint set of `m`, so neither the weights nor x
are re-read. Only `acc` stays live across chunks; `partial`, `sums` and `a0..a3`
become chunk-local and drop into a narrower lane class. On the padded-slot model
this cuts NA = 5 from 104 slots to 56.

Arms `mc4`, `mc3`, `mc2` are greedy chunks of at most 4, 3 and 2. g17s registers
/ spill bytes:

| arm | NA=2 | NA=3 | NA=4 | NA=5 | NA=6 |
|---|---|---|---|---|---|
| `plain` | 83 / 0 | 90 / 0 | 91 / 0 | 98 / 0 | 111 / 0 |
| `mc4` | 83 / 0 | 90 / 0 | 91 / 0 | **126 / 240** | **126 / 288** |
| `mc3` | 83 / 0 | 90 / 0 | 126 / 144 | 126 / 176 | 126 / 224 |
| `mc2` | 83 / 0 | 116 / 0 | 122 / 0 | 126 / 240 | 126 / 288 |

Every chunked arm at NA = 5 and NA = 6 pins at the hard ceiling and spills. The
emitted text grows to about 9 to 10 KB, so the chunk bodies unroll separately
and the allocator pays for both live accumulator sets plus a doubled schedule.
Splitting a native `float4` is worse still: `mc3` at NA = 4 goes 91 -> 126 with
a 144-byte frame.

Where the partition is trivial the arm must be the shipped kernel, and it is.
`mc4` at NA = 2, 3 and 4, `mc3` at NA = 2 and 3, and `mc2` at NA = 2 all emit
byte-identical machine code to the shipped instantiation on both architectures.

### The chunk arms also fail device parity, and the cause is the backend

The full parity sweep is 518 device checks over 25 arms x NA = 5, 6 x seven
scored shapes. 476 pass with zero differing elements. **The 42 failures are
exactly `mc4`, `mc3` and `mc2`, at both widths, on all seven shapes.**

The failure signature is precise and identical everywhere. The fraction of
differing elements equals exactly the fraction of `m` values that are not in the
last chunk: `mc4` at NA = 5 is [4,1] and 80 % differ, `mc3` is [3,2] and 60 %
differ, `mc2` is [2,2,1] and 80 % differ. Every chunk except the last returns
wrong output and the last chunk is exact, independent of chunk width.

**The rewrite is not the cause.** `rps1mc4` carries the same three substitutions
on the smallest row block. It allocates 88 g17s registers at NA = 5 and 98 at
NA = 6 with **zero spill on both architectures**, and it returns **zero
differing elements over all seven scored shapes at both widths**, 4 647 808
elements. Every chunk instantiation that spills fails on every shape; the one
that does not spill passes.

This is a caveat the campaign should carry, because the compile oracle is now
load-bearing for kernel design. **A register census that reports a spill frame
is reporting more than a performance problem on this toolchain. Treat any
non-zero spill as a correctness risk and prove bit-identity on device before
believing a spilling arm.** A register count alone is not a safe gate there.

## 7. What could not be calibrated

`maxTotalThreadsPerThreadgroup` is the one public Metal field a register
allocation could move. It holds at 1024 for every kernel built here, from 14
registers up to 126 registers carrying an 816-byte spill frame. The runtime
spills instead of shrinking the threadgroup, so the field is not register-derived
on this host and gives no register-file estimate.

The step-1 live-float sweep shows no further allocator steps. The reported count
takes both odd and even values across 74 to 126 with no plateau, so the allocator
does not quantize the count. **Occupancy tiers remain unknown; rank variants by
raw register count.** Both backends allocate identically up to 64 live floats on
the scalar probe and diverge only where g16s runs out.

## 8. Suggested follow-ups, not implemented

1. **Measure the crown's two-dispatch cell directly.** Section 4 compares
   one-group arms against the one-group shipped cell, not against the crown's
   `[3,2]` and `[3,3]` split. At NA = 6 the one-group `rps2lazy` reaches 86
   registers at the same 2x row-side traffic the crown pays, which is 19
   modelled resident simdgroups against the crown's 18, and it saves a dispatch
   launch. That single cell is the one place the arithmetic here is not closed.
2. **Report the spill miscompile upstream or pin it.** The failure is
   reproducible from `research/e76_wide_gen.py` plus `research/e76_session.sh
   --mode parity`, and it is a silent wrong-answer bug, not a crash.
3. **Reuse the oracle on the head, not the target.** The census costs about nine
   seconds for 26 arms at five widths across two architectures. The MTP proposal
   head has never been through it.
4. **Check whether the g16s spill at NA = 6 is worth removing on its own.**
   `plain` carries a 16-byte frame there and `lazy` removes 4 registers for
   +5.71 %. A cheaper staging that only removes the frame was not searched.

## Reproduction

```bash
python3 research/e76_wide_gen.py --check          # arms are the shipped body
python3 research/e76_vec_layout.py                # vector padding, no GPU
python3 research/e76_rung1_census.py --out research/e76-artifacts/rung1.json
research/e76_session.sh --mode parity --na "5 6"
research/e76_session.sh --mode timed --na "5" --reps 21 \
  --arms plain,lazy,rps2,rps2lazyw,rps2lazy,rps1,rps1lazyw,rps1lazy
research/e76_session.sh --mode timed --na "6" --reps 21 \
  --arms plain,lazy,rps2,rps2lazyw,rps2lazy,rps1,rps1lazyw,rps1lazy
python3 research/e76_report.py --out research/e76-artifacts/rung1-table.md
```

Pre-checks, all passing on this branch:

```bash
senpai/verify-ranked-score-boundary.sh
senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf
senpai/validate-assignment-scope.sh 41ddc183817979be8d2f0817d79f98b2ddefb984 \
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
```
