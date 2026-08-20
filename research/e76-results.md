# E76: fit the scored cell into the ranked occupancy tier

Base `41ddc183817979be8d2f0817d79f98b2ddefb984`. Host Apple M4 Pro,
`applegpu_g16s`. Ranked host is `applegpu_g17s`. No candidate file changed: this
experiment did not touch `quantized.h`, its generated twin, the wrapper, or
either dispatch table. Every arm lives in a generated research header.

## Answer

**The register bar is reachable at both scored widths, and the register axis is
worth nothing. Every arm loses, and the closest one misses by a factor of 47.**

This report is written against the E77-corrected occupancy law from feedback
`e76-retract-the-occupancy-premise`. No GPU time was spent after that
retraction arrived, and the prediction it asked me not to pre-register is not
recorded here.

`rps1lazy` allocates **75** g17s registers at NA = 5 and **70** at NA = 6, which
is 23 and 41 registers below the shipped cells and 15 and 20 below the crown's
90. Bit-identity holds: zero differing elements on all seven scored shapes at
both widths, over 112 checked arm-width pairs.

None of it converts, and the corrected law makes that unambiguous. The largest
occupancy gain anywhere in the grid is **-0.631 %**, while the cheapest arm that
reduces registers at all costs **+5.03 %**. At the closest point, `rps2lazy` at
NA = 6, occupancy supplies -0.367 % against a measured +17.14 %.

The mechanism behind the cost is structural and is unaffected by the retraction:
**every arm that lowers the register count at NA = 5 or NA = 6 does so by
reducing `rows_per_simd`**, which doubles or quadruples x re-reads while the
weight and scale traffic stays fixed. Section 4 prices that from the shipped
source with no model at all.

Recommendation: **do not spend a ranked slot, and do not spend further GPU time
on the register axis.** The one-group partition is not structurally unavailable
at rank, which is what the brief expected the negative to look like. It is
available, it is bit-identical, and it is simply not worth what it costs.

**Deliverable B has a clean answer.** `lazyfall` removes the 16-byte
`applegpu_g16s` spill frame at NA = 6 while keeping the shipped
`rows_per_simd = 4`, and it lowers the ranked count from 111 to 99 rather than
raising it. It is the only arm that removes the frame without touching the row
block. It was never timed, and section 5b states that gap plainly.

A second result came out of the same instrument and it matters more than the
recommendation. **The register census is a cost instrument, not a correctness
gate.** One arm family compiles clean, with no spill and no diagnostic, and
returns wrong answers on every scored shape. Section 6 characterises it.

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

`research/e76-artifacts/rung1.json` and `rung1-table.md`. 29 arms x NA = 2..6 x
two architectures, in one JIT-string translation unit, from the `_wide` body
extracted verbatim from `quantized.h` lines 968-1066. Four levers cross into the
arm set: the row block (`rps4`, `rps2`, `rps1`), operand staging (eager,
`lazysb`, `lazyw`, `lazy`), accumulator layout (`facc`, `fall`) and
proposal-width chunking (`mc4`, `mc3`, `mc2`), plus the `rps2nu` and `rps1nu`
rolled-loop controls.

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

Arms at or below the 91 bar with no spill. Every arm listed here also returns
zero differing elements against `plain` on all seven priced shapes at its width,
so the answer to the assignment question is yes on registers **and** on device
output. `research/e76_qualify.py` prints this table with its parity and cost
columns joined.

| NA | arms |
|---:|---|
| 5 | `rps1lazy` **75**, `rps1lazyfacc` 75, `rps1lazyfall` 75, `rps1lazyw` 76, `rps1fall` 80, `rps1lazysb` 88, `rps1mc4` 88, `rps1mc2` 88 |
| 6 | `rps1lazyfall` **68**, `rps1lazy` 70, `rps1lazyw` 70, `rps1lazyfacc` 70, `rps2lazy` 86, `rps2lazyw` 88, `rps1fall` 90 |

The two widths differ in a way that decides section 5 before any timing is read.
**At NA = 5 every qualifying arm carries `rps1`.** No arm reaches the bar at the
shipped row block or at a halved one, so clearing 91 at NA = 5 requires
quartering the row block. **At NA = 6 the bar is reachable at `rps2`**, because
`rps2lazy` and `rps2lazyw` clear it while bare `rps2` at 100 does not.

The `rps1mc*` arms are diagnostics for section 6 and not candidates. They carry
the row block that section 4 prices at four times the x traffic, so they inherit
the cost that disqualifies `rps1lazy` and add the chunk overhead on top.

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

## 5. Rung 3, graded against the E77-corrected occupancy law

Feedback `e76-retract-the-occupancy-premise` retracts the 208 KiB register file,
`eps = 0.111`, `kappa = 0.0600`, the graded tier table and the -0.38 % prize. It
also instructs me not to pre-register the prediction that the previous feedback
asked for, so no such prediction appears in this report. The surviving law from
Alphonse's E77 is:

```text
S_local(R)  = floor(384 KiB / (128 * R))
S_ranked(R) = floor(496 KiB / (128 * R))
Omega(S)    = (32 / S) ** gamma,   gamma = 0.01346 +/- 0.00065
```

**None of my measurements change under the retraction.** The register census,
the parity results, the gated timing and the section 4 traffic accounting are
all direct observations. Only the modelled upside changes, and it collapses.

`research/e76_grade.py` reproduces every row of Alphonse's `S` table from these
two register-file sizes, on both architectures, which is the check that I am
applying his law and not a variant of it: local `R` 70, 93, 94, 95, 96 give `S`
43, 33, 32, 32, 32, and ranked `R` 83, 90, 91, 98, 111 give `S` 47, 44, 43, 40,
35.

### The register route is refuted by a factor of about 47

Occupancy gain is priced against the shipped cell at the same width. The cost
column is the same gated local measurement reported in section 3.

| NA | arm | g17s | ranked S | occupancy gain | measured cost | net | verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 5 | `rps1lazy` | 75 | 52 | **-0.353 %** | +64.71 % | +64.13 % | loses |
| 5 | `rps1lazyw` | 76 | 52 | -0.353 % | +62.31 % | +61.74 % | loses |
| 5 | `rps1` | 93 | 42 | -0.066 % | +47.56 % | +47.47 % | loses |
| 5 | `rps2lazyw` | 94 | 42 | -0.066 % | +28.59 % | +28.50 % | loses |
| 5 | `rps2` | 96 | 41 | -0.033 % | +14.16 % | +14.12 % | loses |
| 5 | `lazy` | 97 | 40 | +0.000 % | +15.73 % | +15.73 % | loses |
| 6 | `rps1lazy` | 70 | 56 | **-0.631 %** | +49.76 % | +48.81 % | loses |
| 6 | `rps1lazyw` | 70 | 56 | -0.631 % | +52.36 % | +51.40 % | loses |
| 6 | `rps2lazy` | 86 | 46 | -0.367 % | +17.14 % | +16.71 % | loses |
| 6 | `rps2lazyw` | 88 | 45 | -0.338 % | +17.87 % | +17.47 % | loses |
| 6 | `rps2` | 100 | 39 | -0.146 % | +5.18 % | +5.03 % | loses |
| 6 | `lazy` | 107 | 37 | -0.075 % | +5.71 % | +5.64 % | loses |

**Every arm loses, and none is close.** The largest occupancy gain anywhere in
the grid is -0.631 %. The smallest cost of any arm that reduces registers at all
is +5.03 %. At the closest point, `rps2lazy` at NA = 6, the corrected law
supplies -0.367 % against a measured +17.14 %, short by a factor of **47**.

Under the retracted model this same cell was an exact tie. That is the entire
effect of the retraction on this experiment: one tie becomes a clear loss, and
the recommendation stops depending on a modelled quantity at all.

The whole-grid occupancy span, from the worst spill-free arm at 122 ranked
registers to the best at 50, is **-1.209 %**, and -1.151 % to -1.267 % at one
standard error on `gamma`. That is the entire prize available on this axis
across a register range far wider than any shippable table spans, and it is
consistent in magnitude with Alphonse's 0.52 % for the shipped range.

### The recommendation

**Do not spend a ranked slot on a one-group cell at M = 5 or M = 6, and do not
spend further GPU time on the register axis.** The register question is now
closed as a fact for the record rather than a lever, which is how the retraction
asks for it.

The cheapest arm at each width is the honest headline. At NA = 6 the best net
outcome in the grid is `rps2` at +5.03 %; at NA = 5 it is `rps2` at +14.12 %.
Both are losses against the shipped cell.

M = 9 was not searched, as instructed.

## 5b. Deliverable B: the 16-byte local frame at NA = 6 can be removed

The retraction asks a separate, compile-only question: does any variant remove
the 16-byte `applegpu_g16s` spill frame that our `<T,6,6>` cell carries, without
raising the ranked register count? The frame is an instrument problem, because
the ranked host does not charge it, so it biases every local M = 6 measurement
the campaign makes.

**Yes. `lazyfall` removes it at the shipped row block.**

| arm | rows per simd | g16s regs / frame | g17s regs / spill | vs shipped g17s | parity |
|---|---:|---:|---:|---:|---|
| `plain` (= shipped) | 4 | 96 / **16** | 111 / 0 | — | reference |
| **`lazyfall`** | **4** | **93 / 0** | **99 / 0** | **-12** | clean, 7 shapes |
| `rps2` | 2 | 96 / 0 | 100 / 0 | -11 | clean, 7 shapes |
| `rps2lazy` | 2 | 72 / 0 | 86 / 0 | -25 | clean, 7 shapes |
| `rps1lazy` | 1 | 57 / 0 | 70 / 0 | -41 | clean, 7 shapes |

`lazyfall` is the one that matters, because **it is the only arm that removes
the frame while keeping `rows_per_simd = 4`.** Every other candidate reduces the
row block and therefore pays the x re-read that section 4 prices. `lazyfall`
keeps the shipped row block, so it does not pay it, and it lowers the ranked
count by 12 rather than raising it.

Neither lever does this alone. `fall` by itself makes NA = 6 worse: it spills 48
bytes on **both** architectures. `lazy` by itself keeps the 16-byte frame. Only
the combination clears it.

`lazyfall` at the shipped row block across all five widths, g16s regs / frame
then g17s regs / spill:

| NA | `plain` | `lazyfall` |
|---:|---|---|
| 2 | 70 / 0, 83 / 0 | 78 / 0, 82 / 0 |
| 3 | 93 / 0, 90 / 0 | 62 / 0, **70** / 0 |
| 4 | 94 / 0, 91 / 0 | 85 / 0, 93 / 0 |
| 5 | 95 / 0, 98 / 0 | 87 / 0, **93** / 0 |
| 6 | 96 / **16**, 111 / 0 | 93 / **0**, **99** / 0 |

It lowers the ranked count at NA = 3, 5 and 6 and costs 2 registers at NA = 4.

**One honest gap: `lazyfall` was never timed.** It was not in the rung-2 arm
set, and the retraction forbids opening a timed session, so I did not add one.
Its throughput cost is therefore unmeasured. I will not claim it is free: `lazy`
alone costs +5.71 % at NA = 6 and `lazyfall` carries that staging plus a flat
layout. What can be said from compile-time and parity evidence alone is that it
satisfies both stated requirements and is the only arm that does so without
touching the row block.

### Deliverable A, for the record

The flat-`acc` arm the retraction names is `facc`, and the answer at the two
scored widths is that **it moves neither number**. At NA = 5 and NA = 6 it
allocates exactly the shipped 98 and 111 g17s registers, and at NA = 6 it keeps
the 16-byte g16s frame. At NA = 5 it emits byte-identical g16s machine code to
`plain`, because SROA rebuilds the same scalars from the flat array. Its only
effect anywhere in the grid is to cost 2 registers at NA = 4. Section 1 records
the wider `fall` arm that does move the count, and why it is not profitable.


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

### The chunk arms also fail device parity, and the compiler reports success

The parity sweep covers all arms at NA = 3, 4, 5 and 6 on all seven scored
shapes. Every non-chunk arm returns zero differing elements. **Every multi-chunk
arm at the shipped row block returns wrong output.**

The failure signature is precise and identical everywhere. The fraction of
differing elements equals exactly the fraction of `m` values that are not in the
last non-empty chunk. Every chunk except the last returns wrong output and the
last chunk is exact, independent of chunk width and of width `NA`.

| arm | NA | partition | g17s regs / spill | fraction differing | "all but the last chunk" predicts |
|---|---:|---|---:|---:|---:|
| `mc2` | 3 | [2,1] | 116 / **0** | 66.7 % | 66.7 % |
| `mc2` | 4 | [2,2] | 122 / **0** | 50.0 % | 50.0 % |
| `mc3` | 4 | [3,1] | 126 / 144 | 75.0 % | 75.0 % |
| `mc4` | 5 | [4,1] | 126 / 240 | 80.0 % | 80.0 % |
| `mc3` | 5 | [3,2] | 126 / 176 | 60.0 % | 60.0 % |
| `mc2` | 5 | [2,2,1] | 126 / 240 | 80.0 % | 80.0 % |
| `mc4` | 6 | [4,2] | 126 / 288 | 66.7 % | 66.7 % |
| `mc3` | 6 | [3,3] | 126 / 224 | 50.0 % | 50.0 % |
| `mc2` | 6 | [2,2,2] | 126 / 288 | 66.7 % | 66.7 % |

Three hypotheses were tested and all three are refuted.

**It is not the rewrite.** `rps1mc4` carries the identical three substitutions on
the smallest row block. At NA = 5 it emits exactly the same `[4,1]` partition as
`mc4`. `mc4` fails on all seven shapes; `rps1mc4` returns **zero differing
elements on all seven shapes at NA = 3, 4, 5 and 6**.

**It is not spill.** `mc2` at NA = 3 and NA = 4 carries no spill frame on either
architecture and fails anyway, at exactly the predicted fraction. `mc4` at
NA = 3 and NA = 4 is a single chunk and passes trivially, and it emits
byte-identical machine code to the shipped instantiation.

**It is not register pressure, and it is not the number of chunks.** Crossing the
chunk lever with every row block separates the two candidate causes completely.
`research/e76_chunk_cross.py` prints the table; each cell is 7 priced shapes.

| arm | rows per simd | NA=3 | NA=4 | NA=5 | NA=6 |
|---|---:|---|---|---|---|
| `mc4` | 4 | pass (1 chunk) | pass (1 chunk) | **FAIL 80.00 %** | **FAIL 66.67 %** |
| `mc3` | 4 | pass (1 chunk) | **FAIL 75.00 %** | **FAIL 60.00 %** | **FAIL 50.00 %** |
| `mc2` | 4 | **FAIL 66.67 %** | **FAIL 50.00 %** | **FAIL 80.00 %** | **FAIL 66.67 %** |
| `rps2mc4` | 2 | pass | pass | pass | pass |
| `rps2mc2` | 2 | pass | pass | pass | pass |
| `rps1mc4` | 1 | pass | pass | pass | pass |
| `rps1mc2` | 1 | pass | pass | pass | pass |

112 arm-width pairs were checked over 7 shapes. Exactly 9 pairs differ, and all
9 are `rows_per_simd = 4` with more than one chunk. Multiple chunks alone are
safe: `rps2mc2` and `rps1mc2` at NA = 6 both run the three-chunk `[2,2,2]`
partition and both return zero differing elements.

The cleanest single comparison holds NA, the partition and the substitutions
fixed and varies only the row block. At NA = 3 with the `[2,1]` partition:

| arm | rows per simd | g17s regs / spill | parity |
|---|---:|---:|---|
| `mc2` | 4 | 116 / **0** | **FAIL 66.67 %** |
| `rps2mc2` | 2 | 92 / 0 | pass |
| `rps1mc2` | 1 | 65 / 0 | pass |

The failing arm carries the **most** registers of the three and still has no
spill, so higher pressure is not the trigger and the two lower-pressure arms are
correct.

**The discriminator is `rows_per_simd = 4` — the shipped value — combined with
more than one chunk.** `mc4` and `rps1mc4` at NA = 5 differ in exactly one
`constexpr`. Every row loop in the rewritten body is
`for (int r = 0; r < rows_per_simd; r++)`, so the source has no structural
dependence on that constant. The compiler accepts both, reports registers and
spill for both, and one of them silently returns wrong answers for 80 % of its
output rows.

This is the caveat the campaign should carry, because the compile oracle is now
load-bearing for kernel design. E72 established that AIR is not the backend.
This is the next step down: **the backend can report a clean allocation, with no
spill and no diagnostic, for a kernel that does not compute the right thing. A
register census is a cost instrument, not a correctness gate.** Device parity on
the scored shapes remains the only gate that holds.

Toolchain for the record: `Apple metal version 32023.883 (metalfe-32023.883)`,
target `air64-apple-darwin25.5.0`, macOS 26.5.2.

## 7. What could not be calibrated

`maxTotalThreadsPerThreadgroup` is the one public Metal field a register
allocation could move. It holds at 1024 for every kernel built here, from 14
registers up to 126 registers carrying an 816-byte spill frame. The runtime
spills instead of shrinking the threadgroup, so the field is not register-derived
on this host and gives no register-file estimate.

The step-1 live-float sweep shows no further allocator steps. The reported count
takes both odd and even values across 74 to 126 with no plateau, so the allocator
does not quantize the count. Both backends allocate identically up to 64 live
floats on the scalar probe and diverge only where g16s runs out.

**This section is now partly superseded, and the two results agree.** Alphonse's
E77 measured the register file directly at 384 KiB locally and 496 KiB at rank,
which is what I could not obtain here, and it also found that occupancy rises
smoothly rather than in tiers. My negative result — no plateau anywhere in a
step-1 sweep from 74 to 126 registers — is the same observation from the
compile side, and it is consistent with his finding that the staircase is the
wrong functional form. Section 5 uses his measured constants rather than my
inability to obtain them.

One correction for the record, requested in the retraction and not made in any
scored diff: the `<T,8,4,true>` line at `upstream/main` carries a stale
`3+3+2` comment. The actual partition is `[4,4]`.

## 8. Suggested follow-ups, not implemented

1. **Time `lazyfall` at NA = 6 and adopt it as an instrument fix if it is
   cheap.** This is the one open item from section 5b and it is a measurement I
   was told not to start. `lazyfall` removes the 16-byte local frame at the
   shipped row block, so it is the only frame-free arm whose cost is not
   dominated by x re-reads, and its cost is unmeasured. If it is near free, every
   future local M = 6 measurement gets a control that is not handicapped relative
   to the ranked host, which matters because M = 6 carries 33.4 % of the ranked
   width pool. Report it as an instrument fix, not a score lever: section 5 shows
   its 12-register reduction is worth about -0.1 % on the corrected occupancy
   law, so it cannot be a lever.
2. **Reduce the chunk miscompile to a minimal case and pin or report it.** The
   failure is reproducible from `research/e76_wide_gen.py` plus
   `research/e76_session.sh --mode parity`, and it is a silent wrong-answer bug,
   not a crash. The reduction is not needed for any campaign decision, so I did
   not spend the time, but the toolchain version should be recorded against it
   before the next toolchain change.
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
python3 research/e76_rung1_census.py --na 2 3 4 5 6 \
  --out research/e76-artifacts/rung1.json
research/e76_session.sh --mode parity --na "3 4 5 6"
research/e76_session.sh --mode timed --na "5" --reps 21 \
  --arms plain,lazy,rps2,rps2lazyw,rps2lazy,rps1,rps1lazyw,rps1lazy
research/e76_session.sh --mode timed --na "6" --reps 21 \
  --arms plain,lazy,rps2,rps2lazyw,rps2lazy,rps1,rps1lazyw,rps1lazy
python3 research/e76_report.py --out research/e76-artifacts/rung1-table.md
python3 research/e76_qualify.py                   # arms clearing the 91 bar
python3 research/e76_chunk_cross.py               # the miscompile cross
python3 research/e76_grade.py                     # E77-corrected occupancy law
```

Two traps in this tooling. `e76_rung1_census.py` defaults to `--na 5 6`, so the
low widths that anchor section 1 are dropped unless they are named. `--mode
parity` ignores `--arms` and always sweeps every arm, so a tagged partial run
must be deleted rather than kept beside a full one; `e76_report.parity()` globs
`parity-na*.json` and would otherwise count both.

Pre-checks, all passing on this branch:

```bash
senpai/verify-ranked-score-boundary.sh
senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf
senpai/validate-assignment-scope.sh 41ddc183817979be8d2f0817d79f98b2ddefb984 \
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
```
