# E223: the per-column QMV inner loop

`harness=local` for the static census, `harness=local-microbench` for the
timing. Not gate-qualified. No official or ranked score. No whole-leg number.

## Question

FINDING 573 puts an NA-relief ceiling of 23.066 ms/round on the per-column cost
slope of `qwen_e120_qmv_wide`. Is that slope dominated by per-column register
lane inserts, per-column address arithmetic and separate 8-byte activation
loads? If so, an NA-contiguous activation access path should recover at least
+1.0 ms/round pooled.

## Answer

**No. The hypothesis is refuted, and the refutation generalises.**

The addressable part of the per-column cost is none of the three named
mechanisms. The backend removes or normalises all three. What remains is 16
`bfloat16` to `float` conversions per column per k-block, and removing those
buys no time, because the gather is not instruction-issue-limited.

## Identity tuple

| field | value |
|---|---|
| assignment | PR 221, `e223-percolumn-inner-loop`, revision `e223-r0` |
| base | `senpai/qwen38-mtp-r1` at `a384559870722a20e7c48354afc72e623a9afefb` |
| scored kernel | `qwen_e120_qmv_wide`, `qwen35E120QMVHeader`, `Qwen35.swift:1426-1559` |
| host | Apple M4 Pro |
| local arch | `applegpu_g16s` (96-register cap) |
| ranked arch | `applegpu_g17s` |
| cells | `gdn.in_proj` (k 5120, n 16480, inv 48), `mlp.gate_up` (k 5120, n 34816, inv 64), `mlp.down` (k 17408, n 5120, inv 64) |
| token window | none: standalone kernel microbenchmark, no model loaded |
| reference source | self-generated random affine-4/group-64 cells at scored shapes |
| thermal | ungated by design, ABBA-counterbalanced within one session, entry 33.7C, exit 45.6C |

## Stage 0: where the per-column cost actually is

Zero GPU seconds. AIR census plus a real AGX witness on both arches, calibrated
at 9.2565 bytes per instruction (R2 0.996 on both arches).

The kernel has two unroll regimes. At NA <= 5 it is fully unrolled and costs
**123.2 machine instructions per column per k-block** on `g17s` (1141 B/NA,
R2 0.99999). At NA >= 6 it rolls, code size drops and spilling starts.

Attribution of those 123.2 instructions:

| component | instructions | share | in scope |
|---|---|---|---|
| floating-point mul/add/FMA | 92 | 74% | no, value-fixed |
| `bfloat16` to `float` converts | 16 | 13% | yes |
| device loads | 5 | 4% | yes |
| address and integer residual | 11 | 9% | yes |

The whole addressable access path is 32 of 123.2 instructions, so **26% is the
hard ceiling** for any source-level change that preserves the arithmetic.

AIR's 93 lane-ops of shuffle, control, integer and address work collapse to
about 11 machine instructions, so AIR op counts overstate the addressable
share by roughly 8x. Price from the AGX witness, not from AIR.

### Four mechanisms refuted at the desk

1. **Lane inserts cost about zero.** The backend fully unrolls the `m` loop, so
   `a0[m] = ...` becomes register naming, not an insert.
2. **The load-count lever does not exist at source level.** `narrowload`, a
   method control that reads the same four values as four separate 2-byte
   scalar loads instead of one 8-byte vector load, compiles to
   **bit-identical machine code** (`text_sha8 73628626` at `g17s` NA=6). The
   backend re-coalesces narrow source loads.
3. **`wideload` is dead.** At NA=2, the only width where it stays in the same
   unroll regime, it moves `text_bytes` by -2 B on `g16s` and -8 B on `g17s`.
   At NA >= 3 it changes the unroll regime and worsens spill at the scored
   NA=6.
4. **`hoistbase` is worse, not better.** Hoisting the per-column base pointer
   turns `xbase[NA]` into memory traffic: AIR `address_arith` goes 8 to 16 per
   NA and `device_load` 4 to 8 per NA.

NA-contiguous layouts are structurally dead for a separate reason: the
per-`(k, i, j)` stride is `NA * 4` bytes, a power of two only at NA in
{2, 4, 8}, and at NA=4 the transposed form issues the same 16 loads while
doubling issued bytes. The `xtf32` probe measured 17 scalar loads per NA.

### The one survivor

`xf32flat` / `xf32ship`: activations already `float32` in the shipped `[m, k]`
layout. It removes the 16 converts and adds 2 loads, for
**110.6 instructions per column per k-block** on `g17s`, or -10.2%. Widening
`bfloat16` to `float` is exact and no other line moves, so it is bit-identical
by construction.

## Stage 1: the survivor measured

### Bit-exactness gate

`activationDTypeExactness`, 12 real rows across all three cells at NA 2..5:
`differing == 0` and `non_finite == 0` in every row. 12 positive-control rows
perturb the float32 slab inside the first k-block of the first column only and
move the output in every row, minimum 5120 elements. The gate can fail, and it
does not.

### Timing

48 matched `(cell, NA, G, thermal)` points, 6 blocks with alternating order,
4 chain lengths, calibrated repetition counts. Pooled over `gdn.in_proj` and
`mlp.gate_up`; `mlp.down` is reported per cell and never pooled.

Positive = the float32 arm is faster. Advance bar +1.0, stop bar +0.3.

| arm | measured relief | +/- CI95 | stage-0 desk net |
|---|---|---|---|
| na2/g1/hot | **+0.300** | 0.446 | — |
| na2/g2/cold | +0.031 | 1.424 | — |
| na2/g1/cold | -1.633 | 1.075 | — |
| na2/g2/hot | -2.464 | 0.398 | — |
| na3/g1/hot | -1.727 | 0.427 | — |
| na3/g2/hot | -58.211 | 0.588 | — |
| na4/g1/hot | -6.714 | 0.645 | +0.225 |
| na4/g2/cold | -130.314 | 2.466 | +1.861 |
| na5/g1/cold | -105.050 | 1.564 | -3.299 |
| na5/g2/hot | **-353.040** | 1.690 | +2.300 |

The best arm of the whole grid is +0.300 +/- 0.446 ms/round, which does not
clear the +0.3 stop bar with separation, and its cold twin has the opposite
sign. Nothing reaches +1.0.

## Why the desk was wrong by up to 355 ms/round

The stage-0 desk priced the extra activation bytes with `us_per_issued_mb` from
E221's geometry route, 0.042 to 0.186 us/MB. Under RULE 410 that is a
**cache-served/concurrent** rate, and the tag is correct: of the 850.0 MB of
extra bytes at the worst arm, 0.098 MB are first touch and 849.9 MB, or 99.99%,
are re-reads inside one dispatch.

Tagging is necessary but not sufficient:

| term | value |
|---|---|
| first-touch cost at full `b` = 1.9292 us/MiB | 0.012 ms/round |
| cache-served cost at the measured `0.36*b` bound | 37.8 ms/round |
| **measured loss** | **263.3 ms/round** |

Even the conservative end of the E222 bound under-prices this arm 7x.

**A cache-served coefficient is not transferable across a change in element
size.** A cache-served rate is defined by a residency that the treatment moves.
`bfloat16` to `float` does not merely double byte volume; it doubles the
working set and changes register allocation, so it relocates the cliff whose
position sets the rate. Fitting `b'` more precisely does not fix this class of
arm.

## Mechanism: a local register cliff

RULE 407 witness on the exact timed instantiation. The `xf32ship` variant is
the exact source form `e223ActivationF32Header()` builds, so these are the
kernel that ran. Table path, registers/spill bytes:

| NA | g16s bf16 | g16s f32 | g17s bf16 | g17s f32 |
|---|---|---|---|---|
| 2 | 74/0 | 89/0 | 87/0 | 95/0 |
| 3 | 84/0 | **96/16** | 90/0 | 99/0 |
| 4 | 93/0 | 94/0 | 94/0 | 98/0 |
| 5 | 96/0 | **96/176** | 102/0 | 107/0 |
| 6 | 96/16 | 96/64 | 114/0 | 126/0 |

`g16s` caps at 96 registers. The float32 gather crosses that cap and spills at
NA=3 and NA=5, where the shipped arm spills nothing until NA=6. This predicts
the measured cliff in three independent directions:

- **NA.** NA=2 and NA=4 spill nothing and cost 1.02x to 1.15x, even though
  NA=4 issues twice the extra bytes of NA=2. Byte volume does not drive the
  cliff. NA=3 spills 16 B and NA=5 spills 176 B, giving 1.7x and 3.0x at G=1.
- **G.** G changes no compiled code, so no compile-level term can explain it,
  but spill traffic scales with groups. Every G=2 arm is worse than its G=1
  twin.
- **Cell.** Spill traffic scales with k-blocks, so `mlp.down` at k=17408
  cliffs three widths earlier: 2.20x already at NA=3, G=1.

The effective rate for the extra bytes collapses from about 11.2 TB/s
(NA=2, no spill) to 206 GB/s (NA=5, G=2, spilling), a 54x collapse into DRAM
class.

## The finding that outlives the arm

**The activation gather is not instruction-issue-limited.**

At NA=2, G=1 there is no spill on either arch and the byte cost is smallest.
The arm removes 10.2% of the machine instructions per column per k-block, and
it is 3 to 5% **slower** cold and indistinguishable from zero hot.

Stage 0 offered two readings of the NA slope: op-proportional relief as an
upper bound, and an issue-limited reading predicting exactly zero. **The data
selects zero.** This retires the whole "remove non-arithmetic work from the
inner loop" family, not just this arm. The remaining 74% of the per-column cost
is floating-point work fixed by the value contract.

This also explains the factor 1.65 discrepancy stage 0 could not account for
between E221's average op rate of about 3.6 T ops/s and the marginal rate of
about 5.95 T ops/s: the marginal instruction is not what the kernel is waiting
on.

### The one correction to FINDING 580

FINDING 580 attributed the NA cost slope to per-column register lane inserts,
per-column address arithmetic, and NA separate 8-byte loads. **All three
attributions are wrong.** Stage 0 refutes each one at the desk, and stage 1
refutes the family they belong to by measurement:

- Lane inserts are approximately zero. The backend fully unrolls the `m` loop
  at NA <= 5 and turns every insert into register naming.
- Address arithmetic is 11 of 123.2 machine instructions per column per
  k-block, 9%.
- Device loads are 5 of 123.2, 4%. A `narrowload` method control compiles
  **bit-identical** (`text_sha8 73628626`, `g17s`, NA=6), so there is no
  source-level load-count lever at all.

The slope is real, but 92 of its 123.2 instructions (74%) are floating-point
mul/add/FMA fixed by the value contract. The NA slope is carried by
floating-point work, not by lane inserts, loads, or address arithmetic.

**Important nuance.** This does not collapse FINDING 573's 23.066 ms/round
NA-relief ceiling. The measured NA slope is real cost. What collapses is the
*reachable* part of it through non-arithmetic instruction removal. The ceiling
relocates entirely onto the 74% floating-point share, where the value contract
forbids removal. A future arm must reduce columns, widths, or invocations, not
the instructions inside a column.

### The fitted coefficient, and why it is only a bound

The advisor asked for the fitted marginal worth of one removed non-arithmetic
instruction, so E226 can hold it out as a self-test target. **E223 cannot
identify it as a free parameter.** Removed instructions and extra issued bytes
are exactly collinear across all 48 arms at a single ratio:

```text
0.39375 removed instructions per extra issued byte   (all 48 arms)
```

Removing a bfloat16-to-float convert requires widening the load that feeds it.
Both terms scale as `NA * lane_k_blocks * groups`, so E223 identifies only
their **sum**, which is <= 0 at every arm. The tightest bound comes from the
minimum-byte arms, `NA=2, G=1`, which remove 1.2174e10 instructions per round
pooled:

| arm | measured ms/round | +/- CI95 | upper bound | op-proportional prediction | excluded |
|---|---|---|---|---|---|
| na2/g1/cold | -1.633 | 1.075 | -0.558 | +1.115 | yes |
| na2/g1/hot | +0.300 | 0.446 | +0.747 | +1.290 | yes |

The op-proportional reading is **excluded at both thermal arms**. The upper
bound on the coefficient itself is:

```text
<= 6.13e-08 us per removed instruction   (hot)
 = 0.0613 us per 1e9 removed instructions
```

Cold is negative. For a self-test, E226 should treat any model that prices
non-arithmetic instruction removal above about 0.06 us per 1e9 instructions in
this cell family as falsified by E223.

Five of the six individual `NA=2, G=1` cell measurements overlap zero
(gdn hot -0.292 +/- 5.507, gdn cold -7.032 +/- 7.592, mlp.down hot
-0.282 +/- 10.589, mlp.down cold +2.272 +/- 11.452, gate_up hot
+4.909 +/- 5.622, gate_up cold -20.243 +/- 15.805, all us/invocation).

An independent fact supports the same reading: at NA=2 all three cells run the
weight stream at 183 to 217 GB/s, which is DRAM-class on an M4 Pro. The kernel
is weight-stream bandwidth-bound there, which is why neither removing
instructions nor adding cache-served bytes moves the time.

## Transfer

Local-to-ranked transfer is **blocked** for this arm. On `g17s` the same source
spills nothing at NA 2..6, so the dominant local cost term is absent on the
ranked part, and this host cannot price the arm for M5 in either direction.

That is not a reason to chase it on M5. The doubled activation bytes and the
+8 to +12 register delta remain real on `g17s`, the shipped version would also
need a real f32 buffer, a new kernel argument and a fill pass at 3.8 us or more
per fill over a slab 10x the `xsums` table, and the issue-limited finding kills
the upside independently of any of that.

## Decision

Killed at stage 1. Stage 2, the scored path, is not reached. Under RULE 410 the
decision uses no `b_stream` term, so no `b'` in `[0, b]` can move it:
`decision_uses_b_stream: false`,
`decision_flips_anywhere_in_b_prime: false`.

## Reproduction

```bash
# stage 0, zero GPU seconds
python3 research/e223_census.py --variant base
python3 research/e223_census.py --variant xf32ship
python3 research/e223_price.py

# stage 1 gate, then timing
research/e219_session.sh e223-sanity sanity
research/e219_session.sh e223-xdtype xdtype
python3 research/e223_screen.py research/out/e219/e223-xdtype/xdtype.json \
  --out research/e223-artifacts/e223-screen.json
python3 research/e223_wandb.py
```

## Suggested follow-ups, not implemented

1. **E226 scope.** Fit the cache-served coefficient within a fixed element
   size, and treat any element-size change as needing its own measurement. A
   single `b'` cannot cover both.
2. **A spill-onset guard.** The cheapest reusable product of this experiment
   would be a RULE 407 probe that fails an arm at the desk whenever it moves
   spill onset below the scored NA on either arch. That would have killed this
   arm before any GPU second, and it would have killed E222's arm too.
3. **Attack the 74%.** The only remaining per-column headroom is the
   floating-point work itself. That needs a change to the arithmetic contract,
   not to the access path, so it is a different and much more constrained
   question.
4. **Register headroom at NA=6.** On `g17s` the shipped arm reaches 114
   registers at NA=6 and 126 at NA=7, and the scored `singlePass` m=6 arm
   already runs the rolled form. Whether the rolled regime at the scored width
   is occupancy-limited is a separate, live question this experiment did not
   test.
