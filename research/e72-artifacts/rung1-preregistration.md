# E72 rung 1 — pre-registered ranked sign predictions for `t55`, `t6` and `E55`

Written and committed before reading the ranked receipt for the in-flight
submission `9b241879`. The advisor holds that receipt and has not disclosed it.

- Assignment: `qwen38-r1-e72-crossarch-register-census`, PR 75, revision `r1`
- Base: `senpai/qwen38-mtp-r1` at `1fa4352f097b79edbe508847ffce1b2aab1edfa1`
- Instrument: `research/agx_crossarch.py`, census `research/e72_rung1_census.py`
- Evidence: `research/e72-artifacts/rung1.json`, `rung1-jit.json`, `rung1-wall.json`
- Harness label: `harness=arch-probe`. Nothing here is a timing measurement and
  nothing here is converted into a ranked score.

## The quantity the predictions are derived from

`xcrun metal-tt -arch applegpu_g17s` runs the real AGX generation-17 backend on
this generation-16 host, so the ranked runner's register allocation can be read
here. Two fields of `__GPU_METADATA` are calibrated on every run by
`agx_crossarch.py selftest`: a register count and a spill frame size that is
exactly `4 * N + 16` bytes when a working set of `N` floats spills.

**The two generations do not have the same per-thread register budget.** A
sweep of kernels holding `n` live floats in named scalars, so that no array
promotion heuristic can be mistaken for a spill:

| generation | highest register count with no frame | first frame appears at |
|---|---:|---:|
| `applegpu_g16s`, this host | **96** | 72 live floats |
| `applegpu_g17s`, ranked host | **124** | 96 live floats |

The local budget saturates hard at 96. The ranked budget is about 30 percent
larger.

Register count and spill bytes for the six cells that decide the three
experiments, from the shipped
`qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true>`:

| cell | ships in | g16s regs | g16s spill | g17s regs | g17s spill |
|---|---|---:|---:|---:|---:|
| `<T,5,5>` | ours, `t55` | 95 | 0 | 98 | 0 |
| `<T,5,3>` | frontier | 93 | 0 | 90 | 0 |
| `<T,6,6>` | ours, `t6` | **96** | **16** | **111** | **0** |
| `<T,6,3>` | frontier | 93 | 0 | 90 | 0 |
| `<T,9,5>` | ours, `E55` | 95 | 0 | 98 | 0 |
| `<T,9,3>` | frontier | 93 | 0 | 90 | 0 |

The headers translation unit and the worker's JIT string produce byte-identical
machine code for all 24 kernels on both generations, so these numbers describe
the scored path and not a research probe.

## What this says about H208

H208 proposes that the local group-partition optimum is a ranked pessimum
because our `t6` put the largest ranked width share onto the one variant that
spills.

**The spill is on this host, not on the ranked host.** `<T,6,6>` sits exactly on
the local 96-register wall and carries a 16-byte frame here; on `applegpu_g17s`
it allocates 111 registers with no frame at all, 13 registers below that
generation's measured 124-register ceiling. Every `wide_na3` through `wide_na6`
variant reports exactly 93 to 96 registers on g16s, pinned at the local wall,
while the same variants on g17s rise freely from 90 to 111.

So the stated mechanism of H208 is absent on the ranked target. If the three
partitions are ranked losses, the cause is not NA=6 register spilling.

## Predictions

The direction of the register evidence is that this host is the **harsher** of
the two for wide groups: it is the only one of the pair that clamps and frames
the `<T,6,6>` cell. A variant that won here under a register handicap that the
ranked host does not impose should not invert on the ranked host.

| experiment | predicted ranked sign | quantity predicted from |
|---|---|---|
| `t6`, M=6 one group of 6 | **same sign as local: a win** | `<T,6,6>` allocates 111 of 124 available registers on g17s with 0 spill bytes, against 96 of 96 with a 16-byte frame on g16s. The local measurement carries a penalty the ranked host does not apply. |
| `t55`, M=5 one group of 5 | **same sign as local: a win** | `<T,5,5>` allocates 98 registers with 0 spill bytes on g17s and 95 with 0 on g16s. Neither generation frames, and both are below their own budget. There is no register mechanism that can invert this cell. |
| `E55`, M=9 as 5 plus 4 | **same sign as local: a win** | `<T,9,5>` has the same register signature as `<T,5,5>`: 98 with 0 spill on g17s, 95 with 0 on g16s. Same reasoning. |

I am predicting that reverting the three partitions to the frontier's bytes does
**not** recover the 1.86 percent we lost, and that the regression in `ff73cbbd`
comes from something other than these three partitions.

## Falsification condition

The in-flight submission `9b241879` reverts all three partitions to the
frontier's exact bytes. Our last submission `ff73cbbd` scored 3.17230 with the
three partitions present.

**These predictions are falsified if `9b241879` scores materially above
3.17230.** That result would mean the bundle `t55 + t6 + E55` is a net ranked
loss, which is the opposite of all three predictions above, and it would mean
the cross-architecture register and spill census has no predictive power for
ranked outcomes.

A single ranked pair cannot resolve a difference smaller than run-to-run ranked
noise, so I claim falsification only for a gap larger than the campaign's
observed ranked repeatability, and I treat any smaller gap as undecided.

## Known residual risk, stated in advance

This census reads register count and spill bytes. **It does not read
occupancy**, and no public field in `__GPU_METADATA` reports it. On g17s the
one-group cells demand 98 and 111 registers against 90 for the frontier's
three-row cells. That is a real 9 to 23 percent rise in per-thread register
demand which does not spill but which may still reduce the number of resident
simdgroups per core. The local host cannot exhibit that effect, because its
allocator clamps every one of these cells into the narrow 93 to 96 band.

If the predictions fail, this is the most likely reason, and it would mean the
instrument needs an occupancy term before it can price a partition decision.
That would be a limitation of this census rather than a refutation of the
method.
