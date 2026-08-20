# E61 preregistration — single weight stream at M=6

Written before any E61 measurement. Every number below is a prediction, not a
result. `git log` fixes the order: this file is committed before the first
census, probe or timed leg.

Base `d2139c924c7a7d98ca6026eea63867c2776abbca`, host Apple M4 Pro, 48 GiB,
macOS 26.5.2, Swift 6.3.3. Identity tuple for every timed arm: this host, this
toolchain, the declared proposal head, the public longcopy fixture, 512 decode
tokens, offered depth 8.

## Rung 0.1 — `vec<float,N>` for N = 5, 6, 7, 8

E54 already answered the layout half of this question at compile time
(`research/e54-artifacts/vec5-proof.json`), so the prediction is a
reproduction claim rather than a guess:

| N | sizeof | alignof | AIR alloca type | padding bytes |
| --- | --- | --- | --- | --- |
| 5 | 32 | 32 | `[4 x <5 x float>]` | 12 |
| 6 | 32 | 32 | `[4 x <6 x float>]` | 8 |
| 7 | 32 | 32 | `[4 x <7 x float>]` | 4 |
| 8 | 32 | 32 | `[4 x <8 x float>]` | 0 |

So the advisor's conditional fires: `sizeof(vec<float,5>) == 32`, not 20. The
type does pad to eight lanes **in memory**.

It does not follow that NA = 5, 6, 7, 8 occupy the same registers. The AIR
alloca keeps N lanes, `acc` never escapes the thread, the compiler scalarises
it, and the padding lanes are dead. E54 measured `+21` registers across
NA 4 -> 5, which is one accumulator row and not the `+84` an eight-lane padded
step would cost, and it retracted Law D on exactly that evidence. I therefore
predict the register law survives the padding fact, and I predict the same
`+21` per row at 5 -> 6.

Device predictions for the GPU probe, for every N in 2..8:

- `sizeof` and `alignof` equal to the table above;
- indexed-write lane-bleed mask `0`;
- lane-local arithmetic fault mask `0`;
- every lane m of an NA = N run bitwise equal to the same input row computed by
  the shipped NA = 4 or NA = 2 helper, `max_ulp = 0`;
- all N lane outputs distinct;
- all three deliberate lane faults caught at every N.

## Rung 0.2 — register census

Instrument: `research/e61_reg_census.py`, the E46/E55 compile machinery.
Prediction first, from the advisor's census law

```
reg = 20 + 21*max(NA) + 4*[two distinct NA group sizes]
```

One honest note about that law, recorded before the measurement. The `+4` term
is exactly the constant over-count E54 documented in this instrument on mixed
cells (`<T,5,3>` 87 against 83, `<T,7,4>` 108 against 104, `<T,9,5>` 129
against 125). So the law fits *the instrument*, and the hardware ceiling of a
mixed cell is likely 4 lower than the instrument reports. Both readings are
predicted below, because the campaign's historical numbers are all
instrument-reported and must stay comparable.

| arm | cell under test | predicted cell regs (instrument) | predicted table max (instrument) | predicted hardware max (law, no over-count) |
| --- | --- | --- | --- | --- |
| `shipped` at `d2139c92` | `<T,9,5>` {5,4} | 129 | **129** | 125 |
| `t6` whole table, `case 6` -> `<T,6,6>` | `<T,6,6>` {6} | **146** | **146** | 146 |
| isolated M=6 at IPG=3 | `<T,6,3>` {3,3} | 83 | 83 | 83 |
| isolated M=6 at IPG=6 | `<T,6,6>` {6} | 146 | 146 | 146 |
| isolated `<T,7,7>` | `<T,7,7>` {7} | 167 | 167 | 167 |
| isolated `<T,8,8>` | `<T,8,8>` {8} | 188 | 188 | 188 |

Whole-table entry-point registers, the second reading the census emits. The
recorded entry-to-cell offsets are +55 (108/163), +54 (129/183), +57 (125/182)
and +52 (129/181), so I predict:

- `shipped` at `d2139c92`: `entry_batch0` = **181**, accept 178..184;
- `t6`: `entry_batch0` = **200**, accept 196..205.

Refutation rule: any cell that misses its predicted value by more than 0 is a
miss against the law, and I report it as prominently as a timing number. The
entry predictions are an interval, because the offset is not a law.

Every census arm is marked `family: "census_probe"`, `never_time: True`, except
`shipped` and `t6`, which are the two arms rung 3 would time.

## Rung 1 — extend the E54 lone-group bandwidth ladder

Measured anchors, `research/e54-artifacts/e54-bandwidth.json`:

```
NA=2  223.784   NA=3  199.693   NA=4  175.238   NA=5  150.946   GB/s
steps: -24.091, -24.455, -24.292
```

Linear predictions, extrapolated one and two steps past the last anchor:

- **bw(6) = 126.65 GB/s**
- **bw(7) = 102.35 GB/s**

Break-even thresholds carried verbatim from the assignment: M=6 needs
**114.00**, M=7 needs **106.55**, M=8 needs **100.01**, M=9 needs **92.56**.

Refutation tolerance for the linear model, set before the measurement:
`|measured - predicted| > 5.0 GB/s` refutes linearity at that NA. 5.0 GB/s is
about 4 % of the predicted value and about 20 % of one ladder step, and the
three measured steps agree to 0.36 GB/s, so a 5 GB/s miss cannot be step noise.

Positive control, required in **every** E61 curve leg: the untreated widths
M = 2, 3, 4 re-measure the NA = 2, 3, 4 anchors inside the same leg. Accept if
each is within 2.5 % of its historical value (E54's own cross-arm spread was
<= 1.0 %). If any control misses, the ladder extension is void and I report
that instead of a verdict.

### Direct cell measurement, taken in the same legs

The same legs measure the M=6 cell time directly, which is the quantity the
bandwidth model exists to predict. Predictions:

- `t6` minus `shipped` at M=6: **-9.95 %** (the assignment's model value);
- `t7` minus `shipped` at M=7: **+4.16 %**;
- untreated widths in `t6` versus `shipped`: **0.00 %**, accept +-1.0 %. Any
  systematic negative or positive shift across all seven untreated widths is
  the shared-register-ceiling term at 146, measured directly, and I report it
  as the rung-4 decomposition without needing a ballast arm.

### Decision gate

Preregistered, and deliberately one clause wider than the assignment's:

- **Proceed** to rung 2 and rung 3 if `bw(6) > 114.00 GB/s`.
- **Also proceed** if `bw(6) <= 114.00` but the *directly measured* M=6 cell
  delta is `<= -2.0 %` with the untreated widths inside +-1.0 %. A measured
  cell time is a better estimate of break-even than a model of break-even, and
  refusing a measured win because a model of it failed would be the wrong
  error. I will state which clause fired.
- **Stop** if both clauses fail. Report the ladder extension as the result.
  That closes M=6, M=7, M=8 and M=9 in one measurement, given a monotone
  decreasing ladder.

## Rung 3 — whole-leg prediction

Local width histogram, deterministic across all six E55 legs at 512 tokens:

```
M:      2   4   5   6   7   8   9
rounds: 1   5   5  23   4   6  34      (78 rounds)
```

Time-weighted local QMV shares on the `d2139c92` table, using E54's shipped
per-width verify costs with M=9 taken from the `<T,9,5>` arm:

```
f6 = 26.71 %      f9 = 50.47 %
```

The advisor expected M=6 to be a smaller share locally than at rank. It is
smaller, but only slightly: 26.71 % local against 30.9-34.7 % ranked. The
local instrument is therefore *not* badly mismatched for this cell, unlike
E55.

Whole-leg prediction, using E55's pre-registered transfer constant
`psi_mtp = 0.693391` and the model cell delta:

```
predicted MTP-leg delta = 0.693391 * 0.2671 * (-9.95 %) = -1.84 %
after E55's measured realisation factor 0.946           = -1.74 %
```

Registered before rung 3, from the assignment:

- **Promote** at `<= -0.30 %` whole-leg candidate MTP seconds per token, with
  all three falsifiers inside their nulls and rung 2 clean.
- **Report only** in `(-0.30 %, +0.10 %)`.
- **Stop** at `>= +0.10 %`, at any exactness failure, or at any unclosed row
  ledger.

Falsifiers, unchanged from E55, all predicted to be null:

1. serial-leg seconds per token, predicted delta `0.00 %`, null floor
   `-0.0133 %`;
2. serial round cost, predicted null;
3. seed prefill seconds, predicted null.

Local null floor `0.0629 %`. The predicted effect is 27x the null floor.

## Method commitments

- All three geometry-lever variables exported on every timed leg:
  `DARKBLOOM_STARTUP_MEMORY_PROFILE=full`, `MLX_MAX_MB_PER_BUFFER=512`,
  `MLX_MAX_OPS_PER_BUFFER=50`. Every leg is failed if worker stderr shows
  `mlxfast-worker: low-memory startup profile engaged`.
- Arm identity is asserted from the built worker binary's content, never from
  mtime or provenance (`research/e61_binary_assert.sh`).
- Candidate files are the two quantized twins only. Everything else is under
  `research/`.
- Timing is ungated: `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `official_or_ranked_score=false` are
  preserved verbatim, arms are ABBA-counterbalanced in one session, and entry
  and exit GPU temperature are recorded per arm.

---

## Rung 1b prereg — `t6_rbx`, the occupancy-cliff hypothesis

Registered before `research/e61_reg_census.py` ran on the rbx arms and before
any `t6_rbx` timing leg. Advisor comment 5349751724 supplied the hypothesis and
the reading.

**Hypothesis.** The -33.1 GB/s step from NA=5 to NA=6 is an occupancy cliff
caused by the 125 -> 144 register jump, not a property of streaming six inputs.
The threadgroup is 64 threads and the kernel is bandwidth bound, so losing one
resident threadgroup per multiprocessor removes latency hiding.

**Mechanism under test.** The shipped `qmv_fast_crossrow_affine4_g64_m` wrapper
computes a runtime `first_m = tid.x * IPG` and then chooses the tail NA from
it. The `rbx` form selects the group from `tid.x` first and hands each group a
literal first input row. Group count, per-group NA and per-group first input
row are unchanged, so no output element changes its K accumulation order.

I implemented this from the mechanism the advisor described in prose. I did not
read PR #62; this launch scopes me to `senpai/qwen38-mtp-r1` and my own branch.
So `t6_rbx` is my construction of the described idea, not thorfinn's bytes, and
its register count must be measured rather than assumed to match his 95.

**Preregistered reading, from the advisor.**

- `bw(6)` under `t6_rbx` at or above ~135 GB/s AND the M=6 cell register count
  near 110 or below: the cliff is occupancy. Rung 3 times `t6_rbx`.
- `bw(6)` unchanged within the rung 1 controls: the cliff is intrinsic to six
  streams. Rung 3 times `t6` as planned, and the hypothesis is dead.

**My own additions, registered now.**

- `shipped_rbx` is included so the wrapper rewrite is separable from the
  schedule change. Under `t6_rbx` every width except 6 is treated by the
  wrapper rewrite alone, so those widths are NOT an untreated-width null the
  way they were under `t6`. I will report them as the rbx-wrapper effect and
  not as a null.
- Register prediction for `t6_rbx` at the M=6 cell: I predict a **small**
  reduction only, 135 to 144, because at `<T,6,6>` the shipped wrapper already
  has `TAIL == 0`, so its `else` branch is already dead and the only thing
  `rbx` removes is the runtime input-row base. I expect the large saving the
  advisor quotes for `<T,9,5>` to come mostly from separating two live group
  widths, which `<T,6,6>` does not have. If the count lands near 110 my
  reasoning is wrong and the occupancy reading is live.
- `shipped_rbx` table maximum prediction: 129, unchanged or slightly below.

