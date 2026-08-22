# E117 — the `[4+4]` rate deficit, and what a serialised N-split can do about it

PR https://github.com/morganmcg1/qwen38-challenge_senpai/pull/119

## Summary

E115 found one positive cell: a two-way serialised N-split made `mlp.gate_up`
6.70 % faster at NA=4. This experiment asked whether that cell survives in the
shipped dispatch frame. It does not. It also asked what causes the underlying
rate deficit. The answer moves the whole question onto firmer ground.

1. **The original hypothesis is dead.** At the shipped width M=8, `[4+4]`, the
   serialised N-split makes `mlp.gate_up` **14.276 % ± 0.783 slower**, not
   faster. The kill rule was +2.0 %. `mlp.gate_up` is closed.
2. **The advisor's pricing and mine were both built on a frame error**, now
   recorded as ADVISOR ERROR 77: an isolated one-group cell was weighted by a
   realised-width histogram from a different dispatch grouping. Priced in the
   shipped M frame the mechanism is worth +0.108 % of the local round, half the
   0.20 % bar, before any serialisation is paid.
3. **A different cell replicates and is live.** `gdn.in_proj` at M=8 gains
   **+6.560 % ± 0.803**, replicated across two sessions.
4. **The cause is not what anyone proposed.** The deficit is not a resonance in
   N, not a working-set threshold, not `grid.y`, and not a property of the
   tensor. Within the IPG=4 family the rate collapses onto a single curve in the
   grid volume `M x grid.y`, with a trough at 16384 to 18432 threadgroups.
   `mlp.gate_up` at M=4 and `gdn.in_proj` at M=8 are the same defect reached
   from opposite directions.
5. The N-split is therefore **not a mechanism**. It is a way of moving grid
   volume off the trough, and it pays exactly when the halved volume lands on a
   faster point of the same curve.

## Experiment identity tuple

| field | value |
| --- | --- |
| harness | `local` |
| base SHA | `1d2320bece29cddc94b95e5f99f00331b05a5025` |
| branch | `qwen-thorfinn/e117-gate-up-na4-rate-dip-and-serialised-n-split` |
| rung 0 probe commit | `61cd56fc` |
| rung 0b probe commit | `78795f42` |
| host | `ip-10-231-2-95.ec2.internal` |
| chip | Apple M4 Pro, 20 GPU cores, 48 GiB unified |
| OS | macOS 26.0 |
| toolchain | Swift 6.3.3, `swiftlang-6.3.3.1.3 clang-2100.1.1.101` |
| build | `swift test -c release --force-resolved-versions` |
| thermal mode | ungated, `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, `timing_valid=false` |
| reference source | the probe's own single-dispatch `a_one` output, per cell |
| dispatch | `quantizedMM`, affine 4-bit, group 64, `transpose: true` |

This is a kernel-frame instrument. It does not decode tokens, so there is no
token window, fixture, proposal head, or row digest for any measurement here.
No arm in this experiment is gate-qualified and none is an official score.

## Instrument certificates

Three independent checks say the probe measures the object it claims to.

**Round arithmetic.** `mlp.gate_up` `a_one` at M=8 is 1026.43 µs net. Over 64
layers that is 65,691 µs. Finding 22 puts `mlp.gate_up` at 37.937 % of the local
round, and 37.937 % of edward's rung-2 frame of 171,384 µs is 65,020 µs. The two
agree to **1.03 %**. A standalone kernel probe reproduces a share measured end to
end by a different student on a different harness.

**Three-way cross-session agreement on `lm_head`.** `a_one` net GB/s at
M=2/3/4/5:

| source | M=2 | M=3 | M=4 | M=5 |
| --- | --- | --- | --- | --- |
| E117 rung 0 | 245.4 | 239.4 | 207.0 | 175.4 |
| E115 reverse pass | 244.1 | 238.7 | 206.6 | 175.1 |
| E111 isolated, alphonse | 242.2 | 239.7 | 206.9 | 173.5 |

Three independently written probes agree inside 0.6 % at every width.

**Within-experiment replication.** Rung 0b re-measured two rung-0 cells on
synthetic tensors of the same shape, in a separate process:

| cell | rung 0 | rung 0b | gap |
| --- | --- | --- | --- |
| N=16480 M=8 `a_one` | 172.5 GB/s | 172.7 GB/s | 0.12 % |
| N=34816 M=8 `a_one` | 195.4 GB/s | 195.2 GB/s | 0.10 % |

## Method

`Tests/MLXFastTests/E117WidthFrameProbeTests.swift` drives the real
`quantizedMM` entry point, so the shipped partition table is exercised as
shipped. `quantized.cpp:251-254` dispatches `grid_dims(M, (N+7)/8, B)` with
`group_dims(32, 2, 1)`, so `ntg.x == M`. `quantized.h:1157-1186`
`qmv_fast_crossrow_affine4_g64_m` sets `first_m = tid.x * IPG` and early-returns
when `first_m >= M`, so `M` threadgroups launch in x but only `ceil(M / IPG)` do
work. The realised partition, from `quantized.h:1922-1979`:

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IPG | – | 2 | 3 | 4 | 5 | 3 | 4 | 4 | 3 |
| partition | [1] | [2] | [3] | [4] | [5] | [3+3] | [4+3] | **[4+4]** | [3+3+3] |
| working groups | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 3 |

Arms, all reading the same buffers:

| arm | dispatches | evals | what it is |
| --- | --- | --- | --- |
| `a_one` | 1 | 1 | one dispatch, M rows, full N |
| `c_nsplit` | 2 | 1 | two concurrent dispatches over disjoint N halves |
| `e_nsplit_serial` | 2 | 2 | the same two dispatches, separated by a blocking host `eval` |

`control.small` is a 8192 x 64 tensor whose GPU work is about 1.5 µs. It runs
every arm structure, so its cell time is that structure's host cost. The
analysis subtracts each arm's own control. `e_nsplit_serial` net is therefore
the **idealised free-barrier ceiling**: the GPU-side value of serialising, with
the host round trip removed rather than paid.

Estimator, per CAMPAIGN RULE 39: paired within-block contrast on the net
estimator, mean over blocks, standard error of the per-block contrasts. Each
block times every arm forward and then in reverse, so monotone thermal drift
cancels to first order.

### Named frames

Every percentage of a round in this report names its frame. Mixing two of them
in one sum is the arithmetic error corrected in the rung-0 pricing below.

| frame | round | where it comes from |
| --- | --- | --- |
| `probe-M8` | 173,168 µs | this probe. `mlp.gate_up` costs 64 x 1026.43 = 65,691 µs here, and Finding 22 puts `mlp.gate_up` at 37.937 % of a round, so the implied round is 65,691 / 0.37937. Cross-check: edward's rung-2 frame is 171,384 µs, 1.0 % away. |
| `traced-512` | 160,590 µs | E112, traced local MTP round median, 512 tokens (range 160,590–161,800). |
| `decode-M5` | 102,864 µs | decode-only round at M=5. |
| `E96-anchor` | 127,533 µs | E96. |

`probe-M8` is an **all-M=8 idealisation**: it counts every layer at the widest
shipped width, with the GPU otherwise idle and clocks ramped. `traced-512` is
the closest thing here to a real round. `probe-M8` is 7.8 % larger, which is
the direction and roughly the size one should expect from that idealisation.
Where a rung-1 or rung-0 number is quoted as a share of a round, it is given
in absolute microseconds first and then against both frames, so no conclusion
depends on the choice.

A per-round absolute figure of the form `L x per-layer µs` also assumes every
layer runs at M=8. E109 v2's realised histogram puts 88.49 % of `gdn.in_proj`
and 86.96 % of `mlp.gate_up` streaming time at M=8, so that idealisation is
close but not exact, and it is stated wherever it is used.

## Rung 0 — the dip in the shipped M frame

`mlp.gate_up`, N=34816, K=5120. 8 blocks. `harness=local`. W&B `zbe3jt4y`
https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/zbe3jt4y

| M | partition | grp | `a_one` net µs | GB/s | `e_nsplit_serial` net % | se |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [1] | 1 | 432.26 | 232.0 | −3.109 | 0.221 |
| 2 | [2] | 1 | 434.08 | 231.0 | −10.704 | 6.260 |
| 3 | [3] | 1 | 457.13 | 219.3 | −8.359 | 5.631 |
| 4 | [4] | 1 | 592.09 | **169.4** | **+8.735** | 2.875 |
| 5 | [5] | 1 | 664.87 | 150.8 | +1.906 | 0.455 |
| 6 | [3+3] | 2 | 891.86 | 224.9 | +1.273 | 1.803 |
| 7 | [4+3] | 2 | 962.53 | 208.3 | −12.670 | 0.208 |
| 8 | **[4+4]** | 2 | 1026.43 | **195.4** | **−14.276** | 0.783 |
| 9 | [3+3+3] | 3 | 1280.11 | 235.0 | −9.999 | 0.601 |

The isolated `[4]` cell reproduces E115 exactly at 169.4 GB/s, and the split
pays +8.7 % there against E115's +6.70 %. The instrument agrees with itself. The
shipped width does not agree with the isolated width.

All three estimators agree on the headline: mean −14.276 ± 0.783, reverse
−15.449 ± 0.234, forward −13.166 ± 1.484. The full palindrome mean is used
throughout; reverse-only is reported as a check, not as a fallback.

### The discriminator

```
2 x M=4 [4]        1184.17 us net
M=8   [4+4]        1026.43 us net
ratio               0.8668
M=8 saving         13.32 %
```

`[4+4]` is much better than two `[4]`. Grouping recovers 13.3 % of the isolated
pair cost unaided. In the rate frame the dip halves rather than vanishing:

```
isolated  [3]   219.3 GB/s  ->  [4]   169.4 GB/s     dip 22.75 %
grouped   [3+3] 224.9 GB/s  ->  [4+4] 195.4 GB/s     dip 13.12 %
```

58 % of the relative dip survives grouping and 42 % was an artefact of the
single-group dispatch. The surviving half is real. The serialised N-split cannot
reach it.

### Pricing the null in the shipped frame

E109 v2's realised histogram gives verify widths `{4:1, 5:1, 6:2, 7:1, 8:26}`
over 31 rounds. Weighting each width by `count x a_one_net(M)` gives streaming
shares `{4: 0.0193, 5: 0.0217, 6: 0.0581, 7: 0.0314, 8: 0.8696}`, so 87 % of
`mlp.gate_up` streaming time is spent at M=8.

Priced in absolute microseconds per round, in the `probe-M8` frame defined
above, so that no step of the sum changes frame:

```
gate_up streaming time, 64 layers at M=8               65,691 us
always split           -12.527 % of that              -8,229.1 us
split only where positive, M in {4,5,6}   +0.284 %      +186.6 us
minus 64 boundaries x 1.80 us                           -115.2 us
net                                                      +71.4 us
                                     = +0.041 % of probe-M8   (173,168 us)
                                     = +0.044 % of traced-512 (160,590 us)
```

Zero, against a 0.20 % bar, and that is before any serialisation is paid. This
is the corrected version of the +1.11 % and +1.73 % in the assignment. Two
general lessons:

1. **An isolated per-group cell may not be weighted by a realised width
   histogram unless the dispatch grouping is the same in both frames.**
2. **A gain quoted as a share of one op and a cost quoted in microseconds
   cannot be added until both are in one named frame.** An earlier interim
   comment on this PR added `+0.108 %` (gate_up share, implicitly `probe-M8`)
   to `-0.112 %` (115.2 us over the `decode-M5` round) and reached `-0.004 %`.
   Both terms were individually right and the sum was wrong by the frame
   ratio 173,168 / 102,864 = 1.683. The corrected net is `+0.041 %`, still
   far below the bar, so the rung-0 kill is unchanged.

## Rung 0b — what causes the deficit

13 synthetic tensors at fixed K=5120, M in {2,3,4,5,8}, 6 blocks.
W&B `93mrc16r`
https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/93mrc16r

`a_one` net GB/s. `split@8` is `e_nsplit_serial` net % faster at M=8.
`harness=local`.

| N | grid.y | MB | M=2 | M=3 | M=4 | M=5 | M=8 | split@8 % | se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 1024 | 23.6 | 60.1 † | 40.9 † | 121.8 † | 82.4 † | 185.0 | −13.911 | 2.248 |
| 12288 | 1536 | 35.4 | 201.8 † | 88.9 † | 154.3 † | 143.3 | 194.1 | −11.494 | 0.262 |
| 14336 | 1792 | 41.3 | 202.4 | 88.1 | 166.1 | 147.3 | 193.5 | −8.086 | 0.498 |
| 16384 | 2048 | 47.2 | 216.8 | 197.5 | 173.5 | 153.7 | **173.5** | **+4.884** | 0.302 |
| 16480 | 2060 | 47.5 | 219.2 | 194.2 | 169.5 | 154.3 | **172.7** | **+6.560** | 0.803 |
| 20480 | 2560 | 59.0 | 224.0 | 207.4 | 179.6 | 157.7 | 182.9 | +2.026 | 0.298 |
| 24576 | 3072 | 70.8 | 228.3 | 214.5 | 189.2 | 158.0 | 185.4 | +2.198 | 0.162 |
| 32768 | 4096 | 94.4 | 232.2 | 221.4 | **169.1** | 149.3 | 192.8 | −4.858 † | 6.812 |
| 33792 | 4224 | 97.3 | 231.6 | 217.5 | **169.0** | 151.0 | 195.3 | −13.501 | 1.660 |
| 34816 | 4352 | 100.3 | 229.8 | 221.9 | **169.1** | 150.4 | 195.2 | −13.392 | 1.833 |
| 35840 | 4480 | 103.2 | 229.5 | 221.6 | **168.0** | 151.4 | 195.3 | −12.721 | 1.936 |
| 36864 | 4608 | 106.2 | 235.3 | 203.2 | **169.4** | 151.2 | 195.8 | −15.287 | 0.174 |
| 40960 | 5120 | 118.0 | 204.4 | 198.8 | 178.5 | 153.1 | 198.0 | −10.001 | 0.202 |

† contains a block flagged by harness defect 19.

### The deficit is not in N, in bytes, or in `grid.y`

- **Not a resonance at one N.** 16384 is a clean power of two and 16480 is
  2060 x 8. Both read 173.5 and 172.7. An address-aliasing resonance would not
  treat those alike.
- **Not a working-set threshold.** The deficit sits at 47 MB while 41 MB reads
  193.5 and 94 MB reads 192.8.
- **Not smooth in N.** The M=8 curve rises to 194 at 12288, falls 11 % at
  16384–16480, recovers to 183–185 at 20480–24576, and reaches 195–198 above
  32768. That is a trough with a floor and two shoulders.
- **Not a property of the tensor.** A synthetic 16384 x 5120 dips as hard as
  `gdn.in_proj`.
- **Not bytes and not `grid.y`.** One pair of cells excludes both:

```
N = 32768, K = 5120     94.4 MB     grid.y = 4096
  M = 4  [4]      169.1 GB/s
  M = 8  [4+4]    192.8 GB/s
```

Same tensor, same bytes, same `grid.y`, same IPG, 14 % apart.

### It is the grid volume `M x grid.y`

M=4 and M=8 both partition into groups of four, so they are the same IPG family.
Overlaid on grid volume, the rate curve collapses:

| grid volume | M=4 (N) | M=8 (N) |
| --- | --- | --- |
| 8192 | 173.5 (16384) | 185.0 (8192) |
| 12288 | 189.2 (24576) | 194.1 (12288) |
| 16384 | **169.1** (32768) | **173.5** (16384) |
| 16480 | – | **172.7** (16480) |
| 16896 | **169.0** (33792) | – |
| 17408 | **169.1** (34816) | – |
| 17920 | **168.0** (35840) | – |
| 18432 | **169.4** (36864) | – |
| 20480 | 178.5 (40960) | 182.9 (20480) |

M=8 sits 2.5 % to 6.6 % above M=4 at equal volume, which is the amortisation of
the second working group. The shape is identical at every overlapping point: a
local peak near 12288, a flat trough of about −11 % over 16384 to 18432, and
recovery from 20480 upward.

Within the IPG=4 family, launched threadgroups are always exactly four times the
working threadgroups, so this sweep cannot say which of the two carries the
effect. Crossing to IPG=3 or IPG=5 changes the per-group rate at the same time
and is not a clean discriminator. The defensible observable is `M x grid.y`.

### The split has no free parameters left

If the rate is a function of grid volume alone, a two-way N split moves volume
`V` to two dispatches of `V/2`, so the predicted gain is
`1 − rate(V) / rate(V/2)`:

| N | rate(V) | rate(V/2) | predicted % | measured % |
| --- | --- | --- | --- | --- |
| 16384 | 173.5 | 185.0 | +6.2 | +4.9 |
| 16480 | 172.7 | ~185 | +6.6 | +6.6 |
| 20480 | 182.9 | ~190 | +3.7 | +2.0 |
| 24576 | 185.4 | 194.1 | +4.5 | +2.2 |
| 33792 | 195.3 | ~174 | −12.3 | −13.5 |
| 34816 | 195.2 | ~176 | −11.0 | −13.4 |
| 40960 | 198.0 | 182.9 | −8.3 | −10.0 |

The sign is right in all seven and the magnitude is within a few points. The
split is not a mechanism. It is a move along this curve. A three-way split is
not obviously better, because the curve falls again below volume 8000.

### Transfer risk

The trough is a band, not a point, which is better than a knife-edge. But the
band is in grid volume, and grid volume interacts with the core count of the
part. This host is a 20-core M4 Pro. The ranked runner is M5. There is no
evidence here about where the trough sits on M5. The honest position is that
**the trough location is host-dependent, so this lever is host-tunable but not
host-portable.** A hard-coded N-split predicate chosen on M4 Pro could be worth
+6.5 % here and negative on the ranked runner.

## Frame dilution — every campaign `A` is an upper bound

Requested as a named heading. A campaign `A` is measured on an isolated kernel
cell. In the current tree the same kernel runs beside other work, so the
measured `A` is diluted by the fraction `F` of time the tensor does not own.

Measured current-tree `A_kernel`, defined as the rate ratio `G x t1 / tG` where
`G` is the working group count, on `mlp.gate_up`:

| comparison | `A_kernel` |
| --- | --- |
| `[4+4]` vs `[4]` | **1.1537** |
| `[3+3]` vs `[3]` | 1.0251 |
| `[3+3+3]` vs `[3]` | 1.0713 |

The E115 pre-registration bracket said `A_kernel` is 1.550 at F=20 % and 1.400
at F=40 %, with a ceiling of 1.640 at F=0. The measured values are far below the
whole bracket. **The bracket is a loose upper bound, not an estimate**, and
should be quoted that way. In-dispatch grouping time ratios on the same tensor:

| ratio | value |
| --- | --- |
| `[3+3] / [3]` | 1.9510 |
| `[4+4] / [4]` | 1.7336 |
| `[3+3+3] / [3]` | 2.8003 |
| `[4+3] / ([4] + [3])` | 0.9174 |

## Withdrawing E115's `f = 0.667`

Requested as follow-up 3. **E115's `f = 0.667` is withdrawn.** It plugged the
`b_msplit` **time** ratio 1.960 into a **rate**-ratio identity with
`A_local = 1.640`. As a rate ratio, `b_msplit` is `2 / 1.960 = 1.0204`, and
solving the same identity then gives `f = 31.4`, which is impossible. The
quantity was never a fraction of the round.

Redone in one time frame, with `t2 / t1 = 1 / (1 − 0.180) = 1.2195`:

| group ratio `q2/q1` | source | implied `phi` |
| --- | --- | --- |
| 1.7336 | `[4+4] / [4]` | 0.299 |
| 1.9510 | `[3+3] / [3]` | 0.231 |

Direct check in the M frame, avoiding the identity entirely. `[5]` measures
664.87 µs. An estimated `[3+2]` is 869.37 µs, from `[3] + [2] = 891.21` less the
2.45 % grouping saving. That is 23.52 % faster, and 23.52 % x the 0.59977
wide-branch share is **14.11 % of the round**, against E100's measured 18.0 %.
The narrow-branch `out_proj`/`down_proj` family, 28.666 % of the round, accounts
for the remainder. **Finding 22 and E100 agree**; they partition the round
differently and there was never a contradiction to resolve, only a broken
algebraic bridge between them.

## Instrument work

### Harness defect 16, fixed by construction

Sampling `macmon` runs a subprocess and leaves the GPU idle long enough to drop
its clocks. The DVFS ramp costs a fixed 30 to 80 ms of wall clock and is paid
entirely by whichever arm is timed first, so a palindrome does not cancel it.
E115 inserted a ramp burst of `count / 4` replicates, which is a **fixed
replicate count**, and on `mlp.gate_up` that was only about 43 ms.

The fix: temperature is sampled only at block boundaries, and every sample is
followed by a discarded burst of fixed **wall-clock** duration, 0.30 s, which
dominates the ramp regardless of how fast the cell is. The burst also runs
before replicate calibration, so the replicate count is not chosen from a
low-clock reading.

Residual, median `100 x (forward / reverse − 1)` on `a_one`, against E115's
+61.6 %:

| shape | rung 0 | rung 0b |
| --- | --- | --- |
| `mlp.gate_up` / N=34816 | +0.277 | +0.172 |
| `lm_head` | −0.019 | – |
| `gdn.in_proj` / N=16480 | +1.548 | +0.918 |
| `fa.qkv` / N=14336 | +2.278 | – |
| large N, 20480 to 40960 | – | 0.109 to 0.745 |

### Harness defect 19, now instrumented

A few whole timed blocks read three to four times their neighbours. Rung 0's
`fa.qkv` M=3 read forward 1254.7 µs against reverse 300.0 µs in four blocks of
eight. That is external interruption of a whole timed region, not the
first-position ramp bias, and pooling it corrupts the cell.

Every cell now reports its per-block median, min, max and `max / median`. Any
block above 1.5 x the cell median is flagged and excluded from a parallel
trimmed contrast. Both contrasts are in the artifacts.

Rung 0b: 210 cells, 11 with a flagged block, 11 flagged blocks. **No
load-bearing cell is flagged.** N=16384, N=16480 and N=34816 are clean at every
width. The flag catches the one cell whose standard error blew out, `n32768`
M=8 at ±6.812, and explains the implausible low rates at N=8192 and N=12288,
which are not interpreted anywhere in this report.

### Fidelity

| session | cells | `nsplit_bit_exact` | `positive_control_differs` |
| --- | --- | --- | --- |
| rung 0 | 45 | 45 / 45 | 45 / 45 |
| rung 0b | 70 | 70 / 70 | 70 / 70 |

The positive control concatenates the two halves in swapped order and must move
the digest, so the check can fail.

Slice aliasing: `activeMemory` around the hoisted half views moves by 0 bytes on
`mlp.gate_up` (100.3 MB) and `lm_head` (715.2 MB) in rung 0. In rung 0b eleven of
thirteen shapes read a **negative** delta, between −5.2 MB and −26.2 MB, which is
the allocator releasing cached blocks during the check. No shape read a positive
delta, so no slice copied. The rung-1 probe pins `Memory.cacheLimit = 0` around
the check so the delta reads zero rather than negative.

## Reproduction

```bash
# rung 0, the M frame
research/e117_probe.sh e117-rung0-mframe \
  'mlp.gate_up,lm_head,gdn.in_proj,fa.qkv,control.small' 1,2,3,4,5,6,7,8,9 8

# rung 0b, the N sweep at fixed K
research/e117_probe.sh e117-rung0b-nsweep \
  'n08192:8192:5120,n12288:12288:5120,n14336:14336:5120,n16384:16384:5120,\
n16480:16480:5120,n20480:20480:5120,n24576:24576:5120,n32768:32768:5120,\
n33792:33792:5120,n34816:34816:5120,n35840:35840:5120,n36864:36864:5120,\
n40960:40960:5120,control.small' 2,3,4,5,8 6

python3 research/e117_analysis.py research/out/TAG/cells.json --json OUT.json
python3 research/e117_wandb_log.py research/out/TAG --name NAME --rung N
python3 research/e117_reconcile.py
```

Artifacts under `research/e117-artifacts/`, committed per CAMPAIGN RULE 40:
`rung0-mframe-summary.json`, `rung0-mframe-meta.txt`, `rung0-pricing.txt`,
`rung0b-nsweep-summary.json`, `rung0b-nsweep-meta.txt`.

## Suggested follow-ups, not implemented

1. **Find the trough on the ranked part.** The whole value of this lever depends
   on where the grid-volume trough sits on M5. A one-session N sweep on the
   official runner would settle it. Nothing else in this report transfers
   without it.
2. **Separate launched from working threadgroups.** Within one IPG family the
   two are exactly proportional. A kernel-side experiment that varies the early
   return without changing `ntg.x` would separate them, but that is inside
   `quantized.h`, which I do not own this round. **Reporting, not implementing,
   per the assignment.**
3. **The trough may be reachable without any split.** If the carrier is launched
   threadgroups, then `M x ceil(N/8)` empty-group launches are the cost, and a
   dispatch-shape change in `quantized.cpp` that stops launching `M` groups in x
   when only `ceil(M/IPG)` can work would remove 75 % of them at IPG=4. That is
   also outside my scope this round and is the single largest thing this
   experiment found.
4. **`mlp.gate_up` at M=4 is in the trough on this host.** It is only 1.9 % of
   `gate_up` streaming time locally, so it does not pay. If the realised width
   histogram ever shifts toward M=4, revisit.
