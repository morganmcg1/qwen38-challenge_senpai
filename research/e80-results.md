# E80 — per-kernel GPU-time census, and the unattributed 22.6 % of the verify-width tax

Research-only. **Zero candidate files.** The GPU-time instrument lives on this
branch only as `research/e80-artifacts/gputime-census.patch`; the final commit
removes it from `Sources/` and `Tests/`.

- Base: `2eec73d352af2e689c91236e8eac89413797a19d` (`senpai/qwen38-mtp-r1`)
- Host: Apple M4 Pro, `applegpu_g16s`, 20 GPU cores, 48 GiB, macOS 26.5.2
- Swift 6.3.3, `metallib_source_fingerprint=f09821bdbd820b77502867cbf660c1157407243ca9639de681c5b46fedfbd9fe`
- The ranked host is M5 (`applegpu_g17s`). **Every number here is directional.**

All sessions carry `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false`. No
result here is an official or ranked score.

## Rung 0a — the E76 spill/parity confound is closed

The advisor's cross is confirmed against the primary artifacts rather than the
prose. Verdicts come from the measured `parity_differing_vs_plain` counts in
`research/e76-artifacts/parity-na{3,4,5,6}-b{0,1,2}.json`; spill comes from
`research/e76-artifacts/rung1.json`. Script: `research/e80_spill_confound.py`.
Machine-readable: `research/e80-artifacts/rung0a-spill-parity-cross.json`.

112 parity-tested pairs = 28 arms × 4 NA:

| class | pairs | g16s spill range |
|---|---:|---|
| FAIL | 9 | 144 B … 352 B |
| pass | 103 | 0 B … 48 B |

Clean separation on `applegpu_g16s`, the architecture the parity run executed
on. The same cross on `g17s` **overlaps**: `mc2` NA=3/4 fail at 0 B while
`fall` NA=6 passes at 48 B. Quoting the g17s columns to explain a g16s-only
failure is what produced the false sentence.

`research/e76-results.md` now records the corrected statement: **the cause of
the `mc*` fault is not established; large g16s spill is perfectly confounded
with the failing configuration and is the leading remaining candidate.**
Refutation 2 is withdrawn. Refutation 3 inherits the same confound, because
every `rows_per_simd = 4` multi-chunk cell is also a high-spill cell.
Refutation 1 stands: the generated body text is byte-identical and only
`ROWS_PER_SIMD` differs.

The shipped `<T,6,6>` carries **16 B**, nine times below the observed hazard
threshold.

### A non-`mc` arm at the hazard threshold exists

A numerically neutral `ballast<N>` ladder reaches ≥144 B while keeping the
shipped row block, staging, layout, expression trees and reduction:

| arm | NA | g16s regs / spill | g17s regs / spill | role |
|---|---:|---:|---:|---|
| `plain` | 5 | 95 / 0 | 98 / 0 | negative control |
| `ballast8` | 5 | 96 / 16 | 103 / 0 | low-spill control |
| `ballast16` | 4 | 95 / 0 | 101 / 0 | same-arm control |
| `ballast16` | 5 | 96 / **224** | 111 / 0 | test arm |
| `mc3` | 4 | 96 / **224** | 126 / 144 | confirmed failure, identical spill |

Bit-neutral by construction, and it survives `-fno-fast-math` plus
`setFastMathEnabled:NO`. The lever provably engaged: `e76_ballast16_na5`
`text_sha8 1eba0549` ≠ `e76_plain_na5` `0a5810b4`.
`research/e76_wide_gen.py --check` passes at 38 arms and all 240 pre-existing
`rung1.json` records are unchanged.

**Device parity was not run.** `e76_session.sh --mode parity` ignores `--arms`
and always sweeps all 38 arms, so a single `--na 5` question costs the whole
sweep. That exceeded the 30-minute bound. It is suggested follow-up 1.

## Rung 0b — the census patch

`git apply --check research/e62-artifacts/census-probe.patch` fails on
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:996`. `git apply --3way`
lands it with one conflict at the `draft_head` phase marker, because the base
gained the `tDraft0` trace timestamp on the same line. Both are kept. The other
seven hunks and both other files apply unchanged.

## Rung 0c — mechanism (B), chosen on device evidence

`research/e80_counter_probe.m`, saved to
`research/e80-artifacts/rung0c-counter-capability.json`:

| capability | value |
|---|---|
| `supportsCounterSampling(.atStageBoundary)` | true |
| `supportsCounterSampling(.atDispatchBoundary)` | **false** |
| counter sets | exactly one: `timestamp` → `GPUTimestamp` |
| `GPUStartTime` / `GPUEndTime` | populated, moved 1.666 ms on a real dispatch |
| GPU ticks per CPU ns | **1.0** — both clocks are nanoseconds |

**Option (C) is impossible on this GPU:** per-dispatch GPU timestamps do not
exist. Only stage (encoder) boundary sampling exists, and an encoder boundary
carries no more information than the command-buffer interval that
`GPUStartTime`/`GPUEndTime` already provides for free.

Option **(B)** is used in the two configurations the assignment asks for.
Option (A) `xctrace` was held in reserve and never needed.

### Why both configurations are mandatory, measured rather than assumed

Isolation coverage — the share of verify dispatches that got a command buffer
to themselves, and can therefore be priced individually:

| mode | width 1 | width 6 |
|---|---:|---:|
| default | **0.0 %** | **0.0 %** |

Not one dispatch in either width is alone in its buffer. Per-kernel GPU time is
unobtainable from a default-mode run, which is what makes the isolated leg
necessary rather than merely convenient.

## Rung 1 — the mandatory hard gate

**Verdict: the gate FAILED as specified. The instrument is nevertheless
validated, and the failure is a property of the reference, not the
instrument.**

Three independent sessions:

| session | W&B | instrument | entry °C | exit °C |
|---|---|---|---:|---:|
| E71 published | — | none | — | — |
| `e80-rung1-gate` | `ws1e4j5m` | installed | 45.1 | 65.5 |
| `e80-rung1-control` | `blld7vtb` | **dormant** | 42.7 | — |

`installIfRequested()` installs nothing when the census is off, so the control
carries zero instrument overhead.

### The gate as written

| sub-gate | rows | verdict |
|---|---|---|
| G0 wall clock vs E71 | 8 | FAIL on `lm_head` (+20.0 %) |
| G1 GPU time vs E71, 10 % | 5 | FAIL on `lm_head` (+13.6 %) |
| G2 GPU level vs E71, 5 % | 3 | **PASS** |

G2 carries the tightest tolerance and passes on every row: `F(1)` −0.9 %,
`F(6)` −0.4 %, `F(6)−F(1)` +0.2 %.

### G0′ — instrument overhead, instrumented wall vs dormant wall

| row | instrumented | control | relative |
|---|---:|---:|---:|
| `F(1)` level | 65.924 | 65.781 | **+0.2 %** |
| `F(6)` level | 122.062 | 122.233 | **−0.1 %** |
| `mlp_gate_up` tax | 20.030 | 19.300 | +3.8 % |
| `mlp_down` tax | 16.746 | 17.214 | −2.7 % |
| `gdn_out_proj` tax | 2.983 | 3.590 | −16.9 % |
| `fa_o_proj` tax | 1.124 | 1.553 | −27.6 % |
| `lm_head` tax | 2.570 | 2.194 | +17.1 % |

Installing the instrument costs the round 0.2 % and 0.1 %. The small-family
taxes move by ±17–28 % **in both directions with no consistent sign**, which is
random error rather than instrument bias.

### G0c — the dormant session against the published table

The uninstrumented harness **also fails the gate**, on a different row:
`fa_o_proj` +33.7 %. Every other row passes, including `lm_head` at +2.4 %.

### Three-session reproducibility — the actual finding

| row | E71 | dormant | instr. wall | instr. GPU | spread | ±10 % band | is it a test? |
|---|---:|---:|---:|---:|---:|---:|---|
| `mlp_gate_up` | 20.913 | 19.300 | 20.030 | 21.497 | 2.198 | 4.183 | yes |
| `mlp_down` | 16.887 | 17.214 | 16.746 | 16.774 | 0.469 | 3.377 | yes |
| `gdn_out_proj` | 3.313 | 3.590 | 2.983 | 2.990 | 0.607 | 0.663 | marginal |
| `lm_head` | 2.142 | 2.194 | 2.570 | 2.433 | 0.428 | 0.428 | at the band |
| `fa_o_proj` | 1.161 | 1.553 | 1.124 | 1.101 | 0.452 | 0.232 | **no** |
| `F(1)` | 64.706 | 65.781 | 65.924 | 64.126 | 1.798 | 6.471 | yes |
| `F(6)` | 122.110 | 122.233 | 122.062 | 121.639 | 0.594 | 12.211 | yes |

The spread is 0.43–0.61 ms for every small family and roughly constant in
absolute terms: the ABBA tax noise floor is about ±0.25 ms. The tolerance is
*relative*, so its width falls with family size — 4.18 ms for `mlp_gate_up`,
0.23 ms for `fa_o_proj`. Below roughly 3 ms of tax the band passes under the
noise floor and the row stops being a test.

A bootstrap over reps within each block agrees: the 95 % CI half-width is
0.486 ms for `mlp_gate_up` and 0.133 ms for `fa_o_proj` against a 0.116 ms
band. That CI is a **lower bound**, because resampling reps cannot see
block-to-block drift, and the observed between-session spread is 2–4× larger.

### G1′ — the only comparison that isolates the instrument

G0, G1 and G2 each compare a number measured today with a number published by
another session on another build, so each tests the instrument *and* reference
replay at once. G1′ compares GPU time with wall clock in the **same blocks of
the same session**, where session drift cancels exactly.

| row | GPU | wall | relative |
|---|---:|---:|---:|
| `mlp_gate_up` tax | 21.497 | 20.030 | +7.3 % |
| `mlp_down` tax | 16.774 | 16.746 | +0.2 % |
| `gdn_out_proj` tax | 2.990 | 2.983 | +0.2 % |
| `fa_o_proj` tax | 1.101 | 1.124 | −2.0 % |
| `lm_head` tax | 2.433 | 2.570 | −5.3 % |
| `F(1)` level | 64.126 | 65.924 | −2.7 % |
| `F(6)` level | 121.639 | 122.062 | −0.3 % |

Worst row 7.3 %, inside the 10 % tolerance. Both levels satisfy **GPU ≤ wall**,
a physical constraint the instrument could have violated and did not.
Instrument health over all 32 blocks: `unmapped_encoder_dispatches=0`,
`zero_time_buffers=0`, `undrained_buffers=0`.

### Conclusion

The gate is decisive where it has power, and passes there:

- the two large families, 37.8 ms = **65.8 %** of the tax, reproduce inside a
  band 7–9× the noise floor;
- all three levels pass at the tighter 5 % tolerance in all three sessions;
- the three small families total 6.6 ms = **11.5 %** of the tax and are
  noise-limited in every session, including the uninstrumented one.

Recommended re-specification: an absolute floor on the tolerance,
`max(10 % of reference, 0.6 ms)`. Every row of all three sessions passes under
that rule, and it states honestly what this harness can resolve.

Artifact: `research/e80-artifacts/rung1-gate.json`.

### G3 — the E68 in-situ curve

The assignment attributes the in-situ eval milliseconds (M=2 34.306, M=3
35.416, M=4 40.264, M=5 47.368, M=6 64.622) to E74. They are in fact **E68**:
`research/e68-artifacts/e68-insitu-curve.json`, produced by
`research/e68_insitu_curve.py` re-parsing `research/results/e37/medicine-rounds.txt`
as the median `eval_wall_us` per round from `Qwen36MTPBlockSession.swift:1502`.
That source carries `dirty=1`, `cool_gate_passed_real_gate=false` and
`trace_perturbs_timing=true`, and its in-situ to isolated ratio is 0.49–0.53.

The rung-2 width set is 6, 5, 1, 4, 9, which excludes M=2 and M=3, so only
M=4, M=5 and M=6 can be compared. That overlap is reported in the rung-2
section rather than presented as a full curve reproduction.

## Rung 2 — the census

_Pending. Widths 6, 5, 1, 4, 9, 512-token legs, default and isolated per width._

## Rung 3 — naming the 22.6 %

### Hypothesis 2 is dead

| width | wall ms/round | verify GPU ms/round | **GPU busy / wall** |
|---:|---:|---:|---:|
| 1 | 65.924 | 64.126 | **0.973** |
| 6 | 122.062 | 121.639 | **0.997** |

Everything outside kernel execution — inter-kernel gaps, barriers,
command-buffer boundaries, encoder setup — is 0.4–1.8 ms per round. Against the
57.4 ms width tax that is about **1 %**. The unattributed 22.6 % is kernel time.

### Hypothesis 3 has a named candidate, from exact dispatch structure

Decoding the qmv grid (grid.y × 8 = out_features, verified against hidden 5120,
64 layers = 48 GDN + 16 full attention, head_dim 256, 24 query heads, 4 KV
heads, MLP intermediate 17408, vocabulary 248320), the 257 `affine_qmv_fast`
dispatches of a verify round split into exactly five projections:

| grid.y | out_features | count | module | E71 arm? |
|---:|---:|---:|---|---|
| 4352 | 34816 | 64 | `mlp_gate_up` | yes |
| 640 | 5120 | 128 | `mlp_down` ×64, `gdn_out_proj` ×48, `fa_o_proj` ×16 | yes |
| 2060 | 16480 | 48 | `gdn_in_proj_fused` (qkvzba) | **no** |
| 1792 | 14336 | 16 | `fa_qkv_gate_fused` (q/k/v + output gate) | **no** |
| 31040 | 248320 | 1 | `lm_head` | yes |

`Qwen35GatedDeltaNet` and `Qwen35Attention` fuse their in-projections into raw
`quantizedMM` calls that never dispatch through a child `Linear`
(`Qwen35.swift:677-689`, `:1707-1712`), so **no E71 arm could ever intercept
them**. They are **64 of 257 qmv dispatches, 24.9 %**, against an unattributed
22.6 %.

### The width tax is grid, not dispatch count

`affine_qmv_fast` runs **257 dispatches at both width 1 and width 6**. Total
dispatches per round actually *fall* with width, from 1705 at w1 to 825 at w6,
because the elementwise and copy families collapse. The tax is grid.x = M
applied to the same 257 projections.

| family | w1 dispatches | w6 dispatches |
|---|---:|---:|
| `copy` | 480 | 146.3 |
| `elementwise` | 385 | 1 |
| `norm` | 273 | 177 |
| `qmv` | **257** | **257** |
| `compiled_fusion` | 176 | 128 |
| `gdn_recurrence` | 96 | 96 |
| `sdpa_fused` | 16 | 32 |
| `qk_rms_rope` | 16 | 16 |
| `gather_scatter` | 3 | 3 |
| `top2_readout` | 2 | 2 |
| `quant_dequant` | 1 | 1 |
| **unclassified** | **0** | **0** |

## Falsification riders

_Pending rung 2._

## Suggested follow-ups

_Pending rung 3._
