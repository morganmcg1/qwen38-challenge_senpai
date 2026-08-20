# E80 — per-kernel GPU-time census, and the unattributed 22.6 % of the verify-width tax

Research-only. **Zero candidate files.** The GPU-time instrument lives on this
branch only as `research/e80-artifacts/gputime-census.patch`; the final commit
removes it from `Sources/` and `Tests/`.

- Base: `2eec73d352af2e689c91236e8eac89413797a19d` (`senpai/qwen38-mtp-r1`)
- Host: Apple M4 Pro, `applegpu_g16s`, 20 GPU cores, 48 GiB, macOS 26.5.2
- Swift 6.3.3, `metallib_source_fingerprint=f09821bdbd820b77502867cbf660c1157407243ca9639de681c5b46fedfbd9fe`
- The ranked host is M5 (`applegpu_g17s`). **Every number here is directional.**

`official_or_ranked_score=false` for every session in this file. No result here
is an official or ranked score.

Gate status is **not** uniform and the sets are never pooled:

- The rung 0 and rung 1 sessions carry `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false`.
- The rung 2 census sweep 1 legs carry `cool_gate_passed_real_gate=true` and
  `gate_qualified_for_timing=true`.
- The rung 2 census sweep 2 legs carry `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false`. They are counterbalanced ABBA within one
  session under the standing authorisation in `program.md`, and every leg
  records entry and exit GPU temperature.

### Headline results

1. **The planned NNLS attribution is wrong on this workload and was replaced.**
   A command-buffer interval is a maximum over its dispatches, not a sum. The
   census uses fit-free dominant-dispatch attribution with a published
   partner-spread validation. NNLS would have failed the `rms_norm` rider at
   10.5 % on a pure solver artefact.
2. **The unattributed 22.6 % of the verify-width tax is named.**
   `gdn_in_proj_fused` at 7.820 ms/round and `fa_qkv_gate_fused` at 2.343
   ms/round are 17.75 % of the tax on their own; with the drafting-only non-qmv
   work and an under-count inside the pooled `qmv Mx640x1` the total is 12.65 ms
   = 22.10 %, against E71's 12.609 ms residual. Both are candidate reachable.
3. **The proposal head costs a flat 4.810 ms per draft token** and carries no
   fixed cost: `draft_head_ms = 4.8098 × drafts − 0.113`, R² = 0.999972 over
   widths 4, 5 and 9. It is 15.1 % of the whole round at width 4 and 18.8 % at
   width 9. It is purely bandwidth-bound, streaming 849 MB of bf16 weights per
   draft token at 245 GB/s. The `gemv < 2 %` rider **fails at every width**, and
   fails harder as width grows: 10.30 % at width 4, 13.39 % at width 9.
4. **The local harness cannot run the declared head.** Every local measurement
   in this campaign uses the organizer-pinned bf16 head at 849,398,784 bytes,
   not the 427,742,600-byte artifact that `mtp-head.manifest.json` declares for
   the ranked candidate leg. A bandwidth model puts local pessimism at 5.1 % to
   6.7 % of whole-round GPU time.
5. **H-221 is closed in every form.** The closure gap decomposes into measured
   host dispatch and commit time. The decode path takes **no host
   synchronisation points at all** at any width from 1 to 9, and the measured
   per-dispatch host cost is 0.66–1.55 µs, 220 to 530 times below 0.35 ms.
6. **The rung 1 gate as written is unpassable by any instrument**, including a
   dormant one. A re-spec is recommended.
7. **Disabling the cool gate costs +0.12 % on this host**, measured two
   independent ways: the per-leg `F(1)` thermometer across both packings, and a
   direct gated-versus-ungated replicate at width 5 isolated that differs by
   +0.11 % on the whole round for +7.8 °C of entry temperature.
8. **Deeper drafting has an almost flat ceiling and a steeply rising
   break-even.** The speedup at perfect acceptance spans only 2.683 to 2.850
   across widths 4 to 9, while the acceptance rate needed merely to break even
   climbs from 15.9 % to 27.0 %. This needs no acceptance model: the round cost
   is fixed once the width is chosen, and each leg supplies its own serial
   reference.

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

### The attribution method, and why the planned one was discarded

The census was specified around a non-negative least-squares fit: solve for a
per-kernel nanosecond rate from the measured command-buffer intervals, then
price every dispatch at its fitted rate. That method is **wrong on this
workload** and the census does not use it.

A command buffer is a measured interval. Attribution is the problem of
splitting that interval among the dispatches inside it. NNLS answers the
question only if the interval is a sum of independent per-dispatch costs. On
this device it is not. The falsifying measurement, from the w6 isolated leg:

| buffer contents at M=6 | buffers | measured ns |
|---|---:|---:|
| `qmv 6x4352x1` + `swiglu_fusion 17408x6x1` | 3360 | 746,672 |
| `qmv 6x4352x1` + `residual_rms_norm 6144x1x1` | 2720 | 746,530 |

The two partners differ by a factor of seventeen in element count. The interval
differs by **0.02 %**. The same holds at M=1 (426,836 against 428,752, 0.4 %)
and for `qmv 6x2060x1` across three different partners (370,221 / 376,231 /
369,621, a 1.8 % spread). One dispatch sets the interval and the others are
free inside it.

Fitting a sum to data generated by a maximum produces confident nonsense. NNLS
priced `residual_rms_norm` at 101,783 ns per dispatch, which is 13.0 ms/round
for the `norm` family — while a buffer containing that same norm plus two small
copies measures 6,065 ns in total. The `rms_norm < 3 %` rider would have
**failed at 10.5 %** on a pure artefact of the solver's null space.

**The method used instead — dominant-dispatch attribution.** Charge each
buffer's whole measured interval to the one dispatch that sets it, chosen from
a fixed priority order over kernel families. No solver, no fit, no null space.
The attributed total reproduces the measured phase total exactly by
construction, because every interval is assigned once.

**The validation that makes it honest — the partner-spread check.** For every
owning dispatch the census publishes the interval distribution across the
distinct partner sets it was observed with. If a dispatch really sets its
buffer, that spread is small. Where the spread is large, the census must
explain it before the row is trusted. The check earned its place in this run:
`gemv_al_bfloat16 160x1x1` showed a 2.8x spread, which turned out to be three
different head matrices sharing one grid signature and differing only in the
reduction dimension (see the draft-head section). The spread flagged a real
ambiguity in the key; bandwidth arithmetic resolved it without a fit.

**Reusable instrument, stated for the next experiment.** (1) Record grid,
threadgroup, kernel name and the owning buffer for every dispatch, plus the
buffer's GPU interval. (2) Normalise grid axes that scale with the batch
dimension so one signature covers all widths. (3) Assign each buffer to its
dominant dispatch by family priority. (4) Publish, for every owner, the
interval spread across partner sets and the count of co-dispatches. (5) Treat
any row whose spread exceeds a few per cent as unresolved until an independent
argument — bytes moved, arithmetic, or a shape identity — explains it. (6)
Never publish a per-kernel rate that was inferred rather than measured.

### Sessions and legs

All legs ran on the same host, from the same commit, with the same metallib
fingerprint `f09821bd…fbd9fe`, at 512 decode tokens, and with the target's
draft width forced. Two packings were measured for every width: `default`, the
shipped command-buffer packing, and `isolated`, one dispatch per MLX op.
`isolated` is the attribution leg because its buffers are small enough to be
meaningful; `default` supplies the absolute anchor and the concurrency
discount.

Each leg also runs a serial reference pass in the same session, so every leg
carries its own `F(1)` at width 1 in phase `target_forward`. That makes `F(1)`
an internal thermometer: comparing `F(1)` between a gated and an ungated leg
bounds thermal inflation directly, without leaving the session.

**Gate status is recorded per leg and the two sets are never pooled.**

Sweep 1 ran with the real cool gate enabled, from commit `773dff41`, `dirty=0`.
Every leg below reports `cool_gate_passed_real_gate=true` and
`gate_qualified_for_timing=true`. None is an official or ranked score.

| set | width | packing | rounds | started (UTC) | finished (UTC) | entry °C | exit °C |
|---|---:|---|---:|---|---|---:|---:|
| gated | 6 | default | 95 | 16:37:28 | 16:48:42 | 43.030 | 58.921 |
| gated | 6 | isolated | 95 | 16:49:10 | 17:02:00 | 49.385 | 61.113 |
| gated | 5 | isolated | 110 | 17:06:58 | 17:20:44 | 43.470 | 60.058 |
| gated | 1 | default | 512 | 17:21:12 | 17:39:10 | 49.943 | 57.146 |
| gated | 1 | isolated | 512 | 17:39:39 | 17:56:57 | 48.747 | 58.103 |

Entry-temperature spread across the gated set is **6.9 °C**, from 43.030 to
49.943. The gate passes at those temperatures rather than at 40.0 °C, so the
gated set is not thermally tight either. That is stated here because the
ungated set below is judged against it.

Five legs of sweep 1 never produced drafting rounds:

| width | packing | outcome | entry °C |
|---:|---|---|---:|
| 5 | default | gate unreachable, host floor 42.0 °C against a 40.0 °C target | 49.629 |
| 4 | default | gate failed at 40.157 °C, floor 40.3 °C, waited 260 s | 49.581 |
| 4 | isolated | gate failed | 42.733 |
| 9 | default | serial pass completed, gate failed before the drafting pass | 42.765 |
| 9 | isolated | gate failed | 42.575 |

The width 9 default leg is the one partial: its serial reference pass ran to
completion and contributes 512 rounds of width-1 data, but it produced no
width-9 verify rounds. Those widths were rerun ungated under the standing
authorisation in `program.md`; that set is reported separately below and is
never merged with the gated table.

### Head provenance — which artifact these numbers describe

`metrics.head_provenance_sha256` is `c5791f65bf026de7be0277c34b79156c09879955a1acad7776dddae7c2dd9c2d`
for every leg. Applying the tree digest rule from `mtp-head/README.md` to the
resident cache reproduces that digest exactly, over four records that match
`fixtures/qwen3_8_27b_mtp_head.sha256` hash-for-hash plus the wrapper's own
reference-cache lock.

**The resident head was the organizer-pinned `EigenLabs/Qwen3.8-27B-MTP-bf16@26a328e0`.**
`setup-qwen-mtp.sh:66` hardcodes that repository and never reads
`mtp-head.manifest.json`, so the local wrapper cannot run the declared head.
`metrics.uses_pinned_mtp_head` does not discriminate: `Sources/MLXFastCLI/main.swift:1966`
sets it from `report.usesNativeMTPHead`, which reports whether the head drafted
at all.

This label matters only for the draft-head phase. The head does not appear in
`target_verify`, so the verify-side census, the width tax, and every rider on
the target path are independent of it.

### Family census, width 6, isolated, `target_verify`, 95 rounds, 123.072 ms/round

| family | dispatches/round | GPU ms/round | share |
|---|---:|---:|---:|
| `qmv` | 257.0 | 117.875 | 95.78 % |
| `sdpa` | 32.0 | 2.273 | 1.85 % |
| `gdn_recurrence` | 106.1 | 2.021 | 1.64 % |
| `copy` | 140.2 | 0.554 | 0.45 % |
| `compiled_fusion` | 128.0 | 0.241 | 0.20 % |
| `top2_readout` | 2.0 | 0.062 | 0.05 % |
| `norm` | 193.0 | 0.034 | 0.03 % |
| `quant_dequant` | 1.0 | 0.007 | 0.01 % |
| `gather` | 3.0 | 0.005 | 0.00 % |
| `elementwise` | 1.0 | 0.000 | 0.00 % |
| **total** | **863.3** | **123.072** | **100 %** |

`unclassified_kernels: 0`, verified by an explicit family classifier rather
than assumed: every observed kernel name is matched against the family table
and any miss is reported as `UNCLASSIFIED`. 177.0 dispatches per round never
own a buffer; their time is reported inside the owner that shares the buffer
with them rather than invented.

### Family census, width 1, `target_forward`, 1024 rounds, 65.142 ms/round

| family | dispatches/round | GPU ms/round | share |
|---|---:|---:|---:|
| `qmv` | 257.0 | 61.732 | 94.77 % |
| `gdn_recurrence` | 48.0 | 1.048 | 1.61 % |
| `sdpa` | 16.0 | 0.685 | 1.05 % |
| `copy` | 480.4 | 0.558 | 0.86 % |
| `norm` | 289.0 | 0.330 | 0.51 % |
| `compiled_fusion` | 176.0 | 0.316 | 0.49 % |
| `elementwise` | 385.0 | 0.226 | 0.35 % |
| `depthwise_conv` | 48.0 | 0.201 | 0.31 % |
| `top2_readout` | 2.0 | 0.033 | 0.05 % |
| `quant_dequant` | 1.0 | 0.007 | 0.01 % |
| `gather` | 3.0 | 0.005 | 0.00 % |
| **total** | **1705.4** | **65.142** | **100 %** |

`unclassified_kernels: 0`. 416.0 dispatches per round never own a buffer. Note
the dispatch count **falls** from 1705 at width 1 to 863 at width 6 while GPU
time nearly doubles: the elementwise and copy families collapse into strided
fusions as soon as M exceeds 1.

### The draft head, width 6, isolated, 95 rounds, 24.146 ms/round

The proposal head is 24.146 / 147.217 = **16.4 % of whole-round GPU time** at
width 6. Both packings agree: the `w6|draft_head` phase totals are 2287.3 ms
(default) and 2293.8 ms (isolated), a 0.3 % difference, so this is not an
isolation artefact.

| family | dispatches/round | GPU ms/round | share of head |
|---|---:|---:|---:|
| `gemv` | 24.1 | 16.903 | 70.01 % |
| `qmv` | 5.0 | 5.764 | 23.87 % |
| `steel_gemm` | 3.7 | 0.828 | 3.43 % |
| `sdpa` | 5.0 | 0.210 | 0.87 % |
| `draft_select` | 5.0 | 0.167 | 0.69 % |
| `norm` | 31.9 | 0.152 | 0.63 % |
| `copy` | 26.6 | 0.066 | 0.27 % |
| `quant_dequant` | 5.0 | 0.025 | 0.10 % |
| `gather` | 15.0 | 0.023 | 0.09 % |
| `rope` | 0.9 | 0.006 | 0.03 % |
| `elementwise` | 10.0 | 0.000 | 0.00 % |
| `compiled_fusion` | 10.0 | 0.000 | 0.00 % |
| **total** | **142.2** | **24.146** | **100 %** |

Every `gemv` dispatch is named. `gemv_al_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`
uses threadgroup 32x1x8 and covers 32 output rows per threadgroup, so
`grid.x × 32` is the output width. That maps one-to-one onto the resident
head's tensor table, read directly from the safetensors header:

| head matrix | tensor shape | grid | weight bytes | ns/dispatch | GB/s |
|---|---|---|---:|---:|---:|
| `mlp.{gate,up}_proj` fused | 2 × (17408, 5120) | `1088x1x1` | 356,515,840 | 1,435,656 | 248 |
| `mlp.down_proj` | (5120, 17408) | `160x1x1` | 178,257,920 | 729,837 | 244 |
| `self_attn.{q,k,v}_proj` fused | (14336, 5120) | `448x1x1` | 146,800,640 | 600,168 | 245 |
| `fc` | (5120, 10240) | `160x1x1` | 104,857,600 | 436,837 | 240 |
| `self_attn.o_proj` | (5120, 6144) | `160x1x1` | 62,914,560 | 261,060 | 241 |
| **sum** | | | **849,346,560** | **3,463,558** | **245** |

The byte column sums to the whole head weight file. The fixture pins
`mtp_head.tensor_bytes = 849,398,784`; the 52,224-byte remainder is the five
norm vectors, which the census sees as `rms_looped` at 0.056 ms/round. **The
head streams its complete 849 MB of bf16 weights once per draft token and is
purely bandwidth-bound at 245 GB/s.** Five draft steps per round is 4.25 GB.

The three `160x1x1` rows are the partner-spread check working as designed.
Their intervals cluster at 729,837 / 436,837 / 261,060 ns, a 2.8x spread on one
signature, because `grid.x` encodes the output width only and `down_proj`
(K = 17408), `fc` (K = 10240) and `o_proj` (K = 6144) differ solely in the
reduction dimension. Bytes over time gives 244 / 240 / 241 GB/s for the three,
which resolves the ambiguity without a fit.

The remaining `qmv 1x12292x1`, 5.0 dispatches per round at 1,152,808 ns, is the
quantized readout applied once per draft step. Its cost tracks the target's own
`lm_head` exactly by grid: 31040/12292 = 2.53 against a time ratio of
2,932,685/1,152,808 = 2.54.

### Concurrency discount

| level | width | default ms/round | isolated ms/round | discount |
|---|---:|---:|---:|---:|
| phase | 6 | 121.306 | 123.072 | 0.986 |
| phase | 5 | 94.134 | 95.393 | 0.987 |

Isolated packing inflates measured GPU time by only 1.4 %, so the attribution
leg is a faithful stand-in for the shipped packing at the phase level.

**Per-family concurrency discount is not identifiable in the default packing,
and the census does not publish a fabricated one.** Default buffers almost
always contain a `qmv`, so dominant-dispatch attribution assigns 121.240 of
121.306 ms — 99.9 % — to `qmv` alone, against 117.875 of 123.072 ms (95.8 %)
when isolated. A per-family ratio computed from those two columns reads 0.000
for every non-`qmv` family. That is a statement about the packing, not about
concurrency, and it is reported as such rather than as a measurement.

### Ranked-weighted verify cost, gated set

| width | ranked weight % | verify ms/round | `qmv` share | draft head ms/round | whole round ms/round |
|---:|---:|---:|---:|---:|---:|
| 5 | 24.1 | 95.384 | 95.74 % | 18.967 | 114.352 |
| 6 | 33.4 | 123.072 | 95.78 % | 24.146 | 147.217 |

Covered mass **57.50 %** of the ranked histogram. No missing width is
interpolated. Over that covered mass the weighted verify cost is
**111.467 ms/round** and the weighted whole round is **133.442 ms/round**.

The whole-round riders at width 5 reproduce the width-6 verdicts with the same
signs and nearly the same magnitudes: `copy` 0.242 ms and 0.21 % PASS,
`elementwise` 0.000 PASS, `norm` 0.145 and 0.13 % PASS, `sdpa` 1.841 and 1.61 %
PASS, `gemv` 13.213 and **11.55 % FAIL**, `qmv` 95.871 and 83.84 % PASS. The
`gemv` share is 11.48 % at width 6 and 11.55 % at width 5, which is what a
per-draft-token cost should look like when both the head and the verify grow
with the same width.

### Seed window, and the convergence with the prefill census

The census also covers the 512-token seed, phase `w0|seed_prefill`:

| leg | seed GPU ms | buffers | dispatches |
|---|---:|---:|---:|
| w6 default | 7,988.1 | 132 | 4,530 |
| w6 isolated | 16,042.9 | 4,180 | 9,060 |

The default-packing seed is 7.99 s of GPU time against a serial decode round of
64.522 ms, so the seed costs about 124 serial rounds, or roughly 24 % of a
512-round serial leg. It is a first-class part of every timed leg.

The seed also carries the fusion gate found independently in the prefill work.
`Qwen35.swift:1003` reads `if S <= 9, let fused = fusedInProjections(inputs) {`.
At S = 512 that gate fails, so the seed issues four separate quantized GEMMs per
GDN layer where decode issues one fused `qmv 2060`. The decode-side census
prices the fused form at 48 dispatches per round and 7.820 ms of the width tax,
which is the same object seen from the other end. **Raising the gate above 9 has
never been attempted on the board.** This census does not test that change and
does not widen its scope to it; the connection is recorded so the two results
can be read together.

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

## Rung 2b — the ungated ABBA widths 4, 5 and 9

Five of the ten gated legs in sweep 1 produced no drafting rounds, because the
host temperature floor sat above the 40.0 °C gate. The advisor authorised an
ungated counterbalanced session for the missing widths under the three standing
conditions in `program.md`. Sweep 2 ran six legs from one clean commit
`cdf33bb6` between 18:36:31Z and 19:01:10Z, all exit 0.

**These legs carry `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false`.** They are directional causal evidence inside
their own counterbalanced session. They are never pooled with the gated set, and
no number in this section is an official or ranked score.

### Session record

| order | leg | mode | rounds | start | end | entry °C | exit °C | `dirty` |
|---:|---|---|---:|---|---|---:|---:|---:|
| 1 | `e80-hot-w4-default` | D | 132 | 18:37:19Z | 18:40:55Z | 43.179 | 61.846 | 0 |
| 2 | `e80-hot-w4-isolated` | I | 132 | 18:41:24Z | 18:44:53Z | 51.096 | 65.278 | 0 |
| 3 | `e80-hot-w9-isolated` | I | 66 | 18:45:17Z | 18:48:51Z | 53.568 | 64.316 | 0 |
| 4 | `e80-hot-w9-default` | D | 66 | 18:49:19Z | 18:53:00Z | 52.214 | 63.088 | 0 |
| 5 | `e80-hot-w5-default` | D | 109 | 18:53:28Z | 18:57:08Z | 51.275 | 62.690 | 1 |
| 6 | `e80-hot-w5-isolated` | I | 109 | 18:57:37Z | 19:01:10Z | 51.316 | 64.525 | 6 |

Counterbalancing: mode order is D I I D D I. The default legs sit at positions
1, 4 and 5 with mean position 3.33; the isolated legs sit at 2, 3 and 6 with
mean position 3.67. Monotone thermal drift therefore cancels to first order
between the two packings.

**Entry-temperature spread: 10.4 °C** (43.179 to 53.568). The gated sweep's
spread was 6.9 °C. The next subsection measures what that spread is worth.

### The two `dirty` legs, disclosed

Legs 5 and 6 recorded `dirty=1` and `dirty=6`. `dirty` is
`git status --porcelain | wc -l`, and I was writing research Python in the
worktree while the sweep ran. The six files are exactly the six files in commit
`6c00521`, all under `research/`.

The timed build is unaffected, and the evidence is checkable rather than
asserted:

- all six legs record the same `candidate_sha=cdf33bb628d8672908e092ced6863047289e8af6`;
- `.build-worker/release/mlxfast-runtime-worker` has mtime 18:37:10Z, before the
  first leg started at 18:37:19Z, and it never changed again, so all six legs
  ran one byte-identical binary;
- no dirty path lies under `Sources/`, `Vendor/`, `Tests/` or `mtp-head/`.

### What the cool gate is worth on this host

Every leg runs a 512-round serial pass before it drafts, so each leg carries its
own `F(1)` thermometer measured under its own thermal conditions.

| packing | gated `F(1)` ms | ungated `F(1)` ms | gated mean | ungated mean | delta |
|---|---|---|---:|---:|---:|
| default | 64.522, 64.527 | 64.661, 64.542, 64.595 | 64.525 | 64.599 | **+0.12 %** |
| isolated | 64.464, 64.460 | 64.575, 64.495, 64.556 | 64.462 | 64.542 | **+0.12 %** |

The gated `e80-census-w6-isolated` leg read 65.819 and is excluded from the
isolated mean as an outlier at +2.1 %; including it would move the isolated
delta to −0.57 % and change the sign, which is why it is named rather than
quietly averaged in.

There is also a direct replicate at one width and one packing:

| leg | gate | entry °C | `draft_head` | `target_verify` | whole round |
|---|---|---:|---:|---:|---:|
| `e80-census-w5-isolated` | gated | 43.470 | 18.970 | 95.393 | 114.363 |
| `e80-hot-w5-isolated` | ungated | 51.316 | 19.052 | 95.440 | **114.492** |

**+0.11 % on the whole round for +7.8 °C of entry temperature.** Both readings
agree that the cool gate is worth about +0.12 % on this host. That is below the
±0.25 ms noise floor established in rung 1 and far below the 0.2 ms/round
threshold the advisor set for a result worth building on. The ungated widths are
usable directional evidence. They remain separately labelled.

### Width tax at 4 and 9

`F(1)` for the ungated set is 64.542 ms/round pooled over 1536 serial rounds
inside the three isolated legs.

| width | `F(M)` verify ms | tax ms | named rows | closure |
|---:|---:|---:|---:|---:|
| 4 | 81.011 | 16.469 | 16.408 | 99.63 % |
| 5 | 95.440 | 30.898 | 30.838 | 99.81 % |
| 9 | 165.477 | 100.936 | 100.916 | 99.98 % |

The same five `qmv` grids carry the tax at every width, and their shares are
stable:

| owning dispatch | unit | w4 share | w5 share | w6 share (gated) | w9 share |
|---|---|---:|---:|---:|---:|
| `qmv Mx640x1` | `mlp_down` + `gdn_out_proj` + `fa_o_proj` | 46.64 % | 39.96 % | 39.53 % | 31.91 % |
| `qmv Mx4352x1` | `mlp_gate_up` | 31.02 % | 35.31 % | 35.75 % | 40.77 % |
| `qmv Mx2060x1` | `gdn_in_proj_fused` | 14.11 % | 14.43 % | 13.66 % | 15.04 % |
| `qmv Mx1792x1` | `fa_qkv_gate_fused` | 4.57 % | 4.48 % | 4.09 % | 4.40 % |
| `qmv Mx31040x1` | `lm_head` | 3.02 % | 3.66 % | 3.85 % | 4.54 % |

`gdn_in_proj_fused` and `fa_qkv_gate_fused` together hold **18.7 % of the tax at
width 4, 18.9 % at width 5, 17.8 % at width 6 and 19.4 % at width 9.** The
naming of the unattributed 22.6 % therefore does not depend on width 6; it holds
across the whole ranked histogram.

Two secondary rows only exist while drafting and are worth naming, because they
grow faster than the `qmv` rows at small width:

| owning dispatch | w4 ms | w9 ms |
|---|---:|---:|
| `sdpa_c 24x{4,5}x1` | 1.419 | 3.005 |
| `gdn_prework 32xMx80` | 1.220 | 2.122 |

At width 4 those two are 16.0 % of the tax; at width 9 they are 5.1 %. They are
a fixed drafting overhead that amortises with depth, which is the opposite
behaviour to the `qmv` rows.

### Riders at every ungated width

| family | rider | w4 | w5 | w9 | verdict |
|---|---|---:|---:|---:|---|
| `copy` | ≤ 1 %, ~0.02 % expected | 0.19 % | 0.25 % | 0.39 % | PASS |
| `elementwise` | `unary/binary/ternary_ops` < 3 % | 0.00 % | 0.00 % | 0.00 % | PASS |
| `norm` | `rms_norm` < 3 % | 0.12 % | 0.13 % | 0.13 % | PASS |
| `sdpa` | `sdpa_vector` < 3 % | 1.62 % | 1.61 % | 1.64 % | PASS |
| `gemv` | `gemv` < 2 % | **10.30 %** | **11.56 %** | **13.39 %** | **FAIL** |
| `qmv` | five linear families dominate | 84.83 % | 83.78 % | 82.38 % | PASS |
| remainder | not a sixth linear family | 15.17 % | 16.22 % | 17.62 % | PASS |

`unclassified_kernels` is 0 at every width in both sets.

The `copy` rider is the one to watch. It passes everywhere, but it rises
monotonically with width — 0.19 %, 0.25 %, 0.42 % (w6 gated), 0.39 % — and is
already 20 times the ~0.02 % that ledger 218 expected. It is not near the 1 %
reopening threshold at any measured width, so ledger 218 stays closed, but the
trend is real and worth a line in the ledger.

### Concurrency discount, all three ungated widths

| width | default ms/round | isolated ms/round | phase-level discount |
|---:|---:|---:|---:|
| 4 | 79.440 | 81.011 | 0.981 |
| 5 | 93.327 | 95.440 | 0.978 |
| 9 | 163.399 | 165.477 | 0.987 |

Every width agrees with the gated widths 5 and 6 at 0.987 and 0.986. **Default
packing buys 1.3 % to 2.2 % of overlap and no more.** The verify phase is
essentially serialised already, which is itself a result: there is no large
pool of hidden concurrency to recover by re-packing command buffers.

Per-family discounts remain unidentifiable, for the reason given in rung 2. In
default packing MLX assigns 163.309 of 163.399 ms at width 9 to the `qmv` owner,
because `qmv` dominates almost every mixed buffer, so every other family reads a
discount of exactly 0.000. Those numbers are an artifact of the attribution
under coarse packing, not a measurement, and are not published as one.

### Ranked-weighted cost over the fuller width coverage

The two sets are weighted separately. Widths 4, 5 and 9 carry 44.05 % of the
ranked histogram; widths 5 and 6 carry 57.50 %.

| set | widths | covered mass | weighted verify ms/round | weighted whole round ms/round |
|---|---|---:|---:|---:|
| ungated | 4, 5, 9 | 44.05 % | 99.931 | 119.998 |
| gate-qualified | 5, 6 | 57.50 % | 111.467 | 133.442 |

Taking widths 4, 5, 6 and 9 together spans 77.45 % of the histogram, but doing
so would pool the two gate sets, so the combined figure is deliberately not
computed.

## Proposal-head cost per draft token

This is the largest single actionable finding in the census, and it is the one
that needed widths 4 and 9 to establish.

### The head is a straight line in draft depth with no fixed cost

| width | set | draft tokens | rounds | `draft_head` ms/round | ms per draft token | whole round | head share |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | ungated | 3 | 132 | 14.375 | 4.792 | 95.386 | 15.07 % |
| 5 | ungated | 4 | 109 | 19.052 | 4.763 | 114.492 | 16.64 % |
| 9 | ungated | 8 | 66 | 38.380 | 4.797 | 203.857 | 18.83 % |
| 5 | gated | 4 | 110 | 18.970 | 4.743 | 114.363 | 16.59 % |
| 6 | gated | 5 | 95 | 24.146 | 4.829 | 147.217 | 16.40 % |

Least squares over the three ungated widths:

```text
draft_head_ms = 4.8098 * drafts - 0.1130      R^2 = 0.999972
```

The intercept is 2.3 % of one draft token, so the phase carries **no measurable
fixed cost**. Every millisecond of head time is bought one draft token at a
time. The gated widths are not pooled into that fit, and they land on it anyway:

| width | measured ms | ungated-fit prediction | delta |
|---:|---:|---:|---:|
| 5 | 18.970 | 19.126 | −0.81 % |
| 6 | 24.146 | 23.936 | +0.88 % |

At 4.810 ms per draft token the head moves its whole 849.4 MB of weights once
per draft token, an effective 177 GB/s across the entire phase and 245 GB/s
inside its five `gemv` rows.

### Cost per draft token by family

| width | set | `gemv` | `qmv` | `steel_gemm` | other |
|---:|---|---:|---:|---:|---:|
| 5 | gated | 3.303 | 1.137 | 0.184 | 0.118 |
| 6 | gated | 3.381 | 1.153 | 0.166 | 0.130 |
| 4 | ungated | 3.276 | 1.140 | 0.242 | 0.134 |
| 5 | ungated | 3.308 | 1.141 | 0.184 | 0.131 |
| 9 | ungated | 3.412 | 1.156 | 0.105 | 0.125 |

`gemv` holds within ±2 % and `qmv` within ±0.8 % across a 2.7× change in draft
depth and across both gate sets. The head does exactly the same work per draft
token no matter how deep the schedule goes.

The width-9 leg reproduces the width-6 byte table independently. Its
`gemv_bm8 160x1x1` row shows four distinct partner intervals — 734,064,
435,798, 261,662 and 259,076 ns — against the width-6 leg's 729,837 (`down_proj`),
436,837 (`fc`) and 261,060 ns (`o_proj`). Three matrices share one output width
and differ only in `K`, and both legs price them the same to within 0.6 %. That
is the partner-spread check doing the job it was built for, across two
independent sessions and two gate sets.

### What the declared head would cost instead

The resident artifact holds 849,398,784 bytes. `mtp-head.manifest.json` declares
427,742,600 bytes, **50.36 %** of it. The head phase is bandwidth-bound, so bytes
price it. Two models bracket the answer:

- the **narrow** model scales only `gemv`, which the byte table shows is exactly
  the head weight tensors;
- the **wide** model also scales the phase's `qmv` row, which reads about 281 MB
  of already-quantized weights that the head artifact does not contain.

| width | set | measured head | narrow model | narrow saving | wide saving | narrow share of round | wide share of round |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | ungated | 14.375 | 9.497 | 4.879 | 6.937 | 5.11 % | 7.27 % |
| 5 | ungated | 19.052 | 12.484 | 6.568 | 9.198 | 5.74 % | 8.03 % |
| 5 | gated | 18.970 | 12.411 | 6.559 | 9.182 | 5.74 % | 8.03 % |
| 6 | gated | 24.146 | 15.755 | 8.391 | 11.664 | 5.70 % | 7.92 % |
| 9 | ungated | 38.380 | 24.829 | 13.551 | 18.556 | 6.65 % | 9.10 % |

**Every local drafting measurement on this host is pessimistic by roughly 5 % to
7 % of whole-round GPU time, and the direction is fixed.** The head cost lands
only on the MTP leg, because the serial leg does not draft, so this bias does
not cancel in the local serial-to-MTP ratio.

Both rows are models, not measurements. Each assumes the declared artifact
reaches the same GB/s. One number argues against reading them too literally:
50.36 % of a bf16 artifact is not a 4-bit artifact, since affine-4 group-64
would be about 28 %. The declared head is therefore either wider at lower
precision or a different architecture, and its acceptance behaviour may differ
as well. I cannot resolve its shapes without downloading it, and I did not.

What the census does offer is a free, falsifiable discriminator for any future
ranked-side trace: **the ranked candidate leg's head phase should dispatch
`qmv`, not `gemv`.** MLX chooses `QuantizedLinear` over `Linear` at load time on
the presence of a `.scales` tensor (`Load.swift:250-258`), so the kernel name
alone identifies which artifact is resident, at zero extra GPU time.

## The economics of draft depth

Having whole-round cost at widths 4, 5, 6 and 9 makes one campaign-level
question answerable without any acceptance model at all.

A drafting round at width `M` costs a fixed amount of GPU time whatever the
target accepts, because the target verifies all `M` rows before acceptance is
known. The census measures that cost. Each leg also measures its own serial cost
per token in the same session, on the same host, at the same temperature. Two
numbers follow directly:

- the **ceiling**, `M × F(1) / round`, the speedup if every draft is accepted;
- the **break-even**, the accepted-token count per round at which speedup
  reaches exactly 1.0.

Each row uses its own leg's `F(1)`, so no cross-leg or cross-gate-set comparison
is involved.

| width | set | leg | rounds | own `F(1)` ms | round ms | ceiling | break-even accepted tokens | break-even rate |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 4 | ungated | `e80-hot-w4-isolated` | 132 | 64.575 | 95.386 | 2.708 | 0.477 of 3 | 15.9 % |
| 5 | ungated | `e80-hot-w5-isolated` | 109 | 64.495 | 114.492 | 2.817 | 0.775 of 4 | 19.4 % |
| 5 | gated | `e80-census-w5-isolated` | 109 | 64.464 | 114.352 | 2.819 | 0.774 of 4 | 19.3 % |
| 6 | gated | `e80-census-w6-isolated` | 95 | 65.819 | 147.217 | 2.683 | 1.237 of 5 | 24.7 % |
| 9 | ungated | `e80-hot-w9-isolated` | 66 | 64.556 | 203.857 | 2.850 | 2.158 of 8 | 27.0 % |

**The ceiling is flat. It spans 2.683 to 2.850 across widths 4 to 9, a range of
6.2 %, while the break-even acceptance rate climbs from 15.9 % to 27.0 %.**

That asymmetry is the result. Going deeper buys almost nothing at the top and
costs a great deal at the bottom. Width 9 raises the perfect-acceptance ceiling
over width 4 by 5.2 %, and raises the acceptance rate the schedule must sustain
merely to break even by 11.1 points. The reason is visible in the census: the
`qmv` verify rows grow superlinearly in `M` — the marginal cost of one more
draft token is 10.3 ms between widths 1 and 4, 19.1 ms between 4 and 5, and
22.3 ms between 5 and 9 — while the head adds a further flat 4.810 ms each.

Width 6 reads slightly low. Its leg is the one whose `F(1)` thermometer sat at
65.819 ms, 2.1 % above every other leg. Substituting the typical 64.5 ms would
move its ceiling to 2.628, which would make width 6 the *worst* of the four
measured widths rather than merely unremarkable. I flag this rather than pick a
number: my base's QMV template table specializes width 6 as `<T,6,6>`, whereas
the advisor base at `6acb0d15` specializes it as `<T,6,3>`, so width 6 is
exactly one of the widths where the two bases differ.

Under the narrow declared-head model the whole curve shifts up and keeps its
shape:

| width | set | modelled round ms | ceiling | break-even accepted tokens | break-even rate |
|---:|---|---:|---:|---:|---:|
| 4 | ungated | 90.508 | 2.854 | 0.402 of 3 | 13.4 % |
| 5 | ungated | 107.924 | 2.988 | 0.673 of 4 | 16.8 % |
| 5 | gated | 107.793 | 2.990 | 0.672 of 4 | 16.8 % |
| 6 | gated | 138.826 | 2.845 | 1.109 of 5 | 22.2 % |
| 9 | ungated | 190.307 | 3.053 | 1.948 of 8 | 24.3 % |

The modelled ceiling spans 2.845 to 3.053, still a 7.3 % range. Replacing the
head lowers every break-even by 2.5 to 2.7 points and raises every ceiling by
about 0.14 to 0.20, which is a larger and far more certain gain than anything
available from changing draft depth.

Three caveats bound this section. These are GPU-time ratios on the decode
window only; the ranked leg also times the 512-token seed inside the same
measurement, and the census puts that seed at about 8.0 s in default packing.
The census legs force their draft count, so the acceptance rates above are the
requirement, not an observation. And the ranked host is M5, where the `qmv`
width curve may differ.


## Falsification riders

Every rider is evaluated on the **whole drafting round**, `draft_head` plus
`target_verify`, at width 6 on the gated isolated leg: 147.217 ms/round over 95
rounds. Evaluating a rider on `target_verify` alone is what produced the earlier
`gemv = 0.00 %` reading, because the proposal head is the only phase that
dispatches `gemv`.

| family | rider | ms/round | share of round | verdict |
|---|---|---:|---:|---|
| `copy` | ≤ 1 %, ~0.02 % expected; above 1 % reopens ledger 218 | 0.621 | 0.42 % | **PASS** |
| `elementwise` | `unary`/`binary`/`ternary_ops` < 3 % | 0.000 | 0.00 % | **PASS** |
| `norm` | `rms_norm` < 3 % | 0.186 | 0.13 % | **PASS** |
| `sdpa` | `sdpa_vector` < 3 % | 2.483 | 1.69 % | **PASS** |
| `gemv` | `gemv` < 2 % | 16.903 | 11.48 % | **FAIL** |
| `qmv` | five linear families dominate | 123.639 | 83.98 % | **PASS** |
| remainder | not a sixth linear family | 23.578 | 16.02 % | **PASS** |

**`copy` passes but at 21x the expected value.** The rider anticipated ~0.02 %
and the census measures 0.42 %. Ledger 218 stays closed on the stated
threshold, and this is in the headline because the gap between expectation and
measurement is large enough to matter if the threshold is ever tightened. At
width 1 the picture is different again: `copy` is 480.4 dispatches per round and
0.86 % of a 65.142 ms round. Copies are a width-1 phenomenon that the strided
fusions absorb as soon as M exceeds 1.

**`gemv` fails, and the failure is informative rather than a defect.** The
earlier `0.00 %` reading came from measuring `target_verify` only. The E58
dispatch census recorded `gemv.{h,metal}` at 3.02 % of in-round dispatches and
attributed it to the draft head; that was right about the location and, being a
dispatch count rather than a GPU-time share, understated the cost by almost
four times. `gemv` is 11.48 % of whole-round GPU time and 70.01 % of the
proposal head. **It stays on the live surface.**

The failure carries a provenance label. `gemv` is dispatched because the
resident head has no `.scales` tensors, so every head projection loads as a
plain `Linear`. That is true of the organizer-pinned bf16 head and false of the
declared 4-bit head. So the correct reading is: **`gemv` is live with certainty
on the local surface, and live on the ranked surface only if the ranked
candidate leg also resolves to a head with bf16 tensors.** The checked-in
manifest says it does not.

**The remainder is not a sixth linear family.** Outside `qmv`, the 23.578 ms
breaks down as `gemv` 16.903, `sdpa` 2.483, `gdn_recurrence` 2.021,
`steel_gemm` 0.828, `copy` 0.621, `compiled_fusion` 0.241, `norm` 0.186,
`draft_select` 0.167, `top2_readout` 0.062, `quant_dequant` 0.032, `gather`
0.028. `gemv` and `steel_gemm` are matrix multiplies, but both belong to the
proposal head, not to the target. Inside `target_verify` the remainder outside
`qmv` is 5.197 ms across nine families and contains no matrix multiply at all.

`unclassified_kernels: 0` in every phase of every leg. This is verified rather
than assumed: each observed kernel name is matched against the family table and
any miss is reported explicitly as `UNCLASSIFIED`.

## H-221 addendum — the surviving variant

The 0.35 ms per dispatch figure is formally withdrawn upstream, and this census
is a confirmation of that withdrawal rather than an independent challenge to it.
What follows is the surviving variant: **cost per host synchronisation point.**

### Requirement 1 — the closure gap, decode regime

`host_boundary_ms = wall − Σ GPU` per round, from the 512-token gated default
legs:

| width | wall ms/round | GPU ms/round | gap ms/round | GPU/wall | commits/round | MLX ops/round | ms per MLX op |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 66.776 | 64.522 | **2.254** | 0.9662 | 44.6 | 821.2 | 0.00274 |
| 6 | 148.560 | 145.383 | **3.177** | 0.9786 | 81.0 | 499.1 | 0.00637 |

A 64-token debug leg gave 2.456 and 3.990 for the same two widths, so the gap is
stable in shape across leg lengths. These are decode-regime measurements; the
caveat is stated in full at the end of this section.

### Requirement 2 — the packing bound, which fails

If the gap were paid per command-buffer boundary, adding boundaries would cost
wall time in proportion. It does not:

| width | commits/round default → isolated | wall ms/round default → isolated | ms per extra boundary |
|---:|---|---|---:|
| 1 | 44.6 → 1075.2 | 66.776 → 69.631 | **0.00277** |
| 5 | 71.0 → 778.2 | 115.790 → 115.418 | **−0.00053** |
| 6 | 81.0 → 695.9 | 148.560 → 150.999 | **0.00397** |

Width 5 adds 707 command buffers per round and finishes 0.37 ms **faster**. The
bound is at most 0.004 ms per boundary, about ninety times below 0.35, and the
sign is not even consistent. **A per-command-buffer-boundary cost of 0.35 ms is
dead, plainly.**

### The surviving variant — per host synchronisation point — is also dead

The remaining candidate was not a buffer boundary but a blocking host wait.
`CommandEncoder::synchronize()` at
`Vendor/mlx-swift/.../backend/metal/device.cpp:532-537` does
`end_encoding(); commit(); cbuf->waitUntilCompleted();`. That third call blocks
the host thread, and its count is structural: it depends on how many forced
evaluation points the session takes per round, not on how MLX packs buffers.

The census instrument was extended to measure it directly. A
`waitUntilCompleted` swizzle records `waits` and `wait_ns` per phase alongside
the existing `commits`/`commit_ns` and `dispatches`/`dispatch_ns` counters, so
the closure gap can be decomposed into named host costs instead of being left
as a residual.

**There are no host synchronisation points to price.** Measured across all
eleven timed legs, both packings, both gate sets and every width from 1 to 9:

| `waits` per round | legs and widths |
|---:|---|
| 0.00 | 18 of the 20 leg-and-width rows |
| 0.01 | `e80-hot-w4-default` at width 4; `e80-hot-w5-isolated` at width 5 |

Total blocked time is 0.000 ms per round everywhere, to three decimal places.

The hook is attached and working — it fires on two rows, roughly once per 100
rounds — so this is a measurement of near-zero, not a dead counter. MLX runs the
decode path asynchronously and essentially never blocks the host thread inside
a command buffer.

### The gap, fully decomposed

With `wait_ns` measured at zero, the closure gap resolves into the host time
spent encoding dispatches and committing buffers, both of which the census
records directly:

Default packing is what the candidate ships, so it carries the verdict. Every
drafting width from 1 to 9 is covered.

| leg | gate | width | wall ms | GPU ms | gap ms | dispatch ms | commit ms | wait ms | host sum | unexplained | disp/rd | commits/rd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `e80-census-w1-default` | gated | 1 | 66.970 | 64.527 | 2.443 | 1.203 | 0.608 | 0.000 | 1.811 | 0.632 | 1705.4 | 44.4 |
| `e80-census-w6-default` | gated | 1 | 66.776 | 64.522 | 2.254 | 1.127 | 0.603 | 0.000 | 1.730 | 0.524 | 1705.4 | 44.6 |
| `e80-census-w6-default` | gated | 6 | 148.560 | 145.383 | 3.177 | 1.575 | 1.227 | 0.000 | 2.802 | 0.375 | 1013.5 | 81.0 |
| `e80-hot-w4-default` | ungated | 1 | 67.129 | 64.661 | 2.467 | 1.253 | 0.663 | 0.000 | 1.915 | 0.552 | 1705.4 | 44.3 |
| `e80-hot-w4-default` | ungated | 4 | 96.914 | 94.185 | 2.730 | 1.319 | 1.512 | 0.000 | 2.831 | −0.101 | 856.0 | 67.1 |
| `e80-hot-w5-default` | ungated | 5 | 115.526 | 112.729 | 2.796 | 1.295 | 1.583 | 0.000 | 2.878 | −0.082 | 892.3 | 75.1 |
| `e80-hot-w9-default` | ungated | 9 | 204.851 | 201.893 | 2.958 | 1.573 | 1.802 | 0.000 | 3.375 | −0.417 | 1112.3 | 102.3 |

The whole gap is 2.25–3.18 ms per round across a 3.1× change in round cost.
Named host cost accounts for 74–114 % of it. The residual never exceeds
±0.632 ms per round and its sign is not consistent, which is what a small clock
bias between the host and GPU timebases looks like.

The per-unit costs follow directly:

| unit | width 1 | width 6 | width 9 |
|---|---:|---:|---:|
| host µs per dispatch | 1.127 / 1705.4 = **0.66** | 1.575 / 1013.5 = **1.55** | 1.573 / 1112.3 = **1.41** |
| host µs per commit | 0.603 / 44.6 = **13.5** | 1.227 / 81.0 = **15.1** | 1.802 / 102.3 = **17.6** |
| host µs per synchronisation point | **none exist** | **none exist** | **none exist** |

One instrument caveat belongs here, because the isolated legs look different and
the difference is not physical. In isolated packing `commit_ns` reaches 7.806 ms
per round at width 6 and 9.884 ms at width 9, and `unexplained` goes to −6.014
and −8.820 ms. Forcing one MLX op per command buffer raises commits from about
81 per round to about 700–1080, and the host `commit` call then runs
concurrently with GPU execution rather than adding to the round. `commit_ns`
double-counts that overlap. This is an artifact of the measuring mode. The
H-221 verdict rests on the default rows above, where commits are two orders of
magnitude rarer and the accounting closes.

**H-221 is closed in every form the evidence can address.** A per-dispatch host
cost of 0.35 ms is 220 to 530 times above the measured value. A
per-command-buffer cost of 0.35 ms is 23 times above the measured commit cost
and is separately falsified by the packing experiment, which added 700 buffers
per round at width 5 and got 0.37 ms *faster*. A per-synchronisation-point cost
cannot be priced because the decode path takes no synchronisation points.

These are decode-regime measurements and must not be extrapolated to prefill.
At the 512-row seed the model issues 512 rows per dispatch and is
throughput-bound; at decode width it is launch-bound. The seed is also the
boundary condition that kills the original figure independently: 0.35 ms per
dispatch would consume 793 ms of a 4046.5 ms prefill, and the GEMM roofline
leaves no room for it.

## Per-projection comparison with E71

E71 measured the width tax by nulling one projection arm at a time. This census
measures it by attributing command-buffer intervals. The two methods are
independent and mostly agree:

| projection | E71 arm ms | census ms | delta |
|---|---:|---:|---:|
| `mlp_gate_up` | 20.913 | 20.465 | −2.1 % |
| `mlp_down` | 16.887 | 16.87 | −0.1 % |
| `gdn_out_proj` | 3.313 | 4.33 | **+30.7 %** |
| `fa_o_proj` | 1.161 | 1.25 | +7.7 % |
| `lm_head` | 2.142 | 2.205 | +2.9 % |
| **sum** | **44.416** | **45.12** | **+1.6 %** |

Four of the five rows agree inside 8 % and the sum agrees inside 1.6 %.

**The `gdn_out_proj` row is left open.** The most likely explanation is that
`gdn_out_proj`, `mlp_down` and `fa_o_proj` all dispatch as `qmv Mx640x1` — one
grid signature, three different projections — so E71's arm and this census
divide the same pooled measurement differently. The census separates the pool
fit-free by buffer partner, which at M=1 is exact: the `swiglu` + `g1_copy`
partner gives 63.9 buffers per round at 218,439 ns (`mlp_down`), `v_copy
6144x1x1` gives 48.0 at 81,714 ns (`gdn_out_proj`), and `attn_gate 256x24x1`
gives 15.9 at 81,052 ns (`fa_o_proj`); those reconstruct 19.198 ms against a
measured 19.200 ms. At M=6 one partner group of 68.7 buffers per round mixes
33.3 `mlp_down` with 35.4 `gdn_out_proj` and has to be split by difference,
reconstructing 41.656 against a measured 41.831, a 0.4 % residual. That residual
is far smaller than the 30.7 % discrepancy, so the split is not obviously the
cause. **I do not have a measurement that settles it and I am not claiming
one.** Resolving it needs a hook on `setBuffer:offset:atIndex:` to identify the
weight tensor behind each dispatch directly.

## W&B evidence

One run per timed leg, all in
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group
`e80-per-kernel-gpu-time-census`. Every run carries
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` in its config, so the two gate sets cannot be pooled
by accident in the W&B UI either.

| leg | gate | widths | run id | URL |
|---|---|---|---|---|
| `e80-census-w6-default` | gated | 1, 6 | `azgwyge5` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/azgwyge5 |
| `e80-census-w6-isolated` | gated | 1, 6 | `y0gdzebh` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/y0gdzebh |
| `e80-census-w5-isolated` | gated | 1, 5 | `xe118mah` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/xe118mah |
| `e80-census-w1-default` | gated | 1 | `gcdjmemd` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/gcdjmemd |
| `e80-census-w1-isolated` | gated | 1 | `cowjscbn` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cowjscbn |
| `e80-hot-w4-default` | ungated | 1, 4 | `jok8s3qh` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/jok8s3qh |
| `e80-hot-w4-isolated` | ungated | 1, 4 | `nr4q5wpn` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/nr4q5wpn |
| `e80-hot-w5-default` | ungated | 1, 5 | `d0kmcp1q` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/d0kmcp1q |
| `e80-hot-w5-isolated` | ungated | 1, 5 | `kbgk2bs9` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/kbgk2bs9 |
| `e80-hot-w9-default` | ungated | 1, 9 | `ul97w9b4` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ul97w9b4 |
| `e80-hot-w9-isolated` | ungated | 1, 9 | `xav95tv4` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/xav95tv4 |

Each run logs per-family GPU ms per round and family share, per-width verify,
draft-head and whole-round cost, phase totals, the host closure gap, the
instrument health counters, a `census` table of every owning dispatch, and an
artifact holding the full derived `census.json` plus the leg's verbatim
`meta.txt`.

The rung 1 gate sessions are separate and were reported earlier: `ws1e4j5m`
(`e80-rung1-gate`) and `blld7vtb` (`e80-rung1-control`).

### Reproduction

```bash
# sweep 1, gated, from commit 773dff4
research/e80_census_session.sh e80-census 6,5,1,4,9 512

# sweep 2, ungated ABBA, from commit cdf33bb
research/e80_census_session.sh e80-hot \
  4:default,4:isolated,9:isolated,9:default,5:default,5:isolated 512 1

# analysis, one command per gate set, never pooled
research/e80_blocks.py \
  --isolated research/out/e80-census-w{6,5,1}-isolated/census.jsonl \
  --default  research/out/e80-census-w{6,1}-default/census.jsonl \
  --json research/e80-artifacts/gputime-census-gated.json
research/e80_blocks.py \
  --isolated research/out/e80-hot-w{4,5,9}-isolated/census.jsonl \
  --default  research/out/e80-hot-w{4,5,9}-default/census.jsonl \
  --json research/e80-artifacts/gputime-census-ungated.json

research/e80_head_scaling.py \
  --gated   research/out/e80-census-w{6,5}-isolated \
  --ungated research/out/e80-hot-w{4,5,9}-isolated
research/e80_schedule_economics.py \
  --gated   research/out/e80-census-w{5,6}-isolated \
  --ungated research/out/e80-hot-w{4,5,9}-isolated
research/e80_host_gap.py research/out/e80-{census,hot}-w*
research/e80_wandb_log.py research/out/e80-{census,hot}-w*
```

The instrument itself is `research/e80-artifacts/gputime-census.patch`. Apply it
with `git apply` on top of the base to reproduce a census leg; it is not
committed into `Sources/`.


## Suggested follow-ups

These are proposals. None is implemented here.

### 1. Quantize or shrink the proposal head — largest measured headroom

The head costs a flat **4.810 ms per draft token** with no fixed component, and
it is 15.1 % to 18.8 % of whole-round GPU time across widths 4 to 9. It is
purely bandwidth-bound, streaming 849 MB of bf16 weights per draft token at
245 GB/s. Two distinct questions follow, and they are worth separating:

- **Confirm which head the ranked candidate leg actually loads.** The census
  gives a free provenance check. MLX picks `Linear` or `QuantizedLinear` at load
  time on the presence of a `.scales` tensor (`Load.swift:250-258`), and the two
  dispatch different kernels. `gemv` in the head phase means the bf16 pinned
  head is resident; `qmv` means a quantized head is. Any future ranked-side
  trace answers this at zero extra GPU time.
- **Measure the declared head locally.** `setup-qwen-mtp.sh:66` honours
  `MLXFAST_QWEN_MTP_HEAD_REPO`, so provisioning the declared head needs a byte
  manifest and nothing else. Until that is done, every local `--local-iterate`
  and `--local-submit` ratio carries a systematic bias in one fixed direction:
  the head cost lands entirely on the MTP leg because the serial leg does not
  draft, so it cannot cancel in the ratio. The bias is **5.1 % to 6.7 % of
  whole-round GPU time** under the narrow bandwidth model, and local is always
  the pessimistic side.

Beyond swapping artifacts, the head is a genuine optimization target in its own
right. It is one full-width transformer layer with the target's own dimensions,
run once per draft token, purely to propose. The measured linearity is what
makes this tractable: because the phase has no fixed cost, any byte removed
from the head weights is repaid on **every** draft token at every schedule
depth. Lower precision, a narrower intermediate, fewer heads, or reusing target
state instead of recomputing all convert directly into score at the measured
245 GB/s.

The same linearity carries a warning for schedule design. Deeper drafting buys
accepted tokens but pays 4.810 ms of head time per extra draft token
unconditionally, before the target verifies anything. At width 9 the head alone
is 38.4 ms of a 203.9 ms round.

### 2. Attack `gdn_in_proj_fused` and `fa_qkv_gate_fused`

Together these are **10.163 ms/round, 17.75 % of the width tax**, and they are
the larger part of the previously unattributed 22.6 %. They are candidate
reachable: both are raw `quantizedMM` calls inside `Qwen35GatedDeltaNet` and
`Qwen35Attention` (`Qwen35.swift:677-689`, `:1707-1712`). No E71 arm could ever
reach them because they never dispatch through a child `Linear`, which is
exactly why they stayed unnamed for so long. This is also the same object the
prefill work found unfused at S = 512 behind the `S <= 9` gate at
`Qwen35.swift:1003`, so a change here has two independent reasons to matter.

### 3. Re-measure the qmv table on the reverted organizer table

This census ran on a base whose QMV template table carries `<T,5,5>`, `<T,6,6>`,
`<T,9,5>` with `NA<=6`. The advisor base at `6acb0d15` reverted `quantized.h`
and `mlx-generated/quantized.cpp` to the organizer table, which carries
`<T,5,3>`, `<T,6,3>`, `<T,9,3>` with `NA<=4`. Widths 1 and 4 resolve to the same
specialization under both tables; widths 5, 6 and 9 do not. **I have not
measured which table is faster at those widths, so I do not claim a direction
for the bias — only that one exists.** Since `qmv` is 95.8 % of verify GPU time,
the whole per-width curve should be replayed on the current base before it is
used to choose a draft schedule.

### 4. Fix the cool gate, or the campaign loses widths

Five of ten legs in the gated sweep produced no drafting rounds because the host
floor sat at 40.1–42.0 °C against a 40.0 °C target. The ungated authorisation
rescued this experiment, but an unreachable gate silently biases which widths
get measured toward whichever ones happen to run when the room is cool. The
gated legs that did pass entered at 43.0–49.9 °C, so the gate is not delivering
thermal tightness even when it passes. A gate specified against a measured host
floor, rather than a fixed 40.0 °C, would be both reachable and more honest.
