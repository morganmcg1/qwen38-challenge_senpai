# E16 — Prefill ladder adjudication

- Assignment: `qwen38-r1-e16-prefill-ladder-adjudication` (r1), PR #18
- Student: `qwen-alphonse`
- Base: `senpai/qwen38-mtp-r1` @ `e6e6f81767e84cc8c39b48c09a4f5cac597cdbca`
  (rebased mid-experiment from `e13a6fe0fd62a90d5042860dd01b03b7dfa8bcc4`; PR #13's
  per-depth draft-cost curve merged in between. Q1–Q4 were measured on `e13a6fe`,
  the prefill interval is re-confirmed on `e6e6f81` in Q5, and every ranked
  conversion below uses the post-merge decode window.)
- Host: AWS Mac, Apple M4 Pro (20 GPU / 14 CPU cores), `hw.memsize = 51539607552`,
  macOS 26.5.2 (25F84), Swift 6.3.3, automatic low-memory profile.
  **Not the ranked M5**: every absolute number here is directional.
- W&B group: `qwen38-r1-e16-prefill-ladder-adjudication`
  (project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`)

## Headline

**My merged E12 claim that the 512-row seed prefill is 73.8 % CPU graph
construction and 26.2 % GPU execution is retracted.** The advisor's ceiling
arithmetic was right and my attribution was wrong. With the `asyncEval` rung
ladder removed, the *same total* prefill wall moves wholesale from
`build_us` into `eval_wall_us`:

| arm | rungs | `build_us` (serial / MTP) | `eval_wall_us` (serial / MTP) |
|---|---:|---:|---:|
| `q1on` compiled default, `env=""` | 22 | 2 957 503 / 2 954 419 | 1 046 892 / 1 046 825 |
| `q1off` `DARKBLOOM_QWEN_PREFILL_LADDER=off` | 0 | **1 796 / 1 869** | **4 004 115 / 4 004 676** |
| `q1ctl` `DARKBLOOM_QWEN_PREFILL_LADDER=everyN:3` | 22 | 2 955 463 / 2 956 411 | 1 047 163 / 1 046 901 |

Real CPU graph-construction cost for the whole 64-layer prefill is
**1.8 ms, 0.045 % of `P`**. `begin` is **99.94 % GPU execution**. What E12
called "CPU build time" was the CPU blocking on MLX enqueue back-pressure
behind 22 already-dispatched `asyncEval` rungs — time the GPU was spending
anyway, merely charged to the wrong interval by the instrument.

The ceiling contradiction dissolves with the corrected attribution:
`24.9338 TFLOP / 4.006405 s = 6.2235 TFLOP/s`, i.e. **84.5 % of the measured
7.363 TFLOP/s dense-bf16 ceiling** and adjacent to E3's independent
6.415 TFLOP/s GEMM rate. There is no 3.23× impossibility left to explain.

## Q1 — ladder-on vs ladder-off, both arms

Run: `research/e12-run.sh ladder-sweep 64 1 q1on:default q1off:off q1ctl:everyN:3`
(one build, three arms, six timed phases, 942.8 s, exit 0).

Every timed leg is a 512-token seed followed by a **64-token decode window**;
all ratios below are 64-token-window ratios and are not ranked-equivalent.

| arm | phase | `seed_prefill_seconds` | `decode_seconds` |
|---|---|---:|---:|
| q1on | serial depth-0 | 4.004903913 | 8.211409926 |
| q1on | MTP depth-8 | 4.001740098 | 5.620792031 |
| q1off | serial depth-0 | 4.006404996 | 8.244696021 |
| q1off | MTP depth-8 | 4.007063985 | 5.616096020 |
| q1ctl | serial depth-0 | 4.003096104 | 8.213860035 |
| q1ctl | MTP depth-8 | 4.003799915 | 5.622963905 |

`build_us + eval_wall_us` reconciles with `seed_prefill_seconds` to within
0.5 ms in all six phases, so the two intervals partition `P` exactly and the
migration above is a re-attribution, not a measurement change.

### Same-build noise band

`q1on` and `q1ctl` run the *identical* 22-rung schedule in separate processes
(the trace proves `everyN:3` ≡ compiled default), so their difference is the
noise floor of this instrument:

- `seed_prefill_seconds`: 0.001808 s serial, 0.002060 s MTP
- `mtp_decode_speedup`: 0.000129 (0.0088 %)

### Headline delta

`P_ladder_off − P_ladder_on`, against the two-arm ON mean
(serial 4.004000009, MTP 4.002770007):

- serial **+0.002405 s = +0.060 % of `P`**
- MTP **+0.004294 s = +0.107 % of `P`**

Raw single-arm form (`q1on` vs `q1off`): +0.001501 s serial, +0.005324 s MTP.

The shipped 22-rung ladder is therefore worth **0.06–0.11 % of `P`**, which is
**14–25× below** the assignment's 1.5 %-of-`P` materiality bar (0.06 s) and
roughly one noise band wide. It is not a speedup mechanism; it is a
bookkeeping artifact that moved GPU wall time between two counters.

### Verdict on the 73.8 / 26.2 split

**Retracted.** Corrected: `begin` is 0.045 % CPU graph construction and
99.94 % GPU execution. A correction note has been appended to
`research/e12-r1-seed-prefill-charge-report.md` (commit `f426c16`) which
retracts the split and withdraws the "attack the CPU three-quarters"
next-action while leaving E12's raw measurements, `p(512)` and the area sizing
untouched.

### Scored predictions

| prediction | outcome |
|---|---|
| advisor 1: ladder-off `eval_wall_us` ≥ 3.3 s | **✓** 4.004 s |
| advisor 1: ladder-on `build_us` ≤ 1.0 s | **✗** 2.958 s — enqueue back-pressure is charged to the build interval |
| advisor 2: shipped ladder worth < 3 % of `P` | **✓** 0.06–0.11 % |
| advisor 6: at least one of the five is wrong | **✓** (its own half of 1) |

### Caveat the ratio raises

The `q1off` arm's **serial decode** is 0.405 % slower (8.244696 vs 8.211410 s)
than the ON mean — 13.6× the same-build decode noise (0.0298 %) — even though
the ladder is prefill-only and decode uses the untouched
`{0,1,9,19,29,39,49,57}` rung set. All six timed phases entered the cool gate
at 39.6–40.0 °C, so temperature does not explain it. It inflates the off-arm
`mtp_decode_speedup` to 1.4680 vs 1.4608/1.4608 for the two ON arms. I am
reporting this as unexplained rather than absorbing it. It is a further reason
`mtp_decode_speedup` is the wrong instrument for a prefill question and that
judging on `seed_prefill_seconds` is correct.

## Q4 — is any interior rung schedule materially better than shipped?

Run: `research/e12-run.sh ladder-sweep 64 1 q4a:list:0,1,2,5,11,23,47 q4b:everyN:12`
(one build, two arms, four timed phases, 634.3 s, exit 0, HEAD `b82a51e` ≡ `4864bc5`
sources). Identical binaries to Q1 (worker `42c72d09…`, CLI `0a904c0d…`), so the
arms are directly comparable across the two runs.

| arm | spec | rungs | at | `build_us` s/M | `eval_wall_us` s/M |
|---|---|---:|---|---:|---:|
| q4a | `list:0,1,2,5,11,23,47` | 7 | 0,1,2,5,11,23,47 | 2 205 716 / 2 206 181 | 1 787 581 / 1 787 376 |
| q4b | `everyN:12` | 6 | 0,11,23,35,47,59 | 2 953 026 / 2 953 384 | 1 041 079 / 1 040 912 |

| arm | phase | `seed_prefill_seconds` | vs ON mean | `decode_seconds` |
|---|---|---:|---:|---:|
| q4a | serial | 3.993803024291992 | −0.010196985 s (−0.2547 %) | 8.242900967597961 |
| q4a | MTP | 3.994047999382019 | −0.008722008 s (−0.2179 %) | 5.608191013336182 |
| q4b | serial | 3.994601011276245 | −0.009398998 s (−0.2347 %) | 8.206997990608215 |
| q4b | MTP | 3.994801998138428 | −0.007968009 s (−0.1991 %) | 5.618273973464966 |

**Best interior schedule = 0.2547 % of `P`.** That is ~5.6× the same-build noise
band, so it is probably a real ordering effect, but it is **5.9× below** the
1.5 %-of-`P` materiality bar. No schedule in `{0, 6, 7, 22}` rungs is material.

### The total is rung-invariant; only the split moves

| rungs | last rung | `build_us` + `eval_wall_us` (serial) |
|---:|---:|---:|
| 0 (`off`) | — | 4.005911 |
| 6 (`everyN:12`) | 59 | 3.994105 |
| 7 (`list:…,47`) | 47 | 3.993297 |
| 22 (default) | 57 | 4.004395 |
| 22 (`everyN:3`) | 57 | 4.002626 |

Full spread over a 0→22 rung sweep is **0.0126 s = 0.31 % of `P`**; among the
rung-bearing arms it is 0.0033 s. The split, by contrast, ranges from
`0.0018 / 4.0041` to `2.9530 / 1.0411`. Note that `q4b` has **fewer** rungs than
the default yet nearly the same split, while `q4a` with **more** rungs has a
completely different one: the split is set by *where the last rung sits*
(4 layers of tail for `q4b`, 16 for `q4a`), not by rung count. `build_us` is
measuring enqueue back-pressure, exactly as Q1 concluded. This is the third
independent confirmation of the Q1 verdict.

## Q2 — is E3's 6.415 TFLOP/s a real kernel rate or a harness artifact?

Run: `research/prefill_floor.py --reps 15 --chain 8 --pipeline-reps 5
--measured-prefill-seconds 4.004000009 --out research/floor-e16.json`
(37.9 s, exit 0; an independent smoke run reproduced every headline within 0.3 %).
Standalone MLX Python (mlx 0.29.3), same host, real 4-bit group-64 weights at the
scored geometry.

**Verdict: real.** The per-op harness is *not* materially over-charging.

| mode | prefill seconds | vs worker `P` |
|---|---:|---:|
| worker-measured `P` | 4.004000009 | 1.000× |
| per-op synced sum | 4.100217625000054 | 1.024× |
| **fully pipelined single `eval`** | **4.011978** (build 0.001371 + eval 4.010607) | **1.002×** |
| chained ×8, no intermediate sync | 4.011978 | 0.978× of synced |

The pipelined graph is the same shape the worker actually runs, and it reproduces
the worker wall to **0.2 %** — implying 6.2178 TFLOP/s against the worker's
6.2235 TFLOP/s, **0.09 % apart**. Advisor prediction 3 (the per-op harness
overcharges by little) is **confirmed**.

Where per-op charging *is* wrong, it is only on tiny elementwise ops. Normalized
chain scaling (RAW `chain_scaling` ÷ 8): `residual_add` 0.263, `final_norm` 0.288,
`rms_norm` 0.298, `conv1d_depthwise_k4` 0.563 — i.e. these run ~3.4× cheaper when
pipelined. Every GEMM sits at 0.961–0.992. Total elementwise overcharge is
**0.0654 s ≈ 1.6 % of `P`**, which is why the synced sum is only 2.4 % high.

The GEMM-rate uniformity E3 reported is real and *tightens* when pipelined: the
2.7 % spread (6.295–6.466 TFLOP/s) becomes 1.9 % (6.428–6.551). The smallest
GEMMs gain the most (`out_proj`/`o_proj` 6.295/6.297 → 6.551), so ~0.4 % of `P` is
per-dispatch overhead on small projections rather than arithmetic.

## Q3 — the closing prefill budget

`closure_error_seconds = 0.0` exactly, by construction of the signed identity
`P = gemm_at_ceiling + nongemm + dequant_overhead − overlap_credit`:

| term | seconds | % of `P` |
|---|---:|---:|
| measured `P` | 4.004000009 | 100.000 |
| `gemm_at_ceiling` (all GEMMs at 7.4014 TFLOP/s) | 3.3693021188662633 | 84.148 |
| `nongemm` (kernels with no FLOP account) | 0.21271352600006566 | 5.313 |
| **`floor_subtotal`** | **3.582015644866329** | **89.461** |
| `residual` | 0.42198436413367135 | 10.539 |
| ↳ `dequant_overhead` | 0.5182019801337252 | 12.942 |
| ↳ `overlap_credit` | −0.09621761600005385 | −2.403 |

Supporting measured quantities: `gemm_seconds_measured` 3.8875040989999885,
`gemm_tflop_total` 24.93751230464, `gemm_tflops_achieved` 6.414787398180458,
`ceiling_tflops` 7.401388009998707 (dense bf16 512×5120×17408),
`gemm_fraction_of_ceiling` 0.8667005958226446.

**The residual is fully named.** It is not unexplained overhead: it is the cost of
dequantizing affine 4-bit group-64 weights inside the matmul (12.94 % of `P`),
partly repaid by cross-op overlap the per-op model cannot see (2.40 %). Four
kernel families E3 never charged — `rms_norm` ×128, `residual_add` ×128,
`conv_silu_gate_norm` ×48, `qk_norm_rope_gate` ×16 — add 0.0785 s (1.96 % of `P`)
and close most of E3's former gap.

### Ranked value, on the post-merge decode window

Every ratio in this report is measured on a **64-token decode window**; the
ranked window is 512 tokens. Converting to ranked points uses the post-merge
`p(512) = P / D_mtp(512) = 4.003337 / 12.049719 = 0.3322361` (prefill = 24.938 %
of the MTP leg) and `gain = p · φ · (R − 1)`. I use the advisor's projected
post-merge `R = 3.0972967` (solved from their own prize table; the promoted
on-record `R` is 2.94661597308114, which would make every row below 7.19 %
smaller). One frontier step = 0.0122890 points.

| φ (removable fraction of `P`) | source | points | frontier steps |
|---|---|---:|---:|
| 0.00060 – 0.00107 | **the shipped ladder (Q1)** | 0.000418 – 0.000746 | 0.03 – 0.06 |
| 0.002547 | **best interior schedule (Q4)** | 0.001775 | 0.14 |
| 0.015 | assignment materiality bar | 0.010452 | 0.85 |
| 0.10539 | **Q3 `residual`**: every GEMM at the dense-bf16 ceiling | 0.073436 | 5.98 |
| 0.12942 | Q3 `dequant_overhead` alone, no overlap credit | 0.090180 | 7.34 |

1 % of `P` removed = **0.006968 points = 0.567 frontier steps** on this base
(0.006054860 pre-merge). So the whole scheduling question I was assigned is worth
**0.03–0.14 frontier steps**, while the kernel-efficiency residual it rules out
is worth **~6 steps**. My Q3 envelope (0.1054) is *below* the advisor's
0.1552 GPU-floor-slack row because my directly measured ceiling is 7.4014 rather
than 7.363 TFLOP/s and because the four newly charged elementwise families
(1.96 % of `P`) are real work that no kernel rewrite removes.

## Q5 — post-merge confirmation on `e6e6f81`

PR #13's per-depth draft-cost curve is a decode-path policy change inside
`Qwen36MTPBlockSession`; it does not touch `begin`. The rebuild confirms the blast
radius exactly: the CLI digest is **unchanged** (`0a904c0d…`), only the worker
moved (`42c72d09…` → `3670d6f7…`).

<!-- Q5 arm table appended after the confirmation run -->

## Verdict

**Not useful (publishable negative). Close the enqueue-scheduling area.**

Stop rules, both fired:

- (a) *If ladder-off moves `P` by less than 1.5 %, the CPU-build premise is
  refuted and Phase 2/3 are cancelled.* Fired: 0.060 % / 0.107 %.
- (b) *If no rung schedule beats shipped by more than 1.5 % of `P`, stop.* Fired:
  best of four alternative schedules is 0.2547 %.

Consequently Phase 2 and Phase 3 are cancelled and **I am handing timing slot 2
back to the pool unused.** There is nothing left in enqueue ordering to time:
prefill is ~100 % GPU-resident, the CPU contributes 1.8 ms, a fully pipelined
graph already matches the worker wall to 0.2 %, and the wall is invariant across
0/6/7/22 rungs.

## Where the prefill area actually goes next

The area is *not* dead — only my lever is. Prefill is 24.938 % of the ranked MTP
leg and runs at 86.67 % of the measured dense-bf16 ceiling. The named residual is
**quantized-matmul dequantization**, so the follow-up belongs to the
quantized-kernel owner (`Sources/…/quantized.h` and its generated twin, currently
`qwen-thorfinn`'s surface), not to a scheduling or graph-construction experiment.
Worth ~6 frontier steps if 4-bit group-64 matmul can be brought to the dense rate;
any fraction of that is proportional.

## Open follow-ups I did *not* implement

1. **A ~0.4 % serial-decode band that tracks the prefill split.** Serial
   `decode_seconds` is 8.2070–8.2139 s in the three arms whose prefill tail
   finished inside `build_us` (`eval_wall_us` ≈ 1.04 s) and 8.2429–8.2447 s in the
   two arms with a large trailing `eval` (1.79 s and 4.00 s) — +0.402 %, 13.6× the
   same-build decode noise, with all ten timed phases entering the cool gate at
   39.3–40.0 °C. The MTP leg moves the *opposite* way (−0.152 %), so this is not
   one clean mechanism and I am not claiming allocator state as the cause. It is
   worth 0.4 % of the *decode* leg, i.e. more than everything Q1/Q4 measured, and
   it is cheap to chase: sweep last-rung position with decode fixed.
2. **Per-dispatch overhead on the small projections** (`out_proj`, `o_proj` gain
   4.0 % when pipelined) — ~0.4 % of `P`, a fusion question, still below the bar
   alone but additive with the dequant work.
3. **`DARKBLOOM_QWEN_PREFILL_LADDER` is a 2 706-byte diagnostic knob on a
   submitted path** (`Vendor/…/Qwen35.swift`). It defaults to the shipped ladder
   when unset, so it is score-neutral, and it is the only submitted-path change in
   this branch. Because the verdict is negative, the advisor should decide
   **keep as a documented diagnostic or revert before any official submission**;
   I have not reverted it unilaterally since it is the instrument that produced
   the retraction and it makes the result reproducible.
4. **The fixed-window EOS defect** (diagnosis below) is a correctness bug in the
   scored session that I found while reading `begin`; it needs its own assignment.

## The EOS / fixed-window defect (read-only diagnosis, not fixed)

Line numbers on the post-rebase base `e6e6f81`:

- **`Qwen36MTPBlockSession.swift:778-795`** — on a stop-token *primary* the session
  clears `pendingPrimary`/`pendingTop2`/`pendingHidden` and returns
  `declaredRows: 1`. The next `generateRound` then fails the `guard` at the top of
  the round and throws `.notBegun` — *"MTP round requested before the seed
  prefill"*. **This is the token-#301 failure.** The error message is misleading:
  the session did begin, it deleted its own pending state.
- **`Qwen36MTPBlockSession.swift:1168-1184`** — truncates `committed` at the first
  stop token and decrements `committedTokenCount` without trimming the target
  cache or the pendings, breaking the round-top invariant
  `trimmableOffset() == seedTokenCount + committedTokenCount` and throwing
  `.cacheOffsetInvariant`. Loud failure, not silent divergence.
- `Qwen36MTPReferenceSession.swift` has **no** stop-token logic at all, which is
  the correct reading of the fixed-window contract: the parent owns the window and
  the session keeps decoding past EOS.

Fix shape: **delete both special cases** and reduce `reachedStopToken` to
telemetry. Validation needs a ≥320-token decode window (~10 min) to cross the
failure point, plus exact post-EOS token and row-ledger checks against the public
golden. I did not implement it: it is outside this assignment's six forbidden
items and a half-fix here would be worse than a clean assignment.

## Reproduction

```bash
research/e12-run.sh build
research/e12-run.sh ladder-sweep 64 1 q1on:default q1off:off q1ctl:everyN:3
research/e12-run.sh ladder-sweep 64 1 q4a:list:0,1,2,5,11,23,47 q4b:everyN:12
research/e12-run.sh ladder-sweep 64 1 q5post:default
"${MLX_VENV}"/bin/python3 research/prefill_floor.py --reps 15 --chain 8 \
  --pipeline-reps 5 --measured-prefill-seconds 4.004000009 \
  --out research/floor-e16.json
research/e16_wandb.py --group qwen38-r1-e16-prefill-ladder-adjudication \
  --log RUNLOG [--log RUNLOG …] --floor research/floor-e16.json
```

`DARKBLOOM_QWEN_PREFILL_LADDER` accepts `off`, `everyN:<n>`, `list:<i,j,k…>`, or
nothing (compiled default). It is read once in `Qwen35.swift` when the prefill
ladder is built; unset ⇒ byte-identical behaviour to the shipped build.
