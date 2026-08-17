# E16 — Prefill ladder adjudication

- Assignment: `qwen38-r1-e16-prefill-ladder-adjudication` (r1), PR #18
- Student: `qwen-alphonse`
- Base: `senpai/qwen38-mtp-r1` @ `e13a6fe0fd62a90d5042860dd01b03b7dfa8bcc4`
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

<!-- Q2/Q3/Q4 sections appended after their runs -->
