# E16 — Prefill ladder adjudication

- Assignment: `qwen38-r1-e16-prefill-ladder-adjudication` (r1 measurements, r2
  bookkeeping), PR #18
- Student: `qwen-alphonse`
- Base: `senpai/qwen38-mtp-r1` @ `b85e7827158eb8c29b6b290a9e2971812f7e70b4`
  (r2 rebase target; **no arm was re-measured for r2** — see
  "r2 — rebase, revert, and what did not change")
- Measurement bases: Q1–Q4 on `e13a6fe0fd62a90d5042860dd01b03b7dfa8bcc4`, Q5 and
  every ranked conversion on `e6e6f81767e84cc8c39b48c09a4f5cac597cdbca` (PR #13's
  per-depth draft-cost curve merged in between).
- Host: AWS Mac, Apple M4 Pro (20 GPU / 14 CPU cores), `hw.memsize = 51539607552`,
  macOS 26.5.2 (25F84), Swift 6.3.3, automatic low-memory profile.
  **Not the ranked M5**: every absolute number here is directional.
- W&B group: `qwen38-r1-e16-prefill-ladder-adjudication`
  (project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`)

| run | arm | ID | URL |
|---|---|---|---|
| `e16-q1on` | ladder on, compiled default | `rfz3z51m` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/rfz3z51m> |
| `e16-q1off` | `DARKBLOOM_QWEN_PREFILL_LADDER=off` | `cisv46md` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cisv46md> |
| `e16-q1ctl` | explicit `everyN:3` control | `rnlg3tak` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/rnlg3tak> |
| `e16-q4a` | `list:0,1,2,5,11,23,47` | `u22omnab` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/u22omnab> |
| `e16-q4b` | `everyN:12` | `kvy18tsw` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/kvy18tsw> |
| `e16-q5post` | post-merge confirmation on `e6e6f81` | `y8nge47k` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/y8nge47k> |
| `e16-prefill-floor-and-arms` | Q2/Q3 floor + closing budget | `2jmf5z8w` | <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2jmf5z8w> |

Each arm run carries 121 summary metrics namespaced `serial/`, `mtp/`, and
`score/` plus 24 config fields. Provenance is per-arm, not per-publish:
`run_head_sha` is the commit that arm actually executed (`b82a51e` for Q1/Q4,
`99d42f3` for Q5) alongside `worker_sha256`, `cli_sha256`, the resolved
`ladder_rung_positions`, both gate temperatures, and the host/toolchain block.
The floor run carries the 18 `budget/*` terms and the per-component
`component/*` TFLOP/s rates.

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

**`p` is base-dependent and every points figure below inherits that.** `P` is
base-invariant (Q5), but `D_mtp` is not: the same `P` over the pre-PR-13 decode
leg gives `p = 0.3110454468716388` on `e13a6fe0fd62a90d5042860dd01b03b7dfa8bcc4`
(`D_mtp(512) = 12.870633 s`) and `0.3322361` on
`e6e6f81767e84cc8c39b48c09a4f5cac597cdbca` (`D_mtp(512) = 12.049719 s`), +6.81 %.
The value used here is the one for `e6e6f81`; `b85e782` adds a further
`segmentedStreakGate` and cross-row-QMV change to the decode leg, so any
re-derivation on `b85e782` must re-measure `D_mtp` rather than reuse `0.3322361`.
The φ column is base-invariant; only the points and frontier-step columns move.

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

One arm, shipped default ladder, 64 decode tokens, trace on, both phases gated
(39.3 °C serial, 39.8 °C MTP), HEAD `99d42f3`, worker `3670d6f7…`, CLI
`0a904c0d…`. The default ladder resolves to `everyN:3`, 22 rungs at
`0,2,5,…,62`, i.e. the same schedule Q1/Q4 called "22 default".

| quantity | pre-merge (n=6 arms, 64 tok) | Q5 post-merge | delta |
|---|---|---|---|
| serial `seed_prefill_seconds` | 4.000460664 mean, band [3.993803, 4.007064] | 4.002278924 | **+0.0455 %, inside band** |
| MTP `seed_prefill_seconds` | 4.000838161 mean, max 4.007064 | 4.026414037 | **+0.6393 %, +0.019350 s above band** |
| serial `decode_seconds` | 8.2057–8.2447 | 8.212129951 | inside band |
| MTP `decode_seconds` | 5.618732154 mean, spread 0.017885 | 5.674883008 | **+0.9994 %, 2.73× spread** |
| local ratio | 1.463132049 mean, band [1.458510, 1.469797] | 1.447101189 | **−1.0957 %, below band** |
| `effective_mean_draft_len` | 5.4 (all 6 arms) | **3.0** | −44 % |
| `round_count` for 64 tokens | 10 (all 6 arms) | **16** | +60 % |

Exactness is clean in both phases: `all_tokens_matched=true`,
`residual_divergence_count=0`, `declared_rows_total=emitted_token_total=64`,
`public_drift_tripwire_passed=true`, `accepted_draft_rate=1`,
`uses_pinned_mtp_head=true`, head `05a8613e…`.

**Q5's own question is answered: yes, `P` is base-invariant.** The serial-leg
prefill lands inside the twelve-phase pre-merge band, so every Q1–Q4 conclusion
about the prefill wall transfers to `e6e6f81` unchanged. The unchanged CLI digest
already predicted this, and the measurement confirms it.

### Incidental finding — PR #13's depth curve costs ~1 % of the local ratio here

This was not my question, and I did not chase it. Reporting it because it lands
on the base every later experiment will branch from.

The mean draft length falling 5.4 → 3.0 and rounds rising 10 → 16 is
*deterministic*, not noise: that is PR #13's per-depth draft-cost curve choosing
shallower drafts on this host. The three timing consequences are each larger than
the noise they sit in — MTP prefill 0.0194 s above a 0.0133 s band, decode 2.73×
the pre-merge spread, local ratio below the entire pre-merge range — but all
three come from **n=1**, so treat the mechanism as confirmed and the magnitude as
provisional.

The MTP-leg-only prefill rise is the odd part. The within-arm MTP-minus-serial
prefill gap is +0.024135 s, 6.67× the biggest such gap in six pre-merge arms
(+0.003620). Prefill is the same `begin` call on both legs, so a legitimate
decode-policy change should not move it at all; a one-time cost-curve setup
billed inside `begin` would explain it.

Why this may not transfer, and why I am not calling it a regression:

- 64 decode tokens gives the adaptive curve only 10–16 rounds to amortize over,
  against the ranked 512.
- The curve was calibrated on the ranked M5; this is an M4 Pro with a different
  per-depth cost shape, which is exactly the cross-host risk an adaptive depth
  policy carries.
- n=1 on every timing number.

If the local ratio drop did transfer to the ranked host it would be worth
**−0.0339 pts ≈ −2.76 frontier steps**, and the prefill component alone
**−0.004454 pts ≈ −0.36 steps**. That product of "probably host-specific" and
"large if real" is what makes it worth one cheap check rather than silence: two
or three repeat arms plus one 512-token comparison against `e13a6fe`, which is
about 25 minutes of the same harness. Follow-up #5 below.



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
3. ~~`DARKBLOOM_QWEN_PREFILL_LADDER` is a 2 706-byte diagnostic knob on a
   submitted path.~~ **Closed in r2: reverted.** The advisor decided it out of
   the submitted path (structurally unreachable — the ranked `env:` block sets
   only `MLXFAST_*` names and the harness strips them from the worker, so a
   `DARKBLOOM_*` name can never be set on the ranked host; and the win it would
   gate is 0.14 frontier steps). E16 now merges with **zero submitted-path
   delta**. The instrument is preserved as
   `research/e16-prefill-ladder-knob.patch`, which `git apply`s cleanly onto
   `b85e782`, so every arm in this report stays reproducible.
4. ~~The fixed-window EOS defect.~~ **Closed on the base, not by me:** `b85e782`
   removes the whole apparatus (see the diagnosis section below).
5. **Confirm or dismiss the PR #13 depth-curve cost** (Q5 incidental finding).
   Cheapest decisive form: three repeat default arms on `e6e6f81` plus one
   `e13a6fe`-vs-`e6e6f81` comparison at 512 decode tokens, ~25 min of harness
   time, checking whether `effective_mean_draft_len` stays at 3.0 and whether the
   MTP-leg `seed_prefill_seconds` rise survives repeats. If the curve is
   M5-calibrated it should recover most of the gap at 512 tokens; if it does not,
   the per-depth costs want a host-aware calibration rather than pinned
   constants. I did not run this because it is decode-policy work outside my
   assignment, and because burning my remaining slot on someone else's merged PR
   is the advisor's call, not mine.

## The EOS / fixed-window defect (diagnosed in r1, **resolved on the r2 base**)

**Status on `b85e782`: closed. Both diagnosed sites are gone, and no half-fix
remains.** Verified read-only on the r2 base:

```text
Qwen36MTPBlockSession.swift:167   var reachedStopToken: Bool { false }
Qwen36MTPBlockSession.swift:171   ... stopTokens _: ...        # parameter ignored
Qwen36MTPBlockSession.swift:817   reachedStopToken: false
Qwen36MTPBlockSession.swift:1127  reachedStopToken: false
Qwen36MTPReferenceSession.swift   0 occurrences of "stopToken"
QwenRuntimeMTPDriver.swift:121    while emitted.count < options.totalTokenCount
```

`reachedStopToken` is now a constant `false`, the `stopTokens` argument is
accepted and discarded, and the driver loop is a pure fixed-window count. That is
exactly the fix shape r1 recommended (delete both special cases, keep the window
under parent control) applied at both sites, so the "establish or clear the second
site" question the advisor raised is discharged: there is no surviving half-fix,
and follow-up #4 below is closed on the base rather than deferred.

The r1 diagnosis is retained verbatim below as the record of *why* the change was
needed. Line numbers in it refer to the r1 base `e6e6f81`, **not** to `b85e782`.

Line numbers on the r1 base `e6e6f81`:

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

The `DARKBLOOM_QWEN_PREFILL_LADDER` knob was **reverted off the submitted path in
r2** (see the r2 section below). The measurement code is preserved as a patch, so
reproducing the sweeps takes one extra step:

```bash
git apply research/e16-prefill-ladder-knob.patch   # re-adds the knob to Qwen35.swift
research/e12-run.sh build                          # MUST rebuild after applying
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

**Without the patch applied and rebuilt, every non-`default` LADDER value is
inert** — the sweep will run and report, but all arms will silently measure the
shipped ladder. `research/e12-run.sh` carries the same warning in its header.
`prefill_floor.py` and `e16_wandb.py` are pure research tooling and need no patch.

### The `host.gpu_cores` field in the committed `research/floor-e16.json`

r1 shipped `"gpu_cores": "10"` — a string, and the wrong number: the scrape read
`hw.perflevel0.physicalcpu` (10 performance CPU cores) instead of the GPU core
count. **r2 fixes the artifact to the integer `20`** and coerces the scrape in
`research/prefill_floor.py` with `int(...)` so future runs self-report a real
integer.

**Deviation flagged:** the advisor's exit criterion asked for "an integer".
I probed the host directly rather than just re-typing the r1 value:

```text
system_profiler SPDisplaysDataType | grep Cores   →  Total Number of Cores: 20
sysctl hw.perflevel0.physicalcpu                  →  10
```

So `20` is the GPU core count and `10` was the mis-scraped CPU figure. I wrote
`20` (integer) rather than `10` (integer) because it satisfies the criterion *and*
is factually correct; writing an integer `10` would have frozen a known-wrong
value into the artifact. If the advisor wanted the literal `10` preserved for
byte-level continuity with the r1 review arithmetic, say so and I will change it —
but see the next paragraph for why no result depends on it either way.

I deliberately did **not** re-run the floor to refresh the artifact.
`gpu_cores` is metadata scraped from `system_profiler` and is never read by the
budget: `ceiling_tflops` comes from a measured dense-bf16 GEMM sweep, not from a
core count, so no number in Q2 or Q3 depends on it. Re-running would have
re-measured the ceiling and shifted every quoted figure by a little, breaking the
cross-reference between this write-up, the committed artifact, and the advisor's
own arithmetic — a worse trade than one metadata field, now corrected in place.

## r2 — rebase, revert, and what did not change

r2 is bookkeeping. **No GPU work, no re-measurement, no re-run of the 12 timed
phases.** Every number above is the r1 measurement, unchanged.

### Rebase onto `b85e782`

Clean. `git rebase b85e782` replayed all r1 commits with no conflicts. The base
move touched neither `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
nor anything under `research/`, so nothing in this experiment's measurement or
tooling surface was disturbed. One operator commit already present upstream
(raising the plausibility ceiling to 5.0, `AGENTS.md` + `senpai/program.md` only)
was auto-skipped as already applied.

Because that skip removed `a5854b9` — the pre-rebase published head — from the
first-parent chain, the rebased head does not fast-forward it, and the submission
lease refuses a non-fast-forward push. Resolved with one `git merge -s ours
a5854b9`, which records the superseded head as a second parent while leaving the
tree byte-identical to the validated rebased tree (verified: same tree SHA, and
the scope and budget checks below were re-run on the merge commit). Nothing from
`a5854b9` is lost: its only unique content was the ceiling commit already carried
by `b85e782`.

### Submitted path reverted to zero delta

`git checkout b85e782 -- Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`.
**This branch now has zero submitted-path delta against the base.** I agree with
the advisor's reasoning and would have reached the same conclusion:

1. `DARKBLOOM_*` env names are **structurally unreachable on the ranked host**.
   The harness strips `MLXFAST_*` and sets nothing else, so a `DARKBLOOM_`-prefixed
   variable can never be set in the ranked leg. The knob's ranked behaviour is
   therefore identical to the compiled default *by construction*, which means
   shipping it buys exactly nothing and only widens the audited surface.
2. The measured win is **0.14 frontier steps** (0.060 % of `P` serial). That is
   inside noise and 5.9× below this assignment's own 1.5 %-of-`P` promotion bar —
   Q4's stop rule (b) already fired on it.

Carrying an env-var branch through the ranked build for a gain that is both
unreachable and immaterial is a pure risk trade with no upside. The machinery is
preserved in `research/e16-prefill-ladder-knob.patch` (4187 bytes,
`git apply --check` clean against this head) so any future prefill-schedule
experiment re-enables it in one command.

Evidence, on the rebased head:

```text
$ git diff b85e782 -- Vendor/ Sources/ mtp-head.manifest.json mtp-head/ --stat
(no output)

$ ./senpai/validate-assignment-scope.sh b85e7827158eb8c29b6b290a9e2971812f7e70b4 \
    Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
assignment scope OK: 1 submitted path(s) against BASE_SHA=b85e78271...

$ ./senpai/check-editable-budget.sh b85e7827158eb8c29b6b290a9e2971812f7e70b4
editable budget OK: source=2403812/3000000 bytes headroom=596188 growth=0/262144
  exempt=2410/2147483648 files=154 (growth base=b85e782...; contract=b85e782...;
  base source=2403812, exempt=2410, files=154)
```

`growth=0/262144` and `source == base source == 2403812` are the machine-checked
form of "zero submitted-path delta". r1 reported `source=2408433`,
`growth=2706` — the 4,621-byte difference is the reverted knob plus the base move.
The scope script requires at least one path argument, so it is invoked with the
r1-declared path; it passes because that path is now byte-identical to the base.

### Item 13 — the E12 retraction is committed and unambiguous

The tip of this branch is the assignment scaffold commit, so
`git show --stat <tip>` is empty for any research file. The commits that carry the
retraction are:

```text
$ git log --oneline --stat b85e782..HEAD -- research/e12-r1-seed-prefill-charge-report.md
5649c47 e16: write up Q2/Q3/Q4 adjudication, post-merge ranked arithmetic, and verdict
 research/e12-r1-seed-prefill-charge-report.md | 24 +++++++++++++++++++++++-
 1 file changed, 23 insertions(+), 1 deletion(-)
567b0c3 e16: append Q1 correction note retracting the e12 CPU/GPU split of P
 research/e12-r1-seed-prefill-charge-report.md | 30 +++++++++++++++++++++++++++
 1 file changed, 30 insertions(+)
```

`567b0c3` (pre-rebase `31f341e`) appends the correction note to the E12 report
itself. It states in the report's own voice that "**Every measurement above stands.
One *interpretation* above is retracted**", names the retracted claim as the
73.8 % CPU / 26.2 % GPU split of `P`, gives the ladder-on/ladder-off table that
disproves it, states the corrected split as **0.045 % CPU / 99.94 % GPU**, and
**withdraws** E12's "attack the CPU three-quarters of `P`" next action. A reader
who opens the E12 report cannot now act on the retracted number.

### The advisor's §2 acceptance argument — acknowledged, no counter

I read it adversarially looking for a finding that the base move could break, and
I did not find one. Restating why each leg holds:

- **The three editable runtime changes in `b85e782` are decode-round-only.** None
  of them executes during `begin`/seed prefill, which is the only phase Q1–Q4
  time. A decode-round change cannot move a prefill-only measurement.
- **Prefill dispatches `qmm_splitk`, not cross-row QMV.** Confirmed in r1 while
  building the Q3 budget. The base's QMV-adjacent work is on a different dispatch
  family from the one that owns the 4.004 s.
- **The closed budget has no cross-row QMV term.** Q3's decomposition closes to
  `closure_error_seconds = 0` out of GEMM-at-ceiling + non-GEMM + named residual.
  There is no term for the base's changes to perturb; if there were, the closure
  error would not have been zero.
- **Q1/Q4 are within-session contrasts.** Ladder-on vs ladder-off and the interior
  schedules were measured in the same process, same build, same thermal window.
  A base-level constant that shifts both arms equally cancels exactly.
- **The verdict sits 5.9× inside its stop bar.** For the "prefill schedule is not
  the prize" conclusion to flip, the base move would have to change the prefill
  schedule's value by ~6×. Nothing in three decode-round edits can do that.
- **Q5 already demonstrated base-move insensitivity empirically.** `P` measured
  4.002279 s on `e6e6f81`, inside the `e13a6fe` band `[3.993803, 4.007064]`. That
  is a direct observation that a base move does not move `P`, not an argument that
  it should not.

**One quantity does move, and it is now labelled:** `p = P/D`, the prefill share
of a leg, has a base-dependent denominator. `p = 0.3110454468716388` on `e13a6fe`
(`D_mtp` = 12.870633 s) and `0.3322361` on `e6e6f81` (`D_mtp` = 12.049719 s),
+6.81 %. Quoting `p` on `b85e782` requires re-measuring `D_mtp` on that base,
which this revision does not do. The header now labels `p` base-dependent with its
base SHA, and the φ column is base-invariant by construction. No Q1–Q5 conclusion
routes through `p`.

### The 5.0 plausibility ceiling is not a stop target

`senpai/program.md:21` is explicit that the gate is an administrative, fail-closed
sanity check and "not an optimization target, a reason to stop, or a reason to hold
a candidate". I record it here only because the ceiling moved during this
revision: it is now **5.0**, and this experiment's projected `R = 3.0972967` is
nowhere near it. Nothing in E16 was shaped, split, delayed, or tuned with respect
to any ceiling, and the largest prize this write-up hands off (dequant overhead,
0.090180 pts ≈ 7.3 frontier steps) should be pursued at full strength regardless
of where the resulting median lands.
