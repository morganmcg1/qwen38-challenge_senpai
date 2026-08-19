# E30 — Does the M=9 cliff survive NA=5, and is `vbuild` really host work?

Student `qwen-alphonse` · PR #35 · base `d08feb85bf65959d7eaa1455e36a0173b3edd8d9`
(`UPSTREAM_SHA` per `senpai/frontier-state.json` at that base).
Host: Apple M4 Pro `Mac16,11`, 48 GiB, 20 GPU cores, NAX off.
Local mode: `--local-iterate`, 256 decode tokens, offered depth 8, declared head
`hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827` (sha256 `7bbb40de…`,
270,408,194 B) — **byte-identical to the E29 baseline arm**.
Predictions pre-registered at `3ec7cbe` before the first run; instrument
(`research/e29-run.sh`, `research/e29_analyze.py`) unchanged from E29.

## Headline

1. **The M=9 cliff is closed.** `bound_M8` fell **8.0102 % → 3.1295 %** (repeat
   arm 2.2370 %), the cheapest width **flipped 8 → 9**, and the M=8→M=9 step
   collapsed from **1.2322 → 1.1129 / 1.0993** against a 1.125× row increase.
   M=9 was 9.53 % *more* expensive per row than M=8; it is now 1.08–2.29 %
   *cheaper*. The per-row ordering I reported in E29 has inverted, exactly as the
   assignment predicted from E27's width curve.
2. **`vbuild` is GPU wait, not host graph construction — the host-work
   explanation is now falsified, not merely unsupported.** `vbuild` at M=9 fell
   **−8.69 ms** (repeat **−10.34 ms**) while **every untreated width stayed flat
   within ±1 ms in the same arms**, with zero host-code change, an unchanged
   969-op dispatch inventory, and an unchanged graph shape. Host work cannot fall
   because a GPU kernel stopped re-streaming weights.
3. **E30 earns no speedup of its own and I am not claiming one.** The
   −20.4 ms/round at M=9 and the −5.04 % MTP leg are **E27's** prize, measured
   independently on a fixture E27 did not use. What E30 delivers is a closed
   cliff, a settled attribution, and a cross-fixture confirmation of E27's
   mechanism.

Both prediction sets were adjudicated against the numbers below. **P1's
threshold failed and my registered point estimate was right**; P2's threshold
passed.

## Provenance and why this is a clean natural experiment

`git diff 09fd819..HEAD -- Sources Vendor` (E29's measured tree → this base) is
four files. Only one is live on this host:

| change | live here? | why |
|---|---|---|
| `quantized.{h,cpp}` twins: `<T,9,3>`→`<T,9,5>`, `<T,5,3>`→`<T,5,5>`, `static_assert(NA<=5)` | **yes** | E27's treatment; M=9 goes 3 → 2 weight streams |
| `Qwen36MTPBlockSession` residency wiring | no | `physicalMemory >= 96 GiB`; this host is 48 GiB |
| `RuntimeStartupMemoryPolicy` command-buffer defaults | no | same ≥96 GiB gate |
| my E29 trace + ladder knob | yes, **byte-identical** to E29's | instrument unchanged |

**M=8's dispatch is `<T,8,4,true>` in both trees**, so M=8 is an interleaved,
same-arm internal control for the treated width. Build witness: worker relinked
to `b44e4888…` (was `e567aa90…`), `[1/3] Compiling quantized.cpp`, binary
contains `NA in [2, 5]` and **zero** hits of `NA in [2, 4]`. The trusted CLI is
**byte-identical** to E29's (`d9186c45…`) — the driver is literally the same
binary.

## Both legs, true decode (never a ratio alone)

`decode_seconds` is prefill-inclusive, so the denominator is the sum of the
parent's own `block_request_seconds` (= `decode_seconds − seed_prefill_seconds`
to <1 ms).

| arm | tree | entry GPU °C | **serial leg ms/tok** | **MTP leg ms/tok** | MTP ms/round | true-decode ratio |
|---|---|---:|---:|---:|---:|---:|
| E29 D0 | `09fd819` | 41.57 | 66.8745 | 23.5920 | 172.5586 | 2.8346 |
| E30 D0 | `c264ac6` | 43.38 | **66.8645** (−0.015 %) | **22.4021** (−5.04 %) | 163.8551 | 2.9847 (+5.29 %) |
| E30 D0b | `3dce66b` | 63.48 | **67.0071** (+0.20 %) | **22.3016** (−5.47 %) | 163.1202 | 3.0046 (+6.00 %) |

**The serial leg did not move** (−0.015 % / +0.20 %, inside the ±2 % registered
band and inside my 0.86 % repeat noise floor), because E27 changed only the M=5
and M=9 dispatch and the serial leg runs at M=1. The MTP leg fell 5.0–5.5 %.
That is the shape a legitimate gain has under the corrected scoring model:
**numerator flat, denominator down** — the exact opposite of the ladder, where
the whole ratio move came from the numerator. MTP-leg throughput gain
**+5.31 % / +5.79 %**, which sits inside E27's own end-to-end range of +2.49 %
to +7.02 % and is consistent with this fixture putting 45.5 % of steady rounds
at M=9 versus 43.6 % for E27's high-value fixture.

## Realised depth histogram — identical in all three arms

| M | rounds | share of 33 steady rounds |
|---:|---:|---:|
| 2 | 1 | 3.0 % |
| 6 | 9 | 27.3 % |
| 7 | 3 | 9.1 % |
| 8 | 5 | 15.2 % |
| 9 | 15 | 45.5 % |

`accepted_draft_total` 222, `rejected_draft_total` 6, `accepted_draft_rate`
0.9737, `effective_mean_draft_len` 6.5143 — identical across E29 D0, E30 D0 and
E30 D0b. Every cost number below is re-weightable onto another histogram with
this table.

## Per-width cost, both instruments

Trace (six-way split, steady state, warmup 2 dropped). `unaccounted` 79–85 µs
per arm, +0.001 % of round time, so the tiling still holds.

| M | n | E29 round | **E30 round** | E29 vbuild | **E30 vbuild** | E29 eval | **E30 eval** | E29 ms/tok | **E30 ms/tok** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 70.849 | 72.918 | 35.998 | 37.259 | 34.381 | 33.589 | 35.425 | 36.459 |
| 6 | 9 | 141.199 | 141.795 | 73.118 | 73.077 | 65.327 | 64.893 | 23.533 | 23.632 |
| 7 | 3 | 154.183 | 154.016 | 78.183 | 78.251 | 69.168 | 68.920 | 22.026 | 22.002 |
| **8** | 5 | 167.675 | **167.315** | 83.922 | **83.859** | 75.604 | 76.183 | 20.959 | **20.914** |
| **9** | 15 | 206.607 | **186.200** | 101.720 | **93.029** | 96.511 | **83.287** | 22.956 | **20.689** |

Trusted parent, same rounds, from `effective_draft_lengths × block_request_seconds`
— an instrument I did not write:

| M | n | E29 mean | **E30 D0** | **E30 D0b** | E29 ms/tok | E30 D0 ms/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 71.013 | 73.336 | 72.119 | 35.506 | 36.668 |
| 6 | 9 | 141.539 | 142.248 | 142.033 | 23.590 | 23.708 |
| 7 | 3 | 154.898 | 154.742 | 155.041 | 22.128 | 22.106 |
| 8 | 5 | 168.201 | 167.724 | 169.156 | 21.025 | 20.966 |
| **9** | 15 | 206.759 | **186.649** | **185.620** | 22.973 | **20.739** |

`bound_M8`: trace 8.0102 → 3.1295 / 2.2370; parent 7.8862 → **3.1679 / 2.1519**.
The two instruments agree to **0.04–0.09 pp**, so the primary metric does not
depend on my trace.

### The width-specificity argument (this is the load-bearing evidence)

E29 → E30 D0 / D0b deltas:

| M | round Δ | vbuild Δ | eval Δ | treated? |
|---:|---:|---:|---:|---|
| 2 | +2.07 / +1.16 | +1.26 / +0.17 | −0.79 / +0.62 | no |
| 6 | +0.60 / +0.33 | −0.04 / −0.63 | −0.43 / −0.90 | no |
| 7 | −0.17 / +0.22 | +0.07 / +0.93 | −0.25 / −0.38 | no |
| 8 | −0.36 / +0.89 | −0.06 / +0.54 | +0.58 / −0.55 | **control** |
| **9** | **−20.41 / −21.31** | **−8.69 / −10.34** | **−13.22 / −12.09** | **yes** |

Four untreated widths flat within ±1.3 ms; the one treated width moves ~20 ms.
Session-level drift, thermal state and host load cannot produce that pattern.
In-arm sequence slopes at M=9 are −0.001 / +0.092 / −0.100 ms per round position
(≤1.5 ms across a 15-round span), and **D0b entered 20 °C hotter yet measured
M=9 slightly faster** (185.29 vs 186.20), so the effect is thermally robust in
the direction that matters.

## P1 — `e30/all_at_m8_upper_bound_pct`

```
bound_M8 = 100 x (steady_round_total - full_accept_tokens x ms_per_token(M=8))
                 / steady_round_total
```

| quantity | value |
|---|---:|
| assignment baseline | 8.25 |
| E29 D0 measured | 8.0102 |
| **E30 D0 (primary)** | **3.1295** |
| E30 D0b (repeat) | 2.2370 |
| parent-side cross-check | 3.1679 / 2.1519 |
| two-arm mean (sd) | 2.683 (0.631) |
| advisor threshold "< 2.0" | **not met** |
| my registered point 2.58, range [1.8, 3.6] | **mean error +0.10 pp; both arms inside** |

**Verdict: the advisor's threshold failed and the reason is the one I registered
in advance.** The residual is not M=9 any more — it is the untouched widths. E27
could not have driven the bound below 2.0, because 70 % of the remaining
full-accept headroom is the nine M=6 rounds:

| source of the residual 4.17 % best-width headroom | ms | share |
|---|---:|---:|
| M=6 (9 rounds, 54 tokens @ 23.71 vs 20.74 ms/tok) | 160.3 | 70 % |
| M=2 (1 round, 2 tokens) | 31.8 | 14 % |
| M=7 (3 rounds, 21 tokens) | 28.7 | 12 % |
| M=8 (5 rounds, 40 tokens) | 9.0 | 4 % |

Secondary P1 predictions, all correct:

| prediction | registered | measured |
|---|---:|---:|
| best width flips 8 → 9 | yes | **yes** (20.689 vs 20.914 ms/tok) |
| M=8→M=9 round step | 1.105 | **1.1129 / 1.0993** (mean 1.1061, error +0.001) |
| M=9 per-row vs M=8 | inverts | **+9.53 % → −1.08 % / −2.29 %** |
| best-width headroom | ≈4.3 % | **4.174 % / 4.473 %** |

## P2 — `vbuild` at M=9, and which explanation survives

| quantity | value |
|---|---:|
| E29 D0 | 101.720 ms |
| **E30 D0** | **93.029 ms (−8.691)** |
| E30 D0b | 91.384 ms (**−10.335**) |
| advisor threshold "≤ −5 ms" | **met** |
| my registered point −10.5, range [−16, −4] | **mean −9.51, error +0.99 ms; both arms inside** |
| M=9 `eval_wall` Δ | −13.224 / −12.093 |
| M=9 round Δ | −20.408 / −21.314 |
| vbuild share of the round fall | 42.6 % / 48.5 % (proportional model predicted 51.3 %) |

**The falsification table cell reached is row 1: round falls ≥10 ms *and* vbuild
falls ≥5 ms.** So:

- **`vbuild` is not host graph-construction time.** The one-sided test the
  assignment identified is decisive in the direction that kills the host-work
  reading: nothing on the host changed — same Swift, same 969 dispatches, same
  graph, same head bytes — and `vbuild` still lost 8.7–10.3 ms at the single
  width whose *kernel* changed, while staying flat at the control width in the
  same arm. Host work does not respond to a weight-stream count.
- **The backpressure reading survives quantitatively, not just directionally.**
  My registered point estimate came from assuming the removed GPU time would
  split between `vbuild` and `eval_wall` in the 51.3/48.7 ratio E29 measured at
  M=9. Measured split: 42.6 % / 48.5 % into `vbuild`. A ±9 pp miss on a
  mechanism-derived point estimate is the level of agreement that makes me
  believe the mechanism rather than the number.
- **Saturation is a real but bounded refinement.** D0's slightly-low `vbuild`
  share is consistent with the queue still being past `MAX_ACTIVE_TASKS = 10`
  for part of the window, so the host waits on buffer *count* there rather than
  on total work. The data do not require it and I am not claiming it.
- **Per the advisor's §4, this time it does carry a prize** — the round total
  fell by 20.4 ms, so this is not the "relocation with no prize" case. But the
  prize is E27's, not E30's.

Honest note on dispersion: D0's M=9 `vbuild` sd is 3.34 ms because of exactly
**one** round at 104.9 ms inside an otherwise tight 91.1–93.6 cluster — and that
round's *total* was a normal 186.5 ms, so it is a bucket-boundary artefact, not a
slow round. Excluding it gives 92.17 ms (Δ −9.55). D0b's M=9 `vbuild` spans
90.5–91.9 (sd 0.29). `draft_build` rose +0.90 / +0.71 ms at M=9, which is why the
round fall (−20.4) is slightly smaller than `vbuild`+`eval` (−21.9); consistent
with E29's sync-head probe finding that head-chain GPU time straddles those
buckets.

## Controls

| control | registered band | measured | verdict |
|---|---|---:|---|
| M=8 round (untreated dispatch) | ±3 % of 167.675 | −0.21 % / +0.53 % | **pass** |
| serial leg | ±2 % of 66.8745 | −0.015 % / +0.20 % | **pass** |
| histogram + `accepted_draft_total` | unchanged | identical (222) | **pass** |
| proposal head | identical | sha256 `7bbb40de…`, same bytes | **pass** |
| exactness | matched, 0 divergence | `all_tokens_matched: true`, `residual_divergence_count: 0`, `parity_all_ok: true`, 263/263 rows checked, `max_rejected_tail_logit_delta: 0` | **pass** |

## Limits of this evidence

- **Ungated.** `MLXFAST_LOCAL_COOL_GATE=0` (idle GPU sits at ~43 °C, above the
  40 °C gate, so the real gate can never be satisfied on this host).
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`. Entry
  temperatures 43.38 / 63.48 °C, exit 66.51 / 68.53 °C. Directional causal
  evidence within these sessions; **not** a gate-qualified or ranked score.
- **One public fixture, 256 tokens, M4 Pro.** The ranked run is 8 hidden prompts
  × 512 tokens on M5. E27 already showed its end-to-end value swings +2.49 % to
  +7.02 % with M=9 round mass, so this fixture's 45.5 % is not the ranked
  weighting.
- **`bound_M8` is a full-acceptance upper bound**, not achievable headroom: it
  assumes every round at the chosen width accepts all drafts, while the realised
  rate is 0.9737. Forcing a width also changes acceptance, which this framing
  does not model.
- **No new mechanism was implemented.** No candidate-surface file was touched.

## Reproduction

```bash
bash research/rebuild.sh          # required: the base moved E27's kernel twins
E29_RUNS_ROOT="$PWD/.mlxfast-private/e30/runs" \
  bash research/e29-run.sh D0=1:8:default D0b=1:8:default
python3 research/e30_adjudicate.py \
  --baseline .mlxfast-private/e29/runs/D0 \
  --candidate .mlxfast-private/e30/runs/D0 \
  --candidate .mlxfast-private/e30/runs/D0b \
  --json-out research/results/e30-analysis.json
python3 research/e30_log_wandb.py research/results/e30-analysis.json
```

Two arms ≈ 5 min. E29's `D0` run directory must be preserved; it is the baseline.

## Follow-ups not implemented

- **Depth selection now points at M=6, not M=9.** M=6 is the *most* expensive
  realised width per token (23.63–23.71 ms) and carries 70 % of the residual
  bound, while M=9 is now the cheapest. The schedule question has inverted from
  "stop picking 9" to "why 6". This is edward's surface and I did not touch it.
- **`<T,6,3>` is the only remaining multi-stream width in the realised mix**
  (`ceil(6/3) = 2` streams). E27's own legality table says M=6 admits IPG ∈
  {2,3,4}; IPG=4 was measured worse at NA=4, but NA=5 now permits `<T,6,5>` →
  `ceil(6/5) = 2` streams with a single main body plus tail. Whether that beats
  `<T,6,3>` is unmeasured, and 27 % of this fixture's rounds sit there. Kernel
  surface, twin-locked, and out of E30's scope.
- **The M=2 round costs 36.5 ms/token** — 76 % worse than M=9. One round here,
  but on the ranked 512-token window E21's histogram put 995 rounds at M=2.
- **`ConcurrentContext` barrier suppression** remains unexplored (one user
  tree-wide), unchanged from E29's list.
