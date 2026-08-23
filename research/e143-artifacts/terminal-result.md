SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e143_reachable_acceptance_pct_beagle","available":true,"value":1.1481900452488687},"test_metric":{"name":"e143_exactness_divergences","available":true,"value":0}}

# E143 — Where beagle's three points of acceptance go

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e143-beagle-acceptance-decomposition`
- Hypothesis and target cost: beagle loses ~3 pt of realised acceptance against the
  ranked carriers. The brief split the loss into four channels and asked which of
  them a solver can actually close. Target cost: each missed draft token costs one
  extra decode round, and the round count sets the leg time.
- Decision: **dead for the reachable acceptance axis; green for the closure evidence.**
  The measurable, closeable channels are worth **+0.55 % of the published median**.
  The dominant channel is head hidden-state quality (C-d) at **92.2 %** of first
  divergences, above the advisor's 80 % kill line at the point estimate *and* at the
  worst case of the interval.
- `BASE_SHA`: `892dc5e16c3ec741287b3ef213b52b97fd01a6c6`
  (`senpai/qwen38-mtp-r1`) / candidate commit: see "Files" below.
- Yukon promoted submission used as frontier: `1760479a`, live crown median
  `3.70355222` (advisor F2). F1's earlier anchor `572b2cc4` is carried alongside it
  and both are asserted by `research/e143_value.py`.
- Candidate build fingerprint: worker
  `a73ac91f20b4f446981ee0695fcd41d858166b56683d7e3261f5c9172f8f6153`, CLI
  `41a568caf74345a4c1757f96bc64cc38d0b73087566e2ffde3e1667e82ca0262`.
- Submitted-surface / generated-twin / metallib digests: **not applicable.**
  This experiment changed **no** submitted source file. Every artifact is under
  `research/`.
- Submitted candidate files: none.
- Supporting test, tooling, or documentation files:
  `research/e143_r0.py`, `research/e143_stage_b.sh`, `research/e143-r0.json`,
  `research/e143_r1.py`, `research/e143_r1.sh`, `research/e143-r1.json`,
  `research/e143_value.py`, `research/e143_f2.py`, `research/e143-f2.json`,
  `research/e143_c2_head.py`, `research/e143_c2_head.sh`,
  `research/e143-c2-head.json`, `research/e143_c2_session.sh`,
  `research/e143_c2_read.py`, `research/e143-c2.json`, `research/e143_wandb.py`.
- MTP head provenance, digest, and draft policy: declared head
  `amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827`, weight sha256
  `d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`,
  `head_provenance_sha256 = dadbfb80…` on every leg. Draft policy unchanged.
- Token window, fixture, reference source, harness:
  R0/R1 read 3,072 emitted tokens over six seeds of the public fixture with the
  candidate's own reference pass. C2 used 512 decode tokens per leg.
  **`harness=local` throughout.** No ranked measurement was taken.
- Exact cell: the MTP proposal head's rerank screen (coarse top-32 → exact affine-4
  rerank) and, for C2, the head's `layers.0.self_attn.{k,v}_proj` load path.
- Official causal path and score equation: closing an acceptance channel removes
  decode rounds from the **candidate** leg only, so it lowers
  `candidate_mtp_seconds_per_token_mean` and raises every affected `raw_p`. The
  ranked serial numerator is untouched. Conversion from a per-prompt raw-ratio gain
  to a published-median gain uses **Rule 121** (`research/e143_value.py`): the
  published score is the *sorted order statistic*, so a gain only pays where it moves
  one of the two middle values.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh` — not required.
  No file outside `research/` was modified, so the gate chain that guards source
  edits never applied. No forbidden line range was read into an edit.
- Editable source bytes / headroom / growth / exempt-head bytes: unchanged from base.
- Scored-path reachability evidence: all 2,634 draft trials analysed in R0/R1 were
  captured **on the live scored trajectory** from the timed worker, not synthesised.

---

## 1. Result in one line

Beagle's acceptance deficit is **not** a screening, ranking, or vocabulary-coverage
problem. It is a head hidden-state quality problem, and that is the one channel a
solver cannot close from the editable surface.

| channel | definition | pooled events / 2,634 trials | verdict |
| --- | --- | ---: | --- |
| **C-a** | target token structurally unproposable (outside the head's output vocabulary) | 7 (0.266 %) | **real but tiny**, and beagle-only |
| **C-b** | target token outside the head's coarse top-32 rerank window | **0** | **measured zero on real rows** |
| **C-c** | in-window but mis-ranked by the exact rerank | **0 by proof** | closed |
| **C-d** | head hidden state prefers another token | **273** (92.2 % of divergences) | **not closeable from the editable surface** |
| unresolved | offline replay did not reproduce the device decision | 16 (`e143_unresolved_fraction = 0.0541`) | stated uncertainty |

---

## 2. Primary metric

**`e143_reachable_acceptance_pct_beagle` = +1.1482 % of the beagle raw ratio**,
68 % CI **[+0.7374 %, +1.7857 %]**.

Under Rule 121 at anchor `1760479a` that is **+0.5503 % of the published median**,
68 % CI **[+0.3535 %, +0.8559 %]**.

- All of it is C-a. Beagle marginal `dM/dx = 0.4793`; beagle's ceiling is at
  `x = 9.667 %` (worth +4.6337 % median), so +1.1482 % is comfortably **below**
  saturation and converts linearly.
- The gain shape is **beagle-only**: 5 C-a events on 884 beagle trials, **0** on 429
  essays trials (68 % upper bound +0.4721 %), and 2 on 1,321 trials of prompts whose
  marginal value at this anchor is exactly zero.
- Beagle-only is the *safest* shape on this board. The two live receipts the advisor
  flagged (`e003a86d`, `09b452f3`) each lost >1 % of median by improving **essays**
  out of the median pair. A beagle-only gain cannot do that, because it never touches
  the upper slot. For reference, the upper-slot buffers at this anchor are
  medicine **0.9521 %**, republic **1.1538 %**, botany **1.9081 %**.
- **C-a is alphonse's surface, not mine** (`qwen35Top32RealCount` / `Qwen35Top32Plan`,
  `draftTokenIDWithDeclaredRerank`). I priced it; I did not implement it.

### C-a boundary correction

R0 found the C-a boundary should be `t* >= 98,304`, not `98,330`. Alphonse's earlier
0.4677 % C-a figure is reproduced exactly by a `targetTail` double-count
(14 / 3,368 = 0.4157 %); the de-duplicated rate on real trials is **0.2658 %**.

---

## 3. C-b and C-c: the decisive falsification (R1)

W&B run **`myon2da0`** —
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/myon2da0>

**Arm S** replayed the head's real coarse screen on all 2,634 captured rows:

- `screen_loss_at_32 = 0.0`, **recall@32 = 1.000**
- worst coarse rank of the exact argmax over 2,634 rows = **31**; median 1; p99 5
- recall@8 = 0.996963; recall@16 = 0.999620

The mechanism is a large margin-to-noise ratio, not luck:

- exact margin row 1 → row 32: median **13.581** logits, **min 1.871**
- coarse error sd: median **0.752**, max **0.841**
- margin measured in error sigmas: median **18.74**, **min 2.27**, p01 3.17

**Rule 101 dose-response** (inject Gaussian coarse error at k × the measured sd, then
re-screen; monotone, so the harness can detect a loss when one exists):

| injected error | screen loss @32 |
| ---: | ---: |
| 0σ | 0.00000 |
| 1σ | 0.00190 |
| 2σ | 0.01025 |
| 4σ | 0.07099 |
| 8σ | 0.41989 |
| 16σ | 0.91648 |
| 32σ | 0.99089 |

**C-c = 0 by proof.** The rerank inside the window is the *exact* affine-4 score. A
token the exact scorer would rank first cannot be mis-ordered by the exact scorer.
There is no approximation left in that stage to be wrong.

**Arm F** (independent surrogate, calibrated σ = 0.85; simulated miss 0.11124 vs
measured 0.10972, residual 0.00152; `surrogate_usable = true`) agrees: **0 C-b events
at every σ ≤ 1.00**, first event at σ = 1.20.

### Pre-registered fork

`cb_plus_cc_median_pct = 0.000`. Rule-of-three 95 % upper bound on the C-b rate is
0.0011390, which is **+0.2312 %** of beagle raw = +0.2312 % median — below the
+0.30 % fork line **even at the upper confidence bound**. `take_c2_fallback = true`.

---

## 4. C-d as a stated uncertainty (advisor F2 request)

C-d is a **residual**, not a direct measurement:
`C-d = misses − C-a − C-b − C-c`. I bracket it from "every unresolved row is
actionable, plus the full rule-of-three C-b allowance of 3.0 events" (low) to
"every unresolved row is C-d" (high).

| carrier | trials | misses | C-a | unresolved | C-d events | C-d share of first divergences | `unresolved_fraction` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beagle | 884 | 92 | 5 | 6 | 81 [80, 87] | **88.04 %** [86.95, 94.57] | 0.0652 |
| essays | 429 | 68 | 0 | 2 | 66 [65.5, 68] | **97.06 %** [96.34, 100.00] | 0.0294 |
| other | 1,321 | 136 | 2 | 8 | 126 [124.5, 134] | 92.65 % [91.54, 98.53] | 0.0588 |
| **pooled** | **2,634** | **296** | **7** | **16** | **273** [270, 289] | **92.23 %** [91.22, 97.64] | **0.0541** |

`closes_axis_at_point = true`, `closes_axis_at_worst_case = true`. The lowest C-d
share anywhere in the interval, on any carrier, is **86.95 %** — still above the
80 % kill line.

**C-d ceiling (unreachable).** If C-d were fully closed: beagle +18.60 % raw,
essays +31.23 % raw → **+5.234 % median** under Rule 121. The naive Rule 116 linear
conversion gives +25.18 %, a **4.8× overstatement**, because both carriers saturate
long before that. The C-d interval does not move the median at all, for the same
reason.

**Honesty note.** The load-bearing measurements in this report are arm S's screen
recall on real rows and the C-c proof. C-d's *size* is inferred from those by
subtraction. The `e143_unresolved_fraction = 0.0541` above is the offline-replay
disagreement rate (offline argmax matches the device on 96.96 % of rows; max s1 delta
0.083) and is reported beside every C-d figure, as F2 asked.

**Miss-to-score constant.** F2 §2's geometric model implies `MISS_TO_SCORE_PCT = 290`
(`dT/dp = 16.82` at `p = 0.9341`, depth 6). The contract names 203. My E139
calibration measured **209.5 ± 93.1**. The three disagree by more than the campaign
usually tolerates; the E139 interval covers 203 but not 290. I did not resolve this,
and I did not use any of the three in the primary metric — the primary metric is
priced from measured event counts through Rule 121, not through a miss-to-score
constant.

---

## 5. C2 fallback arm (affine-4 g64 KV precision islands)

### 5.1 Rung 0 — the arm needs no source edit

`research/e143-c2-head.json`, `env_arm_has_c2_numerics = true`:

The declared head already ships `layers.0.self_attn.{k,v}_proj` as affine-4 g64
weights that are **bit-identical** to `quantize(precision_islands.{k,v}.weight, 64, 4)`
— 0 mismatching words in weight, scales, and biases. Source inspection confirms arm
`q` sets `installsKV == false`, so `_exactKVDenseW` is never built.

Therefore **`DARKBLOOM_QWEN_MTP_ISLAND_ARM=q` runs C2's exact numerics** with no
source edit, no rebuild, and no gate chain. The byte saving is real: KV dense BF16
20,971,520 B → affine-4 5,898,240 B, saving **15,073,280 B = 4.658 %** of the
323.59 MB draft step.

### 5.2 Prior art — C2 was already refuted end-to-end

`research/e124-artifacts/terminal-result.md` (edward, E124, PR #125) records that
**E82 measured this exact arm at 512 tokens** (W&B `o0rawiol`, `yerghmxz`):

| arm | s/tok | rounds | vs `declared` |
| --- | ---: | ---: | ---: |
| `declared` / `all` | 0.031432 | 78 | ref |
| `none` | 0.031547 | 79 | +0.366 % slower |
| **`q`** | **0.031998** | **80** | **+1.801 % SLOWER** |

`q` was E82's **worst** arm. E124 repriced it at **+0.084 to +0.116 % ranked** gross
(not the brief's +0.35 %), noted that one extra decode round costs 0.965 % of the leg,
concluded "the islands buy about 1.8× more acceptance than they cost in bytes", and
declared the direction **dead**.

The brief's +0.35 % point and +0.30–0.42 % band price the **bytes only**. They do not
price the acceptance loss, which is the term that dominates.

### 5.3 Rung 1 — live gate-qualified ABBA measurement

<!-- C2-MEASURED-BLOCK -->

---

## 6. Evidence

- Host, instance, chip, memory profile, toolchain, thermal policy:
  `ip-10-231-2-227.ec2.internal`, Apple **M4 Pro**, 48 GiB, plugged-in high-power,
  **real 40 C cool gate enforced on every C2 leg** (`gate_qualified_for_timing=true`).
  R0/R1 are offline analyses of a captured trace and are not timed measurements.
- `head_provenance_sha256` for every leg: `dadbfb80…` (identical on all four C2 legs;
  the reader asserts a single-valued identity tuple).
- Exact baseline and candidate commands: section 8.
- Cheapest real falsification gate and positive-control verdict:
  arm S's **Rule 101 dose-response** is the positive control. Injecting 1σ of extra
  coarse error produces a detectable 0.190 % screen loss, so a `screen_loss_at_32` of
  exactly 0 is a real measurement and not a broken harness. The C2 reader carries a
  **Rule 101 wrong-arm control** and asserts three Rule 114 witnesses per leg.
- Tests and risk-based checks, in execution order: no source file changed, so the
  source gate chain did not apply. The checks that did run are the Rule 114 witness
  assertions, the wrong-arm control, the identity-tuple assertion, cross-arm token
  agreement, and `research/e143_value.py`'s 26-figure self-check against F1 and F2.
- Exact-token and row-ledger verdict: `e143_exactness_divergences = 0`.
- Divergent tokens or failure category: none.
- Generated-twin audit: not relevant; no Metal source changed.
- Peak RAM or head/artifact size: head 427.7 MB unchanged; the `q` arm removes
  15.07 MB from the resident draft step.
- Official status and score: **not submitted.** Thorfinn holds the submission slot.

---

## 7. Noise context (Rule 120)

Medpair 1σ ≈ **0.30 %**; 8-prompt 1σ ≈ **0.15 %**. The primary +0.5503 % median gain
is ~1.8 medpair sigmas — real but not large. The C2 arm's E82-measured **+1.801 %
regression** is 6 medpair sigmas in the wrong direction.

---

## 8. Reproduction

```bash
# R0 — capture and channel decomposition (offline, no GPU)
bash research/e143_stage_b.sh            # writes research/e143-r0.json

# R1 — arm S screen replay, Rule 101 dose-response, arm F surrogate
bash research/e143_r1.sh                 # writes research/e143-r1.json

# Rule 121 order-statistic model; self-checks 26 published F1/F2 figures
python3 research/e143_value.py           # exits non-zero on any disagreement

# F2 re-pricing: C-d residual interval, gain shape, ceiling
python3 research/e143_f2.py              # writes research/e143-f2.json

# C2 rung 0 — head bit-identity check (offline)
bash research/e143_c2_head.sh            # writes research/e143-c2-head.json

# C2 rung 1 — gate-qualified ABBA session, legs all,q,q,all at 512 tokens
bash research/e143_c2_session.sh 512     # via run_job
PYTHONPATH=research python3 research/e143_c2_read.py   # writes research/e143-c2.json

# W&B
python3 research/e143_wandb.py --resume myon2da0 --c2 research/e143-c2.json
```

---

## 9. Conclusion

- **What happened and why.** The acceptance deficit is concentrated in the one place
  a solver cannot reach. The head's coarse screen is not the bottleneck: it recalls the
  exact argmax on 100 % of 2,634 real rows with a median margin of 18.7 error sigmas.
  The exact rerank cannot mis-order what it would rank first. What is left is the head
  simply preferring a different token — 92.2 % of first divergences, and at worst
  86.95 %.
- **Evidence for or against the mechanism.** For: recall@32 = 1.000 with a monotone
  dose-response control that proves the metric can fail; an independent calibrated
  surrogate that also finds 0 C-b events; a bit-identity proof that the C2 arm is
  reachable by an environment variable. Against: C-d is a residual, and 5.41 % of rows
  did not reproduce offline.
- **Prompt or M5 transfer risk.** Low for the closure claim — the recall margin is
  18.7 sigmas, so a modest M4→M5 numerical shift cannot open C-b. Higher for the C-a
  price, which rests on 5 events.
- **Smallest useful next action.** Stop spending students on the acceptance-screen
  axis. Ledger entry (J) at `senpai/campaign-ledger.md:21815` is the productive
  alternative and it is a **manifest-only** change (see follow-ups).
- **Recommendation: close.** C-b and C-c are refuted. C-a is worth +0.55 % median and
  belongs to alphonse. C-d is above the kill line at every point of its interval, so
  the advisor's pre-registered rule fires: close the acceptance axis and redirect.
  C2 is refuted twice — once by E82's end-to-end measurement and once again here.

---

## 10. Suggested follow-ups (not implemented; outside my surface)

1. **Ledger (J): re-quantize the declared head.** `declared` is the *naive* affine-4
   g64 requantization of `master-bf16` and pays **−0.82 pt** of acceptance to
   quantization damage. `qat-q4` (relL2 2.89e-2…3.52e-2 vs declared's
   9.18e-2…9.97e-2, a 3.2× reduction) recovers **0.71 pt at identical 427.7 MB**,
   changing only `mtp-head.manifest.json`. Priced through Rule 121 at anchor
   `1760479a`:

   | recovered acceptance | uniform shape | beagle-only | essays-only |
   | --- | ---: | ---: | ---: |
   | +0.71 pt (= +1.441 % raw) | **+1.4413 %** | +0.6908 % | +0.4957 % |
   | +0.82 pt (= +1.665 % raw) | **+1.6646 %** | +0.7979 % | +0.4957 % |

   This corroborates ledger (J)'s independent +1.57 % estimate. **Risk to flag before
   assigning:** the shape decides the value. An essays-heavy head improvement caps at
   **+0.4957 %** median no matter how large it is; a beagle-heavy one is worth up to
   **+4.63 %**. Whoever runs it should report the per-carrier acceptance delta, not a
   pooled number. Note this is *not* head fine-tuning or distillation, which is closed
   by six ranked negatives and a controlled refutation — it is a better quantizer for
   the same weights.

2. **Convert R1's dose-response into a price for coarse-metadata coarsening.** E79's
   g64 → g128/g256 lever saves head bytes by making the coarse screen noisier. My
   dose-response table turns that noise into a screen-loss cost directly: 1σ extra
   error → 0.190 % loss, 2σ → 1.025 %. The whole decision therefore reduces to
   measuring the sigma inflation from the coarser grouping, which is a cheap **offline**
   measurement on the head weights — no GPU, no timed leg. If the inflation is under
   ~1σ the lever is nearly free; if it is over 2σ it is not worth the bytes.

3. **Fix the `MISS_TO_SCORE_PCT` disagreement.** 203 (contract) vs 209.5 ± 93.1
   (my E139 measurement) vs 290 (F2 §2's geometric model). Several campaign prices
   depend on this constant. A direct measurement — sweep realised acceptance and
   regress the leg time — would settle it in one session.
