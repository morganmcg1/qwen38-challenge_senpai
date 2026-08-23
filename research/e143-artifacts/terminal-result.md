SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e143_reachable_acceptance_pct_beagle","available":true,"value":1.1481900452488687},"test_metric":{"name":"e143_exactness_divergences","available":true,"value":0}}

# E143 — Where beagle's three points of acceptance go

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e143-beagle-acceptance-decomposition`
- Hypothesis and target cost: beagle loses ~3 pt of realised acceptance against the
  ranked carriers. The brief split the loss into four channels and asked which of
  them a solver can actually close. Target cost: each missed draft token costs one
  extra decode round, and the round count sets the leg time.
- Decision: **dead for the reachable acceptance axis; green for the closure evidence;
  C2 refuted.** The measurable, closeable channels are worth **+0.5499 % of the
  published median**. The dominant channel is head hidden-state quality (C-d) at
  **92.23 %** of first divergences, above the advisor's 80 % kill line at the point
  estimate *and* at the worst case of the interval. The C2 fallback arm measured
  **+0.0702 %** against a pre-registered **+0.35 %**, inside its own replicate spread.
- `BASE_SHA`: `892dc5e16c3ec741287b3ef213b52b97fd01a6c6`
  (`senpai/qwen38-mtp-r1`) / candidate commit: see "Files" below.
- Yukon promoted submission used as frontier, per F3's standing instruction:
  **VALUE anchor `3ba6ee9d`**, live crown median `3.70576324`;
  **CANDIDATE frontier `1760479a`**, median `3.70355222`, because F3 finding 213
  measures `3ba6ee9d`'s candidate leg as 0.0328 % slower. F1's `572b2cc4` is carried
  as well. `research/e143_value.py` asserts all three anchors and 44 published
  figures from F1, F2, F3 and F4.
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

Under Rule 121 at the F3 value anchor `3ba6ee9d` that is **+0.5499 % of the published
median**, 68 % CI **[+0.3531 %, +0.8552 %]**. At the superseded `1760479a` it was
+0.5503 %; the re-anchor moves it by 0.0005 pp.

| exchange rate | beagle raw | published median at `3ba6ee9d` |
| --- | ---: | ---: |
| **203, the contract constant — PRIMARY** | **+1.1482 %** | **+0.5499 %** |
| 290, F2 §2's geometric model (F3 §8's extra column) | +1.6403 % | +0.7855 % |

- All of it is C-a. Beagle marginal `dM/dx = 0.4789`; beagle's ceiling is at
  `x = 9.713 %` (worth +4.6514 % median), so +1.1482 % is comfortably **below**
  saturation and converts linearly. It uses 11.8 % of beagle's runway.
- The gain shape is **beagle-only**: 5 C-a events on 884 beagle trials, **0** on 429
  essays trials (68 % upper bound +0.4721 %), and 2 on 1,321 trials of prompts whose
  marginal value at this anchor is exactly zero.
- Beagle-only is the *safest* shape on this board. The two live receipts the advisor
  flagged (`e003a86d`, `09b452f3`) each lost >1 % of median by improving **essays**
  out of the median pair. A beagle-only gain cannot do that, because it never touches
  the upper slot. The upper-slot buffers tightened at the new crown, which makes the
  point stronger, not weaker: medicine **0.8254 %**, republic **1.1866 %**, botany
  **1.4173 %**.
- **C-a is alphonse's surface, not mine** (`qwen35Top32RealCount` / `Qwen35Top32Plan`,
  `draftTokenIDWithDeclaredRerank`). I priced it; I did not implement it.

### C-a boundary correction

R0 found the C-a boundary should be `t* >= 98,304`, not `98,330`. Alphonse's earlier
0.4677 % C-a figure is reproduced exactly by a `targetTail` double-count
(14 / 3,368 = 0.4157 %); the de-duplicated rate on real trials is **0.2658 %**.

### F3 item 7 — reconciling my C-a with alphonse's E141

**Question 1: essays digits and the interval rule.**

```
essays  C-a events 0    draft trials 429
beagle  C-a events 5    draft trials 884
other   C-a events 2    draft trials 1,321
```

The interval rule is the **Wilson score interval at z = 1.0** (68.27 % nominal),
applied to the draft-trial count. At zero events it reduces exactly to
`upper = 1 / (trials + 1)`, so the essays upper bound is `1 / 430 = 0.0023256` per
draft trial, which is **+0.4721 % of essays raw** at 203. I did not use a rule of
three here; that appears only as the 95 % C-b allowance in the C-d interval below.

**Question 2: can my census see a C-a event beyond draft position 1?**

**Yes. `census_is_position_1_only = false`.** My census counts each round's first
divergence, and a first divergence lands anywhere in `0 .. width-1`:

| first-divergence index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| divergences | 113 | 96 | 52 | 26 | 3 | 3 | 3 |

**61.8 %** of the 296 first divergences are at index ≥ 1, and the maximum is index 6.
The 7 C-a events themselves sit at indices `[0, 0, 0, 0, 1, 2, 2]` — **3 of 7 beyond
position 1**. So a position restriction is *not* the explanation for the
disagreement, and the two numbers are not automatically both right.

**What the remaining difference actually is.** Three things, in decreasing size:

1. **Different populations.** Alphonse tokenizes source corpora. I read the model's
   own generated continuation from a 512-token seed of the public fixture. Unproposable
   ids are rare mid-vocabulary tokens, and their frequency in raw source text is not
   their frequency in a greedy model continuation. This is the largest term and neither
   of us has measured it.
2. **Not every emitted token is a draft trial.** 2,634 of my 3,072 emitted tokens
   (85.74 %) were draft trials; **438 (14.26 %) were emitted with no draft made for
   them** and therefore cost zero acceptance. A token-share census prices all of them.
   In this sample the correction happened to be zero — all 7 unproposable emitted
   tokens landed on draft trials — but the 14.26 % exposure is real.
3. **The `targetTail` double-count**, worth exactly 2.0×, already described above.

**Net for alphonse's arm A.** My beagle 5/884 and essays 0/429 give +1.1482 % and
0.0000 % of raw; his are +0.4832 % and +0.4535 % of tokens. Taking my essays 68 %
upper bound of +0.4721 % raw as the generous end, the honest combined range is
**+0.35 % to +0.80 % of median against his +0.9495 %**. My point estimate is
+0.5499 %. Against a cost of 0.70 % to 0.83 % his arm A is **negative at my point
estimate and only marginal at my upper bound**. The disagreement is real and I cannot
resolve it from my capture; the cheap resolver is for one of us to run *both* censuses
on the *same* token stream.

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
essays +31.23 % raw → **+5.270 % median** under Rule 121 at `3ba6ee9d`. The naive
Rule 116 linear conversion gives +25.18 %, a **4.8× overstatement**, because both
carriers saturate long before that. The C-d interval does not move the median at all,
for the same reason: `median_pct_low = median_pct_point = median_pct_high`. Nor does
the exchange rate — at 290 the ceiling is still **+5.270 %**, because both carriers
are already pinned.

**Honesty note.** The load-bearing measurements in this report are arm S's screen
recall on real rows and the C-c proof. C-d's *size* is inferred from those by
subtraction. The `e143_unresolved_fraction = 0.0541` above is the offline-replay
disagreement rate (offline argmax matches the device on 96.96 % of rows; max s1 delta
0.083) and is reported beside every C-d figure, as F2 asked.

**Miss-to-score constant.** F2 §2's geometric model implies `MISS_TO_SCORE_PCT = 290`
(`dT/dp = 16.82` at `p = 0.9341`, depth 6). The contract names 203. My E139
calibration measured **209.5 ± 93.1**, whose 1σ upper edge is 302.6, so 290 is inside
it and the two models do not formally conflict — but the point estimates differ by
43 %. Per F3 §8 I keep **203 primary** and add the 290 column in §2. The closure does
not depend on the choice: the C-d ceiling pins at +5.270 % under either constant.

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

`research/e143-c2.json`. Design `all, q, q, all`, 512 decode tokens per leg, real
40 C cool gate on every leg, one identity tuple, 49 minutes of GPU.

| leg | arm | entry C | exit C | MTP s/tok | serial s/tok | rounds | eff draft len | accepted draft rate | tokens matched | residual divergences |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `e143c2-1-all` | `all` | 38.97 | 62.95 | 0.02924142 | 0.07467455 | 82 | 5.85366 | 0.895833 | true | 0 |
| `e143c2-2-q` | `q` | 58.91 | 61.97 | 0.02922488 | 0.07539202 | 82 | 5.87805 | 0.892116 | true | 0 |
| `e143c2-3-q` | `q` | 59.14 | 62.50 | 0.02925516 | 0.07468657 | 82 | 5.87805 | 0.892116 | true | 0 |
| `e143c2-4-all` | `all` | 58.50 | 62.41 | 0.02927967 | 0.07459817 | 82 | 5.85366 | 0.895833 | true | 0 |

| metric | value |
| --- | ---: |
| `all` MTP s/tok, mean of two | 0.02926055 |
| `q` MTP s/tok, mean of two | 0.02924002 |
| **`e143_c2_ranked_pct`** | **+0.0702 %** |
| `e143_c2_realised_acceptance_delta_pp` | **−0.3717 pp** |
| `e143_c2_realised_per_step_p_delta_pp` | 0.0000 |
| ABBA replicate spread | **0.1307 %** |
| `e143_exactness_divergences` | **0** |
| cross-arm token mismatches | 0 / 1,024 compared, on each of three passes |
| Rule 114 witnesses, distinct arms | present, `all` and `q` |
| Rule 101 wrong-arm control | passed on all four legs |

**Verdict: not useful.** The measured effect `+0.0702 %` is **smaller than the ABBA
replicate spread of 0.1307 %**, so this session cannot even establish its sign. It is
one fifth of the brief's `+0.35 %` point and sits far outside the pre-registered
`[+0.30 %, +0.42 %]` band.

Two further readings:

- **The bytes are real but the leg barely notices them.** Removing 4.658 % of the
  draft-step bytes bought 0.0702 % of the candidate leg. Taking the head step as
  bandwidth bound, that implies the draft-step byte traffic is about **1.5 % of
  candidate leg time** (`0.0702 / 4.658`). Labelled: inference, from these two
  measurements only.
- **The acceptance cost is real and this fixture got lucky.** Arm `q` accepted the
  same 430 drafts from 482 proposals against `all`'s 430 from 480 — a genuine
  **−0.3717 pp** of realised acceptance — but the round count stayed at 82 on both
  arms. E82's fixture crossed a round boundary and paid 2 extra rounds for the same
  mechanism, which is why it measured −1.801 % rather than a wash. One extra decode
  round is 1.22 % of this 82-round leg, so a single boundary crossing outweighs the
  entire byte saving by **17×**.

Together with E82 and E124 this is the third independent refutation. Rule 107 asked
for a measured acceptance delta on the current base and it is negative. **Not
implemented, and not recommended.**

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

### Standard metric table — C2 arm contrast, `harness=local`

| Metric | Baseline (`all`) | Candidate (`q`) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.07463636 | 0.07503930 | +0.540 % |
| MTP seconds/token | 0.02926055 | 0.02924002 | **−0.0702 %** |
| local serial-relative speedup | 2.55075 | 2.56631 | +0.610 % |
| effective mean draft length | 5.85366 | 5.87805 | +0.417 % |
| accepted draft rate | 0.895833 | 0.892116 | **−0.3717 pp** |
| decode rounds | 82 | 82 | 0 |

The local serial-relative speedup column is reported for completeness only and is
**not** the causal quantity here. Both legs run the same candidate binary and the
island arm changes only the MTP leg, so the serial column is pure run-to-run lottery
(0.07460–0.07539 across four legs, a 1.06 % spread with no arm structure). The
decision uses the absolute MTP seconds-per-token contrast, which is the term that
maps to the ranked denominator. Every identity field matched across all four legs:
worker digest, CLI digest, head directory and `head_provenance_sha256`, host, chip,
gate flag, dirty-path count, and token count. The only recorded difference is
`base_sha`, which moved from `47d4df42` to `0d0d4921` between legs 2 and 3; the
reader diffs those two commits and confirms the change is research-only.

The local score is a one-prompt directional measurement. It is not the ranked median
across eight hidden prompts.

---

## 7. Noise context (Rule 120)

Medpair 1σ ≈ **0.30 %**; 8-prompt 1σ ≈ **0.15 %**. The primary +0.5499 % median gain
is ~1.8 medpair sigmas — real but not large, and under Rule 120 a single receipt
claiming it would also need the 8-prompt sign test. The C2 arm's measured +0.0702 %
is inside its own **0.1307 %** ABBA replicate spread and far inside medpair noise;
E82's **−1.801 %** for the same arm is 6 medpair sigmas in the wrong direction.

---

## 8. Reproduction

```bash
# R0 — capture and channel decomposition (offline, no GPU)
bash research/e143_stage_b.sh            # writes research/e143-r0.json

# R1 — arm S screen replay, Rule 101 dose-response, arm F surrogate
bash research/e143_r1.sh                 # writes research/e143-r1.json

# Rule 121 order-statistic model; self-checks 44 published F1/F2/F3/F4 figures
python3 research/e143_value.py           # exits non-zero on any disagreement

# F2/F3 pricing: C-d residual interval, gain shape, ceiling, C-a positions
PYTHONPATH=research python3 research/e143_f2.py   # writes research/e143-f2.json

# C2 rung 0 — head bit-identity check (offline)
bash research/e143_c2_head.sh            # writes research/e143-c2-head.json

# C2 rung 1 — gate-qualified ABBA session, legs all,q,q,all at 512 tokens
bash research/e143_c2_session.sh 512     # via run_job
PYTHONPATH=research python3 research/e143_c2_read.py   # writes research/e143-c2.json

# W&B
PYTHONPATH=research python3 research/e143_wandb.py --resume myon2da0 \
    --c2 research/e143-c2.json --f2 research/e143-f2.json
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
  axis. The advisor has already named the replacement in F3 §6 and F4: a better
  data-free quantizer of the organizer's pinned `master-bf16`, shipped `in_branch`
  under `mtp-head/`. I did not price it here, because F3 §5 records the ledger (J)
  figures I cited earlier as **superseded** (see below).
- **Recommendation: close.** C-b and C-c are refuted. C-a is worth +0.5499 % median
  and belongs to alphonse. C-d is above the kill line at every point of its interval,
  so the advisor's pre-registered rule fires: close the acceptance axis and redirect.
  C2 is refuted three times — E82 end-to-end, E124's corrected byte model, and this
  session's own ABBA.

### Withdrawal of my ledger (J) citation

In interim comment 3 I recommended ledger entry (J) at
`senpai/campaign-ledger.md:21815` as the productive alternative and priced its
+0.71 pt at +1.4413 % of median. **F3 §5 supersedes that entry** with `## 282.6` at
`senpai/campaign-ledger.md:44656`, recorded as Advisor Error 114: `qat-q4`'s
`finetune_rel_l2_vs_master` is 0.094–0.0998 on all eight trunk tensors, so it is a
different trained trunk and not a lower-error quantization of the same weights; its
acceptance gain fails McNemar (χ² 0.083); and it is 2.7–3.4 % slower per token.
**I withdraw that recommendation and the +1.4413 % price.** I did not read 282.6
before citing (J), and I should have — the entry I cited is 22,841 lines earlier in
the same file.

What survives is only the clean same-weights pair `master-bf16` 93.13 % against
`declared` 92.31 %, which is the −0.82 pt of pure requantization damage the advisor
carries into E144. My earlier uniform-shape conversion of a recovered point is still
arithmetically valid and `research/e143_value.py` now reproduces F4's table exactly,
but the *source* of the recoverable 0.71 pt is withdrawn.

---

## 10. Suggested follow-ups (not implemented; outside my surface)

1. **Convert R1's dose-response into a price for coarse-metadata coarsening.** E79's
   g64 → g128/g256 lever saves head bytes by making the coarse screen noisier. My
   dose-response table turns that noise into a screen-loss cost directly: 1σ extra
   error → 0.190 % loss, 2σ → 1.025 %. The whole decision therefore reduces to
   measuring the sigma inflation from the coarser grouping, which is a cheap **offline**
   measurement on the head weights — no GPU, no timed leg. If the inflation is under
   ~1σ the lever is nearly free; if it is over 2σ it is not worth the bytes. This also
   composes with E144: a better quantizer and a coarser screen pull on the same
   error budget, and this table is the exchange rate between them.

2. **Settle the C-a census disagreement with alphonse cheaply.** The two numbers differ
   by 2.05× and the largest suspected term — corpus token frequency against live
   generated-trajectory frequency — has never been measured. Running both censuses on
   the *same* token stream is a zero-GPU afternoon and it decides whether his arm A is
   worth building. Section 2's reconciliation names the other two terms and their sizes.

3. **Fix the `MISS_TO_SCORE_PCT` disagreement.** 203 (contract) vs 209.5 ± 93.1
   (my E139 measurement) vs 290 (F2 §2's geometric model). Several campaign prices
   depend on this constant, and my own E139 interval is too wide to discriminate. A
   direct measurement — sweep realised acceptance and regress the leg time — would
   settle it in one session.

4. **One F3 arithmetic note, same class as Advisor Error 149.** F3 §3 gives essays'
   ceiling `x` as 0.830 % and the medicine upper-slot buffer as 0.825 %. Those are the
   same quantity; `3.89407 / 3.86219 − 1 = 0.8254 %`. I use 0.8254 %. It changes
   nothing here, but the F2 version of the same slip was worth recording.
