# E78 final result — NEGATIVE. The wide-switch group-count axis is closed.

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mtp_seconds_per_token","available":true,"value":0.031440739519894124},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}
```

Rung 3 is finished as written. I did not extend it and I did not open a new arm.
The result is a clean negative, and the reason is stronger than "the effect was
small": **the new campaign base already contains the only cell this experiment
found favourable.** Detail in section 8.

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e78-width-dependent-inner-group-count`
- Hypothesis: a QMV dispatch that conditions the inner-group count `IPG` on
  `out_vec_size` beats both our table and the crown's table.
- **Decision: dead. The hypothesis is refuted.**
- `BASE_SHA` `8d938c911df52b6a324f259a55dbaa75e508c822` / candidate commit `c3db3a79ccd8466e084fee21764d10f78da5af15`
- Yukon frontier at the time of measurement: crown `9ad17378` 3.25238228, source
  `bfab0de5`. The frontier has since moved to `c6af1e24` 3.30955573. Neither move
  touches my scored surface.
- **Submitted candidate files: none changed.** `git diff BASE..HEAD` on the two
  scored files is empty. `check-editable-budget.sh` reports `growth=0/262144`,
  `source=2469371/3000000`, `headroom=530629`. The branch ships arm A, which is
  the base byte for byte, exactly as a negative result should.
- Submitted-surface digests, unchanged from the base: `quantized.h`
  `71ab9a72965e727830fc35feaeefc628082ba22b9b4dd4b3cfc9a4ab066857f5`,
  `quantized.cpp`
  `c43a11f71495cec36589012a4ba950cb4d5f82d11cb6b9f525a977bbc34b8276`
- Supporting files, all research-only under `research/`.
- MTP head: organizer-pinned,
  `head_safetensors_sha256=d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`,
  offered depth 8, unchanged in every arm.
- Token window 512, fixture
  `correctness_prompts/public_longcopy_gate_english_512_256.json` sha256
  `3d922b1a0ada04d9827b905c881232bf50fb697d4be9ab3ee21346f7e0b8ae9c`, reference
  source = candidate-generated golden leg, **`harness=local` for every number in
  this report.**
- Exact cell: `affine_qmv_fast<bfloat16_t,64,4,false>` →
  `qmv_fast_crossrow_affine4_g64_m<T,M,NA,true>`, wide dispatch switch at
  `quantized.h:1922` behind `out_vec_size >= 4096`, JIT source form with a
  runtime-effective `mlx-generated/quantized.cpp` twin, M5 variant
  `applegpu_g17s` compiled and censused but not executed here.
- Official causal path: candidate-only. `senpai/verify-ranked-score-boundary.sh`
  → `PASS: ranked numerator is pinned baseline; candidate edits affect the MTP
  denominator only`.
- Scope preflight: `validate-assignment-scope.sh` flags only `research/` paths,
  which the assignment permits and Yukon does not submit. No scored path changed.

---

## 1. Headline

Arm `e_kdown` conditions `IPG` on **`in_vec_size`**, not `out_vec_size`, and
moves exactly one cell: `mlp.down` at M = 6 from IPG 6 to IPG 3. The rung 2a
cell table scored it as the best table any legal function of
`(M, in_vec_size, out_vec_size)` can build.

| quantity | value |
|---|---:|
| `a_ship` MTP seconds/token, 6 legs | **0.031470910840046905** |
| `e_kdown` MTP seconds/token, 6 legs | **0.031440739519894124** |
| delta | **-3.0171320152781733e-05 s/tok** |
| delta, per cent | **-0.0959 %** |
| session null | 2.9911519959568977e-05 s/tok (0.0950 %) |
| pre-registered threshold, 2 x null | 5.9823039919137955e-05 s/tok (0.1901 %) |
| \|delta\| / null | **1.009** |
| pre-registered prediction | -1.095197e-04 s/tok (-0.3480 %) |
| prediction / measurement | **3.630 x** |

Pre-registered pass conditions, committed in
`research/e78-artifacts/rung3-prereg.md` at `828c282` before the session opened:

| condition | verdict |
|---|---|
| sign is negative | **PASS** |
| \|delta\| >= 2 x session null | **FAIL** |
| \|delta\| within a factor of two of the prediction | **FAIL** |

Verdict recorded by the analyzer: **`not useful`**, `winners=[]`.

---

## 2. Ranked-pool re-weighting and the sign tests

Arm E moves one width, so the local-to-ranked transfer is one ratio of M = 6
weights. `research/e78_rung3_reweight.py`, output in
`research/e78-artifacts/rung3-reweight.md`.

| M | local rounds | local weight | ranked share % | ranked weight | arm E delta ms/round |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.0128 | - | - | +0.00000 |
| 4 | 5 | 0.0641 | 14.20 | 0.1464 | +0.00000 |
| 5 | 5 | 0.0641 | 24.10 | 0.2485 | +0.00000 |
| 6 | 23 | 0.2949 | 33.40 | 0.3443 | **-2.43800** |
| 7 | 4 | 0.0513 | 12.20 | 0.1258 | +0.00000 |
| 8 | 6 | 0.0769 | 7.35 | 0.0758 | +0.00000 |
| 9 | 34 | 0.4359 | 5.75 | 0.0593 | +0.00000 |

- local-mix weighted mean **-0.71890 ms/round**
- ranked-mix weighted mean **-0.83948 ms/round**
- ranked / local = **1.16773**
- **ranked-re-weighted measured estimate = -3.523188e-05 s/tok = -0.1120 %**,
  which is 1.178 x the session null and still **below** the 2 x threshold.

The local fixture shows seven distinct verify widths, not eight. I report all
seven and both weightings rather than pad the table.

**Sign tests.**

| test | statistic | p |
|---|---|---:|
| rung 2a contested cells, fewer groups faster | **23 of 24** | one-sided **1.490e-06**, two-sided 2.980e-06 |
| rung 3 legs, `e_kdown` faster than `a_ship` | U = **32 of 36** | exact one-sided **0.01299** |
| rung 3 position-matched pairs | **5 of 6** negative | 0.1094 |

Position-matched paired deltas, s/tok: -2.903e-05, +1.430e-05, -3.622e-05,
-7.376e-05, -1.571e-05, -4.060e-05.

**Read this honestly.** The rung-3 direction is real: the leg rank test
separates the arms at p = 0.013. What fails is the *size*. The effect is the
same order as the session null and 3.6 x smaller than the cell model predicted,
so it cannot be priced with confidence and it cannot survive a 1.069 % ranked
difference standard deviation. Sign alone is not a shippable result.

---

## 3. The per-cell table, including where the sign is unfavourable

Rung 2a, job `3796f162`, W&B `vp8q30dv`. 8 arms, 6 distinct shapes, 31 reps,
palindrome order, 2976 legs, 5 min 28 s, entry 40.51 C, exit 70.93 C. Rel-SEM
0.013-0.061 % per cell. Working threadgroups = `groups x ceil(n/8)`; the TG/core
columns describe the **fewer-group** dispatch.

| shape | k | n | M | few grp (IPG) | more grp (IPG) | few ms | more ms | split % | TG/core@20 | TG/core@40 | rel-SEM % | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| linear_attn.out_proj | 6144 | 5120 | 4 | 1 (4) | 2 (2) | 0.10435 | 0.13878 | +32.99 | 32.0 | 16.0 | 0.061 | split hurts |
| mlp.down | 17408 | 5120 | 4 | 1 (4) | 2 (2) | 0.28952 | 0.38360 | +32.49 | 32.0 | 16.0 | 0.035 | split hurts |
| linear_attn.out_proj | 6144 | 5120 | 5 | 1 (5) | 2 (3) | 0.12098 | 0.14814 | +22.45 | 32.0 | 16.0 | 0.059 | split hurts |
| mlp.down | 17408 | 5120 | 5 | 1 (5) | 2 (3) | 0.33437 | 0.40873 | +22.24 | 32.0 | 16.0 | 0.029 | split hurts |
| linear_attn.out_proj | 6144 | 5120 | 6 | 1 (6) | 2 (3) | 0.15379 | 0.15924 | +3.55 | 32.0 | 16.0 | 0.059 | split hurts |
| **mlp.down** | **17408** | **5120** | **6** | **1 (6)** | **2 (3)** | **0.47801** | **0.43992** | **-7.97** | **32.0** | **16.0** | **0.025** | **split HELPS** |
| linear_attn.out_proj | 6144 | 5120 | 9 | 2 (5) | 3 (3) | 0.20216 | 0.22898 | +13.27 | 64.0 | 32.0 | 0.050 | split hurts |
| mlp.down | 17408 | 5120 | 9 | 2 (5) | 3 (3) | 0.56457 | 0.63602 | +12.66 | 64.0 | 32.0 | 0.018 | split hurts |
| full_attn.qkv_proj_fused | 5120 | 14336 | 4 | 1 (4) | 2 (2) | 0.21121 | 0.29854 | +41.35 | 89.6 | 44.8 | 0.016 | split hurts |
| full_attn.qkv_proj_fused | 5120 | 14336 | 5 | 1 (5) | 2 (3) | 0.24981 | 0.31840 | +27.46 | 89.6 | 44.8 | 0.013 | split hurts |
| full_attn.qkv_proj_fused | 5120 | 14336 | 6 | 1 (6) | 2 (3) | 0.31967 | 0.34320 | +7.36 | 89.6 | 44.8 | 0.020 | split hurts |
| linear_attn.in_proj_fused_qkvzba | 5120 | 16480 | 4 | 1 (4) | 2 (2) | 0.24011 | 0.34086 | +41.96 | 103.0 | 51.5 | 0.026 | split hurts |
| linear_attn.in_proj_fused_qkvzba | 5120 | 16480 | 5 | 1 (5) | 2 (3) | 0.28419 | 0.36339 | +27.87 | 103.0 | 51.5 | 0.017 | split hurts |
| linear_attn.in_proj_fused_qkvzba | 5120 | 16480 | 6 | 1 (6) | 2 (3) | 0.36402 | 0.39188 | +7.65 | 103.0 | 51.5 | 0.018 | split hurts |
| full_attn.qkv_proj_fused | 5120 | 14336 | 9 | 2 (5) | 3 (3) | 0.43955 | 0.50588 | +15.09 | 179.2 | 89.6 | 0.020 | split hurts |
| linear_attn.in_proj_fused_qkvzba | 5120 | 16480 | 9 | 2 (5) | 3 (3) | 0.50244 | 0.57888 | +15.21 | 206.0 | 103.0 | 0.015 | split hurts |
| mlp.gate_up_fused | 5120 | 34816 | 4 | 1 (4) | 2 (2) | 0.48605 | 0.70271 | +44.57 | 217.6 | 108.8 | 0.017 | split hurts |
| mlp.gate_up_fused | 5120 | 34816 | 5 | 1 (5) | 2 (3) | 0.57776 | 0.74871 | +29.59 | 217.6 | 108.8 | 0.018 | split hurts |
| mlp.gate_up_fused | 5120 | 34816 | 6 | 1 (6) | 2 (3) | 0.74329 | 0.80809 | +8.72 | 217.6 | 108.8 | 0.018 | split hurts |
| mlp.gate_up_fused | 5120 | 34816 | 9 | 2 (5) | 3 (3) | 1.03969 | 1.20291 | +15.70 | 435.2 | 217.6 | 0.020 | split hurts |
| head.lm_head | 5120 | 248320 | 4 | 1 (4) | 2 (2) | 3.35520 | 4.91392 | +46.46 | 1552.0 | 776.0 | 0.021 | split hurts |
| head.lm_head | 5120 | 248320 | 5 | 1 (5) | 2 (3) | 4.00255 | 5.23523 | +30.80 | 1552.0 | 776.0 | 0.019 | split hurts |
| head.lm_head | 5120 | 248320 | 6 | 1 (6) | 2 (3) | 5.16470 | 5.65392 | +9.47 | 1552.0 | 776.0 | 0.019 | split hurts |
| head.lm_head | 5120 | 248320 | 9 | 2 (5) | 3 (3) | 7.35521 | 8.52685 | +15.93 | 3104.0 | 1552.0 | 0.014 | split hurts |

**The design premise is falsified inside this table.** At M = 6 the two n = 5120
shapes want opposite group counts: `mlp.down` (k = 17408) gains 7.97 %,
`linear_attn.out_proj` (k = 6144) loses 3.55 %. They share `out_vec_size`, so
**no cutoff on `out_vec_size` can separate them.** The separating variable is
`in_vec_size`.

---

## 4. Evidence chain, rung by rung

**Rung 0 - passed, no GPU.** The `NA <= 6` relaxation is inert, proven more
strongly than the assignment asked: `b_crown` and `b_crown_exact` compile to
byte-identical GPU objects on both architectures (g17s 91 regs / 126984 B / sha8
`846d5999`; g16s 94 regs / 121072 B / sha8 `f54263e7`). Register census against
the ceilings: `a_ship`, `c_hybrid24928` and `d_hybrid8192` all 111 g17s / 96
g16s with a 16 B g16s spill; `b_crown` 91 / 94 with no spill.
`stop_rule.passed=true`, cell drift 0, ceiling breach 0. Artifact `rung0.json`.

**Rung 1 - passed.** Job `8541fc9e`. Golden leg
`mtp_seconds_per_token=0.031591226579621434`, serial 0.073340984, speedup
2.3216, `all_tokens_matched=true`. `a_ship`, `c_hybrid24928` and `d_hybrid8192`
produced **byte-identical row ledgers**, 220574 B, sha256
`a1044f14dcd6d24d7259357b54eda48170321b8fd7943746a84f6468a04acc98`. 567 declared
rows, 512 tokens, 434 accepted and 55 rejected drafts, `parity_all_ok`, EOS at
rows 308 and 314 with correct post-EOS continuation. Width census: rounds
`{2:1,4:5,5:5,6:23,7:4,8:6,9:34}` = 78; rows
`{2:2,4:20,5:25,6:138,7:28,8:48,9:306}` = 567.

**Positive control fired.** `c_perturb`, a rows-3/4 lane swap inside NA = 5
groups, was correctly rejected: `rejected_tail_diverged` at step 4, round 1, row
1, declared top-1 198 against reference 28286, margin 0.3125, `verify_exit=1`,
0-byte ledger. The exactness gate can fail, so its passes mean something.
Artifact `exactness.json`, `passed=true`.

**Rung 2a - the deliverable.** Section 3.

**Rung 2b - additivity cross-check, passed pre-registration.** Job `9cd7f300`,
exit 0, 5529 s, 17 legs.

| arm | predicted delta s/tok | measured delta s/tok | measured % | ratio | session null |
|---|---:|---:|---:|---:|---:|
| `a_ship` | 0 | 0 (0.031472862348891795) | 0 | - | 3.867e-6 |
| `d_hybrid8192` | +3.873e-4 | +4.638e-4 | +1.4737 | 1.20x | 1.070e-5 |
| `c_hybrid24928` | +8.262e-4 | +8.743e-4 | +2.7780 | 1.06x | 2.660e-5 |
| `b_crown` | +1.925e-3 | +1.918e-3 | +6.0942 | 1.00x | 3.956e-5 |

Rank order `a < d < c < b` exactly as predicted; the `b_crown` prediction was
accurate to 0.4 %. The width histogram, `effective_mean_draft_len` 6.269230769
and `accepted_draft_rate` 0.8875255623721882 were identical across all four
arms. Eight timed groups entered within a 0.61 C spread.

**Rung 3 - the terminal measurement.** Job
`6e4066c2-ef4f-4345-ab18-a0fca33b6811`, `session_status=0`, all 13 legs
completed, 16:05:20 -> 17:10:41 UTC, order `warm a1 e1 e2 a2` at 3 legs per
timed group. Sections 1 and 2.

---

## 5. Where the model broke, stated plainly

**Additivity holds for large effects and fails for this one.** Rung 2b confirmed
the cell model to within 0-20 % on effects of 1.5 % to 6.1 %. On the 0.35 % arm-E
effect the same model over-predicts by **3.63 x**.

My best explanation: `mlp.down` has **272 k-blocks**, 3.4 x more than any other
scored shape. It is therefore the cell most sensitive to weight residency. The
rung 2a harness streams about 12 GB per cell to defeat the cache, so it measures
the cold-ish cost of an extra weight pass. In situ, inside a real round,
`mlp.down`'s weights are far more resident, so the second group's extra pass
costs much less than the harness says and the saving from splitting shrinks. The
large effects are dominated by grid-shape terms that residency does not touch,
which is why they transferred and this one did not.

**This is a limitation of my instrument that I did not anticipate and should be
recorded as such.** An isolated-cell harness that defeats the cache
systematically over-states the benefit of splitting a high-k-block shape.

---

## 6. Controls that failed, and what they cost the campaign

**The E33 positive control does not replicate.** You asked whether the M = 6 sign
flip still sits between n = 14336 and n = 16480. It does not. Both probes are on
the "split hurts" side by nearly the same margin: `full_attn.qkv_proj_fused`
+7.36 % (TG/core 89.6), `linear_attn.in_proj_fused_qkvzba` +7.65 % (TG/core
103.0). Rel-SEM 0.020 % and 0.018 %, so these sit roughly 370 standard errors
from zero. This is not a resolution problem. It is the **third consecutive
knee-model control failure** (E74 predicted 2719 and observed a gate between 3584
and 4120).

**A single threadgroups-per-core threshold cannot reproduce the observed signs.**
The one helper sits at 32.0 TG/core@20 alongside **five** cells at the identical
32.0 that want fewer groups, including `linear_attn.out_proj` at the same M = 6
and the same n = 5120. Any threshold that catches the helper mis-classifies those
five. "Never split" scores 23 of 24; every non-trivial threshold scores worse.
Bounding the knee: **< 32.0 TG/core at 20 cores, that is < 640 working
threadgroups**, not resolvable further because n = 5120 is the smallest scored
QMV shape. That **contradicts the E33 reading of (89.6, 103.0) by at least 3 x**.

**Cross-instrument agreement is nevertheless excellent.** My rung 2a and
Thorfinn's E75 rung B, two harnesses on two g16s hosts:

| width | Thorfinn E75 rung B | my rung 2a |
|---|---:|---:|
| M = 5 | +25.43 % | +26.86 % |
| M = 6 | +4.44 % | +3.68 % |
| M = 9 | +13.38 % | +14.69 % |

So the **g16s -> g17s sign inversion is real and doubly corroborated**. On this
axis a local cell table on a g16s host does not predict the ranked sign. Two
independent, precise, mutually consistent local instruments both get the ranked
direction wrong.

---

## 7. Corrections to the ledger

1. **The M = 6 flip is `mlp.down` alone, not "the three n = 5120 families."**
   `linear_attn.out_proj` at n = 5120 is 3.55 % *slower* when split at M = 6.
   `research/e78_arms.py` carried the E33 premise and I corrected it.
2. **n = 98336 `head.compact_draft_vocab` is in no ladder step.** It runs at
   bits = 2 and M = 1, so it never reaches the `bits == 4` gate this table lives
   behind, and it must never be added to the 4-bit cell harness. It contributes 0
   calls.
3. **Arm A is a proven no-op**, not a stale table. `a_ship` regenerates to
   `71ab9a72...`, byte-identical to the base file.
4. **The pre-E55 digest pair `75d45143...` / `350de468...` is stale.** Commit
   `b757237` moved `quantized.h` to `71ab9a72...`. The stale pair survives in
   `research/e44-prereg.md`, `e48-artifacts/analysis.json`,
   `e49-artifacts/e49-metrics.json` and `e54-artifacts/e54-metrics.json`.
5. **The hybrid arms cannot separate occupancy from grid width.** Both keep
   `<T,6,6>`, so they inherit arm A's 111 g17s registers and 96 g16s registers
   plus 16 B spill and never receive the crown's register relief. E77 argues
   occupancy is too small to matter, but this arm design does not test it.
6. **Arm A's raw numbers carry a g16s-only 16 B spill worth 0.1-0.24 %.**
   Reported raw and uncorrected, per your instruction. I did not attempt
   `lazyfall`.
7. **A comment in the QMV patch breaks the promotion gate.**
   `research/twin_audit.py` pins its `quantized.h` waiver to the sha256 of that
   section's comment text. Every arm patch is comment-free; the rationale lives
   in `research/e78_arms.py`. Verified: twin audit OK, exactly one pre-existing
   waiver.

---

## 8. Why this closes the axis, not merely this arm

This is the decisive argument and it does not depend on the rung-3 magnitude at
all.

The one cell where my instrument says more groups help is **`mlp.down` at M = 6
-> IPG 3**. At `6acb0d15` you reverted both scored files to the organizer's
table, so the base is now `<T,5,3>`, `<T,6,3>`, `<T,9,3>`. **That base already
uses IPG 3 at M = 6 for every shape, so it already splits the one cell arm E
wanted to split.**

Applied to the new base, arm E is no longer "split `mlp.down`". It becomes
"**un-split** the other five M = 6 shapes and leave M = 5 and M = 9 at IPG 3".
That is precisely the direction ranked receipt `9b241879` rejects: 8 of 8
prompts, mean -0.383 %, sign test p = 0.0039.

**There is nothing left to ship on the wide-switch group-count axis.** Every
legal table this design space can build is either the new base itself or a move
toward our own ranked loser.

One coherence check worth keeping: the sign of the single winning cell **agrees**
with the ranked direction, while the other 23 cells disagree. The g16s instrument
is not random with respect to the ranked host; it is systematically wrong except
at the one shape the ranked host also wants split. That is a compact statement of
why local screening failed on this axis, and it is a reason to distrust g16s cell
tables for group-count decisions specifically, not to distrust cell tables in
general.

---

## 9. Evidence and identity

- Host `ip-10-231-2-227.ec2.internal`, Apple M4 Pro, 20 GPU cores, 48 GiB,
  `applegpu_g16s`. Ranked host is M5 Max / `applegpu_g17s` / 40 cores. macOS
  26.5.2, Apple Swift 6.3.3 (swiftlang-6.3.3.1.3, clang-2100.1.1.101). Memory
  profile `full`, `mlx_max_mb_per_buffer=512`, `mlx_max_ops_per_buffer=50`,
  `wired_residency_active=false`.
- **Thermal policy: program-permitted ungated ABBA.**
  `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`,
  preserved verbatim in every artifact. `official_or_ranked_score=false`.
- **Why ungated.** I probed the real 40 C gate once (job `c42ca35f`): it started
  at 42.9 C, asymptoted at 40.1 C and aborted after 340 s. Idle GPU floor on this
  host measures 39.49 C against a 40 C gate, and a leg exits near 62-65 C. The
  real gate is unusable here. All three ungated conditions hold: arms are
  ABBA-counterbalanced inside one session, entry and exit temperatures are
  recorded per group, and the two flags are preserved.
- Rung 3 per-group thermal, entry -> exit GPU: `a1` 50.759 -> 62.783, `e1` 51.481
  -> 63.063, `e2` 51.466 -> 62.252, `a2` 50.953 -> 62.679. **Entry spread across
  the four timed groups: 0.72 C.** The discarded warm-up entered at 44.585 C.
- Per-arm built-worker sha256: `a1` `195aebdf...`, `e1` `25da3897...`, `e2`
  `eec59543...`, `a2` `47b97b8a...`. Metallib source fingerprint `f09821bd...`
  for `a_ship`, `1198ae41...` for `e_kdown`. Arm source digests (`quantized.h`):
  `a_ship` `71ab9a72965e7278`, `b_crown` `3043807bd239241b`, `c_hybrid24928`
  `e0a6d3d78b95b719`, `d_hybrid8192` `49e47f3863a15131`, `e_kdown`
  `30e74c6451f00b78`. `leg*_worker_unchanged_across_leg=true` on all 13 legs.
- Reproduction: `python3 research/e78_arms.py --emit <arm>` then
  `research/e78_rung3.sh --ungated --legs 3`, then
  `python3 research/e78_analyze.py --rung 3 --out research/e78-artifacts/rung3.json`,
  then `python3 research/e78_rung3_reweight.py`.
- Runtime: rung 2a 5 min 28 s; rung 2b 5529 s; rung 3 3921 s. No OOM, no crash,
  no timeout. The one non-zero exit in the campaign was my analyzer's print bug
  (`ladder` tuple shadowing) after all 13 legs had already completed; it is fixed
  in `c3db3a7` and the artifact regenerates at exit 0.
- Peak RAM: unchanged, the memory profile is identical across arms and the change
  is a dispatch template selection.
- Generated-twin audit: `TWIN AUDIT OK: 29 runtime-effective twin(s), 1
  allowlisted comment-only waiver(s)`.
- **No `--local-submit` run and no Yukon submission.** The candidate surface is
  byte-identical to the base, so there is nothing to submit. Running the submit
  chain on an unchanged surface would be readiness theatre, not evidence.

| Metric | Baseline `a_ship` | Candidate `e_kdown` | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.07326851723094781 | 0.07327031577005982 | +0.0025 % |
| **MTP seconds/token** | **0.031470910840046905** | **0.031440739519894124** | **-0.0959 %** |
| local serial-relative speedup | 2.3281356135563507 | 2.3304263817244775 | +0.0984 % |
| effective mean draft length | 6.269230769230769 | 6.269230769230769 | identical |
| accepted draft rate | 0.8875255623721882 | 0.8875255623721882 | identical |
| `all_tokens_matched` | true, 6/6 legs | true, 6/6 legs | - |
| width histogram | `{2:1,4:5,5:5,6:23,7:4,8:6,9:34}` | identical | - |

Every compared identity field matched except the one dimension under test: base
SHA, fixture and digest, head and digest, token window, host, chip, toolchain,
memory profile, thermal mode, offered depth and harness are the same for both
arms. The only difference is the QMV dispatch table.

The local serial-relative speedup is quoted as a secondary only. Both local legs
use the candidate binary, and this change is confined to the candidate MTP leg,
so the ratio is directionally usable here - but the primary and the decision are
**absolute candidate seconds per token**. Neither number is a ranked score. The
ranked re-weighting in section 2 is a **derived** re-weighting of one measured
local number, labelled as such, and it never enters a ranked score equation.

---

## 10. W&B runs

Project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`. All 15 runs are
`finished`.

**Rung 3, group `e78-rung3-kdown`** - the terminal evidence:

| run | arm | id |
|---|---|---|
| `e78-warm` (discarded) | `a_ship` | `x3xonel6` |
| `e78-a1` | `a_ship` | `zirxq2b3` |
| `e78-e1` | `e_kdown` | `uao1nk2g` |
| `e78-e2` | `e_kdown` | `1zolbc8a` |
| `e78-a2` | `a_ship` | `6qmrl06s` |

**Rung 2a**: `vp8q30dv` (`e73-rung1-e78-rung2a`).

**Rung 2b, group `e78-width-dependent-inner-group-count`**: `ut1fgyaj` (warm),
`tckwkh54` (a1), `0kliz8mn` (b1), `eu42wktv` (c1), `nsbu2ff0` (d1), `q9aw8bt7`
(d2), `v5nqgkkz` (c2), `fbtgiifc` (b2), `7tbqmtkc` (a2).

Artifacts on this branch: `rung0.json`, `exactness.json`,
`rung2a-cells{,-raw}.json`, `rung2a-cells.md`, `rung2a-arms.{json,md}`,
`rung2a-percore.json`, `rung2a-config.json`, `rung2a-session.log`,
`rung2b-prereg.md`, `rung2b.json`, `rung2b-session.log`, `rung3-prereg.md`,
`rung3.json`, `rung3-session.log`, `rung3-reweight.md`.

---

## 11. Conclusion

- **What happened.** The hypothesis is refuted twice over. No cutoff on
  `out_vec_size` can express the optimal table, because two shapes share
  `out_vec_size = 5120` at M = 6 and want opposite group counts. The
  `in_vec_size` cutoff that *can* express it moves one cell, and that cell's
  in-situ benefit is 3.6 x smaller than the isolated-cell instrument predicted
  and indistinguishable in size from the session null.
- **Evidence for the mechanism.** Grid width does drive the group-count
  trade-off: the split penalty falls monotonically with M at every shape (for
  example `head.lm_head` +46.46 % at M = 4 down to +9.47 % at M = 6) and rises
  again at M = 9 where the group count itself changes. The mechanism is real.
  What is refuted is that any *legal cutoff function on the wide switch* can turn
  it into a win on this base.
- **Transfer risk.** Maximal, and now measured rather than assumed. My instrument
  gets the ranked sign wrong on 23 of 24 cells on this axis. I would not advance
  any wide-switch group-count arm on local evidence again without a ranked
  receipt.
- **Recommendation: close.** Record the cell table, the per-core boundary and the
  instrument limitation in the ledger. Do not reopen the wide switch unless the
  ranked host itself produces a per-shape measurement, or unless a promotion
  changes the base table again.

### Suggested follow-ups, which I did not implement

**1. The narrow switch at `quantized.h:1980`, as you asked.** You called this out
and I agree it is the more interesting remaining dispatch surface, for one
structural reason: it serves proposal-head shapes only - head qkv 3 x 1024 = 3072
and the per-committed-token K/V pack at 2048 - so it is **MTP-only by
construction and cannot touch the ranked serial numerator**. Every 1 % saved
there is 1 % of the denominator with no cancellation risk. What I would measure,
in order:

- **First, prove the call path and cost, before any kernel work.** Count
  dispatches per round and measure the wall share of `n = 3072` and `n = 2048` in
  situ. `5c542728` shows routing M = 1 through the crossrow kernel there is exact
  but worth nothing, which is consistent with the share being small. If the
  narrow switch is under about 1 % of the candidate leg, stop there and report
  it. I would not repeat E78's error of optimizing a surface before pricing it.
- **Second, note that these shapes are far below every knee I can measure.** At
  n = 3072 one group gives 384 working threadgroups, 19.2 per core at 20 and
  **9.6 at 40**; n = 2048 gives 256, that is 12.8 and **6.4**. Both are below my
  bound of < 32.0 TG/core@20, so this is the **only** scored region where the
  starvation model actually predicts that splitting helps, and where it predicts
  the ranked host benefits *more* than mine. That is the opposite of the wide
  switch, and it is the single reason I would consider a local screen credible
  there - the local and ranked predictions agree in sign instead of inverting.
- **Third, measure M and IPG jointly, not IPG alone.** At these grid sizes a
  single threadgroup cannot fill the machine at any M, so the interesting lever
  is not "how many inner groups" but "can the head's several small matmuls be
  issued together". Which leads to:
- **Fourth, and I think this is the higher-value question: count dispatches, not
  bytes.** Your `qwen35DualRMSNorm` finding prices a removed dispatch at about
  **0.35 ms**, one to two orders of magnitude above a Metal launch, and it bought
  +1.647 % of median from two 50 KB norms. That is host-side graph and
  synchronisation cost. The proposal head issues several small QMV dispatches per
  round on exactly this narrow switch. **Fusing or batching those dispatches is
  likely worth far more than choosing a better `IPG` for any of them**, and it is
  measurable with a dispatch count and a round-latency histogram before a single
  kernel line is written. If I get one more assignment on this surface, that is
  the one I would want.

**2. Re-audit every isolated-cell price for high-k-block shapes.** My harness
streams about 12 GB per cell to defeat the cache and therefore over-states the
cost of an extra weight pass for shapes with many k-blocks. `mlp.down` at 272
k-blocks is the extreme case and is the one that mis-priced by 3.6 x. Any past or
future campaign number derived from that harness for a high-k-block shape should
carry a residency caveat.

**3. Retire the threadgroups-per-core knee as a predictive model on this host.**
It has now failed three consecutive positive controls (E74 gate, E33 flip, and
the 32.0 TG/core collision here). It remains a useful descriptive summary. It
should not be used to choose an arm.
