# E45 — pooled plateau trees cannot separate the step-at-6 and quadratic families

**Stopping early, as instructed: pooling the plateau rows does not separate the
two families, and the reason is structural rather than a shortage of data.** At
every admissible depth mixture the two families' regression designs span the
*same* column space, so they produce identical predictions and identical
residuals while implying `T(6) − T(5)` values that differ by tens of
milliseconds. Worse for the pooling idea specifically, the work-identity filter
that makes pooling legal also forces every pooled tree to carry the *same*
regressor column and the *same* admissible shape box as our own row, so pooling
adds replication in `y` and no new identifying variation at all. No tree count,
no tolerance, and no reduction in replicate noise closes that gap.

The one asymmetry that survives is not pooling: it is the **monotonicity cone**.
The two parameterisations span the same space but occupy different admissible
orthants of it, and that is the only lever in the current evidence that
distinguishes them. It is available on a single row and does not need the pool.

Everything below is `research/`-only analysis of already-published Yukon rows.
**Zero GPU seconds were used.**

---

## 1. Deliverable (a) — the pool, with tree-identity dedup

Six candidate plateau rows were requested. They resolve to **five distinct git
trees**, so the pool actually used is **our row plus five trees, contributing 47
legs**, not six trees and 48 legs.

| row | solver | published score | git tree (16 hex) | legs used |
|---|---|---|---|---|
| `ca9251b8` | `morganmcg1` (ours) | 3.2325084826 | `5862ec15d14537cf` | 8 |
| `0cd0a6b4` | ofou (board crown) | 3.2492939855 | `2d65604a66d48956` | 8 |
| `b0994092` | fkiene | 3.2441790000 | `6e7e83f57f39a864` | 8 |
| `3ac231d5` | Lieisyourlie | 3.2438790000 | `9abb1c8f6a822b31` | 8 |
| `11863aa9` | companygardener | 3.2432620000 | `b8642b81f72ff921` | 8 |
| ~~`4f76de6e`~~ | ~~alfranli123~~ | ~~3.2430010000~~ | `b8642b81f72ff921` | **dropped** |
| `de7981ae` | WillGasser | 3.2407780000 | `bdad982d6ec3f6f0` | 7 |

* **Tree-identity dedup.** `11863aa9` (companygardener) and `4f76de6e`
  (alfranli123) are the *same source tree* `b8642b81f72ff921` submitted twice.
  Treating them as two independent observations would have double-counted one
  piece of evidence. The rule is deterministic — keep the higher published score
  — so `11863aa9` (3.243262) is retained and `4f76de6e` (3.243001) dropped.
  Tree SHAs are resolved from the GitHub commit API and cached under
  `.mlxfast-private/` so the count is reproducible without re-querying.
* **Leg accounting.** `47 = 8 (ours) + 8 + 8 + 8 + 8 + 7`. WillGasser's
  `plutarch` leg is *not* work-identical to ours, so it is excluded rather than
  forced onto our reading; the other seven of its legs are kept. Seven prompts
  are shared by all six trees; `plutarch` is shared by five of them.
* **Why work identity is required.** A leg enters the pool only when
  `effective_mean_draft_len` and `non_drafting_round_count` match ours exactly
  (compared as the published 16-digit float and as an integer). That is what
  licenses treating the depth mixture `rho_p` as shared.

Corpus: 656 submissions, of which **110** rows declare per-prompt MTP head
telemetry. Our row's solver label on the board is `morganmcg1`, not `senpai`.

### Replicate noise, measured from the pool

Same-tree, same-prompt MTP-leg replicates give a **relative sd of 0.0683 %**
over 8 leg pairs (`plutarch` 0.0274, `drama` 0.0060, `travel` 0.0662, `beagle`
0.1144, `medicine` 0.0845, `essays` 0.0183, `republic` 0.1068, `botany`
0.0126 %). This is tighter than the 0.281 % pair noise E43 assumed, and tighter
than the advisor's 0.0995 % MTP replicate sd. Using the *smaller* number makes
the non-separation conclusion harder to reach, so it is the conservative choice
here.

---

## 2. Pooling adds exactly zero identifying information

### 2.1 Empirically: the admissible reading set does not shrink

Re-running the E43 exact enumeration at the E43 headline tolerance
(`0.00562`, single-shape step-at-6 family):

| enumeration | selections | nodes visited |
|---|---|---|
| our row alone | **42** | 7070 |
| all six plateau rows pooled | **42** | 7014 |

The 42 selections are **identical as sets**. Pooling prunes nothing. The E43
pins are reproduced exactly (`beagle` 107, `medicine` 99, `essays` 87,
`republic` 89, `botany` 85) along with the surviving free readings `plutarch`
[461, 474, 487, 500], `drama` [252, 289, 299, 336], `travel` [151, 212, 273].
Both endpoints of the `T(6) − T(5)` bracket also survive pooling: every one of
the six trees is feasible at both endpoint shapes.

### 2.2 Structurally: the pooled design has the same columns as one row

The pooled model is, for tree `t` and prompt `p`,

```text
y_tp = a_t + b_t * x_tp + s_t * g_p
```

with a per-tree intercept/slope/scale and **one shared shape value `g_p` per
prompt**. Family B is indistinguishable from family A exactly when
`u_p = g_B,p − lambda * g_A,p` lies in `span{1, x_t.}` for *every* pooled tree.
Two measured facts settle it:

1. **`x` is bit-identical across all pooled trees** (max gap `0` mean-`M`
   units). This is not a coincidence: `x_p = E[M]_p = n_p + 1`, and work
   identity is *defined* by `n_p` matching exactly. So `span{1, x_t.}` is the
   same 2-plane for every tree, and pooling contributes no regressor variation.
2. **The pooled admissible shape box equals our single row's own bracket**
   (max gap `0`). Same reason: `rho` depends only on the same two matched
   integers, so intersecting the trees' brackets removes nothing.

So the pooled identification problem is *literally* our single row's
identification problem with extra `y` replicates.

### 2.3 The two families are exactly collinear inside the admissible box

Two linear programs, both feasible:

| test | free parameters | feasible? | `lambda` range |
|---|---|---|---|
| single row, `x` column free | `alpha, beta, lambda>0` | **yes** | [1.697, 10⁴ (box)] |
| pooled, offset only (strictest) | `alpha, lambda>0` | **yes** | **[44.159, 129.925]** |

The second row is the version pooling could have broken — it forbids a per-tree
slope from absorbing the offset — and it is still feasible. At a witness inside
that range:

* pooled unconstrained worst relative miss, step-at-6: `0.0294774177`
* pooled unconstrained worst relative miss, quadratic: `0.0294774177`
* gap: `2.75e-14` (predictions agree to `1.8e-12` ms; replicate sd is
  `6.8e-4` in the same units — eleven orders of magnitude larger)
* implied `T(6) − T(5)` at that identical misfit: **differs by up to 69.97 ms**

On the single row the same demonstration gives `pred_gap = 2.84e-14 ms`,
`map_gap = 6.4e-14`, residual difference `3.7e-16`, with `Δ = 54.35` vs
`7.48 ms`, and the map is verified to be exact (family B's fitted coefficients
are reproduced from family A's by the closed-form transform). It is feasible at
**all 42** surviving readings, and also for `step5`↔`step6`, `step6`↔`step7`,
`step5`↔quadratic and `step7`↔quadratic.

**The misfit that pooling would have to detect is zero, not small.** Minimum
detectable effect is therefore unbounded at any tree count. That is the answer
to the assignment's central question.

### 2.4 The one real asymmetry: the monotonicity cone

Same column space, different admissible orthant. Projecting each fit onto its
own family's monotonicity cone (`b ≥ 0, s ≥ 0` for a step; `c ≥ 0, b + 3c ≥ 0`
for a quadratic) at the ray witness gives:

| family | pooled worst-rel after cone projection | trees clamped |
|---|---|---|
| step-at-6 | 0.256576 | 5 of 6 |
| quadratic | 0.029477 | 0 of 6 |

This is the only mechanism in the current evidence that distinguishes the
families, it favours the quadratic reading, and **it does not need the pool** —
it is a property of the parameterisation, available on one row. Note the number
above is at one arbitrary point of the ray, not each family's best shape; the
best-shape comparison is in §5.

---

## 3. Deliverable (c) — the cross-family bracket on the 5→6 increment

`Δ = T(6) − T(5)`, in ms per round, at the E43 primary reading and tolerance:

| admissible family | `Δ` bracket (ms) |
|---|---|
| linear | **infeasible** |
| step at 5 | [0.0000, 3.5503] |
| step at 6 | [18.2019, 80.4828] |
| step at 7 | [0.0000, 3.8505] |
| quadratic | [6.8681, 10.0666] |
| mixture (step-at-6 + quadratic) | [6.8681, 80.4828] |

* Step-at-6 and quadratic are **disjoint** (10.0666 < 18.2019). E43's
  inconclusive verdict was not timidity: the two families genuinely disagree.
* Taking the union over **admissible families and mixtures**, not just over the
  nuisance `s`, the increment bracket is
  **[0.0000, 84.1060] ms** — it **contains zero**. The 5→6 increment is not
  identified at all by the published telemetry: nothing rules out a step at 5 or
  at 7, under which the 5→6 increment can be zero.
* Widening over all 42 surviving readings within the step-at-6 family alone
  gives [15.8466, 84.1060]; the cross-family lower endpoint comes from the
  `step5` / `step7` boundary families.
* Both advisor point estimates sit inside their own family's bracket
  (37.730 in step-at-6, 8.575 in quadratic) and inside the cross-family union.

**Flag on the local ladder.** The `+32.850 ms` local 5→6 target from E38 comes
from a **dropped (pre-rebase) tree** and is not used as evidence anywhere in
this analysis. It is quoted once, here, only so nobody re-imports it as
load-bearing. It happens to fall inside the step-at-6 bracket and outside the
quadratic one, but a measurement from a tree that is no longer on the campaign
base cannot arbitrate between families.

---

## 4. Deliverable (b) — excess and score value under both families

### 4.1 Excess per prompt

Two anchors are reported because they answer different questions:

* **`whole`** — the entire shape term at the leg's depth.
* **`secant`** — only the part above the straight line through depths 1..5, i.e.
  the excess a depth-limiting fix could actually remove.

Findings:

* For **step-at-6** the two anchors are **identical**: a step at 6 has no
  sub-six term, so every millisecond of the shape term is above the secant.
* For **quadratic** the secant is strictly below the whole term at every leg,
  and is **negative at low depth** (`drama` lower endpoint `−3.0736` ms).
  A quadratic penalty is partly already paid inside depths 1..5, so attributing
  all of it to "deep rounds" overstates the recoverable excess.

Reporting only the `whole` anchor would therefore bias the quadratic family's
apparent upside upward. The `secant` anchor is the conservative one and is the
one used for the value table.

### 4.2 Score value under both families

Base score reproduced from the published legs: **3.2325084826** (matches the
board exactly). Crown `0cd0a6b4` is 3.2492939855, a gap of **0.5193 %**.

| family | fraction of its own excess that must be removed to reach the crown |
|---|---|
| step at 6 | **0.011135** (reproduces the E43 figure `0.011134589`) |
| quadratic | **0.023501** |

So the two families disagree by more than a factor of two on how much of the
identified excess a fix must capture — the same ambiguity as the `Δ` bracket,
expressed in score units.

### 4.3 A family-free anchor

Because the family choice is not identified, the most useful number is one that
does not depend on it. Working directly in **leg time fraction** (how much of
each MTP leg's wall time must be removed, irrespective of *why* it is spent):

| target | leg fraction needed |
|---|---|
| one replicate sigma (0.0978 % of score) | **0.000977** |
| the crown gap (0.5193 %) | **0.005166** |

* Only **0.52 % of MTP leg time** needs to be removed to take the crown. That
  is a small, family-independent target.
* The median-of-eight is **saturating**: fixing only the two worst-ratio prompts
  (`beagle` + `medicine`) at a leg fraction of `0.0806` already yields
  **+4.5649 %** and then stops improving, because the median stops moving once
  those two prompts cross their neighbours. Both families' value arms hit the
  same cap. Any fix that concentrates on a couple of prompts therefore runs into
  the median, not into the physics.

---

## 5. Pooled shared-shape threshold

Reported from both sides so the two families are compared as bounds, not as
point estimates:

* **Lower bound** (`family_threshold`) — the smallest per-leg slack at which the
  family can explain every pooled tree, with the shared-shape coupling dropped.
  Sound lower bound.
* **Upper bound** (`search_shape` + `certify_witness`) — an explicit shared shape
  vector and per-tree cone-legal parameter vectors whose worst relative miss is
  the reported value. Sound upper bound.

At the E43 primary reading the step-at-6 pooled upper bound is **0.014516**
(certified), which is *above* the E43 headline tolerance of 0.00562. The
per-family threshold table produced by `--run` is in
`research/e45-pooled-family.json` under `threshold`.

Read this carefully: a pooled upper bound above the tolerance means the *shared*
`rho` assumption plus that family plus that reading cannot jointly hit 2 sigma —
which is a statement about the conjunction, not about the family alone. Since
§2.3 shows the two families have identical unconstrained misfit at a shared
admissible `rho`, any pooled threshold difference between them comes from the
cone projection, i.e. from §2.4, not from the pooling.

---

## 6. Corrections to the campaign record

1. **Five trees, not six.** The plateau is six *rows* but five distinct source
   trees; `11863aa9` and `4f76de6e` are the same tree. Any "six independent
   plateau submissions" statement double-counts.
2. **E43's `s` bracket upper endpoint is `80.482786128094`,** not
   `80.48305253958134` as recorded. My generic N-shape layer reproduces E43's
   hand-written `step_polytope` + `lp_extreme` result to floating-point identity
   (`< 1e-12`), and the `q` brackets agree exactly (max gap `0`), so this is a
   transcription error in the record rather than a modelling difference.
3. **The advisor's residual ratio `1.174289` is the rms-|residual| ratio**
   (quadratic / step-at-6), not the max-relative-residual ratio, which is
   `1.485036`. Both are well inside any reasonable "inconclusive" band; the
   distinction matters only if someone quotes the number as a fit-quality
   threshold.
4. **`+32.850 ms` belongs to a dropped tree** (E38, pre-rebase) — see §3.
5. **Our board solver label is `morganmcg1`.** Row `ca9251b8`.

---

## 7. Assumptions added, and which way each one biases the answer

| assumption | where used | bias direction |
|---|---|---|
| **Shared `rho` across work-identical legs.** Legs with equal `effective_mean_draft_len` and `non_drafting_round_count` are assumed to share the depth mixture. | the whole pooling construction | If false, pooling is *spuriously tight* — it would manufacture separation that is not there. Since the conclusion is *non*-separation, a false assumption here can only strengthen it. |
| **Per-tree relaxation in `pooled_enumerate`.** Each tree's feasibility is tested separately rather than jointly with a shared shape. | the 42-selection enumeration | Sound **superset**: the reported admissible reading set is at least as large as the true one, so "pooling prunes nothing" is not an artefact of over-pruning. |
| **Independent multi-shape box relaxation.** For the mixture family the two shape moments are bracketed independently instead of on their joint attainable set. | mixture `Δ` bracket | **Widens** the bracket — conservative. The reported cross-family width is therefore a *lower* bound on the true width. |
| **`M = 1..5` secant anchor** for excess. | §4.1, §4.2 | **Conservative** versus the "whole shape term" anchor: it attributes less excess to deep rounds, so it makes the required fix fraction larger, not smaller. |
| **Replicate sd 0.0683 %** measured from same-tree leg pairs rather than the advisor's 0.0995 %. | noise comparisons | The **smaller** sd makes non-separation *harder* to conclude, so using it is conservative for this result. |
| **Monotonicity cone** (`b ≥ 0, s ≥ 0`; `c ≥ 0, b + 3c ≥ 0`) as a physical constraint on the fit. | §2.4, §5 | This is the one assumption that *creates* separation. Anyone who wants to act on §2.4 must first defend the cone on physical grounds; it is an assumption about the runtime, not a measurement. |
| **Tolerance `0.00562`** carried over from E43 (2 sigma of its 0.281 % pair noise). | enumeration, brackets | Held fixed deliberately so E45 is comparable with E43. Tightening it shrinks the reading set but cannot break the collinearity of §2.3. |

`excludes_0_contains_local` is **not** used, defended, or relied on anywhere in
this analysis.

---

## 8. What would actually separate the families

Not more trees, and not lower noise. The identification failure is in the
*design*, so it has to be broken by changing the design:

1. **Legs with different `E[M]` at otherwise matched work.** The pool is
   restricted to legs whose `n_p` matches ours *exactly*, which is precisely
   what collapses the regressor variation. Rows that ran a different mean draft
   length would supply the missing variation — at the cost of the shared-`rho`
   assumption, which would then need a parametric link instead of exact
   matching.
2. **A direct local depth ladder on the current base.** Measuring `T(5)` and
   `T(6)` on the same host and tree observes the increment instead of inferring
   it. This is the only clean resolution, it needs GPU time, and it is what the
   `+32.850 ms` figure was trying to be before its tree was dropped.
3. **Defending the monotonicity cone.** If `b ≥ 0` and `s ≥ 0` are physically
   compulsory for a step and `c ≥ 0, b + 3c ≥ 0` for a quadratic, §2.4 already
   discriminates — on a single row. That is an argument about the runtime, not
   more statistics.

Given §4.3, none of this is on the critical path for the score: **0.52 % of MTP
leg time** takes the crown, and that target is family-free.

---

## 9. Reproduction

```bash
cd <repo root>
python3 research/e45_pooled_family.py --self-test   # 108 checks, 0 failed
python3 research/e45_pooled_family.py --census      # the pool and its dedup
python3 research/e45_pooled_family.py --run         # full analysis + JSON
```

* Stdlib only; imports `research/e43_ranked_step.py` for the shared primitives.
* Options: `--tol` (per-leg slack), `--node-cap`, `--no-threshold` to skip the
  threshold search, `--refresh` to re-pull the Yukon corpus, `--refresh-trees`
  to re-resolve commit trees.
* Output: `research/e45-pooled-family.json`.
* Tree SHAs are cached in `.mlxfast-private/e45-trees.json` (untracked).
* The self-test includes the pooling assertions the assignment asked for:
  `distinct_trees_five`, `dedup_pair_is_ledger_160H`, `dedup_keeps_higher_score`,
  `pooled_legs_47`, `pooled_ray_uses_all_legs`, plus the bit-for-bit E43
  cross-checks `e43_polytope_agrees_bit_for_bit` and `e43_q_bracket_agrees`.

## 10. Scope

`git diff efff400c HEAD -- Sources Vendor benchmark.json` is empty — see the
submission comment for the verbatim output. Only `research/` files were added:

* `research/e45_pooled_family.py`
* `research/e45-results.md`
* `research/e45_wandb_log.py`

No GPU seconds were used.

## 11. Suggested follow-ups (not implemented)

1. **Relax work identity to a parametric link.** Pool legs with differing `n_p`
   under an explicit model for how `rho` moves with `n`, which restores
   regressor variation. This is the only way pooling becomes informative, and it
   trades an exact assumption for a modelled one — worth a bounded experiment.
2. **Spend the GPU time on the direct ladder instead.** One matched local
   `T(5)`/`T(6)` measurement on the current base settles in one run what no
   amount of board telemetry can.
3. **Chase the family-free 0.52 %.** The median saturation in §4.3 says a
   two-prompt fix caps out at +4.56 %; a broad fix worth 0.52 % of leg time is
   enough to take the crown and does not require resolving the family question.
4. **Audit the ledger's plateau count.** §6.1 affects any statement that treats
   the plateau as six independent results.
