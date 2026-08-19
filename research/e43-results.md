# E43 — Does the ranked per-round data admit a step at M ≥ 6, and how big?

**PR #48 · assignment `qwen38-r1-e43-ranked-rho-step-vs-linear` r1 · base `04ad6bf1` · zero GPU seconds · `research/` only.**

## The answer, first sentence

**The ranked data is consistent with the full local `+32.850 ms` step. No discount is
justified, and my E34 r2 claim that local overstates the M = 6 step by 19.14 % is
withdrawn.** The ranked bracket on the step magnitude is `s ∈ [14.786, 80.483] ms/round`
— it excludes zero and contains 32.850 comfortably, and under the one named ρ assumption
I report (max-entropy) the ranked point estimate is **36.278 ms**, *larger* than local,
not smaller.

Three qualifications that matter as much as the headline:

1. **Superlinearity in M is decisive and assumption-free.** A straight line in M cannot
   explain the eight ranked per-round costs under *any* admissible ρ and *any* admissible
   round count: it needs **5.70 % per-prompt slack**, which is **20.3 ×** the widest
   measured pair-level noise (0.281 %). This does not depend on a ρ point estimate, and it
   survives the obvious rival — adding a physical rejection cost `r ≥ 0` to the linear
   model does not lower that threshold at all.
2. **"A step at M ≥ 6" is *not* identified against smooth convexity.** Both a step model
   and a plain quadratic explain the row with **zero** slack, and their max-entropy
   residual ratio is 1.174 — below my pre-registered inconclusive threshold of 1.5. The
   ranked telemetry says the curve bends upward; it does not say *where*.
3. **The bracket is wide: 65.70 ms, exactly 2.00 × the local step itself.** Ranked
   telemetry prices this lever only to within a factor of ~5. It is *not* too wide to be
   useful — it excludes zero, so it does not trigger the assignment's "stop early"
   condition — but it cannot adjudicate a 19 % discount, in either direction.

**What this means for thorfinn's K-tiled activation staging: build it.** The interval
excludes no-step, contains the full local magnitude, and the identified per-prompt excess
on the two prompts that can move the score is 9.4–27.4 ms/round = **16–47 % of the scored
leg**. Removing **1.1–3.2 %** of that excess is enough to take the crown.

---

## Scope and reproduction

```bash
python3 research/e43_ranked_step.py --self-test          # 51 checks, 0 failures
python3 research/e43_ranked_step.py --draws 400 --out research/e43-ranked-step.json
python3 research/e43_wandb_log.py
```

Stdlib only (no numpy/scipy on this host). Corpus cached at
`.mlxfast-private/e43-corpus.json` (656 rows, `--refresh` to re-pull). Every score
comparison is restricted to `head_provenance_sha256 == 559b24eb…`.

| | |
|---|---|
| Our row | `ca9251b8` morganmcg1, **rejected**, score **3.23250848**, head `559b24eb` |
| Board top | `0cd0a6b4` ofou, accepted, 3.24929399 |
| Zero-accept anchor | `c91581eb` scarletbright, score 1.209272, k ∈ [1.2059, 1.2115] |
| Shipped-surface diff | `git diff 04ad6bf1 HEAD -- Sources Vendor benchmark.json` → **empty** (shown below) |
| GPU seconds | **0** |
| W&B | run `piz9gjgg` — https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/piz9gjgg |

---

## Method: partial identification, not estimation

ρ(M) is not point-identified, so nothing here is estimated — everything is *bracketed*
over the admissible set. Two published equalities constrain each prompt:

```
effective_mean_draft_len = D / R        (mean of per-round draft counts, 4 dp rounded)
R + A = 512                             (rounds + accepted drafts = decode window)
non_drafting_round_count = #{rounds with 0 drafts}
```

Verified against the trusted harness, `Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift:370-385`:
`effectiveMeanDraftLength` is the mean of per-round draft counts and
`nonDraftingRoundCount` counts zero-draft rounds. **On seven of our eight prompts
`nd = 0`, so ρ(M = 1) = 0 exactly** — a constraint askeladd's published bracket did not
have.

The distribution over M = drafts + 1 ∈ {1..9} then carries two equalities, so **every
vertex of the ρ polytope has at most two support points and enumeration of ordered pairs
is exact, not a search.** All brackets below are vertex-exact.

### Correction to a published bracket

Reproducing askeladd's beagle upper bound `.90654` requires leaving ρ(1) free, which is
`(5.5327 − 1)/5`. With `nd = 0` pinned the correct upper bound is **`.883175`**. His lower
bound `.13318` is confirmed exactly. This is a tightening, not a disagreement:
`self_test` asserts both his published numbers and the corrected one.

### Pre-registration (fixed before fitting, in `PREREG` in the script)

| | |
|---|---|
| Primary tolerance | 0.281 % = widest measured pair-level per-prompt ratio spread (botany) |
| Decisive residual ratio | ≥ **3.0** (the factor my E34 r1 local analysis found) |
| Inconclusive residual ratio | ≤ **1.5** |
| Fair contest | step (a, b, s) vs quadratic (a, b, c) — same parameter count. Linear (a, b) is a 2-parameter straw man and is reported separately. |
| Bracket verdict classes | `contains_0_and_local` → telemetry cannot price the lever, redirect to askeladd's causal route · `excludes_0_contains_local` → consistent with full local step · `excludes_0_excludes_local` → quote the signed discount |

---

## Round recovery without any assumption — the advisor's uniqueness objection, answered

The advisor was right that beagle also admits R = 214 and R = 321, and that askeladd and
I both collapsed the ambiguity with the *same* monotonicity assumption (ledger-149 failure
mode). **The step family's own joint feasibility replaces that assumption entirely.**

Depth-first over all reading combinations, pruning any partial assignment whose polytope
is already empty, with **no T(1) bound and no monotonicity**:

```
42 of 226,017,792 reading combinations are jointly feasible   (7,070 LP nodes, ~16 s)
pinned by the model alone:  beagle=107  medicine=99  essays=87  republic=89  botany=85
still ambiguous:            plutarch {461,474,487,500}  drama {252,289,299,336}  travel {151,212,273}
```

**All five wide prompts are pinned to exactly the integers askeladd and I recovered, and
the monotone-ρ selection is inside the surviving set.** The pins hold at tolerance 0.562 %
and 2 %; at 5 % and 10 % they dissolve (96 and 415 combinations survive), so the pinning
is a real constraint, not an artefact of a tight band.

Under a **linear** family, **zero** combinations are feasible at 0.562 %, 1 %, 2 % or 5 %
— the linear threshold is 5.70 %, so no reading of R can rescue a straight line.

The alternatives are therefore excluded by the *cost model* rather than by an assumption
about the policy. I still carry the caveat the advisor asked for: the pinning is
conditional on T(M) being prompt-independent, which is stated below with its bias
direction.

Per-prompt admissible set at the headline tolerance:

| prompt | R | α | mean M | ms/round | q = P(M≥6) bracket | readings surviving |
|---|---|---|---|---|---|---|
| plutarch | 461 | 0.7183 | 1.1540 | 33.659 | 0.012–0.026 | 4 |
| drama | 252 | 0.4491 | 3.2976 | 40.181 | 0.000–0.324 | 4 |
| travel | 212 | 0.5329 | 3.6557 | 41.995 | 0.000–0.414 | 3 |
| **beagle** | **107** | 0.8351 | 5.5327 | 58.253 | 0.133–0.883 | **1** |
| **medicine** | **99** | 0.8750 | 5.7677 | 58.795 | 0.192–0.942 | **1** |
| republic | 89 | 0.9019 | 6.2697 | 64.338 | 0.317–1.000 | 1 |
| essays | 87 | 0.9004 | 6.4253 | 66.249 | 0.356–1.000 | 1 |
| botany | 85 | 0.8697 | 6.7765 | 66.744 | 0.444–1.000 | 1 |

### T(1) from the zero-accept row, with the direction corrected

Row `c91581eb` commits 512 primary tokens with **zero** accepted drafts on all eight
prompts, so `R = 512` uniquely, `A = 0`, `nd = 502`: ten rounds drafted, all rejected. Its
published ms/token is therefore a **mean** over rounds, which bounds T(1) from **above**,
not below:

```
T(1) ≤ 31.419 ms                        (needs only monotone T — assumption-free)
T(1) ≥ 30.435 ms  at κ = T(9)/T(1) = 2.803 (the local ladder's own ratio)
T(1) ≥ 26.796 ms  at κ = 10
```

I had this backwards in an earlier draft of the script and used the mean as a *floor*. A
larger κ lowers the bound and re-admits readings, so every exclusion made with it is
conservative in that one direction. Note the linear model's implied T(1) is **29.889 ms**,
*below* the assumption-free upper bound of 31.419 — so the straight line is not rejected
by the anchor; it is rejected by the shape of the eight points.

---

## (a) The model comparison, done properly

### (a-i) Assumption-free: minimum slack each family needs to explain the row at all

Minimised over **all 226,017,792 reading combinations and all admissible ρ**. No ρ point
estimate enters.

| family | minimum per-prompt slack | × measured pair noise (0.281 %) | verdict |
|---|---|---|---|
| **linear** `a + bM` | **5.7011 %** | **20.3 ×** | **decisively rejected** |
| **linear + reject cost** `a + bM + r·(rejects/round)`, r ≥ 0 | **5.7011 %** | **20.3 ×** | **decisively rejected — the rival explanation buys nothing** |
| step `a + bM + s·1[M≥6]` | 0.0000 % | 0.0 × | fits exactly |
| quadratic `a + bM + cM²` | 0.0000 % | 0.0 × | fits exactly |

**Superlinearity: decisive. Step vs smooth: no discrimination.**

The third row is the leading rival explanation and I tested it because it would have
destroyed the headline: *the curve is not superlinear in M, it just costs time to throw
rejected drafts away.* `rejects/round` is a published constant per reading, so this family
is exactly as falsifiable as `linear`. It nests `linear` at r = 0, so its threshold can
only be smaller — and it is **identical to 18 binary-search digits**, meaning the optimum
sits at r = 0 and a positive rejection cost provides *no* explanatory power. The reason is
visible in the data: rejects/round is **not** increasing in mean M
(plutarch 0.043, drama 1.266, travel 1.241, republic 0.517, essays 0.540, medicine 0.596,
beagle 0.748, botany 0.753) — it peaks at the *middle* widths, so it cannot manufacture
extra cost at the top.

### (a-ii) Under max-entropy ρ (named assumption)

Max-entropy ρ subject to the two published equalities. This is the narrowing assumption;
it is the least-committal single ρ, and it biases toward *spreading* mass across widths,
which **smears** any true step and therefore biases the fitted `s` **downward** and the
linear model's residual **downward** — i.e. it is conservative for my conclusion.

| model | k | rms | R² | χ²/dof | fitted |
|---|---|---|---|---|---|
| linear | 2 | 2.7237 ms | 0.950905 | 624.3 | `23.820 + 6.068·M` |
| linear + reject | 3 | 0.7658 ms | 0.996119 | 34.1 | see caveat below |
| **step** | 3 | **0.7419 ms** | **0.996358** | **29.8** | `31.268 + 1.452·M + 36.278·q` |
| quadratic | 3 | 0.8712 ms | 0.994977 | 39.1 | `33.639 − 1.930·M + 0.955·M²` |
| step + reject | 4 | 0.6571 ms | 0.997143 | 31.5 | `s = 21.253`, `r = −2.745` |

```
residual ratio  linear / step         = 3.671   ≥ 3.0 pre-registered decisive
residual ratio  quadratic / step      = 1.174   ≤ 1.5 pre-registered inconclusive
residual ratio  linear+reject / step  = 1.032   ≤ 1.5 pre-registered inconclusive
```

At this reading alone a straight line needs **7.825 %** slack = **27.8 ×** the measured
noise.

**The caveat that rescues the headline from the linear+reject row.** Unconstrained WLS
fits the reject coefficient at **−2.745 ms per reject/round** — a *negative* cost for
throwing work away, which is not physical. Rejects/round is anti-correlated with mean M
across these eight prompts, so a sign-free reject term is simply acting as a proxy for the
M-dependence rather than measuring a cost. With the physically required constraint r ≥ 0
the rival collapses (see the assumption-free table above, where it buys nothing).

**Sensitivity of the headline to including the reject term anyway:** the maxent step
magnitude falls from **36.278 ms to 21.253 ms**. That is a large move and I report it
rather than burying it — but 21.253 ms is still comfortably inside the bracket
`[14.786, 80.483]`, still far above zero, and still below the local 32.850. The direction
of the E43 conclusion does not change under the most hostile specification I could
construct.

**Why my E34 r2 fit gave the wrong answer.** I fitted `per_round = 20.543 + 6.792·M`, read
`T(6)/T(5)` off the line, and compared it to a local ratio measured at *fixed* width. The
advisor's objection is exactly right and is now quantified two ways: the straight line is
not a weak model here, it is a **rejected** one (20.3 × noise, assumption-free), and the
x-axis is a *mean* over a mixture, so the ratio of the fitted line at two mean-widths is
not the ratio of T at two widths. Both errors push the same way, toward an apparent
discount that is not there.

χ²/dof of 29.8 even for the winning model says the residual is not pure pair noise:
the step model is the best of three, not a complete description. Candidates for the
missing term are per-prompt T (context length differs) and a rejection-cost term — see
"what would narrow this" below.

---

## (b) The bracket on the ranked step magnitude

### Calibrating the tolerance first

The pre-registered 1σ band is a **Chebyshev** criterion applied simultaneously to eight
prompts, so it rejects a *true* s = 0 far too often. Measured by simulation (400 draws,
generating noise at the measured 0.281 %):

| band | false-positive rate at s = 0 |
|---|---|
| 1.0 σ = 0.281 % | **0.74** — anti-conservative, unusable |
| 1.5 σ = 0.422 % | 0.24 |
| **2.0 σ = 0.562 %** | **0.04** ✔ headline arm |

**MDE:** at the calibrated band, power is 0.04 at s = 0, 0.49 at s = 2 ms and **1.00 at
s = 4 ms**. So an "excludes 0" finding here is not an under-powered null in disguise: the
instrument would have detected a step a *quarter* the size of the smallest bracket end.

### The brackets

| arm | tolerance | s (ms/round) | width | contains 0 | contains local 32.850 |
|---|---|---|---|---|---|
| measured pair noise | 0.281 % | [15.366, 79.808] | 64.44 | no | **yes** |
| **calibrated (fp ≤ 5 %)** | **0.562 %** | **[14.786, 80.483]** | **65.70** | **no** | **yes** |
| model slack | 1 % | [13.882, 81.535] | 67.65 | no | yes |
| model slack | 2 % | [11.819, 83.936] | 72.12 | no | yes |
| **union over all 42 feasible round readings** | 0.562 % | **[11.324, 84.106]** | 72.78 | no | **yes** |

Adding the κ-based T(1) lower bound changes nothing at any arm — the bound is not binding,
so the result does not rest on κ.

**Verdict class `excludes_0_contains_local`** → *"ranked data is consistent with the full
local step; no discount is justified."*

**Bracket width is a first-class result: 65.70 ms = 2.00 × the local step.** Ranked
telemetry can tell us the step is real and can tell us it is not small. It cannot tell us
whether it is 15 ms or 80 ms. Any future claim of a specific ranked/local ratio from
telemetry alone should be rejected on these grounds — including mine.

### What *is* identified: the product `e_p = s · q_p`

`s` and `q_p` are each unidentified, but their product — the ms/round a fix at M ≥ 6 could
actually remove from prompt p — is bracketed much more tightly, because a larger `s`
forces a smaller `q`:

| prompt | e_p (ms/round) | as % of the scored leg |
|---|---|---|
| plutarch | 0.006 – 2.284 | 0.02 – 6.79 % |
| drama | −0.226 – 8.806 | −0.56 – 21.92 % |
| travel | 0.291 – 10.620 | 0.69 – 25.29 % |
| **beagle** | **9.751 – 26.878** | **16.74 – 46.14 %** |
| **medicine** | **9.442 – 27.420** | **16.06 – 46.64 %** |
| republic | 13.166 – 32.963 | 20.46 – 51.23 % |
| essays | 14.513 – 34.874 | 21.91 – 52.64 % |
| botany | 13.736 – 35.369 | 20.58 – 52.99 % |

Union over all 42 feasible round readings: beagle **7.564 – 28.488**, medicine
**7.025 – 29.030** ms/round.

Drama's lower end is **slightly negative (−0.226)**. That is not clamped and not a bug: at
the tolerance band the linear part is allowed to sit marginally above the observed y, so
the step contribution can be marginally negative. It is reported as measured. On the two
prompts that matter it is far from zero.

### Value on the published score

Order-statistic sentence first: **only beagle (79 % of available value) and medicine
(21 %) can move the score; the other six are worth +0.0000 % each.** All figures below
recompute the median rather than differentiating, so saturation against essays at
`raw_p = 3.366118` is honoured.

| arm | beagle leg | medicine leg | score | gain |
|---|---|---|---|---|
| base | — | — | 3.23250848 | — |
| remove 10 % of e_lo | −1.67 % | −1.61 % | 3.269694 | **+1.1504 %** |
| remove 10 % of e_hi | −4.61 % | −4.66 % | 3.318601 | **+2.6633 %** |
| remove 25 % of e_hi | −11.54 % | −11.66 % | 3.380068 | +4.5649 % (saturated) |
| remove 100 % of e_lo | −16.74 % | −16.06 % | 3.380068 | +4.5649 % (saturated) |

**Fraction of the identified excess a fix must actually remove:**

| target | at the low end of e | at the high end of e |
|---|---|---|
| one σ_score (0.0978 %) | **0.60 %** | 0.21 % |
| **crown gap (+0.5193 %)** | **3.15 %** | **1.11 %** |

Even the most conservative arm — remove only 10 % of the *low* end — is **+1.150 %, 2.2 ×
the crown gap**. This is the sentence I would put in front of the K-tiling decision: the
lever is worth building even if ranked reality sits at the pessimistic end of every
bracket in this document.

---

## (c) Ranked φ, as a bracket

φ is the M ≥ 6 share of QMV cost. **Note a definitional split I did not resolve
unilaterally:** `program.md`/ledger 137 define φ at **M = 6**; this assignment says
**M ≥ 6**. Both are reported.

The weighting matters, so three are given. `passes(M) = ceil(M/5)` from the dispatch
table, so under pass-count weighting φ(M≥6) = 2q/(1+q) in closed form (asserted in
`self_test`). The local ladder decomposes as **F = 10.222 ms/row + S = 24.701 ms/extra
pass** (R² 0.9960), giving the mixed weighting.

| prompt | weighting | cell | φ bracket |
|---|---|---|---|
| **beagle** | pass count | M ≥ 6 | **0.2351 – 0.9380** |
| beagle | pass count | M = 6 | 0.0000 – 0.9380 |
| beagle | row share | M ≥ 6 | 0.2166 – 0.9578 |
| beagle | pass + row (local split) | M ≥ 6 | 0.2227 – 0.9488 |
| **medicine** | pass count | M ≥ 6 | **0.3220 – 0.9701** |
| medicine | pass count | M = 6 | 0.0000 – 0.9701 |
| medicine | row share | M ≥ 6 | 0.2995 – 0.9799 |
| medicine | pass + row (local split) | M ≥ 6 | 0.3070 – 0.9755 |

The row-share lower bounds reproduce askeladd's published `.2166` / `.2995` **exactly**,
which is a useful cross-check that our two pipelines agree on the arithmetic.

**Two things to say to askeladd rather than around him:**

1. **If φ is defined at exactly M = 6, ranked telemetry cannot bound it away from zero.**
   A vertex can place all the M ≥ 6 mass at M = 9 and none at M = 6. His injected local
   regression measures the ρ his fixture actually produces and so *can* give a nonzero
   number; the ranked bound simply does not exist. If his measured φ(M = 6) is large, that
   is a statement about ρ, not a contradiction of this bracket.
2. **My ranked bracket cannot falsify his local point measurement, in either direction.**
   Any φ in the ranges above is admissible. The honest joint reading is: use his causal
   number for the *mechanism*, and use my bracket only to check it is not outside what
   ranked telemetry allows. As of this analysis, nothing he could plausibly measure would
   be outside it.

---

## Assumptions ledger — every narrowing assumption and its bias direction

| # | assumption | used for | bias direction | reported without it? |
|---|---|---|---|---|
| 1 | T(M) is prompt-independent | pooling eight prompts into one fit; the round pinning | Prompt-varying T would widen every bracket and could unpin the wide prompts. **This is the load-bearing assumption of the whole analysis.** | No — the 1 % and 2 % model-slack arms are the sensitivity proxy |
| 2 | max-entropy ρ | the (a-ii) point fits only | Smears any true step ⇒ biases fitted `s` **down**, conservative for my conclusion | Yes — (a-i) is assumption-free and is the headline |
| 3 | monotone ρ / round cost | nothing load-bearing; cross-reference only | The advisor's ledger-149 concern | Yes — the 42-selection enumeration replaces it |
| 4 | κ = T(9)/T(1) ≤ 2.803 | the T(1) lower bound | Larger κ lowers the bound and admits more readings; every exclusion is conservative in that direction | Yes — the bound is non-binding; brackets are identical with and without it |
| 5 | pair noise 0.281 % transfers to ranked | tolerance calibration | Measured locally on the fixture. If ranked is noisier the bracket widens | Partially — 1 % and 2 % arms bracket it |
| 6 | `passes(M) = ceil(M/5)` | the φ pass-count weighting | Dispatch-table fact, asserted in self-test | Row-share weighting needs no dispatch model |
| 6b | reject cost r ≥ 0 | ruling out the reject-cost rival assumption-free | Sign is physical: discarded work cannot save time. Sign-free, the rival ties the step model but only at r = −2.745 ms | Yes — the sign-free fit is reported in full |
| 7 | ρ is the same for us and the plateau (ledger 155) | interpreting the deficit as pure time at identical work | Advisor-supplied, `effective_mean_draft_len` byte-identical on 7/8 prompts | Not tested here |

Per the advisor: **no part of the ranked step is attributed to the SDPA fallback**
(ledger 156 — the chunked-SDPA fix at widths 6–9 is already in our binary). The step here
is an unattributed excess at high M; thorfinn's weight stream is the leading candidate.

---

## Honest limitations

- **χ²/dof = 29.8 for the winning model.** The eight points are not explained to pair-noise
  precision by any 3-parameter model tried. There is structure left.
- **The bracket cannot separate a discontinuity from a curve.** If the campaign needs to
  know *where* the cost appears, only a causal local measurement can say.
- **Rejection cost is tested but not settled.** With the physical constraint r ≥ 0 it
  explains nothing; sign-free it fits nearly as well as the step but only with an
  unphysical negative coefficient, and it pulls the maxent step estimate from 36.3 ms to
  21.3 ms. Eight prompts cannot separate two nearly-collinear regressors.
- **One board row.** All eight observations come from `ca9251b8`. Ledger 155 says the six
  plateau rows share our ρ trajectory exactly, so pooling them would add timing replicates
  at identical work — that is the single cheapest way to shrink these brackets and I did
  not do it.
- **Every number is conditional on assumption 1.** If T is prompt-dependent, the pinning
  of beagle to R = 107 is not proven and the brackets widen by an unknown amount.

## Suggested follow-ups (not implemented)

1. **Pool the six plateau rows.** Ledger 155 gives identical ρ across seven prompts and
   seven solvers, so their per-round costs are repeated measurements of the same T under
   different hardware noise. That converts my single-row Chebyshev band into a real
   standard error and should shrink the step bracket substantially — the single highest
   value/cost follow-up in this document.
2. **Break the reject-cost / step collinearity.** The two regressors cannot be separated on
   eight prompts. Pooling the plateau rows (follow-up 1) is one route; a local injected
   experiment that varies the reject rate at *fixed* M is the decisive one.
3. **Test prompt-dependent T** by allowing a per-prompt intercept tied to seed length. If
   assumption 1 fails, everything above needs re-bracketing, so this is the cheapest
   possible attack on my own result.
4. **Resolve the φ definition** (M = 6 vs M ≥ 6) in `program.md` so askeladd's causal
   number and this bracket are comparable without a translation step.

---

## Scope proof

```
$ git diff 04ad6bf1 HEAD -- Sources Vendor benchmark.json
$ echo "exit=$? bytes=0"
exit=0 bytes=0
```

Files added, all under `research/`:

- `research/e43_ranked_step.py` — the analysis (stdlib only, `--self-test` → 51 checks, 0 failures)
- `research/e43-ranked-step.json` — full machine-readable output
- `research/e43_wandb_log.py` — W&B logging
- `research/e43-results.md` — this document

---

_This document was produced by an AI research agent (OpenHands / Senpai `qwen-edward`) on
behalf of the campaign._
