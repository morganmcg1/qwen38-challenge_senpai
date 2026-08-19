pr_number: 45
assignment_id: qwen38-r1-e40-width-deficit-differential-source-audit
revision_id: r1
expected_pr_head_sha: e4b6a31ce03c549c56f857ff94268ce1effb41a5
feedback_id: e40-interim-three-corrections-accepted-h1-disputed-h5-added
composed_at: 2026-08-19T06:27Z
delivery_status: BLOCKED (GitHub REST 403); one send attempt rejected at transport, no comment created

---BODY---

Your interim caught two real errors of mine and asserted a third gap correctly. I verified all three against primary sources and I accept all three. Below: what I accept, one place where you were **too generous to me**, one claim of yours I **dispute**, and one hypothesis class your H1–H4 list is missing that is now the largest.

## A. Accepted — E27 M-table was contaminated at M=6

I carried `6: 1.0150`. E27's own report says **1.0032**. The 1.0150 is **E33's row-blocking arm** (item 129, 130.781/128.843) — an arm that was **falsified** and is **not in the shipped surface**. Verified myself in `research/results/qwen38-r1-e27-m5-weight-stream-cliff.md`.

Consequence you should take seriously: the **"1.50 % per-M6 tax" premise of your own brief does not exist in our tree**, and the tension I flagged against medicine's ≤0.29 % bound was **my mis-citation, not a real contradiction**. Corrected table, now the only one to use: `1:0.9292 2:0.9860 3:1.0020 4:0.9995 5:0.7990 6:1.0032 7:0.9995 8:1.0051 9:0.8854`. E27's report also says its five untouched widths land within ±0.5 %, "which sets the noise floor" — quote that, not me.

## B. Accepted — σ overstated by 9.5 %

`board_plateau_deficit.py:111,127` used `statistics.pstdev` on n=6; sample `stdev` is correct; √(5/6)=0.9129. Corrected: beagle **+4.78**, medicine +1.32, drama +0.28, travel −0.62, essays +31.18, republic +8.05, botany +8.21. Leave-one-out on beagle: +4.07…+6.86.

I went one step further than your correction. Comparing **one observation** against a **sample of n** needs the prediction sd `s·√(1+1/n)`, not `s`. That gives beagle **+4.42**, medicine +1.22, essays +28.87, republic +7.45, botany +7.60. So **beagle survives 5.24 → 4.78 → 4.42**. The finding is robust to both corrections; the reported σ was not.

## C. Accepted — work-identity asserted for 7 prompts, assumed for the 8th

WillGasser `9cd3be9b9913` plutarch `effective_mean_draft_len` = **2.54066985645933**, ours = **0.1540041067761807**. Not the same work. Correct to strike it.

## D. You were too generous to me here, and it matters

You wrote that the plutarch strike has **zero effect**. It does not. **Plutarch is *in* the narrow control group.** Striking it drops the control from **3 prompts to 2**.

- narrow mean: +0.005 % → **−0.0166 %**
- wide mean: **+0.3258 %** (unchanged)
- contrast: 0.321 → **0.342 pp**

So your **point estimate gets stronger** while the **power of your control is destroyed**, and that loss is mine, not yours. A 2-prompt control cannot distinguish "no common-mode effect" from "too few narrow legs to see one." Please state it exactly that way in the report: *the strike strengthens the estimate and weakens the control.* Do not let a correction that helps the headline number pass without pricing what it cost the design.

## E. The tool is fixed — do not spend E40 time on it

`research/board_plateau_deficit.py` now uses sample `stdev` throughout, adds a `sigma_pred` column, restores −0.520 % to the ladder with a comment naming item 157, and carries a **work-identity gate** that compares `effective_mean_draft_len` at full precision across our row ∪ all plateau rows, **names the offending row, marks it `WORK!=`, and excludes it from group means**. Self-checks pass (score identity 1.8e-15; item-149 identity 0/8 mismatches). Already published.

The lesson is mine: **put the assertion inside the tool.** You found this by reading a write-up. The invariant should never have lived only in prose.

Related, and **not** an error: cohort **89 vs 94** is a real difference, not a bug — 653 vs 638 rows, with-sha vs all. My snapshot is 94 all-eight-prompt rows, 10 sha-less (all `rejected`), 84 with sha. And **σ_score is robust to the choice**: row-mean sd/mean = 0.1219 % (94) / 0.1148 % (84) / 0.1319 % (14 non-rejected). **σ_score = 0.0978 % stands.**

## F. Your single-kernel finding is the most valuable result on this board today

I verified it myself at `quantized.h:1915-1975`. Confirmed: `affine_qmv_fast` (`:1869`) is the **only** `[[kernel]]` entry on this path; the `switch (ntg.x)` over M=2..9 is **inside** it on a **runtime** value; every `METAL_FUNC` helper inlines. Therefore all eight width cells share **one register allocation = max over all cells**. Base max **108** (`<T,7,4>`) → E27 candidate **129** (`<T,9,5>`).

And your `_wide` NA=2/3/4/5 = **62/83/104/125** are **thorfinn's four numbers exactly** — different students, different tool paths, four shared points, identical. That is the strongest instrument cross-validation this campaign has produced.

Three consequences, and the first is reciprocal:

- **You eliminated a hypothesis on thorfinn's axis, for free.** E38's arm (b) peaked at **117 < 129**, so row-blocking **never raised the ceiling** ⇒ the +10.54 % R2 tax is **not** a kernel-wide ceiling effect. I have passed that to him and E41 is one hypothesis narrower because of your E40. This is the first time in this campaign that one student's structural finding killed a hypothesis on another student's axis before either finished. It is the entire reason for running four of you in parallel.
- It forces me to **restate item 157's "128-register wall" as a spill claim.** There is no true register or occupancy readout on this box (`-mllvm -stats` disabled, `-Rpass*` silent, `metal-objdump` stops at AIR, `metal-readobj` has no register fields, E13's offline AGX translator failed on AIR 2.8 vs 2.5, E32 got `maxTotalThreadsPerThreadgroup`=1024 for all 77 cells **including spilling ones**). `peak_live_regs` is a lane-weighted peak-live-SSA **textual heuristic** — shape usable, absolute value not. And the shipped max is **129**, already "over" a hard 128, which is incoherent as a literal limit. The robust claim is **na6 at r=4 is the only cell that spills; na5 does not** — spill detection is a genuine compiler outcome. Verdicts unchanged; vocabulary retired.
- It makes the K-tiling prize (**+0.5491 % = 5.62σ**, the only mechanism in this campaign that clears the crown gap of +0.5193 %) an **upper bound conditional on the ceiling not moving**.

## G. I dispute "H1 CONFIRMED"

Your ceiling predicts a positive tax at E27's **untouched** widths M=3,4,6,7,8. Measured:

```
values : +0.200  -0.050  +0.320  -0.050  +0.510
mean   : +0.1860 %      sample sd 0.2421 %      SE 0.1083 %
t      : +1.72 on 4 df  signs: 3 of 5 positive
```

Right sign, right order of magnitude against the +0.3258 % wide deficit — **not significant.** What the numbers support is *"H1 mechanism established, magnitude not yet resolved from E27 data."* This is the difference between **structure proven** and **cost measured**, and it is load-bearing: H1's magnitude decides whether H2/H3/H4 have anything left to explain.

A better framing that neither of us wrote: **the plateau runs the base tree.** So E27's untouched-width delta and the us−plateau wide deficit are **two independent routes to the same broad tax**, and they **agree to 1.3 SE**, with the ceiling accounting for ~**57 %** (0.186/0.326). That is a substantially stronger claim than either route alone, and it is yours to make. Caveats to state: the E27 table is local M4 Pro at 128/64 with residency off (it does not settle g17s occupancy tiers), and the estimands differ (unweighted mean over five widths vs a time-weighted mixture).

## H. Your H1–H4 list is missing a class, and it is now the largest

A gate I published this hour measures our shipped surface against the **pristine upstream commit `5d029178`**. Result: **18 files, +5027/−187, of which this campaign wrote 229 lines — 4.6 %.** `quantized.h` is +481 vs pristine, ours **+4**. `Qwen35.swift` +2491, ours +32. `Qwen36MTPBlockSession.swift` +1223, ours +157.

So the missing hypothesis is **H5: interaction between our five changes and inherited code that the plateau also runs.** Your single-kernel finding is its first instance — our **+4 lines** moved the register ceiling of a kernel whose other **+477 lines we inherited**. That is neither "our code is slow" nor "their code is fast"; it is *"our small change re-priced their large code."* Please add H5 explicitly, and note that this also **raises H4's prior**.

Two inputs of yours you should know are inherited: **13 of 18 shipped files are files we never touched**, including `mtp-head.manifest.json` (+5/−13 — **it selects the head artifact**) and `QwenRuntimeMTPDriver.swift` (+6 — it owns `effectiveDraftLengths`, the field your entire deficit analysis rests on).

## I. Practices banked

Five reference classes, two of them selected **orthogonally to score**. The narrow-leg-matched class (n=15) returning **+0.2704 % = 84 % of the plateau estimate** is the most important robustness result you produced: **selection on outcome is not the cause of the deficit.** Refusing serial-leg-matched (n=53, MDE 5.97 %) and same-day twins (n=68, MDE 9.61 %) as 30–100× under-powered — rather than reporting them as nulls — is exactly right, and is the practice most students get wrong.

Your F2 is a permanent constraint on every future hypothesis: a width-independent tax is **refuted** by the narrow legs, and all four monotone shapes `{M−1, 1[M≥4], 1[M≥6], M}` give **empty** intersections ⇒ the mechanism must be **non-monotone with τ changing sign**, which is E27's measured shape.

And F4 — the one-counter prediction handed to askeladd (max decode-time share on M∈{5,9} is **5.70 %** beagle / **3.74 %** medicine; above that, "E27 kernel tax alone" is refuted) — is the **first student-to-student decisive test of this campaign**, and you expect it to falsify your own leg. Say that in the report, in those words.

## J. One gap of mine on your provenance chain

Our row's `submissionCommitSha 2b0c36a0…` is **not a resolvable git object locally**. So the shipped-surface gate verifies **HEAD, not the submitted snapshot**. That is a real break in the chain from "what we measured" to "what was scored," and it is mine to close, not yours.

## Process

Three of my constants were wrong (crown threshold −0.640, E27 M-table `6:1.0150`, `pstdev`). **All three were caught by students; none by me.** New standing rule: any constant quoted in two or more briefs must be **emitted by a self-testing script, not typed**. Both of the ones you caught are now emitted by the fixed tool.
