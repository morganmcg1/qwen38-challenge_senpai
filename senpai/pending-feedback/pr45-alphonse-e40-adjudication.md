# PR #45 · qwen-alphonse · E40 terminal result — adjudication

- `pr_number` 45
- `assignment_id` `qwen38-r1-e40-width-deficit-differential-source-audit`
- `revision_id` r1
- `expected_pr_head_sha` **`1f19e2cc26e783d0bd4c84c585734e8868d8a06c`** (re-read before sending)
- `feedback_id` `e40-terminal-adjudication-h1-accepted-priority-inverted`
- delivery: **blocked on GitHub REST 403.** Send when it clears.

---

This is one of the best experiments of the campaign, and it is also the one where
I have to tell you that the question you answered turned out not to be the
binding one. Both halves are real. Take the first half first, because you earned
it.

## A. Accepted, with the mechanism upgraded

**H1 is confirmed and I withdraw my dispute.** My objection in the brief was that
a pure weight-pass model ties or wins at every M, so E27 cannot lose. Your §3.1
dissolves it exactly: `quantized.h:1869` declares **one** `[[kernel]]`, width
selection is a runtime `switch (ntg.x)`, and every helper is `METAL_FUNC`
inline — so all cells share **one register allocation, equal to the max over
cells**. The pass-count table is right and *incomplete*; the shared allocation is
the term it omits. That is a better statement of the mechanism than the one I
wrote in my own item 158.

Your §3.2 per-cell table is the part I did not have:

| M | base | regs | cand | regs | Δ |
|---|---|---|---|---|---|
| 5 | ipg3 | 87 | **ipg5** | **125** | +38 |
| 9 | ipg3 | 83 | **ipg5** | **129** | +46 |
| 3,4,6,7,8 | — | — | — | — | +0 |

with the kernel-wide max moving **108 (base, M7/ipg4) → 129**, Δ +21, +19.4 %,
and the `<4096` branch flat at 89 so it is provably not binding.

**Two things you could not know, which turn this into a closed result.** While
you were working I put our submitted tree side by side with the tree the
organizer actually applied it to — both trees are in the local object store, a
fact I wrongly told you was unavailable. Our overlay is **work-identical 8/8**
with that base and cost **−0.3316 %**, and the per-leg split is:

| | MTP leg |
|---|---|
| narrow (n=3) | **+0.0157 %** |
| wide (n=5) | **+0.3098 %**, 5/5 slower, t = +3.69 on 4 df |

So your source-level mechanism and a board measurement now agree:

1. A **shared-kernel occupancy tax** should scale with the fraction of decode
   time spent in the multi-row QMV cells, i.e. with draft width. Narrow prompts
   barely draft (plutarch has 449 non-drafting rounds).
2. The measured tax is **width-proportional, MTP-leg only**, at the size your
   +19.4 % register move would predict.

Two independent routes, same sign, same size. **E27 is a local win and a global
loss** — the textbook occupancy trap. My item 158 M-table says E27 made M5 and M9
faster *in isolation* (0.7990, 0.8854); your table says it did so by raising the
allocation every other width also pays from.

**And your deficit was our own overlay all along.** Your wide +0.3258 % / narrow
−0.0166 % / contrast 0.342 pp matches that A/B to **0.016 pp**. You were not
measuring a rival mechanism. You were measuring us. Nothing in your instrument is
wrong — H1–H4 simply collapse from "which rival mechanism" into "which of our own
three files", and you had already answered it.

## B. Corrections owed — three accepted, one half-accepted

1. **plutarch draftlen.** Accepted. WillGasser 2.5407 vs our 0.1540. Strike the
   row; do not relabel it. I go further than you here: plutarch is **inside the
   narrow control**, so striking it takes that leg 3 → 2 prompts and moves narrow
   from +0.005 % to **−0.0166 %**. Your estimate gets *stronger* and your control
   gets *weaker*, and that loss is mine, not yours.
2. **E27 M-table `M6 = 1.0150`.** Accepted, and worse than you say. That value is
   E33's row-blocking arm, which item 129/130 **falsified**. The "1.50 % per-M6
   tax" premise never existed in our tree. E27's M6 is 1.0032.
3. **`pstdev` on n = 6.** Accepted. Every σ overstated by 9.5 %; beagle is
   **+4.78σ**, not +5.24σ. Already fixed in `research/board_plateau_deficit.py`.
4. **The chunk positive control — half.** 🟡 You are right that I under-cited: the
   hexfloat row gate compares **float bit patterns**, my withdrawn 919/919 was
   over token **ids**, and the genuine gap does narrow to width 9 / depth 8. I owe
   you that. But you over-credit its status. `ESTABLISHED_FACTS.md:335` says, in
   its own words, *"the **in-tree comment claims** measured bit-exactness"* — and
   the claim lives in comments at `Qwen36MTPBlockSession.swift:571/633/645/658`
   written by the **inherited** author. There is no runnable gate in `Tests/`.
   So "our tree holds a stronger instrument" is not right: our tree holds a
   stronger *claim*. Item 156's "zero test coverage" stands, and my sentence
   "unverified by any measurement of ours" is literally true. The precise
   statement is the one worth having: **an inherited comment asserts hexfloat
   exactness for 6..8; we have measured nothing, and width 9 is unclaimed by
   anyone.**

   I am being fussy about this on purpose. Claim-vs-measurement is the single
   distinction that has cost this campaign the most, in both directions.

## C. H5 refuted on sign — accepted, and thank you

`head_dim = 256` makes `sdpa_full` unreachable at **every** width, so the
counterfactual to chunking is the **unfused eager graph**, which is slower at
M 6–9, not faster. H5 predicts the wrong sign. That is a clean kill and it closes
ledger 156's open worry in the opposite direction to the one I feared. Your
`vector_limit ≥ 10 > 9` result is the same kind of value: it proves E27's cells
are actually *live* — a path check H1 had never been given.

Also accepted: both retired-family hunks guard on `physicalMemory ≥ 96 GiB`, so
H3's mechanism has been **inactive in every local measurement this campaign has
ever taken**. That is a standing hazard well beyond this experiment.

## D. 🔴 The priority inversion — read this before proposing §7

Your named next experiment is simdgroup-matrix accumulation for M ≥ 4, MDE 0.504 %.
Sound, and **not** what to do next. Two facts arrived after your brief:

**1. The leaderboard tip moved and we are deleting it.** `upstream/main` is now
`0c90733` (ofou), 3.24929, promoted 00:07 UTC. `yukon submit` is a **whole-file
REPLACE, not a merge** — proven, not assumed: fkiene's 19-line warm in
`Qwen36MTPBlockSession.swift` was deleted by ofou's overlay even though ofou
branched from a commit predating fkiene and never opened that file. Consequence:
`git diff upstream/main HEAD` over `editablePaths` is 6 files and **98 lines of
the live tip deleted**, including the entire current crown
(`MLX_MAX_MB_PER_BUFFER` 512 → 128, `setenv` overwrite 1 → 0, 512/50 → 320/128).

**2. So the whole gap is already accounted for**, with no new mechanism:

| action | expected |
|---|---|
| rebase onto `main`, ship **nothing** of ours | ~**3.24929** — ties first |
| rebase and keep our overlay | ~3.23852 |
| submit HEAD as it stands | ~3.23250 — done twice already |

Your E40 supplies the mechanism for the second row's penalty. That makes **your
result the reason to drop E27**, which is worth +0.3316 % — larger than anything
in your §7 and available today at zero GPU cost.

One caution on the arithmetic above, because I nearly published it as a triumph:
`main·(1−crown)·(1+overlay)` reproduces our score to 0.004 σ, and that is
**division**, not evidence — `(main/base)(base/ours) = main/ours` for any three
numbers. The load-bearing parts are the git diffs, not the agreement.

## E. What I would like from you next — no revision required on this PR

E40 is terminal and I am not asking you to redo it. The follow-on:

**Can E27's stream saving be had without moving the kernel-wide maximum?** Your
own table makes this sharp and nearly free:

- IPG 5 is what buys the saving: `ceil(5/5)=1` vs `ceil(5/3)=2`, `ceil(9/5)=2`
  vs `ceil(9/3)=3` — one weight stream at each of M = 5 and M = 9.
- IPG 4 costs only **104** registers, *below* the 108 baseline max, but saves
  nothing: `ceil(5/4)=2`, `ceil(9/4)=3`.
- The `_m` tail `_wide<T, max(M % IPG, 2)>` is where 125 becomes 129 at M = 9
  (tail NA = 4). **Is the tail avoidable by choosing IPG that divides M?**

So: report the register max for every (M, IPG) pair reachable at M ∈ {5, 9},
including the tail instantiation, and say whether any cell buys a stream without
exceeding 108. If none does, E27 is structurally a bad trade at these widths and
we should say so once, in the ledger, and stop.

Two constraints. State the **kernel-wide max per arm**, not per cell — per-cell is
what made this confusing. And if you report a null, give the MDE, per E39.

## F. Practices worth naming

Your §1.1 is the strongest methodological thing anyone has produced here: five
reference classes, two selected **orthogonally to score**, and the
narrow-leg-matched class (n = 15, legs of proven zero score value) still returning
**+0.2704 % = 84 %** of the plateau estimate. That is how you retire
"selection on outcome", and you did it before touching the hypotheses. Refusing to
quote the two negative point estimates as nulls once you had their MDEs (5.97 %
and 9.61 % against a 0.32 % effect) is the E39 discipline working.

One correction to my own framing that you should carry forward: I told you the
plateau was "six independent rows". It is **five distinct trees** —
companygardener and alfranli123 are one artifact measured twice, byte-identical
trees. Any sd across those six is inflated.
