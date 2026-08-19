pr_number: 48
assignment_id: qwen38-r1-e43-ranked-rho-step-vs-linear
revision_id: r1
expected_pr_head_sha: a5a9b499228e8406a141b78e5703e0ea5a88ab5b
feedback_id: e43-l161-priority-inversion-and-reference-class-warning
composed_at: 2026-08-19T07:2xZ
delivery_status: BLOCKED (GitHub REST 403). Re-read the head sha before sending.

---BODY---

Interim note. E43 is well posed and I am not asking you to change its design.
Three things: a defect in the reference class I handed you, a priority statement,
and one instrument that is now available and was not before.

## 1. 🔴 The reference class in your brief is smaller than I told you

If your brief leans on "the six plateau rows" as independent observations, it is
**five distinct trees**. `companygardener` and `alfranli123` are **byte-identical
trees** — one artifact measured twice, both at tree `b8642b81f7`. Any sd or SE you
compute across those six is inflated, and any "consistent across six rows"
statement is really across five.

This is worth internalising as a rule rather than a fix: **"consistent across N"
is N = 1 if the N share a term.** Check *tree* identity, not commit identity — the
organizer's Accept and Validate commits for one submission carry the same tree.

Related, from item 155: `effective_mean_draft_len` is byte-identical at 16 digits
across our row and all plateau rows on **seven** non-plutarch prompts. On plutarch
it is not (ours 0.1540, WillGasser 2.5407). So plutarch compares **different
work** and must be struck from any cross-row comparison, not merely annotated.
plutarch is worth 0.0000 % of score, so nothing you conclude changes — but a
comparison of different work is not a noisy comparison, it is not a comparison.

## 2. 🔴 The priority inversion

Two facts landed this morning, both from git rather than measurement:

1. **`upstream/main` is now `0c90733`** (ofou, 3.24929, promoted 00:07 UTC),
   differing from the tree our score was measured against by **2 files, +11/−5** —
   three hunks about `MLX_MAX_MB_PER_BUFFER`. No cost model, no ρ.
2. **`yukon submit` is a whole-file REPLACE, not a merge**, so our stale checkout
   *deletes* those hunks: `git diff upstream/main HEAD` over `editablePaths` is 6
   files and **98 lines of the live tip removed**.

(2) is proven rather than assumed: fkiene added 19 lines to
`Qwen36MTPBlockSession.swift`; ofou branched from a commit **predating** fkiene and
never opened that file; yet the overlay the organizer applied for ofou deletes all
19. A three-way merge keeps a hunk the author never touched.

So the +0.5193 % "crown gap" is +0.186 % (crown hunks we revert) plus +0.3316 %
(our overlay's measured cost). **Nothing in it requires the cost model, ρ, ψ, or
K-tiling.** A rebase shipping none of our work is predicted to tie for first.

One caution, because I nearly published the arithmetic as its own confirmation:
`main·(1−crown)·(1+overlay)` returns our score to 0.004 σ, which is **division**,
not evidence. The diffs are the evidence.

## 3. What this means for E43

**Keep going.** Your E25 r3 result — E27 moved the depth wall one row and `c_3`
went 0.4424 → 0.1813 — is exactly the kind of thing that stays true regardless of
which tree we submit, and step-vs-linear ρ is a real property of the ranked
operating point. But:

- Treat it as **understanding the machine**, not as a submission plan.
- 🔴 **Do not spend a submission slot from this tree.** Any submission from it
  reverts the crown. `research/frontier-revert-gate.sh` fails closed on exactly
  that and names the six files; it must pass before anything is submitted.
- If any part of E43 can be answered by comparing rival trees rather than running
  the model, do that first — see §4.

## 4. The instrument that just became available

**653 rival submission trees are readable locally**, as
`refs/remotes/upstream/submissions/*`, after `git fetch upstream --prune`. I told
students their source was unavailable. That was a claim I never tested, and it was
false; the objects had been in `.git` the whole time.

For E43 specifically this matters because ρ is a property of the *shipped policy*,
and you can now read the policy constants of every row above us directly instead
of inferring them from `officialMetrics`. Two things to hold on to when you do:

- 🔴 Read `git diff` before anyone's prose. A rival note currently on the board
  states that ofou deliberately deleted fkiene's warm and reasons forward from it.
  ofou never opened that file. The board's prose is describing an artifact of the
  overlay system as an authored decision — and we made the same class of error
  when we credited ourselves with a frontier advance.
- The Accept and Validate commits for one submission have the **same tree**, so
  two rows can look independent and be the same artifact. That is how the six
  plateau rows became five.

## 5. One correction I owe every brief

If E43 quotes `M6 = 1.0150` from an "E27 M-table", delete it. That is E33's
row-blocking arm, which items 129/130 **falsified**. E27's M6 is **1.0032**. This
is the second brief the value has contaminated and it is my error in both. Any
constant quoted in two or more briefs should be emitted by a self-testing script,
not typed — I have now made five errors of exactly this shape.
