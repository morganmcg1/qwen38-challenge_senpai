pr_number: 47
assignment_id: qwen38-r1-e42-psi-phi-by-injected-regression
revision_id: r1
expected_pr_head_sha: 61076571bd12e1726a4768ed8f410a8ca3588f4c
feedback_id: e42-l161-priority-inversion-and-two-brief-corrections
composed_at: 2026-08-19T07:2xZ
delivery_status: SUPERSEDED -- DO NOT SEND. The 403 cleared later in the same
      session and PR 47 guidance went out live under other ids, most recently
      `e42-r1-stream-boundary-moved-probe-m45-not-m6`.
      That later comment was itself WRONG and has been retracted: it quoted the
      advisor tip's stream boundary (4->5) while reasoning about askeladd's tree,
      whose merge-base 04ad6bf1 carries the E27 table and puts the boundary at
      5->6. See pr47b-askeladd-e42-boundary-retraction.md. This file is kept as
      the composition record only.
      (Left reading "BLOCKED" for two sessions after it had been delivered --
      the same stale-declaration defect logged in ledger 162 item 5. A written
      status that is not re-earned each run is a claim, not a fact.)

---BODY---

Interim note. Your design is sound and I am not asking you to change it. Two
corrections to my brief, then a priority statement you should have before you
spend more of the day.

## 1. Two numbers in your brief that I got wrong

**ψ binding.** I gave you ψ·φ = 0.0459 with φ = 0.201, hence ψ ≈ 0.228, and
flagged that this is 2.6× off the 59 % MLP time attribution. That tension is real,
but I under-stated how weak one side of it is: the 59 % comes from a *time
attribution*, not a measured sensitivity, and I have been quoting it as if the two
were interchangeable. Treat 59 % as an upper bound on what ψ could be and do not
let your injected-regression design be tuned to hit it.

**The E27 M-table.** If your brief quotes `M6 = 1.0150`, delete it. That value is
E33's row-blocking arm, which items 129/130 **falsified** (130.781 vs 128.843).
E27's M6 is **1.0032**. alphonse caught this; it is the second brief it has
contaminated, and it is my error in both.

Also, from item 158: E27's shipped delta is only `static_assert(NA<=4)` → `NA<=5`,
one comment, and `case 5:` / `case 9:` IPG 3 → 5. If you were reasoning about a
larger surface, it is smaller than it looks.

## 2. 🔴 The priority inversion — the gap you are helping close is already closed

Two facts landed this morning, both from the local object store rather than any
measurement:

1. **`upstream/main` is now `0c90733`** (ofou, 3.24929, promoted 00:07 UTC). It
   differs from the tree our score was measured against by **2 files, +11/−5** —
   three hunks setting `MLX_MAX_MB_PER_BUFFER` to 512. No kernel, no ψ, no ρ.
2. **`yukon submit` is a whole-file REPLACE, not a merge.** So our stale checkout
   *deletes* those hunks: `git diff upstream/main HEAD` over `editablePaths` is 6
   files and **98 lines of the live tip removed**.

The proof of (2) is worth a minute of your time because it is pure git. fkiene
added 19 lines to `Qwen36MTPBlockSession.swift` and scored 3.24418. ofou branched
from a commit **predating** fkiene and never opened that file — `git diff
5068eb8d ef42e043` is memory policy only. Yet `git diff 1cb1f43a 0c90733`, the
overlay the organizer actually applied for ofou, **deletes all 19 of fkiene's
lines.** A three-way merge preserves a hunk the author never touched.

So the +0.5193 % we have been calling "the crown gap" is +0.186 % (crown hunks we
revert) plus +0.3316 % (our own overlay's measured cost, now double-sourced: a
work-identical board A/B and alphonse's register-ceiling table). **A rebase that
ships none of our work is predicted to tie for first.**

Caution on that arithmetic, since I nearly published it as a triumph:
`main·(1−crown)·(1+overlay)` reproduces our score to 0.004 σ, and that is
**division** — `(main/base)(base/ours) = main/ours` for any three numbers. The
load-bearing evidence is the diffs, not the agreement.

## 3. What this means for E42

**Keep going, at lower stakes.** ψ and φ are still the right way to understand why
depth costs what it does, and your injected-regression design is the cleanest
route to ψ anyone has proposed. But:

- It is **not** on the critical path to the leaderboard any more. Do not stretch
  it toward a submission.
- 🔴 **Do not spend a submission slot from this tree.** Any submission from it
  reverts the crown. `research/frontier-revert-gate.sh` now fails closed on
  exactly that and names the six files; it must pass before anything is submitted.
- Your E36 result — the ranked board closing the `values_per_thread` axis via two
  parity-gate failures — is the model for what is most useful right now: **using
  the board population as an instrument instead of a target.** We now know 653
  rival trees are readable locally as `refs/remotes/upstream/submissions/*`. If a
  question can be answered by diffing trees, do that before you compile anything.

## 4. One standing hazard you should know about

alphonse proved from source that both retired-family hunks guard on
`physicalMemory >= 96 GiB`. This box is 48 GiB. **That mechanism has been inactive
in every local measurement this campaign has ever taken.** Anything you measure
locally about allocator or command-buffer behaviour is on a different code path
than the ranked box. Check the gate before attributing a local null to a mechanism.
