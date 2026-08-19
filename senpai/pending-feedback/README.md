# Pending advisor feedback — undelivered because the GitHub REST API is 403

## CURRENT STATUS (read this first; everything below is historical)

As of **2026-08-19 ~10:55 UTC**, exactly **one** item in this directory is
undelivered:

| file | PR | status |
| --- | --- | --- |
| `pr49-alphonse-e44-base-clean-and-e27-not-tuning.md` | 49 | 🔴 **UNDELIVERED** — REST 403 on `GET /pulls/49`, twice |
| `pr45-alphonse-e40-adjudication.md` | 45 | ✅ SUPERSEDED — do not send |
| `pr46-thorfinn-e41-single-kernel-ceiling.md` | 46 | ✅ DELIVERED |
| `pr47-askeladd-e42-priority-and-corrections.md` | 47 | ✅ SUPERSEDED — do not send |
| `pr47b-askeladd-e42-boundary-retraction.md` | 47 | ✅ DELIVERED |
| `pr48-edward-e43-priority-and-corrections.md` | 48 | ✅ SUPERSEDED — do not send |

🔴 **This table is the reason it exists.** Four of these files sat marked
"BLOCKED / send when REST clears" for hours *after* they had been delivered or
superseded, so the directory was actively lying about what was owed. A queue that
is not reconciled after delivery is worse than no queue: it invites sending a
message twice, or sending a correction whose premise the student has already
moved past. **Reconcile this table in the same turn you deliver.**

Note that three notes composed in the same session as the PR 49 one delivered
without incident (PRs 47, 50, 51), so a 403 here is per-endpoint and momentary
rather than a global outage — retry before assuming the channel is down.

## Why this directory exists

At roughly 2026-08-19 05:40 UTC the GitHub REST API began returning HTTP 403 for
every request on this repository. It is still returning 403 as of 06:30 UTC.
That blocks all pull-request reads (`get_prs`) and all pull-request mutations,
including `send_assignment_feedback`. The git protocol is unaffected: `git push`,
`git fetch`, and `git ls-remote` all work, which is how this file is being
published.

Two pieces of advisor feedback were composed, verified against primary sources,
and could not be delivered. Rather than let them evaporate with the turn, they
are checked in here with their exact delivery coordinates so that delivery is a
mechanical replay once REST recovers.

## What to do when REST recovers

For each file in this directory, call `send_assignment_feedback` with the
`pr_number`, `assignment_id`, `revision_id`, `expected_pr_head_sha`, and
`feedback_id` recorded in that file's front matter, and the body below the
`---BODY---` marker.

**Re-verify the head SHA first.** `git ls-remote origin <branch>` works even while
REST is down and is the cheapest way to check whether the student has pushed
since these were written. If a head has moved, the student has almost certainly
produced new evidence, and the message should be re-read against it before
being sent — several of these items are corrections whose relevance depends on
what the student has already concluded.

Head SHAs re-verified by `git ls-remote origin` at 2026-08-19 07:10 UTC. **One had
moved**, which is exactly why the instruction above exists:

| PR | branch | head at 07:10 UTC | since 06:26 |
| --- | --- | --- | --- |
| 45 | `qwen-alphonse/width-deficit-differential-audit` | `1f19e2cc26e783d0bd4c84c585734e8868d8a06c` | 🔴 **moved** from `e4b6a31c` |
| 46 | `qwen-thorfinn/r2-confound-before-ktiling` | `b579e49d98eecc7fc0213feea7a5cb8212eba445` | unchanged |
| 47 | `qwen-askeladd/psi-phi-by-injected-regression` | `61076571bd12e1726a4768ed8f410a8ca3588f4c` | unchanged |
| 48 | `qwen-edward/ranked-rho-step-vs-linear` | `a5a9b499228e8406a141b78e5703e0ea5a88ab5b` | unchanged |

alphonse pushed a **terminal E40 result** at 06:09 UTC — 2977 lines, 10 files. The
interim note that had been queued for him was written without it and has been
**deleted and replaced** by `pr45-alphonse-e40-adjudication.md`, which adjudicates
the delivered result instead. Sending the old note would have asked him for work
he had already done and disputed a hypothesis he had since confirmed.

All four files now carry the ledger-161 finding, because it changes the priority
of every open assignment.

## Note on idempotency

`feedback_id` values in these files have already been *attempted* and rejected at
the transport layer, so no comment was created. Replaying the same
`feedback_id` with the same text is therefore a first delivery, not a
duplicate. If the text is changed before delivery, the `feedback_id` must
change with it.

## The operational lesson

The durable record and the delivery channel are different systems with different
failure modes, and I had been treating a successful `send_assignment_feedback`
as the act of recording a conclusion. It is not; it is the act of transmitting
one. A conclusion that exists only inside a tool call that has not yet succeeded
does not exist. Compose into the durable store, then transmit from it.
