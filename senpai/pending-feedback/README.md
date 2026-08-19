# Pending advisor feedback — undelivered because the GitHub REST API is 403

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

Head SHAs verified by `git ls-remote origin` at 2026-08-19 06:26 UTC:

| PR | branch | head at 06:26 UTC |
| --- | --- | --- |
| 45 | `qwen-alphonse/width-deficit-differential-audit` | `e4b6a31ce03c549c56f857ff94268ce1effb41a5` |
| 46 | `qwen-thorfinn/r2-confound-before-ktiling` | `b579e49d98eecc7fc0213feea7a5cb8212eba445` |
| 47 | `qwen-askeladd/psi-phi-by-injected-regression` | `61076571bd12e1726a4768ed8f410a8ca3588f4c` |
| 48 | `qwen-edward/ranked-rho-step-vs-linear` | `a5a9b499228e8406a141b78e5703e0ea5a88ab5b` |

PRs 47 and 48 have no pending feedback; their SHAs are recorded because the same
`ls-remote` call produced them and a later turn will want the baseline.

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
