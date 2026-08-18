# E21 legality argument: depth-preserving row declination

Assignment `qwen38-r1-e21-depth-preserving-row-declination`, PR #25.
Base `c0f7e370921a14f348fa1872f2176b1b43028752`.

This document is required to exist and to be committed **before any timed arm
is spent**. If the argument could not be made cleanly, the assignment says to
stop and report that as the result. It can be made cleanly, and the reason is
narrower and stronger than expected: **declination is not a new mechanism at
all.** It is the selection of a draft count the contract already offers, using
a strict subset of the inputs the shipped schedule already reads.

## 0. The mechanism in one sentence

`costModelDepth(offeredDepth:roundCount:)` is the function bound to
`draftPolicy` at `init`, and its return value *is* the round's draft count at
the single call site

```swift
let draftCount = draftPolicy(depth, roundCount)
```

Declination returns `0` from that function on rounds the schedule predicts will
accept nothing. Nothing else changes.

## 1. Returning zero is an existing, contract-blessed value

`program.md`, "Scored Path", step 2 of the round:

> Choose zero through eight draft tokens within the limit offered by the parent.

and, in "Goal And Score":

> On each round, the candidate may choose any draft count from zero up to the
> limit offered by the parent... Choosing zero is a useful serial control

Zero is not a loophole; it is the low end of the declared choice set. The
shipped `costModelDepth` already returns it:

```swift
let cap = Swift.min(Swift.min(offeredDepth, Qwen36MTPLimits.maxDepth), widthCap)
guard cap > 0 else { return 0 }
```

so the zero-draft round is an already-implemented, already-warmed, already-
exercised code path. E21 does not add a path; it changes *when an existing path
is taken*.

## 2. (a) At which stage is a row declined?

**Before the row exists.** `costModelDepth` runs at scheduling time — round
step 2, "choose the draft count" — which is strictly earlier than round step 3,
"use the MTP head to propose those drafts". A declined row is therefore a draft
row that is never proposed, never built by the head, never appended to the
verify batch, never submitted to the target, and never counted in the row
ledger.

The verify batch of a declining round contains exactly one row: the pending
primary token. That is bit-for-bit the depth-0 serial contract, which the
program explicitly names as a legal control.

There is no intermediate state in which a declined row is "created but
suppressed". The distinction matters for §4: a row that never existed cannot be
an *unevaluated verify row*, because it is never declared to the parent as a
verify row in the first place.

## 3. (b) Proof a declined row can never be emitted

A token leaves the session by exactly two routes:

1. as the committed **primary** token, or
2. as an **accepted draft** — an element of `drafts[]` at an index below
   `acceptedCount`.

Route 2 is closed by construction. `drafts[]` is populated by the head-proposal
loop, which iterates exactly `draftCount` times. With `draftCount == 0`,
`drafts` is empty, so `acceptedCount == 0` and there is no index to emit from.
The round emits exactly one token: the primary, which is the token the serial
target already selected.

The draft chain is linear — each proposal conditions on the previous one — so
the proposable set within a round is always a *prefix*. There is no way to
decline row *k* while proposing row *k+1*. E21 therefore only ever declines
**whole rounds**, which collapses the emission proof to the trivial case above:
the empty prefix emits nothing.

## 4. Against `AGENTS.md:231-233` — unevaluated verify rows

The prohibited list names

> degraded or partial target forwards, **unevaluated verify rows**, omitted
> exact final-logit/top-two work for emitted rows, or fabricated top-two, row,
> acceptance, or ledger evidence

Every clause is about **claiming work that was not done**: declaring a row to
the parent, or emitting a token, without having run the real target forward and
reported its exact evidence.

Declination moves in the opposite direction on every clause:

| clause | E21 behaviour |
| --- | --- |
| degraded/partial target forward | unchanged; the primary row gets the same full forward |
| unevaluated verify rows | strictly **fewer rows declared**; every declared row is still fully evaluated |
| omitted final-logit/top-two for emitted rows | unchanged; the emitted primary still gets exact top-two |
| fabricated row/acceptance/ledger evidence | ledger still closes at `rows_evaluated == rows_declared`, at a smaller honest count |

A round that declares one row and evaluates one row is exact. The prohibition
forbids `declared > evaluated`; declination reduces both together. This is the
same accounting the depth-0 serial control already produces.

## 5. Against `AGENTS.md:242-244` — head may improve proposal quality and cost

> Input-independent weight/kernel/shape tables and ordinary within-request cache
> reuse are legal when they preserve the contract. **A candidate head may
> improve proposal quality and cost; it cannot redefine the target answer.**

Declination is purely a **proposal-cost** decision, which is the sanctioned
half of that sentence. It never touches the target: the emitted token is
selected by the same unchanged target forward over the same unchanged
checkpoint, and its top-two evidence is produced by the same code. Declining
changes how many proposals the candidate pays for. It cannot change what the
target says, because on a declining round the target is asked exactly the
question the serial model asks.

Note also that the head is *not modified* by E21 at all. The pinned head
(`head_provenance.sha256 = 07293af7...`) is byte-identical across both arms;
only the count of head invocations differs.

## 6. Against `AGENTS.md:237-238` — rollback leaving rejected rows reachable

> logical rollback that leaves rejected cache/state rows reachable, or merely
> decrements an offset without trimming or overwriting rejected physical rows

A declining round **performs no rollback, because it creates no rejected
rows.** There is nothing to trim and no offset to decrement. Concretely, on a
declining round the session does not:

* invoke the head, so no head-cache row is appended and none must be replayed;
* append speculative rows to the target KV cache, so the 48 Gated DeltaNet
  recurrent states and 16 full-attention caches advance by exactly the one
  committed primary row — the serial advance;
* take a rollback snapshot, so no snapshot can be stale or partially restored.

The post-round reachable state of a declining round is therefore a **subset**
of the shipped round's reachable state, not a divergence from it. The class of
bug this rule guards against — reachable rejected state — is not merely avoided
but made unrepresentable, since the round never produces a rejected row.

## 7. The real hazard, and why this predicate avoids it

The genuine risk in any "skip work" rule is that the predicate reads something
it must not: the future, the reference rows, the prompt identity, the benchmark
phase, or state carried between requests. `program.md` forbids all of these,
and `Qwen36MTPReferenceSession` may never influence timed generation.

The E21 predicate reads exactly three scalars, and **the shipped schedule
already reads all three at the same point in the same function**:

| scalar | already read by shipped `costModelDepth`? |
| --- | --- |
| `pendingTop2` margin `v[0] - v[1]` | yes — the depth-0 `conf` and depth-1 `conf2` terms |
| `positionAcceptEMA[...]` | yes — `var p = positionAcceptEMA[depth]` |
| `fullAcceptStreak` | yes — selects `widthCap` |

This gives the strongest available form of the argument:

> **The declination predicate is a different function of strictly the same
> inputs the shipped schedule is already trusted to use. It opens no new
> information channel. If the shipped schedule is legal, the predicate is
> legal.**

Each input is also verified to be within-request only:

* `pendingTop2` is the target's own top-two on the *pending primary*, produced
  by the previous round's own verify forward. It is target output about a token
  already committed — not lookahead.
* `positionAcceptEMA` and `fullAcceptStreak` are `private var`s on
  `Qwen36MTPBlockSession`, seeded at declaration and updated only in
  `recordAcceptOutcome`. `begin()` guards `!began` and throws `alreadyBegun`,
  so a single session serves exactly one decode trajectory; a second request
  cannot observe the first one's accumulators. This is ordinary within-request
  reuse, explicitly permitted above.

And the predicate is *phase-blind*: it contains no reference to the fixture,
the prompt, the golden, the run mode, wall-clock time, or the trace gate. It
computes the same output for the same session state regardless of whether the
run is a reference pass, a serial control, or a ranked leg.

## 8. What this argument does **not** license

Stated explicitly so a later reader does not over-extend it:

* It does not license declining a row *after* the head has proposed it in order
  to skip its target check. That would be an unevaluated verify row under §4.
* It does not license mid-chain declination. §3 shows the chain is a prefix; a
  hole in it is not representable.
* It does not license any predicate that reads outside the three scalars in §7.
  Any future signal must re-run this argument, because the "strict subset"
  property in §7 is what carries the whole case.
* It does not license changing what the target answers for an emitted token,
  under any depth.

## 9. Conclusion

Depth-preserving row declination is legal. It selects an already-offered draft
count of zero, at the scheduling stage, from a strict subset of the inputs the
shipped schedule already consumes; it declares fewer rows and evaluates every
row it declares; it cannot emit a row it never created; and it leaves strictly
less reachable state than the shipped path, with no rollback to get wrong.

The experiment may proceed to measurement. Whether it *helps* is an empirical
question, answered by the pre-registered arms in `research/e21-prereg.md`.
