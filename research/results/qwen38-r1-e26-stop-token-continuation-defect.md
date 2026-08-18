# E26 — the stop-token continuation defect

- **Assignment:** `qwen38-r1-e26-stop-token-continuation-defect`, revision `r1`, PR #30
- **Student:** `qwen-askeladd`
- **Base:** `0ac14570a0c26b803c8f84594a307e92402d98cc` (`senpai/qwen38-mtp-r1`)
- **Control arm:** `020c6b5a8a9eb373d95b1ed816bead54fc324947` (base + the compile fix in §10.1)
- **Candidate arm:** `07345f4015e319d201fc59872c3adb7c1ae6e2bc`
- **Host:** M4 Pro, `applegpu_g16s`, NAX off. **The ranked host is M5** — see §11.
- **Track:** `qwen3.8-27b-mtp-v1`; head `559b24eb…` (pinned, §9.4)
- **Result label:** local winner (correctness), **insurance rather than speedup** — see §7

## 1. Report shape

**Question.** At the ranked 512-token decode window, does the editable block
session at the campaign base emit the full window, or does it abort once the
serial trajectory produces a stop token?

**Evidence that made it worth testing.** `Qwen36MTPBlockSession.swift` carries
two stop-token overlay sites that the trusted parent has no counterpart for, and
`program.md` names the resulting failure mode explicitly: *"An editable session
that stops at EOS and later throws `notBegun` is a solver defect, not permission
to shorten the ranked contract."* No campaign record contained a 512-token exact
replay; two ledger rows (`8b85909`, `28e591f`) say *"full 512-token exact replay
still required"*.

**Expected result.** The base aborts inside the ranked window with
`.notBegun`; deleting both overlay sites emits all 512 tokens and matches the
serial trajectory exactly, with unchanged behaviour before the first stop token.

**Smallest decisive test.** One reference golden of 513 rows, then matched
`128:2 / 256:2 / 512:2 / 512:0` legs on both arms against that single golden,
plus a 300–303 boundary sweep on the control to locate the abort exactly.

**Stop or promotion rule.** Pre-registered as four falsifiers (§3). Advance only
if all four survive.

## 2. Headline

The base **cannot complete a ranked-length leg on the public gate fixture at
either draft depth.** It emits exactly **302** tokens and then throws
`.notBegun`. The candidate emits all **512** tokens at depth 2 and at depth 0,
matches the serial trajectory token-for-token, and closes the row ledger.

| primary metric | baseline (`020c6b5`) | candidate (`07345f4`) | direction |
| --- | --- | --- | --- |
| `max_local_decode_tokens_with_all_tokens_matched` | **302** | **512** | maximize |

512 is the ranked window, i.e. the ceiling of this metric, not merely a larger
number than 302.

## 3. Pre-registered falsifiers — all four survived

Pre-registration: PR #30 comment `e26-prereg-r1`, posted before any arm ran.

| # | falsifier | outcome |
| --- | --- | --- |
| F1 | candidate 512 leg not matching, or `residual_divergence_count > 0`, or a `first_divergence_index` | **survived** — both 512 legs `all_tokens_matched=true`, `residual_divergence_count=0`, no divergence index |
| F2 | any bitwise divergence from the control on the stop-free 128/256 legs | **survived** — all 14 compared counters equal and `effective_draft_lengths` element-wise identical |
| F3 | control at 512 not dying at the fixture's first stop index | **survived, with one correction** — it dies, at emitted index **302**, not 301 (§4.2) |
| F4 | row-ledger closure or `cacheOffsetInvariant` failure | **survived** — closure holds on every passing leg |

**Correction recorded on evidence.** The pre-registration named baseline `301`.
The measured boundary is `302`; §4.2 gives the off-by-one that caused the
mis-statement. `primary_metric.baseline` uses the measured `302`.

## 4. What actually breaks

### 4.1 The runtime stop set

`resolveQwenMTPStopTokens` (`Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift:92-117`)
unions the scalar `eos_token_id` from `weights/config.json` (`248044`), the list
`eos_token_id` from `weights/generation_config.json` (`[248046, 248044]`), a
non-null `pad_token_id` (`248044`), and `tokenizer?.eosTokenId`. The files alone
give exactly **`{248044, 248046}`**. The worker resolves this at `:188` and hands
it to both the warmup session (`:195`) and the timed session (`:198`).

### 4.2 The off-by-one, and why the boundary is 302

`QwenRuntimeMTPDriver.swift:216-226` checks emitted token `index` against
`index == 0 ? golden.referenceSeedToken : golden.rows[index-1].sequentialArgmax`.
The emitted stream is therefore
`[referenceSeedToken, golden[0], golden[1], …]`, so

```
emitted-stream index = golden index + 1
```

In the 513-row golden there is exactly **one** stop hit, at golden index **300**
(token `248044`), i.e. **emitted index 301**, the **302nd** emitted token, with
212 reference tokens after it. Independent cross-check: the fixture's own
recorded 1024-token continuation has its first `248044` at index **301**.

A session that commits the stop token and then dies therefore completes **302**
tokens. My pre-registration conflated the golden index with the emitted index.

### 4.3 Site 1 — the live abort

`Qwen36MTPBlockSession.swift:795-813` at the control:

```swift
if stopTokens.contains(primary) {
    reachedStopToken = true
    …
    pendingPrimary = nil; pendingTop2 = nil; pendingHidden = nil
    return Qwen36MTPRoundResult(tokens: committed, declaredRows: 1, …,
                                reachedStopToken: true)
}
```

The primary has already been committed at `:779-781`, so the stop token *is*
emitted. What kills the next round is niling the pending trio: `generateRound`
opens with

```swift
guard began, let primaryPending = pendingPrimary, let tailPending = pendingTop2,
      let hidden = pendingHidden else { throw .notBegun }
```

(`:745-747`). The parent's window loop (`while emitted.count < options.totalTokenCount`,
`QwenRuntimeMTPDriver.swift:121`) does not stop asking, so the next call throws.
The observed failure is exactly this:

```
mlxfast-swift: runtime worker mtp_decode_round failed:
MTP round requested before the seed prefill
```

### 4.4 Site 2 — provably dead, and load-bearing anyway

`:1186-1191` truncates `committed` after a stop token:

```swift
if let stopIndex = committed.firstIndex(where: { stopTokens.contains($0) }) {
    dropped = committed.count - (stopIndex + 1); …
}
```

`dropped == 0` **always**, for two reasons together: Site 1 guarantees `primary`
is not a stop token by the time control reaches Site 2, and the accept loop
(`:1028-1033`) increments `acceptedCount` and then breaks immediately on a stop
token, so a stop token can only ever be the *last* element of `committed`. Its
only live effect is setting `reachedStopToken`, which nothing reads (§5.4).

**The two deletions are a matched pair.** Deleting Site 1 alone makes Site 2
live: the primary would then be a stop token at index 0 of `committed` with
accepted drafts after it, so `dropped > 0` would decrement `committedTokenCount`
without trimming the caches, and the next round would trip the
`cacheOffsetInvariant` throw at `:766-771`. A partial revert fails loudly rather
than silently, which is the behaviour we want from a landmine.

### 4.5 Why the trigger is schedule-dependent, not fixed

`draftPolicy` is **not** a constant 2. The worker installs
`costModelDepth(offeredDepth:)` (`:197`), an adaptive EMA cost model that walks
`depth` while `reach > h*(1+expected)/(1+depth*h)`, with `positionAcceptEMA`
clamped at depth 0/1 by a sigmoid of the top-2 logit margin, and a `widthCap`
that switches on `fullAcceptStreak`. Whether the stop token lands on a *round
primary* (fatal) or inside an *accepted draft prefix* (survivable — the
full-accept and reject branches both repopulate `pendingPrimary` and continue) is
therefore emergent and history-dependent.

Two consequences:

- **At depth 0 the abort is guaranteed** whenever any stop token appears in the
  window, because every emitted token is a round primary. Confirmed: control
  `302:0` passes, `303:0` fails.
- **The trigger probability is coupled to every future scheduling experiment.**
  A hidden prompt that survives today can start aborting after an unrelated
  depth-policy change, and the adaptive policy is free to choose 0 drafts for an
  individual round at any time. This is the strongest argument for removing the
  overlay now rather than when it next fires.

## 5. Why this is misalignment, not alignment

### 5.1 `program.md` names it a defect

> "An editable session that stops at EOS and later throws `notBegun` is a solver
> defect, not permission to shorten the ranked contract."

### 5.2 The serial reference has no stop-token logic at all

`grep -c stopToken Sources/MLXFastModel/Qwen36MTPReferenceSession.swift` → **0**.
Site 2's comment claims it keeps the stop token *"— the same rule the serial
reference applies"*. That is **factually false about the code it cites**, and it
is the sentence most likely to have justified re-adding the overlay.

### 5.3 The trusted side has no stop-token semantics

Whole-tree `grep` for `stopToken` under `Sources/` finds it only in the editable
session, in `QwenRuntimeMTPWorker.swift` (resolve + two session constructors), in
a *comment* in `QwenRuntimeMTPDriver.swift`, and in the **retired** enum case
`stopTokenInsideWindow = "stop_token_inside_window"` (`QwenRuntimeMTP.swift:106-108`,
whose own comment says *"Nothing throws it"*). The driver's operator-ratified
note (`:172-192`) is explicit about what survived retirement:

> "A round must still commit at least one token, **the loop still runs until the
> parent's own configured total is reached** … a trajectory that genuinely
> terminated early cannot short the window silently. **It shows up as a token
> mismatch, which is the stronger signal**, not as a short denominator."

The base's actual behaviour is worse than the mismatch the operator anticipated:
a hard throw that produces no score at all.

### 5.4 `reachedStopToken` is read by nothing

Zero occurrences in either copy of the driver/worker. The pre-existing test
`theTrustedDriverStillOwnsTheWindowLengthByCount` asserts exactly that, so the
field's own comment ("the parent stops asking") was never true — the tripwire
suite already contained the refutation of the code it was guarding.

### 5.5 The session's own bookkeeping models a stop token as ordinary

`recordAcceptOutcome` (`:681-685`) says the position at the accepted count
"observed a failure only if the walk actually rejected there **(not when it ended
early on a committed stop token)**", and `:691` computes `stoppedEarly` from a
stop token *inside the accepted prefix*. The scheduler already treats a stop
token as a normally committed token; only Site 1 disagrees.

## 6. Evidence

Every leg is the trusted CLI's own timed report, checked against **one** shared
513-row golden generated through `Qwen36MTPReferenceSession` (untouched by E26,
so one golden is valid for both arms).

`rc` is the leg exit code; `replay` is `verify_block_replayed_round_count`;
`coff` is `target_cache_offset_final`; `spt` is `parent_measured_seconds_per_token`.

| arm | leg | rc | rounds | acc | rej | replay | rejchk | emit | decl | refchk | coff | match | spt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 128:2 | 0 | 43 | 85 | 0 | 0 | 0 | 128 | 128 | 128 | 640 | true | 0.057842 |
| control | 256:2 | 0 | 86 | 171 | 0 | 0 | 0 | 256 | 257 | 257 | 768 | true | 0.041513 |
| control | 512:2 | **1** | — | — | — | — | — | — | — | — | — | **`.notBegun`** | — |
| control | 512:0 | **1** | — | — | — | — | — | — | — | — | — | **`.notBegun`** | — |
| control | 300:2 | 0 | 100 | 200 | 0 | 0 | 0 | 300 | 300 | 300 | 812 | true | 0.038960 |
| control | 301:2 | 0 | 101 | 200 | 1 | 0 | 1 | 301 | 302 | 302 | 813 | true | 0.039133 |
| control | **302:2** | 0 | 102 | 200 | 1 | 0 | 1 | 302 | 303 | 303 | 814 | true | 0.039398 |
| control | **303:2** | **1** | — | — | — | — | — | — | — | — | — | **`.notBegun`** | — |
| control | 301:0 | 0 | 301 | 0 | 0 | 0 | 0 | 301 | 301 | 301 | 813 | true | 0.078759 |
| control | **302:0** | 0 | 302 | 0 | 0 | 0 | 0 | 302 | 302 | 302 | 814 | true | 0.078756 |
| control | **303:0** | **1** | — | — | — | — | — | — | — | — | — | **`.notBegun`** | — |
| cand | 128:2 | 0 | 43 | 85 | 0 | 0 | 0 | 128 | 128 | 128 | 640 | true | 0.057750 |
| cand | 256:2 | 0 | 86 | 171 | 0 | 0 | 0 | 256 | 257 | 257 | 768 | true | 0.041576 |
| cand | **512:2** | 0 | 176 | 336 | 16 | 5 | 16 | **512** | 528 | 528 | 1024 | **true** | 0.034147 |
| cand | **512:0** | 0 | 512 | 0 | 0 | 0 | 0 | **512** | 512 | 512 | 1024 | **true** | 0.073514 |

**The boundary is exactly 302 at both depths.** Corroborated out of band by wall
clock: the failing control `512:2` took 36 s, the same as the passing `302:2`
(37 s), and the failing `512:0` took 49 s against the passing `302:0` (50 s) — the
abort is at 302, not somewhere later.

**Ledger closure** on every passing leg: `accepted + rejected + tails == declared_rows == reference_checked`,
`tails == round_count`, `coff == 512 + emitted`. Candidate `512:2`:
`336 + 16 + 176 = 528 = declared = reference_checked`, `coff = 1024`.

**Fidelity beyond token equality** on candidate `512:2`: `parity_all_ok=true`,
`residual_divergence_count=0`, all **16** rejected rows reference-checked with
`max_rejected_tail_logit_delta = 0`.

**New code coverage.** Candidate `512:2` is the only leg in this experiment with
`verify_block_replayed_round_count > 0` (5 replays) and a double-digit rejection
count. The rollback/replay/repair machinery past the stop token is territory the
base **cannot reach at all**, and the 128/256 legs never enter it.

### 6.1 F2 in detail

| leg | counter diffs (14 counters) | `effective_draft_lengths` |
| --- | --- | --- |
| 128:2 | none | identical, len 43, sum 85, digest `815843eaa649271a` both arms |
| 256:2 | none | identical, len 86, sum 171, digest `a747b5590815b90e` both arms |

The digest is `sha256(",".join(effective_draft_lengths))[:16]` as computed by
`draft_schedule_digest` in `research/e26_log_wandb.py`, so the equality above is
reproducible from the archived leg JSON rather than from an ad-hoc one-liner.

Because the work is bitwise identical here, the `spt` gap on these legs is a
clean **noise-floor estimate for this host: ±0.16 %** (128:2 −0.16 %, 256:2 +0.15 %).

**Disclosed limitation, pre-registered:** the 128/256 gate has **no power in the
changed region**. Both windows are provably stop-free, so they prove the change
is free, not that it is correct past a stop token. The 512 legs and the boundary
sweep carry that burden.

**Transitive parity substitution.** My harness does not extract a `row_ledger` or
the emitted-token array from the report (§9.2), so "compare emitted token arrays
between arms" is not literally satisfiable. Substituted: both arms are checked
token-for-token against the **same** 513-row golden with
`all_tokens_matched=true`, `residual_divergence_count=0` and
`declared == reference_checked`, which makes them transitively equal on the
overlapping prefix, **plus** element-wise `effective_draft_lengths` equality,
which pins the schedule as well as the tokens.

## 7. This is insurance, and the promoted receipt says so

Honest counter-evidence, stated prominently: **a promoted submission carrying
this defect scored 3.13098700135133 on the ranked 8-prompt run.** Promotion means
every gate passed, so the defect did **not** abort any hidden prompt in at least
one ranked run. Either those windows contain no stop token, or their stop tokens
happened to land inside accepted draft prefixes rather than on round primaries.

So E26 does not repair a currently-failing ranked run, and I claim no score from
it. What it does:

1. **Removes a hard-abort landmine** whose trigger is emergent (§4.5) and coupled
   to every future scheduling experiment, and **guaranteed** at depth 0.
2. **Unblocks 512-token local measurement on this base.** `program.md` requires
   *"measure credible candidates against a fresh same-host base over 512 decode
   tokens"* and says a 256-token result is not a ranked-equivalent headline. On
   the control that instruction is **impossible to execute** — it dies at 302 at
   both depths. Every future candidate on this lineage needs E26 first.
3. **Supplies the evidence the ledger twice recorded as missing** (§8).

This matches the ledger's own judgement of the fix: *"insurance, not speedup: it
enables legs that would otherwise fail, and it does not move the score."*

## 8. The regression, and its exact root cause

`senpai/campaign-ledger.md:159-185` records this defect as **CLOSED** at
`b85e782` (*"the fix is alignment, not a deviation"*), reached by source
inspection. Two things follow from measurement.

**8.1 It regressed.** `b85e782` *is* an ancestor of the assignment base, and it
had the overlay removed — yet the base has it back, in exactly the pre-fix shape:

| commit | Site 1 | Site 2 | `stopTokens` | `reachedStopToken` |
| --- | --- | --- | --- | --- |
| `5d02917` organizer challenge import | 1 | 1 | 6 | 7 |
| `b85e782` merge — ledger's "CLOSED" | 0 | 0 | 1 | 4 |
| `0ac1457` **assignment base** | **1** | **1** | **7** | **7** |
| `07345f4` candidate | 0 | 0 | 5 | 0 |

**8.2 The root cause is the frontier sync.** The overlay is **organizer-supplied**
code — it is present at `5d02917`, the original challenge import, and every
promoted submission is built from that source. So each
`Sync promoted organizer frontier` commit brings it back. Pickaxe on the Site 1
literal gives four completed cycles before mine:

| removed by | re-added by |
| --- | --- |
| `f1a874d` "continue fixed decode windows past EOS" | `330b44e` sync `32b94cb6…` |
| `b219009` (same title) | `d098212` sync `156b5b75…` |
| `8b85909` "Preserve fixed decode windows on promoted frontier" | `29f1ee4` sync `79683c63…` |
| `28e591f` (same title) | **`f04df93` sync `ed4dfd6b…`** |

Restricting to `b85e782..0ac1457`, exactly two commits flip Site 1 back on
relative to their parent, and both are frontier syncs: `29f1ee4` (parent
`1c57496`, Site 1 = 0) and **`f04df93`** (parent `1d573f6`, Site 1 = 0).
`29f1ee4` was compensated by `28e591f`; **`f04df93` was never compensated**, and
that is precisely why the defect is live at the base.

This generalises the ledger's existing standing rule. Alongside *"a frontier sync
may not reduce the test-file count"*, the sync needs a **content** check:
`grep -c 'if stopTokens.contains(primary)' Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
must be `0` after every sync. The test that would have caught it was itself
deleted a fourth time by `bc552e5` — the two defects protect each other, which is
why four fixes did not stick.

## 9. Method, and what I deliberately did not do

### 9.1 Change under test

`020c6b5 → 07345f4` touches exactly two files, 79 insertions / 81 deletions:

- `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (53 lines) — delete Site 1
  and Site 2 and the unread `reachedStopToken` field. Result: `stopTokens` goes
  7 → 5 (declaration, initialiser parameter, assignment, the `stoppedEarly` EMA
  term at `:689`, the accept-loop cap at `:1011`), `reachedStopToken` 7 → **0**.
  The retained sites are the ones that treat a stop token as an ordinary token.
- `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift` (107 lines) — re-adjudicate
  the tripwire on evidence and re-point its source assertions; test renamed
  `theEditableSessionContinuesPastEosWithoutTheOverlay`.

Both arms were committed and the worktree clean before every launch.

**Which SHA was measured.** Every number in this report was produced at
`020c6b5` (control) and `07345f4` (candidate), and those two commits are
immutable — I did not amend them. The branch head is one commit *later* than
`07345f4`, adding this report, `research/e26_log_wandb.py`, and a comment-only
correction to the header and one doc comment of
`Tests/MLXFastTests/QwenMTPFixedWindowTests.swift` (the previous revision of that
header mis-stated the boundary as token 151645 at position 300 and claimed a
tripwire had fired; see §11). That follow-up commit touches **no** file under
`Sources/`, so runtime behaviour at the branch head is bit-identical to the
measured `07345f4`; verify with
`git diff 07345f4..HEAD --stat -- Sources/` (empty) and
`git diff 07345f4..HEAD -U0 -- Tests/ | grep -vE '^[+-][[:space:]]*(//|///)'`
(no non-comment lines).

### 9.2 Harness

`research/e26-legs.sh` (`golden` | `legs ARM TOKENS:DEPTH …`) lives in the
**control** commit, so both arms ran a byte-identical harness. It is research-only,
never compiled, and not in `editablePaths`. The wrapper could not express what
this needed: several token windows against **one** golden, and a failing leg
recorded rather than aborting the sweep, because *a leg that aborts is itself the
observation*.

Known wart, disclosed rather than fixed mid-experiment: the reports contain no
`row_ledger`, no emitted-token array and no `first_divergence_index`, so the
`ledger-*.json` / `accepted-*.json` files are empty and the script's own header
comment overstates what it captures. Fixing it would have dirtied the tree and
broken harness identity between arms. §6.1 gives the substitute. (I am reporting
a false source comment in §5.2; it is only fair to flag my own.)

Also disclosed: the trusted CLI binary is **byte-identical across both arms**
(`.build/release/mlxfast-swift`, mtime `2026-08-18T05:45:59`), because it does not
link `MLXFastModel`. Only the worker differs.

### 9.3 Reproduction

```bash
export MLXFAST_QWEN_MTP_HEAD_DIR="$HOME/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared"

git checkout 020c6b5 && research/rebuild.sh
research/e26-legs.sh golden                                   # once; shared by both arms
research/e26-legs.sh legs base-020c6b5           128:2 256:2 512:2 512:0
research/e26-legs.sh legs base-020c6b5-boundary  300:2 301:2 302:2 301:0 302:0 303:0 303:2

git checkout 07345f4 && research/rebuild.sh
research/e26-legs.sh legs cand-07345f4           128:2 256:2 512:2 512:0
```

Fixture `correctness_prompts/public_longcopy_gate_english_512_1024.json`; 512-token
seed; artifacts under `$HOME/e26-stop-token/` (outside Git). Job IDs: golden
`673c8331`, control `9eb8c1d6`, boundary `1252390d` + `35e92c5e`, candidate
`2b8a5bc1`.

### 9.4 Head provenance

Identical in both arms: single `model.safetensors`, 427,742,600 bytes, sha256
`559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71` — an exact
manifest match, 40 tensors including `draft_lm_head.*` (a genuine q2/q4 rerank
head). Every leg reports `mtp_head_attached=true`,
`uses_native_mtp_head=uses_pinned_mtp_head=true`.

### 9.5 Swift tests

`swift test --force-resolved-versions -c debug`, 672 tests in 41 suites, both
arms. Compared as **sorted failing identifiers**, never totals.

- Candidate `07345f4` (job `a2a6448c`, 40.8 s): 38 issues, **12** distinct
  failing identifiers, **0 compile errors**.
- Control `020c6b5` (job `5261bfa2`, 29.0 s): 38 issues, **12** distinct failing
  identifiers, **0 compile errors**.
- `diff <(sort control/failed.txt) <(sort cand/failed.txt)` is **empty** ⇒ all 12
  are pre-existing on the base and E26 introduces **zero** new failures. They are:
  suites `QwenMTPHeadDeclarationTests`, `QwenMTPScoringSemanticsTests`,
  `QwenMTPTrackNamingTests`; tests
  `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen`,
  `participantDocsExposeDefaultCLIInstallDirectory`,
  `qwen36ConfigContractDigestMatchesTheReferenceManifest`,
  `submissionStaticReviewPromptCoversMeasurementStructureExploitation`,
  `theCheckedInDeclarationSelectsThePinnedHead`,
  `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`,
  `theQwenMTPTrackIsArmedOnQwen38`,
  `theSeededCalibrationExpectationMatchesItsRecordedProvenance`.
- `QwenMTPFixedWindowTests` and `QwenMTPFixedWindowSourceGuardTests` pass in the
  candidate; `theEditableSessionContinuesPastEosWithoutTheOverlay()` passes
  (10.126 s). `QwenMTPFixedWindowSourceGuardTests` also passes in the **control**
  (9.816 s) — as §11 explains, that is by design, because the control's version of
  the guard asserted the truncating shape. A green source guard in both arms is
  therefore not evidence for either arm; the 512-token legs are.

### 9.6 Scope and budget

Two files, both in `editablePaths` groups (`Sources/MLXFastModel/Qwen36MTP*.swift`
and `Tests/`, which Yukon does not submit). The change is net **−2 lines** of
source and removes a field, so it cannot approach the 262,144-byte growth cap or
the per-file and total source limits. No target weights, tokenizer, goldens,
trusted driver, timing, telemetry, workflow, fixture, package graph or submission
manifest were touched. `Package.resolved` unchanged; no
`swift package resolve`/`update` was run.

## 10. Base-hygiene defects found

### 10.1 The base does not compile — reported as PR #30 `issuecomment-5326329091`

At `0ac1457`, `swift test` fails with **0 failing tests and 3 compile errors**:
`Qwen36MTPBlockSession.swift:987,989,991,994 cannot find 'Qwen35Attribution' in scope`
and `:992 cannot infer contextual base in reference to member 'topTwo'`.
`public enum Qwen35Attribution` was added by `70155d7` (E20) in
`Vendor/mlx-swift-lm/.../Qwen35.swift`; at the base the enum has **0** occurrences
while **5** call sites survive. Stale prebuilt `.build-worker` binaries hid it.

This is not cosmetic: `benchmark.json` submits `Sources/MLXFastModel/**`, so a
submission cut from this base **fails its ranked build**. Fixed in the control
commit `020c6b5` by deleting the 5 orphaned call sites and keeping
`let (top2IDs, top2Values) = Self.linearTopTwoRows(verifyLogits)`. The vendor enum
was deliberately not restored — E20's attribution mode is not part of E26.
`grep -rn "Qwen35Attribution" Sources/ Tests/` is now empty. Because the control
arm carries this fix, both E26 arms are compiled from source.

### 10.2 Stale `mlx.metallib`

The worker warns on every run that the metallib was built from different vendored
Metal sources (recorded `3dd0ffd6…`, current `7ed19d64…`). Harmless for E26 — it
is identical in both arms and no Metal source changed — but it **silently masks
any kernel experiment on this base**, which is exactly the false-pass hole the
ledger records as cross-cutting defect 3.

## 11. Limitations

- **Host.** M4 Pro, not the ranked M5. The *correctness* finding is
  host-independent — `.notBegun` is control flow, not numerics — but no timing
  number here transfers.
- **No speed claim, as pre-registered.** `MLXFAST_MACMON_BIN` was deliberately
  omitted from every leg to keep the harness byte-identical between arms, so
  there is **no per-leg thermal record** (`gpu_temp_c=unknown`). The wrapper's
  40 °C cool gate was bypassed because this host idles above it
  (`cool_gate=BYPASSED_this_host_idles_above_40C`); an out-of-band sample at
  09:42Z read GPU 40.58 °C / CPU 42.02 °C / GPU 0.0073 W, confirming no
  concurrent GPU tenant but not licensing a speed claim.
  For completeness only, and **not** a claim: candidate `512:0 / 512:2` gives a
  local serial-to-MTP ratio of **2.153**, and control `302:0 / 302:2` gives
  **1.999**.
- **Hidden prompts.** I cannot see whether their 512-token windows contain stop
  tokens. §7 states the promoted-receipt counter-evidence.
- **Cross-student serialisation.** `research/await-lock-then-run.sh` cannot help
  here: its lock path is under `${MLXFAST_CACHE_ROOT:-$HOME/.cache/mlxfast}` and
  `HOME` is role-scoped, so the lock is role-private and cannot serialise across
  students. Per-leg wall time was my contention tripwire, and the 128/256 legs
  reproducing to ±0.16 % across arms indicates none occurred.
- **`research/capture-cli.sh` not used**, deliberately: my harness writes
  permanent per-leg files and avoids that script's `03-*`/`04-*` leg-mixing
  hazard when several legs share one arm directory.
- **One fixture.** The public gate fixture only. `public_longcopy_gate_english_512_256.json`
  is a null control for this mechanism (0 ids ≥ 248000, max 94032), which is a
  second reason the standard local modes never see this.
- **My own test-file header was wrong first, and I am correcting it here rather
  than quietly.** The revision of
  `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift` that shipped in the measured
  `07345f4` claimed a tripwire had fired and inherited a factual error — "token
  151645 at position 300" — from the pre-existing header. Neither is true: the
  stop token is **248044 at emitted index 301**, and nothing tripped on its own.
  A source guard that asserts today's shape passes for exactly as long as the
  shape holds, and the control's version of that guard
  (`theEditableSessionTruncatesAtEosToday`) asserted the *truncating* shape, so it
  passed at the control **by design** — it documented the defect instead of
  detecting it. The boundary is E26's own bisection (302 pass / 303 abort at both
  depths), not a tripwire. The correction is comment-only and is in the follow-up
  commit described in §9.1.

## 12. Suggested follow-ups (not implemented)

1. **A sync content gate.** Add
   `grep -c 'if stopTokens.contains(primary)' Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
   == 0 to `sync-organizer-frontier`, next to the test-file-count rule. Four
   fixes have not stuck (§8); a source-inspection conclusion in the ledger is
   not a regression gate.
2. **Restore the `.build-worker` freshness check.** §10.1 was invisible because
   stale binaries satisfied the run while the source could not compile. A build
   that is newer than its sources should fail loudly.
3. **Rebuild `mlx.metallib`** (`tools/build-mlx-metallib.sh`) before any kernel
   experiment on this base (§10.2).
4. **Sweep for other orphaned E20 symbols.** `MLX_QWEN_ATTRIB` and the rest of
   the attribution surface may have further half-reverted call sites; §10.1 was
   found by accident.
5. **Fix `research/e26-legs.sh` ledger extraction** (§9.2) so future experiments
   get real `row_ledger` and emitted-token arrays, and correct its header comment.
6. **A throughput idea, unmeasured.** The accept-loop cap at `:1011` still breaks
   the accept walk on a stop token. That is now the only remaining stop-token
   special case in the hot path, and with the window owned by the parent it may
   be unnecessary: letting the walk accept the full verified prefix could recover
   a token per stop-token round. Small, but it is a *speed* question, unlike E26,
   and it should be measured on its own.

## 13. Artifacts

- W&B run `qr1h0gl1` —
  <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/qr1h0gl1>
  (logged by `research/e26_log_wandb.py`: `legs_logged=15 ran=11`, the `legs`
  table, `falsifier/*_survived` all true, `primary/*`, `directional/*`). The four
  legs with `ran=false` are the control's aborts — published, not dropped.
- Leg reports: `$HOME/e26-stop-token/{base-020c6b5,base-020c6b5-boundary,cand-07345f4}/leg-t*-d*.json`
- Golden: `$HOME/e26-stop-token/golden-513.json` (513 rows)
- Swift test sets: `.mlxfast-private/swift-tests/{e26-control-020c6b5,e26-cand-07345f4}/`
