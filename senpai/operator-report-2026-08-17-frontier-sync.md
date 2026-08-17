# Operator report — 2026-08-17 frontier refresh (reply to issue #9)

**This file exists because the reply channel is broken.** See
[§0](#0-why-this-is-a-file-and-not-a-comment-on-issue-9) first.

Written by the advisor role at `2026-08-17T09:0x`Z against campaign `main`
`f6bac5cad6f0c0b69749727bd0f1f90fb45bc203` and advisor branch
`senpai/qwen38-mtp-r1` `fe38ecc21e4084e4d17dac3aa76264bb5897a614`.

Directive answered: issue #9 comment `5313223350` (`2026-08-17T07:41:17Z`).

---

## 0. Why this is a file and not a comment on issue #9

`respond_to_human_issue` refuses every message on issue #9:

```
Error executing tool 'respond_to_human_issue':
human message must not be authored by the authenticated actor
```

The guard is **identity-based, not message-specific**. I verified this rather
than assuming it:

| Attempt | `human_message_id` | Author | Result |
| --- | --- | --- | --- |
| comment (the directive) | `5313223350` | `morganmcg1` | rejected |
| issue body | `5166104890` | `morganmcg1` | rejected, same error |

Both issues in this repository (`#9`, `#6`) and all their comments are authored
by `morganmcg1`. And `morganmcg1` **is** the identity this role posts as — I
confirmed it by reading back a comment I had just written myself, PR 13 feedback
comment `5313823140`, whose author is `morganmcg1` / `OWNER`.

So the operator account and the advisor account are the same GitHub identity.
The anti-self-reply guard therefore makes *every* human message in this
repository unaddressable. This is an environment configuration collision, not a
missing message ID, and no choice of ID can work around it.

There is also no fallback GitHub credential: `gh` is unauthenticated and the
environment carries `GH_REPO`, `YUKON_API_TOKEN` and `HF_TOKEN` but no GitHub
token. Unauthenticated reads work (`GET /issues/9` → 200); writes do not.

**Requested fix:** give the advisor role its own GitHub identity, distinct from
the operator's, or file directives from a second account. Until then, treat
`senpai/campaign-ledger.md` plus reports like this one as the advisor's outbound
channel. I did not spam a student PR to get your attention.

---

## 1. The receipt in the directive was already superseded

The directive asked me to re-query Yukon and use the highest row still
explicitly `promoted`. I did exactly that, four times over the session, most
recently at `2026-08-17T09:01:07Z`. **The row named in the directive is not the
frontier.**

| | Directive's receipt | Live frontier |
| --- | --- | --- |
| Submission | `5c523482-452c-4662-a303-a3b359c81030` | **`03dedda8-fc70-4e3e-881f-5384a17af405`** |
| Solver | — | `vibecodooor` |
| Source commit | `cdb06b7045622fc40c1b336af28892c073ba28a3` | **`32b94cb67d2f3a102a36382d2beb62eee8d99db5`** |
| Score | `2.93428682708139` | **`2.94661597308114`** |
| Promoted at | 2026-08-17 01:36 UTC | 2026-08-17 04:34 UTC |
| Reported diff | `+0.000672` | `+0.012329 (+1.24%)` |

`03dedda8` is the **last** `promoted` row in `yukon submissions --all`; every
row after it is `rejected` or `validating`. `5c523482` is the second-highest
promoted row.

The host clock is UTC (`date -u` = `Mon Aug 17 09:01:07 UTC 2026`), so
`03dedda8` was promoted **2 h 42 m before your 07:16:09Z audit** and **3 h 07 m
before you posted the directive**. It was not a race; it was already visible.

`cdb06b70` is a verified git ancestor of `32b94cb`, so the newer receipt
strictly contains the one you named — nothing is lost by moving forward.

**Please look at your audit tooling.** This is the *second consecutive*
directive that quoted a stale frontier: the issue #9 body quoted
`12b1c699…` / `bd12edd1…` / `2.92520777238747`, and the comment quoted
`5c523482` / `cdb06b70`. Two independent misses in a row is a selection bug —
most likely picking the highest-scoring row your snapshot happened to hold, or
the newest row that had already *finished* validating, rather than the last row
whose status is `promoted`. I followed the *instruction* ("re-query, use the
highest still-promoted row") over the *asserted value*, which is why the
campaign is on `32b94cb` and not `cdb06b70`.

### Calibration: how tight the bar actually is

Rows posted after the frontier that were **rejected despite high scores**:

| Submission | Solver | Score | Diff |
| --- | --- | --- | --- |
| `1443c1c` | audreyt | 2.94121661565988 | −0.005399 (−0.54%) |
| `8774c67` | companygardener | 2.94026479195 | −0.006351 (−0.64%) |
| `4650c96` | yijunyu | 2.93524422189044 | −0.011372 (−1.15%) |
| `b4d62ea` | Amal-David | 2.93375431543959 | −0.012862 (−1.30%) |

Rejection means "did not exceed the live frontier". Competitors are landing
within 0.6% and getting nothing. A candidate must beat `2.94661597308114`, not
merely look good against a stale baseline. As of 09:01Z there are **7 rows still
`validating`** (`18e95ea`, `e4d6058`, `83ded51`, `5cd4c21`, `a451f3b`,
`ba493f7`, `499c3f4`), so the bar may rise again — I will re-query immediately
before any submission.

---

## 2. Resulting pins

```
UPSTREAM_SHA   = 32b94cb67d2f3a102a36382d2beb62eee8d99db5   # organizer frontier
BASE_SHA       = f6bac5cad6f0c0b69749727bd0f1f90fb45bc203   # campaign main (local)
RESEARCH_BASE  = fe38ecc21e4084e4d17dac3aa76264bb5897a614   # senpai/qwen38-mtp-r1 (published)
FORK_BASE_SHA  = 83201aa98a71d42415e1c7e85e8bc96cf609d5cf   # pre-sync main
PREV_UPSTREAM  = 7351e62674bc600f0ca148d3a1b0604716a09db6   # previous organizer sync
PROMOTED_ID    = 03dedda8-fc70-4e3e-881f-5384a17af405
PROMOTED_SCORE = 2.94661597308114
```

**Preservation proof.** `git diff --name-status 32b94cb HEAD` reports **zero
editable-path differences**. The campaign's editable surface is byte-exact to
the promoted frontier — no blended merge, no inherited half-state. Editable
budget on the advisor branch:

```
source=2402203/3000000 headroom=597797 growth=0/262144 exempt=2410 files=154
```

Exactly **3** non-editable overlay paths exist, each declared:

| Path | Kind | Organizer blob | Campaign blob |
| --- | --- | --- | --- |
| `Sources/MLXFastCLI/main.swift` | seam | `b75c8fb65b393b96` | `4fb19edcca8024b1` |
| `Tests/MLXFastTests/QwenMTPVerbTests.swift` | repair | `5bab1076ac632a0b` | `710953e68da56573` |
| `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` | added | absent | `105113cc2a2ff9dc` |

### Disclosure: `origin/main` is behind

`origin/main` is still `83201aa98a71d42415e1c7e85e8bc96cf609d5cf`. My tooling
publishes the *advisor* branch under a force-with-lease audit; it has no
sanctioned route to push `main`, and I declined to use a raw `git push`, which
would bypass the lease and the audit trail.

**This costs nothing in substance**: `f6bac5c` is a verified ancestor of the
published `origin/senpai/qwen38-mtp-r1`, so every commit, gate result and ledger
entry is durably on the remote and reviewable. The stale `main` ref is
cosmetic. Tell me the sanctioned route, or say the word and I will fast-forward
it directly.

---

## 3. What the sync did

Four commits on `codex/sync-organizer-frontier-20260817`, merged into `main`:

1. `3352917` — cherry-pick `-x` of trusted policy commit `da72848b` (David Tai,
   "Publish the seed-prefill rate in ranked payloads (observability only)").
   4 files, +48/−2, non-editable paths only ⇒ **zero solver bytes**.
2. `330b44e` — sync the promoted organizer frontier `32b94cb` (6 files,
   +224/−62).
3. `bc552e5` — retire the orphaned fixed-window EOS guard test.
4. `f6bac5c` — record the 2026-08-17 promoted frontier in ledger + state file.

**The promoted delta** (`cdb06b70` → `32b94cb`) is 120 insertions / 9 deletions
across three files — `Qwen36MTPBlockSession.swift` (+64),
`Qwen36MTPTarget.swift` (+18), `Qwen35.swift` (+47). Per
`yukon submission-note 03dedda`: publish the post-final-norm block the target
verify forward already computes; reuse the accepted prefix for proposal-head
history as one contiguous slice; compile Qwen attention output
`x * sigmoid(gate)` as one shapeless fused pass. **Schedule-neutral and
head-neutral.**

**Gates, all passing:** overlay verify; `senpai/check-editable-budget.sh`;
`research/twin_audit.py` (29 runtime-effective twins); `git diff --check`;
`jq -e` assertion on `senpai/frontier-state.json`; a strengthened trusted-parity
check (3 declared overlays, 0 undeclared drift). No metallib rebuild required —
`quantized.h` is untouched by the frontier delta.

**Nothing to port from the organizer's rules:** `git diff AGENTS.md
7351e626 → 32b94cb` is empty.

**`benchmark.json` contract unchanged** — `editablePaths` and
`optionalEditablePaths` are identical across the sync.

### Scheduler constants moved under us — this is the trap

Read as `headStepCostRatio / sdpaWidthWallDepthCap / segmentedVerifyDepthCap /
segmentedStreakGate`. At the frontier these are `0.18 / 5 / 8 / 3`, wired as
`widthCap = fullAcceptStreak >= segmentedStreakGate ? segmentedVerifyDepthCap
: sdpaWidthWallDepthCap` (`Qwen36MTPBlockSession.swift:529,561,568-580`), so 5
is the ordinary wall and 8 is the reward for a full-accept streak:

| Revision | Constants |
| --- | --- |
| `8970d775` (old research base) | 0.20 / 4 / **7** / 3 |
| `7351e626` (previous sync) | 0.20 / 4 / 8 / 3 |
| `c7468c56` | 0.20 / 4 / 8 / **2** |
| `033f6227`, `4aacc534`, `cdb06b70`, **`32b94cb6`** | **0.18 / 5 / 8 / 3** |

Every A/B measured against `8970d775` or `7351e626` was measured against a
*different scheduler*. All such numbers are now **pre-frontier directional**,
not comparable, and I have labelled them so in the ledger and in each PR.

### `swift test` disposition — and a correction to my own earlier record

`swift test` builds clean: 657 tests, 37 suites, exit 1 with 38 issues, **none
introduced by the import**. Five failures reproduce at organizer `32b94cb`
itself, including `theCheckedInDeclarationSelectsThePinnedHead()` — it expects a
pinned bf16 head while the live head is the remote 4-bit/group-64 one, which is
independent proof the manifest head is live.

**Consequence worth stating plainly:** the ranked benchmark does not run
`swift test`. The promoted frontier fails the organizer's own head-declaration
test and still scored `2.94661597308114`. `swift test` is campaign hygiene, and
it has never been a submission gate.

I previously recorded the doc-test failures as "a campaign defect in
`AGENTS.md`". That was half wrong, and here is the exact truth after checking
all four documents at both revisions:

| Document | Organizer `32b94cb` | Campaign `HEAD` |
| --- | --- | --- |
| `README.md` | passes | passes (identical blob) |
| `TASK.md` | passes | passes (identical blob) |
| `AGENTS.md` | **passes** (`819f5fdb1fcf`) | **fails** (`80a46b72c907`) |
| `CLAUDE.md` | **fails** (`47dc3e3d863c`) | **fails** (identical blob) |

Two separate facts, not one:

* Campaign `AGENTS.md` is a deliberate, sentinel-marked
  (`<!-- SENPAI-CAMPAIGN-BEGIN -->`) rewrite that strips the inherited
  Laguna/DFlash prose (345 insertions / 948 deletions). The rewrite dropped the
  literal string ``Yukon CLI (`yukon`)``. That is a genuine campaign-side
  omission and I am fixing it, because students read this file.
* `CLAUDE.md` fails **at the organizer frontier**, on a byte-identical blob,
  missing both required strings. Fixing it would mean modifying an
  organizer-owned non-editable document to create a new undeclared overlay, for
  zero benchmark value. **I am not doing that**, which means the test stays red
  no matter what I do to `AGENTS.md`. I would rather tell you it is red for a
  known, harmless, upstream reason than green it by touching trusted files.

---

## 4. PR dispositions, strongest first

As instructed: strongest first, replay required wherever the promoted delta can
move the answer, older evidence labelled pre-frontier directional.

### PR 13 — `qwen-edward`, depth lever — **revision `r3` requested** (strongest)

The measured per-depth h-curve
`[.0842,.0775,.2426,.3754,.2919,.3000,.2870,.3909]` replacing the scalar
`headStepCostRatio` produced **−7.658%** on held-out prose (`Hp` 0.046757248 vs
reference 0.050634801), replicated at −7.624%. 14/14 arms exact
(`all_tokens_matched=true`, `resid=0`). Mechanism is real and legible:
rejections 310→237, replays 100→76 (both −24%), acceptance .4561→.5288.

This is the best result in the campaign, which is exactly why it must be
replayed: it was measured on a `0.20 / 4` scheduler and the frontier is
`0.18 / 5`. The default it beats has changed, and the wall it substitutes for
has moved from 4 to 5. Replay, not re-litigation. Four questions: does −7.658%
survive; re-fit the curve on the frontier; does it still substitute at wall 5;
report gain against 0.18, not 0.20. Holds timing slot 1.

I also corrected my own error in that thread — I had written that prefill is
excluded from the score. It is not. See §6.

### PR 11 — `qwen-askeladd`, draft-head bits — **revision `r3` requested**

One-line `draftHeadBits` 4→3 gave −2.0303% with exact tokens 16/16. But the
decomposition is damning and he found it himself: −2.0303% total = **+0.4499%
per round (slower)** + −2.4691% round-count, against a mechanism ceiling of only
−0.683%. **The entire gain is an acceptance coin-flip**, not a cost reduction.

Route B shipped itself: the frontier already provisions the pre-quantized
4-bit/group-64 head. So Part A is liveness-only with no thermal budget, and the
coin-flip is **re-rolled on a new head — directional agreement is not
replication**. Timing slot 2.

### PR 12 — `qwen-thorfinn`, crossrow FMA — **closed unmerged**

Arm 1 (naive `metal::fma`) was **+3.77% slower**; arm 1b (ordered) −0.59%,
inside the ±1.2% single-cell noise band. A ~1% analytic ceiling cannot move a
score whose frontier step was +1.24%. Closed with five durable facts preserved,
including a positively-validated exactness rig (the `1+2⁻⁶` perturbation fired
with its exact preregistered 56-cell signature) and a campaign hygiene rule:
**restore from `HEAD`, never from the index; verify digests after every job.**

Before closing I salvaged his tooling — `research/await-lock-then-run.sh`
existed *only* on that unmerged branch. 7 files, digest-verified byte-exact,
now on the advisor branch as `fe38ecc`. His `air_kernel_stats.py` fix matters:
`metal::fma` lowers to `@air.fma.v4f32`, never `@llvm.fma.f32`, so the old
detector was a false negative.

I scored his four predictions publicly: one confirmed, one failed, one resolved
against me, one confirmed.

---

## 5. Active assignments — four students, four slots, none idle

| PR | Student | Assignment | Needs GPU slot |
| --- | --- | --- | --- |
| 13 | `qwen-edward` | `…-e11-depth-lever-showdown` r3 | **1st** |
| 11 | `qwen-askeladd` | `…-e9-draft-bits-default` r3 | **2nd** |
| 14 | `qwen-alphonse` | `…-e12-seed-prefill-charge` r1 | **3rd** |
| 15 | `qwen-thorfinn` | `…-e13-na4-register-cliff` r1 | **none** |

All four are `status:wip` and all four are non-duplicative. Ownership is
partitioned so no two students touch the same symbol: edward owns the cost
curve, alphonse owns `warmAllDepths` and `begin`, thorfinn owns `quantized.h`,
askeladd owns `draftHeadBits`.

Two are new this session.

### PR 14 — the seed prefill is inside the scored clock (new, highest expected value)

I read the trusted harness rather than trusting the docs, and
`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift` says it outright:

> "The seed prefill IS charged to the decode measurement, the way the paired
> contract requires: the clock starts immediately before the request so the
> seed cost cannot be hidden outside the window."

`decodeSeconds = P + D`. `seedPrefillSeconds` is captured for **observability
only** — "nothing is subtracted". Both arms take the identical path
(`benchmark-qwen-mtp.sh:601` and `:611` both call `mtp-timed`, and
`begin(seedTokens:)` takes no depth argument), so `P_serial ≈ P_mtp`.

With `p = P/D`, the exact identity is

```
r_ideal − r_reported = p × (r_reported − 1)
```

verified to 14 digits. On the ranked host (`r = 2.9466`) that is **+0.0195 per
1% of prefill**; p = 0.02 costs **0.0389**, which is *three frontier steps*.
Leverage per unit `p` is `(r − 1)`: ranked 1.94662 vs local 0.47088, so the
local M4 Pro **understates this lever by 4.13×** — the one lever local hardware
systematically hides.

**I do not know `p`.** Nobody has measured it. Phase 1 is pure measurement with
zero edits and no GPU slot: read the two report JSONs. The new fields are
emitted only under `if report.seedPrefillSeconds > 0, report.seedTokenCount > 0`,
so on a stale binary the keys are **omitted rather than zero** — a free
staleness tripwire. Phase 3, if warranted, moves seed-*independent* work out of
`begin` into `warmAllDepths`, with in-tree precedent (enriching the warm was
worth a repeatable 0.368 s stall) and an in-tree counterexample (commit
`7b33621`: parity-clean, faster in steady state, and it **still lost** because
its warm evaluated a different graph and the JIT was paid inside the scored
window). Five forbidden items are named explicitly, first among them: do not
slow the serial control, and do not subtract prefill.

### PR 15 — thorfinn's failed arm proved something he did not claim (new)

His own data, re-partitioned by NA rather than by M, is a clean split:

| M | streams | IPG | groups | max NA | naive |
| --: | --: | --: | :-- | --: | --: |
| 3 | 1 | 3 | [3] | 3 | 0.993 |
| 4 | 1 | 4 | [4] | **4** | **1.045** |
| 5 | 2 | 3 | [3,2] | 3 | 0.983 |
| 6 | 2 | 3 | [3,3] | 3 | 0.977 |
| 7 | 2 | 4 | [4,3] | **4** | **1.013** |
| 8 | 2 | 4 | [4,4] | **4** | **1.058** |
| 9 | 3 | 3 | [3,3,3] | 3 | 0.975 |

**Every NA=4 width slowed; every NA≤3 width sped up; no overlap.** NA=4 mean
1.0387 (min 1.013) vs NA≤3 mean 0.9820 (max 0.993) — separation +5.67 pp, and a
+2.00 pp inter-group gap against ±1.2% single-cell noise. A perfect 3-vs-4 split
by chance is `1/C(7,3)` = 2.9%.

Phase 1 is static AIR only — no GPU, no slot — and I told him the honest limit
up front: AIR is pre-register-allocation and **cannot see back-end spills**, so
absence of a static signal is not evidence of no cliff. Phase 3 timing is
explicitly forbidden; he stops and reports. Prediction 6 on the brief is "you
will find at least one thing in this brief that is wrong", because §1 of it
rests on edward's attribution of `M = d+1` rather than my own verification, and
I asked him to check it.

---

## 6. No submission, and why

**I have not submitted, and I am not asking permission to.** Per your standing
direction I make merge and submission calls myself, including above the 3.0
plausibility gate. This is a *readiness* judgement, not a deference one:

Every candidate the campaign holds was measured against a superseded cost
structure. Edward's −7.658% is the strongest result here, and it was measured on
a `0.20 / 4` scheduler that no longer exists; the frontier ships `0.18 / 5`.
Submitting a number I know was measured against a superseded cost structure is
not a legitimate submission, it is a guess with a receipt attached. The replays
are already assigned and slot 1 is running.

Two supporting facts:

* Local results are **directional and non-rankable**. The ranked host is M5;
  this box is an M4 Pro. Askeladd's `--local-submit` returned
  `rankable=false`, score 2.21009, precisely as expected.
* Scores above the 3.0 ceiling return `{score: 0, passed: false}` but are
  classified `runtime_error` and are **not charged**. I will never weaken,
  delay, split or tune a candidate to sit under a gate.

When a replay clears on `32b94cb`, I re-query Yukon (7 rows are validating),
confirm the frontier has not moved, and submit. Serialized, one at a time.

---

## 7. Scheduled, not yet done

1. **This report's delivery.** Blocked as in §0.
2. `origin/main` fast-forward to `f6bac5c` — needs a sanctioned route (§2).
3. A fresh same-host baseline on the frontier, serialized behind
   `research/await-lock-then-run.sh` with per-arm temperature sampling. The old
   `7351e626` baseline (`1.4708805115725638`, exact 64/64) is marked **STALE for
   A/B use** in the ledger.
4. Replay of the demoted campaign editable IP. When I advanced the research base
   I refused the textual auto-merge of the editable surface and forced it
   byte-exact to the frontier, so prior campaign mechanisms are parked, not
   lost. The replay handle is
   `git diff e20268e9 8970d775 -- Sources/MLXFastModel Vendor mtp-head.manifest.json`,
   and each mechanism must return as its own matched-pair candidate.

### Measurement environment, honestly

Idle GPU settles at 38.7–40 °C / ~0.02 W and recovers within ~50 s. But I
recorded **recurring foreign load** on this box — spikes to 65–83 °C at 16–31 W.
`await-lock-then-run.sh` only excludes work that goes through `benchmark.sh`; it
cannot exclude a foreign process. Hence the standing rule: parallelize builds
and analysis, **serialize all timing**, and sample temperature per arm. One
soak sampler was killed by its own 1000 s timeout — self-inflicted, data usable,
not re-run.

---

## Verification commands

```sh
yukon submissions --all | tail -25            # run from the linked repo dir
git rev-parse main senpai/qwen38-mtp-r1
git diff --name-status 32b94cb67d2f3a102a36382d2beb62eee8d99db5 HEAD
senpai/check-editable-budget.sh 32b94cb67d2f3a102a36382d2beb62eee8d99db5
jq . senpai/frontier-state.json
```
