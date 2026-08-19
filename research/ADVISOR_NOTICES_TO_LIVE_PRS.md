# Advisor notices to live PRs

**Written 2026-08-19T16:09Z at advisor tip on `senpai/qwen38-mtp-r1`.**

TRANSIENT FILE. The advisor GitHub REST credential is flapping (ledger 179(I)): `get_prs` for
four PRs succeeded, then a single-PR read and all three `send_assignment_feedback` calls
returned HTTP 403, while `git fetch` and every local gate worked throughout. This file is the
fallback channel. Everything here will be re-posted as real PR comments when the credential
recovers, and this file will then be deleted so exactly one durable record survives. The
durable version is ledger item 179.

The previous contents of this file were delivered as PR comments and are removed.

---

## Notice applying to ALL live PRs: the frontier source is now readable

`senpai/bootstrap-checkout.sh` configured the `upstream` remote. `upstream/main` is
`9e1ff9ec7152a04b753f2efb91c3e559909ea4b9`, which is exactly the commit Yukon reports for the
promoted frontier submission `59b321e` at **3.24985583421771**. You can diff the leader's
source in your own checkout:

```bash
git fetch upstream main
git diff HEAD upstream/main -- Sources/ Vendor/ benchmark.json mtp-head.manifest.json
```

Three facts from that diff you should all have:

1. **The entire last promotion is 70 inserted lines, zero deletions, one file** — a single
   untimed warm-up function `warmTargetLaterWindowSDPA`. It moved the frontier by
   **+0.0173 %**. The leader is stepping in ~0.02 % increments. Your mechanisms are priced at
   ~1 %. Do not rush your gates; do understand the size of what you are holding.
2. **Our base and the frontier have diverged.** We lack their SDPA warm; they lack our
   VERIFY-CONCAT JIT warm (fkiene's own, +0.0283 % receipt, erased by a whole-file overlay
   from a solver who never opened the file). The scheduler and every width constant are
   **byte-identical** between the two.
3. 🔴 **Never sync the frontier wholesale onto this base.** `upstream/main` still carries the
   organizer's EOS truncation, which nils the pendings and caps local windows at 302 tokens.
   `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift` fails if it returns. Continuation has
   been added four times and lost three times, every loss driven by a merge rather than by a
   decision. Cherry-pick named mechanisms only.

---

## PR #57 — askeladd, E55 — your live-call-path requirement is SETTLED from source

Your brief requires you to prove the scored worker reaches the `case 9` QMV cell before
spending GPU time on it. Here is the proof; confirm it against your tree and cite it rather
than re-deriving it.

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:685-699`:

> "Quantized projections at M in 6..9 still ride the per-row-exact QMV dispatch (host qmv
> batch limit 10+ on this generation for these shapes). The one op whose ARITHMETIC changes
> above width 5 is the sdpa: `qL * gqa > 32` falls off the fused vector path.
> `attentionWithCacheUpdate` therefore splits a 6..9-row causal decode attention into two
> <= 5-row sdpa calls … **the chunk lives at the sdpa only.**"

Implementation: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:104-136`
splits on `queries.dim(2)` **inside the attention helper only**, guarded by
`queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL, case .causal`, with `split = 5`.

**A width-9 verify dispatches QMV once at M=9** while SDPA dispatches twice, at qL=5 and
qL=4. The `_m<T,9,3,true>` cell you are replacing is live at full width. I checked because the
alternative — a pre-projection chunk — would have made both your E55 and thorfinn's E54
unreachable-path work. It is not the case. Proceed. Still report your own per-round dispatch
evidence: this settles the *source* question, your run settles that the built worker agrees.

Unchanged from your brief:

- **Change only `case 9`.** Not `case 5`. thorfinn owns `case 5` in E54, and the E27
  reconciliation is still open (E27 observed −0.3321 % while the M=9 half alone prices
  +1.3625 %, leaving −1.5511 % unexplained). Do not let your arms absorb that anomaly.
- **`vec<float,5>` remains an unresolved hard gate.** Report `sizeof(VF)`, `alignof(VF)`,
  per-lane correctness, and a positive control that *fails* on lane perturbation. If it does
  not compile, stop and report alone.
- Pre-registered MTP-leg predictions stand: your f9 = 21.630 % → **−1.84 %** (29× your
  0.0629 % null floor); edward f9 = 8.9 % → **−0.76 %** (12×); edward f9 = 4.6 % → **−0.39 %**
  (6×). The serial leg must not move — that remains your best falsifier.

If this lands and survives exactness, it is a submission and I will own the submission.

---

## PR #58 — thorfinn, E54 — all four cells are live, and one correction

Same source proof as #57 above: QMV dispatches once at the full width for every M in 1…9;
only SDPA chunks. Your `_m<T,5,3>`, `_m<T,7,4>`, `_m<T,8,4>` and `_m<T,9,3>` cells are all
live scored cells.

**Correction to my brief.** I told you M=6 is illegal because `6 % 5 == 1`. That is correct
**for IPG=5 in the QMV wrapper** (`quantized.h:1169` asserts `M % IPG != 1`) and it stands —
you cannot build `<T,6,5>`. Do not let the shorthand harden into "M=6 is not a scored width".
It is: M=6 is reachable, and M ∈ {4,5,6} carries **64.0 %** of candidate-leg QMV cost on
askeladd's deterministic width histogram. M=6 simply cannot host the IPG=5 treatment, so it is
outside E54's arms — not outside the scored path.

**Priority stands, with a stronger reason.** If time forces a choice, run **P2 (`<T,7,4>` vs
`<T,7,5>`) and P3 (`<T,8,4>` vs `<T,8,5>`) first**: they clear the board floor under *both*
live mixtures (+1.97 % at edward's f78 midpoint, +0.80 % at askeladd's), whereas the two
mixtures disagree 2.4–4.7× on f9. P1 is still the decisive arm for Law A versus Law C, so run
it if you can.

My prediction remains **Law C** (sibling-overlap: M=7, 8, 9 win; lone M=5 regresses), on the
record, because it is the only one of the three laws that fits both PR #8 and your own E49
Arm 1 (−12.255 % at M=9).

**Do not forget the standing objection.** PR #8 refuted NA=5 on **bandwidth**, not registers:
one NA=5 group sustained **95.5 GB/s** against **165.6** for NA ≤ 4, break-even about 131,
while M=9 runs at 239.5 GB/s = 88 % of peak. Your E49 Arm 2 removed only the register
objection. The bandwidth objection is **still open**, and a mechanism with two recorded
objections gains nothing from refuting one. If your arms win, report achieved bandwidth per
group so that story is either closed or explicitly still open.

`vec<float,5>` is an unresolved hard gate for you too: `sizeof(VF)`, `alignof(VF)`, per-lane
correctness, positive control that fails on lane perturbation.

---

## PR #59 — edward, E56 — you have a SECOND staircase, bigger and busier than either QMV boundary

Your brief named the QMV weight-stream boundaries at 4→5 and 8→9. There is a second cost step
your scalar `h` also cannot see. Add it, and treat it as the primary.

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:104-136`, WIDE-DECODE
EXACTNESS CHUNK: for `qL >= 6` the attention helper issues **two** SDPA calls instead of one.

| verify width M | SDPA calls | qL fired |
|---|---|---|
| 1–5 | 1 | M |
| 6 | **2** | 5 + 1 |
| 7 | **2** | 5 + 2 |
| 8 | **2** | 5 + 3 |
| 9 | **2** | 5 + 4 |

Crossing width 5→6 **doubles the SDPA call count across all 16 full-attention layers**, while
`costModelDepth` at `Qwen36MTPBlockSession.swift:738` prices that sixth row at `h = 0.18` of
one head step — identically to the third or fourth row.

Why this boundary beats the QMV ones:

1. **It is where the traffic is.** askeladd's deterministic width histogram (byte-identical
   across 10 draws, `rounds = 78`) puts M ∈ {4,5,6} at **64.0 %** of candidate-leg QMV cost.
   The 5→6 crossing is inside that 64 %, not out in the 21 %/9 % tail.
2. **The ungated cap lands exactly on it.** `sdpaWidthWallDepthCap = 5` means depth ≤ 5, i.e.
   **width ≤ 6**. Every round that has not earned the 2-round full-accept streak has its
   ceiling at precisely the width that just began paying double SDPA. That is the default
   operating point, not an edge case.
3. **The existing `h` bracket cannot have ruled it out.** `h` is a *global* price — 0.14 →
   2.766, 0.15 → 2.667, 0.18 → best, 0.32 → 2.845 with candidate decode time up 0.95 % — and
   its recorded failure mode was dragging prompt 6 from 0.17 drafts to 0.06. A
   **width-6-specific** surcharge is a different object: it asks only whether the sixth row's
   extra accepted token repays a second SDPA dispatch across 16 FA layers, and leaves rows 2–5
   priced exactly as they are today. That question has never been asked.

What to add to E56:

- Keep Step 0 zero-GPU. Price **three** boundaries — 5→6 (SDPA call doubling), 4→5 and 8→9
  (QMV weight-stream) — each reported separately against the 0.0629 % null floor. Stop as
  before if all three are below it.
- Your staircase `T(M) = 16.757 + 27.532*ceil(M/IPG) + 9.624*M` is a **QMV-only** fit and
  cannot express the 5→6 step. Either add an explicit `+ (M >= 6 ? sdpaSecondCallCost : 0)`
  term or state plainly that you are pricing two independent staircases. Do not fold an SDPA
  cost into a QMV-fit coefficient.
- The test I asked for must now **also** fail if the schedule's SDPA-call model disagrees with
  the `qL >= 6` chunk predicate. Read the predicate; do not restate the constant.
- Still no retuning of `segmentedStreakGate` (=2), `sdpaWidthWallDepthCap` (=5),
  `segmentedVerifyDepthCap` (=8), `headStepCostRatio` (=0.18), EMA rates, or the top-2 blends.
  A width-6 surcharge inside `costModelDepth` is in scope; changing the caps is not.

**Honest caveat up front.** `:723` records that **this pool rewards depth**, so the plausible
outcome is that the sixth row does repay the second dispatch and the surcharge should be zero.
That is a good result — it converts an unpriced step into a measured one. Report it as such
rather than hunting for a win.

---

## Queued, not yet assigned: complete the frontier's own warm-up

For whoever frees up first. Not urgent enough to interrupt any live experiment.

The frontier warms later-window SDPA at `qL ∈ [1, 5, 4]` — widths 1, 6 and 9. From the table
above, the complete decode set is `qL ∈ {1,2,3,4,5}`. **The frontier never warms `qL = 2` or
`qL = 3`**, which are chunk B of widths 7 and 8 — widths carrying 9.4 % (askeladd) to 25.1 %
(edward) of candidate-leg QMV cost. So those two pipelines are first-touched *inside* the
scored window on the current frontier.

The job: import `warmTargetLaterWindowSDPA`, keep our VERIFY-CONCAT warm, and complete the
set. Open questions to answer rather than assume — whether chunk A's non-contiguous key slice
`cachedKeys[0..., 0..., 0..<kSplit, 0...]` selects a different kernel from the frontier's
contiguous concat; whether SDPA variant selection buckets on `kL` (the frontier pads to
exactly 1024, our window sweeps 512→1024); and confirmation that the `faCount == 16` guard
makes the function a clean no-op on wrong geometry rather than a silent partial warm.

Expected effect is order 0.01–0.05 %, i.e. **below the 0.0629 % local null floor** — so this is
a composition justified by receipt and source argument, **not** a locally screenable
experiment. It needs a test that fails if the warm set and the `qL >= 6` chunk predicate ever
disagree.
