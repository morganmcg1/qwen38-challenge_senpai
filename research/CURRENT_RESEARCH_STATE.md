# SENPAI Research State

- **2026-08-19 16:09 UTC**
- Track `qwen3.8-27b-mtp-v1`; advisor branch `senpai/qwen38-mtp-r1`;
  `BASE_SHA = a2c3dbc497fd76b3e4f99c529a3eb5e8b2090abf`;
  `UPSTREAM_SHA = 0c90733d383f6b987a29682bf9eb9458a6172bfa` — the value actually synced into
  this tree. The organizer tip is now `9e1ff9ec7152a04b753f2efb91c3e559909ea4b9` and the
  trusted delta between the two is **empty**, so a future sync has no contract work to do,
  only editable cherry-picks (ledger 179(G)).
- Most recent human research direction: issue #31 asked for a maintenance checkpoint, which
  is complete and closed. **Issue #22 — execute aggressively toward the winning frontier —
  is the standing directive.**

This is a **living document**. The per-experiment record of authority is
[`../senpai/campaign-ledger.md`](../senpai/campaign-ledger.md); durable measurements,
source-line citations and closed questions live in
[`ESTABLISHED_FACTS.md`](ESTABLISHED_FACTS.md). The previous 2955-line version of this file is
retained verbatim as [`RESEARCH_STATE_ARCHIVE_2026-08-19.md`](RESEARCH_STATE_ARCHIVE_2026-08-19.md).
Keep this file to current hypotheses, live slots, and where we go next. Prune it every round.

---

## Where we stand

| quantity | value |
|---|---|
| live promoted frontier | **3.24985583421771** — submission `59b321e`, solver fkiene, commit `9e1ff9ec…` |
| our best official submission | **3.23250848263467** — receipt `ca9251b`, candidate `2b0c36a`, rejected |
| **our deficit** | **0.01734735158304 = 0.534 %** |
| our official submissions | six, under solver `morganmcg1`; four scored, none promoted |
| board floor (between-submission) | 0.7678 % |
| local end-to-end null floor | **0.0629 %** — askeladd's E48 `base2` arm |
| ranked score leverage `psi_mtp` | **0.693391** [0.692292, 0.694490] |
| the leader's last step | **+0.0173 %** |

**The strategic fact of this round: the leader is advancing the board in ~0.02 % increments
while our in-flight mechanisms are priced at ~1 %.** The entire 3.24986 promotion is 70
inserted lines, zero deletions, one file — an untimed warm-up. Our deficit is 31× that step.
We are not chasing an unreachable leader; we are chasing small steps with large tools.

Confirmed causal boundary, re-verified this round (`senpai/verify-ranked-score-boundary.sh`
PASSES): candidate-editable code cannot move the ranked serial numerator, so any compliant
edit that lowers candidate MTP seconds per token improves every affected `raw_p`. Never
subtract a locally measured serial share such as `psi_serial` when pricing official value.

---

## Current research focus and themes

### 1. The QMV dispatch table is the main attack surface, and its cells are provably live

`quantized.h` dispatches `qmv_fast_crossrow_affine4_g64_m<T,M,IPG,true>` on verify width M.
The design rule is `IPG = ceil(M / ceil(M/4))` and the binding constraint is
`static_assert(NA >= 2 && NA <= 4)` at `:980`, with accumulator `typedef vec<float, NA> VF`.
Raising a cell to NA=5 is the mechanism under test.

**Settled this round from source (ledger 179(D)):** a width-M verify dispatches QMV **once at
the full M** for every M in 1…9. Only the SDPA chunks, at `split = 5`, inside
`attentionWithCacheUpdate`. So `case 7`, `case 8` and `case 9` are live scored cells. The
alternative — a pre-projection chunk — would have made two live experiments unreachable-path
work. It is not the case.

Measured: thorfinn's `<T,9,5>` isolated cell win is **−12.255 %** (MDE 0.333 ms, nine
byte-identical controls ≤ 0.33 %). The register-tax objection is **refuted** (no dose-response;
ceiling |dScore| ≤ 0.1435 % shipped-referenced, ≤ 0.0876 % control-free). The **bandwidth**
objection from PR #8 is **still open**: one NA=5 group sustained 95.5 GB/s against 165.6 for
NA ≤ 4, while M=9 already runs at 88 % of peak.

🔴 `vec<float,5>` remains an unresolved hard gate. No local `vec` alias exists in the kernel
headers, so it resolves to `metal::vec`, which MSL specifies only for N ∈ {2,3,4}. thorfinn's
build compiled and ran, so something works. Every NA=5 experiment must report `sizeof(VF)`,
`alignof(VF)`, per-lane correctness, and a positive control that fails on lane perturbation.

### 2. 🔴 THE CENTRAL OPEN DISAGREEMENT — the narrow-width cost split

askeladd (E48) and edward (E53) roughly **swap** the split and **agree** on its total:

| share of candidate-leg QMV cost | askeladd E48 | edward E53 |
|---|---|---|
| M ∈ {4,5,6} | 64.025 % | 65.0–68.9 % — agree |
| M ∈ {7,8} | **9.391 %** | **21.2–25.1 %** |
| M = 9 | **21.630 %** | **4.6–8.9 %** |
| {7,8} + 9 | 31.02 % | 25.8–34.0 % — agree |

Neither is a GPU measurement; both are inferences from published board telemetry, and edward
declined to claim falsification. **This decides which mechanism we ship**, because under
edward's mixture the ranking *inverts*: E49's `<T,9,5>` falls to +0.47…+0.92 % (straddling the
0.7678 % floor) while E44 r2's M ∈ {7,8} rises to +1.13…+1.30 % (clears everywhere).
**PR #57 settles it by direct measurement** — the hypotheses are 2.4–4.7× apart in the
quantity that instrument reads, and all three predictions are ≥ 6× the 0.0629 % null floor.

### 3. The draft-depth schedule is mispriced at TWO staircases

`costModelDepth` at `Qwen36MTPBlockSession.swift:738` is a greedy marginal-cost walk that
prices every extra row with one scalar `h = 0.18`. Two real cost steps are invisible to it:

- **width 5→6 doubles the SDPA call count** across all 16 full-attention layers (the `qL >= 6`
  exactness chunk). This is the higher-traffic boundary: it sits inside the 64 % of QMV cost
  carried by M ∈ {4,5,6}, and `sdpaWidthWallDepthCap = 5` puts the ungated ceiling **exactly**
  on it.
- **QMV weight-stream boundaries at 4→5 and 8→9**, where marginal cost steps 9.624 → 37.156.

`h` is bracketed on both sides (0.14 → 2.766, 0.15 → 2.667, 0.18 → best, 0.32 → 2.845) so the
*global* price is a local optimum. A **width-specific** surcharge has never been tested, and
the bracket cannot have ruled it out. Honest prior: `:723` records that this pool rewards
depth, so the surcharge may well be zero — which is still a result worth having.

### 4. Warm-up and compile placement is the frontier's own current lever

Both the leader's last promotion (+0.0173 %) and the mechanism we hold that they lack
(+0.0283 %) are untimed warm-ups that move first-touch Metal compilation out of the scored
window. This class is real, free, and token-neutral, but sits **below our 0.0629 % local null
floor** — so it is justified by receipt and source argument and **composed into a candidate**,
never screened as a local A/B.

### 5. Measurement discipline that now gates every claim

- The **0.0629 % end-to-end null floor** (E48 `base2`) is 12.2× finer than the board floor and
  292× finer than the old smoke A/A sd. Every effect claim is now expressed as a multiple of it.
- Board intervals are **identification intervals, not standard errors** — 152 content-distinct
  trees reproduce the published telemetry to 16 digits, so the board gives ~1 observation, not
  371 draws. Never attach a sd multiple to a board-derived quantity.
- `psi_serial` is **NOT IDENTIFIABLE** locally: four treatments imply 0.7966/0.8694/1.0414/
  1.2470, two exceeding 1.0. It is unidentified, not refuted.
- Ungated timing (`MLXFAST_LOCAL_COOL_GATE=0`) is permitted only ABBA-counterbalanced, with
  entry/exit temperatures recorded and `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false` preserved verbatim.
- Log W&B **per leg while timing**, never at session end. A launch retag destroyed a workspace
  mid-session and made three legs permanently unrunnable.

---

## Live experiment slots — all four occupied, no GPU idle

| PR | student | experiment | state |
|---|---|---|---|
| #55 | alphonse | E51 — exactness-wall dose ladder; Step 0b accepted (safe math confirmed, refuted my prediction 1) | `status:wip` r1 |
| #57 | askeladd | E55 — compose `<T,9,5>` onto the real shipped table, end-to-end, to a **submittable candidate**; settles theme 2 | `status:wip` r1 |
| #58 | thorfinn | E54 — lone-versus-sibling NA=5 law across M=5/7/8/9; I predicted **Law C** on the record | `status:wip` r1 |
| #59 | edward | E56 — stream-aware draft-depth schedule; now three boundaries including 5→6 | `status:wip` r1 |

Merged this round: **#53** (thorfinn E49, W&B `92a0u0fl`), **#52** (askeladd E48, W&B
`yd949eze`), **#56** (edward E53, zero GPU by design). Base advanced
`981e69a` → `1247c57f` → `a35bb006` → `a2c3dbc4`, each with base-change inertness verified
rather than assumed.

---

## Potential next research directions

Ordered by expected value, not by convenience.

1. **Submit whatever #57 produces, if it survives exactness.** Our last submission missed by
   1.69 %; a 0.39–1.84 % MTP-leg win on a 0.534 % deficit is a submission, not a screen.
   Official evaluation is part of the research loop.
2. **Complete the frontier's warm set** (queued, unassigned). The leader warms `qL ∈ [1,5,4]`;
   the complete decode set is `{1,2,3,4,5}`. `qL = 2` and `qL = 3` — chunk B of widths 7 and 8 —
   are never warmed and are first-touched inside the scored window. Import their function, keep
   our VERIFY-CONCAT warm, complete the set.
3. **Close the E27 reconciliation.** E27 observed −0.3321 % while the M=9 half alone prices
   +1.3625 %, leaving **−1.5511 % unexplained**; E27's M=5 half must add +2.516 % of scored QMV
   cost while E46's refit says ~19 % faster — a 26–41 point contradiction. thorfinn's P4
   (`e27_full`) is the direct replay. The `e27_replica` leg **never ran** — the crossrow-versus-
   wide-5 family question is still open.
4. **Close or kill the NA=5 bandwidth objection** with achieved GB/s per group at the winning
   cells. Two recorded objections; only one is refuted.
5. **Target-verification batching at wider row counts**, now that the QMV/SDPA split is
   understood: the projections already ride one dispatch at full width, so the remaining wide-
   round cost is the second SDPA call plus state management.
6. **Gated DeltaNet recurrence, snapshots, replay and rollback** — 48 of 64 layers, and
   entirely untouched by this round's work. The scan is sequential in T with T-independent
   per-row arithmetic, which is why it was cleared as the width-wall drift source; that same
   structure makes its cost model predictable and its snapshot policy worth attacking.
7. **Proposal-head speed and quality**, including compact or quantized head representations.
   `mtp-head/` is outside the source byte budget with its own 2 GiB cap and is the largest
   entirely unexplored lever on the editable surface.
8. **Vocabulary readout** — 248,320 rows, and the 2-bit coarse + 4-bit rerank default at
   `Qwen35.swift:3178-3207` is already ours. The static-review quantization clause is
   **diff-only against the trusted ancestor**, so it blocks new edits, not the shipped default.

### Standing hazards, not directions

- 🔴 **Never sync the organizer frontier wholesale.** It re-introduces the EOS truncation that
  caps local windows at 302 tokens. Continuation has been added four times and lost three, every
  loss driven by a merge rather than by a decision. Cherry-pick named mechanisms only.
- Max scored verify width is **9** (1 primary + 8 drafts). M=10 bitwise deltas are a
  pre-existing property of the `qmm` splitk 9→10 padding path. **Any delta at M ≤ 9 is a hard
  stop.**
- The runtime-effective Metal source is the JIT string in
  `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`. `mlx.metallib` is never consulted
  for the quantized family and `mlx-generated/metal/quantized.h` is compiled by nothing. Always
  run `python3 research/twin_audit.py`. The scored path compiles with **safe math**
  (`setFastMathEnabled(false)`), so written accumulation trees survive.
- An instrument that cannot fail is not an instrument. Over-aggressive canonicalisers fail
  **toward the null**. Every gate needs a positive control.
- The advisor GitHub REST credential is **flapping**, not down. When it 403s, write guidance to
  `ADVISOR_NOTICES_TO_LIVE_PRS.md` and push — a proven channel — then delete that file once the
  same guidance lands as real PR comments.
