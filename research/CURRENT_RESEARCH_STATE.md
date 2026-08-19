# SENPAI Research State

- **2026-08-19 17:34 UTC**
- Track `qwen3.8-27b-mtp-v1`; advisor branch `senpai/qwen38-mtp-r1`;
  `BASE_SHA = e64961658db3593445d074b5fb3bcbcf0a033c2d`;
  `UPSTREAM_SHA = 0c90733d383f6b987a29682bf9eb9458a6172bfa` — the value actually synced into
  this tree. The organizer tip is `9e1ff9ec7152a04b753f2efb91c3e559909ea4b9` and the trusted
  delta between the two is **empty**, so a future sync has no contract work to do, only
  editable cherry-picks (ledger 179(G)).
- Most recent human research direction: issue #31 asked for a maintenance checkpoint, which is
  complete and closed. **Issue #22 — execute aggressively toward the winning frontier — is the
  standing directive.**

This is a **living document**. The per-experiment record of authority is
[`../senpai/campaign-ledger.md`](../senpai/campaign-ledger.md); durable measurements,
source-line citations and closed questions live in
[`ESTABLISHED_FACTS.md`](ESTABLISHED_FACTS.md). The 2955-line predecessor of this file is
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
| **ranked instrument jitter, per prompt per leg** | **0.2257 %** — measured, n=408, serial leg |
| **rel sd of the published median of 8** | **0.1415 %** point, **0.2636 %** worst case |
| **ranked MDE at 2 sd on the median** | **+0.283 %** point, **+0.527 %** worst case |
| local end-to-end null floor | **0.0629 %** — askeladd's E48 `base2` arm |
| ranked score leverage `psi_mtp` | **0.693391** [0.692292, 0.694490] |
| the leader's last step | **+0.0173 %** |

The entire 3.24986 promotion is 70 inserted lines, zero deletions, one file — an untimed
warm-up. Our deficit is 31× that step.

Confirmed causal boundary, re-verified (`senpai/verify-ranked-score-boundary.sh` PASSES):
candidate-editable code cannot move the ranked serial numerator, so any compliant edit that
lowers candidate MTP seconds per token improves every affected `raw_p`. Never subtract a
locally measured serial share such as `psi_serial` when pricing official value.

### ✅ RESOLVED: the ranked noise model, measured from the pinned serial leg

`research/board_noise_identification.py` settles ledger item 180(H). Both floors the ledger has
been quoting are wrong.

**The identification.** The ranked serial leg is a prebuilt, pinned binary in the runner-owned
baseline workspace, and no candidate edit can move it. So the spread of `serial_spt` across all
408 content-unique board submissions is **pure instrument noise**, with one independent draw per
submission per prompt.

| measurement | value |
|---|---|
| per-prompt rel sd of the serial leg | **0.2257 %** (n=408, all 8 prompts agree to 0.20–0.24 %) |
| distinct serial values per prompt | 407–408 of 408 — effective n is the full sample |
| within-day rel sd | 0.196–0.241 % |
| between-day rel sd | 0.016–0.043 % |
| upper bound on the candidate leg | 0.5505 % (tightest behaviour class, still holds content) |

The within-day term is **10× the between-day term**, so this is iid per-measurement jitter and
not slow thermal drift. The runner's thermal gate works; the instrument still jitters.

**Consequences.** Taking the candidate leg's own jitter at the same relative scale and the two
legs as independent (`corr(serial_spt, mtp_spt) ≈ +0.05` within a behaviour class, so no
common-mode cancellation):

| quantity | point estimate | worst case |
|---|---|---|
| per-prompt `raw_ratio` rel sd | 0.3193 % | 0.5950 % |
| rel sd of the published median of 8 | **0.1415 %** | **0.2636 %** |
| detectable at 2 sd | **+0.283 %** | **+0.527 %** |
| our 0.534 % deficit, in sd | 3.77 | 2.03 |
| P(a redraw of an unchanged tree promotes) | 8.0e-5 | 2.1e-2 |

- Item 148's **0.0693 %** is roughly **3× below** the measured per-leg jitter. It cannot be the
  instrument's floor; it was a lucky pair or a mis-specified comparison.
- Items 166/172's **0.7678 %** is **content granularity**, not noise: `mtp_spt` varies 11.2 %
  and `raw_ratio` 9.9 % across submissions, which is real tree difference.
- **Resubmitting an unchanged tree cannot close the deficit even in the worst case.** The
  "submit more often to win the lottery" reading is dead.
- **This re-prices every banked mechanism.** In median-sd units, using the point estimate: the
  M=9 prize at +1.36 % = **9.6 sd**; E44's ceiling at +0.7437 % = **5.3 sd**; the latch valve at
  ≈+0.5 % = **3.5 sd**; the SDPA chunk at ≈+0.1 % = **0.7 sd**, compose-only.
- ❌ **E29's "4.35 % removable host cost = 31 sd" is RETRACTED** (ledger 181(C)). It was arm 2's
  host-tail *share* of four ladder arms, and ledger item 70 had already ruled the tail an
  accounting artefact: it moved 12× while the round total moved 0.5 %.
- The frontier's own +0.0173 % step is **0.12 sd**. It is not a detectable improvement on this
  instrument; fkiene was promoted by a draw, a real sub-noise gain, or both. **Never quote that
  number as the size of the mechanism it shipped** — the derived "0.02 % per pipeline miss" unit
  was circular and is retracted too.

### 🔴🔴🔴 RESOLVED: what our deficit actually is

`research/order_statistic_targeting.py` plus the 712-tree corpus (ledger 181) settle this.

**Our accept trajectory is already the frontier's.** Our best submission `ca9251b8` and the
promoted frontier report a **byte-identical `mean_draft_len` 8-tuple** and the same head digest.
So do all six trees at the top of the board. Not one diff at the top of the board is a
scheduling or head change. Our `costModelDepth`, `h = 0.18` and every width constant are already
byte-identical to the frontier.

**The whole deficit is candidate-leg execution overhead, and it scales with draft depth:**

| prompt | our `mtp_spt` excess | `mean_draft_len` | sets the median? |
|---|---|---|---|
| `essays` | +0.814 % | 5.43 | no (rank 6) |
| `republic` | +0.594 % | 5.27 | no (rank 7) |
| `botany` | +0.498 % | 5.78 | no (rank 8) |
| **`beagle`** | **+0.454 %** | 4.53 | **YES (rank 4)** |
| **`medicine`** | **+0.291 %** | 4.77 | **YES (rank 5)** |
| `travel` | +0.080 % | 2.66 | no (rank 3) |
| `plutarch` | +0.020 % | 0.15 | no (rank 1) |

Median `mtp_spt` 11.7713 vs 11.7277 ms = **+0.372 %**, slower on **8 of 8** prompts. **This
points at per-round and per-shape dispatch/host overhead, not per-token arithmetic.** Reject any
hypothesis that requires different tokens, different depths, or different arithmetic.

⚠️ **`mean_draft_len` and `1 / leg_time` are collinear on this pool** — deep drafting is what
makes a leg fast — so board data alone cannot separate "cost per wide round" from "one-off cost
per distinct shape visited". Only a local experiment separates them.

### 🔴🔴🔴 Only two prompts convert speed into score, and `beagle` is 98.5 % of the board

`published == median(raw_ratio)` reconstructs to <3.2e-11 for every tree; the median of 8 is the
mean of the 4th and 5th order statistics. Across all 408 scored submissions, central-pair
membership is `beagle` **402 (98.5 %)**, `medicine` 201, `republic` 131, `botany` 67, `essays` 9,
`travel` 4, `plutarch` 2, **`drama` 0**. The pair is `beagle+medicine` for both the frontier and
our best submission.

The pool is bimodal: `drama`, `plutarch`, `travel` cap at ~2.0–2.3 while the other five reach
3.15–3.55, so the three low prompts hold ranks 1–3 in nearly every tree and **can never enter the
median**. `beagle` is the weakest of the five high prompts, which is why it is rank 4 almost
always.

- **`beagle` carries 79 % of our deficit.** `beagle` −0.85 % and `medicine` −0.24 % average to
  −0.545 %, reproducing the observed 0.5367 %.
- A gain confined to `drama`/`plutarch`/`travel` is worth **exactly zero**. This re-derives from a
  second direction why item 146's latch valve simulated at 0.00 %.
- A gain confined to `essays`/`botany` is worth **≈zero** (ranks 6 and 8).
- A `medicine`-only gain saturates after **0.64 %**, when `essays` takes rank 5.

**This is scoring geometry, not permission to specialise.** `program.md` forbids hidden-prompt
specialisation and runtime prompt detection. The legitimate use is to break ties: prefer general
mechanisms that help the deep-drafting mid-speed regime `beagle` and `medicine` occupy.

---

## Current research focus and themes

### 1. The QMV dispatch table is the main attack surface, and its cells are provably live

`quantized.h` dispatches `qmv_fast_crossrow_affine4_g64_m<T,M,IPG,true>` on verify width M.
The design rule is `IPG = ceil(M / ceil(M/4))`; the binding constraint is
`static_assert(NA >= 2 && NA <= 4)` at `:980`, accumulator `typedef vec<float, NA> VF`. Raising
a cell to NA=5 is the mechanism under test.

**Settled from source (ledger 179(D)):** a width-M verify dispatches QMV **once at the full M**
for every M in 1…9. Only the SDPA chunks. So `case 7`, `case 8` and `case 9` are live scored
cells.

Measured: thorfinn's `<T,9,5>` isolated cell win is **−12.255 %**. The register-tax objection
is **refuted**. The **bandwidth** objection from PR #8 is **still open**: one NA=5 group
sustained 95.5 GB/s against 165.6 for NA ≤ 4, while M=9 already runs at 88 % of peak.

🔴 `vec<float,5>` remains an unresolved hard gate: no local `vec` alias exists in the kernel
headers, so it resolves to `metal::vec`, which MSL specifies only for N ∈ {2,3,4}. Every NA=5
experiment must report `sizeof(VF)`, `alignof(VF)`, per-lane correctness, and a positive
control that fails on lane perturbation.

### 2. 🔴 THE CENTRAL OPEN DISAGREEMENT — the narrow-width cost split

askeladd (E48) and edward (E53) roughly **swap** the split and **agree** on its total:

| share of candidate-leg QMV cost | askeladd E48 | edward E53 |
|---|---|---|
| M ∈ {4,5,6} | 64.025 % | 65.0–68.9 % — agree |
| M ∈ {7,8} | **9.391 %** | **21.2–25.1 %** |
| M = 9 | **21.630 %** | **4.6–8.9 %** |
| {7,8} + 9 | 31.02 % | 25.8–34.0 % — agree |

Neither is a GPU measurement. **This decides which mechanism we ship**: under edward's mixture
the ranking *inverts*. **PR #57 settles it by direct measurement** — the hypotheses are 2.4–4.7×
apart and all three predictions are ≥ 6× the 0.0629 % null floor.

### 3. 🔴 The SDPA chunk predicate is WIDER than the constraint that motivated it

Full source proof in [`SDPA_ROUTE_MAP.md`](SDPA_ROUTE_MAP.md). Three routes exist, and the
trusted host dispatcher `scaled_dot_product_attention.cpp` is **not** in `editablePaths`:

| condition | route | threads / threadgroup |
|---|---|---|
| `qL >= 9` | `steel_attention` full attention | — |
| `qL <= 8`, `kL < 1024` | `sdpa_vector`, one pass | fixed **1024**, no `qL` term |
| `qL <= 8`, `kL >= 1024`, arch `'d'`/`'s'` | `sdpa_vector_2pass` | `32 * gqa * qL` |

`qL * gqa <= 32` governs **only** the two-pass route, and that route needs `kL >= 1024`. Our
window is `kL = 512 + tokensCommitted + M`, so `kL >= 1024` arrives only in the **final round or
two**. For essentially the whole scored window, widths 6, 7 and 8 are legal single calls and the
`qL >= 6` chunk splits them for no kernel-family reason — paying two query copies, one extra
SDPA dispatch and one `concatenated`, per full-attention layer, 16 layers deep.

The chunk is still load-bearing at `qL = 9` (steel avoidance) and at `kL >= 1024`
(`utils.h:84-96` **throws** when the thread cap is violated). Correct predicate:
`qL >= 6 && (qL >= 9 || kL >= 1024)`. Bit-exactness of the narrowed form is provable from
`sdpa_vector.h:15-176`: no reduction crosses query rows, the causal predicate is bottom-right
aligned, and chunked and unchunked see identical contributing keys in identical per-thread order.

**Second-order prize, larger than the first.** Deleting this surcharge removes the width 5→6
cost step that E56 (#59) is currently trying to price around, so `costModelDepth` can buy the
sixth row at its true marginal cost — on a pool whose own source comment says it **rewards
depth** (`:723`).

### 4. 🔴🔴🔴 WARM COVERAGE IS THE TOP PRIORITY — it is the only content difference between us and the frontier that can move time

`git diff HEAD upstream/main -- Sources/MLXFastModel/Qwen36MTPBlockSession.swift` resolves to
three content groups. Two are things we removed on purpose: EOS truncation and
`reachedStopToken` (E26; our tests forbid its return) and a stderr trace variant. **The third is
the entire remainder: `warmTargetLaterWindowSDPA`, 70 inserted lines, zero deletions.**

It host-extends throwaway full-attention K/V so `kL == 1024` exactly, fires SDPA at
`qL ∈ {1, 5, 4}`, and discards the outputs. We hold the 512-zero seed warm it builds on
(`:463-475`, identical to theirs); we do **not** hold this. Our warm tops out at
`kL ≈ 512 + width`, so we never create a `kL >= 1024` pipeline before timing. `program.md` puts
seed processing **and** decode in the same timed leg, so any pipeline first-touched in that window
is a real timed cost.

Its board delta is noise (0.12 median sd) but its **leg** evidence is sound: the frontier's
candidate leg improved on **7 of 8** prompts against its parent, median `mtp_spt` −0.102 %.

**A second warm gap nobody has connected (ledger 181(G)).** paul-hf's `0a45fedd` warms the head
flush width over `S = 1 … maxDepth+1` against a 512-row-populated head cache. **We prime the head
at M=512 and fold at M=2** (`:397-406`), so every head QMV width from 3 to 9 is cold at first
use. Absent from our tree **and from the frontier's**.

This is the hypothesis that resolves the collinearity above **without** invoking a per-round
cost: a one-off JIT miss whose *count* equals the number of distinct widths the window visits. On
`essays` (`mean_draft_len` 5.43) widths 3–9 all occur — up to seven misses. On `plutarch` (0.15)
about zero. A one-off cost that is nonetheless ordered by draft depth, at the right total
magnitude (~23 ms on a ~6.2 s leg).

**Corrections to my own route map (ledger 181(E)), both verified in the trusted dispatcher:**

- ❌ **`qL` is NOT in the pipeline identity.** `kname` is the route name plus dtype and the two
  head dims (`:340-348`, `:429-437`); `hash_name` appends only mask mode, `_qt`/`_qnt`, `_c`/`_nc`,
  `_sinks` (`:375-378`). So warming `qL = 4` already creates the pipeline `qL = 2` and `qL = 3`
  use. My "missing `qL = 2` and `qL = 3`" claim was wrong and would have bought a null experiment.
  `qL` reaches identity only via `:746` `do_causal = do_causal_ && q.shape(2) > 1` (so `qL == 1`
  forces `_nc` regardless of the mask argument) and via `blocks`. `{1, 4}` covers both pipelines.
- ✅ **The `blocks` gap is real.** `blocks` is function constant 26 and is appended to
  `hash_name`. On `devc == 's'`, `:446-458` gives 64 promoted to 128 when
  `N > 1024 && n_simds > 4`; `n_simds = 6·qL >= 6` always, so the condition is just `kL > 1024`.
  Padding to exactly 1024 warms 64 and misses 128, while the live window visits both. **One extra
  throwaway dispatch at `kL >= 1025` makes our warm a strictly additive superset of the promoted
  frontier's.** Null if the ranked host is `devc == 'd'` — the arch letter must be reported first,
  which is already Rung 0 of E57.
- ❌ The derived unit "a pipeline miss is worth roughly 0.02 %" was computed from the frontier's
  +0.0173 % board step, which is 0.12 median sd. Circular; retracted. This class is **no longer**
  assumed to sit below the local null floor.

### 5. Gated DeltaNet: the scan is NOT the cost, and the published fix does not apply

Verified geometry: `Hv=48, Hk=16, Dk=Dv=128`; SSM state 3 MiB fp32 per GDN layer; one
recurrence launch reads and writes 6.29 MB, so **302 MB per forward across 48 layers**. Against
a 14,413 MB forward that is **2.1 % of bytes**, and ~91 % of GDN bytes are its three quantized
projections, not the scan.

🔴 **State traffic is FIXED in draft width S.** The kernel loads state into registers once
before the t-loop and stores once after; grid `(32,128,48)` and threadgroup `(32,4,1)` are
independent of `T` (`GatedDelta.swift:54-58, 92-95, 162-163`). **This refutes the published
KVBuffer-style "deferred commit removes `O(m·d²)` per-row state traffic" recommendation for our
tree — our kernel already has the property those papers add.** Do not spend a slot on it.

What actually costs, in priority order:

- **A rejecting round pays three state passes** (verify, replay, next verify) where full
  attention pays two cache writes and one integer decrement: **302 MB + 48 dispatches per
  rejecting round**. Rejecting rounds are the **common case on prose** — per-draft accept is
  0.4685/0.4398 on the two prose proxies against 0.8875 on the copy task.
- **At S=2 the mid-state is written unconditionally and discarded on full accept**:
  3.15 MB/layer, **151 MB/round**. M=2 is 15.8 % of rounds on `natural_history`.
- **`q`/`k`/`g`/`beta` are re-read `Dv = 128×` per head per timestep** — the kernel's indexing
  contains `hk_idx`/`hv_idx` but never `dv_idx`. Order 340 MB/forward at S=9 of cache traffic
  for 8 KB of unique data. A `VPT` (values-per-thread) template is bit-identical by
  construction because each output keeps its own `simd_sum` over the same 32 lanes.
- `snapshotRecurrent` costs **zero** — `arrays[0]?[.ellipsis]` hits `ops.cpp:811-813`
  (`if (!has_neg_strides && out_shape == a.shape()) return a;`) and returns the same array. The
  protection is real but the doc comment's stated mechanism is wrong. Nobody should "optimize"
  this.

**The recurrence kernel's absolute cost has never been measured, and the microbenchmark already
exists**: `sweepGatedDelta` over widths 1…12 with `traffic_bytes` and `flops` at
`Tests/MLXFastTests/QwenQMVCostCurveTests.swift:898-966`, skipped whenever
`MLXFAST_QMV_COST_CURVE_SHAPES_ONLY=1`. That is the cheapest gate in the campaign right now.

Caution: E20 measures the forward at 14,413 MB / 197.45 ms ≈ **73 GB/s effective**, so the
forward is *not* bandwidth-bound and byte counts must not be priced at that average rate.

### 6. The proposal head is NOT unexplored, and its open lever is the shortlist

Correcting this file's previous claim. A non-organizer head **is already declared and in use**
(`mtp-head.manifest.json`, remote `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827`,
427,742,600 bytes). Replacing head **weights** is closed by measurement: two scored
submissions did it and both were rejected (`4437d06` at 2.86127, `9197ed6` at 3.06938).

What survives is head **runtime**:

- **Shortlist containment is the cheapest untried lever.** The proposal is `argmax over exact
  affine-4 logits restricted to the coarse affine-2 top-32` (`Qwen35.swift:3155-3216`). Nobody
  has measured `P(exact argmax ∈ coarse top-32)`. If it is below ~98 %, raising `K` to 64 costs
  ~82 KB of gather (about 0.05 % of the readout's 157 MB) and buys acceptance. The bitmask
  `static_assert`s at `:2506-2508` and `:2594-2596` already admit K=64. **Zero-GPU to falsify.**
- **A flat vocabulary crop is already dead**: halving the compact prefix to 49,152 regressed
  acceptance 1.00 → 0.877 (`Qwen35.swift:2757-2768`). The published FR-Spec / VocabTrim lever is
  therefore already partly harvested by our 98,336-row compact readout; only a *hierarchical*
  shortlist generator remains, and it must be declared and digest-pinned, never derived at load
  time.
- `h = 0.18` decomposes as **~22 % head step, ~78 % extra verify row** (E1: isolated head step
  2.590 ms against a 65.009 ms depth-0 round; 84.4 % of the depth-8 marginal is verify width).
  Measured per-depth `h` is `[0.084, 0.078, 0.243, 0.375, 0.292, 0.300, 0.287, 0.391]` —
  over-priced at d ≤ 1, under-priced at d ≥ 2. E56 (#59) is testing exactly this.
- **Our head does not collapse with depth.** Published vanilla MTP heads reused recursively go
  70 % → 10 % → ~0 % at k=1/2/3; our pooled tape is 0.693 / 0.584 / 0.508 / 0.419 — monotone,
  no cliff. The shipped prior `0.85 * 0.98^i` is the wrong shape in both directions, but the
  head itself is not the depth blocker.
- A free A/B already exists for the head's bf16 precision islands:
  `MLXFAST_QWEN_MTP_EXACT_QKV_ROWS` at `Qwen35.swift:2882`. Its dose has never been swept.

### 7. Measurement discipline that now gates every claim

- 🔴 **A green `--local-iterate` parity line is NOT exactness evidence.** E51 measured it: an
  arm reporting `all_tokens_matched=true`, `residual_divergence_count=0`,
  `public_drift_tripwire_passed=true` had moved declared top-two row evidence at **52 of 64
  positions** with two top-2 identity flips. Every brief touching precision, reduction order,
  packing, recurrence, cache layout or replay must gate on **declared per-position row
  evidence** with a positive control, and must state that the local parity line was not the gate.
- The **0.0629 % end-to-end null floor** (E48 `base2`) is the unit for every effect claim.
- Board intervals are **identification intervals, not standard errors** — 152 content-distinct
  trees reproduce the published telemetry to 16 digits, so the board gives ~1 observation.
- `psi_serial` is **NOT IDENTIFIABLE** locally (four treatments imply 0.7966/0.8694/1.0414/
  1.2470, two exceeding 1.0). Unidentified, not refuted.
- Ungated timing (`MLXFAST_LOCAL_COOL_GATE=0`) only ABBA-counterbalanced, entry/exit
  temperatures recorded, `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
  preserved verbatim.
- Log W&B **per leg while timing**, never at session end.

---

## Live experiment slots

| PR | student | experiment | state |
|---|---|---|---|
| #57 | askeladd | E55 — compose `<T,9,5>` onto the real shipped table to a **submittable candidate**; settles theme 2 | `status:wip` r1 |
| #58 | thorfinn | E54 — lone-versus-sibling NA=5 law across M=5/7/8/9; I predicted **Law C** on the record | `status:wip` r1 |
| #59 | edward | E56 — draft-depth schedule staircases; theme 3 changed what he should conclude | `status:wip` r1 |
| #60 | alphonse | E57 — narrow the SDPA chunk predicate (theme 3) | `status:wip` r1 — **brief needs a correction: `qL` is NOT in the SDPA pipeline identity** |

Merged: **#55** (alphonse E51 — refuted at rung one, instruments retained), **#53** (thorfinn
E49), **#52** (askeladd E48), **#56** (edward E53). Base advanced
`981e69a` → `1247c57f` → `a35bb006` → `a2c3dbc4` → `67b8547` → `daa1d018` → `e6496165`, each
with base-change inertness verified rather than assumed.

---

## Potential next research directions

Ordered by expected value, not by convenience.

1. **Submit whatever #57 produces, if it survives exactness.** A 0.39–1.84 % MTP-leg win on a
   0.534 % deficit is a submission, not a screen. Official evaluation is part of the research
   loop.
2. **🔴 WARM-COVERAGE COMPLETION — the highest-value next assignment.** One hypothesis:
   *incomplete pipeline warm coverage costs timed-window JIT misses, and the cost scales with
   the number of distinct shapes the scored window visits.* This is the only mechanism found so
   far whose predicted signature matches all three independent observations — the frozen accept
   trajectory, the depth-ordered deficit, and the frontier's one 70-line untimed edit. Three
   instances, measured separately **and** together so attribution survives:
   - (a) hand-apply the frontier's `warmTargetLaterWindowSDPA`, reduced to `qL ∈ {1,4}` (the
     `qL=2`/`qL=3` gap does not exist — see below), and **extended with a second dispatch pair
     at `kL >= 1025`** so the `blocks=128` variant is also warmed. The frontier pads to exactly
     `kL == 1024` and therefore warms only `blocks=64`, while the live window
     `kL = 512 + tokensCommitted + M` visits both. This makes our warm a strictly additive
     superset of the promoted frontier's at zero fidelity risk.
   - (b) paul-hf's flush-width head warm (`0a45fedd`, 3.24001): replace the single 2-row head
     "fold" warm with a loop over `S = 1 … maxDepth+1`, each run against a 512-row-populated
     head cache and then trimmed back to 512, so round-2's head QMV families `M = 3…9` for
     `fc`, `kv()` and the island overlay compile **outside** the timed window. We prime the head
     at `M=512` and fold at `M=2` (`Qwen36MTPBlockSession.swift:397-406`), so every head QMV
     width 3–9 is cold at first use. Absent from our tree **and** from the frontier's.
   - (c) **keep VERIFY-CONCAT** (`:359-396`). The board provides no evidence against it: the one
     submission that deleted it (ofou `0cd0a6b4`) branched from a pre-fkiene commit and Yukon
     replaced the whole file, so the deletion was incidental, and that submission's candidate
     leg improved on only 5/8 prompts.

   All three are untimed, zero-arithmetic, discard-output. Instance (b) resolves the
   depth-collinearity **without** a per-round cost: a one-off JIT miss whose *count* equals the
   number of distinct widths visited. `essays` (mean draft length 5.43) visits widths 3–9, up to
   7 misses; `plutarch` (0.15) visits ≈0. That is a one-off cost ordered by draft depth with the
   right total magnitude. Requirements for the assignment: report the `d.get_architecture()`
   arch letter first (the `blocks` axis is null on `devc=='d'`); compare **matched absolute
   candidate seconds/token against a fresh unchanged base**, not the local ratio; ABBA
   counterbalance with entry/exit temperatures; hand-apply every hunk (never copy a rival file —
   see the `reachedStopToken` trap).

   ~~E29 measured a genuinely removable host cost of 4.35 % of decode.~~ **RETRACTED.** The
   4.35 % is arm 2's host-tail *share* of four E29 ladder arms, and ledger item 70 already ruled
   that share an accounting artefact: the shares moved 12× (53.86 / 4.35 / 5.66 / 35.86 %) while
   the round totals moved 0.5 % (6028.7 / 6022.2 / 6015.8 / 5998.3 ms). E31 used the same number
   correctly as a *ceiling* on all host-side work, not as a recovery. What survives is only that
   `mx.compile` of the head step is untouched by the entire 712-tree rival field; its ceiling is
   that same 4.35 % and its prior is ≈0, so it is **speculative, not first**. It also has a
   structural blocker: `CompiledDecode` requires every layer to be a compilable fixed-shape
   cache and explicitly excludes SSM, and our target has 48 Gated DeltaNet layers, so only the
   single-layer head is eligible (`makeMTPCache()` returns exactly one `KVCacheSimple`).
3. ~~Resolve the noise-model inconsistency.~~ **DONE, zero GPU** —
   `research/board_noise_identification.py`, see "Where we stand". The ranked instrument jitters
   at 0.2257 % per prompt per leg, the published median at 0.1415 %, and the answer is that
   mechanism size dominates: a promoting candidate needs a real **+0.53 % or better**. The one
   remaining sub-question worth an hour is whether the candidate leg's own jitter is really the
   same relative scale as the serial leg's; the dataset bounds it above at 0.5505 % but cannot
   identify it from below, because `mean_draft_len` identity does not imply identical candidate
   work.
4. **Land the latch release valve.** `positionAcceptEMA[0] <= 0.18` is an absorbing state:
   it is written only inside `recordAcceptOutcome`, whose single call site is unreachable at
   depth 0. Simulated at **−14.55 % to −18.02 %** when it hits a bankable prompt, observed at
   3/94 ≈ 3.2 % of runs, so it is roughly **+0.5 % expected score per submission** of tail
   insurance for a policy edit with zero exactness risk. Item 146 said "bundle it"; nobody did.
   It is an unpriced liability on every remaining slot. Never worth a dedicated slot; always
   worth composing.
5. **Price GDN rollback economics** (theme 5). Run the existing `sweepGatedDelta` first — it is
   nearly free and it gates everything else in that theme. Then split `rollbackRoundCount` by
   `draftCount` to get the per-width reject rate, which decides whether deleting the S=2 eager
   mid-state wins (break-even is M=2 reject probability ≈ 0.49).
6. **Audit the pre-GDN depthwise conv under multi-row verification.** Published work documents a
   silent temporal misalignment where a masked depthwise convolution over stacked candidate
   tokens captures `{t1,t1}` instead of `{t0,t1}` unless the mask is applied. This is a
   correctness question, not a speed question, and it is cheap.
7. **`MLX_MAX_OPS_PER_BUFFER` 50 → 100 — the second assignment to fire.**
   `RuntimeStartupMemoryPolicy.swift:76`. igneous-prose's `646a3dee` is a 2-line change and had
   the **fastest candidate leg of ranks 13–18** (−0.037 % `mtp_spt` versus its parent, second
   only to the rank-1 tip). Structural key finding:
   `installQwenMTPFullProfileCommandBufferDefaults` force-sets the variable with `overwrite=1`
   from `resolve()` before MLX's first device access, while the trusted worker's own `setenv`
   pair sits behind `guard policy.isLowMemory else { return }`
   (`Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift:487`,
   `Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift:498`). On the ranked 128 GiB box the struct
   constant at `:150` (`maxOperationsPerCommandBuffer: 50`) is therefore **dead** and line 76 is
   the only writer. This is exactly the unswept `[1, floor]` corner E31 named (and predicted a
   0.4 % slowdown for) and which was never run. Sweep `{50, 100, 128, 256}` at a fixed 512 MiB,
   ABBA. It does **not** reopen ledger item 69: that closed the `asyncEval` ladder, which adds
   commits *above* MLX's floor, whereas this moves the floor itself.

   ~~Census the high-scoring NON-promoted rival trees.~~ **DONE, zero GPU** —
   `research/rival_tree_census.py` and `research/corpus_surface_map.py` over all 712 readable
   `upstream/submissions/*` refs. See ledger 181(F)–181(J). Headlines: churn does not predict
   score (Pearson −0.075); the field plays on 27 of 154 editable files and 127 are untouched;
   ranks 1–6 are one promotion chain in which #3, #4 and #6 are regressions against their own
   parents, so **board score is not a valid price for a rival mechanism — `mtp_spt` is**. Both
   named trees were inspected and both are now **do-not-slot**: paul-hf's real content is the
   dead-KV-GEMM elision plus the flush-width head warm (now direction 2b), and Lieisyourlie's
   baked bf16 GDN scale immediates measured −0.0003 and would license FMA contraction on a path
   we already memoise (`Qwen35.swift:743-745`).
8. **Shortlist-containment audit** (theme 6) — zero-GPU falsification of the cheapest remaining
   head lever.
9. **Close the E27 reconciliation.** −1.5511 % remains unexplained; the `e27_replica` leg never
   ran, so the crossrow-versus-wide-5 family question is open. thorfinn's P4 is the direct
   replay. Weaker than it looks: #57 settles the mixture question that gives E27 its relevance.
10. **Close or kill the NA=5 bandwidth objection** with achieved GB/s per group at the winning
    cells. Two recorded objections; only one is refuted.

### Compose-only — worth carrying on any slot, never worth a slot of its own

Each item below was inspected in the 712-tree rival census. Each is small, has a bounded
mechanism, and is priced by `mtp_spt` rather than by board score.

- **Dead-KV-GEMM elision** (paul-hf, `Qwen35.swift:1759-1766`, `:1812`). Provably bit-exact by
  inspection: the `putAlong` index vector is a bijection of `0..<kOut+vOut`, so every base column
  is overwritten and the operation is a scatter onto `zeros`. Confined to the head, so the local
  ratio **is** valid evidence here. ~0.04 %. Precondition: dump the
  `mtp.precision_islands.k.indices` / `v.indices` shapes and ranges and confirm full cover
  (4 KV heads × 256 head dim = 1024 per side, 2048 total).
- **`pendingPrimaryDevice`** (fkiene, `7f777a36`). `Self.devicePrimaryToken(top2IDs, row:)` is a
  pure slice `top2IDs[row..<row+1, 0..<1]`, which makes the verify concat all-device. Integer
  only, identical by construction. 🔴 The same code and comments appear byte-identically under a
  second username (`055bc201`, jonathan308), so shared lineage is likely — count the two refs as
  **one** piece of evidence.
- **Fused last-merge plus final RMSNorm** (`bc0b2fea`). We already ship
  `qwen35FusedResidualRMSNorm` for interior pairs (`:2055`, `:2086`, `:2102`), but
  `Qwen35.swift:2228` is still `hiddenStates = delta.map { base + $0 } ?? base` followed by a
  separate `model.norm(hidden)`. ~15 lines. It runs on **both** legs, so screen it on matched
  absolute time, not the local ratio. Verify `n_reads=4` / `lsize=1024` matches `rms_looped` at
  axis 5120 (`RMS_LOOPED_LIMIT = 4096`, `defines.h:16`, dispatch `normalization.cpp:57-60`).
- **Top-32 finalize k-way merge** (DawgZter, `d909492b`). Zero floating-point arithmetic; pure
  `uint (ord,idx)` selection; 256 → 32 threads; no threadgroup memory and no barriers. It lands
  on `Qwen35.swift:2492-2660`, our leading suspect for the 91-versus-89 row nondeterminism, so
  bundle it with a ten-draw byte-identity replay. Our constants `Tiles=64, K=32, TG=256` satisfy
  their `static_assert(LISTS == 2*SIMD_SIZE)`.
- **The latch release valve** (item 146). Take WillGasser's `latchProbeInterval = 4` as given.

**Demoted out of this list:** Lieisyourlie's `hidden`-deferral (`5ad14a0b`, dropping `hidden`
from our `:513-514` eval list). Its one ranked measurement is negative on **both** score-setting
prompts (beagle +0.049 %, plutarch +0.384 %) and it is a genuine zero-sum deferral, not a
removal.

### Explicitly closed — do not spend a slot

- **The entire lossless-verification-theory family is a no-op at greedy.** Block Verification,
  Traversal Verification, UniVer, hierarchical SD, multi-draft canonical decomposition and
  relatives all recover residual probability mass under *sampling*. At T=0 the target's argmax
  is deterministic and the optimal rule is already "longest matching prefix".
- **Draft trees.** Contraindicated by two independent measured lines, and our schedule is a
  chain by design.
- **KVBuffer-style deferred recurrent-state commit** as a per-row traffic win — our scan already
  has the property (theme 5).
- **Replacement head weights** — two scored rejections (theme 6).
- **A flat vocabulary crop** — measured dead at 49,152 rows (theme 6).
- **Layer-skipping self-drafting** — published α = 0.038 on sequential GDN hybrids like ours, and
  forbidden by the MTP-only rule anyway.
- **`MLX_METAL_GPU_ARCH` nax-off on the ranked leg.** Now that the ranked serial leg is known to
  be a pinned separate binary (item 176), the spoof is candidate-only and may look tempting.
  It fails the fidelity gate by construction: nax-off changes prefill GEMM rounding, which
  perturbs every downstream hidden state and top-2 value that the parent checks.

### Standing hazards, not directions

- 🔴 **Never sync the organizer frontier wholesale.** It re-introduces the EOS truncation that
  caps local windows at 302 tokens. Continuation has been added four times and lost three, every
  loss driven by a merge rather than by a decision. Cherry-pick named mechanisms only.
- 🔴 **Never copy a rival file. Hand-apply every hunk.** `reachedStopToken` is present in **all
  six** top rival trees — 7 sites in the frontier's `Qwen36MTPBlockSession.swift` (`:69`, `:165`,
  `:904`, `:920`, `:986`, `:1297`, `:1309`) — and is completely absent from our `Sources/`.
  Yukon replaces whole files rather than merging them, so importing a rival *file* would silently
  restore all seven sites and reintroduce the `.notBegun` abort that E26 bisected. This hazard
  applies to every import in the compose-only list and to direction 2.
- Max scored verify width is **9**. M=10 bitwise deltas are a pre-existing property of the `qmm`
  splitk 9→10 padding path. **Any delta at M ≤ 9 is a hard stop.**
- The runtime-effective Metal source for the quantized family is the JIT string in
  `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`; `mlx.metallib` is never consulted
  for it and `mlx-generated/metal/quantized.h` is compiled by nothing. Always run
  `python3 research/twin_audit.py`. The GDN scan and mid-state kernels are `MLXFast.metalKernel`
  JIT strings with neither a generated twin nor metallib involvement. The scored path compiles
  with **safe math** (`setFastMathEnabled(false)`), so written accumulation trees survive.
- An instrument that cannot fail is not an instrument. Over-aggressive canonicalisers fail
  **toward the null**. Every gate needs a positive control.
- 🔴 **Open: the drafting schedule may not be deterministic across identical-source runs.**
  E51's A/A control emitted 91 and 89 rows over the same 64 positions with zero shared positions
  disagreeing, i.e. two runs of one binary proposed different drafts. askeladd's E48 width
  histogram is recorded byte-identical across 10 draws, and that histogram underpins the theme-2
  cost mixture. Both cannot describe the same scheduler. Leading suspect, unverified: the
  two-dispatch exact top-32 draft readout (`Qwen35.swift:2492-2660`) breaking a near-tie
  order-sensitively.
- Dead code on the scored path, for the next cleanup PR: the `nConfirmed > 0 && nConfirmed < S`
  split-chunk branch (`Qwen35.swift:1120-1147`), the masked scan variant
  (`GatedDelta.swift:146-152`), `gatedDeltaStepOps` (`:176+`), and `rollbackState`, which is now
  written and then cleared without ever being read.
