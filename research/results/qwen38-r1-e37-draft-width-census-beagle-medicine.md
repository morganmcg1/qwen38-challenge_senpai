# E37 — Dispatched verify-width (M) census on the beagle and medicine proxies

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_row_share_at_M_ge_6_beagle_proxy","available":true,"value":0.0},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- **Student / branch:** `qwen-askeladd` / `qwen-askeladd/draft-width-census-beagle-medicine` (PR #42, revision `r1`)
- **Hypothesis and target cost:** H1 — the local prose proxies reach the ranked
  `M >= 6` operating cell often enough to profile it locally. Target cost is the
  `M >= 6` verify cell, which the advisor brief's main line ("make M=6 a single
  weight pass") depends on.
- **Decision:** **dead** for H1 as stated — the local proxies cannot reach the
  ranked cell, and no local run ever chooses depth > 5. The census is
  nonetheless *positive* for deliverables 3 and 4, which came out stronger than
  the assignment assumed.
- **`BASE_SHA`:** `abf6d79f92b97e3c47856be9c1d7798e6dc5a6b5` · **candidate
  commit:** this branch head · **`UPSTREAM_SHA`:** unchanged, no organizer sync
  performed.
- **Yukon promoted submission / frontier:** not queried and not needed — nothing
  here is a submission candidate. Our best ranked row remains `ca9251b8`
  (3.23250848263467, rank 9, `rejected: score did not improve current best`).
- **Submitted candidate files:** **none.** The diff touches only `research/`.
- **Supporting tooling / documentation files:** `research/e37-run.sh`,
  `research/e37_width_census.py`, `research/e37_wandb_log.py`,
  `research/results/e37/*`, this report.
- **MTP head provenance and draft policy:** declared proposal head from
  `mtp-head.manifest.json` — tree digest
  `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`,
  427 742 600 bytes, staged and verified by `research/fetch-declared-head.sh`.
  The raw `model.safetensors` sha256 is
  `d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`; the
  manifest value is a *tree* digest (sha256 over `"<file sha256>  <path>\n"`),
  so the two differ by construction and both are recorded. Draft policy is the
  shipped `costModelDepth` at its shipped literals.
- **Assignment-scope preflight:** diff confined to `research/`. No
  `Sources/`, `Vendor/`, `mtp-head.manifest.json`, fixture, workflow or trusted
  file was modified. `git status` was clean before every `run_job` launch.
- **Editable source bytes / headroom / growth:** unchanged — zero candidate
  bytes added, so the 3 000 000 / 524 288 / 262 144 limits are untouched.
- **Scored-path reachability evidence:** established by source inspection and
  confirmed by the live trace; see §1.

---

## 1. What M is, and why the census is a census of *dispatched* width

`chosen_depth -> M` is not an assumption. In `Qwen36MTPBlockSession.swift`:

- A **drafting** round (`d >= 1`) issues exactly one verify call,
  `callWithHiddenAndNormed([primary] + drafts, nConfirmed: 1)` (`:1070`), with
  `declaredRows = draftCount + 1` (`:1260`). So **`M = d + 1`**.
- A **skip** round (`d == 0`) issues a single-row forward, `declaredRows = 1`
  (`:943`), so **`M = 1`**, and it returns before writing its trace line. Those
  rounds are invisible in the trace and are recovered from round-counter gaps,
  exactly as `research/depth_histogram.py` already does. Both prompts had
  `implied_d0 = 0`, so no recovery was needed.
- Widths 6..9 are still **one** dispatch. Only the SDPA inside them is split
  into two `<= 5`-row calls (§3), which does not change `M` for the 369 + 32 + 96
  quantised projections per verify forward (E20 §2.8).

`draftPolicy` is wired to `costModelDepth` at `:194`. The active cap is
`cap = min(min(offeredDepth, maxDepth), widthCap)` with
`widthCap = fullAcceptStreak >= segmentedStreakGate(2) ? segmentedVerifyDepthCap(8) : sdpaWidthWallDepthCap(5)`
(`:706-709`).

Scored geometry was used throughout: offered depth **8**
(`benchmark-qwen-mtp.sh:141` default `MLXFAST_QWEN_MTP_DEPTH:-8`, matching the
ranked workflow's `MLXFAST_QWEN_MTP_DEPTH: "8"` — the usage text saying
"2..4 default 2" is stale), 512 decode tokens, declared head.

---

## 2. Deliverable 1+2 — the census. The local proxies cannot reach the ranked cell.

`natural_history` (the beagle proxy):

| M | rounds | round share | rows | row share | token share |
|---:|---:|---:|---:|---:|---:|
| 2 | 41 | 0.1589 | 82 | 0.0993 | 0.1240 |
| 3 | 138 | 0.5349 | 414 | 0.5012 | 0.5610 |
| 4 | 65 | 0.2519 | 260 | 0.3148 | 0.2598 |
| 5 | 14 | 0.0543 | 70 | 0.0847 | 0.0551 |

`max_M = 5`, `mean_M = 3.2016`, **`M >= 6` share = 0.0000 on all three weightings.**

`medicine`:

| M | rounds | round share | rows | row share | token share |
|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 0.0264 | 12 | 0.0145 | 0.0217 |
| 3 | 107 | 0.4714 | 321 | 0.3886 | 0.4320 |
| 4 | 85 | 0.3744 | 340 | 0.4116 | 0.3886 |
| 5 | 21 | 0.0925 | 105 | 0.1271 | 0.1105 |
| 6 | 8 | 0.0352 | 48 | 0.0581 | 0.0473 |

`max_M = 6`, `mean_M = 3.6388`, **`M >= 6` round share 0.0352, row share 0.0581,
token share 0.0473.**

**This reproduces the E25 BASE arm bit-for-bit.** E25's 2026-08-18 BASE run on
the same host with the same declared head recorded `{d1:41, d2:139, d3:66,
d4:14}` over 260 rounds for `natural_history` and 8 rounds at `d=5` out of 229
for `medicine`. The schedule is byte-identical between E25's base `d7619a7f` and
`abf6d79f` (`headStepCostRatio=0.18`, `sdpaWidthWallDepthCap=5`,
`segmentedVerifyDepthCap=8`, `segmentedStreakGate=2`), so this is a
**deterministic property of the policy on these prompts**, not a draw. The one
+95-line delta in `Qwen36MTPBlockSession.swift` between those bases is the
wired-residency warm block, which is a no-op locally (it guards on
`physicalMemory >= 96 GB`; this host has 48 GiB).

**Correction to the record.** edward's "candidate leg caps at
`effective_max_draft_len = 3`" describes E25's *candidate (modified-policy)*
arm, not the shipped BASE arm. Shipped BASE reaches `d=4` (M=5) on the beagle
proxy and `d=5` (M=6) on medicine.

### 2a. The discriminating result: the gate opens, the cost model declines

"Never saw M > 6" has two very different causes. The round trace prints the live
`cap` and `streak` (`:767`), so this is read, not reconstructed:

| arm | drafting rounds | caps seen | gate-open (`cap=8`) rounds | max streak | chose depth > 5 while open |
|---|---:|---|---:|---:|---:|
| natural_history | 260 | 5, 8 | 16 (6.15 %) | 3 | **0** |
| medicine | 229 | 5, 8 | 23 (10.04 %) | 4 | **0** |

The `segmentedVerifyDepthCap = 8` regime **does** open — 39 rounds across the two
prompts — and in **0 of 489 drafting rounds** does `costModelDepth` choose a
depth above 5. So:

- **M = 7, 8, 9 are never dispatched locally at all.** That whole scored path is
  dead code on this host under these prompts, while ranked `botany` runs at
  `n = 5.776 > 5`, which is impossible without it. **A live scored code path
  exists that no local `--local-iterate` run can exercise, validate, or
  regression-test — including for correctness.**
- The binding constraint locally is the **cost model**, not the streak gate.
  Ranked beagle runs the same policy and the same head at mean chosen depth
  **4.533**; the local proxy runs at **2.204**. The head is identical, so the
  entire gap is the prompt.

**H1 is dead, and it is dead structurally.** The proxies are unfaithful in
exactly the dimension E37 was asked to census, so no amount of local running
reaches the cell. Per the advisor's pre-authorisation, this is a complete
terminal result.

---

## 3. Deliverable 3 — warm-coverage audit

### 3a. The width wall is a hard MLX dispatch cliff, independently re-derived

I derived the mechanism from vendored source before reading the shipped comment;
they agree, which is worth recording because the constant is often read as
tuning. In `Vendor/mlx-swift/.../metal/scaled_dot_product_attention.cpp:621-640`,
for Qwen (head_dim **256**, gqa = 24/4 = **6**):

- `sdpa_full_supported_head_dim` requires head_dim ∈ {64, 80, 128} → **false for
  every width**. The steel path is permanently closed to this model.
- `supports_sdpa_vector` requires `q_len * gqa <= 32` → **`q_len <= 5`**.

At `q_len >= 6` neither branch qualifies, `use_fallback` returns true
(`fast.cpp:828` takes the fused path only when it returns false), and SDPA
decomposes into unfused matmul+softmax. `attentionWithCacheUpdate`
(`Qwen35Attention.swift:200`) therefore splits 6..9-row attention into two
`<= 5`-row calls, preserving the fused kernel — the shipped rationale at
`Qwen36MTPBlockSession.swift:648-661` says exactly this, and this derivation was
reached independently before reading it. The `query_sequence_length > 8`
boundary in `supports_sdpa_full` is never reached, so **the operative cliff is
6, from the gqa product, not 8**.

### 3b. Pipeline identity across the scored width range — one PSO on both sides

This was the question blocking the audit's conclusion. E20 §2.8 settled the
target side and I re-verified it, then extended it to the head:

- `get_qmv_batch_limit` (`quantized.cpp:84`) returns **10** for every scored
  shape on `applegpu_g16s` (arch_gen 16 → `else` branch, arch size `s` →
  `default:` case, all `D` ∈ {2560, 5120, 6144, 10240, 17408} with `D > 4096` or
  `O > 4096`). `eval_gpu:1415-1418` routes `M >= vector_limit` to `qmm`/
  `qmm_splitk`, so **every width `M <= 9` reaches `dispatch_qmv`**. Ranked M5
  (gen ≥ 17) takes the same branch, so this transfers.
- `qmv()`'s kernel name is `<mode>_qmv_fast_<type>_gs_64_b_4_batch_0` — it
  **does not encode M** — and `grid_dims(M, ceil(N/8), B)` makes `ntg.x == M`, so
  the E27 crossrow IPG table's `switch (ntg.x)` lives inside a *single*
  compiled pipeline.
- **New:** the proposal head is itself **fully 4-bit group-64 affine quantised**
  (40 tensors, U32 packed weights + BF16 scales/biases; `fc` 10240→5120,
  `draft_lm_head` 2560→**98336** — a reduced draft vocabulary vs the target's
  248320). Every head linear therefore rides the same `dispatch_qmv`, and head
  flush widths `F = 1..9` are all below the limit-10 boundary too.

### 3c. Result: the coverage holes are shape gaps, not pipeline gaps

`warmAllDepthShapes` (`Qwen36MTPBlockSession.swift:290`) covers the 512-row
target forward; 8× 1-row head `mtpForwardWithHidden`; head-history warm at
`F = 512` (`:334-341`) and `F = 2` (`:359-365`); `draftTokenID`; verify widths
**1…9** with `linearTopTwoRows`; and `replayRecurrentPrefix(T = 2..8)` plus a
width-3 verify for `T = 1`.

The live head flush width is `F = 1 + accepted_prev + (#non-drafting rounds
since)`, and the live call is `mtpHeadLastHiddenWithKVOnlyHistory`
(`Qwen36MTPTarget.swift:85`, invoked at `Qwen36MTPBlockSession.swift:1014`),
which splits into `fc`/RMSNorm/embed over `F` rows plus a KV-history append over
`F-1` rows. Against that, the apparent holes are:

| # | uncovered triple | verdict |
|---|---|---|
| G1 | head `fc`/norm/embed at `F = 3..9` (warm has 1, 2, 512) | **shape gap only** — same `dispatch_qmv` PSO |
| G2 | KV-history append at `F-1 = 2..8` (warm has 1, 511) | **shape gap only** — shape-generic copy kernels |
| G3 | warm uses `callWithHidden`, live verify uses `callWithHiddenAndNormed` | no extra pipeline; the normed output is an elementwise epilogue |
| G4 | generic repair `callWithHidden(1+accepted, nConfirmed: 0)` at widths 2..9 | **shape gap only** — same PSO family |

The first flush is exactly `F = 511 + 1 = 512`, so the seed-priming flush *is*
covered.

**This is a negative result and I am reporting it as one.** No Metal pipeline
compile is missed at any scored width. The residual cost of a first-touch shape
is bounded by one-time allocator work, which amortises to nothing over a
512-token window. **There is no warm-coverage headroom to harvest here**, and I
recommend closing that line rather than dressing the gaps up as opportunity.

Conversely, `warmAllDepthShapes` warms widths **7, 8, 9 that the local run never
dispatches** (§2a). That warm work is not wasted on rank, but it is unvalidated
locally.

---

## 4. Deliverable 4 — the real `M >= 6` mass for beagle, with provenance

Three sources, clearly separated:

| source | beagle `M>=6` | medicine `M>=6` | provenance |
|---|---|---|---|
| **measured, this census** | round 0.0000 / row **0.0000** | round 0.0352 / row **0.0581** | local M4 Pro proxy prompts, declared head, 512 tokens, scored geometry |
| **simulated (E34, edward)** | 0.538 | 0.593 | simulation; `research/e34_cost_model.py` is not in this tree (E34 not merged into `abf6d79`), so it could not be re-run or audited here |
| **exact bracket from ranked telemetry** | round ≥ **0.1333**, row ≥ **0.2167** | round ≥ **0.1920**, row ≥ **0.2996** | published `effective_mean_draft_len`; no simulation, no acceptance model |

The bracket needs no assumptions. `effective_mean_draft_len` is the mean chosen
depth over **all** rounds — plutarch's `0.154` with 449 non-drafting rounds fixes
that convention — and the advisor brief states `M = n + 1` directly. That is two
equality constraints on a distribution over depths {0..8}, so the vertices are
supported on at most two depths and the extrema are exact by enumeration
(`ranked_ge6_bound`, verified by vertex enumeration rather than an analytic
guess):

| prompt | n | mean M | round share `M>=6` | row share `M>=6` | floor witness |
|---|---:|---:|---|---|---|
| beagle | 4.533 | 5.53 | [0.1333, 0.9066] | [0.2167, 0.9831] | d4 .8668 / d8 .1333 |
| medicine | 4.768 | 5.77 | [0.1920, 0.9536] | [0.2996, 0.9920] | d4 .8080 / d8 .1920 |

So **ranked beagle spends at least 21.7 % of its dispatched target rows at
`M >= 6`**, guaranteed. E34's 0.538 sits inside the feasible bracket but is an
unvalidated point estimate; the floor is what should be quoted.

Two calibrations on the upper half of the bracket:

- If the depth `<= 5` ceiling observed locally in **489/489** rounds also held on
  rank, beagle's floor would rise to round 0.5330 / row **0.5780** — close to
  E34's 0.538, which suggests that is the assumption E34 encoded.
- But ranked `botany` runs `n = 5.776 > 5`, which **falsifies a universal
  depth-5 ceiling on rank.** So that scenario is an upper bracket for beagle,
  not a prediction.

### Score payoff frame

Score = mean of the 4th and 5th order statistics = (beagle 3.1433 + medicine
3.3553)/2 = **3.24930**, reproducing the board top; our `ca9251b8` at 3.23251 is
**−0.5168 %**. `∂score/∂raw_p` is 0.5 for beagle and medicine, 0 for the other six.

For a **1 %** speedup of a cell holding candidate-leg *time* share φ, on beagle:

| φ | Δraw_p | Δscore | as % of score | vs σ_score = 0.078 % |
|---:|---:|---:|---:|---:|
| 10 % | +0.00315 | +0.00157 | +0.048 % | 0.6σ |
| 25 % | +0.00788 | +0.00394 | +0.121 % | 1.6σ |
| 50 % | +0.01580 | +0.00790 | +0.243 % | 3.1σ |
| 100 % | +0.03175 | +0.01588 | +0.489 % | 6.3σ |

At the guaranteed row-share floor φ ≈ 0.22, **a 1 % speedup of the `M >= 6` cell
is worth ≈ 1.4σ** — real but marginal; roughly **3 %** is needed to be
comfortably decisive. Row share is used as a proxy for time share, which is only
exact if per-row cost is width-independent; E20/E27 show it is not, so treat this
as an order-of-magnitude frame.

Headroom before the marginal weight goes to zero: beagle **+7.87 %** of raw_p
(score ceiling 3.37300, +3.81 %); medicine only **+1.06 %** (I previously carried
+0.64 % — that figure was wrong and is corrected here from the source table).

---

## Evidence

- **Host, memory profile, toolchain, thermal policy:** Apple **M4 Pro**
  `Mac16,11`, `applegpu_g16s`, 20 GPU cores, 48 GiB
  (`hw.memsize = 51539607552`), macOS 26.5.2, Swift 6.3.3. **Not the ranked M5.**
  `MLXFAST_LOCAL_COOL_GATE=0`; entry/exit GPU temperature recorded per arm
  (`natural_history` 39.2 °C → 63.9 °C).
  **`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `timing_claims_permitted=false`, `trace_perturbs_timing=true`** are preserved
  verbatim in every `meta.txt`.
- **Exact reproduction commands** (every GPU touch via the lock wrapper and
  `run_job`, worktree clean at launch):

  ```bash
  bash research/fetch-declared-head.sh
  research/await-lock-then-run.sh 600 bash research/rebuild.sh
  research/await-lock-then-run.sh 600 bash research/e37-run.sh --golden natural_history medicine
  research/await-lock-then-run.sh 900 bash research/e37-run.sh --census natural_history medicine
  python3 research/e37_width_census.py .mlxfast-private/e37/runs \
      natural_history medicine --payoff --json-out research/results/e37/census.json
  python3 research/e37_wandb_log.py research/results/e37/census.json
  ```

- **Tests and risk-based checks:** no `Sources/` change was made, so no Swift
  test was warranted. The runner asserts the golden covers the full 512-token
  window before each census and fails closed otherwise. The census parser reuses
  `research/depth_histogram.py`'s leg-splitting rather than reimplementing it,
  and refuses to proceed if more than one leg carries nonzero depths.
- **Exact-token and row-ledger verdict:** both census legs completed against
  512-step goldens with `exit=0` and no fidelity failure; `implied_d0 = 0` and
  the round ledger closes on both arms (258 rounds / 508 tokens and 227 / 507
  after the 2-round warmup drop; 260 / 512 and 229 / 512 including it).
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not relevant — no Metal source was touched.
- **Peak RAM / artifact size:** not measured; no bearing on a counts-only census.
- **Official status and score:** not submitted; not a submission candidate.
- **Known caveat:** the build emits a stale-`mlx.metallib` warning (recorded
  `3dd0ffd6…` vs current `1e359ea9…`). This is a **pre-existing base condition** —
  the diff touches only `research/` — and it cannot affect *counts*, which are
  decided by host-side policy code, not kernel contents. It would matter for a
  timing claim, and none is made.

| Metric | beagle proxy | medicine | ranked beagle | ranked medicine |
| --- | ---: | ---: | ---: | ---: |
| mean dispatched width M | 3.2016 | 3.6388 | 5.533 | 5.768 |
| max dispatched width M | 5 | 6 | ≥ 6 (inferred) | ≥ 6 (inferred) |
| row share at `M >= 6` | 0.0000 | 0.0581 | ≥ 0.2167 | ≥ 0.2996 |
| rounds choosing depth > 5 | 0 / 260 | 0 / 229 | > 0 (inferred) | > 0 (inferred) |
| depth-8 gate open | 16 (6.15 %) | 23 (10.04 %) | — | — |

No timing row is reported. The trace perturbs the round it counts, so a
serial-relative speedup would be meaningless and is deliberately omitted.

**W&B:** run `afefx5kd` —
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/afefx5kd>
(state `finished`), carrying the width census, gate cross-tab, ranked bracket
tables, and the `e37-width-census` artifact with the raw round traces and
per-arm provenance.

---

## Conclusion

- **What happened and why.** The census ran cleanly at scored geometry and
  answered the reachability question negatively and decisively. The beagle proxy
  never dispatches `M >= 6`; medicine does so in 3.5 % of rounds against a ranked
  floor of ≥ 19.2 %. The cause is not a cap: the depth-8 gate opened 39 times and
  the cost model declined every time, because these prompts are far less
  predictable than the ranked ones (mean chosen depth 2.20 vs 4.533) under an
  identical head and identical policy literals.
- **Evidence for or against the mechanism.** Against H1, conclusively —
  489 drafting rounds, 0 above depth 5. For deliverable 4, the ranked `M >= 6`
  mass is now bounded from published telemetry alone rather than simulated:
  row share ≥ 21.7 % (beagle), ≥ 30.0 % (medicine). For deliverable 3, the warm
  "gaps" are shape gaps, not pipeline gaps, because kernel names across the whole
  scored width range never encode M on either the target or the head side.
- **Prompt or M5 transfer risk.** High and now *quantified*. This is the sharpest
  finding: the widths that decide the ranked score (`M = 6..9`) are the widths
  local iteration cannot produce. Any future change to the `M >= 6` path — the
  advisor brief's main line included — is **locally unfalsifiable**, for speed
  *and for correctness*, and the shipped `sdpaWidthWallDepthCap` comment already
  records that widths 6-9 previously drifted in top-2 values while staying
  invisible to a local argmax check.
- **Smallest useful next action.** Add a forced-depth harness arm
  (`MLXFAST_QWEN_MTP_FORCE_DEPTH` or equivalent) that pins chosen depth to a
  fixed value regardless of the cost model, so widths 6..9 can be exercised
  locally for exactness and cost. Without it the campaign is optimising a cell it
  cannot observe. E25 already built forced-depth arms, so this is likely a small
  port rather than new machinery. **I did not implement this — it is outside the
  assignment's `research/`-only scope, since it needs a `Sources/` hook.**
- **Recommendation: close** H1 and the warm-coverage line. **Carry forward** the
  ranked `M >= 6` floor, the gqa-product cliff at `q_len = 6`, and the
  local-unfalsifiability finding into the ledger, and prioritise the forced-depth
  harness before any further `M >= 6` optimisation work.

### Suggested follow-ups (not implemented)

1. **Forced-depth local arm** — as above; the single highest-value unblocker.
2. **`gqa_factor` is the real lever on the SDPA cliff.** `q_len * gqa <= 32` with
   gqa = 6 gives the wall at 6. Splitting the 24 query heads into two 12-head
   SDPA calls halves the effective product per call and would raise the fused
   ceiling to `q_len = 10`, removing the two-call chunk for all scored widths.
   Whether that is cheaper than the current chunk is unmeasured.
3. **Ask whether `segmentedVerifyDepthCap = 8` earns its keep on rank.** It is
   dead locally; if ranked beagle's mass above depth 5 is small, the cap could be
   simplified away, and if it is large, it deserves dedicated optimisation.
4. **Re-derive E34's simulated masses once E34 merges**, to confirm the
   depth-5-ceiling assumption inferred in §4 and reconcile medicine (0.593
   simulated vs 0.768 implied by that same assumption).
