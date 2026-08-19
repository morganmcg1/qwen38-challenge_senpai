# E37 r2 — Dispatched verify-width (M) census: the benchmark's own fixture *does* reach the ranked regime

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_row_share_at_M_ge_6_benchfixture","available":true,"value":0.9171},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- **Student / branch:** `qwen-askeladd` / `qwen-askeladd/draft-width-census-beagle-medicine` (PR #42, revision **`r2`**)
- **Hypothesis (r1, restated):** H1 — the local prose proxies reach the ranked
  `M >= 6` operating cell often enough to profile it locally.
- **Decision:** **H1 remains falsified** for the two prose proxies. But r1's
  *structural generalisation* of that verdict — "no local instrument can reach
  the cell" — is **withdrawn**: the benchmark's own shipped fixture reaches and
  overshoots the ranked cell, and it does so through an identifiable mechanism.
- **`BASE_SHA`:** `0491f9e5` (rebased; carries thorfinn's E33 and alphonse's
  E35 artifacts) · **candidate commit:** this branch head ·
  **`UPSTREAM_SHA`:** unchanged, no organizer sync performed.
- **Yukon promoted submission / frontier:** not queried live; nothing here is a
  submission candidate. Payoff arithmetic is pinned to the two submission IDs
  recorded in `senpai/frontier-state.json` and read from the cached telemetry.
- **Submitted candidate files:** **none.** The diff touches only `research/`.
- **Supporting tooling / documentation files:** `research/e37-run.sh`,
  `research/e37r2_census.py`, `research/e37_width_census.py` (r1),
  `research/e37_alloc_regime.py`, `research/e37_shipped_surface.py`,
  `research/e37_wandb_log.py`, `research/results/e37/*`, this report.
- **MTP head provenance and draft policy:** declared proposal head from
  `mtp-head.manifest.json` — tree digest
  `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`,
  raw `model.safetensors` sha256
  `d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`, staged and
  verified by `research/fetch-declared-head.sh`. Draft policy is the shipped
  `costModelDepth` at its shipped literals; nothing was forced or overridden.
- **Assignment-scope preflight:** diff confined to `research/`. No `Sources/`,
  `Vendor/`, `mtp-head.manifest.json`, fixture, workflow or trusted file was
  modified — re-derived against the *named* campaign baseline `5273067` in §6e
  (`0 files, +0/−0` on the shipped surface, negative control live).
  `git status` was clean at every `run_job` launch
  (`dirty=0` in each r2 `meta.txt`).
- **Editable source bytes / headroom / growth:** unchanged — **0** candidate
  bytes added against the 3 000 000 / 524 288 / **262 144** limits.
- **Timing claims: none.** Every arm ran with `MLXFAST_LOCAL_COOL_GATE=0` and
  every `meta.txt` preserves `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `timing_claims_permitted=false` verbatim.

---

## 0. What r2 changed, and what I withdraw

| # | advisor request | where |
|---|---|---|
| a | re-census the benchmark's own fixture at 512 tokens from `effectiveDraftLengths` | §2 |
| b | report accept rate beside the histogram; is thorfinn's window degenerate; which side of beagle, and by what mechanism | §2b, §2c, §2d |
| c | deliverables 1–2 on a fixture that reaches the regime; validate or break E34's `.538` / `.593`; measured `w(M=6)` vs thorfinn's 20.1 % and my floor ≥ .2167 | §4, §5 |
| d | withdraw the structural claim; keep the falsified-H1 verdict | §0, below |
| e | *(addendum 1)* say how the split allocator regime bears on the warm-coverage negative | §6d |
| f | *(addendum 2)* the shipped surface is 5 files, +229/−74 against `5273067`, not 4 / +117/−87 | §6e |

**Withdrawn.** r1 §2a concluded: *"M = 7, 8, 9 are never dispatched locally at
all… A live scored code path exists that no local `--local-iterate` run can
exercise."* That is **false, and my own instrument falsifies it.** The shipped
fixture dispatches `M = 9` in **34 of 78 rounds** and `M >= 6` in **67 of 78**.
The claim I can defend is much narrower:

> *These two prose proxies* under-draft by roughly 2× relative to the ranked
> pair (`n` = 2.20 and 2.64 against 4.53 and 4.77, i.e. **48.6 %** and **55.3 %**
> fidelity), so neither is a usable stand-in for the ranked `M >= 6` cell.

**Kept.** H1 as written is falsified: on `natural_history` and `medicine`, 0 of
489 drafting rounds chose depth > 5, and the depth-8 gate opened 39 times and
was declined every time. That result is unchanged and is reproduced here from a
better instrument.

**Also corrected from r1:** r1's histograms were parsed from the phase trace,
which truncates — it saw 258 rounds / 508 tokens for `natural_history` and 227 /
507 for `medicine`. The trusted parent's journal records the true 260 / 512 and
229 / 512. All tables below come from the journal.

---

## 1. The instrument: `effectiveDraftLengths`, and why it is timing-free

**`M = d + 1` is source-proved, not assumed.** `Qwen36MTPBlockSession.swift:1049`
builds `verifyTokens = [primary] + draftIdArrays`, so a round that drafts `d`
tokens verifies exactly `d + 1` rows in one dispatch. A skip round (`d == 0`)
issues a 1-row forward, so `M = 1`.

**The counts come from the trusted parent, not from solver-side printing.**
`QwenRuntimeMTPDriver.swift:295` (trusted) and `:289` (harness) write
`effective_draft_lengths` as `rounds.map { $0.draftTokens.count }` — one element
per round, including skip rounds. r1 needed the phase trace; r2 does not, so
`E37_TRACE` now defaults to **0**.

**`costModelDepth` cannot see a clock.** I read it rather than assuming it
(`Qwen36MTPBlockSession.swift:700-744`). Its only inputs are `fullAcceptStreak`,
`positionAcceptEMA`, the `pendingTop2` margins, the constant
`headStepCostRatio`, and the integer cap. There is **no time source, no counter,
no wall-clock, and no I/O** in the decision. The trace branches only append
strings after the depth is chosen. So the depth vector is a deterministic
function of the token stream, and tracing *cannot* change it.

### 1a. Control: traced vs untraced, element-wise

That argument is also measured. Both proxy arms were re-run untraced and
compared element by element against their r1 traced counterparts:

| arm | traced run | untraced run | rounds | verdict |
|---|---|---|---:|---|
| `natural_history` | `2026-08-19T03:06:41Z` | `2026-08-19T03:59:03Z` | 260 vs 260 | **element-wise IDENTICAL** |
| `medicine` | `2026-08-19T03:10:44Z` | `2026-08-19T04:15:33Z` | 229 vs 229 | **element-wise IDENTICAL** |

**A defect I caught and fixed rather than shipped.** My first r2 pass reported
`medicine` as IDENTICAL when the two directories held *the same run* — the
backup copy compared against itself. `research/e37r2_census.py` now reads the
start instant from `meta.txt` and refuses the comparison as `VACUOUS` unless the
two runs are genuinely distinct; `medicine` was then re-run untraced to make the
control real. The `natural_history` control was valid throughout.

This converts r1's "counts only, trace perturbs timing" caveat into a
**measurement**: the trace perturbs timing, and does not perturb the counts.

---

## 2. The census — three arms, 512 decode tokens, offered depth 8

All three arms: `all_tokens_matched=true`, `residual_divergence_count=0`,
`non_drafting_round_count=0`, and the row ledger closes exactly
(`R + A = 512`; rejected + accepted = offered).

### `benchfixture` — the benchmark's own `--local-iterate` fixture

`correctness_prompts/public_longcopy_gate_english_512_256.json`
(sha256 `3d922b1a…`), 512-token seed used verbatim.

| M | rounds | round share | rows | row share |
|---:|---:|---:|---:|---:|
| 2 | 1 | 0.0128 | 2 | 0.0035 |
| 4 | 5 | 0.0641 | 20 | 0.0353 |
| 5 | 5 | 0.0641 | 25 | 0.0441 |
| **6** | **23** | **0.2949** | 138 | **0.2434** |
| 7 | 4 | 0.0513 | 28 | 0.0494 |
| 8 | 6 | 0.0769 | 48 | 0.0847 |
| **9** | **34** | **0.4359** | 306 | **0.5397** |

78 rounds · mean `M` **7.2692** · max `M` **9** · offered 489 draft tokens,
accepted 434, rejected 55 · **accept rate 0.8875**.
`M >= 6`: round share **0.8590**, row share **0.9171**.

### `medicine` (prose proxy)

| M | rounds | round share | rows | row share |
|---:|---:|---:|---:|---:|
| 2 | 6 | 0.0262 | 12 | 0.0144 |
| 3 | 108 | 0.4716 | 324 | 0.3890 |
| 4 | 86 | 0.3755 | 344 | 0.4130 |
| 5 | 21 | 0.0917 | 105 | 0.1261 |
| 6 | 8 | 0.0349 | 48 | 0.0576 |

229 rounds · mean `M` **3.6376** · max `M` 6 · offered 604, accepted 283,
rejected 321 · **accept rate 0.4685**. `M >= 6`: round **0.0349**, row **0.0576**.

### `natural_history` (the r1 beagle proxy)

| M | rounds | round share | rows | row share |
|---:|---:|---:|---:|---:|
| 2 | 41 | 0.1577 | 82 | 0.0984 |
| 3 | 139 | 0.5346 | 417 | 0.5006 |
| 4 | 66 | 0.2538 | 264 | 0.3169 |
| 5 | 14 | 0.0538 | 70 | 0.0840 |

260 rounds · mean `M` **3.2038** · max `M` 5 · offered 573, accepted 252,
rejected 321 · **accept rate 0.4398**. `M >= 6`: **0.0000** everywhere.

### 2a. Why the fixture reaches the regime — the mechanism

The fixture is a **verbatim copy task**, and that is checkable rather than
inferred: `expected_tokens[1:256]` is a single contiguous 255-token run inside
`prompt_tokens`, and the prompt text opens *"Copy the passage between the tags
exactly."* The proposal head therefore predicts from a source it can already
see, acceptance is near-deterministic, `fullAcceptStreak` stays above
`segmentedStreakGate = 2`, and `costModelDepth` runs at the cap.

This is not a new observation about the fixture — `research/e37_width_census.py`'s
own header, written at E11, already recorded a .89–.95 accept rate on it. What is
new is that its **width** distribution was never censused, and that this is what
makes it the only local instrument that reaches the ranked cell.

### 2b. Is thorfinn's window degenerate? No — but his was

thorfinn measured 54/54 acceptance on a short window and read it as saturation.
At the full 512-token ranked window the fixture is **not** degenerate:

- 55 rejections out of 489 offered draft tokens;
- `M` spans the full 2…9 range;
- 11 of 78 rounds sit below `M = 6`.

54/54 is a **small-window ramp artifact**: the copy region begins early and the
first rounds accept everything, so a short window samples only the ramp. The
mechanism is real; the saturation is not.

### 2c. The bimodality is the two caps, not the prompt

73 % of fixture rounds sit at exactly `M = 6` (23) or `M = 9` (34) — the two
shipped caps saturating:

- `sdpaWidthWallDepthCap = 5` → depth 5 → **`M = 6`** when the streak gate is shut;
- `segmentedVerifyDepthCap = 8` → depth 8 → **`M = 9`** when it is open.

So the fixture spends its time pinned against whichever cap is live. This is the
direct observation that kills r1's structural claim: `M = 7, 8, 9` are not only
reachable locally, they are the **modal** widths on the shipped fixture.

### 2d. Which side of beagle? The ranked pair sits *between* the two instruments

| instrument | mean `M` | accept rate | tokens/round |
|---|---:|---:|---:|
| `natural_history` proxy | 3.20 | 0.4398 | 1.97 |
| `medicine` proxy | 3.64 | 0.4685 | 2.24 |
| **ranked beagle** | **5.53** | **0.8351** | **4.79** |
| **ranked medicine** | **5.77** | **0.8750** | **5.17** |
| `benchfixture` | 7.27 | 0.8875 | 6.56 |

The fixture **overshoots** beagle on width (7.27 vs 5.53) while sitting almost
exactly on it in acceptance (0.8875 vs 0.8351). The proxies undershoot on both.
So the ranked pair is bracketed, and the fixture is the closer of the two
available instruments — but it is an **upper** bound on width, not a match.

---

## 3. Ranked round counts, recovered exactly

`effective_mean_draft_len` is published as an exact rational `n = D/R` where `R`
is the round count and `D` the offered draft tokens. With `R + A = 512`
(validated exactly on all three local arms), `α = A/D <= 1` forces
`R >= 512/(1+n)`, and `ρ = 512/(R · raw_p)` — the per-round cost in units of a
pinned serial token — must be monotone in mean `M`. Enumerating **every**
admissible multiple of the reduced denominator and keeping only readings
consistent with monotone `ρ` leaves **exactly one** solution per prompt:

| prompt | `n = D/R` | R | D | A | α | ρ | tokens/round |
|---|---|---:|---:|---:|---:|---:|---:|
| plutarch | 75/487 | 487 | 75 | 25 | 0.3333 | 0.8370 | 1.051 |
| drama | 193/84 | **252** | 579 | 260 | 0.4491 | 1.0565 | 2.032 |
| travel | 563/212 | 212 | 563 | 300 | 0.5329 | 1.1030 | 2.415 |
| beagle | 485/107 | **107** | 485 | 405 | **0.8351** | 1.5223 | 4.785 |
| medicine | 472/99 | **99** | 472 | 413 | **0.8750** | 1.5414 | 5.172 |
| essays | 472/87 | 87 | 472 | 425 | 0.9004 | 1.7357 | 5.885 |
| republic | 469/89 | 89 | 469 | 423 | 0.9019 | 1.6849 | 5.753 |
| botany | 491/85 | 85 | 491 | 427 | 0.8697 | 1.7464 | 6.024 |

**`drama` is not the smallest admissible multiple.** `R = 168` is arithmetically
legal but gives `ρ = 1.585`, which would place a mean-`M` 3.30 prompt above
beagle's 1.5223 — impossible. `R = 252` (`ρ = 1.0565`) is the unique monotone
reading. My earlier interim PR comment quoted the `R = 168` reading for drama;
that figure is **superseded** by this table. The beagle and medicine figures in
that comment (α = .8351 / .8750) are unchanged and remain correct.

`ρ` against mean `M` fits a straight line:

```
rho(meanM) = 0.5430 + 0.1770 * meanM      R^2 = 0.9711   (8 points, 2 parameters)
```

| prompt | mean M | ρ | residual |
|---|---:|---:|---:|
| plutarch | 1.1540 | 0.8370 | +0.0897 |
| drama | 3.2976 | 1.0565 | −0.0703 |
| travel | 3.6557 | 1.1030 | −0.0872 |
| beagle | 5.5327 | 1.5223 | −0.0002 |
| medicine | 5.7677 | 1.5414 | −0.0227 |
| essays | 6.4253 | 1.7357 | +0.0551 |
| republic | 6.2697 | 1.6849 | +0.0319 |
| botany | 6.7765 | 1.7464 | +0.0037 |

**This fit can fail, and that is the point.** It is over-determined — 8 points,
2 parameters — unlike the `R = (1+αn)/(1+h̄n)` identity I retracted earlier,
which was a re-parameterisation that could not fail. It does not fail: a round
costs ≈ 0.72 pinned-serial-tokens of fixed overhead plus ≈ 0.177 per extra
verified row. It is built **only** from published ranked rationals — no local
timing enters it, and none is claimed.

---

## 4. The measured `w(M=6)`, against the bracket and against thorfinn's 20.1 %

The r1 bracket is reproduced with **tightened support**: the ranked worker
publishes `non_drafting_round_count = 0` for both prompts, so depth 0 is
excluded and the feasible set is a distribution on {1…8} with two equality
constraints. Vertices are supported on at most two depths, so the extrema are
exact by enumeration. The `ρ(M)` fit above additionally converts row share into
**time** share:

| prompt | round share `M>=6` | **row share** | time share |
|---|---|---|---|
| beagle | [0.1332, 0.8832] | **[0.2166, 0.9578]** | [0.1869, 0.9312] |
| medicine | [0.1919, 0.9419] | **[0.2995, 0.9799]** | [0.2621, 0.9667] |

The floors are unchanged from r1 (.2166 / .2995) and the advisor verified them
independently.

**Against the requested comparison:**

| quantity | value | status |
|---|---:|---|
| thorfinn's `w(M=6)` (short window) | 20.1 % | small-window ramp artifact (§2b) |
| my ranked **floor** for `M>=6`, beagle rows | ≥ **21.66 %** | assumption-free, from published rationals |
| **measured** `w(M=6)` on `benchfixture`, rounds | **29.49 %** | this census |
| **measured** `w(M=6)` on `benchfixture`, rows | **24.34 %** | this census |
| measured `M>=6` on `benchfixture`, rows | **91.71 %** | this census |

`w(M=6)` — the weight on *exactly* `M = 6` — lands at 24.3 % of rows on the only
local instrument that reaches the regime, close to thorfinn's 20.1 % and just
above my beagle floor. But the fixture's `M >= 6` mass (91.7 % of rows) sits far
above beagle's floor, and its mean `M` overshoots beagle by 1.74. So the
agreement on `w(M=6)` is **coincidental**: the fixture and ranked beagle put
similar weight on the `M = 6` cap for different reasons — beagle because 6 is
near its mean, the fixture because 6 is its *lower* cap.

**Conclusion for the cell:** ranked beagle spends **at least 21.7 %** of its
dispatched target rows, and at least 18.7 % of its round time, at `M >= 6`. That
floor is the number to plan against; the fixture measurement neither raises nor
lowers it, but it does prove the cell is locally *runnable*.

---

## 5. E34's `.538` / `.593`: not validated, and not refutable by this route

Two separate findings, both negative for the simulation.

**First, `.538` is not the same quantity as `w(M=6)`.** E34's figure is a *round
share for `M >= 6`*, not the weight on exactly `M = 6`. Comparing it to
thorfinn's 20.1 % is a category error and I am flagging it rather than repeating
it. Against the like-for-like bracket, E34's `.538` and `.593` both sit inside
the feasible round-share intervals [0.1332, 0.8832] and [0.1919, 0.9419], so the
bracket cannot reject them.

**Second, the inference class E34 belongs to fails on measured data.** E34 could
not be re-run (`research/e34_cost_model.py` is not in this tree), so I scored the
maximum-entropy estimator — the same class of "infer the distribution from the
mean" argument, and the natural cross-check — against all three measured arms:

| arm | mean d | measured round `M>=6` | maxent | measured row `M>=6` | maxent |
|---|---:|---:|---:|---:|---:|
| `benchfixture` | 6.2692 | **0.8590** | 0.8257 | **0.9171** | 0.9049 |
| `medicine` | 2.6376 | **0.0349** | 0.1587 | **0.0576** | 0.3055 |
| `natural_history` | 2.2038 | **0.0000** | 0.0909 | **0.0000** | 0.1940 |
| ranked beagle | 4.5327 | — | 0.5062 | — | 0.6869 |
| ranked medicine | 4.7677 | — | 0.5510 | — | 0.7225 |

The error **changes sign** between mean d = 2.64 (maxent over-predicts by 4.5×)
and mean d = 6.27 (maxent under-predicts). **Ranked beagle, at 4.53, lies inside
that interval** — exactly where the estimator's bias is unconstrained by any
measurement I have. So the maxent route **cannot validate or break** E34's
numbers at the ranked operating point, and I am reporting that as a clean
negative rather than dressing the coincidence up as agreement.

For completeness, the maxent values on the ranked support {1…8} are .5062
(beagle) and .5510 (medicine) against E34's .538 / .593. The residual difference
is most likely that E34 allowed depth 0, which `non_drafting_round_count = 0`
rules out.

**Verdict: E34's simulated masses are neither validated nor broken.** They are
feasible but unverified, and the assumption-free floors in §4 should be quoted
in their place until the forced-depth harness exists.

---

## 6. Warm-coverage audit (r1, retained) and the allocator addendum

### 6a. The width wall — stated precisely

In `Vendor/mlx-swift/.../metal/scaled_dot_product_attention.cpp:621-640`, for
Qwen (head_dim **256**, gqa = 24/4 = **6**):

- `sdpa_full_supported_head_dim` requires head_dim ∈ {64, 80, 128} → false at
  every width. The steel path is permanently closed to this model.
- `supports_sdpa_vector` requires `q_len * gqa <= 32`.

So the precise statement is: **max legal `q_len` = 5, and the wall bites at 6.**
At `q_len >= 6` neither branch qualifies and SDPA decomposes into unfused
matmul+softmax, so `attentionWithCacheUpdate` (`Qwen35Attention.swift:200`)
splits 6..9-row attention into two `<= 5`-row calls. The `query_sequence_length
> 8` boundary in `supports_sdpa_full` is never reached — the operative cliff
comes from the **gqa product**, not from 8. I derived this from source before
reading the shipped rationale at `Qwen36MTPBlockSession.swift:648-661`, which
agrees.

### 6b. One PSO across the whole scored width range

- `get_qmv_batch_limit` (`quantized.cpp:84`) returns **10** for every scored
  shape on `applegpu_g16s`, and `eval_gpu:1415-1418` routes `M >= vector_limit`
  to `qmm`/`qmm_splitk`, so **every width `M <= 9` reaches `dispatch_qmv`**.
  Ranked M5 (gen ≥ 17) takes the same branch.
- `qmv()`'s kernel name `<mode>_qmv_fast_<type>_gs_64_b_4_batch_0` **does not
  encode M**, and `grid_dims(M, ceil(N/8), B)` makes `ntg.x == M`, so the E27
  crossrow IPG table's `switch (ntg.x)` lives inside a single compiled pipeline.
- The proposal head is itself fully 4-bit group-64 affine quantised (40 tensors;
  `fc` 10240→5120, `draft_lm_head` 2560→**98336**, a reduced draft vocabulary vs
  the target's 248320), so every head linear rides the same `dispatch_qmv`.

### 6c. Result: shape gaps, not pipeline gaps

| # | uncovered triple | verdict |
|---|---|---|
| G1 | head `fc`/norm/embed at `F = 3..9` (warm has 1, 2, 512) | **shape gap only** — same `dispatch_qmv` PSO |
| G2 | KV-history append at `F-1 = 2..8` (warm has 1, 511) | **shape gap only** — shape-generic copy kernels |
| G3 | warm uses `callWithHidden`, live verify uses `callWithHiddenAndNormed` | no extra pipeline; normed output is an elementwise epilogue |
| G4 | generic repair `callWithHidden(1+accepted, nConfirmed: 0)` at widths 2..9 | **shape gap only** — same PSO family |

**Negative result, reported as one:** no Metal pipeline compile is missed at any
scored width, and there is no warm-coverage headroom to harvest. I recommend
closing that line.

### 6d. Addendum — the allocator regime, re-derived rather than adopted

The addendum supplied a local-vs-ranked regime table and concluded my
warm-coverage negative is conservative. The whole claim is about which branch a
`guard` takes, so it is checkable, and I checked it instead of quoting it.
`research/e37_alloc_regime.py` is that check: **13 structural assertions**
(brace-matched function bodies, so a check cannot be satisfied by a coincidental
string elsewhere in the file) over the shipped policy, both MTP worker copies,
the MTP block session and the vendored MLX allocator/device — plus **13 mutation
negative controls**, each a targeted edit that must flip its target check to
`FAIL`. Both halves pass 13/13. Console and JSON are committed.

**Four rows confirm; one is inverted on the path that runs this census.**

| row | addendum | verified — Qwen-MTP worker path | check |
|---|---|---|---|
| `MLX_MAX_MB_PER_BUFFER` | 128 forced / 512 | 128 **forced** (`overwrite=1`) / 512 **default** (`overwrite=0`) | C2, C6 |
| `MLX_MAX_OPS_PER_BUFFER` | 64 forced / 50 | same, with the same forced-vs-default asymmetry | C2, C6 |
| `Memory.cacheLimit` | 6 GiB / MLX default | 6 GiB / MLX default = `min(1.5·maxRec, 0.95·memsize)` = **121.6 GiB** on a 128 GiB box whenever `maxRec >= 81.1 GiB`, which Apple's ~75 %-of-RAM recommendation satisfies | C4, C10 |
| clear cache after warmup | true / false | **inverted**: **no** clear locally, **one** clear on ranked | C7, C8, C9 |
| wired residency | OFF / ON | OFF / ON, same `>= 96 GiB` gate | C9 |

**Why that row inverts.** `clearAllocatorCacheAfterWarmup` is *never read on
this path*. Its only consumers are `LagunaRuntimeWeights.swift:395` and the two
DFlash workers (`:205`, `:204`) — the flag is inert for the MTP worker, which
inlines the policy by hand (trusted `QwenRuntimeMTPWorker.swift:487`, harness
`:498`) and applies only the two command-buffer budgets and the cache limit,
never the clear. The shipped
test suite agrees by omission: it pins the clear-after-warm ordering for the
*Laguna* initializer only. Meanwhile the single post-warm `Memory.clearCache()`
reachable from an MTP session is the *only* `clearCache` in that file
(`Qwen36MTPBlockSession.swift:235`) and sits inside
`wireResidentWeightsIfEnabled()` behind the *same* `physicalMemory >= 96 GiB`
gate as wired residency (`:225`). So the box that clears its allocator cache
after warm is the **ranked** one; this 48 GiB host never clears.

**The conclusion survives, and in a stronger form than the one offered.**

1. **The negative is regime-*invariant*, not merely conservative.** §6c is a
   claim about *pipeline* coverage. `MetalAllocator::clear_cache()` frees
   buffers only, and compiled pipelines live in `Device::library_kernels_`,
   which no allocator knob can reach (C11); `MLX_MAX_*_PER_BUFFER` are
   Device-scope command-buffer commit thresholds read once at Device
   construction and present at no kernel-selection site (C12). No memory-policy
   setting on either box can create, hide or evict a PSO. G1–G4 are shape gaps
   on both boxes or on neither.
2. **The residual first-touch cost is regime-dependent, and this box is the
   punishing one — on the rows that survived.** The ranked box runs a cache
   limit ~20× larger (121.6 GiB against a forced 6 GiB) and keeps the weights
   wired, both of which make a fresh allocation likelier to be served from the
   retained pool there than here. The local policy is in fact harsher than
   *stock MLX on this same host*: unforced, this 48 GiB box would cache up to
   `0.95 × 48 = 45.6 GiB` (binding whenever `maxRec >= 30.4 GiB`), so the 6 GiB
   cap is a 7.6× reduction against its own default, not only against ranked.
   Both multiples are arithmetic on the formula in C10 with the one input I did
   not measure — `maxRec` on either box — entering only through the stated
   inequality. That is the addendum's conclusion and
   it holds — but it is carried by the cache-limit and residency rows, not by
   the clear-after-warm row, which runs the other way, and not by warm-buffer
   retention: a *gap* shape has by construction no warm-phase buffer to retain,
   so retention can only help it through size-bucket reuse. **My bound was taken
   under the less favourable of the two allocator regimes and transfers
   upward.**
3. **The counts are untouched either way.** Memory and command-buffer geometry
   are not inputs to `costModelDepth` (§1), so every histogram above is
   unaffected by which side of either gate the host falls on.

Two side findings, both checkable, neither disturbing anything above:

- **The full profile is applied on no host at all.** All five application sites
  are `isLowMemory`-guarded (C13), so the policy's 320 MB / 128 ops / 32 GiB
  full-profile constants never execute anywhere. This is documented intent —
  `QwenRuntimeDFlashWorker.swift:124` says the full profile "stays a deliberate
  no-op, so the ranked box keeps the stock allocator behavior the pinned
  baseline was measured with" — but a reader of the struct would conclude the
  ranked box runs 320/128, and it does not: it runs 512/50, installed by the
  separate `>= 96 GiB` block inside `resolve()` that fires *before* the
  low/full branch and independently of it.
- **The 64–96 GiB band gets neither treatment** (C1 against C2/C3): such a host
  is above the low-memory threshold and below the command-buffer installer's
  gate, so it runs stock MLX geometry with no campaign setting at all. No
  campaign machine is in that band today; it is a latent trap for anyone who
  reads "ranked-like" as ">= 64 GiB".

### 6e. The shipped-surface baseline — re-derived, not accepted

Addendum item 2 retracts the "frozen at 4 files, +117/−87" constraint. I
re-derived the replacement rather than adopting it. Against the true campaign
baseline `5273067`, `Sources/` + `Vendor/` differ by **5 files, +229/−74** —
the corrected figure exactly — and the fifth file is indeed
`RuntimeStartupMemoryPolicy.swift` (+32/−0), the subsystem §6d is about.

My branch's contribution to that surface is **0 files, +0/−0**, and no
non-`research/` path appears in my diff at all.

`research/e37_shipped_surface.py` records this. It prints **both ends of every
range it compares**, because the failure being retired was a gate that never
named its own baseline, and it carries a negative control: the same
zero-bytes-added predicate, evaluated on a range that genuinely edits shipped
files, must report a violation — it reports 5. A scope check that has never seen
a non-zero input is not evidence, which is the same objection the r2 request
raised against my proxies.

---

## 7. Score payoff — on **our** row, with the corrected σ and gap

All of the following is computed from the exact submission IDs pinned in
`senpai/frontier-state.json` and verified against the published scores:

- our row **`ca9251b8`** — recomputed median **3.23250848260** vs reported
  **3.23250848263** (agreement to 1e-10);
- board top **`0cd0a6b4`** (ofou) — recomputed **3.24929398550** vs reported
  **3.24929398547**;
- every prompt's `raw_p` equals `serial / candidate` exactly (0 mismatches), and
  every `effective_mean_draft_len` matches the §3 table exactly.

**Payoff belongs on our row, not the leader's.** The binding upper neighbour is
whichever prompt sits at rank 6 in **our** ordering. Our ascending order is
plutarch, drama, travel, **beagle**, **medicine**, essays, republic, botany —
so the scored cells are beagle and medicine, and the ceiling is **essays at
`raw_p` = 3.366118**.

| cell | our `raw_p` | ceiling | **headroom** | score if saturated |
|---|---:|---:|---:|---:|
| beagle | 3.120154 | 3.366118 | **+7.883 %** | **+3.8045 %** (41.2σ) |
| medicine | 3.344863 | 3.366118 | **+0.635 %** | **+0.3288 %** (3.6σ) |
| both | — | — | — | +4.1333 % (44.8σ) |

**This corrects r1.** r1 §4 quoted medicine headroom as **+1.06 %** and
"corrected" an earlier +0.64 % to get there. +1.06 % is right *on the leader's
row* (3.390664 / 3.355262 − 1); **+0.635 % is the correct figure on ours**, and
ours is the row we can move. The medicine-saturated score gain of **+0.3288 %**
independently cross-checks alphonse's **+0.318 %** (ledger 132) to within 3 %.

**σ and the gap, both relabelled.**

| quantity | value | provenance |
|---|---:|---|
| σ_score | **0.0923 %** | ledger 131 / E35 — effective σ on the crown's steep profile, where the central pair pins to beagle+medicine and kills the median's averaging. r1's 0.078 % was the organizers' six-session sd *before* that correction. |
| detection threshold (2σ) | **0.185 %** | |
| gap to crown, official `R` | **−0.5166 %** | ratio of published medians; quotable **if labelled** |
| gap to crown, `R'` | **−0.5611 %** | `mean_8(own serial) / cand_p`; **this is the engineerable gap** |

I recomputed `R'` from raw telemetry rather than importing it: it lands at
**−0.5611 %** against ledger 131's −0.561 %, i.e. **6.1σ**. r1 and my earlier
briefs sized work against 0.2587 %; that estimator (`R*`) is retired.

**Independent confirmation of ledger 131's premise:** I scanned all 422 fully
scored rows in the cached telemetry for two submissions sharing a
`submissionCommitSha` or `promotedSourceRef`. There are **zero** such pairs. So
σ_score genuinely cannot be measured from the board and must be imported —
"submit one tree twice" remains the only clean route to a σ of our own.

**What a 1 % speedup of the `M >= 6` beagle cell is worth**, as a function of the
candidate-leg time share φ it holds:

| φ | Δscore | in σ | cell speedup needed for 2σ |
|---:|---:|---:|---:|
| 0.15 | +0.0725 % | 0.79σ | 2.5 % |
| 0.20 | +0.0967 % | 1.05σ | 1.9 % |
| **0.2166** (row-share floor) | **+0.1048 %** | **1.14σ** | **1.8 %** |
| 0.25 | +0.1210 % | 1.31σ | 1.5 % |
| 0.2734 | +0.1323 % | 1.43σ | 1.4 % |
| 0.30 | +0.1452 % | 1.57σ | 1.3 % |
| 0.50 | +0.2425 % | 2.63σ | 0.8 % |

**The E38 number:** at the assumption-free floor, a 1 % speedup of the `M >= 6`
beagle cell buys **≈ 1.1σ** — real but under the 2σ detection threshold. Roughly
**1.8 %** is needed to be decisive. The cell is worth attacking (a fully
saturated beagle cell is +3.80 % of score), but a sub-1 % kernel win inside it is
**not measurable on rank**.

---

## Evidence

- **Host, memory profile, toolchain, thermal policy:** Apple **M4 Pro**
  `Mac16,11`, `applegpu_g16s`, 20 GPU cores, 48 GiB
  (`hw.memsize = 51539607552`), macOS 26.5.2, Swift 6.3.3. **Not the ranked M5.**
  `MLXFAST_LOCAL_COOL_GATE=0` throughout; entry/exit GPU temperature recorded per
  arm (`benchfixture` 37.0 °C → 61.9 °C; `natural_history` 58.8 °C → 64.2 °C;
  `medicine` 37.0 °C → 62.7 °C).
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `timing_claims_permitted=false` are preserved verbatim in every `meta.txt`.
  **No timing figure is reported anywhere in this document.**
- **Exact reproduction commands** (every GPU touch via the lock wrapper and
  `run_job`; worktree clean at launch):

  ```bash
  bash research/fetch-declared-head.sh
  research/await-lock-then-run.sh 600 research/e37-run.sh --golden natural_history medicine
  # r2, untraced (E37_TRACE defaults to 0):
  research/await-lock-then-run.sh 600 research/e37-run.sh --census benchfixture natural_history
  research/await-lock-then-run.sh 600 research/e37-run.sh --census medicine
  python3 research/e37r2_census.py | tee research/results/e37/r2-console.txt
  # addendum, zero GPU, read-only on the tree (both exit non-zero on failure):
  python3 research/e37_alloc_regime.py    | tee research/results/e37/r2-alloc-regime.txt
  python3 research/e37_shipped_surface.py | tee research/results/e37/r2-shipped-surface.txt
  python3 research/e37_wandb_log.py research/results/e37/r2-census.json
  python3 research/e37_wandb_addendum.py  # updates run h977ws5a in place
  ```

- **Committed raw artifacts:** `research/results/e37/r2-census.json` (per-arm
  depth vectors, per-arm `meta.txt`, ranked resolution with the rejected
  alternatives, ρ fit, brackets, payoff) and
  `research/results/e37/r2-console.txt` (full console transcript). The r1
  artifacts (`census.json`, `*-rounds.txt`, `*-meta.txt`) are retained unchanged
  for comparison. Addendum: `r2-alloc-regime.json` / `.txt` (13 checks, 13
  mutation controls, derived regime, cited line numbers) and
  `r2-shipped-surface.json` / `.txt`. The surface JSON's `head` names the commit
  the gate ran against — one commit behind the commit that publishes it, which
  adds `research/` paths only and so cannot move the gated predicate.
- **Metal / twin audit:** not relevant — no Metal source was touched.
  `metallib_fingerprint` is recorded **read-only** in each `meta.txt`
  (`1e359ea9…`). I deliberately did **not** rebuild `mlx.metallib`: a rebuild
  would disturb a sibling's timing lock, and holding it fixed keeps kernel
  content constant across all arms. The build's stale-metallib warning is a
  pre-existing base condition that cannot affect counts, which are decided by
  host-side policy code.
- **Tests and risk-based checks:** no `Sources/` change, so no Swift test was
  warranted. The runner asserts the golden covers the full 512-token window
  before each census and fails closed otherwise; for `benchfixture` the shipped
  fixture supplies its own reference rows and an advisory coverage tripwire is
  recorded. The census tool refuses a traced-vs-untraced comparison when both
  directories hold the same run (§1a).
- **Exact-token and row-ledger verdict:** all arms `exit=0`,
  `all_tokens_matched=true`, `residual_divergence_count=0`,
  `non_drafting_round_count=0`, `R + A = 512` exactly. No divergent tokens.
- **Peak RAM / artifact size:** not measured; no bearing on a counts-only census.
- **Official status and score:** not submitted; not a submission candidate. Our
  best ranked row remains `ca9251b8` (3.23250848263467, rank 9, *rejected: score
  did not improve current best*).

| Metric | `natural_history` | `medicine` | **`benchfixture`** | ranked beagle | ranked medicine |
| --- | ---: | ---: | ---: | ---: | ---: |
| rounds | 260 | 229 | **78** | 107 | 99 |
| mean dispatched width `M` | 3.2038 | 3.6376 | **7.2692** | 5.5327 | 5.7677 |
| max dispatched width `M` | 5 | 6 | **9** | ≤ 9 | ≤ 9 |
| accept rate | 0.4398 | 0.4685 | **0.8875** | 0.8351 | 0.8750 |
| round share `M >= 6` | 0.0000 | 0.0349 | **0.8590** | ≥ 0.1332 | ≥ 0.1919 |
| row share `M >= 6` | 0.0000 | 0.0576 | **0.9171** | ≥ 0.2166 | ≥ 0.2995 |
| `w(M=6)` rows | 0.0000 | 0.0576 | **0.2434** | — | — |
| `all_tokens_matched` | true | true | **true** | — | — |

**W&B:** r2 run `h977ws5a` —
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/h977ws5a>
(state `finished`), carrying the width census, per-arm provenance and thermal
record, the traced-vs-untraced control, the ranked resolution with its rejected
readings, the ρ fit, the brackets, the payoff tables, and the
`e37-width-census` artifact including this report. The addendum was **resumed
onto the same run** rather than forked to a new id — it adds no measurement —
and contributes the `addendum_alloc_checks`, `addendum_alloc_negative_controls`
and `addendum_alloc_regime` tables, the `addendum/*` and `surface/*` summary
keys, and the `e37-addendum` artifact. `research/e37_wandb_addendum.py` re-runs
both gates before logging and refuses to publish a failing one. The r1 run is
`afefx5kd`; its structural-claim summary keys are superseded by `h977ws5a`.

---

## Conclusion

- **What happened.** Re-censusing from the trusted parent's journal instead of
  the phase trace, and adding the benchmark's own fixture as a third arm,
  overturned my r1 structural claim while confirming my r1 verdict. H1 is still
  dead for the prose proxies. But the shipped fixture dispatches `M = 9` in 34 of
  78 rounds, so the `M = 6..9` path is **locally reachable, runnable, and
  regression-testable today** — the campaign is not blind to it.
- **Why the fixture works.** It is a verbatim copy task, so acceptance is
  near-deterministic and the streak gate stays open, pinning `costModelDepth`
  against whichever cap is live. That also bounds its usefulness: it overshoots
  ranked beagle's mean width by 1.74 and is an upper bound, not a match.
- **Evidence quality.** The counts are trusted-parent records, the trace-vs-
  untraced control is element-wise identical on two arms, `costModelDepth` is
  provably timing-free by source read, and the ranked round counts are recovered
  by exhaustive enumeration with a monotonicity constraint rather than by a
  greedy guess. The ρ(M) fit is over-determined and could have failed; it did
  not (R² = 0.971).
- **Transfer risk.** Moderate, and lower than r1 claimed. The fixture reaches the
  regime but from above; the ranked pair remains bracketed rather than matched. A
  forced-depth harness is still the clean instrument, but it is no longer a
  precondition for touching `M >= 6`.
- **The warm-coverage negative is conservative in the direction that matters.**
  Its core — no PSO is missed at any scored width — is invariant to the startup
  memory policy, because allocator knobs cannot reach `Device::library_kernels_`.
  Its residual — the one-time first-touch allocation a shape gap would pay — was
  measured on the box with the smaller allocator cache (a forced 6 GiB against
  MLX's ~120 GiB default) and without wired residency, so it transfers upward to
  the ranked box. One row of the regime table I was given inverts on this code
  path (§6d) and the conclusion is unaffected.
- **Sizing.** Work in the `M >= 6` beagle cell should be sized against
  **σ_score = 0.0923 %** and a **0.561 %** engineerable gap, not against 0.078 %
  and 0.2587 %. At the floor, a 1 % cell win is 1.1σ; ~1.8 % is needed to be
  decisive on rank.
- **Recommendation.** **Close** H1 and the warm-coverage line. **Carry forward**
  the ranked `M >= 6` floors, the ρ(M) round-cost law, the exact ranked round
  counts, the `q_len = 5` / wall-at-6 statement, and — newly — the shipped
  fixture as the campaign's local `M >= 6` instrument.

### Suggested follow-ups (not implemented)

1. **Adopt `benchfixture` as the standing `M >= 6` regression arm.** It is the
   only local prompt that exercises widths 7, 8, 9, including for *exactness*.
   Cheap, already wired, and needs no `Sources/` hook.
2. **Forced-depth local arm** (`MLXFAST_QWEN_MTP_FORCE_DEPTH` or equivalent) to
   pin chosen depth independently of the cost model, so a chosen width can be
   held fixed while a kernel change is measured. Still the cleanest instrument;
   needs a `Sources/` hook, so out of scope here.
3. **Build a copy-task prompt with a *lower* mean width.** The fixture overshoots
   beagle; a copy prompt with periodic interruptions would let the local mean `M`
   be tuned onto 5.53 and turn the bracket into a match.
4. **`gqa_factor` is the real lever on the SDPA cliff.** Splitting the 24 query
   heads into two 12-head SDPA calls halves the product per call and would raise
   the fused ceiling to `q_len = 10`, removing the two-call chunk at all scored
   widths. Whether that is cheaper than the current chunk is unmeasured — and it
   is now locally measurable on `benchfixture`.
5. **Submit one tree twice** to obtain a σ_score of our own. Zero rows on the
   board have ever been resubmitted identically (confirmed here), so the campaign
   is importing a σ it has never measured.
6. **Decide whether `clearAllocatorCacheAfterWarmup` should be honoured by the
   MTP worker.** Today it is set by the low profile and read by nobody on that
   path (§6d, C7/C8), so a local box carries warm scratch into its decode window
   while the ranked box clears it. Either wire it up or delete it, but the
   current state is a flag that documents an intent the worker does not execute.
   Not attempted here: it is a `Sources/` change, and any allocator edit needs a
   gate-qualified timing arm this experiment is not permitted to produce.
7. **Point the shipped-surface gate at a named baseline in CI.** `5273067` is
   the baseline; `research/e37_shipped_surface.py` shows the shape of the check
   (both ends printed, negative control included) and agrees with the advisor's
   corrected 5 / +229/−74. Two independent implementations now report the same
   number, which is the cheapest available guard against a third silent drift.
