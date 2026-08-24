# E166 — Ranked depth optimum: the price is not the lever

**Status: hypothesis refuted at the source level. No candidate advanced. No Yukon submission.**

- Assignment: `e166-ranked-depth-optimum`, revision `r0`, PR #167.
- Base: `4b6eb4f59a1f7b2722b9ec48a46d55f67a233753`.
- Host: AWS Mac, Apple M4 Pro, `harness=local` for every measured leg.
- Every timed leg in this report is **ungated**: `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`. Nothing here is an official or ranked score.

## 1. Question and answer

**Question.** The brief states that the shipped depth policy under-drafts on the
ranked M5 because its cost calibration carries a LAW-354 weight-stream step that
the ranked host does not pay. Substituting the ranked round price
`R(M) = 16.1585 + 5.3350*M` should therefore raise the offered depth toward the
cap on high-acceptance prompts, worth about `+4.19 %` published.

**Answer.** The premise is false in the shipped source, and the briefed change is
a regression. Four independent readings agree:

1. The shipped depth objective contains **no width-conditional term at all**, so a
   width-dependent weight-stream step cannot be inside it.
2. The shipped price is **dimensionless**. Translating it into ranked
   milliseconds makes the shipped policy *more expensive per row* than the truth,
   not cheaper. Correcting the price makes the policy **shallower**, never deeper.
3. At each prompt's measured acceptance rate, the shipped rule **already chooses
   `d = 7`** on all five high-acceptance prompts — the same depth the ranked
   global optimum wants. There is no depth left to gain.
4. On a traced 512-token leg the shipped policy already spends **77 % of its
   rounds at the cap**, mean depth `6.359`. Replaying that leg with the ranked
   price lowers mean depth to `5.821` and makes ranked seconds per token
   **`1.97 %` worse**.

Result for the briefed arm: **`-1.97 %` ranked seconds per token (a regression)**
against a briefed `+4.19 %` gain. The sign is wrong.

The same replay shows where the depth actually is: releasing the two top-2 margin
clamps raises mean depth to `6.667` and improves ranked seconds per token by
**`0.60 %`**. That is the follow-up worth running.

## 2. Step 1 — reading the shipped policy

Source: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`.

### 2.1 The objective

`:1091-1111` performs a **greedy myopic** minimisation of the dimensionless cost

```text
f(d) = (1 + h*d) / (1 + E(d)),    E(d) = sum_{k<d} p_k
```

with `h = headStepCostRatio = 0.18` (`:892`). The walk accepts `d+1` only while
`f` strictly decreases, then stops. The offered depth is finally clamped by
`min(offeredDepth, 8, segmentedVerifyDepthCap = 7)`.

Under `depthPriceArm = .ship` (`:1010`) the per-row price is a **single uniform
constant**. It does not read the row count, the verify width, the layer mix, or
any weight-stream state.

### 2.2 Where the only width-conditional shape lives

The one width-conditional table in the file is
`measuredRawDepthPrice[4] = 0.63287…`, and it is reachable only under
`depthPriceArm = .pbfit`, which is **not selected**. `sdpaWidthWallDepthCap = 5`
is dead code: only its own declaration and one comment reference it.

So the shipped scored path has no place for a LAW-354 step to hide.

### 2.3 The acceptance estimator, which *is* width-sensitive

`p_k` comes from `positionAcceptEMA`:

- prior `0.85 * 0.98^i` (`:856`),
- update rate `alpha = 0.15`,
- optimism cap `0.95` (`:1218`),
- then two top-2 **margin clamps** applied before the walk (`:1090-1098`):
  `p0 <- min(p0, sigmoid(m/2))` and `p1 <- min(p1, sigmoid(m/3))`,
  where `m` is the current top-2 logit margin.

These clamps and the per-position decay are the terms that actually bind the
depth. They are the lever the brief was looking for; the price is not.

### 2.4 Price translation (`harness` labels kept separate)

| harness | law | round at `d=0` (ms) | marginal row (ms) | implied dimensionless ratio | shipped `0.18` expressed in ms | shipped / truth |
|---|---|---|---|---|---|---|
| `ranked` | `R(M) = 16.1585 + 5.3350*M` | 21.4935 | 5.3350 | **0.24821** | 3.86883 | 0.7252 |
| `local` (E159 destepped) | `R(M) = 47.756 + 8.6457*M` | 56.4017 | 8.6457 | **0.15329** | 10.1523 | 1.1743 |
| `local_raw` (LAW 354 step still inside) | raw curve | — | 12.8215 (mean) | — | — | — |

The ranked truth wants `0.24821`; the shipped policy uses `0.18`. A **larger**
ratio is a **more expensive** row, which shortens the greedy walk. The briefed
substitution therefore pushes depth **down**.

`senpai/verify-ranked-score-boundary.sh` passes on this base, and no local
serial share is subtracted anywhere in this report.

### 2.5 Per-prompt depth table (`harness=ranked`)

`q` solved from the exact ranked accept ledger; the shipped greedy rule is
evaluated at that flat `q`.

| prompt | `q` | shipped EDL | ranked global optimum `d*` | shipped greedy `d` | ranked-price greedy `d` |
|---|---|---|---|---|---|
| beagle | 0.9341 | 4.38 | 7 | **7** | 7 |
| essays | 0.9647 | 5.09 | 7 | **7** | 7 |
| republic | 0.9661 | 4.99 | 7 | **7** | 7 |
| medicine | 0.9639 | 5.26 | 7 | **7** | 7 |
| botany | 0.9598 | 6.15 | 7 | **7** | 7 |
| travel | 0.6974 | 2.65 | 2 | 3 | **2** |
| drama | 0.6030 | 2.30 | 2 | 2 | 2 |
| plutarch | 0.3158 | 1.16 | 1 | 1 | 1 |

On every high-acceptance prompt the shipped price and the ranked price agree on
`d = 7`. The single row where the prices disagree is `travel`, and there the
briefed change moves `3 -> 2`.

The gap between "the rule would choose 7" and "the leg realises EDL 4.38-6.15" is
therefore **not** a price gap. It is produced by the estimator: the per-position
EMA decay and the two margin clamps hold `p_k` below the flat `q` the ledger
implies.

## 3. Step 2 — acceptance arithmetic (control; does not kill the experiment)

`q` solved from the E159 accepted counts at pinned depth `d = 1..7`:

```text
d:  1        2        3        4        5        6        7
q:  0.98070  0.96935  0.97328  0.96876  0.95337  0.96222  0.96350
```

- mean `0.96731`, sd `0.00872`, spread `2.83 %`;
- mean over `d <= 3` = `0.97444`, mean over `d >= 4` = `0.96196`.

Acceptance does **not** collapse with depth, so a deep-draft arm is not blocked by
acceptance decay. This control is clean; the experiment dies on the price
argument alone.

E159 exactness on the same fixture: all 11 legs `matched=true`,
`residual_divergence_count=0`, including the `asc7` leg at `d=7`, `M=8`.

## 4. Step 0 — island prelude (FINDING 372 mechanism)

Rebuild at branch HEAD produced `cli_sha256=c81b0779…` and a **new**
`worker_sha256=70ebae60…` (previously `deafb20e…`) although the sources are
byte-identical to `001f8e24`. This is build non-determinism, consistent with
ledger 202(I).

Four legs, ABBA, 128 tokens, pinned depth 4, one binary, one environment
variable apart. **Arm `S` is the control and sets
`DARKBLOOM_QWEN_MTP_ISLAND_ARM=all` (islands on); arm `P` sets `…=none`
(islands off).**

| quantity | S (islands on) | P (islands off) |
|---|---|---|
| rounds | 27 | 27 |
| EDL | 3.814815 | 3.814815 |
| accepted / rejected | 102 / 1 | 102 / 1 |
| rows declared / checked | 130 / 130 | 130 / 130 |
| residual divergences | 0 | 0 |
| width histogram | `{1:1, 2:1, 4:25}` | `{1:1, 2:1, 4:25}` |

The two arms produce **digit-identical ledgers**. This is FINDING 372
**mechanism (b)**: the island transform changes runtime only, never the decision
walk or the accept ledger.

Entry temperatures 58.7-59.4 C, ungated by construction.

### 4.1 Step 0b closed the witness gap and corrected the arm mapping

Step 0 produced no **arm witness**, because
`Qwen35IslandArm.writeWitness`
(`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:2925-2942`) writes only
to `MLX_QWEN_MTP_TRACE_PATH`, which step 0 did not set. Step 0b repeated the ABBA
with that path set. Every leg now carries a read-back witness:

```text
research/out/e166s0w/b0-S  qwen-mtp-island-arm: all  installsQ=true  installsKV=true
research/out/e166s0w/b0-P  qwen-mtp-island-arm: none installsQ=false installsKV=false
research/out/e166s0w/b1-S  qwen-mtp-island-arm: all  installsQ=true  installsKV=true
research/out/e166s0w/b1-P  qwen-mtp-island-arm: none installsQ=false installsKV=false
```

The witness proves the environment variable reached the install site and that the
two arms really differ. Session `e166s0w` reproduces the same four-leg
digit-identical ledger (rounds 27, EDL 3.814815, accepted 102, rejected 1, rows
130/130, residual 0, histogram `{1:1, 2:1, 4:25}`).

### 4.2 Runtime contrast, stated with the correct sign

The harness reports the contrast **relative to the control arm `S`**, so a
positive effect means the *islands-off* arm is slower.

| session | effect on `P` (islands off) | percent | bar | significant at 95 % |
|---|---|---|---|---|
| `e166s0` | `+385.4 us / round` | `+0.427 %` | 180.5 us | no (1 block, CI undefined) |
| `e166s0w` | `+283.6 us / round` | `+0.314 %` | 180.6 us | no (1 block, CI undefined) |

Read plainly: **turning islands off costs `0.31-0.43 %` per round**, so the
island transform is the faster arm. Two independent sessions agree on the sign
and both exceed the 0.2 % bar, but each has a single estimate block, so
`significant_at_95=false` and `clears_bar=false` in both reports. This is
directional evidence only, and it is not part of the E166 hypothesis test.

My step 0 note recorded the raw number `+385.4 us` without resolving which arm
carried the islands. The witness resolves it; the sign above supersedes that
note.

## 5. Traced adaptive leg — the refutation measured, not just derived

Session `e166trace`: 512 decode tokens, offered depth 8, arm `adapt`
(`MLX_QWEN_MTP_TRACE=1`) against an untraced control arm `adaptc`.

Both legs: `matched=true`, 78 rounds, EDL `6.3589743589743586` — **identical to
the last digit**. Seconds per token `0.02869482` traced against `0.02865331`
untraced, so tracing costs `+0.145 %` and does not move the decision walk. The
trace is therefore a faithful record of the shipped policy.

### 5.1 The trace records the decision, not only its outcome

Each round writes `sched=i:p/reach/threshold;…`, where `p` is the post-clamp
per-position accept probability, `reach = prod_{k<=i} p_k`, and `threshold` is
the value `reach` must exceed for the walk to add draft `i`:

```text
threshold_i = h * (1 + E(i)) / (1 + h*i)
```

This is exactly the strict-decrease condition of `f(d) = (1 + h*d) / (1 + E(d))`.

**Positive control.** Rebuilding `p`, `reach` and `threshold` from the model over
all **513 steps of all 78 rounds** gives worst absolute errors of `4.99e-07`,
`1.76e-06` and `8.56e-07` — the trace prints six decimals, so these are printing
residue. `model_reproduces_trace_fields = true`. The replay below uses the
shipped objective, price and clamp, not a paraphrase of them.

**Walk control.** The replayed shipped depth equals the realised depth on
**77 of 77 interior rounds** (`interior_match_rate = 1.0`). The only difference
is round 78, where the policy asks for `d = 7` but the leg emits `d = 1`, because
the 512-token window has one token left. That is a parent window boundary, not a
policy disagreement, and it is reported separately rather than folded into the
match rate.

### 5.2 The shipped policy is already deep

Realised width histogram over 78 rounds:

```text
d:      1   3   4   5   6    7
rounds: 1   4   5   5   3   60
```

Mean realised depth `6.359`; **`60 / 78 = 77 %` of rounds already sit at the
`segmentedVerifyDepthCap = 7`**. There is very little depth headroom for any
price change to recover.

The two top-2 margin clamps bind in **38 of 78 rounds (48.7 %)**. They, not the
price, are the term that holds the remaining rounds short of the cap.

### 5.3 Open-loop counterfactual, priced in the ranked harness

Every EMA is held at its shipped trajectory and one policy term is swapped. The
EMAs are a closed loop in a live arm, so these are first-order readings, not
predictions of a live arm. Cost is priced with `R(M) = 16.1585 + 5.3350*M`
(`harness=ranked`) and the token gain uses the measured E159 acceptance.

| variant | price ratio | margin clamp | mean depth | mean offered rows `M` | ranked round (ms) | ranked s/token (ms) | **ranked s/token vs ship** |
|---|---|---|---|---|---|---|---|
| `ship` | 0.18 | yes | 6.436 | 7.436 | 55.829 | 8.4664 | `0.000 %` |
| **`ranked_price`** (the brief) | 0.24821 | yes | **5.821** | 6.821 | 52.546 | 8.6330 | **`+1.968 %` (worse)** |
| `no_margin_clamp` | 0.18 | no | 6.667 | 7.667 | 57.060 | 8.4153 | **`-0.604 %` (better)** |
| `no_clamp_ranked_price` | 0.24821 | no | 6.333 | 7.333 | 55.282 | 8.4768 | `+0.123 %` |

Read the second row against the brief. The briefed substitution moves the mean
depth **down** from `6.436` to `5.821` and makes ranked seconds per token
**`1.97 %` worse**. The brief predicted `+4.19 %` better. The sign is wrong and
the magnitude is wrong.

The third row is the measured version of my step 1 recommendation: removing the
two margin clamps raises mean depth to `6.667` and improves ranked seconds per
token by `0.60 %`. That is the lever, and it is worth a real arm.

## 6. Gates

| gate | result |
|---|---|
| `senpai/verify-ranked-score-boundary.sh` | PASS |
| `senpai/check-editable-budget.sh` (vs promoted base) | OK — source 2,619,256 / 3,000,000; growth 164,421 / 262,144 |
| `senpai/check-editable-budget.sh` (vs own base `4b6eb4f5`) | growth **0** |
| `python3 research/twin_audit.py` | TWIN AUDIT OK (29 twins) |
| `senpai/validate-assignment-scope.sh 4b6eb4f5 <changed files>` | every changed file outside `editablePaths` |

`git diff --name-only 4b6eb4f5 HEAD` intersected with the 89 `editablePaths`
entries is **empty**. This branch changes no submitted file.

## 7. What I did not do, and why

- **Step 3 (build the ranked-price arm) was not built.** Step 1 is decisive
  before any GPU time: the arm's own arithmetic predicts `+0.00 %` on the five
  high-acceptance prompts and a regression on `travel`. Spending a build and an
  ABBA session to measure a change that the source says cannot move the depth
  would burn the pre-official budget for no decision.
- **No Yukon submission.** The advisor holds the official slot for `edward`.

## 8. Suggested follow-ups (not implemented)

1. **The acceptance estimator, not the price.** The binding terms are the two
   top-2 margin clamps at positions 0 and 1 and the `0.95` optimism cap over a
   `0.98^i` prior. The clamps bind in `38 / 78` traced rounds, and releasing
   them in the open-loop replay is worth `-0.60 %` ranked seconds per token. A
   clamp-relaxation arm is the falsifiable version of the brief's intent and is
   measurable on the same fixture.

   Two cautions for whoever runs it. The replay is **open loop**: it holds the
   EMAs on the shipped trajectory, so a live arm that drafts deeper will also
   feed different acceptance samples back into the EMA and the realised gain may
   differ. And the clamps exist to protect exactness margin on low-confidence
   rounds, so the arm must carry the full exact-token and row-ledger check, not
   only a timing delta. Relaxing rather than removing the clamps — for example
   `sigmoid(m/2) -> sigmoid(m/1.5)` — is the safer first dose.
2. **The ranked price axis is closed on both sides.** Scanning the ratio gives
   `0.14 -> 2.766`, `0.15 -> 2.667`, `0.18` shipped, `0.32 -> 2.84585`. There is
   no price value that raises depth on the high-acceptance prompts, so no further
   price arm is worth a session.
3. **`sdpaWidthWallDepthCap = 5` is dead.** Removing it, or wiring it, is a
   separate decision. It currently misleads a reader into thinking a width wall
   is enforced.
