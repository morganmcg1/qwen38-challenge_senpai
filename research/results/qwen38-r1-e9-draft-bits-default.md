# Result — `qwen38-r1-e9-draft-bits-default` (revision `r2`)

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mtp_seconds_per_token","available":true,"value":0.03331521479412913},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

**Result label: `local winner` — with one honest qualification.** Flipping the
in-tree `draftHeadBits` default from 4 to 3 is worth **−2.0303 %** MTP
seconds/token on the cap-7 base, correctness is exact on every arm, and the
candidate diff is a single submitted file. But the gain is **not** the kernel
speedup the hypothesis assumed: it is an acceptance-rate rise that is
deterministic and bit-exactly reproducible yet **fixture-specific**, and it
landed the other way on one of the two unseen prompts. Project the ranked median
from Part A's three-prompt median of **−1.520 %**, not from −2.0303 %.

- **Student / branch:** `qwen-askeladd` / `qwen-askeladd/draft-bits-multiprompt-default`
- **Hypothesis and target cost:** the 3-bit draft-head readout measured in E6
  (−1.9 % on the one public english fixture) is a real kernel win; flipping the
  in-tree default converts it into scored time, and it survives unseen prompts.
- **Decision:** **green locally.** Candidate is submittable as-is. Ranked
  projection should be discounted — see Conclusion.
- **`BASE_SHA`:** `8970d775a63a28b610fd418c68873c236ce6b86c` (`senpai/qwen38-mtp-r1`,
  carries merged PR #2 ⇒ `segmentedVerifyDepthCap = 7` at
  `Sources/MLXFastModel/Qwen36MTPBlockSession.swift:757`)
- **`UPSTREAM_SHA`:** `7351e62674bc600f0ca148d3a1b0604716a09db6`
- **Candidate commit:** `d320d36` (branch HEAD; the flip itself is `6c2c360`)
- **Yukon promoted frontier used as reference:** submission
  `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, score `2.9042110287045`,
  `sourceRef 7351e626`
- **Submitted candidate files:** **one** —
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` (`draftHeadBits`,
  `else { return 4 }` → `else { return 3 }`)
- **Supporting test, tooling, and documentation files (not submitted):**
  `research/run-draft-bits-arm.sh`, `research/e9_evidence.py`,
  `research/e9_partb.py`, `correctness_prompts/*.json`
- **MTP head provenance and draft policy:** organizer-pinned head throughout —
  `uses_pinned_mtp_head: true`,
  `head_provenance_sha256 = eb481df38267db5c9d9db1f6a813fcc73e762d0af74fdb1bcb061724c815adfe`.
  No `mtp-head.manifest.json` declared, so `mtp-head/` contributes 0 exempt
  bytes. Draft policy unchanged: parent-offered depth 8, session cap 7.
- **Assignment-scope preflight:** `assignment scope OK` — 1 submitted path.
  New goldens live in `correctness_prompts/`, which is **outside**
  `benchmark.json -> editablePaths`, so they cost zero growth.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  `editable budget OK: source=2413591/3000000 bytes headroom=586409
  growth=0/262144 exempt=2410/2147483648 files=154`. The 2410 exempt bytes are
  inherited unchanged from `BASE_SHA`; this candidate declares no
  `mtp-head.manifest.json` and adds no exempt bytes. Growth is exactly zero
  because the flip changes one literal character.
- **Scored-path reachability evidence:** `draftHeadBits` is read inside the
  quantized draft-head readout that `Qwen36MTPBlockSession` invokes every
  drafting round. Reachability is proven *behaviourally*, not by inspection:
  with the flip in place and the override env var unset, the measured
  `accepted_draft_rate` is **bit-identical to 16 digits** to the `env=3` arm and
  **differs** from the `env=4` arm (see "Flip is live" below). A dead code path
  cannot move a 16-digit acceptance figure.

## Evidence

- **Host, memory profile, toolchain, thermal policy:** Apple **M4 Pro**,
  `mem=51539607552`, macOS `26.5.2 (25F84)`. **This is not the ranked M5 host.**
  `cool_gate=disabled_ambient_floor` — the inlet-bound floor sat near 40.7 °C, so
  the 40 °C gate could never arm. **Arms are therefore comparable to each other
  only**, which is why every headline number in this report is a within-session
  A/B on one binary rather than a cross-session absolute.
- **Exact baseline and candidate commands:** driver signature is
  `research/run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA] [MODE]`. The
  literal `default` means "unset the knob", which is exactly what the ranked
  worker sees; any other value is exported as `MLX_QWEN_MTP_DRAFT_BITS`.
  ```bash
  BASE=8970d775a63a28b610fd418c68873c236ce6b86c
  # baseline (cap-7 control, 4-bit forced via the env override)
  research/run-draft-bits-arm.sh 4       e9-cap7-b4             512 "$BASE"
  # candidate (cap-7, in-tree default now 3-bit, override unset)
  research/run-draft-bits-arm.sh default e9-cap7-bdefault       512 "$BASE"
  # Part C submit-mode confirmation
  research/run-draft-bits-arm.sh default e9-cap7-submit-default 512 "$BASE" --local-submit
  ```
  All arms: 512 decode tokens, `dirty=0`. The driver refuses to time a build
  whose worker binary does not contain the `MLX_QWEN_MTP_DRAFT_BITS` symbol, so
  a stale binary cannot silently invalidate an arm.
- **Tests and risk-based checks:** Part C ran `--local-submit` end to end and
  passed (`passed: true`). Its drift tripwire ran against
  `correctness_prompts/public_longcopy_gate_english_512_1024.json` — 1024
  expected tokens over a 512-token prompt, i.e. an M5-generated reference
  checked to `key_len = 1536` — and reported
  `public_drift_tripwire_passed: true`.
- **Exact-token and row-ledger verdict:** **exact on all 16 arms.**
  `all_tokens_matched = true`, `residual_divergence_count = 0` everywhere. Part C
  ledger closes: `declared_rows_total 540`, `emitted_token_total 512`,
  `round_count 79`.
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not relevant — no `.metal` / `.h` / `mlx-generated`
  file was touched.
- **Peak RAM:** `16249552896` B = **15.13 GiB** (Part C). Every arm landed within
  15.10–15.16 GiB; the flip does not change the memory profile.
- **Official status and score:** **not submitted to Yukon.** Part C is
  explicitly `rankable: false`,
  `not_rankable_reason: "candidate-generated reference rows; official scoring
  disabled; ranked run is the only authority"`, `oracle:
  candidate-local-mtp-golden-rows`.

### Headline metric table — cap-7 base, one binary

Baseline = `e9-cap7-b4` (`MLX_QWEN_MTP_DRAFT_BITS=4`).
Candidate = `e9-cap7-bdefault` (override unset; in-tree default is now 3).
Both share `worker_sha256 3a62b25ce75396365342b0bf91bbaa394cfa27f03de1cc838d638b758f81ad49`
and `cli_sha256 f90971df57df19bd4ecc6c76601511e8b7b2759c93b87fba13ead7ca55d243b4`.

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.07345598423853517 | 0.07358258380554616 | +0.172 % (noise) |
| MTP seconds/token | 0.034005630761384964 | 0.03331521479412913 | **−2.0303 %** |
| local serial-relative speedup | 2.160112387091728 | 2.208678054764126 | **+2.2483 %** |
| effective mean draft length | 5.790123456790123 | 5.8354430379746836 | +0.783 % |
| accepted draft rate | 0.9189765458422174 | 0.9392624728850325 | +0.0202859 |

The serial leg is untouched by the flip (serial runs depth 0 and never loads the
draft head). Its 0.172 % spread across the three cap-7 arms is the honest
session noise floor, and the −2.0303 % candidate move is ~12× that.

The local score is a one-prompt directional measurement. It is not the ranked
median across eight hidden prompts.

### Part C — `--local-submit` confirmation (W&B `92n7h4tq`)

`passed: true`, `score` = `mtp_decode_speedup` = **2.2100917410013805**,
`mtp_depth: 8`, head `d320d36`, `dirty=0`, started 23:40:11Z, GPU 44.58 → 57.02 °C.

| Leg | seconds/token | rounds | matched | div |
| --- | ---: | ---: | --- | ---: |
| serial (depth 0) | 0.07355892774648964 | 512 | true | 0 |
| MTP (depth 8, cap 7) | 0.03328320104628801 | 79 | true | 0 |

`accepted_draft_rate = 0.9392624728850325` — **bit-identical to the Part B
candidate arm**. First block 0.17305 s, steady 25.412 ms/token, guardrail
max/p50 1.0125 (serial leg 1.7797). Timing reproduces the Part B candidate
(0.0333152) to **0.096 %** across a different mode *and* a different thermal
slot, which is the strongest stability evidence in this report.

### Full arm table — all 16 arms, full precision

`dep` = parent-offered depth 8 (not the session cap). `match`/`div` are
`all_tokens_matched` / `residual_divergence_count`.

| arm | env | ms/token | accepted_draft_rate | rnds | D | rdouts | ms/round | match | div | worker_sha256 | W&B | base | head |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|
| e6-r1-bits2 | ? | 34.6194 | 0.8902691511387164 | 82 | 5.8902 | 483 | 216.160 | True | 0 | ? | `ue0l9ryy` | b2419f41 | 58e9af2 |
| e6-r1-bits3 | ? | 34.4514 | 0.9094736842105263 | 81 | 5.8642 | 475 | 217.767 | True | 0 | ? | `ey56o2j5` | b2419f41 | 58e9af2 |
| e6-r1-bits4-control | ? | 35.1056 | 0.8902691511387164 | 82 | 5.8902 | 483 | 219.196 | True | 0 | ? | `50ikno4b` | b2419f41 | 26f9d4f |
| e6-r1-bits4-control-b | ? | 35.1193 | 0.8902691511387164 | 82 | 5.8902 | 483 | 219.282 | True | 0 | ? | `hrgew6pe` | b2419f41 | a1081a9 |
| **e9-cap7-b4** | 4 | **34.0056** | 0.9189765458422174 | 81 | 5.7901 | 469 | 214.949 | True | 0 | `3a62b25ce753` | `m4xemao3` | 8970d775 | 36322ce |
| **e9-cap7-bdefault** | unset | **33.3152** | 0.9392624728850325 | 79 | 5.8354 | 461 | 215.916 | True | 0 | `3a62b25ce753` | `dc5kwr7w` | 8970d775 | a9d338c |
| **e9-cap7-submit-default** | unset | **33.2832** | 0.9392624728850325 | 79 | 5.8354 | 461 | 215.709 | True | 0 | `3a62b25ce753` | `92n7h4tq` | 8970d775 | d320d36 |
| e9-flip-b3 | 3 | 34.4224 | 0.9094736842105263 | 81 | 5.8642 | 475 | 217.583 | True | 0 | `b73455224922` | `xextsvtq` | f89b3d60 | 8d979e9 |
| e9-flip-b4 | 4 | 35.1543 | 0.8902691511387164 | 82 | 5.8902 | 483 | 219.500 | True | 0 | `b73455224922` | `dc2yln9r` | f89b3d60 | 8d979e9 |
| e9-flip-bdefault | unset | 34.4325 | 0.9094736842105263 | 81 | 5.8642 | 475 | 217.647 | True | 0 | `b73455224922` | `q36elobx` | f89b3d60 | 8d979e9 |
| e9-narr-b2 | 2 | 52.5561 | 0.4407345575959933 | 249 | 2.4056 | 599 | 108.067 | True | 0 | `369d7adb713b` | `msn7bh1w` | f89b3d60 | 7803f48 |
| e9-narr-b3 | 3 | 52.8506 | 0.4431239388794567 | 251 | 2.3466 | 589 | 107.807 | True | 0 | `369d7adb713b` | `rnk2vqkx` | f89b3d60 | 7803f48 |
| e9-narr-b4 | 4 | 52.8231 | 0.4501718213058419 | 251 | 2.3187 | 582 | 107.751 | True | 0 | `369d7adb713b` | `s55oneg8` | f89b3d60 | 7803f48 |
| e9-tech-b2 | 2 | 42.4325 | 0.6820809248554913 | 158 | 3.2848 | 519 | 137.503 | True | 0 | `369d7adb713b` | `3jiygepd` | f89b3d60 | 46e3d88 |
| e9-tech-b3 | 3 | 41.3435 | 0.7029126213592233 | 150 | 3.4333 | 515 | 141.119 | True | 0 | `369d7adb713b` | `h6aytj05` | f89b3d60 | 46e3d88 |
| e9-tech-b4 | 4 | 41.9817 | 0.6928982725527831 | 151 | 3.4503 | 521 | 142.349 | True | 0 | `369d7adb713b` | `y05wa3s6` | f89b3d60 | 46e3d88 |

W&B project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group
`qwen38-r1-e9-draft-bits-default`; URL form
`https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/<id>`.

### Flip is live — pre-merge bit-exact proof (cap-8 base `f89b3d60`)

The three `e9-flip-*` arms share one binary and differ only in the env override:

| arm | env | accepted_draft_rate | ms/token |
| --- | --- | --- | ---: |
| e9-flip-bdefault | unset | 0.9094736842105263 | 34.4325 |
| e9-flip-b3 | 3 | 0.9094736842105263 | 34.4224 |
| e9-flip-b4 | 4 | 0.8902691511387164 | 35.1543 |

`acceptance(default) == acceptance(env=3)` to 16 digits and `!= acceptance(env=4)`.
The unset-vs-`env=3` timing gap is **0.029 %**, inside the 0.039 % noise floor
for that pair ⇒ **no differential requantization cost** when the value comes
from the in-tree default rather than the env override. This retires the
advisor's "default could be worse than the control" branch and means **Route B
is not forced.**

### Part A — does 3-bit survive unseen prompts? (cap-8 base `f89b3d60`, worker `369d7adb713b`)

Two new goldens were generated for this part. 3-bit vs 4-bit control, same base
and binary within each prompt:

| prompt | ms/token 4-bit | ms/token 3-bit | Δ time | Δ acceptance | D (4-bit) |
| --- | ---: | ---: | ---: | ---: | ---: |
| english (public) | 35.1543 | 34.4224 | **−2.082 %** | +0.019205 | 5.8902 |
| technical (new) | 41.9817 | 41.3435 | **−1.520 %** | +0.010014 | 3.4503 |
| narrative (new) | 52.8231 | 52.8506 | **+0.052 %** | −0.007048 | 2.3187 |

Median across the three prompts = **−1.520 %**. The english figure reproduces
independently across two sessions: the E6 pairing (`ey56o2j5` vs `hrgew6pe`)
gives −1.902 %, so the english effect is −1.9 % to −2.1 %.

**The 3-bit gain tracks D, not a kernel speedup.** Ordering the prompts by
effective mean draft length — 5.89 / 3.45 / 2.32 — reproduces the ordering of
the time delta exactly. A genuine per-readout kernel win would be roughly
constant per readout and would, if anything, help the *low*-D prompts most,
because they issue more readouts per token (599 readouts for 512 narrative
tokens vs 483 for english). It does the opposite.

**Prediction 2 confirmed on both new prompts.** If 3-bit and 2-bit preserved
subgrid rank, their acceptance would be exactly equal. They are not:
technical 0.7029126213592233 vs 0.6820809248554913; narrative
0.4431239388794567 vs 0.4407345575959933. The subgrid-rank-preservation branch
is **dead**.

### Part B — does the cap-7 composition hold? (base `8970d775`)

- Candidate effect on the cap-7 base: **−2.0303 %**.
- Control vs the advisor's independently measured 33.99: **+0.046 %**.
- Candidate vs the advisor's 33.34: **−0.074 %** ⇒ **prediction 4 confirmed.**
  Composition is in fact **0.131 % better** than pure multiplicative against my
  own same-session carry (34.0056 × 0.980983 = 33.3589 predicted vs 33.3152
  measured).
- **Full stack** vs the cap-8 4-bit control: 35.1543 → 33.3152 = **−5.2314 %**.
- **Cap-7's own effect, measured adjacently:** 35.1543 → 34.0056 = **−3.2675 %**.
  Alphonse reported −3.1215 %; this is an independent reproduction of PR #2.
- **Advisor's coupling term confirmed exactly:** D moved
  5.890243902439025 → 5.790123456790123 = **−1.6998 %**, against his predicted
  −1.7 %.

#### Break-even floor — called explicitly

Recomputed on the cap-7 base, because control acceptance moved from 0.8903 to
0.9190 and **any floor calibrated at cap 8 is invalid at cap 7**:

| quantity | value |
| --- | ---: |
| cap-7 3-bit break-even floor | 0.911518 |
| allowed acceptance drop | 0.007459 |
| measured candidate acceptance | 0.939262 |
| margin | **+0.027744** |

**Verdict: PASS**, by 3.7× the allowed drop. Acceptance did not drop at all —
it rose.

#### Honest decomposition — the mechanism did not do the work

Splitting the −2.0303 % into its two multiplicative parts:

| term | value |
| --- | ---: |
| per-round cost (214.949 → 215.916 ms/round) | **+0.450 %** |
| round count (81 → 79) | **−2.469 %** |

The readout share at cap 7 is 2.82 %, so the **mechanism ceiling is −0.683 %**.
The per-round term went the **wrong way**. Part of that is explained: D rose
5.7901 → 5.8354, i.e. +0.78 % more drafts per round, which widens the
ALU-bound verify consistent with Thorfinn's M ≥ 4 result. The residual is
**unclosed** and I am not going to invent a story for it.

So the entire −2.0303 % is a **round-count effect driven by an acceptance
rise**. That rise is deterministic and bit-exactly reproducible on a given
fixture, but it is a quantization coin flip: on narrative it landed the other
way (Δa −0.0070).

### Verdict on the advisor's break-even derivation

**Structurally right, calibrated too lenient.** His cap-8 floors were 0.88224
(3-bit) and 0.87620 (2-bit); mine are 0.88687 and 0.87762. **Both omit the
rollback and re-forward cost of a rejected draft**, and the cap-8 technical
2-bit arm proves it empirically: acceptance 0.6820809248554913 against a floor
of 0.682314 is a margin of **−0.0002**, i.e. nominally break-even, yet the arm
measured **+1.074 % slower** (42.4325 vs 41.9817). A nominally break-even arm
paying a ~1 % penalty is the missing rollback term made visible. Any future
floor should carry it.

### Prediction scorecard

| # | prediction | outcome |
| --- | --- | --- |
| P1 (advisor) | 3-bit is neutral **or better** on unseen prompts | **half right.** Narrative Δa −0.00705 is inside ±0.008, so "neutral" holds on acceptance; "or better" broke on timing (+0.052 %). |
| P2 (advisor) | 3-bit/2-bit acceptance will **not** be exactly equal | **confirmed** on both new prompts ⇒ subgrid-rank branch dead. |
| P3 (advisor) | cap-7 control reproduces | **taken free** — reproduced at −3.2675 % vs Alphonse's −3.1215 %. |
| P4 (advisor) | flip composes ≈ multiplicatively with cap 7 | **confirmed** within −0.074 %. |
| Mine (pre-registered) | unseen prompts come in at "−0.7 % or worse" | **wrong on technical** (−1.520 %, much better), **right on narrative** (+0.052 %). |

### Disclosed hygiene defect

The two cap-7 arms record different `head` (`36322ce` vs `a9d338c`) because I
committed a research script while arm 1 was still timing. Both are `dirty=0`,
both share one `worker_sha256 3a62b25ce753…`, and
`git diff 36322ce a9d338c -- Sources Vendor` is **empty** — binary identity is
the binding proof, and the comparison stands. The lesson was applied to Part C:
`d320d36` was committed *before* launch and the worktree was frozen for the whole
run.

## Conclusion

- **What happened and why:** the flip is live, exact, free in bytes, and worth
  −2.0303 % on the cap-7 base, composing with PR #2 for a −5.2314 % full stack.
  But the decomposition shows the win is **not** the readout kernel. The
  mechanism ceiling is −0.683 % and the per-round term actually regressed
  +0.450 %; the whole gain arrives as a −2.469 % round-count reduction caused by
  acceptance rising 0.0203.
- **Evidence for or against the mechanism:** **against.** The three-prompt Δ
  ordering tracks D (5.89 / 3.45 / 2.32) rather than readout count
  (483 / 521 / 582), which is backwards for a per-readout kernel win. 3-bit
  quantization is perturbing which drafts survive verification, and that
  perturbation is prompt-specific.
- **Prompt or M5 transfer risk:** **high, and this is the main caveat.** Two
  compounding risks: (1) prompt — one of two unseen prompts showed no gain, so
  the ranked median should be projected from **−1.520 %**, not −2.0303 %;
  (2) host — every number here is M4 Pro with the cooling gate ambient-disabled,
  so absolute times do not transfer to the ranked M5 and only the within-session
  A/B structure does.
- **Smallest useful next action:** run the candidate on the ranked M5 against a
  fresh same-host cap-7 control. Everything cheap has been done; the remaining
  uncertainty is host and prompt transfer, which no further M4 work can reduce.
- **Recommendation:** **promote**, with the ranked projection discounted to
  ~−1.5 %. The candidate is one character in one submitted file, costs zero
  bytes of growth, is exact on 16/16 arms, passes `--local-submit` with the
  1024-step drift tripwire green, and reproduces to 0.096 % across modes. The
  downside is bounded by the narrative arm at +0.052 %, which is within noise —
  so the realistic worst case is "no change", not a regression.

## Suggested follow-ups (not implemented)

1. **Route B — manifest-provisioned pre-quantized head.** Now optional rather
   than forced, since Part B proved there is no differential requantization
   cost. It would require lifting the `_draftHeadW == nil` guards at
   `Qwen35.swift:2364` and `:2403`. Worth it only if a *genuine* readout win is
   the target; this experiment says the readout is 2.82 % of the round and the
   ceiling is −0.683 %.
2. **Close the +0.450 % per-round residual.** D rising 0.78 % explains part of
   it. The rest is unexplained and sits directly on the critical path.
3. **More unseen prompts.** Three prompts give a very soft median. Five or six
   would materially firm up the ranked projection, and this experiment now has
   a reusable golden-generation path to do it cheaply.
4. **PR #7 §9 dispatch-grid puzzle** — untouched here, still open.
5. **Re-derive the break-even floor with the rollback term.** The technical
   2-bit arm gives a calibration point: margin −0.0002 ⇒ +1.074 % measured.
