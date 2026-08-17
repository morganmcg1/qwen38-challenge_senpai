SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"g_median_held_out_pct","available":true,"value":5.284272658432601},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

# E17 — does the depth curve's advantage survive the ranked aggregation?

- Student / branch: `qwen-edward` / `qwen-edward/curve-transfer-and-refit` (PR #19, assignment `qwen38-r1-e17-curve-transfer-and-refit` r1)
- Hypothesis and target cost: the merged `headStepCostRatioByDepth` curve beat the retired `headStepCostRatio = 0.18` scalar by `g = 6.378 %` of the MTP leg **on one prompt**. Q1 asks whether that advantage survives `median` over prompts of varied register, or whether it is a best-prompt artifact like E9 r2's sign flip (english −1.902 %/−2.082 %, technical −1.520 %, narrative +0.052 %, median −1.520 %).
- **Decision: green locally, with a hard caveat — measured on the wrong base for a headline, and at n=4 rather than the contracted n≥6.**
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: `e6e6f81767e84cc8c39b48c09a4f5cac597cdbca` / `32b94cb67d2f3a102a36382d2beb62eee8d99db5` / this branch head
- Yukon promoted submission / source ref used as frontier: `03dedda8-fc70-4e3e-881f-5384a17af405`, `sourceRef 32b94cb6`, score `2.94661597308114`
- Submitted candidate files: **none.** The `CURVE` arm is `HEAD` unmodified; the `FLAT18` arm is materialised transiently by `research/e17-build.sh`, built, and reverted by an `EXIT` trap. `git diff --name-only e6e6f81 HEAD` touches zero paths under `Sources/`, `benchmark.json`, `fixtures/`, `mtp-head/`, `tools/`, `Package*`.
- Supporting test, tooling, or documentation files: `research/e17-build.sh`, `research/e17-run.sh`, `research/e17_analyse.py`, `research/e17_wandb.py`, `research/e17_gate_sim.py`, `research/e17_token_check.py`, `research/e17-notes.md`, `research/e17-q4-crossed-design.md`, seven `research/e17_prose_*_512.txt` prompt texts, plus the dated retraction block in `research/e11-notes.md`.
- MTP head provenance and draft policy: pinned head only, `sha256 07293af742df4599d94eda6e9db5782e7f5be10cd1b5fdef7691f4ef404ea81c`, `hf:dwsdubey/qwen3.8-27b-mtp-4bit@34ee76f6`, `238937699` bytes, identical on all 8 timed arms. No `MLXFAST_QWEN_MTP_HEAD_DIR` override on any arm. Draft policy `--mtp-depth 8`, serial control `depth 0`, adaptive gate chooses per round.
- Assignment-scope preflight: `./senpai/validate-assignment-scope.sh e6e6f81767… Sources/MLXFastModel/Qwen36MTPBlockSession.swift` → `assignment scope OK: 1 submitted path(s)`.
- Editable source bytes / headroom / growth / exempt-head bytes: `source=2405727/3000000 headroom=594273 growth=0/262144 exempt=2410/2147483648 files=154`.
- Scored-path reachability evidence: both arms differ only in the `headStepCostRatioByDepth` literal inside `Qwen36MTPBlockSession.swift`, which the gate consumes on every round of the scored session. The two arms produce **different worker binaries** (`1651e64e…` vs `bb7db942…`) and **different depth histograms** on every prompt, so the edited constant provably reaches the timed path.

## Q1 — headline

Per-prompt pairs, `raw_p = serial_spt / mtp_spt` straight from `.parent_measured_seconds_per_token`, **prefill-inclusive, 512 decode tokens after a 512-token seed**, `--local-iterate`:

| prompt | held out | serial (C leg) | serial (F leg) | floor % | mtp CURVE | mtp FLAT18 | raw CURVE | raw FLAT18 | Δraw | g % |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `english` (anchor, in-sample) | no | 0.074378 | 0.074425 | 0.063 | 0.046269 | 0.049585 | 1.607516 | 1.500951 | +0.106565 | **+6.688** |
| `narrative` | yes | 0.074399 | 0.074464 | 0.087 | 0.045837 | 0.048394 | 1.623115 | 1.538689 | +0.084426 | **+5.284** |
| `technical` | yes | 0.074456 | 0.074553 | 0.130 | 0.044686 | 0.046511 | 1.666207 | 1.602927 | +0.063280 | **+3.923** |
| `dramatic` | yes | 0.074301 | 0.074554 | 0.340 | 0.043463 | 0.047777 | 1.709516 | 1.560481 | +0.149034 | **+9.028** |

`g` is the MTP-leg improvement implied by the pair, `g = 1 − mtp_CURVE / mtp_FLAT18`. `floor %` is that prompt's noise floor from the byte-identical serial leg run twice.

**Held-out population (n=3: narrative, technical, dramatic).** Odd n, so the median is the single central order statistic:

- `median(raw_p | CURVE)` = median(1.623115, 1.666207, 1.709516) = **1.666207**
- `median(raw_p | FLAT18)` = median(1.538689, 1.602927, 1.560481) = **1.560481**
- headline delta = **+0.105726** = **+6.775 %** of the FLAT18 median
- **`g_median` = +5.284 %**, spread **+3.923 … +9.028 %**, curve wins **3/3**

**All-4 population, including the in-sample anchor.** Even n, so the even-n rule applies — mean of the two central order statistics, computed **per arm independently**:

- CURVE sorted: 1.607516, **1.623115**, **1.666207**, 1.709516 → median = (1.623115 + 1.666207)/2 = **1.644661** (driven by `narrative` + `technical`)
- FLAT18 sorted: 1.500951, **1.538689**, **1.560481**, 1.602927 → median = (1.538689 + 1.560481)/2 = **1.549585** (driven by `narrative` + `dramatic`)
- headline delta = **+0.095076** = **+6.136 %**
- `g_median` = mean of the two central `g` values = (5.2843 + 6.6880)/2 = **+5.986 %**, curve wins **4/4**

The two central-order-statistic sets differ between arms (`dramatic` is central for FLAT18 but extreme for CURVE), which is exactly why the ratio of medians (+6.136 %) is not the median of the ratios (+5.986 %). Both are reported; neither is cherry-picked.

**I report the held-out `+5.284 %` as the headline.** The anchor prompt turns out to be the *second strongest* of the four, so removing it **lowers** `g_median` from 5.986 % to 5.284 %. The number I am quoting is the more conservative one.

### Stop rule

> *"If `g_median` over >= 6 prompts is **>= 4 %**, the lever is real at scale."*

`g_median` = **+5.284 %** held-out, and **every individual prompt** clears 4 % except `technical` at +3.923 % — which is itself 30× its own noise floor. The ≥4 % branch is met, **at n=3 held-out / n=4 total rather than the contracted n≥6.** Section "Why I stopped at four prompts" states why, and the number is not robust to the missing prompts in the way an n≥6 median would be: with n=3 the median *is* a single observation, and one prompt at, say, +1 % would drag the held-out median to +3.9 %. I am not claiming the n≥6 result. I am claiming that four independent registers all landed between +3.9 % and +9.0 % with 4/4 wins, which makes advisor prediction 1 (`1.5–4 %`) and prediction 2 (`≥1 prompt shows no benefit or a slight regression`) both false on this evidence.

### Convention and window, stated for every ratio

Every `raw_p`, every median, and every `g` above is **prefill-inclusive** and measured over **512 decode tokens** after a 512-token seed, taken directly from `.parent_measured_seconds_per_token` with **no subtraction**, per the section-0 correction. Prefill is ≈3.9944–3.9980 s per leg ≈ 15.8–18.0 % of the MTP leg. The decode-currency→score conversion factor on this base is **0.84228**: a decode-leg `g` becomes a score-leg effect of `g × 0.84228`. Sanity check against the E11 pair: 7.577 % × 0.84228 = 6.382 % vs the 6.378 % measured prefill-inclusive delta ✓. **No number in this report is quoted in the decode-only convention.**

## Mechanism — the main scientific result of the session

The scalar accepts **more** draft tokens on every single prompt and is **slower** on every single prompt.

| prompt | g % | extra verify rows F−C | extra rows % | extra accepts F−C | extra rounds F−C | rows/accept C | rows/accept F | ratio |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `technical` | +3.923 | 56 | +7.80 | +11 | −11 | 2.592 | 2.688 | 1.037 |
| `narrative` | +5.284 | 58 | +7.86 | +3 | −3 | 2.733 | 2.916 | 1.067 |
| `english` | +6.688 | 82 | +11.04 | +1 | −1 | 2.793 | 3.090 | 1.106 |
| `dramatic` | +9.028 | 86 | +11.78 | +15 | −16 | 2.635 | 2.795 | 1.060 |

Sorted by `g`, **extra verify rows is monotone**: `Spearman(g, extra rows %) = +1.000` (n=4) against `+0.400` for rows-per-accept. **The curve wins by declining redundant target verification**, trading cheap extra rounds for expensive saved rows at `M = 4, 5`. The scalar reaches `d5` on `narrative` (1 round) and `dramatic` (9 rounds) and `d4` on all four; the curve's `max_d` is **3** on all four prompts.

This is the same "wins by going shallower" mechanism banked from r1, now confirmed to be **register-independent in sign** across four registers and to have a *quantitative* predictor. It has a direct corollary for Q4: the observation that "the depth-4 gate is closed by only 6.9 %" describes the curve's **most valuable behaviour**, not a defect to be fixed. Opening depth 4 is a win only if the `M=5` cell becomes cheap enough that the deeper round repays the extra rows — and the extra rows are precisely what the curve is currently saving.

## Evidence

- Host, memory profile, toolchain, and thermal policy: `Mac16,11` Apple **M4 Pro**, 14 cores, 48 GiB, macOS 26.5.2, `swift-driver 1.148.6 / Apple Swift 6.3.3 (swiftlang-6.3.3.1.3)`, target `arm64-apple-macosx26.0`. **Not the ranked M5** — all transfer caveats below apply. Thermal policy: every arm serialised through `research/await-lock-then-run.sh`, wrapper 40 °C cool gate before each resident measurement, `mlxfast` orphan check, GPU temperature recorded before and after every arm.
- Exact baseline and candidate commands: `research/e17-build.sh CURVE FLAT18` then, per prompt, `research/e17-run.sh <prompt>` which serialises `await-lock-then-run.sh ./benchmark-qwen-mtp.sh --local-iterate` with `--generate 513 --mtp-depth 8` for the MTP leg and `--mtp-depth 0` for the serial control, and records `meta.txt` including `dirty=$(git status --porcelain | wc -l)`.
- Tests and risk-based checks: no `Sources/` change is proposed, so no new Swift test is warranted. Risk-based checks actually run: `research/e17-build.sh`'s source asserts (vector declaration present, scalar `h` absent, shipped `h[0..3]` literal present for CURVE / flat 0.18 installed twice for FLAT18, and `sdpaWidthWallDepthCap = 5`, `segmentedVerifyDepthCap = 8`, `segmentedStreakGate = 3` unmoved on both arms so a silent base change cannot be read as an `h` effect); a 64-step public-drift tripwire per prompt before timing; and `research/e17_analyse.py --contract`.
- Exact-token and row-ledger verdict: **clean on all 8 arms.** `all_tokens_matched=True`, `residual_divergence_count=0`, `parity_all_ok=True`, 512/512 emitted vs golden steps, `declared_rows == reference_checked_rows` on every arm.
- Divergent tokens or failure category, if any: **none.**
- Generated-twin audit, if relevant: not relevant — no `.metal`/`.h` or `mlx-generated/*.cpp` file was touched.
- Peak RAM or head/artifact size, if relevant: pinned head `238937699` bytes, well inside the 2 GiB exempt cap; `exempt=2410` bytes of in-repo declaration.
- Official status and score, if submitted: **not submitted.** See "Why no headline is proposed from this base".

### Contract item 1 — provenance per prompt, and GPU temperature per arm

| prompt | prompt text | text sha256 | golden sha256 (shared by both arms) | order | gpu °C before → after |
|:--|:--|:--|:--|:--|:--|
| `english` | `research/e11_prose_gate_english_512.txt` | `a75aece96c68a866…548a00c` | `615a1f20cae333fd…5ceb219` | C then F | C 37.86→60.64, F 57.73→59.94 |
| `narrative` | `research/e17_prose_narrative_512.txt` | `651d635ada51fc56…d81cbb` | `f0873582e5de5e21…c945849ac3f904` | F then C | F 41.68→60.17, C 57.32→60.01 |
| `technical` | `research/e17_prose_technical_512.txt` | `c135cec61e796f2c…f990ba774ee6` | `28447e11a48c37f1…3bd5023ce3e5` | C then F | C 41.53→59.45, F 57.04→60.83 |
| `dramatic` | `research/e17_prose_dramatic_512.txt` | `58d2c351bf85356d…3e039174ffa8e` | `e5bd3711ca4fbf42…bbe36bc4702c5` | F then C | F 39.09→60.87, C 57.83→59.97 |

Arms are interleaved **within** prompt as required, in a balanced **ABBA/ABBA** order (`C F | F C | C F | F C`), so each arm ran first twice and second twice and thermal drift cannot correlate with arm. The second arm of a block always starts hot (~57 °C) because the wrapper's 40 °C gate is satisfied before the block, not between arms. That position effect is real and is **balanced by design**, and it happens to work against the conclusion: CURVE ran *second* on `narrative` and `dramatic` (mean g **+7.16 pp**) and *first* on `english` and `technical` (mean g **+5.31 pp**), so the curve won by more when it carried the hot-start handicap. Even the handicapped-position pair mean clears the 4 % stop rule.

Per-prompt noise floors are 0.063 %, 0.087 %, 0.130 %, 0.340 % of the serial leg. The smallest `g` in the table (+3.923 %) is **30×** its prompt's floor; the largest floor (`dramatic`, 0.340 %) is **26×** smaller than that prompt's `g` (+9.028 %). No pair exceeded a clean band, so no pair is reported as a range.

### Contract item 4 — depth, acceptance, replay, rejects per arm per prompt

| prompt | arm | rounds | mean d | max d | depth histogram | accepted/proposed | acc rate | replays | rows |
|:--|:--|--:|--:|--:|:--|--:|--:|--:|--:|
| `english` | CURVE | 246 | 2.020 | 3 | `{1:2, 2:237, 3:7}` | 266/497 | 53.5 % | 74 | 743 |
| `english` | FLAT18 | 245 | 2.367 | 4 | `{1:19, 2:138, 3:67, 4:21}` | 267/580 | 46.0 % | 110 | 825 |
| `narrative` | CURVE | 243 | 2.037 | 3 | `{1:1, 2:232, 3:10}` | 270/495 | 54.5 % | 75 | 738 |
| `narrative` | FLAT18 | 240 | 2.317 | 5 | `{1:20, 2:135, 3:75, 4:9, 5:1}` | 273/556 | 49.1 % | 90 | 796 |
| `technical` | CURVE | 235 | 2.055 | 3 | `{1:1, 2:220, 3:14}` | 277/483 | 57.3 % | 70 | 718 |
| `technical` | FLAT18 | 224 | 2.455 | 4 | `{1:10, 2:117, 3:82, 4:15}` | 288/550 | 52.4 % | 88 | 774 |
| `dramatic` | CURVE | 236 | 2.093 | 3 | `{1:1, 2:212, 3:23}` | 277/494 | 56.1 % | 74 | 730 |
| `dramatic` | FLAT18 | 220 | 2.709 | 5 | `{1:6, 2:97, 3:81, 4:27, 5:9}` | 292/596 | 49.0 % | 97 | 816 |

### Contract item 5 — correctness / hygiene, every timed arm

| arm | matched | div | parity | emitted/golden | declared=checked rows | `mlx_qwen_env` | `dirty` | pinned head | drift tripwire | stall max/p50 |
|:--|:--|--:|:--|:--|:--|:--|--:|:--|:--|--:|
| `english-CURVE` | True | 0 | True | 512/512 | 743=743 | *(empty)* | 0 | yes | True (64 steps) | 1.545× |
| `english-FLAT18` | True | 0 | True | 512/512 | 825=825 | *(empty)* | 0 | yes | True (64 steps) | 1.528× |
| `narrative-CURVE` | True | 0 | True | 512/512 | 738=738 | *(empty)* | 0 | yes | True (64 steps) | 1.547× |
| `narrative-FLAT18` | True | 0 | True | 512/512 | 796=796 | *(empty)* | 0 | yes | True (64 steps) | 1.738× |
| `technical-CURVE` | True | 0 | True | 512/512 | 718=718 | *(empty)* | 0 | yes | True (64 steps) | 1.552× |
| `technical-FLAT18` | True | 0 | True | 512/512 | 774=774 | *(empty)* | 0 | yes | True (64 steps) | 1.718× |
| `dramatic-CURVE` | True | 0 | True | 512/512 | 730=730 | *(empty)* | 0 | yes | True (64 steps) | 1.227× |
| `dramatic-FLAT18` | True | 0 | True | 512/512 | 816=816 | *(empty)* | 0 | yes | True (64 steps) | 1.516× |

`ALL ARMS CLEAN: True`. Stall ratios 1.227–1.738× are all well inside the 4× guardrail. `max_rejected_tail_logit_delta = 0` read directly from `reports/04-mtp-timed.json` on all 8 arms, alongside a non-zero `rejected_rows_reference_checked` on each (CURVE 231/225/206/217, FLAT18 313/283/262/304 for english/narrative/technical/dramatic), so the zero means "every rejected row was checked and none diverged", not "no rejected rows were checked". No timed arm set any `MLX_QWEN_MTP_*` variable, so **no timed arm is a traced run**.

### Contract item 6 — binary freshness, and proof the worker is not a stale twin

| arm | CLI sha256 | worker sha256 | patched-source sha256 |
|:--|:--|:--|:--|
| all CURVE arms | `e1d9980b04e45b68…` | `1651e64e29766973…` | `93d115bf78ba5b6f…` |
| all FLAT18 arms | `e1d9980b04e45b68…` | `bb7db942dbdcf1d9…` | `394291dfa57ba1c7…` |

One distinct worker binary per arm, **the two arms use distinct workers**, and the CLI is identical (the constant lives in the worker, not the CLI — this is the expected pattern, and a *shared* worker hash would have been the stale-twin failure). The source hash differs exactly as the two literals differ. Staleness is therefore excluded by construction rather than by timestamp.

### Contract item 7 — serialisation

Every one of the 16 timed legs went through `research/await-lock-then-run.sh`. One model-holding process at a time; wrapper lock, orphan check and cool gate never bypassed.

### Contract items 9, 10, 11, 12 — paper deliverables

All in `research/e17-notes.md` (§4, §5, §6) and `research/e17-q4-crossed-design.md`, committed on this branch. Summaries:

**Q2 — non-monotonicity diagnosed, gate algebra written down, refit produced.**

The gate algebra, previously held only in my head:

> extend `d → d+1` iff `reach(d) > h[d] · (1 + E_d) / (1 + H_d)`, where `E_d = Σ_{j<d} Π_{k≤j} p_k` and `H_d = Σ_{j<d} h[j]`.

With a *flat* vector `H_d = d·h`, so a flat-0.18 vector reproduces the retired scalar rule `h(1+E)/(1+d·h)` term for term. That identity is why `FLAT18` — not a partial revert — is the correct counterfactual, and it is what `research/e17-build.sh` implements.

Diagnosis: **identification failure at low depth**, confirming advisor prediction 3. `h0` and `h1` are estimated from steps that are near-universal in unforced runs, so depth variation barely identifies them. The shipped `h[0] = 0.0842` is **−13.3 %** against the measured 0.0971 and `h[1] = 0.0775` is **−32.7 %** against the measured 0.1152. Monotone refit: `[0.0971, 0.1152, 0.2482, 0.3761, 0.3761, 0.3761, 0.3761, 0.3909]`.

Pre-timing depth predictions for every candidate vector (contract item 9's "before you time anything"): shipped opens to **d2** (slack −0.0053); refit opens to **d3** (+0.0052); flat 0.18 → **d4**; flat 0.20 → **d3**.

**The conclusion is uncomfortable and worth stating plainly: the fitting bug is accidental extra conservatism at depth 2, and that conservatism is what wins.** The raw monotone refit is predicted at ≈**−0.12 %**, i.e. neutral-to-worse than the shipped curve, because making the coefficients *more accurate* makes the scheduler *less* conservative exactly where conservatism pays. Advisor prediction 4 said a refit would go shallower and move mass `M=4 → M=3` without opening `M=5`; that is **refuted in its specifics** — the refit opens `d3`, it does not close `d2`. So I am **not** proposing the refit as a candidate. The correct reading is that `h` is not a cost estimate to be improved but a *risk price* to be tuned, and the shipped values happen to price depth-2 risk well.

Pricing test (`e17_gate_sim.py --costcheck`), a real out-of-sample check of the algebra: measured `Hp3 → Sp3` = **+7.577 %**; refit predicts **+7.651 %**, shipped **+8.021 %**, flat 0.20 **+4.321 %**, flat 0.18 **+3.990 %**. The refit vector predicts the measured pricing effect to within 0.07 pp.

**Q3 — the 0.18-vs-0.20 null is a real gate flip, predicted analytically before comparison.** Measured **−0.782 %** (0.20 faster) against a predicted **−0.738 … −1.256 %**. That is inside a factor of 2, so **advisor prediction 5 is confirmed**. Consequence, stated against my own interest: the pinned 0.18 scalar is ~0.8 % off *its own* optimum, so the curve's honest advantage over a **well-chosen** scalar is roughly `g_median − 0.8 pp ≈ +4.5 %` held-out, not +5.28 %. It still clears the 4 % stop rule, but only just, and future scalar controls should be **0.20**.

**Q4 — crossed design on paper** in `research/e17-q4-crossed-design.md`, `{shipped curve, refit curve} × {shipped IPG, single-pass M=5}`, with the informative/redundant cells and the counter-argument. The mechanism table above strengthens that counter-argument considerably: the curve's win is *monotone* in verify rows avoided, so the `M=5` cell has to be cheap enough to repay rows the curve is currently declining. **Do not assume the sign.**

**Two arithmetic corrections to the brief**, in `research/e17-notes.md` §6, both verified from source.

**Contract item 12 — the retraction is done.** `research/e11-notes.md` line 788 carries `### RETRACTED 2026-08-17 …` with the wrong claim, the source chain that refutes it, and the corrected two-convention table; the verbatim r3 text is retained at line 883 so the record shows what was said, not just that it was withdrawn. Committed in `82ce6ce`. I accept the section-0 correction in full: *"the scoring expression has no X operand"* does not show X is excluded, because the operands themselves may already contain X. My decode-only `g = 6.378 %` was correct arithmetic about the wrong quantity.

### Contract item 13 — W&B runs

Group `qwen38-r1-e17-curve-transfer-and-refit`, project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`. One run per `(prompt, arm)` carrying the full per-arm config (head sha, worker sha, source sha, golden sha, base sha, dirty count, arm position, thermals) and metrics (both spt legs, `raw_p`, rounds, mean/max depth, per-depth histogram, accepts, rejects, replays, rows, correctness booleans, prefill share), plus one analysis run carrying both medians:

| run | id | state | URL |
|:--|:--|:--|:--|
| `e17-english-CURVE` | `js5je0nm` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/js5je0nm |
| `e17-english-FLAT18` | `wuggzjeo` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/wuggzjeo |
| `e17-narrative-CURVE` | `s1aug405` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/s1aug405 |
| `e17-narrative-FLAT18` | `i81wn9dz` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/i81wn9dz |
| `e17-technical-CURVE` | `b2u43595` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/b2u43595 |
| `e17-technical-FLAT18` | `atrpydja` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/atrpydja |
| `e17-dramatic-CURVE` | `uc5jlr8d` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/uc5jlr8d |
| `e17-dramatic-FLAT18` | `8ocnciqv` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/8ocnciqv |
| `e17-headline` (analysis) | `s9oi4jrb` | finished | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/s9oi4jrb |

`e17-headline` also uploads the per-prompt pair table as an artifact and logs the per-prompt noise floors.

### Contract item 14 — signed falsification statement, and order of information

The statement was signed **before** any timing, as `research/e17-notes.md` §1.6, committed in `82ce6ce`. In summary, I said I would conclude the curve does **not** transfer if any of: `g_median` < 1 %; either sign across prompts; the median driven by a prompt whose pair exceeds its noise floor; or a `g` smaller than 3× that prompt's floor on the median prompt. None occurred.

**Order-of-information disclosure, as requested.** §1.6 and the fixed prompt set, order and generation procedure (§1.2) were committed in `82ce6ce`; the first timed arm started **11:19:41Z**; the quarantined §7 arrived in the advisor comment at **12:27:41Z**, by which point **3 of 4 pairs were already on disk**. So: **yes, the pre-timing statement was already signed when I read §7.** It was unavoidable that §7 arrived inside the same `get_prs` payload as the administrative corrections I did need. Nothing in the prompt set was added, dropped, reordered or re-run after reading it; the fourth pair (`dramatic`) was already running under the pre-committed order.

On the substance of §7: alphonse's E16 claim that the curve **costs** ~1 % of the MTP leg on this same base is **not reproduced here** — I measure the curve *gaining* 3.9–9.0 % on four prompts at 512 tokens. The most likely explanation is a window-length artifact: at 64 tokens the leg is prefill-dominated and 10–16 rounds is far too few for an adaptive gate to express itself, and his own caveats are n=1, 64 tokens, M4 Pro. Folding this in at n=3 and 512 tokens is a named follow-up; attribution for the observation to alphonse. PR #18 is outside this launch's isolation scope so I did not inspect it — I treated the summary as advisor-relayed context only.

## Why no headline is proposed from this base

Per the advisor's §2/§3: the live advisor base is `b85e782`, which moved `segmentedStreakGate 3 → 2` and `qmv_fast_crossrow_affine4_g64_m<T,8,4> → <T,8,3>` **together on purpose**, and those are exactly the two levers my model is about. So:

- **This is a within-session contrast on a single fixed base, `e6e6f81`, and I report it as such.** The blinding held, the arms differ in one literal, and the pairing is clean. As a *mechanism* result it stands.
- **I am not proposing any `R'` or submission number.** Anything headline must be measured on `b85e782`. My `M≥5 = 0.00 %` figure and the `{1:2, 2:231, 3:13}` histogram were both taken at gate 3 and now **understate** firing; the `R'` projection table is stale for the same reason.

### What is salvageable, and what a re-measure costs

**Base-independent (no re-measure needed):** the whole gate algebra and its derivation; the non-monotonicity diagnosis; the monotone refit vector and its pre-timing depth predictions; the pricing test; the 0.18-vs-0.20 gate-flip calculation and its confirmation; both arithmetic corrections; the prefill-dilution factor derivation (the *value* 0.84228 is base-specific, the method is not); the mechanism finding that the curve wins by declining verify rows; and the entire instrument — prompt set, texts, goldens, `e17-build.sh`, `e17-run.sh`, `e17_analyse.py`, `e17_wandb.py`, the ABBA design and the noise-floor discipline. The instrument is the expensive part and it ports unchanged.

**Needs re-measure on `b85e782`:** every number in the per-prompt table and both medians; anything about depth ≥4, `M≥5` or `M=8`; and the `R'` projection.

**Wall-time cost, measured from this session's own timestamps** (2 arms + 2 cool gates per prompt, one prompt per `run_job`):

| prompt | span | minutes |
|:--|--:|--:|
| `english` | 11:19:41 → 11:36:02 | 16.35 |
| `narrative` | 11:37:10 → 11:54:06 | 16.93 |
| `technical` | 11:55:19 → 12:13:44 | 18.42 |
| `dramatic` | 12:18:32 → 12:37:35 | 19.05 |

Mean **17.69 min/prompt**, max **19.05**. So: **6 prompts ≈ 106–114 min; 8 prompts ≈ 142–152 min**; plus **~10–14 min** to build both arms on the new base. **An 8-prompt re-measure on `b85e782` costs ≈152–166 min of exclusive GPU time; a 6-prompt one ≈116–128 min.** `timeout_seconds=1740` per prompt is sufficient with margin. Four prompt texts (`medicine`, `philosophy`, `travel`, `natural_history`) are already committed and unused, so extending to 7–8 prompts costs only machine time, not new authoring.

**I am asking for a formal revision bound to `b85e782` with that budget, rather than shortening the window.** Please also decide whether the re-measure should keep `FLAT18` as the counterfactual or switch to `FLAT20`, given Q3.

## Why I stopped at four prompts

The advisor asked (§5) that I release the GPU to askeladd at my next timed-block boundary, and asked (§3) what a re-measure costs so a formal revision bound to `b85e782` could be issued. The `dramatic` block finished at 12:37:35Z; I verified no `mlxfast` process remained and released the GPU there.

The reasoning, recorded in `research/e17-notes.md` §7.5 at the time: continuing to prompts 5–8 would spend a further ~71–76 GPU-minutes on a base that **cannot carry a headline anyway**, while the ≥4 % stop rule was already met decisively and `dramatic` had just removed the drift confound by winning largest from the hot position. Those same minutes are worth strictly more on `b85e782`. This is a real deviation from the contracted n≥6 and I am flagging it as such rather than presenting n=4 as the plan.

**Disclosure about §5:** I was asked to *"post a one-line note and release the GPU."* I have **no `post_assignment_comment` tool in this session's schema** — my only GitHub write path is the terminal result. I released the GPU on time at 12:37:35Z but **could not post the note**, so askeladd had no signal from me. If GPU handoffs are going to be coordinated by PR comment, the student role needs that tool provisioned; otherwise handoff notes should ride on something both roles can see.

## Metric table

Baseline = `FLAT18` (the retired scalar the curve replaced); candidate = `CURVE` (merged per-depth vector). Both at the **held-out median prompt**, prefill-inclusive, 512 decode tokens.

| Metric | Baseline (FLAT18) | Candidate (CURVE) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.074553 | 0.074456 | −0.130 % (noise floor) |
| MTP seconds/token | 0.046511 | 0.044686 | **−3.923 %** |
| local serial-relative speedup (`raw_p`) | 1.602927 | 1.666207 | **+0.063280** |
| `median(raw_p)` over held-out prompts (n=3) | 1.560481 | 1.666207 | **+0.105726 (+6.775 %)** |
| `g_median` (MTP-leg improvement) | — | — | **+5.284 %** |
| effective mean draft length | 2.455 | 2.055 | −0.400 (curve is **shallower**) |
| accepted draft rate | 52.4 % | 57.3 % | +4.9 pp |
| verify rows evaluated | 774 | 718 | **−56 (−7.24 %)** |

The local score is a directional measurement on four public prompts on M4 Pro. **It is not the ranked median across eight hidden prompts on M5**, and it is not measured on the current advisor base.

## Conclusion

- **What happened and why:** the curve's advantage **does** survive median-over-prompts on four registers — `g_median` **+5.284 %** held-out (+5.986 % all-4), spread +3.923…+9.028 %, 4/4 wins, every prompt ≥26× its own noise floor. The E9-style sign flip did not happen. Advisor predictions 1, 2 and 4 are falsified; 3, 5 and 6 are confirmed.
- **Evidence for or against the mechanism:** strongly for, and now with a quantitative handle. `Spearman(g, extra verify rows %) = +1.000`. The scalar accepts *more* tokens on every prompt and is *slower* on every prompt; the curve's win is bought by **declining redundant target verification** at `M = 4, 5`, not by better acceptance. Against my own case: the pinned 0.18 control is ~0.8 pp worse than the 0.20 scalar, so the honest advantage over a well-chosen scalar is ≈**+4.5 %**; and the monotone refit — the "more accurate" vector — is predicted **neutral-to-worse**, which means the shipped curve's edge is accidental risk-pricing at depth 2, not superior cost modelling. That is a fragile foundation and should be treated as such.
- **Prompt or M5 transfer risk:** prompt risk is now **materially lower** — four registers, same sign, 4/4. Remaining risks, in order: (1) **base risk is the largest** — `b85e782` moved the streak gate and the `M=8` kernel together, and both are inside this mechanism; (2) **M5 risk is untested** — everything here is M4 Pro, and the mechanism is a *cost ratio* between head steps and target verify widths, which is exactly the kind of quantity a different GPU can re-order; (3) **n risk** — n=3 held-out means the median is one observation; (4) the `--local-iterate` goldens are self-generated, so token matching here does not prove a match against the hidden reference.
- **Smallest useful next action:** re-run this exact instrument on `b85e782` for 6–8 prompts (≈116–166 GPU-min, budget above), with `FLAT20` rather than `FLAT18` as the counterfactual, and log the depth histograms at gate 2 so the `M≥5`/`M=8` figures stop being stale. That single run replaces every stale number in this report and produces the first submission-eligible median in the campaign.
- **Recommendation: repeat on the current base.** The curve stays merged — it is an improvement on four registers by a mechanism I can now name and predict. But **no submission number should be built on `e6e6f81`**, and I am not offering one. Please issue the revision bound to `b85e782`.

### Suggested follow-ups I did **not** implement

1. **Fold in alphonse's E16 observation at n=3 / 512 tokens** on `b85e782`. My four-prompt result contradicts it at 512 tokens; the disagreement is worth resolving properly rather than by assertion, and my window-length hypothesis is testable by re-running his 64-token configuration alongside a 512-token one on the same host.
2. **`FLAT20` as the standing scalar control.** Q3 settles that 0.20 beats 0.18; every future curve-vs-scalar contrast should use it, which will shrink reported margins by ~0.8 pp and make them honest.
3. **Price depth-2 risk directly.** Since the shipped curve wins by accidental conservatism at `d2`, the obvious experiment is a *deliberate* one-parameter sweep of `h[1]` alone with everything else shipped, to find whether the accident is at its optimum. That is a cheap, well-identified, one-lever study and it is the natural successor to E17.
4. **`research/ESTABLISHED_FACTS.md:1675` and `research/CURRENT_RESEARCH_STATE.md:83,56`** are stale on the `3.0 → 5.0` ceiling. The advisor claimed those fixes; I am **not** editing them and am not silently assuming they are done.
