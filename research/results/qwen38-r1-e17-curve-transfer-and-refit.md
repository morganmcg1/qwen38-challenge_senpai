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

Per-prompt pairs, `raw_p = serial_spt / mtp_spt` straight from `.parent_measured_seconds_per_token`, **decode-only — which is the scored currency — over 512 decode tokens after a 512-token seed**, `--local-iterate`. *(r2 correction: r1 originally labelled this field prefill-inclusive. The label was wrong; the numbers below are unchanged and were always in score currency. See "Convention and window" below.)*

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

**Corrected in r2 — read this instead of r1's original wording.** Every `raw_p`, every median, and every `g` above is **decode-only**, which `score.json` proves is exactly the scored currency, measured over **512 decode tokens** after a 512-token seed, taken directly from `.parent_measured_seconds_per_token` with **no subtraction and no addition**. So **every number in this report is already in score currency** and the `× 0.84228` factor must **not** be applied to any of them.

What r1 got wrong was only the *label*, not any measured value. r1 called the field prefill-inclusive and derived a "decode-currency → score conversion factor" of `0.84228`. That quantity is real but is the **decode share of the wall-clock leg** (`D_m / T_m`), which describes end-to-end latency and is not scored. The arithmetic r1 offered as a sanity check (7.577 % × 0.84228 = 6.382 % ≈ 6.378 %) is correct algebra with the two labels swapped: `(R_wall − 1)/(R_decode − 1) = D_m/T_m` exactly when both legs prefill in the same time, which they do here. r1's further conclusion that the pair `Sp3 = 1.437971 / Hp3 = 1.521771` arose from "prefill charged twice" is therefore **inverted** — that pair is the legitimate wall-clock ratio, diluted because a fixed ≈4 s seed prefill is not accelerated.

Three independent proofs, in increasing authority: (i) `spt × 512 = 38.05845701694889` matches `decode_seconds = 38.058457016944885` to 1e-11 while `decode + prefill = 42.0525` does not, and `prefill_seconds_per_token = 3.9940489530563354/512` exactly; (ii) `Sources/MLXFastCLI/main.swift:1509` emits `report.decodeSecondsPerToken`, and `QwenRuntimeBenchmark.swift` carries `decodeSecondsPerToken` and `prefillSecondsPerToken` as separate parallel fields; (iii) the run's own `score.json` has `.score == .metrics.mtp_decode_speedup` over these same two fields, gated by `ranked_decode_speedup_floor = 0.9` — `program.md`'s published 0.90 floor.

Across all 20 timed legs measured in r1 and r2, prefill is **3.9937–4.0111 s** per leg: **9.47–9.51 %** of a serial wall-clock leg and **13.60–15.22 %** of an MTP one. That asymmetry is the whole story — a faster MTP leg makes the unaccelerated prefill a larger share, so the wall-clock ratio always understates the scored one. Recomputed on `af80b0fc`, `D_m/T_m` = **0.85570** (CURVE) and **0.86391** (S18), versus 0.84228 on `e6e6f81`. Reported as a **clearly-labelled secondary**, r1's held-out wall-clock `g_median` is **+4.556 %** against the scored **+5.284 %**.

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

---

# r2 — transfer to `af80b0fc`, and what the base actually ships

Assignment `qwen38-r1-e17-curve-transfer-and-refit` revision **r2**, base
`af80b0fc93cf20e8405631bb53365ace21a1f913`, host Apple M4 Pro (Mac16,11, 14 cores, 48 GiB, macOS 26.5.2).
Everything below is measured on that base unless it says otherwise.

The headline of r2 is not the number. It is that **the premise of the assignment was wrong in a way that
made the experiment more valuable, not less**, and that a prediction the advisor asked me to test came out
false. Both are reported before the metric.

## Finding A — the merged depth curve is **NOT** on the live base

The r2 assignment described the control arm as "the shipped CURVE". It is not shipped.
On `af80b0fc`, `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` declares a **scalar**:

- line 530: `headStepCostRatio = 0.18`
- line 629: extend test `h * (1.0 + expected) / (1.0 + Double(depth) * h)` — a *linear* depth price
- line 597: `segmentedStreakGate = 2`

`headStepCostRatioByDepth` and its `cumH` prefix sum exist at `e6e6f817` (r1's base) and are **absent** at
`b85e782`, `d098212`, `3c9317d` and `af80b0fc`. So the curve was never merged forward; the live default is
exactly r1's *losing* arm.

This is not an inference from `git log` alone — it is confirmed by measurement. My control arm `S18`, built
as byte-untouched HEAD, reproduces r1's `FLAT18` **exactly**: same round count, same depth histogram, same
accepted/rejected counts, same declared row total, same replay count. The live default *is* r1's FLAT18.

**Consequence:** restoring the curve is a genuine, submittable source change worth ≈+5–7 % locally, not a
no-op re-measurement. The r2 premise as written is invalid; the corrected question — "does r1's curve
transfer from `e6e6f81` to `af80b0fc`?" — is the one I answered.

## Finding B — the base's own doc comment brackets the scalar with ranked M5 evidence that contradicts the local mechanism

`Qwen36MTPBlockSession.swift:505–596` on this base carries a calibration record from ranked M5 runs:

| `headStepCostRatio` | ranked score |
| --- | --- |
| 0.14 | 2.766 |
| 0.15 | 2.667 |
| **0.18 (shipped)** | **≈2.934 — ranked local optimum** |
| 0.32 (`fc62d1aa`) | **2.84585, −3 %** |

The 0.32 run is the important one. Its serial baseline leg was flat (0.038092 → 0.038070 s/token), drafts
shortened (4.35/4.89/5.78/5.33/5.04 → 3.36/4.01/4.53/4.03/4.76), and candidate decode time **rose 0.95 %**.
The comment's own conclusion: **"this pool rewards depth."**

My CURVE arm shallows: mean drafted depth 2.020 vs 2.367, max depth 3 vs 4, share of rounds at `M ≥ 5`
0.00 % vs 8.57 %. That is the same direction that lost 3 % on ranked M5. The local win is real and large;
the ranked evidence against the mechanism it uses is also real. **These are not reconciled**, and per
pre-registration §6-P3 that is sufficient to bar advancing the arm as submittable on local evidence alone.

Related ranked history in the same region, for whoever picks this up: `segmentedStreakGate = 2` has a
promotion record (newjordan 2.91995 promoted; hadakang 2.92976; `4650c96e` 2.93524 vs 2.93429 base), gate 1
is dead (2.833, −7.1 %), gate 0 ties (2.9200). `sdpaWidthWallDepthCap = 5` has a bitwise justification —
widths 6–9 drift in top-2, and `attentionWithCacheUpdate` splits 6..9-row attention into two ≤5-row `sdpa`
calls, measured bit-exact. Neither was touched.

## The advisor's gate-2 prediction is **falsified**

The advisor predicted that on `af80b0fc` (`segmentedStreakGate` 3 → 2) the curve's depth histogram would
move **deeper** than r1's `{1:2, 2:231, 3:13}`, and that the `M ≥ 5` share would become **> 0.00 %**.

Measured: `{1:2, 2:237, 3:7}` — the histogram did not move deeper; it did not move meaningfully at all.
`M ≥ 5` stayed **exactly 0.00 %**.

**Mechanism, and it is benign.** `segmentedStreakGate` governs *entry to segmented verify*, which the
campaign ledger ties to the width-8 verify kernel. Both of my arms cap at max drafted depth 4 (`M ≤ 5`), so
that path is never reached. **Gate 3 → 2 is inert in this regime.** My pre-registered simulator estimate
put the gate's contribution at ~0.015 mean depth — directionally right and conservative; the truth is 0.000.

Per the advisor's own stop rule, this **stops the `h[1]` sweep**. `H1LO` / `H1MEAS` / `H1HI` were built and
hash-verified and are listed below, but were **never timed and carry no numbers**.

## Finding C — `research/twin_audit.py` is already **RED** on the assigned base

`af80b0fc` fails its own Metal-twin gate: `STALE quantized: section drift in
mlx/backend/metal/kernels/quantized.h`, 1/29 twins stale. The diff is **comment text only** — the
checked-in header has a 3-line condensed comment where regeneration from the runtime-effective JIT twin
produces the original 10-line version. **Zero semantic difference.** I did not fix it: `quantized.h` and
the kernels are outside my assignment's allowed edits. It is reported because a pre-existing red audit
**masks future real drift** for every subsequent experiment.

**Related false alarm that is *not* a defect.** Every arm logs
`mlxfast-swift: warning: mlx.metallib ... built from different vendored Metal sources`
(recorded `6639cc59…`, current `3dd0ffd6…`). The recipe at `tools/build-mlx-metallib.sh:63` hashes *all*
files under `Vendor/mlx-swift/Source/Cmlx/{mlx,mlx-generated}`; exactly two are newer than the metallib —
`quantized.h` and `mlx-generated/quantized.cpp` — both arriving with `d098212` (organizer sync
`156b5b75`). Per `program.md`, a kernel family **with** an `mlx-generated/*.cpp` twin is JIT-compiled from
that C++ source string and is *not* served from `mlx.metallib`. So the stale metallib does **not** mask the
base's promoted quantized-kernel edits, and I deliberately did **not** rebuild it mid-experiment.

## Finding D — the depth curve's cost model is structurally misspecified

The base's own `quantized.h` comment, `research/ESTABLISHED_FACTS.md:1221` and
`senpai/campaign-ledger.md:81` all record crossrow QMV dispatch costs of **319 / 437 / 216 µs for
M = 7 / 8 / 9**. That is a *register cliff*, not a curve: an even 4+4 split at M = 8 needs two simultaneous
`vec<float,4>` accumulators, while M = 9 uses three-lane vectors and is cheaper despite doing more work —
which is why M = 8 is dispatched as 3+3+2.

The extend test at line 629 prices depth **linearly** (`1.0 + Double(depth) * h`). Since `M = depth + 1`, a
linear model **cannot represent a cliff**. That is a clean structural reason why a per-depth vector beats a
scalar, independent of which numbers are in the vector.

**Caveat, stated because it matters:** those microbenchmarks are dispatch-side at widths 7–9, whereas my
arms' hot range is `M = 2–5`, where the cost shape is *unmeasured*. Finding D is therefore a mechanism
**hypothesis** consistent with the win, not a demonstrated cause of it.

**Ledger corroboration and a transfer warning.** Gate-2 and the M = 8 3+3+2 split were *one deliberately
paired* organizer change (gate values: `ef16dea4` = 3, `e6e6f81` = 3, `b85e782` = 2). The ledger warns that
results measured on `ef16dea4` / `e6e6f81` had both halves at the wrong setting, invalidating depth-8 / M = 8
arithmetic, depth histograms and `h(8)` estimates. **r1 was measured on `e6e6f81`** — precisely the transfer
risk P1 existed to detect. The answer: for the depths that `h = 0.18` actually reaches, the transfer is clean.

## Index convention, corrected before the first timed arm

An earlier pre-registration draft of mine asserted `M = depth + 2`. That was **wrong**, and I found it
before spending a single timed leg. Reporting it rather than quietly fixing it:

1. **`M = depth + 1`**, so **`M ≥ 5` ⇔ `depth ≥ 4`**. Proof: `Σ(depth+1)·n` reproduces `declared_rows`
   exactly (S18 825, CURVE 743), while `Σdepth·n` (580 / 497) and `Σ(depth+2)·n` (1070 / 989) do not.
   Cross-check: 21/245 = 8.571 % = S18's measured `M ≥ 5` share.
2. **Marginal `h[i]` prices the step `depth i → i+1`**, taking that round's width from `i+1` to `i+2`. So
   `h[3]` is the first entry that buys `M = 5`, and `h[1]` governs "go to depth 2".

`wide_share()` in the analyser was always computing the right thing (`depth ≥ 4`); only its printed legend
was stale, and that is fixed.

## Q1 (r2) — headline, in score currency

`english`, 512 decode tokens, ABBA position 1 then 2, decode-only (= the scored currency, see the
convention section above):

| arm | serial s/tok | mtp s/tok | `raw_p` | `g%` vs control |
| --- | --- | --- | --- | --- |
| **CURVE** (candidate) | 0.074303 | **0.046275** | **1.60569** | **+6.821 %** |
| `S18` (control, live default) | 0.074333 | 0.049662 | 1.49677 | — |

`d_raw = +0.10892`. Serial-leg floor between the two byte-identical depth-0 legs: **0.041 %**.

Dual convention, both currencies, so the number cannot be misread:

```
prompt   arm    raw scored  raw wall  g% scored  g% wall     D/T   (Rt-1)/(Rd-1)
english  CURVE      1.6057    1.5183     +6.821   +5.926  0.85570      0.85567
english  S18        1.4968    1.4288         --       --  0.86391      0.86314
```

`raw scored` is the decode-only ratio the harness scores. `raw wall` adds the ~4.0 s seed prefill to both
legs; it is *unscored end-to-end latency*, diluted because prefill is unaccelerated. The identity
`(R_wall − 1)/(R_decode − 1) = D_m/T_m` holds to 5 decimal places, which is what a shared-prefill pair
must satisfy.

## P1 — transfer verdict: **CONFIRMED**, and by an unusually strong route

r1's mechanism transfers from `e6e6f81` to `af80b0fc`. The evidence is stronger than a matching headline:
**both arms reproduce their r1 counterparts bit-for-bit in behaviour.**

| arm | rounds | mean depth | max depth | accepted | accept rate | replays | rows checked/declared | `M ≥ 5` | depth histogram |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `S18` (control) | 245 | 2.367 | 4 | 267/580 | 46.0 % | 110 | 825/825 | 8.57 % | `{1:19, 2:138, 3:67, 4:21}` |
| `CURVE` | 246 | 2.020 | 3 | 266/497 | 53.5 % | 74 | 743/743 | 0.00 % | `{1:2, 2:237, 3:7}` |

Identical histograms, rows, accepts, replays and round counts to r1. `g` differs (6.821 % vs r1's 6.688 %)
on **bit-identical behaviour**, so that 0.13 pp gap is pure timing noise, not a base effect.

Stale-binary risk was ruled out from `meta.txt`, not assumed: fresh non-overlapping timestamps
(S18 13:52:17Z → 14:02:25Z, CURVE 14:02:28Z → 14:15:25Z), distinct `worker_sha256` per arm, shared
`cli_sha256 = c9bfcaf9…`, `dirty = 0`, and `mlx_qwen_env=` empty on every arm.

Thermal note: `meta.txt` records `thermal_before = 39.14 C / thermal_after = 60.66 C` (S18) and
`58.13 C / 60.88 C` (CURVE). The `before` value is `e11-run.sh`'s entry snapshot taken *before*
`benchmark.sh` runs its own internal per-leg cooling gates, so each timed leg was still individually gated
below 40 C. Job A's measurements are sound.

## Noise floor — honest, and one estimate short

`S18R` — a byte-identical replicate of the control, built and hash-verified for exactly this purpose — was
**never measured**, because the host became thermally unrunnable (next section). I therefore have **no
arm-level replicate floor** and will not invent one. Two partial estimates:

- **serial floor 0.041 %** — english job A ran two byte-identical depth-0 serial legs in different thermal
  slots (0.074303 vs 0.074333). Bounds *session thermal drift on an identical binary*, but it is a depth-0
  leg and slower per token, so its variance may not transfer to an MTP leg.
- **≈0.13 pp on `g`** — r1 CURVE vs r2 CURVE, bit-identical behaviour, different sessions *and* different
  bases (+6.688 vs +6.821). This is the better arm-level estimate because it compares MTP legs.

The measured effect is ~50–170× the larger of these. That is why I am willing to call P1 confirmed even
without `S18R`.

## Blocker — the host's idle GPU temperature rose above the cooling gate

Two jobs to extend the sweep to `narrative` both failed, and the cause is environmental, not a code defect.

- Job B (`5febb677-7371-47e4-9fee-e7baeab9ee0a`, exit 1): first gate passed at 39.9 C; reference-row
  generation heated the GPU to 58.6 C; the second gate — before the first timed leg — plateaued at 40.3 C
  across 27 polls and aborted at 270 s.
- Retry (`d44f374a-ab85-4647-b79a-6f89ac16cdbf`, exit 1): failed at the **very first** gate, before any GPU
  work, aborting at 180 s with `min seen 40.6 C`.

Diagnosis, checked rather than assumed: no `mlxfast` / `runtime-worker` / `benchmark` processes alive; no
stale lock files; GPU idle at 40.43 C then 40.61 C with `gpu_power = 0.008 W`; CPU only 34.6 C, i.e. cooler
than the GPU, so this is not CPU spillover. The reading is plausible and varying, so this is **not** the
documented sensor quirk at `benchmark.sh:863-870` — that failure mode is the opposite (frozen implausibly
*low*, e.g. a constant 3.657 C). What the timestamps show is a monotone ambient rise:
13:52Z 39.14 C → ~14:16Z 40.3 C → 14:40Z 40.43 C → 14:48Z 40.61 C — about +1.5 C in under an hour. **The
host's idle floor moved above the ≤40 C gate and kept rising.**

The gate constants in `benchmark.sh` are all `readonly` and not env-overridable: `COOL_GATE_TEMP_C=40`,
`POLL=10s`, `ABORT=180s`, `STALL=90s`, `MAX_WAIT=900s`, `PROGRESS_EPSILON_C=0.25`, `FAN_OFFER_STALL=60s`.
The reader is `local_gpu_temp()` at line 448. Interactive fan boost is unavailable under `run_job`.

ABBA rotation is confirmed working: job B's log line 2 reads
`=== e17-run: prompt narrative (index 1) arms CURVE S18 ===`, correctly reversing english's `S18 CURVE`.
The retry deliberately reused the identical command so that the pre-registered `rot = idx % len(pair)`
rotation, and not I, chose arm order.

The thermally aborted partial run is preserved at
`.mlxfast-private/e17/aborted/narrative-CURVE-thermal-abort/` (`meta.txt` with `exit=1`,
`thermal_before=40.30C`, `thermal_after=40.60C`, `started=2026-08-17T14:37:35Z`,
`finished=2026-08-17T14:41:47Z`, `worker_sha256=ed12f46…`, and an empty `reports/`).

### Shortcuts I declined, and why

1. **`MLXFAST_LOCAL_COOL_GATE=0`** — `program.md` forbids bypassing the cooling gate, and a hot-start timing
   would not be comparable with job A's gated legs anyway. Declined.
2. **`MLXFAST_GPU_TEMP_CMD` pointed at a fake reader** — that is measurement fraud. Declined.
3. **Editing `COOL_GATE_TEMP_C`** — trusted timing/telemetry, outside the editable surface. Declined.
4. **Reusing english's reference rows for the narrative arms** to avoid the 58 C generation spike — checked
   and it is *deliberately* unsupported: `benchmark-qwen-mtp.sh:34` requires rows generated by the
   candidate's **own** build, and line 688 gates on it. Declined.

## Correctness contract — every timed arm

Both timed arms pass the full contract, and these are read out of the reports rather than asserted:

`all_tokens_matched = true` on both legs; `parity_all_ok = true`; `residual_divergence_count = 0`;
`reference_checked_row_total == declared_rows_total` (825/825, 743/743);
`rejected_rows_reference_checked == rejected_draft_total` (313/313, 231/231);
`max_rejected_tail_logit_delta = 0` on both; `uses_pinned_mtp_head = true` with
`head_sha256 = 07293af742df4599d94eda6e9db5782e7f5be10cd1b5fdef7691f4ef404ea81c`, `head_bytes = 238937699`,
`head_origin = hf:dwsdubey/qwen3.8-27b-mtp-4bit@34ee76f6c87a438caa28f975c1cea9b0b005bc71`; drift tripwire
passed; `target_cache_offset_final = 1024` and `non_drafting_round_count = 0` on both.
Stall/p50 block latency: 0.12388/0.08109 = 1.53× (S18) and 0.12378/0.08020 = 1.54× (CURVE), far inside the
4× guardrail.

**`max_rejected_tail_logit_delta = 0` is genuine, not a default.**
`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift:497` accumulates a running
`max(|candidate top-1 logit − reference top-1 logit|)` over every reference-checked rejected-tail row
(declared at `QwenRuntimeDFlash.swift:1049/1152/1199/1238`, wired at `QwenRuntimeMTPDriver.swift:283/356`).
Combined with `rejected_rows_reference_checked == rejected_draft_total`, every rejected row was replayed and
matched to the last bit.

Arm identity (all six built on `af80b0fc`, each verified twice including exact source literals; shared
`cli_sha256 = c9bfcaf9c58d5b5bd31466f4bab8c90a5d693bf8f0afd2818840deef0fd060b7`):

| arm | `headStepCostRatio` at l.530–533 | `worker_sha256` | role / status |
| --- | --- | --- | --- |
| `S18` | scalar `0.18`, untouched HEAD | `aa17ce5c064b5d1f3574783364ed861d4372452d651327187be712fd03f61dca` | control, **timed** |
| `CURVE` | `[0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]` + `cumH` | `ed12f4647045de01b72aadbbee29c6e2e29a53631865b360c1d4a295007d2488` | candidate, **timed** |
| `H1LO` | `[0.18, 0.0800, 0.18×6]` | `a6733ca8a7e6057152740d4b42f1a9991c05af98a60ef2ff49eed9bec24391b5` | **never timed, no number** |
| `H1MEAS` | `[0.18, 0.1152, 0.18×6]` | `72cdfc2f058414f44247de6b3a492e4cd428d344dcf328fb528f8fab5234e1e5` | **never timed, no number** |
| `H1HI` | `[0.18, 0.3000, 0.18×6]` | `f226deaa8e777d3cdf79951747e10246fe6884d696b8ca4d51d8560b8908e869` | **never timed, no number** |
| `S18R` | scalar `0.18`, byte-identical copy of `S18` | `aa17ce5c064b5d1f…` (identical to `S18`) | noise floor, **never timed, blocked** |

`S18` is built as *untouched HEAD* rather than by patching in a flat vector, because with a flat vector
`cumH` equals `Double(depth) * h` bitwise only for depths 0–5 and differs by 1 ulp at depths 6–7. Asserting
byte-identity is stronger than asserting numerical equivalence.

The `h[1]` sweep direction I built also **disagrees with the assignment's §3**, and I followed the algebra
under its own escape clause. The thresholds are `d0→1: h[0]`; `d1→2: h[1](1+reach0)/(1+h[0])`;
`d2→3: h[2](1+reach0+reach1)/(1+h[0]+h[1])`. `h[1]` sits in the `1→2` numerator *and* in every deeper
denominator, so raising it closes depth 2 while **opening** depth 3+. Mean depth is therefore **not
monotone in `h[1]`**, and a 3-point bracket cannot identify a non-monotone response. My simulator agrees:
mean drafted depth S18 **3.573** > H1MEAS 3.430 > H1LO 3.366 > H1HI 3.075 > CURVE 2.388, i.e. **0.18 is a
local maximum**, with H1HI strongly bimodal (`d1:32.7 % d2:1.2 % d3:17.0 % d4:26.8 % d5:19.7 %`). These are
synthetic predictions of record, not measurements.

## Provenance disclosure

The two timed arms of job A carry **different `head_sha` values** in `meta.txt` (S18 `73cba45`, CURVE
`d3bed96`) because I committed a W&B-logger generalisation while job A was running. This is **not a
confound** — both binaries were built before the job started and hash-verified at install, `worker_sha256`
is the authoritative identity, `dirty = 0` on both, and the changed file is a Python logger that no arm
executes — but it is a hygiene slip and it is disclosed rather than smoothed over. Rule adopted for future
sessions: **never commit while a job is running.**

## Submitted surface

`git diff --name-only af80b0fc..HEAD` lists **only 20 `research/` files**. The diff filtered to
`Sources/ Vendor/ fixtures/ benchmark.json .github/ correctness_prompts/ benchmark-qwen-mtp.sh` is
**empty** — zero submitted-surface change, so the editable byte budget is exactly the base's and
`validate-assignment-scope.sh` (which takes `BASE_SHA SUBMITTED_PATH …`) is trivially satisfied.
In particular the CURVE vector was **deliberately not** committed into
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`, per pre-registration §6-P3.

## Conclusion (r2)

- **P1 is confirmed.** The curve's mechanism transfers from `e6e6f81` to `af80b0fc`: `+6.821 %` on
  `english`, decode-only, 512 tokens, with both arms reproducing r1 bit-for-bit in behaviour.
- **The advisor's gate-2 prediction is falsified**, with a benign mechanism: the segmented-verify path the
  gate controls is never entered when both arms cap at `M ≤ 5`. Gate 3 → 2 is inert here. The `h[1]` sweep
  is stopped per the advisor's own stop rule.
- **The base does not ship the curve** (Finding A), so this is a live, submittable change worth ≈+5–7 %
  locally — a materially better outcome than the re-measurement the assignment expected.
- **No headline is proposed.** P3 binds: CURVE wins locally *by shallowing*, and the base's own ranked M5
  record shows that direction losing 3 % at `h = 0.32`. Advancing a shallowing arm on local evidence alone
  is exactly what §6-P3 forbids, and I am not doing it.
- **The held-out sweep is blocked** on host thermals, not on code. `S18R` and the three `H1*` arms are
  built, hash-verified, and reported with **no numbers attached**.
- **A pre-existing red `twin_audit.py`** (Finding C) will mask real Metal drift for the next experiment
  until someone with `quantized.h` access clears it.

## Suggested follow-ups (not implemented)

1. **Log per-round `reach`.** The timed reports expose only `block_request_seconds`,
   `effective_draft_lengths` and `head_provenance` as sequences, so a zero-GPU depth-histogram prediction by
   replaying r1's *real* reach distribution is impossible today. Adding `reach` would have answered the
   gate-2 question **without a single timed leg**.
2. **Measure the `M = 2–5` dispatch cost curve directly.** Finding D rests on width-7–9 microbenchmarks,
   while the hot range of these arms is `M = 2–5`, where the shape is unmeasured. This turns a mechanism
   hypothesis into a measured cause.
3. **Build a depth-preserving variant.** The clean way to resolve the ranked/local contradiction is an arm
   that keeps mean depth at ≈2.37 while still declining the redundant rows — that sidesteps P3 instead of
   arguing with it, and it is the only route I can see to a submittable version of this mechanism.
4. **Re-green `twin_audit.py`** in an assignment permitted to touch `quantized.h`. The current drift is
   comment-only and safe to normalise; leaving it red is what is unsafe.
5. **Raise cooling-gate headroom or cool the host.** Every remaining held-out prompt in this experiment is
   blocked on a ~+1.5 C/hour ambient rise that pushed the idle floor above the 40 C gate. No amount of code
   work unblocks it.

