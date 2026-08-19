SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"revision_id":"r3","base_sha":"329d3644dc96972d6843ecfe759141b8b0ab539d","credit":"thorfinn E22 follow-up #1 (two-piece boundary-aware marginal price, arm C); E27 M5 weight-stream fix (thorfinn)","primary_metric":{"name":"e25/measured_row_step_ratio_at_depth_3","available":true,"value":0.1812497742996191},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

## r3 result (measurement only; supersedes the r2 body below)

Re-measured on base `329d3644` (E27 merged), declared `q2-q4-rerank-v1` head
(`d038fd41…`), 16 legs over 8 prompts × {BASE, FORCE}, 512 seed + 512 decode
tokens each. **No arm, no cap, no policy** — `git diff 329d3644 -- Sources/
Vendor/` is empty for the whole revision.

- **Primary metric `e25/measured_row_step_ratio_at_depth_3` = 0.181250**
  (baseline `0.442442`, minimize, **Δ −0.261192**, −59.0 %).
- W&B run [`95umlz5m`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/95umlz5m),
  group `qwen38-r1-e25-per-row-draft-price`, 232 summary keys.
- Evidence: [`research/e25r3-refit-post-force.json`](../e25r3-refit-post-force.json),
  [`research/e25r3-refit-post-base.json`](../e25r3-refit-post-base.json),
  [`research/e25r3-refit-pre.json`](../e25r3-refit-pre.json),
  [`research/e25r3-recost-post.json`](../e25r3-recost-post.json),
  [`research/e25r3-recost-pre.json`](../e25r3-recost-pre.json),
  [`research/e25r3-recost-basetape.json`](../e25r3-recost-basetape.json).
  Pre-registration: [`research/e25-r3-prereg.md`](../e25-r3-prereg.md) (`150c957`,
  committed before the first leg).

**The wall moved one row deeper; it was not removed.** `T(4)` fell
`132.257 → 108.346` ms (−18.08 %) and no other depth moved by more than 0.79 %.
`c_3` dropped under its `1/4` ceiling (`0.4394 → 0.1813`) but `c_4` rose through
its `1/5` ceiling (`0.0896 → 0.3279`). The reachable depth set widened from
`{0..3}` to `{0..4}` — on both instruments, on 8/8 prompts.

**The economic optimum did not move at all.** Realised-rate argmax over
constant-depth policies is depth **2** before and after (bootstrap share 80 %).
Depth 4 improved from −31.0 % to −15.9 % versus depth 2 and still does not pay,
because acceptance saturates near 2.4 tokens/round from depth 3 onward.

**My pre-registered verdict is falsified.** I predicted residual headroom
< +1.0 % and "the lever is closed". Measured: **+3.63 % pooled / +3.71 %
median-of-8** on the r1 1947-round tape re-costed on the post-E27 curve
(from +6.11 / +5.36 pre-E27), and **+3.99 / +4.10** on a fresh post-E27 BASE
tape. My explicit falsifier (headroom > +1.5 %) triggered. I conflated the M=5
dispatch cliff with the mis-calibrated scalar `h = 0.18`; E27 fixed the first
and left the second, and arm C/D's win was mostly the second all along. Arm B
(`DEEP_CAP = 3`) behaved as a cliff effect and did collapse, +3.31 → +1.19 %.

**Everything else I registered was accurate**: advisor band [0.13, 0.23] hit
(0.1813); my `c_3` point 0.1754 (error 0.0059); my `c_4` point 0.3343 (error
0.0064); mean depth 2.30 ± 0.25 (measured 2.4577); depth ≥ 4 share 5–12 %
(measured 9.37 %).

**Exactness receipt:** the pre- and post-E27 FORCE tapes are separate builds yet
reproduce a bit-identical acceptance table (1594 rounds; `p_i` = 0.6926 /
0.5840 / 0.5077 / 0.4190 / 0.3860 …). E27 changed time, not tokens. All 16 legs:
`all_tokens_matched` 16/16, `residual_divergence_count` 0, parity 16/16, rows
closed 16/16, exit 0.

**Base drift checked, not assumed:** the advisor branch moved to `a88b4d33`
mid-run. Both scored-path deltas are inert with no env override, and
`ladderActive = inputs.dim(1) <= 9` is a *context* line in that diff — `<= 9`
was already at `329d3644`. **This curve transfers to `a88b4d33` unchanged.**

- Decision: **not useful as a candidate, valuable as measurement.** Nothing was
  submitted; r3 proposed no code change. Live bar re-anchored to
  **3.24326223889754** (`11863aa9-…`, frontier `5068eb8d`).
- ⚠️ Two of 16 legs recorded `dirty=2/3` from `research/*.py` edits made while
  the last pair was in flight. Worker and source digests are identical across
  all 8 FORCE legs, so the executed code was unchanged; disclosed in §32.3.
- Not gate-qualified: `MLXFAST_LOCAL_COOL_GATE=0` on every leg with all four
  disclosures preserved verbatim; ABBA-counterbalanced; M4 Pro, not the ranked M5.

Full analysis: [`research/e25-results.md`](../e25-results.md) Part III (§24–§33).

## r2 result (superseded by r3 above)

Re-measured on base `d7619a7`, with the declared `q2-q4-rerank-v1` head, all
gates green (`gates.all_pass = true`, zero failures, no missing legs).

- **Primary metric `e25/mtp_true_decode_gain_pct_median_of_8` = +3.1797 %**
  (mean +3.2772, min −1.3644, max +7.2377), **7/8 improved**.
- W&B run [`ydgtamkp`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ydgtamkp),
  group `qwen38-r1-e25-per-row-draft-price`.
- Machine-readable evidence: [`research/e25r2-timed.json`](../e25r2-timed.json)
  (16 timed legs), [`research/e25r2-pool.json`](../e25r2-pool.json) (forced-depth
  price curve), [`research/e25r2-policy.json`](../e25r2-policy.json) (offline replay).

**Deliverable (1) answered: arm D is a hard `DEEP_CAP = 3`, not a price.**
`effective_max_draft_len` collapses 4/5 → 3 on 8/8 prompts,
`candidate_rounds_above_cliff` = 0 on 8/8, and with the measured
`c_3 = 0.4394 > 1/4` depth ≥ 4 fires **0 / 400,000** analytic draws. The wall is
the measured `M = 5` weight-stream pass cliff, not arm D's `max(...)`.

**Three retractions, all mine:**

1. r1's **8/8** sweep does not reproduce — `natural_history` measures −1.364 %.
2. Both **dose-response** claims (above-cliff share, and its host-normalised
   replacement) are withdrawn: the correlation flips sign with subset choice
   (−0.101 → +0.333 → +0.706 → **+0.227** at n=8).
3. r1's claim that the in-source **h-sweep supports arm D** was a cross-era
   comparison and is wrong; that evidence points the other way.

**Measurement-resolution finding (§21.5), the most transferable result here:**
re-running an identical `technical` BASE leg moved it **−3.397 %**, swinging that
prompt's gain by **3.347 pct points**. Implied **σ ≈ 1.73 pct points** per
per-prompt gain, so the headline is **+3.18 ± ~0.77**, is *not* distinguishable
from r1's +3.8346, and `natural_history`'s regression is only **0.79 σ** from
zero — I therefore do **not** claim it as a genuine per-prompt regression. This
is caused by `MLXFAST_LOCAL_COOL_GATE=0`; see follow-up §22.7.

- Decision: **local winner with a proved mechanism and a predicted ranked
  regression — do not submit.** The ranked pool drafts at mean length 5.078
  (implied uniform `p ≈ 0.8168`) versus local `p0 = 0.6926`; arm D caps at 3 and
  would truncate the majority of ranked rounds, where a ~19 % draft shortening
  already cost ~3 % of score. The submission slot is in any case busy
  (receipt `9197ed62-621f-474d-bfba-e1efddd9dd4c`).
- Live ranked bar re-anchored per deliverable (e): **3.2341518328631**
  (submission `942e5ab2-1c46-4c50-b7c3-eaf948878ed0`, frontier `474c7501`).
- Candidate files unchanged from r1: **one** —
  `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (`+36/−2`).

**Credit: this experiment implements thorfinn's E22 follow-up #1 — the two-piece
boundary-aware marginal price, arm C in that proposal.** The idea, the diagnosis
of the scalar `headStepCostRatio` fit's failure mode, and the shape of the fix
are thorfinn's. This report contributes the instrument gate, the measurement of
the price curve, the confound controls, and the timed 8-prompt evaluation.

Full analysis, derivations and legality argument: [`research/e25-results.md`](../e25-results.md).
Machine-readable evidence: [`research/e25-phase0.json`](../e25-phase0.json) (price curve, arm simulation, pre-registration) and [`research/e25-phase1.json`](../e25-phase1.json) (timed matrix, fixed-window accounting).

- Student / branch: `qwen-edward` / `qwen-edward/per-row-draft-price` (PR #29, assignment `qwen38-r1-e25-per-row-draft-price` r1)
- Hypothesis and target cost: the shipped draft price `h / (1 + d·h)` with `h = 0.18` is a smooth scalar fit, but the *measured* per-row cost of the target verify is not smooth — E21's tape shows `T(4) − T(3) ≈ 40 ms` against `T(3) − T(2) ≈ 12 ms`. Replacing the fitted price with the measured one at each depth should stop the gate from buying rows whose real cost exceeds their expected value. Target cost: the rejected-row work, which is 45–55 % of all proposed rows on the local prose pool.
- Decision: **green locally** (8/8 prompts, effect 6–130× the serial noise floor), **with a hard ranked-transfer caveat that argues against submitting** — see §7.3 of the full write-up and "Prompt or M5 transfer risk" below.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: `0d2eef9cac75d890de06a5eef4fd686c3c34c1ef` / `d1530a409848b82a0a1890141c1483875d1e0173` / this branch head (the marker cannot contain its own commit; the head of record is in the typed `submit_experiment_result` payload)
- Yukon promoted submission / source ref used as frontier: `bd007bc7-e8ab-4919-baf4-d5e90068dd83`, `sourceRef d1530a40`, score `3.13098700135133` per [`senpai/frontier-state.json`](../../senpai/frontier-state.json) (`observedAt 2026-08-17T21:24:25Z`). Source comments in the file I edited quote a live accepted bar of `3.14642585386152`; I did not query Yukon to reconcile the two, and did not need to, because I am not recommending a submission.
- Submitted candidate files: **one** — `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (`+36/−2`). Nothing else under `Sources/`, `benchmark.json`, `fixtures/`, `mtp-head/`, `tools/` or `Package*` is touched.
- Supporting test, tooling, or documentation files (not submitted): `research/e25-build.sh`, `research/e25-run.sh`, `research/e25_price.py` (Phase 0: instrument gate, price fit, arm simulation), `research/e25_phase1.py` (Phase 1: reduction, fixed-window accounting, W&B), `research/e25_counters.py`, `research/e25_summary.py`, `research/e25-results.md`, `research/e25-phase0.json`, `research/e25-phase1.json`.
- MTP head provenance and draft policy: pinned organizer head only, `sha256 07293af742df4599d94eda6e9db5782e7f5be10cd1b5fdef7691f4ef404ea81c`, identical on all 16 timed legs. No `MLXFAST_QWEN_MTP_HEAD_DIR` override anywhere; `uses_pinned_mtp_head` true on every run. `--mtp-depth 8` offered, serial control depth 0, the gate chooses per round.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh 0d2eef9c Sources/MLXFastModel/Qwen36MTPBlockSession.swift` → `assignment scope OK: 1 submitted path(s)`.
- Editable source bytes / headroom / growth / exempt-head bytes: `source=2426820/3000000`, `headroom=573180`, `growth=2216/262144`, `exempt=2410/2147483648`, `files=154`.
- Scored-path reachability evidence: `costModelDepth` is called at `Qwen36MTPBlockSession.swift:199`, **before** proposal, inside `Qwen36MTPBlockSession`. `nm -a .build/release/mlxfast-swift | grep -c Qwen36MTPBlockSession` = **0**; the same grep against the runtime worker = **294**. So the trusted driver does not link the edited type and the scored worker does. The build harness asserts a byte-identical trusted-driver digest across both arms and refuses to proceed if either installed driver contains a `Qwen36MTPBlockSession` symbol.

## Evidence

- Host, memory profile, toolchain, and thermal policy: single-GPU Apple M-series host, **not the ranked M5**. Both arms built with `--force-resolved-versions` from one tree, no patching. Cool gate **bypassed** under the advisor's PR #29 §8 authorisation (`MLXFAST_LOCAL_COOL_GATE=0`); every `meta.txt` carries `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, `cool_gate_temp_c=40`, `cool_gate_bypass_reason=host idles above the compile-time 40C gate`. Mitigation is the counterbalanced ABBA schedule within one session plus the reported serial spread; entry/exit temperatures are recorded per arm.
- Exact baseline and candidate commands:
  ```bash
  research/e25-build.sh BASE PRICE
  E11_GOLDEN_DIR=.mlxfast-private/e17/goldens E11_GOLDEN_STEPS=512 \
  E11_BINS_ROOT=$PWD/.mlxfast-private/e25/bins E11_RUNS_ROOT=$PWD/.mlxfast-private/e25/runs \
  E11_TOKENS=512 MLXFAST_LOCAL_COOL_GATE=0 \
    research/e25-run.sh --pairs BASE,PRICE english narrative technical
  # ... dramatic travel philosophy / natural_history medicine, ABBA order index-derived
  python3 research/e25_phase1.py --wandb
  ```
- Tests and risk-based checks: the **instrument gate** ran before any measurement was trusted — feeding `[0.18]×8` through the reimplemented gate reproduces all 1947 taped depths bit-identically (0 mismatches over 6580 walk steps, max |threshold error| `6.67e-7`), which is what licenses the arm simulation. Arm D's d0 and d1 coefficients are bit-identical to the shipped curve by construction. The build harness pins seven shared invariants across arms (`headStepCostRatio = 0.18`, `acceptEMAAlpha = 0.15`, `sdpaWidthWallDepthCap = 5`, `segmentedVerifyDepthCap = 8`, `segmentedStreakGate = 2`, the confidence terms, `reach *= p`, the trace snapshot line) and requires distinct worker+source digests per arm.
- Exact-token and row-ledger verdict: **PASS on all 16 legs.** `all_pass = True`, `failures = []`; `mtp_all_tokens_matched` and `serial_all_tokens_matched` true everywhere, `mtp_parity_all_ok` true, `tokens_emitted_all_legs_512 = True`. Every declared row is accounted for: BASE `6592 = 1947 primary + 4645 proposed`, PRICE `6236 = 1971 primary + 4265 proposed`.
- Divergent tokens or failure category, if any: none.
- Generated-twin audit, if relevant: not relevant — no `.metal`, `.h` or `mlx-generated/*.cpp` file was touched.
- Peak RAM or head/artifact size, if relevant: unchanged; the diff adds a 4-element static `[Double]` table and one function. No new allocation on the round path.

| Metric | Baseline (BASE) | Candidate (PRICE) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token (median of 8) | 0.0745233 | 0.0745504 | +0.036 % (noise floor) |
| MTP seconds/token (median of 8) | 0.0497099 | 0.0473737 | **−4.70 %** |
| local serial-relative speedup (median of 8) | 1.500250 | 1.572157 | **+0.071907** |
| effective mean draft length (median of 8) | 2.3820 | 2.1944 | −0.1876 |
| accepted draft rate (median of 8) | 0.4572 | 0.5008 | **+0.0436** |

The `−4.70 %` MTP row is a **ratio of medians** and the primary metric below is a
**median of per-prompt ratios**; they are different statistics over the same 16
legs and neither is derived from the other. The primary metric is the contracted
one.

**Primary metric, as contracted:** `e25/mtp_true_decode_gain_pct_median_of_8`
= **+3.8346 %** against a baseline of `0.0`, direction maximize. Pooled
`+3.8337 %`, mean `+3.845 %`, range `+2.095 %` (`natural_history`) to `+5.919 %`
(`medicine`). **PRICE wins on 8 of 8 prompts.** `depth_ge_4_realised = 0`,
exactly as pre-registered. Per-prompt table, per-arm counters and the full
fixed-window accounting are in §4 and §5.1 of the write-up.

This is a median over **eight local prose prompts**, which is a stronger design
than a one-prompt directional screen but is still **not** the ranked median over
the eight hidden prompts, and both legs use the same candidate build.

## Conclusion

- What happened and why: replacing the fitted scalar price with the measured
  per-row price makes the gate refuse depth 4 and 5 outright — `max depth` is
  exactly **3** on every PRICE leg while BASE reaches 4 or 5 on every prompt.
  The measured d3 coefficient is `0.442` against the shipped `0.153`, because
  `T(4) − T(3) ≈ 40 ms` is a real cliff that the smooth fit averages away. The
  refused rows were disproportionately rejected rows, so rejected rows fall on
  all 8 prompts while accepted rows fall by at most 9, accept rate rises 8/8,
  replayed rounds fall 8/8, and p50 block latency falls 8/8.
- Evidence for or against the mechanism: strongly for, on this pool. The effect
  is 6–130× the serial noise floor, the sign is consistent 8/8 under a
  counterbalanced schedule, and the realised depth ceiling matches the
  pre-registered prediction exactly. Two honest corrections narrow the claim:
  (1) Phase 0's `+4.688 %` projection is an **upper bound** — it modelled 98
  lost tokens, but the trusted parent owns a fixed 512-token window, so the arm
  loses **0** tokens and instead spends **+24 extra rounds**, which recovers 229
  of the 609 predicted row savings; I tested that explanation rather than
  asserting it (`r = −0.7405`, n = 8, extra rounds vs gain attenuation).
  (2) a prefill double-subtraction bug in my own reducer was found and fixed
  before the matrix was reduced, and `assert_scored_unit()` now pins both legs of
  every run to the trusted `score.json` per-token metric so the class of error
  cannot recur silently. A useful side result: the BASE arm **independently
  regenerated the E21 tape** — 1947 rounds and 4645 proposed rows, to the row.
- Prompt or M5 transfer risk: **high, and it points the wrong way.** Source
  comments record ranked scores for uniform `h = 0.32` → `2.84585` (−3 % vs the
  bar), `h = 0.15` → `2.667`, `h = 0.14` → `2.766`, and the `h = 0.32` note
  observes ranked per-prompt mean drafts of 4.35/4.89/5.78/5.33/5.04 and
  concludes "this pool rewards depth". My tape has n = 167 at d4, n = 9 at d5,
  and nothing above 5; local prose means are ≈ 2.4. Arm D is therefore *more*
  aggressive than `h = 0.32` at exactly the depths the hidden pool occupies, and
  its d4/d5 coefficients are extrapolated from n = 9. A clean 8/8 win at depth
  ≈ 2.2 is weak evidence about a pool at depth ≈ 5.
- Smallest useful next action: force depths 4–6 on a prose tape to get real n at
  d4/d5/d6 and refit. That single measurement decides whether arm D's aggression
  at d3 is right or whether the ranked pool needs a curve with a *falling* tail.
- Recommendation: **do not submit to Yukon without a ranked-representative
  check.** The mechanism is sound and the local result is real, so keep the
  branch and the measured curve; the open question is entirely whether the price
  holds at `d ≥ 4`, and this experiment cannot answer it. Compose later, after
  the deep-depth refit.
