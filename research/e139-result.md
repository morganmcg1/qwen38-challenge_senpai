SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e139_composed_rider_ranked_pct","available":true,"value":0.0},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

# E139 — pricing the two held riders on the zero-noise acceptance channel

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e139-zero-noise-acceptance-instrument-and-composed-riders`
- Hypothesis and target cost: both held riders are byte-model changes whose
  published-median value is `gross byte saving − measured acceptance cost`.
  The acceptance term is what destroyed E136 C1 by 18.4x and the offline corpus
  screen prices it at zero. The decode round count is exactly reproducible for
  a fixed arm, fixture and token window, so one leg per arm resolves the
  acceptance term with no variance at all.
- Decision: **green as a measurement, empty as a composition.** Both riders are
  priced exactly. Neither is composed: rider A is worth zero and rider B's
  literal was transferred to thorfinn in feedback F2.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit:
  - assignment `BASE_SHA` `328c4b9eac1b386f0c0913afcf0c7a64c232e5c0`
  - experiment base for every leg `7279218cde67d81b930d953e56368e15b9c84c39`
  - organizer synced commit `c0dbec051c58bccf5435ee1e1e5b01271dc7e179`;
    live `upstream/main` at report time `b40c28e95cc7488e798f2c90b4984bf73558ff93`
  - candidate commit: **none.** No submitted-surface change is proposed.
- Yukon promoted submission / source ref used as frontier: **ambiguous, flagged
  rather than guessed.** `senpai/frontier-state.json` (observed
  `2026-08-22T18:30:00Z`) records `b6cb0fea-880f-4737-8129-3769e071a808`,
  sourceRef `d44ad22960039ea2eca6c32806af86195e101ecc`, score `3.5250913`. The
  assignment body (written `2026-08-22T20:46Z`) records `623e77af` at
  `3.52085227`. The two disagree in both id and score. Nothing in this result
  depends on which is current, because no submission is proposed. The advisor
  owns the reconciliation.
- Candidate build fingerprint: `worker_sha256`
  `a4b777893ceb3c62eb88f84f8010b47eba23471fd9e8a883d4bea7a3cac7abb6`,
  `cli_sha256`
  `41a568caf74345a4c1757f96bc64cc38d0b73087566e2ffde3e1667e82ca0262`.
  Both constant before and after all 23 legs.
- Submitted-surface / generated-twin / metallib digests: candidate tree
  byte-identical since `7279218c` — `Sources 39e6130dca78c28eb1a499e260edb948cc488fa9`,
  `Vendor 2d04bb204e0767d7e6549099b5e0bf2b9477d308`,
  `Package.swift 2896ef2487c54aaeb91cc18fcb675ed14c2d3ff3`,
  `Package.resolved 0c4a64931660f3e88949e8d6be834812e1fa0d79`,
  `mtp-head.manifest.json 331ce9d19320816390deeff5606795213dd903e0`.
  `metallib_source_fingerprint 2050ebf1c1cf091ebbf35fceb7e9c1a9b399b7ed371b133d10b8546734efe7a7`.
- Submitted candidate files: **none proposed.** The branch does carry one
  submitted-surface file,
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`, which holds the
  two measurement gates. See "The scaffold" below; it must not reach a
  submission unchanged and the advisor owns the decision to keep or strip it.
- Supporting test, tooling, or documentation files: `research/e139_session.sh`,
  `research/e139_analyse.py`, `research/e139_wandb_log.py`,
  `research/e139_probe_margin.py`, `research/e139-acceptance-channel.json`,
  `research/e139-probe-ladder-screen.json`, `research/e139-probe-margin.json`,
  `research/e133_screen.py` (three repairs), `research/e136_*` (cherry-picked
  durable tooling), `senpai/rebuild-and-assert-worker.sh` (one repair).
- MTP head provenance, digest, and draft policy:
  `head_provenance_sha256 dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`
  and `head_manifest_tree_sha256 559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`
  on **every** leg, baseline and arm. The organizer-pinned head is used
  throughout; no candidate head is declared. Draft policy is the shipped
  schedule at offered depth 8, except the three offered-depth control legs.
- Token window, fixture, reference source, and harness: 512 decode tokens
  after a 512-token seed. Two fixtures.
  `benchfixture` = `correctness_prompts/public_longcopy_gate_english_512_256.json`,
  `prompt_sha256 3d922b1a0ada04d9827b905c881232bf50fb697d4be9ab3ee21346f7e0b8ae9c`,
  `golden_sha256 66858d9561663b62e58a97191428da0a3816cdb00f67a93f724c7b2d0cf2301e`.
  `natural_history` = `research/e17_prose_natural_history_512.txt`,
  `prompt_sha256 10251cd46b6c6d4bc768900fd8a9db51b9a0aad522c3eb96d8c758f43456a567`,
  `golden_sha256 80e893da40e5b4d275daf4193f18970a578d66a388ab1fd328e80d097a6f2163`.
  Reference rows are candidate-generated goldens cached per fixture.
  **Harness `local` for every measurement.** The published-median figures are
  labelled `harness=ranked` in the tooling because they are a model of the
  ranked quantity, and they are built only from candidate-leg quantities.
- Exact cell: two proposal-path cells inside the MTP head.
  1. `qwen35DraftSelectedAffine4RerankKernel`, declared at
     `Qwen35.swift:4056`, store at `:4117`, `InT` typedef at `:4143`. Metal JIT
     string compiled into the worker binary.
  2. `qwen35DerivedClusterProbeFraction` at `Qwen35.swift:4880`, consumed at
     `:5923`, over 12,292 centroid rows. Swift constant.
- Official causal path and score equation (`harness=ranked`): both riders act
  only inside the candidate MTP leg. The ranked serial numerator is produced by
  the runner-owned prebuilt baseline workspace, so
  `d ln(ranked baseline serial time)/dx = 0` for every edit here. Any reduction
  in candidate seconds per token therefore raises every affected `raw_p`.
  `senpai/verify-ranked-score-boundary.sh` passes at branch tip.
  **CAMPAIGN RULE 118.** Every contrast here is priced on the candidate leg
  and never on the published median. The model is
  `candidate_leg_medpair_pct = k × gross_byte_pct_local − measured_acceptance_cost_pct`,
  with `k = 0.95`, band `[0.75, 1.15]`. The `0.95` is the **measured**
  candidate-leg medpair transfer from two rival receipts, not an assumed
  constant; its derivation is in "Calibration" below. This experiment
  contributes no serial-leg observation, because it never runs the ranked
  serial workspace, so no serial-leg noise diagnostic is offered.
- Assignment-scope preflight:
  `senpai/validate-assignment-scope.sh 328c4b9e… Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
  → OK.
- Editable source bytes / headroom / growth / exempt-head bytes, measured
  twice because the branch tip moved after the revert:
  - Instrument tip, while the `MLX_E139_*` gates were still present:
    `senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`
    → `source=2617128/3000000 headroom=382872 growth=162293/262144
    exempt=2410 files=154`. Within all three limits.
  - Submitted tip, after the revert:
    `senpai/check-editable-budget.sh 328c4b9eac1b386f0c0913afcf0c7a64c232e5c0`
    → `editable budget OK: source=2613813/3000000 bytes headroom=386187
    growth=0/262144 exempt=2410/2147483648 files=154`. **Growth is exactly
    zero**, which is the arithmetic statement of the same fact as the empty
    candidate diff.
- Scored-path reachability evidence: both cells are witnessed live from the
  run's own round trace, not from the environment the leg was asked with
  (CAMPAIGN RULE 114). Each round emits
  `sel_env=<top32>+e139fp32:<gate>:<drafts through the rerank kernel>+e139p:<gate>:<probes>`.
  `probes = ceil(fraction × 12292)`, so an arm is certified by an integer the
  run derived: `0.25→3073`, `0.15→1844`, `0.10→1230`, `0.09→1107`, `0.08→984`,
  `0.07→861`, `0.06→738`, `0.05→615`, `0.04→492`, `0.03→369`, `0.02→246`. The
  rerank cell is witnessed by a nonzero draft count on every leg. A leg whose
  witnessed gate or probe integer disagreed with its arm would be discarded;
  none did.

## The scaffold, stated plainly

Commit `8e7a9620` adds two environment gates to the submitted-surface file
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`:

```
MLX_E139_FP32_TIEBREAK=1    store the fp32 simd_sum result in exact_scores
MLX_E139_PROBE_FRACTION=p   override the derived-cluster probe fraction
```

Both fail closed on an unrecognised value. **Unset reproduces the shipped
kernel source character for character and the shipped `0.25`**, and the live
evidence for that claim is the strongest available: the two `ship` legs run
with no experiment variable exported are digit-identical to each other on both
fixtures, and every null arm reproduces them digit for digit.

This is an instrument. It must not be composed into a submission: it puts
runtime-configurable behaviour on the scored path.

**It has been reverted at the branch tip, in commit `63599dd0`.** Feedback F5
section 6 put `qwen35DerivedClusterProbeFraction` on loan to thorfinn for
tonight's composition, and `8e7a9620` had rewritten that exact declaration line
to read the resolver. The shipped behaviour was never changed — with the
variable unset the resolver returns `0.25`, and the `ship` legs reproduce the
shipped trajectory digit for digit — but it is textually the same line thorfinn
edits tonight, so leaving it at the tip was a merge hazard on the scored
surface.

After the revert, `git diff 328c4b9e HEAD -- Sources/ Vendor/ Package.swift Package.resolved mtp-head.manifest.json mtp-head/`
prints nothing. **This branch changes zero candidate bytes.** The instrument
stays reachable at `8e7a9620`, and every measurement in this report is
reproducible from that commit. Nothing in this report depends on the tip
carrying it, because no rider is proposed for submission.

## Evidence

- Host, instance, chip, memory profile, toolchain, thermal policy:
  `ip-10-231-2-227.ec2.internal`, Apple M4 Pro, 48 GiB.
  **Not a timing session.** Every leg runs the per-round phase trace, which
  writes a line per round, so `timing_valid=false` is recorded verbatim on
  every leg together with `cool_gate_passed_real_gate=false` and
  `gate_qualified_for_timing=false`. The measured channel is the round count,
  the effective mean draft length and the accepted draft rate, which are
  functions of the arm, the fixture and the token window alone and carry no
  thermal or scheduling content. No number in this report is a score.
- `head_provenance_sha256` for every leg, baseline and candidate:
  `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`, all 23
  legs, both fixtures, both arms and controls. `score.json.uses_pinned_mtp_head`
  is not used.
- Exact baseline and candidate commands:

```bash
# baseline (no experiment variable exported)
research/e139_session.sh ship benchfixture natural_history
E139_REP=2 research/e139_session.sh ship benchfixture natural_history

# riders
research/e139_session.sh fp32 benchfixture natural_history
for arm in p015 p010 p009 p008 p007 p006 p005 p004 p003 p002; do
  research/e139_session.sh "$arm" benchfixture natural_history
done

# offered-depth controls
E139_DEPTH=4 research/e139_session.sh ship benchfixture natural_history
E139_DEPTH=2 research/e139_session.sh ship natural_history

# pricing and publication
python3 research/e139_analyse.py --json research/e139-acceptance-channel.json
python3 research/e139_wandb_log.py --rung channel
python3 research/e139_wandb_log.py --rung ladder
```

- Cheapest real falsification gate and positive-control verdict. The gate is
  the instrument itself, run in both polarities before any rider was priced
  (CAMPAIGN RULE 101):
  - **positive polarity** — the same arm twice must be identical. `ship` × 2 is
    digit-identical on both fixtures, in round count, effective mean draft
    length and accepted draft rate.
  - **negative polarity** — one changed knob must be distinguished. Achieved on
    **three independent knobs**: offered depth (`8→4` moves benchfixture
    `78→110` rounds; `4→2` moves natural_history `260→269`), the fp32 tiebreak
    (distinguished on both fixtures at equal round count), and the probe
    fraction (distinguished at every rung from `0.08` down).
  - **the vacuous case is reported, not hidden.** `offer4` versus `offer8` on
    natural_history is VACUOUS: that fixture never drafts to depth 4, so the
    cap never binds. The tool labels it VACUOUS rather than counting it as a
    passing control.
  - **exact determinism condition:** same worker digest, same fixture, same
    token window, same seed path, and no environment difference that reaches
    the decode path.
- Tests and risk-based checks, in execution order:
  1. `senpai/rebuild-and-assert-worker.sh` with three `--require` witnesses and
     a `--forbid` control → all ok, 81,048 strings extracted, worker
     `a4b77789…`.
  2. positive-polarity determinism pair → identical.
  3. negative-polarity controls on three knobs → distinguished.
  4. 20 rider legs across 10 probe rungs and the fp32 rider → all exact.
  5. `python3 research/twin_audit.py` → OK, 29 runtime-effective twins.
  6. `senpai/validate-assignment-scope.sh` → OK.
  7. `senpai/check-editable-budget.sh 770a3ff2…` → within all three limits.
  8. `senpai/verify-ranked-score-boundary.sh` → PASS.
  9. `swift test --force-resolved-versions` → **755 tests in 69 suites, 41
     issues under 10 names, exit 1.** The documented floor is 40 issues under
     9 names. The tenth name is
     `theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax`
     (`Tests/MLXFastTests/E130WiredResidencySlackTests.swift:235`): *"bound A
     fails: slack 64 MiB is below the largest measured persistent growth 267.79
     MiB plus the measured page tax 0.9746208190917969 MiB."* **It is not
     mine.** The test reads `wiredZHDefaultSlackMB`, declared `64` at
     `Sources/MLXFastModel/Qwen36MTPBlockSession.swift:214`, against constants
     hard-coded in the test file. Neither file is in this branch's diff, and
     after the revert below the branch changes zero candidate bytes at all, so
     the failure is pre-existing on `BASE_SHA`. The other nine names match the
     documented set exactly, at 11, 11, 6, 3, 2, 2, 2, 2 and 1 issues. **The
     campaign's expected floor should be updated to 41 issues under 10 names**;
     a stale expected count will let the next real failure be read as known.
  10. `senpai/entry-point-cliff-census.sh --base 328c4b9e` → **PASS.** Every
      scored entry point unchanged; `applegpu_g16s` 32.000 → 32.000 derived
      simdgroups (+0.00 %), `applegpu_g17s` 40.364 → 40.364 (+0.00 %), 4 → 4
      pipelines.
  11. `senpai/rebuild-and-assert-worker.sh` two-sided on the reverted tip,
      `--require 'float(InT(reduced))'` and `--forbid` on both `MLX_E139_*`
      needles and the rider store → **PASS**. The worker was rebuilt from the
      reverted tree and re-extracted: `require 'float(InT(reduced))': 1`,
      `forbid 'MLX_E139_FP32_TIEBREAK': 0`, `forbid 'MLX_E139_PROBE_FRACTION':
      0`, `forbid 'exact_scores[candidate_base + r] = reduced;': 0`, over
      81,094 extracted strings. The reverted worker fingerprints
      `worker_sha256 4cbbb647a2386ea92ef6e6f5923b2ac50e286ccfaa844d4f79b92af1e6e5e4e7`,
      which differs from the instrument worker used for all 23 timed legs
      (`a4b777893ceb3c62eb88f84f8010b47eba23471fd9e8a883d4bea7a3cac7abb6`)
      exactly as it must. The `require` needle is the base-form rerank store,
      so this is a two-sided assertion: it proves the base code is present, not
      merely that the gate strings are absent.
- Exact-token and row-ledger verdict: **23 of 23 legs pass.**
  `all_tokens_matched=true`, `residual_divergence_count=0`,
  `decode_token_count=512`, `parity_all_ok=true`, and
  `declared_rows_total == reference_checked_row_total` on every leg.
  `dirty_candidate_paths=0` on every leg.
- Divergent tokens or failure category: none. Zero leg failures.
- **The exactness caveat that must not be misread.** `all_tokens_matched`
  compares the candidate stream against **candidate-generated** reference rows.
  It cannot by itself prove anything about the organizer's hidden reference.
  The target stream is unchanged by construction here because neither rider
  touches the target verification path; both act only on which draft tokens are
  proposed. Drafting is free to differ, and at the low probe rungs it does
  differ. Exactness of the emitted stream is not exactness of the draft
  proposals, and nobody should later read it as such.
- **EOS inside the window:** recorded verbatim — no fixture used in this
  experiment emits EOS inside its 512-token window, so post-EOS continuation
  was **not exercised** and is not claimed.
- Generated-twin audit: `python3 research/twin_audit.py` → OK, 29
  runtime-effective twins. No twin changed; neither rider adds, removes or
  edits a Metal kernel.
- Peak RAM or head/artifact size: not measured. No artifact size changed; the
  probe rider changes a loop bound and the fp32 rider changes one store.
- Official status and score: **not submitted.** Held per section 7 item 6 of
  the assignment.

## Results

### Rider A — fp32 rerank tiebreak

| Metric | Baseline (`ship`) | Candidate (`fp32`) | Delta |
| --- | ---: | ---: | ---: |
| rounds, benchfixture | 78 | 78 | 0 |
| rounds, natural_history | 260 | 260 | 0 |
| effective mean draft length, benchfixture | 6.3589743589743586 | 6.3076923076923075 | −0.0513 |
| effective mean draft length, natural_history | 2.2038461538461538 | 2.1884615384615387 | −0.0154 |
| accepted draft rate, benchfixture | 0.87701612903225812 | 0.88211382113821135 | **+0.5098 pp** |
| accepted draft rate, natural_history | 0.43979057591623039 | 0.44288224956063271 | **+0.3092 pp** |
| declared rows, benchfixture | 574 | 570 | −4 |
| declared rows, natural_history | 833 | 829 | −4 |
| **candidate-leg medpair gain** | 0.0 | **0.0000 %** | **0.0000** |

The mechanism is confirmed and behaves exactly as pre-registered: acceptance
rises on both fixtures, the round count does not move, and four proposals
change on each fixture against a predicted three and two. Its entire value is
four fewer verified rows and four fewer head steps per 512 tokens, which is
bounded between 0 % and 0.70 % in time and sits below the 0.5658 % local
session null measured in E136. **Confirmed, and worth zero on the round
channel. Do not compose.**

### Rider B — derived-cluster probe fraction

`net = 0.95 × gross_byte_pct_local − measured_acceptance_cost_pct`, in per cent
of candidate-leg medpair (CAMPAIGN RULE 118). `miss_wg` is the corpus screen's worst-gating probe miss
rate at that rung.

| p | probes | miss_wg | ΔR bf | ΔR nh | gross × 0.95 | net worst | net mean | verdict | screen `T0` |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 1844 | 1.089e-3 | 0 | 0 | 0.3233 | +0.3233 | +0.3233 | NULL | ok |
| 0.10 | 1230 | 2.613e-3 | 0 | 0 | 0.4848 | **+0.4848** | +0.4848 | NULL | ok |
| 0.09 | 1107 | 3.484e-3 | 0 | 0 | 0.5172 | **+0.5172** | +0.5172 | NULL | NO |
| 0.08 | 984 | 3.702e-3 | 0 | +1 | 0.5495 | +0.1649 | +0.3572 | SPLIT | NO |
| 0.07 | 861 | 4.355e-3 | +1 | +1 | 0.5819 | −0.7002 | −0.2515 | RESOLVED | NO |
| 0.06 | 738 | 5.444e-3 | +1 | +1 | 0.6142 | −0.6679 | −0.2192 | RESOLVED | NO |
| 0.05 | 615 | 6.751e-3 | +1 | +1 | 0.6466 | −0.6355 | −0.1868 | RESOLVED | NO |
| 0.04 | 492 | 1.067e-2 | +1 | −1 | 0.6790 | −0.6031 | +0.2302 | RESOLVED | NO |
| 0.03 | 369 | 1.568e-2 | +2 | −1 | 0.7113 | −1.8528 | −0.3785 | RESOLVED | NO |
| 0.02 | 246 | 2.613e-2 | +3 | +6 | 0.7437 | −3.1025 | −2.3333 | RESOLVED | NO |

**Recommended rung `p = 0.10`, worth `+0.4848 %` of candidate-leg medpair (CAMPAIGN RULE 118).**

`p = 0.09` is the measured argmax at `+0.5172 %` and I decline it, for three
stated reasons:

1. **Margin.** The first rung with any measured penalty is `p = 0.08` at
   `miss_wg = 3.702e-3`. `p = 0.09` sits only 6.3 % below that; `p = 0.10` sits
   29 % below it. Across eight hidden prompts, 6.3 % is not a margin.
2. **Gates.** `p = 0.09` fails `T0` and the repaired `T0b-leaf`, which are
   corpus-level statements about unseen prompts. Two live fixtures cannot
   overturn them.
3. **Prize against resolution.** The move is worth `+0.0324 pp`, which is 11.9x
   below this channel's own one-round resolution on natural_history. A
   sub-round debit on a hidden prompt would erase it and the instrument could
   never see it.

**Named reopening condition for `p = 0.09`:** measure it null on at least four
more fixtures spanning effective draft lengths from about 2 to about 6.4, or
produce a per-prompt margin distribution showing every prompt clears the
penalty threshold at that rung. The prize is fixed at `+0.0324 pp`.

Per feedback F2 the literal at `Qwen35.swift:4880` was transferred to thorfinn,
so this rung is his to land.

### The one-round quantisation limit, restated with every exact zero

`ΔR = 0` is exact for the leg that produced it, because the round count is an
integer and it did not move. It does **not** bound a sub-round debit. The
channel resolves no finer than one round: **1.2821 %** on benchfixture (78
rounds) and **0.3846 %** on natural_history (260 rounds).

## Calibration

### The receipt selects the estimator

At `p = 0.15` two independent rival receipts give a median-pair gain of
`+0.3866 %` and `+0.2603 %`, pooled `+0.3235 % ± 0.1140 (2σ)`. Candidate
predictions and the implied local-to-median transfer `k`:

| estimator | predicted | implied `k` |
| --- | ---: | ---: |
| gross byte model | 0.3403 | **0.951** |
| `predicted_pct_gating` | 0.3403 | **0.951** |
| `predicted_pct_pooled` | 0.2725 | 1.187 |
| `predicted_pct_absolute` / `_raw_miss` | 0.1193 | 2.712 |

A candidate-leg byte saving cannot be amplified by the ranked harness, so
`k > 1` is unphysical. The receipt therefore selects the realised-acceptance
estimator and fixes `0.95` as its transfer. The worst-gating absolute-miss
estimator over-charges by 4.7σ at this rung and must not price this family.
Reconstruction: predicted `+0.3233 %` against measured `+0.3235 %`,
`measured / model = 1.0007`, inside the rival 2σ band.

### `MISS_TO_SCORE_PCT` against live decode

`e133_screen.py:160` asserts `MISS_TO_SCORE_PCT = 203.0`. Four live points with
a nonzero screen loss:

| p | loss_wg | fixture | ΔR | cost % | implied `k` |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.02 | 1.132e-2 | benchfixture | +3 | 3.8462 | 339.8 |
| 0.02 | 1.132e-2 | natural_history | +6 | 2.3077 | 203.9 |
| 0.03 | 7.404e-3 | benchfixture | +2 | 2.5641 | 346.3 |
| 0.03 | 7.404e-3 | natural_history | −1 | −0.3846 | −51.9 |

Per-leg mean `k = 209.5 ± 93.1` (1σ, n=4, sample sd 186.3); ratio-of-sums
`k = 222.5`; asserted `203.0` → **0.07σ**. The constant is validated. The
per-leg estimator is not usable: the dispersion is about 90 % of the value.

## Findings recorded against the instruments

- **HARNESS DEFECT 38 (`senpai/rebuild-and-assert-worker.sh`).** Swift stores a
  string literal of 15 UTF-8 bytes or fewer inline in the instruction stream,
  so it never reaches the string table `strings` reads. A `--require` needle
  shorter than 16 bytes therefore fails a correct binary and a `--forbid`
  needle shorter than 16 bytes passes unconditionally — a guard that cannot
  fail. Fixed in `4ab839d3`: short string needles are refused.
- **HARNESS DEFECT 39 (`research/e133_screen.py:~1176`).** `recall` is computed
  from affine-2 scores over **probed rows only**, so it measures survivor
  retention conditional on the probed set and is identically `1.0` whenever the
  survivor width covers the probed row count. It cannot see a leaf miss. The
  screen's own output proves it: `recall = 1.000000` at all 19 rungs including
  `p = 0.005`, where the probe hit rate is `0.913763` and 658 live
  substitutions occur. The correct fields are `probe_hit_rate` and
  `m_absolute`. E136 rung 5a's "recall 1.000000 down to p=0.10", which the
  first two rounds of this assignment's feedback relied on, was the wrong
  column.
- **HARNESS DEFECT 40 (`research/e133_screen.py:1353`).** That defective field
  sits inside a kill rule, `passes_t0b = lowest("recall") >= 0.997`, which
  therefore cannot fail on this ladder and reads `ok` at `p = 0.005`. Added
  `passes_t0b_leaf` in `5c8b242e`, applying the same threshold to
  `probe_hit_rate`. Added rather than substituted, because narrow-width sketch
  families can genuinely drop a probed winner and an in-place substitution
  would silently move verdicts other agents have already recorded. On this
  family the repaired gate agrees with `T0` at all 19 rungs.
- **`research/e133_screen.py:2552-2562` crash (fixed in `454a24a8`).** The F3.3
  whitening block formatted `mean_delta_whitened_minus_plain`, `sign_test_p`
  and `best_clearing_*_predicted_pct` without `fmt_opt`, so a family with zero
  twin cells raised `TypeError` **after all measurement and before the `--out`
  write**, discarding a complete run.
- **HARNESS DEFECT 37 diagnosis (`--local-submit`), handed on rather than
  actioned.** Three source facts, from reading only:
  1. `exit_status=15` is self-inflicted. `workerExitDiagnostic()` at
     `Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift:2537-2551` sends
     SIGTERM to the worker and then reports `terminationStatus`. This rules out
     a jetsam SIGKILL.
  2. The true error is erased twice by `sanitizeWorkerDiagnostic` at
     `:2553-2563`.
  3. A discriminator nobody has tried. The identical reference step completed
     in about 60 s on this 48 GiB M4 Pro **with `MLXFAST_NO_SANDBOX=1`**.
     `benchmark-qwen-mtp.sh` runs it **without** that variable, and
     `writeRuntimeWorkerSandboxProfile` at
     `Sources/MLXFastCLI/main.swift:2605-2652` then writes a Seatbelt profile
     denying every file write except `/dev/null`. The cheapest next test is one
     `--local-submit` with `MLXFAST_NO_SANDBOX=1`. **I did not run it**, per
     feedback F1's instruction not to attempt `--local-submit` a third time.

## Second-order results

- **The realised debit is not monotone in `p`.** At `p = 0.03` benchfixture
  spends two extra rounds while natural_history saves one and its accept rate
  rises 1.69 pp. Each leg is exactly reproducible, so this is structure, not
  noise: the drafting trajectory is a discontinuous function of which leaves
  the probe keeps.
- **The live channel is a step function with wide plateaus.** benchfixture
  round count is monotone in `p` (`78,78,78,78,79,79,79,79,80,81` from 0.15 to
  0.02); natural_history is not (`260,260,260,261,261,261,261,259,259,266`).
  `p = 0.05` (615 probes) and `p = 0.06` (738 probes) are witnessed as
  different arms yet produce identical round counts, identical drafts through
  the rerank kernel (506 and 576), identical mean draft length and identical
  accept rate on both fixtures. 123 probed leaves change nothing. The corpus
  screen's 19-rung resolution is not real at the live level.
- **The screen's `T0` floor is conservative by exactly one rung, and correctly
  signed.** `T0` refused `p = 0.09`, which is live-null on both fixtures — a
  false refusal. `T0` refused `p = 0.08`, which does lose a round — a correct
  refusal. The live knee is `p = 0.08`.
- **Accepted draft rate is not a safe headline for a head-quality change.** At
  `p = 0.02` on natural_history the accept rate rises from 0.4398 to 0.4457
  while the round count rises by 6. The round proxy and the row proxy disagree
  in sign on that fixture: `+2.31 %` by rounds, `−1.80 %` by rows.
  `research/e139_analyse.py` reports both and does not collapse them.
- **Mechanism class.** The probe fraction removes bytes from every draft step
  of every round on every prompt, so it is not concentrated at any verify width
  and cannot be defeated by which two prompts occupy the median pair. Its
  relative effect is uniform in the **head's** time, though the head's share of
  total decode time is not uniform across prompts. The `0.95` is not an
  assumption of uniformity; it is the measured median-pair transfer from two
  real receipts, so whatever prompt-mix nonuniformity exists is already inside
  it. Width-concentrated mechanisms such as a verify-width-6 change are
  essays-weighted and must be priced by a different procedure.

## Pre-registration scorecard

| pre-registered before measurement | measured | verdict |
| --- | --- | --- |
| Rider A: acceptance rises 0 to +0.6 pp, `ΔR` 0 or −1, about 3 changed proposals bf and 2 nh | +0.5098 pp and +0.3092 pp, `ΔR = 0`, 4 and 4 | right |
| Rider B: cost exactly zero at `p = 0.15` and `p = 0.10` on both fixtures, identical round counts | exactly that | right |
| `p = 0.02` control: guaranteed to move, about 23 changed proposals | `ΔR = +3` and `+6` | right |
| `p = 0.03`: `ΔR = +1` bf and `+4` nh | `+2` bf and `−1` nh | wrong on nh |
| `p = 0.06`: `ΔR = 0` | `+1` on both | wrong |

Three right, two wrong. Both misses are the same error: I treated the corpus
miss rate as a smooth predictor of a discrete live quantity.

## W&B

| rung | run id | url |
| --- | --- | --- |
| live acceptance channel, 23 legs, 10 rungs | `han2y2jg` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/han2y2jg |
| corpus probe ladder, 19 rungs | `bzqaguve` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/bzqaguve |

Both runs carry `timing_valid=false`, `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false`
verbatim. Neither is a score. The ladder run logs the defective field under the
name `recall_wg_defective` beside `probe_hit_wg` so the two can never be
confused in a later query. `e139_composed_rider_median_pct` in `han2y2jg`
reports the **rider's** best measured value, not this experiment's composition,
which is empty.

## Conclusion

- **What happened and why.** The decode round count is exactly reproducible for
  a fixed arm, fixture and token window. That makes it a zero-variance
  instrument for the acceptance term, which is the term the offline corpus
  screen prices at zero and which destroyed E136 C1 by 18.4x. Both polarities
  of the instrument were proven before any rider was priced. Rider A moves
  acceptance in the predicted direction and does not move a single round on
  either fixture, so it is worth zero. Rider B is worth `+0.4848 %`
  **candidate-leg medpair** (`harness=ranked`, RULE 118) at the recommended
  rung `p = 0.10`, with a measured acceptance cost of exactly zero there and at
  `p = 0.15`.
- **Evidence for or against the mechanism.** The receipt reconstruction is the
  strongest single piece: an independent ranked candidate-leg medpair
  measurement of the exact edit at `p = 0.15` returns `+0.3235 % ± 0.1140`
  pooled over the two corrected receipt pairs, and this model predicts
  `+0.3233 %`. That is `measured / model = 1.0007`, and it simultaneously
  falsifies the worst-gating absolute-miss estimator at 4.7σ. The live decode
  then validates `MISS_TO_SCORE_PCT = 203.0` at `209.5 ± 93.1`, a factor of 7.5
  away in `p` from the receipt anchor.
- **Prompt or M5 transfer risk.** Two fixtures is a small sample of a
  prompt-level quantity that this experiment has shown to be discrete and
  non-monotone. The `ΔR = 0` at `p = 0.10` is exact for these two fixtures and
  says nothing about a sub-round debit on a hidden prompt; the channel cannot
  resolve one. This is the whole reason `p = 0.09` is declined despite being
  the measured argmax. Machine transfer risk is low for this family: nothing
  here is a timing measurement, and round counts are architecture-invariant
  given the same weights and arithmetic.
- **Smallest useful next action.** Run `p = 0.09` and `p = 0.10` on four more
  fixtures spanning effective draft length 2 to 6.4. That is six legs of GPU,
  about 26 minutes, and it either buys `+0.0324 pp` of candidate-leg medpair or
  closes the question permanently. Separately, and cheaper: one
  `--local-submit` with `MLXFAST_NO_SANDBOX=1` would test the only untried
  discriminator for HARNESS DEFECT 37, which currently blocks the pre-submit
  chain for the whole campaign.
- **Recommendation: close.** Merge the tooling and the three instrument
  repairs. Do not compose either rider from this branch: rider A is worth zero,
  and rider B's literal belongs to thorfinn, who should take `p = 0.10` and
  should know that `p = 0.09` is measured-null on two fixtures with a `+0.0324
  pp` prize behind a 6.3 % margin. Decide whether to keep or strip the Vendor
  instrument gates; they must not reach a submission unchanged.
