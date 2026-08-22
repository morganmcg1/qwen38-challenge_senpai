
# E133 terminal result — C1 is priced, not killed, at `qlowrank256-N4096-p0.35` for +0.678 %. The base miss is a bfloat16 tie, so two other levers are dead.

- **Student / branch:** qwen-askeladd / `qwen-askeladd/e133-sketch-first-draft-readout-offline-screen`
- **Hypothesis and target cost:** replacing the two exact affine-2 matvec passes over the compact draft vocabulary with a compact sketch, keeping the survivors and rescoring them exactly, keeps the exact top-32 shortlist at a byte cost worth about 1 % of the ranked leg. Target cost is the 59,093,760 B draft-readout stage, 18.26 % of the 323.59 MB reconciled per-draft-step budget.
- **Decision:** **green offline** — C1 survives the screen with a specified cell and a full implementer's brief. No scored-surface change was made, so this is not a timed or ranked result.
- **`BASE_SHA`:** `197e0550ab46842b639a4ff4fe3f4889ca3b01ec` (`senpai/qwen38-mtp-r1`)
- **`UPSTREAM_SHA`:** unchanged by this assignment; no organizer sync was performed.
- **Candidate commit:** the head of this branch. There is no candidate build to fingerprint against it, because the tree is byte-identical to `BASE_SHA` on every submitted path.
- **Yukon promoted submission / frontier:** not queried and not needed. This assignment changes zero submitted-surface bytes and forbids an official submission.
- **Candidate build fingerprint:** `cli_sha256=41a568caf74345a4c1757f96bc64cc38d0b73087566e2ffde3e1667e82ca0262`, `worker_sha256=8817a3a0f16e3311feae7e697bce2399dce01b30dfcc2d22a2da56f2ad17b75a`, metallib `2050ebf1...`. This build was used only to capture hidden states in rung 1. No timed leg used it.
- **Submitted-surface / generated-twin / metallib digests:** unchanged. `research/twin_audit.py` reports 29 runtime-effective twins OK.
- **Submitted candidate files:** **none**. Zero submitted-surface bytes changed.
- **Supporting test, tooling, or documentation files:** 26 files, all under `research/`. The main ones are `research/e133_screen.py`, `research/e133_index.py`, `research/e133_capture.sh`, `research/e133_rebuild.sh`, `research/e133_job.sh`, `research/e133_wandb_log.py`, `research/e133_summarise.py`, `research/e133-corpus.json`, `research/e133-validate.json`, `research/e133-attrib.json`, `research/e133-screen.json`.
- **MTP head provenance, digest, and draft policy:** organizer-pinned head, unchanged. No head was declared. Capture used depth 8 with the shipped derived-index draft policy.
- **Token window, fixture, reference source, and harness:** 512-token windows over 22 self-generated E124-domain seeds; reference rows are the parent's own `row_ledger`; harness is **offline**, not `local` and not `ranked`. No benchmark leg ran.
- **Exact cell:** the C1 draft-readout stage of `draftTokenIDWithDeclaredRerank`: stage A `qwen_mtp_cluster_centroid_qmv_a2g64_v1` over 12,292 centroid rows, stage B `qwen_mtp_cluster_row_qmv_a2g64_v1` over 24,584 probed rows, both affine-2 group-64 at K = 5,120 and 1,600 B/row, followed by the 32-row exact affine-4 rerank. M5 variant not applicable: the screen is a numerical model of the readout, not a kernel measurement.
- **Official causal path and score equation:** `harness=ranked`. The candidate edit lies entirely inside the candidate MTP leg, so `d ln(ranked baseline serial time)/dx = 0` and any byte removed lowers candidate seconds per token directly. Price is `pct = MB_removed_per_step x 0.02167 %/MB - net_miss x 203.0 pp`. No local serial share is subtracted anywhere.
- **Assignment-scope preflight:** see Evidence.
- **Editable source bytes / headroom / growth / exempt-head bytes:** see Evidence. `research/` is outside the submitted surface, so this assignment adds zero submitted bytes.
- **Scored-path reachability evidence:** rung 2 proved the offline model reproduces the shipped runtime chain on 99.20 % of 11,244 real draft rows, and reproduces the runtime proposal exactly on 99.72 %. The 32 mismatches all sit at `rank_max = 1`, i.e. exact ties broken differently. Rung 4 then showed that those ties are the dominant feature of the readout: the exact affine-4 head emits `bfloat16`, and 1.1 % to 2.6 % of positions carry a two-way tie at the maximum.

## Evidence

- **Host, chip, memory, toolchain, thermal policy:** Mac16,11 Apple M4 Pro, 20 GPU cores, 48 GiB, W&B host tag `apple-m4-pro-applegpu_g16s-20core-48gib`. Rung 1 took the process lock and the real 40 C cool gate. Rungs 2, 3 and 4 are CPU-only numpy and MLX-on-CPU analysis and have no thermal policy. `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, `official_or_ranked_score=false`, `timing_valid=false` on every published run.
- **`head_provenance_sha256` for every leg:** not applicable. No baseline or candidate benchmark leg ran under this assignment, so there is no `score.json` and no head-provenance pair to report. The capture used the organizer-pinned head with no declaration.
- **Exact baseline and candidate commands:**
  ```
  research/e133_job.sh research/e133_capture.sh all --steps 512 --depth 8
  python3 research/e133_screen.py validate
  python3 research/e133_screen.py attrib --batch 32
  python3 research/e133_screen.py screen --group-size 4 --top 25 \
      --base-sha 197e0550ab46842b639a4ff4fe3f4889ca3b01ec \
      --out research/e133-screen.json --out-full ~/e133-screen-full.json
  python3 research/e133_screen.py selftest
  python3 research/e133_summarise.py
  python3 research/e133_wandb_log.py --rung 1|2|3|4
  ```
- **Cheapest real falsification gate and positive-control verdict:** two controls, in both directions.
  - `exact0`, an arm that scores stage A and stage B with the shipped affine-2 values themselves and differs from the base only by running through the arm pipeline. **77 cells, 0 anomalies**: net miss exactly 0.000000e+00, `m_incremental` exactly 0, recall exactly 1.000000, `m_absolute` exactly 8.663123e-03 on beagle, bit for bit equal to the shipped baseline.
  - `simhash8`, a deliberately damaged 8-bit sketch. It loses the argmax on 11,180 of 11,244 rows, p = 0.9943, `control_can_fail=true`.

  Together these prove the comparison can fail and does not fail spuriously.
- **W&B runs.** Every rung is published. Two runs were superseded during the assignment and both stay listed so the record is complete.

  | rung | run | URL |
  |---|---|---|
  | 1, corpus capture | `7ppyblde` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/7ppyblde |
  | 2, validation (current) | `y8bs5w33` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/y8bs5w33 |
  | 2, validation (superseded by the base-miss fix) | `qn4euyez` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/qn4euyez |
  | **3, cell sweep (current, 2,156 cells)** | **`nl31frlw`** | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/nl31frlw |
  | 3, cell sweep (superseded, 1,176 cells, 3 probe fractions) | `btb5bzz3` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/btb5bzz3 |
  | 4, base-miss attribution | `blldv9cd` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/blldv9cd |

  Rung 3 carries seven tables: `screen_cells` (2,156 rows), `screen_by_stratum`, `screen_survival`, `screen_spectrum`, `screen_byte_ladder`, `screen_probe_ladder`, `screen_whitening`. Rung 4 carries three: `attrib_by_stratum`, `attrib_k_curve`, `attrib_probe_curve`. Every run sets `harness=offline`, `timing_valid=false`, `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, `official_or_ranked_score=false`.

- **Tests and risk-based checks, in execution order:** see the check block below.

### Preflight checks

Run after reverting the rung-1 instrument patch, which is commit `608bdc3d` reverted by `50448a17`. After the revert the diff against `BASE_SHA` touches 26 files and **0 of them are outside `research/`**.

```
python3 research/twin_audit.py
  TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)   exit 0

senpai/verify-ranked-score-boundary.sh
  PASS: ranked numerator is pinned baseline; candidate edits affect the MTP
  denominator only                                                                     exit 0

senpai/validate-assignment-scope.sh 197e0550... <all 26 changed paths>
  all 26 reported "outside benchmark.json editablePaths"                               exit 1

senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf
  editable budget OK: source=2594084/3000000 headroom=405916 growth=139249/262144
  exempt=2410 files=154                                                                exit 0

senpai/entry-point-cliff-census.sh --base 197e0550...
  verdict: PASS, 8 scored entry points, 0/8 register movement,
  g16s 32.000 -> 32.000 derived simdgroups (+0.00 %),
  g17s 38.000 -> 38.000 (+0.00 %)                                                      exit 0

swift test --force-resolved-versions
  737 tests in 67 suites, 9 failing names, 40 issues                                   exit 1

python3 research/e133_screen.py selftest
  SELFTEST PASS                                                                        exit 0
```

**`swift test` reproduces the documented floor exactly and adds nothing.** 9 failing names and 40 issues, against the 9 names and 40 issues recorded in `senpai/known-test-failures.md`. I compared the observed set against the documented set by name, not only by count:

```
documented 9   observed 9
added   (observed, not documented): none
missing (documented, not observed): none
```

The nine are `theCheckedInDeclarationSelectsThePinnedHead`, `startupMemoryPolicyKeepsRanked128GiBProfile`, `qwen36ConfigContractDigestMatchesTheReferenceManifest`, `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`, `theSeededCalibrationExpectationMatchesItsRecordedProvenance`, `theQwenMTPTrackIsArmedOnQwen38`, `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen`, `participantDocsExposeDefaultCLIInstallDirectory` and `submissionStaticReviewPromptCoversMeasurementStructureExploitation`. **Zero added.** The `swift test` run predates this session's `research/`-only edits, and `research/` is not compiled by SwiftPM, so the result stands.

**On the scope script's exit 1.** It validates *proposed submitted paths*, so a non-zero exit means the paths are not on the submitted surface. This assignment proposes zero submitted paths, so exit 1 with all 26 files reported outside is the required result, not a failure. The independent confirmation is the budget line: `source=2594084`, `headroom=405916`, `growth=139249` are byte-for-byte identical to the figures the assignment quotes for the base. **Zero submitted bytes changed.**

- **Exact-token and row-ledger verdict:** no tokens were generated by a candidate, so there is no exact-token verdict to give. The row-ledger check that does apply is the capture alignment: `align_shard()` asserts token-by-token equality between each instrument shard and the parent's `row_ledger`, and all 22 seeds pass. `ledger_verdict_factors_as_live_and_match` is exactly 1.0, i.e. acceptance factorises as `prefix_live AND (token == reference_token)` on every one of the 11,244 rows.
- **Divergent tokens or failure category:** none. No divergence is possible; nothing timed or scored ran.
- **Generated-twin audit:** `research/twin_audit.py` → `TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)`, exit 0.
- **Peak RAM or head/artifact size:** the sweep peaks under 6 GiB of host RAM and takes 656.8 s for 2,156 cells. The capture corpus is 11,244 x 5,120 fp32 = 230 MB plus ledgers, held outside Git at `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e133/`. `~/e133-screen-full.json` is the full un-slimmed sweep, also outside Git.
- **Official status and score:** not submitted. The assignment forbids it and there is nothing to submit.

### Metrics

The template's timing table does not apply: this assignment ran no timed leg. The measured quantities are these. The headline price uses `m_absolute` against the exact affine-4 global argmax, as advisor error 125 requires.

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| **`e133_best_cell_ranked_pct` (primary, maximize)** | 0.0 | **0.678228** | +0.678228 |
| `e133_worst_domain_net_miss_rate` (secondary, minimize, kill 3.0e-3) | — | **6.53310e-04** | 4.6x inside the gate |
| best cell recall into sketch top-N (T0b ≥ 0.997) | — | 0.999564 | +0.002564 |
| draft-readout stage bytes per step | 59,093,760 | 21,610,464 | −37,483,296 |
| stage bytes removed per draft step | 0 | 37.483 MB | 11.584 % of the 323.59 MB step |
| gross gain %, plain byte-and-rate | 0.0 | 0.636 | +0.636 |
| gross gain %, 7 % head share | 0.0 | 0.811 | +0.811 |
| gross gain %, 9 % head share | 0.0 | 1.043 | +1.043 |
| predicted ranked %, absolute price at a 9 % head share | 0.0 | 0.910 | +0.910 |
| predicted ranked %, F1.5 incremental price | 0.0 | 0.767 | +0.767 |
| predicted ranked %, pooled price | 0.0 | 0.714 | +0.714 |
| predicted ranked %, raw-miss bound | 0.0 | 0.590 | +0.590 |
| cells clearing T0 and T0b | 0 / 2,156 | 893 / 2,156 | full C1 398/980, hybridA 415/980, no-centroid 80/196 |
| gate recomputation disagreements | — | 0 | independent recheck in the W&B logger |
| shipped-chain `m_absolute`, beagle | 8.663123e-03 | — | 44 misses of 5,079 |
| of which are an exact bfloat16 tie | — | 42 of 44 | 95.5 % |
| recovery available from `p → 1.00` | — | 0.000e+00 | costs 0.351 pp |
| recovery available from `K → 2,048` | — | 0.000e+00 | costs 0.126 pp |

The primary metric is `predicted_pct_absolute = pct_head_share_7 - 203.0 x net_miss_worst_gating = 0.810850 - 203.0 x 6.533101e-04 = 0.678228`. `net_miss_worst_gating` is the difference of `m_absolute` between the arm and the shipped chain on the worst gating stratum.

Every compared identity field matched: same base, same corpus, same 11,244 samples, same host, same interpreter, same seed. The only dimension that varies across cells is the sketch family, size, survivor width, probe fraction, and stage-A mode. All prices are **inferred** from a byte model and Finding 69's exchange rate, not measured on a GPU; that inference is labelled everywhere it appears.

## Conclusion

### What happened and why

Rung 0 found the E87 corpus absent on this Mac, so rung 1 recaptured it: 22 seeds, 11,244 draft rows, 5,079 `beagle` / 4,592 `min_carriers` / 1,573 `zero_weight`, each gating stratum over the 4,000-sample floor F1 required. Rung 2 validated the offline model against the runtime. Rung 3 swept 2,156 cells. Rung 4 attributed the shipped chain's own miss.

**C1 is not killed. 893 of 2,156 cells clear both T0 and T0b.** The best-priced cell is `qlowrank256-N4096-p0.35`: query-fitted int8 low-rank at r = 256, 264 B/row, survivor width 4,096, probe fraction 0.35. It removes 37.483 MB per draft step for a byte-rate gain of 0.811 % and pays back 0.133 pp in lost acceptance, netting **+0.678 %** on the absolute-miss price and **+0.767 %** on the incremental price.

**The larger result of this assignment is not the cell. It is that the shipped chain's own miss is a bfloat16 tie, and that two levers the campaign was about to pull are therefore worth exactly zero.**

Rung 4 split the base miss into probe loss `P`, centroid-precision loss `C`, and everything downstream `R`, with `causes_sum_to_base_miss = ok` on every stratum.

| stratum | n | `m_absolute` | `P` | `C` | `R` |
|---|---:|---:|---:|---:|---:|
| beagle | 5,079 | 8.6631e-03 | **0.0** | **0.0** | 8.6631e-03 |
| min_carriers | 4,592 | 6.7509e-03 | **0.0** | **0.0** | 6.7509e-03 |
| zero_weight | 1,573 | 5.7216e-03 | **0.0** | **0.0** | 5.7216e-03 |
| essays_bacon | 569 | 8.7873e-03 | **0.0** | **0.0** | 8.7873e-03 |

`probe_hit_rate_affine2 = 1.000000` everywhere, so the probe stage never loses the answer. `R` splits again into rank loss, tie-straddle, and rerank loss: on beagle that is 1, 1 and 42 of 44. `R_rerank` means the row certainly reached the exact rerank and the rerank still returned a different row, which only an exact tie can produce. Measured directly: `exact_score_dtype = mlx.core.bfloat16`, **42 of the 44 beagle misses have gap exactly zero**, and 82 of the 84 misses pooled over the three disjoint strata do. 1.1 % to 2.6 % of positions carry a two-way tie at the maximum and the chain loses roughly half of them.

This reconciles the ledger. Replaying E87's own strict-rank shortlist-survival statistic on the E133 corpus gives `1.9689e-4` on beagle and `0.0` elsewhere, against the ledger's worst-domain `2.266e-4`. **The ledger number reproduces to within one sample.** Both numbers are correct and they measure different quantities: E87 asked whether the argmax survives the shortlist, and it does; I asked what the chain returns, and 98 % of the difference is tie-break order.

Three consequences, each measured rather than argued.

**The probe fraction is dead as an accuracy lever.** `exact0` at p = 0.25, 0.35, 0.50, 0.75, 1.00 gives `8.6631e-03` on beagle at every rung, flat to the last digit, with probe hit 1.000000 at every rung. Probing everything costs 0.351 pp of gross gain and recovers `0.000e+00`. The break-even recovery of 1.7e-3 is unreachable because the maximum achievable recovery is zero.

**The shortlist width is dead.** `m_absolute(K)` for K in {32, 64, 128, 256, 512, 1024, 2048} on the shipped chain recovers `0.0` on the worst gating stratum at every K, while the cost runs from 0.002 pp to 0.126 pp. `K = 64` buys the two beagle rows lost by rank and tie-straddle and then the curve is flat forever. The argmax of the priced curve is `K = 32`, the shipped value. **I recommend no change to `Qwen35.swift:3799`, which also means no work lands in thorfinn's file.** The strict-rank lower bound and the non-strict upper bound both collapse to `0.0` from K = 64 upward while the realised miss does not move, which is the clean proof that the residual is tie resolution.

**A perfect readout is worth about zero, measured.** On the live missed rows, substituting the exact affine-4 argmax for the shipped output never raised acceptance and lowered it on three of four strata: beagle −1.9689e-04 on 34 live rows, min_carriers −1.5244e-03 on 26, zero_weight −6.3573e-04 on 6, essays_bacon −1.7575e-03 on 5. That is what the tie finding predicts: on an exact tie the reference token is itself settled by the runtime's tie order, so "the argmax" is no better a guess than the row the chain already returns. With 5 to 34 live rows per stratum I do not claim the true value is negative. I claim it is not distinguishable from zero and is certainly not the +1.76 pp that `203 x m_absolute` implies. **`203 x m_absolute` is an upper bound that this measurement refutes as an estimate.**

**The sketch's own incremental loss is different, and that is why C1 still stands.** Asked the same question, the selected family's arm-only misses are **0 of 11 ties**. Every one is a real score gap. The sketch makes genuine readout errors that the base chain does not, its measured acceptance cost is small and mixed in sign, and unlike the base miss it is a quantity the byte-for-accuracy trade can actually move.

Four of the assignment's design decisions were falsified by the measurements they asked for.

**The cheapest-clearing-cell selection rule gives away 0.290 pp.** Byte-rate gain falls monotonically with size, 1.238 % at 40 B/row down to 0.337 % at 1,032 B/row, exactly as the corrected payout curve says. The miss penalty falls faster, so the product peaks in the interior. The cheapest clearing candidate at 40 B/row is worth +0.388 %; the best-priced cell at 264 B/row is worth +0.678 %. I report the whole ladder rather than one cheapest cell.

**D5's N = 256 survivor width gives away 0.501 pp, which is still the larger error.** Survivor width is a first-order axis that the design treated as fixed, and it trades directly against sketch size: a wider shortlist tolerates a coarser sketch. Constrained to N ≤ 256 the best clearing candidate is worth +0.177 %; unconstrained it is +0.678 %.

**The probe fraction is a live lever for the sketch even though it is dead for the exact chain,** and the design fixed it at 0.25. Stage A of the sketch is approximate, so it can drop the answer's leaf out of the probe set, which the exact affine-2 centroid score never does. Moving p from 0.25 to 0.35 on the selected family raises beagle probe hit from 0.99961 to 0.99980 and min_carriers from 0.99869 to 0.99956, lowering the worst-gating net miss from 1.5244e-03 to 6.5331e-04. It costs 0.056 pp of gross gain and buys 0.177 pp of acceptance, **net +0.120 pp**.

**Whitening loses, in the direction I pre-registered before the run finished.** Over 462 exactly-paired twin cells, whitened int8 loses on the absolute price 137 times against 66 wins with 259 ties, sign-test p = 6.983e-07, mean delta −0.027 pp. The selftest proves the whitened estimator is algebraically identical before rounding (rel. err 3.1e-16), so the whole difference is quantization. With one per-row max-abs scale the score error variance is `(max_j|a_j t_j|/254)^2 · Σ_j λ_j/t_j^2`, minimised at `t_j = 1`; whitening picks `t_j = sqrt(λ_j)` and inflates the scale everyone pays. The trick is right for a per-coordinate scale and wrong for a per-row scale. `wlowrank` is retained as a measured control and dropped from the candidate set.

Two further results matter for the implementer.

**Captured query energy does not predict the miss.** At only 45.6 % held-out captured energy, a rank-64 basis already reaches net miss 2.178e-04 with recall 0.99961. A spectral argument would say that is impossible. It is possible because the sketch only orders a shortlist of thousands of rows and never has to get the score right — stage C rescores exactly. Rank preservation at coarse resolution is much easier than score preservation. This is the single most useful thing the screen learned and it is why C1 is cheaper than the design assumed.

**The base was not exact and my first instrument did not see it.** `screen.shipped()` originally returned a structural proxy that undercounted the shipped chain's true miss by 44x. Subtracting the 1.97e-04 proxy instead of the 8.663e-03 truth inflated every net figure by about 8.5e-03 and produced a false `0/252` kill verdict, which I retracted in interim 4. Rung 4 then explained the 44x: it is the tie population and nothing else.

### Evidence for or against the mechanism

For: 398 clearing full-C1 cells and 415 clearing hybridA cells across every byte rung from 40 to 1,600 B/row; the `exact0` control clean on 77 cells; the damaged control failing as it must; the `essays_bacon` holdout better than the near-neighbour cross-fit on the selected cell; the arm's incremental loss confirmed to be genuine score error rather than the tie artefact that dominates the base.

Against, or at least limiting: on `essays_bacon` the selected cell's cross-fit net miss is 3.5149e-03, which is above T0. `essays_bacon` is a watch stratum inside `min_carriers` and T0 gates on beagle and min_carriers, so the cell clears as specified, and the holdout figure is 1.7575e-03 with recall 1.00000. Both are 2 and 4 rows in 569 and neither is precise. The `p = 0.25` cell carries the identical pair, so this is a property of the family at 264 B/row and not of the probe move. It is the one place where the selected cell is not comfortable.

Also limiting: the whole price is a byte model. If the true price of a miss is the raw-miss bound, the arm is worth +0.590 % rather than +0.678 %; if it is the F1.5 incremental price, +0.767 %. The band is 0.59 to 0.77 and the point is not the honest answer.

### Prompt or M5 transfer risk

Moderate and named.

1. **The price is a byte model, not a measurement.** Nothing here ran on a GPU. The 0.02167 %/MB coefficient and Finding 69's 203.0 pp per unit miss both carry their own uncertainty, and neither was re-derived by me.
2. **The query basis is domain-specific.** Held-out k256 keeps 0.59 to 0.61 against 0.83 in-fit. The corpus is 22 self-generated seeds covering all eight ranked domains — beagle 5,079, medicine 2,019, essays 1,590, plutarch 563, travel 543, botany 506, republic 477, drama 467 — but the hidden pool is eight prompts I cannot see, and four of those domains carry under 600 samples each. The `essays_bacon` holdout is the strongest evidence I can produce against specialization and it is reassuring on recall, but it is n = 569 with wide intervals. The basis-free fallback costs 0.253 pp and removes this risk entirely.
3. **`beagle` carries 97.9 % of the ranked median** and has the lowest acceptance. It is the worst gating stratum on absolute miss and it is where I gate. That is the right choice, but it means the gate is set by the noisiest stratum.
4. **Dispatch budget goes 7 to 8, at a survivor width 16x wider than D5 sized for** (D5 hazard 5). The screen cannot see dispatch overhead at all. F4 §4 supplies a selection plan that keeps the `:4057` precondition untouched at `N = 4,096, T = 32`, which lowers this risk considerably, but it is still unmeasured. If the widened plan costs anything, `N = 8,192` at 136 B/row is within 0.027 pp of the peak and halves the per-row bytes.
5. **The tie population is a measured property of this build on this host.** `bfloat16` scores at 98,330 rows produce ties at a rate that depends on the score distribution. I did not test whether the M5 build's rerank kernel accumulates identically. Nothing in the selected cell depends on it, but the "a perfect readout is worth zero" conclusion does.

### Smallest useful next action

Implement `qlowrank256-N4096-p0.35` behind an env gate in `Qwen35.swift` and measure one matched absolute-candidate-time pair against a fresh unchanged `BASE_SHA` run. That converts the byte model into a measurement and settles risks 1 and 4 together. It needs thorfinn's file, so it is the advisor's to schedule.

### Recommendation

**Compose later.** C1 is priced and specified, not implemented. Hand the brief below to whoever owns `Qwen35.swift` when E129 lands. Do not spend a slot on the probe fraction or on `qwen35Top32K`.

## Rung 5 — implementer's brief

**Cell.** Family `qlowrank`: query-fitted PCA basis, int8 codes, no whitening. Rank **r = 256**. Survivor width **N = 4,096**. Probe fraction **0.35** (4,303 of 12,292 clusters, 34,424 probed rows). Stage A sketched as well as stage B, i.e. full C1 rather than hybridA.

**Bytes.**

```
per row                       264 B  = 256 int8 codes + 1 fp32 norm + 1 fp32 row.mu
sketch A  12,292 centroid rows      3,245,088 B
sketch B  34,424 probed rows        9,087,936 B
exact C   4,096 survivors affine-2  6,553,600 B   (was 32 x 1600 = 51,200)
projection R[5120,256] bf16         2,621,440 B
mu[5120] bf16                          10,240 B
affine-4 rerank, 32 rows               92,160 B   unchanged, charged in both columns
--------------------------------------------------------------------
stage total today                  59,093,760 B
stage total under C1               21,610,464 B
removed per draft step             37,483,296 B = 37.483 MB = 11.58 % of the 323.59 MB step
```

**Resident cost is larger than D4 assumed.** The sketch must cover every leaf row, not only the probed ones, so it is `(98,336 + 12,292) x 264 = 29,205,792 B` plus `R` and `mu`, i.e. **31.84 MB**. That is **47.4 % of the 64 MiB residency slack** at `Qwen36MTPBlockSession.swift:212`, not the 24 % D4 projected at 132 B/row. It still fits, but it consumes nearly half the slack, and alphonse is separately arguing that slack is undersized by construction. Coordinate before implementing. `wireResidentWeightsIfEnabled()` still picks it up automatically because it sizes from live `Memory.activeMemory` after the warm.

**Price.** Byte-rate gain **0.811 %** at a 7 % head share, 1.043 % at 9 %. Acceptance cost `6.5331e-04 x 203.0 = 0.133 pp`. Predicted ranked **+0.678 %** on the absolute price, **+0.767 %** on the F1.5 incremental price, **+0.714 %** pooled, **+0.590 %** on the raw-miss bound. The band, not the point, is the honest answer.

**Basis provenance, to be recorded verbatim in a source comment as F3.1 condition 3 requires.** Fitted on the query second moment `E[(x-mu)(x-mu)^T]` over hidden states captured from 22 self-generated seeds in the E124 prompt domains, 11,244 samples, at base `197e0550`. Cross-fitted: no gating stratum is ever scored by a basis that saw it. Frozen at build time, no run-time update path, shortlist only, never entering the exact rerank, the target verification, the row ledger, or the top-two evidence.

**D5 hazards this cell actually touches.**

- **Hazard 1, `perm` bypass — TOUCHED, and it is the live one.** My screen hit exactly this defect: 98,336 padded positions map to only 98,330 distinct rows, and comparing permuted positions instead of compact ids silently counts six phantom rows. Every comparison in the screen runs on compact ids. Reuse `:5650-5657` verbatim; do not index the sketch table by a raw permuted index.
- **Hazard 2, integer ties — TOUCHED, and rung 4 raises its priority.** The exact affine-4 head emits `bfloat16` and 1.1 % to 2.6 % of positions carry a two-way tie at the maximum. Whatever tie order the new path uses will change which of two equal-scoring rows is emitted on roughly 1 in 70 positions. Break on index with the `qwen_top32_ordinal` packing at `:3827-3836` and expect the exactness tests to notice.
- **Hazard 3, no silent fallback — TOUCHED.** Follow the `fatalError` pattern at `:5558-5561`.
- **Hazard 4, bit-identity tests — TOUCHED.** `QwenDraftReadoutExactnessTests.swift:939-1012` will fail by design. Add a new arm with a recall gate; never relax the exactness tests.
- **Hazard 5, dispatch budget 7 to 8 — TOUCHED, and D5's arithmetic does not carry, but F4 §4 repairs it.** D5 sizes the top-32 plan at N = 256. The clearing cells need much wider survivor sets. At `N = 4,096`, `cands = 32 * 4,096 = 131,072`, 16x D5's figure. The advisor's F4 §4 plan avoids a redesign: a three-dispatch histogram threshold for selection 1, then a per-threadgroup top-32 union merge for selection 2, which at `N = 4,096, T = 32` gives each group 128 rows and keeps `finPerThread <= 32` so the `:4057` precondition holds untouched. `qwen35Top32K` at `:3799` still moves from a global into a plan field. The screen cannot price the dispatch cost, so this is still the largest single implementation risk.
- **Hazard 6, legality — SATISFIED** subject to the F3.1 provenance comment above.

**What not to do.** Do not whiten and do not use `wlowrank`. Do not pick the cheapest clearing cell; pick 264 B/row. Do not use a bf16 `R` at r = 1024 — that alone is 10.49 MB per step and it is the D3 trap. Do not widen `qwen35Top32K` and do not raise the probe fraction as a way to buy accuracy on the exact chain; both recover exactly zero.

**Ordered fallbacks, if the chosen cell cannot be built.**

| if | use | predicted % (absolute price) | why |
|---|---|---:|---|
| the widened plan costs any dispatch time | `qlowrank128-N8192-p0.5`, 136 B/row | +0.651 | 0.027 pp behind, halves the per-row bytes |
| the top-32 plan cannot widen past N = 1,024 | `qlowrank512-N1024-p0.25`, 520 B/row | +0.491 | costs 0.187 pp against the first choice |
| the plan is stuck at N = 256 | `qlowrank1024-N256-p0.25`, 1032 B/row | +0.177 | drags in a 10.49 MB bf16 `R`; check it still pays |
| the query fit is disallowed | `lowrank256-N8192-p0.5`, 264 B/row | +0.425 | net 7.876e-04, recall 0.99921 |
| stage A must stay exact (hybridA) | `qlowrank256-hybridA-N4096-p0.25`, 264 B/row | +0.468 | net 2.178e-04, recall 0.99956 |

## Suggested follow-ups, not implemented

1. **Measure the widened top-32 plan.** A microbenchmark of `Qwen35Top32Plan` at N = 4,096 and at N = 8,192 against today's N = 32 would settle transfer risk 4 and the D5 hazard-5 correction together, without touching the readout at all. This is the cheapest way to convert the largest remaining unknown into a number, and it needs no capture corpus.
2. **The exact affine-4 rerank emits bfloat16, and that is an unowned lever.** 82 of the 84 shipped-chain misses are an exact tie at the maximum. Nothing in this report can price a change of tie order, because the reference token is itself produced by the runtime's own tie order. The cheaper and better-posed question is whether an fp32 accumulation in `qwen35DraftSelectedAffine4RerankKernel` separates those ties and changes acceptance. It touches 32 rows per step, so it costs almost nothing in bytes, and nobody is looking at it.
3. **Deepen the thin median carriers, not the domain list.** The corpus already covers all eight ranked domains. The gap is depth, not coverage. `republic` at 477 and `botany` at 506 are B2 median carriers thin enough that a 3.0e-3 miss gives about 1.5 expected events, so neither can gate on its own. Another capture pass on those two would promote them from watch strata to gating strata. The sweep itself costs 657 s of CPU, so the cost is the GPU capture, not the analysis.
4. **Re-derive the 0.02167 %/MB coefficient.** Every price in this report is linear in it and I inherited it rather than measuring it.

_This report was created by an AI agent (OpenHands) on behalf of the qwen-askeladd research student role._
