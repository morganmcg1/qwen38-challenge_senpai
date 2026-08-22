# E124 — delete the MTP head precision islands and price the acceptance exchange

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"median_regime_corpus_seeds_passing_criterion","available":true,"value":3},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}
```

- **Student / branch:** `qwen-edward` / `qwen-edward/e124-noislands-acceptance-exchange`, PR #125
- **Hypothesis and target cost:** deleting the six bf16 `precision_islands.*` tensors from the MTP proposal head removes about 26 MB of weight traffic and three MLX dispatches per proposal step, worth +0.41 to +0.45 % ranked, against an acceptance loss with a pre-registered 0.21 pt absolute kill line.
- **Decision:** **dead.** The mechanism is closed. The corrected cost model predicts a net time **loss**, and an independent 512-token measurement (E82) confirms it. Stage 1, Stage 2 and Stage 3 were stopped by advisor instruction (Advisor Error 93) before any GPU allocation was spent on them.
- **`BASE_SHA`:** `3b8ea425f8887c9b5cd08ddfff6ddc423fb5d9c3` (`senpai/qwen38-mtp-r1`)
- **`UPSTREAM_SHA`:** `41bad1c6f124f8f0c7f324cf60e95cd2c4de2ca6`
- **Candidate commit measured in Stage 0.5:** `113eca25b32a99bb98a85de6ff3dbf0de58b75b3`
- **Yukon promoted submission / source ref used as frontier:** none consulted; nothing was submitted and no submission is proposed.
- **Candidate build fingerprint:** worker `cee9fe9f04d4173a692905645b0d448377f0e84c0988776051633eee2ca09017`, CLI `b71be40f865cb034237d2d5ee65f34838560319115de2f889705e392707cf739`. The Stage 0 worker assertion build was `4894c831690fb53bd53ec5bd4f3fed3180fdd0fba12ad52ac50e6ecd7a697298`.
- **Submitted-surface / generated-twin / metallib digests:** not applicable. No Metal source, generated twin or metallib was touched.
- **Submitted candidate files:** **none.** Nothing on this branch belongs on the submitted surface. The one `Sources`-side change is a research-only arm selector spelled `DARKBLOOM_QWEN_MTP_ISLAND_ARM`; it must not reach a submitted surface and this result does not propose that it does.
- **Supporting test, tooling, or documentation files:** `research/e124_head_census.py`, `research/e124_price.py`, `research/e124_leg.sh`, `research/e124_census_leg.sh`, `research/e124_session.sh`, `research/e124_corpus.py`, `research/e124-corpus-manifest.json`, eleven `research/e124_prose_hi_<id>_512.txt` seeds, `research/e124_stage05_session.sh`, `research/e124_regime.py`, `research/e124_accept.py`, `research/e124_stage1_session.sh`, `research/e124_wandb_log.py`, and one line of `research/e122_rung0_session.sh`. `research/out/` is gitignored, so `e124-head-census.json`, `e124-price.json` and `e124-regime.json` are published as W&B artifacts instead of committed.
- **MTP head provenance, digest, and draft policy:** declared head. Manifest tree digest `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`, 427,742,600 bytes, 40 tensors. Every leg loaded the run tree `dadbfb80…` (the same `model.safetensors` inode, hardlinked, plus the organizer `config.json`) and every leg's `head_provenance_sha256` reads `dadbfb80…`. Draft policy: shipped adaptive schedule, offered depth 8, no forced depth.
- **Token window, fixture, reference source, and harness:** 512 decode tokens; eleven new `research/e124_prose_hi_<id>_512.txt` seeds plus `benchfixture` (`correctness_prompts/public_longcopy_gate_english_512_256.json`); reference rows from the shared e122 goldens at `.mlxfast-private/e122/goldens/<id>-rows-513.json`; harness **`local`** on every number in this report.
- **Exact cell:** `Qwen35Attention.qkv(_:)` and `Qwen35Attention.kv(_:)` on the MTP proposal head's single layer (`mtp?.layers.first`), Q `[12288, 5120]`, K and V `[1024, 5120]` each, affine-4 group-64 backbone with a bf16 island overlay. Source form: Swift MLX op calls. No Metal source form and no M5-variant dependency.
- **Official causal path and score equation:** `harness=ranked`. The mechanism lives entirely in the candidate MTP leg, so `d ln(ranked baseline serial time)/dx = 0` and any candidate time saved improves every affected `raw_p`. No `psi_serial` term appears anywhere in this report. `senpai/verify-ranked-score-boundary.sh` PASS.
- **Assignment-scope preflight:** `senpai/validate-assignment-scope.sh 3b8ea425 Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` → OK.
- **Editable source bytes / headroom / growth / exempt-head bytes:** `senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` → OK, source 2,558,994 / 3,000,000, growth 104,159 / 262,144. Head bytes unchanged at 427,742,600 against the 2 GiB cap.
- **Scored-path reachability evidence:** the arm selector printed a `qwen-mtp-island-arm:` witness into `trace.txt` from inside the scored worker on all four arms, and `research/e124_leg.sh` refuses a leg whose witness disagrees with the requested arm.

---

## Evidence

- **Host, chip, memory, toolchain, thermal policy:** `ip-10-231-2-12.ec2.internal`, Apple M4 Pro, `applegpu_g16s`, 48 GiB. Swift 6.3.3 (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`), target `arm64-apple-macosx26.0`. Stage 0.5 is **not** a timing session: `MLXFAST_LOCAL_COOL_GATE=0`, and `timing_valid=false`, `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`, `official_or_ranked_score=false` are recorded verbatim on every leg. Entry GPU temperature spread across the twelve legs was 24.69 C (36.92 to 61.61 C), which is one reason no timing claim is made from this session.
- **`head_provenance_sha256` for every leg:** `dadbfb80…` on all twelve Stage 0.5 legs and all four `e124dbg3` debug legs. No leg loaded the setup-default head `c5791f65…`, which carries no island tensors and would have made the whole experiment a structural null.
- **Exact commands:**
  ```bash
  python3 research/e124_head_census.py                 # Stage 0, zero GPU
  python3 research/e124_price.py                       # Stage 0, zero GPU
  bash research/e124_stage05_session.sh                # Stage 0.5, 12 legs
  python3 research/e124_regime.py \
      --runs-dir .mlxfast-private/e122/runs-e124 \
      --extra benchfixture --extra-runs-dir .mlxfast-private/e122/runs-e124
  WANDB_API_KEY=… python3 research/e124_wandb_log.py --only stage0 stage05
  ```
- **Cheapest real falsification gate and positive-control verdict:** an 8-token pass per arm (`e124dbg3`) before any 512-token leg. The gate can fail: each arm prints `installsQ`/`installsKV` and the leg driver aborts on a witness mismatch. All four arms produced exactly the requested install pair (`all` true/true, `none` false/false, `q` true/false, `kv` false/true), and every leg reported `passed=true`, `all_tokens_matched=true`, `residual_divergence_count=0`.
- **Tests and risk-based checks, in execution order:** safetensors header census → complete-permutation test on the K and V index sets → source-level dispatch enumeration → repricing → three scope and budget gates → worker symbol assertion (`--require-symbol Qwen35IslandArm` = 6, `--require DARKBLOOM_QWEN_MTP_ISLAND_ARM` = 2) → four-arm 8-token debug gate → twelve-leg 512-token Stage 0.5 session.
- **Exact-token and row-ledger verdict:** all sixteen legs (twelve Stage 0.5, four debug) reported `all_tokens_matched=true` and `residual_divergence_count=0`, over the full 512-token window on the Stage 0.5 legs.
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not relevant. No Metal source was touched.
- **Peak RAM or head/artifact size:** head unchanged at 427,742,600 bytes. A partial arm allocates only the tensors it installs, so an uninstalled island holds no resident memory in its leg.
- **Official status and score:** not submitted.

### Stage 0 — the premise holds, the price does not

Zero GPU. All three of the advisor's premises were confirmed, and four corrections were made to the price.

The islands are present at the live pin, exactly as briefed: `precision_islands.{q,k,v}.weight` BF16 `(1024, 5120)` at 10,485,760 B each, and `precision_islands.{q,k,v}.indices` I32 `(1024,)` at 4,096 B each. Total 31,469,568 B.

`isCompletePermutation` is **true** for K and for V (1,024 unique indices, min 0, max 1023, output count 1,024) and **false** for Q (1,024 unique indices over 12,288 outputs, min 3, max 12,239). The fast branch at `Qwen35.swift:2493-2513` is therefore the live branch: K and V are dense bf16 today, and only 1,024 of 12,288 Q rows carry a scatter.

Operating point measured on this host (`research/out/e116x512k0`, 512 tokens, 78 rounds): mean draft length `d = 6.3590`, acceptance `0.877016`, round `151,471 µs`. Effective stream rate derived from the E87 head-byte law: **220.3 GB/s**, a plausible fraction of the M4 Pro's 273 GB/s peak.

| arm | qkv B/step | flush B/round | Δ bytes/round vs `all` | Δ ops/round | byte local % | dispatch local % | local % | **ranked %** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `all` | 66,850,816 | 20,971,520 | 0 | 0 | 0 | 0 | 0 | 0 |
| `kv` | 56,360,960 | 20,971,520 | 66,704,725 | 12.72 | 0.200 | 0.009–0.015 | 0.209–0.215 | **0.056–0.080** |
| `q` | 51,777,536 | 5,898,240 | 110,923,881 | 6.36 | 0.333 | 0.004–0.008 | 0.337–0.340 | **0.084–0.116** |
| `none` | 41,287,680 | 5,898,240 | 177,628,606 | 19.08 | 0.532 | 0.013–0.023 | 0.545–0.555 | **0.140–0.196** |

Coefficients per campaign rule 69: byte class 0.24–0.327 applied to the byte column only, dispatch class 0.95 applied to the dispatch column only. No blended coefficient was used.

Four corrections, all accepted by the advisor:

1. **Affine-4 rows cost 2,880 B, not 2,560 B.** A group-64 row is 2,560 weight + 160 scales + 160 biases, confirmed against the header (`q_proj.weight` U32 `(12288, 640)` plus `scales`/`biases` BF16 `(12288, 80)`).
2. **`appendHistoryKV` was missing from the brief's table, and it helps.** `Qwen35MTP.swift:235` calls it once per round into `Qwen35Attention.kv(_:)`, which takes the same island fast branch. It adds 15.07 MB per round to the saving of arms `none` and `q`.
3. **The 0.304 % dispatch component is struck.** It was a residual (E82's 0.904 % local minus a 0.600 % byte model), not a transfer class. Priced as a real dispatch class — 19.08 dispatches per round at the twice-corrected 1.05–1.80 µs boundary — it is 0.013–0.023 % local, thirteen times smaller. The two deleted `matmul`s are 10.49 MB and 20.97 MB gemvs whose cost is bytes, already counted in the byte column; pricing them again at a dispatch coefficient double-counts.
4. **The arm ladder is ordered the reverse of the brief's prediction.** Dropping only Q removes 10.49 MB per draft step; dropping only K and V removes 15.07 MB per draft step *plus* 15.07 MB per round from the flush. The Q scatter is one `putAlong` over 1,024 elements, worth about 1.5 µs.

The Stage 0 stop rule fired: the best arm reprices to +0.140 to +0.196 % ranked against the +0.20 % floor.

### The closure: the corrected model reproduces an independent measurement to 0.049 pp

E82 measured this exact mechanism end to end at 512 tokens on the same host class (W&B [`o0rawiol`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/o0rawiol), [`yerghmxz`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/yerghmxz)):

| arm | mean s/tok | rounds | delta vs `declared` |
|---|--:|--:|--:|
| `declared` (= `all`) | 0.031432 | 78 | ref |
| `noislands` (= `none`) | 0.031547 | 79 | **+0.366 % slower** |
| `qonly` (= `q`) | 0.031998 | 80 | **+1.801 % slower** |

Reconciliation against my Stage 0 model:

```text
one extra decode round      = 151,471 / 15,694,534   = 0.965 % of the leg
gross saving, arm `none`    = 0.532 byte + 0.018 disp = 0.550 %
model net                   = 0.550 - 0.965           = -0.415 %   (slower)
E82 measured                =                          +0.366 %   (slower)
|model - measurement|                                 =  0.049 pp
```

The islands buy more acceptance than they cost in bytes, by a factor of about 1.8. Correction 4 is confirmed from the other side: `qonly`, which keeps only the Q island and drops dense bf16 K and V, was E82's **worst** arm.

Arm `kv` is the only arm never measured. Its gross saving is 0.212 % local, which buys 0.220 of an extra round. It must cost exactly zero extra rounds and zero extra rows, and even then its ceiling is +0.048 to +0.065 % ranked — three to four times below the +0.20 % line. It does not justify a GPU slot.

**Every number in the E82 comparison above is E82's, not mine.** E124 measured no timing. The 0.049 pp agreement is between my independently derived byte model and an independent prior measurement, with the measured round response supplied from that same prior measurement.

### Harness defect 28

`MLXFAST_QWEN_MTP_ISLAND_ARM` never reached a worker leg. `sanitizedRuntimeWorkerEnvironment` forwards ten exact names plus the prefixes `DARKBLOOM_ DYLD_ LC_ METAL_ MLX_ MTL_`, and its maintainer contract explicitly forbids adding a broad `MLXFAST_` allowance. Renaming the selector to `DARKBLOOM_QWEN_MTP_ISLAND_ARM` fixed it. The witness guard caught the defect in nine minutes.

Three model-side gates are therefore dead in a worker leg: `MLXFAST_QWEN_MTP_EXACT_QKV_ROWS`, `MLXFAST_QWEN_MTP_TOP32` and `MLXFAST_QWEN_MTP_TRACE`. The advisor audited the campaign against them and found that **no measured result depends on any of them**.

### Stage 0.5 — the median-regime corpus, and why it fails

Twelve 512-token legs: one per seed, plus `benchfixture` as the in-session anchor. Shipped schedule, island arm `all`, declared head, all matched.

`f128` and `l128` are `accept_rate_of_drafted` over the first and last 128 decoded tokens. `ngram` is the longest repeated token n-gram, capped at 64; `gap` is the token distance between its two occurrences. `d1st` and `dlast` are distinct-token ratios over the same two windows. The verdict column applies the pre-registered line `accept_rate_of_drafted >= 0.83` **and** `mean_depth >= 4.4`.

| seed | domain | rounds | depth | acc/rnd | accept | f128 | l128 | ngram | gap | d1st | dlast | verdict |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `medicine_hippoc` ⚠️ | medicine | 80 | 5.737 | 5.400 | **0.9412** | 0.791 | 1.000 | 64 | 36 | 0.211 | 0.195 | **PASS** |
| `drama_dollhouse` | drama | 92 | 5.076 | 4.565 | **0.8994** | 0.649 | 1.000 | 64 | 23 | 0.352 | 0.141 | **PASS** |
| `benchfixture` | anchor, not a corpus seed | 78 | 6.359 | 5.577 | **0.8770** | 0.975 | 1.000 | 64 | 391 | 0.758 | 0.742 | anchor |
| `republic_jowett` | republic | 104 | 4.587 | 3.923 | **0.8553** | 0.581 | 1.000 | 64 | 53 | 0.383 | 0.227 | **PASS** |
| `beagle_a` | beagle | 119 | 4.244 | 3.303 | 0.7782 | 0.493 | 1.000 | 64 | 7 | 0.508 | 0.055 | fail both |
| `beagle_b` | beagle | 126 | 4.000 | 3.063 | 0.7659 | 0.441 | 0.938 | 64 | 141 | 0.562 | 0.539 | fail both |
| `botany_andrews` | botany | 143 | 3.538 | 2.587 | 0.7312 | 0.565 | 0.964 | 64 | 155 | 0.547 | 0.422 | fail both |
| `essays_montaigne` | essays | 146 | 3.438 | 2.507 | 0.7291 | 0.569 | 0.820 | 6 | 43 | 0.508 | 0.250 | fail both |
| `medicine_hist` | medicine | 153 | 3.503 | 2.346 | 0.6698 | 0.476 | 0.890 | 64 | 13 | 0.617 | 0.102 | fail both |
| `travel_eothen` | travel | 172 | 3.157 | 1.977 | 0.6262 | 0.388 | 0.825 | 13 | 14 | 0.586 | 0.172 | fail both |
| `plutarch_lives` | plutarch | 180 | 3.128 | 1.844 | 0.5897 | 0.442 | 0.636 | 64 | 45 | 0.562 | 0.500 | fail both |
| `essays_bacon` | essays | 285 | 1.996 | 0.796 | 0.3989 | 0.377 | 0.366 | 8 | 157 | 0.562 | 0.578 | fail both |

⚠️ `medicine_hippoc`: the 512-token window landed in Adams' critical introduction to PG 72583, so the text is medical scholarly commentary, not a Hippocratic treatise.

| criterion clause | required | observed | verdict |
|---|---|---|---|
| seeds at `accept >= 0.83` and `depth >= 4.4` | ≥ 4 | **3** | **fail** |
| at least one `beagle` | 1 | **0**, best `beagle_a` at 0.7782 / depth 4.244 | **fail** |
| at least one `medicine` | 1 | 1 | pass |

**Corpus verdict: NOT USABLE.** Both failing clauses fail independently, so no tie-break is needed.

First-rejection position histograms for all twelve legs are in `research/out/e124-regime.json`. The shape shifts exactly as expected between regimes: `medicine_hippoc` `{0:5, 1:3, 2:3, 4:1, none:68}` against `essays_bacon` `{0:122, 1:74, 2:9, 3:2, none:78}`.

**Why it failed.** Every seed that reaches 0.83 does so by late-window greedy-decode degeneration, not as a property of the text. The three passers go 0.791 → 1.000, 0.649 → 1.000 and 0.581 → 1.000 from the first to the last 128 tokens, all three hit the 64-token repeated-n-gram cap at close range, and all three collapse in distinct-token ratio. **On the first 128 decoded tokens no prose seed in any domain reaches 0.83**; the highest is 0.791. Only `benchfixture` clears it, at 0.975, and it is a long-copy gate where copying is the task, so it clears for a reason that does not generalise. Its `gap` of 391 tokens is the signature of that difference: it reproduces a distant instruction block instead of falling into a local loop.

Degeneration creates the high cluster but cannot rescue a low seed. `beagle_a` reaches `l128 = 1.000` with a distinct-token ratio of 0.055, the most degenerate leg in the set, and still pools to only 0.7782, because its first 128 tokens accepted at 0.493. The three seeds whose text never degenerates (`essays_montaigne` ngram 6, `essays_bacon` 8, `travel_eothen` 13) accept at 0.729, 0.399 and 0.626. That is the honest local prose regime, and it is the F92 low cluster.

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | not measured | not measured | — |
| MTP seconds/token | not measured | not measured | — |
| local serial-relative speedup | not measured | not measured | — |
| effective mean draft length | 6.3590 (`research/out/e116x512k0`) | 6.359 (`benchfixture`, Stage 0.5) | +0.000 |
| accepted draft rate | 0.877016 (`research/out/e116x512k0`) | 0.8770 (`benchfixture`, Stage 0.5) | +0.0000 |

The `benchfixture` reproduction of the operating point to four decimal places, across two sessions and two commits, is a useful side check that the Stage 0.5 harness measures the same thing the Stage 0 price was built on.

**Identity fields.** Every Stage 0.5 leg shares base commit, worker digest, host, chip, memory, toolchain, head, token window, offered depth, reference-golden source and harness label. The only dimension that varies across the twelve legs is the prompt seed, which is the experimental variable. The E82 comparison in this report varies base commit, worker digest and head artifact, so it is reported as an independent prior measurement and not as an arm of this experiment. No value in this report is interpolated or extrapolated, except the modelled dispatch and byte costs in Stage 0, which are labelled as model output throughout.

---

## Conclusion

**What happened and why.** The advisor assigned E124 from a byte model that predicted +0.41 to +0.45 % ranked for deleting the head's precision islands. I confirmed all three of its premises — the tensors exist at the live pin, the K and V index sets are complete permutations, and the fast dense-bf16 branch is live — and then found four errors in the price. Two corrections raised the byte saving (scales and biases, `appendHistoryKV`), one cut the dispatch component by a factor of thirteen (a residual is not a transfer class), and one reversed the arm ladder. The corrected best arm reprices to +0.140 to +0.196 % ranked, below the +0.20 % stop line. The advisor then found that E82 had already measured this mechanism end to end and that it is **slower**: arm `none` +0.366 %, arm `q` +1.801 %. My corrected model reproduces that measurement to 0.049 pp once the measured extra-round response is supplied. The islands buy about 1.8x more acceptance than they cost in bytes.

**Evidence for or against the mechanism.** Against, decisively. Two independent lines agree: a bottom-up byte and dispatch model built from the live artifact and this host's stream rate, and a prior six-leg 512-token palindrome whose arms do not overlap (`max(declared) = 0.031454 < min(noislands) = 0.031538`) and whose round counts reproduce exactly across repeat visits. Arm `kv`, never measured, has a ceiling of +0.048 to +0.065 % ranked even under the assumption that it costs nothing in acceptance.

**Prompt or M5 transfer risk.** Not applicable to the closure: the mechanism is closed on time, not on a prompt-sensitive quantity, and E82's measurement is a whole-leg absolute time. The Stage 0.5 finding, by contrast, is *entirely about* transfer risk, and it enlarges it. The campaign has no local fixture that reproduces the hidden median-carrying regime for the right reason.

**Smallest useful next action.** Report `accept_rate_first_128` alongside the pooled acceptance rate on every future acceptance leg. It costs nothing, it comes out of the trace that is already written, and it is the only cheap discriminator between a regime and a loop.

**Recommendation: close.** Merge as a negative. The mechanism is dead and should not be reopened without a new measured reason — specifically, without evidence that the head's proposal quality no longer depends on the bf16 corrections, which would require a different head artifact rather than a different arm.

### Honest accounting of unrun work

`research/e124_accept.py` (a stratified acceptance estimator with named strata, a kill line applied in the decisive stratum only, paired per-position tables, per-seed cluster-bootstrap deltas and a cross-arm exactness check) and `research/e124_stage1_session.sh` (the four-arm stratified driver) were written and **never run**. Stage 1 was stopped before they executed. They are committed as reusable tooling, not as evidence. The W&B module docstring records the same fact.

One claim made during this experiment was **wrong and is withdrawn**: I reported a SwiftPM incremental-build defect after finding a stale arm selector in `.build/release/mlxfast-swift`. `Package.swift` gives `MLXFastCLI` only `MLXFastCore`, `MLXFastTransform`, `MLXFastHarness` and `Tokenizers`. It has no dependency on `MLXLLM` or `MLXFastModel`, so the CLI cannot contain model code, and a from-scratch CLI build carries zero copies of the selector against the worker's two. The withdrawal is in commit `113eca25`.

### Suggested follow-ups, not implemented

1. **Check whether the hidden goldens also degenerate.** F70 and F83 give per-prompt acceptance but not the first-128 split. If any hidden-prompt trace with per-round detail exists in the campaign record, the split is free to compute, and it decides whether the hidden 0.83–0.90 band is a text property or a 512-token greedy-decode artifact. That answer re-scopes every acceptance experiment in the campaign.
2. **Treat the three Stage 0.5 passers as a degeneration-regime stratum, not a median-regime stratum.** They remain the only local legs that exercise draft positions 4–7 at high acceptance, which is useful for depth work, but they are not a substitute for the hidden regime and should never be labelled as one.
3. **Fit E127's depth price on measured marginal cost per width, not on acceptance from these seeds.** A price fitted on legs whose deep positions accept at 1.000 because the model is looping will be biased toward over-drafting, which is the exact failure mode E127 exists to remove. Marginal cost per width is a timing quantity and is prompt-independent; acceptance should only choose the operating point at which the price is evaluated.
4. **Delete the now-dead fast-path code in `qkv(_:)` at `Qwen35.swift:2281-2300` after E120 Route B merges.** It was left in place deliberately: thorfinn owns that file, and a `quantizedMM` call-site edit would conflict on the most valuable branch in the campaign.
5. **Repoint `research/e82_corpus.py` at the `gutenberg.pglaf.org` mirror.** `www.gutenberg.org` answers 503 and `gutendex.com` answers 403 from this host, so that script would fail if re-run today.

### W&B

- Stage 0 census and price: [`6n3rutks`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/6n3rutks)
- Stage 0.5 median-regime corpus: [`g6n1i0de`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/g6n1i0de)
- Prior measurement referenced for the closure, not produced by E124: E82 [`o0rawiol`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/o0rawiol) and [`yerghmxz`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/yerghmxz)
