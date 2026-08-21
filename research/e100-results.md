SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mtp_seconds_per_token_512tok_depth8","available":true,"value":0.033455},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

**Choice of primary metric.** I use the 512-token **depth-8** MTP seconds/token (baseline
0.033717, candidate 0.033455, **−0.775 %**) rather than the much larger depth-4 result
(0.037185 → 0.032422, **−12.810 %**). Depth 8 is the harness default and the closer analogue
of the ranked schedule, and it is the one arm independently reproduced by a
**gate-qualified** leg. The depth-4 number is the better *mechanism* isolator, because that
session puts ~94 % of round mass at M=5, but it is not the configuration the ranked runner
uses. Quoting the larger number as the headline would overstate transfer. Both are reported
in full in §4.2.

# E100 — Fewer weight streams per round: collapse the M=5 verify width to one x-group

- **Student / branch:** `qwen-alphonse` / `qwen-alphonse/e100-fewer-weight-streams-per-round` (PR #102)
- **Hypothesis and target cost:** At verify width M=5 the scored `affine_qmv_fast` dispatcher splits the rows into two x-groups (`3+2`) and therefore streams the 4-bit backbone weights through the GPU **twice** per projection. Collapsing M=5 into a single 5-wide x-group removes one full weight stream per M=5 round. Measured target cost: the base width law is `round_ms = 61.08 + 14.86 * M`, so a duplicated stream at M=5 is worth roughly 22 ms of a 135 ms round.
- **Decision:** **green locally**, and gate-qualified at 512 tokens.
- **`BASE_SHA`:** `cd0a89dadf543261a91eb6cae07c57b3f3282519` (`senpai/qwen38-mtp-r1`)
- **`UPSTREAM_SHA`:** `8b54ff11c6d686628f6534d7127a261115782757`
- **Byte-budget contract sha:** `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`
- **Candidate commit:** `c049485c4bcd925abfcbd27fe368a4d64294138f`
- **Yukon promoted frontier used:** submission `8819b108-e0cb-411f-8c60-e1629523cb49`, sourceRef `b40c28e95cc7488e798f2c90b4984bf73558ff93`, score `3.32794960796967`
- **Candidate build fingerprint:** worker `.build-worker/release/mlxfast-runtime-worker`, sha256 `fd9e6b24950e8cc41b34574d19fb79a0c4b862f7fc092198e1afd1c3bf9377f3`, mtime `2026-08-21T14:18:53Z`
- **Generated-twin audit:** `TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver`. The `quantized` family reports 1 vendored section and 0 normalized toolchain sections, so the readable `.h` and the JIT `.cpp` twin stay byte-identical.
- **Submitted candidate files (2):**
  - `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`
  - `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
- **Supporting research-only files (not submitted):** `Tests/MLXFastTests/E100StreamCollapseProbeTests.swift`, `research/e100_e2e_leg.sh`, `research/e100_e2e_pair.sh`, `research/e100_e2e_analysis.py`, `research/e100_round_model.py`, `research/e100_presubmit.sh`, `research/e100_reg_census.py`, `research/e100_wandb_log.py`, `research/ranked_stream_ab_board.json`, this file.
- **MTP head provenance:** `head_provenance_sha256 = 62516c6f3799b66c91171ee13aa6816db5af197aa8c527cec0f6bb4026f0c7b7` for **every** leg, base and candidate. `uses_pinned_mtp_head=true`. No proposal head was declared; the organizer-pinned head was used throughout. Draft policy unchanged from base.
- **Token window, fixture, reference source, harness:** 512 seed tokens + 512 decode tokens on the public fixture; reference `candidate-local-mtp-golden-rows`; harness `local`; `rankable=false`.
- **Exact cell:** shape = 4-bit affine group-64 backbone projections with `out_vec_size >= 4096`; width M=5 (`ntg.x == 5`); dispatch family = `affine_qmv_fast<bfloat16_t, 64, 4, false>` reaching `qmv_fast_crossrow_affine4_g64_m`; source form = **JIT string** (Finding 28, see below), not `mlx.metallib`; M5 variant = the `_nax`-era `applegpu_g17s` target.
- **Official causal path and score equation:** the edit lives only in the candidate's target-verification kernels. The ranked numerator is the pinned serial build and is untouched; the edit reduces the ranked **denominator** only. `senpai/verify-ranked-score-boundary.sh` returns `PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only`.
- **Assignment-scope preflight:** `assignment scope OK: 2 submitted path(s) against BASE_SHA=cd0a89da…`
- **Editable byte budget:** `source=2515544/3000000 headroom=484456 growth=60709/262144 exempt=2410/2147483648 files=154`. The candidate contributes **0 bytes** of its own growth; the 60709 is the inherited base-to-contract delta.
- **Scored-path reachability evidence:** the M=5 arm changes end-to-end decode time by −12.81 % at depth 4 and the isolated kernel probe changes the same shapes by −17.7 %. A path that is not executed cannot do that. The forced binary witness (`<T, 5, 5, true>`: 1, `<T, 5, 3, true>`: 0) proves the timed worker contained the arm.

---

## 1. Retraction and root cause — the most important part of this report

**I retract the four earlier E100 end-to-end sessions I reported before slot `a1`. They were base-versus-base and measured nothing.**

### Finding 28 — for the `quantized` kernel family the runtime-effective source is the JIT string linked into the worker binary, not `mlx.metallib`

1. `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp:915` — `get_quantized_kernel()` calls `d.get_library(lib_name, builder)`.
2. `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp:770` — that overload calls `build_library_(builder())` directly. It never opens `default.metallib`. The library name is the fully-templated kernel name, so a metallib copy can never be reached.
3. `metal::quantized()` is the `R"preamble(...)"` literal inside `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`, which is **compiled into** `.build-worker/release/mlxfast-runtime-worker`.
4. `Package.swift:284` excludes `nojit_kernels.cpp` and compiles `jit_kernels.cpp`, confirming the JIT build.

### Harness defect 12

`benchmark-qwen-mtp.sh:200-206` extracts only `metallib_rebuild_required()` from `benchmark.sh`. It never reaches `swift_build_required()` (`benchmark.sh:1791-1806`) or the `swift build --scratch-path .build-worker` at `benchmark.sh:1842`. **`./benchmark-qwen-mtp.sh` therefore never rebuilds the runtime worker.** Editing `quantized.cpp` and running the benchmark measures the previously built worker.

### Timestamp proof

| Artifact | Timestamp |
|---|---|
| `.build-worker/…/Cmlx.build/mlx-generated/quantized.cpp.o` | 2026-08-20 21:11:03 |
| `.build-worker/release/mlxfast-runtime-worker` | 2026-08-21 10:44:44 |
| my arm switches in the retracted sessions | 2026-08-21 11:45 – 13:03 |

Every retracted arm ran a worker built before the first arm switch.

### Forced-rebuild proof

A forced `senpai/rebuild-and-assert-worker.sh` flipped the binary witnesses in 45 s: `qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>` went 0 → 1 and `<T, 5, 3, true>` went 1 → 0. Before the force, the arm string was simply absent from the binary being timed.

### Fix carried into every later leg

`research/e100_e2e_leg.sh` now forces `senpai/rebuild-and-assert-worker.sh` with arm-specific **binary** witnesses and **refuses to time** a mismatched arm. Each leg records `worker_mtime_pre/post`, `worker_sha256_pre/post`, `worker_m5_ipg5`, `worker_m5_ipg3`, `worker_m6_ipg2` and `worker_m6_ipg3`. Per f3 item 1 this guard stays in the merged tree.

### Trap worth recording

`rebuild-and-assert-worker.sh` matches with `grep -c --`, so its needles are **regexes**. A needle like `NA in [2, 5]` matches nothing, because `[2, 5]` is a character class. Use metacharacter-free template strings such as `qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>`.

### Advisor triage of blast radius — accepted, with one caveat

The advisor ruled E76 §3-5, E97, E98 and E102 not at risk, and only E100 before slot `a1` retracted. I accept this. E76's register work is compile-time and never depended on a timed worker; E97/E98/E102 did not edit the `quantized` JIT family.

**Caveat I am obliged to flag:** the general rule is *any* experiment that edits a JIT-family source (`quantized`, `quantized_nax`, `quantized_utils`, `reduce`, `softmax`, `sort`, `steel_*`, `ternary`, `unary`) and times it through `./benchmark-qwen-mtp.sh` without a forced worker rebuild is silently base-versus-base. The triage above is correct for the four named experiments, but the defect is generic. I recommend the guard become the default rather than an E100-local script.

---

## 2. The candidate

Four lines, two files, applied identically to the readable header and its JIT twin.

```diff
-  static_assert(NA >= 2 && NA <= 4, "wide multi-row QMV supports NA in [2, 4]");
+  static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");
```
```diff
         case 5:
-          qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>(
+          qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>(
```

`git diff cd0a89da…HEAD -- Sources Vendor mtp-head.manifest.json mtp-head Package.swift`:

```
 Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp               | 4 ++--
 Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h | 4 ++--
 2 files changed, 4 insertions(+), 4 deletions(-)
```

`<T, 9, 5, true>` was dropped per advisor note f1: M=9 is unreachable because `segmentedVerifyDepthCap = 7`. `case 6` is untouched, so askeladd's E98 metadata-read lines in the same function merge mechanically.

---

## 3. Evidence

### Host, toolchain and thermal policy

- Host `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, GPU family `applegpu_g16s`, 48 GiB (`hw.memsize = 51539607552`).
- Swift 6.3.3 (`swiftlang-6.3.3.1.3`), target `arm64-apple-macosx26.0`.
- **Ranked box is `applegpu_g17s`, not `g16s`.** Every local number transfers with risk; see §6.

Two thermal policies were used and they are never mixed:

| Policy | Legs | `cool_gate_passed_real_gate` | `gate_qualified_for_timing` |
|---|---|---|---|
| `MLXFAST_LOCAL_COOL_GATE=0`, ABBA-counterbalanced | all 12 ABBA legs + dose control | **false** | **false** |
| real 40 C gate | the single `--local-submit` 512-token leg | **true** | **true** |

For every ungated leg `harness=local` and `timing_valid=false`. Entry temperature spread across the 12 ABBA legs was **14.5 C** (36.6 – 51.1 C); exit 58.8 – 61.6 C. An ungated arm is directional causal evidence inside its counterbalanced session. It is not a gate-qualified measurement, is not comparable to a gated historical run, and is never a ranked score. I do not relabel any of it.

### Exact commands

Base and candidate legs:

```bash
research/e100_e2e_pair.sh a1,b1,b2,a2 collapse d8,w512,w512d4     # arm selection per slot
research/e100_e2e_analysis.py --json research/out/e100_e2e_session.json
research/e100_round_model.py  --json research/out/e100_round_model.json
```

Isolated kernel probe:

```bash
MLXFAST_RUN_E100_PROBE=1 MLXFAST_E100_OUT=research/out/e100-probe.json \
  swift test --force-resolved-versions --filter E100StreamCollapseProbeTests
```

Pre-submit chain:

```bash
research/e100_presubmit.sh          # logs to research/out/e100-presubmit/
```

Register census:

```bash
python3 research/e100_reg_census.py --base cd0a89dadf543261a91eb6cae07c57b3f3282519 \
  --out research/out/e100-reg-census.json
```

### Cheapest real falsification gate and positive control

The isolated probe compares base and candidate kernels bit-for-bit on the five scored projection shapes. Verdict: **0 / 45 across-arm bit mismatches**. The positive control — a deliberately wrong reduction — differs on **5 / 5** shapes, so the comparison can detect a difference when one exists.

### Tests and risk-based checks, in execution order

| # | Check | Result |
|---|---|---|
| 1 | `senpai/rebuild-and-assert-worker.sh` with binary witnesses | PASS |
| 2 | `tools/build-mlx-metallib.sh` | PASS |
| 3 | `python3 research/twin_audit.py` | PASS, 29 twins |
| 4 | `senpai/validate-assignment-scope.sh` | PASS, 2 paths |
| 5 | `senpai/check-editable-budget.sh 770a3ff2…` | PASS, 0 candidate growth |
| 6 | `senpai/verify-ranked-score-boundary.sh` | PASS |
| 7 | `swift test --force-resolved-versions` | 40 issues / 9 names, **all pre-existing** |
| 8 | `./benchmark-qwen-mtp.sh --local-submit` (512 tok, real gate) | **PASS** |

Step 7 detail: `Test run with 724 tests in 62 suites failed after 20.566 seconds with 40 issues`. The 9 failing names are `theQwenMTPTrackIsArmedOnQwen38`, `submissionStaticReviewPromptCoversMeasurementStructureExploitation`, `theCheckedInDeclarationSelectsThePinnedHead`, `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`, `theSeededCalibrationExpectationMatchesItsRecordedProvenance`, `startupMemoryPolicyKeepsRanked128GiBProfile`, `qwen36ConfigContractDigestMatchesTheReferenceManifest`, `participantDocsExposeDefaultCLIInstallDirectory` and `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen`. All are organizer contract and documentation tests. **I added zero.**

*Self-correction:* I previously recorded this floor as 42 issues across 11 names. The two extra names, `E95QmvWidthProbeTests` and `E95DonationProbeTests`, are opt-in and did not run in this invocation. The correct comparable floor for a default `swift test` is **40 across 9**.

### Exact-token and row-ledger verdict

Every timed leg in this experiment, in both arms, reported `all_tokens_matched=true`, `residual_divergence_count=0`, `public_drift_tripwire_passed=true`, `passed=true`, `git_dirty_build=0` and `worker_sha256_pre == worker_sha256_post`.

The gate-qualified 512-token leg closes its ledger exactly:

```
reference: rows=513 seed_tokens=512 reference_seed_token=271 self_consistent=true
           (replayed 1 row bit-identically) chain_contradictions=0
serial:    tokens=512 depth=0 rounds=512 reference_checked_rows=512/512 all_tokens_matched=true
MTP:       tokens=512 depth=8 rounds=77  reference_checked_rows=568/568 all_tokens_matched=true
```

The arithmetic is self-consistent in both directions:

- 77 rounds x 6.3766233766 mean drafts = 491 draft rows; 491 + 77 primaries = **568 rows evaluated = 568 rows checked**.
- 491 drafts x 0.8859470468 accepted = 435 accepted drafts; 435 + 77 primaries = **512 emitted tokens**.

**Divergent tokens: none. Failure category: none.**

### Post-EOS continuation — measured, not assumed

EOS for this checkpoint is token id **248044**. In the public golden `correctness_prompts/public_longcopy_gate_english_512_1024.json` (512 prompt tokens, 1024 expected tokens) EOS occurs at generated indices:

```
301, 692, 696, 701, 706, 713, 720, 727
```

Index **301 is inside the 512-token decode window**; surrounding tokens are `[…, 21118, 5426, 13, 248044, 248045, 271, 248068, 198, 760, …]`.

The candidate therefore met EOS at generated token 301 and produced **210 further tokens that all matched exactly**, with the drift tripwire green. This is fixed-window continuation demonstrated on real post-EOS tokens.

The unit-level guards also passed: `Suite QwenMTPFixedWindowTests passed`, covering `eosInsideAnAcceptedPrefixDoesNotEndTheWindow`, `theEditableSessionContinuesPastEosWithoutTheOverlay`, `theRowLedgerClosesOverAWholeWindow`, `theExtraVerifyRowNeverChangesTheCount` and `theCountIsTheLongestCommonPrefixAndEosIsJustAToken`.

### Peak RAM and artifact size

No proposal head was declared, so exempt head bytes stay at the base `2410`. No change to resident model memory: the arm changes only the threadgroup partitioning of an existing kernel.

### Official status

**Not submitted to Yukon.** Per advisor order f3 item 7, Edward's submission `87b654b2-63fd-44f8-a606-6709bad39ed0` went out at 13:47Z with a receipt expected 14:45Z – 15:50Z. The candidate is committed, gated and ready.

---

## 4. Measurements

### 4.1 Headline gate-qualified leg (real 40 C gate, 512 tokens)

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.073810 | 0.073762747924774885 | −0.06 % |
| MTP seconds/token | 0.037185443 | **0.033453777199611068** | **−10.03 %** |
| local serial-relative speedup | 1.984907 | **2.204915381741481** | **+11.08 %** |
| effective mean draft length | 3.936937 | 6.3766233766233764 | (different depth cap) |
| accepted draft rate | 0.919908 | 0.8859470468 | — |

Read that table with care. The gated leg runs at the harness default depth 8, while the strongest ABBA session ran at depth 4. The clean same-depth comparisons are in §4.2. I put the gated leg here because it is the only leg that passed the real thermal gate, and it independently reproduces the depth-8 candidate value: ABBA `w512` collapse mean 0.033455 versus this gated leg 0.0334538.

### 4.2 End-to-end ABBA, 4 slots x 3 sessions (ungated, counterbalanced)

Slots `a1`, `a2` = collapse; `b1`, `b2` = base. ABBA order within each session.

| Session | tokens | depth | base mean s/tok | collapse mean s/tok | **delta** | worst within-arm spread |
|---|--:|--:|--:|--:|--:|--:|
| `d8` | 64 | 8 | 0.087088 | 0.086744 | **−0.395 %** | 0.238 % |
| `w512` | 512 | 8 | 0.033717 | 0.033455 | **−0.775 %** | 0.112 % |
| **`w512d4`** | 512 | 4 | 0.037185 | 0.032422 | **−12.810 %** | 0.057 % |

Individual legs:

```
d8      a1 0.086673  b1 0.086984  b2 0.087191  a2 0.086814
w512    a1 0.033474  b1 0.033716  b2 0.033717  a2 0.033437
w512d4  a1 0.032427  b1 0.037175  b2 0.037196  a2 0.032417
```

In every session the effect is larger than the worst within-arm spread, and in `w512d4` it is larger by a factor of 225.

Local ratio: `w512` 2.184566 → 2.201447 (+0.773 %); `w512d4` 1.984907 → 2.272880 (**+14.508 %**).

**The trajectory is identical in both arms**, which is what makes this a pure cost effect rather than a schedule effect: `d8` draft 5.400000 / accept 1.000; `w512` 6.376623 / 0.885947; `w512d4` 3.936937 / 0.919908 — the same in base and collapse.

**Serial legs are flat**, confirming the arm touches only the MTP denominator: `w512` base 0.073656 versus collapse 0.073650 (−0.008 %); `w512d4` base 0.073810 versus collapse 0.073691 (−0.160 %).

### 4.3 Round-cost model

Pooled prefill **4.031 s**; serial round **65.83 ms**.

| Session | M̄ | rounds | base round | collapse round | delta | spreads |
|---|--:|--:|--:|--:|--:|--:|
| `d8` | 6.400 | 10 | 154.28 ms | 152.08 ms | −1.427 % | 1.33 / 0.90 ms |
| `w512` | 7.377 | 77 | 171.85 ms | 170.11 ms | −1.011 % | 0.00 / 0.25 ms |
| **`w512d4`** | 4.937 | 111 | **135.21 ms** | **113.24 ms** | **−16.250 %** | 0.10 / 0.05 ms |

Base width law from six points: `round_ms = 61.08 + 14.86 * M`. Each extra verify row costs 14.86 ms.

**Cross-session consistency — the strongest single argument for the mechanism.** The three sessions have very different M=5 shares, yet dividing the round saving by the implied M=5 share recovers the same per-round constant:

| Session | round saving | implied M=5 share | saving per M=5 round |
|---|--:|--:|--:|
| `d8` | −2.20 ms | ~10 % | ~22 ms |
| `w512` | −1.74 ms | ~8 % | ~22 ms |
| `w512d4` | −21.97 ms | ~94 % | ~23.4 ms |

Three independent sessions, spanning a 12x range of M=5 exposure, all recover **22 – 23.4 ms per M=5 round**. The isolated kernel probe independently reconstructs **−25.3 ms**. Two unrelated methods agree to within 8 %.

### 4.4 Isolated kernel probe

`Tests/MLXFastTests/E100StreamCollapseProbeTests.swift`, gated by `MLXFAST_RUN_E100_PROBE=1`, output via `MLXFAST_E100_OUT`. It is built through `swift test`, which uses the default `.build` scratch path and therefore performs a real build every time. **The probe was never affected by harness defect 12.**

- Exactness: **0 / 45 across-arm bit mismatches**; positive control differs on 5 / 5 shapes.
- **M=5: −17.7 % ± 1.7 %** (34 clean cells).
- M=9: −9.8 % ± 0.9 %.
- **Byte-identical-dispatch width tax on M=6, 7, 8: +0.475 % ± 0.404 %**, median −0.176 %, n=13. This is the null: widths whose generated code did not change show no effect, so the M=5 effect is not a build or measurement artifact.

Per-shape M=5 improvement: `gate_up` −20.65 %, `in_proj` −25.67 %, `qkv` −14.63 %, `down` −17.75 %, `out_proj` −38.63 %.

Probe-reconstructed target forward at M=5: 114.68 → 89.40 ms, i.e. **−25.3 ms per round**.

### 4.5 Control 2 — the dose leg (reverse direction)

To test whether the effect really is "number of weight streams", I ran the opposite change on the base tree: `case 6` split from 2 x-groups into **3** (`<T, 6, 2, true>`), adding a stream instead of removing one. Commit `e17f1561`, since reverted. Witnesses `worker_m6_ipg2=1`, `worker_m6_ipg3=0`, `all_tokens_matched=true`, entry/exit 40.0 / 59.9 C.

| Metric | base `d8` | dose6 | delta |
|---|--:|--:|--:|
| s/tok | 0.087087680 | 0.088259187 | **+1.345 %** |
| round | 154.26 ms | 161.76 ms | **+7.50 ms, +4.860 %** |

At 22.0 ms per stream, measured independently from the collapse arm, +7.50 ms implies an M=6 round share of **34.1 %**. Ledger 207 records the M=6 share as **33.4 %**. The two agree to 0.7 points.

Three candidate readings of the collapse result were on the table. The dose control settles them:

- **(a) "it is a compiler-scheduling accident at NA=5"** — dead. Adding a stream at a different width M=6 costs time in proportion.
- **(b) "it is a thermal or ordering artifact"** — dead. The sign reverses when the mechanism reverses.
- **(c) "cost scales with the number of weight streams per round"** — **confirmed in both directions.**

*Honesty note:* the dose leg's serial control read 0.129481 versus 0.128622 for the base `d8` leg, +0.67 %. That is leg noise — the `d8` serial within-arm spread was 1.027 %. The +7.50 ms figure uses the pooled prefill, not this leg's serial, so it is unaffected.

### 4.6 Register census (compile-only, both GPU families)

`affine_qmv_fast<bfloat16_t, 64, 4, batched>`. The crossrow path is guarded by `if (!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024)`, so **`batched=false` is the scored instantiation**; `batched=true` never reaches the changed code and stays at 57 registers in every arm.

| Arm | M=5 partition | g16s regs (local) | g16s spill | **g17s regs (ranked)** | **g17s spill** |
|---|---|--:|--:|--:|--:|
| `base` | `3+2` | 94 | 0 B | **91** | **0 B** |
| **`cap5` (this candidate)** | **`5`** | 95 | 0 B | **98** | **0 B** |
| `cap8` (NA up to 8) | `5`, `6`, `7`, `8` | 96 | **96 B** | **126** | **48 B** |

Three things follow.

1. **The advisor's register assumption is exactly right.** He priced the shared `_wide` cap going 91 → 98 g17s registers. That is precisely what the census measures. His **−0.10 %** flat register tax rests on the correct input.
2. **Zero spill at `cap5` on both families.** The tax is occupancy-only; there is no spill penalty to pay.
3. **`cap8` spills on both families** and pushes g17s to 126 registers. This is direct evidence for why the M=6..8 collapse cannot ship as-is, and it converts the "next prize" follow-up from a guess into a costed problem.

**A caveat that runs against my own result.** The ranked box pays **+7** registers (91 → 98) while my local box pays only **+1** (94 → 95). My local measurements therefore under-represent the ranked occupancy tax by roughly a factor of seven. I cannot measure the size of that tax on g17s from here. This supports subtracting the advisor's −0.10 % rather than ignoring it, and it is a reason to treat my local deltas as an upper bound on ranked transfer.

### 4.7 Ranked-corpus re-pricing (612 trees)

`research/ranked_stream_ab_board.json`, outputs `research/out/e100_ranked_stream_ab.txt` and `research/out/e100_na5_board.txt`.

- One-leg empirical null: **1.077 % sd** across 279 pairs.
- Pooled stream-removal effect: **−0.700 % ± 0.285 %, t = −2.46**.
- A constant-per-removal model beats a proportional model, ρ = **0.204**.
- **max-NA ≥ 5 has 0 groups and 0 runs** on the board. Nobody has shipped this.

---

## 5. Ranked projection — the advisor's, re-evaluated, and clearly labelled as inference

This is **inference**, not measurement. Its source tree is the advisor's cost curve and his ranked width shares; its domain is `applegpu_g17s` at 512 tokens across eight hidden prompts. My contribution is the realisation factor only.

- Advisor's pre-registered ranked gain: **+1.9 %**, band **+1.2 % to +2.5 %**, already net of the −0.10 % register tax.
- Re-evaluated at my final realisation factor **0.72**: **+1.69 %**.

The +1.69 % inherits his cost curve and width shares. It is **not** a rescaling of my local −12.81 %, and the local ratio improvement of +14.508 % is a local cancellation term that must never be presented as part of the ranked score equation. The ranked run is the only authority.

**Identity fields compared across all legs:** host, chip, memory profile, toolchain, token window, fixture, proposal head digest, oracle and depth all matched within each session. The one field that does **not** match the ranked box is the GPU family: local `applegpu_g16s` versus ranked `applegpu_g17s`. §4.6 quantifies one concrete consequence of that mismatch.

---

## 6. Conclusion

**What happened and why.** The scored 4-bit projection dispatcher partitioned M=5 verify rows into two x-groups, which streamed the backbone weights through the GPU twice per projection. Collapsing M=5 into one 5-wide x-group removes one full weight stream per M=5 round. Because these projections are bandwidth-bound, removing a stream removes close to its full cost: 22 – 23.4 ms per M=5 round by three independent end-to-end sessions, and 25.3 ms by an isolated kernel probe.

**Evidence for the mechanism.** Four independent lines agree.

1. Cross-session consistency: sessions with 10 %, 8 % and 94 % M=5 exposure all recover the same per-round constant.
2. The isolated probe reproduces the effect on the same shapes with 0 / 45 bit mismatches and a working positive control.
3. The width-tax null: widths whose generated code did not change move by +0.475 % ± 0.404 %, i.e. nothing.
4. The dose control reverses the sign by adding a stream at a different width, and its implied M=6 share matches the ledger to 0.7 points.

**Evidence against, and honest limits.** The corpus re-pricing puts the pooled stream-removal effect at only −0.700 % ± 0.285 %, far below my local −12.81 %. That is expected — the corpus averages over all widths and my `w512d4` session is 94 % M=5 — but it is the number that should temper expectations. The register census shows the ranked box pays seven times my local occupancy tax. My best local session is also the least representative of the ranked depth schedule.

**Prompt and M5 transfer risk: moderate.** The mechanism is a bandwidth property of a shape that the ranked path certainly executes, so I expect the sign to transfer. The magnitude will not. The three risks are: (i) g17s pays +7 registers where g16s pays +1; (ii) ranked M=5 share is much lower than my `w512d4` session; (iii) all end-to-end deltas here are ungated except the depth-8 leg, and the depth-8 gated delta is the small one, −0.775 % on the ABBA pair.

**Smallest useful next action.** Submit it. The mechanism is confirmed in both directions, the candidate is four lines with zero byte growth, it is exactness-clean over a 512-token window that includes real post-EOS continuation, and it composes mechanically with E98. Nothing further can be learned locally; only the M5 runner can price it.

**Recommendation: promote.** Submit to Yukon once Edward's 13:47Z slot clears. If it must be sequenced, note that E100 substitutes with E99 at M=5 rather than adding to it — see the follow-ups.

---

## 7. Suggested follow-ups, not implemented

1. **The depth schedule is now mis-tuned against its own kernel.** The arm moves the G cliff from M=4→5 to M=5→6. After the change, `w512d4` at 113.24 ms/round is *cheaper* than `w512` at 170.11 ms and yields a higher local ratio (2.2729 versus 2.2014). E94 concluded that only depths 3, 6 and 7 were ever cost-optimal under the old curve. Depth 4 is now cheap, so the schedule should be re-fitted against the new curve. This may be worth more than the kernel change itself.
2. **The next prize is M=6 at NA=6**, 2 streams → 1, covering 33.4 % of local width mass. §4.6 now prices the blocker exactly: `cap8` reaches 126 g17s registers with 48 B spill. E76 records g17s `_wide` registers for NA=2..6 as 83 / 90 / 91 / 98 / 111.
3. **Re-open rung 3 (`rows_per_simd` 4 → 2).** The advisor cancelled it citing E76 §3/§4 (`rps2` +14.16 % at NA=5, +5.18 % at NA=6). His own brief §5 says 4 → 2 has never actually been tried and could bring NA=6 and NA=8 inside the register wall. §4.6 raises the value of settling this.
4. **Time the E76 `lazyfall` arm.** It reduced g17s registers 98 → 93 at NA=5 and 111 → 99 at NA=6 with 0 B spill and clean parity on 7 shapes, and it is the only register-reducing arm that keeps `rows_per_simd = 4`. **It was never timed.** It is the cheapest route to follow-up 2.
5. **E76 `facc` is inert.** It produces byte-identical g16s machine code to the shipped kernel at NA=5 (`0a5810b4`). Close it.
6. **Re-price E98 with ρ = 0.204**, not Finding 21's 249.55 GB/s.
7. **Composition with E99** (Edward's margin gate, PR #101, +3.222 % ranked): E100 and E99 **substitute** at M=5 rather than add. If M=5 becomes a single x-group, E99's gate could clamp to depth 4 instead of depth 3. Worth one joint measurement before assuming the gains stack.
8. **Make the worker-rebuild guard the default.** Harness defect 12 is generic to every JIT kernel family, not specific to E100.
