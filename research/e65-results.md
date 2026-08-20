# E65 — Cold kernel first-touch census: results

Assignment: `qwen38-r1-e65-cold-kernel-first-touch-census` (PR #68, rev r1), student `qwen-alphonse`.
Base merged during the assignment: `6cbf1a40632ea44f4eff0406d32eddf72f50d282`.
Host `applegpu_g16s` (Apple M4 Pro, `arch_gen=16`, `devc='s'`).
W&B run `cgckdu51`.

**Verdict: bounded negative.** No exploitable cold-kernel first touch exists in the scored
window. The census found four measured ceilings and one mechanism correction that is worth
more than the original question.

---

## Q1 — Is the SDPA `kL>=1024` two-pass crossing a cold cost? No.

The crossing round is always round 76, the last round of the leg. Excess over same-width peers:

| leg | crossing excess | as share of leg |
|---|---|---|
| r0-01 | +0.25 ms | 0.0015 % |
| r0-02 | +0.41 ms | 0.0023 % |
| r0-04 | +0.00 ms | 0 % |
| r1-01 | +0.30 ms | 0.0018 % |
| r1-02 | +0.32 ms | 0.0019 % |

Clean-vs-repaired offset inside a single width is +/-0.5 ms, larger than the crossing excess.

**Structural argument (generalizes to rank).** Both public fixtures carry exactly 512 prompt
tokens, and the ranked leg is 512 seed + 512 decode. `kL` therefore walks 517 -> 1024 and
touches 1024 only in the final round. One round is the whole exposure locally *and* at rank.
No warm-up can recover a cost that is paid once at the very end of the leg.

**Gate correction.** `scaled_dot_product_attention.cpp:746-748` needs **two** conditions, not
one: `(devc == 'd' || devc == 's') && k.shape(2) >= 1024`. The second disjunct
`(k.shape(1) < q.shape(1) && k.shape(2) >= 4096)` is GQA-true for this model (4 KV vs 24 Q
heads) but requires `kL >= 4096`, which the 1024-token window never reaches. That disjunct is
dead for this track.

## Q2 — Is there a first touch per draft depth? Only at `d=4`, and it is not a cold kernel.

Rule-free test: group rounds by draft depth `d`, compare the first round at a depth against the
median of the rest at that depth. No outlier rule, no tuning.

| leg | `d=4` round-1 excess | max excess at any other depth |
|---|---|---|
| r0-01 | 25.16 ms | 1.75 ms |
| r0-02 | 29.29 ms | 1.18 ms |
| r0-04 (sync-head) | 28.99 ms | 0.92 ms |
| r1-01 | 23.62 ms | 0.14 ms |
| r1-02 | 14.33 ms | 0.21 ms |

`warmAllDepthShapes` already covers `d = 3,5,6,7,8`. Combined excess per leg is 0.129-0.171 %
(r1-02 reaches 0.2616 % including a separate one-off, see Q5).

## Q3 — What is the `d=4` round-1 excess actually made of? A 512-row head prime.

Sub-phase split (`r1probe` arm, both legs `binary_witness=ok`, worker `2af5dd68...`):

| sub-phase | round 1 | median `d=4` peer | excess (leg1 / leg2) |
|---|---|---|---|
| `d_pre`+`d_flush`+`d_head1`+`d_submit1`+`d_chain` | 0.46 ms | 0.32 ms | +0.13 / +0.14 ms |
| **`d_submit2`** | 35.75 / 35.17 ms | 6.23 / 6.26 ms | **+29.52 / +28.91 ms** |
| `draft_build` total | 36.22 / 35.63 ms | 6.56 / 6.58 ms | +29.66 / +29.06 ms |

Round 1 has `headHistoryCache == nil`, so `primeCount = seedTokensForPriming.count - 1 = 511`
and it pushes 512 rows through the MTP head against 1-2 rows later. Peers take the
`flushHidden.count == 1 ? hidden : concatenated(...)` fast path and skip `applyFinalNorm` and
`concatenated` entirely. Two extra op *encodes* cost microseconds, not 29.5 ms, so the excess
is real GPU work on real data.

This explains every null in the assignment: rung-1 prewarm cannot remove real compute on real
data, `--sync-head` did not move it, and the preregistered outlier rule was hunting a kernel
compile that never existed.

## Q4 — Rung 1 (warm the `d=4` shape): falsified, reverted.

base `{25.16, 29.29}` ms vs r1warm `{23.62, 14.33}` ms. Non-overlapping, but n=2 per arm and the
r1warm within-arm spread is 9.29 ms. **No effect claimed.** Reverted in `3cd80e3`; the negative
is recorded in a source comment so the next agent does not retry it blindly.

## Q5 — A stochastic host stall of the same magnitude exists.

r1-02 round 27: +32.7 ms `draft_build` at `d=5`, `over_width_wall=True`, `streak=1`. Not a first
touch — `d=5` first occurs at round 2 and was already warm at 11.43 ms. Seen in 1 leg of 5. This
floors single-leg resolution of any ~25 ms event and is why Q4 cannot be resolved at n=2.

---

## Corrections to standing campaign beliefs

**Correction 1 — `draft_build` is not host graph build.** `d_submit2` is 24.56/24.83 ms of the
25.3 ms median, i.e. **96.9-97.8 %** of the section; all graph-construction phases together are
~0.5 ms. The cost is `async_eval` -> `eval_impl`, where `transforms.cpp:270` blocks in
`scheduler::wait_for_one()` once `n_active_tasks() > MAX_ACTIVE_TASKS = 10`
(`transforms.cpp:25`). It is GPU back-pressure. This retires the standing source comment
claiming per-step cost is "host graph BUILD, not GPU work to overlap".

Corroboration: `--sync-head` transfers 12.8 ms almost exactly between sections
(`draft_build` 6.56 -> 19.33 = +12.77 ms; `verify_build` 69.72 -> 57.30 = -12.42 ms).

**Not separated, and I state it plainly:** the steady-state split of `d_submit2` into host
encode versus GPU wait. Only round 1 is separated.

**Correction 2 — the round-1 excess is irreducible GPU work,** not a warm-up artefact. See Q3.

**Correction 3 — the SDPA two-pass gate has two conditions,** not one. See Q1.

**Correction 4 — MLX's own arch comment is wrong.** `device.cpp` says `case 'g': // base, pro`.
Pro groups with Max under `'s'`, not with base under `'g'`: M4 Pro is `g16s` and M5 Pro is
`g17s`. Any reasoning that assumed "Pro => 'g'" is incorrect.

---

## Bonus — complete timed-leg budget (first in the campaign)

| segment | share of timed leg |
|---|---|
| `verify_build` | 36.0 % |
| `eval_wall` | 32.6 % |
| `begin` prefill + tail | 23.0 % |
| `draft_build` | 8.2 % (97 % of it `d_submit2`) |
| everything else | < 0.3 % |

Timed leg 17.344 / 17.354 s; rounds are 77.0 % of the leg.

## Bonus — prefill roofline (`research/e65_prefill_roofline.py`)

Shapes taken from `Qwen35Weights.swift` and the `Qwen35Config.swift` `expect()` block.

- matmul params 24.35 B, embed + lm_head 2.54 B, total **26.89 B**, which matches the "27B"
  name and validates that the inventory is complete.
- Prefill for 512 tokens = **24.99 TFLOP** (projections 24.93 T, attention 0.052 T,
  lm_head 0.0025 T).
- Measured `begin` 4.0 s => **6.25 TFLOP/s = 83.2 %** of the 7.51 TFLOP/s bf16 peak measured on
  a sibling host.
- At 100 % of peak, prefill would still take 3.33 s, so the **absolute ceiling on any prefill
  work is 0.67 s**.

=> Prefill headroom is in overheads, not in the GEMMs. This rules out a kernel rewrite and makes
the undosed prefill `asyncEval` ladder the right sub-question.

Caveats: the backbone is affine 4-bit group-64, not bf16, and the peak figure comes from a
sibling host.

**Advisor sub-question (a), answered:** `begin` costs the same in serial and MTP sessions —
build 2.945-2.976 s and `eval_wall` 1.047-1.049 s across 3 legs x 2 session types
(6 measurements). There is **no candidate-side prefill tax in `begin`**. The MTP-specific tax is
the 29.5 ms head prime in scored round 1.

**Withdrawn:** my earlier "host is the bottleneck 3x" reading. `Qwen35.swift:2196-2197` sets
`prefillLadder = inputs.dim(1) >= 512` and calls `asyncEval` at layer 0 and every `i % 3 == 2`
(~22 submissions across 64 layers), so GPU prefill runs inside that interval. What survives:
prefill is 23 % of the candidate leg and does **not** cancel at rank, because the numerator is
the pinned serial build: `raw = 38.12 / (17.37 - X)`, so X = 0.5 s is about +3 %.

---

## Ranked-host GPU architecture (optional sub-question) — determinable

| chip | `MTLDevice.architecture.name` | final char |
|---|---|---|
| Apple M5 (base) | `applegpu_g17g` | `g` |
| Apple M5 Pro | `applegpu_g17s` | `s` |
| Apple M5 Max | `applegpu_g17s` | `s` |

Evidence: first-hand printouts in ml-explore/mlx#3897 (base M5 `g17g`; M5 Max `g17s`) and
ml-explore/mlx-lm#1206 (M5 Max `g17s`); the Apple Wiki entry for `Mac17,9` ("Apple G17S"); and
an Xcode 26.4b3 Metal toolchain table. Cross-checked on this host against Apple's own driver
string table in `/System/Library/Extensions/AGXMetal*.bundle`, which contains exactly
`g17g g17p g17s` for G17 and no `g17d`/`g17x`, and against `xcrun metal -arch <name>`, which
accepts those three and rejects the others. The M4 Pro anchor confirms the bundle-to-letter
mapping: this host is served by `AGXMetalG16X` and reports `g16s`, so `AGXMetalG17X` -> `g17s`.

**Residual gap, stated honestly:** the tier of the ranked `m5-qwen38-27b-mtp` host is unknown,
and no first-hand printout for M5 Pro specifically was found. **I did not probe the ranked
runner.**

Consequences for this track:

1. If the ranked host is a **base M5** (`devc == 'g'`), the SDPA two-pass branch at
   `scaled_dot_product_attention.cpp:748` is **never taken** at any window size this track
   reaches. Q1's mechanism would then not merely be too small at rank, it would not exist.
   If the host is M5 Pro/Max (`'s'`), it behaves like this M4 Pro box. Q1's negative holds
   either way.
2. `arch_gen` is 17 for both tiers, so `_nax` gating is tier-independent:
   `device.cpp:926` gives `can_use_nax &= gen >= (arch == 'p' ? 18 : 17)` plus a
   macOS 26.2 availability check. Both `g17g` and `g17s` enable `_nax`.
3. Other tier-sensitive dispatch sites, should a future experiment need them:
   `matmul.cpp:89` (small-device tile params for `'g'`/`'p'`), `matmul.cpp:209` and `:280`
   (larger-device `bk`/`bm` routing for `'s'`/`'c'`/`'d'`), `matmul.cpp:919`
   (`min_tmn_threshold` 2048 vs 1024).

**Actionable tooling, source-verified in our own vendored copy:** MLX reads
`MLX_METAL_GPU_ARCH` (`utils.h:205`) and `device.cpp:560-562` consults it *before* the real
device string. Setting it to `applegpu_g17g` or `applegpu_g17s` forces the corresponding
dispatch predicates on this M4 Pro box. It changes dispatch selection only, never the silicon,
so it is a legality-safe way to check which branch a future candidate would take at rank. I
verified the code path by reading our vendored source; I did not run a timed arm under it.

---

## Methodology disclosures

1. The preregistered `(M, repaired)` 3xIQR outlier rule **missed round 1**. Its `M5/clean` cell
   has n=3, and cells with n<4 have no usable IQR. I left the rule **unchanged** and added the
   depth first-touch test and a labelled pooled channel **beside** it, rather than retuning the
   rule after seeing the data.
2. `e65_run_leg.sh` asserts the timed binary after each run (worker and CLI sha256 against
   `provenance.txt`), writes `binary_witness`, and exits 90 on mismatch.
3. All census legs are **traced**, so their scores are not comparable with untraced receipts.
   Instrumentation is free: base `{2.1947, 2.1869}` vs traced r1probe `{2.1913, 2.1888}`, means
   differ by 0.03 %.
4. Entry GPU temperatures 36.9 / 57.9 / 56.0 C. The coldest leg scored highest. Largest
   same-arm score spread across 3 base legs is **0.36 %**.
5. Projection to rank: the event occurs once per leg locally and once per leg at rank, so
   `f_ranked / f_local = 1` and the projected ranked delta is about the local **+0.15 %**. That
   is roughly 25 % of our 0.5993 % deficit but below the 0.756 % single-run standard deviation.

## Exactness

All 8 legs: `all_tokens_matched=true`, `residual_divergence_count=0`, 512 decode tokens,
`passed=true`.

| leg | score |
|---|---|
| r0-01 / r0-02 / r0-04 | 2.1947 / 2.1869 / 2.1897 |
| r1-01 / r1-02 | 2.1992 / 2.1930 |
| r1b-01 / r1b-02 | 2.1913 / 2.1888 |
| r4-01 `--local-submit` | 2.1938 |

r4-01 also reports `public_drift_tripwire_passed=true` and `uses_pinned_mtp_head=true`, head
`62516c6f...`.

Row ledger: `mtp-verify: rows=513 seed_tokens=512 self_consistent=true chain_contradictions=0`;
`mtp-timed: depth=0 rounds=512 reference_checked_rows=512/512`; `mtp-timed: depth=8 rounds=76
accepted_draft_rate=0.8808 reference_checked_rows=571/571`. Row counts agree exactly.

`e65-r0-03-census640` failed by design: `benchmark-qwen-mtp.sh:120` caps both local modes at 512
decode tokens.

## Gates against base `6cbf1a40632ea44f4eff0406d32eddf72f50d282`

- `python3 research/twin_audit.py` — OK, 29 runtime-effective twins, 1 allowlisted waiver.
- `senpai/validate-assignment-scope.sh` — OK, 1 submitted path.
- `senpai/check-editable-budget.sh` — OK: `source=2463704/3000000 headroom=536296
  growth=1650/262144 exempt=2410 files=154`.
- `senpai/verify-ranked-score-boundary.sh` — PASS.
- `swift test --force-resolved-versions` — 688 tests, 9 failing tests / 40 issues, all
  pre-existing campaign-base failures unrelated to this diff:
  `startupMemoryPolicyKeepsRanked128GiBProfile`,
  `submissionStaticReviewPromptCoversMeasurementStructureExploitation`,
  `qwen36ConfigContractDigestMatchesTheReferenceManifest`,
  `participantDocsExposeDefaultCLIInstallDirectory`,
  `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen`,
  `theCheckedInDeclarationSelectsThePinnedHead`, `theQwenMTPTrackIsArmedOnQwen38`,
  `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues`,
  `theSeededCalibrationExpectationMatchesItsRecordedProvenance`.
  These cover campaign doc markers, the tuned memory profile, the head manifest, and calibration
  digests. None touch `Qwen36MTPBlockSession`. Caveat: the count matches the nine the advisor
  recorded at the previous base, but I did not re-run the base arm as a control on the moved
  base.

Submittable diff is 26 lines in `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`: five
`let tXxx = Self.traceRounds ? DispatchTime.now().uptimeNanoseconds : 0` guards and six trace
fields. No scored-path behaviour change. Everything else is under `research/`.

## Suggested follow-ups (not implemented)

**(a) Shorter head prime.** The 512-row prime is necessary but its *size* may not be. Priming K
rows instead of 511 cannot change emitted tokens, because the target verifies every draft, so a
worse draft costs acceptance rate and not correctness. Stop rule: sweep K and weigh accepted
tokens per round against the 29.5 ms saved. Ceiling about +0.17 %.

**(b) Prefill.** Sub-question (a) is already answered above: no `begin` excess. Sub-question (b)
is the undosed prefill `asyncEval` ladder over {0, 4, 11, 22, dense}, bit-identical by
construction. The roofline says overheads, not kernels, with a 0.67 s ceiling.
