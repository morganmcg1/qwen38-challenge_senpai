# E83 — decompose the untouched 512-token seed prefill

**Verdict: NOT USEFUL. The local seed prefill is GEMM-bound (99.7%) and close
to GEMM-optimal. The two largest named structural levers inside that GEMM
total return 7.1 ms together, which is 0.18% of the local seed leg.**

`harness=local` for every number in this file.
`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`official_or_ranked_score=false`.

## Identity

| field | value |
|---|---|
| assignment | `qwen38-r1-e83-decompose-the-untouched-prefill-leg` r2 |
| branch | `qwen-thorfinn/e83-prefill-decomposition` |
| base | `07c75a708c2347021d3148d7bc87b246ba2aec73` |
| measured on | `6acb0d152da090070b55b5120b338f0a33014e53`, the r1 base |
| host | `ip-10-231-2-95.ec2.internal`, Apple M4 Pro, 20 GPU cores, 48 GiB |
| device | `applegpu_g16s` |
| os / swift | macOS 26.5.2 / 6.3.3 (`swiftlang-6.3.3.1.3`) |
| metallib fingerprint | `7ae5c5a3d8fabe72ee19bfc09dd737281338a6be658deca49ba97eefdbe3611c` |
| prompt | `correctness_prompts/public_longcopy_gate_english_512_1024.json` |
| seed window | 512 tokens |
| head | none attached; the harness times the target path only |

The ranked host is M5 with 128 GiB. Every result here is directional for the
ranked runner and is labelled as such.

## Question

The scored candidate leg pays a 512-token seed prefill on every prompt. E3
measured that leg at 8.6–9.4% of the candidate leg, and 470 board runs had
never changed it. Where does the 4.0 s of local seed prefill go, and is any of
it recoverable?

## Sessions

| rung | job | W&B run | duration | artefact |
|---|---|---|---|---|
| 0 | static | — | — | `research/e83_prefill_accounting.py` |
| 1–2 smoke | `daa5695d` | `vml5xzkj` | n=1, superseded | — |
| 1–2 full | `cf936b97` | [`hl39g0tm`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/hl39g0tm) | 17:08:25→17:24:51Z | `research/results/e83/full.json` |
| 3 gates | `6237c445` | [`l2xex14v`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/l2xex14v) | 19:00:19→19:06:59Z | `research/results/e83/gates.json` |

Reproduce:

```bash
research/e83_prefill.sh full     # rungs 0-2
research/e83_prefill.sh gates    # rung 3
python3 research/e83_report.py research/results/e83/full.json
python3 research/e83_report.py research/results/e83/gates.json
```

## Rung 1 — where the 4.0 s goes

`begin()` unphased median **4046.1 ms** (4031.3–4062.8, n=32).
Phased median 4065.7 ms (n=10); the phase observer costs +19.6 ms.

| phase | median ms | share | dispatches | cmd buffers |
|---|---:|---:|---:|---:|
| `p1_setup` | 0.1 | 0.0% | | |
| `p2_target_forward_build` | 3048.2 | 75.0% | 2233 | 94 |
| `p3_target_forward_eval` | 995.2 | 24.5% | | |
| `p4_tail_norm_lmhead` | 21.3 | 0.5% | | |
| `p5_top_two` | 0.3 | 0.0% | | |
| `p6`, `p7` | 0.0 | 0.0% | | |
| unattributed remainder | 0.6 | 0.0% | | |

The remainder is 0.6 ms across 2233 dispatches, 94 command buffers, and 23
forced evaluation points. Every per-boundary rate is marked **NOT H-221**: the
seed prefill pays no measurable fixed per-boundary tax.

The positive control passed after `de60c02` fixed a verdict-logic bug. A
requested `usleep(20000)` delivers 25.6–29.3 ms on this host, and the control
is scored against the delivered stall, not the requested one.

### Prefill ladder discontinuity

`Qwen35TextModelInner.callAsFunction` arms `prefillLadder` at exactly
`dim(1) >= 512`, which forces 22 `asyncEval` points. Widths 481/489/497 run
ladder-off; 505/511/512 run ladder-on. Fit on the ladder-off side:
`begin_ms = 0.6544 * width + 3727.35`. Residual at width 512 is
**−17.79 ms**, inside the 31.5 ms noise band, and the sign is opposite to
H-221's predicted +7.7 ms. **The prefill ladder does not cost anything
measurable and may pay for itself.**

## Rung 2 — GEMM accounting

In-situ family tax at M=512, with the null-wrapper control (+0.99 ms)
subtracted:

| family | in-situ tax ms | share of `begin()` |
|---|---:|---:|
| `mlp_all` | 2660.6 | 65.8% |
| `gdn_in_qkv` | 391.3 | 9.7% |
| `gdn_out_proj` | 257.2 | 6.4% |
| `gdn_in_z` | 234.2 | 5.8% |
| `fa_o_proj` | 68.7 | 1.7% |
| `gdn_in_ba` | 28.0 | 0.7% |
| **`all_interceptable`** | **3611.0** | **89.2%** |

Isolated roofline, modelled over the whole 512-row seed:

| family | M | K | N | layers | ms/call | TFLOP/s | modelled ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp.down_proj` | 512 | 17408 | 5120 | 64 | 14.435 | 6.32 | 923.8 |
| `mlp.up_proj` | 512 | 5120 | 17408 | 64 | 14.343 | 6.36 | 917.9 |
| `mlp.gate_proj` | 512 | 5120 | 17408 | 64 | 14.263 | 6.40 | 912.8 |
| `gdn.in_proj_qkv` | 512 | 5120 | 10240 | 48 | 8.719 | 6.16 | 418.5 |
| `gdn.in_proj_z` | 512 | 5120 | 6144 | 48 | 5.274 | 6.11 | 253.2 |
| `gdn.out_proj` | 512 | 6144 | 5120 | 48 | 5.268 | 6.12 | 252.8 |
| `fa.qkv_packed` | 512 | 5120 | 14336 | 16 | 11.827 | 6.36 | 189.2 |
| `fa.o_proj` | 512 | 6144 | 5120 | 16 | 5.275 | 6.11 | 84.4 |
| `gdn.in_proj_a` | 512 | 5120 | 48 | 48 | 0.964 | 0.26 | 46.3 |
| `gdn.in_proj_b` | 512 | 5120 | 48 | 48 | 0.758 | 0.33 | 36.4 |
| **512-row GEMM sum** | | | | | | | **4035.4** |
| `fa.sdpa_causal` | 512 | | | 16 | 3.185 | 1.01 | 51.0 |
| `lm_head.tail_row` | 1 | 5120 | 248320 | 1 | 3.903 | 0.65 | 3.9 |
| **all executed isolated work** | | | | | | | **4090.3** |

`mlp.gate_up_fused_unused` (512/5120/34816, 28.244 ms/call, 6.46 TFLOP/s,
1807.6 ms modelled) is excluded: it is the pack the decode path builds and
prefill never calls. The four `.pin1` rows are M=1 controls, not executed work.

**GEMM share = 4035.4 / 4046.1 = 99.7% at 6.18 TFLOP/s.** The non-GEMM
residual against the 512-row GEMM sum is +10.7 ms, inside the 31.5 ms noise
band, so the honest statement is a bound: non-GEMM work in the local seed
prefill costs **at most 32 ms, at most 0.8% of the local leg**.

Add the two executed non-GEMM rows and the isolated total reaches 4090.3 ms,
which is **44 ms more than the measured 4046.1 ms `begin()`**. An isolated
sum that exceeds the thing it models is direct evidence that isolation
over-counts, because every cell pays its own `eval()` round trip while the
real forward hides those round trips behind 2233 dispatches in 94 command
buffers. Keep that sign in mind for every number in this table; rung 3 turns
it into a measured factor.

The stop rule for rungs 0–2 was "GEMM share ≥ 90% ⇒ closed". 99.7% closes it.

### Local → ranked transfer

`g = 7.69` is the local-to-ranked ratio of the whole seed leg: 4.0461 s
measured here against 0.5264 s published by the frontier run. This session's
`begin()` supersedes the campaign's inherited `P = 4.0086 s`, and the 6.15–6.18
TFLOP/s measured below supersedes the inherited 6.415 TFLOP/s. Both inherited
values were `harness=local` on a different tree and neither is cited here.

| g | n (non-GEMM speedup) | ranked non-GEMM share |
|---:|---:|---:|
| 7.69 | 1.0 | 5.7% |
| 7.69 | 1.5 | 3.9% |
| 7.69 | 3.0 | 2.0% |
| 7.69 | 7.69 | 0.8% |

Even the worst row leaves the ranked non-GEMM share small, because the local
non-GEMM bound is itself small.

## Rung 3 — the two prefill-width fusion gates

GEMM-bound is not GEMM-optimal. Two shipped fusions are gated on a row count
and therefore skip the 512-row seed:

- **G1** — `Qwen35.swift` `if S <= 9` on `fusedInProjections`. Raising it makes
  the seed run **one** `quantizedMM` at N=16480 instead of **four** at
  N=10240, 6144, 48, 48.
- **G2** — `Qwen35.swift` `if x.dim(-2) <= 16` on `fusedGateUp`. Raising it
  makes the seed run **one** 5120→34816 projection plus
  `qwen35CompiledFusedSwiGLU` instead of **two** 5120→17408 projections with a
  separate `silu` and multiply.

### Why this was safe to try

The verify ladder runs widths `1…maxDepth+1 == 9`. Both shipped bounds (9, 16)
are at or above that, so **every decode and verify width already fuses**.
Raising a bound changes the 512-row seed and nothing else, by construction.
Both fused packs are built lazily and the warm pass already reaches them at
verify widths, so raising a bound selects an already-resident pack and adds no
memory.

That claim is measured, not assumed: `qwen35FusedPackBuildCount` was 112 before
the first timed arm and 112 after the last one. No arm paid a first-use
allocation inside a timed region.

The three counters and bounds that this section reads live on the candidate
surface, so they are reverted; see "Scope and budget" for the commits that hold
and remove them.

### Design

Four arms, ABBA-counterbalanced (arm order reversed on odd reps), 8 reps, all
inside one session:

| arm | in-proj bound | gate-up bound |
|---|---:|---:|
| `gate_baseline` | 9 | 16 |
| `gate_g1` | 512 | 16 |
| `gate_g2` | 9 | 512 |
| `gate_g1g2` | 512 | 512 |

### Result

| arm | n | median `begin()` ms | range ms | unpaired saving ms |
|---|---:|---:|---|---:|
| `gate_baseline` | 8 | 4042.9 | 4039.9–4065.4 | |
| `gate_g1` | 8 | 4041.7 | 4033.7–4045.3 | +1.1 |
| `gate_g2` | 8 | 4037.0 | 4018.0–4040.7 | +5.9 |
| `gate_g1g2` | 8 | 4036.5 | 4033.0–4041.7 | +6.4 |

Each rep runs every arm, so the paired within-rep delta cancels the monotone
thermal trend that an unpaired median absorbs. Trust this form:

| arm | pairs | median saving ms | min | max | reps faster |
|---|---:|---:|---:|---:|---:|
| `gate_g1` | 8 | **+2.6** | −5.3 | +22.9 | 5/8 |
| `gate_g2` | 8 | **+5.5** | +3.0 | +30.3 | 8/8 |
| `gate_g1g2` | 8 | **+7.1** | −1.8 | +26.1 | 7/8 |

Per-rep deltas, sorted:

```
gate_g1    [-5.3, -0.5, -0.5,  1.2,  4.0,  7.5,  9.8, 22.9]
gate_g2    [ 3.0,  3.4,  3.7,  4.2,  6.8,  9.9, 28.5, 30.3]
gate_g1g2  [-1.8,  3.7,  4.3,  4.4,  9.8, 12.6, 14.5, 26.1]
```

`gate_g2` is the only arm that wins in every rep. `gate_g1` straddles zero.

**Stop rule (fixed before the run): combined saving < 40.7 ms ⇒ not useful,
report the bound and close. Measured +7.1 ms. CLOSED.**

### G1 is not numerically exact

N=48 and N=16480 straddle the `out_vec_size >= 4096` dispatch threshold at
`quantized.h:1917`, so the fused form takes a different reduction path. This
was predicted before the run and is now confirmed:

| arm | first primary token | top-two logit values |
|---|---:|---|
| `gate_baseline` | 271 | `(21, 15.6875)` |
| `gate_g1` | 271 | `(21.125, 15.6875)` |
| `gate_g2` | 271 | `(21, 15.6875)` |
| `gate_g1g2` | 271 | `(21.125, 15.6875)` |

G1 moves the top-1 logit by exactly one bf16 ulp at 21, deterministically, in
all 8 reps. The argmax token is unchanged on this prompt, but the trusted
parent receives the top-two evidence for every row, so a one-ulp shift is a
change to the scored surface and not a free win. G2's widths (17408 and 34816)
are both above 4096, so no path change and no numeric change — the exactness
model held on both sides.

**G1 is therefore disqualified twice over: it is not exact, and it is worth
+2.6 ms.**

### Why the roofline over-predicted by 16×

Predicted from rung 2: G1 ≈ 81 ms, G2 ≈ 34 ms, combined ≈ 115 ms.
Measured: +2.6, +5.5, +7.1 ms.

Two separate errors, and both are useful to record.

**1. The isolated-cell roofline is not a cost model for small-N cells.**
`gdn.in_proj_b` and `gdn.in_proj_a` run at 0.33 and 0.26 TFLOP/s in isolation,
20× below what the same kernel reaches at N≥6144. The isolated harness gives
each cell its own `eval()`, so for a 0.025 TFLOP job the measurement is
dominated by one GPU round trip, not by arithmetic. In situ those dispatches
sit inside a command buffer holding 2233 dispatches behind 94 commits, where
that fixed cost overlaps with neighbouring GEMM work. Rung 2 already had the
cross-check and I did not weight it enough: the **in-situ** `gdn_in_ba` family
tax was **28.0 ms**, against **82.7 ms** isolated. The ladder of
over-estimation is 82.7 → 28.0 → 2.6 ms.

**2. Fusion removes launches and activation re-reads, not FLOP.** The 28.0 ms
in-situ tax is the cost of the two cells *existing*. G1 does not remove them;
it merges them into a GEMM that still computes those 96 columns, which are
0.58% of the fused N=16480 output. The only recoverable terms are 3 fewer
dispatches per GDN layer and 3 fewer reads of the 512×5120 bf16 activation:
3 × 5.24 MB × 48 layers = 755 MB. The measured +2.6 ms is the right order for
that traffic and nothing like 81 ms.

The same correction applies to G2, more mildly. Its predicted 34 ms was
23 ms of "one 34816 GEMM beats two 17408 GEMMs" plus 10.5 ms of SwiGLU
fusion. The 23 ms term is a 1.3% efficiency difference read off two isolated
measurements and is inside that harness's own noise; the 10.5 ms term is real
bandwidth (one materialised 512×17408 bf16 intermediate avoided per layer).
The measured +5.5 ms is consistent with the bandwidth term alone.

**Generalisable rule for this campaign: an isolated-cell roofline over-states
recoverable time whenever the cell does not saturate the GPU, and a fusion
saving is bounded by removed traffic and removed launches, never by the
difference of two isolated cell times.**

## Ranked consequence

+7.1 ms on a 4042.9 ms local seed leg is 0.176%. The seed prefill is 8.6–9.4%
of the candidate leg, so if the fraction transferred unchanged it would move
the candidate leg by about 0.016%. Even at 3× transfer it stays near 0.05%.
That is far below submission noise, and the larger half of it (G1) is not
exact. **No part of E83 should be submitted.**

## Scope and budget

**The branch now changes no candidate byte.** Rung 3 needed one candidate file,
`Qwen35.swift`: three `nonisolated(unsafe)` globals, two call sites reading
them, and one counter increment in each of the three lazy pack-build branches.
The gates are closed, so the instrument has no further use, and a research
instrument does not belong in `Sources/` or `Vendor/`. It is reverted.

| item | commit |
|---|---|
| instrument, as measured on the r1 base | `7ef3f15` |
| instrument, replayed onto this base | `e277761` |
| revert | `313fd74` |

`git diff 07c75a70 HEAD -- Sources/ Vendor/ mtp-head.manifest.json mtp-head/`
prints nothing. `Qwen35.swift` is byte-identical to the base. Everything E83 still
carries is research-only: `Tests/MLXFastTests/E83PrefillDecompositionTests.swift`,
`research/e83_*`, `research/results/e83/`, and `senpai/known-test-failures.md`.

Two consequences of the revert, both recorded so nobody looks for a missing
instrument. The rung-3 arms in the test drove the reverted globals and cannot
compile without them, so that section is removed and the comment points at the
commits above. The `gates` profile in `research/e83_prefill.sh` now exits 2 with
the replay instructions. `research/e83_report.py` is unchanged and still reads
the recorded arms out of `research/results/e83/gates.json`.

### Gates re-verified on the rebased tree

Base `07c75a708c2347021d3148d7bc87b246ba2aec73`, the advisor head named in the
r2 feedback. It descends from `f7f356b2834518ced918f3049ca1b88afb6003f3`, which
adopts organizer commit `8b54ff11c6d686628f6534d7127a261115782757`, and adds two
record-only commits that touch `research/CURRENT_RESEARCH_STATE.md`,
`senpai/campaign-ledger.md` and `senpai/notes/r1-compose-on-8e83c6b3.md`. Both
rebases were clean; the organizer concat block lands far from every E83 region.

`Sources/`, `Vendor/` and `Package.swift` are byte-identical at `f7f356b2`, at
`07c75a70` and at this head — tree `c1280bcf` / `5d60a961` / `2896ef24` in all
three — so the worker gate and the `swift test` gate below apply unchanged to
the final head.

```
senpai/rebuild-and-assert-worker.sh: PASS
  worker_mtime  2026-08-20T20:19:11Z
  worker_sha256 db0dcafe6d58d4df3323584ac099be9c64a1e65c50546b70bf00d7dcf0b46606
  ok require-symbol qwen35DualRMSNormConcat  : 14
  ok forbid-symbol  qwen35FusedInProjMaxRows : 0
  ok forbid-symbol  qwen35FusedGateUpMaxRows : 0
  ok forbid-symbol  qwen35FusedPackBuildCount: 0

python3 research/twin_audit.py: exit 0
  29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)

senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf: exit 0
  source=2484815/3000000 headroom=515185 growth=29980/262144
  exempt=2410/2147483648 files=154

senpai/verify-ranked-score-boundary.sh: PASS
```

Every gate was re-run on the final head at 2026-08-20T20:56Z after the rebase
onto `07c75a70`. The worker gate reports PASS with the same `db0dcafe` digest
and 574,507 extracted symbols; the rebuild is a content-addressed no-op because
no build input moved.

The four witnesses are two-sided, not one-sided, so none of them is a guard
that cannot fail. The stale worker built at 19:25:49Z from the pre-rebase,
pre-revert tree, `70693f8f1d175d86fc955dc98bb180dc0df0f9ec84ea8dd1ef64f3f6c69f8809`,
reports the exact inverse: `qwen35DualRMSNormConcat` 0 and each instrument
symbol 4. The rebuild flips all four.

The reported `growth=29980` is measured against the contract base
`770a3ff2`, so it prices the organizer commits this base adopted, not E83. E83
contributes 0 bytes of candidate growth.

`swift test --force-resolved-versions` on the rebased and reverted branch head:
**705 tests in 53 suites, 40 issues, 9 failing functions**, 20.9 s. The nine
names are exactly the nine in `senpai/known-test-failures.md`:

```
contestantDocsCommandBlocksKeepTheDependencyGraphFrozen
participantDocsExposeDefaultCLIInstallDirectory
qwen36ConfigContractDigestMatchesTheReferenceManifest
startupMemoryPolicyKeepsRanked128GiBProfile
submissionStaticReviewPromptCoversMeasurementStructureExploitation
theCheckedInDeclarationSelectsThePinnedHead
theEvenMedianRuleIsTheMeanOfTheTwoCentralValues
theQwenMTPTrackIsArmedOnQwen38
theSeededCalibrationExpectationMatchesItsRecordedProvenance
```

The name set and the issue count are the gate, and both hold. The test and
suite counts are not the gate: they move with whatever tests a branch adds, and
`senpai/known-test-failures.md` now says so. Neither the organizer sync nor the
revert introduced a new failure.

`swift test` on the gates session: 2 tests, 1 suite, **0 expectation
failures**, 247.5 s.

## Suggested follow-ups (not implemented)

1. **Re-price every rung-2 "named lever" against its in-situ family tax, not
   its isolated cell time.** The isolated table is a good inventory and a bad
   cost model below saturation. The 5120→96 GDN fusion probe (49.5 ms
   modelled) is the clearest remaining example of a number that will not
   survive the same correction.
2. **The 6.18 TFLOP/s ceiling is the real prefill question.** 99.7% of the
   seed leg is one kernel family running at a fixed rate. Anything that moves
   that rate moves the whole leg; nothing structural above it will. This is a
   kernel question on the ranked M5 `_nax` variants, not a scheduling question.
3. **The prefill ladder's −17.79 ms residual deserves one cheap confirmation.**
   If forcing 22 evaluation points genuinely pays for itself, forcing more of
   them may pay again. The effect is inside the noise band at n=6, so it needs
   replicated widths, not a new mechanism.
4. **`mlp.gate_up_fused_unused` at 1807.6 ms modelled is the road not taken at
   seed width and it is now measured as nearly worthless.** Remove the
   temptation from the ledger.
