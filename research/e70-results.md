# E70 — Local-to-ranked dispatch divergence audit

PR #73 · assignment `qwen38-r1-e70-local-ranked-dispatch-divergence-audit` rev `r1`
BASE_SHA `bdfbc4e92c93d216503980fb46258ff0b314145a` · base_ref `senpai/qwen38-mtp-r1`
W&B run [`k3iv3ylg`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/k3iv3ylg)

Host: Apple M4 Pro, 48 GiB, macOS 26.5.2, `applegpu_g16s`, `arch_gen = 16`,
`devc = 's'`, low startup memory profile. Ranked host: M5, `applegpu_g17s`,
`arch_gen = 17`, high memory profile.

Every claim below carries a harness label:
`harness=local` (this M4 Pro), `harness=ranked` (the M5 runner or a ranked
receipt), `harness=arch-probe` (this M4 Pro with `MLX_METAL_GPU_ARCH` forced).

**No candidate file changed.** `git diff BASE..HEAD` touches `research/` and one
new test file only. Full scope evidence is in section 7.

---

## Question

Where does the MLX host dispatcher choose a different kernel on this local M4
Pro than on the ranked M5, is that site on the scored path, and what is the
resulting local measurement worth in ranked score?

## Evidence that made it worth testing

The campaign prices local savings at rank with transfer constants. Those
constants are fitted, and no record proved which code sites actually differ
between the two hosts. `is_nax_available()` keys on `arch_gen >= 17`, so the M5
can reach a whole kernel family this host cannot. Until the divergent set is
enumerated, every local measurement carries an unquantified transfer risk, and
E65's follow-up list contained at least one item whose value depended entirely
on that risk being real.

## Expected result

A small divergent set. The useful outcome is either a named site that
invalidates a standing local measurement, or proof that the divergent set is
narrow enough to retire the transfer-risk caveat for the rest of the campaign.

## Smallest decisive test

Three rungs, cheapest first.

- **Rung 0**, source only: enumerate every architecture-conditional branch in
  the vendored dispatcher, evaluate each predicate for M4 Pro, M5 and base M5,
  and machine-check the table with mutation controls.
- **Rung 1**, empirical: capture the actual Metal kernel name for each scored
  shape twice — once on the real device, once with
  `MLX_METAL_GPU_ARCH=applegpu_g17s`.
- **Rung 2**, consequence: convert only the divergent *and* reachable sites into
  published-score percent using measured ranked receipts.

## Stop or promotion rule

Report and stop. The assignment names the deliverable as the follow-up
experiment stated precisely, not the follow-up experiment run. Section 9 names
two. Neither is started.

---

## 1. Rung 0 — the source table (`harness=local`, source-derived)

`research/e70_dispatch_divergence_audit.py` → `research/e70-rung0.json`.

**35 of 35 structural checks pass. 24 of 24 mutation controls flip.** A mutation
control rewrites one predicate input and asserts the verdict changes, so the
table cannot silently pass by reading the wrong line.

15 sites, not 14. `device.cpp:572` is a second, separate read of the
architecture string and it needs its own row.

| id | file:line | verdict on M4 Pro vs M5 |
|----|-----------|-------------------------|
| S1 | `device.cpp:913` | **diverges** |
| S2 | `device.cpp:560` command-buffer tier | identical vs M5 Pro/Max, differs vs base M5 — **memory profile, not architecture** |
| S3 | `quantized.cpp:84` `get_qmv_batch_limit` | identical, `vector_limit = 10` on all three arms and all 7 linears |
| S4 | `quantized.cpp:697` qmm nax gate | **diverges** |
| S5, S6 | — | unreachable |
| S7 | `matmul.cpp:176` | **diverges** |
| S8 | `matmul.cpp:373` | **diverges** |
| S9 | `matmul.cpp:915` | **diverges** |
| S10–S13 | — | unreachable |
| S14, S15 | `device.cpp:572` and successor | identical vs M5 Pro/Max, differs vs base M5 |

Five sites diverge. Six are unreachable on every host.

`S13 sdpa.cpp:177` is unreachable everywhere:
`sdpa_full_supported_head_dim` accepts head dimensions 64, 80 and 128 only, and
this checkpoint has head dimension 256. No host in this campaign can take that
branch.

## 2. Rung 1 — the measured kernel names (`harness=arch-probe`)

`Tests/MLXFastTests/E70ArchDispatchProbeTests.swift` (503 lines, gated behind
`MLXFAST_E70_ARCH_PROBE=1`), driven by `research/e70_run_arch_probe.sh`, diffed
by `research/e70_rung1_diff.py` → `research/e70-rung1-diff.json`.

**34 cells × 2 arms. Zero aborts. Zero forced-arm failures. Zero cells not
captured.** One cell per process, so a crash in one cell cannot corrupt another.
Job `0eafeb43-9419-4474-817e-38c3b4828f5f`, exit 0, 343 s.

**13 of 34 cells change kernel** under the forced architecture.

### 2.1 The nax cliff is a step at M=10 with a flat shelf to M=32

| M | real `applegpu_g16s` | forced `applegpu_g17s` | changed |
|---|----------------------|------------------------|---------|
| 1, 5, 9 | `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` | same | no |
| 10, 12, 511, 512 | `affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0` | `affine_qmm_t_nax_...bm64_bn64_bk64_wm2_wn2_alN_*` | **yes** |

M ≤ 9 stays on the qmv path on both arms, because `vector_limit = 10` is
identical on both (rung 0, S3). At M = 10 the qmm path opens and the M5 takes
the nax variant this host cannot reach.

### 2.2 Only wide-output families cross

| shape | scored user | real | forced | changed |
|-------|-------------|------|--------|---------|
| K=5120 N=14336 | `fa.qkv` fused | `affine_qmm_t` | `affine_qmm_t_nax ... alN_true` | **yes** |
| K=5120 N=16480 | `gdn.in_proj` fused | `affine_qmm_t` | `affine_qmm_t_nax ... **alN_false**` | **yes** |
| K=5120 N=34816 | `mlp.gate_up` fused | `affine_qmm_t` | `affine_qmm_t_nax ... alN_true` | **yes** |
| K=5120 N=248320 | `lm_head` | `affine_qmm_t` | `affine_qmm_t_nax ... alN_true` | **yes** |
| K=6144 N=5120 | `gdn.out_proj`, `fa.o_proj` | `affine_qmm_t_splitk` | `affine_qmm_t_splitk` | no |
| K=17408 N=5120 | `mlp.down` | `affine_qmm_t_splitk` | `affine_qmm_t_splitk` | no |
| K=5120 N=5120 | square control | `affine_qmm_t_splitk` | `affine_qmm_t_splitk` | no |

**The narrow-output families never reach nax on either arm.** They stay on
`affine_qmm_t_splitk`. This is exactly what the advisor predicted from source.

**`gdn.in_proj` (N=16480) is the only scored family that takes the *unaligned*
nax variant**, `alN_false`, because 16480 is not a multiple of 64. Every other
crossing family is N-aligned. That is a distinct kernel with a distinct
performance profile, and it is invisible on this host.

M=10 and M=12 route identically at every shape, on both arms.

### 2.3 Attention and dense GEMM

- `sdpa_prefill_512`: `block_softmax_precise_bfloat16` +
  `steel_gemm_fused_{nn,nt}_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2` →
  `steel_gemm_fused_nax_{nn,nt}_..._bm64_bn128_bk256_wm2_wn4`. **Changed.**
- `dense_matmul_m511`: `steel_gemm_fused_nt_...bm64_bn64_bk16_wm2_wn2` →
  `steel_gemm_fused_nax_nt_...bm64_bn128_bk256_wm2_wn4_..._align_M_n_align_N_t_align_K_t`.
  **Changed.**
- `sdpa_vector_q1_k768`, `sdpa_vector_q5_k768` →
  `sdpa_vector_bfloat16_t_256_256_nomask_qnt_{nc,c}_nosinks`. Identical.
- `sdpa_vector_*_k1030` → `sdpa_vector_2pass_1_..._128` +
  `sdpa_vector_2pass_2_bfloat16_t_256`. Identical.
- `dense_gemv_m1` → `gemv_al_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`.
  Identical.

### 2.4 Probe validity

`MLX_METAL_GPU_ARCH` is read at `mlx/utils.h:205` and consulted at
`device.cpp:560` before the real device string. `arch_gen_` is parsed from that
same string at `device.cpp:566-572`, and `devc = arch_.back()`. The forced
architecture therefore moves **both** the string and the generation, so one
environment variable covers all 15 sites. Confirmed empirically: 34 forced cells,
zero crashes, zero fallbacks.

The probe changes kernel *selection* only. It does not make this GPU execute the
nax instruction set, so the probe proves **which** kernel the M5 picks and never
**how fast** it runs.

## 3. Verification of the advisor's `qmm_splitk` fork table

Checked line by line against
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:776-810` and
its caller at `:1415-1424`. The table is accurate. Two additions:

- **`mlp.down` needs the alignment loop.** `17408 % 192 ≠ 0`, so the split
  count walks 3 → 2.
- **K=5120 N=5120 also walks 3 → 2** for the same reason.
- **M=10 and M=12 route identically** because `m_tiles = ceil(M/32) = 1` for
  every M ≤ 32. The cliff is a step at M=10 and then a flat shelf all the way to
  M=32, not a ramp. Rung 1 confirms this empirically at both M values.

## 4. The transfer constant (`harness=ranked`)

`research/e70_transfer_constant_provenance.py` →
`research/e70-transfer-constant.json`.

Source data: `research/ranked_stream_ab_board.json`, **471 submissions × 8
prompt keys = 3768 ranked serial legs**. Mean seconds/token 0.03799564539,
sd 0.2225 %, min 0.0377260451, max 0.0385442637. Mean leg 19.4538 s.

### 4.1 🔴 RETRACTION — my refutation of 30.402 ms was wrong

An earlier revision of this report claimed that the 30.402 ms constant at
`campaign-ledger.md:10453` was refuted, because it implies a 16.092 s leg that
none of the 3768 board samples reach. **I retract that claim.** The arithmetic
is right and the population is wrong.

`30.402 ms` is not a pinned serial round. It is the **candidate build's**
depth-0 round. It is calibrated in `research/prompt_round_reconstruction.py`
from `row["mtp_spt"]`, the candidate seconds per token. I compared it against
the pinned serial leg population, which is a different build. The measured
source does exist, and it reproduces:

```bash
python3 research/prompt_round_reconstruction.py \
  --facts research/e53-board-facts.json --submission ca9251b8 --prefill-ms 526.6
```

`calibrate_depth0_ms()` at `:111-156` reads `row["mtp_spt"]` at `:134`, anchors
on plutarch (92.2 % non-drafting, 487 rounds), and resolves one unknown by
fixed point. Recomputed live it returns **30.40205 ms**, which is the ledger
value to 1.6e-4 %. The same script emits an exact score decomposition
(`build × spec × dilution = raw_ratio`) that closes to ~1e-11 on all eight
prompts.

My implied 16.092 s leg is also real. It is the leg **our candidate** would run
on plutarch if every round were depth-0. The measured plutarch candidate leg is
15.517 s, 3.6 % below it, and the gap is exactly what plutarch's 38 drafting
rounds add.

**A model-free bound settles it with no transfer model at all.** plutarch runs
487 rounds, 449 of them depth-0, at a mean round of 30.781 ms. A drafting round
cannot be cheaper than a depth-0 round, so `c1 ≤ 30.781 ms` is hard. 30.402
sits 1.23 % under that ceiling. The pinned serial 36.963 sits 20.09 % over it
and cannot be `c1`. **`R(depth-0) = 65.009 / 30.402 = 2.1383` stands.**

### 4.2 Both constants are correct; they measure different builds

| constant | value | population |
|----------|-------|-----------|
| pinned serial depth-0 round | **36.963 ms** | 3768 board serial legs |
| candidate depth-0 round | **30.402 ms** | reconstruction of `ca9251b8` |
| build factor | **1.2158** | serial ÷ candidate |

The pinned serial round is confirmed to **0.031 %** by three independent
routes: beagle receipt 36.9573, board mean minus the ledger's K high
36.9640, board mean minus K low 36.9689. It remains the best-supported
constant available; it simply is not the denominator of `R`.

My error was to build `tau_depth0_round = 65.0094 / 36.9573 = 1.7590`, which
divides the local **candidate** round by the ranked **serial** round. That
mixes builds and is withdrawn. Every consequence I drew from it — the 4.309×
contrast, the ÷3.91–3.96 divisor, the 1.92–1.94 latency multiplier, and the
"every R-based projection is 9.4–10.4 % too high" claim — is withdrawn with it.

**The surviving defect is real and separate.** Ledger 188(A) still defines `R`
as a **leg** ratio ("Let L be leg time") and then computes it from two **round**
times. That defect does not depend on which round constant is used.

### 4.3 `R` is width dependent — `R(M)` is the pricing table

`R` is not one number. Dividing the local E1 round ladder by the reconstructed
ranked round at the same verify width gives:

| M | prompt | local round ms | ranked round ms | **R(M)** | rounds | tok/rd | candidate leg ms |
|---|--------|---------------:|----------------:|---------:|-------:|-------:|-----------------:|
| 1.0000 | (depth-0 control) | 65.009 | 30.402 | **2.1383** | 512 | 1.0000 | — |
| 1.1540 | plutarch | 65.851 | 30.781 | 2.1394 | 487 | 1.0513 | 15516.8 |
| 3.2976 | drama | 80.212 | 38.092 | 2.1058 | 252 | 2.0317 | 10125.7 |
| 3.6557 | travel | 85.859 | 39.511 | 2.1730 | 212 | 2.4151 | 8903.0 |
| 5.5327 | beagle | 125.800 | 53.332 | 2.3588 | 107 | 4.7850 | 6233.1 |
| 5.7677 | medicine | 130.260 | 53.476 | 2.4358 | 99 | 5.1717 | 5820.7 |
| 6.2697 | republic | 139.927 | 58.421 | 2.3951 | 89 | 5.7528 | 5726.1 |
| 6.4253 | essays | 142.962 | 60.196 | 2.3749 | 87 | 5.8851 | 5763.7 |
| 6.7765 | botany | 149.810 | 60.548 | **2.4742** | 85 | 6.0235 | 5673.2 |

`M = 1 + proposed / rounds` is the verify width. `tok/rd = 1 + accepted /
rounds` is what the 512-token window divides by, so `rounds + accepted = 512`
closes exactly on all eight prompts. Local round cost is E1's depth-0 round plus
its marginal ladder, interpolated at fractional depth.

`R(M)` is flat at **2.11–2.17 for M ≤ 3.66** and flat again at **2.36–2.47 for
M ≥ 5.53**. Group means are 2.1394 and 2.4078, a step of **12.55 %**; measured
edge to edge the step is **8.55 %**, against within-group scatter of 3.14 % and
4.79 %. The step sits exactly where the local ladder jumps `E1(4) = 91.29 ms →
E1(5) = 115.69 ms`, **+26.7 %**.

Two caveats, both carried into rung 2:

1. Ranked rounds are means over a width **mixture**, while local `E1` is a
   fixed width. That biases `R(M)` **downward** for high-spread prompts.
2. The public receipt gives no ranked round census, so the mixture cannot be
   deconvolved.

The caveat cuts **against** the step, so the step is more likely real than not.
The working reading is that our M4 Pro carries an M=4→M=5 width penalty that the
ranked M5 carries much less of. That question is **named as a follow-up in
§9.2 and deliberately NOT opened here.**

### 4.4 The adopted pricing rule

```
delta_ranked_ms = delta_local_ms / R(M)          # M = the width you measured at
delta_score_pct = 100 * delta_ranked_ms * rounds_at_M / ranked_candidate_leg_ms
```

**Report `R(M)` and `M` beside every converted number. An unlabelled conversion
is invalid.** The normalizer is the **candidate** leg, because a candidate-side
saving moves only the score denominator. This form needs no leg ratio and
cannot pick up the wrong leg, so the single `R` is retired in favour of `R(M)`
plus the direct form.

The indicative candidate-leg ratio on beagle is 2.839. It is **not** comparable
with `R(M)`: it is a leg ratio across two different width mixtures. I decline to
promote it to a constant.

## 5. Rung 2 — consequences (`harness=ranked`)

`research/e70_rung2_consequences.py` → `research/e70-rung2.json`.

### 5.1 The median-pair model

Built from the per-prompt ranked receipt of submission `ca9251b8` in ledger
186(B): leg_ms, build, spec and dilution for all eight prompts. Sorted by raw
ratio, the median pair is **beagle (6233 ms) and medicine (5821 ms)**.

**Self-check: the model reproduces the published 3.23250848 to 7.11e-6 relative
error.** Score sensitivity is **0.016631 % per ranked millisecond**.

Both sites are **once-per-leg** costs, so `rounds_at_M = 1`. The 511-row head
prime fires on the first drafting round only; the SDPA fallback fires inside the
single seed prefill. Neither is a per-round cost.

`R(M)` at the median pair, labelled as the rule requires:

| prompt | M | R(M) | rounds | candidate leg ms |
|--------|--:|-----:|-------:|-----------------:|
| beagle | 5.5327 | 2.3588 | 107 | 6233.1 |
| medicine | 5.7677 | 2.4358 | 99 | 5820.7 |

### 5.2 The two divergent, reachable sites

| | S4 decode head prime | S9 prefill SDPA fallback |
|---|---|---|
| local saving if removed entirely | 29.215 ms | ≤ 16.49 ms |
| adopted transfer rate | `tau_prefill` 7.5798 | `tau_prefill` 7.5798 |
| `rounds_at_M` | 1 | 1 |
| ranked saving | 3.854 ms | 2.176 ms |
| **route A, median pair** | **0.0641 %** | **0.0362 %** |
| route B, direct form, same transfer rate | 0.0640 % (−0.12 %) | 0.0361 % (−0.12 %) |
| route B, direct form, `R(M)` | 0.2024 % (3.16×) | not applicable |
| naive, ignoring the transfer | 0.1684 % (2.63× too high) | 0.0951 % (2.63× too high) |
| vs published-score sd 0.756 % | 0.085 sd | 0.048 sd |
| vs our 0.61 % deficit | 10.5 % | 5.9 % |
| steerable by editable code | **no** | **no** |

S9's FLOP accounting: the composed fallback runs 103.1 GFLOP per seed, which is
0.412 % of prefill FLOPs. `R(M)` is **not applicable** to S9: that cost sits
inside the seed prefill, not inside a decode round, so a decode-round ratio is
the wrong category of constant. Prefill has its own measured transfer rate.

**Do the two routes agree?** Yes, on the part that can be compared. Hold the
transfer constant equal and the two leg models differ by **0.12 %** on both
sites. The whole remaining gap is the choice of transfer constant, and §5.3 is
what decides it.

### 5.3 The transfer band, and what the audit buys

S4 is the interesting one. Its price depends entirely on which constant it
takes:

| divisor | value | M | ranked saving | score |
|---------|------:|--:|--------------:|------:|
| `tau_prefill` (adopted) | 7.5798 | 511 | 3.854 ms | **0.0641 %** |
| `R(M)` largest, drama | 2.1058 | 3.30 | 13.874 ms | 0.2307 % |
| `R(depth-0)` | 2.1383 | 1.00 | 13.663 ms | 0.2272 % |
| `R(M)` at the median pair | 2.3588 / 2.4358 | 5.53 / 5.77 | 12.385 / 11.994 ms | 0.2024 %\* |
| `R(M)` smallest, botany | 2.4742 | 6.78 | 11.808 ms | 0.1964 % |

\* route B direct form. Route A with the same two divisors gives 0.2027 %, a
0.15 % difference, which is the same leg-model agreement reported in §5.2.

That is a **3.06–3.60× spread**, and the whole question is whether the head
prime reaches the nax family at rank. **Rung 1 answers it.** The prime runs at
M=511, and rung 1 shows M=511 taking `affine_qmm_t_nax` and
`steel_gemm_fused_nax` under the ranked architecture — exactly the families
that give prefill its 7.58×. The prime is ~100 % GEMM at M=511, against
prefill's 84 % at M=512, so if anything it is *more* compute-bound than
prefill. The `R(M)` decode-round branch is excluded. **0.0641 % is the
number.**

Note what this band is doing. The `R(M)` rows are a **floor** in the sense of a
worst case for the audit's conclusion, not a competing estimate: they price the
prime as if it ran at a decode round's transfer rate. Rung 1 is measured
evidence that it does not.

### 5.4 Upper bound

**If both divergent reachable sites cost exactly zero at rank, the published
score moves 0.1003 %.** That is 0.133 sd of one published score and 16.4 % of
our 0.61 % deficit to the crown.

### 5.5 Rung-2 question 1 — what each divergent site invalidates or weakens

The assignment asks this per site, and asks for bluntness.

**S1 `device.cpp:913`, the master nax switch.** Invalidates nothing on its own.
It is the switch every other site reads, not a dispatch site.

**S4 `quantized.cpp:697`, the qmm nax gate.** Weakens exactly two things, and
invalidates neither.
(i) E65's head-prime pricing, by 2.63×, as section 6.2 sets out.
(ii) E68 rung 1's `C(10) = 271.147 ms` against `C(9) = 163.621 ms`, the +107.526 ms
step that produced the standing rule "never widen the verify past 9". Rung 1
proves four of the seven scored families take `affine_qmm_t_nax` at M=10 on the
ranked host, so **that step is a local-only measurement of kernels the ranked
host does not run.** The rule itself survives, but for a different reason than
the one recorded: `program.md` caps drafts at eight, so M ≤ 9 by contract and the
cliff is unreachable by any legal schedule. The rule should be re-recorded as a
contract fact, not as a measured cost. **It invalidates no promoted result**,
because no promoted candidate ever runs M ≥ 10.
Everything else in the QMV programme is untouched: `t55`, `t6`, E55, E61, E63,
E64 and E69 all live at M ≤ 9, where rung 1 shows `affine_qmv_fast` on both arms.

**S7 `matmul.cpp:176` and S8 `matmul.cpp:373`, the tile parameters.** These are
the two halves of one fork; S7 is ranked-only and S8 is local-only. They weaken
any local claim about dense-GEMM tile behaviour, of which this campaign has
none. **None.**

**S9 `matmul.cpp:915`, the dense family selector.** Weakens E57's eight-dispatch
prefill enumeration (`:10299-10324`) as a *ranked* description: the dispatches
are correctly counted, but two of the families are `steel_gemm_fused_*` locally
and `steel_gemm_fused_nax_*` at rank with different tiles. E57's conclusion —
that prefill attention is composed rather than fused — is unaffected and is in
fact reinforced. It also corrects E65's prefill roofline by 2× (section 6.4).
**No timed result is invalidated**, because 186(C) already treats ranked prefill
as an unmeasurable section and prices it with `tau` rather than with local time.

**S2, S14, S15.** Identical on M5 Pro and M5 Max, divergent against a base M5.
Since the ranked tier is still unconfirmed, these remain **live tier risk on
three sites** rather than a resolved divergence. They invalidate nothing today.

### 5.6 What the width shares do not do

The modelled width shares (M4 14.2, M5 24.1, M6 33.4, M7 12.2, M8 7.35, M9 5.75)
apply to **no** divergent site. Every modelled width is ≤ 9, and the nax cliff
starts at M=10. Those shares are a fitted model output
(`research/e53_width_mixture.py`, ledger 200(D):14830), and ledger 184(D):10219
already proved the ranked width histogram is unidentifiable. They enter no line
of section 5.

## 6. Findings the advisor asked to be written as findings

### 6.1 Named finding — the SDPA transfer risk is retired

> **No local SDPA measurement in this campaign describes a kernel the ranked M5
> does not run.**

Support: `sdpa_vector` at q=1, q=5 and kL=768 is identical on both arms;
`sdpa_vector` at kL=1030 takes the same two-pass pair on both arms; and the one
SDPA cell that *does* change (`sdpa_prefill_512`) is the prefill fallback, which
is not an SDPA kernel at all — it is the composed `steel_gemm_fused` +
`block_softmax` path taken *because* `sdpa_full_supported_head_dim` rejects head
dimension 256 (rung 0, S13, unreachable on **every** host including the M5).

Consequence: the crown holder's `kL = 1025` warm mechanism, adopted into the base
at `4898738e`, is **fully local-comparable**. So is every SDPA measurement since
E65. The transfer-risk caveat attached to those results can be dropped.

### 6.2 This audit closes a target, it does not open one

Stated plainly, as instructed: **result (c) prices E65 follow-up (a) — the
head-prime row-count sweep — DOWN, and the correct action is to not run it.**

The arithmetic lands exactly where the advisor predicted. E65 priced that
follow-up with a ceiling of about **+0.17 % of a leg**, computed from local time
with no transfer rate applied. Rung 2 reproduces that figure as the *naive* row:
**0.1684 %**. Applying the measured `tau = 7.5798` gives **0.0641 %**, so the
published ceiling was **2.63× too high**.

The sweep was also attractive while S4's price could still have been 0.196 % to
0.231 %, which is what a decode-round `R(M)` divisor would give.
Rung 1 excludes that branch, because M=511 provably takes `affine_qmm_t_nax` and
`steel_gemm_fused_nax` under the ranked architecture. The real ceiling is
0.0641 %, which is 0.085 sd of one published score. A sweep cannot recover more
than the whole term, the whole term is under a tenth of a standard deviation,
and the term is not steerable by editable code in any case.

**E65 follow-up (a) is closed by this audit.** The audit removed a candidate
from the queue; it did not add one.

(The 29.215 ms local saving used above is the mean of E65's two prime
measurements, 29.52 ms and 28.91 ms. Using the advisor's single 29.52 ms figure
moves the result to 0.0648 %, which changes nothing.)

### 6.3 The prefill dense bf16 GEMM fallback is real

Recorded, not chased, as instructed. `dense_matmul_m511` and `sdpa_prefill_512`
both fall back to composed `steel_gemm_fused` work that the M5 runs in a wider
nax tiling (`bm64_bn128_bk256_wm2_wn4` against local `bm64_bn64_bk16_wm2_wn2`).
The effect is real and measurable in rung 1. Its priced consequence is S9's
0.0362 %. No follow-up is proposed.

### 6.4 Corrections to standing campaign records, and one retraction of my own

0. 🔴 **RETRACTED, mine.** My claim that the 30.402 ms constant has no measured
   source and is refuted was wrong. The source exists at
   `prompt_round_reconstruction.py:111-156`, 30.402 ms is the **candidate**
   build's depth-0 round, and I compared it against the **pinned serial** leg
   population. `R(depth-0) = 2.1383` stands. `tau_depth0_round = 1.7590` and
   every consequence I drew from it are withdrawn (section 4.1, 4.2).
1. **`R` is width dependent.** A single `R` is the wrong shape. `R(M)` runs
   2.11–2.17 for M ≤ 3.66 and 2.36–2.47 for M ≥ 5.53 (section 4.3). This
   supersedes the flat constant without refuting its depth-0 value.
2. **S2 is not an architecture divergence.** `device.cpp:595-596` reapplies
   `env::max_ops_per_buffer` and `max_mb_per_buffer` after the tier switch, and
   `RuntimeStartupMemoryPolicy.installQwenMTPFullProfileCommandBufferDefaults`
   force-sets 512 MiB / 50 ops at ≥ 96 GiB against 128 MiB / 64 ops below. Local
   at 48 GiB gets 128/64 and rank gets 512/50 for a **memory-profile** reason.
   **Ledger item 115 consequence 2 is wrong; item 94 / E31 is right.** The
   pre-existing failure of `startupMemoryPolicyKeepsRanked128GiBProfile()`
   corroborates the 512/50 pair independently (section 7).
3. **E65's prefill roofline understates prefill attention by 2×**, 0.052 against
   0.1031 TFLOP. The composed fallback runs both matmuls at full qL × kL and
   then masks, so the masked half is computed and discarded.
4. **There are 15 architecture-conditional sites, not 14.** `device.cpp:572` is
   a second, separate read of the architecture string.
5. **188(A) still has a definitional defect.** It defines `R` as a **leg** ratio
   ("Let L be leg time") and then computes it from two **round** times. This
   defect is independent of which round constant is used, and it survives my
   retraction (section 4.2).

## 7. Scope, budget and boundary checks

All run at HEAD against BASE_SHA `bdfbc4e9`.

```
python3 research/twin_audit.py
  TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver

senpai/check-editable-budget.sh bdfbc4e92c93d216503980fb46258ff0b314145a
  editable budget OK: source=2463704/3000000 headroom=536296
  growth=0/262144 exempt=2410 files=154

senpai/verify-ranked-score-boundary.sh
  PASS: ranked numerator is pinned baseline;
  candidate edits affect the MTP denominator only
```

**`growth = 0/262144`.** No candidate byte changed.

`senpai/validate-assignment-scope.sh` requires at least one submitted path
argument and only checks membership in `benchmark.json` `editablePaths`. The
meaningful E70 statement is therefore the stronger one: `git diff BASE..HEAD`
touches **zero** submitted paths. The complete non-`research/` change is
`Tests/MLXFastTests/E70ArchDispatchProbeTests.swift`, +503 lines, and `Tests/`
is not in `editablePaths`.

### `swift test` two-arm control

The assignment states nine tests fail at base and asks for the base arm as a
control. `research/e70_test_control.sh` runs `swift test --force-resolved-versions`
at HEAD, then builds a throwaway detached worktree at BASE_SHA and runs the same
command there, so the two failing sets are compared instead of asserted.

Job `a0c5fa37-8104-4846-94d7-baaeee6c83e4`, exit 0, 119 s. The base arm compiled
1009 files from scratch in its own worktree; the head arm was incremental.

| arm | commit | tests | suites | issues | failing tests |
|-----|--------|-------|--------|--------|---------------|
| HEAD | `ff63cf4` | 689 | 50 | 40 | **9** |
| base | `bdfbc4e9` | 688 | 49 | 40 | **9** |

**The nine failing tests are the same nine on both arms**, and the failing set
diffs empty:

```
contestantDocsCommandBlocksKeepTheDependencyGraphFrozen()
participantDocsExposeDefaultCLIInstallDirectory()
qwen36ConfigContractDigestMatchesTheReferenceManifest()
startupMemoryPolicyKeepsRanked128GiBProfile()
submissionStaticReviewPromptCoversMeasurementStructureExploitation()
theCheckedInDeclarationSelectsThePinnedHead()
theEvenMedianRuleIsTheMeanOfTheTwoCentralValues()
theQwenMTPTrackIsArmedOnQwen38()
theSeededCalibrationExpectationMatchesItsRecordedProvenance()
```

The single extra test and suite at HEAD is `E70ArchDispatchProbeTests`. It
skips by default with `"set MLXFAST_E70_ARCH_PROBE=1 to run the architecture
probe"`, and its suite passes. **This branch introduces no new failure.**

One of the nine corroborates correction 2 of section 6.4 by accident.
`startupMemoryPolicyKeepsRanked128GiBProfile()` fails on both arms with:

```
RuntimeStartupMemoryPolicyTests.swift:83: (policy.maxMegabytesPerCommandBuffer → 512) == 320
RuntimeStartupMemoryPolicyTests.swift:84: (policy.maxOperationsPerCommandBuffer → 50) == 128
```

The live policy returns **512 MiB / 50 ops** at the ranked memory profile. The
test still expects an older 320 MiB / 128 ops. That is the same 512/50 pair the
rung-0 S2 row reads from
`RuntimeStartupMemoryPolicy.installQwenMTPFullProfileCommandBufferDefaults`, so
the failing assertion is independent evidence that S2 is governed by the memory
profile and not by the GPU architecture. The stale expectation is out of scope
for E70 and is left unchanged.

## 8. Reproduction

```bash
# rung 0, source only, no GPU
python3 research/e70_dispatch_divergence_audit.py --json research/e70-rung0.json

# rung 1, both arms, one cell per process   (via run_job, ~343 s)
research/e70_run_arch_probe.sh
python3 research/e70_rung1_diff.py --root research/out/e70-rung1 \
  --json research/e70-rung1-diff.json

# rung 2 and the transfer constant
python3 research/e70_transfer_constant_provenance.py \
  --json research/e70-transfer-constant.json
python3 research/e70_rung2_consequences.py --json research/e70-rung2.json

# W&B
python3 research/e70_wandb_log.py --rung0 research/e70-rung0.json \
  --rung1 research/out/e70-rung1 --rung2 research/e70-rung2.json

# two-arm swift test control   (via run_job)
research/e70_test_control.sh bdfbc4e92c93d216503980fb46258ff0b314145a
```

The W&B backend merges summary metrics instead of replacing them, so the keys
written by the retracted revision cannot be deleted. `e70_wandb_log.py`
overwrites each one with its own retraction text instead, which leaves a
readable record rather than a silent gap. The new `transfer/R_of_M` table
carries the width ladder of §4.3.

## 9. Follow-up experiments, named precisely and NOT started

### 9.1 E71 — the unaligned nax variant

Per the stop rule, this is the deliverable. It is not started.

> **E71 — Does the unaligned nax variant `alN_false` cost `gdn.in_proj` more at
> rank than the aligned variant costs the other three wide-output linears?**
>
> **Site.** `quantized.cpp:697`, the qmm nax gate, at K=5120 N=16480, the fused
> `gdn.in_proj`. Rung 1 proves this is the **only** scored family that takes
> `affine_qmm_t_nax_..._alN_false` on the ranked M5. The other three wide-output
> families (N=14336, 34816, 248320) all take `alN_true`. 16480 = 64 × 257.5, so
> N is not a multiple of the 64-wide nax tile.
>
> **Why it is the right next question.** It is the one dispatch difference this
> audit found that is (i) on the scored path, (ii) invisible on every local
> host, and (iii) **steerable by editable code** — unlike S4 and S9, which are
> both priced and both unsteerable. `gdn.in_proj` runs in all 48 Gated DeltaNet
> layers, three quarters of the model, and
> `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` is in
> `benchmark.json` `editablePaths`.
>
> **Mechanism to test.** The fused projection is built lazily at
> `Qwen35.swift:703-707`, where `_inW`, `_inS` and `_inZ` are the row-wise
> concatenation of the `qkv`, `z`, `b` and `a` quantized weights along axis 0.
> Append 32 zero rows to each of those three tensors, which pads the output
> width from 16480 to 16512 (= 64 × 258), and change the tail slice at
> `Qwen35.swift:688` from `y[.ellipsis, bEnd...]` to `y[.ellipsis, bEnd..<16480]`.
> The pad is paid once at build time; the per-call cost is 32 wasted output
> columns, 32 / 16480 = 0.19 % extra work. Group size 64 runs along K, so
> appending along the N axis needs no requantization.
>
> This moves the family from `affine_qmm_t_nax ... alN_false` to
> `... alN_true` at rank, and changes nothing at all on any local host, which is
> exactly why it must not be judged by a local timing run.
>
> **Why it cannot be decided locally, and what to do about it.** This host never
> executes either nax variant, so no local timing can answer it. The decisive
> measurement is an official ranked submission pair, or a rung-1-style probe
> extended to *time* rather than merely *name* the kernel — which requires M5
> access. Do not start it on a local timing argument.
>
> **Expected size.** Unknown, and that is the point: it is the only site in this
> audit whose ranked cost is not bounded by the 0.1003 % ceiling of section 5.4,
> because that ceiling prices the cost of *lacking* nax, not the cost of taking
> the *wrong* nax variant.
>
> **Stop rule.** If a ranked pair shows less than 0.756 % (one published-score
> sd) of separation, close it. The padding is cheap but not free, and an
> unresolved 0.19 % overhead is not worth carrying.

### 9.2 The M=4 → M=5 width-transfer question — suggested, NOT opened here

Section 4.3 found a step in `R(M)`. The advisor asked me to record it and not
to open it in this PR, so it is recorded and not opened.

> **Suggested experiment — does the local M4 Pro carry an M=4 → M=5 verify-width
> penalty that the ranked M5 does not?**
>
> **Evidence.** `R(M)` sits on two flat shelves, 2.11–2.17 for M ≤ 3.66 and
> 2.36–2.47 for M ≥ 5.53. The 8.55 % edge-to-edge step is about twice the
> within-group scatter. It coincides exactly with the local ladder's own jump
> `E1(4) = 91.29 ms → E1(5) = 115.69 ms`, **+26.7 %**. That local jump is much
> larger than any adjacent marginal step: `E1_MARGINAL[4] = 24.40 ms` against
> 15.77 at depth 3 and 18.98 at depth 5.
>
> **Why it matters.** M=5 and M=6 together carry **57.5 %** of the modelled
> ranked verify-width time. If the local penalty at M=5 is a local artefact,
> every local measurement taken at M ≥ 5 is over-priced when converted to rank,
> and any experiment whose gain comes from *avoiding* width 5 is over-valued.
>
> **What would settle it.** A ranked-side round census, or a rung-1-style probe
> extended to time the M=4 and M=5 verify shapes on the M5. Neither is available
> from public receipts.
>
> **Confound to control first.** Ranked rounds are means over a width mixture
> while local `E1` is a fixed width, which biases `R(M)` downward for
> high-spread prompts. That bias works *against* the step, so it does not
> explain it, but a proper test must model the mixture rather than assume a
> point width.

## 10. Honest limits of this result

- Rung 1 names kernels. It does not time them. Nothing here measures M5 speed.
- Rung 2's median-pair model is validated against exactly one ranked receipt
  (`ca9251b8`, reproduced to 7.11e-6). A different candidate schedule would
  shift the median pair and therefore the sensitivity constant.
- The 0.1003 % upper bound assumes both sites cost *zero* at rank, which is
  physically impossible. The true recoverable amount is strictly smaller.
- `R(M)` is derived, not directly measured. Its ranked side comes from the
  round-count reconstruction of one receipt, and its local side comes from the
  E1 ladder interpolated at fractional depth. Both inputs are measured; the
  quotient is a construction.
- The `R(M)` shelf step is confounded by the ranked width mixture, which is
  unidentifiable from public data (184(D):10219). The confound pushes the step
  down, so the step is a lower bound on whatever effect is there, but the size
  is not settled.
- `tau = 7.5798` is imported from ledger 186(C). This audit justifies applying
  it to S4 but does not re-derive it.
- My own refutation of the 30.402 ms constant was wrong and is retracted in
  §4.1. The corrected reading changes no headline number in §5: S4 is still
  0.0641 %, S9 is still 0.0362 %, and the upper bound is still 0.1003 %. The
  error was in the transfer-constant analysis, not in the priced result.
