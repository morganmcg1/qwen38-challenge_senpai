# E176 result: Q has no decode consumer, and receipt C is the outlier

`assignment_id` `e176-q-decode-consumer-census`, revision `r1`.
Base `senpai/qwen38-mtp-r1` at `b13ad3b875546e784a9ebdc7efa25f20dac6712a`
(r0 was written against `6368dc25265bd05461df4f070a18f4d97a18bc12`).
The r1 branch merges base `b13ad3b8` before submission. That merge changes no
file under `Sources/` or `Vendor/`, so every source citation and line number in
section 1 stays valid on the merged tree.

W&B r1: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/zm2f4q6q>
(run `zm2f4q6q`, corrected scored-tree census).
W&B r0: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/vxdn8h60>
(run `vxdn8h60`, superseded section 1).

Revision r1 rewrites section 1 against the scored vendored tree, because the r0
census read a parallel non-scored implementation. The verdict is unchanged and
now rests on two independent barriers. Sections 2 to 5 and 7 are byte-identical
board measurements. Section 6 is rewritten, because the corrected census kills
the test it proposed. Section 8 gains one note on r1 reproduction scope, and
follow-up 1 of section 9 is replaced for the same reason. FINDINGs 450, 452 and
453 come from this experiment.

Receipt state when this amendment closed: `15017ddf` (E175, A+Q) was still
`validating` on a board fetch at 2026-08-24T11:15Z with 1336 scored rows, so
the section 5 prediction stays registered pre-receipt.

Every number carries a harness label. No GPU decode leg was timed. Task 4, the
local GPU probe, was not needed: the desk answer at the routing threshold is
not close.

## 1. Cell census (`harness=source`, revision r1)

Reproduce with `python3 research/e176_cell_census.py`.

### 1.0 What r1 corrects, and what it does not

The r0 census read the wrong tree. It enumerated
`Sources/MLXFastModel/Qwen35Ops.linear` and its callers. That is a **parallel
implementation**: `Qwen35FastEngine` is referenced only from inside
`Sources/MLXFastModel/` (`Qwen35Cache.swift:66`,
`Qwen35RuntimeWeights.swift:14,61,197`). The scored session
`Qwen36MTPBlockSession.swift` imports `MLXLLM` (line 4) and drives
`Qwen35TextModelInner.callAsFunction` (line 775), so the scored path is the
**vendored** tree
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` plus
`Qwen35MTP.swift`.

Three r0 claims are **withdrawn**:

1. "No file named `Qwen35.swift` exists in this tree." False. It exists at
   `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`. The r0 grep was
   scoped to `Sources/` and the miss was reported as an advisor error. That
   report was wrong.
2. "There is no custom quantized matvec replica in this tree." False.
   `Qwen35CustomQMV.matmul` (`Qwen35.swift:1841`) fronts routed calls through
   `qwen35RoutedQuantizedMM` (`:1895`) and `qwen35RoutedLinear` (`:1918`), and
   dispatches its own Metal kernel `qwen35CustomAffine4QMVKernel`
   (`:1598`, called at `:1881`).
3. The 497-call, 13-cell table as a **scored-path** census. It describes the
   `Sources/MLXFastModel` parallel tree only. Section 1.7 retains it under that
   label.

The verdict does not change. The corrected census reaches the same answer
through **two** independent barriers instead of one, so it is stronger than the
submitted version. The primary metric is restated as **0 of 257**.

Sections 2 to 5, 7 and 8 are board measurements and source-independent. They are
unchanged. Section 6 is rewritten because the corrected census kills its
proposed first test.

### 1.1 The scored entry point and the two barriers

Every quantized projection in a decode round meets one of two gates before any
kernel runs.

**Barrier 1, the candidate-owned replica.** `Qwen35CustomQMV.routable`
(`Qwen35.swift:1767`) accepts a cell when it is affine 4-bit group-64 with
bf16 activations, `k % 512 == 0`, `n % 8 == 0`, `n >= 4096` (`:1780`), width
`m` inside `2...9` (`:1704`), and the input, weight, scales and biases are row
contiguous. An accepted cell runs the replica kernel and **never reaches MLX**.
`routable()` contains **no `arch_gen` term**: dtype, shape, width and
contiguity only. Barrier 1 therefore holds on every GPU generation.

**Barrier 2, the MLX dispatcher.** A cell that reaches MLX enters
`QuantizedMatmul::eval_gpu`
(`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1415`):

```cpp
int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
if (M >= vector_limit) { /* qmm_splitk or qmm */ return; }
dispatch_qmv(...);
```

Q (RULE 377, commits `2679ef6c` and `232b9fda`) edits `qmm_t`, `qmm_t_nax` and
`qmm_t_splitk` only, and entry 258 already proved `qmm_t_splitk` dead on this
model. A cell consumes Q only if it passes both barriers: it must be declined
by the replica **and** reach `M >= vector_limit` with `transpose_ == true`.

### 1.2 The scored target cells: 257 calls, seven fused shapes

Shapes follow from `fixtures/qwen3_6_27b_config.json` `text_config`: hidden
5120, intermediate 17408, vocab 248320, 64 layers = 48 linear-attention + 16
full-attention, head_dim 256, 24 query heads, 4 KV heads,
`attn_output_gate: true`, GDN 16 key heads and 48 value heads at dim 128.

| cell | K | N | calls/round | front | routable at M 2..9 | site |
| --- | --- | --- | --- | --- | --- | --- |
| mlp.gate_up (fused) | 5120 | 34816 | 64 | replica | yes | `Qwen35.swift:1948` |
| mlp.down | 17408 | 5120 | 64 | replica | yes | `Qwen35.swift:1985` |
| gdn.in_proj (fused) | 5120 | 16480 | 48 | replica | yes | `Qwen35.swift:809` |
| gdn.out_proj | 6144 | 5120 | 48 | replica | yes | `Qwen35.swift:1322` |
| fa.qkv (fused) | 5120 | 14336 | 16 | replica | yes | `Qwen35.swift:3098` |
| fa.o_proj | 6144 | 5120 | 16 | replica | yes | `Qwen35.swift:3398` |
| lm_head | 5120 | 248320 | 1 | replica | yes | `Qwen35.swift:5602` |
| | | | **257** | | | |

The count matches the tree's own statement: `Qwen35.swift:1737` records "all
257 wide QMV calls of one decode round" and lists the same seven shapes in its
E120 measurement grid.

Fusion arithmetic, so the shapes can be checked against the config:
GDN `in_proj` = 2·2048 (q, k) + 6144 (v) + 6144 (z) + 48 (b) + 48 (a) = 16480;
FA `qkv` = 12288 (q, doubled by `attn_output_gate`) + 1024 (k) + 1024 (v) =
14336; MLP `gate_up` = 2 × 17408 = 34816.

**Structural consequence: fusion removes the small-N cells.** All seven scored
shapes have N ≥ 4096. The kv projection at N = 1024 and the GDN `b` and `a`
projections at N = 48, nominated in F2 as Q's decode consumers, are not
dispatched at all in this tree: they exist only as row ranges inside
`fa.qkv` and `gdn.in_proj`.

**Variant.** When the exact-KV island fast path is active (`Qwen35.swift:3078`),
`fa.qkv` splits into a quantized q+gate cell at N = 12288, still routable, and
a **BF16 dense** matmul for k and v that is not a quantized dispatch at all.
Neither form creates a sub-4096 quantized cell.

### 1.3 The MTP head cells

Drafting is autoregressive, so the head runs at `M = 1` per draft step; the
session also warms a two-row accept fold, and the seed prime runs the head at
`M ≈ 512`, outside the drafting rounds.

| head cell | K | N | bits | front | kernel at M ≤ 9, gen16 / gen ≥ 17 |
| --- | --- | --- | --- | --- | --- |
| mtp.fc | 10240 | 5120 | 4 | MLX, plain `QuantizedLinear` (`Qwen35MTP.swift:194`) | `dispatch_qmv` |
| mtp.layer fa.qkv | 5120 | 14336 | 4 | replica | replica at M 2..9, `dispatch_qmv` at M = 1 |
| mtp.layer fa.o_proj | 6144 | 5120 | 4 | replica | replica at M 2..9, `dispatch_qmv` at M = 1 |
| mtp.layer mlp.gate_up | 5120 | 34816 | 4 | replica | replica at M 2..9, `dispatch_qmv` at M = 1 |
| mtp.layer mlp.down | 17408 | 5120 | 4 | replica | replica at M 2..9, `dispatch_qmv` at M = 1 |
| draft_lm_head | 5120 | 248320 | 2 | MLX, bits 2 so the replica declines | `dispatch_qmv` |
| draft centroid | 5120 | ~1024 | 2 | MLX, bits 2 so the replica declines | `dispatch_qmv` |

`mtp.fc` is the clearest single demonstration of barrier 2 acting alone: it is
called as `fc(...)`, with no replica in front, and it still never reaches
`qmm_t` at any decode width because its limit is 10.

### 1.4 Barrier 2 measured on this host

`research/e176_arch_probe.swift` reads the Metal device architecture string
first hand and **replays** the `get_qmv_batch_limit` branch table
(`quantized.cpp:84-125`) verbatim for the exact scored shapes. It is a device
metadata query plus arithmetic; it does not call the MLX function itself. An
earlier PR comment described it as calling the real function. That description
was too strong, and this is the accurate one. `harness=local`, exit 0, no
compute dispatched.

```
architecture: applegpu_g16s
arch_gen: 16  arch_size: s
get_qmv_batch_limit(K=5120,  N=34816)  = 10   [mlp.gate_up (fused)]
get_qmv_batch_limit(K=17408, N=5120)   = 10   [mlp.down]
get_qmv_batch_limit(K=5120,  N=16480)  = 10   [gdn.in_proj (fused)]
get_qmv_batch_limit(K=6144,  N=5120)   = 10   [gdn.out_proj, fa.o_proj]
get_qmv_batch_limit(K=5120,  N=14336)  = 10   [fa.qkv (fused)]
get_qmv_batch_limit(K=5120,  N=248320) = 10   [lm_head]
get_qmv_batch_limit(K=10240, N=5120)   = 10   [mtp.fc]
get_qmv_batch_limit(K=5120,  N=12288)  = 10   [fa.q_gate, island path]
get_qmv_batch_limit(K=5120,  N=1024)   = 10   [unfused kv shape]
get_qmv_batch_limit(K=5120,  N=48)     = 10   [unfused gdn b/a shape]
get_qmv_batch_limit(K=2048,  N=2048)   = 18   [control]
get_qmv_batch_limit(K=4096,  N=4096)   = 12   [control]
```

Three readings:

1. Every scored shape returns **10** on this host, `mtp.fc` and the unfused
   N = 1024 and N = 48 shapes included. The guard is a conjunction on `D` **and**
   `O`, and every scored cell has `D = K ≥ 5120 > 4096`, so no scored shape can
   take a narrow branch.
2. The two controls return 18 and 12, so the branch structure is live and 10 is
   the floor on this host, not a constant returned by accident.
3. `arch_gen` is **16** first hand, so this host takes the `18/12/10` table and
   never the `14/10/6` table, which `quantized.cpp:86` guards with
   `arch_gen == 13 || arch_gen == 14`.

The ranked M5 step stays source-derived: `is_nax_available`
(`device.cpp:913-925`) requires `arch_gen >= 17`, so any host that runs
`qmm_t_nax` takes the same `else` branch and the same floor of 10. I cannot
probe the ranked host.

### 1.5 Required statement

Maximum decode width is `M = 1 committed primary + 8 drafts = 9`.

**The set of Q-consuming cells at `M ≤ 5` is EMPTY. It is equally empty at
`M = 6, 7, 8, 9`: 0 of 257 target calls, and 0 of the head cells.** Q's only
live consumer inside a timed leg is the 512-token seed prefill, where the target
and the head both run far above every limit.

Both barriers give this answer independently:

- Barrier 1 removes all 257 target calls at `M = 2...9` on **every** GPU
  generation, because `routable()` has no `arch_gen` term and all seven fused
  cells satisfy its shape gates.
- Barrier 2 removes `M = 1`, `mtp.fc`, the 2-bit draft projections and any
  future fallthrough, because the minimum limit outside `arch_gen` 13/14 is 10
  and `9 < 10`.

The tree says the same thing independently: `Qwen36MTPBlockSession.swift:1049`
and `:1658` both record that projections at `M` in 6..9 stay on a per-row-exact
QMV dispatch.

### 1.6 The M = 9 row, for the reopened declamp question

At `M = 9`, on gen 16 and on gen ≥ 17, the count of scored cells that reach
`qmm_t` or `qmm_t_nax` is **0 of 257**. A declamp to `M = 9` cannot create a Q
consumer and cannot be priced through Q.

The gen 13/14 branch does not change this, and the r0 claim that it does is
**withdrawn**. Barrier 1 has no generation term, so a gen 13/14 host also sends
zero scored target calls to `qmm_t` at `M = 2...9`. The only cells that would
cross a limit of 6 there are the non-replica head cells (`mtp.fc` and the 2-bit
draft projections), and drafting runs them at `M = 1`. No host class in this
campaign has a Q-consuming decode cell.

**Corollary.** Any future `qmm_t` win needs decode `M ≥ 10`, which the
eight-draft cap forbids, or a cell that the replica declines and that also
crosses the limit. The scored tree currently has neither.

**One line towards Entry 379.** The replica width plan (`Qwen35.swift:1715-1728`)
launches the same 2 × 4 active input groups at `M = 7` and `M = 8`, so the
kernel-level target cost should be near flat between them, while the ranked
`R(M)` law prices +7.7 ms. Under this census the difference cannot live in
`qmm_t`; it must live in non-replica round cost — verification, GDN replay,
attention cache or head work. That is E177's open question, not this
experiment's.

### 1.7 Withdrawn r0 table: `Sources/MLXFastModel` parallel tree, NOT the scored path

Retained so the record shows exactly what was measured and where it applies.
These cells exist in `Sources/MLXFastModel` (`Qwen35Ops.linear`,
`Qwen35GatedDelta`, `Qwen35Attention`, `Qwen35MLP`, `Qwen35FastEngine`). The
scored session never calls them.

| cell | K | N | calls/round | source (non-scored tree) |
| --- | --- | --- | --- | --- |
| gdn.in_qkv | 5120 | 10240 | 48 | `Qwen35GatedDelta.swift:241` |
| gdn.in_z | 5120 | 6144 | 48 | `Qwen35GatedDelta.swift:245` |
| gdn.in_b | 5120 | 48 | 48 | `Qwen35GatedDelta.swift:254` |
| gdn.in_a | 5120 | 48 | 48 | `Qwen35GatedDelta.swift:255` |
| gdn.out | 6144 | 5120 | 48 | `Qwen35GatedDelta.swift:346` |
| attn.q | 5120 | 6144 | 16 | `Qwen35Attention.swift:143` |
| attn.k | 5120 | 1024 | 16 | `Qwen35Attention.swift:162` |
| attn.v | 5120 | 1024 | 16 | `Qwen35Attention.swift:163` |
| attn.o | 6144 | 5120 | 16 | `Qwen35Attention.swift:211` |
| mlp.gate | 5120 | 17408 | 64 | `Qwen35MLP.swift:28` |
| mlp.up | 5120 | 17408 | 64 | `Qwen35MLP.swift:29` |
| mlp.down | 17408 | 5120 | 64 | `Qwen35MLP.swift:30` |
| lm_head | 5120 | 248320 | 1 | `Qwen35FastEngine.swift:263` |
| | | | **497** | |

Its `attn.q` row is also wrong for the scored architecture: the output gate
doubles that projection to 12288.

### 1.8 One kernel outside the `qmm_t` family shares an edited symbol

Q adds `QuantizedBlockLoader::shift_dst` (`quantized.h:572`). That struct is
also used by `qmm_n_impl` (`:1686`) and `affine_gather_qmm_rhs` (`:2834`). The
addition is a new method with no new call site in either kernel, and the Q
commit message states the same scope, so their generated code is unchanged.
`qmv` kernels do not use the loader at all. The only channel Q opens outside
`qmm_t` is the JIT source string of the quantized family, which changes the
compiled-library cache key rather than any decode arithmetic.

## 2. Per-prompt null calibration (`harness=ranked`)

Board snapshot: 976 scored 512-token receipts, 192 distinct schedule
fingerprints, 115 in the organizer-main-schedule cluster.

Scalar widths:

- Per-receipt common offset, robust sd **0.1421 %**; pairwise 0.2011 %.
- Per-prompt serial residual sd 0.155 % to 0.171 %.
- **Adopted per-prompt pairwise 1 sigma 0.298 % to 0.314 %**, from
  `sqrt(2 * serial_resid_p^2 + common_null^2)`.
- Contaminated ceiling from the candidate cluster 0.68 % to 0.97 %.

### 2.1 Sign structure of the byte-identical pair, crown `ec24d591` vs A `5a9f130a`

| prompt | serial % | prefill % | decode % | leg % |
| --- | --- | --- | --- | --- |
| plutarch | −0.0935 | −0.2456 | −0.0049 | −0.0131 |
| drama | −0.2427 | −0.1009 | −0.5710 | −0.5440 |
| travel | −0.2137 | −0.3742 | −0.4705 | −0.4641 |
| beagle | −0.1513 | −0.2499 | −0.2473 | −0.2475 |
| republic | −0.4306 | −0.1471 | −0.1742 | −0.1713 |
| essays | **+1.1256** | −0.3802 | −0.1459 | −0.1705 |
| medicine | −0.0075 | −0.0218 | −0.0927 | −0.0852 |
| botany | +0.0541 | −0.2373 | −0.1748 | −0.1815 |
| **mean8** | **+0.0051** | **−0.2196** | **−0.2351** | **−0.2346** |
| **sd8** | 0.4768 | 0.1254 | 0.1916 | 0.1816 |
| **negative** | 6/8 | **8/8** | **8/8** | **8/8** |

This is the decisive property, and a scalar sigma hides it. **The null is not
sign-symmetric.** Two byte-identical candidate builds differ by a coherent
whole-receipt offset of about −0.23 % that appears on all eight prompts in the
candidate channel, with a per-prompt spread of only 0.18 % around it.

Two consequences.

1. A whole-receipt shift of about +-0.25 % in the candidate channel is
   **ordinary**, and averaging over prompts does not reduce it. Any
   cross-receipt claim smaller than that is unsafe.
2. The serial channel does not carry the same offset: mean8 +0.0051 %, 6/8
   negative, with one large excursion at essays of +1.1256 %. The serial leg is
   byte-identical by construction, so that essays value is pure measurement
   noise, and it is why essays is a weak lever throughout this analysis.

## 3. Mechanism fit (`harness=ranked`)

One-parameter fits to the `B - C` vector, 7 dof: prefill (34.7 +- 6.6 ms/leg,
chi2 14.38), round (249.9 +- 56.3 us/round, chi2 22.41), uniform
(−0.5754 +- 0.1093 %, chi2 14.41). The round model is refuted: it needs
plutarch at −0.82 % and plutarch measures +0.05 %, a +3.1 sigma miss.

Two-parameter fit: per leg **+35.5 +- 12.5 ms** (2.8 sigma), per round
**−7.9 +- 107 us** (0.1 sigma, consistent with zero, 95 % upper bound
218 us/round).

**Essays as a falsification lever: it does not work, in either direction.**
Essays sits at −0.39 sigma against zero, and the three one-parameter models
predict −0.687, −0.417 and −0.575 %, a spread of 0.29 pp that is inside the
per-prompt sigma. Section 2.1 explains why: essays carries the largest serial
excursion in the byte-identical control. Plutarch is the only prompt that
discriminates, because its edl of 0.156 makes round count and prefill share
diverge.

### 3.1 The channel split, with no model

Each receipt carries per-prompt `prefill_seconds_per_token`, so
`leg = 512 * mtp_seconds_per_token_mean`,
`prefill = 512 * prefill_seconds_per_token`, `decode = leg - prefill`.

Prefill share of the candidate leg: plutarch 3.42 %, drama 5.71 %, travel
6.47 %, beagle 9.51 %, republic 10.49 %, essays 10.42 %, medicine 10.47 %,
botany 10.60 %; mean7 9.10 %.

`B - A` is the clean Q instrument, because A is organizer-pure and both trees
were inspected first hand:

| prompt | prefill % | decode % | leg % |
| --- | --- | --- | --- |
| plutarch | +1.9225 | −0.0579 | +0.0098 |
| drama | +2.1709 | −0.1685 | −0.0340 |
| travel | +1.7964 | +0.1604 | +0.2684 |
| beagle | +1.7070 | −0.0839 | +0.0888 |
| republic | +2.0347 | +0.1444 | +0.3450 |
| essays | +1.9133 | +0.0178 | +0.2167 |
| medicine | +2.0869 | +0.0709 | +0.2846 |
| botany | +1.8552 | +0.0391 | +0.2331 |
| **mean8** | **+1.9359** | **+0.0153** | **+0.1766** |
| **sd8** | 0.1544 | 0.1136 | 0.1379 |
| **negative** | **0/8** | 3/8 | 1/8 |

Q's prefill effect is 8/8 the same sign. Q's decode effect is 3/8 negative with
a mean of +0.0153 % and a spread of 0.11 %, which is the sign-symmetric null of
section 2.1 with no offset at all. That is what "no consumer" looks like in a
measurement.

Prefill-only closure test, mean7, where a seed-only factor must satisfy
`leg % = prefill % * prefill share`:

| pair | content | prediction | measured leg | residual |
| --- | --- | --- | --- | --- |
| B − A | Q + I | +0.1782 | +0.2004 | **+0.0222** |
| C − A | I | −0.0231 | +0.8765 | +0.8996 |
| B − C | Q as previously derived | +0.1996 | −0.6691 | −0.8687 |
| D − B | E165 − I | +0.0061 | +0.6878 | +0.6817 |
| crown − A | byte-identical null | −0.0199 | −0.2663 | −0.2464 |

Only `B - A` closes, and it closes at the null scale.

**Q is prefill-only: +1.94 % of prefill, which is +10 to +12 ms per leg, and a
decode effect indistinguishable from zero. Net candidate leg +0.20 % slower.**
This reproduces Edward's E175 local +1.936 %, 8/8 same sign (`harness=local`,
W&B `8bc65oel`) to three digits, so the prefill penalty transfers from local to
ranked.

## 4. The quantitative implication: receipt C is the outlier

FINDING 450 makes the Q decode term identically zero. Then `B - C` must equal
its prefill-only prediction, and it does not:

| quantity | value |
| --- | --- |
| prefill share, mean7 | 9.0960 % |
| `B - C` prefill, mean7 | +2.1947 % |
| predicted `B - C` leg | **+0.1996 %** |
| measured `B - C` leg, mean7 | **−0.6691 %** |
| **unexplained, located in C** | **−0.8687 %** |

Q-prefill pushes B the wrong way for FINDING 440: it makes B *slower* than C by
about +0.20 %, not faster by 0.67 %. So `B - C` needs a non-code component of
about **−0.87 %** sitting in receipt C.

The independent check agrees. `C - A` is an inert-instrumentation-only
contrast, yet its decode channel is **+0.9889 % mean7** while its prefill
channel is −0.2315 %. Against the whole-receipt common-offset sigma of
0.1421 % that is **7.0 sigma**, and it is about four times the −0.23 % coherent
offset that section 2.1 measures between two byte-identical builds. `C - A`
decode is 7/8 positive with a spread of 0.56 %, so it is not one bad prompt.

**Conclusion: receipt C is an outlier.** Its candidate decode channel is slow by
about +0.9 % for a reason that is not in its diff. FINDING 440's attribution of
−0.6691 % to Q is therefore conditional on receipt C, and the E175 receipt is
the test.

The `+0.0070 %` additivity residual that validated FINDING 440 cannot detect
this. `(C - A) + (B - C) + (D - B) = D - A` is an identity in receipt values and
holds for any C, however anomalous.

This converges with Edward's independent C/A anomaly of −1.08 % published on an
inert-only contrast. It would also re-open D's E165 verdict, but only after the
E175 receipt confirms the model. One step at a time.

## 5. Registered prediction for E175 (`15017ddf`, frozen `9c4fefe8`)

Applying the measured per-prompt `B - A` channel factors to A's own per-prompt
legs and recomputing the published median gives **3.702087**, against A's
`3.70784519415395`, a change of **−0.155 %**.

| model | predicted published |
| --- | --- |
| advisor, ranked route | 3.7328 / 3.7267 |
| Edward, prefill-adjusted | 3.7218 / 3.7157 |
| **this census, model 3** | **3.7021**, band 3.700 to 3.704 |

The three models are separated by far more than the +-0.25 % whole-receipt null
of section 2.1, so the receipt discriminates cleanly. If E175 lands near 3.702,
Q is a net cost, it should leave the ship set, and receipt C is confirmed as the
outlier. If it lands near 3.727, this census is wrong about the live call path
and section 1 must be re-derived.

## 6. Open conflict: FINDING 444

FINDING 444 derives a local `Q` leg effect near **−0.88 %**. At the 23.4 %
local prefill share, and with the measured local prefill penalty of +1.936 %,
that requires about **−1.33 % of local decode**. Section 1 forbids that channel
on **every** host, not only on gen 16 and gen ≥ 17.

**This conflict is fully unresolved and no candidate explanation survives.**

The r0 result proposed one: a gen 13/14 host, where the MLX limit is 6, so all
scored calls would switch to `qmm_t` at `M ≥ 6`. That explanation is **dead**.
`routable()` (`Qwen35.swift:1767`) has no `arch_gen` term, so barrier 1
intercepts all 257 scored target calls at `M = 2...9` on every generation. A
gen 13/14 host would still send zero scored decode cells to `qmm_t`, so the
architecture probe cannot explain FINDING 444 in either outcome. Do not run it
for that purpose.

Nothing else on the table explains a −1.33 % local decode term either. Q does
not touch `qmv`, `qvm`, `qmm_n`, the crossrow kernels or the replica kernel
(section 1.8), and `qmm_t_splitk` is dead on this model (entry 258). Either the
FINDING 444 derivation carries a confound that its composite arithmetic hides,
or the census misses a live call path that no reading of the scored tree has
produced.

**The settling measurement is direct, and it is the only one I can name.** One
ABBA-counterbalanced local session, organizer-pure against organizer-pure plus
Q, on one host, with **prefill seconds and decode seconds logged separately**
instead of only the leg total. The existing local evidence reports the leg
only, and that is exactly the aggregation that lets a +1.94 % prefill penalty
and a spurious decode term hide inside one number. If that session shows a real
local decode effect, the census is incomplete and section 1 must be re-derived
from a runtime kernel counter rather than from source.

## 7. cap-4 (`harness=ranked`, moot for composition)

Recorded because I measured it before F2 declared the composition dead.

Receipt `90c131dc` is organizer main with one literal changed,
`segmentedVerifyDepthCap 7 -> 4`. Rejected at **3.54742900664627**, which is
**−4.33 %** against A at `3.70784519415395`.

| prompt | A edl | cap-4 edl | prefill % | decode % | leg % |
| --- | --- | --- | --- | --- | --- |
| plutarch | 0.156 | 0.156 | −0.2014 | −0.1145 | −0.1175 |
| drama | 2.298 | 2.298 | +0.1513 | −0.2175 | −0.1963 |
| travel | 2.648 | 2.591 | +0.2160 | −0.0468 | −0.0295 |
| beagle | 4.382 | 3.292 | −0.1937 | **+4.6343** | +4.1686 |
| republic | 4.989 | 3.562 | −0.1887 | **+5.6161** | +5.0001 |
| essays | 5.087 | 3.525 | +0.0070 | **+5.1855** | +4.6421 |
| medicine | 5.256 | 3.597 | +0.0173 | **+4.2324** | +3.7856 |
| botany | 6.148 | 3.827 | −0.0980 | −0.3191 | −0.2955 |
| **mean** | | | **−0.0363** (8) | **+2.7264** (7) | **+2.4393** (7) |

cap-4 is a clean decode-channel factor: prefill flat within the null on all
eight prompts, and the whole cost in decode. It is the exact mirror of Q, which
is prefill-only.

**Ruling, for the record: do not compose.** Q pays nothing at `M <= 5` because
it pays nothing at any decode `M`. cap-4 moves only the decode denominator, the
one channel Q does not touch, so composition cannot create a Q consumer and Q's
fixed prefill penalty rides on top unchanged. Composing the measured `B - A`
factors onto cap-4 gives a published median of **3.542212**, a further
−0.147 %.

**Anomaly worth its own question.** botany is clipped hardest, edl 6.148 to
3.827, and is the one clipped prompt that does not regress, at −0.32 % decode,
while beagle, republic, essays and medicine are clipped less and lose 4 % to
6 %. Whatever makes botany's deep drafts worthless is a real signal about the
adaptive depth walk, and it may be useful to Askeladd's E177 cost(M) fit.

## 8. Reproduction

```bash
python3 research/e176_cell_census.py                 # task 1, scored tree
YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
python3 research/e176_q_consumer_census.py           # tasks 2 and 3
WANDB_API_KEY=... python3 research/e176_wandb.py
swift research/e176_arch_probe.swift                 # Metal metadata only
```

The r1 amendment re-ran `e176_cell_census.py` and `e176_arch_probe.swift`. It
did **not** re-run `e176_q_consumer_census.py`: the board tables of sections 2
to 5 and 7 stay exactly as measured for r0, and only a provenance block was
added to `research/e176-q-census.json`.

## 9. Suggested follow-ups, not implemented

1. Settle FINDING 444 with the direct local pair of section 6: organizer-pure
   against organizer-pure plus Q, ABBA-counterbalanced, prefill and decode
   logged separately. The architecture probe cannot settle it, because barrier 1
   has no generation term.
2. Re-audit every finding derived from receipt C `fda590bb`. The additivity
   check that validated FINDING 440 is an identity and cannot detect a
   contaminated leg.
3. Report the null as a coherent whole-receipt offset of about −0.23 % plus a
   0.18 % per-prompt spread, not as a single symmetric sigma. Sign structure
   across the eight prompts is a stronger test than any mean.
4. A QMM-path win needs decode `M >= 10`, which the eight-draft cap forbids, or
   a change to the qmv path instead. Further `qmm_t` work cannot move ranked
   decode at all.
