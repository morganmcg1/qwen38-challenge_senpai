# E20 — Verify-side layer-family attribution at the shipped width histogram

Assignment: `qwen38-r1-e20-verify-side-layer-family-attribution` (rev `r1`), PR #24.
Base: `c0f7e370921a14f348fa1872f2176b1b43028752` on `senpai/qwen38-mtp-r1`.

**This experiment builds a map, not a speedup.** Deliverable is a four-way split of
verify-side decode work plus a ranked list of levers. A null result publishes.

---

## PART 1 — PRE-REGISTRATION (committed before the first timed run)

Everything below this line and above "PART 2" was written and committed **before any
instrumented binary was built or any GPU timing was taken**, so the measurement is
separable from the narrative in git history.

### 1.1 The four buckets

| # | Family | Instances | What is charged to it |
|---|--------|-----------|-----------------------|
| 1 | GDN / linear attention | 48 layers | `Qwen35GatedDeltaNet.callAsFunction` end-to-end: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, conv1d, the recurrent scan, output norm, `out_proj` |
| 2 | Full attention | 16 layers | `Qwen35Attention.callAsFunction` end-to-end: `q_proj`/`k_proj`/`v_proj`, RoPE, cache update, SDPA (incl. the `split = 5` two-call workaround), output gate, `o_proj` |
| 3 | MLP | 64 blocks | `Qwen35FusedMLP.callAsFunction`: `gate_proj`, `up_proj`, SiLU-mul, `down_proj` |
| 4 | LM head + top-two reducer | 1 each | `model.norm`, `lmHead`, and the two-stage custom-Metal top-two reduction at `Qwen36MTPBlockSession.swift:963` |

Embedding gather, mask construction and the residual adds are **not** in the four
buckets; they are measured into a fifth `other` bucket that is reported separately for
closure accounting. The four pre-registered percentages are shares of the four-family
total (they sum to 100 by construction); `other` is reported as a percentage of the
grand total alongside them.

### 1.2 Pre-registered prediction

| Family | Predicted share of four-family verify work |
|--------|-------------------------------------------:|
| GDN / linear attention (48) | **28 %** |
| Full attention (16) | **8 %** |
| MLP (64) | **59 %** |
| LM head + top-two reducer | **5 %** |
| **sum** | **100 %** |

**Predicted largest term, and why (one line):** *MLP*, because at verify widths
M ≤ 9 the target forward is weight-bandwidth-bound rather than FLOP-bound, and the
three affine-4-bit `17408 × 5120` projections per layer are 65.1 % of the ~14.8 GB of
weight-plus-state bytes a single verify forward must read, so byte share dominates even
after crediting GDN's sequential-in-T scan and full attention's unfused-SDPA dispatch
overhead.

### 1.3 The arithmetic behind the prediction

Geometry from `weights/config.json` and `Sources/MLXFastCore/Constants.swift`:
hidden 5120, intermediate 17408, vocab 248320, 64 layers, `full_attention_interval` 4
(→ 48 GDN + 16 full attention), 24 Q heads / 4 KV heads / head_dim 256,
`linear_num_key_heads` 16 × `linear_key_head_dim` 128 = 2048,
`linear_num_value_heads` 48 × `linear_value_head_dim` 128 = 6144,
`mamba_ssm_dtype` float32, quantization affine 4-bit group-64.

Affine 4-bit g64 costs 0.5 B/param for nibbles plus one fp16 scale and one fp16 bias
per 64 params = **0.5625 B/param**.

Weight bytes read per verify forward (M-independent):

| Family | Per-instance params | ×instances | MB |
|--------|--------------------:|-----------:|---:|
| GDN | `in_proj_qkv` 10240×5120 + `in_proj_z` 6144×5120 + `in_proj_a` 48×5120 + `in_proj_b` 48×5120 + `out_proj` 5120×6144 = 115,834,880 (+82 KB bf16 conv1d/norm) | ×48 | 3131.5 |
| Full attention | `q_proj` 12288×5120 + `k_proj` 1024×5120 + `v_proj` 1024×5120 + `o_proj` 5120×6144 = 104,857,600 | ×16 | 943.7 |
| MLP | 3 × 17408×5120 = 267,386,880 | ×64 | 9625.9 |
| LM head | 248320×5120 = 1,271,398,400 | ×1 | 715.2 |

`q_proj` is 12288 = `heads × head_dim × 2` because `attn_output_gate: true` packs the
per-head gate into the same projection.

State and activation traffic at M = 6, KV length ≈ 576:

* GDN recurrent state 48 v-heads × 128 × 128 fp32 = 3.146 MB/layer, read + write =
  6.29 MB/layer × 48 = **302 MB** if the scan loads state once per forward.
* Full-attention KV cache 4 heads × 256 × 2 tensors × 2 B = 4096 B/token/layer,
  × 576 × 16 = **37.7 MB**.
* Logits M × 248320 × fp32 written then re-read by the reducer ≈ **12 MB**.

Totals → GDN 3433.5 MB (23.2 %), full attention 981.4 MB (6.6 %), MLP 9625.9 MB
(65.1 %), LM head + reducer 727.2 MB (4.9 %); grand total 14.77 GB.

The pre-registered numbers move GDN up 5 pts and full attention up 1.4 pts against that
pure-bandwidth baseline, taking the difference out of MLP and the LM head, for two
source-derived reasons:

1. The GDN scan is sequential in T (established fact, `research/ESTABLISHED_FACTS.md`),
   so it pays ≥ M dispatches per layer and may re-materialise the 3.15 MB state per
   step. If it re-materialises fully at M = 6, GDN's byte share alone rises to 30.1 %.
2. `head_dim = 256` is not in `{64, 80, 128}`, so `supports_sdpa_full` is false at every
   width (`scaled_dot_product_attention.cpp:625-626`), and `supports_sdpa_vector` needs
   `qL × gqa ≤ 32` with `gqa = 6`, i.e. `qL ≤ 5` (`:634-637`). At M ≥ 6 both fused paths
   fail, which is exactly why `AttentionUtils.swift:125` splits at `let split = 5`.
   Full attention therefore pays extra dispatches and a `concatenated` at every width
   this experiment cares about.

**Sensitivity check on the headline claim:** under the most GDN-favourable assumption
(state fully re-materialised every scan step at M = 6) the byte split is GDN 30.1 %,
full attention 6.0 %, MLP 58.6 %, LM head 4.4 %. MLP still leads by 28 pts. The
"MLP is largest" call is therefore robust to the single biggest modelling uncertainty;
the *sizes* of the other three are not, which is what makes measuring worthwhile.

### 1.4 Pre-registered crossrow-QMV dispatch column (source-derived, not measured)

Host side, `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp`:
`get_qmv_batch_limit` (`:84`) returns **10** for every scored shape on this host
(`applegpu_g16s`, arch_gen 16 → `default:` case; all K/D ∈ {5120, 6144, 17408} > 4096
→ `:121`). `eval_gpu:1415` sets `vector_limit = transpose_ ? that : 4` and `:1418`
routes `M >= vector_limit` to qmm, so **every width M ≤ 9 reaches `dispatch_qmv`**.
Ranked M5 (gen ≥ 17) takes the same `default:` branch, so this transfers.

`qmv()` (`:250-260`) sets `bn = 8` and selects `_qmv_fast_` iff `N % 8 == 0 && K % 512 == 0`,
and dispatches `grid_dims = (M, ceil(N/8), B)` — so **`ntg.x == M`**. Device side,
`kernels/quantized.h:1822` then gates crossrow on
`!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024`, and `:1823` splits
wide from narrow at `out_vec_size >= 4096`.

| M | `out_vec_size ≥ 4096` (wide) | `1024 ≤ out_vec_size < 4096` (narrow) | `< 1024` |
|---|---|---|---|
| 1 | *(no case — falls through to plain `qmv_fast`)* | *(no case)* | plain `qmv_fast` |
| 2 | `_g64<T,2>` pair kernel, no `DIRECT_NIBBLES` | `_g64<T,2>` | plain `qmv_fast` |
| 3 | `_g64_m<T,3,3,true>` | `_g64<T,3>` | plain `qmv_fast` |
| 4 | `_g64_m<T,4,4,true>` | `_g64<T,4>` | plain `qmv_fast` |
| 5 | `_g64_m<T,5,3,true>` | `_g64<T,5>` | plain `qmv_fast` |
| 6 | `_g64_m<T,6,3,true>` | `_g64<T,6>` | plain `qmv_fast` |
| 7 | `_g64_m<T,7,4,true>` | `_g64<T,7>` | plain `qmv_fast` |
| 8 | `_g64_m<T,8,3,true>` | `_g64<T,8>` | plain `qmv_fast` |
| 9 | `_g64_m<T,9,3,true>` | `_g64<T,9>` | plain `qmv_fast` |

Mapping the scored projections onto that table (every K here is a multiple of 512 and
every N a multiple of 8, so all take `_qmv_fast_`):

| Family | Projection | N = `out_vec_size` | K | Crossrow class | Dispatches / forward |
|--------|-----------|-------------------:|---:|----------------|---------------------:|
| GDN | `in_proj_qkv` | 10240 | 5120 | **wide** (`true` at M 3–9) | 48 |
| GDN | `in_proj_z` | 6144 | 5120 | **wide** (`true` at M 3–9) | 48 |
| GDN | `out_proj` | 5120 | 6144 | **wide** (`true` at M 3–9) | 48 |
| GDN | `in_proj_a` | 48 | 5120 | **none** (N < 1024) | 48 |
| GDN | `in_proj_b` | 48 | 5120 | **none** (N < 1024) | 48 |
| Full attn | `q_proj` | 12288 | 5120 | **wide** (`true` at M 3–9) | 16 |
| Full attn | `o_proj` | 5120 | 6144 | **wide** (`true` at M 3–9) | 16 |
| Full attn | `k_proj` | 1024 | 5120 | **narrow** (pair kernel, never `true`) | 16 |
| Full attn | `v_proj` | 1024 | 5120 | **narrow** (pair kernel, never `true`) | 16 |
| Full attn | SDPA | — | — | not a QMV | — |
| MLP | `gate_proj` | 17408 | 5120 | **wide** (`true` at M 3–9) | 64 |
| MLP | `up_proj` | 17408 | 5120 | **wide** (`true` at M 3–9) | 64 |
| MLP | `down_proj` | 5120 | 17408 | **wide** (`true` at M 3–9) | 64 |
| LM head | `lm_head` | 248320 | 5120 | **wide** (`true` at M 3–9) | 1 |
| LM head | top-two reducer | — | — | custom Metal, not a QMV | 2 |

Per verify forward at any M in 3–9: **369 wide-crossrow-`true` dispatches**
(GDN 144, full attention 32, MLP 192, LM head 1), **32 narrow-crossrow pair dispatches**
(full attention `k_proj`/`v_proj`), and **96 non-crossrow `qmv_fast` dispatches**
(GDN `in_proj_a`/`in_proj_b`).

Two consequences worth stating before the measurement:

* Full attention is the **only** family with any narrow-crossrow traffic, and GDN is the
  **only** family with any non-crossrow QMV traffic. A crossrow-kernel lever is
  therefore not uniform across families.
* `upstream/main` moved `d1530a4 → 0d800b2` by adding exactly the `true` flag to
  `<T,3,3>`, `<T,4,4>` and `<T,5,3>` (promoted 3.14643, +1.56 %). That six-line change
  touched all 369 wide dispatches, but only at M ∈ {3,4,5}. Which widths actually occur
  is therefore load-bearing, which is why this experiment reports a width-resolved split.

### 1.5 Pre-registered width weighting, with provenance attached to every number

There is **no shipped-default 512-token width histogram in existence** at the time of
writing, so no width weighting used here is authoritative. The two histograms that do
exist are:

* Advisor receipt `e29a3e0d`, **shipped default**, **128-token `--local-submit`**:
  depths `{1:1, 4:1, 5:8, 6:3, 7:5, 8:2}` → widths `{2:1, 5:1, 6:8, 7:3, 8:5, 9:2}`,
  mean depth 5.7; M = 6–9 is 18 of 20 rounds (90 %).
* Edward's **E17 arm S18**, **512-token**, **under his own non-default depth policy**
  (`research/results/qwen38-r1-e17-curve-transfer-and-refit.md`): depths
  `{1:19, 2:138, 3:67, 4:21}` → 226 of 245 rounds at M = 2–5, mode M = 3 ≈ 56 %.

These disagree because they are different policies at different window lengths, not
because either is wrong. This experiment therefore reports the split **width-resolved
and unweighted** as the headline, and any pooled number is labelled with which of the
two histograms weighted it, in the same sentence as the number. Target widths, in
priority order: **M = 6** and **M = 9** (the shipped-default mode region), then
**M = 3** (plausibly the modal ranked width, and the width the promoted frontier commit
just started flagging).

### 1.6 Method, fixed in advance

* **Instrumentation.** A new `MLX_QWEN_ATTRIB=1` gate (the `MLX_` prefix is mandatory:
  `MLXFAST_*` is stripped from the worker environment, `allowedPrefixes =
  ["DARKBLOOM_","DYLD_","LC_","METAL_","MLX_","MTL_"]`). When the gate is off the added
  code must be inert. Forced `eval()` barriers bracket each sub-block inside
  `Qwen35DecoderLayer.callAsFunction`; embed/mask and norm+lmHead get their own buckets;
  the top-two reducer is timed at `Qwen36MTPBlockSession.swift:963`. The asyncEval
  ladder in `Qwen35TextModelInner.callAsFunction` (`:1966`, `:1971`) is bypassed while
  instrumented, because a ladder boundary inside a bucket would misattribute the wait.
* **Keying.** One stderr record per forward, keyed by row count S. Verify forwards have
  S = M, prefill has S = 512, serial and repair forwards have S = 1. Width resolution
  therefore falls out of a single instrumented run at no extra GPU cost.
* **Closure.** Sum of the five buckets is checked against the round's existing
  `verify_build_us + eval_wall_us` counters (`Qwen36MTPBlockSession.swift:1115-1131`).
  A closure gap above 5 % invalidates the split.
* **Prefill.** `decode_seconds` is prefill-inclusive
  (`QwenRuntimeMTPDriver.swift:94/100/197`). Prefill is subtracted before any
  share-of-decode number is computed. No exception.
* **Depth policy is timing-independent.** `costModelDepth`
  (`Qwen36MTPBlockSession.swift:599-635`) reads only `positionAcceptEMA`, logit-margin
  confidence, the streak/width-wall caps and the compile-time `headStepCostRatio` — no
  wall-clock term. Adding `eval()` barriers therefore *should not* move the width
  histogram. This is a prediction, and it is checked by comparing the histograms of the
  gate-ON and gate-OFF arms.
* **Design.** ABBA counterbalanced inside a single session:
  A = gate ON, B = gate OFF, order A B B A. Entry and exit GPU temperature recorded per
  arm. Run-to-run spread from the B B pair is reported next to every effect size.
* **Null control.** Required, because code is being changed: the unmodified base build
  is timed in the same session against the gate-OFF arm, to show the added code is inert
  when the gate is off.
* **Thermal honesty.** Idle GPU on this host sits at ~42.9 °C against
  `COOL_GATE_TEMP_C = 40`, so the real gate is unsatisfiable and
  `MLXFAST_LOCAL_COOL_GATE=0` is used. Consequently
  `cool_gate_passed_real_gate = false` and `gate_qualified_for_timing = false` are
  carried verbatim into the result. These are attribution shares, not a score claim,
  which is why hot-start timing is acceptable here at all.
* **Stopping rule.** Stop when the four-way split's spread is smaller than the smallest
  gap being claimed, or at **3 GPU-hours**, whichever comes first. If the
  instrumentation perturbs total decode time by more than the effect being measured,
  report that and stop.
* **Non-goal.** No speedup is attempted. Levers are proposed, not built.

### 1.7 What would make this experiment a failure rather than a negative result

* Closure gap > 5 % between the bucket sum and `verify_build_us + eval_wall_us`.
* Gate-ON vs gate-OFF width histograms differ (would falsify the timing-independence
  claim and make every gate-ON number unrepresentative).
* Instrumented total decode time inflated by more than the smallest inter-family gap
  being claimed.
* `all_tokens_matched != true` on any timed arm.

---

## PART 2 — MEASUREMENT

*(Not yet run. Nothing above this line may be edited after this commit; corrections go
below it and are labelled as corrections.)*
