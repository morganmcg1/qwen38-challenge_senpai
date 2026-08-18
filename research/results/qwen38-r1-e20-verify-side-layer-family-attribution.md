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

Nothing above this line was edited after the pre-registration commit (`42ad911`).
Corrections and deviations are recorded here and labelled as such.

### 2.0 Deviations from the pre-registration, stated up front

| # | Pre-registered | Actually done | Why |
|---|---|---|---|
| D1 | ABBA over gate-ON / gate-OFF | ABCCBA palindrome over **attribution modes** in one session | The real cool gate is unsatisfiable on this host (idle GPU 42.9 °C vs `COOL_GATE_TEMP_C=40`), so a gate-ON arm cannot exist. The palindrome counterbalances thermal drift across the quantity that actually varies. |
| D2 | 512-token window | **256-token** window | A pre-existing `notBegun` defect in the base makes every >300-token run abort (§2.9). 256 is the fixture's designed decode length. |
| D3 | Closure against `verify_build_us + eval_wall_us` | Closure against per-round **`block_request_seconds`** | Those two fields are not emitted by the trusted CLI. `block_request_seconds` is strictly stronger: it is the parent's own measurement. |
| D4 | Two modes (instrumented / inert) | **Three** modes: fine (mode 1), coarse (mode 3), inert (mode 2) | Mode 3 halves boundary density and turns "does the instrument distort the split?" into a measured quantity instead of an assumption. |
| D5 | Linear boundary-overhead correction | **Mode-2 apportionment** | Three boundary counts falsify the linear `c · evals` model (26–52 % residual). Kept only as a labelled diagnostic. |
| D6 | Four buckets | Four buckets **plus `drain`**, and MLP split by host layer type | `drain` isolates draft-head queue absorption so it cannot contaminate verify work; the MLP split makes mode 1's `gdn + mlp_gdn` exactly comparable to mode 3's `gdn`. |

**Correction to §1.7.** The failure condition "instrumented total decode time inflated by
more than the smallest inter-family gap" was evaluated during the run against a *wrong*
number. The 2.4–2.8× figure quoted in interim #2 compared mode-1 and mode-2 **forward
`total_ns`**, which is apples-to-oranges: mode 2's forward returns before the GPU drains
(forward occupancy 46.3 % of the block, versus 91.1 % in mode 1). Measured against the
parent's block time the instrument costs **+12.4 % at M=9** and **+30.3 % at M=3** —
it scales with round count, not width. This is still a deviation and is still an
inflation, but it is roughly a fifth of what was feared, and §2.6 shows it does not move
the split.

**Correction to the family split published in interim comment #3.** The split I posted
mid-run was produced by a scratch script that parsed only the `qwen-attrib:` records and
silently dropped the `qwen-attrib-span:` records. `top_two_ns` is emitted *only* on the
span line, so every `head_and_top_two` figure in that comment was too small and the
residual `round_overhead` was correspondingly too large. The shipped
`research/e20_analyze.py` merges spans into their parent forward
(`load_attrib`), and every number in this report is regenerated from it. The corrections
are: `head_and_top_two` 3.99 → **4.10 %** of verify-side work, `round_overhead`
0.6715 → **0.6641 s**. Every affected figure moved by ≤ 0.11 pp, which is below the
instrument's own 0.98 pp resolution (§2.6), so **no conclusion, ranking or lever
changes**. Interim comment #4 carries the same correction on the PR. The block-time
results — the null control, the perturbation ladder, the per-token economics and the
crossrow-QMV analysis — never went through the scratch script and are unaffected.

### 2.1 Host, base and provenance

| field | value |
|---|---|
| host | `ip-10-231-2-227.ec2.internal`, Apple M4 Pro (`applegpu_g16s`, arch_gen 16) |
| `BASE_SHA` | `c0f7e370921a14f348fa1872f2176b1b43028752` |
| instrumentation HEAD | `b78f0c91ac2d0ecebda39375be4cc51240fa8b48` |
| fixture | `public_longcopy_gate_english_512_256.json` (512 seed + 256 decode) |
| MTP head | `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head/`, 849,407,066 B, 15 tensors |
| `head_provenance_sha256` | `eb481df38267db5c9d9db1f6a813fcc73e762d0af74fdb1bcb061724c815adfe` |
| head origin | `hf:amal-david/qwen38-mtp-head-q4-qkv-islands-v1@8081fee431e304076b6f6296d6eb5dc7a3fc91af` |
| `mlxfast-swift` digest | `fa42e0f9eaff117c8ad507fa4b3768c8ab4bc0492b6731694db8abc51ab98361` (byte-identical across every arm) |
| worker digest, INSTR | `2e8f9d25952d3436f60dc4fefa63e099ed799e3109c5a359a05dda0dc95d5d58` (110 attribution symbols) |
| worker digest, BASE | `dd59dc7573a5f088c336c9c932d4fc2f609dc592a8cbbc2727bed58aaefdefb4` (0 attribution symbols) |
| `cool_gate_passed_real_gate` | **false** |
| `gate_qualified_for_timing` | **false** |

`MLXFAST_LOCAL_COOL_GATE=0` was required because the idle GPU floor on this host
(~42.9 °C) is above the harness gate (40 °C), so the gate can never be satisfied. The
two flags above are carried verbatim into the submitted result. Entry/exit GPU
temperature is recorded per arm in §2.2.

### 2.2 Arms

Every arm is `BUILD:MODE:DEPTH` at 256 decode tokens. Mode 1 = fine boundaries
(2·layers+3 = 131 evals), mode 3 = coarse (layers+3 = 67), mode 2 = inert logging
(0 evals, unperturbed scored path).

**Session S — shipped depth 8 (`ABCCBA` = modes 1,3,2,2,3,1)**

| arm | mode | block sum (s) | mean block (ms) | GPU °C in → out | dirty |
|---|---|---|---|---|---|
| S1 | 1 | 7.4612 | 226.10 | 40.93 → 64.16 | 0 |
| S2 | 3 | 7.0622 | 214.00 | 61.26 → 65.59 | 0 |
| S3 | 2 | 6.4710 | 196.09 | 62.62 → 66.26 | 0 |
| S4 | 2 | 6.4744 | 196.20 | 63.28 → 65.83 | 0 |
| S5 | 3 | 7.0836 | 214.65 | 63.02 → 65.71 | 0 |
| S6 | 1 | 7.4649 | 226.21 | 62.92 → 65.78 | 0 |

**Session D — offered depth 2 (`ABCCBA`), to resolve M ≤ 3**

| arm | mode | block sum (s) | GPU °C in → out | dirty |
|---|---|---|---|---|
| D1 | 1 | 9.1348 | 42.99 → 63.67 | 0 |
| D2 | 3 | 8.3771 | 61.08 → 65.48 | 0 |
| D3 | 2 | 6.9441 | 62.70 → 66.85 | 0 |
| D4 | 2 | 7.0848 | 63.75 → 66.39 | 0 |
| D5 | 3 | 8.3482 | 63.53 → 66.40 | 0 |
| D6 | 1 | 9.1484 | 62.50 → 64.76 | 0 |

**Session N — null control, offered depth 8 (`ABBA` over *binary*, not mode)**

Two arms run the instrumented binary with `MLX_QWEN_ATTRIB=0` (compiled in, fully
dormant) and two run the pristine `BASE_SHA` binary with zero attribution symbols. This
is the arm the pre-registration did not have and the one that decides whether any of the
above is admissible as a statement about the *shipped* path.

| arm | build | mode | block sum (s) | mean block (ms) | GPU °C in → out | dirty |
|---|---|---|---|---|---|---|
| N1 | INSTR | 0 | 6.5095 | 197.26 | 46.06 → 65.36 | 0 |
| N2 | BASE | — | 6.4993 | 196.95 | 62.31 → 64.95 | 0 |
| N3 | BASE | — | 6.4999 | 196.97 | 62.23 → 65.84 | 0 |
| N4 | INSTR | 0 | 6.5120 | 197.33 | 62.81 → 65.17 | 0 |

Run-to-run spread between palindrome repeats (the arms that differ only by position in
the thermal sequence): mode 1 **0.05 %**, mode 3 **0.30 %**, mode 2 **0.05 %** in
session S; mode 1 **0.15 %**, mode 3 **0.35 %**, mode 2 **2.03 %** in session D. Every
one of these is far below the smallest inter-family gap being claimed (4.81 pp), so the
counterbalanced design did its job. The mode-2 M=3 figure (2.03 %) is the single
noisiest measurement in the experiment and is flagged wherever it is used.

### 2.3 Correctness, every timed arm

| arm | `all_tokens_matched` | `parity_all_ok` | `residual_divergence_count` | `max_rejected_tail_logit_delta` | rounds | accepted draft rate |
|---|---|---|---|---|---|---|
| S1–S6 | true | true | 0 | 0 | 33 | 0.9912 |
| D1–D6 | true | true | 0 | 0 | 86 | 1.0000 |
| N1–N4 | true | true | 0 | 0 | 33 | 0.9912 |

`mtp-verify` reference: 257 rows, `self_consistent=true`, `chain_contradictions=0`.

**Width-histogram invariance.** Within each session the histogram derived from
`effective_draft_lengths` is *identical* across all six arms despite block-time
differences of up to 41 %:

* session S: `{2:1, 5:1, 6:6, 7:3, 8:3, 9:19}` (M=9 dominant, 19/33 rounds)
* session D: `{2:1, 3:85}`
* session N: `{2:1, 5:1, 6:6, 7:3, 8:3, 9:19}` — **identical to session S**, on a
  different day and on a binary that does not contain the instrumentation at all

This is the §1.7 timing-independence check and it passes cleanly. It is also direct
empirical confirmation of the source reading that `costModelDepth`
(`Qwen36MTPBlockSession.swift:599-635`) has **no wall-clock feedback** — the schedule
cannot react to how slow the instrumented build is.

**Row-ledger closure.** `row_ledger` is not present in the timed report
(`retainLedger: false`, `main.swift:1819`), so closure is done against
`effective_draft_lengths`, which reproduces the instrumented scored-forward histogram
exactly. Independently, `sum(block_request_seconds)` equals
`decode_seconds − seed_prefill_seconds` to four decimals on all sixteen timed arms (largest deviation 0.06 ms).

### 2.4 The accounting identity

Mode 2 leaves the scored path unperturbed, but its *forward* clock is not a wall-clock
denominator because the forward returns before the GPU drains. The parent's per-round
`block_request_seconds` is a valid denominator: it is measured outside the candidate
entirely. Round overhead (draft-head graph build, commit, rollback, worker protocol)
sits inside the block but outside the target forward, and the instrument never touches
it, so it is carried across at its **absolute** value:

```
overhead(M)     = block_m1(M) − attributed_m1(M)
target_work(M)  = block_m2(M) − overhead(M)
family_s(M)     = share_m1(M) × target_work(M) × rounds(M)
```

| M | rounds | `block_m1` ms | `attributed_m1` ms | overhead ms | `block_m2` ms | `target_work` ms | instrument inflation | mode-1 occupancy |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 101.43 | 100.77 | 0.65 | 73.76 | 73.11 | +37.5 % | 99.4 % |
| 5 | 1 | 259.74 | 238.32 | 21.42 | 163.12 | 141.70 | +59.2 % | 91.8 % |
| 6 | 6 | 178.72 | 167.38 | 11.35 | 150.03 | 138.68 | +19.1 % | 93.7 % |
| 7 | 3 | 193.85 | 178.13 | 15.73 | 164.62 | 148.89 | +17.8 % | 91.9 % |
| 8 | 3 | 235.89 | 215.79 | 20.10 | 207.94 | 187.83 | +13.4 % | 91.5 % |
| 9 | 19 | 249.49 | 224.94 | 24.55 | 222.00 | 197.45 | +12.4 % | 90.2 % |
| 3 | 85 | 106.35 | 105.55 | 0.80 | 81.65 | 80.85 | +30.3 % | 99.2 % |

Widths with `rounds ≤ 3` (M=2, 5, 7, 8) are noisy and are **not** used for any headline
claim; M=3 (n=85) and M=9 (n=19) carry the conclusions.

Choosing proportional rather than absolute overhead carry-across changes absolute
seconds by ~1.2 % and leaves every split percentage unchanged, because the split is a
ratio taken *within* the forward.

### 2.5 HEADLINE — the four-way split at the shipped width histogram

256-token decode window, depth 8, per window:

| family | seconds | **% of verify-side** | % of decode window |
|---|---|---|---|
| **MLP blocks (64)** | 3.3476 | **61.03 %** | 51.72 % |
| **GDN / linear-attention layers (48)** | 1.4193 | **25.88 %** | 21.93 % |
| **Full-attention layers (16)** | 0.4886 | **8.91 %** | 7.55 % |
| **LM head + two-stage top-two reducer** | 0.2249 | **4.10 %** | 3.47 % |
| embed | 0.0043 | 0.08 % | 0.07 % |
| *drain — draft-head, not verify* | 0.3239 | — | 5.00 % |
| *round overhead — draft-head/protocol* | 0.6641 | — | 10.26 % |
| **decode window total** | **6.4727** | | 100 % |
| **verify-side proper** | **5.4848** | 100 % | **84.74 %** |

Verify-side is **84.74 %** of the decode window. The campaign's prior 85.8 % figure was
a *modelled residual* (`13.996965 − 495 × 4.659177 ms`), not a measurement; this is an
independent direct measurement and it lands 1.1 pp away.

**Versus the pre-registered prediction** (§1.2: GDN 28 / FA 8 / MLP 59 / head 5):

| family | predicted | measured | error |
|---|---|---|---|
| GDN | 28 % | 25.88 % | −2.12 pp |
| full-attn | 8 % | 8.91 % | +0.91 pp |
| MLP | 59 % | 61.03 % | +2.03 pp |
| head+top2 | 5 % | 4.10 % | −0.90 pp |

The weight-traffic model was directionally right and ranked all four families correctly,
but it **understates MLP and overstates GDN**. Pure 4-bit-g64 weight traffic predicts
MLP 66.8 / GDN 21.7 / FA 6.6 / head 5.0 (14.413 GB per forward). The measurement sits
between that and the pre-registration, which is what you expect if GDN and full-attention
carry meaningful *non-GEMM* work (recurrent state update, unfused SDPA) on top of their
weight reads, while MLP is close to pure streaming.

### 2.6 Width resolution — the split is not width-invariant

| family (% of verify) | **M=3** (n=85) | M=6 (n=6) | M=7 (n=3) | M=8 (n=3) | **M=9** (n=19) |
|---|---|---|---|---|---|
| MLP | 58.51 | 59.87 | 60.20 | 61.32 | **61.42** |
| GDN | 28.47 | 26.84 | 26.60 | 25.72 | **25.46** |
| full-attn | 9.45 | 9.28 | 9.16 | 8.70 | **8.82** |
| head+top2 | 3.42 | 3.90 | 3.94 | 4.17 | **4.23** |

Shares move monotonically with M. A single-width answer would have been misleading,
which is exactly why the third point below M=6 was worth the GPU time.

**Absolute per-round cost (ms) on the unperturbed target forward:**

| M | n | depth | GDN | full-attn | MLP | head+top2 | embed | target forward |
|---|---|---|---|---|---|---|---|---|
| 3 | 85 | 2 | 21.06 | 6.99 | 43.26 | 2.53 | 0.115 | 80.85 |
| 6 | 6 | 8 | 34.62 | 11.98 | 77.23 | 5.03 | 0.131 | 138.68 |
| 9 | 19 | 8 | 47.66 | 16.50 | 114.96 | 7.91 | 0.135 | 197.45 |

**Fixed vs per-row structure**, fitting `cost(M) = a + b·M` on the two high-n widths
(M=3 from session D, M=9 from session S — the two ends of the sampled range and the
only two widths with n ≥ 19):

| family | a (fixed, ms/round) | b (ms/row) | fixed share @ M=9 | share of *marginal* cost |
|---|---|---|---|---|
| **MLP** | 7.42 | **11.95** | 6.5 % | **61.5 %** |
| **GDN** | 7.75 | 4.43 | **16.3 %** | 22.8 % |
| full-attn | 2.23 | 1.59 | 13.5 % | 8.2 % |
| head+top2 | −0.16 | 0.90 | ~0 % | 4.6 % |
| embed | 0.11 | 0.00 | 77.7 % | ~0 % |
| **target forward** | **22.54** | **19.43** | 11.4 % | 100 % |

Held-out validation of the linear model on widths not used in the fit: M=6 (n=6)
predicted 139.1 ms vs 138.68 measured (**0.3 %**); M=8 (n=3) 178.0 vs 187.83 (5.2 %);
M=7 (n=3) 158.6 vs 148.89 (6.5 %); M=5 (n=1) 119.7 vs 141.70 (15.5 %). Agreement tracks
sample count, as expected. Refitting *within* session S alone (M=6 and M=9, so no
cross-session anchor) moves MLP's marginal share from 61.5 % to 64.9 % and GDN's from
22.8 % to 22.4 %; the ranking and the qualitative conclusions are unchanged, and the
cross-session fit is preferred because both of its anchors have n ≥ 19.

Two conclusions:

1. **MLP owns 61.5 % of the marginal cost of each extra verified row** and is almost
   pure per-row work (6.5 % fixed). Anything that reduces per-row MLP cost pays off in
   direct proportion to drafting depth.
2. **GDN carries the largest fixed per-round cost** (7.75 ms, 16.3 % of its M=9 total).
   This is the sequential recurrent state update, and it is the component that deeper
   drafting *cannot* amortize — it sets a floor on round latency.

### 2.7 Validity of the attribution itself

**Null control — does the instrumented binary still describe the shipped path?**

Every number above rests on mode 2 (inert logging, 0 eval boundaries) being an honest
stand-in for the unmodified base, because `block_m2` is the denominator of
`target_work`. Session N tests that directly against the pristine `BASE_SHA` binary at
the identical width histogram, round count and acceptance rate:

| M | rounds | BASE (ms) | mode 0 (ms) | mode 2 (ms) | mode 0 − BASE | mode 2 − BASE |
|---|---|---|---|---|---|---|
| 2 | 1 | 73.81 | 73.80 | 73.76 | −0.01 % | −0.07 % |
| 5 | 1 | 189.15 | 188.67 | **163.12** | −0.25 % | **−13.76 %** |
| 6 | 6 | 150.01 | 150.45 | 150.03 | +0.29 % | +0.01 % |
| 7 | 3 | 164.59 | 164.64 | 164.62 | +0.03 % | +0.02 % |
| 8 | 3 | 208.05 | 207.97 | 207.94 | −0.04 % | −0.05 % |
| 9 | 19 | 222.03 | 222.51 | 222.00 | +0.22 % | −0.01 % |

**At every width carrying more than one round, mode 2 reproduces the pristine binary to
within 0.07 %.** The single exception is M=5, which is one round out of thirty-three;
it is the same width that already showed the worst boundary-density disagreement
(0.98 pp) and the worst cost-model residual (18 %). Substituting the BASE M=5 block time
would raise the window total by 0.4 % and move no family share by more than 0.03 pp,
because that round is 1/33 of the window. Whole-window aggregate: mode 2 sits 0.41 %
*below* BASE, which is causally impossible for an arm that only adds work — it is
entirely this one round, and it bounds the absolute-seconds precision of §2.5 rather
than its shares.

Dormant instrumentation (mode 0, compiled in but disabled) costs **+0.17 %** on the
whole window and is within 0.3 % of BASE at every width. Compiling the instrument in
does not itself perturb the path.

**Perturbation ladder.** With a true unmodified reference in hand, the whole instrument
can finally be priced against something real rather than against itself:

| configuration | arms | block sum (s) | vs BASE | repeat spread |
|---|---|---|---|---|
| BASE binary, no instrument compiled | N2+N3 | 6.4996 | — | 0.009 % |
| INSTR mode 0 (compiled, disabled) | N1+N4 | 6.5107 | **+0.17 %** | 0.037 % |
| INSTR mode 2 (inert logging, 0 evals) | S3+S4 | 6.4727 | −0.41 %¹ | 0.053 % |
| INSTR mode 3 (coarse, 67 evals) | S2+S5 | 7.0729 | **+8.82 %** | 0.303 % |
| INSTR mode 1 (fine, 131 evals) | S1+S6 | 7.4631 | **+14.82 %** | 0.049 % |

¹ the M=5 single round, above.

The serial control legs (`--mtp-depth 0`, 256 rounds at M=1) give a second, fully
independent reading of the same quantity — and it is the one that shows the boundary
cost is charged **per forward**, not per row:

| binary / mode | serial ms/token | vs BASE |
|---|---|---|
| BASE | 65.539 | — |
| INSTR mode 0 | 65.665 | +0.19 % |
| INSTR mode 2 | 65.594 | +0.08 % |
| INSTR mode 3 | 81.796 | +24.8 % |
| INSTR mode 1 | 90.187 | **+37.6 %** |

+37.6 % at M=1 against +14.8 % at the depth-8 histogram, and the accounting table
in §2.4 independently reports +37.5 % at M=2 and +12.4 % at M=9. Three separate
measurements of the instrument's cost agree, and all of them say the same thing: the
boundary cost is a fixed per-forward charge that dilutes as rows are added. That is why
it does not bias the split — it is added *outside* the buckets and subtracted as
`overhead(M)` before any family gets its share.

**Boundary-density invariance.** Mode 1 places 131 eval boundaries per forward, mode 3
places 67 — a 2× difference. If boundaries distorted *where* time appears, the two
would disagree. Worst-case disagreement on layer-group shares:

| session | worst disagreement |
|---|---|
| S (depth 8, six widths) | **0.98 pp** (at M=5, n=1) |
| D (depth 2, two widths) | **0.82 pp** (at M=3, n=85) |

At the two high-n widths the disagreement is 0.40 pp (M=9) and 0.82 pp (M=3).

**Window replication.** The 256-token headline reproduces the *independently measured*
64-token calibration headline (GDN 26.5 / FA 9.0 / MLP 60.6 / head 3.9) to within
0.6 pp on every family, across a 4× change in window length and a different width
histogram.

**Error budget vs claim size.** Smallest inter-family gap claimed: 4.81 pp (full-attn
8.91 vs head+top2 4.10). Largest error term: 0.98 pp (boundary density). Run-to-run
spread: ≤ 0.35 % (≤ 2.03 % for the one noisy mode-2 M=3 case). **The pre-registered
stopping rule is satisfied** — the four-way split spread is well below the smallest
claimed gap.

### 2.8 Crossrow-QMV dispatch column (pre-registered in §1.4, unchanged)

Source-derived, constant across every scored width M ∈ 3..9, and re-verified against the
live host (`applegpu_g16s`, arch_gen 16 → `get_qmv_batch_limit` else-branch; every scored
`D` ∈ {5120, 6144, 17408} > 4096 → limit **10**):

| family | wide crossrow (≥4096) | narrow crossrow (1024–4095) | non-crossrow `qmv_fast` |
|---|---|---|---|
| GDN (48 layers) | 144 | 0 | **96** (in_proj_a/b, N=48) |
| full-attention (16) | 32 | **32** (k_proj/v_proj, N=1024) | 0 |
| MLP (64) | **192** | 0 | 0 |
| LM head | 1 | 0 | 0 |
| **total per verify forward** | **369** | **32** | **96** |

Full attention is the only family with narrow-crossrow traffic; GDN is the only family
with non-crossrow QMV; MLP is pure wide-crossrow. The ranked M5 (`gen >= 17`) takes the
same else-branch, so the limit is 10 there too.

**This is the highest-leverage observation in the experiment.** Every scored width
M ≤ 9 is strictly below the batch limit of 10, so **the scored path never reaches the
`qmm` matrix kernel — it always takes the `qmv` vector path.** The measured consequence
is visible in the effective bandwidth of the verify forward, assuming weights are read
once per forward (14.413 GB at 4-bit g64):

| M | target forward | implied effective bandwidth |
|---|---|---|
| 3 | 80.85 ms | 178.3 GB/s |
| 9 | 197.45 ms | 73.0 GB/s |

Utilisation *falls* as M grows. That is the signature of a kernel family that does not
amortise weight reads across rows. A GEMM path would move in the opposite direction.

### 2.9 Defects found in the base (not caused by E20)

**A pre-existing `notBegun` defect blocks every run longer than ~300 tokens.** A
512-token attempt aborted at 152 s in the *serial control* leg with
`runtime worker mtp_decode_round failed: MTP round requested before the seed prefill`.
Root cause is base code at `c0f7e370`, `Qwen36MTPBlockSession.swift:753-771`: the
stop-token branch sets `reachedStopToken` and nils `pendingPrimary`/`pendingTop2`/
`pendingHidden`, but the parent keeps requesting rounds for the full configured window,
so the next round trips the guard at `:702-704`. My instrumentation counted 301 sustained
`rows=1` decode forwards on the serial leg, placing the stop token at decode position
≈ 301 in `public_longcopy_gate_english_512_256.json`. The greedy trajectory is
depth-independent, so every leg fails at the same place. It has never been hit because
`--local-iterate` is 64 tokens and `--local-submit` is 128. `mtp-verify` is unaffected
(`Qwen36MTPReferenceSession` has no stop-token early exit).

`git diff c0f7e370 -- Qwen36MTPBlockSession.swift` shows my only edits to that file are
a comment fix, a `topTwo` boundary, and `markVerifyRound(rows:)` — none touch the
stop-token path.

Per `program.md` this is a solver defect, not permission to shorten the ranked contract.
**I am deliberately not fixing it inside E20** (it is out of scope for an attribution
experiment and would confound every measurement here); it is reported as a campaign-level
follow-up. Consequence for this experiment: the headline is a **256-token** result and
must not be quoted as ranked-equivalent.

**Pre-existing `mlx.metallib` staleness warning** on every run (recorded
`3dd0ffd6…`, current `484f8492…`). Present identically on base and instrumented builds
and on all sixteen timed arms, so it cannot bias the split, but it means these absolute times
are from a build whose metallib does not match the vendored Metal sources on disk.

### 2.10 By-product: the draft head costs ~4.5 ms per proposed token

`drain + round_overhead` is draft-head and protocol work, not verify work. It scales with
the number of tokens the head actually proposes:

| session | mean proposed drafts/round | drain + overhead per round |
|---|---|---|
| D (depth 2) | 1.988 | 7.66 ms |
| S (depth 8) | 6.848 | 29.94 ms |
| fit | | **−1.46 ms fixed + 4.58 ms per proposed draft token** |

A second, independent estimator agrees: taking the per-*width* slope of the same two
buckets across M=3 and M=9 within the §2.6 accounting gives 3.96 ms/row of overhead plus
0.57 ms/row of drain = **4.52 ms per row**. The two differ by 1.3 %.

Combining with the verify fit gives a round model that is exact on both measured depths:

```
block(D) = 34.86 + 23.39·D  ms        (18.87 verify + 4.52 draft head per step)
  D=8 → 221.98 predicted vs 222.00 measured
  D=2 →  81.64 predicted vs  81.65 measured
```

The pristine BASE binary reproduces this independently: fitting the same form on its own
M=6 and M=9 block times gives `block(D) = 29.98 + 24.01·D ms`, a slope within 2.6 % of
the mode-2 fit, with no instrumentation present at all.

Each extra draft step costs ~23.4 ms and buys **at most** one token, so depth pays only
while acceptance stays near 1. Measured on the **pristine BASE binary**, prefill
excluded:

| configuration | tokens/round | ms/token | vs serial |
|---|---|---|---|
| serial control (depth 0) | 1.00 | 65.539 | 1.000× |
| depth 2 | 2.98 | 27.401 | 2.394× |
| depth 8 | 7.76 | 25.389 | **2.581×** |

> **Correction.** Interim comment #3 quoted 98.69 / 27.4 / 28.6 ms per token and
> concluded depth 2 beat depth 8. Those figures used
> `parent_measured_seconds_per_token`, which is **prefill-inclusive** — the 512-token
> seed costs 4.01 s and is charged to the same window, 38 % of it. With prefill removed
> the ordering reverses: depth 8 is 7.9 % faster per token than depth 2 on this fixture.
> The depth-2 and depth-8 serial controls agree to 0.07 % (65.539 vs 65.588 ms/token,
> different sessions and different binaries), so the comparison is sound.

**I am explicitly not making a ranked claim from this.** `public_longcopy_gate_english_512_256`
is a longcopy fixture with abnormally high acceptance (`accepted_draft_rate` 1.0000 at
depth 2, 0.9912 at depth 8); hidden prompts will not behave this way, and both legs here
use the same candidate build. It does say the depth schedule deserves a dedicated
experiment on varied prompts, which I am not running under this assignment.

### 2.11 Suggested levers (proposals only — none implemented)

Ranked by measured ceiling. "Ceiling" = the family's share of verify-side work, i.e. the
gain if that family's cost went to zero.

1. **Route wide-crossrow QMV to a batched/`qmm` path for M ≥ 4** — ceiling 61.0 %
   (MLP alone), realistically the shared per-row cost of MLP + GDN + head = 88 % of
   marginal cost. §2.8 shows the scored path is *always* on the vector kernel because
   every width is below `get_qmv_batch_limit() == 10`, and effective bandwidth falls
   from 179 to 73 GB/s as M goes 3 → 9. This is a small, well-localised change
   (`quantized.cpp` dispatch) with a directly measurable prediction: `b` (ms/row) should
   drop while `a` rises. **Highest expected value of anything in this report.**
2. **Fuse MLP gate and up projections** — ceiling ≈ 40 % of MLP (two of three
   projections). They share an input, so one 5120→34816 dispatch replaces two
   5120→17408 dispatches, halving input reads and one launch per layer × 64 layers.
3. **Attack GDN's 7.75 ms fixed per-round cost** — ceiling 4.0 % of the window, but it
   is the floor on round latency and it does *not* shrink with better drafting, so it
   caps the achievable speedup from any depth work. `mamba_ssm_dtype` is float32; a
   narrower accumulation is the obvious candidate but is numerically risky and would
   need focused exactness checks.
4. **Fused SDPA for head_dim 256** — ceiling 8.9 %. `supports_sdpa_full` requires
   head_dim ∈ {64, 80, 128}, so head_dim 256 fails at every width and the un-split path
   is fully unfused; `AttentionUtils` split=5 is already exactly maximal (guard
   `qL >= 6, qL <= 9`, gqa=6, `qL*gqa <= 32` → qL ≤ 5).
5. **Batch the LM head across rows** — ceiling 4.1 %. Cost is perfectly linear at
   0.90 ms/row with ~zero fixed component, which is consistent with no cross-row reuse
   over a 715 MB readout. Low ceiling; only worth doing if lever 1 makes it nearly free.

I did not implement any of these: this assignment is attribution, and each of the above
is a separate experiment with its own correctness surface.

### 2.12 Reproduction

```bash
# instrumentation build + both attribution sessions (256 decode tokens)
env E20_TOKENS=256 bash research/e20-run.sh \
  S1=INSTR:1:8 S2=INSTR:3:8 S3=INSTR:2:8 S4=INSTR:2:8 S5=INSTR:3:8 S6=INSTR:1:8
env E20_TOKENS=256 bash research/e20-run.sh \
  D1=INSTR:1:2 D2=INSTR:3:2 D3=INSTR:2:2 D4=INSTR:2:2 D5=INSTR:3:2 D6=INSTR:1:2

# null control: instrumentation compiled in but disabled, vs pristine base
env E20_TOKENS=256 bash research/e20-run.sh \
  N1=INSTR:0:8 N2=BASE:0:8 N3=BASE:0:8 N4=INSTR:0:8

# analysis — the analyzer takes arm directories positionally
python3 research/e20_analyze.py .mlxfast-private/e20/runs/S{1,2,3,4,5,6} \
  --json-out research/results/e20-analysis-S.json
python3 research/e20_analyze.py .mlxfast-private/e20/runs/D{1,2,3,4,5,6} \
  --json-out research/results/e20-analysis-D.json
python3 research/e20_analyze.py .mlxfast-private/e20/runs/N{1,2,3,4} \
  --json-out research/results/e20-analysis-N.json

# W&B logging — first file is the headline session, the rest are merged so the
# perturbation ladder and the cross-session fit are computed from logged data
python3 research/e20_log_wandb.py \
  research/results/e20-analysis-S.json \
  research/results/e20-analysis-D.json \
  research/results/e20-analysis-N.json \
  --group qwen38-r1-e20 --base-sha c0f7e370921a14f348fa1872f2176b1b43028752
```

Spec format is `LABEL=BUILD:MODE:DEPTH`. `MLX_QWEN_ATTRIB` selects the mode
(1 = fine, 3 = coarse, 2 = inert, unset/0 = off). Artifacts land per-arm under
`.mlxfast-private/e20/runs/<LABEL>/`.

The three analysis JSONs are committed under `research/results/`, so every table
below can be re-derived without re-running the GPU arms.

The headline split, the accounting-identity table, the per-width tables and the
fixed/per-row fit in §2.4–§2.6 all come from `research/e20_analyze.py`'s
`===== HEADLINE =====` block; nothing in this report is computed by an ad-hoc script
(see the correction in §2.0).

### 2.13 W&B run

| field | value |
|---|---|
| run | [`i5vrwffs`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/i5vrwffs) |
| project | `wandb-applied-ai-team/qwen38-mlx-challenge-senpai` |
| group | `qwen38-r1-e20` |
| state | `finished` |

One run covers all 16 timed arms. The pre-registered prediction of §1.2 is in
`config` as `prereg_share/*`, and the pre-registered crossrow-QMV column of §1.4
is in `config` as `crossrow_qmv/*`, so the comparison is visible without leaving
the run. Tables published:

| table | what it holds |
|---|---|
| `headline_split` | §2.5, per session |
| `headline_accounting_identity` | §2.4, every width in both sessions |
| `headline_marginal_fit` | §2.6 fixed vs per-row cost, within-session and cross-session |
| `perturbation_ladder` | §2.7 instrument cost by depth, build and mode, MTP and serial legs |
| `arms` | §2.2 and §2.3, every arm's timing, thermals, binary digests and correctness |
| `width_histograms` | the shipped width histogram and the instrumented forward counts |
| `family_shares_by_width` | raw scored and warmup shares |
| `mode1_vs_mode3_agreement` | §2.7 validity check, scored and warmup |
| `boundary_overhead_fits` | the falsified linear-in-boundary-count fit |
| `corrected_shares_by_width` | the split that fit implies — kept only so the falsification is auditable |
| `diagnostic_apportioned_family_seconds` | the superseded pooled-share apportionment, audit only |

Summary keys mirror the tables. `headline/*` is session S, `headline/D/*` is
session D, `prereg_error_pp/*` is the §2.5 pre-registration error in percentage
points, and `agreement/max_abs_share_delta` is the §1.7 stopping-rule number
(0.00976, i.e. 0.98 pp, over scored forwards; warmup forwards reach 1.01 pp and
are outside the timed window).
