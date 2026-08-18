# E23 — Verify-forward GPU dispatch inventory

**Assignment:** `qwen38-r1-e23-verify-forward-dispatch-inventory` (PR #27, rev r1)
**Base:** `c0f7e370921a14f348fa1872f2176b1b43028752`
**Type:** zero-GPU static source analysis. No run, no W&B run, no timing claim.
**Deliverable:** [`research/verify_forward_dispatch_inventory.py`](../verify_forward_dispatch_inventory.py)
**Pre-registration:** [`research/e23-prereg.md`](../e23-prereg.md), posted to PR #27 before the
enumerator was written ([comment](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/27#issuecomment-5322384606)).

## Unit of account

One call of `Qwen36MTPBlockSession.swift:951` `model.callWithHiddenAndNormed(...)` with `M` verify
rows, plus the top-two reducer at `:963`. `M = 1 + draftCount`, so `M ∈ [1, 9]`. Counted at
`M ∈ {1, 2, 3, 4, 5, 6, 8, 9}`. A "dispatch" is one Metal kernel launch issued by the MLX backend.

Reproduce:

```bash
python3 research/verify_forward_dispatch_inventory.py --report            # primary
python3 research/verify_forward_dispatch_inventory.py --breakdown         # per-item + citations
python3 research/verify_forward_dispatch_inventory.py --predict           # falsifiable numbers
python3 research/verify_forward_dispatch_inventory.py --verify-citations  # exits 1 on source drift
python3 research/verify_forward_dispatch_inventory.py --selftest          # 38 internal checks
```

## PRIMARY RESULT — unweighted per-M table

Scenario `qmv_limit=10`, `layout_copies=counted` (both knobs are varied under *Sensitivity* below).
This unweighted table is the primary result. Every weighted number in this document is a labelled
secondary sensitivity analysis.

| M | gdn | attn | mlp | envelope | head | **TOTAL** | gdn % | attn % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 912 | 112 | 192 | 261 | 3 | **1480** | 61.6 % | 7.6 % |
| 2 | 960 | 128 | 192 | 261 | 3 | **1544** | 62.2 % | 8.3 % |
| 3 | 432 | 128 | 192 | 261 | 3 | **1016** | 42.5 % | 12.6 % |
| 4 | 432 | 128 | 192 | 261 | 3 | **1016** | 42.5 % | 12.6 % |
| 5 | 432 | 128 | 192 | 261 | 3 | **1016** | 42.5 % | 12.6 % |
| 6 | 432 | 208 | 192 | 261 | 3 | **1096** | 39.4 % | 19.0 % |
| 8 | 432 | 208 | 192 | 261 | 3 | **1096** | 39.4 % | 19.0 % |
| 9 | 432 | 208 | 192 | 261 | 3 | **1096** | 39.4 % | 19.0 % |

### Headline: dispatch count is NON-MONOTONIC in M

**A width-6 verify forward issues 29 % FEWER GPU dispatches than a width-2 one** (1096 vs 1544).
The count *falls* by 34 % going from M=2 to M=3 and then is flat across M ∈ {3,4,5} and again across
M ∈ {6,8,9}, with one +80 step at M=6.

The mechanism is the already-promoted packed GDN prework mixer. At `Qwen35.swift:1008` the branch
`nConfirmed == 1 && S >= 3 && mask == nil` routes into `processChunkStashingPrefix`
(`Qwen35.swift:778`), whose single packed kernel at `:812` replaces roughly eleven satellite launches
per GDN layer that the M≤2 paths still issue individually (concat ×2, conv1d, silu, three reshape
copies, two scalar `asType`, two rmsNorm, two multiplies). Multiplied over 48 GDN layers that is
where the ~450-dispatch cliff comes from.

This inverts the naive cost model. Per *emitted* token the effect is far stronger still: at M=6 the
forward costs 1096 dispatches for up to 6 tokens (≤183/token) against 1544 for up to 2 (≥772/token).
Dispatch count is of course not latency — but it does mean **there is no dispatch-count argument for
preferring narrow verify widths, and a dispatch-count argument against M=2 specifically.**

## Per-family derivation

Full per-item rows with per-M counts, confidence tags and file:line citations are in
`--breakdown`. Summary of where each number comes from:

### GDN, 48 layers — 912 / 960 / 432

Branch selection is on `(nConfirmed, S)`; in the verify forward `S == M` and `nConfirmed == 1`.

| M | branch | per-layer launches |
|---|---|---:|
| 1 | `processChunk` (`Q35:732`) | 19 |
| 2 | inline mid-kernel branch (`Q35:1024+`) | 20 |
| ≥3 | `processChunkStashingPrefix` packed mixer (`Q35:778`) | 9 |

Per-layer at M≥3: fused in-proj `quantizedMM` (1, `Q35:675`) + `z` reshape copy (1, `Q35:981`) +
packed prework mixer (1, `Q35:812`) + prepared recurrence (1, `Q35:868`) + two scalar `asType`
(2, `Q35:811`) + tail rmsNorm & compiled postnorm (2, `Q35:1152`) + outProj (1, `Q35:1157`).

### Full attention, 16 layers — 112 / 128 / 208

Per layer: packed QKV `quantizedMM` (1) + fused QK-RMSNorm-RoPE kernel (1, `Q35:1736`) + 2 KV
slice-update writes (`KVCache.swift:434-435`) + SDPA + post-SDPA transpose/reshape copy (1 at M≥2) +
compiled sigmoid-multiply gate (1, `Q35:1765`) + oProj (1).

The +80 step at M=6 is the SDPA split. `scaled_dot_product_attention.cpp:634` requires
`qL * gqa ≤ 32` for the fused vector kernel; with `gqa = 24/4 = 6` that is **qL ≤ 5**. The guard at
`AttentionUtils.swift:122-125` splits any `qL ≥ 6` at row 5 precisely so both arms stay on the fused
kernel. Each split therefore costs, per layer: +1 SDPA, +2 query contiguity copies
(`scaled_dot_product_attention.cpp:686-718`), +2 concat copies (`AttentionUtils.swift:141`) = +5,
×16 layers = +80.

Note `head_dim = 256` is **not** in `supports_sdpa_full`'s `{64, 80, 128}`, so without that split
`qL ∈ 6..9` would fall off the fused path into an unfused op graph entirely. The split is not an
optimisation, it is load-bearing for correctness of the fast path.

### MLP, 64 blocks — 192, flat in M

All 64 are `Qwen35FusedMLP`; the MoE branch (`Q35:1841-1842`) is dead for this checkpoint
(`numExperts == 0`). The fused guard at `Q35:1263` holds for all M ≤ 16, so every block is
gate_up `quantizedMM` + `qwen35CompiledFusedSwiGLU` + downProj = 3.

### Envelope — 261, flat in M

Embedding 4 (`Q35:1935`, 4-bit gather + scale/bias gathers + dequant) + per-block 2 RMSNorm
(`Q35:1879`, `Q35:1886`) + per-block 2 residual adds (`Q35:1885`, `Q35:1886`) = 4×64 + final norm 1
(`Q35:2346`).

### Head — 3, flat in M *and* in the 248,320-token vocabulary

LM head runs over **all M rows** with no slicing (`Q35:2349`, `[1, M, 248320]`), then the two-launch
top-two reducer (`Qwen36MTPBlockSession.swift:1505`, `:1512`).

## Two corrections to the assignment's premises

**1. QKV is not unfused at dispatch time.** The assignment (inheriting E18) treats the attention QKV
projection as UNFUSED. That is a *checkpoint-layout* fact, not a dispatch fact: at runtime
`Q35:1553-1562` builds one packed `quantizedMM` of out-dim 14336, lazily at `Q35:1577-1585` and
warmed at `Qwen36MTPBlockSession.swift:218`. The scored path issues **one** projection launch per
attention layer, not three. My Prediction 1 leaned on the unfused premise; it was wrong to.

**2. The exact-QKV-rows island is dead in the target.** `replaceExactRows` early-returns at
`Q35:1646`, and the only callers of the `installExactQKVRows` setters (`Q35:1663`, `Q35:1683`) are at
`Q35:2266` against `mtp?.layers.first` — the **proposal head**, never the 16 target full-attention
layers. No verify-forward dispatch comes from it at any M.

## Pre-registered predictions — scored honestly

| # | prediction | outcome |
|---|---|---|
| 1 | GDN dominates at M=8; point 65 %, confirm ≥55 %, refute <45 % | **REFUTED.** 39.4 % at M=8. |
| 2 | GDN 1400–3000, MLP ~400, attn ~200, head <15, total 2000–4000 | **REFUTED.** GDN 432, MLP 192, total 1096. |
| 3 | GDN superconstant in M; attn +16 at M>5; else "total nearly flat" | **PARTIALLY REFUTED.** |
| 4 | no projection leaves the QMV family for M ∈ 1..9 | **CONFIRMED**, conditional on the limit. |

**Prediction 1 — refuted, and instructively so.** GDN's share is 61.6 % at M=1 and 62.2 % at M=2, so
the prediction would have been *confirmed* had I only looked at narrow widths. It fails at M=8
exactly because the packed mixer already fixed the thing I predicted would dominate. I was
predicting the cost structure of a version of this code that no longer exists.

**Prediction 2 — refuted on every line.** I over-predicted the total by roughly 2–3.5×. The two
errors were the unfused-QKV premise above and an assumption that MLP would cost ~6 launches per
block when the fused SwiGLU makes it 3.

**Prediction 3 — the alternative branch was closest, but the truth is stronger.** I wrote that if GDN
were not superconstant the headline would be "total nearly flat in M". GDN is indeed not
superconstant — it is *decreasing*, with a step **down** of 528 dispatches at M=3, which no branch of
my prediction anticipated. The attention step is +80 at M≥6, not the +16 I predicted (I counted one
extra SDPA launch per layer and missed the four copy launches per split).

**Prediction 4 — confirmed, conditional.** `get_qmv_batch_limit` (`quantized.cpp:84-126`) returns 10
on this generation, and `M ≤ 9 < 10` never trips the `qmm_splitk` route at `:1418`. This is the one
prediction whose truth depends on a device property I could not measure without a GPU; see
Sensitivity.

## SECONDARY / SENSITIVITY ONLY

Everything below is a labelled sensitivity analysis. Provenance is stated in the same sentence as
each number, per the calibration rule.

### Scenario knobs

| scenario | M=1 | M=2 | M=3–5 | M=6–9 | headline survives? |
|---|---:|---:|---:|---:|---|
| `qmv_limit=10`, layout copies counted (primary) | 1480 | 1544 | 1016 | 1096 | — |
| `qmv_limit=10`, layout copies zeroed | 1288 | 1336 | 952 | 1000 | yes, M=6 is 25.1 % cheaper than M=2 |
| `qmv_limit=6`, layout copies counted | 1480 | 1544 | 1016 | 1353 | yes, M=6 is 12.4 % cheaper than M=2 |

The non-monotonicity headline is robust to both knobs. Its magnitude is not.

### Width weightings — both are proxies, neither is authoritative

**GAP: no shipped-default 512-token verify-width histogram exists anywhere in the tree.** Until one
is captured, the unweighted per-M table above is the only safe basis for a dispatch-coverage
decision.

- **Weighted by the 128-token `--local-submit` shipped-default receipt (20 verify rounds, ONE public
  fixture):** mean 1114 dispatches per verify forward, of which GDN 41.1 % and attention 17.9 % —
  weights from the 128-token `--local-submit` shipped-default receipt, 20 verify rounds, one public
  fixture. *Per calibration rule (f) this must never size a dispatch-coverage claim*: it is a
  128-token single-prompt screen, and its histogram keys are ambiguous — read here as verify width M,
  but if they are draft counts every weight shifts by +1.
- **Weighted by the E17/S18 512-token accepted-depth histogram, converted `M = depth + 1` (245
  rounds):** mean 1057 dispatches per verify forward, of which GDN 44.7 % and attention 12.1 % —
  weights from the E17/S18 512-token accepted-depth histogram, a *different experiment's schedule*,
  not the shipped default. Mode is M=3 at ~56 %.

Both proxies agree the shipped operating point sits in the cheap M≥3 regime and that GDN is a
plurality but not a majority there. That agreement is worth exactly as much as the two proxies are,
which is not much.

## Falsifiable numbers

A single GPU profiler capture of one verify forward kills any of these outright. From `--predict`:

- **N1** total dispatches at M=6 = **1096** (±5 % for the layout-copy band; zeroed gives 1000)
- **N2** dispatches(M=2) / dispatches(M=6) = **1.409** — narrow verify is *more* dispatch-expensive
- **N3** GDN-family share at M=8 = **39.4 %**
- **N4** exactly **48** `qwen35PackedGDNPrework` dispatches at M≥3, exactly **0** at M≤2
- **N5** exactly **32** `sdpa_vector` dispatches at M≥6 and exactly **16** at M≤5; **zero**
  `sdpa_vector_2pass` and **zero** full-attention SDPA at every M ∈ [1,9]
- **N6** exactly **96** scalar-`asType` copy dispatches per forward at every M (two `invScale`
  constants per GDN layer) — pure waste, hoistable to init
- **N7** **zero** `qmm_splitk` / steel-gemm dispatches at every M ∈ [1,9], iff the device qmv batch
  limit is ≥10; at limit 6 the M≥6 total becomes **1353**
- **N8** exactly **2** top-two reducer dispatches per forward, independent of M and of the vocabulary

## The program can fail

`--verify-citations` re-reads all 47 citations in the live tree and exits 1 if any cited line no
longer says what the inventory claims. It genuinely fired during development: five citations had
drifted (two GDN rows citing the M==2 mid-branch while describing M==1 `processChunk`, the KV-cache
writes, and both envelope rows), and I fixed the inventory rather than the checker. Rows whose
prework is textually duplicated across the M==1 and M==2 branches now carry **both** citations, so
neither site can drift unnoticed.

`--selftest` runs 38 checks, four of which inject deliberate drift (bad line number, drifted text,
unreadable file) and assert the checker returns non-zero. A citation checker that cannot fail proves
nothing.

## Limitations — read before using any number here

1. **This is a static count, not a measurement.** Dispatch count is not latency. A 1096-dispatch
   forward can easily be slower than a 1544-dispatch one if the kernels are bigger. Nothing here
   claims otherwise.
2. **Layout copies are a band, not a point.** Eleven rows are `DERIVED` from stride/contiguity
   arguments I performed by hand against cited MLX rules (`copy.cpp:216-232`,
   `slicing.cpp:14-42`). `--layout-copies zero` gives the lower bound. MLX's lazy graph may also
   fuse or elide some of these; I could not verify that without running.
3. **The deferred rollback tape is excluded.** On the M≥3 mixer path the `convInput` concat at
   `Q35:791` feeds only `PrefixReplayTape`, so under lazy evaluation it is not an ancestor of the
   forward's outputs. Those 96 dispatches are charged to a rollback, not to this forward. If the
   graph is evaluated more eagerly than I assume, add 96 to every M≥3 total.
4. **`vector_limit` is device-dependent.** N7 and the whole `qmv_limit` column rest on
   `get_qmv_batch_limit` returning 10 rather than 6. I read the source; I did not query the device.
5. **KVCache growth-branch frequency is assumed.** The `+4` growth-branch launches
   (`KVCache.swift:402, 413-425`) are excluded from the steady-state count; they fire roughly once
   per `256/M` forwards.
6. **Compiled ops are assumed to be one launch each.** `qwen35Compiled*` and `compiledSilu` are
   counted as 1. If MLX's compile splits any of them, those rows undercount. Relatedly, `compile`
   retraces on shape change (`Transforms+Compile.swift:154`), so a width the session has not seen
   before pays a trace cost this inventory does not model.
7. **One host, one base.** All line numbers are pinned to
   `c0f7e370921a14f348fa1872f2176b1b43028752`.

## Suggested follow-ups (not implemented)

- **N6 is a free 96-dispatch cut at every M.** The two `invScale` scalars per GDN layer are
  input-independent constants recomputed and re-`asType`'d every forward at `Q35:811` (and `:756`).
  Hoisting them to layer init removes 96 tiny dispatches per verify forward, ~8.8 % of the M≥3 total,
  with no numerical change. This is the single cheapest item the inventory found.
- **M=2 is the worst operating point in the table and should probably not be scheduled.** If the
  scheduler ever chooses M=2, M=3 is strictly cheaper in dispatches *and* proposes an extra token.
  Whether that survives a latency measurement is exactly the experiment to run.
- **Capture the missing histogram.** A 512-token shipped-default verify-width histogram would turn
  both weightings above from proxies into evidence, and it is cheap.
- **Profile-verify N4/N5/N8 first.** They are the three cheapest claims to check with a single
  capture and they discriminate the branch structure the rest of the table is built on.
