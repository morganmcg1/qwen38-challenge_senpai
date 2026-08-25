# E206 — cap-8 width-work map

Desk and census work. Zero GPU legs. Zero timing contrasts between caps,
surfaces, or policies. Every number is labelled `harness=local` or
`harness=ranked`, and every ranked number that leaves the measured anchors is
labelled EXTRAPOLATED.

- Assignment: PR #204, revision `e206-r0`.
- Campaign base: `40be5063a136725f0ce73ed81b2f8deac5ce184d`.
- Scope: `research/` only. Unfiltered `editablePaths` diff against the base is
  empty, so the scored surface is byte-identical (RULE 393).
- `senpai/verify-ranked-score-boundary.sh`: PASS.
- `senpai/check-editable-budget.sh`: OK, growth 0 / 262144.
- Reproduce: `python3 research/e206_width_map.py`.
- Artifact: `research/e206-width-work-map.json`.

---

## Deliverable 1 — RULE 395 censuses for both surfaces

### 1a. No census leg was needed

Thorfinn's re-gate run carries the full per-round trace. W&B `7y6ap5l8` logs a
`per_round` media table with columns `[round, drafts, rows,
block_request_seconds]`, 76 rows. The sibling run `exdb3vyt` in the same group
carries a **bit-identical** width sequence, so the census is replicated across
two independent gate-qualified legs.

**The cap-8 surface is exact, not a reconstruction.** `b9ee228c` has a
**zero-byte** scored-surface diff against the campaign base `40be5063` under the
full `editablePaths` filter, and the frozen candidate `c47c7284` is `b9ee228c`
plus the single line `segmentedVerifyDepthCap = 8`. The census therefore
describes `current base + the one submitted line` directly. RULE 386
reconstruction labelling does not apply.

### 1b. Cap-8 census (the surface `aff4ad64` would promote)

`harness=local`. Source: W&B `7y6ap5l8`, replicated by `exdb3vyt`.
512 decode tokens, `--local-submit`, one public fixture, Apple M4 Pro.
`cool_gate_passed_real_gate=true`, `gate_qualified_for_timing=true`,
`all_tokens_matched=true`, `residual_divergence_count=0`, ledger closes
76 + 436 + 85 = 597 declared = 597 reference-checked rows.

| served width qL | 4 | 5 | 6 | 7 | 8 | **9** | total |
|---|---|---|---|---|---|---|---|
| rounds | 3 | 7 | 8 | 4 | 12 | **42** | 76 |
| share of all rounds | 3.95% | 9.21% | 10.53% | 5.26% | 15.79% | **55.26%** | 100% |

Acceptance context: 76 drafting rounds, 0 non-drafting rounds, 521 drafts
proposed, 436 accepted, 85 rejected. Accepted-draft rate **0.83685**. Mean
accepted draft length **5.7368**. EDL (proposed drafts per round) **6.8553**.
Tokens per round 6.7368.

### 1c. Cap-7 census (current shipped cap)

`harness=local`. Source: E202, W&B `lpwbno36`, raw per-round traces at
`research/out/e202/session-submit-abba1/*/trace.txt`, 14 legs × 512 tokens.
Ungated standing mode with flags preserved verbatim
(`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`);
counts do not depend on temperature. Every leg exact, ledger closed at 574/574.
The 14 legs are bit-deterministic, so each leg is one identical repeat.

The assignment quoted the split-branch frame (qL8 88.24%). That frame counts
only the 68 rounds per leg that take the split-cell branch. For comparison with
the cap-8 table I re-emit the census in the **all-rounds** frame, where the
served width is `qL = drafts + 1` on every round including the 10 rounds per leg
that take the non-split branch:

| served width qL | 2 | 4 | 5 | 6 | 7 | **8** | 9 | total |
|---|---|---|---|---|---|---|---|---|
| rounds per leg | 1 | 4 | 5 | 5 | 3 | **60** | 0 | 78 |
| rounds, 14 legs | 14 | 56 | 70 | 70 | 42 | **840** | 0 | 1092 |
| share of all rounds | 1.28% | 5.13% | 6.41% | 6.41% | 3.85% | **76.92%** | **0%** | 100% |

Split-branch frame, for continuity with FINDING 531: qL6 7.35%, qL7 4.41%,
qL8 **88.24%**, qL9 **0%**. Both frames are correct; they differ only in the
denominator.

Acceptance context: 1092 rounds, 0 non-drafting rounds, 6944 drafts proposed,
6090 accepted, 854 rejected. Accepted-draft rate **0.87702**. Mean accepted
draft length **5.5769**. EDL **6.3590**. Tokens per round 6.5769.

Per-width acceptance, cap-7 (14 legs):

| served width | rounds | proposed | accepted | rejected | accept rate | full-accept rounds |
|---|---|---|---|---|---|---|
| qL8 | 840 | 5880 | 5152 | 728 | 0.8762 | 672 |
| qL7 | 42 | 252 | 252 | 0 | 1.0000 | 42 |
| qL6 | 70 | 350 | 294 | 56 | 0.8400 | 42 |
| qL5 | 70 | 280 | 238 | 42 | 0.8500 | 56 |
| qL4 | 56 | 168 | 140 | 28 | 0.8333 | 28 |
| qL2 | 14 | 14 | 14 | 0 | 1.0000 | 14 |

Warm-up is excluded by construction, not by trimming. The one uniform
32-per-width warm-up pass (`6:32|7:32|8:32|9:32`) runs before the first timed
round and emits no round record. Arithmetic closure: the final cumulative
witness `6:112|7:80|8:992|9:32` equals `32 + {80, 48, 960, 0}`. **Zero timed
rounds were dropped**, and qL9 on cap-7 was served only by warm-up.

### 1d. Honest RULE 395 caveat on the cap-7 table

**The cap-7 census is not on the current base.** E202 ran on `7f10e147`. A
project-wide W&B search found **no** 512-token cap-7 census on any base at or
after `b9ee228c`; the only current-base 512-token census in existence is the
cap-8 one above.

The single scored-surface change from `7f10e147` to `40be5063` is the E165 head
prefetch becoming unconditional (the `MLX_E165_HEAD_PREFETCH` arm switch was
deleted; E202 ran with it unset, so prefetch was OFF). INFERRED, not measured:
the prefetch relocates head work within the round (FINDING 528, relocation not
removal). It does not change the proposed tokens, the acceptance outcome, or the
EMA schedule that sets the depth, so the width **counts** should be invariant
even though the block latency is not. Corroboration: `cqfa2d6n` (E167, base
`770a3ff2`, gate-qualified, 512 tokens, cap-7) gives qL3 1, qL4 2, qL5 6, qL6 5,
qL7 7, qL8 56 over 77 rounds — the same qL8-dominant shape on a third base.

If the receipt REJECTS and a cap-7 width table must be priced, one 512-token
census leg on `40be5063` closes this gap. It was outside this assignment's
budget and is not needed for the cap-8 branch of the map.

---

## Deliverable 2 — 9-row dispatch enumeration

All citations are `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
unless another file is named. PROVEN-FROM-SOURCE on the current tip.

### 2a. HEADLINE — the shipped E195 plan does not cover m=9 for any family

The plan is not a table keyed at M=4…9. It is one predicate, **1641**:

```swift
guard m == 6, cell != .mlpDown, cell != .unlisted else { return .staged }
```

**E195's shipped selectivity is m=6 only.** The witness string is
`"selective-m6"` (1647). The manifest note "publishes at M=4…9 with the IPG
table and tablePays≥4" describes the older **E120 sumtable arm**
(`minimumTableWidth = 4`, 2082-2084), not E195 coverage.

The IPG tables (**1582-1584**):

| variant | pairs `(m, IPG)` |
|---|---|
| `.staged` | (2,2) (3,3) (4,4) (5,5) (6,3) (7,4) **(8,4) (9,3)** |
| `.singlePass` | (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) **(9,3)** |

Both tables end at `(9, 3)`. **No IPG=9 kernel instantiation exists anywhere in
the tree**, not even as an unshipped variant. `G(m) = ceil(m / IPG)`
(1562-1565), so `G(9) = 3` under either variant.

Per-family answer at m=9. The routing decision is `(m, cell)` and cell identity
is `(k, n)` only (1591-1616), so the map is a compile-time property of the
checkpoint:

| family | (k, n) | invocations/round | m=9 path | IPG | G |
|---|---|---|---|---|---|
| mlp.gate_up | (5120, 34816) | 64 | `.staged` | 3 | **3** |
| mlp.down | (17408, 5120) | 64 | `.staged` (excluded even at m=6) | 3 | **3** |
| gdn.in_proj | (5120, 16480) | 48 | `.staged` | 3 | **3** |
| gdn.out_proj | (6144, 5120) | 48 | `.staged` (shares the cell with fa.o_proj) | 3 | **3** |
| fa.qkv | (5120, 14336) | 16 | `.staged` | 3 | **3** |
| fa.o_proj | (6144, 5120) | 16 | `.staged` | 3 | **3** |
| lm_head (readout) | (5120, 248320) | 1 | `.staged`, via `routedLMHead` (6176-6178) | 3 | **3** |

m=9 is a **planned** shape, not a fallback and not unplanned: `(9,3)` is an
explicit table entry, `preconditionFailure` at 2035 is unreachable, and the
kernel switch never falls through. It is planned at the **worst pass count in
the whole 2…9 band**.

**The coverage-gap cell list the assignment asked for is therefore all seven
cells, 257 matvecs per round.**

Full routing chain at m=9, for the record: call site → `qwen35RoutedQuantizedMM`
(2257) → `Qwen35CustomQMV.matmul` (2189); `arm = .sumTable` (2014-2018); `arm !=
.off` (2199); `routable` passes for every cell at m ∈ 2…9 (2094-2121, widths
2025); `tablePays(9)` true (2082-2084) → table branch (2206); sidecar lookup
(2214, 2940); `kernelVariant` (2051) → `Qwen35QMVCell.identify` (1607-1616) →
`qwen35QMVVariant(9, cell) = .staged` (1641); `inputsPerGroup = 3` (2033-2038);
`activeInputGroups = 3` (2044-2048); JIT kernel
`qwen35_custom_affine4_g64_qmv_wide_sums_v1` (1847), grid `(3*32, (n/8)*2, 1)`,
threadgroup `(32, 2, 1)` (2170-2183); kernel `case 9` →
`qwen_e120_qmv_m<9, 3, USE_TABLE>` (1660-1668); `qmv_stride = 16` at m=9 (1675).
MLX is never consulted on this path. Even the MLX fallback would stay in the
qmv band: `get_qmv_batch_limit` returns 10 for every scored `(K,N)` on both
gen-16 and gen-17 (`Vendor/mlx-swift/.../quantized.cpp:84-124`), and 9 < 10.

### 2b. The immediate composition opportunity, with the exact cell list

The kernel template already supports an unbalanced last group
(**1543-1557**):

```metal
static_assert(M % IPG != 1, "a one-input tail group is not built");
constexpr int TAIL = M % IPG;
const int first_m = group_x * IPG;
if (first_m >= M) { return; }
if (TAIL == 0 || M - first_m >= IPG) { qwen_e120_qmv_wide<IPG, USE_TABLE>(...); }
else { qwen_e120_qmv_wide<(TAIL >= 2 ? TAIL : 2), USE_TABLE>(...); }
```

So the only forbidden pairing at m=9 is a one-row tail. Options:

| IPG at m=9 | G | tail rows | legal | register regime |
|---|---|---|---|---|
| 3 (shipped) | 3 | 0 | yes | proven |
| 4 | 3 | 1 | **no** — `static_assert` | — |
| **5** | **2** | **4** | **yes** | **IPG=5 already ships and wins at m=5** |
| 6 | 2 | 3 | yes | IPG=6 ships at m=6 (E195) |
| 7 | 2 | 2 | yes | IPG=7 lost 23.62 ms/round at m=7 |
| 8 | 2 | 1 | **no** — `static_assert` | — |
| 9 | 1 | 0 | yes | deepest into the losing regime |

`(9, 3) → (9, 5)` is a **single table entry at 1582**. It removes one of three
weight passes without leaving the IPG≤5 staged regime. It is **not** the E195
single-pass mechanism: the source note at 1626-1633 attributes the m=7 and m=8
losses to `IPG = m` leaving the weight-streaming regime and becoming
register/occupancy bound. `(9,5)` does not do that; it re-groups inside the
regime E120 shipped.

The existing plan is balanced rather than pass-minimal: 6→3+3, 7→4+3, 8→4+4,
9→3+3+3. `(9,5)` trades balance (5+4) for one fewer pass.

Ranked target list for a selective-m9 attack, ordered by the measured local cost
of the third pass (Deliverable 3 for provenance):

1. `mlp.gate_up` (5120, 34816) ×64 — 17.580 ms, 40.6%
2. `mlp.down` (17408, 5120) ×64 — 9.129 ms, 21.1%
3. `gdn.in_proj` (5120, 16480) ×48 — 8.652 ms, 20.0%
4. `fa.qkv` (5120, 14336) ×16 — 3.037 ms, 7.0%
5. `gdn.out_proj` (6144, 5120) ×48 — 2.205 ms, 5.1%
6. `lm_head` (5120, 248320) ×1 — 1.943 ms, 4.5%
7. `fa.o_proj` (6144, 5120) ×16 — 0.755 ms, 1.7%

The top three carry **81.7%**. E195's own precedent says selectivity matters:
holding `mlp.down` staged at m=6 nearly doubled the end-to-end gain
(7.31 vs 3.81 ms/round), so a per-cell m=9 ladder is the right instrument, not
an all-cells switch.

### 2c. x-sums fill dispatches, m=9 vs m=8

The publisher predicate `wants` (2916-2926) is widths 2…9 ∧ `tablePays(m ≥ 4)` ∧
`k % 512 == 0`. **Width never enters beyond `tablePays`, so m=9 publishes exactly
like m=8.**

Counts per round are **identical at m=8 and m=9: 130 standalone fills + 127
sidecar hits** (source note 2886-2891). Structure: sidecar hits = 64
`mlp.gate_up` + 47 `gdn.in_proj` + 16 `fa.qkv`; standalone = 64 `mlp.down` + 64
`out_proj` family + 1 `lm_head` + 1 layer-0 entry (the entry norm fuses only
when `delta != nil`, 4054-4060, and layer 0 gets `delta = nil`, 4243-4246).

The only m=9 deltas are size, not count: `sumsStride(m) = m <= 8 ? 8 : 16`
(1944, 2028), so the table doubles from 10,240 B to 20,480 B at K=5120, and the
fill grid z goes 8 → 9 (2129-2143). The fused-norm epilogue already handles the
stride-16 case (2542) and `take` verifies the size (2947-2951).

### 2d. SDPA / split-cell path at qL=9

At qL=9 there is **no single SDPA call**. `attentionWithCacheUpdate`
(`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:122-141`)
splits the nine query rows into two fused `sdpa_vector` calls of qL=5 and qL=4:

```swift
if queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL, case .causal = mask {
    let split = 5
    let kSplit = kL - (qL - split)
```

The split is what keeps both chunks on the fused kernel. In
`scaled_dot_product_attention.cpp:631-639`, `supports_sdpa_vector` requires
`query_sequence_length * gqa_factor <= 32`; with 24 Q and 4 KV heads the GQA
factor is 6, so the binding constraint is **qL ≤ 5**, not the nominal `qL <= 8`.
An unsplit qL=9 call would fall back to the unfused matmul lambda. Chunk A
(5×6 = 30) and chunk B (4×6 = 24) both stay under 32. `sdpa_full` is unreachable
at every width on this model because it requires head_dim ∈ {64, 80, 128} and
this model's head_dim is 256.

**Nothing on the SDPA route changes at 8 → 9.** Chunk A stays 5 rows, only chunk
B grows 3 → 4 and chunk A's key window moves kL−3 → kL−4. The `use_fallback`
gate, the vector-mode gate (`q_pre.shape(2) <= 8`, both chunks qualify), the
1-pass grid, the 2-pass threadgroup `(32, gqa, qL)`, the slice behaviour, and
the `concatenated` epilogue are all unchanged. The 1-pass vs 2-pass route
(`:746-753`) is driven by kL, not qL, and both chunks cross together.

Two genuine 8 → 9 boundaries exist, both **outside** SDPA:

1. `qwen35GatedDeltaReplayState` guard `k.dim(1) <= 8` (**631**). At T=9 the
   guard fails and the function returns `nil` (**636**), dropping to the
   vendored two-output kernel. This touches the 48 GDN layers on a partial-accept
   replay. It is a live m=9-only behaviour change worth a separate question.
2. The QMV lane stride and x-sums table double at m=9 (2c above).

Do not re-propose the `widthSixWall` two-append fix at qL=9. It is **withdrawn**:
`KVCacheSimple` returns axis-2 slices by construction, the shipped split path
issues zero KV-sized copies, and the candidate fix manufactures a whole-KV copy
twice, losing 1.636–2.300 ms/round (FINDING 496). FINDING 497 also reprices
FINDING 489: the +87.7 / +173.3 µs figures measured the unsplit fallback arm.
The mechanism is width-independent, so it is no more applicable at 9 than at 8.

### 2e. M4 vs ranked M5 `_nax` divergence

- **QMV replica path: no divergence.** There is no device or architecture
  predicate anywhere from `Qwen35CustomQMV.matmul` to the kernel dispatch; every
  switch is an environment variable read at process start (2014, 2903, 1833).
  The m=9 enumeration above is device-independent and **holds on the ranked M5**.
- **SDPA route: no `_nax` divergence.** The only `_nax` attention decision is
  `scaled_dot_product_attention.cpp:177-179`, inside
  `sdpa_full_self_attention_metal`, which is unreachable on this model
  (head_dim 256). `sdpa_vector.h` contains no `_nax` symbols.
- **`get_qmv_batch_limit`** (`quantized.cpp:84-124`) branches on `arch_gen`.
  Gen 16 (M4) and gen 17 (M5) both take the else branch → limit 10 for every
  scored `(K, N)`. The gen-17 value rests on the on-device census recorded in
  `frontier-state.json` (`quantizedMMRouting`, FINDING 390/450), **not** on M4.
  Label any claim that depends on it as census-backed, not source-backed.
- **One live transfer risk, UNVERIFIED:** `char devc = d.get_architecture().back()`
  at `scaled_dot_product_attention.cpp:446` and `:747` selects the 1-pass vs
  2-pass SDPA route and the 2-pass block ladder. The local host is
  `applegpu_g16s` → `'s'`. **The M5 architecture string is not recorded anywhere
  in this checkout.** If M5's last character is neither `'s'` nor `'d'`, M5 stays
  on 1-pass `sdpa_vector` at kL ≥ 1024 while M4 uses 2-pass. Smallest resolving
  read: capture `d.get_architecture()` on the ranked host or find it in a prior
  runner log.
- **All performance numbers behind the shipped plan are M4-local** (the E120
  rung-5d table at 2060-2074 says so explicitly). The m=9 *dispatch* facts
  transfer; the m=9 *cost* has never been measured on M5.

---

## Deliverable 3 — G=3 step exposure map

Desk only. No new timing. `harness=local` unless marked.

### 3a. The step and its provenance

FINDING 484 (E182, W&B `dhi54lrs`, `research/e182-report.json`):
`R_local = 18.92 + 6.51·m + 36.03·G ms`. Fit coefficients a = 18923.6 µs,
b = 6513.5 µs, s = 36026.6 µs; step form preferred over smooth on every GPU
band (`step_rmse` 2864 µs vs `smooth_rmse` 6577 µs; AICc 161.3 vs 176.2) and on
no host band. Limitations recorded with it: full acceptance only (no rejected-row
cost measured), 256-token window, ungated ABBA, M4 Pro.

The measured curve matters more than the fit. E182 unperturbed legs:

| m | round ms | increment |
|---|---|---|
| 8 | 145.774 | +8.413 |
| **9** | **185.826** | **+40.051** |

**The third pass costs +40.051 ms, 12.5% MORE than the second pass
(+35.596 ms).** The single `s = 36.03` coefficient averages them, so the law
understates the m=9 boundary.

### 3b. Independent consistency check at m=9 from existing data

Not a contrast — one within-run comparison of the census leg's own per-round
latencies against the law. `harness=local`, W&B `7y6ap5l8`:

| qL | rounds | measured block ms | law at uniform staged G | G |
|---|---|---|---|---|
| 4 | 3 | 77.45 | 80.99 | 1 |
| 5 | 7 | 109.86 | 87.50 | 1 |
| 6 | 8 | 118.26 | 130.04 | 2 |
| 7 | 4 | 136.31 | 136.55 | 2 |
| 8 | 12 | 145.21 | 143.06 | 2 |
| **9** | **42** | **186.01** | **185.60** | **3** |

The law reproduces the m=9 cell to **0.2%** on the real schedule, and m=7 and
m=8 to under 1.5%. Two honest caveats: m=6 is expected low because E195 makes
six of seven cells G=1 there, and the low widths (n = 3 and 7) sit above the law
because a narrow round follows a rejection and pays repair work that the law
does not model. This is evidence that the G-step object is real at m=9 on a live
schedule; it is **not** a depth-price or policy contrast (RULE 79).

### 3c. Where the third pass lands

Two instruments, in-path and isolated, agree on the family split.

E182 in-path bands, m=8 → m=9:

| band | step ms | share |
|---|---|---|
| gdn_mlp | +19.305 | 47.8% |
| gdn_mixer | +8.359 | 20.7% |
| fa_mlp | +6.431 | 15.9% |
| fa_mixer | +2.648 | 6.6% |
| residual (lm_head + host) | +3.637 | 9.0% |
| round total | **+40.380** | 100% |

E186 isolated layer probes (W&B `fm5fnfmd`, table `e186/fits`, `step_m9_us`):

| family | ms/round | share |
|---|---|---|
| MLP | 25.755 | 64.7% |
| GDN | 9.167 | 23.0% |
| FA | 2.961 | 7.4% |
| lm_head | 1.943 | 4.9% |
| total | **39.827** | 100% |

Cross-instrument agreement on MLP is **0.07%** (25.736 vs 25.755 ms).

Per-cell ranking is in Deliverable 2b. Recorded caveat, preserved: the isolated
per-cell sum is 43.301 ms against the layer-probe reconstruction 39.827 ms, so
it **over-explains by 8.7%** and ranks cells rather than pricing them. FINDING
488's harsher caveat also stands: isolated families over-explain the *ranked*
width term 3.5–4.6×.

### 3d. Weight bytes predict the split

affine 4-bit group 64 → 0.5 + 4/64 = **0.5625 B/element**. One full weight pass
over the seven fused cells:

| cell | K×N | bytes/invocation | inv | GB/round | share |
|---|---|---|---|---|---|
| mlp.gate_up | 5120×34816 | 100,270,080 | 64 | 6.4173 | 44.53% |
| mlp.down | 17408×5120 | 50,135,040 | 64 | 3.2086 | 22.26% |
| gdn.in_proj | 5120×16480 | 47,462,400 | 48 | 2.2782 | 15.81% |
| gdn.out_proj | 6144×5120 | 17,694,720 | 48 | 0.8493 | 5.89% |
| fa.qkv | 5120×14336 | 41,287,680 | 16 | 0.6606 | 4.58% |
| fa.o_proj | 6144×5120 | 17,694,720 | 16 | 0.2831 | 1.96% |
| lm_head | 5120×248320 | 715,161,600 | 1 | 0.7152 | 4.96% |
| **total** | | | | **14.4123** | 100% |

This reproduces FINDING 248 exactly. Family shares by bytes (MLP 66.8%,
GDN 21.7%, FA 6.5%, lm_head 5.0%) match the measured G=3 step split to about
1 pp on every family. FA's excess at m=8 (11.2% measured vs 6.5% bytes) is the
`widthSixWall` SDPA effect, which is flat from m=6 to m=9, so **FA's excess does
not repeat at the G=3 boundary**.

Implied rate for the third pass: 39.827 ms / 14.4123 GB = **2.76 ms/GB**, at the
ceiling of the in-situ per-family rates recorded at M=6 (2.995–5.263 ms/GB). The
third pass is quantitatively "one more full 14.4 GB weight stream".

Worst rate against its byte share: `gdn.in_proj`, 3.80 ms/GB on 15.81% of the
bytes.

### 3e. Local-to-ranked exposure — the assignment's premise needs correction

**FINDING 484's "ranked prices the whole 5→6 crossing at 5.4 ms, so ≥2/3 of the
local step is hidden on M5" is stale.** That 5.4 ms is
`research/e182-ranked_fold.json`, `models.quadratic.r_by_width_ms`
(R(5) 40.338, R(6) 45.732), i.e. the FINDING 456 quadratic that **FINDING 504
falsified at 5.60σ**. The same falsified law produced the 8.848 ms width-9
marginal row.

Measured replacement, `harness=ranked`: FINDING 505, ranked
**dR6 = +8.940 ms**, 1σ [8.345, 9.544], from single-cell bisection off the paid
cap-4 anchor. Against the E186 local dR6 of 37.6726 ms the transfer coefficient
is **k = 0.23731**, 1σ [0.22151, 0.25334] — this is exactly the
`transfer_ratio` E197 stores. The E197 smooth-step refit gives dR6 7.93–8.26 ms,
k = 0.2105–0.2193.

So roughly **76% of a local weight pass is hidden on M5**, and the realized
share is about **54% of the naively R(1)-scaled step, not 33%**. The direction
of the caveat is unchanged; the magnitude is not. Attach this to every number
above.

### 3f. Round-weighted exposure and its ranked price

`harness=local`, MEASURED: third pass 40.051 ms per m=9 round; cap-8 m=9 duty
42/76 = 55.26% → **22.13 ms per average round**.

`harness=ranked`, all EXTRAPOLATED (ranked dR9 has never been measured; a cap-7
round verifies at most 8 rows). Using the E201 / FINDING 523 linearized score
engine (0.019745 published points per ms of transferred dR9 excess; flat cap-8
chain 3.829386):

| scenario | pass removed per m=9 round | round-weighted | cap-8 with third pass | cap-8 without | Δ published |
|---|---|---|---|---|---|
| (a) measured F505 transfer, k = 0.23731 on the 36.03 ms pass term | 8.550 ms | 4.725 ms | 3.6729 | **3.8417** | **+0.1688** |
| (b) E197 refit band, k = 0.2105–0.2193 | 7.584–7.900 ms | 4.191–4.366 ms | 3.6857–3.6919 | 3.8417 | +0.1498 to +0.1560 |
| (c) most pessimistic admissible: the smooth-step law's own dR9 = 1.962 ms total leaves ≤ 0.623 ms of pass component | 0.623 ms | 0.344 ms | 3.8294 | 3.8417 | +0.0123 |

Hard ceiling on (a) if the whole measured 40.051 ms step transfers at the 1σ
upper coefficient: 10.15 ms per m=9 round.

**Reading:** under the measured transfer the m=9 third pass is worth about
**+0.15 to +0.17 published points** on a promoted cap-8 surface, which clears
the crown 3.72911 by a wide margin. Under the smooth-step continuation it is
worth almost nothing. The spread is irreducible on the desk. The in-flight
`aff4ad64` receipt measures dR9 directly and collapses it.

Do not census-scale this to ≈3.985. The local fixture's 55% m=9 duty is roughly
twice the ranked extrapolated tail mass across the eight hidden prompts, and
per-prompt ranked weighting is required.

### 3g. Named risks that can zero or invert the prize

1. **Register/occupancy cliff.** FINDING 500 / E195: an all-cells single-pass
   switch cost +53.96 ms/round at m=8 and +23.62 ms/round at m=7. A G-reduction
   at m=9 risks the same regime exit. `(9,5)` is chosen precisely to stay at
   IPG≤5, but that is an argument, not a measurement.
2. **The 5+4 split is unmeasured end to end.** Only directional isolated
   evidence exists (E73 rung-1 `mlp.down`: m9 IPG5 ≈ 502 µs vs m9 IPG3 ≈ 579 µs,
   −13%), and FINDING 501/517 prove isolated probe wins can invert end to end.
3. **Ranked dR9 is receipt-only knowledge** (FINDING 514).
4. **RULE 79 / FINDING 521.** No local timing leg may decide this; local and
   ranked cost tables gave the same policy opposite signs.
5. **Transfer extrapolation risk.** FINDING 504 killed the last confidently
   extrapolated law at 5.60σ; FINDING 514's leave-one-receipt-out shows no
   family predicts a held-out paid median within the channel.

---

## Deliverable 4 — the conditional width-priority map

### Branch A — `aff4ad64` PROMOTES (cap-8 surface)

| rank | work | scored path | measured cost behind it | exposure caveat |
|---|---|---|---|---|
| 1 | **selective-m9 IPG re-grouping**, `(9,3) → (9,5)` at 1582, applied per cell as a ladder | all 7 fused QMV cells, 257 matvecs/round, on 55.3% of rounds | local third pass 40.051 ms/m9-round; 22.13 ms round-weighted | ranked +0.15 to +0.17 published (a/b) or +0.01 (c); EXTRAPOLATED |
| 2 | the same, restricted to the top three cells `mlp.gate_up`, `mlp.down`, `gdn.in_proj` | 176 matvecs/round | 35.36 of 43.30 ms = 81.7% of the third pass | same, minus 18.3%; E195 precedent says selectivity can nearly double the end-to-end gain |
| 3 | **GDN replay at T=9**: the `k.dim(1) <= 8` guard at 631 fails at nine rows and drops to the vendored two-output kernel | 48 GDN layers, partial-accept rounds only | not yet priced; 85 rejected drafts over 76 rounds means partial accepts are common on cap-8 | unknown; needs a census of partial-accept rounds at qL=9 first |
| 4 | `gdn.in_proj` as a cost outlier — 3.80 ms/GB, the worst rate of the seven against its byte share | 48 invocations/round | 8.652 ms of third pass on 15.81% of bytes | a per-cell question, not a width question |
| — | **do not** re-attack SDPA at qL=9 | — | the path is width-flat 8→9; `widthSixWall` M1 is withdrawn (FINDING 496); the fused row-amortized kernel returned zero (FINDING 522) | — |
| — | **do not** attempt IPG=9 single-pass | — | it is the m=7/8 losing mechanism, one step deeper | — |

Width tables that die under RULE 395 if cap-8 promotes:

- My own E202 recommendations "delete qL=9 work" and "concentrate on qL=8"
  **invert**. qL=9 goes from 0% to 55.3% of rounds; qL=8 falls from 76.9% to
  15.8%.
- FINDING 522(b)'s "optimizing the widest legal shape is worth nothing" is
  cap-7-only and must not be carried across.
- Any kernel priority derived from the cap-7 census (qL8-dominant) is void.
- The cap-7 census itself becomes historical. The cap-8 census in Deliverable 1b
  is already on the promoted surface, so no new leg is needed to replace it.
- E197's ranked law gains its first real m=9 anchor; every dR9 number in §3f
  should be replaced by the receipt's inverted value.

### Branch B — `aff4ad64` REJECTS (cap-7 surface stands)

The qL8-concentration recommendation becomes operative, but the honest answer is
that **the qL=8 width axis is close to exhausted**:

| axis at qL=8 | status |
|---|---|
| E198 dispatch fusion | dead — FINDING 529, `inner` = +7.451 ms/round, MLX already overlaps |
| single-pass QMV at m=7/8 | dead — +23.62 ms/round at m=7, +53.96 at m=8 |
| `widthSixWall` two-append | withdrawn — FINDING 496 |
| SDPA step removal | closed — FINDING 502; only "Route 1" remains open |
| `eval()` / sync restructuring | forbidden — FINDING 530, 419 µs/call is ~12× MUE |

What remains worth doing at qL=8:

1. **Re-derive the cap-7 census on `40be5063` first.** RULE 395 is not satisfied
   by E202's `7f10e147` table. One 512-token leg. Do this before pricing any
   qL=8 work, not after.
2. **`mlp.gate_up` at m=8 remains the largest single cell** (44.5% of pass
   bytes, G=2). The G axis is closed there, but the cell is the right target for
   any non-G mechanism.
3. **Do not spend on qL=9.** On cap-7 it is served only by warm-up, exactly 32
   calls per width per process, and zero timed rounds.
4. The selective-m9 finding does not die; it becomes conditional stock. It
   reopens the moment any cap-8 candidate is reconsidered.

### Decision rule

The map's branch selector is the receipt, not a local measurement. Under RULE 79
no local leg may substitute for it.

---

## What I did not do

- No mechanism was implemented. `(9,5)` is a proposal with a cell list and a
  price, not a change. It needs its own assignment, an exactness gate with real
  floating-point values at the touched cells plus a positive control, and an
  end-to-end ladder rather than an isolated probe.
- No cap-7 census leg on the current base. It is the one gap in Deliverable 1
  and it only matters on Branch B.
- No timing contrast of any kind.

## Suggested follow-ups

1. **E207 candidate: the m=9 IPG ladder.** `(9,3)` → `(9,5)` as a per-cell
   selective plan, gated for exactness, screened as an end-to-end ladder over
   the seven cells in the Deliverable 2b order. Conditional on promotion.
2. **Census the partial-accept rounds at qL=9** before pricing the GDN replay
   guard at 631. The cap-8 `per_round` table has drafts and rows but not the
   accepted count per round, so this needs one traced leg or a richer logger.
3. **Capture `d.get_architecture()` on the ranked M5 host.** It is the one
   unresolved M4/M5 divergence on the attention route and it is a log line, not
   an experiment.
4. **Log the accepted count per round in the standard `per_round` table.** The
   cap-8 census could not produce per-width acceptance because that column does
   not exist. The cap-7 raw trace has it; the W&B table does not.
5. **The E202 raw traces are untracked** and exist only on this host's disk. A
   reduced per-round CSV should be committed if that evidence must survive.
