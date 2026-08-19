# E33 — row-blocked single-pass M=6: the weight pass halved and it cost us 1.5 %

**Verdict: NEGATIVE. Hypothesis falsified. Kill criterion fired.**

`e33/m6_per_row_cost_ratio = 1.0150` against a registered prediction of 0.85 (mine)
and 0.82 (advisor's). The candidate is **1.50 % slower** at M=6, not 15–18 % faster.
The mechanism engaged exactly as designed — M=6 really did become one weight pass —
and it did not pay for itself.

Rung 2 (M=7/8/9) is **not** started, per the assignment's staging rule and because
the per-shape scaling law below predicts the same structural outcome there.

- Assignment: `qwen38-r1-e33-rowblocked-single-pass-m6`, revision `r1`, PR #38
- Base: `senpai/qwen38-mtp-r1` @ `4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549`
- Candidate: `20bdd2595bb962013f18ca4882cb21a4eed6b695`
- Pre-registration: `16c2a0d`, committed **before any code existed**
- Host: Apple M4 Pro `Mac16,11`, 20 GPU cores, 48 GiB, macOS 26.5.2 (25F84),
  Swift 6.3.3, `applegpu_g16s` (arch_gen 16, tier `'s'`)

---

## 1. Pre-declared controls first — 17/17, before any new number

Per the standard set on askeladd's E36 report: every control I registered in
`16c2a0d` or inherited from E27/E32, reported before the headline.

| # | control | expected | measured | verdict |
|---|---|---|---|---|
| 1–6 | untreated widths M ∈ {3,4,5,7,8,9}, cost ratio | \|r−1\| ≤ 0.01 | max **0.0046** (M=9) | **6/6 pass** |
| 7 | AIR `crossrow_m6_ipg3` (E27 shipped cell) | 83 | 83 | pass |
| 8 | AIR `crossrow_m5_ipg5` | 125 | 125 | pass |
| 9 | AIR `crossrow_m7_ipg4` | 108 | 108 | pass |
| 10 | AIR `crossrow_m8_ipg4` | 104 | 104 | pass |
| 11 | AIR `crossrow_m9_ipg5` | 129 | 129 | pass |
| 12 | AIR `crossrow_m4_ipg4` | 104 | 104 | pass |
| 13 | AIR `crossrow_m4_ipg2` | 62 | 62 | pass |
| 14 | production cell `crossrow_m6_ipg6_r2` registers | **117** | **117** | pass |
| 15 | QMV parity grid, all cells | bit-identical | **192/192, 0 differing** | pass |
| 16 | dispatch actually reached at M=6 | qmv, row-blocked | 8/8 shapes, **0 qmm** | pass |
| 17 | cross-session anchor: base M=6 vs E27's published | ≈ equal | 128.843 vs **128.865 ms** (0.017 %) | pass |

Plus row0 bit-exactness against each arm's own M=1: **8/8 shapes at every width
M=1..9 in both arms, max abs delta 0** (18 arm-widths).

Controls 7–13 re-anchor E27's `62/83/104/125` ladder and E32's tail-group cells
byte-for-byte, so the AIR pipeline underneath control 14 is the same instrument
that produced them.

---

## 2. The headline

**`e33/m6_per_row_cost_ratio = 1.0150`** (drift-adjusted 1.0147).

- Registered prediction: **0.85**, band [0.78, 0.92]. Advisor's: **0.82**.
- Registered kill criterion: **ratio ≥ 0.97 → stop, do not start rung 2.**
- Corrected kill criterion from the noise-floor addendum: **0.16 %**.
  The result is **+1.50 %, i.e. 9.4× the corrected single-run detection threshold,
  in the wrong direction.**

Control widths give the noise scale directly on this instrument: median drift
**1.0003**, range 0.9966..1.0046, **max |ratio−1| = 0.0046**. The M=6 effect is
**3.3× the largest control deviation**. It is not noise.

---

## 3. The mechanism did engage — this is not an implementation failure

Three independent readouts confirm the intended change happened:

1. **Dispatch readback**, candidate arm, all 8 scored shapes at M=6:
   `qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true, 2>`
   (base: `qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>`).
2. **Harness `stream_boundaries` moved `[6] → [7]`**: the harness's own weight-pass
   model agrees M=6 stopped being a two-pass width.
3. **Static AIR device loads per output tile**: weight-side **48 → 24** (the pass
   really halved), activation-side 24 → 48, total unchanged at 72.

So the experiment tested what it claimed to test. The hypothesis — that halving
weight traffic at M=6 buys 15–18 % — is what failed.

---

## 4. Assertion the advisor asked for: the timed cell is the cell that ran

`get_qmv_batch_limit` (`Vendor/.../backend/metal/quantized.cpp:84-126`) is consumed
at `:1415` and gated at `:1417` as `if (M >= vector_limit) { ...qmm... }`.

**Source side.** This box is `applegpu_g16s` ⇒ `arch_gen = 16`, `arch_size = 's'`.
That takes the `else` branch (not 13/14) and the `default:` case (not `'d'`), and all
eight scored shapes have `D, O ≥ 5120 > 4096`, so `vector_limit = 10`. `6 < 10`, so
M=6..9 all stay on `qmv`.

**Measured side, which is stronger.** Walking every string in both arms'
`vendored.json`:

```
e33-base-r1 : qmm-named readbacks anywhere = 0 ; 8/8 shapes at M=6 -> _m<T, 6, 3, true>
e33-cand-r1 : qmm-named readbacks anywhere = 0 ; 8/8 shapes at M=6 -> _m<T, 6, 6, true, 2>
```

Every scored shape at every timed width M=1..9 reports a `qmv_fast*` readback and
**not one `qmm*` readback in either arm**. The cell I timed is the cell that
executed, measured rather than inferred.

**The instrument already carried the answer.** Each entry in `vendored.json/shapes`
records `predicted_vector_limit`, and it reads **10 on all eight scored shapes** in
both arms, alongside `qmv_fast: true`. E27's curve harness had been logging exactly
the quantity askeladd escalated, in every arm this campaign has run. Worth noting for
next time: the escalation was still correct — nobody had looked.

Reproduce: `python3 research/e33_dispatch_reached.py`.

---

## 5. Correctness — the load-bearing section, given §0.6

The advisor's point 6 and askeladd's E36 finding 2 make this the most important part
of the submission: FP32 reassociation is licensed on the coarse draft path and
**forbidden on the verify path**, and the local fixture cannot detect a violation.

### 5.1 The distinction that matters: row visits vs the K partition

companygardener's `5c74b78b`/`6154a6f1` failed official parity by cutting
`values_per_thread` 16→8 and block 512→256. **`values_per_thread` *is* the lane→K
partition**: changing it re-partitions the reduction across lanes, so no setting
preserves the order, which is why even the register-cheapest setting still failed.

Row blocking is a different operation. It changes **which output rows a thread
visits**, and it does not touch the K partition at all:

```
block_size = values_per_thread(16) x SIMD_SIZE(32) = 512
```

`block_size` is a function of `values_per_thread` and `SIMD_SIZE` only. It is
**constant in both `NA` and `ROWS_PER_SIMD`**, so every thread walks the same
k-offsets in the same order in both arms. `ROWS_PER_SIMD` appears nowhere in the
k-loop bound, the k-stride, or the lane→k map.

### 5.2 Reassociation proof

Fix an output row `R` and an activation `a`:

1. **Same k-offsets, same order** — by §5.1, `block_size` is invariant.
2. **No cross-activation mixing** — `VF = vec<float, NA>` arithmetic is
   component-wise, and every write into a `VF` is by scalar index (`a0[m] = xc[0]`,
   `sums[m] += ...`). Widening `NA` 3→6 adds independent lanes; it does not touch any
   existing lane's expression tree.
3. **No cross-row reduction** — the `r` loops accumulate into separate registers
   (`acc[r]`, `partial[r]`) and are never reduced across `r`. `packed[r][i]` depends
   only on `row = R`, `xc[*]` only on `a`, `scale_local[r]`/`bias_local[r]` only on
   `R`. Changing `rows_per_simd` 4→2 only changes which `r` index a given row maps
   to.
4. **`simd_sum` pairing untouched** — the host grid is frozen at
   `group_dims(32,2,1)`, so lane membership is identical, and `simd_sum` now reduces
   a value proved identical by (1)–(3).

Output addressing: base writes activations {0,1,2} ∪ {3,4,5} × rows `out_row+[0,4)`;
candidate writes {0..5} × `out_row + 2b + [0,2)` for b ∈ {0,1}. Same 24 elements,
each written exactly once.

### 5.3 Measured: whole-tensor digest, not a token match

`research/run-qmv-parity.sh` digests the **entire output tensor** of one
deterministic `quantized_matmul` per (shape, width, bits) cell — 8 scored shapes ×
widths 1..12 × bits {3,4} = 192 cells, inputs from integer arithmetic and `arange`.

```
cells compared : 192
  bits=3: 96 compared, 0 differing
  bits=4: 96 compared, 0 differing
verdict: BIT-IDENTICAL
```

`covering_cells_by_bits = {"4": 64}` in both arms, so 64 cells reach a crossrow body
and the gate has real power; the 96 bits=3 cells reach none and are a pure
untouched-path control. **Exactly one cell in the 192-cell grid changed its
dispatched kernel** (bits=4, M=6); the other 191 dispatch a byte-identical string.

At the treated width alone that is **2,565,888 float32 values bit-identical**;
66,713,088 across the grid.

This is not subject to the "public fixture passes while a hidden prompt flips a
near-tie argmax" caveat: an argmax flip requires a *value* difference, and there is
none on any element of any scored shape at M=6.

### 5.4 Cross-arm golden: the candidate reproduces the *base's* token stream

The parity grid above is a kernel-level test. The end-to-end analogue that
`benchmark-qwen-mtp.sh` runs is weaker than it looks: it calls `mtp-verify
--generate` on the arm under test and then checks that arm against rows **it
generated itself**, so both arms pass by construction and neither is ever
compared with the other.

I ran the stronger test. Reference rows were generated **once, on the base**, and
then *both* builds were timed against that one file:

```
mtp-verify --generate 65 --mtp-depth 8   @ 4e5dc2b   -> golden-base.json (65 rows,
                                                        reference_self_consistent)
mtp-timed --golden golden-base.json --tokens 64 --mtp-depth 8   @ 4e5dc2b
mtp-timed --golden golden-base.json --tokens 64 --mtp-depth 8   @ 20bdd25
```

Every field of the trusted parent's ledger is identical, including the one that
would move first if the candidate had perturbed a logit anywhere:

| field | base `4e5dc2b` | candidate `20bdd25` |
|---|---|---|
| `all_tokens_matched` | `true` | `true` |
| `parity_all_ok` | `true` | `true` |
| `residual_divergence_count` | 0 | 0 |
| `first_divergence_index` | absent | absent |
| `reference_checked_row_total` | 64 / 64 | 64 / 64 |
| `rejected_rows_reference_checked` | 0 | 0 |
| `max_rejected_tail_logit_delta` | 0 | 0 |
| `declared_rows_total` / `emitted_token_total` | 64 / 64 | 64 / 64 |
| `target_tail_total` | 10 | 10 |
| `target_cache_offset_final` | 576 | 576 |
| `verify_block_replayed_round_count` | 0 | 0 |
| `accepted` / `rejected_draft_total` | 54 / 0 | 54 / 0 |
| `effective_draft_lengths` | `[4,5,5,6,6,6,7,7,7,1]` | `[4,5,5,6,6,6,7,7,7,1]` |

`effective_draft_lengths` matching **element-wise** is the load-bearing one for a
kernel change. The draft schedule is driven by per-position acceptance EMAs and a
cost model, so any change to a target logit — even one too small to flip an
argmax — would have had to move some round's chosen width. None moved.

Provenance: golden and base leg at `4e5dc2b`; candidate leg at `20bdd25`,
`dirty=0`, metallib `ebff5245…`. The base leg's metallib fingerprint was not
recorded (the diagnostic script gained fingerprint capture afterwards); its
chain of custody is that arm `base-4` had just rebuilt to `1e359ea9…` and no
checkout intervened. This does not weaken the claim: the two metallibs are
bit-identical on every scored shape (§5.3), so the base leg would have produced
the same ledger under either.

---

## 6. Coverage — deliverable (a), in general form

```
threadgroups           = ceil(N/8)              [bn = 8]
simdgroups/threadgroup = 2                      [group_dims(32,2,1)]
rows written/simdgroup = (4/r) * r = 4          independent of r
total rows written     = ceil(N/8) * 2 * 4 = 8*ceil(N/8)
```

The `fast` predicate the crossrow gate lives under
(`quantized.cpp:251-254`) requires `N % bn == 0 && K % 512 == 0`, so
`8*ceil(N/8) = N` **exactly**, for every legal N — not just for 17408. All eight
scored shapes satisfy `N % 8 == 0` and `K % 512 == 0`.

For the assignment's specific case: `17408/8 = 2176` threadgroups × 2 simdgroups ×
4 rows = **17408 rows**, none unwritten, none written twice.

A non-tiling `r` cannot be built:

```c++
static_assert(ROWS_PER_SIMD >= 1 && ROWS_PER_SIMD <= 4 && 4 % ROWS_PER_SIMD == 0,
              "row blocks must tile the frozen 4 rows per simdgroup exactly");
```

**M=1 unreachability.** `grid_dims(M, (N+bn-1)/bn, B)` ⇒ `ntg.x == M`. The 4-bit
crossrow gate is two `switch (ntg.x)` tiers (`out_vec_size >= 4096` and else);
**both have cases 2..9 only, no `case 1:` anywhere**, both ending `default: break;`
→ `qmv_fast_impl`. The only M=1 specialisation in the file is
`qmv_fast_singlerow_affine2_g64`, gated on `bits==2 && out_vec_size==98336`, the
coarse draft readout. Curve readback confirms M=1 → `qmv_fast_impl` in both arms.
I changed **only** the `>= 4096` tier; the lower tier's `case 6:` is byte-identical
to base.

---

## 7. Registers — deliverable met, and a model correction I owe the advisor

`crossrow_m6_ipg6_r2`, the **real production cell**, not a probe:

```
peak_live_regs = 117    allocas = 1  ([2 x [4 x i16]])    acc_spill = false
loop_backedges = 3      device_loads = 36                 float_ops = 48
```

Registers 117 vs the **125** the shipped `<T,5,5>` already costs: the register gate
passed with headroom, exactly as pre-registered. The counterfactual is intact —
`crossrow_na6` at r=4 still spills `[4 x <6 x float>]` at 144 regs. askeladd's E36
confirms 117 at all four of vpt ∈ {8,16,32,64}, so the headroom claim holds at four
independent vpt settings.

**Registers were never the problem. This experiment died on time, not on registers.**

### 7.1 The slope law is not affine in `r`

The advisor superseded `slope(r) = 8.36 + 3.19·r` with `slope(r) = 8.00 + 3.30·r`
(max residual 0.40). Refitting askeladd's own published `research/e32-rps-grid.json`
over **every spill-free `coverage_preserving` cell**:

| r | exact law | cells | max residual | measured slope | `8.00+3.30r` |
|---:|---|---:|---:|---:|---:|
| 1 | `regs = 15 + 12·NA` | NA 2..10 (9) | **0.00** | 12 | 11.30 |
| 2 | `regs = 15 + 17·NA` | NA 2..10 (9) | **0.00** | **17** | **14.60** |
| 4 | `regs = 20 + 21·NA` | NA 2..5 (4) | **0.00** | 21 | 21.20 |

The **within-`r` laws are exact, residual 0**. The **cross-`r` slope law is not
affine**: measured slopes are 12 / 17 / 21 at r = 1 / 2 / 4, i.e. +5 per unit r from
1→2 but +2 per unit r from 2→4. It is concave, so no two-parameter affine form can
fit all three; the best possible affine fit is `slope(r) = 10.00 + 2.857·r` with max
residual **1.29**, and the published `8.00 + 3.30·r` has max residual **2.40** at
r = 2 — which is exactly the r our production cell uses, and why the earlier
`16 + 15·NA` predicted 106 where the cell measures 117.

**Recommendation: use the exact per-`r` table and do not interpolate across `r`.**
Only `r ∈ {1, 2, 4}` tile the frozen 4 rows at all, so the table is already complete
and interpolation buys nothing.

Reproduce: `python3 research/e33_regfit.py`.

Consistency check on the tail-group cells: `_m<T,7,4>` peaks at 108 rather than 104
because it also instantiates `_wide<T,3>`; `_m<T,9,5>` peaks at 129 rather than 125
for the same reason. The r=4 law breaking at NA=6 (predicts 146, measures 144) is
the **spill signature**, not model noise.

---

## 8. The cost curve — deliverable (b)

Both arms: `--widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10
--skip-stock`, E27's exact settings. `C_round(M) = Σ_shapes calls_per_verify ×
seconds_per_call`, recomputed identically from each arm's own `vendored.json`.

| M | weight passes (cand) | base C (ms) | cand C (ms) | ratio | base C/M | cand C/M |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 58.676 | 58.996 | 1.0055 | 58.676 | 58.996 |
| 2 | 1 | 63.212 | 63.379 | 1.0026 | 31.606 | 31.690 |
| 3 | 1 | 72.507 | 72.427 | 0.9989 | 24.169 | 24.142 |
| 4 | 1 | 82.774 | 82.493 | 0.9966 | 20.694 | 20.623 |
| 5 | 1 | 96.163 | 96.058 | 0.9989 | 19.233 | 19.212 |
| **6** | **1** (was 2) | **128.843** | **130.781** | **1.0150** | 21.474 | 21.797 |
| 7 | 2 | 138.694 | 138.988 | 1.0021 | 19.813 | 19.855 |
| 8 | 2 | 149.490 | 149.536 | 1.0003 | 18.686 | 18.692 |
| 9 | 2 | 164.443 | 165.198 | 1.0046 | 18.271 | 18.355 |

M=1 is measured but excluded from the control set by pre-registration (E27 showed it
is a coldest-GPU artifact); it came in at 1.0055.

### 8.1 Decomposition: what the row block cost

Base within-stream increments `C(M) − C(M−1)` at constant weight-pass count
(3→4, 4→5, 7→8, 8→9): 10.27, 13.39, 10.80, 14.95 ms → **median 12.09 ms**. That is
the price of one more activation lane with no new weight pass.

| quantity | value |
|---|---:|
| base 5→6 (adds the 2nd weight pass) | +32.680 ms |
| ⇒ second weight pass alone | **+20.59 ms** |
| candidate 5→6 (one pass, row-blocked) | +34.723 ms |
| ⇒ row-blocking cost alone | **+22.63 ms** |
| **net at M=6** | **+1.94 ms (1.0150×)** |

**The row block costs 1.10× what the weight pass it removes was costing.** The saving
is real and was correctly sized; the price of collecting it is slightly larger.

Corroboration from a width the model did not fit: the candidate's 6→7 increment is
only **8.207 ms despite adding a weight pass**, because M=7 is unblocked `<T,7,4>`
and therefore *sheds* the blocking cost at the same time as it adds the pass.

### 8.2 Why it failed: the sign flips with output width `n`, and the losing side is heavier

Per-shape at M=6, sorted by contribution to the regression. `delta` is
`calls_per_verify × Δ seconds_per_call`, so the column sums to the net.

| shape | n | k | calls | base ms | cand ms | ratio | delta ms | % of net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mlp.down** | 5120 | 17408 | 64 | 30.4096 | 32.2093 | **1.0592** | **+1.7997** | **+92.8** |
| linear_attn.out_proj | 5120 | 6144 | 48 | 9.0322 | 9.4770 | 1.0492 | +0.4448 | +22.9 |
| mlp.gate_up_fused | 34816 | 5120 | 64 | 54.0672 | 53.7499 | 0.9941 | −0.3173 | −16.4 |
| full_attn.o_proj | 5120 | 6144 | 16 | 3.0961 | 3.2244 | 1.0414 | +0.1283 | +6.6 |
| linear_attn.in_proj_fused_qkvzba | 16480 | 5120 | 48 | 20.5156 | 20.4064 | 0.9947 | −0.1092 | −5.6 |
| head.lm_head | 248320 | 5120 | 1 | 5.7151 | 5.6182 | 0.9830 | −0.0970 | −5.0 |
| full_attn.qkv_proj_fused | 14336 | 5120 | 16 | 6.0071 | 6.0963 | 1.0148 | +0.0891 | +4.6 |
| head.compact_draft_vocab | 98336 | 5120 | **0** | 0.0000 | 0.0000 | 0.9868 | +0.0000 | +0.0 |

**Total wins −0.52 ms; total losses +2.46 ms; net +1.94 ms. The losing side is 4.7×
the winning side.**

Three things this table says that the unweighted ratios hide:

1. **`mlp.down` alone is 92.8 % of the regression.** It is 23.6 % of `C_round(6)` and
   it takes the largest per-call penalty (+5.9 %). Any account of this failure that
   is not primarily an account of `mlp.down` is the wrong account.
2. **`head.compact_draft_vocab` has `calls_per_verify = 0`.** Its 0.9868 is the second
   best ratio in the table and it is worth **exactly nothing** in `C_round`. I am
   flagging this because quoting it as a win would be the easiest available way to
   make this result look better than it is.
3. **The sign flips cleanly at output width.** All four `n = 5120` and `n = 14336`
   shapes lose; all four `n ≥ 16480` shapes win. Ratio is monotone in `n`, and at
   fixed `n = 5120` monotone in `k`. **It is `n`, not weight bytes, that predicts the
   gain**: `mlp.down` carries 50.1 MB and loses worst, `full_attn.qkv_proj_fused`
   carries 41.3 MB and loses much less.

Interpretation (offered as interpretation — I did not isolate it): the grid is
`grid.y = ceil(N/8)` threadgroups, so `n = 5120` launches only 640 threadgroups on
20 GPU cores. At that occupancy the kernel is latency-bound rather than
bandwidth-bound, so removing weight traffic buys nothing while the extra activation
pass still has to be paid. Only from `n ≳ 16000` is the grid large enough for the
saving to materialise.

**Why this generalises, and why rung 2 is not worth running.** This transformer pairs
every wide projection with an equally-called narrow one:

| pair | calls each |
|---|---:|
| `mlp.gate_up_fused` (n=34816) / `mlp.down` (n=5120) | 64 |
| `linear_attn.in_proj` (n=16480) / `linear_attn.out_proj` (n=5120) | 48 |
| `full_attn.qkv_proj_fused` (n=14336) / `full_attn.o_proj` (n=5120) | 16 |

The call mix is structurally balanced between the shapes this mechanism helps and the
shapes it hurts, **and the hurt side is heavier per call**. A mechanism that trades
narrow-shape cost for wide-shape gain cannot win on this call mix at any M. Rung 2
changes the constant, not the sign.

### 8.3 The ceiling on rescuing this by shape gating

The obvious repair is to apply `ROWS_PER_SIMD = 2` only where it wins. Recomputing
`C_round(6)` with each shape taking its better arm:

| gate | shapes blocked | ratio |
|---|---:|---:|
| `n ≥ 14336` | 5 | 0.9966 (−0.34 %) |
| `n ≥ 16480` | 4 | **0.9959 (−0.41 %)** |
| `n ≥ 34816` | 3 | 0.9968 (−0.32 %) |
| **oracle: per-shape best, not implementable** | — | **0.9959 (−0.41 %)** |

**The best achievable outcome from any shape gating of this mechanism is −0.41 %,
and this instrument's own control band is ±0.46 %.** The ceiling on the repair sits
inside the noise of the instrument that would have to measure it. Transferred through
the advisor's own h̄ arithmetic (`w ≈ 0.30`) that is ≈ **0.12 % of score**, below the
corrected 0.16 % detection threshold.

So the shape-gated variant is not a promising follow-up that I merely lacked time
for. It is closed by the numbers already in hand.

Reproduce: `python3 research/e33_shape_attribution.py`.

### 8.4 Harness-computed corroboration, independent of my recompute

| field | base | candidate |
|---|---|---|
| `stream_boundaries` | `[6]` | **`[7]`** |
| **`staircase rank1st`** | **8/8** | **0/8** |
| `staircase boundaries` | `[6]` | `[7]` |
| `optimal_depth_q100` | 8 | 8 |
| `optimal_speedup_q100` | 2.9233561 | 2.9203692 |
| `optimal_seconds_per_token_q100` | 0.022675627 | **0.022698819** |
| `weighted_qmv_tax_9` | 2.802580 | 2.800162 |
| `per_draft_head_seconds` | 0.0040399457 | 0.0040267540 |
| `per_round_constant_seconds` | 0.0073182379 | 0.0068777274 |
| `stop_rule_branch` | part_b_full | part_b_full |
| `scored_shapes_off_qmv_fast` | 0 | 0 |

**`staircase rank1st` 8/8 → 0/8 is the key independent line.** The candidate *claims*
its boundary moved to `[7]`, but the harness's staircase fit says that claim does not
describe the measured curve on any of the 8 shapes: the big riser is still at M=6
even though only one weight pass is now made there. Two instruments, one conclusion.

### 8.5 Two numbers I could have claimed and am not claiming

Reporting these against my own interest, per the E36 standard:

- **`weighted_qmv_tax_9` improved 2.802580 → 2.800162 (−0.086 %).** I am not claiming
  it. It is an aggregate dominated by widths where the candidate is neutral, it is
  well inside the 0.46 % control drift band, and it is weighted by a flat width mix
  rather than the M=6-dominated ranked round mass. Under ranked weighting the sign
  flips.
- **`per_round_constant_seconds` improved 6.0 % (0.0073182 → 0.0068777).** I am not
  claiming that either. It is a *fitted* parameter, and the fit absorbed the M=6
  change into the constant; the honest read is that the same fit's
  `optimal_seconds_per_token_q100` got **worse** (0.022676 → 0.022699). A fitted
  constant improving while the fit's own objective degrades is an artifact, not a
  gain.

---

## 8b. End-to-end decode — deliverable (d), and why it settles nothing

Four ABBA-counterbalanced `benchmark-qwen-mtp.sh --local-iterate` arms, run in the
order **base, cand, cand, base**, each producing both legs in one process. All four
carry `stale_metallib_warnings=0` (see §9.1 — this gate is not optional here).

| arm | head | metallib | entry °C | exit °C | serial s/tok | MTP s/tok | speedup |
|---|---|---|---:|---:|---:|---:|---:|
| `base-3` | `4e5dc2b` | `1e359ea9…` | 46.926 | 67.174 | 0.12926187 | 0.08794198 | 1.469854 |
| `cand-2` | `20bdd25` | `ebff5245…` | 47.072 | 68.012 | 0.12948109 | 0.08779255 | 1.474853 |
| `cand-3` | `20bdd25` | `ebff5245…` | 54.416 | 67.304 | 0.12919292 | 0.08790588 | 1.469673 |
| `base-4` | `4e5dc2b` | `1e359ea9…` | 49.832 | 67.791 | 0.12961011 | 0.08829142 | 1.467981 |

```
serial leg (M=1)   base 0.12943599  cand 0.12933701  cand/base 0.99924  (-0.076 %)
MTP leg            base 0.08811670  cand 0.08784921  cand/base 0.99696  (-0.304 %)
```

`MLXFAST_LOCAL_COOL_GATE=0`, so per `program.md`: arms are ABBA-counterbalanced in
one session, entry and exit temperatures are recorded above, entry spread is
**7.49 °C**, and **`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`**.
Not a gated or ranked score.

### 8b.1 The serial leg is a real null control, and it is not zero

§4 proves M=1 never reaches the changed switch tier. The serial leg is therefore a
build that *cannot* have changed, measured through the whole instrument — so its
`-0.076 %` is a direct in-experiment read of this instrument's own offset. The
arm-to-arm spread inside a single arm type is larger still: 0.27 % (base serial),
0.22 % (cand serial), 0.40 % (base MTP), 0.13 % (cand MTP).

**With n = 2 per arm, this instrument resolves roughly ±0.3 %.**

### 8b.2 What effect size the E2E was ever going to see

The trusted parent's own journal gives the realised per-round draft schedule, which
was **identical in both arms** (§5.4): `[4,5,5,6,6,6,7,7,7,1]`. Verify width is
`M = 1 + drafts`, so on this 64-token public window the ten rounds ran at

```
M = 2  x1     M = 5  x1     M = 6  x2  <- the treated cell     M = 7  x3     M = 8  x3
```

**The treated cell fires in 2 of 10 rounds.** Weighting the §8 cost curve by that
exact mix:

```
sum C_round(M_i)   base 1281.613 ms   cand 1286.571 ms   ratio 1.00387  (+0.387 % of QMV)
```

and the same journal gives `decode_seconds = 5.6162` for 64 tokens, so the
shapes-only QMV total is **≈ 22.8 % of decode**. The predicted end-to-end penalty is
therefore

```
+4.96 ms on 5616 ms  =  +0.088 %
```

against an instrument that resolves ±0.3 %. **The E2E was under-powered by about
3.5x before it was run.** The observed `-0.304 %` on the MTP leg is consistent with
zero *and* with the prediction, and I am claiming it as neither.

I want to be blunt about this rather than let a favourable-looking `-0.304 %` sit in
a table: **the end-to-end arms do not corroborate the primary metric and do not
contradict it. They are not evidence about this mechanism at all.** What they do
establish is that the candidate runs the real decode path, matches the base's token
stream (§5.4), and produced no crash, no divergence and no schedule change.

The 22.8 % QMV share is itself an estimate: `--shapes-only` times each dispatch in
isolation with hot caches, while in situ these dispatches interleave with the head
chain and may partly overlap. If the true in-situ share is lower, the predicted
effect is *smaller* still, so the conclusion is conservative in the right direction.

### 8b.3 A number I got wrong earlier, corrected against my own interest

In §8.3 I sized the oracle per-shape gating ceiling (`ratio 0.9959` at M=6) as
"≈ 0.12 % of score" using an assumed M=6 round weight `w ≈ 0.30`. Both inputs were
too generous. Measured here, `w = 0.201` on this window, and QMV is only ≈ 22.8 % of
decode, so that ceiling is worth

```
0.41 % x 0.201 x 0.228  =  0.019 % of decode
```

roughly **6x smaller than I claimed**, and about one eightieth of the 0.16 %
single-run detection threshold. The follow-up in §8.3 was already closed; this
closes it by a much larger margin than I first stated.

### 8b.4 🔴 The number you assigned E37 to find — I have a local measurement of it, and it is below 30 %

Your latest comment says I am *assuming* ~30 % round mass at M=6, that askeladd's E37
will measure the real share for beagle, and: *"If it comes back much below 30 %, tell
me and we re-aim rather than pushing on."*

**Telling you now, because this experiment produced the measurement as a by-product.**

On the local public fixture the trusted parent's own journal gives the exact
per-round schedule, identical in both arms:

```
effective_draft_lengths = [4, 5, 5, 6, 6, 6, 7, 7, 7, 1]      (M = drafts + 1)
M-histogram             = {2:1, 5:1, 6:2, 7:3, 8:3}
w(M=6) by QMV round cost = 20.1 %          (assumed: ~30 %)
```

Your `M = chosen_depth + 1` is confirmed here independently: the ten draft counts sum
to 54, `accepted_draft_total = 54`, and 54 + 10 primaries = 64 emitted tokens exactly.

**Four caveats, because this is not beagle and I do not want it used as if it were:**

1. **Wrong head.** You established that round mass is head-dependent and local runs
   resolve `7bbb40de…`/`07293af7…`, never the declared `559b24eb…`. This 20.1 % is
   a within-head fact about the local head.
2. **My window is *wider* than beagle, not narrower.** Local `effective_mean_draft_len
   = 5.4`; beagle's ranked value is **4.5327**, and the ranked pool mean is 5.078. So
   the local histogram sits *above* beagle's. This is the opposite of the "local
   saturates near 2.4" case you quoted from edward's fixture — different fixture,
   different behaviour, and worth knowing before E37 assumes locals under-draft.
3. **Direction of transfer is not determined.** Beagle's lower mean shifts its mass
   down toward M=5/6, which could raise its M=6 share above my 20.1 %. I measured a
   histogram, not a ramp model, so I will not extrapolate the shape.
4. It is a 64-token window with a visible ramp (`4,5,5,6,6,6,7,7,7`); the ranked
   window is 512 tokens where the ramp is amortised over ~8x more rounds.

**What I would ask E37 to do with this.** The blocker is mechanical, not scientific:
`effective_draft_lengths` lives in `${run_dir}/mtp-decode.json`, and
`benchmark-qwen-mtp.sh:447-468` `rm -rf`s that directory in its EXIT trap, which is
why nobody has had this number. `research/e33-diag.sh` in this PR calls the trusted
`mtp-verify`/`mtp-timed` verbs directly and keeps the journal, so askeladd can have
the histogram without re-deriving the plumbing. **He should not spend a work block
rebuilding it.**

### 8b.5 What this does to the M=6 case, stated plainly

You wrote that E27's M-table has `M=5: 0.7990` and `M=9: 0.8854` as large wins and
**`M=6: 1.0032` flat**, and that if beagle's mass sits at M=6 then "the win is
currently zero there". E33 was the attempt to convert that zero into a win.

It went the other way: **1.0032 → 1.0150.** M=6 is not merely the width where we have
no win; on the only mechanism tried it is the width that actively resists the
treatment that worked at M=5. §8.2 says why — the loss is concentrated in the two
narrow-output shapes (`mlp.down` +5.9 %, `linear_attn.out_proj` +4.9 %), and every
shape with `n ≥ 16480` did improve. The M=5 win at NA=5 needed no row blocking at all,
so it never paid the activation re-read that sinks the narrow shapes here.

Combining that with 8b.4: even the oracle per-shape-gated best (0.9959) at a measured
`w = 0.201` is worth ~0.019 % of local decode against a 0.16 % detection threshold.
**I do not think M=6 should be re-aimed at with a variant of this mechanism.** If
E37 returns a beagle share far above 30 %, that changes the value of *some* M=6 win —
it does not resurrect this one, because the −0.41 % oracle ceiling is a property of
the per-width cost table and is already measured.

### 8b.6 On your "keep every win at M ≥ 2"

Recorded against my own result: the candidate's M=1 cost ratio is **1.0055**, i.e.
nominally *slower*, well inside the ±0.46 % control band and not a real effect. The
relevant point is the sign — this change does not produce a serial-leg speedup, so it
carries none of the score-negative risk you flagged. §4 proves M=1 cannot reach the
changed dispatch tier at all, so 1.0055 is instrument noise by construction, which is
also a useful read on the noise of a single control cell.

### 8b.7 GPU timing lock

The primary curve arms and all four E2E arms ran under `benchmark.sh`'s local run
lock (`research/run-qmv-curve.sh:45-80` reuses `acquire_local_run_lock`;
`benchmark-qwen-mtp.sh` acquires it directly). No concurrent model-holding process was
observed at any point, and the cross-session anchor holds: base M=6 measured
128.843 ms here against E27's published 128.865 ms (0.017 %). **No disturbance to
flag.** The `research/e33-diag.sh` runs in §5.4/§8b.4 do *not* take the lock; I make no
timing claim from them, and their numbers appear in this report only as ledger fields.

### 8b.8 A cheaper consequence for whoever plans the next width experiment

On this window the schedule spends 6 of 10 rounds at M ∈ {7, 8} and only 2 at M = 6,
and the ranked window is 512 tokens where the ramp's share is smaller still. Any
experiment that treats **one** width should size itself against the width's measured
round mass, not against its position in the cost curve. E33's own cost curve says
the M = 6 cliff is the largest single step in the ladder (+32.7 ms, §8.1); its round
mass says removing that cliff entirely would be worth ≈ 0.20 x 32.7 / 5616 ≈ 0.12 %
of local decode. That sizing was available *before* the experiment ran, from
`effective_draft_lengths` in any prior local run, and it would have argued for
treating M ∈ {7,8} first.

Reproduction (both legs, one arm):

```bash
git checkout --detach 4e5dc2b            # or the candidate branch
tools/build-mlx-metallib.sh --all-build-roots      # NOT done by the wrapper
MLXFAST_LOCAL_COOL_GATE=0 MLXFAST_SCORE_PATH=out/score.json \
  ./benchmark-qwen-mtp.sh --local-iterate
```

Cross-arm golden and per-round widths:

```bash
research/e33-diag.sh golden          # on the base
research/e33-diag.sh timed base      # on the base
research/e33-diag.sh timed cand      # on the candidate
python3 research/e33_diag_compare.py
```

---

## 9. Provenance, scope and budget — deliverable (e)

| item | value |
|---|---|
| metallib fingerprint, base `4e5dc2b` | `1e359ea9f49b1651b01b68d49333dba557db9a619fc8e7776e5ad9bc5f495e80` |
| metallib fingerprint, candidate | `ebff5245d590fc9b03b25b26e1f6a3f9d3d8baa7793bf48eb16ad8bcdc1e90cc` |
| `python3 research/twin_audit.py` | `TWIN AUDIT OK: 29 runtime-effective twin(s)` |
| `senpai/validate-assignment-scope.sh` | `assignment scope OK: 2 submitted path(s)` |
| `senpai/check-editable-budget.sh` | `source=2460377/3000000 headroom=539623 growth=5088/262144 exempt=2410/2GiB files=154` |

Submitted paths, both edited at the twin-locked offset +13:

- `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
- `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`

Everything else on the branch is research-only and not submitted.

### 9.1 🔴 A trap I fell into, and the guard I now recommend

**`benchmark-qwen-mtp.sh --local-iterate` does not rebuild `mlx.metallib`.** It only
forwards the CLI's warning and runs anyway:

```
mlxfast-swift: warning: mlx.metallib ... was built from different vendored Metal
sources than the ones on disk (recorded ebff5245..., current 1e359ea9...);
rebuild it with tools/build-mlx-metallib.sh ... so kernel edits are not masked
by a stale metallib
```

`research/run-qmv-curve.sh:130` rebuilds (`tools/build-mlx-metallib.sh
--all-build-roots`) and `benchmark.sh:1774` rebuilds when stale. The Qwen MTP
wrapper does neither. So checking out a different commit and running
`--local-iterate` silently times **whichever kernel was compiled last.**

My first two base E2E arms did exactly this: they sat at `4e5dc2b` and executed the
**candidate** metallib. Four warnings per run, and I would have reported them as base
timings. **I discarded them and re-ran with an explicit rebuild.** The arms reported
below all carry `stale_metallib_warnings=0`, which `research/e33-e2e-run.sh` now
enforces as a fail-closed gate (exit 91).

**The primary metric was never affected**, and this is checkable two ways: both curve
arms record **0** such warnings, and — independently — the two arms report *different*
dispatch readbacks at M=6 (`<T,6,3,true>` vs `<T,6,6,true,2>`), which is impossible if
they had shared one metallib. That readback is the guard that made this detectable at
all.

Recommendation for the campaign: either have `benchmark-qwen-mtp.sh` rebuild like its
two sibling harnesses do, or make the warning fatal. It is currently a silent
wrong-answer generator for exactly the base-vs-candidate comparison every experiment
here has to run.

Both curve arms record `cool_gate_vendored=stalled_above_40C`: this host's idle GPU
floor sits above the 40 °C gate. This is the documented pre-existing protocol for
this instrument — all widths are measured round-robin inside one process, so a
thermal floor biases every width equally and cancels in the ratio. Entry temperatures
matched to **0.03 °C** (43.116 vs 43.090 °C); exits 68.72 vs 66.96 °C.

---

## 10. Implementation

Both twins, offset +13:

1. `qmv_fast_crossrow_affine4_g64_wide` gains a 4th template parameter:
   `template <typename T, int NA, bool DIRECT_NIBBLES = false, int ROWS_PER_SIMD = 4>`;
   the body takes `constexpr int rows_per_simd = ROWS_PER_SIMD;`.
2. The NA bound is relaxed to a flat
   `static_assert(NA >= 2 && NA <= 9, "wide multi-row QMV supports NA in [2, 9]");`.
3. New wrapper `qmv_fast_crossrow_affine4_g64_rowblocked<T, NA, DIRECT_NIBBLES,
   ROWS_PER_SIMD>` loops `4 / ROWS_PER_SIMD` blocks, calling `_wide` at
   `out_row + b * ROWS_PER_SIMD`, with the tiling `static_assert` from §6.
4. `qmv_fast_crossrow_affine4_g64_m` gains `int ROWS_PER_SIMD = 4` as a 5th parameter
   and routes both its main and tail groups through `_rowblocked`.
5. Dispatch switch, **`out_vec_size >= 4096` tier only**: `case 6:` changes from
   `<T, 6, 3, true>` to `<T, 6, 6, true, 2>`. Every other case is unchanged.

### 10.1 A design reversal worth recording

My first cut made the `_wide` NA bound depend on `ROWS_PER_SIMD`. That **broke
askeladd's E32 instrument** — its `xctl_e27_spill_na6_r4` negative control stopped
compiling, because that control deliberately instantiates a cell the new bound
rejected. Which `(NA, r)` pairs are spill-free is a *measured toolchain property*,
not a language invariant, and encoding it as a `static_assert` destroys the very
instrument that measures it. Reverted to a flat `NA <= 9`.

E32 was then re-validated end to end: `crossrow_rps_gen.py --check` OK (2 rewrites),
`crossrow_rps_sweep.py` zero non-ok cells, gate validation 10/10, and **all 77 E32
cells reproduce field-for-field**, with `research/e32-rps-grid.json` restored to
askeladd's published bytes.

---

## 11. Honest caveats

1. **This is one host, and it is not the ranked host.** M4 Pro, 20 GPU cores,
   `applegpu_g16s` gen 16. The ranked M5 is gen ≥ 17 with NAX. My §8.2 explanation is
   occupancy-based and 20 cores is a specific number; a wider M5 could move the
   `n` threshold at which row blocking starts to pay. What does **not** depend on core
   count is §8.2's structural argument — the narrow/wide call pairing is a property of
   the model, not the GPU.
2. **The per-shape law in §8.2 is a fit, not an isolation.** I did not run an
   occupancy sweep or a bandwidth counter to prove the latency-bound reading. The
   monotonicity in `n` is measured; the causal story is inference.
3. **I did not run the direct decomposition control.** A `<T,6,3,true,2>` arm (two
   weight passes *and* row blocking) would measure the +22.63 ms blocking cost
   directly instead of extrapolating it from within-stream increments. The verdict
   does not need it — the sign is 3.3× the control band — but the +20.59/+22.63 split
   would be measured rather than inferred.
4. **`crossrow_na_max` reads 4 in both summaries.** This is a cosmetic CLI default in
   the summary writer, not a readback; §4's dispatch evidence is the real instrument.
5. **The cost curve is a shapes-only microbenchmark.** It measures the QMV shapes in
   isolation, not the full round with attention, recurrence and head work interleaved.
   That is what makes it a clean instrument for this question, and it is also why §8's
   ratios are not score ratios. The ≈22.8 % QMV share of decode in §8b.2 inherits the
   same limitation: it is a sum of isolated dispatch times, so in-situ overlap can only
   make it smaller.
6. **The end-to-end arms are under-powered, not null.** §8b.2 shows the predicted
   effect (+0.088 %) is ~3.5× below what four arms can resolve (±0.3 %). Reporting
   `-0.304 %` without that arithmetic would have been the most misleading number in
   this document, because it has the sign I wanted.
7. **The round-mass measurement is on the local head.** `w(M=6) = 20.1 %` (§8b.4) is
   a within-head fact on a 64-token window. It is evidence for E37, not a substitute
   for it.

---

## Reproduction

```bash
# ---- cost curve, both arms (E27's exact settings) --------------------------
git checkout --detach 4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549
research/run-qmv-curve.sh e33-base-r1 4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549 \
  --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock

git checkout qwen-thorfinn/m6-single-pass-rowblocked   # 20bdd259
research/run-qmv-curve.sh e33-cand-r1 4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549 \
  --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock

# ---- primary metric, per-shape attribution, gating ceiling ------------------
python3 research/e33_cost_ratio.py            # C_round(M) both arms, ratio table
python3 research/e33_shape_attribution.py     # per-shape delta + oracle gate ceiling
python3 research/e33_dispatch_reached.py      # qmv-vs-qmm readback assertion
python3 research/e33_regfit.py                # per-r register laws from E32's grid

# ---- correctness -----------------------------------------------------------
research/run-qmv-parity.sh \
  base=4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549 \
  cand=20bdd2595bb962013f18ca4882cb21a4eed6b695
# NOTE: qmv_parity_compare.py never exits nonzero. Parse `verdict:` from its text.

# ---- static AIR register gate ----------------------------------------------
xcrun -sdk macosx metal -std=metal3.1 -S -O2 -DCROSSROW_NA_PROBE_WIDE \
  research/crossrow_na_probe.metal \
  -I Vendor/mlx-swift/Source/Cmlx/mlx -o /tmp/e33_probe.ll
xcrun -sdk macosx metal-opt -passes='default<O3>' -S /tmp/e33_probe.ll \
  -o /tmp/e33_probe.o3.ll
python3 research/air_kernel_stats.py /tmp/e33_probe.o3.ll --match crossrow_

# ---- end to end, ABBA (deliverable (d)) ------------------------------------
# BOTH runner scripts live only on this branch, so copy them outside the
# checkout first -- otherwise they vanish when you detach to the base and the
# two arms are no longer driven by the identical file.
cp research/e33-e2e-run.sh research/e33-diag.sh /tmp/
# e33-e2e-run.sh rebuilds the metallib itself and FAILS CLOSED (exit 91) on any
# stale-metallib warning. See 9.1: this is not optional.
bash /tmp/e33-e2e-run.sh base-3     # detached at 4e5dc2b
bash /tmp/e33-e2e-run.sh cand-2     # at 20bdd259
bash /tmp/e33-e2e-run.sh cand-3     # at 20bdd259
bash /tmp/e33-e2e-run.sh base-4     # detached at 4e5dc2b
python3 research/e33_e2e_compare.py

# ---- cross-arm golden + per-round width histogram --------------------------
# Generates reference rows ONCE on the base and times BOTH builds against that
# one file; also keeps the journal that benchmark-qwen-mtp.sh rm -rf's. Output
# goes to /tmp/e33_diag so the golden survives the checkout between arms.
bash /tmp/e33-diag.sh golden        # detached at 4e5dc2b
bash /tmp/e33-diag.sh timed base    # detached at 4e5dc2b
bash /tmp/e33-diag.sh timed cand    # at 20bdd259
python3 research/e33_diag_compare.py

# ---- provenance ------------------------------------------------------------
tools/build-mlx-metallib.sh --print-fingerprint
python3 research/twin_audit.py
senpai/validate-assignment-scope.sh 4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549 \
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
senpai/check-editable-budget.sh 4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549
```

### W&B runs

| arm | run | URL |
|---|---|---|
| cost curve, base `4e5dc2b` | `2pgw1cmc` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2pgw1cmc |
| cost curve, candidate `20bdd259` | `d3ct2cxl` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/d3ct2cxl |

---

## 12. Suggested follow-ups (not implemented)

**Shape-gated row blocking is NOT on this list.** It is the obvious repair, I costed
it in §8.3, and its unimplementable oracle ceiling is −0.41 % against a ±0.46 %
control band. It is closed, not deferred.

1. **The direct decomposition control**, `<T,6,3,true,2>` — two weight passes *and*
   row blocking. One arm, and it converts the +20.59 / +22.63 ms split in §8.1 from an
   extrapolation into a measurement. Cheap, and it is the only loose end in the
   causal story.
2. **Isolate the occupancy reading in §8.2.** The monotonicity in `n` is measured; the
   latency-bound explanation is inference. A synthetic sweep of `n` at fixed `k` on
   this kernel would turn it into a curve with a located knee, which is reusable by
   any future kernel experiment on these shapes.
3. **Re-test the threshold on M5 before discarding the mechanism entirely.** The `n`
   knee is the whole result and it is the quantity most likely to move with core
   count; if M5's knee falls below 5120 the sign flips everywhere. Needs ranked-box
   time, so it is an advisor decision, not a student one.
4. **`mlp.down` deserves its own assignment.** It is 23.6 % of `C_round(6)`, it is the
   shape that killed this experiment, and it is structurally odd — the only scored
   shape with `n = 5120` and `k = 17408`. Whatever makes it the most penalty-sensitive
   shape in the set is worth understanding independently of row blocking.
5. **Make `benchmark-qwen-mtp.sh` rebuild the metallib, or make the stale warning
   fatal.** Its two siblings both rebuild (`research/run-qmv-curve.sh:130`,
   `benchmark.sh:1774`); the Qwen MTP wrapper does neither and only forwards the CLI's
   warning to a log nobody parses. This silently invalidated two of my arms (§9.1). It
   is a one-line change in campaign plumbing, it protects every future E2E arm by
   *anyone*, and I did not make it because it is outside my assignment's scope.
6. **Widths 7 and 8 carry more round mass than 6 on the measured histogram.** Not a
   recommendation to row-block them — that is rung 2, which this result kills — but if
   a *different* mechanism is proposed for a single width, §8b.4's histogram says
   M ∈ {7,8} is where to point it. Sizing should come from
   `effective_draft_lengths`, which is now cheap to obtain (`research/e33-diag.sh`).

---

## 13. What I would tell the next student

The register model, the dispatch gate, the coverage arithmetic and the reassociation
proof were all correct, and all four were verified before timing. The prediction was
still wrong by 18 %, because **all four are source facts and the ratio is a physical
one.** The one thing none of them contained was how much a re-read of the activation
tile costs at `n = 5120` on a 20-core GPU — and that single unmeasured quantity was
larger than the entire modelled saving.

The advisor wrote this to me before the arms ran: *"a mechanism verified in source is
not a mechanism sized against data."* This result is that sentence, measured.

There is a second, cheaper lesson. I spent the whole experiment believing M=6 carried
~30 % of round mass because that was the number in my brief. It is **20.1 %**
(§8b.4), and I could have measured it on day one from `effective_draft_lengths` in any
prior local run — the only reason nobody had is that the harness deletes the file. So:
**before treating one width, spend ten minutes measuring that width's round mass.**
The same ten minutes also tells you what effect size your end-to-end instrument would
have to resolve, which is how I know my four E2E arms were under-powered by 3.5× —
and I know it *after* running them rather than instead of running them.

Finally: the E2E number came back at −0.304 %, which looks like a win. The primary
metric says +1.50 %, which is a loss. They are not in conflict — the E2E cannot see an
effect this small — but the temptation to lead with the friendly number was real, and
the arithmetic that dissolves it took about five minutes. Do that arithmetic before
you write your headline, not after someone asks.
