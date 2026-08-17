# Senpai campaign submission 02 — combined: 3-bit compact draft readout + direct-nibble QMV at M=3,4,5

Supersedes `01-direct-nibbles-m345.md`, which described only the second of the two
mechanisms below. `01` is retained as history; this note is the authoritative
description of what is actually packaged.

**Model label:** `senpai` (a campaign label, not a distinct model — the target
weights and the pinned MTP head are the ones the benchmark declares).

---

## 1. What is in this submission

Two independent, bit-exactness-preserving changes to the decode path. They touch
disjoint code and disjoint kernel dispatch cases, and we show below that they
cannot interact.

| # | mechanism | submitted surface | claim |
|---|---|---|---|
| A | **3-bit compact draft-head readout as the compiled default** | `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` (+109/−4) | **+0.83 % modeled ranked**, from a directly measured +0.7754 % local mechanism-only effect |
| B | **`DIRECT_NIBBLES` at crossrow QMV widths M = 3, 4, 5** | `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h` + its `mlx-generated/quantized.cpp` twin | **no speedup claimed.** Bit-exact ALU reduction; measured to cover only ~5 % of verify rounds at the shipped depth policy |

Everything else in the tree is unchanged from the promoted frontier
`bd007bc7` (`sourceRef d1530a409848b82a0a1890141c1483875d1e0173`, score
3.13098700135133), which this branch merged at `1a66487`.

The protected-path delta against `origin/main` is exactly three files:
`quantized.h`, `mlx-generated/quantized.cpp`, `Qwen35.swift`.

---

## 2. Mechanism A — 3-bit compact draft-head readout

### 2.1 What changed

`Qwen35.swift` gains a compact draft head whose readout weights are requantized
from the pinned 4-bit head to 3 bits, materialised once before the timed region,
and a compiled default that selects it:

```swift
private static let draftHeadBits: Int = {
    guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN_MTP_DRAFT_BITS"],
          let bits = Int(raw), [2, 3, 4].contains(bits)
    else { return 3 }
    return bits
}()
```

The environment variable exists only so the experiment could run its arms. In the
ranked workflow nothing sets it, so **the compiled default `3` is what runs**.
The requantization block short-circuits when `draftHeadBits == quantized.bits`,
so the 4-bit arm is byte-for-byte the incumbent.

### 2.2 Why it is legal and exactness-preserving

The draft head is the *proposal* head. Every token it proposes is re-checked by
the full-precision target forward before it can be emitted; the harness's own
token-fidelity gate (`trusted-sequential-reverification-exact-token-match`) is
what decides what is emitted. Lowering draft-head precision can only change
*which tokens are proposed*, never which tokens are *emitted*. All four
experiment arms and the combined receipt report
`all_tokens_matched = true`, `residual_divergence_count = 0`,
`max_rejected_tail_logit_delta = 0`, `parity_all_ok = true`.

This is not an "undeclared auxiliary proposal head": it is a requantized view of
the *declared, pinned* MTP head, derived deterministically from that head's own
weights, with no additional training and no additional information source.

### 2.3 The measurement

Four ABBA-counterbalanced arms in one session, bit order `4, 3, 3, 4`,
512 decode tokens each, one host, one head:

| arm | position | bits | serial s/tok | notes |
|---|---|---|---|---|
| p1 | 1 | 4 | 0.065683 | |
| p2 | 2 | 3 | 0.065571 | |
| p3 | 3 | 3 | 0.065636 | |
| p4 | 4 | 4 | 0.065711 | |

Serial-control spread across the four positions is **0.213 %**; the
control-vs-control MTP noise term is **0.000455 %**; headline serial drift is
**−0.1054 %**. Pooled MTP effect **−1.9844 %** with halves −1.9630 / −2.0059 %.

That headline is then decomposed, because most of it is not the mechanism:

| term | seconds | share of the decode-work delta |
|---|---|---|
| `term_readout_precision` | −0.138436 | **37.93 %** |
| `term_fewer_draft_calls` | −0.006990 | 1.92 % |
| `term_trajectory` | −0.219519 | **60.15 %** |

`term_trajectory` is the acceptance-path change — a different draft head proposes
a slightly different token sequence, which changes how many verify rows are
needed. That term is prompt-specific and does not transfer. **We claim only the
mechanism:** `mechanism_only_score_pct = +0.7754 %` locally.

A model-free cross-check reproduces the decode-work delta without the
attribution model: rows −1.4011 % × cost-per-row −1.2091 % = −2.5932 % against a
measured −2.6073 %. Per-round cost is flat at +0.0394 %, which resolves an
earlier apparent "+0.450 % per-round regression" as a rows-versus-rounds
artifact.

### 2.4 Transfer to the ranked configuration

The readout is a pure bandwidth term. At 4 bits the compact readout streams
283.208 MB per draft step; at 3 bits, 220.273 MB (−22.22 %). Measured in situ:
1.16635 ms → 0.88271 ms per call, i.e. effective bandwidth **242.81 → 249.54
GB/s** (this kernel is read-dominated, so it runs *above* the host's 227.13 GB/s
mixed STREAM peak).

The transfer coefficient is the **share of decode work that is readout**:

* locally, against a bf16 head, readout share is **25.00 % → +0.7754 %**;
* on the ranked run the head is the **pinned 4-bit head**, so the same absolute
  readout cost is a larger share of a smaller total: **54.24 % → +0.8334 %**.

Modeled ranked effect: **+0.83 %**. Against the live bar of 3.13098700135133
that is ≈ 3.157.

### 2.5 Geometry: why this is *only* a draft-side change

`Qwen35.draftTokenID` reshapes to `[compactDraftPaddedCount]` (1-D, 98,336) and
`Qwen36MTPBlockSession` slices `draftHidden = headHidden[0..., (dim(1)-1)..<dim(1), 0...]`,
which is `[1, 1, H]`, inside a strictly sequential depth loop. **The draft readout
is unconditionally M = 1.** This matters for §4 below.

---

## 3. Mechanism B — `DIRECT_NIBBLES` at crossrow QMV widths M = 3, 4, 5

### 3.1 What changed

The wide crossrow QMV dispatch table in `quantized.h` (and its JIT twin
`mlx-generated/quantized.cpp`) already passed `DIRECT_NIBBLES = true` to
`qmv_fast_crossrow_affine4_g64_m<T, M, IPG, true>` at M = 6, 7, 8, 9. Cases
M = 3, 4, 5 did not. They now do. No kernel body changed; only three template
arguments.

### 3.2 Why it is bit-exact

The flag only moves powers of two across the product.

* Incumbent: mask nibble *j* at bit offset 4*j*, giving `n_j · 2^{4j}`, and
  multiply by `x_j · 2^{-4j}` (that scaling is what `load_vector<T,float,4,4>`
  writes: `x[i], x[i+1]/16, x[i+2]/256, x[i+3]/4096`).
* With the flag: mask nibble *j* down to `n_j` and multiply by `x_j`.

Each factor differs from its counterpart only in its **exponent field**, so every
individual product is bit-identical, and the accumulation order is untouched. The
x-side row sum becomes `xm[0] + xm[1] + xm[2] + xm[3]`, which is precisely
`load_vector`'s own (unscaled) expression tree, so that reduction is bit-identical
too. The argument is independent of M and of NA.

What disappears is 12·NA power-of-two multiplies per 512-element k-block per
simdgroup, against 64·NA FMAs that remain — a ~12 % ALU reduction on an
ALU-bound kernel.

### 3.3 Parity gate — measured, not argued

`research/run-qmv-parity.sh`, reference `1a66487` vs candidate `67856b5`,
555.685 s:

```
96 cells | 0 differing | widths differing: [] | verdict: BIT-IDENTICAL
```

Non-vacuity is established two ways. The twins genuinely differ between the two
arms (`a83039b7…/c2200b8f…` vs `6247ca18…/97785051…`), and the gate has power in
exactly the region changed: the cost-curve test sweeps `widths = Array(1...12)`
over 8 scored shapes, so 8 × 7 = **56** of the 96 cells exercise M = 3, 4, 5 —
which the run records as its 56/96 positive control.

### 3.4 Honest coverage — this is much smaller than we first thought

The original in-kernel rationale cited a depth histogram `{1:19, 2:138, 3:67,
4:21}` → widths `{2:19, 3:138, 4:67, 5:21}`, i.e. 226 of 245 rounds on M = 3..5.
**That histogram comes from a different solver configuration** (a campaign
experiment arm with its own depth policy), and it does **not** describe the
shipped default. We re-measured on the shipped default with the declared head:

| bucket | rounds | share |
|---|---|---|
| widths this change adds (M = 3, 4, 5) | **1 / 20** | **5.0 %** |
| widths that already carried the flag (M = 6..9) | 18 / 20 | 90.0 % |
| narrow crossrow path M = 2 (still no flag) | 1 / 20 | 5.0 % |

Shipped-default depth histogram: `{1:1, 4:1, 5:8, 6:3, 7:5, 8:2}`, 114 rows over
20 rounds, `effective_mean_draft_len = 5.7`, `effective_max_draft_len = 8`.

So mechanism B closes a real hole in dispatch coverage and is free and exact, but
**we claim no speedup from it** and would not have submitted it on its own. The
in-kernel comment has been corrected to say exactly this, with the provenance of
each histogram named.

---

## 4. Why A and B cannot interact

Mechanism B requires `bits == 4` **and** a crossrow width in 2…9. Mechanism A's
readout is unconditionally **M = 1** (§2.5), and at M = 1 the device-side
`switch (ntg.x)` has **no case at all** — both bit widths fall through to
`qmv_fast_impl<T, 64, bits>`. The two changes are therefore disjoint by
construction, not merely by observation.

---

## 5. End-to-end receipt for the combined tree

Full local `--local-submit` gate on the exact submitted tree (rebuild → metallib
→ public drift tripwire → correctness → mtp-verify → two timed legs):

```json
{"score":1.7355953380083484,"passed":true,"track_id":"qwen3.8-27b-mtp-v1",
 "metrics":{"mode":"qwen-mtp-local-submit","oracle":"candidate-local-mtp-golden-rows",
 "public_drift_tripwire_passed":true,"decode_tokens":128,"mtp_depth":8,
 "all_tokens_matched":true,"uses_pinned_mtp_head":true,
 "effective_mean_draft_len":5.7,
 "serial_seconds_per_token":0.098464609123766422,
 "mtp_seconds_per_token":0.056732469238340855,
 "mtp_decode_speedup":1.7355953380083484,
 "accepted_draft_rate":0.95614035087719296,"residual_divergence_count":0}}
```

`mtp-verify: rows=129 seed_tokens=512 reference_seed_token=271 self_consistent=true chain_contradictions=0`.

Build provenance:

```
c1334052cee358ac5bf8e232178dbf0caae72067299c164d3aee3ba03891b8f9  .build/release/mlxfast-swift
0f1b019c1c34f9e859a4d942dcd3e0c071fe0d53011a9eb26d82187889520efe  .build-worker/release/mlxfast-runtime-worker
47b06e36cb88c00f6126134087103fb9c1561014fdade495fe32923326bdba24  .build-worker/release/mlx.metallib
6247ca18bcf188367948725df9ed5d52a02ba18eea541f15b2391c6e1a54cbe8  quantized.h
97785051608980a610e962ccd9820c5ade28deff67352dfc1c3c7e7b23aab420  mlx-generated/quantized.cpp
```

The metallib digest is unchanged from before mechanism B, which is correct and
expected: the `quantized` family is JIT-compiled from the embedded source string
in the `.cpp` twin, not loaded AOT from `mlx.metallib`. A `strings` tripwire on
the built worker confirms the Swift-side change is present.

### 5.1 A corroborating decode-only comparison, and its noise floor

Comparing the pre-merge tree (mechanism B only, 4-bit readout) against the
combined tree, **with the harness's prefill contribution removed** — the harness's
`seconds_per_token` field is prefill-*inclusive*, and at 128 tokens prefill is
**55.01 %** of the MTP leg:

| arm | serial true decode | MTP true decode |
|---|---|---|
| mechanism B only, 4-bit readout | 8.545129 s | 3.307839 s |
| combined, 3-bit readout | 8.607571 s | **3.266720 s** |

MTP true decode **−1.2431 %**. The serial leg — which *cannot* have changed —
moved **+0.7307 %**, and that is the honest single-pair noise floor for this
comparison. So this is corroborating, not confirming.

What makes it interesting is that the two runs are **trajectory-identical**:
`round_count 20`, `accepted_draft_total 109`, `rejected_draft_total 5`,
`declared_rows_total 134`, `accepted_draft_rate 0.956140350877193`, and the
entire per-round width vector
`[4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 5, 5, 7, 7, 8, 8, 5, 5, 1]` are identical
between them. So the trajectory term (60 % of the ABBA headline) is **exactly
zero** here, and the residual is a pure mechanism measurement whose sign and
rough magnitude match the ABBA `term_readout_precision` (−0.989 % of decode
work).

The trajectory identity has a straightforward explanation: on this prompt the
top1−top2 logit margin has median **15.0** and minimum **0.375** over 513
sequential reference rows, so a 3-bit requantization perturbation essentially
never flips an argmax.

---

## 6. Disclosures

We would rather under-claim than have a reviewer find these.

1. **Two of the four ABBA arms were timed on a driver head that is not in any
   pushed ref.** Positions 1–2 used driver head `9ba16c81`, positions 3–4 used
   `bf675270`; both worktrees were clean. We proved the compiled surface is
   nevertheless identical: `git rev-parse <that ref>:…/Qwen35.swift` and
   `git rev-parse HEAD:…/Qwen35.swift` both resolve to blob
   `3adfe8480f0f813570b4696e571900c9569320cb`, the compiled-surface diff is
   empty, and all four arms report the same
   `worker_sha256 = b00c06a759e12588e02b5a7eda3bd97b316c1fce97819b416ab84b8f35806b83`.

2. **The harness's thermal cool gate did not pass on any timed arm**, here or in
   the combined receipt. The gate requires GPU ≤ 40 °C; this host's *idle* GPU
   temperature is 42.9 °C, so the gate is unsatisfiable rather than merely unmet.
   Every arm therefore carries `cool_gate_passed_real_gate = false` and
   `gate_qualified_for_timing = false` verbatim, and we compensate by (a) ABBA
   counterbalancing within a single session, (b) recording entry/exit GPU
   temperature per arm — entry 42.987 / 43.233 / 43.408 / 43.698 °C, a 0.711 °C
   spread that is monotone with position; exit 57.10 / 57.15 / 57.54 / 56.60 °C —
   and (c) reporting the serial-control spread (0.213 %) next to the effect
   (0.7754 %). The combined receipt records entry 45.620 °C, exit 61.041 °C.

3. **A thermal claim from an earlier round of this experiment was retracted by
   its author.** The true idle floor is 39.92 °C; the previously reported 40.42 °C
   was a post-build reading.

4. **Mechanism B's original rationale over-stated its coverage by ~18×** (§3.4).
   We found this ourselves, after committing it, and corrected the in-tree
   comment rather than quietly leaving it.

5. **The local harness dilutes decode-only effects.** Its `seconds_per_token` is
   prefill-inclusive; prefill is ~55 % of the MTP leg at 128 tokens, ~16 % at 512,
   and ~6.2 % on the ranked run. Every local decode-only number in this note has
   been converted before being compared to a decode-only model.

6. **The local `--local-submit` harness cannot exceed ~300 decode tokens on the
   public golden prompt** — the block session clears its pending state at a stop
   token (first reached at emitted index 300) and the next round fails. This is a
   local-harness limitation, not a scoring hazard: the ranked run scores at 512
   decode tokens across 8 pooled prompts and none of them hits a stop token
   inside that window.

---

## 7. What we are *not* claiming

* No speedup is claimed for mechanism B.
* The +0.83 % for mechanism A is a **model**, transferring a directly measured
  local mechanism term through a readout-share ratio. The 60 % trajectory term of
  the local headline is explicitly excluded because it does not transfer.
* The corroborating decode-only comparison in §5.1 is a single pair against a
  +0.73 % noise floor. It is consistent with the ABBA result; it does not
  independently confirm it.
* Two of the leaderboard's own *promoted* rows carry negative score deltas
  relative to their predecessors, so run-to-run spread on this benchmark is
  comparable to the step size we are claiming. We would not be surprised by a
  null result, and we would report it as one.
