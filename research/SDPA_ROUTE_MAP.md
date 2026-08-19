# SDPA route map for the scored decode path

Advisor, 2026-08-19. Read this before you touch `AttentionUtils.swift`, the
width wall, the warm path, or any draft-depth schedule that prices attention.

Every claim here is read from source in this checkout. No claim is measured.
The measurements this map calls for are named at the end.

---

## ⛔ CORRECTION BANNER — 2026-08-20 — READ THIS FIRST

**The central conclusion of the original map is REFUTED.** E57 (PR #60, merged,
W&B run `g4efi05h`) measured the routing directly and inspected the deciding
source. Ledger items 185(A), 185(B), and 185(C) carry the full record.

Everything below the banner is retained as a record of the reasoning, not as
guidance. Four specific claims are dead:

1. **The `qL >= 6` chunk predicate is CORRECT, not "too wide."** The original
   map read the route selector at `scaled_dot_product_attention.cpp:685` and
   never reached the function that runs **before** it.
   `ScaledDotProductAttention::use_fallback` at
   **`scaled_dot_product_attention.cpp:591-639`** decides first:
   - `supports_sdpa_full` requires `head_dim` in `{64, 80, 128}`
     (`:625-632`). Our head dimension is **256**, so this is **FALSE at every
     width**.
   - `supports_sdpa_vector` requires `qL * gqa <= 32` (`:634-637`) with **no
     `kL` condition whatsoever**. With `gqa = 6` this caps the fused vector
     path at **`qL <= 5`, unconditionally.**

   The rule the original comment quoted is therefore not a route-2 detail that
   needs `kL >= 1024`. It is the **top-level gate on all fused attention**.

2. **The chunk is a DISCOUNT, and it is load-bearing.** Measured dispatches per
   SDPA call (`research/out/e57-rung1/`):

   | shape | dispatches |
   |---|---|
   | unsplit `qL <= 5`, `kL < 1024` | 1 |
   | unsplit `qL <= 5`, `kL >= 1024` | 2 |
   | **unsplit `qL` 6..9** | **8** |
   | chunked `qL` 6..9, `kL < 1024` | 4 (headTransposed) / 6 (contiguous) |
   | chunked `qL` 6..9, `kL >= 1024` | 6 / 8 |

   The unsplit path at `qL >= 6` is an **8-dispatch composed fallback**
   (`arangeint32` x2, `sv_Multiply`, `g2_GreaterEqual`, `steel_gemm_fused_nt`,
   `g2_Select`, `block_softmax_precise`, `steel_gemm_fused_nn`) that
   materialises a `24 x qL x kL` bf16 score tensor. The chunk splits into a
   5-row and a `(qL - 5)`-row call, **both of which are `<= 5`**, so both pass
   `supports_sdpa_vector` and stay fused. Removing the chunk **adds** about 4
   dispatches per full-attention layer, roughly **64 per round**. The original
   map's cost estimate had the **sign backwards**. Arm C (chunk off) fired
   **10957** SDPA dispatches versus the base's **6163**, and it **failed
   correctness** (`rejected_tail_diverged` at step 300, round 37, declared
   top-1 2523 versus reference 248045 at margin **0.125 = 12.5x** the `1e-2`
   `referenceMargin`).

3. **The `qL >= 9` steel / `_nax` route never fires.** Because
   `supports_sdpa_full` is false at head dimension 256, `sdpa_full_self_attention_*`
   is unreachable on this model at every width. The measured proof: unsplit
   `qL = 6, kL = 1030` returned with **8 dispatches**, not a throw and not a
   steel kernel (`research/out/e57-rung1/throw.log:829-830`). Outcome 3 of the
   original plan — a threadgroup-limit throw at `qL >= 6` — is falsified; no
   such throw is reachable. The editability of `steel_attention.cpp` and
   `steel_attention_nax.cpp` is therefore **irrelevant to the scored path**.

4. **`kL >= 1025` is UNREACHABLE, so the `blocks = 128` promotion is dead
   code for us.** A 512-token seed plus 512 generated tokens caps `kL` at
   **exactly 1024**. Measured across all three E57 arms: calls with
   `qL >= 6 && kL >= 1024` = **0 in every leg**; `blocks_64_calls = 16`,
   `blocks_128_calls = **0**`. Only **1 of 76 rounds** reaches `kL = 1024`, and
   it does so at **`qL = 4`**.

   Consequence: the frontier's `warmTargetLaterWindowSDPA`, which pads to
   `kL == 1024`, **warms the only reachable boundary variant**. The frontier is
   correct and my earlier 182(E)/183(E) claim that it "warmed the wrong
   pipeline" is **refuted**.

### One live risk this correction exposes

The base itself is not bit-stable at the `kL = 1024` boundary. E57 Arm A — the
**shipped, passing** base — declared two distinct top-two tuples at positions
**1022** and **1024**, both in round 76, the single round that reaches
`kL = 1024`, at **`qL = 4` with no chunk**. Position 1022 agrees on top-1
(6009) but reports top-2 31098 versus 98138; position 1024 reports
`0x1.4ap+4` versus `0x1.4ep+4`. This tracks the `sdpa_vector` one-pass to
two-pass transition at `:746-753`.

**The ranked 512 + 512 window always reaches this boundary.** That is a latent
near-tie exposure on hidden prompts, in code we ship today, and it is not
caused by the chunk.

### What survives

- The bit-exactness argument for the chunk *on the fused one-pass route*
  (`sdpa_vector.h:15-176`, no reduction crosses query rows, bottom-right
  aligned causal predicate) is **sound and confirmed**: chunk versus unsplit
  showed **0 differing elements at `qL = 5`**, and the A/A control showed 0
  differing elements at every width. At `qL` 6..9 about 63 % of elements
  differ, max absolute 1.95e-3 to 4.88e-3 — because there the comparison is
  fused-vector against the composed fallback, which is a genuine kernel-family
  change.
- The function-constant and `hash_name` inventory in Route 2 and Route 3 below
  is accurate as source reading.
- The `kL = 512 + tokensCommitted + M` derivation is accurate. Its
  **conclusion** about which route serves the window is wrong.

---

## Why this map exists

> ⛔ **Refuted premise. Retained for the record.** See the correction banner.
> The corrected statement is: `AttentionUtils.swift`'s chunk predicate is
> right, the fused path ends at `qL = 5`, and the chunk is a dispatch discount
> that keeps both halves fused.

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift` carries the
WIDE-DECODE EXACTNESS CHUNK. Its comment says:

> the fused sdpa vector path serves `qL * gqa <= 32`; above it the dispatch
> changes kernel family and the accumulation order of every score

The chunk therefore fires for `qL >= 6` (because `gqa = 24/4 = 6`, so
`6 * 6 = 36 > 32`).

**[REFUTED]** The `qL * gqa <= 32` rule is real. The `qL >= 6` predicate that
was derived from it is **too wide**. The rule governs one of the three SDPA
routes, and that route needs `kL >= 1024`, which our scored window reaches only
in its final round or two.

**[CORRECT READING]** The `qL * gqa <= 32` rule is the top-level gate in
`use_fallback` at `:634-637`, it carries no `kL` term, and it makes `qL >= 6`
exactly the right predicate. The comment in `AttentionUtils.swift` was right
all along; this map misread which function enforced it. Note that
`Qwen36MTPBlockSession.swift:670-699` contains two paragraphs that contradict
each other on this point — E57 settles it in favour of the **second**.

## The three routes

> ⛔ **Incomplete.** There is a **fourth** path, and it is the one that runs at
> `qL >= 6`: the **8-dispatch composed fallback** taken when `use_fallback`
> returns true. The three routes below are the three *fused* routes, and all
> three require `qL <= 5` on this model. Read `use_fallback` at
> `:591-639` **before** the selector at `:685`.

Host routing lives in
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`.
That file is **not** in `benchmark.json` `editablePaths`. It is trusted, fixed
host code. We route around it; we do not change it.

**The deciding gate, which this map originally missed** (`:591-639`):

```cpp
// supports_sdpa_full  (:625-632)   requires head_dim in {64, 80, 128}
//                                  -> FALSE for us, head_dim == 256
// supports_sdpa_vector (:634-637)  requires qL * gqa <= 32
//                                  -> with gqa == 6, requires qL <= 5
// if neither holds, use_fallback() == true and the composed 8-dispatch
// fallback runs. No kL term appears in either condition.
```

Only if `use_fallback` returns false do we reach the selector below.

`:685` selects vector mode with no GQA term at all:

```cpp
// We are in vector mode ie single query
if (q_pre.shape(2) <= 8) {
```

`:746-753` then splits vector mode in two:

```cpp
bool do_causal = do_causal_ && q.shape(2) > 1;
char devc = d.get_architecture().back();
if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
    (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
  sdpa_vector_2pass(...);
} else {
  sdpa_vector(...);
}
```

Our GQA geometry is `k.shape(1) = 4`, `q.shape(1) = 24`, so `gqa_factor = 6`.
The second clause needs `kL >= 4096` and never fires: our `kL` tops out near
1024.

### Route 1 - `sdpa_vector`, one pass

Dispatch (`:341-360` region):

```cpp
MTL::Size group_dims(1024, 1, 1);
MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);
```

The threadgroup is a fixed 1024 threads. `qL` enters `grid_dims.y` only.
**Nothing in this route depends on `qL * gqa`.** `qL = 8` is as legal as
`qL = 1`.

Pipeline identity (`sdpa_vector.h:7-13`) is the function constants
`has_mask`, `query_transposed`, `do_causal`, `bool_mask`, `float_mask`,
`has_sinks`. None of them is a function of `qL`. **One pipeline serves every
`qL` on this route.**

### Route 2 - `sdpa_vector_2pass`, two passes

Dispatch (`:418-484`):

```cpp
int gqa_factor = q.shape(1) / k.shape(1);
int n_simds = gqa_factor * q.shape(2);
...
MTL::Size group_dims(32, gqa_factor, q.shape(2));
MTL::Size grid_dims(k.shape(1), q.shape(0), blocks);
```

The threadgroup is `32 * gqa_factor * qL = 32 * n_simds` threads. Metal caps a
threadgroup at 1024, so this route requires `n_simds <= 32`, that is
`qL * gqa <= 32`. **This is the only place the chunk comment's rule lives.**

At `gqa = 6`:

| `qL` | threads per threadgroup | legal |
| --- | --- | --- |
| 1 | 192 | yes |
| 4 | 768 | yes |
| 5 | 960 | yes |
| 6 | 1152 | **no** |
| 7 | 1344 | **no** |
| 8 | 1536 | **no** |

The violation is not silent. `utils.h:84-96`
`check_kernel_threadgroup_size` compares `group_dims` against the pipeline's
`maxTotalThreadsPerThreadgroup` and calls `throw std::runtime_error`. An
unguarded `qL >= 6` call on this route aborts the run with a readable message.

`blocks` is function constant 26 **and** is appended to `hash_name`, so each
distinct `blocks` value is a distinct pipeline that must be created:

```cpp
{&blocks, MTL::DataType::DataTypeInt, 26},
...
hash_name += std::to_string(blocks);
```

On `devc == 's'`, `blocks = 64`, promoted to `128` when
`N > 1024 && n_simds > 4`. `n_simds = 6 * qL > 4` for every `qL >= 1`, so:

| `kL` | `blocks` on `'s'` | pipeline |
| --- | --- | --- |
| exactly 1024 | 64 | A |
| 1025 and above | 128 | B |

On `devc == 'd'`, `blocks = 128` throughout our range. `MLX_SDPA_BLOCKS`
overrides the value; do not set it in a timed arm.

### Route 3 - full attention, `steel_attention`

> ⛔ **DEAD PATH.** `supports_sdpa_full` requires head dimension in
> `{64, 80, 128}` (`:625-632`) and ours is **256**, so `use_fallback` returns
> true long before `q_pre.shape(2) <= 8` is tested. `sdpa_full_self_attention_*`
> is unreachable on this model at every width. Measured: unsplit `qL = 6`,
> `kL = 1030` returned 8 composed-fallback dispatches, no steel kernel and no
> throw. `steel_attention.cpp` and `steel_attention_nax.cpp` are editable and
> irrelevant.

`qL >= 9` fails `q_pre.shape(2) <= 8` and goes to
`sdpa_full_self_attention_metal`. This is a tiled matmul-style kernel whose
`bq` tile is far wider than nine rows. It is a genuinely different kernel
family with a genuinely different accumulation order.

### Route 4 - the composed fallback (the one that actually runs at `qL >= 6`)

Added 2026-08-20 from E57 measurement. When `use_fallback` returns true, MLX
does not dispatch an attention kernel at all. It composes attention out of
eight primitive dispatches:

```text
arangeint32, arangeint32, sv_Multiply, g2_GreaterEqual,
steel_gemm_fused_nt, g2_Select, block_softmax_precise, steel_gemm_fused_nn
```

The two `arange` and the `GreaterEqual`/`Select` pair build the causal mask on
the fly. `steel_gemm_fused_nt` materialises the full `24 x qL x kL` bf16 score
tensor, `block_softmax_precise` normalises it, and `steel_gemm_fused_nn`
applies it to V. At `qL = 9`, `kL = 1024` that intermediate is about 442 KB per
layer per call, and there are 16 full-attention layers.

This is the path the chunk **avoids**. That is why the chunk is a discount.

## What `kL` actually is in the scored window

The seed is 512 tokens and the parent counts 512 generated tokens. After the
seed, cache length is 512. During a verify round of width `M`,
`cache.update()` appends `M` rows, so

```text
kL = 512 + tokensCommitted + M
```

`kL >= 1024` therefore needs `tokensCommitted + M >= 512`: **the last round,
or the last two.** Prefill itself runs at `qL = 512`, which is Route 3, so no
decode-shaped vector pipeline is created there.

**[CORRECTED 2026-08-20]** The derivation above is right and the measurement
sharpens it: `kL` reaches **exactly 1024** and never exceeds it, in **1 round
of 76**, and that round runs at **`qL = 4`**. So `qL >= 6 && kL >= 1024` has
measured frequency **zero**, and `kL >= 1025` is unreachable. Prefill at
`qL = 512` does not take Route 3 either — it takes the composed fallback, like
every other `qL >= 6` shape.

Consequence, for the whole scored window except its final round or two:

- **[REFUTED]** Route 1 is the route.
- **[REFUTED]** `qL = 6, 7, 8` are legal single calls on Route 1.
- **[REFUTED]** The `qL >= 6` chunk splits a legal single call into two calls
  for no kernel-family reason.

**[CORRECT]** Route 1 serves `qL <= 5` only. At `qL` 6..9 the unsplit call
leaves the fused family entirely and costs 8 dispatches. The chunk turns that
into two fused calls at 4-6 dispatches. There is a kernel-family reason, it is
the whole reason, and the chunk is on the cheap side of it.

## What the chunk costs when it is not needed

> ⛔ **SIGN ERROR. This entire section has the cost backwards.** The chunk does
> not cost dispatches; it **saves** about 4 per full-attention layer, roughly 64
> per round. Removing it measured **+77.8 % SDPA dispatches** (6163 -> 10957)
> and **failed correctness**. Retained to show how the error was made: the
> section prices "one call becomes two calls" while the real comparison is
> "eight dispatches become four."

Read the code, not the comment. Per full-attention layer per round, at
`qL in {6, 7, 8}`:

```swift
let outA = MLXFast.scaledDotProductAttention(
    queries: queries[0..., 0..., 0 ..< split, 0...],
    keys: cachedKeys[0..., 0..., 0 ..< kSplit, 0...], ...)
let outB = MLXFast.scaledDotProductAttention(
    queries: queries[0..., 0..., split..., 0...],
    keys: cachedKeys, ...)
return concatenated([outA, outB], axis: 2)
```

1. **Two query copies.** A row slice of a row-contiguous `[1, 24, qL, 256]`
   query is not row-contiguous, and `q_copy_unless` at `:686-698` rejects it:
   with `shape[0] == 1` it requires `strides[2] == shape[3] * shape[1] = 6144`,
   while the slice has `strides[2] == 256`. Both slices take
   `contiguous_copy_gpu`. The unchunked call passes the original query and is
   very likely copy-free.
2. **One extra SDPA dispatch.**
3. **One `concatenated` kernel** over the full `[1, 24, qL, 256]` output.

The comment claims the cost is "one more pass over the KV rows (a few MB)".
**That part of the comment is wrong, in our favour.** Key traffic scales with
the number of `(head, row)` threadgroups either way: unchunked reads
`24 * qL * kL`, chunked reads `24 * 5 * kSplit + 24 * (qL - 5) * kL`, which is
strictly *less*. There is no extra KV pass. The real overhead is the copies,
the concat, and the dispatch count.

Rough size, at `qL = 6`, 16 full-attention layers, fp16:

| item | bytes per layer, read + write |
| --- | --- |
| query slice A copy, 24 x 5 x 256 | 123 KB |
| query slice B copy, 24 x 1 x 256 | 25 KB |
| concat output, 24 x 6 x 256 | 147 KB |
| total | ~295 KB |

Times 16 layers is about 4.7 MB per round, plus 64 extra dispatches per round
(16 layers x [2 copies + 1 sdpa + 1 concat]). At a few hundred GB/s the
traffic is only about 12 microseconds; the dispatch count is the term to fear.

## Calibrating the dispatch-overhead prize against the board

> ✅ **VINDICATED 2026-08-20, against my own later objection.** This section's
> reading of `warmTargetLaterWindowSDPA` is **correct**, and it is the one part
> of the map that E57 confirmed rather than refuted.
>
> `kL = 1024` is the **only** reachable Route 2 boundary in a 512 + 512 window,
> it is touched in exactly **1 round of 76**, and that round is the **last**
> one — so the pipeline is otherwise created inside the timed leg, exactly as
> claimed here. Measured: `blocks_64_calls = 16`, `blocks_128_calls = 0`. The
> frontier's `qL in {1, 5, 4}` choice is also exactly right: `1` is the serial
> step, `5` is chunk A, `4` is chunk B of width 9 — and 4 and 5 are the only
> widths that survive `supports_sdpa_vector`.
>
> **I was wrong later, not here.** Ledger 182(E) and 183(E) claimed the frontier
> "warmed the wrong pipeline" and proposed extending the warm to `kL >= 1025` to
> reach `blocks = 128`. That is **unreachable**: 512 + 512 caps `kL` at exactly
> 1024. The extended-warm proposal is deleted from the compose list. This
> section had it right the first time.
>
> The retraction inside the section — that +0.0173 % is 0.12 median sd and
> cannot price the mechanism — also stands, and remains the correct standard.

The promoted frontier submission `59b321e` at 3.24985583421771 is, in its
entirety, 70 added lines in `Qwen36MTPBlockSession.swift`:
`warmTargetLaterWindowSDPA`, which host-extends throwaway full-attention K/V
to `kL >= 1024`, dispatches attention at `qL in {1, 5, 4}`, and discards the
results. `git diff --stat 0c90733 9e1ff9e` is one file, 70 insertions, 0
deletions.

This map explains that submission exactly. Warming `kL >= 1024` creates the
**Route 2** pipelines - which, per the table above, are otherwise first
touched inside the timed window, in its final round. `qL = 1` is the serial
step, `qL = 5` is chunk A, `qL = 4` is chunk B of width 9.

It scored **+0.0173 %** over its base. **That number must NOT be used to price
the mechanism, and an earlier revision of this document did exactly that.**
`research/board_noise_identification.py` measures the published median's
relative standard deviation at **0.1415 %** (worst case 0.2636 %), so
+0.0173 % is **0.12 median sd** — an unresolvable draw, not a measured effect.
The derived claim that "a pipeline-creation miss inside the scored window is
worth roughly 0.02 %" was circular and is retracted.

The mechanism's own leg evidence is better than its board score. Across the
eight prompts the frontier's candidate leg improved on **7 of 8** against its
parent, median `mtp_spt` 11.7397 -> 11.7277 ms (**-0.102 %**). The published
delta is small only because the pinned serial numerator happened to fall
0.248 % on `medicine`. Direction and leg-level consistency support the
mechanism; the board magnitude is luck.

Two things follow.

**`qL` IS NOT IN THE PIPELINE IDENTITY. An earlier revision of this document
claimed the frontier warm was incomplete because it omits `qL = 2` and
`qL = 3`. That claim was WRONG and is retracted here.** Verified against
`scaled_dot_product_attention.cpp`:

- `kname` for Route 2 is `"sdpa_vector_"` + dtype + `"_"` + `q.shape(-1)` +
  `"_"` + `v.shape(-1)` (`:340-348`), and for Route 3
  `"sdpa_vector_2pass_1_"` + the same three fields (`:429-437`). **Neither
  carries `qL` or `kL`.**
- `hash_name` appends only the mask mode, `_qt`/`_qnt`, `_c`/`_nc` and
  `_sinks`/`_nosinks` (`:375-378`); the function-constant list is
  `has_mask`(20), `query_transposed`(21), `do_causal`(22), `bool_mask`(23),
  `float_mask`(24), `has_sinks`(25) (`:366-373`). **No `qL` term.**

So warming a single `qL = 4` shape already creates the pipeline that `qL = 2`
and `qL = 3` use. `qL` reaches pipeline identity through exactly two indirect
paths, and only two:

1. `:746` `bool do_causal = do_causal_ && q.shape(2) > 1;` forces
   `do_causal = false` at `qL == 1` whatever mask the caller passes. So a
   `qL = 1, .causal` warm and a live `qL = 1, .none` call select the **same**
   `_nc` pipeline, and every `qL >= 2` causal call selects `_c`. Two pipelines,
   not one per width.
2. `blocks` on Route 3 only (see below).

The frontier's `qL in {1, 5, 4}` is therefore itself redundant: `{1, 4}`
covers the same two pipelines.

**One genuine gap remains, and it is the `blocks` axis.** `blocks` is function
constant 26 and is appended to `hash_name`, so each value is a distinct
pipeline. On `devc == 's'`, `:446-458` sets `blocks = 64`, promoted to `128`
when `N > 1024 && n_simds > 4`; `n_simds = gqa_factor * qL = 6 * qL >= 6`, so
the `n_simds` clause is always satisfied and the condition reduces to
`kL > 1024`. The frontier pads to **exactly** 1024, so it warms `blocks = 64`
and never `blocks = 128`. The live window runs `kL = 512 + tokensCommitted + M`,
which takes both `kL == 1024` and `kL > 1024`. **Warming one `kL == 1024` shape
and one `kL >= 1025` shape is a strictly additive superset of the frontier's
own mechanism, at zero fidelity risk.** Null if the ranked host reports
`devc == 'd'`, where `blocks = 128` throughout our range — so the arch letter
must be reported before this is priced.

**Both mechanisms are live, and they are separable but confounded.** An earlier
revision asserted the chunk removal is worth more than the warm import. That
ranking rested on the retracted 0.02 % figure and is withdrawn.

The evidence that actually discriminates is the shape of our deficit. Our best
submission `ca9251b8` carries a `mean_draft_len` 8-tuple **identical to the
frontier's to four decimal places**, so the accept trajectory is not the
difference; our candidate leg is simply slower on **8 of 8** prompts. And the
excess scales with draft depth:

| prompt | our `mtp_spt` excess | `mean_draft_len` |
|---|---|---|
| `essays` | +0.814 % | 5.43 |
| `republic` | +0.594 % | 5.27 |
| `botany` | +0.498 % | 5.78 |
| `beagle` | +0.454 % | 4.53 |
| `medicine` | +0.291 % | 4.77 |
| `travel` | +0.080 % | 2.66 |
| `plutarch` | +0.020 % | 0.15 |

Two hypotheses fit that ordering, and both predict roughly the same total
(~23 ms on a ~6.2 s leg):

- **H1, warm coverage.** A one-off pipeline miss whose *count* rises with the
  number of distinct shapes the window visits. Deep-drafting prompts visit more
  widths, so they pay more misses.
- **H2, chunk overhead.** A per-round cost of 2 query copies + 1 extra SDPA
  dispatch + 1 concat per full-attention layer, over 16 layers, on every
  width-6-to-8 round. Deep-drafting prompts have more such rounds.

**`mean_draft_len` and `1 / leg_time` are collinear on this pool** — deep
drafting is what makes a leg fast — so per-prompt board data cannot separate
H1 from H2. They are independent code changes, both zero-arithmetic, so measure
them separately and together rather than arguing the ranking.

## The correct predicate

> ⛔ **REFUTED, AND THE SHIPPED PREDICATE IS ALREADY CORRECT.** Do not apply the
> narrowed predicate below. It would push widths 6, 7, 8 onto the composed
> fallback: **more** dispatches, a materialised score tensor, and a
> demonstrated correctness failure. E57 Arm B implemented exactly this
> narrowing (chunk restricted to `qL == 9`) and measured **+36.1 % dispatches**,
> **+0.27 %** seconds per token, and **396 of 512 positions** moving their
> declared top-two row evidence with 18 changing a top-two id — while the
> token-match line stayed green. Keep `qL >= 6, qL <= 9` exactly as shipped.

Route 3 avoidance still needs a chunk at `qL = 9`: unchunked, `qL = 9` leaves
vector mode entirely. Route 2's thread cap still needs a chunk at
`qL >= 6` when `kL >= 1024`, or the run throws. Nothing needs a chunk at
`qL in {6, 7, 8}` when `kL < 1024`.

```swift
if queries.dim(0) == 1, kL >= qL, case .causal = mask,
   qL >= 6, qL <= 9,
   qL >= 9 || kL >= 1024
{
    // existing split at row 5
}
```

Keep the `kL >= 1024` arm unconditional rather than testing the architecture
letter. It costs one round and it is the arm that prevents a thrown run on a
`'d'` or `'s'` host.

## Why the narrowed predicate should be bit-exact

`sdpa_vector.h:15-176`. Each `(q_batch_head_idx, q_seq_idx)` pair is its own
threadgroup; **no reduction ever crosses query rows.** The causal predicate is
bottom-right aligned:

```cpp
const int q_seq_idx = tid.y;
...
if (do_causal) {
  use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
}
```

with `tpg.y == qL`, so row `r` of a `qL`-row block attends keys
`0 ... kL - qL + r`. The per-thread key loop is
`for (int i = simd_gid; i < N; i += BN)` with `BN = 32`, and a masked key
leaves `max_score`, `sum_exp_score` and `o[]` untouched.

So for a fixed query row, the ordered sequence of contributing keys seen by a
given thread is exactly "absolute key indices congruent to `simd_gid` mod 32,
not greater than `kL - qL + r`, ascending". Now compare:

- **Chunk A**, rows `0..<5`, `N = kSplit = kL - (qL - 5)`. Row `r` attends
  `0 ... kSplit - 5 + r = kL - qL + r`. Same set, same absolute indices, same
  order. The smaller loop bound removes only iterations that were masked.
- **Chunk B**, rows `5...`, `N = kL`, block height `qL - 5`. Row `r'` attends
  `0 ... kL - (qL - 5) + r'`, and with `r = r' + 5` the unchunked row wants
  `0 ... kL - qL + r' + 5`. Identical.
- **Unchunked**, `N = kL`, block height `qL`. Identical to both by
  construction.

Chunked and unchunked are therefore bit-identical **whenever both use
Route 1**. This is a prediction from source, and it is exactly what the
experiment must falsify or confirm.

The one place it can break is the Route 1 / Route 2 seam. Chunk A passes
`kSplit = kL - (qL - 5)`, up to 4 less than `kL`. In the band
`kL in [1024, 1027]` on a `'d'` or `'s'` host, chunk A can sit at
`kSplit <= 1023` on Route 1 while the unchunked call sits on Route 2. The two
routes have different reduction trees, so bits may differ there. The
recommended predicate keeps chunking whenever `kL >= 1024`, which closes that
seam by construction.

## Measurements this map asks for

> ✅ **ALL FIVE WERE RUN, in E57. The map's hypothesis lost; the measurement
> programme was sound.** Results, in order:
>
> 1. Exactness gate with a positive control: **run.** The A/A control returned 0
>    differing elements at every width, proving the comparison could detect a
>    change; chunk-off then **failed** `rejected_tail_diverged`. The predicted
>    `Maximum threads per threadgroup` throw **did not occur** and is not
>    reachable — the fallback composes instead of throwing.
> 2. Architecture letter and generation: **`applegpu_g16s`**, `devc == 's'`,
>    generation **16**, `_nax` unavailable (needs >= 17), max threads per
>    threadgroup 1024, max threadgroup memory 32768 B. Route 2 **is** reachable
>    here, and it fires in exactly 1 round of 76, at `qL = 4`.
> 3. Dispatch counts: **counted, not inferred.** The predicted delta was "4
>    fewer dispatches at widths 6-8." Measured: **4 more.** Sign error.
> 4. Matched absolute candidate seconds per token: **+0.27 %** for the narrowed
>    predicate. A loss.
> 5. `kL >= 1024` guard exercised: **yes, once per leg**, and it never coincides
>    with `qL >= 6`.
>
> The closing estimate below ("around 0.1 %") was the right order of magnitude
> for the *effect size* and the wrong **sign**. Note that the reasoning behind
> it — that dispatch overhead is worth chasing — was later vindicated from the
> opposite direction: the 22 us/dispatch figure this experiment produced is now
> the basis of E58, and under the ledger-186(D) transfer law it prices at about
> **2.0 % of ranked score**, roughly 7 sd. The map asked the right question
> about dispatches and got the arithmetic backwards.

1. **The exactness check is the gate, and it must be able to fail.** Run the
   full 512-token public-golden check with the narrowed predicate, including
   post-EOS continuation and row-ledger closure. Then run a positive control:
   force the chunk off for `qL >= 6` *including* `kL >= 1024` and confirm the
   run throws the `Maximum threads per threadgroup` error, or that tokens
   diverge. A gate that cannot fail is not a gate.
2. **Report the architecture letter and generation** actually seen by
   `get_architecture()` on the student host, and say whether Route 2 is
   reachable there at all. If the local host is not `'d'` or `'s'`, the local
   run never touches Route 2 and the `kL >= 1024` guard is untested locally -
   say so rather than implying coverage.
3. **Count the dispatches, do not infer them.** Report SDPA calls, query
   copies and concat kernels per round, before and after, at each of
   `qL = 6, 7, 8, 9`. The predicted delta is 4 fewer dispatches per
   full-attention layer at widths 6 to 8 and no change at width 9.
4. **Matched absolute candidate seconds per token** against a fresh
   unchanged-base run on the same host, same token window, same head, ABBA
   counterbalanced. The causal path is confined to the candidate MTP leg, so
   the local serial-to-MTP ratio is also admissible evidence here - but report
   absolute candidate time as the headline.
5. **Assert the guard fires.** A counter proving `kL >= 1024` chunking
   happened at least once in a 512-token run, so the guard is exercised rather
   than merely present.

The local end-to-end null floor is **0.0629 %**. My point estimate for
narrowing the predicate is around 0.1 %, dominated by dispatch overhead on
width-6-to-8 rounds, with real uncertainty in both directions. If it lands
above the floor it is roughly six times the frontier's last published step.

## What this map does not claim

- It does not claim the original chunk was wrong to exist. At `qL = 9` it is
  load-bearing, and at `kL >= 1024` it prevents a thrown run.
- It does not claim the recorded top-2 value drift at the width wall was
  imaginary. It claims the drift can only come from a route change, that the
  route change needs `kL >= 1024` or `qL >= 9`, and that whoever measured the
  drift should be able to say which of those they were in.
- It does not measure anything. Every number above the "Measurements" heading
  is either read from source or an explicit order-of-magnitude estimate,
  labelled as such.
