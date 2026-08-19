# SDPA route map for the scored decode path

Advisor, 2026-08-19. Read this before you touch `AttentionUtils.swift`, the
width wall, the warm path, or any draft-depth schedule that prices attention.

Every claim here is read from source in this checkout. No claim is measured.
The measurements this map calls for are named at the end.

## Why this map exists

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift` carries the
WIDE-DECODE EXACTNESS CHUNK. Its comment says:

> the fused sdpa vector path serves `qL * gqa <= 32`; above it the dispatch
> changes kernel family and the accumulation order of every score

The chunk therefore fires for `qL >= 6` (because `gqa = 24/4 = 6`, so
`6 * 6 = 36 > 32`).

The `qL * gqa <= 32` rule is real. The `qL >= 6` predicate that was derived
from it is **too wide**. The rule governs one of the three SDPA routes, and
that route needs `kL >= 1024`, which our scored window reaches only in its
final round or two.

## The three routes

Host routing lives in
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`.
That file is **not** in `benchmark.json` `editablePaths`. It is trusted, fixed
host code. We route around it; we do not change it.

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

`qL >= 9` fails `q_pre.shape(2) <= 8` and goes to
`sdpa_full_self_attention_metal`. This is a tiled matmul-style kernel whose
`bq` tile is far wider than nine rows. It is a genuinely different kernel
family with a genuinely different accumulation order.

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

Consequence, for the whole scored window except its final round or two:

- Route 1 is the route.
- `qL = 6, 7, 8` are legal single calls on Route 1.
- The `qL >= 6` chunk splits a legal single call into two calls for no
  kernel-family reason.

## What the chunk costs when it is not needed

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
