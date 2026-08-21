# E87 §8 — replacing the arm C selection chains

Design note only. Nothing here is implemented. Written while submission
`84b9ef7b` was validating, so the next assignment can start from a settled plan.

Source read: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`,
selection sites at `:4216`, `:4214` and `:4229`.

## The three chains, and what each one is for

Arm C's proposal path spends about 113.78 microseconds per draft in three
generic selection chains. They are not the same problem.

| chain | call site | keys | selects | measured | ns/key |
|---|---|---:|---:|---:|---:|
| A centroid probe | `:4216` `argPartition(centroidScore, kth: clusters - probes)` | 12,292 bf16 | 3,073 | 41.80 us | 3.401 |
| B index order | `:4214` `MLX.sorted(...)` | 3,073 uint32 | — | 24.40 us | 7.940 |
| C row top-32 | `:4229` `argPartition(rowScore, kth: ...)` | 24,584 bf16 | 32 | 47.58 us | 1.935 |

The declared path selects top-32 from 98,304 rows in 38.19 microseconds with the
custom `qwen35DraftTop32` kernel, which is 0.388 ns/key. Every chain above is
between 5x and 20x that cost per key.

## Chain C is a parameterisation, not new work

`qwen35DraftTop32` (`:3533`) is a two-stage tiled partial-plus-finalize kernel
built from a source string whose shape constants are interpolated at
construction: `REAL_COUNT` from `qwen35Top32RealCount = 98_330` (`:3193`) and
`TOPK` from `qwen35Top32K = 32` (`:3194`).

Chain C is the identical problem at a different width: top-32 from 24,584 bf16
keys. Making the kernel factory take `(realCount, topK)` and instantiating a
second copy at 24,584 is a parameterisation change. At the declared per-key cost
this is 24,584 x 0.388 ns = 9.5 us, so the saving is about **38.1 us/draft**.

This is the highest-confidence part of §8 and it should be built first.

## Chains A and B should be fused, and that is the real win

The advisor priced B as an optional extra. Reading the source, B is not an extra
at all — it is a direct consequence of how A returns its answer, and the right
fix deletes it rather than accelerating it.

The comment at `:4213` states the constraint exactly:

> `gatherQuantizedMM`'s sorted fast path needs indices in value order, while the
> top-C arrive in score order.

So chain A produces 3,073 indices in score order, and chain B exists only to put
them back in ascending index order for `sortedIndices: true`.

Chain A is also the one place where the existing top-k kernel does **not**
transfer. It selects 3,073 of 12,292, which is a 25 % selection, not a small
top-k; a tiled top-K structure with K = 3,073 per tile is the wrong algorithm.

Both facts point at one replacement: a **threshold-and-compact** pass.

1. Find the value `t` that is the 3,073rd largest centroid score. A histogram or
   radix pass over 12,292 bf16 keys does this in one or two cheap sweeps.
2. Sweep the scores in ascending index order and append every index whose score
   passes `t`.

Step 2 emits indices **already in ascending index order**, which is precisely
what `sortedIndices: true` wants. Chain B disappears; it is not replaced by a
faster sort, it is replaced by nothing. Combined saving for A and B is about
**66.2 us/draft**, of which 24.40 us is pure deletion.

Total for all three chains is about **99.5 us/draft**, which reproduces the
advisor's 99.47 us figure by an independent route.

## The exactness constraint, stated precisely

This must be handled explicitly, because it is the one way the change could go
wrong quietly.

Changing the probe set cannot change an emitted token. Stage 2 rescores the
shortlist against exact affine-4 weights and the trusted parent verifies every
row, so a different probe set is an acceptance question, not a correctness
question.

But the advisor's claim is stronger: exactness-preserving **by construction**,
meaning bit-identical proposals. To earn that, the replacement must return the
same 3,073-element set and the same 32-element set as `argPartition`, including
behaviour at ties. bf16 scores over 12,292 keys make ties at the threshold
likely, not hypothetical.

Rule to adopt: break ties by lowest index. An ordered compaction gives that for
free in step 2, and it is stable and deterministic. The count must then be
clamped to exactly 3,073 when more than 3,073 keys pass `t`, dropping the
highest-index ties.

Required gate before any timing: a positive control that compares the new
selection against `MLX.argPartition` on real bf16 score rows captured from a
live draft, not synthetic integers, and that is shown to fail when the
tie-break is deliberately inverted. The tree already has the right harness shape
for this at `:3554-3589`, which benchmarks and cross-checks the existing kernel
against `MLX.argPartition` and is documented as never running on a scored path.

## Value, priced with the measured transfer law

The advisor's §5 transfer factor is 0.30, measured from one mechanism on one
ranked pair.

| target | us/draft | % of round, k-adjusted | published at 0.30 |
|---|---:|---:|---:|
| chain C alone | 38.1 | 0.162 % | +0.049 % |
| chains A and B | 66.2 | 0.281 % | +0.084 % |
| all three | 99.5 | 0.422 % | +0.127 % |

This is below the 0.55 % published detection floor, so it is a rider. Its
strategic value is that it removes 22 generic dispatches per draft, and fixed
per-dispatch cost is the class the transfer law says grows in relative terms on
the faster ranked machine.

## Suggested order

1. Parameterise the top-32 kernel and instantiate it at 24,584. Cheapest, safest,
   38.1 us.
2. Build the threshold-and-compact centroid selection and delete the sort. 66.2
   us, and it removes an algorithm that never fitted the problem.
3. Gate both against `MLX.argPartition` on captured bf16 rows with an inverted
   tie-break positive control, then run one matched timing session.
