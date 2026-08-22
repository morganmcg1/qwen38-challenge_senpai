# E129 — ship the reverted Route B candidate, then template the entry point

PR #128, assignment
`qwen38-r1-e129-ship-the-clean-base-and-template-the-route-b-entry-point-on-m`,
revision `r1`. Base `d46eb29b178c123b6d243127039920872f158440` after the E121
revert merged.

All endpoints below are `harness=local`. Static census endpoints cost zero GPU
seconds and are never timing results. No entry here is an official or ranked
score.

## Rung 0 — the tree that ships

### Rule 55, measured on this branch

```
git diff --stat d46eb29b HEAD -- <89 editablePaths>       (empty)
git diff --stat 2127858b HEAD -- <89 editablePaths>
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift | 650 +, 38 -
```

The submitted surface of this branch is byte-identical to the advisor head.
Against `2127858b`, the tree that rung 5e measured its `off` control against,
exactly one submitted file differs, and it carries Route B plus the E124
island-arm selector. Blob digests at HEAD:

```
quantized.h                 8c56e0e4e2724a8ba798ff04d6f8e90cc4310c9b   pre-E121
mlx-generated/quantized.cpp f3d264775a050f2106d5b38080b397b8e393ac18   pre-E121
Qwen35.swift                f9c272b63fc36b58ffaf9c9fdd53769d2fb43917
```

The E124 selector defaults to `Qwen35IslandArm.all`, is read once in
`Qwen35TextModel.sanitize`, and writes its witness line only when
`DARKBLOOM_QWEN_MTP_ISLAND_ARM` or `MLXFAST_QWEN_MTP_EXACT_QKV_ROWS` is set. On
the default path it installs the same Q, K and V island tensors as before, in
the same complete-permutation branch. It reaches no kernel and no timed round.

### The witness set, and one correction

The three E121 witnesses invert with the revert, as instructed. Verified
against the built worker and against both blobs before use:

| needle | polarity | evidence |
|---|---|---|
| `constexpr bool SHARE_SUMS = NA <= 4;` | forbid | present pre-revert, absent at `2c976286` |
| `threadgroup float sums_xchg[1 * 4 * 32];` | forbid | same |
| `sums[m] += load_vector` | require | absent pre-revert, present at `2c976286` |

**`--require qwen_e120_xsums` cannot be used.** That name exists only inside
`research/e120_census.py`, which wraps the fill body in a synthetic entry point
of that name for the register census. `strings` on the built worker reports 0
copies, so the needle would fail a correct build. The shipped fill pipeline is
`qwen35_custom_affine4_g64_xsums_v1`, and the chain requires that name instead,
together with `qwen35_custom_affine4_g64_qmv_wide_v1` (the Route B replica),
`qwen35_custom_affine4_g64_qmv_wide_sums_v1` and
`inline void qwen_e120_qmv_m(`. All four read 1 copy in the built worker.

### Pre-registration, written before the submission

Base for the prediction is the pre-E121 tree at its mode-corrected score
`3.34136`. The crown is `bc070b7b` at `3.35922017`.

| model | Route B ranked gain | predicted fast | predicted slow |
|---|--:|--:|--:|
| A — uniform transfer of the 5e leg | +4.036 % | 3.4763 | 3.4305 |
| B — ranked width mix | +1.918 % | 3.4054 | 3.3606 |
| C — adverse bracket | +1.321 % | 3.3855 | 3.3409 |

Model C in slow mode is the only cell that misses the crown.

## Rung 1 — per-width entry-point census, `research/e129_entry_point_census.py`

Zero GPU seconds. `xcrun metal-tt` runs the real AGX backend for a named
architecture, wrapped by `research/agx_crossarch.py`. The Metal source is
lifted out of `Qwen35.swift` and the MLX signature generation is reproduced by
`research/e120_g17s_census.py`, which this instrument imports rather than
copies. Output `research/out/e129-entry-point-census.json`, W&B `cym8xztn`.

Residency is `floor(budget / registers)`, budgets 3072 on `applegpu_g16s` and
3968 on `applegpu_g17s`. The budgets are fitted, so the ratio between two
variants is the finding and the absolute count is not.

    Question:        Does templating the Route B QMV entry point on `M` remove
                     the M=5 register maximum from every other width, and is
                     the residency change worth anything at the ranked widths?
    Gate (advisor):  ranked-weighted resident-simdgroup gain below 5 % closes
                     the axis.
    Measured:        g17s +10.11 %, bracket [+7.20 %, +13.37 %].
                     g16s  +7.49 %, bracket [+5.63 %,  +8.19 %].
    Verdict:         PASSES the gate on the ranked architecture, and passes it
                     on the adverse corner of the width mix as well.

### The table

`variant = switch` is the shipped entry point: one pipeline whose register
count is the maximum over its seven inlined branches, so it is the same cell at
every width. `variant = templated` is `M` and `IPG` as template parameters, one
pipeline per width, which is what `MLXFast.metalKernel` compiles and caches per
distinct template value (`metal_kernel.cpp:289-338`). `variant = body` is one
inlined `qwen_e120_qmv_wide<NA>` with no width dispatch above it.

`applegpu_g17s`, the ranked architecture, `sumtable` pipeline:

| M | switch regs / resident | templated regs / resident | templated text B |
|--:|---|---|--:|
| 3 | 102 / 38 | 90 / 44 | 5,662 |
| 4 | 102 / 38 | 94 / 42 | 6,810 |
| 5 | 102 / 38 | 102 / 38 | 7,944 |
| 6 | 102 / 38 | 90 / 44 | 5,662 |
| 7 | 102 / 38 | 94 / 42 | 12,248 |
| 8 | 102 / 38 | 94 / 42 | 6,810 |
| 9 | 102 / 38 | 90 / 44 | 5,648 |

The switch pipeline is 49,718 text bytes. Zero spill in every cell of the whole
census, on both architectures.

`applegpu_g16s`, this host, `sumtable`: switch 96 / 32; templated M=3, 6, 9 at
84 / 36, M=4, 7, 8 at 93 / 33, M=5 at 96 / 32. The `replica_no_table` pipeline
that serves M=3 behaves the same way: g17s switch 101 / 39, templated 89 to 101.

**M is 3 through 9, not 1 through 8.** `Qwen35CustomQMV.widths` is `3...9`; M=1
and M=2 reach other MLX kernels and never enter this entry point. The gate at
`minimumTableWidth = 4` sends M=3 to the `replica_no_table` pipeline and M>=4 to
`sumtable`, so the census weights each width against the pipeline that actually
serves it.

### Three things the table settles

1. **The M=5 branch is the whole tax.** It is the only width whose templated
   register count equals the switch maximum, on both architectures. Templating
   returns 4 to 6 resident simdgroups at every other width and returns nothing
   at M=5.
2. **The switch scaffolding itself is free in registers.** The g17s switch cell
   is 102 registers and the g17s `body NA=5` cell is also 102. The entry point
   costs the maximum over its branches and nothing on top of it. What the
   switch does cost is text: 49,718 bytes against 5,648 to 12,248 per templated
   pipeline, a factor of 4 to 8.8.
3. **M=7 is the expensive templated pipeline.** 12,248 bytes against 6,810 at
   M=8, because `TAIL = 7 % 4 = 3` instantiates a second body at NA=3 beside the
   NA=4 one. It still reads 94 registers, so it keeps the residency gain.

### Weighting, and what is interpolated

Ranked mean verify widths and F83 median weights: beagle 5.382 at 0.4862,
medicine 6.256 at 0.2508, essays 6.087 at 0.1598, botany 7.148 at 0.0124,
republic 5.989 at 0.0100.

Those are means, and the census has only integer rows, so every per-prompt cell
in the JSON is marked `interpolated: true` and the report brackets the
interpolation by sending every prompt to the floor of its mean (adverse) and to
the ceiling (favourable). The bracket is what makes the verdict safe: the gate
holds even in the adverse corner.

**Beagle is the reason the bracket is wide.** Its mean sits at 5.382, on the
one width boundary where the residency step is the largest in the table: M=5
gains nothing and M=6 gains 15.8 % on g17s. Beagle carries 48.6 % of the median
weight. The same boundary halves Route B's own per-width gain. A measured
per-prompt width histogram for beagle would collapse most of this bracket, and
it would do the same for the Route B B-to-C bracket.

### The transfer caution, stated against my own interest

The local fixture runs at mean width 7.359 on `applegpu_g16s`. That cell reads
**+3.12 %**, the smallest number in the whole table. The ranked cell reads
+10.11 %. So a local ABBA on this host measures the weakest corner of the
mechanism, and a null local reading would not falsify the ranked claim. This is
the E121 asymmetry with the sign reversed: E121 looked harmless locally and cost
2.10 % ranked, while templating looks small locally and is largest exactly where
we cannot time it.

Rung 2 therefore cannot be decided by local timing alone. Its ABBA measures
whether the residency gain converts into time at all, on a host where the
residency gain is 3.1 %. If the local leg reads null, the correct conclusion is
"no local effect at the local width mix", not "the mechanism is dead".

## Follow-ups this experiment did not implement

1. **Split the M=5 accumulator group.** M=5 runs one NA=5 group and that single
   branch sets the register maximum. Partitioning 5 as 3+2 would drop its
   branch to NA=3, but it reads the weight rows twice, and the kernel is
   bandwidth bound, so it is a work change and not a codegen change. Price it
   before believing it.
2. **Measure the ranked per-prompt width histogram**, not its mean. It collapses
   the bracket here and the Route B B-to-C bracket at the same time.
3. **Warmup coverage for the templated arm.** `warmAllDepthShapes` warms every
   depth, so the extra JIT compiles land outside the timed window, but rung 2
   must confirm that with the trace before it times anything.
