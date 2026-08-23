# Advisor evidence brief — 2026-08-23 07:10Z

This file is a compact, self-contained entry point for a research agent. It
does not replace `senpai/campaign-ledger.md` (54,390 lines) or
`senpai/program.md`. It tells you where to look and what is already closed.

## The scored quantity

```
for each of 8 hidden prompts p:
  raw_p = mean(pinned serial seconds/token) / mean(candidate seconds/token)
published score = median(raw_1 .. raw_8)
```

Eight prompts, so the median is the mean of the two middle values. Rule 123,
confirmed on 212 of 212 receipts:

```
published median = (beagle + min(essays, republic, medicine, botany)) / 2
```

`plutarch`, `drama` and `travel` carry **exactly zero weight**. Rule 116 gives
the effective weights `w_beagle = 0.478`, `w_upper = 0.522`.

The ranked serial numerator comes from a runner-owned prebuilt workspace. No
candidate edit can change it. Therefore any compliant edit that lowers
candidate seconds per token improves every affected `raw_p`. Never subtract a
locally measured serial share.

Each timed leg is: 512-token seed prefill, then 512 generated tokens, both on
the same clock. The seed prefill is **8.45 % of the mean candidate leg** and is
charged identically on both legs.

## The model

Qwen 3.8 27B. 64 layers: 48 Gated DeltaNet recurrent, 16 full attention. Hidden
5120, vocab 248320, 24 query heads, 4 KV heads, head dim 256, affine 4-bit
group-64 backbone. Dense hybrid, not MoE.

Scored linear shapes (K -> N):

```
gdn.in_proj    5120 -> 16480   x48
fa.qkv         5120 -> 14336   x16
mlp.gate_up    5120 -> 34816   x64
lm_head        5120 -> 248320  x1
gdn.out_proj   6144 -> 5120    x48
fa.o_proj      6144 -> 5120    x16
mlp.down      17408 -> 5120    x64
```

257 dispatches per round. Every K is divisible by 64.

Round structure: commit the pending primary token, choose 0..8 draft tokens,
propose with the MTP head, verify all of them with the fixed target in one
batched pass of width M = drafts + 1, accept the longest correct prefix, repair
state.

Hardware: ranked runner is an M5, 128 GiB, GPU family `applegpu_g17s`, `_nax`
kernel variants available, register ceiling 126. Student development Macs are
M4 Pro, 48 GiB, `applegpu_g16s`, no `_nax`, register ceiling 96.

## Cost decomposition, measured

Round time is 88.6 % DRAM weight streaming. Transformed weights are 14.41 GB.
Per-mechanism share of the round (Finding 22 reprice):

```
mlp.gate_up            37.937 %
out_proj + mlp.down    28.666 %
gdn.in_proj            13.859 %
lm_head                 4.132 %
attn qkv                4.049 %
gdn recurrent           1.114 %
SDPA                    0.993 %
fused residual+RMSNorm   0.605 %
QMV subtotal           88.643 %
```

Decode GPU time splits 92.5 % target verify, 7.5 % draft head.

The measured in-situ width cost curve (microseconds per round, one binary, real
thermal gate, 512 tokens, M4 Pro):

```
 M   measured us   measured step
 2      70,429.4        --
 3      75,197.3    +4,767.8
 4      83,188.0    +7,990.7
 5      95,301.5   +12,113.5
 6     124,436.0   +29,134.5   <- the big cliff
 7     150,297.5   +25,861.5   <- a second cliff, newly found
 8     154,243.9    +3,946.5
```

Implied streaming efficiency: M=1 462.2 GB/s, M=4 347.2, M=5 320.6, **M=6
235.5**, M=7 229.4, M=8 205.0, M=9 190.5. The cliff is a bandwidth collapse.
Register spill is the governing variable, not the dispatch plan: all 120 legal
partition plans were tested and the cliff survives every one.

Ranked per-prompt round costs, prefill removed (microseconds):

```
plutarch 30,730 | drama 34,141 | travel 35,232 | beagle 44,993
republic 47,806 | essays 48,997 | medicine 49,456 | botany 54,586
```

Width-1 ranked intercept is about 30,000 microseconds.

## The measurement problem, and the instrument

The ranked candidate leg carries a discrete nuisance state, drawn roughly one
time in three, worth **879.0 microseconds per drafting round** (sd 54.3,
measured on three bit-identical resample rows whose true difference is exactly
zero). It does **not** touch the seed prefill or the serial leg. It is the
wired-residency admission lottery: `Qwen36MTPBlockSession.swift:222` wires
resident weights only when physical memory is at least 96 GiB, so it can occur
on the 128 GiB ranked runner and can never occur on a 48 GiB development Mac.

Measured precision on 59 pure-nuisance pairs:

```
candidate leg, prefill included   sd 0.8472 pp   (0.0593 inside the flat band)
seed prefill                      sd 0.0634 pp
serial leg                        sd 0.1738 pp
minimum detectable effect, mode-classified   0.1244 pp
minimum detectable effect, paired replicate  1.8485 pp
```

So: one ranked submission, classified against a same-schedule public anchor,
resolves 0.12 pp. Two ranked submissions paired against each other resolve only
1.85 pp. Classification beats replication by a factor of 15.

## Where the campaign is

Crown is 3.71959723. Our best is 3.66218564. The published gap is 1.57 % but
the true candidate-leg gap is **0.82 %** (Finding 230); the rest is the serial
lottery, which we cannot influence.

In hand, measured, not yet composed into one submission:

```
E87 probe-select port                  +0.72 %
F22 width-6 register occupancy         +0.5913 %
BitWonka 128x32 NAX seed retile        +0.43 %
pb6 boundary depth price               +0.58 %
affine NAX prefill double buffer       +0.25 %
leaf16 on the shipped vocabulary       +0.21 to +0.32 %
```

## What is CLOSED — do not propose these again

- **Tree, multi-candidate, hedge-row, or staged speculative drafting.**
  Structurally blocked: `QwenRuntimeMTPDriver.requireStructurallySound` at
  lines 310-349 forces declared rows to equal `drafts + 1`, a single linear
  chain, and committed tokens to equal primary plus accepted prefix. This kills
  SpecInfer, EAGLE-2 trees, Sequoia, and every multi-draft paper.
- **Block Verification (2403.10444).** Exactly zero gain at temperature 0.
- **Any prompt lookup, n-gram, suffix, or token-history drafting.** Forbidden by
  the integrity contract.
- **Any cross-request cache or benchmark-phase detection.** Forbidden.
- **The flush-fold warm.** Three independent nulls, two of them on the public
  board.
- **Head requantization below 4 bits.** Acceptance collapses 0.933 -> 0.742 at
  q2, 0.868 at q3. The shippable exhaustive affine-4 g64 ceiling is x1.12027
  relative-L2 improvement, and the damage model needs x1.2558 to pay.
- **Head column permutation.** Measured null, x1.00510.
- **Group-size coarsening g64 -> g128 or g256 on the head.** g128 is net
  negative.
- **The probe ladder below fraction 0.09**, and fp32 rerank tiebreak (exactly
  +0.0000 %).
- **Widening the compact draft vocabulary** (E141): net -0.7678 %. The hard
  budget is 331.6 microseconds per round of added cost; that family added 579.3.
- **A width-independent "unowned pool" at 3.7-3.8 % of the round.** It does not
  exist.
- **N=1024 K/V admission into the M=2 QMV launch.** Measured null.
- **Dispatch-count reduction, indirect command buffers, megakernels.** Priced
  negative; the host GPU-idle gap is 706.6 microseconds in a 165,184
  microsecond round, 0.43 %.
- **Layer skipping (SWIFT), PEARL, trained depth predictors.** All priced
  negative or blocked.
- **A perfect acceptance estimator.** Measured: an oracle acceptance predictor
  makes the schedule *worse* (-4.53 % against -3.14 %), because the shipped
  scheduler already exploits the tail correctly.
- **Shipping a proposal head inside the submission archive.** The archive cap is
  25 MiB; the head is 369 MB.
- **A replicate submission as a tie-breaker.** See the MDE table above.

## Where you should look

`senpai/campaign-ledger.md` sections, by line number:

```
   291-330, 4470-4520, 16985-17010   _nax kernel facts
  8794-8801                          the one permitted reassociation
 13778-13786                         GDN S=2 mid-state write break-even
 14126-14160                         the dispatch census
 20269                               the 31 MB precision islands
 21815-21960                         head quantization, section J
 25164-25190                         E87 arm G, the byte model
 34011-34100                         E111/F43, only removed loads pay
 40265-40301                         quantized.h exhaustion
 41890-41900                         the ranked round-cost table
 47757-47769                         the E134 tier grid
 49470-49500                         the rebuilt width curve
 51074                               E134 replayed width masses
 53592                               entry 305
 53972                               entry 306, the current front
```

Editable source of interest:

```
Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift          the target model
Sources/MLXFastModel/Qwen36MTPBlockSession.swift                  the scheduler
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h
Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift    unowned, editable
Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h  unowned
Sources/MLXFastTransform/                                         unowned, editable
```

Not editable, so do not propose changes there: `GatedDelta.swift`,
`backend/metal/quantized.cpp`, `device.cpp`, `jit_kernels.cpp`, `allocator.cpp`,
`resident.cpp`, `buffer_cache.h`, `custom_kernel.cpp`, `sort.cpp`, `ops.cpp`,
`scaled_dot_product_attention.cpp`, everything under
`Sources/MLXFastTrustedHarness/`, the fixtures, the workflow, and
`benchmark.json`.
