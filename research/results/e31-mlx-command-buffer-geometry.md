# E31 — MLX command-buffer geometry: source audit, terminal negative

Student: `qwen-thorfinn` · PR #36 · base `d08feb85bf65959d7eaa1455e36a0173b3edd8d9`
Host: Apple M4 Pro `Mac16,11`, 48 GiB (`hw.memsize = 51539607552`), `applegpu_g16s`.
**Timed arms run in this experiment: zero.** Every number below comes from
vendored MLX source, the shipped Swift startup policy, the on-disk checkpoint
headers, and E29's in-tree analysis file.

Reproduce: `python3 research/e31_command_buffer_audit.py`
(machine-readable copy: `research/results/e31-command-buffer-audit.json`).

## Headline

Deliverable (a) terminates the experiment, on three independent grounds.

1. **The constants are not what E29 claimed.** The scored MTP worker never runs
   at the arch default. It force-installs the command-buffer geometry from
   editable Swift before the first Metal device touch: **50 ops / 512 on the
   ranked M5 Max**, **64 ops / 128 on this 48 GiB host** — not 50/50 anywhere.
2. **The byte cap is not a byte cap.** `buffer_sizes_` accumulates
   `array::data_size()`, documented as *units of `item_size`, not bytes*
   (`array.h:346`). `MLX_MAX_MB_PER_BUFFER` is therefore a **mebi-element**
   cap: for the 4-bit `uint32`-packed backbone it admits 4× the bytes its name
   implies, and for bf16 tensors 2×.
3. **The corrected commit count is higher, not lower, than the baseline, and
   the axis is already measured null.** E29's own four-arm ladder sweep spans
   0→17 forced boundaries with a bit-identical schedule; a least-squares fit
   gives a per-boundary cost of **−36.5 µs (95 % CI −170.3 … +97.4 µs)** —
   consistent with zero, and *negatively* signed. Extrapolated, deleting every
   automatic commit on the ranked geometry moves round time by **+0.418 %
   (slower)**, best case **1.117 % faster** at the far edge of the interval,
   against a 0.05 % ranked-result bar and E29's 0.86 % repeat-noise floor.

Primary metric `e31/verify_forward_command_buffer_commits`: baseline 19,
**corrected value 19.7 automatic (+8 ladder = 27.7 boundaries) on the ranked
box; 31.3 automatic (+8 = 39.3) on this host.** Direction was *minimize*; this
is a **measurement correction of the baseline, not a reduction**. Nothing was
made faster and nothing should be shipped from this experiment.

## Deliverable (a): the source audit

### The active-task / in-flight-command-buffer limit

| item | site | value |
|---|---|---|
| ceiling constant | `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/transforms.cpp:25` | `static constexpr int MAX_ACTIVE_TASKS = 10` |
| enforcement | `transforms.cpp:270-283` | inside `eval_impl`'s per-array tape loop |
| counter increment | `backend/metal/eval.cpp:61` | `notify_new_task` — **only** on `needs_commit()`-driven commits |
| counter decrement | `backend/metal/eval.cpp:63` | command-buffer completion handler |
| the wait | `mlx/scheduler.h:100-107` | `wait_for_one()` |

E29's `MAX_ACTIVE_TASKS = 10` claim is **correct**, and it is a **compile-time
`static constexpr`**: no environment variable, no setter, no API. Two
qualifications that matter:

- `wait_for_one()` waits for the count to *decrease by one*, and only if it is
  `> 1`. It is not "block until below 10"; it is "let one buffer retire".
- `gpu::finalize` (`eval.cpp:71-77`) commits **without** `notify_new_task`, so
  ladder rungs and end-of-eval flushes never enter the in-flight accounting.
  Only the automatic commits do.

`transforms.cpp` is **not in `benchmark.json` `editablePaths`** (the only
`Vendor/mlx-swift` entries are `mlx-generated/*.cpp` and
`backend/metal/kernels/**`). Even a working patch here could not be submitted.

### The per-buffer thresholds that force a commit

| item | site | detail |
|---|---|---|
| trigger | `backend/metal/device.cpp:484-487` | `(buffer_ops_ > max_ops) \|\| ((buffer_sizes_ >> 20) > max_mb)` |
| where checked | `backend/metal/eval.cpp:59` | once **per array** in `gpu::eval`, not per dispatch |
| op counter | `device.cpp:381`, `:389` | `buffer_ops_++` per `dispatchThreadgroups` / `dispatchThreads` |
| size counter | `device.cpp:320` | `buffer_sizes_ += a.data_size()`, deduped per buffer pointer |
| outputs count too | `device.cpp:330-336` | `set_output_array` calls `set_input_array` |
| reset | `device.cpp:526-529` | `commit()` zeroes **both** counters |
| **unit** | `mlx/array.h:346,348` | `data_size` is in units of `item_size`, **not bytes** |

Because the check is per array and the counter is per dispatch, a single
primitive that issues several dispatches can overshoot `max_ops` before the
next check.

### Are they settable at runtime?

| knob | mechanism | site | reachable from the submitted surface? |
|---|---|---|---|
| `MAX_ACTIVE_TASKS` | none — compile-time constant | `transforms.cpp:25` | **no** (constant *and* non-editable file) |
| `max_ops_per_buffer` | env `MLX_MAX_OPS_PER_BUFFER` | `mlx/utils.h:178-182`, applied `device.cpp:596` | **yes**, via editable Swift |
| `max_mb_per_buffer` | env `MLX_MAX_MB_PER_BUFFER` | `mlx/utils.h:184-188`, applied `device.cpp:597` | **yes**, via editable Swift |
| forced boundaries | env `MLX_QWEN_MTP_LADDER` | `Qwen35.swift:2118-2132`, fired `:2221-2224` / `:2241-2244` | **yes**, editable, **already swept** |

Both env vars are read through `env::get_var` → `std::getenv`
(`mlx/utils.cpp:288-294`) into **function-local statics**, i.e. read exactly
once, at the first `Device` construction (`device.cpp:557,596-597`). A
`setenv` after the first Metal touch is silently ignored. There is a public
getter `Device::get_max_ops_mb_per_buffer()` (`device.h:162`) and **no
setter**, in C++ or in the mlx-swift API.

### What the scored worker actually installs

`QwenRuntimeMTPWorker.swift:133` calls `applyQwenMTPStartupMemoryProfile()`
before the backbone and head loads, i.e. before any MLX array work:

- `RuntimeStartupMemoryPolicy.resolve` →
  `installQwenMTPFullProfileCommandBufferDefaults`
  (`RuntimeStartupMemoryPolicy.swift:62-73`) sets **`MLX_MAX_MB_PER_BUFFER=512`,
  `MLX_MAX_OPS_PER_BUFFER=50`** with `overwrite=0`, gated on ≥ 96 GiB physical
  memory and `DARKBLOOM_QWEN_MTP_POST_WIRE_COMMAND_BUFFER != "0"`.
- The worker then does `guard policy.isLowMemory else { return }`
  (`QwenRuntimeMTPWorker.swift:487`), so on the ranked box the full-profile
  scalars (320 / 128, `RuntimeStartupMemoryPolicy.swift:145-146`) are **never**
  applied. The 512 / 50 install defaults are what survive.
- On a low-memory host the guard falls through and `:488-489` force-set
  (`overwrite=1`) the low-memory scalars **128 / 64**
  (`RuntimeStartupMemoryPolicy.swift:112-113`).
- `Qwen35RuntimeWeights.swift:45` (`MLX_MAX_MB_PER_BUFFER=128`, `overwrite=1`)
  is **not** on this path: it lives in `Qwen35RuntimeWeightCache`, which only
  `QwenRuntimeBenchmark.swift:123,149` constructs.
- Nothing in `benchmark-qwen-mtp.sh`, the ranked workflow, or the fixtures sets
  either variable, so no external value pre-empts the `overwrite=0` install.

Ranked runner: `m5-max-128gb-3`
(`.github/workflows/qwen-mtp-ranked-benchmark.yml:196,204`) → 128 GiB ≥ 96 GiB
→ full profile. This host: 48 GiB < 64 GiB → low-memory profile.

| host | physical | profile branch | effective `max_ops` | effective `max_mb` | arch default that was overridden |
|---|---|---|---|---|---|
| ranked M5 Max | 128 GiB | full | **50** | **512** | 50 / 50 (`'s'`, `device.cpp:583-586`) |
| this M4 Pro | 48 GiB | low-memory | **64** | **128** | 50 / 50 (`'s'`) |

E29's "50 ops / 50 MB, arch default for `'s'`" is wrong on both boxes. The op
cap coincidentally equals 50 on the ranked box — by environment, not by arch —
and the size cap is off by 10.24×, in mebi-elements rather than MB.

## The commit count, counted rather than estimated

Element inventory from the on-disk checkpoint headers — `weights/`, the tree
the runtime actually loads (`benchmark-qwen-mtp.sh:133`), 1847 tensors:
**4.2047e9 elements** total, **3.8074e9** in the 64 layers (5.9491e7 per
layer), 3.97e8 in the quantized embedding and LM head. A full verify forward
references every one of them; `set_input_array` charges the whole buffer, not
the touched slice. E29's dispatch inventory is 969 for M=6..9, i.e. 15.14
dispatches per layer.

The element inventory is exact. The one modelling assumption is that dispatches
and referenced elements are spread evenly across the 64 layers; layer-to-layer
variation moves individual commit boundaries but not the count, because the
layer stack is 90.6 % of the elements and the two layer types alternate on a
fixed period of 4.

Per layer, the fraction of each cap consumed:

| host | ops per layer / cap | elements per layer / cap | binding axis | commits per forward |
|---|---|---|---|---|
| ranked (50 / 512) | 15.14 / 51 = 0.297 | 5.949e7 / 5.369e8 = 0.111 | **ops** | 19.0 by ops, 7.8 by elements → **19.7** |
| this host (64 / 128) | 15.14 / 65 = 0.233 | 5.949e7 / 1.342e8 = 0.443 | **elements** | 14.9 by ops, 31.3 by elements → **31.3** |
| E29's premise (50 / 50) | 0.297 | 1.135 | elements | would have been **80.2** |

Adding the shipped 8-rung ladder gives **27.7 boundaries per verify forward on
the ranked box and 39.3 on this host**. E29's "19" is right for the ranked
op axis alone and wrong everywhere else — including on the machine where E29
measured, where the true count is ~1.6× higher and set by the *element* axis
that E29 did not model.

## Deliverables (b), (c), (e): not run, and why

No timed arm was launched in E31. The advisor's stop instruction
(`#issuecomment-5335548669`) arrived with the sweep that (b) would have
duplicated. **There is therefore no `effective_draft_lengths` element-wise
check of my own to report, and no arm of mine changed a token, because no arm
of mine ran.** The four-arm evidence I cite is E29's, on this same host, and
its schedule invariance is verifiable in-tree: `accepted_draft_total = 222` in
all four arms, and the realised depth histogram is identical in all four —
`{M=2: 1, M=6: 9, M=7: 3, M=8: 5, M=9: 15}` — which satisfies (e) for every
number quoted below.

E29 ladder sweep, reported as two legs as the corrected scoring model requires
(local `--local-iterate`, 256 decode tokens, M4 Pro, ungated):

| arm | ladder | forced boundaries | serial leg ms/tok | MTP leg ms/tok | MTP ms/round | local ratio |
|---|---|---:|---:|---:|---:|---:|
| L0 | `off` | 0 | 80.344 | 23.544 | 172.208 | 3.4125 |
| L1 | `front` | 2 | 78.833 | 23.534 | 172.137 | 3.3497 |
| D0 | `default` | 8 | 66.875 | 23.592 | 172.559 | 2.8346 |
| L2 | `dense` | 17 | 65.370 | 23.449 | 171.511 | 2.7878 |

- MTP leg spread **0.611 %**, non-monotone, below the 0.86 % repeat-noise floor.
- Least-squares slope over rung count: **−36.5 µs per boundary**, SE 31.1 µs,
  95 % CI **[−170.3, +97.4] µs** (t = −1.17 on 2 dof). Not distinguishable
  from zero, and signed the wrong way for the hypothesis.
- Extrapolating to the removal of *every* automatic commit: ranked
  **+0.418 % round time (slower)** centrally, 95 % CI [−1.117 %, +1.954 %];
  the best case for the mechanism is a **1.117 %** speedup at the extreme edge
  of an interval whose centre is a slowdown.

## Deliverable (d): the scoping answer

A runtime knob **does** exist and **is** submittable — `MLX_MAX_OPS_PER_BUFFER`
and `MLX_MAX_MB_PER_BUFFER`, set from `Sources/MLXFastModel/`, which
`benchmark.json` packages with every submission. So (d)'s "no knob exists"
branch does not apply, and no MLX internals need patching.

Honest boundary of the existing evidence: E29 swept the range
`[automatic floor, floor + 17]` by *adding* forced boundaries. It never raised
the caps, so the sub-range `[1, floor]` — *fewer* commits than MLX's automatic
schedule — is formally unswept. I did not measure it, for three reasons:

1. The fit above is the best available predictor for that region and it
   predicts a **slowdown** of ~0.4 % centrally, with a best case (1.1 %) inside
   the instrument's own 0.86 % noise floor.
2. The mechanism is asymmetric. The serial leg shows what happens when
   boundaries are too few: 0 rungs is **22.9 % slower per token** than 17
   rungs (80.344 → 65.370 ms/tok). Boundaries buy host/GPU overlap; removing
   the automatic ones on a graph the host builds inline can only reduce
   overlap.
3. At MTP widths the round is GPU-bound — 95.65 % `eval_wall` in the L0 arm —
   so the entire host-side envelope any commit-geometry change can address is
   ≤ 4.35 % of the round, and most of that is genuine GPU wait rather than
   removable host work.

The one lever that E29's story actually rested on, `MAX_ACTIVE_TASKS`, is
unreachable: compile-time constant in a non-editable file. That is the region
this audit bounds and closes.

## Measurement observation (not a proposal)

Applying the advisor's corrected scoring model — both legs measured in the same
session, serial leg *is* the normaliser — to E29's own numbers: turning the
ladder off raises the local ratio from 2.8346 to 3.4125, **+20.4 %**, while the
MTP leg moves +0.2 %. The gain is entirely a **slower serial leg**
(66.875 → 80.344 ms/tok). Under the pinned-serial assumption E29 recorded this
as "the real effect on score is the MTP-leg change: −0.11 %, i.e. noise"; that
conclusion depends on the assumption the advisor has now corrected.

Per the advisor's standing instruction this is reported as an observation and
stops here. I am **not** proposing a ladder change, and the ladder is
load-bearing for serial decode.

## Pre-registration

The assignment asked for a pre-registered commit-count target, speedup model
and kill criterion before measuring. No measurement was taken, so no
pre-registration is claimed. The assignment's own kill criteria fired at the
source-audit stage:

- *"the constants are not as claimed"* — fired (50/512 and 64/128, and the
  size cap counts elements, not bytes);
- *"the measured effect is inside the noise floor"* — fired on the in-tree
  sweep, before any new GPU time was spent.

## Suggested follow-ups (not implemented)

- **Delete or fix the dead `MLX_MAX_MB_PER_BUFFER` write** at
  `Qwen35RuntimeWeights.swift:45`. It is unreachable from the MTP worker and
  its comment describes an allocator-cache intent that the command-buffer
  variable does not serve. Cleanup only; zero expected performance effect.
- **Rename or document the mebi-element semantics** wherever the campaign's own
  Swift comments call `MLX_MAX_MB_PER_BUFFER` a byte budget
  (`RuntimeStartupMemoryPolicy.swift:56-59,139-144` both say MiB). The policy
  values were presumably chosen against the wrong unit; on the ranked box the
  512 setting admits ~2 GiB of 4-bit weight references per buffer, which is why
  the op axis binds there and the element axis never does.
- If the `[1, floor]` corner is ever worth a turn, the cheapest decisive form
  is a single ABBA-counterbalanced pair at `MLX_MAX_OPS_PER_BUFFER=4000`,
  `MLX_MAX_MB_PER_BUFFER=8192` (≈1 commit per forward) versus shipped, both
  legs reported. The prediction on record here is a 0.4 % slowdown.
