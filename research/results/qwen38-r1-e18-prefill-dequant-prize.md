# E18 — Prefill dequant prize: dispatch adjudication and overhead decomposition

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"removable_fraction_of_P","available":false,"value":null},"test_metric":{"name":"qmv_parity_cells_differing","available":false,"value":null}}

- Assignment: `qwen38-r1-e18-prefill-dequant-prize` (r1), PR #20
- Student: `qwen-thorfinn`
- Base: `senpai/qwen38-mtp-r1` @ `422db0451e92a8c9b70ff2900874fa3b69fab261`
- Analysis head: `dbfef04721f9d2bed33ebee0c3b2e43a13df39d6` — **every source line
  number in this document resolves against that commit.** The result commit is
  its direct child and adds only this file plus `research/e18_dispatch_table.py`
  (Phase 1 dispatch reproduction) and `research/e18_yukon_prefill.py`
  (§11 ranked-receipt reproduction).
- Host: AWS Mac, Apple M4 Pro, `hw.memsize = 51539607552` (48 GiB), macOS 26.5.2
  (build 25F84), `swift-driver 1.148.6` / Apple Swift 6.3.3
  (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`), target `arm64-apple-macosx26.0`.
  **Not the ranked M5.**
- Thermal sessions: **none**. No timed leg was run, no GPU lock was taken, and I
  remain third in the queue behind PR #17 (`qwen-askeladd`) and PR #19
  (`qwen-edward`).
- Worktree `dirty` flag at analysis time: **dirty** — two untracked research
  scripts, `research/e18_dispatch_table.py` and `research/e18_yukon_prefill.py`,
  both committed alongside this document. **No file under any submitted
  `editablePaths` entry was modified**, so the candidate snapshot at
  `dbfef04…` is byte-identical to the base.
- W&B runs: **none**. This experiment is host-only source adjudication and
  arithmetic; it produced no GPU work to log.
- Roofline used throughout: **226.9 GB/s** (STREAM `227,128,791,836.97 B/s`).

## Headline

**Clean negative, and I am not shipping a kernel edit.**

Phase 1 and Phase 2 both completed. The result is that the prefill dequant prize
described in the assignment is not reachable from inside the editable surface,
for a structural reason rather than a measurement reason: every byte-traffic term
is two orders of magnitude too small to matter, and the one surviving mechanism —
unpack ALU per MAC — has a magnitude set entirely by the GEMM tile height `BM`,
which is chosen by the **read-only** dispatcher and baked into the requested
kernel name before any editable code runs.

Five corrections to the assignment's premises came out of the work. One of them
(Correction 5) contradicts a standing campaign belief and may reopen closed work,
so it is stated in full below rather than buried.

- **H1 → UNRESOLVED**, with the two missing facts named precisely and the
  smallest resolving read identified. Stopping rule (b) fires.
- **H2 → CONFIRMED.** The NA=4/NA=5 cross-row register cliff does not govern the
  prefill GEMM, and the proof is structural — the kernel carrying that mechanism
  is not reachable at prefill at all.
- **Stopping rule (c) is satisfied for every byte-traffic term.** The largest of
  them, all quantized weight traffic combined, is 1.632 % of measured GEMM
  seconds; metadata alone is 0.176 % of P and the split-K round trip is 0.0166 %
  of P.
- **The prize is also 4–8× smaller than the assignment assumed** (§11, added
  late, from ranked-hardware telemetry in public Yukon receipts). Ranked prefill
  is **6.20 %** of the candidate leg, not the 15.8–18.0 % the assignment states,
  and it has not moved across 87 distinct candidate commits from 50 solvers
  spanning a 2.44× decode-score range. Even assuming *all* of φ were removable,
  that is **1.94 frontier steps**, and **0.93–1.17** under the NAX branch —
  against the 7.34 steps the local framing implied. This does not change the
  verdict; it raises confidence in it.

## 1. Corrections to the assignment premises

### Correction 1 — the `split_k = 1` arithmetic is CORRECT; the conclusion drawn from it is not

I reproduced `qmm_splitk`'s selection arithmetic exactly from
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:790-810`:

```text
bm = bn = 32
n_tiles     = ceil(N / 32)
m_tiles     = ceil(M / 32)
current_tgs = n_tiles * m_tiles
split_k     = max(1, 512 / current_tgs)
k_align     = max(group_size, 32)                 # 64 here
split_k     = min(split_k, K / k_align)
while (split_k > 1 && K % (split_k * k_align) != 0) split_k--
if (split_k <= 1) -> delegate to qmm()
```

At M=512, `m_tiles = 16`. Any `N >= 1024` therefore gives `n_tiles >= 32`, so
`current_tgs >= 512` and `512 / current_tgs == 1`. **For every shape the
assignment lists, `split_k == 1` is correct**, and correcting the shapes to the
real unfused ones (Correction 2) does not disturb that — every real projection
with `N >= 1024` still collapses to `qmm()`.

`research/e18_dispatch_table.py` reproduces this on the assignment's own quoted
shapes as a cross-check:

```text
advisor row                   K       N  ntl  mtl   tgs  spk  branch
gate_up_proj               5120   17408  544   16  8704    1  qmm_splitk -> qmm
down_proj                  8704    5120  160   16  2560    1  qmm_splitk -> qmm
qkv_proj                   5120    7168  224   16  3584    1  qmm_splitk -> qmm
o_proj                     3072    5120  160   16  2560    1  qmm_splitk -> qmm
in_proj (gated delta)      5120    8192  256   16  4096    1  qmm_splitk -> qmm
lm_head                    5120  248320    -    -     -    -  dispatch_qmv -> qmv
```

**But the conclusion "the `qmm_t_splitk` kernels are dead code on this model" is
REFUTED.** The assignment's shape list omits two projections with a tiny `N`:

- `linear_attn.in_proj_b` — `K=5120, N=48`
- `linear_attn.in_proj_a` — `K=5120, N=48`

`N = 48` is `linear_num_value_heads` (`Sources/MLXFastModel/Qwen35Config.swift:244-247`),
and both projections really are quantized — `Sources/MLXFastModel/Qwen35Weights.swift:507-508`
registers them through `addQuantized`, and `:255-256` validates them through
`validateQuantizedLinear` with shape `[48, 5120]`. At `N=48`, `n_tiles = 2`,
`current_tgs = 32`, and `split_k` lands on `16`. So
`affine_qmm_t_splitk_bfloat16_gs_64_b_4_alN_false` is dispatched **96 times per
prefill** (2 projections × 48 GatedDeltaNet layers).

That path is also **NAX-immune**: `qmm_splitk` never calls `is_nax_available()`.
The NAX early-return lives only in `qmm()` (`quantized.cpp:697-698`), which
`qmm_splitk` reaches only when `split_k` collapses to 1.

The cost is small — 150,994,944 B of bf16 partial write-plus-read ≈ **0.665 ms ≈
0.0166 % of P**. This is a correction to the record, not a lever. But "dead code"
was load-bearing for the assignment's framing of where prefill time goes, so it
needed checking.

### Correction 2 — the model is UNFUSED, so the assignment's shape list is numerically wrong

There is no fused `gate_up`, no fused `qkv`, and no fused gated-delta `in_proj`
anywhere in the executed path:

- `Sources/MLXFastModel/Qwen35MLP.swift` issues gate, up and down as three
  separate `Qwen35Ops.linear` calls.
- `Sources/MLXFastModel/Qwen35Attention.swift` issues them separately:
  `queryProjection:143`, `keyProjection:162`, `valueProjection:163`,
  `outputProjection:211`.
- `Sources/MLXFastModel/Qwen35GatedDelta.swift` issues five separately:
  `inputQKVProjection:241`, `inputZProjection:245`, `inputBProjection:254`,
  `inputAProjection:255`, `outputProjection:346`.

`Sources/MLXFastModel/Qwen35Ops.swift:29-46` shows `linear()` taking a dense
`matmul` when `scales == nil` and `quantizedMM(transpose: true, mode: .affine)`
otherwise.

So `gate_proj` has `N=17408` (not 34816), `q_proj` has `N=12288` (not a fused
7168), `down_proj` has `K=17408` (not 8704), and `o_proj` has `K=6144` (not
3072). The corrected census per prefill is **400** `affine_qmm_t_*` + **96**
`affine_qmm_t_splitk_*` + **1** `affine_qmv_fast_*`.

### Correction 3 — lm_head at prefill is an M=1 QMV, not an M=512 GEMM

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift` `begin(seedTokens:)`
(`:369-415`) calls `model.callWithHidden(tokens.reshaped([1, 512]))`, builds the
512-row `seedLogits` lm_head projection, and then **deliberately never evaluates
it** — it is a dead lazy graph (`:378-386`). Only a single row is pushed through
`model.applyLMHead(pendingHidden)`. Head-history priming is likewise lazy
(`:839-849`, `primeCount = seedTokens.count - 1`) and lands in the first drafting
round, not in prefill.

This matters for the budget: at `N = 248320` an M=512 lm_head GEMM would have
dominated the census. At M=1 it contributes exactly `2 * 5120 * 248320 =
2,542,796,800` FLOP and dispatches `affine_qmv_fast_bfloat16_gs_64_b_4_batch_0`.

### Correction 4 — the corrected census reproduces E16's FLOP total exactly

```text
GEMM FLOPs (this census)          24,937,512,304,640
GEMM FLOPs (E16 reported)         24,937,512,304,640
relative error                    0.000e+00
```

Zero relative error, not "close". Since the census and E16 were built
independently, and since the M=1 lm_head term is what makes the totals agree,
this simultaneously validates the unfused census (Correction 2) and Correction 3.
I would not have trusted either on its own.

### Correction 5 — the dispatcher is READ-ONLY, but `steel/gemm/**` IS editable

This one contradicts a standing campaign belief and should be propagated.

`benchmark.json` carries **85 vendor file entries plus 5 directory entries**:
`Sources/MLXFastModel`, `Sources/MLXFastTransform`,
`Vendor/.../kernels/steel/attn`, `Vendor/.../kernels/steel/gemm`, and
`mtp-head`. `.github/scripts/run-submission-static-review.sh:405` expands a
directory entry with `find "${editable_path}" -type f -print0`, so **every file
under those directories is editable**, not just files named individually.

Verified status of the paths that matter here:

| path | status |
|---|---|
| `Vendor/.../backend/metal/quantized.cpp` (the dispatcher) | **READ-ONLY** |
| `Vendor/.../backend/metal/matmul.cpp` | READ-ONLY |
| `Vendor/.../backend/metal/device.cpp` | READ-ONLY |
| `Vendor/.../kernels/quantized.h` | EDITABLE (file entry) |
| `Vendor/.../kernels/quantized_nax.h` | EDITABLE (file entry) |
| `Vendor/.../kernels/steel/gemm/nax.h` | **EDITABLE (directory entry)** |
| `Vendor/.../kernels/steel/gemm/mma.h` | **EDITABLE (directory entry)** |
| `Vendor/.../kernels/steel/gemm/loader.h` | **EDITABLE (directory entry)** |
| `Vendor/.../kernels/steel/gemm/gemm_nax.h` | **EDITABLE (directory entry)** |

`steel/gemm/` contains `gemm_nax.h  gemm.h  kernels  loader.h  mma.h  nax.h
params.h  transforms.h`.

The prior campaign conclusion that "no `steel/**` NAX headers are editable" came
from a file-path-only check that never expanded the directory entries. A related
hazard: **`rg` is not installed on this host**, so any campaign conclusion of the
form "grep found nothing" that was reached with `rg` is potentially a false
negative and should be re-checked with `grep -rn`. I hit this myself.

The consequence for E18 is the decisive one. `get_qmv_batch_limit`, the
`M >= vector_limit` branch, the NAX gate, the `qmm_splitk` arithmetic, **and the
tile sizes `bm`/`bn`/`bk` that the dispatcher bakes into the requested kernel
name** all live in the read-only `quantized.cpp`. No host-side dispatch change
and no tile-selection change is submittable.

## 2. Phase 1 — dispatch table

Produced by `research/e18_dispatch_table.py` (committed with this document). The
script ports `get_qmv_batch_limit`, the `eval_gpu` branch, the `qmm_splitk`
arithmetic and the kernel-name builders from `quantized.cpp`, and models four
hosts: the local M4 Pro plus three ranked hypotheses. Run it with `--json` for
machine-readable output or `--M` to sweep the row count.

`Tests/MLXFastTests/QwenQMVCostCurveTests.swift:352-386` contains an independent
Swift reproduction of the same host-dispatch logic (`arch`, `gen`,
`naxAvailable`, `vectorLimit`); my port agrees with it.

Local host, `applegpu_g16s`, gen 16, `is_nax_available() = False`, M=512. Every
row has `vlim = 10` and `m_tiles = 16`.

| projection | K | N | n_tiles | tgs | split_k | branch | kernel |
|---|---|---|---|---|---|---|---|
| linear_attn.in_proj_qkv | 5120 | 10240 | 320 | 5120 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| linear_attn.in_proj_z | 5120 | 6144 | 192 | 3072 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| **linear_attn.in_proj_b** | 5120 | **48** | 2 | 32 | **16** | **qmm_splitk** | `affine_qmm_t_splitk_bfloat16_gs_64_b_4_alN_false` |
| **linear_attn.in_proj_a** | 5120 | **48** | 2 | 32 | **16** | **qmm_splitk** | `affine_qmm_t_splitk_bfloat16_gs_64_b_4_alN_false` |
| linear_attn.out_proj | 6144 | 5120 | 160 | 2560 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| full_attn.q_proj | 5120 | 12288 | 384 | 6144 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| full_attn.k_proj | 5120 | 1024 | 32 | 512 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| full_attn.v_proj | 5120 | 1024 | 32 | 512 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| full_attn.o_proj | 6144 | 5120 | 160 | 2560 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| mlp.gate_proj | 5120 | 17408 | 544 | 8704 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| mlp.up_proj | 5120 | 17408 | 544 | 8704 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| mlp.down_proj | 17408 | 5120 | 160 | 2560 | 1 | qmm_splitk → qmm | `affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0` |
| head.lm_head (M=1) | 5120 | 248320 | – | – | – | dispatch_qmv → qmv | `affine_qmv_fast_bfloat16_gs_64_b_4_batch_0` |

Census per prefill:

```text
   400 dispatches  affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0
                   48x in_proj_qkv, 48x in_proj_z, 48x out_proj,
                   16x q_proj, 16x k_proj, 16x v_proj, 16x o_proj,
                   64x gate_proj, 64x up_proj, 64x down_proj
    96 dispatches  affine_qmm_t_splitk_bfloat16_gs_64_b_4_alN_false
                   48x in_proj_b, 48x in_proj_a
     1 dispatches  affine_qmv_fast_bfloat16_gs_64_b_4_batch_0
                    1x lm_head (M=1)
```

Under a NAX-capable host the 400 `affine_qmm_t_*` dispatches become
`affine_qmm_t_nax_bfloat16_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0`.
The 96 split-K dispatches and the 1 QMV dispatch are **byte-identical either
way**.

Relevant dispatcher line map in `quantized.cpp` (1768 lines), for whoever picks
this up next: `get_qmv_batch_limit:84`, `qmv_quad:177`, `qmv:235`,
`qvm_split_k:298`, `qvm:419`, `qmm_nax:473` (tiles `:491-495`, name `:499-522`),
`gather_qmm_nax:576`, `qmm:682` (NAX early-return `:697-698`, tiles `:717-720`,
name `:722-735`), `qmm_splitk:776` (arithmetic `:790-810`, name `:834-844`,
reduction `:817-873`), `gather_qmm:875`, `gather_qmm_rhs_nax:1090`,
`gather_qmm_rhs:1221`, `dispatch_qmv:1371-1391`,
`QuantizedMatmul::eval_gpu:1393-1461`, `GatherQMM::eval_gpu:1483`.

`get_qmv_batch_limit` returns `{18,12,10}` by default, keyed on
`D<=2048 && O<=2048` / `D<=4096 && O<=4096` / else. Gen 13/14 and size `'d'`
have their own tables. At our shapes the third bucket applies, giving
`vector_limit = 10`.

## 3. H1 verdict — UNRESOLVED, with the missing facts named

**H1: "ranked prefill dispatches through `qmm_nax`, not the non-NAX `qmm`."**

**Verdict: UNRESOLVED.** Per the assignment, an UNRESOLVED verdict with a named
missing fact is a full pass, and stopping rule (b) fires here.

### The assignment's proposed off control does not exist in this build

The assignment names `MLX_METAL_NO_NAX` as the off control. **It cannot be, and
this refutes the §3 premise.** That macro is set by
`target_compile_definitions(mlx PRIVATE ...)` at
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/CMakeLists.txt:176`,
scoped to the CMake `mlx` target. But SwiftPM compiles `device.cpp` in the
`Cmlx` target, whose defines are only `MLX_USE_ACCELERATE`,
`ACCELERATE_NEW_LAPACK`, `_METAL_`, `SWIFTPM_BUNDLE`, `METAL_PATH`
(`Vendor/mlx-swift/Package.swift:218-222`) plus `MLX_VERSION` (`:307`). The macro
is never visible to the code that would honour it.

So `is_nax_available()` (`device.cpp:913-932`) always compiles to its runtime
branch:

```text
__builtin_available(macOS 26.2, *) && gen >= (arch == 'p' ? 18 : 17)
```

There is no compile-time off switch to toggle. (`MLX_METAL_GPU_ARCH` exists as an
env override — `utils.h:205` and `:56` — but `research/CURRENT_RESEARCH_STATE.md:1667`
already flagged it as non-surgical and non-submittable, and I agree: it would
change the arch string globally rather than isolate the GEMM.)

### The two facts that would resolve H1, and why neither is in this repo

`is_nax_available()` needs both:

1. **The ranked box's macOS version** — must be ≥ 26.2.
2. **The ranked box's literal `applegpu_*` architecture string** — supplies both
   `gen` and the `arch` size character.

Neither is emitted anywhere in this checkout:

- `.github/workflows/qwen-mtp-ranked-benchmark.yml:1373` does call
  `sw_vers -buildVersion`, but only feeds the value into a SHA-256 cache
  fingerprint (`:1369-1382`). It is never printed and never asserted.
- `grep -rn 'deviceInfo()' Sources/` returns **zero hits**, and the sealed ranked
  report carries no architecture field.
- The runner has no image pin: `runs-on: [self-hosted, m5-qwen38-27b-mtp]`
  (`:204`). The only description of the box is prose —
  `docs/qwen-mtp-go-live-runbook.md:65-66` names `m5-max-128gb-3` / "Apple M5
  Max".
- `senpai/campaign-ledger.md:169` asserts "ranked M5 crosses it", but that is
  **unsourced prose**, not evidence, and I decline to launder it into a verdict.

**The smallest resolving read is outside this checkout**: a Yukon receipt or a
GitHub Actions job log that captures the ranked box's macOS build string and its
`applegpu_*` architecture. I do not have access to either from here.

A complicating fact worth recording: `fixtures/qwen3_8_27b_mtp_track.json:156`
records **two** ranked boxes with different pinned serial means (`-2` at
`0.03799388007610105` s/tok, `-3` at `0.03793544849031605` s/tok), and `:196`
names box 3. "The ranked host" is not a single machine, so H1 could in principle
have different answers on different runs.

### Labelled inference (not evidence, and not the verdict)

The one thing I can establish is a negative that removes a plausible failure
mode. This local machine is an M4 **Pro** and reports `applegpu_g16s` — so the
`'s'` character is **not** a tier marker distinguishing Max from Pro, and an M5
Max would not be expected to report `'p'`. The most plausible M5 Max string is
therefore `g17s`, which gives `17 >= 17` and passes the generation half of the
gate. Combined with the toolchain evidence below, H1 is *more likely true than
false*.

That is an inference, and it stays labelled as one. It is not sufficient to spend
a kernel edit on, because the whole value of the prefill dequant prize depends on
which branch runs.

What *is* established about the ranked toolchain: the Metal component is pinned
to `com.apple.dt.toolchain.Metal.32023.883` and the workflow fails closed on it
(`:1357-1361`).

### Local-side NAX facts established along the way

- Seven `*_nax.air` files exist under `.build-worker/mlx-metal/`, proving the
  CMake NAX branch was taken when the local metallib was built.
- The `qmm_nax` kernel body has **no internal fallback** — all gating is in the
  caller. Its `aligned = N % 64 == 0` flag selects a variant, it never rejects.
- NAX quantized kernels are **JIT**, not AOT: `jit_kernels.cpp:1116`, with
  `nojit_kernels.cpp` in SwiftPM's `exclude:` list (`Package.swift:284`). The
  runtime-effective sources are therefore
  `mlx-generated/{quantized_nax,fp_quantized_nax,gemm_nax}.cpp`, and any edit to
  the readable headers would have needed the twin update plus
  `python3 research/twin_audit.py`. No edit was made, so no twin audit was
  required.
- There is no MLX JIT disk cache — `Device::get_library` (`device.cpp:770`) is an
  in-memory `library_map_` only.

## 4. H2 verdict — CONFIRMED

**H2: "the NA=4/NA=5 cross-row register-cliff mechanism does not govern the
prefill GEMM."**

**Verdict: CONFIRMED.** This needed no measurement; the dispatcher settles it.

`qmv_fast_crossrow_affine4_g64` is a campaign-added kernel that exists only in
`Vendor/.../kernels/quantized.h` — `:860` (base), `:969` (`_wide`), `:1054`
(`_m`) — and is selected by an in-kernel dispatch switch at `:1811-1904` inside
`[[kernel]] affine_qmv_fast`. `NA` is the accumulator count (`vec<float, NA>`),
as established by E13 (`research/results/qwen38-r1-e13-na4-register-cliff.md`,
my own prior experiment) and E14
(`research/results/qwen38-r1-e14-ipg-weight-passes.md`).

Reachability is the whole argument:

1. `affine_qmv_fast` is reached only through `qmv` ← `dispatch_qmv` ←
   `QuantizedMatmul::eval_gpu`.
2. `eval_gpu` enters `dispatch_qmv` only when `M < vector_limit`.
3. At prefill, all 496 weight projections have `M = 512`, and `vector_limit = 10`
   for every one of them. `512 >= 10`, so **`dispatch_qmv` is never entered.**
4. The single M=1 lm_head call *does* reach `affine_qmv_fast`, but with
   `ntg.x = 1`. The dispatch switch has cases for `2..9` only; `ntg.x = 1` misses
   all of them, hits `default: break;`, and falls through to plain
   `qmv_fast_impl<T, group_size, bits>` (verified at `:1905-1925`).

**No cross-row variant executes during prefill at all.** The transfer from the
NA cliff to the prefill GEMM is exactly zero — not small, not noisy, zero.

There is a corroborating campaign note at `senpai/campaign-ledger.md:176-182`:
even a satisfied NAX gate would not help decode QMV, because `quantized_nax.h`
has no `qmv`, `crossrow`, `affine4` or `qmm_splitk` entry point. Separately, the
string `crossrow` does not appear anywhere in `quantized.cpp`.

## 5. Phase 2 — decomposition of the dequant overhead

Ordered by footprint, as the assignment asked. The path Phase 1 identified for
the local host is non-NAX `affine_qmm_t` at `BM=BN=BK=32`; the NAX arm is
included as **source analysis, labelled not locally measurable**, because
`is_nax_available()` is False on this machine.

### 5.1 Byte-traffic terms — all far below the stopping-rule bar

```text
GEMM FLOPs (this census)          24,937,512,304,640
4-bit code bytes                  12,810,977,280
scale+bias bytes (bf16 pairs)      1,601,372,160
total quantized weight bytes      14,412,349,440
arithmetic intensity              1,730.3 FLOP/byte

weight traffic at roofline         0.063455 s  =  1.632% of measured GEMM seconds
  -> prefill GEMM is COMPUTE-BOUND by 61.3x

scale/bias traffic alone           0.007051 s  =  0.1761% of P
split-K partial round trip         0.000665 s  =  0.0166% of P   (150,994,944 B)
```

| term | bytes | seconds @ 226.9 GB/s | % of P |
|---|---|---|---|
| all quantized weight traffic | 14,412,349,440 | 0.063455 | 1.632 % (of GEMM s) |
| scales + biases only | 1,601,372,160 | 0.007051 | **0.176 %** |
| split-K partial round trip | 150,994,944 | 0.000665 | **0.0166 %** |

**Stopping rule (c) is satisfied for every byte-traffic term.** The bar is ~2 % of
P; metadata reads and the split-K round trip together come to about 0.19 % of P.
Even the physically impossible move of eliminating *all* weight traffic buys
1.632 % of GEMM seconds. At an arithmetic intensity of 1,730 FLOP/byte the
prefill GEMM is compute-bound by 61.3×, so no bandwidth-side construction can
pay for itself here. This kills the "metadata layout" and "scale/bias packing"
families of idea outright.

### 5.2 Unpack ALU per MAC — the only surviving mechanism

The key structural observation is that **the weight tile is dequantized once into
threadgroup memory and then reused `BM` times**. So the unpack cost amortised per
MAC is `ops_per_weight / BM` — independent of `BN` and `BK`.

`dequantize<U, N, bits=4>` (`quantized.h:521-527`, byte-identical to
`quantized_nax.h:524-530`):

```cpp
U s[2] = {scale, scale / static_cast<U>(16.0f)};
for (int i = 0; i < (N / 2); i++) {
  w_local[2*i]     = s[0] * (w[i] & 0x0f) + bias;
  w_local[2*i + 1] = s[1] * (w[i] & 0xf0) + bias;
}
```

Per weight that is: one AND, one int→float convert, one FMA, one threadgroup
store — **4 ops** — plus the amortised `s[1]` multiply and the packed-byte load,
giving **5** as an upper bound. Note the high nibble is deliberately left
**unshifted**, compensated by `s[1] = scale / 16`. There is therefore **no shift
to remove on either lane**; the obvious "strength-reduce the unpack" idea is
already taken.

| kernel | BM×BN×BK | `n_reads` | weights/tile | MACs/tile | ALU/MAC band |
|---|---|---|---|---|---|
| `affine_qmm_t` (non-NAX) | 32×32×32 | 4 | 1024 | 32768 | **12.50–15.62 %** |
| `affine_qmm_t_nax` | 64×64×64 | 16 | 4096 | 262144 | **6.25–7.81 %** |

Structural differences between the two, from primary source:

- Non-NAX `qmm_t_impl` (`quantized.h:1355+`, tile loop `:1420-1480`) stages
  **both** `x` and `w` through threadgroup memory (`Xs`, `Ws`), using
  `mlx::steel::BlockMMA` and `mlx::steel::BlockLoader` for `x` and
  `QuantizedBlockLoader` (`:572`) for `w`. Defaults `BM=32, BK=32, BN=32`,
  `WM=WN=2`, threadgroup size 128.
- NAX `qmm_t_nax_tgp_impl` (`quantized_nax.h:938`) stages **only** `w`; `x` is
  loaded device→register through `NAXTile<T,TM,TK> Atile; Atile.load(x + kk1, K)`
  with no `Xs` staging at all. Defaults `BM=64, BK=64, BN=64`, `WM=WN=2`,
  threadgroup size 128, `AccumType = float`, `SK = 32`, with `dispatch_bool` on
  aligned-M / aligned-N.
- **Neither has double buffering or software pipelining.** Both tile loops are
  strictly `threadgroup_barrier → load → threadgroup_barrier → mma`. Scalar
  dequant and the hardware MMA are therefore serialised within a threadgroup and
  their times **add** rather than overlap. That is what makes the unpack ALU cost
  visible at all.

### 5.3 Why this does not become a kernel edit

The ALU/MAC ratio is `ops_per_weight / BM`. To move it you either reduce
`ops_per_weight` — which the `s[1] = scale/16` trick has already minimised, there
is no shift left to delete and the AND, convert, FMA and store are all load-bearing —
or you raise `BM`.

`BM` is not mine to raise. It is chosen in `qmm()` at `quantized.cpp:717-720` and
in `qmm_nax()` at `:491-495`, both in the **read-only** dispatcher, and it is
then baked into the requested kernel name (`:722-735`, `:499-522`) before any
editable code executes. A kernel template compiled for a different `BM` would
simply never be requested.

Introducing double buffering inside the editable headers is the one remaining
theoretical construction, but it is not a *removable-byte or removable-ALU*
construction — it is a latency-hiding rewrite of the tile loop with a
threadgroup-memory cost, and the assignment gates a kernel edit on having a
concrete removable-byte/ALU construction. It also lands squarely on the
exactness-critical path and would require the full bitwise-vs-`M=1` parity gate.
I am not spending the remaining budget on it, and I would want H1 resolved first
regardless — see 5.4.

### 5.4 The H1 consistency check, and why H1 is the gating question

E16 was measured on **this local host**, where `is_nax_available()` is False, so
its 400 GEMM dispatches ran `affine_qmm_t` at `BM=32`. Its measured residual
φ = 12.942 % of P falls **inside** the BM=32 band (12.50–15.62 %) and **outside**
the BM=64 band (6.25–7.81 %).

That is a consistency check on the ALU model — a reassuring one, since the model
was built from source without reference to φ — and **not** a resolution of H1,
which is about the ranked box.

Its corollary is the important part: **if the ranked host does take the NAX
branch, roughly half of the locally-measured 12.942 % is already structurally
collected there by the wider tile, and the local prize does not transfer.** This
is exactly why H1 gates the value of the whole line of work, and why an
UNRESOLVED H1 is a genuine stop rather than a technicality.

## 6. What alphonse's 12.942 % actually describes

*Attribution: this figure is alphonse's, from E16
(`research/results/qwen38-r1-e16-prefill-ladder-adjudication.md`, PR #18), and is
referenced as "alphonse's attribution" at `senpai/campaign-ledger.md:246-247`.*

E16's budget is the signed identity

```text
P = gemm_at_ceiling + nongemm + dequant_overhead - overlap_credit
```

with these terms:

| term | seconds | % of P |
|---|---|---|
| `P` | 4.004000009 | 100 % |
| `gemm_at_ceiling` | 3.3693021188662633 | 84.148 % |
| `nongemm` | 0.21271352600006566 | 5.313 % |
| `floor_subtotal` | 3.582015644866329 | 89.461 % |
| `residual` | 0.42198436413367135 | 10.539 % |
| `dequant_overhead` | 0.5182019801337252 | **12.942 %** |
| `overlap_credit` | −0.09621761600005385 | −2.403 % |

with `closure_error_seconds = 0.0` **by construction**, and supporting terms
`gemm_seconds_measured = 3.8875040989999885`,
`gemm_tflop_total = 24.93751230464`,
`gemm_tflops_achieved = 6.414787398180458`,
`ceiling_tflops = 7.401388009998707`,
`gemm_fraction_of_ceiling = 0.8667005958226446`.

**Checking the algebra reveals what the number is.**
`dequant_overhead ≡ gemm_seconds_measured − gemm_at_ceiling` exactly:
`3.8875040989999885 − 3.3693021188662633 = 0.5182019801337252`, identical to the
reported figure to the last digit. My script asserts this identity.

So **the 12.942 % is not a measurement of dequantization.** It is the entire gap
between achieved quantized-GEMM throughput (6.4148 TFLOP/s) and a **dense bf16**
reference rate (7.4014 TFLOP/s) that was itself measured at a **single** shape
(512×5120×17408). Everything that makes the real census slower than that one
reference rate is inside the number:

- genuine dequant ALU and metadata cost — the part the assignment is hunting;
- tiling and occupancy differences between the quantized and dense kernels;
- per-dispatch launch overhead across 496 dispatches;
- shape-dependent efficiency across a census whose `N` ranges from 48 to 17408,
  which the single-shape reference cannot represent;
- the split-K reduction round trip on 96 of those dispatches;
- any accumulator-boundary conversion.

E16 labels it a *residual*, which is correct and honest. The consequence for E18
is that 12.942 % is an **upper bound** on the dequant prize, and the removable
part is a strict subset — plausibly a small one, since the closure is by
construction and absorbs every unmodelled effect. Section 5 finds that the
byte-traffic subset is ~0.19 % of P and the ALU subset is bounded by
`ops_per_weight / BM` with no submittable way to change either factor.

For completeness, E16's value conversions, which are what would turn a saved
fraction into score: `p(512) = 4.003337 / 12.049719 = 0.3322361`,
`R = 3.0972967` (projected post-merge), one frontier step = `0.0122890` points,
1 % of P removed = `0.006968` points = `0.567` frontier steps, and the full
`φ = 0.12942` would be `0.090180` points ≈ `7.34` frontier steps. Note `p` is
base-dependent and `b85e782` would require re-measuring `D_mtp`. Since the
removable subset is what matters and it is far below φ, these conversions are
recorded for the next reader rather than claimed.

## 7. Pre-timing falsification statement

**No timed leg was run in E18**, so this statement did not gate anything. I am
recording it anyway so the deliverable is complete and so the next agent inherits
the commitment rather than re-deriving it.

Signed, pre-timing, `qwen-thorfinn`:

> Had E18 reached a timed leg, the claim under test would have been: *a kernel
> edit inside the editable surface reduces measured prefill seconds `P` by at
> least 2 % relative to a fresh, unchanged base measured in the same thermal
> session on the same host, with bitwise-identical tokens.*
>
> I would have counted the experiment **falsified** if any of the following held:
> (a) the candidate's `P` improvement was under 2 % of base `P`; (b) any cell of
> the bitwise-vs-`M=1` parity gate differed, including on a refactor I believed
> to be pure; (c) the positive control (`perturb` ×1.015625f, previously firing
> 56/96) failed to fire, since a silent control means the gate proves nothing;
> (d) the improvement appeared only in the local serial-to-MTP ratio and not in
> absolute candidate seconds per token against a fresh `BASE_SHA` run, since both
> local legs share the candidate build and a general GEMM win cancels in that
> ratio.
>
> I would have reported the result under whichever of those fired, without
> re-scoping the claim after seeing the numbers.

The falsification path that actually fired was upstream of timing: the mechanism
was eliminated by source analysis before any measurement was warranted, under
stopping rules (b) and (c).

## 8. Compliance notes

- **Kernel edits:** none. No file on the exactness-critical path was touched, so
  the bitwise-vs-`M=1` parity gate and its `perturb` positive control were not
  required and were not run.
- **JIT twin rule:** not triggered. No `.metal`/`.h` source was edited, so no
  `mlx-generated/*.cpp` twin update and no `python3 research/twin_audit.py` run
  was required. Recorded for the next agent: the NAX quantized kernels are JIT
  (`jit_kernels.cpp:1116`), so their runtime-effective source is
  `mlx-generated/{quantized_nax,fp_quantized_nax,gemm_nax}.cpp`, and any future
  edit there must update the readable header and the generated twin together.
- **Per-arm worker `sha256`:** not applicable. No arm was run, no worker was
  built, and `benchmark-qwen-mtp.sh` was not invoked.
- **`research/capture-cli.sh`:** not applicable — mandatory for timed runs, and
  no timed run occurred.
- **`research/await-lock-then-run.sh`:** not invoked. **I did not take the GPU
  lock at any point.** I am third in the queue behind PR #17 and PR #19, no
  handoff signal reaches me, and E18 needed no GPU.
- **Four trace files per arm:** not applicable, no arm.
- **Ratio labelling:** no ratio is reported in this document, so neither
  `prefill-inclusive` nor `decode-only` applies. Where §6 quotes E16's
  conversions they are reproduced with E16's own framing and are not restated as
  E18 measurements. For the record, the conversion context is that prefill is
  15.8–18.0 % of the MTP leg and edward's decode-currency→score factor is
  `0.84228`.
- **Roofline:** quoted as **226.9 GB/s** (`227,128,791,836.97 B/s`) everywhere.
  The 273 GB/s figure is not used.
- **Swift invocations:** none were needed. No `swift package resolve` or
  `swift package update` was run.
- **Live campaign facts** as given in the assignment, unchanged by this work:
  frontier receipt `ba493f74-c0fe-440a-a956-f77d26232e54`, source
  `156b5b75bdfac82ae406487f531fd991e7fdfd30`, official score
  `2.95338624520432`, plausibility ceiling `5.0` (operator commit `a5854b97`),
  headroom `+2.047`.

## 9. Reproduction

```bash
git checkout dbfef04721f9d2bed33ebee0c3b2e43a13df39d6
python3 research/e18_dispatch_table.py            # full table, census, budget, ALU model
python3 research/e18_dispatch_table.py --json     # machine-readable
python3 research/e18_dispatch_table.py --M 1      # decode-shaped sweep
```

The script is pure arithmetic over the dispatcher's own logic. It needs no GPU,
no model weights and no lock, and it runs in well under a second.

Every invocation first runs `self_check()`, which asserts the load-bearing
claims of this document and crashes rather than printing a quietly wrong table:

- `dequant_overhead == gemm_seconds_measured - gemm_at_ceiling` **exactly** (§6);
- the census FLOP total matches E16 at **zero** relative error (Correction 4);
- every byte-traffic term is under 2 % of P and the compute-bound factor exceeds
  50× (stopping rule (c), §5.1);
- `split_k == 1` for every projection with `N >= 1024` (Correction 1, first half);
- `split_k == 16` on exactly **96** dispatches (Correction 1, second half);
- φ falls inside the BM=32 band and outside the BM=64 band (§5.4).

If a future base changes any of these, the script fails instead of silently
invalidating the verdict.

## 10. Suggested follow-ups (not implemented)

1. **Resolve H1 with one read outside this checkout — now specifically the
   GitHub Actions job log.** A read showing the ranked box's macOS build string
   and its `applegpu_*` architecture would settle whether the ranked prefill runs
   `affine_qmm_t` at BM=32 or `affine_qmm_t_nax` at BM=64. That single fact
   decides whether roughly half the locally-measured 12.942 % is already
   collected on the ranked host, and therefore whether *any* prefill-dequant work
   transfers. **§11.1 closes the Yukon half of this: the receipt telemetry
   provably carries no host, OS or GPU field**, so the Actions job log is the
   only remaining resolving read. §11.4 supplies indirect support for the NAX
   branch but does not settle it. If it is cheap to add, having the ranked
   workflow print `sw_vers -productVersion` and the device architecture string
   would permanently unblock this class of question.
2. **`steel/gemm/**` is editable (Correction 5).** This is new information and it
   may reopen NAX GEMM work that was previously closed on the belief that those
   headers were frozen. Worth a scan of the closed-experiment list for anything
   whose stated reason for closure was "not editable".
3. **Re-check `rg`-derived campaign negatives.** `rg` is not installed on this
   host. Any conclusion of the form "grep found no hits" that was reached with
   `rg` may be a false negative. I hit exactly this on Correction 5.
4. **Two ranked boxes, not one.** `fixtures/qwen3_8_27b_mtp_track.json:156`
   records two boxes with different pinned serial means. If they differ in OS or
   silicon generation, H1 could have different answers on different ranked runs,
   and a candidate tuned for one branch could regress on the other.
5. **Neither GEMM tile loop is double-buffered.** Both are strictly
   `barrier → load → barrier → mma`, so dequant and MMA serialise. This is the
   only structural inefficiency I found that is reachable from inside the
   editable headers. It is not a removable-byte/ALU construction and so is out of
   scope for E18, but it is a real, specific, source-backed target if someone
   wants to spend a parity-gated kernel edit on latency hiding.

## 11. Ranked-hardware evidence from Yukon receipt telemetry

Added after the sections above were written. While GitHub was unavailable I
queried Yukon, which was up, and found that ranked receipts carry a
`prefill_seconds_per_token` field. This is **ranked-hardware prefill telemetry**,
and it bears directly on this assignment. It changes the size of the prize
without changing any verdict above.

Source: `GET /api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true`
(393 submissions, 107 of them scored). Frontier receipt
`ba493f74-c0fe-440a-a956-f77d26232e54`, solver `yijunyu`, `promotedSourceRef`
`156b5b75bdfa`, `officialScore` `2.95338624520432` — matching the campaign
frontier exactly.

### 11.1 The Yukon receipt does not contain the fact H1 needs

Follow-up 1 above proposed "a Yukon receipt **or** a GitHub Actions job log". The
Yukon half is now closed. The complete `officialMetrics` key set is:

```text
accepted_pair_count, aggregation, baseline_serial_seconds_per_token_mean,
candidate_mtp_seconds_per_token_mean, commit, decode_speedup_ceiling,
decode_speedup_floor, decode_tokens, median_rule, mode,
mtp_decode_speedup_median, mtp_decode_speedup_min,
mtp_decode_speedup_pooled_ratio_of_means, mtp_decode_speedup_raw_median,
mtp_depth, mtp_max_draft_depth, pairs_per_prompt, parity_all_ok, per_prompt,
prefill_seconds_per_token, prompt_count, qwen_mtp_weights_hash, score_anchor,
scoring_normalized
```

A scan of `officialMetrics` for `applegpu`, `gpu`, `macos`, `sw_vers`,
`buildVersion`, `host`, `device`, `arch`, `nax`, `runner`, `m5`, `chip`,
`hardware`, `kernel`, `metal`, `darwin` and `generation` returns **zero hits**,
in keys and in values.

Being precise about the scope, because the same scan over the *whole submission
row* does match: the only hits anywhere in a row are `macos`, `host`, `arch`,
`kernel` and `metal` inside the `note` field — an 11,790-character block of
solver-authored prose describing that solver's own machine. That is free text
submitted by a competitor, not machine-generated telemetry, and it does not
identify the ranked box. The reproduction script asserts both facts: zero
needles in `officialMetrics`, and every whole-row hit confined to `note`.

So no host, OS or GPU field exists in the receipt telemetry. **H1 remains
UNRESOLVED, and the only remaining resolving read is the GitHub Actions job
log.**

### 11.2 Ranked prefill is 6.20 % of the candidate leg, not 15.8–18.0 %

The assignment states prefill is 15.8–18.0 % of the MTP leg. That figure is
local. On ranked hardware:

| quantity | value |
|---|---|
| `prefill_seconds_per_token` | `0.0010361296590417624` |
| seed tokens / decode tokens | 512 / 512 |
| ranked prefill wall seconds | **0.530498** |
| candidate leg (`0.016714676981791854 × 512`) | 8.557915 |
| serial leg (`0.03805219699279405 × 512`) | 19.482725 |
| **prefill share of candidate leg** | **6.1989 %** |
| prefill share of serial leg | 2.7229 % |

Seed and decode windows are both 512, so total prefill seconds is 0.5305
regardless of which token count the per-token figure is normalised by. Per
`fixtures/qwen3_8_27b_mtp_track.json`, seed prefill is charged *inside* the
decode measurement, so the 6.1989 % is a share of a leg that already contains it.
If instead prefill were additive, the share would be 5.84 %. Neither reading
approaches 15.8 %.

### 11.3 Ranked prefill has never moved, on any public submission that reports it

First, the sample. 238 of the 393 submissions carry `officialMetrics`, but only
**107** carry `prefill_seconds_per_token`; on the other 131 the key is present
and `None`. Sorting by `createdAt` shows exactly **one** on/off transition, at
`2026-08-16T13:35:19.579Z` — a clean schema addition, not sporadic reporting. So
the telemetry covers a window of about 23 hours,
`2026-08-16T13:39:59Z … 2026-08-17T13:04:22Z`. That window is short, but it is
not thin: those 107 receipts are **87 distinct candidate commits from 50
distinct solvers** (99 rejected, 8 accepted — rejected runs still executed the
real benchmark and produced real telemetry). I therefore claim "never moved
across every receipt that reports it", not "never moved in the whole
competition".

Across those 107 receipts:

| statistic | value |
|---|---|
| min / max `prefill_seconds_per_token` | `0.0010345071` / `0.0010682998` |
| spread (max/min) | **1.0327×** |
| coefficient of variation | 1.334 % |
| official score range over the same 107 | 1.2098 … 2.9534 (**2.44×**) |
| Pearson *r*(score, prefill) | **−0.0854** |

The decisive pair: the worst-scoring candidate (`1.2098`) and the best-scoring
candidate (`2.9534`) report prefill of `0.0010361077` and `0.0010361297` — equal
to five significant figures, a ratio of `1.0000×`, across a 2.44× spread in
decode score.

Two readings are consistent with this, and the receipt does not document which:
either the reported prefill is measured on the **pinned serial leg** and is
candidate-independent by construction, or it is the candidate's and no public
submission has ever moved it. The second reading is not a weak one — 87 distinct
candidate commits from 50 solvers span a 2.44× decode-score range while their
prefill stays inside 1.03×, so if the number is candidate-side then a large,
diverse population of optimisers has collectively left it untouched.
**Under both readings the ranked prefill workload is ~0.53 s**, because both
legs push the same 512-token seed through the same fixed target checkpoint with
the same 24.94 TFLOP census. I flag the ambiguity rather than assert the stronger
claim.

### 11.4 Compute scales 4.5× more than bandwidth — quantitative support for the NAX branch

This does not resolve H1, but it is the strongest indirect evidence available
without the Actions log.

| | local (M4 Pro, NAX off) | ranked | ratio |
|---|---|---|---|
| prefill wall seconds (512 seed) | 4.004000 (E16) | 0.530498 | — |
| prefill throughput, 24.94 TFLOP census | 6.228 TFLOP/s | **47.008 TFLOP/s** | **7.548×** |
| decode weight-streaming bandwidth | 227.1 GB/s (measured roofline) | ≥ 378.8 GB/s (effective, from serial s/token) | ≥ 1.668× |

The **compute-bound** prefill scales `7.548×` while the **memory-bound** serial
decode scales at most `1.668×` — compute scales **4.53× more than bandwidth**
between the two hosts. Both figures come from the same two machines running the
same target checkpoint.

A pure core-count-and-clock generational step would move both roughly together.
A 4.5× divergence in favour of the compute-bound phase is the signature of a
matrix-accelerator path that is active during the M=512 GEMM prefill and
irrelevant to M=1 decode — where, per §2, the QMV and split-K kernels are
byte-identical with and without NAX. That is exactly the NAX/BM=64 arm of H1.

Caveats, stated plainly: the ranked box's core count and peak bandwidth are not
published in this checkout, the 378.8 GB/s is an effective lower bound rather
than a peak, and the local 4.004 s comes from E16's instrumentation rather than
from a measurement I took. This is corroboration, **not** a substitute for
reading the ranked `applegpu_*` string.

### 11.5 The ranked prize is at most ~1–2 frontier steps, and is not reachable

Combining §11.2 with the §5 bands, and using score `2.95338624520432` and one
frontier step `= 0.0122890` points:

| assumed removable fraction of prefill | Δt | leg reduction | score gain | frontier steps |
|---|---|---|---|---|
| E16 residual φ = 12.942 % (upper bound) | 68.66 ms | 0.8023 % | **+0.02389** | **1.94** |
| BM=32 ALU band, high (15.62 %) | 82.86 ms | 0.9683 % | +0.02888 | 2.35 |
| BM=32 ALU band, low (12.50 %) | 66.31 ms | 0.7749 % | +0.02306 | 1.88 |
| **BM=64 NAX band, high (7.81 %)** | 41.43 ms | 0.4841 % | **+0.01437** | **1.17** |
| **BM=64 NAX band, low (6.25 %)** | 33.16 ms | 0.3874 % | **+0.01149** | **0.93** |

Every row assumes **100 %** of the stated fraction is removed, which is not
achievable: §5 shows `ops_per_weight` is already minimal because
`dequantize<U,N,4>` leaves the high nibble unshifted and compensates with
`s[1] = scale/16`, and §1 Correction 5 shows `BM` is set by the read-only
dispatcher.

The local framing implied φ was worth `7.34` frontier steps. On ranked hardware
the same φ is worth `1.94` steps, and under the NAX branch that §11.4 supports it
is worth `0.93–1.17` steps. **The prize is 4–8× smaller than the local analysis
suggested, and it remains structurally unreachable.** This does not change the
verdict in the Headline; it raises confidence in it.

### 11.6 Side finding: `program.md`'s plausibility ceiling is stale — the live value is 5.0

Not part of E18, but it fell out of the same data and it corrects a source-of-truth
document, so it should not be lost.

`program.md` states the published median has "an operator-set plausibility gate
at `3.0`". The receipts carry the gate as `decode_speedup_ceiling`, and sorting
all 238 metered receipts by `createdAt` shows exactly **one** transition:

```text
2026-08-17T10:58:40.600Z   ceiling 3
2026-08-17T11:10:46.796Z   ceiling 5     <- raised here
... all 11 subsequent receipts, through 2026-08-17T13:04:22.040Z, carry 5
```

So the operator **raised the ceiling from 3 to 5**, and the current live value is
**5.0**. This independently confirms §5.10 of the assignment brief, which already
states the ceiling is `5.0` and attributes the raise to operator commit
`a5854b97`; the receipts add the exact wall-clock moment it took effect. The
assignment's stated ceiling and headroom of `+2.047` are **correct and current**;
it is `program.md` that is stale. The frontier receipt `ba493f74` records `3`
because it ran at `09:57Z`, before the raise — it is a correct snapshot of a
superseded policy, not a contradiction.

I flag this because I initially read the frontier receipt's `3` as contradicting
the assignment and it does not — the assignment was right and my first reading
was wrong. The direction matters: had the true value been
`3`, the frontier at `2.95339` would sit `+0.0466` below the gate — about 3.8
frontier steps of total remaining headroom — and every prize calculation in the
campaign would be competing for a nearly exhausted budget. At `5.0` there is
`+2.047` of headroom and no such squeeze.

Two supporting facts: `decode_speedup_floor` is `0.9` on all 238 receipts and
never changed, and **no receipt has ever reported a raw median above `3.0`**
(max observed `2.9534`). The gate has therefore never actually rejected a
submission; the raise was pre-emptive. Consistent with that, all 99 rejections
among the 107 prefill-reporting receipts carry `rejectionReason` `"score did not
improve current best"`, and the 8 non-rejected ones are exactly the 8 with
`promotionStatus: "promoted"`.

Suggested follow-up for the advisor: update `program.md`'s `3.0` to `5.0`, or
re-derive it from the live receipt field rather than hard-coding it.

### 11.7 Reproduction

```bash
export PATH="${HOME}/.local/bin:${PATH}"
curl -s -H "Authorization: Bearer $YUKON_API_TOKEN" \
  "https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true" \
  -o yukon_subs.json
python3 research/e18_yukon_prefill.py yukon_subs.json
```

`research/e18_yukon_prefill.py` reproduces every number in this section and
self-checks the frontier receipt id, source ref and official score against the
values quoted here.

---

```senpai-result:v1
{
  "schema": "senpai-result:v1",
  "assignment_id": "qwen38-r1-e18-prefill-dequant-prize",
  "revision_id": "r1",
  "student": "qwen-thorfinn",
  "repo": "morganmcg1/qwen38-challenge_senpai",
  "pr_number": 20,
  "base_ref": "senpai/qwen38-mtp-r1",
  "base_sha": "422db0451e92a8c9b70ff2900874fa3b69fab261",
  "analysis_head_sha": "dbfef04721f9d2bed33ebee0c3b2e43a13df39d6",
  "status": "succeeded",
  "label": "not-useful",
  "terminal": true,
  "pending_arms": false,
  "kernel_edit_shipped": false,
  "timed_leg_run": false,
  "gpu_lock_taken": false,
  "wandb_runs": [],
  "yukon_submission_id": null,
  "host": {
    "machine": "Apple M4 Pro",
    "memsize_bytes": 51539607552,
    "macos": "26.5.2",
    "macos_build": "25F84",
    "swift": "6.3.3",
    "is_ranked_m5": false,
    "thermal_sessions": 0,
    "dirty": true
  },
  "hypotheses": {
    "H1": {
      "statement": "ranked prefill dispatches through qmm_nax rather than the non-NAX qmm",
      "verdict": "UNRESOLVED",
      "missing_facts": [
        "ranked box macOS version (gate requires >= 26.2)",
        "ranked box literal applegpu_* architecture string (supplies gen and arch size char)"
      ],
      "smallest_resolving_read": "GitHub Actions job log for the ranked run; neither fact is emitted anywhere in this checkout",
      "yukon_receipt_ruled_out": "officialMetrics provably carries no host, OS or GPU field (section 11.1); the Yukon half of this follow-up is closed",
      "indirect_support_for_nax_branch": "compute scales 4.53x more than bandwidth between local and ranked hosts (section 11.4); corroboration only",
      "assignment_premise_refuted": "MLX_METAL_NO_NAX is not defined in the SwiftPM Cmlx target and cannot serve as the off control"
    },
    "H2": {
      "statement": "the NA=4/NA=5 cross-row register-cliff mechanism does not govern the prefill GEMM",
      "verdict": "CONFIRMED",
      "basis": "structural; affine_qmv_fast is reachable only via dispatch_qmv when M < vector_limit, and all 496 prefill weight projections have M=512 >= vector_limit=10"
    }
  },
  "phase1": {
    "advisor_split_k_arithmetic": "CORRECT",
    "advisor_dead_code_claim": "REFUTED",
    "split_k_16_dispatches_per_prefill": 96,
    "kernel_census": {
      "affine_qmm_t": 400,
      "affine_qmm_t_splitk": 96,
      "affine_qmv_fast": 1
    },
    "model_is_fused": false,
    "lm_head_prefill_rows": 1
  },
  "phase2": {
    "roofline_bytes_per_second": 227128791836.97,
    "gemm_flops": 24937512304640,
    "gemm_flops_e16_relative_error": 0.0,
    "quantized_weight_bytes": 14412349440,
    "scale_bias_bytes": 1601372160,
    "arithmetic_intensity_flop_per_byte": 1730.3,
    "compute_bound_factor": 61.3,
    "weight_traffic_fraction_of_gemm_seconds": 0.01632,
    "scale_bias_fraction_of_P": 0.001761,
    "splitk_roundtrip_fraction_of_P": 0.000166,
    "alu_per_mac_band_bm32": [0.1250, 0.1562],
    "alu_per_mac_band_bm64": [0.0625, 0.0781],
    "e16_phi": 0.12942,
    "e16_identity_verified": "dequant_overhead == gemm_seconds_measured - gemm_at_ceiling exactly",
    "nax_arm_label": "source analysis, not locally measurable"
  },
  "ranked": {
    "source": "public Yukon receipts, GET /api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true",
    "reproduction_script": "research/e18_yukon_prefill.py",
    "frontier_receipt_id": "ba493f74-c0fe-440a-a956-f77d26232e54",
    "frontier_source_ref": "156b5b75bdfac82ae406487f531fd991e7fdfd30",
    "frontier_official_score": 2.95338624520432,
    "prefill_seconds_per_token": 0.0010361296590417624,
    "prefill_wall_seconds": 0.530498,
    "candidate_leg_seconds": 8.557915,
    "serial_leg_seconds": 19.482725,
    "prefill_share_of_candidate_leg": 0.061989,
    "prefill_share_if_additive": 0.058371,
    "assignment_premise_share": [0.158, 0.180],
    "assignment_premise_refuted": true,
    "prefill_tflops": 47.008,
    "local_prefill_tflops": 6.228,
    "compute_scaling_ratio": 7.548,
    "bandwidth_scaling_ratio_lower_bound": 1.668,
    "compute_vs_bandwidth_divergence": 4.53,
    "receipt_has_host_field": false,
    "telemetry_receipts": 107,
    "telemetry_receipts_with_metrics_total": 238,
    "telemetry_independent_commits": 87,
    "telemetry_distinct_solvers": 50,
    "telemetry_window_utc": ["2026-08-16T13:39:59.798Z", "2026-08-17T13:04:22.040Z"],
    "prefill_spread_max_over_min": 1.0327,
    "prefill_coefficient_of_variation": 0.01334,
    "score_range_over_same_receipts": [1.2098, 2.9534],
    "pearson_r_score_vs_prefill": -0.0854,
    "prize_frontier_steps_phi": 1.94,
    "prize_frontier_steps_bm32": [1.88, 2.35],
    "prize_frontier_steps_bm64_nax": [0.93, 1.17],
    "prize_frontier_steps_local_framing": 7.34,
    "decode_speedup_ceiling_current": 5,
    "decode_speedup_ceiling_previous": 3,
    "decode_speedup_ceiling_raised_at_utc": "2026-08-17T11:10:46.796Z",
    "decode_speedup_ceiling_operator_commit": "a5854b97",
    "decode_speedup_ceiling_assignment_value_confirmed": true,
    "decode_speedup_ceiling_program_md_value_stale": 3.0,
    "decode_speedup_floor": 0.9,
    "receipts_ever_exceeding_old_ceiling": 0
  },
  "stopping_rules_fired": ["b", "c"],
  "corrections": [
    "split_k arithmetic correct but qmm_t_splitk is NOT dead code: in_proj_b/in_proj_a fire it 96x per prefill",
    "model is unfused; advisor shape list numerically wrong",
    "lm_head at prefill is M=1 QMV, not M=512 GEMM",
    "corrected census reproduces E16 FLOP total at zero relative error",
    "backend/metal/quantized.cpp is READ-ONLY but kernels/steel/gemm/** IS editable via a benchmark.json directory entry",
    "assignment premise that ranked prefill is 15.8-18.0 percent of the candidate leg is refuted: the frontier receipt puts it at 6.1989 percent",
    "program.md's 3.0 plausibility ceiling is stale; receipts independently confirm assignment section 5.10, with decode_speedup_ceiling raised 3 -> 5 at 2026-08-17T11:10:46.796Z"
  ]
}
```
