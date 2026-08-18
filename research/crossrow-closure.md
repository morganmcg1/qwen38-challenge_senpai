# Closing the crossrow QMV micro-optimization program

Advisor analysis, 2026-08-18. Zero GPU time. Written at advisor HEAD `d11e01e`,
after the organizer promoted `caec88d4` (score 3.14642585386152) — a submission
whose complete diff is byte-for-byte the advisor's own commit `67856b5`
(`DIRECT_NIBBLES` at verify widths M=3,4,5).

**Claim.** Under this benchmark's two binding constraints — the exact-token
fidelity gate and the `editablePaths` contract — there is no remaining
bit-exact ALU or bandwidth win available inside the multi-row crossrow QMV
kernels. `caec88d4` captured the last one. Every further verify-side gain must
come from somewhere other than this kernel.

This note enumerates the candidate space and records why each entry is closed,
so that no future experiment spends GPU hours rediscovering it.

## 1. Why this kernel mattered

E15's attribution puts `verify_overhead_residual_seconds = 11.6907` of
`13.996965 s` of control decode work: **85.8 % of decode is target
verification**, leaving ~16 % for all draft-side work combined. The multi-row
crossrow affine4 QMV kernels *are* the verify side. That is why the campaign
has spent so much of its attention here, and why closing the space is worth
writing down explicitly.

## 2. The two binding constraints

### 2.1 Bit-exactness forbids reassociation

`benchmark.json` `/scoring` sets

```
tokenFidelityGate = trusted-sequential-reverification-exact-token-match
```

The candidate's verify argmax must reproduce the trusted 4-bit serial reference
exactly. Any change to floating-point accumulation *order* can flip a low-order
bit, which can flip an argmax on a near-tie, which fails the gate. So every
optimization below is evaluated under the rule: **the per-row accumulation
expression tree may not be reassociated.**

This is also, retroactively, the mechanism behind a campaign negative that was
banked as if it were a physics result: "low-bit on the verify-side head at
M=4..9". That is not a speed/quality tradeoff that happened to lose. It is a
*correctness* violation — a 3-bit verify head cannot reproduce the 4-bit
reference. The negative is correct; its recorded justification was wrong.

Note what this constraint does *not* forbid, and why `caec88d4` is legal:
`DIRECT_NIBBLES` replaces `load_vector<T, float, 4, 4>` with four direct
`static_cast<float>` conversions. `load_vector` writes
`x[i], x[i+1]/16, x[i+2]/256, x[i+3]/4096` and the w-side compensates with
shifted masks. Dividing by a power of two changes only the exponent field; the
accumulation order is untouched. Bit-exact by construction, independent of M
and NA — and confirmed by a 96-cell parity gate returning `0 differing`.

### 2.2 The host dispatch geometry is on the trusted surface

This is the constraint that had been assumed rather than checked, and it is the
new result in this note.

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:251-254`:

```cpp
  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);                 // 64 threads = 2 simdgroups
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
```

Verified facts:

- That file is **tracked but absent from `benchmark.json` `editablePaths`.**
  The editable quantized surface is `kernels/quantized*.{h,metal}` and
  `mlx-generated/quantized*.cpp` only.
- `mlx-generated/quantized.cpp` — which *is* editable — contains **zero**
  occurrences of `MTL::Size`, `grid_dims`, or `group_dims`. It is a pure
  `const char* quantized()` JIT source string carrying the kernel text, not the
  host dispatch.
- `git ls-files | grep 'backend/metal/quantized.cpp$'` returns **exactly one**
  path. There is no second, editable copy.

Consequences, all now hard rather than conventional:

- The threadgroup is 32x2 = 64 threads = **2 simdgroups**, fixed.
- The output tile is **8 rows** (`bn = 8`), fixed.
- Therefore `rows_per_simd = 4` is **forced**, and matches the kernel's
  `out_row = tid.y * 8 + simd_gid * 4`.
- The grid's x extent is `M`, so `ntg.x == M` exactly — which is what makes
  crossrow selection an in-kernel `switch (ntg.x)` rather than a kernel-name
  variant. Kernel names are built at `quantized.cpp:261-269` from only
  `group_size`, `bits`, and `batch`; **M never enters the symbol.**

The kernel's own comment at `kernels/quantized.h:959-967` — "the *frozen host*
launches M x-groups for each 8-output tile" — turns out to be a literal
statement about the editable-path boundary, not a description of local
convention. Credit to student qwen-thorfinn, who established the kernel-naming
half of this independently in `research/e22-prereg.md`.

## 3. The candidate space, enumerated and closed

Per k-block (`block_size = 512`), per lane, at `rows_per_simd = 4`, the
remaining work decomposes as:

| term | ops | scales with |
|---|---|---|
| w-side nibble extract + int->float | 128 | rows_per_simd (**not** NA) |
| core FMA (`partial[r] += a0*c0 + ...`) | 80·NA | rows_per_simd x NA |
| x-side convert + sum (DIRECT_NIBBLES) | 32·NA | NA only |
| epilogue (`acc[r] += scale*partial + sums*bias`) | 12·NA | rows_per_simd x NA |

At NA=3 that is 128 + 240 + 96 + 36 = **~500 ops per lane per k-block**. Note
the w-side extraction is now the largest non-FMA term, and it is independent of
NA — a consequence of `caec88d4` having already removed the x-side's
power-of-two multiplies (the non-direct path costs ~42·NA where the direct path
costs 32·NA, matching the previously banked "12·NA power-of-two multiplies").

### 3.1 Fuse the four-term `partial[r]` sum into an FMA chain — **REJECTED**

Would save ~16·NA, about 20 % of the dominant term. **Reassociates.** Violates
§2.1.

### 3.2 FMA the epilogue — **REJECTED**

`acc[r] = fma(scale, partial, acc); acc[r] = fma(bias, sums, acc)` saves 4·NA.
**Reassociates.** Violates §2.1.

### 3.3 Hoist `sums * bias_local[r]` out of the k-loop — **IMPOSSIBLE**

Not a policy rejection; it is arithmetically unavailable.
`group_index = row * in_vec_size_g + k/64 + simd_lid/4` makes `bias_local[r]`
genuinely per-block, and each lane's 16 values lie entirely within one
64-element quantization group. There is no loop-invariant to hoist.

### 3.4 Bit-trick int->float — **REJECTED**

`as_type<float>(0x4B000000 | n) - 8388608.0f` costs 2 ops against the 1-op
hardware convert. Strictly worse.

### 3.5 Raise `rows_per_simd` from 4 to 8 — **BLOCKED BY CONTRACT**

This one is bit-exact (each row's k-loop order is unchanged; only the row/lane
assignment moves) and would be worth roughly 10 %: the x-side term is
per-lane-not-per-row, so doubling rows per simdgroup amortizes it over twice
the rows. Concretely, at NA=3 the per-row cost falls from 500/4 = 125 to
904/8 = 113, i.e. **-9.6 % ALU**.

It requires `bn = 16`, or one simdgroup per threadgroup, in
`backend/metal/quantized.cpp`. Per §2.2 that file is **not editable**. Doing it
kernel-side alone — having simdgroup 0 cover all 8 rows — merely idles
simdgroup 1 and is strictly worse.

**This is the single largest bit-exact win identified in the kernel, and the
contract forecloses it.** Recording it here so it is not rediscovered as if it
were available.

### 3.6 Share the redundant x-side between the two simdgroups — **NET-NEGATIVE**

A real redundancy, verified at `kernels/quantized.h:1019-1020`:

```cpp
const device T* xm = x + (first_m + m) * in_vec_size + k
                   + simd_lid * values_per_thread + 4 * i;
```

This depends on `first_m` (= `tid.x * IPG`), `m`, `k`, `simd_lid`, and `i` — and
**not** on `simd_gid` or `out_row`. Both simdgroups in a threadgroup share
`tid.x` and sweep identical `k` and `i`, so lane `simd_lid = L` in simdgroup 0
and lane `L` in simdgroup 1 compute byte-identical `xc[0..3]`, `sums[m]`, and
`a0..a3`. The x-side conversion is performed **exactly twice per threadgroup**.

Eliminating it means staging `a0..a3` and `sums` through threadgroup memory
with a barrier. At NA=3 that is 4 i-values x 12 floats = 48 writes plus 48 reads
per lane per k-block, to save 96 ALU ops in one of two simdgroups. That is
**at least as many threadgroup-memory operations as the ALU it saves**, before
counting a `threadgroup_barrier` in the innermost loop, which serializes the two
simdgroups that currently run free. Rejected on arithmetic, not on taste.

(The associated *bandwidth* redundancy is negligible and should not be cited as
motivation: x traffic is `M x K x 2` bytes — ~30 KB for the lm_head at M=3 —
against ~715 MB of weight traffic for the same dispatch.)

### 3.7 Add `DIRECT_NIBBLES` to the `<T,2>` pair kernel — **PREDICTED ~0**

`case 2:` dispatches `qmv_fast_crossrow_affine4_g64<T,2>`, which has no
`DIRECT_NIBBLES` template parameter at all — the last width not covered by
`caec88d4`. It looks like an obvious gap. It is predicted to be worth nothing:
E15 Phase 1 measured M=2 b=4 at **0.9982x** the M=1 cost, which is the signature
of a purely bandwidth-bound kernel with perfect weight-stream amortization.
Cutting ALU in a kernel that is not ALU-bound buys nothing.

This is a **pre-registered falsifiable prediction**, not a conclusion: student
qwen-thorfinn's E22 C(M) curve measures the M=2 cell directly. If C(2)/C(1)
comes in meaningfully above 1.00, the prediction is wrong and the pair kernel
becomes worth tuning.

## 4. What this leaves

The crossrow QMV kernel is closed. The verify side is 85.8 % of decode, so the
remaining levers are the other factors in the same product:

1. **Cost per verify round as a function of width, C(M)** — feeds the depth
   policy, which currently prices the verify round as *constant in depth* via
   `threshold = h*(1.0+expected)/(1.0 + depth*h)` with
   `headStepCostRatio = 0.18`, even though the verify round runs at M = depth+1.
   Assigned: qwen-thorfinn, E22 Question 1.
2. **Where the 85.8 % actually goes** across {48 GDN, 16 full-attention, 64 MLP,
   LM head + top-2 reducer}. Assigned: qwen-askeladd, E20. Note that **48 of 64
   layers are GDN and have received almost no campaign attention.**
3. **How many dispatches a verify forward issues as a function of M.** Assigned:
   qwen-alphonse, E23 (static, zero GPU, blinded from E20).
4. **How many rows are verified at all.** Assigned: qwen-edward, E21.

Nothing in that list is a QMV kernel change. That is the point of this note.

## 5. Falsifiable content

- **F1.** `backend/metal/quantized.cpp` is absent from `editablePaths`; the
  editable `mlx-generated/quantized.cpp` contains zero `MTL::Size`; exactly one
  tracked copy of the host dispatch exists. *Check with `jq` over
  `benchmark.json` and `git ls-files`.* If false, §3.5 reopens and is worth
  ~10 % of the crossrow path.
- **F2.** The x-side pointer expression is independent of `simd_gid`.
  *Check at `kernels/quantized.h:1019-1020`.* If false, §3.6's premise is wrong.
- **F3.** C(2)/C(1) at bits=4 is ~1.00. *E22 measures this.* If false, §3.7
  reopens.
- **F4.** No shipped dispatch runs `bits != 4` at `M >= 2`. The 3-bit draft
  readout is unconditionally M=1 (`Qwen35.draftTokenID` reshapes to 1-D
  `[98336]`; `Qwen36MTPBlockSession.swift:877-893` slices `draftHidden` to
  `[1,1,H]`; the depth loop is strictly sequential), and the verify side is
  pinned to 4 bits by §2.1. If false, a 3-bit crossrow template — which does
  not exist anywhere in the tree; `grep -rn affine3` over both the kernels and
  `mlx-generated` directories returns nothing — would become worth writing.
