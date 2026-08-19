# E38 pre-registration — row blocks in the idle x-blocks at M=6

Committed **before** any E38 kernel code exists and **before** any E38 measurement.
Every number here is reproducible with `python3 research/e38_prereg.py`, which
self-tests its own invariants and fails closed.

- Assignment: PR #43, `qwen38-r1-e38-rowblocks-in-idle-xblocks-m6`, revision `r1`
- Base: `senpai/qwen38-mtp-r1` @ `54248ce258376db756be02fd65a814a903e2d601`
- Prior art: E33 (`qwen-thorfinn/m6-single-pass-rowblocked`, result `007933a`, base `4e5dc2b`)
- Kernel files are **byte-identical** between `4e5dc2b` and `54248ce`, so E33's
  measured cost tables are a valid comparison basis for E38.

---

## 1. Question

E33 turned M=6 into a single weight pass with `_m<T,6,6,true,2>` and a
**sequential** row-block loop, and measured `1.0150` — 1.5% *slower*. That change
bundled two effects that cannot be separated from a single number:

| effect | E33 |
|---|---|
| (i) activation tile doubled (each row block re-reads its inputs) | 288 → 576 issue units |
| (ii) grid halved (`IPG=6` ⇒ `ceil(M/IPG)=1` working x-block vs the shipped 2) | 2 → 1 x-block |

**E38 unbundles them.** It keeps (i) exactly and undoes (ii) by placing the two
row blocks in two *different* threadgroups — the ones the host already launches
and which today exit immediately at `if (first_m >= M) return;`
(`kernels/quantized.h:1171-1172`; `grid.x == M` from
`backend/metal/quantized.cpp:251-254`).

## 2. Arms

All three measured in **one locked session** against **one fresh base**.

| arm | instantiation | row-block placement | x-blocks | weight passes | act units | MAC units | wgt units | U |
|---|---|---|---|---|---|---|---|---|
| base | `_m<T,6,3,true,4>` | n/a (`rows_per_simd=4`) | 2 | 2 | 288 | 768 | 128 | 1184 |
| **(a) control** | `_m<T,6,3,true,2>` | sequential loop | 2 | 2 | 576 | 768 | 128 | 1472 |
| **(b) repair** | `_m<T,6,6,true,2>` | separate x-blocks | 2 | 1 | 576 | 768 | 64 | 1408 |
| *(E33, measured)* | `_m<T,6,6,true,2>` | sequential loop | **1** | 1 | 576 | 768 | 64 | 1408 |

Two clean isolations fall out, and they are the whole point of the design:

- **(a) − base** = *exactly* the doubled activation reads. Grid, weight traffic
  and MAC work are all identical.
- **(a) − (b)** = *exactly* one weight pass. Grid, activation traffic and MAC
  work are all identical.
- **E33 − (b)** = *exactly* the lost parallelism. Everything else is identical.

`U = 32·rows·NA + 16·rows + 48·NA` per working threadgroup per k-block, from
`values_per_thread=16` and the vector widths in `_wide`; the coefficients are
structural, not fitted.

## 3. Registered prediction — I disagree with the advisor

> **`e38/m6_per_row_cost_ratio` = 0.990, band [0.960, 1.005].**

The assignment expects ≈ **0.84**. I register against it. Two independent routes:

**Route 1 — issue-cost model fitted to the measured base ladder.**
Least squares over M=2..9 gives `T = 34.87 + 0.06441·U + 17.24·(passes−1)` ms,
residuals within ±2.1 ms. Differentially (the intercept and base's own residual
cancel): arm (b) = `1 + (0.06441·224 − 17.24)/128.843` = **0.9782**. E33's
measured excess over that same prediction is the grid-thinning penalty,
**S = +4.75 ms = +3.69%** — real, and exactly what E38 recovers. But the +224
issue units of the doubled activation tile very nearly cancel the saved weight
pass, so the *net* is near parity, not 0.84.

**Route 2 — per-shape residual analysis of E33's own measured table.**
E33 ran `ceil(N/8)` threadgroups; base and E38 both run `2·ceil(N/8)`. Shapes
whose E33 grid already saturated the machine paid no grid penalty, so for *those*
shapes E33's measured ratio **is** the pure traffic effect that E38 keeps:

| shape | `ceil(N/8)` | knee | E33 | E38 prediction | calls |
|---|---:|---|---:|---:|---:|
| mlp.gate_up_fused | 4352 | ok | 0.9941 | 0.9941 | 64 |
| mlp.down | 640 | **below** | 1.0592 | 0.9905 | 64 |
| linear_attn.in_proj_fused | 2060 | ok | 0.9947 | 0.9947 | 48 |
| linear_attn.out_proj | 640 | **below** | 1.0492 | 0.9905 | 48 |
| full_attn.qkv_proj_fused | 1792 | **below** | 1.0148 | 0.9905 | 16 |
| full_attn.o_proj | 640 | **below** | 1.0414 | 0.9905 | 16 |
| head.lm_head | 31040 | ok | 0.9830 | 0.9830 | 1 |
| head.compact_draft_vocab | 12292 | ok | 0.9868 | 0.9868 | 0 |

Blended round ratio = **0.9923**, band 0.9895..0.9939. Note this predicts
`mlp.down` **does** cross back below 1.0 (from 1.0592), which is a per-shape
falsifier independent of the headline.

**The strongest single fact against 0.84:** on `lm_head` — the most DRAM-bound
shape in the mix, with a 31040-row-group grid and therefore *no* occupancy
penalty to recover — halving the weight pass bought only **1.7%**. If doubling
the activation tile were nearly free, that shape alone would already show a large
win in E33's data. It does not.

## 4. Registered within-experiment relations

These are identities over one session's own numbers, with no cross-session model:

| id | relation | registered band | point |
|---|---|---|---|
| R1 | `ratio(a) − ratio(b)` = one full weight stream | [+0.130, +0.200] | +0.1658 |
| R2 | `ratio(a) − 1` = cost of doubling the activation tile | [+0.050, +0.200] | +0.1440 |
| R3 | `1.0150 − ratio(b)` = the parallelism E33 lost | [0.000, +0.200] | +0.0250 |

`ratio(b) = ratio(a) − R1` exactly. **The weakest link in my case is R2**: route 1
assumes an activation-issue unit costs the same as a MAC unit. Arm (a) *measures*
that assumption. If arm (a) lands materially below 1.144, the activation tile is
cheaper than I modelled, `ratio(b)` drops with it, and that is precisely the
pathway by which 0.84 could still be right — with no appeal to my model at all.
Break-even: `ratio(a) = 1.1658` puts arm (b) exactly at parity with base.

R3 is the assignment's actual question stated as a number: my account needs
**+0.025**, the 0.84 account needs **+0.175**. A 7× disagreement about one
quantity, settled by one measurement.

## 5. Controls

- M ∈ {2,3,4,5,7,8,9} are untreated. Predict `|ratio − 1| ≤ 0.0046`, which is
  E33's own worst untreated-M deviation (M=9 at 1.0046).
- **M=1 is a global null**: neither dispatch tier has a `case 1:`, so M=1 never
  reaches the crossrow kernel at all. Any M=1 movement invalidates the run.
  A serial-leg speedup would *lower* the published score, so this is a
  correctness gate and not a bonus (E33: 1.0055).
- Register predictions from E33's measured `r=2` law `regs = 15 + 17·NA`,
  measured by AIR probe **before** the production compile:
  `_wide<T,6,DN,2>` = 117 (band 113–121), `_wide<T,3,DN,2>` = 66 (band 62–70).
  Gate: ≤ 125 (the shipped `_wide<T,5,..,4>` high-water), and no
  `[N x <6 x float>]` accumulator spill in the AIR listing.

## 6. Correctness

The x-block move must be **bit-identical**, not merely close. Argument:

- `block_size = 512` and the lane→k-slice map (`simd_lid*16`) depend only on
  `in_vec_size` and `simd_lid` — never on `tid.x`, `ROWS_PER_SIMD`, or block index.
- `group_index = row·in_vec_size_g + k/64 + simd_lid/4` depends only on the
  physical output row.
- The `VF` ops are component-wise; no cross-`m` mixing exists.
- `simd_sum` reduces the same 32 lanes holding the same k-slices in the same order.

Moving a row block to a different `tid.x` therefore leaves every output element's
floating-point expression tree invariant. Evidence required: `run-qmv-parity.sh`
verdict `BIT-IDENTICAL` for both arms across 192 cells, plus a cross-arm golden
with `all_tokens_matched` and element-wise equal `effectiveDraftLengths`.

Coverage is asserted structurally: `ceil(N/8) · 2 simdgroups · ROW_BLOCKS ·
ROWS_PER_SIMD = N` for every scored shape (all have `N % 8 == 0`).

## 7. E2E decision rule — registered before looking at any E2E number

`ψ = 0.228` is **measured** (QMV share of the decode wall, E33, this host), not
assumed. `φ = 0.201` is the M=6 share of QMV *cost* on my fixture; askeladd's
`≥ 0.217` is a *dispatched-row* share on ranked beagle. **These are not the same
quantity and I do not substitute one for the other.**

Predicted leg movement `= (1 − ratio) · φ · ψ = (1 − ratio) · 4.583%`:

| ratio | leg | score (advisor's `0.4827·ψ·φ·x`) |
|---|---|---|
| 0.84 (assignment) | +0.733% | +0.354% |
| 0.9345 | +0.302% | +0.146% |
| **0.99 (mine)** | **+0.046%** | **+0.022%** |
| 1.0150 (E33) | −0.069% | −0.033% |

E33's `--local-iterate` instrument resolved ±0.30% at n=2 / 64 tokens.

> **RULE: run the 512-token, ≥4-paired-leg, ABBA E2E iff `ratio ≤ 0.9345`.**
> Otherwise the instrument cannot resolve the effect and running it would only
> manufacture a noise number; report this arithmetic instead.

At my registered 0.99 the predicted movement is 0.046% — 6.5× below resolution.
I am registering *now* that I expect to skip the E2E, so that skipping it later
cannot be read as avoiding a bad result.

One consequence worth stating plainly: **even at the assignment's 0.84**, the
projected score gain is +0.354% against a **0.561%** gap to rank 1, so E38 alone
does not take the top slot under the measured ψ. ψ is the swing factor — at
ψ = 0.40 the same ratio would give +0.621%.

## 8. Ship / kill

| verdict | threshold | action |
|---|---|---|
| **decisive** | `ratio ≤ 0.9166` | projected score gain = 2σ (0.185%); one ranked run could confirm it alone |
| **ship** | `ratio ≤ 0.9583` | projected score gain = 1σ (0.092%) |
| **null** | `ratio ≥ 0.9954` | inside the control band; report a null, do not chase |

In every non-ship case the kernel is **reverted** so no slower cell can sit on the
submission path — this is why E33 was closed unmerged. Research artifacts and the
`(g)` side-fix stay regardless.

## 9. Runtime geometry — advisor comment `5337327566`

**Finding, resolved before measurement (this was registered as an open question
and is now answered):** `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` imports
only `CryptoKit, Foundation, MLX, MLXLLM, Testing`. It never constructs
`QwenRuntimeMTPWorker`, so `RuntimeStartupMemoryPolicy.resolve()/apply()` never
runs. **The primary-metric cost-curve arms therefore execute under MLX's own
architecture-derived defaults — a third geometry, distinct from both:**

| | MB/buffer | ops/buffer | cache limit | residency |
|---|---|---|---|---|
| cost curve (primary metric) | MLX default for this arch | MLX default | MLX default | off |
| local decode / E2E | 128 | 64 | 6 GiB, cleared after warmup | off |
| ranked box (≥96 GiB) | 512 | 50 | MLX default | **on** |

MLX's arch defaults are set in `backend/metal/device.cpp:574-596` (`'g'` → 40/40,
`'s'` → 50/50), overridable by env. The exact resolved values are captured in
every arm's meta at measurement time rather than assumed.

This cuts two ways and I will report both: a single-op microbenchmark is largely
*insensitive* to command-buffer batching, which makes the primary ratio more
robust than feared; but it also means the cost-curve ratio has never been
validated against the geometry the E2E leg actually runs in, let alone the ranked
one. `wireResidentWeightsIfEnabled` is ≥96 GiB-gated with no env override and is
**genuinely not testable locally** on a 48 GiB host.

Plan:
1. Record the effective geometry in **every** arm's meta (`MLX_MAX_MB_PER_BUFFER`,
   `MLX_MAX_OPS_PER_BUFFER`, resolved profile, whether the low-memory stderr
   notice appeared, whether the residency ticket was taken).
2. A ranked-geometry arm (`DARKBLOOM_STARTUP_MEMORY_PROFILE=full`,
   `MLX_MAX_MB_PER_BUFFER=512`, `MLX_MAX_OPS_PER_BUFFER=50`) runs **last**, may
   OOM on 48 GiB, and is a **third arm, never a replacement** — the two primary
   ratio arms stay in one geometry so the ratio stays paired.
3. If lock time does not allow (2), (1) alone ships and the gap is stated plainly
   as unmeasured. A terminal E38 result is not delayed for it.

## 10. Fixture honesty

The local `--local-iterate` fixture is degenerate for acceptance: 10 draft counts
summing to 54 with `accepted_draft_total = 54` ⇒ every draft accepted,
`fullAcceptStreak` never reset, `widthCap` pinned at 8. **The accept rate is
reported next to every draft histogram.** It remains the campaign's only
instrument that reaches the width regime (depth 7, M=8), so it is pinned
identically across arms and `effectiveDraftLengths` equality is asserted
element-wise.

## 11. Scope

Editable in this experiment:
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h` and its
twin `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` (offset +13),
`out_vec_size >= 4096` tier only. Not touched: host dispatch, `mtp-head.manifest.json`,
depth-policy constants. `static_assert(M % IPG != 1)` and the new row-block tiling
assert are both kept. Deliverable `(g)` — making `benchmark-qwen-mtp.sh` rebuild
`mlx.metallib` or making its stale warning fatal — lands as a **separate commit**.
