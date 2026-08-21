# E115 — concurrency discriminator and the N-split

Experiment: `qwen38-r1-e115-concurrency-discriminator-and-the-n-split`
Base: `91b51ec3c5c3eb86b917de1efb3de7219dc3eecb` (`senpai/qwen38-mtp-r1`)
Host: Apple M4 Pro, 48 GiB, `ip-10-231-2-95.ec2.internal`. `harness=local`.
Reproduce rung 0: `python3 research/e115_rung0.py`

**Question.** Two concurrent QMV dispatches beat one. Is the cause request-level
overlap (H1), shared-weight caching (H2) or slicing (H3)? If H1 holds, an
N-split buys the overlap prize with no kernel edit.

---

## Rung 0 — the three `A` values and where they come from

### Provenance table

| value | published by | exact input measurement | byte convention | partition map | status |
| --- | --- | --- | --- | --- | --- |
| `A_local = 1.640` | `research/group_scaling.py:14-18` | E100 collapse `c = 0.180` (alphonse, PR #102): end-to-end 3-session ABBA, `research/e100-results.md:265-281`, reconciled with an isolated kernel probe `-17.7 ± 1.7 %`, `:283-294` | logical weight bytes `W = 14.41235e9`; **W cancels** | **pre-E100**, `Gof[5] = 2`, M=5 is `[3+2]` | **derived identity.** `A ≡ 2(1 − c)`; the script prints "matches by construction" |
| `A_ranked` route 1 `= 2.040` | `research/group_scaling.py:28-33` | `A_local` × ranked-to-local `[3] → [3+2]` aggregate scaling. The ranked round times `{1:31177 … 5:53108}` are **evaluations of the fitted lines** `27,181.5 + 3,995.1M` and `16,943.2 + 7,233.0M`, `senpai/campaign-ledger.md:28560-28572` | logical | **pre-E100** | **inferred twice.** Inherits every `A_local` input, then multiplies by a ratio of two fitted lines |
| `A_ranked` route 2 `= 1.994` `[1.964, 2.024]` | `research/group_scaling.py:36-45` | `dW = -0.070 ± 0.360 pp` from receipts `ca9251b8` / `3ff80e86`, `senpai/campaign-ledger.md:31903-31912`; `share = 0.24` | published-percent difference | ranked tree of those receipts (**pre-E100**) | **derived from one receipt difference** whose standard deviation is **5.1×** its point estimate; `share` is assumed, not measured |

The local round times `{1:64445 … 5:126103}` are measured: E92 round-busy,
edward, PR #94, W&B `ytcemy51`, ungated ABBA, 512 tokens,
`senpai/campaign-ledger.md:27529-27533`.

### What that table means

1. **`A_local` is an identity, not a concurrency measurement.**
   `r2 = 2W/t2`, `r1 = W/t1`, `t1 = t2(1 − c)`, so `A = 2t1/t2 = 2(1 − c)`.
   `W`, the byte convention and the rate units all cancel. The coefficient
   carries exactly one measured input, the E100 collapse fraction.
2. **`A_ranked` route 1 is `A_local` in ranked clothing.** It cannot corroborate
   `A_local` because it is a function of it, and its ranked inputs are fitted,
   not measured.
3. **Route 2 is the only ranked evidence, and it is a round-level receipt
   difference.** It excludes `A = 1.64` on the ranked box at ~11.8 σ given
   `share = 0.24`, but it says nothing about the mechanism.
4. **No route measures concurrency.** Every `A` mixes four terms: overlap
   between concurrent dispatches, the per-dispatch `rate(NA)` curve, total
   logical bytes (2 passes against 1) and the extra dispatch boundary. A `[3+2]`
   partition runs its groups at NA=3 and NA=2, which are intrinsically faster
   per byte than one NA=5 group, and that difference sits inside `A` with no way
   to remove it. **E115 rung 1 fixes NA and total bytes and measures overlap
   alone.**
5. **Frame dilution.** `A_local` is a ratio of round times, so with `t = F + q`
   and any non-QMV `F > 0`, `A_round = 2(F+q1)/(F+q2) > A_kernel = 2q1/q2`. At
   `F = 20 %` of the round, `A_kernel = 1.550`; at `F = 40 %`, `A_kernel = 1.400`.
   *Open point:* E100's isolated kernel collapse `-17.7 ± 1.7 %` equals the
   round-level `-18.0 %` almost exactly, which is only possible if QMV holds
   nearly the whole M=5 round. No `A` quotation should treat the round frame and
   the kernel frame as interchangeable.

### Partition map, as the brief asked

`Gof[5] = 2` in `research/group_scaling.py:8` is the **pre-E100** map. The
current tree (`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:1917-1978`)
partitions `M = 1..9` as `[1] [2] [3] [4] [5] [3+3] [4+3] [4+4] [3+3+3]`, so
**M=5 is one group, G=1**. Every number quoted from Finding 31 or Finding 32 is
therefore stated on a partition map this tree no longer has. Two comments inside
`quantized.h` are also stale (`:1953-1958` says "3+3+2" above a `4+4`
instantiation; `:1154-1155` gives a pre-E100 IPG formula) — cite the code, not
the comments.

### Three corrections to the brief's inputs

1. **There is no Finding 47.** The highest finding in `senpai/campaign-ledger.md`
   is Finding 45 at `:34310`. The rate table the brief calls "Finding 47 / E106
   per-width refit" (`253.6 / 245.6 / 211.7 / 178.8 GB/s`) is edward's
   **end-to-end refit**, `:33867-33878`, W&B `19kgn6xi`. A **different** table,
   the **isolated probe** from E111 rung 1 (`242.2 / 239.7 / 206.9 / 173.5 GB/s`,
   `:34057-34058`), also exists. The two must not be merged. I use the isolated
   family when I need an isolated rate and label every use.
2. **Finding 42 is superseded by Finding 44** (`:33837-33866` against
   `:34260-34281`). I price the H1 ceiling from Finding 44 only.
3. **`Qwen35.swift:2995-3000` is not an lm_head call site.** It is a draft-rerank
   Metal source string. The affine-4 `248,320 × 5,120` lm_head sites are
   `:4404-4405` and `:4540-4541`, and the 2-bit `98,336 × 5,120` coarse readout
   at `:4835-4837` is a different tensor. This matters only if rung 2 runs.

I could not trace "Finding 46" or harness defects 7 and 14 to any file in
`senpai/` or `research/`; only harness defect 12 is defined there. I treat them
as advisor-held context and will follow them at rung 2 regardless.

### Cost of the split, in both frames (Rule 34)

`258` extra dispatches × `1.8 us` = `464 us` per round =
**0.364 %** of the E96 anchor decode round (`127,533 us`) =
**0.451 %** of the current-tree decode-only M=5 round (`102,864 us`).

| shape | N | half | quarter | packed MB | us @200 GB/s | one extra boundary |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp.gate_up` | 34,816 | 17,408 | 8,704 | 100.3 | 501.4 | 0.36 % |
| `lm_head` | 248,320 | 124,160 | 62,080 | 715.2 | 3,575.8 | 0.05 % |
| `gdn.in_proj` | 16,480 | 8,240 | 4,120 | 47.5 | 237.3 | 0.76 % |
| `fa.qkv` | 14,336 | 7,168 | **3,584 (narrow)** | 41.3 | 206.4 | 0.87 % |

`fa.qkv` quarters fall below the `out_vec_size >= 4096` branch, so the probe
runs no `f_nsplit4` arm there and records why.

### H1 ceiling, from Finding 44 (`harness=local`)

| NA | shipped `a_base` | load-only | ALU-only | perfect overlap | ceiling for `c_nsplit` |
| --- | --- | --- | --- | --- | --- |
| 4 | 245.9 us | 203.0 us | 33.0 us | 203.0 us | **+17.4 %** |
| 5 | 292.2 us | 207.0 us | 39.1 us | 207.0 us | **+29.2 %** |

---

## Rung 0 — pre-registered predictions

Metric: `pct_faster(arm) = 100 × (1 − net_us(arm) / net_us(a_one))`, per block,
median over blocks, per shape and per NA. `net_us` subtracts the `control.small`
cell of the **same arm structure and width**, which is that structure's host
cost plus about 1.5 us of GPU work.

| observable | H1 request-level overlap | H2 shared-weight caching | H3 slicing |
| --- | --- | --- | --- |
| `c_nsplit_pre` at NA=4 | **≥ +3 %**, typically +5..15 % (ceiling +17.4 %) | **−1.5 %..+1.0 %**, slightly negative from the extra boundary | **+1..8 %** |
| `c_nsplit_pre` against NA | rises with NA: ~+0..3 (NA=2), +3..8 (NA=3), +5..15 (NA=4), +8..25 (NA=5) | flat and ~0 at every NA | roughly flat, or set by grid tail effects rather than by the roofline gap |
| `d_indep` | within 1.5 pp of `c_nsplit_pre` | ≈ 0, same as `c_nsplit_pre` | within 1.5 pp of `c_nsplit_pre` |
| `e_nsplit_serial` | ≈ `a_one` ± 1.5 pp: removing concurrency removes the gain | ≈ `a_one` | ≈ `c_nsplit_pre` ± 1.5 pp: the gain survives without concurrency |
| `b_msplit` net ratio to `a_one` | 1.65..1.85 | < 1.90 (the shared-weight signature) | ≈ 2.0 |
| `f_nsplit4` | ≥ `c_nsplit_pre` | ≈ `a_one` minus three boundaries | ≥ `c_nsplit_pre` |

Two further pre-registered statements:

- **Slice aliasing.** If a first-dim slice aliases the parent buffer,
  `c_nsplit` (slicing inside the timed body) is within 1 pp of `c_nsplit_pre`
  (slices hoisted) and `Memory.activeMemory` does not grow by the tensor size.
  If MLX materialises a copy, `c_nsplit` is at least 50 % slower and the arm is
  dead at rung 1.
- **Kill rule.** `c_nsplit_pre` must be at least **+3 %** at NA=4 on **both**
  `mlp.gate_up` and `lm_head`. Below that I stop at rung 1 and report.

### Disclosure, so the pre-registration is honest

I ran one debug-configuration smoke block on `fa.qkv` at NA=4 while validating
the harness (`research/out/e115-smoke/`, one block, block 0 not discarded, debug
build, `eval` overhead 124 us). It showed `c_nsplit_pre` about **4 % slower**
than `a_one` net of overhead, `e_nsplit_serial` about 7 % slower than
`c_nsplit_pre`, and `b_msplit` about 4 % faster than two serialised passes.
That is one unblocked cell on the smallest qualifying tensor in the wrong build
configuration, so it decides nothing, but it is what I saw before writing this
table. My prior is therefore **H2 or a null**, with a real but small concurrency
benefit relative to forced serialisation. The hypothesis-conditional rows above
are unchanged by it.

---

## Rung 1 — instrument

`Tests/MLXFastTests/E115ConcurrentDispatchProbeTests.swift`, driven by
`research/e115_probe.sh` and analysed by `research/e115_analysis.py`.

The brief asked for the probe under `research/`. I put the MLX-level probe in
`Tests/` and the runner and the analysis under `research/`, because an MLX-level
probe needs the package build graph and `research/` has no Swift target.
`Tests/` is inside this round's allowed scope, is never packaged into a
submission, and is the established pattern for this kind of probe
(`E95QmvWidthProbeTests`, `E100StreamCollapseProbeTests`). A hand-written Metal
replica would measure a kernel I wrote; this measures the shipped
`quantizedMM` dispatch path that a rung-2 call-site change would actually use.

Arms, at fixed NA and fixed total logical weight bytes except `b_msplit`:

| arm | dispatches | weight passes | what it removes or adds |
| --- | --- | --- | --- |
| `a_one` | 1 | 1 | the reference |
| `b_msplit` | 2 concurrent | 2 | the shipped G=2 shape, shared weights |
| `c_nsplit` | 2 concurrent | 1 | the payoff arm, slices built inside the timed body |
| `c_nsplit_pre` | 2 concurrent | 1 | the payoff arm with slices hoisted |
| `d_indep` | 2 concurrent | 1 | **weight sharing removed**: two separate buffers |
| `e_nsplit_serial` | 2 in two evals | 1 | **concurrency removed** |
| `f_nsplit4` | 4 concurrent | 1 | is two streams enough? |
| `control.small` | same structures | — | host cost of each arm structure |

Protocol: palindromic forward-then-reverse arm order inside every block, first
block discarded, per-block paired ratio against `a_one`, median over blocks,
entry and exit GPU temperature per block from `macmon`. No thermal gate:
`cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false` are
recorded in `cells.json` and in `meta.txt`. Exactness is checked in every cell:
the concatenated split must be bit-identical to one dispatch, and a deliberately
wrong split (halves concatenated in swapped order) must change the digest.
