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

---

## Rung 1 — instrument defect found, and the estimator it forced

The palindrome does **not** cancel what I designed it to cancel.

Reading the GPU temperature runs `macmon pipe -s1` as a subprocess. That leaves
the GPU idle for about a second, and the ramp back to full clock costs a fixed
30 to 80 ms of wall clock. A fixed cost is not monotone drift. It is paid
entirely by whichever arm is timed first, so the forward-to-reverse mean does
not remove it: it inflates `a_one` and makes every other arm look better than
it is.

Median `100 × (forward / reverse − 1)` over all kept cells:

| arm | `a_one` | `b_msplit` | `c_nsplit` | `c_nsplit_pre` | `d_indep` | `e_nsplit_serial` | `f_nsplit4` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gap | **+61.6 %** | +0.2 % | +0.1 % | +0.1 % | +0.4 % | +0.0 % | +0.0 % |

Position inside a pass does not matter once the GPU is ramped. `f_nsplit4` is
timed last on the forward pass and first on the reverse pass and the two agree
to 0.3 %. `b_msplit` is second forward and sixth reverse and agrees to 0.2 %.
Only position 1 of the forward pass is contaminated.

The reverse pass therefore measures every arm at full clock, and
`--pass reverse` is the analysis default. Three independent checks say this is
the correct estimator, not a convenient one:

1. **The defect is arm-specific, not shape-specific.** A real property of
   `a_one` would appear in both passes.
2. **The host-cost control becomes physically sensible.** Under the reverse
   estimator every one-eval arm control lands at 94.5 to 109 us against a
   measured `eval` overhead of 100.7 us, and `e_nsplit_serial` lands at 190 us,
   which is two evals. Under the contaminated mean, `a_one` read 149 to 163 us,
   which is *higher* than the two-dispatch arms and physically impossible.
3. **It reproduces an independent prior campaign measurement.** My `a_one` net
   rate on `lm_head` is 244.1 / 238.7 / 206.6 / 175.1 GB/s at NA=2/3/4/5. The
   campaign's isolated one-group table from E111 (`senpai/campaign-ledger.md`
   `:34057-34058`, alphonse) is 242.2 / 239.7 / 206.9 / 173.5 GB/s. That is
   agreement inside 0.8 % at every width, on a probe written independently.
   The contaminated mean gives 221 GB/s at NA=2 and does not match.

This defect is a property of sampling `macmon` between timed regions. Any
future probe in this campaign that samples an external temperature source
between arms inherits it. The cheap fix for a later probe is to ramp for a
fixed **wall-clock** duration of at least 150 ms rather than for a fixed
iteration count, or to sample temperature only at block boundaries outside the
timed sequence.

---

## Rung 1 — result

`harness=local`. `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false`. Apple M4 Pro, 48 GiB,
`ip-10-231-2-95.ec2.internal`, Swift 6.3.3, release build, `git_head`
`b9acf36c`, `git_dirty=0`, base `91b51ec3`. Session 22:55:20Z to 23:07:06Z on
2026-08-21. GPU 36.45 C at entry, 37.93 C at exit. 792 cells, 6 blocks, block 0
discarded, 5 paired blocks per cell.

### The deployable arm buys nothing

`c_nsplit_pre` is the arm a rung-2 call-site change would actually ship:
half-views hoisted out of the hot path, two concurrent `quantizedMM` calls,
one weight buffer. Net percent faster than `a_one`:

| shape | NA=2 | NA=3 | NA=4 | NA=5 |
| --- | --- | --- | --- | --- |
| `mlp.gate_up` | +0.65 | +0.23 | **−1.04** | −0.33 |
| `lm_head` | −0.26 | −1.17 | **−49.69** | −17.36 |
| `gdn.in_proj` | +0.30 | +0.27 | −0.10 | −0.73 |
| `fa.qkv` | −0.18 | +0.49 | −0.19 | −0.72 |

Null on three tensors at every width, and a large loss on `lm_head` at the two
widths that carry 70 % of the local round weight. Per-block spreads are tight:
most cells span under 1.5 pp across the 5 paired blocks.

**Kill rule fails.** It required `c_nsplit` at NA=4 to be at least +3 % on both
`mlp.gate_up` and `lm_head`. Measured: `mlp.gate_up` −1.69 %, `lm_head`
−48.99 %. I stop at rung 1. No rung 2, no rung 3, no submission.

### H1 / H2 / H3, in net percentage points against `a_one`

`total` is `c_nsplit_pre`. `H3 slicing` is `e_nsplit_serial`, the same two
half-N dispatches with concurrency removed. `H1 concurrency` is
`total − H3`. `H2 weight sharing` is `total − d_indep`.

| shape | NA | total | H3 slicing | H1 concurrency | H2 sharing |
| --- | --- | --- | --- | --- | --- |
| `mlp.gate_up` | 2 | +0.65 | −3.99 | +4.64 | +0.77 |
| `mlp.gate_up` | 3 | +0.23 | −5.55 | +5.78 | −0.09 |
| `mlp.gate_up` | 4 | −1.04 | **+6.70** | −7.73 | −0.59 |
| `mlp.gate_up` | 5 | −0.33 | +2.41 | −2.74 | +0.23 |
| `lm_head` | 2 | −0.26 | −3.21 | +2.95 | −0.32 |
| `lm_head` | 3 | −1.17 | −3.53 | +2.35 | −0.11 |
| `lm_head` | 4 | −49.69 | −3.12 | **−46.57** | −0.18 |
| `lm_head` | 5 | −17.36 | −2.76 | −14.60 | −13.01 |
| `gdn.in_proj` | 2 | +0.30 | −15.81 | +16.12 | −0.01 |
| `gdn.in_proj` | 3 | +0.27 | −16.13 | +16.40 | +0.01 |
| `gdn.in_proj` | 4 | −0.10 | −15.43 | +15.33 | −0.11 |
| `gdn.in_proj` | 5 | −0.73 | −14.56 | +13.82 | +0.82 |
| `fa.qkv` | 2 | −0.18 | −18.68 | +18.50 | +0.04 |
| `fa.qkv` | 3 | +0.49 | −17.69 | +18.18 | +0.20 |
| `fa.qkv` | 4 | −0.19 | −17.14 | +16.95 | +0.42 |
| `fa.qkv` | 5 | −0.72 | −16.89 | +16.17 | +0.05 |

**H2 shared-weight caching is dead.** `d_indep` gives each concurrent dispatch
its own half-size weight buffer, which destroys every opportunity to reuse a
cached weight line. It performs the same as the shared-buffer arm: the sharing
column is inside ±0.8 pp in 15 of 16 cells. The 16th is `lm_head` NA=5, which
sits inside the thrash regime described below and is not a clean sharing
signal. The pre-registered discriminator agrees: under H2 I required the
`b_msplit` net ratio to be below 1.90, and it is 1.96.

**H1 request-level overlap is real as a mechanism and worthless as a lever.**
Concurrency is doing genuine work: on `gdn.in_proj` and `fa.qkv` it is worth
+13.8 to +18.5 pp against the same two dispatches run back to back. But the
`total` column shows the catch. The only way to create the second dispatch is
to split, and splitting costs almost exactly what overlapping it repays. On
`fa.qkv` NA=4 the split costs 39.7 us of GPU time, about 20 us for each extra
dispatch, and concurrency hides 39.7 us. Net zero, at every width, on every
tensor. The overlap prize the brief hoped to harvest is the *refund* on a fee
the split itself charges.

**H3 slicing has no independent benefit**, with one exception below. Slices do
alias, so that premise was right: `Memory.activeMemory` grows by 0 bytes on the
100 MB `mlp.gate_up` buffer and by 0 bytes on the 715 MB `lm_head` buffer, and
the three smaller shapes show negative deltas, meaning memory was released. A
copy would have added the full tensor size. But a hoisted, aliased, cost-free
half-view still leaves `total` at zero.

### Why: one dispatch already saturates the memory system

The direct group-scaling measurement explains all three verdicts at once.
`b_msplit` runs two concurrent dispatches of NA rows over the **full** N, so
both read the same weights and the weight traffic doubles. Net time ratio
against `a_one`:

| shape | `[2+2]/[2]` | `[3+3]/[3]` | `[4+4]/[4]` | `[5+5]/[5]` |
| --- | --- | --- | --- | --- |
| `mlp.gate_up` | 2.014 | 2.004 | 1.741 | 1.815 |
| `lm_head` | 1.958 | 1.962 | 2.031 | 2.167 |
| `gdn.in_proj` | 1.864 | 1.848 | 2.114 | 2.032 |
| `fa.qkv` | 1.851 | 1.803 | 1.850 | 2.122 |

Median 1.960 over 16 cells, range 1.741 to 2.167; median 1.940 at NA=4 alone.
Two concurrent readers of the *same* bytes cost two full passes. There is no
bandwidth left for a second dispatch to use, so concurrency can only hide fixed
per-dispatch costs such as launch and grid tail. That is exactly the +16 pp
seen on the small tensors and exactly why it never exceeds the split penalty.

### Secondary result: the first direct measurement of group scaling

Rung 0 established that every campaign value of `A` is inferred. `A_local`
= 1.640 is the identity `2(1 − c)` with the E100 collapse `c` = 0.180.
`A_ranked` is inferred twice, or rests on `dW = −0.070 ± 0.360 pp`. None of
them measures a `[w+w]` partition directly.

The `b_msplit` column above is a direct, isolated, per-tensor measurement of
the `[w+w]` against `[w]` ratio, which is the quantity Finding 31 models:
**1.960, range 1.741 to 2.167**.

That does not contradict `A_local` = 1.640, because `A_local` is a *round*
ratio and a round contains work that does not scale with the group count. If a
fraction `f` of the local round scales with `G`, then
`A_round = f × A_tensor + (1 − f)`, so `f = 0.640 / 0.960 = 0.667`. Two thirds
of the local round is group-scaling quantized matvec work. That is a testable
number and a useful cross-check on Finding 22's per-tensor shares, which sum to
about 60 % of the E96 frame.

Caveat, stated plainly: `b_msplit` is two separate MLX `quantizedMM` calls. The
shipped kernel forms its groups inside one dispatch. The two agree on weight
byte traffic, which is what Finding 31 models, but not necessarily on launch
overhead. Treat 1.960 as a measurement of the byte-traffic component of group
scaling, not as a drop-in replacement for `A` in a round-level equation.

### Two anomalies worth recording

**`mlp.gate_up` at NA=4 and NA=5: the serial split is genuinely faster.**
`e_nsplit_serial` is +6.70 % at NA=4, block range +5.91 to +7.41, and +2.41 %
at NA=5. It is negative at NA=2 and NA=3. `a_one` on `mlp.gate_up` shows a rate
dip at NA=4, 169.4 GB/s against 221.2 GB/s at NA=3, and splitting N in half
side-steps that dip. Concurrency then gives the whole gain back, −7.73 pp. This
is the one cell where the split has an independent benefit, and it is only
reachable if the two half dispatches do **not** overlap.

**`lm_head` concurrency cliff at NA=4.** All three concurrent arms collapse
together: `c_nsplit` −48.99 %, `c_nsplit_pre` −49.69 %, `d_indep` −49.51 %,
while serial `e_nsplit_serial` is only −3.12 %. NA=5 shows a smaller version.
Two concurrent dispatches each streaming a 358 MB half thrash the memory
system. Whatever a future experiment does with `lm_head`, it must not issue two
concurrent large-N reads of it at the scored widths.

### Four-way split

`f_nsplit4` has no matching `control.small` arm, because the control tensor's
quarter is 2048 rows and the probe skips quarters below 4096. Raw percent
faster than `a_one`, host cost included:

| shape | NA=2 | NA=3 | NA=4 | NA=5 |
| --- | --- | --- | --- | --- |
| `mlp.gate_up` | −2.09 | +0.02 | −1.80 | −1.10 |
| `lm_head` | +0.28 | +0.27 | +0.18 | +0.23 |
| `gdn.in_proj` | −2.26 | −2.13 | −1.91 | −1.62 |

No knee. Four streams are not better than two, and on `lm_head` at NA=4 the
four-way split avoids the two-way cliff entirely, which supports a
working-set explanation for that cliff rather than a dispatch-count one.

### Fidelity

Every cell passed. `nsplit_bit_exact` is true for all 20 shape and width
combinations, `positive_control_differs` is true for all 20, and
`nsplit4_bit_exact` is true for all 12 combinations where the quarter split
ran. An N-split of `quantizedMM` is bit-exact; it is simply not faster.

### Scoring the pre-registration

| pre-registered claim | outcome |
| --- | --- |
| H1: `c_nsplit_pre` ≥ +3 % at NA=4 | **falsified**, −1.04 % on `mlp.gate_up` |
| H2: `b_msplit` ratio below 1.90 | **falsified**, 1.960 |
| H3: `e_nsplit_serial` ≈ `c_nsplit_pre` ± 1.5 pp | **falsified**, −15 to −19 pp apart on the small tensors |
| H1: `e_nsplit_serial` ≈ `a_one` ± 1.5 pp | **falsified**, and not in the predicted direction |
| slices alias, `c_nsplit` within 1 pp of `c_nsplit_pre` | **confirmed**, 0 byte growth, arms within 0.7 pp |
| `d_indep` within 1.5 pp of `c_nsplit_pre` | **confirmed**, 15 of 16 cells |

I did not pre-register the outcome that actually occurred: a large split
penalty that concurrency repays almost exactly. My honest prior, recorded
before the run, was "H2 or a null, with a real but small concurrency benefit
relative to forced serialisation". The direction was right and the magnitude
was wrong. The concurrency benefit is large, +16 pp, not small; it is the split
penalty that cancels it.

### Suggested follow-ups, not implemented

1. **Barrier-separated N-split on `mlp.gate_up` at NA=4 and NA=5.** The only
   positive cell in this experiment. `e_nsplit_serial` bought +6.70 % of GPU
   time at NA=4, round weighted +2.93 % across the four widths, but it needs
   the two half dispatches serialised. MLX encodes concurrently by default
   (Finding 17), so this would need `maybeInsertBarrier` between them rather
   than a host round trip. My `e_nsplit_serial` pays a full blocking `eval`,
   and the net estimator removes that host cost, so +6.70 % is the idealised
   GPU-side ceiling for a free barrier. Worth one cheap probe before any
   call-site work.
2. **Explain the `mlp.gate_up` NA=4 rate dip.** 169.4 GB/s against 221.2 GB/s
   at NA=3 is a 23 % loss at the dominant scored width on a tensor that carries
   roughly 38 % of the round. Follow-up 1 is one workaround, but the dip itself
   may have a cheaper fix inside the partition heuristic.
3. **Verify `f` = 0.667 independently.** If two thirds of the local round is
   group-scaling matvec work, that predicts the round-level effect of any
   change to the group partition and should be checked against Finding 22.
4. **Fix the ramp defect campaign-wide.** Ramp for a fixed wall-clock duration,
   or move temperature sampling outside the timed sequence, in any probe that
   reads `macmon` between arms.

