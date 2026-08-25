# QMV geometry byte census

**Purpose.** Any change to the QMV launch geometry (`rows_per_simd`, the
simdgroups per threadgroup, the threadgroup y extent) changes how many times the
kernel re-reads the activation slab and the chunk-sum table. Run this census
FIRST, before pricing a geometry route. E221 was proposed on the premise that
`rows_per_simd` 4 -> 2 leaves device bytes unchanged; that premise is false, and
the census refutes it at the desk for zero GPU seconds (FINDING 576).

**Scope.** This is a research note. It changes no scored file.

**Source.** `qwen35E120QMVHeader` in
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`, lines 1426-1524.
Line cites below are against that range.

## 1. What one lane reads, per k-block

`block_size = values_per_thread * 32 = 512` (line 1445). One simdgroup owns
`rows_per_simd` output rows (line 1443) and walks the whole `k` extent (line
1455). Per lane, per k-block:

| Read | Lines | Bytes per lane per k-block | Scales with |
| --- | --- | --- | --- |
| packed 4-bit weights, `4 x uint16` per row | 1461-1468 | `8 * rows` | `rows` |
| scale and bias, `2 x bfloat16` per row | 1469-1472 | `4 * rows` | `rows` |
| chunk sums, `NA x float` (`USE_TABLE` only) | 1476-1483 | `4 * NA` | **`NA` only** |
| activations, `4 x vec<bfloat16,4>` per column | 1494-1502 | `32 * NA` | **`NA` only** |

Total per lane per k-block:

```text
bytes_per_lane_per_kblock(NA, rows) = 12 * rows + 36 * NA
```

The first term is per-row work. The second term is **per-simdgroup work that
does not shrink when the simdgroup owns fewer rows.**

## 2. The invariant that matters

Normalise to one output row:

```text
bytes_per_output_row_per_kblock(NA, rows) = 12 + 36 * NA / rows
```

Halving `rows` therefore does NOT conserve issued bytes. The weight and
scale/bias term is geometry independent; the activation and chunk-sum term is
multiplied by `4 / rows`, because covering `n` output rows needs `n / rows`
simdgroups (`E219Pipeline.call` grid y extent) and every simdgroup re-reads the
same activation slab and the same chunk-sum table in full.

| `NA` | rows=4 (shipped) | rows=3 | rows=2 | rows=1 | rows2 / rows4 |
| --- | --- | --- | --- | --- | --- |
| 2 | 30 | 36 | 48 | 84 | 1.600 |
| 3 | 39 | 48 | 66 | 120 | 1.692 |
| 4 | 48 | 60 | 84 | 156 | 1.750 |
| 5 | 57 | 72 | 102 | 192 | **1.789** |
| 6 | 66 | 84 | 120 | 228 | 1.818 |
| 7 | 75 | 96 | 138 | 264 | 1.840 |
| 8 | 84 | 108 | 156 | 300 | 1.857 |
| 9 | 93 | 120 | 174 | 336 | 1.871 |

Two consequences:

1. The penalty **grows with `NA`**, so a geometry route aimed at the NA slope
   pays most exactly where the slope is steepest.
2. The excess over rows=4 is linear in `1/rows`: at fixed `NA`, the rows=2
   excess is exactly one third of the rows=1 excess. That lets a measured
   rows=1 arm price a rows=2 arm without running it (see section 4).

## 3. Whole-dispatch and whole-round totals

Total issued read bytes for one dispatch over `n` output rows at extent `k`:

```text
issued(k, n, NA, rows) = (n / rows) * (k / 512) * 32 * (12 * rows + 36 * NA)
                       = (k / 512) * 32 * (12 * n + 36 * NA * n / rows)
```

At the three top scored cells, `G = 1`:

| `NA` | cell | invocations/round | rows=4 | rows=2 | delta |
| --- | --- | --- | --- | --- | --- |
| 4 | mlp.gate_up | 64 | 31.9 GiB | 55.8 GiB | +23.9 GiB |
| 4 | mlp.down | 64 | 15.9 GiB | 27.9 GiB | +12.0 GiB |
| 4 | gdn.in_proj | 48 | 11.3 GiB | 19.8 GiB | +8.5 GiB |
| 4 | **all three** | | **59.1 GiB** | **103.5 GiB** | **+44.3 GiB (x1.750)** |
| 5 | mlp.gate_up | 64 | 37.9 GiB | 67.7 GiB | +29.9 GiB |
| 5 | mlp.down | 64 | 18.9 GiB | 33.9 GiB | +14.9 GiB |
| 5 | gdn.in_proj | 48 | 13.4 GiB | 24.0 GiB | +10.6 GiB |
| 5 | **all three** | | **70.2 GiB** | **125.6 GiB** | **+55.4 GiB (x1.789)** |

**These are ISSUED bytes, not DRAM bytes.** The weight and scale/bias share is
genuine DRAM traffic; the activation slab is `m * k * 2` bytes (at most 174 KB
per cell here) and the chunk-sum table is smaller still, so the re-read share
is L1/L2 and load-issue cost, not DRAM cost. That distinction sets which
measured coefficient prices it: the re-read is priced against the E219
per-lane-issue and cache terms, not against the fitted DRAM-stream coefficient
`b`.

## 4. Pricing the E221 route at the desk

Two independent estimates of `rows_per_simd` 4 -> 2 at `NA = 5`, both against
the E221 relief ceiling of +23.066 ms/round pooled:

| Estimate | Source | Method | Predicted effect |
| --- | --- | --- | --- |
| FINDING 573 `c` term | E219 composition fit | the `c` (per-lane activation/sums issue) term is 46.81 ms/round at NA=5, 55.2% of the pass; the census doubles it | **≈ -46.8 ms/round** |
| FINDING 557 fixed-`G` probe | E213 r1, measured | `(m=9, NA=5, rows=1)` vs `(m=9, NA=5, rows=4)` = **-118.856 ms/round** [-119.124, -118.587]; rows=2 excess is exactly 1/3 of the rows=1 excess (section 2) | **≈ -39.6 ms/round at m=9, ≈ -37.6 pooled** |

Both estimates are net-negative by more than the whole relief ceiling, and both
are far outside the ±0.24 ms/round noise floor. Corroboration from independent
work: E76 measured the `rps2` variant at **+14.16% per verify round** at NA=5 on
gated GPU time, and ranked receipts `afb688fe`, `e617ef07` and `60a5ac1f`
shipped rows=2 and lost +1.79, +1.33 and +2.38 pp respectively.

## 5. How to use this census

Before pricing any QMV geometry route:

1. Recount the four reads in the table in section 1 against the CURRENT header
   text and line numbers. Do not trust these line cites across a base move.
2. Compute `bytes_per_output_row_per_kblock` for the incumbent geometry and for
   the proposal. If it moves, the route is NOT byte-neutral and the assignment's
   cost model must include the delta before any GPU time is spent.
3. Split the delta into DRAM share (weights, scales, biases) and issue/cache
   share (activations, chunk sums). Price each against the E219 coefficient that
   actually covers it.
4. Only then decide whether the register or occupancy relief can repay it.

Related closed axes and findings: FINDING 549 (`rows_per_simd` retune at
rows=4), FINDING 552 (`(NA=7, rows=2, plain)` miscompiles), FINDING 558
(single-pass at `G=1` via rows), FINDING 576 (this census).
`research/e221_register_probe.py` supplies the register-side witness for the
same geometry grid, on the AIR proxy and on the real `applegpu_g16s` and
`applegpu_g17s` backends.
