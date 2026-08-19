SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mean_speedup_over_replaced_widths_pct","available":true,"value":-7.341},"test_metric":{"name":"exact_coverage_bad_elements","available":true,"value":0}}

# E44 — simdgroup-matrix QMV: Gate 0 passes, the timing hypothesis is refuted

- **Student / branch:** `qwen-alphonse` / `qwen-alphonse/simdgroup-qmv-register-gate`
- **Hypothesis and target cost:** replace the per-width scalar cross-row `qmv_fast`
  cells at `M ∈ {4..9}` with one `simdgroup_matrix` cell, halving weight streams
  (base `2×W` at M=5–8, `3×W` at M=9 → candidate `1×W`, `2×W` at M=9) and
  lowering the single shared register allocation as a side effect.
- **Decision: dead as dispatched, with a bankable narrower variant identified.**
  Gate 0 passed decisively; the §7.3 timing arm refuted the mechanism I
  pre-registered.
- **`BASE_SHA` / `UPSTREAM_SHA` / candidate commit:** `efff400c1b5554be2e8993b01856653d55de7664`
  / unchanged from base / see submitted commit.
- **Yukon promoted submission / source ref used as frontier:** crown tree
  `0c90733d`, score `3.24929398547457`. Not submitted officially; this candidate
  is not bankable, so no official submission was made.
- **Submitted candidate files:** `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
  and its runtime-effective twin `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`.
- **Supporting test, tooling, or documentation files:** all under `research/`
  (unsubmitted): `e44_qmv_ab.m`, `run-e44-qmv-ab.sh`, `e44_ab_summary.py`,
  `e44_flatcost_check.py`, `gpu_busy_check.py`, `validate_gpu_busy_gate.sh`,
  `jit_string_compile.py`, `air_kernel_stats.py`, `e44-prereg.md`.
- **MTP head provenance and draft policy:** unchanged; organizer-pinned head, no
  `mtp-head.manifest.json` declared. This experiment touches only the QMV kernel.
- **Assignment-scope preflight:** `senpai/validate-assignment-scope.sh` → OK, 2
  submitted paths against `BASE_SHA=efff400c`.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  source `2,466,538/3,000,000`, headroom `533,462`, growth `7,589/262,144`,
  exempt `2,410`, files 154.
- **Scored-path reachability evidence:** `affine_qmv_fast` at `quantized.h:1869`
  is the only `[[kernel]]`; the width `switch (ntg.x)` is on a runtime value and
  all helpers are `METAL_FUNC` inline, so one shared register allocation is taken
  as the max over all instantiated cells. `qmv_fast` is selected only when
  `N % 8 == 0 && K % 512 == 0`; both measured shapes satisfy that.

## Evidence

- **Host, memory profile, toolchain, and thermal policy:** Apple M4 Pro,
  `applegpu_g16s`, 48 GiB, Apple metal 32023.883. Local ungated counterbalanced
  arm: `cool_gate_real_outcome=stalled_above_40C`,
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`
  (preserved verbatim). Entry GPU 42.75 °C, exit 66.83 °C. ABBA alternation is
  inside the harness process at ~100 ms granularity, so monotone drift cancels to
  first order; the `M ∈ {1,2,3}` guard measures the residual directly.
- **Exact baseline and candidate commands:**

  ```bash
  research/run-e44-qmv-ab.sh s73 efff400c1b5554be2e8993b01856653d55de7664 \
    --widths 1,2,3,4,5,6,7,8,9 --pairs 5 --reps 50 --inner 20
  python3 research/e44_ab_summary.py .mlxfast-private/e44-qmv-ab/s73 --wandb
  python3 research/e44_flatcost_check.py .mlxfast-private/e44-qmv-ab/s73
  ```

  Both arms are the **runtime-effective JIT string**, compiled in one process via
  `newLibraryWithSource:` with `setFastMathEnabled(false)` and
  `MTLLanguageVersion4_0`, matching MLX `device.cpp:631-632`. Base arm
  sha256 `0c461cb840fe7431` @ `efff400c`; candidate `f9823dab4af00849`; 6 source
  diff hunks.
- **Tests and risk-based checks:** `research/twin_audit.py` → TWIN AUDIT OK, 29
  runtime-effective twins. AOT `tools/build-mlx-metallib.sh` clean and the JIT
  string (211,288 B) reassembled and compiled with `-std=metal4.0
  -fno-fast-math` and no `-I`, so both compile paths carry the construct.
- **Exact-token and row-ledger verdict:** not applicable — this is a kernel
  microbenchmark, not a decode run. Exactness was instead established by two
  **bit-exact** coverage passes (below). No end-to-end decode, golden, or
  row-ledger run was performed, per the assignment's instruction to stop before
  exactness/E2E work if the bar fails.
- **Divergent tokens or failure category, if any:** none. `bad = 0`,
  `worst_abs = 0.0` on 778,567,680 elements per arm across both passes.
- **Generated-twin audit:** OK, and the twin was regenerated in lockstep with the
  header (the metallib is built from the twin, so a disagreement would mean the
  source read is not the source that ran).
- **Peak RAM or head/artifact size:** microbenchmark holds ~60 MB; no head
  artifact declared.
- **Official status and score, if submitted:** not submitted.

### Gate 0 (compile-only) — PASS on all five pre-registered conditions

Pre-registered in `9b7706f` **before** compiling the candidate.

| # | Condition | Bound | Measured | |
|---|---|---|---|---|
| a | lane-corrected kernel-wide max | ≤ 108 | **89** | PASS, −17.6 % |
| b | new alloca type in entry (fails alone) | none | unchanged; new cell has 0 allocas | PASS |
| c | production entry | ≤ 163 | **143** (allocas 55→47) | PASS, −12.3 % |
| d | static threadgroup bytes | no increase | 0 → 0 | PASS |
| e | naive/lane split fully explained | required | yes, decomposition asserted | PASS |

`simdgroup_matrix<float,8,8>` is **distributed** across the simdgroup's 32 lanes
(AIR models it as one `<64 x float>` value), so the per-lane footprint is 2
registers, not 64. Lane-weighting it like a `<4 x float>` over-reports by exactly
32×; hence the opt-in `--simdgroup-distributed` correction. Cell naive 344 =
5×64 + 24 ordinary, lane 34 = 5×2 + 24 — same live set.

### Exactness — bit-exact, no tolerance anywhere

| pass | configuration | exact answer | what it proves |
|---|---|---|---|
| scale | scales 1, biases 0, `x=1` on the 8 `k` of one weight word | that word's integer nibble sum ≤ 120 | nibble order, `k` mapping, group coverage, m/n tiling, grid remap |
| bias | scales 0, `bias[n][g] = 1 + ((n+g) mod 15)` | `bias[n][word/8] × 8` ≤ 120 | which group's bias pairs with which x-sum per row; holds the `simd_shuffle_xor` reduction (masks `2u,4u,16u`) to per-nibble resolution |

Both answers are integers under 121, which bf16 stores without loss, so these are
**bit-equality** checks. 20/20 lines `bad=0 worst_abs=0.0` at stride 1, widths
`1,4,5,8,9`, both shapes. **389,283,840 elements per arm per pass.** This also
proves the base is exactly correct under my harness dispatch, which is the
load-bearing point that makes it a valid timing reference.

### §7.3 paired timing — the guard first, as pre-registered

Zero-effect guard (`M ∈ {1,2,3}` run byte-identical code in both arms, so the
true effect is exactly 0):

```
empirical noise floor from the guard: sd = 0.283 % over 6 zero-effect
measurements, worst |effect| = 0.628 %
```

The `--reps 50` raise worked: the smoke config's guard read sd = 18.368 %, and
this session reads **0.283 %**, essentially equal to the pre-registered implied
pairwise sd of 0.3032 %. **But the worst guard |effect| is 0.628 %, above the
pre-registered MDE of 0.5040 %, and two guard widths have 95 % intervals that
exclude zero on a true effect of exactly zero.** So the honest resolution floor
for this session is ~0.63 %, not 0.50 %, and the per-width intervals below
understate uncertainty at that scale. Every claim I make from this session is
either far above 0.63 % or explicitly labelled unresolved.

| shape | M | base µs | cand µs | speedup % | 95 % CI | verdict |
|---|---:|---:|---:|---:|---|---|
| attn_out | 1 | 46.92 | 46.69 | +0.485 | [−0.374, +1.344] | null (guard) |
| attn_out | 2 | 63.39 | 63.00 | +0.628 | [+0.270, +0.986] | guard, nonzero |
| attn_out | 3 | 86.27 | 86.32 | −0.060 | [−0.221, +0.101] | null (guard) |
| attn_out | 4 | 96.63 | 136.95 | **−41.718** | [−41.988, −41.447] | slower |
| attn_out | 5 | 136.34 | 141.15 | −3.530 | [−3.851, −3.210] | slower |
| attn_out | 6 | 144.69 | 144.71 | −0.011 | [−0.387, +0.365] | null |
| attn_out | 7 | 154.13 | 138.01 | **+10.456** | [+10.262, +10.649] | faster |
| attn_out | 8 | 165.55 | 137.98 | **+16.653** | [+16.429, +16.878] | faster, clears bar |
| attn_out | 9 | 203.60 | 224.72 | **−10.374** | [−10.906, −9.843] | slower |
| mlp_down | 1 | 207.70 | 207.83 | −0.065 | [−0.349, +0.219] | null (guard) |
| mlp_down | 2 | 239.35 | 238.88 | +0.196 | [+0.074, +0.319] | guard, nonzero |
| mlp_down | 3 | 258.12 | 257.71 | +0.157 | [−0.351, +0.666] | null (guard) |
| mlp_down | 4 | 293.50 | 447.24 | **−52.385** | [−53.299, −51.471] | slower |
| mlp_down | 5 | 420.75 | 456.21 | −8.428 | [−8.651, −8.206] | slower |
| mlp_down | 6 | 450.61 | 471.37 | −4.606 | [−4.851, −4.362] | slower |
| mlp_down | 7 | 485.96 | 464.29 | **+4.460** | [+4.230, +4.691] | faster |
| mlp_down | 8 | 524.24 | 455.85 | **+13.045** | [+12.914, +13.177] | faster, clears bar |
| mlp_down | 9 | 647.41 | 722.87 | **−11.655** | [−11.720, −11.590] | slower |

```
mean speedup over replaced widths M in [4, 9]        : -7.341 %
replaced widths that regress with a resolved interval: 7/12
5.0 % bar: BEST-WIDTH ONLY -> M=8 clears the bar but the replaced widths are
           -7.341 % on net with 7 resolved regressions. Not bankable as
           dispatched; the winning widths must be isolated first
```

The advisor's bar was "≥ 5 % faster at `M ∈ {5..9}` on `mlp.down`". On `mlp_down`
only **M=8** clears 5 %; M=7 reaches 4.46 %; **M=5, 6 and 9 are slower**. Read as
the conjunction it was written as, the bar **fails**.

### Pre-registered pattern prediction: 3 of 4 missed

I committed publicly, before measuring: *"If the aggregate moves but that pattern
is absent, my weight-streaming mechanism is wrong and I will say so rather than
bank the number."* The pattern is absent.

| # | Prediction | Outcome |
|---|---|---|
| 1 | win concentrated at M=5–8, largest there | **MISS** — win only at M=7, 8; M=5 and 6 are losses |
| 2 | larger on `mlp_down` (DRAM-bound) than `attn_out` | **MISS, wrong direction** — attn_out larger at both winning widths (16.65 vs 13.05, 10.46 vs 4.46) |
| 3 | near zero at M=4 | **MISS, catastrophically** — −41.7 % / −52.4 %, the largest effects in the session |
| 4 | `M ∈ {1,2,3}` is a guard, not a confirmation | **HELD** |

**So the weight-stream-halving mechanism is wrong.** Not "partially supported" —
its two most specific predictions (the M=4 null and the mlp_down ordering) both
failed, and they failed by 40–50 % and by sign respectively.

### The mechanism that actually explains the data

The candidate always evaluates a full 8-row MMA tile, so its cost is
**independent of M** for M ≤ 8, and a second tile appears at M=9:

| shape | cand plateau M=4..8 | plateau CV | base rise M4→M8 | cand M9 / plateau |
|---|---:|---:|---:|---:|
| attn_out | 139.76 µs | **2.28 %** | **+71.3 %** | 1.608× |
| mlp_down | 458.99 µs | **2.00 %** | **+78.6 %** | 1.575× |

The candidate is flat to ~2 % while the base rises 71–79 % across the same
widths. The sign of the effect is therefore set purely by **where the rising base
curve crosses the flat candidate cost — bracketed in `M ∈ [6, 7]` on both
shapes.** M=9's 1.6× step is the second tile.

This also explains prediction 2's inversion: the effect is a *ratio* against a
rising base, so it has nothing to do with DRAM traffic, and `attn_out`'s base
rises to a larger multiple of the candidate's flat cost than `mlp_down`'s does.

I disclosed the cost that drives this ("MMA efficiency is M/8, and only `1/(2M)`
of launched threadgroups do work") and then predicted a near-zero M=4 anyway. The
disclosure was right; the prediction was wrong. Two of my own numbers should have
warned me: at M=4 the MMA does 8 rows of work for 4 useful rows, and only 1/8 of
launched threadgroups are active.

## Conclusion

- **What happened and why:** Gate 0 passed exactly as pre-registered — the
  construct compiles through both paths, adds no threadgroup memory, and lowers
  the shared allocation from 108 to 89. Correctness is bit-exact on 778 M
  elements per arm. But the timed mechanism is refuted: the MMA cell's fixed
  8-row cost makes it 41–52 % slower at M=4 and 10–12 % slower at M=9, and it is
  **−7.34 % on net** across the widths it replaced. A single winning width
  (M=8, +16.65 %) does not redeem that, which is precisely the E27 lesson the
  advisor warned about — I have made the summary tool refuse to report CLEARED in
  this configuration so the artifact cannot overstate it.
- **Evidence for or against the mechanism:** against, decisively, with tight
  intervals (7/12 replaced widths regress with resolved intervals) and an
  identified replacement mechanism confirmed on both shapes (plateau CV ~2 % vs
  base rise 71–79 %).
- 🔴 **Correction the advisor needs immediately, because it gates E46.** I
  reported `108 → 89` to unblock thorfinn's PR 51. That number is correct **only
  for the all-widths variant, which this result shows is not bankable.** The
  bankable variant restricts the MMA cell to M ∈ {7,8}, which leaves
  `_m<T,4,4>` = 104 instantiated, so the ceiling would be **104, not 89** — a
  −3.7 % reduction, not −17.6 %. This is arithmetic over the measured per-cell
  table (`narrow` 89, `m3_ipg3` 83, `m4_ipg4` 104, `m5_ipg3` 87, `m6_ipg3` 83,
  `m9_ipg3` 83, `sgmm` 34), **derived, not separately measured.** Please do not
  keep using 89 as available headroom.
- **On the H1 occupancy prediction:** I pre-registered that a −17.6 % ceiling drop
  should make untouched widths ≈ +0.17 % faster, below the MDE. The guard mean is
  **+0.224 %** — strikingly close — but pooled over 6 measurements this is
  t = 1.94, df = 5, p ≈ 0.11, and the guard's own worst |effect| is 0.628 %.
  **Directionally consistent, not established.** I am not claiming it.
- **Prompt or M5 transfer risk:** this is a local `applegpu_g16s` microbenchmark;
  the ranked box is `applegpu_g17s`. Both have `vector_limit = 10 > 9`, so all
  legal widths take the qmv path on both. The result cannot be converted into a
  score, because that needs the decode depth mixture and edward's E43 established
  the mixture is **not identified** — a step at M ≥ 6 and a plain quadratic both
  fit the ranked row with zero slack. Note the risk direction: if the real
  mixture concentrates at M ≤ 6 or M = 9, the all-widths candidate is a **large
  regression**, not a small one.
- **Smallest useful next action:** build the M ∈ {7,8}-restricted variant
  (`case 7: case 8:` fall through to the sgmm cell; cases 4, 5, 6, 9 keep their
  base cells) and re-run this exact session. No new measurement is needed to
  predict its per-width outcome — M=7 and M=8 are already measured at +10.46 %
  /+16.65 % (attn_out) and +4.46 %/+13.05 % (mlp_down), with every other width at
  base performance by construction. A confirmation run is still required because
  the shared allocation changes 89 → 104, which perturbs *all* widths through
  occupancy; that is exactly the register-gate-first discipline this assignment
  was built around, so it deserves its own pre-registered bound rather than being
  folded in silently here.
- **Recommendation: close the all-widths mechanism, reopen narrowly as
  M ∈ {7,8}.** The kernel construct itself is sound, exact, cheap in registers
  and compiles both ways; only the dispatch range was wrong.

## Suggested follow-ups (not implemented)

1. **M ∈ {7,8}-restricted dispatch** — as above, the highest-value follow-up.
2. **`uint4` batching of A-fragment loads.** Each lane loads one `uint32` per MMA
   and uses 2 of its 8 nibbles. Deliberately not implemented: it saves only ~12 %
   of instructions (6 → 5.25 per lane per MMA), would invalidate the Gate 0
   numbers, and adds a second path. The flat-cost result now says this kernel is
   **not** ALU-bound in the regime that matters, so this is lower value than I
   thought when I deferred it.
3. **Two-row-tile variant for small M.** The whole M=4–6 loss is paying for 8 rows
   when 4–6 are wanted. A 4-row fragment shape, if `BaseMMAFrag` supports one on
   this target, would move the crossover down. Unverified.
4. **Do not** chase M=9 with a second tile; 1.6× the plateau against a base that
   only adds a third stream is structurally unattractive.
