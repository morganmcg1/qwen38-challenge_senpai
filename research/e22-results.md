# E22 results: the C(M) dispatch-cost staircase, M = 1..9 at bits == 4

Assignment: `qwen38-r1-e22-narrow-width-dispatch-cost-curve` (PR #26, revision `r1`).
Branch `qwen-thorfinn/narrow-width-dispatch-cost-curve`, base `senpai/qwen38-mtp-r1`
@ `c0f7e370921a14f348fa1872f2176b1b43028752`.

W&B run: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/zmu41ta1>
(id `zmu41ta1`), tables `e22/cost_curve`, `e22/round_curve`, `e22/dispatch`.

**Host caveat, stated once and meant everywhere below.** Every number here was
measured on an Apple M4 Pro (`mem=51539607552`), not on the ranked M5 host
`m5-qwen38-27b-mtp`. These are per-call kernel microbenchmarks of the same
`qmv_fast_crossrow_affine4_g64*` family the scored worker dispatches, so the
*shape* of the curve transfers as directional evidence. No absolute microsecond
figure and no derived policy number here is a ranked-M5 claim.

## What was asked, and what changed

The A3X arm (low-bit verify-side head at M=4..9) is **dead: H_MISSING**. It is
withdrawn, and the earlier banked negative is reclassified as recorded in the
advisor redirect: it is a **fidelity constraint, not a physics result**. Zero
GPU time was spent on A3X. The full allocation went to Question #1, the C(M)
staircase.

## Reproduction

```bash
research/run-qmv-curve.sh e22-cm-r1 c0f7e370921a14f348fa1872f2176b1b43028752 --reps 21 --inner 10 --skip-stock
research/run-qmv-curve.sh e22-cm-r2 c0f7e370921a14f348fa1872f2176b1b43028752 --reps 21 --inner 10 --skip-stock

python3 research/e22_cm_report.py \
  --r1 .mlxfast-private/qmv-curve/e22-cm-r1/vendored.json \
  --r2 .mlxfast-private/qmv-curve/e22-cm-r2/vendored.json \
  --identity .mlxfast-private/qmv-curve/e22-cm-r1/identity.txt \
  --identity .mlxfast-private/qmv-curve/e22-cm-r2/identity.txt \
  --out .mlxfast-private/qmv-curve/e22-cm-report.json --wandb
```

Recorded identities:

```text
r1  head=9a3a295819054e92335b603bfb4652bb1a362f34 dirty=0  started_utc=2026-08-18T02:26:32Z
    cool_gate_vendored=stalled_above_40C  gpu_temp_c_before=43.007  after=79.284
r2  head=d33c36b5340fbf4f34e6f68191df6bf2bdeb3800 dirty=0  started_utc=2026-08-18T02:34:53Z
    cool_gate_vendored=stalled_above_40C  gpu_temp_c_before=43.394  after=82.833
```

`reps=21 inner=10`, `bits=4`, widths `1..12` plus the far tail. Both runs held
`benchmark.sh`'s run lock. The cool gate stalled in both: this host's **idle GPU
floor is ~43.0-43.4 C against a 40 C target**, so it can never pass. See
"Two harness defects" below.

## The measured curve

### `head.lm_head` (k=5120, n=248320, weight_bytes=715161600, calls_per_verify=1)

| M | C(M) us | C/C(1) | C/M us | str | ipg | spread | kernel |
|---|---|---|---|---|---|---|---|
| 1 | 2856.40 | 1.0000 | 2856.40 | 1 | 1 | 0.69% | `qmv_fast_impl` |
| 2 | 2853.73 | 0.9991 | 1426.86 | 1 | 2 | 0.48% | `qmv_fast_crossrow_affine4_g64<T, 2>` |
| 3 | 2934.80 | 1.0274 | 978.27 | 1 | 3 | 0.46% | `..._m<T, 3, 3, true>` |
| 4 | 3403.91 | 1.1917 | **850.98** | 1 | 4 | 0.31% | `..._m<T, 4, 4, true>` |
| 5 | 5307.85 | 1.8582 | 1061.57 | 2 | 3 | 0.17% | `..._m<T, 5, 3, true>` |
| 6 | 5715.93 | 2.0011 | 952.65 | 2 | 3 | 0.27% | `..._m<T, 6, 3, true>` |
| 7 | 6198.79 | 2.1701 | 885.54 | 2 | 4 | 0.23% | `..._m<T, 7, 4, true>` |
| 8 | 8132.51 | 2.8471 | 1016.56 | 3 | 3 | 0.14% | `..._m<T, 8, 3, true>` |
| 9 | 8616.01 | 3.0164 | 957.33 | 3 | 3 | 0.18% | `..._m<T, 9, 3, true>` |
| 10 | 12479.37 | 4.3689 | 1247.94 | - | - | 0.41% | `qmm` |
| 11 | 12479.02 | 4.3688 | 1134.46 | - | - | 0.46% | `qmm` |
| 12 | 12491.54 | 4.3732 | 1040.96 | - | - | 0.44% | `qmm` |

### `mlp.gate_up_fused` (k=5120, n=34816, weight_bytes=100270080, calls_per_verify=64)

| M | C(M) us | C/C(1) | C/M us | str | ipg | spread |
|---|---|---|---|---|---|---|
| 1 | 407.93 | 1.0000 | 407.93 | 1 | 1 | 3.83% |
| 2 | 411.41 | 1.0085 | 205.71 | 1 | 2 | 0.78% |
| 3 | 450.23 | 1.1037 | 150.07 | 1 | 3 | 0.69% |
| 4 | 521.69 | 1.2789 | **130.42** | 1 | 4 | 1.84% |
| 5 | 787.85 | 1.9313 | 157.57 | 2 | 3 | 1.05% |
| 6 | 846.47 | 2.0750 | 141.08 | 2 | 3 | 0.88% |
| 7 | 915.62 | 2.2445 | 130.80 | 2 | 4 | 0.85% |
| 8 | 1185.14 | 2.9052 | 148.14 | 3 | 3 | 0.54% |
| 9 | 1244.93 | 3.0518 | 138.33 | 3 | 3 | 0.33% |
| 10 | 1830.88 | 4.4882 | 183.09 | - | - | 0.43% |
| 11 | 1832.33 | 4.4917 | 166.58 | - | - | 0.65% |
| 12 | 1833.67 | 4.4950 | 152.81 | - | - | 0.64% |

### Round-weighted (sum over all 8 scored shapes of `calls_per_verify` x C(M))

| M | C_round ms | /C(1) | /M ms | step |
|---|---|---|---|---|
| 1 | 59.871 | 1.0000 | 59.871 | - |
| 2 | 63.886 | 1.0671 | 31.943 | 1.0671 |
| 3 | 72.918 | 1.2179 | 24.306 | 1.1414 |
| 4 | 83.031 | 1.3868 | 20.758 | 1.1387 |
| 5 | 120.886 | 2.0191 | 24.177 | **1.4559** |
| 6 | 129.284 | 2.1594 | 21.547 | 1.0695 |
| 7 | 139.325 | 2.3271 | **19.904** | 1.0777 |
| 8 | 177.758 | 2.9690 | 22.220 | **1.2759** |
| 9 | 186.422 | 3.1137 | 20.714 | 1.0487 |
| 10 | 272.079 | 4.5444 | 27.208 | **1.4595** |
| 11 | 272.173 | 4.5460 | 24.743 | 1.0003 |
| 12 | 272.380 | 4.5495 | 22.698 | 1.0008 |

## Adjudication

### P1 passes; `DIRECT_NIBBLES` on the `<T,2>` pair kernel is worth ~0

The advisor's free falsifiable prediction is **confirmed**. On the two shapes
the brief required, `C(2)/C(1)` is **0.9991** (`head.lm_head`) and **1.0085**
(`mlp.gate_up_fused`) - a second row is free to within the 0.5-0.8% measurement
spread, and on the largest shape it is very slightly *negative*. The positive
control holds and the pair kernel has no headroom to recover.

The round-weighted `C(2)/C(1)` is 1.0671, which is larger than either required
shape. That is not a contradiction: it is dominated by the small-`n` shapes,
and specifically by `mlp.down` (k=17408, n=5120), the only shape whose `C(1)`
anchor is itself unstable between repeats (5.08% at M=1, 3.85% at M=2, versus
<=0.5% everywhere else in the whole grid). Read the round-weighted `C(2)/C(1)`
as "about 1.0-1.07, anchored on the noisiest point in the sweep", and read the
per-shape numbers as the result.

### P2 is refuted; P2' is confirmed. The second step is at M=7->8, not M=8->9

The brief predicted stream boundaries at **M=4->5 and M=8->9**, derived from
`ceil(M/ceil(M/4))`. That formula does not describe the live kernel. Reading the
live dispatch table in `Vendor/.../kernels/quantized.h` (the `out_vec_size >= 4096`
tier at `:1823`, inside the gate at `:1822`), the actual stream counts are:

| M | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| streams | 1 | 1 | 1 | 2 | 2 | 2 | **3** | 3 |
| inputs/group | 2 | 3 | 4 | 3 | 3 | 4 | 3 | 3 |

I pre-registered this correction as **P2'** in `research/e22-prereg-q1-addendum.md`
**before any timed C(M) run**, precisely so the correction could not be a
post-hoc fit. The measurement agrees with P2' and not with P2:

- live boundaries `M=[5, 8]`, steps `{5: 1.4559, 8: 1.2759}` - both large,
  both far outside noise;
- brief boundaries `M=[5, 9]`, where the claimed M=9 step is **1.0487**, which
  is indistinguishable from the ordinary within-stream increments on either
  side of it (1.0695 at M=6, 1.0777 at M=7).

So the staircase is real and lands where the live source says it lands.

### Magnitude, reported and not assumed

The advisor's warning was explicit: the shape is expected, the magnitude is a
live open question, and the campaign has already refuted one over-prediction of
it. Measured, round-weighted:

- **first stream boundary (M=4->5): +45.6%**
- **second stream boundary (M=7->8): +27.6%**

The "stream count drives verify cost" family survives decisively. Measured as
excess over 1.0, the two boundary steps are **3.2x and 2.0x** the largest
ordinary within-stream increment (1.1414 at M=3). Both reproduce across
independent repeats to within 0.25% on the step ratio.

### P3 (the in-source non-monotonicity claim) does not reproduce

The `case 8` comment in `quantized.h` claims `319/437/216 us for M=7/8/9`,
i.e. `C(9) < C(7)`. Measured: the round-weighted curve is **monotone
non-decreasing across the entire range**, with `C(8)/C(7) = 1.27585` and
`C(9)/C(8) = 1.04874`. This is consistent with that comment's own note that the
shipped 3+3+2 split fixed the 4+4 cliff it was describing - the claim is stale,
not wrong-at-the-time. **Propose: refresh that comment.** Not shipped.

### The largest cliff in the whole curve is not a stream boundary at all

The single largest step is **M=9->10 at 1.4595x**, the `qmv` -> `qmm`
transition at `vector_limit = 10` (`quantized.cpp:1415`, branch `:1418`;
K=5120 > 4096 gives `vector_limit = 10`). Above it the curve is flat
(1.0003, 1.0008): `qmm` prices M=10, 11 and 12 identically. This is outside the
assignment's M<=9 window and outside the contract's max draft count of 8, so it
is recorded as context, not as a proposal.

### C(M)/M is non-monotonic, with minima at the IPG=4 widths

Per-row cost has local minima at **M=4 and M=7** and local maxima at **M=5 and
M=8**. Both minima are the `inputs_per_group == 4` widths - the last width
before each stream boundary. Both maxima are the first width after one. On
`head.lm_head` the best per-row cost is M=4 at 850.98 us, 19.8% cheaper per row
than M=5; on `mlp.gate_up_fused`, M=4 at 130.42 us and M=7 at 130.80 us are
effectively tied for best.

## What this says about the S18 cost model (proposal only, nothing shipped)

`Qwen36MTPBlockSession.swift:529` sets `headStepCostRatio = 0.18`, and
`costModelDepth` (`:599-634`) prices depth with a single scalar `h`. Pricing the
measured C(M) against the S18 width mix - quoting the provenance verbatim, **a
512-token measurement under E17's S18 policy**, depth histogram
`{1: 19, 2: 138, 3: 67, 4: 21}`, `M = depth + 1`:

```text
verify seconds if cost were constant in depth: 14668.38 ms
verify seconds at the measured C(M):           19378.16 ms
understates by 32.1% anchored at C(1), 23.8% anchored at C(2)
```

That headline number is, on its own, **not** an indictment of `h = 0.18`, and it
would be dishonest to present it as one. The comment at `:579-587` records a
real, careful sweep: `h = 0.32` scored 2.84585 (-3%) with the baseline leg flat,
`h = 0.15` gave 2.667 and `h = 0.14` gave 2.766. `h` is bracketed on both sides
and 0.18 is a true local optimum. Any story that predicts "0.18 badly
underprices depth, so raise it" is already refuted by that sweep.

The decomposition that reconciles the two:

```text
best-fit constant (what a scalar h already absorbs): 79.09 ms/round
residual a scalar cannot absorb, C(M)/best-fit:
  M=2: 0.808   M=3: 0.922   M=4: 1.050   M=5: 1.528
residual span over the mix: 1.892x
marginal cost of one more draft, C(M+1)/C(M):
  depth 1->2: 1.141   depth 2->3: 1.139   depth 3->4: 1.456   depth 4->5: 1.069
```

A scalar `h` fitted end-to-end **cannot be wrong about the mean**: it absorbs
whatever constant best prices the observed mix, which is exactly why the sweep
found 0.18 bracketed on both sides. What a scalar provably **cannot** express is
the **1.89x residual span**, or equivalently the fact that the marginal price of
one more draft jumps **28%** going from depth 3 to depth 4 (1.139 -> 1.456) and
then falls back. That jump is exactly the M=4->5 stream boundary. Under the S18
mix, 138 of 245 rounds (56%) sit at depth 2 (M=3), 67 sit at depth 3 (M=4) - the
last width before the jump - and 21 sit at depth 4 (M=5), having paid it.

So the testable proposal is not "change `h`". It is: **the depth threshold in
`costModelDepth:628` currently pays a flat marginal price per draft, and the
measurement says the marginal price is flat except for one large step at the
stream boundary.** A depth policy that knows where the boundary is could decline
the boundary-crossing draft while still taking the cheap ones, which a scalar
cannot do at any value of `h`. This is *consistent with* the h-sweep rather than
refuted by it, which is what makes it worth a separate experiment.

**Not shipped.** Per the assignment's scope discipline, this is a proposal for
the advisor to assign or reject; no policy change is in this branch.

## Two in-source claims I propose correcting (not shipped)

1. `quantized.h:1834-1861` sizes `DIRECT_NIBBLES` as "worth ~5% of verify
   rounds", sourced from a **128-token `--local-submit` receipt `e29a3e0d`**.
   Calibration fact (f) forbids sizing a width/depth-coverage claim from a
   `--local-submit` receipt. The direct measurement above puts the `<T,2>` pair
   kernel's headroom at ~0, so the number and its instrument are both suspect.
2. The same comment asserts "ALU-bound" too broadly. The banked bits-grid
   roofline (k=5120, n=98336) puts single-pass-equivalent bandwidth at 242.5 and
   243.7 GB/s for M=1 and M=2 - at roofline, i.e. **bandwidth**-bound - and only
   drops to 205.2 GB/s (-15.4%) at M=4. The ALU-bound regime begins at M>=3 and
   is worst at M=4; it does not describe M=1..2.

## Two harness defects found and fixed en route

Recorded so the advisor knows why the timed runs slipped, per the
error-and-crash instruction.

1. **Cool gate could never pass on this host.** `run-qmv-curve.sh` aborted when
   the GPU would not reach 40 C, but this host's idle floor is ~43.0-43.4 C. Job
   `a3da9784` died at 228.6 s (`started_utc=2026-08-18T02:15:33Z`) having burned
   its budget waiting. Fixed in `9a3a295` by adopting the same
   record-and-proceed behaviour `research/run-draft-bits-sweep.sh:94-98` already
   used: the gate now tees `cool_gate_<phase>`, `gpu_temp_c_before_<phase>` and
   `gpu_temp_c_after_vendored` into `identity.txt` instead of aborting. Both
   reported runs carry `cool_gate_vendored=stalled_above_40C` honestly.
2. **bash 3.2 `set -u` bug.** With `--skip-stock`, `stock_flag` and `wandb_flag`
   were empty arrays, and `"${a[@]}"` on an empty array is "unbound" under
   bash 3.2 - so the summary step died *after* a good measurement had already
   been taken. Fixed in `d33c36b` with `${a[@]+"${a[@]}"}`.

A third defect was in the analysis tooling, not the runner: the advisor's
`ceil(M/4)` stream formula had a **second independent instance** in
`research/qmv_cost_curve_summary.py`, which would have silently re-derived the
refuted P2 boundaries in the summary output and in the W&B plot title. Fixed in
`c457b68` by introducing a `LIVE_STREAMS` table read from the live kernel and
making `stream_boundaries` shape-aware.

Separately, `939c18a` fixed a real parser defect in my own gate reader: the
crossrow `switch` appears once per `out_vec_size` tier, and a flat
last-writer-wins parse let the narrow tier overwrite the wide one.
`CrossrowGate.coveredRows` is now keyed `[bits][out_vec tier][case]` and
`inKernelPath(bits:m:n:)` selects the largest tier <= n. The same commit
corrected the citation the brief carried, `1804-1908` -> live `1822-1955`.
Per the advisor's own adjudication, that citation drift was an advisor error,
not a student error; it is recorded here only because the corrected line
numbers are load-bearing for the P2' argument above.

## Reproducibility

Two independent runs, r1 (`head=9a3a295`) and r2 (`head=d33c36b`); the
difference between those two heads is the bash 3.2 fix, which cannot affect
kernel timing.

Every round-weighted step reproduces:

| M | r1 step | r2 step |
|---|---|---|
| 5 | 1.4559 | 1.4525 |
| 8 | 1.2759 | 1.2756 |
| 9 (non-step) | 1.0487 | 1.0492 |
| 10 | 1.4595 | 1.4574 |

Per-point worst disagreement across the whole 8-shape x 12-width grid is
**5.079%**, confined to `mlp.down` at M=1 (and 3.845% at M=2); every other point
in that shape is <=0.5%, and the two required shapes are within **0.64%**
(`head.lm_head`) and **0.47%** (`mlp.gate_up_fused`) worst-case. The
round-weighted C(1) anchor therefore carries ~1.2% of uncertainty, which is the
only reason to prefer the per-shape P1 numbers over the round-weighted one.

## Suggested follow-ups (not implemented)

1. **Boundary-aware depth policy.** Test whether replacing the scalar marginal
   price in `costModelDepth` with a two-piece price - cheap within a stream,
   expensive at the M=4->5 crossing - beats `h = 0.18` end-to-end. The h-sweep
   cannot answer this; a scalar has no way to express it.
2. **Confirm the boundary on ranked M5.** The stream boundaries follow from the
   dispatch table, which is host-independent, but the 45.6% and 27.6%
   magnitudes are M4 Pro numbers. If the M5 magnitudes are materially smaller,
   proposal 1 loses most of its value, so measure before building.
3. **Ask why M=2 is free on `head.lm_head` but not round-weighted.** The
   round-weighted C(2)/C(1) is carried almost entirely by `mlp.down`'s unstable
   M=1 anchor. A short targeted repeat on that one shape would either stabilise
   it or expose something real about the k=17408 layout.
4. **Refresh the two stale in-source comments** (`case 8` non-monotonicity, and
   the `e29a3e0d`-sourced `DIRECT_NIBBLES` sizing plus the over-broad
   "ALU-bound"). Documentation-only; no behaviour change.
