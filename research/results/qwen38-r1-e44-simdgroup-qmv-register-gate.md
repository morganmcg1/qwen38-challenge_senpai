SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mean_speedup_over_replaced_widths_pct","available":true,"value":11.421},"test_metric":{"name":"exact_coverage_bad_elements","available":true,"value":0}}

# E44 r2 — narrow `M ∈ {7,8}` MMA dispatch: all three gates pass, +11.4 % on the replaced widths, and one blocker before anything can ship

- **Student / branch:** `qwen-alphonse` / `qwen-alphonse/simdgroup-qmv-register-gate`
- **Revision:** r2. r1's full report is retained verbatim at the bottom of this
  file; nothing in it has been edited.
- **Hypothesis:** r1 refuted the all-widths `M ∈ {4..9}` dispatch (−7.341 % on
  net, 7/12 replaced widths regressing). r2 keeps the identical kernel construct
  and narrows the dispatch to the two widths that won in r1, `M ∈ {7,8}`,
  restoring the base scalar cells at `M ∈ {4,5,6,9}`.
- **Decision: the speed mechanism is confirmed and the narrow arm clears the
  5 % bar at +11.421 %, but it is NOT shippable yet.** Its outputs are not
  bit-identical to base at the replaced widths, so the golden-decode exactness
  gate is a real open risk, not a formality.
- **The win is reported per width and per shape, not pooled.** The four
  independently resolved cells and their score conversions are in
  *Width term resolved per width and per shape*; the pooled `+11.421 %` is a
  headline only. Applying askeladd's E42 census (`f{7,8} = 0.1225`) per width
  gives **dScore +0.789 % … +1.228 %** depending on the shape mixture — against
  a 0.7678 % board floor and a 0.5193 % crown gap — and that range **halves to
  +0.386 … +0.601 % if `f` is recomputed on the two prompts that actually set
  the median**. Four caveats are attached to those numbers in that section.
- **`BASE_SHA`:** `9fe0dc5dbdb30af4c807ea71873df99e2da72aa2` (r1 used
  `efff400c1b5554be2e8993b01856653d55de7664`). `UPSTREAM_SHA` unchanged from
  base. The shipped-surface diff `efff400c → 9fe0dc5d` over
  `Sources Vendor benchmark.json mlx-generated` is **empty**, so r1's per-cell
  register table transfers to this base without remeasurement.
- **Submitted candidate files: NONE.** As instructed, both scored files are
  reverted to their base blobs in the final commit and
  `git diff 9fe0dc5d HEAD -- Sources Vendor benchmark.json mlx-generated` is
  **empty**. The measured candidate is preserved as commit `e66ddd0` on this
  branch and can be replayed from it.
- **Yukon:** not submitted. Nothing here is bankable yet.
- **W&B run:** `dn6hk8u7` —
  <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/dn6hk8u7>
  (per-width table, fidelity table, raw pairs, A/A control table, cost model,
  score decomposition, and all four evidence artifacts attached).
  r1's run was `3fi0jrgh`.
- **Reproduction.** From `e66ddd0` (the measured candidate), on this host:
  ```bash
  export MLXFAST_LOCAL_RUN_LOCK_DIR=/tmp/mlxfast-shared   # cross-role GPU lock
  research/run-e44-qmv-ab.sh r2cov 9fe0dc5d --coverage 1 --widths 1,4,6,7,8,9
  research/run-e44-qmv-ab.sh r2ab  9fe0dc5d --pairs 9 --reps 50 --inner 20 \
      --widths 1,2,3,4,5,6,7,8,9
  research/run-e44-qmv-ab.sh r2aa  9fe0dc5d --cand-rev 9fe0dc5d \
      --pairs 9 --reps 50 --inner 20 --widths 1,2,3,4,5,6,7,8,9
  python3 research/e44_ab_summary.py .mlxfast-private/e44-qmv-ab/r2ab \
      --touched 7,8 --control .mlxfast-private/e44-qmv-ab/r2aa
  research/e44_sgmm_air.sh          # Gate A, compile-only
  python3 research/e44_census_score.py   # census -> per-cell score, no GPU
  ```
  Runtime: Gate B 4.6 min, Gate C primary 6.7 min, A/A control 11.0 min.
  Peak memory is negligible (a microbenchmark holding tens of MB); the harness
  refuses to run if a model-holding process is resident.
- **MTP head provenance:** unchanged, organizer-pinned, no
  `mtp-head.manifest.json` declared.
- **Host / toolchain / thermal:** Apple M4 Pro, `applegpu_g16s`, 48 GiB, Apple
  metal 32023.883. The ranked box is M5 Max, `applegpu_g17s`.
  **`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`** —
  preserved verbatim. This host's idle GPU floor sits above the 40 C target, so
  every timed arm here is an ungated, ABBA-counterbalanced local session:
  directional causal evidence, never a score.
- **Pre-registration:** `research/e44-r2-prereg.md`, committed in `502f715`
  **before the narrow variant was compiled**, and posted to the PR. Method
  record in `research/e44-artifacts/r2-method.md`, written while the primary
  timing session was still running.

## Result summary

| gate | pre-registered condition | measured | verdict |
|---|---|---|---|
| A — registers | kernel-wide max lane-corrected ≤ 108 | **104** (`e44_m4_ipg4`) | **pass** |
| B — exact coverage | zero mismatching elements | **0 / 1,009,254,400** | **pass** |
| C — timing | net over `M ∈ {7,8}` ≥ +5 %, all four cells positive and resolved | **+11.421 %**, 4/4 positive and resolved | **pass** |
| C — predictions | no sign flip, no cell below +2 % | 4/4 within 1 pp | **pass** |
| C — flat tile | \|cand(7) − cand(8)\| / cand(8) < 3 % | 0.86 % / 1.33 % | **pass** |
| C — guard | no resolved untouched-width regression | **4/14 guard cells exceed the control floor, mixed sign** | **NOT met** |

Three of the four Gate C conditions passed and the fourth did not. The failure is
analysed honestly below rather than absorbed into the headline: it does not
threaten the +11.421 % width result, but it does mean the arm is not provably
neutral at untouched widths, and it is the reason the ceiling term is reported
with the conservative floor.

## Gate A — register bound (compile-only, zero GPU)

Artifact: `research/e44-artifacts/r2-gate-a-registers.txt`.

| quantity | pre-registered | measured | verdict |
|---|---|---|---|
| kernel-wide max, lane-corrected | ≤ 108 | **104** | pass |
| binding cell identity | `e44_m4_ipg4` | `e44_m4_ipg4` | pass |
| entry-block registers | [143, 163] | **160** (was 163) | pass |
| allocas in the sgmm cell | 0 | **0** | pass |
| new alloca type introduced | none | none (entry 55 → 52) | pass |
| static threadgroup bytes | 0 → 0 | 0 → 0 | pass |

Per-cell lane-corrected footprint: `narrow_*` 89, `m3_ipg3` 83, `m4_ipg4` 104,
`m5_ipg3` 87, `m6_ipg3` 83, `m7_ipg4` 108, `m8_ipg4` 104, `m9_ipg3` 83,
`sgmm` 34 (naive 344, gap 310 = 5 distributed values × 62 lanes, ordinary count
24 identical at both peak points, so rule (e) of the lane-correction contract
holds).

`affine_qmv_fast` has a single `[[kernel]]` entry point and the width `switch` is
on a runtime value with all helpers `METAL_FUNC` inline, so one shared register
allocation is taken as the max over all instantiated cells. Removing the sgmm
cell from `M ∈ {4,5,6,9}` leaves `m7_ipg4` (108) uninstantiated and makes
`m4_ipg4` (104) binding. **Ceiling 108 → 104 = −3.70 %.**

> 🔴 **Retraction that gates E46.** r1 reported the ceiling as `108 → 89`. That
> is correct only for the all-widths variant, which r1 itself proved is not
> bankable. **For the bankable narrow variant the ceiling is 104, a −3.70 %
> reduction, not −17.6 %.** r1 flagged this as derived arithmetic; r2 has now
> measured it directly and it is confirmed. Please do not let E46 size anything
> against 89.

## Gate B — exact integer coverage, and precisely what it does and does not prove

Artifact: `research/e44-artifacts/r2-gate-b-coverage.txt`, run dir
`.mlxfast-private/e44-qmv-ab/r2cov`.

24 cells = 2 shapes × 6 widths {1,4,6,7,8,9} × {scale pass, bias pass}.
**1,009,254,400 elements. `bad=0` and `worst_abs=0.0` on every line of both
arms. 0 / 24 cells with any nonzero.**

The construction forces every scale to 1 and every bias to 0 (scale pass), or
scales to 0 and biases to distinct small integers (bias pass), and sets `x`
one-hot on the eight `k` values packed in a single weight word. The exact answer
is then a small integer ≤ 120, which bfloat16 represents without loss, so the
check demands **bit equality of the stored result**, not a tolerance. Sweeping
the word index covers every `k` at per-nibble resolution, and one dispatch yields
every `(m, n)` at once.

**What this proves:** the index mapping, the `m`/`n` tiling, the nibble
unpacking, the group reduction, and the pairing of each group's bias with the
right partial sum are all exactly correct at every position. A single dropped,
duplicated or misplaced nibble moves the sum by at least 1 and would be visible.

**What this does NOT prove, and the distinction matters:** because every operand
is an exactly-representable small integer, *no rounding occurs anywhere* in
either arm, so reduction order is irrelevant by construction. Gate B is
therefore **silent on floating-point rounding**. It must not be read as "the
outputs are bit-identical". They are not — see the blocker below.

Dispatch was verified mechanically rather than trusted: the summary tool parses
the dispatch tables out of both emitted runtime-effective JIT sources and refuses
to report if they disagree with the claimed touched set. It derives **exactly
{7,8}**, with `M ∈ {4,5,6,9}` byte-identical scalar cells again.

## Gate C — paired timing

Design: 9 pairs × 50 reps × 20 inner, widths 1..9, ABBA alternation **inside one
process** at ~100 ms granularity, so drift between paired arms is far smaller
than between sessions. df = 8. Run dir `.mlxfast-private/e44-qmv-ab/r2ab`.
Entry GPU temperature 42.92 C, exit 69.99 C. GPU-busy precheck: idle
(peak 0 %, mean 0.0 %, 12 samples).

### Replaced widths

| shape | M | base µs | cand µs | speedup % | sd % | 95 % CI | verdict |
|---|---|---|---|---|---|---|---|
| attn_out | 7 | 154.03 | 136.48 | **+11.389** | 0.137 | [+11.284, +11.495] | clears bar |
| attn_out | 8 | 165.96 | 137.66 | **+17.050** | 0.227 | [+16.876, +17.225] | clears bar |
| mlp_down | 7 | 486.07 | 463.74 | **+4.596** | 0.116 | [+4.506, +4.685] | faster |
| mlp_down | 8 | 523.81 | 457.55 | **+12.649** | 0.223 | [+12.478, +12.820] | clears bar |

**Mean over `M ∈ {7,8}`: +11.421 %. Replaced widths regressing with a resolved
interval: 0 / 4. Best: attn_out M=8 at +17.050 %.**

### Pre-registered predictions: 4 / 4 confirmed

| cell | pre-registered | measured | error |
|---|---|---|---|
| attn M7 | +10.46 | +11.389 | +0.93 |
| attn M8 | +16.65 | +17.050 | +0.40 |
| mlp M7 | +4.46 | +4.596 | +0.14 |
| mlp M8 | +13.05 | +12.649 | −0.40 |

The falsification condition was any sign flip or any cell below +2 %. Neither
occurred, and every cell landed within 1 pp of its prediction. This is a clean
contrast with r1, where 3 of 4 pattern predictions missed — because r1's
predictions came from a *mechanism* (weight-stream halving) that was wrong,
whereas r2's come from *r1's own measurements* of the same two widths.

### Flat-tile check: passes on both shapes

Pre-registered as |cand(7) − cand(8)| / cand(8) < 3 %.

| shape | cand plateau µs | cv % | spread % | base rise M7→M8 |
|---|---|---|---|---|
| attn_out | 137.07 | 0.61 | **0.86** | +7.7 % |
| mlp_down | 460.64 | 0.95 | **1.33** | +7.8 % |

Candidate flat, base rising: the fixed 8-row MMA tile mechanism that r1
identified is confirmed on both shapes. The sign of the effect at any width is
set by where the flat candidate curve crosses the rising base curve, which is
exactly why narrowing the dispatch to the widths past the crossing works.

I also corrected the flatness diagnostic itself. It had `M=4..8` hardcoded from
r1 and was averaging three unchanged scalar cells into the narrow arm's
"plateau", reporting a meaningless 15–17 % CV. It now follows the touched
widths, capped at 8 because `M=9` needs a second 8-row tile and is genuinely not
flat. r1's numbers are unchanged by the fix (139.76 µs cv 2.28 %, 458.99 µs
cv 2.00 %).

### The A/A control, and the pre-registered condition that did not hold

The control is the same design with `--cand-rev` set to the base sha, so both
arms are the **same bytes** — verified: `base_metal_sha256 ==
cand_metal_sha256 == 0c461cb8…`, `arm_source_diff_hunks=0`. The summary tool
refuses a control whose arms are not byte-identical. Run dir
`.mlxfast-private/e44-qmv-ab/r2aa`, entry 42.89 C, exit 70.57 C.

**All 18 cells have a true effect of exactly zero. Result: worst |effect| =
0.263 %, sd of cell effects = 0.085 %, and intervals excluding zero = 0 / 18.**

That last number is the important one. **The design does not manufacture
"resolved" effects on a true zero.** So a resolved interval in the primary
cannot be dismissed as a broken statistic — which is inconvenient for me,
because two guard cells in the primary do resolve in the slower direction.

**My pre-registered stop rule required "no resolved untouched-width regression",
and that condition was NOT met.** Reporting it plainly:

| untouched cell | effect | 95 % CI | vs 0.263 % control floor |
|---|---|---|---|
| attn_out M=1 | **+0.539 %** faster | [−1.546, +0.468] on delta | exceeds |
| attn_out M=2 | **−0.663 %** slower | [+0.376, +0.950] on delta | exceeds |
| attn_out M=5 | **+0.341 %** faster | [−0.917, +0.235] on delta | exceeds |
| mlp_down M=1 | **−0.347 %** slower | [−0.194, +0.888] on delta | exceeds |

4 of 14 guard cells exceed the control's floor, and **the signs are mixed: two
faster, two slower**, averaging +0.055 % overall.

Here is what I think is actually going on, stated as a hypothesis rather than a
conclusion. The guard cells share their *cell source* with base, but they are
compiled into a **different binary**: the register allocation changed 108 → 104
and the entry block changed 163 → 160, which shifts code layout for every width.
The A/A control holds the binary fixed and so measures only measurement noise
(0.263 %); the guard varies the binary and picks up layout and allocation
scatter on top. Mixed signs of a few tenths of a percent are the signature of
that, not of a coherent occupancy change — a real ceiling reduction should make
untouched widths uniformly *faster*, and two of these four are slower.

**Two consequences, and I am not softening either:**

1. The pre-registered condition failed. The narrow arm is not perfectly neutral
   at untouched widths in this session, and I cannot rule out a genuine ±0.7 %
   width-dependent effect. This deserves a follow-up rather than a shrug.
2. **The ceiling term therefore cannot be read off the guard at all**, because
   the guard confounds occupancy with code layout. This is why the ceiling term
   stays a bound.

**Choice of floor.** Two floors are available and they disagree:

| floor | measures | value |
|---|---|---|
| A/A control | measurement noise, **identical** binaries | 0.263 % |
| untouched-width guard | noise + layout/allocation, **different** binaries | 0.663 % |

The ceiling effect is claimed to act *through* the binary difference, so
bounding it with the identical-binary floor would assume away exactly the
variability the bound must cover. **The larger, more conservative 0.663 % is
used**, giving `|dScore| ≤ 0.1186 %` rather than the flattering 0.0470 %. I
changed the tool to always take the larger of the two and to print both, so this
cannot be quietly reversed later.

## Score decomposition — two terms, opposite signs, never aggregated

The candidate changes two things at once and they push the score in **opposite
directions**, so a single aggregate speedup number would be uninterpretable.
Reported separately, each with its sign attached.

### Width term — favourable, and powered

`M ∈ {7,8}` exist only on the MTP leg; the serial leg never dispatches them.
With ψ_mtp = 0.6736, removing 1 % of MTP-leg QMV cost is worth **+0.674 %** of
score. The measured win is +11.421 %, so

> **dScore = +7.693 % × f**

where `f` is the share of MTP-leg QMV cost dispatched at `M ∈ {7,8}`.

| f | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|
| dScore | +0.385 % | +0.769 % | +1.923 % | +3.847 % |

The sensitivity table above was written before a census existed. It is
superseded by the measured `f` below and is kept only to show what was claimed
in advance. Note that the risk direction is benign in a way r1's was not: the
narrow arm is *neutral by construction* at every width it does not touch, so a
mixture concentrated at `M ≤ 6` or `M = 9` makes this candidate worthless
rather than harmful, whereas the all-widths arm would have been a large
regression.

### Width term resolved per width and per shape — the re-weightable form

A pooled mean cannot be re-weighted once a real width census lands, so the
primary result is also carried as four independent cells. For one cell,

> `dScore(M, shape) = ψ_mtp × speedup(M, shape) × f(M) × s(shape | M)`

with `ψ_mtp = 0.6736`, `f(M)` the share of MTP-leg QMV cost dispatched at that
width, and `s(shape | M)` that shape's share of the cost within the width.

| shape | M | speedup | 95 % CI | dScore per 1 pp of `f(M)` | dScore at census `f(M)`, `s = 1` |
|---|---|---|---|---|---|
| `attn_out` | 7 | +11.389 % | [+11.284, +11.495] | +0.07672 % | +0.361 % |
| `mlp_down` | 7 | +4.596 % | [+4.506, +4.685] | +0.03096 % | +0.146 % |
| `attn_out` | 8 | +17.050 % | [+16.876, +17.225] | +0.11485 % | +0.867 % |
| `mlp_down` | 8 | +12.649 % | [+12.478, +12.820] | +0.08520 % | +0.643 % |

The two shape rows at a given `M` are **alternatives, not addends**: the last
column asks what that width would be worth if all of its QMV cost were that one
shape. Every cell is independently resolved, so any future census can be applied
directly to this table without re-running anything.

### Applying askeladd's E42 corpus census

askeladd supplied a real dispatch census (tree
`04ad6bf11437c269df85a47e91faa769c74fe6da`): 78 dispatches distributed
1 / 5 / 5 / 23 / 4 / 6 / 34 over `M = 2 / 4 / 5 / 6 / 7 / 8 / 9`, mean `M`
7.269. Cost-weighting those counts with thorfinn's E46 refit
`T = 16.757 + 27.532·ceil(M/IPG) + 9.624·M` gives the cost shares

| M | 2 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| cost share | 0.54 % | 3.50 % | 5.07 % | 25.19 % | 4.71 % | 7.55 % | 53.45 % |

so **`f{7,8} = 0.1225`**. This is the first non-speculative value for `f`. I did
not measure it: askeladd's E48 / PR 52 owns that census and I consumed it rather
than reproducing it.

Applying it per width and then summing, for four mixtures of the two shapes:

| shape mixture | effective speedup over `M ∈ {7,8}` | dScore |
|---|---|---|
| all `mlp_down` | +9.555 % | **+0.789 %** |
| cost-proportional (one call of each per layer) | +10.835 % | **+0.895 %** |
| equal weight per cell | +12.215 % | **+1.009 %** |
| all `attn_out` | +14.875 % | **+1.228 %** |

**Why the per-width form matters, concretely.** Pooling first and weighting
afterwards gives `0.6736 × 11.421 % × 0.1225 = +0.942 %`. Weighting each width
first and then summing gives **+1.009 %** for the very same equal-shape mixture.
The 0.067 pp gap is not rounding: `M = 8` both wins more (+14.85 % mean vs
+7.99 % at `M = 7`) and carries more census cost (7.55 % vs 4.71 %), so the
pooled mean systematically understates the census-weighted value. That gap is
the entire reason the four-cell table above is the artifact to keep and the
pooled `11.421 %` is only a headline.

Four caveats travel with every number in this subsection.

1. **The census is corpus-wide; the score is not.** The published score is the
   median over eight prompts, i.e. the mean of the 4th and 5th order statistics —
   in practice beagle and medicine, not the corpus. beagle's mean draft width is
   5.533 against the corpus 7.269, so `f{7,8}` on the two prompts that actually
   set the score is very likely **lower** than 0.1225. `dScore` is linear in `f`,
   so halving it to 0.06 moves the range to **+0.386 … +0.601 %**. Against the
   0.7678 % board floor and the 0.5193 % crown gap: at the census `f` all four
   mixtures clear both, though the `mlp_down`-dominated end clears the board
   floor by only 0.021 pp and so should not be treated as clearing it at all;
   at the halved `f` **nothing** clears the board floor and only the
   `attn_out`-dominated end still clears the crown gap. The honest statement is
   that this candidate is reliably crown-gap-sized and only conditionally
   board-floor-sized, until the census is recomputed on the two prompts that
   set the median.
2. **Quote the range, not the midpoint.** The shape mixture `s(shape | M)` is
   not identified either, and it moves the answer by a factor of 1.56
   (+0.789 % to +1.228 %). Reporting a single midpoint would hide that.
3. **`T(M)` is a microbenchmark aggregate.** Only its ratios transfer; the cost
   shares inherit the same host-transfer risk as every absolute timing in this
   report.
4. **The bankable register ceiling is 104, not 89.** I retract the "89 as
   headroom" framing from r1. The measured kernel-wide maximum is 104, so the
   ceiling move is −3.70 %, it gates thorfinn's E46 at that value, and by the
   next subsection it is **adverse** — it subtracts from the numbers above
   rather than adding to them.

None of this is bankable while the exactness blocker below is open.

### Ceiling term — adverse, and NOT powered

`qmv_fast` has one shared register allocation, so changing it perturbs **every**
width including `M = 1`, and both legs speed up together. The serial leg is the
more QMV-dominated of the two (ψ_serial = 0.8525 vs ψ_mtp = 0.6736), so a
uniform QMV speedup helps the denominator more than the numerator:

> **dScore = −0.179 % per 1 % of uniform QMV cost removed — ADVERSE.**

The measured guard mean is +0.055 %, which is **below this session's own
resolution floor**. It is therefore reported as a bound:

> **|dScore| ≤ 0.1186 %** — a bound, not a measurement.

This was stated before the numbers existed and has not been revised after
seeing them. At the E40 price of 0.00959 % of cost per 1 % of register, the
−3.70 % ceiling move is worth about **0.0355 %** of score, and resolving that
against a ~0.63 % floor needs on the order of **880 pairs**. This design runs 9.
No claim is made that the ceiling effect was observed.

## 🔴 The blocker: the candidate is not bit-identical to base at the replaced widths

This is the finding that decides whether the +11.4 % is worth anything, and it is
easy to misread Gate B as having settled it. It has not.

On general random inputs, the harness measures each arm against an exact double
reference and against each other:

| shape | M | base max_rel | cand max_rel | **cand vs base max_rel** |
|---|---|---|---|---|
| attn_out | 1–6, 9 | 8.6e−02 … 6.3e−01 | identical | **0.0000e+00** |
| attn_out | **7** | 6.267e−01 | 3.636e−03 | **1.6788e+00** |
| attn_out | **8** | 6.267e−01 | 3.636e−03 | **1.6788e+00** |
| mlp_down | 1–6, 9 | 6.1e−02 … 1.1e+00 | identical | **0.0000e+00** |
| mlp_down | **7** | 1.099e+00 | 3.281e−03 | **5.2381e−01** |
| mlp_down | **8** | 1.099e+00 | 3.692e−03 | **5.2381e−01** |

Two things follow.

1. **At every untouched width the two arms are bit-identical** (`cand_vs_base`
   is exactly zero). That independently confirms those cells are genuinely
   unchanged code, which is what makes them valid as a zero-effect noise floor.

2. **At `M ∈ {7,8}` the candidate differs from base.** It is in fact *far more
   accurate* — its error against the exact double reference drops by roughly two
   orders of magnitude (6.3e−1 → 3.6e−3 max_rel; rms_rel 5.7e−2 → 1.7e−3),
   because the simdgroup MMA accumulates in fp32 whereas the scalar cell
   accumulates in lower precision. **But "more accurate" is not the contract.**
   The contract is reproducing the hidden *serial* token stream, and the serial
   reference is produced by the base numerics. A more accurate kernel that
   disagrees with base is exactly as disqualifying as a less accurate one if it
   flips a single argmax.

**Consequence: this candidate cannot ship on the strength of Gates A–C.** The
next gate is a fixed-window exact decode against the public golden — 512 tokens,
including exact post-EOS continuation and row-ledger closure — comparing the
candidate's token stream to the unchanged base's on this same host. That gate was
out of scope for this assignment and I did not run it.

I want to be explicit that this is a *substantive* risk, not paperwork. The
divergence is concentrated on exactly the two widths the candidate is meant to
accelerate, so there is no version of this mechanism that avoids the question.

## 🔴 Incidental finding: `twin_audit.py` is already RED on the campaign base

Found while running the final preflight, and worth a separate flag because it
degrades a safety gate for everyone, not just this experiment.

```text
base 9fe0dc5d (clean scratch worktree):   TWIN AUDIT FAILED: 1/29 twin(s)
candidate e66ddd0 (this branch):          TWIN AUDIT OK: 29 runtime-effective twin(s)
after reverting both files to base:       TWIN AUDIT FAILED: 1/29 twin(s)
```

Verified decisively: both scored files in my worktree hash-match their base blobs
exactly (`12e2f73d…`, `2429e888…`), and a **clean worktree checked out at
`9fe0dc5d` with none of my changes reproduces the same failure**. This is not
something r2 introduced.

The drift is in `quantized.h` at `case 8:` and is **comment-only** — the
surrounding code is identical in both, which is why runtime behaviour is
unaffected and none of the measurements above are compromised. But the two
comments say **opposite things about the same code**:

| source | comment at `case 8` |
|---|---|
| checked-in `quantized.h` | "**4+4**: two weight streams, receipted on this benchmark (scored 3.195804751396457 as a promoted submission) before a later stale-base REPLACE overlay reverted it; restored here." |
| runtime-effective `mlx-generated/quantized.cpp` | "**3+3+2, not 4+4.** M = 8 is the only hot width whose EVEN split needs two simultaneous `vec<float,4>` accumulators…" |

For the record, the `.h` comment is the accurate one: both dispatch tables — the
one I parse from `quantized.h` and the one I parse from the emitted
runtime-effective JIT string — independently resolve `case 8` to
`e44_m8_ipg4`, i.e. 4+4. The twin's "3+3+2" comment is stale text left behind
when the header comment was edited without regenerating the twin.

**Why it matters:** a gate that is already failing at base cannot detect a real
drift introduced later — the next person to run it sees a red result either way
and learns nothing. It also means the readable source and the runtime-effective
source currently disagree in their explanation of a hot width, which is exactly
the kind of thing that misleads a later experiment.

**I did not fix it.** The fix touches a scored file, and this assignment requires
the final diff against base to be empty. It is a one-line regeneration
(`make_compiled_preamble.sh`) and belongs in its own change so it is reviewable
on its own; my candidate `e66ddd0` already demonstrates that regenerating clears
it.

## Conclusion

- **What happened:** narrowing r1's refuted all-widths dispatch to the two widths
  that actually won turns a −7.341 % net regression into a **+11.421 % net win**
  with **0 / 4** replaced widths regressing, at a register ceiling of 104
  (−3.70 %) and with exact integer coverage over 1.009 billion elements. Every
  pre-registered condition on Gate A and Gate B held, and 4 / 4 pre-registered
  timing predictions were confirmed within 1 pp.
- **Evidence for the mechanism:** strong and now doubly confirmed. The flat-tile
  prediction holds on both shapes (spread 0.86 % / 1.33 % against a base rise of
  +7.7 % / +7.8 %), which is the mechanism r1 identified post hoc and r2
  pre-registered and confirmed.
- **The one pre-registered condition that failed:** "no resolved untouched-width
  regression". 4 of 14 guard cells exceed the A/A control's 0.263 % floor with
  **mixed signs** (attn M=1 +0.539 %, attn M=5 +0.341 %, attn M=2 −0.663 %,
  mlp M=1 −0.347 %). The A/A control resolved 0 of 18 cells on a true zero, so
  this is not a broken statistic; my best explanation is register-allocation and
  code-layout scatter between two genuinely different binaries, but I have not
  proved that and I am not claiming the arm is neutral at untouched widths.
- **What this is worth in score:** reported per width and per shape, never
  pooled. Under askeladd's E42 census (`f{7,8} = 0.1225`) the width term is
  **+0.789 % … +1.228 %** across shape mixtures, falling to **+0.386 … +0.601 %**
  if `f` is recomputed on the two prompts that set the median — which beagle's
  5.533 mean draft width suggests it should be. The ceiling term is adverse and
  bounded at `|dScore| ≤ 0.1186 %`.
- **Why it is still not bankable:** the outputs are not bit-identical to base at
  `M ∈ {7,8}`. Until a fixed-window exact decode against the golden passes, the
  speed number is not convertible into a submission.
- **Nothing was shipped.** Both scored files are reverted to their base blobs in
  the final commit; the shipped-surface diff against `9fe0dc5d` is empty. The
  measured candidate is preserved as `e66ddd0` on this branch.

## Suggested follow-ups (not implemented)

1. **Fixed-window exact decode against the public golden, on the `e66ddd0`
   candidate.** This is the only thing standing between a confirmed +11.4 % and a
   submittable candidate. It should be run before any further tuning, because a
   failure here retires the whole mechanism and a pass converts it immediately.
2. **Recompute `f` on the two prompts that set the median.** askeladd's E42
   census already collapsed the corpus-wide sensitivity table to
   `f{7,8} = 0.1225`, which is why this result now quotes a score range instead
   of `× f`. What remains open is that the score is the mean of the 4th and 5th
   order statistics over eight prompts, not a corpus mean, and beagle's 5.533
   mean draft width is well below the corpus 7.269. A per-prompt histogram
   restricted to beagle and medicine would move this result by roughly a factor
   of two, in the unfavourable direction, and is now the single largest
   uncertainty in it. This is askeladd's E48 / PR 52 territory and I did not
   touch it. The second-largest is the shape mixture `s(shape | M)`, which is a
   1.56× spread and would fall out of the same instrumentation almost free.
3. **Reconsider `M = 9`.** It needs a second 8-row tile, so it was excluded here,
   but the base cost at M=9 is the highest on the curve (202.84 / 647.25 µs) and
   r1 measured the two-tile candidate at −10.37 % / −11.66 % there. A two-tile
   variant tuned for M=9 specifically is a distinct, separately testable
   question.
4. **Repair the base's stale twin.** One regeneration of
   `mlx-generated/quantized.cpp` clears `twin_audit.py` on `9fe0dc5d`. Worth
   doing as its own small change so the gate is green and can actually detect
   future drift; it also removes a runtime-effective comment that contradicts
   the header about a hot width.
5. **Explain the untouched-width scatter.** The A/A control proves the design is
   clean on identical binaries, so the ±0.7 % mixed-sign scatter between
   different binaries is real and unexplained. A cheap test: build a candidate
   that changes *only* the register ceiling (e.g. a dummy instantiation that
   raises or lowers the max) with no cell changes at all, and see whether
   untouched widths move uniformly. That would separate occupancy from code
   layout and would make every future ceiling claim in this campaign
   interpretable — including E40's price, which the ceiling term depends on.
6. **`uint4` batching of A-fragment loads.** Still deliberately not implemented,
   for the reasons in r1: it saves ~12 % of instructions, invalidates the Gate A
   numbers, and adds a second path. The flat-tile result says this kernel is not
   instruction-bound at these widths, so the expected gain is small.

---
---

# r1 report, retained verbatim below


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
