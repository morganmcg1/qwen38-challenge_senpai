# E46 — is the 20.291 ms step a weight STREAM or a GROUP WIDTH?

Assignment `qwen38-r1-e46-stream-vs-groupwidth-fixed-m`, PR #51, revision `r1`,
branch `qwen-thorfinn/stream-vs-groupwidth-fixed-m`, base
`01f69e18f3878c9565fee479581581d85cf481ce`.

Host Apple M4 Pro, 48 GiB. **No result here is gate-qualified**: this host's real
40 °C cool gate is unreachable (floor ≈ 43.3 °C), so every run records
`cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`. These
are directional causal numbers inside one counterbalanced session, never a
ranked score.

## Question

E41 fitted `T(M) = 16.432 + 20.291·streams(M) + 11.798·M` on the NA≤5 table at
`04ad6bf1`. That table has exactly **one** stream boundary, at 5→6, and at that
boundary the stream count and the widest group's row count move **together**
(`<T,5,5>` → `<T,6,3>` is 1→2 streams *and* 5 rows → 3 rows). So `b = 20.291`
was carried by a single contrast and the mechanism was not identified. Three
readings survived E41:

| hypothesis | claim |
|---|---|
| `H_streams` | `T` is set by `ceil(M/IPG)`, the number of threadgroups that each re-read the whole weight tile |
| `H_groupwidth` | `T` is set by the widest group's row count / group balance |
| `H_M6breakpoint` | E43's fitted `36.278·[M≥6]` indicator — a property of the width itself |

E46 separates them **twice at fixed M**, where the `a·M` term cancels exactly,
and re-measures the width curve on the **shipped NA≤4 table**, whose stream
vector `[1,1,2,2,2,2,3]` moves the boundaries to 4→5 and 8→9 and removes the
5→6 boundary entirely.

| contrast | edit | streams | group width | `H_streams` | `H_groupwidth` | `H_M6breakpoint` |
|---|---|---|---|---|---|---|
| **A** | M=6, IPG 3→4 | 2 → 2 | 3+3 → 4+2 | Δ = 0 | Δ > 0 | Δ = 0 |
| **B** | M=8, IPG 4→3 | 2 → 3 | 4+4 → 3+3+2 | Δ = +20.291 ms | Δ < 0 | Δ = 0 |

Step 2 falsifier: on the shipped table, `argmax d1 ∈ {4→5, 8→9}` under
`H_streams` versus `5→6` under `H_M6breakpoint`.

Everything above was committed at `9169403` (`research/e46_prereg.py`,
`research/e46-prereg.json`) **before the first GPU second**, and before the
prior art in §4 was discovered.

## Step 0 — the two dispatch tables, verbatim

`senpai/verify-kernel-table.sh table 04ad6bf1 01f69e18`:

```
04ad6bf: IPG 3 4 5 3 4 4 5   streams 1 1 1 2 2 2 2   boundaries 5->6    NA ceiling 5   twin==hdr yes
01f69e1: IPG 3 4 3 3 4 4 3   streams 1 1 2 2 2 2 3   boundaries 4->5 8->9  NA ceiling 4  twin==hdr yes
PASS kernel-table-gate
```

The two tables genuinely disagree about where the boundaries sit, which is what
makes step 2 a test rather than a re-description.

## Step 1 — compile-only register census, before any GPU time

Stop rule 3 says: hard stop if any arm's measured kernel-wide register max
exceeds 108. The campaign's 108 is the **max over the seven per-width cases**
`..._m<T,M,IPG,true>`, not a per-NA cell — reproduced exactly here, and set at
M=7, the only case carrying both an NA=4 group and an NA=3 tail.

| arm | edits | kernel-wide max | entry point | per-width regs, M=3..9 |
|---|---|---|---|---|
| `A1_m6_ipg3` (shipped) | none | **108** | 163 | 83 104 87 83 108 104 83 |
| `A2_m6_ipg4` | M=6 IPG 4 | **108** | 164 | 83 104 87 **108** 108 104 83 |
| `B1_m8_ipg4` (shipped) | none | **108** | 163 | 83 104 87 83 108 104 83 |
| `B2_m8_ipg3` | M=8 IPG 3 | **108** | 164 | 83 104 87 83 108 **87** 83 |
| `AB_timed_arm` | both | **108** | 165 | 83 104 87 108 108 87 83 |

`any_exceeds_ceiling=false`, `all_arms_equal=true`,
`maxTotalThreadsPerThreadgroup=1024` in every arm (saturated, so this is a
floor-check only). **Stop rule 3 does not fire.**

Honest caveat: the entry-point figure drifts 163 → 165 across arms. That number
comes from a textual scan of the AIR entry point and is a heuristic; the
kernel-wide max, which is the quantity the stop rule names, is identical at 108
in all five builds. I did not treat the entry-point drift as a stop-rule event.

Pipeline: `metal -O2 -S | metal-opt -passes=default<O3>`, recorded in
`research/e46-reg-census.json`.

## Design

Four runs, ABBA within one session:

```
research/run-qmv-curve.sh <tag> 01f69e18f3878c9565fee479581581d85cf481ce \
    --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock
```

order `e46-base-r1 → e46-arm-r1 → e46-arm-r2 → e46-base-r2`.

Both builds occupy mean sweep position 2.5, so **linear session drift cancels
exactly** in the arithmetic mean of each pair. Every arm sweeps the *same*
width list `1..9`, so each width sits at the same sweep position — and
therefore the same thermal position — in every run. Widths 1,2,3,4,5,7,9 are
byte-identical between the two builds and act as untreated controls.

`T(M) = Σ over the 8 scored shapes of calls_per_verify × seconds_per_call`, ms.
All eight shapes have `n ≥ 5120`, so all of them land in the ≥4096-wide m-table
tier and there is no tier blending to disentangle here. Total
`calls_per_verify` is 257, dominated by `mlp.*` (64 each) and
`linear_attn.*` (48 each).

Two measurement caveats worth stating up front:

- **`head.compact_draft_vocab` has `calls_per_verify = 0`**, so it contributes
  nothing to `T(M)`. It still appears in the per-shape sign test, which asks a
  different question — whether the *kernel* slowed on each shape individually —
  and is legitimate evidence there. It is not double-counted into the headline.
- **`nax_available: false` on this host** (Apple M4 Pro, `applegpu_g16s`). The
  ranked M5 runner has the `_nax` variants. What E46 measures is the *shape* of
  the dispatch table's cost function, which is a structural property of
  `ceil(M/IPG)`; the absolute millisecond levels should not be transferred to
  M5 without re-measurement.
- **M=1 is a warmup width, not a usable control.** It is measured first in every
  sweep and absorbs first-touch pipeline cost. Its mean-vs-min spread is 19.4 %
  in `base-r1` and 41.5 % in `arm-r1` — the latter needed a fresh 90 s Swift
  rebuild — against 3.3 % at M=6 and 1.4 % at M=8. The spread falls
  monotonically with M, which is the signature of a first-touch cost rather than
  anything the edit did. The registered `MDE(M)` rule prices M=1 out
  automatically, since the same-build replicate disagrees there by about as much
  as the arm does. Neither contrast width is JIT-contaminated.

Registered decision rules, from `research/e46_prereg.py`:

- `MDE(M) = max(|T_base1 − T_base2|, |T_arm1 − T_arm2|)`; a signed effect
  requires `|Δ(M)| > MDE(M)`. With n=2 the honest floor is the observed
  same-build spread, not a t-statistic.
- Contrast B is a *level*, and levels move between trees, so it is scored in
  bands around E41's coefficient: strict `[15.218, 25.364]`, lenient
  `[10.146, 30.437]`.
- Secondary, distribution-free: per-shape sign of Δ over the 8 scored shapes;
  8/8 is p = 0.0078 two-sided.

The arm edit is two template arguments per file, in the readable header **and**
its runtime-effective generated twin:

```
quantized.h + mlx-generated/quantized.cpp
  qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>  ->  <T, 6, 4, true>
  qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>  ->  <T, 8, 3, true>
```

Contrast B makes M=8 worse **on purpose**. This is scaffolding: the branch
returns to zero scored-surface diff versus `01f69e18` before submission.

## Results

W&B: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/9gc2wstc>
(9 tables). Runs `hdgpg0z7` and `me6l35bb` are `failed` — two aborted attempts
at the logging script itself, superseded by `9gc2wstc`. No timed data differs
between them; only the logger changed.

### Provenance

| leg | order | head | dirty | entry °C | exit °C |
|---|---|---|---|---|---|
| `e46-base-r1` | 1 | `9169403` | 0 | 43.297 | 69.710 |
| `e46-arm-r1` | 2 | `f11d4c9` | 0 | 43.266 | 68.257 |
| `e46-arm-r2` | 3 | `524d926` | 0 | 43.212 | 88.138 |
| `e46-base-r2` | 4 | `16cefd0d` | 0 | 43.251 | 89.494 |

Entry-temperature spread across the four arms: **0.09 °C**. ABBA ordering gives
both builds mean sweep position 2.5, so the arithmetic mean cancels linear
thermal drift exactly. `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` on every leg: this host's idle floor is
≈ 43.2 °C, so the real 40 °C gate is unreachable. Directional causal evidence
within this counterbalanced session only — never a gate-qualified or ranked
number.

The compiled header hashes take exactly two distinct values,
`75d45143…` for both base legs and `57cf15b7…` for both arm legs. Dispatch
readback confirmed every width in every build compiled as designed, and row-0
bitwise fidelity versus M=1 had **0 failures in all four legs**.

### Step 2 — the width curve on the shipped NA≤4 table (base legs only)

| M | streams | IPG | base-r1 | base-r2 | mean (ms) |
|---|---|---|---|---|---|
| 3 | 1 | 3 | 74.274 | 72.636 | 73.455 |
| 4 | 1 | 4 | 82.474 | 82.766 | 82.620 |
| 5 | 2 | 3 | 120.133 | 120.543 | 120.338 |
| 6 | 2 | 3 | 128.737 | 128.850 | 128.794 |
| 7 | 2 | 4 | 138.789 | 138.908 | 138.848 |
| 8 | 2 | 4 | 149.185 | 149.340 | 149.263 |
| 9 | 3 | 3 | 186.195 | 186.001 | 186.098 |

First differences, each with its own replication floor (the two base legs each
measure the whole curve, so their disagreement *is* that step's floor):

| step | measured d1 (ms) | floor (ms) | streams |
|---|---|---|---|
| 3→4 | 9.165 | 1.930 | 1→1 |
| **4→5** | **37.718** | 0.117 | **1→2** |
| 5→6 | 8.456 | 0.296 | 2→2 |
| 6→7 | 10.055 | 0.006 | 2→2 |
| 7→8 | 10.414 | 0.036 | 2→2 |
| **8→9** | **36.835** | 0.349 | **2→3** |

**`argmax(d1) = 4→5`** at 37.718 ms; runner-up **`8→9`** at 36.835 ms. Both are
stream boundaries. The 5→6 transition — the one `H_M6breakpoint` names — is the
**minimum** of `d1`, at 8.456 ms.

This is the literal trigger of pre-registered stop rule 1. I report it as the
number that falsifies `H_M6breakpoint` rather than as a reason to discard the
arm legs, because the fixed-M contrasts are what separate the two surviving
readings and they were already funded.

Refit on the shipped stream vector:

```text
T(M) = 16.757 + 27.532 * streams(M) + 9.624 * M      max|resid| = 0.770 ms
```

Same data refit on an `[M>=6]` indicator instead:

```text
T(M) = 13.254 - 10.410 * [M>=6] + 19.721 * M         max|resid| = 11.348 ms
```

The stream regressor fits **14.7× tighter**, and the indicator's coefficient
comes out *negative* — the M≥6 story has to make the model faster past M=6 to
absorb a curve whose real steps are elsewhere. For reference E41 fitted
`b = 20.291 ms/stream` on the NA≤5 table; the shipped NA≤4 table gives
`b = 27.532 ms/stream`.

### Step 3 — fixed-M contrasts

| M | base (ms) | arm (ms) | Δ (ms) | MDE (ms) | \|Δ\|>MDE | role |
|---|---|---|---|---|---|---|
| 1 | 59.687 | 63.515 | +3.829 | 5.683 | no | control |
| 2 | 64.266 | 65.205 | +0.939 | 0.112 | YES | control |
| 3 | 73.455 | 73.071 | −0.384 | 1.638 | no | control |
| 4 | 82.620 | 82.872 | +0.252 | 0.292 | no | control |
| 5 | 120.338 | 120.484 | +0.146 | 0.409 | no | control |
| **6** | **128.794** | **129.057** | **+0.263** | **0.426** | **no** | **contrast A** |
| 7 | 138.848 | 138.673 | −0.176 | 0.119 | YES | control |
| **8** | **149.263** | **177.210** | **+27.947** | **0.155** | **YES** | **contrast B** |
| 9 | 186.098 | 185.603 | −0.495 | 0.194 | YES | control |

`MDE(M) = max(|base1−base2|, |arm1−arm2|)` as registered.

Three untreated controls (M=2, 7, 9) exceed their own replicate floor, so the
floor alone is not a sufficient bar. The **worst untreated control moves
+3.829 ms (+6.41 %) at M=1**, and every width other than 6 and 8 compiled to
byte-identical code across the two builds — so that 3.829 ms is this session's
own noise on unchanged code, and it is the honest bar a contrast must clear.

**Contrast A (M=6, IPG 3→4, streams 2→2, widest group 3→4 rows):
Δ = +0.263 ms.** Below its MDE of 0.426 ms *and* **14× smaller than the worst
untreated control**. `H_streams` predicts exactly 0 here → consistent.
`H_groupwidth` predicts a positive cost from widening the group → inconsistent.

**Contrast B (M=8, IPG 4→3, streams 2→3, widest group 4→3 rows):
Δ = +27.947 ms (+18.72 %).** 180× its MDE and **7.3× the worst untreated
control**. `H_streams` predicts a positive step; it lands **outside** the strict
band `[15.218, 25.364]` but **inside** the lenient band `[10.146, 30.437]`.
`H_groupwidth` predicts a *negative* step from narrowing the group →
inconsistent. `H_M6breakpoint` predicts 0 at fixed M → inconsistent.

The strict-band miss is itself informative and I do not want to paper over it:
the strict band was built from E41's `b = 20.291 ms/stream` on the **NA≤5**
table, whereas the refit above gives `b = 27.532 ms/stream` on the **shipped
NA≤4** table. Contrast B measured `+27.947`, which is within 1.5 % of that
refitted per-stream coefficient. So the one registered miss is explained by the
band being imported from a different table, not by the mechanism.

### Per-shape sign test (8 scored shapes, distribution-free)

- **Contrast A: 4/8 shapes slower, two-sided p = 1.000.** Largest single move is
  `full_attn.o_proj` at +2.41 %; the two biggest shapes both move *negative*
  (`head.lm_head` −0.24 %, `head.compact_draft_vocab` −0.14 %). This is a coin
  flip, exactly what a true null looks like.
- **Contrast B: 8/8 shapes slower, two-sided p = 0.0078** (the floor for n=8).
  Range +14.83 % (`linear_attn.out_proj`) to +21.34 % (`head.lm_head`). The
  effect is uniform across every shape regardless of `k`, `n`, or weight bytes —
  which is what a whole-tile re-read predicts and what a register-pressure story
  does not.

`head.compact_draft_vocab` has `calls_per_verify = 0`, so it contributes
nothing to `T(M)`; it still appears in the sign test as an independent shape,
and it moves +21.12 % under B.

### JIT-leak check

The arm introduces two instantiations the base never compiles, so a
first-call compile could masquerade as a cost. Mean-vs-min spread at the two
treated widths is small and *lower in the arm than the base* at M=8
(base-r1 3.57 %, arm-r1 1.35 %), and at M=6 it is 1.46–3.67 % across all four
legs. Both contrasts sit in the settled regime.

The M=1 control's +11.48 %-looking behaviour in the raw r1 pair is the same
warmup artifact: mean-vs-min spread there is 19.4 % (base-r1) and 41.5 %
(arm-r1, which followed a fresh 90 s Swift rebuild), decaying monotonically to
3.3 % by M=6 and 1.4 % by M=8. Averaged over the ABBA block M=1 falls to
+3.829 ms, still the noisiest cell on the curve, which is why it is used as the
control bar rather than deleted.

### Independent prior replication (E27 `7b5183d`, different base tree, n=1)

| M | E27 Δ (ms) | E27 % | E46 Δ (ms) | E46 % | role |
|---|---|---|---|---|---|
| 4 | +33.105 | +39.83 | +0.252 | +0.30 | E27-only cell (streams 1→2) |
| 6 | −0.016 | −0.01 | +0.263 | +0.20 | contrast A |
| 7 | +0.035 | +0.02 | −0.176 | −0.13 | control |
| 8 | +28.405 | +19.02 | +27.947 | +18.72 | contrast B |

Contrast A replicates as a **null** and contrast B replicates as a **positive
step within 0.30 pp**, across two different campaign bases, two different
builds, and two different sweep designs. E27's base `d1` also has the same
`argmax 4→5` and the same minimum at `5→6`.

### Verdict

| hypothesis | registered predictions held |
|---|---|
| **`H_streams`** | **3/3** — argmax at a stream boundary; A null; B positive in the lenient band |
| `H_groupwidth` | 0/2 — A positive where it needs positive-from-widening; B positive where it needs negative |
| `H_M6breakpoint` | 1/3 — argmax is 4→5 not 5→6; B is not null at fixed M |

**`H_streams` is the surviving hypothesis, and the mechanism should now be
called the weight-stream count `ceil(M/IPG)` — the number of threadgroups that
each re-read the entire weight tile — not a group-width register cliff and not
an M≥6 breakpoint.**

Practically: **`M` is not the cost driver, `ceil(M/IPG)` is.** Widths that share
a stream count cost within ~10 ms of each other (M=5→8 spans 28.9 ms across
three steps); crossing a stream boundary costs ~27.5 ms in one step. Any future
schedule work should treat draft depth as free *within* a stream band and
expensive only at the boundaries — on this table, at M=5 and M=9.

## Prior art discovered mid-experiment

See §4 of the PR thread. Commit `7b5183d` ("E27 probe arm: IPG falsification
(M4->2, M6->4, M8->3)") had already run both contrasts on the shipped table at
base `d7619a7f`, with artifacts still on disk. Its evidence points the same way,
and E46 is therefore a **pre-registered, position-matched, n=2 replication**
rather than a first observation. The pre-registration predates the discovery
(`9169403` is earlier in the branch history), so the bands and stop rules were
not fitted to it.

E27's three defects, all fixed here:

1. its base swept `widths=1..9` but its arm only `widths=4,6,7,8`, over a
   40.9 → 85.2 °C ramp, so every width sat at a different thermal position in
   the two runs;
2. n = 1, so no replication floor under its −0.016 ms null;
3. a different tree, and no pre-registration.

## Repository defects found (not touched — outside E46's registered edit)

**(a) The `case 8` comment contradicts the `case 8` code.** `quantized.h` says
`// 3+3+2, not 4+4 ... a register cliff, not work scaling` with
`Receipts: 85d5bca3 2.91143, yzxoi 2.92675`, directly above a dispatch to
`<T, 8, 4, true>`, which is 4+4. `git log -S` puts the flip to IPG=4 at
`dccba74 Validate submission 72ce82dc`; the comment survived from the earlier
IPG=3 tree. Making the code match its own comment is exactly contrast B, and it
costs ~19 %. The comment's "register cliff" reasoning is also the group-width
story this experiment falsifies.

**(b) `python3 research/twin_audit.py` is already RED on the campaign base** —
`TWIN AUDIT FAILED: 1/29`. Header and generated twin dispatch **identical
code** but carry **different comments** at `case 8`; the twin's is the correct
one ("4+4: two weight streams, receipted ... scored 3.195804751396457"). Proved
pre-existing by stashing the arm edit and re-running the audit at clean
`9169403`: same `1/29`, same comment-only diff. `9169403` adds only `research/`
files, so the failure is inherited from `01f69e18` itself. A one-line deletion
of the stale header comment would turn the gate green and remove the trap.

## Reproduction

```bash
git checkout 9169403 && python3 research/e46_prereg.py     # the registered predictions
python3 research/e46_reg_census.py                          # the compile-only gate
research/run-qmv-curve.sh e46-base-r1 01f69e18f3878c9565fee479581581d85cf481ce \
    --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock
git checkout f11d4c9                                        # the arm
research/run-qmv-curve.sh e46-arm-r1 ... ; research/run-qmv-curve.sh e46-arm-r2 ...
git checkout <revert>                                       # back to the base surface
research/run-qmv-curve.sh e46-base-r2 ...
python3 research/e46_analyze.py --base1 e46-base-r1 --arm1 e46-arm-r1 \
    --arm2 e46-arm-r2 --base2 e46-base-r2 \
    --e27-base e27-base-r1 --e27-arm e27-ipg-falsify-r1 \
    --json-out research/e46-artifacts/e46-metrics.json
python3 research/e46_wandb_log.py research/e46-artifacts/e46-metrics.json
```
