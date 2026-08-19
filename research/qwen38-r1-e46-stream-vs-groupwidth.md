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

<!-- FILLED FROM research/e46-artifacts/e46-metrics.json AFTER ALL FOUR RUNS -->

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
