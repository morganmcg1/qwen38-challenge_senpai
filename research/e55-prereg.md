# E55 pre-registration — Risk 1 gate PASSED, plus two blockers for the advisor

Written **before any GPU timing second**. `post_assignment_comment` returned HTTP 403
at 2026-08-19T16:2xZ (the same advisor-token 403 recorded at `981e69a`), so this file
is the durable route. It is committed on the assignment branch and is retried through
the typed tool.

**Retry log.** A third `post_assignment_comment` attempt also returned
`GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/57 returned HTTP 403`. The
failure is on the tool's own **read** of the PR, before any mutation. No GitHub
credential is exposed to the student shell (`GET /rate_limit` returns 401), so this is
inside the typed tool's executor and cannot be diagnosed or worked around from here. It
was not worked around. Consequence to note: `submit_experiment_result` reaches GitHub
through the same boundary, so the terminal submission may hit the same 403. If it does,
this branch still carries the full result, because every arm is committed and pushed as
it completes.

- Assignment: `qwen38-r1-e55-compose-m9-two-stream-on-shipped-table`, PR #57, revision `r1`
- Base: `a35bb006fd47785dc916241df63ec8780bda8e5c`
- Instruments commit: `5b421f9`
- Host: **Apple M4 Pro** — not the ranked M5 (`m5-qwen38-27b-mtp`)

---

## 🟢 Risk 1 — `vec<float,5>` compiles, and all five lanes are verified

`vec<float,5>` compiles with **zero diagnostics** (`xcrun metal -std=metal3.1 -O2`).
It is not a padding-free 5-lane type:

| type | `sizeof` | `alignof` |
|---|---|---|
| `vec<float,2>` | 8 | 8 |
| `vec<float,3>` | 16 | 16 |
| `vec<float,4>` | 16 | 16 |
| **`vec<float,5>`** | **32** | **32** |

**`vec<float,5>` pads to 8 lanes.** An `NA=5` accumulator costs the vector-register
footprint of `NA=8` while carrying 5 rows: `acc[rows_per_simd]` is 128 bytes at `NA=5`
against 64 bytes at `NA=4` — 2× the vector bytes for 1.25× the rows. This, and not a
missing type, is the likely reason the `static_assert` bound was 4. It is a cost fact
and it feeds the advisor's Risk 3.

### Lane fidelity, GPU-vs-GPU and bitwise

A CPU reference is not the primary check, because Metal FP contraction would confound
it. The primary check uses the property the brief licenses: the `NA` lanes carry
independent input rows and `simd_sum` reduces along K *within* a row, so **lane `m` of
an `NA=5` run must equal the same row computed by a shipped narrower `NA` on the same
data, bit for bit.**

| check | reference | max_ulp | verdict |
|---|---|---|---|
| lane 0 | `NA=3`, `first_m=0` | 0 | EXACT |
| lane 1 | `NA=3`, `first_m=0` | 0 | EXACT |
| lane 2 | `NA=3`, `first_m=0` | 0 | EXACT |
| lane 3 | `NA=4`, `first_m=0` | 0 | EXACT |
| **lane 4** | `NA=2`, `first_m=3` | **0** | **EXACT** |
| reference cross-check | `NA=3` vs `NA=4`, rows 0–2 | 0 | consistent |

Also verified: indexed write `t[i]=v` touches only lane `i` (bleed mask 0), lane-wise
arithmetic stays lane-local (fault mask 0), and all five lane outputs are **distinct**,
so "EXACT" cannot hide a lane collapse.

### Positive controls — the instrument can fail (ledger 178(E))

| control | caught | offending lanes |
|---|---|---|
| swap lanes 0 and 4 | yes | 0, 4 (rel 2.4, 5.4) |
| zero lane 4 activations | yes | 4 (rel 1.0), **and lane 1 at 8 ulp** |
| leak lane 3 into lane 4 | yes | 4 (rel 4.5) |

🟡 **Caveat from the second control.** Zeroing lane 4 also moved **lane 1 by 8 ulp
(rel 5.6e-7)**, which it should not do by construction. The faithful `NA=5` build is
0 ulp on every lane, so this is not a candidate defect — but `NA=5` codegen is **not
automatically bit-stable under source perturbation**. The full 512-token exactness
check is therefore not a formality, and any `M <= 9` delta is a hard stop.

Reproduce: `research/e55_vec5_probe.metal`, `research/e55_vec5_check.swift`.

---

## The candidate — code-only, zero byte growth

```
case 9:
-  qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>(
+  qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>(

-  static_assert(NA >= 2 && NA <= 4, "wide multi-row QMV supports NA in [2, 4]");
+  static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");
```

in **both** the readable header and the runtime-effective JIT twin.
`research/e55_diff_scope.py` asserts, and is re-run before every arm:

- both twins carry both edits **identically**;
- **every other dispatch cell is unchanged**, including `case 5` and `case 8`;
- **JIT source byte delta = 0** (99358 -> 99358); header byte delta = 0.

No comment was added to either file. `research/twin_audit.py:165` records that
`mlx-generated/quantized.cpp` is Metal source compiled at runtime and that "JIT cost is
inside the timed window, so comment lines here are not provably free". A byte-neutral
diff removes that confound from the measurement.

`setFastMathEnabled(false)` confirmed at `device.cpp:631`.

---

## The padding fact has a shared-kernel consequence, checked before timing

The `M` switch sits **inside one kernel**, so every dispatched width shares that
kernel's register allocation. `vec<float,5>` padding therefore does not price only
`M=9`: `acc[4]` doubling from 64 to 128 vector bytes could raise register pressure and
**slow every width from 2 to 9**, which is the concrete mechanism behind the
"composed cell is slower" branch below.

`research/e55_occupancy.sh` reads `maxTotalThreadsPerThreadgroup` for the shared
`affine_qmv_fast` pipeline on both arms. It normalises the header to `base` first and
then to the candidate, so the answer never depends on entry state, and it needs a Metal
device but **not** the model, so it runs before the expensive arms. E49 Arm 2 bounded
the harm from adding registers to the shipped table at `|dScore| <= 0.0876 %`, but never
measured the composed `NA=5` body's own allocation. An unchanged ceiling does not prove
zero spill; a moved ceiling does prove the effect is not a pure cell substitution, and
would invalidate the sensitivity constant.

The script edits only the readable header and restores it through an `EXIT` trap. A hard
kill would leave the header on `m9two` while the twin stays on `base`; the arm-independent
`research/e55_twin_gate.py` catches exactly that non-comment drift before any arm times.

## Pre-registered predictions

`psi_mtp = 0.693391` (E48), M=9 cell win `12.255 %` (E49 Arm 1).
Sensitivity: **8.49751 % of MTP leg per unit `f9`**.

| hypothesis | `f9` | predicted MTP leg | predicted local `raw_p` | x MTP-leg null |
|---|---|---|---|---|
| E48 (mine) | 21.630 % | **-1.838 %** | +1.838 % | 37.0x |
| edward E53, upper | 8.900 % | **-0.756 %** | +0.756 % | 15.2x |
| edward E53, lower | 4.600 % | **-0.391 %** | +0.391 % | 7.9x |

**Serial leg: 0 %.** The falsifier. The `out_vec_size >= 4096` switch has **no
`case 1`** — `case 2` is its first arm — so `M=1` falls to `default: break` and reaches
`qmv_fast_impl` untouched. A serial-leg move beyond its null means the dispatch story
is wrong.

Null floors, from E48's byte-identical `base2` rebuild: **`raw_p` 0.0629 %**,
**MTP leg +0.0497 %**, serial leg -0.0133 %. Per-leg `raw_p` spread on `base`: 0.105 %.

### Which result selects which mixture

Report `f9_implied = |dMTP %| / 8.49751 %`.

| measured dMTP leg | `f9_implied` | selects |
|---|---|---|
| <= -1.30 % | >= 15.3 % | **E48 mixture** |
| -1.30 % .. -1.05 % | 12.4-15.3 % | between; nearer E48 |
| -1.05 % .. -0.50 % | 5.9-12.4 % | **edward's E53 interval** |
| -0.50 % .. -0.30 % | 3.5-5.9 % | edward's lower bound or below |
| > -0.149 % (3x null) | < 1.8 % | **neither** — the isolated win did not transfer |

**Consistent with neither model**, stated in advance: a *positive* dMTP beyond the null
(the composed cell is slower), or `f9_implied` below 3.5 %, or dMTP more negative than
-2.2 % (`f9_implied` > 26 %). A positive dMTP selects answer 2 of Risk 3 — same family,
and the isolated single-body build was the artefact.

### Stop rules

1. `vec<float,5>` fails to compile, or any lane check fails -> stop. **Cleared: PASS.**
2. dMTP within 3x the null floor -> the win did not transfer. Stop and report.
3. dMTP <= -1 % with `all_tokens_matched` true -> go to the promotion chain.
4. Serial leg moves beyond its null -> stop; the dispatch model is wrong.
5. Any bitwise delta at **M <= 9** -> hard stop.

### Arms

`base` -> `m9two` -> `base2`, 2 legs each, 512-token `--local-iterate`, one fixture,
the **declared** head, `MLXFAST_LOCAL_RUN_LOCK_DIR=/tmp/mlxfast-shared`.

🟡 **Declared deviation from strict ABBA.** Each arm change needs a full two-root
rebuild, so leg-level ABBA is not affordable in one session. This uses E48's bracket
design instead: the candidate sits **between two byte-identical base arms**, which
controls monotone drift to first order and yields the null in the same session. Entry
and exit temperature are reported per arm with the entry-temperature spread, and
`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`official_or_ranked_score=false` are preserved verbatim. Primary is **absolute
candidate seconds per token**; the local ratio is secondary.

---

## 🔴 Blocker 1 — the promotion chain's twin audit cannot pass, and it is not this diff

`research/twin_audit.py` is step 4 of the promotion chain. It **fails** on this
candidate, and it would fail on *any* candidate that edits this section:

```
STALE quantized: section drift in mlx/backend/metal/kernels/quantized.h
@@ -1955,5 +1955,15 @@
         case 8:
-          // 4+4: two weight streams, receipted on this benchmark (scored
...
TWIN AUDIT FAILED: 1/29 twin(s)
```

The reported block is **entirely the pre-existing `case 8` comment divergence**.
Neither edit appears on either side of it, which itself proves both files carry them
identically; `research/e55_diff_scope.py` asserts the drift block holds **0 non-comment
lines**.

Mechanism: on base `a35bb006` the audit **WAIVES** this section (`3005 non-comment
lines identical`). The waiver is pinned to **two whole-body sha256 digests**. A
4-character code edit changes both bodies, the pins miss, and the known divergence
reds. That is correct fail-closed behaviour, with one consequence:

**One pinned digest pair cannot cover both arms of an A/B on this section.** Pinning
the candidate's bodies would red `base` and `base2`; leaving base pinned reds the
candidate.

`KNOWN_COMMENT_DIVERGENCES` was **not** touched and the waiver was **not** re-pinned;
`twin_audit.py:241` calls re-pinning "laundering" and the table encodes an advisor
decision.

Instead: `research/e55_twin_gate.py`, pinned to **the divergence** rather than the
body, so it holds for `base`, `m9two` and `base2` alike. It is strictly narrower in
what it tolerates and keeps the audit's own structural guard:

1. every non-comment line byte-identical — `twin_audit.comment_only_waiver`'s own
   guard, and the property that actually proves the JIT string and the readable header
   compile to the same kernel;
2. every differing line is a whole-line comment;
3. the differing comment lines hash to a pinned digest, so **only** the known `case 8`
   block is tolerated.

Arm-independence verified: the two comment digests are **identical on base and on the
candidate** (`ada98491...` / `5f747f13...`). Falsifiability verified by
`research/e55_twin_gate_negative_control.py`:

| injected fault | caught |
|---|---|
| twin only moves `case 7` to NA=5 | yes, NON-COMMENT drift |
| header only relaxes the NA bound further | yes, NON-COMMENT drift |
| a new unpinned comment in the twin only | yes, does not match the pinned block |
| tree restored, gate passes again | yes |

`research/e42-run.sh` now reads `E42_TWIN_GATE`, **default unchanged**
(`research/twin_audit.py`).

**Advisor decision needed.** `twin_audit.py` stays the promotion gate, so the `case 8`
waiver must be re-expressed before this candidate can pass the chain. Recommendation:
pin it to the divergence as `research/e55_twin_gate.py` does, because whole-body
pinning reds the audit for every experiment on the campaign's hottest file.

## 🔴 Blocker 2 — this candidate necessarily breaks the E27-revert byte-identity property

`twin_audit.py:158-167` records that the E27 revert deliberately restored **both** files
to frontier byte-identity, that the decision "is worth 0.3321 % of score", and that
`research/scored-surface-gate.sh` marks them FRONTIER-TAKEN and **asserts** that
byte-identity. It warns that any edit converts the overlay from "writes back exactly
what is already there" into "**REPLACES the tip's copy of the hottest file in the
competition**", the mechanism by which a submission silently reverts
organizer-accepted work.

**Any candidate that changes `case 9` breaks that property by construction.** This is
inherent to the assignment. It is flagged before measurement, not after. It bears on
the submission decision and is above student scope: this base is `a35bb006` while the
live promoted frontier is `3.24985583421771` (`59b321e`), so if the tip's copy of
`quantized.h` carries work this base lacks, submitting our copy would revert it.

---

## 🟡 Provisioning from zero

The **15:02Z retag wiped this workspace**, as it wiped thorfinn's: no model cache
(`~/.cache/mlxfast/` empty), no `.build/`, no `.build-worker/`.
`research/e55_setup.sh` provisions in three steps, and the third matters:
`setup-qwen-mtp.sh` only ever provisions the **organizer-pinned** head, while
`mtp-head.manifest.json` **declares**
`hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827` (427,742,600 bytes), which is
the head the ranked **candidate** leg executes. `research/fetch-declared-head.sh`
provides it; without that step every timed candidate leg runs the wrong head.

The branch is pushed after every arm, per the advisor's warning about the retag.

---

# ADDENDUM 1 — registered 2026-08-19T17:10Z, after the `base` arm, before any candidate arm

The `base` arm measured the dispatched verify-width histogram. That histogram is a
property of **the fixture and the base build only**: the candidate changes which kernel
serves M=9, never the schedule, so nothing here is fitted to the effect under test.
This addendum is registered **before the `m9two` arm runs**, so the new prediction is a
genuine prediction and not a post-hoc correction.

## The local mixture is far deeper than the ranked mixture

Measured on `base`, 78 rounds, 512 decode tokens:

| width M | rounds | weight streams | stream-cost |
|---|---|---|---|
| 2 | 1 | 1 | 1 |
| 4 | 5 | 1 | 5 |
| 5 | 5 | 2 | 10 |
| 6 | 23 | 2 | 46 |
| 7 | 4 | 2 | 8 |
| 8 | 6 | 2 | 12 |
| **9** | **34** | **3** | **102** |
| total | 78 | — | **184** |

Streams are read from the shipped `out_vec_size >= 4096` switch at
`mlx-generated/quantized.cpp:1934-1980`: each working group re-reads the whole weight
matrix, so streams `= ceil(M / IPG)`. QMV is bandwidth bound at these shapes, so cost
per round is taken proportional to streams.

```
local f9 = 102 / 184 = 55.435 %
```

`accepted_draft_rate = 0.8875` and `effective_mean_draft_len = 6.27` on this fixture.
The pre-registered `f9` values are ranked-mixture quantities: 21.630 % from E48's
score-weighted share, and 4.6-8.9 % from edward's E53 board telemetry. **This one
public English long-copy prompt accepts far more drafts than the board average, so it
dispatches a much deeper width mix, and its local `f9` is 2.6x to 12x the ranked
estimates.**

## Revised prediction for the local instrument

```
predicted local MTP-leg change = -psi_mtp x 12.255 % x local_f9
                               = -8.49751 % x 0.55435
                               = -4.711 %
```

- **Registered prediction: MTP leg −4.711 %**, which is **94.8x** the +0.0497 % MTP-leg
  null floor.
- Serial-leg prediction is **unchanged at 0 %**. It remains the best falsifier.
- Guard band unchanged at `3 x null = 0.1491 %`.

## What this does and does not settle

🟢 **It makes the composition question sharper.** The two competing answers to Risk 3
are now separated by roughly 4.7 percentage points instead of 1.8:

| answer | mechanism | predicted local MTP leg |
|---|---|---|
| 1 — crossrow tier escapes PR #8's group-throughput collapse | the isolated −12.255 % cell win transfers | **≈ −4.71 %** |
| 2 — same family, isolated single-body build was the artefact | no transfer, or a slowdown | **≈ 0 %, or positive** |

🔴 **It does NOT settle the ranked `f9` disagreement, and I am withdrawing that
deliverable as stated.** The advisor's framing was that this instrument reads the
product directly and therefore selects between the E48 and E53 mixtures. That holds
only if the local and ranked width mixtures agree, and they demonstrably do not. The
local leg change prices **local** `f9 = 55.4 %`. Selecting a **ranked** `f9` needs the
ranked width histogram, which this fixture cannot supply.

The pre-registered three-row selection table is therefore **void as a ranked-mixture
selector**. I keep reporting `f9_implied` from the measured leg change, but it must be
read as *local* `f9`, and its agreement with 55.4 % tests the stream-cost model rather
than the board mixture.

🔴 **Consequence for the submission decision.** A local MTP-leg improvement of ~4.7 %
must **not** be priced as a ~4.7 % ranked gain. Under the same `psi_mtp` and cell win,
the ranked prize is `8.49751 % x ranked_f9`: **1.84 %** at E48's mixture, **0.76 %** at
edward's upper bound, **0.39 %** at his lower bound. The local number confirms the
mechanism; the ranked mixture sets the prize. I will state both and promise neither.

## Also confirmed on `base`

- `max_dispatched_width = 9`, and **no width outside the shipped table** — so the
  advisor's M=10 caveat does not arise on this fixture, and any M ≤ 9 bitwise delta
  remains a hard stop.
- `all_tokens_matched = true`, `row_ledger_closes = true`, 567 declared rows over 78
  rounds.
- Leg spread `sd = 0.119 %` on MTP decode seconds, consistent with the E48 null.
- `mtp_seconds_per_token = 0.03343756` against E48's `0.033438` — cross-session
  agreement to about `5e-7`, which is far finer than the effect under test.

## One protocol defect found and fixed before the timed comparison

The `base` arm recorded `thermal_before=unavailable`. `e42-run.sh` resolved macmon only
at `${HOME}/bin/macmon`, while it is installed at `/opt/homebrew/bin/macmon`, so
`sample_thermal` silently degraded to a string. `program.md` permits
`MLXFAST_LOCAL_COOL_GATE=0` **only** when entry and exit GPU temperature are recorded
for every arm, so that arm did not satisfy the ungated protocol even though its timings
were clean.

Fixed: macmon is resolved from `MLXFAST_MACMON_BIN`, then `${HOME}/bin`, then `PATH`,
and the driver now **refuses to time** when no GPU temperature can be read. Verified
reading `gpu_temp=45.09 C`. The thermal-less `base` arm is preserved under
`.mlxfast-private/e55/discarded/base-nothermal/` and **is not used in the comparison**;
`base` is re-run with temperatures so all three arms satisfy the protocol.
