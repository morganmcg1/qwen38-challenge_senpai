# E39 — Statistical power audit of the established-negatives list

- **Student:** qwen-alphonse
- **Assignment:** `qwen38-r1-e39-negatives-power-audit`, revision `r1`, PR #44
- **Base:** `senpai/qwen38-mtp-r1` @ `0491f9e54c4df28f3c69d8b114574fda366c5062`
- **Host:** AWS M4 Pro `Mac16,11`, 48 GiB (`hw.memsize = 51539607552`), 20 cores
- **GPU runs in this experiment: zero.** Analysis only, per the assignment constraint.
- **Diff:** confined to `research/`.
- **Tools:** `research/e39_mde.py` (`--self-test` passes 13 checks),
  `research/e39_residency_audit.py`.

---

## 0. Headline

Three findings, in the order they should change decisions.

1. 🔴🔴 **Entry 6 was never in conflict with the board.** The assignment's
   premise is that the only ranked-positive family contains two of our own
   negatives. For the residency half that is false: our negative is about the
   **with-headroom** arm, and the board positive is **zero-headroom** wiring.
   The competitor who filed the ranked-positive row is the same person who
   reports with-headroom as harmful. They agree. Entry 6 also has **n = 0** —
   never measured by this campaign, on hardware that cannot execute the code.

2. 🔴 **The instrument that reopened the list is itself under-powered.**
   E35's family join has MDE 0.401 % and needs n = 25 rows to reach the
   0.185 % bar. It has 5 rows and **3 distinct contrasts**. Power at its own
   refreshed observed effect is **0.337**. Applying E39's test to E39's
   premise fails it.

3. 🔴 **The board's +0.316 % does not survive.** On a refreshed corpus it is
   **+0.220 %** (|t| 1.54). Decomposed by contrast, the only cleanly
   identified measurement — three independent re-applications of the same
   donor diff — averages **+0.058 %**, roughly a third of the bar. The
   headline was carried by a single 5-mechanism bundle.

**Verdict counts over 23 entries** (22 assigned + 1 I am adding):
CLOSED **9**, UNDER-POWERED **7**, WRONG INSTRUMENT **6**, NO EVIDENCE EXISTS **1**.

---

## 1. Method

### 1.1 The MDE formula, stated once

For an estimator with standard error `se`, the minimum effect detectable at
80 % power and two-sided α = 0.05 is

```
MDE = (z_{0.975} + z_{0.80}) × se = 2.8016 × se
```

with `se` by design:

| design | se |
|---|---|
| paired / mean-of-n | `s / sqrt(n)` |
| two-sample, n per arm | `s × sqrt(2/n)` |
| single observation | `s` |
| slope over `n` levels | supplied directly as the regression se |

The normal multiplier assumes `s` is known. At the sample sizes this campaign
actually uses it is not, so every row also carries an **exact** MDE from the
noncentral-t power function on the real residual df. The inflation is not
cosmetic:

| n (paired) | df | exact / normal |
|---:|---:|---:|
| 2 | 1 | **5.83×** |
| 3 | 2 | 2.02× |
| 10 | 9 | 1.12× |

**Most campaign negatives sit at n = 2.** Quoting the normal MDE there
understates the true detection floor by nearly six-fold, so where an entry
rests on n = 2 I lead with the exact figure.

`research/e39_mde.py` implements the noncentral-t in the standard library (no
scipy). Its self-test reproduces the two numbers the assignment named —
σ_score = 0.0923 % and the E33 E2E instrument at ±0.30 % with n = 2 — and is
additionally validated against two external anchors so the engine is not just
self-consistent:

| check | got | want |
|---|---|---|
| Cohen, two-sample n=64/arm, d=0.5 → power | 0.80146 | 0.80 |
| G\*Power, paired n=2 → standardised MDE | 11.5499 | 11.55 |

### 1.2 The effect that would matter

- **Score claims:** 2σ_score = **0.185 %**.
- **Mechanism claims:** derived per entry from the mechanism's own plausible
  size, not assumed.

### 1.3 Instrument registry

`python3 research/e39_mde.py --audit`:

```
instrument            design      n     sd        MDE(norm)  MDE(exact)  own-sd
------------------------------------------------------------------------------
ranked_score          single      1     0.0923    0.2586     1.5076      True
local_e2e_leg         two_sample  2     0.2974    0.8332     1.6814      True
local_e2e_leg_e29     two_sample  2     0.8600    2.4094     4.8620      True
e29_ladder_slope      slope       4     0.0311    0.0871     0.1758      True
local_microbench      paired      3     0.3000    0.4852     0.9792      False
competitor_ranked_n1  two_sample  1     0.0923    0.3657     n/a         False
board_family_n5       single      5     0.3240    0.9077     1.2186      True
algebraic             single      1     0.0000    0 (exact)  0 (exact)   True
static_analysis       single      1     inf       undefined  undefined   False
hardware_gated_out    single      0     inf       undefined  undefined   False
```

**`own-sd = False` means the noise is borrowed** from the nearest comparable
run rather than measured by the run being judged. Per the evidence contract
that is stated in the row, not a footnote. Two entries in the registry carry
borrowed noise, and both are load-bearing for multiple audit rows.

🔴 **A caveat on the campaign's standing noise floor.** The 0.86 % "same-config
repeat" that `local_e2e_leg_e29` rests on comes from **one pair that is not
same-config**: D0 ran `trace=1` plus a `switch → Set.contains` refactor, N1 ran
`trace=0` pre-refactor (`.mlxfast-private/e29/runs/{D0,N1}/meta.txt`;
`e29-round-overhead-host-graph.md:82-84`). It is an upper bound contaminated by
a code change, and it is the only repeat-noise number the campaign owns for the
local E2E leg.

---

## 2. The audit table

Columns: entry, what was actually run, instrument, n, MDE, the effect that
would matter, verdict. Pre/post-E27 and head are called out where they change
the verdict.

**E27 anchor.** The assignment defines E27 by the fact that it *"changed the
M=5 cell by −20.1 %."* That is `21d98b7`, **2026-08-18 19:35:40 +0000**
("E27 results: NA=5 collapses the M=5 weight-stream cliff, 1.4520 → 1.1607";
the ratio change is −20.06 %). E27 opened earlier — `f0bb949` at 18:47:01 is
its static AIR probe and `0207de6` at 19:10:57 is work item 3 — but neither
carries the M=5 finding, so **19:35:40 is the boundary that matters for
transfer**. Using an opening commit as the anchor would move the cut 48 minutes
earlier and misclassify anything run in that window. E29's timed arms ran
18:48–19:30, i.e. squarely inside it: they are **PRE-E27**, independently
confirmed by `git merge-base --is-ancestor` against E29's base `d7619a7f`,
which does not contain E27's results.

| # | entry | what was run | instrument | n | MDE | effect that matters | verdict |
|---|---|---|---|---|---|---|---|
| 1 | NA=6 at rows=4 | static AIR register/occupancy inventory | static analysis | 0 | n/a — analytic wall | occupancy is discrete | **CLOSED** |
| 2a | rows_per_simd ≠ 4 (structural) | static SIMD-layout analysis | static analysis | 0 | n/a — analytic | layout is discrete | **CLOSED** |
| 2b | rows_per_simd ≠ 4 (E2E leg) | local `--local-iterate` leg | local_e2e_leg | 2 | **1.681 %** exact | 0.185 % | **UNDER-POWERED** |
| 3 | M=7 IPG 3→5 | static instructions-per-group count | static analysis | 0 | n/a — analytic | IPG is compile-time | **CLOSED** |
| 4 | qmm for M ≥ 4 | one un-replicated microbenchmark of **padding across `vector_limit`** | local_microbench (borrowed sd) | 1 | ≥ 0.979 % | **predicted +61 % ceiling** | 🔴 **WRONG INSTRUMENT** |
| 5 | wide target-top-2 reducer (−3.85 %) | competitor's prose note | competitor_ranked_n1 (borrowed sd) | 1 | 0.366 % | 0.185 % | **WRONG INSTRUMENT** |
| 6 | **wired limit with headroom** | **nothing** | hardware_gated_out | **0** | **undefined** | 0.185 % | 🔴 **WRONG INSTRUMENT** |
| 7 | E25 arm D | local fixture, replicated, σ on 2 dof | local_e2e_leg | 3 | 0.485 % | 0.185 % | 🔴 **WRONG INSTRUMENT** |
| 8 | argPartition top-32 | one timed leg, no repeat | local_e2e_leg (borrowed sd) | 1 | ≥ 1.681 % | 0.185 % | **UNDER-POWERED** |
| 9 | 2-bit draft readout | exactness audit (bit-width proof) | algebraic | — | 0 | correctness is binary | **CLOSED** |
| 10 | host round overhead | local leg vs declared noise floor | local_e2e_leg_e29 | 2 | **4.862 %** exact | 0.185 % | **UNDER-POWERED** |
| 11 | **command-buffer geometry (E31)** | **static audit, zero timed arms**; null borrowed from E29's ladder | e29_ladder_slope | 4 arms × 1 | **351.6 µs/boundary** exact | 27.6 µs/boundary | 🔴 **WRONG INSTRUMENT** |
| 12 | depth > 8 | closed-form cap read from source | algebraic | — | 0 | cap is a constant | **CLOSED** ⚠️ |
| 13 | replacement head artifact | one competitor ranked row | competitor_ranked_n1 (borrowed sd) | 1 | 0.366 % | 0.185 % | **UNDER-POWERED** |
| 14 | h 0.18 → 0.16 (−1.164 %) | one competitor ranked row | competitor_ranked_n1 (borrowed sd) | 1 | 0.366 % | 1.164 % observed | **CLOSED** |
| 15 | verify-tree FP32 reassociation | correctness boundary, confirmed 3 ways | algebraic | — | 0 | exactness is binary | **CLOSED** |
| 16 | values_per_thread on verify path | one competitor ranked row | competitor_ranked_n1 (borrowed sd) | 1 | 0.366 % | 0.185 % | **UNDER-POWERED** |
| 17 | prefill (exactly 0.000 %) | accounting identity | algebraic | — | 0 | structurally zero | **CLOSED** |
| 18 | KV-1024 warm | χ² fit; source says *"not proven dead"* | local_e2e_leg | 2 | 1.681 % | 0.185 % | **UNDER-POWERED** |
| 19 | E33 row-blocked M=6 sequential (1.0150) | purpose-built ABBA, pre-registered kill rule | local_e2e_leg | 2 | 0.833 % norm | **1.50 % observed** | **CLOSED** ⚠️ |
| 20 | beagle acceptance, either direction | competitor rows, regex-classified | competitor_ranked_n1 (borrowed sd) | 1 | 0.366 % | 0.185 % | **UNDER-POWERED** |
| 21 | warm coverage / shape gaps (E37 d3) | **E37 was never delivered** | — | — | — | — | 🔴 **NO EVIDENCE EXISTS** |
| 22 | depth constant re-pricing (E35) | 6-row re-pricing, regex-grouped | competitor_ranked_n1 | 6 | 0.149 % | 0.185 % | **CLOSED** ⚠️ |
| **23** | **shipped surface = "4 files, +117/−87"** | **advisor gate, never counted a 5th file** | config echo | — | — | exact file set | 🔴 **WRONG INSTRUMENT** |

### Notes on the ⚠️ rows

- **12** — the constant is `Sources/MLXFastCore/Constants.swift:331`, **not**
  `MLXFastModel`. `MLXFastCore` is **not in `editablePaths`**, so its exclusion
  is load-bearing: the cap is not merely unattractive to change, it is outside
  the submitted surface. The verdict stands but the *reason* in the ledger is
  wrong.
- **19** — the observed 1.50 % comfortably exceeds even the exact-t MDE, so the
  negative is real. But the campaign's framing of E33 as *"3.5× under-powered"*
  is arithmetically wrong: it compares thorfinn's **+0.088 %** prediction to a
  **±0.30 % 1σ resolution**, which is not an MDE. Against the true MDE the
  prediction was **9.5×** under-powered (normal) or **19.1×** (exact). The
  conclusion was right; the power statement understated the problem by ~3–5×.
- **22** — narrower than the list implies. The source insists that removing a
  *policy* constraint stays open (`research/within_head_cost.py:907-919`), and
  the n=6 group is classified by **regex over competitors' self-authored note
  titles** (`:641-644`). CLOSED for "a depth *constant* re-prices 0 of 6 wins";
  **structural depth work remains open.**

### "Was this measured under ranked memory geometry?" — asked for as a column

The advisor asked for this as a ninth column. The table is already eight wide,
so I answer it in grouped form instead; the content is the same.

**For every timed entry on the list, the answer is no.** Every local timing
artifact this campaign owns — E27's M-table, E33's per-width ladder and
per-shape attribution, edward's `S = 23.911 ms`, E29, E25, my own E30 — was
produced at **128 mebi-elements / 64 ops with residency off**, against the
ranked box's **512 / 50 with residency on**.

| entry class | measured under ranked geometry? | does that invalidate it? |
|---|---|---|
| 1, 2a, 3, 9, 12, 15, 17 (static / algebraic) | n/a — nothing was timed | **No.** Analytic results do not depend on dispatch batching. |
| 2b, 7, 8, 10, 18, 19 (local timed, paired ratio) | **No** | **Not automatically.** A *paired ratio* within one fixed geometry is still a valid contrast; both arms saw the same batching. |
| any absolute-millisecond claim | **No** | **Yes, for transfer.** No local absolute figure describes the ranked box. |
| 11 (command-buffer geometry) | **No, and fatally** | **Yes.** Here the geometry *is* the independent variable, so measuring it at the wrong setting is disqualifying, not merely non-transferable. |
| 13, 14, 16, 20, 22 (competitor ranked rows) | **Yes** — they ran on the ranked box | Geometry is not the issue; **n = 1** is. |

The honest summary is narrower than "everything is invalid": the paired-ratio
structure of most campaign work protects it. The systematic question mark falls
on (a) every absolute millisecond figure, and (b) any mechanism that plausibly
interacts with dispatch batching. That second set is not hypothetical — E33's
own finding is that **narrow-output, few-threadgroup, short-kernel shapes lose**
under row blocking, and short kernels are exactly the ones most sensitive to
command-buffer packing. E33's per-shape attribution therefore deserves a replay
under the Correction-1 recipe before it is relied on further; I did not have
the GPU budget to do that here and it is not on the re-test list because it is
a re-measurement of a *positive*, not a negative.

### The entry I am adding — #23, the shipped-surface gate

The assignment states the shipped surface is *"frozen at E27, 4 files,
+117/−87, and I re-verify it on every submission."* Against the true campaign
baseline `5273067` the shipped diff is **5 files, +229/−74**:

```
$ git diff --stat 5273067 HEAD -- Sources/ Vendor/
 Sources/MLXFastModel/Qwen36MTPBlockSession.swift   | 204 ++++++-----
 Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift |  32 ++++
 .../Libraries/MLXLLM/Models/Qwen35.swift           |  51 ++++--
 .../Source/Cmlx/mlx-generated/quantized.cpp        |   8 +-
 .../Cmlx/mlx/mlx/backend/metal/kernels/quantized.h |   8 +-
 5 files changed, 229 insertions(+), 74 deletions(-)
```

| file | diff |
|---|---|
| `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` | **+157 / −47** |
| `Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift` | **+32 / −0** ← never counted by the gate |
| `…/MLXLLM/Models/Qwen35.swift` | +32 / −19 |
| `…/mlx-generated/quantized.cpp` | +4 / −4 |
| `…/metal/kernels/quantized.h` | +4 / −4 |

A gate that reports a stable count while missing a file is a **config echo**,
not a measurement — the same failure mode as an under-powered null, and it is
verifying the wrong thing on every submission.

🔴 **Correction to my own first pass.** I originally reported this entry as
"5 files" while listing only four, having omitted the largest one
(`Qwen36MTPBlockSession.swift`, +157/−47), and I gave no total. I had also
reached for `5068eb8d`, which is **not a valid git object in this checkout**,
and reported that as a blocker instead of using the baseline `5273067` that was
supplied in the assignment feedback. The advisor's figures were right and mine
were incomplete; the verdict is unchanged but the arithmetic above is the
authoritative version. This is the same error class the audit is about — a
check that ran and was believed without anyone asking what it compared — so it
belongs in the record rather than in a silent edit.

---

## 3. 🔴 The residency + command-buffer case in depth

This is the deliverable that matters most, so it gets the full treatment. The
assignment asked four specific questions; each is answered under its own
heading, then I give the two source corrections that change the re-test recipe.

### 3.0 The finding that precedes all four questions

**The board does not contradict entry 6, because they describe opposite arms.**

SSHdotCodes' note on `3a995c2b` (accepted, 3.23222998733732) —
`.mlxfast-private/ranked-telemetry.json`:

> "The related M5-Max benchmark found that a naive limit **with spare capacity
> was harmful** because scored-window temporaries could fit in the wired set and
> cause commit traffic on insert/erase. **The successful mechanism was
> zero-headroom wiring.**"

Our negative: *"raising the wired limit **with headroom** is harmful."*
The board positive: **zero-headroom** wiring.

These are the same claim. The reopening premise for the residency half
dissolves on inspection of the note that generated the row.

And entry 6 has **n = 0** — no experiment, no PR, no results file, no tool, no
raw capture:

- `git log --all -S "wired limit with headroom"` → only ledger commits
  `cf80358` (item 81) and `5154d6d` (item 132).
- `git log --all -S "WIRED_ZH"` → only organizer *"Accept submission"* syncs
  (`86fb1f0`, `474c750`, `c0e34af`). **No campaign branch ever touched it.**
- `"zero-headroom"` appears nowhere in the ledger.

**Falsifiable prediction, stated and then tested:** if residency has never
executed here, no local artifact can contain its telemetry.

| grep across `research/` and `.mlxfast-private/` | files |
|---|---|
| `wired-zh` (the stderr line at `Qwen36MTPBlockSession.swift:272-276`) | **0** |
| `low-memory startup profile` | **10** |

Confirmed. The gate at `Qwen36MTPBlockSession.swift:225-226` is
`physicalMemory >= 96 GiB` against a 48 GiB box, with **only a kill switch**
(`DARKBLOOM_QWEN_MTP_WIRED_ZH != "0"`) and no enable override. claudiodekker's
Laguna port reached it via `DARKBLOOM_WIRED_ZH_MIN_PHYS_GB=32`, which our tree
does not have.

Ledger item 81 (`senpai/campaign-ledger.md:1340-1355`) recorded all of this
correctly, and its purpose was the **opposite of a closure** — it ordered the
mechanism *kept*: "we cannot measure it locally and must not delete it." Entry
6 is an artifact of the list being reconstructed from memory.

### 3.1 Winner's curse and multiplicity over 9 families

Selecting the max of 9 family means under the null:

| quantity | value |
|---|---|
| E[max of 9] on the % scale | **+1.265 %** |
| observed family mean (refreshed) | **+0.220 %** |
| median max-of-9 on the t-scale | **+1.85** |
| observed \|t\| | **1.54** |

The observed maximum is **below** what pure selection over 9 families produces
by chance. Selection alone explains it. The earlier −0.029 % winner's-curse
figure was computed over the **top 6 rows**, which is a different and much
weaker correction than **family** selection.

### 3.2 Confounding with the box — the serial legs are *not* ordinary

The assignment asked me to argue explicitly whether serial +0.071 rules a box
effect in or out. It rules it **in**.

Because `R = serial / mtp`, a **slow** serial leg *raises* the score, and E35's
`R'` deliberately retains that session factor (it is the organizers' own
normaliser). So a positive serial offset sitting beside a positive delta is the
signature of a box effect, not evidence against one.

All 83 head-matched rows: serial offset mean −0.0128 %, sd 0.1159 %.

| id | solver | role | delta | serial | (σ) | serial-corrected |
|---|---|---|---|---|---|---|
| 3a995c2b | SSHdotCodes | INTRODUCE | +0.771 | +0.049 | +0.53 | **+0.722** |
| 4f76de6e | alfranli123 | RESTORE | +0.103 | +0.092 | +0.91 | +0.011 |
| 11863aa9 | companygardener | RESTORE | +0.133 | +0.139 | +1.31 | −0.005 |
| 3ec77796 | xadenryan | RESTORE | −0.063 | −0.113 | −0.86 | +0.050 |
| 0cd0a6b4 | **ofou (crown)** | FORCE | +0.156 | **+0.187** | **+1.73** | **−0.032** |

Four of five offsets are positive and the family mean is **+1.62σ of the
mean-of-5**. Subtracting each row's own offset:

**serial-corrected mean +0.149 %, se 0.144, |t| 1.04 — below the 0.185 % bar.**

The crown is the sharpest case: it drew the slowest serial leg of the field and
is **−0.032 % once corrected**. Its own note (hypothesis H4) predicts a *tie* if
the environment was already 512 — and E31's static audit independently
concludes it was. **The crown is plausibly a no-op that won on a lucky serial
leg.**

### 3.3 Are the 5 rows independent? No — there are 3 contrasts, and the clean one is ~zero

Distinct solvers: 5 of 5. Repeated titles: none. Distinct commits: 5. By every
surface measure the rows look independent. They are not, because the unit that
matters is **which diff was applied to which parent**:

```
INTRODUCE n=1  mean +0.771 %  first appearance, bundled with 4 other mechanisms
          rows: 3a995c2b
RESTORE   n=3  mean +0.058 %  same donor diff re-applied to 942e5ab2 by 3 solvers
          rows: 4f76de6e, 11863aa9, 3ec77796
FORCE     n=1  mean +0.156 %  crown: setenv overwrite 0->1 and 320/128 -> 512/50
          rows: 0cd0a6b4
contrast-clustered mean +0.328 %, se 0.223, |t| 1.47 on 2 df
effective n falls 5 -> 3
```

The three RESTORE rows re-apply **the same two donor files** (from `86fb1f0`)
to **the same `942e5ab2` parent** after a yukon replace-overlay dropped them.
Three solvers, three commits, **one contrast measured three times**.

🔴 **And that is the cleanest identified estimate on offer** — same diff, same
parent, three independent sessions, no bundling. It averages **+0.058 %**,
about **one third of the 0.185 % bar**. The +0.316 % headline was carried
almost entirely by the single INTRODUCE row, where the mechanism is bundled
with four others.

This also matters for us directly: the RESTORE contrast is **worth zero to
us**, because we already carry that code.

### 3.4 Multi-mechanism contamination — the family is not identified

`3a995c2b` bundles **five** mechanisms and supplies the largest delta.
Membership is assigned by **regex over the note title**
(`research/within_head_cost.py:611`); a title naming one mechanism does not
mean the tree contains only that mechanism, and every row builds on an
inherited frontier. **Family attribution is not identified**, exactly as
suspected.

### 3.5 What our two tests actually did, and their MDE

**Test A — "wired limit with headroom":** nothing was run. n = 0, MDE
undefined, mechanism hardware-gated out, claim imported from a sister challenge
about the opposite arm.

**Test B — E31 "command-buffer geometry":**
`research/results/e31-mlx-command-buffer-geometry.md:5` —
**"Timed arms run in this experiment: zero."**

E31 was a **static source audit**, and it got the environment *right* (the
96 GiB gate and the 512/50-vs-128/64 split are documented at `.md:105-115`). Two
premises of the item-69 closure were in fact **corrected by E31 itself**: the
caps *are* settable from editable Swift (`.md:84-89, 195-198`), and the
constants were never 50/50 (`.md:16-19`).

The null is **borrowed from E29**, which fails on six counts:

1. **Wrong lever.** The knob was `MLX_QWEN_MTP_LADDER`, which *adds forced
   `asyncEval` boundaries*. **No arm ever changed
   `MLX_MAX_OPS_PER_BUFFER`/`MLX_MAX_MB_PER_BUFFER`.** E31 declares the useful
   direction — `[1, floor]`, *fewer* commits — **"formally unswept"**
   (`.md:200-204`), and proposes the decisive test (`.md:261-265`). Nobody ran it.
2. **Mechanism mismatch.** `gpu::finalize` commits **bypass** the
   `MAX_ACTIVE_TASKS=10` accounting that automatic `needs_commit()` commits
   enter (`.md:61-64`). Forced rungs ≠ automatic commits.
3. **Wrong regime.** Local: 64 ops / 128 mebi-elements, **element-bound**,
   ≈31.3 commits/forward. Ranked: 50 / 512, **op-bound**, ≈19.7.
4. **Wrong head.** `7bbb40de…`, not the ranked `559b24eb…`.
5. **Invalid thermal protocol.** All arms `cool_gate_passed_real_gate=false`,
   run in **monotone order, not ABBA**, entry temps **41.6 °C for the D0 control
   vs 62.9 / 62.5 / 62.7 °C** for the treatments — a 21 °C cold-start advantage
   for the control. `program.md` permits ungated timing *only* with ABBA
   counterbalancing, so this session does not qualify even as directional
   evidence.
6. **Wrong units.** `MLX_MAX_MB_PER_BUFFER` is a **mebi-element** cap, not
   megabytes.

**MDE arithmetic.** Slope −36.5 µs/boundary, **se 31.1 µs, 2 dof**, 95 % CI
[−170.3, +97.4] µs:

```
MDE(normal)      = 2.8016 × 31.1                    =  87.1 µs/boundary
MDE(exact, 2 df) = noncentral-t bisection at 80 %   = 351.6 µs/boundary
board +0.316 %   = 0.00316 × 172.1 ms / 19.74       =  27.6 µs/boundary
```

| | value |
|---|---|
| target / se | **0.89 — the effect is smaller than one standard error** |
| under-powered (normal) | 3.2× |
| **under-powered (exact, 2 df)** | **12.7×** |
| vs refreshed +0.220 % | **18.3×** |

The closure's headline *"+0.418 % central"* is a **linear extrapolation of a
statistically-zero slope across a mechanism mismatch** — a model, not a
measurement.

### 3.6 Does the +0.316 % survive? No.

| test | result |
|---|---|
| refreshed corpus (629 → 646 rows) | +0.316 % → **+0.220 %**, \|t\| 2.2 → **1.54** |
| serial-corrected | **+0.149 %**, \|t\| 1.04 — **below the bar** |
| contrast-clustered (n 5 → 3) | +0.328 %, se 0.223, \|t\| 1.47 on 2 df |
| **cleanest identified contrast (RESTORE, n=3)** | **+0.058 %** — a third of the bar |
| multiplicity, E[max of 9] | **+1.265 %** — observed is *below* it |
| power of the instrument at its own observed effect | **0.337** |
| n needed to reach the 0.185 % bar | **25 rows**; it has 3 contrasts |

**Killed, not strengthened.** The mechanism may still be real — a null this
weak cannot say — but the board provides no usable evidence for it, and it
cannot be used to reopen entries 6 and 11.

### 3.7 Two source corrections that change the re-test recipe

🔴 **Correction 1 — `DARKBLOOM_STARTUP_MEMORY_PROFILE=full` does not give you
512/50.** `RuntimeStartupMemoryPolicy.swift:66`,
`installQwenMTPFullProfileCommandBufferDefaults`, has **its own independent
96 GiB gate** that `=full` does not bypass, followed by `setenv(..., 0)`
(overwrite = **0**). On a 48 GiB box, `=full` alone leaves the geometry
untouched.

What `=full` *does* is set `isLowMemory = false`, suppressing the forced
low-memory `setenv` that would otherwise stomp a user export. Combined with
overwrite = 0, an explicit export survives:

```bash
DARKBLOOM_STARTUP_MEMORY_PROFILE=full \
MLX_MAX_MB_PER_BUFFER=512 MLX_MAX_OPS_PER_BUFFER=50 \
  ./benchmark-qwen-mtp.sh --local-iterate
```

All three pass the worker allowlist (`QwenRuntimeWorker.swift:2623-2655`:
`DARKBLOOM_ DYLD_ LC_ METAL_ MLX_ MTL_`). The ranked workflow sets none of them.
**This is what makes entry 11 re-testable locally at all** — it is the only way
to put this box in the ranked op-bound regime.

⚠️ **OOM hazard — read before running this on a 48 GiB box.** Setting
`DARKBLOOM_STARTUP_MEMORY_PROFILE=full` does more than unblock the caps: it
takes the whole low-memory branch out of play, which on this host removes two
protections that exist precisely because the box is small —

- the **6 GiB allocator cache cap** (`RuntimeStartupMemoryPolicy.swift:107`,
  `cacheLimitBytes: 6 << 30`) is replaced by the full profile's **32 GiB**
  (`:138`, `cacheLimitBytes: 32 << 30`) — which is *two thirds of this box's
  entire RAM*; and
- **`clearAllocatorCacheAfterWarmup`** flips from `true` (`:114`) to `false`
  (`:147`), so the warmup arena is never released.

On a 48 GiB machine holding ~14 GiB of transformed weights plus head and KV,
that is a real risk of a mid-run OOM, and an OOM part-way through an ABBA
session destroys the counterbalancing rather than just costing one leg. Anyone
running this should raise the caps **without** removing the cache protections
where possible, watch RSS on the first pair, and treat the first run as a
throwaway smoke test. The recipe is correct; it is not free.

🔴 **Correction 2 — the MTP worker does not call `policy.apply()`.**
`QwenRuntimeMTPWorker.swift:481-489` **inlines** it:
`guard policy.isLowMemory else { return }` then `setenv(..., 1)` forcing 128/64.
Same effect, different code path — worth knowing before someone greps for
`apply()` and concludes the policy is inert. On the ranked box
`isLowMemory == false`, so our full-profile **320/128 constants are dead code**;
effective ranked geometry is **512/50**.

Also: `Qwen35RuntimeWeights.swift:45` (`MLX_MAX_MB_PER_BUFFER=128`,
overwrite = 1) is dead on the MTP path but **live on the serial/local-iterate
path** (ofou H3). It is unmeasured by us and is a plausible confounder in
**every local ratio the campaign has taken**.

**Standing gap:** our tree still ships `setenv(..., 0)` and 320/128
(`RuntimeStartupMemoryPolicy.swift:71-72`, `:146-147`) — precisely what the
crown changed. We are synced to promoted `11863aa9`, not to the crown
`0cd0a6b4`.

---

## 4. The ranked re-test list

Ordered by (plausible effect) / (cost of a decisive test). Cost is in local
timed legs unless stated; the named instrument is the **cheapest one that would
actually be decisive**, not the most thorough.

| # | entry | plausible effect | cheapest decisive instrument | cost | why it is decisive |
|---|---|---|---|---|---|
| **1** | **4 — qmm for M ≥ 4** | **up to +61 % on the projection path** | paired microbenchmark of the **right** mechanism: row-batching into `qmm`, n ≥ 5 | ~0 GPU-min, no E2E leg | The closing test measured padding across `vector_limit`. Nobody has ever benchmarked the actual mechanism. Largest unmeasured ceiling on the list, at the lowest cost. |
| **2** | **11 — command-buffer caps** | +0.27 – 0.6 % (Laguna cmdbuf-alone receipt +0.6 %) | local ABBA at `OPS=4000`, `MB=8192` **under the Correction-1 recipe**, 2 pairs ⚠️ **OOM hazard, see §3.7** | 4 legs + 1 throwaway smoke run | Correction 1 puts this box in the ranked op-bound regime, so the mechanism claim transfers. Sweeps the `[1, floor]` direction E31 declared unswept. |
| **3** | **18 — KV-1024 warm** | unquantified; source says *"not proven dead"* | paired local E2E, n = 5 (MDE → 0.527 %) | 10 legs | The χ² fit never had a power statement; n = 5 brings the MDE under the mechanism's own plausible size. |
| **4** | **8 — argPartition top-32** | ~0.2 – 0.5 % | paired local E2E, n = 5 | 10 legs | Currently one leg with **no repeat at all**; any replication is a strict improvement. |
| **5** | **2b — rows_per_simd ≠ 4 (E2E)** | ~0.2 – 0.4 % | paired local E2E, n = 5 | 10 legs | Structural half (2a) stays closed; only the E2E half needs power. |
| **6** | **10 — host round overhead** | ~0.3 % | paired local E2E, n = 5, **ABBA-counterbalanced** | 10 legs | Its MDE (4.86 % exact) is the worst on the list, and its noise floor is the contaminated D0/N1 pair. Needs a real repeat before any claim. |
| **7** | **13 — replacement head artifact** | ~0.2 % | our own head A/B, n = 3 | 6 legs + head build | n=1 competitor row with borrowed σ; head claims do not transfer across provenance. |
| **8** | **16 — values_per_thread (verify path)** | ~0.2 % | paired local microbenchmark, n = 5 | ~0 GPU-min | Cheap, but the plausible effect is small; below the qmm and cmdbuf items. |
| **9** | **20 — beagle acceptance** | prompt-specific | per-prompt acceptance trace, n = 3 | 6 legs | Regex-classified from competitor titles; the "either direction" framing is not a measurement. |
| — | **5 — wide target-top-2 reducer** | −3.85 % claimed | *none proposed* | — | Competitor prose about a mechanism we do not run. Re-test only if we adopt the reducer. |
| — | **6 — wired limit with headroom** | n/a | **none exists on this hardware** | ∞ | 96 GiB gate, no override. Not a power problem. Needs a ≥96 GiB host or an organizer-visible ranked probe. |
| — | **23 — shipped-surface gate** | correctness of every submission | recount the diff against `promotedSourceRef` | ~0 | Not a speed question, but it is wrong on every submission and cheap to fix. |

**Two orderings worth flagging.** Entry 4 outranks entry 11 despite entry 11
being the assignment's focus, because its plausible effect is two orders of
magnitude larger and its decisive test costs no GPU legs at all. And entry 6,
the other focus, is **unrunnable at any n** — it belongs on the strike list, not
the re-test list.

---

## 5. 🔴 The strike list

Entries that should stop being cited as closed. The assignment says: *"I would
rather strike five entries and be honest than keep twenty and be wrong."* Here
are **eight**, in descending order of how much damage the citation is doing.

| # | entry | strike because | what to write instead |
|---|---|---|---|
| **1** | **6 — wired limit with headroom** | **n = 0.** Never measured by us. Competitor claim, sister challenge (Laguna M5-Max), **opposite arm** from the board positive, and the mechanism is hardware-gated out at 96 GiB vs our 48 GiB. Ledger item 81 ordered it *kept*, not closed. | *"Unmeasurable locally; we ship the zero-headroom variant; the with-headroom negative agrees with the board and is not evidence against us."* |
| **2** | **11 — command-buffer geometry** | E31 ran **zero timed arms**. Null borrowed from E29, which swept the wrong lever, in the wrong regime, on the wrong head, under a **non-ABBA ungated protocol with a 21 °C control advantage**. Under-powered **12.7×** even on its own axis. The `[1, floor]` direction is **formally unswept**. | *"Open. The direction that could help has never been swept."* |
| **3** | **21 — warm coverage / shape gaps (E37 d3)** | **E37 was never delivered.** The live record treats `warmAllDepths` as **open and contested** (`senpai/campaign-ledger.md:2304, 2322-2325`). | Delete the entry. |
| **4** | **4 — qmm for M ≥ 4** | Closed by a single un-replicated microbenchmark of a **different mechanism** (padding across `vector_limit`, not row-batching into `qmm`). No σ, no n, against a predicted **+61 %** ceiling. | *"Untested. Highest unmeasured ceiling on the list."* |
| **5** | **7 — E25 arm D** | Well powered locally (+3.18 ± 0.77 %, 4.1σ) and still wrong: **construct validity**. Fixture depth 2.06–2.74 vs scored 4.53–5.78. More replicates would never have caught this — it is the counter-example to *"just add legs."* | *"Measured a depth regime the scored path does not visit."* |
| **6** | **10 — host round overhead** | MDE **4.86 %** (exact) against a 0.185 % bar — the worst instrument on the list — and its noise floor is the **contaminated D0/N1 pair** (differs by `trace` flag *and* a refactor). | *"Uninformative. Needs a genuine same-config repeat first."* |
| **7** | **8 — argPartition top-32** | One timed leg, **no repeat of any kind**. The σ is borrowed. | *"Single observation; no null was established."* |
| **8** | **23 — shipped surface "4 files, +117/−87"** | Wrong baseline. Against `5273067` it is **5 files, +229/−74**, and the gate never counted `RuntimeStartupMemoryPolicy.swift` (**+32/−0**). A gate reporting a stable count while missing a file is a **config echo**. | *"5 files, +229/−74 against `5273067`. Recount on every submission."* |

### Two corrections rather than strikes

- **12** — verdict CLOSED stands, but the constant is in **`MLXFastCore`**, not
  `MLXFastModel`, and that module's absence from `editablePaths` is the
  load-bearing reason.
- **22** — CLOSED only for *"a depth **constant** re-prices 0 of 6 wins."*
  **Structural depth work remains open** (`within_head_cost.py:907-919`), and
  `de7981ae` at 3.24078 is rank 5.

### And one correction to the campaign's power vocabulary

E33 is described as *"3.5× under-powered."* That compares a **+0.088 %**
prediction to a **±0.30 % 1σ resolution**. A 1σ resolution is not an MDE. The
true figures are **9.5×** (normal) and **19.1×** (exact). The same slip — using
1σ where 2.8σ/√n belongs — will understate every future power statement by
about 3–5×, which is why `e39_mde.py` exists.

---

## 6. What I would tell the advisor in one paragraph

The reopening was the right instinct applied to the wrong evidence. Entry 6 was
never in conflict with the board — our negative and the board positive describe
opposite arms of the same mechanism and agree with each other — and it has
n = 0 on hardware that cannot run the code. Entry 11 *is* a genuine hole, but
not for the stated reason: E31 ran nothing, and the E29 null it borrowed is
12.7× under-powered on an axis it never swept, under a thermal protocol that
disqualifies it. Meanwhile the board evidence that motivated all of this is
itself under-powered by exactly the standard this audit applies: refreshed to
+0.220 %, serial-corrected to +0.149 %, cleanest-contrast +0.058 %, power 0.337,
and below the multiplicity floor for a max-of-9. The genuinely valuable output
is not the residency story at all — it is **entry 4**, where a predicted +61 %
ceiling was closed by a microbenchmark of a different mechanism, and which costs
no GPU legs to settle.

---

## 7. Reproduction

```bash
python3 research/e39_mde.py --self-test     # 13 checks, incl. Cohen + G*Power anchors
python3 research/e39_mde.py --audit         # instrument registry table
python3 research/e39_residency_audit.py     # board scrutiny, all 5 sections
```

`e39_mde.py` also supports ad-hoc queries:

```bash
python3 research/e39_mde.py --mde --sd 0.2974 --n 2 --design two_sample
python3 research/e39_mde.py --n-required --sd 0.2974 --target 0.185 --design paired
```

### Key constants used

| constant | value | source |
|---|---|---|
| σ_score | 0.0923 % | median-of-8 on the crown's profile |
| 2σ bar | 0.185 % | assignment |
| engineerable gap | 0.561 % | `R'`, 4.3σ paired |
| E33 pooled sd | 0.29741 % | within-arm-type spreads |
| E29 slope se | 31.1 µs/boundary, 2 dof | `e29-analysis.json` |
| board corpus | 646 submissions, 83 head-matched | refreshed this session |
| ranked head | `559b24eb…` | assignment |
| local heads | `7bbb40de`, `07293af7` | E29/E25 raw captures |

### Deviations and limits

- **No GPU runs**, per the constraint. Every re-test proposal is in §4.
- Entry 23 was **corrected after first submission**. My initial pass used the
  wrong reference (`5068eb8d`, not a valid git object here) and omitted the
  largest shipped file. The figures now in §2 are recomputed against `5273067`
  and agree with the advisor's.
- PR #36's comment thread was not readable from this role, so E31's advisor
  stop-instruction is known only through its citation at
  `e31-mlx-command-buffer-geometry.md:164-166`.
- Family membership on the board comes from **regex over note titles**
  (`within_head_cost.py:611`). I report contrast identity separately precisely
  because titles are not a reliable unit.

## 8. Suggested follow-ups I did not implement

- **Recount the shipped surface** against `promotedSourceRef` and fix the gate
  (entry 23). Cheap, and it is wrong on every submission today.
- **Adopt the crown's constants deliberately or reject them deliberately.** We
  ship `setenv(..., 0)` and 320/128; the crown ships `1` and 512/50. Right now
  we have neither measured it nor matched it.
- **Establish one honest same-config repeat** for the local E2E leg. The whole
  campaign's noise vocabulary rests on the contaminated D0/N1 pair, and
  `local_e2e_leg_e29` (MDE 4.86 % exact) is propagating that into every entry
  that borrows it.
- **Audit `Qwen35RuntimeWeights.swift:45`** — live on the serial/local-iterate
  path, dead on MTP. If it moves the serial leg it has been silently biasing
  every local ratio we have ever taken.
- **Re-derive the E27 transfer rule.** Six entries are pre-E27; I classified
  them by date, but only a width-behaviour entry is genuinely invalidated by
  E27's M=5 change. A per-entry mechanism test would be sharper than a date cut.
