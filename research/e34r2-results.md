# E34 r2 — model autopsy: what the depth cost model gets wrong

Assignment `qwen38-r1-e34-ranked-operating-point-depth-cap`, revision `r2`, PR #39.
Base `senpai/qwen38-mtp-r1` @ `abf6d79f92b97e3c47856be9c1d7798e6dc5a6b5`.

**Zero GPU time was used.** No build, no benchmark, no timed run. The shipped-surface
diff against base is empty; every file changed is under `research/`.

Reproduce everything:

```bash
python3 research/e34r2_model_autopsy.py --self-test   # 0 failures, incl. telemetry re-derivation
python3 research/e34r2_model_autopsy.py --json research/e34r2-model-autopsy.json
```

---

## 1. Primary metric: WITHDRAWN

`e34/predicted_ranked_central_pair_at_best_cap` is withdrawn as a decision input.

| | value | vs baseline `3.24929398547457` |
|---|---|---|
| r1 published value | `3.77855631847542` | +16.29 % |
| repaired model (MT1+MT3 fixed) | `3.563809083470352` | +9.68 % |

Neither number should size a ranked decision, for three independent reasons:

1. The r1 model gets **0 of 6** signs right on the declared-head rows that are
   shallower than ours. It predicts **+0.86 % to +3.05 %** where the board
   measured **−0.32 % to −3.47 %**.
2. Repairing the two provable errors below changes the magnitudes but **not the
   sign record: still 0 of 6**. Overall sign agreement is 6/14 either way — worse
   than a coin.
3. The repaired model also **mis-orders the central pair**. It predicts
   `medicine, republic`; the ranked reality is `beagle, medicine`. A median of
   eight numbers is decided by exactly the ordering the model gets wrong.

I re-issued the number rather than silently dropping it so the withdrawal is
auditable, but it is not defensible against the 15 rows and I am not defending it.

---

## 2. The real deliverable: four errors, and the one that decides the sign

### MT1 — token credit (proved exactly; the largest single error)

`effective_mean_draft_len` counts drafts **PROPOSED**, not accepted.

* Source: `Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift:363-374` —
  *"what the candidate actually **proposed**"*, meaned over all rounds.
* Exact integer identities close on **8/8** local tapes:
  `sum(effective_draft_lengths) == accepted + rejected`,
  `declared_rows_total == round_count + proposed`,
  `emitted_token_total == round_count + accepted`.
* r1 credited `1 + n` emitted tokens per round. That is **identically
  equivalent to asserting a 100.000000 % acceptance rate on every prompt** — the
  script prints `1.000000` for all eight, by construction.
* Independent impossibility: our plutarch row publishes 449 non-drafting rounds.
  Under r1's credit the leg would have only `512/1.154 = 443.67` rounds in total,
  i.e. **−5.33 drafting rounds**.

### MT3 — build asymmetry (now measured, not inferred)

Ranked `raw_p` divides the **pinned serial build** by **our candidate build**.
Both local legs use the candidate build. r1 predicted a ranked ratio using a
local ratio's structure.

The elimination argument put a lower bound of `k ≥ 1.2132` on how much slower the
pinned serial build is at width 1. **Section 3 measures it directly: `k = 1.2090`
(range 1.2059–1.2115 across all eight prompts).** The two agree to 0.35 %; the
direct measurement is authoritative and shows the elimination bound was very
slightly optimistic because it inherited the local ladder.

### MT4 — geometry confound in the ladder that produced the number

The 48 GiB local box takes the low-memory branch at
`Sources/.../QwenRuntimeMTPWorker.swift:487`: `MLX_MAX_MB_PER_BUFFER=128`,
`MLX_MAX_OPS_PER_BUFFER=64`, 6 GiB cache limit, cache cleared after warmup,
residency **off**. The ranked 128 GiB box uses 512/50, MLX default cache,
residency **on**.

`mlp.down` moves 100.3 MB/call at M=6 and ≈50.1 MB at M=5, so under **local**
geometry the calls-per-command-buffer drops 2 → 1 exactly at M=6 (ranked: 10 → 5).
**The buffer-flush boundary and the weight-pass boundary are perfectly confounded
at M=6 locally**, so the step-vs-smooth fit cannot attribute the step to either.
The E33 *structure* survives; the *attribution* does not. Every absolute in the
E33 ladder is a low-memory-profile number. Paired ratios within one geometry
survive; absolutes describing the ranked box do not.

### MT2 — `weight_passes(M)` is shape-blind (secondary)

`research/e34_cost_model.py` returns one pass count per width. The live kernel
selects on `out_vec_size`: ≥4096 uses the IPG table, 1024 ≤ n < 4096 hardcodes
IPG 2 at `quantized.h:872`, below that it is the stock path. Qwen 3.8 has
projections in at least two regimes (`k_proj`/`v_proj` are 1024). Worth fixing
before the model is used again, but it is not what flips the sign.

---

## 3. MT5 — the decisive evidence, obtained with zero GPU time

The ranked telemetry publishes `effective_mean_draft_len` as an **exact
rational** and the window is a fixed 512 tokens with one primary token committed
per round. Therefore

```
512 = rounds + accepted_total       accepted_total ≤ proposed_total = n · rounds
```

`n = p/q` in lowest terms means `rounds` must be a multiple of `q`, which leaves a
handful of integer candidates per prompt. Adding only *"per-round cost is
non-decreasing in mean width"* leaves **exactly one** globally consistent
reconstruction. Everything below comes from our own submission `ca9251b8`, so no
cross-build or cross-machine comparison is involved.

**Validation: the reconstructed median of the eight ratios is `3.23250500`
against the published official score `3.23250848` (they differ only in the 6th
decimal, because `raw_p` is quoted to 5 dp).**

| prompt | mean M | raw_p | rounds | proposed | accepted | acceptance | per-round ms |
|---|---|---|---|---|---|---|---|
| plutarch | 1.1540 | 1.25280 | 487 | 75 | 25 | 33.33 % | 31.862 |
| drama | 3.2976 | 1.91668 | 252 | 579 | 260 | 44.91 % | 40.181 |
| travel | 3.6557 | 2.17980 | 212 | 563 | 300 | 53.29 % | 41.995 |
| **beagle** | 5.5327 | 3.12015 | 107 | 485 | 405 | **83.51 %** | 58.253 |
| **medicine** | 5.7677 | 3.34486 | 99 | 472 | 413 | **87.50 %** | 58.795 |
| republic | 6.2697 | 3.39402 | 89 | 469 | 423 | 90.19 % | 64.338 |
| essays | 6.4253 | 3.36612 | 87 | 472 | 425 | 90.04 % | 66.249 |
| botany | 6.7765 | 3.42536 | 85 | 491 | 427 | 86.97 % | 66.744 |

Linear fit `per_round = 20.543 + 6.792·M` ms, R² = 0.9706.

### 3a. MT3 measured directly

Submission `c91581eb` (scarletbright) proposes 18–27 drafts across 512 rounds and
**accepts zero, on all eight prompts**. Its reconstruction is unique with no
filtering at all: `rounds = 512`, `accepted = 0`. Its candidate leg is therefore
plain serial decoding through a candidate build, at **31.42–31.51 ms/round** —
and it still scores **1.206–1.212**.

> Any model whose denominator is the candidate's own width-1 time must score
> **exactly 1.000** on that row. The board says 1.21.

The pinned serial leg over **88 declared-head submissions × 8 prompts** is
`37.9908 ms/token`, spread across prompts `0.0231 ms`, within-prompt CV
`0.20 %–0.28 %`. The serial leg is both genuinely pinned and essentially
**prompt-independent** — the whole per-prompt spread in `raw_p` comes from the
candidate leg.

### 3b. The elimination now fires on the prompts that matter

For M=6 vs M=5, the wide round wins iff `q > (1 + a₄)·(T(6)/T(5) − 1)`. With the
local step 1.33984 this needs `q > 1` — impossible — once the per-draft
acceptance rate reaches **0.4856**.

In r1 I could only state this conditionally, because the local fixture measures
27.6–35.9 %, which sits *below* the threshold. The reconstruction settles it:
**measured ranked acceptance clears 0.4856 on 6 of 8 prompts, including both
central prompts (beagle 83.5 %, medicine 87.5 %)**.

So on the prompts that decide the median, **no acceptance-side term — higher
average acceptance, streak-path conditional selection, or a perfect head — can
rescue the wide round.** The residual is forced into the **cost** term. That
independently confirms the advisor's occupancy / grid-width direction and rules
out the streak/selection suspicion as *sufficient*.

### 3c. P1 and P2 confirmed by measurement, and exceeded

I registered P1/P2 before looking at this data. Both are confirmed, and P1's
magnitude is exceeded.

| | predicted | measured |
|---|---|---|
| ranked T(6)/T(5) | `< 1.2221` | **`1.1246`** |
| local overstatement of the M=5→6 step | `≥ 9.64 %` | **`19.14 %`** |
| overstatement concentrated at wide widths (P2) | yes | **+1.08 % at mean M<4, +8.49 % at mean M≥5** |

The falsifier (`T(6)/T(5) ≥ 1.30` under ranked conditions) did **not** fire.

**Caveat, stated plainly:** ranked "mean M" is the mean of an adaptive policy's
width distribution, not a fixed width, so this is a ladder in *mean* width. It is
comparable with the fixed-width local ladder only under near-linearity. The claim
that survives without that assumption is the one that matters: **there is no
30 ms cliff between prompts whose mean widths straddle 6.** Locally the 5→6 step
is +32.680 ms; the ranked fit gives +6.792 ms per unit of mean width across the
whole range.

### 3d. Fourth transfer failure: acceptance itself does not transfer

The local fixture's 27.6–35.9 % acceptance sits in the *shallow* regime — near
ranked plutarch (33.3 %) — not near the prompts that decide the median (83.5 %,
87.5 %). Any future model calibrated on local-fixture acceptance will be wrong in
the same direction.

---

## 4. Provenance corrections to the 15-row table

Re-derived against the telemetry cache; all 15 rows now re-verify via
`--verify-telemetry`.

* **Dropped `a874233e`** (jonathan308). It is still `status: validating` with
  null `officialMetrics` and **has no beagle per-prompt entry at all**. The
  quoted pair (n 4.583, raw_p 3.06986) is not a measurement.
* **Added `baa75efa`, `26d0e934`, `a1326b4b`** — scored declared-head rows the
  table omitted. This widens the near-zero end of the sweep from one row to three,
  which is what makes §3a robust.
* Six of the eight per-prompt `raw_p` values I was carrying for our own row were
  slightly off; corrected from the cache. The central pair is unchanged.

---

## 5. Items kept as-is, and one withdrawal

Per the r2 instruction, r1 items (a), (b), (d), (e) and the provenance finding
are unchanged and not reworked. Three housekeeping updates:

* **Withdrawn:** my r1 claim that *"the fixture cannot see this wall."* Askeladd
  is right — shipped BASE reaches M=5/M=6 and thorfinn's journal reaches M=8. My
  "candidate 3" reading was E25's *modified-policy* arm, not the shipped path.
* **Adopted:** σ_score = 0.0923 %, 2σ detection threshold 0.185 %, engineerable
  gap 0.561 % (not my 0.2587 %).
* **Askeladd's bracket quoted alongside my simulated M≥6 round shares:** beagle
  `[0.1332, 0.9065]` vs my 0.538; medicine `[0.1919, 0.9535]` vs my 0.593. Both
  sit inside; the bracket is too wide to discriminate. §3 now supersedes it —
  the round counts are pinned exactly rather than bracketed.
* The **depth constant is closed** (alphonse E35: explains 0 of 6 wins). Nothing
  here re-opens it, and I am **not** proposing to ship the cap change.

Two source defects found while checking my own reasoning:

1. `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1198` — the comment
   *"crossrow for M ≤ 5, per-row qmv_fast above it"* is **stale and wrong**.
   Crossrow is selected inside the kernel at `quantized.h:1917-2015` on
   `ntg.x == M` for **M = 2..9**; stock `qmv_fast` runs only at M ≥ 10 or M = 1.
   The advisor's IPG table and the r1 pass-count reading for M=6 are correct; the
   comment led me to a wrong draft hypothesis that I discarded.
2. `research/ESTABLISHED_FACTS.md:1377-1396` claims NA=5 was refuted and
   "NA_max = 4 restored", which contradicts the live `NA <= 5` at
   `quantized.h:980`.

---

## 6. Honest assessment

The r1 prediction was wrong and is withdrawn. Two provable errors (MT1, MT3) fix
its magnitude and its transfer but leave the sign record untouched at 0/6, which
is why the sign residual is genuinely informative: it cannot be an acceptance-side
artefact on the central prompts, so it belongs to cost.

The part I did not expect is that the ranked cost curve was **measurable all
along** from published telemetry, without a GPU. The reconstruction reproduces
our own official score to six decimals, so it is not a fitted story. It says the
ranked box has **no width cliff at 6** — the local one is a 19 % artefact of the
low-memory command-buffer profile — and that ranked acceptance on deep prompts is
**83–90 %**, roughly 2.5× what the local fixture shows.

## 7. Suggested follow-ups (not implemented)

1. **Retire the local absolute ladder as a ranked predictor.** Rebuild the depth
   model on the reconstruction in §3, which is ranked-native and free. If a local
   ladder is still wanted, measure it under `DARKBLOOM_STARTUP_MEMORY_PROFILE=full
   MLX_MAX_MB_PER_BUFFER=512 MLX_MAX_OPS_PER_BUFFER=50` so the geometry matches.
2. **Run the reconstruction across all 88 declared-head rows**, not just ours.
   That yields a per-solver ranked cost curve and acceptance profile, and would
   show directly which solvers are winning on cost and which on acceptance.
3. **Re-locate the ~1900-threadgroup occupancy knee under ranked geometry.** It
   was found under the low-memory profile; command-buffer packing and occupancy
   are both dispatch-side, so the knee's position is geometry-dependent and is not
   a machine constant.
4. **Fix `weight_passes(M)` for the three `out_vec_size` regimes** (MT2) before
   the cost model is used for another decision.
5. Fix the two stale source claims in §5.

## 8. Addendum — the arch string, and why it is worth more than the rest of r2

The advisor asked (feedback `e34-r2-...`, item 7) for the effective
`get_max_ops_mb_per_buffer()` tuple and the `applegpu_*` arch string, noting that
three experiments are blocked on it. `research/archprobe.m` already existed on the
base — the advisor added it in `8ded37b` (ledger 115-119) for the SDPA two-pass
question — so this extends that tool rather than adding a second one, and runs
it. It enumerates the Metal device only: no compute, no timing, so it does not
contend with the precision-timing slot.

```
device.name         = Apple M4 Pro
architecture.name   = applegpu_g16s
MLX arch_gen        = 16
MLX devc (back)     = 's'
SDPA 2pass at KV>=1024 on this box? YES
_nax gen threshold  = 17
_nax available (arch gate only)? NO
MLX stock budget    = 50 ops / 50 mebi-elements
threadgroup_memory_bytes = 32768
recommended_wset_bytes   = 40200896512
```

The three added lines are the `_nax` gate, its threshold, and the arch-derived
stock command-buffer budget. Everything else was already printed by the existing
tool; the finding below is what the tool says when it is run.

### 8.1 🔴 `_nax` kernels cannot execute on this box

`device.cpp:924-926` gates the whole `_nax` family:

```cpp
auto arch = d.get_architecture().back();
auto gen  = d.get_architecture_gen();
can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
```

`arch_gen_` is parsed positionally from the two digits at `arch_[size-3]` and
`arch_[size-2]` (`device.cpp:563-571`), so `applegpu_g16s` gives `gen = 16` and
`arch = 's'`. The threshold for a non-`p` arch is 17, and **16 >= 17 is false**.

`program.md` states that "`_nax` variants run on the ranked M5 and are
first-class targets". Both statements cannot describe the same kernel family:
either the ranked box runs `_nax` and this box provably does not, or `program.md`
is wrong about the ranked box. Under the first reading — the one `program.md`
asserts — **every local kernel measurement in this campaign was taken on the
non-`nax` path while the ranked score is produced on the `nax` path.** That
covers the E27 causal ladder, E30's per-width absolutes, E33's dispatch-table
work, and my own `T(M)` ladder. `quantized.cpp` references `nax` 23 times, so the
divergence reaches the quantized matvec dispatch this campaign has been tuning.

This is a second, independent local-to-ranked divergence, distinct from the
memory-profile split, and it is the better explanation of §3's result: if the
ranked box dispatches `_nax` quantized matvec, the local dispatch table is not
the ranked dispatch table, and the local M=6 cliff belongs to a kernel family
that never executes at rank. That is consistent with the reconstruction finding
no cliff at mean width 6 in ranked telemetry.

I have **not** measured the ranked M5's arch string; I can only prove the local
side and combine it with `program.md`'s claim. Confirming it needs one line of
worker-log telemetry, which is why item 7 is worth doing.

`MLX_METAL_GPU_ARCH` (`utils.h:205`, consumed at `device.cpp:560`) overrides the
arch string outright, so a forced value can be used to test ranked selection
locally — but it must keep the `<letter><digits><letter>` shape or `arch_gen_`
silently reads 0 and every gen-keyed branch changes at once.

### 8.2 The command-buffer budget is in mebi-elements, and it is dtype-dependent

The advisor's unit correction is confirmed, with the citation: `array.h:346`
states `data_size` is "in units of `item_size` (not bytes)"; `device.cpp:320`
accumulates `buffer_sizes_ += a.data_size()`; `device.cpp:486` tests
`(buffer_sizes_ >> 20) > max_mb`. So the budget is **mebi-elements**.

Two refinements the correction did not carry:

- Because it counts elements, the byte budget depends on dtype. 4-bit affine
  weights packed in `uint32` are 4 B/element, `fp16` activations 2 B/element, so
  512 mebi-elements is about 2 GiB of packed weights or 1 GiB of `fp16`. Any
  reasoning that converts the budget to megabytes is wrong by a dtype-dependent
  factor of 2 to 4.
- `set_input_array` only accumulates on `all_inputs_.insert(a.buffer().ptr()).second`,
  so a buffer referenced many times inside one command buffer is counted **once**,
  and outputs are not counted at all.

🔴 **That second point weakens my own §3 mechanism.** I attributed the local
M=5→6 step to `mlp.down` at 100.3 MB/call dropping calls-per-command-buffer from
2 to 1 exactly at M=6. Under element units with per-buffer dedup, the weight is
counted once per command buffer however many rounds reference it, and only the
activations scale with M, so the flush boundary does not move with M the way I
claimed. The measured result in §3 stands — it comes from ranked telemetry — but
the command-buffer *explanation* for it does not, and §8.1 is the better
candidate. I am withdrawing the flush-boundary attribution.

### 8.3 The stock default on this box is 50/50, not 40/40

`device.cpp:572-595` picks the default budget from the **last character** of the
arch string: `p` 20/40, `g` 40/40, `s` 50/50, `d` 50/50, otherwise 40/40. This
box reports `s`, so its stock default is **50 ops / 50 mebi-elements** — even
though it is a *Pro*, not a Max. The switch's `// max` comment is a label, not a
fact about the part.

Consequences for the four-way table in the advisor's earlier note:

- The local `apply()` path forces 128/64 with `setenv(..., 1)`
  (`RuntimeStartupMemoryPolicy.swift:211-224`), which **raises** the budget from
  the stock 50/50. It is smaller only relative to the ranked 512, not relative to
  stock, so "the referenced-byte budget is 4x smaller locally" is the wrong
  comparison.
- The ranked >= 96 GiB path installs 512/50 with `setenv(..., 0)`
  (`RuntimeStartupMemoryPolicy.swift:66-72`), i.e. pre-existing values win. If the
  ranked arch also ends in `s` or `d`, then `MLX_MAX_OPS_PER_BUFFER=50` is
  **exactly the stock default and installs nothing**, and the only real ranked
  change is mb 50 -> 512.
- The comment at `RuntimeStartupMemoryPolicy.swift:212-214` says the budgets are
  "per-profile absolutes (the pre-policy code force-set them identically)", but
  the >= 96 GiB branch uses no-overwrite semantics while `apply()` uses force.
  The two paths are not symmetric and the comment says they are.

The probe also confirms that a fresh process on this box sees
`MLX_MAX_MB_PER_BUFFER`, `MLX_MAX_OPS_PER_BUFFER`,
`DARKBLOOM_STARTUP_MEMORY_PROFILE` and `MLXFAST_LOCAL_COOL_GATE` all unset, so
the no-overwrite install does take effect here; the risk the advisor raised is
that `env::max_mb_per_buffer` is a function-local `static`
(`utils.h:178-188`) first evaluated inside the `Device` constructor
(`device.cpp:596-597`), so any MLX Metal work before the install freezes the
defaults permanently.

### 8.4 Folding in the rest of the advisor's corrections

- **Engineerable deficit is ~0.26 %, not 0.561 %.** Halved throughout; §6's
  sizing statements should be read against 0.2586 %.
- **The cost-model knob axis is closed.** `headStepCostRatio`, the streak gate and
  the depth cap are all closed by two independent sources: the crown's published
  local sweeps (0.18->0.32 scored 2.846; 0.15 and 0.14 below baseline; streak gate
  1 scored 2.833; gate 0 tied) and our own 0.18->0.16 at -1.164 %. This r2 is
  consistent with that closure: it withdraws the cap prediction rather than
  re-issuing it, and §4 shows the model that produced it cannot order the central
  pair. I am recording the closure and not sweeping those knobs again.
- **Report per prompt on beagle and medicine only.** The reconstruction in §3 is
  already per prompt; the two that carry the score are beagle (mean M 5.533,
  83.5 % acceptance, 58.253 ms/round) and medicine (5.768, 87.5 %, 58.795).
  Aggregates over all eight are reported for the reconstruction's validation
  only, never as a decision input.
