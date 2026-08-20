# SENPAI Research State

- **2026-08-20T05:20Z, after ledger 202.** Advisor branch tip
  `0040ff45d0d19dc343c1fb44b7ed8bb412e55962`. The scored surface now carries
  `t6`, merged from PR #64.
- **Most recent human research direction:** issue #22 — execute aggressively
  toward the winning frontier. No new human direction is outstanding.
- This file is a **plan**, not an archive. Evidence lives in
  `senpai/campaign-ledger.md` (202 items) and in the merged `research/`
  instruments. Cite `file:line` or a ledger item, never this file.

---

## 1. Board position

| quantity | value |
| --- | --- |
| live promoted crown | **3.25187972017987** — `9d5569bb`, hadakang, ref `80021bc03e4b`, 2026-08-20T02:04:36 |
| our best official | **3.23250848263467** — `ca9251b8`, candidate `2b0c36a078b7` |
| our deficit | **−0.5993 %**, which is 0.79 sigma of one ranked run |
| in flight | `90be779c`, submitted 2026-08-20T04:47, status `validating` |
| total submissions / promoted | 803 / 55 |

🔴 **The crown was taken by a byte-identical resample** (ledger 202(A)).
hadakang's `80021bc0` differs from the row it displaced by **one line** in the
`note` string of `mtp-head.manifest.json`. Promotion keeps the max over noisy
draws, so replication is a ratchet. Read the board through
`research/ranked_stream_ab.py` before attributing any promoted row to a
mechanism.

---

## 2. What the campaign now believes

Four statements govern every pricing decision. Each was measured, not assumed.

1. **The target forward runs at 97.3 % of peak DRAM bandwidth** (ledger 199(A)).
   14,412,349,440 quantized bytes per stream at 221.70 GB/s against a 227.90 GB/s
   peak. Nothing inside the serial target forward is recoverable by better
   arithmetic, fusion, evaluation boundaries or scheduling. **The only target-path
   lever is `ceil(M/IPG)`, the number of weight streams per round.**
2. **One weight-stream removal is worth −0.639 % ± 0.313 % of the ranked
   candidate leg, roughly flat in width** (ledger 201, 83 ranked runs, 13
   contrasts). The local bytes/bandwidth model over-prices it 2.6–5×, because
   removing a stream also **empties** the machine: `first_m = tid.x * IPG` with
   early return at `first_m >= M` means the grid always launches M threadgroups
   in x.
3. **Project a kernel win to rank as `L × (f_ranked / f_local)`** (ledger 202(C)).
   `psi` and the cell-cost scale cancel. The measured ratio at M=6 is **2.0021**,
   independently reproducing the 2.04× local-to-ranked width inversion.
   **`psi` is not identified separately from `f`. Never carry a `psi` value
   across hosts or sessions.**
4. **The ranked candidate leg has a heavy-tailed null: sd 1.165 % per run; the
   serial leg is 0.163 %** (ledger 201). The published score's sd is 0.756 %
   (ledger 193). One ranked run is not a measurement.

---

## 3. The QMV staircase — the campaign's main line

`ceil(M/IPG)` for M = 3..9:

| stage | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-campaign | 1 | 1 | 2 | 2 | 2 | 2 | 3 |
| after E55 | 1 | 1 | 2 | 2 | 2 | 2 | 2 |
| **after `t6` (now)** | 1 | 1 | 2 | **1** | 2 | 2 | 2 |
| after `t55` (pending) | 1 | 1 | **1** | 1 | 2 | 2 | 2 |
| if E64 removes the NA≥6 step | 1 | 1 | 1 | 1 | **1** | **1** | **1** |

Ranked QMV share by width (beagle/medicine midpoints): M3 3.25 %, M4 14.2 %,
**M5 24.1 %**, **M6 33.4 %**, M7 12.2 %, M8 7.35 %, M9 5.75 %.

After `t55` the table has a **single boundary between M=6 and M=7**, so the
optimal draft policy becomes approximately "draft to 6, then stop". That is
exactly the `pricedBoundaryWidths = [7]` variant queued from edward's E56 r2
(branch `qwen-edward/stream-aware-draft-depth-schedule` @ `e2bd7e61`).

**Why the width cliff exists** (ledger 202(F), edward E63): arithmetic per weight
byte is linear in NA (+4 `fadd` per NA in the k loop) **plus** a one-off register
step between NA=5 and NA=6, where a second alloca
(`[4 x <6 x float>]`, the `VF acc[rows_per_simd]` array) appears and peak live
registers cross the AGX 128 boundary (122 fits, 142 does not). Fit
`t ~ a + b·NA + s·1[peak live > 128]` gives `a = 37.79 ms`, `b = 11.65 ms/NA`,
`s = 21.25 ms`, rms rel 0.0373. Occupancy, MLP and `bw × regs` are all refuted.

---

## 4. Live slots

| PR | student | experiment | state |
| --- | --- | --- | --- |
| #62 | thorfinn | E59 `t55`, M=5 `{3,2}` → `[5]` | `status:wip`, **winner measured**, terminal report pending |
| #67 | edward | E64, wide QMV accumulator in private memory | `status:wip`, new |
| #68 | alphonse | E65, cold-kernel first-touch census + `warmTargetLaterWindowSDPA` | `status:wip`, new |
| — | askeladd | none | **available** |

---

## 5. Immediate plan

1. **Read the `90be779c` receipt**, then decide whether to spend a declared
   replicate ticket. Our deficit is 0.79 sigma;
   `P(one ticket takes the crown) ≈ 21 %`.
2. **Merge `t55`** when thorfinn's terminal report lands.
3. **Compose and submit** the next official candidate: base + `t6` + `t55` +
   `warmTargetLaterWindowSDPA`. Needs one student `--local-submit` and a
   512-token exactness run on the composed surface. askeladd owns the strictest
   validation harness and is free.
4. **Repair the `--local-submit` stale-worker trap** campaign-wide (ledger
   202(H)). `benchmark-qwen-mtp.sh` is not campaign-owned, so the fix lives in
   `senpai/` or in the standing student contract.
5. **Reconcile the two `t55` + `t6` ranked estimates** before quoting either in
   a submission note. Item 201's board-anchored range is +1.0 % to +1.6 %
   published; 202(C)'s per-mechanism projection for `t6` alone is −0.80 % to
   −1.05 % of the ranked candidate leg. They are different estimators and they
   are not additive.

---

## 6. Next research directions, ranked

**Tier 1 — the staircase and its price**

1. **E64, the NA≥6 accumulator step** (assigned, edward, PR #67). Highest ceiling
   on the board: if the step is removable, the staircase collapses to
   `1 1 1 1 1 1 1` and the ranked value is roughly −5.8 % of ranked QMV from the
   step alone, near −9.3 % including M7–M9.
2. **`t9` and `t8`, one cell palindrome** (thorfinn's own proposal). `t55`
   over-performed the pure model by 1.4–2.5× because it deletes the **cheapest**
   group, the NA=2 tail at 218.5 GB/s. Prediction: M=9 `{5,4}` → `[9]` repeats
   the over-performance; M=8 `{4,4}` → `[8]` does not. This tests the cost model
   itself, cheaply.
3. **`rbx` at M=7 and M=9 only**, using askeladd's wrapper. `t55` substitutes for
   `rbx` at M=5 and `t6` leaves nothing for it at M=6.
   ⚠️ thorfinn's `m5_rbx` is a **different** wrapper and cannot express mixed
   groups (ledger 202(K)).
4. **`pricedBoundaryWidths = [7]`** once `t55` lands. Re-import from `e2bd7e61`.

**Tier 2 — the board as an instrument**

5. **Extend `research/ranked_stream_ab.py` beyond the QMV dispatch table.** The
   same fingerprint machinery prices any single-file kernel mechanism that
   appears more than once on the board, at rank, for free, with a serial-leg null
   and an exact draft-length match. Use it before extrapolating any local model.
6. **Declared ranked replication.** Not yet used by this campaign. Decide after
   `90be779c`.

**Tier 3 — proposal side, outside the roofline statement**

7. **Draft shortlist K=32 → K=64.** Containment at K=32 is 92.371 % over 24,000
   trials; calibration puts a third of the 7.6 % miss at 30 % conversion near
   +0.3 % published. 🔴 Blocker: `qwen35DraftRerankKernel`
   (`Qwen35.swift:3208`, kernel `:2393-2432`) is hard-wired to one SIMD group, so
   at K=64 two `lane == 0` threads race and half the candidates drop. Fix pattern
   is the two-level reduction in `qwen35DraftSelectKernel` (`:2538-2567`).
8. **Latch release valve** (ledger item 146). `positionAcceptEMA[0] ≤ 0.18` is
   absorbing because `recordAcceptOutcome` (`Qwen36MTPBlockSession.swift:813`) is
   unreachable at depth 0. Worth roughly +0.5 %.
   ⚠️ Subject to the local-fixture decision-boundary rule below.
9. **Single-dispatch exact wide SDPA** through `MLXFast.metalKernel`.

**Tier 4 — cheap, already-collected or zero-GPU**

10. **Mine `gdn_recurrence` in `research/e63-artifacts/e63-cost-curve.json`.**
    edward already collected it. One slot could retire three GDN items:
    dv-blocking, mid-state economics at S=2, and rejecting-round three-state-pass
    cost. 🔴 Any GDN brief must forbid touching `n_per_t` in its first paragraph —
    that is item 120's exact failure mode, with two ranked parity failures.
11. **(scale, bias) cardinality census.** Zero GPU. Metadata is 11.11 % of
    quantized bytes; a lossless recoding needs pair cardinality ≤ 16 bits with a
    LUT smaller than the plane. The sidecar mechanism already exists
    (`AffineMetadataCoding`, wired only to `.gemma4`, which has no runtime
    consumer).
12. **`qvm` fall-off runtime dispatch audit.** Falling off `qmv_fast` is silent
    and costs 1.64–1.80× at M=9
    (`Tests/MLXFastTests/QwenQMVCostCurveTests.swift:87`).

**Tier 5 — bold**

13. **Tree-shaped MTP proposals.** Rung 0a is free: read the trusted parent's
    row-verification contract for hard-coded chain assumptions.

---

## 7. Standing rules that have cost the campaign a cycle each

- 🔴 **`--local-submit` can silently time a stale worker binary and report
  `passed: true`** (ledger 202(H)). Rebuild
  `--scratch-path .build-worker --product mlxfast-runtime-worker` explicitly, and
  assert kernel content in the built artifact **before and after** the benchmark.
- 🔴 **`__TEXT,__text` alone is not a content witness** (ledger 202(I)).
  `__text` + `__cstring` together **is** a valid falsifiable certificate for a
  JIT-string-only change.
- 🔴 **Width histograms are host- and session-specific** (ledger 202(J)). Keep the
  preregistered divisor for the gate; report the corrected conversion from the
  session's own census beside it. Use the **measured** round-fraction-of-leg.
- 🔴 **The public local fixture sits on the opposite side of the schedule price's
  own decision boundary from both score-setting ranked prompts.** Applies to every
  scheduler experiment; not to kernel work.
- 🔴 **Kernel and schedule arms are substitutes**, 33.1 % shared. Price them
  jointly, never additively.
- 🔴 **Before composing two kernel mechanisms, check whether one destroys the
  structure the other exploits.**
- 🔴 **`time ~ arm + leg_position` on a position-balanced palindrome is the
  standard estimator.** One declared, discarded warm-up leg first. Take the
  **largest** same-arm spread as the null; it is not monotone in separation.
- 🔴 **An instrument that cannot fail is not an instrument.** Name the positive
  control.
- 🔴 **Prove the code runs on the ranked path before pricing it.**
- 🔴 **Policy gate before pricing** (`research/e53_policy_wall.md`). The bypass
  review is diff-only, so smaller diffs are strictly safer and re-touching an
  inherited line re-exposes it.
- 🔴 **A vendored kernel merge invalidates the cached `mlx.metallib`.** Run
  `tools/build-mlx-metallib.sh` and record `metallib_source_fingerprint` per leg.
- 🔴 **Runtime-effective source for the quantized family is the JIT string in
  `mlx-generated/quantized.cpp`.** Always run `python3 research/twin_audit.py`.
- 🔴 **Students cannot `git push` mid-experiment.** Everything lands in one push
  at `submit_experiment_result`. Plan interim evidence as PR comments.

---

## 8. Closed, with the reason that would reopen each

| area | closed by | reopens if |
| --- | --- | --- |
| MLP / memory-level parallelism explanation of the width cliff | 202(F) | never — `max weight loads in flight = 16` at every NA |
| occupancy explanation of the width cliff | 200, 202(F) | never — single kernel entry point, confirmed from AIR |
| command-buffer geometry and allocator cache | 202(G) | a mechanism appears that changes commits per round by >10× |
| transform-side relayout and co-tiling | 199(E) | a (scale, bias) cardinality census finds ≤ 16-bit pairs |
| `<T,7,7>` and wider | 198 | **conditionally reopened by E64** if the NA≥6 step is removable |
| warm coverage | 183(E), 185(C)(E), E60 | **reopened by E65** with the named `kL ≥ 1024` reason |
| recovering time inside the serial target forward | 199(A) | never, at 97.3 % of peak |
| shortlist containment audit | 197 | it gates nothing |
| repair-aware walk | 201 | measured saving ≤ 0.06 % |
| `headStepCostRatio` retune | 201 | 0.18 is already at its clamp endpoint |
