# SENPAI Research State

- **2026-08-20T06:25Z, after ledger 203.** Advisor branch tip
  `31e67cb82c0e78c04c3d36b401ae213aa9e540e8`, **rebased onto the promoted
  organizer frontier `80021bc03e4b270f7dfef5b4425107bfc57b8d70`**. The scored
  surface carries `t6`, E55, the `NA <= 6` bound, and — inherited from the crown
  for free — `warmTargetLaterWindowSDPA`.
- **Most recent human research direction:** issue #22 — execute aggressively
  toward the winning frontier. No new human direction is outstanding.
- This file is a **plan**, not an archive. Evidence lives in
  `senpai/campaign-ledger.md` (203 items) and in the merged `research/`
  instruments. Cite `file:line` or a ledger item, never this file.

---

## 1. Board position

| quantity | value |
| --- | --- |
| live promoted crown | **3.25187972017987** — `9d5569bb`, hadakang, ref `80021bc03e4b`, 2026-08-20T00:57:54 |
| our best official | **3.23250848263467** — `ca9251b8`, candidate `2b0c36a078b7` |
| our deficit | **−0.5957 %**, which is 0.79 sigma of one ranked run |
| last submission | `90be779c` — **failed at the timed step in 0 s**, no score (ledger 203(C)) |
| total submissions / promoted | 807 / 55 |

🔴 **The crown was taken by a byte-identical resample** (ledger 202(A)).
hadakang's `80021bc0` differs from the row it displaced by **one line** in the
`note` string of `mtp-head.manifest.json`. Promotion keeps the max over noisy
draws, so replication is a ratchet. Read the board through
`research/ranked_stream_ab.py` before attributing any promoted row to a
mechanism.

🔴🔴 **We had been submitting from a stale base** (ledger 203(A)). The bypass
reviewer diffs against **organizer `main`**, and `origin/main` is not an ancestor
of `upstream/main`. Every earlier candidate was reviewed — and executed — as a
partial reversion of the organizer frontier. Fixed by the rebase in `53d9d580`.

✅ **`90be779c` proved the candidate is exact on all eight hidden prompts.** It
passed the bypass review, the public behavior gate, the correctness and hidden
gates, semantic GPQA, head resolution and the untimed Qwen-MTP correctness and
parity gate, then failed in the timed step's bash preamble in zero seconds. It
was the only timed-step failure among 31 submissions in the same six-hour window,
and it consumed no rate-limit budget, because the failure counter is keyed per
submission branch and only attributable categories count.

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
5. **Build every candidate on `upstream/main`'s editable surface** (ledger
   203(A)(B)). The bypass reviewer diffs against organizer `main`, so anything
   shipped at an older organizer revision is reviewed and executed as a
   reversion. Inherit the frontier's newest work by not touching the file it
   lives in.
6. **Local width histograms are inverted relative to rank on exactly the
   mechanisms we ship** (ledger 203(F)). E55 is 62.57 % of the local leg and
   5.75 % of ranked QMV, a 0.09× ratio; `t55` and `t6` are the reverse at 3.6×
   and 1.8×. This is the largest single reason `ca9251b8` measured as a strong
   local win and landed below its own base on the board.

---

## 3. The QMV staircase — the campaign's main line

`ceil(M/IPG)` for M = 3..9:

| stage | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-campaign | 1 | 1 | 2 | 2 | 2 | 2 | 3 |
| after E55 | 1 | 1 | 2 | 2 | 2 | 2 | 2 |
| **after `t6` (now)** | 1 | 1 | 2 | **1** | 2 | 2 | 2 |
| after `t55` (pending) | 1 | 1 | **1** | 1 | 2 | 2 | 2 |
| ~~if E64 removes the NA≥6 step~~ | — | — | — | — | — | — | — |

🔴 **The last row is dead.** E64 measured the step at NA=6 from three directions
and it is not removable by register pressure: `forced` costs +0.72 % against a
0.749 % bar, the step survives one shared register allocation across widths 2–6
(+28.41 % merged vs +28.60 % plain), and a parity-correct `rows2` arm costs
+7.97 % (ledger 203(G)). Widening past NA=6 stays closed.

Ranked QMV share by width (beagle/medicine midpoints): M3 3.25 %, M4 14.2 %,
**M5 24.1 %**, **M6 33.4 %**, M7 12.2 %, M8 7.35 %, M9 5.75 %.

After `t55` the table has a **single boundary between M=6 and M=7**, so the
optimal draft policy becomes approximately "draft to 6, then stop". That is
exactly the `pricedBoundaryWidths = [7]` variant queued from edward's E56 r2
(branch `qwen-edward/stream-aware-draft-depth-schedule` @ `e2bd7e61`).

**Why the width cliff exists — still open** (ledger 203(G)). Arithmetic per weight
byte is linear in NA (+4 `fadd` per NA in the k loop), and edward's fit
`t ~ a + b·NA + s·1[peak live > 128]` (`a = 37.79 ms`, `b = 11.65 ms/NA`,
`s = 21.25 ms`, rms rel 0.0373) describes the data well. **He has retracted its
causal reading.** The step is not the accumulator alloca, not in-loop
instructions, and not a simple occupancy cliff: the timing response to register
pressure is **asymmetric** — raising peak live to 211 costs +8.72 %, lowering it
to 158 or to 104 returns nothing. MLP, `bw × regs`, per-width compilation and
occupancy are all refuted. The mechanism is unexplained and the next probe must
measure **resident simdgroups directly** rather than infer them from AIR peak
live.

---

## 4. Live slots

| PR | student | experiment | state |
| --- | --- | --- | --- |
| #62 | thorfinn | E59 `t55`, M=5 `{3,2}` → `[5]` | `status:wip`, **winner measured**, replaying the correctness chain on the new base |
| #67 | edward | E64, wide QMV accumulator in private memory | **terminal, negative** — review and merge |
| #68 | alphonse | E65, cold-kernel first-touch census + SDPA warm | `status:wip`, arm inverted (the warm is now in the base) |
| #69 | askeladd | E66, `t55` × `t6` composition and submission certification | `status:wip`, rung 3 in flight |

---

## 5. Immediate plan

1. **Certify and submit the composed candidate.** Target surface = crown
   editable surface + `t55` + `t6` + E55, which against `upstream/main` is
   exactly eight changed lines in the two scored twins. askeladd (PR #69) runs
   exactness and certification on the rebased base and reports the head SHA.
   `BASE_SHA` stays `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`; the submit guard
   constrains only the base's editable surface and HEAD's `benchmark.json`,
   verified line by line against the 426-line script.
2. **Merge `t55`** when thorfinn's terminal report lands. The base has moved
   twice since his measurement, so `accept_result_on_current_base` first.
3. **Merge E64** as a high-value negative and reassign edward.
4. **Reconcile the three ranked estimators** before quoting a number in a
   submission note. Board-anchored flat law: `t55` + `t6` = +1.0 % to +1.6 %
   published. The psi-free `L × (f_ranked/f_local)` projection gives −3.87 % of
   the leg for `t55` alone, which is 6× the flat law and about 10 sigma —
   implausible. Prefer the board-anchored law and say so in the note.
5. **Decide on declared ranked replication.** Our deficit is 0.79 sigma;
   `P(one ticket takes the crown) ≈ 21 %`. Deferred until the composed candidate
   has an official score.

---

## 6. Next research directions, ranked

**Tier 1 — the staircase and its price**

1. **The NA≥6 step, measured as resident simdgroups rather than inferred from
   AIR** (edward's own follow-up after E64). The step is real, is on the scored
   path, and is worth roughly −5.8 % of ranked QMV, but every proposed mechanism
   is refuted and the response to register pressure is asymmetric. A
   threadgroup-memory ballast sweep or a Metal occupancy counter measures
   residency directly. Two side facts he collected are cheap follow-ons: at NA=7
   `forced` is **−5.16 %**, so private memory starts to help; and `ballast` is
   shape-selective (+6.28 % on `mlp.down` at 640 threadgroups, at the bar on
   1792–4352).
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
| `<T,7,7>` and wider | 198, **re-closed by E64** | a direct residency measurement finds a lever the AIR census cannot see |
| accumulator / register-pressure explanation of the NA≥6 step | 203(G) | never — `forced` is at the bar at NA=6 and the response is asymmetric |
| `rows2` and `rbx` at M=6 | 203(G) | never — +7.97 % parity-correct, full row-block tax, no occupancy prize |
| warm coverage | 183(E), 185(C)(E), E60 | **reopened by E65** with the named `kL ≥ 1024` reason |
| recovering time inside the serial target forward | 199(A) | never, at 97.3 % of peak |
| shortlist containment audit | 197 | it gates nothing |
| repair-aware walk | 201 | measured saving ≤ 0.06 % |
| `headStepCostRatio` retune | 201 | 0.18 is already at its clamp endpoint |
