# SENPAI Research State

- 2026-08-19, after merging E55 (PR #57), deriving the single-weight-stream
  affordability table from the register census, and attempting an official
  submission that the guard refused.
  Campaign base `d2139c924c7a7d98ca6026eea63867c2776abbca`.
- Most recent human research direction: issue #22 -- execute aggressively toward
  the winning frontier. Issue #31 is complete and closed. No new human direction
  is outstanding.

---

## 🔴🔴 BLOCKED: we have a fully certified candidate and cannot submit it

**One human action unblocks the campaign's highest-value move: advance
`origin/main` so it contains `d2139c924c7a7d98ca6026eea63867c2776abbca`.**

Every pre-submit gate on the current base passes:

| gate | result |
|---|---|
| `python3 research/twin_audit.py` | OK, 29 runtime-effective twins |
| `senpai/verify-ranked-score-boundary.sh` | PASS |
| `senpai/check-editable-budget.sh` | OK, source 2458949/3000000, growth 4891/262144 |
| `senpai/validate-assignment-scope.sh` | OK, 4 submitted paths |
| 512-token PATH C exactness incl. post-EOS, row-ledger closure | PASS (askeladd, byte-identical surface) |
| gate-qualified `--local-submit` | PASS x3 behind the real 40 C gate |
| Yukon frontier check | FRONTIER UNCHANGED |
| duplicate check against our six rows | distinct |
| `mtp-head.manifest.json` and `mtp-head/` vs organizer main | byte-identical |

The guard then refuses:

```
official submit: BASE_SHA is not an ancestor of current origin/main
official submit: campaign history moved; no submission was sent
```

Cause, verified:

- `origin/main` = `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
- advisor branch `senpai/qwen38-mtp-r1` = `d2139c92`.
- merge base = `527306761f70e2c4024f347915328894db80c181` (2026-08-18 18:02).
- `origin/main` has **4 commits we lack**; the advisor branch has **881 commits
  `main` lacks**. This launch has never advanced `origin/main`.
- `senpai/submit-official.sh` hard-codes `SOURCE_BRANCH="main"` and asserts
  `git config yukon.source-branch == "main"` at line 146. There is no override.
- `origin/main`'s `senpai/frontier-state.json` is also stale (records promoted
  `0cd0a6b4` at `3.24929399`, not the live `59b321ee` at `3.24985583`).

**I did not bypass the guard and no agent on this campaign should.** No typed
tool publishes `main`; `publish_advisor_branch` targets only the advisor branch;
raw `git push` is a forbidden reproduction of a typed GitHub mutation; and a
fast-forward merge is impossible anyway because `main` holds four commits we do
not have. Recorded in full as ledger 194(D).

### The unblock is small, and I verified it with `git merge-tree` without mutating anything

Every other guard precondition already passes: `0c90733d` is an ancestor of
`upstream/main` `9e1ff9ec`; organizer `benchmark.json` equals campaign main's;
and the organizer trusted surface has not advanced (`0c90733d..9e1ff9ec` changes
exactly one file, `Qwen36MTPBlockSession.swift`, which is editable). **The
organizer side needs no sync. Only the campaign branch is stale.**

Merging the advisor branch into `main` conflicts on **exactly two files, and
neither is scored**: `senpai/campaign-ledger.md` and
`senpai/frontier-state.json`. Every file on the submitted surface auto-merges.

One silent trap. The auto-merged protected surface is not byte-identical to the
measured base: `quantized.cpp` differs by **comment text only**, because
`origin/main` still carries the stale "3+3+2, not 4+4" block above a call that
has read `<T, 8, 4, true>` for many rounds. The compiled kernel is unchanged, but
the guard's line-383 test is a textual `git diff --quiet`, so a merge resolved
only at the two visible conflicts would be refused again.

**Verified recipe.** Merge, resolve the two bookkeeping files in favour of the
advisor branch, then take the advisor branch's version of every protected path
(`git checkout d2139c92 -- benchmark.json <editablePaths...>`). The test that
must pass before pushing:

```bash
git diff --quiet <merged main> d2139c924c7a7d98ca6026eea63867c2776abbca \
  -- benchmark.json $(python3 -c "import json;print(' '.join(json.load(open('benchmark.json'))['editablePaths']))")
```

An empty diff means the guard will pass. The note is written and ready; after
that, submission is one command with every gate above already green.

---

## Board

| quantity | value |
|---|---|
| live promoted frontier | **3.24985583421771** (submission `59b321ee`, solver fkiene, source `9e1ff9ec` = `upstream/main`) |
| organizer main `0c90733d` | **3.24929399** (submission `0cd0a6b4`, solver ofou) |
| organizer main's own identical-tree replicate `dc70080f` | **3.22945266** (**0.6144 % below its own twin**) |
| our best official submission | **3.23250848263467** (receipt `ca9251b8`, candidate `2b0c36a0`, rejected on score) |
| our deficit to the frontier | **0.5366 %** = **0.71 sd of one ranked run** |
| **sd of ONE official ranked run** | **0.756 %** (18 identical-surface groups, 44 rows, dof 26) |
| **sd of a difference of two ranked runs** | **1.069 %** |
| **ranked MDE, single (S, S^) pair** | **+2.10 %** |
| local end-to-end null floor | **0.0629 %** (**17x more sensitive than ranked**) |
| total board submissions / promoted | 773 / 54 |

### We are not behind the frontier. We are at it, inside the instrument.

Ledger 193 measured the official instrument on pairs of ranked runs whose
**submitted surface is byte-identical**, keyed on the git tree and never on the
announced commit SHA (which recovers zero groups from 512 scored rows).

- Median disagreement of two runs of the **same tree**: **1.113 %**.
- **51.4 %** of identical-surface pairs disagree by more than **1.00 %**.
- Five pairs have a literally **empty** diff and scored
  `+0.0081, +0.1556, -0.6106, -0.6786, -1.2737 %`. One of them was **promoted**.
- The noise sits on the **candidate** leg (median pair delta 0.589 %), not the
  pinned serial leg (0.181 %), a ratio of **3.62x**. The pinned baseline does not
  drift: `+0.000058 %/h` over 109.8 h, `t = 0.48`.

**Organizer main resubmitted its own byte-identical tree and scored 0.6144 %
lower, which is more than our entire deficit.**

P(our candidate outscores the crown on one run), by its true improvement over
the crown: `0 % -> 49.1 %`, `+0.25 % -> 62.1 %`, `+0.50 % -> 73.7 %`,
`+1.00 % -> 90.1 %`, `+1.66 % -> 98.4 %`.

Retracted by 193 and still retracted: the `+0.283 %` ranked MDE (wrong by 7.4x);
the E27 board receipt reading a `-0.3321 %` register-ceiling cost (one pair
against `sd = 1.069 %`); ledger 181(D)'s claim that the frontier's advantage is
one untimed warm. `research/ranked_noise.py` is the single authority.

---

## 🔴 The live scientific lead: single weight streams in the wide QMV table

E55 (merged, PR #57) replaced the two-weight-stream call at `case 9` with the
single-stream `<T,9,5>` form in both runtime-effective twins. Measured
**-4.2952 %** on the MTP leg against a `+0.0497 %` null, bitwise exact at 512
tokens including post-EOS.
W&B [`wxezisvs`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/wxezisvs),
[`f4ej9y1n`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/f4ej9y1n),
[`o8ig3ht7`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/o8ig3ht7).

The template's second parameter is `IPG`, inputs per group, and each x-group
re-reads the whole weight matrix, so **group count is weight stream count**.
Askeladd's register law, `reg = 20 + 21*max(NA) + 4*[two distinct NA sizes]`,
reproduces every shipped cell and both students' measurements with no fitted
parameter.

| M | shipped | groups | regs | one-group form | regs | affordable under the new 129 max? | predicted cell delta |
|---|---|---|---|---|---|---|---|
| 3 | `<T,3,3>` | 1 | 83 | already one stream | - | - | - |
| 4 | `<T,4,4>` | 1 | 104 | already one stream | - | - | - |
| **5** | `<T,5,3>` | 2 | 87 | **`<T,5,5>`** | **125** | **YES** | **-20.15 %** |
| 6 | `<T,6,3>` | 2 | 83 | `<T,6,6>` | 146 | no, +17 | -9.95 % |
| 7 | `<T,7,4>` | 2 | 108 | `<T,7,7>` | 167 | no, +38 | +4.16 % |
| 8 | `<T,8,4>` | 2 | 104 | `<T,8,8>` | 188 | no, +59 | +28.22 % |
| 9 | `<T,9,5>` | 1 | 129 | shipped by E55 | - | - | measured -4.30 % leg |

Break-even against E54's measured lone-group bandwidth ladder
(`223.8 / 199.7 / 175.2 / 150.9` GB/s at `NA = 2..5`; differences
`-24.1 / -24.5 / -24.3`, so near-linear across its measured span):

| M | needs | flagged extrapolation | verdict |
|---|---|---|---|
| 5 | `bw(5) > 120.49` | `150.9` **measured** | PASS |
| 6 | `bw(6) > 114.00` | `126.6` | PASS, margin 11 % |
| 7 | `bw(7) > 106.55` | `102.3` | FAIL |
| 8 | `bw(8) > 100.01` | `78.0` | FAIL |
| 9 | `bw(9) > 92.56` | `53.7` | FAIL |

Two consequences drive the current assignments.

1. **`<T,5,5>` is a one-character diff per twin and is already paid for.** It
   needs no helper change (E55 already raised `NA <= 5`), costs 125 registers
   which is below the 129 the table now pays, and lands on a cell carrying
   **19.4 % to 26.4 %** of ranked QMV time on the two prompts that set the
   published median. Assigned to thorfinn as a new rung-4 arm.
2. **`bw(6)` is the single measurement that closes M=6 through M=9 permanently.**
   Assigned to askeladd as E61 rung 1, with an explicit stop gate at
   `bw(6) <= 114.00`. M=6 is the **largest** ranked QMV width (30.9 % to 34.7 %
   of QMV time), so it is worth the direct measurement even though its one-group
   form needs 17 registers above the current max.

`m5_rbx`'s motivation is retracted, not lost. Before E55 the table max was 108
and `<T,5,5>` at 125 would have raised it; now the max is pinned at 129 by
`case 9`, so if the allocator is per-kernel-max, `m5_rbx` (90) and `<T,5,5>`
(125) must time identically. That is now a free falsifier with a preregistered
tolerance.

---

## Ranked mechanism census: what the board knows that we do not

A frontier-tier agent reconstructed 749 ranked mechanism records from the public
board (`_advisor_scratch/ranked-mechanism-ledger.md`).

- **Both of our failed rows died at the same gate**, "Review submitted code for
  benchmark bypasses (Qwen-MTP policy)", the second most common killer on the
  board. Our current candidate leaves the head manifest and `mtp-head/`
  byte-identical to organizer main, which closes that specific failure mode.
- **Exactly one ranked-measured mechanism is genuinely absent from our tree**:
  `3a7f09f4` (+0.260 %), a one-line change from
  `eval(cache.flatMap { $0.state } + bundle)` to `eval(bundle)` in
  `generateRound`. The line survives verbatim at our
  `Qwen36MTPBlockSession.swift:1123`. Queued as a one-line arm.
- **Our edits deleted no ranked-measured winner.** Same-parent differencing put
  the other candidates negative: DIRECT_NIBBLES `-0.894 %`, bespoke m7 3+2+2
  `-0.841 %`, MTP-head norm transplant `-0.163 %`.
- **Self-criticism.** Ledger 14 closed crossrow QMV on a claimed "~1 % ceiling".
  Rivals then banked roughly `+3` to `+4` cumulative points in that vein. Under
  193 each step is inside the noise, so we did not lose a specific measured gain;
  we abandoned a vein others kept mining. E55 and E61 are us returning to it.

---

## Current research focus

1. **Get `origin/main` moved and submit `d2139c92`.** It is certified and idle.
   Under the 193 instrument this is worth more than any single mechanism now in
   flight.
2. **Finish the single-weight-stream sweep of the QMV width table.** This is the
   only direction with a measured `-4.30 %` leg result behind it and a
   zero-parameter model that predicts where the next win is.
3. **Attack decode-side host cost.** E29 measured about `2.4 ms` per draft step
   of host graph build, roughly `4.35 %` of decode, and nothing has ever acted on
   it. `CompiledDecode.swift` and `CompilableKVCache.swift` are editable and have
   zero ledger mentions.

## In flight

| PR | student | experiment | state |
|---|---|---|---|
| #64 | qwen-askeladd | **E61** single weight stream at M=6, and the first direct price of the register ceiling | assigned |
| #63 | qwen-alphonse | **E60** composite against organizer main, warm-matched to the frontier | running |
| #62 | qwen-thorfinn | **E59** M=5 route, now with `<T,5,5>` as a rung-4 arm | rungs 2b and 3 in flight, rung 4 redesigned |
| #59 | qwen-edward | **E56** stream-aware draft-depth schedule | revision `e56-r2`, must merge `d2139c92` first |

Merged this campaign: #57, #55, #53, #52, #56, #60, #58, #61.

## Next research directions

Ordered by expected value against the `0.5366 %` deficit.

1. **Submit `d2139c92`** once `origin/main` moves. Blocked, not deprioritised.
2. **Generalised x-group `rbx` wrapper at M=5 and M=9.** Thorfinn's `m9_rbx4`
   reaches the E55 schedule at 95 registers instead of 129, and `m5_rbx4` reaches
   the M=5 single-stream schedule at 91. If the ceiling has any real cost, this
   recovers the 108 floor while keeping both wins. If E61 rung 4 shows the
   ceiling is free, this direction closes cleanly.
3. **The `3a7f09f4` one-line arm** at `Qwen36MTPBlockSession.swift:1123`. The
   only genuine absent ranked-measured mechanism. Cheap, needs exactness because
   it interacts with rollback and replay.
4. **`mx.compile` the head draft chain / remove the E29 host graph build.**
   `+0.5 %` to `+2.0 %`. The head is fc plus one full-attention layer plus norms,
   an eligible shape; the full target is ineligible because of its 48 recurrent
   layers.
5. **Certified exact target LM-head screening**, `+2.0 %` to `+2.9 %`. An
   offline input-independent conservative bound plane screens rows of the
   248,320-entry readout; survivors get exact affine-4 with identical per-row
   arithmetic, so the top two are bit-identical. Lands in a new kernel library so
   it has zero interaction with the register table. Free first rung: dump traced
   hidden vectors, compute the plane in Python, measure survivor density. Step 0
   is resolving the `run-submission-static-review.sh:453` NVFP4-envelope
   ambiguity.
6. **Hierarchical certified shortlist for the head's coarse readout**, `+1.5 %`
   to `+2.2 %`. The flat 2-bit scan over 98,336 compact-vocab rows is about 40 %
   of a head step, and the step is about 86 % pure weight streaming at a
   saturated 243 GB/s, so byte cuts transfer nearly 1:1. Changes zero weights, so
   it is not the closed head-replacement direction.
7. **Latch release valve (ledger 146).** `positionAcceptEMA[0] <= 0.18` is an
   absorbing state and `recordAcceptOutcome` is unreachable at depth 0, so a
   prompt that latches never recovers. Simulated `-14.55 %` to `-18.02 %` if it
   hits a bankable prompt at a `~3.2 %` base rate, which is about `+0.5 %`
   expected score per submission as tail insurance at zero exactness risk.
8. **E56 x E59 2x2 factorial.** The two are **substitutive, not additive**: E59
   sets `streams(M5) = 1`, which deletes the 4-to-5 boundary that edward's
   attribution says produces most of E56's gain. Naive summation overstates the
   composite by about `2x`. Assign once both terminal results land.
9. **Smaller command buffers.** E58 falsified *larger* buffers and showed buffer
   boundaries are pipelining opportunities. Fewer than 50 operations per buffer
   is untested end to end. Do not extrapolate E58's one-directional slope.
10. **Composition vehicle for the exact sub-MDE wins**, `+0.2 %` to `+0.5 %`,
    near zero risk: `pendingPrimaryDevice`, dead-KV-GEMM elision, fused
    last-merge plus final RMSNorm, top-32 finalize k-way merge. One PR, one hunk
    per mechanism. Hand-apply hunk by hunk; never file-copy.
11. **Single-dispatch exact wide SDPA** via `MLXFast.metalKernel` at the editable
    chunk site, to lift the 32-lane wall.
12. **GDN rollback economics.** The tape write and `replayRecurrentPrefix` are
    inherited and have never been measured. The scan writes about 600-700 MB per
    round of fp32 mid-states that are discarded on the ~44 % of rounds that fully
    accept.

Deliberately not proposed, with reasons: single QMV cells outside the
single-stream sweep (187(K)); warm coverage (183(E), 185(C)(E)); seed prefill,
scored but unreachable on our gen-16 host (186(C)); SDPA chunk removal, which is
a discount and must be kept (185(B)); KVBuffer (180(D)); head weight
replacement, twice rejected at rank; `MLX_MAX_OPS_PER_BUFFER` enlargement, now
falsified; moment-based board arithmetic (184(D)); `MLX_METAL_GPU_ARCH` nax-off,
which fails exactness by construction because it changes prefill GEMM rounding
and so perturbs the reported top-two evidence.

---

## Standing method rules

- 🔴 **`origin/main` is the branch `senpai/submit-official.sh` trusts.** It is
  currently diverged and blocks every submission. Do not bypass the guard.
- 🔴 **Group count is weight stream count** in the wide QMV template, and
  `reg = 20 + 21*max(NA) + 4*[two distinct NA sizes]` reproduces every shipped
  cell.
- 🔴 **Decide locally, submit to claim.** Ranked is 17x coarser than a local ABBA
  pair. No official submission can validate a mechanism worth less than ~2 %.
- 🔴 **Cadence beats mechanism size at the frontier**, and selection bias means
  `E[observed | true 0, observed > 0] = +0.60 %`. Do not rank rival mechanisms by
  the size of the step that promoted them.
- 🔴 **Measurement power is not payoff.** A ranked run is a poor instrument and
  the only one that pays.
- 🔴 **The ranked replicate key is the git tree of the submitted surface**, never
  the announced commit SHA. Promotion is exactly
  `git merge-base --is-ancestor upstream/submissions/<uuid> upstream/main`.
- 🔴 **Never read mode structure from a deviation about a group extremum.**
  Centre on the mean or use pair deltas.
- 🔴 **Verify the arm from the artifact under the clock**, not from the process
  that produced it.
- 🔴 **Integrate a moved base by merge, not rebase.** A rebase breaks
  published-head ancestry and blocks submission.
- **Command-buffer geometry is part of the experiment identity tuple.** Export
  all three of `DARKBLOOM_STARTUP_MEMORY_PROFILE=full`,
  `MLX_MAX_MB_PER_BUFFER=512`, `MLX_MAX_OPS_PER_BUFFER=50`, and prove it by the
  **absence** of `mlxfast-worker: low-memory startup profile engaged`.
- **A local cost curve is not a ranked cost curve.** Edward measured this host at
  `2.4x` the ranked per-row charge, so a depth-cutting mechanism flatters itself
  locally. A local width histogram is likewise not the ranked width mixture:
  `M = 9` is 53 % local against 3-10 % ranked.
- **A local win measured on the leg has already paid the local prefill
  dilution.** Apply scalars to the leg gain, then price once. Keep leg-reduction
  and `raw_p`-change in separately named functions; all five recorded advisor
  pricing errors are basis confusions, not arithmetic.
- **`--local-submit` at 128 tokens is a worse ranked proxy than `--local-iterate`
  at 512** (56.2 % against 23.4 % prefill). Its score must not enter a pricing
  chain.
- **Check headroom before pricing a per-prompt gain.** A gain above the next
  order statistic buys nothing.
- **Never extrapolate a two-point fit outside its anchor interval**, and mark
  every extrapolated value as extrapolated.
- **A result failing the advisor's gate can be worth more than one that passes.**
  A student deviation that makes a control non-tautological is correct.
- **An instrument that cannot fail is not an instrument.**
- **Ungated timing only** ABBA-counterbalanced, with entry and exit temperatures
  recorded and `cool_gate_passed_real_gate=false` plus
  `gate_qualified_for_timing=false` preserved verbatim.
- **Log W&B per leg while timing**, never at session end.
- **Always run `python3 research/twin_audit.py`** after touching Metal source;
  the runtime-effective source for the quantized family is the JIT string in
  `mlx-generated/quantized.cpp`.
