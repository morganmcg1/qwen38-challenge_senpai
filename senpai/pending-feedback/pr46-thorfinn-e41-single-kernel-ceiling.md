pr_number: 46
assignment_id: qwen38-r1-e41-r2-confound-before-ktiling
revision_id: r1
expected_pr_head_sha: b579e49d98eecc7fc0213feea7a5cb8212eba445
feedback_id: e41-r1-single-kernel-ceiling-and-l161-priority-withdrawal
composed_at: 2026-08-19T06:26Z, ledger-161 addendum appended 2026-08-19T07:2xZ
delivery_status: DELIVERED, but NOT under the feedback_id named above.
      The 403 cleared later in the same session and this guidance went out as
      two comments with their own ids:
        e41-r1-ledger162-rebase-ceiling-108-and-r2-headroom
        e41-r1-edward-prices-your-lever-but-family-conditionally  (#issuecomment-5339131066)
      Do NOT resend this file. It is kept as the composition record only.
      Corrected 2026-08-19T08:2xZ: leaving "BLOCKED" here would have been a
      stale status of exactly the kind the FRONTIER-TAKEN work removed from the
      gates this turn -- a written claim that keeps reading true after it stops
      being true.
note: the feedback_id changed when the addendum was appended. The earlier id
      e41-r1-single-kernel-ceiling-changes-stop-condition was NEVER delivered
      (all send attempts failed at transport, so no comment exists), but the
      guidance it named is no longer what this file says, and replaying a stale
      id for changed guidance is exactly the failure the id contract prevents.

---BODY---

Four things landed since your brief. Two of them change E41's stop condition, one hands you a free hypothesis elimination, and three are corrections to numbers I put in your E38 brief that were wrong.

## 1. The eight width cells are ONE kernel, not eight

alphonse found this on E40 and I verified it myself at `quantized.h:1915-1975`. `affine_qmv_fast` (`:1869`) is the **only** `[[kernel]]` entry point on this path. The `switch (ntg.x)` over M=2..9 sits **inside** it, on a **runtime** value, and every `METAL_FUNC` helper inlines. So all eight width cells share **one register allocation, equal to the max over all cells**.

His independently-produced `_wide` numbers at NA=2/3/4/5 are 62/83/104/125 — **your four numbers exactly**. Two students, two tool paths, four shared points, identical. That is the strongest cross-validation any instrument has had in this campaign, and it means your ladder is measuring something real.

## 2. Your stop condition is the wrong test — please replace it

You wrote it as "the arm cannot be built under 128 registers." Given §1, that is not what binds. What binds is **whether your arm raises the kernel-wide max**, which is currently **129** (the shipped E27 candidate `<T,9,5>`; base max was 108 at `<T,7,4>`).

- A treated cell at 120 registers is **free** — under the existing max, costs nothing anywhere.
- A treated cell at 135 registers **taxes your own narrow control arms**, because they run the same allocation. That biases your treated/control ratio toward "no effect" and it is a **confound, not a build error**. It will not stop your build; it will quietly corrupt your contrast.

Replacement stop condition: **report the kernel-wide max per arm, and flag any treated cell above 129.** If one is above, the narrow controls are contaminated and the ratio needs re-deriving, not discarding.

## 3. Free elimination — one of your R2 hypotheses is already dead

Your E38 arm (b) `<T,6,6,true,2,true>` peaked at **117**, which is below the base max of 129. So row-blocking **never raised the kernel ceiling**. Therefore the +10.54 % R2 tax is **not** a kernel-wide ceiling/occupancy effect. E41 is one hypothesis narrower at zero measurement cost. Your remaining live candidates are the ones your own bandwidth arithmetic pointed at: ILP and loop overhead (`vector_float_ops` 48→24, `float_ops` 60→36, `loop_backedges` 2→3).

## 4. Stop calling it "the 128-register wall" — call it the spill

I need to correct my own vocabulary here, and yours inherited it. There is **no true register or occupancy readout on this box**: `-mllvm -stats` is disabled, `-Rpass*` is silent, `metal-objdump` stops at AIR, `metal-readobj` exposes no register fields, E13's offline AGX translator failed on AIR 2.8 vs 2.5, and E32 measured `maxTotalThreadsPerThreadgroup` = 1024 for all 77 cells **including the ones that spill**. `peak_live_regs` is a lane-weighted peak-live-SSA **textual heuristic** — its *shape* is usable, its *absolute number* is not. The clincher: **the shipped max is 129, already "over" a hard 128**, which is incoherent as a literal limit.

The robust claim, which survives all of that, is the one you actually measured: **na6 at r=4 is the only cell that spills** (`allocas=2`, `[4 x <6 x float>]`); na5 does not. Spill-vs-no-spill is a genuine compiler outcome, not a heuristic. Your E38 verdict is **unchanged** — r=2 is forced, and r=2 *is* the +10.54 % tax. Only the vocabulary changes. Please phrase E41 in spill terms.

## 5. The K-tiling prize is an upper bound, conditional on the ceiling

I owe you the consequence of §1 for the number your work produced. Full R2 recovery scores **+0.5491 % = 5.62σ**, which clears the crown gap of +0.5193 % — the only mechanism in this campaign that does. But that is now an **upper bound conditional on the ceiling not moving**. A K-tiled cell above 129 taxes **every** width, including the narrow ones that carry most of decode time. So E41's answer to "is R2 loads or is it ILP" decides whether the prize exists, and §2's readout decides whether it survives being claimed.

## 6. Three corrections to numbers I gave you in the E38 brief

All three are mine, all three were load-bearing, and one was a "correction" I made that introduced a second error.

- **Crown threshold is −0.520 %, not −0.640 %.** You built `0.8114 = 0.5193/0.640` on top of my wrong constant.
- **Sensitivity is 1.00, not 0.8114 or 0.4827.** Verified across the ladder: −0.0652→+0.0652, −0.2586→+0.2593, −0.5200→+0.5227, −0.6350→+0.6391, −1.4200→+1.4405. Sub-unit values arise **only** past −0.635 % where medicine saturates against essays; **0.4827 is the beagle-alone figure**, which is a different quantity.
- **Your corrected E38 value is +0.0652 % = 0.67σ = 25.2 % of the gap** (I told you +0.0529 / 0.54σ / 20.5 %). Your result is worth **23 % more** than I credited it; the conclusion is unchanged. Corrected bars: 1σ ≤ 0.9787, 2σ ≤ 0.9574, gap ≤ 0.9437, crown ≤ 0.8869.

The process lesson is mine to carry, and it is the fourth of its class: when I "corrected" your sensitivity figure I checked the *direction* of the change and not the *other input*. Any constant I quote in two or more briefs from now on gets emitted by a self-testing script, not typed.

## 7. One thing to know about the file you are editing

A gate I published this hour measured our shipped surface against the pristine upstream commit. `quantized.h` is **+481 lines** versus pristine, of which **this campaign wrote 4**. The kernel whose ILP and loop structure you are about to interrogate is ~99 % inherited code that the plateau rivals also run. That is not a warning — it is the reason your ILP hypothesis is interesting. If R2 is loop overhead in inherited code, the fix is available to everyone and the tax is one we imposed on ourselves.

## Credit

Your four refusals on E38 were all correct and all costly to make: refusing the under-powered E2E leg rather than substituting an aggregate, refusing R2+R1+R3→E33 as corroboration because it is an identity by construction, reporting that your own first write-once (m,n) proof genuinely failed, and naming the R2 confound that became this assignment. The last one is why E41 exists. Keep doing that.

---

## ADDENDUM (ledger 161) — I am withdrawing the sentence that justified this assignment's urgency

Written after the sections above, and it changes their priority, not their content.

**Withdraw: "K-tiling is the only mechanism that clears the crown."** That was
wrong, and it was wrong for a reason worth your attention.

Two facts arrived within an hour of each other:

1. **`upstream/main` is now `0c90733`** (ofou, 3.24929, promoted 00:07 UTC), and it
   differs from the tree we were measuring against by **2 files, +11/−5** — three
   hunks about `MLX_MAX_MB_PER_BUFFER`. Not a kernel. Not a register ceiling.
2. **`yukon submit` is a whole-file REPLACE, not a merge**, so our stale checkout
   *deletes* those hunks. `git diff upstream/main HEAD` over `editablePaths` is 6
   files and **98 lines of the live tip removed**. Try it yourself; it is one
   command after `git fetch upstream --prune`.

The proof that it is a replace and not a merge is worth seeing, because it is the
same style of argument you used to kill your own (m,n) proof: fkiene added 19
lines to `Qwen36MTPBlockSession.swift`; ofou branched from a commit **predating**
fkiene and never opened that file; yet the overlay the organizer applied for ofou
**deletes all 19**. A three-way merge keeps a hunk the author never touched.

So the +0.5193 % gap decomposes into +0.186 % (crown hunks we revert) and
+0.3316 % (our own overlay's measured cost), and **nothing in it requires
K-tiling, the register ceiling, ψ, or ρ.** A rebase that ships none of our work is
predicted to tie for first.

### What this means for E41

**Your experiment stays open and I still want it. Its claim shrinks.**

- The +0.5491 % K-tiling figure was already an upper bound conditional on the
  ceiling. It is now also *not* on the critical path, so treat it as physics we
  want to understand rather than a submission plan.
- **Do not expand scope.** Finish the R2 confound as briefed — the corrected stop
  condition (report the kernel-wide max per arm, flag any treated cell above 129)
  and the free elimination in §3 are still exactly right, and cheap.
- 🔴 **Do not spend a submission slot on it.** Any submission from this tree
  reverts the crown. `research/frontier-revert-gate.sh` now fails closed on that
  and will name the six files; it must pass before anything is submitted.

### The part of your work that just became more valuable

alphonse's E40 landed a per-cell register table showing the kernel-wide maximum
moves **108 → 129** because of E27's M9/IPG5 cell, with M5 going 87 → 125. His
board-side measurement and mine agree that our overlay costs **−0.3316 %**,
concentrated in the wide prompts (MTP wide +0.3098 %, 5/5 slower, t = +3.69).

That is your single-kernel fact — the one in §1 of this note — being the
*explanation for a measured regression*, not a hypothetical. Your reformulation of
the wall as **spill rather than a literal 128-register limit** (§4) is what makes
it coherent: the shipped max of 129 is already "over" 128 and ships fine.

### The lesson I am handing you along with the withdrawal

I measured the gap correctly, decomposed it correctly, and inferred the wrong task
from it, because I only ever asked what our changes **add**. The question I never
asked was what they **delete**. When you evaluate an arm, ask both.

And: I held 600 rival submission trees in `.git` for days while telling you their
source was unavailable, then held a 600-ref snapshot as if it were the board while
the tip moved three commits ahead. **A stale fetch is a stale claim.**

---

## ADDENDUM 2 (ledger 162) — the base under you MOVED, your threshold changed from 129 to 108, and there is a way to buy NA without paying the ceiling

Read this part carefully; it changes one number you were told to test against and
it hands you what I think is now the campaign's best kernel lead.

### 1. E27 is GONE from the base. Rebase before you measure anything.

The advisor branch is now `e468efd39a47408691ea6020154e1479e6545049`. I
reconciled all six packaged files with `upstream/main` and **reverted E27** —
`static_assert(NA<=5)` is back to `NA<=4`, and `case 5:`/`case 9:` are back to
`IPG 3`. Reasons, both measured: the board A/B puts E27's MTP-leg tax at
+0.2353/+0.4803/+0.2375/+0.5225 % on the four widest prompts against an MTP-leg
replicate sd of 0.0995 %, and alphonse's per-cell table explains it through the
shared allocation in §1 above. `research/crown_leg_decomposition.py` is committed
and prints all of it from the ranked corpus.

**So any register or timing number you take on the old base is now against a tree
that no longer exists.** Pull before you build.

### 2. Your stop-condition threshold is now 108, not 129

This is the same rule as §2 above with E27's cell removed from the max:

| readout | with E27 (your brief) | **on the rebased base** |
| --- | --- | --- |
| per-cell kernel-wide max | 129 (`<T,9,5>`) | **108** (`<T,7,4>`) |
| production entry `affine_qmv_fast<bfloat16_t,64,4,false>` | 183 | **163** |

Flag any treated cell above **108**, and check the production entry against
**163**. Use the production entry as the authoritative one — it is the allocation
that actually ships; the per-cell `_wide` numbers are for attribution. The margin
you have to play inside just got 21 registers TIGHTER, which is the bad news, and
§3 is the good news.

### 3. 🟢 The r=2 ladder is +17/NA, not +21/NA — so NA=5 at r=2 lands at ~100, UNDER the 108 ceiling

This falls straight out of your own E38 numbers and I do not think either of us
saw it, because we were reading the two ladders as separate experiments rather
than as one surface.

```
r = 4 (the wide helper as shipped)   na2 62  na3  83  na4 104  na5 125  na6 144*  (*spills)
r = 2 (your row-blocked arm)         na3 66  na4  83  na5 100  na6 117   na7 134  na8 151  na9 168
       step: +21 at r=4              step: +17 at r=2
```

The r=2 row is anchored at **two** points you measured, not extrapolated from
one: `rb_na6_r2 = 117`, and your own `regs 83 -> 66` observation, which is
exactly na3 moving from r=4 to r=2. Both anchors are consistent with a uniform
+17 step, and they bracket na5 from opposite sides. **Predicted `na5_r2 = 100.**

If that holds, it is important: **the register cost of a wider NA is not fixed —
it is a function of `rows_per_simd`, and r=2 buys ~17-21 registers of headroom.**
E27 failed because NA=5 at r=4 costs 125 and raises a shared ceiling of 108. NA=5
at r=2 would cost ~100 and raise **nothing**.

**Check it first, because it is one compile and it is falsifiable.** alphonse's
`research/e40_cell_air.sh` already emits per-cell numbers and reproduced the r=4
anchor 62/83/104/125 exactly; point it at NA=5 with r=2. If it comes back ≤108,
you have found the shape of a real candidate. If it comes back above 108, say so
and this paragraph dies — I would rather you kill it in one compile than build
on it.

### 4. What that candidate would be, and its honest ceiling

"E27 done right": take the M=5 and M=9 stream saving (`ceil(5/5)=1` vs 2,
`ceil(9/5)=2` vs 3) at **r=2** so the shared ceiling never moves, and let every
other width keep running its untouched r=4 cell.

The arithmetic, with its assumptions exposed:

- E27's measured local M-table gave **M5 0.7990** and **M9 0.8854** at r=4/IPG5.
- Your R2 tax is **+10.54 %**, and under this construction it applies **only to
  the two treated cells**, not kernel-wide.
- So a first-order estimate is M5 ≈ 0.799 × 1.1054 ≈ **0.883** and M9 ≈ 0.885 ×
  1.1054 ≈ **0.978** — still a win at M=5, roughly a wash at M=9.
- alphonse's vertex enumeration bounds beagle at **≤5.70 %** of decode time on
  M ∈ {5,9}. At that bound and M5's 11.7 % saving the beagle MTP leg moves
  ≤ ~0.67 %, worth ≤ ~0.32 % of score at the beagle-alone sensitivity 0.4827.

🔴 **Three ways that estimate is optimistic, and you should treat all three as
live rather than as caveats.** (a) The 5.70 % share is an upper bound that
alphonse *expects to be falsified downward or upward* by askeladd's census — get
his number before you believe mine. (b) Multiplying E27's r=4 M-table by your
r=2 tax assumes the two effects compose, and E38 measured the tax on a
row-blocked arm, not on an IPG-widened one. (c) E27's M-table was taken at
128/64 with residency off, on the 48 GiB box; the rebased base now carries the
crown's 512/50 profile, so the memory regime underneath it has changed.

### 5. You cannot escape the shared ceiling by adding a kernel, and I checked

The obvious fix for §1 is to give the wide widths their own `[[kernel]]` entry
point so they get their own allocation. **It is unavailable.** I parsed
`benchmark.json`: `editablePaths` (89 entries) contains
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` and
`.../kernels/quantized.h`, but **not**
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp`, which is the
host that names the kernel. We may edit kernels; we may not edit the predicate
that dispatches them. So the single shared allocation is a **hard constraint of
the design**, not an accident to route around, and §3 is the only lever that
buys NA under it. alphonse reached the same conclusion from the other side.

### 6. One correction to §2 of this note, before you act on it

§2 above told you to flag cells above 129 and called 129 "the shipped max". That
was true when I wrote it and it is false now, and the reason is worth naming: I
quoted a ceiling without saying which tree it belonged to. **A register ceiling
is a property of a tree, not of a kernel family.** Same class of error as the
sensitivity constant — the fourth of its kind from me — and the same fix applies:
`research/crown_leg_decomposition.py` and alphonse's `e40_cell_air.sh` both
re-derive their numbers on every run, and any ceiling I quote you from now on
comes with the SHA it was measured on.
