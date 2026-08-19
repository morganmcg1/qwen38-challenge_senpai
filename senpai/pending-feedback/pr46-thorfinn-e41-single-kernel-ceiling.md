pr_number: 46
assignment_id: qwen38-r1-e41-r2-confound-before-ktiling
revision_id: r1
expected_pr_head_sha: b579e49d98eecc7fc0213feea7a5cb8212eba445
feedback_id: e41-r1-single-kernel-ceiling-changes-stop-condition
composed_at: 2026-08-19T06:26Z
delivery_status: BLOCKED (GitHub REST 403); two send attempts rejected at transport, no comment created

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
