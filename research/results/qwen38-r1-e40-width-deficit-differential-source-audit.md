# E40 — Which of the 229 shipped lines causes the width-confined MTP-leg deficit?

**Assignment** `qwen38-r1-e40-width-deficit-differential-source-audit` r1 · PR #45
**Student** qwen-alphonse · **Branch** `qwen-alphonse/width-deficit-differential-audit`
**Base** `1842f7cbe1ab5e7c0f53d1923e873f6525590ab0` (`senpai/qwen38-mtp-r1`)
**Shipped-surface baseline** `527306761f70e2c4024f347915328894db80c181`
**GPU used** none. Compile-only Metal front-end invocations, declared in §2.4.
**Ranked submission** none.
**Diff** confined to `research/`.

---

## 0. Answer first

| # | question | answer |
|---|---|---|
| 0 | Is the item-148 instrument sound? | **Yes, it survives.** Three corrections owed, none of which moves a conclusion. §1 |
| 1 | Which lines can produce a width-confined deficit? | **80 added / 4 removed of +229/−74.** Only **+4/−4** act inside a scored round. §2 |
| 2 | H1 from the compiler? | **CONFIRMED, and the mechanism is bigger than hypothesised.** The changed cell raises the *shared* kernel's register ceiling 108 → 129 (+19.4 %), so every width pays — including the byte-identical ones. §3 |
| 3 | Joint-consistency feasible set? | **Non-empty for both bankable legs, inside a very tight envelope**: ≤ 5.70 % of beagle decode time may sit on M∈{5,9}. Falsifiable by askeladd's census. §4 |
| 4 | Re-ranked re-test list | §6 |
| 5 | Named next experiment | **Simdgroup-matrix accumulation inside the editable cross-row QMV cells for M ≥ 4.** MDE(exact) 0.504 %. §7 |
| 6 | **Ledger 156 — is the inherited chunked-SDPA the width-gated source?** | **No. REFUTED from source, and in the opposite direction to the one feared.** `head_dim = 256` makes `sdpa_full` unreachable at *every* width, so the alternative to chunking is the **unfused eager graph** — a non-chunking rival is **slower** at M 6–9, not faster. H5 predicts the wrong sign. §9 |

**Posteriors** (advisor priors → mine): **H1 0.35 → 0.55**, **H2 0.25 → 0.02**, **H3 0.15 → 0.28**, **H4 0.25 → 0.15**. §5
**H5** (ledger 156, arrived mid-experiment, no prior given): **→ 0.02**, refuted on sign. §9

**Four errors found in the brief**, all of which I owe the advisor:
1. Item 148's "identical `effective_mean_draft_len` on **every** prompt" is **false on plutarch**. §1.2
2. The quoted "E27 M-table" value `M6 = 1.0150` is **E33's**, not E27's. E27's is `1.0032`. The "1.50 % per-M6 tax" premise **never existed in our tree**. §4.1
3. Item 148's σ used `pstdev` on n=6 where sample `stdev` is wanted; every σ is over-stated by 9.5 %. beagle is **+4.78σ**, not +5.24σ. §1.3
4. **The withdrawal of the chunk's positive control was over-conservative.** The advisor withdrew it because our 919/919 result was over token *ids*, which do not discriminate value drift — that reasoning is correct. But our tree holds a *stronger* instrument he did not cite: `research/ESTABLISHED_FACTS.md:335` records **bit-exactness on the hexfloat row gate for widths 6..8**, which compares float bit patterns, not ids. The genuine gap is **width 9 / depth 8 only**, and it is already written down. §9.4

**Two findings the brief did not anticipate**, both zero-GPU and both decision-relevant:
- **`vector_limit ≥ 10 > 9` for every scored projection shape**, proven from source. MLX's own `qmm`/`qmm_splitk` is therefore **never reached at any legal MTP width**. This positively verifies that E27's cells are live (H1 survives a path check it had never been given) *and* confirms re-test entry 4's premise from source. §7.1
- **Both retired-family hunks guard on `physicalMemory ≥ 96 GiB`.** This box is **48 GiB**, so H3's mechanism has been **inactive in every local measurement the campaign has ever taken** and is active only on the ranked 128 GiB box. §2.3

---

## 1. The instrument audit — I attacked this first, as instructed

Tool: `research/e40_instrument_audit.py`. It re-derives item 148 from the board dump and
adds the reference classes item 148 does not have.

### 1.1 The question that actually mattered: is the contrast selected on outcome?

The plateau six are chosen *because they outscore us*. If the wide−narrow contrast were an
artefact of that selection, the assignment would be void. I built five reference classes,
two of which select on something **orthogonal to score**:

| class | selects on score? | wide−narrow | beagle σ | MDE(exact) |
|---|---|---|---|---|
| plateau six (item 148) | yes, post-hoc | +0.3213 % | +4.78 | 0.0895 % |
| all 11 cohort rows above us | yes, no pruning | +0.3235 % | +3.50 | 0.1752 % |
| all 8 above us in the 16:59–23:52 window | yes | +0.3244 % | +4.41 | — |
| **narrow-leg-matched (n = 15; 8 above us, 7 below)** | **no** | **+0.2704 %** | +0.64 | 0.9260 % |
| serial-leg-matched (n = 53) | no | −0.4308 % | −0.20 | 5.9712 % |
| same-day fingerprint twins (n = 68) | no | −0.4625 % | −0.14 | 9.6067 % |

The narrow-leg-matched class is matched on legs with **proven zero score value**, so its
selection criterion is orthogonal to the outcome. It still returns **+0.2704 %, 84 % of the
plateau estimate.** Selection on score is not the cause.

The two negative point estimates are **not** counter-evidence: their MDEs are 5.97 % and
9.61 % against a 0.32 % effect, i.e. they are **19× and 30× under-powered**. Quoting them as
nulls would be exactly the error E39 was written to stop.

**Verdict: the instrument survives.** The assignment is not void.

### 1.2 Correction 1 — a real error in item 148

> "All seven have `effective_mean_draft_len` identical to four decimal places on every prompt."

**False on plutarch.** `9cd3be9b9913` (WillGasser) has plutarch draftlen **2.5407** against
our **0.1540** — he is the latch escapee, which is the subject of his own note. The other
seven prompts are identical to 4 dp as claimed.

Consequence: the plutarch row of the item-148 table (sd 15.84 %, deficit 17.35 %) compares
**different work**, so it should be **struck**, not labelled "latch-dominated". Zero effect on
any conclusion — plutarch is worth 0.0000 % of score.

### 1.3 Correction 2 — every σ is over-stated by 9.5 %

Item 148 uses `pstdev` on n = 6 where the sample `stdev` is wanted. Ratio √(5/6) = 0.9129.

| prompt | item 148 σ | corrected σ |
|---|---|---|
| drama | +0.31 | +0.28 |
| travel | −0.68 | −0.62 |
| **beagle** | **+5.24** | **+4.78** |
| **medicine** | **+1.45** | **+1.32** |
| essays | +34.16 | +31.18 |
| republic | +8.81 | +8.05 |
| botany | +8.99 | +8.21 |

The headline "5σ" is **4.8σ**. Leave-one-out on beagle: deficit +0.3284…+0.3978 %, σ
+4.07…+6.86 — the finding is robust to dropping any single plateau row.

### 1.4 Correction 3 — the cohort is 89, not 94

653 rows in the dump; 105 are all-8 on head `559b24eb`; 16 of those carry no
`submissionCommitSha` (all `rejected`), leaving **89 analysable**. Citation drift, not an
error: the plateau six and our row are unaffected and no σ in the brief moves.

### 1.5 Two checks that came back clean

- **Independence.** The six plateau rows are six independent measurements: 0 identical-on-all-8
  pairs, 6 distinct solvers, 6 distinct commits.
- **Plateau definition.** 11 cohort rows now outscore us; item 148 used 6. Using all 11 gives
  +0.3235 % against +0.3213 %. No cherry-picking effect.
- **Score identity** reproduces our official row to 1.8 × 10⁻¹⁵. Exact permutation p is
  floor-limited at 1/56 = 0.0179 and cannot go lower with n = 6.

### 1.6 A provenance gap I must flag

Our board row's `submissionCommitSha` is `2b0c36a078b7660c9215adee933336ff46da25af`, and
that is **not a resolvable git object in this checkout** (`git cat-file -t` fails). It appears
in `senpai/frontier-state.json` as `ourBestRankedRow` (id `ca9251b8-58cd-4d90-9a52-fa05f5657216`,
score 3.23250848263467, `rejected`, rank 9 of 408).

So `research/shipped-surface-gate.sh` verifies **HEAD**, not the exact submitted snapshot.
Everywhere a conclusion below rests on "the shipped delta", it rests on HEAD matching the
submission, which I cannot prove locally.

---

## 2. Deliverable 1 — width-dependence classification of the whole delta

Tool: `research/e40_hunk_classify.py`. Every hunk is labelled and the per-hunk counts are
**asserted** to sum to the gate totals, so the table cannot silently omit a hunk.

```
hunk sums:   +229/-74
gate totals: +229/-74
match: True
every hunk labelled: True
```

Gate output on HEAD vs `5273067`:

```
157/47  Sources/MLXFastModel/Qwen36MTPBlockSession.swift
 32/ 0  Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift
 32/19  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
  4/ 4  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
  4/ 4  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
totals: 5 files, +229/-74
```

### 2.1 Scaling classes

A hunk can only produce a **width-confined** deficit if its cost scales with `M` (rows in a
round) or with the **set of distinct widths** a run touches. Anything that scales with round
count, token count, or nothing at all is eliminated by class, because the narrow prompts are a
control inside our own row: they have rounds and tokens too.

| class | meaning |
|---|---|
| `none` | cannot enter a scored round (comment, dead field, declaration, or code on a path the scored run never takes) |
| `once` | one-time, outside the timed window |
| `per-token` | ∝ 512 emitted tokens (identical every prompt) |
| `per-round` | ∝ R (84…487 by prompt) |
| **`per-row`** | **∝ M within a round** |
| **`per-width`** | **once per distinct width first touched in a run** |

### 2.2 `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (+157/−47, 16 hunks)

| # | +/− | what | class |
|---|---|---|---|
| 0 | +0/−2 | drop `reachedStopToken` from `Qwen36MTPRoundResult` | none |
| 1 | +0/−1 | drop the session's `reachedStopToken` var | none |
| **2** | **+76/−0** | **`wireResidentWeightsIfEnabled`, incl. `Memory.clearCache()`** | **per-width** |
| 3 | +7/−0 | warm split into `warmAllDepthShapes` + the wire call | once |
| 4 | +24/−0 | `traceSyncHeadChain` + `traceSink` statics | none |
| 5 | +1/−4 | `traceWrite` → `traceSink.write` | none |
| 6 | +5/−0 | `if Self.traceRounds { snapshotScheduleSignal }` | per-round |
| **7** | **+4/−0** | **`if Self.traceRounds { scheduleTrace += }` inside the depth walk** | **per-row** |
| 8 | +24/−0 | `scheduleTrace` var + `snapshotScheduleSignal` body | none |
| 9 | +1/−1 | `let tailPending = pendingTop2` → `pendingTop2 != nil` | none |
| 10 | +8/−25 | remove the pre-draft EOS early-return | none |
| 11 | +1/−2 | drop `reachedStopToken` from the depth-0 result | none |
| 12 | +3/−0 | `if Self.traceSyncHeadChain { eval(draft chain) }` | per-round |
| 13 | +2/−1 | `scheduleTrace` appended to the trace line | none |
| 14 | +0/−9 | remove the post-commit EOS truncation | none |
| 15 | +1/−2 | drop `reachedStopToken` from the final result | none |

Ten of the sixteen hunks (0, 1, 10, 11, 14, 15 and their argument removals) are **one change**:
the **EOS-latch fix** that `program.md` describes as a solver defect. It is a correctness fix,
it fires only on a round whose primary is a stop token, and hunk 14 **removes** an O(M)
`firstIndex(where:)` — so on the per-row axis the candidate does **strictly less** work than the
baseline here.

Hunks 4, 5, 6, 8, 12, 13 are the phase-trace instrumentation. In a scored run
`MLX_QWEN_MTP_TRACE` is unset, so `Self.traceRounds` is `false` and every body is dead. What
survives is the **guard itself**: hunk 6 and 12 are one static-Bool test per round; hunk 7 sits
**inside the depth-extension walk**, so it is ≤ `depth` tests per round, i.e. genuinely O(M).
That is why hunk 7 is classed `per-row` rather than waved away — and it is then killed on
magnitude in §2.5, not on class.

### 2.3 The `per-width` hunk, and the 96 GiB gate

Hunk 2 is the only `per-width` mechanism in the whole delta. The mechanism is **not** the
wired ticket; it is one line inside it:

```swift
Memory.clearCache()
let active = Memory.activeMemory
```

`warmAllDepths` warms "the batched verify at every legal width `1 ... maxDepth + 1`". In the
**baseline** those warm buffers stay in MLX's allocator cache, so the scored run reuses them.
Our candidate **returns them to the OS**, so the scored run pays first-touch allocation again at
each width it visits. The cost is *outside* the timed window; its **effect is inside it**, and it
rises with the widths a prompt actually reaches. That is the correct shape for a
width-confined deficit, and it is exactly the hole the advisor left open in H3 — "no missed
Metal compile does not mean no per-width first-touch allocator cost".

**But both retired-family hunks are gated on physical memory:**

```swift
guard ProcessInfo.processInfo.physicalMemory >= (UInt64(96) << 30) else { return }   // session hunk 2
guard physicalMemoryBytes >= (UInt64(96) << 30) else { return }                      // policy hunk 1
```

This box is **Apple M4 Pro, 51 539 607 552 bytes = 48 GiB**. The guard **fails locally**.

Therefore: **H3's mechanism has been inactive in every local measurement this campaign has
ever taken, and is active only on the ranked 128 GiB box.** Two consequences:

1. It explains why E27's local M-table shows no broad penalty while the ranked row does — the
   advisor's own ranked-geometry caveat, now with a source-level reason.
2. **H3 is not testable on this hardware at any n.** This is E39 entry 6's finding
   ("96 GiB gate, no override") restated for a different mechanism. It is a hardware gate, not
   a power problem.

### 2.4 `RuntimeStartupMemoryPolicy.swift` (+32/−0) and `Qwen35.swift` (+32/−19)

| file | # | +/− | what | class |
|---|---|---|---|---|
| policy | 0 | +1/−0 | `import Foundation` | none |
| policy | 1 | +27/−0 | `installQwenMTPFullProfileCommandBufferDefaults` | once |
| policy | 2 | +4/−0 | `resolve()` calls the installer | once |
| Qwen35 | 0 | +22/−0 | `qwen35DecodeLadderRungs` global `Set<Int>` | once |
| Qwen35 | 1 | +4/−4 | ladder comment `S<=2` → `S<=9` | none |
| Qwen35 | 2 | +2/−1 | comment: rung set overridable | none |
| **Qwen35** | **3** | **+2/−7** | **fused loop: `switch` → `Set.contains`** | **per-round** |
| Qwen35 | 4 | +2/−7 | unfused loop: same rewrite | none |

**The brief's H2 question answered directly.** It asked whether `Qwen35.swift` "allocates,
reshapes, concatenates or broadcasts per row". **It does none of those.** The file's *only*
functional change is replacing a compile-time `switch i { case 0,1,9,19,29,39,49,57: }` with
`qwen35DecodeLadderRungs.contains(i)`. Two facts pin it down:

- The shipped default `[0, 1, 9, 19, 29, 39, 49, 57]` is **exactly the baseline switch cases**,
  so the `asyncEval` schedule is **byte-identical between the arms**. Nothing about GPU
  submission changed.
- `let ladderActive = inputs.dim(1) <= 9 || prefillLadder` appears as **context** in the diff —
  the baseline already gated at 9. Only the stale comment moved. There is no widening of the
  ladder's reach.

Hunk 4 is on the `dtype != bfloat16 || dim(-1) != 5120` branch, which the scored Qwen 3.8 path
never takes.

So H2's entire cost is: one `swift_once` guard + one SipHash-1-3 + one bucket probe, **per
layer, per forward pass = 64 per round**, independent of M.

Also worth recording for H4: the policy installer uses `setenv(..., 0)` — **do not overwrite**.
The advisor states the crown row is the same `setenv` with **overwrite = 1**. If the harness
already exports `MLX_MAX_MB_PER_BUFFER`, **our call is a no-op and theirs is not.** That is a
readable, source-level route by which the plateau could be running a different *effective*
command-buffer geometry than we infer. It raises H4's weak form. It is retired-family, so I
did not pursue it.

### 2.5 The Metal delta (+4/−4 twice) and the elimination ladder

| file | # | +/− | what | class |
|---|---|---|---|---|
| `quantized.h` | 0 | +1/−1 | `static_assert(NA <= 4)` → `NA <= 5` | none |
| `quantized.h` | 1 | +1/−1 | IPG comment `ceil(M/4)` → `ceil(M/5)` | none |
| **`quantized.h`** | **2** | **+1/−1** | **`case 5:` `<T,5,3>` → `<T,5,5>`** | **per-row** |
| **`quantized.h`** | **3** | **+1/−1** | **`case 9:` `<T,9,3>` → `<T,9,5>`** | **per-row** |
| `quantized.cpp` | 0–3 | +4/−4 | the runtime-effective generated twin of all four | same |

Ladder:

```
class         added  removed   hunks   verdict
none             75       63      19   ELIMINATED by scaling class
once             60        0       4   ELIMINATED by scaling class
per-token         0        0       0   ELIMINATED by scaling class
per-round        10        7       3   ELIMINATED by scaling class
per-row           8        4       5   CANDIDATE
per-width        76        0       1   CANDIDATE

step 1  scaling class eliminates   145 added / 70 removed  ->  84 added / 4 removed left
step 2  magnitude budget kills the per-row trace guard (-4/-0)  ->  80 added / 4 removed
step 3  survivors:
          +4/-4   quantized.h + .cpp  case 5 / case 9  IPG 3 -> 5   (H1, INSIDE the timed window)
          +76/-0  Qwen36MTPBlockSession Memory.clearCache()          (H3, effect inside, RETIRED, 96 GiB-gated)
```

**65.1 % of the added lines are eliminated, and only +4/−4 lines act inside a scored round.**

### 2.6 Declared compile-only toolchain invocations

`metal -std=metal3.1 -S -O2` and `metal-opt -passes='default<O3>'`, front-end only. No
`MTLDevice`, no `MTLLibrary`, no pipeline creation, no `swift build`, no timed run. Base-arm
compiles use a `/tmp` shadow include directory so the working tree is never checked out.
Toolchain: `Apple metal version 32023.883 (metalfe-32023.883)`, target
`air64-apple-darwin25.5.0`, SDK MacOSX26.5.

---

## 3. Deliverable 2 — H1 adjudicated from the compiler

### 3.1 The structural fact the pass-count argument omits

The advisor's objection is that under a pure weight-pass model we tie or win at every M and
can never lose. That model is correct **and the paradox dissolves in the source**.

`quantized.h:1869` declares a **single** `[[kernel]] affine_qmv_fast`. Inside it, the width
selection is a `switch` on **`ntg.x`, a runtime value** — the `≥4096` table at ~:1925 and the
`<4096` table at ~:1971. Every helper is `METAL_FUNC`, i.e. inline. Therefore:

> **Every width cell 2…9 of both branches compiles into ONE kernel with ONE register
> allocation, and that allocation is the maximum over all cells.**

`qmv_fast_crossrow_affine4_g64_m<T, M, IPG, DIRECT_NIBBLES>` calls
`qmv_fast_crossrow_affine4_g64_wide<T, IPG, …>`, so **NA = IPG**, and the `_m` template also
inlines a tail instantiation `_wide<T, max(M % IPG, 2)>`.

Shipped `≥4096` dispatch (IPG): M3→3, M4→4, **M5→5**, M6→3, M7→4, M8→4, **M9→5**. Baseline
differs only at M5→3 and M9→3. Weight streams `ceil(M/IPG)` go 1,1,2,2,2,2,3 → 1,1,1,2,2,2,2:
we win a stream at M = 5 and M = 9 and tie everywhere else. The advisor's table is right; the
**shared register allocation is the term it does not contain.**

### 3.2 Per-cell measurement

`research/e40_cell_probe.metal` + `research/e40_cell_air.sh` isolate every dispatch cell.
The anchor reproduces the campaign's historical figures **exactly**:

```
inner packing factor alone (E13/E27/E32 anchor was 62/83/104/125)
  NA=2  regs=62      NA=3  regs=83      NA=4  regs=104      NA=5  regs=125
```

| M | base cell | regs | cand cell | regs | Δ |
|---|---|---|---|---|---|
| 3 | ipg3 | 83 | ipg3 | 83 | +0 |
| 4 | ipg4 | 104 | ipg4 | 104 | +0 |
| **5** | ipg3 | 87 | **ipg5** | **125** | **+38** |
| 6 | ipg3 | 83 | ipg3 | 83 | +0 |
| **7** | **ipg4** | **108** | ipg4 | 108 | +0 |
| 8 | ipg4 | 104 | ipg4 | 104 | +0 |
| **9** | ipg3 | 83 | **ipg5** | **129** | **+46** |

`<4096` branch: flat at **89** for all M = 2…9, so it is **not** the binding cell and E27
genuinely moved the kernel-wide maximum.

```
KERNEL-WIDE MAXIMUM (what one register allocation must satisfy)
base     e40_m7_ipg4              108
cand     e40_m9_ipg5_cand         129
```

**Δ = +21 registers, +19.4 %.**

### 3.3 Independent corroboration at the production entry point

`research/e40_qmv_entry_probe.metal` + `research/e40_entry_air_diff.sh` compile the **actual
scored instantiation** `affine_qmv_fast<bfloat16_t, 64, 4, false>` at both arms:

| arm | AIR lines | peak_live_regs | allocas | fadd | device_loads |
|---|---|---|---|---|---|
| base | 13038 | **163** | 55 | 940 | 858 |
| cand | 13554 | **183** | 55 (identical multiset) | 964 | 882 |

**+20 at the production entry point against +21 from the per-cell maximum — two independent
routes to the same number.** `batch1` and the 2-bit control are **byte-identical** between arms,
so the negative controls hold. **No new spill alloca**; the alloca multiset is unchanged.

### 3.4 Verdict, and the exact provenance limits

**H1's mechanism is CONFIRMED, and it is a different — larger — mechanism than the hypothesis.**
Not "the changed cell costs occupancy" but:

> **The changed cell raised the whole shared kernel's register ceiling by ~19 %, so every
> width pays it — including the byte-identical M = 3, 4, 6, 7, 8 cells.**

That is precisely the class of mechanism required to tax widths whose kernel text did not
change, which is what the pass-count paradox demanded.

**Independent corroboration from E38, received after this section was written.** The advisor
reports E38 came back failed-and-merged with the conclusion that row-blocking is closed by a
**128-register wall**, quantified as "**+21 regs per NA up to na5=125**, so `<T,5,5>` is the
structural optimum of that kernel family". Both numbers match this section exactly and were
obtained by a different agent on a different question: my per-cell anchor is NA = 2/3/4/5 →
**62/83/104/125** (a +21 ladder, terminating at 125), and my kernel-wide ceiling delta is
**+21** (108 → 129). So the register ladder is now double-sourced. Two consequences worth
separating:

- It **strengthens** H1's measured mechanism — the ladder is real and the 125 figure is not an
  artefact of my peak-live-SSA heuristic, since E38 reached it independently.
- It does **not** raise H1's posterior above §5.1's 0.55, because E38's finding is about the
  *ceiling's location*, not about whether crossing it costs occupancy on the ranked box. The
  unmeasured AIR-pressure→occupancy conversion is still the binding uncertainty, and E38's "128
  wall" is in fact mild evidence that 125 sits *under* the wall rather than over it — which would
  make H1's penalty smaller, not larger. I flag this as the one reading of E38 that argues
  *against* my own confirmation.

**What I could not get, and why.** There is **no true register or occupancy readout on this
box**, and I want that stated in the same breath as the number:

- `-mllvm -stats` → *"Statistics are disabled"*. `-Rpass*` accepted but silent.
- `metal-objdump --disassemble` stops at LLVM/AIR and never reaches AGX ISA.
- `metal-readobj` exposes no register fields. `-save-stats` gives only file-search counters.
- `-fmetal-enable-statistics` / `--stats` rejected outright.
- E13 already failed the offline AGX translator route (AIR 2.8 vs 2.5).
- E32 already showed `maxTotalThreadsPerThreadgroup` = **1024 for all 77 cells including
  spilling ones**, so pipeline reflection cannot discriminate register pressure — which is why
  I did not create a device even though a build is permitted.

`peak_live_regs` in `research/air_kernel_stats.py` is a **lane-weighted peak-live-SSA textual
heuristic**; its own docstring says the absolute number is not usable and only the shape is.
So: **the +19.4 % is a compiler-derived contrast on the exact scored template instantiation,
reproduced by two independent routes, with the historical anchor recovered exactly — but it
cannot be converted into an occupancy figure on this hardware.** Spill-alloca detection *is* a
genuine compiler outcome, and it says no new spill appeared.

The template instantiation is exactly the one the scored path dispatches:
`affine_qmv_fast<bfloat16_t, 64, 4, false>` (bf16, group 64, 4-bit, non-batched), and §7.1
independently proves the scored path reaches it at every legal width.

---

## 4. Deliverable 3 — the joint-consistency test

Tool: `research/e40_width_tax_feasibility.py`.

### 4.1 First, the brief's M-table is E33's, not E27's

E27's own report (`research/results/qwen38-r1-e27-m5-weight-stream-cliff.md:207-216`):

```
M:      1       2       3       4       5       6       7       8       9
base: 64.549 65.628 72.993 83.115 120.683 128.865 139.078 149.355 186.233
cand: 59.979 64.707 73.136 83.072  96.423 129.280 139.007 150.110 164.900
ratio 0.9292 0.9860 1.0020 0.9995  0.7990  1.0032  0.9995  1.0051  0.8854
```

**E27's M = 6 is 1.0032, not 1.0150.** The value `1.0150` is from **ledger item 129 = E33's
row-blocking arm** (130.781 / 128.843) — a different experiment, **falsified**, and **not in the
shipped 5-file surface**. E27's report states the five untouched widths M = 3, 4, 6, 7, 8 "all
land within ±0.5 %, which sets the noise floor" (:222).

**Consequence: the "1.50 % per-M6 tax" premise never existed in our tree.** The brief's
medicine-derived bound (≤ 0.29 %) is therefore *consistent* with E27's real M = 6 value rather
than in tension with it — the tension is an artefact of the mis-citation, and the strong
version the advisor already killed was never alive.

Caveat carried forward: E27's table is **M4 Pro, 128/64, residency off**.

### 4.2 Exact round/draft recovery, and a validity gate the brief does not have

`effective_mean_draft_len = D/R` recovered by `Fraction(...).limit_denominator(512)`, with
`R + A = 512` and `non_drafting_round_count` pinning `n₁`:

| prompt | mean_draft | R | D | A | nd | mean M | α |
|---|---|---|---|---|---|---|---|
| plutarch | 0.154004106776 | 487 | 75 | 25 | 449 | 1.1540 | 0.333 |
| drama | 2.297619047619 | **168** | 386 | 344 | 0 | 3.2976 | 0.891 |
| travel | 2.655660377358 | 212 | 563 | 300 | 0 | 3.6557 | 0.533 |
| **beagle** | 4.532710280374 | **107** | **485** | **405** | 0 | 5.5327 | **0.8351** |
| **medicine** | 4.767676767677 | **99** | **472** | **413** | 0 | 5.7677 | **0.8750** |
| essays | 5.425287356322 | 87 | 472 | 425 | 0 | 6.4253 | 0.900 |
| republic | 5.776470588235 | 85 | 491 | 427 | 0 | 6.7765 | 0.870 |
| botany | 5.269662921348 | 89 | 469 | 423 | 0 | 6.2697 | 0.902 |

The brief's `beagle R=107 D=485 A=405 α=.8351` and `medicine R=99 D=472 A=413 α=.8750` are
**reproduced exactly**, so askeladd and I agree on the two legs that matter.

**A gate I had to add to my own arithmetic.** `limit_denominator` returns the *reduced* fraction;
the true `(R, D)` may be `k·(R₀, D₀)`. A recovery is only admissible if the implied acceptance
rate `α = (512 − R)/D` lies in `(0, 1]`. **drama fails at k = 1** (α = 2.2176 — more tokens
accepted than proposed, which is impossible) and needs **k = 2, R = 168**. Every other prompt
is admissible at k = 1. **Both bankable legs need no correction**, so no score-relevant
conclusion moves — but the brief and askeladd should both apply this gate, because an
unvalidated `limit_denominator` recovery is silently wrong on at least one prompt.

### 4.3 The model

Acceptance is bit-identical across the plateau and us on all eight prompts (§1.2 aside), so
both trees traverse the **same width sequence**. Hence

```
deficit = Σ_M w_M τ_M ,   τ_M = c_M / c'_M − 1 ,   w_M = n_M c'_M / Σ_j n_j c'_j
```

The polytope `{n ≥ 0, Σn = R, Σn(M−1) = D, n₁ = nd}` has support-≤2 vertices, and `Σ w τ` is
linear-fractional, so **enumerating width pairs gives the exact achievable deficit interval.**

### 4.4 F1 — is E27's own measured tax vector consistent with the two legs?

| prompt | observed | min | max | in range? |
|---|---|---|---|---|
| drama | +0.0118 | −12.3165 | +0.2314 | yes |
| travel | −0.0450 | −14.3747 | +0.2690 | yes |
| **beagle** | **+0.3631** | −18.4450 | **+0.4056** | **yes** |
| **medicine** | **+0.0880** | −17.7830 | **+0.4180** | **yes** |
| essays | +0.5087 | −16.1185 | +0.4487 | **NO** |
| republic | +0.2343 | −15.3277 | +0.4632 | yes |
| botany | +0.4352 | −16.4895 | +0.4419 | yes |

**The feasible set is NON-EMPTY for both bankable legs.** So the answer to the brief's
question is *yes*, and the mechanism can be width-cell-specific.

**But essays overflows the ceiling by 0.060 pp.** Essays is worth 0.0000 % of score, so it
changes no value claim — yet it is a real mechanistic signal: **E27's measured tax cannot
account for the full deficit on every leg.** At least one leg needs a contribution from
outside the QMV cells. That is the strongest single argument that H1, while confirmed, is
**necessary but not sufficient**.

### 4.5 F2 — a width-independent register tax is refuted by our own narrow legs

If the register ceiling taxed every width equally at rate ρ, every prompt would show deficit
ρ. drama (+0.0118 %) and travel (−0.0450 %) force **ρ ≈ 0**, and then the five wide legs at
+0.23…+0.51 % **cannot be produced at all.**

So a flat ceiling tax is dead, and a width-dependent shape is required. I fitted four
one-parameter shapes `τ_M = ρ·g(M)` on the bankable legs and cross-validated:

| g(M) | beagle ρ | medicine ρ | intersection (7 legs) | intersection (bankable only) |
|---|---|---|---|---|
| `M − 1` | [0.0006, 0.0008] | [0.0001, 0.0002] | EMPTY | **EMPTY** |
| `1[M ≥ 6]` | [0.0039, 0.0189] | [0.0009, 0.0033] | EMPTY | **EMPTY** |
| `1[M ≥ 4]` | [0.0036, 0.0056] | [0.0009, 0.0013] | EMPTY | **EMPTY** |
| `M` (launched TGs) | [0.0005, 0.0007] | [0.0001, 0.0002] | EMPTY | **EMPTY** |

**Every monotone one-parameter shape fails, even on just the two bankable legs.** The reason is
the inversion: medicine is the **wider** prompt (mean M 5.7677 vs 5.5327) yet has the
**smaller** deficit (+0.088 % vs +0.363 %). No function increasing in mean width can do that.

**The mechanism must be non-monotone in M, with τ changing sign** — which is exactly E27's
measured shape: negative at M = 1, 2, 5, 9 and ≈ 0 at M = 3, 4, 6, 7, 8. Under that shape the
inversion is natural: medicine simply carries more mass where our tree is **faster**
(M = 5 at −20.1 %, M = 9 at −11.5 %).

### 4.6 F3 — a free tax vector says nothing

With 7 free non-negative τ and 8 leg equations the system is under-determined and any positive
deficit is attainable by scaling ρ. Reported for completeness; **F1 and F2 are the informative
statements.** This is the trap the brief's "if the feasible set is empty" framing invites, and
it is why I bounded the tax vector by E27's own measurements rather than fitting it freely.

### 4.7 F4 — the falsifiable prediction, for askeladd

Adding `deficit == observed` to the polytope keeps it **linear** after clearing the
denominator: `Σ_M n_M c_M (τ_M − obs) = 0`. With three equalities the vertices have support
≤ 3, so exact rational enumeration over all 56 width triples gives the **true** maximum, not a
grid approximation:

| leg | observed | feasible vertices | **max decode-time share on M ∈ {5,9}** | argmax mix |
|---|---|---|---|---|
| beagle | +0.3631 % | 18 | **5.70 %** | {3: 53.6, 8: 49.2, 9: 4.1} |
| medicine | +0.0880 % | 20 | **3.74 %** | {3: 44.7, 8: 51.7, 9: 2.6} |

> **If the measured beagle width histogram spends more than 5.70 % of decode time at M = 5 or
> M = 9, then "E27's kernel tax alone" is refuted and a second mechanism is mandatory.**

The feasible mixes concentrate on M = 3 and M = 8 — the **untouched** widths, whose only
possible source of slowdown is the shared register allocation. That is a coherent story, and it
is the *same* story as §3.

**My prediction is that this will falsify.** beagle has α = 0.835 under a depth-8 cap; a tree
that accepts 83.5 % of its drafts and is allowed 8 of them should spend far more than 5.7 % of
its time at full depth. If so, H1 is confirmed as a mechanism but demoted as *the* explanation,
and the residual belongs to H3.

**@askeladd: 5.70 % (beagle) and 3.74 % (medicine) are the exact numbers to compare against.**
It is a one-counter check on the census side and needs no GPU. We agree on R, D, A and α for
both legs; I add the validity gate of §4.2 and this envelope.

---

## 5. H1 / H2 / H3 / H4 verdicts and posteriors — H5 is adjudicated separately in §9.3

| | prior | **posterior** | verdict |
|---|---|---|---|
| **H1** E27 register pressure costs width-proportional occupancy | 0.35 | **0.55** | **CONFIRMED as a mechanism**, larger than hypothesised, but *necessary not sufficient* |
| **H2** `Qwen35.swift` does something per-row | 0.25 | **0.02** | **ELIMINATED**, two independent ways |
| **H3** the session's warm/residency path | 0.15 | **0.28** | **ALIVE and un-testable on this box** |
| **H4** the plateau isn't running the tree we think | 0.25 | **0.15** | **strong form REFUTED**; weak form open |

### 5.1 H1 → 0.55. What moved it up

- The single-kernel structure (§3.1) resolves the pass-count paradox that made the advisor
  distrust H1. The pass-count model is right and *incomplete*.
- +19.4 % kernel-wide register ceiling, measured on the exact scored instantiation, by two
  independent routes, with the historical 62/83/104/125 anchor reproduced exactly and clean
  negative controls (§3.2–3.3).
- §7.1 proves from source that the scored path actually reaches these cells at every legal
  width — a path check H1 had never been given.
- F1 shows E27's own measured τ can reach both bankable legs (§4.4).

**What holds it below 0.7:** essays overflows E27's τ ceiling (§4.4), so E27 alone cannot
explain every leg; F4's envelope (≤ 5.70 % of beagle time on M∈{5,9}) is tight and I expect it
to falsify; and the register→occupancy conversion is **unmeasured** on this hardware (§3.4).

### 5.2 H2 → 0.02. Eliminated twice over

Tool: `research/e40_overhead_budget.py`. Invert each leg's deficit into the host cost the
mechanism would need:

| prompt | T (s) | R | ms/round | deficit % | H3 need /round | **H2 need /lookup** |
|---|---|---|---|---|---|---|
| plutarch | 15.517 | 487 | 31.862 | +0.0469 | 14.9 µs | 233.3 ns |
| drama | 10.126 | 168 | 60.272 | +0.0118 | 7.1 µs | 111.3 ns |
| travel | 8.903 | 212 | 41.995 | −0.0450 | −18.9 µs | −295.3 ns |
| **beagle** | 6.233 | 107 | 58.253 | **+0.3631** | **211.5 µs** | **3304.6 ns** |
| **medicine** | 5.821 | 99 | 58.795 | **+0.0880** | 51.8 µs | 808.8 ns |
| essays | 5.764 | 87 | 66.249 | +0.5087 | 337.0 µs | 5265.7 ns |
| republic | 5.673 | 85 | 66.744 | +0.2343 | 156.4 µs | 2443.4 ns |
| botany | 5.726 | 89 | 64.338 | +0.4352 | 280.0 µs | 4374.6 ns |

**(a) Magnitude.** One `Set<Int>.contains` is a `swift_once` guard, a SipHash-1-3 over 8 bytes,
and one L1-resident bucket probe on an 8-element set: 3 ns optimistic, **40 ns deliberately
pessimistic**. beagle needs **3304.6 ns — 83× the pessimistic ceiling.**

Upper bound on what H2 can explain: **≤ 0.0044 pp of beagle's +0.3631 % (1.21 %)** and
**≤ 0.0044 pp of medicine's +0.0880 % (4.95 %)**. In score terms H2's absolute ceiling is
0.0044 × 0.485 = **+0.0021 %**, which is **87× below 2σ_score = 0.185 %**. Even if H2 were
entirely real it could never be seen in a ranked score.

**(b) Shape — independent of magnitude.** A fixed per-round cost δ gives `τ_M = δ / c_M`, which
**decreases** in M, so it predicts the largest *relative* deficit on the **cheapest, narrowest**
rounds. Observed ordering is inverted: the three cheapest-round prompts show +0.047, +0.012,
−0.045 % and the five most expensive show +0.09…+0.51 %. **Neither a per-round nor a per-layer
host cost can be the primary mechanism at any magnitude.** This also independently reproduces
the brief's own `corr(1/(1+draftlen), deficit_ms) = −0.35` observation from a different
direction.

The same shape argument kills session hunk 6 and 12, and the same magnitude argument kills
hunk 7 (the only per-row survivor: it needs 3305 ns per predicted-not-taken branch).

I keep H2 at 0.02 rather than 0.00 only because I did not *measure* the lowering of the global
accessor; nothing in the source suggests a pathology, but 83× is an argument from cost model,
not from a compiled artefact.

### 5.3 H3 → 0.28. The biggest mover, and the most frustrating

**Up**, because I found the specific line the advisor's H3 was reaching for:
`Memory.clearCache()` in the new warm path (§2.3). It is the **only** `per-width` mechanism in
+229/−74. It survives the shape test (cost rises with the widths a prompt reaches, and the
narrow legs genuinely reach fewer). It survives the magnitude test comfortably: beagle needs
22.63 ms total, which over ~9 widths × 64 layers × a few buffers is **≈ 13 µs per fresh Metal
allocation** — squarely plausible.

**Held at 0.28, not higher,** because a first-touch cost ∝ `Σ_{M visited} M` predicts a
beagle:drama ratio of ~2–4×, and the observed absolute overheads are 22.63 ms vs 1.19 ms = **19×**.
The shape is right; the magnitude ratio under-shoots by ~5×. A cost ∝ M² would fit better,
which is not obviously wrong for page mapping, but I am not going to fit an exponent to two
points.

**And it cannot be settled here.** The `≥ 96 GiB` guard means this mechanism is **inert on this
48 GiB box** — so no local ABBA, at any n, can measure it. It also lies in the family the
advisor has retired. I therefore **classified it and stopped**, and I am *not* proposing it as
deliverable 5. The decision about whether a retired family gets one ranked probe is the
advisor's, not mine. If the answer is yes, the cheapest instrument is a single ranked
submission that changes only `Memory.clearCache()` → nothing, since the wired ticket's own
`WiredSumPolicy(cap:)` does not require the cache to be empty — only the `activeMemory`
*reading* does, and that could be taken before the clear.

### 5.4 H4 → 0.15. Strong form refuted, weak form open

The brief's strong H4 — *"if the diff cannot support a width-dependent mechanism at all, say
so"* — is **refuted**. The diff supports **two**: one confirmed from the compiler (H1) and one
plausible and quantified (H3). This was the stopping rule (b) branch, and it does **not** fire.

The weak form survives and I raised one concrete, readable reason for it: our policy installer
uses `setenv(..., 0)`, the crown's is `setenv(..., 1)`. If the harness exports
`MLX_MAX_MB_PER_BUFFER`, our 512 MiB never applies and theirs does. **Stated as required: this
rests on inference about the plateau's source from published notes, not on source we can
read.** So does anything about their lineage carrying or not carrying E27.

---

## 6. Deliverable 4 — re-test list re-ranked by expected score value

**The re-ranking rule.** From the brief's ladder, score ≈ **0.4827 × (beagle leg %)** for
beagle-only, and ≈ **1.00 × (leg %)** when both bankable legs move (checked against all four
ladder rows: 0.4847, 0.4848, 0.4875, 0.4925 and 0.9994). So

```
E[score value]  =  P(mechanism is real)  x  plausible leg improvement %  x  {0.4827 | 1.00}
```

**This rule was independently confirmed by the advisor after I derived it**, in the ledger-156
corrections: "Score sensitivity to a uniform both-leg MTP speedup is **1.00**, not 0.8114 and not
0.4827 … **0.4827 is the beagle-alone derivative**." The sub-unit both-leg values in the ladder
come from medicine saturating against essays at `raw_p = 3.366118`, which needs a −0.635 % move —
so **`1.00` is correct for every candidate in the table below, all of which are under that
threshold**, and the saturation only matters for a move larger than −0.635 %. I had used
`0.485 | 1.00`; the only change is the third digit of the beagle-alone constant, which reorders
nothing. Two further corrections received and adopted: the crown is reached by **−0.520 %** on
both central legs (not −0.640 %), and thorfinn's M=6 step split (**+15.401** weight stream /
+17.448 residual) supersedes the advisor's earlier +20.590/+12.090 — I use thorfinn's in §9.6.

Entries whose mechanism cannot reach the **wide** verify path on beagle/medicine have
E[score] ≈ 0 no matter how large their effect, because the other six prompts are worth
**0.0000 % each**. `P` values are my subjective credences; the arithmetic is exposed so the
advisor can substitute their own.

| # | entry | reaches wide bankable legs? | plausible leg | P | **E[score]** | cheapest decisive instrument |
|---|---|---|---|---|---|---|
| **1** | **4 — qmm / simdgroup-matrix for M ≥ 4** | **yes — M ≥ 4 *is* the bankable width range** | −3.0 % (conservative; ceiling −61 % on the projection path) | 0.30 | **+0.90 %** | paired local microbenchmark, n = 5, no E2E leg. §7 |
| **2** | **NEW E40-a — split the NA=5 cells off the shared kernel** | yes | −0.36 % (the whole measured deficit) | 0.55 | **+0.20 %** | AIR register ceiling recheck (zero GPU) then 1 paired `--local-iterate` |
| 3 | 5 — wide target-top-2 reducer | yes — "wide" is literally the bankable widths | −1.0 % (discounted from −3.85 % claimed) | 0.12 | +0.12 % | paired microbenchmark of the reducer at M = 4…9, n = 5 |
| 4 | 20 — beagle acceptance | **yes, and beagle is 79 % of all value** | −0.7 % (α .835 → .87 ⇒ R 107 → ~100) | 0.15 | +0.05 % | per-prompt acceptance trace, n = 3 |
| 5 | 13 — replacement head artifact | yes, via acceptance ⇒ width mix | −0.5 % | 0.10 | +0.05 % | our own head A/B, n = 3 + head build |
| 6 | 8 — argPartition top-32 | yes — the top-2 readout is over M rows | −0.35 % | 0.12 | +0.04 % | paired local E2E, n = 5 |
| 7 | 2b — `rows_per_simd ≠ 4` (E2E) | yes, width path | −0.30 % | 0.10 | +0.03 % | paired local E2E, n = 5 |
| 8 | 16 — `values_per_thread`, verify path | yes | −0.20 % | 0.12 | +0.02 % | paired microbenchmark, n = 5 |
| 9 | 18 — KV-1024 warm | **now yes** — E40 shows warm-path changes reach the timed window through the allocator (§2.3) | −0.20 % | 0.10 | +0.02 % | paired local E2E, n = 5 |
| 10 | 10 — host round overhead | **shape-refuted as the deficit cause** (§5.2b); still a real cost | −0.30 % | 0.08 | +0.01 % | paired local E2E, n = 5, ABBA |

**Struck or closed, with the reason:**

| entry | disposition |
|---|---|
| **21 — warm coverage / shape gaps** | **CLOSED WITH EVIDENCE**, not struck for non-existence. E37 delivered it on PR #42, banked 03:44. My E39 entry said "E37 was never delivered" because my base predated it. **The E39 strike-list entry is wrong and this supersedes it.** |
| **11 — command-buffer caps** | **RETIRED by the advisor.** Two independent routes agree; at face value its MTP-leg share is +0.1070 % = 1.1σ. Not re-litigated. |
| **6 — wired limit with headroom** | **RETIRED + hardware-gated.** The 96 GiB guard is now confirmed in source at two sites (§2.3). Unrunnable on this box at any n. |
| **23 — shipped-surface gate** | **standing hygiene.** Not a speed question. It earned its keep here: it is how §2 was verified, and it caught the unresolvable `submissionCommitSha` (§1.6). |
| **H3 / `Memory.clearCache()`** | **NEW, classified, not proposed.** Retired family + 96 GiB gate. Advisor's call. §5.3 |

**Two orderings worth flagging.** Entry 4 stays #1 and gets stronger for a reason the advisor
did not have: it is not merely a width-path entry, it is the **only** entry on the list whose
ceiling exceeds 1 % by an order of magnitude, and §7.1 now proves from source that its
mechanism has **never once executed** on the scored path. And entry 20 rises from 9th to 4th
purely on the order-statistic gate — it is prompt-specific to *the* prompt that carries 79 % of
the value, which under effect/cost ranking looked like a weakness and under expected-score-value
ranking is its main virtue.

---

## 7. Deliverable 5 — the named next experiment

> **Route the M ≥ 4 verify projection through simdgroup-matrix (qmm-style) tiled accumulation
> implemented inside the editable `qmv_fast_crossrow_affine4_g64_*` family, instead of
> NA-packed scalar cross-row accumulation.**

### 7.1 The source finding that makes this concrete — and that verifies H1's path

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1415-1426`:

```cpp
int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
if (M >= vector_limit) {           // matrix-matrix product
  if (transpose_ && B == 1) { qmm_splitk(...); return; }
  qmm(...);
}
```

Every Qwen 3.8 projection has `D > 4096` **or** `O > 4096`, so `get_qmv_batch_limit` always
lands in its third branch:

| proj | D = K | O = N | third branch? |
|---|---|---|---|
| q_proj | 5120 | 6144 | yes |
| k_proj / v_proj | 5120 | 1024 | yes |
| o_proj | 6144 | 5120 | yes |
| mlp.gate / up | 5120 | 17408 | yes |
| mlp.down | 17408 | 5120 | yes |

| arch_gen | arch_size | `vector_limit` |
|---|---|---|
| 13 or 14 | `d` | 12 |
| 13 or 14 | default | **6** |
| ≥ 15 | `d` | 12 |
| ≥ 15 | default | 10 |

`maxDepth + 1 = 9`. **For every modern configuration (arch_gen ≥ 15) `vector_limit` is 10 or
12, both > 9.** Two conclusions, both zero-GPU:

1. **Positive path verification for H1.** Every legal MTP width stays on the **qmv** path, so
   E27's `case 5:` and `case 9:` cells *are* live on the scored surface. §3 is measuring code
   that actually runs. (E27's own M = 9 measurement of 0.8854 on this M4 Pro independently
   confirms `vector_limit > 9` here.)
2. **Entry 4's premise is confirmed from source.** MLX's own `qmm` / `qmm_splitk` is **never
   reached at any legal MTP width**. The +61 % ceiling has never been tested because the
   mechanism has never executed.

**One residual risk, cheaply checkable.** The `arch_gen ∈ {13,14}` + non-`d` cell returns **6**.
If the ranked M5 ever reported that combination, M ≥ 6 would leave qmv and E27's `case 9:`
would be dead code on the ranked box — which would also be an alternative resolution of the
pass-count paradox. It is implausible for an M5 but it has never been verified. **One line of
architecture reporting at worker startup settles it with no timing and no GPU.** I recommend
folding that into whatever runs next.

### 7.2 Why this, and not closing the deficit

Closing the **entire** measured beagle deficit yields only **0.363 %**, so no deficit-repair can
reach the brief's ≥ 1 % bar. The ≥ 1 % must come from a new mechanism, and entry 4 is the only
candidate whose ceiling clears it by an order of magnitude.

It also **subsumes** E40-a: a simdgroup-matrix accumulator uses the matrix register file rather
than 125–129 scalar registers, so it removes the H1 ceiling tax **as a side effect** of the main
mechanism, at every untouched width, without giving back E27's M = 5 (−20.1 %) and M = 9
(−11.5 %) wins. That is strictly better than reverting E27, which would trade a −16.3 %
register-ceiling win for the loss of both.

**Scope is clean and needs no non-editable file.** `kernels/quantized.h` and the
`mlx-generated/quantized.cpp` twin are both editable and both already in the shipped surface.
`vector_limit` itself is in a **non-editable** file — which is precisely why the mechanism must
be implemented *inside* the cross-row cells rather than by re-routing the dispatch. It is also
why E39's closing evidence ("padding across `vector_limit`") measured the wrong thing: padding
M from ≤ 9 up to ≥ 10 is the only *dispatch-level* route, and it doubles the work it is trying
to save. Run `research/twin_audit.py` plus `senpai/validate-assignment-scope.sh` and
`senpai/check-editable-budget.sh` before implementing.

### 7.3 Instrument, MDE, stop rule

**Instrument.** Paired local microbenchmark of the two dominant scored shapes —
`n=5120, k=5120` and `mlp.down n=5120, k=17408` (ledger 129 attributes +92.8 % of E33's net
M = 6 effect to `mlp.down` alone) — at bf16, affine 4-bit group-64, `M ∈ {4,…,9}`,
**ABBA-counterbalanced within one session**, `n = 5` pairs. **No E2E leg for the go/no-go.**

**MDE** (`research/e39_mde.py`, `local_microbench` sd = 0.30, paired, n = 5):

```
MDE (normal, 80 % power, two-sided) = 0.3758 %
MDE (exact noncentral-t, df = 4)    = 0.5040 %      [1.34x the normal figure]
```

Against the ≥ 1 % target that is **2.0× powered**; against the −61 % ceiling, **121× powered**.
Reporting the exact figure, not the normal one, per E39's own finding that the normal
approximation lies at small n.

**Stop rule, pre-registered.** If the simdgroup-matrix variant is **not ≥ 5 % faster** than the
shipped cross-row cell at `M ∈ {5,…,9}` on the `mlp.down` shape, **close entry 4 permanently** —
that discharges E39's "nobody has benchmarked the actual mechanism" and the entry never reopens
without new source evidence. If it is ≥ 5 % faster, escalate to **one** matched
`--local-iterate` pair, then the full exactness chain.

**The real risk is exactness, not speed.** Simdgroup matrix ops change reduction order, and the
trusted parent checks exact top-two row evidence, not just argmax. Validation against the
public golden with exact post-EOS tokens and row-ledger closure must pass **before** any timing
number is believed. That risk is the reason the cheap microbenchmark comes first: it costs no
E2E leg and it can close the entry before any exactness work is spent.

---

## 8. Stopping rule, and what I did not do

Stopped at **(a)**: the diff is classified completely (§2, arithmetic asserted) and one
hypothesis is **confirmed with a mechanism** (§3). Branch (b) did **not** fire — the delta
*does* support a width-dependent mechanism, in fact two — so the campaign is **not** redirected
away from the delta to the kernel. Well inside the 1.5-day bound.

**The ledger-156 lead arrived after (a) had already fired** and was marked "no revision". I
adjudicated it anyway rather than deferring it to a future assignment, because it was fully
answerable from source at zero GPU inside this turn, and because deferring it risked the campaign
spending a GPU leg on a bypass arm that source alone refutes (§9). That is an addition to the
stopping rule's scope, not a violation of it: no new mechanism was implemented and the diff is
still `research/`-only.

**Not done, deliberately:**

- **No GPU, no device, no pipeline.** Compile-only, and I declined even the permitted
  `MTLComputePipelineState` reflection route because E32 already proved it returns 1024 for
  every cell including spilling ones (§3.4).
- **Did not re-open residency / command buffers.** H3 is classified, quantified and handed
  back; I did not design an experiment for it (§5.3).
- **Did not implement anything.** Diff is `research/` only.
- **Did not re-litigate E33.** I cite its M = 6 row only to make the mis-citation reproducible.

**Suggested follow-ups I did not implement:**

1. **One line of architecture reporting at worker startup** (`get_architecture()`,
   `get_architecture_gen()`, and the resolved `vector_limit`). Zero GPU, zero timing, and it
   closes the last residual risk in §7.1 permanently — including the possibility that E27's
   `case 9:` is dead code on the ranked box.
2. **Apply the α-validity gate of §4.2** to every published round-count recovery in the
   campaign. It is silently wrong on drama today and the same bug will recur.
3. **Fix E39's entry 21** on the strike list: it is CLOSED-with-evidence via E37/PR #42, not
   absent. My E39 text is wrong and should be superseded, not merely annotated.
4. If askeladd's census shows beagle spends **> 5.70 %** of decode time at M ∈ {5, 9}, the
   residual is **not** E27, and the only remaining candidate in the delta is H3 — which needs a
   ≥ 96 GiB host or a ranked probe, not another local experiment.
5. **Put width 9 on the exactness gate.** It is the *only* unmeasured width under the inherited
   SDPA chunk (§9.4), the instruction to do so is already written in
   `research/ESTABLISHED_FACTS.md:336-339`, and it is one row-gate run rather than a GPU leg.
   This is the cheapest open correctness risk in the tree.
6. **Give thorfinn the §9.6 cross-check.** His M=6 residual of +17.448 ms is quantitatively
   consistent with the chunk's structural second KV pass (`16 × 4096 × kSplit` bytes). If it
   matches at his measured `kL`, the residual is not addressable and the ψ ≈ 0.59 branch loses
   its mechanism. Free to check, and it bears on three agents' current work.
7. **Run the scope gate against pristine upstream `5d02917`, not only `5273067`.** The
   "+229/−74" denominator silently excludes everything we inherited and ship — which is exactly
   where ledger 156's candidate lived (§9.7).

---

## 9. Addendum — advisor ledger 156: the inherited chunked-SDPA lead, adjudicated

This lead arrived on PR #45 at 05:56 UTC, after §1–§8 were written. It was flagged as a lead
rather than a revision, and it is answerable **entirely from source at zero GPU**, so I
adjudicated it rather than deferring it. The answer is decisive and it is **not** the answer the
brief expected.

### 9.1 The lead is real, and reachability checks out on the scored file

Everything in ledger 156 that I could verify, verified:

| claim | status |
|---|---|
| chunk exists at `AttentionUtils.swift`, gated `qL ≥ 6, qL ≤ 9, kL ≥ qL, B == 1, .causal` | **confirmed**, verbatim |
| introduced by `b6ce964`, `yukon-autoresearch[bot]`, Sat Aug 15 21:15:26 2026 UTC | **confirmed** (`git log -1`) |
| `b6ce964` is an ancestor of HEAD | **confirmed** (`git merge-base --is-ancestor`) |
| it is **not** one of the 5 files in our +229/−74 | **confirmed** — but it **is** `editablePaths[7]`, so it *ships* |
| reachable on the scored verify path | **confirmed** — `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1918` is the **sole** attention entry point of the timed model and calls `attentionWithCacheUpdate` directly |

One correction to the trace: the advisor's `Qwen35Attention:1918` is in the **vendored MLXLLM**
`Qwen35.swift` (the timed target per `program.md`). There is a *second*, similarly named
`Sources/MLXFastModel/Qwen35Attention.swift:200` which also calls the helper. Both reach the
chunk, so the conclusion is unaffected, but only the vendored file is in the shipped surface.

### 9.2 🔴 The decisive fact: `sdpa_full` is unreachable for this model at every width

`ScaledDotProductAttention::use_fallback`
(`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:591-639`)
is the sole dispatch gate. Ported verbatim in `research/e40_sdpa_chunk_price.py`:

```
sdpa_vector_supported_head_dim : head_dim in {64, 96, 128, 256}      :621-624
sdpa_full_supported_head_dim   : head_dim in {64, 80, 128}           :625-626
supports_sdpa_full   = qL > 8  && full_mask && full_hd               :631-632
supports_sdpa_vector = qL <= 8 && qL <= kL && vec_hd
                       && (qL * gqa_factor) <= 32                    :634-637
return !(supports_sdpa_full || supports_sdpa_vector)                 :639
```

Qwen 3.8 27B geometry, from the authoritative fixture
(`fixtures/qwen3_8_27b_mtp_track.json:108-110`): 64 layers, `full_attention_interval` 4 ⇒ **16
full-attention layers**; 24 q / 4 kv heads ⇒ **gqa_factor = 6**; **head_dim = 256**.

**`head_dim = 256` is not in `sdpa_full`'s list. So `supports_sdpa_full` is `false` at every
width, for this checkpoint, permanently** — including prefill. And `qL · 6 ≤ 32 ⇔ qL ≤ 5`:

| qL | qL·gqa | route **without** the chunk | chunk fires? |
|---|---|---|---|
| 1–5 | 6…30 | `sdpa_vector` (fused) | no |
| **6** | 36 | **EAGER FALLBACK** | yes → 5 + 1 |
| **7** | 42 | **EAGER FALLBACK** | yes → 5 + 2 |
| **8** | 48 | **EAGER FALLBACK** | yes → 5 + 3 |
| **9** | 54 | **EAGER FALLBACK** | yes → 5 + 4 |
| 512 (prefill) | 3072 | EAGER FALLBACK | no |

The chunk's halves are `5·6 = 30 ≤ 32` and `(qL−5)·6 ≤ 24 ≤ 32`, so **both** stay fused.

This corrects the brief's framing in one load-bearing way. The advisor wrote that a wide `q` at
`qL ≥ 6` "gives 36 > 32 and cannot" stay on the fused vector kernel — correct — but the implied
alternative was another *kernel family*, and at `qL = 9` specifically that `sdpa_full` would take
over. **It does not.** There is no fused wide path at all. The alternative to chunking is MLX's
unfused eager graph: a materialised `[1, 24, qL, kL]` score tensor with separate
matmul → softmax → matmul. That also explains, at no extra cost, *why* the unfused path drifts —
mega-dmitriy's ~44 % element divergence and polymorf's "41 top-2 VALUE mismatches (ids stable)"
are the eager graph's accumulation order, not a different tuned kernel's.

### 9.3 🔴 Verdict on H5: refuted on **sign**, not on magnitude

> H5: some rivals do not chunk, take the unfused wide path at 6–9, are **faster** there while
> emitting identical ids, and that is our width-gated deficit.

The premise that they emit identical ids is plausible. The premise that they are **faster** is
false from source: their path at every one of M = 6, 7, 8, 9 is the eager fallback, which
materialises scores and cannot beat two fused vector calls. **H5 predicts the wrong sign** — a
non-chunking rival should be *slower* on exactly the wide prompts, so H5 cannot generate a
deficit *for us*. `H5 → 0.02`.

Two further pieces of evidence point the same way, and both are already in our tree:

- **The chunk is promoted prior art we inherited.** `research/ESTABLISHED_FACTS.md:340-345`
  records it as `senpai/qwen38-yukon-submissions-2026-08-16.md` entry **89** (`hadakang`,
  **promoted**, 2.510033): "Cracking the width wall: proven-shape chunking for verify widths
  6–9, depth cap to the trusted maximum". Entry **83** (`polymorf`, **failed**) root-caused the
  same wall to the sdpa `qL` bound. So the advisor's careful hedge — "they probably do [have
  it], since it came from the shared validated-submission lineage everyone forks" — is
  **upgradeable from inference to a named promoted submission.** Common-mode, no differential.
  That note also states in terms: "**We inherited this work; it is not ours and the prize for
  re-doing it is zero.**"
- Ledger 155's 16-digit-identical `effective_mean_draft_len` then closes it: identical
  trajectories plus common-mode code is common-mode cost.

**I therefore recommend against spending the bypass arm's GPU leg.** It would measure a
known-slower, known-inexact alternative, and §9.5 shows our instrument cannot resolve it anyway.

### 9.4 The exactness corner the advisor flagged is provably fine

The advisor named `kL == qL` (where `kSplit` collapses to 5, so chunk A sees a square window) as
"the corner most likely to break the bottom-right-alignment argument". It does not break. Under
bottom-right causal alignment, row `i` of a `(qL, kL)` call sees keys `0 … kL−qL+i`, so:

```
chunk A : qLA = 5,     kLA = kSplit = kL−qL+5
          row i  ->  0 … (kLA−5+i)      = 0 … (kL−qL+i)          identical
chunk B : qLB = qL−5,  kLB = kL
          row 5+j ->  0 … (kL−qLB+j)    = 0 … (kL−qL+(5+j))      identical
```

`research/e40_sdpa_chunk_price.py §2` verifies this exhaustively by index-set comparison for all
`qL ∈ 6..9 × kL ∈ qL..1199`: **0 mismatches.** The `kL == qL` case is included (`qL=9, kL=9 ⇒
kSplit = 5`).

**This proves the mask/window algebra only** — it does not prove bit-equality, because the two
calls have different `kL` and so potentially different tile counts and reduction order. That
residual is where the real answer lives, and it is already recorded rather than open:
`research/ESTABLISHED_FACTS.md:335` states measured **bit-exactness on the hexfloat row gate for
widths 6..8**, and `:336-339` states that **width 9 / depth 8 is covered by the chunk by
construction but has no in-tree measurement**, with the standing instruction that width 9 must be
on the exactness gate because any gate change raises its firing rate. That — one row-gate run at
width 9 — is the entire outstanding exactness question, and it is far cheaper than the bypass arm.

### 9.5 The price the advisor asked for

> "how much does the chunk cost us at widths 6–9, and is that cost the size of the deficit?"

Answered two ways, because the comparator matters:

- **Against the true alternative** (the eager fallback): the price is **negative**. The chunk is
  a saving and an exactness fix.
- **Against a hypothetical single fused wide call** (unreachable — see below): the price is
  **one extra pass over the KV cache**, because chunk A reads `kSplit ≈ kL` rows and chunk B
  reads all `kL`.

With 16 full-attention layers, 4 KV heads, head_dim 256, bf16, K and V ⇒ 4096 bytes per cached
row per layer, so the extra pass is `16 × 4096 × kSplit` bytes:

| kL | extra | µs @ 273 GB/s (this box) |
|---|---|---|
| 512 (leg start) | 31.9 MB | 122.7 |
| 768 (leg mean) | 47.9 MB | 184.1 |
| 1024 (leg end) | 63.9 MB | 245.6 |

Beagle: `R = 107`, leg ≈ 6232 ms (22.63 ms / 0.3631 %). Even if **every** round were wide:

```
f(rounds M>=6)   0.25     0.50     0.75     1.00
total ms         4.92     9.84    14.76    19.68
% of beagle leg  0.0789   0.1578   0.2368   0.3157   <- CEILING
```

**Ceiling 0.3157 % of the beagle leg, against MDE(exact, n=5, paired) = 0.5040 %: ratio 0.63.
The chunk's price is undetectable by our own instrument even at its ceiling**, and the true
price is lower still — KV for one layer at `kL = 768` is only 3.00 MB, chunk A and chunk B run
back to back on the same layer, so an SLC-resident second read makes the real cost far below the
bandwidth bound. That factor is unmeasured, which is another reason not to buy the measurement.

And the lever is **closed anyway**: the `≤ 32` bound is `n_simds = gqa_factor · qL` against a
1024-thread threadgroup (`group_dims(32, gqa_factor, qL)`, `:484`), and both that bound and the
`use_fallback` predicate live in `backend/metal/scaled_dot_product_attention.cpp`, which is
**not in `editablePaths`** — while `sdpa_vector.h` and `scaled_dot_product_attention.metal`
(`editablePaths[75], [76]`) are. **This is the same structural wall as §7.1's `quantized.cpp`:
we may edit the kernel but not the host predicate that decides to dispatch it.** Fewer chunks is
impossible; a cheaper split is not (minimising total KV rows over legal splits saves 1 row in
~1500). `ESTABLISHED_FACTS.md:331-333` separately records that whole-forward segmentation was
measured bit-exact but pays a ~25 ms second weight pass and loses on net — do not re-propose it.

**So: it is not the size of the deficit as a differential (it is common-mode), and as an absolute
it is ≤ 0.32 % of one leg, unreachable, and below our detection threshold.**

### 9.6 What this *is* worth: thorfinn's unexplained M=6 residual

The one genuinely valuable consequence is a cross-connection to the ψ/φ work. thorfinn's M=6
step splits as `+32.850 ms = weight stream +15.401 + residual +17.448`. The chunk fires at
**exactly** M = 6, and it is the only op in the round whose byte traffic steps there:

```
residual 17.448 ms / R = 107 rounds  =  163.1 us/round
source-derived chunk bound           =  183.9 us/round      ratio 0.887
```

**The residual half of the M=6 step is quantitatively consistent with the chunk's second KV
pass.** If that holds, the residual is **structural and irremovable inside the editable
surface** — not a defect, and not a fix that can be priced. Since thorfinn, askeladd and edward
are jointly deciding whether ψ is nearer 0.59 or 0.23 on the assumption that the residual is
addressable, this is decision-relevant to all three.

**Stated as required, this rests on inference: I do not have thorfinn's definition of his step,
so the `/107` normalisation may not be the right one.** It is a coincidence with the right
mechanism and the right threshold, offered as a falsifiable cross-check, not a result. The check
is free — compare his residual against `16 × 4096 × kSplit` bytes at his measured `kL`.

### 9.7 Effect on the rest of this report

None of §1–§8 changes. H5 is a fifth hypothesis about the *inherited* surface rather than the
+229/−74 shipped delta, so it does not disturb the §2 elimination ladder, and it does not
compete with §7's named experiment: entry 4 stays #1 at **+0.90 %** expected score value, versus
a closed lever worth ≤ 0.32 % of one leg. The one thing it does change is that **the "5 files,
+229/−74" framing is now known to be the wrong denominator for this class of question** — the
gate reports what we changed, not what we inherited and ship. The advisor identified that
himself; §9.1 confirms it, and the fix is cheap: run the scope gate against the *pristine
upstream* baseline `5d02917`, not against `5273067`, whenever the question is "what could cause
this", and against `5273067` when the question is "what did we do".

---

## 10. Reproduction

```bash
# instrument audit (board dump -> item 148 re-derivation + reference classes)
curl -s -H "Authorization: Bearer $YUKON_API_TOKEN" \
  'https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?limit=2000' \
  > /tmp/rows_live.json
python3 research/e40_instrument_audit.py /tmp/rows_live.json

# deliverable 1: hunk classification, arithmetic asserted against the gate
bash   research/shipped-surface-gate.sh
python3 research/e40_hunk_classify.py

# deliverable 2: compiler-derived register ceiling (compile-only, zero GPU)
bash research/e40_entry_air_diff.sh      # production entry point, both arms
bash research/e40_cell_air.sh            # every dispatch cell + anchor

# deliverable 3: joint-consistency feasible set
python3 research/e40_width_tax_feasibility.py /tmp/rows_live.json

# H2/H3 magnitude and shape refutation, with the alpha-validity gate
python3 research/e40_overhead_budget.py /tmp/rows_live.json

# section 9 addendum: sdpa dispatch truth table, chunk window-equivalence proof,
# chunk price, and the thorfinn M=6 residual cross-check (all zero GPU)
python3 research/e40_sdpa_chunk_price.py

# deliverable 5 MDE
python3 research/e39_mde.py --mde --sd 0.30 --n 5 --design paired
```

| item | value |
|---|---|
| local host | Apple M4 Pro, 48 GiB (51 539 607 552 bytes) |
| Metal toolchain | Apple metal version 32023.883 (metalfe-32023.883), air64-apple-darwin25.5.0, SDK MacOSX26.5 |
| ranked head | `559b24eb…` |
| our ranked row | `ca9251b8-58cd-4d90-9a52-fa05f5657216`, score 3.23250848263467, rejected, rank 9/408 |
| our `submissionCommitSha` | `2b0c36a078b7660c9215adee933336ff46da25af` — **not resolvable locally** (§1.6) |
| board top | 3.24929398547457 |
| cohort | 89 analysable all-8 rows on `559b24eb` (not 94) |
| GPU minutes | **0** |
