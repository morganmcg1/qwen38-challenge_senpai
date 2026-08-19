# Advisor corrections to the four live assignments, 2026-08-19

**Why this file exists.** The GitHub REST API returned HTTP 403 for the whole
turn after `ca9251b8` resolved, so I could not post these as PR comments. They
are important enough that they must not wait. Detail and receipts are in
`senpai/campaign-ledger.md` items **101–114**; this file is only the "what
changes in your assignment" layer. I will also post them as PR comments once the
API recovers — if you see both, they are the same content.

**All four briefs (PRs #38–#41) contain a calibration section that is now wrong.**
Every one of them says the campaign has never had a valid ranked measurement and
that `ca9251b8` "is still validating." It has resolved. Read §0, then your own
section.

---

## §0 What applies to all four of you

`ca9251b8` = **3.23250848263467**, `rejected` with `score did not improve current
best` — the benign class. It is the **first Senpai row ever to run the declared
head artifact** `559b24eb`.

| quantity | value |
|---|---|
| our score | **3.23250848263467** |
| **rank** | **9 of 408 scored rows** |
| board top | 3.24929399 (`0cd0a6b4`, ofou) |
| official gap to #1 | **0.517 %** |
| **engineering gap after serial-normalising** | **0.258 %** |
| official top-10 span | 0.521 % |
| **serial-normalised top-10 span** | **0.211 %** |
| noise to expect on a ranked confirmation | **±0.2 %** |

1. 🔴 **The "5.862 % median gap" framing is retired.** It was measured on our LR3
   head `2f6805e1` and was almost entirely head artifact, not kernel work. Any
   sizing in your brief derived from it is void — most importantly the
   "+8 % to +25 % of score" figure in E33.
2. 🟢 **Acceptance is a closed lever, by exact measurement.**
   `effective_mean_draft_len` is **bit-identical to the board top on 8/8
   prompts** (`+0.000 %` each): plutarch 0.1540, drama 2.2976, travel 2.6557,
   beagle 4.5327, medicine 4.7677, republic 5.2697, essays 5.4253, botany
   5.7765. `qwen_mtp_weights_hash` identical, `parity_all_ok` true. Of the 88
   rows on this head, **73 carry that exact fingerprint** and span **2.3256 →
   3.2493 (+39.7 %)**. Same tokens, 39.7 % of score range: **per-row cost is the
   entire game.**
3. 🔴 **~60 % of the visible ordering at the top of the board is serial-leg
   variation, not engineering.** Serial per-leg sd 0.18 %, run-level sd 0.10 %,
   `corr(serial, mtp) ≈ +0.04` (no box-speed effect), and
   `corr(serial, score) = +0.695` inside the top cluster. Tools:
   `research/serial_normalise.py`, `research/serial_variance.py`. The competitor
   folklore figure of "±1.5–3 % ranked noise" is contradicted by our own
   population — do not use it.
4. 🔴 **`R = serial / mtp`, so the serial leg is the numerator.** An optimisation
   that fires on the M=1 serial path **lowers** our score. Shape-gate everything
   away from it. Reverting or gating *our own* change is legitimate engineering;
   deliberately injecting a pessimisation is benchmark gaming and
   `Review submitted code for benchmark bypasses` is a live rejection class.
   **But do not hunt for a self-inflicted serial optimisation of ours — I
   believed we had one worth +0.26 % and falsified it myself (ledger 111).**
5. 🔴 **RETRACTION I owe all four of you.** I claimed no note among 627 mentions
   IPG, `rows_per_simd`, `ceil(M/IPG)`, NA, or widths 5/6/7, and I repeated it in
   all four briefs as "nobody else is here." It is false. Correct scan of all
   628 notes: **`crossrow` 175, `qmv_fast` 147, `IPG` 96, `warmAllDepths` 78
   (including ranks 1–4), `weight stream` 62, `sdpaWidthWallDepthCap` 48,
   `values_per_thread` 13, `rows_per_simd` 3.** My grep was broken and I
   published its zero-hit output as fact. **A zero-hit corpus scan is a claim
   about my tooling until it is validated on a known-positive.** What survives is
   narrower and still real: the specific row-blocked `4/r` reformulation and the
   wide-4-bit `values_per_thread` cell appear in no note.
6. 🔴 **A ranked-measured correctness boundary.** FP32 reassociation is licensed
   on the **coarse draft** path and forbidden on the **verify** path. Lieisyourlie
   (rank 8) relies on the former; companygardener tried the latter, scored
   3.22444, was rejected, and their next accepted row (3.24326) says only *"Stay
   off the verify reduction tree."* The public fixture **can pass while a hidden
   prompt flips a near-tie argmax**, so this is not detectable locally.
7. 🔴 **`mtp_depth: 8` and `mtp_max_draft_depth: 8` both appear in
   `officialMetrics`.** Because both are 8 they may be configuration echoes
   rather than realised depth. So my inference "n = 4.533 implies depth 5 at
   p ≈ 0.965" is **unconfirmed, not established** — I stated it as fact in E33
   and E34. Withdraw it until edward settles it from source.
8. 🔴 **`headStepCostRatio` 0.18 → 0.16 is measured at rank: 3.19088 → 3.15370,
   −1.164 %**, one constant, everything else byte-identical (`a1326b4b`).

Where we actually lose, measured. MTP-leg penalty vs the top, by realised draft
length:

| prompt | n | our MTP leg vs top |
|---|---:|---:|
| plutarch | 0.154 | +0.049 % |
| drama | 2.298 | −0.008 % |
| travel | 2.656 | −0.037 % |
| **beagle** | 4.533 | **+0.418 %** |
| medicine | 4.768 | +0.110 % |
| republic | 5.270 | **+0.426 %** |
| essays | 5.425 | **+0.506 %** |
| botany | 5.776 | +0.206 % |

**Zero penalty on every narrow prompt; the entire deficit is on the wide ones.**
Our beagle `mtp_seconds_per_token_mean` (0.0121740) and beagle h̄ (0.16738 vs the
top's 0.16452) are both the **worst of the top 12**.

---

## §1 thorfinn — E33 (PR #38), the main line

- **Corrected sizing, replacing "+8 % to +25 %":** a 5.4 % reduction in h̄ on the
  central pair (your 0.82 ratio on M=6 rounds carrying ~30 % of round mass) gives
  beagle 3.1947 and medicine 3.4215 ⇒ **3.30809, +1.81 % over the top.** Still
  large. But state your kill criterion against a **0.258 % engineering bar and a
  ±0.2 % noise floor**, and note that a clean 0.3 % is now decision-relevant.
- 🟢 §0's per-prompt table is **ranked-side confirmation of your target** — the
  deficit is entirely in the wide widths. Not modelled; measured.
- ⚠️ Counter-weight: E27 made M=5 single-pass and measured −6.56 % E2E locally,
  and we still hold the cluster's **worst** wide widths. Either E27's local gain
  did not transfer, or it did and we would otherwise be much further back. There
  is no control row on this head to distinguish them. **Apply the same suspicion
  to your own local E2E number** — the per-width cost table remains primary.
- 🔴 **Deliverable (c) is upgraded by §0.6.** It is no longer "check tokens
  match": **demonstrate from the code that row blocking preserves the per-row
  K-reduction order and the `simd_sum` pairing exactly.** I expect it does — you
  change which rows a thread visits, not the within-row accumulation order — but
  it must be shown, because the local fixture cannot detect the failure.
- Confirm from the dispatch gate that a row-blocked NA=6 crossrow cell **cannot
  fire at M=1**, and keep treating a serial-leg speedup as a red flag.
- Unchanged: rung 1 = M=6 only, 117 registers vs the 125 the shipped `<T,5,5>`
  already uses; my registered prediction stays **0.82**.

## §2 edward — E34 (PR #39)

- 🟢 **Deliverable (a) is answered on the ranked side, in your favour.** §0.2
  shows our ranked acceptance is *already exactly* the leader's. There is no
  ranked acceptance gap; the 0.6926-vs-0.82 discrepancy is purely a property of
  your local fixture and its head artifact (`d038fd41` ≠ declared `559b24eb` at
  427,742,600 B). (a) becomes a **bounded confirmation write-up** — document what
  `research/fetch-declared-head.sh` fetched (tree, digests, byte counts, file
  count) — and I withdraw "it is worth more than the rest of this assignment."
  **(b)–(e) are the experiment.**
- 🔴 **§0.8 closes the `h` direction at rank.** Treat `headStepCostRatio`
  recalibration as dead. It does **not** close (c): `sdpaWidthWallDepthCap` 5 → 4
  pushes capped rounds *shallower* (M=6 → M=5), the opposite sign, and is now the
  only depth-policy lever with ranked evidence pointing the right way. **(c) is
  your strongest deliverable.**
- 🔴 **§0.7 is the new first step of (d)**, and it flips the conclusion:
  realised depth 5 ⇒ p ≈ 0.965, `0.965⁵ = 0.842`, streaks common, wall rarely
  binds, **(c) worth little**; realised depth 8 ⇒ p ≈ **0.871**,
  `0.871⁵ = 0.501`, two in a row ≈ 0.25, wall binds most rounds, **(c) worth a
  lot**. One source read decides it. Report the read and the arithmetic.
- **Validate before predicting:** your identity must reproduce our beagle
  R = 3.12015 from n = 4.5327 and h̄ = 0.16738 **exactly**. First check.
- 🟢 §0's step pattern — zero penalty on narrow prompts, all of it on wide ones —
  is direct support for the **pass-count-aware cost model in (e)**, now the
  deliverable I most want. A smooth price in depth cannot produce that pattern;
  `ceil(M/IPG)` can.

## §3 alphonse — E35 (PR #40)

- 🔴 **The noise handle I gave you does not exist.** Grouping all 88 rows on head
  `559b24eb` by `submissionCommitSha` yields **no group with more than one row**,
  so the `c0e34afd..5068eb8` same-code pair idea for (d) is void. Replacement,
  measured (ledger 111): serial grand mean 0.0379908, between-run sd 0.1207 %,
  within-run sd 0.1766 %, implied **run-level σ ≈ 0.10 %** on top of per-leg
  0.18 %; `corr(serial, mtp) ≈ +0.04` ⇒ legs independent.
- 🔴 **h̄ inherits the serial leg's noise**, because it is derived from
  `R = serial/mtp`. A mechanism→h̄ join that does not remove the serial numerator
  will attribute a competitor's serial luck to their mechanism. **Compute h̄
  twice — raw, and against the per-prompt grand-mean serial numerator — and
  report both.** If a mechanism's value survives normalisation you have
  something; if not, you have found (d). Spend your effort there, not on breadth.
- Use the **73-row fingerprint-matched** subset as the primary population (I said
  87; the correct counts are 88 on-head, 73 fingerprint-matched).
- Our anchor row: beagle n 4.5327, R 3.12015, **h̄ 0.16738** (worst of top 12);
  medicine n 4.7677, R 3.34486, h̄ 0.14894; top beagle h̄ 0.16452.
- 🟢 **Ground truth for your join:** Lieisyourlie's published dead-axis table —
  fusion restore 3.21191 · AndNormed warm 3.20407 · mid-width nibble 3.18005 ·
  **h 0.18→0.16 3.15370 from base 3.19088** · mid-width on welt 3.12613 · cold
  prior 3.07827. **Your pipeline must recover −1.164 % for the `a1326b4b` pair.**
  If it cannot reproduce a known one-constant result it cannot be trusted on
  unknown ones.
- Corrected anchor: scarletbright's `values_per_thread = 32` (+1.84 %) is on the
  **2-bit single-row draft readout** (`bits == 2 && out_vec_size == 98336 &&
  ntg.x == 1`), not a wide-4-bit result.
- One bounded addition: report **the per-run serial-leg offset beside each
  mechanism**. Any note claiming a mechanism that should touch the M=1 path is
  directly checkable against it.

## §4 askeladd — E36 (PR #41)

- 🔴 **Do (e) FIRST — the answer may exist and may be fatal.** §0.6 plus
  Lieisyourlie's own numerics: *"The wider lane coverage reassociates the FP32
  partial-sum tree (32 products accumulate per lane-interior before the k-block
  add, versus 16 …). That perturbs rounding at the last-ulp level. It is legal
  **for this stage** …"* Raising `values_per_thread` provably changes the
  K-tiling and therefore the accumulation order — licensed on the draft path,
  **forbidden on the verify path, which is where your wide 4-bit crossrow target
  lives.** So the question becomes **"can `values_per_thread` be raised on the
  wide crossrow QMV without altering the per-row reduction order?"** If not, the
  axis is closed on the only path worth optimising, and that is a complete
  terminal negative reached in a day instead of a week.
- 🟢 **Free host-legality result — verify my arithmetic rather than trusting it.**
  Lieisyourlie quotes the promoted contract: *"`values_per_thread = 32` … one
  `uint64` load … **`block_size = 1024`, five k-blocks over `K = 5120`**."*
  Values per k-block per row = `values_per_thread × 32`, and `K` must divide it:

  | vpt | values/k-block | `5120 /` | legal? |
  |---:|---:|---:|---|
  | 8 | 256 | 20 | ✔ |
  | 16 (shipped) | 512 | 10 | ✔ |
  | 32 | 1024 | **5** | ✔ — matches their "five k-blocks" |
  | **64** | 2048 | **2.5** | ✘ `5120 % 2048 = 1024` |

  So **vpt 64 looks host-illegal**, removing a quarter of your grid. Also
  `bytes_per_lane = vpt × bits/8`: at `bits = 2` vpt 32 is 8 bytes (their single
  `uint64`), but at **`bits = 4` vpt 32 is 16 bytes** — twice the shipped
  footprint. Your register model must use the 4-bit figure. **Check the k-block
  loop and the tail/`_m` instantiation: if a ragged final block is handled, vpt
  64 may be legal after all, and that would be a better result than my table.**
  I have been wrong twice about this kernel's structure and you corrected me both
  times.
- 🟢 **Your target survives §0.5.** Lieisyourlie states that *"every `bits == 4`
  dispatch cell, including the live M=8 `4+4`"* stays as shipped; their vpt work
  is confined to the 2-bit single-row readout. The wide 4-bit crossrow MLP QMV —
  59 % of verify time — is untouched by anyone. And "NAX" in their dead-axis list
  is Apple's M5 neural-accelerator path, **not** our NA (ledger lines 258/310),
  so your axis is not pre-killed.
- Re-weight scarletbright's +1.84 % once more: evidence that the axis **does not
  spill** on ranked M5 hardware, and no evidence of its value on a 4-bit verify
  path.
- Their `ntg.x == 1` gate is a textbook instance of §0.4's shape-gating doctrine.

---

## Queued for the next round, not yet assigned

Verified in source this turn (ledger 113–114): the SDPA dispatch switches kernel
families at **KV ≥ 1024** (`scaled_dot_product_attention.cpp:745-753`,
`sdpa_vector` → `sdpa_vector_2pass`) while our shape warm seeds the throwaway
cache at exactly **512** (`Qwen36MTPBlockSession.swift:308`). The track is 512
seed + 512 decode, so the live KV length crosses the boundary **mid-window** and
the first `sdpa_vector_2pass` dispatch lands inside the scored window. We already
fixed the 8 → 512 instance of this same bug and stopped one token short.
`warmAllDepths` is in an **editable** file and currently **times nothing**.
Open uncertainty: the branch is gated on architecture character `'d'`/`'s'`,
unestablished on both hosts — one print settles it.
