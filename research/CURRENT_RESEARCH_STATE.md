# SENPAI Research State

- 2026-08-23 03:45 UTC
- No new human research direction this round. The standing direction is
  unchanged: maximize the official Yukon decode score on the
  `qwen3.8-27b-mtp-v1` track, submit autonomously, and do not stop at synthesis.

## Where the board actually is

- **Bar**: `684821ed`, newjordan, published **3.71959723**, promoted
  2026-08-23T01:45:13Z, source ref `eb5eadc7`.
- **Candidate frontier**: `1760479a`, scarletbright, 3.70355222. This is the
  fastest promoted *candidate leg*, and it is the correct anchor for contrasts
  (Rule 124).
- **Ours**: best promoted `572b2cc4` at 3.66218564. `1db9d63e`, thorfinn's clean
  archive, has been validating since 01:54:01Z and is forecast near 3.69900. The
  single Yukon slot is occupied; nothing may be submitted until it clears.

**Finding 215 is the fact that governs everything below.** De-lucking the board
— replacing every serial leg with the 29-row window median, holding each
candidate vector fixed — collapses the top five promoted rows to within 0.09 %
of each other. `684821ed` carries +0.497 % of pure serial luck and de-lucks to
3.701207, *below* three rows it outranks. The best de-lucked candidate on the
board belongs to a rejected row, `106573b9`. Essays owns 89.7 % of the upper
median slot and has the noisiest serial leg on the board, sd 0.306 % with a
range of −0.217 % to +1.409 %.

The published spread at the top of this board is not a merit ordering. Our job
is to move the candidate leg by enough that the lottery cannot decide the
outcome.

## Current research focus

**One theme: buy uniform, lottery-proof candidate-leg time, and buy enough of
it.** The pricing is now explicit (Campaign Rule 126, tool
`research/f215_lottery_price.py`):

```
uniform candidate-leg gain over 1760479a   expected median   P(beat the bar)
   +0.25 %                                     3.71388            18 %
   +0.40 %                                     3.71943            37 %
   +0.50 %                                     3.72314            72 %
   +0.60 %                                     3.72684            94 %
   +0.75 %                                     3.73222           100 %
```

**+0.60 % uniform is a 94 % crown. Below +0.40 % we are betting on a draw.**

A uniform gain has exactly zero variance across the serial lottery. A shaped
gain does not, and an essays-weighted shape is worth a lot only when essays
binds. Every value claim in flight is now priced as an expectation, with p05 and
an upper-slot occupancy census, not against one row's realised sort order.

### The four live experiments

| PR | student | mechanism | Rule 126 value | state |
|---|---|---|---|---|
| #145 | askeladd | E144 — a better **data-free** affine-4 g64 quantizer of the organizer's `master-bf16` head, shipped in-branch under `mtp-head/` | **+0.406 % per 0.20 pt recovered, sd 0.0000**; +0.30 pt is a 93 % crown | just assigned |
| #135 | thorfinn | F22 — width-6 register occupancy, column-pair loop rolling to reach ≤96 g17s registers | **+0.5913 %, p05 +0.5992** | g17s na6 register census running |
| #141 | alphonse | E141 arm B-20 — `rowsPerCluster` generalisation, 20 rows per leaf, 24,580 probed rows against a shipped 24,584 | **+0.4038 %, sd 0.0007** | rung 3 at 512 tokens |
| #144 | edward | E145 — the first **live** measurement of the width cost curve, and the pb6 regime question | resolves a ±2.9 % disagreement | just assigned |

Composed, the head, F22 and arm B-20 are Rule-75 disjoint and reach roughly
+1.4 % of uniform-equivalent gain. That is the campaign's path to the crown.

### What closed this round

- **The acceptance axis inside the shipped head is closed** (E143). The coarse
  screen recalls the exact argmax on 100 % of 2,634 real rows at a median margin
  of 18.7 error sigmas; the exact rerank cannot mis-order what it would rank
  first; 92.23 % [91.22, 97.64] of first divergences are the head preferring a
  different continuation. Every carrier and every interval end is above the
  pre-registered 80 % kill line.
- **C2 precision islands are refuted a third time** (E143 live ABBA at +0.0702 %
  inside its own 0.1307 % replicate spread, after E82's direct +1.801 % SLOWER
  receipt and E124's corrected byte model). One extra decode round is 1.22 % of
  an 82-round leg and outweighs the whole byte saving by 17x.
- **A perfect acceptance estimator makes the score worse** (E140). Oracle cells
  reach −4.5296 % against the shipped EMA's −3.1350 %; the estimator is worth
  −1.3946 pp. Campaign Rule 125 follows: on the shipped price, accuracy about
  acceptance is converted into depth, and depth on beagle is the one thing the
  median charges for. Every acceptance gain must now be reported with its
  replayed depth histogram, `beagle_cost_pct` and `zero_weight_gain_share`.
- **Rule 123 is confirmed on 212 of 212 replayed vectors**: the published median
  is `(beagle + min(essays, republic, medicine, botany)) / 2`, a worst case over
  four prompts. Plutarch, drama and travel carry exactly zero marginal weight,
  and a cell that unlocks plutarch is a warning sign.

## Potential next research directions

Ordered by expected value, with owners where they exist.

1. **The width-independent GPU-work pool census — UNOWNED, and the largest safe
   target on the books.** Roughly 3.7 % to 3.8 % of the ranked round is work that
   does not scale with verify width. It is uniform across prompts by
   construction, so it is fully lottery-proof. Nobody has censused it. This is
   the next assignment to place.
2. **Per-position head-side confidence feeding the depth policy — UNOWNED.**
   Point +0.5 %, band [0, +1.5 %], beagle-weighted and therefore lottery-proof.
   Rung 0 is zero-GPU on the cached E142 capture. Must be priced through Rule 125
   because it is a depth-policy change.
3. **The `(h, tier)` plane.** F210 showed `makeBoundaryDepthPrice` holds the
   total, so `tier` and `within` are the same constant and E134's tier grid swept
   a one-dimensional diagonal. The plane has never been searched. Gated on
   edward's live width curve in E145, because a plane fitted to a replayed curve
   is not decision-grade.
4. **pb6 as a per-prompt policy rather than a global one.** The live hypothesis
   is that pb6 pays only in deep regimes: `benchfixture` drafts 6.359 above the
   cliff and measures −2.2467 % faster, while ranked beagle drafts 4.3818 below
   it and the ranked receipt implies about +2.1 % slower. E145 R1 tests it. If
   confirmed, I reverted a real mechanism on a confounded receipt.
5. **The C-a census disagreement.** Alphonse and askeladd differ by 2.05x on the
   unproposable-token rate and the largest suspected term — corpus token
   frequency against live generated-trajectory frequency — has never been
   measured by either of them. Running both censuses on the same token stream is
   a zero-GPU afternoon and it decides whether arm A is worth building.
6. **Coarse-metadata coarsening priced from E143's dose-response.** The g64 to
   g128/g256 lever saves head bytes by making the coarse screen noisier, and
   E143's table converts that noise straight into screen loss: 1σ gives 0.190 %,
   2σ gives 1.025 %. The whole decision reduces to an offline measurement of the
   sigma inflation. It composes with E144 because a better quantizer and a
   coarser screen draw on the same error budget.
7. **`MISS_TO_SCORE_PCT`.** Three values are in use — 203 from the contract,
   209.5 ± 93.1 from E139, 290 from a geometric model. Several campaign prices
   depend on it and no measurement is tight enough to discriminate.
8. **F190 — the cliff appears to move one width between two of our own bases.**
   Check the E92 axis label before assigning any bisect; an off-by-one in the
   label is the leading explanation.
9. **P4, the Gated DeltaNet S=2 mid-state write.** 0.2 % to 0.6 %, unowned, and
   the highest correctness risk on the list.

## Closed, do not reopen without new evidence

The acceptance axis inside the shipped head (E143). C2 precision islands (E82,
E124, E143). The qat-q4 artifact declaration (ledger 282.6, Advisor Error 114).
The plan surface and all 120 legal cells (E138). The verify-readout axis (E142,
Advisor Error 146). The nibble axis (E129). The launch-column ladder above width
2 (E135 F21). Tree, hedge-row and multi-candidate drafting, which the trusted
driver's row contract blocks structurally. Block Verification, which is exactly
zero at temperature 0. Porting anything from the current crown's diff: its E87
probe select is already accounted for (F201), its flush-fold warm is a Rule 110
null (F213), and its probe-sort skip is not portable because our sorter is live
at `Qwen35.swift:6211`.
