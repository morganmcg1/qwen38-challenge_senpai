# SENPAI Research State

- 2026-08-20 16:35 UTC
- Most recent human direction: issue #22 — execute aggressively toward the
  winning frontier. No new human instruction since.

## Where the campaign stands

Official frontier `c6af1e24` = **3.30955573** (organizer `88578f9295523af1`,
accepted 2026-08-20T14:40). Our best official run `9b241879` = 3.23588901, a
deficit of **2.29 %**. The ranked slot is free. Nothing of ours is in flight.

The advisor branch tracks the frontier verbatim and adds four items the
frontier does not have: the VERIFY-CONCAT int32 JIT warm, inert trace
instrumentation, inert DepthPrice machinery pinned at `.ship`, and the
stopTokens fixed-window continuation fix. Composed as it stands we would land
at approximately the frontier, so **there is nothing to submit yet**. The slot
stays free until we hold a candidate-side speed win.

## Current focus: the fixed cost of a drafting round

Ledger 221 is the governing result. Four submissions ran the bit-identical
eight-prompt schedule. plutarch — 449 of 487 rounds non-drafting — was flat to
±0.1 %. Every prompt that drafts on every round was 1.2 % to 2.5 % slower. The
spread is a **fixed cost per drafting round of ≈0.86 ms**, near-constant across
prompts whose draft lengths differ by 2.7×.

Removing 0.86 ms per drafting round is worth **+1.43 % of official median**.
The entire remaining schedule axis is worth **+0.46 %**.

The mechanism is already proven once. The only net code difference between
`89cbdc02` (3.255922) and `c6af1e24` (3.309556) is `qwen35DualRMSNorm`, a
two-output Metal kernel that runs the proposal head's two independent pre-fc
RMSNorms in one dispatch instead of two. Eight of eight prompts faster, mean
−1.145 % candidate seconds per token, sign test p = 0.0039, **+1.647 % of
median**. Two 50 KB norms cannot be a bandwidth story, so the saving is an MLX
graph boundary on a path the session synchronises.

**H-221:** the proposal-head path pays of order 0.35 ms per MLX op boundary,
not per kernel launch. Any fusion that removes one op from the once-per-round
head flush is worth of order +0.5 % of median.

## Closed axes — do not reopen without a named new reason

- **Schedule.** +0.46 % total from the new frontier, and two of the four
  relevant per-prompt floors were set by custom heads. beagle is 0.19 % from
  its 601-run floor.
- **Registers** (bound −1.209 %), **occupancy** (0.52 %), **the copy family**
  (ceiling 0.016 %), **low-rank draft readout** (full rank is load-bearing),
  **head fine-tuning and distillation on prose corpora** (six ranked
  negatives), **the depth-price level** (bracketed both sides on rank).
- **Narrow-N M=1 crossrow routing** — exact and free, but a measured null
  (4 prompts faster, 4 slower, mean −0.09 %).

## Next research directions, in priority order

1. **Delete the pre-fc concat.** `qwen35DualRMSNorm` already computes both
   halves; write them into one buffer of width 10240 at column offsets 0 and
   5120 and `fc` consumes it directly. Removes one MLX op boundary AND a real
   `rows × 10240` bf16 copy. Bit-identical by construction, proposal-only,
   about thirty lines. Highest expected value on the board.
2. **Census the whole proposal-head chain for op boundaries.** Count MLX ops
   per drafting round on the head path and price each removable boundary at the
   0.35 ms unit from H-221. The gather, the concat, the KV pack, the per-draft
   norm, and the top-32 rerank are all candidates.
3. **Separate per-boundary host cost from per-kernel GPU time** in the E80
   census. The gap between summed kernel GPU time and wall-clock round time IS
   the boundary overhead, and it is now the campaign's main quantity.
4. **The narrow dispatch switch** at `quantized.h:1980`. Every promoted kernel
   change in this campaign has been on the wide switch behind
   `out_vec_size >= 4096`. The narrow switch serves proposal-head shapes only
   and cannot touch the serial numerator. M=1 there is a null; the rest of the
   table is unexamined.
5. **Prefill.** 8.6–9.4 % of the scored candidate leg, never moved in 470 board
   runs, worth +0.903 % per 10 % cut. Under test as E83. The prior is that it
   is close to the quantized-GEMM roofline and therefore closed, but both
   inherited constants behind that prior were never measured end to end.
6. **Head bytes, not head quality.** Head cost ×0.75 is +2.11 % of median and
   the head step is bandwidth-scaling (2.05× cost for 1.99× traffic). No custom
   head has ever beaten the pinned head on beagle across ~40 digests, so the
   lever is bytes moved per draft step, not acceptance.
7. **Entropy-gated early stopping of the draft chain** (AdaEDL,
   arXiv:2410.18351), training-free, +10–57 % reported. Untried here.

## Live experiments

| PR | student | question |
| --- | --- | --- |
| #85 | thorfinn | E83 — decompose the untouched prefill leg |
| #83 | edward | E80 — per-kernel GPU-time census, name the unattributed 22.6 % |
| #84 | alphonse | E82 — requantize the one genuinely retrained published head |
| #81 | askeladd | E78 — width-dependent QMV inner-group count |

## Standing measurement rules

- **plutarch alone is the host/serial-speed control.** The median-of-three
  build factor is demoted: drama and travel carry the most per-drafting-round
  cost on the board and report it as host speed.
- **Score = (beagle_raw + slowest other wide prompt) / 2.** The 4th slot is
  beagle in 100 % of strong trees. A single prompt's ordinary variation can
  move the median; judge a change by the mean over all eight prompts and a sign
  test, never by the score alone.
- Same-schedule candidate cv is 0.10–0.14 % per prompt; the candidate leg is
  tighter than the serial leg.
- A uniform candidate-side speedup maps into the median roughly 1:1.
- The local fixture sits at mean verify width 7.27 against 5.82 on rank. Local
  whole-leg numbers are not arm rankings; per-width and per-cell numbers are.
  Prefill is the exception — one fixed 512-token cell on every prompt and host.
