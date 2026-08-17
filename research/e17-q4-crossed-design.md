# E17 Q4 -- paper-only crossed design: h curve x verify path

Experiment `qwen38-r1-e17-curve-transfer-and-refit` (PR #19), question Q4. Branch
`qwen-edward/curve-transfer-and-refit`, HEAD `991af64`, base `e6e6f81`. This document is
design and prediction only: no build, no benchmark, no kernel edit was performed. The only
execution was the committed pure-Python simulator (`python3 research/e17_gate_sim.py
--trials 60`, exit 0); every number from it is labeled "simulated, not measured".

The proposed factorial: factor A = h curve {A1 shipped, A2 monotone refit}, factor B =
verify path {B1 shipped IPG dispatch, B2 "thorfinn single-pass M=5"}. Cells: C11=A1B1
(shipped binary), C12=A1B2, C21=A2B1, C22=A2B2.

## What is actually shipped (with file:line evidence)

**The h curve.** `headStepCostRatioByDepth = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919,
0.3000, 0.2870, 0.3909]` (`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:575-578`).
The doc comment (:572-574) records the curve re-measuring at **-5.8% to -6.7%**
seconds/token vs the 0.18 scalar on the held-out prose golden, and records direct
forced-depth marginals **0.0971 / 0.1152 / 0.2482 / 0.3761** for h[0..3]: shipped h[0] is
~15% low, h[1] ~33% low and non-monotone (a fitting artifact). The monotone refit level
(A2) is exactly the curve in `research/e17_gate_sim.py:60-75`: measured marginals for
d0-d3, running-max tail beyond, monotone by construction.

**The gate.** `costModelDepth` (:621-655): per depth, `reach *= p`; extend iff `reach >
h[depth]*(1+expected)/(1+cumH)`; top-2-margin clamps at depth 0 (sigmoid margin/2) and
depth 1 (margin/3) (:644-651). Acceptance prior is EMA `0.85*pow(0.98,i)` (:497) with
`acceptEMAAlpha = 0.15` (:498), optimism transfer capped at 0.95 (`recordAcceptOutcome`
:669+). Caps: `sdpaWidthWallDepthCap = 5` (:610), `segmentedVerifyDepthCap = 8` (:617),
`segmentedStreakGate = 3` (:618), `fullAcceptStreak` (:454, reset on any reject
:1126-1127).

**The depth-3 seal (load-bearing).** Advisor commit `6f13c71` ("prove the h-curve caps
depth at 3", generator `research/depth_lever_reconciliation.py`) proved the shipped curve
structurally seals depth at 3 for every acceptance profile: the 3->4 test needs a
probability product > 0.2673/(1-0.8020) = 1.3499 > 1, impossible; verified against 400k
random acceptance vectors. Corollaries quoted there: `sdpaWidthWallDepthCap` never binds,
`segmentedStreakGate` never matters, `segmentedVerifyDepthCap` is unreachable dead code.
Recomputed this session from the sim's gate algebra (`e17_gate_sim.py:100-115`, and
`threshold_table` :250-279, "term for term the shipped code" :258): passing the depth-3
test requires x = p0*p1*p2 with expected >= 3x, i.e. x > h3/((1+h0+h1+h2) - 3*h3).
Shipped: 0.3754/(1.4043-1.1262) = **1.3500** (matches 6f13c71). Refit:
0.3761/(1.4605-1.1283) = **1.1322**. Both > 1: **both A levels are sealed at depth <= 3**
(paper arithmetic, this session).

**The verify path.** One model call per round: `verifyTokens = concatenated([primary] +
draftIdArrays)` (:958-960), `model.callWithHiddenAndNormed(...)` (:979-982), one blocking
eval (:985-995, "THE ROUND'S SINGLE BLOCKING EVAL"). Rows per round = depth+1,
ledger-legal (:613). Widths 6..9 still ride one call; only sdpa is chunked into two
<=5-row calls inside `attentionWithCacheUpdate` (:965-973, :602).

**IPG.** The quantized projections of that verify call route to the crossrow qmv
(`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`):
`static_assert(NA >= 2 && NA <= 4, ...)` (:980; generated twin
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp:993`). "IPG = ceil(M / ceil(M /
4)): the fewest weight streams reachable at NA <= 4" (:1051-1052);
`qmv_fast_crossrow_affine4_g64_m<T,M,IPG>` (:1054) splits M rows into ceil(M/IPG) working
groups, each re-reading the whole weight tile. Dispatch (out_vec_size >= 4096): M=1..4 ->
1 group; **M=5 -> `<T,5,3>`, 2 groups (:1825-1828)**; M=9 -> `<T,9,3>`, 3 groups
(:1845-1848). The measured depth-cost knee tracks this staircase
(`research/results/qwen38-r1-e1-depth-cost-curve.md:883-887`): largest marginals at d=4
(M=5) and d=8 (M=9).

**The only other M=5 channel.** Head-history flush (:868-931): `flushTokens` = backlog +
primary; the head forward runs at M = flushTokens.count, which reaches 5 after a
full-accept depth-3 round followed by a fresh primary. Only the last row projects through
lm_head (:870-871), so flush lm_head stays M=1, but the head-internal quantized
projections see M=5. Drafting itself is sequential width-1 (:940-949). Flush frequency is
statically unresolvable (e13 follow-up #3, see below).

## What 'thorfinn single-pass M=5' is, as far as the record supports

`qwen-thorfinn` is the fellow student behind PRs #3, #5, #8, #10, #15. PR #5's isolated
cost law (quoted in `research/depth_policy_check.py:249-250`): per-step head floor,
verify ~free to M~=3, then linear ramp, plus stream-boundary bumps at M=5 (d=4) and M=9
(d=8).

**PR #8 `qwen-thorfinn/crossrow-na5` (merged `fa9a216`, 2026-08-16) already built and
measured exactly this idea -- and refuted it.** Commits: `704af6f` (v1, pure wide-5
accumulator: "lift the packing limit to NA=5 so the stream boundary moves"), `0a739c9`
(v2, vec4+scalar tail), `84eedac` (restore NA_max=4). W&B: `bq9xfu6d` (NA=4 control),
`e79lcwx2` (v1), `1y91qkq5` (v2) (`research/ESTABLISHED_FACTS.md:1287-1290`). FACT 2
(:1387-1395): both implementations made boundary widths **1.13-1.54x slower**; the
mechanism fired as designed (weight_streams 2->1 at M=5, 3->2 at M=9); break-even needed
~131 GB/s effective bandwidth per stream, best observed 95.5; through the measured law
`C(d) = V(d+1) + 4.46 + 3.73*d`: **C(4) 127.736 -> 173.012 ms (+35.4%)**, C(8) 213.248 ->
249.020 (+16.8%). "Not marginal." FACT 1 (:1370-1385) explains why: at M=5 the two
streams deliver 262.1 GB/s stream-corrected (96% of 273 peak) vs 165.6 (61%) at M=4 --
the "redundant" second stream IS the memory-level parallelism; collapsing to one group
dropped per-stream bandwidth to 85.6-95.5 GB/s "superlinearly and independently of stream
count". Invariant from the dispute banner (:1296-1315): `nominal x M ~= 692 +- 5.6%`,
i.e. seconds/call scales with M; cross-row weight reuse buys ~nothing at M >= 4. FACT 4
(:1410-1418): v2, a semantically identical refactor, broke bit-exactness at exactly M=5
and M=9 (0/8 shapes) -- any such edit carries a mandatory bitwise-vs-M=1 gate.

**e13 = PR #15 `qwen-thorfinn/na4-register-cliff` (merged `ef16dea`, 2026-08-17, static
analysis only)** then found no register cliff at NA=4 (first discontinuity at NA=6, a
spill), concluded the assert is policy and the binding constraint is the M->IPG dispatch
table, and re-proposed in follow-up #1
(`research/results/qwen38-r1-e13-na4-register-cliff.md:229-232`): "Widen the assert to
NA<=5 and change the dispatch table entry M=5 from IPG=3 to IPG=5 (and M=9 from 3 to
5)... Measure whether collapsing 2 working groups to 1 at M=5 is real or absorbed by L2.
Without the table change the assert edit is a no-op." Notably, e13 does not cite PR #8's
refutation anywhere (checked by grep this session), and its static spill-free finding
does not answer FACT 1's memory-parallelism explanation. e13 P4 confirms M = draftCount+1.

**Reading adopted here** (minimal, consistent with all of the above): "thorfinn
single-pass M=5 verify" = the e13 follow-up #1 edit -- NA=5 accumulator plus dispatch
`<T,5,5>` (and `<T,9,5>`) -- so each quantized projection of a width-5 verify makes one
pass over the weight tile instead of two. Rejected readings: (a) "one model call per
round" -- already shipped (:958-995); (b) M=5 = five drafts -- M is rows = drafts+1 (e13
P4, session :613); (c) an sdpa change -- sdpa is already single-call-chunked and is not
the qmv staircase. Honest record ambiguity: the record contains a measured refutation
(PR #8) and a later static re-proposal (e13) that does not engage it; per
`senpai/program.md`, "Try it again only when the recorded reason for reopening has
changed" -- e13's static finding does not change FACT 1's recorded dynamic reason.

## The 2x2 and what each cell measures

| cell | h curve | verify path | binary delta vs C11 |
|------|---------|-------------|---------------------|
| C11 | shipped | shipped IPG | none (baseline) |
| C21 | monotone refit | shipped IPG | eight Double literals (:575-578), same shape as the funded E17 arms (`research/e17-notes.md:19-24`) |
| C12 | shipped | single-pass M=5 | kernel: assert :980 + twin :993, dispatch :1825-1828 (+:1845-1848), rebuild |
| C22 | monotone refit | single-pass M=5 | both of the above |

**The literal 2x2 is structurally degenerate.** Both A levels are sealed at depth <= 3
(1.3500 and 1.1322 above), so the gate-chosen verify width never exceeds M=4 in any cell,
and dispatch cases M=1..4 are byte-identical between B1 and B2. B is therefore inert on
the gated verify path in **all four cells**. What each cell actually measures:

- C21 - C11: the pure curve-shape effect inside the sealed regime -- a d2<->d3 mix shift
  (simulated, not measured: d3 share 41.5% vs 38.8%, mean depth 2.415 vs 2.388,
  rounds/token 0.3625 vs 0.3648, cost/token -0.12%). This is the only funded-adjacent
  contrast, and it is predicted sub-noise.
- C12 - C11 and C22 - C21: B's only live channel, the head-history flush forward at
  M=flushTokens.count=5 (:868-931), plus any build-to-build variance from relinking the
  kernel library. Flush frequency is unknown (e13 follow-up #3: "statically
  unresolvable"); from the simulated depth histograms a full-accept depth-3 round is
  ~39-41% of rounds, an upper bound on 5-row flushes of roughly ~15% of rounds
  (simulated-derived guess, not measured), and a flush is a head-only forward, small
  against a 64-layer target verify.
- C12 and C22 double as **placebo cells**: the seal proof says the gated verify path
  cannot reach M=5, so any resolvable movement there falsifies the seal proof itself or
  exposes an unmodeled M=5 launch site.

## Predicted cell ordering, with the mechanism for each prediction

Predicted seconds/token, lower = better (all gaps predicted below the ~0.3-0.9%
per-prompt serial noise floor recorded in `research/e17-notes.md:113`; scale anchor: the
measured 0.655-0.898% for the 0.20-vs-0.18 scalar flip, `e17_gate_sim.py:10-11`):

**C21 <~ C11 < C22 ~= C12** -- i.e. four statistical ties, with this sign structure:

1. C21 <~ C11 (refit marginally best): the refit curve prices d3 honestly instead of 33%
   low at h[1]/15% low at h[0]; simulated effect -0.12% cost/token via slightly more d3
   full-accepts and fewer rounds (simulated, not measured). Mechanism: threshold table
   slack, not verify cost.
2. C11 < C12 (shipped curve, B2 slightly worse): the only exposed M=5 site is the head
   flush; PR #8 measured single-pass M=5 as 1.13-1.54x slower at boundary widths, so the
   flush channel gets dearer, weighted by ~15%-of-rounds exposure (guess) times a
   head-sized forward => sub-0.1% of wall time.
3. C22 ~= C12, C22 - C21 slightly larger than C12 - C11: refit has more d3 full-accepts
   (41.5% vs 38.8%, simulated) => marginally more 5-row flushes => marginally more B2
   penalty.

No cell is predicted to beat C11 by a resolvable margin, because the one mechanism B2
could help through (cheaper deep verify) was measured to have the opposite sign (FACT 2)
and the seal keeps deep verify unreachable anyway.

## The interaction hypothesis, with its sign

Define the literal-2x2 interaction on time: I = [t(C22)-t(C12)] - [t(C21)-t(C11)] (curve
effect at B2 minus curve effect at B1). Prediction: **I ~= 0, residue positive on time**
(< 0.1%, far sub-noise): refit's extra d3 full-accepts buy slightly more exposure to B2's
dearer M=5 flush. This is a formal sign, not a measurable one.

The interaction worth having requires a depth-permissive A level, i.e. the funded E17
contrast: {CURVE, FLAT18} x {IPG, single-pass}, with g = curve's gain vs flat. FLAT18
opens the gate to depth 4+ (simulated, not measured: mean depth 3.558, max 7, 51.6% of
rounds at d >= 4, i.e. gated verify at M >= 5). Then:

- **Intended mechanism** (e13 follow-up #1: collapse is real, M=5 gets cheaper): the
  FLAT18 x single-pass cell improves, g shrinks => **interaction negative on g**.
- **PR #8's measured reality** (M=5 gets +35.4% dearer at C(4)): that cell worsens, g
  grows => **interaction positive on g**.

Committed prediction, given FACT 2: **positive on g**. The showing cell is FLAT18 x
single-pass -- the only cell in either factorial with majority >= 5-row gated verify
exposure (51.6% of rounds, simulated). The sealed-curve cells cannot show the interaction
regardless of which way B2 goes; that is precisely why the literal 2x2 cannot settle the
question it appears to pose.

## Honest counter-argument: the curve may win only by going shallower

The simulator says the curve's entire advantage over FLAT18 is depth refusal, not smarter
per-depth pricing (all simulated, not measured): shipped runs **+17.6% more rounds/token**
(0.3648 vs 0.3102) yet costs **-9.0%** (0.4773 vs 0.5248 model-cost/token); head
steps/token -21.1% (0.871 vs 1.104); verify rows/token -12.6% (1.236 vs 1.414). The curve
wins by never buying d >= 4, where h[3]=0.3754+ exceeds the marginal accepted-token value.
The measured r3 headline (g = 6.378% single-prompt, `research/e17-notes.md:121`) is the
curve-vs-flat contrast, and this decomposition says most of it is shallowness.

If B2's intended mechanism were real, that win would be partly an artifact of the current
verify-cost surface. Flip-point arithmetic (paper arithmetic from sim outputs): shipped
leads FLAT18 by 0.0475 cost/token; FLAT18 crosses M=5 in 0.516*0.3102 = 0.1601
rounds/token; so h[3] would need to fall by 0.0475/0.1601 = **0.297**, from 0.376 to
<= 0.079 -- below the measured h[0] = 0.0971 -- before FLAT18 catches shipped at fixed
policies (fixed-histogram approximation; re-equilibration of both gates would move both
arms and soften this). That is an implausibly large drop, and PR #8 measured the delta
with the **opposite sign**: C(4) +35.4% => h[3] rises by ~+0.68 in width-1-verify units
(taking C(0) = 67.0 ms, `research/CURRENT_RESEARCH_STATE.md:1109`). So on this hardware
"going shallower" is not an artifact -- it is correct pricing of a real staircase. The
residual honest caveat: within the sealed regime the refit-vs-shipped contrast has nothing
to do with shallowness (both cap at 3) and is predicted unresolvable; the crossed design
adds no power to detect it.

## What would falsify the interaction

- **Movement in a placebo cell**: any resolvable C12-C11 or C22-C21 gap beyond the
  flush-sized budget falsifies the depth-3 seal (6f13c71; recomputed 1.3500 / 1.1322
  above) or reveals an unmodeled M=5 launch site -- either is a concrete, checkable
  discovery, and the first would also invalidate E17's premise that the curve transfers
  as a sealed policy.
- **Additivity-with-effect**: a nonzero B main effect that is identical across A levels
  within paired noise would be mechanically anomalous (dispatch cases 1..4 are
  byte-identical between B levels), indicting build/measurement variance rather than
  physics; the paired base re-run discipline in `senpai/program.md` is the control.
- **Depth-permissive null**: FLAT18 x single-pass moving < paired noise while forced-M=5
  microbench telemetry confirms the collapse fired (weight_streams 2->1, as PR #8 logged)
  would falsify the simulated 51.6% M>=5 exposure share -- i.e. the sim's depth
  histogram, not the kernel story.
- **Sign flip**: g shrinking under B2 (negative interaction) would mean single-pass M=5
  helps -- a direct contradiction of FACT 2's 1.13-1.54x and FACT 1's bandwidth
  mechanism, credible only if the "absorbed by L2" hypothesis (e13 :229-232) succeeds
  where both PR #8 implementations failed, and only after the mandatory bitwise
  M=1..9-vs-M=1 gate passes (FACT 4 broke exactness at exactly M=5/M=9).

## Cost of running it, and why it was NOT run here

Cost to stand up B2 (per `senpai/program.md` Metal source-form rules): edit
`quantized.h:980` assert + dispatch cases :1825-1828/:1845-1848, mirror the generated
twin `quantized.cpp:993` (JIT family => twin edit + `python3 research/twin_audit.py`, not
a metallib rebuild), release rebuild, then the mandatory hexfloat bit-exactness gate
M=1..9 vs M=1 across the 8 shipped shapes (FACT 4 hazard). Timing: 4 cells x 8 prompts of
paired iterate legs ~= 32 legs plus fresh-base pairs, roughly **2x the funded E17
budget** (E17 funds 2 arms x 8 prompts, `research/e17-notes.md:19-24`), each leg behind
the 40C cooling gate and inside the 30-minute wall-clock cap per job.

Why not run: (1) Q4 is explicitly paper-only -- E17's funded slot is the CURVE-vs-FLAT18
contrast, and this design does not outrank it; (2) the literal 2x2 is a predicted
four-way tie (both A levels sealed), so the information value per leg is near zero;
(3) the reopening rule in `senpai/program.md` -- PR #8's measured refutation stands, e13's
static result does not change the recorded dynamic reason (FACT 1), so re-running
single-pass M=5 without new evidence is exactly the duplicated work the rule forbids;
(4) the B2 cells require the riskiest class of edit in the record (bit-exactness broke at
precisely these widths, FACT 4) in exchange for a predicted null.

## What single cheapest measurement would most reduce the uncertainty

**Free, and first**: parse the per-round `effective_draft_lengths` / accepted-length
telemetry out of the already-funded E17 CURVE and FLAT18 run reports to build the in-vivo
verify-width histogram per arm (e13 follow-up #2, :233-237, asked for exactly this). It
costs zero extra legs and settles the one number every prediction above leans on: the
real fraction of rounds at M >= 5 under FLAT18 (simulated 51.6%) and the real d2/d3 mix
under the curve (simulated 61/39). If FLAT18's M>=5 share is materially lower in vivo,
the depth-permissive interaction cell loses its power and the whole B factor is moot; if
it matches, the flip-point arithmetic above becomes a measured statement about h[3].

Next cheapest (one observability-only `--local-iterate` leg, no kernel change, no
exactness risk): log `flushTokens.count` at the flush site (:868-931) -- e13 follow-up
#3's "statically unresolvable" number -- turning the ~15%-of-rounds flush-exposure guess
into a measurement and bounding the only B2 channel that exists in the sealed cells.
