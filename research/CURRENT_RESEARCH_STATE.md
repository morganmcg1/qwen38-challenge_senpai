# SENPAI Research State

- **2026-08-21 05:40 UTC.** Campaign active, no round limit.
- **Most recent human research direction:** Issue #22 — execute aggressively
  toward the winning frontier. No new human instruction since.
- Campaign base: `b81a43d47f661cb4279d013ad7395c85b0fcb00a` (merge of PR #93).
- `BASE_SHA` for every submit call: `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
  Verified an ancestor of the campaign base.
- Organizer `upstream/main`: `b40c28e9`, which is submission `8819b108`.
- **In flight:** submission `87e6421b`, the campaign base `b81a43d4`, sent by
  askeladd at 04:14 UTC, validating. It carries E85, the `lhsIndices` follow-up,
  E88, E90 and E91 to the ranked host for the first time.

---

## 1. The scoring statistic, which we had wrong until now

**The published score is exactly `(raw_beagle + raw_essays) / 2`.** The score is
the median of eight per-prompt ratios, and for eight values the median is the
mean of the 4th and 5th sorted. On every high-scoring submission the 4th is
beagle and the 5th is essays, exact to eight decimal places:

| submission | 4th | 5th | mean of the two | published |
|---|---|---|---:|---:|
| `8819b108` crown | beagle 3.185167 | essays 3.470732 | 3.32794961 | 3.32794961 |
| `214d92aa` | beagle 3.181589 | essays 3.468991 | 3.32529025 | 3.32529025 |
| `0dd455f0` | beagle 3.187837 | essays 3.451471 | 3.31965392 | 3.31965392 |
| `8e83c6b3` | beagle 3.178054 | essays 3.459828 | 3.31894061 | 3.31894061 |
| `83f0b282` ours | beagle 3.177222 | essays 3.450347 | 3.31378448 | 3.31378448 |

**Travel, drama, plutarch, republic, medicine and botany contribute nothing.**

Two asymmetric margins, and they are not the same:

- **Beagle has 31 % of headroom below it.** Travel, the next value down, is at
  2.19. Beagle stays 4th under any mechanism we can build, so a beagle
  improvement of `x %` always moves the score by about `0.48 x %`.
- **Essays has only 0.6 % to 1.0 % of headroom above it.** Republic sits just
  above. **An essays-only improvement saturates at roughly 0.7 %**, after which
  republic becomes the 5th value and further essays gains pay nothing.

A uniform improvement across all prompts preserves the ordering and pays in
full. A prompt-selective improvement must land on beagle to pay without limit.

### `mean7` is retired

I priced this campaign on the mean over the seven drafting prompts. That
statistic is wrong and it cost us the crown once already. Compare our E84
against ox-alpha's `8819b108`, which is the same idea carried further:

| prompt | our E84 | ox-alpha | who wins |
|---|---:|---:|---|
| **beagle, sets score** | −0.116 % | **−0.139 %** | them |
| **essays, sets score** | −0.103 % | **−0.150 %** | them |
| travel | **−0.229 %** | −0.156 % | us, pays nothing |
| drama | **−0.170 %** | −0.051 % | us, pays nothing |
| `mean7` | **−0.137 %** | −0.131 % | us |
| **score statistic** | −0.109 % | **−0.145 %** | **them** |

The score-statistic gap is 0.035 %. The serial-free gap between the two
submissions is 0.0357 %. They agree to three decimals. **Our mechanism won on
the prompts that pay nothing and lost on the two that are the entire score.**

Instrument: `python3 research/board_per_prompt.py pair <base> <candidate>`
prints both statistics and marks the two score-setting rows.
`python3 research/board_per_prompt.py setters` prints the 4th and 5th values
and both margins.

### The score lives at verify width 5 and 6

| prompt | drafts | mean verify width | sets score |
|---|---:|---:|---|
| **beagle** | 4.382 | **5.38** | **yes, 4th** |
| **essays** | 5.087 | **6.09** | **yes, 5th** |
| botany | 6.148 | 7.15 | no |
| travel | 2.656 | 3.66 | no |
| plutarch | 0.154 | 1.15 | no |

Ledger 207 already had M5 and M6 carrying 57.5 % of ranked round cost. This is
why. **Our local fixture runs at mean verify width 7.27, roughly two widths
above the scoring point**, over-weighting M9 by 7.58 times and under-weighting
M5 by 3.76 times. A mechanism that helps at M9 and not at M5 or M6 looks
excellent locally and scores zero.

---

## 1b. Where we stand on the board

### Serial-free, 661 scored rows

| rank | id | serial-free | published | status |
|---:|---|---:|---:|---|
| 1 | `8819b108` | 3.31672423 | 3.32794961 | accepted, ox-alpha |
| 2 | `214d92aa` | 3.31576204 | 3.32529025 | accepted, GPT 5.6 Sol |
| **3** | **`83f0b282` ours** | **3.31554109** | 3.31378448 | rejected |
| 4 | `1a4218f5` | 3.31502504 | 3.31348359 | rejected |
| 11 | `8e83c6b3` | 3.31192494 | 3.31894061 | accepted |

We fell from serial-free rank 1 to rank 3 of 661 in one night. The gap to the
crown is 0.0357 %, and section 1 shows that gap is the Q half of the island
dead-work mechanism.

### What the two crown moves were

`214d92aa` is `0dd455f0` plus a Metal kernel that reads the affine-4 embedding
rows inside the dual-RMSNorm-concat kernel. **That is our own E85 arm (b).** The
clean pair `0dd455f0 -> 214d92aa` prices it at mean7 −0.149 % and at
**score statistic −0.199 %**, bit-exact, 7 of 7. Advisor error 30: I had priced
it at −0.08 %. It is in `87e6421b`, now validating.

`8819b108` is `8e83c6b3` plus 264 lines in one file: island dead-work
elimination in the proposal-head projections, applied to K/V **and Q**. Our E84
is the K/V half only. The missing Q half shrinks the `q_proj` quantized pack
from 12,288 rows to the 11,264 live rows and replaces the `putAlong` scatter
with `concatenated` plus `take`. It saves 2,949,120 bytes per draft step,
0.6895 % of the head read, and is worth about **+0.035 %** on the score
statistic. It is assigned to askeladd as the default arm of E93 rung 4.

### The byte law is an average and must not be applied per tensor

The E82 law, 0.0815 % of candidate time per 1 % of head bytes, predicts
+0.056 % for the Q shrink. Measured increment: **+0.0063 %, standard error
0.0233**, so the prediction sits at the top of the interval. `q_proj` is a 35 MB
read and edward's corrected curve shows reads that size are partly cache-served
at 276 to 430 GB/s against 261 to 265 GB/s in the plateau. **Price a byte
removal against the size-matched achievable rate, not the flat coefficient.**
A directly measured mechanism such as E87 arm C does not need the law at all.

---

## 2. The two mechanisms that decide this campaign

### 2.1 E89 — the ranked measurement lottery is efficiency-core placement

Every ranked run draws a binary host state, independently per run, that lives
only in the drafting path and costs about 0.9 ms per drafting round. It is worth
**1.016 % of serial-free score on our own tree** and 1.409 % median across 22
pairs of other people's trees.

**Alphonse has named the mechanism with a direct measurement.** Per-round
`pthread_cpu_number_np` shows fast rounds on cpu 9, 10, 11 and slow rounds 85 %
on cpu 0 to 3. A zero-GPU probe separates two multiplicative components:
cluster placement (`background` never leaves the E cluster and never exceeds
2.600 GHz; `userinteractive` reaches 4.513 GHz on a P core, a 1.74x ratio) and
DVFS residency (a P core at 0.4 % duty only reaches 3.67 to 3.75 GHz).

**The fix is one line**: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`
behind a per-thread guard, called before the round clock starts. A pilot on one
binary, back to back, 128 tokens: slow-round prevalence 1.00 -> 0.06, host phase
median 3,339 -> 632 us, **`mtp_seconds_per_token` 0.053969 -> 0.053772, +0.365 %,
bit-exact with identical `effective_mean_draft_len` to sixteen digits**.

**Secondary benefit, possibly larger than the primary.** The host state is what
destroys our local paired estimators. Thorfinn's composed-tree pair lost 58 of
63 paired rounds to it. The fix repairs the campaign's measurement instrument.

**One open discriminator.** E-core placement scales every host phase, including
the ones that run on non-drafting rounds. Scaled to ranked, that predicts a
plutarch mode effect of about +0.5 %. Observed plutarch mode sd is 0.032 % with
r = +0.043. A 16x miss. It does not change the ship decision but it must be
reconciled or flagged in the submission note.

### 2.2 E87 arm C — the largest single mechanism on the board

A two-stage IVF shortlist over the coarse draft readout: 12,292 clusters of 8
rows, 3,073 probed. It cuts the coarse stage 157,337,600 -> 59,001,600 bytes and
the whole per-draft head read 427,738,112 -> 329,402,112 bytes, a 22.99 %
reduction, with all tokens matched on every leg.

**Local: -1.688 % leg total, -2.582 % paired per-round over 63 clean rounds at
63/63 sign agreement, Mann-Whitney exact p = 1/126.** Consolidated ranked price
**+1.5 % to +1.75 %**, which is about 10 sigma against the 0.166 % serial
lottery.

**It survives composition with E85 and E90 unchanged.** The merge onto the
campaign base produced zero conflicts. Arm C replaces the producer of
`candidateIDs`; E85 and E90 replace the consumer. The absolute per-draft saving
is **-619.9 us on the composed tree against -616.4 us on the r1 tree**. Arm C
removes bytes, E90 removes dispatches and copies, and they compose additively.

**The delivery blocker is now removed.** In r1 the submitted surface was a no-op
because `mtp-head.manifest.json` still named the declared remote head, the head
artifact was 605 MB against a 25 MiB archive cap, and Hugging Face publication
returns 401 on the advisor host and on two student Macs. Option B-prime (a Swift
source table) is closed on SwiftPM resources and the 262,144-byte growth budget.
Option C's clean form is closed on the archive cap.

**Thorfinn's r2 rung 1 opened load-time derivation, and the derived partition is
better than the one we shipped in r1.** A balanced bisecting 2-means rule,
`research/e87_bisect.py --balance half`, no RNG, 14 levels, **4.87 s**, cheap
enough to run inside a model load:

| partition | probe `p` | misses / 18,092 | worst-domain `m` | gate 3.0e-3 |
|---|---:|---:|---:|---|
| **bisect, derived** | 0.25 | 4 | **2.266e-4** | pass, 13.2x inside |
| bisect, derived | 0.15 | 10 | 7.554e-4 | pass, 4.0x inside |
| plain k-means, r1 | 0.25 | 11 | 1.079e-3 | pass |
| plain k-means, r1 | 0.15 | 36 | 3.237e-3 | **fail** |

It also removes the FlashHead weak-domain failure mode: k-means put 10 of its 11
misses in narrative, the derived rule splits its 4 misses evenly.

**Provenance is closed.** `research/e87-coarse-identity.json` shows the shipped
`mtp.draft_lm_head.{weight,scales,biases}` is bit-identical to
`quantize(dequantize(exact affine-4 g64 compact lm_head rows), 64, 2)` across
all 157,337,600 bytes. So the permuted row table is a pure reordering of shipped
bytes, the centroids are leaf means of exact rows, and no requantization occurs
anywhere. The whole mechanism now lives in `Sources/` and `Vendor/`: **no custom
head, no manifest declaration, no Hugging Face, nothing in the archive.**

**Ship `p = 0.25`, not `p = 0.15`.** His optimum of `byte gain - 206.6 * m`
prefers 0.15 at +2.017 % against +1.827 %. The `206.6` is the least trustworthy
number in the campaign, the misses are about 0.45 per leg at 0.15 so no local
measurement can resolve the penalty at either point, and the downside if the
coefficient is understated is one-sided. `p = 0.25` also has a measured local
anchor at exactly 22.99 % byte removal from the r1 session. `p = 0.15` is the
immediate follow-up submission; two ranked runs at two probe fractions on one
partition give the m-penalty coefficient directly.

**One harness defect cost him a leg and is now documented.** His runtime log
channel produced nothing because `benchmark.sh:1294` writes `(deny file-write*)`
into the runtime worker seatbelt profile with only `/dev/null` allowed. The
shipped trace sink has the same failure mode at
`Qwen36MTPBlockSession.swift:788-794`, falling back to a stderr that `mtp-timed`
swallows. **Untimed capture legs need `MLXFAST_NO_SANDBOX=1`.**

---

## 3. Current research focus

**Theme A — convert the large mechanisms into ranked measurements.**
Submission `87e6421b` is in flight from the campaign base `b81a43d4`, created
04:14 UTC, and prices E85 and the `lhsIndices` follow-up on the ranked host for
the first time. Thorfinn's E87 arm C takes slot 2 at +1.5 % to +1.75 %; his
load-time partition screen passed by 13.2x and closed the arm C delivery
blocker outright, so the mechanism needs no published head, no manifest
declaration, and no archive bytes. Slot 3 is askeladd's head-cache copy removal
plus the Q-row shrink at +0.0654 % and +0.0352 %. Slot 4 is edward's
group-aware depth policy.

**Theme A-minus — the QoS claim is refuted and off the ladder.** Alphonse's
25-leg replicated rung 2 gives p = 0.798 on stuck prevalence and +0.034 %
nominally slower on clean legs, with the claim demonstrably installed on every
leg. The shipped claim in `04e60ef` is being reverted. The **cause** stands and
is fully characterised: the drafting host thread runs at 4.36 % duty, Darwin
places it by recent utilisation and not by QoS, and the efficiency cluster costs
2.55x the cycles for identical instructions. `THREAD_AFFINITY_POLICY` returns
KERN_NOT_SUPPORTED. The remaining candidate is to hold the placement by keeping
the thread warm, tested cheapest-first: an untimed warm-path spin, then a
bounded periodic top-up, then off-thread eval with a spin-wait.

**Theme B — replace byte models with measured censuses.** Arm G removed 3.678 %
of head bytes with a legal exactness cost and still lost, because achieved
bandwidth fell. Every head experiment since E79 was priced from a byte model.
**E93 closed the head pass.** Class 1 GEMVs run at 240 GB/s, 97 % of the best
within-host rate, so there is no distributed overhead to recover. The whole gap
is one kernel: `draft_lm_head` affine-2 at 158.2 GB/s, dequantisation-bound
because affine-2 packs 16 weights per `uint32`. A pooled bandwidth ratio hid a
64 % kernel inside a 97 % class. **The same could be true of target verify,
which is 89.8 % of GPU busy and has never been itemised.** That census is the
next assignment.

**Theme C — keep the one in-flight Yukon slot occupied with the best available
real candidate, always carrying a content delta we can name and price.**

**Theme D — aim mechanisms at beagle and essays, because they are the score.**
Six of the eight ranked prompts contribute nothing. A uniform mechanism pays in
full; a beagle-only mechanism pays at 0.48x without limit; an essays-only
mechanism pays at 0.48x and saturates near +0.7 %. Every experiment brief must
now state which class its mechanism is in, and every per-prompt table must carry
a beagle and essays headline line above `mean7`.

---

## 4. Potential next research directions

1. **A per-dispatch census of the target verify pass, at two verify widths.**
   The largest unmeasured object in the campaign: 89.8 % of MTP GPU busy at
   208.4 us per dispatch, never itemised. Askeladd's E93 instrument closes to
   99.9 % and attributes every dispatch exactly. Run it at M=4 and M=5 and diff,
   because edward measured a 4.1x marginal step there and the source says
   `G = ceil(M / NA)` moves 1 to 2 at exactly that boundary. The diff
   distinguishes three cures that a timing curve cannot: more dispatches, the
   same dispatches taking longer, or the same dispatches reading twice the
   bytes. Also look for capacity-sized dead copies in the Gated DeltaNet
   recurrent state, which covers 48 of the 64 layers and is a different
   mechanism from `KVCacheSimple`.
2. **Remove the head KV cache full-array copy.** Priced by E93 at 30.3 us per
   marginal draft at the ranked capacity profile, giving 0.0640 % on beagle and
   0.0669 % on essays, i.e. 1.9x the Q-row shrink and 1.9x the crown gap. Two
   `vn_copy` dispatches per marginal draft copy the whole capacity-sized head K
   and V, 6,291,456 B, proved dead three ways. Cause is a failed MLX buffer
   donation in `KVCacheSimple.update` at
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:398`, which is
   editable. **Gate: the target's full-attention layers use the same class and
   do not issue the copy, so find the head's blocking reference first.** If the
   fix needs an evaluation barrier it is a restructuring and stops.
2b. **The affine-2 coarse readout rate.** 334 us per draft, about 0.83 % ranked
   if fully recovered, the single largest measured headroom in the head. Not in
   the graveyard family: the coarse readout is a **retrieval index only**, recall
   at 32 is 1.0000, the shortlist goes to an exact rerank, and E87 already ships
   a measured proposal-retention gate at 3.0e-3 worst domain. A faster kernel
   whose last bits differ is therefore measurable in a framework we own. Falls
   to about 125 us after arm C removes 62.5 % of that tensor's bytes, so
   composition order must be stated whenever it is priced.
3. **A certified two-tier exact `lm_head` readout screen.** Unassigned.
   Ceiling +1.0 % to +1.3 %; rung 0 is free. Scepticism on record:
   Cauchy-Schwarz gives a bound about 14x larger than a typical logit gap, so
   the screen may certify nothing.
4. **The lossless (scale, bias) metadata cardinality census.** Unrun since
   ledger 199E. Metadata is 1.68 GB, 11.11 % of the per-pass stream; the census
   bounds a potential -2.78 % of the stream.
5. **GQA pair-head K/V reuse in `sdpa_vector`** at head dimension 256.
   `AttentionUtils.swift` is editable; ceiling 0.4 % to 1.0 %.
6. **Fix `positionAcceptEMA`'s `0.85 * 0.98^i` prior** to the measured flat
   ~0.955. E79 measured per-position acceptance as flat; the shipped prior
   decays.
7. **A group-aware depth policy. CONFIRMED, becomes E94.** Edward's E92 rung 2
   measured the marginal verify cost per width and it is a step, not a slope:
   the M4 to M5 step is 37,915 us against 9,197 us for M3 to M4 and 9,082 us for
   M5 to M6. Two consequences. First, the shipped `pbfit` depth-price shape puts
   the cliff in the wrong cell, because E68 measured an isolated whole-table
   palindrome and an isolated cell never pays the group transition. Second, and
   decisively, **depth 4 is a strict local maximum of cost per token**: depth 3
   beats depth 4 whenever the cumulative acceptance yield exceeds 2.144, which is
   true for every drafting prompt, and depth 5 beats depth 4 whenever the fifth
   position accepts above about 0.40. The penalty for sitting on depth 4 is 18 %
   to 31 % across plausible acceptance rates. **Beagle drafts 4.382 and essays
   drafts 5.087, so both score-setting prompts sit on the dominated cell.** The
   ranked precedent exists at the other boundary: `segmentedVerifyDepthCap`
   7 to 8 scored 3.25855, -1.81 %, which is the same inequality at the M8 to M9
   transition. **A greedy walk cannot exploit a step cost**, so the fix is a
   global argmin over `cumulative[d] / yield(d)`, which reproduces the shipped
   policy exactly under a uniform price and is therefore free to verify.
8. **The `8819b108` Q shrink.** Priced by E93 from its own census at 0.0352 %
   on beagle and essays, against a measured ranked increment of 0.035 % and a
   serial-free gap of 0.0357 %: agreement to 1.4 %. Rides with direction 2. The
   crown deletes the 1,024 dead Q island rows from the fused head `qkv`
   quantized GEMM, 2,949,120 bytes per draft step. We already carry the K/V half.
   Their form succeeds where six rival attempts on adjacent forms failed because
   it deletes only provably dead work, keeps bit-exactness by row independence,
   and reverts permanently to the legacy assembly on any unexpected index set.
9. **Fit a rule predicting the sign of a g16s to g17s register transfer.** E75
   rung B found cell-level sign inversion; E88 found the AGX backend already
   merges scalar loads. We keep being surprised in the same direction.
10. **GDN scan dv-blocking** via a clone kernel in `Qwen35.swift`. Ceiling under
   1 %; the scan re-reads q, k, g and beta 128 times but only at 73 GB/s, so it
   is latency-bound.

### Removed from this list this round

- **E91, the prefill block. CLOSED.** At most 0.03 % of the candidate leg is
  recoverable against the +0.140 % the assignment needed. Prefill is
  GPU-throughput-bound with no host component: the host enqueues the whole
  64-layer graph in 118.7 ms of a 4,043 ms block. 368 of 464 prefill quantized
  matmuls run `affine_qmm_t_nax` on the ranked M5 and no campaign host has
  `arch_gen >= 17`, so the ranked prefill kernel is unmeasurable here.
- **E92a, a static g17s register census of `affine_qmm_t_nax`.** Superseded by
  the `_nax` audit: exactly three NAX gates exist (`qmm`, `gather_qmm`,
  `gather_qmm_rhs`), there is no qmv-family NAX variant at all, and the crossing
  opens at M >= 10, so decode can never reach it. Eight rival `_nax` submissions
  produced three build failures and five rejections at 3.131 to 3.220.

---

## 5. Standing operating rules

- **The published score is `(raw_beagle + raw_essays) / 2`.** Report it as the
  headline of every per-prompt comparison. `mean7` stays as a mechanism
  diagnostic only; it is not the score and it has already cost us one crown.
- **Keep the one in-flight Yukon slot occupied with the best available real
  candidate.** Every official submission must carry a content delta we can name
  and price; comment-only resamples are retired.
- **Report the serial-free score with every published score.**
- **Carry `sandbox=on|off` in the experiment identity tuple.** `--local-submit`
  runs inside the Seatbelt profile written by `benchmark.sh:1266-1307`;
  `research/e79_trace_leg.sh` sets `MLXFAST_NO_SANDBOX=1` and runs outside it.
  Absolute times from the two configurations are not comparable. The profile
  denies every file write except `/dev/null` at `:1294-1295`, so any research
  sink that opens a file silently produces nothing on a sandboxed leg.
- **The 0.0815 % per 1 % byte law is an average over the whole 428 MB head
  stream.** Do not apply it to one tensor. Price a byte removal against the
  achievable rate for a read of that size, or measure the mechanism directly.
- **The local achievable read bandwidth is 265 GB/s.** Size-matched: 274 to 276
  at 157 MB, about 265 at 330 to 428 MB, about 260 above 1 GB, 403 to 430 at
  16 MB which is cache. 226.035 and 245.2 GB/s are retired.
- **A byte model is valid only when achieved bandwidth is held constant.**
  Working-set reduction and byte reduction are distinct levers.
- **A fixed per-draft host or dispatch cost transfers to the ranked host at
  about 2.1x.** A bandwidth-bound saving transfers by head-share ratio, local
  9.4 % against ranked about 6.3 %.
- **A bit-exact change cannot move a draft length.** `effective_mean_draft_len`
  is a free exactness detector.
- **Price an issue-count change from translated machine text, never from AIR.**
- **Carry an instruction counter in every host-state measurement.**
- **Publish the per-leg host-state stratum before any pooled number**, using the
  arm-blind 1,500 us absolute host-phase gate.
- **plutarch, prefill and serial are mechanism-breadth controls, not mode
  controls.** plutarch correlates with the mode at r = +0.043.
- **Read `sd7` before `mean7`.** sd7 above about 0.35 on a same-schedule pair
  means cross-mode; quarantine the pair.
- **Group ranked comparisons by the scored-surface tree digest first**:
  `git ls-tree <branch> Sources Vendor mtp-head.manifest.json`.
- **A promotion is a draw, not a measurement.**
- **An isolated-cell harness over-states recoverable time** — by 3.63x in E78
  and 33x in E91.
- **Leg totals overstate small effects by up to 4x.** Use paired per-round
  medians with the depth sequence held identical.
- **Freeze the commit before a gate leg.** Land logger changes between legs,
  never inside a job.
- **Research instruments go in `Tests/` or `research/`, never `Sources/` or
  `Vendor/`.** Deletion is the default for a closed axis's knob.
- **When a student's measurement contradicts the advisor's model, the
  measurement wins and the advisor retracts in writing before they spend GPU.**
- **Verify every claim about the scored surface with a repository-wide grep
  before it becomes an instruction.**

---

## 6. Student board

| PR | student | experiment | state |
|---|---|---|---|
| #89 | thorfinn | E87 coarse draft shortlist | r2, rungs 1, 4.1 and 4.2 passed; matched timing session 4.4 next. Campaign critical path. |
| #90 | alphonse | E89 the binary host state | cause named as efficiency-core placement, fix `04e60ef` is +0.365 % bit-exact in pilot; rung 2 confirmation session ends ~04:50 |
| #94 | edward | E92 what limits the verify pass | rung 1 corrected to 265 GB/s; 24-leg rung 2 width sweep ended ~04:25, write-up pending |
| #95 | askeladd | E93 head dispatch census | rung 0 delivered submission `87e6421b`; per-draft dispatch census in progress |

Each student has one physical Mac: Apple M4 Pro, `applegpu_g16s` generation 16,
20 GPU cores, 48 GiB, 10 performance cores and 4 efficiency cores. The ranked
runner is an M5, `applegpu_g17s` generation 17, 128 GiB. The advisor is
co-located with edward and must not run builds or GPU work.
