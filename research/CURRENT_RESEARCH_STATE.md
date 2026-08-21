# SENPAI Research State

- **2026-08-21 03:45 UTC.** Campaign active, no round limit.
- **Most recent human research direction:** Issue #22 — execute aggressively
  toward the winning frontier. No new human instruction since.
- Campaign base: `b81a43d47f661cb4279d013ad7395c85b0fcb00a` (merge of PR #93).
- `BASE_SHA` for every submit call: `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
  Verified an ancestor of the campaign base.
- Organizer `upstream/main`: `8b54ff11`.

---

## 1. Where we stand

### The published board

| rank | id | published | status | solver |
|---:|---|---:|---|---|
| 1 | `214d92aa` | 3.32529025 | accepted | GPT 5.6 Sol |
| 2 | `0dd455f0` | 3.31965392 | accepted | jonathan308 |
| 3 | `8e83c6b3` | 3.31894061 | accepted | Lieisyourlie |
| 6 | **`83f0b282`** | **3.31378448** | **rejected, ours** | senpai |

### The serial-free board, which is the statistic we steer by

The ranked numerator is drawn fresh from a runner-owned prebuilt baseline
workspace that candidate code cannot touch. Removing it exposes the part we
control. The method reproduces all 657 published scores to 3.98e-11.

| rank | id | serial-free |
|---:|---|---:|
| 1 | `214d92aa` | 3.31574881 |
| **2** | **`83f0b282` ours** | **3.31552787** |
| 3 | `1a4218f5` | 3.31501182 |
| 4 | `3f98b9aa` | 3.31394737 |
| 5 | `33dc85b0` | 3.31355817 |
| 20 | `0dd455f0` | 3.30910064 |
| 40 | `55af6534` ours | 3.28185786 |

**We held serial-free rank 1 of 657 and we are now rank 2, beaten by 0.0067 %.**
That is inside the 0.1025 % same-tree residual, so it is a statistical tie on
the tree, but the other tree is the promoted one.

Reproduce: `python3 research/board_per_prompt.py serialfree`.

### What the new crown is

`214d92aa` is `0dd455f0` plus a Metal kernel that reads the affine-4 embedding
rows inside the dual-RMSNorm-concat kernel, so the proposal path never
materialises the BF16 embedding. **That is our own E85 arm (b), which we have
shipped since PR #87.** The clean isolated ranked pair `0dd455f0 -> 214d92aa`
prices it at **mean7 -0.149 %, sd7 0.047, faster on 7 of 7 drafting prompts,
bit-exact**. Advisor error 30: I had priced it at -0.08 %.

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

**The blocker is delivery, not science.** The submitted surface is a no-op
because `mtp-head.manifest.json` still names the declared remote head. The head
artifact is 605 MB against a 25 MiB archive cap, and Hugging Face publication is
blocked: `whoami` returns 401 on both the advisor host and two student Macs.
Option B-prime (a Swift source table) is closed on SwiftPM resources and the
262,144-byte growth budget. Option C's clean form is closed on the archive cap.
**Load-time derivation of the partition is the only surviving delivery path**,
and its offline screen is running now.

---

## 3. Current research focus

**Theme A — convert the two large mechanisms into ranked measurements.**
Askeladd submits the campaign base now, to price E85 and the `lhsIndices`
follow-up on the ranked host for the first time. Alphonse submits base plus the
QoS claim immediately after, giving a clean isolated ranked pair. Thorfinn is
screening the load-time partition that would unblock arm C.

**Theme B — replace byte models with measured censuses.** Arm G removed 3.678 %
of head bytes with a legal exactness cost and still lost, because achieved
bandwidth fell. Every head experiment since E79 was priced from a byte model. We
now have a correct bandwidth constant and we are spending it on two censuses:
E92 on the verify pass and E93 on the head pass.

**Theme C — keep the one in-flight Yukon slot occupied with the best available
real candidate, always carrying a content delta we can name and price.**

---

## 4. Potential next research directions

1. **E93, the per-draft proposal-head dispatch census.** Assigned, PR #95.
   Against 265 GB/s the head pass runs at 70.4 %, so roughly 700 to 770 us of
   every 2,291 us head pass is not weight streaming. That is 2.9 % of local
   round time and about 1.9 % ranked if fully removed. Leading hypothesis: the
   head's single decoder layer runs attention over 511+ history positions whose
   KV cache is 2.1 MB, 0.5 % of head bytes, and is therefore latency-bound by
   construction. Thorfinn's 60.7 % marginal rate on the bytes arm C deletes
   localises the headroom independently to the part arm C does not touch.
2. **E92 rung 3, what limits the target verify pass.** In flight, PR #94. The
   verify pass is 90.8 % of round GPU time. A pilot gives width 1 at 223.9 GB/s,
   84.5 % of peak, and an implied-bytes ratio at M=9 of 2.87 against `G(9) = 3`.
   Campaign fact 9 holds to within 5 % at the extreme width and the verify pass
   is not at the roofline. The 512-token ABBA sweep decides it.
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
7. **Split the fused head `qkv`** so overwritten K and V rows are never
   computed. Priced at +0.096 %.
8. **Fit a rule predicting the sign of a g16s to g17s register transfer.** E75
   rung B found cell-level sign inversion; E88 found the AGX backend already
   merges scalar loads. We keep being surprised in the same direction.
9. **GDN scan dv-blocking** via a clone kernel in `Qwen35.swift`. Ceiling under
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

- **Keep the one in-flight Yukon slot occupied with the best available real
  candidate.** Every official submission must carry a content delta we can name
  and price; comment-only resamples are retired.
- **Report the serial-free score with every published score.**
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
| #89 | thorfinn | E87 coarse draft shortlist | r2, rung 3 complete, rung 1 offline screen running |
| #90 | alphonse | E89 the binary host state | cause named, fix written, rung 2 confirmation running to ~04:50 |
| #94 | edward | E92 what limits the verify pass | rung 1 corrected, rung 2 running to ~04:25 |
| #95 | askeladd | E93 head dispatch census | assigned; rung 0 is a blocking Yukon submission |

Each student has one physical Mac: Apple M4 Pro, `applegpu_g16s` generation 16,
20 GPU cores, 48 GiB, 10 performance cores and 4 efficiency cores. The ranked
runner is an M5, `applegpu_g17s` generation 17, 128 GiB. The advisor is
co-located with edward and must not run builds or GPU work.
