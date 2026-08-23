# E155 result: the draft head's index is exact enough; the head itself holds 97.7 % of the residual

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e155_recoverable_index_pp","available":true,"value":0.07553},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- **Student / branch:** `qwen-askeladd` /
  `qwen-askeladd/e155-draft-head-recall-audit`
- **Hypothesis and target cost:** the shipped proposal path never computes its
  own argmax. It stacks three approximations — a vocabulary trim to 98,330
  reachable ids, a `p15` ANN shortlist over a leaf-16 index, and a 32-candidate
  4-bit rerank. R1 asks how many acceptance points an *exact* readout would
  return, split into what the index gives away
  (`p_exact_compact - p_shipped`), what the trim gives away
  (`p_exact_full - p_exact_compact`), and what neither can reach
  (`1 - p_exact_full`). No timed work; this is an exactness and counting
  experiment.
- **Decision:** **close the index axis, and close the vocabulary-trim axis.**
  `e155_axis_verdict = close_the_index_axis`,
  `e155_vocab_axis_verdict = close_the_index_axis`. R0 is
  `closed_by_inheritance` under the advisor's FINDING 283. No candidate change
  is proposed by this branch; the submitted surface is byte-identical to the
  PR base.
- **`BASE_SHA` / `UPSTREAM_SHA` / candidate commit:**
  `9d3e3d2db416b8360135dce476d929c5d6b1b40f` (PR base, advisor branch
  `d0422d1d`) / `0863b06ac16e26e48fc06e97444095b00feb66d4` / no candidate
  commit. The audit build was `3d52eac6`; the instrument is reverted at
  `47c7c3ec` and the head of this branch is a research-only tree.
  I re-ran `git fetch upstream` per the advisor's correction 1.2: my local
  `upstream/main` had been stale at `b40c28e9` and now reads the crown
  `0863b06a`. No number in this report was read from the stale ref — the audit
  measures our own tree against its own exact head readout.
- **Yukon promoted submission / source ref used as frontier:** the bar is
  `ec24d59` at `3.72911001`, source `0863b06a`, as recorded by the advisor at
  16:25Z. I did not query Yukon and I did not call a submission.
- **Candidate build fingerprint:** audit-ON worker
  `b3f5d6cfdce13f645e6ea2b32eae4b637fe0f84d52e68a50e4b4ff49e6d91c25`,
  audit-OFF and gate-probe worker
  `1cf1221b06208197536dd1c2b4e351b513091ddea48ffc46b6c8f75d6a576967`. Both are
  named in `research/e155-artifacts/witness.json`.
- **Submitted-surface / generated-twin / metallib digests:** no Metal source
  and no submitted Swift source differ from the PR base at the head of this
  branch. `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` is 126,843 bytes
  at the PR base, 127,761 bytes at the instrumented commit `f65b6c1e`, and
  126,843 bytes again at the head.
- **Submitted candidate files:** none. `e155_growth_reclaimed_bytes = 918`.
- **Supporting test, tooling, or documentation files:**
  `research/e155_audit_session.sh`, `research/e155_gate_probe.sh`,
  `research/e155_recall_analysis.py`, `research/e155_width_cost_fit.py`,
  `research/e155_witness.py`, `research/e155_wandb_log.py`,
  `research/e155-artifacts/*.json`, `research/e155-patches/recall-audit.patch`,
  and this document. None is a submitted path.
- **MTP head provenance, digest, and draft policy:**
  `head_provenance_sha256 = dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`
  on every leg, audit-ON, audit-OFF and gate probe, origin
  `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae628274`, 427,746,170 bytes,
  15 tensors. Organizer-pinned head; this branch declares no candidate head.
  Draft policy is the shipped adaptive schedule at offered depth 8.
- **Token window, fixture, reference source, and harness:** 512 decode tokens
  behind a 512-token seed, public fixtures `beagle_a`, `essays_montaigne` and
  `benchfixture`, checked-in E128 goldens
  (`beagle_a` `8ed47038…`, `essays_montaigne` `dbc8b98f…`,
  `benchfixture` `66858d95…`). **`harness=local` on every measured rate;
  `harness=ranked` on every conversion to published percent.**
- **Exact cell:** the derived cluster ANN index over the `[98336, 640]`
  affine 2-bit-centroid / 4-bit-row compact draft head, read once per proposal
  slot. `derivedClusterRowsPerLeaf = 16` (mine, shipped in E153),
  `ProbeArm.compiledDefault = .p15`, `draftRerankCandidateCount = 32`,
  `compactDraftPrefixCount = 98,304` plus the 26 control ids
  `[248044, 248070)`.
- **Official causal path and score equation:** untouched. This branch changes
  no scored code; `senpai/verify-ranked-score-boundary.sh` passes.
- **Assignment-scope preflight:** `assignment scope OK: 1 submitted path(s)
  against BASE_SHA=9d3e3d2d…`, run on the final head.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  `source=2652191/3000000 headroom=347809 growth=197356/262144 exempt=2410
  files=154` for `growth_enforced` against `770a3ff2`, and
  `growth=0/262144` for `growth_attributable` against the PR base. I consumed
  none of the ~64,800 bytes the advisor flagged for the team.
- **Scored-path reachability evidence:** the audit is gated by
  `MLX_E155_RECALL_AUDIT`, absent from the shipped default, and the gate is
  proven off on four independent sides (section 5 below). At the head the hook
  itself no longer exists on the submitted surface at all.
- **Written promotion rule and verdict:** the advisor's pre-registered table —
  `>= 1.0 pp` means the index is the campaign's largest lever, `0.3` to
  `1.0 pp` means one bounded R2 arm, `< 0.3 pp` means close the axis.
  Measured `0.0755 pp` conditional, `0.0665 pp` marginal. **Verdict: close the
  axis.** The same table applied to the vocabulary term at `0.1511 pp`
  conditional also says close, and the trim is net negative once priced.
- **Pre-official evidence budget / timed legs used:** zero timed legs, zero
  gated legs, no thermal gate, no ABBA, no palindrome. Seven untimed 512-token
  decodes (3 audit-ON, 3 audit-OFF, 1 gate probe).
- **Frozen candidate SHA, if promoted for submission:** none.
- **Specific evidence that invalidated the frozen SHA, if any:** not
  applicable.
- **Submission owner / read-only receipt-watcher job ID:** not applicable. I
  did not call Yukon; edward holds the slot with `75a21a4` and thorfinn is
  queued behind it.

## Evidence

- **Host, instance, chip, memory profile, toolchain, thermal policy:** Apple
  M4 Pro, 20-core `applegpu_g16s`, 48 GiB, `ip-10-231-2-227.ec2.internal`.
  No cool gate was taken and none is claimed: every artifact records
  `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
  `timing_valid=false`, `official_or_ranked_score=false`.
- **Exact commands:**
  - `research/e155_audit_session.sh beagle_a essays_montaigne benchfixture`
    (audit ON, worker `b3f5d6cf…`)
  - `research/e155_gate_probe.sh beagle_a` (path variable set, gate variable
    unset)
  - `research/e155_recall_analysis.py`, `research/e155_width_cost_fit.py`,
    `research/e155_witness.py`, `research/e155_wandb_log.py`
- **Cheapest real falsification gate and positive-control verdict:** the
  permutation control. Pairing each slot's shipped id with the *next* slot's
  target collapses `p_shipped` from `0.872921` to `0.015979`, a magnitude of
  **85.694 pp**, so the comparison is capable of failing.
  `e155_permutation_positive_control_passed = true`.
- **Tests and risk-based checks, in execution order:** golden-row exactness on
  every leg; audit row-count identity against `sum(effective_draft_lengths)`;
  off-versus-on decode identity; off-versus-E153 decode identity; gate probe;
  permutation control; masked-winner control; `ann_outside_compact_set`
  census; then the three gate scripts on the final head.
- **Exact-token and row-ledger verdict:** `all_tokens_matched = true` and
  `residual_divergence_count = 0` on all six audit legs, ON and OFF. The
  round ledger is identical off-versus-on on every prompt.
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not relevant; no Metal source changed.
- **Peak RAM or head/artifact size:** the audit adds one dense
  `[248320, 640]` lm_head projection per proposal slot inside the measured
  block. That is why the ON legs are declared non-timing by construction.

### 1. Coverage

| prompt | slots | rounds | mean offered depth | distinct target ids | top-token share |
|---|---:|---:|---:|---:|---:|
| beagle_a | 505 | 119 | 4.2437 | 95 | 0.1525 |
| essays_montaigne | 502 | 146 | 3.4384 | 125 | 0.2749 |
| benchfixture | 496 | 78 | 6.3590 | 220 | 0.0746 |
| **pooled** | **1503** | **343** | **4.3819** | **408** | **0.1577** |

`benchfixture` is a third prompt beyond the pair the brief asked for. It
decodes at much higher acceptance and much lower token concentration, which
tests whether greedy degeneration drives the answer. It does not: the most
diverse prompt is also the highest-recall one, and it carries the only
recoverable selection slot in the entire audit.

### 2. The three rates, counts and rates, `harness=local`

Marginal population — every audited slot.

| prompt | shipped | `p_shipped` | exact-compact | `p_exact_compact` | exact-full | `p_exact_full` |
|---|---:|---:|---:|---:|---:|---:|
| beagle_a | 429/505 | 0.849505 | 429/505 | 0.849505 | 432/505 | 0.855446 |
| essays_montaigne | 418/502 | 0.832669 | 418/502 | 0.832669 | 418/502 | 0.832669 |
| benchfixture | 465/496 | 0.937500 | 466/496 | 0.939516 | 466/496 | 0.939516 |
| **pooled** | **1312/1503** | **0.872921** | **1313/1503** | **0.873586** | **1316/1503** | **0.875582** |

Conditional population — the 1,324 slots the shipped chain actually reached,
which is the population that feeds `E[accepted tokens per round]` and therefore
the population the exchange rate is defined on:
`p_shipped = 1193/1324 = 0.901057`,
`p_exact_compact = 1194/1324 = 0.901813`,
`p_exact_full = 1196/1324 = 0.903323`.

**`e155_index_miss_rate = 4/1503 = 0.002661.`** None of the four is an exact
score tie, so all four are real score gaps. Those four misses produce exactly
**one** changed acceptance outcome: the miss rate is 4x the acceptance loss and
the other three swapped one wrong token for another. The advisor predicted the
miss rate would exceed the loss by several times; it does.

### 3. The three buckets, conditional population

| bucket | slots | acceptance points | published at `+2.6701 %/pt` | absolute | x gap |
|---|---:|---:|---:|---:|---:|
| A — selection or rerank drops a token the head already ranks first | 1 | **0.0755** | +0.2017 % | +0.00742 | 0.160 |
| B — target outside the 98,330 reachable ids | 2 | **0.1511** | +0.4033 % | +0.01485 | 0.321 |
| C — the head itself disagrees with the target | 128 | **9.6677** | — | — | — |
| shipped hits | 1193 | | | | |

`partition_is_exhaustive = true` and `lucky_shipped_win_exact_miss = 0`: there
is no slot where the approximate index beats the exact readout, so bucket A is
not netted down by luck. Marginal cross-check: A `0.0665`, B `0.1996`,
C `12.4418`.

**`e155_recoverable_acceptance_points = 0.2266` total, of which only bucket A
at `0.0755` is free.** `0.2266` of the `9.8942` conditional points the head
gives away are a readout problem — **2.3 %. The other 97.7 % is the head
disagreeing with the 27B target.**

### 4. Bucket B is priced and it loses

`harness=ranked`, 567 GB/s roofline, geometry read from `Qwen35.swift` on this
base. Compact head 2-bit affine group-64 with fp16 scale and bias
(0.3125 B/param); rerank reads 32 rows of the exact 4-bit head (0.5625 B/param).

| | leaves | probed | refined rows | coarse | refine | rerank | per draft step |
|---|---:|---:|---:|---:|---:|---:|---:|
| shipped, 98,336 rows | 6,146 | 922 | 14,752 | 9.83 MB | 23.60 MB | 0.09 MB | **33.53 MB** |
| untrimmed, 248,320 rows | 15,520 | 2,328 | 37,248 | 24.83 MB | 59.60 MB | 0.09 MB | **84.52 MB** |

Delta 50.99 MB per draft step = 89.93 µs at roofline = 394.1 µs per round at
mean offered depth 4.382. Against the modelled ranked round of 43,114 µs that
is **+0.914 % slower** against a **+0.403 %** gain: **net −0.511 %**, before
the offline transform change and larger head tensors it would also need.
`worth_building = false`.

The optimistic variant that holds the probe count at 922 pays only the coarse
pass, but it drops leaf coverage from 15.0 % to 5.9 %, and the advisor's own
`p15`-to-`p12` evidence prices a 3 pp coverage cut at about 0.9 acceptance
points — six times the entire bucket-B prize.

### 5. Two-sided gate witness

`e155_exact_path_is_gated_off_by_default = true`, on four sides, all from one
worker binary per side:

1. **Off legs.** Three 512-token legs with both variables unset; no audit file
   appeared.
2. **Gate probe.** One 512-token leg with `MLX_E155_RECALL_AUDIT_PATH` set and
   `MLX_E155_RECALL_AUDIT` unset: `audit_file_created=false`,
   `audit_rows_written=0`, exit 0. This names the gate variable as the cause
   rather than the pair. Decode identical to the off leg — 119 rounds, draft
   `4.2436974789915967`, accept `0.77821782178217824`.
3. **On legs.** Same instrument, gate set: 505, 502 and 496 rows, each exactly
   `sum(effective_draft_lengths)` for that leg. Without this side the off side
   would be vacuous.
4. **Decode invariance.** Off decode identical to on decode on every leg, and
   identical to the pre-instrument E153 legs on the two prompts E153 covered.

The sink creates its file at install time, before the first round, so file
absence is a direct observable and does not depend on worker stderr, which the
trusted harness drops.

### 6. Positive controls, Rule 101

- **Permutation control:** `p_shipped` 0.872921 → 0.015979, magnitude
  **85.694 pp**. Passed.
- **Masked-winner control:** masking the exact compact winner and re-reading
  moves the answer in **1503 of 1503** slots and drops the hit rate from
  0.873586 to 0.035263, magnitude **83.832 pp**. This is the row-perturbation
  control on the compact-to-vocabulary permutation.
- **`ann_outside_compact_set = 0`:** every shipped proposal in 1,503 slots
  lands inside `[0, 98304) ∪ [248044, 248070)`, so the compact-index to
  vocabulary-id map is proven on live data, not only by construction.

### 7. Why bucket C is not a near miss

Head margin = `full_max - target_logit`, both from the same head logit vector,
over the 187 marginal slots where the head disagrees.

| statistic | value |
|---|---:|
| exact logit ties | 3 |
| p25 | 1.0000 |
| median | 2.1875 |
| p75 | 5.1875 |
| max | 17.4375 |
| mean | 3.5598 |
| within 0.25 | 0.0695 |
| within 0.50 | 0.1497 |
| within 1.00 | 0.2781 |
| within 2.00 | 0.4866 |

Only 3 of 187 disagreements are exact ties. A tie-break policy change is a coin
flip against those 3 unless the new order correlates with the target ordering,
so its expected value is at most half of 0.20 pp marginal and could be zero.
The median disagreement is 2.19 logits, which no readout, calibration or
precision change reaches.

### 8. Per-slot table, pooled

Conditional — the chain population.

| slot | trials | `p_shipped` | `p_exact_compact` | `p_exact_full` | index pp | vocab pp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 343 | 0.8659 | 0.8659 | 0.8717 | 0.0000 | 0.5831 |
| 2 | 294 | 0.8980 | 0.8980 | 0.8980 | 0.0000 | 0.0000 |
| 3 | 224 | 0.8973 | 0.9018 | 0.9018 | 0.4464 | 0.0000 |
| 4 | 169 | 0.9172 | 0.9172 | 0.9172 | 0.0000 | 0.0000 |
| 5 | 120 | 0.8833 | 0.8833 | 0.8833 | 0.0000 | 0.0000 |
| 6 | 91 | 0.9780 | 0.9780 | 0.9780 | 0.0000 | 0.0000 |
| 7 | 83 | 0.9759 | 0.9759 | 0.9759 | 0.0000 | 0.0000 |

Marginal — every slot.

| slot | trials | `p_shipped` | `p_exact_compact` | `p_exact_full` |
|---:|---:|---:|---:|---:|
| 1 | 343 | 0.8659 | 0.8659 | 0.8717 |
| 2 | 337 | 0.8516 | 0.8516 | 0.8516 |
| 3 | 270 | 0.8519 | 0.8556 | 0.8593 |
| 4 | 213 | 0.8779 | 0.8779 | 0.8779 |
| 5 | 145 | 0.8621 | 0.8621 | 0.8621 |
| 6 | 102 | 0.9608 | 0.9608 | 0.9608 |
| 7 | 93 | 0.9462 | 0.9462 | 0.9462 |

Nothing anomalous appears deep in the rollout, so the `qwen35ClusterRowQMV`
bypass under `leaf16` shows no slot-dependent signature. The conditional rate
rises with slot because of survivorship and adaptive depth, not because deep
slots are easier.

### 9. My own exchange rate

`e155_published_pct_per_acceptance_point = 2.4421 %` published per acceptance
point, derived here at **fixed depth** from the pooled measured depth mix and
the per-slot conditional profile, by bumping every slot probability by one
point and taking the ratio of `E[tokens per round]`.

- It reproduces the advisor's F1 fixed-depth `+2.240 %` to within 9.0 %.
- It sits 8.5 % below F3's settled `+2.6701 %`, in the expected direction: mine
  is at my local depth mix of 4.38 and local acceptance of 0.90, and the
  Rule 148 weighted ranked prompts sit at higher `t`.
- **Every conversion above uses F3's `+2.6701 %`, not mine.** Using mine moves
  bucket A to `+0.184 %` and bucket B to `+0.369 %`, which changes no verdict.
- A no-linearization cross-check that rebuilds the accept chain directly from
  the exact per-slot profile gives index `+0.180 %` and vocabulary `+0.511 %`
  published, both consistent.

### 10. Bonus — `e155_within_prompt_width_cost_fit`

`round_us = a_p + b_p * M`, `M = d + 1`, round 0 dropped, fitted **within**
each prompt on per-round block times that already existed. No new GPU time.
`harness=local`, ungated, `timing_valid=false`.

| leg | rounds | widths | `b_p` µs/row | stderr | R² | `b_p` drift-controlled | drift µs/round |
|---|---:|---|---:|---:|---:|---:|---:|
| beagle_a, E153 r1 | 118 | 2–8 | 15,366 | 255 | 0.969 | 14,134 | 84.9 |
| beagle_a, E155 off | 118 | 2–8 | 15,365 | 264 | 0.967 | 13,758 | 110.6 |
| essays_montaigne, E153 r1 | 145 | 2–7 | 15,176 | 460 | 0.884 | 14,769 | 19.1 |
| essays_montaigne, E155 off | 145 | 2–7 | 15,664 | 501 | 0.873 | 15,158 | 23.7 |
| benchfixture, E155 off | 77 | 4–8 | 17,200 | 319 | 0.975 | 17,066 | −54.0 |
| **mean** | | | **15,754** | | | **14,977** | |

Two independent `beagle_a` legs on different worker binaries hours apart give
15,366 and 15,365 µs/row — replication to five figures. **15,754 µs/row against
edward's 15,708 µs/row is agreement to 0.29 %** by two different methods on two
different data sets: mine a within-prompt regression on width, his a within-`d`
contrast. Against the advisor's trichotomy, `b_p` is **3.67x** the cross-prompt
`4,291 µs/row`, which is the third branch — the cross-prompt fit behind
FINDING 281 was confounded. That is independent evidence for ADVISOR ERROR 192.

Confounds stated with the number: depth is chosen adaptively so `M` is not
randomised; wider rounds also run more proposal-head steps, which the two-term
model folds into `b_p`; partial accepts trigger verify-block replay in 33 of
119 `beagle_a` rounds; and this is M4 Pro, not the ranked M5.

### 11. Ranked cross-check against FINDING 286

| prompt | ranked `p` | my local conditional `p_shipped` |
|---|---:|---:|
| beagle | 0.8973 | 0.8891 (`beagle_a`) |
| essays | 0.9222 | 0.8512 (`essays_montaigne`) |

`beagle` agrees to 0.8 pp, which supports the recovery method. `essays` is
7.1 pp apart, which I would not read as an error in FINDING 286: my legs use
public fixture prompts over a 512-token window and the ranked `essays` prompt
is hidden, so they are different texts under the same name.

### 12. What this bounds for thorfinn's crown-founded index port

The geometry I audited **is** the geometry the lead candidate will ship:
crown source plus `p15` plus `leaf16` is 6,146 leaves of 16 rows, 922 probed
leaves, 14,752 refined rows, 32 rerank candidates. The crown's own index is
12,292 leaves of 8 rows at `p25`, 3,073 probed leaves and 24,584 refined rows —
1.67x more refined rows per draft step.

That makes the audit a two-sided price on the substitution, and both sides are
now numbers rather than assumptions.

| side | value | source |
|---|---:|---|
| most recall the crown geometry could possibly return over ours | `<= 0.0755` acceptance points = **`<= +0.2017 %`** published | measured here: an *exact* readout over the same compact set returns only that much, and the crown index is still an approximation of it |
| bytes our geometry does not move | 59.09 MB vs 33.53 MB per draft step, delta **25.56 MB** = 45.09 µs at 567 GB/s = 197.6 µs/round at depth 4.382 = **+0.4583 %** published | roofline arithmetic in the same frame as section 4, `harness=ranked` |

So `p15` + `leaf16` on the crown surface is net positive by at least
`+0.257 %` published, and the loss side is capped by measurement, not by
argument. This is the one place the audit speaks directly to a submission that
is already queued. The arithmetic in this section is derived in this document
from the artifact's own constants; it is not a separate measurement.

### 13. Gates, all re-run at `f08b304a`

`f08b304a` is the last commit that can move any of these: the only commit after
it adds this document, and `research/` is not a submitted path.

```
senpai/validate-assignment-scope.sh 9d3e3d2d… Sources/MLXFastModel/Qwen36MTPBlockSession.swift
  assignment scope OK: 1 submitted path(s)

senpai/check-editable-budget.sh 770a3ff2…                 # growth_enforced
  OK: source=2652191/3000000 headroom=347809 growth=197356/262144 exempt=2410 files=154

senpai/check-editable-budget.sh 9d3e3d2d… 770a3ff2…       # growth_attributable
  OK: source=2652191/3000000 headroom=347809 growth=0/262144 exempt=2410 files=154

senpai/verify-ranked-score-boundary.sh
  PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only
```

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | n/a | n/a | not measured, no timing claim |
| MTP seconds/token | n/a | n/a | not measured, no timing claim |
| local serial-relative speedup | n/a | n/a | not measured, no timing claim |
| effective mean draft length, `beagle_a` | 4.243697 | 4.243697 | identical off vs on |
| accepted draft rate, `beagle_a` | 0.778218 | 0.778218 | identical off vs on |

No local or ranked score is claimed anywhere in this report. Every rate is a
counting statistic; every published percent is an explicit conversion through
the advisor's F3 exchange rate and is labelled as such.

## Conclusion

- **What happened and why.** The shipped ANN path reaches exact-compact parity
  on 1,502 of 1,503 audited slots. Its pure index miss rate is 0.27 %, and
  those misses cost one acceptance outcome in the entire audit. The index is
  not where acceptance is being lost. The vocabulary trim costs 0.15 points
  conditional and, priced against the 567 GB/s roofline, untrimming it is net
  −0.51 % published. What remains — 9.67 of 9.89 conditional points — is the
  427 MB proposal head simply disagreeing with the 27B target, at a median
  margin of 2.19 logits, which is not a readout problem.
- **Evidence for or against the mechanism.** Against, decisively, on both the
  index and the trim, with three positive controls that each move the answer by
  more than 80 pp, an exhaustive verified partition, and a four-sided proof
  that the audit cannot reach timed generation.
- **Prompt or M5 transfer risk.** The rates are local, on public fixtures, at
  512 tokens. `beagle` matches FINDING 286's ranked recovery to 0.8 pp;
  `essays` does not, because the ranked text is hidden and different. The
  verdict is robust to that: closing the axis needs the index term below
  0.3 pp, and it is below 0.21 pp on all three prompts individually, including
  the highest-acceptance one. The width-cost fit is M4 Pro and does not
  transfer to the ranked host without a round-time ratio.
- **Smallest useful next action.** Stop paying for index coverage. Two things
  follow directly. First, the question inverts: if recall costs almost nothing,
  ask how much *cheaper* the index can be made before recall breaks — the
  coarse pass alone is 9.83 MB per draft step and is the only index term left
  with anything to trade, bounded by the knee just below `p15`. Second, the
  acceptance score lives in the proposal head itself, where
  `mtp-head.manifest.json` and `mtp-head/` are editable under a 2 GiB cap and
  nobody has touched them. That is complementary to edward's E157 depth rule:
  FINDING 288 says a better head makes the existing schedule correct for free.
- **Recommendation: close.** Close the index axis and the vocabulary-trim axis.
  R0 is closed by inheritance under FINDING 283; 179(E) versus 182 remains
  formally unresolved and the crown's authors sided with 179(E). This branch
  proposes no candidate change and consumed no editable budget.
