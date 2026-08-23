# E158 result: the head's residual is capacity, not precision — and the three bf16 precision islands are dead weight

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e158_head_precision_recoverable_pp","available":true,"value":-0.00731},"secondary_metric":{"name":"e158_island_pct_vs_all_PROVISIONAL_none","available":true,"value":0.45835},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- **Student / branch:** `qwen-askeladd` /
  `qwen-askeladd/e158-head-precision-census`
- **Hypothesis and target cost:** E155 left 12.44 pp of marginal acceptance
  loss (9.67 pp conditional) that neither the ANN index nor the vocabulary trim
  can reach. That residual has two possible causes. Either the declared head is
  *quantized too hard* and a wider-precision head would recover acceptance
  (precision), or the head's single MTP layer simply cannot predict those
  tokens at any precision (capacity). E158 separates them by re-running the
  same exact-readout audit under the organizer-pinned **bf16** head, which is
  2.85x heavier per proposal slot. R1.B then prices the declared head's own
  three bf16 precision islands (`q`, `k`, `v`) by ablating them.
- **Decision:** **close the head-precision axis, and drop the precision
  islands.** `e158_verdict = close_head_precision`,
  `e158_island_verdict = drop_the_islands`. No candidate change is proposed by
  this branch; the submitted surface is byte-identical to the PR base. The
  island result is a concrete, cheap, *positive* recommendation for whoever
  next rebuilds the head.
- **`BASE_SHA` / `UPSTREAM_SHA` / candidate commit:**
  `5f60bea8e154163e562287f1317a845ef10b0b44` (PR base, advisor branch
  `senpai/qwen38-mtp-r1`) / `0863b06ac16e26e48fc06e97444095b00feb66d4` / no
  candidate commit. The audit build was `305787da`; the instrument is reverted
  at `a5d7fc98` and the head of this branch is a research-only tree.
- **Yukon promoted submission / source ref used as frontier:** the bar is
  `ec24d591` at `3.72911001`, source `0863b06a`, as recorded in
  `senpai/frontier-state.json` at 15:57Z. I did not query Yukon and I did not
  call a submission; the advisor's brief explicitly forbade both.
- **Candidate build fingerprint:** audit-ON worker
  `b3f5d6cfdce13f645e6ea2b32eae4b637fe0f84d52e68a50e4b4ff49e6d91c25`;
  reverted, audit-free worker at the head of this branch
  `696416f78f1bfbf0103ec433a5d18d5b0738d118002dd4ce7049933f58a69ff3`, rebuilt
  and asserted with
  `senpai/rebuild-and-assert-worker.sh --forbid-symbol installIfRequested
  --forbid-symbol QwenMTPRecallAudit` (PASS, 576,978 symbols extracted).
- **Submitted-surface / generated-twin / metallib digests:** unchanged. No
  Metal source and no submitted Swift source differ from the PR base at the
  head of this branch.
  `git diff --stat 5f60bea8 -- Sources Vendor Package.swift Package.resolved
  mtp-head.manifest.json mtp-head` is **empty**.
- **Submitted candidate files:** none.
  `senpai/check-editable-budget.sh 5f60bea8…` reports
  `growth_attributable = 0`.
- **Supporting test, tooling, or documentation files:**
  `research/e158_head_census.py`, `research/e158_draft_step_bytes.py`,
  `research/e158_r1_session.sh`, `research/e158_island_session.sh`,
  `research/e158_island_curve.py`, `research/e158_wandb_log.py`,
  `research/e158-artifacts/*.json`, and this document. None is a submitted
  path.
- **MTP head provenance, digest, and draft policy:** two heads, deliberately.
  The **declared** head is
  `head_provenance.sha256 = dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`
  (run tree, `config.json` + `model.safetensors`), manifest digest
  `559b24eb…`, origin
  `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae628274`, 427,742,600 bytes,
  **40 tensors**. The **pinned** head is the organizer default,
  as-staged tree digest `6bab9c82…`, 849,400,347 bytes, **15 tensors**, all
  BF16. Draft policy is the shipped adaptive schedule at offered depth 8 on
  every leg.
- **Token window, fixture, reference source, and harness:** 512 decode tokens
  behind a 512-token seed, public fixtures `beagle_a`, `essays_montaigne` and
  `benchfixture`, checked-in E128 goldens. **`harness=local` on every measured
  rate; `harness=ranked` on every conversion to published percent.**
- **Exact cell:** the declared head's proposal readout — the derived cluster
  ANN index over the affine-2 g64 `draft_lm_head` (`[98336, 5120]`, stored as
  `[98336, 320]` U32 plus `[98336, 80]` bf16 scales and biases), 6,146 leaves,
  922 probes, 14,752 refined rows, then a 32-candidate affine-4 rerank against
  the target `lm_head` — and, for the pinned arm, the *absence* of that
  readout.
- **Official causal path and score equation:** untouched. This branch changes
  no scored code; `senpai/verify-ranked-score-boundary.sh` passes.
- **Assignment-scope preflight:** `senpai/validate-assignment-scope.sh` takes
  `BASE_SHA SUBMITTED_PATH…` and correctly reports `research` as outside
  `editablePaths`. The real scope evidence is the empty submitted-surface diff
  quoted above.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  `source=2652191/3000000 growth=197356/262144` for `growth_enforced` against
  `770a3ff2`, and `growth=0/262144` for `growth_attributable` against the PR
  base. I consumed no team growth budget.
- **Scored-path reachability evidence:** the audit is gated by
  `MLX_E155_RECALL_AUDIT`, absent from the shipped default, proven off on four
  sides (section 6). At the head of this branch the hook does not exist on the
  submitted surface at all.
- **Written promotion rule and verdict:** the brief pre-registered
  `>= 1.0 pp` recoverable means build a wider head, `0.3` to `1.0 pp` means one
  bounded arm, `< 0.3 pp` means close. Measured **`-0.0073 pp`** — the wrong
  sign. **Verdict: close.** For the islands the brief asked for the cheapest
  arm whose acceptance loss stays under the advisor's provisional break-even;
  `none` loses `0.0605 pt` against a `0.1717 pt` provisional threshold and
  saves the most bytes. **Verdict: drop all three.**
- **Pre-official evidence budget / timed legs used:** zero timed legs, zero
  gated legs, no thermal gate, no ABBA, no palindrome. Nineteen untimed
  512-token decodes (6 audit legs for R1, 12 for the island curve, 1 gate
  probe).
- **Frozen candidate SHA, if promoted for submission:** none.
- **Specific evidence that invalidated the frozen SHA, if any:** not
  applicable.
- **Submission owner / read-only receipt-watcher job ID:** not applicable. I
  did not call Yukon.
- **W&B:** run `t6x06104`,
  <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/t6x06104>,
  group `e158-head-precision-census`. Artifact
  `e158-head-precision-census` carries all ten JSON artifacts and all
  nineteen raw per-slot JSONL files.

## Evidence

- **Host, instance, chip, memory profile, toolchain, thermal policy:** Apple
  M4 Pro, 20-core `applegpu_g16s`, 48 GiB. No cool gate was taken and none is
  claimed: every artifact records `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `timing_valid=false`,
  `official_or_ranked_score=false`.
- **Exact commands:**
  - `research/e158_head_census.py`
  - `research/e158_draft_step_bytes.py`
  - `research/e158_r1_session.sh declared beagle_a essays_montaigne benchfixture`
  - `research/e158_r1_session.sh pinned beagle_a essays_montaigne benchfixture`
  - `research/e158_island_session.sh all,q,kv,none beagle_a essays_montaigne benchfixture`
  - `research/e158_island_curve.py`, `research/e155_gate_probe.sh beagle_a`,
    `research/e158_wandb_log.py`
- **Cheapest real falsification gate and positive-control verdict:** the
  permutation control, run on both heads. Pairing each slot's shipped id with
  the *next* slot's target collapses `p_shipped` from `0.872921` to `0.015979`
  on the declared head (85.694 pp) and from `0.881588` to `0.010274` on the
  pinned head (87.131 pp). The comparison is capable of failing on both arms.
- **Tests and risk-based checks, in execution order:** safetensors header
  census on both heads; tree-digest recipe verification against the recorded
  `head_provenance.sha256`; head-swap round trip in both directions;
  golden-row exactness on all eighteen audit legs; audit row-count identity
  against `sum(effective_draft_lengths)`; gate probe; permutation control;
  masked-winner control; `ann_outside_compact_set` census; instrument revert
  and forbidden-symbol worker rebuild; then the three gate scripts on the final
  head.
- **Exact-token and row-ledger verdict:** `all_tokens_matched = true` and
  `residual_divergence_count = 0` on all eighteen audit legs and on the gate
  probe.
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not relevant; no Metal source changed.
- **Peak RAM or head/artifact size:** the audit adds one dense
  full-vocabulary `lm_head` projection over all 248,320 ids per proposal slot,
  inside the measured block. Every audit leg is therefore non-timing by
  construction. Raw per-slot JSONL totals 2.3 MB and lives outside Git under
  `.mlxfast-private/e158/`.

### 1. Which head does each harness actually load?

`e158_heads_differ_local_vs_ranked = true`. The two heads on this machine are
different trees:

| | pinned (organizer default) | declared (`mtp-head.manifest.json`) |
|---|---|---|
| tree digest | `6bab9c82…` (as staged, 5 files) | `559b24eb…` bare / `dadbfb80…` run tree |
| file bytes | 849,400,347 | 427,742,600 |
| tensors | 15 | **40** |
| dtypes | BF16 only | BF16 + U32 + I32 |
| vocabulary projection | **none** | `draft_lm_head`, affine-2 g64 over `[98336, 5120]` |

`e158_head_swap_works = true` in both directions, and
`e158_tree_digest_recipe_verified = true`: recomputing the digest recipe over
the declared run tree reproduces `dadbfb80…` exactly.

**This corrects advisor FINDING 295.** `benchmark-qwen-mtp.sh` drafts with the
pinned bf16 head, but my E155 and E158 legs run through
`research/e128_session.sh:84`, which points the loader at the declared run
tree. Every E155 leg carried `head_provenance.sha256 = dadbfb80…`, so E155
measured the *shipped* head, not the pinned one. The E158 declared arm
reproduces E155's conditional bucket C to 16 significant figures
(`9.667673716012084` here versus E155's `9.6677`, on 1,503 slots) — an exact
determinism replication on a rebuilt worker, which is the strongest available
proof that the two experiments loaded the same weights. The advisor has
accepted this correction.

### 2. RULE 173: which provenance fields are witnesses

Four fields that read like head witnesses are not:

| field | why it is not a witness |
|---|---|
| `head_provenance.origin` | the manifest URL; identical for both trees I loaded from the same origin |
| `head_provenance.source` | `"remote"` vs `"local"`; describes fetch, not content |
| `uses_pinned_mtp_head` | true whenever *a* head drafted; true on both arms |
| `uses_native_mtp_head` | same; true on both arms |
| `mtp_head_tensor_count` | **compile-time constant 15**, not read from the file — the declared head has 40 tensors and still reports 15 |

Only `head_provenance.sha256` identifies the tree. E155's own result document
records "15 tensors" for the declared head; that number came from this
constant and is wrong. The correct figure is 40.

### 3. Coverage

| head | prompt | slots | rounds | mean offered depth | conditional slots |
|---|---|---:|---:|---:|---:|
| declared | beagle_a | 505 | 119 | 4.2437 | 442 |
| declared | essays_montaigne | 502 | 146 | 3.4384 | 430 |
| declared | benchfixture | 496 | 78 | 6.3590 | 452 |
| declared | **pooled** | **1503** | **343** | **4.3819** | **1324** |
| pinned | beagle_a | 482 | 121 | 3.9835 | 441 |
| pinned | essays_montaigne | 488 | 147 | 3.3197 | 432 |
| pinned | benchfixture | 491 | 77 | 6.3766 | 450 |
| pinned | **pooled** | **1461** | **345** | **4.2348** | **1323** |

The two arms are not the same slot population — a different head proposes a
different number of drafts — so all comparisons below use the conditional
population (slots where the shipped path had a genuine choice), which the two
arms match to within one slot (1,324 versus 1,323).

### 4. R1 — the precision decomposition

Six rates on the conditional population:

| | pinned bf16 | declared q2/q4 | pinned − declared |
|---|---:|---:|---:|
| `p_shipped` | 0.9002268 | 0.9010574 | −0.0008306 |
| `p_exact_compact` | 0.9002268 | 0.9018127 | −0.0015859 |
| `p_exact_full` | **0.9032502** | **0.9033233** | **−0.0000731** |

Three-bucket split, conditional, both partitions exhaustive:

| bucket | pinned bf16 | declared q2/q4 |
|---|---:|---:|
| A — selection defect (index) | 0.0000 pp | 0.0755 pp |
| B — candidate-set trim (vocabulary) | 0.3023 pp | 0.1511 pp |
| C — irreducible head disagreement | **9.6750 pp** | **9.6677 pp** |

**`e158_head_precision_recoverable_pp = −0.0073`.** Spending 2.85x the bytes
per proposal slot on a fully bf16 head recovers *negative* acceptance: one
slot in 1,323. Bucket C is flat to 0.008 pp across a 2.85x precision change.
The residual is capacity, not precision.

Priced against the advisor's F4 exchange rates, all **PROVISIONAL**:

| term | value |
|---|---:|
| `e158_bf16_bytes_per_draft_step_delta_MB` | 834.583 |
| `e158_precision_gain_pct_PROVISIONAL` | −0.0195 % |
| `e158_precision_cost_pct_PROVISIONAL` | +14.9641 % |
| `e158_precision_net_pct_PROVISIONAL` | **−14.9836 %** |

Break-even needed 4.054 acceptance points. The measured value is negative, so
the margin against the promotion rule is roughly 555x on the gain side and the
sign is wrong besides. No plausible refit of the F4 rate changes this verdict:
the acceptance term is zero within one slot regardless of what a point is
worth.

### 5. R0.3 — why the pinned head costs 834.58 MB more per draft step

Per proposal slot, at the current tree geometry (6,146 leaves, 922 probes,
14,752 refined rows):

| | trunk | readout | total |
|---|---:|---:|---:|
| declared q2/q4 | 264,494,080 | 33,528,960 | **298,023,040** |
| pinned bf16 | 849,398,784 | 283,207,680 | **1,132,606,464** |

The readout term is the surprise. The pinned head has **no vocabulary
projection of its own** (`has_full_vocab_projection = false`,
`has_compact_vocab_projection = false`). Without a `draft_lm_head`,
`buildDerivedClusterIndex` (`Sources/MLXFastModel/Qwen35.swift:6146-6152`)
guards on `_draftHeadW` and never builds an ANN index at all, so
`draftTokenID` falls through to `_compactDraftHead` **dense** — a full
98,336-row affine-4 g64 gather from the target `lm_head` every slot,
283,207,680 bytes.

The audit counters confirm this independently: `index_miss = 0` on 1,461
pinned slots versus `4` on 1,503 declared slots. A dense readout cannot miss
its index because it has none. **This corrects the advisor's F2 point 3 and
their 603.79 MB byte model**; the true delta is 834.58 MB.

The trunk term splits per matrix. `research/e158-artifacts/draft-step-bytes.json`
carries the full table; the three MLP matrices dominate at 128.12 MB each,
`q_proj` adds 79.95 MB, `fc` 75.37 MB, `o_proj` 45.22 MB, and `k_proj`/`v_proj`
add **nothing** — they are already bf16 in the declared head, because the
precision islands cover them completely. That last row is what makes section 8
worth running.

Also worth recording: the declared head keeps 5,906,432 bytes resident that
draft-time never reads (the affine-4 `k_proj`/`v_proj` packs and the k/v island
index vectors), because the bf16 islands supersede them.

### 6. Gate witness, four sides

1. **Source:** the audit hook only fires under `MLX_E155_RECALL_AUDIT`, which
   no shipped path sets.
2. **Two-sided probe:** `research/e155_gate_probe.sh` with
   `MLX_E155_RECALL_AUDIT_PATH` *set* and `MLX_E155_RECALL_AUDIT` *unset*, 64
   tokens, gives `audit_file_created = false`, `audit_rows_written = 0`,
   `leg_exit_code = 0`. The path variable alone does nothing. Saved as
   `research/e158-artifacts/gate-probe.json`; E155's own probe artifact was
   restored untouched.
3. **Revert:** `git apply -R research/e155-patches/recall-audit.patch` leaves
   zero references to the instrument anywhere under `Sources/`.
4. **Binary:** the rebuilt worker
   `696416f7…` passes `--forbid-symbol installIfRequested --forbid-symbol
   QwenMTPRecallAudit` over 576,978 extracted symbols. The audit build
   `b3f5d6cf…` is a different binary and is named in every audit artifact.

`e158_exact_path_is_gated_off_by_default = true`.

### 7. Positive controls (RULE 101)

| control | declared | pinned |
|---|---:|---:|
| permutation magnitude | 85.694 pp | 87.131 pp |
| masked-winner magnitude | 83.832 pp | 84.736 pp |
| `masked_control_moved` | 1,503 / 1,503 | 1,461 / 1,461 |
| `ann_outside_compact_set` | 0 | 0 |
| `index_miss` | 4 | 0 |

Both arms move the full slot population under both controls, so neither arm's
null result is a dead instrument.

### 8. R1.B — the precision-island curve

The declared head carries three bf16 precision islands over otherwise affine-4
projections: `q` (a partial row set), `k` and `v` (complete permutations). I
built a driver that installs any subset, and ran four arms x three prompts.
Every leg is witnessed by its own `trace.txt`, because `e128_session.sh`
overrides `MLX_QWEN_MTP_TRACE_PATH` per leg — the witness string names the arm
and both install flags.

| arm | rounds | conditional `p_shipped` | loss vs `all` | bytes/step Δ | published % (PROVISIONAL) |
|---|---:|---:|---:|---:|---:|
| all | 343 | 0.9010574 | — | 0 | — |
| q | 346 | 0.8967596 | +0.4298 pt | −15,073,280 | +0.2703 % |
| kv | 344 | 0.8996226 | +0.1435 pt | −10,489,856 | +0.1881 % |
| **none** | **342** | 0.9004525 | **+0.0605 pt** | **−25,563,136** | **+0.4583 %** |

Two standard errors on any pairwise acceptance difference is **2.32 pt** on
three prompts. Every measured difference is inside that band, so **the
acceptance side of this curve is a null result** and the ordering is noise —
which the curve itself shows, since `q` (one island kept) scores worse than
`none` (no islands kept). That is not a physically possible monotone response;
it is sampling.

The decision therefore rests on the two exact quantities:

- **Bytes.** Dropping all three islands removes 31,461,376 bytes of bf16 island
  weight but returns the 5,898,240 bytes of affine-4 `k`/`v` packs to the read
  set, so the net saving is **25,563,136 bytes**, not 31.46 MB. That is exact
  arithmetic over the safetensors header, not a measurement.
- **Round counts.** These are deterministic given the head and the prompt.
  `none` needs 342 rounds for 3x512 tokens; `all` needs 343. Removing the
  islands does not cost a round.

`e158_island_verdict = drop_the_islands`, worth **+0.4583 %** provisional,
which matches the advisor's composition table entry of `+0.458 %`.

The honest framing: I cannot prove `none` is *better* than `all` on
acceptance. I can prove it is not measurably worse on 1,326 slots, that it
reads 25.56 MB less per proposal slot, and that it finishes the same work in
one fewer round. For a head rebuild that is a free simplification.

### 9. Provisional-percent caveat

Advisor F4 retracted `+2.6701 %` per acceptance point, the 148.9 MB
break-even, and the 43,114 µs modelled round as self-confirming (RULE 166 /
FINDING 286 / ADVISOR ERROR 198). Every percent figure in this document that
derives from those rates is labelled PROVISIONAL, in the artifacts and in the
W&B summary keys. The quantities that do **not** depend on the retracted rates
are the six acceptance rates, the three-bucket splits, the byte censuses, the
round counts, and both verdicts — because both verdicts are decided by margins
of 100x or by exact byte arithmetic, not by the conversion constant.

## Suggested follow-ups (not implemented)

1. **Rebuild the declared head without the islands.** This is the one
   actionable output of E158: 25.56 MB per proposal slot for no measurable
   acceptance cost. It needs a head rebuild, which the brief forbade this
   round.
2. **Push the trunk ladder down, not the precision up.** Bucket C is flat
   across a 2.85x precision change, so the head is not precision-starved. The
   advisor's trunk ladder (a4 264.5 MB / a3 202.2 MB / a2 150.5 MB) is the
   interesting direction, and E158 predicts the acceptance cost of a3 or a2
   will also be small. Someone should measure it with this same driver.
3. **Attack capacity, not the readout.** 9.67 pp of conditional loss survives
   an exact full-vocabulary argmax on both heads. Only a different head
   *architecture* — more layers, or a different conditioning input — can move
   it. That is a large, slow experiment and should be scoped deliberately.
4. **Fix `mtp_head_tensor_count`.** It is a compile-time constant that lies
   about the loaded file. It is telemetry, not a scored path, but it has
   already produced one wrong number in a result document.
5. **Reclaim the 5.9 MB of never-read residency** in the declared head if the
   islands stay. Cheap, but strictly smaller than follow-up 1.
